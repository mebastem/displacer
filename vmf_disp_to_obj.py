#!/usr/bin/env python3
"""
vmf_disp_to_obj.py
------------------
Extracts displacement surfaces from a Source Engine VMF file and writes
them as Wavefront OBJ files, one per displacement tile or merged by proximity.

Requires Python 3.7+, no third-party libraries.

Usage:
    python vmf_disp_to_obj.py <map.vmf> [options]

Options:
    -o, --output DIR        Output directory (default: ./disp_obj)
    --no-merge              Write one OBJ per displacement tile instead of grouping
    --proximity FLOAT       Bounding-box gap threshold for grouping tiles (default: 4.0)
    --weld FLOAT            Vertex merge distance within a group (default: 1.0)
    -v, --verbose           Print per-tile debug info
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Types and math
# ---------------------------------------------------------------------------

Vec3 = Tuple[float, float, float]

def vadd(a: Vec3, b: Vec3) -> Vec3: return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def vsub(a: Vec3, b: Vec3) -> Vec3: return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vscale(v: Vec3, s: float) -> Vec3: return (v[0]*s, v[1]*s, v[2]*s)
def vdot(a: Vec3, b: Vec3) -> float: return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def vlen(v: Vec3) -> float: return math.sqrt(v[0]**2+v[1]**2+v[2]**2)
def vnorm(v: Vec3) -> Vec3:
    l = vlen(v)
    return (v[0]/l, v[1]/l, v[2]/l) if l > 1e-9 else (0., 0., 1.)
def vcross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def vlerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return vadd(vscale(a, 1-t), vscale(b, t))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DispInfo:
    power: int
    start_position: Vec3 = (0., 0., 0.)
    normals:   List[List[Vec3]]  = field(default_factory=list)
    distances: List[List[float]] = field(default_factory=list)
    offsets:   List[List[Vec3]]  = field(default_factory=list)
    alphas:    List[List[float]] = field(default_factory=list)  # blend factor 0-255 per vertex


@dataclass
class TexAxes:
    udir:    Vec3
    uoffset: float
    uscale:  float
    vdir:    Vec3
    voffset: float
    vscale:  float


@dataclass
class DispSide:
    solid_id:      int
    side_id:       int
    plane_pts:     List[Vec3]        # 3 points defining the face plane (VMF CCW winding)
    sibling_planes: List[List[Vec3]] # all planes in the parent solid
    dispinfo:      DispInfo
    material:      str = 'NODRAW'
    tex_axes:      Optional['TexAxes'] = None


@dataclass
class Mesh:
    verts:    List[Vec3]
    normals:  List[Vec3]
    uvs:      List[Tuple[float,float]]
    tris:     List[Tuple[int,int,int]]  # 0-based vertex indices
    material:    str   = 'NODRAW'
    alphas:      List[float] = field(default_factory=list)  # per-vertex blend 0-1
    tex_density: float = 0.0  # texels per Source unit (for auto triplanar scale)

    def bbox(self):
        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        zs = [v[2] for v in self.verts]
        return (min(xs),min(ys),min(zs)),(max(xs),max(ys),max(zs))


# ---------------------------------------------------------------------------
# VMF parser  (minimal recursive key-value parser)
# ---------------------------------------------------------------------------

class VMFParser:
    _STRIP_COMMENTS = re.compile(r'//[^\n]*')
    _TOKEN          = re.compile(r'"[^"]*"|\{|\}|[^\s"{}]+')

    def __init__(self, text: str):
        clean = self._STRIP_COMMENTS.sub('', text)
        self._tok = self._TOKEN.findall(clean)
        self._pos = 0

    def _peek(self): return self._tok[self._pos] if self._pos < len(self._tok) else None
    def _next(self):
        if self._pos >= len(self._tok):
            raise EOFError("unexpected end of VMF")
        t = self._tok[self._pos]; self._pos += 1; return t
    def _uq(self, s): return s.strip('"')

    def _block(self) -> dict:
        t = self._next()
        if t != '{':
            raise ValueError(f"expected '{{', got {t!r}")
        d: dict = {}
        while True:
            p = self._peek()
            if p is None:
                break   # truncated file — return what we have
            if p == '}':
                self._next()
                break
            k = self._uq(self._next())
            p2 = self._peek()
            if p2 is None:
                break
            v = self._block() if p2 == '{' else self._uq(self._next())
            if k in d:
                d[k] = d[k] if isinstance(d[k], list) else [d[k]]
                d[k].append(v)
            else:
                d[k] = v
        return d

    def parse(self) -> List[dict]:
        out = []
        while self._peek():
            p = self._peek()
            if p == '{':
                # anonymous block — skip
                self._block()
                continue
            name = self._uq(self._next())
            if self._peek() != '{':
                continue   # bare key with no block, skip
            try:
                b = self._block()
                b['__name__'] = name
                out.append(b)
            except (EOFError, ValueError):
                break   # truncated/malformed — return what we parsed so far
        return out


# ---------------------------------------------------------------------------
# VMF helpers
# ---------------------------------------------------------------------------

def parse_vec3(s: str) -> Vec3:
    n = re.findall(r'-?[\d.eE+\-]+', s)
    return float(n[0]), float(n[1]), float(n[2])

def parse_plane(s: str) -> List[Vec3]:
    return [parse_vec3(g) for g in re.findall(r'\(([^)]+)\)', s)]

def parse_tex_axis(s: str) -> Optional[Tuple]:
    """Parse VMF '[ux uy uz offset] scale' into (dir, offset, scale)."""
    nums = list(map(float, re.findall(r'-?[\d.eE+\-]+', s)))
    if len(nums) < 5:
        return None
    return (nums[0], nums[1], nums[2]), nums[3], nums[4]

def parse_row_vecs(s: str, n: int) -> List[Vec3]:
    nums = list(map(float, s.split()))
    return [(nums[i*3], nums[i*3+1], nums[i*3+2]) for i in range(n)]

def parse_row_floats(s: str, n: int) -> List[float]:
    return list(map(float, s.split()))[:n]


def read_dispinfo(raw: dict, power: int) -> DispInfo:
    size = (1 << power) + 1
    start = parse_vec3(raw.get('startposition', '0 0 0'))

    def rows_vecs(key, default_vec):
        blk = raw.get(key, {})
        out = []
        for r in range(size):
            s = blk.get(f'row{r}', '')
            out.append(parse_row_vecs(s, size) if s else [default_vec]*size)
        return out

    def rows_floats(key):
        blk = raw.get(key, {})
        out = []
        for r in range(size):
            s = blk.get(f'row{r}', '')
            out.append(parse_row_floats(s, size) if s else [0.]*size)
        return out

    return DispInfo(power=power, start_position=start,
                    normals=rows_vecs('normals', (0., 0., 1.)),
                    distances=rows_floats('distances'),
                    offsets=rows_vecs('offsets', (0., 0., 0.)),
                    alphas=rows_floats('alphas'))


def extract_disp_sides(blocks: List[dict]) -> List[DispSide]:
    """Walk the VMF tree and collect every side that has a dispinfo."""
    results: List[DispSide] = []

    def walk(node: dict, solid_id: int = -1):
        name = node.get('__name__', '')
        if name == 'solid':
            try: solid_id = int(node.get('id', -1))
            except ValueError: pass

        for key, val in node.items():
            if key == '__name__':
                continue
            for item in (val if isinstance(val, list) else [val]):
                if not isinstance(item, dict):
                    continue
                item_name = item.get('__name__', key)

                if item_name == 'solid' or key == 'solid':
                    sid = int(item.get('id', solid_id)) if 'id' in item else solid_id

                    # Grab all side plane lists for this solid up front
                    side_list = item.get('side', [])
                    if not isinstance(side_list, list):
                        side_list = [side_list]
                    all_planes = [parse_plane(s.get('plane', ''))
                                  for s in side_list
                                  if isinstance(s, dict) and s.get('plane')]

                    for s in side_list:
                        if not isinstance(s, dict) or 'dispinfo' not in s:
                            continue
                        side_id  = int(s.get('id', -1)) if 'id' in s else -1
                        pts      = parse_plane(s.get('plane', ''))
                        di_raw   = s['dispinfo']
                        if isinstance(di_raw, list): di_raw = di_raw[0]
                        power    = int(di_raw.get('power', 2))
                        di       = read_dispinfo(di_raw, power)
                        mat      = s.get('material', 'NODRAW').strip()
                        ta       = None
                        u_raw    = parse_tex_axis(s.get('uaxis', ''))
                        v_raw    = parse_tex_axis(s.get('vaxis', ''))
                        if u_raw and v_raw:
                            ta = TexAxes(udir=u_raw[0], uoffset=u_raw[1], uscale=u_raw[2],
                                         vdir=v_raw[0], voffset=v_raw[1], vscale=v_raw[2])
                        results.append(DispSide(solid_id=sid, side_id=side_id,
                                                plane_pts=pts, sibling_planes=all_planes,
                                                dispinfo=di, material=mat, tex_axes=ta))
                    walk(item, sid)

                elif item_name != 'side' and key != 'side':
                    walk(item, solid_id)

    for b in blocks:
        walk(b)
    return results


# ---------------------------------------------------------------------------
# Quad corner recovery — uses VMF plane points directly
# ---------------------------------------------------------------------------


def _plane_nd(pts: List[Vec3]):
    """Return (unit_normal, d) for a plane defined by 3 points."""
    p0, p1, p2 = pts[0], pts[1], pts[2]
    n = vnorm(vcross(vsub(p1, p0), vsub(p2, p0)))
    return n, vdot(n, p0)

def _intersect3(n1, d1, n2, d2, n3, d3) -> Optional[Vec3]:
    """Intersection point of three planes.  Returns None if degenerate."""
    c23 = vcross(n2, n3)
    det = vdot(n1, c23)
    if abs(det) < 1e-9:
        return None
    c31 = vcross(n3, n1)
    c12 = vcross(n1, n2)
    return vscale(vadd(vadd(vscale(c23, d1), vscale(c31, d2)), vscale(c12, d3)), 1.0 / det)

def _dedupe_points(points: List[Vec3], eps: float = 1e-4) -> List[Vec3]:
    """Collapse nearly-identical points while preserving first-seen order."""
    out: List[Vec3] = []
    for p in points:
        if all(vlen(vsub(p, q)) > eps for q in out):
            out.append(p)
    return out


def _order_face_points(points: List[Vec3],
                       face_n: Vec3,
                       winding_ref: Optional[List[Vec3]] = None) -> List[Vec3]:
    """Return face vertices in a stable winding around their centroid."""
    center = (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
        sum(p[2] for p in points) / len(points),
    )

    axis_u = vsub(points[0], center)
    if vlen(axis_u) < 1e-9:
        axis_u = (1., 0., 0.)
    axis_u = vnorm(axis_u)
    axis_v = vnorm(vcross(face_n, axis_u))

    ordered = sorted(
        points,
        key=lambda p: math.atan2(vdot(vsub(p, center), axis_v),
                                 vdot(vsub(p, center), axis_u))
    )

    # Match the VMF plane winding as closely as possible.  The first three plane
    # points are real face corners, so their signed area is the authoritative
    # local orientation for this side.
    ref_pts = winding_ref if winding_ref and len(winding_ref) >= 3 else points
    ref = vdot(vcross(vsub(ref_pts[1], ref_pts[0]), vsub(ref_pts[2], ref_pts[1])), face_n)
    cur = vdot(vcross(vsub(ordered[1], ordered[0]), vsub(ordered[2], ordered[1])), face_n)
    if ref * cur < 0:
        ordered.reverse()
    return ordered


def _brush_face_vertices(plane_pts: List[Vec3],
                         sibling_planes: List[List[Vec3]]) -> Optional[List[Vec3]]:
    """
    Reconstruct this side's full polygon by intersecting the parent brush planes.

    VMF side planes define a convex brush.  Instead of guessing a missing corner
    from arbitrary sibling-plane pairs, derive all brush vertices and keep the
    ones that lie on this face.  This also makes adjacent displacements recover
    shared brush corners from the same plane set.
    """
    planes = []
    for pts in sibling_planes:
        if len(pts) >= 3:
            planes.append(_plane_nd(pts))
    if len(planes) < 4:
        return None

    face_n, face_d = _plane_nd(plane_pts)
    face_idx = None
    for idx, (n, d) in enumerate(planes):
        if abs(vdot(n, face_n) - 1.0) < 1e-6 and abs(d - face_d) < 1e-4:
            face_idx = idx
            break
    if face_idx is None:
        planes.append((face_n, face_d))
        face_idx = len(planes) - 1

    def collect(sign: float) -> List[Vec3]:
        verts: List[Vec3] = []
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                for k in range(j + 1, len(planes)):
                    p = _intersect3(planes[i][0], planes[i][1],
                                    planes[j][0], planes[j][1],
                                    planes[k][0], planes[k][1])
                    if p is None:
                        continue
                    if all(sign * (vdot(n, p) - d) <= 1e-3 for n, d in planes):
                        verts.append(p)
        face_verts = [
            p for p in _dedupe_points(verts)
            if abs(vdot(face_n, p) - face_d) <= 1e-3
        ]
        return face_verts

    face_verts = collect(1.0)
    if len(face_verts) < 3:
        face_verts = collect(-1.0)
    if len(face_verts) < 3:
        return None

    return _order_face_points(face_verts, face_n, plane_pts)


def recover_corners(plane_pts: List[Vec3],
                    sibling_planes: List[List[Vec3]],
                    start_pos: Vec3) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    """
    Recover the 4 grid corners from the VMF plane points.

    p0, p1, p2 are 3 actual face corners stored in the VMF plane.  Prefer
    reconstructing the whole face from the parent brush's convex plane set; only
    fall back to a parallelogram when brush reconstruction is unavailable.

    After recovering all 4 corners we rotate them so the one nearest
    startposition comes first (CW), giving:
        corners[0] = c_start
        corners[1] = c_row   (first CW step from start)
        corners[2] = c_diag
        corners[3] = c_col   (third CW step from start)
    """
    p0, p1, p2 = plane_pts[0], plane_pts[1], plane_pts[2]
    face_n, face_d = _plane_nd([p0, p1, p2])
    corners = _brush_face_vertices(plane_pts, sibling_planes)
    if corners is None:
        p3 = vadd(p0, vsub(p2, p1))
        corners = _order_face_points([p0, p1, p2, p3], face_n, plane_pts)
    elif len(corners) != 4:
        # Source displacements are quadrilateral.  If the brush face has extra
        # clipped vertices, keep the four corners nearest to the displacement's
        # authored plane points and the opposite parallelogram prediction.
        refs = [p0, p1, p2, vadd(p0, vsub(p2, p1))]
        picked: List[Vec3] = []
        for ref in refs:
            p = min(corners, key=lambda c: vlen(vsub(c, ref)))
            if all(vlen(vsub(p, q)) > 1e-4 for q in picked):
                picked.append(p)
        if len(picked) == 4:
            corners = _order_face_points(picked, face_n, plane_pts)
        else:
            raise ValueError(f"displacement side is not a quad ({len(corners)} face vertices)")

    # Rotate so the corner nearest startposition comes first, preserving CW order
    dists = [vlen(vsub(c, start_pos)) for c in corners]
    si = dists.index(min(dists))
    corners = corners[si:] + corners[:si]

    # The triangle builder expects cross(row_axis, col_axis) to face the source
    # side normal.  If the recovered winding is opposite, flip the two adjacent
    # corners so dispinfo rows/columns are not transposed by winding alone.
    row_axis = vsub(corners[1], corners[0])
    col_axis = vsub(corners[3], corners[0])
    if vdot(vcross(row_axis, col_axis), face_n) < 0:
        corners = [corners[0], corners[3], corners[2], corners[1]]

    # corners[0]=c_start, corners[1]=c_row, corners[2]=c_diag, corners[3]=c_col
    return corners[0], corners[1], corners[2], corners[3]


# ---------------------------------------------------------------------------
# Displacement mesh builder
# ---------------------------------------------------------------------------

def build_mesh(ds: DispSide) -> Mesh:
    """
    Build the displaced mesh.

    Grid layout (Source convention):
      (row=0, col=0) = c_start
      (row=0, col=N) = c_col
      (row=N, col=N) = c_diag
      (row=N, col=0) = c_row

    Base position for grid point (row, col):
      left_edge  = lerp(c_start, c_row,  row/N)
      right_edge = lerp(c_col,   c_diag, row/N)
      base       = lerp(left_edge, right_edge, col/N)
    """
    di   = ds.dispinfo
    size = (1 << di.power) + 1
    c_start, c_row, c_diag, c_col = recover_corners(
        ds.plane_pts, ds.sibling_planes, di.start_position)

    verts:  List[Vec3]               = []
    uvs:    List[Tuple[float,float]] = []
    alphas: List[float]              = []

    for row in range(size):
        t = row / (size - 1)
        edge_a = vlerp(c_start, c_row,  t)
        edge_b = vlerp(c_col,   c_diag, t)
        for col in range(size):
            s = col / (size - 1)
            base = vlerp(edge_a, edge_b, s)
            n_raw = di.normals[row][col]
            n_len = vlen(n_raw)
            d     = di.distances[row][col]
            off   = di.offsets[row][col]
            # Only apply displacement if normal is a valid direction.
            # VMF normals should be unit vectors; re-normalize defensively.
            if n_len > 1e-6:
                # Do NOT renormalise here.  Regular VMF normals are already
                # unit-length so it makes no difference.  Hammer-stitched edge
                # vertices store the pre-computed world-space displacement as
                # the "normal" with distance=1.0; renormalising destroys the
                # magnitude and produces massive spikes at seam edges.
                pos = vadd(base, vadd(vscale(n_raw, d), off))
            else:
                pos = vadd(base, off)
            verts.append(pos)
            # UV: use VMF texture projection if available, else fall back to grid 0-1
            if ds.tex_axes:
                ta = ds.tex_axes
                # Drop uoffset/voffset so adjacent tiles with different offsets
                # have continuous UVs at their shared edges (no seam).
                u = vdot(pos, ta.udir) / ta.uscale
                v = vdot(pos, ta.vdir) / ta.vscale
                uvs.append((u, v))
            else:
                uvs.append((s, t))
            # Blend alpha: 0 = basetexture, 1 = basetexture2
            if di.alphas and row < len(di.alphas) and col < len(di.alphas[row]):
                alphas.append(di.alphas[row][col] / 255.0)
            else:
                alphas.append(0.0)

    tris: List[Tuple[int,int,int]] = []
    for row in range(size-1):
        for col in range(size-1):
            i00 = row*size + col
            i10 = (row+1)*size + col
            i01 = row*size + (col+1)
            i11 = (row+1)*size + (col+1)
            if (row+col) % 2 == 0:
                tris += [(i00,i10,i11),(i00,i11,i01)]
            else:
                tris += [(i00,i10,i01),(i10,i11,i01)]

    # Compute proper per-vertex normals from triangle geometry.
    # Area-weighted accumulation so curvier regions have more influence.
    accum: List[Vec3] = [(0., 0., 0.)] * len(verts)
    for (i, j, k) in tris:
        ab = vsub(verts[j], verts[i])
        ac = vsub(verts[k], verts[i])
        fn = vcross(ab, ac)   # unnormalised = area-weighted face normal
        accum[i] = vadd(accum[i], fn)
        accum[j] = vadd(accum[j], fn)
        accum[k] = vadd(accum[k], fn)
    norms: List[Vec3] = [vnorm(n) for n in accum]

    tex_density = 0.0
    if ds.tex_axes:
        ta = ds.tex_axes
        ud = (vlen(ta.udir) / ta.uscale + vlen(ta.vdir) / ta.vscale) / 2.0
        tex_density = ud
    return Mesh(verts=verts, normals=norms, uvs=uvs, tris=tris, material=ds.material,
                alphas=alphas, tex_density=tex_density)


# ---------------------------------------------------------------------------
# Proximity grouping (Union-Find)
# ---------------------------------------------------------------------------

class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[b] = a


def bbox_gap(a, b) -> float:
    d = 0.
    for i in range(3):
        g = max(0., max(a[0][i], b[0][i]) - min(a[1][i], b[1][i]))
        d += g*g
    return math.sqrt(d)


def group_meshes(meshes: List[Mesh], proximity: float) -> List[List[Mesh]]:
    n = len(meshes)
    boxes = [m.bbox() for m in meshes]
    uf = UF(n)
    for i in range(n):
        for j in range(i+1, n):
            # Never mix tiles with different materials — wrong texture for minority tiles
            if meshes[i].material != meshes[j].material:
                continue
            if bbox_gap(boxes[i], boxes[j]) <= proximity:
                uf.union(i, j)
    groups: Dict[int, List[Mesh]] = {}
    for i, m in enumerate(meshes):
        groups.setdefault(uf.find(i), []).append(m)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Vertex welding
# ---------------------------------------------------------------------------

def _bucket(v: Vec3, cell: float):
    return (int(math.floor(v[0]/cell)),
            int(math.floor(v[1]/cell)),
            int(math.floor(v[2]/cell)))


def merge_meshes(meshes: List[Mesh], weld: float) -> Mesh:
    all_v: List[Vec3]               = []
    all_n: List[Vec3]               = []
    all_uv:List[Tuple[float,float]] = []
    all_t: List[Tuple[int,int,int]] = []
    all_a: List[float]              = []
    off = 0
    for m in meshes:
        all_v.extend(m.verts); all_n.extend(m.normals); all_uv.extend(m.uvs)
        all_a.extend(m.alphas if m.alphas else [0.0] * len(m.verts))
        all_t.extend((i+off,j+off,k+off) for i,j,k in m.tris)
        off += len(m.verts)

    cell = max(weld, 0.01)
    bmap: Dict = {}
    for idx, v in enumerate(all_v):
        bmap.setdefault(_bucket(v, cell), []).append(idx)

    def nbrs(b):
        bx,by,bz = b
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    nb=(bx+dx,by+dy,bz+dz)
                    if nb in bmap: yield from bmap[nb]

    # UV weld threshold: only merge vertices whose UVs are also close.
    # Tiles sharing the same texture axes will have matching UVs at seams;
    # tiles with different axes will differ significantly → keep them separate
    # so each tile interpolates its own correct UVs without smearing.
    UV_THRESH = 2.0  # texels

    remap = list(range(len(all_v)))
    done: Set[int] = set()
    for i in range(len(all_v)):
        if i in done: continue
        v = all_v[i]
        for j in nbrs(_bucket(v, cell)):
            if j <= i or j in done: continue
            if vlen(vsub(v, all_v[j])) > weld: continue
            ui, vi_uv = all_uv[i]
            uj, vj_uv = all_uv[j]
            if abs(ui - uj) > UV_THRESH or abs(vi_uv - vj_uv) > UV_THRESH: continue
            remap[j] = i; done.add(j)
        done.add(i)

    canon: Dict[int,int] = {}
    nv: List[Vec3]               = []
    nn: List[Vec3]               = []
    nuv:List[Tuple[float,float]] = []
    na: List[float]              = []
    nacc: Dict[int, List[Vec3]]  = {}
    aacc: Dict[int, List[float]] = {}
    for i in range(len(all_v)):
        c = remap[i]
        if c not in canon:
            canon[c] = len(nv)
            nv.append(all_v[c]); nn.append(all_n[c]); nuv.append(all_uv[c]); na.append(all_a[c])
            nacc[canon[c]] = [all_n[c]]
            aacc[canon[c]] = [all_a[i]]
        else:
            nacc[canon[c]].append(all_n[i])
            aacc[canon[c]].append(all_a[i])

    for ni, lst in nacc.items():
        nn[ni] = vnorm((sum(x[0] for x in lst), sum(x[1] for x in lst), sum(x[2] for x in lst)))
    for ni, lst in aacc.items():
        na[ni] = sum(lst) / len(lst)

    nt: List[Tuple[int,int,int]] = []
    for i,j,k in all_t:
        a,b,c = canon[remap[i]], canon[remap[j]], canon[remap[k]]
        if a != b and b != c and a != c: nt.append((a,b,c))

    # Use the most common material in the group
    from collections import Counter
    dominant_mat = Counter(m.material for m in meshes).most_common(1)[0][0]

    densities = [m.tex_density for m in meshes if m.tex_density > 0]
    avg_density = sum(densities) / len(densities) if densities else 0.0
    return Mesh(verts=nv, normals=nn, uvs=nuv, tris=nt, material=dominant_mat, alphas=na,
                tex_density=avg_density)


# ---------------------------------------------------------------------------
# Clip brush (GoldSrc MAP) generator
# ---------------------------------------------------------------------------

def _prism_brush_z(a: Vec3, b: Vec3, c: Vec3, depth: float, tex: str) -> Optional[str]:
    """
    5-plane triangular prism of uniform thickness.
    Bottom = top shifted straight down by depth, so every triangle is exactly
    `depth` units thick and adjacent prisms share the same vertical side planes.
    """
    fn = vcross(vsub(b, a), vsub(c, a))
    if vlen(fn) < 1e-6:
        return None

    da: Vec3 = (a[0], a[1], a[2] - depth)
    db: Vec3 = (b[0], b[1], b[2] - depth)
    dc: Vec3 = (c[0], c[1], c[2] - depth)

    def fp(v: Vec3) -> str:
        return f'( {round(v[0])} {round(v[1])} {round(v[2])} )'

    def pl(p0: Vec3, p1: Vec3, p2: Vec3) -> str:
        return f'{fp(p0)} {fp(p1)} {fp(p2)} {tex} [ 1 0 0 0 ] [ 0 -1 0 0 ] 0 1 1'

    zp_a: Vec3 = (a[0], a[1], a[2] + 1)
    zp_b: Vec3 = (b[0], b[1], b[2] + 1)
    zp_c: Vec3 = (c[0], c[1], c[2] + 1)

    if fn[2] >= 0:
        # Upward-facing: CCW from above
        # top inward=down, bottom inward=up, sides inward toward centroid
        return (
            '{\n'
            + pl(a,  c,  b)    + '\n'   # top
            + pl(da, db, dc)   + '\n'   # bottom
            + pl(b,  a,  zp_b) + '\n'   # side a-b
            + pl(c,  b,  zp_c) + '\n'   # side b-c
            + pl(a,  c,  zp_a) + '\n'   # side c-a
            + '}'
        )
    else:
        # Downward-facing: CW from above — flip top, bottom, and side windings
        return (
            '{\n'
            + pl(a,  b,  c)    + '\n'   # top   (flipped)
            + pl(da, dc, db)   + '\n'   # bottom (flipped)
            + pl(a,  b,  zp_a) + '\n'   # side a-b
            + pl(b,  c,  zp_b) + '\n'   # side b-c
            + pl(c,  a,  zp_c) + '\n'   # side c-a
            + '}'
        )


def build_clip_brushes_map(sides_and_meshes: List[Tuple['DispSide', 'Mesh']],
                           depth: float = 64.0,
                           tex: str = 'CLIP') -> str:
    """
    Generate a Valve-220 .map file with triangular prism clip brushes.
    One prism per mesh triangle — maximum accuracy, follows every facet
    of the displaced surface.
    """
    valid = [(ds, m) for ds, m in sides_and_meshes if m.verts]
    if not valid:
        return ('// Game: Half-Life\n// Format: Valve\n'
                '// entity 0\n{\n"mapversion" "220"\n"classname" "worldspawn"\n}')

    brush_strs: List[str] = []

    for ds, mesh in valid:
        for ai, bi, ci in mesh.tris:
            a = mesh.verts[ai]
            b = mesh.verts[bi]
            c = mesh.verts[ci]
            brush = _prism_brush_z(a, b, c, depth=depth, tex=tex)
            if brush:
                brush_strs.append(brush)

    n_disps = len(valid)
    header = (
        '// Game: Half-Life\n'
        '// Format: Valve\n'
        f'// {len(brush_strs)} clip brushes from {n_disps} displacement(s)\n'
    )
    entity = (
        '// entity 0\n'
        '{\n'
        '"mapversion" "220"\n'
        '"classname" "worldspawn"\n'
        + '\n'.join(brush_strs)
        + '\n}'
    )
    return header + entity


def build_clip_brushes_map_opt(
        sides_and_meshes: List[Tuple['DispSide', 'Mesh']],
        depth: float = 64.0,
        tex: str = 'CLIP') -> str:
    """
    Optimised clip MAP: one convex-hull brush per displacement tile instead of
    one triangular prism per triangle.  Typically reduces GoldSrc clipnodes by
    ~90%.  Falls back to the per-triangle approach when scipy is unavailable.
    """
    try:
        from scipy.spatial import ConvexHull
        import numpy as np
    except ImportError:
        return build_clip_brushes_map(sides_and_meshes, depth=depth, tex=tex)

    _EMPTY = ('// Game: Half-Life\n// Format: Valve\n'
              '// entity 0\n{\n"mapversion" "220"\n"classname" "worldspawn"\n}')

    valid = [(ds, m) for ds, m in sides_and_meshes if m.verts]
    if not valid:
        return _EMPTY

    def fp(v) -> str:
        return f'( {round(float(v[0]))} {round(float(v[1]))} {round(float(v[2]))} )'

    brush_strs: List[str] = []

    for _ds, mesh in valid:
        # Point cloud: terrain surface + depth-shifted underside
        pts = []
        for vx, vy, vz in mesh.verts:
            pts.append([vx, vy, vz])
            pts.append([vx, vy, vz - depth])

        arr = np.array(pts, dtype=np.float64)
        if arr.shape[0] < 4:
            continue

        try:
            hull = ConvexHull(arr, qhull_options='QJ')
        except Exception:
            continue

        lines = ['{']
        for i, simplex in enumerate(hull.simplices):
            p0 = arr[simplex[0]]
            p1 = arr[simplex[1]]
            p2 = arr[simplex[2]]
            outward = hull.equations[i, :3]
            # Valve-220: (p1-p0)×(p2-p0) = INWARD normal = opposite of outward
            if np.dot(np.cross(p1 - p0, p2 - p0), outward) > 0:
                p1, p2 = p2, p1
            lines.append(
                f'{fp(p0)} {fp(p1)} {fp(p2)} '
                f'{tex} [ 1 0 0 0 ] [ 0 -1 0 0 ] 0 1 1'
            )
        lines.append('}')
        brush_strs.append('\n'.join(lines))

    if not brush_strs:
        return _EMPTY

    n_disps = len(valid)
    header = (
        '// Game: Half-Life\n'
        '// Format: Valve\n'
        f'// {len(brush_strs)} optimised clip brushes from {n_disps} displacement(s)\n'
    )
    entity = (
        '// entity 0\n{\n"mapversion" "220"\n"classname" "worldspawn"\n'
        + '\n'.join(brush_strs)
        + '\n}'
    )
    return header + entity


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------

def _mtl_name(material: str) -> str:
    """Sanitise a Source material path into a valid MTL material name."""
    return material.replace('/', '_').replace('\\', '_').replace(' ', '_')


def write_obj(path: Path, mesh: Mesh, name: str) -> None:
    """
    Write a Wavefront OBJ + companion MTL file.
    - Coordinates kept in Source units; Y/Z swapped for Blender (Z-forward, Y-up).
    - UVs are in Source texel space (dot-product projection); they tile correctly
      once the matching texture is assigned in Blender.
    """
    mtl_filename = path.with_suffix('.mtl').name
    mat_name     = _mtl_name(mesh.material)

    # MTL
    with open(path.with_suffix('.mtl'), 'w') as f:
        f.write(f"# vmf_disp_to_obj — {name}\n")
        f.write(f"newmtl {mat_name}\n")
        f.write(f"# Source material: {mesh.material}\n")
        # map_Kd points to where the user should put the converted texture
        tex_leaf = mesh.material.replace('\\', '/').split('/')[-1]
        f.write(f"map_Kd {tex_leaf}.bmp\n")

    with open(path, 'w') as f:
        f.write(f"# vmf_disp_to_obj — {name}\n")
        f.write(f"mtllib {mtl_filename}\n")
        f.write(f"o {name}\n\n")

        # Vertices: swap Y/Z for Blender Y-up
        for (x, y, z) in mesh.verts:
            f.write(f"v {x:.4f} {z:.4f} {-y:.4f}\n")
        f.write("\n")

        # UVs — texel-space projection from VMF texture axes
        for (u, v) in mesh.uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write("\n")

        # Normals (same axis swap)
        for (nx, ny, nz) in mesh.normals:
            f.write(f"vn {nx:.6f} {nz:.6f} {-ny:.6f}\n")
        f.write("\n")

        f.write(f"usemtl {mat_name}\n")
        for (i, j, k) in mesh.tris:
            i1, j1, k1 = i+1, j+1, k+1
            f.write(f"f {i1}/{i1}/{i1} {j1}/{j1}/{j1} {k1}/{k1}/{k1}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Convert VMF displacement surfaces to Wavefront OBJ.')
    ap.add_argument('vmf', help='Path to .vmf file')
    ap.add_argument('-o', '--output', default='disp_obj',
                    help='Output directory (default: ./disp_obj)')
    ap.add_argument('--no-merge', action='store_true',
                    help='One OBJ per tile, no grouping')
    ap.add_argument('--proximity', type=float, default=4.0,
                    help='Tile grouping proximity in Source units (default: 4.0)')
    ap.add_argument('--weld', type=float, default=1.0,
                    help='Vertex weld tolerance in Source units (default: 1.0)')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    vmf_path = Path(args.vmf)
    if not vmf_path.exists():
        sys.exit(f"Error: {vmf_path} not found")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {vmf_path} ...")
    text = vmf_path.read_text(encoding='utf-8', errors='replace')

    print("Parsing VMF ...")
    blocks = VMFParser(text).parse()

    print("Extracting displacements ...")
    sides = extract_disp_sides(blocks)
    if not sides:
        sys.exit("No displacement surfaces found.")
    print(f"Found {len(sides)} displacement surface(s).")

    print("Building meshes ...")
    meshes: List[Mesh] = []
    labels: List[str]  = []
    for ds in sides:
        try:
            meshes.append(build_mesh(ds))
            labels.append(f"disp_{ds.solid_id}_{ds.side_id}")
        except Exception as e:
            print(f"  WARNING: solid={ds.solid_id} side={ds.side_id}: {e}")

    # ---- Mode A: one OBJ per tile -------------------------------------------
    if args.no_merge:
        print(f"Writing {len(meshes)} OBJ files ...")
        for m, lbl in zip(meshes, labels):
            write_obj(out / f"{lbl}.obj", m, lbl)
            if args.verbose:
                print(f"  {lbl}.obj  {len(m.verts)}v {len(m.tris)}t")
        print(f"\nDone -> {out}/")
        return

    # ---- Mode B: group by proximity, weld seams -----------------------------
    print(f"Grouping by proximity ({args.proximity} units) ...")
    groups = group_meshes(meshes, args.proximity)
    print(f"{len(groups)} group(s) found.  Welding seams ({args.weld} units) ...")

    for gi, grp in enumerate(groups):
        name = f"terrain_group_{gi}"
        merged = merge_meshes(grp, args.weld)
        write_obj(out / f"{name}.obj", merged, name)
        print(f"  {name}.obj  ({len(grp)} tile(s), "
              f"{len(merged.verts)}v, {len(merged.tris)}t)")
        if args.verbose:
            for m, lbl in zip(meshes, labels):
                if m in grp:
                    print(f"    {lbl}")

    print(f"\nDone -> {out}/")
    print()
    print("Import into Blender:")
    print("  File > Import > Wavefront (.obj)")
    print("  Forward axis: -Z   Up axis: Y   (Blender default OBJ settings)")


if __name__ == '__main__':
    main()
