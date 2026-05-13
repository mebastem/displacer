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


@dataclass
class DispSide:
    solid_id:      int
    side_id:       int
    plane_pts:     List[Vec3]        # 3 points defining the face plane (VMF CCW winding)
    sibling_planes: List[List[Vec3]] # all planes in the parent solid
    dispinfo:      DispInfo


@dataclass
class Mesh:
    verts:    List[Vec3]
    normals:  List[Vec3]
    uvs:      List[Tuple[float,float]]
    tris:     List[Tuple[int,int,int]]  # 0-based vertex indices

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
        t = self._tok[self._pos]; self._pos += 1; return t
    def _uq(self, s): return s.strip('"')

    def _block(self) -> dict:
        assert self._next() == '{', "expected {"
        d: dict = {}
        while self._peek() != '}':
            k = self._uq(self._next())
            v = self._block() if self._peek() == '{' else self._uq(self._next())
            if k in d:
                d[k] = d[k] if isinstance(d[k], list) else [d[k]]
                d[k].append(v)
            else:
                d[k] = v
        self._next()  # consume '}'
        return d

    def parse(self) -> List[dict]:
        out = []
        while self._peek():
            name = self._uq(self._next())
            b = self._block(); b['__name__'] = name
            out.append(b)
        return out


# ---------------------------------------------------------------------------
# VMF helpers
# ---------------------------------------------------------------------------

def parse_vec3(s: str) -> Vec3:
    n = re.findall(r'-?[\d.eE+\-]+', s)
    return float(n[0]), float(n[1]), float(n[2])

def parse_plane(s: str) -> List[Vec3]:
    return [parse_vec3(g) for g in re.findall(r'\(([^)]+)\)', s)]

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
                    offsets=rows_vecs('offsets', (0., 0., 0.)))


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
                        results.append(DispSide(solid_id=sid, side_id=side_id,
                                                plane_pts=pts, sibling_planes=all_planes,
                                                dispinfo=di))
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

def recover_corners(plane_pts: List[Vec3],
                    sibling_planes: List[List[Vec3]],
                    start_pos: Vec3) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    """
    Recover the 4 grid corners from the VMF plane points.

    p0, p1, p2  are 3 of the 4 actual face corners stored in the VMF plane
    (CW order viewed from outside the brush).  The 4th corner is found by
    intersecting the two sibling brush planes that share edges with p0 and p2
    respectively.  If that fails we fall back to the parallelogram formula.

    After recovering all 4 corners we rotate them so the one nearest
    startposition comes first (CW), giving:
        corners[0] = c_start
        corners[1] = c_row   (first CW step from start)
        corners[2] = c_diag
        corners[3] = c_col   (third CW step from start)
    """
    p0, p1, p2 = plane_pts[0], plane_pts[1], plane_pts[2]

    # --- Find the true 4th corner via sibling planes -------------------------
    face_n, face_d = _plane_nd([p0, p1, p2])
    THRESH = 1.0   # Source units; plane points are exact integers typically

    def on_sib(pt, sn, sd) -> bool:
        return abs(vdot(sn, pt) - sd) < THRESH

    # Collect non-parallel sibling planes.
    non_par = []
    for sib in sibling_planes:
        if len(sib) < 3:
            continue
        sn, sd = _plane_nd(sib)
        if 1.0 - abs(vdot(face_n, sn)) < 1e-6:
            continue   # parallel to face (face itself or back face)
        non_par.append((sn, sd))

    # Try every pair of sibling planes.  Each pair's 3-plane intersection with
    # the face plane gives a candidate vertex.  We keep the candidate that is
    # (a) not one of the 3 known corners and (b) closest to start_pos.
    p3: Optional[Vec3] = None
    best_dist = float('inf')
    for i in range(len(non_par)):
        for j in range(i + 1, len(non_par)):
            cand = _intersect3(face_n, face_d,
                               non_par[i][0], non_par[i][1],
                               non_par[j][0], non_par[j][1])
            if cand is None:
                continue
            # Skip if it coincides with a known corner
            if min(vlen(vsub(cand, pt)) for pt in [p0, p1, p2]) < THRESH:
                continue
            d = vlen(vsub(cand, start_pos))
            if d < best_dist:
                best_dist = d
                p3 = cand

    if p3 is None:
        # Fallback: parallelogram assumption (correct for rectangular brushes)
        p3 = vadd(p0, vsub(p2, p1))

    # Sort all 4 corners into the correct CW order (viewed from outside).
    # p0,p1,p2 are already CW; we find where p3 belongs by checking which
    # insertion position produces a convex polygon (all cross products point
    # in the same inward direction as face_n).
    def _try_order(pts):
        n = len(pts)
        for k in range(n):
            a, b, c = pts[k], pts[(k+1) % n], pts[(k+2) % n]
            if vdot(vcross(vsub(b, a), vsub(c, a)), face_n) < -1e-6:
                return False
        return True

    corners: Optional[List[Vec3]] = None
    for pos in range(4):
        cand = [p0, p1, p2]
        cand.insert(pos, p3)
        if _try_order(cand):
            corners = cand
            break
    if corners is None:
        corners = [p0, p1, p2, p3]   # shouldn't happen for valid geometry

    # Rotate so the corner nearest startposition comes first, preserving CW order
    dists = [vlen(vsub(c, start_pos)) for c in corners]
    si = dists.index(min(dists))
    corners = corners[si:] + corners[:si]

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

    verts: List[Vec3]               = []
    uvs:   List[Tuple[float,float]] = []

    for row in range(size):
        t = row / (size - 1)
        edge_a = vlerp(c_start, c_row,  t)
        edge_b = vlerp(c_col,   c_diag, t)
        for col in range(size):
            s = col / (size - 1)
            base = vlerp(edge_a, edge_b, s)
            n   = di.normals[row][col]
            d   = di.distances[row][col]
            off = di.offsets[row][col]
            pos = vadd(base, vadd(vscale(n, d), off))
            verts.append(pos)
            uvs.append((s, t))

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

    return Mesh(verts=verts, normals=norms, uvs=uvs, tris=tris)


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
    off = 0
    for m in meshes:
        all_v.extend(m.verts); all_n.extend(m.normals); all_uv.extend(m.uvs)
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

    remap = list(range(len(all_v)))
    done: Set[int] = set()
    for i in range(len(all_v)):
        if i in done: continue
        v = all_v[i]
        for j in nbrs(_bucket(v, cell)):
            if j <= i or j in done: continue
            if vlen(vsub(v, all_v[j])) <= weld:
                remap[j] = i; done.add(j)
        done.add(i)

    canon: Dict[int,int] = {}
    nv: List[Vec3]               = []
    nn: List[Vec3]               = []
    nuv:List[Tuple[float,float]] = []
    nacc: Dict[int, List[Vec3]]  = {}
    for i in range(len(all_v)):
        c = remap[i]
        if c not in canon:
            canon[c] = len(nv)
            nv.append(all_v[c]); nn.append(all_n[c]); nuv.append(all_uv[c])
            nacc[canon[c]] = [all_n[c]]
        else:
            nacc[canon[c]].append(all_n[i])

    for ni, lst in nacc.items():
        nn[ni] = vnorm((sum(x[0] for x in lst), sum(x[1] for x in lst), sum(x[2] for x in lst)))

    nt: List[Tuple[int,int,int]] = []
    for i,j,k in all_t:
        a,b,c = canon[remap[i]], canon[remap[j]], canon[remap[k]]
        if a != b and b != c and a != c: nt.append((a,b,c))

    return Mesh(verts=nv, normals=nn, uvs=nuv, tris=nt)


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------

def write_obj(path: Path, mesh: Mesh, name: str) -> None:
    """
    Write a Wavefront OBJ.
    - Coordinates are kept in Source units (no scaling).
    - Y-up: Source uses Z-up, so we swap Y and Z so Blender shows it correctly
      when imported with default settings (Z-forward, Y-up).
    """
    with open(path, 'w') as f:
        f.write(f"# vmf_disp_to_obj — {name}\n")
        f.write(f"o {name}\n\n")

        # Vertices: swap Y/Z to convert Source (Z-up) to Blender (Y-up)
        for (x, y, z) in mesh.verts:
            f.write(f"v {x:.4f} {z:.4f} {-y:.4f}\n")
        f.write("\n")

        # UVs
        for (u, v) in mesh.uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write("\n")

        # Normals (same axis swap)
        for (nx, ny, nz) in mesh.normals:
            f.write(f"vn {nx:.6f} {nz:.6f} {-ny:.6f}\n")
        f.write("\n")

        # Faces — OBJ is 1-indexed; format: v/vt/vn
        f.write("usemtl displacement\n")
        for (i, j, k) in mesh.tris:
            # Each vertex uses its own index for pos, uv, and normal
            # (they share the same index since we store them in lockstep)
            i1,j1,k1 = i+1, j+1, k+1
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