#!/usr/bin/env python3
"""
vmf_disp_to_smd.py
------------------
Converts Source Engine (.vmf) displacement surfaces into GoldSrc-compatible
Valve SMD meshes.  Adjacent displacement tiles are automatically grouped by
proximity and merged into single models, with shared edge vertices welded
together to eliminate seams.

Usage:
    python vmf_disp_to_smd.py <map.vmf> [options]

Options:
    -o, --output DIR        Output directory (default: ./disp_models)
    -s, --scale FLOAT       Unit scale factor Source→GoldSrc (default: 0.05)
                            Typical range: 0.05–0.1 depending on map scale.
    -t, --texture NAME      Texture name written into SMD/QC (default: "displacement")
    --proximity FLOAT       Max bounding-box distance (Source units) between two tiles
                            to consider them part of the same group (default: 4.0)
    --weld FLOAT            Max distance (Source units, pre-scale) between two vertices
                            to merge them when stitching tiles together (default: 1.0)
    --no-merge              Disable grouping/welding; output one SMD per displacement
    --no-qc                 Skip generating .qc files
    --poly-limit INT        Warn if a group exceeds this triangle count (default: 20000)
    -v, --verbose           Print per-tile and per-group debug info

Output per group:
    terrain_group_<N>.smd   — merged mesh in reference pose
    terrain_group_<N>.qc    — studiomdl compile script

Without --no-merge, individual per-tile files are NOT written.
With    --no-merge, outputs are named disp_<solid_id>_<side_id>.smd/.qc
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Vec3 helpers
# ---------------------------------------------------------------------------

Vec3 = Tuple[float, float, float]

def vec3_add(a: Vec3, b: Vec3) -> Vec3:    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def vec3_sub(a: Vec3, b: Vec3) -> Vec3:    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vec3_dot(a: Vec3, b: Vec3) -> float:    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def vec3_scale(v: Vec3, s: float) -> Vec3: return (v[0]*s, v[1]*s, v[2]*s)
def vec3_len(v: Vec3) -> float:            return math.sqrt(v[0]**2+v[1]**2+v[2]**2)

def vec3_norm(v: Vec3) -> Vec3:
    l = vec3_len(v)
    return (v[0]/l, v[1]/l, v[2]/l) if l > 1e-9 else (0.0, 0.0, 1.0)

def vec3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return vec3_add(vec3_scale(a, 1-t), vec3_scale(b, t))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DispInfo:
    power: int
    start_position: Vec3 = (0, 0, 0)
    normals:   List[List[Vec3]]  = field(default_factory=list)
    distances: List[List[float]] = field(default_factory=list)
    offsets:   List[List[Vec3]]  = field(default_factory=list)


@dataclass
class Side:
    solid_id: int
    side_id: int
    plane_points: List[Vec3] = field(default_factory=list)
    dispinfo: Optional[DispInfo] = None
    # All planes of the parent brush solid (including this face's own plane).
    # Used to clip this face's infinite plane down to its actual quad corners.
    sibling_planes: List[List[Vec3]] = field(default_factory=list)


@dataclass
class TileMesh:
    """Fully built mesh for one displacement tile, stored in raw Source units."""
    side: Side
    vertices:     List[Vec3]
    triangles:    List[Tuple[int, int, int]]
    vert_normals: List[Vec3]
    uv_coords:    List[Tuple[float, float]]

    def bbox(self) -> Tuple[Vec3, Vec3]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# ---------------------------------------------------------------------------
# VMF parser
# ---------------------------------------------------------------------------

class VMFParser:
    _COMMENT = re.compile(r'//[^\n]*')
    _TOKEN   = re.compile(r'"[^"]*"|\{|\}|[^\s"{}]+')

    def __init__(self, text: str):
        self._tokens = self._TOKEN.findall(self._COMMENT.sub('', text))
        self._pos = 0

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        t = self._tokens[self._pos]; self._pos += 1; return t

    def _strip(self, s: str) -> str:
        return s.strip('"')

    def _parse_block(self) -> dict:
        result: dict = {}
        assert self._consume() == '{', "Expected '{'"
        while self._peek() != '}':
            key = self._strip(self._consume())
            value = self._parse_block() if self._peek() == '{' else self._strip(self._consume())
            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(value)
            else:
                result[key] = value
        self._consume()
        return result

    def parse(self) -> List[dict]:
        blocks = []
        while self._peek() is not None:
            name = self._strip(self._consume())
            block = self._parse_block()
            block['__name__'] = name
            blocks.append(block)
        return blocks


# ---------------------------------------------------------------------------
# VMF extraction
# ---------------------------------------------------------------------------

def parse_vec3(s: str) -> Vec3:
    n = re.findall(r'-?[\d.eE+\-]+', s)
    return (float(n[0]), float(n[1]), float(n[2]))

def parse_plane(s: str) -> List[Vec3]:
    return [parse_vec3(g) for g in re.findall(r'\(([^)]+)\)', s)]

def parse_row_vecs(s: str, n: int) -> List[Vec3]:
    nums = [float(x) for x in s.split()]
    return [(nums[i*3], nums[i*3+1], nums[i*3+2]) for i in range(n)]

def parse_row_floats(s: str, n: int) -> List[float]:
    return [float(x) for x in s.split()][:n]


def extract_dispinfo(d: dict, power: int) -> DispInfo:
    size = (1 << power) + 1
    start_pos = parse_vec3(d.get('startposition', '0 0 0'))

    def get_rows(block_key, parser, default):
        blk = d.get(block_key, {})
        rows = []
        for r in range(size):
            s = blk.get(f'row{r}', '')
            rows.append(parser(s, size) if s else [default] * size)
        return rows

    normals   = get_rows('normals',   parse_row_vecs,   (0.0, 0.0, 1.0))
    distances = get_rows('distances', parse_row_floats, 0.0)
    offsets   = get_rows('offsets',   parse_row_vecs,   (0.0, 0.0, 0.0))
    return DispInfo(power=power, start_position=start_pos,
                    normals=normals, distances=distances, offsets=offsets)


def extract_sides(blocks: List[dict]) -> List[Side]:
    sides: List[Side] = []

    def walk(block: dict, solid_id: int = -1):
        name = block.get('__name__', '')
        if name == 'solid':
            try: solid_id = int(block.get('id', -1))
            except ValueError: pass

        for k, v in block.items():
            if k == '__name__': continue
            for item in (v if isinstance(v, list) else [v]):
                if not isinstance(item, dict): continue
                item_name = item.get('__name__', k)

                if item_name == 'solid' or k == 'solid':
                    sid = int(item.get('id', solid_id)) if 'id' in item else solid_id

                    # Collect ALL side planes in this solid first, then process
                    # displacement sides — so we can pass sibling planes in.
                    all_side_items = item.get('side', [])
                    if not isinstance(all_side_items, list):
                        all_side_items = [all_side_items]
                    all_planes = [parse_plane(s.get('plane', ''))
                                  for s in all_side_items
                                  if isinstance(s, dict) and s.get('plane')]

                    for s_item in all_side_items:
                        if not isinstance(s_item, dict): continue
                        side_id   = int(s_item.get('id', -1)) if 'id' in s_item else -1
                        plane_pts = parse_plane(s_item.get('plane', ''))
                        disp = None
                        if 'dispinfo' in s_item:
                            di_raw = s_item['dispinfo']
                            if isinstance(di_raw, list): di_raw = di_raw[0]
                            power = int(di_raw.get('power', 2))
                            disp = extract_dispinfo(di_raw, power)
                        if disp is not None:
                            sides.append(Side(solid_id=sid, side_id=side_id,
                                              plane_points=plane_pts, dispinfo=disp,
                                              sibling_planes=all_planes))
                    # Still recurse for nested solids
                    walk(item, sid)
                elif item_name != 'side' and k != 'side':
                    walk(item, solid_id)

    for block in blocks:
        walk(block)
    return sides


# ---------------------------------------------------------------------------
# Displacement geometry builder
# ---------------------------------------------------------------------------

def vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def build_tile_mesh(side: Side) -> TileMesh:
    """
    Build the displaced mesh in raw Source units (scale applied later).

    Corner layout (matching Source engine convention):
        c0 = startposition corner
        c1 = corner along the first edge from c0  (col direction)
        c2 = diagonally opposite corner
        c3 = corner along the second edge from c0 (row direction)

    Grid traversal:
        row increases c0→c3 (one edge of the quad)
        col increases c0→c1 (adjacent edge)

    This matches how Source stores displacement normals/distances in row-major
    order starting from startposition.
    """
    di = side.dispinfo
    size = (1 << di.power) + 1

    # Recover the 4 face corners.
    # The VMF stores 3 of the 4 corners (p0,p1,p2).  We find the true 4th
    # corner by intersecting the two sibling brush planes that share edges with
    # p0 and p2 respectively.  Falls back to parallelogram if that fails.
    p0, p1, p2 = side.plane_points[0], side.plane_points[1], side.plane_points[2]

    def _plane_nd(pts):
        a, b, c = pts[0], pts[1], pts[2]
        n = vec3_norm(vec3_cross(vec3_sub(b, a), vec3_sub(c, a)))
        return n, vec3_dot(n, a)

    def _intersect3(n1, d1, n2, d2, n3, d3):
        c23 = vec3_cross(n2, n3)
        det = vec3_dot(n1, c23)
        if abs(det) < 1e-9:
            return None
        c31 = vec3_cross(n3, n1)
        c12 = vec3_cross(n1, n2)
        return vec3_scale(
            vec3_add(vec3_add(vec3_scale(c23, d1), vec3_scale(c31, d2)), vec3_scale(c12, d3)),
            1.0 / det)

    face_n, face_d = _plane_nd([p0, p1, p2])
    THRESH = 1.0

    # Collect non-parallel sibling planes
    non_par = []
    for sib in side.sibling_planes:
        if len(sib) < 3:
            continue
        sn, sd = _plane_nd(sib)
        if 1.0 - abs(vec3_dot(face_n, sn)) < 1e-6:
            continue
        non_par.append((sn, sd))

    # Try every pair; find the candidate vertex that is not a known corner
    # and is closest to start_position (= the true 4th corner C3).
    p3 = None
    best_dist = float('inf')
    for i in range(len(non_par)):
        for j in range(i + 1, len(non_par)):
            cand = _intersect3(face_n, face_d,
                               non_par[i][0], non_par[i][1],
                               non_par[j][0], non_par[j][1])
            if cand is None:
                continue
            if min(vec3_len(vec3_sub(cand, pt)) for pt in [p0, p1, p2]) < THRESH:
                continue
            d = vec3_len(vec3_sub(cand, di.start_position))
            if d < best_dist:
                best_dist = d
                p3 = cand

    if p3 is None:
        p3 = vec3_add(p0, vec3_sub(p2, p1))

    # Insert p3 at the correct position in the CW sequence [p0,p1,p2].
    def _try_order(pts):
        n = len(pts)
        for k in range(n):
            a, b, c = pts[k], pts[(k+1) % n], pts[(k+2) % n]
            if vec3_dot(vec3_cross(vec3_sub(b, a), vec3_sub(c, a)), face_n) < -1e-6:
                return False
        return True

    corners = None
    for pos in range(4):
        cand = [p0, p1, p2]
        cand.insert(pos, p3)
        if _try_order(cand):
            corners = cand
            break
    if corners is None:
        corners = [p0, p1, p2, p3]

    # Rotate so the corner nearest startposition is first, preserving CW order.
    dists = [vec3_len(vec3_sub(c, di.start_position)) for c in corners]
    si    = dists.index(min(dists))
    corners = corners[si:] + corners[:si]
    c_start, c_row, c_diag, c_col = corners[0], corners[1], corners[2], corners[3]

    vertices:  List[Vec3]               = []
    uv_coords: List[Tuple[float, float]] = []

    for row in range(size):
        t = row / (size - 1)
        edge_a = vec3_lerp(c_start, c_row,  t)   # row direction edge
        edge_b = vec3_lerp(c_col,   c_diag, t)   # opposite edge
        for col in range(size):
            s = col / (size - 1)
            base      = vec3_lerp(edge_a, edge_b, s)
            n         = di.normals[row][col]
            d         = di.distances[row][col]
            off       = di.offsets[row][col]
            displaced = vec3_add(base, vec3_add(vec3_scale(n, d), off))
            vertices.append(displaced)
            uv_coords.append((s, t))

    triangles: List[Tuple[int, int, int]] = []
    for row in range(size - 1):
        for col in range(size - 1):
            i00 = row * size + col
            i10 = (row+1) * size + col
            i01 = row * size + (col+1)
            i11 = (row+1) * size + (col+1)
            if (row + col) % 2 == 0:
                triangles += [(i00, i10, i11), (i00, i11, i01)]
            else:
                triangles += [(i00, i10, i01), (i10, i11, i01)]

    # Compute proper per-vertex normals from triangle geometry
    # (area-weighted accumulation gives smooth normals at shared vertices).
    accum: List[Vec3] = [(0., 0., 0.)] * len(vertices)
    for (i, j, k) in triangles:
        ab = vec3_sub(vertices[j], vertices[i])
        ac = vec3_sub(vertices[k], vertices[i])
        fn = vec3_cross(ab, ac)
        accum[i] = vec3_add(accum[i], fn)
        accum[j] = vec3_add(accum[j], fn)
        accum[k] = vec3_add(accum[k], fn)
    vert_normals: List[Vec3] = [vec3_norm(n) for n in accum]

    return TileMesh(side=side, vertices=vertices, triangles=triangles,
                    vert_normals=vert_normals, uv_coords=uv_coords)


# ---------------------------------------------------------------------------
# Proximity grouping  (Union-Find)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[b] = a


def bbox_distance(a: Tuple[Vec3, Vec3], b: Tuple[Vec3, Vec3]) -> float:
    """
    Minimum axis-aligned distance between two bounding boxes.
    Returns 0.0 if they overlap or touch on any axis.
    """
    dist_sq = 0.0
    for i in range(3):
        gap = max(0.0, max(a[0][i], b[0][i]) - min(a[1][i], b[1][i]))
        dist_sq += gap * gap
    return math.sqrt(dist_sq)


def group_tiles(tiles: List[TileMesh], proximity: float) -> List[List[TileMesh]]:
    """
    Group tiles whose bounding boxes are within `proximity` Source units of
    each other.  Uses Union-Find for O(n²) worst case but n is small in practice.
    """
    n = len(tiles)
    bboxes = [t.bbox() for t in tiles]
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i+1, n):
            if bbox_distance(bboxes[i], bboxes[j]) <= proximity:
                uf.union(i, j)

    groups: Dict[int, List[TileMesh]] = {}
    for i, tile in enumerate(tiles):
        groups.setdefault(uf.find(i), []).append(tile)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Vertex welding  (spatial hash grid)
# ---------------------------------------------------------------------------

def _bucket(v: Vec3, cell: float) -> Tuple[int, int, int]:
    return (int(math.floor(v[0] / cell)),
            int(math.floor(v[1] / cell)),
            int(math.floor(v[2] / cell)))


def merge_tiles(tiles: List[TileMesh], weld_threshold: float, scale: float):
    """
    Merge a group of tiles into one mesh:
      1. Concatenate all vertices (still in Source units).
      2. Build a spatial hash grid; merge vertex pairs within weld_threshold.
      3. Average normals at welded vertices (smooth seam shading).
      4. Remap + deduplicate triangles; discard degenerate ones.
      5. Apply scale to final vertex positions.

    Returns (vertices, triangles, vert_normals, uv_coords) — all scaled.
    """
    all_verts:  List[Vec3]               = []
    all_norms:  List[Vec3]               = []
    all_uvs:    List[Tuple[float,float]] = []
    all_tris:   List[Tuple[int,int,int]] = []

    offset = 0
    for tile in tiles:
        all_verts.extend(tile.vertices)
        all_norms.extend(tile.vert_normals)
        all_uvs.extend(tile.uv_coords)
        all_tris.extend((i+offset, j+offset, k+offset) for (i,j,k) in tile.triangles)
        offset += len(tile.vertices)

    total = len(all_verts)
    cell  = max(weld_threshold, 0.01)

    # Build spatial hash
    bucket_map: Dict[Tuple[int,int,int], List[int]] = {}
    for idx, v in enumerate(all_verts):
        bucket_map.setdefault(_bucket(v, cell), []).append(idx)

    def neighbours(b):
        bx, by, bz = b
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = (bx+dx, by+dy, bz+dz)
                    if nb in bucket_map:
                        yield from bucket_map[nb]

    # remap[i] = canonical index for vertex i
    remap: List[int] = list(range(total))
    processed: Set[int] = set()

    for idx in range(total):
        if idx in processed:
            continue
        v = all_verts[idx]
        for nb in neighbours(_bucket(v, cell)):
            if nb <= idx or nb in processed:
                continue
            if vec3_len(vec3_sub(v, all_verts[nb])) <= weld_threshold:
                remap[nb] = idx
                processed.add(nb)
        processed.add(idx)

    # Build new vertex arrays — canonical vertices only
    canonical: Dict[int, int] = {}   # old canonical idx → new idx
    new_verts:  List[Vec3]               = []
    new_norms:  List[Vec3]               = []
    new_uvs:    List[Tuple[float,float]] = []
    norm_accum: Dict[int, List[Vec3]]    = {}

    for i in range(total):
        c = remap[i]
        if c not in canonical:
            canonical[c] = len(new_verts)
            new_verts.append(vec3_scale(all_verts[c], scale))
            new_norms.append(all_norms[c])
            new_uvs.append(all_uvs[c])
            norm_accum[canonical[c]] = [all_norms[c]]
        else:
            norm_accum[canonical[c]].append(all_norms[i])

    # Average normals at welded vertices for smooth shading across seams
    for new_idx, nlist in norm_accum.items():
        ax = sum(n[0] for n in nlist)
        ay = sum(n[1] for n in nlist)
        az = sum(n[2] for n in nlist)
        new_norms[new_idx] = vec3_norm((ax, ay, az))

    # Remap triangles; discard degenerate triangles (all three verts became one)
    new_tris: List[Tuple[int,int,int]] = []
    for (i, j, k) in all_tris:
        ni = canonical[remap[i]]
        nj = canonical[remap[j]]
        nk = canonical[remap[k]]
        if ni != nj and nj != nk and ni != nk:
            new_tris.append((ni, nj, nk))

    return new_verts, new_tris, new_norms, new_uvs


# ---------------------------------------------------------------------------
# SMD / QC writers
# ---------------------------------------------------------------------------

SMD_HEADER = """\
version 1
nodes
0 "root" -1
end
skeleton
time 0
0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
end
triangles
"""

def write_smd(path: Path, vertices: List[Vec3], triangles: List[Tuple],
              vert_normals: List[Vec3], uv_coords: List[Tuple],
              texture: str) -> None:
    with open(path, 'w') as f:
        f.write(SMD_HEADER)
        for (i0, i1, i2) in triangles:
            f.write(f"{texture}\n")
            for idx in (i0, i1, i2):
                x, y, z    = vertices[idx]
                nx, ny, nz = vert_normals[idx]
                u, v       = uv_coords[idx]
                f.write(f"0  {x:.6f} {y:.6f} {z:.6f}  "
                        f"{nx:.6f} {ny:.6f} {nz:.6f}  "
                        f"{u:.6f} {v:.6f}\n")
        f.write("end\n")


QC_TEMPLATE = """\
$modelname "{model_name}.mdl"
$cd "."
$cdtexture "."
$scale 1.0
$bbox 0 0 0 0 0 0
$cbox 0 0 0 0 0 0
$eyeposition 0 0 0

$body "Body" "{smd_name}"

$sequence "idle" "{smd_name}" fps 1
"""

def write_qc(path: Path, model_name: str, smd_name: str) -> None:
    with open(path, 'w') as f:
        f.write(QC_TEMPLATE.format(model_name=model_name, smd_name=smd_name))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Convert Source VMF displacements to GoldSrc SMD, '
                    'merging adjacent tiles into seamless groups.'
    )
    ap.add_argument('vmf',                            help='Path to the .vmf file')
    ap.add_argument('-o', '--output', default='disp_models',
                    help='Output directory (default: ./disp_models)')
    ap.add_argument('-s', '--scale',  type=float, default=0.05,
                    help='Unit scale Source→GoldSrc (default: 0.05)')
    ap.add_argument('-t', '--texture', default='displacement',
                    help='Texture name for SMD/QC (default: "displacement")')
    ap.add_argument('--proximity', type=float, default=4.0,
                    help='Max bbox gap in Source units to group tiles (default: 4.0)')
    ap.add_argument('--weld',      type=float, default=1.0,
                    help='Vertex weld tolerance in Source units, pre-scale (default: 1.0)')
    ap.add_argument('--no-merge',  action='store_true',
                    help='Output one SMD per tile, no grouping or welding')
    ap.add_argument('--no-qc',     action='store_true',
                    help='Skip .qc file generation')
    ap.add_argument('--poly-limit', type=int, default=20000,
                    help='Warn if a group exceeds this triangle count (default: 20000)')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    vmf_path = Path(args.vmf)
    if not vmf_path.exists():
        print(f"Error: file not found: {vmf_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {vmf_path} ...")
    text = vmf_path.read_text(encoding='utf-8', errors='replace')

    print("Parsing VMF ...")
    blocks = VMFParser(text).parse()

    print("Extracting displacement sides ...")
    sides = extract_sides(blocks)
    if not sides:
        print("No displacement surfaces found in this VMF.")
        sys.exit(0)
    print(f"Found {len(sides)} displacement surface(s).")

    # Build individual tile meshes in Source units
    tiles: List[TileMesh] = []
    for side in sides:
        try:
            tiles.append(build_tile_mesh(side))
        except Exception as e:
            print(f"  WARNING: skipping solid {side.solid_id} / side {side.side_id}: {e}")

    # ------------------------------------------------------------------
    # Mode A: no merging
    # ------------------------------------------------------------------
    if args.no_merge:
        print(f"Writing {len(tiles)} individual SMD file(s) ...")
        for tile in tiles:
            name = f"disp_{tile.side.solid_id}_{tile.side.side_id}"
            verts_scaled = [vec3_scale(v, args.scale) for v in tile.vertices]
            write_smd(out_dir / f"{name}.smd", verts_scaled, tile.triangles,
                      tile.vert_normals, tile.uv_coords, args.texture)
            if not args.no_qc:
                write_qc(out_dir / f"{name}.qc", name, name)
            if args.verbose:
                print(f"  {name}.smd  ({len(tile.vertices)} verts, "
                      f"{len(tile.triangles)} tris)")
        _print_footer(out_dir, no_merge=True)
        return

    # ------------------------------------------------------------------
    # Mode B: proximity grouping + vertex welding
    # ------------------------------------------------------------------
    print(f"Grouping by proximity (threshold: {args.proximity} Source units) ...")
    groups = group_tiles(tiles, args.proximity)
    print(f"Formed {len(groups)} group(s).  Merging and welding "
          f"(weld tolerance: {args.weld} Source units) ...")

    for g_idx, group in enumerate(groups):
        name = f"terrain_group_{g_idx}"
        if args.verbose:
            ids = [(t.side.solid_id, t.side.side_id) for t in group]
            print(f"  Group {g_idx}: {len(group)} tile(s) — solid/side IDs: {ids}")

        verts, tris, norms, uvs = merge_tiles(group, args.weld, args.scale)

        if len(tris) > args.poly_limit:
            print(f"  WARNING: group {g_idx} has {len(tris)} triangles "
                  f"(poly limit: {args.poly_limit}).")
            print(f"           GoldSrc may reject this model.  "
                  f"Try lowering --proximity to split it into smaller groups.")

        write_smd(out_dir / f"{name}.smd", verts, tris, norms, uvs, args.texture)
        if not args.no_qc:
            write_qc(out_dir / f"{name}.qc", name, name)

        tile_count = len(group)
        print(f"  {name}.smd  "
              f"({tile_count} tile{'s' if tile_count != 1 else ''} merged, "
              f"{len(verts)} verts, {len(tris)} tris)")
        if args.verbose:
            for t in group:
                size = (1 << t.side.dispinfo.power) + 1
                print(f"    solid={t.side.solid_id} side={t.side.side_id} "
                      f"power={t.side.dispinfo.power} grid={size}x{size}")

    _print_footer(out_dir, no_merge=False)


def _print_footer(out_dir: Path, no_merge: bool) -> None:
    print(f"\nDone. Output in: {out_dir}/")
    print()
    print("Compile with studiomdl (GoldSrc):")
    if no_merge:
        print("  studiomdl.exe disp_models/disp_<solidID>_<sideID>.qc")
    else:
        print("  for %f in (disp_models\\*.qc) do studiomdl.exe %f   [Windows]")
        print("  for f in disp_models/*.qc; do studiomdl $f; done      [Linux]")
    print()
    print("Tuning flags:")
    print("  --scale 0.1          models too small? scale up")
    print("  --proximity 128      merge cave walls further apart")
    print("  --proximity 1        keep terrain patches separate")
    print("  --weld 0.5           tighter seam stitching")
    print("  --weld 4.0           looser (float drift in some maps)")
    print("  --no-merge           one file per tile, no welding")


if __name__ == '__main__':
    main()