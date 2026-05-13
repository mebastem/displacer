#!/usr/bin/env python3
"""
bsp_disp_to_obj.py
------------------
Reads displacement surfaces directly from a Source Engine BSP file and
exports them as Wavefront OBJ files.

No VMF needed. Reads LUMP_DISPINFO, LUMP_DISP_VERTS, and face/vertex lumps
directly. Vertex positions are reconstructed from the base face corners
(which the BSP stores exactly) plus the displacement offset vectors.

Usage:
    python bsp_disp_to_obj.py <map.bsp> [options]

Options:
    -o, --output DIR      Output directory (default: ./disp_obj)
    --no-merge            One OBJ per tile instead of grouping
    --proximity FLOAT     Grouping distance in Source units (default: 512.0)
    --weld FLOAT          Vertex weld distance in Source units (default: 1.0)
    -v, --verbose         Print per-tile info

Python 3.7+, no dependencies.
"""

import argparse
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

Vec3 = Tuple[float, float, float]

def vadd(a: Vec3, b: Vec3) -> Vec3:    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def vsub(a: Vec3, b: Vec3) -> Vec3:    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vscale(v: Vec3, s: float) -> Vec3: return (v[0]*s, v[1]*s, v[2]*s)
def vlen(v: Vec3) -> float:            return math.sqrt(v[0]**2+v[1]**2+v[2]**2)
def vnorm(v: Vec3) -> Vec3:
    l = vlen(v); return (v[0]/l, v[1]/l, v[2]/l) if l > 1e-9 else (0., 0., 1.)
def vlerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return vadd(vscale(a, 1-t), vscale(b, t))

# ---------------------------------------------------------------------------
# BSP lump indices (Source Engine)
# ---------------------------------------------------------------------------

LUMP_VERTICES   = 3
LUMP_EDGES      = 12
LUMP_SURFEDGES  = 13
LUMP_FACES      = 7
LUMP_ORIGFACES  = 27
LUMP_DISPINFO   = 26
LUMP_DISP_VERTS = 33

# ---------------------------------------------------------------------------
# BSP reader
# ---------------------------------------------------------------------------

class BSP:
    def __init__(self, path: Path):
        self.data = path.read_bytes()
        if self.data[:4] != b'VBSP':
            raise ValueError("Not a valid BSP file (missing VBSP magic)")
        self.version = struct.unpack_from('<i', self.data, 4)[0]
        print(f"  BSP version: {self.version}")

    def lump_bytes(self, index: int) -> bytes:
        off, length = struct.unpack_from('<ii', self.data, 8 + index * 16)
        return self.data[off: off + length]

    # -- Vertices -------------------------------------------------------------

    def read_vertices(self) -> List[Vec3]:
        raw = self.lump_bytes(LUMP_VERTICES)
        n   = len(raw) // 12
        return [struct.unpack_from('<3f', raw, i*12) for i in range(n)]

    # -- Edges & surfedges ----------------------------------------------------

    def read_edges(self) -> List[Tuple[int,int]]:
        raw = self.lump_bytes(LUMP_EDGES)
        n   = len(raw) // 4
        return [struct.unpack_from('<HH', raw, i*4) for i in range(n)]

    def read_surfedges(self) -> List[int]:
        raw = self.lump_bytes(LUMP_SURFEDGES)
        n   = len(raw) // 4
        return [struct.unpack_from('<i', raw, i*4)[0] for i in range(n)]

    # -- Faces ----------------------------------------------------------------

    def _parse_faces(self, raw: bytes) -> List[dict]:
        FSIZE = 56
        n = len(raw) // FSIZE
        result = []
        for i in range(n):
            b = i * FSIZE
            first_edge = struct.unpack_from('<i', raw, b + 8)[0]
            num_edges  = struct.unpack_from('<h', raw, b + 12)[0]
            disp_idx   = struct.unpack_from('<h', raw, b + 26)[0]
            result.append({'first_edge': first_edge,
                           'num_edges':  num_edges,
                           'dispinfo':   disp_idx})
        return result

    def read_faces(self) -> List[dict]:
        return self._parse_faces(self.lump_bytes(LUMP_FACES))

    def read_origfaces(self) -> List[dict]:
        raw = self.lump_bytes(LUMP_ORIGFACES)
        if len(raw) == 0:
            return []
        return self._parse_faces(raw)

    def face_verts(self, face: dict, verts: List[Vec3],
                   edges: List, surfedges: List[int]) -> List[Vec3]:
        result = []
        for i in range(face['num_edges']):
            se = surfedges[face['first_edge'] + i]
            vi = edges[se][0] if se >= 0 else edges[-se][1]
            result.append(verts[vi])
        return result

    # -- DispInfo -------------------------------------------------------------

    def read_dispinfos(self) -> List[dict]:
        """
        Parse LUMP_DISPINFO.
        ddispinfo_t layout (Source BSP v19-21), 176 bytes:
          offset  0: startPosition    vec3  (3 floats)
          offset 12: dispVertStart    int
          offset 16: dispTriStart     int
          offset 20: power            int
          offset 24: minTess          int
          offset 28: smoothingAngle   float
          offset 32: contents         int
          offset 36: mapFace          unsigned short
          ... (rest not needed)
        """
        raw  = self.lump_bytes(LUMP_DISPINFO)
        SIZE = 176
        n    = len(raw) // SIZE
        result = []
        for i in range(n):
            b  = i * SIZE
            sx, sy, sz    = struct.unpack_from('<3f', raw, b +  0)
            vert_start    = struct.unpack_from('<i',  raw, b + 12)[0]
            power         = struct.unpack_from('<i',  raw, b + 20)[0]
            map_face      = struct.unpack_from('<H',  raw, b + 36)[0]
            result.append({
                'index':      i,
                'start_pos':  (sx, sy, sz),
                'vert_start': vert_start,
                'power':      power,
                'map_face':   map_face,
            })
        return result

    # -- DispVerts ------------------------------------------------------------

    def read_disp_verts(self) -> List[Tuple[Vec3, float, float]]:
        """
        Parse LUMP_DISP_VERTS.
        dDispVert layout, 20 bytes:
          vec    vec3  (displacement vector = normal * distance)
          dist   float (magnitude, same info as vec but scalar)
          alpha  float (blend alpha, 0-1)
        """
        raw  = self.lump_bytes(LUMP_DISP_VERTS)
        SIZE = 20
        n    = len(raw) // SIZE
        result = []
        for i in range(n):
            b  = i * SIZE
            vx, vy, vz = struct.unpack_from('<3f', raw, b)
            dist       = struct.unpack_from('<f',  raw, b + 12)[0]
            alpha      = struct.unpack_from('<f',  raw, b + 16)[0]
            result.append(((vx, vy, vz), dist, alpha))
        return result


# ---------------------------------------------------------------------------
# Tile mesh
# ---------------------------------------------------------------------------

class TileMesh:
    def __init__(self, verts, normals, uvs, tris):
        self.verts   = verts
        self.normals = normals
        self.uvs     = uvs
        self.tris    = tris

    def bbox(self):
        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        zs = [v[2] for v in self.verts]
        return (min(xs),min(ys),min(zs)), (max(xs),max(ys),max(zs))


def build_base_grid(corners: List[Vec3], start_pos: Vec3, power: int) -> List[Vec3]:
    """
    Build the undisplaced base grid from the 4 face corner vertices.

    The BSP stores the exact face corner vertices — no reconstruction needed.
    We just need to rotate them so the corner nearest startposition is first,
    then bilinearly interpolate across the grid.

    Grid layout:
      (row=0, col=0) = corner nearest startposition
      (row=0, col=N) = next CCW corner
      (row=N, col=N) = diagonal corner
      (row=N, col=0) = remaining corner
    """
    # Take exactly 4 corners (faces with T-junctions may have more verts)
    if len(corners) < 4:
        return []
    if len(corners) > 4:
        corners = corners[:4]

    # Rotate so corner nearest startposition is first (preserving CCW winding)
    dists = [vlen(vsub(c, start_pos)) for c in corners]
    si    = dists.index(min(dists))
    corners = corners[si:] + corners[:si]
    c0, c1, c2, c3 = corners

    size = (1 << power) + 1
    grid = []
    for row in range(size):
        t      = row / (size - 1)
        left   = vlerp(c0, c3, t)
        right  = vlerp(c1, c2, t)
        for col in range(size):
            s = col / (size - 1)
            grid.append(vlerp(left, right, s))
    return grid


def build_tile(base_grid: List[Vec3], disp_verts: List,
               vert_start: int, power: int) -> TileMesh:
    """
    Build final mesh by adding displacement vectors to base grid positions.

    disp_verts[i] = (vec, dist, alpha)
    vec is the displacement offset in world space (already direction * distance).
    Final position = base_grid[i] + vec.
    """
    size  = (1 << power) + 1
    total = size * size

    verts:   List[Vec3]               = []
    normals: List[Vec3]               = []
    uvs:     List[Tuple[float,float]] = []

    for i in range(total):
        base        = base_grid[i]
        dvec, _, __ = disp_verts[vert_start + i]
        pos         = vadd(base, dvec)
        # Use the displacement vector as the surface normal direction.
        # If zero (flat base), fall back to computing from neighbours later.
        n = vnorm(dvec) if vlen(dvec) > 0.1 else (0., 0., 1.)
        row = i // size
        col = i %  size
        verts.append(pos)
        normals.append(n)
        uvs.append((col / (size-1), row / (size-1)))

    # Compute proper per-vertex normals from triangle geometry
    # (displacement vec direction is unreliable when dist ≈ 0)
    accum = [(0., 0., 0.)] * total
    tris  = []
    for row in range(size-1):
        for col in range(size-1):
            i00 = row*size + col
            i10 = (row+1)*size + col
            i01 = row*size + (col+1)
            i11 = (row+1)*size + (col+1)
            if (row+col) % 2 == 0:
                pairs = [(i00,i10,i11),(i00,i11,i01)]
            else:
                pairs = [(i00,i10,i01),(i10,i11,i01)]
            tris.extend(pairs)
            for (a, b, c) in pairs:
                ab  = vsub(verts[b], verts[a])
                ac  = vsub(verts[c], verts[a])
                # cross product (face normal, unnormalised — weighted by area)
                fn  = (ab[1]*ac[2]-ab[2]*ac[1],
                       ab[2]*ac[0]-ab[0]*ac[2],
                       ab[0]*ac[1]-ab[1]*ac[0])
                for idx in (a, b, c):
                    accum[idx] = vadd(accum[idx], fn)

    normals = [vnorm(n) for n in accum]

    return TileMesh(verts=verts, normals=normals, uvs=uvs, tris=tris)


# ---------------------------------------------------------------------------
# Load all tiles from BSP
# ---------------------------------------------------------------------------

def load_tiles(bsp: BSP, verbose: bool = False) -> List[TileMesh]:
    vertices  = bsp.read_vertices()
    edges     = bsp.read_edges()
    surfedges = bsp.read_surfedges()
    dispinfos = bsp.read_dispinfos()
    dv        = bsp.read_disp_verts()

    # m_iMapFace is a DIRECT index into LUMP_FACES — just read the face list
    # and index into it. Do NOT scan faces looking for a matching dispinfo field.
    faces = bsp.read_faces()

    tiles   = []
    skipped = 0
    for di in dispinfos:
        idx       = di['index']
        power     = di['power']
        map_face  = di['map_face']

        if map_face < 0 or map_face >= len(faces):
            skipped += 1
            if verbose:
                print(f"  disp {idx:4d}: map_face={map_face} out of range "
                      f"(total faces={len(faces)}), skipped")
            continue

        face    = faces[map_face]
        corners = bsp.face_verts(face, vertices, edges, surfedges)

        if len(corners) < 4:
            skipped += 1
            if verbose:
                print(f"  disp {idx:4d}: only {len(corners)} face verts, skipped")
            continue

        base_grid = build_base_grid(corners, di['start_pos'], power)
        if not base_grid:
            skipped += 1
            continue

        tile = build_tile(base_grid, dv, di['vert_start'], power)
        tiles.append(tile)

        if verbose:
            bb = tile.bbox()
            print(f"  disp {idx:4d}  power={power}  face={map_face}  "
                  f"verts={len(tile.verts)}  tris={len(tile.tris)}  "
                  f"Z={bb[0][2]:.0f}→{bb[1][2]:.0f}")

    if skipped:
        print(f"  ({skipped} tiles skipped)")
    return tiles


# ---------------------------------------------------------------------------
# Proximity grouping (Union-Find)
# ---------------------------------------------------------------------------

class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[b] = a


def group_tiles(tiles: List[TileMesh], proximity: float) -> List[List[TileMesh]]:
    n     = len(tiles)
    boxes = [t.bbox() for t in tiles]
    uf    = UF(n)
    for i in range(n):
        for j in range(i+1, n):
            a, b = boxes[i], boxes[j]
            d = 0.
            for k in range(3):
                g = max(0., max(a[0][k], b[0][k]) - min(a[1][k], b[1][k]))
                d += g*g
            if math.sqrt(d) <= proximity:
                uf.union(i, j)
    groups: Dict[int, List[TileMesh]] = {}
    for i, t in enumerate(tiles):
        groups.setdefault(uf.find(i), []).append(t)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Vertex welding
# ---------------------------------------------------------------------------

def _bucket(v: Vec3, cell: float):
    return (int(math.floor(v[0]/cell)),
            int(math.floor(v[1]/cell)),
            int(math.floor(v[2]/cell)))


def merge_tiles(tiles: List[TileMesh], weld: float) -> TileMesh:
    all_v:  List[Vec3]               = []
    all_n:  List[Vec3]               = []
    all_uv: List[Tuple[float,float]] = []
    all_t:  List[Tuple[int,int,int]] = []
    off = 0
    for t in tiles:
        all_v.extend(t.verts);  all_n.extend(t.normals); all_uv.extend(t.uvs)
        all_t.extend((i+off, j+off, k+off) for i,j,k in t.tris)
        off += len(t.verts)

    cell = max(weld, 0.01)
    bmap: Dict = {}
    for idx, v in enumerate(all_v):
        bmap.setdefault(_bucket(v, cell), []).append(idx)

    def nbrs(bk):
        bx, by, bz = bk
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    nb = (bx+dx, by+dy, bz+dz)
                    if nb in bmap: yield from bmap[nb]

    remap = list(range(len(all_v)))
    done:  Set[int] = set()
    for i in range(len(all_v)):
        if i in done: continue
        v = all_v[i]
        for j in nbrs(_bucket(v, cell)):
            if j <= i or j in done: continue
            if vlen(vsub(v, all_v[j])) <= weld:
                remap[j] = i; done.add(j)
        done.add(i)

    canon: Dict[int,int] = {}
    nv:   List[Vec3]               = []
    nn:   List[Vec3]               = []
    nuv:  List[Tuple[float,float]] = []
    nacc: Dict[int, List[Vec3]]    = {}
    for i in range(len(all_v)):
        c = remap[i]
        if c not in canon:
            canon[c] = len(nv)
            nv.append(all_v[c]); nn.append(all_n[c]); nuv.append(all_uv[c])
            nacc[canon[c]] = [all_n[c]]
        else:
            nacc[canon[c]].append(all_n[i])

    for ni, lst in nacc.items():
        nn[ni] = vnorm((sum(x[0] for x in lst),
                        sum(x[1] for x in lst),
                        sum(x[2] for x in lst)))

    nt: List[Tuple[int,int,int]] = []
    for i, j, k in all_t:
        a, b, c = canon[remap[i]], canon[remap[j]], canon[remap[k]]
        if a != b and b != c and a != c:
            nt.append((a, b, c))

    return TileMesh(verts=nv, normals=nn, uvs=nuv, tris=nt)


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------

def write_obj(path: Path, tile: TileMesh, name: str) -> None:
    """
    Write Wavefront OBJ.
    Source is Z-up; convert to Blender Y-up:
      Blender X =  Source X
      Blender Y =  Source Z
      Blender Z = -Source Y
    """
    with open(path, 'w') as f:
        f.write(f"# bsp_disp_to_obj — {name}\n")
        f.write(f"o {name}\n\n")
        for (x, y, z) in tile.verts:
            f.write(f"v {x:.4f} {z:.4f} {-y:.4f}\n")
        f.write("\n")
        for (u, v) in tile.uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write("\n")
        for (nx, ny, nz) in tile.normals:
            f.write(f"vn {nx:.6f} {nz:.6f} {-ny:.6f}\n")
        f.write("\n")
        f.write("usemtl displacement\n")
        for (i, j, k) in tile.tris:
            i1, j1, k1 = i+1, j+1, k+1
            f.write(f"f {i1}/{i1}/{i1} {j1}/{j1}/{j1} {k1}/{k1}/{k1}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Extract Source BSP displacement surfaces to Wavefront OBJ.')
    ap.add_argument('bsp',  help='Path to the .bsp file')
    ap.add_argument('-o', '--output',  default='disp_obj',
                    help='Output directory (default: ./disp_obj)')
    ap.add_argument('--no-merge',      action='store_true',
                    help='One OBJ per tile, no grouping')
    ap.add_argument('--proximity',     type=float, default=512.0,
                    help='Grouping distance in Source units (default: 512.0)')
    ap.add_argument('--weld',          type=float, default=1.0,
                    help='Vertex weld distance in Source units (default: 1.0)')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    bsp_path = Path(args.bsp)
    if not bsp_path.exists():
        sys.exit(f"Error: {bsp_path} not found")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {bsp_path} ...")
    bsp   = BSP(bsp_path)

    print("Loading displacement tiles ...")
    tiles = load_tiles(bsp, verbose=args.verbose)

    if not tiles:
        sys.exit("No displacement surfaces found.")
    print(f"Loaded {len(tiles)} tile(s).")

    if args.no_merge:
        print(f"Writing {len(tiles)} OBJ files ...")
        for i, tile in enumerate(tiles):
            name = f"disp_{i:04d}"
            write_obj(out / f"{name}.obj", tile, name)
            if args.verbose:
                print(f"  {name}.obj  {len(tile.verts)}v {len(tile.tris)}t")
    else:
        print(f"Grouping by proximity ({args.proximity} units) ...")
        groups = group_tiles(tiles, args.proximity)
        print(f"{len(groups)} group(s). Welding ({args.weld} units) ...")
        for gi, grp in enumerate(groups):
            name   = f"terrain_group_{gi}"
            merged = merge_tiles(grp, args.weld)
            write_obj(out / f"{name}.obj", merged, name)
            print(f"  {name}.obj  "
                  f"({len(grp)} tile(s), {len(merged.verts)}v, {len(merged.tris)}t)")

    print(f"\nDone → {out}/")
    print()
    print("Import into Blender: File → Import → Wavefront (.obj)")
    print("  Forward: -Z   Up: Y")


if __name__ == '__main__':
    main()