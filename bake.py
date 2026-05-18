"""
bake.py — UV atlas generation and texture baking for displacement groups.

For each group of displacement tiles:
  1. Assigns every tile a non-overlapping region in a square UV atlas.
  2. For each pixel in that region, bilinearly interpolates the world-space
     position from the displaced vertex grid.
  3. Samples $basetexture and $basetexture2 at (world_x * seamless_scale,
     world_y * seamless_scale), blends them using the per-vertex alpha grid.
  4. Does the same for $bumpmap / $bumpmap2.
  5. Returns OBJ (with atlas UVs) + baked diffuse PNG + baked normal PNG.

Requires: Pillow, numpy
"""

import io
import math
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from PIL import Image
    BAKE_AVAILABLE = True
except ImportError:
    BAKE_AVAILABLE = False

from vmf_disp_to_obj import DispSide, Mesh
from vmt_parser import leaf_no_ext, seamless_scale, is_blend


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(data: bytes) -> 'Image.Image':
    return Image.open(io.BytesIO(data)).convert('RGB')


def _arr(img: 'Image.Image') -> 'np.ndarray':
    return np.array(img, dtype=np.float32)


def _sample(arr: 'np.ndarray', u: 'np.ndarray', v: 'np.ndarray') -> 'np.ndarray':
    """Vectorised nearest-neighbour sample with repeat wrapping. Returns float32 (H,W,3)."""
    h, w = arr.shape[:2]
    px = np.floor(np.abs(u) * w).astype(np.int32) % w
    py = np.floor(np.abs(v) * h).astype(np.int32) % h
    return arr[py, px]  # shape (..., 3)


# ---------------------------------------------------------------------------
# Main bake entry point
# ---------------------------------------------------------------------------

def bake_groups(
    groups: List[Tuple[List[DispSide], List[Mesh]]],
    vmt_by_leaf: Dict[str, dict],   # leaf_no_ext(mat) → parsed VMT dict
    tex_by_leaf: Dict[str, bytes],  # leaf_no_ext(filename) → raw bytes
    resolution: int = 2048,
) -> Tuple[List[Dict], str]:  # (results, report_text)
    """
    Returns one dict per group:
        name         str
        obj          str   — OBJ file content (atlas UVs, Blender Y-up)
        mtl          str   — MTL file content
        diffuse      bytes — PNG
        normal       bytes or None — PNG
    """
    if not BAKE_AVAILABLE:
        raise RuntimeError('Pillow and numpy are required for texture baking. '
                           'Run: pip install Pillow numpy')

    report_lines = [
        '=== Displacer Bake Report ===',
        f'VMTs loaded:    {list(vmt_by_leaf.keys()) or "NONE — upload the .vmt files!"}',
        f'Textures loaded:{list(tex_by_leaf.keys()) or "NONE"}',
        '',
    ]

    results = []

    for gi, (tiles, meshes) in enumerate(groups):
        name = f'terrain_group_{gi}'

        # --- resolve material & VMT ---
        from collections import Counter
        mat_leaf = leaf_no_ext(Counter(ds.material for ds in tiles).most_common(1)[0][0]) if tiles else ''
        vmt = vmt_by_leaf.get(mat_leaf, {})
        sc    = seamless_scale(vmt)
        blend = is_blend(vmt) and sc > 0

        report_lines.append(f'Group {gi} — {name} ({len(tiles)} tiles):')
        report_lines.append(f'  VMT found : {bool(vmt)} — shader={vmt.get("__shader__","?")} scale={sc} blend={blend}')
        if vmt:
            report_lines.append(f'  $basetexture  = {vmt.get("$basetexture","MISSING")}')
            report_lines.append(f'  $basetexture2 = {vmt.get("$basetexture2","MISSING")}')
            report_lines.append(f'  $bumpmap      = {vmt.get("$bumpmap","MISSING")}')
            report_lines.append(f'  $bumpmap2     = {vmt.get("$bumpmap2","MISSING")}')

        # --- load texture arrays ---
        def get_arr(key: str, warn_tag: str) -> Optional['np.ndarray']:
            path = vmt.get(key, '')
            lf   = leaf_no_ext(path)
            if not lf:
                report_lines.append(f'  {warn_tag}: key missing from VMT')
                return None
            for attempt in [lf, lf + '.bmp', lf + '.png', lf + '.jpg', lf + '.jpeg']:
                if attempt in tex_by_leaf:
                    report_lines.append(f'  {warn_tag}: FOUND → {attempt}')
                    return _arr(load_image(tex_by_leaf[attempt]))
            report_lines.append(f'  {warn_tag}: NOT FOUND (looked for "{lf}" in uploaded textures)')
            return None

        t1 = get_arr('$basetexture',  't1 (base)')
        t2 = get_arr('$basetexture2', 't2 (blend)') if blend else None
        n1 = get_arr('$bumpmap',      'n1 (normal1)')
        n2 = get_arr('$bumpmap2',     'n2 (normal2)') if blend else None
        report_lines.append('')

        has_normal = (n1 is not None or n2 is not None)

        # --- atlas layout ---
        N    = len(tiles)
        cols = max(1, math.ceil(math.sqrt(N)))
        rows = max(1, math.ceil(N / cols))

        diffuse_out = np.full((resolution, resolution, 3), 64,          dtype=np.float32)
        normal_out  = np.full((resolution, resolution, 3), [128,128,255], dtype=np.float32) \
                      if has_normal else None

        tile_uvs_list: List[List[Tuple[float, float]]] = []

        for ti, (ds, mesh) in enumerate(zip(tiles, meshes)):
            size = (1 << ds.dispinfo.power) + 1
            ag   = ds.dispinfo.alphas   # [row][col] 0-255, may be empty

            tc, tr = ti % cols, ti // cols
            pad = 1
            px0 = int(tc * resolution / cols) + pad
            px1 = int((tc + 1) * resolution / cols) - pad
            py0 = int(tr * resolution / rows) + pad
            py1 = int((tr + 1) * resolution / rows) - pad
            px1 = max(px0 + 1, px1)
            py1 = max(py0 + 1, py1)
            pw, ph = px1 - px0, py1 - py0

            # Build vertex world-position grids (Source XY for seamless UV)
            gx = np.zeros((size, size), dtype=np.float32)
            gy = np.zeros((size, size), dtype=np.float32)
            ga = np.zeros((size, size), dtype=np.float32)
            for r in range(size):
                for c in range(size):
                    vx, vy, _ = mesh.verts[r * size + c]
                    gx[r, c] = vx
                    gy[r, c] = vy
                    if ag and r < len(ag) and c < len(ag[r]):
                        ga[r, c] = ag[r][c]

            # Pixel-space coordinate grids → continuous grid coords
            lu = np.linspace(0, 1, pw, dtype=np.float32)
            lv = np.linspace(0, 1, ph, dtype=np.float32)
            lu_g, lv_g = np.meshgrid(lu, lv)          # (ph, pw)
            gc_g = lu_g * (size - 1)
            gr_g = lv_g * (size - 1)

            c0 = np.clip(gc_g.astype(np.int32), 0, size - 2)
            r0 = np.clip(gr_g.astype(np.int32), 0, size - 2)
            c1 = np.clip(c0 + 1, 0, size - 1)
            r1 = np.clip(r0 + 1, 0, size - 1)
            fc = gc_g - c0
            fr = gr_g - r0

            def bilerp(g):
                return (g[r0, c0] * (1-fc) * (1-fr) +
                        g[r0, c1] *    fc  * (1-fr) +
                        g[r1, c0] * (1-fc) *    fr  +
                        g[r1, c1] *    fc  *    fr)

            wx = bilerp(gx)   # (ph, pw)  Source world X
            wy = bilerp(gy)   # (ph, pw)  Source world Y

            # Seamless UV from world position
            if sc > 0:
                uw = (wx * sc) % 1.0;  uw[uw < 0] += 1.0
                vw = (wy * sc) % 1.0;  vw[vw < 0] += 1.0
            else:
                uw, vw = lu_g, lv_g

            # Sample base texture
            pix = _sample(t1, uw, vw) if t1 is not None \
                  else np.full((ph, pw, 3), 128, dtype=np.float32)

            # Blend with second texture using alpha grid
            if blend and t2 is not None:
                alpha = np.clip(bilerp(ga) / 255.0, 0, 1)[:, :, np.newaxis]
                pix2  = _sample(t2, uw, vw)
                pix   = pix * (1 - alpha) + pix2 * alpha

            diffuse_out[py0:py1, px0:px1] = np.clip(pix, 0, 255)

            # Normal map
            if normal_out is not None:
                npix = _sample(n1, uw, vw) if n1 is not None \
                       else np.full((ph, pw, 3), [128, 128, 255], dtype=np.float32)
                if blend and n2 is not None:
                    npix2 = _sample(n2, uw, vw)
                    npix  = npix * (1 - alpha) + npix2 * alpha
                normal_out[py0:py1, px0:px1] = np.clip(npix, 0, 255)

            # Atlas UV per vertex — V flipped for OBJ convention
            uvs_for_tile = []
            for r in range(size):
                for c in range(size):
                    u_a = (px0 + (c / (size-1)) * pw) / resolution
                    v_a = 1.0 - (py0 + (r / (size-1)) * ph) / resolution
                    uvs_for_tile.append((u_a, v_a))
            tile_uvs_list.append(uvs_for_tile)

        # --- encode images ---
        def to_png(arr):
            buf = io.BytesIO()
            Image.fromarray(arr.astype(np.uint8), 'RGB').save(buf, format='PNG')
            return buf.getvalue()

        diff_png = to_png(diffuse_out)
        norm_png = to_png(normal_out) if normal_out is not None else None

        obj_str, mtl_str = _build_obj(tiles, meshes, tile_uvs_list, name)

        results.append({
            'name':    name,
            'obj':     obj_str,
            'mtl':     mtl_str,
            'diffuse': diff_png,
            'normal':  norm_png,
        })

    return results, '\n'.join(report_lines) + '\n'


# ---------------------------------------------------------------------------
# OBJ builder with atlas UVs
# ---------------------------------------------------------------------------

def _build_obj(
    tiles:         List[DispSide],
    meshes:        List[Mesh],
    tile_uvs_list: List[List[Tuple[float, float]]],
    group_name:    str,
) -> Tuple[str, str]:
    mat_name = f'{group_name}_baked'
    tex_name = f'{group_name}_diffuse.png'

    lines = [
        f'# Displacer baked export — {group_name}',
        f'mtllib {group_name}.mtl',
        f'o {group_name}',
        '',
    ]

    v_off = 0
    all_verts, all_uvs, all_norms, all_tris = [], [], [], []

    for mesh, tile_uvs in zip(meshes, tile_uvs_list):
        for x, y, z in mesh.verts:
            all_verts.append((x, z, -y))        # Y/Z swap for Blender
        for uv in tile_uvs:
            all_uvs.append(uv)
        for nx, ny, nz in mesh.normals:
            all_norms.append((nx, nz, -ny))
        for a, b, c in mesh.tris:
            all_tris.append((a + v_off, b + v_off, c + v_off))
        v_off += len(mesh.verts)

    for x, y, z in all_verts:
        lines.append(f'v {x:.4f} {y:.4f} {z:.4f}')
    lines.append('')
    for u, v in all_uvs:
        lines.append(f'vt {u:.6f} {v:.6f}')
    lines.append('')
    for nx, ny, nz in all_norms:
        lines.append(f'vn {nx:.6f} {ny:.6f} {nz:.6f}')
    lines.append('')
    lines.append(f'usemtl {mat_name}')
    for a, b, c in all_tris:
        a1, b1, c1 = a+1, b+1, c+1
        lines.append(f'f {a1}/{a1}/{a1} {b1}/{b1}/{b1} {c1}/{c1}/{c1}')

    obj_str = '\n'.join(lines) + '\n'
    mtl_str = '\n'.join([
        f'newmtl {mat_name}',
        'Ka 1.000 1.000 1.000',
        'Kd 1.000 1.000 1.000',
        'Ks 0.000 0.000 0.000',
        f'map_Kd {tex_name}',
    ]) + '\n'

    return obj_str, mtl_str
