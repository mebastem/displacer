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
    """Vectorised bilinear sample with repeat wrapping. Returns float32 (H,W,3)."""
    h, w = arr.shape[:2]
    # Wrap to [0,1)
    u = u % 1.0; u[u < 0] += 1.0
    v = v % 1.0; v[v < 0] += 1.0
    # Continuous pixel coords
    fx = u * w - 0.5;  fx[fx < 0] += w
    fy = v * h - 0.5;  fy[fy < 0] += h
    x0 = np.floor(fx).astype(np.int32) % w
    y0 = np.floor(fy).astype(np.int32) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    tx = (fx - np.floor(fx))[..., np.newaxis]
    ty = (fy - np.floor(fy))[..., np.newaxis]
    return (arr[y0, x0] * (1 - tx) * (1 - ty) +
            arr[y0, x1] *      tx  * (1 - ty) +
            arr[y1, x0] * (1 - tx) *      ty  +
            arr[y1, x1] *      tx  *      ty)


# ---------------------------------------------------------------------------
# Main bake entry point
# ---------------------------------------------------------------------------

def bake_groups(
    groups: List[Tuple[List[DispSide], List[Mesh]]],
    vmt_by_leaf: Dict[str, dict],
    tex_by_leaf: Dict[str, bytes],
    resolution: int = 2048,
    progress_cb=None,   # callable(msg: str) for live progress updates
) -> Tuple[List[Dict], str]:
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

    def _prog(msg):
        print(f'[bake] {msg}', flush=True)
        if progress_cb:
            progress_cb(msg)

    # Split large groups so each sub-group gets enough atlas pixels per tile
    MAX_TILES = 64   # 8×8 at 2048px → 256px per tile
    flat: List[Tuple[List, List]] = []
    for tiles, meshes in groups:
        for i in range(0, max(1, len(tiles)), MAX_TILES):
            flat.append((tiles[i:i+MAX_TILES], meshes[i:i+MAX_TILES]))

    _prog(f'Starting bake: {len(flat)} sub-group(s) (from {len(groups)} proximity group(s))')
    results = []

    for gi, (tiles, meshes) in enumerate(flat):
        name = f'terrain_group_{gi}'
        _prog(f'Sub-group {gi+1}/{len(flat)}: resolving material ({len(tiles)} tiles)…')

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

        # Each tile gets at least 128px but atlas is capped at 4096
        tile_px  = max(128, min(512, resolution // max(cols, rows)))
        auto_res = min(4096, max(resolution, cols * tile_px, rows * tile_px))
        # Round up to next power of two
        auto_res = 1 << (auto_res - 1).bit_length()
        res = auto_res
        report_lines.append(f'  Atlas resolution: {res}px ({cols}×{rows} tile grid, ~{res//cols}px/tile)')

        diffuse_out = np.full((res, res, 3), 64,          dtype=np.float32)
        normal_out  = np.full((res, res, 3), [128,128,255], dtype=np.float32) \
                      if has_normal else None

        _prog(f'Sub-group {gi+1}/{len(flat)}: atlas {res}px, sampling {len(tiles)} tiles…')
        tile_uvs_list: List[List[Tuple[float, float]]] = []

        for ti, (ds, mesh) in enumerate(zip(tiles, meshes)):
            if ti % 50 == 0:
                _prog(f'Sub-group {gi+1}/{len(flat)}: tile {ti+1}/{len(tiles)}…')
            size = (1 << ds.dispinfo.power) + 1
            ag   = ds.dispinfo.alphas   # [row][col] 0-255, may be empty

            tc, tr = ti % cols, ti // cols
            pad = 1
            px0 = int(tc * res / cols) + pad
            px1 = int((tc + 1) * res / cols) - pad
            py0 = int(tr * res / rows) + pad
            py1 = int((tr + 1) * res / rows) - pad
            px1 = max(px0 + 1, px1)
            py1 = max(py0 + 1, py1)
            pw, ph = px1 - px0, py1 - py0

            # Build vertex world-position and normal grids
            gx  = np.zeros((size, size), dtype=np.float32)
            gy  = np.zeros((size, size), dtype=np.float32)
            gz  = np.zeros((size, size), dtype=np.float32)
            gnx = np.zeros((size, size), dtype=np.float32)
            gny = np.zeros((size, size), dtype=np.float32)
            gnz = np.zeros((size, size), dtype=np.float32)
            ga  = np.zeros((size, size), dtype=np.float32)
            for r in range(size):
                for c in range(size):
                    idx = r * size + c
                    vx, vy, vz = mesh.verts[idx]
                    nx, ny, nz = mesh.normals[idx]
                    gx[r, c] = vx;  gy[r, c] = vy;  gz[r, c] = vz
                    gnx[r, c] = nx; gny[r, c] = ny; gnz[r, c] = nz
                    if ag and r < len(ag) and c < len(ag[r]):
                        ga[r, c] = ag[r][c]

            # Pixel-space → continuous grid coords
            lu = np.linspace(0, 1, pw, dtype=np.float32)
            lv = np.linspace(0, 1, ph, dtype=np.float32)
            lu_g, lv_g = np.meshgrid(lu, lv)   # (ph, pw)
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

            wx  = bilerp(gx)   # (ph, pw)  Source world X
            wy  = bilerp(gy)   # (ph, pw)  Source world Y
            wz  = bilerp(gz)   # (ph, pw)  Source world Z
            wnx = bilerp(gnx)  # surface normal X
            wny = bilerp(gny)
            wnz = bilerp(gnz)

            def _triplanar(arr, scale):
                """Sample arr with triplanar projection, matching the viewer's GLSL shader."""
                if arr is None:
                    return np.full((ph, pw, 3), 128, dtype=np.float32)
                if scale <= 0:
                    return _sample(arr, lu_g, lv_g)
                # Three projections: XY (top), XZ (front), YZ (side)
                # Source → Blender axis swap used in viewer: (x, z, -y)
                # Viewer GLSL: XY plane = worldPos.xz * scale (Source X, -Y=Blender Z)
                #              XZ plane = worldPos.xy * scale
                #              YZ plane = worldPos.zy * scale
                s_xy = _sample(arr, wx * scale, -wy * scale)   # top face
                s_xz = _sample(arr, wx * scale,  wz * scale)   # front face
                s_yz = _sample(arr, wy * scale,  wz * scale)   # side face
                # Blend weights: pow(abs(normal), 6), normalised
                ax = np.abs(wnx) ** 6
                ay = np.abs(wny) ** 6
                az = np.abs(wnz) ** 6
                total = ax + ay + az + 1e-8
                wx_ = (ax / total)[..., np.newaxis]
                wy_ = (ay / total)[..., np.newaxis]
                wz_ = (az / total)[..., np.newaxis]
                # Map viewer axes: wnz→top (XY), wny→front (XZ), wnx→side (YZ)
                return s_xy * wz_ + s_xz * wy_ + s_yz * wx_

            # Sample base texture with triplanar
            pix = _triplanar(t1, sc)

            # Blend with second texture using alpha grid
            if blend and t2 is not None:
                alpha = np.clip(bilerp(ga) / 255.0, 0, 1)[:, :, np.newaxis]
                pix2  = _triplanar(t2, sc)
                pix   = pix * (1 - alpha) + pix2 * alpha

            diffuse_out[py0:py1, px0:px1] = np.clip(pix, 0, 255)

            # Normal map
            if normal_out is not None:
                npix = _triplanar(n1, sc) if n1 is not None \
                       else np.full((ph, pw, 3), [128, 128, 255], dtype=np.float32)
                if blend and n2 is not None:
                    npix2 = _triplanar(n2, sc)
                    npix  = npix * (1 - alpha) + npix2 * alpha
                normal_out[py0:py1, px0:px1] = np.clip(npix, 0, 255)

            # Atlas UV per vertex — V flipped for OBJ convention
            uvs_for_tile = []
            for r in range(size):
                for c in range(size):
                    u_a = (px0 + (c / (size-1)) * pw) / res
                    v_a = 1.0 - (py0 + (r / (size-1)) * ph) / res
                    uvs_for_tile.append((u_a, v_a))
            tile_uvs_list.append(uvs_for_tile)

        # --- encode images ---
        _prog(f'Sub-group {gi+1}/{len(flat)}: encoding PNG…')
        def to_png(arr):
            buf = io.BytesIO()
            Image.fromarray(arr.astype(np.uint8), 'RGB').save(buf, format='PNG')
            return buf.getvalue()

        diff_png = to_png(diffuse_out)
        norm_png = to_png(normal_out) if normal_out is not None else None

        _prog(f'Sub-group {gi+1}/{len(flat)}: building OBJ…')
        obj_str, mtl_str = _build_obj(tiles, meshes, tile_uvs_list, name)

        _prog(f'Sub-group {gi+1}/{len(flat)}: done ✓')
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
