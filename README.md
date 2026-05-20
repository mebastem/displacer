# Displacer

Reads displacement surfaces out of Source Engine VMF files, shows them in a first-person 3D viewer with live texture blending, and bakes them to atlased OBJ/PNG ready for Blender or GoldSrc.

## Requirements

- Python 3.10+
- Flask — `pip install flask`
- Pillow + numpy — only needed for baking — `pip install Pillow numpy`

### With [uv](https://docs.astral.sh/uv/)

```
uv sync
```

### With pip

```
pip install flask Pillow numpy
```

## Running the viewer

```
python app.py          # or: uv run python app.py
```

Opens at `http://127.0.0.1:5000`. Drag a `.vmf` onto the page or click **Load VMF**.

## Camera

The viewer is first-person spectator — click the viewport to lock the cursor, then:

| Input | Action |
|-------|--------|
| W A S D | Move |
| Mouse | Look |
| E / Q | Move up / down |
| Scroll wheel | Adjust move speed |
| F or Home | Snap to geometry |
| Esc | Release cursor |

## Toolbar

### View

| Button | Key | What it does |
|--------|-----|-------------|
| Wire | Z | Overlay triangle grid |
| Flat | L | Full-brightness lighting (no shading) |
| Lights | K | Show light entity positions as colored spheres |
| Props | — | Show prop_static / prop_dynamic origins as pink cubes |
| Reset | F | Return camera to the saved position |

### Assets

| Button | What it does |
|--------|-------------|
| VMTs | Upload `.vmt` files to enable texture preview and blending |
| Textures | Upload `.bmp` / `.png` / `.jpg` textures to display on the mesh |
| Sky | Upload 6 skybox images — auto-detected from Source suffixes (`_ft`, `_bk`, `_lf`, `_rt`, `_up`, `_dn`) |

### Export

| Button | What it does |
|--------|-------------|
| Picker | Click a tile in the viewport to inspect its material, power, vert/tri count, and texture density |
| Bake | Upload VMTs + textures, bake an atlased OBJ/PNG package, download as zip |
| Export ⚙ | Export OBJ for current selection or all groups; configurable vert limits for GoldSrc |

Press **?** at any time to see the full keyboard shortcut reference.

## Live texture preview

1. Click **VMTs** and upload the `.vmt` files for your displacement materials
2. Click **Textures** and upload the `.bmp` / `.png` images they reference
3. The viewport shader handles:
   - **WorldVertexTransition** — blends `$basetexture` → `$basetexture2` using the per-vertex alpha grid baked into the VMF
   - **Seamless UV** — derived from `$seamless_scale` in the VMT
   - **Non-seamless UV** — normalized to texel space so the texture tiles correctly regardless of face dimensions

## Sidebar

Displacement groups are listed by material. Click a material to isolate its tiles; Shift/Ctrl+click to select multiple. The search box filters by name in real time. **Show all** clears the filter.

## Baked export

Click **Bake**, upload VMTs and source textures, pick a resolution. You get a zip with one set per group:

```
group_N.obj            mesh with atlas UVs (Blender Y-up)
group_N.mtl            material pointing at the baked diffuse
group_N_diffuse.png    baked diffuse atlas
group_N_normal.png     baked normal map (if bumpmaps were uploaded)
_bake_report.txt       which VMTs / textures were found or missing
```

## CLI — VMF → OBJ (no server)

```
python vmf_disp_to_obj.py yourmap.vmf
```

Options:

```
-o, --output DIR      Output directory (default: ./disp_obj)
--no-merge            One OBJ per tile instead of grouping by proximity
--proximity FLOAT     Grouping distance in Source units (default: 4.0)
--weld FLOAT          Vertex weld tolerance in Source units (default: 1.0)
-v, --verbose         Print per-tile debug info
```

## Blender import settings

**File → Import → Wavefront (.obj)**
- Forward axis: **-Z**
- Up axis: **Y**

## File overview

| File | Role |
|------|------|
| `app.py` | Flask server — `/process` parses VMF, `/bake` runs the atlas pipeline |
| `vmf_disp_to_obj.py` | VMF parser and displacement mesh builder |
| `vmt_parser.py` | VMT parser — extracts shader params, seamless scale, blend detection |
| `bake.py` | UV atlas baking — triplanar projection → diffuse + normal PNG |
| `templates/index.html` | Single-page viewer — Three.js, GLSL shaders, all UI |
