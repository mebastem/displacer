# Displacer

Extracts displacement surfaces from Source Engine VMF files, previews them in a 3D web viewer with live texture blending, and bakes them to atlased OBJ/PNG for use in GoldSrc / Blender.

## Tools

| File | What it does |
|------|-------------|
| `vmf_disp_to_obj.py` | Core parser — VMF → displacement meshes |
| `vmt_parser.py` | VMT material file parser (seamless scale, blend detection) |
| `bake.py` | UV atlas baking — diffuse + normal PNG from uploaded textures |
| `app.py` | Web viewer + REST API (`/process`, `/bake`) |

## Requirements

- Python 3.14+
- Flask (`pip install flask`)
- Pillow + numpy — only needed for texture baking (`pip install Pillow numpy`)

### With [uv](https://docs.astral.sh/uv/) (recommended)

```
uv sync
```

### With pip

```
pip install flask Pillow numpy
```

## Web Viewer

**With uv:**
```
uv run python app.py
```

**With Python directly:**
```
python app.py
```

Opens at `http://127.0.0.1:5000` automatically.

### Loading a map

- Drag and drop a `.vmf` file or click **Load VMF**
- Import Settings let you choose between proximity grouping or per-tile mode

### Sidebar

- Displacement groups are listed **by material** — click a material to isolate its tiles
- **Shift/Ctrl+click** for multi-material selection
- **Search box** filters materials by name in real time
- **Show all** clears the active filter

### Viewport controls

| Input | Action |
|-------|--------|
| Scroll | Zoom |
| Left-drag | Orbit |
| Right-drag | Pan |
| **F** / Reset View | Snap camera to geometry |

### Toolbar buttons

| Button | What it does |
|--------|-------------|
| Load VMF | Open a new VMF (also available via drag-and-drop) |
| Load VMT | Upload `.vmt` material files to enable texture preview |
| Load Textures | Upload `.bmp`/`.png`/`.jpg` textures to display on the mesh |
| Wireframe | Overlay the triangle grid |
| Flat Light | Disable directional lighting so textures read at full brightness |
| Export OBJ | Download current selection (or all groups) as OBJ files |
| Bake Textures | Upload VMTs + textures and bake an atlased OBJ/PNG package |

### Live texture preview

1. Click **Load VMT** and upload the `.vmt` file(s) for your displacement material
2. Click **Load Textures** and upload the referenced `.bmp`/`.png` textures
3. The viewer applies a GLSL shader with:
   - **Triplanar world-space projection** — eliminates stretching on steep faces
   - **Per-vertex alpha blending** — `WorldVertexTransition` materials blend `$basetexture` → `$basetexture2` using the VMF alpha grid
   - **Seamless UV** derived from `$seamless_scale`

### Texture baking (`/bake`)

Click **Bake Textures**, upload VMTs and source textures, choose a resolution, and download `baked.zip` containing per-group:

- `terrain_group_N.obj` — mesh with atlas UVs (Blender Y-up)
- `terrain_group_N.mtl` — material referencing the baked diffuse
- `terrain_group_N_diffuse.png` — baked diffuse atlas
- `terrain_group_N_normal.png` — baked normal map (if bumpmaps were provided)
- `_bake_report.txt` — lists which VMTs and textures were found/missing

## CLI — VMF → OBJ

**With uv:**
```
uv run python vmf_disp_to_obj.py yourmap.vmf
```

**With Python directly:**
```
python vmf_disp_to_obj.py yourmap.vmf
```

Options:

```
-o, --output DIR      Output directory (default: ./disp_obj)
--no-merge            One OBJ per tile instead of grouping by proximity
--proximity FLOAT     Grouping threshold in Source units (default: 4.0)
--weld FLOAT          Vertex weld tolerance in Source units (default: 1.0)
--stitch FLOAT        Bridge close unmatched boundary edges (default: 128.0, 0 disables)
-v, --verbose         Print per-tile debug info
```

## Blender Import Settings

**File > Import > Wavefront (.obj)**
- Forward axis: **-Z**
- Up axis: **Y**
