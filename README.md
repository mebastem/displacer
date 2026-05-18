# Displacer

Extracts displacement surfaces from Source Engine VMF files and exports them for use in GoldSrc / Blender.

## Tools

| Script | What it does |
|--------|-------------|
| `vmf_disp_to_obj.py` | CLI — converts VMF displacements to Wavefront OBJ |
| `app.py` | Web viewer — drag-and-drop VMF, inspect 3D in browser |

## Requirements

- Python 3.14+
- Flask (`pip install flask`)

### With [uv](https://docs.astral.sh/uv/) (recommended)

```
uv sync
```

### With pip

```
pip install flask
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

- Drag and drop a `.vmf` file or click **Load VMF**
- Groups of displacement tiles are color-coded in the sidebar
- Click a group to isolate it, **Show all** to bring everything back
- **Wireframe** toggle overlays the mesh grid
- **Export OBJ** downloads the current selection (or all groups) as OBJ files
- **Reset View** / press **F** to snap the camera back to the loaded geometry

Controls: scroll to zoom, left-drag to orbit, right-drag to pan.

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
