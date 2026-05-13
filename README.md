# Source Displacement Extractor

Extracts displacement surfaces from Source Engine map files and exports them for use in Blender or GoldSrc.

## Tools

| Script | Input | Output | Use |
|--------|-------|--------|-----|
| `vmf_disp_to_obj.py` | `.vmf` | `.obj` | Displacement surfaces → Blender |
| `bsp_disp_to_obj.py` | `.bsp` | `.obj` | Compiled BSP → Blender (authoritative) |

## Requirements

- Python 3.7+
- No third-party libraries

## Usage

### VMF → OBJ (Blender)
```
py -3 vmf_disp_to_obj.py yourmap.vmf
```

Options:
```
-o, --output DIR      Output directory (default: ./disp_obj)
--no-merge            One OBJ per tile instead of grouping by proximity
--proximity FLOAT     Grouping threshold in Source units (default: 4.0)
--weld FLOAT          Vertex weld tolerance in Source units (default: 1.0)
-v, --verbose         Print per-tile debug info
```

### BSP → OBJ (Blender)
```
py -3 bsp_disp_to_obj.py yourmap.bsp
```

## Blender Import Settings

**File > Import > Wavefront (.obj)**
- Forward axis: **-Z**
- Up axis: **Y**
