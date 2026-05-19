"""
app.py  —  Displacement Viewer
Flask backend: accepts a VMF upload, extracts displacements, returns geometry JSON.
"""

import io
import json
import queue
import threading
import zipfile

from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
from vmf_disp_to_obj import (
    VMFParser, extract_disp_sides, build_mesh, group_meshes, merge_meshes
)
from vmt_parser import parse_vmt, leaf_no_ext

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512 MB

_last_vmf: str = ''   # cached after /process so /bake doesn't need re-upload


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'vmf' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['vmf']

    try:
        content = f.read().decode('utf-8', errors='replace')
    except Exception as e:
        return jsonify({'error': f'Could not read file: {e}'}), 400

    global _last_vmf
    _last_vmf = content

    try:
        parser = VMFParser(content)
        blocks = parser.parse()
        sides  = extract_disp_sides(blocks)
        lights = extract_lights(blocks)
    except Exception as e:
        return jsonify({'error': f'VMF parse error: {e}'}), 400

    if not sides:
        return jsonify({'error': 'No displacement surfaces found in this VMF.'}), 400

    no_merge  = request.form.get('no_merge',  'false').lower() == 'true'
    proximity = float(request.form.get('proximity', 4.0))
    weld      = float(request.form.get('weld', 1.0))

    meshes = []
    side_ids = []  # parallel list: solid_id/side_id for each mesh
    warnings = []
    for ds in sides:
        try:
            meshes.append(build_mesh(ds))
            side_ids.append((ds.solid_id, ds.side_id))
        except Exception as e:
            warnings.append({
                'solid_id': ds.solid_id,
                'side_id': ds.side_id,
                'error': str(e),
            })

    if not meshes:
        return jsonify({
            'error': 'Displacement surfaces were found, but none could be converted.',
            'warnings': warnings,
        }), 400

    def mesh_to_arrays(mesh):
        verts = []
        for x, y, z in mesh.verts:
            verts += [x, z, -y]
        norms = []
        for nx, ny, nz in mesh.normals:
            norms += [nx, nz, -ny]
        uvs = []
        for u, v in mesh.uvs:
            uvs += [u, v]
        indices = []
        for a, b, c in mesh.tris:
            indices += [a, b, c]
        alphas = list(mesh.alphas) if mesh.alphas else [0.0] * len(mesh.verts)
        return verts, norms, uvs, indices, alphas

    out_groups = []

    mesh_to_ids = {id(m): ids for m, ids in zip(meshes, side_ids)}

    if no_merge:
        # Each tile is its own group — no merging, no welding
        groups = [[m] for m in meshes]
        weld = 0.0  # single-tile groups need no welding
    else:
        groups = group_meshes(meshes, proximity=proximity)

    for i, grp in enumerate(groups):
            tile_map = []
            tri_offset = 0
            for m in grp:
                ids = mesh_to_ids[id(m)]
                tile_map.append({
                    'solid_id':  ids[0],
                    'side_id':   ids[1],
                    'material':  m.material,
                    'tri_start': tri_offset,
                    'tri_count': len(m.tris),
                })
                tri_offset += len(m.tris)

            merged = merge_meshes(grp, weld=weld)
            grp_ids = [mesh_to_ids[id(m)] for m in grp]
            verts, norms, uvs, indices, alphas = mesh_to_arrays(merged)
            out_groups.append({
                'name':      f'group_{i}',
                'tiles':     len(grp),
                'material':  merged.material,
                'solid_ids': grp_ids,
                'tile_map':  tile_map,
                'verts':     verts,
                'normals':   norms,
                'uvs':       uvs,
                'indices':   indices,
                'alphas':      alphas,
                'tex_density': merged.tex_density,
            })

    return jsonify({
        'filename':    f.filename,
        'total_disps': len(sides),
        'converted_disps': len(meshes),
        'warnings':    warnings,
        'groups':      out_groups,
        'lights':      lights,
    })


_LIGHT_CLASSES = {'light', 'light_spot', 'light_environment', 'light_dynamic'}

def _parse_color(s, default='255 255 255 200'):
    parts = (s or default).split()
    try:
        r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
        brightness = float(parts[3]) if len(parts) >= 4 else 200.0
    except (IndexError, ValueError):
        r, g, b, brightness = 255.0, 255.0, 255.0, 200.0
    return [round(r / 255.0, 4), round(g / 255.0, 4), round(b / 255.0, 4)], brightness

def _parse_vec3f(s, default='0 0 0'):
    parts = (s or default).split()
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except (IndexError, ValueError):
        return [0.0, 0.0, 0.0]

def extract_lights(blocks):
    lights = []

    def walk(node):
        name = node.get('__name__', '')
        if name == 'entity':
            raw_cls = node.get('classname', '')
            if isinstance(raw_cls, list): raw_cls = raw_cls[0]
            cls = raw_cls.lower()
            if cls in _LIGHT_CLASSES:
                color, brightness = _parse_color(node.get('_light'))
                angles = _parse_vec3f(node.get('angles', '0 0 0'))
                origin = _parse_vec3f(node.get('origin', '0 0 0'))
                entry = {
                    'classname': cls,
                    'origin':    origin,
                    'color':     color,
                    'brightness': brightness,
                    'angles':    angles,           # [pitch, yaw, roll]
                }
                if 'pitch' in node:               # explicit pitch override
                    try: entry['pitch'] = float(node['pitch'])
                    except ValueError: pass
                if cls == 'light_spot':
                    try: entry['cone']       = float(node.get('_cone', 45))
                    except ValueError: entry['cone'] = 45.0
                    try: entry['inner_cone'] = float(node.get('_inner_cone', 30))
                    except ValueError: entry['inner_cone'] = 30.0
                if cls == 'light_environment':
                    amb_color, amb_brightness = _parse_color(node.get('_ambient'), '128 128 128 100')
                    entry['ambient_color']      = amb_color
                    entry['ambient_brightness'] = amb_brightness
                lights.append(entry)
            return

        for val in node.values():
            for item in (val if isinstance(val, list) else [val]):
                if isinstance(item, dict):
                    walk(item)

    for b in blocks:
        walk(b)
    return lights


def _prepare_bake_inputs(vmf_text, vmt_files, tex_files, resolution):
    """Parse inputs and return (groups, vmt_by_leaf, tex_by_leaf, error_str)."""
    vmt_by_leaf = {}
    for f in vmt_files:
        try:
            text = f.read().decode('utf-8', errors='replace')
            vmt_by_leaf[leaf_no_ext(f.filename)] = parse_vmt(text)
        except Exception:
            pass

    tex_by_leaf = {}
    for f in tex_files:
        tex_by_leaf[leaf_no_ext(f.filename)] = f.read()

    try:
        blocks = VMFParser(vmf_text).parse()
        sides  = extract_disp_sides(blocks)
    except Exception as e:
        return None, None, None, f'VMF parse error: {e}'

    if not sides:
        return None, None, None, 'No displacements found in cached VMF'

    meshes, valid_sides = [], []
    for ds in sides:
        try:
            meshes.append(build_mesh(ds))
            valid_sides.append(ds)
        except Exception:
            pass

    if not meshes:
        return None, None, None, 'No meshes could be built'

    side_map   = {id(m): ds for m, ds in zip(meshes, valid_sides)}
    raw_groups = group_meshes(meshes, proximity=4.0)
    groups     = [([side_map[id(m)] for m in grp], grp) for grp in raw_groups]
    return groups, vmt_by_leaf, tex_by_leaf, None


@app.route('/bake', methods=['POST'])
def bake():
    global _last_vmf
    if not _last_vmf:
        return jsonify({'error': 'No VMF loaded — open a VMF in the viewer first'}), 400

    vmt_files  = request.files.getlist('vmts')
    tex_files  = request.files.getlist('textures')
    resolution = int(request.form.get('resolution', 2048))

    groups, vmt_by_leaf, tex_by_leaf, err = _prepare_bake_inputs(
        _last_vmf, vmt_files, tex_files, resolution)
    if err:
        return jsonify({'error': err}), 400

    q = queue.Queue()

    def run():
        try:
            from bake import bake_groups
            baked, report = bake_groups(
                groups, vmt_by_leaf, tex_by_leaf,
                resolution=resolution,
                progress_cb=lambda msg: q.put(('progress', msg)),
            )
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('_bake_report.txt', report)
                for item in baked:
                    n = item['name']
                    zf.writestr(f'{n}.obj',         item['obj'])
                    zf.writestr(f'{n}.mtl',         item['mtl'])
                    zf.writestr(f'{n}_diffuse.png', item['diffuse'])
                    if item['normal']:
                        zf.writestr(f'{n}_normal.png', item['normal'])
            buf.seek(0)
            q.put(('done', buf.getvalue()))
        except Exception as e:
            import traceback
            q.put(('error', f'{e}\n{traceback.format_exc()}'))

    t = threading.Thread(target=run, daemon=True)
    t.start()

    fname = (request.form.get('filename') or 'terrain').replace('.vmf', '') + '_baked.zip'

    @stream_with_context
    def generate():
        while True:
            kind, data = q.get()
            if kind == 'progress':
                yield f"data: {json.dumps({'progress': data})}\n\n"
            elif kind == 'error':
                yield f"data: {json.dumps({'error': data})}\n\n"
                return
            elif kind == 'done':
                import base64
                yield f"data: {json.dumps({'done': True, 'filename': fname, 'zip': base64.b64encode(data).decode()})}\n\n"
                return

    return Response(generate(), mimetype='text/event-stream',
                    headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})


if __name__ == '__main__':
    import webbrowser, threading
    def _open():
        import time; time.sleep(0.8)
        webbrowser.open('http://127.0.0.1:5000')
    threading.Thread(target=_open, daemon=True).start()
    app.run(debug=False, port=5000)
