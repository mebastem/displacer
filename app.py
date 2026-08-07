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
    VMFParser, extract_disp_sides, build_mesh, group_meshes, merge_meshes,
    build_clip_brushes_map, build_clip_brushes_map_strip, estimate_clip_strip,
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
        props  = extract_props(blocks)
        fog    = extract_fog(blocks)
    except Exception as e:
        return jsonify({'error': f'VMF parse error: {e}'}), 400

    if not sides:
        return jsonify({'error': 'No displacement surfaces found in this VMF.'}), 400

    no_merge  = request.form.get('no_merge',  'false').lower() == 'true'
    proximity = float(request.form.get('proximity', 64.0))
    weld      = float(request.form.get('weld', 1.0))

    meshes = []
    side_ids = []  # parallel list: solid_id/side_id for each mesh
    side_meta = []  # parallel: extra per-tile info (power, etc.)
    warnings = []
    for ds in sides:
        try:
            meshes.append(build_mesh(ds))
            side_ids.append((ds.solid_id, ds.side_id))
            side_meta.append({'power': ds.dispinfo.power})
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

    mesh_to_ids  = {id(m): ids  for m, ids  in zip(meshes, side_ids)}
    mesh_to_meta = {id(m): meta for m, meta in zip(meshes, side_meta)}

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
                ids  = mesh_to_ids[id(m)]
                meta = mesh_to_meta[id(m)]
                tile_map.append({
                    'solid_id':   ids[0],
                    'side_id':    ids[1],
                    'material':   m.material,
                    'tri_start':  tri_offset,
                    'tri_count':  len(m.tris),
                    'vert_count': len(m.verts),
                    'power':      meta['power'],
                    'tex_density': round(m.tex_density, 4),
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
        'props':       props,
        'fog':         fog,
    })


def _parse_rgb255(s, default='200 200 200'):
    if isinstance(s, list): s = s[0] if s else default   # duplicate key → coerce
    parts = (s or default).split()
    try:
        r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        r, g, b = 200.0, 200.0, 200.0
    return [round(r / 255.0, 4), round(g / 255.0, 4), round(b / 255.0, 4)]

def extract_fog(blocks):
    fog = None

    def walk(node):
        nonlocal fog
        name = node.get('__name__', '')
        if name == 'entity':
            raw_cls = node.get('classname', '')
            if isinstance(raw_cls, list): raw_cls = raw_cls[0]
            cls = raw_cls.lower()
            if cls == 'env_fog_controller':
                color_str = node.get('fogcolor', '200 200 200')
                if isinstance(color_str, list): color_str = color_str[0]
                start = node.get('fogstart', '500')
                end   = node.get('fogend',   '2000')
                maxd  = node.get('fogmaxdensity', '1.0')
                if isinstance(start, list): start = start[0]
                if isinstance(end,   list): end   = end[0]
                if isinstance(maxd,  list): maxd  = maxd[0]
                fog = {
                    'color':      _parse_rgb255(color_str),
                    'start':      float(start or 500),
                    'end':        float(end   or 2000),
                    'maxDensity': float(maxd  or 1.0),
                }
            return
        if name == 'world':
            if node.get('fog_enable', '0') == '1':
                color_str = node.get('fog_color', '200 200 200')
                if isinstance(color_str, list): color_str = color_str[0]
                start = node.get('fog_start', '500')
                end   = node.get('fog_end',   '2000')
                maxd  = node.get('fog_maxdensity', '1.0')
                if isinstance(start, list): start = start[0]
                if isinstance(end,   list): end   = end[0]
                if isinstance(maxd,  list): maxd  = maxd[0]
                fog = {
                    'color':      _parse_rgb255(color_str),
                    'start':      float(start or 500),
                    'end':        float(end   or 2000),
                    'maxDensity': float(maxd  or 1.0),
                }
            return
        for val in node.values():
            for item in (val if isinstance(val, list) else [val]):
                if isinstance(item, dict):
                    walk(item)

    for b in blocks:
        walk(b)
    return fog

_LIGHT_CLASSES = {'light', 'light_spot', 'light_environment', 'light_dynamic'}

_PROP_CLASSES = {
    'prop_static', 'prop_dynamic', 'prop_dynamic_override',
    'prop_physics', 'prop_physics_override', 'prop_ragdoll', 'prop_detail',
}

def extract_props(blocks):
    props = []

    def walk(node):
        name = node.get('__name__', '')
        if name == 'entity':
            raw_cls = node.get('classname', '')
            if isinstance(raw_cls, list): raw_cls = raw_cls[0]
            cls = raw_cls.lower()
            if cls in _PROP_CLASSES:
                origin = _parse_vec3f(node.get('origin', '0 0 0'))
                model  = node.get('model', '')
                if isinstance(model, list): model = model[0]
                props.append({'classname': cls, 'origin': origin, 'model': model})
            return

        for val in node.values():
            for item in (val if isinstance(val, list) else [val]):
                if isinstance(item, dict):
                    walk(item)

    for b in blocks:
        walk(b)
    return props

def _parse_color(s, default='255 255 255 200'):
    if isinstance(s, list): s = s[0] if s else default   # duplicate key → coerce
    parts = (s or default).split()
    try:
        r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
        brightness = float(parts[3]) if len(parts) >= 4 else 200.0
    except (IndexError, ValueError):
        r, g, b, brightness = 255.0, 255.0, 255.0, 200.0
    return [round(r / 255.0, 4), round(g / 255.0, 4), round(b / 255.0, 4)], brightness

def _parse_vec3f(s, default='0 0 0'):
    if isinstance(s, list): s = s[0] if s else default   # duplicate key → coerce
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


@app.route('/export_clip', methods=['POST'])
def export_clip():
    global _last_vmf
    if not _last_vmf:
        return jsonify({'error': 'No VMF loaded — open a VMF in the viewer first'}), 400

    depth = float(request.form.get('depth', 64.0))
    tex   = request.form.get('tex', 'CLIP').strip() or 'CLIP'

    try:
        blocks = VMFParser(_last_vmf).parse()
        sides  = extract_disp_sides(blocks)
    except Exception as e:
        return jsonify({'error': f'VMF parse error: {e}'}), 400

    if not sides:
        return jsonify({'error': 'No displacements found in VMF'}), 400

    pairs = []
    for ds in sides:
        try:
            pairs.append((ds, build_mesh(ds)))
        except Exception:
            pass

    if not pairs:
        return jsonify({'error': 'No meshes could be built'}), 400

    map_text = build_clip_brushes_map(pairs, depth=depth, tex=tex)

    fname = (request.form.get('filename') or 'terrain').replace('.vmf', '') + '_clip.map'
    buf = io.BytesIO(map_text.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=fname,
                     mimetype='text/plain')


_MAX_CLIPNODES = 32767
_clip_cache = {'vmf': None, 'pairs': None}


def _clip_pairs():
    """(DispSide, Mesh) pairs for the cached VMF, memoised so repeated estimate
    calls during a slider drag don't re-parse + re-mesh the whole map each time."""
    global _clip_cache
    if _clip_cache['vmf'] != _last_vmf:
        blocks = VMFParser(_last_vmf).parse()
        sides  = extract_disp_sides(blocks)
        pairs  = []
        for ds in sides:
            try:
                pairs.append((ds, build_mesh(ds)))
            except Exception:
                pass
        _clip_cache = {'vmf': _last_vmf, 'pairs': pairs}
    return _clip_cache['pairs']


@app.route('/export_clip_opt', methods=['POST'])
def export_clip_opt():
    global _last_vmf
    if not _last_vmf:
        return jsonify({'error': 'No VMF loaded — open a VMF in the viewer first'}), 400

    depth = float(request.form.get('depth', 64.0))
    tex   = request.form.get('tex', 'CLIP').strip() or 'CLIP'
    tol   = float(request.form.get('tol', 4.0))
    estimate_only = request.form.get('estimate_only', 'false').lower() == 'true'

    try:
        pairs = _clip_pairs()
    except Exception as e:
        return jsonify({'error': f'VMF parse error: {e}'}), 400

    if not pairs:
        return jsonify({'error': 'No displacements found in VMF'}), 400

    # Optional scope filter: only clip the given (solid_id, side_id) tiles
    # (e.g. the currently-visible material in the viewer).
    sides_json = request.form.get('sides', '').strip()
    if sides_json:
        try:
            wanted = set((int(a), int(b)) for a, b in json.loads(sides_json))
            pairs = [(ds, m) for ds, m in pairs if (ds.solid_id, ds.side_id) in wanted]
        except Exception:
            pass
        if not pairs:
            return jsonify({'error': 'No displacements in the selected scope'}), 400

    if estimate_only:
        st = estimate_clip_strip(pairs, tol=tol)
        cn = st['faces'] * 3
        return jsonify({
            'brushes':   st['brushes'],
            'faces':     st['faces'],
            'clipnodes': cn,
            'limit':     _MAX_CLIPNODES,
            'headroom':  _MAX_CLIPNODES - cn,
            'fits':      cn <= _MAX_CLIPNODES,
        })

    map_text, _st = build_clip_brushes_map_strip(pairs, tol=tol, depth=depth, tex=tex)

    fname = (request.form.get('filename') or 'terrain').replace('.vmf', '') + '_clip_strip.map'
    buf = io.BytesIO(map_text.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=fname,
                     mimetype='text/plain')


if __name__ == '__main__':
    import webbrowser, threading, os
    # Only open browser in the main process, not the reloader child
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        def _open():
            import time; time.sleep(1.2)
            webbrowser.open('http://127.0.0.1:5000')
        threading.Thread(target=_open, daemon=True).start()
    app.run(debug=True, use_reloader=True, port=5000)
