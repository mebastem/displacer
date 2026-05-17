"""
app.py  —  Displacement Viewer
Flask backend: accepts a VMF upload, extracts displacements, returns geometry JSON.
"""

from flask import Flask, request, jsonify, render_template
from vmf_disp_to_obj import (
    VMFParser, extract_disp_sides, build_mesh, group_meshes, merge_meshes
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256 MB


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

    try:
        parser = VMFParser(content)
        blocks = parser.parse()
        sides  = extract_disp_sides(blocks)
    except Exception as e:
        return jsonify({'error': f'VMF parse error: {e}'}), 400

    if not sides:
        return jsonify({'error': 'No displacement surfaces found in this VMF.'}), 400

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
        return verts, norms, uvs, indices

    out_groups = []

    # group_meshes returns groups of meshes; track which side IDs are in each group
    mesh_to_ids = {id(m): ids for m, ids in zip(meshes, side_ids)}
    groups = group_meshes(meshes, proximity=4.0)

    for i, grp in enumerate(groups):
            # Build tile_map before merging so we know each tile's triangle range
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

            merged = merge_meshes(grp, weld=1.0)
            grp_ids = [mesh_to_ids[id(m)] for m in grp]
            verts, norms, uvs, indices = mesh_to_arrays(merged)
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
            })

    return jsonify({
        'filename':    f.filename,
        'total_disps': len(sides),
        'converted_disps': len(meshes),
        'warnings':    warnings,
        'groups':      out_groups,
    })


if __name__ == '__main__':
    import webbrowser, threading
    def _open():
        import time; time.sleep(0.8)
        webbrowser.open('http://127.0.0.1:5000')
    threading.Thread(target=_open, daemon=True).start()
    app.run(debug=False, port=5000)
