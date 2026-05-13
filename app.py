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
    for ds in sides:
        try:
            meshes.append(build_mesh(ds))
        except Exception:
            pass

    groups = group_meshes(meshes, proximity=4.0)

    out_groups = []
    for i, grp in enumerate(groups):
        merged = merge_meshes(grp, weld=1.0)

        # Flatten into typed arrays for Three.js BufferGeometry.
        # Convert Source (Z-up) → Three.js (Y-up): (x, y, z) → (x, z, -y)
        verts = []
        for x, y, z in merged.verts:
            verts += [x, z, -y]

        norms = []
        for nx, ny, nz in merged.normals:
            norms += [nx, nz, -ny]

        uvs = []
        for u, v in merged.uvs:
            uvs += [u, v]

        indices = []
        for a, b, c in merged.tris:
            indices += [a, b, c]

        out_groups.append({
            'name':     f'group_{i}',
            'tiles':    len(grp),
            'material': merged.material,
            'verts':    verts,
            'normals':  norms,
            'uvs':      uvs,
            'indices':  indices,
        })

    return jsonify({
        'filename':    f.filename,
        'total_disps': len(sides),
        'groups':      out_groups,
    })


if __name__ == '__main__':
    import webbrowser, threading
    def _open():
        import time; time.sleep(0.8)
        webbrowser.open('http://127.0.0.1:5000')
    threading.Thread(target=_open, daemon=True).start()
    app.run(debug=False, port=5000)
