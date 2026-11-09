"""Side server for segmenters that cannot share the PyTorch server's environment.

Run one per environment:
    <env python> side_server.py <group> --port 8011 --token-file ../server-pytorch/secret.txt
Groups: microsam (micro-SAM vit_b_lm, box + point), cellpose3 (cyto3, nuclei), stardist (2D_versatile_fluo).

The main server forwards a method here when LABKIT_SIDE_BACKENDS lists its group, e.g.
    LABKIT_SIDE_BACKENDS="microsam=http://127.0.0.1:8011,cellpose3=http://127.0.0.1:8012,stardist=http://127.0.0.1:8013"

API (header X-API-Token):
    GET  /methods                      -> {"methods": [...], "prompt": {...}}
    POST /segment/<method>?box=x0,y0,x1,y1&point=x,y   body: TIFF image -> body: uint16 TIFF label image
The box is inclusive (the server's convention); the point is optional.

Model code and pre-processing match labkit-evaluation/model-pilot-2026-11-07/scripts/run_models.py,
which produced the study candidates.
"""
import argparse, io, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import tifffile


def pnorm(img, lo=1, hi=99.8):
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img-a)/max(b-a, 1e-6), 0, 1)


def rgb8(img):
    return np.repeat((pnorm(img)*255).astype(np.uint8)[..., None], 3, axis=2)


def build(group):
    """Return {method_id: (prompt_kind, lazy builder -> fun(img, box_inclusive, point))}."""
    if group == 'microsam':
        def microsam():
            from micro_sam import util
            p = util.get_sam_model(model_type='vit_b_lm')
            def f(img, box, pt):
                p.set_image(rgb8(img))
                x0, y0, x1, y1 = box
                masks, _, _ = p.predict(point_coords=None if pt is None else np.array([pt], np.float32),
                                        point_labels=None if pt is None else np.array([1]),
                                        box=np.array([x0, y0, x1+1, y1+1], np.float32), multimask_output=False)
                return masks[0].astype(np.uint16)
            return f
        return {'microsam.vit_b_lm': ('box', microsam)}
    if group == 'cellpose3':
        def make(kind):
            def b():
                from cellpose import models
                m = models.Cellpose(model_type=kind, gpu=False)
                return lambda img, box, pt: m.eval(img, diameter=None, channels=[0, 0])[0].astype(np.uint16)
            return b
        return {'cellpose3.cyto3': ('crop', make('cyto3')), 'cellpose3.nuclei': ('crop', make('nuclei'))}
    if group == 'stardist':
        def sd():
            from stardist.models import StarDist2D
            m = StarDist2D.from_pretrained('2D_versatile_fluo')
            return lambda img, box, pt: m.predict_instances(pnorm(img), verbose=False)[0].astype(np.uint16)
        return {'stardist.versatile-fluo': ('crop', sd)}
    raise SystemExit(f'unknown group {group}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('group'); ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--token-file', required=True)
    ap.add_argument('--preload', action='store_true', help='load the first method before serving (slow imports)')
    a = ap.parse_args()
    token = open(a.token_file).read().strip()
    registry = build(a.group); loaded = {}; lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _reply(self, code, body, ctype='application/json'):
            self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def _auth(self):
            if self.headers.get('X-API-Token', '') != token:
                self._reply(401, b'{"detail": "bad token"}'); return False
            return True

        def do_GET(self):
            if not self._auth(): return
            if urlparse(self.path).path == '/methods':
                return self._reply(200, json.dumps({'group': a.group, 'methods': sorted(registry),
                                                    'prompt': {k: v[0] for k, v in registry.items()}}).encode())
            self._reply(404, b'{}')

        def do_POST(self):
            if not self._auth(): return
            u = urlparse(self.path); parts = u.path.strip('/').split('/')
            if len(parts) != 2 or parts[0] != 'segment' or parts[1] not in registry:
                return self._reply(404, b'{"detail": "unknown method"}')
            q = parse_qs(u.query)
            try:
                box = [int(v) for v in q['box'][0].split(',')]
                pt = [int(v) for v in q['point'][0].split(',')] if 'point' in q else None
                img = tifffile.imread(io.BytesIO(self.rfile.read(int(self.headers['Content-Length']))))
                with lock:
                    if parts[1] not in loaded:
                        loaded.clear(); loaded[parts[1]] = registry[parts[1]][1]()
                    lab = np.asarray(loaded[parts[1]](img, box, pt), np.uint16)
                out = io.BytesIO(); tifffile.imwrite(out, lab, photometric='minisblack')
                self._reply(200, out.getvalue(), 'application/octet-stream')
            except Exception as exc:  # report, keep serving
                self._reply(500, json.dumps({'detail': str(exc)[:300]}).encode())

        def log_message(self, *args):
            pass

    if a.preload:
        first = sorted(registry)[0]; loaded[first] = registry[first][1]()
    print(f'side server {a.group} on {a.host}:{a.port}: {sorted(registry)}', flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()
