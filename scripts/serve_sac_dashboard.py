#!/usr/bin/env python3
"""Local, read-only SAC progress/results viewer. Does not import JAX or train."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.sac_dashboard_data import experiments, inside, snapshot


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, outputs, **kwargs):
        self.outputs = outputs
        super().__init__(*args, **kwargs)

    def respond(self, data, content_type, code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/api/runs":
                data = dict(runs=[p.name for p in experiments(self.outputs)])
            elif url.path == "/api/run":
                name = query.get("id", [""])[0]
                available = {p.name: p for p in experiments(self.outputs)}
                if name not in available:
                    raise ValueError("unknown experiment")
                data = snapshot(available[name], self.outputs)
            elif url.path == "/media":
                path = inside(self.outputs / query.get("path", [""])[0], self.outputs)
                if path.suffix.lower() not in {".gif", ".png", ".jpg", ".json"} or not path.is_file():
                    raise ValueError("unsupported artifact")
                self.respond(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                return
            elif url.path in ("/", "/app.js", "/style.css"):
                name = "index.html" if url.path == "/" else url.path[1:]
                path = REPO / "web/sac_dashboard" / name
                self.respond(path.read_bytes(), (mimetypes.guess_type(name)[0] or "text/plain") + "; charset=utf-8")
                return
            else:
                self.send_error(404)
                return
            self.respond(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.respond(json.dumps({"error": str(exc)}, ensure_ascii=False).encode(), "application/json; charset=utf-8", 400)

    def log_message(self, fmt, *args):
        if len(args) > 1 and str(args[1]) != "200":
            super().log_message(fmt, *args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, default=REPO / "outputs")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, outputs=args.outputs.resolve()))
    print(f"SAC dashboard: http://localhost:{server.server_port}\n읽기 전용 · 5초 갱신 · Ctrl+C는 웹만 종료", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
