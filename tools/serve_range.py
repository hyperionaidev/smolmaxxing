#!/usr/bin/env python3
"""Tiny HTTP file server with Range support (so tools/tvfetch on the TV can resume big downloads).
Serves one directory, read-only, GET only. usage: python3 -I serve_range.py <dir> <bind-ip> <port>"""
import http.server, os, re, sys

ROOT = os.path.realpath(sys.argv[1])

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        p = os.path.realpath(os.path.join(ROOT, self.path.lstrip("/")))
        if not p.startswith(ROOT + os.sep) or not os.path.isfile(p):
            self.send_error(404); return
        size = os.path.getsize(p); start = 0
        m = re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
        if m: start = int(m.group(1))
        if start >= size:
            self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return
        self.send_response(206 if m else 200)
        self.send_header("Content-Length", str(size - start))
        if m: self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            while chunk := f.read(1 << 20):
                self.wfile.write(chunk)

http.server.ThreadingHTTPServer((sys.argv[2], int(sys.argv[3])), H).serve_forever()
