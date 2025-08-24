# api/index.py
from http.server import BaseHTTPRequestHandler
import json, traceback

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            data = {"ok": True, "message": "Hello from Python on Vercel"}
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            # 回 500，但把堆疊寫到日誌，避免白畫面
            traceback.print_exc()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": "internal"}).encode("utf-8"))
