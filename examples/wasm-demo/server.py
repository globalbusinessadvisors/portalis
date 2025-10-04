#!/usr/bin/env python3
"""Simple HTTP server for WASM demo"""
import http.server
import socketserver
import os

PORT = 8000
os.chdir('/workspace/portalis/examples/wasm-demo')

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for WASM
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"🚀 Server running at http://localhost:{PORT}")
    print(f"📁 Serving: {os.getcwd()}")
    print(f"\n🌐 Open http://localhost:{PORT} in your browser\n")
    httpd.serve_forever()
