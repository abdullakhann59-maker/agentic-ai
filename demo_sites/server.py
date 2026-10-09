"""Local demo websites so the scraper demo never depends on the internet.

Run:  python demo_sites/server.py          (serves http://localhost:8001)

Pages:
  /static/     plain HTML table (static site)
  /iframe/     page with 2 content iframes (one nested) + 1 tiny ad iframe
  /dynamic/    products loaded by JavaScript from /api/products (dynamic site with an API)
  /embedded/   products embedded as JSON inside a <script> tag (Next.js style)
  /captcha/    a page that shows a CAPTCHA wall
  /injection/  a page with hidden prompt-injection text
  /private/    disallowed by robots.txt
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SITE_DIR = Path(__file__).parent / "site"
CATEGORIES = ["Laptop", "Phone", "Headphones", "Monitor", "Keyboard", "Mouse", "Tablet", "Camera"]


def products(page: int, per_page: int = 8, total_pages: int = 3) -> dict:
    items = []
    if 1 <= page <= total_pages:
        for i in range(per_page):
            n = (page - 1) * per_page + i + 1
            cat = CATEGORIES[(n - 1) % len(CATEGORIES)]
            items.append({
                "id": n,
                "title": f"{cat} Model {100 + n}",
                "category": cat,
                "price_inr": 1999 + n * 750,
                "rating": round(3.5 + (n % 15) / 10, 1),
                "in_stock": n % 4 != 0,
            })
    return {"page": page, "total_pages": total_pages, "count": len(items), "products": items}


ROBOTS = "User-agent: *\nDisallow: /private/\n"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/robots.txt":
            return self._send(ROBOTS.encode(), "text/plain")
        if url.path == "/api/products":
            page = int(parse_qs(url.query).get("page", ["1"])[0])
            return self._send(json.dumps(products(page)).encode(), "application/json")
        if url.path == "/embedded/":
            html = (SITE_DIR / "embedded" / "index.html").read_text()
            html = html.replace("__DATA__", json.dumps({"props": {"pageProps": products(1)}}))
            return self._send(html.encode(), "text/html; charset=utf-8")
        return super().do_GET()


def start(port: int = 8001) -> ThreadingHTTPServer:
    """Start in a background thread (used by tests)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"Demo sites running on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
