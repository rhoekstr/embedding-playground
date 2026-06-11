#!/usr/bin/env python3
"""
Package the playground as ONE self-contained HTML file for distribution.

Inlines embeddings.js and app.js into index.html and writes
dist/embedding-playground.html. The result opens by double-click from
file:// with no server, no install, no network — hand it to participants
by email or shared drive.

Run:  python3 build/make_dist.py   (after preprocess.py if data changed)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "embedding-playground.html"

html = (ROOT / "index.html").read_text(encoding="utf-8")

for name in ("embeddings.js", "app.js"):
    src = (ROOT / name).read_text(encoding="utf-8")
    if "</script" in src.lower():
        sys.exit(f"{name} contains '</script' — would break inlining; fix before packaging.")
    tag = f'<script src="{name}"></script>'
    if tag not in html:
        sys.exit(f"index.html is missing the expected tag: {tag}")
    html = html.replace(tag, f"<script>\n/* inlined {name} */\n{src}\n</script>")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
