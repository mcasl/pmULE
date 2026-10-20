"""Incrusta en los HTML los <script src="https://..."> que Quarto no pudo
incrustar en el clúster (los nodos de cálculo no tienen salida a Internet: con
embed-resources, jquery y requirejs quedan enlazados al CDN).

Uso: python scripts/cluster/incrusta_cdn.py docs/a.html docs/b.html ...
Las bibliotecas se descargan una vez y se guardan en scripts/cluster/vendor/.
"""
import re
import sys
import urllib.request
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"
PATRON = re.compile(r'<script([^>]*?)\ssrc="(https://[^"]+)"([^>]*)>\s*</script>')


def contenido(url: str) -> str:
    VENDOR.mkdir(exist_ok=True)
    local = VENDOR / re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[1])
    if not local.exists():
        with urllib.request.urlopen(url, timeout=30) as r:
            local.write_bytes(r.read())
    return local.read_text(encoding="utf-8")


def incrusta(ruta: Path) -> int:
    html = ruta.read_text(encoding="utf-8")
    n = 0

    def sustituye(m):
        nonlocal n
        n += 1
        codigo = contenido(m.group(2)).replace("</script", "<\\/script")
        return f"<script{m.group(1)}{m.group(3)}>{codigo}</script>"

    nuevo = PATRON.sub(sustituye, html)
    if n:
        ruta.write_text(nuevo, encoding="utf-8")
    return n


if __name__ == "__main__":
    total = sum(incrusta(Path(p)) for p in sys.argv[1:])
    print(f"Scripts del CDN incrustados: {total}")
