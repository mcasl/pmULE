"""Hace deterministas los identificadores aleatorios de los HTML renderizados.

Regenerar un cuaderno sin cambios de contenido debe dar el mismo HTML: así ``git`` no
acumula blobs nuevos de varios MB en cada renderizado. El único azar que queda en el
HTML son los identificadores de las tablas de pandas (``Styler``: ``T_xxxxx_...``, cinco
cifras hexadecimales con ``uuid4``), que aquí se renumeran por orden de primera aparición
en el fichero (``T_00000``, ``T_00001``, ...) en el CSS, el ``id`` de la tabla y el de
cada celda. Además Quarto escribe en orden variable los atributos ``data-execution_count``,
``data-outputid`` y ``data-quarto-private-1`` de cada celda; se dejan en orden alfabético.
Es idempotente.

Uso: python scripts/normaliza_ids.py docs/a.html docs/b.html ...
"""
import re
import sys
from pathlib import Path

PATRON_STYLER = re.compile(r"\bT_([0-9a-f]{5})(?![0-9A-Za-z])")
ATRIBUTO_CELDA = r'\sdata-(?:execution_count|outputid|quarto-private-1)="[^"]*"'
PATRON_ATRIBUTOS = re.compile(f"(?:{ATRIBUTO_CELDA})+")
PATRON_UN_ATRIBUTO = re.compile(r'\s(data-(?:execution_count|outputid|quarto-private-1))="([^"]*)"')


def _ordena_atributos(m):
    attrs = sorted(PATRON_UN_ATRIBUTO.findall(m.group(0)))
    return "".join(f' {k}="{v}"' for k, v in attrs)


def normaliza(texto: str) -> str:
    """Renumera los ``T_xxxxx`` de ``texto`` por orden de primera aparición y ordena los atributos de celda de Quarto."""
    orden = {}

    def nuevo(m):
        return "T_%05x" % orden.setdefault(m.group(1), len(orden))

    return PATRON_ATRIBUTOS.sub(_ordena_atributos, PATRON_STYLER.sub(nuevo, texto))


def normaliza_fichero(ruta: Path) -> bool:
    """Reescribe ``ruta`` si cambia algo; devuelve si cambió."""
    texto = ruta.read_text(encoding="utf-8")
    nuevo = normaliza(texto)
    if nuevo != texto:
        ruta.write_text(nuevo, encoding="utf-8")
        return True
    return False


if __name__ == "__main__":
    cambiados = sum(normaliza_fichero(Path(p)) for p in sys.argv[1:])
    print(f"Ids de tablas normalizados en {cambiados} de {len(sys.argv) - 1} HTML")
