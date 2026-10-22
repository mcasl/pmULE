"""Muestra el calendario del proyecto siempre con el visor interactivo.

En las celdas que calculan ``cal = obj.calendar(args)`` y terminan mostrando la tabla
(la última línea es el propio nombre ``cal``) sustituye esa última línea por
``obj.display_calendario(args)`` —el calendario interactivo junto al grafo PERT (o Roy,
en la colección ``03_roy``)—. La asignación se conserva, porque el resto del cuaderno usa
la tabla; las opciones ``#|`` (pie y etiqueta) también. Idempotente. Las celdas en las
que la tabla se transforma antes de mostrarla (``.drop``, ``.rename``…) se listan sin tocar.

Uso (desde la raíz):
    python -m scripts.colecciones.calendario_visor notebooks/Ejercicio_*.py notebooks/colecciones/*/*.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ASIGNA = re.compile(r"^(\w+) *= *(\w+)\.calendar\((.*)\)\s*$")


def _balanceado(s: str) -> bool:
    n = 0
    for ch in s:
        n += (ch == "(") - (ch == ")")
        if n < 0:
            return False
    return n == 0


ECO = re.compile(r"^(\w+)(\[\[.*\]\])?$")          # calendario   o   calendario[['H_total', ...]]
BARRA = re.compile(r"^(\w+)\.calendar\((.*?)\)(\[\[.*\]\])?$")     # obj.calendar(args)[[...]]


def convierte(texto: str, grafo: str = "pert"):
    """Devuelve (texto nuevo, celdas convertidas, celdas dudosas)."""
    celdas = re.split(r"(?m)^(?=# %%)", texto)
    hechas, dudosas = 0, []
    extra = f', grafo="{grafo}"' if grafo != "pert" else ""
    previas = {}                                    # variable -> (objeto, argumentos) de celdas anteriores
    for k, c in enumerate(celdas):
        if not c.startswith("# %%") or "[markdown]" in c.splitlines()[0]:
            continue
        anteriores = dict(previas)
        for l in c.split("\n"):
            m = ASIGNA.match(l)
            if m and _balanceado(m.group(3)):
                previas[m.group(1)] = (m.group(2), m.group(3))
        if "display_calendario" in c:
            continue
        lineas = c.rstrip("\n").split("\n")
        ultima = lineas[-1].strip()
        nueva = None
        m = BARRA.match(ultima)
        if m and _balanceado(m.group(2)):
            nueva = f"{m.group(1)}.display_calendario({m.group(2)}{extra})"
        else:
            e = ECO.match(ultima)
            if e:
                var = e.group(1)
                for l in lineas[:-1]:
                    m = ASIGNA.match(l)
                    if m and m.group(1) == var:
                        if _balanceado(m.group(3)):
                            nueva = f"{m.group(2)}.display_calendario({m.group(3)}{extra})"
                        break
                else:
                    if var in anteriores:           # calculado en una celda anterior
                        obj, args = anteriores[var]
                        nueva = f"{obj}.display_calendario({args}{extra})"
                    elif any(re.match(rf"^{re.escape(var)} *= *\w+\.calendar\(", l) for l in lineas):
                        dudosas.append(var)
        if nueva:
            lineas[-1] = nueva
            celdas[k] = "\n".join(lineas) + c[len(c.rstrip("\n")):]
            hechas += 1
    return "".join(celdas), hechas, dudosas


def main(rutas):
    total = 0
    for r in rutas:
        p = Path(r)
        grafo = "roy" if "03_roy" in p.parts else "pert"
        nuevo, n, dud = convierte(p.read_text(encoding="utf-8"), grafo)
        if n:
            p.write_text(nuevo, encoding="utf-8")
            total += n
        if dud:
            print(f"revisar {p}: {dud}")
    print(f"{total} celdas convertidas")


if __name__ == "__main__":
    main(sys.argv[1:])
