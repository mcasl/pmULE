"""Deja solo el visor paso a paso de Zaderenko en los cuadernos.

Sustituye la celda que muestra ``obj.zaderenko(...)`` (la matriz ya resuelta) por
``obj.display_zaderenko_pasos(...)`` —el visor cuyo primer paso es el resultado final—,
conservando las opciones ``#|`` de la celda (pie y etiqueta de la tabla), y elimina
la celda del visor que añadió ``pasos.py`` con su introducción. Idempotente.

Uso (desde la raíz):
    python -m scripts.colecciones.zaderenko_visor notebooks/Ejercicio_*.py notebooks/colecciones/*/*.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .pasos import _celdas, _es_codigo, INTRO

LLAMADA = re.compile(r"^(\w+)\.zaderenko\((.*)\)\s*$", re.S | re.M)
INTRO_NUEVA = ("Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; "
               "a partir del siguiente se construye desde el principio (avance con los botones):")


def _nueva(celda: str) -> str:
    return celda.rstrip("\n") + "\n\n"


def convierte(texto: str):
    cabecera, celdas = _celdas(texto)
    if not celdas:
        return texto, []
    sin, i = [], 0
    while i < len(celdas):
        c = celdas[i]
        cuerpo = c.rstrip("\n").split("\n")
        codigo = "\n".join(l for l in cuerpo[1:] if not l.startswith("#|")).strip()
        m = LLAMADA.search(codigo) if _es_codigo(c) else None
        if not m:
            i += 1
            continue
        obj, args = m.group(1), m.group(2)
        previo = codigo[:m.start()].rstrip("\n")        # líneas anteriores de la misma celda
        opciones = [l for l in cuerpo[1:] if l.startswith("#|")]
        # visor de pasos.py (y su introducción) que va detrás
        v = next((j for j in range(i + 1, len(celdas))
                  if _es_codigo(celdas[j]) and f"{obj}.display_zaderenko_pasos(" in celdas[j]), None)
        if v is not None:
            visor = "\n".join(l for l in celdas[v].rstrip("\n").split("\n")[1:] if not l.startswith("#|"))
            quitar = {v}
            if v - 1 > i and INTRO["zaderenko"] in celdas[v - 1]:
                quitar.add(v - 1)
            celdas = [x for k, x in enumerate(celdas) if k not in quitar]
        else:
            visor = f"{obj}.display_zaderenko_pasos({args})"
        celdas[i] = _nueva("\n".join([cuerpo[0]] + opciones + ([previo] if previo else []) + [visor]))
        celdas.insert(i, _nueva(f"# %% [markdown]\n# {INTRO_NUEVA}"))
        i += 2
    return (cabecera or "") + "".join(celdas), sin


def main(rutas):
    for ruta in rutas:
        p = Path(ruta)
        t = p.read_text()
        if ".zaderenko(" not in t or "scratch" in p.parts:
            continue
        nuevo, _ = convierte(t)
        if nuevo != t:
            p.write_text(nuevo)
            print("convertido", ruta)


if __name__ == "__main__":
    main(sys.argv[1:])
