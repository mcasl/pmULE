"""Deja solo el visor paso a paso de PERT y Roy en los cuadernos.

Sustituye la celda que dibuja ``obj.pert(durations=...)`` / ``obj.roy(durations=...)``
(el grafo ya resuelto) por ``obj.display_pert_pasos(...)`` / ``obj.display_roy_pasos(...)``
—visores cuyo primer paso es el grafo resuelto—, conservando las opciones ``#|`` de la
celda (pie y etiqueta de la figura) y elimina el visor que añadió ``pasos.py`` con su
introducción. Si la llamada guarda un fichero (``filename=``) se mantiene, sin mostrarse
(``_ = ...``), delante del visor. Los grafos sin duraciones no se tocan. Idempotente.

Uso (desde la raíz):
    python -m scripts.colecciones.pert_roy_visor notebooks/Ejercicio_*.py notebooks/colecciones/*/*.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .pasos import _celdas, _es_codigo, _durations, INTRO

INTRO_NUEVA = {
    "pert": "Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales "
            "paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se "
            "calcula desde el principio (avance con los botones):",
    "roy": "Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. "
           "El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde "
           "el principio (avance con los botones):",
}


def _nueva(celda: str) -> str:
    return celda.rstrip("\n") + "\n\n"


def _partes(celda: str):
    cuerpo = celda.rstrip("\n").split("\n")
    opciones = [l for l in cuerpo[1:] if l.startswith("#|")]
    codigo = "\n".join(l for l in cuerpo[1:] if not l.startswith("#|")).strip()
    return cuerpo[0], opciones, codigo


def convierte(texto: str, metodo: str):
    cabecera, celdas = _celdas(texto)
    if not celdas:
        return texto, []
    sin, i = [], 0
    while i < len(celdas):
        c = celdas[i]
        if not _es_codigo(c):
            i += 1
            continue
        cab, opciones, codigo = _partes(c)
        m = re.fullmatch(rf"(\w+)\.{metodo}\((.*)\)", codigo, re.S)
        d = _durations(codigo, metodo) if m else None
        if not (m and d):
            i += 1
            continue
        obj, expr = d
        # visor de pasos.py (y su introducción) que va detrás, antes de otro grafo con tiempos
        v = None
        for j in range(i + 1, len(celdas)):
            if not _es_codigo(celdas[j]):
                continue
            _, _, cj = _partes(celdas[j])
            if re.fullmatch(rf"\w+\.{metodo}\(.*\)", cj, re.S) and _durations(cj, metodo):
                break
            if f"{obj}.display_{metodo}_pasos(" in cj:
                v = j
                break
        if v is not None:
            visor = _partes(celdas[v])[2]
            quitar = {v}
            if v - 1 > i and INTRO[metodo] in celdas[v - 1]:
                quitar.add(v - 1)
            celdas = [x for k, x in enumerate(celdas) if k not in quitar]
        else:
            visor = f"{obj}.display_{metodo}_pasos({expr})"
        if "filename" in codigo:
            visor = f"_ = {codigo}\n{visor}"
        celdas[i] = _nueva("\n".join([cab] + opciones + [visor]))
        celdas.insert(i, _nueva(f"# %% [markdown]\n# {INTRO_NUEVA[metodo]}"))
        i += 2
    return (cabecera or "") + "".join(celdas), sin


def main(rutas):
    for ruta in rutas:
        p = Path(ruta)
        t = p.read_text()
        if "scratch" in p.parts or not re.search(r"\.(pert|roy)\(", t):
            continue
        nuevo = t
        for metodo in ("pert", "roy"):
            nuevo, _ = convierte(nuevo, metodo)
        if nuevo != t:
            p.write_text(nuevo)
            print("convertido", ruta)


if __name__ == "__main__":
    main(sys.argv[1:])
