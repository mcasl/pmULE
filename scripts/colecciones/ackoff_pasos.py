"""Sustituye la tabla de ``ackoff()`` de los cuadernos por el visor paso a paso.

Para cada ``tabla, ... = obj.ackoff(args)`` (columna 0) busca dónde se muestra
``tabla`` —última línea de la misma celda o una celda posterior que solo contiene
``tabla``— y lo cambia por ``obj.display_ackoff_pasos(args)`` (sin ``report``),
conservando las opciones ``#|`` de la celda. Los pares «introducción + visor»
añadidos en una versión anterior se eliminan. Es idempotente.

Uso (desde la raíz):
    python -m scripts.colecciones.ackoff_pasos notebooks/Ejercicio_*.py notebooks/colecciones/*/*.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .pasos import _celdas, _es_codigo, _argumento_completo

ASIGNA = re.compile(r"^(\w+)\s*,[^=\n]*=\s*(\w+)\.ackoff\(", re.M)
INTRO_ANTIGUA = "Construcción de la tabla paso a paso"


def _argumentos(texto: str):
    """Argumentos de nivel superior de una llamada (texto entre sus paréntesis)."""
    partes, nivel, ini, i = [], 0, 0, 0
    while i < len(texto):
        c = texto[i]
        if c in "([{":
            nivel += 1
        elif c in ")]}":
            nivel -= 1
        elif c in "'\"":
            i = texto.index(c, i + 1)
        elif c == "," and nivel == 0:
            partes.append(texto[ini:i].strip())
            ini = i + 1
        i += 1
    ultimo = texto[ini:].strip()
    if ultimo:
        partes.append(ultimo)
    return [p for p in partes if p]


def _llamada(objeto: str, args_texto: str) -> str:
    args = [a for a in _argumentos(args_texto) if not re.match(r"report\s*=", a)]
    args = [re.sub(r"\s*\n\s*", " ", a) for a in args]
    return f"{objeto}.display_ackoff_pasos(\n" + ",\n".join("    " + a for a in args) + ")"


def _nueva(celda: str) -> str:
    return celda.rstrip("\n") + "\n\n"


def convierte(texto: str):
    """(texto nuevo, sin_reconocer) para un cuaderno .py."""
    cabecera, original = _celdas(texto)
    if not original:
        return texto, []
    # se quitan los pares «introducción + visor» de la versión anterior
    celdas, k = [], 0
    while k < len(original):
        c = original[k]
        if INTRO_ANTIGUA in c and c.startswith("# %% [markdown]"):
            k += 1
            if k < len(original) and "display_ackoff_pasos(" in original[k]:
                k += 1
            continue
        celdas.append(c)
        k += 1

    sin = []
    for i, c in enumerate(celdas):
        m = ASIGNA.search(c) if _es_codigo(c) else None
        if not m or m.group(1) == "_":
            continue
        var, obj = m.group(1), m.group(2)
        nueva = _llamada(obj, _argumento_completo(c[m.end():]))
        lineas = c.rstrip("\n").split("\n")
        if lineas[-1].strip() in (var, f"display({var})"):      # se muestra en la misma celda
            celdas[i] = _nueva("\n".join(lineas[:-1] + [nueva]))
            continue
        hecho = False
        for j in range(i + 1, len(celdas)):                      # o en una celda posterior
            if not _es_codigo(celdas[j]):
                continue
            if ASIGNA.search(celdas[j]):
                break
            cuerpo = celdas[j].rstrip("\n").split("\n")
            codigo = "\n".join(l for l in cuerpo[1:] if not l.startswith("#|")).strip()
            if codigo in (var, f"display({var})"):
                opciones = [l for l in cuerpo[1:] if l.startswith("#|")]
                celdas[j] = _nueva("\n".join([cuerpo[0]] + opciones + [nueva]))
                hecho = True
                break
        if not hecho:
            sin.append((var, obj))
    return (cabecera or "") + "".join(celdas), sin


def main(rutas):
    for ruta in rutas:
        p = Path(ruta)
        t = p.read_text()
        if ".ackoff(" not in t or "scratch" in p.parts:
            continue
        nuevo, sin = convierte(t)
        if nuevo != t:
            p.write_text(nuevo)
            print("convertido", ruta)
        for s in sin:
            print("  SIN RECONOCER", ruta, s)


if __name__ == "__main__":
    main(sys.argv[1:])
