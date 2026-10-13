"""
examen.py — monta un examen (o una hoja de problemas) a partir de las
plantillas de las colecciones, con datos nuevos, en dos cuadernos: el
enunciado (sin resolución) y la solución completa.

Uso (desde la raíz del repositorio):

    python -m scripts.colecciones.examen --nombre parcial1 --semilla 99 \\
        --problemas 05:holgura-compartida 08:dos-rutas-criticas 12:tendencia

    python -m scripts.colecciones.examen --nombre hoja3 --problemas 05 08 10 12   # variante al azar

Cada problema se indica como `<prefijo de colección>[:<variante>]`. Salida en
notebooks/examenes/<nombre>/<nombre>_enunciado.{py,ipynb} y
<nombre>_solucion.{py,ipynb}; se renderizan con
`./render_colecciones.sh notebooks/examenes/<nombre>/*.ipynb` → docs/examenes/<nombre>/.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .generador import REPO, SRC, NOTEBOOKS, generar_ejercicio, _emparejar   # noqa: E402
from .cuaderno import render, partir_enunciado, md, code                      # noqa: E402

sys.path.insert(0, str(SRC))
from .plantillas import PLANTILLAS                                            # noqa: E402

EXAMENES = NOTEBOOKS / "examenes"

LIMPIAR = "for _v in [v for v in list(globals()) if v.startswith('pregunta_')]:\n    del globals()[_v]"


def _plantilla(prefijo):
    for p in PLANTILLAS:
        if p.META["slug"].startswith(prefijo):
            return p
    raise SystemExit(f"No hay ninguna colección que empiece por {prefijo!r}")


def _variante_k(plantilla, nombre, rng):
    nombres = [v.nombre for v in plantilla.VARIANTES]
    if nombre is None:
        return int(rng.integers(0, len(nombres))) + 1
    if nombre not in nombres:
        raise SystemExit(f"La colección {plantilla.META['slug']} no tiene la variante {nombre!r}; "
                         f"disponibles: {', '.join(nombres)}")
    return nombres.index(nombre) + 1


def montar(nombre, problemas, semilla, contexto=True):
    rng = np.random.default_rng(semilla)
    enunciado, solucion = [], []
    resumen = []
    for i, spec in enumerate(problemas, 1):
        prefijo, _, var = spec.partition(":")
        plantilla = _plantilla(prefijo)
        k = _variante_k(plantilla, var or None, rng)
        rng_i = np.random.default_rng(int(rng.integers(0, 2**31)))
        celdas, subtitulo, variante = generar_ejercicio(plantilla, k, rng_i, con_contexto=contexto)
        enun, resol = partir_enunciado(celdas)
        # quarto exige etiquetas únicas: se les añade el número de problema
        def etiquetar(cs):
            return [(t, re.sub(r"^(#\| label: \S+)$", rf"\1-p{i}", x, flags=re.M) if t == "code" else x) for t, x in cs]
        enun, resol = etiquetar(enun), etiquetar(resol)
        # el primer md "# Enunciado" se sustituye por la cabecera del problema
        enun = [c for c in enun if not (c[0] == "md" and c[1].strip() == "# Enunciado")]
        cabecera = [md(f"# Problema {i}"), code(LIMPIAR)]
        enunciado += cabecera + enun
        solucion += cabecera + enun + [md(f"## Solución del problema {i}")] + [
            (t, x) for t, x in resol if not (t == "md" and x.strip().startswith("# Resolución"))]
        resumen.append((i, plantilla.META["titulo"], variante.nombre, variante.nivel, subtitulo))
    carpeta = EXAMENES / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    sub = "; ".join(f"P{i}: {t} ({v})" for i, t, v, _, _ in resumen)
    for sufijo, celdas in (("enunciado", enunciado), ("solucion", solucion)):
        ruta = carpeta / f"{nombre}_{sufijo}.py"
        ruta.write_text(render(celdas, titulo=f"{nombre} — {sufijo}", subtitulo=sub,
                               coleccion="examen", variante=",".join(p for p in problemas), nivel=0),
                        encoding="utf8")
        _emparejar(ruta)
        print(f"  + {ruta.relative_to(REPO)}")
    for i, t, v, nivel, s in resumen:
        print(f"    Problema {i}: {t} — {v} (nivel {nivel}): {s}")
    return carpeta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", required=True)
    p.add_argument("--problemas", nargs="+", required=True, help="<colección>[:<variante>] …")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--sin-contexto", action="store_true")
    args = p.parse_args(argv)
    montar(args.nombre, args.problemas, args.semilla, contexto=not args.sin_contexto)


if __name__ == "__main__":
    main()
