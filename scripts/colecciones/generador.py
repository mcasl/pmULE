"""
generador.py — genera las colecciones de ejercicios cortos por competencia
en notebooks/colecciones/<coleccion>/<Prefijo>_NN.{py,ipynb} y actualiza
docs/index.md y notebooks/colecciones/README.md.

Uso (desde la raíz del repositorio):

    python -m scripts.colecciones.generador                 # todas las colecciones, 3 ejercicios cada una
    python -m scripts.colecciones.generador --solo 05 09    # solo esas colecciones
    python -m scripts.colecciones.generador --n 5           # 5 ejercicios por colección
    python -m scripts.colecciones.generador --ejecutar      # además ejecuta los .ipynb (nbconvert)
    python -m scripts.colecciones.generador --semilla 7     # otra familia de proyectos

Los cuadernos son deterministas: la misma semilla produce los mismos datos.
Un ejercicio ya existente no se reescribe salvo con --forzar.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
SRC = REPO / "src"
NOTEBOOKS = REPO / "notebooks"
COLECCIONES = NOTEBOOKS / "colecciones"
DOCS = REPO / "docs"
INDEX = DOCS / "index.md"
MARCA_INI = "<!-- colecciones:inicio -->"
MARCA_FIN = "<!-- colecciones:fin -->"

sys.path.insert(0, str(SRC))

from .cuaderno import render                       # noqa: E402
from .plantillas import PLANTILLAS                 # noqa: E402


def generar_coleccion(plantilla, n, semilla, forzar, ejecutar, indice):
    meta = plantilla.META
    carpeta = COLECCIONES / meta["slug"]
    carpeta.mkdir(parents=True, exist_ok=True)
    generados = []
    for k in range(1, n + 1):
        nombre = f"{meta['prefijo']}_{k:02d}"
        ruta_py = carpeta / f"{nombre}.py"
        if ruta_py.exists() and not forzar:
            print(f"  = {ruta_py.relative_to(REPO)} (existe; use --forzar para regenerar)")
            generados.append((nombre, _leer_subtitulo(ruta_py)))
            continue
        rng = np.random.default_rng(semilla * 1000 + indice * 100 + k)
        celdas, subtitulo = plantilla.generar(k, rng)
        texto = render(celdas, titulo=f"{meta['prefijo']} {k:02d}", subtitulo=subtitulo)
        ruta_py.write_text(texto, encoding="utf8")
        print(f"  + {ruta_py.relative_to(REPO)}  —  {subtitulo}")
        generados.append((nombre, subtitulo))
        _emparejar(ruta_py)
        if ejecutar:
            _ejecutar(ruta_py.with_suffix(".ipynb"))
    return generados


def _leer_subtitulo(ruta_py):
    m = re.search(r'^# subtitle: "(.*)"$', ruta_py.read_text(encoding="utf8"), re.M)
    return m.group(1) if m else ""


def _emparejar(ruta_py):
    """Crea/actualiza el .ipynb gemelo (jupytext lee el jupytext.toml de notebooks/)."""
    try:
        subprocess.run([sys.executable, "-m", "jupytext", "--quiet", "--to", "ipynb", str(ruta_py)],
                       check=True, cwd=ruta_py.parent)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"    (no se pudo crear el .ipynb: {exc})")


def _ejecutar(ruta_ipynb):
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--ExecutePreprocessor.timeout=600", str(ruta_ipynb)]
    r = subprocess.run(cmd, cwd=ruta_ipynb.parent, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ERROR al ejecutar {ruta_ipynb.name}:\n{r.stderr[-2500:]}")
    else:
        print(f"    ejecutado {ruta_ipynb.name}")


def escribir_readme(resumen):
    lineas = ["# Colecciones de ejercicios cortos por competencia", "",
              "Cada carpeta es una colección de ejercicios breves que ejercitan una sola competencia. "
              "Los cuadernos siguen el guion de los `Ejercicio_NN` (enunciado con `pregunta_N`, tabla de datos, "
              "resolución por apartados con pmULE) y están emparejados con jupytext (`.py` ↔ `.ipynb`).", "",
              "Se generan con `python -m scripts.colecciones.generador` desde la raíz del repositorio "
              "(ver la ayuda del script). Los datos de cada ejercicio son deterministas (semilla fija) y se "
              "eligen por muestreo hasta que el proyecto presenta los rasgos que la competencia necesita "
              "(ficticias, varias rutas críticas, holguras distintas, sobrecoste, …).", "",
              "Para renderizar a HTML en `docs/colecciones/`: `./render_colecciones.sh` (todos) o "
              "`./render_colecciones.sh notebooks/colecciones/05_holguras/Holguras_01.ipynb`.", ""]
    for meta, generados in resumen:
        lineas.append(f"## {meta['slug']} — {meta['titulo']}")
        lineas.append("")
        lineas.append(meta["descripcion"])
        lineas.append("")
        for nombre, sub in generados:
            lineas.append(f"- `{nombre}` — {sub}")
        lineas.append("")
    (COLECCIONES / "README.md").write_text("\n".join(lineas), encoding="utf8")


def actualizar_index(resumen):
    bloque = [MARCA_INI, "", "# Colecciones de ejercicios cortos por competencia", "",
              "Ejercicios breves, de una sola competencia, para practicar cada paso por separado. "
              "Los cuadernos están en `notebooks/colecciones/`.", ""]
    for meta, generados in resumen:
        bloque.append(f"## {meta['titulo']}")
        bloque.append("")
        bloque.append(meta["descripcion"] + "  ")
        bloque.append("")
        for nombre, sub in generados:
            bloque.append(f"[{nombre.replace('_', ' ')}](colecciones/{meta['slug']}/{nombre}.html) — {sub}  ")
        bloque.append("")
    bloque.append(MARCA_FIN)
    nuevo = "\n".join(bloque)
    texto = INDEX.read_text(encoding="utf8") if INDEX.exists() else "# Ejercicios resueltos\n\n"
    if MARCA_INI in texto and MARCA_FIN in texto:
        texto = texto[:texto.index(MARCA_INI)] + nuevo + texto[texto.index(MARCA_FIN) + len(MARCA_FIN):]
    else:
        texto = texto.rstrip("\n") + "\n\n" + nuevo + "\n"
    INDEX.write_text(texto, encoding="utf8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solo", nargs="*", default=None, help="prefijos numéricos de las colecciones (05 09 …)")
    p.add_argument("--n", type=int, default=None, help="ejercicios por colección (por defecto, el de cada plantilla)")
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--forzar", action="store_true", help="regenerar los que ya existen")
    p.add_argument("--ejecutar", action="store_true", help="ejecutar los .ipynb con nbconvert")
    p.add_argument("--sin-index", action="store_true", help="no tocar docs/index.md ni el README")
    args = p.parse_args(argv)

    resumen = []
    for indice, plantilla in enumerate(PLANTILLAS, 1):
        meta = plantilla.META
        if args.solo and not any(meta["slug"].startswith(s) for s in args.solo):
            # las colecciones no seleccionadas siguen apareciendo en el índice si ya existen
            existentes = sorted((COLECCIONES / meta["slug"]).glob(f"{meta['prefijo']}_*.py"))
            if existentes:
                resumen.append((meta, [(r.stem, _leer_subtitulo(r)) for r in existentes]))
            continue
        print(f"[{meta['slug']}] {meta['titulo']}")
        n = args.n or meta["n_defecto"]
        generados = generar_coleccion(plantilla, n, args.semilla, args.forzar, args.ejecutar, indice)
        resumen.append((meta, generados))
    if not args.sin_index:
        escribir_readme(resumen)
        actualizar_index(resumen)
        print(f"README e índice actualizados ({INDEX.relative_to(REPO)})")


if __name__ == "__main__":
    main()
