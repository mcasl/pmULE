"""
generador.py — genera las colecciones de ejercicios cortos por competencia
en notebooks/colecciones/<coleccion>/<Prefijo>_NN.{py,ipynb} y actualiza
docs/index.md, notebooks/colecciones/README.md y notebooks/colecciones/catalogo.json.

Uso (desde la raíz del repositorio):

    python -m scripts.colecciones.generador                 # todas las colecciones, una vez cada variante
    python -m scripts.colecciones.generador --listar        # colecciones y variantes disponibles
    python -m scripts.colecciones.generador --solo 05 09    # solo esas colecciones
    python -m scripts.colecciones.generador --n 12          # 12 ejercicios por colección (las variantes se repiten cíclicamente)
    python -m scripts.colecciones.generador --ejecutar      # además ejecuta los .ipynb (nbconvert)
    python -m scripts.colecciones.generador --semilla 7     # otra familia de proyectos
    python -m scripts.colecciones.generador --forzar        # reescribir los que ya existen

Cada colección tiene una lista de variantes (`VARIANTES` en su plantilla); el
ejercicio k usa la variante ((k-1) mod número de variantes) + 1. Los cuadernos
son deterministas: la misma semilla produce los mismos datos. Un ejercicio ya
existente no se reescribe salvo con --forzar.

Para montar un examen a partir de las plantillas: `python -m scripts.colecciones.examen`.
"""
from __future__ import annotations

import argparse
import json
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
CATALOGO = COLECCIONES / "catalogo.json"
MARCA_INI = "<!-- colecciones:inicio -->"
MARCA_FIN = "<!-- colecciones:fin -->"
PROB_CONTEXTO = 0.5          # fracción de cuadernos con contexto narrativo

sys.path.insert(0, str(SRC))

from . import contextos                             # noqa: E402
from .cuaderno import render                        # noqa: E402
from .plantillas import PLANTILLAS                  # noqa: E402


def semilla_de(semilla, indice, k):
    return semilla * 1000 + indice * 100 + k


def generar_ejercicio(plantilla, k, rng, con_contexto=True):
    """Genera las celdas de un ejercicio (sin escribirlo). Devuelve (celdas, subtitulo, variante)."""
    celdas, subtitulo, variante = plantilla.generar(k, rng)
    if con_contexto and rng.random() < PROB_CONTEXTO:
        celdas = contextos.inyectar(celdas, rng)
    return celdas, subtitulo, variante


def generar_coleccion(plantilla, n, semilla, forzar, ejecutar, indice):
    meta = plantilla.META
    carpeta = COLECCIONES / meta["slug"]
    carpeta.mkdir(parents=True, exist_ok=True)
    entradas = []
    for k in range(1, n + 1):
        nombre = f"{meta['prefijo']}_{k:02d}"
        ruta_py = carpeta / f"{nombre}.py"
        if ruta_py.exists() and not forzar:
            print(f"  = {ruta_py.relative_to(REPO)} (existe; use --forzar para regenerar)")
            entradas.append(_entrada_desde_fichero(meta, nombre, ruta_py))
            continue
        rng = np.random.default_rng(semilla_de(semilla, indice, k))
        celdas, subtitulo, variante = generar_ejercicio(plantilla, k, rng)
        texto = render(celdas, titulo=f"{meta['prefijo']} {k:02d}", subtitulo=subtitulo,
                       coleccion=meta["slug"], variante=variante.nombre, nivel=variante.nivel)
        ruta_py.write_text(texto, encoding="utf8")
        print(f"  + {ruta_py.relative_to(REPO)}  —  [{variante.nombre}, nivel {variante.nivel}] {subtitulo}")
        entradas.append(dict(nombre=nombre, coleccion=meta["slug"], variante=variante.nombre,
                             nivel=variante.nivel, subtitulo=subtitulo,
                             ruta=str(ruta_py.relative_to(REPO)), semilla=semilla_de(semilla, indice, k)))
        _emparejar(ruta_py)
        if ejecutar:
            _ejecutar(ruta_py.with_suffix(".ipynb"))
    return entradas


def _leer_cabecera(ruta_py):
    texto = ruta_py.read_text(encoding="utf8")
    campos = {}
    for clave in ("subtitle", "variante", "nivel", "coleccion"):
        m = re.search(rf'^# {clave}: "?(.*?)"?$', texto, re.M)
        if m:
            campos[clave] = m.group(1)
    return campos


def _entrada_desde_fichero(meta, nombre, ruta_py):
    c = _leer_cabecera(ruta_py)
    return dict(nombre=nombre, coleccion=meta["slug"], variante=c.get("variante", ""),
                nivel=int(c.get("nivel", 0) or 0), subtitulo=c.get("subtitle", ""),
                ruta=str(ruta_py.relative_to(REPO)), semilla=None)


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


def _variantes_md(plantilla):
    lineas = []
    for v in plantilla.VARIANTES:
        lineas.append(f"  - `{v.nombre}` (nivel {v.nivel}): {v.descripcion}")
    return lineas


def escribir_readme(resumen):
    lineas = ["# Colecciones de ejercicios cortos por competencia", "",
              "Cada carpeta es una colección de ejercicios breves que ejercitan una sola competencia. "
              "Los cuadernos siguen el guion de los `Ejercicio_NN` (enunciado con `pregunta_N`, tabla de datos, "
              "resolución por apartados con pmULE) y están emparejados con jupytext (`.py` ↔ `.ipynb`).", "",
              "Se generan con `python -m scripts.colecciones.generador` desde la raíz del repositorio "
              "(`--listar` muestra las variantes; ver la ayuda del script). Cada colección tiene varias *variantes*, "
              "una por concepto que se quiere comprobar; el ejercicio k usa la variante k (cíclicamente). "
              "Los datos de cada ejercicio son deterministas (semilla fija) y se eligen por muestreo hasta que el "
              "proyecto presenta los rasgos que la variante necesita (ficticias, varias rutas críticas, holguras "
              "distintas, sobrecoste, …). La mitad de los cuadernos llevan además un contexto narrativo con la "
              "descripción de cada actividad.", "",
              "`catalogo.json` lista todos los ejercicios con su colección, variante y nivel; "
              "`python -m scripts.colecciones.examen` monta exámenes (enunciado y solución por separado) "
              "a partir de las mismas plantillas con datos nuevos.", "",
              "Para renderizar a HTML en `docs/colecciones/`: `./render_colecciones.sh` (todos) o "
              "`./render_colecciones.sh notebooks/colecciones/05_holguras/Holguras_01.ipynb`.", ""]
    for plantilla, entradas in resumen:
        meta = plantilla.META
        lineas.append(f"## {meta['slug']} — {meta['titulo']}")
        lineas.append("")
        lineas.append(meta["descripcion"])
        lineas.append("")
        lineas.append("Variantes:")
        lineas += _variantes_md(plantilla)
        lineas.append("")
        for e in entradas:
            lineas.append(f"- `{e['nombre']}` [{e['variante']}, nivel {e['nivel']}] — {e['subtitulo']}")
        lineas.append("")
    (COLECCIONES / "README.md").write_text("\n".join(lineas), encoding="utf8")


def actualizar_index(resumen):
    bloque = [MARCA_INI, "", "# Colecciones de ejercicios cortos por competencia", "",
              "Ejercicios breves, de una sola competencia, para practicar cada paso por separado. "
              "Cada colección tiene varias variantes (un concepto que comprobar en cada una) y tres niveles. "
              "Los cuadernos están en `notebooks/colecciones/`.", ""]
    for plantilla, entradas in resumen:
        meta = plantilla.META
        bloque.append(f"## {meta['titulo']}")
        bloque.append("")
        bloque.append(meta["descripcion"] + "  ")
        bloque.append("")
        for e in entradas:
            bloque.append(f"[{e['nombre'].replace('_', ' ')}](colecciones/{meta['slug']}/{e['nombre']}.html) "
                          f"— *{e['variante']}* (nivel {e['nivel']}): {e['subtitulo']}  ")
        bloque.append("")
    examenes = sorted((NOTEBOOKS / "examenes").glob("*/*_enunciado.ipynb")) if (NOTEBOOKS / "examenes").exists() else []
    if examenes:
        bloque += ["## Exámenes montados con las plantillas", "",
                   "Generados con `python -m scripts.colecciones.examen` (enunciado y solución por separado, datos nuevos).  ", ""]
        for e in examenes:
            nombre = e.parent.name
            bloque.append(f"[{nombre}](examenes/{nombre}/{nombre}_enunciado.html) — enunciado · "
                          f"[solución](examenes/{nombre}/{nombre}_solucion.html)  ")
        bloque.append("")
    bloque.append(MARCA_FIN)
    nuevo = "\n".join(bloque)
    texto = INDEX.read_text(encoding="utf8") if INDEX.exists() else "# Ejercicios resueltos\n\n"
    if MARCA_INI in texto and MARCA_FIN in texto:
        texto = texto[:texto.index(MARCA_INI)] + nuevo + texto[texto.index(MARCA_FIN) + len(MARCA_FIN):]
    else:
        texto = texto.rstrip("\n") + "\n\n" + nuevo + "\n"
    INDEX.write_text(texto, encoding="utf8")


def escribir_catalogo(resumen):
    catalogo = []
    for plantilla, entradas in resumen:
        meta = plantilla.META
        for e in entradas:
            catalogo.append(dict(e, titulo_coleccion=meta["titulo"]))
    CATALOGO.write_text(json.dumps(catalogo, ensure_ascii=False, indent=1), encoding="utf8")


def listar():
    for plantilla in PLANTILLAS:
        meta = plantilla.META
        print(f"[{meta['slug']}] {meta['titulo']} — {len(plantilla.VARIANTES)} variantes")
        for v in plantilla.VARIANTES:
            print(f"    {v.nombre:28s} nivel {v.nivel}  {v.descripcion}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solo", nargs="*", default=None, help="prefijos numéricos de las colecciones (05 09 …)")
    p.add_argument("--n", type=int, default=None, help="ejercicios por colección (por defecto, una vez cada variante)")
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--forzar", action="store_true", help="regenerar los que ya existen")
    p.add_argument("--ejecutar", action="store_true", help="ejecutar los .ipynb con nbconvert")
    p.add_argument("--sin-index", action="store_true", help="no tocar docs/index.md, el README ni el catálogo")
    p.add_argument("--listar", action="store_true", help="listar colecciones y variantes")
    args = p.parse_args(argv)
    if args.listar:
        listar()
        return

    resumen = []
    for indice, plantilla in enumerate(PLANTILLAS, 1):
        meta = plantilla.META
        if args.solo and not any(meta["slug"].startswith(s) for s in args.solo):
            existentes = sorted((COLECCIONES / meta["slug"]).glob(f"{meta['prefijo']}_*.py"))
            if existentes:
                resumen.append((plantilla, [_entrada_desde_fichero(meta, r.stem, r) for r in existentes]))
            continue
        print(f"[{meta['slug']}] {meta['titulo']}")
        n = args.n or len(plantilla.VARIANTES)
        entradas = generar_coleccion(plantilla, n, args.semilla, args.forzar, args.ejecutar, indice)
        resumen.append((plantilla, entradas))
    if not args.sin_index:
        escribir_readme(resumen)
        actualizar_index(resumen)
        escribir_catalogo(resumen)
        print(f"README, catálogo e índice actualizados ({INDEX.relative_to(REPO)})")


if __name__ == "__main__":
    main()
