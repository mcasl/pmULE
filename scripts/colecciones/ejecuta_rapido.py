"""
ejecuta_rapido.py — ejecuta un cuaderno jupytext (py:percent) celda a celda SIN LaTeX ni
graphviz (los mismos stubs que los tests de regresión) y muestra en texto lo que cada celda
produce: los Markdown, las tablas (pandas) y las salidas impresas. Sirve para comprobar en
segundos que una plantilla o un cuaderno editado no falla y que sus frases de conclusión
dicen lo que deben, sin esperar al render de Quarto.

Uso (desde la raíz del repositorio):
    .venv/bin/python -m scripts.colecciones.ejecuta_rapido notebooks/colecciones/05_holguras/Holguras_01.py
    .venv/bin/python -m scripts.colecciones.ejecuta_rapido --solo-errores notebooks/colecciones/05_holguras/*.py
    .venv/bin/python -m scripts.colecciones.ejecuta_rapido --filas 12 notebooks/Ejercicio_05.py

Opciones: --solo-errores (no muestra salidas; resume errores), --filas N (filas máximas por tabla, 20).
Código de salida 1 si algún cuaderno falla.
"""
from __future__ import annotations

import ast
import contextlib
import io
import linecache
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tests_ejercicios.nb_runner import _install_stubs, read_cells, _patch_chdir, NOTEBOOKS, SEED   # noqa: E402


def _texto(obj, filas: int) -> str | None:
    """Representación en texto de lo que Jupyter mostraría."""
    import pandas as pd
    if obj is None:
        return None
    if hasattr(obj, "data") and isinstance(getattr(obj, "data"), str):      # Markdown, HTML, Math
        return obj.data
    if type(obj).__name__ == "Styler" and hasattr(obj, "data"):
        obj = obj.data
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        with pd.option_context("display.max_colwidth", None, "display.width", 250, "display.max_rows", filas):
            return obj.to_string(max_rows=filas) if isinstance(obj, pd.DataFrame) else obj.to_string(max_rows=filas)
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "_repr_html_") or hasattr(obj, "_repr_svg_"):
        return f"<{type(obj).__name__}>"
    r = repr(obj)
    return r if len(r) < 400 else r[:400] + " …"


def ejecutar(ruta: Path, filas: int = 20, mostrar: bool = True) -> dict | None:
    """Ejecuta el cuaderno; devuelve None si va bien o el dict del error."""
    pmule = _install_stubs()
    import numpy as np, random
    np.random.seed(SEED); random.seed(SEED)
    salidas = []

    def display(*objs, **k):
        for o in objs:
            t = _texto(o, filas)
            if t is not None:
                salidas.append(t)

    ns: dict = {"__name__": "__nb__", "__file__": str(ruta)}
    cwd = os.getcwd()
    workdir = Path(tempfile.mkdtemp(prefix="pmule_er_"))
    for entry in NOTEBOOKS.iterdir():
        if entry.is_dir() and not entry.name.startswith((".", "_", "scratch", "Ejercicio_")):
            (workdir / entry.name).symlink_to(entry, target_is_directory=True)
    os.chdir(workdir)
    _patch_chdir(ns)
    error = None
    try:
        for idx, code in read_cells(ruta):
            salidas.clear()
            fname = f"{ruta.stem}[c{idx}]"
            lineas = [l for l in code.splitlines() if not l.lstrip().startswith(("%", "!"))]
            fuente = "\n".join(lineas)
            linecache.cache[fname] = (len(fuente), None, fuente.splitlines(True), fname)
            buf = io.StringIO()
            try:
                arbol = ast.parse(fuente, fname)
                ultimo = arbol.body[-1] if arbol.body and isinstance(arbol.body[-1], ast.Expr) else None
                cuerpo = ast.Module(body=arbol.body[:-1] if ultimo else arbol.body, type_ignores=[])
                with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ns["display"] = display                      # el cuaderno lo reimporta; se vuelve a poner
                    exec(compile(cuerpo, fname, "exec"), ns)
                    ns["display"] = display
                    if ultimo is not None:
                        valor = eval(compile(ast.Expression(ultimo.value), fname, "eval"), ns)
                        display(valor)
            except Exception as exc:                              # noqa: BLE001
                import traceback
                error = {"celda": idx, "tipo": type(exc).__name__, "msg": str(exc)[:500],
                         "fuente": code.strip()[:400], "traza": traceback.format_exc()[-1500:]}
                break
            if mostrar:
                primera = code.strip().splitlines()[0][:90]
                impreso = buf.getvalue().strip()
                if impreso or salidas:
                    print(f"\n── [c{idx}] {primera}")
                    if impreso:
                        print(impreso)
                    for s in salidas:
                        print(s)
    finally:
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)
    return error


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    solo_errores = "--solo-errores" in args
    filas = 20
    if "--filas" in args:
        i = args.index("--filas")
        filas = int(args[i + 1])
        del args[i:i + 2]
    rutas = [Path(a).resolve() for a in args if not a.startswith("--")]
    fallos = 0
    for r in rutas:
        if not solo_errores:
            print(f"\n════════ {r} ════════")
        err = ejecutar(r, filas=filas, mostrar=not solo_errores)
        if err:
            fallos += 1
            print(f"\nERROR en {r} (celda {err['celda']}): {err['tipo']}: {err['msg']}")
            print("  fuente:", err["fuente"].replace("\n", "\n          "))
            print("  traza:", err["traza"].replace("\n", "\n         "))
        elif solo_errores:
            print(f"OK {r}")
    print(f"\n{len(rutas) - fallos} de {len(rutas)} cuaderno(s) sin errores")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
