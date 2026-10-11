"""
nb_runner.py — Ejecuta un cuaderno jupytext (notebooks/Ejercicio_NN.py) celda
a celda, con las salidas gráficas anuladas, y fotografía las variables que
cambian tras cada celda.

Los cuadernos son la "receta": no se reescribe su lógica, se ejecuta tal cual
(derivación de columnas, parámetros de cada llamada, etc.). Lo que se captura
son los resultados intermedios que el propio cuaderno calcula.

Uso programático:
    snap = run_notebook("03")          # dict serializable (ver serialize.py)
"""
from __future__ import annotations

import contextlib
import io
import json
import linecache
import os
import re
import sys
import types
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
REPO = SRC.parent
NOTEBOOKS = REPO / "notebooks"
GOLDEN_DIR = HERE / "golden"

SEED = 20261011   # semilla global previa a cada cuaderno (16 y 17 no fijan la suya)

# Nombres que nunca se fotografían (ruido de infraestructura)
SKIP_NAMES = {"display", "HTML", "Markdown", "Math", "Latex", "Image", "IFrame",
              "TexFragment", "TexDocument", "pd", "np", "nx", "sym", "scipy", "norm",
              "plt", "matplotlib", "os", "sys", "math", "handcalc", "Int64Dtype",
              "pinta_tikz", "plot_cargas", "plot_curvas_valor_ganado", "beautify",
              "colorea_minimo_fila", "PredecessorTable", "ProjectGraph", "EarnedValue"}


# ---------------------------------------------------------------------------
# Stubs de salida (TikZ, graphviz, matplotlib, IPython)
# ---------------------------------------------------------------------------

class _Silent:
    """Objeto que absorbe cualquier llamada/atributo y devuelve None o self."""
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return None
    def __getattr__(self, name): return _Silent()
    def __repr__(self): return "<stub>"


def _install_stubs():
    # jupyter_tikz → sin LaTeX
    jt = types.ModuleType("jupyter_tikz")
    jt.TexFragment = _Silent
    jt.TexDocument = _Silent
    sys.modules["jupyter_tikz"] = jt
    # handcalcs.render registra magics de IPython; aquí no hay IPython
    hr = types.ModuleType("handcalcs.render")
    sys.modules["handcalcs.render"] = hr
    # matplotlib sin ventana
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import pmule
    pmule.TexFragment = _Silent
    pmule.display = lambda *a, **k: None
    pmule.to_jup = lambda *a, **k: None
    pmule.pinta_tikz = lambda *a, **k: None
    pmule.plot_cargas = lambda *a, **k: None
    pmule.plot_curvas_valor_ganado = lambda *a, **k: None
    pmule.ProjectGraph.pert = lambda self, *a, **k: None     # graphviz
    pmule.ProjectGraph.roy = lambda self, *a, **k: None
    return pmule


# ---------------------------------------------------------------------------
# Lectura de celdas jupytext (formato percent)
# ---------------------------------------------------------------------------

_CELL_RE = re.compile(r"^# %%(.*)$")


def read_cells(path: Path) -> list[tuple[int, str]]:
    """Devuelve [(indice_celda, codigo)] solo para celdas de código no vacías."""
    cells, current, kind, idx = [], [], None, -1
    for line in path.read_text(encoding="utf8").splitlines():
        m = _CELL_RE.match(line)
        if m:
            if kind == "code":
                cells.append((idx, "\n".join(current)))
            current, idx = [], idx + 1
            kind = "markdown" if ("[markdown]" in m.group(1) or "[raw]" in m.group(1)) else "code"
            continue
        if kind == "code":
            current.append(line)
    if kind == "code":
        cells.append((idx, "\n".join(current)))
    return [(i, c) for i, c in cells if c.strip() and not all(
        l.strip().startswith("#") or not l.strip() for l in c.splitlines())]


# ---------------------------------------------------------------------------
# Ejecución con captura
# ---------------------------------------------------------------------------

def _snapshotable(name, value):
    if name.startswith("_") or name in SKIP_NAMES:
        return False
    if isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)):
        return False
    if isinstance(value, _Silent):
        return False
    return True


def run_notebook(num: str, stop_on_error: bool = True) -> dict:
    from . import serialize as ser
    path = NOTEBOOKS / f"Ejercicio_{num}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    pmule = _install_stubs()
    import numpy as np, random
    np.random.seed(SEED); random.seed(SEED)

    ns: dict = {"__name__": "__nb__", "__file__": str(path)}
    prev: dict[str, str] = {}
    result = {"notebook": f"Ejercicio_{num}", "cells": [], "error": None}

    cwd = os.getcwd()
    # Los cuadernos leen datos con rutas relativas (csv_files/…) y algunos
    # escriben ficheros (savefig, PDFs). Se ejecutan en un directorio temporal
    # con enlaces a los subdirectorios de datos, para no ensuciar notebooks/.
    import shutil, tempfile
    workdir = Path(tempfile.mkdtemp(prefix="pmule_nb_"))
    for entry in NOTEBOOKS.iterdir():
        if entry.is_dir() and not entry.name.startswith((".", "_", "scratch", "Ejercicio_")):
            (workdir / entry.name).symlink_to(entry, target_is_directory=True)
    os.chdir(workdir)
    # Algunos cuadernos hacen os.chdir a un directorio temporal del autor.
    # Si no existe (otra máquina), se redirige a un temporal local.
    _patch_chdir(ns)
    try:
        for idx, code in read_cells(path):
            buf = io.StringIO()
            fname = f"Ejercicio_{num}[c{idx}]"
            # inspect.getsource (handcalcs, dataclasses…) necesita el código en linecache
            linecache.cache[fname] = (len(code), None, code.splitlines(True), fname)
            try:
                with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(compile(code, fname, "exec"), ns)
            except Exception as exc:          # noqa: BLE001
                result["error"] = {"cell": idx, "type": type(exc).__name__,
                                   "msg": str(exc)[:300], "source": code[:300]}
                if stop_on_error:
                    break
                continue
            changed = {}
            for name, value in list(ns.items()):
                if not _snapshotable(name, value):
                    continue
                try:
                    js = ser.to_json(value)
                except (ser.NotSerializable, Exception):   # noqa: BLE001
                    continue
                key = json.dumps(js, sort_keys=True, default=str)
                if prev.get(name) != key:
                    prev[name] = key
                    changed[name] = js
            if changed:
                result["cells"].append({"index": idx,
                                        "source": code.strip().splitlines()[0][:100],
                                        "vars": changed})
    finally:
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def _patch_chdir(ns):
    """os.chdir tolerante: si la ruta no existe, usa un temporal y no falla."""
    import tempfile
    real_chdir = os.chdir
    fallback = Path(tempfile.gettempdir()) / "pmule_nb_tmp"
    fallback.mkdir(exist_ok=True)

    def chdir(p):
        p = os.path.expanduser(str(p))
        real_chdir(p if os.path.isdir(p) else fallback)

    import types as _t
    os_proxy = _t.ModuleType("os")
    os_proxy.__dict__.update(os.__dict__)
    os_proxy.chdir = chdir
    ns["__os_proxy__"] = os_proxy
    # El cuaderno hace `import os` en su primera celda: interceptamos ese nombre
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "os":
            return os_proxy
        return real_import(name, *a, **k)
    ns["__builtins__"] = dict(vars(builtins), __import__=fake_import)


def list_notebooks() -> list[str]:
    return sorted(p.stem.split("_", 1)[1] for p in NOTEBOOKS.glob("Ejercicio_*.py"))


def golden_path(num: str) -> Path:
    return GOLDEN_DIR / f"ejercicio_{num}.json"
