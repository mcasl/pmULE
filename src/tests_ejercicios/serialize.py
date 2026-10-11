"""
serialize.py — Serialización JSON estable de los objetos que aparecen en los
cuadernos (DataFrame, Series, ndarray, Styler, ProjectGraph, ...) y
comparación con tolerancia numérica.
"""
from __future__ import annotations

import hashlib
import math
import numbers

import numpy as np
import pandas as pd

MAX_STR = 20_000          # strings más largos se guardan como hash
MAX_SEQ = 2_000           # listas/dicts más largos se descartan
MAX_CELLS = 5_000         # DataFrame/Series/ndarray más grandes → resumen estadístico
FLOAT_TOL = 1e-6


# ---------------------------------------------------------------------------
# Serialización
# ---------------------------------------------------------------------------

class NotSerializable(Exception):
    pass


def _scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        if math.isnan(f):
            return {"__nan__": True}
        if math.isinf(f):
            return {"__inf__": 1 if f > 0 else -1}
        return f
    if isinstance(v, str):
        # LaTeX generado por handcalcs (@handcalc): es presentación y su
        # formato cambia entre versiones ($$, \\[10pt]…). No se fotografía;
        # los valores numéricos que calcula se capturan como variables aparte.
        t = v.strip()
        if t.startswith("$$"):
            t = t[2:].lstrip()
        if t.startswith("\\begin{aligned}"):
            raise NotSerializable("handcalcs latex")
        if len(v) > MAX_STR:
            return {"__hash__": hashlib.sha1(v.encode()).hexdigest(), "len": len(v)}
        return v
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return {"__ts__": str(v)}
    raise NotSerializable(type(v).__name__)


def _label(x):
    """Etiquetas de índice/columnas → str (para que 1 y '1' no diverjan)."""
    if isinstance(x, tuple):
        return "|".join(str(i) for i in x)
    return str(x)


def to_json(v, depth=0):
    """Convierte v a una estructura JSON-serializable o lanza NotSerializable."""
    if depth > 4:
        raise NotSerializable("depth")
    tname = type(v).__name__

    # Objetos de pmule --------------------------------------------------
    if tname == "ProjectGraph":
        G = v.pert_graph
        return {"__type__": "ProjectGraph",
                "nodes": [str(n) for n in v.nodes],
                "edges": sorted([str(a), str(b), str(d.get("activity", ""))]
                                for a, b, d in G.edges(data=True))}
    if tname == "PredecessorTable":
        return {"__type__": "PredecessorTable",
                "immediate": {k: sorted(s) for k, s in v.immediate_predecessor.items()}}
    if tname == "MonteCarloResult":
        return {"__type__": "MonteCarloResult", "n": v.n_samples,
                "tau_mean": v.tau_mean, "tau_std": v.tau_std,
                "q05": v.tau_q05, "q50": v.tau_q50, "q95": v.tau_q95,
                "sensitivity": {k: float(x) for k, x in v.sensitivity_mean.items()}}
    if tname == "PSPLIBInstance":
        return {"__type__": "PSPLIBInstance", "n_jobs": v.n_jobs, "horizon": v.horizon}
    if tname == "Styler":
        out = to_json(v.data, depth + 1)
        out["__type__"] = "Styler"
        return out

    # pandas / numpy -----------------------------------------------------
    if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)) and v.size > MAX_CELLS:
        return _summary(v)
    if isinstance(v, pd.DataFrame):
        return {"__type__": "DataFrame",
                "index": [_label(i) for i in v.index],
                "columns": [_label(c) for c in v.columns],
                "data": [[_scalar(x) if not isinstance(x, (list, tuple, dict, set, frozenset))
                          else to_json(x, depth + 1) for x in row]
                         for row in v.itertuples(index=False, name=None)]}
    if isinstance(v, pd.Series):
        return {"__type__": "Series", "name": _label(v.name) if v.name is not None else None,
                "index": [_label(i) for i in v.index],
                "data": [_scalar(x) if not isinstance(x, (list, tuple, dict, set, frozenset))
                         else to_json(x, depth + 1) for x in v.tolist()]}
    if isinstance(v, pd.Index):
        return {"__type__": "Index", "data": [_label(i) for i in v]}
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "OUS":
            return {"__type__": "ndarray", "shape": list(v.shape),
                    "data": [str(x) for x in v.ravel().tolist()]}
        return {"__type__": "ndarray", "shape": list(v.shape),
                "data": [_scalar(x) for x in v.ravel().tolist()]}

    # contenedores ---------------------------------------------------------
    if isinstance(v, dict):
        if len(v) > MAX_SEQ:
            raise NotSerializable("dict too long")
        return {"__type__": "dict", "items": [[_label(k), to_json(x, depth + 1)] for k, x in v.items()]}
    if isinstance(v, (list, tuple)):
        if len(v) > MAX_SEQ:
            raise NotSerializable("seq too long")
        return {"__type__": "list", "data": [to_json(x, depth + 1) for x in v]}
    if isinstance(v, (set, frozenset)):
        return {"__type__": "set", "data": sorted(str(x) for x in v)}

    # sympy y similares: por su representación
    mod = type(v).__module__ or ""
    if mod.startswith("sympy"):
        return {"__type__": "sympy", "repr": str(v)}

    return _scalar(v)


def _summary(v):
    """Resumen estadístico de objetos grandes (p. ej. muestras Monte Carlo):
    forma, estadísticos globales y media/desviación a lo largo del eje corto
    (por columna si hay ≤ 64 columnas, por fila si hay ≤ 64 filas)."""
    if isinstance(v, np.ndarray):
        v = pd.DataFrame(v.reshape(v.shape[0], -1) if v.ndim > 1 else v)
    if isinstance(v, pd.Series):
        v = v.to_frame()
    num = v.select_dtypes("number")
    arr = num.to_numpy(dtype=float)
    out = {"__type__": "summary", "shape": list(v.shape),
           "columns": [_label(c) for c in v.columns][:64]}
    if arr.size:
        flat = arr[np.isfinite(arr)]
        q = np.quantile(flat, [0.25, 0.5, 0.75]) if flat.size else [math.nan] * 3
        out["global"] = {"mean": _scalar(float(flat.mean())), "std": _scalar(float(flat.std(ddof=1)) if flat.size > 1 else 0.0),
                         "min": _scalar(float(flat.min())), "q25": _scalar(float(q[0])),
                         "q50": _scalar(float(q[1])), "q75": _scalar(float(q[2])),
                         "max": _scalar(float(flat.max())), "n": int(flat.size)}
        if arr.shape[1] <= 64:
            out["per_column"] = {_label(c): {"mean": _scalar(float(np.nanmean(arr[:, j]))),
                                             "std": _scalar(float(np.nanstd(arr[:, j], ddof=1)))}
                                 for j, c in enumerate(num.columns)}
        elif arr.shape[0] <= 64:
            out["per_row"] = {_label(i): {"mean": _scalar(float(np.nanmean(arr[k]))),
                                          "std": _scalar(float(np.nanstd(arr[k], ddof=1)))}
                              for k, i in enumerate(num.index)}
    return out


# ---------------------------------------------------------------------------
# Comparación
# ---------------------------------------------------------------------------

def _num(x):
    if isinstance(x, dict):
        if "__nan__" in x:
            return math.nan
        if "__inf__" in x:
            return math.inf * x["__inf__"]
    return x


def diff(golden, actual, path="", tol=FLOAT_TOL):
    """Lista de discrepancias (strings) entre dos estructuras JSON."""
    out = []
    g, a = _num(golden), _num(actual)
    if isinstance(g, float) or isinstance(a, float):
        if isinstance(g, (int, float)) and isinstance(a, (int, float)) and not isinstance(g, bool):
            if (math.isnan(g) and math.isnan(a)) or g == a:
                return out
            if math.isfinite(g) and math.isfinite(a) and abs(g - a) <= tol * max(1.0, abs(g)):
                return out
            return [f"{path}: {g!r} != {a!r}"]
    if type(g) != type(a):
        return [f"{path}: tipo {type(g).__name__} != {type(a).__name__} ({str(g)[:40]!r} vs {str(a)[:40]!r})"]
    if isinstance(g, dict):
        for k in sorted(set(g) | set(a)):
            if k not in g or k not in a:
                out.append(f"{path}.{k}: solo en {'golden' if k in g else 'actual'}")
            else:
                out += diff(g[k], a[k], f"{path}.{k}", tol)
        return out
    if isinstance(g, list):
        if len(g) != len(a):
            return [f"{path}: longitud {len(g)} != {len(a)}"]
        for i, (x, y) in enumerate(zip(g, a)):
            out += diff(x, y, f"{path}[{i}]", tol)
            if len(out) > 20:
                out.append(f"{path}: ... (más diferencias)")
                return out
        return out
    if g != a:
        return [f"{path}: {str(g)[:60]!r} != {str(a)[:60]!r}"]
    return out
