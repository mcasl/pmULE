"""
compare_outputs.py — Contrasta las salidas de los cuadernos .ipynb entre dos
versiones (p. ej. un commit antiguo y la copia de trabajo), celda a celda.

Sirve para validar que, tras cambiar la librería y re-ejecutar los cuadernos,
los resultados publicados no han cambiado salvo donde se esperaba.

    python -m tests_ejercicios.compare_outputs a33af95            # todos
    python -m tests_ejercicios.compare_outputs a33af95 03 04 06   # algunos
    python -m tests_ejercicios.compare_outputs a33af95 --verbose  # muestra diffs

Las celdas se emparejan por su código fuente (no por posición). Se comparan
las salidas text/plain y text/html (sin etiquetas), normalizando espacios y
números (p. ej. 18 == 18.0, redondeo a 4 decimales).
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import subprocess
import sys

from .nb_runner import NOTEBOOKS, REPO, list_notebooks

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")
_STYLE = re.compile(r"<style.*?</style>", re.S)
_ID = re.compile(r"T_[0-9a-f]{5,}")     # ids aleatorios de Styler


def _norm_num(m):
    s = m.group(0)
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f.is_integer() else f"{f:.4f}".rstrip("0").rstrip(".")


def normalize(text: str) -> str:
    text = _STYLE.sub("", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _ID.sub("T_x", text)
    text = _NUM.sub(_norm_num, text)
    return _WS.sub(" ", text).strip()


def cell_outputs(nb: dict, raw: bool = False) -> dict[str, str]:
    """{codigo_normalizado: salida_normalizada} para las celdas de código.
    raw=True → salida sin normalizar (HTML/texto crudo) para parsear tablas."""
    out = {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]).strip()
        if not src or src.startswith("%"):
            continue
        chunks = []
        for o in c.get("outputs", []):
            d = o.get("data", {})
            if "text/html" in d:
                chunks.append("".join(d["text/html"]))
            elif "text/plain" in d:
                chunks.append("".join(d["text/plain"]))
            elif "text/markdown" in d:
                chunks.append("".join(d["text/markdown"]))
            elif o.get("output_type") == "stream":
                chunks.append("".join(o.get("text", "")))
            elif o.get("output_type") == "error":
                chunks.append("ERROR: " + o.get("ename", ""))
        key = _WS.sub(" ", src)
        out[key] = "\n".join(chunks) if raw else normalize("\n".join(chunks))
    return out


# ---------------------------------------------------------------------------
# Clasificación de diferencias
# ---------------------------------------------------------------------------

NEW_COLUMNS = {"H_libre", "H_independiente"}   # añadidas a calendar() en 2026-10
_AT = re.compile(r"@\S+")


def _tables(raw_html: str):
    """Tablas HTML → lista de DataFrames normalizados (o None si no hay)."""
    import io
    import pandas as pd
    if "<table" not in raw_html:
        return None
    try:
        dfs = pd.read_html(io.StringIO(raw_html))
    except Exception:          # noqa: BLE001
        return None
    out = []
    for df in dfs:
        df = df.copy()
        # Cabeceras multinivel (fila con el nombre del índice): quedarse con el
        # primer nivel que no sea "Unnamed"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [next((str(l) for l in col if not str(l).startswith("Unnamed")), str(col[0]))
                          for col in df.columns]
        df.columns = [str(c) for c in df.columns]
        df = df.drop(columns=[c for c in df.columns if c in NEW_COLUMNS], errors="ignore")
        # primera columna = etiquetas (índice); quitar filas de ficticias
        first = df.columns[0]
        df[first] = df[first].astype(str)
        df = df[~df[first].str.startswith("@")].reset_index(drop=True)
        df = df.rename(columns={first: "idx"})
        out.append(df)
    return out


def _same_frames(a, b) -> bool:
    import pandas as pd
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            pd.testing.assert_frame_equal(x, y, check_dtype=False, check_like=True, atol=1e-6)
        except AssertionError:
            return False
    return True


def _multiset(text: str):
    return sorted(_NUM.findall(_AT.sub("", text)))


def classify(src: str, old_raw: str, new_raw: str, old_n: str, new_n: str) -> str:
    """Etiqueta la causa de una diferencia de salida."""
    # (a) solo cambian nombres de ficticias / nombre del índice
    strip = lambda t: _WS.sub(" ", _AT.sub("", t).replace("activity", "")).strip()
    if strip(old_n) == strip(new_n):
        return "ficticias"
    # (b) tablas: iguales tras quitar columnas nuevas y filas de ficticias
    ta, tb = _tables(old_raw), _tables(new_raw)
    if ta and tb and _same_frames(ta, tb):
        return "columnas_nuevas"
    # (c) tablas indexadas por nodo (Zaderenko, early/late de nodos): el
    #     conjunto de eventos depende de cómo se construya el grafo (Roy vs
    #     Fondahl), así que no son comparables entre versiones. Lo invariante
    #     son las tablas por actividad, que sí se comparan estrictamente.
    if "zaderenko" in src or "['nodes']" in src:
        return "nodos"
    if _multiset(old_n) == _multiset(new_n):
        return "reordenacion"
    return "REAL"


def load_old(rev: str, num: str) -> dict | None:
    rel = f"notebooks/Ejercicio_{num}.ipynb"
    try:
        raw = subprocess.check_output(["git", "-C", str(REPO), "show", f"{rev}:{rel}"],
                                      stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return json.loads(raw)


def load_new(num: str) -> dict:
    return json.loads((NOTEBOOKS / f"Ejercicio_{num}.ipynb").read_text(encoding="utf8"))


def compare(rev: str, num: str, verbose: bool) -> dict:
    old = load_old(rev, num)
    if old is None:
        return {"num": num, "status": "sin version antigua"}
    new = load_new(num)
    o, n = cell_outputs(old), cell_outputs(new)
    o_raw, n_raw = cell_outputs(old, raw=True), cell_outputs(new, raw=True)
    common = [k for k in n if k in o]
    same = [k for k in common if o[k] == n[k]]
    changed = [k for k in common if o[k] != n[k] and (o[k] or n[k])]
    res = {"num": num, "status": "ok", "celdas_comunes": len(common), "iguales": len(same),
           "distintas": len(changed), "solo_nuevas": len([k for k in n if k not in o]),
           "solo_antiguas": len([k for k in o if k not in n]), "detalle": [], "clases": {}}
    for k in changed:
        cls = classify(k, o_raw[k], n_raw[k], o[k], n[k])
        res["clases"][cls] = res["clases"].get(cls, 0) + 1
        if cls != "REAL" and not verbose:
            continue
        entry = {"codigo": k[:90], "antes": o[k][:160], "ahora": n[k][:160], "clase": cls}
        if verbose:
            entry["diff"] = "\n".join(difflib.unified_diff(
                o[k].split(" "), n[k].split(" "), lineterm="", n=2))[:1200]
        res["detalle"].append(entry)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rev", help="revisión git con los cuadernos antiguos")
    ap.add_argument("nums", nargs="*")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    total_changed = 0
    for num in args.nums or list_notebooks():
        r = compare(args.rev, num, args.verbose)
        if r["status"] != "ok":
            print(f"Ej{num}: {r['status']}")
            continue
        n_real = r["clases"].get("REAL", 0)
        flag = "!!" if n_real else "  "
        expl = ", ".join(f"{v} {k}" for k, v in sorted(r["clases"].items()))
        print(f"{flag} Ej{num}: {r['iguales']}/{r['celdas_comunes']} iguales"
              + (f"; {r['distintas']} distintas ({expl})" if r["distintas"] else "")
              + (f"; {r['solo_nuevas']} nuevas" if r["solo_nuevas"] else "")
              + (f"; {r['solo_antiguas']} desaparecidas" if r["solo_antiguas"] else ""))
        total_changed += n_real
        for d in r["detalle"]:
            print(f"      · [{d['clase']}] {d['codigo']}")
            print(f"        antes: {d['antes']}")
            print(f"        ahora: {d['ahora']}")
            if args.verbose and d.get("diff"):
                print("        " + d["diff"].replace("\n", "\n        "))
    return 1 if total_changed else 0


if __name__ == "__main__":
    sys.exit(main())
