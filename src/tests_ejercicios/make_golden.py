"""
make_golden.py — Genera (o actualiza) los ficheros golden/ejercicio_NN.json
ejecutando cada cuaderno con nb_runner.

    python -m tests_ejercicios.make_golden            # todos los que no existen
    python -m tests_ejercicios.make_golden --all      # regenera todos
    python -m tests_ejercicios.make_golden 03 04      # solo esos

Regenerar un golden es una decisión consciente: significa aceptar que el
comportamiento nuevo es el correcto. Revisa el diff de git del JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import nb_runner


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("nums", nargs="*", help="números de ejercicio (p.ej. 03 04)")
    ap.add_argument("--all", action="store_true", help="regenerar todos")
    args = ap.parse_args(argv)

    nums = args.nums or nb_runner.list_notebooks()
    nb_runner.GOLDEN_DIR.mkdir(exist_ok=True)
    failures = 0
    for num in nums:
        out = nb_runner.golden_path(num)
        if out.exists() and not args.all and not args.nums:
            print(f"  = {out.name} ya existe (usa --all para regenerar)")
            continue
        t0 = time.time()
        snap = nb_runner.run_notebook(num)
        n_vars = sum(len(c["vars"]) for c in snap["cells"])
        status = "OK " if snap["error"] is None else "ERR"
        print(f"  {status} Ejercicio_{num}: {len(snap['cells'])} celdas con cambios, "
              f"{n_vars} variables, {time.time() - t0:.1f}s"
              + (f"  ← {snap['error']['type']} en celda {snap['error']['cell']}: {snap['error']['msg'][:80]}"
                 if snap["error"] else ""))
        if snap["error"]:
            failures += 1
        out.write_text(json.dumps(snap, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
