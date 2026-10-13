"""
make_golden.py — Congela en golden/simplify.json el resultado de
PredecessorTable.create_project(simplify=True) para cada caso de cases.py.

    python -m tests_simplify.make_golden

Regenerar el golden es una decisión consciente: significa aceptar que el
comportamiento nuevo es el correcto. Revisa el diff de git del JSON.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pmule import PredecessorTable, ProjectGraph

from .cases import all_cases

GOLDEN = Path(__file__).resolve().parent / "golden" / "simplify.json"


def snapshot(project: ProjectGraph) -> dict:
    """Resumen comparable de un grafo PERT simplificado.

    Las predecesoras (distantes e inmediatas) son la semántica del proyecto y
    no dependen del orden en que se simplifiquen las ficticias; el número de
    ficticias/nudos y los arcos sí pueden depender de ese orden.
    """
    G = project.pert_graph
    return {
        "n_dummies": len(project.dummies),
        "n_nodes": G.number_of_nodes(),
        "distant_predecessor": {k: sorted(v) for k, v in
                                sorted(project.distant_predecessor(dummies=False).items())},
        "immediate_predecessor": {a: sorted(project.calculate_immediate_predecessor_of(a))
                                  for a in sorted(project.actual_activities)},
        "edges": sorted([s, t, d["activity"]] for s, t, d in G.edges(data=True)),
        "single_source_sink_acyclic": ProjectGraph._is_single_source_sink(project),
    }


def build(preds: dict) -> ProjectGraph:
    table = PredecessorTable.from_dict_of_sets({k: set(v) for k, v in preds.items()})
    return table.create_project(simplify=True)


def main():
    out = {}
    for name, preds in all_cases().items():
        t0 = time.time()
        snap = snapshot(build(preds))
        snap["predecessors"] = preds
        snap["seconds"] = round(time.time() - t0, 3)
        out[name] = snap
        print(f"  {name:32s} dummies={snap['n_dummies']:3d} nodes={snap['n_nodes']:3d} "
              f"{snap['seconds']:7.2f}s", flush=True)
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"-> {GOLDEN} ({len(out)} casos)")


if __name__ == "__main__":
    main()
