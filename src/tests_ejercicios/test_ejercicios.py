"""
test_ejercicios.py — Los cuadernos Ejercicio_NN como tests de regresión.

Cada cuaderno se ejecuta una vez por sesión (nb_runner) y cada variable
intermedia que el cuaderno calcula se compara con el golden congelado en
golden/ejercicio_NN.json. Un test por (cuaderno, celda, variable) para que el
fallo señale exactamente qué resultado ha cambiado.

Regenerar golden (tras un cambio deliberado de comportamiento):
    python -m tests_ejercicios.make_golden --all
"""
from __future__ import annotations

import functools
import json

import pytest

from tests_ejercicios import nb_runner
from tests_ejercicios import serialize as ser


def _load_golden():
    out = {}
    for p in sorted(nb_runner.GOLDEN_DIR.glob("ejercicio_*.json")):
        out[p.stem.split("_", 1)[1]] = json.loads(p.read_text(encoding="utf8"))
    return out


GOLDEN = _load_golden()


@functools.lru_cache(maxsize=None)
def _run(num: str) -> dict:
    return nb_runner.run_notebook(num, stop_on_error=False)


def _cases():
    for num, g in GOLDEN.items():
        for cell in g["cells"]:
            for name in cell["vars"]:
                yield pytest.param(num, cell["index"], name,
                                   id=f"Ej{num}-c{cell['index']}-{name}")


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num", sorted(GOLDEN), ids=lambda n: f"Ej{n}")
def test_notebook_runs_without_error(num):
    """El cuaderno se ejecuta de principio a fin (como cuando se renderiza)."""
    snap = _run(num)
    assert snap["error"] is None, (
        f"Ejercicio_{num} falla en la celda {snap['error']['cell']}: "
        f"{snap['error']['type']}: {snap['error']['msg']}\n--- celda ---\n{snap['error']['source']}")


@pytest.mark.parametrize("num,cell,name", list(_cases()))
def test_intermediate_result(num, cell, name):
    """La variable `name` tras la celda `cell` coincide con el golden."""
    snap = _run(num)
    actual_cells = {c["index"]: c["vars"] for c in snap["cells"]}
    golden_cells = {c["index"]: c["vars"] for c in GOLDEN[num]["cells"]}
    if cell not in actual_cells or name not in actual_cells[cell]:
        pytest.fail(f"Ejercicio_{num}: la celda {cell} ya no produce/cambia `{name}` "
                    f"(¿cambió el cuaderno? regenera el golden si es intencionado)")
    diffs = ser.diff(golden_cells[cell][name], actual_cells[cell][name], path=name)
    assert not diffs, f"Ejercicio_{num} celda {cell}:\n  " + "\n  ".join(diffs[:15])


def test_golden_covers_all_notebooks():
    """Cada cuaderno Ejercicio_NN.py tiene su golden (y viceversa)."""
    nbs = set(nb_runner.list_notebooks())
    assert nbs == set(GOLDEN), f"sin golden: {sorted(nbs - set(GOLDEN))}; golden huérfano: {sorted(set(GOLDEN) - nbs)}"
