"""
cases.py — Catálogo de tablas de precedencias para probar la simplificación
de ficticias (create_project(simplify=True)).

Cada caso es {actividad: [predecesoras inmediatas]}. Incluye casos límite
escritos a mano y familias aleatorias deterministas (semilla fija), de modo
que el catálogo es reproducible y el golden no depende del azar.
"""
from __future__ import annotations

import random

HANDMADE = {
    # --- triviales -----------------------------------------------------------
    "single_activity": {"A": []},
    "two_independent": {"A": [], "B": []},
    "five_independent": {k: [] for k in "ABCDE"},
    "chain_four": {"A": [], "B": ["A"], "C": ["B"], "D": ["C"]},

    # --- abanicos ------------------------------------------------------------
    "fan_out": {"A": [], "B": ["A"], "C": ["A"], "D": ["A"]},
    "fan_in": {"A": [], "B": [], "C": [], "D": ["A", "B", "C"]},
    "fan_out_fan_in": {"A": [], "B": ["A"], "C": ["A"], "D": ["A"],
                       "E": ["B", "C", "D"]},

    # --- clásicos AoA --------------------------------------------------------
    "diamond": {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]},
    "n_network": {"A": [], "B": [], "C": ["A"], "D": ["A", "B"]},
    "w_network": {"A": [], "B": [], "C": [], "D": ["A", "B"], "E": ["B", "C"]},
    "classic_fondahl_six": {"A": [], "B": [], "C": ["A"], "D": ["A", "B"],
                            "E": ["B"], "F": ["C", "D", "E"]},
    "textbook_ten": {"A": [], "B": [], "C": ["A"], "D": ["A", "B"], "E": ["B"],
                     "F": ["C"], "G": ["C", "D"], "H": ["D", "E"],
                     "I": ["F", "G"], "J": ["G", "H"]},

    # --- conjuntos de predecesoras idénticos (bug histórico del ciclo) -------
    "shared_predecessor_set": {"A": [], "B": [], "C": ["A", "B"], "D": ["A", "B"]},
    "shared_preds_with_sink": {"A": [], "B": [], "C": ["A", "B"], "D": ["A", "B"],
                               "E": ["C", "D"]},
    "shared_single_pred_three": {"A": [], "B": ["A"], "C": ["A"], "D": ["A"],
                                 "E": ["B"], "F": ["B"]},

    # --- varias actividades finales desde el mismo nodo (arcos paralelos) ----
    "parallel_ends": {"A": [], "B": ["A"], "C": ["A"]},
    "parallel_starts_and_ends": {"A": [], "B": [], "C": ["A", "B"],
                                 "D": ["C"], "E": ["C"]},
    "parallel_middle": {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"],
                        "E": ["D"]},

    # --- redundancia transitiva en la entrada (la tabla la reduce) -----------
    "transitive_redundant_input": {"A": [], "B": ["A"], "C": ["A", "B"],
                                   "D": ["A", "B", "C"]},

    # --- bipartitos completos y escaleras ------------------------------------
    "k33": {"A": [], "B": [], "C": [],
            "D": ["A", "B", "C"], "E": ["A", "B", "C"], "F": ["A", "B", "C"]},
    "ladder": {"A1": [], "B1": [], "A2": ["A1"], "B2": ["A1", "B1"],
               "A3": ["A2", "B2"], "B3": ["B2"], "A4": ["A3"], "B4": ["A3", "B3"]},
    "zigzag": {"A": [], "B": [], "C": ["A"], "D": ["A", "B"], "E": ["C", "D"],
               "F": ["D"], "G": ["E", "F"], "H": ["F"]},

    # --- PREDS03 de test_regressions (ejercicio con varias convergencias) ----
    "crossing_merges": {"A": [], "B": [], "C": [], "D": ["A"], "E": ["A", "B"],
                        "F": ["B", "C"], "G": ["C"], "H": ["D", "E"],
                        "I": ["E", "F"], "J": ["F", "G"], "K": ["H", "I", "J"]},
}


def random_dag(n: int, max_preds: int, seed: int) -> dict:
    """DAG aleatorio: la actividad i elige 0..max_preds predecesoras entre las anteriores."""
    rng = random.Random(seed)
    return {f"A{i:03d}": [f"A{j:03d}" for j in rng.sample(range(i), min(i, rng.randint(0, max_preds)))]
            for i in range(n)}


def layered_dag(layers: int, width: int, density: float, seed: int) -> dict:
    """DAG por niveles: cada actividad depende de un subconjunto (no vacío) del nivel anterior."""
    rng = random.Random(seed)
    result, prev = {}, []
    for lv in range(layers):
        cur = [f"L{lv}_{k}" for k in range(width)]
        for name in cur:
            preds = [p for p in prev if rng.random() < density]
            if prev and not preds:
                preds = [rng.choice(prev)]
            result[name] = preds
        prev = cur
    return result


def all_cases() -> dict:
    cases = dict(HANDMADE)
    for n, max_preds, seeds in [(10, 2, range(5)), (20, 3, range(5)),
                                (30, 3, range(4)), (40, 4, range(3)), (50, 3, range(2))]:
        for s in seeds:
            cases[f"random_n{n}_p{max_preds}_s{s}"] = random_dag(n, max_preds, seed=1000 * n + s)
    for layers, width, density, s in [(3, 3, 1.0, 0), (4, 4, 0.5, 1), (5, 4, 0.4, 2),
                                      (6, 5, 0.3, 3), (4, 6, 0.6, 4)]:
        cases[f"layered_{layers}x{width}_d{density}_s{s}"] = layered_dag(layers, width, density, seed=s)
    return cases
