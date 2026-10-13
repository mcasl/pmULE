"""
test_simplify.py — Regresión de la simplificación de ficticias.

Compara create_project(simplify=True) con golden/simplify.json, generado con
la implementación anterior (_simplify_legacy). Se exige:

  * la misma semántica: predecesoras distantes e inmediatas idénticas al golden
    y a la tabla de partida;
  * un grafo válido: acíclico, con un único origen y un único sumidero, y con
    todas las actividades reales;
  * el MISMO grafo: _simplify_local reproduce las decisiones y el orden de
    _simplify_legacy, así que arcos, nombres de ficticias y numeración de
    nudos deben coincidir exactamente (los cuadernos y docs dependen de ello).

Regenerar golden (tras un cambio deliberado de comportamiento):
    python -m tests_simplify.make_golden
"""
from __future__ import annotations

import json

import pytest

from pmule import PredecessorTable, ProjectGraph, _simplify_legacy, _simplify_local
from tests_simplify.cases import all_cases, random_dag
from tests_simplify.make_golden import GOLDEN, build, snapshot

GOLD = json.loads(GOLDEN.read_text(encoding="utf8"))
CASES = all_cases()


def test_golden_covers_all_cases():
    assert set(GOLD) == set(CASES)


@pytest.fixture(scope="module")
def snaps():
    return {name: snapshot(build(preds)) for name, preds in CASES.items()}


@pytest.mark.parametrize("name", sorted(CASES))
def test_same_precedences_as_golden(name, snaps):
    assert snaps[name]["distant_predecessor"] == GOLD[name]["distant_predecessor"]
    assert snaps[name]["immediate_predecessor"] == GOLD[name]["immediate_predecessor"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_same_precedences_as_table(name, snaps):
    table = PredecessorTable.from_dict_of_sets({k: set(v) for k, v in CASES[name].items()})
    expected = {k: sorted(v) for k, v in sorted(table.distant_predecessor.items())}
    assert snaps[name]["distant_predecessor"] == expected


@pytest.mark.parametrize("name", sorted(CASES))
def test_valid_graph(name, snaps):
    assert snaps[name]["single_source_sink_acyclic"]
    assert set(snaps[name]["immediate_predecessor"]) == set(CASES[name])


@pytest.mark.parametrize("name", sorted(CASES))
def test_same_graph_as_golden(name, snaps):
    assert snaps[name]["n_dummies"] == GOLD[name]["n_dummies"]
    assert snaps[name]["n_nodes"] == GOLD[name]["n_nodes"]
    assert snaps[name]["edges"] == GOLD[name]["edges"]


def _edges(project):
    return sorted(project.pert_graph.edges(data="activity"))


@pytest.mark.parametrize("seed", range(30))
def test_identical_to_legacy(seed):
    """Sobre el mismo grafo de Fondahl sin simplificar, la versión nueva y la
    antigua producen exactamente el mismo grafo (antes del renumerado final)."""
    n, max_preds = [(12, 2), (18, 3), (25, 4)][seed % 3]
    preds = random_dag(n, max_preds, seed=500 + seed)
    table = PredecessorTable.from_dict_of_sets({k: set(v) for k, v in preds.items()})
    raw = table.create_project_fondahl(simplify=False)
    new, old = _simplify_local(raw), _simplify_legacy(raw)
    assert _edges(new) == _edges(old)
    assert list(new.pert_graph.nodes) == list(old.pert_graph.nodes)
    assert new.distant_predecessor() == table.distant_predecessor


def test_input_project_not_mutated():
    table = PredecessorTable.from_dict_of_sets({k: set(v) for k, v in CASES["k33"].items()})
    raw = table.create_project_fondahl(simplify=False)
    edges_before = sorted(raw.pert_graph.edges(data="activity"))
    _simplify_local(raw)
    assert sorted(raw.pert_graph.edges(data="activity")) == edges_before


def test_large_instance_is_fast():
    """500 actividades: la versión antigua tardaría horas; la nueva, < 10 s."""
    import time
    preds = random_dag(500, 3, seed=7)
    t0 = time.time()
    project = build(preds)
    assert time.time() - t0 < 10
    assert ProjectGraph._is_single_source_sink(project)
    assert len(project.actual_activities) == 500
