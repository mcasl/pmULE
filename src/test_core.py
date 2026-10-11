"""
test_core.py — Tests de caracterización del núcleo clásico de pmULE
====================================================================
Cubre: calculate_pert, calendar, duration, critical_path, zaderenko,
       paths/path_matrix, gantt, gantt_cargas, ackoff, standard_deviation,
       incidence_matrix, nullspace, PredecessorTable, helpers de formato.

Todos los valores esperados están calculados a mano sobre el proyecto
de 8 actividades definido en PREDS/DUR (camino crítico A-D-G-H = 18).
"""

import os
import sys
import types
import tempfile
import textwrap
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import pmule
from pmule import (
    ProjectGraph, PredecessorTable,
    truncate_and_pad, significant_values,
    highlight_maximum, highlight_minimum, highlight_first_minimum,
    read_rcp, make_Roy, SVD,
)

# ---------------------------------------------------------------------------
# Proyecto de referencia
# ---------------------------------------------------------------------------

PREDS = {
    "A": set(), "B": set(),
    "C": {"A"}, "D": {"A", "B"}, "E": {"B"},
    "F": {"C", "D"}, "G": {"D", "E"}, "H": {"F", "G"},
}
DUR = {"A": 3, "B": 2, "C": 4, "D": 5, "E": 3, "F": 2, "G": 6, "H": 4}
RES = {"A": 2, "B": 1, "C": 3, "D": 2, "E": 1, "F": 2, "G": 3, "H": 1}

# Valores esperados (a mano)
ES = {"A": 0, "B": 0, "C": 3, "D": 3, "E": 2, "F": 8, "G": 8, "H": 14}
LS = {"A": 0, "B": 1, "C": 8, "D": 3, "E": 5, "F": 12, "G": 8, "H": 14}
EF = {k: ES[k] + DUR[k] for k in DUR}
LF = {k: LS[k] + DUR[k] for k in DUR}
H_TOTAL = {"A": 0, "B": 1, "C": 5, "D": 0, "E": 3, "F": 4, "G": 0, "H": 0}
PROJECT_DURATION = 18
CRITICAL = ["A", "D", "G", "H"]
PATHS_NO_DUMMIES = [
    ["A", "C", "F", "H"], ["A", "D", "F", "H"], ["A", "D", "G", "H"],
    ["B", "D", "F", "H"], ["B", "D", "G", "H"], ["B", "E", "G", "H"],
]
LOADS = [3, 3, 3, 6, 6, 5, 5, 2, 5, 5, 3, 3, 3, 3, 1, 1, 1, 1]
SUM_SQUARES = 243


def project():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


def data_df():
    return pd.DataFrame({"duration": DUR, "resources": RES})


# ---------------------------------------------------------------------------
# calculate_pert / duration
# ---------------------------------------------------------------------------

class TestCalculatePert(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.res = self.pg.calculate_pert(DUR)

    def test_returns_dict_with_nodes_and_activities(self):
        self.assertIn("nodes", self.res)
        self.assertIn("activities", self.res)

    def test_source_early_zero_and_sink_early_is_duration(self):
        nodes = self.res["nodes"]
        self.assertEqual(nodes["early"].iloc[0], 0)
        self.assertEqual(nodes["early"].max(), PROJECT_DURATION)

    def test_sink_late_equals_early(self):
        nodes = self.res["nodes"]
        sink = self.pg.nodes[-1]
        self.assertEqual(nodes.loc[sink, "early"], nodes.loc[sink, "late"])

    def test_late_never_before_early(self):
        nodes = self.res["nodes"]
        self.assertTrue((nodes["late"] >= nodes["early"]).all())

    def test_total_floats_match_hand_calculation(self):
        H = self.res["activities"]["H_total"]
        for act, expected in H_TOTAL.items():
            self.assertEqual(H[act], expected, msg=act)

    def test_dummies_have_zero_or_positive_float(self):
        H = self.res["activities"]["H_total"]
        for d in self.pg.dummies:
            self.assertGreaterEqual(H[d], 0)

    def test_missing_activities_default_to_zero(self):
        res = self.pg.calculate_pert({"A": 3})
        self.assertEqual(res["nodes"]["early"].max(), 3)

    def test_duration_method(self):
        self.assertEqual(self.pg.duration(DUR), PROJECT_DURATION)

    def test_duration_matches_tau(self):
        self.assertAlmostEqual(self.pg.duration(DUR), self.pg.tau(DUR))


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

class TestCalendar(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.cal = self.pg.calendar(pd.Series(DUR))

    def test_index_is_activities(self):
        self.assertEqual(sorted(self.cal.index), sorted(DUR))

    def test_columns(self):
        for col in ("inicio_mas_temprano", "inicio_mas_tardio",
                    "fin_mas_temprano", "fin_mas_tardio", "H_total", "duracion"):
            self.assertIn(col, self.cal.columns)

    def test_early_start(self):
        for a, v in ES.items():
            self.assertEqual(self.cal.loc[a, "inicio_mas_temprano"], v, a)

    def test_late_start(self):
        for a, v in LS.items():
            self.assertEqual(self.cal.loc[a, "inicio_mas_tardio"], v, a)

    def test_early_finish(self):
        for a, v in EF.items():
            self.assertEqual(self.cal.loc[a, "fin_mas_temprano"], v, a)

    def test_late_finish(self):
        for a, v in LF.items():
            self.assertEqual(self.cal.loc[a, "fin_mas_tardio"], v, a)

    def test_total_float(self):
        for a, v in H_TOTAL.items():
            self.assertEqual(self.cal.loc[a, "H_total"], v, a)

    def test_float_equals_ls_minus_es(self):
        diff = self.cal["inicio_mas_tardio"] - self.cal["inicio_mas_temprano"]
        pd.testing.assert_series_equal(
            diff.astype(float), self.cal["H_total"].astype(float), check_names=False)


# ---------------------------------------------------------------------------
# paths / path_matrix / critical_path
# ---------------------------------------------------------------------------

class TestPathsAndCriticalPath(unittest.TestCase):

    def setUp(self):
        self.pg = project()

    def test_paths_without_dummies(self):
        paths = self.pg.paths(dummies=False)
        self.assertEqual(sorted(paths.values()), PATHS_NO_DUMMIES)

    def test_paths_keys_are_route_n(self):
        for k in self.pg.paths():
            self.assertRegex(k, r"^Route_\d+$")

    def test_paths_with_dummies_contain_at_names(self):
        paths = self.pg.paths(dummies=True)
        all_acts = set().union(*paths.values())
        self.assertTrue(any(a.startswith("@") for a in all_acts))

    def test_path_matrix_shape(self):
        R = self.pg.path_matrix(dummies=False)
        self.assertEqual(R.shape, (6, 8))

    def test_path_matrix_binary(self):
        R = self.pg.path_matrix(dummies=False)
        self.assertTrue(((R == 0) | (R == 1)).all().all())

    def test_path_matrix_row_sums_equal_path_length(self):
        R = self.pg.path_matrix(dummies=False)
        self.assertTrue((R.sum(axis=1) == 4).all())

    def test_path_matrix_SVD_returns_three_arrays(self):
        U, S, Vt = self.pg.path_matrix_SVD(dummies=False)
        self.assertEqual(len(S), 6)

    def test_critical_path_exact(self):
        cp = self.pg.critical_path(DUR)
        self.assertEqual(len(cp), 1)
        self.assertEqual(list(cp.values())[0], CRITICAL)

    def test_critical_path_two_paths_when_tied(self):
        dur = dict(DUR); dur["B"] = 3   # B-D-G-H = 18 también
        cp = self.pg.critical_path(dur)
        self.assertEqual(len(cp), 2)

    def test_critical_activities_have_zero_float(self):
        H = self.pg.calculate_pert(DUR)["activities"]["H_total"]
        for a in CRITICAL:
            self.assertEqual(H[a], 0)


# ---------------------------------------------------------------------------
# zaderenko
# ---------------------------------------------------------------------------

class TestZaderenko(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.z = self.pg.zaderenko(DUR)

    def test_shape(self):
        n = len(self.pg.nodes)
        self.assertEqual(self.z.shape, (n + 1, n + 1))

    def test_has_early_column_and_late_row(self):
        self.assertIn("early", self.z.columns)
        self.assertIn("late", self.z.index)

    def test_durations_in_cells(self):
        n2e = self.pg.nodes_from_edges
        for act, d in DUR.items():
            s, t = n2e[act]
            self.assertEqual(float(self.z.loc[s, t]), d, act)

    def test_sink_early_is_project_duration(self):
        sink = self.pg.nodes[-1]
        self.assertEqual(float(self.z.loc[sink, "early"]), PROJECT_DURATION)

    def test_late_row_sink(self):
        sink = self.pg.nodes[-1]
        self.assertEqual(float(self.z.loc["late", sink]), PROJECT_DURATION)


# ---------------------------------------------------------------------------
# gantt / gantt_cargas
# ---------------------------------------------------------------------------

class TestGantt(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.data = data_df()

    def test_returns_styler_and_drawing(self):
        styler, dibujo = self.pg.gantt(self.data, "duration", "resources")
        self.assertTrue(hasattr(styler, "data"))
        self.assertIsInstance(dibujo, str)

    def test_periods_span_project_duration(self):
        styler, _ = self.pg.gantt(self.data, "duration", "resources")
        period_cols = [c for c in styler.data.columns if isinstance(c, (int, np.integer))]
        self.assertEqual(len(period_cols), PROJECT_DURATION)

    def test_activity_occupies_its_early_window(self):
        styler, _ = self.pg.gantt(self.data, "duration", "resources")
        g = styler.data
        for a in DUR:
            occupied = [p for p in range(1, PROJECT_DURATION + 1) if g.loc[a, p] != ""]
            self.assertEqual(occupied, list(range(ES[a] + 1, EF[a] + 1)), a)

    def test_loads_with_total_fila(self):
        styler, _ = self.pg.gantt(self.data, "duration", "resources", total="fila")
        g = styler.data
        self.assertIn("Total", g.index)
        got = [int(g.loc["Total", p]) for p in range(1, PROJECT_DURATION + 1)]
        self.assertEqual(got, LOADS)

    def test_cuadrados_row(self):
        styler, _ = self.pg.gantt(self.data, "duration", "resources",
                                  total="fila", cuadrados=True)
        g = styler.data
        self.assertIn("Cuadrados", g.index)
        got = [int(g.loc["Cuadrados", p]) for p in range(1, PROJECT_DURATION + 1)]
        self.assertEqual(got, [x * x for x in LOADS])

    def test_holguras_column(self):
        styler, _ = self.pg.gantt(self.data, "duration", "resources", holguras=True)
        g = styler.data
        self.assertIn("H_total", g.columns)
        for a, h in H_TOTAL.items():
            self.assertEqual(float(g.loc[a, "H_total"]), h, a)

    def test_no_resource_label(self):
        styler, _ = self.pg.gantt(self.data, "duration")
        self.assertEqual(len(styler.data), 8)

    def test_resource_label_names(self):
        styler, _ = self.pg.gantt(self.data, "duration", resource_label="names")
        g = styler.data
        self.assertEqual(g.loc["A", 1], "A")

    def test_tikz_returns_tikz_code(self):
        _, dibujo = self.pg.gantt(self.data, "duration", "resources", tikz=True)
        self.assertIn("tikzpicture", dibujo)

    def test_gantt_cargas_sum_of_squares(self):
        styler, _ = self.pg.gantt_cargas(self.data, "duration", "resources", report=False)
        self.assertEqual(float(styler.data.loc["Cuadrados", "H_total"]), SUM_SQUARES)


# ---------------------------------------------------------------------------
# ackoff
# ---------------------------------------------------------------------------

class TestAckoff(unittest.TestCase):

    MIN = {"A": 1, "B": 1, "C": 2, "D": 3, "E": 1, "F": 1, "G": 4, "H": 2}
    COST = {"A": 10, "B": 8, "C": 6, "D": 12, "E": 5, "F": 7, "G": 9, "H": 11}

    def setUp(self):
        self.pg = project()

    def test_three_steps_characterization(self):
        styler, best, dur, avail = self.pg.ackoff(DUR, self.MIN, self.COST, reduction=3)
        self.assertEqual(best, {0: ["G"], 1: ["G"], 2: ["A"]})
        self.assertEqual(dur["G"], 4)
        self.assertEqual(dur["A"], 2)
        self.assertEqual(avail["G"], 0)

    def test_each_step_reduces_duration_by_one(self):
        _, best, dur, _ = self.pg.ackoff(DUR, self.MIN, self.COST, reduction=3)
        self.assertEqual(self.pg.duration(dur.to_dict()), PROJECT_DURATION - 3)

    def test_zero_reduction_returns_unchanged(self):
        _, best, dur, _ = self.pg.ackoff(DUR, self.MIN, self.COST, reduction=0)
        self.assertEqual(best, {})
        self.assertEqual(dur.to_dict(), DUR)

    def test_accepts_series(self):
        _, best, _, _ = self.pg.ackoff(pd.Series(DUR), pd.Series(self.MIN),
                                       pd.Series(self.COST), reduction=1)
        self.assertEqual(best, {0: ["G"]})

    def test_stops_when_nothing_reducible(self):
        mind = dict(DUR)  # nada reducible
        _, best, dur, _ = self.pg.ackoff(DUR, mind, self.COST, reduction=5)
        self.assertEqual(best, {})

    def test_inputs_not_mutated(self):
        d = dict(DUR)
        self.pg.ackoff(d, self.MIN, self.COST, reduction=2)
        self.assertEqual(d, DUR)

    def test_report_prints(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.pg.ackoff(DUR, self.MIN, self.COST, reduction=1, report=True)
        self.assertIn("Best option", buf.getvalue())


# ---------------------------------------------------------------------------
# standard_deviation
# ---------------------------------------------------------------------------

class TestStandardDeviation(unittest.TestCase):

    def setUp(self):
        self.pg = project()

    def test_unit_variances(self):
        std, table = self.pg.standard_deviation(DUR, {k: 1 for k in DUR})
        self.assertAlmostEqual(std, 2.0)   # 4 actividades críticas con var 1

    def test_only_critical_variances_count(self):
        var = {k: 0 for k in DUR}; var["C"] = 100   # C no es crítica
        std, _ = self.pg.standard_deviation(DUR, var)
        self.assertAlmostEqual(std, 0.0)

    def test_critical_variance(self):
        var = {"A": 1, "D": 4, "G": 9, "H": 2}
        std, _ = self.pg.standard_deviation(DUR, var)
        self.assertAlmostEqual(std, 16 ** 0.5)

    def test_report_prints(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.pg.standard_deviation(DUR, {k: 1 for k in DUR}, report=True)
        self.assertIn("standard deviation", buf.getvalue())


# ---------------------------------------------------------------------------
# incidence_matrix / nullspace
# ---------------------------------------------------------------------------

class TestIncidence(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.H = self.pg.incidence_matrix()

    def test_shape(self):
        n_edges = self.pg.pert_graph.number_of_edges()
        n_nodes = self.pg.pert_graph.number_of_nodes()
        self.assertEqual(self.H.shape, (n_edges, n_nodes))

    def test_each_row_has_one_minus_one_and_one_plus_one(self):
        self.assertTrue((self.H.sum(axis=1) == 0).all())
        self.assertTrue(((self.H == 1).sum(axis=1) == 1).all())
        self.assertTrue(((self.H == -1).sum(axis=1) == 1).all())

    def test_rows_are_all_activities(self):
        self.assertEqual(sorted(self.H.index), self.pg.activities)

    def test_orientation_tail_minus_head_plus(self):
        n2e = self.pg.nodes_from_edges
        s, t = n2e["A"]
        self.assertEqual(self.H.loc["A", s], -1)
        self.assertEqual(self.H.loc["A", t], 1)

    def test_nullspace_drops_two_columns(self):
        N = self.pg.nullspace()
        self.assertEqual(N.shape[1], self.H.shape[1] - 2)

    def test_nullspace_drops_source_and_sink(self):
        N = self.pg.nullspace()
        self.assertNotIn(self.pg.nodes[0], N.columns)
        self.assertNotIn(self.pg.nodes[-1], N.columns)


# ---------------------------------------------------------------------------
# PredecessorTable
# ---------------------------------------------------------------------------

class TestPredecessorTable(unittest.TestCase):

    def test_from_dict_of_strings_parsing(self):
        pt = PredecessorTable.from_dict_of_strings(
            {"A": "", "B": "A", "C": "A, B", "D": "B;C", "E": "---"})
        self.assertEqual(pt.immediate_predecessor["C"], {"B"})   # simplificado
        self.assertEqual(pt.immediate_predecessor["D"], {"C"})
        self.assertEqual(pt.immediate_predecessor["E"], set())

    def test_from_dict_of_sets_adds_undeclared_predecessors(self):
        pt = PredecessorTable.from_dict_of_sets({"B": {"A"}})
        self.assertIn("A", pt.immediate_predecessor)
        self.assertEqual(pt.immediate_predecessor["A"], set())

    def test_simplify_removes_transitive(self):
        pt = PredecessorTable.from_dict_of_sets({"A": set(), "B": {"A"}, "C": {"A", "B"}})
        self.assertEqual(pt.immediate_predecessor["C"], {"B"})

    def test_no_simplify_keeps_transitive(self):
        pt = PredecessorTable.from_dict_of_sets(
            {"A": set(), "B": {"A"}, "C": {"A", "B"}}, simplify=False)
        self.assertEqual(pt.immediate_predecessor["C"], {"A", "B"})

    def test_distant_predecessor(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        self.assertEqual(pt.distant_predecessor["H"], set("ABCDEFG"))
        self.assertEqual(pt.distant_predecessor["D"], {"A", "B"})

    def test_start_and_end_activities(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        self.assertEqual(pt.start_activities, {"A", "B"})
        self.assertEqual(pt.end_activities, {"H"})

    def test_activity_names(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        self.assertEqual(pt.activity_names, sorted(PREDS))

    def test_augmented_predecessor_has_dummies(self):
        pt = PredecessorTable.from_dict_of_sets({"A": set(), "B": {"A"}})
        aug = pt.augmented_predecessor
        self.assertTrue(any(k.startswith("@") for k in aug))
        self.assertTrue(len(pt.dummies) > 0)

    def test_nodes_property(self):
        pt = PredecessorTable.from_dict_of_sets({"A": set(), "B": {"A"}})
        self.assertIn("@∇StartOfProject", pt.nodes)
        self.assertIn("ΔEndOfProject", pt.nodes)
        self.assertIn("ΔA", pt.nodes)

    def test_from_dataframe_of_strings(self):
        df = pd.DataFrame({"act": ["A", "B", "C"], "pred": ["", "A", "A,B"]})
        pt = PredecessorTable.from_dataframe_of_strings(df, "act", "pred")
        self.assertEqual(pt.immediate_predecessor["C"], {"B"})

    def test_from_project_roundtrip(self):
        pg = project()
        pt = PredecessorTable.from_project(pg)
        self.assertEqual(pt.immediate_predecessor, PREDS)

    def test_copy_independent(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        pt2 = pt.copy()
        self.assertEqual(pt.immediate_predecessor, pt2.immediate_predecessor)
        self.assertIsNot(pt.immediate_predecessor, pt2.immediate_predecessor)

    def test_linkage_matrices(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        M = pt.immediate_linkage_matrix
        self.assertEqual(M.shape, (8, 8))
        self.assertTrue(M.loc["D", "A"])
        self.assertFalse(M.loc["A", "D"])
        D = pt.distant_linkage_dataframe
        self.assertTrue(D.loc["H", "A"])

    def test_create_project_roy(self):
        """Construcción PERT por el método de Roy: mismo resultado que Fondahl."""
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        pg = pt.create_project_roy()
        self.assertEqual(pg.actual_activities, sorted(PREDS))
        self.assertEqual(pg.duration(DUR), PROJECT_DURATION)
        self.assertEqual(pg.distant_predecessor(), pt.distant_predecessor)

    def test_create_project_roy_no_simplify(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        pg = pt.create_project_roy(simplify=False)
        self.assertGreater(len(pg.dummies), len(pt.create_project_roy().dummies))

    def test_create_project_fondahl(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        pg = pt.create_project_fondahl()
        self.assertEqual(pg.actual_activities, sorted(PREDS))
        self.assertEqual(pg.duration(DUR), PROJECT_DURATION)


# ---------------------------------------------------------------------------
# ProjectGraph: predecesores, contract/remove, graph helpers
# ---------------------------------------------------------------------------

class TestProjectGraphStructure(unittest.TestCase):

    def setUp(self):
        self.pg = project()

    def test_nodes_topological_and_int_like(self):
        nodes = self.pg.nodes
        self.assertEqual(nodes, [str(i) for i in range(1, len(nodes) + 1)])

    def test_activities_dummies_actual(self):
        self.assertEqual(self.pg.actual_activities, sorted(PREDS))
        self.assertEqual(set(self.pg.activities), set(self.pg.dummies) | set(PREDS))

    def test_nodes_from_edges_inverse_of_edges_from_nodes(self):
        n2e, e2n = self.pg.nodes_from_edges, self.pg.edges_from_nodes
        for act, edge in n2e.items():
            self.assertEqual(e2n[edge], act)

    def test_distant_predecessor_matches_table(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        self.assertEqual(self.pg.distant_predecessor(), pt.distant_predecessor)

    def test_distant_predecessor_dataframe_format(self):
        df = self.pg.distant_predecessor(format="DataFrame")
        self.assertEqual(df.loc["A", "predecessors"], "----")
        self.assertEqual(df.loc["D", "predecessors"], "A, B")

    def test_calculate_distant_predecessor_of(self):
        self.assertEqual(self.pg.calculate_distant_predecessor_of("H"), set("ABCDEFG"))

    def test_remove_activities(self):
        pg2 = self.pg.remove_activities(["H"])
        self.assertNotIn("H", pg2.activities)
        self.assertIn("H", self.pg.activities)  # original intacto

    def test_remove_unknown_is_noop(self):
        pg2 = self.pg.remove_activities(["ZZZ"])
        self.assertEqual(pg2.activities, self.pg.activities)

    def test_contract_activities_reduces_nodes(self):
        pg2 = self.pg.contract_activities(["H"])
        self.assertEqual(len(pg2.nodes), len(self.pg.nodes) - 1)

    def test_contract_unknown_is_noop(self):
        pg2 = self.pg.contract_activities(["ZZZ"])
        self.assertEqual(len(pg2.nodes), len(self.pg.nodes))

    def test_copy_is_independent(self):
        pg2 = self.pg.copy()
        pg2.pert_graph.add_node("999")
        self.assertNotIn("999", self.pg.pert_graph.nodes)

    def test_simplify_idempotent(self):
        pg2 = self.pg.simplify()
        self.assertEqual(pg2.actual_activities, self.pg.actual_activities)
        self.assertEqual(pg2.distant_predecessor(), self.pg.distant_predecessor())

    def test_from_dict_of_strings(self):
        pg = ProjectGraph.from_dict_of_strings(
            {k: ",".join(sorted(v)) for k, v in PREDS.items()})
        self.assertEqual(pg.duration(DUR), PROJECT_DURATION)

    def test_from_dataframe_of_strings(self):
        df = pd.DataFrame({"act": list(PREDS), "pred": [",".join(sorted(v)) for v in PREDS.values()]})
        pg = ProjectGraph.from_dataframe_of_strings(df, "act", "pred")
        self.assertEqual(pg.duration(DUR), PROJECT_DURATION)

    def test_none_graph(self):
        pg = ProjectGraph(None)
        self.assertEqual(len(pg.pert_graph.nodes), 0)


# ---------------------------------------------------------------------------
# Helpers de módulo
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_truncate_and_pad(self):
        self.assertEqual(truncate_and_pad(3.14159, 2), "3.14")
        self.assertEqual(truncate_and_pad(3.7, 0), "4")
        self.assertEqual(truncate_and_pad("2.5", 1), "2.5")
        self.assertEqual(truncate_and_pad(np.int64(5), 1), "5.0")
        self.assertEqual(truncate_and_pad(None, 2), "0.00")
        self.assertEqual(truncate_and_pad(float("nan"), 1), "0.0")

    def test_truncate_negative_digits_raises(self):
        with self.assertRaises(ValueError):
            truncate_and_pad(1.0, -1)

    def test_significant_values(self):
        self.assertEqual(significant_values(np.array([5, 3, 1, 1]), 0.5), 1)
        self.assertEqual(significant_values(np.array([5, 3, 1, 1]), 0.9), 3)

    def test_highlight_maximum(self):
        out = highlight_maximum(pd.Series([1, 3, 2]))
        self.assertEqual(out[1], "background-color: greenyellow;")
        self.assertEqual(out[0], "")

    def test_highlight_minimum(self):
        out = highlight_minimum(pd.Series([1, 3, 2]))
        self.assertEqual(out[0], "background-color: yellow;")

    def test_highlight_first_minimum(self):
        out = highlight_first_minimum(pd.Series([2, 1, 1], index=list("abc")))
        self.assertEqual(out, ["", "background-color: yellow", ""])

    def test_make_Roy(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        g = make_Roy(pt.immediate_linkage_matrix)
        self.assertIn(("start", "A"), g.edges)
        self.assertIn(("H", "finish"), g.edges)
        self.assertIn(("A", "D"), g.edges)

    def test_SVD_padding(self):
        R = project().path_matrix(dummies=False).values
        out = SVD(R)
        self.assertEqual(out["S"].shape, R.shape)
        np.testing.assert_allclose(out["U"] @ out["S"] @ out["VT"], R, atol=1e-10)
        tall = R.T  # m > n
        out2 = SVD(tall)
        self.assertEqual(out2["S"].shape, tall.shape)
        np.testing.assert_allclose(out2["U"] @ out2["S"] @ out2["VT"], tall, atol=1e-10)

    def test_pretty_latex_and_jup(self):
        R = project().path_matrix(dummies=False).values
        self.assertIsInstance(pmule.pretty_latex(R), str)
        self.assertIsInstance(pmule.pretty(R, dec=2, latex=True), str)

    def test_tanto_por_uno(self):
        out = pmule.tanto_por_uno(np.array([1, 2, 3, 4]))
        self.assertTrue(np.all(out <= 1))


# ---------------------------------------------------------------------------
# read_rcp
# ---------------------------------------------------------------------------

RCP = textwrap.dedent("""\
    4 2
    10 8
    3 2 1 2 2 3
    2 1 1 1 4
    4 2 2 1 4
    1 0 0 0
""")


class TestReadRcp(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".rcp", delete=False)
        self.tmp.write(RCP); self.tmp.close()
        self.data = read_rcp(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_counts(self):
        self.assertEqual(self.data["num_activities"], 4)
        self.assertEqual(self.data["num_resources"], 2)

    def test_availability(self):
        self.assertEqual(list(self.data["resource_availability"]["Availability"]), [10, 8])

    def test_predecessors_from_successors(self):
        p = self.data["predecessors"]
        self.assertEqual(p["A1"], "")
        self.assertEqual(p["A2"], "A1")
        self.assertEqual(p["A3"], "A1")
        self.assertEqual(p["A4"], "A2,A3")

    def test_durations_and_resources(self):
        self.assertEqual(self.data["duration"].loc["A3", "duration"], 4)
        self.assertEqual(self.data["resources"].loc["A1", "Resource 2"], 1)

    def test_builds_project(self):
        pg = ProjectGraph.from_dict_of_strings(self.data["predecessors"])
        d = self.data["duration"]["duration"].to_dict()
        self.assertEqual(pg.duration(d), 3 + 4 + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
