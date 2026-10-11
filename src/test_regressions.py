"""
test_regressions.py — Tests de regresión para bugs encontrados en la auditoría
================================================================================
Cada clase documenta un bug concreto. Los tests fallan con el código previo
a la corrección y pasan después.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from pmule import ProjectGraph, PredecessorTable

PREDS = {
    "A": set(), "B": set(),
    "C": {"A"}, "D": {"A", "B"}, "E": {"B"},
    "F": {"C", "D"}, "G": {"D", "E"}, "H": {"F", "G"},
}
DUR = {"A": 3, "B": 2, "C": 4, "D": 5, "E": 3, "F": 2, "G": 6, "H": 4}
DUR_F = {k: float(v) for k, v in DUR.items()}
DUR_F["A"] = 3.5


def project():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


class TestBug1_FloatDtype(unittest.TestCase):
    """calculate_pert infería el dtype del primer valor (una ficticia → int) y
    asignaba floats sobre Series int: FutureWarning en pandas 2.x, error en 3.0."""

    def setUp(self):
        self.pg = project()

    def _no_future_warning(self, fn):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            return fn()

    def test_calculate_pert_float_no_warning(self):
        res = self._no_future_warning(lambda: self.pg.calculate_pert(DUR_F))
        self.assertAlmostEqual(res["nodes"]["early"].max(), 18.5)

    def test_calculate_pert_float_values_exact(self):
        res = self.pg.calculate_pert(DUR_F)
        H = res["activities"]["H_total"]
        self.assertAlmostEqual(H["B"], 1.5)   # LS(B)=1.5
        self.assertAlmostEqual(H["A"], 0.0)

    def test_zaderenko_float_no_warning(self):
        z = self._no_future_warning(lambda: self.pg.zaderenko(DUR_F))
        self.assertAlmostEqual(float(z.loc["late", self.pg.nodes[-1]]), 18.5)

    def test_duration_float_no_warning(self):
        d = self._no_future_warning(lambda: self.pg.duration(DUR_F))
        self.assertAlmostEqual(d, 18.5)

    def test_calendar_float_no_warning(self):
        cal = self._no_future_warning(lambda: self.pg.calendar(pd.Series(DUR_F)))
        self.assertAlmostEqual(cal.loc["H", "fin_mas_tardio"], 18.5)

    def test_integer_input_stays_integer_valued(self):
        res = self.pg.calculate_pert(DUR)
        self.assertEqual(res["nodes"]["early"].max(), 18)
        self.assertTrue(float(res["nodes"]["early"].max()).is_integer())


class TestBug2_CriticalPathAlignment(unittest.TestCase):
    """critical_path fallaba con 'matrices are not aligned' si el dict no
    coincidía exactamente con actual_activities."""

    def setUp(self):
        self.pg = project()

    def test_partial_durations(self):
        cp = self.pg.critical_path({"A": 3, "D": 5, "G": 6, "H": 4})
        self.assertEqual(list(cp.values())[0], ["A", "D", "G", "H"])

    def test_padded_with_dummies(self):
        padded = {k: 0 for k in self.pg.activities}
        padded.update(DUR)
        cp = self.pg.critical_path(padded)
        self.assertEqual(list(cp.values())[0], ["A", "D", "G", "H"])

    def test_series_input(self):
        cp = self.pg.critical_path(pd.Series(DUR))
        self.assertEqual(len(cp), 1)

    def test_standard_deviation_with_padded_durations(self):
        padded = {k: 0 for k in self.pg.activities}
        padded.update(DUR)
        std, _ = self.pg.standard_deviation(padded, {k: 1 for k in DUR})
        self.assertAlmostEqual(std, 2.0)


class TestBug3_CalendarAcceptsDict(unittest.TestCase):
    """calendar() exigía una Series (usaba .index); con dict → AttributeError."""

    def setUp(self):
        self.pg = project()

    def test_dict_input(self):
        cal = self.pg.calendar(DUR)
        self.assertEqual(cal.loc["H", "fin_mas_tardio"], 18)

    def test_dict_and_series_equivalent(self):
        c1 = self.pg.calendar(DUR)
        c2 = self.pg.calendar(pd.Series(DUR))
        pd.testing.assert_frame_equal(
            c1.sort_index().astype(float), c2.sort_index().astype(float))

    def test_partial_dict(self):
        cal = self.pg.calendar({"A": 3})
        self.assertEqual(cal.loc["A", "fin_mas_temprano"], 3)
        self.assertIn("B", cal.index)   # las ausentes aparecen con duración 0


class TestBug4_ImmediatePredecessorsThroughDummies(unittest.TestCase):
    """calculate_immediate_predecessor_of(dummies=False) devolvía nombres de
    ficticias en vez de resolverlas hasta las actividades reales."""

    def setUp(self):
        self.pg = project()

    def test_D_has_A_and_B(self):
        self.assertEqual(self.pg.calculate_immediate_predecessor_of("D"), {"A", "B"})

    def test_F_has_C_and_D(self):
        self.assertEqual(self.pg.calculate_immediate_predecessor_of("F"), {"C", "D"})

    def test_G_has_D_and_E(self):
        self.assertEqual(self.pg.calculate_immediate_predecessor_of("G"), {"D", "E"})

    def test_start_activity_empty(self):
        self.assertEqual(self.pg.calculate_immediate_predecessor_of("A"), set())

    def test_no_dummy_names_returned(self):
        for a in self.pg.actual_activities:
            for p in self.pg.calculate_immediate_predecessor_of(a):
                self.assertFalse(p.startswith("@"), f"{a} -> {p}")

    def test_matches_predecessor_table(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        for a in PREDS:
            self.assertEqual(self.pg.calculate_immediate_predecessor_of(a),
                             pt.immediate_predecessor[a], a)

    def test_dummies_true_returns_direct_arcs(self):
        preds = self.pg.calculate_immediate_predecessor_of("D", dummies=True)
        self.assertTrue(all(p.startswith("@") for p in preds))
        self.assertEqual(len(preds), 2)


class TestBug5_NullspaceNumericOrder(unittest.TestCase):
    """incidence_matrix/nullspace ordenaban nodos como strings: con ≥10 nodos
    se eliminaba la columna '9' en lugar del sumidero."""

    def setUp(self):
        chain_preds = {chr(65 + i): ({chr(64 + i)} if i else set()) for i in range(12)}
        self.pg = ProjectGraph.from_dict_of_sets(chain_preds)
        self.assertGreaterEqual(len(self.pg.nodes), 13)

    def test_incidence_columns_numeric_order(self):
        H = self.pg.incidence_matrix()
        self.assertEqual(list(H.columns), self.pg.nodes)

    def test_nullspace_drops_true_source_and_sink(self):
        N = self.pg.nullspace()
        self.assertNotIn(self.pg.nodes[0], N.columns)
        self.assertNotIn(self.pg.nodes[-1], N.columns)
        self.assertIn("9", N.columns)


class TestBug6_CycleDetection(unittest.TestCase):
    """PredecessorTable con ciclo o auto-predecesora → RecursionError."""

    def test_two_cycle_raises_value_error(self):
        with self.assertRaises(ValueError) as cm:
            PredecessorTable.from_dict_of_sets({"A": {"B"}, "B": {"A"}})
        self.assertIn("ciclo", str(cm.exception).lower())

    def test_self_predecessor_raises_value_error(self):
        with self.assertRaises(ValueError):
            PredecessorTable.from_dict_of_sets({"A": {"A"}})

    def test_long_cycle(self):
        with self.assertRaises(ValueError):
            PredecessorTable.from_dict_of_strings({"A": "C", "B": "A", "C": "B"})

    def test_project_graph_constructors_propagate(self):
        with self.assertRaises(ValueError):
            ProjectGraph.from_dict_of_sets({"A": {"B"}, "B": {"A"}})

    def test_acyclic_still_fine(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        self.assertEqual(pt.start_activities, {"A", "B"})


class TestBug10_AckoffNothingReducible(unittest.TestCase):
    """ackoff() reventaba con ValueError cuando ninguna actividad crítica
    tenía margen de reducción (DataFrame sin columnas en apply)."""

    def setUp(self):
        self.pg = project()
        self.cost = {k: 10 for k in DUR}

    def test_all_at_minimum(self):
        _, best, dur, _ = self.pg.ackoff(DUR, dict(DUR), self.cost, reduction=5)
        self.assertEqual(best, {})
        self.assertEqual(dur.to_dict(), DUR)

    def test_only_noncritical_reducible(self):
        mind = dict(DUR); mind["C"] = 1; mind["E"] = 1   # C y E no son críticas
        _, best, dur, _ = self.pg.ackoff(DUR, mind, self.cost, reduction=5)
        self.assertEqual(best, {})

    def test_stops_midway_gracefully(self):
        mind = dict(DUR); mind["H"] = 3   # solo H reducible, 1 periodo
        _, best, dur, _ = self.pg.ackoff(DUR, mind, self.cost, reduction=5)
        self.assertEqual(best, {0: ["H"]})
        self.assertEqual(dur["H"], 3)


class TestBug11_DistantLinkageMatrixAPI(unittest.TestCase):
    """distant_linkage_matrix devuelve Styler (se conserva por compatibilidad
    con notebooks que usan .data). Se añade distant_linkage_dataframe y
    display_distant_linkage_matrix() para tener una API paralela a la de
    immediate_linkage_matrix."""

    def test_styler_kept_for_compat(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        S = pt.distant_linkage_matrix
        self.assertTrue(hasattr(S, "data"))
        self.assertTrue(S.data.astype(bool).loc["H", "A"])

    def test_dataframe_variant(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        D = pt.distant_linkage_dataframe
        self.assertIsInstance(D, pd.DataFrame)
        self.assertTrue(D.loc["H", "A"])
        self.assertFalse(D.loc["A", "H"])
        self.assertEqual(D.dtypes.unique().tolist(), [np.dtype(bool)])

    def test_display_variants_exist(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        self.assertTrue(hasattr(pt.display_immediate_linkage_matrix(), "data"))
        self.assertTrue(hasattr(pt.display_distant_linkage_matrix(), "data"))


class TestBug8_MutationDocumented(unittest.TestCase):
    """desplazar()/nivelar() mutan self.pert_graph. Comportamiento conservado
    por decisión de diseño; aquí se fija y se exige que esté documentado."""

    def test_desplazar_mutates_self(self):
        pg = project()
        data = pd.DataFrame({"duration": DUR, "resources": {k: 1 for k in DUR}})
        n0 = len(pg.nodes)
        pg.desplazar(data, "duration", "resources", report=False, C=1)
        self.assertEqual(len(pg.nodes), n0 + 1)
        self.assertIn("💤C", pg.activities)

    def test_docstrings_warn(self):
        self.assertIn("muta", ProjectGraph.desplazar.__doc__.lower())
        self.assertIn("muta", ProjectGraph.nivelar.__doc__.lower())


class TestBug12_EarnedValueHorizon(unittest.TestCase):
    """EarnedValue.calcula_gantts: si la ejecución real era más larga que la
    planificada, PV acumulado terminaba en NaN. El horizonte sigue siendo el
    de la ejecución real (= fecha de estado); PV se prolonga plana en el BAC."""

    def _data(self, actual):
        return pd.DataFrame({
            "planned": DUR, "actual": actual,
            "PV": {a: 100 * DUR[a] for a in DUR},
            "AC": {a: 110 * DUR[a] for a in DUR},
            "pct": {"A": 1, "B": 1, "C": 1, "D": 0.5, "E": 1, "F": 0, "G": 0, "H": 0},
        })

    def test_actual_longer_than_planned_pv_plateaus(self):
        from pmule import EarnedValue
        longer = {"A": 3, "B": 3, "C": 4, "D": 6, "E": 3, "F": 2, "G": 6, "H": 4}  # 19
        out = EarnedValue(project()).calcula_gantts(
            self._data(longer), "planned", "actual", "PV", "AC", "pct")
        acc = out["acumulados"]
        self.assertFalse(acc.isna().any().any())
        self.assertEqual(len(acc), 19)
        self.assertAlmostEqual(float(acc["PV"].iloc[-1]), 2900.0)
        self.assertAlmostEqual(float(acc["PV"].loc[18]), float(acc["PV"].loc[19]))

    def test_actual_shorter_than_planned_is_status_date(self):
        """Las duraciones reales describen el trabajo hasta la fecha de estado:
        `acumulados` termina ahí y PV es el planificado *hasta esa fecha*
        (no el BAC). Es lo que los cuadernos usan para CPI/SPI."""
        from pmule import EarnedValue
        shorter = {"A": 2, "B": 2, "C": 4, "D": 4, "E": 3, "F": 2, "G": 5, "H": 3}  # 14
        out = EarnedValue(project()).calcula_gantts(
            self._data(shorter), "planned", "actual", "PV", "AC", "pct")
        acc = out["acumulados"]
        self.assertEqual(len(acc), 14)
        self.assertFalse(acc.isna().any().any())
        pv_planned_14 = float(out["Gantt_PV"].data.loc["Total", list(range(1, 15))].sum())
        self.assertAlmostEqual(float(acc["PV"].iloc[-1]), pv_planned_14)
        self.assertLess(float(acc["PV"].iloc[-1]), 2900.0)

    def test_same_horizon_unchanged(self):
        from pmule import EarnedValue
        out = EarnedValue(project()).calcula_gantts(
            self._data(dict(DUR)), "planned", "actual", "PV", "AC", "pct")
        self.assertEqual(len(out["acumulados"]), 18)
        self.assertAlmostEqual(float(out["acumulados"]["PV"].iloc[-1]), 2900.0)


class TestBug13_ParallelEndActivitiesKeepSingleSink(unittest.TestCase):
    """Dos actividades finales con las mismas predecesoras (I y J en el
    Ejercicio 03) acababan como dos sumideros: al simplificar, la ficticia que
    conectaba la cabeza de J al sumidero se *eliminaba* porque la comprobación
    solo miraba predecesoras. Luego calculate_pert fallaba con
    'min() iterable argument is empty'."""

    PREDS03 = {'A': set(), 'B': set(), 'C': {'B'}, 'D': {'A'}, 'E': {'B', 'D'},
               'F': {'B', 'C', 'D'}, 'G': set(), 'H': {'G'},
               'I': {'E', 'F', 'H'}, 'J': {'E', 'F', 'H'}}
    DUR03 = {'A': 2, 'B': 4, 'C': 1, 'D': 3, 'E': 7, 'F': 4, 'G': 1, 'H': 11, 'I': 2, 'J': 8}

    def _check(self, pg):
        G = pg.pert_graph
        self.assertEqual(sum(1 for n in G if G.out_degree(n) == 0), 1, "un solo sumidero")
        self.assertEqual(sum(1 for n in G if G.in_degree(n) == 0), 1, "un solo origen")
        self.assertEqual(pg.actual_activities, sorted(self.PREDS03))
        self.assertEqual(pg.duration(self.DUR03), 20)      # (A-D-E | G-H) = 12, + J = 8
        self.assertEqual(pg.distant_predecessor(),
                         PredecessorTable.from_dict_of_sets(self.PREDS03).distant_predecessor)

    def test_fondahl_default(self):
        self._check(ProjectGraph.from_dict_of_sets(self.PREDS03))

    def test_roy_method(self):
        self._check(PredecessorTable.from_dict_of_sets(self.PREDS03).create_project_roy())

    def test_zaderenko_and_calendar_work(self):
        pg = ProjectGraph.from_dict_of_sets(self.PREDS03)
        z = pg.zaderenko(self.DUR03)
        self.assertEqual(float(z.loc["late", pg.nodes[-1]]), 20)
        cal = pg.calendar(self.DUR03)
        self.assertEqual(cal.loc["I", "H_total"], 6)        # J tarda 8, I tarda 2
        self.assertEqual(cal.loc["J", "H_total"], 0)

    def test_parallel_activities_not_collapsed(self):
        """Caso mínimo: A y B ambas finales, sin predecesoras → necesitan una ficticia."""
        pg = ProjectGraph.from_dict_of_sets({"A": set(), "B": set()})
        self._assert_single(pg)
        self.assertEqual(pg.duration({"A": 3, "B": 5}), 5)

    def test_three_parallel_end_activities(self):
        pg = ProjectGraph.from_dict_of_sets({"X": set(), "A": {"X"}, "B": {"X"}, "C": {"X"}})
        self._assert_single(pg)
        self.assertEqual(len(pg), 4)
        self.assertEqual(pg.duration({"X": 1, "A": 1, "B": 2, "C": 3}), 4)

    def _assert_single(self, pg):
        G = pg.pert_graph
        self.assertEqual(sum(1 for n in G if G.out_degree(n) == 0), 1)
        self.assertEqual(sum(1 for n in G if G.in_degree(n) == 0), 1)

    def test_simplify_method_also_safe(self):
        pg = ProjectGraph.from_dict_of_sets(self.PREDS03, simplify=False).simplify()
        self._check(pg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
