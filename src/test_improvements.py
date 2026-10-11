"""
test_improvements.py — Tests de las mejoras de la auditoría
============================================================
_normalize_durations (validación), holguras libre/independiente,
monte_carlo vectorizado (equivalencia con tau/tau_sensitivity), dunders.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from pmule import ProjectGraph, MonteCarloResult

PREDS = {
    "A": set(), "B": set(),
    "C": {"A"}, "D": {"A", "B"}, "E": {"B"},
    "F": {"C", "D"}, "G": {"D", "E"}, "H": {"F", "G"},
}
DUR = {"A": 3, "B": 2, "C": 4, "D": 5, "E": 3, "F": 2, "G": 6, "H": 4}
H_LIBRE = {"A": 0, "B": 0, "C": 1, "D": 0, "E": 3, "F": 4, "G": 0, "H": 0}
H_INDEP = {"A": 0, "B": 0, "C": 1, "D": 0, "E": 2, "F": 0, "G": 0, "H": 0}


def project():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


class TestNormalizeDurations(unittest.TestCase):

    def setUp(self):
        self.pg = project()

    def test_dict_to_series_with_dummies(self):
        s = self.pg._normalize_durations(DUR)
        self.assertEqual(list(s.index), self.pg.activities)
        for d in self.pg.dummies:
            self.assertEqual(s[d], 0)

    def test_without_dummies(self):
        s = self.pg._normalize_durations(DUR, dummies=False)
        self.assertEqual(list(s.index), self.pg.actual_activities)

    def test_series_input(self):
        s = self.pg._normalize_durations(pd.Series(DUR))
        self.assertEqual(s["H"], 4)

    def test_int_dtype_preserved(self):
        self.assertTrue(np.issubdtype(self.pg._normalize_durations(DUR).dtype, np.integer))

    def test_float_dtype_when_any_float(self):
        d = dict(DUR); d["A"] = 3.5
        self.assertTrue(np.issubdtype(self.pg._normalize_durations(d).dtype, np.floating))

    def test_numpy_ints_stay_int(self):
        d = {k: np.int64(v) for k, v in DUR.items()}
        self.assertTrue(np.issubdtype(self.pg._normalize_durations(d).dtype, np.integer))

    def test_missing_default_zero(self):
        s = self.pg._normalize_durations({"A": 3})
        self.assertEqual(s["B"], 0)

    def test_nan_treated_as_zero(self):
        s = self.pg._normalize_durations({"A": float("nan"), "B": 2})
        self.assertEqual(s["A"], 0)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            self.pg._normalize_durations({"A": -1})

    def test_unknown_activity_warns(self):
        with self.assertWarns(UserWarning):
            self.pg._normalize_durations({"ZZZ": 3})

    def test_unknown_can_be_silenced(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.pg._normalize_durations({"ZZZ": 3}, warn_unknown=False)

    def test_dataframe_rejected(self):
        with self.assertRaises(TypeError):
            self.pg._normalize_durations(pd.DataFrame({"duration": DUR}))

    def test_public_methods_warn_on_unknown(self):
        bad = dict(DUR); bad["Ab"] = 1   # error tipográfico típico
        with self.assertWarns(UserWarning):
            self.pg.duration(bad)
        with self.assertWarns(UserWarning):
            self.pg.critical_path(bad)

    def test_public_methods_raise_on_negative(self):
        bad = dict(DUR); bad["A"] = -3
        for fn in (self.pg.duration, self.pg.critical_path, self.pg.zaderenko,
                   self.pg.calendar, self.pg.calculate_pert):
            with self.assertRaises(ValueError, msg=fn.__name__):
                fn(bad)


class TestFreeAndIndependentFloat(unittest.TestCase):

    def setUp(self):
        self.cal = project().calendar(DUR)

    def test_columns_present(self):
        self.assertIn("H_libre", self.cal.columns)
        self.assertIn("H_independiente", self.cal.columns)

    def test_free_float_values(self):
        for a, v in H_LIBRE.items():
            self.assertEqual(self.cal.loc[a, "H_libre"], v, a)

    def test_independent_float_values(self):
        for a, v in H_INDEP.items():
            self.assertEqual(self.cal.loc[a, "H_independiente"], v, a)

    def test_ordering_total_ge_free_ge_independent(self):
        self.assertTrue((self.cal["H_total"] >= self.cal["H_libre"]).all())
        self.assertTrue((self.cal["H_libre"] >= self.cal["H_independiente"]).all())

    def test_nonnegative(self):
        self.assertTrue((self.cal["H_independiente"] >= 0).all())
        self.assertTrue((self.cal["H_libre"] >= 0).all())

    def test_chain_all_zero(self):
        pg = ProjectGraph.from_dict_of_sets({"A": set(), "B": {"A"}, "C": {"B"}})
        cal = pg.calendar({"A": 1, "B": 2, "C": 3})
        self.assertTrue((cal[["H_total", "H_libre", "H_independiente"]] == 0).all().all())

    def test_free_float_definition(self):
        """H_libre = ES(sucesoras) - EF."""
        pg = project()
        cal = pg.calendar(DUR)
        n2e = pg.nodes_from_edges
        early = pg.calculate_pert(DUR)["nodes"]["early"]
        for a in DUR:
            s, t = n2e[a]
            self.assertEqual(cal.loc[a, "H_libre"], early[t] - cal.loc[a, "fin_mas_temprano"], a)


class TestMonteCarloVectorized(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.dists = {k: ("normal", v, 0.8) for k, v in DUR.items()}

    def test_fast_for_large_n(self):
        import time
        t0 = time.time()
        r = self.pg.monte_carlo(self.dists, n_samples=50_000, seed=1)
        self.assertLess(time.time() - t0, 5.0)
        self.assertEqual(len(r.tau_samples), 50_000)

    def test_tau_matches_scalar_tau_per_sample(self):
        """Cada τ simulada coincide con tau() evaluada en esa muestra."""
        rng = np.random.default_rng(3)
        samples = {k: rng.normal(v, 0.8, 20) for k, v in DUR.items()}
        dists = {k: (lambda size, _s=s: _s[:size]) for k, s in samples.items()}
        r = self.pg.monte_carlo(dists, n_samples=20)
        for i in range(20):
            d = {k: float(samples[k][i]) for k in DUR}
            self.assertAlmostEqual(r.tau_samples[i], self.pg.tau(d), places=10)

    def test_sensitivity_matches_scalar_mean(self):
        """sensitivity_mean = media de tau_sensitivity() sobre las muestras."""
        rng = np.random.default_rng(5)
        n = 30
        samples = {k: rng.normal(v, 1.0, n) for k, v in DUR.items()}
        dists = {k: (lambda size, _s=s: _s[:size]) for k, s in samples.items()}
        r = self.pg.monte_carlo(dists, n_samples=n)
        acc = pd.Series(0.0, index=self.pg.actual_activities)
        for i in range(n):
            d = {k: float(samples[k][i]) for k in DUR}
            acc = acc.add(self.pg.tau_sensitivity(d), fill_value=0.0)
        expected = (acc / n)
        for a in expected.index:
            self.assertAlmostEqual(r.sensitivity_mean[a], expected[a], places=10, msg=a)

    def test_seed_reproducible(self):
        r1 = self.pg.monte_carlo(self.dists, n_samples=500, seed=11)
        r2 = self.pg.monte_carlo(self.dists, n_samples=500, seed=11)
        np.testing.assert_array_equal(r1.tau_samples, r2.tau_samples)

    def test_unknown_activity_warns(self):
        d = dict(self.dists); d["ZZZ"] = ("normal", 1, 0.1)
        with self.assertWarns(UserWarning):
            self.pg.monte_carlo(d, n_samples=10, seed=0)

    def test_unknown_distribution_kind_raises(self):
        d = dict(self.dists); d["A"] = ("weibull", 1, 2)
        with self.assertRaises(ValueError):
            self.pg.monte_carlo(d, n_samples=10, seed=0)

    def test_scalar_callable_without_size(self):
        import random
        random.seed(0)
        d = {k: (lambda _v=v: _v + random.random()) for k, v in DUR.items()}
        r = self.pg.monte_carlo(d, n_samples=50)
        self.assertEqual(len(r.tau_samples), 50)
        self.assertTrue((r.tau_samples >= 18).all())

    def test_sensitivity_sums_to_critical_path_length_in_expectation(self):
        """Σ_i sensibilidad_i = longitud media (en nº de actividades) de la ruta crítica = 4."""
        r = self.pg.monte_carlo(self.dists, n_samples=2000, seed=2)
        self.assertAlmostEqual(r.sensitivity_mean.sum(), 4.0, places=6)


class TestDunders(unittest.TestCase):

    def setUp(self):
        self.pg = project()

    def test_repr(self):
        self.assertEqual(repr(self.pg), "ProjectGraph(activities=8, dummies=4, nodes=9)")

    def test_len(self):
        self.assertEqual(len(self.pg), 8)

    def test_contains(self):
        self.assertIn("C", self.pg)
        self.assertNotIn("Z", self.pg)
        self.assertIn(self.pg.dummies[0], self.pg)

    def test_eq_copy_and_simplify(self):
        self.assertEqual(self.pg, self.pg.copy())
        self.assertEqual(self.pg, self.pg.simplify())

    def test_eq_different_construction_methods(self):
        from pmule import PredecessorTable
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        self.assertEqual(pt.create_project_roy(), pt.create_project_fondahl())

    def test_neq_different_structure(self):
        other = ProjectGraph.from_dict_of_sets({**PREDS, "H": {"F"}})
        self.assertNotEqual(self.pg, other)

    def test_neq_other_type(self):
        self.assertNotEqual(self.pg, "ProjectGraph")

    def test_unhashable(self):
        with self.assertRaises(TypeError):
            hash(self.pg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
