"""
test_analysis.py — Tests for pmULE analysis methods
=====================================================
Covers: tau, tau_sensitivity, svd_approximation_error,
        PSPLIBInstance.read, ProjectGraph.from_sm, plot_tikz, _pert_layout
"""

import sys
import os
import types
import pathlib
import tempfile
import unittest
import textwrap

# pmule ya no importa pygraphviz a nivel de módulo (import lazy en
# plot_pert/plot_roy). Este mock solo se usa, de forma local, en
# test_no_graphviz_import; NO se instala en sys.modules globalmente para
# no contaminar otros ficheros de test que sí usan graphviz.
pgv_mock = types.ModuleType("pygraphviz")
pgv_mock.AGraph = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmule import ProjectGraph, PSPLIBInstance

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chain(n=3):
    """Chain project A→B→C→… with n activities."""
    acts = [chr(65 + i) for i in range(n)]
    preds = {a: {acts[i - 1]} if i > 0 else set() for i, a in enumerate(acts)}
    return ProjectGraph.from_dict_of_sets(preds, simplify=True), acts


def make_diamond():
    """A→{B,C}→D — two parallel paths."""
    return ProjectGraph.from_dict_of_sets(
        {"A": set(), "B": {"A"}, "C": {"A"}, "D": {"B", "C"}},
        simplify=True
    )


MINI_SM = textwrap.dedent("""\
    file with basedata            : mini.sm
    initial value random generator: 1
    ************************************************************************
    projects                      :  1
    jobs (incl. supersource/sink ):  4
    horizon                       : 20
    RESOURCES
    - renewable                 :  1   R
    - nonrenewable              :  0   N
    - doubly constrained        :  0   D
    ************************************************************************
    PROJECT INFORMATION:
    pronr.  #jobs rel.date duedate tardcost  MPM-Time
        1      2      0        8        1         8
    ************************************************************************
    PRECEDENCE RELATIONS:
    jobnr.    #modes  #successors   successors
       1        1          1           2
       2        1          1           3
       3        1          1           4
       4        1          0
    ************************************************************************
    REQUESTS/DURATIONS:
    jobnr. mode duration  R 1
    ------------------------------------------------------------------------
       1      1     0       0
       2      1     3       2
       3      1     5       1
       4      1     0       0
    ************************************************************************
    RESOURCEAVAILABILITIES:
      R 1
        3
    ************************************************************************
""")


# ---------------------------------------------------------------------------
# tau tests
# ---------------------------------------------------------------------------

class TestTau(unittest.TestCase):

    def test_chain_tau_equals_sum(self):
        proj, acts = make_chain(3)
        durations = {a: 1.0 for a in acts}
        # path A→B→C: tau = 3
        self.assertAlmostEqual(proj.tau(durations), 3.0)

    def test_diamond_tau_equals_longer_path(self):
        proj = make_diamond()
        # Path A→B→D: 1+4+2=7, A→C→D: 1+1+2=4 → tau=7
        durations = {"A": 1, "B": 4, "C": 1, "D": 2}
        self.assertAlmostEqual(proj.tau(durations), 7.0)

    def test_tau_zero_durations(self):
        proj = make_diamond()
        durations = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.assertAlmostEqual(proj.tau(durations), 0.0)

    def test_tau_single_activity(self):
        proj = ProjectGraph.from_dict_of_sets({"X": set()}, simplify=True)
        self.assertAlmostEqual(proj.tau({"X": 7.5}), 7.5)

    def test_tau_returns_float(self):
        proj, acts = make_chain(2)
        result = proj.tau({a: 1 for a in acts})
        self.assertIsInstance(result, float)

    def test_tau_missing_key_treated_as_zero(self):
        proj, acts = make_chain(3)
        # Only supply first activity
        result = proj.tau({acts[0]: 5})
        self.assertGreaterEqual(result, 5.0)

    def test_tau_longer_chain_correct(self):
        proj, acts = make_chain(5)
        durations = {a: float(i + 1) for i, a in enumerate(acts)}
        expected = sum(range(1, 6))  # 1+2+3+4+5 = 15
        self.assertAlmostEqual(proj.tau(durations), expected)

    def test_tau_consistent_with_path_matrix(self):
        proj = make_diamond()
        durations = {"A": 2, "B": 3, "C": 1, "D": 4}
        R = proj.path_matrix(dummies=False)
        t = np.array([durations.get(c, 0) for c in R.columns])
        expected = float(np.max(R.values @ t))
        self.assertAlmostEqual(proj.tau(durations), expected)


# ---------------------------------------------------------------------------
# tau_sensitivity tests
# ---------------------------------------------------------------------------

class TestTauSensitivity(unittest.TestCase):

    def test_returns_series(self):
        proj = make_diamond()
        result = proj.tau_sensitivity({"A": 1, "B": 4, "C": 1, "D": 2})
        self.assertIsInstance(result, pd.Series)

    def test_critical_activities_have_weight_one(self):
        # Chain: single critical path → all activities weight 1
        proj, acts = make_chain(3)
        sens = proj.tau_sensitivity({a: 1 for a in acts})
        for a in acts:
            if a in sens.index:
                self.assertAlmostEqual(sens[a], 1.0)

    def test_noncritical_activity_zero(self):
        proj = make_diamond()
        # Path A→B→D dominates (4 vs 1)
        sens = proj.tau_sensitivity({"A": 1, "B": 4, "C": 1, "D": 2})
        if "C" in sens.index:
            self.assertAlmostEqual(sens["C"], 0.0)

    def test_weights_between_zero_and_one(self):
        proj = make_diamond()
        sens = proj.tau_sensitivity({"A": 1, "B": 2, "C": 2, "D": 1})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in sens.values))

    def test_sorted_descending(self):
        proj = make_diamond()
        sens = proj.tau_sensitivity({"A": 1, "B": 4, "C": 1, "D": 2})
        vals = list(sens.values)
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_tied_paths_equal_weights(self):
        # Both paths same length → all activities weight 1 (both critical)
        proj = make_diamond()
        sens = proj.tau_sensitivity({"A": 1, "B": 2, "C": 2, "D": 1})
        # A and D are on both paths → weight 1.0
        if "A" in sens.index:
            self.assertAlmostEqual(sens["A"], 1.0)
        if "D" in sens.index:
            self.assertAlmostEqual(sens["D"], 1.0)

    def test_index_matches_path_matrix_columns(self):
        proj = make_diamond()
        R = proj.path_matrix(dummies=False)
        sens = proj.tau_sensitivity({"A": 1, "B": 2, "C": 2, "D": 1})
        self.assertEqual(set(sens.index), set(R.columns))


# ---------------------------------------------------------------------------
# svd_approximation_error tests
# ---------------------------------------------------------------------------

class TestSvdApproximationError(unittest.TestCase):

    def test_returns_dict_with_expected_keys(self):
        proj = make_diamond()
        result = proj.svd_approximation_error({"A": 1, "B": 2, "C": 1, "D": 2})
        expected_keys = {"tau_exact", "tau_svd", "abs_error", "rel_error", "sigma_ratio"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_tau_exact_matches_tau_method(self):
        proj = make_diamond()
        durations = {"A": 1, "B": 3, "C": 2, "D": 1}
        err = proj.svd_approximation_error(durations)
        self.assertAlmostEqual(err["tau_exact"], proj.tau(durations))

    def test_abs_error_nonnegative(self):
        proj = make_diamond()
        err = proj.svd_approximation_error({"A": 1, "B": 2, "C": 1, "D": 2})
        self.assertGreaterEqual(err["abs_error"], 0.0)

    def test_rel_error_between_0_and_1_ish(self):
        proj = make_diamond()
        err = proj.svd_approximation_error({"A": 1, "B": 2, "C": 1, "D": 2})
        self.assertGreaterEqual(err["rel_error"], 0.0)

    def test_sigma_ratio_positive(self):
        proj = make_diamond()
        err = proj.svd_approximation_error({"A": 1, "B": 2, "C": 1, "D": 2})
        self.assertGreater(err["sigma_ratio"], 0.0)

    def test_chain_sigma_ratio_inf(self):
        # Single-path chain has rank-1 R matrix → σ₂=0 → ratio=inf
        proj, acts = make_chain(3)
        err = proj.svd_approximation_error({a: 1 for a in acts})
        self.assertEqual(err["sigma_ratio"], float("inf"))

    def test_chain_exact_equals_svd(self):
        # For a rank-1 route matrix, SVD approx = exact (up to signs/scaling)
        proj, acts = make_chain(2)
        err = proj.svd_approximation_error({a: 1.0 for a in acts})
        # rel_error should be small for rank-1
        self.assertAlmostEqual(err["sigma_ratio"], float("inf"))


# ---------------------------------------------------------------------------
# PSPLIBInstance tests
# ---------------------------------------------------------------------------

class TestPSPLIBInstance(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sm", mode="w",
                                               delete=False)
        self.tmp.write(MINI_SM)
        self.tmp.close()
        self.path = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_n_jobs(self):
        inst = PSPLIBInstance.read(self.path)
        self.assertEqual(inst.n_jobs, 4)

    def test_horizon(self):
        inst = PSPLIBInstance.read(self.path)
        self.assertEqual(inst.horizon, 20)

    def test_mpm_time(self):
        inst = PSPLIBInstance.read(self.path)
        self.assertEqual(inst.mpm_time, 8)

    def test_predecessors_is_dict_of_sets(self):
        inst = PSPLIBInstance.read(self.path)
        self.assertIsInstance(inst.predecessors, dict)
        for v in inst.predecessors.values():
            self.assertIsInstance(v, set)

    def test_predecessors_source_has_no_preds(self):
        inst = PSPLIBInstance.read(self.path)
        self.assertEqual(inst.predecessors.get("1", set()), set())

    def test_chain_predecessors_correct(self):
        inst = PSPLIBInstance.read(self.path)
        # 1→2→3→4 chain
        self.assertIn("1", inst.predecessors.get("2", set()))
        self.assertIn("2", inst.predecessors.get("3", set()))
        self.assertIn("3", inst.predecessors.get("4", set()))

    def test_durations_dict(self):
        inst = PSPLIBInstance.read(self.path)
        self.assertEqual(inst.durations["2"], 3)
        self.assertEqual(inst.durations["3"], 5)

    def test_resource_availabilities(self):
        inst = PSPLIBInstance.read(self.path)
        self.assertEqual(inst.resource_availabilities, [3])

    def test_resources_per_job(self):
        inst = PSPLIBInstance.read(self.path)
        self.assertIsInstance(inst.resources["2"], list)
        self.assertEqual(inst.resources["2"][0], 2)


# ---------------------------------------------------------------------------
# ProjectGraph.from_sm tests
# ---------------------------------------------------------------------------

class TestFromSm(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sm", mode="w",
                                               delete=False)
        self.tmp.write(MINI_SM)
        self.tmp.close()
        self.path = pathlib.Path(self.tmp.name)
        self.inst = PSPLIBInstance.read(self.path)
        self.proj = ProjectGraph.from_sm(self.path)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_returns_project_graph(self):
        self.assertIsInstance(self.proj, ProjectGraph)

    def test_graph_has_nodes(self):
        self.assertGreater(len(self.proj.pert_graph.nodes), 0)

    def test_tau_correct(self):
        tau = self.proj.tau(self.inst.durations)
        # chain: 0+3+5+0 = 8
        self.assertAlmostEqual(tau, 8.0)

    def test_horizon_stored_in_graph(self):
        self.assertEqual(self.proj.pert_graph.graph.get("horizon"), 20)

    def test_mpm_time_stored_in_graph(self):
        self.assertEqual(self.proj.pert_graph.graph.get("mpm_time"), 8)

    def test_n_jobs_stored_in_graph(self):
        self.assertEqual(self.proj.pert_graph.graph.get("n_jobs"), 4)


# ---------------------------------------------------------------------------
# _pert_layout tests
# ---------------------------------------------------------------------------

class TestPertLayout(unittest.TestCase):

    def test_returns_dict(self):
        proj, _ = make_chain(3)
        pos = proj._pert_layout()
        self.assertIsInstance(pos, dict)

    def test_all_nodes_have_positions(self):
        proj = make_diamond()
        pos = proj._pert_layout()
        for node in proj.pert_graph.nodes:
            self.assertIn(node, pos)

    def test_positions_are_2_tuples(self):
        proj, _ = make_chain(3)
        for v in proj._pert_layout().values():
            self.assertEqual(len(v), 2)

    def test_x_coordinates_nonnegative(self):
        proj, _ = make_chain(4)
        for x, y in proj._pert_layout().values():
            self.assertGreaterEqual(x, 0.0)

    def test_source_x_is_zero(self):
        proj = make_diamond()
        pos = proj._pert_layout()
        source = proj.nodes[0]
        self.assertAlmostEqual(pos[source][0], 0.0)

    def test_without_durations_all_x_zero(self):
        """No durations → all activities have duration 0 → all events at x=0."""
        proj = make_diamond()
        pos = proj._pert_layout()  # no durations
        for x, _ in pos.values():
            self.assertAlmostEqual(x, 0.0)

    def test_with_durations_x_increases(self):
        """With durations the sink node must be to the right of the source."""
        proj, acts = make_chain(3)
        dur = {a: 2.0 for a in acts}
        pos = proj._pert_layout(durations=dur)
        xs = sorted(set(x for x, _ in pos.values()))
        self.assertGreater(xs[-1], xs[0])

    def test_with_durations_sink_x_equals_sum(self):
        """Chain A(3)→B(4)→C(2): sink x = (3+4+2)*x_scale."""
        proj, acts = make_chain(3)
        durs = {a: d for a, d in zip(acts, [3.0, 4.0, 2.0])}
        pos = proj._pert_layout(durations=durs)
        sink = proj.nodes[-1]
        x_scale = 2.0
        expected_x = 9.0 * x_scale
        self.assertAlmostEqual(pos[sink][0], expected_x, places=5)

    def test_dummies_get_zero_duration(self):
        """Dummy edges (activity starts with @) must not contribute to early time."""
        proj = make_diamond()
        acts_real = [a for a in proj.activities if not a.startswith("@")]
        dur = {a: 5.0 for a in acts_real}
        pos = proj._pert_layout(durations=dur)
        # source always x=0
        source = proj.nodes[0]
        self.assertAlmostEqual(pos[source][0], 0.0)


# ---------------------------------------------------------------------------
# plot_tikz tests
# ---------------------------------------------------------------------------

class TestPlotTikz(unittest.TestCase):

    def test_returns_string(self):
        proj, _ = make_chain(3)
        result = proj.plot_tikz()
        self.assertIsInstance(result, str)

    def test_contains_tikzpicture(self):
        proj, _ = make_chain(3)
        tikz = proj.plot_tikz()
        self.assertIn(r"\begin{tikzpicture}", tikz)
        self.assertIn(r"\end{tikzpicture}", tikz)

    def test_contains_node_commands(self):
        proj, _ = make_chain(3)
        tikz = proj.plot_tikz()
        self.assertIn(r"\node", tikz)

    def test_contains_draw_commands(self):
        proj, _ = make_chain(3)
        tikz = proj.plot_tikz()
        self.assertIn(r"\draw", tikz)

    def test_svd_mode_returns_valid_tikz(self):
        """svd=True must produce valid TikZ (no crash, correct delimiters)."""
        proj = make_diamond()
        tikz = proj.plot_tikz(svd=True)
        self.assertIn(r"\begin{tikzpicture}", tikz)
        self.assertIn(r"\end{tikzpicture}", tikz)

    def test_no_graphviz_import(self):
        import unittest.mock as mock
        proj, _ = make_chain(3)
        with mock.patch.dict(sys.modules, {"pygraphviz": pgv_mock}):
            tikz = proj.plot_tikz()
        self.assertIn(r"\begin{tikzpicture}", tikz)

    def test_node_count_reasonable(self):
        proj, _ = make_chain(3)
        tikz = proj.plot_tikz()
        node_count = tikz.count(r"\node")
        self.assertGreaterEqual(node_count, 2)

    def test_svd_false_no_line_width_variation(self):
        proj, _ = make_chain(3)
        tikz = proj.plot_tikz(svd=False)
        self.assertNotIn("line width=", tikz)

    def test_durations_appear_in_edge_labels(self):
        """When durations are passed, edge labels include the duration value."""
        proj, acts = make_chain(3)
        dur = {a: 7 for a in acts}
        tikz = proj.plot_tikz(durations=dur)
        self.assertIn("(7)", tikz)

    def test_activity_names_in_edge_labels(self):
        """Activity names appear in edge labels for real (non-dummy) arcs."""
        proj, acts = make_chain(2)
        dur = {a: 1 for a in acts}
        tikz = proj.plot_tikz(durations=dur)
        for act in acts:
            self.assertIn(act, tikz)

    def test_dummies_get_dashed_style(self):
        """Diamond project generates dummy arcs — they must use dashed style."""
        proj = make_diamond()
        tikz = proj.plot_tikz()
        self.assertIn("dashed", tikz)

    def test_durations_none_still_valid(self):
        """plot_tikz() with no durations must still produce valid TikZ."""
        proj = make_diamond()
        tikz = proj.plot_tikz()
        self.assertIn(r"\begin{tikzpicture}", tikz)

    def test_x_positions_spread_with_durations(self):
        """Nodes must NOT all share x=0 when durations are provided."""
        proj, acts = make_chain(4)
        dur = {a: 3 for a in acts}
        tikz = proj.plot_tikz(durations=dur)
        # Extract all "at (Xcm," values
        import re
        xs = [float(m) for m in re.findall(r"at \(([\d.]+)cm,", tikz)]
        self.assertGreater(max(xs) - min(xs), 0)



# ---------------------------------------------------------------------------
# MonteCarloResult + monte_carlo() tests
# ---------------------------------------------------------------------------

from pmule import MonteCarloResult


def make_project_8act():
    """Proyecto de 8 actividades para tests de Monte Carlo."""
    preds = {
        "A": set(), "B": set(),
        "C": {"A"}, "D": {"A", "B"}, "E": {"B"},
        "F": {"C", "D"}, "G": {"D", "E"}, "H": {"F", "G"},
    }
    return ProjectGraph.from_dict_of_sets(preds, simplify=True)


DISTS_DETERM = {
    "A": ("normal", 3, 1e-9),  # casi constante
    "B": ("normal", 2, 1e-9),
    "C": ("normal", 4, 1e-9),
    "D": ("normal", 5, 1e-9),
    "E": ("normal", 3, 1e-9),
    "F": ("normal", 2, 1e-9),
    "G": ("normal", 6, 1e-9),
    "H": ("normal", 4, 1e-9),
}


class TestMonteCarloResult(unittest.TestCase):
    """Tests unitarios de la clase MonteCarloResult."""

    def setUp(self):
        self.proj = make_project_8act()
        self.result = self.proj.monte_carlo(DISTS_DETERM, n_samples=200, seed=0)

    def test_returns_monte_carlo_result(self):
        self.assertIsInstance(self.result, MonteCarloResult)

    def test_tau_samples_shape(self):
        self.assertEqual(self.result.tau_samples.shape, (200,))

    def test_tau_samples_are_positive(self):
        self.assertTrue(np.all(self.result.tau_samples > 0))

    def test_tau_mean_close_to_deterministic(self):
        """Con distribuciones casi constantes, τ̄ ≈ τ determinista."""
        tau_det = self.proj.tau({"A":3,"B":2,"C":4,"D":5,"E":3,"F":2,"G":6,"H":4})
        self.assertAlmostEqual(self.result.tau_mean, tau_det, delta=0.05)

    def test_tau_std_near_zero_for_degenerate(self):
        self.assertLess(self.result.tau_std, 0.01)

    def test_sensitivity_mean_is_series(self):
        self.assertIsInstance(self.result.sensitivity_mean, pd.Series)

    def test_sensitivity_values_in_unit_interval(self):
        for v in self.result.sensitivity_mean:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_critical_activity_has_high_sensitivity(self):
        """D es siempre crítica en este proyecto — sensibilidad debe ser ≈ 1."""
        self.assertGreater(self.result.sensitivity_mean.get("D", 0), 0.95)

    def test_summary_has_correct_shape(self):
        s = self.result.summary()
        self.assertEqual(s.shape, (1, 7))
        for col in ("mean", "std", "q05", "q50", "q95", "min", "max"):
            self.assertIn(col, s.columns)

    def test_repr_contains_n_and_tau(self):
        r = repr(self.result)
        self.assertIn("n=200", r)
        self.assertIn("τ̄", r)


class TestMonteCarlo(unittest.TestCase):
    """Tests del método ProjectGraph.monte_carlo()."""

    def setUp(self):
        self.proj = make_project_8act()

    def _dists(self, sigma=0.5):
        return {
            "A": ("normal", 3, sigma),
            "B": ("normal", 2, sigma),
            "C": ("normal", 4, sigma),
            "D": ("normal", 5, sigma),
            "E": ("normal", 3, sigma),
            "F": ("normal", 2, sigma),
            "G": ("normal", 6, sigma),
            "H": ("normal", 4, sigma),
        }

    def test_seed_reproducibility(self):
        r1 = self.proj.monte_carlo(self._dists(), n_samples=100, seed=7)
        r2 = self.proj.monte_carlo(self._dists(), n_samples=100, seed=7)
        np.testing.assert_array_equal(r1.tau_samples, r2.tau_samples)

    def test_different_seeds_differ(self):
        r1 = self.proj.monte_carlo(self._dists(), n_samples=100, seed=1)
        r2 = self.proj.monte_carlo(self._dists(), n_samples=100, seed=2)
        self.assertFalse(np.allclose(r1.tau_samples, r2.tau_samples))

    def test_uniform_distribution(self):
        dists = {"A": ("uniform", 2, 4), "B": ("uniform", 1, 3),
                 "C": ("uniform", 3, 5), "D": ("uniform", 4, 6),
                 "E": ("uniform", 2, 4), "F": ("uniform", 1, 3),
                 "G": ("uniform", 5, 7), "H": ("uniform", 3, 5)}
        r = self.proj.monte_carlo(dists, n_samples=500, seed=0)
        self.assertEqual(len(r.tau_samples), 500)

    def test_triangular_distribution(self):
        dists = {k: ("triangular", 2, 3, 5) for k in
                 ["A","B","C","D","E","F","G","H"]}
        r = self.proj.monte_carlo(dists, n_samples=500, seed=0)
        self.assertIsInstance(r, MonteCarloResult)

    def test_lognormal_distribution(self):
        import math
        mu, sigma = math.log(3), 0.2
        dists = {k: ("lognormal", mu, sigma) for k in
                 ["A","B","C","D","E","F","G","H"]}
        r = self.proj.monte_carlo(dists, n_samples=500, seed=0)
        self.assertTrue(np.all(r.tau_samples > 0))

    def test_pert_distribution(self):
        dists = {k: ("pert", 2, 3, 5) for k in
                 ["A","B","C","D","E","F","G","H"]}
        r = self.proj.monte_carlo(dists, n_samples=500, seed=0)
        self.assertIsInstance(r, MonteCarloResult)

    def test_callable_distribution(self):
        rng = np.random.default_rng(0)
        dists = {k: lambda size, _r=rng: _r.normal(3, 0.5, size)
                 for k in ["A","B","C","D","E","F","G","H"]}
        r = self.proj.monte_carlo(dists, n_samples=300, seed=None)
        self.assertEqual(len(r.tau_samples), 300)

    def test_missing_activity_ignored(self):
        """Si falta una actividad en distributions, su duración es 0."""
        dists_partial = {"H": ("normal", 4, 0.1)}  # solo H
        r = self.proj.monte_carlo(dists_partial, n_samples=100, seed=0)
        self.assertIsInstance(r, MonteCarloResult)

    def test_n_samples_stored(self):
        r = self.proj.monte_carlo(self._dists(), n_samples=333, seed=0)
        self.assertEqual(r.n_samples, 333)

    def test_seed_stored(self):
        r = self.proj.monte_carlo(self._dists(), n_samples=50, seed=99)
        self.assertEqual(r.seed, 99)

    def test_tau_quantiles_ordered(self):
        r = self.proj.monte_carlo(self._dists(), n_samples=1000, seed=0)
        self.assertLessEqual(r.tau_q05, r.tau_q50)
        self.assertLessEqual(r.tau_q50, r.tau_q95)

    def test_chain_project_monte_carlo(self):
        """Cadena determinista: τ̄ ≈ suma de medias."""
        proj, acts = make_chain(4)
        dists = {a: ("normal", 3.0, 1e-9) for a in acts}
        r = proj.monte_carlo(dists, n_samples=100, seed=0)
        self.assertAlmostEqual(r.tau_mean, 12.0, delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
