"""
test_extended.py — Cobertura de métodos de visualización, nivelación,
asignación, valor ganado y helpers de display.
=====================================================================
Tests de humo y de propiedades (no de píxeles): comprueban que las
rutas de código se ejecutan, devuelven el tipo esperado y los valores
numéricos que llevan dentro son coherentes.
"""

import os
import sys
import shutil
import tempfile
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import pmule
from pmule import ProjectGraph, PredecessorTable, EarnedValue

PREDS = {
    "A": set(), "B": set(),
    "C": {"A"}, "D": {"A", "B"}, "E": {"B"},
    "F": {"C", "D"}, "G": {"D", "E"}, "H": {"F", "G"},
}
DUR = {"A": 3, "B": 2, "C": 4, "D": 5, "E": 3, "F": 2, "G": 6, "H": 4}
RES = {"A": 2, "B": 1, "C": 3, "D": 2, "E": 1, "F": 2, "G": 3, "H": 1}
LOADS = [3, 3, 3, 6, 6, 5, 5, 2, 5, 5, 3, 3, 3, 3, 1, 1, 1, 1]


def project():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


def data_df():
    return pd.DataFrame({"duration": DUR, "resources": RES})


def _graphviz_available():
    try:
        import pygraphviz  # noqa
    except ImportError:
        return False
    return shutil.which("dot") is not None


# ---------------------------------------------------------------------------
# gantt: variantes de total / acumulado / extra rows-cols / tikz
# ---------------------------------------------------------------------------

class TestGanttVariants(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.data = data_df()

    def test_total_columna(self):
        g, _ = self.pg.gantt(self.data, "duration", "resources", total="columna")
        self.assertIn("Total", g.data.columns)
        self.assertEqual(int(g.data.loc["A", "Total"]), 2 * 3)

    def test_total_ambas(self):
        g, _ = self.pg.gantt(self.data, "duration", "resources", total="ambas")
        self.assertIn("Total", g.data.columns)
        self.assertIn("Total", g.data.index)

    def test_acumulado_fila(self):
        g, _ = self.pg.gantt(self.data, "duration", "resources", total="fila", acumulado=True)
        self.assertIn("Acumulado", g.data.index)
        acc = [int(g.data.loc["Acumulado", p]) for p in range(1, 19)]
        self.assertEqual(acc, list(np.cumsum(LOADS)))

    def test_acumulado_ambas(self):
        g, _ = self.pg.gantt(self.data, "duration", "resources", total="ambas", acumulado=True)
        self.assertIn("Acumulado", g.data.index)
        self.assertEqual(int(g.data.loc["Acumulado", 18]), sum(LOADS))

    def test_cuadrados_ambas(self):
        g, _ = self.pg.gantt(self.data, "duration", "resources", total="ambas", cuadrados=True)
        self.assertIn("Cuadrados", g.data.index)

    def test_extra_rows_and_cols(self):
        extra_rows = pd.DataFrame({p: [p] for p in range(1, 19)}, index=["Periodo"])
        extra_cols = pd.DataFrame({"Nota": {a: "x" for a in DUR}})
        g, _ = self.pg.gantt(self.data, "duration", "resources",
                             extra_rows=extra_rows, extra_cols=extra_cols)
        self.assertIn("Periodo", g.data.index)
        self.assertIn("Nota", g.data.columns)

    def test_tikz_with_holguras_and_total(self):
        _, dibujo = self.pg.gantt(self.data, "duration", "resources",
                                  total="fila", holguras=True, tikz=True)
        self.assertIn(r"\begin{tikzpicture}", dibujo)
        self.assertIn(r"\end{tikzpicture}", dibujo)
        for a in DUR:
            self.assertIn(a, dibujo)

    def test_tikz_with_extra_rows_cols(self):
        extra_rows = pd.DataFrame({p: [p] for p in range(1, 19)}, index=["Periodo"])
        extra_cols = pd.DataFrame({"Nota": {a: 1 for a in DUR}})
        _, dibujo = self.pg.gantt(self.data, "duration", "resources", tikz=True,
                                  total="ambas", extra_rows=extra_rows, extra_cols=extra_cols)
        self.assertIn("Periodo", dibujo)
        self.assertIn("Nota", dibujo)

    def test_tikz_params_colors(self):
        params = {"critical_background_color": "red", "regular_background_color": "blue"}
        _, dibujo = self.pg.gantt(self.data, "duration", "resources", tikz=True, params=params)
        self.assertIn("red", dibujo)

    def test_zero_duration_activity_excluded(self):
        d = self.data.copy(); d.loc["C", "duration"] = 0
        g, _ = self.pg.gantt(d, "duration", "resources")
        self.assertNotIn("C", g.data.index)

    def test_gantt_cargas_tikz(self):
        g, dibujo = self.pg.gantt_cargas(self.data, "duration", "resources",
                                         report=False, tikz=True)
        self.assertIn("tikzpicture", dibujo)
        self.assertEqual(float(g.data.loc["Cuadrados", "H_total"]), 243)

    def test_gantt_cargas_report_prints(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(pmule, "display"):
            self.pg.gantt_cargas(self.data, "duration", "resources", report=True, tikz=False)
        self.assertIn("243", buf.getvalue())


# ---------------------------------------------------------------------------
# gantt: flechas de prelación (params={"arrows": True})
# ---------------------------------------------------------------------------

# Predecesoras inmediatas del proyecto de referencia → 10 pares
PRECEDENCES = [("A", "C"), ("A", "D"), ("B", "D"), ("B", "E"), ("C", "F"),
               ("D", "F"), ("D", "G"), ("E", "G"), ("F", "H"), ("G", "H")]


class TestGanttArrows(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.data = data_df()

    def _tikz(self, pg=None, data=None, **params):
        _, dibujo = (pg or self.pg).gantt(data if data is not None else self.data,
                                          "duration", "resources", tikz=True,
                                          total="fila", holguras=True, params=params)
        return dibujo

    def test_default_has_no_arrows(self):
        self.assertEqual(self._tikz().count("\\draw[->"), 0)

    def test_default_output_unchanged_by_feature(self):
        """Sin el parámetro, el TikZ es byte a byte el mismo que con arrows=False."""
        self.assertEqual(self._tikz(), self._tikz(arrows=False))

    def test_one_arrow_per_precedence(self):
        tikz = self._tikz(arrows=True)
        self.assertEqual(tikz.count("\\draw[->"), len(PRECEDENCES))

    def test_precedences_helper(self):
        self.assertEqual(self.pg._gantt_precedences(set(DUR)), PRECEDENCES)

    def test_critical_arrows_colored(self):
        """A→D, D→G, G→H: ambas críticas y sin hueco → color crítico."""
        tikz = self._tikz(arrows=True)
        self.assertEqual(tikz.count("red!70!black"), 3)

    def test_gap_arrows_have_elbow_and_adjacent_enter_from_edge(self):
        tikz = self._tikz(arrows=True)
        arrows = [l for l in tikz.splitlines() if "\\draw[->" in l]
        # con hueco: 4 puntos (sale, codo, codo, entra); sin hueco: 3 puntos
        n_pts = [l.count("--") + 1 for l in arrows]
        self.assertEqual(sorted(set(n_pts)), [3, 4])

    def test_arrows_drawn_before_bars(self):
        tikz = self._tikz(arrows=True)
        self.assertLess(tikz.index("%Flechas de prelaci"), tikz.index("%Dibujo de las actividades"))

    def test_style_params(self):
        tikz = self._tikz(arrows=True, arrow_width="2pt", arrow_color="blue", arrow_corner="0pt")
        self.assertIn("line width=2pt", tikz)
        self.assertIn("->, blue,", tikz)
        self.assertIn("rounded corners=0pt", tikz)

    def test_zero_duration_predecessor_is_bridged(self):
        """C con duración 0 no tiene barra: F hereda la flecha desde A (predecesora de C)."""
        d = self.data.copy(); d.loc["C", "duration"] = 0
        present = set(d.index[d["duration"] > 0])
        pairs = self.pg._gantt_precedences(present)
        self.assertNotIn(("C", "F"), pairs)
        self.assertIn(("A", "F"), pairs)
        self.assertEqual(len(pairs), len(PRECEDENCES) - 1)     # A→C y C→F → A→F
        self.assertEqual(self._tikz(data=d, arrows=True).count("\\draw[->"), len(pairs))

    def test_after_desplazar_wait_bar_receives_arrow_and_no_same_row_arrow(self):
        pg = project()
        new_data, _, _ = pg.desplazar(self.data, "duration", "resources", report=False, C=1)
        _, tikz = pg.gantt(new_data, "duration", "resources", tikz=True, params={"arrows": True})
        pairs = pg._gantt_precedences(set(new_data.index))
        self.assertIn(("A", "💤C"), pairs)
        self.assertIn(("💤C", "C"), pairs)                 # existe como prelación…
        self.assertEqual(tikz.count("\\draw[->"), len(pairs) - 1)   # …pero no se dibuja (misma fila)

    def test_gantt_cargas_and_nivelar_propagate_params(self):
        _, tikz = self.pg.gantt_cargas(self.data, "duration", "resources", report=False,
                                       tikz=True, params={"arrows": True})
        self.assertGreater(tikz.count("\\draw[->"), 0)


# ---------------------------------------------------------------------------
# desplazar / evaluar_* / nivelar
# ---------------------------------------------------------------------------

class TestDesplazar(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.data = data_df()

    def test_shift_adds_wait_activity(self):
        new_data, g, _ = self.pg.desplazar(self.data, "duration", "resources",
                                            report=False, C=2)
        self.assertIn("💤C", new_data.index)
        self.assertEqual(new_data.loc["💤C", "duration"], 2)
        self.assertEqual(new_data.loc["💤C", "resources"], 0)

    def test_shift_delays_start_in_gantt(self):
        new_data, g, _ = self.pg.desplazar(self.data, "duration", "resources",
                                            report=False, C=2)
        occupied = [p for p in range(1, 19) if g.data.loc["C", p] not in ("", 0)]
        self.assertEqual(occupied[0], 3 + 2 + 1)

    def test_shift_twice_accumulates(self):
        d1, _, _ = self.pg.desplazar(self.data, "duration", "resources", report=False, C=1)
        d2, _, _ = self.pg.desplazar(d1, "duration", "resources", report=False, C=1)
        self.assertEqual(d2.loc["💤C", "duration"], 2)
        self.assertEqual(self.pg.activities.count("💤C"), 1)

    def test_missing_resource_label_raises(self):
        with self.assertRaises(ValueError):
            self.pg.desplazar(self.data, "duration", "nope", report=False, C=1)

    def test_cuadrados_false_branch(self):
        new_data, g, _ = self.pg.desplazar(self.data, "duration", "resources",
                                            report=False, cuadrados=False, C=1)
        self.assertNotIn("Cuadrados", g.data.index)

    def test_tikz_branch(self):
        _, _, dibujo = self.pg.desplazar(self.data, "duration", "resources",
                                         report=False, tikz=True, C=1)
        self.assertIn("tikzpicture", dibujo)

    def test_project_duration_unchanged_within_float(self):
        self.pg.desplazar(self.data, "duration", "resources", report=False, C=5)
        d = dict(DUR); d["💤C"] = 5
        self.assertEqual(self.pg.duration(d), 18)

    def test_evaluar_desplazamiento_does_not_mutate(self):
        n0 = len(self.pg.nodes)
        carga, dibujo = self.pg.evaluar_desplazamiento(self.data, "duration", "resources",
                                                        report=False, tikz=False, C=1)
        self.assertEqual(len(self.pg.nodes), n0)
        self.assertIsInstance(float(carga), float)

    def test_evaluar_desplazamiento_value(self):
        """Desplazar C 1 periodo: cargas 4..7 → 3,6,6,5,5,5 ... recalculado."""
        carga, _ = self.pg.evaluar_desplazamiento(self.data, "duration", "resources",
                                                   report=False, tikz=False, C=1)
        loads = LOADS[:]
        loads[3] -= 3; loads[7] += 3        # C pasa de 4-7 a 5-8
        self.assertEqual(float(carga), sum(x * x for x in loads))

    def test_evaluar_rango_shape_and_first_value(self):
        sq, _ = self.pg.evaluar_rango_de_desplazamientos(
            self.data, "duration", "resources", "C", report=False, tikz=False)
        self.assertEqual(list(sq.index), list(range(0, 6)))   # holgura C = 5
        self.assertEqual(float(sq.loc[0, "Suma_de_cuadrados"]), 243)

    def test_evaluar_rango_report_prints(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(pmule, "display"):
            self.pg.evaluar_rango_de_desplazamientos(
                self.data, "duration", "resources", "E", report=True, tikz=False)
        self.assertIn("Desplazamiento", buf.getvalue())

    def test_nivelar_reduces_sum_of_squares(self):
        pg = project()
        desp, cuadrados, my_data, g, dibujo = pg.nivelar(self.data, "duration", "resources")
        self.assertTrue((desp["desplazamientos"] >= 0).all())
        self.assertLessEqual(float(g.data.loc["Cuadrados", "H_total"]), 243)
        self.assertTrue(hasattr(cuadrados, "data"))

    def test_nivelar_only_noncritical_considered(self):
        desp, *_ = project().nivelar(self.data, "duration", "resources")
        self.assertEqual(set(desp.index), {"B", "C", "E", "F"})

    def test_nivelar_report(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            project().nivelar(self.data, "duration", "resources", report=True)
        self.assertIn("Desplazamiento óptimo", buf.getvalue())

    def test_nivelar_all_critical_chain(self):
        pg = ProjectGraph.from_dict_of_sets({"A": set(), "B": {"A"}, "C": {"B"}})
        data = pd.DataFrame({"duration": {"A": 1, "B": 2, "C": 3},
                             "resources": {"A": 1, "B": 1, "C": 1}})
        desp, *_ = pg.nivelar(data, "duration", "resources")
        self.assertEqual(len(desp), 0)


# ---------------------------------------------------------------------------
# asignar
# ---------------------------------------------------------------------------

class TestAsignar(unittest.TestCase):

    def setUp(self):
        self.data = data_df()

    def test_requires_maximo(self):
        with self.assertRaises(ValueError):
            project().asignar(self.data, "duration", "resources")

    def test_bad_labels(self):
        with self.assertRaises(ValueError):
            project().asignar(self.data, "dur", "resources", maximo=5)
        with self.assertRaises(ValueError):
            project().asignar(self.data, "duration", "res", maximo=5)

    def test_scalar_limit_respected(self):
        out = project().asignar(self.data, "duration", "resources", maximo=4, report=False)
        self.assertIsInstance(out, tuple)
        gantt = next(o for o in out if hasattr(o, "data") and "Total" in getattr(o.data, "index", []))
        totals = gantt.data.loc["Total"].drop("H_total", errors="ignore")
        totals = pd.to_numeric(totals, errors="coerce").dropna()
        self.assertTrue((totals <= 4).all(), totals.to_dict())

    def test_limit_large_enough_keeps_duration(self):
        out = project().asignar(self.data, "duration", "resources", maximo=6, report=False)
        gantt = next(o for o in out if hasattr(o, "data") and "Total" in getattr(o.data, "index", []))
        periods = [c for c in gantt.data.columns if isinstance(c, (int, np.integer))]
        self.assertEqual(len(periods), 18)

    def test_series_limit(self):
        lim = pd.Series(5, index=range(1, 40))
        out = project().asignar(self.data, "duration", "resources", maximo=lim, report=False)
        self.assertIsInstance(out, tuple)

    def test_report_displays(self):
        with mock.patch.object(pmule, "display") as disp:
            project().asignar(self.data, "duration", "resources", maximo=5, report=True)
        self.assertTrue(disp.called)


# ---------------------------------------------------------------------------
# pert() / roy() con graphviz
# ---------------------------------------------------------------------------

@unittest.skipUnless(_graphviz_available(), "pygraphviz/dot no disponibles")
class TestGraphvizPlots(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pert_svg(self):
        out = self.pg.pert(os.path.join(self.tmp, "p.svg"), durations=DUR)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "p.svg")))
        self.assertIn("svg", type(out).__name__.lower())

    def test_pert_svg_without_durations(self):
        out = self.pg.pert(os.path.join(self.tmp, "p0.svg"))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "p0.svg")))

    def test_pert_png(self):
        out = self.pg.pert(os.path.join(self.tmp, "p.png"), durations=pd.Series(DUR))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "p.png")))

    def test_pert_pdf_creates_svg_too(self):
        self.pg.pert(os.path.join(self.tmp, "p.pdf"), durations=DUR)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "p.svg")))

    def test_pert_default_filename(self):
        cwd = os.getcwd(); os.chdir(self.tmp)
        try:
            self.pg.pert(durations=DUR)
            self.assertTrue(os.path.exists("output_pert_figure.svg"))
        finally:
            os.chdir(cwd)

    def test_roy_png(self):
        out = self.pg.roy(os.path.join(self.tmp, "r.png"), durations=dict(DUR))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "r.png")))

    def test_roy_without_durations(self):
        self.pg.roy(os.path.join(self.tmp, "r0.png"))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "r0.png")))

    def test_pert_after_desplazar_shows_wait_activity(self):
        self.pg.desplazar(data_df(), "duration", "resources", report=False, C=1)
        d = dict(DUR); d["💤C"] = 1
        self.pg.pert(os.path.join(self.tmp, "w.svg"), durations=d)
        self.assertIn("💤C", open(os.path.join(self.tmp, "w.svg"), encoding="utf8").read())


class TestLazyGraphvizImport(unittest.TestCase):

    def test_import_error_message(self):
        import builtins
        real_import = builtins.__import__
        def fake(name, *a, **k):
            if name == "pygraphviz":
                raise ImportError("no")
            return real_import(name, *a, **k)
        with mock.patch("builtins.__import__", side_effect=fake):
            with self.assertRaises(ImportError) as cm:
                project().pert("x.svg", durations=DUR)
            self.assertIn("pygraphviz", str(cm.exception))
            with self.assertRaises(ImportError):
                project().roy("x.png")


# ---------------------------------------------------------------------------
# EarnedValue
# ---------------------------------------------------------------------------

class TestEarnedValue(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.data = pd.DataFrame({
            "planned": DUR,
            "actual": {"A": 3, "B": 3, "C": 4, "D": 6, "E": 3, "F": 2, "G": 6, "H": 4},
            "PV": {a: 100 * DUR[a] for a in DUR},
            "AC": {a: 110 * DUR[a] for a in DUR},
            "pct": {"A": 1, "B": 1, "C": 1, "D": 0.5, "E": 1, "F": 0, "G": 0, "H": 0},
        })
        self.ev = EarnedValue(self.pg)

    def test_returns_expected_keys(self):
        out = self.ev.calcula_gantts(self.data, "planned", "actual", "PV", "AC", "pct")
        for k in ("Gantt_PV", "Gantt_AC", "Gantt_EV", "acumulados"):
            self.assertIn(k, out)

    def test_pv_cumulative_equals_budget(self):
        out = self.ev.calcula_gantts(self.data, "planned", "actual", "PV", "AC", "pct")
        total_pv = self.data["PV"].sum()
        self.assertAlmostEqual(float(out["acumulados"]["PV"].iloc[-1]), total_pv)

    def test_ev_le_pv_and_ac_cumulative_monotone(self):
        out = self.ev.calcula_gantts(self.data, "planned", "actual", "PV", "AC", "pct")
        acc = out["acumulados"].astype(float)
        self.assertTrue((acc.diff().dropna() >= -1e-9).all().all())
        self.assertLessEqual(acc["EV"].iloc[-1], acc["PV"].iloc[-1] + 1e-9)

    def test_percentage_over_one_is_rescaled(self):
        d = self.data.copy(); d["pct"] = d["pct"] * 100
        out1 = self.ev.calcula_gantts(d, "planned", "actual", "PV", "AC", "pct")
        out2 = self.ev.calcula_gantts(self.data, "planned", "actual", "PV", "AC", "pct")
        pd.testing.assert_frame_equal(out1["acumulados"].astype(float),
                                      out2["acumulados"].astype(float))

    def test_tikz_drawings(self):
        out = self.ev.calcula_gantts(self.data, "planned", "actual", "PV", "AC", "pct", tikz=True)
        self.assertTrue(any("tikz" in str(v) for v in out.values() if isinstance(v, str)))

    def test_plot_curvas_valor_ganado(self):
        import matplotlib; matplotlib.use("Agg")
        out = self.ev.calcula_gantts(self.data, "planned", "actual", "PV", "AC", "pct")
        with mock.patch("matplotlib.pyplot.show"):
            pmule.plot_curvas_valor_ganado(out)


# ---------------------------------------------------------------------------
# Helpers de display / matplotlib / tikz
# ---------------------------------------------------------------------------

class TestDisplayHelpers(unittest.TestCase):

    def setUp(self):
        self.pg = project()

    def test_display_path_matrix(self):
        st = self.pg.display_path_matrix(dummies=False)
        self.assertTrue(hasattr(st, "data"))
        self.assertEqual(st.data.shape, (6, 8))

    def test_display_linkage_matrices(self):
        pt = PredecessorTable.from_dict_of_sets(PREDS)
        self.assertTrue(hasattr(pt.display_immediate_linkage_matrix(), "data"))
        self.assertTrue(hasattr(pt.display_distant_linkage_matrix(), "data"))

    def test_display_critical_path(self):
        with mock.patch.object(pmule, "display") as disp:
            self.pg.display_critical_path(DUR)
        self.assertEqual(disp.call_count, 1)

    def test_ordena_rutas(self):
        R = self.pg.path_matrix(dummies=False)
        imp_act = np.arange(1, 9)
        imp_rutas = np.arange(6)
        st = pmule.ordena_rutas(R.astype(float), imp_act, imp_rutas)
        self.assertEqual(list(st.data["Importancia"]), [5, 4, 3, 2, 1, 0])

    def test_highlight_blue_red_dataframe_and_array(self):
        df = pd.DataFrame([[1, -1], [0, 1]])
        self.assertTrue(hasattr(pmule.highlight_blue_red(df), "data"))
        self.assertTrue(hasattr(pmule.highlight_blue_red(np.array([[1, -1]])), "data"))

    def test_beautify(self):
        out = pmule.beautify(np.array([[1.0, 2.0]]), 3)
        self.assertEqual(type(out).__name__, "Math")
        self.assertIn("array", out.data)
        self.assertIn("3", out.data)

    def test_pretty_variants(self):
        R = self.pg.path_matrix(dummies=False).values
        self.assertIsInstance(pmule.pretty(R, dec=0, latex=True), str)
        self.assertIsInstance(pmule.pretty(R.astype(float) / 3, dec=2, latex=True), str)
        pmule.pretty(R, dec=0, latex=False)   # ruta jupyter (display)

    def test_colorea_minimo_fila(self):
        df = pd.DataFrame([[3, 1, 2], [5, 5, 4]])
        st = pmule.colorea_minimo_fila(df)
        self.assertTrue(hasattr(st, "data"))

    def test_plot_cargas_scalar_and_vector_limit(self):
        import matplotlib; matplotlib.use("Agg")
        g, _ = self.pg.gantt_cargas(data_df(), "duration", "resources", report=False)
        with mock.patch("matplotlib.pyplot.show"):
            pmule.plot_cargas(g)
            pmule.plot_cargas(g, limite_maximo_recursos=4)
            pmule.plot_cargas(g, limite_maximo_recursos=[4, 5])          # se extiende
            pmule.plot_cargas(g, limite_maximo_recursos=np.full(30, 5))  # se recorta

    def test_pinta_tikz_calls_texfragment(self):
        with mock.patch.object(pmule, "TexFragment") as TF, mock.patch.object(pmule, "display"):
            pmule.pinta_tikz(r"\begin{tikzpicture}\end{tikzpicture}")
            pmule.pinta_tikz(r"\begin{tikzpicture}\end{tikzpicture}", filename="f")
            self.assertEqual(TF.call_count, 2)
            TF.return_value.run_latex.assert_any_call(save_pdf="f.pdf", save_svg="f.svg")

    def test_plot_tikz_with_path_compiles_via_texfragment(self):
        with mock.patch.object(pmule, "TexFragment") as TF, mock.patch.object(pmule, "display"):
            self.pg.plot_tikz(durations=DUR, path="/tmp/x.pdf")
            TF.return_value.run_latex.assert_called_once_with(save_pdf="/tmp/x.pdf")

    def test_latex_array_mimebundle(self):
        arr = pmule.LatexArray(np.array([[1, 2], [3, 4]]))
        data, _meta = arr._repr_mimebundle_()
        self.assertIn("text/latex", data)
        self.assertIn("bmatrix", data["text/latex"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
