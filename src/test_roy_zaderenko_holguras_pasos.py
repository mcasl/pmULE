"""
test_roy_zaderenko_holguras_pasos.py — Roy, Zaderenko y holguras paso a paso
============================================================================
Proyecto de la lección 11 de los apuntes (nave industrial): duración 31,
camino crítico A-D-F-J-I.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_pert_pasos import DUR, project


class TestRoyPasos(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.pasos = self.pg.roy_pasos(DUR)
        self.cal = self.pg.calendar(DUR)

    def test_minimos_y_maximos_coinciden_con_el_calendario(self):
        mn = dict(zip(self.pasos["minimos"]["nodo"], self.pasos["minimos"]["valor"]))
        mx = dict(zip(self.pasos["maximos"]["nodo"], self.pasos["maximos"]["valor"]))
        for act in DUR:
            self.assertEqual(mn[act], self.cal.loc[act, "inicio_mas_temprano"])
            self.assertEqual(mx[act], self.cal.loc[act, "inicio_mas_tardio"])
        self.assertEqual(mn["finish"], 31)
        self.assertEqual(mx["start"], 0)

    def test_calculo_escrito(self):
        calc = dict(zip(self.pasos["maximos"]["nodo"], self.pasos["maximos"]["calculo"]))
        self.assertEqual(calc["C"], "TC* = mín{26 - 7, 10 - 7} = 3")
        lat = self.pg.roy_pasos(DUR, formato="latex")["minimos"]
        self.assertIn(r"T_{\mathrm{Inicio}} = 0", lat["calculo"].iloc[0])

    def test_fotogramas(self):
        n = len(self.pasos["minimos"])
        frames = self.pg.plot_tikz_roy_pasos(DUR)
        self.assertEqual(len(frames), 2 * n + 2)
        self.assertIn("duración = 31", frames[-1])
        c = self.pg.plot_tikz_roy_pasos(DUR, beamer=True, etiquetas={"start": "Start"})
        self.assertIn(rf"\only<{2 * n + 2}>", c)
        self.assertIn("{Start}", c)


class TestZaderenkoPasos(unittest.TestCase):

    def test_fotogramas_y_colores(self):
        pg = project()
        fotos = pg.zaderenko_pasos(DUR)
        n = len(pg.nodes)
        self.assertEqual(len(fotos), 2 * n + 2)
        self.assertNotIn("textcolor", fotos[0])
        self.assertIn("red!75!black", fotos[2])
        self.assertIn("31", fotos[-1])
        b = pg.zaderenko_pasos_beamer(DUR, textos={"columna": "Column"})
        self.assertIn(rf"\only<{2 * n + 2}>", b)
        self.assertIn("Column", b)


class TestHolgurasPasos(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.h = self.pg.holguras_pasos(DUR)

    def test_valores(self):
        self.assertEqual(list(self.h.index), sorted(DUR))
        self.assertEqual(tuple(self.h.loc["E", ["HT", "HL", "HI"]]), (17, 0, -1))
        self.assertEqual(tuple(self.h.loc["B", ["HT", "HL", "HI"]]), (7, 6, 6))
        ref = self.pg.calculate_pert(DUR)["activities"]["H_total"]
        for act in DUR:
            self.assertEqual(self.h.loc[act, "HT"], ref[act])

    def test_ficticias_y_fotogramas(self):
        self.assertGreater(len(self.pg.holguras_pasos(DUR, ficticias=True)), len(self.h))
        frames = self.pg.plot_tikz_holguras_pasos(DUR)
        self.assertEqual(len(frames), len(self.h) + 2)
        self.assertIn("H^{T}", frames[1])


class TestZaderenkoHtml(unittest.TestCase):

    def test_visor_html(self):
        pg = project()
        h = pg.zaderenko_pasos_html(DUR)
        n = len(pg.nodes)
        self.assertEqual(h.count('class="pmule-paso"'), 2 * n + 2)
        self.assertIn("pmulePaso", h)
        self.assertEqual(h, pg.zaderenko_pasos_html(DUR))      # determinista


class TestConstruccion(unittest.TestCase):

    def setUp(self):
        self.pg = project()

    def test_orden(self):
        orden = self.pg.orden_construccion()
        self.assertEqual(sorted(orden), sorted(DUR))
        self.assertLess(orden.index("A"), orden.index("C"))
        self.assertLess(orden.index("F"), orden.index("G"))

    def test_fotogramas(self):
        n = len(DUR)
        pert = self.pg.plot_tikz_construccion_pert()
        roy = self.pg.plot_tikz_construccion_roy()
        self.assertEqual(len(pert), n + 2)
        self.assertEqual(len(roy), n + 2)
        self.assertIn("dashed", pert[-1])          # ficticias en el grafo final
        m = self.pg.matriz_prelaciones_beamer()
        self.assertIn(rf"\only<{n + 1}>", m)


if __name__ == "__main__":
    unittest.main()
