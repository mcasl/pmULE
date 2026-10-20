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



class TestVisoresTiempos(unittest.TestCase):
    """Visores HTML con botones de los tiempos del PERT y del Roy (fotogramas TikZ → SVG)."""

    @classmethod
    def setUpClass(cls):
        import shutil
        if not (shutil.which("pdflatex") and shutil.which("pdftocairo")):
            raise unittest.SkipTest("hace falta pdflatex y pdftocairo")
        cls.pg = project()

    def _fotos(self, html):
        return html.count('class="pmule-paso"')

    def test_pert_visor(self):
        html = self.pg.pert_pasos_html(DUR)
        n = len(self.pg.nodes)
        m = len(self.pg.pert_graph.edges)
        total = 2 * n + m + 2                   # vacío + n + n + m holguras + final
        self.assertEqual(self._fotos(html), total)
        self.assertEqual(html.count("data:image/svg+xml;base64,"), total)
        self.assertIn("siguiente", html)

    def test_roy_visor(self):
        html = self.pg.roy_pasos_html(DUR)
        n = len(DUR) + 2                                    # actividades + Inicio + Fin
        self.assertEqual(self._fotos(html), 2 * n + 2)
        self.assertEqual(html.count("data:image/svg+xml;base64,"), 2 * n + 2)

    def test_svgs_validos(self):
        import base64, re
        html = self.pg.pert_pasos_html(DUR)
        for b64 in re.findall(r'base64,([A-Za-z0-9+/=]+)"', html):
            self.assertIn("<svg", base64.b64decode(b64).decode("utf8"))

    def test_display_sin_visor_conserva_la_lista(self):
        self.pg.display_pert_pasos(DUR, visor=False)
        self.pg.display_roy_pasos(DUR, visor=False)


class TestTrazadoPropio(unittest.TestCase):
    """Los fotogramas TikZ usan la disposición propia (sin graphviz) y sin colisiones."""

    @classmethod
    def setUpClass(cls):
        cls.pg = project()

    def test_pert_curvas_y_ficticias_con_la_notacion_de_pert(self):
        fotos = self.pg.plot_tikz_pasos(DUR)
        self.assertIn(".. controls", fotos[-1])
        self.assertIn(r"@3$\to$5 (0)", fotos[-1])
        self.assertIn(r"@7$\to$8 (0)", fotos[-1])

    def test_pert_layout_capas_conserva_el_dibujo_anterior(self):
        fotos = self.pg.plot_tikz_pasos(DUR, layout="capas")
        self.assertNotIn(".. controls", fotos[-1])
        self.assertEqual(len(fotos), 2 * len(self.pg.nodes) + len(self.pg.pert_graph.edges) + 2)

    def test_roy_curvas(self):
        fotos = self.pg.plot_tikz_roy_pasos(DUR)
        self.assertIn(".. controls", fotos[-1])
        self.assertNotIn(".. controls", self.pg.plot_tikz_roy_pasos(DUR, layout="capas")[-1])

    def test_los_nodos_no_se_solapan(self):
        tr = self.pg._trazado_pert(dict(DUR))
        pts = list(tr["pos"].values())
        for i, (x1, y1) in enumerate(pts):
            for x2, y2 in pts[i + 1:]:
                self.assertGreater(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5, 1.1)


class TestDisposicionPropia(unittest.TestCase):
    """Sin graphviz: la disposición propia no pisa nodos con flechas ni con nombres."""

    PRE = {"A": "", "B": "", "C": "", "D": "A", "E": "A,B", "F": "B,C", "G": "D,E", "H": "E,F",
           "I": "C", "J": "G,H", "K": "G,I", "L": "H,I", "M": "J,K", "N": "K,L"}

    def _proyecto_denso(self):
        from pmule import PredecessorTable
        pg = PredecessorTable.from_dict_of_strings(self.PRE).create_project()
        return pg, {k: 2 + i % 7 for i, k in enumerate(self.PRE)}

    def test_no_lanza_procesos(self):
        import subprocess
        orig = subprocess.run
        subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe lanzar procesos"))
        try:
            project().plot_tikz_pasos(DUR)
            project().plot_tikz_roy_pasos(DUR)
        finally:
            subprocess.run = orig

    def test_ninguna_flecha_ni_nombre_pisa_un_nodo(self):
        import pmule
        pg, dur = self._proyecto_denso()
        tr = pg._trazado_pert(dict(dur))
        for (u, v), inter in tr["rutas"].items():
            pts = [tr["pos"][u]] + inter + [tr["pos"][v]]
            for q in pmule._muestras_ruta(pts, tr["curva"][(u, v)], 30):
                for n, (cx, cy) in tr["pos"].items():
                    if n not in (u, v):
                        self.assertFalse(abs(q[0] - cx) < 0.55 and abs(q[1] - cy) < 0.55,
                                         f"la flecha {u}->{v} pisa el nodo {n}")
        for (u, v), (lx, ly) in tr["etiquetas"].items():
            for n, (cx, cy) in tr["pos"].items():
                self.assertFalse(abs(lx - cx) < 0.85 and abs(ly - cy) < 0.72,
                                 f"el nombre de {u}->{v} pisa el nodo {n}")

    def test_roy_denso(self):
        pg, dur = self._proyecto_denso()
        self.assertEqual(len(pg.plot_tikz_roy_pasos(dur)), 2 * (len(dur) + 2) + 2)


class TestPertRoyTikzPorDefecto(unittest.TestCase):
    """pert() y roy() dibujan con TikZ por defecto (motor="tikz"), sin graphviz."""

    @classmethod
    def setUpClass(cls):
        import shutil
        if not (shutil.which("pdflatex") and shutil.which("pdftocairo")):
            raise unittest.SkipTest("hace falta pdflatex y pdftocairo")

    def test_pert_devuelve_svg_y_guarda_pdf(self):
        import tempfile, os
        d = tempfile.mkdtemp()
        out = project().pert(os.path.join(d, "p.pdf"), durations=DUR, ranksep=1, nodesep=1)
        self.assertIn("<svg", out.data)
        self.assertTrue(os.path.exists(os.path.join(d, "p.pdf")))

    def test_roy_devuelve_svg(self):
        self.assertIn("<svg", project().roy(durations=dict(DUR), ranksep=1.2).data)
        self.assertIn("<svg", project().roy().data)

    def test_critico_en_rojo_brillante_con_nodos_rellenos(self):
        cod = project().plot_tikz_pert(DUR)
        self.assertIn("fill=white!90!red, draw=red, very thick", cod)
        self.assertIn("red, very thick", cod)
        self.assertNotIn("red!75!black", cod)
        cod = project().plot_tikz_roy(DUR)
        self.assertIn("fill=white!90!red, draw=red, very thick", cod)
        self.assertNotIn("red!75!black", cod)

    def test_sin_duraciones_no_hay_tiempos_ni_rojo(self):
        cod = project().plot_tikz_pert(False)
        self.assertNotIn("red", cod)
        self.assertIn("green!50!black", cod)           # ficticias
        self.assertNotIn("(0)", cod)

    def test_actividad_de_espera_con_emoji_compila(self):
        out = project().pert(durations=dict(DUR, **{"💤C": 1}))   # warn por clave desconocida
        self.assertIn("<svg", out.data)

    def test_motor_graphviz_se_conserva(self):
        with self.assertRaises(ValueError):
            project().pert(motor="otro")


if __name__ == "__main__":
    unittest.main()


class TestSvgAnchoColumna(unittest.TestCase):
    def test_quita_width_height_y_conserva_viewbox(self):
        import pmule
        svg = '<svg xmlns="x" width="300pt" height="100pt" viewBox="0 0 300 100"><rect width="3" height="4"/></svg>'
        out = pmule._svg_ancho_columna(svg)
        raiz = out[:out.index(">")]
        self.assertNotIn("width=", raiz)
        self.assertNotIn("height=", raiz)
        self.assertIn('viewBox="0 0 300 100"', raiz)
        self.assertIn('<rect width="3" height="4"/>', out)
