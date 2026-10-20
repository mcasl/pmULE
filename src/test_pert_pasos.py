"""
test_pert_pasos.py — Cálculo y dibujo del PERT paso a paso (uso docente)
========================================================================
Proyecto de la lección 11 de los apuntes (nave industrial): duración 31,
camino crítico A-D-F-J-I.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from pmule import ProjectGraph

PREDS = {
    "A": set(), "B": set(), "C": {"A"}, "D": {"A"}, "E": {"B", "C"},
    "F": {"B", "C", "D"}, "G": {"F"}, "H": {"E"}, "I": {"G", "J"}, "J": {"F"},
}
DUR = {"A": 2, "B": 3, "C": 7, "D": 8, "E": 3, "F": 9, "G": 8, "H": 2,
       "I": 2, "J": 10}


def project():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


class TestPertPasos(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.pasos = self.pg.pert_pasos(DUR)
        self.ref = self.pg.calculate_pert(DUR)["nodes"]

    def test_un_paso_por_suceso_en_cada_recorrido(self):
        n = len(self.pg.nodes)
        self.assertEqual(len(self.pasos["tempranos"]), n)
        self.assertEqual(len(self.pasos["tardios"]), n)
        self.assertEqual(list(self.pasos["tempranos"].index), list(range(1, n + 1)))

    def test_valores_coinciden_con_calculate_pert(self):
        for clave, col in (("tempranos", "early"), ("tardios", "late")):
            serie = self.pasos[clave].set_index("suceso")["valor"]
            for nodo in self.pg.nodes:
                self.assertEqual(serie[nodo], self.ref.loc[nodo, col], (clave, nodo))

    def test_orden_de_los_recorridos(self):
        self.assertEqual(list(self.pasos["tempranos"]["suceso"]), self.pg.nodes)
        self.assertEqual(list(self.pasos["tardios"]["suceso"]), self.pg.nodes[::-1])

    def test_extremos(self):
        primero = self.pasos["tempranos"].iloc[0]
        self.assertEqual(primero["valor"], 0)
        self.assertEqual(primero["calculo"], f"t{primero['suceso']} = 0")
        ultimo = self.pasos["tempranos"].iloc[-1]
        self.assertEqual(ultimo["valor"], 31)
        fin = self.pasos["tardios"].iloc[0]
        self.assertEqual(fin["calculo"], f"t{fin['suceso']}* = t{fin['suceso']} = 31")
        self.assertEqual(self.pasos["tardios"].iloc[-1]["valor"], 0)

    def test_terminos_y_texto(self):
        for _, fila in self.pasos["tempranos"].iloc[1:].iterrows():
            self.assertEqual(fila["valor"], max(t for _, t in fila["terminos"]))
            if len(fila["terminos"]) > 1:
                self.assertIn("máx{", fila["calculo"])
        for _, fila in self.pasos["tardios"].iloc[1:].iterrows():
            self.assertEqual(fila["valor"], min(t for _, t in fila["terminos"]))
            if len(fila["terminos"]) > 1:
                self.assertIn("mín{", fila["calculo"])

    def test_suceso_de_F_tiene_temprano_10(self):
        # El suceso en que empieza F es máx{2+8, 9+0} = 10 (D y la ficticia)
        fila = self.pasos["tempranos"].loc[self.pasos["tempranos"]["valor"] == 10].iloc[0]
        self.assertEqual(len(fila["terminos"]), 2)
        self.assertIn("2 + 8", fila["calculo"])
        self.assertIn("9 + 0", fila["calculo"])

    def test_formato_latex(self):
        pasos = self.pg.pert_pasos(DUR, formato="latex")
        textos = " ".join(pasos["tempranos"]["calculo"]) + " ".join(pasos["tardios"]["calculo"])
        self.assertIn(r"\max\{", textos)
        self.assertIn(r"\min\{", textos)
        self.assertIn("^{*}", textos)

    def test_formato_invalido(self):
        with self.assertRaises(ValueError):
            self.pg.pert_pasos(DUR, formato="html")

    def test_duraciones_en_series_y_decimales(self):
        dur = pd.Series({k: v / 2 for k, v in DUR.items()})
        pasos = self.pg.pert_pasos(dur)
        self.assertAlmostEqual(pasos["tempranos"].iloc[-1]["valor"], 15.5)
        self.assertIn("15.5", pasos["tempranos"].iloc[-1]["calculo"])


class TestPlotTikzPasos(unittest.TestCase):

    def setUp(self):
        self.pg = project()
        self.n = len(self.pg.nodes)
        self.m = len(self.pg.pert_graph.edges)

    def test_numero_de_fotogramas(self):
        frames = self.pg.plot_tikz_pasos(DUR)
        self.assertEqual(len(frames), 2 * self.n + self.m + 2)
        for c in frames:
            self.assertTrue(c.startswith(r"\begin{tikzpicture}"))
            self.assertTrue(c.rstrip().endswith(r"\end{tikzpicture}"))

    def test_nodos_definidos_antes_que_los_arcos(self):
        c = self.pg.plot_tikz_pasos(DUR)[0]
        self.assertLess(c.rfind(r"\node[circle"), c.find(r"\draw[->"))

    def test_relleno_progresivo(self):
        frames = self.pg.plot_tikz_pasos(DUR)
        # El fotograma vacío no tiene tiempos; el último paso hacia delante
        # muestra la duración del proyecto; el final marca el camino crítico.
        self.assertNotIn("(210:0.27)", frames[0])
        self.assertIn("= 31$", frames[self.n])
        self.assertIn("red, very thick", frames[-1])
        self.assertNotIn("red, very thick", frames[self.n])

    def test_beamer(self):
        c = self.pg.plot_tikz_pasos(DUR, beamer=True)
        self.assertIsInstance(c, str)
        self.assertEqual(c.count(r"\only<"), 2 * self.n + self.m + 2)
        self.assertIn(rf"\only<{2 * self.n + self.m + 2}>", c)

    def test_posiciones_y_textos(self):
        pos = {nodo: (i, 0) for i, nodo in enumerate(self.pg.nodes)}
        frames = self.pg.plot_tikz_pasos(
            DUR, posiciones=pos,
            textos={"inicio": "Project network", "final": "Critical path: {duracion}"})
        self.assertIn("Project network", frames[0])
        self.assertIn("Critical path: 31", frames[-1])
        self.assertIn("at (0.00cm, 0.00cm)", frames[0])

    def test_posiciones_incompletas(self):
        with self.assertRaises(ValueError):
            self.pg.plot_tikz_pasos(DUR, posiciones={self.pg.nodes[0]: (0, 0)})

    def test_grafo_construido_a_mano(self):
        import networkx as nx
        G = nx.DiGraph()
        for u, v, a in [("1", "2", "A"), ("1", "3", "B"), ("2", "3", "@2-3"), ("3", "4", "C")]:
            G.add_edge(u, v, activity=a)
        pg = ProjectGraph(G)
        pasos = pg.pert_pasos({"A": 2, "B": 1, "C": 4})
        self.assertEqual(pasos["tempranos"].iloc[-1]["valor"], 6)
        self.assertEqual(pasos["tempranos"].iloc[2]["calculo"], "t3 = máx{2 + 0, 0 + 1} = 2")


if __name__ == "__main__":
    unittest.main()


class TestHolgurasPasos(unittest.TestCase):
    def setUp(self):
        self.pg = project()
        self.pasos = self.pg.pert_pasos(DUR)
        self.res = self.pg.calculate_pert(DUR)

    def test_una_fila_por_actividad_con_la_holgura_total(self):
        h = self.pasos["holguras"]
        self.assertEqual(len(h), len(self.pg.pert_graph.edges))
        esperado = self.res["activities"]["H_total"]
        for _, fila in h.iterrows():
            self.assertEqual(fila["valor"], esperado[fila["actividad"]])

    def test_formula_en_el_calculo(self):
        fila = self.pg.pert_pasos(DUR, formato="latex")["holguras"].iloc[0]
        self.assertIn("t_{", fila["calculo"])
        self.assertIn(f"= {fila['valor']}", fila["calculo"])

    def test_fotogramas_de_holguras_y_rojo_progresivo(self):
        n_h = len(self.pasos["holguras"])
        n_sucesos = len(self.pasos["tempranos"])
        cods = self.pg.plot_tikz_pasos(DUR)
        self.assertEqual(len(cods), 1 + 2 * n_sucesos + n_h + 1)
        rojos = [c.count("fill=white!90!red") for c in cods[1 + 2 * n_sucesos: -1]]
        self.assertEqual(rojos, sorted(rojos))           # nunca se quita el rojo
        self.assertEqual(cods[1 + 2 * n_sucesos - 1].count("fill=white!90!red"), 0)
        self.assertGreater(rojos[-1], 0)
