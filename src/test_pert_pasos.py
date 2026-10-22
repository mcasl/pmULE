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
        self.assertIn(r"= \mathbf{31}$", frames[self.n])
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
        fotos_h = cods[1 + 2 * n_sucesos: -1]
        for c in fotos_h:                              # actividad y sus 2 sucesos en naranja
            self.assertEqual(c.count("fill=orange!20"), 2)
            self.assertIn("blue!80!black, ultra thick]", c)
        rojas = [c.count(", red, very thick") for c in fotos_h]
        self.assertEqual(rojas, sorted(rojas))          # las flechas rojas no se quitan
        self.assertEqual(cods[1 + 2 * n_sucesos - 1].count(", red, very thick"), 0)
        self.assertGreater(rojas[-1], 0)


class TestAckoffPasos(unittest.TestCase):
    def setUp(self):
        self.pg = project()
        self.d = pd.Series(DUR, dtype=float)
        self.mn = (self.d * 0.5).round()
        self.c = pd.Series({k: i + 1 for i, k in enumerate(DUR)}, dtype=float)

    def test_fotogramas_y_contenido(self):
        reduccion = 3
        html = self.pg.ackoff_pasos_html(self.d, self.mn, self.c, reduction=reduccion)
        n = html.count('class="pmule-paso"')
        self.assertEqual(n, 2 + reduccion)            # completa + datos + un paso por iteración
        self.assertIn("siguiente", html)
        self.assertIn("Iteración 0", html)
        self.assertIn("background:#ffff00", html)      # actividad acortada
        self.assertIn("background:#adff2f", html)      # ruta crítica
        self.assertIn("background:#fde3cc", html)      # celdas de cruce ruta crítica × actividad acortada

    def test_primer_fotograma_es_la_tabla_completa_y_el_segundo_solo_datos(self):
        html = self.pg.ackoff_pasos_html(self.d, self.mn, self.c, reduction=2)
        fotos = html.split('class="pmule-paso"')[1:]
        self.assertIn("background:#ffff00", fotos[0])
        self.assertNotIn("background:#ffff00", fotos[1])
        self.assertNotIn("background:#adff2f", fotos[1])
        self.assertGreater(fotos[0].count("<td>") + fotos[0].count("<td style"), 0)

    def test_contador_empieza_en_menos_uno(self):
        html = self.pg.ackoff_pasos_html(self.d, self.mn, self.c, reduction=3)
        self.assertIn("paso -1 de 3", html)
        self.assertIn('data-inicio="-1"', html)
        import pmule
        otro = pmule._html_pasos(["<p>a</p>", "<p>b</p>"])
        self.assertIn('data-inicio="1"', otro)
        self.assertIn("paso 1 de 2", otro)

    def test_la_columna_nueva_no_va_en_verde(self):
        html = self.pg.ackoff_pasos_html(self.d, self.mn, self.c, reduction=3)
        fotos = html.split('class="pmule-paso"')[1:]
        verdes = [f.count("background:#adff2f") + f.count("background:#e2f5b5") for f in fotos]
        self.assertEqual(verdes[1], 0)                       # solo datos
        # iteración r: se resaltan las críticas de las columnas 0..r, no las de la nueva
        self.assertEqual(verdes[2], 1)                       # columna 0: Route_5
        self.assertEqual(verdes[3], 2)                       # columnas 0 y 1
        self.assertEqual(verdes[4], 4)                       # 0, 1 y 2 (en la 2 empatan dos rutas)
        self.assertEqual(verdes[0], 6)                       # tabla completa: también la columna 3 (dos rutas)

    def test_solo_se_sombrean_las_celdas_de_cruce(self):
        html = self.pg.ackoff_pasos_html(self.d, self.mn, self.c, reduction=3)
        fotos = html.split('class="pmule-paso"')[1:]
        # iteración 0: crítica Route_5, se acorta A -> una sola celda
        self.assertEqual(fotos[2].count("background:#fde3cc"), 1)
        # iteración 2: críticas Route_3 y Route_5, se acorta F -> dos celdas
        self.assertEqual(fotos[4].count("background:#fde3cc"), 2)

    def test_costes_infinitos_no_rompen_el_visor(self):
        c = self.c.copy()
        c["A"] = float("inf")
        html = self.pg.ackoff_pasos_html(self.d, self.mn, c, reduction=2)
        self.assertIn("∞", html)


class TestColoresPorFlecha(unittest.TestCase):
    def setUp(self):
        import pmule
        self.pm = pmule
        self.pg = project()
        self.n = len(self.pg.nodes)

    def test_calculo_con_un_color_por_termino(self):
        pasos = self.pg.pert_pasos(DUR, formato="latex", colores=["red", "blue"])
        fila = pasos["tempranos"].iloc[2]                      # suceso con dos actividades que llegan
        self.assertEqual(len(fila["arcos"]), 2)
        self.assertIn(r"\textcolor{red}{", fila["calculo"])
        self.assertIn(r"\textcolor{blue}{", fila["calculo"])
        sin = self.pg.pert_pasos(DUR, formato="latex")["tempranos"].iloc[2]["calculo"]
        self.assertNotIn("textcolor", sin)
        self.assertEqual(self.pg.pert_pasos(DUR)["tempranos"].iloc[2]["calculo"].count("textcolor"), 0)

    def test_fotogramas_pert_cada_flecha_su_color(self):
        cods = self.pg.plot_tikz_pasos(DUR)
        paleta = self.pm.PALETA_FLECHAS
        # suceso 3 del ejemplo tiene dos actividades que llegan: dos colores distintos
        doble = [c for c in cods[1:1 + self.n + 1]
                 if paleta[0] + ", ultra thick" in c and paleta[1] + ", ultra thick" in c]
        self.assertTrue(doble)
        c = doble[0]
        self.assertIn(rf"\textcolor{{{paleta[0]}}}{{", c)          # valor del nodo y fórmula
        self.assertGreaterEqual(c.count(rf"\textcolor{{{paleta[1]}}}"), 2)

    def test_roy_colores_en_nodos_y_formula(self):
        paleta = self.pm.PALETA_FLECHAS
        fotos = self.pg.plot_tikz_roy_pasos(DUR)
        con_dos = [c for c in fotos if paleta[0] + ", ultra thick" in c and paleta[1] + ", ultra thick" in c]
        self.assertTrue(con_dos)
        self.assertIn(rf"\textcolor{{{paleta[1]}}}{{", con_dos[0])

    def test_holgura_en_un_color(self):
        paleta = self.pm.PALETA_FLECHAS
        cods = self.pg.plot_tikz_pasos(DUR)
        h = cods[1 + 2 * self.n]                               # primera holgura
        self.assertIn(rf"\textcolor{{{paleta[0]}}}{{\mathbf{{t_", h)


def test_construccion_pert_ejercicio4():
    import re
    import pandas as pd
    from pmule import PredecessorTable
    d = pd.DataFrame([('A', '---', 3), ('B', '---', 2), ('C', 'B', 4), ('D', 'A', 4), ('E', 'A', 2),
                      ('F', 'C,D', 5), ('G', 'C,D', 1), ('H', 'E', 3), ('I', 'F,G', 1),
                      ('J', 'F,G', 2), ('K', 'F,G,H', 1)],
                     columns=['activity', 'predecessor', 'duration']).set_index('activity')
    t = PredecessorTable.from_dataframe_of_strings(d, activity='activity', predecessor='predecessor')
    cods = t.create_project().plot_tikz_construccion(tabla=t)
    assert len(cods) == 12                       # iniciales + 9 actividades + 1 fusión + final
    assert "iniciales" in cods[0] and "A y B" in cods[0]
    assert any("son iguales" in c for c in cods)  # fusión de C y D
    assert "draw=white, thick" in cods[0]            # lo aún no calculado, en blanco
    assert "blue!70!black" in cods[2]                # previas de la nueva actividad
    assert "ultra thick" not in cods[-1]          # fotograma final sin resaltes
    assert "§" not in "".join(cods)
    assert "{10}" in cods[-1] and "{10}" not in cods[3]   # números de nodo solo al final


def test_zaderenko_tikz_ejercicio4():
    import pandas as pd
    from pmule import PredecessorTable
    d = pd.DataFrame([('A', '---', 3), ('B', '---', 2), ('C', 'B', 4), ('D', 'A', 4), ('E', 'A', 2),
                      ('F', 'C,D', 5), ('G', 'C,D', 1), ('H', 'E', 3), ('I', 'F,G', 1),
                      ('J', 'F,G', 2), ('K', 'F,G,H', 1)],
                     columns=['activity', 'predecessor', 'duration']).set_index('activity')
    t = PredecessorTable.from_dataframe_of_strings(d, activity='activity', predecessor='predecessor')
    p = t.create_project()
    cods = p.plot_tikz_zaderenko(d['duration'])
    assert len(cods) == 2 * 10 + 2                # inicio + 10 sucesos hacia delante y atrás + final
    assert all(r"\begin{scope}[xshift=" in c for c in cods)
    assert r"\max" in cods[5]                      # cálculo del paso
    assert "red" not in cods[-1]                   # sin rojo de actividades críticas
    assert "orange!18" in cods[3] and "orange!18" not in cods[0]


def test_calendario_interactivo_ejercicio4():
    import pandas as pd
    from pmule import PredecessorTable
    d = pd.DataFrame([('A', '---', 3), ('B', '---', 2), ('C', 'B', 4), ('D', 'A', 4), ('E', 'A', 2),
                      ('F', 'C,D', 5), ('G', 'C,D', 1), ('H', 'E', 3), ('I', 'F,G', 1),
                      ('J', 'F,G', 2), ('K', 'F,G,H', 1)],
                     columns=['activity', 'predecessor', 'duration']).set_index('activity')
    t = PredecessorTable.from_dataframe_of_strings(d, activity='activity', predecessor='predecessor')
    p = t.create_project()
    dur, te, tl, filas = p._calendario_datos(d['duration'])
    cal = p.calendar(d['duration'])
    columnas = {'ES': 'inicio_mas_temprano', 'EF': 'fin_mas_temprano', 'LS': 'inicio_mas_tardio',
                'LF': 'fin_mas_tardio', 'HT': 'H_total', 'HL': 'H_libre', 'HI': 'H_independiente',
                'dur': 'duracion'}
    for a, fila in filas.items():                 # las fórmulas dan lo mismo que calendar()
        for c, k in columnas.items():
            assert float(fila[c]) == float(cal.loc[a, k]), (a, c)
    codigos, indice, _ = p.plot_tikz_calendario(d['duration'])
    assert len(indice) == 11 * 8 and 1 < len(codigos) <= 1 + 11 * 8
    assert codigos[indice[('F', 'HT')]] != codigos[0]
    # variante Roy: mismas celdas y mismos valores que calendar() (holguras por actividades)
    codigos_r, indice_r, (filas_r, extra_r, celdas_r) = p.plot_tikz_calendario(d['duration'], grafo='roy')
    assert len(indice_r) == 11 * 8
    for (a, c), info in celdas_r.items():
        assert float(info['res']) == float(filas_r[a][c]) == float(cal.loc[a, columnas[c]]), (a, c)
    assert cal.loc['G', 'H_libre'] == 4 and cal.loc['I', 'H_libre'] == 1
    import pytest
    with pytest.raises(ValueError):
        p.plot_tikz_calendario(d['duration'], grafo='otro')


def test_plot_tikz_normal():
    import pytest
    from pmule import plot_tikz_normal
    from scipy.stats import norm
    c = plot_tikz_normal(14, 2.8, hasta=9)
    assert c.startswith(r"\begin{tikzpicture}") and c.rstrip().endswith(r"\end{tikzpicture}")
    assert r"P(T\le 9)" in c and f"{norm.cdf(9, 14, 2.8):.4f}" in c and "orange!50" in c
    c = plot_tikz_normal(14, 2.8, entre=(10, 12))
    assert f"{norm.cdf(12, 14, 2.8) - norm.cdf(10, 14, 2.8):.4f}" in c
    c = plot_tikz_normal(14, 2.8, hasta=[9, 12])         # áreas anidadas, una leyenda por cada una
    assert c.count(r"P(T\le") == 2
    c = plot_tikz_normal(14, 2.8, percentil=0.9)
    assert f"{norm.ppf(0.9, 14, 2.8):.2f}" in c and "0.9000" in c
    assert r"P(T\ge 15)" in plot_tikz_normal(14, 2.8, desde=15)
    with pytest.raises(ValueError):
        plot_tikz_normal(14, 2.8)
