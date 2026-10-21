"""Figura de las matrices de enlace (cuadro de prelaciones): modo letra / modo marca
y reordenación opcional por precedencia."""
import unittest
import pandas as pd
from pmule import PredecessorTable, MatrizEnlace, _codigo_matriz_tikz, _geometria_matriz


def tabla(filas):
    d = pd.DataFrame(filas, columns=["activity", "predecessor", "duration"]).set_index("activity")
    return PredecessorTable.from_dataframe_of_strings(d, activity="activity", predecessor="predecessor")


CORTOS = [("C", "A,B", 1), ("A", "---", 1), ("B", "A", 1), ("D", "C", 1)]
LARGOS = [("Cimentación", "Excavación", 4), ("Excavación", "---", 3), ("Muros", "Cimentación", 2)]


class TestMatrizEnlace(unittest.TestCase):
    def test_modo_segun_longitud(self):
        self.assertEqual(_geometria_matriz(tabla(CORTOS).immediate_linkage_matrix)[0], "letra")
        self.assertEqual(_geometria_matriz(tabla(LARGOS).immediate_linkage_matrix)[0], "marca")

    def test_codigo_letra_y_marca(self):
        c = _codigo_matriz_tikz(tabla(CORTOS).immediate_linkage_matrix)
        self.assertNotIn("rotate=90", c)
        self.assertIn(r"\node[narosc,font=\bfseries]", c)
        l = _codigo_matriz_tikz(tabla(LARGOS).immediate_linkage_matrix)
        self.assertIn("rotate=90", l)
        self.assertIn("line cap=round", l)

    def test_por_defecto_no_reordena(self):
        t = tabla(LARGOS)
        m = MatrizEnlace(t.immediate_linkage_matrix)
        self.assertEqual(list(m.data.index), list(t.immediate_linkage_matrix.index))

    def test_ordenar_deja_triangular_inferior(self):
        for filas in (CORTOS, LARGOS):
            t = tabla(filas)
            m = MatrizEnlace(t.immediate_linkage_matrix, ordenar=True)
            M = m.data.to_numpy()
            for i in range(len(M)):
                for j in range(i, len(M)):
                    self.assertFalse(M[i, j], (filas[0], i, j))
        self.assertEqual(list(MatrizEnlace(tabla(LARGOS).immediate_linkage_matrix, ordenar=True).data.index),
                         ["Excavación", "Cimentación", "Muros"])

    def test_display_con_parametro(self):
        t = tabla(CORTOS)
        self.assertEqual(list(t.display_immediate_linkage_matrix(ordenar=True).data.index), list("ABCD"))
        self.assertTrue(hasattr(t.display_distant_linkage_matrix(ordenar=True), "data"))

    def test_html_incrusta_svg(self):
        h = MatrizEnlace(tabla(CORTOS).immediate_linkage_matrix)._repr_html_()
        self.assertIn("data:image/svg+xml;base64,", h)
        self.assertIn("overflow-x:auto", h)


if __name__ == "__main__":
    unittest.main()
