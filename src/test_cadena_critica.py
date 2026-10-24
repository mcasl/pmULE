"""test_cadena_critica.py — plan de cadena crítica (CCPM) con el ejemplo de la nave de la lección 12."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from pmule import ProjectGraph, grafico_fiebre_tikz

PREDS = {"A": "", "B": "A", "C": "", "D": "B, C", "E": "D", "F": "D", "G": "E, F"}
AGRESIVA = {"A": 5, "B": 10, "C": 17, "D": 10, "E": 13, "F": 8, "G": 3}
SEGURA = {"A": 6, "B": 12, "C": 19, "D": 12, "E": 15, "F": 10, "G": 4}
RECURSO = {"A": "", "B": "", "C": "", "D": "", "E": "X", "F": "X", "G": ""}


def nave():
    proyecto = ProjectGraph.from_dict_of_strings(PREDS)
    datos = pd.DataFrame({"agresiva": AGRESIVA, "segura": SEGURA, "recurso": RECURSO})
    return proyecto, datos


class TestNaveLeccion12(unittest.TestCase):
    """Cifras del ejemplo de la lección 12 (cuadrilla X: primero F y después E)."""

    def setUp(self):
        proyecto, datos = nave()
        self.plan = proyecto.plan_cadena_critica(datos, recurso="recurso", secuencia_recursos={"X": ["F", "E"]})

    def test_cadena_y_colchon_del_proyecto(self):
        p = self.plan
        self.assertEqual(p.cadena, ["C", "D", "F", "E", "G"])
        self.assertEqual(p.duracion_cadena, 51)
        self.assertEqual(int(p.margen[p.cadena].sum()), 9)
        self.assertAlmostEqual(p.calculo_colchon_proyecto["cortar_pegar"], 4.5)
        self.assertAlmostEqual(p.calculo_colchon_proyecto["raiz"], np.sqrt(17))
        self.assertEqual(p.colchon_proyecto, 5)
        self.assertEqual(p.compromiso, 56)
        self.assertEqual(p.duracion_tradicional, 60)
        self.assertEqual(p.cadenas_empatadas, 1)
        self.assertEqual(list(p.enlaces_recurso.itertuples(index=False, name=None)), [("F", "E", "X")])
        self.assertEqual(list(p.enlaces_cadena["motivo"]), ["precedencia", "precedencia", "recurso", "precedencia"])

    def test_colchon_de_alimentacion_y_alap(self):
        p = self.plan
        self.assertEqual(len(p.alimentacion), 1)
        ca = p.alimentacion.iloc[0]
        self.assertEqual((ca["desde"], ca["antes_de"], ca["cadena_alimentacion"]), ("B", "D", "A - B"))
        self.assertAlmostEqual(ca["bruto"], 1.5)
        self.assertEqual((ca["requerido"], ca["disponible"], ca["adoptado"]), (2, 2, 2))
        self.assertEqual(int(p.inicios["A"]), 0)
        self.assertEqual(int(p.inicios["B"]), 5)
        self.assertEqual(int(p.inicios["D"]), 17)
        self.assertEqual(int(p.inicios["E"]), 35)

    def test_colchon_de_recursos(self):
        self.assertEqual(list(self.plan.colchones_recursos["antes_de"]), ["F"])

    def test_seguimiento_tres_informes(self):
        p = self.plan
        i20 = p.seguimiento(20, terminadas={"A": 5, "B": 15, "C": 19}, en_curso={"D": 9})
        self.assertEqual(i20["fin_previsto"], 53)
        self.assertAlmostEqual(i20["pct_consumo"], 40)
        self.assertAlmostEqual(i20["pct_avance"], 100 * 18 / 51)
        self.assertEqual(i20["zona"], "amarilla")
        i30 = p.seguimiento(30, terminadas={"A": 5, "B": 15, "C": 19, "D": 29}, en_curso={"F": 7})
        self.assertEqual(i30["fin_previsto"], 53)
        self.assertEqual(i30["zona"], "verde")
        i45 = p.seguimiento(45, terminadas={"A": 5, "B": 15, "C": 19, "D": 29, "F": 37}, en_curso={"E": 8})
        self.assertEqual(i45["fin_previsto"], 56)
        self.assertAlmostEqual(i45["pct_consumo"], 100)
        self.assertAlmostEqual(i45["pct_avance"], 100 * 40 / 51)
        self.assertEqual(i45["zona"], "roja")
        self.assertAlmostEqual(i45["bpi"], 0)
        tikz = grafico_fiebre_tikz([i20, i30, i45])
        self.assertIn("día 45", tikz)
        self.assertTrue(tikz.strip().endswith(r"\end{tikzpicture}"))

    def test_alimentacion_retrasada_consume_colchon(self):
        p = self.plan
        # B termina el día 18 en lugar del 15: consume su CA (2 días) y 1 día del colchón del proyecto
        inf = p.seguimiento(18, terminadas={"A": 7, "B": 18, "C": 17})
        self.assertEqual(inf["fin_previsto"], 52)
        fila = inf["alimentacion"].iloc[0]
        self.assertAlmostEqual(fila["consumido"], 3)

    def test_gantt_tikz(self):
        tikz = self.plan.gantt_tikz()
        self.assertIn("orange!45", tikz)
        self.assertIn("dashed", tikz)
        self.assertIn("{CP}", tikz)
        self.assertIn("{CA1}", tikz)

    def test_red_nivelada(self):
        red = self.plan.red_nivelada()
        self.assertEqual(red.loc["E", "predecessor"], "D,F")
        self.assertEqual(red.loc["A", "predecessor"], "---")


class TestSinSecuenciaImpuesta(unittest.TestCase):
    def test_planificador_elige_el_orden(self):
        proyecto, datos = nave()
        plan = proyecto.plan_cadena_critica(datos, recurso="recurso")
        # prioridad: menor LS sin recursos -> E antes que F; la cadena mide lo mismo
        self.assertEqual(plan.cadena, ["C", "D", "E", "F", "G"])
        self.assertEqual(plan.duracion_cadena, 51)
        self.assertEqual(plan.compromiso, 56)

    def test_sin_recursos_es_la_ruta_critica(self):
        proyecto, datos = nave()
        plan = proyecto.plan_cadena_critica(datos)
        self.assertEqual(plan.cadena, ["C", "D", "E", "G"])
        self.assertEqual(plan.duracion_cadena, 43)
        # dos colchones de alimentación: A-B antes de D y F antes de G
        self.assertEqual(list(plan.alimentacion["antes_de"]), ["D", "G"])
        self.assertEqual(int(plan.alimentacion.loc["CA2", "requerido"]), 1)
        self.assertEqual(int(plan.inicios["F"]), 40 - 1 - 8)

    def test_raiz_y_colchon_que_no_cabe(self):
        proyecto = ProjectGraph.from_dict_of_strings({"A": "", "B": "", "C": "A, B"})
        datos = pd.DataFrame({"agresiva": {"A": 6, "B": 5, "C": 4}, "segura": {"A": 9, "B": 9, "C": 6}})
        plan = proyecto.plan_cadena_critica(datos, metodo="raiz")
        self.assertEqual(plan.cadena, ["A", "C"])
        self.assertEqual(plan.colchon_proyecto, 4)          # sqrt(9 + 4) = 3,61
        ca = plan.alimentacion.iloc[0]
        self.assertEqual((ca["requerido"], ca["disponible"], ca["adoptado"]), (4, 1, 1))
        self.assertEqual(int(plan.inicios["B"]), 0)

    def test_simular(self):
        proyecto, datos = nave()
        plan = proyecto.plan_cadena_critica(datos, recurso="recurso", secuencia_recursos={"X": ["F", "E"]})
        relevos = plan.simular(20000, semilla=1)
        trad = plan.simular(20000, semilla=1, regimen="tradicional")
        self.assertEqual(relevos.shape, (20000,))
        self.assertTrue(np.all(trad >= plan.duracion_tradicional - 1e-9))
        self.assertGreater(np.mean(relevos <= plan.compromiso), 0.85)
        self.assertLess(np.mean(relevos <= plan.duracion_cadena), 0.6)
        self.assertLess(np.mean(trad <= plan.duracion_tradicional), np.mean(relevos <= plan.compromiso))


if __name__ == "__main__":
    unittest.main()
