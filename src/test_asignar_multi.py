"""test_asignar_multi.py — asignación con varios recursos limitados."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from pmule import ProjectGraph

PREDS = {"A": set(), "B": set(), "C": set(), "D": {"A"}, "E": {"B", "C"}, "F": {"D", "E"}}
DUR = {"A": 2, "B": 3, "C": 2, "D": 2, "E": 3, "F": 2}
R1 = {"A": 2, "B": 2, "C": 1, "D": 1, "E": 1, "F": 2}
R2 = {"A": 1, "B": 0, "C": 2, "D": 1, "E": 1, "F": 1}
R3 = {"A": 0, "B": 1, "C": 1, "D": 2, "E": 1, "F": 1}
LABELS = ["R1", "R2", "R3"]


def proyecto():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


def datos():
    return pd.DataFrame({"duration": DUR, "R1": R1, "R2": R2, "R3": R3})


class TestEjemploTresRecursos(unittest.TestCase):

    def setUp(self):
        self.pg = proyecto()
        self.n_edges = self.pg.pert_graph.number_of_edges()
        self.out = self.pg.asignar_multi(datos(), "duration", LABELS, [4, 3, 2])

    def test_inicios_y_duracion(self):
        self.assertEqual(self.out.inicios.to_dict(),
                         {"A": 2, "B": 0, "C": 0, "D": 6, "E": 3, "F": 8})
        self.assertEqual(self.out.duracion, 10)
        self.assertEqual(self.out.esperas.to_dict(), {"A": 2, "D": 2})

    def test_recurso_que_bloquea(self):
        t = self.out.traza
        a0 = t[(t.periodo == 1) & (t.actividad == "A")].iloc[0]
        self.assertEqual((a0.decision, a0.bloqueada_por), ("espera", "R1"))
        d5 = t[(t.periodo == 5) & (t.actividad == "D")].iloc[0]
        self.assertEqual((d5.decision, d5.bloqueada_por), ("espera", "R3"))

    def test_cargas_respetan_limites(self):
        self.assertTrue((self.out.cargas.values <= self.out.limites.values).all())

    def test_no_muta_el_proyecto(self):
        self.assertEqual(self.pg.pert_graph.number_of_edges(), self.n_edges)

    def test_gantt_coincide_con_inicios(self):
        for rl in LABELS:
            g = self.out.gantts[rl].data
            tot = pd.to_numeric(g.loc["Total"].drop("H_total", errors="ignore"),
                                errors="coerce").dropna()
            self.assertTrue((tot.values <= self.out.limites[rl].values[:len(tot)]).all())
        cal = self.out.proyecto.calendar(self.out.data["duration"])
        for a, s in self.out.inicios.items():
            self.assertEqual(cal.loc[a, "inicio_mas_temprano"], s)


class TestValidacion(unittest.TestCase):

    def test_errores(self):
        pg = proyecto()
        with self.assertRaises(ValueError):
            pg.asignar_multi(datos(), "duration", LABELS)
        with self.assertRaises(ValueError):
            pg.asignar_multi(datos(), "duration", None, 3)
        with self.assertRaises(ValueError):
            pg.asignar_multi(datos(), "dur", LABELS, 3)
        with self.assertRaises(ValueError):
            pg.asignar_multi(datos(), "duration", ["R1", "RX"], 3)
        with self.assertRaises(ValueError):
            pg.asignar_multi(datos(), "duration", LABELS, [4, 3])
        with self.assertRaises(ValueError):      # A pide 2 de R1 y el límite es 1
            pg.asignar_multi(datos(), "duration", LABELS, [1, 3, 2])

    def test_formas_de_limite(self):
        pg = proyecto()
        a = pg.asignar_multi(datos(), "duration", LABELS, {"R1": 4, "R2": 3, "R3": 2})
        b = pg.asignar_multi(datos(), "duration", LABELS, [4, 3, 2])
        c = pg.asignar_multi(datos(), "duration", LABELS,
                             pd.DataFrame({"R1": [4] * 12, "R2": [3] * 12, "R3": [2] * 12},
                                          index=range(1, 13)))
        self.assertEqual(a.inicios.to_dict(), b.inicios.to_dict())
        self.assertEqual(a.inicios.to_dict(), c.inicios.to_dict())

    def test_un_solo_recurso_y_escalar(self):
        out = proyecto().asignar_multi(datos(), "duration", "R1", 3)
        self.assertTrue((out.cargas["R1"] <= 3).all())

    def test_limite_variable(self):
        out = proyecto().asignar_multi(datos(), "duration", ["R1"],
                                       {"R1": pd.Series([2, 2, 2, 2, 4], index=range(1, 6))})
        self.assertTrue((out.cargas.values <= out.limites.values).all())


class TestInvariantesAleatorios(unittest.TestCase):

    def _random_project(self, rng, n=9):
        names = [chr(65 + i) for i in range(n)]
        preds = {}
        for i, a in enumerate(names):
            k = rng.integers(0, min(i, 3) + 1)
            preds[a] = set(rng.choice(names[:i], size=k, replace=False)) if i and k else set()
        dur = {a: int(rng.integers(1, 5)) for a in names}
        return preds, dur

    def test_factible_y_respeta_precedencias(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            preds, dur = self._random_project(rng)
            pg = ProjectGraph.from_dict_of_sets(preds, simplify=True)
            data = pd.DataFrame({"duration": dur})
            for rl in LABELS:
                data[rl] = rng.integers(0, 4, size=len(dur))
            lim = [int(data[rl].max()) + int(rng.integers(0, 3)) for rl in LABELS]
            out = pg.asignar_multi(data, "duration", LABELS, lim)
            self.assertTrue((out.cargas.values <= out.limites.values + 1e-9).all())
            for a, ps in preds.items():
                for p in ps:
                    self.assertLessEqual(out.fines[p], out.inicios[a])
            self.assertGreaterEqual(out.duracion, max(pg.calendar(dur)["fin_mas_temprano"]))
            # la duración del Gantt dibujado coincide
            g = out.gantts["R1"].data
            periodos = [c for c in g.columns if isinstance(c, (int, np.integer))]
            self.assertEqual(len(periodos), out.duracion)

    def test_limites_holgados_dan_el_cpm(self):
        rng = np.random.default_rng(11)
        for _ in range(15):
            preds, dur = self._random_project(rng)
            pg = ProjectGraph.from_dict_of_sets(preds, simplify=True)
            data = pd.DataFrame({"duration": dur})
            for rl in LABELS:
                data[rl] = rng.integers(0, 4, size=len(dur))
            out = pg.asignar_multi(data, "duration", LABELS, 10_000)
            self.assertEqual(out.duracion, int(pg.calendar(dur)["fin_mas_temprano"].max()))
            self.assertEqual(len(out.esperas), 0)


if __name__ == "__main__":
    unittest.main()
