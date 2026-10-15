"""test_crashing_recursos.py — crashing con recursos limitados y Monte Carlo con recursos."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from pmule import ProjectGraph, CrashingRecursos, MonteCarloRecursosResult

# X→Y es el camino crítico sin recursos (10); Z y W compiten por R (límite 1): con recursos T = 12
PREDS = {"X": set(), "Y": {"X"}, "Z": set(), "W": set()}
DATA = pd.DataFrame({"duration": {"X": 5, "Y": 5, "Z": 6, "W": 6},
                     "R": {"X": 0, "Y": 0, "Z": 1, "W": 1}})
MIND = {"X": 4, "Y": 4, "Z": 5, "W": 5}
UCR = {"X": 1, "Y": 2, "Z": 4, "W": 3}


def pg_ejemplo():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


class TestEjemploReferencia(unittest.TestCase):

    def test_sin_recursos_manda_otro_camino(self):
        pg = pg_ejemplo()
        self.assertEqual(pg.duration(DATA["duration"]), 10)
        out = pg.asignar_multi(DATA, "duration", ["R"], [1], dibujar=False)
        self.assertEqual(out.duracion, 12)

    def test_ackoff_sin_recursos_no_ve_el_cuello_de_botella(self):
        r = pg_ejemplo().crashing_recursos(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=11)
        self.assertEqual(r.duracion_inicial, 12)
        self.assertEqual(r.duracion_final, 11)
        self.assertEqual(r.coste_total, 3)
        p = r.pasos.loc[1]
        self.assertEqual(p["actividades"], "W")                 # la reducción barata que de verdad sirve
        self.assertEqual(p["ackoff_sin_recursos"], "X")         # lo que elegiría Ackoff sin recursos (cuesta 1, no sirve)
        self.assertEqual(p["cadena_critica"], "W, Z")
        self.assertIsInstance(r, CrashingRecursos)

    def test_dos_pasos_y_parada(self):
        r = pg_ejemplo().crashing_recursos(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=10)
        self.assertEqual(r.duracion_final, 10)
        self.assertEqual(r.coste_total, 7)                       # W (3) + Z (4)
        sin_limite = pg_ejemplo().crashing_recursos(DATA, "duration", ["R"], [1], MIND, UCR)
        self.assertEqual(sin_limite.duracion_final, 10)
        # a los 10 hay dos cadenas críticas (X→Y y Z→W) y Z, W ya están al mínimo: nada de una unidad sirve
        self.assertIn("ninguna reducción de una unidad acorta el plazo", sin_limite.motivo_parada)

    def test_reduccion_y_objetivo_excluyentes(self):
        with self.assertRaises(ValueError):
            pg_ejemplo().crashing_recursos(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=10, reduccion=1)
        r = pg_ejemplo().crashing_recursos(DATA, "duration", ["R"], [1], MIND, UCR, reduccion=1)
        self.assertEqual(r.duracion_final, 11)

    def test_modelo_con_recurso_acortar_consume_el_recurso_escaso(self):
        # acortar W exigiría 2 unidades de R (límite 1): ya no es posible; se acorta Z
        extra = {"W": {"R": 1}}
        r = pg_ejemplo().crashing_recursos(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=11,
                                           demanda_extra=extra)
        self.assertEqual(r.pasos.loc[1, "actividades"], "Z")
        self.assertEqual(r.coste_total, 4)
        # con un límite 2 sí cabe, y entonces W es de nuevo la mejor opción
        r2 = pg_ejemplo().crashing_recursos(DATA, "duration", ["R"], [2], MIND, UCR, objetivo=6,
                                            demanda_extra=extra)
        self.assertEqual(r2.duracion_inicial, 10)

    def test_plan_final_coincide_con_asignar_multi(self):
        r = pg_ejemplo().crashing_recursos(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=10, dibujar=True)
        self.assertEqual(r.asignacion.duracion, 10)
        self.assertTrue((r.asignacion.cargas.values <= r.asignacion.limites.values).all())

    def test_validacion(self):
        pg = pg_ejemplo()
        with self.assertRaises(ValueError):
            pg.crashing_recursos(DATA, "duration", ["R"], None, MIND, UCR, objetivo=10)
        with self.assertRaises(ValueError):
            pg.crashing_recursos(DATA, "duration", ["Q"], [1], MIND, UCR, objetivo=10)
        with self.assertRaises(ValueError):
            pg.crashing_recursos(DATA, "duration", ["R"], [1], {**MIND, "X": 9}, UCR, objetivo=10)


class TestCadenaCritica(unittest.TestCase):

    def test_ejemplo(self):
        c = pg_ejemplo().cadena_critica_recursos(DATA, "duration", ["R"], [1])
        self.assertEqual((c["duracion"], c["duracion_sin_recursos"]), (12, 10))
        self.assertEqual(c["actividades"], ["W", "Z"])
        self.assertEqual(c["arcos"].to_dict("records"), [{"de": "W", "a": "Z", "motivo": "recurso"}])   # W y Z empatan en LS: gana el nombre

    def test_sin_limite_efectivo_es_el_camino_critico(self):
        c = pg_ejemplo().cadena_critica_recursos(DATA, "duration", ["R"], [100])
        self.assertEqual(c["actividades"], ["X", "Y"])
        self.assertEqual(c["duracion"], c["duracion_sin_recursos"])
        self.assertEqual(c["arcos"].to_dict("records"), [{"de": "X", "a": "Y", "motivo": "precedencia"}])


class TestExacto(unittest.TestCase):

    def test_optimo_del_ejemplo(self):
        pg = pg_ejemplo()
        e11 = pg.crashing_recursos_exacto(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=11)
        e10 = pg.crashing_recursos_exacto(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=10)
        self.assertEqual((e11["estado"], round(e11["coste"], 6)), ("óptimo", 3.0))
        self.assertEqual(round(e10["coste"], 6), 7.0)
        self.assertEqual(pg.crashing_recursos_exacto(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=9)["estado"],
                         "infactible")

    def test_el_exacto_respeta_restricciones(self):
        pg = pg_ejemplo()
        e = pg.crashing_recursos_exacto(DATA, "duration", ["R"], [1], MIND, UCR, objetivo=11)
        ini, dur = e["inicios"], e["duraciones"]
        self.assertTrue(((ini + dur) <= 11).all())
        self.assertGreaterEqual(ini["Y"], ini["X"] + dur["X"])
        z, w = ini["Z"], ini["W"]
        self.assertTrue(z + dur["Z"] <= w or w + dur["W"] <= z)      # no se solapan (límite 1)

    def test_heuristica_nunca_mejora_al_exacto(self):
        rng = np.random.default_rng(3)
        comparadas = 0
        for _ in range(12):
            n = 6
            names = [chr(65 + i) for i in range(n)]
            preds = {a: (set(rng.choice(names[:i], size=int(rng.integers(0, min(i, 2) + 1)), replace=False))
                         if i else set()) for i, a in enumerate(names)}
            pg = ProjectGraph.from_dict_of_sets(preds, simplify=True)
            dur = {a: int(rng.integers(2, 5)) for a in names}
            data = pd.DataFrame({"duration": dur,
                                 "R1": {a: int(rng.integers(0, 3)) for a in names},
                                 "R2": {a: int(rng.integers(0, 3)) for a in names}})
            lim = [max(2, int(data["R1"].max())), max(2, int(data["R2"].max()))]
            mind = {a: max(1, d - int(rng.integers(0, 3))) for a, d in dur.items()}
            ucr = {a: float(rng.integers(1, 6)) for a in names}
            r = pg.crashing_recursos(data, "duration", ["R1", "R2"], lim, mind, ucr)
            if r.duracion_final >= r.duracion_inicial:
                continue
            e = pg.crashing_recursos_exacto(data, "duration", ["R1", "R2"], lim, mind, ucr,
                                            objetivo=r.duracion_final)
            self.assertEqual(e["estado"], "óptimo")
            self.assertLessEqual(e["coste"], r.coste_total + 1e-6)
            comparadas += 1
        self.assertGreaterEqual(comparadas, 4)


class TestMonteCarloRecursos(unittest.TestCase):

    def setUp(self):
        self.pg = pg_ejemplo()

    def test_duraciones_constantes(self):
        dist = {a: ("uniform", d, d) for a, d in DATA["duration"].items()}
        r = self.pg.monte_carlo_recursos(dist, DATA, ["R"], [1], n_samples=20, seed=1)
        self.assertTrue((r.tau_libre_samples == 10).all())
        self.assertTrue((r.tau_recursos_samples == 12).all())
        self.assertEqual(r.criticidad.loc["Z", "con recursos"], 1.0)
        self.assertEqual(r.criticidad.loc["X", "con recursos"], 0.0)
        self.assertEqual(r.criticidad.loc["X", "sin recursos"], 1.0)
        self.assertEqual(r.criticidad.loc["Z", "sin recursos"], 0.0)
        self.assertEqual(r.bloqueos["R"], 1.0)
        self.assertIsInstance(r, MonteCarloRecursosResult)

    def test_limites_holgados_igualan_al_cpm(self):
        dist = {a: ("triangular", max(d - 1, 1), d, d + 2) for a, d in DATA["duration"].items()}
        r = self.pg.monte_carlo_recursos(dist, DATA, ["R"], [100], n_samples=200, seed=2)
        self.assertTrue(np.array_equal(r.tau_libre_samples, r.tau_recursos_samples))
        self.assertEqual(r.bloqueos["R"], 0.0)

    def test_recursos_nunca_acortan_y_la_normal_subestima(self):
        dist = {a: ("triangular", max(d - 1, 1), d, d + 3) for a, d in DATA["duration"].items()}
        r = self.pg.monte_carlo_recursos(dist, DATA, ["R"], [1], n_samples=300, seed=3)
        self.assertTrue((r.tau_recursos_samples >= r.tau_libre_samples).all())
        self.assertGreater(r.summary().loc["con recursos", "media"], r.summary().loc["sin recursos", "media"])
        self.assertGreater(r.prob_superar(12), r.prob_superar(12, con_recursos=False))

    def test_tau_libre_es_el_cpm_de_las_duraciones_redondeadas(self):
        dist = {a: ("triangular", max(d - 1, 1), d, d + 3) for a, d in DATA["duration"].items()}
        r = self.pg.monte_carlo_recursos(dist, DATA, ["R"], [1], n_samples=100, seed=4)
        rng = np.random.default_rng(4)
        D = np.column_stack([np.maximum(np.rint(rng.triangular(max(d - 1, 1), d, d + 3, 100)), 1.0)
                             for d in DATA.loc[["W", "X", "Z", "Y"], "duration"]])   # orden en que se muestrea
        esperado = np.maximum(D[:, 1] + D[:, 3], np.maximum(D[:, 0], D[:, 2]))       # X→Y frente a W y Z en paralelo
        self.assertTrue(np.array_equal(r.tau_libre_samples, esperado))

    def test_errores(self):
        with self.assertRaises(ValueError):
            self.pg.monte_carlo_recursos({"X": ("uniform", 1, 2)}, DATA, ["R"], [1], n_samples=5)
        with self.assertRaises(ValueError):
            self.pg.monte_carlo_recursos({}, DATA, ["R"], None, n_samples=5)


class TestAnomalias(unittest.TestCase):

    def test_acortar_una_actividad_puede_alargar_el_proyecto(self):
        """Anomalía de Graham: con reglas de prioridad, reducir una duración puede aumentar T."""
        rng = np.random.default_rng(11)
        halladas = 0
        for _ in range(250):
            n = 7
            names = [chr(65 + i) for i in range(n)]
            preds = {a: (set(rng.choice(names[:i], size=int(rng.integers(0, min(i, 2) + 1)), replace=False))
                         if i else set()) for i, a in enumerate(names)}
            pg = ProjectGraph.from_dict_of_sets(preds, simplify=True)
            dur = {a: int(rng.integers(2, 7)) for a in names}
            data = pd.DataFrame({"duration": dur, "R": {a: int(rng.integers(1, 4)) for a in names}})
            lim = [max(3, int(data["R"].max()))]
            base = pg.asignar_multi(data, "duration", ["R"], lim, dibujar=False).duracion
            for a in names:
                if dur[a] <= 1:
                    continue
                d2 = data.copy()
                d2.loc[a, "duration"] -= 1
                if pg.asignar_multi(d2, "duration", ["R"], lim, dibujar=False).duracion > base:
                    halladas += 1
                    break
            if halladas:
                break
        self.assertGreaterEqual(halladas, 1)


if __name__ == "__main__":
    unittest.main()
