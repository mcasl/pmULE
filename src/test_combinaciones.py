"""test_combinaciones.py — VAN, crashing, valor ganado y recursos combinados."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from pmule import ProjectGraph, MonteCarloVANRecursosResult, MonteCarloValorGanadoResult

PREDS = {"X": set(), "Y": {"X"}, "Z": set(), "W": set()}
DATA = pd.DataFrame({"duration": {"X": 5, "Y": 5, "Z": 6, "W": 6},
                     "R": {"X": 0, "Y": 0, "Z": 1, "W": 1}})
MIND = {"X": 4, "Y": 4, "Z": 5, "W": 5}
UCR = {"X": 1, "Y": 2, "Z": 4, "W": 3}
BAC = {"X": 10.0, "Y": 10.0, "Z": 12.0, "W": 12.0}
CTE = {a: ("triangular", d, d, d) for a, d in DATA["duration"].items()}


def pg():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


def van_mensual(inicios, durs, bac, tau, F, n, tasa):
    """VAN por flujo mensual explícito (referencia independiente de la forma cerrada)."""
    km = (1 + tasa) ** (1 / 12) - 1
    total = 0.0
    for a in bac:
        for m in range(int(inicios[a]) + 1, int(inicios[a]) + int(durs[a]) + 1):
            total -= bac[a] / durs[a] / (1 + km) ** m
    for m in range(int(tau) + 1, int(tau) + n + 1):
        total += F / (1 + km) ** m
    return total


class TestVANRecursos(unittest.TestCase):

    def test_coincide_con_flujo_mensual(self):
        r = pg().van_recursos(DATA, "duration", ["R"], [1], BAC, 8.0, 12, 0.12)
        e = r["escenarios"]
        libre = van_mensual({"X": 0, "Y": 5, "Z": 0, "W": 0}, DATA["duration"], BAC, 10, 8.0, 12, 0.12)
        self.assertAlmostEqual(e.loc["sin límite", "VAN"], libre, places=9)
        self.assertEqual(e.loc["sin límite", "duración"], 10)
        self.assertEqual(e.loc["con límite", "duración"], 12)
        ini = pg().asignar_multi(DATA, "duration", ["R"], [1], dibujar=False).inicios.to_dict()
        con = van_mensual(ini, DATA["duration"], BAC, 12, 8.0, 12, 0.12)
        self.assertAlmostEqual(e.loc["con límite", "VAN"], con, places=9)

    def test_ampliar_no_empeora_la_duracion(self):
        r = pg().van_recursos(DATA, "duration", ["R"], [1], BAC, 8.0, 12, 0.12, ampliaciones=2)
        a = r["ampliaciones"]
        self.assertEqual(a.loc[0, "duración"], 12)
        self.assertEqual(a.loc[1, "duración"], 10)
        self.assertTrue((a["duración"].diff().dropna() <= 0).all())
        self.assertGreater(a.loc[1, "VAN"], a.loc[0, "VAN"])

    def test_limite_holgado_no_cambia(self):
        r = pg().van_recursos(DATA, "duration", ["R"], [5], BAC, 8.0, 12, 0.12)["escenarios"]
        self.assertAlmostEqual(r.loc["con límite", "VAN"], r.loc["sin límite", "VAN"], places=9)

    def test_monte_carlo_determinista_iguala_al_exacto(self):
        mc = pg().monte_carlo_van_recursos(CTE, DATA[["R"]], ["R"], [1], BAC, 8.0, 12, 0.12, n_samples=20, seed=1)
        e = pg().van_recursos(DATA, "duration", ["R"], [1], BAC, 8.0, 12, 0.12)["escenarios"]
        self.assertIsInstance(mc, MonteCarloVANRecursosResult)
        self.assertTrue(np.allclose(mc.van_libre_samples, e.loc["sin límite", "VAN"]))
        self.assertTrue(np.allclose(mc.van_recursos_samples, e.loc["con límite", "VAN"]))

    def test_monte_carlo_aleatorio_con_recursos_pierde(self):
        dist = {a: ("triangular", max(1, d - 1), d, d + 2) for a, d in DATA["duration"].items()}
        mc = pg().monte_carlo_van_recursos(dist, DATA[["R"]], ["R"], [1], BAC, 8.0, 12, 0.12, n_samples=300, seed=3)
        self.assertGreaterEqual(mc.tau_recursos_samples.min(), 0)
        self.assertTrue((mc.tau_recursos_samples >= mc.tau_libre_samples - 1e-9).all())
        self.assertLess(mc.van_recursos_samples.mean(), mc.van_libre_samples.mean())
        self.assertEqual(list(mc.summary().index), ["VAN sin recursos", "VAN con recursos", "pérdida por recursos"])


class TestCrashingVAN(unittest.TestCase):

    def test_sin_ingresos_no_conviene_reducir(self):
        t = pg().crashing_van(DATA, "duration", MIND, UCR, BAC, 0.0, 12, 0.12)
        self.assertEqual(t["VAN"].idxmax(), 0)
        self.assertTrue((t["coste de reducción"].diff().dropna() >= 0).all())

    def test_ingresos_altos_conviene_reducir_todo(self):
        t = pg().crashing_van(DATA, "duration", MIND, UCR, BAC, 1000.0, 12, 0.12)
        self.assertEqual(t["VAN"].idxmax(), t.index.max())
        self.assertEqual(t.loc[0, "duración"], 10)
        self.assertLess(t["duración"].iloc[-1], 10)

    def test_con_recursos_parte_de_la_duracion_con_recursos(self):
        t = pg().crashing_van(DATA, "duration", MIND, UCR, BAC, 100.0, 12, 0.12, resource_labels=["R"], maximos=[1])
        self.assertEqual(t.loc[0, "duración"], 12)
        self.assertLessEqual(t["duración"].iloc[-1], 10)
        self.assertEqual(t.loc[1, "coste de reducción"], 3)


class TestMonteCarloCrashing(unittest.TestCase):

    def test_determinista_escalon(self):
        t = pg().monte_carlo_crashing(CTE, DATA, "duration", MIND, UCR, plazo=9, reduccion_max=3, n_samples=50, seed=0)
        self.assertEqual(t.loc[0, "P(τ ≤ plazo)"], 0.0)
        self.assertEqual(t.loc[1, "P(τ ≤ plazo)"], 1.0)
        self.assertEqual(t.loc[1, "duración determinista"], 9)
        self.assertEqual(t.loc[1, "τ media"], 9)

    def test_probabilidad_creciente_con_numeros_comunes(self):
        dist = {a: ("triangular", max(1, d - 1), d, d + 2) for a, d in DATA["duration"].items()}
        t = pg().monte_carlo_crashing(dist, DATA, "duration", MIND, UCR, plazo=10, reduccion_max=4, n_samples=2000, seed=2)
        self.assertTrue((t["P(τ ≤ plazo)"].diff().dropna() >= -1e-12).all())
        self.assertTrue((t["τ media"].diff().dropna() <= 1e-12).all())


class TestValorGanadoMC(unittest.TestCase):

    def test_sin_incertidumbre_las_previsiones_son_exactas(self):
        r = pg().monte_carlo_valor_ganado(CTE, DATA["duration"], BAC, t_control=4, n_samples=30, seed=0)
        self.assertIsInstance(r, MonteCarloValorGanadoResult)
        self.assertTrue(np.allclose(r.spi, 1.0))
        self.assertTrue(np.allclose(r.cpi, 1.0))
        self.assertTrue(np.allclose(r.tau_pred_samples, r.tau_samples))
        c = r.calidad_prevision()
        self.assertAlmostEqual(c.loc["duración", "sesgo"], 0.0)
        self.assertAlmostEqual(c.loc["coste", "sesgo"], 0.0)

    def test_duraciones_el_doble_dan_spi_menor_que_uno(self):
        dist = {a: ("triangular", 2 * d, 2 * d, 2 * d) for a, d in DATA["duration"].items()}
        r = pg().monte_carlo_valor_ganado(dist, DATA["duration"], BAC, t_control=4, n_samples=10, seed=0)
        self.assertTrue((r.spi < 1).all())
        self.assertTrue((r.tau_pred_samples > r.T_plan).all())

    def test_coste_sobrecoste_baja_el_cpi(self):
        fc = {a: ("triangular", 1.2, 1.2, 1.2) for a in DATA.index}
        r = pg().monte_carlo_valor_ganado(CTE, DATA["duration"], BAC, t_control=4, n_samples=10, seed=0, factor_coste=fc)
        self.assertTrue(np.allclose(r.cpi, 1 / 1.2))
        self.assertTrue(np.allclose(r.eac_pred_samples, sum(BAC.values()) * 1.2))

    def test_con_dispersion_hay_error_de_prevision(self):
        dist = {a: ("triangular", max(1, d - 2), d, d + 4) for a, d in DATA["duration"].items()}
        r = pg().monte_carlo_valor_ganado(dist, DATA["duration"], BAC, t_control=5, n_samples=1000, seed=4)
        self.assertGreater(r.calidad_prevision().loc["duración", "error absoluto medio"], 0)


class TestValorGanadoRecursos(unittest.TestCase):

    def test_spi_menor_que_uno_con_recursos(self):
        t = pg().valor_ganado_recursos(DATA, "duration", ["R"], [1], BAC)
        self.assertEqual(t.attrs["T_plan"], 10)
        self.assertEqual(t.attrs["T_real"], 12)
        self.assertTrue((t["EV"] <= t["PV"] + 1e-9).all())
        self.assertAlmostEqual(t["PV"].iloc[-1], sum(BAC.values()))
        self.assertAlmostEqual(t["EV"].iloc[-1], sum(BAC.values()))
        self.assertLess(t.loc[10, "SPI"], 1.0)
        self.assertGreater(t.loc[10, "duración prevista"], 10)

    def test_sin_limite_efectivo_spi_uno(self):
        t = pg().valor_ganado_recursos(DATA, "duration", ["R"], [5], BAC)
        self.assertTrue(np.allclose(t["SPI"].dropna(), 1.0))


class TestValorGanadoMCRecursos(unittest.TestCase):

    def test_determinista_con_recursos_la_linea_base_realista_es_exacta(self):
        r = pg().monte_carlo_valor_ganado_recursos(CTE, DATA["duration"], DATA[["R"]], ["R"], [1], BAC,
                                                  t_control=6, n_samples=20, seed=0)
        self.assertEqual(r.T_libre, 10)
        self.assertEqual(r.T_recursos, 12)
        self.assertTrue(np.allclose(r.spi_recursos, 1.0))
        self.assertTrue(np.allclose(r.pred_recursos_samples, 12))
        self.assertTrue(np.all(r.spi_libre < 1.0))
        self.assertTrue(np.allclose(r.tau_samples, 12))

    def test_con_dispersion_la_linea_base_realista_predice_mejor(self):
        dist = {a: ("triangular", max(1, d - 1), d, d + 2) for a, d in DATA["duration"].items()}
        r = pg().monte_carlo_valor_ganado_recursos(dist, DATA["duration"], DATA[["R"]], ["R"], [1], BAC,
                                                  t_control=6, n_samples=500, seed=2)
        q = r.calidad_prevision()
        self.assertLess(q.loc["línea base con recursos", "error absoluto medio"],
                        q.loc["línea base sin recursos", "error absoluto medio"])
        self.assertEqual(r.summary().shape[0], 5)


if __name__ == "__main__":
    unittest.main()
