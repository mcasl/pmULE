"""test_monte_carlo_van.py — distribución del VAN con duraciones aleatorias."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from pmule import ProjectGraph, MonteCarloVANResult

PREDS = {"A": set(), "B": set(), "C": {"A"}, "D": {"A", "B"}, "E": {"B"},
         "F": {"C", "D"}, "G": {"D", "E"}, "H": {"F", "G"}}
DUR = {"A": 3, "B": 2, "C": 4, "D": 5, "E": 3, "F": 2, "G": 6, "H": 4}
BAC = {"A": 30, "B": 20, "C": 40, "D": 50, "E": 30, "F": 20, "G": 60, "H": 40}
F, N, K = 25.0, 36, 0.10


def proyecto():
    return ProjectGraph.from_dict_of_sets(PREDS, simplify=True)


def fijas(dur=DUR):
    return {a: ("uniform", d, d) for a, d in dur.items()}


def van_determinista(pg, dur, bac, F, N, k):
    """VAN por el procedimiento del ejercicio: flujo mensual explícito."""
    cal = pg.calendar(dur)
    T = int(pg.duration(dur))
    km = (1 + k) ** (1 / 12) - 1
    pagos = {m: 0.0 for m in range(1, T + 1)}
    for a, d in dur.items():
        es = int(cal.loc[a, "inicio_mas_temprano"])
        for m in range(es + 1, es + d + 1):
            pagos[m] += bac[a] / d
    flujos = [0.0] + [-pagos[m] for m in range(1, T + 1)] + [F] * N
    return sum(f / (1 + km) ** t for t, f in enumerate(flujos))


class TestEquivalenciaDeterminista(unittest.TestCase):

    def test_duraciones_constantes_dan_el_van_del_flujo_mensual(self):
        pg = proyecto()
        r = pg.monte_carlo_van(fijas(), BAC, F, N, K, n_samples=50, seed=1)
        esperado = van_determinista(pg, DUR, BAC, F, N, K)
        self.assertTrue(np.allclose(r.van_samples, esperado), (r.van_samples[:3], esperado))
        self.assertTrue((r.tau_samples == 18).all())

    def test_otro_proyecto_de_duraciones(self):
        pg = proyecto()
        dur = {**DUR, "G": 9, "C": 1}
        r = pg.monte_carlo_van(fijas(dur), BAC, 31.0, 24, 0.07, n_samples=10, seed=2)
        self.assertTrue(np.allclose(r.van_samples, van_determinista(pg, dur, BAC, 31.0, 24, 0.07)))

    def test_tau_coincide_con_monte_carlo_sin_redondeo(self):
        pg = proyecto()
        dist = {a: ("triangular", max(d - 1, 0.5), d, d + 3) for a, d in DUR.items()}
        mc = pg.monte_carlo(dist, n_samples=500, seed=5)
        r = pg.monte_carlo_van(dist, BAC, F, N, K, n_samples=500, seed=5, redondear=False)
        self.assertTrue(np.allclose(r.tau_samples, mc.tau_samples))


class TestComportamiento(unittest.TestCase):

    def setUp(self):
        self.pg = proyecto()
        self.dist = {a: ("triangular", max(d - 1, 1), d, d + 4) for a, d in DUR.items()}

    def test_reproducible_con_semilla(self):
        a = self.pg.monte_carlo_van(self.dist, BAC, F, N, K, n_samples=300, seed=9)
        b = self.pg.monte_carlo_van(self.dist, BAC, F, N, K, n_samples=300, seed=9)
        self.assertTrue(np.array_equal(a.van_samples, b.van_samples))

    def test_mas_incertidumbre_en_duraciones_baja_la_media(self):
        base = self.pg.monte_carlo_van(fijas(), BAC, F, N, K, n_samples=10, seed=1)
        r = self.pg.monte_carlo_van(self.dist, BAC, F, N, K, n_samples=4000, seed=1)
        self.assertLess(r.van_mean, base.van_mean)      # E[T] > T(E[d]) y el VAN decrece con T
        self.assertGreater(r.van_std, 0)

    def test_van_decrece_con_la_duracion_del_proyecto(self):
        r = self.pg.monte_carlo_van(self.dist, BAC, F, N, K, n_samples=3000, seed=3)
        self.assertLess(np.corrcoef(r.tau_samples, r.van_samples)[0, 1], -0.5)

    def test_penalizacion_resta_solo_con_retraso(self):
        sin = self.pg.monte_carlo_van(self.dist, BAC, F, N, K, n_samples=500, seed=4)
        con = self.pg.monte_carlo_van(self.dist, BAC, F, N, K, n_samples=500, seed=4,
                                      plazo=18, penalizacion_mensual=10)
        self.assertTrue((con.van_samples <= sin.van_samples + 1e-9).all())
        sin_retraso = sin.tau_samples <= 18
        self.assertTrue(np.allclose(con.van_samples[sin_retraso], sin.van_samples[sin_retraso]))
        self.assertLess(con.van_mean, sin.van_mean)

    def test_ingreso_y_coste_aleatorios_ensanchan_la_distribucion(self):
        base = self.pg.monte_carlo_van(self.dist, BAC, F, N, K, n_samples=4000, seed=6)
        ruido = self.pg.monte_carlo_van(self.dist, BAC, ("normal", F, 5.0), N, K, n_samples=4000, seed=6,
                                        bac_distributions={"D": ("triangular", 0.9, 1.0, 1.5)})
        self.assertGreater(ruido.van_std, base.van_std)
        self.assertIn("ingreso mensual", ruido.tornado().index)

    def test_tornado_y_resumen(self):
        r = self.pg.monte_carlo_van(self.dist, BAC, F, N, K, n_samples=4000, seed=7)
        t = r.tornado()
        # las actividades del camino crítico A-D-G-H (18) mandan; E y F tienen holgura
        top = [i.split()[-1] for i in t.index[:4]]
        self.assertEqual(set(top), {"A", "D", "G", "H"})
        self.assertTrue((t["Spearman con el VAN"].iloc[:4] < 0).all())
        s = r.summary()
        self.assertAlmostEqual(s.loc["VAN", "media"], r.van_mean)
        self.assertLessEqual(r.cvar(0.05), r.cuantil(0.05))
        self.assertIsInstance(r, MonteCarloVANResult)

    def test_alargar_una_actividad_con_holgura_mejora_el_van(self):
        """E tiene holgura: alargarla (sin llegar a ser crítica) no retrasa el cobro y solo difiere su pago."""
        pg = self.pg
        v3 = pg.monte_carlo_van(fijas(), BAC, F, N, K, n_samples=5, seed=1).van_samples[0]
        v4 = pg.monte_carlo_van(fijas({**DUR, "E": 4}), BAC, F, N, K, n_samples=5, seed=1).van_samples[0]
        self.assertGreater(v4, v3)

    def test_triangular_degenerada_es_constante(self):
        pg = proyecto()
        dist = {a: ("triangular", d, d, d) for a, d in DUR.items()}
        r = pg.monte_carlo_van(dist, BAC, F, N, K, n_samples=20, seed=1)
        self.assertTrue(np.allclose(r.van_samples, van_determinista(pg, DUR, BAC, F, N, K)))

    def test_errores(self):
        with self.assertRaises(ValueError):
            self.pg.monte_carlo_van({"A": ("uniform", 1, 2)}, BAC, F, N, K, n_samples=10)
        with self.assertRaises(ValueError):
            self.pg.monte_carlo_van(self.dist, {"A": 1}, F, N, K, n_samples=10)


if __name__ == "__main__":
    unittest.main()
