"""Criticidad blanda (softmax) por DP: coincide con la enumeracion de R, y sus limites son el conteo y el CPM."""
import unittest
import numpy as np
from pmule import ProjectGraph, RoyDAG


class TestSoftCriticality(unittest.TestCase):
    PREDS = {"A": set(), "B": set(), "C": {"A", "B"}, "D": {"B"}, "E": {"C"}, "F": {"C", "D"}, "G": {"E", "F"}, "H": {"D"}}
    T = {"A": 3, "B": 2, "C": 4, "D": 5, "E": 6, "F": 1, "G": 2, "H": 7}

    def setUp(self):
        self.proj = ProjectGraph.from_dict_of_sets(self.PREDS, simplify=True)
        self.R = self.proj.path_matrix(dummies=False)
        self.t = np.array([self.T[a] for a in self.R.columns], float)

    def _enum(self, beta):
        L = self.R.values @ self.t; w = np.exp(beta * (L - L.max())); w /= w.sum()
        return w @ self.R.values, L.max() + np.log(np.exp(beta * (L - L.max())).sum()) / beta

    def test_coincide_con_enumeracion(self):
        for beta in (0.1, 0.5, 2.0, 7.0):
            g, tb = self._enum(beta)
            s = self.proj.soft_criticality(self.T, beta).reindex(self.R.columns).values
            np.testing.assert_allclose(s, g, atol=1e-12)
            self.assertAlmostEqual(self.proj.tau_soft(self.T, beta), tb, places=10)

    def test_gradiente_es_la_criticidad_blanda(self):
        beta, h = 0.8, 1e-6; s = self.proj.soft_criticality(self.T, beta)
        for a in self.PREDS:
            up = dict(self.T); up[a] += h; dn = dict(self.T); dn[a] -= h
            fd = (self.proj.tau_soft(up, beta) - self.proj.tau_soft(dn, beta)) / (2 * h)
            self.assertAlmostEqual(fd, s[a], places=6)

    def test_limite_beta_cero_es_conteo_de_cadenas(self):
        dag = self.proj.roy_dag(); s = self.proj.soft_criticality(self.T, 0.0)
        esperado = self.R.values.sum(axis=0) / len(self.R)
        np.testing.assert_allclose(s.reindex(self.R.columns).values, esperado, atol=1e-12)
        np.testing.assert_allclose(dag.pathcount / dag.n_paths, s.reindex(dag.acts).values, atol=1e-12)

    def test_limite_beta_grande_es_fraccion_de_rutas_criticas(self):
        L = self.R.values @ self.t; crit = np.isclose(L, L.max()); esperado = self.R.values[crit].mean(axis=0)
        s = self.proj.soft_criticality(self.T, 200.0).reindex(self.R.columns).values      # sin desbordamiento (dominio log)
        np.testing.assert_allclose(s, esperado, atol=1e-6)
        self.assertAlmostEqual(self.proj.tau_soft(self.T, 200.0), L.max(), places=6)

    def test_empates_reparten_a_partes_iguales(self):
        proj = ProjectGraph.from_dict_of_sets({"A": set(), "B": {"A"}, "C": {"A"}, "D": {"B", "C"}}, simplify=True)
        s = proj.soft_criticality({"A": 1, "B": 3, "C": 3, "D": 1}, 50.0)
        self.assertAlmostEqual(s["B"], 0.5, places=9); self.assertAlmostEqual(s["C"], 0.5, places=9)
        self.assertAlmostEqual(s["A"], 1.0, places=9); self.assertAlmostEqual(s["D"], 1.0, places=9)

    def test_varios_beta_a_la_vez(self):
        betas = [0.0, 0.3, 3.0]; df = self.proj.soft_criticality(self.T, betas)
        for b in betas: np.testing.assert_allclose(df[b].values, self.proj.soft_criticality(self.T, b).values, atol=1e-14)

    def test_rutas_exponenciales(self):
        """30 rombos en serie (2**30 rutas): sin R; con duraciones iguales cada rama de cada rombo tiene criticidad 1/2."""
        preds = {"x0": set()}
        for i in range(30):
            preds[f"a{i}"] = {f"x{i}"}; preds[f"b{i}"] = {f"x{i}"}; preds[f"x{i + 1}"] = {f"a{i}", f"b{i}"}
        dag = RoyDAG(list(preds), [(p, a) for a, ps in preds.items() for p in ps])
        crit, tb = dag.soft_criticality(np.ones(dag.n), [0.0, 1.0])
        for j in range(2):
            for k, a in enumerate(dag.acts):
                if a[0] in "ab": self.assertAlmostEqual(crit[k, j], 0.5, places=9)
                else: self.assertAlmostEqual(crit[k, j], 1.0, places=9)
        # con una rama mas larga y beta alto, la mayor parte de la masa va a ella
        t = np.array([2.0 if a.startswith("a") else 1.0 for a in dag.acts]); crit, tb = dag.soft_criticality(t, 8.0)
        self.assertTrue(all(crit[k, 0] > 0.999 for k, a in enumerate(dag.acts) if a.startswith("a")))
        self.assertAlmostEqual(tb[0], dag.longest_path(t), delta=30 * np.log(2) / 8.0 + 1e-9)

    def test_beta_negativo_rechazado(self):
        with self.assertRaises(ValueError): self.proj.soft_criticality(self.T, -1.0)

    def test_auto_usa_beta_c_sobre_mu_con_suelo(self):
        mu = np.mean([self.T[a] for a in self.proj.roy_dag().acts])
        for cv, c in ((0.25, 4.0), (0.5, 2.0), (1.0, 1.0), (3.0, 1.0), (7.3, 1.0)):
            esperado = self.proj.soft_criticality(self.T, c / mu)
            pd_auto = self.proj.soft_criticality_auto(self.T, cv)
            np.testing.assert_allclose(pd_auto.reindex(esperado.index).values, esperado.values, atol=1e-14)

    def test_auto_rechaza_cv_no_positivo(self):
        for cv in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError): self.proj.soft_criticality_auto(self.T, cv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
