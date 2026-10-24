"""Regresion de las mejoras de rendimiento (24-oct-2026) que deben ser EXACTAS.

- ``RoyDAG.critical_mask`` (barridos in situ y por bloques de columnas), ``RoyDAG.longest_path_batch`` y
  ``tau_rank_k_roy`` por lotes dan exactamente lo mismo, bit a bit, que el calculo directo anterior.
- ``PredecessorTable`` con bitsets da las mismas predecesoras lejanas e inmediatas que la formula con conjuntos.
"""

import random
import time
import unittest

import numpy as np

from pmule import PredecessorTable, ProjectGraph, RoyDAG, implicit_svd_roy, tau_rank_k_roy
from pmule import _bits_a_conjunto, _bloque_columnas, _indices_de_bits, _reduccion_transitiva_desde_ancestros


def dag_aleatorio(n, seed, ventana=6, grado=2.0):
    """DAG con actividades A0..A{n-1}: las aristas solo van de i a j > i (orden topologico creciente)."""
    rng = random.Random(seed)
    acts = [f"A{i}" for i in range(n)]
    aristas = []
    for j in range(1, n):
        cand = list(range(max(0, j - ventana), j))
        k = min(len(cand), max(1, int(rng.expovariate(1.0 / grado)) if rng.random() < 0.9 else 0))
        for i in rng.sample(cand, k):
            aristas.append((acts[i], acts[j]))
    return acts, aristas


# --- referencias: el codigo anterior, tal cual ---
def critical_mask_ref(d, T):
    T = np.asarray(T, float)
    n, m = T.shape
    EF = np.empty((n, m))
    for i in range(n):
        p = d.pred[i]
        EF[i] = (EF[p].max(axis=0) if p.size else 0.0) + T[i]
    tau = EF[d.snk > 0].max(axis=0)
    LS = np.empty((n, m))
    succ = d.A.tolil().rows
    for i in range(n - 1, -1, -1):
        s = succ[i]
        LS[i] = (LS[s].min(axis=0) if s else tau) - T[i]
    return (LS + T - EF) < 1e-9 * np.maximum(tau, 1.0), tau


def tau_rank_k_ref(d, t, V, ks=None):
    t = np.asarray(t, float)
    ks = ks or range(1, V.shape[1] + 1)
    c = V.T @ t
    return {k: d.longest_path(V[:, :k] @ c[:k]) for k in ks}


class TestBarridosExactos(unittest.TestCase):
    def test_critical_mask_identico_al_calculo_directo(self):
        for seed in range(25):
            acts, aristas = dag_aleatorio(5 + seed * 3, seed)
            d = RoyDAG(acts, aristas)
            T = np.random.default_rng(seed).uniform(0.5, 4.0, (d.n, 37))
            ref_mask, ref_tau = critical_mask_ref(d, T)
            for bloque in (None, 1, 5, 36, 37, 1000):
                mask, tau = d.critical_mask(T, block=bloque)
                self.assertTrue(np.array_equal(mask, ref_mask), f"mascara distinta (seed {seed}, bloque {bloque})")
                self.assertTrue(np.array_equal(tau, ref_tau), f"tau distinto (seed {seed}, bloque {bloque})")

    def test_critical_mask_con_empates_exactos(self):
        """Duraciones enteras: muchas rutas empatadas, que es donde la comparacion con tolerancia importa."""
        for seed in range(10):
            acts, aristas = dag_aleatorio(40, 100 + seed)
            d = RoyDAG(acts, aristas)
            T = np.random.default_rng(seed).integers(1, 4, (d.n, 50)).astype(float)
            ref_mask, ref_tau = critical_mask_ref(d, T)
            mask, tau = d.critical_mask(T, block=8)
            self.assertTrue(np.array_equal(mask, ref_mask))
            self.assertTrue(np.array_equal(tau, ref_tau))

    def test_longest_path_batch_identico_columna_a_columna(self):
        for seed in range(25):
            acts, aristas = dag_aleatorio(4 + seed * 3, 200 + seed)
            d = RoyDAG(acts, aristas)
            W = np.random.default_rng(seed).normal(0.0, 3.0, (d.n, 23))        # pesos negativos incluidos
            ref = np.array([d.longest_path(W[:, j]) for j in range(W.shape[1])])
            for bloque in (None, 1, 4, 23):
                self.assertTrue(np.array_equal(d.longest_path_batch(W, block=bloque), ref), f"seed {seed}, bloque {bloque}")

    def test_tau_rank_k_por_lotes_identico_al_bucle(self):
        for seed in range(12):
            acts, aristas = dag_aleatorio(30 + seed * 4, 300 + seed)
            d = RoyDAG(acts, aristas)
            _, V, _ = implicit_svd_roy(d, k=min(10, d.n - 2), dense_max=0)
            t = np.random.default_rng(seed).uniform(1.0, 9.0, d.n)
            self.assertEqual(tau_rank_k_roy(d, t, V), tau_rank_k_ref(d, t, V))
            ks = [1, 3, 5]
            self.assertEqual(tau_rank_k_roy(d, t, V, ks=ks), tau_rank_k_ref(d, t, V, ks=ks))

    def test_succ_se_calcula_una_vez(self):
        acts, aristas = dag_aleatorio(20, 1)
        d = RoyDAG(acts, aristas)
        self.assertIs(d.succ, d.succ)
        self.assertEqual(sum(len(s) for s in d.succ), len(aristas))

    def test_tamano_de_bloque(self):
        self.assertEqual(_bloque_columnas(300, 100000), 4096)
        self.assertEqual(_bloque_columnas(1500, 100000), 1024)
        self.assertEqual(_bloque_columnas(300, 1000), 1000)         # N pequeno: un solo bloque
        for n in (1, 10, 300, 5000, 10**6):
            b = _bloque_columnas(n, 10**7)
            self.assertTrue(512 <= b <= 8192 and (b & (b - 1)) == 0, (n, b))


def inmediatas_de_referencia(pred):
    """Formula con conjuntos: ancestros menos los ancestros de sus ancestros."""
    lej = {}

    def anc(x):
        if x not in lej:
            r = set(pred[x])
            for p in pred[x]:
                r |= anc(p)
            lej[x] = r
        return lej[x]
    for x in pred:
        anc(x)
    inm = {x: lej[x] - set().union(*(lej[a] for a in lej[x])) for x in pred}
    return lej, inm


def pred_aleatoria(n, seed, ventana=8, grado=2.5):
    rng = random.Random(seed)
    nombres = [f"T{i:03d}" for i in range(n)]
    pred = {nombres[0]: set()}
    for j in range(1, n):
        cand = nombres[max(0, j - ventana):j]
        k = rng.randint(0, min(len(cand), int(grado) + 2))
        pred[nombres[j]] = set(rng.sample(cand, k))
    return pred


class TestPredecessorTableBitsets(unittest.TestCase):
    def test_lejanas_e_inmediatas_iguales_a_la_formula_con_conjuntos(self):
        for seed in range(60):
            pred = pred_aleatoria(5 + seed % 40, seed)
            lej, inm = inmediatas_de_referencia(pred)
            t = PredecessorTable(pred, simplify=True)
            self.assertEqual(t.distant_predecessor, lej)
            self.assertEqual(t.immediate_predecessor, inm)
            self.assertEqual(list(t.immediate_predecessor), sorted(pred))        # orden alfabetico conservado
            for x in pred:
                self.assertEqual(t.calculate_immediate_predecessor_of(x), inm[x])

    def test_sin_simplificar_conserva_las_predecesoras_dadas(self):
        pred = pred_aleatoria(30, 7)
        t = PredecessorTable(pred, simplify=False)
        self.assertEqual(t.immediate_predecessor, dict(sorted(pred.items())))
        self.assertEqual(t.distant_predecessor, inmediatas_de_referencia(pred)[0])

    def test_conjuntos_devueltos_independientes(self):
        t = PredecessorTable(pred_aleatoria(20, 3), simplify=True)
        k = next(x for x, v in t.distant_predecessor.items() if v)
        antes = set(t.distant_predecessor[k])
        t.immediate_predecessor[k].add("zzz")
        self.assertEqual(t.distant_predecessor[k], antes)

    def test_grafo_denso_de_800_actividades_es_rapido(self):
        pred = pred_aleatoria(800, 11, ventana=60, grado=12)
        t0 = time.perf_counter()
        t = PredecessorTable(pred, simplify=True)
        dt = time.perf_counter() - t0
        self.assertEqual(len(t.immediate_predecessor), 800)
        self.assertLess(dt, 5.0, f"PredecessorTable tardo {dt:.1f} s")


class TestProyectosGrandes(unittest.TestCase):
    """Cierre en bits de ProjectGraph (distant_predecessor, roy_dag) y simplificacion de ficticias incremental."""

    def test_bits_a_conjunto(self):
        rng = random.Random(5)
        nombres = [f"N{i:04d}" for i in range(700)]
        self.assertEqual(_indices_de_bits(0), [])
        self.assertEqual(_bits_a_conjunto(0, nombres), set())
        for k in (1, 7, 64, 300, 700):
            idx = sorted(rng.sample(range(700), k))
            mask = sum(1 << i for i in idx)
            self.assertEqual(_indices_de_bits(mask), idx)
            self.assertEqual(_bits_a_conjunto(mask, nombres), {nombres[i] for i in idx})

    def test_roy_dag_igual_a_la_reduccion_desde_los_ancestros(self):
        for seed in range(25):
            pred = pred_aleatoria(8 + seed * 3, 500 + seed, ventana=3 + seed % 9, grado=2 + seed % 4)
            pg = ProjectGraph.from_dict_of_sets(pred, simplify=True)
            dp = pg.distant_predecessor(dummies=False)
            acts = list(dp)
            ref = _reduccion_transitiva_desde_ancestros(acts, dp)
            dag = pg.roy_dag()
            self.assertEqual(sorted(dag.acts), acts)      # RoyDAG las reordena topologicamente
            self.assertEqual(sorted((dag.acts[i], dag.acts[j]) for i, j in zip(*dag.A.nonzero())), sorted(ref))

    def test_distant_predecessor_con_y_sin_ficticias(self):
        for seed in range(15):
            pred = pred_aleatoria(10 + seed * 4, 600 + seed, ventana=4 + seed % 8, grado=3)
            pg = ProjectGraph.from_dict_of_sets(pred, simplify=True)
            con = pg.distant_predecessor(dummies=True)
            sin = pg.distant_predecessor(dummies=False)
            self.assertEqual(sorted(con), pg.activities)
            self.assertEqual(sorted(sin), pg.actual_activities)
            for a in sin:
                self.assertEqual(sin[a], {x for x in con[a] if x[0] != "@"})
            # las predecesoras distantes de las actividades reales son las de la tabla de precedencias original
            self.assertEqual(sin, PredecessorTable(pred, simplify=True).distant_predecessor)

    def test_proyecto_de_800_actividades_se_crea_y_se_analiza_rapido(self):
        pred = pred_aleatoria(800, 17, ventana=12, grado=5)
        t0 = time.perf_counter()
        pg = ProjectGraph.from_dict_of_sets(pred, simplify=True)
        t1 = time.perf_counter()
        dp = pg.distant_predecessor(dummies=False)
        dag = pg.roy_dag()
        t2 = time.perf_counter()
        self.assertEqual(len(dp), 800)
        self.assertEqual(dag.n, 800)
        self.assertLess(t1 - t0, 20.0, f"crear el proyecto tardo {t1 - t0:.1f} s")
        self.assertLess(t2 - t1, 5.0, f"distant_predecessor + roy_dag tardaron {t2 - t1:.1f} s")


if __name__ == "__main__":
    unittest.main()
