"""Regresion: los predecesores lejanos no deben depender del numero de rutas.

La version recursiva sin memoria de ``PredecessorTable.calculate_distant_predecessor_of``
hacia un trabajo proporcional al numero de rutas del grafo (exponencial en una escalera
de rombos). Estos tests fijan el resultado correcto y el coste lineal.
"""

import random
import time
import tracemalloc
import unittest

import networkx as nx

from pmule import PredecessorTable, ProjectGraph, _reduccion_transitiva_desde_ancestros


def escalera_de_rombos(k):
    """k rombos en serie: 2**k rutas, 3k+1 actividades."""
    pred = {"S": set()}
    ant = "S"
    for i in range(k):
        a, b, c = f"a{i}", f"b{i}", f"c{i}"
        pred[a] = {ant}
        pred[b] = {ant}
        pred[c] = {a, b}
        ant = c
    return pred


def lejanos_de_referencia(pred):
    """Cierre transitivo ingenuo (solo para grafos pequenos)."""
    def anc(x):
        r = set(pred[x])
        for p in pred[x]:
            r |= anc(p)
        return r
    return {x: anc(x) for x in pred}


class TestPredecesoresLejanos(unittest.TestCase):
    def test_coincide_con_referencia_en_grafos_aleatorios(self):
        rng = random.Random(7)
        for _ in range(60):
            n = rng.randint(2, 14)
            nombres = [f"N{i:02d}" for i in range(n)]
            pred = {x: {y for y in nombres[:i] if rng.random() < 0.35} for i, x in enumerate(nombres)}
            tabla = PredecessorTable(pred, simplify=True)
            esperado = lejanos_de_referencia(pred)
            self.assertEqual(tabla.distant_predecessor, dict(sorted(esperado.items())))
            for x in nombres:
                self.assertEqual(tabla.calculate_distant_predecessor_of(x), esperado[x])

    def test_escalera_con_2_a_la_40_rutas_es_rapida(self):
        pred = escalera_de_rombos(40)
        t0 = time.perf_counter()
        tabla = PredecessorTable(pred, simplify=True)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 5.0, f"tardo {dt:.1f} s: coste ligado al numero de rutas")
        ultimo = "c39"
        self.assertEqual(tabla.distant_predecessor[ultimo], set(pred) - {ultimo})
        self.assertEqual(len(tabla.distant_predecessor["S"]), 0)

    def test_cadena_larga_sin_limite_de_recursion(self):
        n = 3000  # mayor que el limite de recursion de Python
        pred = {f"T{i:05d}": ({f"T{i - 1:05d}"} if i else set()) for i in range(n)}
        tabla = PredecessorTable(pred, simplify=False)
        self.assertEqual(len(tabla.distant_predecessor[f"T{n - 1:05d}"]), n - 1)

    def test_conjuntos_devueltos_son_independientes(self):
        tabla = PredecessorTable(escalera_de_rombos(3), simplify=True)
        d = tabla.calculate_distant_predecessor_of("c2")
        d.add("intruso")
        self.assertNotIn("intruso", tabla.distant_predecessor["c2"])
        self.assertNotIn("intruso", tabla.calculate_distant_predecessor_of("c2"))
        self.assertNotIn("intruso", tabla.distant_predecessor["c1"])
        otro = {k: id(v) for k, v in tabla.distant_predecessor.items()}
        self.assertEqual(len(set(otro.values())), len(otro))

    def test_ciclo_sigue_dando_valueerror(self):
        with self.assertRaises(ValueError):
            PredecessorTable({"A": {"C"}, "B": {"A"}, "C": {"B"}}, simplify=True)
        tabla = PredecessorTable({"A": set(), "B": {"A"}}, simplify=True)
        tabla.immediate_predecessor["A"] = {"B"}  # se introduce un ciclo a posteriori
        with self.assertRaises(ValueError):
            tabla.calculate_distant_predecessor_of("B")

    def test_memoria_no_supera_la_del_resultado(self):
        pred = escalera_de_rombos(60)  # 181 actividades, 2**60 rutas
        tracemalloc.start()
        tabla = PredecessorTable(pred, simplify=False)
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elementos = sum(len(v) for v in tabla.distant_predecessor.values())
        self.assertLess(pico, 400 * max(elementos, 1) + 2_000_000)


def por_capas(n, ancho, prob, semilla=0):
    """Grafo tipo PSPLIB: cada actividad depende de actividades de las 3 capas anteriores."""
    rng = random.Random(semilla)
    capas = [list(range(i, min(i + ancho, n))) for i in range(0, n, ancho)]
    pred = {f"A{i:04d}": set() for i in range(n)}
    for li in range(1, len(capas)):
        cand = [u for lj in range(max(0, li - 3), li) for u in capas[lj]]
        for v in capas[li]:
            k = max(1, sum(rng.random() < prob for _ in cand))
            pred[f"A{v:04d}"] = {f"A{u:04d}" for u in rng.sample(cand, min(k, len(cand)))}
    return pred


class TestProjectGraphDistantPredecessor(unittest.TestCase):
    def test_distant_predecessor_coincide_con_el_calculo_por_actividad(self):
        for semilla in range(8):
            pg = ProjectGraph.from_dict_of_sets(por_capas(25, 4, 0.4, semilla), simplify=True)
            for dummies in (False, True):
                esperado = {
                    a: pg.calculate_distant_predecessor_of(a, dummies=dummies)
                    for a in (pg.activities if dummies else pg.actual_activities)
                }
                self.assertEqual(pg.distant_predecessor(dummies=dummies), esperado)

    def test_diccionarios_de_nodos_y_arcos_son_inversos(self):
        pg = ProjectGraph.from_dict_of_sets(por_capas(25, 4, 0.4, 3), simplify=True)
        nfe, efn = pg.nodes_from_edges, pg.edges_from_nodes
        self.assertEqual(len(nfe), len(pg.activities))
        for actividad, (u, v) in nfe.items():
            self.assertEqual(efn[(u, v)], actividad)

    def test_proyecto_de_300_actividades_es_rapido(self):
        t0 = time.perf_counter()
        pg = ProjectGraph.from_dict_of_sets(por_capas(300, 10, 0.2), simplify=True)
        dist = pg.distant_predecessor(dummies=False)
        dt = time.perf_counter() - t0
        self.assertEqual(len(dist), 300)
        self.assertLess(dt, 20.0, f"tardo {dt:.1f} s")

    def test_grafo_con_ciclo_usa_el_recorrido_por_actividad(self):
        # assess_dummy prueba contracciones que pueden crear ciclos: no debe fallar.
        pg = ProjectGraph.from_dict_of_sets(por_capas(12, 3, 0.5, 1), simplify=True)
        g = pg.pert_graph.copy()
        u, v = next(iter(g.edges))
        g.add_edge(v, u, activity="CICLO")
        ciclico = ProjectGraph(g)
        resultado = ciclico.distant_predecessor(dummies=True)
        for a in ciclico.activities:
            self.assertEqual(resultado[a], ciclico.calculate_distant_predecessor_of(a, dummies=True))


class TestReduccionTransitiva(unittest.TestCase):
    def test_coincide_con_networkx_en_grafos_aleatorios(self):
        rng = random.Random(11)
        for _ in range(80):
            n = rng.randint(1, 18)
            nombres = [f"N{i:02d}" for i in range(n)]
            pred = {x: {y for y in nombres[:i] if rng.random() < 0.4} for i, x in enumerate(nombres)}
            cierre = lejanos_de_referencia(pred)
            G = nx.DiGraph(); G.add_nodes_from(nombres)
            G.add_edges_from((p, a) for a, ps in cierre.items() for p in ps)
            esperado = set(nx.transitive_reduction(G).edges())
            self.assertEqual(set(_reduccion_transitiva_desde_ancestros(nombres, cierre)), esperado)

    def test_roy_dag_conserva_las_precedencias_inmediatas(self):
        for semilla in range(5):
            pg = ProjectGraph.from_dict_of_sets(por_capas(30, 5, 0.35, semilla), simplify=True)
            dag = pg.roy_dag()
            obtenidas = {a: set() for a in dag.acts}
            for u, v in zip(*dag.A.nonzero()):
                obtenidas[dag.acts[v]].add(dag.acts[u])
            tabla = PredecessorTable(pg.distant_predecessor(dummies=False), simplify=True)
            self.assertEqual(obtenidas, tabla.immediate_predecessor)

    def test_roy_dag_de_600_actividades_es_rapido(self):
        pg = ProjectGraph.from_dict_of_sets(por_capas(600, 12, 0.15), simplify=True)
        t0 = time.perf_counter()
        pg.roy_dag()
        dt = time.perf_counter() - t0
        self.assertLess(dt, 5.0, f"roy_dag tardo {dt:.1f} s")


if __name__ == "__main__":
    unittest.main()
