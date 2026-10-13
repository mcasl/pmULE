"""
proyectos.py — generación de proyectos aleatorios pequeños con control de
rasgos (número de ficticias, de rutas, de caminos críticos, holguras, …).

Cada plantilla pide un proyecto con `buscar(rng, acepta, **kwargs)`, que
muestrea proyectos hasta que `acepta(info)` es True. `info` reúne la tabla de
prelaciones (dict de sets), la tabla de datos y los resultados calculados con
pmULE (calendario, rutas, matriz de caminos, …).
"""
from __future__ import annotations

import contextlib
import io
import string
import warnings

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pmule import PredecessorTable  # noqa: E402  (sys.path lo fija generador.py)


# ---------------------------------------------------------------------------
# Grafo de prelaciones
# ---------------------------------------------------------------------------
def dag_aleatorio(rng, n, n_inicio=2, n_fin=None, max_pred=3, ventana=4):
    """Prelaciones inmediatas (dict actividad -> set) sin redundancias, con
    `n_inicio` actividades iniciales y, si se indica, `n_fin` finales."""
    nombres = list(string.ascii_uppercase[:n])
    preds = {a: set() for a in nombres}
    for i in range(n_inicio, n):
        k = int(rng.integers(1, max_pred + 1))
        lo = max(0, i - ventana)
        cand = list(range(lo, i))
        elegidos = rng.choice(cand, size=min(k, len(cand)), replace=False)
        preds[nombres[i]] = {nombres[j] for j in elegidos}
    # garantizar que las iniciales son exactamente las n_inicio primeras
    for i in range(n_inicio, n):
        if not preds[nombres[i]]:
            preds[nombres[i]] = {nombres[i - 1]}
    if n_fin is not None:
        for _ in range(3 * n):
            suc = sucesoras(preds)
            finales = [a for a in nombres if not suc[a]]
            if len(finales) <= n_fin:
                break
            a = finales[0]
            idx = nombres.index(a)
            posteriores = [b for b in nombres[idx + 1:]]
            if not posteriores:
                break
            b = posteriores[int(rng.integers(0, len(posteriores)))]
            preds[b].add(a)
    return reducir(preds)


def sucesoras(preds):
    suc = {a: set() for a in preds}
    for a, ps in preds.items():
        for p in ps:
            suc[p].add(a)
    return suc


def grafo(preds):
    G = nx.DiGraph()
    G.add_nodes_from(preds)
    G.add_edges_from((p, a) for a, ps in preds.items() for p in ps)
    return G


def reducir(preds):
    """Elimina las prelaciones redundantes (reducción transitiva)."""
    TR = nx.transitive_reduction(grafo(preds))
    return {a: set(TR.predecessors(a)) for a in preds}


def redundantes_posibles(preds):
    """Pares (p, a) tales que p es antecesora distante pero no inmediata de a."""
    G = grafo(preds)
    pares = []
    for a in preds:
        ancestros = nx.ancestors(G, a)
        for p in ancestros - preds[a]:
            pares.append((p, a))
    return pares


def a_strings(preds):
    return {a: (",".join(sorted(ps)) if ps else "---") for a, ps in preds.items()}


# ---------------------------------------------------------------------------
# Datos de las actividades
# ---------------------------------------------------------------------------
def duraciones(rng, nombres, lo=1, hi=8):
    return pd.Series(rng.integers(lo, hi + 1, size=len(nombres)), index=nombres, name="duration")


def tres_estimaciones(rng, D):
    """(a, m, b) enteros cuya media PERT (a+4m+b)/6 es exactamente D."""
    a, m, b = {}, {}, {}
    for act, d in D.items():
        d = int(d)
        forma = rng.choice(["sim", "sim", "der", "izq"])
        if forma == "der":            # cola derecha larga: m = D - 1
            p = int(rng.integers(1, 4))
            q = p + 4
        elif forma == "izq":          # cola izquierda larga: m = D + 1
            q = int(rng.integers(1, 3))
            p = q + 4
        else:                         # simétrica: m = D
            p = q = int(rng.integers(1, 4))
        if d - p < 1:                 # a >= 1: volver a la simétrica más estrecha
            p = q = max(d - 1, 0)
        a[act], b[act] = d - p, d + q
        m[act] = (6 * d - a[act] - b[act]) / 4
        assert float(m[act]).is_integer(), (d, a[act], b[act])
        m[act] = int(m[act])
        assert a[act] <= m[act] <= b[act], (a[act], m[act], b[act])
    return pd.Series(a, name="a"), pd.Series(m, name="m"), pd.Series(b, name="b")


def tres_estimaciones_triangular(rng, D):
    """(a, m, b) enteros cuya media triangular (a+m+b)/3 es exactamente D."""
    a, m, b = {}, {}, {}
    for act, d in D.items():
        d = int(d)
        forma = rng.choice(["sim", "sim", "der", "izq"])
        if forma == "der":            # m = D - 1, cola derecha
            p = int(rng.integers(1, 4)); q = p + 1
        elif forma == "izq":          # m = D + 1, cola izquierda
            p = int(rng.integers(2, 5)); q = p - 1
        else:
            p = q = int(rng.integers(1, 4))
        if d - p < 1:
            p = q = max(d - 1, 0)
        a[act], b[act] = d - p, d + q
        m[act] = 3 * d - a[act] - b[act]
        assert a[act] <= m[act] <= b[act], (a[act], m[act], b[act])
    return pd.Series(a, name="a"), pd.Series(m, name="m"), pd.Series(b, name="b")


def desviaciones(rng, nombres):
    return pd.Series(np.round(rng.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], size=len(nombres)), 1),
                     index=nombres, name="standard_deviation")


def recursos(rng, nombres, hi=5):
    return pd.Series(rng.integers(1, hi + 1, size=len(nombres)), index=nombres, name="resources")


def costes_reduccion(rng, D, n_inf=0):
    """duración tope, coste normal, coste tope y CUR."""
    tope, cn, ct, cur = {}, {}, {}, {}
    inf_en = set(rng.choice(list(D.index), size=n_inf, replace=False)) if n_inf else set()
    for act, d in D.items():
        d = int(d)
        red = int(rng.integers(0, min(3, d - 1) + 1)) if d > 1 else 0
        tope[act] = d - red
        cn[act] = int(rng.integers(2, 10)) * 10
        if act in inf_en:
            tope[act] = d
        if tope[act] == d:
            cur[act] = np.inf if act in inf_en else 0.0
            ct[act] = cn[act]
        else:
            cur[act] = int(rng.integers(1, 10))
            ct[act] = cn[act] + cur[act] * red
    return (pd.Series(tope, name="cap_duration"), pd.Series(cn, name="cost_normal"),
            pd.Series(ct, name="cost_cap"), pd.Series(cur, name="ucr"))


# ---------------------------------------------------------------------------
# Evaluación con pmULE
# ---------------------------------------------------------------------------
def analizar(preds, D=None):
    """Construye el proyecto con pmULE y devuelve un dict con los resultados."""
    with contextlib.redirect_stdout(io.StringIO()):
        tabla = PredecessorTable.from_dict_of_strings(a_strings(preds))
        proyecto = tabla.create_project()
    info = dict(preds=preds, tabla=tabla, proyecto=proyecto,
                n_act=len(preds), dummies=list(proyecto.dummies),
                n_dummies=len(proyecto.dummies), n_nodos=len(proyecto.nodes),
                rutas=proyecto.paths(dummies=False),
                P=proyecto.path_matrix(dummies=False),
                iniciales=[a for a, ps in preds.items() if not ps],
                finales=[a for a, s in sucesoras(preds).items() if not s])
    info["n_rutas"] = len(info["rutas"])
    if D is not None:
        D = D.reindex(list(preds)).astype(int)
        with contextlib.redirect_stdout(io.StringIO()):
            cal = proyecto.calendar(D)
            L = (info["P"] @ D[info["P"].columns]).astype(float)
        T = float(L.max())
        criticas = L[np.isclose(L, T)]
        info.update(D=D, calendario=cal, L=L, T=T, n_criticas=int(len(criticas)),
                    rutas_criticas=list(criticas.index),
                    segunda=float(sorted(L.values)[-len(criticas) - 1]) if len(L) > len(criticas) else None)
    return info


def buscar(rng, acepta, n=9, n_inicio=2, n_fin=None, con_duraciones=True,
           max_intentos=3000, **kw):
    """Muestrea proyectos hasta que `acepta(info)` sea True."""
    ultimo = None
    for _ in range(max_intentos):
        preds = dag_aleatorio(rng, n, n_inicio=n_inicio, n_fin=n_fin, **kw)
        D = duraciones(rng, list(preds)) if con_duraciones else None
        try:
            info = analizar(preds, D)
        except Exception:
            continue
        ultimo = info
        if acepta(info):
            return info
    raise RuntimeError("No se encontró un proyecto con los rasgos pedidos; "
                       f"último: dummies={ultimo['n_dummies']} rutas={ultimo['n_rutas']}")


def tabla_base(info, **columnas):
    """DataFrame de enunciado: predecessor (+ columnas extra), índice 'activity'."""
    df = pd.DataFrame({"predecessor": pd.Series(a_strings(info["preds"]))})
    for nombre, serie in columnas.items():
        df[nombre] = pd.Series(serie).reindex(df.index)
    df.index.name = "activity"
    return df
