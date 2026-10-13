"""
cuaderno.py — utilidades para emitir cuadernos jupytext (py:percent) con el
mismo guion que los Ejercicio_NN de pmULE: cabecera quarto, imports, enunciado
con `pregunta_N`, tabla de datos literal y resolución por apartados.
"""
from __future__ import annotations

import math
from textwrap import dedent

import numpy as np
import pandas as pd

CABECERA = """\
# ---
# title: {titulo}
# subtitle: "{subtitulo}"
# coleccion: {coleccion}
# variante: {variante}
# nivel: {nivel}
# lang: es
# documentclass: scrreprt
# papersize: A4
# classoption: []
# execute:
#     echo: False
#     enabled: true
#     freeze: false
# toc: true
# format:
#     html:
#         theme: cosmo
#         embed-resources: true
#         fig-responsive: true
# warning: false
# message: false
# jupyter: python3
# ---
"""

PREAMBULO = """\
import os, sys
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, Markdown, Math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Versión local de la librería: se busca src/pmule.py subiendo desde el directorio actual
_aqui = os.path.abspath('.')
while _aqui != os.path.dirname(_aqui) and not os.path.isfile(os.path.join(_aqui, 'src', 'pmule.py')):
    _aqui = os.path.dirname(_aqui)
sys.path.append(os.path.join(_aqui, 'src'))
from pmule import (PredecessorTable, ProjectGraph, EarnedValue,
                   pinta_tikz, plot_cargas, plot_curvas_valor_ganado,
                   tanto_por_uno, ordena_rutas)

GANTT_PARAMS = dict(arrows=True)   # flechas de prelación en los diagramas de Gantt (rojas las críticas)
"""

DETALLES = r'''
# Funciones auxiliares para mostrar los cálculos intermedios paso a paso
from itertools import combinations as _combinations


def detalle_tiempos_nodos(proyecto, duraciones):
    """Tiempos tempranos y tardíos de los nodos con los candidatos que se comparan en cada uno."""
    G = proyecto.pert_graph
    d = {data['activity']: (0 if data['activity'].startswith('@') else int(duraciones[data['activity']]))
         for _, _, data in G.edges(data=True)}
    nodos = list(proyecto.nodes)
    early, filas = {}, []
    for j in nodos:
        entrantes = [(i, data['activity']) for i, _, data in G.in_edges(j, data=True)]
        if not entrantes:
            early[j] = 0
            filas.append({'nodo': j, 'candidatos t_i + d_ij': 'nodo inicial', 'tiempo temprano = máx': 0})
            continue
        cand = {f'{a}: {early[i]} + {d[a]} = {early[i] + d[a]}': early[i] + d[a] for i, a in entrantes}
        early[j] = max(cand.values())
        filas.append({'nodo': j, 'candidatos t_i + d_ij': '; '.join(cand), 'tiempo temprano = máx': early[j]})
    tempranos = pd.DataFrame(filas).set_index('nodo')
    late, filas = {}, []
    for i in reversed(nodos):
        salientes = [(j, data['activity']) for _, j, data in G.out_edges(i, data=True)]
        if not salientes:
            late[i] = early[i]
            filas.append({'nodo': i, 'candidatos t*_j − d_ij': 'nodo final (= su tiempo temprano)', 'tiempo tardío = mín': late[i]})
            continue
        cand = {f'{a}: {late[j]} − {d[a]} = {late[j] - d[a]}': late[j] - d[a] for j, a in salientes}
        late[i] = min(cand.values())
        filas.append({'nodo': i, 'candidatos t*_j − d_ij': '; '.join(cand), 'tiempo tardío = mín': late[i]})
    tardios = pd.DataFrame(filas).set_index('nodo')
    return tempranos, tardios


def detalle_holguras(calendario, prelaciones):
    """Cómo se obtiene cada fecha y cada holgura a partir de las vecinas."""
    pred = prelaciones.immediate_predecessor
    suc = {a: sorted(s for s, ps in pred.items() if a in ps) for a in calendario.index}
    fin = int(calendario['fin_mas_temprano'].max())
    filas = []
    for a in calendario.index:
        ES, EF = int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'fin_mas_temprano'])
        LS, LF = int(calendario.loc[a, 'inicio_mas_tardio']), int(calendario.loc[a, 'fin_mas_tardio'])
        dur = int(calendario.loc[a, 'duracion'])
        ps, ss = sorted(pred[a]), suc[a]
        ef_pred = {p: int(calendario.loc[p, 'fin_mas_temprano']) for p in ps}
        es_suc = {s: int(calendario.loc[s, 'inicio_mas_temprano']) for s in ss}
        lf_pred = {p: int(calendario.loc[p, 'fin_mas_tardio']) for p in ps}
        ls_suc = {s: int(calendario.loc[s, 'inicio_mas_tardio']) for s in ss}
        min_es_suc = min(es_suc.values()) if es_suc else fin
        max_lf_pred = max(lf_pred.values()) if lf_pred else 0
        filas.append({
            'actividad': a,
            'ES = máx EF pred.': (f"máx({', '.join(f'{p}: {v}' for p, v in ef_pred.items())}) = {ES}" if ps else f'inicial → {ES}'),
            'EF = ES + d': f'{ES} + {dur} = {EF}',
            'LF = mín LS suc.': (f"mín({', '.join(f'{s}: {v}' for s, v in ls_suc.items())}) = {LF}" if ss else f'final → {LF}'),
            'LS = LF − d': f'{LF} − {dur} = {LS}',
            'H_total = LS − ES': f'{LS} − {ES} = {LS - ES}',
            'H_libre = mín ES suc. − EF': f"{min_es_suc}{'' if ss else ' (fin)'} − {EF} = {min_es_suc - EF}",
            'H_indep. = máx(0, mín ES suc. − máx LF pred. − d)': f'máx(0, {min_es_suc} − {max_lf_pred} − {dur}) = {max(0, min_es_suc - max_lf_pred - dur)}',
        })
    return pd.DataFrame(filas).set_index('actividad')


def detalle_rutas(R, valores, nombre='duración', decimales=0):
    """Suma término a término de un valor (duraciones, varianzas…) a lo largo de cada ruta."""
    filas = {}
    for r in R.index:
        acts = list(R.columns[R.loc[r] == 1])
        total = float(sum(valores[a] for a in acts))
        filas[r] = {'actividades': ' - '.join(acts),
                    f'{nombre} = suma': ' + '.join(f'{a}({valores[a]:.{decimales}f})' for a in acts) + f' = {total:.{decimales}f}',
                    nombre: total}
    return pd.DataFrame(filas).T


def detalle_estimaciones(datos, a='a', m='m', b='b'):
    """Media y varianza PERT de cada actividad con los números sustituidos."""
    filas = {}
    for act, fila in datos.iterrows():
        A, M, B = float(fila[a]), float(fila[m]), float(fila[b])
        filas[act] = {'D = (a + 4m + b)/6': f'({A:g} + 4·{M:g} + {B:g})/6 = {(A + 4 * M + B) / 6:.2f}',
                      'σ² = ((b − a)/6)²': f'(({B:g} − {A:g})/6)² = {((B - A) / 6) ** 2:.3f}'}
    return pd.DataFrame(filas).T


def detalle_ackoff(proyecto, duraciones, topes, cur, pasos):
    """En cada paso de Ackoff-Sasieni: rutas críticas, opciones consideradas (actividades comunes y parejas)
    con su coste, y la elegida."""
    R = proyecto.path_matrix(dummies=False)
    d = duraciones.astype(float).copy()
    filas = []
    for paso, elegidas in sorted(pasos.items()):
        L = R @ d[R.columns]
        T = float(L.max())
        criticas = [r for r in L.index if np.isclose(L[r], T)]
        conjuntos = [set(R.columns[R.loc[r] == 1]) for r in criticas]
        con_margen = {x for x in R.columns if d[x] > topes[x] and np.isfinite(cur[x])}
        opciones = {x: float(cur[x]) for x in set.intersection(*conjuntos) & con_margen}
        if len(conjuntos) > 1:
            cand = sorted(set().union(*conjuntos) & con_margen)
            for p, q in _combinations(cand, 2):
                if all(p in s or q in s for s in conjuntos) and p not in opciones and q not in opciones:
                    opciones[f'{p}+{q}'] = float(cur[p] + cur[q])
        orden = sorted(opciones.items(), key=lambda kv: kv[1])
        filas.append({'paso': paso + 1, 'duración': int(T), 'rutas críticas': ', '.join(criticas),
                      'opciones (coste)': '; '.join(f'{k} ({v:.0f})' for k, v in orden[:6]) + (' …' if len(orden) > 6 else ''),
                      'elegida': ', '.join(elegidas), 'coste del paso': float(sum(cur[x] for x in elegidas))})
        for x in elegidas:
            d[x] -= 1
    return pd.DataFrame(filas).set_index('paso')


def detalle_valor_ganado(datos_vg, calendario, semana):
    """PV, EV y AC actividad a actividad en el punto de control."""
    filas = {}
    for a, fila in datos_vg.iterrows():
        es, dur = int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'duracion'])
        frac = min(max(semana - es, 0), dur) / dur
        filas[a] = {'BAC': float(fila['BAC']), 'planificado hasta la semana': f'{min(max(semana - es, 0), dur)} de {dur}',
                    'PV = BAC·fracción': f'{fila["BAC"]:.0f}·{frac:.2f} = {fila["BAC"] * frac:.1f}',
                    'realizado (%)': int(fila['performed']),
                    'EV = BAC·%': f'{fila["BAC"]:.0f}·{fila["performed"] / 100:.2f} = {fila["BAC"] * fila["performed"] / 100:.1f}',
                    'AC': float(fila['AC']), 'PV_i': fila['BAC'] * frac, 'EV_i': fila['BAC'] * fila['performed'] / 100}
    t = pd.DataFrame(filas).T
    return t


def muestra_formula(nombre, expresion, valor, unidades=''):
    """Una línea «nombre = expresión = valor» en Markdown."""
    return Markdown(f'**{nombre}** = {expresion} = **{valor}**{(" " + unidades) if unidades else ""}')
'''

TEMP = """\
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp
"""

BMATRIX = r'''
def bmatrix(a, dec=0):
    """Matriz LaTeX (bmatrix) a partir de un DataFrame o array."""
    a = np.asarray(getattr(a, 'values', a), dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    filas = [' & '.join(f'{x:.{dec}f}' for x in fila) + r'\\' for fila in a]
    return r'\begin{bmatrix}' + '\n' + '\n'.join(filas) + '\n' + r'\end{bmatrix}'
'''.strip("\n")


def md(texto: str) -> tuple:
    return ("md", dedent(texto).strip("\n"))


def code(texto: str) -> tuple:
    return ("code", dedent(texto).strip("\n"))


def preguntas(lista: list[str]) -> tuple:
    """Celda que define pregunta_1..N y las muestra."""
    def _esc(p):
        return p.replace("\\", "\\\\").replace('"', '\\"')
    lineas = [f'pregunta_{i} = "{i}. {_esc(p)}  "' for i, p in enumerate(lista, 1)]
    lineas.append("")
    lineas.append("for item in sorted(dir()):")
    lineas.append("    if item.startswith('pregunta_'):")
    lineas.append("        display(Markdown(eval(item)))")
    return ("code", "\n".join(lineas))


def apartado(n: int) -> list[tuple]:
    return [md(f"## Apartado {n}"), code(f"display(Markdown(pregunta_{n}))")]


def _lit(v) -> str:
    if isinstance(v, str):
        return repr(v)
    if v is None:
        return "None"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isinf(v):
            return "np.inf"
        if float(v).is_integer():
            return str(int(v))
        return repr(round(float(v), 4))
    return repr(v)


def literal_dataframe(df: pd.DataFrame, nombre: str = "datos_enunciado",
                      indice: str = "activity") -> str:
    """Tabla de datos como literal Python alineado, al estilo de los Ejercicio_NN."""
    df = df.copy()
    if indice not in df.columns:
        df.index.name = indice
        df = df.reset_index()
    cols = list(df.columns)
    filas = []
    for _, fila in df.iterrows():
        filas.append([_lit(fila[c]) for c in cols])
    anchos = [max(len(f[j]) for f in filas + [[c for c in cols]]) for j in range(len(cols))]
    cab = "#  " + ", ".join(c.ljust(anchos[j]) for j, c in enumerate(cols))
    lineas = [f"{nombre} = pd.DataFrame([", cab]
    for f in filas:
        lineas.append("    (" + ", ".join(v.ljust(anchos[j]) for j, v in enumerate(f)) + "),")
    lineas.append("], columns=[" + ", ".join(repr(c) for c in cols) + f"]).set_index({indice!r})")
    lineas.append(nombre)
    return "\n".join(lineas)


def literal_dict(d: dict, nombre: str) -> str:
    lineas = [f"{nombre} = {{"]
    for k, v in d.items():
        lineas.append(f"    {k!r}: {v!r},")
    lineas.append("}")
    return "\n".join(lineas)


def _celda_md(texto: str) -> str:
    cuerpo = "\n".join(("# " + l) if l.strip() else "#" for l in texto.split("\n"))
    return "# %% [markdown]\n" + cuerpo


def _celda_code(texto: str) -> str:
    return "# %%\n" + texto


class Variante:
    """Una variante de una plantilla: nombre corto (slug), nivel 1-3, descripción y
    función generadora ``fn(rng) -> (celdas, subtitulo)``.

    ``grafo`` decide qué grafo dibujan las celdas emitidas con :func:`fig_grafo`:
    ``'pert'`` o ``'roy'`` (fijo, cuando la competencia lo exige), ``'ambos'`` (los dos)
    o ``'alterno'`` (PERT en la primera vuelta de la colección, Roy en la segunda, …;
    con ``--repeticiones 2`` cada variante sale una vez con cada grafo)."""

    def __init__(self, nombre: str, nivel: int, descripcion: str, fn, grafo: str = "alterno"):
        assert grafo in ("pert", "roy", "ambos", "alterno"), grafo
        self.nombre, self.nivel, self.descripcion, self.fn, self.grafo = nombre, nivel, descripcion, fn, grafo


ESTADO = {"grafo": "pert"}   # grafo activo mientras se genera un cuaderno (lo fija despachar)


def fig_grafo(durations: str = "datos_proyecto['duration']", cap: str = "Grafo PERT",
              label: str = "fig-pert", obj: str = "proyecto", tipo: str | None = None,
              ranksep: float = 1) -> list[tuple]:
    """Celda(s) de figura con el grafo activo (PERT, Roy o ambos). ``cap`` se escribe
    pensando en el PERT; para el Roy se adapta («PERT» → «Roy», sin «de los nodos»)."""
    tipo = tipo or ESTADO["grafo"]
    tipos = ["pert", "roy"] if tipo == "ambos" else [tipo]
    celdas = []
    for t in tipos:
        if t == "roy":
            c = cap.replace("PERT", "Roy").replace("tiempos de los nodos", "tiempos")
            llamada = f"{obj}.roy(durations={durations}, ranksep=1.2, nodesep=1.2)"
        else:
            c = cap
            llamada = f"{obj}.pert(durations={durations}, ranksep={ranksep})"
        lab = label if len(tipos) == 1 else f"{label}-{t}"
        celdas.append(code(f'#| fig-cap: "{c}"\n#| label: {lab}\n{llamada}'))
    return celdas


def elegir(variantes: list, k: int):
    return variantes[(k - 1) % len(variantes)]


def partir_enunciado(celdas: list[tuple]) -> tuple[list, list]:
    """Separa (enunciado, resolución) en la celda markdown '# Resolución'."""
    for i, (tipo, texto) in enumerate(celdas):
        if tipo == "md" and texto.strip().startswith("# Resolución"):
            return celdas[:i], celdas[i:]
    return celdas, []


def render(celdas: list[tuple], titulo: str, subtitulo: str, coleccion: str = "",
           variante: str = "", nivel: int = 0) -> str:
    """Devuelve el texto completo del cuaderno en formato py:percent."""
    partes = [CABECERA.format(titulo=titulo, subtitulo=subtitulo, coleccion=coleccion,
                              variante=variante, nivel=nivel).rstrip("\n")]
    partes.append(_celda_code(PREAMBULO))
    partes.append(_celda_code(DETALLES.strip("\n")))
    partes.append(_celda_code(TEMP))
    for tipo, texto in celdas:
        partes.append(_celda_md(texto) if tipo == "md" else _celda_code(texto))
    return "\n\n".join(partes) + "\n"


def despachar(variantes: list, k: int, rng):
    """Ejecuta la variante k (cíclica) y devuelve (celdas, subtitulo, variante).
    Fija el grafo activo: el de la variante o, si es 'alterno', PERT/Roy según la vuelta."""
    v = elegir(variantes, k)
    vuelta = (k - 1) // len(variantes)
    ESTADO["grafo"] = v.grafo if v.grafo != "alterno" else ("pert" if vuelta % 2 == 0 else "roy")
    celdas, sub = v.fn(rng)
    return celdas, sub, v
