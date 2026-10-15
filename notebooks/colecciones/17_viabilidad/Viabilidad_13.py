# ---
# title: Viabilidad 13
# subtitle: "límite de 3 operarios, ingresos de 65 k€/mes durante 36 meses"
# coleccion: 17_viabilidad
# variante: recursos-van
# nivel: 3
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

# %%
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


# %%
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

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Una empresa va a ejecutar el proyecto de inversión cuyo cuadro de prelaciones, duración (meses), operarios necesarios y presupuesto (BAC, k€) de cada actividad se adjunta. El coste de cada actividad se desembolsa uniformemente a lo largo de su duración. Cuando el proyecto termina, la instalación genera un flujo de caja neto de 65 k€ al mes durante 36 meses. La empresa descuenta al 12% anual y dispone como máximo de **3 operarios** al mes. Contratar un operario más cuesta **0.5 k€ por mes** mientras dure el proyecto.

# %%
pregunta_1 = "1. Calcule la duración y el VAN del proyecto sin tener en cuenta el límite de operarios.  "
pregunta_2 = "2. Aplique el algoritmo de asignación con el límite de operarios. ¿Cuánto se alarga el proyecto y cuánto cuesta en VAN? Separe el efecto sobre el VAN de los costes y el de los ingresos y explique cuál domina.  "
pregunta_3 = "3. ¿Compensa contratar un operario más (límite 4)? Construya la tabla de duración y VAN al ampliar el límite en 0, 1, 2 y 3 operarios y compare la mejora del VAN con el coste de contratar (sin descontar, por simplicidad).  "
pregunta_4 = "4. Calcule el flujo mensual mínimo (umbral del VAN = 0) con y sin el límite. ¿Cuánto sube el umbral por culpa del límite?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, operarios, BAC
    ('A'     , '---'      , 2       , 1        , 200),
    ('B'     , '---'      , 3       , 1        , 150),
    ('C'     , '---'      , 1       , 1        , 175),
    ('D'     , 'A,B,C'    , 1       , 2        , 175),
    ('E'     , 'D'        , 6       , 1        , 125),
    ('F'     , 'E'        , 6       , 2        , 125),
    ('G'     , 'D'        , 2       , 1        , 175),
    ('H'     , 'G'        , 3       , 3        , 175),
    ('I'     , 'F,H'      , 5       , 1        , 100),
], columns=['activity', 'predecessor', 'duration', 'operarios', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
F, N, k, L, coste_operario = 65.0, 36, 0.12, 3, 0.5
BAC = datos_proyecto['BAC']

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "VAN sin y con límite de operarios"
#| label: tbl-escenarios
res = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, F, N, k, ampliaciones=3)
escenarios = res['escenarios'].round(2)
escenarios

# %%
Markdown(f'Sin límite: {escenarios.loc["sin límite", "duración"]:.0f} meses y VAN {escenarios.loc["sin límite", "VAN"]:.1f} k€.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
dif = escenarios.loc['con límite'] - escenarios.loc['sin límite']
Markdown(f'Con {L} operarios el proyecto dura {escenarios.loc["con límite", "duración"]:.0f} meses (+{dif["duración"]:.0f}); '
         f'el VAN pasa de {escenarios.loc["sin límite", "VAN"]:.1f} a {escenarios.loc["con límite", "VAN"]:.1f} k€ ({dif["VAN"]:.1f}). '
         f'Efecto en los ingresos: {dif["VAN ingresos"]:.1f} k€ (se cobran más tarde); en los costes: {-dif["VAN costes"]:.1f} k€ '
         '(las actividades que esperan se pagan más tarde y el coste se descuenta más). El retraso de los ingresos domina.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Duración y VAN al ampliar el límite de operarios"
#| label: tbl-ampliaciones
amp = res['ampliaciones'].copy()
amp['coste de contratar'] = amp.index * coste_operario * amp['duración']
amp['VAN neto de contratar'] = amp['VAN'] - amp['coste de contratar']
amp.round(2)

# %%
mejor = amp['VAN neto de contratar'].idxmax()
Markdown(f'La mejor opción es ampliar el límite en {mejor} operario(s): VAN neto {amp.loc[mejor, "VAN neto de contratar"]:.1f} k€ '
         f'frente a {amp.loc[0, "VAN neto de contratar"]:.1f} k€ sin ampliar. Se compara la mejora del VAN por adelantar los ingresos '
         'con lo que cuesta mantener a los operarios todo el proyecto.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
e0 = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, 0.0, N, k, ampliaciones=0)['escenarios']['VAN']
e1 = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, 1.0, N, k, ampliaciones=0)['escenarios']['VAN']
umbral = -e0 / (e1 - e0)          # el VAN es lineal en F
Markdown(f'Flujo mensual mínimo: {umbral["sin límite"]:.2f} k€ sin límite y {umbral["con límite"]:.2f} k€ con límite '
         f'(+{umbral["con límite"] - umbral["sin límite"]:.2f} k€ al mes): el límite exige más ingresos para recuperar lo mismo.')
