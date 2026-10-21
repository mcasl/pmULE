# ---
# title: ValorGanado 21
# subtitle: "límite de 3 operarios; dos líneas base; control en la semana 11"
# coleccion: 12_valor_ganado
# variante: ev-recursos-incertidumbre
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
# Considere el proyecto cuyo cuadro de prelaciones, estimaciones de duración (mínima *a*, más probable *m* y máxima *b*, semanas, distribución triangular; *duration* es la duración esperada), operarios necesarios y presupuesto (BAC, k€) se adjunta; el coste se reparte uniformemente en cada actividad. La empresa dispone de **3 operarios** por semana y las actividades se ejecutan en cuanto el límite lo permite. Se barajan dos líneas base con las duraciones esperadas: la *ingenua* (calendario más temprano, sin recursos) y la *realista* (calendario respetando el límite de operarios).

# %%
pregunta_1 = "1. Calcule la duración y el PV de cada línea base y explique en qué se diferencian.  "
pregunta_2 = "2. Simule 3 000 ejecuciones (duraciones triangulares, replanificando con el límite de operarios) y calcule el SPI en la semana 11 con cada línea base. Describa sus distribuciones.  "
pregunta_3 = "3. Prevea la duración con T/SPI con cada línea base y compárela con la duración real de cada ejecución: sesgo, error absoluto medio y correlación. ¿Qué línea base permite prever mejor y por qué?  "
pregunta_4 = "4. Repita el análisis en tres fechas de control. ¿Cómo evoluciona el error de cada línea base con el avance del proyecto?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, a, m, b , operarios, BAC
    ('A'     , '---'      , 3       , 1, 3, 5 , 2        , 125),
    ('B'     , '---'      , 3       , 1, 3, 5 , 2        , 50 ),
    ('C'     , '---'      , 7       , 4, 7, 10, 2        , 150),
    ('D'     , 'B,C'      , 3       , 1, 3, 5 , 3        , 100),
    ('E'     , 'D'        , 2       , 1, 2, 3 , 3        , 175),
    ('F'     , 'E'        , 7       , 6, 7, 8 , 1        , 50 ),
    ('G'     , 'F'        , 2       , 1, 2, 3 , 2        , 200),
    ('H'     , 'A,G'      , 1       , 1, 1, 1 , 3        , 150),
    ('I'     , 'F'        , 2       , 1, 2, 3 , 1        , 100),
], columns=['activity', 'predecessor', 'duration', 'a', 'm', 'b', 'operarios', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
limite, t_control = [3], 11
BAC = datos_proyecto['BAC']
dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}
demanda = datos_proyecto[['operarios']]

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones esperadas (línea base ingenua)"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario de la línea base ingenua (sin recursos)"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]

# %% [markdown]
# La línea base realista aplica el algoritmo de asignación a las duraciones esperadas: traza semana a semana y Gantt (filas 💤: esperas).

# %%
res = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt de la línea base realista (con el límite de operarios)"
#| label: fig-ganttRealista
pinta_tikz(res.dibujos['operarios'])

# %%
#| tbl-cap: "Valor planificado de cada actividad en la semana de control con cada línea base"
#| label: tbl-pvLineas
frac_ingenua = ((t_control - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion'])
frac_realista = ((t_control - res.inicios.reindex(datos_proyecto.index)).clip(lower=0).clip(upper=datos_proyecto['duration']) / datos_proyecto['duration'])
pv_lineas = pd.DataFrame({'inicio ingenuo': calendario['inicio_mas_temprano'].astype(int), 'inicio realista': res.inicios.reindex(datos_proyecto.index).astype(int),
                          'PV ingenuo': BAC * frac_ingenua, 'PV realista': BAC * frac_realista})
pv_lineas.loc['Total'] = ['', '', pv_lineas['PV ingenuo'].sum(), pv_lineas['PV realista'].sum()]
pv_lineas

# %%
r = proyecto.monte_carlo_valor_ganado_recursos(dists, datos_proyecto['duration'], demanda, ['operarios'], limite, BAC, t_control, n_samples=3000, seed=11)
Markdown(f'Línea base ingenua: {r.T_libre:.0f} semanas, PV({t_control}) = {r.pv_libre:.1f} k€. '
         f'Línea base realista: {r.T_recursos:.0f} semanas, PV({t_control}) = {r.pv_recursos:.1f} k€. '
         'La realista programa las esperas por falta de operarios; la ingenua las ignora y parte de un plan imposible.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "SPI, duración real y previsión T/SPI con cada línea base"
#| label: tbl-spiRecursos
r.summary().round(3)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Calidad de la previsión de duración"
#| label: tbl-calidadRecursos
calidad = r.calidad_prevision().round(3)
calidad

# %%
mejor = calidad['error absoluto medio'].idxmin()
Markdown(f'Mejor previsión: {mejor} (error absoluto medio {calidad.loc[mejor, "error absoluto medio"]:.2f} semanas). '
         'Con la línea base ingenua el SPI cae por debajo de 1 aunque el equipo trabaje como se esperaba, y T/SPI sobrestima la duración; '
         'la realista mide solo las desviaciones que no estaban ya previstas.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
filas = {}
for t in sorted({max(2, round(t_control * 0.5)), t_control, round(t_control * 1.5)}):
    rr = proyecto.monte_carlo_valor_ganado_recursos(dists, datos_proyecto['duration'], demanda, ['operarios'], limite, BAC, t, n_samples=3000, seed=11)
    q = rr.calidad_prevision()['error absoluto medio']
    filas[t] = {'sin recursos': q['línea base sin recursos'], 'con recursos': q['línea base con recursos']}
evolucion = pd.DataFrame(filas).T.rename_axis('semana de control')
display(evolucion.round(3))
peor = (evolucion['sin recursos'] > evolucion['con recursos']).all()
Markdown(('La línea base ingenua da peor previsión en todas las fechas de control. ' if peor else 'La línea base ingenua da peor previsión en las primeras fechas. ') +
         'Su error es sobre todo sistemático (sesgo), porque compara con un plan imposible; el de la realista es ruido de las duraciones y no desaparece del todo hasta el final.')
