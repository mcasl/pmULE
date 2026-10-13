# ---
# title: MonteCarlo 08
# subtitle: "8 actividades; plazo del cliente 34; percentiles y retraso esperado"
# coleccion: 09_montecarlo
# variante: percentiles
# nivel: 2
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
# Considere el proyecto cuyo cuadro de prelaciones y estimaciones optimista, más probable y pesimista (en semanas) se adjunta; las duraciones siguen distribuciones triangulares independientes. El cliente ha fijado la entrega en la semana 34 con una penalización por cada semana de retraso.

# %%
pregunta_1 = "1. Calcule la duración del proyecto con las duraciones medias de las triangulares, $(a + m + b)/3$, y el camino crítico.  "
pregunta_2 = "2. Simule 20000 réplicas con la matriz de caminos y obtenga los percentiles 10, 50, 80, 90 y 95 de la duración. ¿Qué significa cada uno para el jefe de proyecto?  "
pregunta_3 = "3. ¿Qué probabilidad hay de pagar penalización? ¿Cuántas semanas de retraso se superan sólo con probabilidad 0,05? Calcule el retraso medio esperado (media de máx(0, T − plazo)).  "
pregunta_4 = "4. ¿Qué fecha habría que haber comprometido para que la probabilidad de penalización fuera del 10 %? Compare con la duración media: ¿cuántas semanas de *margen* supone?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b 
    ('A'     , '---'      , 5, 6, 10),
    ('B'     , '---'      , 6, 9, 9 ),
    ('C'     , 'A,B'      , 3, 6, 9 ),
    ('D'     , 'B'        , 2, 2, 5 ),
    ('E'     , 'C'        , 3, 4, 8 ),
    ('F'     , 'C'        , 2, 5, 5 ),
    ('G'     , 'E,F'      , 1, 4, 7 ),
    ('H'     , 'D,G'      , 6, 8, 10),
], columns=['activity', 'predecessor', 'a', 'm', 'b']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = datos_proyecto.eval('(a + m + b) / 3')
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
N, plazo = 20000, 34

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Media de cada triangular"
#| label: tbl-medias
pd.DataFrame({'(a + m + b)/3': [f"({r['a']:g} + {r['m']:g} + {r['b']:g})/3 = {r['duration']:.2f}" for _, r in datos_proyecto.iterrows()]}, index=datos_proyecto.index).T

# %%
#| fig-cap: "Grafo PERT con las duraciones medias"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'].round(1), ranksep=1)

# %%
path_matrix = proyecto.path_matrix(dummies=False)
duraciones_rutas_medias = path_matrix @ datos_proyecto['duration'][path_matrix.columns]
duracion_media = float(duraciones_rutas_medias.max())
Markdown(f'Con las medias, la ruta más larga es {duraciones_rutas_medias.idxmax()} ({" - ".join(path_matrix.columns[path_matrix.loc[duraciones_rutas_medias.idxmax()] == 1])}) y la duración {duracion_media:.2f} semanas.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
rng = np.random.default_rng(9265)
duraciones_mc = pd.DataFrame({act: rng.triangular(fila['a'], fila['m'], fila['b'], N) for act, fila in datos_proyecto.iterrows()})
duraciones_rutas_mc = duraciones_mc[path_matrix.columns] @ path_matrix.T
T_mc = duraciones_rutas_mc.max(axis='columns')
percentiles = T_mc.quantile([0.10, 0.50, 0.80, 0.90, 0.95])
percentiles.index = [f'P{int(p * 100)}' for p in percentiles.index]
percentiles.round(2).to_frame('duración (semanas)').T

# %%
#| fig-cap: "Función de distribución simulada de la duración"
#| label: fig-distribucion
from matplotlib import pyplot as plt
x = np.sort(T_mc.values); y = np.arange(1, N + 1) / N
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y)
for nombre, v in percentiles.items():
    ax.axvline(v, color='grey', linestyle=':', alpha=0.7); ax.text(v, 0.02, nombre, rotation=90, va='bottom')
ax.axvline(plazo, color='red', linestyle='--', label=f'plazo del cliente ({plazo})')
ax.set_xlabel('Duración (semanas)'); ax.set_ylabel('P(T ≤ t)'); ax.legend(); plt.show()

# %% [markdown]
# El percentil $p$ es la duración que no se supera con probabilidad $p$: P50 es la mediana (tan probable acabar antes como después), P80 la fecha que se cumple cuatro de cada cinco veces, P95 una fecha «segura». La diferencia P95 − P50 mide cuánto margen hay que añadir a una previsión central para hacerla fiable.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
p_penalizacion = float((T_mc > plazo).mean())
retraso = (T_mc - plazo).clip(lower=0)
retraso_95 = float(retraso.quantile(0.95))
retraso_medio = float(retraso.mean())
Markdown(f'''P(T > {plazo}) = {p_penalizacion:.3f}: se paga penalización en el {p_penalizacion * 100:.1f} % de las réplicas. El retraso supera {retraso_95:.2f} semanas sólo con probabilidad 0,05 (percentil 95 de máx(0, T − {plazo})). El retraso medio esperado es {retraso_medio:.3f} semanas: multiplicado por la penalización semanal da el coste esperado de la cláusula.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
fecha_10 = float(T_mc.quantile(0.90))
Markdown(f'''Para que P(T > fecha) = 0,10 hay que comprometer el percentil 90: {fecha_10:.2f} semanas, es decir, {fecha_10 - duracion_media:.2f} semanas de margen sobre la duración media ({duracion_media:.2f}). Comprometer la media supone penalización aproximadamente la mitad de las veces (P(T > media) = {float((T_mc > duracion_media).mean()):.3f}, algo más de 0,5 por el sesgo de fusión de las rutas).''')
