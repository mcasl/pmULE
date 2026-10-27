# ---
# title: MonteCarlo 16
# subtitle: "9 actividades; plazo del cliente 33; percentiles y retraso esperado"
# coleccion: 09_montecarlo
# variante: percentiles
# nivel: 2
# grafo: roy
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
pd.set_option('display.max_colwidth', None)   # las columnas «… = suma» no se truncan con «...»

# Versión local de la librería: se busca src/pmule.py subiendo desde el directorio actual
_aqui = os.path.abspath('.')
while _aqui != os.path.dirname(_aqui) and not os.path.isfile(os.path.join(_aqui, 'src', 'pmule.py')):
    _aqui = os.path.dirname(_aqui)
sys.path.append(os.path.join(_aqui, 'src'))
from pmule import (PredecessorTable, ProjectGraph, EarnedValue,
                   pinta_tikz, plot_cargas, plot_curvas_valor_ganado,
                   tanto_por_uno, ordena_rutas,
                   explica_camino_critico, explica_rutas, explica_suma, explica_sigma_ruta,
                   explica_probabilidad, explica_intervalo, explica_percentil,
                   plural, lista_y, ordena_ruta, limpia_tabla, gantt_tramos_tikz)

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
                      'opciones (coste)': '; '.join(f'{k} ({round(v, 2):g})' for k, v in orden[:6]) + (' …' if len(orden) > 6 else ''),
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
# Considere el proyecto cuyo cuadro de prelaciones y estimaciones optimista, más probable y pesimista (en semanas) se adjunta; las duraciones siguen distribuciones triangulares independientes. El cliente ha fijado la entrega en la semana 33 con una penalización por cada semana de retraso.

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
    ('A'     , '---'      , 2, 4, 9 ),
    ('B'     , '---'      , 3, 6, 9 ),
    ('C'     , 'A,B'      , 4, 5, 6 ),
    ('D'     , 'C'        , 2, 4, 9 ),
    ('E'     , 'A'        , 6, 9, 9 ),
    ('F'     , 'C,E'      , 4, 6, 8 ),
    ('G'     , 'D,F'      , 3, 4, 5 ),
    ('H'     , 'G'        , 6, 7, 11),
    ('I'     , 'F'        , 4, 6, 8 ),
], columns=['activity', 'predecessor', 'a', 'm', 'b']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = datos_proyecto.eval('(a + m + b) / 3')
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
N, plazo = 20000, 33

# %%
def tabla_rutas(R, medias, varianzas=None, calendario=None, dec=0, dec_var=3, nombre='media'):
    """Media (y varianza) de cada ruta con la suma término a término; actividades en orden de ejecución."""
    filas = {}
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario)
        m = float(sum(float(medias[a]) for a in acts))
        fila = {'actividades': ' – '.join(acts),
                f'{nombre} = suma': ' + '.join(f'{a}({float(medias[a]):.{dec}f})' for a in acts) + f' = {m:.{dec}f}',
                nombre: m}
        if varianzas is not None:
            v = float(sum(float(varianzas[a]) for a in acts))
            fila['varianza = suma'] = ' + '.join(f'{a}({float(varianzas[a]):.{dec_var}f})' for a in acts) + f' = {v:.{dec_var}f}'
            fila['varianza'] = v
            fila['σ = √varianza'] = v ** 0.5
        filas[r] = fila
    t = pd.DataFrame(filas).T
    for c in (nombre, 'varianza', 'σ = √varianza'):
        if c in t:
            t[c] = t[c].astype(float)
    return t

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Media de cada triangular"
#| label: tbl-medias
pd.DataFrame({'(a + m + b)/3': [f"({r['a']:g} + {r['m']:g} + {r['b']:g})/3 = {r['duration']:.2f}" for _, r in datos_proyecto.iterrows()]}, index=datos_proyecto.index).T

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con las duraciones medias"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'].round(1))

# %%
#| tbl-cap: "Duración media de cada ruta"
#| label: tbl-rutas
path_matrix = proyecto.path_matrix(dummies=False).drop_duplicates()   # rutas que sólo difieren en ficticias, una vez
calendario = proyecto.calendar(datos_proyecto['duration'])
duraciones_rutas_medias = path_matrix @ datos_proyecto['duration'][path_matrix.columns]
duracion_media = float(duraciones_rutas_medias.max())
rutas = tabla_rutas(path_matrix, datos_proyecto['duration'], calendario=calendario, dec=2, nombre='duración media')
limpia_tabla(rutas)

# %%
criticas_txt = [rutas.loc[r, 'actividades'] for r in rutas.index if np.isclose(rutas.loc[r, 'duración media'], duracion_media)]
Markdown(f'**Respuesta:** con las medias de las triangulares la duración del proyecto es {duracion_media:.2f} semanas y el camino crítico '
         f'{lista_y(criticas_txt)}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# En cada réplica se sortea la duración de cada actividad con su triangular, se suman las duraciones a lo largo de cada ruta y la duración del proyecto es la de la ruta más larga. El percentil $p$ de las réplicas es el valor que deja por debajo la proporción $p$ de ellas: se ordenan de menor a mayor y se toma la que ocupa la posición $p\cdot N$ (interpolando).

# %%
rng = np.random.default_rng(7911)
duraciones_mc = pd.DataFrame({act: rng.triangular(fila['a'], fila['m'], fila['b'], N) for act, fila in datos_proyecto.iterrows()})
duraciones_rutas_mc = duraciones_mc[path_matrix.columns] @ path_matrix.T
T_mc = duraciones_rutas_mc.max(axis='columns')
percentiles = T_mc.quantile([0.10, 0.50, 0.80, 0.90, 0.95])
percentiles.index = [f'P{int(round(p * 100))}' for p in percentiles.index]
pd.DataFrame({'posición en la muestra ordenada': [f'≈ {int(round(p * N))} de {N}' for p in (0.10, 0.50, 0.80, 0.90, 0.95)],
              'duración (semanas)': percentiles.round(2).values}, index=percentiles.index).T

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

# %%
pc = percentiles
Markdown(f'''**Respuesta:** P10 = {pc['P10']:.2f}, P50 = {pc['P50']:.2f}, P80 = {pc['P80']:.2f}, P90 = {pc['P90']:.2f} y P95 = {pc['P95']:.2f} semanas. '''
         f'''Para el jefe de proyecto, P10 es una fecha optimista (sólo 1 de cada 10 réplicas termina antes); P50, la mediana (tan probable '''
         f'''acabar antes como después); P80, una fecha que se cumple cuatro de cada cinco veces; P90 y P95, fechas «seguras» que sólo se '''
         f'''superan el 10 % y el 5 % de las veces. La diferencia P95 − P50 = {pc['P95'] - pc['P50']:.2f} semanas es el margen que hay que '''
         '''añadir a una previsión central para hacerla fiable.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Se paga penalización en las réplicas que superan el plazo del cliente; el retraso de cada réplica es $\max(0, T - \text{plazo})$ y se estudia como cualquier otra variable simulada.

# %%
superan = int((T_mc > plazo).sum())
p_penalizacion = float((T_mc > plazo).mean())
retraso = (T_mc - plazo).clip(lower=0)
retraso_95 = float(retraso.quantile(0.95))
retraso_medio = float(retraso.mean())
Markdown(f'''P(T > {plazo}) ≈ {superan}/{N} = {p_penalizacion:.4f}: se paga penalización en el {p_penalizacion * 100:.1f} % de las réplicas. '''
         f'''El percentil 95 del retraso es {retraso_95:.2f} semanas{" (= P95 − plazo = " + f"{pc['P95']:.2f} − {plazo}" + ")" if retraso_95 > 0 else " (el plazo ya es mayor que el P95)"}. '''
         f'''El retraso medio es la media de las {N} réplicas de máx(0, T − {plazo}) (las que terminan a tiempo cuentan 0): {retraso.sum():.1f}/{N} = {retraso_medio:.3f} semanas.\n\n'''
         f'''**Respuesta:** probabilidad de pagar penalización {p_penalizacion:.4f}; el retraso supera {retraso_95:.2f} semanas sólo con '''
         f'''probabilidad 0,05; retraso medio esperado {retraso_medio:.3f} semanas (multiplicado por la penalización semanal da el coste esperado de la cláusula).''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
fecha_10 = float(T_mc.quantile(0.90))
fecha_entera = int(np.ceil(round(fecha_10, 2)))
p_entera = float((T_mc > fecha_entera).mean())
p_media = float((T_mc > duracion_media).mean())
Markdown(f'''Para que P(T > fecha) = 0,10 hay que comprometer el percentil 90 de las réplicas: {fecha_10:.2f} semanas, es decir, semana '''
         f'''{fecha_entera} en semanas enteras (P(T > {fecha_entera}) ≈ {p_entera:.3f}). Comprometer la media ({duracion_media:.2f}) supondría '''
         f'''penalización en el {p_media:.1%} de los casos'''
         + (', algo más de la mitad por el sesgo de fusión de las rutas.' if p_media > 0.5 else '.') + '''\n\n'''
         f'''**Respuesta:** habría que haber comprometido {fecha_10:.2f} semanas (semana {fecha_entera}), con un margen de '''
         f'''{fecha_10:.2f} − {duracion_media:.2f} = {fecha_10 - duracion_media:.2f} semanas sobre la duración media.''')
