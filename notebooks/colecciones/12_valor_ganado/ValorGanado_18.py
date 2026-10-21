# ---
# title: ValorGanado 18
# subtitle: "9 actividades; qué vigilar cruzando SPI/CPI por actividad con la holgura"
# coleccion: 12_valor_ganado
# variante: vigilar
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por actividad (BAC, k€, repartido uniformemente) se adjunta, junto con el seguimiento al final de la semana 10. El jefe de proyecto quiere una lista corta de actividades sobre las que actuar.

# %%
pregunta_1 = "1. Calcule PV, EV y AC de cada actividad y los índices CPI y SPI *por actividad*; añada la holgura total planificada de cada una.  "
pregunta_2 = "2. Clasifique las actividades en cuatro grupos: críticas con retraso (SPI < 1), no críticas con retraso, actividades con sobrecoste (CPI < 1) sin retraso, y el resto. ¿Cuáles amenazan el plazo? ¿Cuáles sólo el presupuesto?  "
pregunta_3 = "3. Calcule los índices globales del proyecto. ¿Qué actividades explican la mayor parte de SV y de CV (ordene por contribución)?  "
pregunta_4 = "4. Proponga las tres actividades a vigilar y justifique el criterio: ¿por qué un SPI bajo en una actividad con mucha holgura preocupa menos que en una crítica, y por qué con el CPI no hay esa distinción?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 3       , 50 ),
    ('B'     , '---'      , 1       , 10 ),
    ('C'     , '---'      , 1       , 10 ),
    ('D'     , 'A,B'      , 6       , 40 ),
    ('E'     , 'A,B'      , 8       , 40 ),
    ('F'     , 'C,E'      , 1       , 10 ),
    ('G'     , 'E'        , 6       , 50 ),
    ('H'     , 'D,G'      , 1       , 10 ),
    ('I'     , 'F,H'      , 1       , 40 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la producción de un cortometraje; las actividades son las siguientes (las duraciones del cuadro están en días).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Guion definitivo',
    'B': 'Casting',
    'C': 'Localizaciones y permisos',
    'D': 'Diseño de vestuario y atrezo',
    'E': 'Plan de rodaje',
    'F': 'Ensayos con los actores',
    'G': 'Rodaje de exteriores',
    'H': 'Rodaje en plató',
    'I': 'Grabación de sonido directo',
}, name='Descripción')
descripciones.to_frame()

# %%
#| tbl-cap: "Seguimiento en la semana de control"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 4              , 60, 100      ),
    ('B'     , 1              , 12, 100      ),
    ('C'     , 1              , 10, 100      ),
    ('D'     , 6              , 22, 50       ),
    ('E'     , 6              , 16, 38       ),
    ('F'     , 0              , 0 , 0        ),
    ('G'     , 0              , 0 , 0        ),
    ('H'     , 0              , 0 , 0        ),
    ('I'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
semana_control = 10

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy de la planificación con el camino crítico"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario planificado (inicios más tempranos y holguras)"
#| label: tbl-calendario
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'duracion']]

# %% [markdown]
# PV de cada actividad = BAC por la fracción planificada transcurrida en la semana de control; EV = BAC por el porcentaje realizado; AC = coste real anotado:

# %%
#| tbl-cap: "Cómo se obtiene el PV y el EV de cada actividad"
#| label: tbl-valorGanadoPaso
datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)
detalle = detalle_valor_ganado(datos_valor_ganado, calendario, semana_control)
detalle[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]

# %%
#| tbl-cap: "Valor ganado por actividad"
#| label: tbl-porActividad
datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)
detalle = detalle_valor_ganado(datos_valor_ganado, calendario, semana_control)
por_actividad = pd.DataFrame({'PV': detalle['PV_i'].astype(float), 'EV': detalle['EV_i'].astype(float), 'AC': seguimiento['AC'].astype(float),
                              'H_total': calendario['H_total']})
por_actividad['CPI'] = por_actividad['EV'] / por_actividad['AC'].replace(0, np.nan)
por_actividad['SPI'] = por_actividad['EV'] / por_actividad['PV'].replace(0, np.nan)
por_actividad.round(3)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
p = por_actividad
grupos = {
    'críticas con retraso (amenazan el plazo)': p.index[(p['H_total'] == 0) & (p['SPI'] < 1)],
    'no críticas con retraso': p.index[(p['H_total'] > 0) & (p['SPI'] < 1)],
    'sobrecoste sin retraso (amenazan el presupuesto)': p.index[(p['CPI'] < 1) & ~(p['SPI'] < 1)],
}
grupos['resto'] = p.index.difference(pd.Index([]).append([pd.Index(v) for v in grupos.values()]))
pd.Series({k: ', '.join(v) or '—' for k, v in grupos.items()}, name='actividades').to_frame()

# %%
retraso_no_critica = p.index[(p['H_total'] > 0) & (p['SPI'] < 1)]
retraso_semanas = ((1 - p.loc[retraso_no_critica, 'SPI']) * (datos_proyecto.loc[retraso_no_critica, 'duration'])).round(1)
Markdown(f'''Las no críticas retrasadas sólo amenazan el plazo si su retraso supera su holgura. Estimación grosera del retraso (semanas de trabajo que faltan respecto al plan): {', '.join(f'{a}: {r} frente a holgura {int(p.loc[a, "H_total"])}' for a, r in retraso_semanas.items()) or 'no hay'}.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
PV, EV, AC = p['PV'].sum(), p['EV'].sum(), p['AC'].sum()
contrib = pd.DataFrame({'SV_i = EV − PV': p['EV'] - p['PV'], 'CV_i = EV − AC': p['EV'] - p['AC']})
display(Markdown(f'Proyecto: PV = {PV:.1f}, EV = {EV:.1f}, AC = {AC:.1f}; SV = {EV - PV:.1f}, CV = {EV - AC:.1f}; SPI = {EV / PV:.3f}, CPI = {EV / AC:.3f}.'))
display(contrib.sort_values('SV_i = EV − PV').round(1).T)
Markdown(f'Las variaciones del proyecto son la suma de las de sus actividades: SV = {contrib.iloc[:, 0].sum():.1f}, CV = {contrib.iloc[:, 1].sum():.1f}. '
         f'La mayor contribución negativa a SV es de {contrib.iloc[:, 0].idxmin()} y a CV de {contrib.iloc[:, 1].idxmin()}.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
puntuacion = pd.DataFrame({'critica': (p['H_total'] == 0).astype(int), 'SV_i': contrib.iloc[:, 0], 'CV_i': contrib.iloc[:, 1]})
puntuacion['prioridad'] = (-puntuacion['SV_i'].clip(upper=0) * (2 + puntuacion['critica'] * 3) - puntuacion['CV_i'].clip(upper=0))
vigilar = puntuacion.sort_values('prioridad', ascending=False).index[:3].tolist()
display(puntuacion.sort_values('prioridad', ascending=False).round(1))
Markdown(f'''Actividades a vigilar: **{', '.join(vigilar)}** (criterio: retraso ponderado por criticidad más sobrecoste). Un SPI bajo en una actividad con holgura puede absorberse sin mover el plazo; en una crítica se traslada entero al final. El dinero, en cambio, no tiene holgura: cada euro de sobrecoste, crítico o no, se suma al coste final, por lo que el CPI se vigila en todas.''')
