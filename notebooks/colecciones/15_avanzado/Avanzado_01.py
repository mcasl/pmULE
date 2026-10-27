# ---
# title: Avanzado 01
# subtitle: "8 actividades; 4 relaciones PDM no ordinarias (1 CC, 1 FF)"
# coleccion: 15_avanzado
# variante: pdm-desfases
# nivel: 3
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
# Considere el proyecto cuyas actividades y duraciones (días) se adjuntan. Sus prelaciones no son todas del tipo habitual: el segundo cuadro las describe con el *método de diagramación por precedencias* (PDM), donde cada relación tiene un tipo y un desfase (en días):
#
# - **FC** (fin-comienzo) + d: la sucesora empieza al menos d días después de que termine la predecesora (FC + 0 es la prelación ordinaria);
# - **CC** (comienzo-comienzo) + d: la sucesora empieza al menos d días después de que *empiece* la predecesora (solape);
# - **FF** (fin-fin) + d: la sucesora termina al menos d días después de que termine la predecesora;
# - **CF** (comienzo-fin) + d: la sucesora termina al menos d días después de que empiece la predecesora.
#
# Las actividades no se interrumpen.

# %%
pregunta_1 = "1. Escriba, para cada tipo de relación, la condición que impone sobre el inicio más temprano de la sucesora (ES) en función de ES, EF y la duración. Calcule el paso hacia delante (ES, EF) en orden topológico, anotando los candidatos de cada actividad.  "
pregunta_2 = "2. Escriba las condiciones del paso hacia atrás (LS, LF) para cada tipo y calcúlelo. Obtenga las holguras totales y la duración del proyecto.  "
pregunta_3 = "3. ¿Cuál es el camino crítico? ¿Hay alguna actividad crítica que no esté *entera* en él, es decir, cuya criticidad venga de un solape (CC o FF)? ¿Qué pasaría con el plazo si esa actividad se alargara?  "
pregunta_4 = "4. Sustituya todas las relaciones por FC + 0 (las prelaciones clásicas) y calcule la duración con la librería. ¿Por qué difiere? ¿Qué relaciones acortan y cuáles alargan el proyecto?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Actividades y duraciones"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 7       ),
    ('B'     , '---'      , 3       ),
    ('C'     , 'A,B'      , 3       ),
    ('D'     , 'C'        , 6       ),
    ('E'     , 'B'        , 5       ),
    ('F'     , 'C'        , 3       ),
    ('G'     , 'D,E,F'    , 8       ),
    ('H'     , 'G'        , 8       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %%
#| tbl-cap: "Relaciones PDM"
#| label: tbl-relaciones
relaciones = pd.DataFrame([
#  id, predecessor, successor, type, lag
    (1 , 'A'        , 'C'      , 'FC', 0  ),
    (2 , 'B'        , 'C'      , 'FC', 1  ),
    (3 , 'C'        , 'D'      , 'FC', 0  ),
    (4 , 'B'        , 'E'      , 'CC', 1  ),
    (5 , 'C'        , 'F'      , 'FF', 1  ),
    (6 , 'D'        , 'G'      , 'FC', 0  ),
    (7 , 'E'        , 'G'      , 'FC', 0  ),
    (8 , 'F'        , 'G'      , 'FC', 0  ),
    (9 , 'G'        , 'H'      , 'CF', 10 ),
], columns=['id', 'predecessor', 'successor', 'type', 'lag']).set_index('id')
relaciones

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Condiciones sobre la sucesora $a$ con predecesora $p$ y desfase $\ell$:
#
# | tipo | condición | sobre ES(a) |
# |---|---|---|
# | FC | $ES_a \ge EF_p + \ell$ | $EF_p + \ell$ |
# | CC | $ES_a \ge ES_p + \ell$ | $ES_p + \ell$ |
# | FF | $EF_a \ge EF_p + \ell$ | $EF_p + \ell - d_a$ |
# | CF | $EF_a \ge ES_p + \ell$ | $ES_p + \ell - d_a$ |
#
# ES es el máximo de los candidatos (y 0 si no hay ninguno); EF = ES + d.

# %%
def calendario_pdm(datos, relaciones):
    '''Pasos hacia delante y hacia atrás con relaciones FC/CC/FF/CF y desfases (sin interrumpir actividades).'''
    d = datos['duration'].astype(int).to_dict()
    G = nx.DiGraph(); G.add_nodes_from(d)
    for _, r in relaciones.iterrows():
        G.add_edge(r['predecessor'], r['successor'], tipo=r['type'], lag=int(r['lag']))
    orden = list(nx.topological_sort(G))
    ES, EF, como = {}, {}, {}
    for a in orden:
        cands = {}
        for p, _, e in G.in_edges(a, data=True):
            tipo, lag = e['tipo'], e['lag']
            if tipo == 'FC': v, f = EF[p] + lag, f'EF({p}) + {lag} = {EF[p]} + {lag}'
            elif tipo == 'CC': v, f = ES[p] + lag, f'ES({p}) + {lag} = {ES[p]} + {lag}'
            elif tipo == 'FF': v, f = EF[p] + lag - d[a], f'EF({p}) + {lag} − d({a}) = {EF[p]} + {lag} − {d[a]}'
            else: v, f = ES[p] + lag - d[a], f'ES({p}) + {lag} − d({a}) = {ES[p]} + {lag} − {d[a]}'
            cands[f'{tipo}+{lag} desde {p}: {f} = {v}'] = v
        ES[a] = max([0] + list(cands.values())); EF[a] = ES[a] + d[a]
        if not cands:
            como[a] = 'sin predecesoras: 0'
        else:
            como[a] = '; '.join(cands) + (' (negativo: no puede empezar antes del inicio, 0)' if max(cands.values()) < 0 else '')
    T = max(EF.values())
    LS, LF, como_t = {}, {}, {}
    for a in reversed(orden):
        cands = {}
        for _, s, e in G.out_edges(a, data=True):
            tipo, lag = e['tipo'], e['lag']
            if tipo == 'FC': v, f = LS[s] - lag, f'LS({s}) − {lag} = {LS[s]} − {lag}'
            elif tipo == 'CC': v, f = LS[s] - lag + d[a], f'LS({s}) − {lag} + d({a}) = {LS[s]} − {lag} + {d[a]}'
            elif tipo == 'FF': v, f = LF[s] - lag, f'LF({s}) − {lag} = {LF[s]} − {lag}'
            else: v, f = LF[s] - lag + d[a], f'LF({s}) − {lag} + d({a}) = {LF[s]} − {lag} + {d[a]}'
            cands[f'{tipo}+{lag} hacia {s}: {f} = {v}'] = v
        LF[a] = min([T] + list(cands.values())); LS[a] = LF[a] - d[a]
        if not cands:
            como_t[a] = f'sin sucesoras: T = {T}'
        else:
            como_t[a] = '; '.join(cands) + (f' (todos superan T = {T}: no puede acabar después del fin del proyecto)' if min(cands.values()) > T else '')
    cal = pd.DataFrame({'ES': ES, 'EF': EF, 'LS': LS, 'LF': LF}).loc[list(d)]
    cal['H_total'] = cal['LS'] - cal['ES']
    detalle = pd.DataFrame({'ES = máx(…)': como, 'LF = mín(…)': como_t}).loc[list(d)]
    return cal, detalle, T

# %%
calendario, detalle, duracion_pdm = calendario_pdm(datos_proyecto, relaciones)
_d = datos_proyecto['duration'].astype(int)

# %% [markdown]
# Paso hacia delante en orden topológico: para cada actividad, un candidato por cada relación que llega a ella (tipo + desfase, con los números sustituidos); ES es el mayor y EF = ES + d.

# %%
#| tbl-cap: "Paso hacia delante: candidatos de ES de cada actividad"
#| label: tbl-candidatos
pd.DataFrame({'candidatos de ES': detalle['ES = máx(…)'], 'ES = máx': calendario['ES'],
              'EF = ES + d': [f"{calendario.loc[a, 'ES']} + {_d[a]} = {calendario.loc[a, 'EF']}" for a in calendario.index]})

# %%
_ult = calendario.index[calendario['EF'] == duracion_pdm].tolist()
Markdown(f'**Respuesta:** ES y EF son los de la tabla. La duración del proyecto es el mayor EF: T = máx(EF) = {duracion_pdm} días (EF de {lista_y(_ult)}).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Hacia atrás, cada relación acota la predecesora: FC: $LF_p \le LS_a - \ell$; CC: $LS_p \le LS_a - \ell$ (luego $LF_p \le LS_a - \ell + d_p$); FF: $LF_p \le LF_a - \ell$; CF: $LS_p \le LF_a - \ell$ (luego $LF_p \le LF_a - \ell + d_p$). Además ninguna actividad puede acabar después del fin del proyecto (LF ≤ T). LF es el menor de los candidatos, LS = LF − d y la holgura total es LS − ES.

# %%
#| tbl-cap: "Paso hacia atrás: candidatos de LF y holgura total"
#| label: tbl-calendarioPdm
pd.DataFrame({'candidatos de LF': detalle['LF = mín(…)'], 'LF = mín': calendario['LF'],
              'LS = LF − d': [f"{calendario.loc[a, 'LF']} − {_d[a]} = {calendario.loc[a, 'LS']}" for a in calendario.index],
              'H_total = LS − ES': [f"{calendario.loc[a, 'LS']} − {calendario.loc[a, 'ES']} = {calendario.loc[a, 'H_total']}" for a in calendario.index]})

# %%
criticas = calendario.index[calendario['H_total'] == 0].tolist()
Markdown(f'**Respuesta:** la duración del proyecto es {duracion_pdm} días; actividades críticas (holgura total 0): {lista_y(criticas)}; '
         + 'con holgura: ' + lista_y([f'{a} ({calendario.loc[a, "H_total"]})' for a in calendario.index if a not in criticas]) + '.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Una relación es *activa* si se cumple con igualdad en el calendario temprano (es la que fija la fecha de la sucesora). El camino crítico es la cadena de relaciones activas entre actividades críticas, desde el inicio hasta el fin del proyecto.

# %%
rel_crit = relaciones[relaciones['predecessor'].isin(criticas) & relaciones['successor'].isin(criticas)].copy()
def activa(r):
    p, s, t, l = r['predecessor'], r['successor'], r['type'], int(r['lag'])
    ES, EF = calendario['ES'], calendario['EF']
    return {'FC': ES[s] == EF[p] + l, 'CC': ES[s] == ES[p] + l, 'FF': EF[s] == EF[p] + l, 'CF': EF[s] == ES[p] + l}[t]
rel_crit['activa'] = rel_crit.apply(activa, axis=1).map({True: 'sí', False: 'no'})
display(rel_crit.rename(columns={'predecessor': 'predecesora', 'successor': 'sucesora', 'type': 'tipo', 'lag': 'desfase'}))
Ga = nx.DiGraph(); Ga.add_nodes_from(criticas)
Ga.add_edges_from((r.predecessor, r.successor, {'et': f'{r.type}+{r.lag}'}) for r in rel_crit.itertuples() if r.activa == 'sí')
_ini = [a for a in criticas if calendario.loc[a, 'ES'] == 0 and Ga.in_degree(a) == 0]
_fin = [a for a in criticas if Ga.out_degree(a) == 0]
cadenas = []
for i in _ini:
    for f in _fin:
        for camino in nx.all_simple_paths(Ga, i, f) if i != f else [[i]]:
            cadenas.append(camino[0] + ''.join(f' →({Ga.edges[u, v]["et"]}) {v}' for u, v in zip(camino, camino[1:])))
display(Markdown('**Camino crítico:** ' + '; '.join(cadenas) + '.'))

# %% [markdown]
# Para saber qué pasa con el plazo si una actividad crítica se alarga (o se acorta), se repite el cálculo con su duración cambiada en un día. Con relaciones FC el plazo se mueve lo mismo que la actividad; con CC, FF o CF puede no moverse, o incluso moverse al revés.

# %%
def _T_con(a, delta):
    dd = datos_proyecto.copy(); dd.loc[a, 'duration'] = int(dd.loc[a, 'duration']) + delta
    return calendario_pdm(dd, relaciones)[2] if dd.loc[a, 'duration'] > 0 else None
sens = pd.DataFrame({'duración': [int(_d[a]) for a in criticas],
                     'T si se alarga 1 día': [_T_con(a, 1) for a in criticas],
                     'T si se acorta 1 día': [_T_con(a, -1) for a in criticas]}, index=criticas)
display(sens)
_no_retrasa = [a for a in criticas if sens.loc[a, 'T si se alarga 1 día'] <= duracion_pdm]
_al_reves = [a for a in criticas if sens.loc[a, 'T si se alarga 1 día'] < duracion_pdm]
_no_adelanta = [a for a in criticas if sens.loc[a, 'T si se acorta 1 día'] is not None and sens.loc[a, 'T si se acorta 1 día'] >= duracion_pdm]
_solapes = rel_crit[(rel_crit['activa'] == 'sí') & rel_crit['type'].isin(['CC', 'FF', 'CF'])]
_txt = ('Relaciones activas del camino crítico que no son fin-comienzo: ' + lista_y([f'{r.predecessor}→{r.successor} ({r.type}+{r.lag})' for r in _solapes.itertuples()]) + '. '
        if len(_solapes) else 'Todas las relaciones activas del camino crítico son FC. ')
if _no_retrasa:
    _txt += (f'**Respuesta:** {lista_y(_no_retrasa)} {"es crítica" if len(_no_retrasa) == 1 else "son críticas"} sin estar entera{"s" if len(_no_retrasa) > 1 else ""} en el camino crítico: '
             f'si se alarga{"n" if len(_no_retrasa) > 1 else ""} un día el plazo no se retrasa (sigue en {duracion_pdm} días o menos)'
             + (f'; alargar {lista_y(_al_reves)} incluso lo acorta, porque su relación CF/FF fija su *fin* y, al ser más larga, puede empezar antes' if _al_reves else '')
             + '. Su criticidad viene sólo de su inicio o sólo de su fin (la paradoja del PDM).')
else:
    _txt += f'**Respuesta:** todas las actividades críticas lo son enteras: alargar cualquiera de ellas un día retrasa el plazo a {duracion_pdm + 1} días.'
if _no_adelanta:
    _txt += f' Acortar {lista_y(_no_adelanta)} un día no adelanta el fin del proyecto.'
Markdown(_txt)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
tabla_fc = datos_proyecto.copy()
tabla_fc['predecessor'] = [','.join(sorted(relaciones.loc[relaciones['successor'] == a, 'predecessor'])) or '---' for a in tabla_fc.index]
proyecto_fc = PredecessorTable.from_dataframe_of_strings(tabla_fc, activity='activity', predecessor='predecessor').create_project()
duracion_fc = proyecto_fc.duration(tabla_fc['duration'])
calendario_fc = proyecto_fc.calendar(tabla_fc['duration'])
comparacion = pd.DataFrame({'ES (PDM)': calendario['ES'], 'ES (todo FC+0)': calendario_fc['inicio_mas_temprano'], 'H_total (PDM)': calendario['H_total'], 'H_total (todo FC+0)': calendario_fc['H_total']})
display(comparacion)
display(explica_camino_critico(proyecto_fc, tabla_fc['duration'], 'días'))

# %% [markdown]
# Para ver qué relaciones acortan y cuáles alargan el proyecto, se cambia cada relación que no es FC + 0, una a una, por FC + 0 (las demás se dejan como están) y se recalcula la duración PDM: si con FC + 0 el proyecto dura más, esa relación lo acortaba; si dura menos, lo alargaba.

# %%
efectos = []
for i, r in relaciones.iterrows():
    if r['type'] == 'FC' and int(r['lag']) == 0:
        continue
    rel2 = relaciones.copy(); rel2.loc[i, 'type'] = 'FC'; rel2.loc[i, 'lag'] = 0
    T2 = calendario_pdm(datos_proyecto, rel2)[2]
    efectos.append({'relación': f"{r['predecessor']}→{r['successor']} ({r['type']}+{r['lag']})", 'T con esa relación como FC+0': T2,
                    'efecto de la relación sobre T': duracion_pdm - T2})
efectos = pd.DataFrame(efectos).set_index('relación')
display(efectos)
_acorta = [f'{k}: {plural(-v, "día")}' for k, v in efectos['efecto de la relación sobre T'].items() if v < 0]
_alarga = [f'{k}: {plural(v, "día")}' for k, v in efectos['efecto de la relación sobre T'].items() if v > 0]
_nada = [k for k, v in efectos['efecto de la relación sobre T'].items() if v == 0]
Markdown(f'**Respuesta:** con todas las relaciones FC + 0 el proyecto dura {duracion_fc} días frente a {duracion_pdm} con PDM '
         + (f'(PDM acorta {plural(duracion_fc - duracion_pdm, "día")}). ' if duracion_pdm < duracion_fc else f'(PDM alarga {plural(duracion_pdm - duracion_fc, "día")}). ' if duracion_pdm > duracion_fc else '(la misma duración). ')
         + 'Difiere porque CC, FF y CF permiten solapar la sucesora con la predecesora y los desfases añaden esperas. '
         + 'Acortan el proyecto: ' + ('; '.join(_acorta) or 'ninguna') + '. Lo alargan: ' + ('; '.join(_alarga) or 'ninguna') + '.' + (f' No influyen por separado (no fijan la fecha de ninguna actividad crítica, o hay otra relación que fija la misma fecha): {lista_y(_nada)}.' if _nada else '')
         + ' Los efectos de cada relación por separado no tienen por qué sumar la diferencia total: al cambiar varias a la vez cambia el camino crítico.')

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy equivalente con todas las relaciones FC + 0"
#| label: fig-pertFc
proyecto_fc.display_roy_pasos(tabla_fc['duration'])

