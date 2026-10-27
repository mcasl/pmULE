# ---
# title: Comprension 04
# subtitle: "8 actividades; dos planes con violaciones (E, F y F)"
# coleccion: 14_comprension
# variante: comparar-planificaciones
# nivel: 1
# grafo: pert
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (semanas) se adjunta. Dos jefes de obra han propuesto sendas programaciones, dadas por la semana de inicio de cada actividad (segundo cuadro). Las actividades no se interrumpen.

# %%
pregunta_1 = "1. Para cada plan, compruebe actividad por actividad si respeta las prelaciones: una actividad no puede empezar antes de que terminen todas sus predecesoras. Liste las violaciones.  "
pregunta_2 = "2. Calcule la duración del proyecto que resulta de cada plan (fin de la última actividad) y compárela con la duración mínima del proyecto.  "
pregunta_3 = "3. Corrija cada plan con el mínimo cambio: adelante o retrase sólo las actividades necesarias (y sus sucesoras si hace falta) y recalcule la duración.  "
pregunta_4 = "4. ¿Qué plan es «más tardío» (deja menos margen) y qué riesgo supone? ¿Qué plan deja más holgura para absorber imprevistos?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 7       ),
    ('B'     , '---'      , 5       ),
    ('C'     , '---'      , 4       ),
    ('D'     , 'A,B,C'    , 3       ),
    ('E'     , 'D'        , 1       ),
    ('F'     , 'E'        , 1       ),
    ('G'     , 'D'        , 7       ),
    ('H'     , 'F'        , 3       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la organización de un congreso científico; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Definición del programa y fechas',
    'B': 'Reserva de la sede',
    'C': 'Convocatoria de comunicaciones',
    'D': 'Página web e inscripciones',
    'E': 'Contratación de ponentes invitados',
    'F': 'Revisión de los trabajos recibidos',
    'G': 'Elaboración del programa definitivo',
    'H': 'Contratación de catering y viajes',
}, name='Descripción')
descripciones.to_frame()

# %%
#| tbl-cap: "Programaciones propuestas (semana de inicio)"
#| label: tbl-planes
planes = pd.DataFrame([
#  activity, start_plan_1, start_plan_2
    ('A'     , 0           , 0           ),
    ('B'     , 2           , 0           ),
    ('C'     , 3           , 0           ),
    ('D'     , 7           , 7           ),
    ('E'     , 9           , 12          ),
    ('F'     , 9           , 14          ),
    ('G'     , 10          , 10          ),
    ('H'     , 14          , 12          ),
], columns=['activity', 'start_plan_1', 'start_plan_2']).set_index('activity')
planes

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion_minima = proyecto.duration(datos_proyecto['duration'])
pred = prelaciones.immediate_predecessor
NOMBRE = {'start_plan_1': 'Plan 1', 'start_plan_2': 'Plan 2'}

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
def violaciones(inicio):
    fin = inicio + datos_proyecto['duration']
    filas = []
    for a in inicio.index:
        for p in sorted(pred[a]):
            if fin[p] > inicio[a]:
                filas.append({'actividad': a, 'empieza en': int(inicio[a]), 'predecesora': p,
                              'la predecesora termina en': f"{int(inicio[p])} + {int(datos_proyecto.loc[p, 'duration'])} = {int(fin[p])}",
                              'semanas de solape': int(fin[p] - inicio[a])})
    return pd.DataFrame(filas)
_resp = []
for plan in ('start_plan_1', 'start_plan_2'):
    v = violaciones(planes[plan])
    display(Markdown(f'**{NOMBRE[plan]}**: {plural(len(v), "violación", "violaciones")}'))
    if len(v):
        display(v.set_index('actividad'))
    _resp.append(f'{NOMBRE[plan]}: ' + (', '.join([f"{r['actividad']} empieza en {r['empieza en']} y su predecesora {r['predecesora']} no termina hasta {int(planes.loc[r['predecesora'], plan] + datos_proyecto.loc[r['predecesora'], 'duration'])}" for _, r in v.iterrows()])
                                        if len(v) else 'ninguna violación'))
Markdown('Para cada actividad se compara su inicio con el fin (inicio + duración) de cada predecesora. **Respuesta:** ' + '. '.join(_resp) + '.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# El proyecto termina cuando termina su última actividad: fin del plan = máx(inicio + duración) sobre todas las actividades. La tabla da el fin de cada actividad en cada plan.

# %%
fines = planes.add(datos_proyecto['duration'], axis='rows')
display(pd.DataFrame({f'fin en {NOMBRE[p]}': [f"{int(planes.loc[a, p])} + {int(datos_proyecto.loc[a, 'duration'])} = {int(fines.loc[a, p])}" for a in planes.index]
                      for p in planes.columns}, index=planes.index))
duraciones = fines.max()
_txt = []
for p in planes.columns:
    ult = fines.index[fines[p] == duraciones[p]].tolist()
    comp = 'igual que la' if duraciones[p] == duracion_minima else (f'{int(duraciones[p] - duracion_minima)} más que la' if duraciones[p] > duracion_minima else f'{int(duracion_minima - duraciones[p])} menos que la')
    _txt.append(f'{NOMBRE[p]} terminaría en la semana {int(duraciones[p])} (fin de {lista_y(ult)}), {comp} duración mínima')
Markdown(f'La duración mínima del proyecto (camino crítico) es {duracion_minima} semanas. **Respuesta:** ' + '; '.join(_txt) + '. '
         'Pero un plan con violaciones no es ejecutable tal cual: su duración aparente es engañosa y hay que corregirlo (apartado 3).')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Corrección mínima: se recorren las actividades en orden topológico (cada una después de sus predecesoras) y cada una empieza en el máximo entre la fecha propuesta y el fin de sus predecesoras (ya corregidas). Así sólo se retrasan las actividades que lo necesitan y, en cadena, sus sucesoras.

# %%
G = nx.DiGraph(); G.add_nodes_from(datos_proyecto.index); G.add_edges_from((p, a) for a, ps in pred.items() for p in ps)
orden = list(nx.topological_sort(G))
def corregir(inicio):
    nuevo = inicio.copy()
    for a in orden:
        fin_pred = max([nuevo[p] + datos_proyecto.loc[p, 'duration'] for p in pred[a]], default=0)
        nuevo[a] = max(int(inicio[a]), int(fin_pred))
    return nuevo
corregidos = pd.DataFrame({plan: corregir(planes[plan]) for plan in planes.columns})
cambios = (corregidos != planes)
display(pd.concat({'inicio propuesto': planes.rename(columns=NOMBRE), 'inicio corregido': corregidos.rename(columns=NOMBRE)}, axis=1))
fines_c = corregidos.add(datos_proyecto['duration'], axis='rows').max()
_txt = []
for p in planes.columns:
    mov = [f'{a} pasa de {int(planes.loc[a, p])} a {int(corregidos.loc[a, p])}' for a in cambios.index[cambios[p]]]
    dif = int(fines_c[p] - duracion_minima)
    _txt.append(f'{NOMBRE[p]}: ' + (lista_y(mov) if mov else 'no hay que mover nada') + f'; termina en la semana {int(fines_c[p])}'
                + (' (la duración mínima)' if dif == 0 else f' ({plural(dif, "semana")} más que la duración mínima)'))
Markdown('**Respuesta:** ' + '; '.join(_txt) + '.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# El margen de cada actividad en un plan es lo que puede retrasarse sin mover el fin *de ese plan*: su inicio más tardío (recalculado para la duración del plan corregido) menos el inicio programado. Un margen 0 significa que la actividad empieza en su inicio más tardío.

# %%
margen = pd.DataFrame({plan: (calendario['inicio_mas_tardio'] + (fines_c[plan] - duracion_minima) - corregidos[plan]) for plan in corregidos.columns})
display(margen.rename(columns=lambda c: f'margen restante ({NOMBRE[c]})').T)
sin_margen = {p: margen.index[margen[p] <= 0].tolist() for p in margen.columns}
mas_tardio = max(margen.columns, key=lambda p: len(sin_margen[p]))
otro = [p for p in margen.columns if p != mas_tardio][0]
_extra = '; '.join(f'{NOMBRE[p]} termina además {plural(int(fines_c[p] - duracion_minima), "semana")} después de la duración mínima ({duracion_minima})'
                   for p in margen.columns if fines_c[p] > duracion_minima)
Markdown('Actividades sin margen: ' + '; '.join(f'{NOMBRE[p]}: {len(sin_margen[p])} ({lista_y(sin_margen[p])})' for p in margen.columns)
         + '. Suma de márgenes: ' + ', '.join(f'{NOMBRE[p]}: {int(margen[p].sum())}' for p in margen.columns) + ' semanas. '
         f'**Respuesta:** el plan más tardío es el {NOMBRE[mas_tardio]}: {len(sin_margen[mas_tardio])} de sus {len(margen)} actividades empiezan en su inicio más tardío, '
         'de modo que casi cualquier retraso se traslada al fin del proyecto (han consumido su holgura antes de empezar). '
         + (f'El {NOMBRE[otro]} deja más holgura para absorber imprevistos ({int(margen[otro].sum())} semanas de margen sumado frente a {int(margen[mas_tardio].sum())}), '
            'a cambio de necesitar antes los recursos y de adelantar los pagos.' if margen[otro].sum() > margen[mas_tardio].sum() else
            f'Aun así, el margen sumado es mayor o igual en el {NOMBRE[mas_tardio]} ({int(margen[mas_tardio].sum())} frente a {int(margen[otro].sum())} semanas), concentrado en pocas actividades.') + (f' Ojo: {_extra}.' if _extra else ''))
