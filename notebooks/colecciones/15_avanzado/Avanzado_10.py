# ---
# title: Avanzado 10
# subtitle: "9 actividades; inicio 2026-03-16, 2 festivos"
# coleccion: 15_avanzado
# variante: fechas-calendario
# nivel: 2
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en **días laborables**) se adjunta. El proyecto empieza el lunes 2026-03-16. Se trabaja de lunes a viernes; son festivos los días 2026-03-26 y 2026-04-01.

# %%
pregunta_1 = "1. Calcule el calendario del proyecto en días laborables (inicio y fin más tempranos y más tardíos, holgura total) y su duración.  "
pregunta_2 = "2. Traduzca el calendario a fechas: fecha de inicio y de fin (último día trabajado) más tempranas y más tardías de cada actividad. ¿En qué fecha termina el proyecto? ¿Cuántos días naturales dura?  "
pregunta_3 = "3. ¿Qué actividades críticas atraviesan un festivo o un fin de semana? ¿Cambia eso su duración en días laborables? ¿Y la fecha de fin del proyecto?  "
pregunta_4 = "4. Se declara un nuevo festivo el 2026-03-27. ¿Qué fechas cambian y cuánto se retrasa la entrega? ¿Depende de que el festivo caiga en una actividad crítica?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 7       ),
    ('B'     , '---'      , 7       ),
    ('C'     , '---'      , 2       ),
    ('D'     , 'A'        , 5       ),
    ('E'     , 'B,C,D'    , 3       ),
    ('F'     , 'C,D'      , 1       ),
    ('G'     , 'F'        , 1       ),
    ('H'     , 'E,F'      , 7       ),
    ('I'     , 'G,H'      , 8       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
inicio = np.datetime64('2026-03-16')
festivos = [np.datetime64('2026-03-26'), np.datetime64('2026-04-01')]

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT con tiempos de los nodos y camino crítico"
#| label: fig-grafo
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos paso a paso (avance con los botones):

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario en días laborables"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = proyecto.duration(datos_proyecto['duration'])
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# El día laborable $k$ (contando desde 0) es `busday_offset(inicio, k)`, que salta fines de semana y festivos. Una actividad que empieza en el laborable ES y dura d ocupa los laborables ES, …, ES + d − 1: su último día trabajado es `busday_offset(inicio, EF − 1)`.

# %%
def fecha(k):
    return np.busday_offset(inicio, int(k), roll='forward', holidays=festivos)
fechas = pd.DataFrame({
    'inicio temprano': [fecha(k) for k in calendario['inicio_mas_temprano']],
    'fin temprano': [fecha(k - 1) for k in calendario['fin_mas_temprano']],
    'inicio tardío': [fecha(k) for k in calendario['inicio_mas_tardio']],
    'fin tardío': [fecha(k - 1) for k in calendario['fin_mas_tardio']]}, index=calendario.index)
fechas

# %%
DIAS = {'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miércoles', 'Thursday': 'jueves', 'Friday': 'viernes', 'Saturday': 'sábado', 'Sunday': 'domingo'}
fin_proyecto = fecha(duracion - 1)
dias_naturales = int((fin_proyecto - inicio).astype(int)) + 1
Markdown(f'El proyecto termina el {fin_proyecto} ({DIAS[pd.Timestamp(fin_proyecto).day_name()]}): {duracion} días laborables son {dias_naturales} días naturales, '
         f'{dias_naturales - duracion} de ellos no trabajados ({int(np.busday_count(inicio, fin_proyecto + np.timedelta64(1, "D"), weekmask="Sat Sun"))} de fin de semana y {len([f for f in festivos if inicio <= f <= fin_proyecto])} festivos).')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
def dias_no_laborables(ini, fin):
    todos = np.arange(ini, fin + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    return int(sum(1 for x in todos if not np.is_busday(x, holidays=festivos)))
criticas = calendario.index[calendario['H_total'] == 0]
cruce = pd.DataFrame({'inicio': fechas.loc[criticas, 'inicio temprano'], 'fin': fechas.loc[criticas, 'fin temprano'],
                      'días no laborables que atraviesa': [dias_no_laborables(fechas.loc[a, 'inicio temprano'], fechas.loc[a, 'fin temprano']) for a in criticas],
                      'incluye festivo': [any(fechas.loc[a, 'inicio temprano'] <= f <= fechas.loc[a, 'fin temprano'] for f in festivos) for a in criticas]})
display(cruce)
Markdown('La duración en días laborables no cambia: los festivos y fines de semana sólo estiran la actividad en el calendario natural. '
         'La fecha de fin del proyecto sí depende de cuántos días no laborables caen dentro del horizonte, estén donde estén.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
festivo_extra = np.datetime64('2026-03-27')
festivos2 = sorted(festivos + [festivo_extra])
def fecha2(k):
    return np.busday_offset(inicio, int(k), roll='forward', holidays=festivos2)
fechas2 = pd.DataFrame({'fin temprano (antes)': fechas['fin temprano'], 'fin temprano (con el nuevo festivo)': [fecha2(k - 1) for k in calendario['fin_mas_temprano']]})
fechas2['cambia'] = fechas2.iloc[:, 0] != fechas2.iloc[:, 1]
display(fechas2)
fin2 = fecha2(duracion - 1)
Markdown(f'''El nuevo festivo desplaza un día natural todas las fechas posteriores a él ({", ".join(fechas2.index[fechas2["cambia"]])}), críticas o no: la entrega pasa del {fin_proyecto} al {fin2}. No depende de la criticidad, porque el festivo no consume días laborables de ninguna actividad: desplaza el calendario entero. Sólo si el festivo cayera después del fin del proyecto no afectaría.''')
