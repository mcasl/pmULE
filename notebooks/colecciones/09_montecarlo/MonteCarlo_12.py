# ---
# title: MonteCarlo 12
# subtitle: "7 actividades, 3 rutas, 5 réplicas calculadas a mano"
# coleccion: 09_montecarlo
# variante: a-mano
# nivel: 1
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones medias (en días) se adjunta. Para entender el método de Monte Carlo se han generado a mano sólo 5 réplicas de las duraciones (cada columna es una réplica: la duración que cada actividad ha tenido en esa simulación).

# %%
#| tbl-cap: "Cuadro de prelaciones y duraciones medias"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 3       ),
    ('B'     , '---'      , 6       ),
    ('C'     , '---'      , 4       ),
    ('D'     , 'A,B,C'    , 7       ),
    ('E'     , 'D'        , 4       ),
    ('F'     , 'E'        , 4       ),
    ('G'     , 'F'        , 6       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la rehabilitación de una nave industrial para un taller; las actividades son las siguientes (las duraciones del cuadro están en días).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Levantamiento topográfico y mediciones',
    'B': 'Proyecto de ejecución y licencia',
    'C': 'Desmontaje de instalaciones antiguas',
    'D': 'Demolición de tabiquería interior',
    'E': 'Refuerzo de la estructura metálica',
    'F': 'Reparación de la cubierta',
    'G': 'Solera y canalizaciones',
}, name='Descripción')
descripciones.to_frame()

# %%
#| tbl-cap: "Duraciones simuladas (una columna por réplica)"
#| label: tbl-replicas
replicas = pd.DataFrame([
#  activity, r1, r2, r3, r4, r5
    ('A'     , 3 , 2 , 4 , 4 , 4 ),
    ('B'     , 5 , 7 , 5 , 6 , 7 ),
    ('C'     , 4 , 5 , 4 , 4 , 4 ),
    ('D'     , 6 , 8 , 8 , 6 , 8 ),
    ('E'     , 5 , 5 , 4 , 3 , 3 ),
    ('F'     , 4 , 4 , 5 , 5 , 5 ),
    ('G'     , 7 , 5 , 7 , 6 , 6 ),
], columns=['activity', 'r1', 'r2', 'r3', 'r4', 'r5']).set_index('activity')
replicas

# %%
pregunta_1 = "1. Dibuje el grafo Roy, enumere las rutas y escriba la matriz de rutas R (una fila por ruta, una columna por actividad).  "
pregunta_2 = "2. Escriba la matriz de tiempos T con las réplicas como columnas y calcule a mano el producto R·T. ¿Qué es cada elemento del resultado?  "
pregunta_3 = "3. Obtenga la duración del proyecto en cada réplica como el máximo de cada columna (τ = ‖R·t‖∞ para cada columna t). Calcule la media y la desviación típica muestral de las cinco duraciones.  "
pregunta_4 = "4. Para cada réplica, ¿qué ruta ha sido la crítica? Calcule el índice de criticidad de cada ruta y de cada actividad con estas cinco réplicas. Compare con el camino crítico de las duraciones medias.  "
pregunta_5 = "5. ¿Qué cambia al pasar de 5 réplicas a 10 000? Compruébelo generando las réplicas con una distribución normal de media la duración media y desviación típica 1.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
def bmatrix(a, dec=0):
    """Matriz LaTeX (bmatrix) a partir de un DataFrame o array."""
    a = np.asarray(getattr(a, 'values', a), dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    filas = [' & '.join(f'{x:.{dec}f}' for x in fila) + r'\\' for fila in a]
    return r'\begin{bmatrix}' + '\n' + '\n'.join(filas) + '\n' + r'\end{bmatrix}'

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con las duraciones medias"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# Cada fila de la matriz de rutas R es una ruta y cada columna una actividad: 1 si la actividad está en la ruta y 0 si no.

# %%
#| tbl-cap: "Matriz de rutas R"
#| label: tbl-matrizRutas
R = proyecto.path_matrix(dummies=False)
rutas = proyecto.paths(dummies=False)
proyecto.display_rutas("roy")

# %%
#| tbl-cap: "Rutas del proyecto y su duración con las duraciones medias"
#| label: tbl-rutas
tabla_rutas = explica_rutas(proyecto, datos_proyecto['duration'], unidad='días')
tabla_rutas

# %%
Markdown(f'''**Respuesta:** hay {len(R)} rutas: ''' + '; '.join(f"{r}: {tabla_rutas.loc[r, 'ruta']}" for r in R.index)
         + '. La matriz R es la de la tabla anterior (filas = rutas, columnas = actividades).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# La matriz de tiempos T tiene una fila por actividad y una columna por réplica (es la tabla de réplicas del enunciado). Cada elemento de R·T se obtiene multiplicando una fila de R por una columna de T, es decir, sumando las duraciones de esa réplica de las actividades marcadas con 1:

# %%
T = replicas.loc[R.columns]
RT = R @ T
from pmule import display_producto_rutas
display_producto_rutas(R, T, RT)

# %%
#| tbl-cap: "R·T: duración de cada ruta (filas) en cada réplica (columnas), término a término"
#| label: tbl-RT
pd.DataFrame({j: [' + '.join(f'{T.loc[a, j]}' for a in R.columns[R.loc[r] == 1]) + f' = {RT.loc[r, j]}' for r in R.index]
              for j in T.columns}, index=R.index)

# %%
r0, j0 = R.index[0], T.columns[0]
acts0 = list(R.columns[R.loc[r0] == 1])
Markdown(f'''**Respuesta:** el elemento (r, j) de R·T es la duración de la ruta r en la réplica j: la suma de las duraciones, en esa '''
         f'''réplica, de las actividades que forman la ruta. Por ejemplo, {r0} en {j0}: '''
         + ' + '.join(f'{a}({T.loc[a, j0]})' for a in acts0) + f''' = {RT.loc[r0, j0]} días. Una fila de R selecciona las actividades '''
         '''de la ruta; una columna de T es un escenario completo.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
tau = RT.max()
mas_largas = {j: [r for r in RT.index if RT.loc[r, j] == tau[j]] for j in RT.columns}
pd.DataFrame({'τ = máximo de la columna': tau, 'rutas más largas': {j: ', '.join(v) for j, v in mas_largas.items()}}).T

# %%
media = tau.mean(); desviacion = tau.std(ddof=1)
dif2 = [(x - media) ** 2 for x in tau]
Markdown(f'''Media = ({" + ".join(str(int(x)) for x in tau)}) / {len(tau)} = {int(tau.sum())} / {len(tau)} = {media:.2f} días.\n\n'''
         f'''Desviación típica muestral = √(Σ(τ − media)² / (n − 1)) = √((''' + ' + '.join(f'{d:.2f}' for d in dif2)
         + f''') / {len(tau) - 1}) = √({sum(dif2):.2f} / {len(tau) - 1}) = {desviacion:.2f} días.\n\n'''
         f'''**Respuesta:** duraciones del proyecto en las réplicas: {", ".join(f"{j} = {int(tau[j])}" for j in tau.index)}; media {media:.2f} días '''
         f'''y desviación típica muestral {desviacion:.2f} días (con las duraciones medias el proyecto duraría '''
         f'''{proyecto.duration(datos_proyecto["duration"])} días).''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Una ruta es crítica en una réplica si es la más larga en ella (si hay empate, lo son todas las empatadas). El índice de criticidad de una ruta es la fracción de réplicas en que es crítica; el de una actividad, la fracción de réplicas en que está en alguna ruta crítica.

# %%
es_critica = RT.eq(tau, axis='columns')            # rutas × réplicas: ¿es la más larga?
criticidad_rutas = es_critica.sum(axis='columns')
criticidad_actividades = ((R.T.astype(int) @ es_critica.astype(int)) > 0).sum(axis='columns')
pd.DataFrame({'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in R.index],
              'veces crítica': criticidad_rutas, 'índice de criticidad': criticidad_rutas / len(tau)}, index=R.index)

# %%
pd.DataFrame({'veces en una ruta crítica': criticidad_actividades, 'índice de criticidad': criticidad_actividades / len(tau)}).T

# %%
camino_medio = proyecto.critical_path(datos_proyecto['duration'])
rutas_medias = [r for r in R.index if set(R.columns[R.loc[r] == 1]) in [set(a) for a in camino_medio.values()]]
empates = [j for j, v in mas_largas.items() if len(v) > 1]
nota_empate = (f' En {lista_y(empates)} hay empate entre rutas, que cuentan como críticas a la vez; por eso los índices de las rutas pueden sumar más de 1.'
               if empates else '')
Markdown(f'''**Respuesta:** rutas críticas por réplica: {"; ".join(f"{j}: {', '.join(v)}" for j, v in mas_largas.items())}.{nota_empate} '''
         f'''Con las duraciones medias la ruta crítica es {lista_y(rutas_medias)} ({", ".join(" – ".join(a) for a in camino_medio.values())}), '''
         f'''que en las réplicas ha sido la más larga {int(criticidad_rutas[rutas_medias].max())} de {len(tau)} veces. Con tan pocas réplicas los '''
         f'''índices sólo pueden valer múltiplos de {1/len(tau):.1f}.''')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# Se repite exactamente el mismo cálculo (R·T y máximo por columnas) con 10 000 columnas, cada una generada con duraciones normales de media la duración media y desviación típica 1.

# %%
rng = np.random.default_rng(1)
N = 10000
T_grande = pd.DataFrame(np.clip(rng.normal(datos_proyecto.loc[R.columns, 'duration'].values[:, None], 1.0, (len(R.columns), N)), 0.5, None), index=R.columns)
tau_grande = (R @ T_grande).max()
criticidad_grande = R.loc[(R @ T_grande).idxmax()].mean()
pd.DataFrame({'5 réplicas': [media, desviacion], '10 000 réplicas': [tau_grande.mean(), tau_grande.std(ddof=1)]}, index=['media', 'desviación típica']).round(2)

# %%
pd.DataFrame({'5 réplicas': criticidad_actividades / len(tau), '10 000 réplicas': criticidad_grande}).round(3).T

# %%
Markdown(f'''**Respuesta:** la operación es exactamente la misma (R·T y máximo por columnas); sólo cambia el número de columnas. '''
         f'''Con 10 000 réplicas la media es {tau_grande.mean():.2f} días y la desviación típica {tau_grande.std(ddof=1):.2f} (con 5: {media:.2f} y '''
         f'''{desviacion:.2f}). Con muchas réplicas estos valores se estabilizan (su error estadístico decrece como 1/√N: el error típico de la media '''
         f'''pasa de unos {desviacion / np.sqrt(len(tau)):.2f} días con 5 réplicas a unos {tau_grande.std(ddof=1) / np.sqrt(N):.3f} con 10 000) y los '''
         '''índices de criticidad dejan de ser múltiplos gruesos de 0.2.''')
