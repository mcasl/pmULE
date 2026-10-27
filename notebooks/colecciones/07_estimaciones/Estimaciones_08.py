# ---
# title: Estimaciones 08
# subtitle: "9 actividades, 2 de duración fija"
# coleccion: 07_estimaciones
# variante: deterministas
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
# Se dispone de las tres estimaciones (optimista, más probable, pesimista; en semanas) de cada actividad. Algunas actividades tienen duración fija (las tres estimaciones coinciden).

# %%
pregunta_1 = "1. Calcule la duración media y la varianza de cada actividad. ¿Qué varianza tienen las actividades de duración fija?  "
pregunta_2 = "2. Determine el camino crítico y la duración media del proyecto.  "
pregunta_3 = "3. Calcule la varianza del proyecto. ¿Qué actividades contribuyen a ella? ¿Contribuye la actividad no crítica más variable? ¿Y las críticas de duración fija?  "
pregunta_4 = "4. Si pudiera reducir a cero la varianza de una sola actividad, ¿cuál elegiría para hacer el plazo más predecible? ¿Y cuál para acortarlo?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b
    ('A'     , '---'      , 3, 4, 5),
    ('B'     , '---'      , 6, 6, 6),
    ('C'     , 'A,B'      , 1, 2, 3),
    ('D'     , 'C'        , 1, 1, 7),
    ('E'     , 'C'        , 2, 5, 8),
    ('F'     , 'C'        , 1, 2, 5),
    ('G'     , 'E,F'      , 3, 3, 9),
    ('H'     , 'D,E'      , 6, 6, 6),
    ('I'     , 'G'        , 1, 3, 5),
], columns=['activity', 'predecessor', 'a', 'm', 'b']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde al lanzamiento de una aplicación móvil; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de requisitos',
    'B': 'Diseño de la arquitectura',
    'C': 'Diseño de interfaz y prototipo',
    'D': 'Base de datos y modelo',
    'E': 'API del servidor',
    'F': 'Pantallas principales',
    'G': 'Módulo de pagos',
    'H': 'Notificaciones y sincronización',
    'I': 'Pruebas de integración',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Con las tres estimaciones $a$, $m$, $b$ del cuadro, el método PERT da para cada actividad $D = (a + 4m + b)/6$ y $\sigma^2 = ((b - a)/6)^2$:

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = datos_proyecto.eval('(a + 4 * m + b) / 6').round().astype(int)
datos_proyecto['variance'] = datos_proyecto.eval('((b - a) / 6) ** 2')

# %%
#| tbl-cap: "Media y varianza PERT de cada actividad con los números sustituidos"
#| label: tbl-estimaciones
detalle_estimaciones(datos_proyecto)

# %%
fijas = [x for x in datos_proyecto.index if datos_proyecto.loc[x, 'a'] == datos_proyecto.loc[x, 'b']]
Markdown(f'''Cuando $a = m = b$ la varianza es $((b - a)/6)^2 = 0$: la actividad no aporta incertidumbre, aunque sí aporta duración.

**Respuesta:** las actividades de duración fija son {lista_y(fijas)} y su varianza es 0 ('''
         + lista_y([f'{x} dura siempre {plural(datos_proyecto.loc[x, "duration"], "semana")}' for x in fijas])
         + '''). La media y la varianza de las demás actividades son las de la tabla.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con duraciones medias y camino crítico"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
duracion_media = proyecto.duration(datos_proyecto['duration'])
calendario = proyecto.calendar(datos_proyecto['duration'])
criticas = ordena_ruta(sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)), calendario)
display(explica_camino_critico(proyecto, datos_proyecto['duration']))
Markdown(f'**Respuesta:** camino crítico {" – ".join(criticas)}; duración media del proyecto {plural(duracion_media, "semana")}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# La varianza del proyecto se aproxima por la de la ruta crítica: la suma de las varianzas de sus actividades (que se suponen independientes).

# %%
#| tbl-cap: "Varianza de la ruta crítica"
#| label: tbl-varianzaRuta
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['variance'])
varianza_rutas.relabel_index(['actividades', 'varianza', 'suma de varianzas'], axis='columns')

# %%
#| tbl-cap: "Varianza de cada actividad y si es crítica"
#| label: tbl-varianzas
contribuyen = datos_proyecto.loc[criticas, 'variance']
mas_variable_no_critica = datos_proyecto['variance'].drop(criticas).idxmax()
limpia_tabla(pd.DataFrame({'crítica': datos_proyecto.index.isin(criticas), 'holgura total': calendario['H_total'],
                           'varianza': datos_proyecto['variance']}, index=datos_proyecto.index), decimales=3).T

# %%
varianza_proyecto = float(contribuyen.sum())
aportan = [a for a in criticas if contribuyen[a] > 0]
fijas_criticas = [a for a in criticas if contribuyen[a] == 0]
h_nc = int(calendario.loc[mas_variable_no_critica, 'H_total'])
if h_nc <= 2:
    matiz = (f'Sin embargo su holgura total es sólo de {plural(h_nc, "semana")}: en la realidad (y en una simulación de Monte Carlo) '
             'un retraso de ese tamaño la haría crítica, así que conviene vigilarla.')
else:
    matiz = (f'Con {plural(h_nc, "semana")} de holgura total es poco probable que llegue a ser crítica, '
             'así que despreciar su varianza es razonable.')
Markdown(f'''{explica_sigma_ruta(criticas, datos_proyecto["variance"], nombre="σ_T")} semanas, es decir, σ²_T = {varianza_proyecto:.3f}.

**Respuesta:** la varianza del proyecto es {varianza_proyecto:.3f} (σ = {np.sqrt(varianza_proyecto):.3f} semanas). Contribuyen {lista_y(aportan)}, las críticas con varianza no nula. La no crítica más variable, {mas_variable_no_critica} (σ² = {datos_proyecto.loc[mas_variable_no_critica, "variance"]:.3f}), no contribuye en la aproximación PERT porque no está en la ruta crítica. {matiz} {('Las críticas de duración fija (' + lista_y(fijas_criticas) + ') aportan duración pero no varianza.') if fijas_criticas else 'Ninguna crítica es de duración fija.'}''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
maxima = contribuyen.max()
mas_variables = [a for a in criticas if np.isclose(contribuyen[a], maxima)]
# si la actividad se hiciera de duración fija igual a su valor más probable m
efecto = {}
for x in criticas:
    d2 = datos_proyecto['duration'].copy(); d2[x] = int(datos_proyecto.loc[x, 'm'])
    efecto[x] = int(proyecto.duration(d2))
mejor = min(efecto, key=lambda x: (efecto[x], -contribuyen[x]))
tabla_efecto = limpia_tabla(pd.DataFrame({'m': datos_proyecto.loc[criticas, 'm'], 'media D': datos_proyecto.loc[criticas, 'duration'],
                                          'varianza': contribuyen,
                                          'σ²_T sin su varianza': varianza_proyecto - contribuyen,
                                          'duración si se fija en m': pd.Series(efecto)}), decimales=3)
display(tabla_efecto.T)
if len(mas_variables) == 1:
    quien = f'{mas_variables[0]} (la crítica de mayor varianza, {maxima:.3f}): sin ella'
else:
    quien = f'{lista_y(mas_variables, "o")} (empatadas como críticas de mayor varianza, {maxima:.3f} cada una): sin la de cualquiera de ellas'
texto_pred = (f'{quien} σ²_T bajaría de {varianza_proyecto:.3f} '
              f'a {varianza_proyecto - maxima:.3f} (σ_T = {np.sqrt(varianza_proyecto - maxima):.3f} semanas)')
if efecto[mejor] < duracion_media:
    texto_acortar = (f'Anular la varianza manteniendo la media no cambia la duración media ({duracion_media}). Si anularla supone fijar la '
                     f'actividad en su valor más probable m, la que más acorta es {mejor}: m = {int(datos_proyecto.loc[mejor, "m"])} < D = '
                     f'{datos_proyecto.loc[mejor, "duration"]}, y el proyecto pasaría a durar {efecto[mejor]} semanas.')
else:
    texto_acortar = (f'Anular la varianza manteniendo la media no cambia la duración media ({duracion_media}), y fijar una crítica en su '
                     f'valor más probable tampoco la acorta (ninguna crítica tiene m < D): para acortar el plazo hay que reducir la duración '
                     f'de alguna crítica, no su varianza.')
Markdown(f'''**Respuesta:** para hacer el plazo más predecible elegiría {texto_pred}. {texto_acortar} '''
         'Reducir la varianza de una no crítica no cambia nada en el análisis PERT.')
