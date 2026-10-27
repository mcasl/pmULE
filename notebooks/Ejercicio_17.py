# ---
# title: Ejercicio 17
# lang: es
# documentclass: scrreprt
# papersize: A4
# classoption: []
# execute:
#     echo: True
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
#| echo: False
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, plot_cargas, explica_percentil, ordena_ruta, plural, lista_y

# %%
#| echo: false
#| output: false
# #%%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# # Enunciado
# ## Cálculo de la duración de un proyecto usando el Método de Montecarlo

# %% [markdown]
# Considere el proyecto cuya información se adjunta.

# %%
#| echo: False
pregunta_1 = f"1. Dibuje el grafo PERT del proyecto.  "
pregunta_2 = f"2. Determine la matriz de caminos del proyecto.  "
pregunta_3 = f"3. Calcule, utilizando la matriz de caminos del proyecto, los tiempos tempranos de cada nodo para 1000 iteraciones del método de MonteCarlo.  "
pregunta_4 = f"4. Determina la duración media y la desviación típica de la duración del proyecto.  "
pregunta_5 = f"5. Determine la duración para una probabilidad de completar antes el proyecto del 98%.  "

# %%
#| echo: False
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))
display(Markdown(pregunta_5))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import numpy as np
import matplotlib as plt
# %matplotlib inline

datos_enunciado = pd.DataFrame([
#  actividad, predecesora, duracion media, desviación típica
     ('A',    '---  ', 10, 1.5 ),
     ('B',    '---  ',  5, 0.4 ),
     ('C',    'A    ',  6, 0.8 ),
     ('D',    'A    ',  2, 0.7 ),
     ('E',    'B,C,D',  7, 1.8 ),
     ('F',    'B,D  ',  8, 0.8 ),
     ('G',    'E,F  ',  4, 1.4 ),
     ('H',    'B    ',  7, 1.6 ),
] ,columns=['activity', 'predecessor', 'mean', 'stdev']).set_index('activity')

datos_enunciado

# %% [markdown]
# ## Solución
# ### Apartado 1

# %%
#| echo: False
Markdown(pregunta_1)

# %%
datos_proyecto =  datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
proyecto.pert(ranksep=0.6, nodesep=0.6)

# %% [markdown]
# ### Apartado 2

# %%
#| echo: False
Markdown(pregunta_2)

# %%
path_matrix = proyecto.path_matrix(dummies=False)
proyecto.display_rutas("pert")
_cal_medio = proyecto.calendar(datos_proyecto['mean'])
_rutas = {r: ordena_ruta(list(path_matrix.columns[path_matrix.loc[r] == 1]), _cal_medio) for r in path_matrix.index}
display(Markdown(
    f"**Respuesta:** la matriz de caminos tiene una fila por ruta del nodo inicial al final y una columna por actividad "
    f"(1 si la actividad está en la ruta). Hay {plural(len(_rutas), 'ruta')}:\n\n"
    + "\n".join(f"- {r}: {' – '.join(acts)}" for r, acts in _rutas.items())))

# %% [markdown]
# ### Apartado 3

# %%
#| echo: False
Markdown(pregunta_3)

# %% [markdown]
# #### Generación aleatoria de las duraciones de las actividades

# %%
number_of_samples = 1000

# %%
activity_names = datos_proyecto.index
np.random.seed(2025)   # reproducibilidad de la simulación
duration = pd.DataFrame([np.random.normal(size = number_of_samples,
                                          loc  = datos_proyecto.loc[task, 'mean',],
                                          scale= datos_proyecto.loc[task, 'stdev'])
                         for task in activity_names
                        ],
                        index=activity_names)

# %%
duration.T.head()

# %% [markdown]
# #### Cálculo de la duración del proyecto

# %% [markdown]
# La duración del proyecto se obtiene multiplicando la matriz de caminos por las duraciones de las actividades. El resultado es la duración de las rutas en cada iteración.

# %%
duraciones_caminos = path_matrix @ duration   # Multiplicación matricial
_d0 = duration[0].round(2)
display(Markdown(
    "Cada elemento es la suma de las duraciones de las actividades de la ruta en esa iteración. Por ejemplo, en la "
    "iteración 0:\n\n"
    + "\n".join(f"- {r} ({' – '.join(acts)}): {' + '.join(format(_d0[a], '.2f') for a in acts)} = "
                 f"{duraciones_caminos.loc[r, 0]:.2f}" for r, acts in _rutas.items())))
display(duraciones_caminos.T.head().round(2))

# %% [markdown]
# La duración del proyecto en cada iteración se obtiene calculando el máximo de la duración de los caminos en esa iteración.

# %%
duracion_proyecto = duraciones_caminos.max(axis="rows").to_frame(name='duracion_proyecto')
display(duracion_proyecto.head(10).round(2))
import networkx as _nx
_G = proyecto.pert_graph
_inicio = [n for n in _G.nodes if _G.in_degree(n) == 0][0]
_nodos = sorted(_G.nodes, key=int)
_t_nodos = {}
_rutas_nodo = {}
for _n in _nodos:
    if _n == _inicio:
        _t_nodos[f't_{_n}'] = pd.Series(0.0, index=duration.columns)
        continue
    _filas = []
    for _camino in _nx.all_simple_paths(_G, _inicio, _n):
        _acts = [_G.edges[u, v]['activity'] for u, v in zip(_camino[:-1], _camino[1:])]
        _filas.append([a for a in _acts if not a.startswith('@')])
    _rutas_nodo[_n] = _filas
    _pm = pd.DataFrame([[int(a in f) for a in activity_names] for f in _filas], columns=activity_names)
    _t_nodos[f't_{_n}'] = (_pm @ duration).max(axis='rows')
_t_nodos = pd.DataFrame(_t_nodos)
display(Markdown(
    "Lo mismo vale para cualquier nodo: su tiempo temprano es el máximo de las duraciones de los caminos que van del nodo "
    "inicial a él (matriz de caminos parcial hasta ese nodo). Caminos que llegan a cada nodo:\n\n"
    + "\n".join(f"- nodo {n}: " + "; ".join(' – '.join(f) for f in filas) for n, filas in _rutas_nodo.items())
    + f"\n\n**Respuesta:** tiempos tempranos de cada nodo en las primeras iteraciones (el del nodo final, "
    f"t_{_nodos[-1]}, es la duración del proyecto y coincide con la columna anterior):"))
display(_t_nodos.head(10).round(2))

# %% [markdown]
# ### Apartado 4

# %%
#| echo: False
Markdown(pregunta_4)

# %%
from pmule import display_histograma_tikz
display_histograma_tikz(duracion_proyecto['duracion_proyecto'])

# %%
duracion_proyecto.describe()

# %%
_x = duracion_proyecto['duracion_proyecto']
Markdown(
    f"Con las {number_of_samples} duraciones simuladas: media $\\bar x = \\sum x_i / n$ = {_x.sum():.2f} / {number_of_samples} = "
    f"{_x.mean():.2f}; desviación típica muestral $s = \\sqrt{{\\sum (x_i - \\bar x)^2 / (n-1)}}$ = "
    f"√({((_x - _x.mean()) ** 2).sum():.2f} / {number_of_samples - 1}) = {_x.std():.2f}.\n\n"
    f"**Respuesta:** la duración media del proyecto es **{_x.mean():.1f}** periodos y su desviación típica, **{_x.std():.1f}** periodos.")

# %% [markdown]
# ### Apartado 5

# %%
#| echo: False
Markdown(pregunta_5)

# %% [markdown]
# Podemos ordenar los valores en sentido ascendente y quedarnos con aquél valor superior al 98% de las muestras.

# %%
resultado_1 = round(duracion_proyecto.sort_values(ascending=True, by='duracion_proyecto').iloc[int(0.98*number_of_samples),0], 1)

# %%
Markdown(f"Con las {number_of_samples} duraciones ordenadas de menor a mayor se toma la que ocupa la posición "
         f"0.98 × {number_of_samples} + 1 = {int(0.98 * number_of_samples) + 1} (por debajo quedan {int(0.98 * number_of_samples)} "
         f"iteraciones, el 98 %). De esta manera obtenemos un resultado igual a {resultado_1} periodos.")

# %% [markdown]
# O bien, utilizando la función `quantile`

# %%
resultado_2 = round(duracion_proyecto.quantile(q=0.98, interpolation='higher'), 1)
from pmule import display_histograma_tikz
display_histograma_tikz(duracion_proyecto['duracion_proyecto'], percentil=0.98)

# %%
Markdown(f"Utilizando la función `quantile` se obtiene igualmente un valor de {float(resultado_2.iloc[0])} periodos.\n\n"
         f"Como comprobación, si se aproxima la duración por una normal con la media y la desviación típica del apartado 4: "
         + explica_percentil(round(_x.mean(), 2), round(_x.std(), 2), 0.98, 'periodos')
         + ", muy parecido al resultado de la simulación.\n\n"
         f"**Respuesta:** el proyecto se completa antes de **{float(resultado_2.iloc[0])}** periodos con una probabilidad del 98 %.")

# %%

# %%
