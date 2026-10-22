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

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, plot_cargas

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
proyecto.display_path_matrix(dummies=False)

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
duraciones_caminos.T.head()

# %% [markdown]
# La duración del proyecto en cada iteración se obtiene calculando el máximo de la duración de los caminos en esa iteración.

# %%
duracion_proyecto = duraciones_caminos.max(axis="rows").to_frame(name='duracion_proyecto')
duracion_proyecto.head(10)

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
Markdown(f"La duración media del proyecto es: {round(duracion_proyecto.mean(), 1)}, la desviación típica de la duración del proyecto es: {round(duracion_proyecto.std(), 1)}")

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
Markdown(f"Podemos ordenar los valores en sentido ascendente y quedarnos con aquél valor superior al 98% de las muestras. De esta manera obtenemos un resultado igual a {resultado_1} periodos")

# %% [markdown]
# O bien, utilizando la función `quantile`

# %%
resultado_2 = round(duracion_proyecto.quantile(q=0.98, interpolation='higher'), 1)
from pmule import display_histograma_tikz
display_histograma_tikz(duracion_proyecto['duracion_proyecto'], percentil=0.98)

# %%
Markdown(f"O bien, utilizando la función `quantile`, obteniendo igualmente un valor de {resultado_2} periodos.")

# %%

# %%
