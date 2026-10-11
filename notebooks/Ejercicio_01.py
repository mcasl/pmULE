# ---
# title: Ejercicio 01
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
from IPython.display import display, HTML, Markdown, Math, Latex
import pandas as pd
import numpy as np
import sympy as sym
from scipy.stats import norm
import math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, pinta_tikz,  beautify


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %%
def bmatrix(a): # bmatrix esta sacada de: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    """Returns a LaTeX bmatrix
    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    a = a.values
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


# %% [markdown]
# # Enunciado

# %% [markdown]
# Utilizando los valores que se proporcionan de la duración de las actividades, se pide:

# %%
pregunta_1 = Markdown("1.- Dibujar el grafo PERT y determinar el cuadro de ancestros (prelaciones transitivas)  ")
pregunta_2 = Markdown("2.- Determinar la matriz de caminos del proyecto  ")
pregunta_3 = Markdown("3.- Utilizar la matriz de caminos del proyecto para calcular su duración  ")
pregunta_4 = Markdown("4.- Calcular la duración para probabilidad 95% de fin de proyecto utilizando la ley de distribución normal  ")
pregunta_5 = Markdown("5.- Calcular la duración para probabilidad 95% de fin de proyecto utilizando los resultados numéricos obtenidos en las distintas iteraciones del método de MonteCarlo  ")

# %%
for item in dir():
    if item.startswith('pregunta_'):
        display(eval(item))

# %%
duraciones=pd.DataFrame.from_records([
( 8.747,  13.512,   7.919,  6.359,  8.835,  8.398,  10.201,  4.238,  1.716),
(10.367,  12.390,   7.782,  4.897,  8.747,  7.388,   8.980,  3.645,  1.932),
( 8.329,  11.379,   7.075,  5.388,  9.697,  8.341,   9.345,  4.305,  2.589),
(13.191,   9.785,   5.011,  4.946,  9.557,  6.871,   9.014,  3.533,  1.238),
(10.659,  13.125,   7.620,  3.623,  8.311,  9.433,   8.628,  3.373,  2.297),
( 8.359,  11.955,   6.944,  4.585,  8.293,  9.980,   9.094,  4.146,  2.166),
(10.975,  11.984,   6.844,  4.606,  9.365,  7.633,   8.098,  3.778,  2.532),
(11.477,  12.944,   5.529,  4.941,  9.769,  6.956,   9.733,  4.001,  1.848),
(11.152,  12.821,   6.522,  6.100,  8.888,  8.570,   9.077,  4.037,  2.185),
( 9.389,  12.594,   7.418,  5.763,  9.881,  7.865,  10.086,  3.705,  2.134)
], columns=list('ABCDEFGHI'))


datos_enunciado = pd.DataFrame()
datos_enunciado['predecesoras'] = {
    "A": "---  ",
    "B": "---  ",
    "C": "---  ",
    "D": "A    ",
    "E": "B    ",
    "F": "C,D,E",
    "G": "A    ",
    "H": "B    ",
    "I": "F    ",
}
datos_enunciado.index.name='actividad'
proyecto = ProjectGraph.from_dict_of_strings(datos_enunciado['predecesoras'])

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado
pd.concat([datos_enunciado, duraciones.T], axis=1)

# %% [markdown]
# # Resolución:
# ## Apartado 1

# %%
display(pregunta_1)

# %% [markdown]
# ### Cuadros de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto,
                                                         activity='actividad',
                                                         predecessor='predecesoras')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos
#
# Utilizando el cuadro de prelaciones expandido podemos dibujar el grafo Pert correspondiente al proyecto.

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto.pert(durations=False, ranksep=1)

# %%
#| tbl-cap: "Cuadro de ancestros (prelaciones transitivas)"
#| label: tbl-CuadroAncestros
proyecto.distant_predecessor(format='DataFrame')

# %% [markdown]
# ## Apartado 2

# %%
display(pregunta_2)

# %%
#| tbl-cap: "Matriz de rutas"
#| label: tbl-MatrixRutas
proyecto.display_path_matrix(dummies=False)

# %% [markdown]
# ## Apartado 3

# %%
display(pregunta_3)

# %% [markdown]
# Multiplicando la matriz de rutas con las duraciones de las actividades, se obtienen la duración de las rutas en cada iteración:

# %%
path_matrix = proyecto.path_matrix(dummies=False)
D = duraciones.T
duraciones_de_caminos = (path_matrix @ D)

# %%
# LaTex for pretty print, the calculations are already done
Math(  bmatrix(path_matrix) + " \\times " + bmatrix(D) + " = " + bmatrix(duraciones_de_caminos) )

# %% [markdown]
# Las duraciones de las rutas son entonces:

# %%
#| tbl-cap: "Duraciones de las rutas en cada iteración"
#| label: tbl-DuracionesCaminos
duraciones_de_caminos

# %% [markdown]
# La duración del proyecto se obtiene, en cada iteración, como el máximo de las duraciones de las rutas:

# %%
#| tbl-cap: "Duraciones del proyecto en cada iteración"
#| label: tbl-DuracionesProyecto
duraciones_de_proyecto = duraciones_de_caminos.max(axis="rows").to_frame(name='Duración proyecto')
duraciones_de_proyecto.T

# %%
#| tbl-cap: "Cuadro resumen de descriptores estadísticos de la duración del proyecto"
#| label: tbl-DescriptoresEstadisticos
duraciones_de_proyecto.describe().round(2)

# %%
media_duracion_proyecto = duraciones_de_proyecto.mean().iloc[0]
desviacion_tipica=duraciones_de_proyecto.std().iloc[0]
print(f'La duración media del proyecto es {media_duracion_proyecto:.2f} y la desviación típica es {desviacion_tipica:.2f}')

# %% [markdown]
# ## Apartado 4

# %%
display(pregunta_4)

# %%
from scipy.stats import norm

# %%
percentil = 0.95
duracion_para_percentil = round(norm.ppf(percentil, loc=media_duracion_proyecto, scale=desviacion_tipica), 2)
Markdown(f'Si utilizamos la ley normal de distribución, para una duración media {round(media_duracion_proyecto, 2)} y desviación típica {round(desviacion_tipica,2)} la duración para probabilidad igual a {percentil*100} por ciento es {duracion_para_percentil} periodos')

# %% [markdown]
# ## Apartado 5

# %%
display(pregunta_5)

# %%
duracion_para_percentil_montecarlo =round(float(duraciones_de_proyecto.quantile(percentil).iloc[0]),2)

# %%
Markdown(f'Si utilizamos la colección de resultados del método de MonteCarlo, la duración para el cuantil {percentil*100} por ciento es {duracion_para_percentil_montecarlo} periodos.')

# %%

# %%

# %%

# %%

# %%
