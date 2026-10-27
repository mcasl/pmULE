# ---
# title: Ejercicio 24
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
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, EarnedValue


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %%
# bmatrix esta sacada de: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
def bmatrix(a):
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

from pmule import *
# Alargamos el ancho de las celdas a un tanto por ciento
from IPython.display import display, HTML, Markdown, Math, Latex
display(HTML("<style>.container { width:80% !important; }</style>"))
import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np
import sympy as sym

proyecto = ProjectGraph.from_dict_of_strings({
    "A": "---",
    "B": "---",
    "C": "B",
    "D": "---",
    "E": "A,B",
    "F": "C",
    "G": "C",
    "H": "D,E",
    "I": "E,F",
    "J": "G,H",
    "K": "H,I",
    "L": "I",
    "M": "J",
    "N": "K",
})


# %%
duraciones = {
 'A': 4,
 'B': 6,
 'C': 9,
 'D': 6,
 'E': 11,
 'F': 4,
 'G': 15,
 'H': 24,
 'I': 6,
 'J': 13,
 'K': 6,
 'L': 11,
 'M': 13,
 'N': 6}

# %%
rutas = pd.Series({key: ', '.join(value) for key, value in proyecto.paths(dummies=False).items()}).to_frame().rename({0:'Activities'}, axis=1)

# %% [markdown]
# # Ejercicio:
# A partir de los datos de las posibles rutas entre el nodo inicial y final del proyecto, y de los datos de duraciones y costes siguientes:

# %%
#| tbl-cap: "Información sobre las rutas del proyecto"
#| label: tbl-datosRutas
rutas

# %%
#| tbl-cap: "Duración de las actividades del proyecto"
#| label: tbl-datosDuraciones
pd.Series(duraciones).to_frame(name='duración')

# %% [markdown]
# Se pide:

# %%
pregunta_1 = "1. Determinar la matriz de caminos del proyecto  "
pregunta_2 = "2. Utilizar la matriz de caminos del proyecto para calcular su duración  "
pregunta_3 = "3. Calcular utilizando el método de Zaderenko los tiempos tempranos y tardíos  "
pregunta_4 = "4. Cuadro de prelaciones transitivas (ancestros)  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %% [markdown]
# Realizar los cálculos utilizando números enteros con redondeo siempre hacia infinito positivo.

# %% [markdown]
# # Solución

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Resulta inmediato, a partir de los datos proporcionados en el enunciado sobre las rutas del proyecto, construir la matriz pedida.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-MatrizPrelaciones
proyecto.display_rutas("pert")
_pm = proyecto.path_matrix(dummies=False)
display(Markdown(f"**Respuesta:** la matriz de caminos tiene una fila por ruta ({len(_pm.index)}) y una columna por actividad "
                 f"({len(_pm.columns)}); el elemento vale 1 si la actividad pertenece a la ruta y 0 si no."))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Multiplicando la matriz de rutas por el vector (columna) de duraciones de las actividades proporciona el vector (columna) de duraciones de los respectivos caminos. El mayor de estos valores es la duración del proyecto.

# %%
D = pd.Series(duraciones).to_frame()
path_matrix = proyecto.path_matrix(dummies=False)
duraciones_caminos = path_matrix @ D

# %%
# LaTex for pretty print, the calculations are already done
from pmule import display_producto_rutas
display_producto_rutas(path_matrix, D, duraciones_caminos)

# %%
_dur_rutas = duraciones_caminos.iloc[:, 0]
_Tmax = _dur_rutas.max()
_cal = proyecto.calendar(duraciones)
_acts = lambda r: ordena_ruta(list(_pm.columns[_pm.loc[r] == 1]), _cal)
Markdown(f"La mayor duración de los caminos es {duraciones_caminos.values.max()}: máx({', '.join(format(v, 'g') for v in _dur_rutas)}) = "
         f"{_Tmax:g}, que corresponde a "
         + lista_y([f"{r} ({' – '.join(_acts(r))} = {' + '.join(format(duraciones[a], 'g') for a in _acts(r))})"
                    for r in _dur_rutas.index if _dur_rutas[r] == _Tmax])
         + f" (camino crítico).\n\n**Respuesta:** la duración del proyecto es **{_Tmax:g}** unidades de tiempo.")

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Contruimos primero el grafo Pert a partir de los datos del enunciado sobre las rutas del proyecto.

# %%
#| fig-cap: "Grafo Pert"
#| label: fig-PertSinTiempos
proyecto.pert(durations=False, ranksep=1)

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Cálculo de los tiempos tempranos y tardíos con el método de Zaderenko"
#| label: tbl-MatrizZaderenko
proyecto.display_zaderenko_tikz(duraciones)
_nodos = proyecto.calculate_pert(durations=duraciones)['nodes']
display(Markdown("**Respuesta:** tiempos temprano / tardío de cada nodo: "
                 + "; ".join(f"nodo {n}: {r.early:g} / {r.late:g}" for n, r in _nodos.iterrows())
                 + f". Los nodos con ambos tiempos iguales están en el camino crítico; la duración del proyecto es "
                 f"{_nodos['early'].max():g}."))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación de tiempos y rutas del camino crítico"
#| label: fig-PertConTiempos
proyecto.display_pert_pasos(duraciones)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Para construir el cuadro de ancestros prestamos atención a todas las actividades aguas arriba de las que depende cada actividad (sus precedentes directas, las precedentes de estas, y así sucesivamente).

# %%
#| tbl-cap: "Cuadro de prelaciones transitivas (ancestros)"
#| label: tbl-CuadroAncestros
proyecto.display_distant_linkage_matrix()

# %%

# %%

# %%

# %%

# %%
