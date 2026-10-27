# ---
# title: Ejercicio 23
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

from pmule import PredecessorTable, ProjectGraph


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


from pmule import *
# Alargamos el ancho de las celdas a un tanto por ciento
from IPython.display import display, HTML, Markdown, Math, Latex
display(HTML("<style>.container { width:70% !important; }</style>"))

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np
import sympy as sym
import matplotlib

datos_enunciado = pd.DataFrame(data=np.array([
       [ 2,  9,  3,  1, 10,  5],
       [ 3, 13,  5,  2, 12,  4],
       [ 4, 17,  7,  3, 43, 35],
       [ 3, 13,  5,  1, 54, 35],
       [ 5, 21,  9,  2, 23, 25],
       [ 2,  9,  3,  1, 34, 23],
       [ 7, 29, 13,  3, 23, 12],
       [ 3, 13,  5,  2, 54, 34],
       [ 6, 25, 11,  3, 65, 44],
       [ 3, 13,  5,  2, 67, 55],
       [ 5, 21,  9,  1, 34, 23],
       [ 6, 25, 11,  3, 23, 12],
       [ 3, 13,  5,  1, 12, 10]]),
   columns=['a', 'b', 'm', 'Dtope', 'Ctope', 'Cmedia'],
   index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'],
   dtype=np.int32)

proyecto = ProjectGraph.from_dict_of_strings({
    "A": "---",
    "B": "---",
    "C": "B",
    "D": "---",
    "E": "A,C",
    "F": "C",
    "G": "C",
    "H": "D,E",
    "I": "E,F",
    "J": "G,I",
    "K": "I,G",
    "L": "J,H",
    "M": "K",
})

#proyecto.pert("pert_2023-12.png", durations=False, ranksep=1)

rutas = pd.Series({key: ','.join(value) for key, value in proyecto.paths(dummies=False).items()}).to_frame().rename({0:'Activities'}, axis=1)

# %% [markdown]
# # Ejercicio:
# A partir de los datos de las posibles rutas entre el nodo inicial y final del proyecto, y de los datos de duraciones y costes siguientes:

# %%
rutas

# %%
datos_enunciado

# %% [markdown]
# Se pide:

# %%
pregunta_1 = "1. Determinar la matriz de caminos del proyecto  "
pregunta_2 = "2. Utilizar la matriz de caminos del proyecto para calcular su duración  "
pregunta_3 = "3. Dibujar los grafos Pert y Roy, y determinar la tabla de prelaciones transitivas (ancestros)  "
pregunta_4 = "4. Calcular utilizando el método de Zaderenko los tiempos tempranos y tardíos  "
pregunta_5 = "5. Calcular las holguras totales de las actividades  "
pregunta_6 = "6. Calcular los costes unitarios de reducción  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))
display(Markdown(pregunta_5))
display(Markdown(pregunta_6))

# %% [markdown]
# Realizar los cálculos utilizando números enteros con redondeo siempre hacia infinito positivo.

# %% [markdown]
# # Resolución

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
proyecto.display_rutas("pert")
_pm = proyecto.path_matrix(dummies=False)
display(Markdown(f"**Respuesta:** la matriz de caminos tiene una fila por ruta ({len(_pm.index)}) y una columna por actividad "
                 f"({len(_pm.columns)}); el elemento vale 1 si la actividad pertenece a la ruta y 0 si no. Se obtiene "
                 f"directamente de la lista de rutas del enunciado."))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# En el apartado anterior, hemos determinado la matriz de caminos. Necesitamos ahora calcular las duraciones medias de las actividades.

# %%
#| tbl-cap: "Cálculo de la duración media de las actividades"
#| label: tbl-calculos Interesantes
datos_proyecto = datos_enunciado.copy()
datos_proyecto['Dmedia'] = datos_proyecto.eval("ceil((a + 4*m +b )/6)")
display(Markdown("Duración media PERT redondeada hacia arriba: $D_{media} = \\lceil (a + 4m + b)/6 \\rceil$:"))
display(pd.DataFrame({
    'a': datos_proyecto['a'], 'm': datos_proyecto['m'], 'b': datos_proyecto['b'],
    'Dmedia = ⌈(a + 4m + b)/6⌉': [f"⌈({r.a} + 4×{r.m} + {r.b})/6⌉ = ⌈{(r.a + 4 * r.m + r.b) / 6:.2f}⌉ = {r.Dmedia:g}"
                                  for r in datos_proyecto.itertuples()]}, index=datos_proyecto.index))

# %% [markdown]
# Con estos valores, podemos multiplicar la matriz de rutas por la columna de duraciones de las actividades para obtener la duración de cada una de las rutas. 

# %%
duraciones = datos_proyecto['Dmedia']#.to_dict()

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
_crit = [r for r in _dur_rutas.index if _dur_rutas[r] == _Tmax]
_cal = proyecto.calendar(duraciones)
display(Markdown(
    f"La duración del proyecto es la máxima de los caminos: máx({', '.join(format(v, 'g') for v in _dur_rutas)}) = "
    f"**{_Tmax:g}** periodos, que corresponde a "
    + lista_y([f"{r} ({' – '.join(ordena_ruta(list(_pm.columns[_pm.loc[r] == 1]), _cal))} = "
               f"{' + '.join(format(duraciones[a], 'g') for a in ordena_ruta(list(_pm.columns[_pm.loc[r] == 1]), _cal))})"
               for r in _crit])
    + f" (camino crítico).\n\n**Respuesta:** la duración del proyecto es **{_Tmax:g}** periodos."))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# ### Grafo Pert con indicación de los números de los nodos

# %%
#| fig-cap: "Grafo Pert"
#| label: fig-PertSinTiempos
proyecto.pert(durations=False, ranksep=0.6)

# %% [markdown]
# ### Grafo Roy

# %%
#| fig-cap: "Grafo Roy"
#| label: fig-RoySinTiempos
proyecto.roy(durations=False, ranksep=0.6)

# %% [markdown]
# ### Cuadro de prelaciones transitivas (ancestros)

# %% [markdown]
# El cuadro de ancestros, o de prelaciones transitivas, recoge todos las actividades previas a una actividad dada. Puede ser de utilidad para comparar dos grafos PERT o Roy y comprobar si se trata de distintas representaciones de un mismo proyecto.

# %%
#| fig-cap: "Grafo PERT y cuadro de prelaciones transitivas (ancestros)"
#| label: fig-prelacionesTransitivas
proyecto.display_ancestros('pert')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_tikz(duraciones)

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
proyecto.calculate_pert(durations=duraciones)['nodes']

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
#| tbl-cap: "Valores de la holgura total de las actividades"
#| label: tbl-Holguras
display(proyecto.calculate_pert(duraciones)['activities'])
_nodos = proyecto.calculate_pert(duraciones)['nodes']
_filas = {}
for _i, _j, _att in proyecto.pert_graph.edges(data=True):
    _a = _att['activity']
    _d = 0 if _a.startswith('@') else duraciones[_a]
    _filas[_a] = {'nodos': f"{_i} → {_j}", 'temprano inicio t_i': _nodos.loc[_i, 'early'], 'tardío fin t*_j': _nodos.loc[_j, 'late'],
                  'duración': _d, 'H_total = t*_j − t_i − d': f"{_nodos.loc[_j, 'late']:g} − {_nodos.loc[_i, 'early']:g} − {_d:g} = "
                  f"{_nodos.loc[_j, 'late'] - _nodos.loc[_i, 'early'] - _d:g}"}
_det = pd.DataFrame(_filas).T.loc[ordena_ruta(list(duraciones.index), _cal)]
_h = {a: float(str(_det.loc[a, 'H_total = t*_j − t_i − d']).split('=')[-1]) for a in _det.index}
display(Markdown("La holgura total de una actividad que va del nodo i al nodo j es $H_T = t^*_j - t_i - d$ (tiempo tardío del nodo "
                 "final menos tiempo temprano del nodo inicial menos su duración):"))
display(limpia_tabla(_det.infer_objects()))
display(Markdown(f"**Respuesta:** las actividades con holgura total nula son las críticas: "
                 f"{lista_y([a for a, h in _h.items() if h == 0])}; el resto tiene holgura "
                 + lista_y([f"{a} = {h:g}" for a, h in _h.items() if h != 0]) + " periodos."))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con tiempos, y rutas críticas"
#| label: fig-PertConTiempos
proyecto.display_pert_pasos(duraciones)

# %%

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con tiempos, y rutas críticas"
#| label: fig-RoyConTiempos
proyecto.display_roy_pasos(duraciones)

# %% [markdown]
# ## Apartado 6

# %%
display(Markdown(pregunta_6))

# %% [markdown]
# Utilizamos la fómula $\mathrm{CUR=\frac{C_{tope}-C_{media}}{D_{media}-D_{tope}}}$

# %%
datos_proyecto['CUR'] = datos_proyecto.eval("ceil((Ctope-Cmedia)/(Dmedia-Dtope))")
display(Markdown("redondeando hacia arriba el cociente, como pide el enunciado:"))
display(pd.DataFrame({
    'Ctope': datos_proyecto['Ctope'], 'Cmedia': datos_proyecto['Cmedia'], 'Dmedia': datos_proyecto['Dmedia'],
    'Dtope': datos_proyecto['Dtope'],
    'CUR = ⌈(Ctope − Cmedia)/(Dmedia − Dtope)⌉': [
        f"⌈({r.Ctope} − {r.Cmedia})/({r.Dmedia:g} − {r.Dtope})⌉ = ⌈{(r.Ctope - r.Cmedia) / (r.Dmedia - r.Dtope):.2f}⌉ = {r.CUR + 0:g}"
        for r in datos_proyecto.itertuples()]}, index=datos_proyecto.index))
_anom = list(datos_proyecto.index[datos_proyecto['Ctope'] < datos_proyecto['Cmedia']])
if _anom:
    display(Markdown(f"*Nota*: en {lista_y(_anom)} el coste a duración tope es menor que el coste a duración media, lo que no es "
                     f"habitual (acortar la actividad la abarataría); con el redondeo hacia arriba su CUR resulta "
                     f"{lista_y([format(datos_proyecto.loc[a, 'CUR'] + 0, 'g') for a in _anom])}."))
