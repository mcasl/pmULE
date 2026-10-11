# ---
# title: Ejercicio 20
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

from pmule import PredecessorTable, ProjectGraph, pinta_tikz

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# # Enunciado
#
# Considere el proyecto cuya información se adjunta.  

# %%
pregunta_1 = "1.  Dibuje los grafos Roy y Pert. Determine el camino crítico.  "
pregunta_2 = "2.  Reduzca la duración del proyecto dos unidades de tiempo con el menor incremento de coste posible.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

import pandas as pd
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, b, m, a, coste_duracion_pert, coste_duracion_tope, duracion_tope
     ('A', '---  ',   6,  2.5, 2, 1600, 1800, 2),
     ('B', '---  ',   6,    6, 6, 2800, 2820, 1),
     ('C', '---  ',   1,    1, 1,  600,  600, 1),
     ('D', '---  ',   3,    3, 3,  600,  607, 2),
     ('E', 'A,B  ',  12,    5, 4, 1600, 2200, 2),
     ('F', 'A,B  ',  12,  2.5, 2, 2000, 2200, 1),
     ('G', 'B,C,P',   6, 1.25, 1,  600,  700, 1),
     ('H', 'B,C,P ',  20,   15, 10, 3500, 3620, 3),
     ('I', 'D    ',  28,   18, 8, 3500, 3503, 17),
     ('J', 'E,M  ',   4,    4, 4, 1500, 1500, 2),
     ('K', 'F    ',   7,    7, 7, 2000, 2000, 3),
     ('L', 'J,K,G',   1,    1, 1, 1000, 1000, 1),
     ('M', 'F    ',   3,    3, 3,  500,  500, 1),
     ('N', 'J,K,G',   1,    1, 1, 1000, 1000, 1),
     ('P', 'D    ',   3,    3, 3, 1000, 1150, 1),

] ,columns=['actividad', 'precedentes', 'b', 'm', 'a',
            'coste_duracion_pert', 'coste_duracion_tope', 'duracion_tope']).set_index('actividad')

datos_enunciado

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadro de prelaciones
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto=datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='actividad', predecessor='precedentes')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo Roy

# %%
#| fig-cap: "Grafo Roy"
#| label: fig-RoySinTiempos
proyecto = prelaciones.create_project()
proyecto.roy(ranksep=1.0)

# %% [markdown]
# ### Grafo Pert

# %%
#| fig-cap: "Grafo Pert"
#| label: fig-PertSinTiempos
proyecto.pert(ranksep=1.0)

# %% [markdown]
# ### Precálculo de variables de interés

# %% [markdown]
# A partir de los datos del enunciado podemos calcular la duración media, y el coste unitario de reducción que utilizaremos para contestar las preguntas realizadas.

# %%
#| tbl-cap: "Cálculo de la duración media y coste unitario de reducción"
#| label: tbl-precalculos
datos_proyecto['duracion'] = datos_proyecto.eval('(b + 4*m +a)/6')
datos_proyecto['cur'] = datos_proyecto.eval('(coste_duracion_tope - coste_duracion_pert)/(duracion - duracion_tope)')
datos_proyecto[['b', 'm', 'a', 'duracion', 'cur']]

# %% [markdown]
# ### Matriz de Zaderenko

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.zaderenko(datos_proyecto['duracion'])

# %% [markdown]
# Los tiempos tempranos y tardíos han resultado ser

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(datos_proyecto['duracion'])
resultados_pert['nodes']

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ### Holguras

# %%
#| tbl-cap: "Valores de la holgura total de las actividades"
#| label: tbl-Holguras
resultados_pert['activities']

# %% [markdown]
# ### Camino crítico

# %%
proyecto.display_critical_path(datos_proyecto['duracion'])

# %% [markdown]
# ### Grafo PERT con indicación de tiempos y rutas del camino crítico

# %%
#| fig-cap: "Grafo Pert con tiempos, y rutas críticas"
#| label: fig-PertConTiempos
proyecto.pert(durations=datos_proyecto['duracion'])

# %% [markdown]
# ### Grafo Roy con indicación de tiempos y rutas del camino crítico

# %%
#| fig-cap: "Grafo Roy con tiempos, y rutas críticas"
#| label: fig-RoyConTiempos
proyecto.roy(durations=datos_proyecto['duracion'])

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Apartado 2.

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Se aborda la solución de este apartado mediante la técnica de reducción de la duración del proyecto con mínimo coste.

# %%
#| include: false
#| echo: false
ackoff, best_option, duration, available = proyecto.ackoff( durations=datos_proyecto['duracion'],
                                                            min_durations=datos_proyecto['duracion_tope'],
                                                            costs=datos_proyecto['cur'],
                                                            reduction=2)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
ackoff

# %% [markdown]
# Las actividades a recortar en cada iteración son:

# %%
for item, value in best_option.items():
    print(f"Actividades a recortar en iteración {item}: {value}")

# %% [markdown]
# El grafo Roy con indicación de tiempos y rutas críticas tras la reducción de la duración es el siguiente:

# %%
#| fig-cap: "Grafo Roy con indicación de tiempos tras la reducción del proyecto"
#| label: fig-RoyReduccionTiempos
proyecto.roy(durations=duration)

# %%
