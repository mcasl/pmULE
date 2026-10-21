# ---
# title: Ejercicio 22
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

# %% [markdown]
# # Enunciado
#
# Considere el proyecto cuya información se adjunta.

# %%
pregunta_1 = "1. Dibuje los grafos Pert y Roy.  Determine el camino crítico.  "
pregunta_2 = "2. Reduzca la duración del proyecto dos unidades de tiempo con el menor incremento de coste posible.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

import pandas as pd
datos_enunciado = pd.DataFrame([
#  actividad, precedentes, b, m, a, coste_duracion_pert, coste_duracion_tope, duracion_tope
     ('A', '---  ',   6,  2.5, 2, 1600, 1800, 2),
     ('B', '---  ',   6,    6, 6, 2800, 2820, 1),
     ('C', '---  ',   1,    1, 1,  600,  600, 1),
     ('D', '---  ',   3,    3, 3,  600,  607, 2),
     ('E', 'A,B  ',  12,    5, 4, 1600, 2200, 2),
     ('F', 'A,B  ',  12,  2.5, 2, 2000, 2200, 1),
     ('G', 'B,C  ',   6, 1.25, 1,  600,  700, 1),
     ('H', 'B,C  ',  20,   15, 10, 3500, 3620, 3),
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
tabla = datos_enunciado.copy()
tabla.columns=['precedentes', 'b', 'm', 'a', 
                                        'coste duracion pert', 'coste duracion tope', 'duracion tope']
tabla

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Matriz de prelaciones

# %% [markdown]
# Empezamos determinando la matriz de prelaciones. Nos sirve para construir el grafo, si leemos las columnas, y para comprobar que no hemos cometido errores, si leemos las columnas.

# %%
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_enunciado, activity='actividad', predecessor='precedentes')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# {{< pagebreak >}}
#

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
proyecto = prelaciones.create_project()

# %%
proyecto.pert()

# %% [markdown]
# ### Grafo Roy

# %%
proyecto = prelaciones.create_project()

# %%
proyecto.roy()

# %% [markdown]
# ### Cálculo de variables de interés

# %% [markdown]
# A partir de los datos del enunciado podemos calcular la duración media $D=\frac{a+4m+b}{6}$, y el coste unitario de reducción $\frac{\Delta Coste}{\Delta duración}$ que utilizaremos para contestar las preguntas realizadas.

# %%
#| tbl-cap: "Cálculo de la duración media y el coste unitario de reducción"
#| label: tbl-datosInteresantes
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duracion'] = datos_proyecto.eval('(b + 4*m +a)/6')
datos_proyecto['cur']      = datos_proyecto.eval('(coste_duracion_tope - coste_duracion_pert)/(duracion - duracion_tope)')
datos_proyecto[['b', 'm', 'a', 'duracion', 'cur']]

# %% [markdown]
# ### Cuadro de ancestros

# %% [markdown]
# {{< pagebreak >}}
# El cuadro de prelaciones transitivas (o de ancestros) nos permite comparar distintas representaciones gráficas del mismo proyecto.

# %%
#| tbl-cap: "Determinación de las prelaciones transitivas"
#| label: tbl-PrelacionesTransitivas
prelaciones_distantes = pd.Series({key: ' '.join(sorted(value)) for key,value in proyecto.distant_predecessor().items()})
prelaciones_distantes = pd.DataFrame(prelaciones_distantes)
prelaciones_distantes.columns=['Precedentes']
prelaciones_distantes

# %% [markdown]
# ### Cálculo de tiempos tempranos  y tardíos con el método de Zaderenko

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_pasos(datos_proyecto['duracion'])

# %% [markdown]
# De manera condensada, se muestran los tiempos tempranos y tardíos en la siguiente tabla:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duracion'])
resultados_pert['nodes']

# %% [markdown]
# ### Holguras

# %% [markdown]
# Calculados los tiempos tempranos y tardíos, podemos calcular las holguras totales de las actividades.

# %%
#| tbl-cap: "Valores de la holgura total de las actividades"
#| label: tbl-Holguras
resultados_pert['activities']

# %% [markdown]
# ### Camino crítico

# %% [markdown]
# Las actividades con holgura total igual a cero constituyen el camino crítico. En este caso el camino crítico está formado por las siguientes rutas:

# %%
proyecto.display_critical_path(durations=datos_proyecto['duracion'])


# %% [markdown]
# ### Grafo PERT con indicación de tiempos y camino crítico

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con tiempos, y rutas críticas"
#| label: fig-PertConTiempos
proyecto.display_pert_pasos(datos_proyecto['duracion'])

# %% [markdown]
# ### Grafo Roy con indicación de tiempos y camino crítico

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con tiempos, y rutas críticas"
#| label: fig-RoyConTiempos
proyecto.display_roy_pasos(datos_proyecto['duracion'])

# %% [markdown]
# {{< pagebreak >}}
#

# %% [markdown]
# ## Apartado 2.

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Se aborda la solución de este apartado mediante la técnica de reducción de la duración del proyecto con mínimo coste.

# %%
#| include: false
#| echo: false
ackoff, best_option, durations, periods_available = proyecto.ackoff(durations=datos_proyecto['duracion'],
                         min_durations=datos_proyecto['duracion_tope'],
                         costs=datos_proyecto['cur']
                        )

# %%
for item, value in best_option.items():
    print(f"Actividades a recortar en iteración {item}: {value}")

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos(
    durations=datos_proyecto['duracion'],
    min_durations=datos_proyecto['duracion_tope'],
    costs=datos_proyecto['cur'])

# %% [markdown]
# La suma del sobrecoste de la primera y segunda iteración es de 18.

# %% [markdown]
# #### Proyecto con la duración reducida

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con tiempos, y rutas críticas, para el proyecto con duración reducida"
#| label: fig-PertConTiemposReducido
proyecto.display_pert_pasos(durations)

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con tiempos, y rutas críticas, para el proyecto con duración reducida"
#| label: fig-RoyConTiemposReducido
proyecto.display_roy_pasos(durations)

# %%

# %%
