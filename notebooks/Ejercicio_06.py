# ---
# title: Ejercicio 06
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

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, plot_cargas

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# # Enunciado
# Dado un proyecto con los datos que se muestran en la tabla siguiente, se pide:

# %%
periodo_probabilidad_b = 9
limite_maximo_recursos = [4]
pregunta_1 = "1. Dibujar el diagrama Pert y determinar el camino crítico.  "
pregunta_2 = f"2. Probabilidad de terminar el proyecto antes de {periodo_probabilidad_b} periodos.  "
pregunta_3 = "3. Reducir la duración del proyecto hasta 10 periodos con el mínimo incremento posible de coste.  "
pregunta_4 = f"4. Si el número máximo de recursos durante las 6 primeras semanas es {limite_maximo_recursos[0]}, determinar si será posible realizar el proyecto en el tiempo obtenido en el apartado 1.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
# Alargamos el ancho de las celdas al 100%
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import pandas as pd
import numpy as np

datos_enunciado= pd.DataFrame([
#actividad, precedentes, duracion, desviacion,  cur, recursos, duracion_tope
  (   'A',     '--- ',       3,         2,        5,      1,       1  ),
  (   'B',     '--- ',       2,         1,        5,      1,       1  ),
  (   'C',     'B   ',       4,         2,       25,      1,       2  ),
  (   'D',     'A   ',       2,         1,      -10,      2,       1  ),
  (   'E',     'A   ',       4,         0,        1,      1,       4  ),
  (   'F',     'E   ',       3,         1,        5,      1,       2  ),
  (   'G',     'C,D ',       1,         0,      -20,      2,       1  ),
  (   'H',     'C,D ',       5,         3,        1,      1,       2  ),
  (   'I',     'F,G,H',      1,         0,        1,      1,       1  ),
  (   'J',     'F,G  ',      2,         0,       -1,      1,       2 ),
  (   'K',     'F,G  ',      1,         0,        5,      2,       1  ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation', 'ucr', 'resources', 'cap_duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadros de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.
#

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos
#
# Utilizando el cuadro de prelaciones expandido podemos dibujar el grafo Pert correspondiente al proyecto.

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto_apartado_1 = datos_proyecto.copy() # La necesitaremos en el apartado 4

# %%
proyecto = prelaciones.create_project()

# %%
#| fig-cap: "Grafo Pert con indicación con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto.pert(ranksep=1.5, nodesep=0.7)

# %% [markdown]
# Alternativamente podríamos haber representado su grafo Roy.
#

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-roy
proyecto.roy(ranksep=1.2, nodesep=1)

# %% [markdown]
# ### Matriz de Zaderenko
#
# A continuación podemos determinar la duración del proyecto calculando los tiempos tempranos y tardíos de su grafo Pert. Se hace en este caso mediante el algoritmo de Zaderenko:

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_pasos(datos_proyecto['duration'])

# %% [markdown]
# Los tiempos tempranos y tardíos obtenidos para cada nodo son:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Duración media del proyecto

# %%
duracion_media_proyecto = proyecto.duration(durations=datos_proyecto['duration'])
Markdown(f"De acuerdo a la duración obtenida para el nodo final del proyecto, la duración media del proyecto es {duracion_media_proyecto} periodos.")

# %% [markdown]
# ### Calendario del proyecto
# Las fechas de inicio y fin más tempranas y tardías de cada actividad son:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario

# %% [markdown]
# #### Holgura total de las actividades
# Del cálculo de las holguras totales obtenemos:

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
resultados_pert['activities'][['H_total']]

# %% [markdown]
# #### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas, cuyas actividades se listan en orden alfabético:

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación con indicación del camino crítico"
#| label: fig-pertcritico
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con indicación con indicación del camino crítico"
#| label: fig-Roycritico
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# Las rutas del camino crítico son:

# %%
proyecto.display_critical_path(durations=datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duracion_media_proyecto = proyecto.duration(durations=datos_proyecto['duration'])

# %%
Markdown(f"La duración media del proyecto la conocemos del primer apartado. Tiene valor {duracion_media_proyecto} periodos.")

# %% [markdown]
# Para determinar la desviación típica debemos obtener en cada rama del camino crítico la suma de las varianzas, y quedarnos con la mayor.

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, caminos_varianza = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas2
caminos_varianza

# %%
display(Markdown(f"""La varianza de la duración del proyecto es la máxima varianza de las rutas críticas. Para este proyecto es {round(desviacion_proyecto**2,2)}.  
Habitualmente será útil calcular su raiz cuadrada, la desviación típica, para ser utilizada como parámetro de la ley de distribución normal.  """))
display(Markdown(f"La desviación típica es {round(desviacion_proyecto,2)}"""))

# %% [markdown]
# Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad:

# %%
import scipy
from scipy.stats import norm

probabilidad_antes_b = norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)

Markdown(f"Para un proyecto con duración media {duracion_media_proyecto} y desviación típica {round(desviacion_proyecto, 2)} el valor de la probabilidad de terminar antes del periodo {periodo_probabilidad_b} es {round(probabilidad_antes_b * 100, 2)} por ciento.")

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                min_durations=datos_proyecto['cap_duration'],
                costs=datos_proyecto['ucr'],
                reduction=2)

# %%
#| tbl-cap: "Cálculo de la reducción de la duración del proyecto mediante el algoritmo de Ackoff Sasieni"
#| label: tbl-ackoff
proyecto.display_ackoff_pasos(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %% [markdown]
# Los desplazamientos indicados al aplicar el algoritmo de reducción con mínimo coste son:

# %%
for iteracion, actividades in best_option.items():
    display(Markdown(f"- Iteración {iteracion}: " + ", ".join(actividades) + " <br>"))

# %%
Markdown(f"El coste de la reducción ha sido de {2 * datos_proyecto.loc['A', 'ucr'] + 2 * datos_proyecto.loc['H', 'ucr']} unidades monetarias.")

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# ### Situación inicial de partida

# %% [markdown]
# Representamos la demanda de recursos por periodo utilizando un diagrama de Gantt.

# %%
#| fig-cap: "Diagrama de Gantt con indicación de las cargas por periodo."
#| label: fig-ganttPartida
proyecto_asignacion = proyecto.copy()
gantt_df, dibujo = proyecto_asignacion.gantt(datos_proyecto, 'duration', 'resources', total='fila', holguras = True, tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original, con indicación del límite máximo de recursos"
#| label: fig-preNivelacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=limite_maximo_recursos)

# %% [markdown]
# Se concluye por tanto que la restricción de recursos impuesta no implica ningún problema.

# %%
