# ---
# title: Ejercicio 09
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
#
# Considere el proyecto cuya información se adjunta.

# %%
periodo_probabilidad_a = 10
periodo_probabilidad_b = 12
limite_maximo_recursos = [10]*5 + [20]
pregunta_1 = "1. Dibuje los diagramas Pert y Roy, y determine el camino crítico.  "
pregunta_2 = f"2. ¿Cuál será la probabilidad de terminar el proyecto entre {periodo_probabilidad_a} y {periodo_probabilidad_b} periodos. ¿Y la de terminar en 9 periodos?.  "
pregunta_3 = "3. Como no está conforme con la duración del proyecto que resulta, desea reducir la duración del proyecto en 2 periodos con el mínimo incremento de coste posible. ¿Qué tareas deben ser reducidas?. Razónese.  "
pregunta_4 = f"""4. Debido a que la empresa está realizando otros dos proyectos al mismo tiempo, la disponibilidad de mano de obra no es todo lo grande que sería de desear, por lo que sólo se puede contar con {limite_maximo_recursos[0]} operarios durante las cinco primeras semanas, y con 15 operarios el resto de semanas .  
    - ¿Se encuentra bien nivelado el recurso “mano de obra”?.  ¿Por qué?.
    - ¿Se encuentra bien asignado este recurso?. ¿Por qué?.
    - En caso de que no estuvieran bien asignados, proponer una posible modificación de la planificación de las tareas para una correcta asignación.  """

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

# Alargamos el ancho de las celdas al 100%
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, desviacion, cur,   recursos, duracion_tope
  (   'A',      '---  ',      3,         1,        25,       3,        2    ),
  (   'B',      '---  ',      1,         1,        20,       1,        1    ),
  (   'C',      '---  ',      2,         1,         7,       6,        1    ),
  (   'D',      '---  ',      1,         1,        25,       1,        1    ),
  (   'E',      'A,B  ',      3,         2,        10,       2,        2    ),
  (   'F',      'B,C  ',      5,         1,        25,       5,        3    ),
  (   'G',      'C,D  ',      1,         1,        30,       2,        1    ),
  (   'H',      'E    ',      1,         1,         5,       5,        1    ),
  (   'I',      'G    ',      3,         1,        50,       3,        2    ),
  (   'J',      'F,H,I',      2,         1,         5,      10,        1    ),
  (   'K',      'G    ',      1,         1,        10,       5,        1    ),
] ,columns=['activity', 'predecessor', 'duration', 'standard_deviation', 'ucr', 'resources', 'cap_duration']).set_index('activity')

datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
from pmule import PredecessorTable, ProjectGraph

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
#

# %% [markdown]
# ### Cuadro de prelaciones
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto.pert(ranksep=1.5, nodesep=0.8)

# %% [markdown]
# ### Matriz de Zaderenko

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.zaderenko(durations=datos_proyecto['duration'])

# %% [markdown]
# Algoritmo de Zaderenko paso a paso (avance con los botones):

# %%
proyecto.display_zaderenko_pasos(datos_proyecto['duration'])

# %% [markdown]
# Los tiempos tempranos y tardios han resultado ser:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
resultados_pert['nodes']

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duration']))

# %%
Markdown(f"La información del nodo final del proyecto nos informa de la duración media del proyecto: {duracion_media_proyecto} periodos")

# %% [markdown]
# ### Holguras
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
resultados_pert['activities'][['H_total']]

# %% [markdown]
# ### Camino crítico

# %% [markdown]
# Las rutas del camino crítico son en este caso:

# %%
camino_critico = proyecto.critical_path(durations=datos_proyecto['duration'])
for key, value in camino_critico.items():
    display(Markdown(f"{key}: \t " + ", ".join(value)))

# %% [markdown]
# ### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto, con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario

# %% [markdown]
# ### Grafo PERT con indicación de tiempos y rutas críticas

# %% [markdown]
# A continuación se muestra el grafo Pert del proyecto, con indicación de las actividades críticas:

# %%
#| fig-cap: "Grafo Pert con indicación del camino crítico"
#| label: fig-pertrutasCriticas
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1.5, nodesep=1)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos paso a paso (avance con los botones):

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# A continuación se muestra el grafo Roy del proyecto, con indicación de las actividades críticas:

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación de las actividades críticas"
#| label: fig-RoyCríticas
proyecto.roy(ranksep=1.25, nodesep=1.25, durations=datos_proyecto['duration'])

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso (avance con los botones):

# %%
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duracion_media_proyecto = proyecto.duration(durations=datos_proyecto['duration'])

# %%
Markdown(f"Ya conocemos el valor medio de la duración del proyecto. Es el valor de la duración del proyecto que hemos calculado con duraciones medias de las actividades: {duracion_media_proyecto} periodos")

# %% [markdown]
# Para determinar la desviación típica de la duración del proyecto debemos obtener, en cada rama del camino crítico, la suma de las varianzas, y quedarnos con la mayor.

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_caminos

# %%
Markdown(f"El proyecto sigue una distribución normal con media {duracion_media_proyecto} y desviación típica {round(desviacion_proyecto, 2)}.")

# %%
Markdown(f"Al coincidir el valor de la media con una de las duraciones para las que nos piden el valor de la probabilidad, nos permite responder responder sin realizar cálculo alguno que la probabilidad de terminar antes de {duracion_media_proyecto} es del 50%, pues en la distribución normal la media por simetría deja un 50% de casos a cada uno de sus lados.")

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(norm.cdf(12, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)
probabilidad_de_terminar_antes_de_a = round(norm.cdf(10, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)
probabilidad_de_terminar_entre_a_y_b = round(probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a, 2)

# %%
Markdown(f"""Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad de terminar entre los periodos {periodo_probabilidad_a} y {periodo_probabilidad_b}.  
- La probabilidad de terminar antes de {periodo_probabilidad_a} periodos es  {probabilidad_de_terminar_antes_de_a}  
- La probabilidad de terminar antes de {periodo_probabilidad_b} periodos es {probabilidad_de_terminar_antes_de_b}  

Por tanto, a partir de la diferencia de estos dos valores tenemos la probabilidad de terminar dentro del mencionado intervalo: {int(probabilidad_de_terminar_entre_a_y_b*100)} por ciento.
""")

# %% [markdown]
# ## Apartado 3.
#

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Aplicamos en este apartado el algoritmo de Ackoff y Sasieni para la reducción del proyecto con mínimo incremento de coste:

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                                                           min_durations=datos_proyecto['cap_duration'],
                                                           costs=datos_proyecto['ucr'],
                                                           reduction=2
                                                           )

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %% [markdown]
# Las actividades a recortar en cada iteración son:

# %%
for iteracion, actividades in best_option.items():
    display(Markdown(f"- Iteración {iteracion}: " + ", ".join(actividades) + "<br>"))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
proyecto_asignacion = proyecto.copy()

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
gantt_recursos, dibujo = proyecto_asignacion.gantt(datos_proyecto,
                                'duration',
                                'resources',
                                total='fila',
                                holguras=True,
                                tikz=True)
pinta_tikz(dibujo)

# %% [markdown]
# La fila "Total" obtenida, que podemos representar gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_recursos, limite_maximo_recursos)

# %% [markdown]
# La demanda del recurso no se encuentra nivelada.
# Además, con el máximo disponible de 10 recursos durante los primeros 5 periodos el proyecto no es ejecutable en su estado actual y debe aplicarse el algoritmo de asignación.

# %% [markdown]
# ### Algoritmo de asignación

# %%
duracion_y_recursos = datos_proyecto[['duration', 'resources']]
datos_proyecto_tras_asignar, gantt_df, dibujo  = proyecto_asignacion.asignar(
                                    data=duracion_y_recursos,
                                    duration_label='duration',
                                    resource_label='resources',
                                    maximo=limite_maximo_recursos,
                                    report=False
                                    )

# %%
#| fig-cap: "Diagrama de Gantt del proyecto una vez aplicado el algoritmo de asignación"
#| label: fig-postAsignacion
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)

# %% [markdown]
# En esta situación el proyecto ya es realizable por haberse impuesto el límite de recursos tan sólo en los primeros 5 periodos. La ausencia de límite en el resto de periodos se ha resuelto estableciendo un límite de 100 recursos, que no llega a superarse, para el resto del proyecto.

# %% [markdown]
# En este caso concreto, la duración del proyecto no ha cambiado tras la asignación, ni han aparecido nuevas rutas en el camino crítico:

# %%
proyecto.display_critical_path(durations=duracion_y_recursos['duration'])

# %% [markdown]
# No obstante, sí ha disminuido la holgura total de las actividades.

# %%
