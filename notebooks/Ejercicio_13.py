# ---
# title: Ejercicio 13
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

# %% [markdown]
# # Enunciado
# Considere la información del proyecto cuya información aparece en la tabla adjunta. Considere en cada apartado que el proyecto se encuentra en su situación inalterada mostrada en los datos del proyecto.

# %%
periodo_a_probabilidad = 17
periodo_b_probabilidad = 19
duracion_objetivo = 18
limite_maximo_recursos = 6
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} semanas?  "
pregunta_2 = f"2. Reduzca, con mínimo incremento de coste la duración media del proyecto a {duracion_objetivo} semanas. ¿Cuál será ahora la probabilidad de terminar en {duracion_objetivo} semanas? ¿Por qué?  "
pregunta_3 = f"3. ¿Se encuentran nivelados los recursos consumidos por el proyecto? De no ser así proceda a su nivelación.  "
pregunta_4 = f"4. Si su empresa sólo puede destinar a {limite_maximo_recursos} personas al proyecto, ¿se encuentran los recursos de mano de obra correctamente asignados? Reprográmelo si fuera necesario de modo que se cumpla esta restricción.  "

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

import pandas as pd
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, varianza,  cur,  recursos, duracion_tope
     ('A',       '---  ',      2,        1,     -100,     2,         1        ),
     ('B',       '---  ',      4,        1,       30,     4,         2        ),
     ('C',       'B    ',      1,        1,   np.inf,     2,         1        ),
     ('D',       'A    ',      3,        1,       50,     2,         1        ),
     ('E',       'B,D  ',      7,        1,       20,     2,         4        ),
     ('F',       'C    ',      4,        1,       10,     2,         2        ),
     ('G',       '---  ',      1,        2,      100,     4,         1        ),
     ('H',       'G    ',     11,        2,       10,     2,         8        ),
     ('I',       'E,F,H',      2,        1,       50,     2,         1        ),
     ('J',       'E,F,H',      8,        1,        5,     2,         7        ),
] ,columns=['activity', 'predecessor', 'duration', 'variance', 'ucr', 'resources', 'crash_duration']).set_index('activity')

datos_enunciado

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
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %%
proyecto = prelaciones.create_project()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto.display_pert_construccion(tabla=prelaciones)

# %% [markdown]
# ### Matriz de Zaderenko

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duration'])

# %% [markdown]
# Los tiempos tempranos y tardios han resultado ser:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Duración media del proyecto

# %% [markdown]
# La duración media del proyecto es:

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duration']))
Markdown(f"La información del nodo final del proyecto nos informa de la duración media del proyecto: {duracion_media_proyecto} periodos")

# %% [markdown]
# ### Holgura total de las actividades
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
resultados_pert['activities']

# %% [markdown]
# ### Camino crítico
#
# El camino crítico consta de las siguientes rutas:

# %%
proyecto.display_critical_path(durations=datos_proyecto['duration'])

# %% [markdown]
# ### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto, con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ### Grafo PERT con indicación de tiempos y rutas del camino crítico de acuerdo a la planificación inicial
# A continuación se muestra el grafo Pert del proyecto, con indicación de las actividades críticas:

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación del camino crítico"
#| label: fig-pertrutasCriticas
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# ### Caracterización de la duración del proyecto mediante una distribución Normal
# A fin de calcular la probabilidad pedida, es necesario caracterizar la distribución Normal de la duración del proyecto. Esto consiste en determinar los valores de sus parámetros: media y desviación típica de la duración del proyecto.

# %%
duracion_media_proyecto = int(proyecto.duration(datos_proyecto['duration']))
Markdown(f"La duración media del proyecto ya la tenemos calculada: {duracion_media_proyecto} periodos")

# %% [markdown]
# Para determinar la desviación típica debemos obtener en cada rama del camino crítico la suma de las varianzas, y quedarnos con la mayor.

# %%
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_rutas

# %%
Markdown(f"El proyecto sigue una distribución normal con media {duracion_media_proyecto} periodos y desviación típica {round(desviacion_proyecto, 2)}.")

# %% [markdown]
# ### Probabilidad de terminar entre 17 y 19 periodos

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(norm.cdf(periodo_b_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)
probabilidad_de_terminar_antes_de_a = round(norm.cdf(periodo_a_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)

probabilidad_de_terminar_entre_a_y_b = probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, entre=(periodo_a_probabilidad, periodo_b_probabilidad))

# %%
Markdown(f"""Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad de terminar entre los periodos {periodo_a_probabilidad} y {periodo_b_probabilidad}.  
- La probabilidad de terminar antes de {periodo_a_probabilidad} periodos es  {probabilidad_de_terminar_antes_de_a}  
- La probabilidad de terminar antes de {periodo_b_probabilidad} periodos es {probabilidad_de_terminar_antes_de_b}  

Por tanto, a partir de la diferencia de estos dos valores tenemos la probabilidad de terminar dentro del mencionado intervalo: {int(probabilidad_de_terminar_entre_a_y_b*100)} por ciento.
""")

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# ### Reducción a la duración objetivo con mínimo incremento de coste

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                min_durations=datos_proyecto['crash_duration'],
                costs=datos_proyecto['ucr'],
                reduction=2)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['crash_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %% [markdown]
# ### Probabilidad de terminar antes de 18 periodos
# Al ser 18 el valor de la media de la duración, la probabilidad pedida es del 50%.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# ### Análisis de la situación inicial

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
proyecto_nivelacion = proyecto.copy()
duration_resources = datos_enunciado[['duration','resources']].copy()
gantt_df, dibujo = proyecto_nivelacion.gantt(data=duration_resources,
                          duration_label='duration',
                          resource_label='resources',
                          holguras=True,
                          total='fila', 
                          tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su planificación inicial"
#| label: fig-preNivelaciónBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None)

# %% [markdown]
# La demanda de recursos actual no se encuentra nivelada. Se procede a aplicar el algoritmo de nivelación.

# %% [markdown]
# ### Nivelación de los recursos

# %%
proyecto_nivelacion.display_nivelar_pasos_tikz(datos_proyecto, duration_label='duration', resource_label='resources')
desplazamientos, cuadrados, my_data, gantt_df, dibujo = proyecto_nivelacion.nivelar(data=datos_proyecto,
                           duration_label='duration',
                           resource_label='resources',
                           report=False
                           )

# %% [markdown]
# Se procede a valorar los posibles desplazamientos de cada actividad no crítica, considerando estas en el orden correspondiente a su fecha de finalización, de más pronto a más tarde (a igualdad, primero la de mayor holgura). El visor anterior muestra el valor de la suma de cuadrados de las cargas para cada actividad, en filas según el mencionado orden, y posible desplazamiento, en columnas; el desplazamiento que minimiza la suma de cuadrados se señala en verde y queda fijado antes de estudiar la siguiente actividad.

# %%
#| include: false
cuadrados   # la tabla de sumas de cuadrados ya se muestra en el visor anterior

# %% [markdown]
# El resultado de estos desplazamientos se muestra en la siguiente figura.

# %%
#| fig-cap: "Resultado de aplicar el algoritmo de nivelación al proyecto"
#| label: fig-GanttTrasNivelacion
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras aplicar el algoritmo de nivelación"
#| label: fig-postNivelaciónBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
proyecto_asignacion = proyecto_nivelacion.copy()
duration_resources = datos_enunciado[['duration','resources']].copy()
gantt_df, dibujo = proyecto_asignacion.gantt(duration_resources, 'duration', 'resources', total='fila', holguras=True, tikz=True)
pinta_tikz(dibujo)

# %% [markdown]
# La fila "Total" obtenida, que podemos representar gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo y el valor límite establecido (línea horizontal discontinua).

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)

# %%
Markdown(f"El proyecto requiere en algunos periodos más de {limite_maximo_recursos} personas. Es necesario aplicar el algorimo de asignación, según se muestra en la siguiente figura.")

# %%
duracion_y_recursos = datos_proyecto[['duration', 'resources']]
my_data, gantt_df, dibujo  = proyecto_asignacion.asignar(
                                    data=duracion_y_recursos,
                                    duration_label='duration',
                                    resource_label='resources',
                                    maximo=[limite_maximo_recursos],
                                    report=False
                                    )

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la asignación"
#| label: fig-ganttPostAsignación
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)

# %% [markdown]
# Tras aplicar el algoritmo de asignación el proyecto es compatible con los límites establecidos.

# %%
