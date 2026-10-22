# ---
# title: Ejercicio 03
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
#     warning: false
# message: false
# jupyter: python3
# ---

# %%
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, colorea_minimo_fila, plot_cargas

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# ## Enunciado
# Considere la información del proyecto cuya información aparece en la tabla adjunta:

# %%
periodo_probabilidad_b = 19
periodo_probabilidad_a = 17
limite_maximo_recursos = [4]*3 + [8]
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto entre {periodo_probabilidad_a} y {periodo_probabilidad_b} semanas?  "
pregunta_2 = "2. Como la probabilidad antes calculada le parece baja, reduzca, con mínimo coste la duración media del proyecto a 18 semanas. ¿Cuál será ahora la probabilidad de terminar en 18 semanas? ¿Por qué?  "
pregunta_3 = "3. Partiendo del proyecto en su situación inicial del apartado 1 ¿Se encuentran nivelados los recursos consumidos? De no ser así proceda a su nivelación.  "
pregunta_4 = "4. Partiendo del proyecto en su situación inicial del apartado 1, modifique la planificación del proyecto para poder ejecutarlo con tan sólo 4 recursos como máximo durante los primeros tres periodos, y 8 recursos durante el resto del proyecto.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
# Alargamos el ancho de las celdas
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import pandas as pd
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, varianza,  cur,  recursos, duracion_tope
     ('A',       '---  ',      2,        1,     -100,     2,         1        ),
     ('B',       '---  ',      4,        1,       30,     2,         2        ),
     ('C',       'B    ',      1,        1,   np.inf,     2,         1        ),
     ('D',       'A    ',      3,        1,       50,     2,         1        ),
     ('E',       'B,D  ',      7,        1,       20,     2,         4        ),
     ('F',       'B,C,D ',     4,        1,       10,     1,         2        ),
     ('G',       '---  ',      1,        2,      100,     4,         1        ),
     ('H',       'G    ',     11,        2,       10,     2,         8        ),
     ('I',       'E,F,H',      2,        1,       50,     2,         1        ),
     ('J',       'E,F,H',      8,        1,        5,     2,         7        ),
] ,columns=['activity', 'predecessor', 'duration', 'variance', 'ucr', 'resources', 'cap_duration']).set_index('activity')

datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')

# %% [markdown]
# ### Cuadros de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos
#
# Utilizando el cuadro de prelaciones expandido podemos dibujar el grafo Pert correspondiente al proyecto.

# %%
#| fig-cap: "Grafo Pert con indicación con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto = prelaciones.create_project()
proyecto.display_pert_construccion(tabla=prelaciones)

# %% [markdown]
# Alternativamente podríamos haber representado su grafo Roy.

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-roy
proyecto.roy(ranksep=1.2, nodesep=1.2)

# %% [markdown]
# ### Matriz de Zaderenko
# A continuación podemos determinar la duración del proyecto calculando los tiempos tempranos y tardíos de su grafo Pert. Se hace en este caso mediante el algoritmo de Zaderenko:

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duration'])

# %% [markdown]
# Los tiempos tempranos y tardíos obtenidos para cada nodo son:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
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
# ### Varianza de la duración del proyecto
# Para determinar la varianza del proyecto es necesario conocer la varianza de cada una de las rutas del camino crítico. Es necesario, por tanto, conocer cuales son las actividades con holgura total cero.

# %% [markdown]
# #### Holgura total de las actividades
# Del cálculo de las holguras totales obtenemos:

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
resultados_pert['activities']

# %% [markdown]
# #### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas, cuyas actividades se listan en orden alfabético:

# %%
camino_critico = proyecto.display_critical_path(durations=datos_proyecto['duration'])

# %% [markdown]
# #### Grafo PERT con indicación del camino crítico

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Alternativamente, se muestra el grafo roy correspondiente con indicación del camino crítico.

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# #### Varianza de cada rama
#
# Se calcula entonces la varianza de cada una de las rutas del camino crítico como suma de las varianzas de las actividades que lo componen.

# %%
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_caminos

# %%
display(Markdown(f"""La varianza de la duración del proyecto es la máxima varianza de las rutas críticas. Para este proyecto es {round(desviacion_proyecto**2,2)}.  
Habitualmente será útil calcular su raiz cuadrada, la desviación típica, para ser utilizada como parámetro de la ley de distribución normal.  """))
display(Markdown(f"La desviación típica es {round(desviacion_proyecto,2)}"""))


# %% [markdown]
# #### Probabilidad de terminar en el periodo solicitado

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)
probabilidad_de_terminar_antes_de_a = norm.cdf(periodo_probabilidad_a, loc=duracion_media_proyecto, scale=desviacion_proyecto)


probabilidad_de_terminar_entre_a_y_b = round(float(probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a), 2)

Markdown(f"Para un proyecto con duración media {duracion_media_proyecto} y desviación típica {round(desviacion_proyecto, 2)} el valor de la probabilidad de terminar en el periodo comprendido entre {periodo_probabilidad_a} y {periodo_probabilidad_b} periodos es {round(probabilidad_de_terminar_entre_a_y_b * 100, 2)} por ciento.")

# %% [markdown]
# ## Apartado 2
#

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# ### Reducción de la duración del proyecto con mínimo coste

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                min_durations=datos_proyecto['cap_duration'],
                costs=datos_proyecto['ucr'],
                reduction=2,
                report=False,
                )

# %%
#| tbl-cap: "Cálculo de la reducción de la duración del proyecto mediante el algoritmo de Ackoff Sasieni"
#| label: tbl-ackoff
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %% [markdown]
# Los desplazamientos indicados al aplicar el algoritmo de reducción con mínimo coste son:

# %%
for iteracion, actividades in best_option.items():
    display(Markdown(f"Iteración {iteracion}: " + ", ".join(actividades)))

# %% [markdown]
# ### Probabilidad de terminar antes de 18 periodos
# Al ser 18 el valor de la media de la duración, la probabilidad pedida es del 50%.

# %% [markdown]
# ### Probabilidad de terminar entre 19 y 17 semanas

# %% [markdown]
# #### Duración media del proyecto

# %%
duracion_media_proyecto = proyecto.duration(duration)

# %%
Markdown(f"Nótese como al haber reducido la duración del proyecto, la duración media a utilizar en el cálculo de la probabilidad es la nueva duración recortada de {duracion_media_proyecto} periodos.")

# %% [markdown]
# #### Varianza de la duración del proyecto

# %% [markdown]
# Al proyecto le ha surgido una nueva rama. Debemos calcular la varianza de esta nueva rama y actualizar el valor de la varianza del proyecto si fuera menor que la de esta rama.

# %%
# duration ha sido calculada en el Ackoff Sasieni
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=duration,
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas2
varianza_caminos

# %%
display(Markdown(f"""La varianza de la duración del proyecto es la máxima varianza de las rutas críticas. Para este proyecto es {round(desviacion_proyecto**2,2)}.  
Habitualmente será útil calcular su raiz cuadrada, la desviación típica, para ser utilizada como parámetro de la ley de distribución normal.  """))
display(Markdown(f"La desviación típica es {round(desviacion_proyecto,2)}"""))


# %% [markdown]
# #### Probabilidad de terminar en el periodo solicitado

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)
probabilidad_de_terminar_antes_de_a = norm.cdf(periodo_probabilidad_a, loc=duracion_media_proyecto, scale=desviacion_proyecto)

probabilidad_de_terminar_entre_a_y_b = probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a

Markdown(f"Para un proyecto con duración media {duracion_media_proyecto} y desviación típica {round(desviacion_proyecto, 2)} el valor de la probabilidad de terminar en el periodo comprendido entre {periodo_probabilidad_a} y {periodo_probabilidad_b} periodos es {round(probabilidad_de_terminar_entre_a_y_b * 100, 2)} por ciento.")

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# ### Análisis de la situación inicial

# %% [markdown]
# Para responder este apartado es necesario representar la demanda inicial de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto antes de la nivelación"
#| label: fig-ganttPrenivelacion
proyecto_nivelacion = proyecto.copy()
datos_proyecto=datos_enunciado.copy()
gantt_df, dibujo = proyecto_nivelacion.gantt_cargas(datos_proyecto, 'duration', 'resources', tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original"
#| label: fig-preNivelacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None)

# %% [markdown]
# La demanda de recursos actual no se encuentra nivelada. Se procede a aplicar el algoritmo de nivelación.

# %% [markdown]
# ### Nivelación

# %% [markdown]
# La siguiente tabla indica el valor de la suma de los cuadrados del consumo de los recursos para cada unos de los posibles desplazamientos, en el orden de desplazamiento de las actividades utilizado. El orden seguido ha sido el de las actividades no críticas en orden creciente de periodo de finalización. Se señala en color amarillo el desplazamiento que en cada caso minimiza la suma de cuadrados.

# %%
proyecto_nivelacion = proyecto.copy()
datos_proyecto = datos_enunciado.copy()

desplazamientos, cuadrados, my_data, gantt_df, dibujo = proyecto_nivelacion.nivelar(
                            data=datos_proyecto,
                            duration_label='duration',
                            resource_label='resources',
                            
                           )

# %%
#| tbl-cap: "Suma de los cuadrados de las cargas para cada uno de los desplazamientos posibles con indicación del mínimo en cada actividad"
#| label: tbl-CuadradosNivelacion
cuadrados

# %% [markdown]
# Atendiendo a los valores de las sumas de cuadrados, los desplazamientos indicados para la nivelación son:

# %%
#| tbl-cap: "Desplazamientos para la nivelación"
#| label: tbl-desplazamientosNivelación
desplazamientos

# %% [markdown]
# El resultado de realizar estos desplazamientos se muestra en el siguiente diagrama de Gantt.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la nivelación"
#| label: fig-ganttPostnivelacion
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras realizar la nivelación"
#| label: fig-postNivelacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# ### Análisis de la situación de partida

# %% [markdown]
# Recuperamos el estado inicial del proyecto, como se ve en el siguiente diagrama de Gantt.

# %%
#| fig-cap: "Diagrama de Gantt previo a la asignación"
#| label: fig-ganttPreAsignacion

proyecto_asignacion = proyecto.copy()
datos_proyecto = datos_enunciado.copy()
gantt_df, dibujo = proyecto_nivelacion.gantt(datos_proyecto, 'duration', 'resources', total='fila', holguras=True, tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original, con indicación del límite máximo de recursos"
#| label: fig-preNivelacionBarras2
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=limite_maximo_recursos)

# %% [markdown]
# ### Algoritmo de asignación de Wiest-Levy

# %% [markdown]
# A continuación se ejecuta el algoritmo de asignación.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la asignación"
#| label: fig-ganttPreAsignación
datos_proyecto_tras_asignar, gantt_df, dibujo = proyecto_asignacion.asignar(data=datos_proyecto,
                                              duration_label='duration',
                                              resource_label='resources',
                                              maximo = limite_maximo_recursos,
                                              report = False,
                                              params = dict(
                                                  period_width=1
                                                  )
                                              )
pinta_tikz(dibujo)

# %% [markdown]
# Tras esta actuación el proyecto ya es compatible con la restricción impuesta en los recursos.

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras el algoritmo de asignación, con indicación del límite máximo de recursos"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=limite_maximo_recursos)

# %%

# %%
