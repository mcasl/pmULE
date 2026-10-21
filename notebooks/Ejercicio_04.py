# ---
# title: Ejercicio 04
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
# # Enunciado
# Considere la información del proyecto cuya información aparece en la tabla adjunta:

# %%
periodo_probabilidad_b = 9
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto antes de {periodo_probabilidad_b} periodos?  "
pregunta_2 = "2. Indique las actuaciones a llevar a cabo para reducir la duración del proyecto hasta 10 periodos con mínimo sobrecoste.  "
pregunta_3 = "3. Nivele los recursos del proyecto.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))

# %%
# Alargamos el ancho de las celdas
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, desviacion,  cur,  recursos, duracion_tope
     ('A',       '---  ',     3,        2,          5,       1,        1      ),
     ('B',       '---  ',     2,        1,          5,       1,        1      ),
     ('C',       'B    ',     4,        2,         25,       1,        2      ),
     ('D',       'A    ',     4,        0,          1,       1,        1      ),
     ('E',       'A    ',     2,        1,        -10,       2,        1      ),
     ('F',       'C,D  ',     5,        3,          1,       1,        2      ),
     ('G',       'C,D  ',     1,        0,        -20,       2,        1      ),
     ('H',       'E    ',     3,        1,          5,       1,        2      ),
     ('I',       'F,G  ',     1,        0,          5,       2,        1      ),
     ('J',       'F,G  ',     2,        0,         -1,       1,        1      ),
     ('K',       'F,G,H',     1,        0,          1,       1,        1      ),
] ,columns=['activity', 'predecessor', 'duration', 'standard_deviation', 'ucr', 'resources', 'cap_duration']).set_index('activity')

datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadro de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto=datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %%
proyecto = prelaciones.create_project()

# %% [markdown]
# ### Grafo PERT con numeración de nodos
#
# Utilizando el cuadro de prelaciones expandido podemos dibujar el grafo Pert correspondiente al proyecto.

# %%
#| fig-cap: "Grafo Pert con indicación con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto.pert()

# %% [markdown]
# Alternativamente podríamos haber representado su grafo Roy.

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-roy
proyecto.roy(ranksep=1.2, nodesep=1.2)

# %% [markdown]
# ### Matriz de Zaderenko

# %% [markdown]
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
duracion_media_proyecto = int(float(proyecto.duration(datos_proyecto['duration'])))

# %%
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
resultados_pert['activities'][['H_total']]

# %% [markdown]
# #### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas, cuyas actividades se listan en orden alfabético:

# %%
camino_critico = proyecto.critical_path(datos_proyecto['duration'])
for ruta, actividades in camino_critico.items():
    display(Markdown(f"{ruta}: " + ", ".join(actividades)))

# %% [markdown]
# #### Grafo PERT con indicación del camino crítico

# %%
#| fig-cap: "Grafo Pert del proyecto con indicación del camino crítico"
#| label: fig-pertCritico
proyecto.pert(durations=datos_proyecto['duration'])

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos paso a paso (avance con los botones):

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Alternativamente, se muestra el grafo roy correspondiente con indicación del camino crítico.

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación del camino crítico"
#| label: fig-royCritico
proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso (avance con los botones):

# %%
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# #### Varianza de cada rama
#
# Se calcula entonces la varianza de cada una de las rutas del camino crítico como suma de las varianzas de las actividades que lo componen.

# %% [markdown]
# El enunciado nos aporta la desviación típica, la varianza se calcula elevando al cuadrado.

# %%
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto.eval('standard_deviation ** 2'))

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_caminos

# %%
display(Markdown(f"""La varianza de la duración del proyecto es la máxima varianza de las rutas críticas. Para este proyecto es {round(desviacion_proyecto**2,2)}.  
Habitualmente será útil calcular su raiz cuadrada, la desviación típica, para ser utilizada como parámetro de la ley de distribución normal.  """))
display(Markdown(f"La desviación típica es {round(desviacion_proyecto,2)}"""))


# %% [markdown]
# ### Probabilidad de terminar antes de 9 periodos

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(float(norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)), 2)

Markdown(f"Para un proyecto con duración media {duracion_media_proyecto} y desviación típica {round(desviacion_proyecto, 2)} el valor de la probabilidad de terminar antes de {periodo_probabilidad_b} periodos es {round(probabilidad_de_terminar_antes_de_b * 100, 2)} por ciento.")

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                min_durations=datos_proyecto['cap_duration'],
                costs=datos_proyecto['ucr'],
                report=False,
                reduction=4)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoff
proyecto.display_ackoff_pasos(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=4)

# %%
for iteracion, actividades in best_option.items():
    display(Markdown(f"- Iteración {iteracion}: " + ", ".join(actividades) + "<br>"))

# %% [markdown]
# Tras esta reducción ya se ha alcanzado el objetivo de 10 periodos de duración.
# Se muestra la representación Pert y Roy del proyecto con las nuevas duración y la indicación de los caminos críticos.

# %%
proyecto.pert(durations=duration)

# %%
proyecto.roy(durations=duration, ranksep=1, nodesep=1)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# ### Análisis de la situación inicial

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto antes de la nivelación"
#| label: fig-ganttPrenivelacion
proyecto_nivelacion = proyecto.copy()
datos_proyecto = datos_enunciado.copy()

gantt_recursos, dibujo = proyecto_nivelacion.gantt_cargas(datos_proyecto, 'duration', 'resources', tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto antes de la nivelación"
#| label: fig-ganttPrenivelacionCargas
plot_cargas(gantt_df=gantt_recursos, limite_maximo_recursos=None);

# %% [markdown]
# La demanda de recursos actual no se encuentra nivelada. Se procede a aplicar el algoritmo de nivelación.

# %% [markdown]
# ### Orden de desplazamientos

# %% [markdown]
# Se aplican las siguientes reglas:  
#    - Se desplazan primero aquellas no críticas aun no desplazadas con su final más temprano más próximo al principio del proyecto.  
#    - En caso de empate se preferirá mover primero la de mayor holgura.  
#    - De persistir el empate se moverá la de menor duración.  
#    - Descartamos del análisis las actividades f1 y f2 por no consumir recursos.  
#    - En caso de persistir el empate, resulta indiferente una u otra.  

# %%
desplazamientos, cuadrados, my_data, gantt_df, dibujo = proyecto_nivelacion.nivelar(
                            data=datos_proyecto,
                            duration_label='duration',
                            resource_label='resources',
                            
                           )

# %% [markdown]
# La siguiente tabla indica el valor de la suma de los cuadrados del consumo de los recursos para cada unos de los posibles desplazamientos, en el orden de desplazamiento de las actividades utilizado. El orden seguido ha sido el de las actividades no críticas en orden creciente de periodo de finalización. Se señala en color amarillo el desplazamiento que en cada caso minimiza la suma de cuadrados.

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
# La planificación del proyecto nivelado se muestra en el siguiente diagrama de Gantt.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la nivelación"
#| label: fig-ganttPostnivelacion
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras la nivelación"
#| label: fig-ganttPostNivelacionCargas
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None);

# %%
