# ---
# title: Ejercicio 05
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
periodo_probabilidad_b = 12

pregunta_1 = "1.  Dibuje el diagrama Pert y determine el camino crítico.  "
pregunta_2 = "2.  Indique el coste de realización para duración PERT  "
pregunta_3 = f"3.  A la vista de la planificación dada, su jefe le dice que el proyeco tiene que ejecutarse en {periodo_probabilidad_b} o menos periodos. ¿Con qué probabilidad estima que se verán cumplidos los deseos de su jefe?  "
pregunta_4 = "4.  Reduzca la duración del proyecto dos unidades de tiempo con el menor incremento de coste posible.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
import pandas as pd
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, b, m, a, coste_duracion_pert, coste_duracion_tope, duracion_tope
     ('A', '---  ',   6,  2.5, 2, 1600, 1800, 2),
     ('B', '---  ',   2,    2, 2, 2900, 2900, 1),
     ('C', '---  ',   1,    1, 1,  600,  600, 1),
     ('D', '---  ',   1,    1, 1,  600,  600, 1),
     ('E', 'A,B  ',  12,    5, 4, 1600, 2200, 2),
     ('F', 'A,B  ',  12,  2.5, 2, 2000, 2200, 1),
     ('G', 'B,C  ',   6, 1.25, 1,  600,  700, 1),
     ('H', 'B,C  ',  20,   13, 6, 3500, 4900, 3),
     ('I', 'D    ',  30,   12, 6, 3500, 4300, 4),
     ('J', 'E,M  ',   4,    4, 4, 1500, 1500, 2),
     ('K', 'F    ',   7,    7, 7, 2000, 2000, 3),
     ('L', 'J,K,G',   1,    1, 1, 1000, 1000, 1),
     ('M', 'F    ',   3,    3, 3,  500,  500, 1),

] ,columns=['activity', 'predecessors', 'b', 'm', 'a',
            'cost_pert_duration', 'cost_cap_duration', 'cap_duration']).set_index('activity')

datos_enunciado['b'] = datos_enunciado['b'].astype(int)
datos_enunciado

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadro de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.
#

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessors')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Precálculo de variables de interés

# %% [markdown]
# A partir de los datos del enunciado podemos calcular la duración media, la desviación típica y el coste unitario de reducción que utilizaremos para contestar las preguntas realizadas.

# %%
display(Math(r"$duration=\frac{a+4m+b}{6}$"))
display(Math(r"$standard ~ deviation=\frac{b-a}{6}$"))
display(Math(r"$ucr=\frac{C_{cap}-C_{pert}}{d_{cap}-d_{pert}}$"))

# %% [markdown]
# {{< pagebreak >}}

# %%
datos_proyecto['duration'] = datos_proyecto.eval('(b + 4*m +a)/6')
datos_proyecto['standard_deviation'] = datos_proyecto.eval('(b -a)/6').round(2)
datos_proyecto['ucr'] = datos_proyecto.eval('(cost_cap_duration - cost_pert_duration)/(duration - cap_duration)').round(2)

# %%
#| tbl-cap: "Duración y desviación típica de las actividades calculadas"
#| label: tbl-datosInteresantes
datos_proyecto[['b', 'm', 'a', 'duration', 'standard_deviation', 'ucr']]

# %% [markdown]
# ### Grafo PERT con numeración de nodos
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
proyecto.roy(ranksep=1, nodesep=1)

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
resultados_pert = proyecto.calculate_pert(datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Duración media del proyecto

# %%
duracion_media_proyecto = int(float(proyecto.duration(datos_proyecto['duration'])))
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
# ### Holgura total de las actividades
# De acuerdo a los cálculos realizados, la holgura total de las actividades es:

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
resultados_pert['activities']

# %% [markdown]
# ### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas, cuyas actividades se listan en orden alfabético:

# %%
camino_critico = proyecto.critical_path(datos_proyecto['duration'])
proyecto.display_critical_path(datos_proyecto['duration'])

# %% [markdown]
# ##### Grafo PERT con indicación de tiempos y camino crítico

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert del proyecto con indicación del camino crítico"
#| label: fig-pertCritico
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Alternativamente, se muestra el grafo roy correspondiente con indicación del camino crítico.

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación del camino crítico"
#| label: fig-royCritico
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Para responder este apartado tan sólo es necesario sumar los costes indicados en la columna coste_duracion_pert.

# %%
Markdown(f"La suma de los costes es {int(datos_proyecto['cost_pert_duration'].sum())}")

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# ### Duración media del proyecto

# %% [markdown]
# La duración media del proyecto ya la tenemos calculada:

# %%
duracion_media_proyecto = int(proyecto.duration(datos_proyecto['duration']))

# %%
Markdown(f"La media de la duración del proyecto es {duracion_media_proyecto}")

# %% [markdown]
# ### Varianza de la duración del proyecto
#
# Se calcula entonces la varianza de cada una de las rutas del camino crítico como suma de las varianzas de las actividades que lo componen. Se indica en color la ruta con máxima varianza, que determina la varianza del proyecto.

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_caminos

# %%
Markdown(f"La desviación típica de la duración del proyecto es la raiz cuadrada de su varianza. En este caso la desviación típica tiene valor {round(float(desviacion_proyecto), 2)}")

# %% [markdown]
# ### Cálculo de la probabilidad

# %% [markdown]
# Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad:

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)
probabilidad_de_terminar_antes_de_b  = round(float(probabilidad_de_terminar_antes_de_b * 100), 2)

# %%
Markdown(f"La probabilidad pedida es {probabilidad_de_terminar_antes_de_b} por ciento.")

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Apartado 4.

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Se aborda la solución de este apartado mediante la técnica de reducción de la duración del proyecto con mínimo coste.

# %%
ackoff, best_option, durations, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                                                            min_durations=datos_proyecto['cap_duration'],
                                                            costs=datos_proyecto['ucr'],
                                                            reduction=2)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoff
proyecto.display_ackoff_pasos(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %% [markdown]
# Las actividades reducidas en cada una de las iteraciones son:

# %%
for iteracion, actividades in best_option.items():
    display(Markdown(f"- Iteración {iteracion}: " + ", ".join(actividades) + "<br>"))

# %%

# %%
