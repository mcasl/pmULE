# ---
# title: Ejercicio 12
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
# ## Enunciado
#
# Considere la información del proyecto cuya información aparece en la tabla adjunta:  
#

# %%
periodo_a_probabilidad = 10
periodo_b_probabilidad = 12
duracion_objetivo = 9
limite_maximo_recursos = [8]*8 + [16]
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} periodos?  "
pregunta_2 = f"2. Indique cuales serían las actuaciones a llevar a cabo para reducir con mínimo sobrecoste la duración media del proyecto a {duracion_objetivo} periodos.  "
pregunta_3 = f"3. Indique cuales serían las actuaciones para poder ejecutar el proyecto con un límite en los recursos de {limite_maximo_recursos[0]} recursos por periodo durante los primeros ocho periodos, y de {limite_maximo_recursos[-1]} recursos por periodo en los siguientes periodos.  "
pregunta_4 =  "4. Calcule el valor del CPI y del SPI para el punto de control de 11 periodos.  "

# %%

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, duracion_tope, desviación, coste_planificado, coste_real, duracion_real, realizado, cur, recursos
     ('A',      'D      ',       2,         1,              0.2,         60,              80,          2,           100,   80,    4 ),
     ('B',      'A      ',       2,         1,              0.3,         70,              60,          2,            80,   10,    2 ),
     ('C',      'B      ',       3,         1,              0.4,         50,               0,          0,             0,   40,    1 ),
     ('D',      '---    ',       4,         3,              0.5,         40,              50,          5,           100,   20,    4 ),
     ('E',      'A      ',       1,         1,              0.1,         30,              30,          3,           100,   30,    3 ),
     ('F',      'D,E    ',       2,         1,              0.3,         10,              10,          3,            80,   40,    5 ),
     ('G',      '---    ',       4,         1,              0.1,         40,              50,          3,           100,   20,    4 ),
     ('H',      'G      ',       4,         2,              0.5,        110,             140,          4,            90,   50,    2 ),
     ('I',      'D      ',       2,         1,              0.1,         90,              70,          1,            50,   60,    3 ),
     ('J',      'F,H,I,K',       2,         1,              0.4,         50,               0,          0,             0,   70,    1 ),
     ('K',      'D,G    ',       4,         1,              0.2,         40,              30,          2,            60,   90,    2 ),
     ('L',      'H,K    ',       1,         1,              0.2,         30,               0,          0,             0,   10,    3 ),
] ,columns=['activity', 'predecessor', 'duration', 'cap_duration', 'standard_deviation', 'planned_value', 'actual_cost',
            'actual_duration', 'performed', 'ucr', 'resources']).set_index('activity')
tabla = datos_enunciado.copy()
tabla.columns = ["Pred.", 'Duration', 'Cap <br> duration', 'Standard <br> deviation', 'PV', 'AC',
            'Actual <br> duration', 'Performed', 'UCR', 'Resources']
display(tabla.style.set_properties(**{'text-align': 'center'}).format(precision=2))

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Solución
# ### Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadro de prelaciones
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
datos_proyecto=datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto = prelaciones.create_project()
proyecto.pert()

# %% [markdown]
# {{< pagebreak >}}

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

# %% [markdown]
# ### Duración media del proyecto

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duration']))
Markdown(f"La información del nodo final del proyecto nos informa de la duración media del proyecto: {duracion_media_proyecto} periodos")

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ### Holgura total de las actividades
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
resultados_pert['activities']

# %% [markdown]
# ### Camino crítico
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
calendario

# %% [markdown]
# ### Grafo PERT con indicación de tiempos y rutas del camino crítico de acuerdo a la planificación inicial
# A continuación se muestra el grafo Pert del proyecto, con indicación de las actividades críticas:

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
proyecto.pert(durations=datos_proyecto['duration'])

# %% [markdown]
# ### Caracterización de la duración del proyecto mediante una distribución Normal
# A fin de calcular la probabilidad pedida, es necesario caracterizar la distribución Normal de la duración del proyecto. Esto consiste en determinar los valores de sus parámetros: media y desviación típica de la duración del proyecto.

# %%
duracion_media_proyecto = int(proyecto.duration(datos_proyecto['duration']))
Markdown(f"La duración media del proyecto ya la tenemos calculada: {duracion_media_proyecto} periodos")

# %% [markdown]
# Para determinar la desviación típica debemos obtener en cada rama del camino crítico la suma de las varianzas, y quedarnos con la mayor.

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                   variances=datos_proyecto['variance']) 
desviacion_proyecto = round(desviacion_proyecto, 2)

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_rutas

# %%
Markdown(f"El proyecto sigue una distribución normal con media {duracion_media_proyecto} periodos y desviación típica {round(desviacion_proyecto, 2)}.")


# %% [markdown]
# #### Probabilidad de terminar entre 10 y 12 periodos

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(norm.cdf(periodo_b_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)
probabilidad_de_terminar_antes_de_a = round(norm.cdf(periodo_a_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)

probabilidad_de_terminar_entre_a_y_b = probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a


# %%
Markdown(f"""Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad de terminar entre los periodos {periodo_a_probabilidad} y {periodo_b_probabilidad}.  
- La probabilidad de terminar antes de {periodo_a_probabilidad} periodos es  {probabilidad_de_terminar_antes_de_a}  
- La probabilidad de terminar antes de {periodo_b_probabilidad} periodos es {probabilidad_de_terminar_antes_de_b}  

Por tanto, a partir de la diferencia de estos dos valores tenemos la probabilidad de terminar dentro del mencionado intervalo: {int(probabilidad_de_terminar_entre_a_y_b*100)} por ciento.
""")

# %% [markdown]
# ### Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# #### Reducción a 9 periodos

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                min_durations=datos_proyecto['cap_duration'],
                costs=datos_proyecto['ucr'],
                reduction = duracion_media_proyecto - duracion_objetivo)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
ackoff

# %% [markdown]
# ### Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
proyecto_asignacion = proyecto.copy()
gantt, dibujo = proyecto_asignacion.gantt(datos_proyecto,
                          'duration',
                          'resources',
                          total='fila',
                          holguras=True, tikz=True)
pinta_tikz(dibujo)

# %% [markdown]
# La fila “Total” obtenida, que podemos representar gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo.
#

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt, limite_maximo_recursos)

# %% [markdown]
# La demanda de recursos actual infringe el límite impuesto. Es necesario aplicar el algoritmo de asignación.

# %% [markdown]
# ### Algoritmo de asignación

# %%
duracion_y_recursos = datos_proyecto[['duration', 'resources']]
datos_proyecto_tras_asignacion, gantt_df, dibujo  = proyecto_asignacion.asignar(
                                    data=duracion_y_recursos,
                                    duration_label='duration',
                                    resource_label='resources',
                                    maximo=limite_maximo_recursos,
                                    report=False,
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
# ### Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
PV = int(datos_proyecto['planned_value'].sum())
EV = int(datos_proyecto.eval('planned_value * performed / 100').sum())
AC = int(datos_proyecto['actual_cost'].sum())
CPI = float(EV/AC)
SPI=float(EV/PV)

# %%
Markdown(f"""Utilizando CPI=EV/AC y SPI=EV/PV, tenemos:  
PV={PV}  
EV={EV}  
AC={AC}  
CPI={round(CPI,2)}  
SPI={round(SPI,2)}""")

# %%

# %%

# %%
