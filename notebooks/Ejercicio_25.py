# ---
# title: Ejercicio 25
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
# Considere la información del proyecto cuya información aparece en la tabla adjunta:    

# %%
periodo_a_probabilidad = 12
periodo_b_probabilidad = 14
duracion_objetivo = 10
limite_maximo_recursos = [8]*7 + [16]
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} periodos?  "
pregunta_2 = f"2. Respecto a la situación original del enunciado, indique mediante el método de Ackoff-Sasieni cuales serían las actuaciones a llevar a cabo para reducir con mínimo sobrecoste la duración media del proyecto a {duracion_objetivo} periodos.  "
pregunta_3 = f"3. Respecto a la situación original del enunciado, indique cuales serían las actuaciones para poder ejecutar el proyecto con un límite en los recursos de {limite_maximo_recursos[0]} recursos por periodo durante los primeros siete periodos, y {limite_maximo_recursos[-1]} recursos por periodo posteriormente.  "
pregunta_4 = f"4. Respecto a la situación original del enunciado, calcule el valor del CPI y del SPI para el punto de control cuyos datos se proporcionan.  "

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

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, duracion_tope, desviación, coste_planificado, coste_real, duracion_real, realizado, cur, recursos
     ('A',      'D      ',       2,         1,              0.2,         80,              80,          2,           100,   80,    4 ),
     ('B',      'A      ',       2,         1,              0.3,         70,              60,          2,            80,   10,    2 ),
     ('C',      'B      ',       3,         1,              0.4,         60,               0,          0,             0,   40,    1 ),
     ('D',      '---    ',       4,         3,              0.5,         40,              50,          5,           100,   20,    4 ),
     ('E',      'A      ',       1,         1,              0.1,         40,              30,          3,           100,   30,    3 ),
     ('F',      'D,E    ',       2,         1,              0.3,         10,              10,          3,            80,   40,    5 ),
     ('G',      '---    ',       4,         1,              0.1,         40,              50,          3,           100,   20,    4 ),
     ('H',      'I    ',       4,         2,              0.5,        100,             140,          4,            90,   50,    2 ),
     ('I',      'D      ',       2,         1,              0.1,         50,              70,          1,            50,   60,    3 ),
     ('J',      'F,H,K  ',       2,         1,              0.4,         50,               0,          0,             0,   70,    1 ),
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
# # Solución
# ## Apartado 1
#

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadro de prelaciones expandido

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto=datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert"
#| label: fig-PertSinTiempos
proyecto = prelaciones.create_project()
proyecto.pert()

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ### Matriz de Zaderenko

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duration'])

# %% [markdown]
# Los tiempos tempranos y tardíos han resultado ser

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Holgura total de las actividades

# %% [markdown]
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valores de la holgura total de las actividades"
#| label: tbl-Holguras
resultados_pert['activities'][['H_total']]

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
calendario

# %% [markdown]
# ### Caracterización de la duración del proyecto mediante una distribución Normal

# %% [markdown]
# A fin de calcular la probabilidad pedida, es necesario caracterizar la distribución Normal de la duración del proyecto. Esto consiste en determinar los valores de sus parámetros: media y desviación típica de la duración del proyecto.

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duration']))

# %%
Markdown(f"De los tiempos obtenidos en el método de Zaderenko, ya sabemos que la duración media del proyecto es {duracion_media_proyecto} periodos.")

# %% [markdown]
# ### Camino crítico
# El camino crítico es:

# %%
proyecto.display_critical_path(durations=datos_proyecto['duration'])

# %% [markdown]
# ### Grafo PERT con indicación del camino crítico

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con tiempos, y rutas críticas"
#| label: fig-PertConTiempos
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# La desviación típica puede calcularse para cada ruta del camino crítico. En este caso sólo existe una ruta

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                   variances=datos_proyecto['variance']) 
desviacion_proyecto = round(desviacion_proyecto, 2)

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_caminos

# %%
Markdown(f"Por tanto, la desviación típica del proyecto resulta ser $\\sigma$ = {desviacion_proyecto}")

# %% [markdown]
# ### Probabilidad de terminar en el intervalo especificado

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(norm.cdf(periodo_b_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto)*100, 1)
probabilidad_de_terminar_antes_de_a = round(norm.cdf(periodo_a_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto)*100, 1)

probabilidad_de_terminar_entre_a_y_b = probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a
probabilidad_de_terminar_entre_a_y_b = round(probabilidad_de_terminar_entre_a_y_b, 1)

# %%
Markdown(f"""Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad de terminar entre los periodos {periodo_a_probabilidad} y {periodo_b_probabilidad}.  
- La probabilidad de terminar antes de {periodo_a_probabilidad} periodos es  {probabilidad_de_terminar_antes_de_a} por ciento    
- La probabilidad de terminar antes de {periodo_b_probabilidad} periodos es {probabilidad_de_terminar_antes_de_b} por ciento    

Por tanto, a partir de la diferencia de estos dos valores tenemos la probabilidad de terminar dentro del mencionado intervalo: {probabilidad_de_terminar_entre_a_y_b} por ciento.
""")

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
datos_proyecto = datos_enunciado.copy()
proyecto = prelaciones.create_project()

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations     = datos_proyecto['duration'],
                                                           min_durations = datos_proyecto['cap_duration'],
                                                           costs         = datos_proyecto['ucr'],
                                                           reduction     = duracion_media_proyecto - duracion_objetivo
                                                          )

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos_tikz(
    durations     = datos_proyecto['duration'],
    min_durations = datos_proyecto['cap_duration'],
    costs         = datos_proyecto['ucr'],
    reduction     = duracion_media_proyecto - duracion_objetivo)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
proyecto_asignacion = proyecto.copy()
gantt, dibujo = proyecto_asignacion.gantt(datos_proyecto,
                          'duration',
                          'resources',
                          total='fila',
                          holguras=True, 
                          tikz=True)

# %%
#| fig-cap: "Diagrama de Gantt del proyecto de acuerdo a su definición original."
#| label: fig-ganttPreAsignación
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original"
#| label: fig-preAsignacionBarras
plot_cargas(gantt, limite_maximo_recursos)

# %% [markdown]
# La demanda de recursos actual infringe el límite impuesto. Es necesario aplicar el algoritmo de asignación.

# %% [markdown]
# {{< pagebreak >}}

# %%
duration_resources = datos_proyecto[['duration','resources']].copy()
duration_resources, df, dibujo = proyecto_asignacion.asignar(data=duration_resources,
                                                            duration_label='duration',
                                                            resource_label='resources',
                                                            report=False,
                                                            maximo=limite_maximo_recursos,
                                                            )

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la asignación"
#| label: fig-ganttPostAsignación
pinta_tikz(dibujo)

# %% [markdown]
# {{< pagebreak >}}

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(df, limite_maximo_recursos)

# %% [markdown]
# Tras aplicar el algoritmo de asignación el proyecto ya es compatible con los límites establecidos.

# %% [markdown]
# ## Apartado 4

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
