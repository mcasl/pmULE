# ---
# title: Ejercicio 30
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
from IPython.display import display, HTML, Markdown, Math, Latex
import handcalcs.render
from handcalcs import handcalc
from scipy.stats import norm

sys.path.append('../src')
from pmule import PredecessorTable, ProjectGraph, EarnedValue, pinta_tikz, plot_cargas, plot_curvas_valor_ganado

# %% [markdown]
# # Enunciado
#
# Considere el proyecto cuya información se adjunta y realice los siguientes apartados.

# %%
periodo_a_probabilidad_normal = 10
periodo_b_probabilidad_normal = 18
probabilidad_exito_normal = 0.85

periodo_a_probabilidad_montecarlo = 13
periodo_b_probabilidad_montecarlo = 18
probabilidad_exito_montecarlo = 0.85

pregunta_1 = Markdown( "1. Indique el coste de realización del proyecto para la duración inicialmente planificada")
pregunta_2 = Markdown( "2. Dibuje el grafo Pert y Roy del proyecto. Utilizando las duraciones medias de las actividades, determine el camino crítico e indique el calendario de ejecución de las actividades.  ")
pregunta_3 = Markdown(f"""3. Utilizando el método de aproximación a la normal:  
                        - determine la probabilidad de terminar antes de {periodo_a_probabilidad_normal} periodos.  
                        - ¿cuál es la duración del proyecto para la cual podemos esperar que el proyecto finalice antes de esa fecha con un {probabilidad_exito_normal*100}% de probabilidad?""")
pregunta_4 = Markdown(f"""4. Utilizando el método de MonteCarlo:  
                        - determine la duración media y desviación típica de la duración del proyecto.  
                        - determine la probabilidad de que el proyecto finalice en menos de {periodo_a_probabilidad_montecarlo} periodos.  
                        - determine la duración del proyecto para la cual podemos esperar que el proyecto finalice antes de esa fecha con un {probabilidad_exito_montecarlo*100}% de probabilidad?.  """)



# %%
for item in dir():
    if item.startswith('pregunta_'):
        display(eval(item))

# %%
# %%capture
# %mkdir -p ~/temp 
# %cd ~/temp

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

import pandas as pd
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, b, m, a, BAC, coste_duracion_tope, duracion_tope, duracionReal, AC, PorcentajeRealizado, recursos
     ('A', '---  ',   6,  2.5, 2,      1600, 1800, 2, 5, 1800, 100, 5 ),
     ('B', '---  ',   4,    3, 2,      2900, 3100, 1, 3, 3200, 100, 5 ),
     ('C', '---  ',   3,    2, 1,       600,  800, 1, 2,  550, 100, 5 ),
     ('D', 'B    ',   3,    2, 1,       600,  700, 1, 1,  540, 100, 10 ),
     ('E', 'D,C  ',   7,    5, 3,      1600, 2200, 2, 5,  750,  50, 5 ),
     ('F', 'A,D,C',   4,    3, 2,      2000, 2800, 2, 4,  500,  40, 5 ),
     ('G', 'D,C  ',   6,  1.25, 1,       600,  700, 1, 5,  200,  50, 10 ),
     ('H', 'F,E,G',  10,    8, 6,       3500, 4900, 3, 0,    0,   0, 5 ),
     ('I', 'E,G  ',   8,    7, 6,       3700, 4300, 4, 0,    0,   0, 5 ),

] ,columns=['activity', 'predecessors', 'b', 'm', 'a',
            'BAC', 'cost_cap_duration', 'cap_duration', 'actual_duration', 'AC', 'performed', 'resources']).set_index('activity')

datos_enunciado['b'] = datos_enunciado['b'].astype(int)
datos_enunciado

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# # Solución

# %% [markdown]
# ## Parte 1. Análisis desde el origen hasta el presente.

# %% [markdown]
# ### Apartado 1

# %%
display(pregunta_1)

# %% [markdown]
# Para responder este apartado tan sólo es necesario sumar los costes indicados en la columna relativa al coste inicialmente planificado (BAC).

# %%
Markdown(f"La suma de los costes es {int(datos_enunciado['BAC'].sum())}")

# %% [markdown]
# ### Apartado 2

# %%
display(pregunta_2)

# %% [markdown]
# ##### Precálculo de la duración media de las actividades, desviación típica, y coste unitario de reducción.

# %% [markdown]
# Es conveniente realizar el cálculo de los valores que vamos a necesitar en diferentes apartados:  
# - duración media de las actividades  
# - desviación estándar de la duración de las actividades  
# - varianza de la duración de las actividades  
# - coste unitario de reducción.  

# %%
#| tbl-cap: "Cálculo de la duración media, desviación típica, varianza, y coste unitario de reducción."
#| label: tbl-calculosInteresantes
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = datos_proyecto.eval('(b + 4*m +a)/6')
datos_proyecto['standard_deviation'] = datos_proyecto.eval('(b -a)/6')
datos_proyecto['variance'] = datos_proyecto['standard_deviation']**2
datos_proyecto['ucr'] = datos_proyecto.eval('(cost_cap_duration - BAC)/(duration - cap_duration)').round(2)
datos_proyecto[['b', 'm', 'a', 'duration', 'standard_deviation', 'variance', 'ucr']]

# %% [markdown]
# Con esta información podemos completar el cuadro de datos del enunciado

# %%
#| tbl-cap: "Datos del proyecto."
#| label: tbl-DatosProyectoComplementados
datos_proyecto

# %% [markdown]
# #### Cuadros de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessors')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# #### Dibuje el diagrama de Gantt del proyecto planificado

# %%

prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessors')
prelaciones.display_immediate_linkage_matrix()
proyecto = prelaciones.create_project()

# %%
#| fig-cap: "Diagrama de Gantt del proyecto en su estado inicial"
#| label: fig-GanttInicial
gantt, dibujo = proyecto.gantt(datos_proyecto,
                          'duration',
                          'resources',
                          total='fila',
                          holguras=True,
                          tikz=True, 
                          params=dict(
                             period_width=1
                          ),
              )
pinta_tikz(dibujo)

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# #### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto.display_pert_construccion(tabla=prelaciones)

# %% [markdown]
# #### Grafo Roy

# %%
#| fig-cap: "Grafo Roy con indicación de los números de nodos"
#| label: fig-RoyNumeracionNodos
proyecto.roy()

# %% [markdown]
# #### Matriz de Zaderenko
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
resultados_pert = proyecto.calculate_pert(datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# #### Holgura total de las actividades
# Conocidos los tiempos tempranos y tardíos se puede proceder a calcular la holgura total de las actividades.

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
resultados_pert['activities']

# %% [markdown]
# #### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas, cuyas actividades se listan en orden alfabético:

# %%
proyecto.display_critical_path(datos_proyecto['duration'])

# %% [markdown]
# #### Grafo PERT con indicación de tiempos, y rutas del camino crítico

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert del proyecto con indicación del camino crítico"
#| label: fig-pertCritico
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# #### Grafo Roy con indicación de tiempos, y rutas del camino crítico

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación del camino crítico"
#| label: fig-royCritico
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# #### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
duration_label='duration'
calendario = proyecto.calendar(datos_proyecto[duration_label]).drop(columns='duracion').rename(columns={'inicio_mas_temprano':'Inicio más temprano',
                                                                    'inicio_mas_tardio':'Inicio más tardio',
                                                                    'fin_mas_temprano':'Fin más temprano',
                                                                    'fin_mas_tardio':'Fin más tardio',
                                                                    'H_total': 'Holgura total'
                                                                   })
proyecto.display_calendario(datos_proyecto[duration_label])

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Apartado 3

# %%
display(pregunta_3)

# %% [markdown]
# ##### Cálculo de la media de la duración del proyecto

# %%
duraciones_medias = datos_proyecto['duration']
duracion_media_proyecto = int(proyecto.duration(durations=duraciones_medias))

# %% [markdown]
# La duración media del proyecto la tenemos ya calculada al haber calculado el calendario de ejecución del proyecto para duraciones medias de las actividades.

# %%
Markdown(f"La duración media del proyecto es {duracion_media_proyecto} periodos")

# %% [markdown]
# ##### Cálculo de la desviación típica del proyecto

# %% [markdown]
# Calculamos la varianza para cada rama como el máximo valor de las varianzas de las rutas del camino crítico.

# %%
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations = datos_proyecto['duration'],
                                                  variances = datos_proyecto['variance'])
varianza_caminos

# %%
Markdown(f"La desviación típica de la duración del proyecto es la máxima desviación típica de todas las rutas críticas. Para este proyecto, haciendo la raíz cuadrada del valor de la varianza, tenemos el valor de desviación típica  {round(desviacion_proyecto,2)}.  ")

# %% [markdown]
# #### Calculo de la probabilidad de que el proyecto finalice antes del periodo indicado

# %% [markdown]
# Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad:

# %%
probability_TLC = norm.cdf(periodo_a_probabilidad_normal, loc=duracion_media_proyecto, scale=desviacion_proyecto)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, hasta=periodo_a_probabilidad_normal)
Markdown(f"""Utilizando, por ejemplo, una calculadora, para un proyecto con:  
                - duración media {duracion_media_proyecto} y  
                - desviación típica {round(desviacion_proyecto, 2)}  
**el valor de la probabilidad** de que el proyecto finalice antes de {periodo_a_probabilidad_normal} periodos **es {round(probability_TLC * 100, 2)} por ciento**.""")

# %% [markdown]
# #### Cálculo de la duración del proyecto para la cual se espera que el proyecto haya finalizado antes de esa fecha con la probabilidad requerida

# %%
fecha_exito_normal = round(norm.ppf(probabilidad_exito_normal, loc=duracion_media_proyecto, scale=desviacion_proyecto), 1)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, percentil=probabilidad_exito_normal)
Markdown(f"Con una calculadora, por ejemplo, obtenemos que el número de periodos necesarios para alcanzar una tasa de éxito del {probabilidad_exito_normal*100} por ciento es {fecha_exito_normal} periodos.  ")

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Apartado 4

# %%
display(pregunta_4)

# %%
num_of_samples = 100000

names_of_the_tasks = datos_proyecto.index
np.random.seed(2025)
durations_montecarlo = pd.DataFrame([np.random.normal(size = num_of_samples,
                                                       loc  = datos_proyecto.duration[task],
                                                       scale= datos_proyecto.standard_deviation[task])
                                      for task in names_of_the_tasks],
                                     index=names_of_the_tasks).T.round(1)

# %%
durations_montecarlo

# %% [markdown]
# ### Determinamos la matriz de caminos del proyecto.

# %%
path_matrix = proyecto.path_matrix(dummies=False)
proyecto.display_path_matrix(dummies=False)

# %% [markdown]
# ### Cálculo de la duración del proyecto

# %%
paths_durations_per_iteration = durations_montecarlo @ path_matrix.T
paths_durations_per_iteration

# %%
project_duration_per_iteration = paths_durations_per_iteration.max(axis="columns")
project_duration_per_iteration.T

# %% [markdown]
# ### Determina la duración media y la desviación típica de la duración del proyecto.

# %%
project_duration_per_iteration.plot(kind='density');

# %%
project_duration_per_iteration.describe()

# %%
project_mean_duration = project_duration_per_iteration.mean()
project_std_deviation_duration = project_duration_per_iteration.std()

# %%
f"La duración media del proyecto es: {project_mean_duration}, la desviación típica de la duración del proyecto es: {project_std_deviation_duration}"

# %% [markdown]
# ### ¿Cuál es la probabilidad de que el proyecto finalice en menos de 13 periodos de tiempo?

# %% [markdown]
# #### Utilizando la distribución normal

# %%
period = 13
probability = norm.cdf(period, loc=project_mean_duration, scale=project_std_deviation_duration)
from pmule import display_normal_tikz
display_normal_tikz(project_mean_duration, project_std_deviation_duration, hasta=period)
Markdown(f"Para un proyecto con duración media {project_mean_duration} y desviación típica {round(project_std_deviation_duration, 2)} el valor de la probabilidad pedida es {round(probability * 100, 2)} por ciento.")

# %% [markdown]
# #### Utilizando los valores de Montecarlo

# %%
probability_montecarlo = (project_duration_per_iteration <= period).mean()
Markdown(f"Considerando la secuencia de valores obtenidos por el método de MonteCarlo, el valor de la probabilidad pedida es {round(probability_montecarlo * 100, 2)} por ciento.")

# %% [markdown]
# ### ¿Cuál es la duración del proyecto para la cual podemos garantizar que el proyecto va finalizar antes de esa fecha con un 85% de probabilidad?

# %% [markdown]
# #### Utilizando la distribución normal

# %%
prob=0.85
from pmule import display_normal_tikz
display_normal_tikz(project_mean_duration, project_std_deviation_duration, percentil=prob)
norm.ppf(prob, loc=project_mean_duration, scale=project_std_deviation_duration)

# %% [markdown]
# #### Utilizando los valores de Montecarlo:

# %% [markdown]
# Podemos ordenar los valores en sentido ascendente y quedarnos con aquél valor superior al 85% de las muestras.

# %%
project_duration_per_iteration.sort_values(ascending=True).iloc[int(prob*num_of_samples)]

# %% [markdown]
# O bien, utilizando la función `quantile`

# %%
project_duration_per_iteration.quantile(q=prob, interpolation='higher')

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Reducción de la duración del proyecto con mínimo incremento de coste: Algoritmo de Ackoff y Sasieni de todo el proyecto

# %%
ackoff, best_option, durations, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                                                            min_durations=datos_proyecto['cap_duration'],
                                                            costs=datos_proyecto['ucr'],
                                                            reduction=20)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=20)

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Nivelación

# %%
proyecto_nivelacion = proyecto.copy()

desplazamientos, cuadrados, my_data, gantt_df, dibujo = proyecto_nivelacion.nivelar(data=datos_proyecto,
                           duration_label='duration',
                           resource_label='resources',
                           #report=False,
                           )
proyecto.copy().display_nivelar_pasos_tikz(datos_proyecto, duration_label='duration', resource_label='resources')

# %%
#| include: false
cuadrados   # la tabla de sumas de cuadrados ya se muestra en el visor anterior

# %%
pinta_tikz(dibujo)

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Asignación

# %% [markdown]
# ##### El número máximo de recursos es 10

# %%
proyecto_asignacion = proyecto.copy()
my_data, gantt_df, dibujo = proyecto_asignacion.asignar(data=datos_proyecto,
                                              duration_label='duration',
                                              resource_label='resources',
                                              maximo = [10],
                                              report = False,
                                              params = dict(
                                                  period_width=1
                                                  )
                                              )
pinta_tikz(dibujo)

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Valor ganado

# %%
vg = EarnedValue(proyecto)

gantts = vg.calcula_gantts(datos_proyecto,
                           planned_durations_label='duration',
                           actual_durations_label='actual_duration',
                           PV_label='BAC',
                           AC_label='AC',
                           percentage_complete_label='performed',
                           tikz=True,
                params=dict(
                        period_width=1.5,
                        row_height=0.7,
                        background_horizontal_line_color='black',
                        activity_relative_height=0.6,
                        critical_background_color='white!90!green',
                        regular_background_color='white!90!green',
                        regular_text_color='black',
                        critical_text_color='black',
                        activity_inner_text_style=r"\bfseries\normalsize",),
)

# %%
#| out-height: 80%
#| fig-cap: "PV: Valor planificado"

pinta_tikz(gantts['gantt_PV'])

# %%
#| out-height: 80%
#| fig-cap: "AC: Coste real"

pinta_tikz(gantts['gantt_AC'])

# %%
#| out-height: 80%
#| fig-cap: "EV: Valor ganado"

pinta_tikz(gantts['gantt_EV'])

# %%
#| out-width: 50%
#| fig-cap: "Evolución en el tiempo de los parámetros del método del valor ganado"
plot_curvas_valor_ganado(gantts)

# %%
acumulados = gantts['acumulados']
EV = acumulados.tail(1)['EV'].iloc[0]
AC = acumulados.tail(1)['AC'].iloc[0]
PV = acumulados.tail(1)['PV'].iloc[0]
BAC = round(float(datos_enunciado['BAC'].sum()),2)

# %%
Markdown(f"""Utilizando: \\
PV={round(PV,0)} \\
AC={AC} \\
EV={EV} \\
y el BAC={BAC} obtenido sumando los costes planificados, obtenemos:
""")

# %%
from handcalcs.decorator import handcalc

@handcalc(jupyter_display=False) 
def my_calculation(EV,AC,PV,BAC):
    CPI = EV/AC
    SPI = EV/PV
    EAC = BAC / CPI
    CV = EV - AC
    CV_relativo = CV/EV *100
    SV = EV - PV
    SV_relativo = SV/PV *100
    VAC = BAC - EAC
    PC = EV / BAC
    PS = AC / BAC
    PP = PV / BAC
    TCPI_BAC = (BAC - EV)/(BAC - AC)
    TCPI_EAC = (BAC - EV)/(EAC - AC)
    LRE = 35000
    TCPI_LRE = (BAC - EV)/(LRE - AC)
latex_code, _ = my_calculation(EV,AC,PV,BAC)
Math((latex_code))

# %% [markdown]
# {{< pagebreak >}}

# %%
#| echo: false
#| output: false

### Preguntas cuestionario

from cuestionario import *

variance_label = 'variance'

M = ProjectQuestionMaker(proyecto)

M.duration(data=datos_proyecto,
           durations_label=duration_label,
           question='¿Cuál es la duración del proyecto utilizando valores medios en las duraciones de las actividades?')

M.standard_deviation(data=datos_proyecto,
                     durations_label=duration_label,
                     variances_label=variance_label,
                     question=None)

M.distant_predecessor(question=None)

M.path_matrix(question=None)

M.paths_duration(data=datos_proyecto,
                 durations_label=duration_label,
                 question=None,
                 dec=0)

M.date_for_probability(data=datos_proyecto,
                       duration_label=duration_label,
                       variance_label=variance_label,
                       probability=probabilidad_exito_normal,
                       question=None)

M.calendar(data=datos_proyecto,
           durations_label=duration_label,
           question=None)

M.probability_for_interval(data=datos_proyecto,
                           duration_label=duration_label,
                           variance_label=variance_label,
                           lower_bound=45,
                           upper_bound=50,
                           question=None,
                           dec=2)

#print(M.write_test())

# %%
#| echo: false
#| output: false

def generate_numerical_cloze_questions(df, tolerance=0.1):
    """
    Generate Moodle CLOZE numeric questions from a Pandas DataFrame with numeric values.

    Args:
        df (pd.DataFrame): A Pandas DataFrame with numeric values.
        tolerance (float): Accepted error tolerance for NUMERIC questions.

    Returns:
        str: A Moodle CLOZE formatted question containing numeric questions for each cell.
    """
    question_text = "Complete the following table with the correct numeric values:\n\n"

    for index, row in df.iterrows():
        row_text = f"{str(index)}:\n"
        for col in df.columns:
            # Add a NUMERIC question for each cell with a tolerance
            row_text += f" {col}: {{1:NUMERICAL:={row[col]}:{tolerance}}} "
        question_text += row_text + "\n"

    return question_text

calendar_questions = generate_numerical_cloze_questions(calendario)
#print(calendar_questions)

# %% [markdown]
# ## Parte 2. Análisis desde el presente hasta la finalización.

# %% [markdown]
# Cálculos sobre lo que falta por completar del proyecto. El proyecto avanza a ritmo de lo ya completado, y las tareas sin comenzar se presuponen que siguen la planificación inicial

# %% [markdown]
# Observamos el diagrama de Gantt de lo realizado hasta el momento

# %%
gantt, dibujo = proyecto.gantt(datos_proyecto,
                          'actual_duration',
                          'names',
                          total=None,
                          holguras=True,
                          tikz=True, 
                          params=dict(
                             period_width=0.7,
                             critical_background_color='white!90!green',
                              critical_text_color='black',
                              
                          ),
              )
pinta_tikz(dibujo)

# %% [markdown]
# Recalculamos las duraciones de las tareas del proyecto, las ya realizadas con su duración real, las comenzadas siguen a ritmo de lo realizado y las que aún no han comenzado se cree que se desarrollarán según lo planificado

# %%
datos_proyecto_actualizado = datos_proyecto.copy()

empezadas = datos_proyecto_actualizado['performed'] > 0
datos_proyecto_actualizado.loc[empezadas, 'duration']  = 100 * datos_proyecto.loc[empezadas, 'actual_duration'] / datos_proyecto.loc[empezadas, 'performed']
datos_proyecto_actualizado.loc[~empezadas, 'duration'] = datos_proyecto.loc[~empezadas, 'duration']

# %%
datos_proyecto_actualizado

# %%
gantt, dibujo = proyecto.gantt(datos_proyecto_actualizado,
                          'duration',
                          'names',
                          total=None,
                          holguras=True,
                          tikz=True, 
                          params=dict(
                             period_width=0.7,
                             critical_background_color='white!90!green',
                              critical_text_color='black',
                              
                          ),
              )
pinta_tikz(dibujo)

# %% [markdown]
# ##### Definimos un nuevo proyecto que consiste en las tareas (o parte de las tareas) que faltan por completar. Sobre este nuevo proyecto es sobre el que nos piden los cálculos.

# %%
datos_proyecto_simplificado = (datos_proyecto
                                  .query('performed != 100')
                                  .rename(columns={'duration':'durationBaseline'})
                                  .drop(['resources', 'standard_deviation', 'a', 'm', 'b', 'BAC', 'cost_cap_duration', 'AC'], axis=1)
                                  .copy()
                              )
datos_proyecto_simplificado['duration'] = datos_proyecto_simplificado['durationBaseline'] 

datos_proyecto_simplificado.loc[['E', 'F', 'G'], 'predecessors'] = '----'
datos_proyecto_simplificado.loc[['E', 'F', 'G'], 'cap_duration'] = 1
empezadas = datos_proyecto_simplificado['performed'] > 0
datos_proyecto_simplificado.loc[empezadas,:] = datos_proyecto_simplificado.eval('duration = (100 - performed) * actual_duration / performed')
#datos_proyecto_simplificado.drop(['actual_duration',  'performed'], axis=1, inplace=True)
datos_proyecto_simplificado

# %%
prelaciones_simplificado = PredecessorTable.from_dataframe_of_strings(datos_proyecto_simplificado, activity='activity', predecessor='predecessors')
proyecto_simplificado    = prelaciones_simplificado.create_project()

# %%
gantt, dibujo = proyecto.gantt(datos_proyecto_simplificado,
                          'duration',
                          'names',
                          total=None,
                          holguras=True,
                          tikz=True, 
                          params=dict(
                             period_width=0.7,
                             critical_background_color='white!90!green',
                              critical_text_color='black',
                              
                          ),
              )
pinta_tikz(dibujo)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| out-width: 100%
#| fig-cap: "Pert del resto del proyecto por completar"
_ = proyecto_simplificado.pert(filename="pert_2.0.2_simplificado.pdf", durations=datos_proyecto_simplificado['duration'])
proyecto_simplificado.display_pert_pasos(datos_proyecto_simplificado['duration'])

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ### Reducción de la duración del proyecto con mínimo incremento de coste: Ackoff y Sasieni de lo que falta por completar del proyecto

# %%
#| tab-width: 80%
#| tab-cap: "Algoritmo de Ackoff-Sassieni"

ackoff, best_option, duration, available = proyecto_simplificado.ackoff(durations     = datos_proyecto_simplificado['duration'],
                            min_durations = datos_proyecto_simplificado['cap_duration'],
                            costs         = datos_proyecto_simplificado['ucr'],
                            reduction=50
                            )

# %%
proyecto_simplificado.display_ackoff_pasos_tikz(
    durations     = datos_proyecto_simplificado['duration'],
    min_durations = datos_proyecto_simplificado['cap_duration'],
    costs         = datos_proyecto_simplificado['ucr'],
    reduction=50)

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ### Montecarlo para lo que falta por completar del proyecto

# %%
num_of_samples = 3

names_of_the_tasks = datos_proyecto.index
np.random.seed(2025)
durations_montecarlo = pd.DataFrame([np.random.normal(size = num_of_samples,
                                                       loc  = datos_proyecto.duration[task],
                                                       scale= datos_proyecto.standard_deviation[task])
                                      for task in names_of_the_tasks],
                                     index=names_of_the_tasks).T.round(1)

# %%
durations_montecarlo

# %% [markdown]
# #### Determinamos la matriz de caminos del proyecto.

# %%
proyecto = proyecto_simplificado.copy()
path_matrix = proyecto.path_matrix(dummies=False)
proyecto.display_path_matrix(dummies=False)

# %% [markdown]
# #### Cálculo de la duración del proyecto

# %%
datos_proyecto_restante = datos_proyecto.loc[path_matrix.columns,:]
restantes = datos_proyecto_restante.query('performed != 100')
durations_montecarlo = durations_montecarlo.reindex(restantes.index, axis=1)
durations_montecarlo = durations_montecarlo.eval('E = E * 0.5')
durations_montecarlo = durations_montecarlo.eval('F = F * 0.6')
durations_montecarlo = durations_montecarlo.eval('G = G * 0.5')
durations_montecarlo

# %%
paths_durations_per_iteration = durations_montecarlo @ path_matrix.T
paths_durations_per_iteration

# %%
project_duration_per_iteration = paths_durations_per_iteration.max(axis="columns")
project_duration_per_iteration.T

# %% [markdown]
# #### Determina la duración media y la desviación típica de la duración del proyecto.

# %%
project_duration_per_iteration.plot(kind='density');

# %%
project_duration_per_iteration.describe()

# %%
project_mean_duration = round(project_duration_per_iteration.mean(), 2)
project_std_deviation_duration = round(project_duration_per_iteration.std(), 2)

# %%
Markdown(f"La duración media del proyecto es: {project_mean_duration}, la desviación típica de la duración del proyecto es: {project_std_deviation_duration}")

# %% [markdown]
# #### ¿Cuál es la probabilidad de que el proyecto finalice en menos de 13 periodos de tiempo?
#
# Utilizando la distribución normal

# %%
period = 10
probability = norm.cdf(period, loc=project_mean_duration, scale=project_std_deviation_duration)
from pmule import display_normal_tikz
display_normal_tikz(project_mean_duration, project_std_deviation_duration, hasta=period)
Markdown(f"Para un proyecto con duración media {project_mean_duration} y desviación típica {round(project_std_deviation_duration, 2)} el valor de la probabilidad pedida es {round(probability * 100, 2)} por ciento.")

# %% [markdown]
# Utilizando los valores de Montecarlo

# %%
resultado = (project_duration_per_iteration <= period).mean()*100
Markdown(f"La probabilidad pedida es {resultado} por ciento")

# %% [markdown]
# #### ¿Cuál es la duración del proyecto para la cual podemos garantizar que el proyecto va finalizar antes de esa fecha con un 85% de probabilidad?
#
# Utilizando la distribución normal

# %%
resultado = norm.ppf(0.85, loc=project_mean_duration, scale=project_std_deviation_duration)
from pmule import display_normal_tikz
display_normal_tikz(project_mean_duration, project_std_deviation_duration, percentil=0.85)
Markdown(f"{resultado}")

# %% [markdown]
# Utilizando los valores de Montecarlo:

# %% [markdown]
# Podemos ordenar los valores en sentido ascendente y quedarnos con aquél valor superior al 85% de las muestras.

# %%
resultado_1 = project_duration_per_iteration.sort_values(ascending=True).iloc[int(0.85*num_of_samples)]
Markdown(f"{resultado_1}")

# %% [markdown]
# O bien, utilizando la función `quantile`

# %%
resultado_2 = project_duration_per_iteration.quantile(q=0.85, interpolation='higher')
Markdown(f"{resultado_2}")

# %%

# %%
