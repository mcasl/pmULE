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
from pmule import (explica_camino_critico, explica_probabilidad, explica_percentil, explica_sigma_ruta, explica_suma,
                   ordena_ruta, plural, lista_y, limpia_tabla)

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
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
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
Markdown(f"La suma de los costes es {int(datos_enunciado['BAC'].sum())}: "
         + " + ".join(str(v) for v in datos_enunciado['BAC']) + f" = {int(datos_enunciado['BAC'].sum())}.\n\n"
         f"**Respuesta:** el coste del proyecto con la duración planificada es **{int(datos_enunciado['BAC'].sum())}** unidades monetarias.")

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
display(Markdown("Fórmulas (estimación PERT con a optimista, m más probable y b pesimista): duración media $d = (a + 4m + b)/6$, "
                 "desviación típica $\\sigma = (b - a)/6$, varianza $\\sigma^2$ y coste unitario de reducción "
                 "ucr = (coste a duración tope − coste planificado) / (d − duración tope):"))
display(pd.DataFrame({
    'd = (a + 4m + b)/6': [f"({r.a:g} + 4×{r.m:g} + {r.b:g})/6 = {r.duration:g}" for r in datos_proyecto.itertuples()],
    'σ = (b − a)/6': [f"({r.b:g} − {r.a:g})/6 = {r.standard_deviation:.3f}" for r in datos_proyecto.itertuples()],
    'σ²': [f"{r.variance:.3f}" for r in datos_proyecto.itertuples()],
    'ucr': [f"({r.cost_cap_duration} − {r.BAC})/({r.duration:g} − {r.cap_duration}) = {r.ucr:g}" for r in datos_proyecto.itertuples()],
}, index=datos_proyecto.index))

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
display(Markdown("Diagrama de Gantt con las duraciones medias, cada actividad en su fecha más temprana; los números son los "
                 "recursos por periodo, la fila *Total* la carga del proyecto en cada periodo y las barras finas, las holguras."))

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
proyecto.display_roy_construccion(tabla=prelaciones)

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
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas, cuyas actividades se listan en orden de ejecución:

# %%
display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'periodos'))

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
display(Markdown("**Respuesta:** el camino crítico es "
                 + lista_y([' – '.join(ordena_ruta(r, proyecto.calendar(datos_proyecto['duration'])))
                            for r in proyecto.critical_path(datos_proyecto['duration']).values()])
                 + f", con una duración de {proyecto.duration(datos_proyecto['duration']):g} periodos; el calendario de ejecución "
                 "es la tabla anterior (inicios y fines más tempranos y más tardíos de cada actividad)."))

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
_rutas_criticas = [ordena_ruta(r, proyecto.calendar(datos_proyecto['duration'])) for r in proyecto.critical_path(datos_proyecto['duration']).values()]
Markdown(f"La desviación típica de la duración del proyecto es la máxima desviación típica de todas las rutas críticas. Para este proyecto, haciendo la raíz cuadrada del valor de la varianza, tenemos el valor de desviación típica  {round(desviacion_proyecto,2)}: "
         + "; ".join(explica_sigma_ruta(r, datos_proyecto['variance']) for r in _rutas_criticas) + ".")

# %% [markdown]
# #### Calculo de la probabilidad de que el proyecto finalice antes del periodo indicado

# %% [markdown]
# Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad:

# %%
probability_TLC = norm.cdf(periodo_a_probabilidad_normal, loc=duracion_media_proyecto, scale=desviacion_proyecto)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, hasta=periodo_a_probabilidad_normal)
Markdown(f"""Utilizando, por ejemplo, una calculadora o una hoja de cálculo, para un proyecto con duración media {duracion_media_proyecto} y desviación típica {round(desviacion_proyecto, 2)}:

{explica_probabilidad(duracion_media_proyecto, desviacion_proyecto, periodo_a_probabilidad_normal)}

**Respuesta:** la probabilidad de que el proyecto finalice antes de {periodo_a_probabilidad_normal} periodos **es {round(probability_TLC * 100, 2)} por ciento**
(prácticamente imposible: {periodo_a_probabilidad_normal} periodos está {(duracion_media_proyecto - periodo_a_probabilidad_normal) / desviacion_proyecto:.1f} desviaciones típicas por debajo de la media).""")

# %% [markdown]
# #### Cálculo de la duración del proyecto para la cual se espera que el proyecto haya finalizado antes de esa fecha con la probabilidad requerida

# %%
fecha_exito_normal = round(norm.ppf(probabilidad_exito_normal, loc=duracion_media_proyecto, scale=desviacion_proyecto), 1)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, percentil=probabilidad_exito_normal)
Markdown(f"Con una calculadora o una hoja de cálculo: "
         + explica_percentil(duracion_media_proyecto, desviacion_proyecto, probabilidad_exito_normal, 'periodos')
         + f".\n\n**Respuesta:** el número de periodos necesarios para alcanzar una tasa de éxito del {probabilidad_exito_normal*100:g} por ciento es **{fecha_exito_normal}** periodos.")

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
display(Markdown(f"Se simulan {num_of_samples} iteraciones: en cada una (fila) la duración de cada actividad se genera con una "
                 f"normal de media d y desviación típica σ (tabla del apartado 2), redondeada a un decimal."))
display(durations_montecarlo)

# %% [markdown]
# ### Determinamos la matriz de caminos del proyecto.

# %%
path_matrix = proyecto.path_matrix(dummies=False)
proyecto.display_rutas("pert")

# %% [markdown]
# ### Cálculo de la duración del proyecto

# %%
paths_durations_per_iteration = durations_montecarlo @ path_matrix.T
_cal_medio = proyecto.calendar(datos_proyecto['duration'])
_rutas_mc = {r: ordena_ruta(list(path_matrix.columns[path_matrix.loc[r] == 1]), _cal_medio) for r in path_matrix.index}
display(Markdown("Multiplicando las duraciones simuladas por la matriz de caminos (traspuesta) se obtiene la duración de cada ruta "
                 "en cada iteración. Por ejemplo, en la iteración 0:\n\n"
                 + "\n".join(f"- {r} ({' – '.join(a)}): {' + '.join(format(durations_montecarlo.loc[0, x], 'g') for x in a)} = "
                              f"{paths_durations_per_iteration.loc[0, r]:.1f}" for r, a in _rutas_mc.items())))
display(paths_durations_per_iteration)

# %%
project_duration_per_iteration = paths_durations_per_iteration.max(axis="columns")
display(Markdown(f"La duración del proyecto en cada iteración es la de su ruta más larga; en la iteración 0: "
                 f"máx({', '.join(format(v, '.1f') for v in paths_durations_per_iteration.loc[0])}) = "
                 f"{project_duration_per_iteration[0]:.1f} periodos."))
display(project_duration_per_iteration.T)

# %% [markdown]
# ### Determina la duración media y la desviación típica de la duración del proyecto.

# %%
from pmule import display_histograma_tikz
display_histograma_tikz(project_duration_per_iteration)

# %%
project_duration_per_iteration.describe()

# %%
project_mean_duration = project_duration_per_iteration.mean()
project_std_deviation_duration = project_duration_per_iteration.std()

# %%
Markdown(f"Con las {num_of_samples} duraciones simuladas: media $\\bar x = \\sum x_i / n$ = {project_duration_per_iteration.sum():.1f} / "
         f"{num_of_samples} = {project_mean_duration:.2f}; desviación típica muestral $s = \\sqrt{{\\sum (x_i - \\bar x)^2 / (n-1)}}$ = "
         f"{project_std_deviation_duration:.2f}.\n\n"
         f"**Respuesta:** la duración media del proyecto es **{project_mean_duration:.2f}** periodos y la desviación típica, "
         f"**{project_std_deviation_duration:.2f}** periodos (muy parecidas a las de la aproximación normal: {duracion_media_proyecto} y "
         f"{desviacion_proyecto:.2f}).")

# %% [markdown]
# ### ¿Cuál es la probabilidad de que el proyecto finalice en menos de 13 periodos de tiempo (plazo del enunciado)?

# %% [markdown]
# #### Utilizando la distribución normal

# %%
period = 13
probability = norm.cdf(period, loc=project_mean_duration, scale=project_std_deviation_duration)
from pmule import display_normal_tikz
display_normal_tikz(project_mean_duration, project_std_deviation_duration, hasta=period)
Markdown(f"Para un proyecto con duración media {project_mean_duration:.2f} y desviación típica {round(project_std_deviation_duration, 2)} "
         f"(media y desviación de la simulación): {explica_probabilidad(round(project_mean_duration, 2), round(project_std_deviation_duration, 2), period)}. "
         f"El valor de la probabilidad pedida es {round(probability * 100, 2)} por ciento.")

# %% [markdown]
# #### Utilizando los valores de Montecarlo

# %%
probability_montecarlo = (project_duration_per_iteration <= period).mean()
from pmule import display_histograma_tikz
display_histograma_tikz(project_duration_per_iteration, project_mean_duration, project_std_deviation_duration, hasta=period)
Markdown(f"Considerando la secuencia de valores obtenidos por el método de MonteCarlo, el valor de la probabilidad pedida es {round(probability_montecarlo * 100, 2)} por ciento: "
         f"iteraciones con duración ≤ {period} / total = {int((project_duration_per_iteration <= period).sum())} / {num_of_samples} "
         f"(la menor duración simulada es {project_duration_per_iteration.min():.1f} periodos).\n\n"
         f"**Respuesta:** la probabilidad de terminar en menos de {period} periodos es {round(probability_montecarlo * 100, 2)} % "
         f"(prácticamente nula).")

# %% [markdown]
# ### ¿Cuál es la duración del proyecto para la cual podemos garantizar que el proyecto va finalizar antes de esa fecha con un 85% de probabilidad?

# %% [markdown]
# #### Utilizando la distribución normal

# %%
prob=0.85
from pmule import display_normal_tikz
display_normal_tikz(project_mean_duration, project_std_deviation_duration, percentil=prob)
display(Markdown(explica_percentil(round(project_mean_duration, 2), round(project_std_deviation_duration, 2), prob, 'periodos') + "."))

# %% [markdown]
# #### Utilizando los valores de Montecarlo:

# %% [markdown]
# Podemos ordenar los valores en sentido ascendente y quedarnos con aquél valor superior al 85% de las muestras.

# %%
from pmule import display_histograma_tikz
display_histograma_tikz(project_duration_per_iteration, project_mean_duration, project_std_deviation_duration, percentil=prob)
display(Markdown(f"Con las {num_of_samples} duraciones ordenadas de menor a mayor, se toma la de la posición "
                 f"{prob} × {num_of_samples} + 1 = {int(prob * num_of_samples) + 1}: "
                 f"**{project_duration_per_iteration.sort_values(ascending=True).iloc[int(prob*num_of_samples)]:.1f}** periodos."))

# %% [markdown]
# O bien, utilizando la función `quantile`

# %%
Markdown(f"`quantile(q={prob})` = {project_duration_per_iteration.quantile(q=prob, interpolation='higher'):.1f} periodos.\n\n"
         f"**Respuesta:** con un {prob:.0%} de probabilidad el proyecto termina antes de "
         f"**{project_duration_per_iteration.quantile(q=prob, interpolation='higher'):.1f}** periodos según la simulación "
         f"({norm.ppf(prob, loc=project_mean_duration, scale=project_std_deviation_duration):.2f} con la aproximación normal).")

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
_T0 = proyecto.duration(datos_proyecto['duration'])
_filas, _acum = [], 0
for _k, _acts in best_option.items():
    _c = float(datos_proyecto.loc[_acts, 'ucr'].sum())
    _acum += _c
    _filas.append({'iteración': _k, 'se acortan 1 periodo': lista_y(_acts),
                   'coste': ' + '.join(format(x, 'g') for x in datos_proyecto.loc[_acts, 'ucr']) + (f" = {_c:g}" if len(_acts) > 1 else ''),
                   'duración': f"{_T0 - _k - 1:g}", 'coste acumulado': f"{_acum:g}"})
display(Markdown(f"Se pide reducir hasta 20 periodos, pero el algoritmo se detiene tras {plural(len(best_option), 'iteración', 'iteraciones')}, "
                 f"cuando alguna ruta crítica ya no tiene actividades que puedan acortarse. Resumen de las iteraciones:"))
display(pd.DataFrame(_filas).set_index('iteración'))
display(Markdown(f"**Resultado:** la duración mínima alcanzable es {proyecto.duration(durations):g} periodos (desde {_T0:g}), con un "
                 f"incremento de coste de {_acum:g} unidades monetarias; duraciones finales: "
                 + ", ".join(f"{a} = {durations[a]:g}" for a in durations.index) + "."))

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
_desp = desplazamientos.iloc[:, 0]
_ss_ini = float(cuadrados.data.iloc[0, 0])
_ss_fin = float(pd.to_numeric(gantt_df.data.loc['Cuadrados'].drop('H_total', errors='ignore'), errors='coerce').sum())
_movidas = _desp[_desp > 0]
display(Markdown(
    "La nivelación busca la programación que minimiza la suma de los cuadrados de la carga de cada periodo (misma duración del "
    "proyecto, desplazando actividades dentro de su holgura). Se recorren las actividades con holgura y, para cada una, se elige el "
    "desplazamiento con menor suma de cuadrados.\n\n"
    f"**Resultado:** " + (lista_y([f"{a} se retrasa {plural(int(d), 'periodo')}" for a, d in _movidas.items()]) if len(_movidas) else
                          "no se desplaza ninguna actividad")
    + f"; las demás ({lista_y([a for a in _desp.index if _desp[a] == 0])}) se quedan en su fecha más temprana. "
    f"La suma de cuadrados de las cargas pasa de {_ss_ini:g} a {_ss_fin:g}."))

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Asignación

# %% [markdown]
# ##### El número máximo de recursos es 10

# %%
proyecto_asignacion = proyecto.copy()
proyecto_asignacion.display_asignar_pasos_tikz(data=datos_proyecto, duration_label='duration', resource_label='resources', maximo=[10])
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
_cal0 = proyecto.calendar(datos_proyecto['duration'])
_gd = gantt_df.data
_per = [c for c in _gd.columns if str(c) != 'H_total']
_ini = {a: int(min(c for c in _per if pd.notna(pd.to_numeric(_gd.loc[a, c], errors='coerce')))) - 1 for a in datos_proyecto.index}
_ret = {a: _ini[a] - int(_cal0.loc[a, 'inicio_mas_temprano']) for a in datos_proyecto.index}
_carga = pd.to_numeric(_gd.loc['Total', _per], errors='coerce').fillna(0)
_fin = max(int(c) for c in _per if _carga[c] > 0)
display(Markdown(
    "**Resultado:** para no superar 10 recursos por periodo hay que retrasar "
    + lista_y([f"{a} {plural(r, 'periodo')} (empieza en {_ini[a]} en lugar de {int(_cal0.loc[a, 'inicio_mas_temprano'])})"
               for a, r in _ret.items() if r > 0])
    + f". La carga máxima pasa a ser {_carga.max():g} y el proyecto termina en el periodo {_fin}, "
    + (f"{plural(_fin - int(_T0), 'periodo')} más tarde que sin límite de recursos ({int(_T0)})." if _fin > _T0 else
       "sin alargarse.")))

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
_t_vg = int(acumulados.index[-1])
_dp = datos_proyecto
Markdown(f"""El punto de control es el fin del calendario real (mismo grafo con las duraciones reales): periodo {_t_vg}. En esa fecha:

- PV (valor planificado) = coste planificado de lo que, según el plan, debía estar hecho en el periodo {_t_vg} (suma hasta esa columna del Gantt PV) = **{round(PV,0):g}**
- EV (valor ganado) = Σ coste planificado × % realizado = {' + '.join(f"{b}×{p/100:g}" for b, p in zip(_dp['BAC'], _dp['performed']) if p > 0)} = **{EV:g}**
- AC (coste real) = {' + '.join(str(v) for v in _dp['AC'] if v > 0)} = **{AC:g}**

Utilizando: \\
PV={round(PV,0)} \\
AC={AC} \\
EV={EV} \\
y el BAC={BAC} obtenido sumando los costes planificados, obtenemos (el LRE, estimación del contratista del coste final, se toma igual a 35000):
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
display(Math((latex_code)))
_CPI, _SPI = EV / AC, EV / PV
display(Markdown(
    f"**Interpretación:** CPI = {_CPI:.3f} {'≥ 1: el trabajo hecho ha costado algo menos de lo presupuestado' if _CPI >= 1 else '< 1: sobrecoste'}"
    f" (CV = {EV - AC:g}); SPI = {_SPI:.3f} {'< 1: el proyecto va retrasado' if _SPI < 1 else '≥ 1: no hay retraso'} "
    f"(SV = {EV - PV:g}: se ha hecho el {_SPI:.0%} de lo planificado). Si se mantiene el CPI, el coste final será "
    f"EAC = {BAC / _CPI:.0f}."))

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
_curso = datos_proyecto.query('0 < performed < 100')
display(Markdown(
    "Actividades en curso (a ritmo de lo realizado): duración total prevista = 100 × duración real / % realizado:\n\n"
    + "\n".join(f"- {a}: 100 × {r.actual_duration} / {r.performed} = **{100 * r.actual_duration / r.performed:g}** periodos "
                 f"(planificada {r.duration:g})" for a, r in _curso.iterrows())
    + "\n\nLas terminadas toman su duración real ("
    + ", ".join(f"{a} = {r.actual_duration}" for a, r in datos_proyecto.query('performed == 100').iterrows())
    + ") y las no empezadas, la planificada (" + ", ".join(f"{a} = {r.duration:g}" for a, r in datos_proyecto.query('performed == 0').iterrows())
    + ")."))
display(datos_proyecto_actualizado[['actual_duration', 'performed', 'duration']])

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
_T_prev = proyecto.duration(datos_proyecto_actualizado['duration'])
display(Markdown(f"Con estas duraciones el proyecto terminaría en el periodo **{_T_prev:g}**, frente a los "
                 f"{duracion_media_proyecto} planificados: {plural(_T_prev - duracion_media_proyecto, 'periodo')} de retraso."))

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
_en_curso = list(datos_proyecto_simplificado.index[empezadas])
display(Markdown(
    f"Se quitan las actividades terminadas. Las que están en curso ({lista_y(_en_curso)}) pasan a no tener precedentes (las suyas ya "
    f"terminaron), se les asigna una duración tope de 1 periodo y su duración es la que les falta, a ritmo de lo realizado: "
    f"(100 − % realizado) × duración real / % realizado:\n\n"
    + "\n".join(f"- {a}: (100 − {r.performed}) × {r.actual_duration} / {r.performed} = **{r.duration:g}**"
                 for a, r in datos_proyecto_simplificado.loc[_en_curso].iterrows())
    + "\n\nLas no empezadas conservan su duración media y sus prelaciones."))
display(datos_proyecto_simplificado)

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
display(explica_camino_critico(proyecto_simplificado, datos_proyecto_simplificado['duration'], 'periodos'))
display(Markdown(f"Es lo que falta desde el punto de control (periodo {_t_vg}): el proyecto terminaría en "
                 f"{_t_vg} + {proyecto_simplificado.duration(datos_proyecto_simplificado['duration']):g} = "
                 f"{_t_vg + proyecto_simplificado.duration(datos_proyecto_simplificado['duration']):g}"
                 + (f", que coincide con la previsión anterior." if _t_vg + proyecto_simplificado.duration(datos_proyecto_simplificado['duration']) == _T_prev else ".")))

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
_T0r = proyecto_simplificado.duration(datos_proyecto_simplificado['duration'])
_filas, _acum = [], 0
for _k, _acts in best_option.items():
    _c = float(datos_proyecto_simplificado.loc[_acts, 'ucr'].sum())
    _acum += _c
    _filas.append({'iteración': _k, 'se acortan 1 periodo': lista_y(_acts),
                   'coste': ' + '.join(format(x, 'g') for x in datos_proyecto_simplificado.loc[_acts, 'ucr']) + (f" = {_c:g}" if len(_acts) > 1 else ''),
                   'duración restante': f"{_T0r - _k - 1:g}", 'fin del proyecto': f"{_t_vg + _T0r - _k - 1:g}", 'coste acumulado': f"{_acum:g}"})
display(Markdown("Resumen de las iteraciones (el algoritmo se detiene cuando ya no puede acortarse alguna ruta crítica):"))
display(pd.DataFrame(_filas).set_index('iteración'))
display(Markdown(f"**Resultado:** lo que falta puede reducirse de {_T0r:g} a {proyecto_simplificado.duration(duration):g} periodos "
                 f"(fin en el periodo {_t_vg + proyecto_simplificado.duration(duration):g}) con un incremento de coste de {_acum:g}. "
                 f"Cada fila de la tabla da el coste de terminar en esa fecha."))

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
display(Markdown(f"Se simulan solo {plural(num_of_samples, 'iteración', 'iteraciones')} (ejemplo para hacerlo a mano; con tan pocas "
                 f"iteraciones los resultados son muy imprecisos), generando las duraciones como en el apartado 4:"))
display(durations_montecarlo)

# %% [markdown]
# #### Determinamos la matriz de caminos del proyecto.

# %%
proyecto = proyecto_simplificado.copy()
path_matrix = proyecto.path_matrix(dummies=False)
proyecto.display_rutas("pert")

# %% [markdown]
# #### Cálculo de la duración del proyecto

# %%
datos_proyecto_restante = datos_proyecto.loc[path_matrix.columns,:]
restantes = datos_proyecto_restante.query('performed != 100')
durations_montecarlo = durations_montecarlo.reindex(restantes.index, axis=1)
durations_montecarlo = durations_montecarlo.eval('E = E * 0.5')
durations_montecarlo = durations_montecarlo.eval('F = F * 0.6')
durations_montecarlo = durations_montecarlo.eval('G = G * 0.5')
display(Markdown(
    "Para las actividades en curso, la duración que falta se toma proporcional a la duración simulada: (1 − % realizado) × "
    "duración simulada ("
    + "; ".join(f"{a}: {1 - restantes.loc[a, 'performed'] / 100:g} × {a}" for a in restantes.index if restantes.loc[a, 'performed'] > 0)
    + "). *Nota*: esta hipótesis (proporcional a la duración planificada) no es la misma que la del cálculo determinista anterior "
    "(proporcional al ritmo real), por eso las duraciones restantes simuladas son menores."))
display(durations_montecarlo)

# %%
paths_durations_per_iteration = durations_montecarlo @ path_matrix.T
_cal_r = proyecto.calendar(datos_proyecto_simplificado['duration'])
_rutas_r = {r: ordena_ruta(list(path_matrix.columns[path_matrix.loc[r] == 1]), _cal_r) for r in path_matrix.index}
display(Markdown("Duración de cada ruta en cada iteración (suma de las duraciones de sus actividades); en la iteración 0:\n\n"
                 + "\n".join(f"- {r} ({' – '.join(a)}): {' + '.join(format(durations_montecarlo.loc[0, x], 'g') for x in a)} = "
                              f"{paths_durations_per_iteration.loc[0, r]:.2f}" for r, a in _rutas_r.items())))
display(paths_durations_per_iteration)

# %%
project_duration_per_iteration = paths_durations_per_iteration.max(axis="columns")
display(Markdown("Duración de lo que falta en cada iteración = máximo de sus rutas: "
                 + "; ".join(f"iteración {i}: máx({', '.join(format(v, '.2f') for v in paths_durations_per_iteration.loc[i])}) = "
                             f"{project_duration_per_iteration[i]:.2f}" for i in paths_durations_per_iteration.index)))
display(project_duration_per_iteration.T)

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
Markdown(f"Media = ({' + '.join(format(v, '.2f') for v in project_duration_per_iteration)}) / {num_of_samples} = {project_mean_duration}; "
         f"desviación típica muestral = √(Σ(x − media)² / (n − 1)) = {project_std_deviation_duration}.\n\n"
         f"**Respuesta:** La duración media de lo que falta del proyecto es: {project_mean_duration}, la desviación típica es: {project_std_deviation_duration} periodos.")

# %% [markdown]
# #### ¿Cuál es la probabilidad de que lo que falta del proyecto finalice en menos de 10 periodos de tiempo?
#
# Utilizando la distribución normal

# %%
period = 10
probability = norm.cdf(period, loc=project_mean_duration, scale=project_std_deviation_duration)
from pmule import display_normal_tikz
display_normal_tikz(project_mean_duration, project_std_deviation_duration, hasta=period)
Markdown(f"Para un proyecto con duración media {project_mean_duration} y desviación típica {round(project_std_deviation_duration, 2)}: "
         f"{explica_probabilidad(project_mean_duration, project_std_deviation_duration, period)}. "
         f"El valor de la probabilidad pedida es {round(probability * 100, 2)} por ciento.")

# %% [markdown]
# Utilizando los valores de Montecarlo

# %%
resultado = (project_duration_per_iteration <= period).mean()*100
Markdown(f"La probabilidad pedida es {resultado:.1f} por ciento: {int((project_duration_per_iteration <= period).sum())} de las "
         f"{num_of_samples} iteraciones duran {period} periodos o menos. La diferencia con la aproximación normal se debe a que "
         f"{plural(num_of_samples, 'iteración', 'iteraciones')} no bastan para estimar una probabilidad.")

# %% [markdown]
# #### ¿Cuál es la duración del proyecto para la cual podemos garantizar que el proyecto va finalizar antes de esa fecha con un 85% de probabilidad?
#
# Utilizando la distribución normal

# %%
resultado = norm.ppf(0.85, loc=project_mean_duration, scale=project_std_deviation_duration)
from pmule import display_normal_tikz
display_normal_tikz(project_mean_duration, project_std_deviation_duration, percentil=0.85)
Markdown(explica_percentil(project_mean_duration, project_std_deviation_duration, 0.85, 'periodos') + ".")

# %% [markdown]
# Utilizando los valores de Montecarlo:

# %% [markdown]
# Podemos ordenar los valores en sentido ascendente y quedarnos con aquél valor superior al 85% de las muestras.

# %%
resultado_1 = project_duration_per_iteration.sort_values(ascending=True).iloc[int(0.85*num_of_samples)]
Markdown(f"Ordenadas: {', '.join(format(v, '.2f') for v in project_duration_per_iteration.sort_values())}; posición "
         f"0.85 × {num_of_samples} + 1 = {int(0.85 * num_of_samples) + 1} → **{resultado_1:.2f}** periodos.")

# %% [markdown]
# O bien, utilizando la función `quantile`

# %%
resultado_2 = project_duration_per_iteration.quantile(q=0.85, interpolation='higher')
Markdown(f"`quantile(q=0.85)` = {resultado_2:.2f} periodos.\n\n"
         f"**Respuesta:** lo que falta del proyecto termina con un 85 % de probabilidad antes de {resultado_2:.2f} periodos según "
         f"la simulación ({norm.ppf(0.85, loc=project_mean_duration, scale=project_std_deviation_duration):.2f} con la aproximación normal); "
         f"con solo {num_of_samples} iteraciones, el percentil simulado es simplemente la mayor de las duraciones.")

# %%

# %%
