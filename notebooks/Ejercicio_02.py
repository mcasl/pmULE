# ---
# title: Ejercicio 02
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
import pandas as pd
import numpy as np


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

periodo_probabilidad_tau = 10
limite_maximo_recursos = [4]*3 + [6]
pregunta_1 = "1. Determine para cada actividad su duración optimista. ¿Influye esta duración optimista en la duración mínima del proyecto?  "
pregunta_2 = "2. Determine la duración media del proyecto y su varianza utilizando para ello la aproximación del teorema del límite central.  "
pregunta_3 = f"3. Determine la probabilidad de que el proyecto se termine antes de {periodo_probabilidad_tau} periodos.  "
pregunta_4 = "4. Planifique el proyecto teniendo en cuenta que durante las primeras tres semanas dispondrá de cuatro trabajadores y a partir de la cuarta de seis trabajadores.  "
pregunta_5 = "5. ¿Sería suficiente disponer de 4 recursos durante todo el proyecto? ¿Por qué?  "


# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))
display(Markdown(pregunta_5))

# %%
datos_enunciado = pd.DataFrame([
#  nombre, anteriores, duracion, duracion_pesimista, duracion_modal, recursos
     ('A',       '---  ',     2,         2,                    2,           1  ),
     ('B',       'A    ',     2,         3,                    2,           2  ),
     ('C',       'B    ',     3,         4,                    3,           2  ),
     ('D',       'B,J  ',     4,         5,                 17/4,           1  ),
     ('E',       'F,B,J',     1,         1,                    1,           1  ),
     ('F',       'A    ',     2,         3,                    2,           2  ),
     ('G',       'C,D  ',     1,         1,                    1,           1  ),
     ('H',       'E    ',     4,         5,                    4,           1  ),
     ('I',       'B,J  ',     1,         1,                    1,           1  ),
     ('J',       'A    ',     2,         2,                    2,           1  ),
] ,columns=['activity', 'predecessor', 'duration', 'pesimistic_duration', 'modal_duration', 'resources'])
datos_enunciado.set_index('activity', inplace=True)
datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1
#

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Si entendemos la duración mínima de un proyecto como la duración media obtenida al utilizar para su cálculo las duraciones tope de las actividades, la duración optimista no causa efecto directo en esta duración. La duración máxima o Pert de una actividad, y su duración tope, son valores medios para los que a su vez podemos entender que existen valores optimístas y pesimístas de la duración. No debemos confundir el cambio en la duración de una actividad provocado de manera consciente mediante el cambio del nivel de esfuerzo aplicado, calidad de herramientas, equipos, técnicas; con la observación de que en un caso concreto la duración resulte ser aleatoriamente mayor o menor.

# %% [markdown]
# En lo relativo al cálculo solicitado, A partir de los datos del enunciado y de la relación $D=\frac{a+4m+b}{6}$ obtenemos la duración optimista $a=6D-b-4m$.

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['optimistic_duration'] = datos_proyecto.eval('6*duration - pesimistic_duration - 4*modal_duration')
datos_proyecto[['optimistic_duration']]

# %% [markdown]
# ## Apartado 2
#

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# ### Cuadro de prelaciones

# %% [markdown]
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto = prelaciones.create_project()
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
duraciones_medias = datos_proyecto['duration']
resultados_pert = proyecto.calculate_pert(durations=duraciones_medias)
resultados_pert['nodes']

# %%
duracion_media_proyecto = int(proyecto.duration(durations=duraciones_medias))

# %%
Markdown(f"La información del nodo final del proyecto nos informa de la duración media del proyecto: {duracion_media_proyecto} periodos")

# %% [markdown]
# ### Calendario del proyecto
# Las fechas de inicio y fin más tempranas y tardías de cada actividad son:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ### Otras representaciones alternativas
# Podemos igualmente representar el cronograma del proyecto bien con un diagrama de Gantt, o un grafo Roy.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto"
#| label: fig-gantt
gantt_df, dibujo = proyecto.gantt(datos_proyecto,
                                  duration_label='duration',
                                  tikz=True,
                                  params=dict(
                                     critical_background_color='white!90!green',
                                     regular_background_color='white!90!green',
                                     period_width=1.5,
                                 )
                            )
pinta_tikz(dibujo)

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-Roy
proyecto.roy(ranksep=1.25, nodesep=1.25)

# %% [markdown]
# ### Holgura total de las actividaes
# Para determinar la varianza de la duración del proyecto es necesario determinar la varianza de cada rama del camino crítico. Para ello nos hará falta calcular la holgura total de cada actividad. Aquellas actividades con holgura total cero formarán el camino crítico. 

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
resultados_pert['activities'][['H_total']]

# %% [markdown]
# Las actividades con holgura total igual a cero forman el camino crítico.

# %% [markdown]
# ### Diagrama de red con indicación del camino crítico

# %% [markdown]
# El proyecto dado puede ser representado utilizando distintas técnicas. A continuación se muestra el grafo Pert del mismo.

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación del camino crítico"
#| label: fig-pertrutasCriticas
proyecto.display_pert_pasos(duraciones_medias)

# %% [markdown]
# Las rutas del camino crítico son en este caso:

# %%
#| tbl-cap: "Rutas del camino crítico"
#| label: tbl-rutasCriticas
camino_critico = proyecto.critical_path(durations=duraciones_medias)
for key, value in camino_critico.items():
    print(f"{key}: ", ", ".join(value))

# %% [markdown]
# Alternativamente, podríamos haber representado el proyecto a traves de su grafo Roy. Se muestra a continuación con indicación de las actividades críticas:

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación de las actividades críticas"
#| label: fig-RoyCríticas
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ### Varianza del proyecto

# %% [markdown]
# Para determinar la varianza del proyecto es necesario conocer la varianza de cada una de las rutas del camino crítico, y para ello la varianza de las actividades que las componen.
# La varianza de cada actividad no es aportada por el enunciado como dato, pero sí se puede calcular a partir de la expresión $\sigma^2 = \left(\frac{b-a}{6} \right)$

# %%
#| tbl-cap: "Cálculo de la varianza de la duración de las actividades"
#| label: tbl-varianzas
datos_proyecto['variance'] = (((datos_proyecto['pesimistic_duration'] - datos_proyecto['optimistic_duration'])/6)**2).round(2)
datos_proyecto[['pesimistic_duration', 'optimistic_duration', 'variance']]

# %% [markdown]
# Conocidos estos valores podemos calcular, mediante el teorema del límite central, la varianza de cada ruta como suma de las varianzas de sus actividades.

# %%
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations = datos_proyecto['duration'],
                                                  variances = datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_caminos

# %%
display(Markdown(f"""La varianza de la duración del proyecto es la máxima varianza de las rutas críticas. Para este proyecto es {round(desviacion_proyecto**2,2)}.  
Habitualmente será útil calcular su raiz cuadrada, la desviación típica, para ser utilizada como parámetro de la ley de distribución normal.  """))
display(Markdown(f"La desviación típica es {desviacion_proyecto}"""))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_tau = norm.cdf(periodo_probabilidad_tau, loc=duracion_media_proyecto, scale=desviacion_proyecto)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, hasta=periodo_probabilidad_tau)
Markdown(f"Para un proyecto con duración media {duracion_media_proyecto} y desviación típica {round(desviacion_proyecto, 2)} el valor de la probabilidad de terminar antes de {periodo_probabilidad_tau} periodos es {round(probabilidad_de_terminar_antes_de_tau * 100, 2)} por ciento.")

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# ### Análisis de la situación inicial

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
datos_proyecto[['duration',
                'pesimistic_duration',
                'resources',
                'optimistic_duration']] = datos_proyecto[['duration', 'pesimistic_duration', 
                                                            'resources', 'optimistic_duration']].astype(int)
proyecto_asignacion = proyecto.copy()
gantt_df, dibujo = proyecto.gantt(datos_proyecto,
                      duration_label='duration',
                      resource_label='resources',
                      holguras=True,
                      total='fila',
                      tikz=True,
                      params=dict(
                        period_width=1.5,
                        row_height=1,
                        #background_horizontal_line_color='gray',
                        #background_vertical_bars_color='white!92!cyan',
                        activity_relative_height=0.6,
                        critical_background_color='white!90!red',
                        regular_background_color='white!90!green',
                        regular_text_color='black',
                        critical_text_color='black',
                        activity_inner_text_style=r"\bfseries\large",),
                    )
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)

# %% [markdown]
# La demanda de recursos actual infringe el límite impuesto en el enunciado. Es necesario aplicar el algoritmo de asignación. Representamos a continuación la nueva planificación atendiendo a los cambios introducidos de acuerdo al algoritmo de asignación.

# %% [markdown]
# ### Algoritmo de asignación de Wiest-Levy

# %%
#| fig-cap: "Planificación del proyecto tras aplicar el algoritmo de asignación"
#| label: fig-asignacion46
proyecto_asignacion = proyecto.copy()
datos_proyecto_tras_asignacion, gantt_df, dibujo = proyecto_asignacion.asignar(data=datos_proyecto,
                                              duration_label='duration',
                                              resource_label='resources',
                                              maximo = [4,4,4,6],
                                              report = False,
                                              params = dict(
                                                  period_width=1.5
                                                  )
                                              )
pinta_tikz(dibujo)

# %% [markdown]
# La zona sombreada indica el desplazamiento realizado en las actividades. De acuerdo a los cambios introducidos el proyecto ya es compatible con las restricciones impuestas a los recursos.

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# Se analiza ahora la posibilidad de reprogramar el proyecto atendiendo al criterio de que el consumo de recursos no supere el  límite de 4 recursos a lo largo de todo el proyecto impuesto por el enunciado del ejercicio. El siguiente diagrama de Gantt muestra la nueva planificación obtenida al aplicar el algoritmo de asignación con el límite impuesto.

# %%
#| fig-cap: "Planificación del proyecto tras aplicar el algoritmo de asignación con el nuevo límite"
#| label: fig-asignacion444
proyecto_asignacion = proyecto.copy()
datos_proyecto_tras_asignar, gantt_df, dibujo = proyecto_asignacion.asignar(data=datos_proyecto,
                                              duration_label='duration',
                                              resource_label='resources',
                                              maximo = [4,4,4],
                                              report = False,
                                              params = dict(
                                                  period_width=1.5
                                                  )
                                              )
pinta_tikz(dibujo)

# %% [markdown]
# En estas condiciones el proyecto ya es viable con 4 recursos. No obstante su duración se ha prolongado hasta 11 periodos.

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras la nueva configuración"
#| label: fig-postAsignacionBarras2
plot_cargas(gantt_df, 4)

# %%

# %%
