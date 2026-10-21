# ---
# title: Ejercicio 10
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
# Se pretende construir una planta desalinizadora de agua, con el fin de proporcionar agua potable a ciertas poblaciones en las que escasea. Usted, como encargado de la planificación de la ejecución de este proyecto, decide que la descomposición en tareas y sus precedencias es la que se muestra en la tabla adjunta, donde la duración se expresa en meses.  
#
# Tras un detenido análisis del proyecto en cuestión, usted ha llegado a la conclusión de que no es posible que la tarea J empiece nada más terminar la tarea A, sino que es absolutamente necesario que para que se pueda empezar a ejecutar la tarea J debe de haber transcurrido un mes después de terminar la tarea A. Se pide:  

# %%
pregunta_1 = "1. Dibujar el diagrama PERT del proyecto y caracteríce su duración mediante una distribución aleatoria Normal.  "
pregunta_2 = "2. Como no está conforme con la duración del proyecto que resulta, desea reducir la duración del proyecto en 1 mes con el mínimo incremento de coste posible. ¿Qué tareas deben ser reducidas?.  "
pregunta_3 = """3. Debido al racionamiento de combustible diesel que sufre la isla, durante el proyecto sólo será posible disponer de 250 litros de gasoil cada semana; y debido a la falta de previsión de los responsables del proyecto, que no solicitaron permiso para el almacenamiento de combustible, no es posible guardar el sobrante de una semana para consumirlo la siguiente. Por otro lado las necesidades de consumo de gasoil que usted planifica son las que se muestran en la tabla.  
- ¿Se encuentra bien nivelado el consumo de gasoil?.  
- ¿Se encuentra bien asignado?.  
- Si no es así ¿qué recomienda hacer para asignar correctamente el consumo de combustible?.  
- ¿Qué consecuencias traerá la asignación correcta del combustible en este proyecto en concreto?.  """

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

import pandas as pd
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes,        duracion, varianza, coste,   litros, duracion_tope
  ('A',       '---  ',               5,        2,    '600-60t',   210,    2),
  ('B',       '---  ',               5,        1,    '300-20t',   150,    2),
  ('C',       'B    ',               1,        1,    '100-30t',    50,    1),
  ('D',       'B    ',               5,        1,    '300-25t',    75,    1),
  ('E',       'B    ',               2,        1,    '200-30t',    25,    1),
  ('F',       'C,D,E',               1,        1,    '100-10t',    75,    1),
  ('G',       'D,E  ',               2,        1,    '200-10t',    75,    2),
  ('H',       'E    ',               2,        1,    '150-50t',     0,    1),
  ('I',       'K    ',               4,        1,    '500-50t',   100,    3),
  ('💤J',      'A    ',               1,        0,         None,     0,    1),
  ('J',       '💤J   ',               2,        1,    '100-10t',    50,    1),
  ('K',       'A    ',               3,        1,    '300-10t',    50,    3),
] ,columns=['actividad', 'precedentes', 'duracion', 'varianza', 'coste', 'litros', 'duracion_tope']).set_index('actividad')

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
prelaciones = PredecessorTable.from_dataframe_of_strings(data=datos_enunciado, activity='actividad', predecessor='precedentes')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto = prelaciones.create_project()
datos_proyecto = datos_enunciado.copy()
proyecto.pert(nodesep=0.6)

# %% [markdown]
# ### Matriz de Zaderenko

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_pasos(datos_proyecto['duracion'])

# %% [markdown]
# Los tiempos tempranos y tardios han resultado ser:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duracion'])
resultados_pert['nodes']

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duracion']))
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

# %% [markdown]
# El camino crítico consta de las siguientes rutas:

# %%
camino_critico = proyecto.critical_path(durations=datos_proyecto['duracion'])
for key, value in camino_critico.items():
    display(Markdown(f"{key}: \t " + ", ".join(value)))

# %% [markdown]
# ### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto, con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duracion'])
calendario

# %% [markdown]
# ### Grafo PERT con indicación de tiempos y rutas del camino crítico de acuerdo a la planificación inicial
# A continuación se muestra el grafo Pert del proyecto, con indicación de las actividades críticas:

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación del camino crítico"
#| label: fig-pertrutasCriticas
proyecto.display_pert_pasos(datos_proyecto['duracion'])

# %% [markdown]
# ### Caracterización de la duración del proyecto mediante una distribución Normal
# La caracterización de la distribución Normal consiste en determinar los valores de sus parámetros: media y desviación típica de su duración.

# %%
duracion_media_proyecto = int(proyecto.duration(datos_proyecto['duracion']))

# %%
Markdown(f"La duración media del proyecto ya la tenemos calculada: {duracion_media_proyecto} periodos")

# %% [markdown]
# Para determinar la desviación típica debemos obtener en cada rama del camino crítico la suma de las varianzas, y quedarnos con la mayor.

# %%
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duracion'],
                                                   variances=datos_proyecto['varianza'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_rutas

# %%
Markdown(f"El proyecto sigue una distribución normal con media {duracion_media_proyecto} periodos y desviación típica {desviacion_proyecto}.")


# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Para resolver la reducción del proyecto con mínimo coste es necesario obtener el valor del coste unitario de reducción a partir de los datos del enunciado.

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto[['coste']]

# %% [markdown]
# Para obtener el coste unitario de reducción debemos quedarnos únicamente con el coeficiente que multiplica a la variable duración 't'.

# %%
datos_proyecto['cur'] = ( datos_proyecto.loc[:, 'coste']
                                        .str.split('[-+]', expand=True)[1]
                                        .str.split('t', expand=True)[0]
                                        .astype(float))
datos_proyecto[['cur']].T

# %% [markdown]
# Se aborda la solución de este apartado mediante la técnica de reducción de la duración del proyecto con mínimo coste.

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duracion'],
                min_durations=datos_proyecto['duracion_tope'],
                costs=datos_proyecto['cur'],
                reduction=1)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos(
    durations=datos_proyecto['duracion'],
    min_durations=datos_proyecto['duracion_tope'],
    costs=datos_proyecto['cur'],
    reduction=1)

# %% [markdown]
# En la Ruta 5 (B-D-G) la actividad más barata es la G pero ya está en su duración tope. La siguiente más barata es la actividad B que dispone de 3 periodos de reducción.
#
# Se propone reducir la B y la I.

# %% [markdown]
# ## Apartado 3.

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Consideramos el coste mensual cuatro veces el semanal:

# %%
#| tbl-cap: "Cuadro de coste mensual del combustible"
#| label: tbl-costeMensual
datos_proyecto['litros_por_mes'] = 4 * datos_enunciado['litros']   # Se considera el gasto mensual 4 veces el coste semanal
datos_proyecto[['litros_por_mes']].T

# %%
gantt_recursos, dibujo = proyecto.gantt(datos_proyecto,
                                'duracion',
                                'litros_por_mes',
                                total='fila',
                                holguras=True,
                                tikz=True)

# %% [markdown]
# De acuerdo con la planificación inicial, el consumo de combustible resulta ser:

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
pinta_tikz(dibujo)

# %% [markdown]
# La zona rayada antes de J representa la espera de 1 periodo mencionada en el enunciado.

# %% [markdown]
# La fila "Total" obtenida, que podemos representar gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_recursos, limite_maximo_recursos=1000)

# %% [markdown]
# La demanda del recurso no se encuentra nivelada.
# Además, con el máximo disponible de 1000 litros por mes el proyecto no es ejecutable en su estado actual y debe aplicarse el algoritmo de asignación.

# %% [markdown]
# ### Algoritmo de asignación

# %% [markdown]
# Durante los primeros seis periodos no es viable ejecutar A y B. Como ambas son críticas y de misma duración resulta indiferente desplazar una u otra pues en ambos casos esta acción conlleva un retraso del proyecto. Resolvemos el ejercicio desplazando la actividad B.

# %%
proyecto_asignacion = proyecto.copy()
duracion_y_recursos = datos_proyecto[['duracion', 'litros_por_mes']]
datos_proyecto_tras_asignar, gantt_df, dibujo  = proyecto_asignacion.asignar(
                                    data=duracion_y_recursos,
                                    duration_label='duracion',
                                    resource_label='litros_por_mes',
                                    maximo=[1000],
                                    report=False
                                    )


# %%
#| fig-cap: "Diagrama de Gantt del proyecto una vez aplicado el algoritmo de asignación"
#| label: fig-postAsignacion
pinta_tikz(dibujo)

# %% [markdown]
# La zona rayada representa el desplazamiento de la tarea _B_.

# %%
#| fig-cap: "Diagrama de cargas tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos=1000)

# %% [markdown]
# Con esta única acción la demanda de recursos del proyecto ya no supera las 1000 unidades que nos ponía de límite en el enunciado. La duración no obstante ha subido a 17 meses y el camino crítico ha cambiado.

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
proyecto_asignacion.display_pert_pasos(datos_proyecto_tras_asignar['duracion'])

# %%
camino_critico = proyecto_asignacion.critical_path(durations=datos_proyecto_tras_asignar['duracion'])
for key, value in camino_critico.items():
    display(Markdown(f"{key}: \t " + ", ".join(value)))

# %% [markdown]
# Este resultado cumple los requerimientos expresados en el enunciado. No obstante, resultará sorprendente que la actividad 💤B sea crítica al inicio del proyecto cuando representa una espera, mientras la actividad A dispone de holgura total 5 5 y no es crítica.
# Realmente, al imponer el enunciado en este apartado un límite máximo de recursos que impide que A y B puedan ejecutarse simultaneamente, ha sido equivalente a añadir una restricción de prelación entre A y B que obliga que B se ejecute después de A; y por tanto A podría considerarse crítica en esa nueva configuración.
#
# Podría abundarse más en otras posibles consideraciones y alternativas de solución pero éstas escapan del alcance de la aplicación estricta del método de asignación de  Wiest-Levy básico explicado en el curso.

# %%
