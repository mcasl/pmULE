# ---
# title: ejemplo_parcial — enunciado
# subtitle: "P1: Holguras y camino crítico (holgura-compartida); P2: Probabilidad de cumplir el plazo (aproximación normal) (dos-rutas-criticas); P3: Reducción de la duración con mínimo coste (barata-no-optima); P4: Método del valor ganado (earned-schedule)"
# coleccion: examen
# variante: 05:holgura-compartida,08:dos-rutas-criticas,10:barata-no-optima,12:earned-schedule
# nivel: 0
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
import os, sys
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, Markdown, Math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Versión local de la librería: se busca src/pmule.py subiendo desde el directorio actual
_aqui = os.path.abspath('.')
while _aqui != os.path.dirname(_aqui) and not os.path.isfile(os.path.join(_aqui, 'src', 'pmule.py')):
    _aqui = os.path.dirname(_aqui)
sys.path.append(os.path.join(_aqui, 'src'))
from pmule import (PredecessorTable, ProjectGraph, EarnedValue,
                   pinta_tikz, plot_cargas, plot_curvas_valor_ganado,
                   tanto_por_uno, ordena_rutas)


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Problema 1

# %%
for _v in [v for v in list(globals()) if v.startswith('pregunta_')]:
    del globals()[_v]

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta.

# %%
pregunta_1 = "1. Calcule el calendario del proyecto con las holguras total, libre e independiente.  "
pregunta_2 = "2. A y C son consecutivas y tienen la misma holgura total. El responsable de A pide retrasarla 3 semanas y el de C también. ¿Pueden concederse las dos peticiones sin alargar el proyecto? Razónelo antes de calcular.  "
pregunta_3 = "3. Compruébelo: recalcule el calendario con A retrasada 3 semanas y después con las dos retrasadas.  "
pregunta_4 = "4. ¿Qué holgura expresa cuánto puede retrasarse cada una sin quitarle margen a la otra? ¿Cómo repartiría las 3 semanas entre A y C?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado-p1
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 1       ),
    ('B'     , '---'      , 8       ),
    ('C'     , 'A'        , 4       ),
    ('D'     , 'A'        , 2       ),
    ('E'     , 'B,C,D'    , 8       ),
    ('F'     , 'D'        , 2       ),
    ('G'     , 'C,F'      , 1       ),
    ('H'     , 'E'        , 2       ),
    ('I'     , 'F,H'      , 2       ),
    ('J'     , 'G,I'      , 7       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a el lanzamiento de una aplicación móvil; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones-p1
descripciones = pd.Series({
    'A': 'Estudio de requisitos',
    'B': 'Diseño de la arquitectura',
    'C': 'Diseño de interfaz y prototipo',
    'D': 'Base de datos y modelo',
    'E': 'API del servidor',
    'F': 'Pantallas principales',
    'G': 'Módulo de pagos',
    'H': 'Notificaciones y sincronización',
    'I': 'Pruebas de integración',
    'J': 'Pruebas con usuarios',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Problema 2

# %%
for _v in [v for v in list(globals()) if v.startswith('pregunta_')]:
    del globals()[_v]

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en semanas) se adjunta.

# %%
pregunta_1 = "1. Determine la duración media del proyecto y su(s) camino(s) crítico(s).  "
pregunta_2 = "2. Calcule la varianza y la desviación típica de la duración del proyecto. ¿Qué ruta crítica debe usarse para ello? Razone la respuesta.  "
pregunta_3 = "3. ¿Qué probabilidad hay de terminar el proyecto en 36 semanas o menos?  "
pregunta_4 = "4. ¿Qué probabilidad hay de terminar entre las semanas 33 y 36?  "
pregunta_5 = "5. ¿Qué duración puede comprometerse con el cliente con una probabilidad del 95% de cumplirla?  "
pregunta_6 = "6. ¿Qué probabilidad hay de terminar en la duración media o antes? ¿Puede garantizarse esa duración con un 95% de probabilidad sin modificar el proyecto?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado-p2
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 5       , 1                 ),
    ('B'     , '---'      , 5       , 0.8               ),
    ('C'     , 'A,B'      , 4       , 0.1               ),
    ('D'     , 'A'        , 4       , 1                 ),
    ('E'     , 'C'        , 8       , 0.3               ),
    ('F'     , 'E'        , 8       , 0.8               ),
    ('G'     , 'F'        , 3       , 0.6               ),
    ('H'     , 'F'        , 6       , 0.2               ),
    ('I'     , 'G,H'      , 3       , 0.4               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Problema 3

# %%
for _v in [v for v in list(globals()) if v.startswith('pregunta_')]:
    del globals()[_v]

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste unitario de reducción (k€/semana) se adjunta. Las actividades con CUR infinito no pueden acelerarse.

# %%
pregunta_1 = "1. Determine la duración del proyecto y sus rutas críticas. ¿Qué actividades son comunes a todas ellas?  "
pregunta_2 = "2. La actividad crítica más barata de acelerar es B. Un compañero propone acortarla una semana para reducir el proyecto una semana. Prediga, sin calcular, si funcionará.  "
pregunta_3 = "3. Compruébelo: acorte B una semana y recalcule la duración.  "
pregunta_4 = "4. Aplique el algoritmo de Ackoff-Sasieni para reducir el proyecto dos semanas con mínimo coste y explique por qué en cada paso elige lo que elige (actividad común frente a pareja de actividades).  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado-p3
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, ucr   
    ('A'     , '---'      , 3       , 3           , np.inf),
    ('B'     , '---'      , 3       , 1           , 1     ),
    ('C'     , '---'      , 2       , 1           , 1     ),
    ('D'     , 'A,B,C'    , 8       , 6           , 9     ),
    ('E'     , 'D'        , 3       , 3           , np.inf),
    ('F'     , 'B'        , 2       , 2           , np.inf),
    ('G'     , 'E,F'      , 6       , 4           , 9     ),
    ('H'     , 'E'        , 6       , 3           , 2     ),
    ('I'     , 'G'        , 7       , 7           , np.inf),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Problema 4

# %%
for _v in [v for v in list(globals()) if v.startswith('pregunta_')]:
    del globals()[_v]

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por actividad (BAC, en k€) se adjunta. El coste planificado de cada actividad se reparte uniformemente a lo largo de su duración.

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado-p4
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 7       , 40 ),
    ('B'     , '---'      , 6       , 20 ),
    ('C'     , 'A,B'      , 2       , 10 ),
    ('D'     , 'A,B'      , 2       , 40 ),
    ('E'     , 'B'        , 8       , 20 ),
    ('F'     , 'C'        , 1       , 30 ),
    ('G'     , 'D,F'      , 6       , 40 ),
    ('H'     , 'E,G'      , 4       , 60 ),
    ('I'     , 'H'        , 2       , 20 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# Al final de la semana 11 se realiza un punto de control. Para cada actividad se anotan las semanas trabajadas hasta ese momento, el coste real incurrido (k€) y el porcentaje realizado.

# %%
#| tbl-cap: "Datos del punto de control (final de la semana 11)"
#| label: tbl-seguimiento-p4
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 10             , 48, 100      ),
    ('B'     , 6              , 22, 100      ),
    ('C'     , 1              , 6 , 50       ),
    ('D'     , 1              , 10, 25       ),
    ('E'     , 5              , 12, 42       ),
    ('F'     , 0              , 0 , 0        ),
    ('G'     , 0              , 0 , 0        ),
    ('H'     , 0              , 0 , 0        ),
    ('I'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %%
pregunta_1 = "1. Dibuje los diagramas de Gantt del valor planificado y del valor ganado y calcule PV, EV, AC, CPI y SPI en el punto de control.  "
pregunta_2 = "2. Calcule la programación ganada (ES), el retraso en semanas y el índice SPI(t).  "
pregunta_3 = "3. Estime la duración final del proyecto con SPI y con SPI(t). ¿Por qué difieren? ¿Cuál es más fiable?  "
pregunta_4 = "4. ¿Qué valor tomará el SPI clásico cuando el proyecto termine, aunque acabe tarde? ¿Y el SPI(t)?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))
