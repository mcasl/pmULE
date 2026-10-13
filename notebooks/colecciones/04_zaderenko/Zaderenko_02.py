# ---
# title: Zaderenko 02
# subtitle: "9 actividades, 9 nodos, 1 ficticias"
# coleccion: 04_zaderenko
# variante: zaderenko-b
# nivel: 1
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
# # Enunciado

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta.

# %%
pregunta_1 = "1. Dibuje el grafo PERT y numere sus nodos.  "
pregunta_2 = "2. Construya la matriz de Zaderenko y calcule con ella los tiempos más tempranos y más tardíos de cada nodo. ¿Cuál es la duración del proyecto?  "
pregunta_3 = "3. Calcule el calendario del proyecto: inicio y fin más tempranos y más tardíos de cada actividad.  "
pregunta_4 = "4. Dibuje el diagrama de Gantt con las actividades en su inicio más temprano e indique sobre él las holguras.  "
pregunta_5 = "5. La actividad F va a durar 17 días más de lo previsto. Recalcule la duración del proyecto y explique el resultado a partir de la holgura de F.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 8       ),
    ('B'     , '---'      , 1       ),
    ('C'     , 'A,B'      , 3       ),
    ('D'     , 'C'        , 2       ),
    ('E'     , 'D'        , 6       ),
    ('F'     , 'B'        , 8       ),
    ('G'     , 'E'        , 8       ),
    ('H'     , 'F,G'      , 8       ),
    ('I'     , 'H'        , 5       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la puesta en marcha de una línea de envasado; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Especificación de la línea',
    'B': 'Adjudicación de los equipos',
    'C': 'Obra civil y bancadas',
    'D': 'Recepción de la llenadora',
    'E': 'Recepción de la etiquetadora',
    'F': 'Montaje mecánico',
    'G': 'Instalación neumática',
    'H': 'Cableado y armarios de control',
    'I': 'Programación de autómatas',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelacionesExpandido
prelaciones.display_immediate_linkage_matrix()

# %%
#| fig-cap: "Grafo PERT con numeración de nodos"
#| label: fig-pert
proyecto.pert(durations=False, ranksep=1)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# En la matriz de Zaderenko la casilla (i, j) contiene la duración de la actividad que va del nodo i al nodo j (cero para las ficticias). El tiempo más temprano de un nodo es el máximo, sobre su columna, de «tiempo temprano del origen más duración»; el más tardío es el mínimo, sobre su fila, de «tiempo tardío del destino menos duración». Los tempranos se calculan de izquierda a derecha y los tardíos de derecha a izquierda, partiendo del tiempo temprano del último nodo.

# %%
#| tbl-cap: "Matriz de Zaderenko"
#| label: tbl-zaderenko
proyecto.zaderenko(durations=datos_proyecto['duration'])

# %%
#| tbl-cap: "Tiempos tempranos y tardíos de los nodos"
#| label: tbl-tiemposNodos
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
resultados_pert['nodes']

# %%
duracion_proyecto = proyecto.duration(datos_proyecto['duration'])
Markdown(f'La duración del proyecto es el tiempo más temprano del último nodo: {duracion_proyecto} días.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Para una actividad que va del nodo i al nodo j: el inicio más temprano es el tiempo temprano de i, el fin más temprano es ese inicio más la duración, el fin más tardío es el tiempo tardío de j y el inicio más tardío es ese fin menos la duración.

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]

# %%
#| fig-cap: "Grafo PERT con tiempos de los nodos y camino crítico"
#| label: fig-pertTiempos
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| fig-cap: "Diagrama de Gantt (inicios más tempranos) con indicación de las holguras"
#| label: fig-gantt
gantt_df, dibujo = proyecto.gantt(datos_proyecto, 'duration', holguras=True, tikz=True)
pinta_tikz(dibujo)

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
holgura = int(calendario.loc['F', 'H_total'])
duraciones_nuevas = datos_proyecto['duration'].copy()
duraciones_nuevas['F'] += 17
duracion_nueva = proyecto.duration(duraciones_nuevas)
Markdown(f'La holgura total de F es {holgura} días. Un retraso de 17 días '
         f'{"no " if 17 <= holgura else ""}supera esa holgura, y la duración del proyecto pasa de '
         f'{duracion_proyecto} a {duracion_nueva} días.')

# %%
#| tbl-cap: "Calendario con la nueva duración"
#| label: tbl-calendarioNuevo
proyecto.calendar(duraciones_nuevas)[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]
