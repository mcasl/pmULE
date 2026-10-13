# ---
# title: Recursos 02
# subtitle: "8 actividades, asignación con límite 6→8"
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

# Versión local de la librería (notebooks/colecciones/<coleccion>/ -> ../../../src)
sys.path.append(os.path.abspath('../../../src'))
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y número de operarios necesarios por actividad se adjunta. La empresa sólo dispone de 6 operarios durante las 3 primeras semanas y de 8 a partir de entonces.

# %%
pregunta_1 = "1. Dibuje el diagrama de cargas con las actividades en su inicio más temprano y compruebe si se respeta el límite de operarios.  "
pregunta_2 = "2. Reprograme el proyecto con el algoritmo de asignación de Wiest-Levy para respetar el límite. ¿Qué actividades se desplazan y cuánto se alarga el proyecto?  "
pregunta_3 = "3. ¿Y si se dispusiera de 6 operarios durante todo el proyecto? Compare las dos duraciones.  "
pregunta_4 = "4. Explique por qué la asignación puede alargar el proyecto y la nivelación nunca.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources
    ('A'     , '---'      , 2       , 4        ),
    ('B'     , '---'      , 3       , 4        ),
    ('C'     , '---'      , 2       , 2        ),
    ('D'     , 'A,C'      , 5       , 4        ),
    ('E'     , 'B'        , 6       , 3        ),
    ('F'     , 'B,D'      , 6       , 1        ),
    ('G'     , 'C'        , 3       , 4        ),
    ('H'     , 'D,E,G'    , 8       , 2        ),
], columns=['activity', 'predecessor', 'duration', 'resources']).set_index('activity')
datos_enunciado

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
#| fig-cap: "Grafo PERT con tiempos y camino crítico"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %%
#| fig-cap: "Diagrama de Gantt con las cargas de operarios (inicios más tempranos)"
#| label: fig-ganttInicial
proyecto_inicial = proyecto.copy()
gantt_inicial, dibujo = proyecto_inicial.gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True)
pinta_tikz(dibujo)

# %%
limite_maximo_recursos = [6] * 3 + [8]

# %%
#| fig-cap: "Diagrama de cargas con el límite de operarios"
#| label: fig-cargasInicial
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite_maximo_recursos)

# %%
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
limite = pd.Series([limite_maximo_recursos[min(t - 1, len(limite_maximo_recursos) - 1)] for t in range(1, len(cargas) + 1)], index=cargas.index)
exceso = cargas[cargas > limite]
Markdown(f'El límite se supera en las semanas {", ".join(str(t) for t in exceso.index)}: hacen falta hasta {int(cargas.max())} operarios.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# El algoritmo de Wiest-Levy recorre el calendario periodo a periodo: cuando la demanda supera el límite, aplaza las actividades con más holgura (y, si no basta, también críticas, lo que alarga el proyecto). Las filas 💤 marcan las esperas introducidas.

# %%
#| fig-cap: "Diagrama de Gantt tras la asignación"
#| label: fig-ganttAsignado
proyecto_asignado = proyecto.copy()
datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',
    maximo=limite_maximo_recursos, report=False, params=dict(period_width=1))
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas tras la asignación"
#| label: fig-cargasAsignado
plot_cargas(gantt_df=gantt_asignado, limite_maximo_recursos=limite_maximo_recursos)

# %%
duracion_inicial = proyecto.duration(datos_proyecto['duration'])
duracion_asignada = len(gantt_asignado.data.columns.drop('H_total'))
esperas = datos_asignados[datos_asignados.index.str.startswith('💤')]
Markdown(f'El proyecto pasa de {duracion_inicial} a {duracion_asignada} semanas. Esperas introducidas: '
         f'{", ".join(f"{i[1:]} ({int(fila.duration)} sem.)" for i, fila in esperas.iterrows()) or "ninguna"}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| fig-cap: "Diagrama de Gantt con el límite constante"
#| label: fig-ganttLimiteFijo
limite_fijo = [6]
proyecto_fijo = proyecto.copy()
datos_fijo, gantt_fijo, dibujo = proyecto_fijo.asignar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',
    maximo=limite_fijo, report=False, params=dict(period_width=1))
pinta_tikz(dibujo)

# %%
duracion_fija = len(gantt_fijo.data.columns.drop('H_total'))
Markdown(f'Con 6 operarios durante todo el proyecto la duración sería de {duracion_fija} semanas, frente a {duracion_asignada} con el refuerzo a partir de la semana 3.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La nivelación sólo usa las holguras: desplaza actividades no críticas sin tocar el plazo. La asignación impone una restricción dura; si en algún periodo ni siquiera aplazando todas las actividades con holgura cabe la demanda, tiene que aplazar actividades críticas y el proyecto se alarga.
