# ---
# title: Recursos 04
# subtitle: "9 actividades; combustible con suministro de 70 l/semana"
# coleccion: 11_recursos
# variante: consumible
# nivel: 2
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

GANTT_PARAMS = dict(arrows=True)   # flechas de prelación en los diagramas de Gantt (rojas las críticas)


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (semanas) se adjunta. Cada actividad consume cada semana la cantidad de combustible indicada (litros/semana). El suministrador entrega como máximo 70 litros por semana.

# %%
pregunta_1 = "1. Dibuje el diagrama de consumo semanal con las actividades en su inicio más temprano. ¿En qué semanas se supera el suministro?  "
pregunta_2 = "2. Si no se puede almacenar combustible de una semana para la siguiente, reprograme el proyecto para respetar el suministro (algoritmo de asignación). ¿Cuánto se alarga?  "
pregunta_3 = "3. Si en cambio se dispone de un depósito donde guardar lo no consumido, la restricción pasa a ser acumulada: el consumo acumulado hasta cada semana no puede superar el suministro acumulado. Compruebe si la programación original la cumple.  "
pregunta_4 = "4. Compare ambas situaciones. ¿Qué tipo de recurso son los operarios de los ejercicios anteriores?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources
    ('A'     , '---'      , 3       , 50       ),
    ('B'     , '---'      , 5       , 10       ),
    ('C'     , 'A'        , 5       , 50       ),
    ('D'     , 'B,C'      , 1       , 30       ),
    ('E'     , 'D'        , 2       , 30       ),
    ('F'     , 'D'        , 5       , 20       ),
    ('G'     , 'C'        , 5       , 50       ),
    ('H'     , 'G'        , 1       , 40       ),
    ('I'     , 'E,F,H'    , 1       , 10       ),
], columns=['activity', 'predecessor', 'duration', 'resources']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
suministro = 70

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Diagrama de Gantt con el consumo semanal"
#| label: fig-ganttInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Consumo semanal frente al suministro"
#| label: fig-cargasInicial
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[suministro])

# %%
consumo = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
exceso = consumo[consumo > suministro]
Markdown(f'Se supera el suministro en las semanas {", ".join(str(t) for t in exceso.index)} (máximo {int(consumo.max())} litros).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| fig-cap: "Diagrama de Gantt tras la asignación"
#| label: fig-ganttAsignado
proyecto_asignado = proyecto.copy()
datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',
    maximo=[suministro], report=False, params=dict(period_width=1, **GANTT_PARAMS))
pinta_tikz(dibujo)

# %%
duracion_inicial = proyecto.duration(datos_proyecto['duration'])
duracion_asignada = len(gantt_asignado.data.columns.drop('H_total'))
Markdown(f'Sin almacenamiento el proyecto pasa de {duracion_inicial} a {duracion_asignada} semanas.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
acumulado = pd.DataFrame({'consumo acumulado': consumo.cumsum(), 'suministro acumulado': suministro * np.arange(1, len(consumo) + 1)})
acumulado['cumple'] = acumulado['consumo acumulado'] <= acumulado['suministro acumulado']
acumulado.T

# %%
#| fig-cap: "Consumo y suministro acumulados"
#| label: fig-acumulado
ax = acumulado[['consumo acumulado', 'suministro acumulado']].plot(figsize=(7, 4)); ax.set_xlabel('semana'); ax.set_ylabel('litros');

# %%
Markdown('Con depósito la programación original ' + ('es viable: el depósito absorbe los picos con lo sobrante de las semanas de menos consumo.' if acumulado['cumple'].all() else f'tampoco es viable: en la semana {acumulado.index[~acumulado["cumple"]][0]} el consumo acumulado supera lo entregado hasta entonces.'))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Un recurso no almacenable (mano de obra, maquinaria, un suministro que no cabe en depósito) impone una restricción semana a semana; uno almacenable impone una restricción acumulada, mucho menos exigente. Los operarios son del primer tipo: las horas de un operario que no se usan esta semana no se guardan para la siguiente. Por eso los recursos renovables se tratan con asignación y los consumibles con balances acumulados (y, si hace falta, un depósito dimensionado con la diferencia máxima entre las dos curvas).
