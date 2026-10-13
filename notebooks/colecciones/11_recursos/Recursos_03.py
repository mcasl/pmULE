# ---
# title: Recursos 03
# subtitle: "8 actividades, dos recursos y propuesta de desplazamiento"
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (semanas) se adjunta. Cada actividad necesita un número de operarios y, algunas, una máquina excavadora (columna *machines*). El jefe de obra propone retrasar A 2 semana(s) y C 1 semana(s) para suavizar la demanda de operarios.

# %%
pregunta_1 = "1. Dibuje los diagramas de cargas de operarios y de máquinas con las actividades en su inicio más temprano. ¿Cuántas máquinas hacen falta como máximo?  "
pregunta_2 = "2. Evalúe la propuesta del jefe de obra: ¿respeta las holguras? ¿Reduce la suma de cuadrados de las cargas de operarios?  "
pregunta_3 = "3. Compare la propuesta con el resultado del algoritmo de nivelación. ¿Cuál es mejor?  "
pregunta_4 = "4. ¿Qué ocurre con la carga de máquinas tras la nivelación de operarios? ¿Puede una nivelación empeorar el otro recurso?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources, machines
    ('A'     , '---'      , 7       , 1        , 0       ),
    ('B'     , '---'      , 6       , 4        , 2       ),
    ('C'     , 'A,B'      , 7       , 3        , 2       ),
    ('D'     , 'B'        , 7       , 1        , 0       ),
    ('E'     , 'D'        , 5       , 4        , 0       ),
    ('F'     , 'D'        , 4       , 4        , 0       ),
    ('G'     , 'D'        , 2       , 1        , 2       ),
    ('H'     , 'C,E,F'    , 1       , 4        , 1       ),
], columns=['activity', 'predecessor', 'duration', 'resources', 'machines']).set_index('activity')
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
#| fig-cap: "Diagrama de cargas de operarios"
#| label: fig-cargasInicial
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=None)

# %%
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
semanas_pico = cargas.index[cargas == cargas.max()].tolist()
Markdown(f'El máximo de operarios es {int(cargas.max())}, en las semanas {", ".join(map(str, semanas_pico))}. '
         f'La media es {cargas.mean():.2f} operarios/semana.')

# %%
#| fig-cap: "Diagrama de Gantt con las cargas de máquinas"
#| label: fig-ganttMaquinas
gantt_maquinas, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'machines', report=False, tikz=True)
pinta_tikz(dibujo)

# %%
cargas_maquinas = gantt_maquinas.data.loc['Total'].drop('H_total').astype(float)
Markdown(f'Hacen falta como máximo {int(cargas_maquinas.max())} máquina(s).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
holguras = proyecto.calendar(datos_proyecto['duration'])['H_total']
propuesta = {'A': 2, 'C': 1}
pd.DataFrame({'desplazamiento propuesto': pd.Series(propuesta), 'holgura total': holguras[list(propuesta)]})

# %%
#| fig-cap: "Diagrama de Gantt con la propuesta del jefe de obra"
#| label: fig-ganttPropuesta
suma_cuadrados_inicial = float((cargas ** 2).sum())
proyecto_propuesta = proyecto.copy()
suma_cuadrados_propuesta, dibujo = proyecto_propuesta.evaluar_desplazamiento(
    datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, **propuesta)
pinta_tikz(dibujo)

# %%
Markdown(f'La propuesta respeta las holguras (cada desplazamiento es menor o igual que la holgura total) y la suma de cuadrados '
         f'pasa de {suma_cuadrados_inicial:.0f} a {suma_cuadrados_propuesta:.0f}: '
         f'{"mejora" if suma_cuadrados_propuesta < suma_cuadrados_inicial else "no mejora"} el nivelado.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Desplazamientos del algoritmo de nivelación"
#| label: tbl-desplazamientos
proyecto_nivelado = proyecto.copy()
desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')
desplazamientos[desplazamientos['desplazamientos'] > 0]

# %%
#| fig-cap: "Diagrama de cargas de operarios tras la nivelación"
#| label: fig-cargasNivelado
plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=None)

# %%
cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)
suma_cuadrados_nivelado = float((cargas_niveladas ** 2).sum())
pd.DataFrame({'suma de cuadrados': [suma_cuadrados_inicial, suma_cuadrados_propuesta, suma_cuadrados_nivelado],
              'máximo de operarios': [cargas.max(), None, cargas_niveladas.max()]},
             index=['inicial', 'propuesta del jefe de obra', 'nivelación'])

# %%
Markdown(f'El algoritmo de nivelación {"mejora" if suma_cuadrados_nivelado < suma_cuadrados_propuesta else "no mejora"} la propuesta: '
         f'prueba sistemáticamente todos los desplazamientos de cada actividad no crítica, aunque, al hacerlo actividad a actividad, tampoco garantiza el óptimo global.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| fig-cap: "Cargas de máquinas tras la nivelación de operarios"
#| label: fig-maquinasNivelado
datos_maquinas = datos_nivelados.copy()
datos_maquinas['machines'] = datos_proyecto['machines'].reindex(datos_maquinas.index).fillna(0).astype(int)
gantt_maquinas_niv, dibujo = proyecto_nivelado.copy().gantt_cargas(datos_maquinas, 'duration', 'machines', report=False, tikz=True)
pinta_tikz(dibujo)

# %%
cargas_maquinas_niv = gantt_maquinas_niv.data.loc['Total'].drop('H_total').astype(float)
pd.DataFrame({'antes': [cargas_maquinas.max(), (cargas_maquinas ** 2).sum()],
              'después': [cargas_maquinas_niv.max(), (cargas_maquinas_niv ** 2).sum()]},
             index=['máximo de máquinas', 'suma de cuadrados (máquinas)']).astype(int)

# %% [markdown]
# La nivelación optimiza un único recurso; los desplazamientos que allanan la demanda de operarios pueden concentrar la de máquinas. Con varios recursos hay que nivelar con una función objetivo combinada o comprobar cada recurso tras nivelar el principal.
