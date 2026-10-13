# ---
# title: Roy 03
# subtitle: "8 actividades, 3 ficticias en PERT, con espera y solape"
# coleccion: 03_roy
# variante: esperas-y-solapes
# nivel: 3
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta.

# %%
pregunta_1 = "1. Dibuje el grafo Roy del proyecto (actividades en los nodos).  "
pregunta_2 = "2. Dibuje el grafo PERT equivalente. Compare el número de nodos y de arcos de ambos grafos y explique a qué se debe la diferencia.  "
pregunta_3 = "3. Calcule sobre el grafo Roy los tiempos más tempranos y más tardíos de inicio y fin de cada actividad, y señale el camino crítico en ambos grafos.  "
pregunta_4 = "4. Replantee el proyecto si, además, entre el final de B y el comienzo de D deben transcurrir 3 semanas de espera, y C puede comenzar cuando A está a la mitad de su ejecución. Dibuje el nuevo grafo Roy y calcule la nueva duración del proyecto.  "

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
    ('C'     , 'A,B'      , 7       ),
    ('D'     , 'B'        , 6       ),
    ('E'     , 'C'        , 2       ),
    ('F'     , 'C,D'      , 7       ),
    ('G'     , 'C,D'      , 1       ),
    ('H'     , 'E,F,G'    , 8       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la producción de un cortometraje; las actividades son las siguientes (las duraciones del cuadro están en días).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Guion definitivo',
    'B': 'Casting',
    'C': 'Localizaciones y permisos',
    'D': 'Diseño de vestuario y atrezo',
    'E': 'Plan de rodaje',
    'F': 'Ensayos con los actores',
    'G': 'Rodaje de exteriores',
    'H': 'Rodaje en plató',
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

# %% [markdown]
# En el grafo Roy cada actividad es un nodo y cada prelación inmediata un arco. Se añaden dos nodos auxiliares, inicio y fin, para que el grafo tenga un único origen y un único destino. No hacen falta actividades ficticias.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelacionesExpandido
prelaciones.display_immediate_linkage_matrix()

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-roy
proyecto.roy(ranksep=1.2, nodesep=1.2)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| fig-cap: "Grafo PERT equivalente"
#| label: fig-pert
proyecto.pert(durations=False, ranksep=1)

# %%
n_act = len(proyecto)
n_prelaciones = sum(len(p) for p in prelaciones.immediate_predecessor.values())
n_arcos_roy = n_prelaciones + len(prelaciones.start_activities) + len(prelaciones.end_activities)
pd.DataFrame({'nodos': [n_act + 2, len(proyecto.nodes)],
              'arcos': [n_arcos_roy, n_act + len(proyecto.dummies)],
              'ficticias': [0, len(proyecto.dummies)]},
             index=['Roy', 'PERT'])

# %% [markdown]
# En Roy el número de nodos es fijo (actividades más inicio y fin) y los arcos son las prelaciones. En PERT los arcos son las actividades y los nodos los sucesos; cuando una prelación no puede expresarse compartiendo nodos aparece una ficticia, que es el precio que paga el PERT por representar las actividades como arcos.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Sobre el grafo Roy, el inicio más temprano de una actividad es el máximo de los fines más tempranos de sus predecesoras; el fin más tardío, el mínimo de los inicios más tardíos de sus sucesoras. Son los mismos números que da el PERT.

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario

# %%
duracion_proyecto = proyecto.duration(datos_proyecto['duration'])
camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])
Markdown(f'La duración del proyecto es {duracion_proyecto} semanas.')

# %%
#| fig-cap: "Grafo Roy con tiempos y camino crítico"
#| label: fig-royCritico
proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)

# %%
#| fig-cap: "Grafo PERT con tiempos y camino crítico"
#| label: fig-pertCritico
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Las dos condiciones nuevas son relaciones con desfase. Con actividades en los nodos se modelan con actividades auxiliares:
#
# - La espera entre el final de B y el comienzo de D es una actividad de espera, 💤D, de duración 3 y sin consumo de recursos, que sucede a B y precede a D.
# - Que C pueda comenzar a mitad de A es un solape comienzo-comienzo: se parte A en A1 y A2 (de 4 y 4 semanas) y C pasa a suceder a A1, mientras que las demás sucesoras de A siguen esperando al final de A2.

# %%
prelaciones_modificadas = {
    'B': '---',
    'C': 'A1,B',
    'D': '💤D',
    'E': 'C',
    'F': 'C,D',
    'G': 'C,D',
    'H': 'E,F,G',
    '💤D': 'B',
    'A1': '---',
    'A2': 'A1',
}
duraciones_modificadas = {
    'B': 1,
    'C': 7,
    'D': 6,
    'E': 2,
    'F': 7,
    'G': 1,
    'H': 8,
    'A1': 4,
    'A2': 4,
    '💤D': 3,
}
prelaciones_mod = PredecessorTable.from_dict_of_strings(prelaciones_modificadas)
proyecto_mod = prelaciones_mod.create_project()
duraciones_mod = pd.Series(duraciones_modificadas, name='duration')

# %%
#| fig-cap: "Grafo Roy del proyecto con la espera y el solape"
#| label: fig-royModificado
proyecto_mod.roy(durations=duraciones_mod, ranksep=1.2, nodesep=1.2)

# %%
#| tbl-cap: "Calendario del proyecto modificado"
#| label: tbl-calendarioModificado
proyecto_mod.calendar(duraciones_mod)

# %%
duracion_mod = proyecto_mod.duration(duraciones_mod)
Markdown(f'La duración pasa de {duracion_proyecto} a {duracion_mod} semanas: la espera añade tiempo si B-D está en una ruta crítica y el solape lo quita si A-C lo está. Aquí el efecto neto es de {duracion_mod - duracion_proyecto:+d} semanas.')
