# ---
# title: Holguras 06
# subtitle: "8 actividades; calendario con errores en B y C"
# coleccion: 05_holguras
# variante: diagnostico
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

GANTT_PARAMS = dict(arrows=True)   # flechas de prelación en los diagramas de Gantt (rojas las críticas)


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Un alumno ha entregado el calendario que se adjunta para el proyecto cuyo cuadro de prelaciones y duraciones (semanas) también se da. Contiene dos errores.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 2       ),
    ('B'     , '---'      , 7       ),
    ('C'     , '---'      , 8       ),
    ('D'     , 'A,B,C'    , 2       ),
    ('E'     , 'A'        , 6       ),
    ('F'     , 'D'        , 6       ),
    ('G'     , 'F'        , 6       ),
    ('H'     , 'G'        , 3       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la rehabilitación de una nave industrial para un taller; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Levantamiento topográfico y mediciones',
    'B': 'Proyecto de ejecución y licencia',
    'C': 'Desmontaje de instalaciones antiguas',
    'D': 'Demolición de tabiquería interior',
    'E': 'Refuerzo de la estructura metálica',
    'F': 'Reparación de la cubierta',
    'G': 'Solera y canalizaciones',
    'H': 'Cerramientos y carpintería exterior',
}, name='Descripción')
descripciones.to_frame()

# %%
#| tbl-cap: "Calendario entregado (con errores)"
#| label: tbl-calendarioEntregado
calendario_entregado = pd.DataFrame([
#  activity, inicio_mas_temprano, fin_mas_temprano, inicio_mas_tardio, fin_mas_tardio, H_total, H_libre
    ('A'     , 0                  , 2               , 6                , 8             , 6      , 0      ),
    ('B'     , 0                  , 7               , 1                , 8             , 1      , 2      ),
    ('C'     , 0                  , 8               , 2                , 8             , 0      , 0      ),
    ('D'     , 8                  , 10              , 8                , 10            , 0      , 0      ),
    ('E'     , 2                  , 8               , 19               , 25            , 17     , 17     ),
    ('F'     , 10                 , 16              , 10               , 16            , 0      , 0      ),
    ('G'     , 16                 , 22              , 16               , 22            , 0      , 0      ),
    ('H'     , 22                 , 25              , 22               , 25            , 0      , 0      ),
], columns=['activity', 'inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'H_libre']).set_index('activity')
calendario_entregado

# %%
pregunta_1 = "1. Sin recalcular todo el proyecto, localice los errores comprobando las relaciones que debe cumplir cualquier calendario: EF = ES + d, LF = LS + d, H_total = LS − ES, 0 ≤ H_libre ≤ H_total, y ES de cada actividad igual al máximo EF de sus predecesoras.  "
pregunta_2 = "2. Explique qué significa cada error (qué se ha calculado mal).  "
pregunta_3 = "3. Calcule el calendario correcto y compárelo.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
c = calendario_entregado.copy()
d = datos_proyecto['duration']

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
comprobaciones = pd.DataFrame({
    'EF = ES + d': c['fin_mas_temprano'] == c['inicio_mas_temprano'] + d,
    'LF = LS + d': c['fin_mas_tardio'] == c['inicio_mas_tardio'] + d,
    'H_total = LS − ES': c['H_total'] == c['inicio_mas_tardio'] - c['inicio_mas_temprano'],
    'H_libre ≤ H_total': c['H_libre'] <= c['H_total'],
    'ES = máx EF pred.': [c.loc[a, 'inicio_mas_temprano'] == max((c.loc[p, 'fin_mas_temprano'] for p in prelaciones.immediate_predecessor[a]), default=0) for a in c.index]})
comprobaciones[~comprobaciones.all(axis=1)]

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# - En B la holgura libre es mayor que la total, lo que es imposible: la holgura libre es una parte de la total (la que no afecta a las sucesoras).
# - En C el inicio más tardío no cuadra con el fin más tardío menos la duración (ni con H_total = LS − ES): se ha copiado mal o se ha calculado el tardío sin restar la duración.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Calendario correcto"
#| label: tbl-calendarioCorrecto
calendario = proyecto.calendar(d)
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'H_libre']]

# %%
difieren = (calendario[c.columns] != c).any(axis=1)
pd.concat({'entregado': c[difieren], 'correcto': calendario.loc[difieren, c.columns]}, axis=1)
