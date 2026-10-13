# ---
# title: Holguras 05
# subtitle: "9 actividades; ordenación por vigilancia y actividades sin margen"
# coleccion: 05_holguras
# variante: vigilancia
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


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta. El jefe de proyecto sólo puede hacer seguimiento estrecho de cuatro actividades.

# %%
pregunta_1 = "1. Calcule el calendario del proyecto con sus holguras.  "
pregunta_2 = "2. Ordene las actividades de mayor a menor necesidad de vigilancia (primero las críticas; entre las demás, menor holgura total primero y, a igualdad, menor holgura libre). ¿Cuáles son las cuatro elegidas?  "
pregunta_3 = "3. ¿Qué actividades *no críticas* no pueden retrasarse ni una semana sin obligar a mover alguna sucesora? ¿Por qué no son críticas si no admiten retraso?  "
pregunta_4 = "4. ¿Qué actividad puede retrasarse más semanas sin que nadie lo note? ¿Cuántas?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 6       ),
    ('B'     , '---'      , 3       ),
    ('C'     , '---'      , 2       ),
    ('D'     , 'B'        , 4       ),
    ('E'     , 'B'        , 8       ),
    ('F'     , 'A,B,C'    , 7       ),
    ('G'     , 'D,F'      , 8       ),
    ('H'     , 'G'        , 7       ),
    ('I'     , 'E'        , 6       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la apertura de una clínica veterinaria; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de mercado y local',
    'B': 'Proyecto de reforma y licencia',
    'C': 'Obra de reforma',
    'D': 'Instalación de quirófano y radiología',
    'E': 'Mobiliario y equipamiento clínico',
    'F': 'Software de gestión',
    'G': 'Selección de personal',
    'H': 'Acuerdos con proveedores',
    'I': 'Registro sanitario',
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
#| fig-cap: "Grafo PERT con tiempos y camino crítico"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %%
#| tbl-cap: "Calendario y holguras"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Actividades ordenadas por necesidad de vigilancia"
#| label: tbl-orden
orden = calendario.sort_values(['H_total', 'H_libre', 'duracion'], ascending=[True, True, False])
orden['prioridad'] = range(1, len(orden) + 1)
orden[['prioridad', 'H_total', 'H_libre', 'H_independiente', 'duracion']]

# %%
Markdown(f'Las cuatro actividades a vigilar son **{", ".join(orden.index[:4])}**. '
         'Si hay más de cuatro críticas, el criterio de desempate es la duración (una actividad larga tiene más ocasiones de retrasarse).')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
sin_margen = calendario.index[(calendario['H_total'] > 0) & (calendario['H_libre'] == 0)].tolist()
Markdown(f'''Las actividades {", ".join(sin_margen)} tienen holgura libre cero pero holgura total positiva: cualquier retraso suyo mueve el inicio más temprano de una sucesora, pero esa sucesora tiene margen suficiente para absorberlo sin que el proyecto se alargue. No son críticas porque el plazo no depende de ellas; lo que no admiten es retrasarse sin molestar.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
mas_libre = calendario['H_libre'].idxmax()
Markdown(f'{mas_libre} puede retrasarse hasta {int(calendario.loc[mas_libre, "H_libre"])} semanas sin mover a ninguna otra actividad ni al plazo (su holgura libre).')
