# ---
# title: Caminos 01
# subtitle: "8 actividades, 6 rutas"
# coleccion: 06_matriz_caminos
# variante: matriz-y-duracion
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

GANTT_PARAMS = dict(arrows=True)   # flechas de prelación en los diagramas de Gantt (rojas las críticas)


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta.

# %%
pregunta_1 = "1. Dibuje el grafo PERT y enumere todas sus rutas entre el nodo inicial y el final.  "
pregunta_2 = "2. Construya la matriz de caminos del proyecto (una fila por ruta, una columna por actividad).  "
pregunta_3 = "3. Calcule la duración de cada ruta como producto de la matriz de caminos por el vector de duraciones, y la duración del proyecto como el máximo. Identifique la ruta crítica.  "
pregunta_4 = "4. Calcule la sensibilidad de la duración del proyecto a cada actividad: ¿en cuánto aumenta la duración si una actividad dura un día más? ¿Y si dura un día menos?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 8       ),
    ('B'     , '---'      , 2       ),
    ('C'     , '---'      , 5       ),
    ('D'     , 'A,B,C'    , 6       ),
    ('E'     , 'D'        , 8       ),
    ('F'     , 'D'        , 5       ),
    ('G'     , 'E'        , 7       ),
    ('H'     , 'F,G'      , 6       ),
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
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
def bmatrix(a, dec=0):
    """Matriz LaTeX (bmatrix) a partir de un DataFrame o array."""
    a = np.asarray(getattr(a, 'values', a), dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    filas = [' & '.join(f'{x:.{dec}f}' for x in fila) + r'\\' for fila in a]
    return r'\begin{bmatrix}' + '\n' + '\n'.join(filas) + '\n' + r'\end{bmatrix}'

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT con numeración de nodos"
#| label: fig-pert
proyecto.pert(durations=False, ranksep=1)

# %% [markdown]
# Las rutas se enumeran recorriendo el grafo desde el nodo inicial: en cada nodo se abre una rama por cada arco saliente. Las ficticias se recorren pero no se anotan.

# %%
#| tbl-cap: "Rutas del proyecto"
#| label: tbl-rutas
rutas = proyecto.paths(dummies=False)
pd.Series({r: ' - '.join(a) for r, a in rutas.items()}).to_frame('actividades')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Matriz de caminos"
#| label: tbl-matrizCaminos
path_matrix = proyecto.path_matrix(dummies=False)
proyecto.display_path_matrix(dummies=False)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
duraciones = datos_proyecto['duration'][path_matrix.columns]
duraciones_rutas = path_matrix @ duraciones
Math(bmatrix(path_matrix) + r' \times ' + bmatrix(duraciones) + ' = ' + bmatrix(duraciones_rutas))

# %%
duracion_proyecto = int(duraciones_rutas.max())
ruta_critica = duraciones_rutas.idxmax()
Markdown(f'La duración del proyecto es {duracion_proyecto} días, la de la ruta más larga, {ruta_critica}: '
         f'{" - ".join(rutas[ruta_critica])}. Coincide con el tiempo más temprano del último nodo: {proyecto.duration(duraciones)}.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La duración del proyecto es $\tau = \max_r (P\,d)_r$. Si una actividad crítica dura un día más, la ruta crítica se alarga un día y con ella el proyecto: su sensibilidad es 1. Si una actividad no crítica dura un día más, la ruta más larga no cambia mientras no se agote la holgura: sensibilidad 0. Al acortar una actividad crítica el proyecto sólo se acorta hasta que otra ruta pasa a ser la más larga.

# %%
sensibilidad = proyecto.tau_sensitivity(duraciones.to_dict())
holgura = duracion_proyecto - duraciones_rutas
segunda = holgura[holgura > 0].min() if (holgura > 0).any() else None
pd.DataFrame({'sensibilidad (+1 día)': sensibilidad.reindex(duraciones.index)}).T

# %%
comprobacion = {}
for act in duraciones.index:
    mas = duraciones.copy(); mas[act] += 1
    menos = duraciones.copy(); menos[act] -= 1
    comprobacion[act] = {'+1 día': int((path_matrix @ mas).max()) - duracion_proyecto,
                         '-1 día': int((path_matrix @ menos).max()) - duracion_proyecto}
pd.DataFrame(comprobacion)

# %%
Markdown(f'Al acortar una actividad crítica el proyecto se reduce un día sólo si ninguna otra ruta alcanza la duración máxima; '
         f'la segunda ruta más larga está a {segunda} día(s) de la crítica, así que esa es la reducción máxima que puede conseguirse '
         'acortando una única actividad crítica que no pertenezca también a esa segunda ruta.')
