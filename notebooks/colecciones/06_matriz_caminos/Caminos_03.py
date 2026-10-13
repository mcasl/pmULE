# ---
# title: Caminos 03
# subtitle: "9 actividades, 8 rutas, reconstrucción desde las rutas y SVD"
# coleccion: 06_matriz_caminos
# variante: rutas-a-grafo-svd
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
# De un proyecto sólo se conocen sus rutas entre el nodo inicial y el final y las duraciones de sus actividades (en días).

# %%
pregunta_1 = "1. Construya la matriz de caminos del proyecto.  "
pregunta_2 = "2. Calcule con ella la duración del proyecto e identifique la ruta crítica.  "
pregunta_3 = "3. Reconstruya el cuadro de prelaciones inmediatas y dibuje los grafos PERT y Roy. Compruebe que sus rutas son las del enunciado.  "
pregunta_4 = "4. Calcule la sensibilidad de la duración del proyecto a cada actividad: ¿en cuánto aumenta la duración si una actividad dura un día más?  "
pregunta_5 = "5. Calcule la descomposición en valores singulares de la matriz de caminos y ordene rutas y actividades según el primer vector singular. Compare con la ruta crítica.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Rutas del proyecto"
#| label: tbl-datosRutas
rutas_enunciado = {
    'Route_1': ['A', 'C', 'D', 'F', 'H', 'I'],
    'Route_2': ['A', 'C', 'D', 'G', 'I'],
    'Route_3': ['A', 'C', 'E', 'F', 'H', 'I'],
    'Route_4': ['A', 'C', 'E', 'G', 'I'],
    'Route_5': ['B', 'C', 'D', 'F', 'H', 'I'],
    'Route_6': ['B', 'C', 'D', 'G', 'I'],
    'Route_7': ['B', 'C', 'E', 'F', 'H', 'I'],
    'Route_8': ['B', 'C', 'E', 'G', 'I'],
}
pd.Series({r: ' - '.join(a) for r, a in rutas_enunciado.items()}).to_frame('actividades')

# %%
#| tbl-cap: "Duración de las actividades"
#| label: tbl-datosDuraciones
datos_enunciado = pd.DataFrame([
#  activity, duration
    ('A'     , 6       ),
    ('B'     , 1       ),
    ('C'     , 8       ),
    ('D'     , 1       ),
    ('E'     , 5       ),
    ('F'     , 1       ),
    ('G'     , 3       ),
    ('H'     , 7       ),
    ('I'     , 7       ),
], columns=['activity', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la instalación de una planta fotovoltaica sobre cubierta; las actividades son las siguientes (las duraciones del cuadro están en días).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de sombras y producción',
    'B': 'Cálculo estructural de la cubierta',
    'C': 'Tramitación de permisos',
    'D': 'Acopio de módulos e inversores',
    'E': 'Montaje de la estructura soporte',
    'F': 'Colocación de módulos',
    'G': 'Cableado de corriente continua',
    'H': 'Instalación de inversores',
    'I': 'Cuadro de protecciones y contador',
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

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# La matriz de caminos tiene un 1 en la fila de una ruta y la columna de una actividad cuando la actividad pertenece a la ruta.

# %%
actividades = sorted(set(a for acts in rutas_enunciado.values() for a in acts))
path_matrix = pd.DataFrame(0, index=list(rutas_enunciado), columns=actividades)
for ruta, acts in rutas_enunciado.items():
    path_matrix.loc[ruta, acts] = 1
path_matrix

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duraciones = datos_enunciado['duration'].reindex(actividades)
duraciones_rutas = path_matrix @ duraciones
Math(bmatrix(path_matrix) + r' \times ' + bmatrix(duraciones) + ' = ' + bmatrix(duraciones_rutas))

# %%
duracion_proyecto = int(duraciones_rutas.max())
ruta_critica = duraciones_rutas.idxmax()
Markdown(f'La duración del proyecto es {duracion_proyecto} días, la de la ruta más larga, {ruta_critica}: '
         f'{" - ".join(rutas_enunciado[ruta_critica])}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Las predecesoras inmediatas de una actividad son las que la preceden inmediatamente en alguna ruta; las rutas son todas las secuencias posibles, así que el conjunto de parejas consecutivas de todas las rutas es exactamente el cuadro de prelaciones inmediatas. Las actividades que abren alguna ruta son iniciales.

# %%
prelaciones_dict = {a: set() for a in actividades}
for acts in rutas_enunciado.values():
    for anterior, posterior in zip(acts[:-1], acts[1:]):
        prelaciones_dict[posterior].add(anterior)
datos_proyecto = pd.DataFrame({'predecessor': pd.Series({a: ','.join(sorted(p)) or '---' for a, p in prelaciones_dict.items()}),
                               'duration': duraciones})
datos_proyecto.index.name = 'activity'
datos_proyecto

# %%
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %%
#| fig-cap: "Grafo PERT reconstruido, con tiempos y ruta crítica"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %%
#| fig-cap: "Grafo Roy reconstruido"
#| label: fig-roy
proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)

# %%
rutas_grafo = {' - '.join(a) for a in proyecto.paths(dummies=False).values()}
rutas_dadas = {' - '.join(a) for a in rutas_enunciado.values()}
print('Las rutas del grafo coinciden con las del enunciado:', rutas_grafo == rutas_dadas)

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

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# La descomposición en valores singulares $P = U\,S\,V^T$ ordena la información de la matriz de caminos por importancia. El primer vector singular por la izquierda, $u_1$, pondera las rutas, y el primero por la derecha, $v_1$, las actividades: las actividades que aparecen en muchas rutas y las rutas que contienen muchas actividades compartidas reciben más peso. Es una ordenación estructural, que no usa las duraciones, y por eso no tiene por qué coincidir con la ruta crítica.

# %%
U, S, Vt = np.linalg.svd(path_matrix.values.astype(float))
u1 = np.abs(U[:, 0]); v1 = np.abs(Vt[0, :])
importancia_rutas = pd.Series(u1 / u1.sum(), index=path_matrix.index)
importancia_actividades = pd.Series(v1 / v1.sum(), index=path_matrix.columns)
Math(r'S = ' + bmatrix(S.round(2), dec=2))

# %%
#| tbl-cap: "Rutas y actividades ordenadas por su peso en el primer vector singular"
#| label: tbl-svd
ordena_rutas(path_matrix.copy(), importancia_actividades.values, importancia_rutas.values)

# %%
Markdown(f'La ruta dominante según la SVD es {importancia_rutas.idxmax()} y la ruta crítica es {ruta_critica}. '
         f'La actividad de mayor peso es {importancia_actividades.idxmax()}, que aparece en '
         f'{int(path_matrix[importancia_actividades.idxmax()].sum())} de las {len(path_matrix)} rutas.')
