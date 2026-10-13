# ---
# title: PERT 03
# subtitle: "10 actividades, 3 ficticias, 5 rutas"
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
# Considere el proyecto definido por el siguiente cuadro de prelaciones inmediatas.

# %%
pregunta_1 = "1. Dibuje el grafo PERT del proyecto utilizando el mínimo número de actividades ficticias y numere sus nodos de modo que todo arco vaya de un número menor a uno mayor.  "
pregunta_2 = "2. Para cada actividad ficticia, explique qué prelación transmite y por qué no puede evitarse.  "
pregunta_3 = "3. Enumere todas las rutas desde el nodo inicial al final e indique cuántas hay.  "
pregunta_4 = "4. Construya el cuadro de ancestros y compruebe que el grafo lo respeta.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de prelaciones del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor
    ('A'     , '---'      ),
    ('B'     , '---'      ),
    ('C'     , 'A'        ),
    ('D'     , 'A,B'      ),
    ('E'     , 'C,D'      ),
    ('F'     , 'B'        ),
    ('G'     , 'E'        ),
    ('H'     , 'F'        ),
    ('I'     , 'F,G'      ),
    ('J'     , 'H,I'      ),
], columns=['activity', 'predecessor']).set_index('activity')
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

# %% [markdown]
# Partimos del cuadro de prelaciones expandido: las filas dan las predecesoras inmediatas y las columnas las sucesoras. Dos actividades pueden compartir nodo final si tienen exactamente las mismas sucesoras, y nodo inicial si tienen las mismas predecesoras; en los demás casos hacen falta ficticias.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelacionesExpandido
prelaciones.display_immediate_linkage_matrix()

# %%
#| fig-cap: "Grafo PERT con numeración de nodos; las ficticias se dibujan discontinuas"
#| label: fig-pert
proyecto.pert(durations=False, ranksep=1)

# %%
print(f'El grafo tiene {len(proyecto.nodes)} nodos, {len(proyecto)} actividades reales '
      f'y {len(proyecto.dummies)} ficticias: {", ".join(proyecto.dummies)}')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Cada ficticia une el nodo final de unas actividades con el nodo inicial de otras. La prelación que transmite es «todas las que entran en su nodo origen preceden a todas las que salen de su nodo destino».

# %%
grafo = proyecto.pert_graph
def actividades_que_entran(nodo):
    return sorted(d['activity'] for _, _, d in grafo.in_edges(nodo, data=True) if not d['activity'].startswith('@'))
def actividades_que_salen(nodo):
    return sorted(d['activity'] for _, _, d in grafo.out_edges(nodo, data=True) if not d['activity'].startswith('@'))

filas = []
for origen, destino, d in grafo.edges(data=True):
    if d['activity'].startswith('@'):
        filas.append({'ficticia': d['activity'],
                      'entran en el nodo origen': ', '.join(actividades_que_entran(origen)),
                      'salen del nodo destino': ', '.join(actividades_que_salen(destino))})
pd.DataFrame(filas).set_index('ficticia')

# %% [markdown]
# Si en lugar de la ficticia se fusionaran ambos nodos, las actividades que salen del nodo destino pasarían a tener como predecesoras a todas las que entran en el origen *y* a las que entran en el destino, que no es lo que dice el cuadro; y si se eliminara sin fusionar, se perdería la prelación. Por eso ninguna puede evitarse.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Rutas del grafo (sin ficticias)"
#| label: tbl-rutas
rutas = proyecto.paths(dummies=False)
pd.Series({ruta: ' - '.join(acts) for ruta, acts in rutas.items()}).to_frame('actividades')

# %%
print(f'El grafo tiene {len(rutas)} rutas entre el nodo inicial y el final.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Cuadro de ancestros"
#| label: tbl-ancestros
proyecto.distant_predecessor(format='DataFrame')

# %% [markdown]
# Para cada actividad, el conjunto de actividades que aparecen antes que ella en alguna ruta debe coincidir con su fila del cuadro de ancestros:

# %%
ancestros = proyecto.distant_predecessor()
comprobacion = {}
for act in datos_proyecto.index:
    por_rutas = set()
    for acts in rutas.values():
        if act in acts:
            por_rutas |= set(acts[:acts.index(act)])
    comprobacion[act] = por_rutas == set(ancestros[act])
pd.Series(comprobacion).to_frame('coincide')
