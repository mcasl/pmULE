# ---
# title: PERT 05
# subtitle: "9 actividades, 7 ficticias sin simplificar frente a 2"
# coleccion: 02_pert
# variante: ficticias-sobrantes
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
# Un método sistemático para dibujar el grafo PERT consiste en dar a cada actividad su propio nodo de inicio y de fin y unir con una ficticia cada prelación. El resultado es correcto, pero tiene muchas más ficticias de las necesarias. Considere el cuadro de prelaciones adjunto.

# %%
pregunta_1 = "1. Dibuje el grafo PERT por el método sistemático (sin fusionar nodos). ¿Cuántas ficticias tiene?  "
pregunta_2 = "2. Simplifíquelo: una ficticia puede eliminarse fusionando sus dos nodos cuando al hacerlo no se crea ninguna prelación nueva ni desaparece ninguna, no se juntan dos arcos entre los mismos nodos y el grafo sigue teniendo un único origen y un único final. ¿Cuántas ficticias quedan?  "
pregunta_3 = "3. Compruebe que ambos grafos tienen el mismo cuadro de ancestros y las mismas rutas (sin ficticias).  "

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
    ('C'     , '---'      ),
    ('D'     , 'A,C'      ),
    ('E'     , 'B'        ),
    ('F'     , 'D,E'      ),
    ('G'     , 'F'        ),
    ('H'     , 'F'        ),
    ('I'     , 'G,H'      ),
], columns=['activity', 'predecessor']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT sin simplificar"
#| label: fig-pertSinSimplificar
proyecto_bruto = prelaciones.create_project(simplify=False)
proyecto_bruto.pert(durations=False, ranksep=1)

# %%
Markdown(f'El grafo sin simplificar tiene {len(proyecto_bruto.nodes)} nodos y {len(proyecto_bruto.dummies)} ficticias: {", ".join(proyecto_bruto.dummies)}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# La librería aplica exactamente esas reglas (`simplify`): recorre las ficticias y contrae las que pueden contraerse sin alterar las prelaciones distantes ni la estructura.

# %%
#| fig-cap: "Grafo PERT simplificado"
#| label: fig-pertSimplificado
proyecto = prelaciones.create_project()
proyecto.pert(durations=False, ranksep=1)

# %%
Markdown(f'Tras simplificar quedan {len(proyecto.nodes)} nodos y {len(proyecto.dummies)} ficticias: {", ".join(proyecto.dummies) or "ninguna"}. '
         f'Se han eliminado {len(proyecto_bruto.dummies) - len(proyecto.dummies)}.')

# %% [markdown]
# Las ficticias que quedan son las que transmiten una prelación que no puede expresarse compartiendo nodos: cada una une el final de un grupo de actividades con el inicio de otro grupo que depende de ellas *y de otras más*.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
mismo_ancestros = proyecto.distant_predecessor() == proyecto_bruto.distant_predecessor()
rutas = {' - '.join(a) for a in proyecto.paths(dummies=False).values()}
rutas_bruto = {' - '.join(a) for a in proyecto_bruto.paths(dummies=False).values()}
print('Mismo cuadro de ancestros:', mismo_ancestros)
print('Mismas rutas:', rutas == rutas_bruto, f'({len(rutas)} rutas)')
