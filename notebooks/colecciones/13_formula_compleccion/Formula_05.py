# ---
# title: Formula 05
# subtitle: "instancia pat1 de Patterson (14 actividades)"
# coleccion: 13_formula_compleccion
# variante: patterson
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
# El fichero `pat1.rcp` del conjunto de instancias de Patterson (formato RCP: primera línea, número de actividades y de recursos; segunda, disponibilidad de cada recurso; después, una línea por actividad con su duración, sus necesidades de recursos, el número de sucesoras y sus sucesoras) describe un proyecto con actividades ficticias de inicio y fin de duración cero:

# %%
#| tbl-cap: "Contenido del fichero"
#| label: tbl-fichero
contenido_rcp = '''14      3
2       1       2
0       0       0       0       3       2       3       4
6       1       0       0       2       9       10
4       0       0       0       3       5       6       7
3       0       0       0       2       8       11
1       0       0       0       1       10
6       1       0       1       1       12
2       1       0       0       2       8       11
1       0       0       0       1       13
4       0       1       1       1       14
3       0       0       1       1       12
2       0       0       1       1       12
3       0       1       0       1       13
5       0       0       0       1       14
0       0       0       0       0
'''
print(contenido_rcp)

# %%
pregunta_1 = "1. Lea el fichero con `read_rcp` (escribiéndolo antes en disco), construya el cuadro de prelaciones y el proyecto. ¿Cuántas actividades reales, ficticias, nodos y rutas tiene?  "
pregunta_2 = "2. Calcule τ = ‖R·t‖∞ y compárelo con la duración obtenida por el cálculo de tiempos (CPM). Indique las actividades críticas.  "
pregunta_3 = "3. Calcule la SVD de R y el cociente σ₁/σ₂. Ordene las actividades por |v₁| y mida la precisión del ranking: fracción de las k primeras (k = número de críticas) que son críticas.  "
pregunta_4 = "4. Compare el tamaño de R con el del grafo. ¿Cómo crece el número de rutas con el tamaño del proyecto y qué implica para proyectos de cientos de actividades?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
from pmule import read_rcp
with open('pat1.rcp', 'w') as f:
    f.write(contenido_rcp)
datos_rcp = read_rcp('pat1.rcp')
datos_proyecto = pd.DataFrame({'predecessor': pd.Series({k: (v or '---') for k, v in datos_rcp['predecessors'].items()}),
                               'duration': datos_rcp['duration']['duration']})
datos_proyecto.index.name = 'activity'
datos_proyecto.T

# %%
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
t = datos_proyecto['duration'][R.columns]
Markdown(f'{len(proyecto)} actividades (incluidas las dos ficticias del formato, de duración 0), {len(proyecto.dummies)} ficticias añadidas por el grafo, {len(proyecto.nodes)} nodos y {R.shape[0]} rutas.')

# %%
#| fig-cap: "Grafo PERT de la instancia"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=0.6, nodesep=0.3)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
Rt = R @ t
tau = float(Rt.max())
cpm = proyecto.duration(datos_proyecto['duration'])
criticas = sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts))
Markdown(f'τ = ‖R·t‖∞ = {tau:.0f}; CPM = {cpm}; coinciden: {tau == cpm}. Actividades críticas: {", ".join(criticas)}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)
v1 = np.abs(Vt[0, :])
ranking = pd.Series(v1, index=R.columns).sort_values(ascending=False)
k = len(criticas)
precision = len(set(ranking.index[:k]) & set(criticas)) / k
display(pd.DataFrame({'v₁': ranking.round(3), 'crítica': [a in criticas for a in ranking.index]}).T)
Markdown(f'σ₁/σ₂ = {S[0]/S[1]:.2f}. La mejor clasificada, {ranking.index[0]}, {"es" if ranking.index[0] in criticas else "no es"} crítica; precisión en las {k} primeras: {precision:.2f}.')

# %% [markdown]
# En el conjunto de Patterson (110 instancias) la actividad mejor clasificada por $v_1$ es crítica en el 77 % de los casos, y la identificación falla sobre todo cuando σ₁/σ₂ está cerca de 1 (dos direcciones dominantes). El ranking es informativo, no determinista.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
Markdown(f'''El grafo tiene {len(proyecto.nodes)} nodos y {len(proyecto.activities)} arcos, pero R tiene {R.shape[0]} filas: el número de rutas crece de forma combinatoria con la anchura del grafo (en las instancias de 51 actividades de Patterson hay hasta varios miles). Por eso la fórmula es útil como herramienta analítica (sensibilidad, estructura, simulación vectorizada), mientras que para calcular τ en proyectos grandes basta el cálculo de tiempos hacia delante, que es lineal en el número de arcos.''')
