# ---
# title: Formula 06
# subtitle: "10 actividades, 7 rutas; criticidad Monte Carlo frente a |v₁|"
# coleccion: 13_formula_compleccion
# variante: svd-vs-montecarlo
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
# Dos formas de medir la importancia de una actividad: el índice de criticidad de Monte Carlo (fracción de réplicas en que está en la ruta más larga; usa las distribuciones de duración) y el peso |v₁| de la SVD de la matriz de rutas (usa sólo la estructura). Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (normal) se adjunta.

# %%
pregunta_1 = "1. Calcule el índice de criticidad de cada actividad con 20 000 réplicas (`monte_carlo`).  "
pregunta_2 = "2. Calcule |v₁| de la SVD de R y ordene las actividades.  "
pregunta_3 = "3. Compare los dos rankings: correlación de Spearman y actividades en las que más discrepan. ¿Qué información tiene uno que no tenga el otro?  "
pregunta_4 = "4. Repita el Monte Carlo multiplicando todas las desviaciones típicas por 3. ¿Se parece más o menos el índice de criticidad al ranking SVD? ¿Por qué?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 8       , 0.8               ),
    ('B'     , '---'      , 1       , 0.4               ),
    ('C'     , '---'      , 8       , 0.3               ),
    ('D'     , 'A,B,C'    , 1       , 1                 ),
    ('E'     , 'B,C'      , 8       , 0.6               ),
    ('F'     , 'E'        , 3       , 0.1               ),
    ('G'     , 'D'        , 5       , 0.1               ),
    ('H'     , 'E,G'      , 4       , 0.1               ),
    ('I'     , 'F,H'      , 4       , 1                 ),
    ('J'     , 'I'        , 8       , 0.3               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
t = datos_proyecto['duration'][R.columns]

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
dists = {a: ('normal', float(datos_proyecto.loc[a, 'duration']), float(datos_proyecto.loc[a, 'standard_deviation'])) for a in R.columns}
mc = proyecto.monte_carlo(dists, n_samples=20000, seed=5)
criticidad = mc.sensitivity_mean.reindex(R.columns)
criticidad.sort_values(ascending=False).round(3).to_frame('índice de criticidad').T

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)
v1 = pd.Series(np.abs(Vt[0, :]), index=R.columns)
v1.sort_values(ascending=False).round(3).to_frame('|v₁|').T

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
from scipy.stats import spearmanr
rho = spearmanr(criticidad, v1).correlation
comparacion = pd.DataFrame({'criticidad MC': criticidad, '|v₁|': v1})
comparacion['rango MC'] = comparacion['criticidad MC'].rank(ascending=False)
comparacion['rango SVD'] = comparacion['|v₁|'].rank(ascending=False)
comparacion['diferencia de rango'] = (comparacion['rango MC'] - comparacion['rango SVD']).abs()
display(comparacion.sort_values('diferencia de rango', ascending=False).round(3))
Markdown(f'Correlación de Spearman entre ambos rankings: ρ = {rho:.2f}.')

# %% [markdown]
# Monte Carlo sabe qué rutas son largas y variables; la SVD sólo sabe qué actividades son estructuralmente centrales. Una actividad por la que pasan muchas rutas cortas tiene |v₁| alto y criticidad baja; una actividad de una única ruta muy larga, lo contrario. Coinciden cuando la estructura y las duraciones apuntan a lo mismo.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
dists3 = {a: ('normal', m, 3 * s) for a, (_, m, s) in dists.items()}
mc3 = proyecto.monte_carlo(dists3, n_samples=20000, seed=5)
criticidad3 = mc3.sensitivity_mean.reindex(R.columns)
rho3 = spearmanr(criticidad3, v1).correlation
pd.DataFrame({'criticidad (σ)': criticidad, 'criticidad (3σ)': criticidad3, '|v₁|': v1}).round(3).T

# %%
cambio = 'sube' if rho3 > rho + 0.03 else ('baja' if rho3 < rho - 0.03 else 'apenas cambia')
Markdown(f'''Con σ triplicada ρ pasa de {rho:.2f} a {rho3:.2f} ({cambio}). Lo esperable es que con más incertidumbre más rutas tengan ocasión de ser la más larga y el índice de criticidad se reparta según cuántas rutas pasan por cada actividad, que es justo lo que mide la SVD, de modo que los rankings se acerquen; con poca incertidumbre la criticidad se concentra en la ruta crítica determinista y la estructura importa menos. Si aquí el cambio es pequeño es porque la estructura y las duraciones ya apuntan en la misma dirección: las actividades centrales son también las de la ruta crítica.''')
