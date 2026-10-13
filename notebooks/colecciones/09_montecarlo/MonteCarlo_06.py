# ---
# title: MonteCarlo 06
# subtitle: "9 actividades; convergencia de la media y del percentil 90 con N"
# coleccion: 09_montecarlo
# variante: convergencia
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (semanas, distribución normal) se adjunta. Se quiere saber cuántas réplicas de Monte Carlo hacen falta para estimar con precisión la duración del proyecto.

# %%
pregunta_1 = "1. Construya la matriz de rutas y simule el proyecto con N = 20, 200, 2 000 y 20 000 réplicas. Anote en cada caso la media, la desviación típica, el percentil 90 y la probabilidad de terminar en 34 semanas.  "
pregunta_2 = "2. Repita cada simulación 10 veces con semillas distintas y calcule la dispersión (desviación típica) de la media estimada entre repeticiones. ¿Cómo decrece con N?  "
pregunta_3 = "3. Compare esa dispersión con la fórmula del error típico de la media, σ/√N. ¿Cuántas réplicas hacen falta para que el error de la media sea menor de 0,1 semanas?  "
pregunta_4 = "4. ¿Necesita el percentil 90 más o menos réplicas que la media para la misma precisión? ¿Por qué?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 3       , 0.6               ),
    ('B'     , '---'      , 1       , 1                 ),
    ('C'     , '---'      , 2       , 0.1               ),
    ('D'     , 'A,B,C'    , 8       , 0.3               ),
    ('E'     , 'D'        , 7       , 0.4               ),
    ('F'     , 'E'        , 4       , 0.6               ),
    ('G'     , 'F'        , 2       , 0.1               ),
    ('H'     , 'F'        , 4       , 0.4               ),
    ('I'     , 'G,H'      , 7       , 0.6               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
mu = datos_proyecto.loc[R.columns, 'duration'].values[:, None]
sigma = datos_proyecto.loc[R.columns, 'standard_deviation'].values[:, None]
plazo = 34
def simular(N, semilla):
    rng = np.random.default_rng(semilla)
    T = np.clip(rng.normal(mu, sigma, (len(R.columns), N)), 0, None)
    return (R.values @ T).max(axis=0)

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
tamanos = [20, 200, 2000, 20000]
filas = {}
for N in tamanos:
    tau = simular(N, 0)
    filas[N] = {'media': tau.mean(), 'desviación típica': tau.std(ddof=1), 'percentil 90': np.quantile(tau, 0.9), f'P(T ≤ {plazo})': (tau <= plazo).mean()}
pd.DataFrame(filas).T.round(3)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
repeticiones = 10
disp = {}
for N in tamanos:
    medias = [simular(N, s).mean() for s in range(1, repeticiones + 1)]
    p90 = [np.quantile(simular(N, s), 0.9) for s in range(1, repeticiones + 1)]
    disp[N] = {'dispersión de la media': np.std(medias, ddof=1), 'dispersión del percentil 90': np.std(p90, ddof=1)}
dispersion = pd.DataFrame(disp).T
dispersion.round(4)

# %% [markdown]
# Al multiplicar N por 10 la dispersión se divide aproximadamente por √10 ≈ 3,16.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
sigma_tau = simular(20000, 0).std(ddof=1)
dispersion['σ/√N'] = [sigma_tau / np.sqrt(N) for N in tamanos]
N_necesario = int(np.ceil((sigma_tau / 0.1) ** 2))
display(dispersion.round(4))
Markdown(f'Con σ_τ ≈ {sigma_tau:.2f}, para un error típico de 0,1 semanas hacen falta N ≥ (σ/0,1)² ≈ {N_necesario} réplicas.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# El percentil 90 se estima con las réplicas de la cola, que son pocas (el 10 % de N), y su error es mayor que el de la media para el mismo N; la tabla de dispersiones lo muestra. Por eso las decisiones basadas en percentiles altos (plazos comprometidos al 90–95 %) necesitan más réplicas que las basadas en la media.
