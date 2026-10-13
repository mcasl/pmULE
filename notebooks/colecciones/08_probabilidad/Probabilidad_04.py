# ---
# title: Probabilidad 04
# subtitle: "9 actividades; segunda ruta a 1 semana(s) de la crítica y mucho más variable"
# coleccion: 08_probabilidad
# variante: casi-critica-variable
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en semanas) se adjunta.

# %%
pregunta_1 = "1. Determine la duración media del proyecto, la ruta crítica y la ruta que le sigue en duración.  "
pregunta_2 = "2. Calcule la probabilidad de terminar en 22 semanas o menos según el método PERT (aproximación normal por la ruta crítica).  "
pregunta_3 = "3. Calcule la misma probabilidad considerando únicamente la segunda ruta. ¿Qué ruta da la probabilidad más baja? ¿Cuál es entonces una cota razonable de la probabilidad de cumplir el plazo?  "
pregunta_4 = "4. Compruebe con Monte Carlo (10 000 réplicas, duraciones normales) cuál es la probabilidad real y explique por qué el método PERT se equivoca aquí más de lo habitual.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 7       , 0.3               ),
    ('B'     , '---'      , 3       , 0.2               ),
    ('C'     , '---'      , 3       , 1.5               ),
    ('D'     , 'C'        , 5       , 1.8               ),
    ('E'     , 'A,B,C'    , 4       , 0.3               ),
    ('F'     , 'B'        , 7       , 0.2               ),
    ('G'     , 'D'        , 1       , 1.8               ),
    ('H'     , 'F,G'      , 5       , 0.2               ),
    ('I'     , 'E,H'      , 5       , 0.2               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
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
    'I': 'Programación de autómatas',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
medias = R @ datos_proyecto['duration'][R.columns]
varianzas = R @ datos_proyecto['variance'][R.columns]

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT con tiempos y camino crítico"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %%
#| tbl-cap: "Duración media y varianza de cada ruta"
#| label: tbl-rutas
rutas = pd.DataFrame({'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in R.index],
                      'media': medias, 'varianza': varianzas, 'σ': np.sqrt(varianzas)}, index=R.index).sort_values('media', ascending=False)
rutas.round(3)

# %%
critica, segunda = rutas.index[0], rutas.index[1]
Markdown(f'La ruta crítica es {critica} ({rutas.loc[critica, "media"]:.0f} semanas, σ = {rutas.loc[critica, "σ"]:.2f}); '
         f'la segunda, {segunda} ({rutas.loc[segunda, "media"]:.0f} semanas, σ = {rutas.loc[segunda, "σ"]:.2f}), es mucho más variable.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
plazo = 22
p_critica = norm.cdf((plazo - rutas.loc[critica, 'media']) / rutas.loc[critica, 'σ'])
Markdown(f'Por la ruta crítica: z = ({plazo} − {rutas.loc[critica, "media"]:.0f}) / {rutas.loc[critica, "σ"]:.2f} = {(plazo - rutas.loc[critica, "media"]) / rutas.loc[critica, "σ"]:.2f}, P = {p_critica:.3f}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
p_segunda = norm.cdf((plazo - rutas.loc[segunda, 'media']) / rutas.loc[segunda, 'σ'])
Markdown(f'''Por la segunda ruta: P = {p_segunda:.3f}. Para cumplir el plazo deben cumplirlo *todas* las rutas, así que la probabilidad real no puede superar la menor de las dos, {min(p_critica, p_segunda):.3f}; y como las rutas comparten actividades y no son independientes, tampoco es simplemente el producto. {"La ruta crítica sigue mandando también en probabilidad." if p_critica < p_segunda else "Es la segunda ruta, y no la crítica, la que manda en probabilidad, aunque no sea la más larga en media."}''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
rng = np.random.default_rng(7)
N = 10000
muestras = pd.DataFrame({a: np.clip(rng.normal(datos_proyecto.loc[a, 'duration'], datos_proyecto.loc[a, 'standard_deviation'], N), 0, None) for a in R.columns})
duracion_mc = (muestras @ R.T).max(axis='columns')
p_mc = float((duracion_mc <= plazo).mean())
pd.Series({'PERT (ruta crítica)': p_critica, 'segunda ruta': p_segunda, 'Monte Carlo': p_mc}).round(3).to_frame(f'P(T ≤ {plazo})')

# %% [markdown]
# El método PERT sólo mira la ruta de mayor duración media. Cuando otra ruta casi tan larga es mucho más variable, es *ella* la que más veces hace fallar el plazo: en muchas réplicas resulta la más larga. La probabilidad real es menor que la de cada ruta por separado, porque basta que una de ellas se pase. La regla práctica es examinar todas las rutas casi críticas y quedarse con la probabilidad más baja, o directamente simular.
