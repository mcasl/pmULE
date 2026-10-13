# ---
# title: Reduccion 01
# subtitle: "8 actividades, reducción de 3 semanas"
# coleccion: 10_reduccion
# variante: cur-y-ackoff
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
# Considere el proyecto cuyo cuadro de prelaciones se adjunta, con la duración normal y la duración tope (mínima) de cada actividad, en semanas, y el coste (en k€) de ejecutarla en cada una de esas duraciones. Se supone coste lineal entre ambas.

# %%
pregunta_1 = "1. Calcule el coste unitario de reducción (CUR) de cada actividad.  "
pregunta_2 = "2. Determine la duración del proyecto, su camino crítico y el coste del proyecto con las duraciones normales.  "
pregunta_3 = "3. Reduzca la duración del proyecto en 3 semanas con el mínimo sobrecoste aplicando el algoritmo de Ackoff-Sasieni paso a paso. Indique en cada paso qué actividad(es) se acortan y por qué.  "
pregunta_4 = "4. Indique el coste final del proyecto y su(s) nuevo(s) camino(s) crítico(s). ¿Por qué en algún paso ha hecho falta acortar más de una actividad a la vez?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, cost_normal, cost_cap
    ('A'     , '---'      , 3       , 3           , 60         , 60      ),
    ('B'     , '---'      , 5       , 5           , 60         , 60      ),
    ('C'     , '---'      , 6       , 3           , 70         , 85      ),
    ('D'     , 'A,B,C'    , 3       , 2           , 30         , 34      ),
    ('E'     , 'A,B,C'    , 4       , 1           , 50         , 62      ),
    ('F'     , 'D,E'      , 3       , 1           , 20         , 22      ),
    ('G'     , 'D'        , 7       , 4           , 50         , 68      ),
    ('H'     , 'F'        , 3       , 1           , 70         , 88      ),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'cost_normal', 'cost_cap']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a el lanzamiento de una aplicación móvil; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de requisitos',
    'B': 'Diseño de la arquitectura',
    'C': 'Diseño de interfaz y prototipo',
    'D': 'Base de datos y modelo',
    'E': 'API del servidor',
    'F': 'Pantallas principales',
    'G': 'Módulo de pagos',
    'H': 'Notificaciones y sincronización',
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

# %% [markdown]
# $CUR = \dfrac{\text{coste tope} - \text{coste normal}}{\text{duración normal} - \text{duración tope}}$. Las actividades cuya duración tope coincide con la normal no pueden acortarse: su CUR es infinito.

# %%
reducible = datos_proyecto['duration'] > datos_proyecto['cap_duration']
datos_proyecto['ucr'] = np.where(reducible,
                                 (datos_proyecto['cost_cap'] - datos_proyecto['cost_normal']) / (datos_proyecto['duration'] - datos_proyecto['cap_duration']).replace(0, np.nan),
                                 np.inf)
datos_proyecto[['duration', 'cap_duration', 'cost_normal', 'cost_cap', 'ucr']]

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| fig-cap: "Grafo PERT con las duraciones normales y el camino crítico"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %%
duracion_normal = proyecto.duration(datos_proyecto['duration'])
coste_normal = int(datos_proyecto['cost_normal'].sum())
camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])
Markdown(f'Con las duraciones normales el proyecto dura {duracion_normal} semanas y cuesta {coste_normal} k€.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# En cada paso del algoritmo de Ackoff-Sasieni se acorta una semana la ruta (o rutas) crítica(s) eligiendo, entre las actividades críticas con margen, la combinación de menor CUR que reduzca *todas* las rutas críticas a la vez. Cuando hay varias rutas críticas sin actividades comunes baratas hay que acortar una actividad de cada una, y el coste del paso es la suma. La tabla muestra, para cada ruta, el CUR de sus actividades y su duración tras cada paso; las últimas filas, el margen de reducción que le queda a cada actividad.

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'], reduction=3, report=False)
tabla_ackoff

# %%
resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],
                        'sobrecoste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},
                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))
resumen['duración'] = duracion_normal - resumen.index
resumen['sobrecoste acumulado'] = resumen['sobrecoste del paso'].cumsum()
resumen

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
duracion_reducida = proyecto.duration(duraciones_reducidas)
coste_final = coste_normal + resumen['sobrecoste del paso'].sum()
camino_critico_reducido = proyecto.display_critical_path(duraciones_reducidas)
pasos_dobles = [p for p, acts in pasos.items() if len(acts) > 1]
Markdown(f'''El proyecto pasa a durar {duracion_reducida} semanas con un coste de {coste_normal} + {resumen["sobrecoste del paso"].sum():.0f} = {coste_final:.0f} k€. Al acortar la ruta crítica, otras rutas con holgura pequeña alcanzan su duración y pasan a ser críticas; a partir de ese momento reducir una sola actividad no basta, salvo que sea común a todas las rutas críticas, y hay que acortar una actividad de cada ruta (paso(s) {", ".join(str(p + 1) for p in pasos_dobles)}), con el coste sumado.''')

# %%
#| fig-cap: "Grafo PERT con las duraciones reducidas"
#| label: fig-pertReducido
proyecto.pert(durations=duraciones_reducidas, ranksep=1)
