# ---
# title: Reduccion 05
# subtitle: "8 actividades; deshacer la reducción más cara al relajar el plazo"
# coleccion: 10_reduccion
# variante: desaceleracion
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

GANTT_PARAMS = dict(arrows=True)   # flechas de prelación en los diagramas de Gantt (rojas las críticas)


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste unitario de reducción (k€/semana) se adjunta. El proyecto se había reducido cuatro semanas con el algoritmo de Ackoff-Sasieni cuando el cliente comunica que acepta la entrega una semana más tarde.

# %%
pregunta_1 = "1. Reduzca el proyecto cuatro semanas con el algoritmo de Ackoff-Sasieni y anote el sobrecoste de cada paso.  "
pregunta_2 = "2. Hay que devolver una semana. Un compañero propone deshacer el primer paso («fue el primero, es el más fácil de anular»). Otro propone deshacer el último. ¿Da lo mismo? Razónelo antes de calcular.  "
pregunta_3 = "3. Compruebe ambas opciones: duración resultante y dinero recuperado en cada caso. ¿Puede deshacerse un paso intermedio? ¿Qué regla general se deduce?  "
pregunta_4 = "4. ¿Por qué el sobrecoste marginal de cada semana de reducción es creciente (o al menos no decreciente) en Ackoff-Sasieni? ¿Qué forma tiene la curva coste-duración?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, ucr   
    ('A'     , '---'      , 2       , 1           , 2     ),
    ('B'     , '---'      , 5       , 4           , 5     ),
    ('C'     , 'A,B'      , 3       , 2           , 4     ),
    ('D'     , 'C'        , 8       , 6           , 1     ),
    ('E'     , 'B'        , 4       , 4           , np.inf),
    ('F'     , 'D'        , 3       , 3           , np.inf),
    ('G'     , 'D,E'      , 2       , 1           , 8     ),
    ('H'     , 'F,G'      , 4       , 2           , 8     ),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
duracion_normal = proyecto.duration(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'], reduction=4, report=False)
tabla_ackoff

# %%
resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],
                        'coste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},
                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))
sobrecoste = float(resumen['coste del paso'].sum())
display(resumen)
Markdown(f'Duración {duracion_normal} → {proyecto.duration(duraciones_reducidas)} semanas con un sobrecoste de {sobrecoste:.0f} k€.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# No da lo mismo. Deshacer un paso es devolver una semana a las actividades que se acortaron en él; cualquiera de los pasos devuelve una semana al proyecto (todas las rutas críticas se alargan una semana), pero el dinero recuperado es el coste de ese paso. Como los pasos de Ackoff-Sasieni son cada vez más caros, deshacer el último recupera más dinero que deshacer el primero.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
filas = {}
for paso, acts in pasos.items():
    d2 = duraciones_reducidas.copy()
    for a in acts:
        d2[a] += 1
    filas[paso + 1] = {'actividades devueltas': ', '.join(acts), 'duración resultante': proyecto.duration(d2),
                       'dinero recuperado': float(sum(datos_proyecto.loc[acts, 'ucr']))}
pd.DataFrame(filas).T

# %%
costes_pasos = {p: float(sum(datos_proyecto.loc[acts, 'ucr'])) for p, acts in pasos.items()}
mejor = max(costes_pasos, key=costes_pasos.get)
Markdown(f'''Todos los pasos devuelven el proyecto a {proyecto.duration(duraciones_reducidas) + 1} semanas, pero el que más dinero recupera es el paso {mejor + 1} ({costes_pasos[mejor]:.0f} k€). Un paso intermedio también puede deshacerse: lo que importa no es el orden en que se dieron sino su coste. Regla: para relajar el plazo se deshacen las reducciones de mayor coste primero (recorrer la tabla de Ackoff-Sasieni hacia atrás).''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# En cada paso el algoritmo elige la opción más barata disponible; las opciones baratas se agotan (las actividades llegan a su tope) o dejan de bastar (aparecen nuevas rutas críticas que exigen acortar varias actividades a la vez), así que el coste del paso siguiente nunca es menor. La curva coste-duración es por tanto convexa: cada semana adicional de reducción cuesta más que la anterior, y por eso la decisión de hasta dónde reducir se toma comparando el coste marginal con lo que vale cada semana (penalización, coste indirecto, prima).
