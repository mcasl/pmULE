# ---
# title: Reduccion 02
# subtitle: "8 actividades, costes indirectos de 9 k€/semana"
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
# Considere el proyecto cuyo cuadro de prelaciones se adjunta, con la duración normal y la duración tope de cada actividad (semanas) y el coste directo (k€) de ejecutarla en cada una. Además, el proyecto tiene unos costes indirectos (oficina, alquileres, seguros) de 9 k€ por semana de duración.

# %%
pregunta_1 = "1. Calcule el coste unitario de reducción (CUR) de cada actividad.  "
pregunta_2 = "2. Determine la duración normal del proyecto y su coste total (directo más indirecto).  "
pregunta_3 = "3. Aplique el algoritmo de Ackoff-Sasieni hasta alcanzar la duración tope del proyecto, anotando en cada paso el sobrecoste directo.  "
pregunta_4 = "4. Construya la curva de coste total frente a duración y determine la duración de coste mínimo. ¿Qué regla de parada se deduce comparando el CUR del paso con el coste indirecto semanal?  "
pregunta_5 = "5. ¿Por qué no se puede reducir el proyecto por debajo de su duración tope aunque queden actividades con margen de reducción?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, cost_normal, cost_cap
    ('A'     , '---'      , 3       , 1           , 90         , 98      ),
    ('B'     , '---'      , 6       , 5           , 30         , 35      ),
    ('C'     , '---'      , 2       , 1           , 40         , 42      ),
    ('D'     , 'B'        , 6       , 3           , 60         , 84      ),
    ('E'     , 'A,C,D'    , 5       , 3           , 90         , 108     ),
    ('F'     , 'E'        , 4       , 2           , 60         , 64      ),
    ('G'     , 'E'        , 8       , 7           , 20         , 28      ),
    ('H'     , 'G'        , 8       , 6           , 70         , 72      ),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'cost_normal', 'cost_cap']).set_index('activity')
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
coste_indirecto_semanal = 9
duracion_normal = proyecto.duration(datos_proyecto['duration'])
coste_directo_normal = int(datos_proyecto['cost_normal'].sum())
camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])
Markdown(f'Duración normal: {duracion_normal} semanas. Coste directo {coste_directo_normal} k€ + indirecto '
         f'{coste_indirecto_semanal} × {duracion_normal} = {coste_indirecto_semanal * duracion_normal} k€; total '
         f'{coste_directo_normal + coste_indirecto_semanal * duracion_normal} k€.')

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
    costs=datos_proyecto['ucr'], reduction=100, report=False)
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
curva = pd.DataFrame({'duración': [duracion_normal] + resumen['duración'].tolist(),
                      'coste directo': [coste_directo_normal] + (coste_directo_normal + resumen['sobrecoste acumulado']).tolist()})
curva['coste indirecto'] = coste_indirecto_semanal * curva['duración']
curva['coste total'] = curva['coste directo'] + curva['coste indirecto']
curva = curva.set_index('duración')
curva

# %%
#| fig-cap: "Curva coste-duración del proyecto"
#| label: fig-curvaCoste
ax = curva.plot(marker='o', figsize=(7, 4))
ax.set_ylabel('k€'); ax.set_xlabel('duración (semanas)'); ax.invert_xaxis();

# %%
duracion_optima = int(curva['coste total'].idxmin())
Markdown(f'''El coste total es mínimo, {curva["coste total"].min():.0f} k€, para una duración de {duracion_optima} semanas. Cada semana de reducción cuesta el CUR del paso y ahorra {coste_indirecto_semanal} k€ de indirectos, así que compensa reducir mientras el sobrecoste del paso sea menor que {coste_indirecto_semanal}; como los pasos de Ackoff-Sasieni tienden a ser cada vez más caros, en cuanto el CUR del paso supera el coste indirecto semanal se para (conviene comprobar la tabla completa por si un paso posterior volviera a ser barato).''')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
Markdown(f'''La duración tope del proyecto, {duracion_normal - len(pasos)} semanas, se alcanza cuando todas las actividades de alguna ruta crítica están en su duración tope: esa ruta ya no puede acortarse y marca la duración mínima, aunque otras actividades (de otras rutas, o críticas pero no de esa ruta) conserven margen. Margen que queda por actividad: {{a: int(m) for a, m in margen_restante[margen_restante > 0].items()}}.''')
