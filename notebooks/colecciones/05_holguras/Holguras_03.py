# ---
# title: Holguras 03
# subtitle: "10 actividades, 2 ruta(s) crítica(s)"
# coleccion: 05_holguras
# variante: varias-criticas
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta.

# %%
pregunta_1 = "1. Calcule para cada actividad sus fechas más tempranas y más tardías de inicio y fin, y sus holguras total, libre e independiente.  "
pregunta_2 = "2. Determine el camino crítico. ¿Cuántas rutas críticas tiene el proyecto?  "
pregunta_3 = "3. ¿Qué actividades pueden retrasarse sin afectar ni al plazo del proyecto ni al inicio más temprano de ninguna otra actividad? ¿Cuáles pueden retrasarse sin afectar al plazo pero obligando a retrasar a alguna sucesora? Justifíquelo con las holguras.  "
pregunta_4 = "4. La actividad F se retrasa tanto como permite su holgura total (3 semanas). Compruebe que el plazo no cambia y determine qué ocurre con las fechas y holguras de sus sucesoras.  "
pregunta_5 = "5. Suponga que las predecesoras de I terminan en su fecha más tardía. ¿Cuánto puede retrasarse entonces I sin afectar a nadie? Relaciónelo con su holgura independiente.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 8       ),
    ('B'     , '---'      , 8       ),
    ('C'     , '---'      , 3       ),
    ('D'     , 'A,B,C'    , 1       ),
    ('E'     , 'D'        , 8       ),
    ('F'     , 'B'        , 6       ),
    ('G'     , 'C,F'      , 5       ),
    ('H'     , 'E,F'      , 8       ),
    ('I'     , 'G'        , 3       ),
    ('J'     , 'H,I'      , 3       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
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

# %%
#| fig-cap: "Grafo PERT con tiempos de los nodos y camino crítico"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %% [markdown]
# Con las fechas de cada actividad (ES, EF, LS, LF) y las de sus vecinas se definen las tres holguras:
#
# - **Total**: $H_t = LS - ES = LF - EF$. Retraso máximo sin alargar el proyecto, suponiendo que las predecesoras acaban lo antes posible y las sucesoras pueden empezar lo más tarde posible.
# - **Libre**: $H_l = \min_{s \in suc} ES_s - EF$. Retraso máximo sin mover el inicio más temprano de ninguna sucesora.
# - **Independiente**: $H_i = \max\bigl(0,\ \min_{s \in suc} ES_s - \max_{p \in pred} LF_p - D\bigr)$. Retraso disponible incluso si las predecesoras acaban lo más tarde posible y las sucesoras deben empezar lo antes posible.
#
# Siempre $H_t \ge H_l \ge H_i$.

# %%
#| tbl-cap: "Calendario y holguras del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duracion_proyecto = proyecto.duration(datos_proyecto['duration'])
rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])
for nombre, acts in rutas_criticas.items():
    print(f'{nombre}: {" - ".join(acts)}')
Markdown(f'La duración del proyecto es {duracion_proyecto} semanas y hay {len(rutas_criticas)} ruta(s) crítica(s). '
         'Las actividades críticas son las de holgura total nula; cuando varias rutas comparten la duración máxima, '
         'todas ellas son críticas y un retraso en cualquiera alarga el proyecto.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
sin_afectar_a_nadie = calendario.index[calendario['H_libre'] > 0].tolist()
solo_sucesoras = calendario.index[(calendario['H_total'] > calendario['H_libre'])].tolist()
Markdown(f'''
- Pueden retrasarse (hasta su holgura libre) sin mover a ninguna otra actividad: **{", ".join(sin_afectar_a_nadie)}**.
- Tienen holgura total mayor que la libre, así que agotar la total obliga a retrasar sucesoras aunque el plazo se mantenga: **{", ".join(solo_sucesoras)}**.
''')

# %%
#| tbl-cap: "Actividades cuyo retraso repercute en sus sucesoras"
#| label: tbl-holgurasDistintas
calendario.loc[solo_sucesoras, ['H_total', 'H_libre', 'H_independiente']]

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Un retraso de F igual a su holgura total equivale a alargar su duración en 3 semanas. Recalculamos el calendario con esa duración.

# %%
duraciones_retraso = datos_proyecto['duration'].copy()
duraciones_retraso['F'] += 3
calendario_retraso = proyecto.calendar(duraciones_retraso)
sucesoras = sorted(s for s, ps in prelaciones.immediate_predecessor.items() if 'F' in ps)
comparacion = pd.concat({'antes': calendario.loc[sucesoras, ['inicio_mas_temprano', 'H_total', 'H_libre']],
                         'después': calendario_retraso.loc[sucesoras, ['inicio_mas_temprano', 'H_total', 'H_libre']]}, axis=1)
comparacion

# %%
Markdown(f'La duración del proyecto sigue siendo {proyecto.duration(duraciones_retraso)} semanas, '
         f'pero F ha consumido holgura que compartía con sus sucesoras: las que tenían inicio más temprano '
         f'anterior al nuevo fin de F se retrasan y su holgura total se reduce; alguna pasa a ser crítica.')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
fila = calendario.loc['I']
predecesoras = sorted(prelaciones.immediate_predecessor['I'])
sucesoras_V = sorted(s for s, ps in prelaciones.immediate_predecessor.items() if 'I' in ps)
fin_tardio_pred = int(calendario.loc[predecesoras, 'fin_mas_tardio'].max())
inicio_temprano_suc = int(calendario.loc[sucesoras_V, 'inicio_mas_temprano'].min())
margen = inicio_temprano_suc - fin_tardio_pred - int(fila['duracion'])
Markdown(f'''Si las predecesoras de I ({", ".join(predecesoras)}) acaban en su fecha más tardía, I no puede empezar antes de la semana {fin_tardio_pred}. Para no mover a sus sucesoras ({", ".join(sucesoras_V)}) debe acabar antes de la semana {inicio_temprano_suc}. Le quedan {inicio_temprano_suc} − {fin_tardio_pred} − {int(fila["duracion"])} = {margen} semanas, es decir, su holgura independiente, {int(fila["H_independiente"])} (si el cálculo sale negativo la holgura es cero). Su holgura libre, {int(fila["H_libre"])}, supone en cambio que las predecesoras acaban lo antes posible.''')
