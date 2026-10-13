# ---
# title: ValorGanado 05
# subtitle: "9 actividades; punto de control en la semana 10: plazo bien coste mal"
# coleccion: 12_valor_ganado
# variante: plazo-bien-coste-mal
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por actividad (BAC, k€) se adjunta; el coste planificado de cada actividad se reparte uniformemente en su duración. Al final de la semana 10 se realiza un punto de control.

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 6       , 10 ),
    ('B'     , '---'      , 7       , 50 ),
    ('C'     , '---'      , 7       , 20 ),
    ('D'     , 'A'        , 3       , 40 ),
    ('E'     , 'B,C'      , 2       , 50 ),
    ('F'     , 'C'        , 1       , 10 ),
    ('G'     , 'C'        , 1       , 40 ),
    ('H'     , 'D,E,G'    , 6       , 30 ),
    ('I'     , 'F,H'      , 6       , 60 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %%
#| tbl-cap: "Datos del punto de control (final de la semana 10)"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 4              , 11, 100      ),
    ('B'     , 4              , 60, 100      ),
    ('C'     , 4              , 26, 100      ),
    ('D'     , 3              , 44, 100      ),
    ('E'     , 2              , 65, 100      ),
    ('F'     , 1              , 15, 100      ),
    ('G'     , 1              , 44, 100      ),
    ('H'     , 3              , 25, 75       ),
    ('I'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %%
pregunta_1 = "1. Calcule PV, EV y AC en el punto de control, y los índices CPI y SPI.  "
pregunta_2 = "2. El director informa al comité de que «vamos adelantados». ¿Es correcta la lectura? ¿Qué parte de la situación oculta?  "
pregunta_3 = "3. ¿Qué actividades explican el sobrecoste? Compare, actividad por actividad, el coste real con el valor ganado.  "
pregunta_4 = "4. Estime el coste final (EAC = BAC/CPI) y la variación al final (VAC). ¿Compensa el adelanto el sobrecoste? ¿Con qué dato del contrato lo decidiría?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
BAC_total = float(datos_proyecto['BAC'].sum())
duracion_planificada = proyecto.duration(datos_proyecto['duration'])
semana_control = 10
datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)
gantts = EarnedValue(proyecto).calcula_gantts(datos_valor_ganado, planned_durations_label='duration',
                                             actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',
                                             percentage_complete_label='performed', tikz=True, params=dict(GANTT_PARAMS))

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "PV: valor planificado"
#| label: fig-ganttPV
pinta_tikz(gantts['gantt_PV'])

# %%
#| fig-cap: "EV: valor ganado"
#| label: fig-ganttEV
pinta_tikz(gantts['gantt_EV'])

# %%
acumulados = gantts['acumulados']
PV = float(acumulados.loc[semana_control, 'PV'])
EV = float((datos_valor_ganado['BAC'] * datos_valor_ganado['performed'] / 100).sum())
AC = float(datos_valor_ganado['AC'].sum())
CPI, SPI = EV / AC, EV / PV
pd.Series({'PV': PV, 'EV': EV, 'AC': AC, 'CV = EV − AC': EV - AC, 'SV = EV − PV': EV - PV, 'CPI': CPI, 'SPI': SPI}).round(3).to_frame('valor').T

# %%
#| fig-cap: "Curvas S"
#| label: fig-curvasS
plot_curvas_valor_ganado(gantts)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
Markdown(f'''El adelanto es real (SPI = {SPI:.2f} ≥ 1: se ha ganado valor por {EV:.0f} frente a {PV:.0f} previstos), pero ha costado {AC:.0f}: CPI = {CPI:.2f} < 1, cada euro gastado rinde {CPI:.2f}. La lectura del director oculta el sobrecoste, que es lo que determinará si el proyecto acaba en presupuesto.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
por_actividad = datos_valor_ganado[['BAC', 'performed', 'AC']].copy()
por_actividad['EV'] = por_actividad['BAC'] * por_actividad['performed'] / 100
calendario = proyecto.calendar(datos_proyecto['duration'])
frac_planificada = ((semana_control - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion'])
por_actividad['PV'] = por_actividad['BAC'] * frac_planificada
por_actividad['SV'] = por_actividad['EV'] - por_actividad['PV']
por_actividad['CV'] = por_actividad['EV'] - por_actividad['AC']
por_actividad.round(1)

# %%
caras = por_actividad[por_actividad['CV'] < -0.5].sort_values('CV')
Markdown(f'Las actividades con CV negativa explican el sobrecoste: {", ".join(caras.index)}; las de mayor peso son {", ".join(caras.index[:2])}.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
EAC = BAC_total / CPI
VAC = BAC_total - EAC
Markdown(f'''EAC = {BAC_total:.0f} / {CPI:.2f} = {EAC:.0f} k€, VAC = {VAC:.0f} k€ (sobrecoste previsto). Que el adelanto compense depende de lo que valga el tiempo en el contrato: una prima por entrega anticipada o una penalización evitada por semana. Si la prima por semana adelantada multiplicada por las semanas que se ganarían supera {-VAC:.0f} k€, compensa; si no, se está comprando tiempo demasiado caro.''')
