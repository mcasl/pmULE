# ---
# title: ValorGanado 01
# subtitle: "9 actividades, un punto de control (semana 10), EAC/VAC/TCPI"
# coleccion: 12_valor_ganado
# variante: indicadores-eac
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por actividad (BAC, en k€) se adjunta. El coste planificado de cada actividad se reparte uniformemente a lo largo de su duración.

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 3       , 50 ),
    ('B'     , '---'      , 2       , 40 ),
    ('C'     , 'A,B'      , 2       , 10 ),
    ('D'     , 'A,B'      , 4       , 60 ),
    ('E'     , 'C'        , 2       , 20 ),
    ('F'     , 'D,E'      , 4       , 60 ),
    ('G'     , 'E'        , 5       , 60 ),
    ('H'     , 'G'        , 4       , 30 ),
    ('I'     , 'F,H'      , 4       , 30 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# Al final de la semana 10 se realiza un punto de control. Para cada actividad se anotan las semanas trabajadas hasta ese momento, el coste real incurrido (k€) y el porcentaje realizado.

# %%
#| tbl-cap: "Datos del punto de control (final de la semana 10)"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 2              , 55, 100      ),
    ('B'     , 2              , 48, 100      ),
    ('C'     , 2              , 10, 100      ),
    ('D'     , 3              , 78, 100      ),
    ('E'     , 3              , 18, 100      ),
    ('F'     , 3              , 72, 100      ),
    ('G'     , 3              , 40, 75       ),
    ('H'     , 0              , 0 , 0        ),
    ('I'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %%
pregunta_1 = "1. Dibuje los tres diagramas de Gantt del valor ganado: valor planificado (PV), valor ganado (EV) y coste real (AC).  "
pregunta_2 = "2. Calcule PV, EV y AC en el punto de control, las variaciones CV y SV y los índices CPI y SPI. Interprete la situación del proyecto.  "
pregunta_3 = "3. Estime el coste final del proyecto (EAC) con tres hipótesis: (a) el resto se ejecuta según presupuesto, (b) se mantiene la eficiencia en coste actual, (c) se mantienen las eficiencias en coste y en plazo. Calcule la variación al final (VAC) en cada caso.  "
pregunta_4 = "4. Calcule el índice de rendimiento necesario para terminar en presupuesto (TCPI). ¿Es realista?  "

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

# %%
#| fig-cap: "Grafo PERT de la planificación"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# En el Gantt del valor planificado cada casilla lleva el coste planificado de la semana (BAC / duración). En el del valor ganado, el BAC multiplicado por el porcentaje realizado se reparte entre las semanas trabajadas. En el del coste real, el coste incurrido entre las semanas trabajadas.

# %%
datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)
vg = EarnedValue(proyecto)
gantts = vg.calcula_gantts(datos_valor_ganado, planned_durations_label='duration',
                           actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',
                           percentage_complete_label='performed', tikz=True,
                           params=dict(period_width=1, row_height=0.7, activity_relative_height=0.6, **GANTT_PARAMS,
                                       critical_background_color='white!90!green', regular_background_color='white!90!green',
                                       regular_text_color='black', critical_text_color='black'))

# %%
#| fig-cap: "PV: valor planificado"
#| label: fig-ganttPV
pinta_tikz(gantts['gantt_PV'])

# %%
#| fig-cap: "EV: valor ganado"
#| label: fig-ganttEV
pinta_tikz(gantts['gantt_EV'])

# %%
#| fig-cap: "AC: coste real"
#| label: fig-ganttAC
pinta_tikz(gantts['gantt_AC'])

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
semana_control = 10
acumulados = gantts['acumulados']
PV = float(acumulados.loc[semana_control, 'PV'])
EV = float((datos_valor_ganado['BAC'] * datos_valor_ganado['performed'] / 100).sum())
AC = float(datos_valor_ganado['AC'].sum())
CV, SV = EV - AC, EV - PV
CPI, SPI = EV / AC, EV / PV
pd.Series({'PV': PV, 'EV': EV, 'AC': AC, 'CV': CV, 'SV': SV, 'CPI': CPI, 'SPI': SPI, 'BAC': BAC_total}).round(3).to_frame('valor').T

# %%
#| fig-cap: "Curvas S del proyecto"
#| label: fig-curvasS
plot_curvas_valor_ganado(gantts)

# %%
Markdown(f'''En la semana {semana_control} se ha ganado valor por {EV:.1f} k€ frente a los {PV:.1f} planificados (SV = {SV:.1f}, SPI = {SPI:.2f}: {"retraso" if SPI < 1 else "adelanto"}) y se han gastado {AC:.1f} k€ (CV = {CV:.1f}, CPI = {CPI:.2f}: por cada euro gastado se obtienen {CPI:.2f} euros de valor, {"sobrecoste" if CPI < 1 else "ahorro"}).''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Las tres previsiones habituales del coste final:
#
# - (a) $EAC = AC + (BAC - EV)$: el trabajo restante costará lo presupuestado.
# - (b) $EAC = BAC / CPI = AC + (BAC - EV)/CPI$: se mantiene la eficiencia en coste.
# - (c) $EAC = AC + (BAC - EV)/(CPI \cdot SPI)$: el retraso también encarece el resto.
#
# En todas, $VAC = BAC - EAC$.

# %%
previsiones = pd.DataFrame({
    '(a) resto según presupuesto': [AC + (BAC_total - EV)],
    '(b) misma eficiencia en coste': [BAC_total / CPI],
    '(c) misma eficiencia en coste y plazo': [AC + (BAC_total - EV) / (CPI * SPI)]},
    index=['EAC']).T
previsiones['VAC'] = BAC_total - previsiones['EAC']
previsiones.round(1)

# %%
Markdown(f'La hipótesis (a) es la más optimista: supone que los problemas no se repetirán. La (c) es la más pesimista. Lo habitual es tomar (b) como previsión central: {BAC_total / CPI:.1f} k€ frente a los {BAC_total:.0f} presupuestados.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# $TCPI = \dfrac{BAC - EV}{BAC - AC}$: eficiencia en coste que habría que mantener en lo que queda para terminar exactamente en presupuesto.

# %%
TCPI = (BAC_total - EV) / (BAC_total - AC)
Markdown(f'TCPI = ({BAC_total:.0f} − {EV:.1f}) / ({BAC_total:.0f} − {AC:.1f}) = {TCPI:.2f}. '
         f'Hasta ahora el proyecto rinde CPI = {CPI:.2f}; pasar a {TCPI:.2f} exige una mejora de '
         f'{(TCPI / CPI - 1) * 100:.0f} %, {"poco realista sin cambios de fondo" if TCPI > 1.1 else "alcanzable con medidas correctoras"}.')
