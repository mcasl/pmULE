# ---
# title: ValorGanado 06
# subtitle: "8 actividades; resolución con CPI y SPI intercambiados"
# coleccion: 12_valor_ganado
# variante: diagnostico
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
# Para el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por actividad (k€) se adjunta, con el punto de control de la semana 8 que también se da, un alumno ha entregado esta resolución:
#
# > *PV = 100, EV = 92, AC = 122. CV = AC − EV = 30 (positiva: ahorro). CPI = EV / PV = 0.92; SPI = EV / AC = 0.75. Como CPI < 1 el proyecto va retrasado y como SPI < 1 va con sobrecoste. EAC = BAC × CPI = 220 k€.*

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 4       , 30 ),
    ('B'     , '---'      , 1       , 20 ),
    ('C'     , '---'      , 4       , 10 ),
    ('D'     , 'A,B'      , 6       , 20 ),
    ('E'     , 'C'        , 6       , 40 ),
    ('F'     , 'C,D'      , 1       , 40 ),
    ('G'     , 'E,F'      , 6       , 50 ),
    ('H'     , 'E'        , 4       , 30 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %%
#| tbl-cap: "Datos del punto de control (final de la semana 8)"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 6              , 39, 100      ),
    ('B'     , 1              , 30, 100      ),
    ('C'     , 4              , 15, 100      ),
    ('D'     , 2              , 6 , 25       ),
    ('E'     , 4              , 32, 67       ),
    ('F'     , 0              , 0 , 0        ),
    ('G'     , 0              , 0 , 0        ),
    ('H'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %%
pregunta_1 = "1. Compruebe PV, EV y AC.  "
pregunta_2 = "2. Localice todos los errores de la resolución (hay al menos cuatro) y explique cada uno.  "
pregunta_3 = "3. Escriba la resolución correcta: CV, SV, CPI, SPI, EAC y VAC, con su interpretación.  "

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
semana_control = 8
datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)
gantts = EarnedValue(proyecto).calcula_gantts(datos_valor_ganado, planned_durations_label='duration',
                                             actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',
                                             percentage_complete_label='performed', tikz=False)

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
PV = float(gantts['acumulados'].loc[semana_control, 'PV'])
EV = float((datos_valor_ganado['BAC'] * datos_valor_ganado['performed'] / 100).sum())
AC = float(datos_valor_ganado['AC'].sum())
Markdown(f'PV = {PV:.0f}, EV = {EV:.0f}, AC = {AC:.0f}: los tres valores del alumno son correctos.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# 1. **CV = AC − EV**: la variación de coste es CV = EV − AC. Con el signo cambiado un sobrecoste parece un ahorro.
# 2. **CPI = EV / PV**: eso es el SPI. El CPI compara valor ganado con coste real: CPI = EV / AC.
# 3. **SPI = EV / AC**: eso es el CPI. SPI = EV / PV.
# 4. **Interpretación cruzada**: CPI mide coste y SPI mide plazo; al intercambiarlos se intercambia también la lectura.
# 5. **EAC = BAC × CPI**: la previsión es EAC = BAC / CPI (si el rendimiento es malo, CPI < 1, el coste final es mayor, no menor).

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
CV, SV = EV - AC, EV - PV
CPI, SPI = EV / AC, EV / PV
EAC = BAC_total / CPI
VAC = BAC_total - EAC
pd.Series({'CV = EV − AC': CV, 'SV = EV − PV': SV, 'CPI = EV / AC': CPI, 'SPI = EV / PV': SPI, 'EAC = BAC / CPI': EAC, 'VAC = BAC − EAC': VAC}).round(3).to_frame('valor').T

# %%
Markdown(f'''CPI = {CPI:.2f} ({"sobrecoste" if CPI < 1 else "ahorro"}) y SPI = {SPI:.2f} ({"retraso" if SPI < 1 else "adelanto"}). Previsión de coste final {EAC:.0f} k€ frente a {BAC_total:.0f} presupuestados (VAC = {VAC:.0f}).''')
