# ---
# title: ValorGanado 04
# subtitle: "9 actividades; punto de control en la semana 11: coste bien plazo mal"
# coleccion: 12_valor_ganado
# variante: coste-bien-plazo-mal
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


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por actividad (BAC, k€) se adjunta; el coste planificado de cada actividad se reparte uniformemente en su duración. Al final de la semana 11 se realiza un punto de control.

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 6       , 50 ),
    ('B'     , '---'      , 1       , 30 ),
    ('C'     , '---'      , 6       , 50 ),
    ('D'     , 'A,B,C'    , 4       , 30 ),
    ('E'     , 'D'        , 3       , 10 ),
    ('F'     , 'C'        , 4       , 10 ),
    ('G'     , 'C'        , 7       , 60 ),
    ('H'     , 'E,G'      , 8       , 10 ),
    ('I'     , 'F,H'      , 1       , 60 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la apertura de una clínica veterinaria; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de mercado y local',
    'B': 'Proyecto de reforma y licencia',
    'C': 'Obra de reforma',
    'D': 'Instalación de quirófano y radiología',
    'E': 'Mobiliario y equipamiento clínico',
    'F': 'Software de gestión',
    'G': 'Selección de personal',
    'H': 'Acuerdos con proveedores',
    'I': 'Registro sanitario',
}, name='Descripción')
descripciones.to_frame()

# %%
#| tbl-cap: "Datos del punto de control (final de la semana 11)"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 9              , 45, 100      ),
    ('B'     , 1              , 30, 100      ),
    ('C'     , 9              , 50, 100      ),
    ('D'     , 2              , 7 , 25       ),
    ('E'     , 0              , 0 , 0        ),
    ('F'     , 2              , 3 , 33       ),
    ('G'     , 2              , 8 , 14       ),
    ('H'     , 0              , 0 , 0        ),
    ('I'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %%
pregunta_1 = "1. Calcule PV, EV y AC en el punto de control, y los índices CPI y SPI.  "
pregunta_2 = "2. El director informa al comité de que «el proyecto va bien: hemos gastado menos de lo previsto». ¿Es correcta la lectura? ¿Qué parte de la situación oculta?  "
pregunta_3 = "3. ¿Qué actividades explican el retraso? Compare, actividad por actividad, el valor ganado con el valor planificado.  "
pregunta_4 = "4. Estime el coste final (EAC = BAC/CPI) y la duración final con SPI. ¿Qué previsión se fiaría más y cuál menos?  "

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
semana_control = 11
datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)
gantts = EarnedValue(proyecto).calcula_gantts(datos_valor_ganado, planned_durations_label='duration',
                                             actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',
                                             percentage_complete_label='performed', tikz=True)

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
Markdown(f'''Gastar menos de lo previsto (AC = {AC:.0f} < PV = {PV:.0f}) no dice nada por sí solo: se ha gastado menos porque se ha hecho menos. La comparación correcta es entre lo gastado y lo *ganado*: CPI = {CPI:.2f} {">" if CPI > 1 else "<"} 1, así que el coste sí va bien, pero SPI = {SPI:.2f} < 1: el proyecto va retrasado, se ha ganado sólo el {EV/PV*100:.0f} % del valor previsto. La frase del director confunde la variación de gasto (AC − PV) con la de coste (EV − AC).''')

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
retrasadas = por_actividad[por_actividad['SV'] < -0.5].sort_values('SV')
Markdown(f'Las actividades con SV negativa explican el retraso: {", ".join(retrasadas.index)}. '
         f'Las de mayor peso son {", ".join(retrasadas.index[:2])} (SV = {retrasadas["SV"].iloc[0]:.1f} y {retrasadas["SV"].iloc[1]:.1f} k€).' if len(retrasadas) > 1 else 'Una sola actividad explica el retraso.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
EAC = BAC_total / CPI
IEAC = duracion_planificada / SPI
Markdown(f'''EAC = {BAC_total:.0f} / {CPI:.2f} = {EAC:.0f} k€ (ahorro de {BAC_total - EAC:.0f}); duración prevista {duracion_planificada} / {SPI:.2f} = {IEAC:.1f} semanas. La previsión de coste es la más fiable (el CPI suele ser estable); la de plazo con SPI es la menos: el SPI tiende a 1 al final aunque el retraso persista, y además el retraso real depende de si las actividades retrasadas son críticas, cosa que el SPI (en euros) no distingue. Para el plazo conviene la programación ganada o replanificar lo que falta.''')
