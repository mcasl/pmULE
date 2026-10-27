# ---
# title: Ejercicio 21
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
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math
import handcalcs.render
from handcalcs import handcalc

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, EarnedValue, pinta_tikz, lista_y, limpia_tabla

# %% [markdown]
# ## Enunciado
#
# Supongamos un proyecto para pintar la fachada de un edificio de planta cuadrada. Cada fachada se estima que tarde dos días en pintarse y conlleve un gasto de 2000 euros. Las fachadas se pintan consecutivamente. Hoy es el final del día 6 y el trabajo se encuentra de la siguiente manera:  
#     - Fachada 1: Acabado en el día 2, 2000 euros.  
#     - Fachada 2: Acabado en el día 5, 2400 euros.  
#     - Fachada 3: 50% acabado, 1200 euros.  
#     - Fachada 4: sin empezar.  
# El coste final estimado por el contratista, LRE, es de 8200 euros.  
# Se pide realizar un analisis del valor ganado del proyecto a fecha actual (fin del día 6).  
# Se admite un error de precisión de 0.02.  
# De las fórmulas estudiadas para el coste estimado final por el promotor utilice: EAC=BAC/CPI.  

# %%
# Alargamos el ancho de las celdas al 100%
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas

import numpy as np

datos_enunciado = pd.DataFrame([
#     actividad,   precedentes,  duracion,  coste_planificado,     coste_real,  duracion_real,  realizado
     ('Fachada·1', '---      ',       2,            2000,              2000,          2,       100,   ),
     ('Fachada·2', 'Fachada·1',       2,            2000,              2400,          3,       100,   ),
     ('Fachada·3', 'Fachada·2',       2,            2000,              1200,          1,        50,   ),
     ('Fachada·4', 'Fachada·3',       2,            2000,                 0,          0,           0,   ),
] ,columns=['actividad', 'precedentes', 'duracion',  'coste_planificado', 'coste_real', 'duracion_real',
            'realizado']).set_index('actividad')
datos_enunciado

# %% [markdown]
# ### Resolución
#

# %%
datos_proyecto=datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='actividad', predecessor='precedentes')
prelaciones.display_immediate_linkage_matrix()

# %%
proyecto = prelaciones.create_project()
proyecto.pert()

# %% [markdown]
# #### Gantt Valor Planificado

# %%
vg = EarnedValue(proyecto)

# %%
datos_proyecto['duracion'] = datos_proyecto['duracion'].astype(float)
gantts = vg.calcula_gantts(data=datos_proyecto,
                           planned_durations_label = 'duracion',
                           actual_durations_label='duracion_real',
                           PV_label='coste_planificado',
                           AC_label='coste_real',
                           percentage_complete_label='realizado',
                           tikz=True,
                           params=dict(
                               period_width=2,
                           )
                          )
_cal_plan = proyecto.calendar(datos_proyecto['duracion'])
_cal_real = proyecto.calendar(datos_proyecto['duracion_real'])
_emp = datos_proyecto['duracion_real'] > 0
display(Markdown(
    "Los diagramas reparten un importe de cada fachada uniformemente a lo largo de su duración: el coste planificado "
    "sobre la duración planificada (Gantt PV, *Planned Value*), el valor ganado EV = % realizado × coste planificado "
    "sobre la duración real (Gantt EV, *Earned Value*) y el coste real sobre la duración real (Gantt AC, *Actual Cost*). "
    "El calendario real se deduce del enunciado (cada fachada empieza al acabar la anterior):"))
display(limpia_tabla(pd.DataFrame({
    'días plan.': [f"{int(_cal_plan.loc[a, 'inicio_mas_temprano']) + 1}–{int(_cal_plan.loc[a, 'fin_mas_temprano'])}" for a in datos_proyecto.index],
    'PV/día': datos_proyecto['coste_planificado'] / datos_proyecto['duracion'],
    'días reales': [("—" if not _emp[a] else f"{int(_cal_real.loc[a, 'fin_mas_temprano'])}"
                     if int(_cal_real.loc[a, 'inicio_mas_temprano']) + 1 == int(_cal_real.loc[a, 'fin_mas_temprano'])
                     else f"{int(_cal_real.loc[a, 'inicio_mas_temprano']) + 1}–{int(_cal_real.loc[a, 'fin_mas_temprano'])}")
                    for a in datos_proyecto.index],
    'EV = % × coste plan.': datos_proyecto['realizado'] / 100 * datos_proyecto['coste_planificado'],
    'EV/día': (datos_proyecto['realizado'] / 100 * datos_proyecto['coste_planificado'] / datos_proyecto['duracion_real']).where(_emp),
    'AC': datos_proyecto['coste_real'],
    'AC/día': (datos_proyecto['coste_real'] / datos_proyecto['duracion_real']).where(_emp),
}, index=datos_proyecto.index)))

# %%
pinta_tikz(gantts['gantt_PV'])

# %% [markdown]
# #### Gantt Valor Ganado

# %%
pinta_tikz(gantts['gantt_EV'])

# %% [markdown]
# #### Gantt Coste Real

# %%
pinta_tikz(gantts['gantt_AC'])

# %%
acumulados = gantts['acumulados']
acumulados.plot();

# %%
acumulados.tail(1)

# %% [markdown]
# #### Cálculo de parámetros diversos

# %%
EV = acumulados.tail(1)['EV'].iloc[0]
AC = acumulados.tail(1)['AC'].iloc[0]
PV = acumulados.tail(1)['PV'].iloc[0]

# %%
BAC = int(datos_enunciado.loc[:, 'coste_planificado'].sum())

# %%
_t = int(acumulados.index[-1])
_dp = datos_proyecto
_pv_i = [min(max(_t - _cal_plan.loc[a, 'inicio_mas_temprano'], 0), _dp.loc[a, 'duracion']) / _dp.loc[a, 'duracion'] * _dp.loc[a, 'coste_planificado']
         for a in _dp.index]
Markdown(f"""Valores a fin del día {_t}:

- PV (valor planificado) = coste planificado de lo que debía estar hecho el día {_t} = {' + '.join(format(v, 'g') for v in _pv_i)} = **{PV:g}** €
- EV (valor ganado) = Σ % realizado × coste planificado = {' + '.join(f"{r:g} % × {c:g}" for r, c in zip(_dp['realizado'], _dp['coste_planificado']))} = **{EV:g}** €
- AC (coste real) = {' + '.join(format(v, 'g') for v in _dp['coste_real'])} = **{AC:g}** €
- BAC (presupuesto total, *Budget At Completion*) = Σ costes planificados = {' + '.join(format(v, 'g') for v in _dp['coste_planificado'])} = **{BAC}** €

Con ellos se obtienen los índices (CPI y SPI: índices de rendimiento del coste y del plazo; EAC: coste estimado final; CV y SV:
variaciones del coste y del plazo; VAC: variación a la conclusión; PC, PS y PP: porcentajes completado, gastado y
planificado sobre el BAC; TCPI: índice de rendimiento del coste necesario en el trabajo restante para acabar con el BAC, el EAC o el LRE):
""")

# %%

# %%
from handcalcs.decorator import handcalc

@handcalc(jupyter_display=False) 
def my_calculation(EV,AC,PV,BAC):
    CPI = EV/AC
    SPI = EV/PV
    EAC = BAC / CPI
    CV = EV - AC
    CV_relativo = CV/EV *100
    SV = EV - PV
    SV_relativo = SV/PV *100
    VAC = BAC - EAC
    PC = EV / BAC
    PS = AC / BAC
    PP = PV / BAC
    TCPI_BAC = (BAC - EV)/(BAC - AC)
    TCPI_EAC = (BAC - EV)/(EAC - AC)
    LRE = 8200
    TCPI_LRE = (BAC - EV)/(LRE - AC)
latex_code, _ = my_calculation(EV,AC,PV,BAC)
display(Math((latex_code)))
_r = dict(CPI=EV / AC, SPI=EV / PV, CV=EV - AC, SV=EV - PV, PC=EV / BAC, PS=AC / BAC, PP=PV / BAC, LRE=8200)
_r.update(EAC=BAC / _r['CPI'], CV_relativo=_r['CV'] / EV * 100)
_r.update(VAC=BAC - _r['EAC'], TCPI_EAC=(BAC - EV) / (_r['EAC'] - AC), TCPI_LRE=(BAC - EV) / (_r['LRE'] - AC))
display(Markdown(
    f"**Interpretación:**\n\n"
    f"- CPI = {_r['CPI']:.3f} < 1 y CV = {_r['CV']:g} €: **sobrecoste**; se ha gastado un {-_r['CV_relativo']:.0f} % más de lo que vale lo hecho.\n"
    f"- SPI = {_r['SPI']:.3f} < 1 y SV = {_r['SV']:g} €: **retraso**; se ha hecho el {_r['SPI']:.1%} de lo planificado "
    f"(PC = {_r['PC']:.1%} del proyecto completado frente al PP = {_r['PP']:.0%} previsto, con un PS = {_r['PS']:.0%} del presupuesto ya gastado).\n"
    f"- Si se mantiene el CPI actual, el coste final será EAC = {_r['EAC']:g} €, {-_r['VAC']:g} € por encima del BAC (VAC = {_r['VAC']:g} €).\n"
    f"- Para acabar con el LRE de {_r['LRE']} € que da el contratista, el trabajo restante debería ejecutarse con un rendimiento "
    f"TCPI_LRE = {_r['TCPI_LRE']:.3f}, es decir, un {(_r['TCPI_LRE'] / _r['CPI'] - 1):.0%} más eficiente que hasta ahora "
    f"(CPI = {_r['CPI']:.3f})"
    + (", cuando hasta ahora ninguna fachada ha costado menos de lo que vale lo realizado" if all(_dp['coste_real'] >= _dp['realizado'] / 100 * _dp['coste_planificado']) else "")
    + f". Con el CPI actual el coste final "
    f"sería {_r['EAC']:g} € (TCPI_EAC = CPI = {_r['TCPI_EAC']:.3f})."))

# %% [markdown]
# **Conclusión:** el proyecto va retrasado y con sobrecoste, y la diferencia entre el rendimiento en costes conseguido
# (CPI) y el que exigiría el LRE (TCPI del LRE) indica que el LRE proporcionado por el contratista no es verosímil:
# el coste final esperable es el EAC calculado.

# %%

# %%

# %%
