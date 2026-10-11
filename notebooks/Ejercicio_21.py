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

from pmule import PredecessorTable, ProjectGraph, EarnedValue

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

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
pd.set_option('future.no_silent_downcasting', True)

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

# %%
os.chdir("/Users/mcasl/temp")
TexFragment(gantts['gantt_PV']).run_latex()

# %% [markdown]
# #### Gantt Valor Ganado

# %%
TexFragment(gantts['gantt_EV']).run_latex()

# %% [markdown]
# #### Gantt Coste Real

# %%
TexFragment(gantts['gantt_AC']).run_latex()

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
Markdown(f"""Utilizando: \\
PV={PV} \\
AC={AC} \\
EV={EV} \\
y el BAC={BAC} obtenido sumando los costes planificados, obtenemos:
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
Math((latex_code))

# %% [markdown]
# La diferencia de rendimiento en costes indica que el LRE proporcionado no es verosimil.

# %%

# %%

# %%
