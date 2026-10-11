# ---
# title: Ejercicio 14
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
# %%capture
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, EarnedValue
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# ## Enunciado
# Considere la información del proyecto cuya información aparece en la tabla adjunta:  

# %%
pregunta_1 = "1.  Calcule los tres diagramas de Gantt correspondientes al análisis de valor ganado.  "
pregunta_2 = "2.  Calcule EV, PV, AC, CPI y SPI para el punto de control actual.  "
pregunta_3 = "3.  Calcule utilizando el método de MonteCarlo la duración media y la desviación típica del resto de proyecto que falta por completar. Suponga para ello que la duración aleatoria de las actividades ya empezadas es proporcional a su duración planificada original.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))

# %%
# Alargamos el ancho de las celdas al 100%
from IPython.display import display, HTML, Math, Latex, Markdown
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, desviación, coste_planificado, coste_real, duracion_real, realizado
     ('A',      '---    ',       2,          0.5,         60,              80,          2,           100,   ),
     ('B',      '---    ',       2,          0.2,         60,              60,          3,           100,   ),
     ('C',      '---    ',       3,          0.6,         60,              80,          4,           100,   ),
     ('D',      'A      ',       4,          0.1,         40,              60,          5,           100,   ),
     ('E',      'A      ',       1,          0.3,         30,              36,          3,           100,   ),
     ('F',      'E, B   ',       2,          0.7,         12,              12,          3,           100,   ),
     ('G',      'E, B   ',       1,          0.2,         39,              54,          3,           100,   ),
     ('H',      'G, C   ',       4,          0.3,         36,              72,          2,            50,   ),
     ('I',      'D, F   ',       1,          0.7,         90,              90,          2,            60,   ),
     ('J',      'D, F, H',       2,          0.3,         50,               0,          0,             0,   ),
     ('K',      'D, F, H',       2,          0.9,         40,               0,          0,             0,   ),
     ('L',      'I, J   ',       2,          0.3,         30,               0,          0,             0,   ),
     ('M',      'K      ',       2,          0.6,         30,               0,          0,             0,   ),
] ,columns=['actividad', 'precedentes', 'duracion', 'desviacion',
            'coste_planificado', 'coste_real', 'duracion_real', 'realizado']).set_index('actividad')
datos_enunciado

# %%
numero_de_muestras = 10

nombres_de_actividades = datos_enunciado.index
np.random.seed(2025)
duraciones_montecarlo = pd.DataFrame([np.random.normal(size = numero_de_muestras,
                                                       loc  = datos_enunciado.duracion[task],
                                                       scale= datos_enunciado.desviacion[task])
                                      for task in nombres_de_actividades],
                                     index=nombres_de_actividades).T.round(1)

# %%
duraciones_montecarlo

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ### Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='actividad', predecessor='precedentes')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# La clase PertGraph permite visualizar el grafo PERT, el diagrama de Gantt así como realizar el cálculo de tiempos tempranos, tardíos y de las holguras totales.

# %%
proyecto = prelaciones.create_project()
proyecto.pert()

# %% [markdown]
# #### Gantt Valor Planificado

# %%
vg = EarnedValue(proyecto)

# %%
gantts = vg.calcula_gantts(datos_proyecto,
                           planned_durations_label    = 'duracion',
                           actual_durations_label     = 'duracion_real',
                           PV_label                   = 'coste_planificado',
                           AC_label                   = 'coste_real',
                           percentage_complete_label  = 'realizado'
                          )

# %%
TexFragment(gantts['gantt_PV']).run_latex()

# %% [markdown]
# #### Gantt Valor Ganado

# %%
TexFragment(gantts['gantt_EV']).run_latex()

# %% [markdown]
# #### Gantt Coste Real

# %%
TexFragment(gantts['gantt_AC']).run_latex()

# %% [markdown]
# ### Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
acumulados = gantts['acumulados']
acumulados.plot();

# %%
acumulados.tail(1)

# %%
EV = acumulados.tail(1)['EV'].iloc[0]
AC = acumulados.tail(1)['AC'].iloc[0]
PV = acumulados.tail(1)['PV'].iloc[0]
CPI = EV/AC
SPI=EV/PV

# %%
Markdown(f"Se obtiene CPI=EV/AC={round(CPI,2)} y SPI=EV/PV={round(SPI,2)}")

# %% [markdown]
# ### Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# #### Fragmento de proyecto por realizar

# %%
prelaciones_restantes = PredecessorTable.from_dict_of_strings(
    { 'H': '',
      'I': '',
      'J': 'H',
      'K': 'H',
      'L': 'I, J',
      'M': 'K'
    }
    )


# %%
proyecto_restante = prelaciones_restantes.create_project()
proyecto_restante.pert(duraciones=False)

# %% [markdown]
# #### Cálculo de las duraciones del resto del proyecto

# %%
datos_proyecto_restante = datos_proyecto.copy().drop(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
restantes = datos_proyecto_restante.query('realizado != 100')
duraciones_montecarlo = duraciones_montecarlo.reindex(restantes.index, axis=1)
duraciones_montecarlo = duraciones_montecarlo.eval('H = H * 0.5')
duraciones_montecarlo = duraciones_montecarlo.eval('I = I * 0.4')
duraciones_montecarlo

# %% [markdown]
# #### Cálculo de la duración del proyecto

# %% [markdown]
# La matrix de rutas es

# %%
proyecto_restante.path_matrix(dummies=False)

# %% [markdown]
# Multiplicando la matriz de rutas por las duraciones de las actividades, obtenemos las duraciones de las rutas en cada iteración.

# %%
duraciones_caminos =  proyecto_restante.path_matrix(dummies=False) @ duraciones_montecarlo.T
duraciones_caminos

# %% [markdown]
# Considerando en cada iteración el máximo de las duraciones de las rutas obtenemos los valores de las duraciones del proyecto.

# %%
duraciones_proyecto = pd.DataFrame(duraciones_caminos.max(axis="rows"))
duraciones_proyecto.columns=['Duración']
duraciones_proyecto

# %% [markdown]
# Con estos valores podemos describir la distribución de la duración del proyecto a través de su media y desviación típica.

# %%
duraciones_proyecto.plot(kind='density');

# %%
Markdown(f"La duración media es {round(float(duraciones_proyecto.mean().iloc[0]), 2)}")

# %%
Markdown(f"La desviación típida es {round(float(duraciones_proyecto.std().iloc[0]), 2)}")

# %%

# %%

# %%
