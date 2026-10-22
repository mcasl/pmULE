# ---
# title: Ejercicio 16
# lang: es
# documentclass: scrreprt
# papersize: A4
# classoption: []
# execute:
#     echo: True
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
#| echo: False
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, plot_cargas

# %%
#| echo: false
#| output: false
# #%%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# # Enunciado
# ## Cálculo de la duración de un proyecto usando el Método de Montecarlo

# %% [markdown]
# Considere el proyecto cuya información se adjunta.

# %%
#| echo: False
pregunta_1 = f"1. Dibuje el grafo PERT del proyecto.  "
pregunta_2 = f"2. Determine las ecuaciones de los tiempos tempranos de cada nodo.  "
pregunta_3 = f"3. Calcule, utilizando las ecuaciones, los tiempos tempranos de cada nodo para 1000 iteraciones del método de MonteCarlo.  "
pregunta_4 = f"4. Determina la duración media y la desviación típica de la duración del proyecto.  "
pregunta_5 = f"5. Determine la duración para una probabilidad de completar antes el proyecto del 98%.  "

# %%
#| echo: False
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))
display(Markdown(pregunta_5))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import numpy as np
import matplotlib as plt
# %matplotlib inline

datos_enunciado = pd.DataFrame([
#  actividad, predecesora, duracion media, desviación típica
     ('A',    '---  ', 10, 1.5 ),
     ('B',    '---  ',  5, 0.4 ),
     ('C',    'A    ',  6, 0.8 ),
     ('D',    'A    ',  2, 0.7 ),
     ('E',    'B,C,D',  7, 1.8 ),
     ('F',    'B,D  ',  8, 0.8 ),
     ('G',    'E,F  ',  4, 1.4 ),
     ('H',    'B    ',  7, 1.6 ),
] ,columns=['activity', 'predecessor', 'mean', 'stdev']).set_index('activity')

datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
#| echo: False
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
datos_proyecto =  datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
proyecto.display_pert_construccion(tabla=prelaciones)

# %% [markdown]
# ## Apartado 2

# %%
#| echo: False
display(Markdown(pregunta_2))

# %% [markdown]
# $$\begin{aligned}
# t_1 &= 0\\
# t_2 &= t_1 + B\\
# t_3 &= t_1 + A\\
# t_4 &= \max \left[ t_2 + d_{@∇B⤑ΔF} \; , \;  t_3 + D \right]           \\
# t_5 &= \max \left[ t_3 + C          \; , \;   t_4 + d_{@∇D⤑ΔE} \right] \\
# t_6 &= \max \left[ t_5 + E          \; , \;   t_4 + F \right]          \\
# t_7 &= \max \left[ t_6+ G           \; , \;   t_2 + H \right]          \\
# \end{aligned}
# $$

# %% [markdown]
# ## Apartado 3

# %%
#| echo: False
display(Markdown(pregunta_3))

# %% [markdown]
# ### Generación aleatoria de las duraciones de las actividades

# %%
number_of_samples = 1000

# %%
activity_names = datos_proyecto.index
np.random.seed(2025)   # reproducibilidad de la simulación
duration = pd.DataFrame([np.random.normal(size = number_of_samples,
                                          loc  = datos_proyecto.loc[task, 'mean',],
                                          scale= datos_proyecto.loc[task, 'stdev'])
                         for task in activity_names
                        ],
                        index=activity_names).T

# %%
duration.head()


# %% [markdown]
# ### Cálculo de los tiempos tempranos

# %%
def row_max(A,B):
    return pd.concat([A,B], axis=1).max(axis=1)


# %%
t_early = duration.copy()
t_early.eval("""t_1 = 0
                t_2 = t_1 + B

                t_3 = t_1 + A

                input_D = t_3 + D
                input_f_B_F = t_2 + 0
                t_4 = @row_max(input_D, input_f_B_F)

                input_C = t_3 + C
                input_f_D_E = t_4 + 0
                t_5 = @row_max(input_C, input_f_D_E)

                input_E = t_5 + E
                input_F = t_4 + F
                t_6 = @row_max(input_E, input_F)

                input_G = t_6 + G
                input_H = t_2 + H
                t_7 = @row_max(input_G, input_H)
                """, inplace=True)

# %%
tiempos_nodos = ['t_'+ str(num) for num in range(1, 8)]
t_early.loc[0:10, : ]

# %% [markdown]
# ## Apartado 4

# %%
#| echo: False
display(Markdown(pregunta_4))

# %%
from pmule import display_histograma_tikz
display_histograma_tikz(t_early['t_7'])

# %%
t_early['t_7'].describe().round(2)

# %%
f"La duración media del proyecto es:{round(t_early['t_7'].mean(), 1)}, la desviación típica de la duración del proyecto es:{round(t_early['t_7'].std(), 1)}"

# %% [markdown]
# ## Apartado 5

# %%
#| echo: False
display(Markdown(pregunta_5))

# %%
resultado_1 = round(t_early['t_7'].sort_values(ascending=True).iloc[int(0.98*number_of_samples)],1)

# %%
Markdown(f"Podemos ordenar los valores en sentido ascendente y quedarnos con aquél valor superior al 98% de las muestras. De esta manera obtenemos un resultado igual a {resultado_1} periodos")

# %%
resultado_2 = round(t_early['t_7'].quantile(q=0.98, interpolation='higher'), 1)
from pmule import display_histograma_tikz
display_histograma_tikz(t_early['t_7'], percentil=0.98)

# %%
Markdown(f"O bien, utilizando la función `quantile`, obteniendo igualmente un valor de {resultado_2} periodos.")

# %%

# %%

# %%
