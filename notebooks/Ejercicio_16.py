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

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, plot_cargas, explica_percentil, plural, lista_y

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
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
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
display(Markdown(
    f"**Respuesta:** el grafo PERT tiene {plural(proyecto.pert_graph.number_of_nodes(), 'nodo')} (sucesos, numerados de 1 a "
    f"{proyecto.pert_graph.number_of_nodes()}) y {plural(len(proyecto.dummies), 'actividad ficticia', 'actividades ficticias')} "
    f"({lista_y(proyecto.dummies)}), necesarias para representar las prelaciones sin duplicar actividades."))

# %% [markdown]
# ## Apartado 2

# %%
#| echo: False
display(Markdown(pregunta_2))

# %% [markdown]
# El tiempo temprano de un nodo es el máximo, sobre todas las flechas que llegan a él, del tiempo temprano del nodo de
# origen más la duración de la actividad: $t_j = \max_i \left( t_i + d_{ij} \right)$. Las actividades ficticias
# ($d_{@∇B⤑ΔF}$ y $d_{@∇D⤑ΔE}$) tienen duración 0. Recorriendo los nodos en orden:
#
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
_r = t_early.loc[0].round(2)
display(Markdown(
    "Cada fila de la tabla anterior es una iteración: se aplican las ecuaciones a sus duraciones. Por ejemplo, en la "
    "iteración 0 (duraciones redondeadas a dos decimales):\n\n"
    f"- t_1 = 0\n"
    f"- t_2 = t_1 + B = 0 + {_r.B} = {_r.t_2}\n"
    f"- t_3 = t_1 + A = 0 + {_r.A} = {_r.t_3}\n"
    f"- t_4 = máx(t_2 + 0, t_3 + D) = máx({_r.t_2}, {_r.t_3} + {_r.D}) = máx({_r.input_f_B_F}, {_r.input_D}) = {_r.t_4}\n"
    f"- t_5 = máx(t_3 + C, t_4 + 0) = máx({_r.t_3} + {_r.C}, {_r.t_4}) = máx({_r.input_C}, {_r.input_f_D_E}) = {_r.t_5}\n"
    f"- t_6 = máx(t_5 + E, t_4 + F) = máx({_r.t_5} + {_r.E}, {_r.t_4} + {_r.F}) = máx({_r.input_E}, {_r.input_F}) = {_r.t_6}\n"
    f"- t_7 = máx(t_6 + G, t_2 + H) = máx({_r.t_6} + {_r.G}, {_r.t_2} + {_r.H}) = máx({_r.input_G}, {_r.input_H}) = **{_r.t_7}**\n\n"
    "t_7 (nodo final) es la duración del proyecto en esa iteración. Primeras iteraciones (duraciones y tiempos tempranos):"))

# %%
tiempos_nodos = ['t_'+ str(num) for num in range(1, 8)]
display(t_early.loc[0:10, list(activity_names) + tiempos_nodos].round(2))

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
Markdown(
    f"Con las {number_of_samples} duraciones simuladas del proyecto (columna t_7): media $\\bar x = \\sum x_i / n$ = "
    f"{t_early['t_7'].sum():.2f} / {number_of_samples} = {t_early['t_7'].mean():.2f}; desviación típica muestral "
    f"$s = \\sqrt{{\\sum (x_i - \\bar x)^2 / (n-1)}}$ = √({((t_early['t_7'] - t_early['t_7'].mean()) ** 2).sum():.2f} / "
    f"{number_of_samples - 1}) = {t_early['t_7'].std():.2f}.\n\n"
    f"**Respuesta:** la duración media del proyecto es **{round(t_early['t_7'].mean(), 1)}** periodos y su desviación típica, "
    f"**{round(t_early['t_7'].std(), 1)}** periodos.")

# %% [markdown]
# ## Apartado 5

# %%
#| echo: False
display(Markdown(pregunta_5))

# %%
resultado_1 = round(t_early['t_7'].sort_values(ascending=True).iloc[int(0.98*number_of_samples)],1)

# %%
Markdown(f"Podemos ordenar los valores en sentido ascendente y quedarnos con aquél valor superior al 98% de las muestras. De esta manera obtenemos un resultado igual a {resultado_1} periodos"
         f" (con las {number_of_samples} duraciones ordenadas de menor a mayor, se toma la que ocupa la posición "
         f"0.98 × {number_of_samples} + 1 = {int(0.98 * number_of_samples) + 1}: por debajo de ella quedan "
         f"{int(0.98 * number_of_samples)} iteraciones, el 98 %).")

# %%
resultado_2 = round(t_early['t_7'].quantile(q=0.98, interpolation='higher'), 1)
from pmule import display_histograma_tikz
display_histograma_tikz(t_early['t_7'], percentil=0.98)

# %%
Markdown(f"O bien, utilizando la función `quantile`, obteniendo igualmente un valor de {resultado_2} periodos.\n\n"
         f"Como comprobación, si se aproxima la duración por una normal con la media y la desviación típica del apartado 4: "
         + explica_percentil(t_early['t_7'].mean().round(2), t_early['t_7'].std().round(2), 0.98, 'periodos')
         + f", muy parecido al resultado de la simulación.\n\n"
         f"**Respuesta:** el proyecto se completa antes de **{resultado_2}** periodos con una probabilidad del 98 %.")

# %%

# %%

# %%
