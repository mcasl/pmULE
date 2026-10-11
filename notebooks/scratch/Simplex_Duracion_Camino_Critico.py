# %% [markdown]
# ## Uso de Programación Lineal para el cálculo de la duración del proyecto y del camino crítico

# %%
# !apt install libgraphviz-dev              > /dev/null
# %pip install pygraphviz "pandas[excel]"   > /dev/null
# %pip install numpyarray_to_latex          > /dev/null
# %pip install -q amplpy                    > /dev/null

repo_url = 'https://github.com/mcasl/pmULE'
repo_name = "pmULE"

import os, sys

if not os.path.exists(repo_name):
    print(f"Cloning repository '{repo_name}'...")
    # !git clone {repo_url}
    print("Repository cloned successfully!")
else:
    print(f"Good! No need to download '{repo_name}' again.")

from google.colab import files

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph

# %%
from pmule import *
# Alargamos el ancho de las celdas a un tanto por ciento
from IPython.display import display, HTML, Markdown, Math, Latex
display(HTML("<style>.container { width:60% !important; }</style>"))

import pandas as pd
import numpy as np
import sympy as sym
import matplotlib
import seaborn as sns

datos_enunciado = pd.DataFrame(data=np.array([
       [ 8, 4, 2],
       [10, 5, 4],
       [12, 6, 3],
       [10, 6, 4],
       [ 7, 5, 5],
       [ 7, 4, 5],
       [14, 9, 3],
       [12, 8, 2],
       [10, 7, 1],
       ]),
   columns=['Dmax', 'Dtope', 'UCR'],
   index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
   dtype=np.int32)
datos_enunciado

# %%
proyecto = ProjectGraph.from_dict_of_strings({
    "A": "---",
    "B": "---",
    "C": "---",
    "D": "A",
    "E": "B",
    "F": "C,D,E",
    "G": "A",
    "H": "B",
    "I": "F,G",
})

proyecto.pert("pert_2024-11-simplex_CriticalPath_no_durations.svg", durations=False, ranksep=1)


# %% [markdown]
# ## Cálculo de los tiempos tempranos y tardíos por el método de Zaderenko

# %%
proyecto.zaderenko(durations=datos_enunciado["Dmax"])


# %%
proyecto.pert("pert_2024-11-simplex_CriticalPath_Dmax.png", durations=datos_enunciado["Dmax"], ranksep=1)

# %% [markdown]
# ## Cálculo de la duración del proyecto mediante Programación Lineal

# %% [markdown]
# $$\max z= 8x_A + 10x_B + 12x_C+10x_D+7x_E+7xF+14x_G+12x_H+10x_I$$
# sujeto a:
#
# \begin{eqnarray}
# 1               & = & x_A + x_B + x_C   \\
# x_B             & = & x_E + x_H \\
# x_A             & = & x_D + x_G \\
# x_C+x_E+x_D     & = & x_F \\
# x_F+x_G         & = & x_I \\
# x_H+x_I         & = & 1 \\
# \end{eqnarray}

# %% [markdown]
# De manera matricial:

# %% [markdown]
# $$\max z= \begin{bmatrix} 8	\\ 10	\\ 	12	\\ 	10	\\ 	7	\\ 	7	\\ 	14	\\ 	12	\\ 	10	\\
#   \end{bmatrix}^T \begin{bmatrix}x_A	\\ 	x_B	\\ 	x_C	\\ 	x_D	\\ 	x_E	\\ 	x_F	\\  x_G	\\ 	x_H	\\ 	x_I\end{bmatrix}$$
# sujeto a:
#
# $$\begin{bmatrix}
#  1  & 1  &  1 &  0 &  0 &  0 &  0 &  0 &  0 \\
#  1  & 0  &  0 & -1 &  0 &  0 & -1 &  0 &  0 \\
#  0  & 1  &  0 &  0 & -1 &  0 &  0 & -1 &  0 \\
#  0  & 0  &  1 &  1 &  1 & -1 &  0 &  0 &  0 \\
#  0  & 0  &  0 &  0 &  0 &  1 &  1 &  0 & -1 \\
#  0  & 0  &  0 &  0 &  0 &  0 &  0 &  1 &  1 \\
# \end{bmatrix}
# \begin{bmatrix}x_A	\\ 	x_B	\\ 	x_C	\\ 	x_D	\\ 	x_E	\\ 	x_F	\\  x_G	\\ 	x_H	\\ 	x_I\end{bmatrix}=\begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 1\end{bmatrix}$$
#

# %% [markdown]
# ### Representación utilizando AMPL

# %%
from amplpy import AMPL, ampl_notebook
ampl = ampl_notebook(
    modules=["highs"], # pick from over 20 modules including most commercial and open-source solvers
    license_uuid="default")
    #    license_uuid=userdata.get("licencia_ampl")) # your license UUID (e.g., free ampl.com/ce or ampl.com/courses licenses)

SOLVER = "highs"

# %%
# Sección del modelo
# %%ampl_eval

param m;
param n;
param t{1..n};
param b{1..m};
param a{1..m,1..n};

var x{1..n} >= 0;

maximize Duración: sum {j in 1..n} t[j] * x[j];
subject to restr {i in 1..m}:
    sum {j in 1..n}  a[i,j] * x[j] <= b[i];


# %%
# Sección de datos
# %%ampl_eval
data;
param n:=9;
param m:=6;
param t:= 1 8
          2 10
          3 12
          4 10
          5 7
          6 7
          7 14
          8 12
          9 10;

param b:= 1 1
          2 0
          3 0
          4 0
          5 0
          6 1;

param a:    1  2  3  4  5  6  7  8  9 :=
        1   1  1  1  0  0  0  0  0  0
        2   1  0  0 -1  0  0 -1  0  0
        3   0  1  0  0 -1  0  0 -1  0
        4   0  0  1  1  1 -1  0  0  0
        5   0  0  0  0  0  1  1  0 -1
        6   0  0  0  0  0  0  0  1  1 ;

# %%
ampl.option["solver"] = SOLVER
ampl.solve()

# %% [markdown]
#
# La duración del proyecto es:

# %%
ampl.get_value("Duración")

# %%
x = pd.Series(ampl.get_solution(zeros=True, flat=False)['x'], name='x')
x

# %% [markdown]
# La rama del camino crítico que ha encontrado (puede haber múltiples pero la solución será sólo una de ellas):

# %%
[activity for (activity, is_critical) in zip(datos_enunciado.index, x)
                                      if is_critical]
