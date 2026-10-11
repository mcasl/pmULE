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
# Alargamos el ancho de las celdas a un tanto por ciento
from IPython.display import display, HTML, Markdown, Math, Latex, SVG
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

# %% [markdown]
# ## Cálculo del PERT con duraciones máximas

# %%
proyecto.pert("pert_2024-11-duraciones_maximas.svg", durations=datos_enunciado["Dmax"], ranksep=1)

# %% [markdown]
# ## Cálculo del PERT con duraciones mínimas
#

# %%
proyecto.pert("pert_2024-11-duraciones_minimas.svg", durations=datos_enunciado["Dtope"], ranksep=1)

# %% [markdown]
# # Reducción de la duración del proyecto con mínimo coste

# %% [markdown]
# ## Project Crashing utilizando el algoritmo de Ackoff-Sassieni

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_enunciado['Dmax'].to_dict(),
                min_durations=datos_enunciado['Dtope'].to_dict(),
                costs=datos_enunciado['UCR'].to_dict())
ackoff

# %% [markdown]
# ## Project Crashing utilizando Programación Lineal

# %% [markdown]
# $$\min z= \mathrm{c}^T \left( \mathrm{T} - \mathrm{d} \right)  $$
# sujeto a:
# \begin{eqnarray}
# 0 & = & y_A  \\
# 0 & = & y_B \\
# 0 & = &  y_C  \\
# 0 & \le & y_A + d_A  & \le  & y_D \\
# 0 & \le & y_B + d_B  & \le  & y_E \\
# 0 & \le & y_C + d_C  & \le  & y_F  \\
# 0 & \le & y_D + d_D  & \le  & y_F  \\
# 0 & \le & y_E + d_E  & \le  & y_F  \\
# 0 & \le & y_A + d_A  & \le  & y_G  \\
# 0 & \le & y_B + d_B  & \le  & y_H  \\
# 0 & \le & y_F + d_F  & \le  & y_I  \\
# 0 & \le & y_G + d_G  & \le  & y_I  \\
# 0 & \le & y_H + d_H  & \le  & y_{Project} \\
# 0 & \le & y_I + d_I  & \le  & y_{Project}\\
# t_i  & \le  & d_i & \le & T_i \\
# 0    & \le  & t_i \\
# \end{eqnarray}
#

# %% [markdown]
# Representación del problema de programación lineal usando AMPL:

# %%
from amplpy import AMPL, ampl_notebook
ampl = ampl_notebook(
    modules=["highs"], # pick from over 20 modules including most commercial and open-source solvers
    license_uuid="default")
#    license_uuid=userdata.get("licencia_ampl")) # Should a license be required
SOLVER = "highs"

# %%
# %%ampl_eval

# Definición del modelo
set ACTIVITIES;

param T {ACTIVITIES} >= 0; # Normal duration
param t {ACTIVITIES} >= 0; # Crash duration
param c {ACTIVITIES};      # Unitary cost of reduction
param y_Project >= 0;      # Project crash duration

var d {ACTIVITIES} >= 0;   # Activity duration
var y {ACTIVITIES} >= 0;   # Earliest starting time of an activity, i.e. the early time of the starting node

minimize Overcost: sum {i in ACTIVITIES} c[i]*(T[i] - d[i]);

subject to  Feasible_Durations {i in ACTIVITIES}: t[i] <= d[i] <= T[i];

subject to  A_is_starting_activity: y['A'] = 0;
subject to  B_is_starting_activity: y['B'] = 0;
subject to  C_is_starting_activity: y['C'] = 0;
subject to  D_after_A             : y['D'] >= y['A'] + d['A'];
subject to  E_after_B             : y['E'] >= y['B'] + d['B'];
subject to  F_after_C             : y['F'] >= y['C'] + d['C'];
subject to  F_after_D             : y['F'] >= y['D'] + d['D'];
subject to  F_after_E             : y['F'] >= y['E'] + d['E'];
subject to  G_after_A             : y['G'] >= y['A'] + d['A'];
subject to  H_after_B             : y['H'] >= y['B'] + d['B'];
subject to  I_after_F             : y['I'] >= y['F'] + d['F'];
subject to  I_after_G             : y['I'] >= y['G'] + d['G'];
subject to  Project_is_finished_H  : y['H'] + d['H'] <= y_Project;
subject to  Project_is_finished_I  : y['I'] + d['I'] <= y_Project;

# %%
# Carga de datos del problema
ampl.set['ACTIVITIES'] = datos_enunciado.index.values
ampl.param['T'] = datos_enunciado["Dmax"].values
ampl.param['t'] = datos_enunciado["Dtope"].values
ampl.param['c'] = datos_enunciado["UCR"].values
ampl.param['y_Project'] = 21


# %%
ampl.option["solver"] = SOLVER
ampl.solve()

# %%
solucion = pd.DataFrame(ampl.get_solution(flat=False, zeros=True))
solucion['reduccion'] = datos_enunciado['Dmax'] - solucion['d']

# %%
solucion

# %%
# Pert de duraciones mínimas calculado anteriormente, mostrado nuevamente para comparar
SVG("pert_2024-11-duraciones_minimas.svg")

# %%
# Proyecto con duraciones reducidas utilizando Ackoff-Sassieni
duracion_ackoff = ackoff.data.loc[14,:]['A':'I'] + datos_enunciado['Dtope']
proyecto.pert(filename="pert_ackoff.svg", durations=duracion_ackoff)

# %%
# Proyecto con duraciones reducidas utilizando Programación Lineal. Coincidente en este caso con el obtenido con Ackoff-Sassieni
proyecto.pert(filename="pert_simplex_crashing.svg", durations=solucion['d'])

# %%
