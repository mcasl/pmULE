# ---
# title: Formula 01
# subtitle: "7 actividades, 4 rutas; τ como norma infinito"
# coleccion: 13_formula_compleccion
# variante: tau-norma-infinito
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


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# La duración de un proyecto puede escribirse como una fórmula cerrada: si $R$ es la matriz de rutas (una fila por cada ruta del nodo inicial al final, una columna por actividad, con un 1 donde la actividad pertenece a la ruta) y $t$ el vector de duraciones, entonces
#
# $$\tau = \lVert R\,t \rVert_\infty = \max_r \sum_i R_{ri}\,t_i .$$
#
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (días) se adjunta.

# %%
pregunta_1 = "1. Dibuje el grafo PERT, enumere sus rutas y construya la matriz de rutas R. ¿Qué dimensiones tiene?  "
pregunta_2 = "2. Calcule el vector R·t y su norma infinito. Compruebe que coincide con el tiempo más temprano del último nodo obtenido por el método de Zaderenko.  "
pregunta_3 = "3. ¿Por qué la norma infinito y no otra? ¿Qué daría ‖R·t‖₁? ¿Y qué significaría el mínimo de R·t?  "
pregunta_4 = "4. Escriba τ en función de t_E (la duración de E) dejando fijas las demás, y dibuje esa función. ¿Es lineal? ¿Dónde cambia de pendiente?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 8       ),
    ('B'     , '---'      , 6       ),
    ('C'     , 'A,B'      , 2       ),
    ('D'     , 'C'        , 5       ),
    ('E'     , 'C'        , 2       ),
    ('F'     , 'E'        , 4       ),
    ('G'     , 'D,F'      , 4       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
def bmatrix(a, dec=0):
    """Matriz LaTeX (bmatrix) a partir de un DataFrame o array."""
    a = np.asarray(getattr(a, 'values', a), dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    filas = [' & '.join(f'{x:.{dec}f}' for x in fila) + r'\\' for fila in a]
    return r'\begin{bmatrix}' + '\n' + '\n'.join(filas) + '\n' + r'\end{bmatrix}'

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
t = datos_proyecto['duration'][R.columns]

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT"
#| label: fig-pert
proyecto.pert(durations=False, ranksep=1)

# %%
#| tbl-cap: "Matriz de rutas R"
#| label: tbl-R
rutas = proyecto.paths(dummies=False)
display(pd.Series({r: ' - '.join(a) for r, a in rutas.items()}).to_frame('actividades'))
proyecto.display_path_matrix(dummies=False)

# %%
Markdown(f'R tiene {R.shape[0]} filas (rutas) y {R.shape[1]} columnas (actividades).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
Rt = R @ t
tau = float(Rt.max())
Math(bmatrix(R) + r' \times ' + bmatrix(t) + ' = ' + bmatrix(Rt) + r' \quad\Rightarrow\quad \tau = ' + f'{tau:.0f}')

# %%
#| tbl-cap: "Matriz de Zaderenko"
#| label: tbl-zaderenko
proyecto.zaderenko(durations=datos_proyecto['duration'])

# %%
Markdown(f'‖R·t‖∞ = {tau:.0f}; tiempo más temprano del último nodo = {proyecto.duration(datos_proyecto["duration"])}; '
         f'la librería (`proyecto.tau`) da {proyecto.tau(t.to_dict()):.0f}. Las tres coinciden: el cálculo de tiempos hacia delante es una forma de calcular el máximo sin enumerar las rutas.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''‖R·t‖₁ = {float(Rt.abs().sum()):.0f} sumaría las duraciones de todas las rutas, contando varias veces cada actividad compartida: no tiene sentido como plazo. El mínimo de R·t, {float(Rt.min()):.0f}, es la ruta más corta: el tiempo que *como mínimo* ocupa cualquier cadena del proyecto, pero el proyecto no termina hasta que acaba la última ruta. El plazo es el máximo porque todas las rutas deben completarse; de ahí la norma infinito.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
k = 'E'
valores = np.arange(0, int(t[k]) + 8)
taus = []
for v in valores:
    t2 = t.copy(); t2[k] = v
    taus.append(float((R @ t2).max()))
pendiente = np.diff(taus)
codo = int(valores[np.argmax(pendiente > 0)]) if (pendiente > 0).any() else None
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(valores, taus, marker='o'); ax.axvline(int(t[k]), color='grey', linestyle='--')
ax.set_xlabel(f'duración de {k}'); ax.set_ylabel('τ'); plt.show()

# %%
Markdown(f'''τ es el máximo de varias funciones lineales de t_{k} (una por ruta): las rutas que no contienen a {k} son constantes y las que la contienen tienen pendiente 1. El resultado es lineal a trozos y convexo: pendiente 0 mientras otra ruta manda y pendiente 1 a partir de t_{k} = {codo} (cuando la ruta más larga que pasa por {k} alcanza a la crítica). La duración actual de {k} es {int(t[k])}: su sensibilidad es {int(pendiente[list(valores).index(int(t[k]))])}.''')
