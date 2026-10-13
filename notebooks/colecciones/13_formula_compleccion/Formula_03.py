# ---
# title: Formula 03
# subtitle: "10 actividades, 6 rutas; SVD de R y ranking frente al conjunto crítico"
# coleccion: 13_formula_compleccion
# variante: svd-rango-1
# nivel: 3
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
# La matriz de rutas $R$ describe la estructura del proyecto sin usar duraciones. Su descomposición en valores singulares $R = U\,\Sigma\,V^T$ permite resumir esa estructura: $u_1$ pondera las rutas y $v_1$ las actividades, y el cociente $\sigma_1/\sigma_2$ indica cuánto domina esa primera dirección. Considere el proyecto cuyo cuadro se adjunta.

# %%
pregunta_1 = "1. Construya R y calcule su SVD. Dé los valores singulares y el cociente σ₁/σ₂.  "
pregunta_2 = "2. Ordene las actividades por |v₁| y las rutas por |u₁|. ¿Qué tienen en común las actividades mejor situadas?  "
pregunta_3 = "3. Con las duraciones, calcule τ exacto y la aproximación de rango 1, τ₁ = σ₁ · máx(u₁) · (v₁ᵀ t) (R ≈ σ₁ u₁ v₁ᵀ). ¿Qué error relativo comete?  "
pregunta_4 = "4. Compare el ranking por |v₁| con el conjunto de actividades críticas: ¿es crítica la actividad mejor clasificada? ¿Qué fracción de las k mejor clasificadas son críticas (k = número de críticas)? ¿Por qué la SVD, que no usa duraciones, puede acertar?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 5       ),
    ('B'     , '---'      , 6       ),
    ('C'     , 'A,B'      , 3       ),
    ('D'     , 'C'        , 7       ),
    ('E'     , 'A,B'      , 3       ),
    ('F'     , 'C'        , 8       ),
    ('G'     , 'D,E'      , 7       ),
    ('H'     , 'G'        , 3       ),
    ('I'     , 'F,H'      , 4       ),
    ('J'     , 'I'        , 7       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la organización de un congreso científico; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Definición del programa y fechas',
    'B': 'Reserva de la sede',
    'C': 'Convocatoria de comunicaciones',
    'D': 'Página web e inscripciones',
    'E': 'Contratación de ponentes invitados',
    'F': 'Revisión de los trabajos recibidos',
    'G': 'Elaboración del programa definitivo',
    'H': 'Contratación de catering y viajes',
    'I': 'Edición del libro de actas',
    'J': 'Preparación de salas y audiovisuales',
}, name='Descripción')
descripciones.to_frame()

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
#| tbl-cap: "Matriz de rutas R"
#| label: tbl-R
proyecto.display_path_matrix(dummies=False)

# %%
U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)
u1, v1 = U[:, 0], Vt[0, :]
if u1.sum() < 0:
    u1, v1 = -u1, -v1          # R es no negativa: u1 y v1 pueden tomarse no negativos
display(Math(r'\Sigma = ' + bmatrix(np.diag(S), dec=2)))
Markdown(f'σ₁ = {S[0]:.3f}, σ₂ = {S[1]:.3f}; σ₁/σ₂ = {S[0]/S[1]:.2f}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
peso_actividades = pd.Series(v1, index=R.columns).sort_values(ascending=False)
peso_rutas = pd.Series(u1, index=R.index).sort_values(ascending=False)
apariciones = R.sum()
display(pd.DataFrame({'v₁': peso_actividades.round(3), 'rutas en las que aparece': apariciones[peso_actividades.index]}).T)
pd.DataFrame({'u₁': peso_rutas.round(3), 'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in peso_rutas.index], 'nº actividades': R.loc[peso_rutas.index].sum(axis=1)})

# %% [markdown]
# Las actividades con mayor $v_1$ son las que aparecen en más rutas, sobre todo en las rutas largas y con muchas actividades compartidas (las que a su vez tienen mayor $u_1$): $v_1$ es una medida de centralidad estructural. Las actividades iniciales o finales únicas aparecen en todas las rutas y encabezan la lista.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
tau = float((R @ t).max())
tau_1 = float(S[0] * u1.max() * (v1 @ t.values))
Markdown(f'τ = {tau:.0f}; aproximación de rango 1: τ₁ = {S[0]:.3f} × {u1.max():.3f} × {v1 @ t.values:.3f} = {tau_1:.2f}; error relativo {abs(tau_1 - tau) / tau * 100:.1f} %.')

# %% [markdown]
# La aproximación sustituye cada fila de $R$ por un múltiplo de $v_1$: funciona bien cuando las rutas se parecen entre sí (σ₁ ≫ σ₂) y peor cuando el proyecto tiene varias ramas muy distintas. El τ exacto es siempre la norma infinito de $R\,t$; la SVD es un resumen.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
criticas = set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)
k = len(criticas)
top_k = list(peso_actividades.index[:k])
precision = len(set(top_k) & criticas) / k
pd.DataFrame({'ranking por v₁': peso_actividades.index, 'crítica': [a in criticas for a in peso_actividades.index]}).set_index('ranking por v₁').T

# %%
Markdown(f'''La actividad mejor clasificada, {peso_actividades.index[0]}, {"es" if peso_actividades.index[0] in criticas else "no es"} crítica. Entre las {k} primeras hay {len(set(top_k) & criticas)} críticas (precisión {precision:.2f}). La SVD sólo ve la estructura: una actividad por la que pasan muchas rutas tiene muchas ocasiones de estar en la más larga, sea cual sea el vector de duraciones; por eso acierta a menudo, pero no siempre, porque la ruta crítica concreta la fijan las duraciones.''')
