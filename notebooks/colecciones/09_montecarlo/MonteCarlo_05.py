# ---
# title: MonteCarlo 05
# subtitle: "8 actividades con distribuciones mixtas (1 fijas)"
# coleccion: 09_montecarlo
# variante: mixtas
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones medias (semanas) se adjunta. Cada actividad sigue una distribución distinta, según la información disponible sobre ella: normal (media y desviación típica), triangular (mínimo, moda, máximo), uniforme (mínimo, máximo) o fija (duración cierta).

# %%
#| tbl-cap: "Cuadro de prelaciones y duraciones medias"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 7       ),
    ('B'     , '---'      , 6       ),
    ('C'     , '---'      , 4       ),
    ('D'     , 'B'        , 4       ),
    ('E'     , 'A,C'      , 5       ),
    ('F'     , 'B,E'      , 5       ),
    ('G'     , 'E'        , 7       ),
    ('H'     , 'D,F,G'    , 5       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %%
#| tbl-cap: "Distribución de la duración de cada actividad"
#| label: tbl-distribuciones
distribuciones = pd.DataFrame([
#  activity, distribucion, parametros           
    ('A'     , 'fija'      , '7'                  ),
    ('B'     , 'triangular', 'a = 5, m = 6, b = 7'),
    ('C'     , 'triangular', 'a = 3, m = 4, b = 5'),
    ('D'     , 'normal'    , 'μ = 4, σ = 0.5'     ),
    ('E'     , 'triangular', 'a = 4, m = 5, b = 6'),
    ('F'     , 'uniforme'  , 'a = 3, b = 7'       ),
    ('G'     , 'triangular', 'a = 6, m = 7, b = 8'),
    ('H'     , 'triangular', 'a = 3, m = 5, b = 7'),
], columns=['activity', 'distribucion', 'parametros']).set_index('activity')
distribuciones

# %%
pregunta_1 = "1. Calcule la media y la varianza de cada actividad según su distribución y compruebe que las medias coinciden con las duraciones medias del cuadro.  "
pregunta_2 = "2. Determine la ruta crítica con las medias y la probabilidad de terminar en la duración media más dos semanas según la aproximación normal.  "
pregunta_3 = "3. Simule el proyecto con 20 000 réplicas usando la distribución de cada actividad (las fijas no varían). Compare media, desviación típica y la probabilidad anterior.  "
pregunta_4 = "4. ¿Qué actividad tiene mayor índice de criticidad sin ser crítica en media? ¿Influye en ello que alguna actividad sea de duración fija?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
import re
def parametros(texto):
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', texto)]

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
filas = {}
for act, fila in distribuciones.iterrows():
    p = parametros(fila['parametros'])
    if fila['distribucion'] == 'normal':
        media, var = p[0], p[1] ** 2
    elif fila['distribucion'] == 'triangular':
        a, m, b = p; media, var = (a + m + b) / 3, (a*a + m*m + b*b - a*b - a*m - b*m) / 18
    elif fila['distribucion'] == 'uniforme':
        a, b = p; media, var = (a + b) / 2, (b - a) ** 2 / 12
    else:
        media, var = p[0], 0.0
    filas[act] = {'distribución': fila['distribucion'], 'media': media, 'varianza': var}
resumen = pd.DataFrame(filas).T
resumen['media del cuadro'] = datos_proyecto['duration']
datos_proyecto['variance'] = resumen['varianza'].astype(float)
resumen

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| fig-cap: "Grafo PERT con duraciones medias y camino crítico"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %%
duracion_media = proyecto.duration(datos_proyecto['duration'])
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['variance'])
plazo = duracion_media + 2
p_normal = float(norm.cdf((plazo - duracion_media) / desviacion_proyecto)) if desviacion_proyecto > 0 else 1.0
Markdown(f'Duración media {duracion_media}, σ = {desviacion_proyecto:.3f}; P(T ≤ {plazo}) = {p_normal:.3f} por la aproximación normal.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# `monte_carlo` acepta una distribución por actividad: `('normal', μ, σ)`, `('triangular', a, m, b)`, `('uniform', a, b)` o una función; una duración fija es una función que siempre devuelve el mismo valor.

# %%
dists = {}
for act, fila in distribuciones.iterrows():
    p = parametros(fila['parametros'])
    if fila['distribucion'] == 'normal':
        dists[act] = ('normal', p[0], p[1])
    elif fila['distribucion'] == 'triangular':
        dists[act] = ('triangular', p[0], p[1], p[2])
    elif fila['distribucion'] == 'uniforme':
        dists[act] = ('uniform', p[0], p[1])
    else:
        dists[act] = (lambda valor: (lambda size=None: np.full(size, valor) if size else valor))(p[0])
resultado = proyecto.monte_carlo(dists, n_samples=20000, seed=11)
p_mc = float((resultado.tau_samples <= plazo).mean())
display(resultado.summary().round(2))
pd.DataFrame({'aproximación normal': [duracion_media, desviacion_proyecto, p_normal], 'Monte Carlo': [resultado.tau_mean, resultado.tau_std, p_mc]},
             index=['media', 'desviación típica', f'P(T ≤ {plazo})']).round(3)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
calendario = proyecto.calendar(datos_proyecto['duration'])
crit = (resultado.sensitivity_mean * 100).round(1).rename('índice de criticidad (%)')
tabla = pd.concat([crit, calendario['H_total'].rename('holgura'), resumen['distribución'], resumen['varianza'].astype(float).round(2)], axis=1).sort_values('índice de criticidad (%)', ascending=False)
tabla

# %%
vigilar = tabla[tabla['holgura'] > 0]['índice de criticidad (%)'].idxmax()
Markdown(f'''La actividad no crítica con mayor índice de criticidad es {vigilar} ({tabla.loc[vigilar, "índice de criticidad (%)"]} %). Las actividades de duración fija nunca cambian de ruta por sí mismas: su índice depende sólo de lo que hagan las demás; una ruta con muchas actividades fijas tiene poca varianza y rara vez «adelanta» a la crítica.''')
