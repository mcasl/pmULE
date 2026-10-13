# ---
# title: MonteCarlo 03
# subtitle: "9 actividades, 6 rutas, distribución Beta-PERT"
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

# Versión local de la librería (notebooks/colecciones/<coleccion>/ -> ../../../src)
sys.path.append(os.path.abspath('../../../src'))
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
# Considere el proyecto cuyo cuadro de prelaciones y estimaciones optimista, más probable y pesimista (en semanas) se adjunta. Suponga que la duración de cada actividad sigue una distribución Beta-PERT de parámetros $(a, m, b)$.

# %%
pregunta_1 = "1. Dibuje el grafo PERT, construya la matriz de caminos y calcule la duración del proyecto con las duraciones medias.  "
pregunta_2 = "2. Genere 10000 réplicas de las duraciones de las actividades y, con la matriz de caminos, obtenga la duración del proyecto en cada una. Calcule su media, desviación típica y percentiles 5, 50 y 95.  "
pregunta_3 = "3. Calcule la probabilidad de terminar en 23 semanas o menos y la duración que puede comprometerse con un 90% de probabilidad. Compare con la aproximación normal por el camino crítico y explique las diferencias.  "
pregunta_4 = "4. Calcule el índice de criticidad de cada ruta y de cada actividad. ¿Qué actividad no crítica convendría vigilar?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b 
    ('A'     , '---'      , 5, 8, 11),
    ('B'     , '---'      , 3, 4, 5 ),
    ('C'     , '---'      , 5, 6, 7 ),
    ('D'     , 'B'        , 3, 4, 5 ),
    ('E'     , 'C'        , 1, 3, 5 ),
    ('F'     , 'D,E'      , 1, 6, 11),
    ('G'     , 'D'        , 3, 6, 9 ),
    ('H'     , 'D,E'      , 5, 7, 9 ),
    ('I'     , 'A,F,G,H'  , 4, 6, 8 ),
], columns=['activity', 'predecessor', 'a', 'm', 'b']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %% [markdown]
# Con el método PERT la media de cada actividad es $(a + 4m + b)/6$ y su desviación típica se aproxima por $(b - a)/6$.

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = datos_proyecto.eval('(a + 4 * m + b) / 6').round().astype(int)
datos_proyecto['standard_deviation'] = datos_proyecto.eval('(b - a) / 6')
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
datos_proyecto

# %%
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT con las duraciones medias"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %%
#| tbl-cap: "Matriz de caminos"
#| label: tbl-matrizCaminos
path_matrix = proyecto.path_matrix(dummies=False)
proyecto.display_path_matrix(dummies=False)

# %%
duraciones = datos_proyecto['duration'][path_matrix.columns]
duraciones_rutas_medias = path_matrix @ duraciones
duracion_media = int(duraciones_rutas_medias.max())
ruta_critica = duraciones_rutas_medias.idxmax()
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])
pd.concat([duraciones_rutas_medias.rename('duración media'),
           (path_matrix @ datos_proyecto['variance'][path_matrix.columns]).rename('varianza')], axis=1)

# %%
Markdown(f'Con las duraciones medias el proyecto dura {duracion_media} semanas; la ruta crítica es {ruta_critica} y su desviación típica {desviacion_proyecto:.3f}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# La Beta-PERT de parámetros $(a, m, b)$ es $X = a + (b - a)\,Y$ con $Y \sim \mathrm{Beta}(\alpha, \beta)$, $\alpha = 1 + 4\frac{m-a}{b-a}$ y $\beta = 1 + 4\frac{b-m}{b-a}$; su media es exactamente $(a+4m+b)/6$.

# %%
N = 10000
rng = np.random.default_rng(557)
def muestras_beta_pert(a, m, b, size):
    if b == a:
        return np.full(size, float(a))
    alfa = 1 + 4 * (m - a) / (b - a)
    beta = 1 + 4 * (b - m) / (b - a)
    return a + (b - a) * rng.beta(alfa, beta, size)
duraciones_mc = pd.DataFrame({act: muestras_beta_pert(fila['a'], fila['m'], fila['b'], N)
                              for act, fila in datos_proyecto.iterrows()})
duraciones_mc = duraciones_mc[path_matrix.columns]

# %%
#| tbl-cap: "Medias y desviaciones típicas muestrales frente a las teóricas"
#| label: tbl-comprobacionMuestras
pd.DataFrame({'media teórica': datos_proyecto['duration'], 'media muestral': duraciones_mc.mean(),
              'σ teórica': datos_proyecto['standard_deviation'], 'σ muestral': duraciones_mc.std()}).round(2).T

# %% [markdown]
# Las medias muestrales coinciden con las teóricas. Las desviaciones típicas salen algo mayores que $(b-a)/6$: esa fórmula es una aproximación; la desviación exacta de la Beta-PERT simétrica es $(b-a)/\sqrt{28} \approx (b-a)/5.3$.

# %% [markdown]
# La duración de cada ruta en cada réplica es el producto de las duraciones por la traspuesta de la matriz de caminos; la del proyecto, el máximo por filas.

# %%
duraciones_rutas_mc = duraciones_mc @ path_matrix.T
duracion_proyecto_mc = duraciones_rutas_mc.max(axis='columns')
pd.concat([duraciones_rutas_mc, duracion_proyecto_mc.rename('Proyecto')], axis=1).head().round(2)

# %%
#| tbl-cap: "Descriptores de la duración simulada del proyecto"
#| label: tbl-descriptores
duracion_proyecto_mc.describe(percentiles=[0.05, 0.5, 0.95]).round(2).to_frame('Duración del proyecto').T

# %%
#| fig-cap: "Distribución simulada de la duración frente a la aproximación normal"
#| label: fig-histograma
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(duracion_proyecto_mc, bins=60, density=True, alpha=0.5, color='skyblue', label='Monte Carlo (Beta-PERT)')
x = np.linspace(duracion_proyecto_mc.min(), duracion_proyecto_mc.max(), 300)
ax.plot(x, norm.pdf(x, duracion_media, desviacion_proyecto), color='navy', label='Aproximación normal (camino crítico)')
ax.axvline(23, color='red', linestyle='--', label='Plazo (23)')
ax.set_xlabel('Duración del proyecto (semanas)'); ax.legend(); plt.show()

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
plazo, probabilidad_objetivo = 23, 0.9
p_mc = float((duracion_proyecto_mc <= plazo).mean())
t_mc = float(duracion_proyecto_mc.quantile(probabilidad_objetivo))
p_normal = float(norm.cdf((plazo - duracion_media) / desviacion_proyecto))
t_normal = float(duracion_media + norm.ppf(probabilidad_objetivo) * desviacion_proyecto)
pd.DataFrame({'Aproximación normal': [duracion_media, desviacion_proyecto, p_normal * 100, t_normal],
              'Monte Carlo': [duracion_proyecto_mc.mean(), duracion_proyecto_mc.std(), p_mc * 100, t_mc]},
             index=['Duración media', 'Desviación típica', f'P(T ≤ {plazo}) (%)', f'Duración para {int(probabilidad_objetivo*100)}%']).round(2)

# %% [markdown]
# La aproximación normal sólo mira la ruta crítica. En cada réplica, sin embargo, la duración del proyecto es el máximo de todas las rutas, y el máximo de varias variables aleatorias tiene una media mayor que cualquiera de ellas: las rutas casi críticas empujan la duración hacia arriba (*sesgo de fusión*). Por eso Monte Carlo da una duración media algo mayor y una probabilidad de cumplir el plazo algo menor: el método PERT clásico es optimista. La desviación típica simulada suele ser menor, porque cuando una ruta sale corta otra ocupa su lugar y la cola izquierda se recorta.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# El índice de criticidad de una ruta es la fracción de réplicas en que es la más larga; el de una actividad, la fracción de réplicas en que pertenece a la ruta más larga.

# %%
#| tbl-cap: "Índice de criticidad de las rutas"
#| label: tbl-criticidadRutas
ruta_critica_mc = duraciones_rutas_mc.idxmax(axis='columns')
criticidad_rutas = (ruta_critica_mc.value_counts(normalize=True) * 100).reindex(path_matrix.index, fill_value=0.0)
actividades_ruta = path_matrix.apply(lambda fila: ' - '.join(fila.index[fila == 1]), axis=1)
pd.DataFrame({'actividades': actividades_ruta, 'duración media': duraciones_rutas_medias,
              'criticidad (%)': criticidad_rutas}).sort_values('criticidad (%)', ascending=False).round(1)

# %%
#| tbl-cap: "Índice de criticidad de las actividades frente a su holgura total"
#| label: tbl-criticidadActividades
criticidad_actividades = path_matrix.loc[ruta_critica_mc].mean() * 100
calendario = proyecto.calendar(datos_proyecto['duration'])
pd.DataFrame({'holgura total': calendario['H_total'], 'criticidad (%)': criticidad_actividades}).round(1).T

# %% [markdown]
# La librería hace el mismo cálculo de forma vectorizada: la sensibilidad empírica media de `monte_carlo` es precisamente el índice de criticidad de cada actividad.

# %%
resultado = proyecto.monte_carlo({act: ('pert', fila['a'], fila['m'], fila['b']) for act, fila in datos_proyecto.iterrows()}, n_samples=N, seed=557)
display(resultado.summary().round(2))
(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%) según monte_carlo').T

# %%
vigilar = criticidad_actividades[calendario['H_total'] > 0].sort_values(ascending=False)
Markdown(f'''La actividad no crítica con mayor índice de criticidad es **{vigilar.index[0]}** ({vigilar.iloc[0]:.1f} % de las réplicas en la ruta más larga, con holgura {int(calendario.loc[vigilar.index[0], "H_total"])}). El análisis determinista la da por holgada, pero un retraso pequeño la convierte en crítica: es la que conviene vigilar.''')
