# ---
# title: MonteCarlo 02
# subtitle: "9 actividades, 6 rutas, distribución triangular"
# coleccion: 09_montecarlo
# variante: triangular
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

GANTT_PARAMS = dict(arrows=True)   # flechas de prelación en los diagramas de Gantt (rojas las críticas)


# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones y estimaciones optimista, más probable y pesimista (en semanas) se adjunta. Suponga que la duración de cada actividad sigue una distribución triangular de parámetros $(a, m, b)$.

# %%
pregunta_1 = "1. Dibuje el grafo PERT, construya la matriz de caminos y calcule la duración del proyecto con las duraciones medias.  "
pregunta_2 = "2. Genere 10000 réplicas de las duraciones de las actividades y, con la matriz de caminos, obtenga la duración del proyecto en cada una. Calcule su media, desviación típica y percentiles 5, 50 y 95.  "
pregunta_3 = "3. Calcule la probabilidad de terminar en 33 semanas o menos y la duración que puede comprometerse con un 90% de probabilidad. Compare con la aproximación normal por el camino crítico y explique las diferencias.  "
pregunta_4 = "4. Calcule el índice de criticidad de cada ruta y de cada actividad. ¿Qué actividad no crítica convendría vigilar?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b
    ('A'     , '---'      , 1, 4, 7),
    ('B'     , '---'      , 5, 6, 7),
    ('C'     , '---'      , 3, 6, 6),
    ('D'     , 'A,B,C'    , 3, 6, 6),
    ('E'     , 'D'        , 6, 6, 9),
    ('F'     , 'E'        , 3, 3, 6),
    ('G'     , 'F'        , 1, 3, 5),
    ('H'     , 'E'        , 7, 8, 9),
    ('I'     , 'H'        , 4, 5, 9),
], columns=['activity', 'predecessor', 'a', 'm', 'b']).set_index('activity')
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
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %% [markdown]
# Para la distribución triangular la media es $(a + m + b)/3$ y la varianza $(a^2 + m^2 + b^2 - ab - am - bm)/18$.

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = datos_proyecto.eval('(a + m + b) / 3').round().astype(int)
datos_proyecto['variance'] = datos_proyecto.eval('(a**2 + m**2 + b**2 - a*b - a*m - b*m) / 18')
datos_proyecto['standard_deviation'] = np.sqrt(datos_proyecto['variance'])
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
# Cada réplica es un vector de duraciones generado con la distribución triangular de mínimo $a$, moda $m$ y máximo $b$ (cuando $a = m = b$ la duración es fija).

# %%
N = 10000
rng = np.random.default_rng(5943)
def muestras_triangular(a, m, b, size):
    return np.full(size, float(a)) if a == b else rng.triangular(a, m, b, size)
duraciones_mc = pd.DataFrame({act: muestras_triangular(fila['a'], fila['m'], fila['b'], N)
                              for act, fila in datos_proyecto.iterrows()})
duraciones_mc = duraciones_mc[path_matrix.columns]

# %%
#| tbl-cap: "Medias y desviaciones típicas muestrales frente a las teóricas"
#| label: tbl-comprobacionMuestras
pd.DataFrame({'media teórica': datos_proyecto['duration'], 'media muestral': duraciones_mc.mean(),
              'σ teórica': datos_proyecto['standard_deviation'], 'σ muestral': duraciones_mc.std()}).round(2).T

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
ax.hist(duracion_proyecto_mc, bins=60, density=True, alpha=0.5, color='skyblue', label='Monte Carlo (triangular)')
x = np.linspace(duracion_proyecto_mc.min(), duracion_proyecto_mc.max(), 300)
ax.plot(x, norm.pdf(x, duracion_media, desviacion_proyecto), color='navy', label='Aproximación normal (camino crítico)')
ax.axvline(33, color='red', linestyle='--', label='Plazo (33)')
ax.set_xlabel('Duración del proyecto (semanas)'); ax.legend(); plt.show()

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
plazo, probabilidad_objetivo = 33, 0.9
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
resultado = proyecto.monte_carlo({act: ('triangular', fila['a'], fila['m'], fila['b']) for act, fila in datos_proyecto.iterrows()}, n_samples=N, seed=5943)
display(resultado.summary().round(2))
(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%) según monte_carlo').T

# %%
vigilar = criticidad_actividades[calendario['H_total'] > 0].sort_values(ascending=False)
Markdown(f'''La actividad no crítica con mayor índice de criticidad es **{vigilar.index[0]}** ({vigilar.iloc[0]:.1f} % de las réplicas en la ruta más larga, con holgura {int(calendario.loc[vigilar.index[0], "H_total"])}). El análisis determinista la da por holgada, pero un retraso pequeño la convierte en crítica: es la que conviene vigilar.''')
