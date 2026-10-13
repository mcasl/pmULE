# ---
# title: Roy 04
# subtitle: "8 actividades, cálculo manual sobre Roy; predecesora dominante de E"
# coleccion: 03_roy
# variante: predecesora-dominante
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta.

# %%
pregunta_1 = "1. Dibuje el grafo Roy y calcule sobre él, actividad a actividad y en orden topológico, el inicio y el fin más tempranos: el inicio más temprano es el mayor de los fines más tempranos de las predecesoras.  "
pregunta_2 = "2. Calcule del mismo modo, de atrás hacia delante, los fines e inicios más tardíos.  "
pregunta_3 = "3. Para cada actividad con varias predecesoras indique cuál de ellas fija su inicio más temprano (la *predecesora dominante*). En particular, ¿qué predecesora de E manda?  "
pregunta_4 = "4. Si se quisiera adelantar el inicio de E un día, ¿qué actividad habría que acortar? ¿Serviría de algo acortar sus otras predecesoras?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 4       ),
    ('B'     , '---'      , 6       ),
    ('C'     , '---'      , 1       ),
    ('D'     , 'A,B,C'    , 6       ),
    ('E'     , 'A,B,C'    , 3       ),
    ('F'     , 'D,E'      , 5       ),
    ('G'     , 'F'        , 4       ),
    ('H'     , 'G'        , 6       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a el lanzamiento de una aplicación móvil; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de requisitos',
    'B': 'Diseño de la arquitectura',
    'C': 'Diseño de interfaz y prototipo',
    'D': 'Base de datos y modelo',
    'E': 'API del servidor',
    'F': 'Pantallas principales',
    'G': 'Módulo de pagos',
    'H': 'Notificaciones y sincronización',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
predecesoras = prelaciones.immediate_predecessor
sucesoras = {a: sorted(s for s, ps in predecesoras.items() if a in ps) for a in datos_proyecto.index}
duracion = datos_proyecto['duration']

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-roy
proyecto.roy(ranksep=1.2, nodesep=1.2)

# %% [markdown]
# Recorriendo las actividades en un orden en que cada una aparece después de sus predecesoras (orden topológico):

# %%
G = nx.DiGraph([(p, a) for a, ps in predecesoras.items() for p in ps])
G.add_nodes_from(datos_proyecto.index)
orden = list(nx.topological_sort(G))
ES, EF, filas = {}, {}, []
for a in orden:
    ps = sorted(predecesoras[a])
    ES[a] = max((EF[p] for p in ps), default=0)
    EF[a] = ES[a] + int(duracion[a])
    filas.append({'actividad': a, 'predecesoras (EF)': ', '.join(f'{p} ({EF[p]})' for p in ps) or '—',
                  'ES = máx': ES[a], 'duración': int(duracion[a]), 'EF = ES + d': EF[a]})
pd.DataFrame(filas).set_index('actividad')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duracion_proyecto = max(EF.values())
LS, LF, filas = {}, {}, []
for a in reversed(orden):
    ss = sucesoras[a]
    LF[a] = min((LS[s] for s in ss), default=duracion_proyecto)
    LS[a] = LF[a] - int(duracion[a])
    filas.append({'actividad': a, 'sucesoras (LS)': ', '.join(f'{s} ({LS[s]})' for s in ss) or '—',
                  'LF = mín': LF[a], 'duración': int(duracion[a]), 'LS = LF − d': LS[a]})
pd.DataFrame(filas).set_index('actividad')

# %%
calendario = proyecto.calendar(duracion)
coincide = all(calendario.loc[a, 'inicio_mas_temprano'] == ES[a] and calendario.loc[a, 'inicio_mas_tardio'] == LS[a] for a in orden)
Markdown(f'La duración del proyecto es {duracion_proyecto} días. El cálculo coincide con el de la librería: {coincide}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
dominantes = {}
for a in datos_proyecto.index:
    ps = sorted(predecesoras[a])
    if len(ps) >= 2:
        maximo = max(EF[p] for p in ps)
        dominantes[a] = ', '.join(p for p in ps if EF[p] == maximo)
pd.Series(dominantes).to_frame('predecesora dominante')

# %% [markdown]
# La predecesora dominante de E es B: su fin más temprano es el mayor de entre las predecesoras y por tanto el que fija el inicio de E. Las demás predecesoras terminan antes y tienen margen (holgura libre) respecto a E.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
objetivo, dominante = 'E', 'B'
otras = [p for p in sorted(predecesoras[objetivo]) if p != dominante]
margen = {p: EF[dominante] - EF[p] for p in otras}
d2 = duracion.copy(); d2[dominante] -= 1
ES_nuevo = int(proyecto.calendar(d2).loc[objetivo, 'inicio_mas_temprano'])
Markdown(f'''Para adelantar {objetivo} hay que acortar {dominante} (o alguna de sus antecesoras en la cadena que la hace llegar la última): acortándola un día, {objetivo} empieza en {ES_nuevo} en vez de {ES[objetivo]}. Acortar {", ".join(otras)} no sirve: terminan {", ".join(f"{p} {m} día(s)" for p, m in margen.items())} antes que {dominante} y su holgura sólo aumentaría. Sólo cuando {dominante} se acorte tanto que otra predecesora pase a ser la última cambiará la dominante.''')
