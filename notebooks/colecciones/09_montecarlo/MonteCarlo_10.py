# ---
# title: MonteCarlo 10
# subtitle: "8 actividades, 6 rutas, distribución triangular"
# coleccion: 09_montecarlo
# variante: triangular
# nivel: 2
# grafo: roy
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
pd.set_option('display.max_colwidth', None)   # las columnas «… = suma» no se truncan con «...»

# Versión local de la librería: se busca src/pmule.py subiendo desde el directorio actual
_aqui = os.path.abspath('.')
while _aqui != os.path.dirname(_aqui) and not os.path.isfile(os.path.join(_aqui, 'src', 'pmule.py')):
    _aqui = os.path.dirname(_aqui)
sys.path.append(os.path.join(_aqui, 'src'))
from pmule import (PredecessorTable, ProjectGraph, EarnedValue,
                   pinta_tikz, plot_cargas, plot_curvas_valor_ganado,
                   tanto_por_uno, ordena_rutas,
                   explica_camino_critico, explica_rutas, explica_suma, explica_sigma_ruta,
                   explica_probabilidad, explica_intervalo, explica_percentil,
                   plural, lista_y, ordena_ruta, limpia_tabla, gantt_tramos_tikz)

GANTT_PARAMS = dict(arrows=True)   # flechas de prelación en los diagramas de Gantt (rojas las críticas)


# %%
# Funciones auxiliares para mostrar los cálculos intermedios paso a paso
from itertools import combinations as _combinations


def detalle_tiempos_nodos(proyecto, duraciones):
    """Tiempos tempranos y tardíos de los nodos con los candidatos que se comparan en cada uno."""
    G = proyecto.pert_graph
    d = {data['activity']: (0 if data['activity'].startswith('@') else int(duraciones[data['activity']]))
         for _, _, data in G.edges(data=True)}
    nodos = list(proyecto.nodes)
    early, filas = {}, []
    for j in nodos:
        entrantes = [(i, data['activity']) for i, _, data in G.in_edges(j, data=True)]
        if not entrantes:
            early[j] = 0
            filas.append({'nodo': j, 'candidatos t_i + d_ij': 'nodo inicial', 'tiempo temprano = máx': 0})
            continue
        cand = {f'{a}: {early[i]} + {d[a]} = {early[i] + d[a]}': early[i] + d[a] for i, a in entrantes}
        early[j] = max(cand.values())
        filas.append({'nodo': j, 'candidatos t_i + d_ij': '; '.join(cand), 'tiempo temprano = máx': early[j]})
    tempranos = pd.DataFrame(filas).set_index('nodo')
    late, filas = {}, []
    for i in reversed(nodos):
        salientes = [(j, data['activity']) for _, j, data in G.out_edges(i, data=True)]
        if not salientes:
            late[i] = early[i]
            filas.append({'nodo': i, 'candidatos t*_j − d_ij': 'nodo final (= su tiempo temprano)', 'tiempo tardío = mín': late[i]})
            continue
        cand = {f'{a}: {late[j]} − {d[a]} = {late[j] - d[a]}': late[j] - d[a] for j, a in salientes}
        late[i] = min(cand.values())
        filas.append({'nodo': i, 'candidatos t*_j − d_ij': '; '.join(cand), 'tiempo tardío = mín': late[i]})
    tardios = pd.DataFrame(filas).set_index('nodo')
    return tempranos, tardios


def detalle_holguras(calendario, prelaciones):
    """Cómo se obtiene cada fecha y cada holgura a partir de las vecinas."""
    pred = prelaciones.immediate_predecessor
    suc = {a: sorted(s for s, ps in pred.items() if a in ps) for a in calendario.index}
    fin = int(calendario['fin_mas_temprano'].max())
    filas = []
    for a in calendario.index:
        ES, EF = int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'fin_mas_temprano'])
        LS, LF = int(calendario.loc[a, 'inicio_mas_tardio']), int(calendario.loc[a, 'fin_mas_tardio'])
        dur = int(calendario.loc[a, 'duracion'])
        ps, ss = sorted(pred[a]), suc[a]
        ef_pred = {p: int(calendario.loc[p, 'fin_mas_temprano']) for p in ps}
        es_suc = {s: int(calendario.loc[s, 'inicio_mas_temprano']) for s in ss}
        lf_pred = {p: int(calendario.loc[p, 'fin_mas_tardio']) for p in ps}
        ls_suc = {s: int(calendario.loc[s, 'inicio_mas_tardio']) for s in ss}
        min_es_suc = min(es_suc.values()) if es_suc else fin
        max_lf_pred = max(lf_pred.values()) if lf_pred else 0
        filas.append({
            'actividad': a,
            'ES = máx EF pred.': (f"máx({', '.join(f'{p}: {v}' for p, v in ef_pred.items())}) = {ES}" if ps else f'inicial → {ES}'),
            'EF = ES + d': f'{ES} + {dur} = {EF}',
            'LF = mín LS suc.': (f"mín({', '.join(f'{s}: {v}' for s, v in ls_suc.items())}) = {LF}" if ss else f'final → {LF}'),
            'LS = LF − d': f'{LF} − {dur} = {LS}',
            'H_total = LS − ES': f'{LS} − {ES} = {LS - ES}',
            'H_libre = mín ES suc. − EF': f"{min_es_suc}{'' if ss else ' (fin)'} − {EF} = {min_es_suc - EF}",
            'H_indep. = máx(0, mín ES suc. − máx LF pred. − d)': f'máx(0, {min_es_suc} − {max_lf_pred} − {dur}) = {max(0, min_es_suc - max_lf_pred - dur)}',
        })
    return pd.DataFrame(filas).set_index('actividad')


def detalle_rutas(R, valores, nombre='duración', decimales=0):
    """Suma término a término de un valor (duraciones, varianzas…) a lo largo de cada ruta."""
    filas = {}
    for r in R.index:
        acts = list(R.columns[R.loc[r] == 1])
        total = float(sum(valores[a] for a in acts))
        filas[r] = {'actividades': ' - '.join(acts),
                    f'{nombre} = suma': ' + '.join(f'{a}({valores[a]:.{decimales}f})' for a in acts) + f' = {total:.{decimales}f}',
                    nombre: total}
    return pd.DataFrame(filas).T


def detalle_estimaciones(datos, a='a', m='m', b='b'):
    """Media y varianza PERT de cada actividad con los números sustituidos."""
    filas = {}
    for act, fila in datos.iterrows():
        A, M, B = float(fila[a]), float(fila[m]), float(fila[b])
        filas[act] = {'D = (a + 4m + b)/6': f'({A:g} + 4·{M:g} + {B:g})/6 = {(A + 4 * M + B) / 6:.2f}',
                      'σ² = ((b − a)/6)²': f'(({B:g} − {A:g})/6)² = {((B - A) / 6) ** 2:.3f}'}
    return pd.DataFrame(filas).T


def detalle_ackoff(proyecto, duraciones, topes, cur, pasos):
    """En cada paso de Ackoff-Sasieni: rutas críticas, opciones consideradas (actividades comunes y parejas)
    con su coste, y la elegida."""
    R = proyecto.path_matrix(dummies=False)
    d = duraciones.astype(float).copy()
    filas = []
    for paso, elegidas in sorted(pasos.items()):
        L = R @ d[R.columns]
        T = float(L.max())
        criticas = [r for r in L.index if np.isclose(L[r], T)]
        conjuntos = [set(R.columns[R.loc[r] == 1]) for r in criticas]
        con_margen = {x for x in R.columns if d[x] > topes[x] and np.isfinite(cur[x])}
        opciones = {x: float(cur[x]) for x in set.intersection(*conjuntos) & con_margen}
        if len(conjuntos) > 1:
            cand = sorted(set().union(*conjuntos) & con_margen)
            for p, q in _combinations(cand, 2):
                if all(p in s or q in s for s in conjuntos) and p not in opciones and q not in opciones:
                    opciones[f'{p}+{q}'] = float(cur[p] + cur[q])
        orden = sorted(opciones.items(), key=lambda kv: kv[1])
        filas.append({'paso': paso + 1, 'duración': int(T), 'rutas críticas': ', '.join(criticas),
                      'opciones (coste)': '; '.join(f'{k} ({round(v, 2):g})' for k, v in orden[:6]) + (' …' if len(orden) > 6 else ''),
                      'elegida': ', '.join(elegidas), 'coste del paso': float(sum(cur[x] for x in elegidas))})
        for x in elegidas:
            d[x] -= 1
    return pd.DataFrame(filas).set_index('paso')


def detalle_valor_ganado(datos_vg, calendario, semana):
    """PV, EV y AC actividad a actividad en el punto de control."""
    filas = {}
    for a, fila in datos_vg.iterrows():
        es, dur = int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'duracion'])
        frac = min(max(semana - es, 0), dur) / dur
        filas[a] = {'BAC': float(fila['BAC']), 'planificado hasta la semana': f'{min(max(semana - es, 0), dur)} de {dur}',
                    'PV = BAC·fracción': f'{fila["BAC"]:.0f}·{frac:.2f} = {fila["BAC"] * frac:.1f}',
                    'realizado (%)': int(fila['performed']),
                    'EV = BAC·%': f'{fila["BAC"]:.0f}·{fila["performed"] / 100:.2f} = {fila["BAC"] * fila["performed"] / 100:.1f}',
                    'AC': float(fila['AC']), 'PV_i': fila['BAC'] * frac, 'EV_i': fila['BAC'] * fila['performed'] / 100}
    t = pd.DataFrame(filas).T
    return t


def muestra_formula(nombre, expresion, valor, unidades=''):
    """Una línea «nombre = expresión = valor» en Markdown."""
    return Markdown(f'**{nombre}** = {expresion} = **{valor}**{(" " + unidades) if unidades else ""}')

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Considere el proyecto cuyo cuadro de prelaciones y estimaciones optimista, más probable y pesimista (en semanas) se adjunta. Suponga que la duración de cada actividad sigue una distribución triangular de parámetros $(a, m, b)$.

# %%
pregunta_1 = "1. Dibuje el grafo Roy, construya la matriz de caminos y calcule la duración del proyecto con las duraciones medias.  "
pregunta_2 = "2. Genere 10000 réplicas de las duraciones de las actividades y, con la matriz de caminos, obtenga la duración del proyecto en cada una. Calcule su media, desviación típica y percentiles 5, 50 y 95.  "
pregunta_3 = "3. Calcule la probabilidad de terminar en 37 semanas o menos y la duración que puede comprometerse con un 90% de probabilidad. Compare con la aproximación normal por el camino crítico y explique las diferencias.  "
pregunta_4 = "4. Calcule el índice de criticidad de cada ruta y de cada actividad. ¿Qué actividad no crítica convendría vigilar?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b 
    ('A'     , '---'      , 1, 3, 5 ),
    ('B'     , '---'      , 5, 7, 9 ),
    ('C'     , '---'      , 6, 8, 10),
    ('D'     , 'A,B,C'    , 4, 7, 10),
    ('E'     , 'D'        , 2, 5, 8 ),
    ('F'     , 'E'        , 6, 9, 9 ),
    ('G'     , 'F'        , 4, 7, 10),
    ('H'     , 'F'        , 4, 9, 11),
], columns=['activity', 'predecessor', 'a', 'm', 'b']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la puesta en marcha de una línea de envasado; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Especificación de la línea',
    'B': 'Adjudicación de los equipos',
    'C': 'Obra civil y bancadas',
    'D': 'Recepción de la llenadora',
    'E': 'Recepción de la etiquetadora',
    'F': 'Montaje mecánico',
    'G': 'Instalación neumática',
    'H': 'Cableado y armarios de control',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %% [markdown]
# Para la distribución triangular de mínimo $a$, moda $m$ y máximo $b$ la media es $(a + m + b)/3$ y la varianza $(a^2 + m^2 + b^2 - ab - am - bm)/18$. Sustituyendo:

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = datos_proyecto.eval('(a + m + b) / 3').round().astype(int)
datos_proyecto['variance'] = datos_proyecto.eval('(a**2 + m**2 + b**2 - a*b - a*m - b*m) / 18')
datos_proyecto['standard_deviation'] = np.sqrt(datos_proyecto['variance'])
pd.DataFrame({'media = (a + m + b)/3': [f"({r.a:g} + {r.m:g} + {r.b:g})/3 = {(r.a + r.m + r.b) / 3:.2f}" for r in datos_proyecto.itertuples()],
              'varianza = (a² + m² + b² − ab − am − bm)/18': [f"({r.a:g}² + {r.m:g}² + {r.b:g}² − {r.a:g}·{r.b:g} − {r.a:g}·{r.m:g} − {r.b:g}·{r.m:g})/18 = {r.variance:.3f}" for r in datos_proyecto.itertuples()],
              'σ': datos_proyecto['standard_deviation'].round(3).values}, index=datos_proyecto.index)

# %%
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %%
def tabla_rutas(R, medias, varianzas=None, calendario=None, dec=0, dec_var=3, nombre='media'):
    """Media (y varianza) de cada ruta con la suma término a término; actividades en orden de ejecución."""
    filas = {}
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario)
        m = float(sum(float(medias[a]) for a in acts))
        fila = {'actividades': ' – '.join(acts),
                f'{nombre} = suma': ' + '.join(f'{a}({float(medias[a]):.{dec}f})' for a in acts) + f' = {m:.{dec}f}',
                nombre: m}
        if varianzas is not None:
            v = float(sum(float(varianzas[a]) for a in acts))
            fila['varianza = suma'] = ' + '.join(f'{a}({float(varianzas[a]):.{dec_var}f})' for a in acts) + f' = {v:.{dec_var}f}'
            fila['varianza'] = v
            fila['σ = √varianza'] = v ** 0.5
        filas[r] = fila
    t = pd.DataFrame(filas).T
    for c in (nombre, 'varianza', 'σ = √varianza'):
        if c in t:
            t[c] = t[c].astype(float)
    return t

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con las duraciones medias"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# La matriz de caminos tiene una fila por ruta y una columna por actividad (1 si la actividad está en la ruta):

# %%
#| tbl-cap: "Matriz de caminos"
#| label: tbl-matrizCaminos
path_matrix = proyecto.path_matrix(dummies=False).drop_duplicates()   # rutas que sólo difieren en ficticias, una vez
proyecto.display_rutas("roy")

# %% [markdown]
# Duración media y varianza de cada ruta (producto de la matriz de caminos por las medias y por las varianzas de las actividades):

# %%
calendario = proyecto.calendar(datos_proyecto['duration'])
duraciones = datos_proyecto['duration'][path_matrix.columns]
duraciones_rutas_medias = path_matrix @ duraciones
duracion_media = int(duraciones_rutas_medias.max())
ruta_critica = duraciones_rutas_medias.idxmax()
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])
rutas = tabla_rutas(path_matrix, datos_proyecto['duration'], datos_proyecto['variance'], calendario)
limpia_tabla(rutas, decimales=3)

# %%
acts_critica = ordena_ruta(list(path_matrix.columns[path_matrix.loc[ruta_critica] == 1]), calendario)
Markdown(f'''{explica_sigma_ruta(acts_critica, datos_proyecto["variance"], nombre="σ_T")}.\n\n'''
         f'''**Respuesta:** con las duraciones medias el proyecto dura {plural(duracion_media, "semana")}; la ruta crítica es '''
         f'''{" – ".join(acts_critica)}, con desviación típica σ_T = {desviacion_proyecto:.3f} semanas.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Cada réplica es un vector de duraciones generado con la distribución triangular de mínimo $a$, moda $m$ y máximo $b$ (cuando $a = m = b$ la duración es fija).

# %%
N = 10000
rng = np.random.default_rng(8006)
def muestras_triangular(a, m, b, size):
    return np.full(size, float(a)) if a == b else rng.triangular(a, m, b, size)
duraciones_mc = pd.DataFrame({act: muestras_triangular(fila['a'], fila['m'], fila['b'], N)
                              for act, fila in datos_proyecto.iterrows()})
duraciones_mc = duraciones_mc[path_matrix.columns]

# %% [markdown]
# Comprobación de que las muestras reproducen la media y la dispersión de cada actividad:

# %%
#| tbl-cap: "Medias y desviaciones típicas muestrales frente a las teóricas"
#| label: tbl-comprobacionMuestras
pd.DataFrame({'media teórica': datos_proyecto['duration'], 'media muestral': duraciones_mc.mean(),
              'σ teórica': datos_proyecto['standard_deviation'], 'σ muestral': duraciones_mc.std()}).round(2).T

# %% [markdown]
# La duración de cada ruta en cada réplica es la suma de las duraciones de sus actividades (producto de las duraciones por la traspuesta de la matriz de caminos), y la del proyecto es la de la ruta más larga: el máximo por filas. Las cinco primeras réplicas:

# %%
duraciones_rutas_mc = duraciones_mc @ path_matrix.T
duracion_proyecto_mc = duraciones_rutas_mc.max(axis='columns')
pd.concat([duraciones_rutas_mc, duracion_proyecto_mc.rename('Proyecto (máximo)')], axis=1).head().round(2)

# %% [markdown]
# Los descriptores se calculan sobre las réplicas de la duración del proyecto. El percentil $p$ es el valor que deja por debajo la proporción $p$ de las réplicas: se ordenan de menor a mayor y se toma la que ocupa la posición $p\cdot N$ (interpolando entre las dos vecinas).

# %%
#| tbl-cap: "Descriptores de la duración simulada del proyecto"
#| label: tbl-descriptores
descriptores = duracion_proyecto_mc.describe(percentiles=[0.05, 0.5, 0.95])
descriptores.index = ['réplicas', 'media', 'desviación típica', 'mínimo', 'P5', 'P50', 'P95', 'máximo']
descriptores.round(2).to_frame('Duración del proyecto (semanas)').T

# %%
ordenadas = np.sort(duracion_proyecto_mc.values)
Markdown(f'''Por ejemplo, el P95 está entre la réplica {int(0.95 * (N - 1)) + 1} y la {int(0.95 * (N - 1)) + 2} de las {N} ordenadas '''
         f'''({ordenadas[int(0.95 * (N - 1))]:.2f} y {ordenadas[int(0.95 * (N - 1)) + 1]:.2f}).\n\n'''
         f'''**Respuesta:** media {descriptores["media"]:.2f} semanas, desviación típica {descriptores["desviación típica"]:.2f}; '''
         f'''percentiles P5 = {descriptores["P5"]:.2f}, P50 = {descriptores["P50"]:.2f} y P95 = {descriptores["P95"]:.2f} semanas.''')

# %%
#| fig-cap: "Distribución simulada de la duración frente a la aproximación normal"
#| label: fig-histograma
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(duracion_proyecto_mc, bins=60, density=True, alpha=0.5, color='skyblue', label='Monte Carlo (triangular)')
x = np.linspace(duracion_proyecto_mc.min(), duracion_proyecto_mc.max(), 300)
ax.plot(x, norm.pdf(x, duracion_media, desviacion_proyecto), color='navy', label='Aproximación normal (camino crítico)')
ax.axvline(37, color='red', linestyle='--', label='Plazo (37)')
ax.set_xlabel('Duración del proyecto (semanas)'); ax.legend(); plt.show()

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Con las réplicas, la probabilidad de cumplir el plazo es la proporción de réplicas que no lo superan, y la duración que se cumple con una probabilidad dada es el percentil empírico correspondiente. Con la aproximación normal, $T \sim N(\mu, \sigma_T)$ con la media y la desviación típica de la ruta crítica (apartado 1).

# %%
plazo, probabilidad_objetivo = 37, 0.9
cumplen = int((duracion_proyecto_mc <= plazo).sum())
p_mc = float((duracion_proyecto_mc <= plazo).mean())
t_mc = float(duracion_proyecto_mc.quantile(probabilidad_objetivo))
p_normal = float(norm.cdf(plazo, duracion_media, desviacion_proyecto))
t_normal = float(norm.ppf(probabilidad_objetivo, duracion_media, desviacion_proyecto))
pct = int(probabilidad_objetivo * 100)
Markdown(f'''Monte Carlo: {cumplen} de las {N} réplicas duran {plazo} semanas o menos, luego P(T ≤ {plazo}) ≈ {cumplen}/{N} = **{p_mc:.4f}**; '''
         f'''el percentil {pct} de las réplicas (posición ≈ {int(probabilidad_objetivo * N)} de {N} ordenadas) es **{t_mc:.2f}** semanas.\n\n'''
         f'''Aproximación normal:\n\n- {explica_probabilidad(duracion_media, desviacion_proyecto, plazo)}\n'''
         f'''- {explica_percentil(duracion_media, desviacion_proyecto, probabilidad_objetivo, "semanas")}''')

# %%
#| tbl-cap: "Aproximación normal frente a Monte Carlo"
#| label: tbl-comparacion
pd.DataFrame({'Aproximación normal': [duracion_media, desviacion_proyecto, p_normal * 100, t_normal],
              'Monte Carlo': [duracion_proyecto_mc.mean(), duracion_proyecto_mc.std(), p_mc * 100, t_mc]},
             index=['Duración media', 'Desviación típica', f'P(T ≤ {plazo}) (%)', f'Duración para {pct}%']).round(2)

# %%
ruta_critica_mc = duraciones_rutas_mc.idxmax(axis='columns')
criticidad_rutas = (ruta_critica_mc.value_counts(normalize=True) * 100).reindex(path_matrix.index, fill_value=0.0)
dm = duracion_proyecto_mc.mean() - duracion_media
ds = duracion_proyecto_mc.std() - desviacion_proyecto
otras = 100 - criticidad_rutas[ruta_critica]
if dm > 0.05:
    t_media = (f'Monte Carlo da una duración media {dm:.2f} semanas mayor. En cada réplica la duración del proyecto es el máximo de todas '
               f'las rutas y en el {otras:.1f} % de las réplicas manda otra ruta distinta de la crítica: las rutas casi críticas empujan la '
               'duración hacia arriba (*sesgo de fusión*), algo que la aproximación normal, que sólo mira la ruta crítica, no recoge.')
else:
    t_media = (f'Las medias casi coinciden (diferencia {dm:+.2f} semanas): la ruta crítica es la más larga en el {100 - otras:.1f} % de las '
               'réplicas, así que el máximo de las rutas apenas se separa de ella y el sesgo de fusión es pequeño.')
if p_mc < p_normal:
    t_prob = (f'La probabilidad de cumplir el plazo es menor con Monte Carlo ({p_mc:.4f} frente a {p_normal:.4f}) y la duración al {pct} % '
              f'mayor ({t_mc:.2f} frente a {t_normal:.2f}): la aproximación normal por el camino crítico es optimista.')
else:
    t_prob = (f'La probabilidad de cumplir el plazo es {"igual" if np.isclose(p_mc, p_normal, atol=5e-4) else "algo mayor"} con Monte Carlo ({p_mc:.4f} frente a {p_normal:.4f}): '
              'con una sola ruta dominante la diferencia queda dentro del error de muestreo y de la forma de las distribuciones.')
dist_pert = False
if dist_pert and ds > 0.02:
    t_sd = (f'La desviación típica simulada es mayor ({duracion_proyecto_mc.std():.2f} frente a {desviacion_proyecto:.2f}) porque la aproximación '
            'PERT σ = (b − a)/6 subestima la dispersión real de la Beta-PERT (ver la comprobación de las muestras).')
elif ds < -0.02:
    t_sd = (f'La desviación típica simulada es menor ({duracion_proyecto_mc.std():.2f} frente a {desviacion_proyecto:.2f}): cuando la ruta '
            'crítica sale corta otra ocupa su lugar y la cola izquierda se recorta.')
else:
    t_sd = (f'Las desviaciones típicas son parecidas ({duracion_proyecto_mc.std():.2f} frente a {desviacion_proyecto:.2f}).')
Markdown(f'''{t_media} {t_prob} {t_sd}\n\n'''
         f'''**Respuesta:** P(T ≤ {plazo}) ≈ {p_mc:.4f} con Monte Carlo y {p_normal:.4f} con la aproximación normal; la duración que se cumple con un '''
         f'''{pct} % de probabilidad es {t_mc:.2f} semanas con Monte Carlo y {t_normal:.2f} con la normal (en semanas enteras, '''
         f'''{int(np.ceil(round(t_mc, 2)))} y {int(np.ceil(round(t_normal, 2)))}).''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# El índice de criticidad de una ruta es la fracción de réplicas en que es la más larga; el de una actividad, la fracción de réplicas en que pertenece a la ruta más larga.

# %%
#| tbl-cap: "Índice de criticidad de las rutas"
#| label: tbl-criticidadRutas
pd.DataFrame({'actividades': rutas['actividades'], 'duración media': duraciones_rutas_medias,
              'veces la más larga': ruta_critica_mc.value_counts().reindex(path_matrix.index, fill_value=0),
              'criticidad (%)': criticidad_rutas}).sort_values('criticidad (%)', ascending=False).round(1)

# %%
#| tbl-cap: "Índice de criticidad de las actividades frente a su holgura total"
#| label: tbl-criticidadActividades
criticidad_actividades = path_matrix.loc[ruta_critica_mc].mean() * 100
pd.DataFrame({'holgura total': calendario['H_total'], 'criticidad (%)': criticidad_actividades}).round(1).T

# %% [markdown]
# La librería hace el mismo cálculo de forma vectorizada: la sensibilidad empírica media de `monte_carlo` es precisamente el índice de criticidad de cada actividad.

# %%
resultado = proyecto.monte_carlo({act: ('triangular', fila['a'], fila['m'], fila['b']) for act, fila in datos_proyecto.iterrows()}, n_samples=N, seed=8006)
display(resultado.summary().round(2).rename(columns={'mean': 'media', 'std': 'desv. típica', 'q05': 'P5', 'q50': 'P50', 'q95': 'P95', 'min': 'mínimo', 'max': 'máximo'}))
(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%) según monte_carlo').T

# %%
vigilar = criticidad_actividades[calendario['H_total'] > 0].sort_values(ascending=False)
rutas_txt = '; '.join(f"{rutas.loc[r, 'actividades']}: {criticidad_rutas[r]:.1f} %" for r in criticidad_rutas.sort_values(ascending=False).index if criticidad_rutas[r] > 0)
if len(vigilar) and vigilar.iloc[0] > 0:
    x = vigilar.index[0]
    t_vig = (f'La actividad no crítica que convendría vigilar es **{x}**: está en la ruta más larga en el {vigilar.iloc[0]:.1f} % de las réplicas '
             f'aunque tiene {plural(int(calendario.loc[x, "H_total"]), "semana")} de holgura total. El análisis determinista la da por holgada, '
             'pero un retraso pequeño la convierte en crítica.')
else:
    t_vig = 'Ninguna actividad no crítica llega a estar en la ruta más larga en las réplicas: no hay ninguna que vigilar especialmente.'
Markdown(f'''**Respuesta:** índices de criticidad de las rutas (las que alguna vez son la más larga): {rutas_txt}. Los de las actividades '''
         f'''son los de la tabla. {t_vig}''')
