# ---
# title: MonteCarlo 06
# subtitle: "9 actividades; convergencia de la media y del percentil 90 con N"
# coleccion: 09_montecarlo
# variante: convergencia
# nivel: 3
# grafo: pert
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (semanas, distribución normal) se adjunta. Se quiere saber cuántas réplicas de Monte Carlo hacen falta para estimar con precisión la duración del proyecto.

# %%
pregunta_1 = "1. Construya la matriz de rutas y simule el proyecto con N = 20, 200, 2 000 y 20 000 réplicas. Anote en cada caso la media, la desviación típica, el percentil 90 y la probabilidad de terminar en 34 semanas.  "
pregunta_2 = "2. Repita cada simulación 10 veces con semillas distintas y calcule la dispersión (desviación típica) de la media estimada entre repeticiones. ¿Cómo decrece con N?  "
pregunta_3 = "3. Compare esa dispersión con la fórmula del error típico de la media, σ/√N. ¿Cuántas réplicas hacen falta para que el error de la media sea menor de 0,1 semanas?  "
pregunta_4 = "4. ¿Necesita el percentil 90 más o menos réplicas que la media para la misma precisión? ¿Por qué?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 3       , 0.6               ),
    ('B'     , '---'      , 1       , 1                 ),
    ('C'     , '---'      , 2       , 0.1               ),
    ('D'     , 'A,B,C'    , 8       , 0.3               ),
    ('E'     , 'D'        , 7       , 0.4               ),
    ('F'     , 'E'        , 4       , 0.6               ),
    ('G'     , 'F'        , 2       , 0.1               ),
    ('H'     , 'F'        , 4       , 0.4               ),
    ('I'     , 'G,H'      , 7       , 0.6               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
mu = datos_proyecto.loc[R.columns, 'duration'].values[:, None]
sigma = datos_proyecto.loc[R.columns, 'standard_deviation'].values[:, None]
plazo = 34
def simular(N, semilla):
    rng = np.random.default_rng(semilla)
    T = np.clip(rng.normal(mu, sigma, (len(R.columns), N)), 0, None)
    return (R.values @ T).max(axis=0)

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
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones medias y el camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Matriz de rutas"
#| label: tbl-matrizRutas
proyecto.display_rutas("pert")

# %%
#| tbl-cap: "Media y varianza de cada ruta (sumas de las de sus actividades)"
#| label: tbl-rutas
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
calendario = proyecto.calendar(datos_proyecto['duration'])
limpia_tabla(tabla_rutas(R.drop_duplicates(), datos_proyecto['duration'], datos_proyecto['variance'], calendario, dec_var=2), decimales=2)

# %% [markdown]
# Cada réplica es un vector de duraciones normales; la duración de cada ruta es R·t y la del proyecto, el máximo. Para cada tamaño N se repite el mismo cálculo con más columnas (semilla 0). La probabilidad es la proporción de réplicas que no superan el plazo y el percentil 90, el valor que deja por debajo el 90 % de las réplicas ordenadas.

# %%
tamanos = [20, 200, 2000, 20000]
filas = {}
for N in tamanos:
    tau = simular(N, 0)
    filas[N] = {'media': tau.mean(), 'desviación típica': tau.std(ddof=1), 'percentil 90': np.quantile(tau, 0.9), f'P(T ≤ {plazo})': (tau <= plazo).mean()}
resultados = pd.DataFrame(filas).T
resultados.index.name = 'N'
resultados.round(3)

# %%
r = resultados
Markdown(f'''**Respuesta:** con N = 20 000 la media es {r.loc[20000, 'media']:.3f} semanas, la desviación típica {r.loc[20000, 'desviación típica']:.3f}, '''
         f'''el percentil 90 {r.loc[20000, 'percentil 90']:.3f} y P(T ≤ {plazo}) = {r.loc[20000, f'P(T ≤ {plazo})']:.3f}. Con N = 20 las estimaciones son poco fiables '''
         f'''(media {r.loc[20, 'media']:.3f}, σ {r.loc[20, 'desviación típica']:.3f}, P {r.loc[20, f'P(T ≤ {plazo})']:.3f}); entre N = 2 000 y N = 20 000 '''
         f'''la media sólo cambia {abs(r.loc[20000, 'media'] - r.loc[2000, 'media']):.3f} semanas y la probabilidad '''
         f'''{abs(r.loc[20000, f'P(T ≤ {plazo})'] - r.loc[2000, f'P(T ≤ {plazo})']):.3f}.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Para cada N se repite la simulación con las semillas 1 a 10 y se calcula la desviación típica de las 10 medias (y de los 10 percentiles 90) obtenidos: mide cuánto varía la estimación de una simulación a otra.

# %%
repeticiones = 10
disp = {}
for N in tamanos:
    medias = [simular(N, s).mean() for s in range(1, repeticiones + 1)]
    p90 = [np.quantile(simular(N, s), 0.9) for s in range(1, repeticiones + 1)]
    disp[N] = {'dispersión de la media': np.std(medias, ddof=1), 'dispersión del percentil 90': np.std(p90, ddof=1)}
dispersion = pd.DataFrame(disp).T
dispersion.index.name = 'N'
dispersion.round(4)

# %%
cocientes = [dispersion['dispersión de la media'].iloc[i] / dispersion['dispersión de la media'].iloc[i + 1] for i in range(len(tamanos) - 1)]
Markdown(f'''Cociente entre la dispersión de la media con N y con 10·N: ''' + '; '.join(f'{tamanos[i]} → {tamanos[i + 1]}: {c:.2f}' for i, c in enumerate(cocientes))
         + f'''.\n\n**Respuesta:** la dispersión de la media pasa de {dispersion["dispersión de la media"].iloc[0]:.4f} (N = 20) a '''
         f'''{dispersion["dispersión de la media"].iloc[-1]:.4f} (N = 20 000): al multiplicar N por 10 se divide aproximadamente por √10 ≈ 3.16 '''
         f'''(cociente medio {np.mean(cocientes):.2f}; con sólo 10 repeticiones cada cociente tiene bastante error), es decir, decrece como 1/√N.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# El error típico de la media de N réplicas independientes es $\sigma_\tau/\sqrt{N}$, donde $\sigma_\tau$ es la desviación típica de la duración del proyecto (se estima con las 20 000 réplicas). Para que sea menor que un error $e$ hace falta $N \ge (\sigma_\tau / e)^2$.

# %%
sigma_tau = simular(20000, 0).std(ddof=1)
dispersion['σ/√N'] = [sigma_tau / np.sqrt(N) for N in tamanos]
N_necesario = int(np.ceil((sigma_tau / 0.1) ** 2))
display(dispersion.round(4))
Markdown(f'''La columna σ/√N ({", ".join(f"{x:.4f}" for x in dispersion["σ/√N"])}) tiene el mismo orden de magnitud que la dispersión observada de la media '''
         f'''({", ".join(f"{x:.4f}" for x in dispersion["dispersión de la media"])}): la fórmula predice bien el error.\n\n'''
         f'''Con σ_τ = {sigma_tau:.3f}: N ≥ ({sigma_tau:.3f} / 0.1)² = {(sigma_tau / 0.1) ** 2:.1f}.\n\n'''
         f'''**Respuesta:** hacen falta al menos {N_necesario} réplicas para que el error típico de la media sea menor de 0,1 semanas.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
ratio = dispersion['dispersión del percentil 90'] / dispersion['dispersión de la media']
Markdown(f'''Cociente entre la dispersión del percentil 90 y la de la media: {", ".join(f"N = {N}: {x:.2f}" for N, x in ratio.items())}.\n\n'''
         f'''**Respuesta:** el percentil 90 necesita más réplicas. Se estima con las réplicas de la cola, que son pocas (el 10 % de N), y su error es '''
         f'''unas {ratio.mean():.1f} veces el de la media para el mismo N; como el error decrece como 1/√N, para igualar la precisión de la media '''
         f'''harían falta unas {ratio.mean() ** 2:.1f} veces más réplicas. Por eso las decisiones basadas en percentiles altos (plazos comprometidos '''
         '''al 90–95 %) necesitan más réplicas que las basadas en la media.''')
