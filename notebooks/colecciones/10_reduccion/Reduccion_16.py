# ---
# title: Reduccion 16
# subtitle: "duraciones triangulares; plazo de 18 semanas; reducción y probabilidad"
# coleccion: 10_reduccion
# variante: crashing-incertidumbre
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
                      'opciones (coste)': '; '.join(f'{k} ({v:.0f})' for k, v in orden[:6]) + (' …' if len(orden) > 6 else ''),
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
# Considere el proyecto cuyo cuadro de prelaciones, estimaciones de duración (mínima *a*, más probable *m* y máxima *b*, semanas, distribución triangular), duración tope y coste unitario de reducción (k€/semana) se adjunta; la columna *duration* es la duración esperada. Las actividades con CUR infinito no se pueden acelerar. El contrato exige entregar en **18 semanas**.

# %%
pregunta_1 = "1. Con las duraciones esperadas el proyecto dura 20 semanas. Simule 5 000 réplicas y estime la probabilidad de entregar en 18 semanas sin acelerar.  "
pregunta_2 = "2. Aplique Ackoff-Sasieni para reducir el proyecto hasta 18 semanas con las duraciones esperadas. ¿Qué probabilidad de cumplir el plazo da ahora esa reducción? Compruébelo simulando.  "
pregunta_3 = "3. Construya la tabla de probabilidad de cumplir el plazo, duración media simulada y coste de reducción en función de las semanas recortadas (de 0 a 8). Use los mismos números aleatorios para todas las filas.  "
pregunta_4 = "4. ¿Cuántas semanas hay que recortar para tener un 90 % de probabilidad de cumplir? ¿Cuánto cuesta frente a la reducción determinista? ¿Cuánto cuesta cada punto porcentual de probabilidad ganado en cada tramo?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, a, m, b , cap_duration, ucr   
    ('A'     , '---'      , 3       , 1, 2, 6 , 1           , 6     ),
    ('B'     , '---'      , 8       , 6, 8, 10, 5           , 6     ),
    ('C'     , 'A'        , 3       , 1, 3, 5 , 3           , np.inf),
    ('D'     , 'B,C'      , 7       , 5, 7, 9 , 6           , 8     ),
    ('E'     , 'A'        , 3       , 1, 3, 5 , 2           , 8     ),
    ('F'     , 'C,E'      , 5       , 3, 5, 7 , 3           , 6     ),
    ('G'     , 'C'        , 6       , 4, 6, 8 , 6           , np.inf),
    ('H'     , 'D,F,G'    , 5       , 3, 6, 6 , 2           , 5     ),
], columns=['activity', 'predecessor', 'duration', 'a', 'm', 'b', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
plazo = 18
dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}
tope, ucr = datos_proyecto['cap_duration'], datos_proyecto['ucr']

# %% [markdown]
# La duración esperada de una triangular es $(a + m + b)/3$; el cuadro la da ya redondeada en *duration*. Con esas duraciones esperadas se calcula el calendario determinista.

# %%
#| tbl-cap: "Media de cada triangular frente a la duración esperada del cuadro"
#| label: tbl-medias
pd.DataFrame({'(a + m + b)/3': [f"({r['a']:g} + {r['m']:g} + {r['b']:g})/3 = {(r['a'] + r['m'] + r['b']) / 3:.2f}" for _, r in datos_proyecto.iterrows()],
              'duration': datos_proyecto['duration']}, index=datos_proyecto.index).T

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones esperadas y el camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Duración esperada de cada ruta"
#| label: tbl-rutas
R = proyecto.path_matrix(dummies=False)
detalle_rutas(R, datos_proyecto['duration'][R.columns])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cada réplica sortea una duración por actividad con su triangular; la duración del proyecto es el máximo de las rutas (τ = ‖R·t‖∞). La probabilidad de cumplir es la fracción de réplicas con τ ≤ plazo.

# %%
mc0 = proyecto.monte_carlo(dists, n_samples=5000, seed=7)
display(mc0.summary().round(2))
p0 = float((mc0.tau_samples <= plazo).mean())
Markdown(f'Sin acelerar: duración media simulada {mc0.tau_mean:.2f} semanas (determinista: {proyecto.duration(datos_proyecto["duration"])}) y P(τ ≤ {plazo}) = {p0:.3f}.')

# %%
#| fig-cap: "Distribución simulada de la duración sin acelerar"
#| label: fig-histograma
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(mc0.tau_samples, bins=40, density=True, alpha=0.6)
ax.axvline(plazo, color='red', linestyle='--', label=f'plazo ({plazo})')
ax.set_xlabel('duración del proyecto (semanas)'); ax.legend(); plt.show()

# %%
tabla = proyecto.monte_carlo_crashing(dists, datos_proyecto, 'duration', tope, ucr, plazo=plazo,
                                       reduccion_max=8, n_samples=5000, seed=7)
Markdown(f'`monte_carlo_crashing` hace lo mismo para cada número de semanas recortadas, con los mismos números aleatorios: '
         f'sin recortar, duración media {tabla.loc[0, "τ media"]:.2f} y P(τ ≤ {plazo}) = {tabla.loc[0, "P(τ ≤ plazo)"]:.3f}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
r_det = int(tabla.index[tabla['duración determinista'] <= plazo].min())
Markdown(f'Con las duraciones esperadas hay que recortar {r_det} semana(s) para llegar a {plazo}.')

# %% [markdown]
# En cada paso del algoritmo de Ackoff-Sasieni se acorta una semana la ruta (o rutas) crítica(s) eligiendo, entre las actividades críticas con margen, la combinación de menor CUR que reduce *todas* las rutas críticas a la vez.

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni con las duraciones esperadas hasta el plazo"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=tope, costs=ucr, reduction=r_det, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=tope,
    costs=ucr,
    reduction=r_det)

# %%
#| tbl-cap: "Opciones consideradas en cada paso"
#| label: tbl-ackoffPaso
detalle_ackoff(proyecto, datos_proyecto['duration'], tope, ucr, pasos)

# %%
resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],
                        'coste del paso': [float(sum(ucr[acts])) for acts in pasos.values()]},
                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))
resumen['sobrecoste acumulado'] = resumen['coste del paso'].cumsum()
resumen

# %%
Markdown(f'Reducir las duraciones esperadas hasta {plazo} semanas exige {r_det} semana(s) de reducción y {tabla.loc[r_det, "coste"]:.0f} k€ '
         f'(duración determinista {proyecto.duration(duraciones_reducidas)}); pero la probabilidad real de cumplir es {tabla.loc[r_det, "P(τ ≤ plazo)"]:.3f}: '
         'con la duración esperada justo en el plazo se cumple poco más de la mitad de las veces (y menos, por el sesgo de fusión de las rutas), '
         'muy lejos del 90 % que suele exigirse.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Para cada número de semanas recortadas $r$, la tabla da las reducciones que aplica Ackoff-Sasieni (paso a paso, separadas por |), el coste, la duración determinista, la media y el percentil 90 simulados y la probabilidad de cumplir, siempre con las mismas 5 000 réplicas de duraciones (a cada réplica se le restan las semanas recortadas, sin bajar de la duración tope).

# %%
#| tbl-cap: "Probabilidad de cumplir el plazo según las semanas recortadas"
#| label: tbl-crashingMC
acortadas = {0: '—'}
for r in tabla.index[1:]:
    _, p_r, _, _ = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=tope, costs=ucr, reduction=int(r), report=False)
    acortadas[r] = ' | '.join(', '.join(p_r[j]) for j in sorted(p_r))
tabla.insert(0, 'actividades acortadas (paso a paso)', pd.Series(acortadas))
tabla.round(3)

# %%
#| fig-cap: "Probabilidad de cumplir y coste de reducción según las semanas recortadas"
#| label: fig-probCoste
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(tabla.index, tabla['P(τ ≤ plazo)'], marker='o', label='P(τ ≤ plazo)'); ax.axhline(0.9, color='grey', linestyle=':')
ax.set_xlabel('semanas recortadas'); ax.set_ylabel('probabilidad')
ax2 = ax.twinx(); ax2.plot(tabla.index, tabla['coste'], marker='s', color='tab:red', label='coste (k€)'); ax2.set_ylabel('k€')
ax.legend(loc='upper left'); ax2.legend(loc='lower right'); plt.show()

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
r90 = int(tabla.index[tabla['P(τ ≤ plazo)'] >= 0.9].min())
marginal = (tabla['coste'].diff() / (100 * tabla['P(τ ≤ plazo)'].diff())).rename('k€ por punto de probabilidad')
display(marginal.round(2).to_frame())
Markdown(f'Para el 90 % hay que recortar {r90} semanas ({tabla.loc[r90, "coste"]:.0f} k€) en vez de las {r_det} deterministas '
         f'({tabla.loc[r_det, "coste"]:.0f} k€): la diferencia, {tabla.loc[r90, "coste"] - tabla.loc[r_det, "coste"]:.0f} k€, es lo que cuesta '
         'la seguridad. Los últimos puntos de probabilidad son los más caros.')
