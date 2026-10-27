# ---
# title: MonteCarlo 18
# subtitle: "9 actividades triangulares; límite 3 operarios, plazo 23"
# coleccion: 09_montecarlo
# variante: recursos-limitados
# nivel: 3
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
# Considere el proyecto cuyo cuadro de prelaciones, duración de cada actividad (mínima *a*, más probable *m* y máxima *b*, en semanas, distribución triangular) y operarios necesarios se adjunta. Cada actividad necesita un número fijo de operarios durante toda su duración y la empresa dispone como máximo de 3 operarios por semana. El contrato exige terminar en **23 semanas**.

# %%
pregunta_1 = "1. Calcule con las duraciones más probables la duración del proyecto sin límite de recursos y con el límite de operarios (algoritmo de asignación).  "
pregunta_2 = "2. Simule 2000 réplicas: en cada una sortee las duraciones, calcule la duración sin recursos y **replanifique** con el límite. Compare media, desviación típica y percentil 90 de ambas distribuciones.  "
pregunta_3 = "3. Estime la probabilidad de cumplir el plazo de 23 semanas ignorando los recursos y respetándolos. ¿Qué conclusión sacaría un director que solo mirase el camino crítico?  "
pregunta_4 = "4. Calcule el índice de criticidad de cada actividad sin y con recursos, y la frecuencia con que el límite de operarios retrasa alguna actividad. ¿Qué actividades pasan a mandar al introducir el límite?  "
pregunta_5 = "5. ¿Cómo cambia la probabilidad de cumplir si se amplía el límite en un operario? Decida si merece la pena.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b , operarios
    ('A'     , '---'      , 1, 3, 5 , 1        ),
    ('B'     , '---'      , 2, 4, 6 , 2        ),
    ('C'     , '---'      , 5, 6, 9 , 2        ),
    ('D'     , 'A'        , 6, 8, 10, 3        ),
    ('E'     , 'A'        , 3, 4, 6 , 3        ),
    ('F'     , 'B,E'      , 2, 4, 6 , 3        ),
    ('G'     , 'C,E'      , 5, 7, 8 , 2        ),
    ('H'     , 'D,F'      , 3, 4, 6 , 3        ),
    ('I'     , 'G,H'      , 3, 5, 8 , 3        ),
], columns=['activity', 'predecessor', 'a', 'm', 'b', 'operarios']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
limite = [3]
plazo = 23
datos_proyecto['duration'] = datos_proyecto['m']
demanda = datos_proyecto[['predecessor', 'duration', 'operarios']]
dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con las duraciones más probables y el camino crítico (sin límite)"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario sin límite de recursos"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion_libre = proyecto.duration(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %%
#| fig-cap: "Carga de operarios con los inicios más tempranos frente al límite"
#| label: fig-cargasInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(demanda.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite)

# %% [markdown]
# El algoritmo de asignación recorre las semanas: cada actividad disponible (predecesoras terminadas) empieza, por orden de prioridad (menor inicio más tardío), si caben sus operarios durante toda su duración; si no, espera. Traza semana a semana y Gantt resultante (filas 💤: esperas):

# %%
res0 = proyecto.copy().asignar_multi(demanda.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios y las duraciones más probables"
#| label: fig-ganttAsignado
pinta_tikz(res0.dibujos['operarios'])

# %%
#| tbl-cap: "Inicio de cada actividad sin y con límite"
#| label: tbl-inicios
pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res0.inicios, 'fin con límite': res0.fines,
              'espera (semanas)': res0.esperas}).fillna(0).astype(int).T

# %%
Markdown(f'**Respuesta:** con las duraciones más probables el proyecto dura {plural(duracion_libre, "semana")} sin límite de recursos '
         f'y {plural(res0.duracion, "semana")} con {limite[0]} operarios (las esperas de la tabla anterior lo retrasan {res0.duracion - duracion_libre}).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# En cada réplica se sortean las duraciones (triangulares, redondeadas a semanas enteras), se calcula la duración sin recursos (camino más largo) y se vuelve a aplicar el algoritmo de asignación con el límite: dos duraciones por réplica. `monte_carlo_recursos` hace las 2000 réplicas y guarda ambas. Los percentiles son los de las réplicas ordenadas (el P90 deja por debajo el 90 % de ellas).

# %%
#| tbl-cap: "Distribución de la duración con y sin límite de recursos"
#| label: tbl-mcRecursos
mc = proyecto.copy().monte_carlo_recursos(dists, demanda[['operarios']].copy(), ['operarios'], limite, n_samples=2000, seed=11)
resumen_mc = mc.summary().rename(columns={'q05': 'P5', 'q50': 'P50', 'q95': 'P95'})
resumen_mc.insert(4, 'P90', [np.quantile(mc.tau_libre_samples, 0.9), np.quantile(mc.tau_recursos_samples, 0.9), np.quantile(mc.retraso_samples, 0.9)])
resumen_mc.round(2)

# %%
#| fig-cap: "Distribución simulada de la duración sin y con el límite de operarios"
#| label: fig-histograma
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.arange(min(mc.tau_libre_samples.min(), mc.tau_recursos_samples.min()) - 0.5, max(mc.tau_libre_samples.max(), mc.tau_recursos_samples.max()) + 1.5, 1)
ax.hist(mc.tau_libre_samples, bins=bins, alpha=0.5, label='sin límite')
ax.hist(mc.tau_recursos_samples, bins=bins, alpha=0.5, label=f'con {limite[0]} operarios')
ax.axvline(plazo, color='red', linestyle='--', label=f'plazo ({plazo})')
ax.set_xlabel('duración del proyecto (semanas)'); ax.set_ylabel('réplicas'); ax.legend(); plt.show()

# %%
rl = resumen_mc
Markdown(f'''**Respuesta:** la media sube de {rl.loc['sin recursos', 'media']:.2f} a {rl.loc['con recursos', 'media']:.2f} semanas, '''
         f'''la desviación típica pasa de {rl.loc['sin recursos', 'desv. típica']:.2f} a {rl.loc['con recursos', 'desv. típica']:.2f} y el percentil 90 de '''
         f'''{rl.loc['sin recursos', 'P90']:.0f} a {rl.loc['con recursos', 'P90']:.0f} semanas. El límite añade esperas (de media '''
         f'''{rl.loc['retraso por recursos', 'media']:.2f} semanas) que dependen de qué actividades coinciden en cada réplica.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
p_libre = 1 - mc.prob_superar(plazo, con_recursos=False)
p_res = 1 - mc.prob_superar(plazo)
n_libre = int((mc.tau_libre_samples <= plazo).sum()); n_res = int((mc.tau_recursos_samples <= plazo).sum())
Markdown(f'''Sin recursos, {n_libre} de las 2000 réplicas terminan en {plazo} semanas o menos: P ≈ {n_libre}/2000 = {p_libre:.3f}. '''
         f'''Con el límite, {n_res} de 2000: P ≈ {p_res:.3f}.\n\n'''
         f'''**Respuesta:** P(T ≤ {plazo}) = {p_libre:.3f} ignorando los recursos y {p_res:.3f} respetándolos. Un director que sólo mirase '''
         f'''el camino crítico creería que cumple el plazo {p_libre:.0%} de las veces, cuando con {limite[0]} operarios sólo lo cumple el '''
         f'''{p_res:.1%}: el riesgo real de incumplir es {1 - p_res:.1%} y no {1 - p_libre:.1%}.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# El índice de criticidad de una actividad es el porcentaje de réplicas en que determina la duración: sin recursos, si está en la ruta más larga; con recursos, si forma parte de la cadena de actividades y esperas por el recurso que fija el final del proyecto.

# %%
#| tbl-cap: "Criticidad sin y con recursos"
#| label: tbl-criticidadRecursos
criticidad = (mc.criticidad * 100).round(1)
criticidad['cambio'] = criticidad.iloc[:, -1] - criticidad.iloc[:, 0]
criticidad.sort_values(criticidad.columns[1], ascending=False)

# %%
#| tbl-cap: "Frecuencia con que cada recurso bloquea alguna actividad"
#| label: tbl-bloqueos
(mc.bloqueos * 100).round(1).to_frame('% de réplicas')

# %%
pasan = criticidad[criticidad['cambio'] >= 20].sort_values('cambio', ascending=False)
bajan = criticidad[criticidad['cambio'] <= -20].sort_values('cambio')
fmt = lambda t: lista_y([f"{a} ({t.loc[a].iloc[0]:.1f} % → {t.loc[a].iloc[1]:.1f} %)" for a in t.index])
Markdown(f'''**Respuesta:** el límite de {limite[0]} operarios retrasa alguna actividad en el {float(mc.bloqueos.iloc[0]) * 100:.1f} % de las réplicas. '''
         + (f'''Pasan a mandar (su índice sube 20 puntos o más) {fmt(pasan)}: en el CPM no siempre estaban en la ruta más larga, pero con el límite forman parte de la cadena de esperas por el recurso que fija el final del proyecto. ''' if len(pasan) else
            'Ninguna actividad gana más de 20 puntos de criticidad. ')
         + (f'''En cambio pierden peso {fmt(bajan)}.''' if len(bajan) else ''))

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
mc2 = proyecto.copy().monte_carlo_recursos(dists, demanda[['operarios']].copy(), ['operarios'], [limite[0] + 1], n_samples=2000, seed=11)
p_res2 = 1 - mc2.prob_superar(plazo)
if p_res2 >= 0.8:
    decision = (f'Con un operario más el plazo pasa a ser razonablemente seguro ({p_res2:.1%}): merece la pena si su coste es menor que '
                'lo que se espera ahorrar en penalizaciones.')
elif p_res2 - p_res >= 0.2:
    decision = (f'Un operario más mejora mucho la probabilidad (+{(p_res2 - p_res) * 100:.1f} puntos), aunque el plazo sigue sin ser seguro '
                f'({p_res2:.1%}): merece la pena si su coste es menor que las penalizaciones evitadas, y conviene combinarlo con otras medidas.')
else:
    decision = (f'Un operario más apenas mejora la probabilidad (+{(p_res2 - p_res) * 100:.1f} puntos, hasta {p_res2:.1%}): por sí solo no resuelve '
                'el problema y sólo merecería la pena si es muy barato; hay que buscar otras medidas (más recursos, acortar actividades críticas o '
                'renegociar el plazo).')
Markdown(f'''Con {limite[0] + 1} operarios P(T ≤ {plazo}) pasa de {p_res:.3f} a {p_res2:.3f}; la media baja de '''
         f'''{mc.tau_recursos_samples.mean():.2f} a {mc2.tau_recursos_samples.mean():.2f} semanas.\n\n**Respuesta:** {decision}''')
