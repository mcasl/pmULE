# ---
# title: CadenaCritica 18
# subtitle: "ruta tradicional de 6 actividades; compromiso 42 frente a 49 días"
# coleccion: 21_cadena_critica
# variante: parkinson-relevos
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
# Un proyecto puede programarse de dos maneras. Para cada actividad, su responsable ha dado una estimación **segura** (con la que cumpliría aproximadamente nueve de cada diez veces, el percentil 90) y una estimación **agresiva pero alcanzable** (la mediana, el percentil 50), ambas en días. Las actividades no comparten recursos.
#
# - **Cronograma tradicional**: estimaciones seguras, cada actividad con su fecha de inicio (la más temprana); el responsable empieza en esa fecha (no antes, aunque pueda) y, por la ley de Parkinson, nunca entrega antes de su estimación segura: si termina antes, no lo comunica.
# - **Cadena crítica**: estimaciones agresivas, colchones por cortar y pegar (redondeados al día superior); las actividades de la cadena empiezan en cuanto terminan sus predecesoras (relevos) y las de alimentación no antes de su inicio del plan.

# %%
pregunta_1 = "1. Calcule la duración del cronograma tradicional y su ruta crítica. ¿Cuántas actividades tiene?  "
pregunta_2 = "2. Obtenga el plan de cadena crítica: cadena, colchones y fecha comprometida.  "
pregunta_3 = "3. Simule 20000 ejecuciones (semilla 6280) en las que las duraciones son independientes y lognormales, con mediana la estimación agresiva y percentil 90 la segura (σ del logaritmo = ln(segura/agresiva)/1,2816; constante si coinciden), en los dos regímenes. ¿Con qué probabilidad se cumple la fecha prometida en cada caso? Compare también las duraciones medias.  "
pregunta_4 = "4. Si cada actividad cumple su estimación segura con probabilidad 0,9 de forma independiente, ¿qué probabilidad hay de que ninguna actividad de la ruta crítica tradicional la supere? Compárela con la simulada y explique por qué el cronograma tradicional, con un margen en cada actividad, protege tan poco la fecha de entrega.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado (días)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, agresiva, segura
    ('A'     , '---'      , 2       , 3     ),
    ('B'     , '---'      , 9       , 12    ),
    ('C'     , '---'      , 3       , 5     ),
    ('D'     , 'A,B'      , 3       , 5     ),
    ('E'     , 'C,D'      , 3       , 4     ),
    ('F'     , 'B,C'      , 9       , 14    ),
    ('G'     , 'E'        , 5       , 8     ),
    ('H'     , 'F,G'      , 6       , 9     ),
    ('I'     , 'H'        , 9       , 11    ),
], columns=['activity', 'predecessor', 'agresiva', 'segura']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la puesta en marcha de una línea de envasado; las actividades son las siguientes (las duraciones del cuadro están en días).

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
    'I': 'Programación de autómatas',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
N, semilla = 20000, 6280

# %%
# Funciones auxiliares de la cadena crítica: cálculos intermedios con los números sustituidos
from math import ceil, sqrt
from pmule import grafico_fiebre_tikz
pd.set_option('display.max_colwidth', None)
GANTT_CC = dict(GANTT_PARAMS, period_width=0.6, activity_inner_text_style=r'\bfseries\small')


def sin_decimales(tabla):
    """Números enteros sin «.0» al mostrar una tabla."""
    return tabla.map(lambda v: f'{v:g}' if isinstance(v, float) else v)


def tabla_margenes(datos, actividades):
    """Margen retirado (segura − agresiva) de cada actividad."""
    t = datos.loc[list(actividades), ['segura', 'agresiva']].copy()
    t['margen = segura − agresiva'] = [f'{s} − {a} = {s - a}' for s, a in zip(t['segura'], t['agresiva'])]
    return t.T


def detalle_colchon(margenes, nombre='colchón'):
    """Colchón por cortar y pegar y por la raíz de la suma de cuadrados (redondeo al día superior)."""
    ms = [int(m) for m in margenes]
    suma, cuad = sum(ms), sum(m * m for m in ms)
    cp, rz = 0.5 * suma, sqrt(cuad)
    return pd.DataFrame({nombre: {
        'cortar y pegar = 0,5·Σ márgenes': f"0,5·({' + '.join(map(str, ms))}) = 0,5·{suma} = {cp:g} → {ceil(cp - 1e-9)}",
        'raíz = √(Σ márgenes²)': f"√({' + '.join(f'{m}²' for m in ms)}) = √{cuad} = {rz:.2f} → {ceil(rz - 1e-9)}"}})


def tabla_alimentacion(plan):
    """Colchones de alimentación: arco en el que desemboca la cadena de alimentación, colchón
    requerido, holgura disponible y colchón adoptado."""
    filas = {}
    for nombre, fila in plan.alimentacion.iterrows():
        ms = [int(x) for x in str(fila['margenes']).split(' + ')]
        if plan.metodo_alimentacion == 'cortar_pegar':
            calc = f"0,5·({' + '.join(map(str, ms))}) = {0.5 * sum(ms):g} → {int(fila['requerido'])}"
        else:
            calc = f"√({' + '.join(f'{m}²' for m in ms)}) = {sqrt(sum(m * m for m in ms)):.2f} → {int(fila['requerido'])}"
        entrada = int(plan.inicio_temprano[fila['antes_de']]) if fila['antes_de'] != 'fin' else int(plan.duracion_cadena)
        fin_p = int(plan.inicio_temprano[fila['desde']] + plan.agresiva[fila['desde']])
        filas[nombre] = {'arco': f"{fila['desde']} → {fila['antes_de']}",
                         'cadena de alimentación': fila['cadena_alimentacion'],
                         'colchón requerido': calc,
                         'holgura disponible = inicio de la cadena − fin más temprano': f"{entrada} − {fin_p} = {int(fila['disponible'])}",
                         'colchón adoptado = mín(requerido, disponible)': int(fila['adoptado'])}
    return pd.DataFrame(filas).T


def detalle_alap(plan):
    """Programación lo más tarde posible de las actividades de alimentación (de la última a la primera)."""
    cadena = set(plan.cadena)
    ca = {(fila['desde'], fila['antes_de']): (n, int(fila['adoptado'])) for n, fila in plan.alimentacion.iterrows()}
    suc = {a: sorted(b for b in plan.predecesoras if a in plan.predecesoras[b]) for a in plan.predecesoras}
    filas = {}
    for a in reversed(plan.orden):
        if a in cadena:
            continue
        cands = []
        for b in suc[a]:
            if b in cadena:
                n, fb = ca[(a, b)]
                cands.append(f'{b} − {n}: {int(plan.inicios[b])} − {fb} = {int(plan.inicios[b]) - fb}')
            else:
                cands.append(f'{b}: {int(plan.inicios[b])}')
        if not suc[a]:
            n, fb = ca[(a, 'fin')]
            cands.append(f'fin − {n}: {plan.duracion_cadena} − {fb} = {plan.duracion_cadena - fb}')
        fin = int(plan.inicios[a] + plan.agresiva[a])
        filas[a] = {'fin = mín(inicio de las sucesoras)': f"mín({'; '.join(cands)}) = {fin}",
                    'inicio = fin − duración': f'{fin} − {int(plan.agresiva[a])} = {int(plan.inicios[a])}',
                    'inicio más temprano': int(plan.inicio_temprano[a])}
    return pd.DataFrame(filas).T


def resumen_informes(plan, informes):
    """Cálculos de cada informe de avance con los números sustituidos."""
    T, CP = plan.duracion_cadena, plan.colchon_proyecto
    columnas = {}
    for inf in informes:
        det = inf['detalle']
        curso = det[det['estado'] == 'en curso']
        pend = det[det['estado'] == 'pendiente']
        partes = [f"{r:g} ({a})" for a, r in curso['restante'].items()] + [f"{int(d)} ({a})" for a, d in pend['agresiva'].items()]
        hecho = float(inf['avance'])
        largas = [f"a {a} le quedan {r:g} días, más que su duración agresiva: se cuentan {int(d)}"
                  for a, r, d in zip(curso.index, curso['restante'], curso['agresiva']) if r > d]
        pa, pc = inf['pct_avance'], inf['pct_consumo']
        columnas[f"día {inf['dia']}"] = {
            'fin previsto = día + restante de la cadena': f"{inf['dia']} + {' + '.join(partes) if partes else '0'} = {inf['fin_previsto']:g}",
            'colchón consumido': f"{inf['fin_previsto']:g} − {T} = {inf['consumo']:g} días → {inf['consumo']:g}/{CP} = {pc:.1f} %",
            'cadena completada': (f"({T} − {T - hecho:g})/{T} = {pa:.1f} %"
                                  + (f" ({'; '.join(largas)})" if largas else '')),
            'BPI = % colchón restante / % cadena restante': (f"(100 − {pc:.1f})/(100 − {pa:.1f}) = {inf['bpi']:.2f}" if pa < 100 else '—'),
            'límite verde = 10 + 0,6·avance': f"10 + 0,6·{pa:.1f} = {inf['limite_verde']:.1f} %",
            'límite rojo = 30 + 0,6·avance': f"30 + 0,6·{pa:.1f} = {inf['limite_roja']:.1f} %",
            'zona': inf['zona']}
    return pd.DataFrame(columnas)

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
datos_proyecto['duration'] = datos_proyecto['segura']

# %%
#| fig-cap: "Grafo Roy con las estimaciones seguras (cronograma tradicional)"
#| label: fig-pert
proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso (avance con los botones):

# %%
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario del cronograma tradicional"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'], grafo="roy")

# %%
ruta_tradicional = [a for a in calendario.sort_values('inicio_mas_temprano').index if calendario.loc[a, 'H_total'] == 0]
T_tradicional = int(calendario['fin_mas_temprano'].max())
k = len(ruta_tradicional)
Markdown(f'El cronograma tradicional dura **{T_tradicional} días**; ruta crítica {" - ".join(ruta_tradicional)} ({k} actividades).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')
cadena, T, CP = plan.cadena, plan.duracion_cadena, plan.colchon_proyecto
display(detalle_colchon(plan.margen[cadena], 'colchón del proyecto').loc[['cortar y pegar = 0,5·Σ márgenes']])
display(tabla_alimentacion(plan))
Markdown(f'Cadena {" - ".join(cadena)} ({T} días), colchón del proyecto {CP} días: compromiso **{plan.compromiso} días**.')

# %%
#| fig-cap: "Gantt del plan de cadena crítica"
#| label: fig-ganttCCPM
pinta_tikz(plan.gantt_tikz(params=GANTT_CC))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# En el régimen tradicional cada actividad empieza en $\max$(su fecha del cronograma, fin de sus predecesoras) y dura $\max$(duración simulada, estimación segura): los adelantos se pierden y los retrasos se transmiten. En el de relevos, cada actividad de la cadena empieza al terminar sus predecesoras y dura lo que dura.

# %%
T_trad = plan.simular(N, semilla=semilla, regimen='tradicional')
T_rel = plan.simular(N, semilla=semilla, regimen='relevos')
pd.DataFrame({'tradicional (Parkinson)': {'fecha prometida': T_tradicional, 'P(cumplir)': float((T_trad <= T_tradicional).mean()),
                                          'duración media': float(T_trad.mean()), 'percentil 90': float(np.percentile(T_trad, 90))},
              'cadena crítica (relevos)': {'fecha prometida': plan.compromiso, 'P(cumplir)': float((T_rel <= plan.compromiso).mean()),
                                           'duración media': float(T_rel.mean()), 'percentil 90': float(np.percentile(T_rel, 90))},
              'relevos con la fecha tradicional': {'fecha prometida': T_tradicional, 'P(cumplir)': float((T_rel <= T_tradicional).mean()),
                                                   'duración media': float(T_rel.mean()), 'percentil 90': float(np.percentile(T_rel, 90))}}).astype(float).round(3).pipe(sin_decimales)

# %%
#| fig-cap: "Duración simulada del proyecto en los dos regímenes"
#| label: fig-regimenes
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.6))
bins = np.linspace(min(T_rel.min(), T_trad.min()), max(T_rel.max(), T_trad.max()), 60)
ax.hist(T_trad, bins=bins, alpha=0.6, density=True, label='tradicional (Parkinson)')
ax.hist(T_rel, bins=bins, alpha=0.6, density=True, label='cadena crítica (relevos)')
ax.axvline(T_tradicional, color='tab:blue', linestyle='--'); ax.axvline(plan.compromiso, color='tab:orange', linestyle='--')
ax.set_xlabel('duración (días)'); ax.set_ylabel('densidad'); ax.legend(); plt.show()

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
Markdown(f'''La probabilidad de que ninguna de las {k} actividades de la ruta crítica supere su estimación segura es 
0,9^{k} = **{0.9 ** k:.3f}**. Con la ley de Parkinson y las fechas fijas, cualquier exceso de una actividad crítica retrasa el proyecto y ningún adelanto lo compensa, así que la probabilidad de cumplir no puede ser mayor que esa (la simulada es {float((T_trad <= T_tradicional).mean()):.3f}: también fallan las no críticas que se pasan de su holgura).

Cada estimación segura protege su actividad, no la fecha de entrega: el margen se pierde porque los adelantos no se transmiten (ley de Parkinson), porque quien tiene holgura empieza tarde (síndrome del estudiante) y porque en las convergencias manda el camino que llega más tarde. La cadena crítica retira esos márgenes y los concentra en colchones: promete {T_tradicional - plan.compromiso} días antes y cumple con probabilidad {float((T_rel <= plan.compromiso).mean()):.3f}. La última columna de la tabla muestra que el problema no estaba en las estimaciones: con las mismas duraciones, ejecutando por relevos, la fecha tradicional se cumpliría con probabilidad {float((T_rel <= T_tradicional).mean()):.3f}.''')

# %% [markdown]
# ## Autoevaluación

# %% [markdown]
# Preguntas de respuesta múltiple con los datos de este ejercicio (la correcta, marcada con ✅, es la primera de cada lista). La celda las exporta también en formato GIFT, con las opciones barajadas, a `~/temp/cuestionario_cadena_critica.gift`, para importarlas en Moodle.

# %%
preguntas_test = [
    ('Si cada una de las 6 actividades de la ruta crítica tradicional cumple su estimación segura con probabilidad 0,9, de forma independiente, ¿qué probabilidad hay de que ninguna la supere?',
     ['0,531', '0,900', '0,400', '0,590'],
     '0,9^6 = 0,531.'),
    ('En la simulación del régimen tradicional con la ley de Parkinson, ¿con qué probabilidad se cumple la fecha prometida?',
     ['0,52', '0,85', '0,90', '1,00'],
     '0,520: como mucho 0,9^6, porque cualquier exceso en la ruta crítica se transmite y ningún adelanto lo compensa.'),
    ('¿Qué describe la ley de Parkinson?',
     ['el trabajo se alarga hasta ocupar el tiempo disponible: quien termina antes no lo comunica', 'con holgura, las actividades se empiezan en el último momento', 'en las uniones de caminos manda el que llega más tarde', 'repartir a una persona entre varias tareas las retrasa todas'],
     'Las otras opciones son el síndrome del estudiante, el sesgo de convergencia y la multitarea.'),
]
TITULO_TEST = 'Cadena crítica — Parkinson frente a relevos (ruta de 6 actividades)'
import os, re


def a_gift(preguntas_test, titulo, semilla=1):
    """Preguntas en formato GIFT (Moodle): caracteres especiales escapados y opciones barajadas."""
    esc = lambda s: re.sub(r'([~=#{}:])', r'\\\1', str(s))
    r = np.random.default_rng(semilla)
    lineas = [f'$CATEGORY: {titulo}', '']
    for i, (q, ops, fb) in enumerate(preguntas_test, 1):
        cuerpo = ' '.join(('=' if j == 0 else '~') + esc(ops[j]) for j in r.permutation(len(ops)))
        lineas.append(f'::{esc(titulo)} P{i}::{esc(q)} {{{cuerpo} ####{esc(fb)}}}')
        lineas.append('')
    return '\n'.join(lineas)


gift = a_gift(preguntas_test, TITULO_TEST)
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_cadena_critica.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)
