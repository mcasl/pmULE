# ---
# title: CadenaCritica 26
# subtitle: "día 12: SPI 0.94 y zona verde (alimentación retrasada dentro de su colchón)"
# coleccion: 21_cadena_critica
# variante: bpi-frente-a-spi
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
# Un proyecto se ha programado con el método de la cadena crítica (estimaciones agresivas, colchones por cortar y pegar redondeados al día superior, actividades de alimentación lo más tarde posible). La cadena crítica es C – D – E – G – H – I, de 43 días; el colchón del proyecto es de 12 días y la fecha comprometida, el día 55. El cuadro recoge las estimaciones, el inicio de cada actividad en el plan (días desde el instante 0) y su presupuesto (miles de euros). La oficina de proyectos sigue el proyecto con el valor ganado, tomando como línea base el plan de cadena crítica (el presupuesto de cada actividad se reparte uniformemente a lo largo de su duración agresiva, desde su inicio en el plan) y midiendo el avance físico de una actividad en curso como días trabajados / (días trabajados + días restantes).

# %%
#| tbl-cap: "Cuadro de datos del enunciado (días)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, agresiva, segura, presupuesto, inicio_plan
    ('A'     , '---'      , 4       , 5     , 12         , 12         ),
    ('B'     , '---'      , 5       , 7     , 5          , 6          ),
    ('C'     , '---'      , 8       , 12    , 32         , 0          ),
    ('D'     , 'C'        , 9       , 15    , 36         , 8          ),
    ('E'     , 'A,B,D'    , 6       , 8     , 6          , 17         ),
    ('F'     , 'B'        , 9       , 13    , 36         , 11         ),
    ('G'     , 'E,F'      , 5       , 8     , 20         , 23         ),
    ('H'     , 'G'        , 6       , 10    , 6          , 28         ),
    ('I'     , 'H'        , 9       , 13    , 27         , 34         ),
], columns=['activity', 'predecessor', 'agresiva', 'segura', 'presupuesto', 'inicio_plan']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde al lanzamiento de una aplicación móvil; las actividades son las siguientes (las duraciones del cuadro están en días).

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
    'I': 'Pruebas de integración',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# El día 12 se recibe el informe de avance del cuadro. Las actividades de la cadena empiezan en cuanto terminan sus predecesoras; para prever las que no han empezado se usan las duraciones agresivas y, para las de alimentación, no antes de su inicio en el plan.

# %%
#| tbl-cap: "Informe de avance"
#| label: tbl-informe
informe = dict(dia=12, terminadas={'B': 12, 'C': 6}, trabajados={'D': 6}, en_curso={'D': 7})
pd.DataFrame({'día 12': {'A': '—', 'B': 'terminó el día 12', 'C': 'terminó el día 6', 'D': 'lleva 6 días, le quedan 7', 'E': '—', 'F': '—', 'G': '—', 'H': '—', 'I': '—'}})

# %%
pregunta_1 = "1. Calcule el valor planificado (PV) y el valor ganado (EV) en el día del informe, la variación del cronograma (SV = EV − PV) y el índice de desempeño del cronograma (SPI = EV/PV). ¿Qué concluiría un informe basado sólo en el valor ganado?  "
pregunta_2 = "2. Calcule el fin previsto de la cadena, el colchón del proyecto consumido, la cadena completada, el BPI y la zona del gráfico de fiebre (zonas 10 → 70 % y 30 → 90 %). ¿Qué concluye la gestión por colchones?  "
pregunta_3 = "3. Calcule el SPI de las actividades de la cadena crítica y el de las actividades de alimentación por separado y explique por qué discrepan los dos diagnósticos.  "
pregunta_4 = "4. ¿Qué indicador debe guiar las decisiones del jefe de proyecto para proteger la fecha comprometida? ¿Qué limitaciones tiene aquí el SPI?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
datos_proyecto['duration'] = datos_proyecto['agresiva']

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


def detalle_pv_ev(plan, datos, dia, terminadas, trabajados, en_curso):
    """PV con el plan de cadena crítica como línea base (reparto uniforme en la duración agresiva desde el inicio
    del plan) y EV con el avance físico (días trabajados / (trabajados + restantes) para las que están en curso)."""
    filas = {}
    for a in sorted(plan.agresiva.index):
        bac, ini, d = float(datos.loc[a, 'presupuesto']), int(plan.inicios[a]), int(plan.agresiva[a])
        hecho_plan = min(max(dia - ini, 0), d)
        pv = bac * hecho_plan / d
        if a in terminadas:
            pct, txt = 1.0, '100 %'
        elif a in en_curso:
            w, r = trabajados[a], en_curso[a]
            pct, txt = w / (w + r), f'{w}/({w} + {r}) = {100 * w / (w + r):.1f} %'
        else:
            pct, txt = 0.0, '0 %'
        filas[a] = {'cadena': 'sí' if a in plan.cadena else '', 'presupuesto': f'{bac:g}',
                    'plan': f'[{ini}, {ini + d})', 'PV = presupuesto·días previstos/duración': f'{bac:g}·{hecho_plan}/{d} = {pv:.2f}',
                    'avance físico': txt, 'EV = presupuesto·avance': f'{bac:g}·{pct:.3f} = {bac * pct:.2f}',
                    'PV_i': pv, 'EV_i': bac * pct}
    return pd.DataFrame(filas).T


def indices(tabla, filtro=None):
    t = tabla if filtro is None else tabla[filtro]
    pv, ev = float(t['PV_i'].sum()), float(t['EV_i'].sum())
    return pv, ev, (ev / pv if pv else float('nan'))

# %%
#| fig-cap: "Grafo Roy con las estimaciones agresivas y la cadena crítica"
#| label: fig-pert
proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso (avance con los botones):

# %%
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')
assert plan.cadena == ['C', 'D', 'E', 'G', 'H', 'I'] and plan.colchon_proyecto == 12
T, CP = plan.duracion_cadena, plan.colchon_proyecto
dia, terminadas, trabajados, en_curso = informe['dia'], informe['terminadas'], informe['trabajados'], informe['en_curso']

# %%
#| fig-cap: "Gantt del plan de cadena crítica (línea base del valor ganado)"
#| label: fig-ganttCCPM
pinta_tikz(plan.gantt_tikz(params=GANTT_CC))

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Valor planificado y valor ganado de cada actividad (miles de euros)"
#| label: tbl-pvev
tabla_vg = detalle_pv_ev(plan, datos_proyecto, dia, terminadas, trabajados, en_curso)
tabla_vg.drop(columns=['PV_i', 'EV_i'])

# %%
PV, EV, SPI = indices(tabla_vg)
Markdown(f'PV = {PV:.2f}; EV = {EV:.2f}; SV = EV − PV = {EV - PV:+.2f} miles de euros; **SPI = {EV:.2f}/{PV:.2f} = {SPI:.3f}**. '
         + ('Según el valor ganado el proyecto va adelantado: se ha hecho más trabajo del previsto a estas alturas.' if SPI >= 1 else
            f'Según el valor ganado el proyecto va retrasado: se ha hecho sólo el {100 * SPI:.1f} % del trabajo previsto a estas alturas.'))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
s = plan.seguimiento(dia, terminadas, en_curso)
display(sin_decimales(s['detalle'][['estado', 'agresiva', 'completado', 'restante', 'fin previsto']]))
resumen = resumen_informes(plan, [s])
resumen

# %%
#| fig-cap: "Gráfico de fiebre del informe"
#| label: fig-fiebre
pinta_tikz(grafico_fiebre_tikz([s]))

# %%
Markdown(f"Según la gestión por colchones el proyecto está en zona **{s['zona']}**: se ha consumido el {s['pct_consumo']:.1f} % del "
         f"colchón con el {s['pct_avance']:.1f} % de la cadena completada (BPI {s['bpi']:.2f}). "
         + ('La fecha comprometida está amenazada.' if s['zona'] != 'verde' else 'La fecha comprometida no corre peligro.'))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "SPI de la cadena crítica y de la alimentación"
#| label: tbl-spiPartes
partes = {'cadena crítica': tabla_vg['cadena'] == 'sí', 'alimentación': tabla_vg['cadena'] != 'sí', 'todo el proyecto': None}
spis = {k: indices(tabla_vg, f) for k, f in partes.items()}
pd.DataFrame({k: {'PV': round(pv, 2), 'EV': round(ev, 2), 'SPI = EV/PV': f'{ev:.2f}/{pv:.2f} = {spi:.3f}'} for k, (pv, ev, spi) in spis.items()})

# %%
Markdown(f"El SPI global ({spis['todo el proyecto'][2]:.3f}) mezcla dos cosas: la cadena crítica va al ritmo del plan "
         f"(SPI {spis['cadena crítica'][2]:.3f}) y la alimentación va retrasada (SPI {spis['alimentación'][2]:.3f}), pero su "
         "retraso cabe en sus colchones de alimentación: por eso no llega a la cadena ni consume colchón del proyecto. El valor "
         "ganado pesa igual un día de retraso en cualquier actividad; la gestión por colchones sólo cuenta el que amenaza la entrega.")

# %%
#| tbl-cap: "Consumo de los colchones de alimentación"
#| label: tbl-consumoCA
ca = s['alimentacion'].set_index('colchon')[['desde', 'antes_de', 'fin_previsto', 'adoptado', 'consumido']]
ca.index.name = None
sin_decimales(ca.rename(columns={'antes_de': 'antes de', 'fin_previsto': 'fin previsto', 'adoptado': 'colchón (días)'}).astype(object))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Para proteger la fecha comprometida manda la gestión por colchones: el BPI y el gráfico de fiebre miran sólo lo que fija la entrega (la cadena crítica, y la alimentación cuando desborda su colchón) y comparan lo que queda de colchón con lo que queda de cadena. La Guía del PMBOK presenta el BPI precisamente como una señal pensada para proteger la fecha de entrega y para decidir las prioridades del trabajo diario, a diferencia del SPI.
#
# Limitaciones del SPI en este caso: (1) suma en dinero todas las actividades, críticas o no, de modo que el trabajo hecho fuera de la cadena compensa el retraso de la cadena (o el retraso de la alimentación dentro de su colchón parece un problema); (2) depende de la línea base: con las actividades de alimentación lo más tarde posible, empezarlas antes eleva el EV sin cambiar la fecha de fin; (3) tiende a 1 al final del proyecto aunque se termine tarde, porque al terminar EV = PV = presupuesto. El valor ganado sigue siendo útil para controlar el coste (CPI) y el gasto, pero el plazo se gobierna con los colchones.

# %% [markdown]
# ## Autoevaluación

# %% [markdown]
# Preguntas de respuesta múltiple con los datos de este ejercicio (la correcta, marcada con ✅, es la primera de cada lista). La celda las exporta también en formato GIFT, con las opciones barajadas, a `~/temp/cuestionario_cadena_critica.gift`, para importarlas en Moodle.

# %%
preguntas_test = [
    ('¿Cuánto vale el SPI de todo el proyecto en el día del informe?',
     ['0,94', '1,01', '0,56', '1,06'],
     'SPI = EV/PV = 53,62/57,00 = 0,941.'),
    ('¿En qué zona del gráfico de fiebre está el informe?',
     ['verde', 'amarilla', 'roja', 'no se puede saber sin el valor ganado'],
     'Consumo 16,7 % con un 23,3 % de cadena completada.'),
    ('¿Por qué discrepan el SPI y la gestión por colchones en este informe?',
     ['la alimentación va retrasada, pero su retraso cabe en sus colchones de alimentación', 'la cadena crítica va retrasada', 'el colchón del proyecto ya está agotado', 'el SPI sólo mide la cadena crítica'],
     'Un retraso de la alimentación que no desborda su colchón no llega a la cadena ni consume colchón del proyecto.'),
    ('¿Qué indicador debe guiar las decisiones para proteger la fecha comprometida?',
     ['el BPI y el gráfico de fiebre', 'el SPI de todo el proyecto', 'el SPI de las actividades de alimentación', 'el CPI del valor ganado'],
     'El BPI mira sólo lo que fija la entrega; el valor ganado sigue siendo útil para controlar el coste.'),
]
TITULO_TEST = 'Cadena crítica — BPI frente a SPI (día 12)'
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
