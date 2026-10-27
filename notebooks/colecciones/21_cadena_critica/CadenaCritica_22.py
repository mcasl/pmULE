# ---
# title: CadenaCritica 22
# subtitle: "día 8: SPI 1.06 y zona roja (alimentación adelantada, cadena retrasada)"
# coleccion: 21_cadena_critica
# variante: bpi-frente-a-spi
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
# Un proyecto se ha programado con el método de la cadena crítica (estimaciones agresivas, colchones por cortar y pegar redondeados al día superior, actividades de alimentación lo más tarde posible). La cadena crítica es A – D – E – G – H, de 29 días; el colchón del proyecto es de 7 días y la fecha comprometida, el día 36. El cuadro recoge las estimaciones, el inicio de cada actividad en el plan (días desde el instante 0) y su presupuesto (miles de euros). La oficina de proyectos sigue el proyecto con el valor ganado, tomando como línea base el plan de cadena crítica (el presupuesto de cada actividad se reparte uniformemente a lo largo de su duración agresiva, desde su inicio en el plan) y midiendo el avance físico de una actividad en curso como días trabajados / (días trabajados + días restantes).

# %%
#| tbl-cap: "Cuadro de datos del enunciado (días)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, agresiva, segura, presupuesto, inicio_plan
    ('A'     , '---'      , 9       , 14    , 27         , 0          ),
    ('B'     , '---'      , 4       , 6     , 12         , 4          ),
    ('C'     , '---'      , 8       , 10    , 16         , 9          ),
    ('D'     , 'A,B'      , 6       , 9     , 6          , 9          ),
    ('E'     , 'D'        , 3       , 5     , 9          , 15         ),
    ('F'     , 'B,C'      , 5       , 7     , 10         , 17         ),
    ('G'     , 'E'        , 6       , 8     , 6          , 18         ),
    ('H'     , 'F,G'      , 5       , 6     , 15         , 24         ),
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
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# El día 8 se recibe el informe de avance del cuadro. Las actividades de la cadena empiezan en cuanto terminan sus predecesoras; para prever las que no han empezado se usan las duraciones agresivas y, para las de alimentación, no antes de su inicio en el plan.

# %%
#| tbl-cap: "Informe de avance"
#| label: tbl-informe
informe = dict(dia=8, terminadas={'B': 5}, trabajados={'A': 8, 'C': 8}, en_curso={'A': 6, 'C': 4})
pd.DataFrame({'día 8': {'A': 'lleva 8 días, le quedan 6', 'B': 'terminó el día 5', 'C': 'lleva 8 días, le quedan 4', 'D': '—', 'E': '—', 'F': '—', 'G': '—', 'H': '—'}})

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
        acts = str(fila['cadena_alimentacion']).split(' - ')
        if len(acts) == len(ms):
            marg = '; '.join(f"{a}: {int(plan.segura[a])} − {int(plan.agresiva[a])} = {m}" for a, m in zip(acts, ms))
        else:
            marg = ' + '.join(map(str, ms))
        if plan.metodo_alimentacion == 'cortar_pegar':
            calc = f"0,5·({' + '.join(map(str, ms))}) = {0.5 * sum(ms):g} → {int(fila['requerido'])}"
        else:
            calc = f"√({' + '.join(f'{m}²' for m in ms)}) = {sqrt(sum(m * m for m in ms)):.2f} → {int(fila['requerido'])}"
        entrada = int(plan.inicio_temprano[fila['antes_de']]) if fila['antes_de'] != 'fin' else int(plan.duracion_cadena)
        fin_p = int(plan.inicio_temprano[fila['desde']] + plan.agresiva[fila['desde']])
        filas[nombre] = {'arco': f"{fila['desde']} → {fila['antes_de']}",
                         'cadena de alimentación': fila['cadena_alimentacion'],
                         'márgenes = segura − agresiva': marg,
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
        partes_c = ([f"{min(r, d):g} ({a})" for a, r, d in zip(curso.index, curso['restante'], curso['agresiva'])]
                    + [f"{int(d)} ({a})" for a, d in pend['agresiva'].items()])
        hecho = float(inf['avance'])
        largas = [f"a {a} le quedan {r:g} días, más que su duración agresiva: se cuentan {int(d)}"
                  for a, r, d in zip(curso.index, curso['restante'], curso['agresiva']) if r > d]
        pa, pc = inf['pct_avance'], inf['pct_consumo']
        columnas[f"día {inf['dia']}"] = {
            'fin previsto = día + restante de la cadena': f"{inf['dia']} + {' + '.join(partes) if partes else '0'} = {inf['fin_previsto']:g}",
            'colchón consumido': f"{inf['fin_previsto']:g} − {T} = {inf['consumo']:g} días → {inf['consumo']:g}/{CP} = {pc:.1f} %",
            'cadena completada = (T − restante)/T': ((f"({T} − ({' + '.join(partes_c)}))/{T} = ({T} − {T - hecho:g})/{T} = {pa:.1f} %"
                                                      if len(partes_c) > 1 else
                                                      f"({T} − {partes_c[0]})/{T} = {pa:.1f} %" if partes_c else f"{T}/{T} = {pa:.1f} %")
                                                     + (f" ({'; '.join(largas)})" if largas else '')),
            'BPI = % colchón restante / % cadena restante': (f"(100 − {pc:.1f})/(100 − {pa:.1f}) = {inf['bpi']:.2f}" if pa < 100 else '—'),
            'límite verde = 10 + 0,6·avance': f"10 + 0,6·{pa:.1f} = {inf['limite_verde']:.1f} %",
            'límite rojo = 30 + 0,6·avance': f"30 + 0,6·{pa:.1f} = {inf['limite_roja']:.1f} %",
            'zona': zona_txt(pc, inf['limite_verde'], inf['limite_roja'], inf['zona'])}
    return pd.DataFrame(columnas)


def zona_txt(pc, lv, lr, zona):
    """Comparación del consumo con los dos límites que decide la zona."""
    if zona == 'verde':
        return f'{pc:.1f} % < {lv:.1f} % (límite verde) → verde'
    nota = ' (justo en la frontera: cuenta como la zona superior)' if min(abs(pc - lv), abs(pc - lr)) < 1e-6 else ''
    if zona == 'roja':
        return f'{pc:.1f} % ≥ {lr:.1f} % (límite rojo) → roja' + nota
    return f'{lv:.1f} % ≤ {pc:.1f} % < {lr:.1f} % → amarilla' + nota


def tabla_rutas(proyecto, R, valores):
    """Rutas con sus actividades en orden de ejecución y su duración como suma término a término."""
    filas, cal = {}, proyecto.calendar(valores)
    for i, r in enumerate(R.index, 1):
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        total = int(sum(int(valores[a]) for a in acts))
        filas[f'ruta {i}'] = {'actividades': ' - '.join(acts),
                              'duración = suma': ' + '.join(f'{a}({int(valores[a])})' for a in acts) + f' = {total}'}
    return pd.DataFrame(filas).T


def margenes_txt(plan, acts):
    """«4 + 2 + 3 = 9» con los márgenes de las actividades dadas."""
    ms = [int(plan.margen[a]) for a in acts]
    return f"{' + '.join(map(str, ms))} = {sum(ms)}" if len(ms) > 1 else f'{sum(ms)}'


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

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las estimaciones agresivas y la cadena crítica"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')
assert plan.cadena == ['A', 'D', 'E', 'G', 'H'] and plan.colchon_proyecto == 7
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

# %% [markdown]
# Con la línea base del plan de cadena crítica, el **valor planificado** (PV) de una actividad el día del informe es su presupuesto por la fracción de su duración agresiva que el plan prevé hecha ese día: días previstos = mín(máx(día − inicio en el plan, 0), duración). El **valor ganado** (EV) es su presupuesto por su avance físico: 100 % si ha terminado, días trabajados/(trabajados + restantes) si está en curso y 0 % si no ha empezado. PV y EV del proyecto son las sumas de los de las actividades.

# %%
#| tbl-cap: "Valor planificado y valor ganado de cada actividad (miles de euros)"
#| label: tbl-pvev
tabla_vg = detalle_pv_ev(plan, datos_proyecto, dia, terminadas, trabajados, en_curso)
tabla_vg.drop(columns=['PV_i', 'EV_i'])

# %%
PV, EV, SPI = indices(tabla_vg)
suma = lambda col: ' + '.join(f'{v:.2f} ({a})' for a, v in tabla_vg[col].items() if v > 0) or '0'
Markdown(f'PV = {suma("PV_i")} = {PV:.2f}; EV = {suma("EV_i")} = {EV:.2f} miles de euros (sólo se suman las actividades con valor no nulo).\n\n'
         f'**Respuesta:** SV = EV − PV = {EV:.2f} − {PV:.2f} = {EV - PV:+.2f} miles de euros; **SPI = {EV:.2f}/{PV:.2f} = {SPI:.3f}**. '
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
Markdown(f"**Respuesta:** según la gestión por colchones el proyecto está en zona **{s['zona']}**: se ha consumido el {s['pct_consumo']:.1f} % del "
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
def suma_parte(f, col):
    t = tabla_vg if f is None else tabla_vg[f]
    v = [f'{x:.2f} ({a})' for a, x in t[col].items() if x > 0]
    return ' + '.join(v) + (f' = {float(t[col].sum()):.2f}' if len(v) > 1 else '') if v else '0.00'
pd.DataFrame({k: {'PV': suma_parte(partes[k], 'PV_i'), 'EV': suma_parte(partes[k], 'EV_i'), 'SPI = EV/PV': f'{ev:.2f}/{pv:.2f} = {spi:.3f}'}
              for k, (pv, ev, spi) in spis.items()})

# %%
Markdown(f"**Respuesta:** el SPI global ({spis['todo el proyecto'][2]:.3f}) mezcla dos cosas: la alimentación va por delante del plan "
         f"(SPI {spis['alimentación'][2]:.3f}): sus actividades se están haciendo antes de lo que prevé el plan, que las programa "
         "lo más tarde posible; la "
         f"cadena crítica va por detrás (SPI {spis['cadena crítica'][2]:.3f}). El trabajo adelantado en la alimentación suma valor "
         "ganado, pero no acerca la fecha de entrega, que sólo depende de la cadena y de sus colchones; el retraso de la cadena, "
         "en cambio, consume colchón del proyecto día a día.")

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
     ['1,06', '0,64', '1,89', '0,95'],
     'SPI = EV/PV = 38,10/36,00 = 1,058.'),
    ('¿En qué zona del gráfico de fiebre está el informe?',
     ['roja', 'verde', 'amarilla', 'no se puede saber sin el valor ganado'],
     'Consumo 71,4 % con un 10,3 % de cadena completada.'),
    ('¿Por qué discrepan el SPI y la gestión por colchones en este informe?',
     ['el trabajo adelantado en la alimentación compensa en el SPI el retraso de la cadena crítica', 'la cadena crítica va adelantada', 'el colchón del proyecto está mal dimensionado', 'el SPI no incluye las actividades de la cadena crítica'],
     'El SPI suma en dinero todas las actividades; la fecha sólo depende de la cadena y de sus colchones.'),
    ('¿Qué indicador debe guiar las decisiones para proteger la fecha comprometida?',
     ['el BPI y el gráfico de fiebre', 'el SPI de todo el proyecto', 'el SPI de las actividades de alimentación', 'el CPI del valor ganado'],
     'El BPI mira sólo lo que fija la entrega; el valor ganado sigue siendo útil para controlar el coste.'),
]
TITULO_TEST = 'Cadena crítica — BPI frente a SPI (día 8)'
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
