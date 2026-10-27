# ---
# title: CadenaCritica 20
# subtitle: "día 12: 1 colchón de alimentación desbordado; CP consumido 33 %"
# coleccion: 21_cadena_critica
# variante: consumo-alimentacion
# nivel: 2
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
# Un proyecto se ha programado con el método de la cadena crítica (estimaciones agresivas, colchones por cortar y pegar redondeados al día superior, actividades de alimentación lo más tarde posible). La cadena crítica es B – F – G – H – I, de 24 días; el colchón del proyecto es de 6 días y la fecha comprometida, el día 30. El cuadro recoge las estimaciones y el inicio de cada actividad en el plan (días contados desde el instante 0); el segundo cuadro, los colchones de alimentación del plan, cada uno en el arco por el que su cadena de alimentación desemboca en la cadena crítica (o en el final del proyecto).

# %%
#| tbl-cap: "Cuadro de datos del enunciado (días)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, agresiva, segura, inicio_plan
    ('A'     , '---'      , 4       , 7     , 4          ),
    ('B'     , '---'      , 6       , 10    , 0          ),
    ('C'     , '---'      , 2       , 3     , 6          ),
    ('D'     , 'A,C'      , 6       , 10    , 8          ),
    ('E'     , 'C'        , 7       , 10    , 9          ),
    ('F'     , 'B'        , 6       , 8     , 6          ),
    ('G'     , 'F'        , 6       , 9     , 12         ),
    ('H'     , 'D,E,G'    , 2       , 3     , 18         ),
    ('I'     , 'H'        , 4       , 6     , 20         ),
], columns=['activity', 'predecessor', 'agresiva', 'segura', 'inicio_plan']).set_index('activity')
datos_enunciado

# %%
#| tbl-cap: "Colchones de alimentación del plan (días)"
#| label: tbl-colchonesPlan
colchones_plan = pd.DataFrame([
#  colchón, arco   , días
    ('CA1'  , 'D → H', 4   ),
    ('CA2'  , 'E → H', 2   ),
], columns=['colchón', 'arco', 'días']).set_index('colchón')
colchones_plan

# %% [markdown]
# El día 12 se recibe un informe de avance: qué actividades han terminado (y cuándo), cuántos días estima su responsable que le quedan a cada actividad en curso y qué actividades no han empezado. Las actividades de la cadena empiezan en cuanto terminan sus predecesoras y las de alimentación no antes de su inicio en el plan.

# %%
#| tbl-cap: "Informe de avance"
#| label: tbl-informe
informe = dict(dia=12, terminadas={'A': 11, 'B': 7, 'C': 8, 'F': 11}, en_curso={'D': 8, 'E': 5, 'G': 6})
pd.DataFrame({'día 12': {'A': 'terminó el día 11', 'B': 'terminó el día 7', 'C': 'terminó el día 8', 'D': 'en curso, le quedan 8', 'E': 'en curso, le quedan 5', 'F': 'terminó el día 11', 'G': 'en curso, le quedan 6', 'H': '—', 'I': '—'}})

# %%
pregunta_1 = "1. Calcule la fecha prevista de fin de cada actividad que no ha terminado (recorrido hacia delante desde el informe con las duraciones agresivas de las que no han empezado).  "
pregunta_2 = "2. ¿Cuánto ha consumido cada colchón de alimentación, en días y en porcentaje? ¿Alguno queda desbordado? ¿Cuántos días pasan entonces a la cadena crítica?  "
pregunta_3 = "3. Calcule el fin previsto del proyecto y el colchón del proyecto consumido. ¿Qué parte de ese consumo se debe a la propia cadena y qué parte al colchón de alimentación desbordado? Calcule la cadena completada y sitúe el informe en el gráfico de fiebre (zonas 10 → 70 % y 30 → 90 %).  "
pregunta_4 = "4. Un técnico puede reforzar una de las actividades en curso. Con la regla de la gestión por colchones —prioridad a la actividad cuyo colchón (el del proyecto para las de la cadena; el de alimentación al que llega, para las demás) está más consumido, en porcentaje—, ¿a cuál debe ir? ¿Por qué no basta con mirar las holguras del calendario?  "

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


def detalle_prevision(plan, dia, terminadas, en_curso):
    """Recorrido hacia delante desde el informe: fin previsto de cada actividad con sus candidatos.
    La cadena empieza en cuanto terminan sus predecesoras; la alimentación, no antes de su inicio del plan."""
    fin, filas = {}, {}
    for a in plan.orden:
        d = int(plan.agresiva[a])
        if a in terminadas:
            fin[a] = terminadas[a]
            filas[a] = {'estado': 'terminada', 'fin previsto': f'terminó el día {fin[a]}'}
        elif a in en_curso:
            fin[a] = dia + en_curso[a]
            filas[a] = {'estado': 'en curso', 'fin previsto': f'{dia} + {en_curso[a]} (restante) = {fin[a]}'}
        else:
            cands = [f'informe {dia}'] + [f'{p} {fin[p]}' for p in sorted(plan.predecesoras[a])]
            ini = max([dia] + [fin[p] for p in plan.predecesoras[a]])
            if a not in plan.cadena:
                cands.append(f'plan {int(plan.inicios[a])}')
                ini = max(ini, int(plan.inicios[a]))
            fin[a] = ini + d
            filas[a] = {'estado': 'pendiente', 'fin previsto': f"máx({'; '.join(cands)}) + {d} = {ini} + {d} = {fin[a]}"}
        filas[a]['de la cadena'] = 'sí' if a in plan.cadena else ''
    return pd.DataFrame(filas).T[['de la cadena', 'estado', 'fin previsto']], fin


def tabla_consumo_alimentacion(plan, fin):
    """Consumo de cada colchón de alimentación: fin previsto de la actividad que desemboca en él frente al
    inicio del colchón en el plan."""
    filas = {}
    for n, f in plan.alimentacion.iterrows():
        entrada = int(plan.inicios[f['antes_de']]) if f['antes_de'] != 'fin' else int(plan.duracion_cadena)
        ini = entrada - int(f['adoptado'])
        cons = max(0, fin[f['desde']] - ini)
        exceso = max(0, cons - int(f['adoptado']))
        filas[n] = {'arco': f"{f['desde']} → {f['antes_de']}", 'colchón (días)': int(f['adoptado']),
                    'el colchón empieza': f'{entrada} − {int(f["adoptado"])} = {ini}',
                    'consumido = máx(0, fin previsto − inicio del colchón)': f'máx(0, {fin[f["desde"]]} − {ini}) = {cons}',
                    '% consumido': round(100 * cons / int(f['adoptado']), 1) if f['adoptado'] else float('nan'),
                    'pasa a la cadena': exceso}
    return pd.DataFrame(filas).T


def fin_solo_cadena(plan, dia, terminadas, en_curso):
    """Fin previsto de la cadena si las cadenas de alimentación no la retrasaran."""
    fin = {}
    for a in plan.cadena:
        if a in terminadas:
            fin[a] = terminadas[a]
        elif a in en_curso:
            fin[a] = dia + en_curso[a]
        else:
            fin[a] = max([dia] + [fin[p] for p in plan.predecesoras[a] if p in fin]) + int(plan.agresiva[a])
    return fin[plan.cadena[-1]]


def detalle_solo_cadena(plan, dia, terminadas, en_curso):
    """Recorrido de la cadena sola desde el informe (sin los retrasos de la alimentación), con los números."""
    fin, pasos = {}, []
    for a in plan.cadena:
        d = int(plan.agresiva[a])
        if a in terminadas:
            fin[a] = terminadas[a]
        elif a in en_curso:
            fin[a] = dia + en_curso[a]
            pasos.append(f'{a} (en curso) {dia} + {en_curso[a]} = {fin[a]}')
        else:
            prev = [p for p in plan.predecesoras[a] if p in fin]
            ini = max([dia] + [fin[p] for p in prev])
            fin[a] = ini + d
            pasos.append(f'{a} {ini} + {d} = {fin[a]}')
    return '; '.join(pasos)


def colchones_que_protegen(plan, actividad):
    """Colchón que protege a una actividad: el del proyecto si es de la cadena; si no, los colchones de
    alimentación de los arcos por los que llega a la cadena (o al final)."""
    if actividad in plan.cadena:
        return ['CP']
    cadena = set(plan.cadena)
    suc = {x: [y for y in plan.predecesoras if x in plan.predecesoras[y]] for x in plan.predecesoras}
    vistos, pila = set(), [actividad]
    while pila:
        x = pila.pop()
        if x not in vistos:
            vistos.add(x)
            pila += [y for y in suc[x] if y not in cadena]
    return [n for n, f in plan.alimentacion.iterrows() if f['desde'] in vistos]

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las estimaciones agresivas y la cadena crítica"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')
assert plan.cadena == ['B', 'F', 'G', 'H', 'I'] and plan.colchon_proyecto == 6
T, CP = plan.duracion_cadena, plan.colchon_proyecto
tabla_alimentacion(plan)

# %%
#| fig-cap: "Gantt del plan de cadena crítica: cadena en rojo, alimentación en verde, colchones en naranja"
#| label: fig-ganttCCPM
pinta_tikz(plan.gantt_tikz(params=GANTT_CC))

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Fin previsto de cada actividad"
#| label: tbl-prevision
dia, terminadas, en_curso = informe['dia'], informe['terminadas'], informe['en_curso']
prevision, fin_previsto = detalle_prevision(plan, dia, terminadas, en_curso)
prevision

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Un colchón de alimentación empieza en el plan cuando termina la última actividad de su cadena de alimentación (su inicio es el de la actividad de la cadena a la que protege menos su tamaño). Todo lo que esa actividad termine más tarde lo consume; si consume más que su tamaño, el exceso retrasa a la actividad de la cadena crítica y pasa a consumir colchón del proyecto.

# %%
#| tbl-cap: "Consumo de los colchones de alimentación"
#| label: tbl-consumoCA
consumo_ca = tabla_consumo_alimentacion(plan, fin_previsto)
sin_decimales(consumo_ca)

# %%
desbordados = consumo_ca[consumo_ca['pasa a la cadena'] > 0]
Markdown('**Respuesta:** ' + '; '.join(f'{n} ({f["arco"]}) consume {f["consumido = máx(0, fin previsto − inicio del colchón)"].split(" = ")[-1]} de {f["colchón (días)"]} días ({f["% consumido"]:g} %)' for n, f in consumo_ca.iterrows()) + '. '
         + '; '.join(f'{n} ({f["arco"]}) se desborda: consume {f["consumido = máx(0, fin previsto − inicio del colchón)"].split(" = ")[-1]} días de {f["colchón (días)"]} '
                   f'y pasan {plural(int(f["pasa a la cadena"]), "día")} a la cadena crítica' for n, f in desbordados.iterrows()) + '.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
s = plan.seguimiento(dia, terminadas, en_curso)
solo = fin_solo_cadena(plan, dia, terminadas, en_curso)
Markdown(f"Fin previsto del proyecto: **día {s['fin_previsto']:g}** (el mayor de los fines previstos del apartado 1, el de {max(fin_previsto, key=fin_previsto.get)}); colchón consumido "
         f"{s['fin_previsto']:g} − {T} = {plural(s['consumo'], 'día')}, {s['consumo']:g}/{CP} = **{s['pct_consumo']:.1f} %**. Si las cadenas "
         f"de alimentación no la retrasaran (cada actividad de la cadena empezaría al terminar su predecesora en la cadena: "
         f"{detalle_solo_cadena(plan, dia, terminadas, en_curso)}), la cadena terminaría el día {solo}.\n\n"
         f"**Respuesta:** del consumo de {plural(s['consumo'], 'día')}, {solo} − {T} = {plural(solo - T, 'día')} se {'debe' if solo - T == 1 else 'deben'} a la propia cadena y "
         f"{s['fin_previsto']:g} − {solo} = {plural(s['fin_previsto'] - solo, 'día')} al colchón de alimentación desbordado.")

# %%
#| tbl-cap: "Cadena completada, BPI y zona"
#| label: tbl-zona
resumen_informes(plan, [s]).loc[['cadena completada = (T − restante)/T', 'BPI = % colchón restante / % cadena restante', 'límite verde = 10 + 0,6·avance', 'límite rojo = 30 + 0,6·avance', 'zona']]

# %%
#| fig-cap: "Gráfico de fiebre del informe"
#| label: fig-fiebre
pinta_tikz(grafico_fiebre_tikz([s]))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Prioridad de las actividades en curso según el estado de su colchón"
#| label: tbl-prioridad
pct_ca = consumo_ca['% consumido'].astype(float)
filas = {}
for a in sorted(en_curso):
    cs = colchones_que_protegen(plan, a)
    pcts = {c: (s['pct_consumo'] if c == 'CP' else float(pct_ca[c])) for c in cs}
    filas[a] = {'colchón que la protege': ', '.join(cs), '% consumido': round(max(pcts.values()), 1)}
prioridad = pd.DataFrame(filas).T.sort_values('% consumido', ascending=False)
prioridad['prioridad'] = range(1, len(prioridad) + 1)
sin_decimales(prioridad)

# %%
primera = prioridad.index[0]
Markdown(f'**Respuesta:** el técnico debe reforzar **{primera}**, cuyo colchón ({prioridad.loc[primera, "colchón que la protege"]}) está '
         f'consumido en un {prioridad.loc[primera, "% consumido"]:.1f} %: es la que más amenaza la fecha comprometida. '
         'Las holguras del calendario no sirven para decidirlo: en el plan de cadena crítica las actividades de alimentación '
         'están programadas lo más tarde posible y su margen no es una holgura, sino su colchón; y la cadena crítica no tiene '
         'holgura, pero su protección es el colchón del proyecto. El porcentaje de colchón consumido da una medida común para '
         'comparar actividades de cadenas distintas, y la misma señal para todo el equipo.')

# %% [markdown]
# ## Autoevaluación

# %% [markdown]
# Preguntas de respuesta múltiple con los datos de este ejercicio (la correcta, marcada con ✅, es la primera de cada lista). La celda las exporta también en formato GIFT, con las opciones barajadas, a `~/temp/cuestionario_cadena_critica.gift`, para importarlas en Moodle.

# %%
preguntas_test = [
    ('¿Cuántos días pasan a la cadena crítica desde el colchón de alimentación CA1 (D → H)?',
     ['2 días', '6 días', '4 días', '0 días'],
     'Consume 6 días de un colchón de 4: el exceso, 2, retrasa a la actividad de la cadena en la que desemboca.'),
    ('¿Cuántos días del colchón del proyecto se consumen por culpa de las cadenas de alimentación?',
     ['2 días', '0 días', '3 días', '1 días'],
     'Fin previsto 26; si la alimentación no la retrasara, la cadena terminaría el día 24: 2 días se deben a la alimentación.'),
    ('Según la gestión por colchones, ¿qué actividad en curso debe reforzarse primero?',
     ['D', 'E', 'G', 'B'],
     'Es la de colchón más consumido en porcentaje (150,0 %).'),
]
TITULO_TEST = 'Cadena crítica — consumo de los colchones de alimentación (día 12)'
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
