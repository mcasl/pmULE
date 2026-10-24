# ---
# title: CadenaCritica 07
# subtitle: "cadena de 6 actividades, CP de 11 días; informes en zona verde → verde → amarilla"
# coleccion: 21_cadena_critica
# variante: seguimiento-fiebre
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
# Un proyecto se ha programado con el método de la cadena crítica (estimaciones agresivas, colchones por cortar y pegar redondeados al día superior, actividades de alimentación lo más tarde posible). La cadena crítica es A – C – E – F – G – I, de 38 días; el colchón del proyecto es de 11 días y la fecha comprometida, el día 49. El cuadro recoge las estimaciones y el inicio de cada actividad en el plan (días contados desde el inicio del proyecto, que es el instante 0).

# %%
#| tbl-cap: "Cuadro de datos del enunciado (días)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, agresiva, segura, inicio_plan
    ('A'     , '---'      , 8       , 12    , 0          ),
    ('B'     , '---'      , 7       , 9     , 13         ),
    ('C'     , 'A'        , 4       , 7     , 8          ),
    ('D'     , 'A'        , 9       , 13    , 19         ),
    ('E'     , 'C'        , 9       , 14    , 12         ),
    ('F'     , 'B,E'      , 7       , 11    , 21         ),
    ('G'     , 'F'        , 5       , 7     , 28         ),
    ('H'     , 'D,F'      , 3       , 5     , 28         ),
    ('I'     , 'G,H'      , 5       , 8     , 33         ),
], columns=['activity', 'predecessor', 'agresiva', 'segura', 'inicio_plan']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la rehabilitación de una nave industrial para un taller; las actividades son las siguientes (las duraciones del cuadro están en días).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Levantamiento topográfico y mediciones',
    'B': 'Proyecto de ejecución y licencia',
    'C': 'Desmontaje de instalaciones antiguas',
    'D': 'Demolición de tabiquería interior',
    'E': 'Refuerzo de la estructura metálica',
    'F': 'Reparación de la cubierta',
    'G': 'Solera y canalizaciones',
    'H': 'Cerramientos y carpintería exterior',
    'I': 'Instalación eléctrica',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# Durante la ejecución se reciben tres informes de avance. En cada uno se indica qué actividades han terminado (y en qué día), cuántos días estima el responsable que le quedan a cada actividad en curso, y qué actividades no han empezado. Las actividades de la cadena empiezan en cuanto terminan sus predecesoras (corredor de relevos).

# %%
#| tbl-cap: "Informes de avance"
#| label: tbl-informes
informes = [
    dict(dia=14, terminadas={'A': 6}, en_curso={'B': 7, 'C': 1}),
    dict(dia=27, terminadas={'A': 6, 'B': 21, 'C': 15, 'D': 26, 'E': 24}, en_curso={'F': 5}),
    dict(dia=39, terminadas={'A': 6, 'B': 21, 'C': 15, 'D': 26, 'E': 24, 'F': 32, 'G': 36, 'H': 38}, en_curso={'I': 6}),
]
tabla_informes = pd.DataFrame({f"día {i['dia']}": {a: (f"terminó el día {i['terminadas'][a]}" if a in i['terminadas'] else (f"en curso, le quedan {i['en_curso'][a]}" if a in i['en_curso'] else '—')) for a in datos_enunciado.index} for i in informes})
tabla_informes

# %%
pregunta_1 = "1. Para cada informe, calcule la fecha prevista de fin de la cadena y el colchón consumido, en días y en porcentaje del colchón del proyecto.  "
pregunta_2 = "2. Calcule el porcentaje de cadena crítica completada en cada informe (duración de la cadena menos su duración restante, entre la duración de la cadena; de una actividad en curso no se cuenta como restante más que su duración agresiva) y el índice de protección del colchón, BPI (porcentaje de colchón restante entre porcentaje de cadena restante).  "
pregunta_3 = "3. Sitúe los informes en el gráfico de fiebre (zona verde por debajo de la recta que va del 10 % de consumo con la cadena sin empezar al 70 % con la cadena terminada; zona roja por encima de la recta que va del 30 % al 90 %; amarilla entre ambas) e indique la zona de cada uno.  "
pregunta_4 = "4. ¿Qué debe hacer el jefe de proyecto tras cada informe? ¿Por qué se pide a los responsables la duración restante y no se controla si cada actividad cumple su fecha de fin?  "

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

# %%
#| fig-cap: "Grafo PERT con las estimaciones agresivas y la cadena crítica"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos paso a paso (avance con los botones):

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')
assert plan.cadena == ['A', 'C', 'E', 'F', 'G', 'I'] and plan.colchon_proyecto == 11
T, CP = plan.duracion_cadena, plan.colchon_proyecto
Markdown(f'Plan: cadena {" - ".join(plan.cadena)} ({T} días), colchón del proyecto {CP} días, compromiso el día {plan.compromiso}.')

# %%
#| fig-cap: "Gantt del plan de cadena crítica"
#| label: fig-ganttCCPM
pinta_tikz(plan.gantt_tikz(params=GANTT_CC))

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# La fecha prevista de fin de la cadena es la del informe más lo que le queda a la cadena: la duración restante estimada de la actividad en curso más las duraciones agresivas de las que no han empezado. Todo lo que pase de la duración de la cadena es colchón consumido.

# %%
seguimientos = [plan.seguimiento(i['dia'], i['terminadas'], i['en_curso']) for i in informes]
for s in seguimientos:
    display(Markdown(f"**Día {s['dia']}** — estado de la cadena crítica:"))
    display(sin_decimales(s['detalle'][['estado', 'agresiva', 'completado', 'restante', 'fin previsto']]))
resumen = resumen_informes(plan, seguimientos)

# %%
#| tbl-cap: "Fecha prevista de fin y colchón consumido"
#| label: tbl-consumo
resumen.loc[['fin previsto = día + restante de la cadena', 'colchón consumido']]

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Lo completado de la cadena es su duración menos lo que le queda: cuenta entera cada actividad terminada (por su duración agresiva, aunque haya tardado más o menos) y, de la que está en curso, su duración agresiva menos lo que le queda (cero si le queda más que eso).

# %%
#| tbl-cap: "Cadena completada y BPI"
#| label: tbl-bpi
resumen.loc[['cadena completada', 'BPI = % colchón restante / % cadena restante']]

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Límites de las zonas para el avance de cada informe y zona"
#| label: tbl-zonas
resumen.loc[['límite verde = 10 + 0,6·avance', 'límite rojo = 30 + 0,6·avance', 'zona']]

# %%
#| fig-cap: "Gráfico de fiebre con los tres informes"
#| label: fig-fiebre
pinta_tikz(grafico_fiebre_tikz(seguimientos))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
acciones = {'verde': 'no hace falta actuar: el colchón se consume a un ritmo menor que el avance de la cadena',
            'amarilla': 'preparar un plan de recuperación (refuerzos, solapes, reasignar a quien termine), sin ejecutarlo todavía',
            'roja': 'ejecutar el plan de recuperación: si no, la fecha comprometida está en peligro'}
Markdown('\n'.join(f"- Día {s['dia']} (zona {s['zona']}, BPI {s['bpi']:.2f}): {acciones[s['zona']]}." for s in seguimientos))

# %% [markdown]
# Con estimaciones agresivas, la mitad de las actividades terminarán después de su duración prevista: controlar el cumplimiento de cada fecha de fin generaría alarmas continuas y presiones para volver a inflar las estimaciones. Lo que importa es si queda colchón suficiente para lo que queda de cadena, y para eso hace falta la **duración restante** de lo que está en curso: con ella se calcula el fin previsto, el consumo del colchón y el BPI, una única señal que todo el equipo entiende.

# %% [markdown]
# ## Autoevaluación

# %% [markdown]
# Preguntas de respuesta múltiple con los datos de este ejercicio (la correcta, marcada con ✅, es la primera de cada lista). La celda las exporta también en formato GIFT, con las opciones barajadas, a `~/temp/cuestionario_cadena_critica.gift`, para importarlas en Moodle.

# %%
preguntas_test = [
    ('En el informe del día 14, ¿cuál es la fecha prevista de fin de la cadena?',
     ['día 41', 'día 38', 'día 49', 'día 44'],
     'Fecha del informe más la duración restante de la actividad en curso más las duraciones agresivas de las pendientes: día 41.'),
    ('En el informe del día 39, ¿qué porcentaje de la cadena crítica está completado?',
     ['86,8 %', '84,2 %', '102,6 %', '63,6 %'],
     '(duración de la cadena − duración restante)/duración = 86,8 %; de una actividad en curso no se cuenta como restante más que su duración agresiva.'),
    ('¿Cuál es el BPI en el informe del día 39?',
     ['2,76', '0,73', '0,36', '2,81'],
     'BPI = % de colchón restante / % de cadena restante = (100 − 63,6)/(100 − 86,8).'),
    ('¿En qué zona del gráfico de fiebre está el informe del día 39?',
     ['amarilla', 'verde', 'roja', 'no se puede saber sin el valor ganado'],
     'Límite verde 10 + 0,6·86,8 = 62,1 %; límite rojo 30 + 0,6·86,8 = 82,1 %; consumo 63,6 %.'),
]
TITULO_TEST = 'Cadena crítica — seguimiento (cadena de 38 días, CP de 11)'
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
