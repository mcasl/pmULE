# ---
# title: Recursos 07
# subtitle: "8 actividades; 4 operarios; interrumpir o no interrumpir"
# coleccion: 11_recursos
# variante: interrumpibles
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y operarios necesarios se adjunta, con 4 operarios disponibles. Se discute si merece la pena organizar el trabajo de modo que una actividad pueda interrumpirse y reanudarse más tarde (sin coste añadido) cuando los operarios hagan falta en otra más urgente.

# %%
pregunta_1 = "1. Programe el proyecto con un planificador en paralelo y la regla «menor holgura total primero» suponiendo que las actividades **no** pueden interrumpirse. ¿Cuánto dura?  "
pregunta_2 = "2. Repita suponiendo que sí pueden interrumpirse: en cada semana se vuelven a ordenar *todas* las actividades disponibles, también las empezadas, y una actividad en curso puede quedar parada si entra otra más prioritaria. ¿Cuánto dura? ¿Qué actividades se interrumpen y cuántas veces?  "
pregunta_3 = "3. Compare las dos programaciones semana a semana. ¿De dónde sale la ganancia (si la hay)?  "
pregunta_4 = "4. ¿Qué costes ocultos tiene interrumpir una actividad en la práctica? ¿Cuándo compensa?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources
    ('A'     , '---'      , 6       , 4        ),
    ('B'     , '---'      , 5       , 1        ),
    ('C'     , 'A,B'      , 3       , 4        ),
    ('D'     , 'C'        , 6       , 2        ),
    ('E'     , 'B'        , 4       , 1        ),
    ('F'     , 'C'        , 3       , 1        ),
    ('G'     , 'E,F'      , 2       , 4        ),
    ('H'     , 'D,G'      , 3       , 4        ),
], columns=['activity', 'predecessor', 'duration', 'resources']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion_minima = proyecto.duration(datos_proyecto['duration'])
limite = 4

# %%
def planificar(datos, prelaciones, limite, prioridad, interrumpible=False, max_periodos=200):
    '''Programación en paralelo periodo a periodo con un recurso renovable limitado.
    prioridad(a, t) -> número: menor = antes. Devuelve (semanas trabajadas por actividad, inicio, fin).'''
    pred = prelaciones.immediate_predecessor
    restante = datos['duration'].astype(int).to_dict()
    rec = datos['resources'].astype(int).to_dict()
    trabajo = {a: [] for a in restante}
    en_curso = set()
    for t in range(1, max_periodos + 1):
        terminadas = {a for a, r in restante.items() if r == 0}
        if len(terminadas) == len(restante):
            break
        elegibles = [a for a in restante if restante[a] > 0 and set(pred[a]) <= terminadas]
        if not interrumpible:
            # las ya empezadas siguen; las nuevas se ordenan por prioridad
            fijas = [a for a in elegibles if a in en_curso]
            nuevas = sorted([a for a in elegibles if a not in en_curso], key=lambda a: prioridad(a, t))
            orden = fijas + nuevas
        else:
            orden = sorted(elegibles, key=lambda a: prioridad(a, t))
        disponible = limite
        activas = []
        for a in orden:
            if rec[a] <= disponible:
                activas.append(a); disponible -= rec[a]
        for a in activas:
            restante[a] -= 1; trabajo[a].append(t)
        en_curso = {a for a in activas if restante[a] > 0}
    inicio = pd.Series({a: (min(s) if s else np.nan) for a, s in trabajo.items()})
    fin = pd.Series({a: (max(s) if s else np.nan) for a, s in trabajo.items()})
    return trabajo, inicio, fin

def gantt_texto(trabajo, limite):
    '''Tabla actividad × semana con el recurso usado y fila de carga total.'''
    T = max(max(s) for s in trabajo.values() if s)
    g = pd.DataFrame('', index=list(trabajo), columns=range(1, T + 1), dtype=object)   # object: admite enteros y texto
    for a, semanas in trabajo.items():
        for t in semanas:
            g.loc[a, t] = int(datos_proyecto.loc[a, 'resources'])
    carga = pd.Series({t: sum(int(datos_proyecto.loc[a, 'resources']) for a, s in trabajo.items() if t in s) for t in g.columns})
    g.loc['Total'] = carga
    g.loc['≤ límite'] = ['sí' if c <= limite else 'NO' for c in carga]
    return g

# %%
regla = lambda a, t: (calendario.loc[a, 'H_total'], datos_proyecto.loc[a, 'duration'], a)

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT con tiempos y camino crítico (sin límite de operarios)"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos paso a paso (avance con los botones):

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario sin límite de recursos (la regla de prioridad usa la holgura total)"
#| label: tbl-calendario
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]

# %%
#| fig-cap: "Cargas con los inicios más tempranos frente al límite"
#| label: fig-cargasInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[limite])

# %% [markdown]
# El planificador recorre las semanas: una actividad es *disponible* cuando han terminado sus predecesoras; las disponibles se ordenan por la regla (menor holgura total, luego menor duración) y entran mientras quepan en el límite. Sin interrupciones, las ya empezadas van primero.

# %%
#| tbl-cap: "Programación sin interrupciones"
#| label: tbl-sinInterrupcion
trabajo_no, inicio_no, fin_no = planificar(datos_proyecto, prelaciones, limite, regla, interrumpible=False)
gantt_texto(trabajo_no, limite)

# %%
Markdown(f'Sin interrupciones el proyecto dura {int(fin_no.max())} semanas (mínimo sin límite de recursos: {duracion_minima}).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Programación con interrupciones permitidas"
#| label: tbl-conInterrupcion
trabajo_si, inicio_si, fin_si = planificar(datos_proyecto, prelaciones, limite, regla, interrumpible=True)
gantt_texto(trabajo_si, limite)

# %%
def interrupciones(semanas):
    return sum(1 for s1, s2 in zip(semanas, semanas[1:]) if s2 > s1 + 1)
inter = pd.Series({a: interrupciones(s) for a, s in trabajo_si.items()})
inter = inter[inter > 0]
Markdown(f'''Con interrupciones el proyecto dura {int(fin_si.max())} semanas. {('Se interrumpen ' + ', '.join(f'{a} ({v} vez/veces)' for a, v in inter.items()) + '.') if len(inter) else 'Ninguna actividad llega a interrumpirse.'}''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Inicio y fin de cada actividad en las dos programaciones"
#| label: tbl-comparacion
pd.DataFrame({'inicio (no interr.)': inicio_no.astype(int), 'fin (no interr.)': fin_no.astype(int),
              'inicio (interr.)': inicio_si.astype(int), 'fin (interr.)': fin_si.astype(int),
              'semanas trabajadas (interr.)': {a: len(s) for a, s in trabajo_si.items()}})

# %% [markdown]
# Si hay ganancia, viene de que al permitir interrumpir no hay que esperar a que termine una actividad poco urgente para liberar operarios: una actividad crítica puede entrar en cuanto está disponible. Si no la hay, es que la regla de prioridad ya había evitado bloquear el recurso con actividades holgadas. Interrumpir nunca empeora la duración con este planificador (el no interrumpible es un caso particular), pero puede fragmentar mucho el trabajo.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Interrumpir tiene costes que el modelo ignora: volver a montar y desmontar (tiempos de preparación), pérdida de ritmo y de aprendizaje, el material en curso inmovilizado, la coordinación de los operarios y la dificultad de seguimiento. Compensa cuando el ahorro de plazo vale más que esos costes: actividades largas, modulares (pueden partirse en lotes) y sin costes de preparación apreciables; no compensa en tareas con fraguados, curados o procesos continuos.
