# ---
# title: Recursos 13
# subtitle: "9 actividades; 5 operarios; tres reglas de prioridad frente a Wiest-Levy"
# coleccion: 11_recursos
# variante: reglas-prioridad
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y operarios necesarios se adjunta. Sólo se dispone de 5 operarios y las actividades no pueden interrumpirse una vez empezadas.

# %%
pregunta_1 = "1. Calcule el calendario sin restricción de recursos y dibuje las cargas con los inicios más tempranos. ¿En qué semanas se superan los 5 operarios?  "
pregunta_2 = "2. Programe el proyecto periodo a periodo con un planificador en paralelo: en cada semana se ordenan las actividades disponibles según una regla de prioridad y se van admitiendo mientras quepan en el límite. Aplique tres reglas: (a) menor holgura total, (b) menor duración (SPT) y (c) mayor número de operarios. Dé la duración con cada una.  "
pregunta_3 = "3. Compare con el resultado del algoritmo de Wiest-Levy de la librería (`asignar`).  "
pregunta_4 = "4. ¿Alguna regla garantiza la duración mínima? ¿Qué información usa la regla de la holgura que no usan las otras?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources
    ('A'     , '---'      , 2       , 4        ),
    ('B'     , '---'      , 3       , 1        ),
    ('C'     , '---'      , 5       , 2        ),
    ('D'     , 'A'        , 4       , 1        ),
    ('E'     , 'A'        , 5       , 2        ),
    ('F'     , 'B,C,D'    , 2       , 1        ),
    ('G'     , 'E'        , 3       , 4        ),
    ('H'     , 'F,G'      , 3       , 4        ),
    ('I'     , 'F'        , 3       , 2        ),
], columns=['activity', 'predecessor', 'duration', 'resources']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
limite = 5

# %%
def semanas_txt(lista):
    """«la semana 6», «las semanas 1, 2 y 5» o «ninguna semana»."""
    lista = [str(int(x)) for x in lista]
    if not lista:
        return 'ninguna semana'
    return ('la semana ' if len(lista) == 1 else 'las semanas ') + lista_y(lista)


def explica_cuadrados(cargas):
    """Σ c² agrupando las semanas de igual carga: «6·3² + 4·5² = 54 + 100 = 154»."""
    c = pd.Series(cargas).astype(int)
    cuenta = c.value_counts().sort_index()
    terminos = [f'{n}·{v}²' if n > 1 else f'{v}²' for v, n in cuenta.items()]
    parciales = [f'{n * v * v}' for v, n in cuenta.items()]
    return f'Σ c² = {" + ".join(terminos)} = {" + ".join(parciales)} = **{int((c ** 2).sum())}**'


def explica_nivelacion(cuadrados, calendario):
    """Una línea por actividad recorrida: desplazamientos probados, suma mínima y elección."""
    tabla = getattr(cuadrados, 'data', cuadrados)
    lineas, elegidos = [], {}
    for a, fila in tabla.iterrows():
        v = fila.dropna()
        k = int(v.index.max())
        H = int(calendario.loc[a, 'H_total'])
        minimo = v.min()
        mejor = int(v.idxmin())
        empates = [int(x) for x in v.index[v == minimo]]
        if k == 0:
            probado = 'solo puede probarse 0'
        else:
            probado = f'se prueban 0 a {k}'
        if k < H:
            probado += (f' (no {H}: los desplazamientos ya aplicados a actividades que comparten su holgura '
                        f'le han dejado {plural(k, "semana")})')
        empate = (f'; empatan {lista_y([str(x) for x in empates])} y se toma el menor' if len(empates) > 1 else '')
        lineas.append(f'- {a} (holgura total {H}): {probado}; suma mínima {int(minimo)} con {mejor}{empate} → '
                      + (f'se desplaza **{plural(mejor, "semana")}**' if mejor > 0 else 'no se desplaza'))
        elegidos[a] = mejor
    return '\n'.join(lineas), {a: d for a, d in elegidos.items() if d > 0}


def tabla_asignacion(proyecto, datos, maximo, recurso='resources', duracion='duration'):
    """Semanas en que la demanda supera el límite: orden de prioridad, qué entra y qué se aplaza y por qué."""
    pasos, finales, _, _ = proyecto.copy().asignar_pasos(datos.copy(), duracion, recurso, maximo)
    filas = []
    for ps in pasos:
        if ps['total'] <= ps['limite']:
            continue
        t, L = ps['periodo'], int(ps['limite'])
        orden, aplazadas, acum = [], [], 0
        for a, f in ps['candidatas'].iterrows():
            r = int(f[t])
            orden.append(f'{a} (en curso, {r})' if f['Empezadas'] else f'{a} (H={int(f["H_total"])}, d={int(f[duracion])}, {r})')
            if a in ps['se_programan']:
                acum += r
            else:
                aplazadas.append(f'{a}: {acum} + {r} = {acum + r} > {L}')
        filas.append({'semana': t, 'límite': L, 'demanda': int(ps['total']),
                      'orden de prioridad (holgura, duración, recurso)': ' › '.join(orden),
                      'entran': lista_y(sorted(ps['se_programan'])) if ps['se_programan'] else '—',
                      'se aplazan una semana (no caben)': '; '.join(aplazadas)})
    return pd.DataFrame(filas).set_index('semana') if filas else Markdown('En ninguna semana la demanda supera el límite: no hay que aplazar nada.')


def primera_critica_aplazada(proyecto, datos, maximo, recurso='resources', duracion='duration'):
    """(semana, actividad) de la primera actividad sin holgura que la asignación tiene que aplazar, o None."""
    pasos, _, _, _ = proyecto.copy().asignar_pasos(datos.copy(), duracion, recurso, maximo)
    for ps in pasos:
        for a in ps['candidatas'].index:
            if a in ps['se_retrasan'] and float(ps['candidatas'].loc[a, 'H_total']) == 0:
                return ps['periodo'], a
    return None


def esperas_asignacion(datos_asignados):
    """{actividad: semanas de espera} a partir de las filas 💤 que añade la asignación."""
    e = datos_asignados[datos_asignados.index.str.startswith('💤')]
    return {i[1:]: int(f.duration) for i, f in e.iterrows()}

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con tiempos y camino crítico"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario sin restricción de recursos"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion_minima = proyecto.duration(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %%
#| fig-cap: "Cargas con los inicios más tempranos"
#| label: fig-cargasInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[limite])

# %%
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
exceso = cargas[cargas > limite]
Markdown(f'**Respuesta:** sin límite el proyecto dura {duracion_minima} semanas; con los inicios más tempranos se superan los {limite} operarios en '
         f'{semanas_txt(exceso.index)} (cargas {", ".join(f"{int(c)}" for c in exceso)}).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# El planificador recorre las semanas: una actividad es *disponible* cuando han terminado sus predecesoras; las que ya están en curso siguen (no se interrumpen) y las nuevas entran por orden de prioridad mientras quepan en el límite (si una no cabe, se prueba la siguiente). A igualdad de prioridad decide el orden alfabético.

# %%
def planificar(datos, prelaciones, limite, prioridad, interrumpible=False, max_periodos=200, registro=None):
    '''Programación en paralelo periodo a periodo con un recurso renovable limitado.
    prioridad(a, t) -> número: menor = antes. Devuelve (semanas trabajadas por actividad, inicio, fin).
    Si se pasa una lista en registro, añade una entrada por semana con el orden, las que entran y las que esperan.'''
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
        activas, esperan = [], []
        for a in orden:
            if rec[a] <= disponible:
                activas.append(a); disponible -= rec[a]
            else:
                esperan.append((a, rec[a], limite - disponible))
        if registro is not None:
            registro.append(dict(semana=t, orden=list(orden), activas=list(activas), esperan=esperan,
                                 en_curso=set(en_curso)))
        for a in activas:
            restante[a] -= 1; trabajo[a].append(t)
        en_curso = {a for a in activas if restante[a] > 0}
    inicio = pd.Series({a: (min(s) if s else np.nan) for a, s in trabajo.items()})
    fin = pd.Series({a: (max(s) if s else np.nan) for a, s in trabajo.items()})
    return trabajo, inicio, fin

def tabla_decisiones(registro, clave, limite):
    '''Semanas en que alguna actividad disponible tiene que esperar: orden, las que trabajan y por qué esperan las demás.'''
    filas = []
    for r in registro:
        if not r['esperan']:
            continue
        etiqueta = lambda a: ', '.join(x for x in ('en curso' if a in r['en_curso'] else clave(a), f"{int(datos_proyecto.loc[a, 'resources'])} op.") if x)
        orden = ' › '.join(f'{a} ({etiqueta(a)})' for a in r['orden'])
        filas.append({'semana': r['semana'], 'orden de prioridad': orden,
                      'trabajan': ', '.join(r['activas']) or '—',
                      'esperan (no caben)': '; '.join(f'{a}: {usado} + {q} = {usado + q} > {limite}' for a, q, usado in r['esperan'])})
    return pd.DataFrame(filas).set_index('semana') if filas else Markdown('Ninguna actividad disponible tiene que esperar.')

def gantt_texto(trabajo, limite):
    '''Gantt de pmule: semanas trabajadas por cada actividad (en tramos si se interrumpe) con los operarios
    que usa en cada semana, y la carga total frente al límite.'''
    T = max(max(s) for s in trabajo.values() if s)
    tramos = {}
    for a, semanas in trabajo.items():
        q = int(datos_proyecto.loc[a, 'resources'])
        seg = []
        for t in sorted(semanas):
            if seg and seg[-1][0] + seg[-1][1] == t - 1:
                seg[-1] = (seg[-1][0], seg[-1][1] + 1, q)
            else:
                seg.append((t - 1, 1, q))
        tramos[a] = seg
    carga = [sum(int(datos_proyecto.loc[a, 'resources']) for a, s in trabajo.items() if t in s) for t in range(1, T + 1)]
    extra = pd.DataFrame([carga, ['sí' if c <= limite else 'NO' for c in carga]], index=['Total', r'$\le$ límite'], columns=range(1, T + 1))
    return pinta_tikz(gantt_tramos_tikz(tramos, horizonte=T, extra_rows=extra, params={'segments_text_per_period': True}))

# %%
reglas = {
    '(a) menor holgura total': lambda a, t: (calendario.loc[a, 'H_total'], a),
    '(b) menor duración (SPT)': lambda a, t: (datos_proyecto.loc[a, 'duration'], a),
    '(c) más operarios primero': lambda a, t: (-datos_proyecto.loc[a, 'resources'], a),
}
claves = {
    '(a) menor holgura total': lambda a: f"H={int(calendario.loc[a, 'H_total'])}",
    '(b) menor duración (SPT)': lambda a: f"d={int(datos_proyecto.loc[a, 'duration'])}",
    '(c) más operarios primero': lambda a: '',
}
resultados = {}
for nombre, regla in reglas.items():
    registro = []
    trabajo, inicio, fin = planificar(datos_proyecto, prelaciones, limite, regla, registro=registro)
    resultados[nombre] = dict(trabajo=trabajo, inicio=inicio, fin=fin, duracion=int(fin.max()), registro=registro)
pd.DataFrame({nombre: r['inicio'].astype(int).rename('inicio') for nombre, r in resultados.items()}).assign(**{'sin límite': calendario['inicio_mas_temprano'] + 1}).T

# %% [markdown]
# La tabla anterior da la semana de inicio de cada actividad con cada regla. Para cada regla, primero las semanas en que alguna actividad disponible tiene que esperar (con el orden de prioridad, las que trabajan y la suma que no cabe) y después el Gantt de operarios:

# %%
#| tbl-cap: "Esperas con la regla (a) menor holgura total"
#| label: tbl-decisionesa
tabla_decisiones(resultados['(a) menor holgura total']['registro'], claves['(a) menor holgura total'], limite)

# %%
#| tbl-cap: "Programación con la regla (a) menor holgura total"
#| label: tbl-reglaa
gantt_texto(resultados['(a) menor holgura total']['trabajo'], limite)

# %%
#| tbl-cap: "Esperas con la regla (b) menor duración (SPT)"
#| label: tbl-decisionesb
tabla_decisiones(resultados['(b) menor duración (SPT)']['registro'], claves['(b) menor duración (SPT)'], limite)

# %%
#| tbl-cap: "Programación con la regla (b) menor duración (SPT)"
#| label: tbl-reglab
gantt_texto(resultados['(b) menor duración (SPT)']['trabajo'], limite)

# %%
#| tbl-cap: "Esperas con la regla (c) más operarios primero"
#| label: tbl-decisionesc
tabla_decisiones(resultados['(c) más operarios primero']['registro'], claves['(c) más operarios primero'], limite)

# %%
#| tbl-cap: "Programación con la regla (c) más operarios primero"
#| label: tbl-reglac
gantt_texto(resultados['(c) más operarios primero']['trabajo'], limite)

# %%
duraciones = pd.Series({nombre: r['duracion'] for nombre, r in resultados.items()}, name='duración (semanas)')
duraciones['sin límite de recursos'] = duracion_minima
display(duraciones.to_frame().T)
d = duraciones.drop('sin límite de recursos')
mejores = [k for k, v in d.items() if v == d.min()]
Markdown('**Respuesta:** ' + '; '.join(f'{k}: {v} semanas' for k, v in d.items()) + f' (sin límite, {duracion_minima}). '
         + ('La mejor es la ' if len(mejores) == 1 else 'Las mejores son ') + lista_y(mejores) + f', con {d.min()} semanas.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# La librería (`asignar`) también recorre las semanas, pero parte del calendario sin límite y lo va corrigiendo: solo considera en cada semana las actividades que el calendario vigente sitúa en ella y aplaza una semana las que no caben.
#
# Regla del algoritmo: en cada semana se toman las actividades que el calendario vigente sitúa en ella y se ordenan así: primero las ya empezadas (no se interrumpen); después, menor holgura total (recalculada tras los aplazamientos anteriores) y, a igualdad, menor duración. Se admiten en ese orden mientras la suma de recursos no supere el límite; las que no caben se aplazan una semana (y con ellas sus sucesoras, si no tienen holgura). En la traza, *Total* es la demanda de esa semana antes de aplazar.
#
# Es, por tanto, parecida a la regla (a), pero con la holgura recalculada tras cada aplazamiento y desempate por duración. Traza semana a semana:

# %%
_ = proyecto.copy().asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources', maximo=[limite], report=True)

# %%
#| tbl-cap: "Semanas con conflicto en la asignación de la librería"
#| label: tbl-conflictos
tabla_asignacion(proyecto, datos_proyecto, [limite])

# %%
#| fig-cap: "Asignación de la librería (Wiest-Levy)"
#| label: fig-ganttAsignado
proyecto_asignado = proyecto.copy()
datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',
    maximo=[limite], report=False, params=dict(period_width=1, **GANTT_PARAMS))
pinta_tikz(dibujo)

# %%
duracion_libreria = len(gantt_asignado.data.columns.drop('H_total'))
comparacion = 'igual que' if duracion_libreria == d.min() else ('mejor que' if duracion_libreria < d.min() else 'peor que')
Markdown(f'**Respuesta:** la librería obtiene {duracion_libreria} semanas, {comparacion} la mejor regla ({d.min()}); las reglas dan '
         f'{"; ".join(f"{k}: {v}" for k, v in d.items())}. Wiest-Levy también es una heurística, de modo que ninguna de las cuatro cifras es necesariamente el óptimo.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
da = d['(a) menor holgura total']
nota = ('Aquí la regla de la holgura da el mejor resultado' if da == d.min() else
        f'Aquí, de hecho, la regla de la holgura ({da} semanas) no es la mejor: {lista_y(mejores)} {"da" if len(mejores) == 1 else "dan"} {d.min()} semanas')
Markdown(f'''**Respuesta:** ninguna regla garantiza el óptimo: el problema de programación con recursos limitados es NP-duro y las reglas de prioridad son heurísticas; para proyectos pequeños se puede comprobar por enumeración o programación entera. La regla de la holgura mínima es la única que usa la *estructura* del proyecto (qué actividades tienen margen y cuáles arrastrarían el plazo); SPT y «más operarios» solo miran la actividad aislada. Por eso suele funcionar mejor, aunque no siempre: la holgura se calculó sin recursos y deja de ser exacta en cuanto se aplaza algo. {nota}.''')
