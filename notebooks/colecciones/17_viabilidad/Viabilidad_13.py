# ---
# title: Viabilidad 13
# subtitle: "límite de 3 operarios, ingresos de 65 k€/mes durante 36 meses"
# coleccion: 17_viabilidad
# variante: recursos-van
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
# Una empresa va a ejecutar el proyecto de inversión cuyo cuadro de prelaciones, duración (meses), operarios necesarios y presupuesto (BAC, k€) de cada actividad se adjunta. El coste de cada actividad se desembolsa uniformemente a lo largo de su duración. Cuando el proyecto termina, la instalación genera un flujo de caja neto de 65 k€ al mes durante 36 meses. La empresa descuenta al 12% anual y dispone como máximo de **3 operarios** al mes. Contratar un operario más cuesta **0.5 k€ por mes** mientras dure el proyecto.

# %%
pregunta_1 = "1. Calcule la duración y el VAN del proyecto sin tener en cuenta el límite de operarios.  "
pregunta_2 = "2. Aplique el algoritmo de asignación con el límite de operarios. ¿Cuánto se alarga el proyecto y cuánto cuesta en VAN? Separe el efecto sobre el VAN de los costes y el de los ingresos y explique cuál domina.  "
pregunta_3 = "3. ¿Compensa contratar un operario más (límite 4)? Construya la tabla de duración y VAN al ampliar el límite en 0, 1, 2 y 3 operarios y compare la mejora del VAN con el coste de contratar (sin descontar, por simplicidad).  "
pregunta_4 = "4. Calcule el flujo mensual mínimo (umbral del VAN = 0) con y sin el límite. ¿Cuánto sube el umbral por culpa del límite?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, operarios, BAC
    ('A'     , '---'      , 2       , 1        , 200),
    ('B'     , '---'      , 3       , 1        , 150),
    ('C'     , '---'      , 1       , 1        , 175),
    ('D'     , 'A,B,C'    , 1       , 2        , 175),
    ('E'     , 'D'        , 6       , 1        , 125),
    ('F'     , 'E'        , 6       , 2        , 125),
    ('G'     , 'D'        , 2       , 1        , 175),
    ('H'     , 'G'        , 3       , 3        , 175),
    ('I'     , 'F,H'      , 5       , 1        , 100),
], columns=['activity', 'predecessor', 'duration', 'operarios', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
def van(flujos, k):
    '''Valor actual neto de una lista de flujos (el primero en t = 0) a la tasa k.'''
    return float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))

def tir(flujos):
    '''Tasa que anula el VAN (por bisección entre −99 % y 1000 %).'''
    from scipy.optimize import brentq
    return float(brentq(lambda k: van(flujos, k), -0.99, 10.0))

def tabla_descuento(flujos, k):
    '''Año, flujo, factor de descuento, valor actual y valores acumulados (simple y descontado).'''
    t = pd.DataFrame({'flujo': list(flujos)}, index=pd.Index(range(len(flujos)), name='año'))
    t['(1 + k)^-t'] = [(1 + k) ** -i for i in t.index]
    t['valor actual'] = t['flujo'] * t['(1 + k)^-t']
    t['acumulado'] = t['flujo'].cumsum()
    t['acumulado descontado'] = t['valor actual'].cumsum()
    return t

def payback(flujos, k=None):
    '''Periodo de recuperación (años, con interpolación lineal dentro del año); descontado si se da k.'''
    acum = 0.0
    for t, f in enumerate(flujos):
        v = f if k is None else f / (1 + k) ** t
        if t > 0 and acum < 0 <= acum + v:
            return float(t - 1 + (-acum) / v)
        acum += v
    return float('inf')

def anualidad_equivalente(VAN, k, n):
    '''Anualidad constante durante n años cuyo valor actual es VAN.'''
    return float(VAN * k / (1 - (1 + k) ** -n))

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
F, N, k, L, coste_operario = 65.0, 36, 0.12, 3, 0.5
BAC = datos_proyecto['BAC']

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con el camino crítico (sin límite de operarios)"
#| label: fig-grafo
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario sin límite de recursos (meses)"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion_libre = int(proyecto.duration(datos_proyecto['duration']))
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# El coste mensual de cada actividad es su BAC entre su duración; la fila *Total* del Gantt es el desembolso de cada mes. Cada mes es un periodo: en los meses 1…T el flujo es −desembolso y en los T+1…T+N, +F; la tasa mensual equivalente cumple $(1 + k_m)^{12} = 1 + k$.

# %%
#| fig-cap: "Gantt de inicios más tempranos con el coste mensual de cada actividad (k€)"
#| label: fig-ganttCostes
datos_gantt = datos_proyecto.copy()
datos_gantt['coste_mensual'] = datos_gantt['BAC'] / datos_gantt['duration']
gantt_costes, dibujo = proyecto.gantt(datos_gantt, 'duration', 'coste_mensual', total='ambas', acumulado=True, tikz=True,
                                      params=dict(inner_text_digits=1, **GANTT_PARAMS))
pinta_tikz(dibujo)

# %%
def desembolsos(inicios, duraciones):
    '''Desembolso de cada mes repartiendo el BAC de cada actividad uniformemente desde su inicio (contado desde 0).'''
    T = int(max(inicios[a] + duraciones[a] for a in datos_proyecto.index))
    pagos = pd.Series(0.0, index=range(1, T + 1))
    for a, fila in datos_proyecto.iterrows():
        es, d = int(inicios[a]), int(duraciones[a])
        for m in range(es + 1, es + d + 1):
            pagos[m] += fila['BAC'] / d
    return pagos
def flujo_mensual(pagos):
    return [0.0] + [-float(x) for x in pagos] + [F] * N
k_mensual = (1 + k) ** (1 / 12) - 1
pagos_libre = desembolsos(calendario['inicio_mas_temprano'], datos_proyecto['duration'])
flujos_libre = flujo_mensual(pagos_libre)
VAN_libre = van(flujos_libre, k_mensual)
display(pd.DataFrame({'desembolso (k€)': pagos_libre}).T.round(1))
Markdown(f'k mensual = (1 + {k})^(1/12) − 1 = {k_mensual:.4%}. Sin límite: {duracion_libre} meses de inversión ({pagos_libre.sum():.0f} k€ = Σ BAC) y '
         f'VAN = {VAN_libre:.1f} k€ (valor actual de los ingresos {van([0.0] * (duracion_libre + 1) + [F] * N, k_mensual):.1f} menos el de los desembolsos '
         f'{-van([0.0] + [-float(x) for x in pagos_libre], k_mensual):.1f}).')

# %%
#| tbl-cap: "VAN sin y con límite de operarios (van_recursos)"
#| label: tbl-escenarios
res = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, F, N, k, ampliaciones=3)
escenarios = res['escenarios'].round(2)
escenarios

# %%
Markdown(f'La librería da lo mismo: sin límite {escenarios.loc["sin límite", "duración"]:.0f} meses y VAN {escenarios.loc["sin límite", "VAN"]:.1f} k€ '
         f'(VAN de los ingresos {escenarios.loc["sin límite", "VAN ingresos"]:.1f}, de los costes {escenarios.loc["sin límite", "VAN costes"]:.1f}).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| fig-cap: "Carga de operarios con los inicios más tempranos frente al límite"
#| label: fig-cargasInicial
gantt_op, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)
plot_cargas(gantt_df=gantt_op, limite_maximo_recursos=[L])

# %% [markdown]
# El algoritmo de asignación recorre los meses; cada actividad disponible (predecesoras terminadas) empieza, por orden de prioridad (menor inicio más tardío), si caben sus operarios durante toda su duración; si no, espera. Traza mes a mes y Gantt resultante (filas 💤: esperas):

# %%
res_lim = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', ['operarios'], [L], report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios"
#| label: fig-ganttLimite
pinta_tikz(res_lim.dibujos['operarios'])

# %%
#| tbl-cap: "Inicio de cada actividad sin y con límite"
#| label: tbl-iniciosLimite
pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res_lim.inicios.reindex(datos_proyecto.index),
              'espera (meses)': res_lim.esperas.reindex(datos_proyecto.index).fillna(0).astype(int)}).T

# %%
#| tbl-cap: "Desembolso mensual sin y con límite (k€)"
#| label: tbl-desembolsosLimite
pagos_lim = desembolsos(res_lim.inicios, datos_proyecto['duration'])
flujos_lim = flujo_mensual(pagos_lim)
VAN_lim = van(flujos_lim, k_mensual)
pd.DataFrame({'sin límite': pagos_libre, 'con límite': pagos_lim}).fillna(0).T.round(1)

# %%
dif = escenarios.loc['con límite'] - escenarios.loc['sin límite']
Markdown(f'Con {L} operarios el proyecto dura {res_lim.duracion} meses (+{res_lim.duracion - duracion_libre}): los ingresos empiezan en el mes {res_lim.duracion + 1} en vez del {duracion_libre + 1}. '
         f'VAN = {VAN_lim:.1f} k€ (la librería: {escenarios.loc["con límite", "VAN"]:.1f}) frente a {VAN_libre:.1f} sin límite: {dif["VAN"]:.1f} k€. '
         f'Efecto en los ingresos: {dif["VAN ingresos"]:.1f} k€ (se cobran más tarde); en los costes: {-dif["VAN costes"]:.1f} k€ '
         '(las actividades que esperan se pagan más tarde y el coste se descuenta más). El retraso de los ingresos domina.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Duración y VAN al ampliar el límite de operarios"
#| label: tbl-ampliaciones
amp = res['ampliaciones'].copy()
amp['coste de contratar'] = amp.index * coste_operario * amp['duración']
amp['VAN neto de contratar'] = amp['VAN'] - amp['coste de contratar']
amp.round(2)

# %%
mejor = amp['VAN neto de contratar'].idxmax()
Markdown(f'La mejor opción es ampliar el límite en {mejor} operario(s): VAN neto {amp.loc[mejor, "VAN neto de contratar"]:.1f} k€ '
         f'frente a {amp.loc[0, "VAN neto de contratar"]:.1f} k€ sin ampliar. Se compara la mejora del VAN por adelantar los ingresos '
         'con lo que cuesta mantener a los operarios todo el proyecto.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
e0 = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, 0.0, N, k, ampliaciones=0)['escenarios']['VAN']
e1 = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, 1.0, N, k, ampliaciones=0)['escenarios']['VAN']
umbral = -e0 / (e1 - e0)          # el VAN es lineal en F
Markdown(f'Flujo mensual mínimo: {umbral["sin límite"]:.2f} k€ sin límite y {umbral["con límite"]:.2f} k€ con límite '
         f'(+{umbral["con límite"] - umbral["sin límite"]:.2f} k€ al mes): el límite exige más ingresos para recuperar lo mismo.')
