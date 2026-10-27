# ---
# title: Viabilidad 07
# subtitle: "9 actividades, 15 meses de inversión (I0 = 1250 k€), 60 k€/mes durante 48 meses, k = 12%"
# coleccion: 17_viabilidad
# variante: vinculo-proyecto
# nivel: 2
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
# Una empresa va a ejecutar el proyecto de inversión cuyo cuadro de prelaciones, duraciones (meses), presupuesto por actividad (BAC, k€), duración tope y coste unitario de reducción (k€/mes) se adjunta. El coste de cada actividad se desembolsa uniformemente a lo largo de su duración. Cuando el proyecto termina, la instalación genera un flujo de caja neto de 60 k€ al mes durante 48 meses. La empresa descuenta al 12% anual.

# %%
pregunta_1 = "1. Calcule el calendario del proyecto y, a partir del Gantt de inicios más tempranos, el desembolso de cada mes de la fase de inversión.  "
pregunta_2 = "2. Construya el flujo de caja mensual completo (inversión y explotación) y calcule el VAN con la tasa mensual equivalente. ¿Es viable?  "
pregunta_3 = "3. El proyecto sufre un retraso de 3 meses (los desembolsos se producen igual, pero los ingresos empiezan más tarde). ¿Cuánto VAN se pierde? ¿Cuánto vale, por tanto, cada mes de adelanto?  "
pregunta_4 = "4. Aplique Ackoff-Sasieni para reducir la duración del proyecto mes a mes hasta tres meses. Compare el sobrecoste de cada mes con la ganancia de VAN por adelantar los ingresos: ¿cuántos meses conviene acortar?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC, cap_duration, ucr   
    ('A'     , '---'      , 3       , 175, 2           , 8     ),
    ('B'     , '---'      , 3       , 200, 1           , 9     ),
    ('C'     , '---'      , 2       , 75 , 2           , np.inf),
    ('D'     , 'A,B,C'    , 3       , 125, 3           , np.inf),
    ('E'     , 'C'        , 3       , 125, 2           , 1     ),
    ('F'     , 'D'        , 2       , 100, 1           , 9     ),
    ('G'     , 'E,F'      , 7       , 100, 5           , 8     ),
    ('H'     , 'D,E'      , 6       , 200, 6           , np.inf),
    ('I'     , 'H'        , 2       , 150, 2           , np.inf),
], columns=['activity', 'predecessor', 'duration', 'BAC', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde al lanzamiento de una aplicación móvil; las actividades son las siguientes (las duraciones del cuadro están en meses).

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

_SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
def _num(x, d=1):
    return f'{round(float(x), d) + 0.0:.{d}f}'.replace('-', '−')
def _con_signo(x, d=1):
    return f' − {_num(-x, d)}' if x < 0 else f' + {_num(x, d)}'

def explica_van(flujos, k, unidad='k€', nombre='VAN'):
    '''VAN término a término: Σ flujo_t / (1 + k)^t con los números sustituidos.'''
    base = f'{1 + k:.6g}'
    terminos = _num(flujos[0]) + ''.join(f' {"−" if f < 0 else "+"} {_num(abs(f))}/{base}{str(t).translate(_SUP)}' for t, f in enumerate(flujos) if t > 0)
    actuales = _num(flujos[0]) + ''.join(_con_signo(f / (1 + k) ** t) for t, f in enumerate(flujos) if t > 0)
    return f'{nombre} = Σ flujo_t/(1 + k)^t = {terminos} = {actuales} = **{_num(van(flujos, k))} {unidad}**'

def explica_tir(flujos, unidad='k€'):
    '''Cómo se obtiene la TIR: horquilla de dos tasas enteras con VAN de signo contrario, interpolación y comprobación.'''
    r = tir(flujos)
    a = np.floor(r * 100) / 100; b = a + 0.01
    va, vb = van(flujos, a), van(flujos, b)
    aprox = a + 0.01 * va / (va - vb)
    return (f'La TIR es la tasa que anula el VAN. Probando tasas: VAN({a:.0%}) = {_num(va)} {unidad} y VAN({b:.0%}) = {_num(vb)} {unidad} tienen signo distinto, '
            f'así que la TIR está entre {a:.0%} y {b:.0%}; interpolando linealmente, TIR ≈ {a:.0%} + 1 % × {_num(va)}/({_num(va)} − ({_num(vb)})) = {aprox:.2%}. '
            f'Afinando (bisección, o =TIR(rango de flujos) en la hoja de cálculo): **TIR = {r:.2%}**. Comprobación: VAN({r:.2%}) = {abs(van(flujos, r)):.4f} ≈ 0.')

def explica_payback(flujos, k=None, unidad='k€'):
    '''Año en que el acumulado (simple o descontado) se hace positivo e interpolación lineal dentro de ese año.'''
    acum, nombre = 0.0, 'acumulado' + ('' if k is None else ' descontado')
    for t, f in enumerate(flujos):
        v = f if k is None else f / (1 + k) ** t
        if t > 0 and acum < 0 <= acum + v:
            return (f'el {nombre} pasa de {_num(acum)} (año {t - 1}) a {_num(acum + v)} (año {t}); dentro del año {t} se recuperan los {_num(-acum)} {unidad} '
                    f'que faltaban con un flujo{"" if k is None else " descontado"} de {_num(v)}: payback = {t - 1} + {_num(-acum)}/{_num(v)} = **{t - 1 + (-acum) / v:.2f} años**')
        acum += v
    return f'el {nombre} no llega a ser positivo: la inversión no se recupera'

def explica_van_mensual(pagos, F, N, km, unidad='k€'):
    '''VAN mensual = VA de los ingresos (renta de N meses desde el mes T + 1) − VA de los desembolsos de los meses 1…T.'''
    pagos = [float(x) for x in pagos]; T = len(pagos); base = f'{1 + km:.5f}'
    va_des = sum(x / (1 + km) ** m for m, x in enumerate(pagos, 1))
    a_N = (1 - (1 + km) ** -N) / km
    va_ing = F * a_N * (1 + km) ** -T
    t_des = ' + '.join(f'{_num(x)}/{base}{str(m).translate(_SUP)}' for m, x in list(enumerate(pagos, 1))[:3])
    if T > 3:
        t_des += f' + … + {_num(pagos[-1])}/{base}{str(T).translate(_SUP)}'
    return (f'- VA de los desembolsos = Σ desembolso_m/(1 + k_m)^m = {t_des} = **{_num(va_des)} {unidad}**.\n'
            f'- VA de los ingresos: renta de N = {N} meses de {_num(F)} {unidad} que empieza en el mes T + 1 = {T + 1}: '
            f'F · [1 − (1 + k_m)^−N]/k_m · (1 + k_m)^−T = {_num(F)} × [1 − {base}^−{N}]/{km:.6f} × {base}^−{T} = {_num(F)} × {a_N:.3f} × {(1 + km) ** -T:.4f} = **{_num(va_ing)} {unidad}**.\n'
            f'- VAN = {_num(va_ing)} − {_num(va_des)} = **{_num(va_ing - va_des)} {unidad}**.')

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
F, N, k, retraso = 60.0, 48, 0.12, 3
k_mensual = (1 + k) ** (1 / 12) - 1
I0 = float(datos_proyecto['BAC'].sum())

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Calendario del proyecto (meses)"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = int(proyecto.duration(datos_proyecto['duration']))
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con el camino crítico"
#| label: fig-grafo
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# El coste mensual de cada actividad es su BAC dividido por su duración. Escribiéndolo en cada casilla del Gantt de inicios más tempranos, el desembolso de cada mes es la suma de su columna (fila *Total*) y la fila *Acumulado* muestra lo invertido hasta ese mes; la columna *Total* de cada actividad recupera su BAC.

# %%
#| fig-cap: "Gantt con el coste mensual de cada actividad (k€): la fila Total es el desembolso del mes"
#| label: fig-ganttCostes
datos_gantt = datos_proyecto.copy()
datos_gantt['coste_mensual'] = datos_gantt['BAC'] / datos_gantt['duration']
gantt_costes, dibujo = proyecto.gantt(datos_gantt, 'duration', 'coste_mensual', total='ambas', acumulado=True, tikz=True,
                                      params=dict(inner_text_digits=1, **GANTT_PARAMS))
pinta_tikz(dibujo)

# %%
def desembolsos(cal, dur, duraciones=None):
    '''Desembolso de cada mes (1..T) repartiendo el BAC de cada actividad uniformemente.'''
    duraciones = datos_proyecto['duration'] if duraciones is None else duraciones
    pagos = pd.Series(0.0, index=range(1, int(dur) + 1))
    for a, fila in datos_proyecto.iterrows():
        es, d = int(cal.loc[a, 'inicio_mas_temprano']), int(duraciones[a])
        for m in range(es + 1, es + d + 1):
            pagos[m] += fila['BAC'] / d
    return pagos
pagos = desembolsos(calendario, duracion)
fila_total = gantt_costes.data.loc['Total'].drop('Total', errors='ignore').astype(float)
display(pd.DataFrame({'desembolso (k€)': pagos, 'fila Total del Gantt': fila_total.values[:len(pagos)]}).round(1).T)
Markdown(f'**Respuesta:** la fase de inversión dura {duracion} meses; el desembolso de cada mes es la fila *desembolso* de la tabla (suma de la columna del Gantt) y en total suma {pagos.sum():.0f} k€ (= Σ BAC = {I0:.0f}).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Cada mes es un periodo: en los meses 1…T el flujo es −desembolso; en los T+1…T+N, +F. La tasa mensual equivalente cumple $(1 + k_m)^{12} = 1 + k$. El VAN se separa en el valor actual (VA) de los ingresos, una renta constante de N meses que empieza en el mes T + 1, y el de los desembolsos.

# %%
def flujo_mensual(pagos, inicio_ingresos, sobrecoste=0.0):
    T = len(pagos)
    f = [0.0] + [-float(x) for x in pagos] + [0.0] * (inicio_ingresos - T) + [F] * N
    f[1] -= sobrecoste
    return f
flujos = flujo_mensual(pagos, duracion)
VAN = van(flujos, k_mensual)
resumen = pd.DataFrame({'mes': range(len(flujos)), 'flujo (k€)': flujos}).set_index('mes')
display(resumen.T.iloc[:, :duracion + 4].round(1))
va_ingresos = van([0.0] * (duracion + 1) + [F] * N, k_mensual)
Markdown(f'k mensual = (1 + {k})^(1/12) − 1 = {k_mensual:.4%}.\n\n' + explica_van_mensual(pagos, F, N, k_mensual)
         + f'\n\n**Respuesta:** VAN = {VAN:.1f} k€: ' + ('positivo, el proyecto es viable.' if VAN > 0 else 'negativo, el proyecto no es viable.'))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Si los ingresos empiezan m meses más tarde, la misma renta se descuenta m meses más: su valor actual se multiplica por (1 + k_m)^−m. Los desembolsos no cambian, así que el VAN perdido es VA ingresos × [1 − (1 + k_m)^−m].

# %%
VAN_retraso = van(flujo_mensual(pagos, duracion + retraso), k_mensual)
perdida = VAN - VAN_retraso
valor_mes = [VAN - van(flujo_mensual(pagos, duracion + m), k_mensual) for m in range(1, retraso + 1)]
display(pd.DataFrame({'meses de retraso': range(1, retraso + 1),
                      'VAN perdido = VA ingresos × [1 − (1 + k_m)^−m]': [f'{va_ingresos:.1f} × [1 − {1 + k_mensual:.5f}^−{m}] = {va_ingresos * (1 - (1 + k_mensual) ** -m):.1f}' for m in range(1, retraso + 1)]}).set_index('meses de retraso'))
Markdown(f'''**Respuesta:** con {retraso} meses de retraso el VAN baja de {VAN:.1f} a {VAN_retraso:.1f} k€: se pierden {perdida:.1f} k€, unos {perdida / retraso:.1f} k€ por mes. Por tanto, cada mes de adelanto vale aproximadamente {valor_mes[0]:.1f} k€ (el primero; los siguientes, algo menos, porque cada mes se descuenta sobre un valor ya descontado): es el precio máximo que merece la pena pagar por acortar el proyecto un mes.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni (hasta tres meses)"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_reducidas, margen = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'], reduction=3, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=3)

# %% [markdown]
# Para cada mes acortado se recalcula el calendario con las nuevas duraciones, los desembolsos (el BAC no cambia, pero se reparte en menos meses) y el VAN, restando el sobrecoste de reducción acumulado en el mes 1. La *ganancia del mes* es lo que añade ese mes al VAN del anterior.

# %%
filas = []
acumulado = 0.0
d = datos_proyecto['duration'].copy()
van_previo = VAN
for paso, acts in sorted(pasos.items()):
    coste = float(sum(datos_proyecto.loc[a, 'ucr'] for a in acts)); acumulado += coste
    for a in acts: d[a] -= 1
    p2 = proyecto.copy(); cal2 = p2.calendar(d); dur2 = int(p2.duration(d))
    VAN2 = van(flujo_mensual(desembolsos(cal2, dur2, d), dur2, sobrecoste=acumulado), k_mensual)
    filas.append({'meses acortados': paso + 1, 'actividades': ', '.join(acts), 'sobrecoste del mes (k€)': coste,
                  'sobrecoste acumulado': acumulado, 'duración': dur2, 'VAN (k€)': VAN2, 'ganancia del mes': VAN2 - van_previo,
                  'ganancia de VAN frente a no acortar': VAN2 - VAN})
    van_previo = VAN2
decision = pd.DataFrame(filas).set_index('meses acortados')
display(decision.round(1))
mejor = int(decision['VAN (k€)'].idxmax()) if decision['VAN (k€)'].max() > VAN else 0
_txt = (f'Cada mes acortado cuesta su sobrecoste ({", ".join(f"{c:.0f}" for c in decision["sobrecoste del mes (k€)"])} k€) y gana adelantar un mes los ingresos (≈ {valor_mes[0]:.1f} k€), aunque también adelanta algo los desembolsos: la ganancia neta del mes es la columna *ganancia del mes*. ')
if len(decision) < 3:
    _txt += f'Ackoff-Sasieni sólo permite acortar {plural(len(decision), "mes", "meses")}: después no queda ninguna actividad (o combinación) que acorte todas las rutas críticas sin pasar de su duración tope. '
_txt += f'**Respuesta:** conviene acortar {plural(mejor, "mes", "meses")}: el VAN máximo es {max(VAN, decision["VAN (k€)"].max()):.1f} k€ (sin acortar, {VAN:.1f}). '
_no = decision.index[decision['ganancia del mes'] <= 0].tolist()
_txt += (f'A partir del mes {_no[0]} el sobrecoste supera lo que vale adelantar los ingresos.' if _no else
         'Todos los meses posibles compensan: su sobrecoste es menor que lo que vale adelantar un mes los ingresos.')
Markdown(_txt)
