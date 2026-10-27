# ---
# title: Viabilidad 11
# subtitle: "8 actividades, 18 meses, I0 = 1075 k€, ingreso 30 ± 10 k€/mes durante 60 meses, k = 10%; distribución del VAN"
# coleccion: 17_viabilidad
# variante: van-montecarlo
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
# Una empresa va a ejecutar el proyecto de inversión cuyo cuadro de prelaciones se adjunta, con la duración esperada de cada actividad (meses), sus tres estimaciones (optimista $a$, más probable $m$, pesimista $b$, con distribución triangular) y su presupuesto (BAC, k€). El coste de cada actividad se desembolsa uniformemente a lo largo de su duración. Cuando el proyecto termina, la instalación genera un flujo de caja neto mensual de 30 k€ durante 60 meses, pero ese flujo es incierto: entre 20 y 40 k€ (triangular simétrica). La empresa descuenta al 10% anual.

# %%
pregunta_1 = "1. Con las duraciones esperadas, calcule el calendario, el camino crítico y el VAN determinista del proyecto (flujo mensual, tasa mensual equivalente).  "
pregunta_2 = "2. Simule 5 000 réplicas: duraciones triangulares e ingreso mensual incierto. De cada réplica obtenga la duración del proyecto y su VAN. Dibuje los histogramas y calcule la media, la desviación típica, P(VAN < 0) y el percentil 5 %.  "
pregunta_3 = "3. ¿Por qué la media del VAN simulado no coincide con el VAN determinista? Separe los dos efectos simulando por separado solo la incertidumbre de las duraciones y solo la del ingreso.  "
pregunta_4 = "4. ¿Qué variables explican la variabilidad del VAN? Calcule la correlación de Spearman del VAN con cada duración y con el ingreso, y compárela con el índice de criticidad de cada actividad. ¿Hay alguna actividad cuya demora *mejore* el VAN? ¿Por qué?  "
pregunta_5 = "5. La empresa puede (a) encargar un estudio de mercado que fije el ingreso en su valor central, o (b) contratar a un subcontratista que fije la duración de la actividad más influyente en su valor esperado. Evalúe qué cambia con cada opción (y con ambas) en la media, la desviación y P(VAN < 0), y qué compra realmente cada una.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, a, m, b , BAC
    ('A'     , '---'      , 4       , 1, 5, 6 , 200),
    ('B'     , '---'      , 6       , 3, 5, 10, 50 ),
    ('C'     , '---'      , 2       , 1, 2, 3 , 200),
    ('D'     , 'A,B,C'    , 6       , 5, 5, 8 , 100),
    ('E'     , 'A,C'      , 6       , 3, 7, 8 , 175),
    ('F'     , 'D,E'      , 5       , 3, 5, 7 , 75 ),
    ('G'     , 'E'        , 5       , 4, 4, 7 , 150),
    ('H'     , 'F,G'      , 1       , 1, 1, 1 , 125),
], columns=['activity', 'predecessor', 'duration', 'a', 'm', 'b', 'BAC']).set_index('activity')
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
F, delta, N, k = 30.0, 10.0, 60, 0.1
k_mensual = (1 + k) ** (1 / 12) - 1
I0 = float(datos_proyecto['BAC'].sum())

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Calendario con las duraciones esperadas (meses)"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = int(proyecto.duration(datos_proyecto['duration']))
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con el camino crítico"
#| label: fig-grafo
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
pagos = pd.Series(0.0, index=range(1, duracion + 1))
for act, fila in datos_proyecto.iterrows():
    es, d = int(calendario.loc[act, 'inicio_mas_temprano']), int(fila['duration'])
    for mes in range(es + 1, es + d + 1):
        pagos[mes] += fila['BAC'] / d
flujos = [0.0] + [-float(x) for x in pagos] + [F] * N
VAN_det = van(flujos, k_mensual)
criticas = calendario.index[calendario['H_total'] == 0].tolist()
display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'meses'))
Markdown(f'Desembolso de cada mes (BAC/duración de cada actividad en curso, con los inicios más tempranos): ' + ', '.join(f'{x:.1f}' for x in pagos) + f' k€ (Σ = Σ BAC = {I0:.0f} k€). '
         f'k mensual = (1 + {k})^(1/12) − 1 = {k_mensual:.4%}.\n\n' + explica_van_mensual(pagos, F, N, k_mensual)
         + f'\n\n**Respuesta:** duración {duracion} meses; camino crítico {" – ".join(ordena_ruta(criticas, calendario))}; VAN determinista = {VAN_det:.1f} k€.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
distribuciones = {act: ('triangular', float(f['a']), float(f['m']), float(f['b'])) for act, f in datos_proyecto.iterrows()}
ingreso = ('triangular', F - delta, F, F + delta)
mc = proyecto.monte_carlo_van(distribuciones, datos_proyecto['BAC'], ingreso, N, k, n_samples=5000, seed=2025)
resumen = mc.summary().round(2)
resumen

# %%
#| fig-cap: "Distribución de la duración del proyecto y del VAN (5 000 réplicas)"
#| label: fig-histogramas
import matplotlib.pyplot as plt
fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6))
ejes[0].hist(mc.tau_samples, bins=range(int(mc.tau_samples.min()), int(mc.tau_samples.max()) + 2), color='#4c78a8', edgecolor='white')
ejes[0].axvline(duracion, color='k', ls='--', label=f'determinista ({duracion})'); ejes[0].set_xlabel('duración (meses)'); ejes[0].legend()
ejes[1].hist(mc.van_samples, bins=40, color='#4c78a8', edgecolor='white')
ejes[1].axvline(VAN_det, color='k', ls='--', label='VAN determinista'); ejes[1].axvline(0, color='#c44', label='VAN = 0')
ejes[1].set_xlabel('VAN (k€)'); ejes[1].legend(); plt.tight_layout()

# %%
Markdown(f'La duración media es {mc.tau_samples.mean():.2f} meses frente a {duracion} del plan; el VAN medio es {mc.van_mean:.1f} k€ '
         f'(determinista {VAN_det:.1f}), con desviación {mc.van_std:.1f}. P(VAN < 0) = {mc.prob_negativo:.1%} y el percentil 5 % es {mc.cuantil(0.05):.1f} k€. '
         f'Las probabilidades salen de contar réplicas: P(VAN < 0) = (réplicas con VAN < 0)/5000 = {int(round(mc.prob_negativo * 5000))}/5000; el percentil 5 % deja por debajo el 5 % de los VAN simulados (250 réplicas). '
         f'**Respuesta:** media {mc.van_mean:.1f} k€, desviación típica {mc.van_std:.1f} k€, P(VAN < 0) = {mc.prob_negativo:.1%}, percentil 5 % = {mc.cuantil(0.05):.1f} k€.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
fijas = {act: ('uniform', float(f['duration']), float(f['duration'])) for act, f in datos_proyecto.iterrows()}
solo_duraciones = proyecto.monte_carlo_van(distribuciones, datos_proyecto['BAC'], F, N, k, n_samples=5000, seed=2025)
solo_ingreso = proyecto.monte_carlo_van(fijas, datos_proyecto['BAC'], ingreso, N, k, n_samples=5000, seed=2025)
filas = {'VAN determinista': (VAN_det, 0.0, float(VAN_det < 0))}
for nombre, r in {'solo duraciones inciertas': solo_duraciones, 'solo ingreso incierto': solo_ingreso, 'ambas': mc}.items():
    filas[nombre] = (r.van_mean, r.van_std, r.prob_negativo)
tabla = pd.DataFrame(filas, index=['VAN medio', 'desviación', 'P(VAN < 0)']).T
display(tabla.round(3))
ee_ing, ee_dur = 2 * solo_ingreso.van_std / np.sqrt(5000), 2 * solo_duraciones.van_std / np.sqrt(5000)
Markdown(f'La incertidumbre del ingreso (simétrica y lineal en el VAN) ensancha la distribución pero no mueve la media: '
         f'{solo_ingreso.van_mean:.1f} ± {ee_ing:.1f} (dos errores estándar de simulación) frente a {VAN_det:.1f}. Las duraciones sí la bajan, hasta '
         f'{solo_duraciones.van_mean:.1f} ± {ee_dur:.1f}: el proyecto termina cuando termina el último camino, así que la duración media es '
         f'{solo_duraciones.tau_samples.mean():.2f} > {duracion} (caminos casi críticos que se fusionan) y cada mes de retraso retrasa un flujo descontado. '
         'Es el mismo sesgo de la aproximación normal, ahora medido en k€. '
         f'**Respuesta:** la media simulada ({mc.van_mean:.1f}) queda por debajo del VAN determinista ({VAN_det:.1f}) por las duraciones (el proyecto dura de media más de lo planificado y los ingresos llegan más tarde); '
         'la incertidumbre del ingreso sólo añade dispersión.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
tornado = mc.tornado()
criticidad = proyecto.monte_carlo(distribuciones, n_samples=5000, seed=2025).sensitivity_mean
comparacion = pd.DataFrame({'Spearman con el VAN': tornado['Spearman con el VAN']})
comparacion['índice de criticidad'] = [criticidad.get(v.split()[-1], np.nan) if v.startswith('duración') else np.nan for v in comparacion.index]
display(comparacion.round(3))
_dur = comparacion.loc[[v for v in comparacion.index if v.startswith('duración')]]
positivas = [v.split()[-1] for v, r in _dur['Spearman con el VAN'].items() if r > 0]
_top = _dur['Spearman con el VAN'].abs().idxmax().split()[-1]
Markdown(f'Manda el ingreso mensual (Spearman {tornado.loc["ingreso mensual", "Spearman con el VAN"]:.3f}); entre las duraciones, la de más peso es {_top} '
         f'(Spearman {_dur.loc["duración " + _top, "Spearman con el VAN"]:.3f}, índice de criticidad {_dur.loc["duración " + _top, "índice de criticidad"]:.3f}): las actividades que más veces son críticas son las que, al alargarse, retrasan el cobro. '
         + (f'**Respuesta:** tienen correlación positiva (su demora mejora el VAN) {lista_y(positivas)}, con índices de criticidad '
            + ', '.join(f'{a}: {_dur.loc["duración " + a, "índice de criticidad"]:.3f}' for a in positivas)
            + '. Son actividades que casi nunca son críticas: si se alargan no retrasan el cobro y desplazan el pago de su BAC hacia meses más descontados, '
              f'así que el VAN mejora ligeramente; el efecto es pequeño (|Spearman| ≤ {max(abs(_dur.loc["duración " + a, "Spearman con el VAN"]) for a in positivas):.3f}).'
            if positivas else '**Respuesta:** ninguna duración tiene correlación positiva con el VAN: toda demora lo empeora o no lo afecta.'))

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
influyente = tornado.index[[v.startswith('duración') for v in tornado.index]][0].split()[-1]
centrada = dict(distribuciones); centrada[influyente] = ('uniform', float(datos_proyecto.loc[influyente, 'duration']), float(datos_proyecto.loc[influyente, 'duration']))
opciones = {'sin hacer nada': mc,
            '(a) fijar el ingreso': proyecto.monte_carlo_van(distribuciones, datos_proyecto['BAC'], F, N, k, n_samples=5000, seed=2025),
            f'(b) fijar la duración de {influyente}': proyecto.monte_carlo_van(centrada, datos_proyecto['BAC'], ingreso, N, k, n_samples=5000, seed=2025),
            '(a) y (b)': proyecto.monte_carlo_van(centrada, datos_proyecto['BAC'], F, N, k, n_samples=5000, seed=2025)}
tabla_op = pd.DataFrame({n: (r.van_mean, r.van_std, r.prob_negativo, r.cvar(0.05)) for n, r in opciones.items()},
                        index=['VAN medio', 'desviación', 'P(VAN < 0)', 'CVaR 5 %']).T
display(tabla_op.round(3))
mejor_riesgo = tabla_op.drop(index=['sin hacer nada', '(a) y (b)'])['P(VAN < 0)'].idxmin()
Markdown(f'La actividad más influyente es {influyente}. Fijar el ingreso o una duración cambia poco el VAN medio '
         f'({tabla_op.loc["sin hacer nada", "VAN medio"]:.1f} → entre {tabla_op["VAN medio"].drop("sin hacer nada").min():.1f} y {tabla_op["VAN medio"].drop("sin hacer nada").max():.1f}): '
         f'lo que compra es seguridad. La opción que más reduce P(VAN < 0) es {mejor_riesgo}. Si la empresa es neutral al riesgo, solo vale la pena pagar '
         'por lo que sube la media; si le preocupa la cola (P(VAN < 0), CVaR, el VAN medio del 5 % de peores réplicas), se paga por reducir dispersión. '
         f'**Respuesta:** (a) deja la desviación en {tabla_op.loc["(a) fijar el ingreso", "desviación"]:.1f} k€ y P(VAN < 0) en {tabla_op.loc["(a) fijar el ingreso", "P(VAN < 0)"]:.1%}; '
         f'(b) en {tabla_op.iloc[2]["desviación"]:.1f} k€ y {tabla_op.iloc[2]["P(VAN < 0)"]:.1%}; las dos, en {tabla_op.loc["(a) y (b)", "desviación"]:.1f} k€ y {tabla_op.loc["(a) y (b)", "P(VAN < 0)"]:.1%} '
         f'(sin hacer nada: {tabla_op.loc["sin hacer nada", "desviación"]:.1f} k€ y {tabla_op.loc["sin hacer nada", "P(VAN < 0)"]:.1%}). Ninguna sube apreciablemente el VAN medio: lo que se compra es reducción del riesgo'
         + (f'; (b) apenas lo reduce, porque la duración de {influyente} pesa poco frente al ingreso.' if tabla_op.iloc[2]['desviación'] > 0.9 * tabla_op.loc['sin hacer nada', 'desviación'] else '.'))
