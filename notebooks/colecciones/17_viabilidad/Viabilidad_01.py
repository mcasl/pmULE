# ---
# title: Viabilidad 01
# subtitle: "una línea de envasado automatizada: I0 = 760 k€, 7 años, k = 12%; VAN 392, TIR 23.8%"
# coleccion: 17_viabilidad
# variante: caso-base
# nivel: 1
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
# Una empresa estudia invertir en una línea de envasado automatizada. Los datos del caso se adjuntan (importes en k€; precio y coste variable en € por unidad). La inversión se amortiza linealmente durante la vida del proyecto hasta su valor residual, que se recupera al final por su valor contable. El circulante necesario cada año es una fracción de las ventas de ese año, se aporta al principio del año (el del primer año, en el momento 0) y se recupera íntegramente al final.

# %%
pregunta_1 = "1. Construya la tabla de flujos de caja libre año a año: ventas, costes, EBITDA, amortización, BAI, impuesto, BDI y flujo de caja libre (FCL).  "
pregunta_2 = "2. Calcule el VAN a la tasa de descuento dada, mostrando el factor de descuento y el valor actual de cada flujo. ¿Es viable el proyecto?  "
pregunta_3 = "3. Calcule la TIR. ¿Cómo se interpreta frente a la tasa de descuento?  "
pregunta_4 = "4. Calcule el periodo de recuperación simple y el descontado. ¿Por qué el descontado es mayor? ¿Qué información da el payback que no dan el VAN ni la TIR, y cuál omite?  "
pregunta_5 = "5. ¿Qué supuesto del caso le parece más frágil? Compruebe cuánto cambia el VAN si las unidades del año 1 son un 15 % menores.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del caso"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  parametro                                              , valor
    ('inversión inicial (k€)'                               , 760  ),
    ('vida del proyecto (años)'                             , 7    ),
    ('precio (€ por miles de unidade)'                      , 3    ),
    ('miles de unidades vendidas el año 1'                  , 360  ),
    ('crecimiento anual de las unidades'                    , 0.05 ),
    ('coste variable unitario (€)'                          , 2    ),
    ('costes fijos anuales (k€)'                            , 100  ),
    ('valor residual al final (k€)'                         , 75   ),
    ('circulante necesario (fracción de las ventas del año)', 0.08 ),
    ('tipo impositivo'                                      , 0.25 ),
    ('tasa de descuento'                                    , 0.12 ),
], columns=['parametro', 'valor']).set_index('parametro')
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

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Orden de cálculo, año a año (k€): ventas = unidades × precio; EBITDA = ventas − costes variables − costes fijos; BAI = EBITDA − amortización (lineal sobre inversión − valor residual); impuesto = tipo × BAI si BAI > 0; BDI = BAI − impuesto. El flujo de caja libre suma al BDI la amortización (no es un pago) y resta las inversiones en activo fijo y en circulante; en el último año se recuperan el circulante y el valor residual.

# %%
p = datos_enunciado['valor'].to_dict()
I0, n, precio = p['inversión inicial (k€)'], int(p['vida del proyecto (años)']), [v for c, v in p.items() if c.startswith('precio')][0]
u1 = [v for c, v in p.items() if 'año 1' in c][0]
g, cv, CF = p['crecimiento anual de las unidades'], p['coste variable unitario (€)'], p['costes fijos anuales (k€)']
VR, w, t_imp, k = p['valor residual al final (k€)'], p['circulante necesario (fracción de las ventas del año)'], p['tipo impositivo'], p['tasa de descuento']

# %%
#| tbl-cap: "Flujos de caja libre del proyecto (k€)"
#| label: tbl-flujos
anos = list(range(0, n + 1))
unidades = pd.Series([0] + [u1 * (1 + g) ** (t - 1) for t in range(1, n + 1)], index=anos)
ventas = unidades * precio
costes_variables = unidades * cv
costes_fijos = pd.Series([0] + [CF] * n, index=anos)
EBITDA = ventas - costes_variables - costes_fijos
amortizacion = pd.Series([0] + [(I0 - VR) / n] * n, index=anos)
BAI = EBITDA - amortizacion
impuesto = BAI.clip(lower=0) * t_imp; impuesto[0] = 0
BDI = BAI - impuesto
circulante = pd.Series([w * ventas[t + 1] for t in range(n)] + [0], index=anos)   # saldo en el momento t: el que necesitan las ventas del año t + 1
variacion_circulante = circulante.diff().fillna(circulante[0])   # aportaciones al principio de cada año; en t = n (saldo 0) se recupera todo
inversion = pd.Series([I0] + [0] * n, index=anos)
valor_residual = pd.Series([0] * n + [VR], index=anos)
FCL = BDI + amortizacion - variacion_circulante - inversion + valor_residual
tabla = pd.DataFrame({'unidades': unidades, 'ventas': ventas, 'costes variables': costes_variables, 'costes fijos': costes_fijos,
                      'EBITDA': EBITDA, 'amortización': amortizacion, 'BAI': BAI, 'impuesto': impuesto, 'BDI': BDI,
                      '+ amortización': amortizacion, '− Δ circulante': -variacion_circulante, '− inversión': -inversion,
                      '+ valor residual': valor_residual, 'FCL': FCL})
tabla.index.name = 'año'
flujos = FCL.tolist()
tabla.round(1).T

# %%
def _fcl_txt(t):
    return (f'FCL_{t} = BDI + amortización − Δ circulante − inversión + valor residual = {_num(BDI[t])}{_con_signo(amortizacion[t])}'
            f'{_con_signo(-variacion_circulante[t])}{_con_signo(-inversion[t])}{_con_signo(valor_residual[t])} = **{_num(FCL[t])}**')
Markdown('Siglas: EBITDA, resultado antes de intereses, impuestos y amortizaciones; BAI, beneficio antes de impuestos; BDI, beneficio después de impuestos; '
         'FCL, flujo de caja libre (lo que el proyecto deja disponible cada año, en k€). Cálculo detallado del año 1 y de los años 0 y n:\n\n'
         f'- Ventas_1 = unidades × precio = {_num(unidades[1])} × {precio:g} = {_num(ventas[1])}; costes variables = {_num(unidades[1])} × {cv:g} = {_num(costes_variables[1])}; '
         f'EBITDA_1 = {_num(ventas[1])} − {_num(costes_variables[1])} − {_num(CF)} = {_num(EBITDA[1])}.\n'
         f'- Amortización anual = (I0 − VR)/n = ({_num(I0, 0)} − {_num(VR, 0)})/{n} = {_num(amortizacion[1])}; BAI_1 = {_num(EBITDA[1])} − {_num(amortizacion[1])} = {_num(BAI[1])}; '
         f'impuesto_1 = {t_imp:g} × {_num(BAI[1])} = {_num(impuesto[1])}; BDI_1 = {_num(BAI[1])} − {_num(impuesto[1])} = {_num(BDI[1])}.\n'
         f'- Circulante (se aporta al principio de cada año, es decir, al final del anterior): en t = 0 el del año 1, {w:g} × {_num(ventas[1])} = {_num(variacion_circulante[0])}; '
         f'en cada momento t = 1…{n - 1}, el aumento que necesitan las ventas del año siguiente, {w:g} × (ventas_t+1 − ventas_t) '
         f'(en t = 1: {w:g} × ({_num(ventas[2])} − {_num(ventas[1])}) = {_num(variacion_circulante[1])}); en t = {n} se recupera todo el circulante acumulado, {w:g} × {_num(ventas[n])} = {_num(w * ventas[n])}.\n'
         f'- {_fcl_txt(0)}; {_fcl_txt(1)}; {_fcl_txt(n)}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# VAN (valor actual neto): cada flujo se lleva al momento 0 dividiéndolo por (1 + k)^t, o multiplicándolo por el factor de descuento (1 + k)^−t, y se suman todos. La tabla da el factor y el valor actual de cada año (las dos últimas columnas se usan en el apartado 4).

# %%
#| tbl-cap: "Descuento de los flujos"
#| label: tbl-descuento
descuento = tabla_descuento(flujos, k)
descuento.round({'flujo': 1, '(1 + k)^-t': 4, 'valor actual': 1, 'acumulado': 1, 'acumulado descontado': 1})

# %%
VAN = van(flujos, k)
Markdown(explica_van(flujos, k) + '. **Respuesta:** ' + (f'el VAN es positivo ({VAN:.1f} k€): el proyecto recupera la inversión, remunera el capital al {k:.0%} y además crea {VAN:.0f} k€ de valor; es viable.'
         if VAN > 0 else f'el VAN es negativo ({VAN:.1f} k€): no es viable a esa tasa.'))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
TIR = tir(flujos)
Markdown(explica_tir(flujos) + f' **Respuesta:** la TIR ({TIR:.2%}) es la rentabilidad anual que da el propio proyecto; '
         + (f'como supera la tasa de descuento ({k:.0%}), el proyecto rinde más de lo que se exige al dinero y el VAN es positivo' if TIR > k else
            f'como no llega a la tasa de descuento ({k:.0%}), el VAN es negativo')
         + ': las dos reglas coinciden en un proyecto aislado con flujos convencionales (un desembolso y luego cobros). El gráfico muestra el VAN para cada tasa: corta el eje en la TIR.')

# %%
#| fig-cap: "Perfil del VAN frente a la tasa de descuento"
#| label: fig-perfil
from matplotlib import pyplot as plt
ks = np.linspace(0, max(0.4, TIR + 0.1), 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ks * 100, [van(flujos, x) for x in ks]); ax.axhline(0, color='grey')
ax.axvline(k * 100, color='tab:green', linestyle='--', label=f'tasa de descuento {k:.0%}: VAN = {VAN:.0f}')
ax.axvline(TIR * 100, color='tab:red', linestyle=':', label=f'TIR = {TIR:.1%}')
ax.set_xlabel('tasa de descuento (%)'); ax.set_ylabel('VAN (k€)'); ax.legend(); plt.show()

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# El periodo de recuperación (*payback*) es el momento en que el flujo acumulado deja de ser negativo; dentro del año en que cambia de signo se interpola linealmente (se supone que el flujo de ese año llega repartido de forma uniforme). El descontado hace lo mismo con los valores actuales.

# %%
pb, pbd = payback(flujos), payback(flujos, k)
display(descuento[['flujo', 'acumulado', 'valor actual', 'acumulado descontado']].round(1))
Markdown(f'Payback simple: {explica_payback(flujos)}. Payback descontado: {explica_payback(flujos, k)}. '
         f'**Respuesta:** {pb:.2f} años el simple y {pbd:.2f} el descontado. El descontado es mayor porque cada flujo descontado es menor que el nominal (se divide por (1 + k)^t > 1), '
         'así que hace falta más tiempo para cubrir la inversión. El payback informa de la liquidez y del riesgo de plazo (cuánto tiempo está el dinero expuesto), que no dan el VAN ni la TIR; '
         f'pero ignora los flujos posteriores a la recuperación (aquí, los de los años {int(np.ceil(pb)) + 1} a {n}, con el valor residual y la recuperación del circulante) y, en su versión simple, el valor del tiempo; '
         'por eso se usa como criterio complementario, nunca único.')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
u1_bajo = u1 * 0.85
def modelo(u1, precio, cv, CF, g=g, I0=I0, n=n, VR=VR, w=w, t_imp=t_imp):
    unidades = pd.Series([0] + [u1 * (1 + g) ** (t - 1) for t in range(1, n + 1)])
    ventas = unidades * precio; EBITDA = ventas - unidades * cv - pd.Series([0] + [CF] * n)
    amort = pd.Series([0] + [(I0 - VR) / n] * n); BAI = EBITDA - amort
    imp = BAI.clip(lower=0) * t_imp; imp[0] = 0
    circ = pd.Series([w * ventas[t + 1] for t in range(n)] + [0])
    dcirc = circ.diff().fillna(circ[0])
    return (BAI - imp + amort - dcirc - pd.Series([I0] + [0] * n) + pd.Series([0] * n + [VR])).tolist()
flujos_bajo = modelo(u1_bajo, precio, cv, CF)
VAN_bajo = van(flujos_bajo, k)
display(pd.DataFrame({'FCL caso base': flujos, f'FCL con {u1_bajo:g} unidades el año 1': flujos_bajo}, index=pd.Index(range(n + 1), name='año')).T.round(1))
Markdown(f'Con las unidades del año 1 un 15 % menores ({u1:g} × 0.85 = {u1_bajo:g}; las de los años siguientes bajan en la misma proporción) se repite el modelo: '
         + explica_van(flujos_bajo, k) + f'. **Respuesta:** el VAN pasa de {VAN:.1f} a {VAN_bajo:.1f} k€ ({(VAN_bajo - VAN) / VAN:+.0%}): '
         + ('sigue siendo viable. ' if VAN_bajo > 0 else 'deja de ser viable. ')
         + 'Las unidades vendidas (y el precio) son el supuesto más frágil: son previsiones de mercado, mientras que la inversión y los costes fijos están casi cerrados por contrato; '
         + f'un 15 % menos de volumen se lleva el {abs(VAN_bajo - VAN) / VAN:.0%} del VAN.')
