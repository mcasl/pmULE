# ---
# title: Viabilidad 10
# subtitle: "una línea de envasado automatizada: préstamo del 70% al 5% a 4 años; TIR proyecto 15.6%"
# coleccion: 17_viabilidad
# variante: financiacion
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
# Una empresa estudia invertir en una línea de envasado automatizada (datos adjuntos; mismas hipótesis que en el caso base). Financiaría el 70% de la inversión inicial con un préstamo al 5% anual, amortizable en 4 años por cuotas constantes (sistema francés); el resto lo aportan los socios, que exigen una rentabilidad del 16%. Los intereses son deducibles en el impuesto de sociedades.

# %%
pregunta_1 = "1. Calcule el flujo de caja libre del proyecto (sin deuda), su VAN a la tasa dada y su TIR.  "
pregunta_2 = "2. Construya el cuadro de amortización del préstamo: cuota, intereses, amortización de capital y capital vivo de cada año.  "
pregunta_3 = "3. Construya el flujo de caja del accionista: FCL + préstamo recibido − cuota + ahorro fiscal de los intereses. Calcule su VAN a la rentabilidad exigida por los socios y su TIR.  "
pregunta_4 = "4. Compare la TIR del proyecto con la del accionista y explique el efecto apalancamiento. ¿Qué ocurriría si el tipo del préstamo fuera del 23% (hipótesis extrema, para ver el efecto)? Recalcule.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del caso"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  parametro                                              , valor
    ('inversión inicial (k€)'                               , 950  ),
    ('vida del proyecto (años)'                             , 6    ),
    ('precio (€ por miles de unidade)'                      , 3    ),
    ('miles de unidades vendidas el año 1'                  , 210  ),
    ('crecimiento anual de las unidades'                    , 0    ),
    ('coste variable unitario (€)'                          , 1    ),
    ('costes fijos anuales (k€)'                            , 135  ),
    ('valor residual al final (k€)'                         , 140  ),
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

# %%
VAN, TIR = van(flujos, k), tir(flujos)
Markdown(explica_van(flujos, k) + '.\n\n' + explica_tir(flujos)
         + f'\n\n**Respuesta:** VAN del proyecto = {VAN:.1f} k€ al {k:.0%}; TIR del proyecto = {TIR:.2%}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Cuota constante $C = P \cdot \dfrac{i}{1 - (1 + i)^{-n_p}}$; cada año los intereses son $i$ por el capital vivo al inicio y la amortización es la cuota menos los intereses.

# %%
alfa, i, n_p, ke = 0.7, 0.05, 4, 0.16
def cuadro_prestamo(P, i, n_p):
    C = P * i / (1 - (1 + i) ** -n_p)
    vivo, filas = P, []
    for t in range(1, n_p + 1):
        interes = vivo * i; amort = C - interes; vivo -= amort
        filas.append({'año': t, 'cuota': C, 'intereses': interes, 'amortización': amort, 'capital vivo': max(vivo, 0)})
    return pd.DataFrame(filas).set_index('año')
prestamo = alfa * I0
cuadro = cuadro_prestamo(prestamo, i, n_p)
_c1 = cuadro.iloc[0]
display(Markdown(f'Préstamo = {alfa:.0%} × {I0:.0f} = {prestamo:.1f} k€; cuota = {prestamo:.1f} × {i} / (1 − (1 + {i})^−{n_p}) = {_c1["cuota"]:.2f} k€. '
                 f'Año 1: intereses = {i} × {prestamo:.2f} = {_c1["intereses"]:.2f}; amortización = {_c1["cuota"]:.2f} − {_c1["intereses"]:.2f} = {_c1["amortización"]:.2f}; '
                 f'capital vivo = {prestamo:.2f} − {_c1["amortización"]:.2f} = {_c1["capital vivo"]:.2f}. Los años siguientes se calculan igual a partir del capital vivo del año anterior; '
                 f'en el año {n_p} el capital vivo llega a 0 (la suma de las amortizaciones, {cuadro["amortización"].sum():.2f}, devuelve el préstamo).'))
cuadro.round(2)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# El accionista recibe el préstamo en t = 0 (pone sólo el resto de la inversión), paga cada año la cuota y se ahorra el impuesto sobre los intereses (escudo fiscal = tipo impositivo × intereses). Su flujo se descuenta a la rentabilidad que exigen los socios.

# %%
def flujo_accionista(cuadro, prestamo):
    f = pd.Series(flujos, index=range(n + 1)).astype(float)
    f[0] += prestamo
    for t, fila in cuadro.iterrows():
        f[t] += -fila['cuota'] + t_imp * fila['intereses']
    return f
fa = flujo_accionista(cuadro, prestamo)
detalle = pd.DataFrame({'FCL proyecto': flujos, '+ préstamo': [prestamo] + [0] * n,
                        '− cuota': [0] + [-cuadro['cuota'].get(t, 0) for t in range(1, n + 1)],
                        '+ ahorro fiscal intereses': [0] + [t_imp * cuadro['intereses'].get(t, 0) for t in range(1, n + 1)],
                        'flujo del accionista': fa.tolist()}, index=pd.Index(range(n + 1), name='año'))
display(detalle.round(1).T)
VAN_acc, TIR_acc = van(fa.tolist(), ke), tir(fa.tolist())
Markdown(f'Año 1: {flujos[1]:.1f} − {cuadro.loc[1, "cuota"]:.1f} + {t_imp} × {cuadro.loc[1, "intereses"]:.1f} = {fa[1]:.1f}. '
         + explica_van(fa.tolist(), ke, nombre='VAN del accionista') + '.\n\n' + explica_tir(fa.tolist())
         + f'\n\n**Respuesta:** VAN para el accionista (al {ke:.0%}) = {VAN_acc:.1f} k€; TIR del accionista = {TIR_acc:.2%}.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
i_alto = 0.23
cuadro_alto = cuadro_prestamo(prestamo, i_alto, n_p)
fa_alto = flujo_accionista(cuadro_alto, prestamo)
TIR_alto = tir(fa_alto.tolist())
coste_deuda_neto = i * (1 - t_imp); coste_alto_neto = i_alto * (1 - t_imp)
display(pd.DataFrame({'proyecto (sin deuda)': {'TIR': TIR}, f'accionista, préstamo al {i:.0%}': {'TIR': TIR_acc},
                      f'accionista, préstamo al {i_alto:.0%}': {'TIR': TIR_alto}}).style.format('{:.2%}'))
display(Markdown(f'Con el préstamo al {i_alto:.0%}: cuota = {prestamo:.1f} × {i_alto} / (1 − (1 + {i_alto})^−{n_p}) = {cuadro_alto["cuota"].iloc[0]:.2f} k€; '
                 'flujo del accionista = ' + ', '.join(f'{x:.1f}' for x in fa_alto) + '. ' + explica_tir(fa_alto.tolist())))
Markdown(f'''**Respuesta:** la deuda cuesta {i:.0%}, es decir, {i} × (1 − {t_imp}) = {coste_deuda_neto:.2%} tras impuestos, menos que la rentabilidad del proyecto ({TIR:.2%}): la diferencia se la queda el accionista, que ha puesto sólo el {1 - alfa:.0%} del capital, y su TIR sube a {TIR_acc:.2%}. Es el apalancamiento financiero positivo. Con el préstamo al {i_alto:.0%} ({i_alto} × (1 − {t_imp}) = {coste_alto_neto:.2%} neto, por encima de la TIR del proyecto) el efecto se invierte: la TIR del accionista {"cae a" if TIR_alto < TIR_acc else "queda en"} {TIR_alto:.2%}{", por debajo de la del proyecto" if TIR_alto < TIR else ""}. La condición es que la rentabilidad del proyecto supere el coste de la deuda después de impuestos. El apalancamiento amplifica en los dos sentidos, y además concentra el riesgo: las cuotas se pagan aunque las ventas fallen.''')
