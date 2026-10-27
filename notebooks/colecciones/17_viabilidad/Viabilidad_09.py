# ---
# title: Viabilidad 09
# subtitle: "una línea de envasado automatizada: umbrales de precio, volumen y coste variable; escenarios; inflación 3%"
# coleccion: 17_viabilidad
# variante: sensibilidad
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
# Una empresa estudia invertir en una línea de envasado automatizada (datos adjuntos; mismas hipótesis contables que en el caso base: amortización lineal hasta el valor residual, circulante como fracción de las ventas aportado al principio de cada año y recuperado al final, valor residual cobrado por su valor contable). Antes de decidir, la dirección quiere saber *cuánto pueden equivocarse* las previsiones sin que el proyecto deje de ser viable.

# %%
pregunta_1 = "1. Calcule los flujos de caja libre y el VAN del caso base.  "
pregunta_2 = "2. Umbrales de rentabilidad: ¿qué precio mínimo, qué volumen mínimo del año 1 y qué coste variable máximo hacen el VAN igual a cero (manteniendo lo demás)? Exprese cada uno como variación porcentual respecto al caso base. ¿A qué variable es más sensible el proyecto?  "
pregunta_3 = "3. Escenarios: pesimista (precio −10 %, unidades −10 %, costes fijos +10 %), base y optimista (precio +10 %, unidades +10 %, costes fijos −10 %). VAN y TIR de cada uno.  "
pregunta_4 = "4. Los datos están en euros constantes. Si se espera una inflación del 3% anual que afecta por igual a precios y costes, ¿cómo deben construirse los flujos en euros corrientes y qué tasa nominal hay que usar para que el VAN no cambie? Compruébelo (relación de Fisher).  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del caso"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  parametro                                              , valor
    ('inversión inicial (k€)'                               , 1110 ),
    ('vida del proyecto (años)'                             , 6    ),
    ('precio (€ por miles de unidade)'                      , 3    ),
    ('miles de unidades vendidas el año 1'                  , 370  ),
    ('crecimiento anual de las unidades'                    , 0.08 ),
    ('coste variable unitario (€)'                          , 2    ),
    ('costes fijos anuales (k€)'                            , 120  ),
    ('valor residual al final (k€)'                         , 110  ),
    ('circulante necesario (fracción de las ventas del año)', 0.05 ),
    ('tipo impositivo'                                      , 0.25 ),
    ('tasa de descuento'                                    , 0.1  ),
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
def modelo(precio=precio, u1=u1, cv=cv, CF=CF, g=g, I0=I0, n=n, VR=VR, w=w, t_imp=t_imp, inflacion=0.0):
    '''Flujos de caja libre del modelo; con inflación, en euros corrientes (precios y costes crecen cada año).'''
    unidades = pd.Series([0] + [u1 * (1 + g) ** (t - 1) for t in range(1, n + 1)])
    infl = pd.Series([(1 + inflacion) ** t for t in range(n + 1)])
    ventas = unidades * precio * infl; EBITDA = ventas - unidades * cv * infl - pd.Series([0] + [CF] * n) * infl
    amort = pd.Series([0] + [(I0 - VR) / n] * n); BAI = EBITDA - amort
    imp = BAI.clip(lower=0) * t_imp; imp[0] = 0
    circ = pd.Series([w * ventas[t + 1] for t in range(n)] + [0])
    dcirc = circ.diff().fillna(circ[0])
    return (BAI - imp + amort - dcirc - pd.Series([I0] + [0] * n) + pd.Series([0] * n + [VR])).tolist()
VAN = van(modelo(), k)
Markdown(explica_van(modelo(), k) + '.\n\n' + explica_tir(modelo())
         + f'\n\n**Respuesta:** VAN del caso base = {VAN:.1f} k€ (TIR {tir(modelo()):.2%}): ' + ('viable.' if VAN > 0 else 'no viable.'))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Cada umbral es el valor de la variable x que hace VAN(x) = 0 con las demás fijas. El VAN es casi lineal en cada variable, así que el umbral se aproxima bien con una regla de tres: VAN(x) ≈ VAN_base + (x − x_base) · ΔVAN/Δx, donde ΔVAN/Δx se mide cambiando la variable un 1 %; después se afina por bisección y se comprueba que el VAN en el umbral es 0. La sensibilidad se lee en la variación porcentual que anula el VAN: cuanto menor, más frágil es el proyecto frente a esa variable.

# %%
from scipy.optimize import brentq
precio_umbral = brentq(lambda x: van(modelo(precio=x), k), precio * 0.2, precio * 2)
u1_umbral = brentq(lambda x: van(modelo(u1=x), k), u1 * 0.2, u1 * 2)
cv_umbral = brentq(lambda x: van(modelo(cv=x), k), cv * 0.2, max(cv * 5, precio * 0.99))
_base = {'precio': precio, 'unidades año 1': u1, 'coste variable': cv}
_arg = {'precio': 'precio', 'unidades año 1': 'u1', 'coste variable': 'cv'}
_umb = {'precio': precio_umbral, 'unidades año 1': u1_umbral, 'coste variable': cv_umbral}
filas = {}
for v, x0 in _base.items():
    pend = (van(modelo(**{_arg[v]: x0 * 1.01}), k) - VAN) / (x0 * 0.01)
    filas[v] = {'caso base': x0, 'ΔVAN por unidad de la variable': pend,
                'umbral ≈ base − VAN / pendiente': f'{x0:g} − {VAN:.1f}/({_num(pend, 2)}) = {x0 - VAN / pend:.3f}',
                'umbral exacto (VAN = 0)': _umb[v], 'VAN en el umbral': van(modelo(**{_arg[v]: _umb[v]}), k),
                'variación admisible': _umb[v] / x0 - 1}
umbrales = pd.DataFrame(filas).T
for c in umbrales.columns.drop('umbral ≈ base − VAN / pendiente'):
    umbrales[c] = umbrales[c].astype(float)
display(umbrales.style.format({'caso base': '{:.3f}', 'ΔVAN por unidad de la variable': '{:.2f}', 'umbral exacto (VAN = 0)': '{:.3f}',
                               'VAN en el umbral': '{:.4f}', 'variación admisible': '{:+.1%}'}))
mas_sensible = umbrales['variación admisible'].astype(float).abs().idxmin()
Markdown('**Respuesta:** ' + '; '.join(f'{v}: {"mínimo" if v != "coste variable" else "máximo"} {float(umbrales.loc[v, "umbral exacto (VAN = 0)"]):.3f} ({float(umbrales.loc[v, "variación admisible"]):+.1%} respecto a {_base[v]:g})' for v in umbrales.index)
         + f'. El proyecto es más sensible a la variable **{mas_sensible}**: basta una variación del {float(umbrales.loc[mas_sensible, "variación admisible"]):+.1%} para anular el VAN. '
         'El diagrama de araña muestra lo mismo: la recta más inclinada (en % de variación) es la de la variable más sensible.')

# %%
#| fig-cap: "VAN frente a la variación de cada variable (diagrama de araña)"
#| label: fig-arana
from matplotlib import pyplot as plt
xs = np.linspace(-0.3, 0.3, 25)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs * 100, [van(modelo(precio=precio * (1 + x)), k) for x in xs], label='precio')
ax.plot(xs * 100, [van(modelo(u1=u1 * (1 + x)), k) for x in xs], label='unidades')
ax.plot(xs * 100, [van(modelo(cv=cv * (1 + x)), k) for x in xs], label='coste variable')
ax.plot(xs * 100, [van(modelo(CF=CF * (1 + x)), k) for x in xs], label='costes fijos')
ax.plot(xs * 100, [van(modelo(I0=I0 * (1 + x)), k) for x in xs], label='inversión')
ax.axhline(0, color='grey'); ax.set_xlabel('variación respecto al caso base (%)'); ax.set_ylabel('VAN (k€)'); ax.legend(); plt.show()

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Cada escenario repite el modelo completo con los tres datos cambiados a la vez; la tabla da sus flujos de caja libre y, debajo, el VAN término a término y la TIR.

# %%
escenarios = {'pesimista': dict(precio=precio * 0.9, u1=u1 * 0.9, CF=CF * 1.1), 'base': {}, 'optimista': dict(precio=precio * 1.1, u1=u1 * 1.1, CF=CF * 0.9)}
display(pd.DataFrame({e: modelo(**kw) for e, kw in escenarios.items()}, index=pd.Index(range(n + 1), name='año')).T.round(1))
tabla_esc = pd.DataFrame({e: {'VAN (k€)': van(modelo(**kw), k), 'TIR': tir(modelo(**kw))} for e, kw in escenarios.items()}).T
display(tabla_esc.style.format({'VAN (k€)': '{:.1f}', 'TIR': '{:.2%}'}))
Markdown('\n'.join(f'- {e} (precio {kw.get("precio", precio):g}, unidades {kw.get("u1", u1):g}, costes fijos {kw.get("CF", CF):g}): ' + explica_van(modelo(**kw), k) + f'; TIR = {tir(modelo(**kw)):.2%}.'
                   for e, kw in escenarios.items() if e != 'base')
         + f'\n\n**Respuesta:** VAN y TIR: pesimista {tabla_esc.loc["pesimista", "VAN (k€)"]:.1f} k€ y {tabla_esc.loc["pesimista", "TIR"]:.2%}; '
         f'base {VAN:.1f} k€ y {tabla_esc.loc["base", "TIR"]:.2%}; optimista {tabla_esc.loc["optimista", "VAN (k€)"]:.1f} k€ y {tabla_esc.loc["optimista", "TIR"]:.2%}. '
         f'El escenario pesimista {"mantiene" if tabla_esc.loc["pesimista", "VAN (k€)"] > 0 else "destruye"} la viabilidad'
         f'{" (su TIR queda por encima" if tabla_esc.loc["pesimista", "TIR"] > k else " (su TIR queda por debajo"} de la tasa del {k:.0%}). '
         'Los escenarios combinan desviaciones simultáneas, que es lo que ocurre en la práctica (un mal año afecta a precio y volumen a la vez); '
         'el análisis de una variable cada vez las subestima.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# En euros corrientes, todo importe que sigue a los precios se multiplica en el año t por (1 + π)^t (ventas, costes variables y fijos y, con ellas, el circulante); la amortización y el valor residual, que dependen del precio de compra, no cambian. Esos flujos se descuentan a la tasa nominal, que incluye la inflación: relación de Fisher, (1 + k_nominal) = (1 + k_real)(1 + π).

# %%
inflacion = 0.03
k_nominal = (1 + k) * (1 + inflacion) - 1
flujos_corrientes = modelo(inflacion=inflacion)
VAN_corriente = van(flujos_corrientes, k_nominal)
VAN_corriente_mal = van(flujos_corrientes, k)
display(pd.DataFrame({'euros constantes': modelo(), 'euros corrientes': flujos_corrientes}, index=pd.Index(range(n + 1), name='año')).T.round(1))
Markdown(f'''k_nominal = (1 + {k})(1 + {inflacion}) − 1 = {k_nominal:.2%}. Flujos corrientes descontados a la tasa nominal: ''' + explica_van(flujos_corrientes, k_nominal) + f'''. **Respuesta:** los flujos corrientes se construyen inflando cada año precios y costes con (1 + {inflacion})^t y se descuentan al {k_nominal:.2%}. Con ello el VAN es {VAN_corriente:.1f} k€, frente a {VAN:.1f} con flujos constantes y tasa real: no coinciden exactamente porque no todo crece con la inflación: la amortización (y su ahorro fiscal) y el valor residual son importes nominales fijos, y el circulante crece con las ventas nominales; {"la inflación perjudica al proyecto" if VAN_corriente < VAN else "la inflación favorece ligeramente al proyecto"} vía impuestos y circulante ({VAN_corriente - VAN:+.1f} k€). El error grave sería mezclar: flujos corrientes con tasa real daría {VAN_corriente_mal:.1f} k€, un VAN inflado.''')
