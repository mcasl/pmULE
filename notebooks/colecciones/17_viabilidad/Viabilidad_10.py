# ---
# title: Viabilidad 10
# subtitle: "una línea de envasado automatizada: préstamo del 70% al 5% a 4 años; TIR proyecto 15.6%"
# coleccion: 17_viabilidad
# variante: financiacion
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
circulante = pd.Series([w * ventas[1]] + [w * ventas[t] for t in range(1, n + 1)], index=anos)   # el del año 1 se aporta en t = 0
variacion_circulante = circulante.diff().fillna(circulante[0]); variacion_circulante[n] -= circulante[n]   # se recupera al final
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
VAN, TIR = van(flujos, k), tir(flujos)
Markdown(f'VAN del proyecto = {VAN:.1f} k€ al {k:.0%}; TIR del proyecto = {TIR:.2%}.')

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
display(Markdown(f'Préstamo = {alfa:.0%} × {I0:.0f} = {prestamo:.1f} k€; cuota = {prestamo:.1f} × {i} / (1 − (1 + {i})^−{n_p}) = {cuadro["cuota"].iloc[0]:.2f} k€.'))
cuadro.round(2)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

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
Markdown(f'VAN para el accionista (al {ke:.0%}) = {VAN_acc:.1f} k€; TIR del accionista = {TIR_acc:.2%}.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
i_alto = 0.23
cuadro_alto = cuadro_prestamo(prestamo, i_alto, n_p)
fa_alto = flujo_accionista(cuadro_alto, prestamo)
coste_deuda_neto = i * (1 - t_imp); coste_alto_neto = i_alto * (1 - t_imp)
display(pd.DataFrame({'proyecto (sin deuda)': {'TIR': TIR}, f'accionista, préstamo al {i:.0%}': {'TIR': TIR_acc},
                      f'accionista, préstamo al {i_alto:.0%}': {'TIR': tir(fa_alto.tolist())}}).round(4))
Markdown(f'''La deuda cuesta {i:.0%} ({coste_deuda_neto:.2%} tras impuestos), menos que la rentabilidad del proyecto ({TIR:.2%}): la diferencia se la queda el accionista, que ha puesto sólo el {1 - alfa:.0%} del capital, y su TIR sube a {TIR_acc:.2%}. Es el apalancamiento financiero positivo. Con el préstamo al {i_alto:.0%} ({coste_alto_neto:.2%} neto, por encima de la TIR del proyecto) el efecto se invierte: la TIR del accionista {"cae a" if tir(fa_alto.tolist()) < TIR else "baja a"} {tir(fa_alto.tolist()):.2%}{", por debajo de la del proyecto" if tir(fa_alto.tolist()) < TIR else ""}. La condición es que la rentabilidad del proyecto supere el coste de la deuda después de impuestos. El apalancamiento amplifica en los dos sentidos, y además concentra el riesgo: las cuotas se pagan aunque las ventas fallen.''')
