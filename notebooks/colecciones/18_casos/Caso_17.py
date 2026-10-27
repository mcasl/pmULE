# ---
# title: Caso 17
# subtitle: "comité de inversión: una nave logística con reparto de última milla; VAN 339 k€ (33%), precio −12% con prob. 40%"
# coleccion: 18_casos
# variante: comite-inversion
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
# El comité de inversiones de su empresa estudia invertir en una nave logística con reparto de última milla. Usted ha preparado el caso base (datos adjuntos; mismas hipótesis que en la colección de viabilidad: amortización lineal hasta el valor residual, circulante como fracción de las ventas aportado al principio del año y recuperado al final, impuesto sobre el beneficio positivo). Antes de la reunión recibe estos mensajes.

# %%
#| tbl-cap: "Datos del caso base"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  parametro                                              , valor
    ('inversión inicial (k€)'                               , 1040 ),
    ('vida del proyecto (años)'                             , 5    ),
    ('precio (€ por miles de entrega)'                      , 4    ),
    ('miles de entregas vendidas el año 1'                  , 270  ),
    ('crecimiento anual de las unidades'                    , 0.05 ),
    ('coste variable unitario (€)'                          , 2    ),
    ('costes fijos anuales (k€)'                            , 245  ),
    ('valor residual al final (k€)'                         , 155  ),
    ('circulante necesario (fracción de las ventas del año)', 0.08 ),
    ('tipo impositivo'                                      , 0.25 ),
    ('tasa de descuento'                                    , 0.06 ),
], columns=['parametro', 'valor']).set_index('parametro')
datos_enunciado

# %% [markdown]
# > **De: Director comercial**
# > Vuestro volumen es conservador: con la red que tenemos vendemos un **10% más** el primer año, seguro. Lo que sí me preocupa es el precio: si entra el competidor que está sondeando el mercado, el precio puede caer un **12%** desde el primer año. Le doy a eso una probabilidad del 40%.

# %% [markdown]
# > **De: Director de producción**
# > Las materias primas están subiendo. Yo no contaría con el coste variable del caso base; pondría al menos un **10% más**. Lo demás me parece bien.

# %% [markdown]
# > **De: Dirección financiera**
# > El banco ofrece financiar el 50% de la inversión con un préstamo al 6% a 4 años, cuota constante. Quiero saber qué hace eso con la rentabilidad del accionista, y que no nos deslumbre: el proyecto tiene que ser bueno sin deuda.

# %% [markdown]
# > **De: Cliente clave**
# > Estamos dispuestos a firmar un contrato a 5 años por el **40 % del volumen** del caso base a un precio fijo un **6% inferior** al previsto, pase lo que pase en el mercado. Es nuestra oferta final.

# %% [markdown]
# > **De: Consejero delegado**
# > Regla de la casa: no aprobamos proyectos cuyo VAN esperado no supere el 10 % de la inversión, ni con una probabilidad apreciable de VAN negativo sin una cobertura. Quiero una recomendación clara el jueves.

# %% [markdown]
# Responda a las preguntas siguientes **antes** de decidir. En la penúltima se le pide la decisión y cómo la comunicaría a cada uno de los implicados (director comercial, director de producción, dirección financiera, cliente clave, consejero delegado).

# %%
pregunta_1 = "1. Calcule los flujos de caja libre del caso base, el VAN, la TIR y los periodos de recuperación simple y descontado. ¿Pasa la regla del consejero delegado?  "
pregunta_2 = "2. Evalúe por separado cada mensaje como escenario: precio −6% y −12%, volumen +10%, coste variable +10%. Calcule además los umbrales (precio mínimo, volumen mínimo, coste variable máximo con VAN = 0) y diga a qué variable es más sensible el proyecto.  "
pregunta_3 = "3. Con la probabilidad del 40% que da el director comercial a la caída del precio, calcule el VAN esperado sin el contrato del cliente clave y con él (40 % del volumen a precio −6% fijo; el resto a precio de mercado en cada escenario). ¿Cuánto cuesta la seguridad que ofrece el contrato y cuánto riesgo elimina?  "
pregunta_4 = "4. Construya el cuadro del préstamo y el flujo del accionista (con ahorro fiscal de los intereses). Compare la TIR del proyecto con la del accionista y explique el apalancamiento. ¿Tiene razón Dirección financiera en que el proyecto debe ser bueno sin deuda?  "
pregunta_5 = "5. Haga una recomendación al comité (aprobar, aprobar con contrato, rechazar, pedir más información) y redacte en tres o cuatro líneas la respuesta a cada implicado.  "
pregunta_6 = "6. ¿Qué fallos del proceso de evaluación revela el caso (quién aporta las hipótesis, cómo se tratan los riesgos de precio y coste, cómo se separa la decisión de inversión de la de financiación)?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

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
#| tbl-cap: "Flujos de caja libre del caso base (k€)"
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
def modelo(precio=precio, u1=u1, cv=cv, CF=CF, g=g, I0=I0, n=n, VR=VR, w=w, t_imp=t_imp):
    '''Flujos de caja libre del modelo con los parámetros que se indiquen (los demás, del caso base).'''
    unidades = pd.Series([0] + [u1 * (1 + g) ** (t - 1) for t in range(1, n + 1)])
    ventas = unidades * precio; EBITDA = ventas - unidades * cv - pd.Series([0] + [CF] * n)
    amort = pd.Series([0] + [(I0 - VR) / n] * n); BAI = EBITDA - amort
    imp = BAI.clip(lower=0) * t_imp; imp[0] = 0
    circ = pd.Series([w * ventas[1]] + [w * ventas[t] for t in range(1, n + 1)])
    dcirc = circ.diff().fillna(circ[0]); dcirc[n] -= circ[n]
    return (BAI - imp + amort - dcirc - pd.Series([I0] + [0] * n) + pd.Series([0] * n + [VR])).tolist()
VAN, TIR = van(flujos, k), tir(flujos)
td = tabla_descuento(flujos, k)
display(td.round(2))
def explica_payback(col):
    t = int((td[col] >= 0).idxmax()); prev = float(td.loc[t - 1, col]); f = float(td.loc[t, col] - td.loc[t - 1, col])
    return f'{t - 1} + {-prev:.1f} / {f:.1f} = {t - 1 + (-prev) / f:.2f} años'
display(Markdown('VAN = Σ FCLₜ·(1 + k)⁻ᵗ = ' + ' + '.join(f'({v:.1f})' for v in td['valor actual']) + f' = **{VAN:.1f} k€**. '
                 f'La TIR es la tasa que anula el VAN: con k = {TIR:.4f}, VAN = {van(flujos, TIR):.2f} ≈ 0. '
                 'Recuperación (año en que el acumulado se hace positivo, interpolando dentro del año: años completos + lo que falta / flujo del año): '
                 f'simple {explica_payback("acumulado")}; descontada {explica_payback("acumulado descontado")}.'))
Markdown(f'''VAN = {VAN:.1f} k€ al {k:.0%} ({VAN / I0:.1%} de la inversión; la regla pide el 10 %: {"pasa" if VAN >= 0.1 * I0 else "no pasa"}); TIR = {TIR:.2%}; recuperación simple {payback(flujos):.2f} años, descontada {payback(flujos, k):.2f}.''')

# %% [markdown]
# **Respuesta:** el caso base es claramente rentable: VAN 338.9 k€, que es 338.9 / 1040 = 32.6% de la inversión, así que pasa la primera parte de la regla (VAN ≥ 10 % de la inversión); TIR 15.1% frente al 6% exigido. El consejero delegado pide además que no haya una probabilidad apreciable de VAN negativo, y eso es lo que los mensajes ponen en duda.

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Escenarios de cada mensaje"
#| label: tbl-escenarios
x, y, z = 0.06, 0.1, 0.1
escenarios = {'caso base': modelo(), f'precio −{x:.0%}': modelo(precio=precio * (1 - x)), f'precio −{2 * x:.0%}': modelo(precio=precio * (1 - 2 * x)),
              f'volumen +{y:.0%}': modelo(u1=u1 * (1 + y)), f'coste variable +{z:.0%}': modelo(cv=cv * (1 + z))}
tabla_esc = pd.DataFrame({nombre: {'VAN (k€)': van(f, k), 'TIR': tir(f), 'VAN / inversión': van(f, k) / I0} for nombre, f in escenarios.items()}).T
tabla_esc.round(3)

# %%
#| tbl-cap: "Umbrales (VAN = 0) y variación admisible"
#| label: tbl-umbrales
from scipy.optimize import brentq
precio_umbral = brentq(lambda v: van(modelo(precio=v), k), precio * 0.2, precio * 2)
u1_umbral = brentq(lambda v: van(modelo(u1=v), k), u1 * 0.2, u1 * 2)
cv_umbral = brentq(lambda v: van(modelo(cv=v), k), cv * 0.2, max(cv * 5, precio * 0.99))
umbrales = pd.DataFrame({'caso base': [precio, u1, cv], 'umbral (VAN = 0)': [precio_umbral, u1_umbral, cv_umbral]},
                        index=['precio', 'unidades año 1', 'coste variable'])
umbrales['variación admisible'] = umbrales['umbral (VAN = 0)'] / umbrales['caso base'] - 1
display(umbrales.round(3))
mas_sensible = umbrales['variación admisible'].abs().idxmin()
Markdown(f'**Respuesta:** la variable más sensible es **{mas_sensible}**: una variación del {umbrales.loc[mas_sensible, "variación admisible"]:+.1%} anula el VAN.')

# %% [markdown]
# Cada mensaje por separado:
#
# - caso base: VAN 339 k€ (+33% de la inversión).
# - precio −6%: VAN 115 k€ (+11% de la inversión).
# - precio −12%: VAN -111 k€ (-11% de la inversión).
# - volumen +10%: VAN 524 k€ (+50% de la inversión).
# - coste variable +10%: VAN 151 k€ (+15% de la inversión).
#
# El mensaje del director comercial tiene dos caras: su volumen convierte el proyecto en muy bueno, pero su escenario de precio lo hunde; el coste variable de producción se come buena parte del VAN. El proyecto es más sensible al precio que al volumen: un 1 % de precio cambia los ingresos de cada unidad en 0.04 y todo eso va al margen, mientras que un 1 % de volumen sólo añade el margen de contribución de esas unidades (precio − coste variable = 4 − 2 = 2 por unidad); el coste variable actúa como un precio negativo. Por eso la variación admisible del precio (umbrales) es la más pequeña.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "VAN esperado sin y con el contrato del cliente clave"
#| label: tbl-contrato
pi = 0.4
# sin contrato: todo el volumen a precio de mercado (base, o −2x si entra el competidor)
van_sin = {'mercado estable': van(modelo(), k), 'cae el precio': van(modelo(precio=precio * (1 - 2 * x)), k)}
# con contrato: 40 % a precio fijo (1 − x); el 60 % restante a mercado. Precio medio = 0.4·(1 − x)·p + 0.6·p_mercado
van_con = {'mercado estable': van(modelo(precio=precio * (0.4 * (1 - x) + 0.6)), k),
           'cae el precio': van(modelo(precio=precio * (0.4 * (1 - x) + 0.6 * (1 - 2 * x))), k)}
contrato = pd.DataFrame({'sin contrato': van_sin, 'con contrato': van_con}).T
contrato['VAN esperado'] = (1 - pi) * contrato['mercado estable'] + pi * contrato['cae el precio']
contrato['peor caso'] = contrato['cae el precio']
display(contrato.round(1))
display(Markdown(f'Precio medio con el contrato = 0.4·(1 − {x:g})·p + 0.6·p_mercado: {0.4 * (1 - x) + 0.6:.3f}·p si el mercado aguanta y '
                 f'{0.4 * (1 - x) + 0.6 * (1 - 2 * x):.3f}·p si cae un {2 * x:.0%}. VAN esperado = (1 − π)·VAN(estable) + π·VAN(cae):\n\n'
                 + '\n'.join(f'- {nombre}: {1 - pi:g} × {fila["mercado estable"]:.1f} + {pi:g} × ({fila["cae el precio"]:.1f}) = **{fila["VAN esperado"]:.1f} k€**'
                             for nombre, fila in contrato.iterrows())))
Markdown(f'''**Respuesta:** con probabilidad {pi:.0%} de caída del precio, VAN esperado **{contrato.loc["sin contrato", "VAN esperado"]:.1f}** k€ sin contrato y **{contrato.loc["con contrato", "VAN esperado"]:.1f}** con él. El contrato cuesta {van_sin["mercado estable"] - van_con["mercado estable"]:.1f} k€ de VAN si el mercado aguanta y ahorra {van_con["cae el precio"] - van_sin["cae el precio"]:.1f} si cae.''')

# %% [markdown]
# El contrato es un seguro: se paga una prima (89.4 k€ de VAN si el mercado aguanta) a cambio de recortar la pérdida en el escenario malo (92.1 k€ menos de pérdida). En valor esperado, la prima cuesta 0.6 × 89.4 = 53.6 k€ y el ahorro vale 0.4 × 92.1 = 36.9 k€, así que sale mejor **sin contrato** (159.0 frente a 142.2 k€): como en casi todo seguro, la prima esperada supera a la pérdida esperada que evita (la caída tiene probabilidad 40% < 50 % y prima y ahorro son de tamaño parecido). Pero la regla del consejero mira también al peor caso: con el contrato el peor caso sigue siendo negativo (-19 k€ frente a -111 sin él). El 40 % que ofrece el cliente no basta para cubrir el riesgo de precio. El VAN es nulo con un precio un 9.1% menor que el previsto (precio umbral del apartado 2); el peor caso sólo deja de ser negativo si el precio medio no baja de ese umbral. Para lograrlo habría que contratar al menos el 48% del volumen con el mismo descuento del 6% (despejando f en f·(1 − 0.06) + (1 − f)·(1 − 0.12) ≥ 0.909), o el 40 % con un descuento de como mucho el 4.7%.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
alfa, i, n_p = 0.5, 0.06, 4
def cuadro_prestamo(P, i, n_p):
    C = P * i / (1 - (1 + i) ** -n_p)
    vivo, filas = P, []
    for t in range(1, n_p + 1):
        interes = vivo * i; amort = C - interes; vivo -= amort
        filas.append({'año': t, 'cuota': C, 'intereses': interes, 'amortización': amort, 'capital vivo': max(vivo, 0)})
    return pd.DataFrame(filas).set_index('año')
def flujo_accionista(cuadro, prestamo):
    f = pd.Series(flujos, index=range(n + 1)).astype(float)
    f[0] += prestamo
    for t, fila in cuadro.iterrows():
        f[t] += -fila['cuota'] + t_imp * fila['intereses']
    return f
prestamo = alfa * I0
cuadro = cuadro_prestamo(prestamo, i, n_p)
display(cuadro.round(2))
fa = flujo_accionista(cuadro, prestamo)
display(pd.DataFrame({'FCL proyecto': flujos, 'flujo del accionista': fa.tolist()}, index=pd.Index(range(n + 1), name='año')).round(1).T)
TIR_acc = tir(fa.tolist())
display(Markdown(f'Cuota constante C = P·i / (1 − (1 + i)⁻ⁿ) = {prestamo:.1f} × {i:g} / (1 − {1 + i:g}^−{n_p}) = **{cuadro.cuota.iloc[0]:.2f} k€**. '
                 f'Flujo del accionista: en el año 0, FCL + préstamo = {flujos[0]:.1f} + {prestamo:.1f} = {fa[0]:.1f}; en el año 1, '
                 f'FCL − cuota + {t_imp:g} × intereses = {flujos[1]:.1f} − {cuadro.cuota.iloc[0]:.2f} + {t_imp:g} × {cuadro.intereses.iloc[0]:.2f} = {fa[1]:.1f} '
                 '(los intereses se deducen del impuesto: es el ahorro fiscal); igual en los demás años del préstamo.'))
Markdown(f'''TIR del proyecto {TIR:.2%}; TIR del accionista con el préstamo {TIR_acc:.2%}. Coste de la deuda después de impuestos {i * (1 - t_imp):.2%}, {"menor" if i * (1 - t_imp) < TIR else "mayor"} que la TIR del proyecto: el apalancamiento {"amplifica la rentabilidad del accionista" if i * (1 - t_imp) < TIR else "la reduce"}.''')

# %% [markdown]
# La TIR del accionista (20.6%) es mayor que la del proyecto (15.1%) porque la deuda cuesta 6% × (1 − 0.25) = 4.50% neto, menos de lo que rinde el proyecto: el accionista pone sólo el 50% de la inversión y se queda con la diferencia. Pero el apalancamiento amplifica en los dos sentidos: en el escenario de caída del precio las cuotas se pagan igual y la pérdida del accionista es proporcionalmente mayor. Dirección financiera tiene razón: la decisión de invertir se toma sobre el flujo de caja libre (sin deuda) y la de financiar, después; una TIR del accionista alta no convierte en bueno un proyecto malo.
#
# **Respuesta:** TIR del proyecto 15.07% y del accionista 20.55%; sí, Dirección financiera tiene razón: el proyecto se juzga sin deuda.

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# Recomendación defendible: **aprobar condicionado a cerrar una cobertura de precio mayor que la ofrecida (contraoferta al cliente clave: al menos el 48% del volumen a −6%, o el 40 % a −4.7% como mucho) y a revisar el coste variable con producción**. El caso base pasa la regla del 10 % (sí: 33%), pero con el riesgo de precio el VAN esperado baja a 159 k€ (15% de la inversión; pasa) y hay un 40% de probabilidad de VAN -111 k€, y la regla también mira eso. El volumen del director comercial no se incorpora al caso base (es una opinión sin contrato detrás) pero sí como escenario favorable. Mensajes:
#
# - *Director comercial:* «Tu volumen daría VAN 524 k€; tu riesgo de precio, -111. El segundo pesa más que el primero en la decisión: ¿puedes convertir parte del volumen extra en contratos firmados?»
# - *Director de producción:* «Con tu coste variable el VAN baja a 151 k€. Necesito la estimación documentada (proveedores, índices) para llevarla al caso base o como escenario.»
# - *Dirección financiera:* «TIR del accionista 20.6% frente a 15.1% del proyecto; de acuerdo en decidir sobre el FCL sin deuda. Si se aprueba, el préstamo mejora la rentabilidad del capital, con el riesgo que conlleva.»
# - *Cliente clave:* «Valoramos el contrato. El 40 % a −6% cubre poco riesgo para lo que cuesta; proponemos el 50% del volumen con el mismo descuento, o el 40 % con un descuento del 4.5%.»
# - *Consejero delegado:* «VAN base 339 k€ (33% de la inversión); VAN esperado 159 sin cobertura y 142 con el contrato ofrecido; peor caso -111 sin cubrir y -19 con él. Recomiendo aprobar condicionado a una cobertura mayor.»

# %% [markdown]
# ## Apartado 6

# %%
display(Markdown(pregunta_6))

# %% [markdown]
# - **Hipótesis sin dueño.** El caso base lo preparó una persona; comercial y producción tenían información que lo cambia y la aportaron el día antes del comité. Las hipótesis deben recogerse de cada área con su justificación y su rango, antes de calcular.
# - **Riesgo tratado como opinión.** «Puede caer un X % con probabilidad π» es una hipótesis valiosa si se formaliza (escenarios con probabilidades, árbol de decisión, o simulación) en lugar de discutirse en la reunión.
# - **Coberturas.** Un contrato a precio fijo es un seguro con prima: se evalúa comparando el valor esperado y el peor caso con y sin él, y se negocia (porcentaje, descuento, duración).
# - **Inversión y financiación por separado.** La bondad del proyecto se juzga sobre el flujo de caja libre; la estructura financiera cambia el reparto entre accionista y banco y el riesgo, no el valor que crea el proyecto.
