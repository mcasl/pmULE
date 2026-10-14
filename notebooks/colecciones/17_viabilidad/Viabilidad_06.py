# ---
# title: Viabilidad 06
# subtitle: "una flota de vehículos eléctricos de alquiler: I0 = 1390 k€, 5 años, k = 8%; VAN 227, TIR 13.1%"
# coleccion: 17_viabilidad
# variante: caso-base
# nivel: 1
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
# Una empresa estudia invertir en una flota de vehículos eléctricos de alquiler. Los datos del caso se adjuntan (importes en k€; precio y coste variable en € por unidad). La inversión se amortiza linealmente durante la vida del proyecto hasta su valor residual, que se recupera al final por su valor contable. El circulante necesario cada año es una fracción de las ventas de ese año, se aporta al principio del año (el del primer año, en el momento 0) y se recupera íntegramente al final.

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
    ('inversión inicial (k€)'                               , 1390 ),
    ('vida del proyecto (años)'                             , 5    ),
    ('precio (€ por miles de días de alquiler)'             , 71   ),
    ('miles de días de alquiler vendidas el año 1'          , 9    ),
    ('crecimiento anual de las unidades'                    , 0.08 ),
    ('coste variable unitario (€)'                          , 12   ),
    ('costes fijos anuales (k€)'                            , 185  ),
    ('valor residual al final (k€)'                         , 140  ),
    ('circulante necesario (fracción de las ventas del año)', 0.08 ),
    ('tipo impositivo'                                      , 0.25 ),
    ('tasa de descuento'                                    , 0.08 ),
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

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Descuento de los flujos"
#| label: tbl-descuento
descuento = tabla_descuento(flujos, k)
descuento.round(2)

# %%
VAN = van(flujos, k)
Markdown(f'''VAN = Σ FCL_t / (1 + {k})^t = ''' + ' + '.join(f'{f:.1f}/{(1 + k) ** t:.3f}' for t, f in enumerate(flujos)) + f''' = **{VAN:.1f} k€**. {("Es positivo: el proyecto devuelve la inversión, remunera el capital al " + f"{k:.0%}" + " y además crea " + f"{VAN:.0f}" + " k€ de valor; es viable.") if VAN > 0 else "Es negativo: no es viable a esa tasa."}''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
TIR = tir(flujos)
comprobacion = van(flujos, TIR)
Markdown(f'TIR = {TIR:.2%} (VAN a esa tasa: {comprobacion:.4f} ≈ 0). Es la rentabilidad del proyecto; como supera la tasa de descuento ({k:.0%}), '
         'el VAN es positivo: las dos reglas coinciden en un proyecto aislado con flujos convencionales (un desembolso y luego cobros).')

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

# %%
pb, pbd = payback(flujos), payback(flujos, k)
display(descuento[['flujo', 'acumulado', 'valor actual', 'acumulado descontado']].round(1))
Markdown(f'''Payback simple: el acumulado cambia de signo en el año {int(np.ceil(pb))}; interpolando, {pb:.2f} años. Payback descontado: {pbd:.2f} años (mayor, porque los flujos descontados son menores y tardan más en cubrir la inversión). El payback mide liquidez y riesgo de plazo (cuánto tiempo está el dinero expuesto), pero ignora los flujos posteriores a la recuperación y, en su versión simple, el valor del tiempo; por eso se usa como criterio complementario, nunca único.''')

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
    circ = pd.Series([w * ventas[1]] + [w * ventas[t] for t in range(1, n + 1)])
    dcirc = circ.diff().fillna(circ[0]); dcirc[n] -= circ[n]
    return (BAI - imp + amort - dcirc - pd.Series([I0] + [0] * n) + pd.Series([0] * n + [VR])).tolist()
VAN_bajo = van(modelo(u1_bajo, precio, cv, CF), k)
Markdown(f'''Con un 15 % menos de unidades el VAN pasa de {VAN:.1f} a {VAN_bajo:.1f} k€ ({(VAN_bajo - VAN) / VAN:+.0%}): {"sigue siendo viable" if VAN_bajo > 0 else "deja de ser viable"}. Las unidades vendidas (y el precio) son el supuesto más frágil: son previsiones de mercado, mientras que la inversión y los costes fijos están casi cerrados por contrato.''')
