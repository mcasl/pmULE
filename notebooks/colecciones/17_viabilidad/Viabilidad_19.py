# ---
# title: Viabilidad 19
# subtitle: "A y B (5 años) con conflicto VAN/TIR (Fisher 10.7%), C de 4 años; k = 6%"
# coleccion: 17_viabilidad
# variante: alternativas
# nivel: 2
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
# Una empresa debe elegir **una sola** de tres inversiones excluyentes (flujos de caja en k€ al final de cada año; el año 0 es la inversión). A y B duran 5 años; C dura 4 y puede repetirse indefinidamente en las mismas condiciones. La tasa de descuento es del 6%.

# %%
pregunta_1 = "1. Calcule el VAN, la TIR y el payback de A y de B. ¿Qué proyecto prefiere cada criterio?  "
pregunta_2 = "2. Explique por qué VAN y TIR pueden discrepar entre proyectos excluyentes. Calcule la tasa de Fisher (la que iguala los VAN) a partir de los flujos diferenciales A − B y dibuje los dos perfiles de VAN.  "
pregunta_3 = "3. ¿Para qué tasas de descuento se prefiere A y para cuáles B? ¿Cuál es la decisión correcta a la tasa dada y por qué manda el VAN?  "
pregunta_4 = "4. C tiene menor vida. Compare C con la mejor de A y B mediante la anualidad equivalente (o la cadena de repeticiones). ¿Cambia la decisión?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Flujos de caja de las alternativas (k€)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  year, A   , B   , C   
    (0   , -650, -650, -390),
    (1   , 120 , 250 , 150 ),
    (2   , 145 , 205 , 150 ),
    (3   , 185 , 175 , 150 ),
    (4   , 230 , 145 , 150 ),
    (5   , 290 , 120 , 0   ),
], columns=['year', 'A', 'B', 'C']).set_index('year')
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
k, n, m = 0.06, 5, 4
A, B = datos_enunciado['A'].tolist(), datos_enunciado['B'].tolist()
C = datos_enunciado['C'].tolist()[:m + 1]

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# VAN (valor actual neto) a la tasa de descuento, TIR (tasa interna de rentabilidad, la que anula el VAN) y periodo de recuperación (payback) de cada proyecto, con los cálculos:

# %%
Markdown('\n'.join(f'**{p}.** ' + explica_van(f, k) + '. ' + explica_tir(f) + f' Payback: {explica_payback(f)}; descontado: {explica_payback(f, k)}.\n'
                   for p, f in (('A', A), ('B', B))))

# %%
#| tbl-cap: "Criterios para A y B"
#| label: tbl-criterios
criterios = pd.DataFrame({p: {'VAN (k€)': van(f, k), 'TIR': tir(f), 'payback (años)': payback(f), 'payback descontado': payback(f, k)}
                          for p, f in (('A', A), ('B', B))}).T
criterios.style.format({'VAN (k€)': '{:.1f}', 'TIR': '{:.2%}', 'payback (años)': '{:.2f}', 'payback descontado': '{:.2f}'})

# %%
_v, _t, _p = criterios['VAN (k€)'].idxmax(), criterios['TIR'].idxmax(), criterios['payback (años)'].idxmin()
Markdown(f'**Respuesta:** el VAN prefiere {_v} ({criterios.loc[_v, "VAN (k€)"]:.1f} frente a {criterios["VAN (k€)"].min():.1f} k€); '
         f'la TIR prefiere {_t} ({criterios.loc[_t, "TIR"]:.2%} frente a {criterios["TIR"].min():.2%}); '
         f'el payback prefiere {_p} (recupera la inversión en {criterios.loc[_p, "payback (años)"]:.2f} años frente a {criterios["payback (años)"].max():.2f}). '
         + ('VAN y TIR discrepan.' if _v != _t else 'VAN y TIR coinciden.'))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# La TIR es una medida *relativa* (rentabilidad por euro y año) y supone la reinversión de los flujos a la propia TIR; el VAN es *absoluto* (euros de valor creado a la tasa de descuento). Cuando los proyectos tienen distinto perfil temporal (A cobra tarde, B cobra pronto) sus perfiles de VAN se cruzan: a la tasa de Fisher ambos valen lo mismo. Como VAN_A − VAN_B es el VAN de los flujos diferenciales A − B, la tasa de Fisher es la TIR de esos flujos.

# %%
diferencial = [a - b for a, b in zip(A, B)]
fisher = tir(diferencial)
display(pd.DataFrame({'A': A, 'B': B, 'A − B': diferencial}, index=pd.Index(range(n + 1), name='año')).T)
Markdown('Flujos A − B: ' + explica_tir(diferencial).replace('La TIR es la tasa que anula el VAN', 'Su TIR es la tasa que anula su VAN') + '\n\n'
         + f'**Respuesta:** tasa de Fisher = {fisher:.2%}; a esa tasa VAN_A = {van(A, fisher):.1f} k€ = VAN_B = {van(B, fisher):.1f} k€ (los perfiles se cortan en el gráfico).')

# %%
#| fig-cap: "Perfiles de VAN de A y B"
#| label: fig-perfiles
from matplotlib import pyplot as plt
ks = np.linspace(0, 0.4, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for nombre, f in (('A', A), ('B', B)):
    ax.plot(ks * 100, [van(f, x) for x in ks], label=f'{nombre} (TIR {tir(f):.1%})')
ax.axhline(0, color='grey'); ax.axvline(fisher * 100, color='grey', linestyle=':', label=f'Fisher {fisher:.1%}')
ax.axvline(k * 100, color='tab:green', linestyle='--', label=f'tasa de descuento {k:.0%}')
ax.set_xlabel('tasa (%)'); ax.set_ylabel('VAN (k€)'); ax.legend(); plt.show()

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Respuesta:** para tasas por debajo de la de Fisher ({fisher:.2%}) se prefiere A (sus flujos tardíos valen más cuando el descuento es suave); por encima, B (hasta su TIR, {tir(B):.2%}; por encima de ambas TIR ninguno es viable). A la tasa dada ({k:.0%} < {fisher:.2%}) la decisión correcta es **A**, la de mayor VAN ({van(A, k):.1f} frente a {van(B, k):.1f} k€): el objetivo es maximizar el valor creado en euros, y la TIR, al ser relativa e implicar reinversión a la propia TIR, no compara bien proyectos de distinto perfil temporal.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Un VAN de 4 años y uno de 6 no son comparables: el de menor vida deja libre el dinero antes y puede repetirse. La anualidad equivalente reparte cada VAN en una renta constante durante su vida, AE = VAN · k / [1 − (1 + k)^−n]; la cadena de repeticiones suma el VAN de cada repetición descontado hasta una vida común (el mínimo común múltiplo de las vidas).

# %%
VAN_C = van(C, k)
AE = {'A': anualidad_equivalente(van(A, k), k, n), 'B': anualidad_equivalente(van(B, k), k, n), 'C': anualidad_equivalente(VAN_C, k, m)}
import math
L = math.lcm(n, m)
VAN_C_cadena = sum(VAN_C / (1 + k) ** (j * m) for j in range(L // m))
VAN_A_cadena = sum(van(A, k) / (1 + k) ** (j * n) for j in range(L // n))
display(pd.DataFrame({'VAN (k€)': {'A': van(A, k), 'B': van(B, k), 'C': VAN_C}, 'vida': {'A': n, 'B': n, 'C': m},
                      'AE = VAN · k / [1 − (1 + k)^−n]': {p: f'{v:.1f} × {k} / [1 − {1 + k:g}^−{vida}] = {AE[p]:.2f}' for p, v, vida in (('A', van(A, k), n), ('B', van(B, k), n), ('C', VAN_C, m))}}).round({'VAN (k€)': 1}))
_cad = lambda v, vida: ' + '.join([f'{v:.1f}'] + [f'{v:.1f}/{1 + k:g}{str(j * vida).translate(_SUP)}' for j in range(1, L // vida)])
mejor_AB = 'A' if van(A, k) >= van(B, k) else 'B'
Markdown(f'VAN de C: ' + explica_van(C, k) + f'. Cadenas de repeticiones hasta {L} años: A, {_cad(van(A, k), n)} = {VAN_A_cadena:.1f} k€; '
         f'C, {_cad(VAN_C, m)} = {VAN_C_cadena:.1f} k€. '
         + (f'**Respuesta:** C pasa a ser la mejor opción: su anualidad equivalente ({AE["C"]:.1f} k€/año) supera la de {mejor_AB} ({AE[mejor_AB]:.1f}), aunque su VAN aislado sea menor.' if AE['C'] > max(AE['A'], AE['B']) else
            f'**Respuesta:** la decisión no cambia, sigue siendo {max(AE, key=AE.get)}: su anualidad equivalente ({max(AE.values()):.1f} k€/año) supera la de C ({AE["C"]:.1f}); la cadena de repeticiones da la misma conclusión.'))
