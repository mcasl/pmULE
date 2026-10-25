# ---
# title: Caso 17
# subtitle: "comité de inversión: una nave logística con reparto de última milla; VAN 340 k€ (33%), precio −12% con prob. 40%"
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
display(tabla_descuento(flujos, k).round(2))
Markdown(f'''VAN = {VAN:.1f} k€ al {k:.0%} ({VAN / I0:.1%} de la inversión; la regla pide el 10 %: {"pasa" if VAN >= 0.1 * I0 else "no pasa"}); TIR = {TIR:.2%}; recuperación simple {payback(flujos):.2f} años, descontada {payback(flujos, k):.2f}.''')

# %% [markdown]
# El caso base es claramente rentable: VAN 340 k€ (33% de la inversión; supera el 10 % de la regla), TIR 15.1% frente al 6% exigido. El consejero delegado pide además que no haya una probabilidad apreciable de VAN negativo, y eso es lo que los mensajes ponen en duda.

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
Markdown(f'La variable más sensible es **{mas_sensible}**: una variación del {umbrales.loc[mas_sensible, "variación admisible"]:+.1%} anula el VAN.')

# %% [markdown]
# Cada mensaje por separado:
#
# - caso base: VAN 340 k€ (+33% de la inversión).
# - precio −6%: VAN 116 k€ (+11% de la inversión).
# - precio −12%: VAN -110 k€ (-11% de la inversión).
# - volumen +10%: VAN 525 k€ (+50% de la inversión).
# - coste variable +10%: VAN 152 k€ (+15% de la inversión).
#
# El mensaje del director comercial tiene dos caras: su volumen convierte el proyecto en muy bueno, pero su escenario de precio lo hunde; el coste variable de producción se come buena parte del VAN. El proyecto es más sensible al precio que al volumen (el volumen arrastra costes variables; el precio va directo al margen), y el coste variable actúa como un precio negativo.

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
Markdown(f'''Con probabilidad {pi:.0%} de caída del precio: VAN esperado **{contrato.loc["sin contrato", "VAN esperado"]:.1f}** k€ sin contrato y **{contrato.loc["con contrato", "VAN esperado"]:.1f}** con él. El contrato cuesta {van_sin["mercado estable"] - van_con["mercado estable"]:.1f} k€ de VAN si el mercado aguanta y ahorra {van_con["cae el precio"] - van_sin["cae el precio"]:.1f} si cae.''')

# %% [markdown]
# El contrato es un seguro: se paga una prima cierta (89 k€ de VAN si el mercado aguanta) a cambio de recortar la pérdida en el escenario malo (92 k€ menos de pérdida). En valor esperado sale mejor **sin contrato** (160 frente a 143 k€): como casi todo seguro, la prima vale más que la pérdida esperada que evita (el VAN es casi lineal en el precio y la caída tiene probabilidad 40% < 50 %). Pero la regla del consejero mira también al peor caso: con el contrato el peor caso sigue siendo negativo (-18 k€ frente a -110 sin él). El 40 % que ofrece el cliente no basta para cubrir el riesgo de precio: cubre menos de la mitad del volumen. Una contraoferta razonable es ampliar el porcentaje contratado o acortar el descuento.

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
Markdown(f'''TIR del proyecto {TIR:.2%}; TIR del accionista con el préstamo {TIR_acc:.2%}. Coste de la deuda después de impuestos {i * (1 - t_imp):.2%}, {"menor" if i * (1 - t_imp) < TIR else "mayor"} que la TIR del proyecto: el apalancamiento {"amplifica la rentabilidad del accionista" if i * (1 - t_imp) < TIR else "la reduce"}.''')

# %% [markdown]
# La TIR del accionista (20.7%) es mayor que la del proyecto (15.1%) porque la deuda cuesta 4.5% neto, menos de lo que rinde el proyecto: el accionista pone el 50% del capital y se queda con la diferencia. Pero el apalancamiento amplifica en los dos sentidos: en el escenario de caída del precio las cuotas se pagan igual y la pérdida del accionista es proporcionalmente mayor. Dirección financiera tiene razón: la decisión de invertir se toma sobre el flujo de caja libre (sin deuda) y la de financiar, después; una TIR del accionista alta no convierte en bueno un proyecto malo.

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# Recomendación defendible: **aprobar condicionado a cerrar una cobertura de precio mayor que la ofrecida (contraoferta al cliente clave: más volumen contratado o menos descuento) y a revisar el coste variable con producción**. El caso base pasa la regla del 10 % (sí: 33%), pero con el riesgo de precio el VAN esperado baja a 160 k€ (15% de la inversión; pasa) y hay un 40% de probabilidad de VAN -110 k€, y la regla también mira eso. El volumen del director comercial no se incorpora al caso base (es una opinión sin contrato detrás) pero sí como escenario favorable. Mensajes:
#
# - *Director comercial:* «Tu volumen daría VAN 525 k€; tu riesgo de precio, -110. El segundo pesa más que el primero en la decisión: ¿puedes convertir parte del volumen extra en contratos firmados?»
# - *Director de producción:* «Con tu coste variable el VAN baja a 152 k€. Necesito la estimación documentada (proveedores, índices) para llevarla al caso base o como escenario.»
# - *Dirección financiera:* «TIR del accionista 20.7% frente a 15.1% del proyecto; de acuerdo en decidir sobre el FCL sin deuda. Si se aprueba, el préstamo mejora la rentabilidad del capital, con el riesgo que conlleva.»
# - *Cliente clave:* «Valoramos el contrato. El 40 % a −6% cubre poco riesgo para lo que cuesta; proponemos un 60 % con el mismo descuento, o el 40 % con un descuento menor.»
# - *Consejero delegado:* «VAN base 340 k€ (33% de la inversión); VAN esperado 160 sin cobertura y 143 con el contrato ofrecido; peor caso -110 sin cubrir y -18 con él. Recomiendo aprobar condicionado a una cobertura mayor.»

# %% [markdown]
# ## Apartado 6

# %%
display(Markdown(pregunta_6))

# %% [markdown]
# - **Hipótesis sin dueño.** El caso base lo preparó una persona; comercial y producción tenían información que lo cambia y la aportaron el día antes del comité. Las hipótesis deben recogerse de cada área con su justificación y su rango, antes de calcular.
# - **Riesgo tratado como opinión.** «Puede caer un X % con probabilidad π» es una hipótesis valiosa si se formaliza (escenarios con probabilidades, árbol de decisión, o simulación) en lugar de discutirse en la reunión.
# - **Coberturas.** Un contrato a precio fijo es un seguro con prima: se evalúa comparando el valor esperado y el peor caso con y sin él, y se negocia (porcentaje, descuento, duración).
# - **Inversión y financiación por separado.** La bondad del proyecto se juzga sobre el flujo de caja libre; la estructura financiera cambia el reparto entre accionista y banco y el riesgo, no el valor que crea el proyecto.
