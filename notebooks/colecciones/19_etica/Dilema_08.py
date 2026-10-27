# ---
# title: Dilema 08
# subtitle: "pagar a 90 días: +5.5 k€ para el contratista, −11.4 k€ para las subcontratas; máximo legal 60"
# coleccion: 19_etica
# variante: pago-subcontratistas
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
# Un contratista principal ejecuta el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y coste subcontratado de cada actividad (k€) se adjuntan; las subcontratas facturan a final de cada mes (cuatro semanas) la parte proporcional de lo ejecutado. El contrato con el cliente paga al contratista a 30 días.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, coste_subcontrata
    ('A'     , '---'      , 2       , 50               ),
    ('B'     , '---'      , 4       , 70               ),
    ('C'     , '---'      , 5       , 30               ),
    ('D'     , 'A,B'      , 2       , 30               ),
    ('E'     , 'C,D'      , 2       , 40               ),
    ('F'     , 'E'        , 1       , 30               ),
    ('G'     , 'E'        , 4       , 80               ),
    ('H'     , 'F,G'      , 4       , 110              ),
], columns=['activity', 'predecessor', 'duration', 'coste_subcontrata']).set_index('activity')
datos_enunciado

# %% [markdown]
# El coste de capital del contratista es del 8% anual; las subcontratas, pequeñas empresas, se financian al 18% anual.

# %% [markdown]
# > **De: Director financiero**
# > Pagaremos a las subcontratas a 90 días en vez de a 30: cobramos del cliente a 30 y nos financiamos con ellas dos meses. Mejora nuestro VAN y no cuesta nada; si alguna protesta, siempre hay otra que acepta.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C5 — Beneficio propio a costa de otro.** Obtener una ventaja (dinero, plazo, reputación) trasladando el coste a otra parte que no lo ha aceptado. *(cf. Código Ético del PMI, 5.3.2 y 3.3.2.)*
# - **C10 — Buena fe en la negociación.** No revelar la propia alternativa o el precio de reserva es legítimo; afirmar hechos falsos para presionar (una oferta inexistente) no lo es (C1). *(cf. Código Ético del PMI, 3.3.1 y 5.3.1; ICB4, Personas 9 «negociación».)*
# - **Norma — plazo de pago entre empresas: 30 días naturales si no se pacta otro y, por pacto, nunca más de 60** *(Ley 3/2004 de lucha contra la morosidad, art. 4.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Construya el calendario mensual de facturas de las subcontratas y calcule cuánto gana el contratista (en valor actual, a su coste de capital) pagando a 90 días en lugar de a 30, y cuánto pagando a 60.  "
pregunta_2 = "2. (A) Calcule lo que pierden las subcontratas (en valor actual, a su coste de financiación) en cada caso y el resultado neto para el conjunto. ¿Por qué no es un juego de suma cero?  "
pregunta_3 = "3. (B) Según la norma de referencia, ¿es legal pagar a 90 días? ¿Y a 60? ¿Qué criterio incumple, en cada caso, la propuesta tal como está formulada («si alguna protesta, siempre hay otra que acepta»)?  "
pregunta_4 = "4. (C) ¿Qué plazo de pago propondría y cómo lo justificaría ante el director financiero? Aplique la rúbrica.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
k_contratista, k_subcontratas, plazo_legal_maximo = 0.08, 0.18, 60
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = int(proyecto.duration(datos_proyecto['duration']))
m_c, m_s = (1 + k_contratista) ** (1 / 12) - 1, (1 + k_subcontratas) ** (1 / 12) - 1
meses = int(np.ceil(duracion / 4))
factura = pd.Series(0.0, index=pd.Index(range(1, meses + 1), name='mes'))
reparto = pd.DataFrame(0.0, index=datos_proyecto.index, columns=factura.index)
for a, fila in datos_proyecto.iterrows():
    es, d = int(calendario.loc[a, 'inicio_mas_temprano']), int(fila['duration'])
    for w in range(es, es + d):
        factura[w // 4 + 1] += fila['coste_subcontrata'] / d
        reparto.loc[a, w // 4 + 1] += fila['coste_subcontrata'] / d
def valor_actual(dias, m):
    '''Valor actual de las facturas pagadas «dias» días después de cada fin de mes, al tipo mensual m.'''
    return float(sum(f / (1 + m) ** (t + dias / 30) for t, f in factura.items()))

# %%
def porc(p, d=1):
    """Probabilidad o fracción como porcentaje: 0.453 -> «45.3 %»."""
    return f'{100 * float(p):.{d}f} %'

def tabla_md(filas, columnas):
    """Tabla Markdown a partir de una lista de filas."""
    esc = lambda x: str(x).replace('|', '\\|')
    lineas = ['| ' + ' | '.join(columnas) + ' |', '|' + '---|' * len(columnas)]
    lineas += ['| ' + ' | '.join(esc(x) for x in f) + ' |' for f in filas]
    return '\n'.join(lineas)

def explica_indicadores(ind, BAC=None, titulo=''):
    """CV, SV, CPI y SPI (y EAC y VAC si se da el BAC) con la fórmula y la sustitución."""
    PV, EV, AC = float(ind['PV']), float(ind['EV']), float(ind['AC'])
    lineas = [f'- CV = EV − AC = {EV:.1f} − {AC:.1f} = **{EV - AC:.1f}** k€',
              f'- SV = EV − PV = {EV:.1f} − {PV:.1f} = **{EV - PV:.1f}** k€',
              f'- CPI = EV / AC = {EV:.1f} / {AC:.1f} = **{EV / AC:.3f}**',
              f'- SPI = EV / PV = {EV:.1f} / {PV:.1f} = **{EV / PV:.3f}**']
    if BAC is not None:
        eac = BAC / (EV / AC)
        lineas += [f'- EAC = BAC / CPI = {BAC:.0f} / {EV / AC:.3f} = **{eac:.1f}** k€',
                   f'- VAC = BAC − EAC = {BAC:.0f} − {eac:.1f} = **{BAC - eac:.1f}** k€']
    return (f'{titulo}\n\n' if titulo else '') + '\n'.join(lineas)

def rubrica_caso(afectados, alternativas, decision):
    """La rúbrica del bloque C aplicada al caso: afectados con cifras, alternativas con las tres pruebas y decisión."""
    partes = ['**La rúbrica aplicada a este caso** (lo que debe contener una respuesta que la supere):', '',
              '*(1) Afectados y qué gana o pierde cada uno:*', '']
    partes += [f'- **{a}**: {b}' for a, b in afectados]
    partes += ['', '*(2) y (3) Alternativas y las tres pruebas:*', '',
               tabla_md(alternativas, ['alternativa', 'legalidad', 'publicidad', 'reversibilidad']), '',
               '*(4) Decisión y comunicación* (un camino defendible, no el único): ' + decision]
    return Markdown('\n'.join(partes))

def sin_ceros(valores):
    """Los términos no nulos de una suma (para escribirla sin «+ 0.0»)."""
    v = pd.Series(valores).astype(float)
    return v[v != 0]

def camino_critico_sigma(proyecto, duraciones, varianzas):
    """Camino crítico en orden de ejecución (el de mayor varianza si hay varios) y su desviación típica."""
    cal = proyecto.calendar(duraciones)
    rutas = [ordena_ruta([a for a in r if not str(a).startswith('@')], cal) for r in proyecto.critical_path(duraciones).values()]
    acts = max(rutas, key=lambda r: sum(float(varianzas[a]) for a in r))
    return acts, float(sum(float(varianzas[a]) for a in acts)) ** 0.5

def rutas_media_varianza(proyecto, duraciones, varianzas):
    """Todas las rutas en orden de ejecución con su duración media y su varianza, de más larga a más corta."""
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    filas = []
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        filas.append((' – '.join(acts), float(sum(float(duraciones[a]) for a in acts)), float(sum(float(varianzas[a]) for a in acts))))
    return pd.DataFrame(filas, columns=['ruta', 'media', 'varianza']).sort_values(['media', 'varianza'], ascending=False).reset_index(drop=True)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT del proyecto"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Con inicios más tempranos, cada actividad factura su coste repartido por igual entre sus semanas (coste / duración por semana); el mes 1 son las semanas 1–4, el mes 2 las semanas 5–8, y así sucesivamente. La tabla reparte el coste de cada actividad entre los meses; la fila «factura» es la suma de cada mes.

# %%
#| tbl-cap: "Facturación mensual de las subcontratas por actividad (k€)"
#| label: tbl-facturas
tabla_facturas = reparto.copy()
tabla_facturas['semanas (inicio–fin)'] = [(lambda i, f: f'{i}' if i == f else f'{i}–{f}')(int(calendario.loc[a, 'inicio_mas_temprano']) + 1, int(calendario.loc[a, 'fin_mas_temprano'])) for a in reparto.index]
tabla_facturas.loc['factura'] = list(factura) + ['']
limpia_tabla(tabla_facturas, 2)

# %%
def tabla_va(m, nombre):
    t = pd.DataFrame({'factura': factura})
    for dias in (30, 60, 90):
        t[f'factor a {dias} días = 1/(1+m)^(t+{dias // 30})'] = [1 / (1 + m) ** (mes + dias / 30) for mes in factura.index]
        t[f'VA a {dias} días'] = t['factura'] * t[f'factor a {dias} días = 1/(1+m)^(t+{dias // 30})']
    t.loc['Total'] = [factura.sum()] + [np.nan if c.startswith('factor') else t[c].sum() for c in t.columns[1:]]
    return t
va_c = tabla_va(m_c, 'contratista')
display(Markdown(f'''Tipo mensual equivalente del contratista: m = (1 + {k_contratista})^(1/12) − 1 = **{m_c:.5f}**. La factura del mes t pagada a d días se descuenta t + d/30 meses: VA = Σ factura_t / (1 + m)^(t + d/30). Valor actual de los pagos para el contratista:'''))
limpia_tabla(va_c, 4)

# %%
tabla = pd.DataFrame({dias: {'valor actual para el contratista': valor_actual(dias, m_c),
                             'valor actual para las subcontratas': valor_actual(dias, m_s)} for dias in (30, 60, 90)}).T
tabla.index.name = 'días de pago'
tabla['ganancia del contratista frente a 30'] = tabla.loc[30, 'valor actual para el contratista'] - tabla['valor actual para el contratista']
tabla['pérdida de las subcontratas frente a 30'] = tabla.loc[30, 'valor actual para las subcontratas'] - tabla['valor actual para las subcontratas']
tabla['neto conjunto'] = tabla['ganancia del contratista frente a 30'] - tabla['pérdida de las subcontratas frente a 30']
g = tabla['ganancia del contratista frente a 30']
vc = tabla['valor actual para el contratista']
Markdown(f'''Pagar más tarde reduce el valor actual de lo que paga el contratista; esa reducción es su ganancia:

- A 90 días: {vc[30]:.2f} − {vc[90]:.2f} = **{g[90]:.2f} k€**.
- A 60 días: {vc[30]:.2f} − {vc[60]:.2f} = **{g[60]:.2f} k€**.

**Respuesta.** Facturas mensuales de {lista_y([f'{v:.2f}' for v in factura])} k€ (total {factura.sum():.0f} k€). Pagar a 90 días en vez de a 30 vale {g[90]:.2f} k€ para el contratista y pagar a 60, {g[60]:.2f} k€: un {100 * g[90] / factura.sum():.1f} % y un {100 * g[60] / factura.sum():.1f} % de lo facturado.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
va_s = tabla_va(m_s, 'subcontratas')
display(Markdown(f'''Las subcontratas se financian al {porc(k_subcontratas, 0)}: m = (1 + {k_subcontratas})^(1/12) − 1 = **{m_s:.5f}**. Valor actual de los cobros para las subcontratas:'''))
limpia_tabla(va_s, 4)

# %%
#| tbl-cap: "Ganancia, pérdida y neto frente a pagar a 30 días (k€)"
#| label: tbl-neto
display(tabla.round(2))
pe = tabla['pérdida de las subcontratas frente a 30']
vs = tabla['valor actual para las subcontratas']
Markdown(f'''- Pérdida de las subcontratas a 90 días: {vs[30]:.2f} − {vs[90]:.2f} = **{pe[90]:.2f} k€**; a 60 días: {vs[30]:.2f} − {vs[60]:.2f} = **{pe[60]:.2f} k€**.
- Neto conjunto a 90 días: {g[90]:.2f} − {pe[90]:.2f} = **{g[90] - pe[90]:.2f} k€**; a 60 días: {g[60]:.2f} − {pe[60]:.2f} = **{g[60] - pe[60]:.2f} k€**.

No es un juego de suma cero porque cada parte valora el dinero aplazado a su propio coste de financiación: dos meses de aplazamiento valen para el contratista lo que le cuesta financiarse ({porc(k_contratista, 0)} anual) y le cuestan a la subcontrata lo que le cuesta a ella ({porc(k_subcontratas, 0)}). Mientras {porc(k_subcontratas, 0)} > {porc(k_contratista, 0)}, lo que pierde la subcontrata es más de lo que gana el contratista.

**Respuesta.** Las subcontratas pierden {pe[90]:.2f} k€ (90 días) y {pe[60]:.2f} k€ (60 días); el conjunto pierde {-(g[90] - pe[90]):.2f} k€ y {-(g[60] - pe[60]):.2f} k€: el aplazamiento no crea valor, traslada dinero de quien se financia caro a quien se financia barato y por el camino destruye una parte.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Norma** (Ley 3/2004, art. 4). Entre empresas, el plazo de pago es de 30 días si no se pacta otro y, por pacto, nunca más de {plazo_legal_maximo}. Pagar a 90 días **es ilegal**, aunque la subcontrata lo acepte: el pacto no puede superar el máximo. Pagar a 60 es **legal** si se pacta.

**Criterio C5 (beneficio propio a costa de otro).** La propuesta obtiene {g[90]:.2f} k€ trasladando a las subcontratas un coste mayor ({pe[90]:.2f} k€) que no han aceptado: «si alguna protesta, siempre hay otra que acepta» describe una imposición, no un acuerdo. **Criterio C10 (buena fe en la negociación).** Negociar el plazo es legítimo; imponerlo con la amenaza de sustituir a quien proteste, no; y decir que «no cuesta nada» es falso: cuesta {pe[90]:.2f} k€ a las subcontratas.

**Respuesta.** A 90 días: ilegal (más de {plazo_legal_maximo} días) y, además, C5 y C10. A 60 días: legal si se pacta libremente; formulado como en el enunciado (impuesto con la amenaza de cambiar de proveedor) sigue incumpliendo C5 (gana {g[60]:.2f} k€ a costa de {pe[60]:.2f} k€ ajenos) y C10.''')

# %%
preguntas_test = [
    ('El plazo máximo de pago entre empresas que admite la Ley 3/2004 por pacto es…', ['60 días', '30 días', '90 días', 'el que acepte el proveedor'],
     'Art. 4: 30 por defecto; por pacto, nunca más de 60.'),
    ('Aplazar el pago a proveedores que se financian más caro que el contratista…', ['destruye valor en el conjunto', 'es neutro: lo que uno gana lo pierde el otro', 'crea valor', 'no tiene efecto financiero'],
     'El contratista gana menos de lo que pierden las subcontratas.'),
    ('«Si alguna protesta, siempre hay otra que acepta» describe…', ['una imposición: C5 y C10', 'una negociación de buena fe', 'una reserva declarada', 'un conflicto de interés'],
     'Se traslada un coste a quien no lo ha aceptado libremente.'),
]
import os
def a_gift(preguntas_test, titulo, semilla=1):
    r = np.random.default_rng(semilla)
    lineas = [f'$CATEGORY: {titulo}', '']
    for i, (q, ops, fb) in enumerate(preguntas_test, 1):
        correcta = ops[0]
        todas = [ops[j] for j in r.permutation(len(ops))]
        cuerpo = ' '.join((f'={v}' if v == correcta else f'~{v}') for v in todas)
        lineas.append(f'::{titulo} P{i}::{q} {{{cuerpo} ####{fb}}}')
        lineas.append('')
    return '\n'.join(lineas)
gift = a_gift(preguntas_test, 'Ética 19 — pago a subcontratistas')
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_etica.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: (1) identificar a todos los afectados y qué pierde o gana cada uno; (2) formular al menos dos alternativas además de la que se propone en el enunciado; (3) someter cada alternativa a las tres pruebas —legalidad (¿cumple la norma?), publicidad (¿la defendería si se hiciera pública?) y reversibilidad (¿la aceptaría en el papel del afectado?)—; (4) decidir y redactar la comunicación con las cifras del bloque A. Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.

# %%
rubrica_caso(
    [('Subcontratas', f'pierden {pe[90]:.2f} k€ (a 90) o {pe[60]:.2f} k€ (a 60) en valor actual; tensiones de tesorería en empresas pequeñas'),
     ('Contratista', f'gana {g[90]:.2f} k€ (a 90) o {g[60]:.2f} k€ (a 60), y arriesga sanción, reclamaciones de intereses de demora y peores proveedores'),
     ('Cliente y proyecto', 'una subcontrata ahogada financieramente entrega peor o más tarde')],
    [('Pagar a 90 días impuesto (propuesta)', f'no: supera los {plazo_legal_maximo} días', 'no', 'no: la empresa no aceptaría que su cliente le pagara a 90'),
     ('Pagar a 30 días', 'sí', 'sí', 'sí'),
     (f'Pactar 60 días compensando a la subcontrata ({pe[60]:.2f} k€)', 'sí', 'sí', f'sí, pero la compensación ({pe[60]:.2f}) supera la ganancia ({g[60]:.2f}): no conviene a nadie'),
     ('Financiar el desfase de tesorería con el banco', 'sí', 'sí', 'sí')],
    f'''pagar a 30 días. Al director financiero: la ganancia del aplazamiento es pequeña ({g[90]:.2f} k€ a 90 días), menor que lo que pierden las subcontratas ({pe[90]:.2f} k€), y a 90 días además es ilegal; a 60 días no hay precio que compense a la subcontrata y deje ganancia al contratista, porque el aplazamiento no crea valor que repartir. Prueba de reversibilidad: ¿aceptaría la empresa que el cliente le pagara a 90?''')
