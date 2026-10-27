# ---
# title: Caso 11
# subtitle: "límite de 3 operarios, plazo mes 16, penalización 6 k€/mes y VAN"
# coleccion: 18_casos
# variante: contingencia-planta
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
# Usted dirige la construcción de una planta cuyo cuadro de prelaciones, duraciones (meses; *a*, *m* y *b* son las estimaciones mínima, más probable y máxima de una distribución triangular y *duration* la duración esperada), operarios por actividad, presupuesto (BAC, k€), duración tope y coste unitario de reducción (CUR, k€ por mes) se adjuntan. La cuadrilla fija es de **3 operarios**. Cuando la planta se entrega, genera un flujo neto de 50 k€ al mes durante 36 meses; la empresa descuenta al 10% anual. El viernes recibe estos mensajes.

# %%
#| tbl-cap: "Datos del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, a, m, b, operarios, BAC, cap_duration, ucr   
    ('A'     , '---'      , 3       , 2, 3, 4, 3        , 175, 2           , 2     ),
    ('B'     , '---'      , 1       , 1, 1, 1, 1        , 200, 1           , np.inf),
    ('C'     , '---'      , 2       , 1, 2, 3, 3        , 175, 2           , np.inf),
    ('D'     , 'A,C'      , 3       , 1, 4, 4, 1        , 200, 2           , 4     ),
    ('E'     , 'B,D'      , 4       , 3, 3, 6, 2        , 75 , 3           , 9     ),
    ('F'     , 'C'        , 5       , 4, 5, 6, 1        , 75 , 5           , np.inf),
    ('G'     , 'F'        , 1       , 1, 1, 1, 1        , 200, 1           , np.inf),
    ('H'     , 'G'        , 1       , 1, 1, 1, 2        , 125, 1           , np.inf),
    ('I'     , 'E,H'      , 5       , 3, 5, 7, 2        , 150, 4           , 8     ),
], columns=['activity', 'predecessor', 'duration', 'a', 'm', 'b', 'operarios', 'BAC', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# > **De: Cliente**
# > El contrato fija la entrega en el mes 16 con una penalización de **6 k€ por mes de retraso**. Les pedimos que nos confirmen la probabilidad real de cumplirlo, no la fecha del plan.

# %% [markdown]
# > **De: Director financiero**
# > No autorizaré ninguna medida cuyo VAN esperado, neto de penalizaciones y de lo que cueste, no supere al de no hacer nada. Y quiero saber la probabilidad de que el VAN sea negativo en cada opción.

# %% [markdown]
# > **De: Jefe de obra**
# > Podemos subcontratar **un operario más** durante todo el proyecto por 2 k€ al mes. Con la cuadrilla actual, por mucho que acorten las actividades, a veces se quedan esperando por falta de gente.

# %% [markdown]
# > **De: Contratista**
# > Podemos acelerar actividades con horas extra y refuerzos a los CUR indicados en la tabla (hasta la duración tope). Díganos cuántos meses necesita recortar y le decimos qué actividades.

# %% [markdown]
# Responda a las preguntas siguientes **antes** de decidir. En la penúltima se le pide la decisión y cómo la comunicaría a cada uno de los implicados (cliente, director financiero, jefe de obra, contratista).

# %%
pregunta_1 = "1. Calcule la duración con las duraciones esperadas sin límite de recursos y con el límite de operarios. ¿Cuál es la cadena crítica con recursos y qué actividades esperan?  "
pregunta_2 = "2. Simule 3 000 ejecuciones con duraciones triangulares y con el límite de operarios (replanificando en cada réplica) y estime la probabilidad de entregar en el mes 16, el VAN esperado neto de penalización y P(VAN < 0) si no se hace nada.  "
pregunta_3 = "3. Evalúe cuatro opciones con los mismos números aleatorios: (A) no hacer nada, (B) contratar un operario más, (C) acelerar dos meses con el contratista (reducción con recursos limitados) y (D) las dos cosas a la vez. Tabla de VAN esperado neto, P(VAN < 0), probabilidad de cumplir el plazo, duración media y coste de la aceleración.  "
pregunta_4 = "4. ¿Qué opción recomienda al director financiero? ¿Y al cliente? ¿Se suman los efectos de B y C en D? Explique por qué.  "
pregunta_5 = "5. Redacte en tres o cuatro líneas la respuesta a cada implicado.  "
pregunta_6 = "6. ¿Qué ha cambiado respecto a un análisis determinista con el camino crítico? Señale al menos tres decisiones que habría tomado mal sin recursos, sin incertidumbre o sin VAN.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
def proyecto_de(predecesoras, duraciones, nombres):
    '''Tabla de datos y proyecto pmULE a partir de listas de predecesoras y duraciones.'''
    df = pd.DataFrame({'predecessor': list(predecesoras), 'duration': list(duraciones)}, index=list(nombres))
    df.index.name = 'activity'
    p = PredecessorTable.from_dataframe_of_strings(df, activity='activity', predecessor='predecessor').create_project()
    return df, p

def camino_critico(p, d):
    return '; '.join('–'.join(ruta) for ruta in p.critical_path(d).values())

def resumen_prob(p, d, sd, plazo):
    '''Duración media, desviación típica del camino crítico (la mayor si hay varios), P(T ≤ plazo) y retraso esperado.'''
    R = p.path_matrix(dummies=False)
    L = R @ d.reindex(R.columns)
    V = R @ (sd.reindex(R.columns).fillna(0) ** 2)
    T = float(L.max()); sigma = float(np.sqrt(V[L == T].max()))
    z = (plazo - T) / sigma
    retraso_esp = sigma * (norm.pdf(z) - z * norm.cdf(-z))      # E[máx(0, X − plazo)] con X ~ N(T, σ)
    return T, sigma, float(norm.cdf(z)), float(retraso_esp)

def suma_caminos_criticos(p, d, unidad='semanas'):
    '''Cada camino crítico, en orden de ejecución, escrito como suma de duraciones.'''
    cal = p.calendar(d)
    T = float(p.duration(d))
    return '; '.join('–'.join(ordena_ruta(r, cal)) + ' = ' + ' + '.join(f'{float(d[a]):g}' for a in ordena_ruta(r, cal)) + f' = {T:g} {unidad}'
                     for r in p.critical_path(d).values())

def pasos_ackoff(p, d, tope, cur, pasos):
    '''Paso a paso de Ackoff-Sasieni: caminos críticos, opciones que los cortan todos (con su CUR) y la elegida.'''
    from itertools import combinations
    R = p.path_matrix(dummies=False)
    d = d.astype(float).copy(); filas = []; acumulado = 0.0
    for paso, elegidas in sorted(pasos.items()):
        L = R @ d[R.columns]; T = float(L.max())
        cal = p.calendar(d)
        criticas = [ordena_ruta(list(R.columns[R.loc[r] == 1]), cal) for r in L.index if np.isclose(L[r], T)]
        conjuntos = [set(c) for c in criticas]
        margen = {x for x in R.columns if d[x] > tope[x] and np.isfinite(cur[x])}
        opciones = {x: float(cur[x]) for x in set.intersection(*conjuntos) & margen}
        if len(conjuntos) > 1:
            for a, b in combinations(sorted(set().union(*conjuntos) & margen), 2):
                if all(a in s or b in s for s in conjuntos) and a not in opciones and b not in opciones:
                    opciones[f'{a}+{b}'] = float(cur[a] + cur[b])
        coste = float(sum(cur[x] for x in elegidas)); acumulado += coste
        filas.append({'paso': paso + 1, 'duración antes': int(T), 'caminos críticos': '; '.join('–'.join(c) for c in criticas),
                      'opciones con margen (CUR, k€)': '; '.join(f'{k} ({v:g})' for k, v in sorted(opciones.items(), key=lambda kv: kv[1])),
                      'se acorta': ' + '.join(elegidas), 'coste del paso (k€)': coste, 'coste acumulado (k€)': acumulado,
                      'duración después': int(T) - 1})
        for x in elegidas:
            d[x] -= 1
    return pd.DataFrame(filas).set_index('paso')

def explica_retraso_esperado(media, sigma, plazo, unidad='semanas'):
    '''E[máx(0, T − plazo)] con T ~ N(media, σ), sin tipificar: (μ − plazo)·P(T > plazo) + σ²·f(plazo).'''
    p_mas = 1 - norm.cdf(plazo, media, sigma); f = norm.pdf(plazo, media, sigma)
    hoja = lambda x: f'{x:g}'.replace('.', ',')
    return (f'E[máx(0, T − {plazo:g})] = (μ − plazo)·P(T > plazo) + σ²·f(plazo) = ({media:g} − {plazo:g})·{p_mas:.4f} + {sigma:.3f}²·{f:.4f} '
            f'= **{(media - plazo) * p_mas + sigma ** 2 * f:.2f} {unidad}**, donde f(plazo) es la densidad normal en el plazo: '
            f'`=DISTR.NORM.N({hoja(plazo)}; {hoja(media)}; {hoja(round(sigma, 3))}; FALSO)` = {f:.4f}')

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
L, plazo, penalizacion, coste_operario, F, N, k = 3, 16, 6.0, 2.0, 50.0, 36, 0.1
base = datos_proyecto[['predecessor', 'duration', 'operarios']]
tope, ucr = datos_proyecto['cap_duration'], datos_proyecto['ucr']
dist_base = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con las duraciones esperadas y el camino crítico (sin límite de operarios)"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario sin límite de recursos (meses)"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %%
#| fig-cap: "Carga de operarios con los inicios más tempranos frente al límite"
#| label: fig-cargasInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(base.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[L])

# %% [markdown]
# El algoritmo de asignación recorre los meses: cada actividad disponible (predecesoras terminadas) empieza, por orden de prioridad (menor inicio más tardío), si caben sus operarios durante toda su duración; si no, espera. Traza mes a mes y Gantt resultante (filas 💤: esperas):

# %%
res0 = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], [L], report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios y las duraciones esperadas"
#| label: fig-ganttAsignado
pinta_tikz(res0.dibujos['operarios'])

# %%
#| tbl-cap: "Inicio de cada actividad sin y con límite"
#| label: tbl-inicios
pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res0.inicios, 'fin con límite': res0.fines,
              'espera (meses)': res0.esperas}).fillna(0).astype(int).T

# %%
duracion_libre = proyecto.duration(datos_proyecto['duration'])
cadena = proyecto.copy().cadena_critica_recursos(base.copy(), 'duration', ['operarios'], [L])
display(cadena['arcos'].rename(columns={'motivo': 'motivo (precedencia o recurso)'}))
_esp = res0.esperas[res0.esperas > 0]
_cadena = sorted(cadena['actividades'], key=lambda a: (res0.inicios[a], a))
Markdown(f'**Respuesta:** sin límite de operarios el proyecto dura {suma_caminos_criticos(proyecto, datos_proyecto["duration"], "meses")}; '
         f'con {L} operarios dura **{res0.duracion} meses** ({res0.duracion - duracion_libre:g} más). La cadena crítica con recursos '
         f'(las actividades que fijan esa duración, encadenadas por precedencias o porque una espera a que otra libere '
         f'operarios: arcos «recurso» de la tabla) es {"–".join(_cadena)}. Esperan por falta de operarios: '
         + lista_y([f'{a} ({plural(int(v), "mes", "meses")})' for a, v in _esp.items()]) + '.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
def evaluar(L_op, red, n=3000, semilla=21):
    '''VAN neto (tras penalización y operarios extra) y probabilidad de cumplir de una opción (mismos números aleatorios).'''
    dist = {a: ('triangular', max(1.0, d[1] - red[a]), max(1.0, d[2] - red[a]), max(1.0, d[3] - red[a])) for a, d in dist_base.items()}
    bac = datos_proyecto['BAC'] + ucr.replace(np.inf, 0.0) * red
    mc = proyecto.monte_carlo_van_recursos(dist, datos_proyecto[['operarios']], ['operarios'], [L_op], bac, F, N, k, n_samples=n, seed=semilla)
    tau = mc.tau_recursos_samples
    neto = mc.van_recursos_samples - penalizacion * np.maximum(tau - plazo, 0.0) - coste_operario * (L_op - L) * tau
    return {'VAN medio': neto.mean(), 'P(VAN<0)': (neto < 0).mean(), 'P(τ≤plazo)': (tau <= plazo).mean(),
            'τ medio': tau.mean(), 'coste reducción': float((ucr.replace(np.inf, 0.0) * red).sum())}

# %%
ceros = pd.Series(0.0, index=datos_proyecto.index)
A = evaluar(L, ceros)
Markdown(f'En cada una de las 3000 réplicas se sortean las duraciones (triangulares, redondeadas a meses enteros), se '
         f'replanifica con {L} operarios y se obtiene la duración τ. El VAN de la réplica descuenta, a la tasa mensual '
         f'equivalente (1 + k_m)¹² = 1 + {k:g}, los costes (el BAC de cada actividad repartido uniformemente en su duración) y '
         f'el flujo de {F:g} k€/mes durante {N} meses desde el mes τ; al VAN se le resta la penalización, {penalizacion:g} × máx(0, τ − {plazo}) '
         f'(sin descontar, como aproximación). Contando las réplicas:\n\n'
         f'**Respuesta:** si no se hace nada, P(τ ≤ {plazo}) = **{A["P(τ≤plazo)"]:.3f}** (duración media {A["τ medio"]:.2f} meses), '
         f'VAN esperado neto de penalización = **{A["VAN medio"]:.1f} k€** y P(VAN < 0) = **{A["P(VAN<0)"]:.3f}**.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Las reducciones de C y D se calculan con `crashing_recursos`: en cada paso se programa con el límite, se halla la cadena crítica con recursos, se prueban como candidatos sus actividades reducibles (sueltas, por parejas y el corte de coste mínimo) replanificando con cada uno, y se elige el más barato que de verdad acorta el proyecto replanificado. Paso a paso, con el límite actual (C) y con un operario más (D):

# %%
#| tbl-cap: "Crashing con recursos para la opción C (límite actual)"
#| label: tbl-crashingC
cr_c = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], [L], tope, ucr, reduccion=2)
cr_c.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]

# %%
#| tbl-cap: "Crashing con recursos para la opción D (un operario más)"
#| label: tbl-crashingD
cr_d = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], [L + 1], tope, ucr, reduccion=2)
cr_d.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]

# %%
red_c = pd.Series(cr_c.reducciones, dtype=float).reindex(datos_proyecto.index).fillna(0.0)
red_d = pd.Series(cr_d.reducciones, dtype=float).reindex(datos_proyecto.index).fillna(0.0)
opciones = {'A: no hacer nada': (L, ceros), 'B: un operario más': (L + 1, ceros),
            'C: acelerar dos meses': (L, red_c), 'D: operario más y acelerar': (L + 1, red_d)}
tabla = pd.DataFrame({n: evaluar(Lo, r) for n, (Lo, r) in opciones.items()}).T
tabla.rename(columns={'VAN medio': 'VAN esperado neto (k€)', 'P(VAN<0)': 'P(VAN < 0)', 'P(τ≤plazo)': f'P(τ ≤ {plazo})',
                      'τ medio': 'duración media (meses)', 'coste reducción': 'coste de la aceleración (k€)'}).astype(float).round(3)

# %%
display(pd.DataFrame({'C: reducciones': red_c[red_c > 0], 'D: reducciones': red_d[red_d > 0]}).fillna(0).astype(int).T)
def _coste(red):
    r = red[red > 0]
    return ' + '.join(f'{ucr[a]:g}·{int(v)}' for a, v in r.items()) + f' = {float((ucr[r.index] * r).sum()):g} k€'
def _difiere(cr):
    f = cr.pasos[cr.pasos['ackoff_sin_recursos'].astype(str) != cr.pasos['actividades'].astype(str)]
    return '; '.join(f'en el paso {i} Ackoff sin recursos habría acortado {r.ackoff_sin_recursos}, pero con el límite de operarios '
                     f'eso no acorta el proyecto replanificado y se acorta {r.actividades}' for i, r in f.iterrows())
_dc, _dd = _difiere(cr_c), _difiere(cr_d)
Markdown(f'Coste de la aceleración (Σ CUR × meses recortados): C, {_coste(red_c)}; D, {_coste(red_d)}. '
         + (f'Opción C: {_dc}. ' if _dc else 'Opción C: cada paso acorta lo mismo que Ackoff sin recursos. ')
         + (f'Opción D: {_dd}. ' if _dd else 'Opción D: cada paso acorta lo mismo que Ackoff sin recursos. ')
         + 'El coste de la aceleración se suma al BAC de las actividades acortadas y el del operario extra '
         f'({coste_operario:g} k€/mes × τ, sin descontar) se resta del VAN, junto con la penalización.\n\n'
         '**Respuesta:** la tabla de opciones anterior da, para cada una, el VAN esperado neto, P(VAN < 0), la probabilidad de cumplir '
         f'el plazo, la duración media y el coste de la aceleración; se comparan en el apartado siguiente.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
mejor = tabla['VAN medio'].idxmax()
cliente = tabla['P(τ≤plazo)'].idxmax()
A0 = tabla.loc['A: no hacer nada']
dB = tabla.loc['B: un operario más', 'VAN medio'] - A0['VAN medio']
dC = tabla.loc['C: acelerar dos meses', 'VAN medio'] - A0['VAN medio']
dD = tabla.loc['D: operario más y acelerar', 'VAN medio'] - A0['VAN medio']
pB = tabla.loc['B: un operario más', 'P(τ≤plazo)'] - A0['P(τ≤plazo)']
pC = tabla.loc['C: acelerar dos meses', 'P(τ≤plazo)'] - A0['P(τ≤plazo)']
pD = tabla.loc['D: operario más y acelerar', 'P(τ≤plazo)'] - A0['P(τ≤plazo)']
mejores_A = [n for n in tabla.index if tabla.loc[n, 'VAN medio'] > A0['VAN medio']]
if np.isclose(dD, dB + dC, atol=0.5):
    _suma = 'aproximadamente sí: los efectos casi se suman'
elif dD < dB + dC:
    _suma = ('no: juntas valen menos que la suma por separado, porque las dos medidas atacan el mismo problema (la '
             'duración con recursos): una vez que el operario extra elimina esperas, acelerar recorta menos meses útiles, '
             'y el coste de cada medida sí se paga entero')
else:
    _suma = ('no: juntas valen más que la suma por separado, porque con un operario más las actividades aceleradas '
             'ya no se quedan esperando por operarios y el recorte llega de verdad al final del proyecto')
Markdown(f'Cambio de cada opción respecto a A (no hacer nada): VAN esperado neto B {dB:+.1f}, C {dC:+.1f}, D {dD:+.1f} k€; '
         f'probabilidad de cumplir el plazo B {pB:+.3f}, C {pC:+.3f}, D {pD:+.3f}.\n\n'
         f'**Respuesta:**\n\n'
         f'- Al director financiero: **{mejor}**, la de mayor VAN esperado neto ({tabla.loc[mejor, "VAN medio"]:.1f} k€ frente a '
         f'{A0["VAN medio"]:.1f} sin hacer nada; P(VAN < 0) = {tabla.loc[mejor, "P(VAN<0)"]:.3f}). Superan a no hacer nada: {lista_y(mejores_A)}.\n'
         f'- Al cliente, que pregunta por la probabilidad de cumplir: con la opción recomendada es {tabla.loc[mejor, "P(τ≤plazo)"]:.2f}; '
         f'la más alta la da {cliente} ({tabla.loc[cliente, "P(τ≤plazo)"]:.2f})'
         + (', que es la misma.' if cliente == mejor else f', pero cuesta {tabla.loc[mejor, "VAN medio"] - tabla.loc[cliente, "VAN medio"]:.1f} k€ de VAN esperado.') + '\n'
         f'- ¿Se suman B y C en D? B + C = ({dB:+.1f}) + ({dC:+.1f}) = {dB + dC:+.1f} k€ frente a D = {dD:+.1f} k€: {_suma}.')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
_A, _M = tabla.loc['A: no hacer nada'], tabla.loc[mejor]
_red = {'C: acelerar dos meses': red_c, 'D: operario más y acelerar': red_d}.get(mejor)
_acel = lista_y([f'{a} ({plural(int(v), "mes", "meses")})' for a, v in _red[_red > 0].items()]) if _red is not None else ''
_operario = mejor.startswith(('B', 'D'))
Markdown(f'- *Cliente:* «Con la cuadrilla actual la probabilidad de entregar en el mes {plazo} es de {_A["P(τ≤plazo)"]:.0%} '
         f'(duración media {_A["τ medio"]:.1f} meses). Con las medidas que vamos a tomar sube al {_M["P(τ≤plazo)"]:.0%}, con una '
         f'duración media de {_M["τ medio"]:.1f} meses. Les informaremos cada mes de cómo evoluciona.»\n'
         f'- *Director financiero:* «Propongo {mejor[3:]}: VAN esperado neto de penalizaciones y costes {_M["VAN medio"]:.1f} k€, '
         f'frente a {_A["VAN medio"]:.1f} sin hacer nada; probabilidad de VAN negativo {_M["P(VAN<0)"]:.1%} (sin hacer nada, '
         f'{_A["P(VAN<0)"]:.1%}). Adjunto la tabla de las cuatro opciones.»\n'
         + (f'- *Jefe de obra:* «Contratamos el operario adicional ({coste_operario:g} k€/mes): la simulación lo respalda.»\n' if _operario else
            f'- *Jefe de obra:* «De momento no contratamos el operario adicional: con {coste_operario:g} k€/mes, en la simulación no compensa '
            f'(opción B: {tabla.loc["B: un operario más", "VAN medio"]:.1f} k€ frente a {_M["VAN medio"]:.1f}). Vigilemos las esperas de la cadena crítica.»\n')
         + (f'- *Contratista:* «Necesitamos acortar {_acel}, con un coste de {_M["coste reducción"]:g} k€ a los CUR de la tabla.»' if _red is not None else
            '- *Contratista:* «Por ahora no aceleramos: con el límite de operarios, recortar duraciones no compensa su coste. Les avisaremos si cambia.»'))

# %% [markdown]
# ## Apartado 6

# %%
display(Markdown(pregunta_6))

# %%
_ack = [str(x) for x in cr_c.pasos['ackoff_sin_recursos']]; _real = [str(x) for x in cr_c.pasos['actividades']]
_l = [f'- **Sin recursos** se habría dicho que el proyecto dura {duracion_libre:g} meses y que el plazo del mes {plazo} se '
      f'cumple; con {L} operarios dura {res0.duracion}, y sin hacer nada sólo se cumple el {A["P(τ≤plazo)"]:.0%} de las veces.']
if _ack != _real:
    _l.append(f'- **Sin recursos**, Ackoff habría acortado {lista_y(_ack)}; con el límite de operarios lo que de verdad acorta es '
              f'{lista_y(_real)}: se habría pagado una reducción que no adelanta la entrega.')
else:
    _l.append(f'- Aquí Ackoff sin recursos y la reducción con recursos eligen lo mismo ({lista_y(_real)}), pero sólo la segunda '
              'comprueba que el recorte llega al final con el límite de operarios.')
_l.append(f'- **Sin incertidumbre** se habría prometido una fecha, no una probabilidad: la duración media simulada sin hacer nada '
          f'es {A["τ medio"]:.2f} meses (con duraciones esperadas, {res0.duracion}) y P(τ ≤ {plazo}) = {A["P(τ≤plazo)"]:.2f}.')
_l.append(f'- **Sin VAN** se habría elegido la opción que más acorta ({tabla["τ medio"].idxmin()}) y no la que más valor crea '
          f'({mejor})' + ('.' if np.isclose(dD, dB + dC, atol=0.5) else
                         f'; y juzgando cada medida por separado no se ve que B + C ({dB + dC:+.1f} k€) no es D ({dD:+.1f} k€).'))
Markdown('\n'.join(_l))
