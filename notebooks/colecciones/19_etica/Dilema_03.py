# ---
# title: Dilema 03
# subtitle: "peores casos entregados + 10 %: 27 semanas frente a 18 esperadas y 20 con el 80 %"
# coleccion: 19_etica
# variante: colchones-ocultos
# nivel: 2
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
# En la reunión de planificación cada responsable de actividad entrega una única duración (columna `estimacion_entregada`, semanas). Usted, que ha hablado con ellos uno a uno, sabe que cada uno ha dado su **peor caso** (la estimación pesimista) «por si acaso», sin decirlo. Las tres estimaciones reales de cada actividad (optimista, más probable y pesimista) se adjuntan.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b , estimacion_entregada
    ('A'     , '---'      , 5, 5, 11, 11                  ),
    ('B'     , '---'      , 3, 9, 9 , 9                   ),
    ('C'     , '---'      , 4, 5, 12, 12                  ),
    ('D'     , 'A,B'      , 1, 1, 1 , 1                   ),
    ('E'     , 'D'        , 5, 5, 11, 11                  ),
    ('F'     , 'C,D'      , 2, 8, 8 , 8                   ),
    ('G'     , 'D'        , 3, 3, 9 , 9                   ),
    ('H'     , 'F,G'      , 1, 2, 3 , 3                   ),
], columns=['activity', 'predecessor', 'a', 'm', 'b', 'estimacion_entregada']).set_index('activity')
datos_enunciado

# %% [markdown]
# > **De: Jefe de proyecto**
# > Con las estimaciones entregadas el proyecto sale en 24 semanas. Yo le añado un 10 % y prometo 27 al cliente; así nunca llegamos tarde. Lo del colchón de cada uno no hace falta contarlo, cada cual gestiona su margen.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C8 — Reserva declarada frente a colchón oculto.** Una reserva de contingencia dimensionada con un nivel de confianza y comunicada es una práctica correcta; el mismo margen escondido en las estimaciones individuales, sin decirlo, es retención de información (C4). *(cf. PMBOK, reservas de contingencia; Código Ético del PMI, 5.2.3.)*
# - **C4 — Retención de información.** Callar un hecho conocido que la otra parte necesitaría para decidir (un riesgo, una holgura, una discrepancia). *(cf. Código Ético del PMI, 5.3.1 y 5.2.3.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Con las tres estimaciones calcule la duración esperada, la desviación típica del camino crítico y la duración que se cumpliría con probabilidad 0,8 (reserva de contingencia necesaria para ese nivel de confianza).  "
pregunta_2 = "2. (A) Calcule la duración con las estimaciones entregadas y con el 10 % del jefe de proyecto, y compare el margen total oculto con la reserva necesaria. ¿Por qué sumar los peores casos de cada actividad sobrestima tanto?  "
pregunta_3 = "3. (B) ¿Qué criterio distingue el margen que propone el jefe de proyecto de una reserva de contingencia correcta? ¿Qué habría que cambiar en el informe de planificación para que cumpla el criterio?  "
pregunta_4 = "4. (C) ¿Cómo plantearía la planificación al equipo y al cliente? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = (datos_proyecto['a'] + 4 * datos_proyecto['m'] + datos_proyecto['b']) / 6
datos_proyecto['standard_deviation'] = (datos_proyecto['b'] - datos_proyecto['a']) / 6
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
duracion = float(proyecto.duration(datos_proyecto['duration']))
sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])
T80 = duracion + norm.ppf(0.8) * sigma
reserva = T80 - duracion

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones medias"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Media y varianza PERT de cada actividad"
#| label: tbl-estimaciones
detalle_estimaciones(datos_enunciado)

# %%
#| tbl-cap: "Duración y varianza de cada ruta"
#| label: tbl-rutas
R = proyecto.path_matrix(dummies=False)
display(detalle_rutas(R, datos_proyecto['duration'], 'duración', 2))
detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)

# %%
Markdown(f'''Duración esperada **{duracion:.2f}** semanas, σ = {sigma:.2f}; con probabilidad 0,8 se termina en **{T80:.2f}** semanas: la reserva de contingencia para ese nivel de confianza es **{reserva:.2f}** semanas (z₀,₈ · σ = 0,84 · {sigma:.2f}).''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duracion_entregada = int(proyecto.duration(datos_proyecto['estimacion_entregada']))
duracion_jp = int(np.ceil(1.1 * duracion_entregada))
camino_entregado = ' / '.join('–'.join(r) for r in proyecto.critical_path(datos_proyecto['estimacion_entregada']).values())
Markdown(f'''Con las estimaciones entregadas: **{duracion_entregada}** semanas (ruta {camino_entregado}); con el 10 % del jefe de proyecto: **{duracion_jp}**. Margen oculto total: {duracion_jp - duracion:.1f} semanas frente a las {reserva:.2f} que justifica el 80 %: **{(duracion_jp - duracion) / reserva:.1f} veces** la reserva necesaria.''')

# %% [markdown]
# Sumar los peores casos sobrestima porque las desviaciones no se suman: se suman las **varianzas**, y la raíz de la suma es mucho menor que la suma de las raíces. Que todas las actividades de una ruta salgan a la vez en su peor caso es un suceso improbable; el margen que protege al proyecto se calcula sobre la ruta, no actividad a actividad, y una sola vez.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# **C8.** Una reserva de contingencia correcta se **dimensiona** con un nivel de confianza (1.2 semanas para el 80 %) y se **declara**: el cliente ve la duración esperada (18 semanas), la reserva y a quién pertenece. El margen del enunciado se esconde dos veces (cada responsable en su estimación y el jefe de proyecto en su 10 %), no se dimensiona y no se comunica: es **C4**, retención de información, y cuesta 9 semanas de plazo al cliente. Para cumplir el criterio, el informe de planificación debe llevar las tres estimaciones por actividad, la duración esperada, la reserva calculada y su nivel de confianza.

# %%
preguntas_test = [
    ('Una reserva de contingencia cumple el criterio C8 cuando…', ['se dimensiona con un nivel de confianza y se declara', 'cada responsable la incluye en su estimación', 'el jefe de proyecto la añade sin comunicarla', 'es como mínimo un 10 %'],
     'Lo que distingue la reserva del colchón es que se calcula y se comunica.'),
    ('Sumar los peores casos de las actividades de la ruta crítica…', ['sobrestima la duración porque las desviaciones no se suman', 'da la duración con probabilidad 0,8', 'da la duración esperada', 'subestima la duración'],
     'Se suman varianzas; la desviación de la ruta es la raíz de esa suma.'),
    ('Esconder un margen en cada estimación sin decirlo es…', ['C4, retención de información', 'C1, afirmación falsa', 'C8, reserva declarada', 'C6, conflicto de interés'],
     'Se calla un dato (el margen) que el cliente y el jefe de proyecto necesitan para decidir.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — colchones ocultos')
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_etica.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
print(gift)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: (1) identificar a todos los afectados y qué pierde o gana cada uno; (2) formular al menos dos alternativas además de la que se propone en el enunciado; (3) someter cada alternativa a las tres pruebas —legalidad (¿cumple la norma?), publicidad (¿la defendería si se hiciera pública?) y reversibilidad (¿la aceptaría en el papel del afectado?)—; (4) decidir y redactar la comunicación con las cifras del bloque A. Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.

# %% [markdown]
# Al equipo: pedir las tres estimaciones a cada responsable y explicar que el margen se calcula una vez, sobre la ruta; los colchones individuales no protegen más y alargan el plazo. Al cliente: duración esperada de 18 semanas, reserva de 1.2 semanas para el 80 % y compromiso en 20 semanas, con la reserva visible y gestionada por el jefe de proyecto. Prometer 27 semanas «para no llegar nunca tarde» supera la prueba de legalidad pero no la de publicidad: si el cliente viera la cuenta, no aceptaría pagar el plazo de dos colchones.
