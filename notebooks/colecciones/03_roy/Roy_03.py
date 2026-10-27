# ---
# title: Roy 03
# subtitle: "8 actividades, 3 ficticias en PERT, con espera y solape"
# coleccion: 03_roy
# variante: esperas-y-solapes
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta.

# %%
pregunta_1 = "1. Dibuje el grafo Roy del proyecto (actividades en los nodos).  "
pregunta_2 = "2. Dibuje el grafo PERT equivalente. Compare el número de nodos y de arcos de ambos grafos y explique a qué se debe la diferencia.  "
pregunta_3 = "3. Calcule sobre el grafo Roy los tiempos más tempranos y más tardíos de inicio y fin de cada actividad, y señale el camino crítico en ambos grafos.  "
pregunta_4 = "4. Replantee el proyecto si, además, entre el final de B y el comienzo de D deben transcurrir 3 semanas de espera, y C puede comenzar cuando A está a la mitad de su ejecución. Dibuje el nuevo grafo Roy y calcule la nueva duración del proyecto.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 8       ),
    ('B'     , '---'      , 1       ),
    ('C'     , 'A,B'      , 7       ),
    ('D'     , 'B'        , 6       ),
    ('E'     , 'C'        , 2       ),
    ('F'     , 'C,D'      , 7       ),
    ('G'     , 'C,D'      , 1       ),
    ('H'     , 'E,F,G'    , 8       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la producción de un cortometraje; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Guion definitivo',
    'B': 'Casting',
    'C': 'Localizaciones y permisos',
    'D': 'Diseño de vestuario y atrezo',
    'E': 'Plan de rodaje',
    'F': 'Ensayos con los actores',
    'G': 'Rodaje de exteriores',
    'H': 'Rodaje en plató',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %%
def explica_ficticias(proyecto, prelaciones):
    """Una línea por ficticia del PERT: nodos que une, prelación que transmite y por qué no
    basta con hacer empezar las sucesoras en el nodo donde terminan las predecesoras."""
    G = proyecto.pert_graph
    pred = {a: set(ps) for a, ps in prelaciones.immediate_predecessor.items()}
    R = nx.DiGraph([(p, a) for a, ps in pred.items() for p in ps])
    R.add_nodes_from(pred)
    anc = {a: nx.ancestors(R, a) for a in pred}
    def entran(nodo, salvo=None):
        acts = set()
        for i, _, d in G.in_edges(nodo, data=True):
            if d['activity'] != salvo:
                acts |= entran(i) if d['activity'].startswith('@') else {d['activity']}
        return acts
    def salen(nodo, salvo=None):
        acts = set()
        for _, j, d in G.out_edges(nodo, data=True):
            if d['activity'] != salvo:
                acts |= salen(j) if d['activity'].startswith('@') else {d['activity']}
        return acts
    def arcos_reales(nodo, sentido):
        aristas = G.in_edges(nodo, data=True) if sentido == 'in' else G.out_edges(nodo, data=True)
        return {(u if sentido == 'in' else v): d['activity'] for u, v, d in aristas if not d['activity'].startswith('@')}
    pl = lambda acts, s, p: s if len(acts) == 1 else p
    lineas = []
    arcos = sorted(proyecto.nodes_from_edges.items(), key=lambda kv: (int(kv[1][0]), int(kv[1][1])))
    for nombre, (i, j) in arcos:
        if not nombre.startswith('@'):
            continue
        antes, despues = sorted(entran(i)), sorted(salen(j))
        # prelaciones inmediatas del cuadro que pasan por la ficticia (las demás ya se deducen de otras)
        pares = [(x, y) for x in antes for y in despues if x in pred[y]] or [(x, y) for x in antes for y in despues]
        despues = sorted({y for _, y in pares})
        grupos = {}
        for y in despues:
            grupos.setdefault(tuple(sorted(x for x, z in pares if z == y)), []).append(y)
        transmite = '; '.join(f'{lista_y(xs)} {pl(xs, "precede", "preceden")} a {lista_y(ys)}' for xs, ys in grupos.items())
        # Si los nodos i y j fueran uno solo, lo que llega a j precedería a lo que sale de i
        llegan_j, salen_i = sorted(entran(j, salvo=nombre)), sorted(salen(i, salvo=nombre))
        falsas = [(x, y) for y in salen_i for x in llegan_j if x not in anc[y]]
        if falsas:
            x_s = sorted({x for x, _ in falsas})
            y_s = sorted({y for _, y in falsas})
            directas_j, directas_i = set(arcos_reales(j, 'in').values()), set(arcos_reales(i, 'out').values())
            llega = (f'{pl(x_s, "llega", "llegan")} al nodo {j}' if set(x_s) <= directas_j else
                     f'{pl(x_s, "llega", "llegan")} al nodo {j} (directamente o por otra ficticia)')
            sale = (f'{pl(y_s, "sale", "salen")} de ese nodo' if set(y_s) <= directas_i else
                    f'{pl(y_s, "empieza", "empiezan")} a partir de ese nodo (directamente o tras otra ficticia)')
            motivo = (f'{lista_y(despues)} {pl(despues, "espera", "esperan")} además a {lista_y(x_s)}, que {llega}; '
                      f'si {lista_y(despues)} {pl(despues, "saliera", "salieran")} del nodo {i}, {lista_y(x_s)} '
                      f'{pl(x_s, "precedería", "precederían")} también a {lista_y(y_s)}, que {sale} y no '
                      f'{pl(y_s, "depende", "dependen")} de {pl(x_s, "ella", "ellas")}')
        else:
            gemelas = None
            ini_i, ini_j = arcos_reales(i, 'in'), arcos_reales(j, 'in')
            fin_i, fin_j = arcos_reales(i, 'out'), arcos_reales(j, 'out')
            for u in sorted(set(ini_i) & set(ini_j)):
                gemelas = gemelas or (ini_i[u], ini_j[u], u, j, f'empiezan en el mismo nodo ({u})')
            for v in sorted(set(fin_i) & set(fin_j)):
                gemelas = gemelas or (fin_i[v], fin_j[v], i, v, f'terminan en el mismo nodo ({v})')
            if gemelas:
                a, b, n1, n2, verbo = gemelas
                a, b = sorted([a, b])
                motivo = (f'{a} y {b} {verbo}; sin la ficticia irían '
                          f'las dos del nodo {n1} al {n2}, dos arcos entre el mismo par de nodos que no se podrían distinguir')
            else:
                motivo = 'sin ella habría que fundir los nodos y se perdería la estructura del grafo'
        lineas.append(f'- Ficticia {i} → {j}: transmite que {transmite}. Hace falta porque {motivo}.')
    return lineas

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# En el grafo Roy cada actividad es un nodo y cada prelación inmediata un arco. Se añaden dos nodos auxiliares, inicio y fin, para que el grafo tenga un único origen y un único destino. No hacen falta actividades ficticias.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelacionesExpandido
prelaciones.display_immediate_linkage_matrix()

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-roy
proyecto.display_roy_construccion(tabla=prelaciones)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| fig-cap: "Grafo PERT equivalente"
#| label: fig-pert
proyecto.pert(durations=False, ranksep=1)

# %%
n_act = len(proyecto)
n_prelaciones = sum(len(p) for p in prelaciones.immediate_predecessor.values())
n_arcos_roy = n_prelaciones + len(prelaciones.start_activities) + len(prelaciones.end_activities)
pd.DataFrame({'nodos': [n_act + 2, len(proyecto.nodes)],
              'arcos': [n_arcos_roy, n_act + len(proyecto.dummies)],
              'ficticias': [0, len(proyecto.dummies)]},
             index=['Roy', 'PERT'])

# %%
_ini, _fin = sorted(prelaciones.start_activities), sorted(prelaciones.end_activities)
_lineas = explica_ficticias(proyecto, prelaciones)
Markdown(f'''**Roy**: un nodo por actividad más los nodos de inicio y fin: {n_act} + 2 = **{n_act + 2} nodos**. Los arcos son las
{n_prelaciones} prelaciones inmediatas del cuadro, más {plural(len(_ini), 'arco')} desde el inicio hasta
{'la actividad inicial' if len(_ini) == 1 else 'las actividades iniciales'} ({lista_y(_ini)}) y {plural(len(_fin), 'arco')} desde
{'la actividad final' if len(_fin) == 1 else 'las actividades finales'} ({lista_y(_fin)}) hasta el fin:
{n_prelaciones} + {len(_ini)} + {len(_fin)} = **{n_arcos_roy} arcos**.

**PERT**: los arcos son las {n_act} actividades más las ficticias: {n_act} + {len(proyecto.dummies)} = **{n_act + len(proyecto.dummies)} arcos**;
los nodos son los sucesos (inicio y fin de grupos de actividades), **{len(proyecto.nodes)} nodos**, porque las actividades con las
mismas predecesoras salen de un mismo nodo y las que tienen las mismas sucesoras llegan a un mismo nodo.

La diferencia se debe a qué representa cada elemento: en el Roy las prelaciones son arcos y nunca hacen falta ficticias; en el
PERT las prelaciones se expresan compartiendo nodos, y cuando una prelación no puede expresarse así aparece una ficticia.
''' + (f'''Este PERT necesita {plural(len(_lineas), 'ficticia')}:

''' + '\n'.join(_lineas) if _lineas else 'Este PERT no necesita ninguna ficticia.'))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Sobre el grafo Roy, el inicio más temprano de una actividad es el máximo de los fines más tempranos de sus predecesoras; el fin más tardío, el mínimo de los inicios más tardíos de sus sucesoras. Son los mismos números que da el PERT.

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'], grafo="roy")

# %% [markdown]
# Cómo sale cada fecha (ES/EF: inicio y fin más tempranos; LS/LF: inicio y fin más tardíos): para cada actividad, la tabla muestra los fines más tempranos de sus predecesoras que se comparan (el máximo es su ES) y los inicios más tardíos de sus sucesoras (el mínimo es su LF).

# %%
#| tbl-cap: "Cálculo paso a paso de las fechas"
#| label: tbl-fechasPaso
detalle_holguras(calendario, prelaciones)[['ES = máx EF pred.', 'EF = ES + d', 'LF = mín LS suc.', 'LS = LF − d']]

# %% [markdown]
# El calendario incluye también las holguras. La *total* (LS − ES) es lo que puede retrasarse una actividad sin alargar el proyecto; la *libre* (mín ES de las sucesoras − EF) lo que puede retrasarse sin retrasar el inicio más temprano de ninguna sucesora; la *independiente* (máx(0, mín ES suc. − máx LF pred. − d)) la que conserva aunque sus predecesoras acaben lo más tarde posible y sus sucesoras empiecen lo antes posible. Las actividades de holgura total nula son las críticas y forman el camino crítico.

# %%
duracion_proyecto = proyecto.duration(datos_proyecto['duration'])
explica_camino_critico(proyecto, datos_proyecto['duration'], 'semanas')

# %%
_criticas = ordena_ruta(calendario.index[calendario['H_total'] == 0], calendario)
_holgadas = ordena_ruta(calendario.index[calendario['H_total'] > 0], calendario)
_txt = ', '.join(f'{a} ({int(calendario.loc[a, "H_total"])})' for a in _holgadas)
Markdown(f'**Respuesta:** el proyecto dura {plural(duracion_proyecto, "semana")} y son críticas {lista_y(_criticas)} '
         f'(holgura total 0). En el grafo Roy el camino crítico pasa por los nodos de esas actividades y en el '
         f'PERT por sus arcos (en rojo en las dos figuras siguientes). '
         + (f'Las demás tienen holgura total positiva (entre paréntesis, en semanas): {_txt}.' if _holgadas else 'No hay actividades con holgura.'))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con tiempos y camino crítico"
#| label: fig-royCritico
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con tiempos y camino crítico"
#| label: fig-pertCritico
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Las dos condiciones nuevas son relaciones con desfase. Con actividades en los nodos se modelan con actividades auxiliares:
#
# - La espera entre el final de B y el comienzo de D es una actividad de espera, 💤D, de duración 3 y sin consumo de recursos, que sucede a B y precede a D.
# - Que C pueda comenzar a mitad de A es un solape comienzo-comienzo: se parte A en A1 y A2 (de 4 y 4 semanas) y C pasa a suceder a A1, mientras que las demás sucesoras de A siguen esperando al final de A2.

# %%
prelaciones_modificadas = {
    'B': '---',
    'C': 'A1,B',
    'D': '💤D',
    'E': 'C',
    'F': 'C,D',
    'G': 'C,D',
    'H': 'E,F,G',
    '💤D': 'B',
    'A1': '---',
    'A2': 'A1',
}
duraciones_modificadas = {
    'B': 1,
    'C': 7,
    'D': 6,
    'E': 2,
    'F': 7,
    'G': 1,
    'H': 8,
    '💤D': 3,
    'A1': 4,
    'A2': 4,
}
prelaciones_mod = PredecessorTable.from_dict_of_strings(prelaciones_modificadas)
proyecto_mod = prelaciones_mod.create_project()
duraciones_mod = pd.Series(duraciones_modificadas, name='duration')

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto con la espera y el solape"
#| label: fig-royModificado
proyecto_mod.display_roy_pasos(duraciones_mod)

# %%
#| tbl-cap: "Calendario del proyecto modificado"
#| label: tbl-calendarioModificado
proyecto_mod.display_calendario(duraciones_mod, grafo="roy")

# %%
duracion_mod = proyecto_mod.duration(duraciones_mod)
explica_camino_critico(proyecto_mod, duraciones_mod, 'semanas')

# %% [markdown]
# Para ver qué aporta cada cambio se calcula también el proyecto con sólo uno de los dos.

# %%
prelaciones_solo_espera = {
    'A': '---',
    'B': '---',
    'C': 'A,B',
    'D': '💤D',
    'E': 'C',
    'F': 'C,D',
    'G': 'C,D',
    'H': 'E,F,G',
    '💤D': 'B',
}
duraciones_solo_espera = {
    'A': 8,
    'B': 1,
    'C': 7,
    'D': 6,
    'E': 2,
    'F': 7,
    'G': 1,
    'H': 8,
    '💤D': 3,
}
prelaciones_solo_solape = {
    'B': '---',
    'C': 'A1,B',
    'D': 'B',
    'E': 'C',
    'F': 'C,D',
    'G': 'C,D',
    'H': 'E,F,G',
    'A1': '---',
    'A2': 'A1',
}
duraciones_solo_solape = {
    'B': 1,
    'C': 7,
    'D': 6,
    'E': 2,
    'F': 7,
    'G': 1,
    'H': 8,
    'A1': 4,
    'A2': 4,
}
def _duracion(prel, dur):
    return int(PredecessorTable.from_dict_of_strings(prel).create_project().duration(pd.Series(dur)))
T_espera = _duracion(prelaciones_solo_espera, duraciones_solo_espera)
T_solape = _duracion(prelaciones_solo_solape, duraciones_solo_solape)
X, Y, W, Z, espera, mitad = 'B', 'D', 'A', 'C', 3, 4
c = calendario
margen = int(c.loc[Y, 'inicio_mas_tardio'] - c.loc[X, 'fin_mas_temprano'])
enlace_critico = (c.loc[W, 'H_total'] == 0 and c.loc[Z, 'H_total'] == 0
                  and c.loc[W, 'fin_mas_temprano'] == c.loc[Z, 'inicio_mas_temprano'])
_e = (f'- **Sólo la espera**: la prelación {X} → {Y} tenía un margen LS({Y}) − EF({X}) = '
      f'{int(c.loc[Y, "inicio_mas_tardio"])} − {int(c.loc[X, "fin_mas_temprano"])} = {plural(margen, "semana")}; '
      f'una espera de {plural(espera, "semana")} alarga el proyecto máx(0, {espera} − {margen}) = '
      f'{max(0, espera - margen)}: de {duracion_proyecto} a **{T_espera}** semanas.')
if T_solape < duracion_proyecto:
    _por = (f'la prelación {W} → {Z} está en el camino crítico, así que el proyecto se acorta '
            f'{plural(duracion_proyecto - T_solape, "semana")} (hasta que otra ruta pasa a ser la más larga)')
elif enlace_critico:
    _por = f'aunque la prelación {W} → {Z} es crítica, hay otra ruta crítica que no pasa por ella'
else:
    _por = f'la prelación {W} → {Z} no está en el camino crítico'
_s = (f'- **Sólo el solape**: {Z} puede empezar {plural(int(c.loc[W, "duracion"]) - mitad, "semana")} antes '
      f'de que termine {W} (tras {W}1, de {mitad}); {_por}: el proyecto dura **{T_solape}** semanas.')
Markdown(_e + '\n' + _s + f'\n\n**Respuesta:** con la espera y el solape a la vez el proyecto dura '
         f'**{plural(duracion_mod, "semana")}**, ' + ('las mismas que antes.' if duracion_mod == duracion_proyecto else
         f'{plural(abs(duracion_mod - duracion_proyecto), "semana")} {"menos" if duracion_mod < duracion_proyecto else "más"} que las {duracion_proyecto} iniciales.'))
