# ---
# title: PERT 04
# subtitle: "9 actividades, 1 pareja con mismas predecesoras y 3 con mismas sucesoras"
# coleccion: 02_pert
# variante: compartir-nodos
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
# Considere el proyecto definido por el siguiente cuadro de prelaciones inmediatas. En un grafo PERT dos actividades pueden salir del mismo nodo sólo si tienen exactamente las mismas predecesoras, y llegar al mismo nodo sólo si tienen exactamente las mismas sucesoras.

# %%
pregunta_1 = "1. ¿Pueden las actividades D y E compartir nodo inicial? ¿Y nodo final? Razónelo con el cuadro de prelaciones expandido.  "
pregunta_2 = "2. Misma pregunta para A y B, y para F y I.  "
pregunta_3 = "3. Enumere todas las parejas de actividades que pueden compartir nodo inicial y todas las que pueden compartir nodo final. Si una pareja puede compartir ambos, ¿qué ocurre?  "
pregunta_4 = "4. Dibuje el grafo PERT y compruebe sobre él las respuestas anteriores.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de prelaciones del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor
    ('A'     , '---'      ),
    ('B'     , '---'      ),
    ('C'     , '---'      ),
    ('D'     , 'A,B,C'    ),
    ('E'     , 'A,B,C'    ),
    ('F'     , 'D,E'      ),
    ('G'     , 'C'        ),
    ('H'     , 'F'        ),
    ('I'     , 'G,H'      ),
], columns=['activity', 'predecessor']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la organización de un congreso científico; las actividades son las siguientes.

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Definición del programa y fechas',
    'B': 'Reserva de la sede',
    'C': 'Convocatoria de comunicaciones',
    'D': 'Página web e inscripciones',
    'E': 'Contratación de ponentes invitados',
    'F': 'Revisión de los trabajos recibidos',
    'G': 'Elaboración del programa definitivo',
    'H': 'Contratación de catering y viajes',
    'I': 'Edición del libro de actas',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
predecesoras = prelaciones.immediate_predecessor
sucesoras = {a: {s for s, ps in predecesoras.items() if a in ps} for a in datos_proyecto.index}
def analiza(a, b):
    return pd.Series({'predecesoras de ' + a: ', '.join(sorted(predecesoras[a])) or '—',
                      'predecesoras de ' + b: ', '.join(sorted(predecesoras[b])) or '—',
                      'mismas predecesoras': 'sí' if predecesoras[a] == predecesoras[b] else 'no',
                      'sucesoras de ' + a: ', '.join(sorted(sucesoras[a])) or '—',
                      'sucesoras de ' + b: ', '.join(sorted(sucesoras[b])) or '—',
                      'mismas sucesoras': 'sí' if sucesoras[a] == sucesoras[b] else 'no'})
def concluye(a, b):
    ini, fin = predecesoras[a] == predecesoras[b], sucesoras[a] == sucesoras[b]
    t = (f'**{a} y {b}**: ' + ('tienen las mismas predecesoras (misma fila del cuadro expandido), luego '
         'pueden compartir nodo inicial; ' if ini else 'sus predecesoras son distintas (filas distintas), '
         'luego no pueden compartir nodo inicial; ')
         + ('tienen las mismas sucesoras (misma columna), luego podrían compartir nodo final' if fin else
            'sus sucesoras son distintas (columnas distintas), luego no pueden compartir nodo final'))
    if ini and fin:
        t += ('; pero no pueden compartir los dos a la vez (serían dos arcos entre el mismo par de nodos): '
              'una de las dos necesita una ficticia')
    return Markdown(t + '.')

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelacionesExpandido
prelaciones.display_immediate_linkage_matrix()

# %%
analiza('D', 'E').to_frame('').T

# %%
concluye('D', 'E')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
analiza('A', 'B').to_frame('').T

# %%
concluye('A', 'B')

# %%
analiza('F', 'I').to_frame('').T

# %%
concluye('F', 'I')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
acts = list(datos_proyecto.index)
parejas_inicio = [(a, b) for i, a in enumerate(acts) for b in acts[i+1:] if predecesoras[a] == predecesoras[b]]
parejas_fin = [(a, b) for i, a in enumerate(acts) for b in acts[i+1:] if sucesoras[a] == sucesoras[b]]
ambas = [p for p in parejas_inicio if p in parejas_fin]
_par = lambda ps: ', '.join(f'{a}–{b}' for a, b in ps) or 'ninguna'
Markdown(f'- Pueden compartir nodo inicial (mismas predecesoras): {_par(parejas_inicio)}.\n'
         f'- Pueden compartir nodo final (mismas sucesoras): {_par(parejas_fin)}.\n'
         f'- Cumplen las dos condiciones: {_par(ambas)}.')

# %% [markdown]
# Las actividades iniciales comparten todas el nodo origen (no tienen predecesoras) y las finales el nodo final (no tienen sucesoras), por eso aparecen en las listas aunque tengan `---`.

# %%
Markdown('**Respuesta:** ' + (f'las parejas {_par(ambas)} podrían compartir los dos nodos, pero entonces '
         'serían dos arcos entre el mismo par de nodos y el grafo no distinguiría una actividad de la otra: '
         'en cada una de esas parejas, una actividad termina en un nodo propio unido al común por una ficticia.'
         if ambas else 'ninguna pareja cumple las dos condiciones, así que no hay arcos paralelos que evitar.'))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Construcción del grafo con el visor: primero el grafo terminado; con los botones, fila a fila desde el principio y, al final, la comprobación de las sucesoras columna a columna.

# %%
#| fig-cap: "Construcción del grafo PERT a partir del cuadro de prelaciones"
#| label: fig-pert
proyecto.display_pert_construccion()

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

# %%
_nodos = proyecto.nodes_from_edges
_ini = {}
for a in datos_proyecto.index:
    _ini.setdefault(_nodos[a][0], []).append(a)
_lineas = explica_ficticias(proyecto, prelaciones)
Markdown('En el grafo salen del mismo nodo: ' + '; '.join(f'{lista_y(v)} (nodo {k})' for k, v in sorted(_ini.items(), key=lambda kv: int(kv[0])) if len(v) > 1)
         + f'. Tiene {plural(len(_lineas), "ficticia")}' + (':\n\n' + '\n'.join(_lineas) if _lineas else '.'))
