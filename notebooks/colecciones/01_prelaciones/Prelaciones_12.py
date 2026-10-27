# ---
# title: Prelaciones 12
# subtitle: "9 actividades, grafo con una prelación de más y otra de menos"
# coleccion: 01_prelaciones
# variante: grafo-erroneo
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
# Un compañero ha dibujado el grafo PERT que se muestra a continuación a partir del cuadro de prelaciones adjunto. Sospechamos que el grafo no refleja exactamente el cuadro.

# %%
#| tbl-cap: "Cuadro de prelaciones (correcto)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor
    ('A'     , '---'      ),
    ('B'     , '---'      ),
    ('C'     , '---'      ),
    ('D'     , 'A,C'      ),
    ('E'     , 'A,C'      ),
    ('F'     , 'B,D'      ),
    ('G'     , 'E,F'      ),
    ('H'     , 'G'        ),
    ('I'     , 'H'        ),
], columns=['activity', 'predecessor']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la apertura de una clínica veterinaria; las actividades son las siguientes.

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de mercado y local',
    'B': 'Proyecto de reforma y licencia',
    'C': 'Obra de reforma',
    'D': 'Instalación de quirófano y radiología',
    'E': 'Mobiliario y equipamiento clínico',
    'F': 'Software de gestión',
    'G': 'Selección de personal',
    'H': 'Acuerdos con proveedores',
    'I': 'Registro sanitario',
}, name='Descripción')
descripciones.to_frame()

# %%
#| fig-cap: "Grafo PERT dibujado por el compañero"
#| label: fig-pertErroneo
prelaciones_del_grafo = {
    'A': '---',
    'B': '---',
    'C': '---',
    'D': 'A,C',
    'E': 'A,C',
    'F': 'B,D',
    'G': 'D,E',
    'H': 'G',
    'I': 'H',
}
proyecto_dibujado = ProjectGraph.from_dict_of_strings(prelaciones_del_grafo)
proyecto_dibujado.pert(durations=False, ranksep=1)

# %%
pregunta_1 = "1. Construya el cuadro de ancestros a partir del cuadro de prelaciones y, por separado, a partir del grafo dibujado (siguiendo los arcos hacia atrás).  "
pregunta_2 = "2. Compare ambos cuadros e identifique qué prelaciones sobran y cuáles faltan en el grafo.  "
pregunta_3 = "3. Dibuje el grafo PERT correcto. ¿Cambia el número de ficticias?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Cuadro de ancestros según el cuadro de prelaciones"
#| label: tbl-ancestrosCuadro
proyecto.display_distant_linkage_matrix()

# %%
#| tbl-cap: "Cuadro de ancestros según el grafo dibujado"
#| label: tbl-ancestrosGrafo
proyecto_dibujado.display_distant_linkage_matrix()

# %%
anc_cuadro = proyecto.distant_predecessor()
anc_grafo = proyecto_dibujado.distant_predecessor()
difieren = [a for a in datos_proyecto.index if anc_cuadro[a] != anc_grafo[a]]
Markdown(f'**Respuesta:** los dos cuadros de ancestros difieren en '
         f'{"la fila" if len(difieren) == 1 else "las filas"} de {lista_y(difieren)}; las demás filas coinciden.'
         if difieren else '**Respuesta:** los dos cuadros de ancestros coinciden fila a fila.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Una prelación que está en el cuadro y no en el grafo se detecta porque alguna actividad tiene menos ancestros en el grafo; una que sobra, porque tiene más. Para localizar la prelación inmediata responsable se comparan las predecesoras inmediatas de las actividades cuyo conjunto de ancestros difiere.

# %%
pd.DataFrame({'ancestros según el cuadro': {a: ', '.join(sorted(anc_cuadro[a])) for a in difieren},
              'ancestros según el grafo': {a: ', '.join(sorted(anc_grafo[a])) for a in difieren}})

# %%
inm_cuadro = prelaciones.immediate_predecessor
inm_grafo = PredecessorTable.from_dict_of_strings(prelaciones_del_grafo).immediate_predecessor
faltan = sorted((p, a) for a in inm_cuadro for p in inm_cuadro[a] - inm_grafo.get(a, set()))
sobran = sorted((p, a) for a in inm_grafo for p in inm_grafo[a] - inm_cuadro.get(a, set()))
def _por_que(p, a, anc):
    return (f'{p} {"sí" if p in anc[a] else "no"} es ancestro de {a}')
_l = [f'- Falta **{p} → {a}**: está en el cuadro, pero en el grafo {a} no la tiene como predecesora inmediata '
      f'(según el grafo, {_por_que(p, a, anc_grafo)}).' for p, a in faltan]
_l += [f'- Sobra **{p} → {a}**: el grafo hace que {p} preceda directamente a {a}, y en el cuadro no es así '
       f'(según el cuadro, {_por_que(p, a, anc_cuadro)}).' for p, a in sobran]
Markdown('\n'.join(_l))

# %%
_errores = {('F', 'G'), ('D', 'G')}
_otras = [f'{p} → {a}' for p, a in faltan + sobran if (p, a) not in _errores]
Markdown('**Respuesta:** en el grafo falta la prelación F → G y sobra la prelación D → G.' + (f' Las demás diferencias de predecesoras inmediatas ({lista_y(_otras)}) son consecuencia de estas dos: al '
  'añadir o quitar una prelación cambia cuáles son redundantes.' if _otras else ''))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| fig-cap: "Grafo PERT correcto"
#| label: fig-pertCorrecto
proyecto.pert(durations=False, ranksep=1)

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
_lineas = explica_ficticias(proyecto, prelaciones)
_n, _m = len(proyecto.dummies), len(proyecto_dibujado.dummies)
Markdown(('\n'.join(_lineas) + '\n\n' if _lineas else '') + f'**Respuesta:** el grafo correcto tiene '
         f'{plural(_n, "ficticia")} frente a {_m} del dibujado: '
         + ('el número no cambia.' if _n == _m else f'{"una más" if _n - _m == 1 else (f"{_n - _m} más" if _n > _m else ("una menos" if _m - _n == 1 else f"{_m - _n} menos"))}.'))
