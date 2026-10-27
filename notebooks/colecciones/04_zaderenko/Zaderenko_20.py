# ---
# title: Zaderenko 20
# subtitle: "9 actividades; duración 29, plazo impuesto 27 (1 rutas lo exceden)"
# coleccion: 04_zaderenko
# variante: plazo-impuesto
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta. El cliente exige que esté terminado en **27 semanas**.

# %%
pregunta_1 = "1. Calcule con Zaderenko los tiempos de los nodos y el calendario del proyecto. ¿Cuál es su duración? ¿Se cumple el plazo del cliente?  "
pregunta_2 = "2. Recalcule los tiempos tardíos tomando como tiempo tardío del último nodo el plazo impuesto (27) en lugar de la duración del proyecto. ¿Qué actividades tienen ahora holgura total negativa y qué significa?  "
pregunta_3 = "3. ¿Cuántas rutas del proyecto exceden el plazo y en cuánto cada una? ¿Hay alguna actividad que esté en todas ellas? ¿Cuántas semanas habría que acortarla para cumplir el plazo si fuera la única que se acorta?  "
pregunta_4 = "4. Si en lugar de acortar actividades se negociara el plazo, ¿cuál es el mínimo plazo que no obliga a acortar nada?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 8       ),
    ('B'     , '---'      , 2       ),
    ('C'     , 'A,B'      , 2       ),
    ('D'     , 'C'        , 1       ),
    ('E'     , 'D'        , 7       ),
    ('F'     , 'E'        , 7       ),
    ('G'     , 'E'        , 7       ),
    ('H'     , 'F'        , 2       ),
    ('I'     , 'G,H'      , 2       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
plazo = 27

# %%
_COLS_CAL = {'inicio_mas_temprano': 'ES', 'fin_mas_temprano': 'EF', 'inicio_mas_tardio': 'LS',
             'fin_mas_tardio': 'LF', 'H_total': 'H total'}


def tiempos_nodos(proyecto, duraciones):
    """Tiempos temprano y tardío de cada nodo (sucesos del grafo PERT)."""
    n = proyecto.calculate_pert(durations=duraciones)['nodes']
    t = pd.DataFrame({'tiempo temprano': n['early'].astype(int).values, 'tiempo tardío': n['late'].astype(int).values},
                     index=pd.Index([str(i) for i in n.index], name='nodo'))
    return t


def arcos_pert(proyecto):
    """Actividad → arco (nodo origen → nodo destino); las ficticias al final."""
    arcos = sorted(proyecto.nodes_from_edges.items(), key=lambda kv: (kv[0].startswith('@'), int(kv[1][0]), int(kv[1][1]), kv[0]))
    t = pd.DataFrame({'nodo origen i': [i for _, (i, j) in arcos], 'nodo destino j': [j for _, (i, j) in arcos]},
                     index=pd.Index([f'ficticia {a}' if a.startswith('@') else a for a, _ in arcos], name='actividad'))
    return t


def respuesta_duracion(proyecto, duraciones, unidad='días'):
    t = tiempos_nodos(proyecto, duraciones)
    finales = [str(i) for i in proyecto.pert_graph.nodes if proyecto.pert_graph.out_degree(i) == 0]
    f = finales[0]
    return Markdown(f"**Respuesta:** la duración del proyecto es el tiempo temprano del último nodo, el {f}: "
                    f"**{int(t.loc[f, 'tiempo temprano'])} {unidad}** (y su tiempo tardío se iguala a ese valor para "
                    f"empezar el cálculo hacia atrás).")


def fechas_desde_nodos(proyecto, duraciones):
    """Fechas y holgura total de cada actividad i → j a partir de los tiempos de sus nodos:
    ES = t_i, EF = t_i + d, LF = t*_j, LS = t*_j − d, H total = t*_j − t_i − d."""
    t = tiempos_nodos(proyecto, duraciones)
    filas = []
    for a, (i, j) in sorted(proyecto.nodes_from_edges.items()):
        if a.startswith('@'):
            continue
        d, ti, tj = int(duraciones[a]), int(t.loc[i, 'tiempo temprano']), int(t.loc[j, 'tiempo tardío'])
        filas.append({'actividad': a, 'arco i → j': f'{i} → {j}', 'ES = t_i': f'{ti}', 'EF = t_i + d': f'{ti} + {d} = {ti + d}',
                      'LF = t*_j': f'{tj}', 'LS = t*_j − d': f'{tj} − {d} = {tj - d}',
                      'H total = t*_j − t_i − d': f'{tj} − {ti} − {d} = {tj - ti - d}'})
    return pd.DataFrame(filas).set_index('actividad')


def respuesta_calendario(calendario, unidad='días'):
    crit = calendario.index[calendario['H_total'] == 0].tolist()
    holg = calendario.index[calendario['H_total'] > 0].tolist()
    return Markdown(f"**Respuesta:** el calendario es el de la tabla. Las actividades críticas (holgura total 0) son "
                    f"{lista_y(crit)}; las demás pueden retrasarse sin alargar el proyecto hasta su holgura total: "
                    + lista_y(f"{a} {plural(int(calendario.loc[a, 'H_total']), unidad[:-1] if unidad.endswith('s') else unidad)}" for a in holg) + '.')

def respuesta_critico(proyecto, duraciones, unidad='días'):
    """Camino(s) crítico(s) como suma y respuesta."""
    display(explica_camino_critico(proyecto, duraciones, unidad=unidad))
    cal = proyecto.calendar(duraciones)
    rutas = [' – '.join(ordena_ruta(r, cal)) for r in proyecto.critical_path(duraciones).values()]
    return Markdown(f"**Respuesta:** {plural(len(rutas), 'camino crítico', 'caminos críticos')}: {lista_y(rutas)}; "
                    f"son las actividades con holgura total 0 en el calendario.")


def deduce_prelaciones(matriz):
    """Cuadro de prelaciones a partir de la matriz de Zaderenko con nombres: (datos, tabla del razonamiento)."""
    arcos = {}
    for i in matriz.index:
        for j in matriz.columns:
            txt = matriz.loc[i, j]
            if txt:
                arcos[(str(i), str(j))] = ('ficticia', 0) if txt == 'ficticia' else (txt.split(' ')[0], int(txt.split('(')[1].rstrip(')')))

    def entran(nodo):
        acts, via = set(), []
        for (i, j), (nombre, _) in sorted(arcos.items()):
            if j == nodo:
                if nombre == 'ficticia':
                    sub, _ = entran(i)
                    acts |= sub
                    via.append(f"ficticia {i} → {j}, que trae {lista_y(sorted(sub))} (lo que llega al nodo {i})")
                else:
                    acts.add(nombre)
                    via.append(f"{nombre} (arco {i} → {j})")
        return acts, via

    filas, prel, dur = [], {}, {}
    for (i, j), (nombre, d) in sorted(arcos.items(), key=lambda kv: kv[1][0]):
        if nombre == 'ficticia':
            continue
        acts, via = entran(i)
        prel[nombre], dur[nombre] = ','.join(sorted(acts)) or '---', d
        filas.append({'actividad': nombre, 'arco': f'{i} → {j}',
                      'llega al nodo origen': '; '.join(via) or f'nada: el nodo {i} es el inicial',
                      'predecesoras inmediatas': ', '.join(sorted(acts)) or '— (inicial)'})
    datos = pd.DataFrame({'predecessor': pd.Series(prel), 'duration': pd.Series(dur)}).sort_index()
    datos.index.name = 'activity'
    return datos, pd.DataFrame(filas).set_index('actividad')


def compara_matriz(matriz, proyecto):
    """¿Reproduce el grafo reconstruido los arcos de la matriz del enunciado?"""
    enun = {}
    for i in matriz.index:
        for j in matriz.columns:
            txt = matriz.loc[i, j]
            if txt and txt != 'ficticia':
                enun[txt.split(' ')[0]] = (str(i), str(j))
    rec = {a: ij for a, ij in proyecto.nodes_from_edges.items() if not a.startswith('@')}
    n_fict_e = int((matriz == 'ficticia').sum().sum())
    n_fict_r = sum(a.startswith('@') for a in proyecto.nodes_from_edges)
    distintos = [a for a in sorted(enun) if enun[a] != rec.get(a)]
    if not distintos:
        txt = (f"**Respuesta:** sí la reproduce: las {len(enun)} actividades están en los mismos arcos (i → j) que en la "
               f"matriz del enunciado, con la misma numeración de nodos, y hay {n_fict_r} ficticias "
               f"({'las mismas' if n_fict_r == n_fict_e else f'{n_fict_e} en el enunciado'}).")
    else:
        txt = (f"**Respuesta:** el grafo tiene la misma estructura, pero la numeración de algunos nodos difiere (hay varias "
               f"numeraciones válidas): " + '; '.join(f"{a}: {enun[a][0]} → {enun[a][1]} en el enunciado, "
                                                       f"{rec[a][0]} → {rec[a][1]} en el grafo" for a in distintos) + '.')
    return Markdown(txt)


def rutas_por_actividad(proyecto, duraciones):
    """Para cada actividad, la ruta más larga que pasa por ella y su duración."""
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    d = pd.Series(duraciones)
    largo = {r: int(sum(d[a] for a in R.columns[R.loc[r] == 1])) for r in R.index}
    res = {}
    for a in R.columns:
        rs = [r for r in R.index if R.loc[r, a] == 1]
        r = max(rs, key=lambda r: largo[r])
        res[a] = (' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)), largo[r])
    return res, max(largo.values())


def respuesta_forma(proyecto, duraciones, unidad='días'):
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    n_acts = R.sum(axis=1)
    mas = n_acts[n_acts == n_acts.max()].index
    nombres = [' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)) for r in mas]
    criticas = [' – '.join(ordena_ruta(r, cal)) for r in proyecto.critical_path(duraciones).values()]
    coincide = set(nombres) & set(criticas)
    return Markdown(f"**Respuesta:** mirando solo la forma, las candidatas son las rutas con más actividades: "
                    f"{lista_y(nombres)} ({int(n_acts.max())} actividades). Con las duraciones (apartado 3) la ruta más "
                    f"larga resulta ser {lista_y(criticas)}"
                    + (', que sí está entre ellas.' if coincide else
                       ', que no está entre ellas: la forma del grafo no basta, hacen falta las duraciones.'))


def respuesta_prediccion(proyecto, duraciones, casos, unidad='días'):
    """Predicción de cada caso sumando las duraciones de las rutas (sin tiempos de los nodos)."""
    mayor, T = rutas_por_actividad(proyecto, duraciones)
    L = [f"**Respuesta** (la ruta más larga del proyecto suma {T} {unidad}):", '']
    for letra, (a, extra) in zip('abc', casos):
        ruta, l = mayor[a]
        margen = T - l
        if margen == 0:
            pred = f"está en la ruta más larga, {ruta} ({l}): cualquier aumento alarga el plazo, **sí cambia** (+{extra})"
        elif extra <= margen:
            pred = (f"la ruta más larga que pasa por {a} es {ruta}, con {l}, {T} − {l} = {plural(margen, unidad[:-1])} menos que la "
                    f"más larga: el aumento de {extra} cabe ({extra} ≤ {margen}), **no cambia**")
        else:
            pred = (f"la ruta más larga que pasa por {a} es {ruta}, con {l}, {T} − {l} = {plural(margen, unidad[:-1])} menos que la "
                    f"más larga: el aumento de {extra} no cabe ({extra} > {margen}), **sí cambia** (+{extra - margen})")
        L.append(f"- ({letra}) {a} dura {extra} {unidad} más: {pred}.")
    return Markdown('\n'.join(L))


def respuesta_comprobacion(calendario, T0, casos, unidad='días'):
    L = []
    for letra, (a, extra) in zip('abc', casos):
        h = int(calendario.loc[a, 'H_total'])
        if extra <= h:
            L.append(f"- ({letra}) {a}: holgura total {h}, aumento {extra} ≤ {h}: el plazo sigue en {T0} {unidad}.")
        else:
            L.append(f"- ({letra}) {a}: holgura total {h}, aumento {extra} > {h}: el plazo pasa a {T0} + ({extra} − {h}) = {T0 + extra - h} {unidad}.")
    return Markdown('\n'.join(L) + "\n\n**Respuesta:** las tres predicciones del apartado 2 se confirman. El dato que permitía acertar es la "
                    "**holgura total** de cada actividad: el plazo cambia si y sólo si el aumento la supera (en una crítica es "
                    "0 y cualquier aumento cuenta), y el proyecto se alarga exactamente el exceso.")

def nodos_con_plazo(proyecto, duraciones, plazo):
    """Tiempos de los nodos con el tiempo tardío del último nodo igual al plazo impuesto."""
    t = tiempos_nodos(proyecto, duraciones)
    k = int(proyecto.duration(duraciones)) - int(plazo)
    t[f'tardío con plazo = tardío − {k}'] = t['tiempo tardío'] - k
    return t.T


def respuesta_negativas(con_plazo, k, unidad='semanas'):
    neg = con_plazo.index[con_plazo['H_total'] < 0].tolist()
    return Markdown(f"**Respuesta:** {plural(len(neg), 'actividad tiene', 'actividades tienen')} holgura total negativa: "
                    + lista_y(f"{a} ({int(con_plazo.loc[a, 'H_total'])})".replace('(-', '(−') for a in neg)
                    + f". Una holgura −h significa que, aun empezando lo antes posible, la actividad acaba h {unidad} después de "
                    f"lo que permite el plazo (su fin más temprano supera en h su nuevo fin más tardío): la ruta más larga que "
                    f"pasa por ella excede el plazo en h {unidad}. Las críticas tienen −{k}, todo el exceso del proyecto; para "
                    f"cumplir el plazo hay que recuperar ese tiempo acortando actividades de esas rutas.")


def rutas_exceden(proyecto, duraciones, plazo):
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    d = pd.Series(duraciones)
    filas = []
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        L = int(sum(d[a] for a in acts))
        if L > plazo:
            filas.append({'ruta': ' – '.join(acts), 'duración = suma': ' + '.join(str(int(d[a])) for a in acts) + f' = {L}',
                          'exceso = duración − plazo': f'{L} − {plazo} = {L - plazo}', '_exceso': L - plazo, '_acts': acts})
    return pd.DataFrame(filas).sort_values('_exceso', ascending=False, kind='stable').reset_index(drop=True)


def respuesta_exceden(tabla, duraciones, unidad='semanas'):
    d = pd.Series(duraciones)
    n = len(tabla)
    m = int(tabla['_exceso'].max())
    comunes = [a for a in tabla['_acts'].iloc[0] if all(a in acts for acts in tabla['_acts'])]
    txt = (f"**Respuesta:** {plural(n, 'ruta excede', 'rutas exceden')} el plazo: "
           + lista_y(f"{r} en {e}" for r, e in zip(tabla['ruta'], tabla['_exceso'])) + '. ')
    if comunes:
        txt += ((f"{lista_y(comunes)} {'está' if len(comunes) == 1 else 'están'} en todas ellas. " if n > 1 else
                 f"Al ser la única, cualquiera de sus actividades ({lista_y(comunes)}) está en todas. ")
                + f"Si se acorta una sola, hay "
                f"que acortarla {m} {unidad}, el mayor exceso (con menos, la ruta que más excede seguiría por encima del "
                f"plazo; las rutas que no exceden no importan): ")
        txt += '; '.join((f"{a} pasaría de {int(d[a])} a {int(d[a]) - m}" if int(d[a]) > m else
                          f"{a} no puede (solo dura {int(d[a])})") for a in comunes) + '.'
    else:
        txt += 'Ninguna actividad está en todas ellas: habrá que acortar al menos dos actividades distintas.'
    return Markdown(txt)


def fines_gantt(datos):
    t = pd.DataFrame({'inicio': datos['start'].astype(int), 'duración': datos['duration'].astype(int),
                      'fin = inicio + duración': [f"{int(s)} + {int(d)} = {int(s + d)}" for s, d in zip(datos['start'], datos['duration'])]})
    t.index.name = 'actividad'
    return t


def respuesta_cadenas(cadenas):
    nombres = [' – '.join(c) for c in cadenas]
    if len(cadenas) == 1:
        return Markdown(f"**Respuesta:** hay una sola cadena de actividades ajustadas de la semana 0 al final: **{nombres[0]}**. "
                        "Toda ruta crítica es una cadena ajustada (en ella cada actividad empieza justo cuando acaba la "
                        "anterior) y el proyecto tiene al menos una, así que esta cadena es la ruta crítica, y es única.")
    return Markdown(f"**Respuesta:** hay {len(cadenas)} cadenas ajustadas de la semana 0 al final: {lista_y(nombres)}. "
                    "Al menos una es ruta crítica (toda ruta crítica es una cadena ajustada), pero con solo el Gantt no se "
                    "sabe cuáles: un eslabón puede ser una coincidencia de fechas entre actividades que no son "
                    "predecesoras. Sí puede haber más de una ruta crítica, si existen las prelaciones de varias cadenas.")


def respuesta_recuperado(datos, ajustadas, prelaciones, proyecto, calendario, cadenas):
    pred = prelaciones.immediate_predecessor
    G = nx.DiGraph([(p, a) for a, ps in pred.items() for p in ps])
    G.add_nodes_from(datos.index)
    iguales = int((calendario['inicio_mas_temprano'].astype(int) == datos['start'].astype(int)).sum())
    L = [f"Los inicios más tempranos calculados con el cuadro recuperado coinciden con los del Gantt en "
         f"{iguales} de {len(datos)} actividades: " + ('**es compatible**.' if iguales == len(datos) else '**no es compatible**.')]
    ocultas = [(p, a) for a in datos.index for p in sorted(pred[a]) if p not in ajustadas[a]]
    if ocultas:
        L.append('- Predecesoras reales que no son ajustadas (acaban antes de que empiece la sucesora; el Gantt no las '
                 'delata): ' + '; '.join(f"{p} → {a} ({p} acaba en {int(datos.loc[p, 'end'])}, {a} empieza en {int(datos.loc[a, 'start'])})" for p, a in ocultas) + '.')
    no_inm = [(p, a) for a in datos.index for p in ajustadas[a] if p not in pred[a] and p in nx.ancestors(G, a)]
    if no_inm:
        L.append('- Ajustadas que son antecesoras pero no predecesoras inmediatas: ' + '; '.join(f'{p} de {a}' for p, a in no_inm) + '.')
    coinc = [(p, a) for a in datos.index for p in ajustadas[a] if p not in nx.ancestors(G, a)]
    if coinc:
        L.append('- Ajustadas por coincidencia de fechas (no preceden a la actividad): ' + '; '.join(f'{p} de {a}' for p, a in coinc) + '.')
    if not (ocultas or no_inm or coinc):
        L.append('- Aquí las predecesoras ajustadas coinciden exactamente con las reales.')
    criticas = [' – '.join(ordena_ruta(r, calendario)) for r in proyecto.critical_path(calendario['duracion']).values()]
    nombres = [' – '.join(c) for c in cadenas]
    L.append(f"\n**Respuesta:** el cuadro recuperado {'reproduce' if iguales == len(datos) else 'no reproduce'} el Gantt. "
             f"{'La ruta crítica real es' if len(criticas) == 1 else 'Las rutas críticas reales son'} {lista_y(criticas)}"
             + (f"; de las cadenas deducidas, {lista_y([c for c in nombres if c not in criticas])} no lo "
                f"{'es' if len([c for c in nombres if c not in criticas]) == 1 else 'son'}." if any(c not in criticas for c in nombres) else
                ', la misma que se dedujo del Gantt.')
             + " El Gantt no determina el cuadro de prelaciones: oculta las prelaciones con holgura y puede sugerir "
             "otras por coincidencia de fechas.")
    return Markdown(L[0] + '\n\n' + '\n'.join(L[1:]))

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Algoritmo de Zaderenko paso a paso, con el grafo PERT arriba y la matriz debajo. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko"
#| label: tbl-zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duration'])

# %% [markdown]
# Tiempos de los nodos con los candidatos que se comparan en cada uno (tempranos: máximo de «tiempo temprano del origen + duración»; tardíos: mínimo de «tiempo tardío del destino − duración»):

# %%
#| tbl-cap: "Tiempos tempranos: candidatos en cada nodo (de izquierda a derecha)"
#| label: tbl-tempranosPaso
tiempos_tempranos, tiempos_tardios = detalle_tiempos_nodos(proyecto, datos_proyecto['duration'])
tiempos_tempranos

# %%
#| tbl-cap: "Tiempos tardíos: candidatos en cada nodo (de derecha a izquierda)"
#| label: tbl-tardiosPaso
tiempos_tardios

# %% [markdown]
# Para una actividad que va del nodo i al nodo j (con $t_i$ el tiempo temprano de i y $t^*_j$ el tardío de j): el inicio más temprano es $ES = t_i$, el fin más temprano $EF = t_i + d$, el fin más tardío $LF = t^*_j$ y el inicio más tardío $LS = t^*_j - d$; la holgura total es $H_t = t^*_j - t_i - d$. Con los números de los nodos:

# %%
#| tbl-cap: "Fechas de cada actividad a partir de los tiempos de sus nodos"
#| label: tbl-fechasPaso
fechas_desde_nodos(proyecto, datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion_proyecto = proyecto.duration(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %%
Markdown(f'**Respuesta:** la duración del proyecto es el tiempo temprano del último nodo, **{duracion_proyecto} semanas**; '
         f'supera el plazo del cliente ({plazo}) en {duracion_proyecto} − {plazo} = {plural(duracion_proyecto - plazo, "semana")}: **no se cumple**.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Los tiempos tempranos no dependen del plazo. Los tardíos sí: si el último nodo debe alcanzarse en la semana del plazo, todos los tiempos tardíos (y los inicios y fines más tardíos de las actividades) se adelantan en la diferencia entre la duración y el plazo, y la holgura total de cada actividad disminuye en esa misma cantidad. Tiempos de los nodos:

# %%
#| tbl-cap: "Tiempos de los nodos con el plazo impuesto"
#| label: tbl-nodosPlazo
retraso = int(duracion_proyecto - plazo)
nodos_con_plazo(proyecto, datos_proyecto['duration'], plazo)

# %% [markdown]
# Fechas de las actividades con esos tiempos tardíos (LS y LF bajan en la diferencia; H total = LS − ES):

# %%
#| tbl-cap: "Calendario con el plazo impuesto"
#| label: tbl-calendarioPlazo
con_plazo = calendario[['inicio_mas_temprano', 'fin_mas_temprano']].copy()
con_plazo['inicio_mas_tardio'] = calendario['inicio_mas_tardio'] - retraso
con_plazo['fin_mas_tardio'] = calendario['fin_mas_tardio'] - retraso
con_plazo['H_total'] = con_plazo['inicio_mas_tardio'] - con_plazo['inicio_mas_temprano']
con_plazo.astype(int).rename(columns=_COLS_CAL).rename_axis('actividad')

# %%
respuesta_negativas(con_plazo, retraso)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Rutas cuya duración (suma de las duraciones de sus actividades, en orden de ejecución) supera el plazo:

# %%
#| tbl-cap: "Rutas que exceden el plazo"
#| label: tbl-rutasExceden
exceden = rutas_exceden(proyecto, datos_proyecto['duration'], plazo)
exceden.drop(columns=['_exceso', '_acts'])

# %%
respuesta_exceden(exceden, datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
Markdown(f'**Respuesta:** {duracion_proyecto} semanas, la duración del proyecto. Con ese plazo el tiempo tardío del último '
         f'nodo coincide con su tiempo temprano, las críticas tienen holgura total 0 y las demás positiva; con cualquier '
         f'plazo menor habría holguras negativas y habría que acortar alguna actividad.')
