# ---
# title: Formula 12
# subtitle: "7 actividades, 7 nodos; incidencia y núcleo"
# coleccion: 13_formula_compleccion
# variante: nucleo-incidencia
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
# La matriz de incidencia $H$ del grafo PERT tiene una fila por arco (actividades y ficticias) y una columna por nodo, con −1 en el nodo origen del arco y +1 en el destino. Cada ruta es una cadena de arcos del nodo inicial al final. Considere el proyecto cuyo cuadro se adjunta.

# %%
pregunta_1 = "1. Dibuje el grafo PERT numerado y escriba la matriz de incidencia H (incluyendo las ficticias).  "
pregunta_2 = "2. Construya la matriz de rutas R *con* las ficticias (duración 0) y compruebe que R·H = 0 salvo en las columnas del nodo inicial y del final. ¿Por qué?  "
pregunta_3 = "3. Elija un vector y con un valor entero por nodo interior (0 en el inicial y el final) y defina la perturbación δ = H·y de las duraciones de los arcos. Compruebe que todas las rutas cambian de duración en la misma cantidad y que τ no cambia. Interprételo: ¿qué se ha hecho con los «tiempos de los nodos»?  "
pregunta_4 = "4. ¿Qué dimensión tiene el espacio de perturbaciones que dejan invariantes todas las duraciones de ruta? ¿Qué significa para la identificación de duraciones a partir de observaciones de rutas?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 8       ),
    ('B'     , '---'      , 3       ),
    ('C'     , '---'      , 6       ),
    ('D'     , 'A,B,C'    , 3       ),
    ('E'     , 'D'        , 7       ),
    ('F'     , 'C'        , 8       ),
    ('G'     , 'E,F'      , 4       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
def bmatrix(a, dec=0):
    """Matriz LaTeX (bmatrix) a partir de un DataFrame o array."""
    a = np.asarray(getattr(a, 'values', a), dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    filas = [' & '.join(f'{x:.{dec}f}' for x in fila) + r'\\' for fila in a]
    return r'\begin{bmatrix}' + '\n' + '\n'.join(filas) + '\n' + r'\end{bmatrix}'

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %%
def _num(x, dec=0):
    x = float(x)
    return (f'{x:.{dec}f}' if dec else f'{x:g}').replace('-', '−')


def producto_Rt(proyecto, R, t, base, decimales=0):
    """Cada fila de R por el vector t: (R·t)_r = Σ_i R_ri·t_i, con los términos no nulos (actividades de la
    ruta, en orden de ejecución) y el valor; marca las filas que dan el máximo."""
    cal = proyecto.calendar(base)
    orden = {a: k for k, a in enumerate(ordena_ruta(list(R.columns), cal))}
    filas = {}
    for r in R.index:
        acts = sorted([a for a in R.columns if R.loc[r, a] == 1], key=lambda a: orden.get(a, 0))
        total = float(sum(float(t[a]) for a in acts))
        filas[r] = {'actividades': ' – '.join(acts),
                    'fila de R · t = suma': ' + '.join(f'{a}({_num(t[a], decimales)})' for a in acts) + f' = {_num(total, decimales)}',
                    'R·t': total}
    tabla = pd.DataFrame(filas).T
    tabla['R·t'] = tabla['R·t'].astype(float)
    tabla['máximo'] = np.where(np.isclose(tabla['R·t'], tabla['R·t'].max()), '← τ', '')
    return tabla


def explica_tau(tabla, unidad='', decimales=0, nombre='τ'):
    """«τ = ‖R·t‖∞ = máx(19, 20, 17) = 20, que da la ruta Route_2 (A – C – E)»."""
    v = tabla['R·t']
    T = float(v.max())
    rutas = [f"{r} ({tabla.loc[r, 'actividades']})" for r in v.index if np.isclose(v[r], T)]
    u = f' {unidad}' if unidad else ''
    quien = (f'que corresponde a la ruta {rutas[0]}' if len(rutas) == 1
             else f'alcanzado a la vez por {len(rutas)} rutas: {lista_y(rutas)}')
    return (f"{nombre} = ‖R·t‖∞ = máx({', '.join(_num(x, decimales) for x in v)}) = **{_num(T, decimales)}{u}**, {quien}")

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT numerado"
#| label: fig-pert
proyecto.pert(durations=False, ranksep=1)

# %%
#| tbl-cap: "Matriz de incidencia H (arcos × nodos)"
#| label: tbl-H
H = proyecto.incidence_matrix()
H

# %%
Markdown(f'**Respuesta:** H tiene {H.shape[0]} filas (las {len(proyecto.actual_activities)} actividades y {H.shape[0] - len(proyecto.actual_activities)} '
         f'ficticias) y {H.shape[1]} columnas (nodos); cada fila tiene un −1 en el nodo donde empieza el arco y un +1 en el nodo donde termina.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Matriz de rutas con las ficticias (una columna por arco, en el mismo orden que las filas de H):

# %%
R = proyecto.path_matrix(dummies=True)[H.index]
display(R)
RH = R @ H
display(RH)
interiores = H.columns[1:-1]
r0 = R.index[0]
arcos_r0 = [a for a in R.columns if R.loc[r0, a] == 1]
nodo = interiores[0]
ejemplo = ' + '.join(f'({_num(H.loc[a, nodo])})' for a in arcos_r0 if H.loc[a, nodo] != 0)
cero = bool((RH[interiores] == 0).all().all())
Markdown(f'''Ejemplo: la fila {r0} de R tiene un 1 en los arcos {lista_y(arcos_r0)}; multiplicada por la columna del nodo {nodo} de H solo suman los arcos de la ruta que tocan ese nodo: {ejemplo or '0 (la ruta no pasa por el nodo)'} = {_num(RH.loc[r0, nodo])}.\n\n**Respuesta:** R·H es {'cero' if cero else 'distinto de cero'} en todas las columnas interiores; en la del nodo inicial vale {_num(RH.iloc[0, 0])} y en la del final {_num(RH.iloc[0, -1])} para todas las rutas.''')

# %% [markdown]
# Una ruta entra y sale de cada nodo interior exactamente una vez, así que la suma de sus entradas de incidencia es +1 − 1 = 0; sólo sale del inicial (−1) y sólo entra en el final (+1).

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
y_interior = {
    '2': 2,
    '3': 1,
    '4': -1,
    '5': -1,
    '6': -1,
}
y = pd.Series(0, index=H.columns, dtype=int)
for nodo, valor in y_interior.items():
    y[nodo] = valor
display(y.to_frame('y (0 en el nodo inicial y en el final)').T)
delta = H @ y
t = pd.Series({a: (datos_proyecto.loc[a, 'duration'] if not a.startswith('@') else 0) for a in H.index})
t_perturbado = t + delta
origen = {a: H.columns[H.loc[a] == -1][0] for a in H.index}
destino = {a: H.columns[H.loc[a] == 1][0] for a in H.index}
pd.DataFrame({'t': t, 'δ = y(destino) − y(origen)': [f'{_num(y[destino[a]])} − ({_num(y[origen[a]])}) = {_num(delta[a])}' for a in H.index],
              't + δ': t_perturbado}, index=H.index).T

# %%
Rt, Rt2 = R @ t, R @ t_perturbado
def _suma(v, r):
    acts = sorted([a for a in R.columns if R.loc[r, a] == 1], key=lambda a: list(H.columns).index(origen[a]))
    return ' + '.join(f'{a}({_num(v[a])})' for a in acts) + f' = {_num(sum(v[a] for a in acts))}'
display(pd.DataFrame({'R·t = suma': [_suma(t, r) for r in R.index], 'R·(t+δ) = suma': [_suma(t_perturbado, r) for r in R.index],
                      'diferencia': (Rt2 - Rt).astype(int)}, index=R.index))
Markdown(f'''**Respuesta:** τ = máx({', '.join(_num(x) for x in Rt)}) = {float(Rt.max()):.0f} ({Rt.idxmax()}) antes y máx({', '.join(_num(x) for x in Rt2)}) = {float(Rt2.max()):.0f} después: las duraciones de todas las rutas cambian en {int((Rt2 - Rt).iloc[0])} (= y_final − y_inicial = 0), así que τ es invariante. Con los «tiempos de los nodos» se ha hecho lo que explica el párrafo siguiente: desplazar el reloj de cada nodo interior.''')

# %% [markdown]
# δ = H·y añade a cada arco la diferencia de «potencial» entre su destino y su origen. Es lo mismo que mover el reloj de cada nodo: si al nodo j se le suma y_j a su tiempo, los arcos que llegan a j parecen y_j más largos y los que salen y_j más cortos, y a lo largo de cualquier ruta cerrada entre inicio y final todo se cancela. (Algunas duraciones perturbadas pueden salir negativas: es un ejercicio algebraico, no un calendario.)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
N = proyecto.nullspace()
rango = np.linalg.matrix_rank(N.values)
rango_R = np.linalg.matrix_rank(R.values.astype(float))
nucleo_R = len(t) - rango_R
Markdown(f'''**Respuesta:** las columnas interiores de H generan un espacio de dimensión {rango} (tantas como nodos interiores, {N.shape[1]}) contenido en el núcleo de R. Comprobación: R tiene {len(t)} columnas (arcos) y rango {rango_R}, así que su núcleo tiene dimensión {len(t)} − {rango_R} = **{nucleo_R}**{' = número de nodos interiores: esas son todas las perturbaciones posibles' if nucleo_R == rango else ''}. Hay por tanto {nucleo_R} direcciones independientes en las que pueden moverse las duraciones de los {len(t)} arcos sin que ninguna ruta cambie. Consecuencia: observando sólo duraciones de rutas no pueden identificarse las duraciones de los arcos; sólo se identifican salvo esas {nucleo_R} direcciones.''')
