# ---
# title: Formula 17
# subtitle: "8 actividades, 4 rutas; τ como norma infinito"
# coleccion: 13_formula_compleccion
# variante: tau-norma-infinito
# nivel: 1
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
# La duración de un proyecto puede escribirse como una fórmula cerrada: si $R$ es la matriz de rutas (una fila por cada ruta del nodo inicial al final, una columna por actividad, con un 1 donde la actividad pertenece a la ruta) y $t$ el vector de duraciones, entonces
#
# $$\tau = \lVert R\,t \rVert_\infty = \max_r \sum_i R_{ri}\,t_i .$$
#
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (días) se adjunta.

# %%
pregunta_1 = "1. Dibuje el grafo PERT, enumere sus rutas y construya la matriz de rutas R. ¿Qué dimensiones tiene?  "
pregunta_2 = "2. Calcule el vector R·t y su norma infinito. Compruebe que coincide con el tiempo más temprano del último nodo obtenido por el método de Zaderenko.  "
pregunta_3 = "3. ¿Por qué la norma infinito y no otra? ¿Qué daría ‖R·t‖₁? ¿Y qué significaría el mínimo de R·t?  "
pregunta_4 = "4. Escriba τ en función de t_G (la duración de G) dejando fijas las demás, y dibuje esa función. ¿Es lineal? ¿Dónde cambia de pendiente?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 8       ),
    ('B'     , '---'      , 5       ),
    ('C'     , 'A,B'      , 5       ),
    ('D'     , 'B'        , 8       ),
    ('E'     , 'C'        , 7       ),
    ('F'     , 'D'        , 7       ),
    ('G'     , 'E'        , 3       ),
    ('H'     , 'D,G'      , 8       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la instalación de una planta fotovoltaica sobre cubierta; las actividades son las siguientes (las duraciones del cuadro están en días).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de sombras y producción',
    'B': 'Cálculo estructural de la cubierta',
    'C': 'Tramitación de permisos',
    'D': 'Acopio de módulos e inversores',
    'E': 'Montaje de la estructura soporte',
    'F': 'Colocación de módulos',
    'G': 'Cableado de corriente continua',
    'H': 'Instalación de inversores',
}, name='Descripción')
descripciones.to_frame()

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
R = proyecto.path_matrix(dummies=False)
t = datos_proyecto['duration'][R.columns]

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
#| fig-cap: "Grafo Roy"
#| label: fig-pert
proyecto.roy(durations=False, ranksep=1.2, nodesep=1.2)

# %%
#| tbl-cap: "Matriz de rutas R"
#| label: tbl-R
rutas = proyecto.paths(dummies=False)
display(pd.Series({r: ' - '.join(a) for r, a in rutas.items()}).to_frame('actividades'))
proyecto.display_rutas("roy")

# %%
Markdown(f'**Respuesta:** el grafo tiene {R.shape[0]} rutas del nodo inicial al final (tabla); R tiene {R.shape[0]} filas (una por ruta) '
         f'y {R.shape[1]} columnas (una por actividad real; las ficticias duran 0 y no se incluyen): dimensiones {R.shape[0]} × {R.shape[1]}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
Rt = R @ t
tau = float(Rt.max())
from pmule import display_producto_rutas
display_producto_rutas(R, t, Rt)

# %% [markdown]
# Cada componente de $R\,t$ es el producto de una fila de $R$ por $t$: como la fila tiene un 1 en las actividades de la ruta y un 0 en las demás, el producto es la suma de las duraciones de esa ruta. La norma infinito es el mayor de esos valores.

# %%
producto = producto_Rt(proyecto, R, t, datos_proyecto['duration'])
display(producto)
Markdown(explica_tau(producto, 'días') + '.')

# %% [markdown]
# Algoritmo de Zaderenko paso a paso, con el grafo PERT arriba y la matriz debajo. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko"
#| label: tbl-zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duration'])

# %%
T_zad = proyecto.duration(datos_proyecto['duration'])
Markdown(f'**Respuesta:** R·t = ({", ".join(_num(x) for x in Rt)}) y ‖R·t‖∞ = {tau:.0f} días; el tiempo más temprano del último nodo por Zaderenko es {T_zad} días '
         f'y la librería (`proyecto.tau`) da {proyecto.tau(t.to_dict()):.0f}. Las tres coinciden: el cálculo de tiempos hacia delante es una forma de calcular el '
         'máximo de las rutas sin enumerarlas.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Respuesta:** ‖R·t‖₁ = {' + '.join(_num(x) for x in Rt)} = {float(Rt.abs().sum()):.0f} sumaría las duraciones de todas las rutas, contando varias veces cada actividad compartida: no tiene sentido como plazo. El mínimo de R·t, {float(Rt.min()):.0f} ({Rt.idxmin()}), es la ruta más corta: el tiempo que *como mínimo* ocupa cualquier cadena del proyecto, pero el proyecto no termina hasta que acaba la última ruta. El plazo es el máximo porque todas las rutas deben completarse; de ahí la norma infinito.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
k = 'G'
valores = np.arange(0, int(t[k]) + 8)
taus = []
for v in valores:
    t2 = t.copy(); t2[k] = v
    taus.append(float((R @ t2).max()))
pendiente = np.diff(taus)
codo = int(valores[np.argmax(pendiente > 0)]) if (pendiente > 0).any() else None
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(valores, taus, marker='o'); ax.axvline(int(t[k]), color='grey', linestyle='--')
ax.set_xlabel(f'duración de {k}'); ax.set_ylabel('τ'); plt.show()

# %%
con_k = [r for r in R.index if R.loc[r, k] == 1]
sin_k = [r for r in R.index if R.loc[r, k] == 0]
resto = (Rt[con_k] - t[k])                     # lo que suman las demás actividades de cada ruta con k
c1 = float(resto.max())
c0 = float(Rt[sin_k].max()) if sin_k else None
if c0 is None:
    formula = f'τ(t_{k}) = {_num(c1)} + t_{k}: todas las rutas pasan por {k}, así que τ es lineal con pendiente 1 para cualquier duración'
elif c0 - c1 <= 0:
    formula = (f'τ(t_{k}) = máx({_num(c0)}, {_num(c1)} + t_{k}) = {_num(c1)} + t_{k} para cualquier t_{k} ≥ 0, porque la ruta más larga con {k} ({resto.idxmax()}) '
               f'ya supera sin {k} a la más larga sin {k} ({Rt[sin_k].idxmax()}, {_num(c0)}): pendiente 1 en todo el dominio, sin cambio de pendiente')
else:
    codo_f = c0 - c1
    formula = (f'τ(t_{k}) = máx({_num(c0)}, {_num(c1)} + t_{k}), donde {_num(c0)} es la ruta más larga sin {k} ({Rt[sin_k].idxmax()}) y {_num(c1)} lo que suman las demás '
               f'actividades de la ruta más larga con {k} ({resto.idxmax()}). Es lineal a trozos y convexa: pendiente 0 mientras t_{k} < {_num(c0)} − {_num(c1)} = {_num(c0 - c1)} '
               f'(manda la ruta sin {k}) y pendiente 1 a partir de ahí; el cambio de pendiente está en t_{k} = {_num(codo_f)}')
actual = c1 + t[k]
if c0 is None or actual > c0:
    sens = f'con t_{k} = {int(t[k])}, {k} está en la ruta crítica: alargarla o acortarla un día cambia τ un día (pendiente 1)'
elif actual < c0:
    sens = f'con t_{k} = {int(t[k])}, {k} no es crítica: puede alargarse hasta {_num(c0 - c1)} sin cambiar τ (pendiente 0)'
else:
    sens = f'con t_{k} = {int(t[k])} el valor actual está justo en el codo: alargarla aumenta τ, acortarla no lo reduce'
lineal = c0 is None or c0 - c1 <= 0
previo = 'En el rango de duraciones posibles es lineal; ' if lineal else 'No es lineal (es el máximo de funciones lineales); '
Markdown(f'**Respuesta:** {formula}. {previo}{sens}.')
