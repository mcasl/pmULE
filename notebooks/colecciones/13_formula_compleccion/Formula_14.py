# ---
# title: Formula 14
# subtitle: "10 actividades, 11 rutas; criticidad Monte Carlo frente a |v₁|"
# coleccion: 13_formula_compleccion
# variante: svd-vs-montecarlo
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
# Dos formas de medir la importancia de una actividad: el índice de criticidad de Monte Carlo (fracción de réplicas en que está en la ruta más larga; usa las distribuciones de duración) y el peso |v₁| de la SVD de la matriz de rutas (usa sólo la estructura). Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (normal) se adjunta.

# %%
pregunta_1 = "1. Calcule el índice de criticidad de cada actividad con 20 000 réplicas (`monte_carlo`).  "
pregunta_2 = "2. Calcule |v₁| de la SVD de R y ordene las actividades.  "
pregunta_3 = "3. Compare los dos rankings: correlación de Spearman y actividades en las que más discrepan. ¿Qué información tiene uno que no tenga el otro?  "
pregunta_4 = "4. Repita el Monte Carlo multiplicando todas las desviaciones típicas por 3. ¿Se parece más o menos el índice de criticidad al ranking SVD? ¿Por qué?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 3       , 0.1               ),
    ('B'     , '---'      , 8       , 0.5               ),
    ('C'     , '---'      , 1       , 1                 ),
    ('D'     , 'A'        , 4       , 0.1               ),
    ('E'     , 'B,C,D'    , 8       , 1                 ),
    ('F'     , 'D'        , 4       , 0.6               ),
    ('G'     , 'E,F'      , 3       , 0.5               ),
    ('H'     , 'E'        , 7       , 0.3               ),
    ('I'     , 'G'        , 2       , 0.6               ),
    ('J'     , 'F,H'      , 4       , 0.3               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

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
dists = {a: ('normal', float(datos_proyecto.loc[a, 'duration']), float(datos_proyecto.loc[a, 'standard_deviation'])) for a in R.columns}
mc = proyecto.monte_carlo(dists, n_samples=20000, seed=5)
criticidad = mc.sensitivity_mean.reindex(R.columns)
criticidad.sort_values(ascending=False).round(3).to_frame('índice de criticidad').T

# %%
orden_mc = list(criticidad.sort_values(ascending=False).index)
Markdown(f'''**Respuesta:** el índice de criticidad es la fracción de las 20 000 réplicas en que la actividad está en la ruta más larga. Las más críticas son {lista_y([f'{a} ({criticidad[a]:.3f})' for a in orden_mc[:3]])}; las menos, {lista_y([f'{a} ({criticidad[a]:.3f})' for a in orden_mc[-2:]])}.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)
v1 = pd.Series(np.abs(Vt[0, :]), index=R.columns)
v1.sort_values(ascending=False).round(3).to_frame('|v₁|').T

# %%
Markdown(f'**Respuesta:** ordenadas por |v₁| de mayor a menor: {", ".join(f"{a} ({v1[a]:.3f})" for a in v1.sort_values(ascending=False).index)} '
         '(las que tienen el mismo valor ocupan posiciones equivalentes en la estructura).')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
from scipy.stats import spearmanr
rho = spearmanr(criticidad, v1).correlation
comparacion = pd.DataFrame({'criticidad MC': criticidad, '|v₁|': v1})
comparacion['rango MC'] = comparacion['criticidad MC'].rank(ascending=False)
comparacion['rango SVD'] = comparacion['|v₁|'].rank(ascending=False)
comparacion['diferencia de rango'] = (comparacion['rango MC'] - comparacion['rango SVD']).abs()
display(comparacion.sort_values('diferencia de rango', ascending=False).round(3))
dif = comparacion.sort_values('diferencia de rango', ascending=False)
maxdif = dif['diferencia de rango'].iloc[0]
peores = dif.index[np.isclose(dif['diferencia de rango'], maxdif)].tolist()
detalle = lista_y([f"{a} (puesto {_num(dif.loc[a, 'rango MC'], 1)} por Monte Carlo y {_num(dif.loc[a, 'rango SVD'], 1)} por SVD)" for a in peores])
Markdown(f'**Respuesta:** la correlación de Spearman entre ambos rankings es ρ = **{rho:.2f}**. Donde más discrepan es en {detalle}.')

# %% [markdown]
# Monte Carlo sabe qué rutas son largas y variables; la SVD sólo sabe qué actividades son estructuralmente centrales. Una actividad por la que pasan muchas rutas cortas tiene |v₁| alto y criticidad baja; una actividad de una única ruta muy larga, lo contrario. Coinciden cuando la estructura y las duraciones apuntan a lo mismo.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
dists3 = {a: ('normal', m, 3 * s) for a, (_, m, s) in dists.items()}
mc3 = proyecto.monte_carlo(dists3, n_samples=20000, seed=5)
criticidad3 = mc3.sensitivity_mean.reindex(R.columns)
rho3 = spearmanr(criticidad3, v1).correlation
pd.DataFrame({'criticidad (σ)': criticidad, 'criticidad (3σ)': criticidad3, '|v₁|': v1}).round(3).T

# %%
cambio = 'sube' if rho3 > rho + 0.03 else ('baja' if rho3 < rho - 0.03 else 'apenas cambia')
Markdown(f'''**Respuesta:** con σ triplicada ρ pasa de {rho:.2f} a {rho3:.2f} ({cambio}). Lo esperable es que con más incertidumbre más rutas tengan ocasión de ser la más larga y el índice de criticidad se reparta según cuántas rutas pasan por cada actividad, que es justo lo que mide la SVD, de modo que los rankings se acerquen; con poca incertidumbre la criticidad se concentra en la ruta crítica determinista y la estructura importa menos. Si aquí el cambio es pequeño es porque la estructura y las duraciones ya apuntan en la misma dirección: las actividades centrales son también las de la ruta crítica.''')
