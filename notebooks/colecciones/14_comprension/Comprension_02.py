# ---
# title: Comprension 02
# subtitle: "8 actividades; duraciones de D y C en intervalos; cotas de τ"
# coleccion: 14_comprension
# variante: datos-insuficientes
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
# Considere el proyecto cuyo cuadro de prelaciones se adjunta. De dos actividades sólo se conoce un intervalo de duraciones posibles (en semanas), anotado como `mín-máx`.

# %%
pregunta_1 = "1. Sin conocer las dos duraciones, ¿qué puede afirmarse con seguridad sobre la duración del proyecto? Dé la mejor cota inferior y la mejor cota superior y diga con qué combinación de valores se alcanza cada una.  "
pregunta_2 = "2. ¿Hay actividades que son críticas *para cualquier* valor de las duraciones desconocidas? ¿Y actividades que no lo son nunca? Use la matriz de rutas.  "
pregunta_3 = "3. Para qué valores de la duración de C deja de tener holgura (se vuelve crítica), suponiendo D en su valor mínimo? ¿Y suponiéndola en su valor máximo?  "
pregunta_4 = "4. ¿Qué dato adicional reduciría más la incertidumbre sobre la duración del proyecto: conocer la duración de D o la de C? Justifíquelo con las cotas.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , '6'     ),
    ('B'     , '---'      , '5'     ),
    ('C'     , '---'      , '1-7'   ),
    ('D'     , 'B,C'      , '1-5'   ),
    ('E'     , 'A,D'      , '8'     ),
    ('F'     , 'E'        , '6'     ),
    ('G'     , 'F'        , '2'     ),
    ('H'     , 'F'        , '6'     ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
def rango(s):
    s = str(s)
    return (int(s.split('-')[0]), int(s.split('-')[1])) if '-' in s else (int(s), int(s))
minimos = datos_proyecto['duration'].map(lambda s: rango(s)[0])
maximos = datos_proyecto['duration'].map(lambda s: rango(s)[1])
X, Y = 'D', 'C'

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# La duración del proyecto es la de la ruta más larga: τ = ‖R·t‖∞, el máximo de las duraciones de las rutas (R·t, una fila de la matriz de rutas R por ruta). Alargar una actividad nunca acorta ninguna ruta, así que τ crece con cada duración: la cota inferior se alcanza con las dos desconocidas en su mínimo y la superior con las dos en su máximo, sin necesidad de probar otras combinaciones. La tabla da la duración de cada ruta en esos dos casos.

# %%
Rt_min = R @ minimos[R.columns]; Rt_max = R @ maximos[R.columns]
cota_inf, cota_sup = int(Rt_min.max()), int(Rt_max.max())
_acts = {r: list(R.columns[R.loc[r] == 1]) for r in R.index}
_suma = lambda r, t: ' + '.join(str(int(t[a])) for a in _acts[r]) + f' = {int(sum(t[a] for a in _acts[r]))}'
display(pd.DataFrame({'actividades': {r: ' – '.join(_acts[r]) for r in R.index},
                      f'duración con {X} = {minimos[X]}, {Y} = {minimos[Y]}': {r: _suma(r, minimos) for r in R.index},
                      f'duración con {X} = {maximos[X]}, {Y} = {maximos[Y]}': {r: _suma(r, maximos) for r in R.index}}))
_r_inf = [' – '.join(_acts[r]) for r in R.index if Rt_min[r] == cota_inf]
_r_sup = [' – '.join(_acts[r]) for r in R.index if Rt_max[r] == cota_sup]
Markdown(f'**Respuesta:** {cota_inf} ≤ τ ≤ {cota_sup} semanas. La cota inferior, {cota_inf}, se alcanza con {X} = {minimos[X]} y {Y} = {minimos[Y]} '
         f'(la ruta más larga es entonces {lista_y(_r_inf, "o")}); la superior, {cota_sup}, con {X} = {maximos[X]} y {Y} = {maximos[Y]} '
         f'(ruta {lista_y(_r_sup, "o")}). Ningún valor intermedio puede dar menos de {cota_inf} ni más de {cota_sup}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Una actividad es crítica para un juego de duraciones si está en alguna ruta de duración máxima. Se prueban todas las combinaciones enteras de las dos duraciones desconocidas; la tabla muestra las cuatro combinaciones extremas como ejemplo:

# %%
from itertools import product
def _criticas(dX, dY):
    t = minimos.copy(); t[X], t[Y] = dX, dY
    Rt = R @ t[R.columns]
    return int(Rt.max()), set(R.columns[(R.loc[Rt == Rt.max()].sum() > 0)])
display(pd.DataFrame([{X: dX, Y: dY, 'τ': _criticas(dX, dY)[0], 'actividades críticas': ', '.join(sorted(_criticas(dX, dY)[1]))}
                      for dX in (minimos[X], maximos[X]) for dY in (minimos[Y], maximos[Y])]).set_index([X, Y]))
siempre, nunca = set(R.columns), set(R.columns)
_n = 0
for dX, dY in product(range(minimos[X], maximos[X] + 1), range(minimos[Y], maximos[Y] + 1)):
    criticas_ahora = _criticas(dX, dY)[1]; _n += 1
    siempre &= criticas_ahora; nunca -= criticas_ahora
_depende = sorted(set(R.columns) - siempre - nunca)
Markdown(f'Recorriendo las {_n} combinaciones posibles: **críticas para cualquier valor:** {lista_y(sorted(siempre))}; '
         f'**nunca críticas:** {lista_y(sorted(nunca))}' + (f'; las demás ({lista_y(_depende)}) lo son o no según los valores.' if _depende else '.'))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# La holgura total de C es lo que le falta a la ruta más larga que pasa por C para igualar la duración del proyecto: H_total(C) = τ − (duración de la ruta más larga que contiene C). La tabla da esa holgura para cada duración posible de C, con D en su mínimo y en su máximo (las demás actividades con su duración conocida).

# %%
def holgura_Y(dX, dY):
    t = minimos.copy(); t[X], t[Y] = dX, dY
    return int(proyecto.calendar(t).loc[Y, 'H_total'])
tabla = pd.DataFrame({f'{X} = {dX}': {dY: holgura_Y(dX, dY) for dY in range(minimos[Y], maximos[Y] + 1)} for dX in (minimos[X], maximos[X])})
tabla.index.name = f'duración de {Y}'
display(tabla.T.rename_axis(f'holgura total de {Y}'))
umbrales = {col: next((dY for dY, h in tabla[col].items() if h == 0), None) for col in tabla.columns}
_u = list(umbrales.values())
_txt = '; '.join(f'con {col}, a partir de {Y} = {u}' if u is not None else f'con {col}, nunca (su holgura mínima es {int(tabla[col].min())})'
                 for col, u in umbrales.items())
_misma_ruta = bool(((R[X] == 1) & (R[Y] == 1)).any())
if _u[0] == _u[1]:
    _por = (f'El umbral no depende de {X}: {X} y {Y} están en una misma ruta, de modo que alargar {X} alarga a la vez la ruta de {Y} y la duración del proyecto.'
            if _misma_ruta and X in siempre else f'El umbral no depende de la duración de {X}.')
elif _u[1] is None or (_u[0] is not None and _u[1] > _u[0]):
    _por = f'Cuanto más dura {X}, más larga es la ruta crítica sin {Y} y más holgura le queda a {Y}: se vuelve crítica más tarde.'
else:
    _por = f'Cuanto más dura {X}, antes se vuelve crítica {Y}.'
Markdown(f'**Respuesta:** {Y} pasa a tener holgura total 0 (se vuelve crítica) {_txt}. {_por}')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Se compara el intervalo [cota inferior, cota superior] de τ que queda al conocer cada dato: fijada una de las dos duraciones en cada uno de sus valores posibles, la otra sigue variando entre su mínimo y su máximo.

# %%
filas = []
for fija, otra in ((X, Y), (Y, X)):
    for v in range(minimos[fija], maximos[fija] + 1):
        lo, hi = minimos.copy(), maximos.copy(); lo[fija] = hi[fija] = v
        a, b = int((R @ lo[R.columns]).max()), int((R @ hi[R.columns]).max())
        filas.append({'dato conocido': f'{fija} = {v}', 'actividad conocida': fija, 'cota inferior': a, 'cota superior': b, 'anchura': b - a})
intervalos = pd.DataFrame(filas).set_index('dato conocido')
display(intervalos.drop(columns='actividad conocida'))
amp = intervalos.groupby('actividad conocida')['anchura'].agg(['mean', 'max'])
mejor = X if amp.loc[X, 'mean'] < amp.loc[Y, 'mean'] else Y if amp.loc[Y, 'mean'] < amp.loc[X, 'mean'] else None
_txt = (f'Sin conocer ninguna, τ está entre {cota_inf} y {cota_sup} (anchura {cota_sup - cota_inf} semanas). '
        f'Conociendo {X}, la anchura media es {amp.loc[X, "mean"]:.1f} semanas (como mucho {int(amp.loc[X, "max"])}); '
        f'conociendo {Y}, {amp.loc[Y, "mean"]:.1f} semanas (como mucho {int(amp.loc[Y, "max"])}). ')
if mejor:
    otra = Y if mejor == X else X
    display(Markdown(_txt + f'**Respuesta:** conviene averiguar primero la duración de {mejor}: es la que más estrecha el intervalo de la duración del proyecto. '
             f'Fijado {mejor}, queda la incertidumbre de {otra} (rango de {maximos[otra] - minimos[otra]} semanas), que deja una anchura media de {amp.loc[mejor, "mean"]:.1f}; '
             f'fijado {otra}, la de {mejor} (rango de {maximos[mejor] - minimos[mejor]} semanas) deja {amp.loc[otra, "mean"]:.1f}. '
             'Una actividad sólo traslada a τ la parte de su rango en la que está en la ruta más larga: mientras tiene holgura, alargarla no alarga el proyecto.'))
else:
    display(Markdown(_txt + '**Respuesta:** los dos datos reducen la incertidumbre por igual.'))
