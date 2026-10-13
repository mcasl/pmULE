# ---
# title: Comprension 02
# subtitle: "8 actividades; duraciones de D y C en intervalos; cotas de τ"
# coleccion: 14_comprension
# variante: datos-insuficientes
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
# τ = ‖R·t‖∞ es creciente en cada duración, luego la cota inferior se alcanza con todas las desconocidas en su mínimo y la superior con todas en su máximo: no hace falta probar combinaciones.

# %%
Rt_min = R @ minimos[R.columns]; Rt_max = R @ maximos[R.columns]
cota_inf, cota_sup = int(Rt_min.max()), int(Rt_max.max())
display(pd.DataFrame({'actividades': {r: ' - '.join(R.columns[R.loc[r] == 1]) for r in R.index}, 'duración mínima': Rt_min, 'duración máxima': Rt_max}))
Markdown(f'{cota_inf} ≤ τ ≤ {cota_sup}. La cota inferior se alcanza con {X} = {minimos[X]} y {Y} = {minimos[Y]} (ruta {Rt_min.idxmax()}); la superior con {X} = {maximos[X]} y {Y} = {maximos[Y]} (ruta {Rt_max.idxmax()}).')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Una actividad es crítica siempre si pertenece a *toda* ruta que pueda ser la más larga para alguna combinación de valores. Como sólo hay dos duraciones inciertas, basta probar las cuatro combinaciones extremas (τ es lineal a trozos y el máximo de R·t cambia de ruta sólo en esos vértices o entre ellos) y, para estar seguros, una rejilla de valores intermedios.

# %%
from itertools import product
siempre, nunca = set(R.columns), set(R.columns)
for dX, dY in product(range(minimos[X], maximos[X] + 1), range(minimos[Y], maximos[Y] + 1)):
    t = minimos.copy(); t[X], t[Y] = dX, dY
    Rt = R @ t[R.columns]
    criticas_ahora = set(R.columns[(R.loc[Rt == Rt.max()].sum() > 0)])
    siempre &= criticas_ahora; nunca -= criticas_ahora
Markdown(f'Críticas para cualquier valor: {", ".join(sorted(siempre)) or "ninguna"}. Nunca críticas: {", ".join(sorted(nunca)) or "ninguna"}. '
         f'Las demás ({", ".join(sorted(set(R.columns) - siempre - nunca))}) lo son o no según los valores.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
def holgura_Y(dX, dY):
    t = minimos.copy(); t[X], t[Y] = dX, dY
    return int(proyecto.calendar(t).loc[Y, 'H_total'])
tabla = pd.DataFrame({f'{X} = {dX}': {dY: holgura_Y(dX, dY) for dY in range(minimos[Y], maximos[Y] + 1)} for dX in (minimos[X], maximos[X])})
tabla.index.name = f'duración de {Y}'
display(tabla.T)
umbrales = {col: next((dY for dY, h in tabla[col].items() if h == 0), None) for col in tabla.columns}
Markdown(f'{Y} se vuelve crítica a partir de ' + '; '.join(f'{Y} = {u} cuando {col}' if u is not None else f'nunca cuando {col}' for col, u in umbrales.items()) + '. '
         f'Cuanto más dura {X} (crítica), más holgura tiene {Y} y más tarde se vuelve crítica.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
def amplitud(fija):
    # anchura del intervalo de τ cuando se conoce la actividad `fija` (promedio sobre sus valores posibles)
    anchuras = []
    for v in range(minimos[fija], maximos[fija] + 1):
        lo, hi = minimos.copy(), maximos.copy(); lo[fija] = hi[fija] = v
        anchuras.append(int((R @ hi[R.columns]).max() - (R @ lo[R.columns]).max()))
    return float(np.mean(anchuras))
amp = pd.Series({f'conociendo {X}': amplitud(X), f'conociendo {Y}': amplitud(Y), 'sin conocer ninguna': float(cota_sup - cota_inf)})
display(amp.to_frame('anchura del intervalo de τ').T)
Markdown(f'Conviene averiguar primero {X if amp.iloc[0] < amp.iloc[1] else Y}: es la que más estrecha el intervalo de la duración. '
         f'En general, la incertidumbre de una actividad crítica se traslada entera a τ; la de una actividad con holgura sólo cuenta cuando agota la holgura.')
