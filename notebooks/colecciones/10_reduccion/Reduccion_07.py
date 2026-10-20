# ---
# title: Reduccion 07
# subtitle: "8 actividades; CUR por tramos, reducir 3 semanas"
# coleccion: 10_reduccion
# variante: cur-por-tramos
# nivel: 3
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones normal y tope (semanas) se adjunta. Las actividades reducibles pueden acortarse hasta dos semanas, pero el coste no es lineal: la primera semana cuesta `ucr_1` k€ y la segunda `ucr_2` k€ (más cara: horas extra, turno de noche…). Las actividades con CUR infinito no pueden acelerarse.

# %%
pregunta_1 = "1. Calcule la duración normal del proyecto y sus rutas críticas.  "
pregunta_2 = "2. Reduzca el proyecto 3 semanas con mínimo coste aplicando Ackoff-Sasieni paso a paso con el *coste marginal*: el de la semana que toca acortar a cada actividad. Muestre en cada paso las rutas críticas, las opciones y su coste.  "
pregunta_3 = "3. ¿Qué habría resultado de aplicar el algoritmo con un CUR único por actividad igual a la media $(ucr_1 + ucr_2)/2$? Compare el coste real de esa solución con el del apartado anterior.  "
pregunta_4 = "4. ¿Por qué con costes crecientes por tramos el procedimiento semana a semana sigue siendo válido, y qué fallaría si la segunda semana fuera *más barata* que la primera?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, ucr_1 , ucr_2 
    ('A'     , '---'      , 7       , 5           , 5     , 9     ),
    ('B'     , '---'      , 6       , 4           , 7     , 11    ),
    ('C'     , 'A,B'      , 7       , 5           , 7     , 10    ),
    ('D'     , 'A,B'      , 6       , 6           , np.inf, np.inf),
    ('E'     , 'C,D'      , 6       , 4           , 5     , 10    ),
    ('F'     , 'E'        , 4       , 4           , np.inf, np.inf),
    ('G'     , 'E'        , 5       , 3           , 5     , 9     ),
    ('H'     , 'F'        , 5       , 3           , 8     , 12    ),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'ucr_1', 'ucr_2']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
reduccion = 3

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo PERT con las duraciones normales"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos paso a paso (avance con los botones):

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
duracion_normal = proyecto.duration(datos_proyecto['duration'])
rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])
for nombre, acts in rutas_criticas.items():
    print(f'{nombre}: {" - ".join(acts)}')
Markdown(f'Duración normal: {duracion_normal} semanas.')

# %%
#| tbl-cap: "Duración de cada ruta con las duraciones normales"
#| label: tbl-rutas
detalle_rutas(R, datos_proyecto['duration'][R.columns])

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# En cada paso el coste marginal de una actividad es `ucr_1` si aún no se ha acortado y `ucr_2` si ya se acortó una semana (infinito si no le queda margen). Las opciones son las actividades comunes a todas las rutas críticas y las parejas que cubren todas ellas; se elige la de menor coste marginal.

# %%
from itertools import combinations
def reducir_por_tramos(d0, tope, ucr_1, ucr_2, semanas):
    d = d0.astype(float).copy()
    filas = []
    for paso in range(1, semanas + 1):
        acortado = d0 - d
        marginal = pd.Series({a: (np.inf if d[a] <= tope[a] else (ucr_1[a] if acortado[a] == 0 else ucr_2[a])) for a in d.index})
        L = R @ d[R.columns]
        T = L.max(); criticas = [r for r in L.index if np.isclose(L[r], T)]
        conjuntos = [set(R.columns[R.loc[r] == 1]) for r in criticas]
        opciones = {}
        for a in set.intersection(*conjuntos):
            if marginal[a] < np.inf: opciones[a] = marginal[a]
        for p, q in combinations(sorted(set().union(*conjuntos)), 2):
            if marginal[p] < np.inf and marginal[q] < np.inf and all(p in s or q in s for s in conjuntos):
                opciones[f'{p}+{q}'] = marginal[p] + marginal[q]
        if not opciones:
            filas.append({'paso': paso, 'duración': int(T), 'rutas críticas': ', '.join(criticas), 'opciones (coste marginal)': 'ninguna', 'elegida': '—', 'coste': np.nan})
            break
        elegida = min(opciones, key=opciones.get)
        filas.append({'paso': paso, 'duración': int(T), 'rutas críticas': ', '.join(criticas),
                      'opciones (coste marginal)': '; '.join(f'{k} ({v:.0f})' for k, v in sorted(opciones.items(), key=lambda kv: kv[1])[:6]),
                      'elegida': elegida, 'coste': opciones[elegida]})
        for a in elegida.split('+'):
            d[a] -= 1
    return pd.DataFrame(filas).set_index('paso'), d
pasos_tramos, d_tramos = reducir_por_tramos(datos_proyecto['duration'], datos_proyecto['cap_duration'],
                                            datos_proyecto['ucr_1'], datos_proyecto['ucr_2'], reduccion)
pasos_tramos

# %%
coste_tramos = float(pasos_tramos['coste'].sum())
acortadas = (datos_proyecto['duration'] - d_tramos)
Markdown(f'Se acortan {", ".join(f"{a} ({int(x)})" for a, x in acortadas[acortadas > 0].items())}; duración {proyecto.duration(d_tramos.astype(int))} semanas; sobrecoste {coste_tramos:.0f} k€.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
ucr_medio = ((datos_proyecto['ucr_1'] + datos_proyecto['ucr_2']) / 2)
tabla_ackoff, pasos, d_medio, _ = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
                                                  costs=ucr_medio, reduction=reduccion, report=False)
acortadas_medio = (datos_proyecto['duration'] - d_medio.reindex(datos_proyecto.index)).astype(int)
def coste_real(acortadas):
    return float(sum((datos_proyecto.loc[a, 'ucr_1'] if x >= 1 else 0) + (datos_proyecto.loc[a, 'ucr_2'] if x >= 2 else 0) for a, x in acortadas.items()))
pd.DataFrame({'con coste marginal': [', '.join(f'{a} ({int(x)})' for a, x in acortadas[acortadas > 0].items()), coste_tramos],
              'con CUR medio': [', '.join(f'{a} ({int(x)})' for a, x in acortadas_medio[acortadas_medio > 0].items()), coste_real(acortadas_medio)]},
             index=['actividades acortadas', 'coste real (k€)'])

# %% [markdown]
# Con el CUR medio el algoritmo puede preferir acortar dos semanas una actividad que parece barata «de media» cuando en realidad su segunda semana es cara, o evitar una cuya primera semana era la más barata de todas. El coste real se obtiene valorando cada semana con su tramo.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Con costes marginales *crecientes* la función de coste de cada actividad es convexa, y en ese caso acortar semana a semana eligiendo siempre la opción marginal más barata es óptimo (es la misma razón por la que el CUR constante funciona). Si la segunda semana fuera más barata que la primera (coste cóncavo: un coste fijo de movilización y luego barato), el paso a paso fallaría: la primera semana parecería cara y nunca se emprendería, aunque acortar dos de golpe fuera lo mejor; habría que evaluar las dos semanas juntas.
