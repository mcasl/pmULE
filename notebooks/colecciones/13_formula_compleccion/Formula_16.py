# ---
# title: Formula 16
# subtitle: "9 actividades; 2 rutas con margen ≤ 2"
# coleccion: 13_formula_compleccion
# variante: casi-criticas
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (días) se adjunta. El vector $R\,t$ contiene la duración de todas las rutas, no sólo de la crítica: la diferencia $\tau - (R\,t)_r$ es el *margen* de la ruta $r$ (cuánto puede alargarse sin cambiar τ).

# %%
pregunta_1 = "1. Calcule $R\\,t$, τ y el margen de cada ruta. Ordene las rutas de menor a mayor margen.  "
pregunta_2 = "2. Llame *casi críticas* a las rutas con margen ≤ 2. ¿Qué actividades pertenecen a alguna ruta casi crítica sin ser críticas? Obténgalas con $R^\\top$: sume las filas de $R$ correspondientes a las rutas casi críticas y mire qué columnas son positivas.  "
pregunta_3 = "3. ¿Cuál es el menor retraso de una sola actividad no crítica que cambiaría la ruta crítica? Compruébelo recalculando τ.  "
pregunta_4 = "4. Un segundo proyecto tiene las mismas prelaciones pero las actividades de la ruta crítica duran 2 días más. Calcule sus márgenes. ¿Cuál de los dos proyectos es más *sensible* (más rutas pueden convertirse en críticas con pequeños retrasos)? ¿Cuál tiene más varianza de τ si todas las duraciones son inciertas? Razónelo y, si quiere, compruébelo con una simulación breve.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 3       ),
    ('B'     , '---'      , 8       ),
    ('C'     , '---'      , 7       ),
    ('D'     , 'A,B,C'    , 6       ),
    ('E'     , 'A,B,C'    , 8       ),
    ('F'     , 'C'        , 3       ),
    ('G'     , 'E'        , 8       ),
    ('H'     , 'D,G'      , 2       ),
    ('I'     , 'F,H'      , 7       ),
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
    'I': 'Cuadro de protecciones y contador',
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
delta = 2

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Duración y margen de cada ruta"
#| label: tbl-margenes
Rt = R @ t
tau = float(Rt.max())
margen = (tau - Rt)
tabla = pd.DataFrame({'actividades': {r: ' - '.join(acts) for r, acts in proyecto.paths(dummies=False).items()}, 'R·t': Rt, 'margen = τ − R·t': margen}).sort_values('margen = τ − R·t')
tabla

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
casi = tabla.index[tabla['margen = τ − R·t'] <= delta].tolist()
criticas = tabla.index[tabla['margen = τ − R·t'] == 0].tolist()
veces = R.loc[casi].sum()               # columnas de R^T · indicador de rutas casi críticas
en_criticas = R.loc[criticas].sum() > 0
casi_sin_ser = veces.index[(veces > 0) & ~en_criticas].tolist()
display(pd.DataFrame({'en rutas casi críticas (nº)': veces, 'crítica': en_criticas}).T)
Markdown(f'Rutas casi críticas (margen ≤ {delta}): {", ".join(casi)}. Actividades en alguna de ellas sin ser críticas: {", ".join(casi_sin_ser) or "ninguna"}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Si la actividad $i$ se retrasa $x$ días, cada ruta que la contiene se alarga $x$; la primera no crítica en alcanzar τ es la de menor margen entre las que contienen $i$. El menor retraso que cambia la ruta crítica es, por tanto, el mínimo sobre las actividades no críticas del menor margen de sus rutas (más una unidad si se quiere que *supere* a la crítica).

# %%
no_criticas = [a for a in R.columns if not en_criticas[a]]
umbral = pd.Series({a: float(margen[R[a] == 1].min()) for a in no_criticas}).sort_values()
a_min = umbral.index[0]; x = int(umbral.iloc[0]) + 1
t_retrasada = t.copy(); t_retrasada[a_min] += x
Rt_r = R @ t_retrasada
display(umbral.to_frame('margen mínimo de sus rutas').T)
Markdown(f'Retrasando {a_min} {x} días: τ pasa de {tau:.0f} a {float(Rt_r.max()):.0f} y la ruta crítica es {Rt_r.idxmax()} (antes {Rt.idxmax()}).')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
t2 = t.copy()
for act in R.columns[R.loc[Rt.idxmax()] == 1]:
    t2[act] += 2
Rt2 = R @ t2
margen2 = Rt2.max() - Rt2
comparacion = pd.DataFrame({'margen (proyecto 1)': margen, 'margen (proyecto 2)': margen2}).sort_values('margen (proyecto 1)')
display(comparacion)
n_casi1, n_casi2 = int((margen <= delta).sum()), int((margen2 <= delta).sum())
Markdown(f'Rutas con margen ≤ {delta}: {n_casi1} en el proyecto 1 y {n_casi2} en el 2. El proyecto 1 es más sensible: más rutas compiten por ser la crítica.')

# %%
rng = np.random.default_rng(7)
N = 5000
def simula(tv):
    X = pd.DataFrame(rng.normal(tv.values, 0.25 * tv.values, size=(N, len(tv))), columns=tv.index).clip(lower=0.1)
    return (X @ R.T).max(axis='columns')
T1, T2 = simula(t), simula(t2)
pd.DataFrame({'proyecto 1': [tau, T1.mean(), T1.std(), T1.mean() - tau], 'proyecto 2': [float(Rt2.max()), T2.mean(), T2.std(), T2.mean() - float(Rt2.max())]},
             index=['τ determinista', 'media simulada', 'desv. típica simulada', 'sesgo de fusión (media − τ)']).round(2)

# %% [markdown]
# Con duraciones inciertas, el proyecto con más rutas casi críticas sufre más *sesgo de fusión*: la media simulada supera a τ porque en cada réplica gana la ruta que haya salido más larga entre varias candidatas. La desviación típica, en cambio, suele ser algo menor en el proyecto sensible (el máximo de varias variables se dispersa menos que una sola) y mayor en el proyecto 2, donde τ depende casi siempre de la misma ruta y hereda toda su varianza. La sensibilidad se lee directamente en el vector de márgenes $\tau\mathbf{1} - R\,t$: cuantos más ceros y valores pequeños tenga, más frágil es el camino crítico.
