# ---
# title: Formula 03
# subtitle: "10 actividades, 6 rutas; SVD de R y ranking frente al conjunto crítico"
# coleccion: 13_formula_compleccion
# variante: svd-rango-1
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
# La matriz de rutas $R$ describe la estructura del proyecto sin usar duraciones. Su descomposición en valores singulares $R = U\,\Sigma\,V^T$ permite resumir esa estructura: $u_1$ pondera las rutas y $v_1$ las actividades, y el cociente $\sigma_1/\sigma_2$ indica cuánto domina esa primera dirección. Considere el proyecto cuyo cuadro se adjunta.

# %%
pregunta_1 = "1. Construya R y calcule su SVD. Dé los valores singulares y el cociente σ₁/σ₂.  "
pregunta_2 = "2. Ordene las actividades por |v₁| y las rutas por |u₁|. ¿Qué tienen en común las actividades mejor situadas?  "
pregunta_3 = "3. Con las duraciones, calcule τ exacto y la aproximación de rango 1, τ₁ = σ₁ · máx(u₁) · (v₁ᵀ t) (R ≈ σ₁ u₁ v₁ᵀ). ¿Qué error relativo comete?  "
pregunta_4 = "4. Compare el ranking por |v₁| con el conjunto de actividades críticas: ¿es crítica la actividad mejor clasificada? ¿Qué fracción de las k mejor clasificadas son críticas (k = número de críticas)? ¿Por qué la SVD, que no usa duraciones, puede acertar?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 5       ),
    ('B'     , '---'      , 6       ),
    ('C'     , 'A,B'      , 3       ),
    ('D'     , 'C'        , 7       ),
    ('E'     , 'A,B'      , 3       ),
    ('F'     , 'C'        , 8       ),
    ('G'     , 'D,E'      , 7       ),
    ('H'     , 'G'        , 3       ),
    ('I'     , 'F,H'      , 4       ),
    ('J'     , 'I'        , 7       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
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
    'J': 'Preparación de salas y audiovisuales',
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
#| tbl-cap: "Matriz de rutas R"
#| label: tbl-R
proyecto.display_rutas("pert")

# %%
U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)
u1, v1 = U[:, 0], Vt[0, :]
if u1.sum() < 0:
    u1, v1 = -u1, -v1          # R es no negativa: u1 y v1 pueden tomarse no negativos
display(Math(r'\Sigma = ' + bmatrix(np.diag(S), dec=2)))
Markdown(f'**Respuesta:** valores singulares {", ".join(f"{x:.3f}" for x in S)}; σ₁/σ₂ = {S[0]:.3f}/{S[1]:.3f} = **{S[0]/S[1]:.2f}**.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
peso_actividades = pd.Series(v1, index=R.columns).sort_values(ascending=False)
peso_rutas = pd.Series(u1, index=R.index).sort_values(ascending=False)
apariciones = R.sum()
display(pd.DataFrame({'v₁': peso_actividades.round(3), 'rutas en las que aparece': apariciones[peso_actividades.index]}).T)
pd.DataFrame({'u₁': peso_rutas.round(3), 'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in peso_rutas.index], 'nº actividades': R.loc[peso_rutas.index].sum(axis=1)})

# %%
en_todas = [a for a in R.columns if apariciones[a] == R.shape[0]]
top = list(peso_actividades.index[:3])
t_todas = (f'{lista_y(en_todas)} {"aparece" if len(en_todas) == 1 else "aparecen"} en las {R.shape[0]} rutas y '
           f'{"encabeza" if len(en_todas) == 1 else "encabezan"} la lista. ' if en_todas else 'Ninguna actividad está en todas las rutas. ')
Markdown(f'''**Respuesta:** las mejor situadas son {lista_y([f"{a} (v₁ = {peso_actividades[a]:.3f}, en {int(apariciones[a])} rutas)" for a in top])}: lo que tienen en común es que aparecen en muchas rutas, sobre todo en las rutas largas y con muchas actividades compartidas (las de mayor u₁: {peso_rutas.index[0]}, con {int(R.loc[peso_rutas.index[0]].sum())} actividades). {t_todas}v₁ es una medida de centralidad estructural: no depende de las duraciones.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# τ exacto: cada fila de $R$ por $t$ es la duración de una ruta, y τ es la mayor.

# %%
producto = producto_Rt(proyecto, R, t, datos_proyecto['duration'])
display(producto)
tau = float((R @ t).max())
tau_1 = float(S[0] * u1.max() * (v1 @ t.values))
v1t = ' + '.join(f'{x:.3f}·{int(d)}' for x, d in zip(v1, t.values))
Markdown(f'''**Respuesta:** {explica_tau(producto)}. Aproximación de rango 1: v₁ᵀt = {v1t} = {v1 @ t.values:.3f}, y τ₁ = σ₁ · máx(u₁) · (v₁ᵀ t) = {S[0]:.3f} × {u1.max():.3f} × {v1 @ t.values:.3f} = **{tau_1:.2f}**. Error relativo = |τ₁ − τ|/τ = |{tau_1:.2f} − {tau:.0f}|/{tau:.0f} = **{abs(tau_1 - tau) / tau * 100:.1f} %**.''')

# %% [markdown]
# La aproximación sustituye cada fila de $R$ por un múltiplo de $v_1$: funciona bien cuando las rutas se parecen entre sí (σ₁ ≫ σ₂) y peor cuando el proyecto tiene varias ramas muy distintas. El τ exacto es siempre la norma infinito de $R\,t$; la SVD es un resumen.

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
criticas = set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)
k = len(criticas)
top_k = list(peso_actividades.index[:k])
precision = len(set(top_k) & criticas) / k
pd.DataFrame({'ranking por v₁': peso_actividades.index, 'v₁': peso_actividades.round(3).values,
              'crítica': ['sí' if a in criticas else 'no' for a in peso_actividades.index]}).set_index('ranking por v₁').T

# %%
corte = peso_actividades.iloc[k - 1]
empate_corte = k < len(peso_actividades) and np.isclose(peso_actividades.iloc[k], corte)
nota = (f' Ojo: en el puesto {k} hay un empate de v₁ ({corte:.3f}) con actividades que quedan fuera, así que la precisión depende de cómo se ordenen los empatados.'
        if empate_corte else '')
Markdown(f'''**Respuesta:** la actividad mejor clasificada, {peso_actividades.index[0]}, {"es" if peso_actividades.index[0] in criticas else "no es"} crítica. Hay k = {k} actividades críticas ({lista_y(ordena_ruta(sorted(criticas), proyecto.calendar(datos_proyecto['duration'])))}); entre las {k} primeras del ranking ({lista_y(top_k)}) hay {len(set(top_k) & criticas)} críticas: precisión {len(set(top_k) & criticas)}/{k} = **{precision:.2f}**.{nota} La SVD sólo ve la estructura: una actividad por la que pasan muchas rutas tiene muchas ocasiones de estar en la más larga, sea cual sea el vector de duraciones; por eso acierta a menudo, pero no siempre, porque la ruta crítica concreta la fijan las duraciones.''')
