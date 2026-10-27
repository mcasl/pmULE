# ---
# title: Caminos 07
# subtitle: "8 actividades, 5 rutas, reconstrucción desde las rutas y SVD"
# coleccion: 06_matriz_caminos
# variante: rutas-a-grafo-svd
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
# De un proyecto sólo se conocen sus rutas entre el nodo inicial y el final y las duraciones de sus actividades (en días).

# %%
pregunta_1 = "1. Construya la matriz de caminos del proyecto.  "
pregunta_2 = "2. Calcule con ella la duración del proyecto e identifique la ruta crítica.  "
pregunta_3 = "3. Reconstruya el cuadro de prelaciones inmediatas y dibuje los grafos PERT y Roy. Compruebe que sus rutas son las del enunciado.  "
pregunta_4 = "4. Calcule la sensibilidad de la duración del proyecto a cada actividad: ¿en cuánto aumenta la duración si una actividad dura un día más?  "
pregunta_5 = "5. Calcule la descomposición en valores singulares de la matriz de caminos y ordene rutas y actividades según el primer vector singular. Compare con la ruta crítica.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Rutas del proyecto"
#| label: tbl-datosRutas
rutas_enunciado = {
    'Route_1': ['A', 'C', 'E', 'H'],
    'Route_2': ['A', 'D', 'H'],
    'Route_3': ['B', 'C', 'E', 'H'],
    'Route_4': ['B', 'F', 'G'],
    'Route_5': ['B', 'F', 'H'],
}
pd.Series({r: ' - '.join(a) for r, a in rutas_enunciado.items()}).to_frame('actividades')

# %%
#| tbl-cap: "Duración de las actividades"
#| label: tbl-datosDuraciones
datos_enunciado = pd.DataFrame([
#  activity, duration
    ('A'     , 8       ),
    ('B'     , 2       ),
    ('C'     , 4       ),
    ('D'     , 7       ),
    ('E'     , 1       ),
    ('F'     , 4       ),
    ('G'     , 2       ),
    ('H'     , 4       ),
], columns=['activity', 'duration']).set_index('activity')
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

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# La matriz de caminos tiene un 1 en la fila de una ruta y la columna de una actividad cuando la actividad pertenece a la ruta.

# %%
actividades = sorted(set(a for acts in rutas_enunciado.values() for a in acts))
path_matrix = pd.DataFrame(0, index=list(rutas_enunciado), columns=actividades)
for ruta, acts in rutas_enunciado.items():
    path_matrix.loc[ruta, acts] = 1
from pmule import matriz_caminos_tikz
matriz_caminos_tikz(path_matrix)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duraciones = datos_enunciado['duration'].reindex(actividades)
duraciones_rutas = path_matrix @ duraciones
from pmule import display_producto_rutas
display_producto_rutas(path_matrix, duraciones, duraciones_rutas)

# %% [markdown]
# Cada fila del producto es la suma de las duraciones de las actividades de esa ruta:

# %%
#| tbl-cap: "Duración de cada ruta, sumando sus actividades"
#| label: tbl-rutasPaso-a
pd.DataFrame({r: {'ruta': ' – '.join(acts),
                  'duración (días) = suma': ' + '.join(str(int(duraciones[a])) for a in acts) + f' = {int(duraciones_rutas[r])}'}
              for r, acts in rutas_enunciado.items()}).T

# %%
duracion_proyecto = int(duraciones_rutas.max())
criticas = [r for r in duraciones_rutas.index if duraciones_rutas[r] == duracion_proyecto]
ruta_critica = criticas[0]
Markdown(f'La duración del proyecto es la mayor de las duraciones de las rutas: T = máx('
         f'{", ".join(str(int(x)) for x in duraciones_rutas)}) = **{duracion_proyecto} días**.\n\n'
         f'**Respuesta:** el proyecto dura {duracion_proyecto} días; '
         + ('; '.join(f'{r}: {" – ".join(rutas_enunciado[r])}' for r in criticas))
         + (' es la ruta crítica.' if len(criticas) == 1 else ' son rutas críticas.'))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Las predecesoras inmediatas de una actividad son las que la preceden inmediatamente en alguna ruta; las rutas son todas las secuencias posibles, así que el conjunto de parejas consecutivas de todas las rutas es exactamente el cuadro de prelaciones inmediatas. Las actividades que abren alguna ruta son iniciales.

# %%
prelaciones_dict = {a: set() for a in actividades}
for acts in rutas_enunciado.values():
    for anterior, posterior in zip(acts[:-1], acts[1:]):
        prelaciones_dict[posterior].add(anterior)
datos_proyecto = pd.DataFrame({'predecessor': pd.Series({a: ','.join(sorted(p)) or '---' for a, p in prelaciones_dict.items()}),
                               'duration': duraciones})
datos_proyecto.index.name = 'activity'
datos_proyecto

# %%
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT reconstruido, con tiempos y ruta crítica"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy reconstruido"
#| label: fig-roy
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
rutas_grafo = {' - '.join(a) for a in proyecto.paths(dummies=False).values()}
rutas_dadas = {' - '.join(a) for a in rutas_enunciado.values()}
Markdown('**Respuesta:** el cuadro de prelaciones reconstruido es el de la tabla anterior, y las rutas de los grafos '
         + ('coinciden con las del enunciado.' if rutas_grafo == rutas_dadas else 'NO coinciden con las del enunciado.'))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La duración del proyecto es $\tau = \max_r (P\,d)_r$. Si una actividad crítica dura un día más, la ruta crítica se alarga un día y con ella el proyecto: su sensibilidad es 1. Si una actividad no crítica dura un día más, la ruta más larga no cambia mientras no se agote la holgura: sensibilidad 0. Al acortar una actividad crítica el proyecto sólo se acorta hasta que otra ruta pasa a ser la más larga.

# %% [markdown]
# Para cada actividad basta mirar la ruta más larga que la contiene: su *margen* hasta T (que es la holgura total de la actividad) dice si un día más alarga el proyecto, y la ruta más larga que *no* la contiene dice si un día menos lo acorta.

# %%
T = duracion_proyecto
L = duraciones_rutas.astype(int)
sensibilidad = proyecto.tau_sensitivity(duraciones.to_dict())
filas = {}
for act in duraciones.index:
    con = [r for r in L.index if path_matrix.loc[r, act] == 1]
    sin = [r for r in L.index if path_matrix.loc[r, act] == 0]
    r_con = max(con, key=lambda r: L[r])
    L_sin = int(L[sin].max()) if sin else 0
    margen = T - int(L[r_con])
    mas = 1 if margen == 0 else 0
    menos = (max(T - 1, L_sin) - T) if margen == 0 else 0
    filas[act] = {'ruta más larga con ella': f'{r_con} ({int(L[r_con])})',
                  'margen = T − esa ruta': f'{T} − {int(L[r_con])} = {margen}',
                  'ruta más larga sin ella': L_sin if sin else '—',
                  '+1 día': f'{mas:+d}' if mas else '0', '−1 día': f'{menos:+d}' if menos else '0',
                  'tau_sensitivity': int(round(float(sensibilidad[act])))}
sens = pd.DataFrame(filas).T
sens

# %%
crit = [a for a in sens.index if sens.loc[a, '+1 día'] == '+1']
resto = [a for a in sens.index if a not in crit]
baja = [a for a in crit if sens.loc[a, '−1 día'] == '-1']
_resto = lista_y([f"{a} (margen {sens.loc[a, 'margen = T − esa ruta'].split(' = ')[-1]})" for a in resto])
Markdown(f'La columna de la librería (`tau_sensitivity`) coincide con la de +1 día.\n\n'
         f'**Respuesta:** si una actividad dura un día más, el proyecto pasa de {T} a {T + 1} días cuando es crítica '
         f'({lista_y(crit)}: margen 0) y no cambia en las demás' + (f', {_resto}, porque su margen absorbe el día.' if resto else '.')
         + f' Si dura un día menos, el proyecto se acorta un día con '
         + (f'{lista_y(baja)}, porque la ruta más larga que no las contiene dura menos de {T}' if baja else 'ninguna')
         + (f'; con {lista_y([a for a in crit if a not in baja])} no cambia, porque otra ruta de {T} días no la contiene' if len(baja) < len(crit) else '')
         + '; acortar una actividad no crítica no cambia nada.')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# La descomposición en valores singulares $P = U\,S\,V^T$ ordena la información de la matriz de caminos por importancia. El primer vector singular por la izquierda, $u_1$, pondera las rutas, y el primero por la derecha, $v_1$, las actividades: las actividades que aparecen en muchas rutas y las rutas que contienen muchas actividades compartidas reciben más peso. Es una ordenación estructural, que no usa las duraciones, y por eso no tiene por qué coincidir con la ruta crítica.

# %%
U, S, Vt = np.linalg.svd(path_matrix.values.astype(float))
u1 = np.abs(U[:, 0]); v1 = np.abs(Vt[0, :])
importancia_rutas = pd.Series(u1 / u1.sum(), index=path_matrix.index)
importancia_actividades = pd.Series(v1 / v1.sum(), index=path_matrix.columns)
Math(r'S = ' + bmatrix(S.round(2), dec=2))

# %%
#| tbl-cap: "Rutas y actividades ordenadas por su peso en el primer vector singular"
#| label: tbl-svd
ordena_rutas(path_matrix.copy(), importancia_actividades.values, importancia_rutas.values)

# %% [markdown]
# Cómo sale cada importancia: como $P\,v_1 = s_1\,u_1$, el peso de una ruta es la suma de los pesos $v_1$ de sus actividades dividida por $s_1$; la columna «Importancia» de la tabla anterior es ese peso normalizado para que las rutas sumen 1 ($u_1/\sum u_1$), y cada celda de actividad es $v_1/\sum v_1$ si la actividad está en la ruta.

# %%
_v = pd.Series(v1, index=path_matrix.columns)
_u = pd.Series(u1, index=path_matrix.index)
display(Markdown(f'$s_1$ = {S[0]:.3f}; pesos $v_1$ de las actividades: '
                 + ', '.join(f'{a} {x:.3f}' for a, x in _v.sort_values(ascending=False).items()) + '.'))
_tabla = pd.DataFrame({r: {'ruta': ' – '.join(rutas_enunciado[r]),
                           'u₁ = (suma de v₁)/s₁': '(' + ' + '.join(f'{_v[a]:.3f}' for a in rutas_enunciado[r]) + f')/{S[0]:.3f} = {_u[r]:.3f}',
                           'importancia = u₁/Σu₁': round(float(importancia_rutas[r]), 3),
                           'duración': int(duraciones_rutas[r])} for r in path_matrix.index}).T
_tabla.sort_values('importancia = u₁/Σu₁', ascending=False)

# %%
_dom = [r for r in importancia_rutas.index if np.isclose(importancia_rutas[r], importancia_rutas.max())]
_act = [a for a in importancia_actividades.index if np.isclose(importancia_actividades[a], importancia_actividades.max())]
_veces = int(path_matrix[_act[0]].sum())
Markdown('**Respuesta:** ' + (f'la ruta dominante según la SVD es {_dom[0]}' if len(_dom) == 1 else
         f'empatan como dominantes según la SVD las rutas {lista_y(_dom)}')
         + f' (importancia {importancia_rutas.max():.3f}); la ruta crítica, {ruta_critica} ({duracion_proyecto} días), '
         + ('es una de ellas' if ruta_critica in _dom and len(_dom) > 1 else ('coincide con ella' if ruta_critica in _dom else 'no coincide'))
         + '. ' + (f'La actividad de mayor peso es {_act[0]}' if len(_act) == 1 else f'Las actividades de mayor peso son {lista_y(_act)}')
         + f', que aparece{"" if len(_act) == 1 else "n"} en {_veces} de las {len(path_matrix)} rutas. La SVD sólo mira la '
         'estructura (en cuántas rutas está cada actividad); la ruta crítica depende además de las duraciones.')
