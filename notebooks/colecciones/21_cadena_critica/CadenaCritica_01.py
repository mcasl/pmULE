# ---
# title: CadenaCritica 01
# subtitle: "4 tareas en turnos de 2 días; cambio de tarea de 1 día"
# coleccion: 21_cadena_critica
# variante: multitarea
# nivel: 1
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
# Un técnico de la oficina técnica tiene asignadas 4 tareas, cada una de un proyecto distinto, con las duraciones (días de trabajo efectivo) del cuadro. Los 4 jefes de proyecto le presionan para que «avance un poco en lo de todos».

# %%
pregunta_1 = "1. Si hace las tareas una detrás de otra en el orden del cuadro, sin interrupciones, ¿qué día termina cada una? ¿Cuál es el tiempo medio de terminación?  "
pregunta_2 = "2. Si reparte su tiempo en turnos de 2 días, rotando entre las tareas pendientes en el orden del cuadro (multitarea), y cambiar de tarea no le cuesta nada, ¿qué día termina cada una? ¿Cuánto se retrasa cada tarea y cuánto el tiempo medio respecto al apartado 1?  "
pregunta_3 = "3. Repita los dos cálculos suponiendo que cada vez que pasa de una tarea a otra pierde 1 día en retomarla (preparación, recuperar el hilo). ¿Cuándo termina el último trabajo en cada caso?  "
pregunta_4 = "4. ¿En qué orden habría que hacerlas, sin multitarea, para minimizar el tiempo medio de terminación? ¿Qué recomienda el método de la cadena crítica sobre la multitarea y por qué?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Tareas y duraciones (días)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  tarea, duration
    ('T1' , 7       ),
    ('T2' , 8       ),
    ('T3' , 7       ),
    ('T4' , 9       ),
], columns=['tarea', 'duration']).set_index('tarea')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
def multitarea(duraciones, bloque=None, cambio=0):
    """Turnos de una persona con varias tareas. Sin multitarea (bloque=None) las hace una detrás de
    otra en el orden dado; con multitarea trabaja `bloque` días en cada tarea pendiente, rotando en
    ese orden. Cada vez que pasa de una tarea a otra pierde `cambio` días.
    Devuelve (día en que termina cada tarea, tabla de turnos)."""
    restante = dict(duraciones)
    t, turnos, fin, anterior = 0, [], {}, None
    while restante:
        for tarea in list(duraciones):
            if tarea not in restante:
                continue
            if anterior is not None and tarea != anterior and cambio:
                turnos.append(('cambio', t, cambio))
                t += cambio
            trabajo = restante[tarea] if bloque is None else min(bloque, restante[tarea])
            turnos.append((tarea, t, trabajo))
            t += trabajo
            restante[tarea] -= trabajo
            anterior = tarea
            if restante[tarea] == 0:
                fin[tarea] = t
                del restante[tarea]
    return pd.Series(fin).reindex(list(duraciones)), pd.DataFrame(turnos, columns=['tarea', 'inicio', 'días'])


def dibuja_turnos(turnos, tareas, titulo, ax):
    """Diagrama de barras de los turnos (los cambios, en gris)."""
    colores = dict(zip(tareas, ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']))
    for i, tarea in enumerate(tareas):
        tramos = [(r.inicio, r.días) for r in turnos.itertuples() if r.tarea == tarea]
        ax.broken_barh(tramos, (i - 0.35, 0.7), color=colores[tarea])
    cambios = [(r.inicio, r.días) for r in turnos.itertuples() if r.tarea == 'cambio']
    if cambios:
        ax.broken_barh(cambios, (len(tareas) - 0.35, 0.7), color='lightgrey')
    ax.set_yticks(range(len(tareas) + (1 if cambios else 0)))
    ax.set_yticklabels(list(tareas) + (['cambios'] if cambios else []))
    ax.set_xlim(0, float((turnos['inicio'] + turnos['días']).max()) + 0.5)
    ax.invert_yaxis(); ax.set_title(titulo, fontsize=10); ax.grid(axis='x', alpha=0.3)


def sin_decimales(tabla):
    """Números enteros sin «.0» al mostrar una tabla."""
    return tabla.map(lambda v: f'{v:g}' if isinstance(v, float) else v)

# %%
duraciones = datos_enunciado['duration'].to_dict()
tareas = list(duraciones)
bloque, cambio = 2, 1

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Sin multitarea cada tarea termina cuando han terminado ella y todas las anteriores: los fines son las sumas acumuladas de las duraciones.

# %%
#| tbl-cap: "Sin multitarea: día de terminación de cada tarea"
#| label: tbl-secuencial
fin_seguidas, turnos_seguidas = multitarea(duraciones)
pd.DataFrame({'fin = suma acumulada': [' + '.join(str(duraciones[x]) for x in tareas[:i + 1]) + f' = {fin_seguidas[t]}' for i, t in enumerate(tareas)]}, index=tareas).T

# %%
Markdown(f'Tiempo medio de terminación sin multitarea: ({" + ".join(str(v) for v in fin_seguidas)})/{len(tareas)} = **{fin_seguidas.mean():.2f} días**; el último trabajo termina el día {fin_seguidas.max()}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Con multitarea se trabaja 2 días en cada tarea pendiente, por turno, en el orden del cuadro; una tarea sale de la rotación cuando termina. La tabla de turnos recoge cada tramo de trabajo.

# %%
#| tbl-cap: "Turnos con multitarea (sin pérdida al cambiar)"
#| label: tbl-turnos
fin_multi, turnos_multi = multitarea(duraciones, bloque)
turnos_multi.assign(fin=turnos_multi['inicio'] + turnos_multi['días']).T

# %%
#| tbl-cap: "Día de terminación con y sin multitarea"
#| label: tbl-comparacion
comparacion = pd.DataFrame({'sin multitarea': fin_seguidas, 'con multitarea': fin_multi})
comparacion['retraso'] = comparacion['con multitarea'] - comparacion['sin multitarea']
comparacion.loc['media'] = comparacion.mean()
sin_decimales(comparacion.astype(float).round(2))

# %%
retrasadas = [t for t in tareas if comparacion.loc[t, 'retraso'] > 0]
Markdown(f'El último trabajo termina igual (día {fin_multi.max()}: el trabajo total es el mismo), pero '
         f'{", ".join(retrasadas)} terminan más tarde: el tiempo medio pasa de {fin_seguidas.mean():.2f} a {fin_multi.mean():.2f} días '
         f'(+{fin_multi.mean() - fin_seguidas.mean():.2f}). Cada proyecto recibe su resultado más tarde y, si la tarea es crítica, '
         'el proyecto entero se retrasa.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Con pérdida en cada cambio de tarea"
#| label: tbl-cambio
fin_seguidas_c, turnos_seguidas_c = multitarea(duraciones, None, cambio)
fin_multi_c, turnos_multi_c = multitarea(duraciones, bloque, cambio)
n_cambios = pd.Series({'sin multitarea': int((turnos_seguidas_c['tarea'] == 'cambio').sum()),
                       'con multitarea': int((turnos_multi_c['tarea'] == 'cambio').sum())})
tabla_c = pd.DataFrame({'sin multitarea': fin_seguidas_c, 'con multitarea': fin_multi_c})
tabla_c.loc['media'] = tabla_c.mean()
tabla_c.loc['cambios de tarea'] = n_cambios
tabla_c.loc['días perdidos'] = n_cambios * cambio
sin_decimales(tabla_c.astype(float).round(2))

# %%
#| fig-cap: "Turnos de trabajo: sin multitarea y con multitarea (con pérdida en cada cambio)"
#| label: fig-turnos
from matplotlib import pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
dibuja_turnos(turnos_seguidas_c, tareas, 'Sin multitarea', axs[0])
dibuja_turnos(turnos_multi_c, tareas, f'Multitarea en turnos de {bloque} días', axs[1])
axs[1].set_xlabel('días'); plt.tight_layout(); plt.show()

# %%
Markdown(f'Con la pérdida por cambio, la multitarea ya no sólo retrasa las tareas intermedias: el último trabajo pasa del día '
         f'{fin_seguidas_c.max()} al {fin_multi_c.max()} ({n_cambios["con multitarea"]} cambios frente a {n_cambios["sin multitarea"]}), '
         'es decir, la persona produce menos. Es el efecto que Goldratt atribuye a la multitarea mal entendida.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
orden_spt = sorted(tareas, key=lambda t: (duraciones[t], t))
fin_spt, _ = multitarea({t: duraciones[t] for t in orden_spt})
Markdown(f'Ordenando de la más corta a la más larga ({" → ".join(orden_spt)}) el tiempo medio de terminación es '
         f'{fin_spt.mean():.2f} días, el mínimo posible sin multitarea (frente a {fin_seguidas.mean():.2f} en el orden del cuadro): '
         'es la regla de la duración más corta primero: si dos tareas seguidas están en orden larga–corta, intercambiarlas '
         'adelanta la corta lo que dura la larga y retrasa la larga sólo lo que dura la corta.')

# %% [markdown]
# El método de la cadena crítica pide **evitar la multitarea** en las actividades de la cadena: quien trabaja en una actividad crítica se dedica a ella hasta terminarla y la entrega en cuanto acaba, como el corredor de relevos. La multitarea alarga todas las actividades que se reparten (cada una termina cuando termina su último turno), retrasa la entrega a las sucesoras y, con la pérdida de cada cambio, reduce la capacidad real del recurso. Para decidir qué tarea va primero, la prioridad la marca el estado de los colchones de cada proyecto, no quien más presiona.
