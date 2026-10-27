# ---
# title: Probabilidad 04
# subtitle: "9 actividades; segunda ruta a 1 semana de la crítica y mucho más variable"
# coleccion: 08_probabilidad
# variante: casi-critica-variable
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en semanas) se adjunta.

# %%
pregunta_1 = "1. Determine la duración media del proyecto, la ruta crítica y la ruta que le sigue en duración.  "
pregunta_2 = "2. Calcule la probabilidad de terminar en 22 semanas o menos según el método PERT (aproximación normal por la ruta crítica).  "
pregunta_3 = "3. Calcule la misma probabilidad considerando únicamente la segunda ruta. ¿Qué ruta da la probabilidad más baja? ¿Cuál es entonces una cota razonable de la probabilidad de cumplir el plazo?  "
pregunta_4 = "4. Compruebe con Monte Carlo (10 000 réplicas, duraciones normales) cuál es la probabilidad real y explique por qué el método PERT se equivoca aquí más de lo habitual.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 7       , 0.3               ),
    ('B'     , '---'      , 3       , 0.2               ),
    ('C'     , '---'      , 3       , 1.5               ),
    ('D'     , 'C'        , 5       , 1.8               ),
    ('E'     , 'A,B,C'    , 4       , 0.3               ),
    ('F'     , 'B'        , 7       , 0.2               ),
    ('G'     , 'D'        , 1       , 1.8               ),
    ('H'     , 'F,G'      , 5       , 0.2               ),
    ('I'     , 'E,H'      , 5       , 0.2               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la puesta en marcha de una línea de envasado; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Especificación de la línea',
    'B': 'Adjudicación de los equipos',
    'C': 'Obra civil y bancadas',
    'D': 'Recepción de la llenadora',
    'E': 'Recepción de la etiquetadora',
    'F': 'Montaje mecánico',
    'G': 'Instalación neumática',
    'H': 'Cableado y armarios de control',
    'I': 'Programación de autómatas',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
R = proyecto.path_matrix(dummies=False)
calendario = proyecto.calendar(datos_proyecto['duration'])

# %%
def tabla_rutas(R, medias, varianzas=None, calendario=None, dec=0, dec_var=3, nombre='media'):
    """Media (y varianza) de cada ruta con la suma término a término; actividades en orden de ejecución."""
    filas = {}
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario)
        m = float(sum(float(medias[a]) for a in acts))
        fila = {'actividades': ' – '.join(acts),
                f'{nombre} = suma': ' + '.join(f'{a}({float(medias[a]):.{dec}f})' for a in acts) + f' = {m:.{dec}f}',
                nombre: m}
        if varianzas is not None:
            v = float(sum(float(varianzas[a]) for a in acts))
            fila['varianza = suma'] = ' + '.join(f'{a}({float(varianzas[a]):.{dec_var}f})' for a in acts) + f' = {v:.{dec_var}f}'
            fila['varianza'] = v
            fila['σ = √varianza'] = v ** 0.5
        filas[r] = fila
    t = pd.DataFrame(filas).T
    for c in (nombre, 'varianza', 'σ = √varianza'):
        if c in t:
            t[c] = t[c].astype(float)
    return t

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con tiempos y camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Media y varianza de cada ruta (sumas de las de sus actividades), ordenadas de mayor a menor duración media:

# %%
#| tbl-cap: "Duración media y varianza de cada ruta"
#| label: tbl-rutas
rutas = tabla_rutas(R, datos_proyecto['duration'], datos_proyecto['variance'], calendario).sort_values('media', ascending=False)
limpia_tabla(rutas, decimales=3)

# %%
critica, segunda = rutas.index[0], rutas.index[1]
txt_c, txt_s = rutas.loc[critica, 'actividades'], rutas.loc[segunda, 'actividades']
mu_c, mu_s = float(rutas.loc[critica, 'media']), float(rutas.loc[segunda, 'media'])
s_c, s_s = float(rutas.loc[critica, 'σ = √varianza']), float(rutas.loc[segunda, 'σ = √varianza'])
Markdown(f'''**Respuesta:** la duración media del proyecto es {plural(mu_c, "semana")}; la ruta crítica es {txt_c} (σ = {s_c:.3f}) y la que le sigue es {txt_s}, con {plural(mu_s, "semana")} de media (sólo {plural(mu_c - mu_s, "semana")} menos) pero una desviación típica mucho mayor (σ = {s_s:.3f}).''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
plazo = 22
p_critica = norm.cdf(plazo, mu_c, s_c)
Markdown(f'''Por la ruta crítica {txt_c}: {explica_probabilidad(mu_c, s_c, plazo)}.\n\n'''
         f'''**Respuesta:** según el método PERT, P(T ≤ {plazo}) = {p_critica:.4f} ({p_critica:.1%}).''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
p_segunda = norm.cdf(plazo, mu_s, s_s)
cota = min(p_critica, p_segunda)
manda = ('la segunda ruta, y no la crítica, es la que da la probabilidad más baja: manda en probabilidad aunque no sea la más larga en media'
         if p_segunda < p_critica else 'la ruta crítica sigue dando la probabilidad más baja')
Markdown(f'''Por la segunda ruta {txt_s} (duración L₂): {explica_probabilidad(mu_s, s_s, plazo, nombre="L₂")}.\n\n'''
         f'''Para cumplir el plazo deben cumplirlo *todas* las rutas a la vez, así que la probabilidad real no puede superar la menor de las dos; '''
         f'''y como las rutas comparten actividades y no son independientes, tampoco es simplemente el producto.\n\n'''
         f'''**Respuesta:** {manda} ({p_segunda:.4f} frente a {p_critica:.4f}). Una cota razonable de la probabilidad de cumplir el plazo es '''
         f'''P(T ≤ {plazo}) ≤ {cota:.4f}.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# En cada réplica se sortea la duración de cada actividad con su normal (recortada en 0), se suman las duraciones a lo largo de cada ruta y la duración del proyecto es la de la ruta más larga en esa réplica. La probabilidad simulada es la proporción de réplicas que terminan dentro del plazo.

# %%
rng = np.random.default_rng(7)
N = 10000
muestras = pd.DataFrame({a: np.clip(rng.normal(datos_proyecto.loc[a, 'duration'], datos_proyecto.loc[a, 'standard_deviation'], N), 0, None) for a in R.columns})
duraciones_rutas_mc = muestras @ R.T
duracion_mc = duraciones_rutas_mc.max(axis='columns')
p_mc = float((duracion_mc <= plazo).mean())
cumplen = int((duracion_mc <= plazo).sum())
mas_larga = duraciones_rutas_mc.idxmax(axis='columns')
frac_segunda = float((mas_larga == segunda).mean())
frac_critica = float((mas_larga == critica).mean())
pd.Series({'PERT (ruta crítica)': p_critica, 'segunda ruta': p_segunda, 'Monte Carlo': p_mc}).round(4).to_frame(f'P(T ≤ {plazo})')

# %%
error_tipico = np.sqrt(cota * (1 - cota) / N)
if p_mc <= cota:
    frase_cota = f'queda por debajo de la cota del apartado 3, {cota:.4f}, como debe ser'
else:
    frase_cota = (f'supera en {p_mc - cota:.4f} la cota del apartado 3, {cota:.4f}: es error de muestreo, porque con {N} réplicas '
                  f'una proporción tiene un error típico de √(p(1 − p)/N) = √({cota:.4f} · {1 - cota:.4f} / {N}) = {error_tipico:.4f}')
culpa = ('es *ella* la que más veces hace fallar el plazo' if p_segunda < p_critica else
         'en muchas réplicas resulta la más larga y alarga el proyecto')
Markdown(f'''Monte Carlo: {cumplen} de las {N} réplicas terminan en {plazo} semanas o menos, luego P(T ≤ {plazo}) ≈ {cumplen}/{N} = **{p_mc:.4f}**, que {frase_cota}. La segunda ruta {txt_s} resulta la más larga en el {frac_segunda:.1%} de las réplicas y la crítica {txt_c} en el {frac_critica:.1%}.\n\n**Respuesta:** la probabilidad real es ≈ {p_mc:.4f}, frente al {p_critica:.4f} del método PERT, que la sobrestima en {p_critica - p_mc:.4f}. El método PERT sólo mira la ruta de mayor duración media ({txt_c}, σ = {s_c:.3f}); aquí otra ruta casi tan larga ({plural(mu_c - mu_s, "semana")} menos) es mucho más variable (σ = {s_s:.3f}) y {culpa}. La regla práctica es examinar todas las rutas casi críticas y quedarse con la probabilidad más baja, o directamente simular.''')
