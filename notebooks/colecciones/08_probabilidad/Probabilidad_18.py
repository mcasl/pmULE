# ---
# title: Probabilidad 18
# subtitle: "10 actividades, 2 rutas críticas, σ = 1.04"
# coleccion: 08_probabilidad
# variante: dos-rutas-criticas
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en semanas) se adjunta.

# %%
pregunta_1 = "1. Determine la duración media del proyecto y su(s) camino(s) crítico(s).  "
pregunta_2 = "2. Calcule la varianza y la desviación típica de la duración del proyecto. ¿Qué ruta crítica debe usarse para ello? Razone la respuesta.  "
pregunta_3 = "3. ¿Qué probabilidad hay de terminar el proyecto en 32 semanas o menos?  "
pregunta_4 = "4. ¿Qué probabilidad hay de terminar entre las semanas 30 y 33?  "
pregunta_5 = "5. ¿Qué duración puede comprometerse con el cliente con una probabilidad del 90% de cumplirla?  "
pregunta_6 = "6. ¿Qué probabilidad hay de terminar en la duración media o antes? ¿Puede garantizarse esa duración con un 90% de probabilidad sin modificar el proyecto?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 1       , 0.3               ),
    ('B'     , '---'      , 2       , 0.3               ),
    ('C'     , '---'      , 2       , 0.5               ),
    ('D'     , 'B,C'      , 4       , 0.3               ),
    ('E'     , 'D'        , 8       , 0.8               ),
    ('F'     , 'A,E'      , 8       , 0.3               ),
    ('G'     , 'F'        , 6       , 0.4               ),
    ('H'     , 'F'        , 7       , 0.4               ),
    ('I'     , 'F'        , 8       , 0.1               ),
    ('J'     , 'H,I'      , 1       , 0.1               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con tiempos de los nodos y camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Holgura total de cada actividad (sin las ficticias), leída del grafo resuelto:

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
holguras = resultados_pert['activities'].T
holguras = holguras[[c for c in holguras.columns if not str(c).startswith('@')]]   # sin las ficticias @i→j
holguras.loc[['H_total']].rename(index={'H_total': 'holgura total'})

# %%
duracion_media = proyecto.duration(datos_proyecto['duration'])
rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])
calendario = proyecto.calendar(datos_proyecto['duration'])
actividades_criticas = ordena_ruta(holguras.columns[holguras.loc['H_total'] == 0], calendario)
rutas_criticas_txt = [' – '.join(ordena_ruta(acts, calendario)) for acts in rutas_criticas.values()]
display(Markdown(f'Las actividades con holgura total nula son {lista_y(actividades_criticas)}; el resto puede retrasarse '
                 'hasta su holgura total sin alargar el proyecto.'))
display(explica_camino_critico(proyecto, datos_proyecto['duration']))
Markdown(f'**Respuesta:** la duración media del proyecto es {plural(duracion_media, "semana")}; '
         + (f'el camino crítico es {rutas_criticas_txt[0]}.' if len(rutas_criticas_txt) == 1 else
            f'hay {len(rutas_criticas_txt)} caminos críticos: {lista_y(rutas_criticas_txt)}.'))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Por el teorema central del límite, la duración de una ruta es aproximadamente normal con media la suma de las medias y varianza la suma de las varianzas de sus actividades ($\sigma_i^2$ = desviación típica al cuadrado). Para el proyecto se toma la ruta crítica; si hay varias, la de mayor varianza, que es la hipótesis más prudente (la verdadera distribución del máximo de varias rutas no es normal, y el método PERT la aproxima así).

# %% [markdown]
# Varianza de cada ruta crítica, suma de las varianzas de sus actividades (resaltada la mayor):

# %%
#| tbl-cap: "Varianza de las rutas críticas"
#| label: tbl-varianzaRutas
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                                   variances=datos_proyecto['variance'])
varianza_rutas.relabel_index(['actividades', 'varianza', 'suma de varianzas'], axis='columns')   # Styler: resalta la mayor varianza

# %%
var_criticas = {r: float(datos_proyecto.loc[list(acts), 'variance'].sum()) for r, acts in rutas_criticas.items()}
ruta_var = max(var_criticas, key=var_criticas.get)
acts_var = ordena_ruta(rutas_criticas[ruta_var], calendario)
txt_var = ' – '.join(acts_var)
varianza_proyecto = var_criticas[ruta_var]
if len(var_criticas) == 1:
    eleccion = f'Sólo hay una ruta crítica, {txt_var}; su varianza es la del proyecto.'
else:
    otras = [r for r in var_criticas if r != ruta_var]
    comparacion_txt = lista_y([f"{var_criticas[r]:.3f} de {' – '.join(ordena_ruta(rutas_criticas[r], calendario))}" for r in otras])
    if all(np.isclose(var_criticas[r], varianza_proyecto) for r in otras):
        eleccion = (f'Las rutas críticas tienen la misma varianza ({varianza_proyecto:.3f}), así que da igual cuál se use: '
                    f'por ejemplo {txt_var}.')
    else:
        eleccion = (f'Hay {len(var_criticas)} rutas críticas con la misma media ({duracion_media} semanas) pero distinta varianza: '
                    f'{txt_var} tiene varianza {varianza_proyecto:.3f}, frente a {comparacion_txt}. Se usa la de mayor varianza, {txt_var}, '
                    'porque es la hipótesis más prudente: da la normal más dispersa y, para plazos por encima de la media, '
                    'la menor probabilidad de cumplirlos.')
Markdown(f'''{eleccion}\n\n{explica_sigma_ruta(acts_var, datos_proyecto["variance"], nombre="σ_T")} semanas.\n\n'''
         f'''**Respuesta:** varianza del proyecto σ²_T = {varianza_proyecto:.3f} y desviación típica σ_T = {desviacion_proyecto:.3f} semanas'''
         + (f''' (ruta {txt_var}).''' if len(var_criticas) > 1 else '.'))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# La duración del proyecto se modela como $T \sim N(\mu, \sigma_T)$, con la media y la desviación típica de los apartados 1 y 2. La probabilidad se obtiene directamente con la función de distribución normal de la hoja de cálculo (o de la calculadora), sin tipificar.

# %%
plazo = 32
probabilidad_plazo = norm.cdf(plazo, duracion_media, desviacion_proyecto)
Markdown(f'''{explica_probabilidad(duracion_media, desviacion_proyecto, plazo)}.\n\n'''
         f'''**Respuesta:** la probabilidad de terminar en {plazo} semanas o menos es {probabilidad_plazo:.4f} ({probabilidad_plazo:.1%}).''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Terminar entre las dos semanas es terminar después de la primera y no más tarde de la segunda: la probabilidad es la diferencia de las dos funciones de distribución.

# %%
t_a, t_b = 30, 33
p_b = norm.cdf(t_b, duracion_media, desviacion_proyecto)
p_a = norm.cdf(t_a, duracion_media, desviacion_proyecto)
Markdown(f'''{explica_intervalo(duracion_media, desviacion_proyecto, t_a, t_b)}.\n\n'''
         f'''**Respuesta:** la probabilidad de terminar entre las semanas {t_a} y {t_b} es {p_b - p_a:.4f} ({p_b - p_a:.1%}).''')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# Se busca el plazo $t$ que se cumple con la probabilidad pedida, $P(T \le t) = p$: es el percentil $p$ de la normal, que da la inversa de la función de distribución.

# %%
probabilidad_objetivo = 0.9
duracion_comprometida = norm.ppf(probabilidad_objetivo, duracion_media, desviacion_proyecto)
semanas_enteras = int(np.ceil(duracion_comprometida))
Markdown(f'''{explica_percentil(duracion_media, desviacion_proyecto, probabilidad_objetivo, 'semanas')}.\n\n'''
         f'''Como el calendario se mide en semanas enteras, se redondea hacia arriba a {semanas_enteras} semanas: '''
         f'''{explica_probabilidad(duracion_media, desviacion_proyecto, semanas_enteras)} ≥ {probabilidad_objetivo:g}.\n\n'''
         f'''**Respuesta:** puede comprometerse una duración de {plural(semanas_enteras, "semana")} '''
         f'''({duracion_comprometida:.2f} semanas sin redondear).''')

# %% [markdown]
# ## Apartado 6

# %%
display(Markdown(pregunta_6))

# %%
p_media = norm.cdf(duracion_media, duracion_media, desviacion_proyecto)
margen = duracion_comprometida - duracion_media
media_necesaria = duracion_media - margen
Markdown(f'''{explica_probabilidad(duracion_media, desviacion_proyecto, duracion_media)}: la media de una normal es también su mediana, así que sólo la mitad de las veces se terminaría en la duración media o antes. Esto no depende de σ_T: con la media en {duracion_media} semanas, P(T ≤ {duracion_media}) = 0.5 por pequeña que sea la varianza, así que reducir la varianza no basta.\n\nPara que {duracion_media} semanas se cumplan con un {probabilidad_objetivo:.0%} tienen que ser el percentil {probabilidad_objetivo * 100:.0f} de la duración. Según el apartado anterior, ese percentil queda {duracion_comprometida:.2f} − {duracion_media} = {margen:.2f} semanas por encima de la media; con la misma σ_T, la media debería bajar a {duracion_media} − {margen:.2f} = {media_necesaria:.2f} semanas.\n\n**Respuesta:** P(T ≤ {duracion_media}) = {p_media:.1f}. No puede garantizarse esa duración con un {probabilidad_objetivo:.0%} sin modificar el proyecto: habría que acortar la duración media al menos {margen:.2f} semanas (hasta {media_necesaria:.2f}, acortando actividades críticas) manteniendo la varianza.''')
