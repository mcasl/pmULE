# ---
# title: Avanzado 03
# subtitle: "9 actividades; indirectos 8, plazo 28, penalización 18, prima 2"
# coleccion: 15_avanzado
# variante: decision-economica
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas), coste normal (k€) y coste unitario de reducción (k€/semana) se adjunta. Además:
#
# - los costes indirectos (oficina, alquileres, supervisión) son de 8 k€ por semana de duración;
# - el contrato fija la entrega en la semana 28, con una penalización de 18 k€ por cada semana de retraso;
# - y una prima de 2 k€ por cada semana de adelanto sobre ese plazo.

# %%
pregunta_1 = "1. Calcule la duración normal del proyecto y su coste total con esa duración (directo + indirecto + penalización o prima).  "
pregunta_2 = "2. Obtenga con Ackoff-Sasieni el sobrecoste directo acumulado de cada semana de reducción hasta la duración tope.  "
pregunta_3 = "3. Construya la tabla de coste total frente a duración, con las cuatro componentes, y determine la duración óptima. Dibuje las curvas.  "
pregunta_4 = "4. Analice la decisión paso a paso: ¿qué se compara en cada semana de reducción antes y después del plazo contractual? ¿A partir de qué prima por semana de adelanto convendría reducir una semana más que en el óptimo?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, cost_normal, ucr   
    ('A'     , '---'      , 6       , 5           , 70         , 5     ),
    ('B'     , '---'      , 4       , 4           , 40         , np.inf),
    ('C'     , '---'      , 5       , 2           , 70         , 8     ),
    ('D'     , 'A,B,C'    , 8       , 6           , 50         , 3     ),
    ('E'     , 'D'        , 4       , 3           , 20         , 8     ),
    ('F'     , 'E'        , 4       , 2           , 90         , 8     ),
    ('G'     , 'E'        , 5       , 2           , 30         , 8     ),
    ('H'     , 'G'        , 6       , 6           , 70         , np.inf),
    ('I'     , 'F,G'      , 7       , 6           , 50         , 6     ),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'cost_normal', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
coste_indirecto_semanal, plazo_contrato, penalizacion_semanal, prima_semanal = 8, 28, 18, 2

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones normales y el camino crítico"
#| label: fig-grafo
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
duracion_normal = proyecto.duration(datos_proyecto['duration'])
coste_directo_normal = float(datos_proyecto['cost_normal'].sum())
def coste_plazo(T):
    return penalizacion_semanal * max(0, T - plazo_contrato) - prima_semanal * max(0, plazo_contrato - T)
def txt_plazo(T):
    if T > plazo_contrato:
        return f'penalización {penalizacion_semanal}·({T} − {plazo_contrato}) = {penalizacion_semanal * (T - plazo_contrato)}'
    if T < plazo_contrato:
        return f'prima −{prima_semanal}·({plazo_contrato} − {T}) = −{prima_semanal * (plazo_contrato - T)}'
    return 'ni penalización ni prima (entrega en plazo) = 0'
_sg = lambda v: f'+ {v:.0f}' if v >= 0 else f'− {-v:.0f}'
total_normal = coste_directo_normal + coste_indirecto_semanal * duracion_normal + coste_plazo(duracion_normal)
display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'semanas'))
display(Markdown('Coste directo normal: ' + explica_suma(datos_proyecto['cost_normal'], 'directo', 'k€')))
Markdown(f'''**Respuesta:** duración normal {duracion_normal} semanas. Coste total = directo {coste_directo_normal:.0f} + indirecto {coste_indirecto_semanal}·{duracion_normal} = {coste_indirecto_semanal * duracion_normal} + {txt_plazo(duracion_normal)} → {coste_directo_normal:.0f} + {coste_indirecto_semanal * duracion_normal} {_sg(coste_plazo(duracion_normal))} = **{total_normal:.0f} k€**.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Ackoff-Sasieni reduce el proyecto semana a semana: en cada paso hay que acortar una semana todas las rutas críticas, y se elige la actividad (o la combinación de actividades) con margen de reducción que lo consigue con el menor coste unitario. La primera tabla da la duración de cada ruta, para identificar las rutas por su nombre; la segunda, el algoritmo completo; la tercera, las opciones que se comparan en cada paso.

# %%
explica_rutas(proyecto, datos_proyecto['duration'])

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni hasta la duración tope"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_tope, margen_restante = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'], reduction=100, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=100)

# %%
#| tbl-cap: "Opciones consideradas en cada paso"
#| label: tbl-ackoffPaso
detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)

# %%
resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],
                        'sobrecoste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},
                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))
resumen['duración'] = duracion_normal - resumen.index
resumen['sobrecoste acumulado'] = resumen['sobrecoste del paso'].cumsum()
resumen['acumulado = suma de pasos'] = [' + '.join(f'{v:.0f}' for v in resumen['sobrecoste del paso'].iloc[:i + 1]) + f' = {resumen["sobrecoste acumulado"].iloc[i]:.0f}' for i in range(len(resumen))]
display(resumen)
Markdown(f'**Respuesta:** la tabla da el sobrecoste directo acumulado de cada semana de reducción; hasta la duración tope '
         f'({int(resumen["duración"].iloc[-1])} semanas, {plural(len(resumen), "semana")} de reducción) el sobrecoste acumulado es {resumen["sobrecoste acumulado"].iloc[-1]:.0f} k€.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Para cada duración: coste directo = normal + sobrecoste acumulado; indirecto = coste semanal × duración; penalización = penalización semanal × semanas de retraso sobre el plazo; prima = −(prima semanal × semanas de adelanto). El coste total es la suma de las cuatro columnas.

# %%
#| tbl-cap: "Coste total según la duración"
#| label: tbl-costeTotal
curva = pd.DataFrame({'duración': [duracion_normal] + resumen['duración'].tolist(),
                      'coste directo': [coste_directo_normal] + (coste_directo_normal + resumen['sobrecoste acumulado']).tolist()}).set_index('duración')
curva['coste indirecto'] = coste_indirecto_semanal * curva.index
curva['penalización'] = [penalizacion_semanal * max(0, T - plazo_contrato) for T in curva.index]
curva['prima (negativa)'] = [-prima_semanal * max(0, plazo_contrato - T) for T in curva.index]
curva['coste total'] = curva.sum(axis=1)
curva

# %%
#| fig-cap: "Componentes del coste frente a la duración"
#| label: fig-curvas
ax = curva.plot(marker='o', figsize=(8, 4.5)); ax.axvline(plazo_contrato, color='grey', linestyle=':', label='plazo contractual')
ax.set_ylabel('k€'); ax.set_xlabel('duración (semanas)'); ax.invert_xaxis(); ax.legend(); ax.grid(alpha=0.3);

# %%
duracion_optima = int(curva['coste total'].idxmin())
_f = curva.loc[duracion_optima]
_k = duracion_normal - duracion_optima
Markdown(f'En la duración óptima, {duracion_optima} semanas ({plural(_k, "semana")} de reducción): directo {coste_directo_normal:.0f} + {_f["coste directo"] - coste_directo_normal:.0f} = {_f["coste directo"]:.0f}; '
         f'indirecto {coste_indirecto_semanal}·{duracion_optima} = {_f["coste indirecto"]:.0f}; {txt_plazo(duracion_optima)}; '
         f'total {_f["coste directo"]:.0f} + {_f["coste indirecto"]:.0f} {_sg(_f["penalización"] + _f["prima (negativa)"])} = {_f["coste total"]:.0f} k€. '
         f'**Respuesta:** la duración óptima es {duracion_optima} semanas, con un coste total de {curva["coste total"].min():.0f} k€ (frente a {total_normal:.0f} k€ sin reducir, un ahorro de {total_normal - curva["coste total"].min():.0f} k€), '
         f'{"antes del plazo contractual" if duracion_optima < plazo_contrato else "justo en el plazo contractual" if duracion_optima == plazo_contrato else "con retraso sobre el plazo contractual"} ({plazo_contrato} semanas).')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Cada semana de reducción cuesta el sobrecoste del paso de Ackoff-Sasieni y ahorra el coste indirecto semanal **más** la penalización semanal mientras el proyecto esté por detrás del plazo, o **más** la prima mientras esté por delante. La reducción compensa mientras el sobrecoste del paso sea menor que ese ahorro.

# %%
decision = resumen[['duración', 'sobrecoste del paso']].copy()
decision.insert(0, 'de → a', [f'{T + 1} → {T}' for T in decision['duración']])
decision['ahorro por la semana'] = [coste_indirecto_semanal + (penalizacion_semanal if T + 1 > plazo_contrato else prima_semanal) for T in decision['duración']]
decision['ahorro = indirecto + penalización o prima'] = [f'{coste_indirecto_semanal} + {penalizacion_semanal} (penalización) = {coste_indirecto_semanal + penalizacion_semanal}' if T + 1 > plazo_contrato
                                                         else f'{coste_indirecto_semanal} + {prima_semanal} (prima) = {coste_indirecto_semanal + prima_semanal}' for T in decision['duración']]
decision['conviene'] = (decision['sobrecoste del paso'] < decision['ahorro por la semana']).map({True: 'sí', False: 'no'})
display(decision.drop(columns='ahorro por la semana'))
_antes = coste_indirecto_semanal + penalizacion_semanal; _despues = coste_indirecto_semanal + prima_semanal
_txt = (f'Mientras el proyecto va por detrás del plazo ({plazo_contrato} semanas), cada semana reducida ahorra {coste_indirecto_semanal} + {penalizacion_semanal} = {_antes} k€; '
        f'una vez alcanzado, sólo {coste_indirecto_semanal} + {prima_semanal} = {_despues} k€, y se compara con el sobrecoste del paso. ')
paso_siguiente = resumen[resumen['duración'] == duracion_optima - 1]
if len(paso_siguiente) and duracion_optima <= plazo_contrato:
    sobre = float(paso_siguiente['sobrecoste del paso'].iloc[0])
    prima_necesaria = sobre - coste_indirecto_semanal
    _txt += (f'**Respuesta:** se reduce hasta {duracion_optima} semanas; reducir una semana más (a {duracion_optima - 1}) costaría {sobre:.0f} k€ y ahorraría {coste_indirecto_semanal} de indirectos más la prima p: '
             f'compensa si {sobre:.0f} < {coste_indirecto_semanal} + p, es decir, con una prima de más de {sobre:.0f} − {coste_indirecto_semanal} = {max(prima_necesaria, 0):.0f} k€/semana (ahora es {prima_semanal}).')
elif len(paso_siguiente):
    _txt += f'**Respuesta:** se reduce hasta {duracion_optima} semanas; la semana siguiente sigue estando por detrás del plazo, así que la prima no interviene en esa decisión.'
else:
    _txt += f'**Respuesta:** se reduce hasta {duracion_optima} semanas, la duración tope: ninguna prima permitiría reducir más.'
Markdown(_txt)
