# ---
# title: Dilema 04
# subtitle: "riesgo en G (30%, +2 semanas): P(T ≤ 23) pasa de 0.83 a 0.62, umbral 70%"
# coleccion: 19_etica
# variante: riesgo-silenciado
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
# El comité de inversión decide el jueves si autoriza la fase de ejecución de un proyecto. Su regla, escrita en el procedimiento, es autorizar sólo si la probabilidad de terminar en el plazo contractual de 23 semanas es al menos del 70%. El cuadro de prelaciones, las duraciones medias y las desviaciones típicas (semanas) se adjuntan.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 3       , 1                 ),
    ('B'     , '---'      , 4       , 0.2               ),
    ('C'     , 'A,B'      , 7       , 0.1               ),
    ('D'     , 'C'        , 3       , 0.8               ),
    ('E'     , 'D'        , 2       , 0.3               ),
    ('F'     , 'D'        , 3       , 0.5               ),
    ('G'     , 'E'        , 6       , 0.3               ),
    ('H'     , 'E,F'      , 2       , 0.3               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# El lunes el jefe de proyecto recibe una llamada: el único proveedor cualificado de la actividad G ha entrado en concurso de acreedores. Su estimación, contrastada con compras, es que con probabilidad 30% habrá que cambiar de proveedor, lo que añadiría 2 semanas a G.

# %% [markdown]
# > **De: Jefe de proyecto (al responsable de riesgos)**
# > El informe para el comité está cerrado desde el viernes con la probabilidad que calculamos; no voy a reabrirlo por una llamada. Si el proveedor cae, lo gestionamos como riesgo en ejecución. No lo menciones el jueves.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C4 — Retención de información.** Callar un hecho conocido que la otra parte necesitaría para decidir (un riesgo, una holgura, una discrepancia). *(cf. Código Ético del PMI, 5.3.1 y 5.2.3.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule la probabilidad de terminar en 23 semanas o menos sin el riesgo, con la aproximación normal del PERT y con una simulación Monte Carlo (20 000 réplicas, duraciones normales).  "
pregunta_2 = "2. (A) Incorpore el riesgo a la simulación (la duración de G es normal más 2 semanas con probabilidad 30%) y recalcule la probabilidad. ¿Cambia la decisión del comité según su regla?  "
pregunta_3 = "3. (B) ¿Qué criterio de referencia incumple no mencionar el riesgo el jueves? ¿Es relevante que el riesgo pueda «gestionarse en ejecución»? ¿Qué convierte a la omisión en material?  "
pregunta_4 = "4. (C) ¿Qué hace el responsable de riesgos antes del jueves y qué dice al jefe de proyecto y al comité? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
plazo, umbral, X, p_riesgo, delta = 23, 0.7, 'G', 0.3, 2
datos_proyecto = datos_enunciado.copy()
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = int(proyecto.duration(datos_proyecto['duration']))
sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])
p_pert = float(norm.cdf(plazo, duracion, sigma))

# %%
def porc(p, d=1):
    """Probabilidad o fracción como porcentaje: 0.453 -> «45.3 %»."""
    return f'{100 * float(p):.{d}f} %'

def tabla_md(filas, columnas):
    """Tabla Markdown a partir de una lista de filas."""
    esc = lambda x: str(x).replace('|', '\\|')
    lineas = ['| ' + ' | '.join(columnas) + ' |', '|' + '---|' * len(columnas)]
    lineas += ['| ' + ' | '.join(esc(x) for x in f) + ' |' for f in filas]
    return '\n'.join(lineas)

def explica_indicadores(ind, BAC=None, titulo=''):
    """CV, SV, CPI y SPI (y EAC y VAC si se da el BAC) con la fórmula y la sustitución."""
    PV, EV, AC = float(ind['PV']), float(ind['EV']), float(ind['AC'])
    lineas = [f'- CV = EV − AC = {EV:.1f} − {AC:.1f} = **{EV - AC:.1f}** k€',
              f'- SV = EV − PV = {EV:.1f} − {PV:.1f} = **{EV - PV:.1f}** k€',
              f'- CPI = EV / AC = {EV:.1f} / {AC:.1f} = **{EV / AC:.3f}**',
              f'- SPI = EV / PV = {EV:.1f} / {PV:.1f} = **{EV / PV:.3f}**']
    if BAC is not None:
        eac = BAC / (EV / AC)
        lineas += [f'- EAC = BAC / CPI = {BAC:.0f} / {EV / AC:.3f} = **{eac:.1f}** k€',
                   f'- VAC = BAC − EAC = {BAC:.0f} − {eac:.1f} = **{BAC - eac:.1f}** k€']
    return (f'{titulo}\n\n' if titulo else '') + '\n'.join(lineas)

def rubrica_caso(afectados, alternativas, decision):
    """La rúbrica del bloque C aplicada al caso: afectados con cifras, alternativas con las tres pruebas y decisión."""
    partes = ['**La rúbrica aplicada a este caso** (lo que debe contener una respuesta que la supere):', '',
              '*(1) Afectados y qué gana o pierde cada uno:*', '']
    partes += [f'- **{a}**: {b}' for a, b in afectados]
    partes += ['', '*(2) y (3) Alternativas y las tres pruebas:*', '',
               tabla_md(alternativas, ['alternativa', 'legalidad', 'publicidad', 'reversibilidad']), '',
               '*(4) Decisión y comunicación* (un camino defendible, no el único): ' + decision]
    return Markdown('\n'.join(partes))

def sin_ceros(valores):
    """Los términos no nulos de una suma (para escribirla sin «+ 0.0»)."""
    v = pd.Series(valores).astype(float)
    return v[v != 0]

def camino_critico_sigma(proyecto, duraciones, varianzas):
    """Camino crítico en orden de ejecución (el de mayor varianza si hay varios) y su desviación típica."""
    cal = proyecto.calendar(duraciones)
    rutas = [ordena_ruta([a for a in r if not str(a).startswith('@')], cal) for r in proyecto.critical_path(duraciones).values()]
    acts = max(rutas, key=lambda r: sum(float(varianzas[a]) for a in r))
    return acts, float(sum(float(varianzas[a]) for a in acts)) ** 0.5

def rutas_media_varianza(proyecto, duraciones, varianzas):
    """Todas las rutas en orden de ejecución con su duración media y su varianza, de más larga a más corta."""
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    filas = []
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        filas.append((' – '.join(acts), float(sum(float(duraciones[a]) for a in acts)), float(sum(float(varianzas[a]) for a in acts))))
    return pd.DataFrame(filas, columns=['ruta', 'media', 'varianza']).sort_values(['media', 'varianza'], ascending=False).reset_index(drop=True)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con el camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Duración y varianza de cada ruta"
#| label: tbl-rutas
R = proyecto.path_matrix(dummies=False)
display(detalle_rutas(R, datos_proyecto['duration'], 'duración'))
detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)

# %%
dists = {act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}
sin_riesgo = proyecto.monte_carlo(dict(dists), n_samples=20000, seed=2024)
tau0 = np.asarray(sin_riesgo.tau_samples, dtype=float)
p_mc0 = float((tau0 <= plazo).mean())
camino, _ = camino_critico_sigma(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])
Markdown(f'''Camino crítico **{' – '.join(camino)}**: duración esperada {' + '.join(str(int(datos_proyecto.loc[a, 'duration'])) for a in camino)} = **{duracion}** semanas; {explica_sigma_ruta(camino, datos_proyecto['variance'])} semanas.

- PERT (aproximación normal): {explica_probabilidad(duracion, sigma, plazo)}.
- Monte Carlo ({len(tau0)} réplicas con duraciones normales, semilla 2024): el proyecto termina en {plazo} semanas o menos en {int((tau0 <= plazo).sum())} réplicas: P(T ≤ {plazo}) = {int((tau0 <= plazo).sum())} / {len(tau0)} = **{p_mc0:.3f}**.

**Respuesta.** Sin el riesgo, P(T ≤ {plazo}) = {p_pert:.3f} con el PERT y {p_mc0:.3f} con la simulación; {'las dos superan' if min(p_pert, p_mc0) >= umbral else 'no superan las dos'} el umbral del comité ({porc(umbral, 0)}), {'así que el comité autorizaría' if min(p_pert, p_mc0) >= umbral else 'así que la autorización ya sería dudosa'}.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
rng_mc = np.random.default_rng(2024)
mu_X, sd_X = float(datos_proyecto.loc[X, 'duration']), float(datos_proyecto.loc[X, 'standard_deviation'])
def duracion_X_con_riesgo(size=None):
    '''Normal más delta semanas con probabilidad p_riesgo (cambio de proveedor).'''
    return rng_mc.normal(mu_X, sd_X, size) + delta * (rng_mc.random(size) < p_riesgo)
dists_riesgo = dict(dists); dists_riesgo[X] = duracion_X_con_riesgo
con_riesgo = proyecto.monte_carlo(dists_riesgo, n_samples=20000, seed=2024)
tau1 = np.asarray(con_riesgo.tau_samples, dtype=float)
p_mc1 = float((tau1 <= plazo).mean())
comparacion = pd.DataFrame({'sin el riesgo': sin_riesgo.summary().iloc[0], 'con el riesgo': con_riesgo.summary().iloc[0]}).T
comparacion[f'P(T ≤ {plazo})'] = [p_mc0, p_mc1]
comparacion.round(3)

# %%
cambia = p_mc0 >= umbral > p_mc1
p_menos = float(norm.cdf(plazo - delta, duracion, sigma))
aprox = (1 - p_riesgo) * p_pert + p_riesgo * p_menos
Markdown(f'''La tabla resume la duración simulada (media, desviación típica, percentiles 5, 50 y 95, mínimo y máximo). Con el riesgo, el proyecto termina en {plazo} semanas o menos en {int((tau1 <= plazo).sum())} de {len(tau1)} réplicas: **P(T ≤ {plazo}) = {p_mc1:.3f}**, {'por debajo' if p_mc1 < umbral else 'por encima'} del umbral del {porc(umbral, 0)}.

Comprobación a mano con el PERT: {X} es crítica, así que cuando el riesgo se materializa las {delta} semanas van enteras al plazo y el proyecto debe acabar en {plazo} − {delta} = {plazo - delta} semanas «de las normales». Por la probabilidad total:

P(T ≤ {plazo}) ≈ (1 − {p_riesgo}) · P(T ≤ {plazo}) + {p_riesgo} · P(T ≤ {plazo - delta}) = {1 - p_riesgo:.2f} · {p_pert:.4f} + {p_riesgo:.2f} · {p_menos:.4f} = **{aprox:.3f}**, con P(T ≤ {plazo - delta}) = `=DISTR.NORM.N({plazo - delta}; {duracion}; {str(round(sigma, 4)).replace('.', ',')}; VERDADERO)`; en línea con la simulación ({p_mc1:.3f}).

**Respuesta.** Con el riesgo la probabilidad baja de {p_mc0:.3f} a **{p_mc1:.3f}**. {'**La decisión cambia**: según su regla, el comité no autorizaría con la información del lunes (o exigiría antes un plan de mitigación).' if cambia else 'La decisión del comité no cambia según su regla, aunque el dato sigue siendo relevante.'}''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Criterio C4 (retención de información).** El comité decide con una regla escrita (autorizar si P ≥ {porc(umbral, 0)}) sobre una cifra, y el jefe de proyecto conoce desde el lunes un hecho —la quiebra del único proveedor de {X}, con un {porc(p_riesgo, 0)} de añadir {delta} semanas— que mueve esa cifra de {p_mc0:.3f} a {p_mc1:.3f}. Callarlo el jueves es callar un hecho conocido que la otra parte necesita para decidir.

Que el riesgo pueda «gestionarse en ejecución» no es relevante: la decisión de autorizar con ese riesgo es del comité, no del jefe de proyecto, y omitirlo le quita la posibilidad de tomarla con los datos. Que el informe estuviera «cerrado desde el viernes» tampoco: la información llegó antes de la decisión.

**Respuesta.** Incumple **C4**. La omisión es **material** porque la decisión cambia con el dato: la probabilidad pasa {'de un lado al otro' if cambia else 'cerca'} del umbral ({p_mc0:.3f} → {p_mc1:.3f} frente a {porc(umbral, 0)}). Ese es el test objetivo; si el riesgo dejara la probabilidad por encima del umbral, informar seguiría siendo exigible (proporcionar información exacta a tiempo, cf. PMI 5.2.3), pero no cambiaría la decisión.''')

# %%
preguntas_test = [
    ('No comunicar al comité un riesgo conocido que cambia su decisión es…', ['C4, retención de información', 'C2, verdad a medias', 'C8, reserva declarada', 'gestión de riesgos en ejecución'],
     'Se calla un hecho conocido que la otra parte necesita para decidir.'),
    ('La omisión es material cuando…', ['la decisión cambia con el dato omitido', 'el riesgo tiene probabilidad mayor que 0,5', 'el informe ya está cerrado', 'el dato llega por teléfono'],
     'El test objetivo es si la cifra que gobierna la decisión cruza el umbral.'),
    ('Si G no fuera crítica, el efecto del riesgo en el plazo sería…', ['menor, absorbido en parte por su holgura', 'el mismo', 'mayor', 'nulo siempre'],
     'Un retraso x en una actividad con holgura H retrasa el proyecto máx(0, x − H).'),
]
import os
def a_gift(preguntas_test, titulo, semilla=1):
    r = np.random.default_rng(semilla)
    lineas = [f'$CATEGORY: {titulo}', '']
    for i, (q, ops, fb) in enumerate(preguntas_test, 1):
        correcta = ops[0]
        todas = [ops[j] for j in r.permutation(len(ops))]
        cuerpo = ' '.join((f'={v}' if v == correcta else f'~{v}') for v in todas)
        lineas.append(f'::{titulo} P{i}::{q} {{{cuerpo} ####{fb}}}')
        lineas.append('')
    return '\n'.join(lineas)
gift = a_gift(preguntas_test, 'Ética 19 — riesgo silenciado')
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_etica.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: (1) identificar a todos los afectados y qué pierde o gana cada uno; (2) formular al menos dos alternativas además de la que se propone en el enunciado; (3) someter cada alternativa a las tres pruebas —legalidad (¿cumple la norma?), publicidad (¿la defendería si se hiciera pública?) y reversibilidad (¿la aceptaría en el papel del afectado?)—; (4) decidir y redactar la comunicación con las cifras del bloque A. Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.

# %%
rubrica_caso(
    [('Comité de inversión', f'autoriza con P = {p_mc0:.2f} creyendo que cumple su regla, cuando con el dato del lunes es {p_mc1:.2f}'),
     ('Jefe de proyecto', 'evita reabrir el informe y quizá un retraso en la autorización; se expone a que el comité lo descubra después'),
     ('Responsable de riesgos', 'si calla, comparte la omisión; si habla, se enfrenta al jefe de proyecto'),
     ('Cliente del contrato', f'soporta el riesgo de no tener la obra en {plazo} semanas sin que nadie haya preparado una mitigación')],
    [('Callar el jueves y «gestionarlo en ejecución» (propuesta)', 'no hay norma legal directa', 'no: el comité no lo aceptaría si lo supiera', 'no'),
     ('Llevar al comité las dos probabilidades y un plan de mitigación con su coste', 'sí', 'sí', 'sí'),
     ('Pedir al comité que aplace la decisión hasta tener un proveedor alternativo precalificado', 'sí', 'sí', 'sí'),
     ('Si el jefe de proyecto se niega, elevarlo a quien corresponda en la organización', 'sí', 'sí', 'sí')],
    f'''antes del jueves, recalcular (este apartado) y redactar una página con P = {p_mc0:.2f} sin el riesgo y {p_mc1:.2f} con él, y un plan de mitigación (proveedor alternativo precalificado, acopio anticipado o acortar otra actividad crítica) con su coste; llevarla primero al jefe de proyecto: «el informe tiene una cifra que sabemos que ya no es cierta». Al comité, con o sin su acuerdo: la actualización y el plan. Lo que no cabe: callar el jueves.''')
