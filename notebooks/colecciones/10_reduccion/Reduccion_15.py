# ---
# title: Reduccion 15
# subtitle: "9 actividades; límite 3 operarios, reducir dos semanas (Ackoff sin recursos falla)"
# coleccion: 10_reduccion
# variante: crashing-recursos
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
# Considere el proyecto cuyo cuadro de prelaciones, duración normal y tope (semanas), operarios por actividad y coste unitario de reducción (k€/semana) se adjunta. Cada actividad necesita un número fijo de operarios durante toda su duración y la empresa dispone como máximo de 3 operarios en cada semana. Las actividades con CUR infinito no pueden acelerarse. Se quiere reducir la duración del proyecto **dos semanas** respetando el límite de operarios.

# %%
pregunta_1 = "1. Determine la duración del proyecto sin límite de recursos y con el límite de operarios (algoritmo de asignación). Indique la cadena crítica con recursos: qué actividades fijan la duración y qué esperas por recurso intervienen.  "
pregunta_2 = "2. Aplique el algoritmo de Ackoff-Sasieni ignorando los recursos para reducir dos semanas, aplique esas reducciones y replanifique con el límite de operarios. ¿Se ha conseguido reducir la duración del proyecto con recursos? ¿Por qué?  "
pregunta_3 = "3. Reduzca ahora dos semanas teniendo en cuenta el límite: pruebe en cada paso los candidatos (cortes de la cadena crítica con recursos, actividades sueltas y parejas) y elija el más barato que de verdad acorte el proyecto replanificado. Compare el sobrecoste con el del apartado anterior.  "
pregunta_4 = "4. Suponga que acelerar una actividad exige un operario más por cada semana recortada (modelo con recurso). ¿Cambia la solución?  "
pregunta_5 = "5. Alternativa: contratar un operario más (límite 4) sin acelerar nada. ¿Qué duración se obtiene? ¿Compensa frente a acelerar?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, operarios, cap_duration, ucr   
    ('A'     , '---'      , 2       , 2        , 1           , 3     ),
    ('B'     , '---'      , 7       , 1        , 4           , 3     ),
    ('C'     , 'A'        , 4       , 2        , 4           , np.inf),
    ('D'     , 'C'        , 6       , 1        , 4           , 6     ),
    ('E'     , 'C'        , 4       , 1        , 2           , 2     ),
    ('F'     , 'B,E'      , 1       , 2        , 1           , np.inf),
    ('G'     , 'F'        , 2       , 3        , 1           , 7     ),
    ('H'     , 'D,F'      , 1       , 1        , 1           , np.inf),
    ('I'     , 'G'        , 7       , 1        , 5           , 6     ),
], columns=['activity', 'predecessor', 'duration', 'operarios', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la rehabilitación de una nave industrial para un taller; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Levantamiento topográfico y mediciones',
    'B': 'Proyecto de ejecución y licencia',
    'C': 'Desmontaje de instalaciones antiguas',
    'D': 'Demolición de tabiquería interior',
    'E': 'Refuerzo de la estructura metálica',
    'F': 'Reparación de la cubierta',
    'G': 'Solera y canalizaciones',
    'H': 'Cerramientos y carpintería exterior',
    'I': 'Instalación eléctrica',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
limite = [3]
base = datos_proyecto[['predecessor', 'duration', 'operarios']]

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones normales y el camino crítico (sin límite de operarios)"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario sin límite de recursos"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion_libre = proyecto.duration(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %%
#| fig-cap: "Gantt de inicios más tempranos con los operarios de cada actividad (fila Total: carga semanal)"
#| label: fig-ganttInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Carga de operarios frente al límite"
#| label: fig-cargasInicial
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite)

# %%
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
exceso = cargas[cargas > limite[0]]
Markdown(f'Sin límite el proyecto dura {duracion_libre} semanas, pero en las semanas {", ".join(str(t) for t in exceso.index)} harían falta '
         f'{", ".join(f"{int(c)}" for c in exceso)} operarios, más de los {limite[0]} disponibles: hay que reprogramar.')

# %% [markdown]
# El algoritmo de asignación recorre las semanas una a una. En cada semana son *candidatas* las actividades no empezadas cuyas predecesoras han terminado; se ordenan por prioridad (menor inicio más tardío en el calendario sin recursos, después menor duración) y cada una empieza si, en todas las semanas que ocuparía, los operarios ya comprometidos más los suyos no superan el límite; si no, espera. La traza muestra la decisión de cada semana y, entre paréntesis, el recurso que bloquea a las que esperan.

# %%
res0 = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios (filas 💤: esperas por falta de operarios)"
#| label: fig-ganttAsignado
pinta_tikz(res0.dibujos['operarios'])

# %%
#| tbl-cap: "Calendario con recursos: inicio, fin y espera de cada actividad"
#| label: tbl-calendarioRecursos
calendario_recursos = pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res0.inicios,
                                    'fin con límite': res0.fines, 'espera (semanas)': res0.esperas}).fillna(0).astype(int)
calendario_recursos

# %%
#| tbl-cap: "Carga de operarios semana a semana frente al límite"
#| label: tbl-cargasRecursos
pd.concat([res0.cargas['operarios'].rename('operarios ocupados'), res0.limites['operarios'].rename('límite')], axis=1).T

# %%
cadena = proyecto.copy().cadena_critica_recursos(base.copy(), 'duration', ['operarios'], limite)
Markdown(f'Sin límite: {duracion_libre} semanas. Con {limite[0]} operarios: {res0.duracion} semanas ({res0.duracion - duracion_libre} más). '
         f'Cadena crítica con recursos: {", ".join(sorted(cadena["actividades"]))}.')

# %%
#| tbl-cap: "Arcos de la cadena crítica con recursos"
#| label: tbl-cadena
cadena['arcos'].rename(columns={'de': 'de', 'a': 'a', 'motivo': 'motivo (precedencia o recurso)'})

# %% [markdown]
# La cadena crítica con recursos se lee hacia atrás desde la última actividad: cada arco une una actividad con otra que empieza justo cuando ella termina, bien porque es su predecesora (*precedencia*), bien porque le libera los operarios que esperaba (*recurso*). Los arcos de recurso son esperas, no prelaciones: la duración la fija esa cadena (precedencias más esperas), no el camino crítico del grafo.

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Ackoff-Sasieni sólo mira el grafo: acorta en cada paso, una semana, la combinación más barata de actividades que reduce todas las rutas críticas *sin recursos*. La tabla muestra, para cada ruta, el CUR de sus actividades y su duración tras cada paso.

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni sin tener en cuenta los recursos"
#| label: tbl-ackoff
tabla_ack, pasos_ack, dur_ack, margen = proyecto.ackoff(durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'], costs=datos_proyecto['ucr'], reduction=2, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %%
#| tbl-cap: "Opciones consideradas en cada paso"
#| label: tbl-ackoffPaso
detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos_ack)

# %%
resumen_ack = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos_ack.values()],
                            'coste del paso': [float(sum(datos_proyecto.loc[acts, 'ucr'])) for acts in pasos_ack.values()]},
                           index=pd.Index(range(1, len(pasos_ack) + 1), name='paso'))
coste_ack = float(resumen_ack['coste del paso'].sum())
resumen_ack

# %% [markdown]
# Aplicadas esas reducciones, se vuelve a programar con el límite de operarios:

# %%
res_ack = proyecto.copy().asignar_multi(base.assign(duration=dur_ack.reindex(base.index).astype(int)), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios tras las reducciones de Ackoff-Sasieni"
#| label: fig-ganttAckoff
pinta_tikz(res_ack.dibujos['operarios'])

# %%
#| tbl-cap: "Duración e inicio de cada actividad antes y después de aplicar Ackoff-Sasieni (con el límite)"
#| label: tbl-inicioAckoff
pd.DataFrame({'duración normal': datos_proyecto['duration'], 'duración Ackoff': dur_ack.reindex(datos_proyecto.index).astype(int),
              'inicio con límite (antes)': res0.inicios, 'inicio con límite (después)': res_ack.inicios,
              'fin con límite (antes)': res0.fines, 'fin con límite (después)': res_ack.fines}).T

# %%
Markdown(f'Ackoff sin recursos reduce el proyecto a {proyecto.duration(dur_ack)} semanas sobre el papel (coste {coste_ack:.0f} k€), '
         f'pero con el límite de operarios la duración es {res_ack.duracion} semanas (antes {res0.duracion}). '
         f'Las actividades acortadas ({", ".join(sorted(set(a for acts in pasos_ack.values() for a in acts)))}) están en el camino crítico del grafo, '
         f'pero lo que manda con recursos es la cadena {", ".join(sorted(cadena["actividades"]))}: '
         'el dinero se gasta sin acortar lo que de verdad fija el plazo.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# `crashing_recursos` repite en cada paso cuatro operaciones: (1) programa con el límite, (2) halla la cadena crítica con recursos, (3) evalúa como candidatos las actividades reducibles de esa cadena una a una, por parejas y el corte de coste mínimo de la cadena, **replanificando** con cada candidato, y (4) se queda con el más barato que de verdad acorta la duración replanificada. La tabla muestra, paso a paso, la cadena crítica, lo que habría elegido Ackoff-Sasieni sin recursos, cuántos candidatos se evaluaron y cuántas reducciones de una sola actividad no acortaron nada.

# %%
#| tbl-cap: "Crashing con recursos limitados, paso a paso"
#| label: tbl-crashing
cr = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], limite,
                                       datos_proyecto['cap_duration'], datos_proyecto['ucr'], reduccion=2)
cr.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'reducciones_sin_efecto', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]

# %%
#| tbl-cap: "Duraciones finales y semanas recortadas a cada actividad"
#| label: tbl-duracionesCrashing
pd.DataFrame({'duración normal': datos_proyecto['duration'], 'duración tope': datos_proyecto['cap_duration'],
              'duración final': cr.duraciones.reindex(datos_proyecto.index).astype(int), 'recortado': cr.reducciones.reindex(datos_proyecto.index).astype(int),
              'CUR': datos_proyecto['ucr']}).T

# %%
datos_cr = base.assign(duration=cr.duraciones.reindex(base.index).astype(int))
res_cr = proyecto.copy().asignar_multi(datos_cr, 'duration', ['operarios'], limite, report=False, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios tras el crashing con recursos"
#| label: fig-ganttCrashing
pinta_tikz(res_cr.dibujos['operarios'])

# %%
Markdown(f'Duración {res0.duracion} → {cr.duracion_final} semanas (parada: «{cr.motivo_parada}») con un sobrecoste de {cr.coste_total:.0f} k€, '
         f'frente a {coste_ack:.0f} k€ de Ackoff sin recursos, que no consiguió reducir la duración con recursos '
         f'(quedó en {res_ack.duracion}). La diferencia está en qué actividades se acortan: las de la cadena crítica con recursos, '
         'comprobando en cada paso que la replanificación realmente termina antes.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# En el modelo *con recurso*, acelerar una actividad consume operarios: cada semana recortada añade un operario a su demanda, de modo que la reducción puede chocar con el mismo límite que intenta esquivar.

# %%
#| tbl-cap: "Crashing con recursos cuando acelerar consume un operario por semana recortada"
#| label: tbl-crashingRecurso
extra = {a: {'operarios': 1} for a in datos_proyecto.index if np.isfinite(datos_proyecto.loc[a, 'ucr'])}
cr_b = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], limite,
                                         datos_proyecto['cap_duration'], datos_proyecto['ucr'], reduccion=2,
                                         demanda_extra=extra)
cr_b.pasos[['cadena_critica', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]

# %%
#| tbl-cap: "Demanda de operarios de cada actividad tras las reducciones (modelo con recurso)"
#| label: tbl-demandaRecurso
pd.DataFrame({'operarios (normal)': datos_proyecto['operarios'], 'operarios (tras recortar)': cr_b.demanda['operarios'].reindex(datos_proyecto.index),
              'recortado': cr_b.reducciones.reindex(datos_proyecto.index).astype(int)}).T

# %%
if cr_b.duracion_final <= res0.duracion - 2 and cr_b.coste_total == cr.coste_total and (cr_b.reducciones == cr.reducciones).all():
    cierre = 'Aun consumiendo el recurso, el límite todavía permite la reducción pedida con las mismas actividades y el mismo coste: la demanda extra cabe en las semanas en que se aplica.'
elif cr_b.duracion_final <= res0.duracion - 2:
    cierre = 'Aun consumiendo el recurso, el límite todavía permite la reducción pedida, pero con otras actividades o con un coste distinto.'
else:
    cierre = 'Como acelerar consume el recurso escaso, el límite frena también las reducciones: no se alcanza lo pedido sin ampliar el recurso.'
Markdown(f'Modelo con recurso: parada «{cr_b.motivo_parada}», duración final {cr_b.duracion_final} semanas, '
         f'sobrecoste {cr_b.coste_total:.0f} k€ (modelo con dinero: {cr.duracion_final} semanas y {cr.coste_total:.0f} k€). {cierre}')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
mas = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], [limite[0] + 1], report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con un operario más, sin acelerar ninguna actividad"
#| label: fig-ganttMasOperarios
pinta_tikz(mas.dibujos['operarios'])

# %%
Markdown(f'Con {limite[0] + 1} operarios, sin acelerar nada: {mas.duracion} semanas (antes {res0.duracion}). '
         f'Se ahorran {res0.duracion - mas.duracion} semana(s). Reducir {res0.duracion - cr.duracion_final} semanas acelerando cuesta '
         f'{cr.coste_total:.0f} k€. Compensa contratar si el coste del operario adicional durante las {mas.duracion} semanas del proyecto es menor que ese sobrecoste '
         '(y la comparación es justa solo con el mismo número de semanas ahorradas).')
