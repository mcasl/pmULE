# ---
# title: Recursos 10
# subtitle: "9 actividades, dos recursos y propuesta de desplazamiento"
# coleccion: 11_recursos
# variante: dos-recursos
# nivel: 3
# grafo: roy
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (semanas) se adjunta. Cada actividad necesita un número de operarios y, algunas, una máquina excavadora (columna *machines*). El jefe de obra propone retrasar H 2 semanas y B 1 semana para suavizar la demanda de operarios.

# %%
pregunta_1 = "1. Dibuje los diagramas de cargas de operarios y de máquinas con las actividades en su inicio más temprano. ¿Cuántas máquinas hacen falta como máximo?  "
pregunta_2 = "2. Evalúe la propuesta del jefe de obra: ¿respeta las holguras? ¿Reduce la suma de cuadrados de las cargas de operarios?  "
pregunta_3 = "3. Compare la propuesta con el resultado del algoritmo de nivelación. ¿Cuál es mejor?  "
pregunta_4 = "4. ¿Qué ocurre con la carga de máquinas tras la nivelación de operarios? ¿Puede una nivelación empeorar el otro recurso?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources, machines
    ('A'     , '---'      , 2       , 4        , 2       ),
    ('B'     , '---'      , 7       , 1        , 0       ),
    ('C'     , 'A'        , 4       , 4        , 0       ),
    ('D'     , 'C'        , 2       , 2        , 2       ),
    ('E'     , 'D'        , 8       , 4        , 0       ),
    ('F'     , 'B,D'      , 1       , 4        , 2       ),
    ('G'     , 'F'        , 8       , 4        , 2       ),
    ('H'     , 'F'        , 7       , 1        , 1       ),
    ('I'     , 'E,G'      , 3       , 1        , 2       ),
], columns=['activity', 'predecessor', 'duration', 'resources', 'machines']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %%
def semanas_txt(lista):
    """«la semana 6», «las semanas 1, 2 y 5» o «ninguna semana»."""
    lista = [str(int(x)) for x in lista]
    if not lista:
        return 'ninguna semana'
    return ('la semana ' if len(lista) == 1 else 'las semanas ') + lista_y(lista)


def explica_cuadrados(cargas):
    """Σ c² agrupando las semanas de igual carga: «6·3² + 4·5² = 54 + 100 = 154»."""
    c = pd.Series(cargas).astype(int)
    cuenta = c.value_counts().sort_index()
    terminos = [f'{n}·{v}²' if n > 1 else f'{v}²' for v, n in cuenta.items()]
    parciales = [f'{n * v * v}' for v, n in cuenta.items()]
    return f'Σ c² = {" + ".join(terminos)} = {" + ".join(parciales)} = **{int((c ** 2).sum())}**'


def explica_nivelacion(cuadrados, calendario):
    """Una línea por actividad recorrida: desplazamientos probados, suma mínima y elección."""
    tabla = getattr(cuadrados, 'data', cuadrados)
    lineas, elegidos = [], {}
    for a, fila in tabla.iterrows():
        v = fila.dropna()
        k = int(v.index.max())
        H = int(calendario.loc[a, 'H_total'])
        minimo = v.min()
        mejor = int(v.idxmin())
        empates = [int(x) for x in v.index[v == minimo]]
        if k == 0:
            probado = 'solo puede probarse 0'
        else:
            probado = f'se prueban 0 a {k}'
        if k < H:
            probado += (f' (no {H}: los desplazamientos ya aplicados a actividades que comparten su holgura '
                        f'le han dejado {plural(k, "semana")})')
        empate = (f'; empatan {lista_y([str(x) for x in empates])} y se toma el menor' if len(empates) > 1 else '')
        lineas.append(f'- {a} (holgura total {H}): {probado}; suma mínima {int(minimo)} con {mejor}{empate} → '
                      + (f'se desplaza **{plural(mejor, "semana")}**' if mejor > 0 else 'no se desplaza'))
        elegidos[a] = mejor
    return '\n'.join(lineas), {a: d for a, d in elegidos.items() if d > 0}


def tabla_asignacion(proyecto, datos, maximo, recurso='resources', duracion='duration'):
    """Semanas en que la demanda supera el límite: orden de prioridad, qué entra y qué se aplaza y por qué."""
    pasos, finales, _, _ = proyecto.copy().asignar_pasos(datos.copy(), duracion, recurso, maximo)
    filas = []
    for ps in pasos:
        if ps['total'] <= ps['limite']:
            continue
        t, L = ps['periodo'], int(ps['limite'])
        orden, aplazadas, acum = [], [], 0
        for a, f in ps['candidatas'].iterrows():
            r = int(f[t])
            orden.append(f'{a} (en curso, {r})' if f['Empezadas'] else f'{a} (H={int(f["H_total"])}, d={int(f[duracion])}, {r})')
            if a in ps['se_programan']:
                acum += r
            else:
                aplazadas.append(f'{a}: {acum} + {r} = {acum + r} > {L}')
        filas.append({'semana': t, 'límite': L, 'demanda': int(ps['total']),
                      'orden de prioridad (holgura, duración, recurso)': ' › '.join(orden),
                      'entran': lista_y(sorted(ps['se_programan'])) if ps['se_programan'] else '—',
                      'se aplazan una semana (no caben)': '; '.join(aplazadas)})
    return pd.DataFrame(filas).set_index('semana') if filas else Markdown('En ninguna semana la demanda supera el límite: no hay que aplazar nada.')


def primera_critica_aplazada(proyecto, datos, maximo, recurso='resources', duracion='duration'):
    """(semana, actividad) de la primera actividad sin holgura que la asignación tiene que aplazar, o None."""
    pasos, _, _, _ = proyecto.copy().asignar_pasos(datos.copy(), duracion, recurso, maximo)
    for ps in pasos:
        for a in ps['candidatas'].index:
            if a in ps['se_retrasan'] and float(ps['candidatas'].loc[a, 'H_total']) == 0:
                return ps['periodo'], a
    return None


def esperas_asignacion(datos_asignados):
    """{actividad: semanas de espera} a partir de las filas 💤 que añade la asignación."""
    e = datos_asignados[datos_asignados.index.str.startswith('💤')]
    return {i[1:]: int(f.duration) for i, f in e.iterrows()}

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con tiempos y camino crítico"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario: fechas y holgura total de cada actividad"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# En el Gantt cada actividad empieza en su inicio más temprano y la casilla lleva los operarios que ocupa; la fila *Total* suma la columna (carga de la semana) y la fila *Cuadrados* eleva cada carga al cuadrado (su suma, en la última columna, es la medida de desnivel que usa la nivelación).

# %%
#| fig-cap: "Diagrama de Gantt con las cargas de operarios (inicios más tempranos)"
#| label: fig-ganttInicial
proyecto_inicial = proyecto.copy()
gantt_inicial, dibujo = proyecto_inicial.gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas de operarios"
#| label: fig-cargasInicial
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=None)

# %%
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
semanas_pico = cargas.index[cargas == cargas.max()].tolist()
Markdown(f'Cargas semana a semana: {", ".join(str(int(c)) for c in cargas)} (total {int(cargas.sum())} operarios·semana en {len(cargas)} semanas, '
         f'media {cargas.sum():.0f} / {len(cargas)} = {cargas.mean():.2f} operarios/semana).\n\n'
         f'**Respuesta:** el máximo es de **{int(cargas.max())} operarios**, en {semanas_txt(semanas_pico)}.')

# %%
#| fig-cap: "Diagrama de Gantt con las cargas de máquinas"
#| label: fig-ganttMaquinas
gantt_maquinas, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'machines', report=False, tikz=True, params=GANTT_PARAMS)
pinta_tikz(dibujo)

# %%
cargas_maquinas = gantt_maquinas.data.loc['Total'].drop('H_total').astype(float)
pico_maq = cargas_maquinas.index[cargas_maquinas == cargas_maquinas.max()].tolist()
Markdown(f'**Respuesta:** hacen falta como máximo **{plural(int(cargas_maquinas.max()), "máquina")}**, en {semanas_txt(pico_maq)}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
holguras = proyecto.calendar(datos_proyecto['duration'])['H_total']
propuesta = {'H': 2, 'B': 1}
pd.DataFrame({'desplazamiento propuesto': pd.Series(propuesta), 'holgura total': holguras[list(propuesta)]})

# %%
#| fig-cap: "Diagrama de Gantt con la propuesta del jefe de obra"
#| label: fig-ganttPropuesta
suma_cuadrados_inicial = float((cargas ** 2).sum())
proyecto_propuesta = proyecto.copy()
suma_cuadrados_propuesta, _ = proyecto_propuesta.evaluar_desplazamiento(
    datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=False, **propuesta)
p_prop = proyecto.copy()
datos_prop, gantt_prop, dibujo = p_prop.desplazar(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True,
                                                  params=GANTT_PARAMS, **propuesta)
pinta_tikz(dibujo)

# %%
cargas_prop = gantt_prop.data.loc['Total'].drop('H_total').astype(float)
duracion_inicial = proyecto.duration(datos_proyecto['duration'])
duracion_prop = len(cargas_prop)
cada_una = all(propuesta[a] <= holguras[a] for a in propuesta)
respeta = cada_una and duracion_prop == duracion_inicial
comprobacion = '; '.join(f'{a}: {d} ≤ {int(holguras[a])}' if d <= holguras[a] else f'{a}: {d} > {int(holguras[a])}' for a, d in propuesta.items())
Markdown(f'''Cada desplazamiento frente a su holgura total: {comprobacion}. Aplicados los dos a la vez, el proyecto dura {duracion_prop} semanas (sin desplazar, {duracion_inicial}){": no se alarga" if duracion_prop == duracion_inicial else ": las dos actividades comparten holgura y juntas la superan"}.\n\nCargas con la propuesta: {explica_cuadrados(cargas_prop)} (antes {suma_cuadrados_inicial:.0f}); máximo {int(cargas_prop.max())} operarios (antes {int(cargas.max())}).\n\n**Respuesta:** la propuesta {"respeta" if respeta else "no respeta"} las holguras y la suma de cuadrados pasa de {suma_cuadrados_inicial:.0f} a {suma_cuadrados_propuesta:.0f}: {"mejora" if suma_cuadrados_propuesta < suma_cuadrados_inicial else "no mejora"} el nivelado.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Suma de cuadrados para cada desplazamiento posible en la nivelación"
#| label: tbl-cuadrados
proyecto_nivelado = proyecto.copy()
desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')
cuadrados

# %%
calendario_inicial = proyecto.calendar(datos_proyecto['duration'])
texto_niv, elegidos = explica_nivelacion(cuadrados, calendario_inicial)
Markdown('El algoritmo recorre las no críticas por fin más temprano y aplica a cada una el desplazamiento de menor suma de cuadrados:\n\n' + texto_niv)

# %%
#| fig-cap: "Diagrama de cargas de operarios tras la nivelación"
#| label: fig-cargasNivelado
plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=None)

# %%
cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)
suma_cuadrados_nivelado = float((cargas_niveladas ** 2).sum())
pd.DataFrame({'suma de cuadrados': [suma_cuadrados_inicial, suma_cuadrados_propuesta, suma_cuadrados_nivelado],
              'máximo de operarios': [cargas.max(), cargas_prop.max(), cargas_niveladas.max()]},
             index=['inicial', 'propuesta del jefe de obra', 'nivelación']).astype(int)

# %%
mejor = 'la nivelación' if suma_cuadrados_nivelado < suma_cuadrados_propuesta else ('la propuesta' if suma_cuadrados_propuesta < suma_cuadrados_nivelado else 'ninguna (empatan)')
Markdown(f'''**Respuesta:** es mejor {mejor}: suma de cuadrados {suma_cuadrados_nivelado:.0f} con la nivelación ({lista_y([f'{a} {plural(d, "semana")}' for a, d in elegidos.items()]) if elegidos else 'sin desplazamientos'}) frente a {suma_cuadrados_propuesta:.0f} con la propuesta y máximo de {int(cargas_niveladas.max())} frente a {int(cargas_prop.max())} operarios. El algoritmo prueba sistemáticamente todos los desplazamientos de cada actividad no crítica; aun así, al decidir actividad a actividad, tampoco garantiza el óptimo global.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| fig-cap: "Cargas de máquinas tras la nivelación de operarios"
#| label: fig-maquinasNivelado
datos_maquinas = datos_nivelados.copy()
datos_maquinas['machines'] = datos_proyecto['machines'].reindex(datos_maquinas.index).fillna(0).astype(int)
gantt_maquinas_niv, dibujo = proyecto_nivelado.copy().gantt_cargas(datos_maquinas, 'duration', 'machines', report=False, tikz=True, params=GANTT_PARAMS)
pinta_tikz(dibujo)

# %%
cargas_maquinas_niv = gantt_maquinas_niv.data.loc['Total'].drop('H_total').astype(float)
pd.DataFrame({'antes': [cargas_maquinas.max(), (cargas_maquinas ** 2).sum()],
              'después': [cargas_maquinas_niv.max(), (cargas_maquinas_niv ** 2).sum()]},
             index=['máximo de máquinas', 'suma de cuadrados (máquinas)']).astype(int)

# %%
q0, q1 = int((cargas_maquinas ** 2).sum()), int((cargas_maquinas_niv ** 2).sum())
M0, M1 = int(cargas_maquinas.max()), int(cargas_maquinas_niv.max())
efecto = ('empeora' if (q1 > q0 or M1 > M0) else ('mejora' if (q1 < q0 or M1 < M0) else 'no cambia'))
Markdown(f'''**Respuesta:** en este proyecto la carga de máquinas {efecto} con la nivelación de operarios (máximo {M0} → {M1}, suma de cuadrados {q0} → {q1}), {"pero no está garantizado" if efecto != "empeora" else "como efecto secundario"}: la nivelación optimiza un único recurso y los desplazamientos que allanan la demanda de operarios pueden concentrar la de máquinas. Sí, una nivelación puede empeorar el otro recurso; con varios recursos hay que nivelar con una función objetivo combinada o comprobar cada recurso tras nivelar el principal.''')
