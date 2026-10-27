# ---
# title: Recursos 18
# subtitle: "9 actividades, asignación con límite 5→8"
# coleccion: 11_recursos
# variante: asignacion
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y número de operarios necesarios por actividad se adjunta. La empresa sólo dispone de 5 operarios durante las 3 primeras semanas y de 8 a partir de entonces.

# %%
pregunta_1 = "1. Dibuje el diagrama de cargas con las actividades en su inicio más temprano y compruebe si se respeta el límite de operarios.  "
pregunta_2 = "2. Reprograme el proyecto con el algoritmo de asignación de Wiest-Levy para respetar el límite. ¿Qué actividades se desplazan y cuánto se alarga el proyecto?  "
pregunta_3 = "3. ¿Y si se dispusiera de 5 operarios durante todo el proyecto? Compare las dos duraciones.  "
pregunta_4 = "4. Explique por qué la asignación puede alargar el proyecto y la nivelación nunca.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources
    ('A'     , '---'      , 6       , 4        ),
    ('B'     , '---'      , 3       , 1        ),
    ('C'     , 'A'        , 2       , 2        ),
    ('D'     , 'A,B'      , 3       , 3        ),
    ('E'     , 'B,C'      , 4       , 2        ),
    ('F'     , 'B'        , 2       , 3        ),
    ('G'     , 'F'        , 2       , 3        ),
    ('H'     , 'E,G'      , 1       , 2        ),
    ('I'     , 'D,H'      , 1       , 2        ),
], columns=['activity', 'predecessor', 'duration', 'resources']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la instalación de una planta fotovoltaica sobre cubierta; las actividades son las siguientes (las duraciones del cuadro están en semanas).

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
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con tiempos y camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

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
limite_maximo_recursos = [5] * 3 + [8]

# %%
#| fig-cap: "Diagrama de cargas con el límite de operarios"
#| label: fig-cargasInicial
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite_maximo_recursos)

# %%
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
limite = pd.Series([limite_maximo_recursos[min(t - 1, len(limite_maximo_recursos) - 1)] for t in range(1, len(cargas) + 1)], index=cargas.index)
exceso = cargas[cargas > limite]
detalle = '; '.join(f'semana {int(t)}: {int(c)} > {int(limite[t])}' for t, c in exceso.items())
if len(exceso):
    resp = (f'no se respeta: la carga supera el límite en {semanas_txt(exceso.index)} ({detalle}); '
            f'con todo en su inicio más temprano harían falta hasta {int(cargas.max())} operarios.')
else:
    pico = int(cargas.idxmax())
    resp = (f'sí se respeta: la carga no supera el límite en ninguna semana. El pico, {int(cargas.max())} operarios en la semana {pico}, '
            f'cae ya en el tramo con límite {int(limite[pico])}, y en las semanas con límite {int(limite.min())} la carga no pasa de '
            f'{int(cargas[limite == limite.min()].max())}.')
Markdown('**Respuesta:** ' + resp)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# El algoritmo de asignación (Wiest-Levy en la librería) recorre el calendario semana a semana y aplaza lo que no cabe; si no basta con consumir holguras, aplaza también actividades críticas y el proyecto se alarga. Las filas 💤 del Gantt marcan las esperas introducidas.
#
# Regla del algoritmo: en cada semana se toman las actividades que el calendario vigente sitúa en ella y se ordenan así: primero las ya empezadas (no se interrumpen); después, menor holgura total (recalculada tras los aplazamientos anteriores) y, a igualdad, menor duración. Se admiten en ese orden mientras la suma de recursos no supere el límite; las que no caben se aplazan una semana (y con ellas sus sucesoras, si no tienen holgura). En la traza, *Total* es la demanda de esa semana antes de aplazar.

# %% [markdown]
# Traza semana a semana (qué se programa y qué se aplaza una semana):

# %%
_ = proyecto.copy().asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',
                           maximo=limite_maximo_recursos, report=True)

# %% [markdown]
# Por qué se aplaza cada actividad: en las semanas en que la demanda supera el límite, la tabla da el orden de prioridad (con su holgura, duración y operarios), las que entran y, para cada aplazada, la suma que no cabe.

# %%
#| tbl-cap: "Semanas con conflicto en la asignación"
#| label: tbl-conflictos
tabla_asignacion(proyecto, datos_proyecto, limite_maximo_recursos)

# %%
#| fig-cap: "Diagrama de Gantt tras la asignación"
#| label: fig-ganttAsignado
proyecto_asignado = proyecto.copy()
datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',
    maximo=limite_maximo_recursos, report=False, params=dict(period_width=1, **GANTT_PARAMS))
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas tras la asignación"
#| label: fig-cargasAsignado
plot_cargas(gantt_df=gantt_asignado, limite_maximo_recursos=limite_maximo_recursos)

# %%
duracion_inicial = proyecto.duration(datos_proyecto['duration'])
duracion_asignada = len(gantt_asignado.data.columns.drop('H_total'))
esperas = esperas_asignacion(datos_asignados)
cargas_asig = gantt_asignado.data.loc['Total'].drop('H_total').astype(float)
txt_esperas = lista_y([f"{a} {plural(e, 'semana')}" for a, e in esperas.items()]) if esperas else 'ninguna actividad'
if not esperas:
    resp = f'no hace falta desplazar ninguna actividad: el proyecto mantiene sus {duracion_inicial} semanas.'
elif duracion_asignada > duracion_inicial:
    resp = (f'se desplazan {txt_esperas}; el proyecto pasa de {duracion_inicial} a {duracion_asignada} semanas '
            f'(se alarga {plural(duracion_asignada - duracion_inicial, "semana")}) y el pico de operarios queda en {int(cargas_asig.max())}.')
else:
    resp = (f'se desplazan {txt_esperas}, dentro de su holgura: el proyecto mantiene sus {duracion_inicial} semanas '
            f'y el pico de operarios queda en {int(cargas_asig.max())}.')
Markdown('**Respuesta:** ' + resp)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
_ = proyecto.copy().asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',
                           maximo=[5], report=True)

# %%
#| tbl-cap: "Semanas con conflicto con el límite constante"
#| label: tbl-conflictosFijo
tabla_asignacion(proyecto, datos_proyecto, [5])

# %%
#| fig-cap: "Diagrama de Gantt con el límite constante"
#| label: fig-ganttLimiteFijo
limite_fijo = [5]
proyecto_fijo = proyecto.copy()
datos_fijo, gantt_fijo, dibujo = proyecto_fijo.asignar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',
    maximo=limite_fijo, report=False, params=dict(period_width=1, **GANTT_PARAMS))
pinta_tikz(dibujo)

# %%
duracion_fija = len(gantt_fijo.data.columns.drop('H_total'))
esperas_fijo = esperas_asignacion(datos_fijo)
txt_esperas_fijo = lista_y([f"{a} {plural(e, 'semana')}" for a, e in esperas_fijo.items()]) if esperas_fijo else 'ninguna'
Markdown(f'Esperas con el límite constante: {txt_esperas_fijo}.\n\n'
         f'**Respuesta:** con 5 operarios durante todo el proyecto la duración sería de {duracion_fija} semanas, frente a {duracion_asignada} con el refuerzo a 8 operarios a partir de la semana 4; '
         + (f'{plural(duracion_fija - duracion_asignada, "semana")} más ({duracion_fija - duracion_inicial} más que sin límite).' if duracion_fija > duracion_asignada else
            'la misma duración; en este proyecto el refuerzo no acorta el plazo.'))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
critica = primera_critica_aplazada(proyecto, datos_proyecto, limite_maximo_recursos)
motivo = (f'en la semana {critica[0]} la actividad {critica[1]}, sin holgura, no cabía y hubo que aplazarla' if critica else
          'hubo que aplazar actividades más allá de su holgura')
Markdown(f'''La nivelación sólo usa las holguras: desplaza actividades no críticas sin tocar el plazo, y si un desplazamiento alargara el proyecto no se prueba. La asignación impone una restricción dura (el límite de cada semana); si ni siquiera aplazando todas las actividades con holgura cabe la demanda, tiene que aplazar actividades críticas y el proyecto se alarga. **Respuesta:** {f'aquí la asignación ha alargado el proyecto de {duracion_inicial} a {duracion_asignada} semanas porque {motivo}' if duracion_asignada > duracion_inicial else ('aquí la asignación no ha necesitado alargar el proyecto: le ha bastado con consumir holguras' if esperas else 'aquí, con los límites del enunciado, no ha hecho falta aplazar nada')}; la nivelación, por construcción, nunca cambia la duración.''')
