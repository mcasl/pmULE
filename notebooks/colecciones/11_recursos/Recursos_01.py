# ---
# title: Recursos 01
# subtitle: "9 actividades, nivelación"
# coleccion: 11_recursos
# variante: nivelacion
# nivel: 1
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y número de operarios necesarios por actividad se adjunta. Todas las actividades se han programado en su inicio más temprano.

# %%
pregunta_1 = "1. Dibuje el diagrama de Gantt y el diagrama de cargas. ¿Cuál es el número máximo de operarios necesario y en qué semanas?  "
pregunta_2 = "2. ¿Está nivelado el consumo de recursos? Calcule la suma de cuadrados de las cargas como medida de desnivel.  "
pregunta_3 = "3. Nivele los recursos desplazando actividades no críticas dentro de su holgura, sin alargar el proyecto, con el algoritmo de mínimos cuadrados. Indique qué actividades desplaza y cuánto.  "
pregunta_4 = "4. Compare el diagrama de cargas antes y después: máximo de operarios y suma de cuadrados.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources
    ('A'     , '---'      , 7       , 3        ),
    ('B'     , '---'      , 1       , 1        ),
    ('C'     , '---'      , 8       , 2        ),
    ('D'     , 'B,C'      , 2       , 3        ),
    ('E'     , 'B,C'      , 7       , 4        ),
    ('F'     , 'E'        , 2       , 1        ),
    ('G'     , 'A,D,E'    , 2       , 1        ),
    ('H'     , 'D'        , 2       , 3        ),
    ('I'     , 'F,G,H'    , 3       , 4        ),
], columns=['activity', 'predecessor', 'duration', 'resources']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la producción de un cortometraje; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Guion definitivo',
    'B': 'Casting',
    'C': 'Localizaciones y permisos',
    'D': 'Diseño de vestuario y atrezo',
    'E': 'Plan de rodaje',
    'F': 'Ensayos con los actores',
    'G': 'Rodaje de exteriores',
    'H': 'Rodaje en plató',
    'I': 'Grabación de sonido directo',
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
#| fig-cap: "Diagrama de cargas de operarios"
#| label: fig-cargasInicial
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=None)

# %%
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
semanas_pico = cargas.index[cargas == cargas.max()].tolist()
Markdown(f'Cargas semana a semana: {", ".join(str(int(c)) for c in cargas)} (total {int(cargas.sum())} operarios·semana en {len(cargas)} semanas, '
         f'media {cargas.sum():.0f} / {len(cargas)} = {cargas.mean():.2f} operarios/semana).\n\n'
         f'**Respuesta:** el máximo es de **{int(cargas.max())} operarios**, en {semanas_txt(semanas_pico)}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# La suma de cuadrados de las cargas, $\sum_t c_t^2$, penaliza los picos: para una misma carga total es mínima cuando el perfil es plano (todas las semanas con la carga media). Es la función objetivo de la nivelación.

# %%
suma_cuadrados_inicial = float((cargas ** 2).sum())
plano = len(cargas) * cargas.mean() ** 2
Markdown(f'''{explica_cuadrados(cargas)} (agrupando las semanas de igual carga).\n\nSi los {int(cargas.sum())} operarios·semana se repartieran por igual en las {len(cargas)} semanas, cada una tendría {cargas.mean():.2f} y la suma sería {len(cargas)} · {cargas.mean():.2f}² = {plano:.0f}.\n\n**Respuesta:** no está nivelado: la carga va de {int(cargas.min())} a {int(cargas.max())} operarios y su suma de cuadrados es {suma_cuadrados_inicial:.0f}, {suma_cuadrados_inicial - plano:.0f} por encima de la de un perfil plano ({plano:.0f}).''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# El algoritmo recorre las actividades no críticas (por orden de fin más temprano) y, para cada una, prueba todos los desplazamientos dentro de la holgura que le queda, eligiendo el que minimiza la suma de cuadrados (a igualdad, el menor desplazamiento).

# %%
#| tbl-cap: "Actividades no críticas en el orden en que las recorre el algoritmo (fin más temprano creciente; a igualdad, más holgura primero)"
#| label: tbl-ordenNivelacion
no_criticas = calendario[calendario['H_total'] > 0].sort_values(['fin_mas_temprano', 'H_total'], ascending=[True, False])
no_criticas[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']].T

# %% [markdown]
# Para cada una de ellas se prueba cada desplazamiento posible (las columnas de la tabla siguiente), se recalculan las cargas y su suma de cuadrados, y se aplica el desplazamiento de suma mínima (resaltado) antes de pasar a la siguiente actividad; por eso el resultado depende del orden y la holgura disponible puede ser menor que la del calendario inicial.

# %%
#| tbl-cap: "Suma de cuadrados para cada desplazamiento posible"
#| label: tbl-cuadrados
proyecto_nivelado = proyecto.copy()
desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(
    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')
cuadrados

# %%
texto_niv, elegidos = explica_nivelacion(cuadrados, calendario)
Markdown(texto_niv + '\n\n**Respuesta:** ' + (('se desplazan ' + lista_y([f'{a} {plural(d, "semana")}' for a, d in elegidos.items()])
         + ', sin alargar el proyecto.') if elegidos else 'ningún desplazamiento mejora la suma de cuadrados.'))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| fig-cap: "Diagrama de Gantt tras la nivelación"
#| label: fig-ganttNivelado
gantt_nivelado, dibujo = proyecto_nivelado.gantt_cargas(datos_nivelados, 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas tras la nivelación"
#| label: fig-cargasNivelado
plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=None)

# %%
cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)
pd.DataFrame({'antes': [cargas.max(), (cargas ** 2).sum(), len(cargas)],
              'después': [cargas_niveladas.max(), (cargas_niveladas ** 2).sum(), len(cargas_niveladas)]},
             index=['máximo de operarios', 'suma de cuadrados', 'duración']).astype(int)

# %%
m0, m1 = int(cargas.max()), int(cargas_niveladas.max())
s0, s1 = int((cargas ** 2).sum()), int((cargas_niveladas ** 2).sum())
Markdown(f'''Después de nivelar: {explica_cuadrados(cargas_niveladas)}.\n\n**Respuesta:** la suma de cuadrados baja de {s0} a {s1} ({s0 - s1} menos) y el máximo de operarios {f"baja de {m0} a {m1}" if m1 < m0 else f"se mantiene en {m0}"}; la duración sigue siendo de {len(cargas_niveladas)} semanas porque solo se han movido actividades dentro de su holgura.{" El pico no baja: la nivelación minimiza la suma de cuadrados, no el máximo, y no ha encontrado desplazamientos que vacíen las semanas de carga máxima." if m1 == m0 else ""}''')
