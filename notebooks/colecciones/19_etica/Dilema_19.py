# ---
# title: Dilema 19
# subtitle: "4 operarios, pico 7: 110 horas extra por operario frente a 80 legales; con límite, 23 semanas"
# coleccion: 19_etica
# variante: horas-extra
# nivel: 2
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
# Una empresa de montajes ejecuta el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y operarios necesarios por actividad se adjunta. La plantilla disponible es de **4 operarios** a jornada completa (40 horas semanales); no hay otros proyectos en marcha.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources
    ('A'     , '---'      , 4       , 2        ),
    ('B'     , '---'      , 2       , 4        ),
    ('C'     , 'A,B'      , 5       , 4        ),
    ('D'     , 'C'        , 3       , 3        ),
    ('E'     , 'B'        , 6       , 1        ),
    ('F'     , 'C,E'      , 1       , 4        ),
    ('G'     , 'D'        , 5       , 1        ),
    ('H'     , 'F,G'      , 3       , 2        ),
], columns=['activity', 'predecessor', 'duration', 'resources']).set_index('activity')
datos_enunciado

# %% [markdown]
# > **De: Jefe de proyecto**
# > El cliente quiere las 20 semanas del plan con inicios más tempranos y no vamos a contratar a nadie. Las semanas en que hagan falta más de los que somos, se hacen horas extra y ya está: se pagan y punto.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C7 — Cumplimiento de la ley y de las normas de seguridad.** El proyecto no puede planificarse incumpliendo una norma legal o de seguridad aunque salga más barato o más rápido; una actividad exigida por una norma no es una opción de compresión. *(cf. Código Ético del PMI, 2.3.1 y 2.2.1; ICB4, Perspectiva 3 «cumplimiento, estándares y regulaciones»; PMBOK 8, principio de calidad integrada.)*
# - **Norma — jornada ordinaria máxima: 40 horas semanales de trabajo efectivo de promedio en cómputo anual** *(Estatuto de los Trabajadores, art. 34.1.)*
# - **Norma — máximo de horas extraordinarias: 80 por trabajador y año, sin contar las compensadas con descanso en los cuatro meses siguientes** *(Estatuto de los Trabajadores (RDL 2/2015), art. 35.2.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Dibuje el diagrama de cargas con inicios más tempranos y obtenga, semana a semana, el exceso de operarios sobre los 4 disponibles. Traduzca el exceso a horas extraordinarias por operario en el año (reparto uniforme entre la plantilla).  "
pregunta_2 = "2. (A) Calcule la duración del proyecto si se respeta el límite de 4 operarios (asignación con límite) y, alternativamente, cuántos operarios haría falta contratar para cumplir las semanas del plan sin horas extra.  "
pregunta_3 = "3. (B) Según el criterio y la norma de referencia, ¿puede ejecutarse el plan del jefe de proyecto? Justifique con la cifra.  "
pregunta_4 = "4. (C) ¿Qué propone al jefe de proyecto y al cliente? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
plantilla, jornada, maximo_horas_extra = 4, 40, 80
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = int(proyecto.duration(datos_proyecto['duration']))

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

def rangos(semanas):
    '''«3, 5–8 y 10»: semanas consecutivas agrupadas.'''
    semanas, grupos = sorted(int(s) for s in semanas), []
    for s in semanas:
        if grupos and s == grupos[-1][1] + 1:
            grupos[-1][1] = s
        else:
            grupos.append([s, s])
    return lista_y([f'{a}' if a == b else f'{a}–{b}' for a, b in grupos])

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Gantt con inicios más tempranos y carga de operarios"
#| label: fig-ganttCargas
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas frente a la plantilla disponible"
#| label: fig-cargas
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[plantilla])

# %% [markdown]
# La tabla siguiente da, semana a semana, los operarios necesarios con inicios más tempranos (fila «Total» del Gantt), el exceso sobre la plantilla, exceso = máx(0, necesarios − 4), las horas extra que ese exceso supone (exceso × 40 h) y su reparto uniforme entre los 4 operarios (horas extra / 4).

# %%
#| tbl-cap: "Exceso de operarios y horas extraordinarias semana a semana"
#| label: tbl-exceso
exceso = (cargas - plantilla).clip(lower=0)
tabla = pd.DataFrame({'operarios necesarios': cargas, 'exceso sobre la plantilla': exceso,
                      'horas extra de la semana': exceso * jornada, 'horas extra por operario': exceso * jornada / plantilla})
tabla.loc['Total'] = tabla.sum()
horas_persona = float(tabla.loc['Total', 'horas extra por operario'])
limpia_tabla(tabla, 1).T

# %%
semanas_exceso = [s for s in exceso.index if exceso[s] > 0]
op_semanas = float(exceso.sum())
horas_totales = op_semanas * jornada
semana_max = max(semanas_exceso, key=lambda s: exceso[s])
Markdown(f'''Pico de **{int(cargas.max())}** operarios frente a {plantilla}; hay exceso en {plural(len(semanas_exceso), 'semana')} ({'semana' if len(semanas_exceso) == 1 else 'semanas'} {rangos(semanas_exceso)}).

- Exceso total: {' + '.join(str(int(exceso[s])) for s in semanas_exceso)} = **{op_semanas:.0f} operario-semanas**.
- Horas extra: {op_semanas:.0f} × {jornada} h = **{horas_totales:.0f} h**.
- Por operario (reparto uniforme): {horas_totales:.0f} / {plantilla} = **{horas_persona:.0f} h** en el año (el proyecto dura {duracion} semanas, menos de un año).
- En la peor semana (la {semana_max}) cada operario trabajaría {jornada} + {exceso[semana_max] * jornada / plantilla:.0f} = {jornada + exceso[semana_max] * jornada / plantilla:.0f} h.

**Respuesta.** El plan exige **{horas_persona:.0f} horas extraordinarias por operario** en el año, {'más que' if horas_persona > maximo_horas_extra else 'dentro de'} las {maximo_horas_extra} que permite la ley.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
proyecto_asignado = proyecto.copy()
datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(data=datos_proyecto.copy(), duration_label='duration',
                                                                  resource_label='resources', maximo=[plantilla], report=False, params=GANTT_PARAMS)
duracion_con_limite = len(gantt_asignado.data.columns.drop('H_total'))
contratar = int(cargas.max()) - plantilla
por_nivel = {int(k): [s for s in semanas_exceso if exceso[s] == k] for k in sorted(set(exceso[semanas_exceso]))}
horas_legales = maximo_horas_extra * plantilla
op_sem_legales = horas_legales / jornada
Markdown(f'''**Con el límite de {plantilla} operarios** (asignación: cada semana entran las actividades por orden de prioridad mientras caben en la plantilla; las demás esperan) el proyecto dura **{duracion_con_limite}** semanas, {plural(duracion_con_limite - duracion, 'semana')} más que las {duracion} del plan (Gantt siguiente).

**Para cumplir las {duracion} semanas sin horas extra** hay que cubrir el exceso con personal contratado: como máximo {int(cargas.max())} − {plantilla} = **{contratar}** operarios a la vez, y no todas las semanas: {'; '.join(f'{plural(k, "operario")} {"en la semana" if len(v) == 1 else "en las semanas"} {rangos(v)}' for k, v in por_nivel.items())}, en total {op_semanas:.0f} operario-semanas de refuerzo.

Combinación legal: las {maximo_horas_extra} h por operario permiten {maximo_horas_extra} × {plantilla} = {horas_legales} h = {op_sem_legales:.0f} operario-semanas en horas extra; el resto, {max(op_semanas - op_sem_legales, 0):.0f} operario-semanas, con refuerzo.

**Respuesta.** Con el límite de {plantilla} operarios el proyecto dura {duracion_con_limite} semanas; para cumplir {duracion} sin horas extra hay que contratar {plural(contratar, 'operario')} en el pico ({op_semanas:.0f} operario-semanas en total).''')

# %%
#| fig-cap: "Gantt con asignación limitada a la plantilla"
#| label: fig-ganttAsignado
pinta_tikz(dibujo)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Criterio C7 y norma.** El Estatuto de los Trabajadores limita las horas extraordinarias a {maximo_horas_extra} por trabajador y año (art. 35.2), sin contar las compensadas con descanso, y la jornada ordinaria a {jornada} horas semanales de promedio (art. 34.1). El plan del jefe de proyecto exige {horas_persona:.0f} h por operario, {horas_persona - maximo_horas_extra:+.0f} h sobre el máximo, con semanas de {jornada + exceso[semana_max] * jornada / plantilla:.0f} h. Según C7, un plan que incumple una norma no es una opción de planificación, por mucho que las horas «se paguen»: pagar no convierte en legal lo que la ley limita.

**Respuesta.** {'**No puede ejecutarse**: ' + f'{horas_persona:.0f} h > {maximo_horas_extra} h por operario y año, salvo que el exceso se compense con descanso (lo que alargaría el proyecto igualmente).' if horas_persona > maximo_horas_extra else 'Puede ejecutarse legalmente: ' + f'{horas_persona:.0f} h ≤ {maximo_horas_extra}; la pregunta pasa al bloque C (si conviene).'}''')

# %%
preguntas_test = [
    ('El máximo de horas extraordinarias por trabajador y año fijado por el Estatuto de los Trabajadores es…', ['80', '40', '120', 'no hay máximo si se pagan'],
     'Art. 35.2 ET: 80 horas al año, sin contar las compensadas con descanso.'),
    ('Un plan que exige más horas extra de las legales…', ['no es una opción: se descarta por C7', 'es válido si el cliente lo acepta', 'es válido si se pagan', 'depende del jefe de proyecto'],
     'El cumplimiento legal es condición previa, no un coste más en la tabla.'),
    ('Para cumplir el plazo sin horas extra la alternativa objetiva es…', ['contratar operarios para las semanas de pico o aceptar la duración con límite', 'reducir la calidad', 'no hacer el diagrama de cargas', 'cambiar las prelaciones'],
     'El diagrama de cargas da el pico y la asignación con límite da la nueva duración.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — horas extra')
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
    [('Operarios', f'{horas_persona:.0f} h extra cada uno y semanas de hasta {jornada + exceso[semana_max] * jornada / plantilla:.0f} h: salud, seguridad y descanso'),
     ('Cliente', f'quiere {duracion} semanas; con la plantilla serían {duracion_con_limite}'),
     ('Empresa', f'ahorra la contratación de {op_semanas:.0f} operario-semanas a cambio de un incumplimiento sancionable y de un riesgo de accidentes'),
     ('Jefe de proyecto', 'cumple el plazo «sobre el papel» con un plan que no puede firmar')],
    [(f'Horas extra sin límite: {horas_persona:.0f} h por operario (propuesta)', f'no: supera las {maximo_horas_extra} h del art. 35.2', 'no', 'no: ningún operario aceptaría esas semanas si pudiera elegir'),
     (f'Aceptar {duracion_con_limite} semanas con la plantilla', 'sí', 'sí', 'sí, si el cliente lo acepta informado'),
     (f'Contratar el refuerzo: {op_semanas:.0f} operario-semanas (hasta {contratar} a la vez)', 'sí', 'sí', 'sí'),
     (f'Combinar: {op_sem_legales:.0f} operario-semanas en horas extra (≤ {maximo_horas_extra} h por operario) y el resto con refuerzo', 'sí', 'sí', 'sí')],
    f'''al jefe de proyecto, la tabla de opciones con su coste y su fecha ({duracion_con_limite} semanas con la plantilla; {duracion} contratando {op_semanas:.0f} operario-semanas; o la combinación dentro de las {maximo_horas_extra} h); al cliente, la opción elegida con su precio y su fecha. Lo que no se le cuenta es que el plazo «se consigue» incumpliendo la ley, porque no está en la tabla.''')
