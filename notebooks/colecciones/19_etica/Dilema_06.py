# ---
# title: Dilema 06
# subtitle: "adelantar 2 semanas: 6 k€ por Ackoff frente a «quitar I» (coste esperado del defecto 20 k€)"
# coleccion: 19_etica
# variante: inspeccion-suprimida
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
# Una instalación industrial termina con las **pruebas de presión y la inspección reglamentaria** (actividad I), exigidas por la normativa de seguridad antes de la puesta en servicio. El cuadro de prelaciones, las duraciones (semanas), las duraciones mínimas alcanzables y el coste unitario de reducción (k€ por semana) se adjuntan; I no se puede acortar.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, ucr   
    ('A'     , '---'      , 2       , 2           , 0     ),
    ('B'     , '---'      , 4       , 1           , 2     ),
    ('C'     , 'A,B'      , 3       , 2           , 6     ),
    ('D'     , 'A'        , 2       , 2           , 0     ),
    ('E'     , 'A'        , 3       , 1           , 5     ),
    ('F'     , 'D'        , 7       , 7           , 0     ),
    ('G'     , 'C,D,E'    , 3       , 1           , 1     ),
    ('H'     , 'E'        , 8       , 6           , 3     ),
    ('I'     , 'F,G,H'    , 2       , 2           , np.inf),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# El cliente exige adelantar la puesta en servicio **2 semanas**. En una reunión, el director de operaciones propone lo siguiente.

# %% [markdown]
# > **De: Director de operaciones**
# > Lo más barato es quitar I: son 2 semanas a coste cero. Las pruebas las hacemos «en marcha» el primer mes de servicio, que en la práctica es lo mismo, y el inspector firma después. Lo otro es pagar reducciones.

# %% [markdown]
# Según el responsable de calidad, con probabilidad 10% la inspección detecta un defecto que, si se pusiera en servicio sin corregir, costaría 200 k€ (parada, reparación y daños).

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C7 — Cumplimiento de la ley y de las normas de seguridad.** El proyecto no puede planificarse incumpliendo una norma legal o de seguridad aunque salga más barato o más rápido; una actividad exigida por una norma no es una opción de compresión. *(cf. Código Ético del PMI, 2.3.1 y 2.2.1; ICB4, Perspectiva 3 «cumplimiento, estándares y regulaciones»; PMBOK 8, principio de calidad integrada.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Obtenga con el algoritmo de Ackoff-Sasieni el coste mínimo de adelantar 2 semanas acortando actividades (sin tocar la inspección) y las actividades que se acortan en cada paso.  "
pregunta_2 = "2. (A) Calcule el coste esperado de poner en servicio sin la inspección (probabilidad del defecto por su coste) y compárelo con el coste de la reducción legítima. ¿Qué «ahorra» de verdad la propuesta?  "
pregunta_3 = "3. (B) Según el criterio de referencia, ¿es la supresión de la inspección una opción de compresión? ¿Qué distingue comprimir el cronograma de recortar el alcance?  "
pregunta_4 = "4. (C) ¿Qué responde al director de operaciones y qué ofrece al cliente? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
inspeccion, semanas_pedidas, p_defecto, coste_defecto = 'I', 2, 0.1, 200
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

# %% [markdown]
# El algoritmo de Ackoff-Sasieni acorta el proyecto semana a semana: en cada paso busca, entre las combinaciones de actividades que acortan a la vez **todas** las rutas críticas, la de menor coste (suma de los costes unitarios de reducción, `ucr`), respetando las duraciones mínimas (`cap_duration`). La inspección I tiene `ucr` infinito: no puede elegirse nunca.

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'], reduction=semanas_pedidas, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=semanas_pedidas)

# %%
#| tbl-cap: "Opciones consideradas en cada paso"
#| label: tbl-ackoffPaso
detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)

# %%
coste_reduccion = float(sum(datos_proyecto.loc[acts, 'ucr'].sum() for acts in pasos.values()))
duracion_reducida = int(proyecto.duration(duraciones_reducidas))
costes_paso = [float(datos_proyecto.loc[acts, 'ucr'].sum()) for acts in pasos.values()]
lineas = [f'- Paso {i}: se acorta {lista_y(acts)} una semana, coste ' + (' + '.join(f"{datos_proyecto.loc[a, 'ucr']:g}" for a in acts) + ' = ' if len(acts) > 1 else '')
          + f'{c:g} k€; el proyecto pasa a {duracion - i} semanas.' for i, (acts, c) in enumerate(zip(pasos.values(), costes_paso), 1)]
Markdown('\n'.join(lineas) + f'''\n\n**Respuesta.** Adelantar {semanas_pedidas} semanas sin tocar {inspeccion} cuesta {' + '.join(f'{c:g}' for c in costes_paso)} = **{coste_reduccion:g} k€** (de {duracion} a {duracion_reducida} semanas).''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
coste_esperado_sin_inspeccion = p_defecto * coste_defecto
d_insp = int(datos_proyecto.loc[inspeccion, 'duration'])
Markdown(f'''Coste esperado de poner en servicio sin la inspección = P(defecto) × coste del defecto = {p_defecto} × {coste_defecto} = **{coste_esperado_sin_inspeccion:g} k€**, frente a los {coste_reduccion:g} k€ de la reducción legítima ({coste_esperado_sin_inspeccion / coste_reduccion:.1f} veces más).

**Respuesta.** La propuesta «ahorra» los {coste_reduccion:g} k€ de la reducción {'(y gana ' + plural(d_insp, 'semana') + ', las ' + str(semanas_pedidas) + ' pedidas)' if d_insp == semanas_pedidas else '(y gana ' + plural(d_insp, 'semana') + ' en lugar de las ' + str(semanas_pedidas) + ' pedidas)'}, pero no los ahorra de verdad: los cambia por un coste esperado de {coste_esperado_sin_inspeccion:g} k€, incierto y que no paga quien decide sino otros (usuarios, cliente, aseguradora). {'En valor esperado es peor negocio: ' + f'{coste_esperado_sin_inspeccion:g} > {coste_reduccion:g} k€.' if coste_esperado_sin_inspeccion > coste_reduccion else 'Aunque en valor esperado saliera más barato, el criterio del apartado 3 no depende de esta cuenta.'}''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Criterio C7 (cumplimiento de la ley y de las normas de seguridad).** El enunciado dice que {inspeccion} (pruebas de presión e inspección reglamentaria) está exigida por la normativa de seguridad **antes** de la puesta en servicio. Según C7, una actividad exigida por una norma no es una opción de compresión: por eso en la tabla de Ackoff su coste de reducción es infinito, y no «cero» como sugiere el director de operaciones. Hacer las pruebas «en marcha» el primer mes no es lo mismo: incumple el «antes», y el inspector que firma después firma algo que no ha visto.

Comprimir el cronograma es hacer el **mismo alcance** en menos tiempo pagando por ello (acortar duraciones con más recursos, solapar actividades), como en el apartado 1: {coste_reduccion:g} k€ por {semanas_pedidas} semanas. Suprimir {inspeccion} es **recortar el alcance**; y como {inspeccion} es un requisito legal, es además incumplir la norma.

**Respuesta.** No: la supresión de {inspeccion} no es una opción de compresión (C7). La respuesta no depende del cálculo del apartado 2; ese cálculo sólo muestra que, además, la propuesta {'es peor negocio' if coste_esperado_sin_inspeccion > coste_reduccion else 'no aporta un ahorro que la justifique'}.''')

# %%
preguntas_test = [
    ('Suprimir una actividad exigida por una norma para acortar el proyecto es…', ['recortar el alcance e incumplir la norma (C7)', 'una opción de compresión más', 'legítimo si el cliente lo pide', 'una reducción a coste cero'],
     'Comprimir es hacer el mismo alcance en menos tiempo; esto cambia el alcance y viola la norma.'),
    ('En el algoritmo de Ackoff-Sasieni una actividad que no puede acortarse se modela con…', ['coste unitario de reducción infinito', 'coste cero', 'duración mínima cero', 'una ficticia'],
     'Infinito la saca de las opciones; cero la convertiría en la primera elegida.'),
    ('El coste esperado de poner en servicio sin inspección se calcula como…', ['probabilidad del defecto por su coste', 'coste del defecto', 'coste de la inspección', 'semanas ahorradas por el CUR'],
     'Es un valor esperado: p · K.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — inspección suprimida')
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
    [('Cliente', f'quiere adelantar {semanas_pedidas} semanas; sin la inspección asume, sin saberlo, un {porc(p_defecto, 0)} de un defecto de {coste_defecto} k€'),
     ('Usuarios y terceros', 'soportan el riesgo de seguridad de una instalación a presión sin probar'),
     ('Inspector', 'se le pide firmar algo que no ha visto'),
     ('Empresa', f'«ahorra» {coste_reduccion:g} k€ y se expone a la sanción, a la responsabilidad por el defecto y a perder el cliente')],
    [(f'Suprimir {inspeccion} y probar «en marcha» (propuesta)', 'no: la norma exige la inspección antes de la puesta en servicio', 'no', 'no: el usuario no aceptaría la instalación sin probar'),
     (f'Acortar {semanas_pedidas} semanas con Ackoff por {coste_reduccion:g} k€', 'sí', 'sí', 'sí'),
     ('Adelantar menos semanas por menos dinero (la curva coste-semanas)', 'sí', 'sí', 'sí'),
     (f'Negociar con el cliente la fecha o el reparto del coste de {coste_reduccion:g} k€', 'sí', 'sí', 'sí')],
    f'''al director de operaciones: la inspección no está en la tabla de opciones; la tabla real es «{semanas_pedidas} semanas por {coste_reduccion:g} k€» acortando {lista_y(sorted(set(a for acts in pasos.values() for a in acts)))}, o menos semanas por menos dinero. Al cliente: la oferta de reducción con su precio; la puesta en servicio sigue exigiendo la inspección. Prueba de publicidad: ¿explicaría al inspector, o a los usuarios, que las pruebas se hicieron después de arrancar?''')
