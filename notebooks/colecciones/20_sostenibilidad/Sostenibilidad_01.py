# ---
# title: Sostenibilidad 01
# subtitle: "adelantar 3 semanas: 11 k€ y 28 t frente a 14 k€ y 12 t con el carbono a 300 €/t"
# coleccion: 20_sostenibilidad
# variante: compresion-con-carbono
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
# El cliente exige adelantar la entrega de un proyecto de montaje industrial **3 semanas**. El cuadro de prelaciones, las duraciones (semanas), las duraciones mínimas alcanzables, el coste unitario de reducción (`ucr`, k€ por semana) y las **emisiones adicionales** de cada semana que se acorta (`emisiones_reduccion`, t CO₂ por semana: grupos electrógenos para el doble turno, transporte aéreo de componentes, hormigón de fraguado rápido) se adjuntan.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, ucr, emisiones_reduccion
    ('A'     , '---'      , 8       , 6           , 3  , 12                 ),
    ('B'     , '---'      , 4       , 1           , 5  , 8                  ),
    ('C'     , '---'      , 7       , 5           , 1  , 4                  ),
    ('D'     , 'A,B,C'    , 3       , 2           , 4  , 0                  ),
    ('E'     , 'B'        , 1       , 1           , 0  , 20                 ),
    ('F'     , 'E'        , 7       , 4           , 3  , 8                  ),
    ('G'     , 'C'        , 1       , 1           , 0  , 0                  ),
    ('H'     , 'D,G'      , 8       , 6           , 7  , 0                  ),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'ucr', 'emisiones_reduccion']).set_index('activity')
datos_enunciado

# %% [markdown]
# La organización tiene fijado en su política un **precio sombra del carbono de 300 €/t CO₂** para las decisiones de proyecto. En el mercado voluntario hay créditos de compensación a 8 €/t.

# %% [markdown]
# > **De: Director de operaciones**
# > Comprime por el método de siempre: Ackoff con los costes unitarios y listo, que es lo más barato. Las toneladas que salgan las compensamos con créditos a 8 €/t y el proyecto queda «neutro». El precio sombra es para la memoria de sostenibilidad, no para la obra.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 20 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **S2 — Externalidad internalizada.** Cuando la organización ha fijado un precio para una externalidad (precio sombra del carbono, canon, tasa), toda decisión de planificación —comprimir, comprar, diseñar— se compara por el coste total: coste del proyecto más el valor de la externalidad a ese precio. Con el precio fijado, la alternativa correcta es la de menor coste total, y comprar compensaciones a un precio inferior no cambia la cuenta. *(cf. PMBOK 8 §3.7 (estrategias para abordar las externalidades); LCSP art. 148.2 (externalidades medioambientales monetizables); precio sombra del carbono del BEI.)*
# - **S1 — Jerarquía de respuesta a los impactos (pirámide de sostenibilidad).** Ante un impacto negativo del proyecto las estrategias se ordenan de menos a más deseable: compensar o contrarrestar el daño, restaurarlo, minimizarlo y evitarlo. Compensar cuando se podía minimizar o evitar es quedarse en el escalón más bajo; clasificar una propuesta en su escalón es una pregunta con respuesta única. *(cf. PMBOK 8, principio «integrar la sostenibilidad en todas las áreas del proyecto» (§3.7, figura 3-7).)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Obtenga con el algoritmo de Ackoff-Sasieni el plan de reducción de 3 semanas de menor coste monetario (sin precio del carbono): actividades acortadas en cada paso, coste y emisiones adicionales del plan.  "
pregunta_2 = "2. (A) Repita el algoritmo con el coste unitario total (ucr + emisiones × 300 €/t) y compare los dos planes: coste monetario, emisiones y coste total al precio sombra. ¿Cuánto costaría compensar las emisiones de cada plan a 8 €/t y cuánto valen al precio sombra?  "
pregunta_3 = "3. (B) Según los criterios de referencia, ¿qué plan es el correcto y por qué el argumento «compensamos a 8 €/t» no cambia la respuesta? ¿En qué escalón de la pirámide está cada propuesta?  "
pregunta_4 = "4. (C) ¿Qué propone al director de operaciones y qué ofrece al cliente? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
semanas_pedidas, precio_sombra, precio_creditos = 3, 300, 8
def pasos_txt(pasos):
    '''«paso 1: A; paso 2: A y C»: actividades acortadas en cada paso de Ackoff-Sasieni.'''
    return '; '.join(f'paso {k + 1}: ' + lista_y(v) for k, v in sorted(pasos.items()))
def suma_txt(valores, decimales=0):
    '''«3 + 4 + 4 = 11»: suma término a término.'''
    vals = [float(v) for v in valores]
    return ' + '.join(f'{v:.{decimales}f}' for v in vals) + f' = {sum(vals):.{decimales}f}'
def suma_acts(acts, serie, decimales=0):
    '''«A(3) + A(3) + C(1) = 7»: suma de una columna sobre una lista de actividades (con repeticiones).'''
    vals = [float(serie[a]) for a in acts]
    return ' + '.join(f'{a}({v:.{decimales}f})' for a, v in zip(acts, vals)) + f' = {sum(vals):.{decimales}f}'
def criticas_txt(proyecto, duraciones):
    '''Caminos críticos en orden de ejecución: «A–C–F / B–D–F».'''
    cal = proyecto.calendar(duraciones)
    return ' / '.join('–'.join(ordena_ruta(list(r), cal)) for r in proyecto.critical_path(duraciones).values())
def miles(x, decimales=0):
    '''«25 400»: separador de miles con espacio.'''
    return f'{x:,.{decimales}f}'.replace(',', ' ')
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = int(proyecto.duration(datos_proyecto['duration']))
datos_proyecto['ucr_total'] = datos_proyecto['ucr'] + datos_proyecto['emisiones_reduccion'] * precio_sombra / 1000
def resumen_plan(pasos):
    acts = [a for paso in pasos.values() for a in paso]
    coste = float(datos_proyecto.loc[acts, 'ucr'].sum())
    emisiones = float(datos_proyecto.loc[acts, 'emisiones_reduccion'].sum())
    return pd.Series({'coste monetario (k€)': coste, 'emisiones adicionales (t CO₂)': emisiones,
                      'valor al precio sombra (k€)': emisiones * precio_sombra / 1000,
                      'coste total al precio sombra (k€)': coste + emisiones * precio_sombra / 1000,
                      'compensar a precio de créditos (k€)': emisiones * precio_creditos / 1000})

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
#| tbl-cap: "Ackoff-Sasieni con el coste unitario monetario"
#| label: tbl-ackoffDinero
tabla_1, pasos_1, duraciones_1, margen_1 = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'], reduction=semanas_pedidas, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=semanas_pedidas)

# %%
#| tbl-cap: "Opciones consideradas en cada paso (coste monetario)"
#| label: tbl-ackoffPaso1
detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos_1)

# %%
plan_1 = resumen_plan(pasos_1)
acts_1 = [a for _, paso in sorted(pasos_1.items()) for a in paso]
display(plan_1.round(2).to_frame('plan 1: coste monetario').T)
Markdown(f'''Plan 1 (por el coste monetario): {pasos_txt(pasos_1)}. El coste del plan es la suma de los costes unitarios de las semanas acortadas, y sus emisiones adicionales, la suma de las emisiones de esas semanas:\n\n- coste = {suma_acts(acts_1, datos_proyecto['ucr'], 1)} k€;\n- emisiones = {suma_acts(acts_1, datos_proyecto['emisiones_reduccion'])} t CO₂.\n\n**Respuesta:** el plan de menor coste monetario ({pasos_txt(pasos_1)}) cuesta **{plan_1.iloc[0]:.1f} k€** y añade **{plan_1.iloc[1]:.0f} t CO₂**.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Coste unitario total de las actividades que pueden acortarse"
#| label: tbl-ucrTotal
reducibles = datos_proyecto.index[datos_proyecto['cap_duration'] < datos_proyecto['duration']]
costes_unitarios = pd.DataFrame({'ucr (k€/semana)': datos_proyecto.loc[reducibles, 'ucr'],
                                 'emisiones (t/semana)': datos_proyecto.loc[reducibles, 'emisiones_reduccion'],
                                 'emisiones × precio sombra (k€/semana)': datos_proyecto.loc[reducibles, 'emisiones_reduccion'] * precio_sombra / 1000,
                                 'ucr total (k€/semana)': datos_proyecto.loc[reducibles, 'ucr_total']})
display(costes_unitarios.round(2))
_a = costes_unitarios['emisiones (t/semana)'].idxmax()
Markdown(f'''El coste unitario total de cada actividad es ucr + emisiones × precio sombra / 1000; por ejemplo, para {_a}: {datos_proyecto.loc[_a, 'ucr']:g} + {datos_proyecto.loc[_a, 'emisiones_reduccion']:g} × {precio_sombra} / 1000 = **{datos_proyecto.loc[_a, 'ucr_total']:.2f} k€** por semana. Con estos costes se repite el algoritmo (en la tabla de opciones los costes van redondeados a enteros; los exactos son los de esta tabla).''')

# %%
#| tbl-cap: "Ackoff-Sasieni con el coste unitario total (ucr + emisiones × precio sombra)"
#| label: tbl-ackoffTotal
tabla_2, pasos_2, duraciones_2, margen_2 = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr_total'], reduction=semanas_pedidas, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr_total'],
    reduction=semanas_pedidas)

# %%
#| tbl-cap: "Opciones consideradas en cada paso (coste total)"
#| label: tbl-ackoffPaso2
detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr_total'], pasos_2)

# %%
#| tbl-cap: "Comparación de los dos planes"
#| label: tbl-planes
plan_2 = resumen_plan(pasos_2)
comparacion = pd.DataFrame({'plan 1 (ucr)': plan_1, 'plan 2 (ucr + carbono)': plan_2})
display(comparacion.round(2))
acts_2 = [a for _, paso in sorted(pasos_2.items()) for a in paso]
def linea_plan(n, plan, acts):
    return (f'- Plan {n}: coste {suma_acts(acts, datos_proyecto["ucr"], 1)} k€; emisiones {suma_acts(acts, datos_proyecto["emisiones_reduccion"])} t; '
            f'coste total = {plan.iloc[0]:.1f} + {plan.iloc[1]:.0f} × {precio_sombra} / 1000 = {plan.iloc[0]:.1f} + {plan.iloc[2]:.1f} = **{plan.iloc[3]:.1f} k€**; '
            f'compensar a {precio_creditos} €/t: {plan.iloc[1]:.0f} × {precio_creditos} / 1000 = {plan.iloc[4]:.2f} k€.')
salen = sorted(set(acts_1) - set(acts_2)); entran = sorted(set(acts_2) - set(acts_1))
Markdown(f'''Plan 2 (por el coste total): {pasos_txt(pasos_2)}.\n\n''' + linea_plan(1, plan_1, acts_1) + '\n' + linea_plan(2, plan_2, acts_2) + f'''\n\nEl algoritmo es el mismo; cambia el coste unitario con el que se ordenan las opciones. Con el carbono dentro del coste, ''' + ((lista_y(salen) + (' deja' if len(salen) == 1 else ' dejan') + ' de elegirse' + (' y entra' + ('' if len(entran) == 1 else 'n') + ' ' + lista_y(entran) if entran else '')) if salen else 'cambia el orden de los pasos') + f'''.\n\n**Respuesta:** el plan 2 cuesta **{plan_2.iloc[0] - plan_1.iloc[0]:.1f} k€ más** en dinero y emite **{plan_1.iloc[1] - plan_2.iloc[1]:.0f} t menos**; al precio sombra es **{plan_1.iloc[3] - plan_2.iloc[3]:.1f} k€ más barato** en total ({plan_2.iloc[3]:.1f} frente a {plan_1.iloc[3]:.1f} k€). Compensar las emisiones costaría {plan_1.iloc[4]:.2f} k€ (plan 1) y {plan_2.iloc[4]:.2f} k€ (plan 2) a precio de créditos; al precio sombra valen {plan_1.iloc[2]:.1f} y {plan_2.iloc[2]:.1f} k€.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# **S2.** La organización ha fijado el precio de la externalidad (300 €/t), de modo que el criterio de decisión es el coste total: el plan 2 (17.6 k€) frente al plan 1 (19.4 k€). El argumento de los créditos a 8 €/t sustituye el precio que la organización ha decidido por el precio que a uno le conviene: si valiera, el precio sombra no serviría para nada. Comprar créditos no reduce las emisiones del proyecto; las compensa, y a un precio que no es el de la política. **S1.** El plan 1 más créditos está en el escalón más bajo de la pirámide (**compensar**); el plan 2 **minimiza** las emisiones de la compresión; **evitar** sería negociar con el cliente las 3 semanas (lección 19, C13 y C10) o acortar por medios sin emisiones (solapes, más personal en jornada ordinaria). La clasificación no es opinable: compensar es compensar.
#
# **Respuesta:** el plan correcto es el **plan 2** (acorta A, D y H), el de menor coste total al precio sombra; los créditos a 8 €/t no cambian la elección porque no son el precio que la organización ha fijado. Plan 1 con créditos: compensar; plan 2: minimizar.

# %%
preguntas_test = [
    ('Con un precio sombra del carbono fijado por la organización, el plan de compresión correcto es…', ['el de menor coste total (dinero + emisiones al precio sombra)', 'el de menor coste monetario, compensando después', 'el de menos emisiones a cualquier coste', 'el que pida el cliente'],
     'S2: la externalidad con precio entra en el coste de la decisión.'),
    ('Comprar créditos de compensación para las emisiones de un plan está, en la pirámide de la sostenibilidad del PMBOK 8, en el escalón…', ['más bajo: compensar', 'de minimizar', 'de restaurar', 'de evitar'],
     'La figura 3-7 ordena compensar < restaurar < minimizar < evitar.'),
    ('Incorporar el precio del carbono a Ackoff-Sasieni consiste en…', ['sumar al coste unitario de reducción las emisiones por semana multiplicadas por el precio', 'restar el precio de los créditos del coste unitario', 'eliminar las actividades con emisiones', 'cambiar el número de semanas pedidas'],
     'El algoritmo no cambia; cambia el coste con el que ordena las opciones.'),
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
gift = a_gift(preguntas_test, 'Sostenibilidad 20 — compresión con carbono')
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_sostenibilidad.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: (1) identificar a todos los afectados, incluidos los que no están en la mesa (la comunidad, los trabajadores de la cadena, quien opere o desmantele lo construido, las generaciones siguientes) y qué pierde o gana cada uno con las cifras del bloque A; (2) formular al menos dos alternativas además de la propuesta, situándolas en la pirámide (compensar, restaurar, minimizar, evitar); (3) someter cada alternativa a tres pruebas —legalidad (¿cumple la norma?), publicidad (¿la defendería ante el inspector, el cliente o la prensa?) y ciclo de vida (¿quién paga el impacto y cuándo?)—; (4) decidir y redactar la comunicación con las cifras. Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.

# %% [markdown]
# Al director de operaciones: el plan 2 con sus cifras (cuesta 3.0 k€ más y emite 16 t menos); el precio sombra es la política de la empresa y se aplica en la obra o no es una política. Al cliente: el precio de cada semana con el coste total, y la alternativa de no comprimir o comprimir menos. Prueba de publicidad: ¿pondría en la memoria de sostenibilidad que el proyecto se llamó «neutro» por 0.2 k€ de créditos mientras se elegía el plan que más emitía?
