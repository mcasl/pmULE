# ---
# title: Dilema 01
# subtitle: "informe con avance = gastado/presupuesto: SPI 1.05 y CPI 1,00 frente a 0.89 y 0.85 verificados"
# coleccion: 19_etica
# variante: informe-maquillado
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
# Usted acaba de incorporarse al equipo de un proyecto como responsable de control. El cuadro de prelaciones, las duraciones planificadas (semanas) y el presupuesto de cada actividad (BAC, k€) se adjuntan; el coste planificado se reparte uniformemente en la duración de cada actividad. El plazo es de 20 semanas.

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 5       , 40 ),
    ('B'     , '---'      , 2       , 30 ),
    ('C'     , '---'      , 5       , 40 ),
    ('D'     , 'B'        , 7       , 70 ),
    ('E'     , 'A,C'      , 7       , 60 ),
    ('F'     , 'D'        , 4       , 50 ),
    ('G'     , 'C'        , 3       , 30 ),
    ('H'     , 'D,G'      , 7       , 60 ),
    ('I'     , 'E,F,H'    , 4       , 20 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# Al final de la semana 10 el jefe de proyecto prepara el informe mensual para el comité. Su método para el porcentaje realizado de cada actividad es dividir el coste incurrido entre el presupuesto (columna `performed_informe`). Usted ha recorrido el proyecto con los responsables y ha anotado el avance físico verificado (columna `performed_verificado`).

# %%
#| tbl-cap: "Seguimiento al final de la semana 10: semanas trabajadas, coste incurrido (k€) y dos versiones del avance"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed_informe, performed_verificado
    ('A'     , 5              , 40, 100              , 100                 ),
    ('B'     , 2              , 30, 100              , 100                 ),
    ('C'     , 5              , 40, 100              , 100                 ),
    ('D'     , 7              , 70, 100              , 100                 ),
    ('E'     , 5              , 54, 90               , 36                  ),
    ('F'     , 1              , 14, 28               , 15                  ),
    ('G'     , 3              , 30, 100              , 100                 ),
    ('H'     , 1              , 11, 18               , 9                   ),
    ('I'     , 0              , 0 , 0                , 0                   ),
], columns=['activity', 'actual_duration', 'AC', 'performed_informe', 'performed_verificado']).set_index('activity')
seguimiento

# %% [markdown]
# > **De: Jefe de proyecto**
# > El informe va con mi método de siempre: avance = gastado / presupuesto. Así los índices salen razonables y no hay que discutir porcentajes con nadie. Sé que lo que hay en obra va algo más lento, pero el comité sólo mira el SPI y el CPI, y con tu versión nos ponen en rojo.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C1 — Afirmación falsa.** Comunicar como cierto un dato que quien lo comunica sabe que no lo es (un porcentaje de avance, una causa, una oferta que no existe). *(cf. Código Ético del PMI, estándar obligatorio 5.3.1; ICB4, Personas 2 «integridad personal y fiabilidad».)*
# - **C3 — Información fuera de contexto.** Presentar un dato cierto en un marco que lleva a una conclusión equivocada (un índice calculado sobre una línea base reescrita). *(cf. Código Ético del PMI, 5.3.1.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule PV, EV, AC, CV, SV, CPI y SPI en la semana de control con la versión del informe y con la verificada, actividad a actividad y en total, y la previsión de coste final EAC = BAC / CPI con cada una.  "
pregunta_2 = "2. (A) Explique por qué con el método del informe el CPI vale exactamente 1 sea cual sea el sobrecoste, y qué magnitud mide en realidad ese «porcentaje realizado».  "
pregunta_3 = "3. (B) ¿Qué criterio de referencia incumple el informe tal como lo quiere enviar el jefe de proyecto y qué forma de engaño es? ¿Qué cifras tendría que contener el informe para cumplir el criterio?  "
pregunta_4 = "4. (C) Usted firma el informe junto al jefe de proyecto. ¿Qué hace y qué dice a cada parte (jefe de proyecto, comité)? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
semana_control, plazo = 10, 20
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = int(proyecto.duration(datos_proyecto['duration']))
BAC_total = float(datos_proyecto['BAC'].sum())
def indicadores(seguimiento, calendario_plan, semana):
    datos_vg = pd.concat([datos_proyecto, seguimiento], axis=1)
    detalle = detalle_valor_ganado(datos_vg, calendario_plan, semana)
    PV, EV, AC = float(detalle['PV_i'].sum()), float(detalle['EV_i'].sum()), float(detalle['AC'].sum())
    return detalle, pd.Series({'PV': PV, 'EV': EV, 'AC': AC, 'CV': EV - AC, 'SV': EV - PV, 'CPI': EV / AC, 'SPI': EV / PV})

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
#| fig-cap: "Grafo PERT de la planificación"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# **Notación.** BAC: presupuesto de cada actividad; PV (*planned value*): coste presupuestado del trabajo programado hasta la semana de control; EV (*earned value*): valor ganado, BAC × porcentaje realizado; AC (*actual cost*): coste real incurrido; CV = EV − AC (variación de coste) y SV = EV − PV (variación de plazo); CPI = EV / AC y SPI = EV / PV (índices de rendimiento del coste y del cronograma; menos de 1 es desfavorable); EAC = BAC / CPI (previsión del coste final si se mantiene el CPI) y VAC = BAC − EAC. PV y AC no dependen del método de medir el avance; EV sí, y por eso se calcula dos veces: con la columna `performed_informe` y con `performed_verificado`.

# %%
#| tbl-cap: "PV, EV y AC actividad a actividad con el avance del informe (avance = AC / BAC)"
#| label: tbl-detalleInforme
seg_informe = seguimiento[['actual_duration', 'AC']].assign(performed=seguimiento['performed_informe'])
seg_verificado = seguimiento[['actual_duration', 'AC']].assign(performed=seguimiento['performed_verificado'])
detalle_inf, ind_inf = indicadores(seg_informe, calendario, semana_control)
detalle_ver, ind_ver = indicadores(seg_verificado, calendario, semana_control)
detalle_inf[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]

# %%
#| tbl-cap: "PV, EV y AC con el avance físico verificado"
#| label: tbl-detalleVerificado
detalle_ver[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]

# %%
PV_txt = explica_suma(sin_ceros(detalle_ver['PV_i']), 'PV', 'k€', 1)
EVi_txt = explica_suma(sin_ceros(detalle_inf['EV_i']), 'EV (informe)', 'k€', 1)
EVv_txt = explica_suma(sin_ceros(detalle_ver['EV_i']), 'EV (verificado)', 'k€', 1)
AC_txt = explica_suma(sin_ceros(seguimiento['AC']), 'AC', 'k€', 0)
Markdown(f'''Totales en la semana {semana_control} (sumas de las columnas de las tablas):

- {PV_txt}
- {AC_txt}
- {EVi_txt}
- {EVv_txt}

{explica_indicadores(ind_inf, BAC_total, '**Con el avance del informe:**')}

{explica_indicadores(ind_ver, BAC_total, '**Con el avance verificado:**')}''')

# %%
#| tbl-cap: "Indicadores y previsión de coste final con las dos versiones"
#| label: tbl-indicadores
comparacion = pd.DataFrame({'informe (avance = AC / BAC)': ind_inf, 'verificado': ind_ver})
comparacion.loc['EAC = BAC / CPI'] = [BAC_total / ind_inf['CPI'], BAC_total / ind_ver['CPI']]
comparacion.loc['VAC = BAC − EAC'] = BAC_total - comparacion.loc['EAC = BAC / CPI']
comparacion.round(3)

# %%
en_curso = [a for a in seguimiento.index if 0 < seguimiento.loc[a, 'performed_verificado'] < 100]
sobregasto = [a for a in en_curso if seguimiento.loc[a, 'AC'] > detalle_ver.loc[a, 'EV_i'] + 0.05]
eac_v = BAC_total / ind_ver['CPI']
Markdown(f'''**Respuesta.** Con el informe, SPI = {ind_inf["SPI"]:.2f} y CPI = {ind_inf["CPI"]:.3f} (prácticamente 1: el proyecto parece en plazo y en presupuesto) y EAC = {BAC_total / ind_inf["CPI"]:.1f} k€. Con el avance verificado, SPI = {ind_ver["SPI"]:.2f} y CPI = {ind_ver["CPI"]:.2f} ({'retrasado' if ind_ver['SPI'] < 1 else 'en plazo'} y {'con sobrecoste' if ind_ver['CPI'] < 1 else 'sin sobrecoste'}), y el coste final previsto pasa de {BAC_total:.0f} a **{eac_v:.1f} k€** ({eac_v - BAC_total:.1f} k€ de desviación, VAC = {BAC_total - eac_v:.1f} k€). PV y AC son iguales en las dos versiones; sólo cambia EV, en {ind_inf["EV"] - ind_ver["EV"]:.1f} k€, y la diferencia procede de las actividades en curso {lista_y(sobregasto)}, que han gastado más de lo que han avanzado.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
redondeo = [a for a in seguimiento.index if abs(detalle_inf.loc[a, 'EV_i'] - seguimiento.loc[a, 'AC']) > 0.05]
x = max(en_curso, key=lambda a: detalle_inf.loc[a, 'EV_i'] - detalle_ver.loc[a, 'EV_i'])
bac_x, ac_x = float(datos_proyecto.loc[x, 'BAC']), float(seguimiento.loc[x, 'AC'])
texto_redondeo = (f'''En los datos el CPI del informe sale {ind_inf["CPI"]:.3f} y no exactamente 1 sólo porque el porcentaje se redondea a un entero ({', '.join(f"{a}: {seguimiento.loc[a, 'AC']:.0f}/{datos_proyecto.loc[a, 'BAC']:.0f} = {100 * seguimiento.loc[a, 'AC'] / datos_proyecto.loc[a, 'BAC']:.1f} % → {seguimiento.loc[a, 'performed_informe']} %" for a in redondeo)}).''' if redondeo else 'En los datos, EV = AC actividad a actividad y el CPI del informe es exactamente 1.')
Markdown(f'''Si el porcentaje realizado se define como AC / BAC, para cada actividad EV = BAC · (AC / BAC) = AC; sumando, EV = AC y CPI = EV / AC = 1, se haya gastado lo que se haya gastado. {texto_redondeo}

Por ejemplo, {x}: el informe declara {seguimiento.loc[x, "performed_informe"]} % = {ac_x:.0f} / {bac_x:.0f}, luego EV = {bac_x:.0f} · {seguimiento.loc[x, "performed_informe"] / 100:.2f} = {detalle_inf.loc[x, "EV_i"]:.1f} ≈ AC = {ac_x:.0f}; el avance verificado es {seguimiento.loc[x, "performed_verificado"]} %, es decir, EV = {detalle_ver.loc[x, "EV_i"]:.1f} k€.

Por la misma razón, el SPI del informe es EV / PV ≈ AC / PV = {ind_inf["AC"]:.1f} / {ind_inf["PV"]:.1f} = {ind_inf["AC"] / ind_inf["PV"]:.3f}: compara lo gastado con lo planificado, y una actividad que gasta de más aparece como adelantada.

**Respuesta.** El CPI del método del informe vale 1 por construcción (EV = AC). Ese «porcentaje realizado» no mide avance físico: mide **consumo de presupuesto** (AC / BAC). El método anula las dos señales que el valor ganado existe para dar: el sobrecoste (CPI {ind_ver["CPI"]:.2f} verificado) y el retraso (SPI {ind_ver["SPI"]:.2f} verificado).''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Criterio C1 (afirmación falsa).** El jefe de proyecto sabe que el avance real es menor («sé que lo que hay en obra va algo más lento») y que con el avance verificado el SPI es {ind_ver["SPI"]:.2f} y el CPI {ind_ver["CPI"]:.2f} («con tu versión nos ponen en rojo»). Aun así, quiere comunicar al comité un valor ganado de {ind_inf["EV"]:.1f} k€ cuando el verificado es {ind_ver["EV"]:.1f} k€. No es una estimación discutible: el método se elige porque da el resultado deseado.

**Criterio C3 (información fuera de contexto).** Los índices del informe salen de los datos («ciertos» aritméticamente), pero se presentan como índices de valor ganado cuando el método no mide avance; el comité, que «sólo mira el SPI y el CPI», concluiría que el proyecto va en plazo y en presupuesto.

**Respuesta.** El informe tal como se quiere enviar incumple **C1** (comunica un avance que se sabe falso) y **C3** (índices en un marco que lleva a una conclusión equivocada). Para cumplir los criterios debe contener los indicadores con el avance verificado: EV {ind_ver["EV"]:.1f} k€, SPI {ind_ver["SPI"]:.2f}, CPI {ind_ver["CPI"]:.2f}, EAC {eac_v:.1f} k€ y VAC {BAC_total - eac_v:.1f} k€; la versión del gasto sólo puede aparecer con su nombre real, «consumo de presupuesto» ({ind_inf["AC"]:.0f} de {BAC_total:.0f} k€).''')

# %%
preguntas_test = [
    ('Con el método «avance = gastado / presupuesto», el CPI vale…', ['1 por construcción (salvo redondeo)', 'menos de 1 si hay sobrecoste', 'más de 1 si hay sobrecoste', 'lo mismo que con el avance físico'],
     'EV = BAC·(AC/BAC) = AC, luego CPI = EV/AC = 1 sea cual sea el gasto.'),
    ('Enviar al comité un avance que se sabe superior al real es, según los criterios de la lección…', ['C1, afirmación falsa', 'C4, retención de información', 'C8, reserva declarada', 'ningún criterio: es una estimación'],
     'Es un dato que quien lo comunica sabe falso; no una omisión ni una estimación.'),
    ('¿Qué indicador cambia entre las dos versiones del informe?', ['sólo EV (y con él CV, SV, CPI, SPI y EAC)', 'sólo AC', 'PV y AC', 'ninguno'],
     'PV depende del plan y AC de lo gastado; el porcentaje de avance sólo entra en EV.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — informe maquillado')
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
    [('Comité', f'decide con SPI {ind_inf["SPI"]:.2f} y CPI {ind_inf["CPI"]:.2f} en lugar de {ind_ver["SPI"]:.2f} y {ind_ver["CPI"]:.2f}; no puede reaccionar a un sobrecoste previsto de {eac_v - BAC_total:.1f} k€'),
     ('Jefe de proyecto', 'evita «ponerse en rojo» este mes, a cambio de un informe que se desmonta en cuanto alguien mire las actividades en curso'),
     ('Usted (firma el informe)', 'su firma avala un EV que sabe falso'),
     ('Responsables de ' + lista_y(sobregasto), 'sin el aviso pierden el tiempo de corregir el sobregasto mientras aún hay margen')],
    [('Firmar el informe con avance = AC / BAC (propuesta)', 'no hay norma legal en juego', 'no: el comité no aceptaría el método si lo conociera', 'no: usted no querría decidir con ese informe'),
     ('Enviar el avance verificado con nota de método y plan de recuperación', 'sí', 'sí', 'sí'),
     ('Enviar las dos versiones, cada una con su nombre (avance físico y consumo de presupuesto)', 'sí', 'sí, si el comité ve cuál mide avance', 'sí'),
     ('No firmar y elevar la discrepancia por escrito si el jefe de proyecto se niega', 'sí', 'sí', 'sí, aunque tensa la relación')],
    f'''proponer al jefe de proyecto enviar el avance verificado con una nota de método (quién lo midió y cómo) y un plan de recuperación para {lista_y(sobregasto)}; si se niega, no firmar, dejar constancia escrita de la discrepancia ({ind_inf["EV"] - ind_ver["EV"]:.1f} k€ de valor declarado y no ejecutado) y elevarla a quien corresponda en la organización. Al comité: SPI {ind_ver["SPI"]:.2f}, CPI {ind_ver["CPI"]:.2f}, EAC {eac_v:.1f} k€ y las medidas. Lo que no cabe: firmar y «ya lo corregiremos el mes que viene».''')
