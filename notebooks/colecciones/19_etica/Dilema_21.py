# ---
# title: Dilema 21
# subtitle: "arranque 2 semanas tarde por el cliente: SPI 0.85 contra la línea base y 1,00 contra la desplazada sin aprobar"
# coleccion: 19_etica
# variante: linea-base-reescrita
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
# El cuadro de prelaciones, las duraciones planificadas (semanas) y el presupuesto de cada actividad (BAC, k€) de un proyecto se adjuntan; el coste planificado se reparte uniformemente en la duración. La línea base aprobada por el cliente fija el inicio en la semana 0 y el fin en la semana 22.

# %%
#| tbl-cap: "Línea base aprobada"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 6       , 20 ),
    ('B'     , '---'      , 4       , 30 ),
    ('C'     , '---'      , 2       , 20 ),
    ('D'     , 'A,B,C'    , 1       , 70 ),
    ('E'     , 'A,C'      , 7       , 60 ),
    ('F'     , 'D,E'      , 6       , 60 ),
    ('G'     , 'D'        , 7       , 70 ),
    ('H'     , 'F'        , 3       , 70 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto arrancó 2 semanas tarde porque el cliente entregó los permisos con retraso; desde entonces todo avanza al ritmo previsto. Al final de la semana 12 el seguimiento es el siguiente (semanas trabajadas, coste incurrido y avance verificado).

# %%
#| tbl-cap: "Seguimiento al final de la semana 12"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 6              , 20, 100      ),
    ('B'     , 4              , 30, 100      ),
    ('C'     , 2              , 20, 100      ),
    ('D'     , 1              , 70, 100      ),
    ('E'     , 4              , 34, 57       ),
    ('F'     , 0              , 0 , 0        ),
    ('G'     , 3              , 30, 43       ),
    ('H'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %% [markdown]
# > **De: Jefe de proyecto**
# > El SPI sale mal por culpa del cliente. Vamos a desplazar la línea base 2 semanas —es lo que de verdad pasó— y el informe sale con SPI = 1. No hace falta abrir una solicitud de cambio: es un ajuste técnico, no un cambio de alcance.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C9 — Línea base y control de cambios.** La línea base sólo cambia mediante una solicitud de cambio aprobada; medir el avance contra una línea base reescrita sin aprobación es presentar un índice fuera de contexto (C3). *(cf. PMBOK 8, dominio de gobernanza (control integrado de cambios).)*
# - **C3 — Información fuera de contexto.** Presentar un dato cierto en un marco que lleva a una conclusión equivocada (un índice calculado sobre una línea base reescrita). *(cf. Código Ético del PMI, 5.3.1.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule PV, EV, AC, SV, SPI, CV y CPI en la semana 12 contra la línea base aprobada y la fecha de fin prevista.  "
pregunta_2 = "2. (A) Repita el cálculo contra la línea base desplazada 2 semanas. ¿Qué indicadores cambian y cuáles no? ¿Cambia la fecha de fin prevista?  "
pregunta_3 = "3. (B) Según el criterio de referencia, ¿puede el jefe de proyecto desplazar la línea base como propone? ¿Qué forma de engaño sería el informe con SPI = 1 y qué procedimiento haría legítimo el cambio?  "
pregunta_4 = "4. (C) ¿Qué informe envía y qué propone al cliente? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
semana_control, desplazamiento = 12, 2
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
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy de la línea base"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# **Notación.** BAC: presupuesto de cada actividad; PV (*planned value*): coste presupuestado del trabajo programado hasta la semana de control; EV (*earned value*): valor ganado, BAC × porcentaje realizado; AC (*actual cost*): coste real incurrido; CV = EV − AC (variación de coste) y SV = EV − PV (variación de plazo); CPI = EV / AC y SPI = EV / PV (índices de rendimiento del coste y del cronograma; menos de 1 es desfavorable); EAC = BAC / CPI (previsión del coste final si se mantiene el CPI) y VAC = BAC − EAC. PV se calcula con las fechas de la línea base: cada actividad reparte su BAC uniformemente entre su inicio más temprano y su fin.

# %%
#| tbl-cap: "PV, EV y AC contra la línea base aprobada"
#| label: tbl-detalleBase
detalle_base, ind_base = indicadores(seguimiento, calendario, semana_control)
detalle_base[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]

# %%
fin_previsto = duracion + desplazamiento
Markdown(f'''Totales en la semana {semana_control}:

- {explica_suma(sin_ceros(detalle_base['PV_i']), 'PV', 'k€', 1)}
- {explica_suma(sin_ceros(detalle_base['EV_i']), 'EV', 'k€', 1)}
- {explica_suma(sin_ceros(seguimiento['AC']), 'AC', 'k€', 0)}

{explica_indicadores(ind_base, titulo='**Contra la línea base aprobada:**')}

Fecha de fin prevista: el retraso es sólo de arranque ({desplazamiento} semanas) y desde entonces todo avanza al ritmo previsto, así que el proyecto terminará {desplazamiento} semanas después de lo planificado: {duracion} + {desplazamiento} = **semana {fin_previsto}**. (La previsión genérica duración / SPI = {duracion} / {ind_base["SPI"]:.3f} = {duracion / ind_base["SPI"]:.1f} semanas no sirve aquí: supone que el retraso seguirá creciendo al ritmo actual, y el enunciado dice que no.)

**Respuesta.** PV = {ind_base["PV"]:.1f}, EV = {ind_base["EV"]:.1f} y AC = {ind_base["AC"]:.1f} k€; SV = {ind_base["SV"]:.1f} k€ y SPI = {ind_base["SPI"]:.2f} (retraso); CV = {ind_base["CV"]:.1f} k€ y CPI = {ind_base["CPI"]:.2f} (el coste va según lo hecho); fin previsto en la semana {fin_previsto}, frente a la {duracion} de la línea base.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Desplazar la línea base equivale a sumar el desplazamiento a todas las fechas del calendario; en la semana de control, PV se calcula entonces con las semanas transcurridas desde el nuevo inicio.

# %%
#| tbl-cap: "PV, EV y AC contra la línea base desplazada"
#| label: tbl-detalleDesplazada
calendario_desplazado = calendario.copy()
for col in ['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio']:
    calendario_desplazado[col] = calendario[col] + desplazamiento
detalle_desp, ind_desp = indicadores(seguimiento, calendario_desplazado, semana_control)
detalle_desp[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]

# %%
#| tbl-cap: "Indicadores contra las dos líneas base"
#| label: tbl-comparacionLB
comparacion = pd.DataFrame({'línea base aprobada': ind_base, 'línea base desplazada': ind_desp})
comparacion.round(3)

# %%
Markdown(f'''- {explica_suma(sin_ceros(detalle_desp['PV_i']), 'PV (desplazada)', 'k€', 1)}

{explica_indicadores(ind_desp, titulo='**Contra la línea base desplazada:**')}

**Respuesta.** Sólo cambia PV ({ind_base["PV"]:.1f} → {ind_desp["PV"]:.1f} k€) y, con él, SV ({ind_base["SV"]:.1f} → {ind_desp["SV"]:.1f} k€) y SPI ({ind_base["SPI"]:.2f} → {ind_desp["SPI"]:.2f}). EV, AC, CV y CPI no cambian, porque no dependen de la línea base. La fecha de fin prevista **no cambia**: sigue siendo la semana {fin_previsto}. El retraso desaparece del informe, no del proyecto.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Criterio C9 (línea base y control de cambios).** La línea base es un compromiso aprobado por el cliente (fin en la semana {duracion}); sólo cambia mediante una solicitud de cambio aprobada, no por un «ajuste técnico» del jefe de proyecto. Que la causa del retraso sea del cliente (permisos entregados {desplazamiento} semanas tarde) no cambia el procedimiento: es precisamente el motivo de una solicitud de cambio (o de una reclamación) en la que el cliente reconoce esas {desplazamiento} semanas y aprueba el nuevo fin en la semana {fin_previsto}.

**Criterio C3 (información fuera de contexto).** Sin ese trámite, el informe con SPI = {ind_desp["SPI"]:.2f} da una cifra aritméticamente correcta contra una referencia que nadie ha aprobado, presentada como si fuera el plan: el cliente leería «en plazo» un proyecto que acabará en la semana {fin_previsto} y no en la {duracion}.

**Respuesta.** No puede desplazarla por su cuenta (C9). El informe con SPI = {ind_desp["SPI"]:.2f} sería **información fuera de contexto (C3)**. Lo haría legítimo una **solicitud de cambio aprobada** que mueva la línea base {desplazamiento} semanas; con ella, el mismo SPI ≈ 1 sería correcto.''')

# %%
preguntas_test = [
    ('Una línea base puede cambiarse…', ['mediante una solicitud de cambio aprobada', 'por decisión del jefe de proyecto si la causa es del cliente', 'nunca', 'siempre que no cambie el alcance'],
     'C9: la línea base sólo cambia por el control integrado de cambios.'),
    ('Al desplazar la línea base, el indicador que NO cambia es…', ['CPI', 'PV', 'SV', 'SPI'],
     'PV depende de la línea base; EV y AC no, así que CV y CPI se mantienen.'),
    ('Un informe con SPI = 1 contra una línea base desplazada sin aprobación es…', ['C3, información fuera de contexto', 'C1, afirmación falsa', 'correcto: el retraso era del cliente', 'C6, conflicto de interés'],
     'La cifra es correcta contra una referencia que no es el plan aprobado.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — línea base reescrita')
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
    [('Cliente', f'recibe «SPI = {ind_desp["SPI"]:.2f}» y planifica con la semana {duracion}, cuando el fin previsto es la {fin_previsto}'),
     ('Contratista', f'sin solicitud de cambio pierde la prueba de que las {desplazamiento} semanas de retraso son imputables al cliente (y la compensación de la espera)'),
     ('Jefe de proyecto', 'evita un informe en rojo este mes a costa de su credibilidad cuando la fecha real aparezca')],
    [('Desplazar la línea base sin aprobación e informar SPI ≈ 1 (propuesta)', 'no infringe una ley, pero sí el procedimiento contractual de cambios', 'no', 'no: el cliente no aceptaría que le cambien la referencia sin preguntar'),
     (f'Informar SPI {ind_base["SPI"]:.2f} contra la línea base aprobada, con la causa documentada', 'sí', 'sí', 'sí'),
     (f'Además, presentar una solicitud de cambio por causa del cliente: fin en la semana {fin_previsto}', 'sí', 'sí', 'sí')],
    f'''informe con SPI {ind_base["SPI"]:.2f} contra la línea base aprobada, la causa documentada (permisos entregados {desplazamiento} semanas tarde) y la fecha prevista, semana {fin_previsto}. Al cliente: solicitud de cambio que mueva la línea base {desplazamiento} semanas y, si procede, la compensación de los costes de la espera; una vez aprobada, los informes siguientes miden contra la nueva línea base.''')
