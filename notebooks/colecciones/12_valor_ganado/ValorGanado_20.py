# ---
# title: ValorGanado 20
# subtitle: "límite de 3 operarios y línea base sin recursos"
# coleccion: 12_valor_ganado
# variante: ev-recursos
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
# Considere el proyecto cuyo cuadro de prelaciones, duración (semanas), operarios necesarios y presupuesto (BAC, k€) de cada actividad se adjunta; el coste se reparte uniformemente en cada actividad. La línea base se preparó con el calendario más temprano, sin mirar los recursos, pero la empresa dispone como máximo de **3 operarios** por semana: las actividades se ejecutan con el algoritmo de asignación (cada una empieza en cuanto puede), sin que nadie trabaje más lento de lo previsto. El coste real coincide con el presupuesto.

# %%
pregunta_1 = "1. Calcule la línea base (duración y PV semana a semana) y el calendario real con el límite de operarios.  "
pregunta_2 = "2. Calcule EV, SPI y la duración prevista T/SPI semana a semana. ¿Desde qué semana cae el SPI por debajo de 1 y qué actividad es la primera en esperar?  "
pregunta_3 = "3. En la semana 7 (mitad de la línea base) ¿qué duración prevé el SPI? Compárela con la duración real. ¿Qué previsión es la acertada y por qué?  "
pregunta_4 = "4. ¿Qué debería haber hecho el equipo al preparar la línea base? Calcule el SPI semana a semana con la línea base que respeta el límite de operarios y explique la diferencia.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, operarios, BAC
    ('A'     , '---'      , 1       , 3        , 125),
    ('B'     , '---'      , 5       , 1        , 50 ),
    ('C'     , '---'      , 5       , 1        , 100),
    ('D'     , 'B,C'      , 2       , 3        , 50 ),
    ('E'     , 'A,B'      , 4       , 2        , 75 ),
    ('F'     , 'C'        , 3       , 1        , 125),
    ('G'     , 'E'        , 3       , 3        , 175),
    ('H'     , 'D,F,G'    , 2       , 3        , 175),
], columns=['activity', 'predecessor', 'duration', 'operarios', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
limite = [3]
t_control = 7
BAC = datos_proyecto['BAC']

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| fig-cap: "Grafo PERT de la línea base (sin recursos) con el camino crítico"
#| label: fig-pert
proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)

# %%
#| tbl-cap: "Calendario de la línea base (inicios más tempranos, sin mirar los operarios)"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]

# %%
#| fig-cap: "Carga de operarios de la línea base frente al límite"
#| label: fig-cargasInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite)

# %% [markdown]
# La ejecución real sigue el algoritmo de asignación: semana a semana, cada actividad disponible (predecesoras terminadas) empieza si caben sus operarios durante toda su duración; si no, espera. Traza semana a semana y Gantt resultante (filas 💤: esperas):

# %%
res = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt de la ejecución real con el límite de operarios"
#| label: fig-ganttReal
pinta_tikz(res.dibujos['operarios'])

# %%
#| tbl-cap: "Inicio de cada actividad en la línea base y en la ejecución real"
#| label: tbl-inicios
pd.DataFrame({'inicio línea base': calendario['inicio_mas_temprano'].astype(int), 'inicio real': res.inicios, 'fin real': res.fines,
              'espera (semanas)': res.esperas}).fillna(0).astype(int).T

# %% [markdown]
# El PV de cada semana es la suma, sobre las actividades, del BAC por la fracción planificada transcurrida (línea base); el EV, lo mismo con el calendario real (nadie trabaja más despacio, así que la fracción ejecutada es la real del calendario con recursos).

# %%
tabla = proyecto.valor_ganado_recursos(datos_proyecto, 'duration', ['operarios'], limite, BAC)
T_plan, T_real = tabla.attrs['T_plan'], tabla.attrs['T_real']
Markdown(f'Línea base {T_plan:.0f} semanas; con {limite[0]} operarios el calendario real dura {T_real:.0f}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "PV, EV, SPI y duración prevista semana a semana"
#| label: tbl-valorGanado
tabla.round(3)

# %%
primera = tabla.index[tabla['SPI'] < 0.999].min()
esperas = res.traza[res.traza.decision == 'espera'].sort_values('periodo')
primera_actividad = esperas.iloc[0]['actividad']
Markdown(f'El SPI cae por debajo de 1 en la semana {primera}; la primera actividad que espera es {primera_actividad} '
         f'(semana {int(esperas.iloc[0]["periodo"])}).')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
fila = tabla.loc[t_control]
Markdown(f'En la semana {t_control}: SPI = {fila["SPI"]:.3f}, duración prevista T/SPI = {fila["duración prevista"]:.1f} semanas; '
         f'duración real {T_real:.0f}: el SPI {"sobrestima" if fila["duración prevista"] > T_real else "subestima"} el retraso. '
         'Supone que la fracción del plan que se logra cada semana se mantiene hasta el final, cuando el límite de recursos '
         'solo frena algunos tramos del proyecto.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
tabla2 = tabla[['PV', 'EV']].copy()
tabla2['PV con recursos'] = tabla['EV']        # línea base replanificada con el límite
tabla2['SPI con línea base realista'] = tabla2['EV'] / tabla2['PV con recursos']
display(tabla2.round(3))
Markdown('Con la línea base calculada respetando el límite de operarios el SPI es 1 en todas las semanas: los problemas de recursos '
         'se detectan al planificar (con la asignación), no al controlar. Un SPI < 1 desde la primera espera es síntoma de una '
         'línea base irrealista, no de un equipo lento.')
