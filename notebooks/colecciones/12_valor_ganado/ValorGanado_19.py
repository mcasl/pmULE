# ---
# title: ValorGanado 19
# subtitle: "duraciones triangulares y costes inciertos; controles en las semanas [6, 12, 17]"
# coleccion: 12_valor_ganado
# variante: prevision-incertidumbre
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
# Considere el proyecto cuyo cuadro de prelaciones, estimaciones de duración (mínima *a*, más probable *m* y máxima *b*, semanas, distribución triangular) y presupuesto (BAC, k€) se adjunta. Se toma como línea base el calendario más temprano con las duraciones más probables *m*; el coste se reparte uniformemente en cada actividad. Los costes reales de cada actividad pueden desviarse: factor multiplicativo triangular (0,9; 1,0; 1,4).

# %%
pregunta_1 = "1. Calcule la línea base: duración prevista y valor planificado (PV) en cada fecha de control.  "
pregunta_2 = "2. Simule 3 000 ejecuciones y evalúe SPI y CPI en las semanas 6, 12 y 17. Describa la distribución de ambos índices en cada fecha.  "
pregunta_3 = "3. En cada fecha, prevea la duración con T/SPI y el coste con BAC/CPI y compárelos con lo que ocurre de verdad en cada ejecución: sesgo, error absoluto medio y correlación. ¿Cómo cambia la calidad de la previsión con el avance?  "
pregunta_4 = "4. Un jefe de proyecto activa una alarma si SPI < 0.97. En la fecha intermedia, ¿qué probabilidad de terminar con retraso hay con y sin alarma? ¿Qué proporción de proyectos con retraso avisa la alarma (sensibilidad) y qué proporción de alarmas son falsas?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, a, m, b , BAC
    ('A'     , '---'      , 1       , 1, 1, 1 , 150),
    ('B'     , '---'      , 2       , 1, 1, 4 , 200),
    ('C'     , '---'      , 6       , 3, 6, 9 , 75 ),
    ('D'     , 'A,B,C'    , 2       , 1, 1, 4 , 150),
    ('E'     , 'A'        , 4       , 1, 5, 6 , 50 ),
    ('F'     , 'C'        , 2       , 1, 2, 3 , 100),
    ('G'     , 'D,E'      , 7       , 5, 6, 10, 50 ),
    ('H'     , 'F,G'      , 3       , 2, 2, 5 , 175),
    ('I'     , 'H'        , 5       , 2, 5, 8 , 150),
], columns=['activity', 'predecessor', 'duration', 'a', 'm', 'b', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la instalación de una planta fotovoltaica sobre cubierta; las actividades son las siguientes (las duraciones del cuadro están en días).

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
puntos = [6, 12, 17]
umbral = 0.97
dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}
factor_coste = {a: ('triangular', 0.9, 1.0, 1.4) for a in datos_proyecto.index}
plan = datos_proyecto['m']

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT de la línea base (duraciones más probables) con el camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(plan)

# %%
#| tbl-cap: "Calendario de la línea base"
#| label: tbl-calendario
calendario = proyecto.calendar(plan)
T_plan = proyecto.duration(plan)
proyecto.display_calendario(plan)

# %% [markdown]
# El valor planificado de cada actividad en una fecha es su BAC por la fracción de su duración que, según el calendario, debería haberse ejecutado; el PV del proyecto es la suma.

# %%
#| tbl-cap: "Valor planificado de cada actividad en cada fecha de control (k€)"
#| label: tbl-pvFechas
pv_fechas = pd.DataFrame({f'semana {t}': datos_proyecto['BAC'] * ((t - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion']) for t in puntos})
pv_fechas.loc['Total (PV)'] = pv_fechas.sum()
pv_fechas.round(1)

# %%
Markdown(f'Línea base: {T_plan} semanas, BAC = {datos_proyecto["BAC"].sum():.0f} k€; PV en las fechas de control: '
         + ', '.join(f'semana {t}: {v:.1f} k€' for t, v in zip(puntos, pv_fechas.loc["Total (PV)"])) + '.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
resultados = {t: proyecto.monte_carlo_valor_ganado(dists, plan, datos_proyecto['BAC'], t, n_samples=3000, seed=11, factor_coste=factor_coste)
              for t in puntos}
pd.concat({f'semana {t}': r.summary().round(3) for t, r in resultados.items()})

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Calidad de las previsiones en cada fecha de control"
#| label: tbl-calidad
pd.concat({f'semana {t}': r.calidad_prevision().round(3) for t, r in resultados.items()})

# %%
mae = {t: r.calidad_prevision().loc['duración', 'error absoluto medio'] for t, r in resultados.items()}
Markdown('El error absoluto medio de la previsión de duración pasa de ' + ' a '.join(f'{v:.2f}' for v in mae.values()) +
         ' semanas al avanzar el proyecto: cuanto más hecho, menos queda por incertidumbre; y T/SPI es sesgado al principio, '
         'porque el avance temprano incluye la parte aún no afectada por los retrasos de las actividades futuras.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
r = resultados[puntos[1]]
retraso = r.tau_samples > r.T_plan
alarma = r.spi < umbral
p_con, p_sin = retraso[alarma].mean(), retraso[~alarma].mean()
sens = alarma[retraso].mean()
falsas = (~retraso[alarma]).mean()
Markdown(f'P(retraso | alarma) = {p_con:.3f}; P(retraso | sin alarma) = {p_sin:.3f}; sensibilidad = {sens:.3f}; '
         f'alarmas falsas = {falsas:.3f}. La alarma discrimina, pero no es infalible: el SPI en una fecha es una señal con ruido.')
