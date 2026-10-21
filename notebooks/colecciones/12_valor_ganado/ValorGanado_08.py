# ---
# title: ValorGanado 08
# subtitle: "7 actividades; perfiles de desembolso no uniformes; control en la semana 8"
# coleccion: 12_valor_ganado
# variante: coste-no-uniforme
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas), presupuesto (BAC, k€) y *perfil de desembolso* se adjunta. En las actividades de perfil `uniforme` el coste se reparte por igual entre sus semanas; en las de perfil `inicio` la mitad del BAC se desembolsa en la primera semana (compra de materiales) y el resto por igual entre las demás; en las de perfil `final` la mitad se desembolsa en la última semana (pago a la entrega) y el resto por igual entre las anteriores. Al final de la semana 8 se ha anotado el seguimiento adjunto (semanas trabajadas, coste real en k€ y porcentaje realizado).

# %%
pregunta_1 = "1. Calcule el calendario de inicios más tempranos y, para cada actividad, el coste planificado de cada semana según su perfil. Compruebe que la suma de cada fila es su BAC.  "
pregunta_2 = "2. Calcule el PV de cada actividad y el total en la semana 8. Compare con el PV que resultaría suponiendo todos los perfiles uniformes.  "
pregunta_3 = "3. Calcule EV y AC, y los índices CPI y SPI con el PV correcto y con el PV uniforme. ¿Cambia el diagnóstico?  "
pregunta_4 = "4. ¿Por qué el EV no depende del perfil de desembolso y el PV sí? ¿Qué perfil hace parecer «adelantado» un proyecto que va en plazo?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC, profile   
    ('A'     , '---'      , 7       , 60 , 'uniforme'),
    ('B'     , '---'      , 2       , 60 , 'uniforme'),
    ('C'     , '---'      , 6       , 20 , 'uniforme'),
    ('D'     , 'A,B,C'    , 4       , 60 , 'inicio'  ),
    ('E'     , 'B,C'      , 5       , 70 , 'final'   ),
    ('F'     , 'C'        , 4       , 40 , 'uniforme'),
    ('G'     , 'E,F'      , 5       , 20 , 'final'   ),
], columns=['activity', 'predecessor', 'duration', 'BAC', 'profile']).set_index('activity')
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
}, name='Descripción')
descripciones.to_frame()

# %%
#| tbl-cap: "Seguimiento en la semana de control"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 8              , 58, 80       ),
    ('B'     , 2              , 66, 100      ),
    ('C'     , 8              , 20, 100      ),
    ('D'     , 0              , 0 , 0        ),
    ('E'     , 0              , 0 , 0        ),
    ('F'     , 0              , 0 , 0        ),
    ('G'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
semana_control = 8
BAC_total = float(datos_proyecto['BAC'].sum())

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT de la planificación"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario de inicios más tempranos"
#| label: tbl-calendario
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'duracion']]

# %% [markdown]
# Con el inicio más temprano de cada actividad se reparte su BAC semana a semana según su perfil; la fila *suma* comprueba que se recupera el BAC.

# %%
def perfil_semanal(BAC, d, perfil):
    d = int(d)
    if perfil == 'uniforme' or d == 1:
        return [BAC / d] * d
    resto = [BAC / 2 / (d - 1)] * (d - 1)
    return [BAC / 2] + resto if perfil == 'inicio' else resto + [BAC / 2]
duracion_planificada = proyecto.duration(datos_proyecto['duration'])
coste_semanal = pd.DataFrame(0.0, index=datos_proyecto.index, columns=range(1, duracion_planificada + 1))
for a, fila in datos_proyecto.iterrows():
    es = int(calendario.loc[a, 'inicio_mas_temprano'])
    for j, c in enumerate(perfil_semanal(float(fila['BAC']), fila['duration'], fila['profile'])):
        coste_semanal.loc[a, es + j + 1] = c
coste_semanal['suma'] = coste_semanal.sum(axis=1)
coste_semanal.round(1)

# %%
Markdown(f'Cada fila suma su BAC: {bool(np.allclose(coste_semanal["suma"], datos_proyecto["BAC"]))}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
semanas = [c for c in coste_semanal.columns if c != 'suma' and c <= semana_control]
PV_i = coste_semanal[semanas].sum(axis=1)
frac = ((semana_control - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion'])
PV_uniforme_i = datos_proyecto['BAC'] * frac
comparacion = pd.DataFrame({'perfil': datos_proyecto['profile'], 'PV según perfil': PV_i, 'PV uniforme': PV_uniforme_i})
comparacion.loc['Total'] = ['', PV_i.sum(), PV_uniforme_i.sum()]
comparacion.round(1)

# %%
PV, PV_uniforme = float(PV_i.sum()), float(PV_uniforme_i.sum())
Markdown(f'PV = {PV:.1f} k€ con los perfiles reales frente a {PV_uniforme:.1f} con reparto uniforme; la diferencia está sólo en las actividades en curso con perfil no uniforme.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
EV = float((datos_proyecto['BAC'] * seguimiento['performed'] / 100).sum())
AC = float(seguimiento['AC'].sum())
indices = pd.DataFrame({'PV según perfil': [PV, EV, AC, EV / AC, EV / PV], 'PV uniforme': [PV_uniforme, EV, AC, EV / AC, EV / PV_uniforme]},
                       index=['PV', 'EV', 'AC', 'CPI = EV / AC', 'SPI = EV / PV']).round(3)
indices

# %%
s1, s2 = EV / PV, EV / PV_uniforme
Markdown(f'''SPI = {EV:.1f} / {PV:.1f} = {s1:.3f} con el perfil correcto y {EV:.1f} / {PV_uniforme:.1f} = {s2:.3f} con el uniforme: {'el diagnóstico de plazo cambia de signo' if (s1 < 1) != (s2 < 1) else 'el signo del diagnóstico no cambia, pero sí su magnitud'}. El CPI no cambia: no depende del PV.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# El EV es el BAC por el porcentaje *físico* realizado: mide trabajo hecho, no dinero desembolsado, y por eso no depende de cuándo se planeó pagar. El PV sí es dinero planificado hasta la fecha. Si una actividad en plazo tiene perfil `final` (el gran pago al terminar), su PV a mitad de camino es pequeño y el EV (proporcional al avance físico) lo supera: el proyecto parece adelantado sin estarlo. Con perfil `inicio` ocurre lo contrario. Para que el SPI mida plazo, el PV debe seguir el mismo criterio que el EV.
