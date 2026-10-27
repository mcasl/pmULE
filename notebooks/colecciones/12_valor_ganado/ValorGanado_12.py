# ---
# title: ValorGanado 12
# subtitle: "8 actividades, tres puntos de control (semanas 5, 9, 13)"
# coleccion: 12_valor_ganado
# variante: tendencia
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por actividad (BAC, en k€) se adjunta. El coste planificado de cada actividad se reparte uniformemente a lo largo de su duración.

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 2       , 60 ),
    ('B'     , '---'      , 7       , 50 ),
    ('C'     , 'A,B'      , 1       , 30 ),
    ('D'     , 'A'        , 6       , 30 ),
    ('E'     , 'A'        , 1       , 20 ),
    ('F'     , 'C,E'      , 7       , 40 ),
    ('G'     , 'C,D,E'    , 2       , 20 ),
    ('H'     , 'F,G'      , 1       , 40 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la rehabilitación de una nave industrial para un taller; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Levantamiento topográfico y mediciones',
    'B': 'Proyecto de ejecución y licencia',
    'C': 'Desmontaje de instalaciones antiguas',
    'D': 'Demolición de tabiquería interior',
    'E': 'Refuerzo de la estructura metálica',
    'F': 'Reparación de la cubierta',
    'G': 'Solera y canalizaciones',
    'H': 'Cerramientos y carpintería exterior',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# Se han realizado tres puntos de control, al final de las semanas 5, 9 y 13. En cada uno se anotó, para cada actividad, las semanas trabajadas hasta ese momento, el coste real incurrido (k€) y el porcentaje realizado.

# %%
#| tbl-cap: "Punto de control 1 (final de la semana 5)"
#| label: tbl-seguimiento1
seguimiento_1 = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 4              , 78, 100      ),
    ('B'     , 5              , 30, 50       ),
    ('C'     , 0              , 0 , 0        ),
    ('D'     , 1              , 4 , 17       ),
    ('E'     , 1              , 26, 100      ),
    ('F'     , 0              , 0 , 0        ),
    ('G'     , 0              , 0 , 0        ),
    ('H'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento_1

# %%
#| tbl-cap: "Punto de control 2 (final de la semana 9)"
#| label: tbl-seguimiento2
seguimiento_2 = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 4              , 78, 100      ),
    ('B'     , 9              , 54, 90       ),
    ('C'     , 0              , 0 , 0        ),
    ('D'     , 5              , 22, 83       ),
    ('E'     , 1              , 26, 100      ),
    ('F'     , 0              , 0 , 0        ),
    ('G'     , 0              , 0 , 0        ),
    ('H'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento_2

# %%
#| tbl-cap: "Punto de control 3 (final de la semana 13)"
#| label: tbl-seguimiento3
seguimiento_3 = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 4              , 78, 100      ),
    ('B'     , 10             , 60, 100      ),
    ('C'     , 1              , 36, 100      ),
    ('D'     , 6              , 27, 100      ),
    ('E'     , 1              , 26, 100      ),
    ('F'     , 2              , 8 , 22       ),
    ('G'     , 2              , 15, 50       ),
    ('H'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento_3

# %%
pregunta_1 = "1. Calcule PV, EV y AC en cada punto de control.  "
pregunta_2 = "2. Calcule CV, SV, CPI y SPI en cada punto de control y represente su evolución.  "
pregunta_3 = "3. Calcule en cada punto la estimación del coste final EAC = BAC / CPI y comente la tendencia.  "
pregunta_4 = "4. A la vista de la evolución, ¿el proyecto se está recuperando o deteriorando? ¿Qué decisiones tomaría tras el último punto de control?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %% [markdown]
# **Notación** (siglas del PMBOK): BAC (*budget at completion*), presupuesto total del proyecto; PV (*planned value*), valor planificado: presupuesto del trabajo que, según el plan, debía estar hecho en la fecha de control; EV (*earned value*), valor ganado: presupuesto del trabajo realmente hecho; AC (*actual cost*), coste real de ese trabajo. Con ellos: CV = EV − AC (*cost variance*), variación del coste; SV = EV − PV (*schedule variance*), variación del plazo; CPI = EV / AC (*cost performance index*), índice de rendimiento del coste; SPI = EV / PV (*schedule performance index*), índice de rendimiento del plazo. Una variación negativa o un índice menor que 1 indica sobrecoste (CV, CPI) o retraso (SV, SPI).

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
BAC_total = float(datos_proyecto['BAC'].sum())
duracion_planificada = proyecto.duration(datos_proyecto['duration'])

# %%
def suma_vg(detalle, columna, nombre):
    """«PV = A(50.0) + B(40.0) + … = 261.0 k€» con las actividades que aportan algo."""
    v = detalle[columna].astype(float)
    nz = v[v.abs() > 1e-9]
    if len(nz) == 0:
        return f'{nombre} = **0** k€'
    otras = ' (las demás aportan 0)' if len(nz) < len(v) else ''
    return f'{nombre} = ' + ' + '.join(f'{a}({x:.1f})' for a, x in nz.items()) + f' = **{v.sum():.1f}** k€{otras}'


def lectura_vg(PV, EV, AC):
    """Lectura del coste y del plazo con los números: (texto coste, texto plazo)."""
    CV, SV, CPI, SPI = EV - AC, EV - PV, EV / AC, EV / PV
    if abs(CV) < 1e-9:
        coste = f'CV = 0 y CPI = 1: el coste va **según presupuesto**'
    elif CV < 0:
        coste = (f'CV = {CV:.1f} k€ < 0 y CPI = {CPI:.3f} < 1: va **por encima del presupuesto** (sobrecoste de '
                 f'{-CV:.1f} k€; cada k€ gastado ha producido solo {CPI:.3f} k€ de valor)')
    else:
        coste = (f'CV = {CV:.1f} k€ > 0 y CPI = {CPI:.3f} > 1: va **por debajo del presupuesto** (ahorro de '
                 f'{CV:.1f} k€; cada k€ gastado ha producido {CPI:.3f} k€ de valor)')
    if abs(SV) < 1e-9:
        plazo = 'SV = 0 y SPI = 1: va **en plazo**'
    elif SV < 0:
        plazo = (f'SV = {SV:.1f} k€ < 0 y SPI = {SPI:.3f} < 1: va **retrasado** (se ha hecho el {100 * SPI:.1f} % '
                 f'del trabajo que el plan preveía para esta fecha)')
    else:
        plazo = (f'SV = {SV:.1f} k€ > 0 y SPI = {SPI:.3f} > 1: va **adelantado** (se ha hecho el {100 * SPI:.1f} % '
                 f'del trabajo que el plan preveía para esta fecha)')
    return coste, plazo


def explica_vg(detalle, PV, EV, AC, respuesta=True):
    """Sumas por actividad, CV, SV, CPI y SPI con los números sustituidos y su lectura."""
    CV, SV, CPI, SPI = EV - AC, EV - PV, EV / AC, EV / PV
    lineas = []
    if detalle is not None:
        lineas += ['Sumando las columnas de la tabla:', '',
                   '- ' + suma_vg(detalle, 'PV_i', 'PV'), '- ' + suma_vg(detalle, 'EV_i', 'EV'),
                   '- ' + suma_vg(detalle, 'AC', 'AC'), '', 'Variaciones e índices:', '']
    lineas += [f'- CV = EV − AC = {EV:.1f} − {AC:.1f} = **{CV:.1f}** k€',
               f'- SV = EV − PV = {EV:.1f} − {PV:.1f} = **{SV:.1f}** k€',
               f'- CPI = EV / AC = {EV:.1f} / {AC:.1f} = **{CPI:.3f}**',
               f'- SPI = EV / PV = {EV:.1f} / {PV:.1f} = **{SPI:.3f}**']
    if respuesta:
        coste, plazo = lectura_vg(PV, EV, AC)
        lineas += ['', f'**Respuesta:** PV = {PV:.1f} k€, EV = {EV:.1f} k€ y AC = {AC:.1f} k€. En coste, {coste}. '
                       f'En plazo, {plazo}.']
    return Markdown('\n'.join(lineas))


def explica_eac(BAC, PV, EV, AC):
    """Las tres previsiones de EAC con ETC y VAC, el orden entre ellas y por qué."""
    CPI, SPI = EV / AC, EV / PV
    resto = BAC - EV
    hip = {'(a)': (1.0, '1'), '(b)': (CPI, f'CPI = {CPI:.3f}'),
           '(c)': (CPI * SPI, f'CPI · SPI = {CPI:.3f} · {SPI:.3f} = {CPI * SPI:.3f}')}
    eac = {h: AC + resto / d for h, (d, _) in hip.items()}
    lin = [f'Trabajo pendiente valorado a precio de presupuesto: BAC − EV = {BAC:.0f} − {EV:.1f} = {resto:.1f} k€. '
           'Cada hipótesis lo divide por un índice distinto para obtener lo que costará de verdad (ETC, *estimate to '
           'complete*); después EAC = AC + ETC (*estimate at completion*) y VAC = BAC − EAC (*variance at completion*):', '']
    for h, (d, txt) in hip.items():
        etc = resto / d
        lin.append(f'- **{h}** índice {txt}: ETC = {resto:.1f} / {d:.3f} = {etc:.1f} k€; EAC = {AC:.1f} + {etc:.1f} = '
                   f'**{eac[h]:.1f}** k€; VAC = {BAC:.0f} − {eac[h]:.1f} = **{BAC - eac[h]:.1f}** k€'
                   + (f' (equivale a BAC / CPI = {BAC:.0f} / {CPI:.3f})' if h == '(b)' else ''))
    orden = sorted(hip, key=lambda h: eac[h])
    lin += ['', f'Cuanto menor es el índice que divide al trabajo pendiente, mayor es la previsión. Aquí los índices son '
                f'{lista_y([f"{h} {hip[h][0]:.3f}" for h in hip])}, así que de menor a mayor EAC el orden es '
                f'{lista_y(orden)}.']
    expl = []
    if CPI < 1:
        expl.append(f'Con CPI = {CPI:.3f} < 1, la (a) supone que el sobrecoste sufrido hasta ahora no se repetirá en lo que queda')
    elif CPI > 1:
        expl.append(f'Con CPI = {CPI:.3f} > 1, la (a) renuncia al ahorro que se viene obteniendo y es más prudente que la (b)')
    if SPI < 1:
        expl.append(f'con SPI = {SPI:.3f} < 1 el retraso añade un factor que encarece el resto, y la (c) queda por encima de la (b)')
    elif SPI > 1:
        expl.append(f'con SPI = {SPI:.3f} > 1 el adelanto abarata el resto en la (c), que queda por debajo de la (b)')
    if expl:
        lin[-1] += ' ' + '; '.join(expl) + '.'
    lin += ['', f'**Respuesta:** EAC (a) = {eac["(a)"]:.1f} k€ (VAC = {BAC - eac["(a)"]:.1f}), (b) = {eac["(b)"]:.1f} k€ '
                f'(VAC = {BAC - eac["(b)"]:.1f}), (c) = {eac["(c)"]:.1f} k€ (VAC = {BAC - eac["(c)"]:.1f}). La más optimista '
                f'es la {orden[0]} y la más pesimista la {orden[-1]}. Como previsión central se suele tomar la (b): '
                f'**{eac["(b)"]:.1f} k€ frente a los {BAC:.0f} k€ presupuestados**.']
    return Markdown('\n'.join(lin))


def explica_tcpi(BAC, EV, AC):
    """TCPI con los números y si es realista frente al CPI."""
    CPI = EV / AC
    resto, fondos = BAC - EV, BAC - AC
    if fondos <= 0:
        return Markdown(f'TCPI = (BAC − EV) / (BAC − AC) = ({BAC:.0f} − {EV:.1f}) / ({BAC:.0f} − {AC:.1f}) = {resto:.1f} / {fondos:.1f}: '
                        f'ya se ha gastado todo el presupuesto y queda trabajo por valor de {resto:.1f} k€. **Respuesta:** terminar en '
                        'presupuesto es imposible; hay que pedir más fondos (rehacer la línea base de coste).')
    TCPI = resto / fondos
    mejora = (TCPI / CPI - 1) * 100
    if TCPI <= CPI:
        juicio = (f'basta con mantener el rendimiento actual (CPI = {CPI:.3f} ≥ TCPI): **es realista**')
    elif mejora <= 10:
        juicio = (f'exige mejorar el rendimiento actual (CPI = {CPI:.3f}) un {mejora:.0f} %: **es alcanzable** con medidas correctoras')
    else:
        juicio = (f'exige mejorar el rendimiento actual (CPI = {CPI:.3f}) un {mejora:.0f} % (TCPI / CPI − 1 = {TCPI:.3f} / {CPI:.3f} − 1): '
                  '**no es realista** sin cambios de fondo (alcance, método o contrato); lo prudente es aceptar un EAC mayor que el BAC')
    return Markdown(f'TCPI = (BAC − EV) / (BAC − AC) = ({BAC:.0f} − {EV:.1f}) / ({BAC:.0f} − {AC:.1f}) = {resto:.1f} / {fondos:.1f} = '
                    f'**{TCPI:.3f}**: queda trabajo por valor de {resto:.1f} k€ y solo quedan {fondos:.1f} k€ de presupuesto, así que '
                    f'cada k€ que se gaste a partir de ahora tendría que producir {TCPI:.3f} k€ de valor.\n\n'
                    f'**Respuesta:** TCPI = {TCPI:.3f}; {juicio}.')


def explica_es(pv_acumulado, EV, AT):
    """Programación ganada: (ES, texto con la interpolación). pv_acumulado indexado por semana (incluye la 0)."""
    pv = pv_acumulado.astype(float)
    C = int(pv[pv <= EV + 1e-9].index.max())
    if C + 1 not in pv.index:
        return float(C), f'El EV = {EV:.1f} k€ alcanza el PV acumulado de la última semana del plan: ES = {C}.'
    PV_C, PV_C1 = float(pv[C]), float(pv[C + 1])
    ES = C + (EV - PV_C) / (PV_C1 - PV_C)
    txt = (f'El EV = {EV:.1f} k€ queda entre el PV acumulado de la semana {C} ({PV_C:.1f}) y el de la semana {C + 1} '
           f'({PV_C1:.1f}), así que C = {C} y ES = {C} + ({EV:.1f} − {PV_C:.1f}) / ({PV_C1:.1f} − {PV_C:.1f}) = '
           f'{C} + {EV - PV_C:.1f} / {PV_C1 - PV_C:.1f} = **{ES:.2f}** semanas: el valor ganado hasta la semana {AT} '
           f'se había previsto ganarlo en la semana {ES:.2f}.')
    return ES, txt

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy de la planificación"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
seguimientos = {5: seguimiento_1, 9: seguimiento_2, 13: seguimiento_3}
filas = {}
curvas = {}
for semana, seguimiento in seguimientos.items():
    datos_vg = pd.concat([datos_proyecto, seguimiento], axis=1)
    gantts = EarnedValue(proyecto.copy()).calcula_gantts(datos_vg, planned_durations_label='duration',
                                                          actual_durations_label='actual_duration', PV_label='BAC',
                                                          AC_label='AC', percentage_complete_label='performed', tikz=False)
    PV = float(gantts['acumulados'].loc[semana, 'PV'])
    EV = float((datos_vg['BAC'] * datos_vg['performed'] / 100).sum())
    AC = float(datos_vg['AC'].sum())
    filas[semana] = {'PV': PV, 'EV': EV, 'AC': AC}
indicadores = pd.DataFrame(filas).T
indicadores.index.name = 'semana'

# %% [markdown]
# De dónde sale cada cifra en cada punto de control: el valor planificado de cada actividad es su BAC por la fracción de su duración que debería haberse ejecutado (según el calendario de inicios más tempranos); el valor ganado, su BAC por el porcentaje realizado; el coste real, lo gastado en ella. Los del proyecto son las sumas.

# %%
calendario_plan = proyecto.calendar(datos_proyecto['duration'])
for semana, seguimiento in seguimientos.items():
    display(Markdown(f'**Punto de control: final de la semana {semana}**'))
    det = detalle_valor_ganado(pd.concat([datos_proyecto, seguimiento], axis=1), calendario_plan, semana)
    display(det[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']])
    display(Markdown('\n'.join(['- ' + suma_vg(det, c, n) for c, n in (('PV_i', 'PV'), ('EV_i', 'EV'), ('AC', 'AC'))])))

# %%
#| tbl-cap: "PV, EV y AC del proyecto en cada punto de control (k€)"
#| label: tbl-pvevac
indicadores.round(1)

# %%
Markdown('**Respuesta:** ' + '; '.join(f'semana {s}: PV = {f.PV:.1f}, EV = {f.EV:.1f} y AC = {f.AC:.1f} k€' for s, f in indicadores.iterrows()) + '.')

# %% [markdown]
# Los tres diagramas de Gantt del valor ganado en el último punto de control:

# %%
datos_vg_ultimo = pd.concat([datos_proyecto, seguimientos[max(seguimientos)]], axis=1)
gantts_ultimo = EarnedValue(proyecto.copy()).calcula_gantts(datos_vg_ultimo, planned_durations_label='duration',
                                                             actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',
                                                             percentage_complete_label='performed', tikz=True,
                                                             params=dict(period_width=1, row_height=0.7, activity_relative_height=0.6, **GANTT_PARAMS,
                                       critical_background_color='white!90!green', regular_background_color='white!90!green',
                                       regular_text_color='black', critical_text_color='black'))

# %%
#| fig-cap: "PV: valor planificado (último punto de control)"
#| label: fig-ganttPV
pinta_tikz(gantts_ultimo['gantt_PV'])

# %%
#| fig-cap: "EV: valor ganado (último punto de control)"
#| label: fig-ganttEV
pinta_tikz(gantts_ultimo['gantt_EV'])

# %%
#| fig-cap: "AC: coste real (último punto de control)"
#| label: fig-ganttAC
pinta_tikz(gantts_ultimo['gantt_AC'])

# %%
#| fig-cap: "Curvas S hasta el último punto de control"
#| label: fig-curvasS
plot_curvas_valor_ganado(gantts_ultimo)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
indicadores['CV'] = indicadores['EV'] - indicadores['AC']
indicadores['SV'] = indicadores['EV'] - indicadores['PV']
indicadores['CPI'] = indicadores['EV'] / indicadores['AC']
indicadores['SPI'] = indicadores['EV'] / indicadores['PV']
for semana, f in indicadores.iterrows():
    display(Markdown(f'**Semana {semana}**'))
    display(explica_vg(None, f.PV, f.EV, f.AC, respuesta=False))

# %%
#| tbl-cap: "Variaciones (k€) e índices en cada punto de control"
#| label: tbl-indicadores
indicadores.round(3)

# %%
#| fig-cap: "Evolución de CPI y SPI"
#| label: fig-evolucion
ax = indicadores[['CPI', 'SPI']].plot(marker='o', figsize=(7, 4))
ax.axhline(1, color='grey', linestyle='--'); ax.set_ylabel('índice');

# %%
serie = lambda x: ' → '.join(f'{v:.3f}' for v in x)
def bajo_uno(x, malo):
    debajo = [str(s) for s, v in x.items() if v < 1]
    if len(debajo) == len(x):
        return f'por debajo de 1 en los tres controles ({malo} en todos)'
    if not debajo:
        return 'igual o por encima de 1 en los tres controles'
    return f'por debajo de 1 ({malo}) solo en la semana {lista_y(debajo)}'
Markdown(f'''**Respuesta:** CPI {serie(indicadores['CPI'])}, {bajo_uno(indicadores['CPI'], 'sobrecoste')}; SPI {serie(indicadores['SPI'])}, {bajo_uno(indicadores['SPI'], 'retraso')}. La gráfica muestra esa evolución frente a la referencia 1.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
indicadores['EAC'] = BAC_total / indicadores['CPI']
indicadores['VAC'] = BAC_total - indicadores['EAC']
display(Markdown('\n'.join(f'- Semana {s}: EAC = BAC / CPI = {BAC_total:.0f} / {f.CPI:.3f} = **{f.EAC:.1f}** k€; '
                            f'VAC = BAC − EAC = {BAC_total:.0f} − {f.EAC:.1f} = {f.VAC:.1f} k€' for s, f in indicadores.iterrows())))
indicadores[['CPI', 'EAC', 'VAC']].round({'CPI': 3, 'EAC': 1, 'VAC': 1})

# %%
eac = indicadores['EAC']
pasos = ['sube' if b > a + 0.05 else ('baja' if b < a - 0.05 else 'se mantiene') for a, b in zip(eac.iloc[:-1], eac.iloc[1:])]
if all(p == 'sube' for p in pasos):
    tendencia = 'la previsión empeora en cada control: el CPI baja y cada vez se prevé un coste final mayor'
elif all(p == 'baja' for p in pasos):
    tendencia = 'la previsión mejora en cada control: el CPI sube y cada vez se prevé un coste final menor'
else:
    tendencia = 'la previsión no evoluciona en un solo sentido (' + ' y luego '.join(pasos) + ')'
Markdown(f'''**Respuesta:** EAC {" → ".join(f"{v:.1f}" for v in eac)} k€ frente a un BAC de {BAC_total:.0f} k€: {tendencia}. En el último control se prevé terminar con {abs(indicadores["VAC"].iloc[-1]):.1f} k€ {"más" if indicadores["VAC"].iloc[-1] < 0 else "menos"} de lo presupuestado.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
cpi = indicadores['CPI']; spi = indicadores['SPI']; sem = list(indicadores.index)
def evolucion(x, nombre):
    tramos = [f'{"mejora" if b > a else "empeora"} de {a:.3f} a {b:.3f} entre las semanas {s0} y {s1}' for (s0, a), (s1, b) in zip(list(x.items())[:-1], list(x.items())[1:])]
    return f'el {nombre} ' + ' y ' .join(tramos)
ultimo = seguimientos[max(seguimientos)]
pendientes = [a for a in ordena_ruta(list(ultimo.index), calendario_plan) if ultimo.loc[a, 'performed'] < 100]
criticas_pend = [a for a in pendientes if calendario_plan.loc[a, 'H_total'] == 0]
coste_mal, plazo_mal = cpi.iloc[-1] < 1, spi.iloc[-1] < 1
sentido_c = 'mejora' if cpi.iloc[-1] > cpi.iloc[-2] else 'empeora'
sentido_s = 'mejora' if spi.iloc[-1] > spi.iloc[-2] else 'empeora'
estado = ('se está deteriorando en coste y en plazo' if sentido_c == sentido_s == 'empeora' else
          'se está recuperando en coste y en plazo' if sentido_c == sentido_s == 'mejora' else
          f'no se recupera del todo: el coste {sentido_c} y el plazo {sentido_s}')
aviso_spi = (' La mejora del SPI en el último tramo puede ser en parte aparente: el SPI tiende a 1 al acercarse el final aunque el retraso persista; '
             'para el plazo es preferible la programación ganada.' if spi.iloc[-1] > spi.iloc[-2] else '')
decisiones = []
if plazo_mal:
    decisiones.append(f'recuperar plazo actuando sobre las actividades críticas pendientes ({lista_y(criticas_pend)}), acelerándolas si el coste de hacerlo '
                      'compensa la penalización o los costes indirectos que se evitan' if criticas_pend else
                      'recuperar plazo en las actividades pendientes que hayan consumido su holgura')
if coste_mal:
    decisiones.append(f'analizar las causas del sobrecoste (CPI = {cpi.iloc[-1]:.3f}) para que no se repita en las actividades pendientes '
                      f'({lista_y(pendientes)}) y revisar el EAC con la dirección ({indicadores["EAC"].iloc[-1]:.1f} k€ frente a {BAC_total:.0f})')
Markdown(f'''Evolución: {evolucion(cpi, "CPI")}; {evolucion(spi, "SPI")}.{aviso_spi}

**Respuesta:** comparando los dos últimos controles el proyecto {estado} (CPI {cpi.iloc[-2]:.3f} → {cpi.iloc[-1]:.3f}, SPI {spi.iloc[-2]:.3f} → {spi.iloc[-1]:.3f}); en el último control sigue {"con sobrecoste" if coste_mal else "sin sobrecoste"} y {"con retraso" if plazo_mal else "sin retraso"}. Decisiones tras el último control: {"; ".join(decisiones) if decisiones else "mantener el seguimiento"}.''')
