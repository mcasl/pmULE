# ---
# title: ValorGanado 09
# subtitle: "9 actividades; qué vigilar cruzando SPI/CPI por actividad con la holgura"
# coleccion: 12_valor_ganado
# variante: vigilar
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por actividad (BAC, k€, repartido uniformemente) se adjunta, junto con el seguimiento al final de la semana 10. El jefe de proyecto quiere una lista corta de actividades sobre las que actuar.

# %%
pregunta_1 = "1. Calcule PV, EV y AC de cada actividad y los índices CPI y SPI *por actividad*; añada la holgura total planificada de cada una.  "
pregunta_2 = "2. Clasifique las actividades en cuatro grupos: críticas con retraso (SPI < 1), no críticas con retraso, actividades con sobrecoste (CPI < 1) sin retraso, y el resto. ¿Cuáles amenazan el plazo? ¿Cuáles sólo el presupuesto?  "
pregunta_3 = "3. Calcule los índices globales del proyecto. ¿Qué actividades explican la mayor parte de SV y de CV (ordene por contribución)?  "
pregunta_4 = "4. Proponga las tres actividades a vigilar y justifique el criterio: ¿por qué un SPI bajo en una actividad con mucha holgura preocupa menos que en una crítica, y por qué con el CPI no hay esa distinción?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Planificación del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, BAC
    ('A'     , '---'      , 8       , 10 ),
    ('B'     , '---'      , 1       , 40 ),
    ('C'     , '---'      , 2       , 50 ),
    ('D'     , 'A'        , 6       , 40 ),
    ('E'     , 'B'        , 4       , 60 ),
    ('F'     , 'B,C'      , 3       , 30 ),
    ('G'     , 'E'        , 1       , 20 ),
    ('H'     , 'G'        , 6       , 20 ),
    ('I'     , 'F,H'      , 7       , 40 ),
], columns=['activity', 'predecessor', 'duration', 'BAC']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la apertura de una clínica veterinaria; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Estudio de mercado y local',
    'B': 'Proyecto de reforma y licencia',
    'C': 'Obra de reforma',
    'D': 'Instalación de quirófano y radiología',
    'E': 'Mobiliario y equipamiento clínico',
    'F': 'Software de gestión',
    'G': 'Selección de personal',
    'H': 'Acuerdos con proveedores',
    'I': 'Registro sanitario',
}, name='Descripción')
descripciones.to_frame()

# %%
#| tbl-cap: "Seguimiento en la semana de control"
#| label: tbl-seguimiento
seguimiento = pd.DataFrame([
#  activity, actual_duration, AC, performed
    ('A'     , 10             , 10, 100      ),
    ('B'     , 2              , 52, 100      ),
    ('C'     , 2              , 50, 100      ),
    ('D'     , 0              , 0 , 0        ),
    ('E'     , 6              , 78, 100      ),
    ('F'     , 3              , 27, 100      ),
    ('G'     , 2              , 24, 100      ),
    ('H'     , 0              , 0 , 0        ),
    ('I'     , 0              , 0 , 0        ),
], columns=['activity', 'actual_duration', 'AC', 'performed']).set_index('activity')
seguimiento

# %% [markdown]
# # Resolución

# %% [markdown]
# **Notación** (siglas del PMBOK): BAC (*budget at completion*), presupuesto total del proyecto; PV (*planned value*), valor planificado: presupuesto del trabajo que, según el plan, debía estar hecho en la fecha de control; EV (*earned value*), valor ganado: presupuesto del trabajo realmente hecho; AC (*actual cost*), coste real de ese trabajo. Con ellos: CV = EV − AC (*cost variance*), variación del coste; SV = EV − PV (*schedule variance*), variación del plazo; CPI = EV / AC (*cost performance index*), índice de rendimiento del coste; SPI = EV / PV (*schedule performance index*), índice de rendimiento del plazo. Una variación negativa o un índice menor que 1 indica sobrecoste (CV, CPI) o retraso (SV, SPI).

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
semana_control = 10

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
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT de la planificación con el camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario planificado (inicios más tempranos y holguras)"
#| label: tbl-calendario
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# PV de cada actividad = BAC por la fracción planificada transcurrida en la semana de control; EV = BAC por el porcentaje realizado; AC = coste real anotado:

# %%
#| tbl-cap: "Cómo se obtiene el PV y el EV de cada actividad"
#| label: tbl-valorGanadoPaso
datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)
detalle = detalle_valor_ganado(datos_valor_ganado, calendario, semana_control)
detalle[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]

# %% [markdown]
# Con ellos, para cada actividad CPI = EV / AC y SPI = EV / PV; quedan sin definir (—) cuando el denominador es 0 (actividad sin coste real o sin trabajo planificado todavía). La holgura total sale del calendario planificado.

# %%
#| tbl-cap: "Valor ganado por actividad"
#| label: tbl-porActividad
por_actividad = pd.DataFrame({'PV': detalle['PV_i'].astype(float), 'EV': detalle['EV_i'].astype(float), 'AC': seguimiento['AC'].astype(float),
                              'H_total': calendario['H_total']})
por_actividad['CPI'] = por_actividad['EV'] / por_actividad['AC'].replace(0, np.nan)
por_actividad['SPI'] = por_actividad['EV'] / por_actividad['PV'].replace(0, np.nan)
limpia_tabla(por_actividad.rename(columns={'H_total': 'holgura total'}), decimales=3)

# %%
p = por_actividad
lineas = []
for a, f in p.iterrows():
    cpi = f'CPI = {f.EV:.1f} / {f.AC:.1f} = {f.CPI:.3f}' if f.AC > 0 else 'CPI sin definir (AC = 0)'
    spi = f'SPI = {f.EV:.1f} / {f.PV:.1f} = {f.SPI:.3f}' if f.PV > 0 else 'SPI sin definir (PV = 0)'
    lineas.append(f'- {a}: {cpi}; {spi}; holgura total {int(f.H_total)}')
Markdown('\n'.join(lineas))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
grupos = {
    'críticas con retraso (amenazan el plazo)': p.index[(p['H_total'] == 0) & (p['SPI'] < 1)],
    'no críticas con retraso': p.index[(p['H_total'] > 0) & (p['SPI'] < 1)],
    'sobrecoste sin retraso (amenazan el presupuesto)': p.index[(p['CPI'] < 1) & ~(p['SPI'] < 1)],
}
grupos['resto'] = p.index.difference(pd.Index([]).append([pd.Index(v) for v in grupos.values()]))
pd.Series({k: ', '.join(v) or '—' for k, v in grupos.items()}, name='actividades').to_frame()

# %% [markdown]
# Una no crítica retrasada solo amenaza el plazo si su retraso supera su holgura total. El retraso de cada actividad, en semanas de trabajo, es la diferencia entre la fracción planificada y la realizada por su duración: $(PV_i - EV_i)/BAC_i \cdot d_i$.

# %%
dur = datos_proyecto['duration']; bac = datos_proyecto['BAC']
retraso = ((p['PV'] - p['EV']) / bac * dur)
no_criticas = list(grupos['no críticas con retraso'])
criticas_r = list(grupos['críticas con retraso (amenazan el plazo)'])
lineas = [f'- {a}: ({p.loc[a, "PV"]:.1f} − {p.loc[a, "EV"]:.1f}) / {bac[a]:.0f} · {dur[a]} = {retraso[a]:.1f} semanas, '
          f'{"más" if retraso[a] > p.loc[a, "H_total"] else "no más"} que su holgura de {plural(int(p.loc[a, "H_total"]), "semana")}'
          for a in criticas_r + no_criticas]
amenazan = criticas_r + [a for a in no_criticas if retraso[a] > p.loc[a, 'H_total']]
absorben = [a for a in no_criticas if retraso[a] <= p.loc[a, 'H_total']]
solo_coste = list(grupos['sobrecoste sin retraso (amenazan el presupuesto)'])
tambien_coste = [a for a in criticas_r + no_criticas if p.loc[a, 'CPI'] < 1]
exceden = [a for a in no_criticas if retraso[a] > p.loc[a, 'H_total']]
partes = ([f'{lista_y(criticas_r)}, por ser {"crítica" if len(criticas_r) == 1 else "críticas"} y estar {"retrasada" if len(criticas_r) == 1 else "retrasadas"}'] if criticas_r else []) \
         + ([f'{lista_y(exceden)}, cuyo retraso ya supera su holgura'] if exceden else [])
Markdown('\n'.join(lineas) + '\n\n'
         + f'**Respuesta:** amenazan el plazo {"; ".join(partes) if partes else "ninguna actividad"}'
         + (f'; {lista_y(absorben)} se {"retrasa" if len(absorben) == 1 else "retrasan"} dentro de su holgura y de momento no lo mueve{"" if len(absorben) == 1 else "n"}' if absorben else '')
         + f'. Amenazan solo el presupuesto {lista_y(solo_coste) if solo_coste else "ninguna"} (CPI < 1 sin retraso)'
         + (f'; además, {lista_y(tambien_coste)} {"tiene" if len(tambien_coste) == 1 else "tienen"} retraso y sobrecoste a la vez' if tambien_coste else '') + '.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
PV, EV, AC = p['PV'].sum(), p['EV'].sum(), p['AC'].sum()
explica_vg(detalle, PV, EV, AC)

# %% [markdown]
# Las variaciones del proyecto son la suma de las de sus actividades, así que cada actividad contribuye con su SV_i = EV_i − PV_i y su CV_i = EV_i − AC_i:

# %%
contrib = pd.DataFrame({'SV_i = EV − PV': p['EV'] - p['PV'], 'CV_i = EV − AC': p['EV'] - p['AC']})
display(limpia_tabla(contrib.sort_values('SV_i = EV − PV').T, decimales=1))
def ranking(serie, total, nombre):
    neg = serie[serie < -0.05].sort_values()
    if neg.empty:
        return f'ninguna actividad tiene {nombre} negativa'
    partes = [f'{a} ({v:.1f} k€, {100 * v / total:.0f} %)' for a, v in neg.items()]
    empate = [a for a, v in neg.items() if abs(v - neg.iloc[0]) < 0.05]
    nota = f' ({lista_y(empate)} empatan en cabeza)' if len(empate) > 1 else ''
    return ' > '.join(partes) + nota
SVt, CVt = contrib.iloc[:, 0].sum(), contrib.iloc[:, 1].sum()
Markdown(f'''Comprobación: SV = {SVt:.1f} k€ y CV = {CVt:.1f} k€, las mismas que con los totales.

**Respuesta:** SPI = {EV / PV:.3f} y CPI = {EV / AC:.3f}. Contribuciones a SV (de mayor a menor peso): {ranking(contrib.iloc[:, 0], SVt, 'SV')}. Contribuciones a CV: {ranking(contrib.iloc[:, 1], CVt, 'CV')}.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Criterio de prioridad (uno razonable entre varios): el retraso de una actividad crítica pesa más que el de una con holgura, y el sobrecoste pesa igual en todas. Con las contribuciones negativas en k€:
#
# $$\text{prioridad} = |SV_i^-| \cdot (2 + 3 \cdot \text{crítica}) + |CV_i^-|$$
#
# donde $SV_i^-$ y $CV_i^-$ son la parte negativa de cada variación y *crítica* vale 1 si la holgura total es 0 (el retraso de una crítica pesa 5 veces su valor, el de una no crítica 2 veces).

# %%
puntuacion = pd.DataFrame({'crítica': (p['H_total'] == 0).astype(int), 'SV_i': contrib.iloc[:, 0], 'CV_i': contrib.iloc[:, 1]})
puntuacion['prioridad'] = (-puntuacion['SV_i'].clip(upper=0) * (2 + puntuacion['crítica'] * 3) - puntuacion['CV_i'].clip(upper=0))
orden = puntuacion.sort_values('prioridad', ascending=False)
vigilar = [a for a in orden.index if orden.loc[a, 'prioridad'] > 0.05][:3]
display(limpia_tabla(orden.assign(**{'crítica': orden['crítica'].map({1: 'sí', 0: 'no'})}), decimales=1))
calc = [f'{a}: {abs(min(orden.loc[a, "SV_i"], 0)):.1f}·{2 + 3 * orden.loc[a, "crítica"]} + {abs(min(orden.loc[a, "CV_i"], 0)):.1f} = {orden.loc[a, "prioridad"]:.1f}' for a in vigilar]
Markdown(f'''{"; ".join(calc)}.

**Respuesta:** actividades a vigilar: **{lista_y(vigilar)}**. Un SPI bajo en una actividad con holgura puede absorberse sin mover el plazo (su retraso solo importa cuando supera la holgura); en una crítica se traslada entero al final. El dinero, en cambio, no tiene holgura: cada euro de sobrecoste, crítico o no, se suma al coste final, por lo que el CPI se vigila igual en todas.''')
