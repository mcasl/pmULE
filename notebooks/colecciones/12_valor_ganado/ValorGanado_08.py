# ---
# title: ValorGanado 08
# subtitle: "7 actividades; perfiles de desembolso no uniformes; control en la semana 8"
# coleccion: 12_valor_ganado
# variante: coste-no-uniforme
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
# El proyecto corresponde a la instalación de una planta fotovoltaica sobre cubierta; las actividades son las siguientes (las duraciones del cuadro están en semanas).

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

# %% [markdown]
# **Notación** (siglas del PMBOK): BAC (*budget at completion*), presupuesto total del proyecto; PV (*planned value*), valor planificado: presupuesto del trabajo que, según el plan, debía estar hecho en la fecha de control; EV (*earned value*), valor ganado: presupuesto del trabajo realmente hecho; AC (*actual cost*), coste real de ese trabajo. Con ellos: CV = EV − AC (*cost variance*), variación del coste; SV = EV − PV (*schedule variance*), variación del plazo; CPI = EV / AC (*cost performance index*), índice de rendimiento del coste; SPI = EV / PV (*schedule performance index*), índice de rendimiento del plazo. Una variación negativa o un índice menor que 1 indica sobrecoste (CV, CPI) o retraso (SV, SPI).

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
semana_control = 8
BAC_total = float(datos_proyecto['BAC'].sum())

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
#| fig-cap: "Grafo PERT de la planificación"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario de inicios más tempranos"
#| label: tbl-calendario
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# Con el inicio más temprano de cada actividad se reparte su BAC semana a semana según su perfil (una actividad que empieza en el instante ES ocupa las semanas ES + 1 a ES + duración); la columna *suma* comprueba que se recupera el BAC.

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
num = lambda x: f'{x:.1f}' if abs(10 * x - round(10 * x)) < 1e-9 else f'{x:.2f}'
rango = lambda a, b: f'la semana {a}' if a == b else f'cada una de las semanas {a} a {b}'
lineas = []
for a, f in datos_proyecto.iterrows():
    es, d, B = int(calendario.loc[a, 'inicio_mas_temprano']), int(f['duration']), float(f['BAC'])
    if f['profile'] == 'uniforme' or d == 1:
        continue
    resto = B / 2 / (d - 1)
    otras = rango(es + 2, es + d) if f['profile'] == 'inicio' else rango(es + 1, es + d - 1)
    clave = f'semana {es + 1}' if f['profile'] == 'inicio' else f'semana {es + d}'
    lineas.append(f'- {a} (perfil {f["profile"]}, BAC = {B:.0f}, {d} semanas, semanas {es + 1} a {es + d}): {B:.0f} / 2 = {B / 2:.1f} en la {clave} '
                  f'y ({B:.0f} − {B / 2:.1f}) / {d - 1} = {num(resto)} en {otras}; suma {B / 2:.1f} + {d - 1}·{num(resto)} = {B:.1f}')
ok = bool(np.allclose(coste_semanal['suma'], datos_proyecto['BAC']))
Markdown('Las de perfil uniforme reciben BAC / duración cada semana. Las de perfil no uniforme:\n\n' + '\n'.join(lineas)
         + f'\n\n**Respuesta:** el calendario es el de la tabla anterior y el reparto semanal el de la tabla de costes; '
         + ('todas las filas suman su BAC.' if ok else 'hay filas que no suman su BAC: revisar el reparto.'))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# El PV de cada actividad en la semana de control es la suma de sus costes planificados hasta esa semana (columnas 1 a la semana de control de la tabla anterior); con reparto uniforme sería BAC por la fracción de duración transcurrida.

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
difieren = [a for a in datos_proyecto.index if abs(PV_i[a] - PV_uniforme_i[a]) > 1e-6]
lineas = []
for a in difieren:
    es, d, B = int(calendario.loc[a, 'inicio_mas_temprano']), int(datos_proyecto.loc[a, 'duration']), float(datos_proyecto.loc[a, 'BAC'])
    k = min(max(semana_control - es, 0), d)
    lineas.append(f'- {a} (perfil {datos_proyecto.loc[a, "profile"]}, {k} de {d} semanas transcurridas): según perfil '
                  f'{" + ".join(num(x) for x in coste_semanal.loc[a, semanas] if x > 0)} = {PV_i[a]:.1f}; uniforme {B:.0f}·{k}/{d} = {PV_uniforme_i[a]:.1f}')
Markdown('Solo cambian las actividades en curso con perfil no uniforme (las terminadas suman su BAC y las no empezadas, 0 con los dos criterios):\n\n'
         + '\n'.join(lineas)
         + f'\n\n- PV según perfil = {" + ".join(f"{x:.1f}" for x in PV_i if x > 0)} = **{PV:.1f}** k€\n'
         + f'- PV uniforme = {" + ".join(f"{x:.1f}" for x in PV_uniforme_i if x > 0)} = **{PV_uniforme:.1f}** k€\n\n'
         + f'**Respuesta:** PV = {PV:.1f} k€ con los perfiles reales frente a {PV_uniforme:.1f} k€ con reparto uniforme '
         + f'({abs(PV - PV_uniforme):.1f} k€ {"más" if PV > PV_uniforme else "menos"}), por {lista_y(difieren)}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
EV_i = datos_proyecto['BAC'] * seguimiento['performed'] / 100
EV = float(EV_i.sum())
AC = float(seguimiento['AC'].sum())
indices = pd.DataFrame({'PV según perfil': [PV, EV, AC, EV / AC, EV / PV], 'PV uniforme': [PV_uniforme, EV, AC, EV / AC, EV / PV_uniforme]},
                       index=['PV', 'EV', 'AC', 'CPI = EV / AC', 'SPI = EV / PV']).round(3)
indices

# %%
s1, s2 = EV / PV, EV / PV_uniforme
lee = lambda s: 'retraso' if s < 1 else ('adelanto' if s > 1 else 'en plazo')
Markdown(f'''- EV = Σ BAC·% = {" + ".join(f"{a}({x:.1f})" for a, x in EV_i.items() if x > 0)} = **{EV:.1f}** k€
- AC = Σ coste real = {" + ".join(f"{a}({x:.0f})" for a, x in seguimiento['AC'].items() if x > 0)} = **{AC:.1f}** k€
- CPI = EV / AC = {EV:.1f} / {AC:.1f} = **{EV / AC:.3f}** (el mismo con los dos PV: no depende del PV)
- SPI con el PV correcto = {EV:.1f} / {PV:.1f} = **{s1:.3f}** ({lee(s1)}); con el PV uniforme = {EV:.1f} / {PV_uniforme:.1f} = **{s2:.3f}** ({lee(s2)})

**Respuesta:** {'el diagnóstico de plazo cambia: ' + lee(s2) + ' con el PV uniforme, ' + lee(s1) + ' con el correcto' if lee(s1) != lee(s2) else 'el diagnóstico de plazo (' + lee(s1) + ') no cambia, pero sí su magnitud: SPI ' + format(s1, '.3f') + ' frente a ' + format(s2, '.3f')}; el de coste (CPI = {EV / AC:.3f}, {"sobrecoste" if EV < AC else "ahorro"}) es el mismo.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# El EV es el BAC por el porcentaje *físico* realizado: mide trabajo hecho, no dinero desembolsado, y por eso no depende de cuándo se planeó pagar. El PV sí es dinero planificado hasta la fecha. Si una actividad en plazo tiene perfil `final` (el gran pago al terminar), su PV a mitad de camino es pequeño y el EV (proporcional al avance físico) lo supera: el proyecto parece adelantado sin estarlo. Con perfil `inicio` ocurre lo contrario. Para que el SPI mida plazo, el PV debe seguir el mismo criterio que el EV.

# %%
ejemplos = [f'{a} (perfil {datos_proyecto.loc[a, "profile"]}): PV según perfil {PV_i[a]:.1f} frente a {PV_uniforme_i[a]:.1f} uniforme' for a in difieren]
Markdown(f'En este proyecto se ve en {"; ".join(ejemplos)}. **Respuesta:** el EV no depende del perfil porque mide avance físico; '
         'el perfil `final` hace parecer adelantado un proyecto que va en plazo (y el `inicio`, retrasado).' if ejemplos else
         '**Respuesta:** el EV no depende del perfil porque mide avance físico; el perfil `final` hace parecer adelantado un proyecto que va en plazo.')
