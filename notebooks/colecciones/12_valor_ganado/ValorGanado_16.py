# ---
# title: ValorGanado 16
# subtitle: "9 actividades (ocultas); curvas S hasta la semana 14 de 20"
# coleccion: 12_valor_ganado
# variante: inverso-desde-curvas
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
# De un proyecto con presupuesto total BAC = 320 k€ y duración planificada de 20 semanas sólo se dispone del informe de seguimiento adjunto: los valores acumulados al final de cada semana, hasta la 14, del valor planificado (PV), el valor ganado (EV) y el coste real (AC), en k€. No se conoce el cuadro de actividades.

# %%
pregunta_1 = "1. Dibuje las tres curvas S. Calcule CV, SV, CPI y SPI en cada semana y represente la evolución de los dos índices.  "
pregunta_2 = "2. ¿En qué semana empezó el proyecto a retrasarse? ¿Y a tener sobrecoste? ¿Fueron simultáneos?  "
pregunta_3 = "3. Con los datos de la semana 14, estime el coste final (EAC = BAC / CPI) y la duración final por programación ganada: ES es la semana en que el PV planificado alcanzaba el EV actual; SPI(t) = ES / semana; duración ≈ duración planificada / SPI(t).  "
pregunta_4 = "4. ¿Qué no puede saberse sin el cuadro de actividades? Ponga un ejemplo de dos situaciones distintas que darían las mismas curvas.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Informe de seguimiento (valores acumulados, k€)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  week, PV   , EV   , AC 
    (1   , 18.7 , 18.5 , 22 ),
    (2   , 37.3 , 37.5 , 44 ),
    (3   , 56   , 56   , 67 ),
    (4   , 69.2 , 67.6 , 83 ),
    (5   , 82.5 , 79.5 , 99 ),
    (6   , 93.8 , 89.4 , 113),
    (7   , 105  , 99   , 127),
    (8   , 108.8, 108.6, 140),
    (9   , 112.5, 112.5, 145),
    (10  , 116.2, 116.4, 150),
    (11  , 120  , 120  , 155),
    (12  , 138  , 129.8, 165),
    (13  , 161  , 140  , 177),
    (14  , 184  , 157.2, 198),
], columns=['week', 'PV', 'EV', 'AC']).set_index('week')
datos_enunciado

# %% [markdown]
# # Resolución

# %% [markdown]
# **Notación** (siglas del PMBOK): BAC (*budget at completion*), presupuesto total del proyecto; PV (*planned value*), valor planificado: presupuesto del trabajo que, según el plan, debía estar hecho en la fecha de control; EV (*earned value*), valor ganado: presupuesto del trabajo realmente hecho; AC (*actual cost*), coste real de ese trabajo. Con ellos: CV = EV − AC (*cost variance*), variación del coste; SV = EV − PV (*schedule variance*), variación del plazo; CPI = EV / AC (*cost performance index*), índice de rendimiento del coste; SPI = EV / PV (*schedule performance index*), índice de rendimiento del plazo. Una variación negativa o un índice menor que 1 indica sobrecoste (CV, CPI) o retraso (SV, SPI).

# %%
BAC_total, duracion_planificada = 320, 20
curvas = datos_enunciado.copy().astype(float)

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

# %%
#| fig-cap: "Curvas S del proyecto"
#| label: fig-curvasS
from matplotlib import pyplot as plt
ax = curvas.plot(marker='o', figsize=(8, 4)); ax.set_xlabel('semana'); ax.set_ylabel('k€ acumulados'); ax.grid(alpha=0.3); plt.show()

# %% [markdown]
# Cada fila de la tabla siguiente aplica las cuatro fórmulas a los valores acumulados de esa semana; debajo, el cálculo completo de la última semana como ejemplo.

# %%
#| tbl-cap: "Indicadores semana a semana"
#| label: tbl-indicadores
ind = curvas.copy()
ind['CV = EV − AC'] = ind['EV'] - ind['AC']
ind['SV = EV − PV'] = ind['EV'] - ind['PV']
ind['CPI = EV / AC'] = ind['EV'] / ind['AC']
ind['SPI = EV / PV'] = ind['EV'] / ind['PV']
ind.round(3)

# %%
t_ult = int(curvas.index[-1])
display(Markdown(f'**Semana {t_ult}**'))
explica_vg(None, curvas.loc[t_ult, 'PV'], curvas.loc[t_ult, 'EV'], curvas.loc[t_ult, 'AC'])

# %%
#| fig-cap: "Evolución de CPI y SPI"
#| label: fig-indices
ax = ind[['CPI = EV / AC', 'SPI = EV / PV']].plot(marker='o', figsize=(8, 4)); ax.axhline(1, color='grey', linestyle='--'); ax.set_xlabel('semana'); plt.show()

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
umbral = 0.97
def inicio(serie, nombre, problema):
    s = next((int(t) for t, v in serie.items() if v < umbral), None)
    if s is None:
        return None, f'el {nombre} no baja de {umbral} en ninguna semana: no hay {problema} apreciable'
    previo = f' (en la semana {s - 1} era {serie[s - 1]:.3f})' if s - 1 in serie.index else ' (desde la primera semana)'
    vuelve = [int(x) for x, v in serie.items() if x > s and v >= umbral]
    nota = (f'; se recupera por encima de {umbral} en la semana {vuelve[0]}, pero termina en {serie.iloc[-1]:.3f}' if vuelve else
            f' y ya no vuelve a superar {umbral} (termina en {serie.iloc[-1]:.3f})')
    return s, f'el {nombre} baja de {umbral} en la semana {s}: {serie[s]:.3f}{previo}{nota}'
semana_retraso, txt_r = inicio(ind['SPI = EV / PV'], 'SPI', 'retraso')
semana_sobrecoste, txt_c = inicio(ind['CPI = EV / AC'], 'CPI', 'sobrecoste')
if semana_retraso == semana_sobrecoste:
    simult = 'Fueron simultáneos.'
elif semana_retraso is None or semana_sobrecoste is None:
    simult = 'No fueron simultáneos: solo aparece uno de los dos problemas.'
else:
    primero = 'el sobrecoste' if semana_sobrecoste < semana_retraso else 'el retraso'
    simult = (f'No fueron simultáneos: {primero} apareció {plural(abs(semana_retraso - semana_sobrecoste), "semana")} antes. Un problema de plazo '
              'no implica uno de coste ni al revés (se puede ir despacio gastando poco, o gastar mucho para ir rápido).')
Markdown(f'''Se toma como señal un índice por debajo de {umbral} (unas centésimas por debajo de 1 son ruido de redondeo): {txt_r}; {txt_c}.

**Respuesta:** retraso desde la semana {semana_retraso}, sobrecoste desde la semana {semana_sobrecoste}. {simult}''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
t = int(curvas.index[-1])
PV, EV, AC = curvas.loc[t, 'PV'], curvas.loc[t, 'EV'], curvas.loc[t, 'AC']
CPI, SPI = EV / AC, EV / PV
EAC = BAC_total / CPI
pv_acumulado = pd.concat([pd.Series({0: 0.0}), curvas['PV']])
ES, texto_es = explica_es(pv_acumulado, EV, t)
SPI_t = ES / t
duracion_estimada = duracion_planificada / SPI_t
Markdown(f'''Coste: CPI = EV / AC = {EV:.1f} / {AC:.1f} = {CPI:.3f}; EAC = BAC / CPI = {BAC_total:.0f} / {CPI:.3f} = **{EAC:.1f}** k€ (VAC = {BAC_total:.0f} − {EAC:.1f} = {BAC_total - EAC:.1f} k€).

Plazo. {texto_es} SPI(t) = ES / {t} = {ES:.2f} / {t} = {SPI_t:.3f}; duración estimada = {duracion_planificada} / {SPI_t:.3f} = **{duracion_estimada:.1f}** semanas.

**Respuesta:** coste final previsto {EAC:.1f} k€ ({abs(BAC_total - EAC):.1f} k€ {"por encima" if EAC > BAC_total else "por debajo"} del presupuesto) y duración prevista {duracion_estimada:.1f} semanas ({abs(duracion_estimada - duracion_planificada):.1f} {"más" if duracion_estimada > duracion_planificada else "menos"} que las {duracion_planificada} planificadas).''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Las curvas agregan todas las actividades: no dicen *qué* actividades van mal ni si son críticas. Un SPI = 0,85 puede deberse a una actividad crítica retrasada (el plazo final peligra) o a varias no críticas aplazadas dentro de su holgura (el plazo no peligra): las mismas curvas, pronósticos opuestos. Igual con el coste: un CPI bajo puede venir de una actividad ya terminada (el daño está hecho y no se repetirá) o de una en curso (seguirá gastando de más). Por eso el informe debe acompañarse del detalle por actividad.
