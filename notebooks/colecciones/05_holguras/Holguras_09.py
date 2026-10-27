# ---
# title: Holguras 09
# subtitle: "10 actividades, 1 ruta crítica"
# coleccion: 05_holguras
# variante: tres-holguras-b
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
# Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta.

# %%
pregunta_1 = "1. Calcule para cada actividad sus fechas más tempranas y más tardías de inicio y fin, y sus holguras total, libre e independiente.  "
pregunta_2 = "2. Determine el camino crítico. ¿Cuántas rutas críticas tiene el proyecto?  "
pregunta_3 = "3. ¿Qué actividades pueden retrasarse sin afectar ni al plazo del proyecto ni al inicio más temprano de ninguna otra actividad? ¿Cuáles pueden retrasarse sin afectar al plazo pero obligando a retrasar a alguna sucesora? Justifíquelo con las holguras.  "
pregunta_4 = "4. La actividad E se retrasa tanto como permite su holgura total (3 semanas). Compruebe que el plazo no cambia y determine qué ocurre con las fechas y holguras de sus sucesoras.  "
pregunta_5 = "5. Suponga que las predecesoras de G terminan en su fecha más tardía. ¿Cuánto puede retrasarse entonces G sin afectar a nadie? Relaciónelo con su holgura independiente.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration
    ('A'     , '---'      , 6       ),
    ('B'     , '---'      , 4       ),
    ('C'     , '---'      , 8       ),
    ('D'     , 'C'        , 7       ),
    ('E'     , 'A,B,C'    , 5       ),
    ('F'     , 'B,C'      , 2       ),
    ('G'     , 'E,F'      , 1       ),
    ('H'     , 'D,F'      , 2       ),
    ('I'     , 'G,H'      , 8       ),
    ('J'     , 'I'        , 2       ),
], columns=['activity', 'predecessor', 'duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# El proyecto corresponde a la organización de un congreso científico; las actividades son las siguientes (las duraciones del cuadro están en semanas).

# %%
#| tbl-cap: "Descripción de las actividades"
#| label: tbl-descripciones
descripciones = pd.Series({
    'A': 'Definición del programa y fechas',
    'B': 'Reserva de la sede',
    'C': 'Convocatoria de comunicaciones',
    'D': 'Página web e inscripciones',
    'E': 'Contratación de ponentes invitados',
    'F': 'Revisión de los trabajos recibidos',
    'G': 'Elaboración del programa definitivo',
    'H': 'Contratación de catering y viajes',
    'I': 'Edición del libro de actas',
    'J': 'Preparación de salas y audiovisuales',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %%
_COLS_CAL = {'inicio_mas_temprano': 'ES', 'fin_mas_temprano': 'EF', 'inicio_mas_tardio': 'LS',
             'fin_mas_tardio': 'LF', 'H_total': 'H total', 'H_libre': 'H libre', 'H_independiente': 'H independiente',
             'duracion': 'duración'}


def _sucesoras(prelaciones):
    pred = prelaciones.immediate_predecessor
    return {a: sorted(s for s, ps in pred.items() if a in ps) for a in pred}


def _v(cal, a, col):
    return int(cal.loc[a, col])


def _verbo(n, singular, plural_):
    return singular if n == 1 else plural_


def _min_es_suc(cal, suc, a):
    """(mín ES de las sucesoras, expresión «mín(E: 12, G: 12)», «9 (ES de G)» o «19 (fin del proyecto)»)."""
    if not suc[a]:
        fin = int(cal['fin_mas_temprano'].max())
        return fin, f'{fin} (fin del proyecto)'
    es = {s: _v(cal, s, 'inicio_mas_temprano') for s in suc[a]}
    m = min(es.values())
    if len(es) == 1:
        return m, f'{m} (ES de {suc[a][0]})'
    return m, 'mín(' + ', '.join(f'{s}: {v}' for s, v in es.items()) + ')'


def respuesta_rutas(tabla_rutas, unidad='semana'):
    """Respuesta al «¿cuántas rutas críticas?» a partir de la tabla de explica_rutas."""
    dur = tabla_rutas['duración'].astype(float)
    T = int(round(dur.max()))
    criticas = tabla_rutas.loc[tabla_rutas['crítica'] == 'sí', 'ruta'].tolist()
    resto = tabla_rutas[tabla_rutas['crítica'] == 'no']
    acts = []
    for r in criticas:
        for a in r.split(' – '):
            if a not in acts:
                acts.append(a)
    n = len(criticas)
    txt = (f"**Respuesta:** el proyecto tiene **{plural(n, 'ruta crítica', 'rutas críticas')}**, "
           f"{lista_y('**' + r + '**' for r in criticas)}, "
           + (f"que suma {plural(T, unidad)}" if n == 1 else f"que suman {'ambas' if n == 2 else 'todas'} {plural(T, unidad)}")
           + f", la duración del proyecto. Actividades críticas (holgura total nula): {lista_y(sorted(acts))}.")
    if len(resto) == 1:
        segunda = int(round(float(resto['duración'].iloc[0])))
        txt += (f" La otra ruta, {resto['ruta'].iloc[0]}, suma {segunda} y queda a "
                f"{plural(T - segunda, unidad)} de ser crítica.")
    elif len(resto):
        d2 = resto['duración'].astype(float)
        segunda = int(round(d2.max()))
        sig = resto.loc[np.isclose(d2, d2.max()), 'ruta'].tolist()
        txt += (f" Las otras {plural(len(resto), 'ruta suma', 'rutas suman')} menos; "
                + (f"la más larga de ellas, {sig[0]} ({segunda}), queda" if len(sig) == 1
                   else f"las más largas de ellas, {lista_y(sig)} ({segunda}), quedan")
                + f" a {plural(T - segunda, unidad)} de ser crítica{'s' if len(sig) > 1 else ''}.")
    else:
        txt += ' Todas las rutas del grafo son críticas.'
    return Markdown(txt)


def respuesta_retrasables(calendario, prelaciones, unidad='semana'):
    """Qué actividades pueden retrasarse sin afectar a nadie, cuáles solo arrastrando a sus
    sucesoras y cuáles no admiten retraso, con las holguras sustituidas."""
    cal, suc = calendario, _sucesoras(prelaciones)
    criticas = [a for a in cal.index if _v(cal, a, 'H_total') == 0]
    libres = [a for a in cal.index if _v(cal, a, 'H_libre') > 0]
    arrastran = [a for a in cal.index if _v(cal, a, 'H_total') > _v(cal, a, 'H_libre')]
    L = ['**Respuesta:**', '']
    L.append(f"- **No admiten ningún retraso** sin alargar el plazo las críticas (holgura total 0): {lista_y(criticas)}.")
    L.append('- **Pueden retrasarse sin afectar ni al plazo ni al inicio más temprano de ninguna otra actividad** '
             'las de holgura libre positiva, como mucho su holgura libre (H libre = mín ES de las sucesoras − EF): '
             + ('; '.join(f"**{a}**, hasta {plural(_v(cal, a, 'H_libre'), unidad)} "
                          f"(H libre = {_min_es_suc(cal, suc, a)[1]} − {_v(cal, a, 'fin_mas_temprano')} = {_v(cal, a, 'H_libre')})"
                          for a in libres) if libres else 'ninguna') + '.')
    L.append('- **Pueden retrasarse sin afectar al plazo pero obligando a retrasar a alguna sucesora** las de holgura '
             'total mayor que la libre: ' + (lista_y(f'**{a}**' for a in arrastran) if arrastran else 'ninguna') + '.')
    for a in arrastran:
        ht, hl, ef = _v(cal, a, 'H_total'), _v(cal, a, 'H_libre'), _v(cal, a, 'fin_mas_temprano')
        mueve = []
        for s in suc[a]:
            es = _v(cal, s, 'inicio_mas_temprano')
            if es < ef + ht:
                t = es - ef
                mueve.append(f'cualquier retraso suyo mueve a {s} (ES de {s} = {es} = EF de {a})' if t == 0 else
                             f'un retraso de más de {plural(t, unidad)} mueve a {s} (ES de {s} = {es} = EF de {a} + {t})')
        L.append(f"    - {a}: H total = {ht} > H libre = {hl}. Puede retrasarse hasta {plural(ht, unidad)} sin alargar el plazo"
                 + (f' (y hasta {plural(hl, unidad)} sin mover a nadie)' if hl else '') + ', pero ' + '; '.join(mueve) + '.')
    return Markdown('\n'.join(L))


def analiza_retraso(proyecto, prelaciones, calendario, duraciones, X, h, unidad='semana'):
    """Retrasa X h unidades (alarga su duración) y devuelve (tabla antes/después de las actividades
    que cambian, texto de la respuesta, duraciones con el retraso)."""
    cal0, suc, pred = calendario, _sucesoras(prelaciones), prelaciones.immediate_predecessor
    d1 = duraciones.copy()
    d1[X] += h
    cal1 = proyecto.calendar(d1)
    cols = list(_COLS_CAL)[:6]
    cambian = [a for a in cal0.index if any(_v(cal0, a, c) != _v(cal1, a, c) for c in cols)]
    tabla = pd.concat({'antes': cal0.loc[cambian, cols].astype(int).rename(columns=_COLS_CAL),
                       'después': cal1.loc[cambian, cols].astype(int).rename(columns=_COLS_CAL)}, axis=1)
    tabla.index.name = 'actividad'
    us = plural(2, unidad).split(' ', 1)[1]
    T0, T1 = int(proyecto.duration(duraciones)), int(proyecto.duration(d1))
    finales = [a for a in cal1.index if not suc[a]]
    fines = ', '.join(f"{a}: {_v(cal1, a, 'fin_mas_temprano')}" for a in finales)
    L = [f"**Respuesta.** **El plazo no cambia**: el fin del proyecto es el mayor fin más temprano de las actividades "
         f"finales, máx({fines}) = **{T1}** {us}, igual que antes ({T0})."]
    hl = _v(cal0, X, 'H_libre')
    tx = (f"- **{X}** acaba en {_v(cal1, X, 'fin_mas_temprano')} en lugar de {_v(cal0, X, 'fin_mas_temprano')} "
          f"y su holgura total pasa de {h} a {_v(cal1, X, 'H_total')}: la ha consumido entera. ")
    if hl == 0:
        tx += 'Como su holgura libre era 0, todo el retraso empuja a sus sucesoras.'
    else:
        tx += (f"De las {plural(h, unidad)} de retraso, {hl} {_verbo(hl, 'cabe', 'caben')} en su holgura libre "
               f"y {h - hl} {_verbo(h - hl, 'empuja', 'empujan')} a sus sucesoras.")
    L.append(tx)
    movidas = [a for a in cambian if a != X and _v(cal1, a, 'inicio_mas_temprano') != _v(cal0, a, 'inicio_mas_temprano')]

    def cadena(a):
        camino = [a]
        while camino[-1] != X:
            b = camino[-1]
            es_b = _v(cal1, b, 'inicio_mas_temprano')
            p = [p for p in sorted(pred[b]) if (p == X or p in movidas) and _v(cal1, p, 'fin_mas_temprano') == es_b][0]
            camino.append(p)
        return ' → '.join(reversed(camino))

    def cambios(a, columnas):
        return ', '.join(f'{_COLS_CAL[c]} {_v(cal0, a, c)} → {_v(cal1, a, c)}' for c in columnas
                         if _v(cal0, a, c) != _v(cal1, a, c))

    if movidas:
        partes = []
        for a in movidas:
            p = f"{a} ({cambios(a, ['inicio_mas_temprano', 'H_total', 'H_libre'])}"
            p += ')' if a in suc[X] else f'; le llega en cadena: {cadena(a)})'
            partes.append(p)
        L.append(f"- **Se retrasa{'n' if len(movidas) > 1 else ''}** (cambia su inicio más temprano): " + '; '.join(partes) + '.')
    quietas = [s for s in suc[X] if s not in movidas]
    for s in quietas:
        es, ef_x = _v(cal1, s, 'inicio_mas_temprano'), _v(cal1, X, 'fin_mas_temprano')
        fija = [p for p in sorted(pred[s]) if p != X and _v(cal1, p, 'fin_mas_temprano') == es][0]
        L.append(f"- {s}, sucesora de {X}, no se mueve: su inicio más temprano ({es}) lo fija {fija}, que acaba en {es}, "
                 + (f"y {X} acaba ahora en {ef_x}, antes." if ef_x < es else
                    f"y {X} acaba ahora justo entonces, en {ef_x}: por eso {s} no se retrasa, pero {X} ya no tiene ningún margen."))
    pierden = [a for a in cambian if a != X and a not in movidas and _v(cal1, a, 'H_total') < _v(cal0, a, 'H_total')]
    if pierden:
        L.append(f"- **{_verbo(len(pierden), 'Pierde', 'Pierden')} holgura sin moverse** "
                 + '; '.join(f"{a} ({cambios(a, ['inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'H_libre'])})" for a in pierden) + ': '
                 f"{_verbo(len(pierden), 'va', 'van')} antes de {X} en la misma cadena y, como {X} dura ahora más, "
                 f"su fin más tardío se adelanta.")
    ganan = [a for a in cambian if a != X and a not in movidas and a not in pierden]
    if ganan:
        L.append(f"- **{_verbo(len(ganan), 'Gana', 'Ganan')} holgura libre** "
                 + '; '.join(f"{a} ({cambios(a, ['H_libre'])})" for a in ganan) + ': alguna de sus sucesoras empieza ahora más tarde.')
    nuevas = [a for a in cal0.index if _v(cal0, a, 'H_total') > 0 and _v(cal1, a, 'H_total') == 0]
    conservan = [a for a in cambian if _v(cal1, a, 'H_total') > 0]
    tx = (f"- **{_verbo(len(nuevas), 'Pasa', 'Pasan')} a ser crítica{'s' if len(nuevas) > 1 else ''}** (holgura total 0): "
          f"{lista_y(nuevas)}.")
    if conservan:
        tx += (f" {_verbo(len(conservan), 'Conserva', 'Conservan')} parte de su holgura: "
               + lista_y(f"{a} (H total {_v(cal1, a, 'H_total')})" for a in conservan) + '.')
    L.append(tx)
    r0 = {frozenset(r) for r in proyecto.critical_path(duraciones).values()}
    r1 = [r for r in proyecto.critical_path(d1).values()]
    rn = [' – '.join(ordena_ruta(r, cal1)) for r in r1 if frozenset(r) not in r0]
    L.append(f"- **{_verbo(len(rn), 'Aparece', 'Aparecen')} {plural(len(rn), 'ruta crítica nueva', 'rutas críticas nuevas')}** "
             f"({lista_y(rn)}): el proyecto pasa de {plural(len(r0), 'ruta crítica', 'rutas críticas')} a {len(r1)}:")
    return tabla, L[0] + '\n\n' + '\n'.join(L[1:]), d1


def respuesta_independiente(calendario, prelaciones, V, unidad='semana'):
    """Retraso disponible para V si sus predecesoras acaban en su fecha más tardía (holgura independiente)."""
    cal, suc = calendario, _sucesoras(prelaciones)
    ps = sorted(prelaciones.immediate_predecessor[V])
    lf = {p: _v(cal, p, 'fin_mas_tardio') for p in ps}
    inicio = max(lf.values())
    fin, fin_txt = _min_es_suc(cal, suc, V)
    if fin_txt.startswith('mín'):
        fin_txt += f' = {fin}'
    d, ef, es = _v(cal, V, 'duracion'), _v(cal, V, 'fin_mas_temprano'), _v(cal, V, 'inicio_mas_temprano')
    hi, hl = _v(cal, V, 'H_independiente'), _v(cal, V, 'H_libre')
    margen = fin - inicio - d
    lf_txt = (f'máx(' + ', '.join(f'LF de {p}: {v}' for p, v in lf.items()) + f') = {inicio}') if len(lf) > 1 else f'LF de {ps[0]} = {inicio}'
    txt = (f"Si {_verbo(len(ps), 'la predecesora', 'las predecesoras')} de {V} ({lista_y(ps)}) {_verbo(len(ps), 'acaba', 'acaban')} en su fecha más tardía, {V} no puede empezar hasta "
           f"{lf_txt}. Para no mover el inicio más temprano de ninguna sucesora debe acabar, como tarde, en {fin_txt}. "
           f"Le queda {fin} − {inicio} − {d} = {margen}")
    if margen < 0:
        txt = txt[:-len(str(margen))] + f"−{-margen} < 0: ningún margen, y la holgura independiente es máx(0, −{-margen}) = 0."
    else:
        txt += f", que es su holgura independiente."
    txt += (f" Su holgura libre, {fin} − {ef} = {hl}, supone en cambio que las predecesoras acaban lo antes posible "
            f"(entonces {V} empieza en {es} y acaba en {ef}).")
    txt += ((f"\n\n**Respuesta:** {V} puede retrasarse **{plural(hi, unidad)}** sin afectar a nadie aunque sus predecesoras "
             f"agoten su holgura" if hi else f"\n\n**Respuesta:** {V} **no puede retrasarse nada** (0 {plural(2, unidad).split()[1]}) sin "
             f"afectar a alguna sucesora si sus predecesoras agotan su holgura")
            + f": es su holgura independiente ({hi}), menor que su holgura libre ({hl}).")
    return Markdown(txt)


def respuesta_compartida(calendario, prelaciones, X, Y, h, unidad='semana'):
    """Razonamiento previo (solo con el calendario original) de si caben los dos retrasos."""
    cal, suc = calendario, _sucesoras(prelaciones)
    ef_x, es_y, ht_y = _v(cal, X, 'fin_mas_temprano'), _v(cal, Y, 'inicio_mas_temprano'), _v(cal, Y, 'H_total')
    g = es_y - ef_x
    empuje = max(0, h - g)
    queda = ht_y - empuje
    alarga = max(0, h - queda)
    txt = (f"**Respuesta:** {'no' if alarga else 'sí'}. {X} {'e' if Y.startswith('I') else 'y'} {Y} tienen la misma holgura "
           f"total ({h}) porque es la holgura de la cadena en la que están. {X} tiene holgura libre 0 "
           f"(H libre = {_min_es_suc(cal, suc, X)[1]} − {ef_x} = {_v(cal, X, 'H_libre')})")
    if g == 0:
        txt += (f" y {Y} empieza en cuanto acaba {X} (ES de {Y} = {es_y} = EF de {X}): cada {unidad} de retraso de {X} "
                f"retrasa el inicio de {Y} y le quita una {unidad} de holgura.")
    else:
        txt += (f"; {Y} empieza {plural(g, unidad)} después de que acabe {X} (ES de {Y} = {es_y} = EF de {X} + {g}): "
                f"solo el retraso de {X} que pase de {plural(g, unidad)} retrasa a {Y}.")
    txt += (f" Si {X} gasta {'su' if h == 1 else 'sus'} {plural(h, unidad)}, {Y} empieza {plural(empuje, unidad)} más tarde y le "
            f"{_verbo(queda, 'queda', 'quedan')} {ht_y} − {empuje} = {plural(queda, unidad)} de holgura total")
    if alarga:
        txt += (f"; si además {Y} se retrasa {plural(h, unidad)}, el proyecto se alarga {h} − {queda} = {plural(alarga, unidad)}.")
    else:
        txt += f", suficientes para su retraso de {plural(h, unidad)}."
    return Markdown(txt)


def respuesta_reparto(proyecto, calendario, duraciones, X, Y, h, unidad='semana'):
    """Holgura libre de X, independiente de Y y repartos (x, y) que respetan el plazo."""
    cal = calendario
    hl_x, hi_y = _v(cal, X, 'H_libre'), _v(cal, Y, 'H_independiente')
    T = int(proyecto.duration(duraciones))
    ymax = {}
    for x in range(h + 1):
        d = duraciones.copy()
        d[X] += x
        ymax[x] = _v(proyecto.calendar(d), Y, 'H_total') if int(proyecto.duration(d)) == T else None
    conj = 'e' if Y.startswith('I') else 'y'
    txt = (f"La holgura **libre** de {X} ({hl_x}) es lo que {X} puede retrasarse sin mover el inicio de {Y} ni quitarle "
           f"margen; la holgura **independiente** de {Y} ({hi_y}) es lo que {Y} puede retrasarse aunque {X} agote su "
           f"holgura y acabe lo más tarde posible.")
    if hl_x == 0 and hi_y == 0:
        txt += ' Las dos son 0: ninguna de las dos puede retrasarse sin gastar holgura que también necesita la otra.'
    if all(v is not None and x + v == h for x, v in ymax.items()):
        reparto = (f"si {X} se retrasa x y {Y} se retrasa y, el plazo se respeta mientras x + y ≤ {h}"
                   + (f" (por ejemplo, {(h + 1) // 2} para {X} {conj} {h // 2} para {Y}, o más para la que tenga más riesgo de retrasarse)"
                      if h > 1 else f": la única {unidad} se concede a una de las dos, la que más la necesite"))
    else:
        reparto = (f"con {X} retrasada x, a {Y} le queda de holgura total: "
                   + ', '.join(f'x = {x} → {v}' for x, v in ymax.items() if v is not None))
    txt += (f"\n\n**Respuesta:** la holgura libre (de {X}) y la independiente (de {Y}). Las {plural(h, unidad)} son un "
            f"recurso compartido de la cadena: {reparto}; lo que no cabe es que cada una disponga de {plural(h, unidad)} "
            f"por su cuenta.") if h > 1 else (f"\n\n**Respuesta:** la holgura libre (de {X}) y la independiente (de {Y}). "
            f"Hay una sola {unidad} de holgura compartida: {reparto}.")
    return Markdown(txt)


def respuesta_vigilancia(orden, n=4):
    """Las n actividades a vigilar según la ordenación, explicando los desempates."""
    crit = [a for a in orden.index if int(orden.loc[a, 'H_total']) == 0]
    elegidas = list(orden.index[:n])
    nc = len(crit)
    dur = lambda a: int(orden.loc[a, 'duracion'])
    if nc > n:
        txt = (f"Hay {nc} actividades críticas (holgura total 0), más de las {n} que se pueden vigilar; entre ellas se "
               f"desempata por duración (una actividad larga tiene más ocasiones de retrasarse): "
               + ', '.join(f'{a} ({dur(a)})' for a in crit) + '.')
        ult = dur(elegidas[-1])
        emp = [a for a in crit if dur(a) == ult]
        if any(a not in elegidas for a in emp):
            txt += (f" {lista_y(emp)} empatan con {plural(ult, 'semana')}: la última plaza puede ser para cualquiera "
                    f"de ellas (la tabla pone primero {elegidas[-1]}).")
    elif nc == n:
        txt = f"Las {n} críticas ({lista_y(crit)}) ocupan justo las {n} plazas."
    else:
        resto = [a for a in orden.index if a not in crit]
        txt = (f"Las {nc} críticas ({lista_y(crit)}) van primero; las {n - nc} plazas restantes son para las no críticas "
               f"de menor holgura total (a igualdad, menor holgura libre): "
               + ', '.join(f"{a} (H total {int(orden.loc[a, 'H_total'])}, H libre {int(orden.loc[a, 'H_libre'])})" for a in resto[:n - nc + 1])
               + '.')
        ult = elegidas[-1]
        clave = lambda a: (int(orden.loc[a, 'H_total']), int(orden.loc[a, 'H_libre']))
        emp = [a for a in resto if clave(a) == clave(ult)]
        if any(a not in elegidas for a in emp):
            txt += (f" {lista_y(emp)} empatan en las dos holguras: se desempata por duración (la más larga primero).")
    txt += f"\n\n**Respuesta:** las cuatro actividades a vigilar son **{lista_y(elegidas)}**."
    return Markdown(txt)


def respuesta_sin_margen(calendario, prelaciones, unidad='semana'):
    """No críticas con holgura libre nula: a quién mueven y por qué no son críticas."""
    cal, suc = calendario, _sucesoras(prelaciones)
    sin = [a for a in cal.index if _v(cal, a, 'H_total') > 0 and _v(cal, a, 'H_libre') == 0]
    L = []
    for a in sin:
        ef = _v(cal, a, 'fin_mas_temprano')
        pegadas = [s for s in suc[a] if _v(cal, s, 'inicio_mas_temprano') == ef]
        L.append(f"- **{a}**: H libre = {_min_es_suc(cal, suc, a)[1]} − {ef} = 0, pero H total = LS − ES = "
                 f"{_v(cal, a, 'inicio_mas_tardio')} − {_v(cal, a, 'inicio_mas_temprano')} = {_v(cal, a, 'H_total')}. "
                 f"Cualquier retraso suyo mueve a {lista_y(pegadas)} (empieza{'n' if len(pegadas) > 1 else ''} en {ef}, "
                 f"cuando acaba {a}), que lo absorbe{'n' if len(pegadas) > 1 else ''} con su holgura total ("
                 + ', '.join(f"{s}: {_v(cal, s, 'H_total')}" for s in pegadas) + ').')
    L.append(f"\n**Respuesta:** {lista_y(sin)} no {'admite' if len(sin) == 1 else 'admiten'} ni una {unidad} de retraso "
             f"sin mover a una sucesora (holgura libre 0), pero no {'es crítica' if len(sin) == 1 else 'son críticas'} porque "
             f"su holgura total es positiva: mientras el retraso no la supere, el plazo no cambia. Crítica significa que "
             f"el plazo depende de la actividad (holgura total 0), no que su retraso mueva a otra actividad.")
    return Markdown('\n'.join(L))


def respuesta_mas_libre(calendario, prelaciones, unidad='semana'):
    """Actividad (o actividades) con mayor holgura libre."""
    cal, suc = calendario, _sucesoras(prelaciones)
    m = int(cal['H_libre'].max())
    quien = [a for a in cal.index if _v(cal, a, 'H_libre') == m]
    txt = (f"**Respuesta:** {lista_y(quien)}, hasta **{plural(m, unidad)}**: {'es la' if len(quien) == 1 else 'son las'} "
           f"de mayor holgura libre (" + '; '.join(f"{a}: H libre = {_min_es_suc(cal, suc, a)[1]} − {_v(cal, a, 'fin_mas_temprano')} = {m}" for a in quien)
           + "). Ese retraso no mueve el inicio más temprano de ninguna otra actividad ni el plazo.")
    for a in quien:
        hi = _v(cal, a, 'H_independiente')
        if hi < m:
            txt += (f" Eso supone que las predecesoras de {a} acaban lo antes posible; si acabaran lo más tarde posible, "
                    f"solo le quedaría su holgura independiente ({hi}).")
    return Markdown(txt)


def respuesta_comprobaciones(c, d, prelaciones):
    """Relaciones que no cumple el calendario entregado, con los números."""
    L = []
    for a in c.index:
        ES, EF, LS, LF = (int(c.loc[a, k]) for k in ['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio'])
        Ht, Hl, da = int(c.loc[a, 'H_total']), int(c.loc[a, 'H_libre']), int(d[a])
        if EF != ES + da:
            L.append(f"- {a}: ES + d = {ES} + {da} = {ES + da} ≠ EF = {EF}")
        if LF != LS + da:
            L.append(f"- {a}: LS + d = {LS} + {da} = {LS + da} ≠ LF = {LF}")
        if Ht != LS - ES:
            L.append(f"- {a}: LS − ES = {LS} − {ES} = {LS - ES} ≠ H total = {Ht}")
        if not 0 <= Hl <= Ht:
            L.append(f"- {a}: H libre = {Hl} > H total = {Ht}")
        ps = sorted(prelaciones.immediate_predecessor[a])
        es_ok = max((int(c.loc[p, 'fin_mas_temprano']) for p in ps), default=0)
        if ES != es_ok:
            L.append(f"- {a}: máx EF de las predecesoras = {es_ok} ≠ ES = {ES}")
    malas = sorted({l.split(':')[0][2:] for l in L})
    return Markdown(f"**Respuesta:** los errores están en {lista_y(malas)}. Relaciones que fallan:\n\n" + '\n'.join(L))


def respuesta_errores(c, calendario, prelaciones, e1, e2):
    """Qué se ha calculado mal en cada actividad errónea y cuál es el valor correcto."""
    cal, suc = calendario, _sucesoras(prelaciones)
    hl_ok = _v(cal, e1, 'H_libre')
    t1 = (f"- **{e1}**: la holgura libre entregada ({int(c.loc[e1, 'H_libre'])}) es mayor que la total "
          f"({int(c.loc[e1, 'H_total'])}), lo que es imposible: la libre es la parte de la total que no afecta a las "
          f"sucesoras (H libre ≤ H total). Bien calculada, H libre = mín ES de las sucesoras − EF = "
          f"{_min_es_suc(cal, suc, e1)[1]} − {_v(cal, e1, 'fin_mas_temprano')} = **{hl_ok}**.")
    LS, LF, ES, EF = (int(c.loc[e2, k]) for k in ['inicio_mas_tardio', 'fin_mas_tardio', 'inicio_mas_temprano', 'fin_mas_temprano'])
    d, Ht = _v(cal, e2, 'duracion'), int(c.loc[e2, 'H_total'])
    t2 = (f"- **{e2}**: el inicio más tardío entregado ({LS}) no cumple LS = LF − d ({LF} − {d} = {LF - d}) ni "
          f"H total = LS − ES ({LS} − {ES} = {LS - ES}, pero la tabla dice {Ht}). En cambio LF, EF y H total sí son "
          f"coherentes entre sí (LF − EF = {LF} − {EF} = {LF - EF} = H total), así que el dato erróneo es LS: debe ser "
          f"LF − d = **{LF - d}**"
          + (f" ({e2} es crítica: su inicio más tardío coincide con el más temprano)." if Ht == 0 else '.'))
    return Markdown(t1 + '\n' + t2)


def respuesta_total_vs_libre(calendario, prelaciones, X, unidad='día'):
    """Por qué la holgura total de X es mayor que la libre, con las fechas de sus sucesoras."""
    cal, suc = calendario, _sucesoras(prelaciones)
    ef, ht, hl = _v(cal, X, 'fin_mas_temprano'), _v(cal, X, 'H_total'), _v(cal, X, 'H_libre')
    ls = {s: _v(cal, s, 'inicio_mas_tardio') for s in suc[X]}
    lf = min(ls.values())
    ls_txt = (f'mín(' + ', '.join(f'{s}: {v}' for s, v in ls.items()) + ')') if len(ls) > 1 else f'{lf} (LS de {suc[X][0]})'
    txt = (f"{X} acaba lo más pronto en {ef}. La holgura libre se mide contra el inicio **más temprano** de sus sucesoras: "
           f"H libre = {_min_es_suc(cal, suc, X)[1]} − {ef} = {hl}. La holgura total, contra su inicio **más tardío** "
           f"(el fin más tardío de {X} es el mínimo LS de sus sucesoras): H total = LF − EF = {ls_txt} − {ef} = {ht}.")
    txt += (f"\n\n**Respuesta:** {X} puede retrasarse hasta {plural(ht, unidad)} sin alargar el proyecto, pero "
            + (f"cualquier retraso suyo" if hl == 0 else f"pasados {plural(hl, unidad)}, el retraso")
            + f" retrasa el inicio más temprano de {lista_y(s for s in suc[X] if _v(cal, s, 'inicio_mas_temprano') < ef + ht)}, "
            f"que se come así parte de su propia holgura: la diferencia, {ht} − {hl} = {plural(ht - hl, unidad)}, es "
            f"holgura que {X} comparte con sus sucesoras.")
    return Markdown(txt)

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con tiempos y camino crítico"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# Con las fechas de cada actividad (ES, EF, LS, LF) y las de sus vecinas se definen las tres holguras:
#
# - **Total**: $H_t = LS - ES = LF - EF$. Retraso máximo sin alargar el proyecto, suponiendo que las predecesoras acaban lo antes posible y las sucesoras pueden empezar lo más tarde posible.
# - **Libre**: $H_l = \min_{s \in suc} ES_s - EF$. Retraso máximo sin mover el inicio más temprano de ninguna sucesora.
# - **Independiente**: $H_i = \max\bigl(0,\ \min_{s \in suc} ES_s - \max_{p \in pred} LF_p - D\bigr)$. Retraso disponible incluso si las predecesoras acaban lo más tarde posible y las sucesoras deben empezar lo antes posible.
#
# Siempre $H_t \ge H_l \ge H_i$.

# %%
#| tbl-cap: "Calendario y holguras del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# Con los números sustituidos, actividad a actividad (ES, EF, LS y LF son inicio y fin más tempranos y más tardíos; «pred.» y «suc.» las predecesoras y sucesoras inmediatas):

# %%
#| tbl-cap: "Cálculo paso a paso de fechas y holguras"
#| label: tbl-holgurasPaso
detalle_holguras(calendario, prelaciones)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Duración de cada ruta del grafo, sumando las duraciones de sus actividades en orden de ejecución; las críticas son las de mayor duración:

# %%
#| tbl-cap: "Todas las rutas del grafo, con su duración como suma"
#| label: tbl-rutas
tabla_rutas = explica_rutas(proyecto, datos_proyecto['duration'])
tabla_rutas.drop(columns='duración')

# %%
display(explica_camino_critico(proyecto, datos_proyecto['duration']))
respuesta_rutas(tabla_rutas)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Holguras de las actividades no críticas (las críticas tienen las tres holguras nulas):

# %%
#| tbl-cap: "Holguras de las actividades no críticas"
#| label: tbl-holgurasDistintas
con_holgura = calendario.index[calendario['H_total'] > 0].tolist()
calendario.loc[con_holgura, ['H_total', 'H_libre', 'H_independiente']].astype(int).rename(columns=_COLS_CAL).rename_axis('actividad')

# %%
respuesta_retrasables(calendario, prelaciones)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Un retraso de E igual a su holgura total equivale a alargar su duración en 3 semanas. Recalculamos el calendario con esa duración y comparamos *todas* las actividades cuyas fechas u holguras cambian (no sólo las sucesoras inmediatas de E: el retraso se propaga en cadena, y también cambian las fechas más tardías de las que van antes).

# %%
#| tbl-cap: "Actividades cuyas fechas u holguras cambian al retrasar la actividad"
#| label: tbl-comparacionRetraso
comparacion, texto_retraso, duraciones_retraso = analiza_retraso(proyecto, prelaciones, calendario, datos_proyecto['duration'], 'E', 3)
comparacion

# %%
display(Markdown(texto_retraso))
explica_camino_critico(proyecto, duraciones_retraso)

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
respuesta_independiente(calendario, prelaciones, 'G')
