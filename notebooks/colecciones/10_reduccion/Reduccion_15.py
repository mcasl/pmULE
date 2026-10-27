# ---
# title: Reduccion 15
# subtitle: "9 actividades; límite 3 operarios, reducir dos semanas (Ackoff sin recursos falla)"
# coleccion: 10_reduccion
# variante: crashing-recursos
# nivel: 3
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
# Considere el proyecto cuyo cuadro de prelaciones, duración normal y tope (semanas), operarios por actividad y coste unitario de reducción (k€/semana) se adjunta. Cada actividad necesita un número fijo de operarios durante toda su duración y la empresa dispone como máximo de 3 operarios en cada semana. Las actividades con CUR infinito no pueden acelerarse. Se quiere reducir la duración del proyecto **dos semanas** respetando el límite de operarios.

# %%
pregunta_1 = "1. Determine la duración del proyecto sin límite de recursos y con el límite de operarios (algoritmo de asignación). Indique la cadena crítica con recursos: qué actividades fijan la duración y qué esperas por recurso intervienen.  "
pregunta_2 = "2. Aplique el algoritmo de Ackoff-Sasieni ignorando los recursos para reducir dos semanas, aplique esas reducciones y replanifique con el límite de operarios. ¿Se ha conseguido reducir la duración del proyecto con recursos? ¿Por qué?  "
pregunta_3 = "3. Reduzca ahora dos semanas teniendo en cuenta el límite: pruebe en cada paso los candidatos (cortes de la cadena crítica con recursos, actividades sueltas y parejas) y elija el más barato que de verdad acorte el proyecto replanificado. Compare el sobrecoste con el del apartado anterior.  "
pregunta_4 = "4. Suponga que acelerar una actividad exige un operario más por cada semana recortada (modelo con recurso). ¿Cambia la solución?  "
pregunta_5 = "5. Alternativa: contratar un operario más (límite 4) sin acelerar nada. ¿Qué duración se obtiene? ¿Compensa frente a acelerar?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, operarios, cap_duration, ucr   
    ('A'     , '---'      , 2       , 2        , 1           , 3     ),
    ('B'     , '---'      , 7       , 1        , 4           , 3     ),
    ('C'     , 'A'        , 4       , 2        , 4           , np.inf),
    ('D'     , 'C'        , 6       , 1        , 4           , 6     ),
    ('E'     , 'C'        , 4       , 1        , 2           , 2     ),
    ('F'     , 'B,E'      , 1       , 2        , 1           , np.inf),
    ('G'     , 'F'        , 2       , 3        , 1           , 7     ),
    ('H'     , 'D,F'      , 1       , 1        , 1           , np.inf),
    ('I'     , 'G'        , 7       , 1        , 5           , 6     ),
], columns=['activity', 'predecessor', 'duration', 'operarios', 'cap_duration', 'ucr']).set_index('activity')
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
    'I': 'Instalación eléctrica',
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
limite = [3]
base = datos_proyecto[['predecessor', 'duration', 'operarios']]

# %%
from itertools import product as _product


def _num(x):
    return f'{float(x):g}'.replace('-', '−')


def rangos(numeros):
    """[1, 2, 3, 5] → «1 a 3 y 5»."""
    ns, partes = sorted(int(x) for x in numeros), []
    for x in ns:
        if partes and x == partes[-1][1] + 1:
            partes[-1][1] = x
        else:
            partes.append([x, x])
    return lista_y([f'{a}' if a == b else (f'{a} y {b}' if b == a + 1 else f'{a} a {b}') for a, b in partes])


def tabla_rutas(proyecto, valores, nombre='duración', decimales=0, base=None):
    """Cada ruta con sus actividades en orden de ejecución y su duración como suma término a término."""
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(base if base is not None else valores)
    filas = {}
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        total = float(sum(float(valores[a]) for a in acts))
        filas[r] = {'actividades': ' – '.join(acts),
                    f'{nombre} = suma': ' + '.join(f'{a}({float(valores[a]):.{decimales}f})' for a in acts)
                                        + f' = {total:.{decimales}f}',
                    nombre: total}
    t = pd.DataFrame(filas).T
    t[nombre] = t[nombre].astype(float)
    return t


def opciones_ackoff(conjuntos, d, topes, cur):
    """Combinaciones mínimas (una actividad con margen por ruta crítica, sin sobrantes) y su coste."""
    elegibles = [[a for a in sorted(s) if d[a] > topes[a] and np.isfinite(cur[a])] for s in conjuntos]
    if any(not e for e in elegibles):
        return []
    combos = {frozenset(c) for c in _product(*elegibles)}
    minimos = [c for c in combos if not any(o < c for o in combos)]
    return sorted(((tuple(sorted(c)), float(sum(cur[a] for a in c))) for c in minimos),
                  key=lambda x: (x[1], len(x[0]), x[0]))


def info_ackoff(proyecto, duraciones, topes, cur, pasos):
    """Estado de cada paso: duración, rutas críticas, opciones, elegidas y duraciones tras el paso."""
    R = proyecto.path_matrix(dummies=False)
    d = pd.Series(duraciones, dtype=float).copy()
    topes, cur = pd.Series(topes, dtype=float), pd.Series(cur, dtype=float)
    info = []
    for k in sorted(pasos):
        L = R @ d[R.columns]
        T = float(L.max())
        crit = [r for r in L.index if np.isclose(L[r], T)]
        conj = [set(R.columns[R.loc[r] == 1]) for r in crit]
        info.append(dict(T=T, crit=crit, opciones=opciones_ackoff(conj, d, topes, cur),
                         elegidas=tuple(sorted(pasos[k])), d_antes=d.copy()))
        for a in pasos[k]:
            d[a] -= 1
        info[-1]['d_despues'] = d.copy()
    return info


def describe_ruta(proyecto, r, base):
    """«Route_3 (A – C – F)»: actividades de la ruta en orden de ejecución."""
    R = proyecto.path_matrix(dummies=False)
    return f"{r} ({' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), proyecto.calendar(base)))})"


def rutas_sin_margen(proyecto, d, topes, cur):
    """Rutas críticas (con las duraciones d) en las que ninguna actividad puede acortarse más."""
    R = proyecto.path_matrix(dummies=False)
    d = pd.Series(d, dtype=float)
    L = R @ d[R.columns]
    crit = [r for r in L.index if np.isclose(L[r], L.max())]
    return [r for r in crit if not any(d[a] > topes[a] and np.isfinite(cur[a]) for a in R.columns[R.loc[r] == 1])]


def narra_ackoff(proyecto, duraciones, topes, cur, pasos, unidad='semana', moneda='k€', max_opciones=5):
    """Markdown con el razonamiento de cada paso (o grupo de pasos iguales) de Ackoff-Sasieni."""
    R = proyecto.path_matrix(dummies=False)
    topes, cur = pd.Series(topes, dtype=float), pd.Series(cur, dtype=float)
    orden = ordena_ruta(list(R.columns), proyecto.calendar(pd.Series(duraciones)))
    pos = {a: i for i, a in enumerate(orden)}
    ruta = lambda r: f"{r} ({' – '.join(sorted(R.columns[R.loc[r] == 1], key=pos.get))})"

    def criticas(dd):
        L = R @ dd[R.columns]
        return float(L.max()), [r for r in L.index if np.isclose(L[r], L.max())]

    def opc(acts, coste):
        if len(acts) == 1:
            return f'{acts[0]} ({_num(coste)})'
        return f"{' + '.join(acts)} ({' + '.join(_num(cur[a]) for a in acts)} = {_num(coste)})"

    info = info_ackoff(proyecto, duraciones, topes, cur, pasos)
    if not info:
        return Markdown('No se ha dado ningún paso: ninguna combinación de actividades con margen acorta todas las rutas críticas.')
    grupos = []
    for x in info:
        if grupos and grupos[-1][-1]['elegidas'] == x['elegidas'] and grupos[-1][-1]['crit'] == x['crit']:
            grupos[-1].append(x)
        else:
            grupos.append([x])
    lineas, acum, paso = [], 0.0, 1
    for g in grupos:
        x0, n = g[0], len(g)
        el = x0['elegidas']
        c = float(sum(cur[a] for a in el))
        acum += n * c
        crit, ops = x0['crit'], x0['opciones']
        cab = (f'**Paso {paso}**' if n == 1 else f'**Pasos {paso} y {paso + 1}**' if n == 2
               else f'**Pasos {paso} a {paso + n - 1}**')
        cab += f" ({plural(x0['T'], unidad)} → {plural(x0['T'] - n, unidad)})."
        t_rutas = (f'Ruta crítica: {ruta(crit[0])}.' if len(crit) == 1
                   else f'Rutas críticas: {lista_y([ruta(r) for r in crit])}.')
        t_ops = '; '.join(opc(a, cc) for a, cc in ops[:max_opciones])
        if len(ops) > max_opciones:
            t_ops += f' y {len(ops) - max_opciones} más, de coste igual o mayor'
        quien = 'la ruta crítica' if len(crit) == 1 else 'todas las rutas críticas a la vez'
        t_ops = f'Opciones que acortan {quien} (CUR en {moneda}/{unidad}): {t_ops}.'
        if len(crit) == 1:
            tipo = 'la actividad de la ruta crítica con margen y menor CUR'
        elif len(el) == 1:
            tipo = 'actividad común a todas las rutas críticas'
        else:
            tipo = 'una combinación que acorta todas las rutas críticas'
        t_el = f"Se elige **{' + '.join(el)}** ({tipo}), la opción más barata: {_num(c)} {moneda} por {unidad}"
        empates = [' + '.join(a) for a, cc in ops if np.isclose(cc, c) and a != el]
        if empates:
            t_el += (f" (empata con {lista_y(empates)}: cuestan lo mismo en este paso y el algoritmo "
                     "toma la primera que encuentra)")
        if len(crit) > 1 and len(el) > 1:
            comunes = [(a, cc) for a, cc in ops if len(a) == 1]
            if comunes:
                t_el += f"; la actividad común más barata, {comunes[0][0][0]}, cuesta {_num(comunes[0][1])}"
            else:
                t_el += '; no hay ninguna actividad común a todas las rutas críticas que tenga margen'
        d_fin = g[-1]['d_despues']
        motivos = []
        llegan = [a for a in el if d_fin[a] <= topes[a]]
        if llegan:
            motivos.append(f"{lista_y(f'{a} llega a su duración tope ({plural(topes[a], unidad)})' for a in llegan)}")
        T_n, crit_n = criticas(d_fin)
        nuevas = [r for r in crit_n if r not in crit]
        if nuevas:
            motivos.append(f"{lista_y([ruta(r) for r in nuevas])} {'alcanza' if len(nuevas) == 1 else 'alcanzan'} "
                           f"la duración del proyecto ({plural(T_n, unidad)}) y {'pasa' if len(nuevas) == 1 else 'pasan'} "
                           "a ser crítica" + ('' if len(nuevas) == 1 else 's'))
        dejan = [r for r in crit if r not in crit_n]
        if dejan:
            motivos.append(f"{lista_y(dejan)} se acorta{'' if len(dejan) == 1 else 'n'} dos {unidad}s en el paso "
                           f"y deja{'' if len(dejan) == 1 else 'n'} de ser crítica" + ('' if len(dejan) == 1 else 's'))
        if g is grupos[-1] and not motivos:
            motivos.append('se alcanza la reducción pedida')
        t_red = f"Se {'reduce' if n == 1 else 'reducen'} {plural(n, unidad)}" + (f": {'; '.join(motivos)}." if motivos else '.')
        t_coste = f"Sobrecoste: {n} × {_num(c)} = {_num(n * c)} {moneda}; acumulado: **{_num(acum)} {moneda}**."
        lineas.append(f'- {cab} {t_rutas} {t_ops} {t_el}. {t_red} {t_coste}')
        paso += n
    T_ini, T_fin = info[0]['T'], info[-1]['T'] - 1
    costes = [_num(sum(cur[a] for a in x['elegidas'])) for x in info]
    cierre = (f"**Resultado:** la duración pasa de {plural(T_ini, unidad)} a {plural(T_fin, unidad)} con un sobrecoste "
              f"de {' + '.join(costes)} = **{_num(acum)} {moneda}**.")
    T_n, crit_n = criticas(info[-1]['d_despues'])
    conj = [set(R.columns[R.loc[r] == 1]) for r in crit_n]
    if not opciones_ackoff(conj, info[-1]['d_despues'], topes, cur):
        bloq = [r for r, s in zip(crit_n, conj)
                if not any(info[-1]['d_despues'][a] > topes[a] and np.isfinite(cur[a]) for a in s)]
        cierre += (f" No puede reducirse más: en {lista_y([ruta(r) for r in bloq])} ninguna actividad tiene margen "
                   "(todas están en su duración tope o no son acelerables), así que esa ruta fija la duración mínima.")
    return Markdown('\n'.join(lineas) + '\n\n' + cierre)

# %%
from itertools import combinations as _comb


def _predecesoras(datos):
    return {a: [p.strip() for p in str(datos.loc[a, 'predecessor']).split(',') if p.strip() not in ('', '---')]
            for a in datos.index}


def explica_cadena(cadena, res, datos, unidad='semana'):
    """Cadena crítica con recursos en orden de ejecución (→ precedencia, ⇒ recurso), su suma y las esperas."""
    preds = _predecesoras(datos)
    ini, fin = res.inicios, res.fines
    acts = sorted(cadena['actividades'], key=lambda a: (ini[a], fin[a], a))
    arcos = {(f['de'], f['a']): f['motivo'] for _, f in cadena['arcos'].iterrows()}
    texto = f'{acts[0]} ({int(ini[acts[0]])}–{int(fin[acts[0]])})'
    for x, y in zip(acts, acts[1:]):
        texto += (' ⇒ ' if arcos.get((x, y)) == 'recurso' else ' → ') + f'{y} ({int(ini[y])}–{int(fin[y])})'
    duraciones = [int(fin[a] - ini[a]) for a in acts]
    simple = (all((x, y) in arcos for x, y in zip(acts, acts[1:])) and int(ini[acts[0]]) == 0
              and int(fin[acts[-1]]) == int(res.duracion))
    suma = f" Duración = {' + '.join(str(d) for d in duraciones)} = {int(res.duracion)} {unidad}s." if simple else ''
    esperas = []
    for (x, y), motivo in sorted(arcos.items(), key=lambda kv: (ini[kv[0][1]], kv[0])):
        if motivo != 'recurso':
            continue
        listo = max([int(fin[p]) for p in preds[y]], default=0)
        espera = int(ini[y]) - listo
        if espera > 0:
            esperas.append(f'{y} podría empezar en {listo} (cuando terminan sus predecesoras), pero espera '
                           f'{plural(espera, unidad)}, hasta {int(ini[y])}, a que {x} libere los operarios')
        else:
            esperas.append(f'{y} empieza en {int(ini[y])}, justo cuando {x} le libera los operarios')
    t_esp = ('Esperas por recurso: ' + '; '.join(esperas) + '.') if esperas else \
        'No hay arcos de recurso: la cadena es un camino del grafo.'
    return f'{texto} (→ precedencia, ⇒ espera por recurso).{suma} {t_esp}'


def candidatos_crashing(proyecto, base, limite, topes, cur, elegidos, extra=0, unidad='semana'):
    """Para cada paso: cadena crítica con recursos, cada candidato (actividades reducibles de la cadena,
    sueltas y por parejas) con su coste y la duración que da al replanificar, y el elegido."""
    red = {a: 0 for a in base.index}
    for paso, el in enumerate(elegidos, 1):
        el = tuple(sorted(el))
        datos = base.copy()
        datos['duration'] = [int(base.loc[a, 'duration']) - red[a] for a in base.index]
        datos['operarios'] = [base.loc[a, 'operarios'] + extra * red[a] for a in base.index]
        cad = proyecto.cadena_critica_recursos(datos, 'duration', ['operarios'], limite)
        T = int(cad['duracion'])
        reducibles = [a for a in sorted(cad['actividades']) if datos.loc[a, 'duration'] > topes[a] and np.isfinite(cur[a])]
        cands = [(a,) for a in reducibles] + list(_comb(reducibles, 2))
        if el not in cands:
            cands.append(el)
        filas = []
        for c in cands:
            d2 = datos.copy()
            for a in c:
                d2.loc[a, 'duration'] -= 1
                d2.loc[a, 'operarios'] += extra
            coste = float(sum(cur[a] for a in c))
            if (d2['operarios'] > limite[0]).any():
                T2, txt = np.nan, f'necesita más de {limite[0]} operarios: no cabe'
            else:
                T2 = int(proyecto.asignar_multi(d2, 'duration', ['operarios'], limite, dibujar=False).duracion)
                txt = f'acorta a {T2}' if T2 < T else 'no acorta'
            filas.append({'candidato': ' + '.join(c),
                          'coste (k€)': _num(coste) if len(c) == 1 else f"{' + '.join(_num(cur[a]) for a in c)} = {_num(coste)}",
                          'duración replanificada': '—' if np.isnan(T2) else T2, 'resultado': txt,
                          'elegido': '✔' if c == el else '', '_c': coste})
        tabla = pd.DataFrame(filas).sort_values(['_c', 'candidato']).set_index('candidato')
        baratos = tabla[tabla['_c'] < tabla.loc[' + '.join(el), '_c'] - 1e-9]
        display(Markdown(f"**Paso {paso}** (duración con recursos {plural(T, unidad)}). Cadena crítica con recursos: "
                         f"{lista_y(sorted(cad['actividades']))}; reducibles en ella: {lista_y(reducibles)}. Candidatos, de más "
                         "barato a más caro:"))
        display(tabla.drop(columns='_c'))
        t = f"Se elige **{' + '.join(el)}** ({_num(tabla.loc[' + '.join(el), '_c'])} k€), el candidato más barato que de verdad acorta el proyecto replanificado"
        if len(baratos):
            t += f"; los más baratos ({lista_y(list(baratos.index))}) no lo acortan o no caben en el límite"
        display(Markdown(t + '.'))
        for a in el:
            red[a] += 1

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con las duraciones normales y el camino crítico (sin límite de operarios)"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario sin límite de recursos"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion_libre = proyecto.duration(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %%
#| fig-cap: "Gantt de inicios más tempranos con los operarios de cada actividad (fila Total: carga semanal)"
#| label: fig-ganttInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Carga de operarios frente al límite"
#| label: fig-cargasInicial
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite)

# %%
cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)
exceso = cargas[cargas > limite[0]]
def _activas(p):
    return [a for a in datos_proyecto.index if calendario.loc[a, 'inicio_mas_temprano'] < p <= calendario.loc[a, 'fin_mas_temprano']]
detalle_exceso = '; '.join(f'semana {int(t)}: ' + ' + '.join(f"{a}({int(datos_proyecto.loc[a, 'operarios'])})" for a in _activas(int(t))) + f' = {int(c)}'
                           for t, c in exceso.items())
display(explica_camino_critico(proyecto, datos_proyecto['duration']))
Markdown(f'Sin límite el proyecto dura {duracion_libre} semanas, pero con los inicios más tempranos se superan los {limite[0]} operarios disponibles '
         f'({detalle_exceso} operarios): hay que reprogramar.')

# %% [markdown]
# El algoritmo de asignación recorre las semanas una a una. En cada semana son *candidatas* las actividades no empezadas cuyas predecesoras han terminado; se ordenan por prioridad (menor inicio más tardío en el calendario sin recursos, después menor duración) y cada una empieza si, en todas las semanas que ocuparía, los operarios ya comprometidos más los suyos no superan el límite; si no, espera. La traza muestra la decisión de cada semana y, entre paréntesis, el recurso que bloquea a las que esperan.

# %%
res0 = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios (filas 💤: esperas por falta de operarios)"
#| label: fig-ganttAsignado
pinta_tikz(res0.dibujos['operarios'])

# %%
#| tbl-cap: "Calendario con recursos: inicio, fin y espera de cada actividad"
#| label: tbl-calendarioRecursos
calendario_recursos = pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res0.inicios,
                                    'fin con límite': res0.fines, 'espera (semanas)': res0.esperas}).fillna(0).astype(int)
calendario_recursos

# %%
#| tbl-cap: "Carga de operarios semana a semana frente al límite"
#| label: tbl-cargasRecursos
pd.concat([res0.cargas['operarios'].rename('operarios ocupados'), res0.limites['operarios'].rename('límite')], axis=1).T

# %%
cadena = proyecto.copy().cadena_critica_recursos(base.copy(), 'duration', ['operarios'], limite)
Markdown(f'Sin límite: {duracion_libre} semanas. Con {limite[0]} operarios: {res0.duracion} semanas ({res0.duracion - duracion_libre} más). '
         f'Actividades de la cadena crítica con recursos: {lista_y(sorted(cadena["actividades"]))}; sus arcos:')

# %%
#| tbl-cap: "Arcos de la cadena crítica con recursos"
#| label: tbl-cadena
cadena['arcos'].rename(columns={'de': 'de', 'a': 'a', 'motivo': 'motivo (precedencia o recurso)'})

# %% [markdown]
# La cadena crítica con recursos se lee hacia atrás desde la última actividad: cada arco une una actividad con otra que empieza justo cuando ella termina, bien porque es su predecesora (*precedencia*), bien porque le libera los operarios que esperaba (*recurso*). Los arcos de recurso son esperas, no prelaciones: la duración la fija esa cadena (precedencias más esperas), no el camino crítico del grafo.

# %%
Markdown(f'**Respuesta:** sin límite de recursos el proyecto dura {duracion_libre} semanas; con {limite[0]} operarios, {res0.duracion} semanas '
         f'({plural(res0.duracion - duracion_libre, "semana")} más). La cadena crítica con recursos es '
         f'{explica_cadena(cadena, res0, datos_proyecto)}')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Ackoff-Sasieni sólo mira el grafo: acorta en cada paso, una semana, la combinación más barata de actividades que reduce todas las rutas críticas *sin recursos*. La tabla muestra, para cada ruta, el CUR de sus actividades y su duración tras cada paso.

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni sin tener en cuenta los recursos"
#| label: tbl-ackoff
tabla_ack, pasos_ack, dur_ack, margen = proyecto.ackoff(durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'], costs=datos_proyecto['ucr'], reduction=2, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %%
#| tbl-cap: "Opciones consideradas en cada paso"
#| label: tbl-ackoffPaso
detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos_ack)

# %%
resumen_ack = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos_ack.values()],
                            'coste del paso': [float(sum(datos_proyecto.loc[acts, 'ucr'])) for acts in pasos_ack.values()]},
                           index=pd.Index(range(1, len(pasos_ack) + 1), name='paso'))
coste_ack = float(resumen_ack['coste del paso'].sum())
resumen_ack

# %% [markdown]
# Aplicadas esas reducciones, se vuelve a programar con el límite de operarios:

# %%
res_ack = proyecto.copy().asignar_multi(base.assign(duration=dur_ack.reindex(base.index).astype(int)), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios tras las reducciones de Ackoff-Sasieni"
#| label: fig-ganttAckoff
pinta_tikz(res_ack.dibujos['operarios'])

# %%
#| tbl-cap: "Duración e inicio de cada actividad antes y después de aplicar Ackoff-Sasieni (con el límite)"
#| label: tbl-inicioAckoff
pd.DataFrame({'duración normal': datos_proyecto['duration'], 'duración Ackoff': dur_ack.reindex(datos_proyecto.index).astype(int),
              'inicio con límite (antes)': res0.inicios, 'inicio con límite (después)': res_ack.inicios,
              'fin con límite (antes)': res0.fines, 'fin con límite (después)': res_ack.fines}).T

# %%
datos_ack = base.assign(duration=dur_ack.reindex(base.index).astype(int))
cadena_ack = proyecto.copy().cadena_critica_recursos(datos_ack.copy(), 'duration', ['operarios'], limite)
acortadas_ack = sorted(set(a for acts in pasos_ack.values() for a in acts))
fuera = [a for a in acortadas_ack if a not in cadena['actividades']]
dentro = [a for a in acortadas_ack if a in cadena['actividades']]
if fuera and not dentro:
    por_que = (f'{lista_y(fuera)} no {"está" if len(fuera) == 1 else "están"} en la cadena crítica con recursos del apartado 1, así que acortar'
               f'{"la" if len(fuera) == 1 else "las"} adelanta otras actividades, pero no lo que fija el plazo con recursos')
else:
    por_que = (f'aunque {lista_y(dentro)} {"está" if len(dentro) == 1 else "están"} en la cadena crítica con recursos, al replanificar cambian las '
               'esperas por operarios y otra cadena pasa a fijar el plazo')
costes_ack = ' + '.join(_num(c) for c in resumen_ack['coste del paso'])
logrado = 'no' if res_ack.duracion >= res0.duracion else 'solo en parte'
Markdown(f'''**Respuesta:** {logrado}. Ackoff-Sasieni sin recursos acorta {lista_y(acortadas_ack)} con un coste de {costes_ack} = {_num(coste_ack)} k€ y el grafo baja a {proyecto.duration(dur_ack)} semanas sobre el papel, pero al replanificar con {limite[0]} operarios el proyecto dura {res_ack.duracion} semanas (antes {res0.duracion}): {por_que}. Con las duraciones de Ackoff la cadena que fija el plazo es {explica_cadena(cadena_ack, res_ack, datos_proyecto)} El dinero se gasta sin acortar lo que de verdad fija el plazo.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# `crashing_recursos` repite en cada paso cuatro operaciones: (1) programa con el límite, (2) halla la cadena crítica con recursos, (3) evalúa como candidatos las actividades reducibles de esa cadena una a una, por parejas y el corte de coste mínimo de la cadena, **replanificando** con cada candidato, y (4) se queda con el más barato que de verdad acorta la duración replanificada. La tabla muestra, paso a paso, la cadena crítica, lo que habría elegido Ackoff-Sasieni sin recursos, cuántos candidatos se evaluaron y cuántas reducciones de una sola actividad no acortaron nada.

# %%
#| tbl-cap: "Crashing con recursos limitados, paso a paso"
#| label: tbl-crashing
cr = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], limite,
                                       datos_proyecto['cap_duration'], datos_proyecto['ucr'], reduccion=2)
COLUMNAS_CR = {'cadena_critica': 'cadena crítica con recursos', 'ackoff_sin_recursos': 'Ackoff sin recursos elegiría',
               'candidatos_evaluados': 'candidatos evaluados', 'reducciones_sin_efecto': 'actividades sueltas que no acortan',
               'actividades': 'se acorta', 'coste_paso': 'coste del paso', 'coste_acumulado': 'coste acumulado',
               'duracion_antes': 'duración antes', 'duracion_despues': 'duración después'}
cr.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'reducciones_sin_efecto', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']].rename(columns=COLUMNAS_CR)

# %% [markdown]
# Detalle de cada paso: todos los candidatos evaluados (actividades reducibles de la cadena crítica con recursos, sueltas y por parejas), su coste y la duración que resulta al replanificar con el límite de operarios.

# %%
candidatos_crashing(proyecto, base, limite, datos_proyecto['cap_duration'], datos_proyecto['ucr'],
                    [acts.split(', ') for acts in cr.pasos['actividades']])

# %%
#| tbl-cap: "Duraciones finales y semanas recortadas a cada actividad"
#| label: tbl-duracionesCrashing
pd.DataFrame({'duración normal': datos_proyecto['duration'], 'duración tope': datos_proyecto['cap_duration'],
              'duración final': cr.duraciones.reindex(datos_proyecto.index).astype(int), 'recortado': cr.reducciones.reindex(datos_proyecto.index).astype(int),
              'CUR': datos_proyecto['ucr']}).T

# %%
datos_cr = base.assign(duration=cr.duraciones.reindex(base.index).astype(int))
res_cr = proyecto.copy().asignar_multi(datos_cr, 'duration', ['operarios'], limite, report=False, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios tras el crashing con recursos"
#| label: fig-ganttCrashing
pinta_tikz(res_cr.dibujos['operarios'])

# %%
recortes = cr.reducciones[cr.reducciones > 0]
lista_cr = lista_y([a + ' ' + plural(x, 'semana') for a, x in recortes.items()])
costes_cr = ' + '.join(_num(c) for c in cr.pasos['coste_paso'])
logro = ('se consigue la reducción pedida' if cr.duracion_final <= res0.duracion - 2
         else f'solo se consigue {plural(res0.duracion - cr.duracion_final, "semana")} (parada: «{cr.motivo_parada}»)')
Markdown(f'**Respuesta:** se acortan {lista_cr}: la duración con recursos pasa de {res0.duracion} a {cr.duracion_final} semanas ({logro}) con un sobrecoste de '
         f'{costes_cr} = **{_num(cr.coste_total)} k€**, frente a los {_num(coste_ack)} k€ de Ackoff sin recursos, que no acortaban nada (el proyecto quedaba en {res_ack.duracion}). '
         'Cuesta más, pero es dinero que sí acorta el proyecto: se acortan actividades de la cadena crítica con recursos y se comprueba en cada paso, replanificando, que el proyecto termina antes.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# En el modelo *con recurso*, acelerar una actividad consume operarios: cada semana recortada añade un operario a su demanda, de modo que la reducción puede chocar con el mismo límite que intenta esquivar.

# %%
#| tbl-cap: "Crashing con recursos cuando acelerar consume un operario por semana recortada"
#| label: tbl-crashingRecurso
extra = {a: {'operarios': 1} for a in datos_proyecto.index if np.isfinite(datos_proyecto.loc[a, 'ucr'])}
cr_b = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], limite,
                                         datos_proyecto['cap_duration'], datos_proyecto['ucr'], reduccion=2,
                                         demanda_extra=extra)
cr_b.pasos[['cadena_critica', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']].rename(columns=COLUMNAS_CR)

# %%
candidatos_crashing(proyecto, base, limite, datos_proyecto['cap_duration'], datos_proyecto['ucr'],
                    [acts.split(', ') for acts in cr_b.pasos['actividades']], extra=1)

# %%
#| tbl-cap: "Demanda de operarios de cada actividad tras las reducciones (modelo con recurso)"
#| label: tbl-demandaRecurso
pd.DataFrame({'operarios (normal)': datos_proyecto['operarios'], 'operarios (tras recortar)': cr_b.demanda['operarios'].reindex(datos_proyecto.index),
              'recortado': cr_b.reducciones.reindex(datos_proyecto.index).astype(int)}).T

# %%
recortes_b = cr_b.reducciones[cr_b.reducciones > 0]
lista_b = lista_y([a + ' ' + plural(x, 'semana') for a, x in recortes_b.items()]) if len(recortes_b) else 'ninguna actividad'
mismas = bool((cr_b.reducciones == cr.reducciones).all())
if cr_b.duracion_final <= res0.duracion - 2 and mismas:
    t = (f'no cambia: se acortan las mismas actividades ({lista_b}), con el mismo coste ({_num(cr_b.coste_total)} k€) y la misma duración final '
         f'({cr_b.duracion_final} semanas); el operario extra cabe en las semanas en que trabajan las actividades aceleradas.')
elif cr_b.duracion_final <= res0.duracion - 2:
    t = (f'sí cambia: se alcanza igualmente la reducción pedida ({cr_b.duracion_final} semanas), pero acortando {lista_b} en lugar de {lista_cr}, '
         f'con un sobrecoste de {_num(cr_b.coste_total)} k€ (modelo con dinero: {_num(cr.coste_total)} k€). Los candidatos que antes servían dejan de '
         'hacerlo porque el operario adicional no cabe en el límite o retrasa a otra actividad (tablas anteriores).')
else:
    t = (f'sí cambia: acortando {lista_b} solo se llega a {cr_b.duracion_final} semanas, {plural(res0.duracion - cr_b.duracion_final, "semana")} de las 2 pedidas, '
         f'con {_num(cr_b.coste_total)} k€; después ninguna reducción de una semana acorta el proyecto, porque el operario extra que exige acelerar no cabe '
         'en el límite o retrasa a otra actividad. Sin ampliar el recurso no se alcanza lo pedido.')
Markdown(f'**Respuesta:** {t}')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
mas = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], [limite[0] + 1], report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con un operario más, sin acelerar ninguna actividad"
#| label: fig-ganttMasOperarios
pinta_tikz(mas.dibujos['operarios'])

# %%
objetivo = res0.duracion - 2
if mas.duracion <= objetivo:
    t = (f'con {limite[0] + 1} operarios y sin acelerar nada el proyecto dura {mas.duracion} semanas (antes {res0.duracion}): ahorra '
         f'{plural(res0.duracion - mas.duracion, "semana")} y ya cumple el objetivo de {objetivo} semanas. Acelerar con {limite[0]} operarios cuesta '
         f'{_num(cr.coste_total)} k€, así que compensa contratar si el operario adicional cuesta menos de {_num(cr.coste_total)} k€ en las {mas.duracion} semanas '
         f'del proyecto, es decir, menos de {_num(cr.coste_total)}/{mas.duracion} = {cr.coste_total / mas.duracion:.2f} k€ por semana.')
else:
    cr_mas = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], [limite[0] + 1],
                                               datos_proyecto['cap_duration'], datos_proyecto['ucr'], objetivo=objetivo)
    rec_mas = cr_mas.reducciones[cr_mas.reducciones > 0]
    t = (f'con {limite[0] + 1} operarios y sin acelerar nada el proyecto dura {mas.duracion} semanas (antes {res0.duracion}): ahorra '
         f'{plural(res0.duracion - mas.duracion, "semana")}, pero no llega al objetivo de {objetivo} semanas, así que por sí solo no basta. ')
    if cr_mas.duracion_final <= objetivo:
        t += (f'Para llegar habría que acelerar además {lista_y([a + " " + plural(x, "semana") for a, x in rec_mas.items()])} ({_num(cr_mas.coste_total)} k€, '
              f'frente a {_num(cr.coste_total)} k€ acelerando con {limite[0]} operarios): contratar compensa si el operario adicional durante las {objetivo} '
              f'semanas cuesta menos que la diferencia, {_num(cr.coste_total - cr_mas.coste_total)} k€.')
    else:
        t += f'Ni acelerando con {limite[0] + 1} operarios se llega al objetivo (se queda en {cr_mas.duracion_final} semanas).'
Markdown(f'**Respuesta:** {t}')
