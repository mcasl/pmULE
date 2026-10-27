# ---
# title: Reduccion 05
# subtitle: "8 actividades; deshacer la reducción más cara al relajar el plazo"
# coleccion: 10_reduccion
# variante: desaceleracion
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste unitario de reducción (k€/semana) se adjunta. El proyecto se había reducido cuatro semanas con el algoritmo de Ackoff-Sasieni cuando el cliente comunica que acepta la entrega una semana más tarde.

# %%
pregunta_1 = "1. Reduzca el proyecto cuatro semanas con el algoritmo de Ackoff-Sasieni y anote el sobrecoste de cada paso.  "
pregunta_2 = "2. Hay que devolver una semana. Un compañero propone deshacer el primer paso («fue el primero, es el más fácil de anular»). Otro propone deshacer el último. ¿Da lo mismo? Razónelo antes de calcular.  "
pregunta_3 = "3. Compruebe ambas opciones: duración resultante y dinero recuperado en cada caso. ¿Puede deshacerse un paso intermedio? ¿Qué regla general se deduce?  "
pregunta_4 = "4. ¿Por qué el sobrecoste marginal de cada semana de reducción es creciente (o al menos no decreciente) en Ackoff-Sasieni? ¿Qué forma tiene la curva coste-duración?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, ucr   
    ('A'     , '---'      , 2       , 1           , 2     ),
    ('B'     , '---'      , 5       , 4           , 5     ),
    ('C'     , 'A,B'      , 3       , 2           , 4     ),
    ('D'     , 'C'        , 8       , 6           , 1     ),
    ('E'     , 'B'        , 4       , 4           , np.inf),
    ('F'     , 'D'        , 3       , 3           , np.inf),
    ('G'     , 'D,E'      , 2       , 1           , 8     ),
    ('H'     , 'F,G'      , 4       , 2           , 8     ),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
duracion_normal = proyecto.duration(datos_proyecto['duration'])

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

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones normales y el camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Duración de cada ruta con las duraciones normales"
#| label: tbl-rutas
R = proyecto.path_matrix(dummies=False)
tabla_rutas(proyecto, datos_proyecto['duration'])

# %%
explica_camino_critico(proyecto, datos_proyecto['duration'])

# %% [markdown]
# En cada paso del algoritmo se acorta una semana la ruta crítica (o las rutas críticas) eligiendo, entre las actividades críticas con margen, la combinación de menor CUR que reduce *todas* las rutas críticas a la vez. La tabla muestra, para cada ruta, el CUR de sus actividades y su duración tras cada paso.

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'], reduction=4, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=4)

# %% [markdown]
# Lo que se compara en cada paso: las rutas críticas en ese momento, las opciones que acortan todas a la vez (una actividad común con margen, o una pareja con una actividad de cada ruta) con su coste, y la elegida (la más barata).

# %%
#| tbl-cap: "Opciones consideradas en cada paso de Ackoff-Sasieni"
#| label: tbl-ackoffPaso
detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)

# %% [markdown]
# Razonamiento de cada paso (los pasos consecutivos que acortan lo mismo se agrupan):

# %%
narra_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)

# %%
resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],
                        'coste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},
                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))
resumen['sobrecoste acumulado'] = resumen['coste del paso'].cumsum()
sobrecoste = float(resumen['coste del paso'].sum())
display(resumen)
Markdown(f'**Respuesta:** duración {duracion_normal} → {proyecto.duration(duraciones_reducidas)} semanas; sobrecoste de cada paso: '
         f'{"; ".join(f"paso {p}: {_num(c)} k€" for p, c in resumen["coste del paso"].items())}; total '
         f'{" + ".join(_num(c) for c in resumen["coste del paso"])} = **{_num(sobrecoste)} k€**.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# **Respuesta:** no da lo mismo. Deshacer un paso es devolver una semana a las actividades que se acortaron en él: el proyecto vuelve a alargarse una semana (las rutas críticas recuperan esa semana), pero el dinero que se recupera es el coste de *ese* paso. Como en Ackoff-Sasieni los pasos suelen ser cada vez más caros (apartado 4), deshacer el último recupera más dinero que deshacer el primero; el apartado siguiente lo comprueba con los números.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
filas = {}
for paso, acts in pasos.items():
    d2 = duraciones_reducidas.copy()
    for a in acts:
        d2[a] += 1
    filas[paso + 1] = {'actividades devueltas': ', '.join(acts), 'duración resultante': proyecto.duration(d2),
                       'dinero recuperado': float(sum(datos_proyecto.loc[acts, 'ucr']))}
pd.DataFrame(filas).T

# %%
costes_pasos = {p + 1: float(sum(datos_proyecto.loc[acts, 'ucr'])) for p, acts in pasos.items()}
acts_pasos = {p + 1: ' + '.join(acts) for p, acts in pasos.items()}
T_red = proyecto.duration(duraciones_reducidas)
durs = {p: int(f['duración resultante']) for p, f in pd.DataFrame(filas).T.iterrows()}
ultimo = max(costes_pasos)
maximo = max(costes_pasos.values())
mejores = [p for p, c in costes_pasos.items() if np.isclose(c, maximo)]
iguales = [p for p, t in durs.items() if t == T_red + 1]
if len(iguales) == len(durs):
    t_dur = f'Todos los pasos devuelven el proyecto a {T_red + 1} semanas, así que cualquiera cumple el nuevo plazo.'
else:
    otros = [f'el paso {p} lo deja en {t} semanas' for p, t in durs.items() if t != T_red + 1]
    t_dur = f'Los pasos {rangos(iguales)} devuelven el proyecto a {T_red + 1} semanas; {lista_y(otros)}.'
t_mejor = (f'el que más recupera es el paso {mejores[0]} ({acts_pasos[mejores[0]]}, {_num(maximo)} k€)' if len(mejores) == 1
           else f'los que más recuperan son los pasos {rangos(mejores)} ({_num(maximo)} k€ cada uno; da igual cuál se deshaga)')
Markdown(f'''**Respuesta:** deshacer el primer paso ({acts_pasos[1]}) recupera {_num(costes_pasos[1])} k€; deshacer el último ({acts_pasos[ultimo]}) recupera {_num(costes_pasos[ultimo])} k€. {t_dur} De todos, {t_mejor}. Un paso intermedio también puede deshacerse: lo que importa no es el orden en que se dieron sino su coste. Regla general: para relajar el plazo se deshacen primero las reducciones más caras (se recorre la tabla de Ackoff-Sasieni hacia atrás, comprobando que la duración vuelve a la deseada).''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# En cada paso el algoritmo elige la opción más barata disponible; las opciones baratas se agotan (las actividades llegan a su tope) o dejan de bastar (aparecen nuevas rutas críticas que exigen acortar varias actividades a la vez), así que normalmente el paso siguiente no es más barato. La curva coste-duración es entonces convexa: cada semana adicional de reducción cuesta al menos lo que la anterior, y por eso la decisión de hasta dónde reducir se toma comparando el coste marginal con lo que vale cada semana (penalización, coste indirecto, prima). No es un teorema: el algoritmo es voraz y, si un paso acorta dos actividades de una misma ruta, esa ruta puede dejar de ser crítica y el paso siguiente volver a ser barato.

# %%
costes = list(resumen['coste del paso'])
if all(b >= a for a, b in zip(costes, costes[1:])):
    t = f'aquí se cumple: los costes de los pasos son {" ≤ ".join(_num(c) for c in costes)} k€, y la curva coste-duración es convexa.'
else:
    bajan = [i + 2 for i, (a, b) in enumerate(zip(costes, costes[1:])) if b < a]
    t = (f'aquí no se cumple del todo: los costes de los pasos son {", ".join(_num(c) for c in costes)} k€ y el paso {rangos(bajan)} es más barato que el anterior; '
         'por eso hay que mirar la tabla completa antes de decidir.')
Markdown(f'**Respuesta:** {t}')
