# ---
# title: Reduccion 02
# subtitle: "8 actividades, costes indirectos de 9 k€/semana"
# coleccion: 10_reduccion
# variante: costes-indirectos
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
# Considere el proyecto cuyo cuadro de prelaciones se adjunta, con la duración normal y la duración tope de cada actividad (semanas) y el coste directo (k€) de ejecutarla en cada una. Además, el proyecto tiene unos costes indirectos (oficina, alquileres, seguros) de 9 k€ por semana de duración.

# %%
pregunta_1 = "1. Calcule el coste unitario de reducción (CUR) de cada actividad.  "
pregunta_2 = "2. Determine la duración normal del proyecto y su coste total (directo más indirecto).  "
pregunta_3 = "3. Aplique el algoritmo de Ackoff-Sasieni hasta alcanzar la duración tope del proyecto, anotando en cada paso el sobrecoste directo.  "
pregunta_4 = "4. Construya la curva de coste total frente a duración y determine la duración de coste mínimo. ¿Qué regla de parada se deduce comparando el CUR del paso con el coste indirecto semanal?  "
pregunta_5 = "5. ¿Por qué no se puede reducir el proyecto por debajo de su duración tope aunque queden actividades con margen de reducción?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, cap_duration, cost_normal, cost_cap
    ('A'     , '---'      , 3       , 1           , 90         , 98      ),
    ('B'     , '---'      , 6       , 5           , 30         , 35      ),
    ('C'     , '---'      , 2       , 1           , 40         , 42      ),
    ('D'     , 'B'        , 6       , 3           , 60         , 84      ),
    ('E'     , 'A,C,D'    , 5       , 3           , 90         , 108     ),
    ('F'     , 'E'        , 4       , 2           , 60         , 64      ),
    ('G'     , 'E'        , 8       , 7           , 20         , 28      ),
    ('H'     , 'G'        , 8       , 6           , 70         , 72      ),
], columns=['activity', 'predecessor', 'duration', 'cap_duration', 'cost_normal', 'cost_cap']).set_index('activity')
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
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

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
# $CUR = \dfrac{\text{coste tope} - \text{coste normal}}{\text{duración normal} - \text{duración tope}}$. Las actividades cuya duración tope coincide con la normal no pueden acortarse: su CUR es infinito.

# %%
reducible = datos_proyecto['duration'] > datos_proyecto['cap_duration']
datos_proyecto['ucr'] = np.where(reducible,
                                 (datos_proyecto['cost_cap'] - datos_proyecto['cost_normal']) / (datos_proyecto['duration'] - datos_proyecto['cap_duration']).replace(0, np.nan),
                                 np.inf)
def _cur_txt(f):
    if f['duration'] == f['cap_duration']:
        return 'duración tope = normal: no se puede acortar → ∞'
    return (f"({f['cost_cap']:g} − {f['cost_normal']:g}) / ({f['duration']:g} − {f['cap_duration']:g}) = "
            f"{f['cost_cap'] - f['cost_normal']:g} / {f['duration'] - f['cap_duration']:g} = {(f['cost_cap'] - f['cost_normal']) / (f['duration'] - f['cap_duration']):g}")
pd.DataFrame({'duración normal': datos_proyecto['duration'], 'duración tope': datos_proyecto['cap_duration'],
              'coste normal (k€)': datos_proyecto['cost_normal'], 'coste tope (k€)': datos_proyecto['cost_cap'],
              'CUR = Δcoste / Δduración (k€/semana)': datos_proyecto.apply(_cur_txt, axis=1)})

# %%
acelerables = datos_proyecto.index[np.isfinite(datos_proyecto['ucr'])].tolist()
no_acelerables = [a for a in datos_proyecto.index if a not in acelerables]
Markdown('**Respuesta:** CUR (k€/semana): ' + '; '.join(f'{a} = {datos_proyecto.loc[a, "ucr"]:g}' for a in acelerables)
         + (f'; {lista_y(no_acelerables)}: ∞ (no acelerable{"" if len(no_acelerables) == 1 else "s"}).' if no_acelerables else '.'))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones normales y el camino crítico"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario con las duraciones normales"
#| label: tbl-calendario
calendario_normal = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# La duración del proyecto es la de su ruta más larga. Conviene tener a la vista la duración de *todas* las rutas: al acortar la crítica, las que están a una o dos semanas de ella serán las primeras en volverse críticas y obligarán a acortar varias actividades a la vez.

# %%
#| tbl-cap: "Duración de cada ruta con las duraciones normales"
#| label: tbl-rutasNormal
R = proyecto.path_matrix(dummies=False)
rutas_normal = tabla_rutas(proyecto, datos_proyecto['duration'])
rutas_normal['margen hasta la crítica'] = (rutas_normal['duración'].max() - rutas_normal['duración']).astype(int)
rutas_normal

# %%
coste_indirecto_semanal = 9
duracion_normal = proyecto.duration(datos_proyecto['duration'])
coste_directo_normal = int(datos_proyecto['cost_normal'].sum())
display(explica_camino_critico(proyecto, datos_proyecto['duration']))
display(Markdown(f'Coste directo con las duraciones normales: {explica_suma(datos_proyecto["cost_normal"], "coste directo", "k€")}.'))
Markdown(f'**Respuesta:** duración normal {duracion_normal} semanas. Coste total = directo + indirecto = {coste_directo_normal} + '
         f'{coste_indirecto_semanal} × {duracion_normal} = {coste_directo_normal} + {coste_indirecto_semanal * duracion_normal} = '
         f'**{coste_directo_normal + coste_indirecto_semanal * duracion_normal} k€**.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# En cada paso del algoritmo de Ackoff-Sasieni se acorta una semana la ruta crítica (o las rutas críticas) eligiendo, entre las actividades críticas con margen, la combinación de menor CUR que reduzca *todas* las rutas críticas a la vez. Cuando hay varias rutas críticas sin actividades comunes baratas hay que acortar una actividad de cada una, y el coste del paso es la suma. La tabla muestra, para cada ruta, el CUR de sus actividades y su duración tras cada paso; las últimas filas, el margen de reducción que le queda a cada actividad.

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'], reduction=100, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=100)

# %% [markdown]
# Lo que se compara en cada paso: las rutas críticas en ese momento, las opciones que acortan todas a la vez (una actividad común con margen, o una pareja con una actividad de cada ruta) con su coste, y la elegida (la más barata).

# %%
#| tbl-cap: "Opciones consideradas en cada paso de Ackoff-Sasieni"
#| label: tbl-ackoffPaso
detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)

# %% [markdown]
# Razonamiento de cada paso (los pasos consecutivos que acortan lo mismo se agrupan): rutas críticas, opciones con su coste, cuál se elige y por qué, cuánto se reduce y hasta dónde, y el sobrecoste acumulado.

# %%
narra_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)

# %%
resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],
                        'sobrecoste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},
                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))
resumen['duración'] = duracion_normal - resumen.index
resumen['sobrecoste acumulado'] = resumen['sobrecoste del paso'].cumsum()
resumen

# %% [markdown]
# Comprobación paso a paso: la duración de cada ruta tras cada paso es la que figura en las columnas numeradas de la tabla de Ackoff-Sasieni; tras el último paso, las rutas que empatan en la duración máxima son las críticas del proyecto reducido.

# %%
#| tbl-cap: "Duración de cada ruta tras la reducción"
#| label: tbl-rutasReducidas
rutas_reducidas = tabla_rutas(proyecto, duraciones_reducidas, base=datos_proyecto['duration'])
rutas_reducidas['crítica'] = np.where(rutas_reducidas['duración'] == rutas_reducidas['duración'].max(), 'sí', '')
rutas_reducidas

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La tabla recorre la curva paso a paso: cada paso suma su CUR al coste directo y ahorra una semana de costes indirectos, así que el coste total varía en CUR − coste indirecto semanal.

# %%
curva = pd.DataFrame({'duración': [duracion_normal] + resumen['duración'].tolist(),
                      'CUR del paso': [np.nan] + resumen['sobrecoste del paso'].tolist(),
                      'coste directo': [coste_directo_normal] + (coste_directo_normal + resumen['sobrecoste acumulado']).tolist()})
curva['coste indirecto'] = coste_indirecto_semanal * curva['duración']
curva['coste total'] = curva['coste directo'] + curva['coste indirecto']
curva['variación del total = CUR − indirecto'] = ['—'] + [f'{c:g} − {coste_indirecto_semanal} = {_num(c - coste_indirecto_semanal)}' for c in resumen['sobrecoste del paso']]
curva = curva.set_index('duración')
limpia_tabla(curva, decimales=1)

# %%
#| fig-cap: "Curva coste-duración del proyecto"
#| label: fig-curvaCoste
ax = curva[['coste directo', 'coste indirecto', 'coste total']].plot(marker='o', figsize=(7, 4))
ax.set_ylabel('k€'); ax.set_xlabel('duración (semanas)'); ax.invert_xaxis();

# %%
minimo = float(curva['coste total'].min())
optimas = [int(x) for x in curva.index[np.isclose(curva['coste total'], minimo)]]
duracion_optima = max(optimas)
cur_pasos = resumen['sobrecoste del paso']
baratos = [str(p) for p, c in cur_pasos.items() if c < coste_indirecto_semanal and not np.isclose(c, coste_indirecto_semanal)]
iguales = [str(p) for p, c in cur_pasos.items() if np.isclose(c, coste_indirecto_semanal)]
caros = [str(p) for p, c in cur_pasos.items() if c > coste_indirecto_semanal and not np.isclose(c, coste_indirecto_semanal)]
def _clase(ps, verbo_1, verbo_n, resto_1, resto_n):
    uno = len(ps) == 1
    return f"{'el paso' if uno else 'los pasos'} {rangos(ps)} {verbo_1 if uno else verbo_n} {resto_1 if uno else resto_n}"
clases = []
if baratos:
    clases.append(_clase(baratos, 'cuesta', 'cuestan', f'menos de {coste_indirecto_semanal} k€ y baja el total', f'menos de {coste_indirecto_semanal} k€ y bajan el total'))
if iguales:
    clases.append(_clase(iguales, 'cuesta', 'cuestan', f'exactamente {coste_indirecto_semanal} k€ y lo deja igual', f'exactamente {coste_indirecto_semanal} k€ y lo dejan igual'))
if caros:
    clases.append(_clase(caros, 'cuesta', 'cuestan', f'más de {coste_indirecto_semanal} k€ y lo sube', f'más de {coste_indirecto_semanal} k€ y lo suben'))
vuelve = bool(caros) and any(c < coste_indirecto_semanal for c in cur_pasos[cur_pasos.index > int(caros[0])])
if len(optimas) == 1:
    resp = f'el coste total es mínimo, **{minimo:g} k€**, con una duración de **{optimas[0]} semanas**'
else:
    resp = (f'el coste total mínimo, **{minimo:g} k€**, se alcanza con {lista_y(sorted(optimas, reverse=True), "o")} semanas (empate: esos pasos cuestan '
            f'lo mismo que el indirecto que ahorran); la más larga, **{duracion_optima} semanas**, consigue ese coste con menos aceleración')
regla = (f'Regla de parada: compensa dar un paso mientras su CUR sea menor que el coste indirecto semanal ({coste_indirecto_semanal} k€), da igual si es igual y no compensa si es mayor. '
         f'Aquí {lista_y(clases)}. ')
regla += ('Ojo: tras un paso más caro que el indirecto vuelve a haber uno más barato, así que no basta con parar en el primer paso caro: hay que mirar la curva completa.' if vuelve
          else 'Como los CUR de los pasos no decrecen, basta con parar en el primer paso cuyo CUR supera el coste indirecto semanal.')
Markdown(f'**Respuesta:** {resp}. {regla}')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
bloqueadas = rutas_sin_margen(proyecto, duraciones_reducidas, datos_proyecto['cap_duration'], datos_proyecto['ucr'])
con_margen = {a: int(m) for a, m in margen_restante.items() if m > 0 and np.isfinite(datos_proyecto.loc[a, 'ucr'])}
cal_normal = proyecto.calendar(datos_proyecto['duration'])
def _estado(a):
    if not np.isfinite(datos_proyecto.loc[a, 'ucr']):
        return f'{a} no acelerable'
    return f"{a} en su tope, {int(duraciones_reducidas[a])}"
detalle_bloq = '; '.join(describe_ruta(proyecto, r, datos_proyecto['duration']) + ' no tiene ninguna actividad con margen (' + '; '.join(
    _estado(a) for a in ordena_ruta(list(R.columns[R.loc[r] == 1]), cal_normal)) + ')' for r in bloqueadas)
if con_margen:
    uno = len(con_margen) == 1
    margen_txt = (('Sigue' if uno else 'Siguen') + ' teniendo margen ' + lista_y([f'{a} ({plural(m, "semana")})' for a, m in con_margen.items()])
                  + (', pero no está en esa ruta: acortarla' if uno else ', pero ninguna de ellas está en esa ruta: acortarlas') + ' no acorta la ruta más larga.')
else:
    margen_txt = 'Ya no le queda margen a ninguna actividad.'
Markdown(f'''**Respuesta:** la duración tope del proyecto es {proyecto.duration(duraciones_reducidas)} semanas ({duracion_normal} − {len(pasos)} pasos). En ella, la ruta crítica {detalle_bloq}, así que no puede acortarse y fija la duración mínima del proyecto. {margen_txt}''')
