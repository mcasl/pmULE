# ---
# title: Reduccion 18
# subtitle: "duraciones triangulares; plazo de 21 semanas; reducción y probabilidad"
# coleccion: 10_reduccion
# variante: crashing-incertidumbre
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
# Considere el proyecto cuyo cuadro de prelaciones, estimaciones de duración (mínima *a*, más probable *m* y máxima *b*, semanas, distribución triangular), duración tope y coste unitario de reducción (k€/semana) se adjunta; la columna *duration* es la duración esperada. Las actividades con CUR infinito no se pueden acelerar. El contrato exige entregar en **21 semanas**.

# %%
pregunta_1 = "1. Con las duraciones esperadas el proyecto dura 23 semanas. Simule 5 000 réplicas y estime la probabilidad de entregar en 21 semanas sin acelerar.  "
pregunta_2 = "2. Aplique Ackoff-Sasieni para reducir el proyecto hasta 21 semanas con las duraciones esperadas. ¿Qué probabilidad de cumplir el plazo da ahora esa reducción? Compruébelo simulando.  "
pregunta_3 = "3. Construya la tabla de probabilidad de cumplir el plazo, duración media simulada y coste de reducción en función de las semanas recortadas (de 0 a 8). Use los mismos números aleatorios para todas las filas.  "
pregunta_4 = "4. ¿Cuántas semanas hay que recortar para tener un 90 % de probabilidad de cumplir? ¿Cuánto cuesta frente a la reducción determinista? ¿Cuánto cuesta cada punto porcentual de probabilidad ganado en cada tramo?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, a, m, b , cap_duration, ucr   
    ('A'     , '---'      , 6       , 3, 7, 8 , 5           , 3     ),
    ('B'     , '---'      , 5       , 4, 5, 6 , 3           , 1     ),
    ('C'     , 'A,B'      , 3       , 2, 3, 4 , 2           , 3     ),
    ('D'     , 'C'        , 7       , 4, 8, 9 , 6           , 2     ),
    ('E'     , 'C'        , 3       , 1, 3, 5 , 1           , 4     ),
    ('F'     , 'B'        , 3       , 2, 3, 4 , 1           , 3     ),
    ('G'     , 'C'        , 8       , 6, 8, 10, 6           , 6     ),
    ('H'     , 'D,F,G'    , 3       , 2, 3, 4 , 2           , 1     ),
    ('I'     , 'E,H'      , 3       , 1, 3, 5 , 3           , np.inf),
], columns=['activity', 'predecessor', 'duration', 'a', 'm', 'b', 'cap_duration', 'ucr']).set_index('activity')
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
}, name='Descripción')
descripciones.to_frame()

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
plazo = 21
dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}
tope, ucr = datos_proyecto['cap_duration'], datos_proyecto['ucr']

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
# La duración esperada de una triangular es $(a + m + b)/3$; el cuadro la da ya redondeada en *duration*. Con esas duraciones esperadas se calcula el calendario determinista.

# %%
#| tbl-cap: "Media de cada triangular frente a la duración esperada del cuadro"
#| label: tbl-medias
pd.DataFrame({'(a + m + b)/3': [f"({r['a']:g} + {r['m']:g} + {r['b']:g})/3 = {(r['a'] + r['m'] + r['b']) / 3:.2f}" for _, r in datos_proyecto.iterrows()],
              'duration': datos_proyecto['duration']}, index=datos_proyecto.index).T

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con las duraciones esperadas y el camino crítico"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Duración esperada de cada ruta"
#| label: tbl-rutas
R = proyecto.path_matrix(dummies=False)
tabla_rutas(proyecto, datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cada réplica sortea una duración por actividad con su triangular; la duración del proyecto es el máximo de las rutas (τ = ‖R·t‖∞). La probabilidad de cumplir es la fracción de réplicas con τ ≤ plazo.

# %%
mc0 = proyecto.monte_carlo(dists, n_samples=5000, seed=7)
display(mc0.summary().round(2))
p0 = float((mc0.tau_samples <= plazo).mean())
Markdown(f'Sin acelerar: duración media simulada {mc0.tau_mean:.2f} semanas (determinista: {proyecto.duration(datos_proyecto["duration"])}) y P(τ ≤ {plazo}) = {p0:.3f}.')

# %%
#| fig-cap: "Distribución simulada de la duración sin acelerar"
#| label: fig-histograma
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(mc0.tau_samples, bins=40, density=True, alpha=0.6)
ax.axvline(plazo, color='red', linestyle='--', label=f'plazo ({plazo})')
ax.set_xlabel('duración del proyecto (semanas)'); ax.legend(); plt.show()

# %%
tabla = proyecto.monte_carlo_crashing(dists, datos_proyecto, 'duration', tope, ucr, plazo=plazo,
                                       reduccion_max=8, n_samples=5000, seed=7)
p0_ent = float(tabla.loc[0, 'P(τ ≤ plazo)'])
Markdown(f'''**Respuesta:** sin acelerar, la probabilidad de entregar en {plazo} semanas es P(τ ≤ {plazo}) ≈ **{p0:.3f}** ({100 * p0:.1f} % de las 5 000 réplicas), con una duración media simulada de {mc0.tau_mean:.2f} semanas frente a las {proyecto.duration(datos_proyecto["duration"])} deterministas. Para los apartados siguientes se usa `monte_carlo_crashing`, que repite la simulación para cada número de semanas recortadas con los mismos números aleatorios y **redondea cada duración sorteada a semanas enteras** (mínimo 1), como el resto del curso; con ese redondeo, sin recortar sale una media de {tabla.loc[0, "τ media"]:.2f} semanas y P(τ ≤ {plazo}) = {p0_ent:.3f}. La diferencia con {p0:.3f} se debe al redondeo: una duración simulada de {plazo}.4 semanas, por ejemplo, cuenta como {plazo} y cumple el plazo.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
r_det = int(tabla.index[tabla['duración determinista'] <= plazo].min())
Markdown(f'Con las duraciones esperadas el proyecto dura {proyecto.duration(datos_proyecto["duration"])} semanas: hay que recortar '
         f'{proyecto.duration(datos_proyecto["duration"])} − {plazo} = {plural(r_det, "semana")} para llegar a {plazo}.')

# %% [markdown]
# En cada paso del algoritmo de Ackoff-Sasieni se acorta una semana la ruta crítica (o las rutas críticas) eligiendo, entre las actividades críticas con margen, la combinación de menor CUR que reduce *todas* las rutas críticas a la vez.

# %%
#| tbl-cap: "Algoritmo de Ackoff-Sasieni con las duraciones esperadas hasta el plazo"
#| label: tbl-ackoff
tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(
    durations=datos_proyecto['duration'], min_durations=tope, costs=ucr, reduction=r_det, report=False)
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=tope,
    costs=ucr,
    reduction=r_det)

# %%
#| tbl-cap: "Opciones consideradas en cada paso"
#| label: tbl-ackoffPaso
detalle_ackoff(proyecto, datos_proyecto['duration'], tope, ucr, pasos)

# %%
narra_ackoff(proyecto, datos_proyecto['duration'], tope, ucr, pasos)

# %%
resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],
                        'coste del paso': [float(sum(ucr[acts])) for acts in pasos.values()]},
                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))
resumen['sobrecoste acumulado'] = resumen['coste del paso'].cumsum()
resumen

# %%
p_det = float(tabla.loc[r_det, 'P(τ ≤ plazo)'])
if p_det < 0.5:
    matiz = 'menos de la mitad de las veces: con la duración esperada justo en el plazo, el sesgo de fusión (en cada réplica manda la ruta que más se alarga) empuja la media por encima del plazo'
elif p_det < 0.75:
    matiz = 'poco más de la mitad de las veces: la duración esperada queda justo en el plazo y, por el sesgo de fusión de las rutas, muchas réplicas lo superan'
else:
    matiz = 'algo más de lo esperable al poner la media justo en el plazo, por el redondeo a semanas enteras, pero aún por debajo de lo que suele exigirse'
Markdown(f'''**Respuesta:** reducir las duraciones esperadas hasta {plazo} semanas exige {plural(r_det, "semana")} de reducción y {' + '.join(_num(c) for c in resumen['coste del paso'])} = {tabla.loc[r_det, "coste"]:.0f} k€ (duración determinista {proyecto.duration(duraciones_reducidas)} semanas); pero, simulando, la probabilidad de cumplir es solo P(τ ≤ {plazo}) = **{p_det:.3f}**: se cumple {matiz}. Está lejos del 90 % que suele exigirse.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Para cada número de semanas recortadas $r$, la tabla da las reducciones que aplica Ackoff-Sasieni (una por paso, separadas por «;»), el coste acumulado, la duración determinista, la media y el percentil 90 simulados y la probabilidad de cumplir, siempre con las mismas 5 000 réplicas de duraciones redondeadas a semanas enteras (a cada réplica se le restan las semanas recortadas a cada actividad, sin bajar de 1 semana).

# %%
#| tbl-cap: "Probabilidad de cumplir el plazo según las semanas recortadas"
#| label: tbl-crashingMC
acortadas = {0: '—'}
for r in tabla.index[1:]:
    _, p_r, _, _ = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=tope, costs=ucr, reduction=int(r), report=False)
    acortadas[r] = '; '.join(' + '.join(p_r[j]) for j in sorted(p_r))
tabla.insert(0, 'actividades acortadas (paso a paso)', pd.Series(acortadas))
tabla.round(3)

# %%
#| fig-cap: "Probabilidad de cumplir y coste de reducción según las semanas recortadas"
#| label: fig-probCoste
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(tabla.index, tabla['P(τ ≤ plazo)'], marker='o', label='P(τ ≤ plazo)'); ax.axhline(0.9, color='grey', linestyle=':')
ax.set_xlabel('semanas recortadas'); ax.set_ylabel('probabilidad')
ax2 = ax.twinx(); ax2.plot(tabla.index, tabla['coste'], marker='s', color='tab:red', label='coste (k€)'); ax2.set_ylabel('k€')
ax.legend(loc='upper left'); ax2.legend(loc='lower right'); plt.show()

# %%
r_max = int(tabla.index.max())
tope_txt = '' if r_max >= 8 else (f' No se puede recortar más de {plural(r_max, "semana")}: después no queda ninguna combinación con margen que acorte '
                                   'todas las rutas críticas (Ackoff-Sasieni se detiene).')
P_ = tabla['P(τ ≤ plazo)']; C_ = tabla['coste']; M_ = tabla['τ media']
Markdown(f'**Respuesta:** la tabla anterior. Al recortar, la probabilidad de cumplir sube de {P_[0]:.3f} (sin recortar) a {P_[r_max]:.3f} '
         f'({plural(r_max, "semana")} recortadas, {C_[r_max]:.0f} k€) y la duración media simulada baja de {M_[0]:.2f} a {M_[r_max]:.2f} semanas,'
         + (' siempre algo por encima de la determinista (sesgo de fusión).' if (M_ >= tabla['duración determinista'] - 1e-9).all() else '.') + tope_txt)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Coste de cada punto porcentual de probabilidad en cada tramo: lo que cuesta recortar esa semana dividido entre los puntos que gana la probabilidad, $\Delta\text{coste} / (100\,\Delta P)$.

# %%
r90 = int(tabla.index[tabla['P(τ ≤ plazo)'] >= 0.9].min())
P_ = tabla['P(τ ≤ plazo)']; C_ = tabla['coste']
filas = {}
for r in tabla.index[1:]:
    dc, dp = C_[r] - C_[r - 1], 100 * (P_[r] - P_[r - 1])
    filas[f'{r - 1} → {r}'] = {'Δcoste (k€)': f'{_num(C_[r])} − {_num(C_[r - 1])} = {_num(dc)}',
                               'Δprobabilidad (puntos)': f'{100 * P_[r]:.1f} − {100 * P_[r - 1]:.1f} = {dp:.1f}',
                               'k€ por punto': (f'{_num(dc)} / {dp:.1f} = {dc / dp:.2f}' if dp > 0 else 'no gana probabilidad')}
marginal = pd.DataFrame(filas).T
marginal.index.name = 'semanas recortadas'
display(marginal)
por_punto = {k: (C_[int(k.split(' → ')[1])] - C_[int(k.split(' → ')[0])]) / (100 * (P_[int(k.split(' → ')[1])] - P_[int(k.split(' → ')[0])]))
             for k in filas if P_[int(k.split(' → ')[1])] > P_[int(k.split(' → ')[0])]}
caro = max(por_punto, key=por_punto.get)
fin_txt = (': los últimos puntos, cuando la probabilidad ya está cerca de 1, son los más caros.' if caro == list(por_punto)[-1]
           else '. En general los últimos puntos, cuando la probabilidad ya está cerca de 1, son los más caros.')
barato = min(por_punto, key=por_punto.get)
Markdown(f'''**Respuesta:** para tener al menos un 90 % de probabilidad hay que recortar **{plural(r90, "semana")}** (P = {P_[r90]:.3f}), con un coste de **{C_[r90]:.0f} k€**, frente a {plural(r_det, "semana")} y {C_[r_det]:.0f} k€ de la reducción determinista hasta el plazo (P = {P_[r_det]:.3f}): la diferencia, {C_[r90] - C_[r_det]:.0f} k€, es lo que cuesta la seguridad. El punto de probabilidad más barato se compra en el tramo {barato} ({por_punto[barato]:.2f} k€) y el más caro en el tramo {caro} ({por_punto[caro]:.2f} k€){fin_txt}''')
