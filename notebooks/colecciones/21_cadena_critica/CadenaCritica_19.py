# ---
# title: CadenaCritica 19
# subtitle: "4 proyectos y un tambor; turnos de 2 días; colchón del tambor de 2 días"
# coleccion: 21_cadena_critica
# variante: multiproyecto-tambor
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
# Una empresa tiene en cartera 4 proyectos que dependen de un mismo equipo escaso, la unidad de validación de software: es el **tambor**, el recurso que limita cuántos proyectos puede sacar adelante la organización, y sólo atiende una tarea a la vez. Cada proyecto se ha planificado con el método de la cadena crítica y su cadena pasa una vez por el tambor. El cuadro resume, en días y con estimaciones agresivas, lo que dura la cadena de cada proyecto antes de su tarea en el tambor (*antes*), la tarea del tambor (*tambor*), lo que queda después (*despues*) y su colchón del proyecto (*CP*). Los proyectos están en el orden de prioridad fijado por la dirección. Las demás actividades de cada proyecto no compiten por el tambor.

# %%
pregunta_1 = "1. Si se lanzan los 4 proyectos el día 0 y el tambor reparte su tiempo en turnos de 2 días entre los proyectos que le esperan (multitarea, rotando en el orden de prioridad; cada proyecto entra en la rotación cuando termina su parte *antes*), ¿qué día termina cada proyecto? Compárelo con la fecha que se le comprometería si tuviera el tambor libre (antes + tambor + después + CP).  "
pregunta_2 = "2. Repita el cálculo si, con todos los proyectos lanzados también el día 0, el tambor no hace multitarea: cuando queda libre atiende, hasta terminarla, la tarea del proyecto de mayor prioridad que le espera.  "
pregunta_3 = "3. Escalone los proyectos por el tambor: sus tareas se programan en el orden de prioridad, una detrás de otra, con un **colchón de capacidad** tras cada una igual al 25 % de su duración (redondeado al día superior); cada proyecto se lanza de modo que su parte *antes* termine 2 días (**colchón del tambor**) antes de su turno en el tambor; el primero se lanza el día 0. Dé el día de lanzamiento, el inicio en el tambor, el fin previsto y la fecha comprometida de cada proyecto, y dibuje el programa.  "
pregunta_4 = "4. Compare los tres regímenes: fin de cada proyecto, duración media de un proyecto (de su lanzamiento a su fin) y máximo de proyectos abiertos a la vez. ¿Qué gana la organización con el escalonamiento? ¿Para qué sirven el colchón de capacidad y el colchón del tambor?  "
pregunta_5 = "5. Llega un proyecto nuevo, P5, con antes = 7, tambor = 5, después = 7 y CP = 7 días, de prioridad menor que los anteriores. Con el programa escalonado, ¿qué día puede lanzarse y qué fecha puede comprometerse con su cliente?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cartera de proyectos (días, estimaciones agresivas; en orden de prioridad)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  proyecto, antes, tambor, despues, CP
    ('P1'    , 2    , 8     , 2      , 3 ),
    ('P2'    , 10   , 4     , 5      , 6 ),
    ('P3'    , 8    , 4     , 3      , 6 ),
    ('P4'    , 5    , 5     , 6      , 7 ),
], columns=['proyecto', 'antes', 'tambor', 'despues', 'CP']).set_index('proyecto')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
# Funciones auxiliares: varios proyectos que comparten el recurso tambor
from math import ceil

COLORES_TRAMO = {'antes': 'green!25', 'tambor': 'red!30', 'despues': 'green!25', 'CP': 'orange!45',
                 'DB': 'orange!15', 'CB': 'black!12'}


def sin_decimales(tabla):
    """Números enteros sin «.0» al mostrar una tabla."""
    return tabla.map(lambda v: f'{v:g}' if isinstance(v, float) else v)


def tambor_sin_escalonar(datos, turno=None):
    """Todos los proyectos se lanzan el día 0 y llegan al tambor al terminar su parte «antes».
    turno=None: cuando el tambor queda libre atiende, hasta terminarla, la tarea del proyecto de mayor
    prioridad que le espera. turno=q: multitarea, turnos de q días rotando en el orden de prioridad entre
    los proyectos que le esperan. Devuelve (fin de la tarea del tambor de cada proyecto, tramos)."""
    orden = list(datos.index)
    llegada = {p: int(datos.loc[p, 'antes']) for p in orden}
    resto = {p: int(datos.loc[p, 'tambor']) for p in orden}
    t, tramos, fin, ultimo = 0, [], {}, -1
    while resto:
        esperan = [i for i, p in enumerate(orden) if p in resto and llegada[p] <= t]
        if not esperan:
            t = min(llegada[p] for p in resto)
            continue
        if turno is None:
            i = esperan[0]
        else:
            siguientes = [i for i in esperan if i > ultimo]
            i = siguientes[0] if siguientes else esperan[0]
        p = orden[i]
        w = resto[p] if turno is None else min(turno, resto[p])
        tramos.append((p, t, w))
        t += w
        resto[p] -= w
        ultimo = i
        if resto[p] == 0:
            fin[p] = t
            del resto[p]
    return pd.Series(fin).reindex(orden), tramos


def escalonar(datos, colchon_tambor, fraccion):
    """Programa del tambor en el orden de prioridad, con colchón de capacidad tras cada tarea, y
    lanzamiento de cada proyecto para que llegue al tambor con su colchón del tambor por delante."""
    filas, libre, libre_txt = [], None, ''
    for p, f in datos.iterrows():
        a, t, b, cp = (int(f[c]) for c in ('antes', 'tambor', 'despues', 'CP'))
        llega = a + colchon_tambor
        if libre is None:
            s = llega
            calc = f'{a} + {colchon_tambor} = {s}'
        else:
            s = max(llega, libre)
            calc = f'máx({a} + {colchon_tambor}; {libre_txt}) = máx({llega}; {libre}) = {s}'
        cb = ceil(fraccion * t - 1e-9)
        filas.append(dict(proyecto=p, a=a, t=t, b=b, cp=cp, inicio_tambor=s, calc_inicio=calc,
                          lanzamiento=s - colchon_tambor - a, fin_tambor=s + t, fin=s + t + b,
                          compromiso=s + t + b + cp, colchon_capacidad=cb, libre_despues=s + t + cb))
        libre = s + t + cb
        libre_txt = f'{s} + {t} + {cb} = {libre}'
    return pd.DataFrame(filas).set_index('proyecto')


def coma(x):
    return f'{x:g}'.replace('.', ',')


def tabla_escalonado(esc, colchon_tambor, fraccion):
    """Cálculos del escalonamiento con los números sustituidos."""
    filas = {}
    for p, f in esc.iterrows():
        filas[p] = {'inicio en el tambor = máx(antes + colchón del tambor; tambor libre)': f['calc_inicio'],
                    'lanzamiento = inicio en el tambor − colchón del tambor − antes':
                        f"{f['inicio_tambor']} − {colchon_tambor} − {f['a']} = {f['lanzamiento']}",
                    'fin previsto = inicio en el tambor + tambor + después':
                        f"{f['inicio_tambor']} + {f['t']} + {f['b']} = {f['fin']}",
                    'fecha comprometida = fin previsto + CP': f"{f['fin']} + {f['cp']} = {f['compromiso']}",
                    f'colchón de capacidad = ⌈{coma(fraccion)}·tambor⌉': f"⌈{coma(fraccion)}·{f['t']}⌉ = {f['colchon_capacidad']}",
                    'tambor libre para el siguiente = inicio + tambor + colchón de capacidad':
                        f"{f['inicio_tambor']} + {f['t']} + {f['colchon_capacidad']} = {f['libre_despues']}"}
    return pd.DataFrame(filas)


def tabla_sin_escalonar(datos, fin_tambor):
    """Fin de cada proyecto lanzado el día 0 y comparación con el compromiso calculado con el tambor libre."""
    filas = {}
    for p, f in datos.iterrows():
        a, t, b, cp = (int(f[c]) for c in ('antes', 'tambor', 'despues', 'CP'))
        fin = int(fin_tambor[p]) + b
        libre = a + t + b + cp
        filas[p] = {'llega al tambor (= antes)': a, 'termina en el tambor': int(fin_tambor[p]),
                    'fin = termina en el tambor + después': f'{int(fin_tambor[p])} + {b} = {fin}',
                    'compromiso con el tambor libre = antes + tambor + después + CP': f'{a} + {t} + {b} + {cp} = {libre}',
                    'retraso sobre el compromiso': max(0, fin - libre)}
    return pd.DataFrame(filas)


def en_curso_a_la_vez(inicios, fines):
    """Máximo de proyectos abiertos a la vez (intervalos [inicio, fin))."""
    eventos = sorted([(int(s), 1) for s in inicios] + [(int(f), -1) for f in fines])
    n = m = 0
    for _, d in eventos:
        n += d
        m = max(m, n)
    return m


def gantt_tramos_tikz(filas, horizonte):
    """Diagrama de barras por tramos (TikZ): filas = [(etiqueta, [(inicio, días, tipo, texto), ...]), ...]."""
    u = min(0.34, 15.0 / max(horizonte, 1))
    n = len(filas)
    paso = 5 if horizonte <= 60 else 10
    L = [rf'\begin{{tikzpicture}}[x={u:.4f}cm, y=-0.62cm, font=\small]']
    for x in range(0, horizonte + 1, paso):
        L.append(rf'\draw[black!15] ({x},-0.55) -- ({x},{n - 0.45});')
        L.append(rf'\node[font=\scriptsize, black!60] at ({x},-0.85) {{{x}}};')
    for i, (etiqueta, tramos) in enumerate(filas):
        L.append(rf'\node[anchor=east, font=\small\bfseries] at (-0.6,{i}) {{{etiqueta}}};')
        for s, d, tipo, texto in tramos:
            if d <= 0:
                continue
            borde = ', dashed' if tipo in ('DB', 'CB') else ''
            L.append(rf'\draw[fill={COLORES_TRAMO[tipo]}, draw=black!55{borde}] ({s},{i - 0.32}) rectangle ({s + d},{i + 0.32});')
            if texto != '':
                L.append(rf'\node[font=\scriptsize] at ({s + d / 2:.2f},{i}) {{{texto}}};')
    L.append(r'\end{tikzpicture}')
    return '\n'.join(L)


def filas_sin_escalonar(datos, fin_tambor, tramos):
    filas = []
    for p, f in datos.iterrows():
        a, b = int(f['antes']), int(f['despues'])
        mios = [(s, w, 'tambor', '') for q, s, w in tramos if q == p]
        filas.append((p, [(0, a, 'antes', a)] + mios + [(int(fin_tambor[p]), b, 'despues', b)]))
    filas.append(('Tambor', [(s, w, 'tambor', q) for q, s, w in tramos]))
    return filas


def filas_escalonado(esc, colchon_tambor):
    filas = []
    for p, f in esc.iterrows():
        lz, a, s, t, b, cp = (int(f[c]) for c in ('lanzamiento', 'a', 'inicio_tambor', 't', 'b', 'cp'))
        filas.append((p, [(lz, a, 'antes', a), (lz + a, colchon_tambor, 'DB', colchon_tambor), (s, t, 'tambor', t),
                          (s + t, b, 'despues', b), (s + t + b, cp, 'CP', cp)]))
    filas.append(('Tambor', [x for p, f in esc.iterrows() for x in
                             ((int(f['inicio_tambor']), int(f['t']), 'tambor', p),
                              (int(f['fin_tambor']), int(f['colchon_capacidad']), 'CB', ''))]))
    return filas

# %%
datos = datos_enunciado.copy()
turno, colchon_tambor, fraccion = 2, 2, 0.25
nuevo = dict(nombre='P5', antes=7, tambor=5, despues=7, CP=7)

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cada proyecto llega al tambor el día *antes* (se lanzó el día 0). El tambor trabaja 2 días seguidos en el primer proyecto que espera, pasa al siguiente en el orden de prioridad, y así sucesivamente; un proyecto sale de la rotación cuando termina su tarea. La tabla recoge los turnos.

# %%
#| tbl-cap: "Turnos del tambor con multitarea"
#| label: tbl-turnosMulti
fin_multi, tramos_multi = tambor_sin_escalonar(datos, turno)
pd.DataFrame(tramos_multi, columns=['proyecto', 'inicio', 'días']).assign(fin=lambda t: t['inicio'] + t['días']).rename(index=lambda i: i + 1).T

# %%
#| tbl-cap: "Fin de cada proyecto con multitarea en el tambor"
#| label: tbl-finMulti
tabla_sin_escalonar(datos, fin_multi)

# %%
#| fig-cap: "Todos los proyectos lanzados el día 0 con multitarea en el tambor (tareas del tambor en rojo)"
#| label: fig-multi
fin_proy_multi = fin_multi + datos['despues']
pinta_tikz(gantt_tramos_tikz(filas_sin_escalonar(datos, fin_multi, tramos_multi), int((fin_proy_multi).max())))

# %%
compromiso_libre = datos[['antes', 'tambor', 'despues', 'CP']].sum(axis=1)
fuera = [p for p in datos.index if fin_proy_multi[p] > compromiso_libre[p]]
Markdown(f'Con multitarea, {len(fuera)} de los {len(datos)} proyectos ({", ".join(fuera)}) terminan después de la fecha '
         'que se les comprometería con el tambor libre: ni siquiera su colchón del proyecto absorbe la espera.' if fuera else
         'Todos los proyectos cumplirían la fecha calculada con el tambor libre, pero consumiendo su colchón.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Fin de cada proyecto sin multitarea en el tambor (todos lanzados el día 0)"
#| label: tbl-finSecuencial
fin_sec, tramos_sec = tambor_sin_escalonar(datos, None)
fin_proy_sec = fin_sec + datos['despues']
display(pd.DataFrame(tramos_sec, columns=['proyecto', 'inicio', 'días']).assign(fin=lambda t: t['inicio'] + t['días']).rename(index=lambda i: i + 1).T)
tabla_sin_escalonar(datos, fin_sec)

# %%
antes_ = [p for p in datos.index if fin_proy_sec[p] < fin_proy_multi[p]]
ult_s, ult_m = int(fin_proy_sec.max()), int(fin_proy_multi.max())
frase = (f'el último proyecto termina el día {ult_s} frente al {ult_m}' if ult_s != ult_m else
         f'el último proyecto termina el mismo día ({ult_s}), porque el tambor hace el mismo trabajo')
Markdown(f'Sin multitarea terminan antes {", ".join(antes_)} '
         f'(media de los fines: {fin_proy_sec.mean():.2f} frente a {fin_proy_multi.mean():.2f} días) y ' + frase + ': repartir el '
         'tambor no adelanta a nadie y retrasa a casi todos, porque cada tarea del tambor termina en su último turno y está '
         'en la cadena crítica de su proyecto.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# El tambor marca el ritmo de la organización: sus tareas se programan primero, en el orden de prioridad y sin interrupción, dejando tras cada una el colchón de capacidad. Después cada proyecto se lanza hacia atrás desde su turno (la *cuerda* que tira del lanzamiento): su parte *antes* debe terminar el colchón del tambor antes de que le toque.

# %%
#| tbl-cap: "Escalonamiento por el tambor"
#| label: tbl-escalonado
esc = escalonar(datos, colchon_tambor, fraccion)
tabla_escalonado(esc, colchon_tambor, fraccion)

# %%
#| fig-cap: "Programa escalonado: parte antes y después en verde, colchón del tambor y colchón del proyecto en naranja, tarea del tambor en rojo y colchones de capacidad en gris"
#| label: fig-escalonado
pinta_tikz(gantt_tramos_tikz(filas_escalonado(esc, colchon_tambor), int(esc['compromiso'].max())))

# %%
Markdown('Lanzamientos: ' + '; '.join(f'{p} el día {int(f.lanzamiento)}, compromiso el día {int(f.compromiso)}' for p, f in esc.iterrows()) + '.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Comparación de los tres regímenes"
#| label: tbl-regimenes
ceros = pd.Series(0, index=datos.index)
regimenes = {'multitarea, todos el día 0': (ceros, fin_proy_multi), 'sin multitarea, todos el día 0': (ceros, fin_proy_sec),
             'escalonados por el tambor': (esc['lanzamiento'], esc['fin'])}
comparacion = pd.DataFrame({r: {**{f'fin de {p}': int(f[p]) for p in datos.index},
                                'duración media (fin − lanzamiento)': round(float((f - l).mean()), 2),
                                'máximo de proyectos abiertos a la vez': en_curso_a_la_vez(l, f),
                                'último fin': int(f.max())} for r, (l, f) in regimenes.items()})
sin_decimales(comparacion.astype(float))

# %%
dur = comparacion.loc['duración media (fin − lanzamiento)']
ultimo = comparacion.loc['último fin']
frase_ultimo = (f'El último proyecto termina el día {int(ultimo.iloc[2])}, '
                + (f'{int(ultimo.iloc[2] - ultimo.iloc[1])} días después que sin escalonar: es el precio de los colchones de capacidad y del tambor, la reserva que hace fiables las fechas comprometidas. '
                   if ultimo.iloc[2] > ultimo.iloc[1] else 'no más tarde que sin escalonar. '))
Markdown(f'Con todos los proyectos lanzados a la vez hay {len(datos)} abiertos desde el día 0; escalonados, como máximo '
         f"{int(comparacion.loc['máximo de proyectos abiertos a la vez'].iloc[2])}. Cada proyecto pasa abierto, de media, {dur.iloc[2]:.2f} días frente a {dur.iloc[1]:.2f} sin multitarea "
         f'y {dur.iloc[0]:.2f} con multitarea: inmoviliza menos tiempo a su equipo y menos dinero, y su fecha no depende de cómo se '
         'reparta el tambor. ' + frase_ultimo)

# %% [markdown]
# La organización no puede terminar más proyectos de los que permite el tambor: lanzar más trabajo del que puede atender sólo alarga las colas y empuja a la multitarea. Escalonando, el tambor trabaja sin interrupciones y sin repartirse, y cada proyecto entra cuando el tambor podrá atenderlo. El **colchón de capacidad** deja al tambor un margen tras cada tarea: si una se alarga, no arrastra a los proyectos siguientes, cuyas fechas se han comprometido contando con su turno. El **colchón del tambor** protege el turno: el trabajo previo llega con unos días de adelanto, de modo que un retraso en la parte *antes* no deja al tambor parado esperando.

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
libre = int(esc['libre_despues'].iloc[-1])
s_n = max(nuevo['antes'] + colchon_tambor, libre)
lanz_n = s_n - colchon_tambor - nuevo['antes']
fin_n = s_n + nuevo['tambor'] + nuevo['despues']
Markdown(f"El tambor queda libre, tras la última tarea y su colchón de capacidad, el día {libre}. "
         f"Inicio de {nuevo['nombre']} en el tambor: máx({nuevo['antes']} + {colchon_tambor}; {libre}) = **{s_n}**; "
         f"lanzamiento: {s_n} − {colchon_tambor} − {nuevo['antes']} = **día {lanz_n}**; fin previsto {s_n} + {nuevo['tambor']} + "
         f"{nuevo['despues']} = {fin_n}; fecha comprometida {fin_n} + {nuevo['CP']} = **día {fin_n + nuevo['CP']}**. "
         "Lanzarlo antes no lo adelantaría: sólo lo tendría abierto esperando al tambor.")

# %% [markdown]
# ## Autoevaluación

# %% [markdown]
# Preguntas de respuesta múltiple con los datos de este ejercicio (la correcta, marcada con ✅, es la primera de cada lista). La celda las exporta también en formato GIFT, con las opciones barajadas, a `~/temp/cuestionario_cadena_critica.gift`, para importarlas en Moodle.

# %%
preguntas_test = [
    ('En el programa escalonado, ¿qué día se lanza P4?',
     ['día 17', 'día 19', 'día 0', 'día 24'],
     'Entra en el tambor el día 24; se lanza 24 − 2 (colchón del tambor) − 5 (antes) = 17.'),
    ('¿Qué fecha puede comprometerse para el proyecto nuevo P5?',
     ['día 50', 'día 26', 'día 48', 'día 43'],
     'El tambor queda libre el día 31; P5 entra el día máx(7 + 2; 31) = 31 y se compromete para 31 + 5 + 7 + 7 = 50.'),
    ('Con todos los proyectos lanzados el día 0 y multitarea en el tambor, ¿qué día termina P1?',
     ['día 20', 'día 12', 'día 18', 'día 21'],
     'Su tarea del tambor termina el día 18, en su último turno; después quedan 2 días.'),
    ('¿Para qué sirve el colchón de capacidad?',
     ['para que el retraso de una tarea del tambor no arrastre a los proyectos siguientes', 'para proteger la fecha de entrega de cada proyecto', 'para que el tambor pueda hacer multitarea', 'para lanzar antes los proyectos'],
     'Las fechas de los proyectos siguientes se comprometen contando con su turno en el tambor; la fecha de entrega la protege el colchón del proyecto.'),
]
TITULO_TEST = 'Cadena crítica — varios proyectos y un tambor (4 proyectos)'
import os, re


def a_gift(preguntas_test, titulo, semilla=1):
    """Preguntas en formato GIFT (Moodle): caracteres especiales escapados y opciones barajadas."""
    esc = lambda s: re.sub(r'([~=#{}:])', r'\\\1', str(s))
    r = np.random.default_rng(semilla)
    lineas = [f'$CATEGORY: {titulo}', '']
    for i, (q, ops, fb) in enumerate(preguntas_test, 1):
        cuerpo = ' '.join(('=' if j == 0 else '~') + esc(ops[j]) for j in r.permutation(len(ops)))
        lineas.append(f'::{esc(titulo)} P{i}::{esc(q)} {{{cuerpo} ####{esc(fb)}}}')
        lineas.append('')
    return '\n'.join(lineas)


gift = a_gift(preguntas_test, TITULO_TEST)
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_cadena_critica.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)
