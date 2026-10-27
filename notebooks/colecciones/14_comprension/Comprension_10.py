# ---
# title: Comprension 10
# subtitle: "9 actividades; 5 preguntas de respuesta múltiple con exportación GIFT"
# coleccion: 14_comprension
# variante: tipo-test
# nivel: 1
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
# Sobre el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (semanas) se adjunta, responda al cuestionario. Cada pregunta tiene una sola respuesta correcta. Al final se pide generar el cuestionario en formato GIFT para importarlo en Moodle.

# %%
pregunta_1 = "1. ¿Cuál es la duración del proyecto?  "
pregunta_2 = "2. ¿Cuál es la holgura total de D?  "
pregunta_3 = "3. ¿Cuántas rutas tiene el grafo (del nodo inicial al final, sin contar ficticias como actividades)?  "
pregunta_4 = "4. ¿Cuántas actividades ficticias necesita el grafo PERT con el mínimo número de ellas?  "
pregunta_5 = "5. ¿Cuál es, aproximadamente, la probabilidad de terminar en 37 semanas o menos (aproximación normal por el camino crítico)?  "
pregunta_6 = "6. Genere el fichero GIFT con las cinco preguntas, sus opciones y la retroalimentación.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 6       , 0.2               ),
    ('B'     , '---'      , 5       , 0.2               ),
    ('C'     , 'A,B'      , 6       , 0.4               ),
    ('D'     , 'A'        , 2       , 0.4               ),
    ('E'     , 'C,D'      , 4       , 0.3               ),
    ('F'     , 'E'        , 6       , 0.4               ),
    ('G'     , 'E'        , 8       , 0.4               ),
    ('H'     , 'F,G'      , 4       , 0.6               ),
    ('I'     , 'H'        , 7       , 0.2               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])
duracion = int(proyecto.duration(datos_proyecto['duration']))
R = proyecto.path_matrix(dummies=False)
sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])
X, plazo = 'D', 37
p_plazo = float(norm.cdf((plazo - duracion) / sigma))

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con el camino crítico"
#| label: fig-pert
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# Cada pregunta del test lleva la respuesta correcta calculada y tres distractores *plausibles*: errores típicos (sumar todas las duraciones, confundir holgura libre y total, contar las ficticias como rutas, usar la varianza en lugar de la desviación típica…). En cada apartado se justifica la respuesta correcta con los números del proyecto y se explica de dónde sale cada opción incorrecta.

# %%
dur = datos_proyecto['duration']
h_total, h_libre = int(calendario.loc[X, 'H_total']), int(calendario.loc[X, 'H_libre'])
n_rutas, n_fict = len(R), len(proyecto.dummies)
camino = ordena_ruta(list(proyecto.critical_path(dur).values())[0], calendario)
Rt = R @ dur[R.columns]
_rutas = {r: ' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario)) for r in R.index}
p_mal = float(norm.cdf(plazo, duracion, sigma ** 2))

def opciones(correcta, distractores, motivo_otro):
    # distractores: lista de (valor, motivo); se descartan los repetidos y se dejan 4 opciones
    vistos, lista, motivos = {correcta}, [correcta], {}
    for v, m in distractores:
        if v not in vistos and len(lista) < 4:
            vistos.add(v); lista.append(v); motivos[v] = m or motivo_otro
    return lista, motivos

def ruta_de(v):
    rs = [_rutas[r] for r in R.index if int(Rt[r]) == v]
    return f'es la duración de {lista_y(rs, "o")}, que no es la ruta más larga' if rs else None

explicaciones = {}
ops1, mot1 = opciones(duracion, [(int(dur.sum()), f'suma de las duraciones de todas las actividades ({" + ".join(str(int(d)) for d in dur)} = {int(dur.sum())}): ignora que las actividades en paralelo se solapan'),
                                 (duracion - 1, ruta_de(duracion - 1)), (duracion + 2, None), (duracion + 1, None), (duracion - 2, ruta_de(duracion - 2))],
                      'no es la duración de ninguna ruta del grafo')
explicaciones[1] = (f'La duración del proyecto es la de la ruta más larga (camino crítico) {" – ".join(camino)}: '
                    f'{" + ".join(str(int(dur[a])) for a in camino)} = **{duracion} semanas**.', mot1)
_r = calendario.loc[X]
_suc = sorted(s for s, ps in prelaciones.immediate_predecessor.items() if X in ps)
ops2, mot2 = opciones(h_total, [(h_libre, f'es la holgura libre de {X}, no la total' + (f' (mín ES de sus sucesoras − EF = mín(' + ', '.join(str(int(calendario.loc[s, "inicio_mas_temprano"])) for s in _suc) + f') − {int(_r.fin_mas_temprano)} = {h_libre})' if _suc else '')),
                                (0, f'sería la holgura de una actividad crítica, y {X} no lo es'), (h_total + 2, None), (h_total + 1, None), (h_total + 3, None)],
                      f'no corresponde a ninguna holgura de {X}')
explicaciones[2] = (f'Holgura total = inicio más tardío − inicio más temprano: H_total({X}) = LS − ES = {int(_r.inicio_mas_tardio)} − {int(_r.inicio_mas_temprano)} = **{h_total} semanas** '
                    f'(equivale a LF − EF = {int(_r.fin_mas_tardio)} − {int(_r.fin_mas_temprano)}).', mot2)
ops3, mot3 = opciones(n_rutas, [(n_rutas + n_fict, f'resulta de sumar las {n_fict} ficticias a las {n_rutas} rutas, como si cada ficticia fuera una ruta más'),
                                (n_rutas - 1, None), (n_rutas + 2, None), (n_rutas + 1, None), (n_rutas + 3, None)],
                      'no coincide con la enumeración de rutas')
explicaciones[3] = (f'Enumerando los caminos del nodo inicial al final (las ficticias forman parte de los caminos, pero no son actividades) hay **{n_rutas} rutas**: '
                    + '; '.join(_rutas.values()) + '.', mot3)
_G = proyecto.pert_graph
_fic = []
for u, v, d in _G.edges(data=True):
    if str(d['activity']).startswith('@'):
        llegan = sorted(dd['activity'] for _, _, dd in _G.in_edges(u, data=True) if not str(dd['activity']).startswith('@'))
        salen = sorted(dd['activity'] for _, _, dd in _G.out_edges(v, data=True) if not str(dd['activity']).startswith('@'))
        _fic.append(f'{d["activity"]} (del nodo {u} al {v}: ' + (f'{lista_y(salen)} espera{"n" if len(salen) > 1 else ""} también a {lista_y(llegan)}' if llegan and salen else 'separa dos actividades que irían entre los mismos nodos') + ')')
ops4, mot4 = opciones(n_fict, [(0, 'sin ficticias no pueden representarse las prelaciones en las que dos actividades comparten sólo parte de sus predecesoras'),
                               (n_fict + 1, None), (n_fict + 2, None), (n_fict + 3, None)],
                      'sobran ficticias: con menos ya se representan todas las prelaciones')
explicaciones[4] = (f'El grafo PERT con el mínimo número de ficticias tiene **{n_fict}**: ' + '; '.join(_fic) + '.', mot4)
ops5, mot5 = opciones(round(p_plazo, 2), [(round(p_mal, 2), f'usa la varianza σ² = {sigma ** 2:.3f} en lugar de la desviación típica σ = {sigma:.3f}'),
                                          (0.5, f'sería la probabilidad de terminar en la duración media, {duracion} semanas, no en {plazo}'),
                                          (round(1 - p_plazo, 2), f'es la probabilidad de *no* cumplir el plazo, P(T > {plazo}) = 1 − {p_plazo:.4f}'),
                                          (0.95, None), (round(p_plazo / 2, 2), None)],
                      'no sale de ningún cálculo con la normal de la ruta crítica')
_var = datos_proyecto['variance']
explicaciones[5] = (f'La duración del proyecto se aproxima por una normal de media la del camino crítico ({duracion} semanas) y desviación típica la raíz de la suma de las varianzas de sus actividades: '
                    f'{explica_sigma_ruta(camino, _var)} semanas. Entonces ' + explica_probabilidad(duracion, sigma, plazo) + f', que redondeada es **{round(p_plazo, 2)}**.', mot5)
preguntas_test = [
    ('¿Cuál es la duración del proyecto (semanas)?', ops1),
    (f'¿Cuál es la holgura total de {X} (semanas)?', ops2),
    ('¿Cuántas rutas tiene el grafo del nodo inicial al final?', ops3),
    ('¿Cuántas actividades ficticias necesita el PERT con el mínimo número de ellas?', ops4),
    (f'¿Probabilidad aproximada de terminar en {plazo} semanas o menos?', ops5),
]

def resuelve(k):
    q, ops = preguntas_test[k - 1]
    texto, motivos = explicaciones[k]
    return Markdown(f'**{q}**\n\n' + '\n'.join(f'- {"✅" if j == 0 else "◻️"} {o}' for j, o in enumerate(ops)) + '\n\n'
                    + f'**Respuesta: {ops[0]}.** {texto}\n\n' + 'Opciones incorrectas:\n\n' + '\n'.join(f'- {v}: {m}.' for v, m in motivos.items()))

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
resuelve(1)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
resuelve(2)

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
resuelve(3)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
resuelve(4)

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %%
resuelve(5)

# %% [markdown]
# ## Apartado 6

# %%
display(Markdown(pregunta_6))

# %% [markdown]
# El formato GIFT de Moodle escribe cada pregunta como `::título::texto {=correcta ~incorrecta ~incorrecta ####retroalimentación general}`; cada opción puede llevar su propia retroalimentación tras `#`. Los caracteres especiales de GIFT (`~ = # { } :`) se escapan con `\` y las opciones se barajan al generar el fichero. Este es el fichero generado:

# %%
import os, re
def _gift_esc(t):
    t = re.sub(r'\*\*|`', '', str(t)).replace('*', '')
    return re.sub(r'([~=#{}:])', r'\\\1', t)
def a_gift(preguntas_test, explicaciones, titulo, semilla=1):
    r = np.random.default_rng(semilla)
    lineas = [f'$CATEGORY: {titulo}', '']
    for i, (q, ops) in enumerate(preguntas_test, 1):
        texto, motivos = explicaciones[i]
        orden = list(r.permutation(len(ops)))
        cuerpo = ' '.join(f'={_gift_esc(ops[j])}' if j == 0 else f'~{_gift_esc(ops[j])}#{_gift_esc(motivos[ops[j]])}' for j in orden)
        lineas.append(f'::{_gift_esc(titulo)} P{i}::{_gift_esc(q)} {{{cuerpo} ####{_gift_esc(texto)}}}')
        lineas.append('')
    return '\n'.join(lineas)
gift = a_gift(preguntas_test, explicaciones, 'Proyecto test')
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
Markdown('```text\n' + gift + '\n```')
