# ---
# title: Dilema 02
# subtitle: "«la ruta crítica acaba en la semana 22» frente a P(T ≤ 22) = 0.50 (PERT) y 0.45 (MC)"
# coleccion: 19_etica
# variante: verdad-a-medias
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
# El cliente de un proyecto de instalación pregunta por escrito si la obra estará terminada en la semana 22, porque ese día tiene comprometida la apertura. El cuadro de prelaciones y las tres estimaciones de duración de cada actividad (optimista, más probable y pesimista, en semanas) se adjuntan.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b 
    ('A'     , '---'      , 4, 5, 6 ),
    ('B'     , '---'      , 1, 5, 9 ),
    ('C'     , '---'      , 2, 3, 4 ),
    ('D'     , 'C'        , 6, 8, 10),
    ('E'     , 'A,B'      , 1, 5, 9 ),
    ('F'     , 'B'        , 6, 6, 12),
    ('G'     , 'D'        , 1, 3, 5 ),
    ('H'     , 'D,F'      , 1, 8, 9 ),
    ('I'     , 'E,G,H'    , 1, 3, 5 ),
], columns=['activity', 'predecessor', 'a', 'm', 'b']).set_index('activity')
datos_enunciado

# %% [markdown]
# > **De: Director comercial**
# > Contéstale que «según la planificación, la ruta crítica termina en la semana 22», que es verdad y es justo la 22. No le des probabilidades: si le hablas de un 60 % se pone nervioso y empieza a pedir penalizaciones.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C2 — Verdad a medias.** Comunicar un dato cierto omitiendo la parte que cambiaría su interpretación (dar la fecha de la ruta crítica sin la probabilidad de cumplirla). *(cf. Código Ético del PMI, 5.3.1.)*
# - **C4 — Retención de información.** Callar un hecho conocido que la otra parte necesitaría para decidir (un riesgo, una holgura, una discrepancia). *(cf. Código Ético del PMI, 5.3.1 y 5.2.3.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule la duración esperada del proyecto, la desviación típica del camino crítico y la probabilidad de terminar en la semana 22 o antes con la aproximación normal del PERT.  "
pregunta_2 = "2. (A) Repita el cálculo con una simulación Monte Carlo (20 000 réplicas, duraciones normales con la media y la desviación típica PERT de cada actividad) y obtenga además las semanas que se cumplirían con probabilidad 0,5 y 0,8. ¿Por qué la simulación da menos probabilidad que el PERT?  "
pregunta_3 = "3. (B) La frase «la ruta crítica termina en la semana 22» es cierta. ¿Cumple el criterio de referencia como respuesta a la pregunta del cliente? ¿Qué forma de engaño es y qué dato falta para que deje de serlo?  "
pregunta_4 = "4. (C) Redacte la respuesta al cliente y la nota interna al director comercial. Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
fecha_cliente = 22
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = (datos_proyecto['a'] + 4 * datos_proyecto['m'] + datos_proyecto['b']) / 6
datos_proyecto['standard_deviation'] = (datos_proyecto['b'] - datos_proyecto['a']) / 6
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
duracion = float(proyecto.duration(datos_proyecto['duration']))
sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])
p_pert = float(norm.cdf(fecha_cliente, duracion, sigma))

# %%
def porc(p, d=1):
    """Probabilidad o fracción como porcentaje: 0.453 -> «45.3 %»."""
    return f'{100 * float(p):.{d}f} %'

def tabla_md(filas, columnas):
    """Tabla Markdown a partir de una lista de filas."""
    esc = lambda x: str(x).replace('|', '\\|')
    lineas = ['| ' + ' | '.join(columnas) + ' |', '|' + '---|' * len(columnas)]
    lineas += ['| ' + ' | '.join(esc(x) for x in f) + ' |' for f in filas]
    return '\n'.join(lineas)

def explica_indicadores(ind, BAC=None, titulo=''):
    """CV, SV, CPI y SPI (y EAC y VAC si se da el BAC) con la fórmula y la sustitución."""
    PV, EV, AC = float(ind['PV']), float(ind['EV']), float(ind['AC'])
    lineas = [f'- CV = EV − AC = {EV:.1f} − {AC:.1f} = **{EV - AC:.1f}** k€',
              f'- SV = EV − PV = {EV:.1f} − {PV:.1f} = **{EV - PV:.1f}** k€',
              f'- CPI = EV / AC = {EV:.1f} / {AC:.1f} = **{EV / AC:.3f}**',
              f'- SPI = EV / PV = {EV:.1f} / {PV:.1f} = **{EV / PV:.3f}**']
    if BAC is not None:
        eac = BAC / (EV / AC)
        lineas += [f'- EAC = BAC / CPI = {BAC:.0f} / {EV / AC:.3f} = **{eac:.1f}** k€',
                   f'- VAC = BAC − EAC = {BAC:.0f} − {eac:.1f} = **{BAC - eac:.1f}** k€']
    return (f'{titulo}\n\n' if titulo else '') + '\n'.join(lineas)

def rubrica_caso(afectados, alternativas, decision):
    """La rúbrica del bloque C aplicada al caso: afectados con cifras, alternativas con las tres pruebas y decisión."""
    partes = ['**La rúbrica aplicada a este caso** (lo que debe contener una respuesta que la supere):', '',
              '*(1) Afectados y qué gana o pierde cada uno:*', '']
    partes += [f'- **{a}**: {b}' for a, b in afectados]
    partes += ['', '*(2) y (3) Alternativas y las tres pruebas:*', '',
               tabla_md(alternativas, ['alternativa', 'legalidad', 'publicidad', 'reversibilidad']), '',
               '*(4) Decisión y comunicación* (un camino defendible, no el único): ' + decision]
    return Markdown('\n'.join(partes))

def sin_ceros(valores):
    """Los términos no nulos de una suma (para escribirla sin «+ 0.0»)."""
    v = pd.Series(valores).astype(float)
    return v[v != 0]

def camino_critico_sigma(proyecto, duraciones, varianzas):
    """Camino crítico en orden de ejecución (el de mayor varianza si hay varios) y su desviación típica."""
    cal = proyecto.calendar(duraciones)
    rutas = [ordena_ruta([a for a in r if not str(a).startswith('@')], cal) for r in proyecto.critical_path(duraciones).values()]
    acts = max(rutas, key=lambda r: sum(float(varianzas[a]) for a in r))
    return acts, float(sum(float(varianzas[a]) for a in acts)) ** 0.5

def rutas_media_varianza(proyecto, duraciones, varianzas):
    """Todas las rutas en orden de ejecución con su duración media y su varianza, de más larga a más corta."""
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    filas = []
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        filas.append((' – '.join(acts), float(sum(float(duraciones[a]) for a in acts)), float(sum(float(varianzas[a]) for a in acts))))
    return pd.DataFrame(filas, columns=['ruta', 'media', 'varianza']).sort_values(['media', 'varianza'], ascending=False).reset_index(drop=True)

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo PERT con las duraciones medias"
#| label: fig-pert
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Cada actividad tiene duración media D = (a + 4m + b)/6 y varianza σ² = ((b − a)/6)² (a: optimista, m: más probable, b: pesimista). La duración esperada del proyecto es la de la ruta más larga con esas medias, y su varianza es la suma de las varianzas de esa ruta.

# %%
#| tbl-cap: "Media y varianza PERT de cada actividad"
#| label: tbl-estimaciones
detalle_estimaciones(datos_enunciado)

# %%
#| tbl-cap: "Duración y varianza de cada ruta"
#| label: tbl-rutas
R = proyecto.path_matrix(dummies=False)
display(detalle_rutas(R, datos_proyecto['duration'], 'duración', 2))
detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)

# %%
camino, _ = camino_critico_sigma(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])
Markdown(f'''Camino crítico: **{' – '.join(camino)}**, con duración esperada {' + '.join(f"{datos_proyecto.loc[a, 'duration']:.2f}" for a in camino)} = **{duracion:.2f}** semanas y {explica_sigma_ruta(camino, datos_proyecto['variance'])} semanas.

Con la aproximación normal del PERT: {explica_probabilidad(duracion, sigma, fecha_cliente)}.

**Respuesta.** Duración esperada {duracion:.2f} semanas, σ = {sigma:.3f} semanas y P(T ≤ {fecha_cliente}) = **{p_pert:.3f}** con el PERT.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
dists = {act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}
resultado = proyecto.monte_carlo(dists, n_samples=20000, seed=2024)
tau = np.asarray(resultado.tau_samples, dtype=float)
p_mc = float((tau <= fecha_cliente).mean())
p50, p80 = np.percentile(tau, 50), np.percentile(tau, 80)
display(resultado.summary().round(2))
Markdown(f'''Monte Carlo ({len(tau)} réplicas, semilla 2024): en {int((tau <= fecha_cliente).sum())} réplicas el proyecto termina en la semana {fecha_cliente} o antes, luego **P(T ≤ {fecha_cliente}) = {int((tau <= fecha_cliente).sum())} / {len(tau)} = {p_mc:.3f}** (PERT: {p_pert:.3f}). Semana que se cumple con probabilidad 0,5 (mediana de las réplicas): **{p50:.2f}**; con probabilidad 0,8 (percentil 80): **{p80:.2f}**. Con el PERT serían {explica_percentil(duracion, sigma, 0.5, 'semanas')} y {explica_percentil(duracion, sigma, 0.8, 'semanas')}.''')

# %%
#| tbl-cap: "Índice de criticidad de las actividades (porcentaje de réplicas en que la actividad está en la ruta más larga)"
#| label: tbl-criticidad
(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%)').T

# %%
rutas = rutas_media_varianza(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])
segunda = rutas.iloc[1]
p_segunda = float(norm.cdf(fecha_cliente, segunda['media'], segunda['varianza'] ** 0.5))
criticidad = resultado.sensitivity_mean * 100
fuera = [a for a in criticidad.index if a not in camino and criticidad[a] >= 1]
crit_min = float(min(criticidad[a] for a in camino))
Markdown(f'''La simulación da menos probabilidad que el PERT ({p_mc:.3f} frente a {p_pert:.3f}) porque el PERT sólo mira la ruta {' – '.join(camino)} y supone que es siempre la más larga. No lo es: la siguiente ruta, {segunda['ruta']}, tiene media {segunda['media']:.2f} semanas y varianza {segunda['varianza']:.3f}, y por sí sola cumpliría la semana {fecha_cliente} con probabilidad {p_segunda:.3f}. El proyecto termina en plazo sólo si **todas** las rutas terminan a tiempo, y eso es menos probable que lo que da una sola ruta (sesgo de fusión). En la tabla de criticidad se ve: las actividades del camino crítico PERT están en la ruta más larga en un {crit_min:.1f} % de las réplicas como mínimo, y {lista_y(fuera) if fuera else 'ninguna otra actividad'} {'lo están en ' + lista_y([f"{criticidad[a]:.1f} %" for a in fuera]) + ' de las réplicas' if fuera else 'aparece'}.

**Respuesta.** P(T ≤ {fecha_cliente}) = {p_mc:.3f} con la simulación (PERT {p_pert:.3f}); se cumplen {p50:.2f} semanas con probabilidad 0,5 y {p80:.2f} con probabilidad 0,8 (la semana entera que se puede comprometer con al menos un 80 % es la **{int(np.ceil(p80))}**).''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
semana_80 = int(np.ceil(p80))
Markdown(f'''**Criterio C2 (verdad a medias).** La frase «la ruta crítica termina en la semana {duracion:.0f}» es cierta: es la duración esperada. Pero el cliente no pregunta por la media, sino si puede contar con la semana {fecha_cliente}, y la respuesta es que eso ocurre con probabilidad {p_mc:.3f} ({porc(p_mc)}; {p_pert:.3f} con el PERT): **{porc(1 - p_mc)} de no llegar**. Dar la media sin la dispersión lleva al cliente a creer que la fecha está asegurada; la parte omitida cambia la interpretación del dato, que es justo lo que define C2. **Criterio C4 (retención de información).** Además, el director comercial pide callar la probabilidad a sabiendas («no le des probabilidades») y es el dato que el cliente necesita para decidir sobre su apertura. No es C1: no se afirma nada falso.

**Respuesta.** No cumple el criterio: es una **verdad a medias (C2)** y, al ocultar el dato a propósito, **retención de información (C4)**. El dato que falta es la **probabilidad de cumplir la semana {fecha_cliente}** ({p_mc:.3f}) o, equivalentemente, la semana que sí se puede comprometer con un 80 % de confianza: la {semana_80}.''')

# %%
preguntas_test = [
    ('«La ruta crítica termina en la semana 22» como respuesta a «¿estará en la semana 22?» es…', ['C2, verdad a medias', 'C1, afirmación falsa', 'una respuesta completa', 'C6, conflicto de interés'],
     'El dato es cierto pero omite la dispersión, que cambia la conclusión.'),
    ('El dato que convierte la respuesta en honesta es…', ['la probabilidad de cumplir la fecha', 'la suma de las duraciones', 'el número de rutas', 'el coste del proyecto'],
     'La media sin la probabilidad no dice si la fecha es alcanzable.'),
    ('La simulación Monte Carlo da menos probabilidad que el PERT porque…', ['hay rutas casi críticas (sesgo de fusión)', 'usa menos réplicas', 'las duraciones son enteras', 'el PERT es siempre pesimista'],
     'El PERT sólo mira la ruta de mayor media; en muchas réplicas la más larga es otra.'),
]
import os
def a_gift(preguntas_test, titulo, semilla=1):
    r = np.random.default_rng(semilla)
    lineas = [f'$CATEGORY: {titulo}', '']
    for i, (q, ops, fb) in enumerate(preguntas_test, 1):
        correcta = ops[0]
        todas = [ops[j] for j in r.permutation(len(ops))]
        cuerpo = ' '.join((f'={v}' if v == correcta else f'~{v}') for v in todas)
        lineas.append(f'::{titulo} P{i}::{q} {{{cuerpo} ####{fb}}}')
        lineas.append('')
    return '\n'.join(lineas)
gift = a_gift(preguntas_test, 'Ética 19 — verdad a medias')
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_etica.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: (1) identificar a todos los afectados y qué pierde o gana cada uno; (2) formular al menos dos alternativas además de la que se propone en el enunciado; (3) someter cada alternativa a las tres pruebas —legalidad (¿cumple la norma?), publicidad (¿la defendería si se hiciera pública?) y reversibilidad (¿la aceptaría en el papel del afectado?)—; (4) decidir y redactar la comunicación con las cifras del bloque A. Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.

# %%
rubrica_caso(
    [('Cliente', f'organiza la apertura en la semana {fecha_cliente} creyendo que es segura, cuando hay un {porc(1 - p_mc)} de no llegar'),
     ('Director comercial', 'evita hoy una conversación sobre penalizaciones; la tendrá después, y con la confianza del cliente perdida'),
     ('Empresa y equipo', f'cargan con la reclamación si la obra acaba después de la {fecha_cliente}; pierden la ocasión de vender una reducción de plazo')],
    [('Responder sólo «la ruta crítica termina en la semana ' + f'{duracion:.0f}»' + ' (propuesta)', 'sí', 'no: el cliente no la aceptaría si supiera la probabilidad', 'no: nadie querría recibir esa respuesta'),
     (f'Dar la probabilidad ({p_mc:.2f}) y la semana comprometible con un 80 % ({semana_80})', 'sí', 'sí', 'sí'),
     (f'Ofrecer además acortar actividades para cumplir la {fecha_cliente}, con su coste (lección 10)', 'sí', 'sí', 'sí')],
    f'''al cliente: «La planificación da una duración esperada de {duracion:.0f} semanas; la probabilidad de terminar en la semana {fecha_cliente} es del {porc(p_mc, 0)}. La semana que podemos comprometer con un 80 % de confianza es la {semana_80}. Si la apertura exige la {fecha_cliente}, podemos estudiar qué actividades acortar y a qué coste.» Nota interna al director comercial: si el cliente descubre después que la fecha tenía un {porc(1 - p_mc, 0)} de riesgo pedirá las penalizaciones igualmente, y con razón; la probabilidad no es una debilidad comercial sino la base para vender la reducción de plazo.''')
