# ---
# title: Probabilidad 13
# subtitle: "10 actividades; σ máxima admisible de E para P ≥ 90 % en 28 semanas"
# coleccion: 08_probabilidad
# variante: varianza-maxima
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
# Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en semanas) se adjunta. La actividad E se va a subcontratar y la desviación típica de su duración depende del contratista elegido. El cliente exige terminar en 28 semanas con una probabilidad de al menos el 90 %.

# %%
pregunta_1 = "1. Determine la duración media del proyecto y la ruta crítica. ¿Está en ella la actividad subcontratada?  "
pregunta_2 = "2. Calcule la varianza del proyecto en función de la desviación típica σ de E.  "
pregunta_3 = "3. ¿Qué desviación típica máxima puede tener E para cumplir la exigencia del cliente?  "
pregunta_4 = "4. Si ningún contratista garantiza esa dispersión, ¿cuántas semanas habría que reducir la duración media del proyecto (manteniendo las varianzas) para cumplir con el contratista de σ = 1,5?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %%
#| tbl-cap: "Cuadro de datos del enunciado (σ de E por determinar)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, standard_deviation
    ('A'     , '---'      , 3       , 0.1               ),
    ('B'     , '---'      , 7       , 0.1               ),
    ('C'     , 'A'        , 6       , 0.6               ),
    ('D'     , 'C'        , 3       , 0.6               ),
    ('E'     , 'D'        , 2       , np.nan               ),
    ('F'     , 'B,E'      , 5       , 1                 ),
    ('G'     , 'E'        , 3       , 0.3               ),
    ('H'     , 'G'        , 1       , 0.1               ),
    ('I'     , 'H'        , 2       , 0.4               ),
    ('J'     , 'I'        , 6       , 0.1               ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation']).set_index('activity')
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
X, plazo, probabilidad_objetivo = 'E', 28, 0.9

# %%
def tabla_rutas(R, medias, varianzas=None, calendario=None, dec=0, dec_var=3, nombre='media'):
    """Media (y varianza) de cada ruta con la suma término a término; actividades en orden de ejecución."""
    filas = {}
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario)
        m = float(sum(float(medias[a]) for a in acts))
        fila = {'actividades': ' – '.join(acts),
                f'{nombre} = suma': ' + '.join(f'{a}({float(medias[a]):.{dec}f})' for a in acts) + f' = {m:.{dec}f}',
                nombre: m}
        if varianzas is not None:
            v = float(sum(float(varianzas[a]) for a in acts))
            fila['varianza = suma'] = ' + '.join(f'{a}({float(varianzas[a]):.{dec_var}f})' for a in acts) + f' = {v:.{dec_var}f}'
            fila['varianza'] = v
            fila['σ = √varianza'] = v ** 0.5
        filas[r] = fila
    t = pd.DataFrame(filas).T
    for c in (nombre, 'varianza', 'σ = √varianza'):
        if c in t:
            t[c] = t[c].astype(float)
    return t

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

# %%
duracion_media = proyecto.duration(datos_proyecto['duration'])
calendario = proyecto.calendar(datos_proyecto['duration'])
criticas = ordena_ruta(sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)), calendario)
display(explica_camino_critico(proyecto, datos_proyecto['duration']))
Markdown(f'**Respuesta:** duración media {plural(duracion_media, "semana")}; ruta crítica {" – ".join(criticas)}. '
         f'La actividad subcontratada, {X}, {"sí" if X in criticas else "no"} está en ella.')

# %% [markdown]
# Media de cada ruta y varianza de sus actividades de desviación conocida (la de E se cuenta como 0 hasta conocerla):

# %%
#| tbl-cap: "Media de cada ruta y varianza de las actividades de desviación conocida"
#| label: tbl-rutas
R = proyecto.path_matrix(dummies=False)
sd_conocida = pd.to_numeric(datos_proyecto['standard_deviation'], errors='coerce').fillna(0.0)
rutas = tabla_rutas(R, datos_proyecto['duration'], sd_conocida ** 2, calendario).sort_values('media', ascending=False)
limpia_tabla(rutas.rename(columns={'varianza = suma': f'varianza sin {X} = suma', 'varianza': f'varianza sin {X}'}).drop(columns='σ = √varianza'), decimales=3)

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
resto = [a for a in criticas if a != X]
var_resto = float((datos_proyecto.loc[resto, 'standard_deviation'] ** 2).sum())
Markdown(f'''La varianza del proyecto es la de la ruta crítica, la suma de las varianzas de sus actividades. Sin {X}: '''
         f'''{explica_suma((datos_proyecto.loc[resto, "standard_deviation"] ** 2).round(3), "σ²_resto", decimales=3)}; '''
         f'''a ella se suma la varianza de {X}, σ² (σ_X en lo que sigue).\n\n**Respuesta:** σ²_T = {var_resto:.3f} + σ², es decir, σ_T = √({var_resto:.3f} + σ²).''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Exigir $P(T \le \text{plazo}) \ge p$ equivale a que el plazo sea al menos el percentil $p$ de $T$. En cualquier normal ese percentil está a un número fijo de desviaciones típicas por encima de la media, $k_p$, el percentil de la normal de media 0 y desviación 1 (`=INV.NORM.ESTAND(p)` o `=INV.NORM(p; 0; 1)` en la hoja de cálculo). Por tanto hace falta $\mu + k_p\,\sigma_T \le \text{plazo}$, es decir, $\sigma_T \le (\text{plazo} - \mu)/k_p$, y como $\sigma_T^2 = \sigma^2_{resto} + \sigma_X^2$, $\sigma_X^2 \le \sigma_T^2 - \sigma^2_{resto}$.

# %%
z_p = norm.ppf(probabilidad_objetivo)
sigma_T_max = (plazo - duracion_media) / z_p
var_X_max = sigma_T_max ** 2 - var_resto
sigma_X_max = np.sqrt(var_X_max) if var_X_max > 0 else 0.0
p_txt = f'{probabilidad_objetivo:g}'.replace('.', ',')
if var_X_max > 0:
    comprobacion = (f'Comprobación con σ_X = {sigma_X_max:.3f}: σ_T = √({var_resto:.3f} + {sigma_X_max:.3f}²) = {sigma_T_max:.3f} y '
                    f'{explica_probabilidad(duracion_media, sigma_T_max, plazo)}.')
    respuesta = f'la desviación típica de {X} no puede superar **{sigma_X_max:.2f} semanas**'
else:
    comprobacion = 'Ni con σ_X = 0 se alcanza la probabilidad exigida: el resto de la ruta crítica ya es demasiado variable.'
    respuesta = 'ningún contratista permite cumplir la exigencia, ni siquiera uno sin variabilidad'
Markdown(f'''k_p = `=INV.NORM.ESTAND({p_txt})` = {z_p:.4f}.\n\n'''
         f'''σ_T ≤ ({plazo} − {duracion_media}) / {z_p:.4f} = {sigma_T_max:.3f}; σ_X² ≤ {sigma_T_max:.3f}² − {var_resto:.3f} = {var_X_max:.3f}'''
         + (f''', luego σ_X ≤ √{var_X_max:.3f} = {sigma_X_max:.3f}.''' if var_X_max > 0 else '.') + '\n\n' + comprobacion
         + f'''\n\n**Respuesta:** {respuesta}.''')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Con la desviación típica del contratista, $\sigma_T$ queda fijada y lo que se ajusta es la media: hace falta $\mu \le \text{plazo} - k_p\,\sigma_T$.

# %%
sigma_contratista = 1.5
sigma_T = np.sqrt(var_resto + sigma_contratista ** 2)
media_necesaria = plazo - z_p * sigma_T
reduccion = max(int(np.ceil(duracion_media - media_necesaria)), 0)
nueva_media = duracion_media - reduccion
segunda = float(rutas['media'].iloc[1]) if len(rutas) > 1 else 0.0
if segunda <= nueva_media:
    nota_ruta = f'La ruta crítica no cambia: la siguiente ruta dura {segunda:g} semanas de media, no más que {nueva_media}.'
else:
    nota_ruta = (f'Ojo: la siguiente ruta dura {segunda:g} semanas de media, más que {nueva_media}; habría que acortarla también '
                 'y recalcular la varianza con la nueva ruta crítica.')
Markdown(f'''σ_T = √({var_resto:.3f} + 1.5²) = √{var_resto + 2.25:.3f} = {sigma_T:.3f}; hace falta μ ≤ {plazo} − {z_p:.4f} × {sigma_T:.3f} = {media_necesaria:.2f} semanas. '''
         f'''Como las duraciones son enteras, la media debe bajar de {duracion_media} a {nueva_media} semanas. Comprobación: '''
         f'''{explica_probabilidad(nueva_media, sigma_T, plazo)}. {nota_ruta}\n\n'''
         f'''**Respuesta:** habría que reducir la duración media del proyecto en {plural(reduccion, "semana")} (acortando actividades críticas).''')
