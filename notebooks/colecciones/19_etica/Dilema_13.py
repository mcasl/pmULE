# ---
# title: Dilema 13
# subtitle: "baja 29.0 % (umbral 27.9 %) y plazo 14 con P = 0.12: temeraria y sin buena fe"
# coleccion: 19_etica
# variante: baja-temeraria
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
# Su empresa prepara la oferta a una licitación pública cuyo único criterio de adjudicación es el precio; el presupuesto base es de 920 k€ y el plazo de ejecución se fija en la oferta. El cuadro de prelaciones y las tres estimaciones de duración de cada actividad (semanas) proceden del equipo técnico; el coste estimado de ejecución es de 749 k€. Por experiencia en licitaciones anteriores se conocen las bajas (porcentaje de descuento sobre el presupuesto) que suelen presentar los tres competidores habituales.

# %%
#| tbl-cap: "Cuadro de datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, a, m, b 
    ('A'     , '---'      , 6, 6, 12),
    ('B'     , '---'      , 1, 1, 1 ),
    ('C'     , 'A,B'      , 4, 5, 12),
    ('D'     , 'B'        , 2, 8, 8 ),
    ('E'     , 'B'        , 1, 2, 3 ),
    ('F'     , 'D,E'      , 1, 2, 3 ),
    ('G'     , 'F'        , 1, 3, 5 ),
    ('H'     , 'C,D'      , 1, 2, 3 ),
    ('I'     , 'F,H'      , 1, 1, 1 ),
], columns=['activity', 'predecessor', 'a', 'm', 'b']).set_index('activity')
datos_enunciado

# %%
#| tbl-cap: "Bajas habituales de los competidores"
#| label: tbl-ofertas
bajas_competidores = pd.DataFrame([
#  oferta        , baja (%)
    ('Competidor 1', 14.6    ),
    ('Competidor 2', 12.3    ),
    ('Competidor 3', 15.6    ),
], columns=['oferta', 'baja (%)']).set_index('oferta')
bajas_competidores

# %% [markdown]
# > **De: Director comercial**
# > Vamos con una baja del 29.0 % y un plazo de 14 semanas: así ganamos seguro. Ya sé que el equipo dice que no llega; cuando estemos dentro se renegocia con modificados y prórrogas, como siempre.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C13 — Compromisos de buena fe.** Prometer un plazo o un precio que se sabe inalcanzable, contando con renegociarlo después, es un compromiso que no se hace de buena fe. *(cf. Código Ético del PMI, 5.2.4 y 2.2.3.)*
# - **C1 — Afirmación falsa.** Comunicar como cierto un dato que quien lo comunica sabe que no lo es (un porcentaje de avance, una causa, una oferta que no existe). *(cf. Código Ético del PMI, estándar obligatorio 5.3.1; ICB4, Personas 2 «integridad personal y fiabilidad».)*
# - **Norma — presunción de oferta anormalmente baja cuando el único criterio es el precio: un licitador, baja superior a 25 puntos sobre el presupuesto; dos, la que sea inferior en más de 20 puntos a la otra; tres o más, las inferiores en más de 10 puntos a la media de las ofertas (dejando fuera de la media las ofertas superiores a ella en más de 10 puntos y, si quedan menos de tres, tomando las tres más bajas). La presunción obliga a pedir justificación y, si no convence, a excluir la oferta** *(Reglamento general de la Ley de Contratos (RD 1098/2001), art. 85; LCSP art. 149.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Con las cuatro ofertas (los tres competidores y la nuestra), calcule la media de las bajas con la regla de exclusión de ofertas altas y el umbral a partir del cual una oferta se presume anormalmente baja. ¿Incurre la nuestra en la presunción? ¿Qué margen deja el precio ofertado sobre el coste estimado?  "
pregunta_2 = "2. (A) Calcule la probabilidad de ejecutar en 14 semanas con la aproximación normal del PERT, y el plazo que se podría ofertar con una probabilidad de 0,8.  "
pregunta_3 = "3. (B) Según la norma y los criterios de referencia, ¿qué consecuencia tiene la presunción de anormalidad? ¿Qué incumple ofertar un plazo que se sabe inalcanzable contando con renegociarlo?  "
pregunta_4 = "4. (C) ¿Qué oferta (precio y plazo) propondría y cómo lo defendería ante el director comercial? Aplique la rúbrica.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
presupuesto, coste_estimado, baja_propuesta, plazo_oferta = 920, 749.0, 29.0, 14
datos_proyecto = datos_enunciado.copy()
datos_proyecto['duration'] = (datos_proyecto['a'] + 4 * datos_proyecto['m'] + datos_proyecto['b']) / 6
datos_proyecto['standard_deviation'] = (datos_proyecto['b'] - datos_proyecto['a']) / 6
datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
duracion = float(proyecto.duration(datos_proyecto['duration']))
sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])

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
# Con cuatro o más licitadores y el precio como único criterio, la norma de referencia se aplica así: (1) media de las bajas de todas las ofertas; (2) si alguna oferta es superior a esa media en más de 10 puntos (es decir, su baja es más de 10 puntos **menor** que la media), se excluye y se recalcula la media con las restantes (y, si quedan menos de tres, con las tres ofertas más bajas, que son las de mayor baja); (3) se presume anormalmente baja la oferta cuya baja supera la media en más de 10 puntos.

# %%
def umbral_temeridad(bajas):
    '''Cuatro o más licitadores (art. 85.4 RGLCAP): media de las bajas sin las ofertas que superan la media en más
    de 10 puntos (bajas 10 puntos menores que la media); si quedan menos de tres, media de las tres ofertas más bajas.'''
    bajas = np.asarray(bajas, dtype=float)
    media = bajas.mean()
    altas = bajas < media - 10
    if altas.any():
        resto = bajas[~altas]
        if len(resto) < 3:
            resto = np.sort(bajas)[-3:]
        media = resto.mean()
    return media + 10, media
bajas = bajas_competidores['baja (%)'].tolist() + [baja_propuesta]
umbral, media = umbral_temeridad(bajas)
media0 = float(np.mean(bajas))
excluidas = [b for b in bajas if b < media0 - 10]
precio_oferta = presupuesto * (1 - baja_propuesta / 100)
margen = precio_oferta - coste_estimado
baja_sin_perdida = 100 * (1 - coste_estimado / presupuesto)
paso2 = (f'Ninguna baja es menor que {media0:.2f} − 10 = {media0 - 10:.2f}: no se excluye ninguna oferta y la media se mantiene.' if not excluidas else
         f'Las bajas {lista_y([f"{b:g}" for b in excluidas])} son menores que {media0:.2f} − 10 = {media0 - 10:.2f}: se excluyen y la nueva media es **{media:.2f} %**.')
Markdown(f'''1. Media de las cuatro bajas: ({' + '.join(f"{b:g}" for b in bajas)}) / {len(bajas)} = **{media0:.2f} %**.
2. {paso2}
3. Umbral de presunción: {media:.2f} + 10 = **{umbral:.2f} %**. Nuestra baja, {baja_propuesta:g} %, {'**supera** el umbral: se presume anormalmente baja' if baja_propuesta > umbral else 'no supera el umbral'}.

Precio ofertado = {presupuesto} · (1 − {baja_propuesta:g}/100) = **{precio_oferta:.1f} k€**; margen sobre el coste estimado = {precio_oferta:.1f} − {coste_estimado:g} = **{margen:+.1f} k€** ({'pérdidas' if margen < 0 else 'beneficio'}). La mayor baja sin perder dinero sería 100 · (1 − {coste_estimado:g}/{presupuesto}) = {baja_sin_perdida:.1f} %.

**Respuesta.** Media de las bajas {media:.2f} %, umbral {umbral:.2f} %; la oferta del {baja_propuesta:g} % {'incurre' if baja_propuesta > umbral else 'no incurre'} en la presunción y deja un margen de {margen:+.1f} k€ sobre el coste.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
camino, _ = camino_critico_sigma(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])
p_plazo = float(norm.cdf(plazo_oferta, duracion, sigma))
plazo_80 = float(norm.ppf(0.8, duracion, sigma))
Markdown(f'''Camino crítico {' – '.join(camino)}: duración esperada {' + '.join(f"{datos_proyecto.loc[a, 'duration']:g}" for a in camino)} = **{duracion:.2f}** semanas; {explica_sigma_ruta(camino, datos_proyecto['variance'])} semanas.

- {explica_probabilidad(duracion, sigma, plazo_oferta)}.
- Plazo con probabilidad 0,8: {explica_percentil(duracion, sigma, 0.8, 'semanas')}.

**Respuesta.** La probabilidad de ejecutar en {plazo_oferta} semanas es **{p_plazo:.3f}** ({porc(p_plazo)}); el plazo que se cumple con probabilidad 0,8 es {plazo_80:.2f} semanas, es decir, se podrían ofertar **{int(np.ceil(plazo_80))} semanas** (con {int(np.floor(plazo_80))} la probabilidad sería {norm.cdf(np.floor(plazo_80), duracion, sigma):.3f}, menor que 0,8).''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Norma** (RGLCAP art. 85; LCSP art. 149). La presunción de anormalidad obliga a la mesa a pedir al licitador que justifique su oferta; si la justificación no convence, la oferta se excluye. Con un margen de {margen:+.1f} k€ sobre el propio coste estimado, la justificación sería difícil: «ganar seguro» no es lo que la norma prevé, sino un trámite que puede acabar en exclusión.

**Criterio C13 (compromisos de buena fe).** Ofertar {plazo_oferta} semanas con una probabilidad de cumplirlas de {p_plazo:.3f}, sabiendo que «el equipo dice que no llega» y contando con «renegociar con modificados y prórrogas», es asumir un compromiso que se sabe inalcanzable: no es de buena fe. **Criterio C1 (afirmación falsa).** La oferta afirma un plazo que quien la firma no espera cumplir. Que «siempre se haga» no cambia el criterio, y los modificados y las prórrogas tienen causas tasadas en la ley, entre las que no está haber ofertado lo que no se podía cumplir.

**Respuesta.** La presunción de anormalidad abre un trámite de justificación y, si no convence, la **exclusión**. Ofertar un plazo inalcanzable (P = {p_plazo:.3f}) contando con renegociarlo incumple **C13** y **C1**.''')

# %%
preguntas_test = [
    ('Con cuatro o más licitadores y sólo el precio como criterio, se presume anormal la oferta que…', ['baja más de 10 puntos por debajo de la media de las ofertas', 'es la más baja', 'baja más del 25 %', 'baja más de 20 puntos que la siguiente'],
     'Art. 85.4 RGLCAP, con la regla de exclusión de las ofertas altas.'),
    ('La consecuencia de la presunción de anormalidad es…', ['pedir justificación y excluir si no convence', 'la exclusión automática', 'la adjudicación con penalidad', 'ninguna si el licitador es solvente'],
     'La presunción abre un trámite de justificación; no es automática ni inocua.'),
    ('Ofertar un plazo que se sabe inalcanzable contando con renegociarlo es…', ['C13: compromiso sin buena fe (y C1)', 'C4: retención de información', 'una práctica comercial legítima', 'C8: reserva declarada'],
     'El compromiso se asume sabiendo que no se va a cumplir.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — oferta temeraria')
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
baja_defendible = min(umbral, baja_sin_perdida)
rubrica_caso(
    [('Administración contratante', f'adjudica a una oferta con {margen:+.1f} k€ de margen y un {porc(1 - p_plazo, 0)} de probabilidad de no cumplir el plazo: riesgo de abandono, modificados y retrasos'),
     ('Competidores', 'pierden una licitación ante una oferta que no se podía cumplir'),
     ('Empresa y equipo', f'si gana, ejecuta con pérdidas ({margen:+.1f} k€) y penalidades probables; si la excluyen, pierde la licitación igualmente'),
     ('Usuarios de la obra', 'reciben la obra más tarde de lo prometido')],
    [(f'Baja del {baja_propuesta:g} % y plazo de {plazo_oferta} semanas (propuesta)', 'presunción de anormalidad: justificación y posible exclusión', 'no', 'no: la empresa no aceptaría un proveedor que le hiciera esto'),
     (f'Baja ≤ {baja_defendible:.1f} % (bajo el umbral y sin pérdidas) y plazo de {int(np.ceil(plazo_80))} semanas (80 %)', 'sí', 'sí', 'sí'),
     (f'Plazo de {duracion:.0f} semanas con la reserva declarada y su probabilidad', 'sí', 'sí', 'sí'),
     ('Si el plazo corto es decisivo, ofertarlo con el coste de comprimir (lección 10) incluido en el precio', 'sí', 'sí', 'sí'),
     ('No presentarse si no se puede competir sin pérdidas', 'sí', 'sí', 'sí')],
    f'''una baja que no supere ni el umbral ({umbral:.2f} %) ni la que deja el precio por debajo del coste ({baja_sin_perdida:.1f} %), y el plazo de {int(np.ceil(plazo_80))} semanas, que se cumple con un 80 % de probabilidad. Al director comercial: la oferta temeraria puede quedar fuera en la mesa; si entra, los modificados no cubren un plazo que no se podía cumplir y la probabilidad del {porc(1 - p_plazo, 0)} de retraso se paga en penalidades y reputación. Prueba de publicidad: ¿firmaría la oferta si la mesa pudiera leer su correo?''')
