# ---
# title: Sostenibilidad 17
# subtitle: "1200 t compensadas a 6 €/t (7 k€) frente a reducir el 30%; «neutro» prohibido por la Directiva 2024/825"
# coleccion: 20_sostenibilidad
# variante: neutro-en-carbono
# nivel: 2
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
# Antes de empezar la obra, el departamento de marketing quiere anunciar el proyecto como **«proyecto neutro en carbono»**. El inventario de emisiones previsto (t CO₂) por fuente se adjunta, junto con las medidas de reducción que el equipo ha estudiado, con la reducción que consigue cada una (t CO₂) y su coste (k€).

# %%
#| tbl-cap: "Inventario de emisiones previsto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  fuente                           , t CO2, alcance
    ('gasóleo de la maquinaria propia', 290  , 1      ),
    ('electricidad de la obra'        , 80   , 2      ),
    ('hormigón y acero comprados'     , 710  , 3      ),
    ('transporte contratado'          , 100  , 3      ),
    ('viajes del equipo'              , 20   , 3      ),
], columns=['fuente', 't CO2', 'alcance']).set_index('fuente')
datos_enunciado

# %%
#| tbl-cap: "Medidas de reducción estudiadas"
#| label: tbl-medidas
medidas = pd.DataFrame([
#  medida                                              , reduccion_t, coste_k
    ('hormigón con cemento de bajo contenido en clínker' , 209        , 43     ),
    ('acero de horno eléctrico (chatarra reciclada)'     , 216        , 49     ),
    ('conexión a red en lugar de grupo electrógeno'      , 42         , 6      ),
    ('biocombustible HVO en la maquinaria propia'        , 143        , 45     ),
    ('transporte a carga completa y proveedores cercanos', 35         , 3      ),
    ('electricidad con garantía de origen renovable'     , 64         , 3      ),
    ('oficina de obra con fotovoltaica y aerotermia'     , 9          , 7      ),
    ('reutilización de encofrados y tierras en la obra'  , 32         , 5      ),
], columns=['medida', 'reduccion_t', 'coste_k']).set_index('medida')
medidas

# %% [markdown]
# En el mercado voluntario hay créditos de compensación a **6 €/t**. La organización aplica un precio sombra del carbono de **100 €/t** a sus decisiones y se ha comprometido públicamente a reducir las emisiones de sus proyectos un **30%** respecto al diseño inicial.

# %% [markdown]
# > **De: Director de marketing**
# > Compramos créditos por las 1200 t a 6 €/t, que es lo que cuesta un fin de semana de publicidad, y lo anunciamos como «el primer proyecto neutro en carbono del grupo» y «construcción sostenible». Las medidas de reducción, las que salgan gratis; el resto no se nota.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 20 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **S7 — Afirmaciones ambientales.** Una afirmación ambiental sobre un proyecto o producto debe ser específica y demostrable. Están prohibidas las afirmaciones genéricas («sostenible», «verde», «respetuoso con el clima») sin un desempeño reconocido que las respalde y las que presentan un producto como de impacto neutro, reducido o positivo sobre el clima basándose en la compensación de emisiones. Se puede decir cuánto se ha reducido y cuánto se compensa, con cifras; no se puede decir «neutro» por haber comprado créditos. *(cf. Directiva (UE) 2024/825 (anexo I de la Directiva 2005/29/CE, aplicable desde el 27-9-2026); cf. C1–C3 de la lección 19.)*
# - **S1 — Jerarquía de respuesta a los impactos (pirámide de sostenibilidad).** Ante un impacto negativo del proyecto las estrategias se ordenan de menos a más deseable: compensar o contrarrestar el daño, restaurarlo, minimizarlo y evitarlo. Compensar cuando se podía minimizar o evitar es quedarse en el escalón más bajo; clasificar una propuesta en su escalón es una pregunta con respuesta única. *(cf. PMBOK 8, principio «integrar la sostenibilidad en todas las áreas del proyecto» (§3.7, figura 3-7).)*
# - **S2 — Externalidad internalizada.** Cuando la organización ha fijado un precio para una externalidad (precio sombra del carbono, canon, tasa), toda decisión de planificación —comprimir, comprar, diseñar— se compara por el coste total: coste del proyecto más el valor de la externalidad a ese precio. Con el precio fijado, la alternativa correcta es la de menor coste total, y comprar compensaciones a un precio inferior no cambia la cuenta. *(cf. PMBOK 8 §3.7 (estrategias para abordar las externalidades); LCSP art. 148.2 (externalidades medioambientales monetizables); precio sombra del carbono del BEI.)*
# - **Norma o dato de referencia — prohibidas las afirmaciones ambientales genéricas sin un desempeño ambiental excelente reconocido que las respalde, y las afirmaciones de que un producto tiene un impacto neutro, reducido o positivo sobre el clima basadas en la compensación de emisiones; aplicable desde el 27 de septiembre de 2026** *(Directiva (UE) 2024/825, que añade los puntos 4 bis y 4 ter al anexo I de la Directiva 2005/29/CE; transposición antes del 27-3-2026.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule el coste de compensar todas las emisiones a 6 €/t y el coste unitario (€/t) de cada medida de reducción. ¿Qué medidas cuestan menos que el precio sombra de 100 €/t, cuánto reducen en conjunto y cuánto cuestan?  "
pregunta_2 = "2. (A) Obtenga la combinación de medidas de menor coste que alcanza la reducción del 30% comprometida, su coste y lo que costaría compensar el resto a precio de créditos y al precio sombra.  "
pregunta_3 = "3. (B) Según los criterios y la norma de referencia, ¿puede anunciarse el proyecto como «neutro en carbono» comprando los créditos? ¿Y como «construcción sostenible»? ¿Qué afirmación sí podría hacerse? ¿En qué escalón de la pirámide queda cada propuesta?  "
pregunta_4 = "4. (C) ¿Qué propone al director de marketing y a la dirección? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
precio_creditos, precio_sombra, objetivo = 6, 100, 0.3
inventario = datos_enunciado.copy()
huella = float(inventario['t CO2'].sum())
medidas = medidas.copy()
medidas['coste unitario (€/t)'] = medidas['coste_k'] * 1000 / medidas['reduccion_t']
medidas = medidas.sort_values('coste unitario (€/t)')

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Medidas ordenadas por coste unitario (curva de coste marginal de reducción)"
#| label: tbl-macc
medidas['reducción acumulada (t)'] = medidas['reduccion_t'].cumsum()
medidas['coste acumulado (k€)'] = medidas['coste_k'].cumsum()
medidas['rentable al precio sombra'] = medidas['coste unitario (€/t)'] <= precio_sombra
display(medidas.round(1))
rentables = medidas[medidas['rentable al precio sombra']]
coste_compensar_todo = huella * precio_creditos / 1000
Markdown(f'''Compensar las {huella:.0f} t a {precio_creditos} €/t cuesta **{coste_compensar_todo:.1f} k€**. Las medidas con coste unitario ≤ {precio_sombra} €/t son {len(rentables)}: reducen **{rentables["reduccion_t"].sum():.0f} t** ({rentables["reduccion_t"].sum() / huella:.0%}) por **{rentables["coste_k"].sum():.0f} k€**; al precio sombra esas toneladas valen {rentables["reduccion_t"].sum() * precio_sombra / 1000:.1f} k€.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
from itertools import combinations
objetivo_t = objetivo * huella
mejor = None
for n in range(1, len(medidas) + 1):
    for combo in combinations(medidas.index, n):
        red = medidas.loc[list(combo), 'reduccion_t'].sum(); coste = medidas.loc[list(combo), 'coste_k'].sum()
        if red >= objetivo_t and (mejor is None or coste < mejor[1]):
            mejor = (combo, coste, red)
combo, coste_combo, red_combo = mejor
resto = huella - red_combo
display(medidas.loc[list(combo), ['reduccion_t', 'coste_k', 'coste unitario (€/t)']].round(1))
Markdown(f'''Reducir al menos {objetivo_t:.0f} t ({objetivo:.0%}) cuesta como mínimo **{coste_combo:.0f} k€** con las medidas de la tabla ({red_combo:.0f} t, {red_combo / huella:.0%}). Quedan {resto:.0f} t: compensarlas cuesta {resto * precio_creditos / 1000:.1f} k€ a precio de créditos y valen {resto * precio_sombra / 1000:.1f} k€ al precio sombra.''')

# %% [markdown]
# La combinación óptima se busca por enumeración (ocho medidas, 255 combinaciones); con medidas divisibles bastaría recorrer la curva de coste marginal por orden de coste unitario. Obsérvese que la medida más barata por tonelada no siempre entra en la combinación óptima para un objetivo dado: las medidas son indivisibles.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# **Norma y S7.** «Proyecto neutro en carbono» basado en compensaciones es exactamente la afirmación que la Directiva (UE) 2024/825 añade a la lista de prácticas prohibidas (impacto neutro, reducido o positivo por compensación): **no puede anunciarse** así, con independencia del precio de los créditos. «Construcción sostenible» es una afirmación genérica sin un desempeño reconocido que la respalde: **tampoco**. Lo que sí puede afirmarse es lo específico y demostrable: «el proyecto reduce sus emisiones un 30% respecto al diseño inicial mediante [medidas]; las 840 t restantes se compensan con créditos verificados de [proyecto y estándar]». **S1.** Compensar está en el escalón más bajo; las medidas del apartado 2 **minimizan**; cambiar el diseño para no necesitar el hormigón o el acero **evita**. **S2.** Con el precio sombra de 100 €/t, las medidas con coste unitario inferior (99 t por 6 k€) se ejecutan aunque no se anuncie nada: son decisiones rentables según la política de la organización, no publicidad.

# %%
preguntas_test = [
    ('Anunciar un proyecto como «neutro en carbono» porque se han comprado créditos de compensación es, desde el 27-9-2026…', ['una práctica comercial prohibida por la Directiva (UE) 2024/825', 'legal si los créditos están verificados', 'legal si se dice el precio pagado', 'obligatorio en los contratos públicos'],
     'S7: afirmación de impacto neutro basada en compensación.'),
    ('Una afirmación ambiental genérica («sostenible», «verde») sin desempeño reconocido que la respalde…', ['está prohibida', 'es correcta si el proyecto compensa', 'es correcta en publicidad pero no en contratos', 'sólo se prohíbe en envases'],
     'Anexo I de la Directiva 2005/29/CE, punto añadido por la 2024/825.'),
    ('Con un precio sombra fijado, las medidas de reducción que se ejecutan son…', ['las de coste unitario menor o igual que el precio sombra', 'las gratuitas', 'las que pida marketing', 'ninguna si se compensa'],
     'S2: la externalidad con precio decide.'),
    ('En la pirámide de la sostenibilidad, cambiar el diseño para no necesitar un material emisor es…', ['evitar', 'compensar', 'restaurar', 'minimizar'],
     'El escalón más deseable es evitar el impacto.'),
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
gift = a_gift(preguntas_test, 'Sostenibilidad 20 — neutro en carbono')
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_sostenibilidad.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: (1) identificar a todos los afectados, incluidos los que no están en la mesa (la comunidad, los trabajadores de la cadena, quien opere o desmantele lo construido, las generaciones siguientes) y qué pierde o gana cada uno con las cifras del bloque A; (2) formular al menos dos alternativas además de la propuesta, situándolas en la pirámide (compensar, restaurar, minimizar, evitar); (3) someter cada alternativa a tres pruebas —legalidad (¿cumple la norma?), publicidad (¿la defendería ante el inspector, el cliente o la prensa?) y ciclo de vida (¿quién paga el impacto y cuándo?)—; (4) decidir y redactar la comunicación con las cifras. Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.

# %% [markdown]
# Al director de marketing: la frase que puede usarse (específica, con cifras) y la que no; la sanción por práctica desleal y el daño reputacional cuando un competidor o una asociación de consumidores pida la cuenta. A la dirección: ejecutar las medidas rentables al precio sombra (6 k€), decidir si se cumple el compromiso del 30% con la combinación del apartado 2 y, sólo después, compensar el resto con créditos verificados y decirlo con esa palabra. Prueba de publicidad, en sentido literal: ¿resistiría el anuncio una pregunta de un periodista sobre cuántas toneladas se han reducido y cuántas se han comprado?
