# ---
# title: Sostenibilidad 02
# subtitle: "A 150 k€ frente a B 240 k€ de inversión; ciclo de vida 481 frente a 460 k€ con el carbono a 250 €/t (equilibrio 59 €/t)"
# coleccion: 20_sostenibilidad
# variante: coste-ciclo-de-vida
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
# Un proyecto de reforma de un edificio público incluye la sustitución de la climatización. Hay dos alternativas que cumplen las prestaciones: **A**, el equipo estándar, y **B**, el equipo de alta eficiencia. Los datos de cada una se adjuntan; la vida útil es de 12 años en los dos casos, la tasa de descuento del 4%, el precio de la electricidad de 0.12 €/kWh y el factor de emisión de la red de 0.15 t CO₂ por MWh. La organización aplica un precio sombra del carbono de 250 €/t.

# %%
#| tbl-cap: "Datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  concepto                      , A (estándar), B (eficiente)
    ('inversión (k€)'              , 150         , 240          ),
    ('consumo eléctrico (MWh/año)' , 190         , 114          ),
    ('mantenimiento (k€/año)'      , 5           , 5            ),
    ('retirada al fin de vida (k€)', 5           , 8            ),
], columns=['concepto', 'A (estándar)', 'B (eficiente)']).set_index('concepto')
datos_enunciado

# %% [markdown]
# > **De: Director de compras**
# > Adjudicamos por precio, como siempre: A cuesta 150 y B 240, un 60% más. Lo que consuma después no es de este proyecto, es de mantenimiento, y el carbono es un número de la memoria, no dinero.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 20 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **S3 — Coste del ciclo de vida.** Dos alternativas se comparan por el coste de todo su ciclo de vida, descontado: adquisición, uso (energía, consumibles), mantenimiento, fin de vida y las externalidades que tengan precio; no por el precio de compra. Elegir por precio de adquisición cuando el uso domina el coste es un error de método, no una opinión. *(cf. LCSP art. 148; PMBOK 8 §3.7.3 (finanzas: inversiones en sostenibilidad con ahorro a largo plazo).)*
# - **S2 — Externalidad internalizada.** Cuando la organización ha fijado un precio para una externalidad (precio sombra del carbono, canon, tasa), toda decisión de planificación —comprimir, comprar, diseñar— se compara por el coste total: coste del proyecto más el valor de la externalidad a ese precio. Con el precio fijado, la alternativa correcta es la de menor coste total, y comprar compensaciones a un precio inferior no cambia la cuenta. *(cf. PMBOK 8 §3.7 (estrategias para abordar las externalidades); LCSP art. 148.2 (externalidades medioambientales monetizables); precio sombra del carbono del BEI.)*
# - **S4 — Orientación temporal: el proyecto responde de sus resultados.** La dirección del proyecto puede orientarse al proceso (hasta la entrega), a los entregables (su vida útil) o a los resultados (lo que dejan a la comunidad y al futuro). Un caso de negocio que termina en la puesta en servicio deja fuera costes y beneficios que pertenecen al proyecto (operación, desmantelamiento, residuos finales); si una obligación futura es consecuencia de la inversión, su valor actual forma parte del coste. *(cf. PMBOK 8 §3.7.1 (figura 3-8); Plan General de Contabilidad, norma de valoración 2.ª (obligaciones de desmantelamiento).)*
# - **Norma o dato de referencia — el coste del ciclo de vida comprende los costes de adquisición, de utilización (energía y otros recursos), de mantenimiento y de fin de vida (recogida y reciclado), y puede incluir las externalidades medioambientales (emisiones, contaminación) cuando su valor monetario pueda determinarse y verificarse** *(Ley 9/2017 de Contratos del Sector Público, art. 148.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule el coste del ciclo de vida (valor actual) de cada alternativa sin el carbono: inversión, energía, mantenimiento y retirada. ¿Cuál es más barata y por cuánto?  "
pregunta_2 = "2. (A) Añada el carbono al precio sombra y recalcule. Obtenga el precio del carbono con el que las dos alternativas tendrían el mismo coste del ciclo de vida, y las emisiones de cada una en la vida útil.  "
pregunta_3 = "3. (B) Según los criterios de referencia, ¿qué alternativa se elige y con qué criterio? ¿Es correcto decir que el consumo «no es de este proyecto»?  "
pregunta_4 = "4. (C) ¿Cómo plantea la adjudicación al director de compras y al órgano de contratación? Aplique la rúbrica.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
N, k, precio_energia, factor, precio_sombra = 12, 0.04, 0.12, 0.15, 250
datos = datos_enunciado.copy()
anualidad = (1 - (1 + k) ** -N) / k
def ciclo_de_vida(col, precio_carbono):
    I, c, m, fin = datos[col]
    energia = c * precio_energia * anualidad
    mantenimiento = m * anualidad
    carbono = c * factor * precio_carbono / 1000 * anualidad
    retirada = fin / (1 + k) ** N
    return pd.Series({'inversión': I, 'energía (VA)': energia, 'mantenimiento (VA)': mantenimiento,
                      'retirada (VA)': retirada, 'carbono (VA)': carbono,
                      'coste del ciclo de vida': I + energia + mantenimiento + retirada + carbono})

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Coste del ciclo de vida sin el carbono (k€, valor actual)"
#| label: tbl-lccSin
sin_carbono = pd.DataFrame({col: ciclo_de_vida(col, 0) for col in datos.columns})
display(sin_carbono.round(1))
dif0 = sin_carbono.loc['coste del ciclo de vida', 'A (estándar)'] - sin_carbono.loc['coste del ciclo de vida', 'B (eficiente)']
Markdown(f'''Factor de anualidad {anualidad:.3f} ({N} años al {k:.0%}). Sin el carbono, B es ''' + ('**más barata**' if dif0 > 0 else '**más cara**') + f''' que A en el ciclo de vida: {abs(dif0):.1f} k€ de diferencia, aunque cueste {datos.iloc[0, 1] - datos.iloc[0, 0]:.0f} k€ más al comprarla.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Coste del ciclo de vida con el carbono al precio sombra (k€, valor actual)"
#| label: tbl-lccCon
con_carbono = pd.DataFrame({col: ciclo_de_vida(col, precio_sombra) for col in datos.columns})
display(con_carbono.round(1))
dif = con_carbono.loc['coste del ciclo de vida', 'A (estándar)'] - con_carbono.loc['coste del ciclo de vida', 'B (eficiente)']
emisiones = datos.loc['consumo eléctrico (MWh/año)'] * factor * N
delta_c = (datos.iloc[1, 0] - datos.iloc[1, 1]) * factor / 1000 * anualidad
p_equilibrio = -dif0 / delta_c
Markdown(f'''Con el carbono a {precio_sombra} €/t, B es más barata en **{dif:.1f} k€**. Emisiones en {N} años: A {emisiones.iloc[0]:.0f} t, B {emisiones.iloc[1]:.0f} t. Las dos alternativas se igualan con un precio del carbono de **{p_equilibrio:.0f} €/t**''' + (' (negativo: B gana incluso sin carbono).' if p_equilibrio < 0 else '; por encima gana B.'))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# **S3.** El criterio es el coste del ciclo de vida, no el precio de adquisición: la LCSP (art. 148) lo define con los mismos sumandos del apartado 1 y permite añadir las externalidades con precio. Sin el carbono A es más barata por 6.3 k€, pero **S2** obliga a contar la externalidad al precio que la organización ha fijado, y con 250 €/t B gana por 20.5 k€ (el precio de equilibrio es 59 €/t): la respuesta es **B**. **S4.** «El consumo no es de este proyecto» confunde el ciclo de vida del proceso con el de los entregables: quien decide el equipo decide la factura de energía de 12 años, y la decisión se toma ahora. Que el coste lo pague otro presupuesto (mantenimiento) no lo saca del cálculo; lo saca de la vista.

# %%
preguntas_test = [
    ('El coste del ciclo de vida de un equipo, según la LCSP (art. 148), incluye…', ['adquisición, uso, mantenimiento, fin de vida y, si tienen precio, las externalidades', 'sólo el precio de adquisición', 'adquisición y mantenimiento del primer año', 'el precio más el IVA'],
     'S3: todos los costes del ciclo, descontados.'),
    ('Con un precio sombra del carbono fijado, la decisión entre dos equipos se toma…', ['por el coste del ciclo de vida incluyendo el carbono a ese precio', 'por el precio de compra', 'por las emisiones solamente', 'por el precio de los créditos de compensación'],
     'S2: la externalidad con precio entra en la cuenta.'),
    ('Decir que el consumo del equipo «no es de este proyecto» es…', ['orientarse sólo al proceso e ignorar el ciclo de vida de los entregables (S4)', 'correcto: el proyecto termina en la entrega', 'correcto si lo paga otro presupuesto', 'un criterio de la LCSP'],
     'La decisión del equipo fija el coste de uso de toda la vida útil.'),
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
gift = a_gift(preguntas_test, 'Sostenibilidad 20 — coste del ciclo de vida')
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_sostenibilidad.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
print(gift)

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: (1) identificar a todos los afectados, incluidos los que no están en la mesa (la comunidad, los trabajadores de la cadena, quien opere o desmantele lo construido, las generaciones siguientes) y qué pierde o gana cada uno con las cifras del bloque A; (2) formular al menos dos alternativas además de la propuesta, situándolas en la pirámide (compensar, restaurar, minimizar, evitar); (3) someter cada alternativa a tres pruebas —legalidad (¿cumple la norma?), publicidad (¿la defendería ante el inspector, el cliente o la prensa?) y ciclo de vida (¿quién paga el impacto y cuándo?)—; (4) decidir y redactar la comunicación con las cifras. Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.

# %% [markdown]
# Al director de compras: la tabla del apartado 2; B cuesta 90 k€ más hoy y 20 k€ menos en 12 años al precio sombra (-6 k€ sin él). Al órgano de contratación: que el pliego fije el coste del ciclo de vida como criterio de adjudicación con los parámetros publicados (vida útil, tasa, precio de la energía, factor de emisión y precio del carbono), que es lo que la ley permite y la política de la organización exige. Prueba del ciclo de vida: ¿quién paga los 76 MWh al año de diferencia durante 12 años?
