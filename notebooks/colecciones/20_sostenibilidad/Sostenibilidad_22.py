# ---
# title: Sostenibilidad 22
# subtitle: "VAN 256 k€ sin el fin de vida frente a 192 k€ con desmantelamiento (320 k€ en el año 20); provisión según el PGC"
# coleccion: 20_sostenibilidad
# variante: desmantelamiento
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
# El caso de negocio de una planta industrial (instalación en suelo rústico arrendado) presenta la inversión y los flujos de caja de la explotación que se indican. El contrato de arrendamiento y la autorización obligan, al final de la vida útil, a **desmantelar la instalación y restaurar el terreno**; el equipo ha estimado ese coste y el valor de los materiales que se recuperarían.

# %%
#| tbl-cap: "Datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  concepto                                                , valor
    ('inversión inicial (k€)'                                , 2100 ),
    ('flujo de caja neto anual (k€)'                         , 240  ),
    ('años de explotación'                                   , 20   ),
    ('coste de desmantelamiento y restauración al final (k€)', 320  ),
    ('valor de los materiales recuperados al final (k€)'     , 20   ),
    ('coste de capital'                                      , 0.08 ),
], columns=['concepto', 'valor']).set_index('concepto')
datos_enunciado

# %% [markdown]
# > **De: Director financiero**
# > El caso de negocio termina en la puesta en servicio más los 20 años de explotación; el desmantelamiento es dentro de 20 años y no es de este proyecto ni de esta decisión. Quien explote la planta entonces, que lo pague. Con el VAN tal como está, se aprueba.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 20 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **S4 — Orientación temporal: el proyecto responde de sus resultados.** La dirección del proyecto puede orientarse al proceso (hasta la entrega), a los entregables (su vida útil) o a los resultados (lo que dejan a la comunidad y al futuro). Un caso de negocio que termina en la puesta en servicio deja fuera costes y beneficios que pertenecen al proyecto (operación, desmantelamiento, residuos finales); si una obligación futura es consecuencia de la inversión, su valor actual forma parte del coste. *(cf. PMBOK 8 §3.7.1 (figura 3-8); Plan General de Contabilidad, norma de valoración 2.ª (obligaciones de desmantelamiento).)*
# - **S1 — Jerarquía de respuesta a los impactos (pirámide de sostenibilidad).** Ante un impacto negativo del proyecto las estrategias se ordenan de menos a más deseable: compensar o contrarrestar el daño, restaurarlo, minimizarlo y evitarlo. Compensar cuando se podía minimizar o evitar es quedarse en el escalón más bajo; clasificar una propuesta en su escalón es una pregunta con respuesta única. *(cf. PMBOK 8, principio «integrar la sostenibilidad en todas las áreas del proyecto» (§3.7, figura 3-7).)*
# - **Norma o dato de referencia — forma parte del precio de adquisición de un inmovilizado la estimación inicial del valor actual de las obligaciones de desmantelamiento o retiro y de rehabilitación del lugar, cuando constituyan obligaciones incurridas como consecuencia de su uso** *(Plan General de Contabilidad (RD 1514/2007), norma de registro y valoración 2.ª, apartado 1.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule el VAN del caso de negocio tal como lo presenta el director financiero y el VAN incluyendo el desmantelamiento y la recuperación de materiales al final de la vida útil. ¿Cambia la decisión?  "
pregunta_2 = "2. (A) Calcule el valor actual de la obligación de desmantelamiento (la provisión que se reconoce al inicio) y la cantidad anual constante que, capitalizada al coste de capital, reuniría el coste del desmantelamiento al final. ¿Qué parte de la inversión «escondida» representa?  "
pregunta_3 = "3. (B) Según el criterio y la norma de referencia, ¿puede presentarse el caso de negocio sin el fin de vida? ¿A quién se traslada el coste si no se provisiona, y en qué escalón de la pirámide queda cada opción?  "
pregunta_4 = "4. (C) Redacte el párrafo del caso de negocio y lo que diría al director financiero. Aplique la rúbrica.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
inversion, flujo_anual, anos, desmantelamiento, recuperado, k = 2100, 240, 20, 320, 20, 0.08
def van(flujos, k):
    return float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))
flujos_sin = [-inversion] + [flujo_anual] * anos
flujos_con = list(flujos_sin); flujos_con[-1] += recuperado - desmantelamiento
van_sin, van_con = van(flujos_sin, k), van(flujos_con, k)

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Flujos y valor actual con y sin el fin de vida"
#| label: tbl-flujos
detalle = pd.DataFrame({'flujo sin fin de vida': flujos_sin, 'flujo con fin de vida': flujos_con,
                        'factor 1/(1+k)^t': [1 / (1 + k) ** t for t in range(anos + 1)]}, index=pd.Index(range(anos + 1), name='año'))
detalle['VA sin'] = detalle['flujo sin fin de vida'] * detalle['factor 1/(1+k)^t']
detalle['VA con'] = detalle['flujo con fin de vida'] * detalle['factor 1/(1+k)^t']
detalle.loc['Total'] = [None, None, None, detalle['VA sin'].sum(), detalle['VA con'].sum()]
display(detalle.round({'flujo sin fin de vida': 1, 'flujo con fin de vida': 1, 'factor 1/(1+k)^t': 4, 'VA sin': 2, 'VA con': 2}))
a = (1 - (1 + k) ** -anos) / k
neto_final = recuperado - desmantelamiento
Markdown(f'''Como los {anos} flujos de la explotación son iguales, su valor actual es flujo × factor de anualidad, a = (1 − (1 + k)^−N) / k = (1 − {1 + k:g}^−{anos}) / {k:g} = {a:.4f}:\n\n- VAN sin el fin de vida = −{inversion} + {flujo_anual} × {a:.4f} = **{van_sin:.1f} k€**.\n- El año {anos} se añade el saldo del fin de vida, materiales recuperados − desmantelamiento = {recuperado} − {desmantelamiento} = {neto_final} k€, que vale hoy {neto_final} / {1 + k:g}^{anos} = {neto_final / (1 + k) ** anos:.1f} k€: VAN con el fin de vida = {van_sin:.1f} + ({neto_final / (1 + k) ** anos:.1f}) = **{van_con:.1f} k€**.\n\nLa tabla comprueba las dos cifras año a año.\n\n**Respuesta:** VAN tal como lo presenta el director financiero, {van_sin:.1f} k€; con el fin de vida, {van_con:.1f} k€. ''' + ('**La decisión cambia**: con el fin de vida el proyecto deja de crear valor (VAN < 0).' if van_con < 0 else '**La decisión no cambia** (el VAN sigue siendo positivo), pero el margen baja.'))

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
provision = desmantelamiento / (1 + k) ** anos
cuota_anual = desmantelamiento * k / ((1 + k) ** anos - 1)
Markdown(f'''Valor actual de la obligación (provisión inicial): {desmantelamiento} / (1 + {k:g})^{anos} = {desmantelamiento} / {(1 + k) ** anos:.4f} = **{provision:.1f} k€**, que el PGC incorpora al coste del activo con su provisión: es una inversión «escondida» del {provision:.1f} / {inversion} = **{provision / inversion:.1%}** de la inversión inicial.\n\nCuota anual que, capitalizada al {k:.0%}, reúne los {desmantelamiento} k€ en el año {anos} (factor de fondo de amortización): {desmantelamiento} · {k:g} / ({1 + k:g}^{anos} − 1) = {desmantelamiento * k:g} / {(1 + k) ** anos - 1:.4f} = **{cuota_anual:.1f} k€ al año** ({cuota_anual / flujo_anual:.1%} del flujo anual de {flujo_anual} k€).\n\n**Respuesta:** provisión de {provision:.1f} k€ ({provision / inversion:.1%} de la inversión) y cuota anual de {cuota_anual:.1f} k€.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# **Norma y S4.** La obligación de desmantelar nace con la inversión (el contrato y la autorización la imponen): el PGC incorpora al coste del inmovilizado el valor actual de esa obligación (68.7 k€) y reconoce la provisión desde el primer día. Un caso de negocio sin el fin de vida está orientado a los entregables, no a los resultados; «dentro de 20 años» no lo saca del proyecto, porque la decisión que lo causa se toma hoy. Con el fin de vida el VAN sigue siendo positivo (192.0 k€): el proyecto se aprueba, pero con la cifra correcta y la provisión. **S1.** No provisionar traslada el coste a quien explote la planta en el año 20 —o, si entonces no hay nadie, al arrendador y a la comunidad, que se quedan con la instalación abandonada—: no es compensar ni minimizar, es trasladar. Desmantelar y restaurar es el escalón de **restaurar**; diseñar la planta para su desmontaje y reutilización (y abaratar los 320 k€) sería **minimizar**.
#
# **Respuesta:** no puede presentarse sin el fin de vida (S4 y PGC); si no se provisiona, el coste se traslada a quien explote la planta al final o, si no hay nadie, al arrendador y a la comunidad (trasladar: fuera de la pirámide); desmantelar y restaurar es restaurar, y diseñar para el desmontaje, minimizar.

# %%
preguntas_test = [
    ('La estimación del coste de desmantelamiento de una instalación, cuando la obligación nace de su uso…', ['forma parte del coste del activo por su valor actual, con la provisión correspondiente', 'se contabiliza cuando se desmantela', 'no se contabiliza si faltan más de 10 años', 'la paga quien explote la planta al final'],
     'PGC, norma de registro y valoración 2.ª, apartado 1.'),
    ('Un caso de negocio que termina en la puesta en servicio más la explotación, sin el fin de vida, está orientado…', ['a los entregables, no a los resultados', 'a los resultados', 'al proceso', 'a la sostenibilidad'],
     'S4: orientación temporal de la figura 3-8 del PMBOK 8.'),
    ('La cuota anual constante que reúne un importe D al cabo de N años al tipo k es…', ['D·k / ((1+k)^N − 1)', 'D / N', 'D / (1+k)^N', 'D·k'],
     'Factor de fondo de amortización.'),
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
gift = a_gift(preguntas_test, 'Sostenibilidad 20 — desmantelamiento')
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
# Párrafo para el caso de negocio (modelo): «La inversión de 2100 k€ genera 240 k€ anuales durante 20 años; al final, el desmantelamiento y la restauración del terreno cuestan 320 k€ (valor actual 69 k€), que se provisionan desde el inicio, y los materiales recuperados valen 20 k€. VAN: 192 k€. Se recomienda aprobar el proyecto con la provisión constituida». Al director financiero: el VAN «tal como está» (256 k€) es el VAN de otro proyecto, uno sin obligación de desmantelar; presentarlo así al comité es una verdad a medias (lección 19, C2). Prueba del ciclo de vida: ¿quién paga, y con qué, en el año 20?
