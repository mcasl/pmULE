# ---
# title: Dilema 14
# subtitle: "530 h (26 k€) de A a B: CPI de A de 0.79 a 1.01; no subvencionable"
# coleccion: 19_etica
# variante: imputacion-cruzada
# nivel: 1
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
# Una empresa de ingeniería ejecuta a la vez dos proyectos: **A**, financiado con una subvención pública que paga los costes justificados hasta el presupuesto aprobado (BAC), y **B**, para un cliente privado. Los indicadores de valor ganado en la fecha de control (k€) se adjuntan. Las horas se imputan por parte de horas a 50 €/h.

# %%
#| tbl-cap: "Valor ganado en la fecha de control (k€)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  proyecto             , BAC, PV , EV   , AC   
    ('A (subvencionado)'  , 160, 112, 95.9 , 121.8),
    ('B (cliente privado)', 540, 324, 337.7, 300.1),
], columns=['proyecto', 'BAC', 'PV', 'EV', 'AC']).set_index('proyecto')
datos_enunciado

# %% [markdown]
# > **De: Director de la oficina técnica**
# > A se ha pasado y la subvención no cubre más. Reimputad 530 horas de A al proyecto B, que va sobrado: las han hecho las mismas personas y para la empresa el coste total es el mismo. En la justificación de la subvención A sale a presupuesto y nadie se entera.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C1 — Afirmación falsa.** Comunicar como cierto un dato que quien lo comunica sabe que no lo es (un porcentaje de avance, una causa, una oferta que no existe). *(cf. Código Ético del PMI, estándar obligatorio 5.3.1; ICB4, Personas 2 «integridad personal y fiabilidad».)*
# - **C5 — Beneficio propio a costa de otro.** Obtener una ventaja (dinero, plazo, reputación) trasladando el coste a otra parte que no lo ha aceptado. *(cf. Código Ético del PMI, 5.3.2 y 3.3.2.)*
# - **Norma — sólo son subvencionables los gastos que responden de manera indubitada a la naturaleza de la actividad subvencionada y se realizan en el plazo fijado; el gasto imputado indebidamente se reintegra con intereses** *(Ley 38/2003 General de Subvenciones, arts. 31 y 37.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule CV, CPI, EAC y VAC de cada proyecto con los datos reales, y el importe de las horas que se reimputarían.  "
pregunta_2 = "2. (A) Recalcule los indicadores tras la reimputación. ¿Cuánto cambia cada CPI y cuánto el total de la empresa? ¿Cuántas horas habría que mover para que A saliera exactamente a CPI = 1?  "
pregunta_3 = "3. (B) Según la norma y los criterios de referencia, ¿puede justificarse en la subvención el coste reimputado? ¿Qué incumple la propuesta y qué consecuencias tiene?  "
pregunta_4 = "4. (C) ¿Qué alternativas tiene la empresa ante el sobrecoste de A y qué propondría al director? Aplique la rúbrica.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
horas, tarifa = 530, 50
vg = datos_enunciado.copy()
def indicadores(vg):
    t = vg.copy()
    t['CV'] = t['EV'] - t['AC']; t['CPI'] = t['EV'] / t['AC']
    t['EAC = BAC / CPI'] = t['BAC'] / t['CPI']; t['VAC'] = t['BAC'] - t['EAC = BAC / CPI']
    t.loc['Total'] = t.sum(); t.loc['Total', 'CPI'] = t.loc['Total', 'EV'] / t.loc['Total', 'AC']
    return t
coste_horas = horas * tarifa / 1000

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Indicadores con los datos reales"
#| label: tbl-reales
reales = indicadores(vg)
display(reales.round(2))
Markdown(f'''Horas a reimputar: {horas} × {tarifa} € = **{coste_horas:.1f} k€**. A: CPI {reales.iloc[0]["CPI"]:.2f}, previsión de coste final {reales.iloc[0]["EAC = BAC / CPI"]:.0f} k€ sobre {reales.iloc[0]["BAC"]:.0f} de subvención.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Indicadores tras reimputar las horas de A a B"
#| label: tbl-reimputado
vg2 = vg.copy(); vg2.iloc[0, vg2.columns.get_loc('AC')] -= coste_horas; vg2.iloc[1, vg2.columns.get_loc('AC')] += coste_horas
reimputado = indicadores(vg2)
comparacion = pd.DataFrame({'CPI real': reales['CPI'], 'CPI reimputado': reimputado['CPI'], 'AC real': reales['AC'], 'AC reimputado': reimputado['AC']})
display(comparacion.round(3))
horas_cpi_1 = (vg.iloc[0]['AC'] - vg.iloc[0]['EV']) * 1000 / tarifa
Markdown(f'''El total de la empresa no cambia (AC {reales.loc["Total", "AC"]:.1f} k€ y CPI {reales.loc["Total", "CPI"]:.2f} en los dos casos): sólo cambia quién paga. Para que A saliera a CPI = 1 habría que mover {horas_cpi_1:.0f} horas.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# **Norma.** Sólo es subvencionable el gasto que responde de manera indubitada a la actividad subvencionada (Ley 38/2003, art. 31): las horas de A son gasto de A, y presentarlas como gasto de B —o, lo que es lo mismo, presentar A «a presupuesto» ocultando que parte de su coste se ha cargado a otro proyecto— no hace que A cueste menos: hace que la justificación sea falsa. La consecuencia es el reintegro de lo indebidamente justificado con intereses y, según el importe y la intención, la infracción o el fraude de subvenciones. **C1.** La justificación afirma un coste que se sabe falso. **C5.** Y el beneficio (cobrar la subvención completa) se obtiene a costa del financiador público y, de paso, del cliente de B, cuyo proyecto aparece más caro de lo que es.

# %%
preguntas_test = [
    ('Reimputar a otro proyecto horas trabajadas en el subvencionado…', ['hace falsa la justificación: no es gasto subvencionable', 'es neutro porque el coste total no cambia', 'es correcto si lo hacen las mismas personas', 'es una reserva de contingencia'],
     'Ley 38/2003, art. 31: el gasto debe responder de manera indubitada a la actividad subvencionada.'),
    ('Tras la reimputación, el CPI del conjunto de la empresa…', ['no cambia', 'mejora', 'empeora', 'pasa a 1'],
     'Se mueve coste entre proyectos; la suma de AC y de EV es la misma.'),
    ('La consecuencia de justificar un gasto no subvencionable es…', ['el reintegro con intereses (y, en su caso, sanción)', 'ninguna si el proyecto termina', 'una penalidad por demora', 'la pérdida de la holgura'],
     'Art. 37 Ley 38/2003.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — imputación cruzada')
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

# %% [markdown]
# Alternativas reales: asumir el sobrecoste de A con fondos propios (es lo que significa una subvención a presupuesto cerrado); solicitar al órgano concedente una modificación o ampliación si la causa lo justifica y el programa lo permite; reducir el alcance restante de A de acuerdo con el financiador; revisar por qué A se ha desviado para que la siguiente justificación sea limpia. Al director: la reimputación no ahorra nada a la empresa (apartado 2), convierte un sobrecoste en un fraude y pone en riesgo la subvención entera y las siguientes. Prueba de publicidad: ¿cómo se leería la justificación en una auditoría que cruce los partes de horas de A y de B?
