# ---
# title: Dilema 09
# subtitle: "contrato privado: la Oferta 3 queda 9.8 puntos por debajo con los criterios del pliego"
# coleccion: 19_etica
# variante: adjudicacion-conflicto
# nivel: 1
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
# Usted forma parte de la mesa que adjudica un subcontrato de instalación en un contrato privado (la empresa adjudicadora es una sociedad mercantil). El pliego publicado fija los criterios y sus pesos: precio 50 puntos (proporcional inverso: la oferta más baja obtiene el máximo), plazo 30 puntos (proporcional inverso) y calidad técnica 20 puntos (valorada por un comité técnico, ya puntuada). Se han recibido tres ofertas (precio en k€, plazo en semanas). Cada semana de plazo cuesta además al proyecto 8 k€ de costes indirectos.

# %%
#| tbl-cap: "Ofertas recibidas"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  oferta    , precio, plazo, calidad
    ('Oferta 1', 209   , 9    , 17     ),
    ('Oferta 2', 208.5 , 14   , 20     ),
    ('Oferta 3', 218.9 , 12   , 17     ),
], columns=['oferta', 'precio', 'plazo', 'calidad']).set_index('oferta')
datos_enunciado

# %% [markdown]
# El jefe de proyecto, que preside la mesa, es cuñado del gerente de la empresa que presenta la **Oferta 3**. No lo ha comunicado a nadie.

# %% [markdown]
# > **De: Jefe de proyecto**
# > Propongo adjudicar a la Oferta 3: ya han trabajado con nosotros y sé que responden. El pliego es orientativo; la experiencia previa con la casa vale más que dos puntos de diferencia.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C6 — Conflicto de interés no revelado.** Participar en una decisión (adjudicar, contratar, certificar) teniendo un interés personal en el resultado sin haberlo comunicado a los afectados y sin que estos hayan consentido; lo exigible es revelarlo y abstenerse. *(cf. Código Ético del PMI, 4.3.1–4.3.3; ICB4, Perspectiva 3; en contratos públicos, LCSP art. 64.)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule la puntuación de cada oferta con los criterios y pesos publicados y ordene las ofertas.  "
pregunta_2 = "2. (A) Cuantifique lo que costaría al proyecto adjudicar a la Oferta 3 en lugar de a la mejor puntuada (diferencia de precio más coste de las semanas de plazo adicionales).  "
pregunta_3 = "3. (B) Según el criterio de referencia, ¿qué debe hacer el jefe de proyecto, con independencia de qué oferta sea mejor? ¿Puede la mesa aplicar «la experiencia previa con la casa»? ¿Cambia algo que el contrato sea público o privado?  "
pregunta_4 = "4. (C) ¿Qué hace usted en la mesa y qué dice al jefe de proyecto? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
pesos = dict(precio=50, plazo=30, calidad=20)
oferta_cunado, coste_semana = 'Oferta 3', 8
ofertas = datos_enunciado.copy()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Puntuación con los criterios publicados"
#| label: tbl-puntuacion
puntuacion = pd.DataFrame({
    'precio = 50 · mín / precio': pesos['precio'] * ofertas['precio'].min() / ofertas['precio'],
    'plazo = 30 · mín / plazo': pesos['plazo'] * ofertas['plazo'].min() / ofertas['plazo'],
    'calidad': ofertas['calidad'].astype(float)})
puntuacion['total'] = puntuacion.sum(axis=1)
puntuacion = puntuacion.sort_values('total', ascending=False)
mejor = puntuacion.index[0]
display(puntuacion.round(2))
Markdown(f'''Gana la **{mejor}** con {puntuacion.loc[mejor, "total"]:.2f} puntos; la {oferta_cunado} obtiene {puntuacion.loc[oferta_cunado, "total"]:.2f} ({puntuacion.loc[mejor, "total"] - puntuacion.loc[oferta_cunado, "total"]:.2f} menos).''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
dif_precio = float(ofertas.loc[oferta_cunado, 'precio'] - ofertas.loc[mejor, 'precio'])
dif_plazo = int(ofertas.loc[oferta_cunado, 'plazo'] - ofertas.loc[mejor, 'plazo'])
sobrecoste = dif_precio + dif_plazo * coste_semana
Markdown(f'''Adjudicar a la {oferta_cunado}: {dif_precio:+.1f} k€ de precio y {dif_plazo:+d} semanas de plazo ({dif_plazo * coste_semana:+.0f} k€ de indirectos): **{sobrecoste:+.1f} k€** para el proyecto frente a la mejor oferta.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# **C6.** El jefe de proyecto tiene un conflicto de interés (parentesco con el gerente de una licitadora) y lo exigible no depende de qué oferta sea mejor: **revelarlo** a la mesa y a su organización y **abstenerse** de la decisión (4.3.1 y 4.3.2 del Código del PMI); si además la propuesta beneficia al familiar, es adjudicar por consideraciones personales (4.3.3). La «experiencia previa con la casa» no está en el pliego: introducir un criterio no publicado es aplicar las reglas con favoritismo (4.3.5). En un contrato privado no hay un procedimiento administrativo que lo anule, pero el criterio es el mismo y la empresa suele tenerlo en su código de conducta; en los dos casos la respuesta del bloque B es idéntica.

# %%
preguntas_test = [
    ('Quien tiene un conflicto de interés en una adjudicación debe…', ['revelarlo y abstenerse', 'votar sólo si su candidata es la mejor', 'callarlo si no influye', 'cambiar los pesos del pliego'],
     'C6: revelar y abstenerse, con independencia del resultado.'),
    ('Aplicar «la experiencia previa con la casa» cuando no está en el pliego es…', ['favoritismo: criterio no publicado', 'una buena práctica', 'legítimo si lo aprueba la mesa', 'parte del criterio de calidad'],
     'Los criterios de adjudicación son los publicados; añadir otro cambia las reglas después de jugar.'),
    ('¿Cambia la obligación de revelar el conflicto si el contrato es privado?', ['no: el criterio es el mismo', 'sí: sólo aplica a contratos públicos', 'sí: en privado basta con no votar', 'depende del importe'],
     'La LCSP añade consecuencias administrativas, pero el criterio ético no depende del tipo de contrato.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — adjudicación')
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
# En la mesa: pedir que conste la puntuación del apartado 1 y la propuesta de adjudicar a la mejor oferta; si el jefe de proyecto insiste, preguntar en acta si algún miembro tiene relación con las licitadoras. Al jefe de proyecto, antes: «tienes que decirlo tú y apartarte; si la Oferta 3 fuera realmente mejor, la mesa lo verá sin ti». Si no lo hace, comunicarlo al órgano competente (2.3.2 y 2.3.3). Prueba de publicidad: ¿cómo se leería la adjudicación, con +33.9 k€ de sobrecoste, si el parentesco saliera en el periódico?
