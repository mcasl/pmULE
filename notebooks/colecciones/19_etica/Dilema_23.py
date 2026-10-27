# ---
# title: Dilema 23
# subtitle: "contrato privado: la Oferta 1 queda 15.0 puntos por debajo con los criterios del pliego"
# coleccion: 19_etica
# variante: adjudicacion-conflicto
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
# Usted forma parte de la mesa que adjudica un subcontrato de instalación en un contrato privado (la empresa adjudicadora es una sociedad mercantil). El pliego publicado fija los criterios y sus pesos: precio 50 puntos (proporcional inverso: la oferta más baja obtiene el máximo), plazo 30 puntos (proporcional inverso) y calidad técnica 20 puntos (valorada por un comité técnico, ya puntuada). Se han recibido tres ofertas (precio en k€, plazo en semanas). Cada semana de plazo cuesta además al proyecto 5 k€ de costes indirectos.

# %%
#| tbl-cap: "Ofertas recibidas"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  oferta    , precio, plazo, calidad
    ('Oferta 1', 511.3 , 15   , 18     ),
    ('Oferta 2', 488   , 13   , 10     ),
    ('Oferta 3', 480.5 , 9    , 18     ),
], columns=['oferta', 'precio', 'plazo', 'calidad']).set_index('oferta')
datos_enunciado

# %% [markdown]
# El jefe de proyecto, que preside la mesa, es cuñado del gerente de la empresa que presenta la **Oferta 1**. No lo ha comunicado a nadie.

# %% [markdown]
# > **De: Jefe de proyecto**
# > Propongo adjudicar a la Oferta 1: ya han trabajado con nosotros y sé que responden. El pliego es orientativo; la experiencia previa con la casa vale más que unos puntos de diferencia en el baremo.

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
pregunta_2 = "2. (A) Cuantifique lo que costaría al proyecto adjudicar a la Oferta 1 en lugar de a la mejor puntuada (diferencia de precio más coste de las semanas de plazo adicionales).  "
pregunta_3 = "3. (B) Según el criterio de referencia, ¿qué debe hacer el jefe de proyecto, con independencia de qué oferta sea mejor? ¿Puede la mesa aplicar «la experiencia previa con la casa»? ¿Cambia algo que el contrato sea público o privado?  "
pregunta_4 = "4. (C) ¿Qué hace usted en la mesa y qué dice al jefe de proyecto? Aplique la rúbrica de la lección.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
pesos = dict(precio=50, plazo=30, calidad=20)
oferta_cunado, coste_semana, contrato_publico = 'Oferta 1', 5, False
ofertas = datos_enunciado.copy()

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
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# Con el criterio proporcional inverso, la mejor oferta en precio (la más baja) recibe los 50 puntos y cada una de las demás, 50 · precio mínimo / su precio; igual con el plazo y 30 puntos. La calidad ya viene puntuada sobre 20.

# %%
#| tbl-cap: "Puntuación con los criterios publicados"
#| label: tbl-puntuacion
p_min, t_min = ofertas['precio'].min(), ofertas['plazo'].min()
puntuacion = pd.DataFrame({
    'precio': pesos['precio'] * p_min / ofertas['precio'],
    'plazo': pesos['plazo'] * t_min / ofertas['plazo'],
    'calidad': ofertas['calidad'].astype(float)})
puntuacion['total'] = puntuacion.sum(axis=1)
puntuacion = puntuacion.sort_values('total', ascending=False)
mejor = puntuacion.index[0]
detalle = pd.DataFrame({
    f"precio = {pesos['precio']} · mín / precio": [f"{pesos['precio']} · {p_min:g} / {ofertas.loc[o, 'precio']:g} = {puntuacion.loc[o, 'precio']:.2f}" for o in puntuacion.index],
    f"plazo = {pesos['plazo']} · mín / plazo": [f"{pesos['plazo']} · {t_min} / {ofertas.loc[o, 'plazo']} = {puntuacion.loc[o, 'plazo']:.2f}" for o in puntuacion.index],
    'calidad': [f"{puntuacion.loc[o, 'calidad']:.0f}" for o in puntuacion.index],
    'total = suma': [f"{puntuacion.loc[o, 'precio']:.2f} + {puntuacion.loc[o, 'plazo']:.2f} + {puntuacion.loc[o, 'calidad']:.0f} = {puntuacion.loc[o, 'total']:.2f}" for o in puntuacion.index]},
    index=puntuacion.index)
detalle

# %%
dif_puntos = puntuacion.loc[mejor, 'total'] - puntuacion.loc[oferta_cunado, 'total']
Markdown(f'''**Respuesta.** Orden: {', '.join(f"{i}.ª {o} ({puntuacion.loc[o, 'total']:.2f} puntos)" for i, o in enumerate(puntuacion.index, 1))}. Gana la **{mejor}**; la {oferta_cunado}, que propone el jefe de proyecto, queda {dif_puntos:.2f} puntos por debajo.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
dif_precio = float(ofertas.loc[oferta_cunado, 'precio'] - ofertas.loc[mejor, 'precio'])
dif_plazo = int(ofertas.loc[oferta_cunado, 'plazo'] - ofertas.loc[mejor, 'plazo'])
sobrecoste = dif_precio + dif_plazo * coste_semana
Markdown(f'''- Diferencia de precio: {ofertas.loc[oferta_cunado, 'precio']:g} − {ofertas.loc[mejor, 'precio']:g} = {dif_precio:+.1f} k€.
- Diferencia de plazo: {ofertas.loc[oferta_cunado, 'plazo']} − {ofertas.loc[mejor, 'plazo']} = {dif_plazo:+d} semanas, que a {coste_semana} k€ por semana de indirectos son {dif_plazo:+d} × {coste_semana} = {dif_plazo * coste_semana:+d} k€.
- Total: {dif_precio:+.1f} {'+' if dif_plazo * coste_semana >= 0 else '−'} {abs(dif_plazo * coste_semana)} = **{sobrecoste:+.1f} k€**.

**Respuesta.** Adjudicar a la {oferta_cunado} en lugar de a la {mejor} {'costaría al proyecto ' + f'{sobrecoste:.1f} k€ más' if sobrecoste > 0 else 'no costaría más al proyecto (' + f'{sobrecoste:+.1f} k€' + '), pero sigue siendo peor según el pliego'}.''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Criterio C6 (conflicto de interés no revelado).** El jefe de proyecto preside la mesa y es cuñado del gerente de la empresa que presenta la {oferta_cunado}; tiene, por tanto, un interés personal en el resultado, y no lo ha comunicado a nadie. Lo exigible según C6 no depende de qué oferta sea mejor: **revelarlo** a la mesa y a su organización y **abstenerse** de la decisión. Proponer además la oferta del familiar es decidir por consideraciones personales (cf. PMI 4.3.3).

**«La experiencia previa con la casa».** No es uno de los criterios publicados (precio, plazo y calidad); aplicarlo después de abrir las ofertas cambia las reglas a favor de un licitador (cf. PMI 4.3.5). La mesa sólo puede aplicar el pliego, y con el pliego gana la {mejor} por {dif_puntos:.2f} puntos.

**Contrato privado.** No hay un procedimiento administrativo que anule la adjudicación, pero el criterio ético es el mismo y la empresa suele recogerlo en su código de conducta. 

**Respuesta.** El jefe de proyecto debe revelar el parentesco y abstenerse (C6), sea cual sea la mejor oferta. La mesa no puede aplicar «la experiencia previa», porque no está en el pliego. Que el contrato sea {'público' if contrato_publico else 'privado'} cambia las consecuencias jurídicas, no la respuesta: en los dos casos hay que revelar, abstenerse y adjudicar con los criterios publicados.''')

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

# %%
rubrica_caso(
    [('Proyecto y empresa adjudicadora', f'{sobrecoste:+.1f} k€ si se adjudica a la {oferta_cunado}' + ('' if sobrecoste > 0 else ' (no hay sobrecoste, pero sí una adjudicación contra el pliego)')),
     ('Licitadora mejor puntuada', f'la {mejor} pierde un contrato que ganó con las reglas publicadas'),
     ('Jefe de proyecto', 'favorece a su familia y arriesga su puesto y la validez de la adjudicación'),
     ('Usted y la mesa', 'si callan sabiendo el parentesco, comparten la decisión')],
    [(f'Adjudicar a la {oferta_cunado} por «experiencia» (propuesta)', 'no en contrato público; contraria al pliego en ambos casos', 'no', 'no: ninguna licitadora aceptaría perder así'),
     (f'Que el jefe de proyecto revele y se abstenga, y la mesa adjudique a la {mejor} con el pliego', 'sí', 'sí', 'sí'),
     ('Si quiere valorar la experiencia, incluirla como criterio publicado en la próxima licitación', 'sí', 'sí', 'sí'),
     ('Si no revela, comunicarlo al órgano competente de la organización', 'sí', 'sí', 'sí')],
    f'''en la mesa, pedir que conste la puntuación del apartado 1 y la propuesta de adjudicar a la {mejor}; antes, al jefe de proyecto: «tienes que decirlo tú y apartarte; si la {oferta_cunado} fuera realmente mejor, la mesa lo verá sin ti». Si no lo hace, comunicarlo a quien corresponda. Prueba de publicidad: ¿cómo se leería la adjudicación, con {dif_puntos:.2f} puntos menos y {sobrecoste:+.1f} k€, si el parentesco saliera en la prensa?''')
