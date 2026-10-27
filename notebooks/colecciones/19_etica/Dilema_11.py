# ---
# title: Dilema 11
# subtitle: "VAN restante -59 k€ (cancelar) aunque los flujos nominales (300) superen la inversión que falta (290)"
# coleccion: 19_etica
# variante: coste-hundido
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
# Un proyecto de desarrollo de producto lleva gastados los importes que se indican (k€). Para terminarlo falta la inversión que se indica, tras la cual el producto generaría el flujo de caja anual previsto durante los años indicados. El comité de dirección debe decidir si continúa o cancela.

# %%
#| tbl-cap: "Datos del enunciado"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  concepto                     , valor
    ('gastado hasta hoy (hundido)', 540  ),
    ('inversión que falta'        , 290  ),
    ('flujo anual previsto'       , 50   ),
    ('años de explotación'        , 6    ),
    ('coste de capital'           , 0.08 ),
], columns=['concepto', 'valor']).set_index('concepto')
datos_enunciado

# %% [markdown]
# > **De: Jefe de proyecto (borrador para el comité)**
# > Hemos invertido 540 k€; cancelar ahora es tirarlos. Con 290 k€ más terminamos y los 50 k€ al año durante 6 años (300 k€) recuperan la inversión. Propongo continuar.

# %% [markdown]
# ## Criterio de referencia
#
# Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 (redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta se justifica citando el criterio y la fuente, no con una opinión.
#
# - **C12 — Decidir sobre el futuro, no sobre lo gastado.** Continuar un proyecto se justifica sólo por el valor actual de los flujos que quedan; lo ya gastado no es argumento ni a favor ni en contra, y decir «así recuperamos la inversión» cuando el VAN restante es negativo es una afirmación falsa (C1). *(cf. Criterio económico de la lección 17; Código Ético del PMI, 5.2.1.)*
# - **C11 — Reconocer los errores propios.** Quien decide asume las consecuencias de sus decisiones; atribuir a otros un error propio, o seguir adelante para no reconocerlo, incumple este criterio. *(cf. Código Ético del PMI, 2.2.4; ICB4, Personas 2, indicador «asumir la responsabilidad de las propias decisiones».)*

# %% [markdown]
# Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).

# %%
pregunta_1 = "1. (A) Calcule el VAN de los flujos que quedan (la inversión que falta y los flujos de explotación) al coste de capital, y el VAN del proyecto completo contando lo ya gastado.  "
pregunta_2 = "2. (A) ¿Qué decisión (continuar o cancelar) recomienda cada VAN? ¿Cuál es el criterio correcto y por qué el otro no lo es? ¿Es cierto que los flujos «recuperan la inversión»?  "
pregunta_3 = "3. (B) Según el criterio de referencia, ¿es correcta la propuesta del jefe de proyecto? ¿Qué afirmación del borrador incumple un criterio y cuál?  "
pregunta_4 = "4. (C) Redacte el párrafo que llevaría al comité (sea cual sea la decisión) y lo que diría al jefe de proyecto. Aplique la rúbrica.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
gastado, inversion_restante, flujo_anual, anos, k = 540, 290, 50, 6, 0.08
flujos_restantes = [-inversion_restante] + [flujo_anual] * anos
def van(flujos, k):
    return float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))
van_restante = van(flujos_restantes, k)
van_total = van_restante - gastado
suma_nominal = flujo_anual * anos
va_flujos = van_restante + inversion_restante

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
# VAN (valor actual neto) de los flujos que quedan: la inversión que falta se paga hoy (año 0) y cada flujo anual se descuenta al coste de capital k con el factor 1/(1 + k)^t. Lo ya gastado no forma parte de esos flujos: es un coste hundido, que no se recupera ni se evita con la decisión.

# %%
#| tbl-cap: "Flujos que quedan y su valor actual (k€)"
#| label: tbl-flujos
detalle = pd.DataFrame({'flujo': [float(f) for f in flujos_restantes], 'factor 1/(1+k)^t': [1 / (1 + k) ** t for t in range(anos + 1)]},
                       index=pd.Index([str(t) for t in range(anos + 1)], name='año'))
detalle['valor actual'] = detalle['flujo'] * detalle['factor 1/(1+k)^t']
detalle.loc['Total'] = [float(sum(flujos_restantes)), np.nan, detalle['valor actual'].sum()]
limpia_tabla(detalle, 4)

# %%
factor_anualidad = (1 - (1 + k) ** -anos) / k
Markdown(f'''- Valor actual de los flujos de explotación: {flujo_anual} · (1/{1 + k:g} + 1/{1 + k:g}² + … + 1/{1 + k:g}^{anos}) = {flujo_anual} · (1 − {1 + k:g}^−{anos}) / {k:g} = {flujo_anual} · {factor_anualidad:.4f} = **{va_flujos:.2f} k€**.
- VAN de lo que queda = −{inversion_restante} + {va_flujos:.2f} = **{van_restante:.2f} k€**.
- VAN del proyecto completo = VAN de lo que queda − lo gastado = {van_restante:.2f} − {gastado} = **{van_total:.2f} k€**.

**Respuesta.** VAN de los flujos que quedan: {van_restante:.2f} k€; VAN del proyecto completo contando lo gastado: {van_total:.2f} k€.''')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
rec_restante = va_flujos >= inversion_restante
rec_total_nominal = suma_nominal >= gastado + inversion_restante
Markdown(f'''- El VAN de lo que queda ({van_restante:.2f} k€) es {'positivo' if van_restante > 0 else 'negativo'}: recomienda **{'continuar' if van_restante > 0 else 'cancelar'}**.
- El VAN total ({van_total:.2f} k€) es {'positivo' if van_total > 0 else 'negativo'}: recomendaría {'continuar' if van_total > 0 else 'cancelar'}.

El criterio correcto es el **VAN de lo que queda**. Lo gastado ({gastado} k€) es el mismo si se continúa y si se cancela: no cambia con la decisión y, por tanto, no puede inclinarla. El VAN total responde a otra pregunta, si el proyecto completo fue buena inversión ({'sí' if van_total > 0 else 'no'}), que ya no se puede cambiar.

¿«Recuperan la inversión»? Depende de qué inversión, y hay que descontar:

- La que falta: los flujos valen hoy {va_flujos:.2f} k€ {'≥' if rec_restante else '<'} {inversion_restante} k€: {'sí la recuperan' if rec_restante else 'no la recuperan'} {('(la suma sin descontar, ' + str(suma_nominal) + ' k€, también la supera)') if rec_restante else ('(sólo la suma sin descontar, ' + str(suma_nominal) + ' k€, la supera: de ahí la apariencia de que la recupera)' if suma_nominal >= inversion_restante else '(ni siquiera la suma sin descontar, ' + str(suma_nominal) + ' k€, llega)')}.
- La total, {gastado} + {inversion_restante} = {gastado + inversion_restante} k€: los flujos valen hoy {va_flujos:.2f} k€, así que **no** (VAN total {van_total:.2f} k€){'; ni siquiera sumando sin descontar: ' + str(suma_nominal) + ' k€' if not rec_total_nominal else '; sólo la suma sin descontar (' + str(suma_nominal) + ' k€) lo aparenta'}.

**Respuesta.** El VAN de lo que queda recomienda {'continuar' if van_restante > 0 else 'cancelar'} y el VAN total, {'continuar' if van_total > 0 else 'cancelar'}; el correcto es el primero. {'La afirmación del borrador es cierta sólo para la inversión que falta; para la inversión total es falsa.' if rec_restante else 'La afirmación del borrador es falsa: descontados, los flujos no cubren ni la inversión que falta.'}''')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
Markdown(f'''**Criterio C12 (decidir sobre el futuro, no sobre lo gastado).** El borrador argumenta con lo gastado («hemos invertido {gastado} k€; cancelar ahora es tirarlos»), que no es un argumento ni a favor ni en contra, y con una suma sin descontar ({flujo_anual} × {anos} = {suma_nominal} k€). La decisión se toma con el VAN de lo que queda: {van_restante:.2f} k€, que recomienda **{'continuar' if van_restante > 0 else 'cancelar'}**. {'La conclusión del borrador (continuar) es correcta, pero por razones equivocadas.' if van_restante > 0 else 'La propuesta del borrador (continuar) es incorrecta.'}

**Criterio C1 (afirmación falsa).** «Recuperan la inversión» {'es falso si se refiere, como sugiere el borrador, a lo invertido en total (' + str(gastado + inversion_restante) + ' k€ frente a ' + f'{va_flujos:.2f}' + ' k€ de valor actual).' if rec_restante else 'es falso: descontados al ' + porc(k, 0) + ', los flujos valen ' + f'{va_flujos:.2f}' + ' k€, menos que los ' + str(inversion_restante) + ' k€ que faltan (C12 lo califica expresamente de C1).'} **Criterio C11 (reconocer los errores propios).** Si el motivo real de seguir es no reconocer que el proyecto fue mala inversión (VAN total {van_total:.2f} k€), el criterio pide lo contrario: reconocerlo y decidir el resto con los números.

**Respuesta.** {'La propuesta (continuar) es correcta en su conclusión, pero su argumentación incumple C12' if van_restante > 0 else 'La propuesta es incorrecta: incumple C12'}; la afirmación «recuperan la inversión» incumple **C1**, y seguir para no admitir el error incumpliría **C11**.''')

# %%
preguntas_test = [
    ('La decisión de continuar o cancelar un proyecto en marcha se toma con…', ['el VAN de los flujos que quedan', 'el VAN incluyendo lo ya gastado', 'la suma sin descontar de los flujos futuros', 'lo gastado hasta hoy'],
     'C12: lo gastado es un coste hundido y no cambia con la decisión.'),
    ('«Cancelar ahora es tirar lo invertido» es…', ['un argumento irrelevante: lo invertido se pierde igual en ambos casos', 'el argumento decisivo', 'correcto si el VAN total es positivo', 'una reserva de contingencia'],
     'El coste hundido es el mismo continuando o cancelando.'),
    ('Sumar los flujos futuros sin descontar para decir que «recuperan la inversión»…', ['sobrestima su valor: hay que descontarlos al coste de capital', 'es correcto si los flujos son constantes', 'subestima su valor', 'es el criterio del payback descontado'],
     'Un euro dentro de n años vale menos que uno hoy.'),
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
gift = a_gift(preguntas_test, 'Ética 19 — coste hundido')
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
decision = 'continuar' if van_restante > 0 else 'cancelar'
rubrica_caso(
    [('Empresa (accionistas)', ('continuar aporta ' if van_restante > 0 else 'continuar resta ') + f'{abs(van_restante):.2f} k€ de valor desde hoy; los {gastado} k€ gastados están perdidos en cualquier caso'),
     ('Comité de dirección', 'necesita la cuenta correcta (VAN de lo que queda) para decidir, no la del borrador'),
     ('Jefe de proyecto y equipo', 'reconocer que el proyecto fue mala inversión les cuesta reputación a corto plazo; defenderlo con una cuenta falsa, más a largo plazo')],
    [('Enviar el borrador tal cual (propuesta)', 'sí', 'no: no resiste que alguien rehaga la cuenta', 'no: el comité no querría decidir con ella'),
     (f'Recomendar {decision} con el VAN de lo que queda y explicar el coste hundido', 'sí', 'sí', 'sí'),
     ('Presentar además el análisis de por qué falló la estimación inicial (lecciones aprendidas)', 'sí', 'sí', 'sí')]
    + ([] if van_restante > 0 else [('Cancelar y buscar comprador para el trabajo hecho', 'sí', 'sí', 'sí')]),
    (f'''párrafo para el comité: «Visto desde el inicio, el proyecto tiene un VAN de {van_total:.0f} k€: esa pérdida ya está hecha. La decisión de hoy es otra: terminarlo requiere {inversion_restante} k€ y los flujos previstos valen {va_flujos:.0f} k€ actuales, así que continuar {'aporta ' + f'{van_restante:.0f}' + ' k€. Recomendamos continuar y revisar por qué falló la estimación inicial.»' if van_restante > 0 else 'destruye ' + f'{-van_restante:.0f}' + ' k€ más. Recomendamos cancelar, documentar lo aprendido y, si hay comprador para el trabajo hecho, venderlo.»'} Al jefe de proyecto: reconocer el error es lo que protege su credibilidad para el siguiente proyecto; defender el proyecto con una cuenta que no se sostiene la destruye en cuanto alguien la rehace.'''))
