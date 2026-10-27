# ---
# title: Caso 20
# subtitle: "recursos compartidos: préstamo de 2 operarios en 7–8 (2 sem. de retraso) frente a 10–11"
# coleccion: 18_casos
# variante: recursos-compartidos
# nivel: 2
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
# Usted dirige un proyecto de instalaciones cuyo cuadro de prelaciones, duraciones (semanas) y operarios necesarios por actividad se adjunta. La empresa le ha asignado una cuadrilla fija de **11 operarios**, dimensionada justo para el pico de carga del plan de inicios más tempranos. El contrato con el cliente fija la entrega en la semana 16 con una penalización de **4 k€ por semana** de retraso. El proyecto empieza el lunes que viene (semana 1). El viernes recibe estos mensajes.

# %%
#| tbl-cap: "Plan del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, resources
    ('A'     , '---'      , 6       , 3        ),
    ('B'     , '---'      , 5       , 2        ),
    ('C'     , 'A,B'      , 3       , 4        ),
    ('D'     , 'B'        , 5       , 3        ),
    ('E'     , 'B'        , 4       , 4        ),
    ('F'     , 'C,E'      , 7       , 2        ),
    ('G'     , 'D'        , 3       , 3        ),
], columns=['activity', 'predecessor', 'duration', 'resources']).set_index('activity')
datos_enunciado

# %% [markdown]
# > **De: Jefa del proyecto Delta**
# > Tengo un problema serio: necesito **2 operarios de tu cuadrilla durante las semanas 7 a 8** (ambas incluidas) para un trabajo que no puedo retrasar. Antes de la semana 7 no me sirven. Si no los tengo, mi proyecto se retrasa esas 2 semanas y la penalización de mi cliente es de 10 k€ por semana; cada semana que tarden en llegar a partir de entonces es una semana de retraso para mí. ¿Me los puedes dejar?

# %% [markdown]
# > **De: Director de operaciones**
# > Lo del proyecto Delta es prioritario para la empresa: es un cliente estratégico y la penalización es alta. Entiendo que tu proyecto también tiene plazo. Dime cuánto te cuesta prestarlos y busquemos la opción más barata para la empresa en conjunto, no para cada proyecto por separado.

# %% [markdown]
# > **De: Cliente**
# > Les recordamos que la fecha de entrega (semana 16) está ligada a la apertura de nuestras instalaciones y no es negociable. La penalización de 4 k€ por semana se aplicará desde el primer día de retraso.

# %% [markdown]
# > **De: Jefe de obra**
# > He mirado el plan. En las semanas 7, 8 vamos al máximo de gente y la ventana que pide Delta las pisa; si nos quitan 2 operarios ahí, algo se retrasa. En cambio, a partir de la semana 10 tenemos más margen: ahí podríamos prestarlos sin que el plazo se mueva, si a Delta le vale.

# %% [markdown]
# > **De: Recursos humanos**
# > Podemos subcontratar operarios a 2.0 k€ por operario y semana, pero necesitamos **una semana de preaviso** para tenerlos en obra: como muy pronto, el lunes de la semana 2. Decidid pronto.

# %% [markdown]
# Responda a las preguntas siguientes **antes** de decidir. En la penúltima se le pide la decisión y cómo la comunicaría a cada uno de los implicados (jefa del proyecto Delta, director de operaciones, cliente, jefe de obra, recursos humanos).

# %%
pregunta_1 = "1. Dibuje el Gantt de inicios más tempranos con las cargas y el diagrama de cargas. ¿En qué semanas se alcanza el máximo de 11 operarios y qué actividades coinciden en ellas? ¿Cuáles son críticas?  "
pregunta_2 = "2. Simule el préstamo de 2 operarios en las semanas 7–8 con un planificador en paralelo (como Wiest-Levy) con un límite de recursos por periodo (11 fuera de la ventana, 9 dentro). ¿Qué actividades esperan, cuánto se alarga el proyecto y cuánto cuesta en penalizaciones?  "
pregunta_3 = "3. Repita la simulación para todas las ventanas posibles de 2 semanas consecutivas. ¿En cuáles se puede prestar sin retrasar el proyecto? Explique el resultado a partir de las cargas y las holguras. ¿Tiene razón el jefe de obra?  "
pregunta_4 = "4. Construya la tabla de opciones (negarse, prestar como se pide, prestar en la ventana del jefe de obra, subcontratar) con el coste de cada una para su proyecto, para el proyecto Delta y para la empresa. ¿Qué opción minimiza el coste de la empresa? ¿Coincide con la que minimiza el de su proyecto?  "
pregunta_5 = "5. Tome una decisión y redacte en tres o cuatro líneas el mensaje de respuesta a cada implicado.  "
pregunta_6 = "6. ¿Qué fallo de organización revela el caso y cómo se evitaría la próxima vez (reparto de recursos entre proyectos, reservas, prioridades)?  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
plazo, penalizacion, penalizacion_delta, coste_sub, q, ventana = 16, 4, 10, 2.0, 2, (7, 8)
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
calendario = proyecto.calendar(datos_proyecto['duration'])

# %% [markdown]
# El préstamo es un límite de recursos *por periodo*. El algoritmo de asignación de la librería (`asignar`, Wiest-Levy) no interrumpe actividades en curso, así que cuando el límite baja a mitad de una actividad no puede respetarlo; se usa aquí un planificador en paralelo equivalente que sólo deja empezar una actividad si cabe en todas las semanas que va a ocupar.

# %%
def programar(limite, L, registro=None):
    '''Planificador en paralelo con límite por periodo: la semana s admite limite[s] operarios (L si no está en limite).
    Recorre las semanas; en cada una, por orden de holgura total, entra cada actividad disponible (predecesoras terminadas)
    si cabe en el límite de TODAS las semanas que ocuparía; si no, espera. Devuelve los inicios y fines de cada actividad.
    Si se pasa una lista en registro, anota cada espera: la primera semana en la que la actividad no cabe y por qué.'''
    D = datos_proyecto['duration'].astype(int); R = datos_proyecto['resources'].astype(int)
    preds = prelaciones.immediate_predecessor
    orden = list(calendario['H_total'].sort_values(kind='stable').index)
    carga, inicio, fin, t = {}, {}, {}, 0
    while len(inicio) < len(D):
        for a in orden:
            if a in inicio or any(p not in fin or fin[p] > t for p in preds[a]):
                continue
            semanas = range(t + 1, t + int(D[a]) + 1)
            if all(carga.get(s, 0) + int(R[a]) <= int(limite.get(s, L)) for s in semanas):
                inicio[a], fin[a] = t, t + int(D[a])
                for s in semanas:
                    carga[s] = carga.get(s, 0) + int(R[a])
            elif registro is not None:
                s = next(s for s in semanas if carga.get(s, 0) + int(R[a]) > int(limite.get(s, L)))
                registro.append({'semana en que podría empezar': t + 1, 'actividad': a, 'operarios': int(R[a]),
                                 'semanas que ocuparía': f'{t + 1}–{t + int(D[a])}', 'primera semana sin sitio': s,
                                 'carga ya asignada + operarios > límite': f'{carga.get(s, 0)} + {int(R[a])} = {carga.get(s, 0) + int(R[a])} > {int(limite.get(s, L))}'})
        t += 1
    return pd.Series(inicio).reindex(D.index), pd.Series(fin).reindex(D.index)

def proyecto_programado(inicio, fin):
    '''Tabla con actividades de espera 💤 (duración = lo que la actividad espera tras sus predecesoras) para dibujar la programación.'''
    preds = prelaciones.immediate_predecessor
    filas = {}
    for a in datos_proyecto.index:
        ps = sorted(preds[a])
        listo = max([int(fin[p]) for p in ps], default=0)
        if int(inicio[a]) > listo:
            filas['💤' + a] = {'predecessor': ','.join(ps) or '---', 'duration': int(inicio[a]) - listo, 'resources': 0}
            ps = ['💤' + a]
        filas[a] = {'predecessor': ','.join(ps) or '---', 'duration': int(datos_proyecto.loc[a, 'duration']), 'resources': int(datos_proyecto.loc[a, 'resources'])}
    df = pd.DataFrame(filas).T; df.index.name = 'activity'
    df['duration'] = df['duration'].astype(int); df['resources'] = df['resources'].astype(int)
    return df, PredecessorTable.from_dataframe_of_strings(df, activity='activity', predecessor='predecessor').create_project()

def simular_prestamo(ventana, q, L):
    '''Límite L fuera de la ventana y L − q dentro. Devuelve (duración, esperas, dibujo del Gantt con la fila Máximo).'''
    limite = {s: L - q for s in range(ventana[0], ventana[1] + 1)}
    inicio, fin = programar(limite, L)
    df_prog, p_prog = proyecto_programado(inicio, fin)
    T_prog = int(fin.max())
    maximo = pd.DataFrame({s: [limite.get(s, L)] for s in range(1, T_prog + 1)}, index=['Máximo'])
    _, dibujo = p_prog.gantt(df_prog, 'duration', 'resources', total='fila', tikz=True, extra_rows=maximo, params=dict(period_width=1, **GANTT_PARAMS))
    esperas = {a[1:]: int(d) for a, d in df_prog['duration'].items() if a.startswith('💤')}
    return T_prog, esperas, dibujo

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Gantt de inicios más tempranos con los operarios de cada actividad; fila Total, carga por semana"
#| label: fig-ganttCargas
gantt_cargas, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas con la cuadrilla disponible"
#| label: fig-cargas
cargas = gantt_cargas.data.loc['Total'].drop('H_total').astype(float)
L = int(cargas.max())
plot_cargas(gantt_df=gantt_cargas, limite_maximo_recursos=[L])

# %%
picos = [int(t) for t in cargas.index[cargas == L]]
en_pico = {t: [a for a in datos_proyecto.index if calendario.loc[a, 'inicio_mas_temprano'] < t <= calendario.loc[a, 'fin_mas_temprano']] for t in picos}
criticas = list(calendario.index[calendario['H_total'] == 0])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# **Respuesta:** el plan de inicios más tempranos llega al máximo de 11 operarios en las semanas 7, 8 y 9; en todas coinciden C, D y E. Las actividades críticas (holgura total 0) son A, B, C, E y F. La ventana que pide Delta (7–8) incluye las semanas de pico 7 y 8. Prestar 2 operarios en una semana en la que la carga supera 9 obliga a aplazar alguna actividad; que eso retrase el proyecto depende de si la actividad aplazada tiene holgura suficiente (y de si al aplazarla arrastra a otras hasta agotar la suya).

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| fig-cap: "Programación con 9 operarios en las semanas 7–8 y 11 en las demás (fila Máximo); las filas 💤 son las esperas"
#| label: fig-ganttPrestamo
T_prestamo, esperas, dibujo = simular_prestamo(ventana, q, L)
pinta_tikz(dibujo)

# %% [markdown]
# El planificador recorre las semanas; al principio de cada una ordena las actividades disponibles (predecesoras terminadas) por holgura total, de menor a mayor (en caso de empate, por orden alfabético), y deja empezar cada una si cabe en el límite de todas las semanas que ocuparía (9 dentro de la ventana 7–8, 11 fuera). Cada vez que una actividad disponible no cabe, se anota la primera semana en la que se pasaría del límite:

# %%
#| tbl-cap: "Esperas con el préstamo pedido: por qué no cabe cada actividad"
#| label: tbl-esperasPrestamo
registro = []
_ = programar({s: L - q for s in range(ventana[0], ventana[1] + 1)}, L, registro)
pd.DataFrame(registro).set_index('semana en que podría empezar') if registro else Markdown('Ninguna actividad tiene que esperar.')

# %%
_ret = T_prestamo - plazo
_lineas = [f'- {a} espera {plural(v, "semana")} y tenía holgura total {int(calendario.loc[a, "H_total"])}.' for a, v in esperas.items()]
_criticas = [a for a, v in esperas.items() if calendario.loc[a, 'H_total'] == 0 and v == _ret]
if _ret == 0:
    _porque = 'las holguras absorben todas las esperas'
elif _criticas:
    _porque = lista_y(_criticas) + (' es crítica' if len(_criticas) == 1 else ' son críticas') + ': su espera pasa entera al plazo'
else:
    _porque = f'las esperas superan en {plural(_ret, "semana")} la holgura de las rutas que las contienen'
_quien = (('espera ' if len(esperas) == 1 else 'esperan ') + lista_y(list(esperas))) if esperas else 'no espera ninguna actividad'
Markdown('\n'.join(_lineas) + f'\n\n**Respuesta:** con el préstamo en las semanas {ventana[0]}–{ventana[1]} {_quien}; '
         f'el proyecto pasa de {plazo} a **{T_prestamo} semanas** ({_porque}): {plural(_ret, "semana")} de retraso y '
         f'{penalizacion} × {_ret} = **{penalizacion * _ret} k€** de penalización.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Duración del proyecto según la ventana del préstamo"
#| label: tbl-ventanas
w = 2
filas = []
for a in range(1, plazo - w + 2):
    T_a, esperas_a, _ = simular_prestamo((a, a + w - 1), q, L)
    cmax = max(int(cargas[t]) for t in range(a, a + w))
    hol = ', '.join(f'{x} ({v} sem.; holgura {int(calendario.loc[x, "H_total"])})' for x, v in esperas_a.items())
    if not esperas_a:
        motivo = f'carga máxima {cmax} ≤ {L - q}: no hay que aplazar nada'
    elif T_a == plazo:
        motivo = f'carga máxima {cmax} > {L - q}; esperan {hol}: lo absorbe la holgura'
    else:
        motivo = f'carga máxima {cmax} > {L - q}; esperan {hol}: no basta la holgura'
    filas.append({'semanas': f'{a}–{a + w - 1}', 'carga antes del préstamo': ', '.join(str(int(cargas[t])) for t in range(a, a + w)),
                  'duración': T_a, 'retraso': T_a - plazo, 'explicación': motivo})
ventanas = pd.DataFrame(filas).set_index('semanas')
ventanas

# %% [markdown]
# Se puede prestar sin mover el plazo en 1–2, 2–3, 3–4, 4–5, 5–6, 6–7, 10–11, 11–12, 12–13, 13–14, 14–15 y 15–16; retrasan el proyecto todas las que empiezan entre la semana 7 y la 9 (7–8 a 9–10). En las ventanas sin retraso, o la carga de esas semanas no pasa de 9 y no hay que aplazar nada, o lo que se aplaza tiene holgura suficiente (columna «explicación»). La ventana que pide Delta (7–8) cuesta 2 semanas porque su carga pasa de 9 y obliga a aplazar C, que es crítica.
#
# **Respuesta:** el jefe de obra tiene razón: a partir de la semana 10 todas las ventanas de 2 semanas dejan el plazo en 16 semanas, así que prestar desde entonces no cuesta nada a este proyecto (a Delta le cuesta esperar 3 semanas).

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Opciones y su coste (k€) para cada parte"
#| label: tbl-opciones
T_alt, _, _ = simular_prestamo((10, 11), q, L)
espera_delta = 10 - ventana[0]
subcontratar_posible = ventana[0] >= 2     # una semana de preaviso
opciones = pd.DataFrame([
    {'opción': 'negarse', 'mi proyecto': 0, 'proyecto Delta': penalizacion_delta * w, 'detalle': f'Delta se retrasa {w} semanas'},
    {'opción': f'prestar en {ventana[0]}–{ventana[1]}', 'mi proyecto': penalizacion * (T_prestamo - plazo), 'proyecto Delta': 0, 'detalle': f'mi proyecto se retrasa {T_prestamo - plazo}'},
    {'opción': 'prestar en 10–11', 'mi proyecto': penalizacion * (T_alt - plazo), 'proyecto Delta': penalizacion_delta * espera_delta, 'detalle': f'Delta espera {espera_delta} semana' + ('s' if espera_delta != 1 else '')},
    {'opción': f'subcontratar {q} para Delta', 'mi proyecto': 0, 'proyecto Delta': coste_sub * q * w if subcontratar_posible else np.inf,
     'detalle': 'preaviso suficiente' if subcontratar_posible else 'no llegan a tiempo (preaviso de 1 semana)'},
]).set_index('opción')
opciones.insert(2, 'empresa', opciones['mi proyecto'] + opciones['proyecto Delta'])
opciones

# %% [markdown]
# Coste de cada opción (penalizaciones y subcontratación, en k€):
#
# - Negarse: Delta se retrasa las 2 semanas → 10 × 2 = **20 k€** (mi proyecto, 0).
# - Prestar en 7–8: mi proyecto se retrasa 2 semanas → 4 × 2 = **8 k€** (Delta, 0).
# - Prestar en 10–11: mi plazo no se mueve (apartado 3); Delta espera 10 − 7 = 3 semanas → 10 × 3 = **30 k€**.
# - Subcontratar: 2 k€ por operario y semana × 2 × 2 semanas = **8.0 k€** (mi proyecto, 0); llega a tiempo porque la ventana empieza en la semana 7 y RR. HH. puede tenerlos desde la semana 2.
#
# **Respuesta:** para la empresa la opción más barata es **prestar en 7–8** y **subcontratar 2 para Delta** (8.0 k€, empatadas). Para mi proyecto cuestan 0 negarse, prestar en 10–11 y que se subcontrate para Delta; la única que me cuesta es prestar como se pide (8 k€). Por tanto el óptimo de la empresa es también uno de los óptimos de mi proyecto, pero no el único: si yo eligiera negarme, la empresa pagaría 12.0 k€ más. El director de operaciones pide la columna «empresa»: ahí está el conflicto entre la optimización local (cada jefe de proyecto defiende su plazo) y la global.

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# Una decisión defendible: no prestar en 7–8, pedir hoy a RR. HH. la subcontratación de 2 operarios para Delta (8.0 k€, mucho menos que cualquier penalización) y ofrecer además los míos desde la semana 10 por si fallara. Mensajes:
#
# - *Jefa de Delta:* «En 7–8 prestar 2 operarios me cuesta 2 semanas de plazo (8 k€). Lo más barato es que RR. HH. subcontrate 2 para ti (8.0 k€); lo pido hoy. Si fallara, te presto los míos desde la semana 10.»
# - *Director de operaciones:* «Adjunto la tabla con el coste de cada opción para los dos proyectos y para la empresa; la más barata en conjunto es subcontratar 2 para Delta (8.0 k€). Si decides otra, pido que la penalización resultante se registre como decisión de empresa.»
# - *Cliente:* «Mantenemos la entrega en la semana 16.»
# - *Jefe de obra:* «Tenías razón con la ventana 10–11; va en la propuesta como alternativa. Prepara el plan de personal con los dos escenarios.»
# - *RR. HH.:* «Necesito 2 operarios subcontratados para las semanas 7–8; confirmad hoy para respetar el preaviso.»

# %% [markdown]
# ## Apartado 6

# %%
display(Markdown(pregunta_6))

# %% [markdown]
# El fallo no es de cálculo sino de organización: dos proyectos comparten un recurso sin que nadie mire el perfil de carga *conjunto*. Lo que lo evita: un diagrama de cargas de la cartera (suma de los proyectos) revisado cada semana por el director de operaciones; reglas explícitas de prioridad entre proyectos antes de que surja el conflicto (no durante); una reserva de capacidad o un acuerdo marco de subcontratación con preaviso corto; y planes nivelados, no de inicios más tempranos, que rebajan los picos y dejan margen para prestar. Y, por supuesto, que la petición llegue con más de un fin de semana de antelación.
