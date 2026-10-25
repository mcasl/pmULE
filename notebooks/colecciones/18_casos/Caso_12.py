# ---
# title: Caso 12
# subtitle: "límite de 3 operarios, plazo mes 18, penalización 10 k€/mes y VAN"
# coleccion: 18_casos
# variante: contingencia-planta
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
# Usted dirige la construcción de una planta cuyo cuadro de prelaciones, duraciones (meses; *a*, *m* y *b* son las estimaciones mínima, más probable y máxima de una distribución triangular y *duration* la duración esperada), operarios por actividad, presupuesto (BAC, k€), duración tope y coste unitario de reducción (CUR, k€ por mes) se adjuntan. La cuadrilla fija es de **3 operarios**. Cuando la planta se entrega, genera un flujo neto de 25 k€ al mes durante 48 meses; la empresa descuenta al 12% anual. El viernes recibe estos mensajes.

# %%
#| tbl-cap: "Datos del proyecto"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  activity, predecessor, duration, a, m, b, operarios, BAC, cap_duration, ucr   
    ('A'     , '---'      , 3       , 1, 2, 6, 3        , 150, 3           , np.inf),
    ('B'     , '---'      , 2       , 1, 2, 3, 1        , 100, 1           , 5     ),
    ('C'     , 'A,B'      , 1       , 1, 1, 1, 1        , 150, 1           , np.inf),
    ('D'     , 'C'        , 5       , 3, 5, 7, 2        , 75 , 5           , np.inf),
    ('E'     , 'D'        , 6       , 2, 7, 9, 1        , 50 , 4           , 7     ),
    ('F'     , 'B'        , 8       , 7, 8, 9, 1        , 75 , 6           , 6     ),
    ('G'     , 'D'        , 5       , 2, 4, 9, 1        , 100, 2           , 5     ),
    ('H'     , 'E,F,G'    , 2       , 1, 2, 3, 2        , 125, 1           , 3     ),
], columns=['activity', 'predecessor', 'duration', 'a', 'm', 'b', 'operarios', 'BAC', 'cap_duration', 'ucr']).set_index('activity')
datos_enunciado

# %% [markdown]
# > **De: Cliente**
# > El contrato fija la entrega en el mes 18 con una penalización de **10 k€ por mes de retraso**. Les pedimos que nos confirmen la probabilidad real de cumplirlo, no la fecha del plan.

# %% [markdown]
# > **De: Director financiero**
# > No autorizaré ninguna medida cuyo VAN esperado, neto de penalizaciones y de lo que cueste, no supere al de no hacer nada. Y quiero saber la probabilidad de que el VAN sea negativo en cada opción.

# %% [markdown]
# > **De: Jefe de obra**
# > Podemos subcontratar **un operario más** durante todo el proyecto por 3 k€ al mes. Con la cuadrilla actual, por mucho que acorten las actividades, a veces se quedan esperando por falta de gente.

# %% [markdown]
# > **De: Contratista**
# > Podemos acelerar actividades con horas extra y refuerzos a los CUR indicados en la tabla (hasta la duración tope). Díganos cuántos meses necesita recortar y le decimos qué actividades.

# %% [markdown]
# Responda a las preguntas siguientes **antes** de decidir. En la penúltima se le pide la decisión y cómo la comunicaría a cada uno de los implicados (cliente, director financiero, jefe de obra, contratista).

# %%
pregunta_1 = "1. Calcule la duración con las duraciones esperadas sin límite de recursos y con el límite de operarios. ¿Cuál es la cadena crítica con recursos y qué actividades esperan?  "
pregunta_2 = "2. Simule 3 000 ejecuciones con duraciones triangulares y con el límite de operarios (replanificando en cada réplica) y estime la probabilidad de entregar en el mes 18, el VAN esperado neto de penalización y P(VAN < 0) si no se hace nada.  "
pregunta_3 = "3. Evalúe cuatro opciones con los mismos números aleatorios: (A) no hacer nada, (B) contratar un operario más, (C) acelerar dos meses con el contratista (reducción con recursos limitados) y (D) las dos cosas a la vez. Tabla de VAN esperado neto, P(VAN < 0), probabilidad de cumplir el plazo, duración media y coste de la aceleración.  "
pregunta_4 = "4. ¿Qué opción recomienda al director financiero? ¿Y al cliente? ¿Se suman los efectos de B y C en D? Explique por qué.  "
pregunta_5 = "5. Redacte en tres o cuatro líneas la respuesta a cada implicado.  "
pregunta_6 = "6. ¿Qué ha cambiado respecto a un análisis determinista con el camino crítico? Señale al menos tres decisiones que habría tomado mal sin recursos, sin incertidumbre o sin VAN.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()
L, plazo, penalizacion, coste_operario, F, N, k = 3, 18, 10.0, 3.0, 25.0, 48, 0.12
base = datos_proyecto[['predecessor', 'duration', 'operarios']]
tope, ucr = datos_proyecto['cap_duration'], datos_proyecto['ucr']
dist_base = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| fig-cap: "Grafo Roy con las duraciones esperadas y el camino crítico (sin límite de operarios)"
#| label: fig-pert
proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso (avance con los botones):

# %%
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %%
#| tbl-cap: "Calendario sin límite de recursos (meses)"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]

# %%
#| fig-cap: "Carga de operarios con los inicios más tempranos frente al límite"
#| label: fig-cargasInicial
gantt_inicial, dibujo = proyecto.copy().gantt_cargas(base.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)
plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[L])

# %% [markdown]
# El algoritmo de asignación recorre los meses: cada actividad disponible (predecesoras terminadas) empieza, por orden de prioridad (menor inicio más tardío), si caben sus operarios durante toda su duración; si no, espera. Traza mes a mes y Gantt resultante (filas 💤: esperas):

# %%
res0 = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], [L], report=True, tikz=True, params=GANTT_PARAMS)

# %%
#| fig-cap: "Gantt con el límite de operarios y las duraciones esperadas"
#| label: fig-ganttAsignado
pinta_tikz(res0.dibujos['operarios'])

# %%
#| tbl-cap: "Inicio de cada actividad sin y con límite"
#| label: tbl-inicios
pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res0.inicios, 'fin con límite': res0.fines,
              'espera (meses)': res0.esperas}).fillna(0).astype(int).T

# %%
duracion_libre = proyecto.duration(datos_proyecto['duration'])
cadena = proyecto.copy().cadena_critica_recursos(base.copy(), 'duration', ['operarios'], [L])
display(cadena['arcos'].rename(columns={'motivo': 'motivo (precedencia o recurso)'}))
esperas = res0.traza[res0.traza.decision == 'espera']
Markdown(f'Sin límite {duracion_libre} meses; con {L} operarios {res0.duracion}. Cadena crítica con recursos: '
         f'{", ".join(sorted(cadena["actividades"]))}. Esperan: {", ".join(sorted(set(esperas.actividad)))}.')

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
def evaluar(L_op, red, n=3000, semilla=21):
    '''VAN neto (tras penalización y operarios extra) y probabilidad de cumplir de una opción (mismos números aleatorios).'''
    dist = {a: ('triangular', max(1.0, d[1] - red[a]), max(1.0, d[2] - red[a]), max(1.0, d[3] - red[a])) for a, d in dist_base.items()}
    bac = datos_proyecto['BAC'] + ucr.replace(np.inf, 0.0) * red
    mc = proyecto.monte_carlo_van_recursos(dist, datos_proyecto[['operarios']], ['operarios'], [L_op], bac, F, N, k, n_samples=n, seed=semilla)
    tau = mc.tau_recursos_samples
    neto = mc.van_recursos_samples - penalizacion * np.maximum(tau - plazo, 0.0) - coste_operario * (L_op - L) * tau
    return {'VAN medio': neto.mean(), 'P(VAN<0)': (neto < 0).mean(), 'P(τ≤plazo)': (tau <= plazo).mean(),
            'τ medio': tau.mean(), 'coste reducción': float((ucr.replace(np.inf, 0.0) * red).sum())}

# %%
ceros = pd.Series(0.0, index=datos_proyecto.index)
A = evaluar(L, ceros)
Markdown(f'Sin hacer nada: P(entregar en el mes {plazo}) = {A["P(τ≤plazo)"]:.3f}; duración media {A["τ medio"]:.2f} meses; '
         f'VAN esperado neto {A["VAN medio"]:.1f} k€; P(VAN<0) = {A["P(VAN<0)"]:.3f}.')

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Las reducciones de C y D se calculan con `crashing_recursos`: en cada paso se programa con el límite, se halla la cadena crítica con recursos, se prueban como candidatos sus actividades reducibles (sueltas, por parejas y el corte de coste mínimo) replanificando con cada uno, y se elige el más barato que de verdad acorta el proyecto replanificado. Paso a paso, con el límite actual (C) y con un operario más (D):

# %%
#| tbl-cap: "Crashing con recursos para la opción C (límite actual)"
#| label: tbl-crashingC
cr_c = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], [L], tope, ucr, reduccion=2)
cr_c.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]

# %%
#| tbl-cap: "Crashing con recursos para la opción D (un operario más)"
#| label: tbl-crashingD
cr_d = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], [L + 1], tope, ucr, reduccion=2)
cr_d.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]

# %%
red_c = pd.Series(cr_c.reducciones, dtype=float).reindex(datos_proyecto.index).fillna(0.0)
red_d = pd.Series(cr_d.reducciones, dtype=float).reindex(datos_proyecto.index).fillna(0.0)
opciones = {'A: no hacer nada': (L, ceros), 'B: un operario más': (L + 1, ceros),
            'C: acelerar dos meses': (L, red_c), 'D: operario más y acelerar': (L + 1, red_d)}
tabla = pd.DataFrame({n: evaluar(Lo, r) for n, (Lo, r) in opciones.items()}).T
tabla

# %%
display(pd.DataFrame({'C: reducciones': red_c[red_c > 0], 'D: reducciones': red_d[red_d > 0]}).fillna(0).astype(int).T)
Markdown('Las reducciones se calculan con `crashing_recursos`: se acorta lo que de verdad acorta el proyecto con el límite de operarios, no lo más barato.')

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
mejor = tabla['VAN medio'].idxmax()
ok_cliente = tabla[tabla['P(τ≤plazo)'] >= 0.7].index.tolist()
suma = (tabla.loc['B: un operario más', 'VAN medio'] - tabla.loc['A: no hacer nada', 'VAN medio']
        + tabla.loc['C: acelerar dos meses', 'VAN medio'] - tabla.loc['A: no hacer nada', 'VAN medio'])
real = tabla.loc['D: operario más y acelerar', 'VAN medio'] - tabla.loc['A: no hacer nada', 'VAN medio']
Markdown(f'Mayor VAN esperado: {mejor} ({tabla.loc[mejor, "VAN medio"]:.1f} k€). Opciones con P(cumplir) ≥ 70 %: {ok_cliente or "ninguna"}. '
         f'Respecto a A, B y C por separado suponen {suma:+.1f} k€ en total y juntas (D) {real:+.1f} k€: los efectos no se suman, '
         'porque las dos medidas atacan el mismo cuello de botella (esperas por operarios y duración de la cadena crítica).')

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# **Cliente:** con la cuadrilla actual la probabilidad de cumplir es baja; la opción recomendada la sube a la cifra de la tabla. **Director financiero:** el VAN esperado neto de penalización y de coste es el de la tabla, con la probabilidad de VAN negativo indicada. **Jefe de obra:** se contrata un operario sólo si la tabla lo respalda; las esperas se concentran en la cadena crítica con recursos. **Contratista:** se aceleran únicamente las actividades que de verdad acortan el proyecto con el límite de operarios.

# %% [markdown]
# ## Apartado 6

# %%
display(Markdown(pregunta_6))

# %% [markdown]
# Con el camino crítico y duraciones esperadas se habría dicho que el plazo se cumple (sin límite de recursos y sin incertidumbre); se habría acelerado la actividad más barata aunque no acorte el proyecto; y se habría juzgado cada medida por separado sin ver que se solapan. Los recursos fijan la duración real, la incertidumbre convierte la fecha en una probabilidad y el VAN obliga a comparar el coste de cada medida con lo que vale cobrar antes.
