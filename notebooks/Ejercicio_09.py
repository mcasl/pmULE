# ---
# title: Ejercicio 09
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
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, plot_cargas
from pmule import explica_camino_critico, explica_probabilidad, explica_intervalo, ordena_ruta, lista_y, plural

def _cargas(gantt, limites=None):
    """Carga de recursos por periodo (y, si se da, frente al límite, que se prolonga con su último valor)."""
    import numpy as _np, pandas as _pd
    _c = gantt.data.loc['Total'].drop('H_total').astype(float)
    _t = _pd.DataFrame({'Periodo': range(1, len(_c) + 1), 'Carga': _c.astype(int).values})
    if limites is not None:
        _l = list(limites) if isinstance(limites, (list, tuple)) else [limites]
        _t['Límite'] = (_l + [_l[-1]] * len(_c))[:len(_c)]
        _t['¿Supera?'] = _np.where(_t['Carga'] > _t['Límite'], 'sí', 'no')
    return _t.set_index('Periodo').T

def _tabla_varianzas(caminos, varianzas, calendario):
    """Varianza de cada ruta crítica, con las actividades en orden de ejecución y la suma término a término."""
    import pandas as _pd
    _f = {}
    for _r, _acts in caminos.items():
        _acts = ordena_ruta(_acts, calendario)
        _f[_r] = {'Ruta': ' – '.join(_acts),
                  'Varianza = suma': ' + '.join(f"σ²({a})" for a in _acts) + ' = '
                                     + ' + '.join(f"{varianzas[a]:g}" for a in _acts) + f" = {sum(varianzas[a] for a in _acts):g}"}
    return _pd.DataFrame(_f).T

def _respuesta_ackoff(best_option, costes, T0, T1, unidad='periodos'):
    """Coste de cada iteración de Ackoff-Sasieni y resumen de las reducciones."""
    from collections import Counter as _Counter
    _ci = {it: sum(costes[a] for a in acts) for it, acts in best_option.items()}
    _n = _Counter(a for acts in best_option.values() for a in acts)
    return Markdown(
        'Coste de cada iteración (suma de los costes unitarios de reducción de las actividades acortadas un periodo):\n\n'
        + '\n'.join(f"- Iteración {it}: " + ' + '.join(f"{a}({costes[a]:g})" for a in acts) + f" = {_ci[it]:g}" for it, acts in best_option.items())
        + "\n\n**Respuesta:** hay que reducir " + lista_y([f"{a} {plural(n, 'periodo')}" for a, n in _n.items()])
        + f"; la duración pasa de {T0:g} a {T1:g} {unidad} con un sobrecoste mínimo de "
        + ' + '.join(f"({c:g})" for c in _ci.values()) + f" = **{sum(_ci.values()):g}** unidades monetarias.")
# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# # Enunciado
#
# Considere el proyecto cuya información se adjunta.

# %%
periodo_probabilidad_a = 10
periodo_probabilidad_b = 12
limite_maximo_recursos = [10]*5 + [15]
pregunta_1 = "1. Dibuje los diagramas Pert y Roy, y determine el camino crítico.  "
pregunta_2 = f"2. ¿Cuál será la probabilidad de terminar el proyecto entre {periodo_probabilidad_a} y {periodo_probabilidad_b} periodos. ¿Y la de terminar en 9 periodos?.  "
pregunta_3 = "3. Como no está conforme con la duración del proyecto que resulta, desea reducir la duración del proyecto en 2 periodos con el mínimo incremento de coste posible. ¿Qué tareas deben ser reducidas?. Razónese.  "
pregunta_4 = f"""4. Debido a que la empresa está realizando otros dos proyectos al mismo tiempo, la disponibilidad de mano de obra no es todo lo grande que sería de desear, por lo que sólo se puede contar con {limite_maximo_recursos[0]} operarios durante las cinco primeras semanas, y con 15 operarios el resto de semanas .  
    - ¿Se encuentra bien nivelado el recurso “mano de obra”?.  ¿Por qué?.
    - ¿Se encuentra bien asignado este recurso?. ¿Por qué?.
    - En caso de que no estuvieran bien asignados, proponer una posible modificación de la planificación de las tareas para una correcta asignación.  """

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

# Alargamos el ancho de las celdas al 100%
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, desviacion, cur,   recursos, duracion_tope
  (   'A',      '---  ',      3,         1,        25,       3,        2    ),
  (   'B',      '---  ',      1,         1,        20,       1,        1    ),
  (   'C',      '---  ',      2,         1,         7,       6,        1    ),
  (   'D',      '---  ',      1,         1,        25,       1,        1    ),
  (   'E',      'A,B  ',      3,         2,        10,       2,        2    ),
  (   'F',      'B,C  ',      5,         1,        25,       5,        3    ),
  (   'G',      'C,D  ',      1,         1,        30,       2,        1    ),
  (   'H',      'E    ',      1,         1,         5,       5,        1    ),
  (   'I',      'G    ',      3,         1,        50,       3,        2    ),
  (   'J',      'F,H,I',      2,         1,         5,      10,        1    ),
  (   'K',      'G    ',      1,         1,        10,       5,        1    ),
] ,columns=['activity', 'predecessor', 'duration', 'standard_deviation', 'ucr', 'resources', 'cap_duration']).set_index('activity')

datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
from pmule import PredecessorTable, ProjectGraph

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
proyecto = prelaciones.create_project()

# %% [markdown]
#

# %% [markdown]
# ### Cuadro de prelaciones
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto.display_pert_construccion(tabla=prelaciones)

# %% [markdown]
# ### Matriz de Zaderenko

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duration'])

# %% [markdown]
# Los tiempos tempranos y tardios han resultado ser:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
resultados_pert['nodes']

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duration']))

# %%
Markdown(f"El tiempo más temprano del nodo final del proyecto es su duración media: **{duracion_media_proyecto} periodos**.")

# %% [markdown]
# ### Holguras
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
display(resultados_pert['activities'][['H_total']].rename(columns={'H_total': 'Holgura total'}))
display(Markdown('(Las filas que empiezan por @ son actividades ficticias.) Actividades con holgura total nula: **'
                 + lista_y([a for a in ordena_ruta(datos_proyecto.index, proyecto.calendar(datos_proyecto['duration'])) if resultados_pert['activities'].loc[a, 'H_total'] == 0]) + '**.'))
# %% [markdown]
# ### Camino crítico

# %% [markdown]
# Las rutas del camino crítico (actividades en orden de ejecución) son en este caso:

# %%
camino_critico = proyecto.critical_path(durations=datos_proyecto['duration'])
for key, value in camino_critico.items():
    pass
explica_camino_critico(proyecto, datos_proyecto['duration'], 'periodos')
# %% [markdown]
# ### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto, con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ### Grafo PERT con indicación de tiempos y rutas críticas

# %% [markdown]
# A continuación se muestra el grafo Pert del proyecto, con indicación de las actividades críticas:

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación del camino crítico"
#| label: fig-pertrutasCriticas
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# A continuación se muestra el grafo Roy del proyecto, con indicación de las actividades críticas:

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación de las actividades críticas"
#| label: fig-RoyCríticas
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duracion_media_proyecto = proyecto.duration(durations=datos_proyecto['duration'])

# %%
Markdown(f"Ya conocemos el valor medio de la duración del proyecto: es la duración calculada con las duraciones medias de las actividades, μ = **{duracion_media_proyecto:g} periodos**.")

# %% [markdown]
# Para determinar la desviación típica de la duración del proyecto debemos obtener, en cada rama del camino crítico, la suma de las varianzas de sus actividades (la varianza es el cuadrado de la desviación típica del enunciado), y quedarnos con la mayor.

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
_tabla_varianzas(camino_critico, datos_proyecto['variance'], calendario)
# %%
_v = varianza_caminos.data['Variance'].astype(float)
Markdown(f"Varianza del proyecto: σ² = máx({', '.join(f'{x:g}' for x in _v)}) = {desviacion_proyecto**2:g}; desviación típica σ = √{desviacion_proyecto**2:g} = {desviacion_proyecto:.2f}. "
         f"La duración del proyecto se aproxima por una normal de media **{duracion_media_proyecto:g}** y desviación típica **{desviacion_proyecto:.2f}**.")

# %%
Markdown(f"Como la media coincide con una de las duraciones por las que se pregunta ({duracion_media_proyecto:g} periodos), la probabilidad de terminar en "
         f"{duracion_media_proyecto:g} periodos es del 50 % sin necesidad de cálculo: la normal es simétrica y deja la mitad de los casos a cada lado de su media. "
         f"Comprobación: " + explica_probabilidad(duracion_media_proyecto, desviacion_proyecto, duracion_media_proyecto))

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(norm.cdf(12, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)
probabilidad_de_terminar_antes_de_a = round(norm.cdf(10, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)
probabilidad_de_terminar_entre_a_y_b = round(probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a, 2)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, entre=(10, 12))

# %%
Markdown(f"""Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, la probabilidad de terminar entre {periodo_probabilidad_a} y {periodo_probabilidad_b} periodos es la diferencia entre la probabilidad de terminar antes de {periodo_probabilidad_b} y la de terminar antes de {periodo_probabilidad_a}:

{explica_intervalo(duracion_media_proyecto, desviacion_proyecto, periodo_probabilidad_a, periodo_probabilidad_b)}

**Respuesta:** la probabilidad de terminar entre {periodo_probabilidad_a} y {periodo_probabilidad_b} periodos es **{probabilidad_de_terminar_entre_a_y_b:.2f}** (≈ {int(round(probabilidad_de_terminar_entre_a_y_b*100))} %), y la de terminar en {duracion_media_proyecto:g} periodos, **0.50** (50 %).
""")

# %% [markdown]
# ## Apartado 3.
#

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Aplicamos en este apartado el algoritmo de Ackoff y Sasieni para la reducción del proyecto con mínimo incremento de coste:

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                                                           min_durations=datos_proyecto['cap_duration'],
                                                           costs=datos_proyecto['ucr'],
                                                           reduction=2
                                                           )

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %% [markdown]
# Las actividades a recortar en cada iteración son:

# %%
for iteracion, actividades in best_option.items():
    pass
_d = datos_proyecto['duration'].astype(float).copy()
_cal = proyecto.calendar(datos_proyecto['duration'])
_razones = []
for _it, _acts in best_option.items():
    _cc = proyecto.critical_path(_d)
    _rutas = '; '.join(f"{r} ({' – '.join(ordena_ruta(a, _cal))})" for r, a in _cc.items())
    _cubre = {r: [x for x in _acts if x in a] for r, a in _cc.items()}
    _razones.append(f"- Iteración {_it}: el proyecto dura {proyecto.duration(_d):g} periodos y tiene {plural(len(_cc), 'ruta crítica', 'rutas críticas')}: {_rutas}. "
                    f"Hay que acortar a la vez todas las rutas críticas; la combinación más barata entre las actividades que aún pueden reducirse es "
                    f"**{lista_y(_acts)}** (" + '; '.join(f"{r}: {lista_y(x)}" for r, x in _cubre.items()) + ")"
                    + (f", porque {_acts[0]} pertenece a todas las rutas críticas y es la más barata." if len(_acts) == 1 and len(_cc) > 1 else "."))
    _d[list(_acts)] -= 1
display(Markdown('\n'.join(_razones)))
_respuesta_ackoff(best_option, datos_proyecto['ucr'], proyecto.duration(datos_proyecto['duration']), proyecto.duration(duration))
# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
proyecto_asignacion = proyecto.copy()

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
gantt_recursos, dibujo = proyecto_asignacion.gantt(datos_proyecto,
                                'duration',
                                'resources',
                                total='fila',
                                holguras=True,
                                tikz=True)
pinta_tikz(dibujo)

# %% [markdown]
# La fila "Total" obtenida, que podemos representar gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_recursos, limite_maximo_recursos)
_lim_enunciado = [limite_maximo_recursos[0]] * 5 + [15]
_c0 = _cargas(gantt_recursos, _lim_enunciado)
display(Markdown(f"Carga de cada periodo con la programación más temprana, frente a la disponibilidad del enunciado ({_lim_enunciado[0]} operarios las 5 primeras semanas y 15 después):"))
display(_c0)
_niv = proyecto.copy().nivelar(data=datos_enunciado, duration_label='duration', resource_label='resources')
_sc = _niv[1].data if hasattr(_niv[1], 'data') else _niv[1]
_movn = {a: int(v) for a, v in _niv[0].iloc[:, 0].items() if v != 0}
_exc = [p for p in _c0.columns if _c0.loc['¿Supera?', p] == 'sí']
display(Markdown(
    f"- **¿Bien nivelado?** No: la carga oscila entre {_c0.loc['Carga'].min()} y {_c0.loc['Carga'].max()} operarios y su suma de cuadrados es "
    + ' + '.join(f"{c}²" for c in _c0.loc['Carga']) + f" = {int((_c0.loc['Carga'] ** 2).sum())}; aplicando el algoritmo de nivelación por mínimos cuadrados, "
    + "desplazando " + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _movn.items()]) + f" dentro de su holgura baja a {float(_sc.iloc[-1].min()):g}, "
    + "luego la programación más temprana no es la mejor nivelada.\n"
    + (f"- **¿Bien asignado?** No: se supera la disponibilidad en {plural(len(_exc), 'periodo')} ("
       + lista_y([f"periodo {p}: {_c0.loc['Carga', p]} > {_c0.loc['Límite', p]}" for p in _exc]) + "), así que hay que aplicar el algoritmo de asignación."
       if _exc else "- **¿Bien asignado?** Sí: la carga no supera la disponibilidad en ningún periodo.")))
# %% [markdown]
# Como la programación inicial no respeta la disponibilidad de operarios, se aplica el algoritmo de asignación.

# %% [markdown]
# ### Algoritmo de asignación

# %%
duracion_y_recursos = datos_proyecto[['duration', 'resources']]
proyecto_asignacion.display_asignar_pasos_tikz(data=duracion_y_recursos, duration_label='duration', resource_label='resources', maximo=limite_maximo_recursos)
datos_proyecto_tras_asignar, gantt_df, dibujo  = proyecto_asignacion.asignar(
                                    data=duracion_y_recursos,
                                    duration_label='duration',
                                    resource_label='resources',
                                    maximo=limite_maximo_recursos,
                                    report=False
                                    )

# %%
#| fig-cap: "Diagrama de Gantt del proyecto una vez aplicado el algoritmo de asignación"
#| label: fig-postAsignacion
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)
_c1 = _cargas(gantt_df, _lim_enunciado)
display(_c1)
# %% [markdown]
# En esta situación el proyecto ya es realizable: la carga no supera los 10 operarios en las 5 primeras semanas ni los 15 en las siguientes (tabla anterior).

# %% [markdown]
# Duración y caminos críticos tras la asignación (las esperas 💤 son los retrasos introducidos):

# %%
_T1 = proyecto_asignacion.duration(datos_proyecto_tras_asignar['duration'])
_desp = {a[1:]: int(d) for a, d in datos_proyecto_tras_asignar['duration'].items() if str(a).startswith('💤')}
display(explica_camino_critico(proyecto, duracion_y_recursos['duration'], 'periodos'))
_cal1 = proyecto_asignacion.calendar(datos_proyecto_tras_asignar['duration'])
_hb = {a: (int(calendario.loc[a, 'H_total']), int(_cal1.loc[a, 'H_total'])) for a in datos_proyecto.index
       if _cal1.loc[a, 'H_total'] < calendario.loc[a, 'H_total']}
display(Markdown(
    "**Respuesta (propuesta de asignación):** retrasar " + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _desp.items()])
    + f" respecto de su inicio más temprano. La duración del proyecto {'no cambia' if _T1 == duracion_media_proyecto else 'pasa a ' + str(_T1)} "
    + f"({_T1:g} periodos) porque los retrasos se absorben con la holgura de esas actividades; a cambio disminuye la holgura total de "
    + lista_y([f"{a} (de {h0} a {h1})" for a, (h0, h1) in _hb.items()]) + "."))
# %% [markdown]
# Al consumir parte de la holgura, esas actividades quedan más cerca de ser críticas: cualquier retraso adicional en ellas puede retrasar el proyecto.

# %%
