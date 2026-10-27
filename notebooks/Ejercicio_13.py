# ---
# title: Ejercicio 13
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
from pmule import explica_camino_critico, explica_intervalo, explica_probabilidad, ordena_ruta, lista_y, plural

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

def _respuesta_nivelacion(cuadrados, desplazamientos):
    """Respuesta de la nivelación: suma de cuadrados inicial y final y actividades desplazadas."""
    _sc = cuadrados.data if hasattr(cuadrados, 'data') else cuadrados
    _sc0, _scf = float(_sc.iloc[0, 0]), float(_sc.iloc[-1].min())
    _mov = {a: int(v) for a, v in desplazamientos.iloc[:, 0].items() if v != 0}
    if not _mov:
        return Markdown(f"**Respuesta:** los recursos ya estaban nivelados: ningún desplazamiento reduce la suma de cuadrados ({_sc0:g}).")
    return Markdown(f"**Respuesta:** para nivelar los recursos se desplaza " + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _mov.items()])
                    + f" respecto de su inicio más temprano; la suma de cuadrados de las cargas baja de {_sc0:g} a **{_scf:g}**. "
                    "Las demás actividades no críticas no se desplazan por sí mismas (solo lo que las arrastren sus predecesoras) porque desplazarlas no reduce más la suma de cuadrados; "
                    "como solo se mueven actividades dentro de su holgura, la duración del proyecto no cambia.")

def _respuesta_ackoff(best_option, costes, T0, T1, unidad='periodos'):
    """Coste de cada iteración de Ackoff-Sasieni y resumen de las reducciones."""
    from collections import Counter as _Counter
    _ci = {it: sum(costes[a] for a in acts) for it, acts in best_option.items()}
    _n = _Counter(a for acts in best_option.values() for a in acts)
    return Markdown(
        'Coste de cada iteración (suma de los costes unitarios de reducción de las actividades acortadas un periodo):\n\n'
        + '\n'.join(f"- Iteración {it}: " + ' + '.join(f"{a}({costes[a]:g})" for a in acts) + f" = {_ci[it]:g}" for it, acts in best_option.items())
        + "\n\n**Respuesta:** hay que reducir " + lista_y([f"{a} {plural(n, unidad[:-1])}" for a, n in _n.items()])
        + f"; la duración pasa de {T0:g} a {T1:g} {unidad} con un sobrecoste mínimo de "
        + ' + '.join(f"({c:g})" for c in _ci.values()) + f" = **{sum(_ci.values()):g}** unidades monetarias.")
# %% [markdown]
# # Enunciado
# Considere la información del proyecto cuya información aparece en la tabla adjunta. Considere en cada apartado que el proyecto se encuentra en su situación inalterada mostrada en los datos del proyecto.

# %%
periodo_a_probabilidad = 17
periodo_b_probabilidad = 19
duracion_objetivo = 18
limite_maximo_recursos = 6
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} semanas?  "
pregunta_2 = f"2. Reduzca, con mínimo incremento de coste la duración media del proyecto a {duracion_objetivo} semanas. ¿Cuál será ahora la probabilidad de terminar en {duracion_objetivo} semanas? ¿Por qué?  "
pregunta_3 = f"3. ¿Se encuentran nivelados los recursos consumidos por el proyecto? De no ser así proceda a su nivelación.  "
pregunta_4 = f"4. Si su empresa sólo puede destinar a {limite_maximo_recursos} personas al proyecto, ¿se encuentran los recursos de mano de obra correctamente asignados? Reprográmelo si fuera necesario de modo que se cumpla esta restricción.  "

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

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, varianza,  cur,  recursos, duracion_tope
     ('A',       '---  ',      2,        1,     -100,     2,         1        ),
     ('B',       '---  ',      4,        1,       30,     4,         2        ),
     ('C',       'B    ',      1,        1,   np.inf,     2,         1        ),
     ('D',       'A    ',      3,        1,       50,     2,         1        ),
     ('E',       'B,D  ',      7,        1,       20,     2,         4        ),
     ('F',       'C    ',      4,        1,       10,     2,         2        ),
     ('G',       '---  ',      1,        2,      100,     4,         1        ),
     ('H',       'G    ',     11,        2,       10,     2,         8        ),
     ('I',       'E,F,H',      2,        1,       50,     2,         1        ),
     ('J',       'E,F,H',      8,        1,        5,     2,         7        ),
] ,columns=['activity', 'predecessor', 'duration', 'variance', 'ucr', 'resources', 'crash_duration']).set_index('activity')

datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadro de prelaciones
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %%
proyecto = prelaciones.create_project()

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

# %% [markdown]
# ### Duración media del proyecto

# %% [markdown]
# La duración media del proyecto es:

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duration']))
Markdown(f"El tiempo más temprano del nodo final del proyecto es su duración media: **{duracion_media_proyecto} semanas**.")

# %% [markdown]
# ### Holgura total de las actividades
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
display(resultados_pert['activities'].rename(columns={'H_total': 'Holgura total'}))
display(Markdown('(Las filas que empiezan por @ son actividades ficticias.) Actividades con holgura total nula: **'
                 + lista_y([a for a in ordena_ruta(datos_proyecto.index, proyecto.calendar(datos_proyecto['duration'])) if resultados_pert['activities'].loc[a, 'H_total'] == 0]) + '**.'))
# %% [markdown]
# ### Camino crítico
#
# El camino crítico consta de las siguientes rutas (actividades en orden de ejecución):

# %%
explica_camino_critico(proyecto, datos_proyecto['duration'], 'semanas')
# %% [markdown]
# ### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto, con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ### Grafo PERT con indicación de tiempos y rutas del camino crítico de acuerdo a la planificación inicial
# A continuación se muestra el grafo Pert del proyecto, con indicación de las actividades críticas:

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación del camino crítico"
#| label: fig-pertrutasCriticas
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# ### Caracterización de la duración del proyecto mediante una distribución Normal
# A fin de calcular la probabilidad pedida, es necesario caracterizar la distribución Normal de la duración del proyecto. Esto consiste en determinar los valores de sus parámetros: media y desviación típica de la duración del proyecto.

# %%
duracion_media_proyecto = int(proyecto.duration(datos_proyecto['duration']))
Markdown(f"La duración media del proyecto ya la tenemos calculada: μ = **{duracion_media_proyecto} semanas**.")

# %% [markdown]
# Para determinar la desviación típica debemos obtener en cada rama del camino crítico la suma de las varianzas de sus actividades (dato del enunciado), y quedarnos con la mayor.

# %%
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
_cc1 = proyecto.critical_path(datos_proyecto['duration'])
_tabla_varianzas(_cc1, datos_proyecto['variance'], calendario)
# %%
_v = varianza_rutas.data['Variance'].astype(float)
Markdown(f"Varianza del proyecto: σ² = máx({', '.join(f'{x:g}' for x in _v)}) = {desviacion_proyecto**2:g}; desviación típica σ = √{desviacion_proyecto**2:g} = {desviacion_proyecto:.2f} semanas. "
         f"La duración del proyecto sigue aproximadamente una normal de media **{duracion_media_proyecto} semanas** y desviación típica **{desviacion_proyecto:.2f}**.")

# %% [markdown]
# ### Probabilidad de terminar entre 17 y 19 periodos

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(norm.cdf(periodo_b_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)
probabilidad_de_terminar_antes_de_a = round(norm.cdf(periodo_a_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)

probabilidad_de_terminar_entre_a_y_b = probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, entre=(periodo_a_probabilidad, periodo_b_probabilidad))

# %%
Markdown(f"""Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, la probabilidad de terminar entre {periodo_a_probabilidad} y {periodo_b_probabilidad} semanas es la diferencia entre la de terminar antes de {periodo_b_probabilidad} y la de terminar antes de {periodo_a_probabilidad}:

{explica_intervalo(duracion_media_proyecto, desviacion_proyecto, periodo_a_probabilidad, periodo_b_probabilidad)}

**Respuesta:** la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} semanas es **{norm.cdf(periodo_b_probabilidad, duracion_media_proyecto, desviacion_proyecto) - norm.cdf(periodo_a_probabilidad, duracion_media_proyecto, desviacion_proyecto):.2f}** (≈ {int(round(probabilidad_de_terminar_entre_a_y_b*100))} %).
""")

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# ### Reducción a la duración objetivo con mínimo incremento de coste

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                min_durations=datos_proyecto['crash_duration'],
                costs=datos_proyecto['ucr'],
                reduction=2)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['crash_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)
display(_respuesta_ackoff(best_option, datos_proyecto['ucr'], proyecto.duration(datos_proyecto['duration']), proyecto.duration(duration), 'semanas'))
display(Markdown("(El coste unitario negativo de A, dato del enunciado, significa que reducirla ahorra dinero; C, con coste infinito, no puede reducirse.)"))
_cc2 = proyecto.critical_path(duration)
_sd2, _vr2 = proyecto.standard_deviation(durations=duration, variances=datos_proyecto['variance'])
display(Markdown(f"Tras la reducción las rutas críticas son " + lista_y([' – '.join(ordena_ruta(a, calendario)) for a in _cc2.values()])
                 + f"; la mayor varianza es {_sd2**2:g}, σ = {_sd2:.2f}, y " + explica_probabilidad(proyecto.duration(duration), _sd2, duracion_objetivo)
                 + f".\n\n**Respuesta:** la probabilidad de terminar en {duracion_objetivo} semanas es ahora del **50 %**, porque {duracion_objetivo} semanas es exactamente la nueva duración media."))

# %% [markdown]
# ### Probabilidad de terminar antes de 18 semanas
# Al ser 18 semanas la nueva duración media del proyecto, y ser la ley normal simétrica respecto de su media, la probabilidad de terminar en 18 semanas es del 50 %, cualquiera que sea la desviación típica (como confirma el cálculo anterior con la nueva desviación típica).

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# ### Análisis de la situación inicial

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
proyecto_nivelacion = proyecto.copy()
duration_resources = datos_enunciado[['duration','resources']].copy()
gantt_df, dibujo = proyecto_nivelacion.gantt(data=duration_resources,
                          duration_label='duration',
                          resource_label='resources',
                          holguras=True,
                          total='fila', 
                          tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su planificación inicial"
#| label: fig-preNivelaciónBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None)
_c3 = _cargas(gantt_df)
display(_c3)
display(Markdown('Suma de cuadrados de las cargas: ' + ' + '.join(f"{c}²" for c in _c3.loc['Carga'])
                 + f" = **{int((_c3.loc['Carga'] ** 2).sum())}**."))
# %% [markdown]
# Para saber si la demanda está nivelada se aplica el algoritmo de nivelación por mínimos cuadrados: si algún desplazamiento de las actividades no críticas dentro de su holgura reduce la suma de cuadrados de las cargas, la demanda inicial no estaba nivelada.

# %% [markdown]
# ### Nivelación de los recursos

# %%
proyecto_nivelacion.display_nivelar_pasos_tikz(datos_proyecto, duration_label='duration', resource_label='resources')
desplazamientos, cuadrados, my_data, gantt_df, dibujo = proyecto_nivelacion.nivelar(data=datos_proyecto,
                           duration_label='duration',
                           resource_label='resources',
                           report=False
                           )

# %% [markdown]
# Se procede a valorar los posibles desplazamientos de cada actividad no crítica, considerando estas en el orden correspondiente a su fecha de finalización, de más pronto a más tarde (a igualdad, primero la de mayor holgura). El visor anterior muestra el valor de la suma de cuadrados de las cargas para cada actividad, en filas según el mencionado orden, y posible desplazamiento, en columnas; el desplazamiento que minimiza la suma de cuadrados se señala en verde y queda fijado antes de estudiar la siguiente actividad.

# %%
#| include: false
cuadrados   # la tabla de sumas de cuadrados ya se muestra en el visor anterior

# %% [markdown]
# El resultado de estos desplazamientos se muestra en la siguiente figura.

# %%
#| fig-cap: "Resultado de aplicar el algoritmo de nivelación al proyecto"
#| label: fig-GanttTrasNivelacion
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras aplicar el algoritmo de nivelación"
#| label: fig-postNivelaciónBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None)
display(_cargas(gantt_df))
_respuesta_nivelacion(cuadrados, desplazamientos)
# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
proyecto_asignacion = proyecto.copy()
duration_resources = datos_enunciado[['duration','resources']].copy()
gantt_df, dibujo = proyecto_asignacion.gantt(duration_resources, 'duration', 'resources', total='fila', holguras=True, tikz=True)
pinta_tikz(dibujo)

# %% [markdown]
# La fila "Total" obtenida, que podemos representar gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo y el valor límite establecido (línea horizontal discontinua).

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)
_c4 = _cargas(gantt_df, limite_maximo_recursos)
display(_c4)
_exc = [p for p in _c4.columns if _c4.loc['¿Supera?', p] == 'sí']
# %%
Markdown((f"Los recursos **no** están correctamente asignados: el proyecto requiere más de {limite_maximo_recursos} personas en {plural(len(_exc), 'semana')} ("
          + lista_y([f"semana {p}: {_c4.loc['Carga', p]}" for p in _exc]) + "). Es necesario aplicar el algoritmo de asignación, según se muestra en la siguiente figura.")
         if _exc else f"La carga no supera {limite_maximo_recursos} personas en ninguna semana: los recursos están bien asignados.")

# %%
duracion_y_recursos = datos_proyecto[['duration', 'resources']]
proyecto_asignacion.display_asignar_pasos_tikz(data=duracion_y_recursos, duration_label='duration', resource_label='resources', maximo=[limite_maximo_recursos])
my_data, gantt_df, dibujo  = proyecto_asignacion.asignar(
                                    data=duracion_y_recursos,
                                    duration_label='duration',
                                    resource_label='resources',
                                    maximo=[limite_maximo_recursos],
                                    report=False
                                    )

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la asignación"
#| label: fig-ganttPostAsignación
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)
_c5 = _cargas(gantt_df, limite_maximo_recursos)
display(_c5)
_desp = {a[1:]: int(d) for a, d in my_data['duration'].items() if str(a).startswith('💤')}
_T5 = len(_c5.columns)
display(Markdown(
    "**Respuesta:** para no superar las " + str(limite_maximo_recursos) + " personas hay que retrasar "
    + lista_y([f"{a} {plural(d, 'semana')}" for a, d in _desp.items()])
    + f" respecto de su inicio más temprano (decisiones semana a semana en el visor del algoritmo de asignación). La carga máxima pasa a ser "
    + f"{_c5.loc['Carga'].max()} ≤ {limite_maximo_recursos} y el proyecto pasa de {duracion_media_proyecto} a **{plural(_T5, 'semana')}**."))
# %% [markdown]
# Tras aplicar el algoritmo de asignación el proyecto es compatible con los límites establecidos.

# %%
