# ---
# title: Ejercicio 03
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
#     warning: false
# message: false
# jupyter: python3
# ---

# %%
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, colorea_minimo_fila, plot_cargas
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
# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# ## Enunciado
# Considere la información del proyecto cuya información aparece en la tabla adjunta:

# %%
periodo_probabilidad_b = 19
periodo_probabilidad_a = 17
limite_maximo_recursos = [4]*3 + [8]
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto entre {periodo_probabilidad_a} y {periodo_probabilidad_b} semanas?  "
pregunta_2 = "2. Como la probabilidad antes calculada le parece baja, reduzca, con mínimo coste la duración media del proyecto a 18 semanas. ¿Cuál será ahora la probabilidad de terminar en 18 semanas? ¿Por qué?  "
pregunta_3 = "3. Partiendo del proyecto en su situación inicial del apartado 1 ¿Se encuentran nivelados los recursos consumidos? De no ser así proceda a su nivelación.  "
pregunta_4 = "4. Partiendo del proyecto en su situación inicial del apartado 1, modifique la planificación del proyecto para poder ejecutarlo con tan sólo 4 recursos como máximo durante los primeros tres periodos, y 8 recursos durante el resto del proyecto.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
# Alargamos el ancho de las celdas
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, varianza,  cur,  recursos, duracion_tope
     ('A',       '---  ',      2,        1,     -100,     2,         1        ),
     ('B',       '---  ',      4,        1,       30,     2,         2        ),
     ('C',       'B    ',      1,        1,   np.inf,     2,         1        ),
     ('D',       'A    ',      3,        1,       50,     2,         1        ),
     ('E',       'B,D  ',      7,        1,       20,     2,         4        ),
     ('F',       'B,C,D ',     4,        1,       10,     1,         2        ),
     ('G',       '---  ',      1,        2,      100,     4,         1        ),
     ('H',       'G    ',     11,        2,       10,     2,         8        ),
     ('I',       'E,F,H',      2,        1,       50,     2,         1        ),
     ('J',       'E,F,H',      8,        1,        5,     2,         7        ),
] ,columns=['activity', 'predecessor', 'duration', 'variance', 'ucr', 'resources', 'cap_duration']).set_index('activity')

datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')

# %% [markdown]
# ### Cuadros de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos
#
# Utilizando el cuadro de prelaciones expandido podemos dibujar el grafo Pert correspondiente al proyecto.

# %%
#| fig-cap: "Grafo Pert con indicación con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto = prelaciones.create_project()
proyecto.display_pert_construccion(tabla=prelaciones)

# %% [markdown]
# Alternativamente podríamos haber representado su grafo Roy.

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-roy
proyecto.display_roy_construccion(tabla=prelaciones)

# %% [markdown]
# ### Matriz de Zaderenko
# A continuación podemos determinar la duración del proyecto calculando los tiempos tempranos y tardíos de su grafo Pert. Se hace en este caso mediante el algoritmo de Zaderenko:

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duration'])

# %% [markdown]
# Los tiempos tempranos y tardíos obtenidos para cada nodo son:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Duración media del proyecto

# %%
duracion_media_proyecto = proyecto.duration(durations=datos_proyecto['duration'])
Markdown(f"De acuerdo a la duración obtenida para el nodo final del proyecto, la duración media del proyecto es {duracion_media_proyecto} periodos.")

# %% [markdown]
# ### Calendario del proyecto
# Las fechas de inicio y fin más tempranas y tardías de cada actividad son:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ### Varianza de la duración del proyecto
# Para determinar la varianza del proyecto es necesario conocer la varianza de cada una de las rutas del camino crítico. Es necesario, por tanto, conocer cuales son las actividades con holgura total cero.

# %% [markdown]
# #### Holgura total de las actividades
# Del cálculo de las holguras totales obtenemos:

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
display(resultados_pert['activities'].rename(columns={'H_total': 'Holgura total'}))
display(Markdown('(Las filas que empiezan por @ son actividades ficticias.) Actividades con holgura total nula: **'
                 + lista_y([a for a in ordena_ruta(datos_proyecto.index, calendario) if resultados_pert['activities'].loc[a, 'H_total'] == 0]) + '**.'))
# %% [markdown]
# #### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas (actividades en orden de ejecución):
# %%
camino_critico = display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'semanas'))
# %% [markdown]
# #### Grafo PERT con indicación del camino crítico

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Alternativamente, se muestra el grafo roy correspondiente con indicación del camino crítico.

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# #### Varianza de cada rama
#
# Se calcula entonces la varianza de cada una de las rutas del camino crítico como suma de las varianzas de las actividades que lo componen.

# %%
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
_cc1 = proyecto.critical_path(datos_proyecto['duration'])
_tabla_varianzas(_cc1, datos_proyecto['variance'], calendario)
# %%
_v = varianza_caminos.data['Variance'].astype(float)
display(Markdown(f"""La varianza de la duración del proyecto es, por prudencia, la máxima varianza de las rutas críticas:
σ² = máx({", ".join(f"{x:g}" for x in _v)}) = **{desviacion_proyecto**2:g}**.
Su raíz cuadrada, la desviación típica, es el parámetro de la ley normal: σ = √{desviacion_proyecto**2:g} = **{desviacion_proyecto:.2f}** semanas."""))
# %% [markdown]
# #### Probabilidad de terminar en el periodo solicitado

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)
probabilidad_de_terminar_antes_de_a = norm.cdf(periodo_probabilidad_a, loc=duracion_media_proyecto, scale=desviacion_proyecto)


probabilidad_de_terminar_entre_a_y_b = round(float(probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a), 2)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, entre=(periodo_probabilidad_a, periodo_probabilidad_b))
display(Markdown(explica_intervalo(duracion_media_proyecto, desviacion_proyecto, periodo_probabilidad_a, periodo_probabilidad_b)
                 + f"\n\n**Respuesta:** con duración media {duracion_media_proyecto:g} semanas y desviación típica {desviacion_proyecto:.2f}, "
                   f"la probabilidad de terminar el proyecto entre {periodo_probabilidad_a} y {periodo_probabilidad_b} semanas es "
                   f"**{probabilidad_de_terminar_entre_a_y_b:.2f}** (≈ {probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a:.1%})."))
# %% [markdown]
# ## Apartado 2
#

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# ### Reducción de la duración del proyecto con mínimo coste

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                min_durations=datos_proyecto['cap_duration'],
                costs=datos_proyecto['ucr'],
                reduction=2,
                report=False,
                )

# %%
#| tbl-cap: "Cálculo de la reducción de la duración del proyecto mediante el algoritmo de Ackoff Sasieni"
#| label: tbl-ackoff
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %% [markdown]
# Las reducciones indicadas en cada iteración del algoritmo de reducción con mínimo coste, y su coste, son:

# %%
for iteracion, actividades in best_option.items():
    pass
_costes_it = {it: sum(datos_proyecto.loc[a, 'ucr'] for a in acts) for it, acts in best_option.items()}
display(Markdown(
    'Cada iteración acorta un periodo la combinación de actividades de menor coste unitario de reducción que reduce '
    'todas las rutas críticas:\n\n'
    + '\n'.join(f"- Iteración {it}: " + ' + '.join(f"{a}({datos_proyecto.loc[a, 'ucr']:g})" for a in acts)
                 + f" = {_costes_it[it]:g}" for it, acts in best_option.items())
    + f"\n\n**Respuesta (reducción):** se reduce un periodo " + lista_y(sorted({a for acts in best_option.values() for a in acts}))
    + f"; la duración media pasa de {proyecto.duration(datos_proyecto['duration']):g} a {proyecto.duration(duration):g} semanas, con un coste de "
    + ' + '.join(f"({c:g})" for c in _costes_it.values()) + f" = **{sum(_costes_it.values()):g}** unidades monetarias "
    + "(el coste unitario negativo de A, dato del enunciado, significa que reducirla ahorra dinero)."))
# %% [markdown]
# ### Probabilidad de terminar antes de 18 semanas
# Al ser 18 semanas la nueva duración media, y ser la ley normal simétrica respecto de su media, la probabilidad de terminar en 18 semanas es del 50 %, sea cual sea la desviación típica (se comprueba numéricamente más abajo, una vez calculada la nueva desviación típica).
# %% [markdown]
# ### Probabilidad de terminar entre 17 y 19 semanas
# %% [markdown]
# #### Duración media del proyecto

# %%
duracion_media_proyecto = proyecto.duration(duration)

# %%
Markdown(f"Nótese como al haber reducido la duración del proyecto, la duración media a utilizar en el cálculo de la probabilidad es la nueva duración recortada de {duracion_media_proyecto} periodos.")

# %% [markdown]
# #### Varianza de la duración del proyecto

# %% [markdown]
# Al reducir la duración pueden aparecer nuevas rutas críticas. Debemos calcular su varianza y actualizar la varianza del proyecto si alguna de ellas tuviera una varianza mayor.
# %%
# duration ha sido calculada en el Ackoff Sasieni
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=duration,
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas2
_cc2 = proyecto.critical_path(duration)
_nuevas = [r for r in _cc2 if r not in _cc1]
display(Markdown(('Nuevas rutas críticas tras la reducción: ' + lista_y([f"{r} ({' – '.join(ordena_ruta(_cc2[r], calendario))})" for r in _nuevas]) + '.')
                 if _nuevas else 'No aparecen nuevas rutas críticas.'))
_tabla_varianzas(_cc2, datos_proyecto['variance'], calendario)
# %%
_v = varianza_caminos.data['Variance'].astype(float)
display(Markdown(f"""La varianza de la duración del proyecto es la máxima varianza de las rutas críticas:
σ² = máx({", ".join(f"{x:g}" for x in _v)}) = **{desviacion_proyecto**2:g}**, y la desviación típica σ = √{desviacion_proyecto**2:g} = **{desviacion_proyecto:.2f}** semanas."""))
# %% [markdown]
# #### Probabilidad de terminar en el periodo solicitado

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)
probabilidad_de_terminar_antes_de_a = norm.cdf(periodo_probabilidad_a, loc=duracion_media_proyecto, scale=desviacion_proyecto)

probabilidad_de_terminar_entre_a_y_b = probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, entre=(periodo_probabilidad_a, periodo_probabilidad_b))
_T18 = 18
display(Markdown(explica_probabilidad(duracion_media_proyecto, desviacion_proyecto, _T18) + "\n\n"
                 + explica_intervalo(duracion_media_proyecto, desviacion_proyecto, periodo_probabilidad_a, periodo_probabilidad_b)
                 + f"\n\n**Respuesta:** tras la reducción, la probabilidad de terminar en {_T18} semanas es "
                   f"**{norm.cdf(_T18, duracion_media_proyecto, desviacion_proyecto):.0%}**, porque {_T18} semanas es exactamente la duración media "
                   f"y la normal es simétrica. La probabilidad de terminar entre {periodo_probabilidad_a} y {periodo_probabilidad_b} semanas sube a "
                   f"**{probabilidad_de_terminar_entre_a_y_b:.2%}** porque el intervalo queda ahora centrado en la media."))
# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# ### Análisis de la situación inicial

# %% [markdown]
# Para responder este apartado es necesario representar la demanda inicial de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto antes de la nivelación"
#| label: fig-ganttPrenivelacion
proyecto_nivelacion = proyecto.copy()
datos_proyecto=datos_enunciado.copy()
gantt_df, dibujo = proyecto_nivelacion.gantt_cargas(datos_proyecto, 'duration', 'resources', tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original"
#| label: fig-preNivelacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None)
_c3 = _cargas(gantt_df)
display(_c3)
display(Markdown('Suma de cuadrados de las cargas: ' + ' + '.join(f"{c}²" for c in _c3.loc['Carga'])
                 + f" = **{int((_c3.loc['Carga'] ** 2).sum())}**."))
# %% [markdown]
# Para saber si la demanda está nivelada se aplica el algoritmo de nivelación por mínimos cuadrados: si algún desplazamiento de las actividades no críticas dentro de su holgura reduce la suma de cuadrados de las cargas, la demanda inicial no estaba nivelada.
# %% [markdown]
# ### Nivelación

# %% [markdown]
# La siguiente tabla indica el valor de la suma de los cuadrados del consumo de los recursos para cada unos de los posibles desplazamientos, en el orden de desplazamiento de las actividades utilizado. El orden seguido ha sido el de las actividades no críticas en orden creciente de periodo de finalización. Se señala en color amarillo el desplazamiento que en cada caso minimiza la suma de cuadrados.

# %%
proyecto_nivelacion = proyecto.copy()
datos_proyecto = datos_enunciado.copy()

proyecto_nivelacion.display_nivelar_pasos_tikz(datos_proyecto, duration_label='duration', resource_label='resources')
desplazamientos, cuadrados, my_data, gantt_df, dibujo = proyecto_nivelacion.nivelar(
                            data=datos_proyecto,
                            duration_label='duration',
                            resource_label='resources',
                            
                           )

# %%
#| include: false
cuadrados   # la tabla de sumas de cuadrados ya se muestra en el visor anterior

# %% [markdown]
# Atendiendo a los valores de las sumas de cuadrados, los desplazamientos indicados para la nivelación son:

# %%
#| tbl-cap: "Desplazamientos para la nivelación"
#| label: tbl-desplazamientosNivelación
desplazamientos

# %% [markdown]
# El resultado de realizar estos desplazamientos se muestra en el siguiente diagrama de Gantt.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la nivelación"
#| label: fig-ganttPostnivelacion
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras realizar la nivelación"
#| label: fig-postNivelacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=None)
_sc = cuadrados.data if hasattr(cuadrados, 'data') else cuadrados
_sc0 = float(_sc.iloc[0, 0])
_scf = float(_sc.iloc[-1].min())
_mov = {a: int(v) for a, v in desplazamientos.iloc[:, 0].items() if v != 0}
display(Markdown(
    (f"**Respuesta:** los recursos **no** estaban nivelados: la suma de cuadrados inicial es {_sc0:g} y desplazando "
     + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _mov.items()])
     + f" baja a **{_scf:g}**. Las demás actividades no críticas no se desplazan por sí mismas (solo lo que las arrastren sus predecesoras) porque ningún desplazamiento reduce más la suma de cuadrados; "
     "como solo se mueven actividades dentro de su holgura, la duración del proyecto no cambia.")
    if _mov else f"**Respuesta:** los recursos ya estaban nivelados: ningún desplazamiento reduce la suma de cuadrados ({_sc0:g})."))
# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# ### Análisis de la situación de partida

# %% [markdown]
# Recuperamos el estado inicial del proyecto, como se ve en el siguiente diagrama de Gantt.

# %%
#| fig-cap: "Diagrama de Gantt previo a la asignación"
#| label: fig-ganttPreAsignacion

proyecto_asignacion = proyecto.copy()
datos_proyecto = datos_enunciado.copy()
gantt_df, dibujo = proyecto_nivelacion.gantt(datos_proyecto, 'duration', 'resources', total='fila', holguras=True, tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original, con indicación del límite máximo de recursos"
#| label: fig-preNivelacionBarras2
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=limite_maximo_recursos)
_c4 = _cargas(gantt_df, limite_maximo_recursos)
display(_c4)
_exc = [p for p in _c4.columns if _c4.loc['¿Supera?', p] == 'sí']
display(Markdown((f"Con la programación más temprana se supera el límite en {plural(len(_exc), 'periodo')}: "
                  + lista_y([f"periodo {p} (carga {_c4.loc['Carga', p]} > {_c4.loc['Límite', p]})" for p in _exc])
                  + "; hay que aplicar el algoritmo de asignación.") if _exc else "La carga no supera el límite en ningún periodo."))
# %% [markdown]
# ### Algoritmo de asignación de Wiest-Levy

# %% [markdown]
# A continuación se ejecuta el algoritmo de asignación.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la asignación"
#| label: fig-ganttPreAsignación
proyecto_asignacion.display_asignar_pasos_tikz(data=datos_proyecto, duration_label='duration', resource_label='resources', maximo=limite_maximo_recursos)
datos_proyecto_tras_asignar, gantt_df, dibujo = proyecto_asignacion.asignar(data=datos_proyecto,
                                              duration_label='duration',
                                              resource_label='resources',
                                              maximo = limite_maximo_recursos,
                                              report = False,
                                              params = dict(
                                                  period_width=1
                                                  )
                                              )
pinta_tikz(dibujo)

# %% [markdown]
# Tras esta actuación el proyecto ya es compatible con la restricción impuesta en los recursos.

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras el algoritmo de asignación, con indicación del límite máximo de recursos"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=limite_maximo_recursos)
_c4b = _cargas(gantt_df, limite_maximo_recursos)
_Ta = int(proyecto_asignacion.duration(datos_proyecto_tras_asignar['duration']))
_desp = {a[1:]: int(d) for a, d in datos_proyecto_tras_asignar['duration'].items() if str(a).startswith('💤')}
display(Markdown(
    "**Respuesta:** el algoritmo de asignación retrasa " + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _desp.items()])
    + f" respecto de su inicio más temprano; con esta planificación la carga de los tres primeros periodos es "
    + lista_y([str(c) for c in _c4b.loc['Carga'].iloc[:3]]) + f" (≤ {limite_maximo_recursos[0]}) y la máxima posterior es "
    + f"{_c4b.loc['Carga'].iloc[3:].max()} (≤ {limite_maximo_recursos[-1]}). El precio es el plazo: el proyecto pasa de "
    + f"{proyecto.duration(datos_enunciado['duration']):g} a **{plural(_Ta, 'semana')}**."))
# %%

# %%
