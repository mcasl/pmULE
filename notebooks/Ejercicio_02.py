# ---
# title: Ejercicio 02
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
import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np


# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, pinta_tikz, plot_cargas
from pmule import explica_camino_critico, explica_probabilidad, ordena_ruta, lista_y, plural

def _cargas(gantt, limites):
    """Carga de recursos por periodo frente al límite (lista que se prolonga con su último valor)."""
    _c = gantt.data.loc['Total'].drop('H_total').astype(float)
    _l = list(limites) if isinstance(limites, (list, tuple)) else [limites]
    _l = (_l + [_l[-1]] * len(_c))[:len(_c)]
    _t = pd.DataFrame({'Periodo': range(1, len(_c) + 1), 'Carga': _c.astype(int).values, 'Límite': _l})
    _t['¿Supera?'] = np.where(_t['Carga'] > _t['Límite'], 'sí', 'no')
    return _t.set_index('Periodo').T

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# # Enunciado
#
# Considere el proyecto cuya información se adjunta.

# %%

periodo_probabilidad_tau = 10
limite_maximo_recursos = [4]*3 + [6]
pregunta_1 = "1. Determine para cada actividad su duración optimista. ¿Influye esta duración optimista en la duración mínima del proyecto?  "
pregunta_2 = "2. Determine la duración media del proyecto y su varianza utilizando para ello la aproximación del teorema del límite central.  "
pregunta_3 = f"3. Determine la probabilidad de que el proyecto se termine antes de {periodo_probabilidad_tau} periodos.  "
pregunta_4 = "4. Planifique el proyecto teniendo en cuenta que durante las primeras tres semanas dispondrá de cuatro trabajadores y a partir de la cuarta de seis trabajadores.  "
pregunta_5 = "5. ¿Sería suficiente disponer de 4 recursos durante todo el proyecto? ¿Por qué?  "


# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))
display(Markdown(pregunta_5))

# %%
datos_enunciado = pd.DataFrame([
#  nombre, anteriores, duracion, duracion_pesimista, duracion_modal, recursos
     ('A',       '---  ',     2,         2,                    2,           1  ),
     ('B',       'A    ',     2,         3,                    2,           2  ),
     ('C',       'B    ',     3,         4,                    3,           2  ),
     ('D',       'B,J  ',     4,         5,                 17/4,           1  ),
     ('E',       'F,B,J',     1,         1,                    1,           1  ),
     ('F',       'A    ',     2,         3,                    2,           2  ),
     ('G',       'C,D  ',     1,         1,                    1,           1  ),
     ('H',       'E    ',     4,         5,                    4,           1  ),
     ('I',       'B,J  ',     1,         1,                    1,           1  ),
     ('J',       'A    ',     2,         2,                    2,           1  ),
] ,columns=['activity', 'predecessor', 'duration', 'pesimistic_duration', 'modal_duration', 'resources'])
datos_enunciado.set_index('activity', inplace=True)
datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1
#

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# En el método PERT la duración media $D$ de cada actividad se obtiene a partir de su duración optimista $a$, la más probable o modal $m$ y la pesimista $b$ mediante $D=\frac{a+4m+b}{6}$. Como el enunciado da $D$ (columna `duration`), $m$ y $b$, despejamos la duración optimista: $a=6D-b-4m$.

# %% [markdown]
# Sustituyendo los datos de cada actividad:

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto['optimistic_duration'] = datos_proyecto.eval('6*duration - pesimistic_duration - 4*modal_duration')
_t1 = pd.DataFrame({
    'D (media)': datos_proyecto['duration'],
    'b (pesimista)': datos_proyecto['pesimistic_duration'],
    'm (modal)': datos_proyecto['modal_duration'],
    'a = 6D − b − 4m': [f"6·{d:g} − {b:g} − 4·{m:g} = {a:g}" for d, b, m, a in
                        datos_proyecto[['duration', 'pesimistic_duration', 'modal_duration', 'optimistic_duration']].values],
})
display(_t1)
display(Markdown(
    '**Respuesta:** las duraciones optimistas son '
    + ', '.join(f"{k} = {v:g}" for k, v in datos_proyecto['optimistic_duration'].items()) + '.\n\n'
    'La duración optimista **no influye** en la duración mínima del proyecto, entendida como la duración media que se obtiene '
    'al calcular el proyecto con las duraciones tope de las actividades. La duración media (o PERT) de una actividad y su '
    'duración tope son valores medios, y para cada uno de ellos cabe a su vez hablar de valores optimistas y pesimistas. '
    'No debe confundirse el cambio de duración de una actividad provocado conscientemente (cambiando el nivel de esfuerzo, '
    'la calidad de las herramientas, los equipos o las técnicas) con la observación de que, en un caso concreto, la duración '
    'resulte ser aleatoriamente mayor o menor: la duración optimista describe esta variabilidad aleatoria (interviene en la '
    'media y en la varianza), no la reducción por aceleración.'))

# %% [markdown]
# ## Apartado 2
#

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# ### Cuadro de prelaciones

# %% [markdown]
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto = prelaciones.create_project()
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
duraciones_medias = datos_proyecto['duration']
resultados_pert = proyecto.calculate_pert(durations=duraciones_medias)
resultados_pert['nodes']

# %%
duracion_media_proyecto = int(proyecto.duration(durations=duraciones_medias))

# %%
Markdown(f"El tiempo más temprano del nodo final del proyecto (nodo {resultados_pert['nodes'].index[-1]}) es su duración media: "
         f"**{duracion_media_proyecto} periodos**.")

# %% [markdown]
# ### Calendario del proyecto
# Las fechas de inicio y fin más tempranas y tardías de cada actividad son:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ### Otras representaciones alternativas
# Podemos igualmente representar el cronograma del proyecto bien con un diagrama de Gantt, o un grafo Roy.

# %%
#| fig-cap: "Diagrama de Gantt del proyecto"
#| label: fig-gantt
gantt_df, dibujo = proyecto.gantt(datos_proyecto,
                                  duration_label='duration',
                                  tikz=True,
                                  params=dict(
                                     critical_background_color='white!90!green',
                                     regular_background_color='white!90!green',
                                     period_width=1.5,
                                 )
                            )
pinta_tikz(dibujo)

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-Roy
proyecto.display_roy_construccion(tabla=prelaciones)

# %% [markdown]
# ### Holgura total de las actividaes
# Para determinar la varianza de la duración del proyecto es necesario determinar la varianza de cada rama del camino crítico. Para ello nos hará falta calcular la holgura total de cada actividad. Aquellas actividades con holgura total cero formarán el camino crítico. 

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
display(resultados_pert['activities'][['H_total']].rename(columns={'H_total': 'Holgura total'}))
_criticas = [a for a in ordena_ruta(list(resultados_pert['activities'].index), calendario)
             if resultados_pert['activities'].loc[a, 'H_total'] == 0]
_no_criticas = [a for a in datos_proyecto.index if a not in _criticas]
display(Markdown(f"(Las filas que empiezan por @ son actividades ficticias.) Actividades reales con holgura total nula: "
                 f"**{lista_y(_criticas)}**; con holgura: " + lista_y([f"{a} ({resultados_pert['activities'].loc[a, 'H_total']:g})" for a in _no_criticas]) + "."))

# %% [markdown]
# Las actividades con holgura total igual a cero forman el camino crítico: un retraso en cualquiera de ellas retrasa el proyecto.

# %% [markdown]
# ### Diagrama de red con indicación del camino crítico

# %% [markdown]
# El proyecto dado puede ser representado utilizando distintas técnicas. A continuación se muestra el grafo Pert del mismo.

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación del camino crítico"
#| label: fig-pertrutasCriticas
proyecto.display_pert_pasos(duraciones_medias)

# %% [markdown]
# Las rutas del camino crítico son en este caso:

# %%
#| tbl-cap: "Rutas del camino crítico"
#| label: tbl-rutasCriticas
camino_critico = proyecto.critical_path(durations=duraciones_medias)
for key, value in camino_critico.items():
    pass
explica_camino_critico(proyecto, duraciones_medias, 'periodos')

# %% [markdown]
# Alternativamente, podríamos haber representado el proyecto a traves de su grafo Roy. Se muestra a continuación con indicación de las actividades críticas:

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación de las actividades críticas"
#| label: fig-RoyCríticas
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ### Varianza del proyecto

# %% [markdown]
# Para determinar la varianza del proyecto es necesario conocer la varianza de cada una de las rutas del camino crítico, y para ello la varianza de las actividades que las componen.
# La varianza de cada actividad no es aportada por el enunciado como dato, pero sí se puede calcular a partir de la expresión $\sigma^2 = \left(\frac{b-a}{6} \right)^2$ (redondeada a dos decimales):

# %%
#| tbl-cap: "Cálculo de la varianza de la duración de las actividades"
#| label: tbl-varianzas
datos_proyecto['variance'] = (((datos_proyecto['pesimistic_duration'] - datos_proyecto['optimistic_duration'])/6)**2).round(2)
pd.DataFrame({
    'b (pesimista)': datos_proyecto['pesimistic_duration'],
    'a (optimista)': datos_proyecto['optimistic_duration'],
    'σ² = ((b − a)/6)²': [f"(({b:g} − {a:g})/6)² = {v:.2f}" for b, a, v in
                          datos_proyecto[['pesimistic_duration', 'optimistic_duration', 'variance']].values],
})

# %% [markdown]
# Conocidos estos valores podemos calcular, mediante el teorema del límite central, la varianza de cada ruta como suma de las varianzas de sus actividades.

# %%
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations = datos_proyecto['duration'],
                                                  variances = datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
_vr = {}
for _r, _acts in camino_critico.items():
    _acts = ordena_ruta(_acts, calendario)
    _vr[_r] = {'Ruta': ' – '.join(_acts),
               'Varianza = suma': ' + '.join(f"σ²({a})" for a in _acts) + ' = '
                                  + ' + '.join(f"{datos_proyecto.loc[a, 'variance']:g}" for a in _acts)
                                  + f" = {sum(datos_proyecto.loc[a, 'variance'] for a in _acts):.2f}"}
pd.DataFrame(_vr).T

# %%
_vmax = varianza_caminos.data['Variance'].astype(float)
_rmax = [r for r in _vmax.index if np.isclose(_vmax[r], _vmax.max())]
display(Markdown(f"""Cuando hay varias rutas críticas se toma, por prudencia, la de **mayor varianza**: 
σ²(proyecto) = máx({", ".join(f"{v:.2f}" for v in _vmax)}) = **{desviacion_proyecto**2:.2f}** ({lista_y(_rmax)}).  
Su raíz cuadrada, la desviación típica, es el parámetro que se usa en la ley normal: σ = √{desviacion_proyecto**2:.2f} = **{desviacion_proyecto:.2f}**.

**Respuesta:** por el teorema del límite central la duración del proyecto se aproxima por una normal de media **{duracion_media_proyecto} periodos** 
(la duración del camino crítico) y varianza **{desviacion_proyecto**2:.2f}** (desviación típica {desviacion_proyecto:.2f} periodos)."""))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_tau = norm.cdf(periodo_probabilidad_tau, loc=duracion_media_proyecto, scale=desviacion_proyecto)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, hasta=periodo_probabilidad_tau)
display(Markdown(explica_probabilidad(duracion_media_proyecto, desviacion_proyecto, periodo_probabilidad_tau)
                 + f"\n\n**Respuesta:** la probabilidad de terminar el proyecto antes de {periodo_probabilidad_tau} periodos es "
                   f"**{probabilidad_de_terminar_antes_de_tau:.4f}** ({probabilidad_de_terminar_antes_de_tau * 100:.2f} %)."))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# ### Análisis de la situación inicial

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
datos_proyecto[['duration',
                'pesimistic_duration',
                'resources',
                'optimistic_duration']] = datos_proyecto[['duration', 'pesimistic_duration', 
                                                            'resources', 'optimistic_duration']].astype(int)
proyecto_asignacion = proyecto.copy()
gantt_df, dibujo = proyecto.gantt(datos_proyecto,
                      duration_label='duration',
                      resource_label='resources',
                      holguras=True,
                      total='fila',
                      tikz=True,
                      params=dict(
                        period_width=1.5,
                        row_height=1,
                        #background_horizontal_line_color='gray',
                        #background_vertical_bars_color='white!92!cyan',
                        activity_relative_height=0.6,
                        critical_background_color='white!90!red',
                        regular_background_color='white!90!green',
                        regular_text_color='black',
                        critical_text_color='black',
                        activity_inner_text_style=r"\bfseries\large",),
                    )
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)
_c0 = _cargas(gantt_df, limite_maximo_recursos)
_exc = [p for p in _c0.columns if _c0.loc['¿Supera?', p] == 'sí']
display(Markdown('Carga de recursos de cada periodo con la programación más temprana, frente al límite disponible:'))
display(_c0)
display(Markdown((f"Se supera el límite en {plural(len(_exc), 'periodo')}: "
                  + lista_y([f"periodo {p} (carga {_c0.loc['Carga', p]} > {_c0.loc['Límite', p]})" for p in _exc]) + '.')
                 if _exc else 'La carga no supera el límite en ningún periodo.'))

# %% [markdown]
# Como la demanda de recursos infringe el límite impuesto en el enunciado, es necesario aplicar el algoritmo de asignación. Representamos a continuación la nueva planificación atendiendo a los cambios introducidos de acuerdo al algoritmo de asignación.

# %% [markdown]
# ### Algoritmo de asignación de Wiest-Levy

# %%
#| fig-cap: "Planificación del proyecto tras aplicar el algoritmo de asignación"
#| label: fig-asignacion46
proyecto_asignacion = proyecto.copy()
proyecto_asignacion.display_asignar_pasos_tikz(data=datos_proyecto, duration_label='duration', resource_label='resources', maximo=[4,4,4,6])
datos_proyecto_tras_asignacion, gantt_df, dibujo = proyecto_asignacion.asignar(data=datos_proyecto,
                                              duration_label='duration',
                                              resource_label='resources',
                                              maximo = [4,4,4,6],
                                              report = False,
                                              params = dict(
                                                  period_width=1.5
                                                  )
                                              )
pinta_tikz(dibujo)
_T4 = int(proyecto_asignacion.duration(datos_proyecto_tras_asignacion['duration']))
_desp4 = {a[1:]: int(d) for a, d in datos_proyecto_tras_asignacion['duration'].items() if str(a).startswith('💤')}

# %% [markdown]
# La zona sombreada indica el desplazamiento realizado en las actividades. De acuerdo a los cambios introducidos el proyecto ya es compatible con las restricciones impuestas a los recursos.

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos)
_c4 = _cargas(gantt_df, limite_maximo_recursos)
display(_c4)
display(Markdown(
    f"**Respuesta:** el algoritmo de asignación retrasa " + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _desp4.items()])
    + f" respecto de su inicio más temprano; con esa planificación la carga máxima es {_c4.loc['Carga'].max()} y "
    + ("no se supera el límite en ningún periodo" if (_c4.loc['¿Supera?'] == 'no').all() else "aún se supera el límite")
    + f", y el proyecto pasa de {duracion_media_proyecto} a **{plural(_T4, 'periodo')}**"
    + (f" ({plural(_T4 - duracion_media_proyecto, 'periodo')} más)." if _T4 > duracion_media_proyecto else ".")))

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# Se analiza ahora la posibilidad de reprogramar el proyecto atendiendo al criterio de que el consumo de recursos no supere el  límite de 4 recursos a lo largo de todo el proyecto impuesto por el enunciado del ejercicio. El siguiente diagrama de Gantt muestra la nueva planificación obtenida al aplicar el algoritmo de asignación con el límite impuesto.

# %%
#| fig-cap: "Planificación del proyecto tras aplicar el algoritmo de asignación con el nuevo límite"
#| label: fig-asignacion444
proyecto_asignacion = proyecto.copy()
datos_proyecto_tras_asignar, gantt_df, dibujo = proyecto_asignacion.asignar(data=datos_proyecto,
                                              duration_label='duration',
                                              resource_label='resources',
                                              maximo = [4,4,4],
                                              report = False,
                                              params = dict(
                                                  period_width=1.5
                                                  )
                                              )
pinta_tikz(dibujo)

# %% [markdown]
# El diagrama de cargas de esta nueva planificación es:

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras la nueva configuración"
#| label: fig-postAsignacionBarras2
plot_cargas(gantt_df, 4)
_T5 = int(proyecto_asignacion.duration(datos_proyecto_tras_asignar['duration']))
_desp5 = {a[1:]: int(d) for a, d in datos_proyecto_tras_asignar['duration'].items() if str(a).startswith('💤')}
_c5 = _cargas(gantt_df, 4)
_rmax = int(datos_proyecto['resources'].max())
display(Markdown(
    f"**Respuesta:** sí, 4 recursos serían suficientes para ejecutar el proyecto, porque ninguna actividad necesita por sí sola más de "
    f"{_rmax} recursos ({_rmax} ≤ 4), de modo que el algoritmo siempre puede encontrar una programación que respete el límite: "
    f"retrasando " + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _desp5.items()])
    + f" la carga máxima es {_c5.loc['Carga'].max()} ≤ 4. El precio es el plazo: el proyecto se prolonga hasta **{plural(_T5, 'periodo')}**, "
    f"frente a {duracion_media_proyecto} sin restricción de recursos y {_T4} con los recursos del apartado 4. "
    f"Si el plazo de {duracion_media_proyecto} periodos fuese obligado, 4 recursos no bastarían."))

# %%

# %%
