# ---
# title: Ejercicio 25
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

from pmule import (PredecessorTable, ProjectGraph, pinta_tikz, plot_cargas, explica_camino_critico, explica_suma,
                   explica_sigma_ruta, explica_intervalo, explica_probabilidad, ordena_ruta, plural, lista_y)

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# # Enunciado
#
# Considere la información del proyecto cuya información aparece en la tabla adjunta:    

# %%
periodo_a_probabilidad = 12
periodo_b_probabilidad = 14
duracion_objetivo = 10
limite_maximo_recursos = [8]*7 + [16]
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} periodos?  "
pregunta_2 = f"2. Respecto a la situación original del enunciado, indique mediante el método de Ackoff-Sasieni cuales serían las actuaciones a llevar a cabo para reducir con mínimo sobrecoste la duración media del proyecto a {duracion_objetivo} periodos.  "
pregunta_3 = f"3. Respecto a la situación original del enunciado, indique cuales serían las actuaciones para poder ejecutar el proyecto con un límite en los recursos de {limite_maximo_recursos[0]} recursos por periodo durante los primeros siete periodos, y {limite_maximo_recursos[-1]} recursos por periodo posteriormente.  "
pregunta_4 = f"4. Respecto a la situación original del enunciado, calcule el valor del CPI y del SPI para el punto de control cuyos datos se proporcionan.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado
import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, duracion_tope, desviación, coste_planificado, coste_real, duracion_real, realizado, cur, recursos
     ('A',      'D      ',       2,         1,              0.2,         80,              80,          2,           100,   80,    4 ),
     ('B',      'A      ',       2,         1,              0.3,         70,              60,          2,            80,   10,    2 ),
     ('C',      'B      ',       3,         1,              0.4,         60,               0,          0,             0,   40,    1 ),
     ('D',      '---    ',       4,         3,              0.5,         40,              50,          5,           100,   20,    4 ),
     ('E',      'A      ',       1,         1,              0.1,         40,              30,          3,           100,   30,    3 ),
     ('F',      'D,E    ',       2,         1,              0.3,         10,              10,          3,            80,   40,    5 ),
     ('G',      '---    ',       4,         1,              0.1,         40,              50,          3,           100,   20,    4 ),
     ('H',      'I    ',       4,         2,              0.5,        100,             140,          4,            90,   50,    2 ),
     ('I',      'D      ',       2,         1,              0.1,         50,              70,          1,            50,   60,    3 ),
     ('J',      'F,H,K  ',       2,         1,              0.4,         50,               0,          0,             0,   70,    1 ),
     ('K',      'D,G    ',       4,         1,              0.2,         40,              30,          2,            60,   90,    2 ),
     ('L',      'H,K    ',       1,         1,              0.2,         30,               0,          0,             0,   10,    3 ),
] ,columns=['activity', 'predecessor', 'duration', 'cap_duration', 'standard_deviation', 'planned_value', 'actual_cost',
            'actual_duration', 'performed', 'ucr', 'resources']).set_index('activity')
tabla = datos_enunciado.copy()
tabla.columns = ["Pred.", 'Duration', 'Cap <br> duration', 'Standard <br> deviation', 'PV', 'AC',
            'Actual <br> duration', 'Performed', 'UCR', 'Resources']
display(tabla.style.set_properties(**{'text-align': 'center'}).format(precision=2))

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# # Solución
# ## Apartado 1
#

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadro de prelaciones expandido

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto=datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert"
#| label: fig-PertSinTiempos
proyecto = prelaciones.create_project()
proyecto.pert()

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ### Matriz de Zaderenko

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duration'])

# %% [markdown]
# Los tiempos tempranos y tardíos han resultado ser

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Holgura total de las actividades

# %% [markdown]
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valores de la holgura total de las actividades"
#| label: tbl-Holguras
resultados_pert['activities'][['H_total']]

# %% [markdown]
# ### Camino crítico
#
# El camino crítico consta de las siguientes rutas:

# %%
display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'periodos'))

# %% [markdown]
# ### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto, con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ### Caracterización de la duración del proyecto mediante una distribución Normal

# %% [markdown]
# A fin de calcular la probabilidad pedida, es necesario caracterizar la distribución Normal de la duración del proyecto. Esto consiste en determinar los valores de sus parámetros: media y desviación típica de la duración del proyecto.

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duration']))

# %%
Markdown(f"De los tiempos obtenidos en el método de Zaderenko, ya sabemos que la duración media del proyecto es {duracion_media_proyecto} periodos.")

# %% [markdown]
# ### Media de la duración del proyecto

# %%
_ruta_critica = ordena_ruta(list(proyecto.critical_path(datos_proyecto['duration']).values())[0], calendario)
display(Markdown("La media de la duración del proyecto es la suma de las duraciones medias de la ruta crítica: "
                 + explica_suma(datos_proyecto['duration'], 'μ', 'periodos', actividades=_ruta_critica) + "."))

# %% [markdown]
# ### Grafo PERT con indicación del camino crítico

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con tiempos, y rutas críticas"
#| label: fig-PertConTiempos
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# La desviación típica puede calcularse para cada ruta del camino crítico. En este caso sólo existe una ruta

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                   variances=datos_proyecto['variance']) 
desviacion_proyecto = round(desviacion_proyecto, 2)

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
varianza_caminos

# %%
Markdown(f"Por tanto, la desviación típica del proyecto resulta ser $\\sigma$ = {desviacion_proyecto}: "
         + explica_sigma_ruta(_ruta_critica, datos_proyecto['variance'], 2) + f" (se usa σ = {desviacion_proyecto}).")

# %% [markdown]
# ### Probabilidad de terminar en el intervalo especificado

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(norm.cdf(periodo_b_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto)*100, 1)
probabilidad_de_terminar_antes_de_a = round(norm.cdf(periodo_a_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto)*100, 1)

probabilidad_de_terminar_entre_a_y_b = probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a
probabilidad_de_terminar_entre_a_y_b = round(probabilidad_de_terminar_entre_a_y_b, 1)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, entre=(periodo_a_probabilidad, periodo_b_probabilidad))

# %%
Markdown(f"""Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, T ~ N(μ = {duracion_media_proyecto}; σ = {desviacion_proyecto}), podemos hacer el cálculo de la probabilidad de terminar entre los periodos {periodo_a_probabilidad} y {periodo_b_probabilidad}:

- {explica_probabilidad(duracion_media_proyecto, desviacion_proyecto, periodo_a_probabilidad)}
- {explica_probabilidad(duracion_media_proyecto, desviacion_proyecto, periodo_b_probabilidad)}

Por tanto, a partir de la diferencia de estos dos valores tenemos la probabilidad de terminar dentro del mencionado intervalo:
{explica_intervalo(duracion_media_proyecto, desviacion_proyecto, periodo_a_probabilidad, periodo_b_probabilidad)}.

**Respuesta:** la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} periodos es del {probabilidad_de_terminar_entre_a_y_b} %.
""")

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
datos_proyecto = datos_enunciado.copy()
proyecto = prelaciones.create_project()

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations     = datos_proyecto['duration'],
                                                           min_durations = datos_proyecto['cap_duration'],
                                                           costs         = datos_proyecto['ucr'],
                                                           reduction     = duracion_media_proyecto - duracion_objetivo
                                                          )

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos_tikz(
    durations     = datos_proyecto['duration'],
    min_durations = datos_proyecto['cap_duration'],
    costs         = datos_proyecto['ucr'],
    reduction     = duracion_media_proyecto - duracion_objetivo)
_lineas = []
_d = duracion_media_proyecto
for _k, _acts in best_option.items():
    _c = datos_proyecto.loc[_acts, 'ucr']
    _lineas.append(f"- Iteración {_k}: se acorta un periodo {lista_y(_acts)} (coste {' + '.join(format(x, 'g') for x in _c) + ' = ' if len(_c) > 1 else ''}"
                   f"{_c.sum():g}); la duración pasa de {_d} a {_d - 1}.")
    _d -= 1
_red = (datos_proyecto['duration'] - duration).loc[lambda x: x > 0]
_coste = sum(datos_proyecto.loc[a, 'ucr'].sum() for a in best_option.values())
display(Markdown(
    "\n".join(_lineas) + "\n\n"
    f"**Respuesta:** para pasar de {duracion_media_proyecto} a {proyecto.duration(duration):g} periodos hay que acortar "
    + lista_y([f"{a} {plural(n, 'periodo')} (de {datos_proyecto.loc[a, 'duration']} a {duration[a]})" for a, n in _red.items()])
    + f", con un sobrecoste mínimo de {' + '.join(format(datos_proyecto.loc[a, 'ucr'].sum(), 'g') for a in best_option.values())} = **{_coste:g}**."))

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
proyecto_asignacion = proyecto.copy()
gantt, dibujo = proyecto_asignacion.gantt(datos_proyecto,
                          'duration',
                          'resources',
                          total='fila',
                          holguras=True, 
                          tikz=True)

# %%
#| fig-cap: "Diagrama de Gantt del proyecto de acuerdo a su definición original."
#| label: fig-ganttPreAsignación
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original"
#| label: fig-preAsignacionBarras
plot_cargas(gantt, limite_maximo_recursos)
_carga = pd.to_numeric(gantt.data.loc['Total'].drop('H_total', errors='ignore'), errors='coerce').dropna()
_limite = lambda p: limite_maximo_recursos[min(int(p), len(limite_maximo_recursos)) - 1]
_exceso = [(p, c, _limite(p)) for p, c in _carga.items() if c > _limite(p)]
display(Markdown(
    "Con todas las actividades en su fecha más temprana, la carga supera el límite en "
    + lista_y([f"el periodo {p} ({c:g} > {l})" for p, c, l in _exceso]) + "." if _exceso else
    "La carga no supera el límite en ningún periodo."))

# %% [markdown]
# La demanda de recursos actual infringe el límite impuesto. Es necesario aplicar el algoritmo de asignación.

# %% [markdown]
# {{< pagebreak >}}

# %%
duration_resources = datos_proyecto[['duration','resources']].copy()
proyecto_asignacion.display_asignar_pasos_tikz(data=duration_resources, duration_label='duration', resource_label='resources', maximo=limite_maximo_recursos)
duration_resources, df, dibujo = proyecto_asignacion.asignar(data=duration_resources,
                                                            duration_label='duration',
                                                            resource_label='resources',
                                                            report=False,
                                                            maximo=limite_maximo_recursos,
                                                            )

# %%
#| fig-cap: "Diagrama de Gantt del proyecto tras la asignación"
#| label: fig-ganttPostAsignación
pinta_tikz(dibujo)

# %% [markdown]
# {{< pagebreak >}}

# %%
#| fig-cap: "Diagrama de cargas del proyecto tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(df, limite_maximo_recursos)
_dfa = df.data
_periodos = [c for c in _dfa.columns if str(c) != 'H_total']
_inicio_nuevo = {a: int(min(c for c in _periodos if pd.notna(pd.to_numeric(_dfa.loc[a, c], errors='coerce')))) - 1
                 for a in datos_proyecto.index}
_retrasos = {a: _inicio_nuevo[a] - int(calendario.loc[a, 'inicio_mas_temprano']) for a in datos_proyecto.index}
_retrasos = {a: r for a, r in _retrasos.items() if r > 0}
_carga2 = pd.to_numeric(_dfa.loc['Total', _periodos], errors='coerce').fillna(0)
_fin = max(int(c) for c in _periodos if _carga2[c] > 0)
display(Markdown(
    "**Respuesta:** hay que retrasar "
    + lista_y([f"{a} {plural(r, 'periodo')} (empieza en {_inicio_nuevo[a]} en lugar de {int(calendario.loc[a, 'inicio_mas_temprano'])}; "
               f"holgura total {int(calendario.loc[a, 'H_total'])})" for a, r in _retrasos.items()])
    + f". Con ello la carga máxima en los siete primeros periodos es {_carga2[_periodos[:7]].max():g} ≤ {limite_maximo_recursos[0]} "
    f"y después {_carga2[_periodos[7:]].max():g} ≤ {limite_maximo_recursos[-1]}, y el proyecto termina en el periodo {_fin} "
    + (f"(su duración no cambia, porque los retrasos caben en las holguras)." if _fin == duracion_media_proyecto
       else f"({plural(_fin - duracion_media_proyecto, 'periodo')} más que sin límite de recursos).")))

# %% [markdown]
# Tras aplicar el algoritmo de asignación el proyecto ya es compatible con los límites establecidos (véase la respuesta
# debajo del diagrama de cargas).

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
PV = int(datos_proyecto['planned_value'].sum())
EV = int(datos_proyecto.eval('planned_value * performed / 100').sum())
AC = int(datos_proyecto['actual_cost'].sum())
CPI = float(EV/AC)
SPI=float(EV/PV)

# %%
_cal_real = proyecto.calendar(datos_proyecto['actual_duration'])
_t_control = int(_cal_real['fin_mas_temprano'].max())
_dp = datos_proyecto
Markdown(f"""El punto de control es el fin del calendario real (mismas prelaciones con las duraciones reales): periodo {_t_control}.
Como el plan terminaba en el periodo {duracion_media_proyecto} ≤ {_t_control}, en esa fecha debía estar hecho todo el proyecto y el
valor planificado es el presupuesto completo.

- PV (valor planificado) = Σ coste planificado = {' + '.join(str(v) for v in _dp['planned_value'])} = **{PV}**
- EV (valor ganado) = Σ coste planificado × % realizado = {' + '.join(f"{v}×{p/100:g}" for v, p in zip(_dp['planned_value'], _dp['performed']) if p > 0)} = **{EV}**
- AC (coste real) = Σ coste real = {' + '.join(str(v) for v in _dp['actual_cost'] if v > 0)} = **{AC}**

Utilizando CPI=EV/AC y SPI=EV/PV, tenemos:

- CPI (índice de rendimiento del coste) = {EV} / {AC} = **{round(CPI,2)}** {'< 1: sobrecoste' if CPI < 1 else '≥ 1: sin sobrecoste'}
- SPI (índice de rendimiento del plazo) = {EV} / {PV} = **{round(SPI,2)}** {'< 1: retraso' if SPI < 1 else '≥ 1: sin retraso'}

**Respuesta:** CPI = {round(CPI,2)} y SPI = {round(SPI,2)}.""")

# %%

# %%
