# ---
# title: Ejercicio 11
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
from pmule import explica_camino_critico, explica_intervalo, ordena_ruta, lista_y, plural

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
# Considere la información del proyecto cuya información aparece en la tabla adjunta:

# %%
periodo_a_probabilidad = 7
periodo_b_probabilidad = 9
duracion_objetivo = 6
limite_maximo_recursos = [8]*4 + [16]
pregunta_1 = f"1. ¿Cuál es la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} periodos?  "
pregunta_2 = f"2. Indique cuales serían las actuaciones a llevar a cabo para reducir con mínimo sobrecoste la duración media del proyecto a {duracion_objetivo} periodos.  "
pregunta_3 = f"3. Indique cuales serían las actuaciones para poder ejecutar el proyecto con un límite en los recursos de {limite_maximo_recursos[0]} recursos por periodo durante los primeros cuatro periodos, y de {limite_maximo_recursos[-1]} recursos por periodo en los siguientes periodos "
pregunta_4 =  "4. Calcule el valor del CPI y del SPI para el punto de control de 8 periodos.  "

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
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, duracion_tope, desviación, coste_planificado, coste_real, duracion_real, realizado, cur, recursos
     ('A',      '---  ',       2,         1,              0.2,         60,              80,          2,           100,   80,    4 ),
     ('B',      'A    ',       2,         1,              0.3,         70,              60,          2,            80,   10,    2 ),
     ('C',      'B    ',       3,         1,              0.4,         50,               0,          0,             0,   40,    1 ),
     ('D',      '---  ',       4,         3,              0.5,         40,              50,          5,           100,   20,    4 ),
     ('E',      'A    ',       1,         1,              0.1,         30,              30,          3,           100,   30,    3 ),
     ('F',      'D,E  ',       2,         1,              0.3,         10,              10,          3,            80,   40,    5 ),
     ('G',      '---  ',       1,         1,              0.1,         40,              50,          3,           100,   20,    4 ),
     ('H',      'G    ',       4,         2,              0.5,        110,             140,          4,            90,   50,    2 ),
     ('I',      'D    ',       1,         1,              0.1,         90,              70,          1,            50,   60,    3 ),
     ('J',      'H,I,K',       2,         1,              0.4,         50,               0,          0,             0,   70,    1 ),
     ('K',      'D,G  ',       2,         1,              0.2,         40,              30,          2,            60,   90,    2 ),
     ('L',      'H,K  ',       2,         1,              0.2,         30,               0,          0,             0,   10,    3 ),
] ,columns=['actividad', 'precedentes', 'duracion', 'duracion_tope', 'desviacion', 'coste_planificado', 'coste_real',
            'duracion_real', 'realizado', 'cur', 'recursos']).set_index('actividad')
datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='actividad', predecessor='precedentes')

# %% [markdown]
# ### Cuadro de prelaciones
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
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
proyecto.display_zaderenko_tikz(datos_proyecto["duracion"])

# %% [markdown]
# Los tiempos tempranos y tardios han resultado ser:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto["duracion"])
resultados_pert['nodes']

# %% [markdown]
# ### Duración media del proyecto

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duracion']))
Markdown(f"El tiempo más temprano del nodo final del proyecto es su duración media: **{duracion_media_proyecto} periodos**.")

# %% [markdown]
# ### Holgura total de las actividades
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
display(resultados_pert['activities'].rename(columns={'H_total': 'Holgura total'}))
display(Markdown('(Las filas que empiezan por @ son actividades ficticias.) Actividades con holgura total nula: **'
                 + lista_y([a for a in ordena_ruta(datos_proyecto.index, proyecto.calendar(datos_proyecto['duracion'])) if resultados_pert['activities'].loc[a, 'H_total'] == 0]) + '**.'))
# %% [markdown]
# ### Camino crítico
#
# El camino crítico consta de las siguientes rutas (actividades en orden de ejecución):

# %%
camino_critico = proyecto.critical_path(durations=datos_proyecto['duracion'])
for key, value in camino_critico.items():
    pass
explica_camino_critico(proyecto, datos_proyecto['duracion'], 'periodos')
# %% [markdown]
# ### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto, con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duracion'])
proyecto.display_calendario(datos_proyecto['duracion'])

# %% [markdown]
# ### Grafo PERT con indicación de tiempos y rutas del camino crítico de acuerdo a la planificación inicial
# A continuación se muestra el grafo Pert del proyecto, con indicación de las actividades críticas:

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación del camino crítico"
#| label: fig-pertrutasCriticas
proyecto.display_pert_pasos(datos_proyecto['duracion'])

# %% [markdown]
# ### Caracterización de la duración del proyecto mediante una distribución Normal
# A fin de calcular la probabilidad pedida, es necesario caracterizar la distribución Normal de la duración del proyecto. Esto consiste en determinar los valores de sus parámetros: media y desviación típica de la duración del proyecto.

# %%
duracion_media_proyecto = int(proyecto.duration(datos_proyecto['duracion']))

Markdown(f"La duración media del proyecto ya la tenemos calculada: μ = **{duracion_media_proyecto} periodos**.")

# %% [markdown]
# Para determinar la desviación típica debemos obtener en cada rama del camino crítico la suma de las varianzas de sus actividades (cuadrado de la desviación típica del enunciado), y quedarnos con la mayor.

# %%
datos_proyecto['varianza'] = datos_proyecto.eval('desviacion ** 2')
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duracion'],
                                                   variances=datos_proyecto['varianza'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
display(Markdown('Varianzas de las actividades críticas: ' + ', '.join(f"σ²({a}) = {datos_proyecto.loc[a, 'desviacion']:g}² = {datos_proyecto.loc[a, 'varianza']:g}"
                 for a in ordena_ruta(sorted({x for r in camino_critico.values() for x in r}), calendario)) + '.'))
_tabla_varianzas(camino_critico, datos_proyecto['varianza'].round(4), calendario)
# %%
_v = varianza_rutas.data['Variance'].astype(float)
Markdown(f"Varianza del proyecto: σ² = máx({', '.join(f'{x:g}' for x in _v)}) = {desviacion_proyecto**2:.2f}; desviación típica σ = √{desviacion_proyecto**2:.2f} = {desviacion_proyecto:.2f}. "
         f"La duración del proyecto sigue aproximadamente una normal de media **{duracion_media_proyecto} periodos** y desviación típica **{desviacion_proyecto:.2f}**.")


# %% [markdown]
# #### Probabilidad de terminar entre 7 y 9 periodos

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = round(norm.cdf(periodo_b_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)
probabilidad_de_terminar_antes_de_a = round(norm.cdf(periodo_a_probabilidad, loc=duracion_media_proyecto, scale=desviacion_proyecto), 2)

probabilidad_de_terminar_entre_a_y_b = probabilidad_de_terminar_antes_de_b - probabilidad_de_terminar_antes_de_a
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, entre=(periodo_a_probabilidad, periodo_b_probabilidad))


# %%
Markdown(f"""Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, la probabilidad de terminar entre {periodo_a_probabilidad} y {periodo_b_probabilidad} periodos es la diferencia entre la de terminar antes de {periodo_b_probabilidad} y la de terminar antes de {periodo_a_probabilidad}:

{explica_intervalo(duracion_media_proyecto, desviacion_proyecto, periodo_a_probabilidad, periodo_b_probabilidad)}

**Respuesta:** la probabilidad de terminar el proyecto entre {periodo_a_probabilidad} y {periodo_b_probabilidad} periodos es **{norm.cdf(periodo_b_probabilidad, duracion_media_proyecto, desviacion_proyecto) - norm.cdf(periodo_a_probabilidad, duracion_media_proyecto, desviacion_proyecto):.2f}** (≈ {int(round(probabilidad_de_terminar_entre_a_y_b*100))} %).
""")

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
ackoff, best_option, duration, available =proyecto.ackoff(durations=datos_proyecto['duracion'],
                                                          min_durations=datos_proyecto['duracion_tope'],
                                                          costs=datos_proyecto['cur'],
                                                          reduction=duracion_media_proyecto - duracion_objetivo)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duracion'],
    min_durations=datos_proyecto['duracion_tope'],
    costs=datos_proyecto['cur'],
    reduction=duracion_media_proyecto - duracion_objetivo)

# %% [markdown]
# Las actividades a recortar en cada iteración, y su coste, son:

# %%
for iteracion, actividades in best_option.items():
    pass
_respuesta_ackoff(best_option, datos_proyecto['cur'], proyecto.duration(datos_proyecto['duracion']), proyecto.duration(duration))
# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
proyecto_asignacion = proyecto.copy()
gantt_recursos, dibujo = proyecto_asignacion.gantt(datos_proyecto,
                                                  'duracion',
                                                  'recursos',
                                                  total='fila',
                                                  holguras = True,
                                                  tikz=True)
pinta_tikz(dibujo)

# %% [markdown]
# La fila "Total" obtenida, que podemos representar gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_recursos, limite_maximo_recursos)
_c0 = _cargas(gantt_recursos, limite_maximo_recursos)
display(_c0)
_exc = [p for p in _c0.columns if _c0.loc['¿Supera?', p] == 'sí']
display(Markdown((f"Se supera el límite en {plural(len(_exc), 'periodo')}: "
                  + lista_y([f"periodo {p} (carga {_c0.loc['Carga', p]} > {_c0.loc['Límite', p]})" for p in _exc]) + '.')
                 if _exc else 'La carga no supera el límite en ningún periodo.'))
# %% [markdown]
# Como la demanda de recursos infringe el límite impuesto, es necesario aplicar el algoritmo de asignación.

# %% [markdown]
# ### Algoritmo de asignación

# %%
duracion_y_recursos = datos_proyecto[['duracion', 'recursos']]
proyecto_asignacion.display_asignar_pasos_tikz(data=duracion_y_recursos, duration_label='duracion', resource_label='recursos', maximo=limite_maximo_recursos)
my_data, gantt_df, dibujo  = proyecto_asignacion.asignar(
                                    data=duracion_y_recursos,
                                    duration_label='duracion',
                                    resource_label='recursos',
                                    maximo=limite_maximo_recursos,
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
_c1 = _cargas(gantt_df, limite_maximo_recursos)
display(_c1)
_T3 = proyecto_asignacion.duration(my_data['duracion'])
_desp = {a[1:]: int(d) for a, d in my_data['duracion'].items() if str(a).startswith('💤')}
display(Markdown(
    "**Respuesta:** hay que retrasar " + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _desp.items()])
    + f" respecto de su inicio más temprano (decisiones periodo a periodo en el visor del algoritmo de asignación). Con ello la carga máxima es "
    + f"{_c1.loc['Carga'].iloc[:4].max()} ≤ {limite_maximo_recursos[0]} en los cuatro primeros periodos y {_c1.loc['Carga'].iloc[4:].max()} ≤ {limite_maximo_recursos[-1]} después, "
    + f"y el proyecto pasa de {duracion_media_proyecto} a **{plural(_T3, 'periodo')}**."))
# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
PV = float(datos_proyecto['coste_planificado'].sum())
EV = float(datos_proyecto.eval('coste_planificado * realizado / 100').sum())
AC = float(datos_proyecto['coste_real'].sum())
CPI = EV/AC
SPI=EV/PV

# %% [markdown]
# En el punto de control se comparan tres magnitudes:
#
# - **PV** (*planned value*, valor planificado): coste presupuestado del trabajo que debería estar hecho en el punto de control. Como el proyecto, según la planificación, dura 8 periodos, en el periodo 8 todas las actividades deberían estar terminadas y PV es la suma de la columna `coste_planificado`.
# - **EV** (*earned value*, valor ganado): coste presupuestado del trabajo realmente hecho, es decir, `coste_planificado` × `realizado`/100 sumado para todas las actividades.
# - **AC** (*actual cost*, coste real): lo realmente gastado, la suma de la columna `coste_real`.
#
# Con ellas, el índice de rendimiento del coste es $CPI=EV/AC$ y el índice de rendimiento del cronograma es $SPI=EV/PV$.
#
# Con estas consideraciones, los valores resultantes son:

# %%
_d = datos_proyecto
Markdown(f"""
- PV = {' + '.join(f"{x:g}" for x in _d['coste_planificado'])} = **{PV:g}**
- EV = {' + '.join(f"{c:g}·{r:g}%" for c, r in zip(_d['coste_planificado'], _d['realizado']))} = {' + '.join(f"{c * r / 100:g}" for c, r in zip(_d['coste_planificado'], _d['realizado']))} = **{EV:g}**
- AC = {' + '.join(f"{x:g}" for x in _d['coste_real'])} = **{AC:g}**
- CPI = EV/AC = {EV:g}/{AC:g} = **{CPI:.2f}**
- SPI = EV/PV = {EV:g}/{PV:g} = **{SPI:.2f}**

**Respuesta:** CPI = {CPI:.2f} y SPI = {SPI:.2f}. {'Como CPI < 1, el proyecto gasta más de lo presupuestado para el trabajo realizado' if CPI < 1 else 'Como CPI ≥ 1, el proyecto no gasta más de lo presupuestado para el trabajo realizado'} 
(por cada unidad gastada se obtienen {CPI:.2f} unidades de valor), y {'como SPI < 1, va retrasado: se ha realizado solo el ' + f"{SPI:.0%}" + ' del trabajo planificado hasta el periodo 8' if SPI < 1 else 'como SPI ≥ 1, va adelantado o en plazo'}.""")

# %%
