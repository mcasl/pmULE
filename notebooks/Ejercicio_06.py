# ---
# title: Ejercicio 06
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
from pmule import explica_camino_critico, explica_probabilidad, ordena_ruta, lista_y, plural

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
# Dado un proyecto con los datos que se muestran en la tabla siguiente, se pide:

# %%
periodo_probabilidad_b = 9
limite_maximo_recursos = [4]
pregunta_1 = "1. Dibujar el diagrama Pert y determinar el camino crítico.  "
pregunta_2 = f"2. Probabilidad de terminar el proyecto antes de {periodo_probabilidad_b} periodos.  "
pregunta_3 = "3. Reducir la duración del proyecto hasta 10 periodos con el mínimo incremento posible de coste.  "
pregunta_4 = f"4. Si el número máximo de recursos durante las 6 primeras semanas es {limite_maximo_recursos[0]}, determinar si será posible realizar el proyecto en el tiempo obtenido en el apartado 1.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
# Alargamos el ancho de las celdas al 100%
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np

datos_enunciado= pd.DataFrame([
#actividad, precedentes, duracion, desviacion,  cur, recursos, duracion_tope
  (   'A',     '--- ',       3,         2,        5,      1,       1  ),
  (   'B',     '--- ',       2,         1,        5,      1,       1  ),
  (   'C',     'B   ',       4,         2,       25,      1,       2  ),
  (   'D',     'A   ',       2,         1,      -10,      2,       1  ),
  (   'E',     'A   ',       4,         0,        1,      1,       4  ),
  (   'F',     'E   ',       3,         1,        5,      1,       2  ),
  (   'G',     'C,D ',       1,         0,      -20,      2,       1  ),
  (   'H',     'C,D ',       5,         3,        1,      1,       2  ),
  (   'I',     'F,G,H',      1,         0,        1,      1,       1  ),
  (   'J',     'F,G  ',      2,         0,       -1,      1,       2 ),
  (   'K',     'F,G  ',      1,         0,        5,      2,       1  ),
], columns=['activity', 'predecessor', 'duration', 'standard_deviation', 'ucr', 'resources', 'cap_duration']).set_index('activity')
datos_enunciado

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadros de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.
#

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos
#
# Utilizando el cuadro de prelaciones expandido podemos dibujar el grafo Pert correspondiente al proyecto.

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto_apartado_1 = datos_proyecto.copy() # La necesitaremos en el apartado 4

# %%
proyecto = prelaciones.create_project()

# %%
#| fig-cap: "Grafo Pert con indicación con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto.display_pert_construccion(tabla=prelaciones)

# %% [markdown]
# Alternativamente podríamos haber representado su grafo Roy.
#

# %%
#| fig-cap: "Grafo Roy del proyecto"
#| label: fig-roy
proyecto.display_roy_construccion(tabla=prelaciones)

# %% [markdown]
# ### Matriz de Zaderenko
#
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
resultados_pert = proyecto.calculate_pert(datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Duración media del proyecto

# %%
duracion_media_proyecto = proyecto.duration(durations=datos_proyecto['duration'])
Markdown(f"El tiempo más temprano del nodo final del proyecto es su duración media: **{duracion_media_proyecto:g} periodos**.")

# %% [markdown]
# ### Calendario del proyecto
# Las fechas de inicio y fin más tempranas y tardías de cada actividad son:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# #### Holgura total de las actividades
# Del cálculo de las holguras totales obtenemos:

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
display(resultados_pert['activities'][['H_total']].rename(columns={'H_total': 'Holgura total'}))
display(Markdown('(Las filas que empiezan por @ son actividades ficticias.) Actividades con holgura total nula: **'
                 + lista_y([a for a in ordena_ruta(datos_proyecto.index, calendario) if resultados_pert['activities'].loc[a, 'H_total'] == 0]) + '**.'))
# %% [markdown]
# #### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Los visores siguientes lo muestran sobre los grafos Pert y Roy; a continuación se enumeran sus rutas.

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert con indicación con indicación del camino crítico"
#| label: fig-pertcritico
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy con indicación con indicación del camino crítico"
#| label: fig-Roycritico
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# Las rutas del camino crítico (actividades en orden de ejecución) son:
# %%
explica_camino_critico(proyecto, datos_proyecto['duration'], 'periodos')
# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
duracion_media_proyecto = proyecto.duration(durations=datos_proyecto['duration'])

# %%
Markdown(f"La duración media del proyecto la conocemos del primer apartado: μ = **{duracion_media_proyecto:g} periodos**.")

# %% [markdown]
# Para determinar la desviación típica debemos obtener en cada rama del camino crítico la suma de las varianzas de sus actividades (el enunciado da la desviación típica $\sigma$; la varianza es $\sigma^2$), y quedarnos con la mayor.

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, caminos_varianza = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas2
_tabla_varianzas(proyecto.critical_path(datos_proyecto['duration']), datos_proyecto['variance'], calendario)
# %%
_v = caminos_varianza.data['Variance'].astype(float)
display(Markdown(f"""La varianza de la duración del proyecto es la máxima varianza de las rutas críticas: σ² = máx({', '.join(f'{x:g}' for x in _v)}) = **{desviacion_proyecto**2:g}**.  
Su raíz cuadrada, la desviación típica, es el parámetro de la ley normal: σ = √{desviacion_proyecto**2:g} = **{desviacion_proyecto:.2f}** periodos."""))

# %% [markdown]
# Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad:

# %%
import scipy
from scipy.stats import norm

probabilidad_antes_b = norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, hasta=periodo_probabilidad_b)

Markdown(explica_probabilidad(duracion_media_proyecto, desviacion_proyecto, periodo_probabilidad_b)
         + f"\n\n**Respuesta:** la probabilidad de terminar el proyecto antes de {periodo_probabilidad_b} periodos es "
           f"**{probabilidad_antes_b:.4f}** ({probabilidad_antes_b * 100:.2f} %).")

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                min_durations=datos_proyecto['cap_duration'],
                costs=datos_proyecto['ucr'],
                reduction=2)

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

# %%
_respuesta_ackoff(best_option, datos_proyecto['ucr'], proyecto.duration(datos_proyecto['duration']), proyecto.duration(duration))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# ### Situación inicial de partida

# %% [markdown]
# Representamos la demanda de recursos por periodo utilizando un diagrama de Gantt.

# %%
#| fig-cap: "Diagrama de Gantt con indicación de las cargas por periodo."
#| label: fig-ganttPartida
proyecto_asignacion = proyecto.copy()
gantt_df, dibujo = proyecto_asignacion.gantt(datos_proyecto, 'duration', 'resources', total='fila', holguras = True, tikz=True)
pinta_tikz(dibujo)

# %%
#| fig-cap: "Diagrama de cargas del proyecto de acuerdo a su definición original, con indicación del límite máximo de recursos"
#| label: fig-preNivelacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=limite_maximo_recursos)
_n6 = 6
_c = _cargas(gantt_df)
_c6 = _c.loc['Carga'].iloc[:_n6]
display(Markdown(f"(La línea del límite se dibuja en todo el proyecto, pero el enunciado solo limita los recursos durante las {_n6} primeras semanas.) "
                 f"Cargas de las semanas 1 a {_n6} con la programación más temprana y las duraciones del apartado 1: "
                 + ', '.join(str(x) for x in _c6) + f"; máximo = {_c6.max()} frente a un límite de {limite_maximo_recursos[0]}."))
# %% [markdown]
# A partir de la semana 7 no hay límite de recursos, por lo que solo hay que comprobar las seis primeras semanas.

# %%
_ok = _c6.max() <= limite_maximo_recursos[0]
display(Markdown(
    f"**Respuesta:** {'sí' if _ok else 'no'} es posible realizar el proyecto en los {duracion_media_proyecto:g} periodos del apartado 1: "
    + (f"la carga máxima de las {_n6} primeras semanas es {_c6.max()} ≤ {limite_maximo_recursos[0]}, así que la programación más temprana ya respeta el límite y no hace falta retrasar ninguna actividad."
       if _ok else f"en las {_n6} primeras semanas la carga llega a {_c6.max()} > {limite_maximo_recursos[0]}; habría que retrasar actividades.")))
