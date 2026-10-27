# ---
# title: Ejercicio 07
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
from pmule import explica_camino_critico, ordena_ruta, lista_y, plural

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
# Considere el proyecto cuya información se adjunta. Realice cada apartado partiendo en todos los casos de la situación original del proyecto indicada en los datos del enunciado.

# %%
limite_maximo_recursos = [3]*7 + [10]
pregunta_1 = "1. Dibuje el grafo PERT y determine el camino crítico.  "
pregunta_2 = "2. Indique sus actuaciones si existiera un límite de disponibilidad de recursos durante los primeros 7 periodos igual a 3 unidades, y de 10 unidades para los siguientes periodos.  "
pregunta_3 = "3. Indique sus actuaciones para reducir la duración del proyecto hasta 28 periodos con un mínimo sobrecoste añadido.  "
pregunta_4 = "4. Indique sus actuaciones para reducir la duración del proyecto hasta su duración tope con un mínimo sobrecoste añadido.  "
pregunta_5 = """5. Duración del proyecto si se introducen los siguientes cambios:
      - Entre el final de la actividad H y el comienzo de la L deben transcurrir 2 periodos.
      - La tarea K se inicia cuando F está a mitad de su ejecución"""

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))
display(Markdown(pregunta_5))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas

import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, duracion_tope, cur, recursos
     ('A',       '---  ',     4,         1,          25,     2    ),
     ('B',       'A    ',     4,         1,          25,     1    ),
     ('C',       'A    ',     5,         2,          20,     2    ),
     ('D',       'A    ',     4,         1,          10,     2    ),
     ('E',       'A    ',     3,         2,          50,     2    ),
     ('F',       'D,E  ',     6,         3,          30,     4    ),
     ('G',       'F    ',     4,         1,          25,     1    ),
     ('H',       'B,C,G',     7,         4,          20,     1    ),
     ('I',       'C,G  ',     5,         2,          20,     3    ),
     ('J',       'I    ',     4,         1,          25,     4    ),
     ('K',       'F    ',     4,         1,          25,     1    ),
     ('L',       'H    ',     8,         5,          10,     2    ),
     ('M',       'I,K  ',     4,         1,          25,     1    ),

] ,columns=['activity', 'predecessor', 'duration', 'cap_duration', 'ucr', 'resources']).set_index('activity')

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
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
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
resultados_pert = proyecto.calculate_pert(datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Holgura total de las actividades
# Conocidos los tiempos tempranos y tardíos se puede proceder a calcular la holgura total de las actividades.

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
display(resultados_pert['activities'].rename(columns={'H_total': 'Holgura total'}))
display(Markdown('(Las filas que empiezan por @ son actividades ficticias.) Actividades con holgura total nula: **'
                 + lista_y([a for a in ordena_ruta(datos_proyecto.index, proyecto.calendar(datos_proyecto['duration'])) if resultados_pert['activities'].loc[a, 'H_total'] == 0]) + '**.'))
# %% [markdown]
# ### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas (actividades en orden de ejecución):

# %%
camino_critico = proyecto.critical_path(datos_proyecto['duration'])
display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'periodos'))
display(Markdown('**Respuesta:** el grafo PERT es el de las figuras anteriores y el camino crítico es el indicado.'))
# %% [markdown]
# ##### Grafo PERT con indicación de tiempos

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert del proyecto con indicación del camino crítico"
#| label: fig-pertCritico
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Alternativamente, se muestra el grafo Roy correspondiente con indicación del camino crítico.

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación del camino crítico"
#| label: fig-royCritico
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ### Calendario del proyecto
# Se muestra a continuación el calendario del proyecto con indicación de las fechas de inicio y fin más tempranas y tardías de cada actividad:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# #### Situación inicial
# Para responder este apartado es necesario representar la demanda de recursos a lo largo del tiempo.

# %%
proyecto_asignacion = proyecto.copy()
gantt_df, dibujo = proyecto_asignacion.gantt(datos_proyecto, 'duration', 'resources', total='fila', holguras = True, tikz=True,
                                                  params=dict(period_width=0.75))

# %%
#| fig-cap: "Diagrama de Gantt del proyecto en su estado inicial"
#| label: fig-preAsignacionGantt
pinta_tikz(dibujo)

# %% [markdown]
# La fila "Total", representada gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=limite_maximo_recursos)
_c0 = _cargas(gantt_df, limite_maximo_recursos)
display(_c0)
_exc = [p for p in _c0.columns if _c0.loc['¿Supera?', p] == 'sí']
display(Markdown((f"Se supera el límite en {plural(len(_exc), 'periodo')}: "
                  + lista_y([f"periodo {p} (carga {_c0.loc['Carga', p]} > {_c0.loc['Límite', p]})" for p in _exc]) + '.')
                 if _exc else 'La carga no supera el límite en ningún periodo.'))
# %% [markdown]
# Como la demanda de recursos infringe el límite impuesto durante los primeros 7 periodos, es necesario aplicar el algoritmo de asignación. En el algoritmo se usan los límites del enunciado: 3 recursos en los 7 primeros periodos y 10 a partir del octavo.

# %% [markdown]
# #### Algoritmo de asignación
# Se procede a aplicar el algoritmo de asignación sobre la planificación inicial del proyecto. El siguiente diagrama de Gantt muestra los desplazamientos realizados (área rayada al comienzo de las actividades).

# %%
duracion_y_recursos = datos_proyecto[['duration', 'resources']]
proyecto_asignacion.display_asignar_pasos_tikz(data=duracion_y_recursos, duration_label='duration', resource_label='resources', maximo=limite_maximo_recursos)
my_data, gantt_df, dibujo  = proyecto_asignacion.asignar(
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
plot_cargas(gantt_df=gantt_df, limite_maximo_recursos=limite_maximo_recursos)
_c2 = _cargas(gantt_df, limite_maximo_recursos)
display(_c2)
_T2 = int(proyecto_asignacion.duration(my_data['duration']))
_desp = {a[1:]: int(d) for a, d in my_data['duration'].items() if str(a).startswith('💤')}
_T0 = int(proyecto.duration(datos_proyecto['duration']))
display(Markdown(
    "**Respuesta:** hay que retrasar " + lista_y([f"{a} {plural(d, 'periodo')}" for a, d in _desp.items()])
    + f" respecto de su inicio más temprano (ver el visor del algoritmo de asignación). Con ello la carga máxima de los 7 primeros periodos es "
    + f"{_c2.loc['Carga'].iloc[:7].max()} ≤ 3 y la de los siguientes {_c2.loc['Carga'].iloc[7:].max()} ≤ 10, y el proyecto pasa de {_T0} a "
    + f"**{plural(_T2, 'periodo')}**."))

# %% [markdown]
# Con las operaciones indicadas el proyecto sería compatible con la restricción en el límite de recursos disponibles impuesta en el enunciado del ejercicio.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
proyecto_ackoff = proyecto.copy()
ackoff, best_option, duration, available = proyecto_ackoff.ackoff(durations=datos_proyecto['duration'],
                                                           min_durations=datos_proyecto['cap_duration'],
                                                           costs=datos_proyecto['ucr'],
                                                           reduction=5
                                                           )

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto_ackoff.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=5)

# %%
for iteracion, actividades in best_option.items():
    pass
_respuesta_ackoff(best_option, datos_proyecto['ucr'], proyecto.duration(datos_proyecto['duration']), proyecto_ackoff.duration(duration))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# La duración tope del proyecto es la que se obtiene con todas las actividades en su duración tope; pero acortar todas las actividades sería ineficiente, porque se pagaría la reducción de actividades que no son críticas. Por eso se aplica el algoritmo de Ackoff-Sasieni periodo a periodo hasta que alguna ruta crítica ya no pueda acortarse más: así solo se reducen, y solo lo necesario, las actividades que acortan el proyecto.

# %% [markdown]
# ::: {style="max-width: 100%; overflow-x: auto;"}

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni para reducción a tiempo tope del proyecto"
#| label: tbl-ackoffƒTope
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                                                           min_durations=datos_proyecto['cap_duration'],
                                                           costs=datos_proyecto['ucr'],
                                                           reduction=18,
                                                           report=False
                                                           )


proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=18)

# %%
for iteracion, actividades in best_option.items():
    pass
_Ttope = proyecto.duration(datos_proyecto['cap_duration'])
display(_respuesta_ackoff(best_option, datos_proyecto['ucr'], proyecto.duration(datos_proyecto['duration']), proyecto.duration(duration)))
display(Markdown(f"Comprobación: con todas las actividades en su duración tope el proyecto dura {_Ttope:g} periodos, "
                 + ("la misma duración alcanzada por el algoritmo" if np.isclose(_Ttope, proyecto.duration(duration)) else "distinta de la alcanzada por el algoritmo")
                 + f"; las actividades no reducidas (" + lista_y([a for a in datos_proyecto.index if duration[a] == datos_proyecto.loc[a, 'duration']])
                 + ") se mantienen en su duración normal, sin sobrecoste."))

# %% [markdown]
# ## Apartado 5

# %%
display(Markdown(pregunta_5))

# %% [markdown]
# La solución de este apartado precisa:  
#    - Introducir una actividad ficticia (que llamaremos 💤L) tras el final de H y anterior a L con duración 2 periodos.  
#    - Partir la actividad F (6 periodos) en dos partes F1 y F2 de 3 periodos cada una e iniciar K tras el final de F1.    
#
# El grafo PERT resultante de considerar estos cambios es el siguiente:

# %%
prelaciones_modificadas =  PredecessorTable.from_dict_of_sets(
    {
          'A': set(),
          'B': {'A'},
          'C': {'A'},
          'D': {'A'},
          'E': {'A'},
          'F1': {'D', 'E'},        # This is different. F1 represents the first part of F
          'F2': {'F1'},            # This is different. F2 represents the second part of F
          'G': {'F2'},             # This is different
          'H': {'B', 'C', 'G'},
          'I': {'C', 'G'},
          'J': {'I'},
          'K': {'F1'},           # This is different
          '💤L': {'H'},         # This is different. zzzL represent the waiting period before L can start.
          'L': {'💤L'},
          'M': {'I', 'K'}
    }
)

# %%
proyecto_modificado = prelaciones_modificadas.create_project()
#proyecto_modificado.pert()

# %%
duracion_modificado = datos_proyecto['duration'].drop('F')
duracion_modificado['F1'] = 3
duracion_modificado['F2'] = 3
duracion_modificado['💤L'] = 2

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert del proyecto modificado con indicación del camino crítico"
#| label: fig-pertcriticoModificado
proyecto_modificado.display_pert_pasos(duracion_modificado)

# %%
gantt_df, dibujo = proyecto_modificado.gantt(duracion_modificado.to_frame(),
                                             duration_label='duration',
                                             tikz=True,
                                             params=dict(
                                                 period_width=0.75,
                                             )
                                        )

# %% [markdown]
# Esta nueva planificación puede ser también representada en el siguiente diagrama de Gantt, donde la zona rayada representa la espera de dos periodos (💤L).

# %%
#| fig-cap: "Diagrama de Gantt del proyecto modificado"
#| label: fig-GanttModificado
pinta_tikz(dibujo)

# %%
duracion_proyecto_modificado = proyecto_modificado.duration(durations=duracion_modificado.to_frame()['duration'])
display(explica_camino_critico(proyecto_modificado, duracion_modificado, 'periodos'))
Markdown(f"**Respuesta:** la duración del proyecto tras estos cambios es **{duracion_proyecto_modificado:g} periodos** "
         f"(frente a {proyecto.duration(datos_proyecto['duration']):g} en la situación original): la espera 💤L de {duracion_modificado['💤L']:g} periodos "
         + ("está en el camino crítico y alarga el proyecto, " if any('💤L' in r for r in proyecto_modificado.critical_path(duracion_modificado).values()) else "no está en el camino crítico, ")
         + "mientras que adelantar el inicio de K no acorta el proyecto porque K "
         + ("no es crítica." if not any('K' in r for r in proyecto_modificado.critical_path(duracion_modificado).values()) else "es crítica."))

# %%
