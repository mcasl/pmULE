# ---
# title: Ejercicio 10
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
# # Enunciado
#
# Se pretende construir una planta desalinizadora de agua, con el fin de proporcionar agua potable a ciertas poblaciones en las que escasea. Usted, como encargado de la planificación de la ejecución de este proyecto, decide que la descomposición en tareas y sus precedencias es la que se muestra en la tabla adjunta, donde la duración se expresa en meses.  
#
# Tras un detenido análisis del proyecto en cuestión, usted ha llegado a la conclusión de que no es posible que la tarea J empiece nada más terminar la tarea A, sino que es absolutamente necesario que para que se pueda empezar a ejecutar la tarea J debe de haber transcurrido un mes después de terminar la tarea A. Se pide:  

# %%
pregunta_1 = "1. Dibujar el diagrama PERT del proyecto y caracteríce su duración mediante una distribución aleatoria Normal.  "
pregunta_2 = "2. Como no está conforme con la duración del proyecto que resulta, desea reducir la duración del proyecto en 1 mes con el mínimo incremento de coste posible. ¿Qué tareas deben ser reducidas?.  "
pregunta_3 = """3. Debido al racionamiento de combustible diesel que sufre la isla, durante el proyecto sólo será posible disponer de 250 litros de gasoil cada semana; y debido a la falta de previsión de los responsables del proyecto, que no solicitaron permiso para el almacenamiento de combustible, no es posible guardar el sobrante de una semana para consumirlo la siguiente. Por otro lado las necesidades de consumo de gasoil que usted planifica son las que se muestran en la tabla.  
- ¿Se encuentra bien nivelado el consumo de gasoil?.  
- ¿Se encuentra bien asignado?.  
- Si no es así ¿qué recomienda hacer para asignar correctamente el consumo de combustible?.  
- ¿Qué consecuencias traerá la asignación correcta del combustible en este proyecto en concreto?.  """

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))

# %%
#| tbl-cap: "Cuadro de datos del enunciado del ejercicio"
#| label: tbl-datosEnunciado

import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes,        duracion, varianza, coste,   litros, duracion_tope
  ('A',       '---  ',               5,        2,    '600-60t',   210,    2),
  ('B',       '---  ',               5,        1,    '300-20t',   150,    2),
  ('C',       'B    ',               1,        1,    '100-30t',    50,    1),
  ('D',       'B    ',               5,        1,    '300-25t',    75,    1),
  ('E',       'B    ',               2,        1,    '200-30t',    25,    1),
  ('F',       'C,D,E',               1,        1,    '100-10t',    75,    1),
  ('G',       'D,E  ',               2,        1,    '200-10t',    75,    2),
  ('H',       'E    ',               2,        1,    '150-50t',     0,    1),
  ('I',       'K    ',               4,        1,    '500-50t',   100,    3),
  ('💤J',      'A    ',               1,        0,         None,     0,    1),
  ('J',       '💤J   ',               2,        1,    '100-10t',    50,    1),
  ('K',       'A    ',               3,        1,    '300-10t',    50,    3),
] ,columns=['actividad', 'precedentes', 'duracion', 'varianza', 'coste', 'litros', 'duracion_tope']).set_index('actividad')

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
prelaciones = PredecessorTable.from_dataframe_of_strings(data=datos_enunciado, activity='actividad', predecessor='precedentes')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Grafo PERT con numeración de nodos

# %%
#| fig-cap: "Grafo Pert con indicación de los números de nodos"
#| label: fig-pertNumeracionNodos
proyecto = prelaciones.create_project()
datos_proyecto = datos_enunciado.copy()
proyecto.display_pert_construccion(tabla=prelaciones)

# %% [markdown]
# ### Matriz de Zaderenko

# %% [markdown]
# Matriz de Zaderenko paso a paso. El primer paso muestra el resultado final; a partir del siguiente se construye desde el principio (avance con los botones):

# %%
#| tbl-cap: "Matriz de Zaderenko para el cálculo de tiempos tempranos y tardíos"
#| label: tbl-Zaderenko
proyecto.display_zaderenko_tikz(datos_proyecto['duracion'])

# %% [markdown]
# Los tiempos tempranos y tardios han resultado ser:

# %%
#| tbl-cap: "Valores de los tiempos tempranos y tardíos de los nodos"
#| label: tbl-TempranosTardios
resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duracion'])
resultados_pert['nodes']

# %%
duracion_media_proyecto = int(proyecto.duration(durations=datos_proyecto['duracion']))
Markdown(f"El tiempo más temprano del nodo final del proyecto es su duración media: **{duracion_media_proyecto} meses**.")

# %% [markdown]
# ### Holgura total de las actividades
# A partir de los tiempos tempranos y tardíos, y de las duraciones de las actividades, podemos determinar la holgura total de las actividades y con ellas las rutas del camino crítico.

# %%
#| tbl-cap: "Valor de la holgura total de las actividades"
#| label: tbl-htotal
display(resultados_pert['activities'].rename(columns={'H_total': 'Holgura total'}))
display(Markdown('(Las filas que empiezan por @ son actividades ficticias; 💤J es la espera de un mes entre A y J.) Actividades con holgura total nula: **'
                 + lista_y([a for a in ordena_ruta(datos_proyecto.index, proyecto.calendar(datos_proyecto['duracion'])) if resultados_pert['activities'].loc[a, 'H_total'] == 0]) + '**.'))
# %% [markdown]
# ### Camino crítico

# %% [markdown]
# El camino crítico consta de las siguientes rutas (actividades en orden de ejecución):

# %%
camino_critico = proyecto.critical_path(durations=datos_proyecto['duracion'])
for key, value in camino_critico.items():
    pass
explica_camino_critico(proyecto, datos_proyecto['duracion'], 'meses')
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
# La caracterización de la distribución Normal consiste en determinar los valores de sus parámetros: media y desviación típica de su duración.

# %%
duracion_media_proyecto = int(proyecto.duration(datos_proyecto['duracion']))

# %%
Markdown(f"La duración media del proyecto ya la tenemos calculada: μ = **{duracion_media_proyecto} meses**.")

# %% [markdown]
# Para determinar la desviación típica debemos obtener en cada rama del camino crítico la suma de las varianzas de sus actividades (dato del enunciado), y quedarnos con la mayor.

# %%
desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duracion'],
                                                   variances=datos_proyecto['varianza'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
_tabla_varianzas(camino_critico, datos_proyecto['varianza'], calendario)
# %%
_v = varianza_rutas.data['Variance'].astype(float)
Markdown(f"Varianza del proyecto: σ² = máx({', '.join(f'{x:g}' for x in _v)}) = {desviacion_proyecto**2:g} meses²; "
         f"desviación típica σ = √{desviacion_proyecto**2:g} = {desviacion_proyecto:g} meses.\n\n"
         f"**Respuesta:** la duración del proyecto se caracteriza por una normal de media **{duracion_media_proyecto} meses** y desviación típica "
         f"**{desviacion_proyecto:g} meses**, T ~ N({duracion_media_proyecto}; {desviacion_proyecto:g}).")


# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# Para resolver la reducción del proyecto con mínimo coste es necesario obtener el valor del coste unitario de reducción a partir de los datos del enunciado.

# %%
datos_proyecto = datos_enunciado.copy()
datos_proyecto[['coste']]

# %% [markdown]
# El coste de cada actividad viene dado como función lineal de su duración $t$ (por ejemplo, $600-60t$): cada mes que se reduce la duración el coste aumenta en el coeficiente de $t$. El coste unitario de reducción (CUR) es, por tanto, ese coeficiente (en valor absoluto):

# %%
datos_proyecto['cur'] = ( datos_proyecto.loc[:, 'coste']
                                        .str.split('[-+]', expand=True)[1]
                                        .str.split('t', expand=True)[0]
                                        .astype(float))
datos_proyecto[['cur']].T

# %% [markdown]
# Se aborda la solución de este apartado mediante la técnica de reducción de la duración del proyecto con mínimo coste.

# %%
ackoff, best_option, duration, available = proyecto.ackoff(durations=datos_proyecto['duracion'],
                min_durations=datos_proyecto['duracion_tope'],
                costs=datos_proyecto['cur'],
                reduction=1)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoffƒ
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duracion'],
    min_durations=datos_proyecto['duracion_tope'],
    costs=datos_proyecto['cur'],
    reduction=1)
_cc = proyecto.critical_path(datos_proyecto['duracion'])
_lin = []
for _r, _acts in _cc.items():
    _acts = ordena_ruta(_acts, calendario)
    _red = [a for a in _acts if datos_proyecto.loc[a, 'duracion'] > datos_proyecto.loc[a, 'duracion_tope']]
    _nored = [a for a in _acts if a not in _red and not str(a).startswith('💤')]
    _best = min(_red, key=lambda a: datos_proyecto.loc[a, 'cur'])
    _lin.append(f"- {_r} ({' – '.join(_acts)}): reducibles " + lista_y([f"{a} (CUR {datos_proyecto.loc[a, 'cur']:g}, {plural(datos_proyecto.loc[a, 'duracion'] - datos_proyecto.loc[a, 'duracion_tope'], 'mes disponible', 'meses disponibles')})" for a in _red])
                + (f"; {lista_y(_nored)} ya {'está' if len(_nored) == 1 else 'están'} en su duración tope" if _nored else '')
                + f" → la más barata es **{_best}**.")
_acts_red = [a for acts in best_option.values() for a in acts]
_coste = sum(datos_proyecto.loc[a, 'cur'] for a in _acts_red)
display(Markdown('\n'.join(_lin) + f"\n\n**Respuesta:** deben reducirse un mes las tareas **{lista_y(_acts_red)}**, con un incremento de coste de "
                 + ' + '.join(f"{a}({datos_proyecto.loc[a, 'cur']:g})" for a in _acts_red) + f" = **{_coste:g}** unidades monetarias; "
                 f"el proyecto pasa de {proyecto.duration(datos_proyecto['duracion']):g} a {proyecto.duration(duration):g} meses."))
# %% [markdown]
# Para acortar el proyecto un mes hay que acortar a la vez todas las rutas críticas. No hay ninguna actividad común a las dos, así que en cada una se elige la actividad más barata de las que aún pueden reducirse (duración mayor que la duración tope), como se razona a continuación.

# %% [markdown]
# ## Apartado 3.

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# Los litros de la tabla del enunciado son consumos semanales y la planificación está en meses: consideramos que un mes son cuatro semanas, de modo que el consumo mensual de cada actividad es 4 veces el semanal y el límite disponible es 4 × 250 = 1000 litros al mes.

# %%
#| tbl-cap: "Cuadro de coste mensual del combustible"
#| label: tbl-costeMensual
datos_proyecto['litros_por_mes'] = 4 * datos_enunciado['litros']   # Se considera el gasto mensual 4 veces el coste semanal
datos_proyecto[['litros_por_mes']].T

# %%
gantt_recursos, dibujo = proyecto.gantt(datos_proyecto,
                                'duracion',
                                'litros_por_mes',
                                total='fila',
                                holguras=True,
                                tikz=True)

# %% [markdown]
# De acuerdo con la planificación inicial, el consumo de combustible resulta ser:

# %%
#| fig-cap: "Distribución inicial del consumo de recursos"
#| label: fig-recursosInicial
pinta_tikz(dibujo)

# %% [markdown]
# La zona rayada antes de J representa la espera de 1 periodo mencionada en el enunciado.

# %% [markdown]
# La fila "Total" obtenida, que podemos representar gráficamente en la siguiente figura, proporciona la información del consumo previsto de recursos a lo largo del tiempo.

# %%
#| fig-cap: "Diagrama de cargas antes de aplicar el algoritmo de asignación"
#| label: fig-preAsignacionBarras
plot_cargas(gantt_recursos, limite_maximo_recursos=1000)
_lim = 4 * 250
_c0 = _cargas(gantt_recursos, _lim)
display(_c0)
_niv = proyecto.copy().nivelar(data=datos_proyecto[['duracion', 'litros_por_mes']], duration_label='duracion', resource_label='litros_por_mes')
_sc = _niv[1].data if hasattr(_niv[1], 'data') else _niv[1]
_movn = {a: int(v) for a, v in _niv[0].iloc[:, 0].items() if v != 0}
_exc = [p for p in _c0.columns if _c0.loc['¿Supera?', p] == 'sí']
display(Markdown(
    f"- **¿Bien nivelado?** " + ("No" if _movn else "Sí") + f": el consumo mensual oscila entre {_c0.loc['Carga'].min()} y {_c0.loc['Carga'].max()} litros "
    + f"(suma de cuadrados {int((_c0.loc['Carga'] ** 2).sum())}); "
    + (("aplicando el algoritmo de nivelación por mínimos cuadrados, desplazando " + lista_y([f"{a} {plural(d, 'mes', 'meses')}" for a, d in _movn.items()])
        + f" dentro de su holgura la suma de cuadrados baja a {float(_sc.iloc[-1].min()):g}.") if _movn else
       ("el algoritmo de nivelación por mínimos cuadrados no encuentra ningún desplazamiento de las actividades no críticas ("
        + lista_y(list(_niv[0].index)) + ") que reduzca la suma de cuadrados, así que, dentro de lo que permite la nivelación (sin alargar el proyecto), "
        "el consumo ya está nivelado. Su irregularidad (los picos de los primeros meses) se debe a actividades críticas, que no pueden moverse sin retrasar el proyecto.")) + "\n"
    + (f"- **¿Bien asignado?** No: se superan los {_lim} litros mensuales en {plural(len(_exc), 'mes', 'meses')} ("
       + lista_y([f"mes {p}: {_c0.loc['Carga', p]}" for p in _exc]) + ").")
    if _exc else f"- **¿Bien asignado?** Sí: no se superan los {_lim} litros mensuales."))
# %% [markdown]
# Como la programación inicial supera los 1000 litros mensuales disponibles, el proyecto no es ejecutable en su estado actual y debe aplicarse el algoritmo de asignación.

# %% [markdown]
# ### Algoritmo de asignación

# %% [markdown]
# Durante los primeros cinco meses no es viable ejecutar A y B a la vez (840 + 600 = 1440 > 1000 litros). Como ambas son críticas y de la misma duración (5 meses), el criterio de prioridad del algoritmo no las distingue y resulta indiferente desplazar una u otra, pues en ambos casos esta acción conlleva un retraso del proyecto. Resolvemos el ejercicio desplazando la actividad B.

# %%
proyecto_asignacion = proyecto.copy()
duracion_y_recursos = datos_proyecto[['duracion', 'litros_por_mes']]
proyecto_asignacion.display_asignar_pasos_tikz(data=duracion_y_recursos, duration_label='duracion', resource_label='litros_por_mes', maximo=[1000])
datos_proyecto_tras_asignar, gantt_df, dibujo  = proyecto_asignacion.asignar(
                                    data=duracion_y_recursos,
                                    duration_label='duracion',
                                    resource_label='litros_por_mes',
                                    maximo=[1000],
                                    report=False
                                    )


# %%
#| fig-cap: "Diagrama de Gantt del proyecto una vez aplicado el algoritmo de asignación"
#| label: fig-postAsignacion
pinta_tikz(dibujo)

# %% [markdown]
# La zona rayada representa el desplazamiento de la tarea _B_.

# %%
#| fig-cap: "Diagrama de cargas tras aplicar el algoritmo de asignación"
#| label: fig-postAsignacionBarras
plot_cargas(gantt_df, limite_maximo_recursos=1000)
display(_cargas(gantt_df, _lim))
_T3 = proyecto_asignacion.duration(datos_proyecto_tras_asignar['duracion'])
# %% [markdown]
# Con esta única acción (retrasar B) la demanda de recursos del proyecto ya no supera los 1000 litros mensuales que permite el enunciado (tabla anterior). La duración, no obstante, aumenta y el camino crítico cambia, como se calcula a continuación.

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
proyecto_asignacion.display_pert_pasos(datos_proyecto_tras_asignar['duracion'])

# %%
camino_critico = proyecto_asignacion.critical_path(durations=datos_proyecto_tras_asignar['duracion'])
for key, value in camino_critico.items():
    pass
display(explica_camino_critico(proyecto_asignacion, datos_proyecto_tras_asignar['duracion'], 'meses'))
_cal3 = proyecto_asignacion.calendar(datos_proyecto_tras_asignar['duracion'])
_desp = {a[1:]: int(d) for a, d in datos_proyecto_tras_asignar['duracion'].items() if str(a).startswith('💤') and a[1:] in datos_proyecto.index and a != '💤J'}
display(Markdown(
    "**Respuesta:** para asignar correctamente el combustible hay que retrasar " + lista_y([f"{a} {plural(d, 'mes', 'meses')}" for a, d in _desp.items()])
    + f" (hasta que termine A). Consecuencia: como B era crítica, el proyecto pasa de {duracion_media_proyecto} a **{_T3:g} meses** "
    + f"({plural(_T3 - duracion_media_proyecto, 'mes', 'meses')} más), y cambia el camino crítico: ahora es el indicado arriba, mientras que A pasa a tener "
    + f"una holgura total de {plural(_cal3.loc['A', 'H_total'], 'mes', 'meses')}."))
# %% [markdown]
# Este resultado cumple los requerimientos expresados en el enunciado. No obstante, resultará sorprendente que la actividad 💤B sea crítica al inicio del proyecto cuando representa una espera, mientras la actividad A dispone de holgura total y no es crítica.
# Realmente, al imponer el enunciado en este apartado un límite máximo de recursos que impide que A y B puedan ejecutarse simultaneamente, ha sido equivalente a añadir una restricción de prelación entre A y B que obliga que B se ejecute después de A; y por tanto A podría considerarse crítica en esa nueva configuración.
#
# Podría abundarse más en otras posibles consideraciones y alternativas de solución pero éstas escapan del alcance de la aplicación estricta del método de asignación de  Wiest-Levy básico explicado en el curso.

# %%
