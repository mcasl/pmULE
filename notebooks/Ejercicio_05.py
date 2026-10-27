# ---
# title: Ejercicio 05
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

sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, pinta_tikz
from pmule import explica_camino_critico, explica_probabilidad, ordena_ruta, lista_y, plural

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
periodo_probabilidad_b = 12

pregunta_1 = "1.  Dibuje el diagrama Pert y determine el camino crítico.  "
pregunta_2 = "2.  Indique el coste de realización para duración PERT  "
pregunta_3 = f"3.  A la vista de la planificación dada, su jefe le dice que el proyeco tiene que ejecutarse en {periodo_probabilidad_b} o menos periodos. ¿Con qué probabilidad estima que se verán cumplidos los deseos de su jefe?  "
pregunta_4 = "4.  Reduzca la duración del proyecto dos unidades de tiempo con el menor incremento de coste posible.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))
display(Markdown(pregunta_4))

# %%
import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas
import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, b, m, a, coste_duracion_pert, coste_duracion_tope, duracion_tope
     ('A', '---  ',   6,  2.5, 2, 1600, 1800, 2),
     ('B', '---  ',   2,    2, 2, 2900, 2900, 1),
     ('C', '---  ',   1,    1, 1,  600,  600, 1),
     ('D', '---  ',   1,    1, 1,  600,  600, 1),
     ('E', 'A,B  ',  12,    5, 4, 1600, 2200, 2),
     ('F', 'A,B  ',  12,  2.5, 2, 2000, 2200, 1),
     ('G', 'B,C  ',   6, 1.25, 1,  600,  700, 1),
     ('H', 'B,C  ',  20,   13, 6, 3500, 4900, 3),
     ('I', 'D    ',  30,   12, 6, 3500, 4300, 4),
     ('J', 'E,M  ',   4,    4, 4, 1500, 1500, 2),
     ('K', 'F    ',   7,    7, 7, 2000, 2000, 3),
     ('L', 'J,K,G',   1,    1, 1, 1000, 1000, 1),
     ('M', 'F    ',   3,    3, 3,  500,  500, 1),

] ,columns=['activity', 'predecessors', 'b', 'm', 'a',
            'cost_pert_duration', 'cost_cap_duration', 'cap_duration']).set_index('activity')

datos_enunciado['b'] = datos_enunciado['b'].astype(int)
datos_enunciado

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# # Solución
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %% [markdown]
# ### Cuadro de prelaciones expandido
#
# Comenzamos construyendo el cuadro de prelaciones. Este cuadro nos permitirá construir los grafos Pert o Roy, si atendemos a la información de las filas; o comprobar si el grafo obtenido es correcto, atendiendo a la información de las columnas.
#

# %%
#| tbl-cap: "Cuadro de prelaciones expandido"
#| label: tbl-prelaciones
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessors')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# ### Precálculo de variables de interés

# %% [markdown]
# A partir de los datos del enunciado (duraciones optimista $a$, más probable $m$ y pesimista $b$; costes con duración PERT y con duración tope) calculamos, para cada actividad, la duración media $D$, la desviación típica $\sigma$ y el coste unitario de reducción (CUR), es decir, lo que cuesta acortar la actividad un periodo:

# %%
display(Math(r"$D=\frac{a+4m+b}{6}$"))
display(Math(r"$\sigma=\frac{b-a}{6}$"))
display(Math(r"$CUR=\frac{C_{tope}-C_{PERT}}{D_{PERT}-d_{tope}}$"))

# %% [markdown]
# {{< pagebreak >}}

# %%
datos_proyecto['duration'] = datos_proyecto.eval('(b + 4*m +a)/6')
datos_proyecto['standard_deviation'] = datos_proyecto.eval('(b -a)/6').round(2)
datos_proyecto['ucr'] = datos_proyecto.eval('(cost_cap_duration - cost_pert_duration)/(duration - cap_duration)').round(2)

# %%
#| tbl-cap: "Duración y desviación típica de las actividades calculadas"
#| label: tbl-datosInteresantes
def _cur(r):
    if r['duration'] == r['cap_duration']:
        return f"no reducible (D = d tope = {r['cap_duration']:g})"
    return (f"({r['cost_cap_duration']:g} − {r['cost_pert_duration']:g})/({r['duration']:g} − {r['cap_duration']:g}) = {r['ucr']:.2f}")
pd.DataFrame({
    'D = (a + 4m + b)/6': [f"({r['a']:g} + 4·{r['m']:g} + {r['b']:g})/6 = {r['duration']:g}" for _, r in datos_proyecto.iterrows()],
    'σ = (b − a)/6': [f"({r['b']:g} − {r['a']:g})/6 = {r['standard_deviation']:.2f}" for _, r in datos_proyecto.iterrows()],
    'CUR': [_cur(r) for _, r in datos_proyecto.iterrows()],
}, index=datos_proyecto.index)
# %% [markdown]
# ### Grafo PERT con numeración de nodos
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
resultados_pert = proyecto.calculate_pert(datos_proyecto['duration'])
resultados_pert['nodes']

# %% [markdown]
# ### Duración media del proyecto

# %%
duracion_media_proyecto = int(float(proyecto.duration(datos_proyecto['duration'])))
Markdown(f"El tiempo más temprano del nodo final del proyecto es su duración media: **{duracion_media_proyecto} periodos**.")

# %% [markdown]
# ### Calendario del proyecto
# Las fechas de inicio y fin más tempranas y tardías de cada actividad son:

# %%
#| tbl-cap: "Calendario del proyecto"
#| label: tbl-calendario
calendario = proyecto.calendar(datos_proyecto['duration'])
proyecto.display_calendario(datos_proyecto['duration'])

# %% [markdown]
# ### Holgura total de las actividades
# De acuerdo a los cálculos realizados, la holgura total de las actividades es:

# %%
#| tbl-cap: "Holgura total de las actividades"
#| label: tbl-holguras
display(resultados_pert['activities'].rename(columns={'H_total': 'Holgura total'}))
display(Markdown('(Las filas que empiezan por @ son actividades ficticias.) Actividades con holgura total nula: **'
                 + lista_y([a for a in ordena_ruta(datos_proyecto.index, calendario) if resultados_pert['activities'].loc[a, 'H_total'] == 0]) + '**.'))
# %% [markdown]
# ### Camino crítico
# Las actividades con holgura total cero forman el camino crítico. Según el grafo Pert del proyecto, éste está compuesto por las siguientes rutas (actividades en orden de ejecución):

# %%
camino_critico = proyecto.critical_path(datos_proyecto['duration'])
display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'periodos'))
display(Markdown(f"**Respuesta:** el grafo PERT es el de las figuras anteriores y el camino crítico lo forman las {len(camino_critico)} rutas anteriores "
                 f"(actividades críticas: {lista_y([a for a in ordena_ruta(datos_proyecto.index, calendario) if any(a in r for r in camino_critico.values())])})."))
# %% [markdown]
# ##### Grafo PERT con indicación de tiempos y camino crítico

# %% [markdown]
# Cálculo de los tiempos tempranos y tardíos de los sucesos y de las holguras totales paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Pert del proyecto con indicación del camino crítico"
#| label: fig-pertCritico
proyecto.display_pert_pasos(datos_proyecto['duration'])

# %% [markdown]
# Alternativamente, se muestra el grafo roy correspondiente con indicación del camino crítico.

# %% [markdown]
# Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso. El primer paso muestra el grafo resuelto; a partir del siguiente se calcula desde el principio (avance con los botones):

# %%
#| fig-cap: "Grafo Roy del proyecto con indicación del camino crítico"
#| label: fig-royCritico
proyecto.display_roy_pasos(datos_proyecto['duration'])

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %% [markdown]
# El coste del proyecto con todas las actividades en su duración PERT es la suma de los costes de la columna `cost_pert_duration`:

# %%
Markdown(' + '.join(f"{a}({c:g})" for a, c in datos_proyecto['cost_pert_duration'].items())
         + f" = **{int(datos_proyecto['cost_pert_duration'].sum())}**\n\n**Respuesta:** el coste de realización con duración PERT "
         f"({duracion_media_proyecto} periodos) es de **{int(datos_proyecto['cost_pert_duration'].sum())}** unidades monetarias.")

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# ### Duración media del proyecto

# %% [markdown]
# La duración media del proyecto ya la tenemos calculada:

# %%
duracion_media_proyecto = int(proyecto.duration(datos_proyecto['duration']))

# %%
Markdown(f"La media de la duración del proyecto es μ = **{duracion_media_proyecto} periodos**.")

# %% [markdown]
# ### Varianza de la duración del proyecto
#
# Se calcula entonces la varianza de cada una de las rutas del camino crítico como suma de las varianzas ($\sigma^2$, con la $\sigma$ redondeada a dos decimales de la tabla anterior) de las actividades que la componen. La ruta con máxima varianza determina la varianza del proyecto.

# %%
datos_proyecto['variance'] = datos_proyecto.eval('standard_deviation ** 2')
desviacion_proyecto, varianza_caminos = proyecto.standard_deviation(durations=datos_proyecto['duration'],
                                                  variances=datos_proyecto['variance'])

# %%
#| tbl-cap: "Varianza de las rutas del camino crítico"
#| label: tbl-varianzaRutas
_tabla_varianzas(camino_critico, datos_proyecto['variance'].round(2), calendario)
# %%
_v = varianza_caminos.data['Variance'].astype(float)
Markdown(f"Varianza del proyecto: σ² = máx({', '.join(f'{x:g}' for x in _v)}) = **{desviacion_proyecto**2:g}** "
         f"({lista_y([r for r in _v.index if np.isclose(_v[r], _v.max())])}). "
         f"La desviación típica es su raíz cuadrada: σ = √{desviacion_proyecto**2:g} = **{float(desviacion_proyecto):.2f}** periodos.")

# %% [markdown]
# ### Cálculo de la probabilidad

# %% [markdown]
# Una vez caracterizada la distribución normal con la que aproximamos la duración del proyecto, podemos hacer el cálculo de la probabilidad:

# %%
import scipy
from scipy.stats import norm

probabilidad_de_terminar_antes_de_b = norm.cdf(periodo_probabilidad_b, loc=duracion_media_proyecto, scale=desviacion_proyecto)
from pmule import display_normal_tikz
display_normal_tikz(duracion_media_proyecto, desviacion_proyecto, hasta=periodo_probabilidad_b)
probabilidad_de_terminar_antes_de_b  = round(float(probabilidad_de_terminar_antes_de_b * 100), 2)

# %%
Markdown(explica_probabilidad(duracion_media_proyecto, desviacion_proyecto, periodo_probabilidad_b)
         + f"\n\n**Respuesta:** la probabilidad de terminar en {periodo_probabilidad_b} periodos o menos es del **{probabilidad_de_terminar_antes_de_b} %**: "
           "los deseos del jefe tienen pocas probabilidades de cumplirse con la planificación actual.")

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ## Apartado 4.

# %%
display(Markdown(pregunta_4))

# %% [markdown]
# Se aborda la solución de este apartado mediante la técnica de reducción de la duración del proyecto con mínimo coste.

# %%
ackoff, best_option, durations, available = proyecto.ackoff(durations=datos_proyecto['duration'],
                                                            min_durations=datos_proyecto['cap_duration'],
                                                            costs=datos_proyecto['ucr'],
                                                            reduction=2)

# %%
#| tbl-cap: "Cuadro del algoritmo de Ackoff Sasieni"
#| label: tbl-ackoff
proyecto.display_ackoff_pasos_tikz(
    durations=datos_proyecto['duration'],
    min_durations=datos_proyecto['cap_duration'],
    costs=datos_proyecto['ucr'],
    reduction=2)

# %% [markdown]
# Las actividades reducidas en cada una de las iteraciones, y su coste, son:

# %%
for iteracion, actividades in best_option.items():
    pass
display(_respuesta_ackoff(best_option, datos_proyecto['ucr'], proyecto.duration(datos_proyecto['duration']), proyecto.duration(durations)))
_inc = sum(datos_proyecto.loc[a, 'ucr'] for acts in best_option.values() for a in acts)
display(Markdown(f"El coste total del proyecto pasa a ser {int(datos_proyecto['cost_pert_duration'].sum())} + {_inc:g} = "
                 f"**{datos_proyecto['cost_pert_duration'].sum() + _inc:g}** unidades monetarias. "
                 "(Las actividades con CUR = 0 se pueden acortar sin coste: su coste con duración tope es igual al de la duración PERT.)"))
# %%

# %%
