# ---
# title: Ejercicio 14
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
# %%capture
from jupyter_tikz import TexFragment
import os, sys
from IPython.display import display, HTML, Markdown, Math

# To find local version of the library
sys.path.append('../src')

from pmule import PredecessorTable, ProjectGraph, EarnedValue, pinta_tikz, plural, lista_y, ordena_ruta, limpia_tabla
# %mkdir -p ~/temp
# %cd ~/temp

# %% [markdown]
# ## Enunciado
# Considere la información del proyecto cuya información aparece en la tabla adjunta:  

# %%
pregunta_1 = "1.  Calcule los tres diagramas de Gantt correspondientes al análisis de valor ganado.  "
pregunta_2 = "2.  Calcule EV, PV, AC, CPI y SPI para el punto de control actual.  "
pregunta_3 = "3.  Calcule utilizando el método de MonteCarlo la duración media y la desviación típica del resto de proyecto que falta por completar. Suponga para ello que la duración aleatoria de las actividades ya empezadas es proporcional a su duración planificada original.  "

# %%
display(Markdown(pregunta_1))
display(Markdown(pregunta_2))
display(Markdown(pregunta_3))

# %%
# Alargamos el ancho de las celdas al 100%
from IPython.display import display, HTML, Math, Latex, Markdown
import pandas as pd
pd.set_option('display.max_colwidth', None)   # sin recortar con «…» los textos largos de las tablas

import numpy as np

datos_enunciado = pd.DataFrame([
#  actividad, precedentes, duracion, desviación, coste_planificado, coste_real, duracion_real, realizado
     ('A',      '---    ',       2,          0.5,         60,              80,          2,           100,   ),
     ('B',      '---    ',       2,          0.2,         60,              60,          3,           100,   ),
     ('C',      '---    ',       3,          0.6,         60,              80,          4,           100,   ),
     ('D',      'A      ',       4,          0.1,         40,              60,          5,           100,   ),
     ('E',      'A      ',       1,          0.3,         30,              36,          3,           100,   ),
     ('F',      'E, B   ',       2,          0.7,         12,              12,          3,           100,   ),
     ('G',      'E, B   ',       1,          0.2,         39,              54,          3,           100,   ),
     ('H',      'G, C   ',       4,          0.3,         36,              72,          2,            50,   ),
     ('I',      'D, F   ',       1,          0.7,         90,              90,          2,            60,   ),
     ('J',      'D, F, H',       2,          0.3,         50,               0,          0,             0,   ),
     ('K',      'D, F, H',       2,          0.9,         40,               0,          0,             0,   ),
     ('L',      'I, J   ',       2,          0.3,         30,               0,          0,             0,   ),
     ('M',      'K      ',       2,          0.6,         30,               0,          0,             0,   ),
] ,columns=['actividad', 'precedentes', 'duracion', 'desviacion',
            'coste_planificado', 'coste_real', 'duracion_real', 'realizado']).set_index('actividad')
datos_enunciado

# %%
numero_de_muestras = 10

nombres_de_actividades = datos_enunciado.index
np.random.seed(2025)
duraciones_montecarlo = pd.DataFrame([np.random.normal(size = numero_de_muestras,
                                                       loc  = datos_enunciado.duracion[task],
                                                       scale= datos_enunciado.desviacion[task])
                                      for task in nombres_de_actividades],
                                     index=nombres_de_actividades).T.round(1)

# %%
duraciones_montecarlo

# %% [markdown]
# {{< pagebreak >}}

# %% [markdown]
# ### Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
datos_proyecto = datos_enunciado.copy()
prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='actividad', predecessor='precedentes')
prelaciones.display_immediate_linkage_matrix()

# %% [markdown]
# Grafo PERT del proyecto con las duraciones planificadas. Su calendario de fechas más tempranas es la referencia del
# valor planificado (PV); el valor ganado (EV) y el coste real (AC) se reparten según el calendario real, que se obtiene
# con el mismo grafo y las duraciones reales.

# %%
proyecto = prelaciones.create_project()
proyecto.pert()

# %% [markdown]
# #### Gantt Valor Planificado

# %%
vg = EarnedValue(proyecto)

# %%
gantts = vg.calcula_gantts(datos_proyecto,
                           planned_durations_label    = 'duracion',
                           actual_durations_label     = 'duracion_real',
                           PV_label                   = 'coste_planificado',
                           AC_label                   = 'coste_real',
                           percentage_complete_label  = 'realizado'
                          )
_cal_plan = proyecto.calendar(datos_proyecto['duracion'])
_cal_real = proyecto.calendar(datos_proyecto['duracion_real'])
_t_control = int(gantts['acumulados'].index.max())
_T_plan = proyecto.duration(datos_proyecto['duracion'])
_BAC = datos_proyecto['coste_planificado'].sum()
_frac = datos_proyecto['realizado'] / 100
_reparto = pd.DataFrame({
    'duración plan.': datos_proyecto['duracion'],
    'inicio plan.': _cal_plan['inicio_mas_temprano'],
    'fin plan.': _cal_plan['fin_mas_temprano'],
    'coste plan.': datos_proyecto['coste_planificado'],
    'PV/periodo': datos_proyecto['coste_planificado'] / datos_proyecto['duracion'],
    'duración real': datos_proyecto['duracion_real'],
    'inicio real': _cal_real['inicio_mas_temprano'].where(datos_proyecto['duracion_real'] > 0),
    'fin real': _cal_real['fin_mas_temprano'].where(datos_proyecto['duracion_real'] > 0),
    '% realizado': datos_proyecto['realizado'],
    'EV': _frac * datos_proyecto['coste_planificado'],
    'EV/periodo': (_frac * datos_proyecto['coste_planificado'] / datos_proyecto['duracion_real']).where(datos_proyecto['duracion_real'] > 0),
    'AC': datos_proyecto['coste_real'],
    'AC/periodo': (datos_proyecto['coste_real'] / datos_proyecto['duracion_real']).where(datos_proyecto['duracion_real'] > 0),
})
display(Markdown(
    f"Los tres diagramas se construyen repartiendo un importe de cada actividad **uniformemente** a lo largo de su duración:\n\n"
    f"- **Gantt del valor planificado (PV)**: el coste planificado se reparte sobre la duración planificada, en el calendario "
    f"planificado (fechas más tempranas); PV por periodo = coste planificado / duración planificada. El proyecto planificado dura "
    f"{plural(_T_plan, 'periodo')} y su presupuesto total es BAC (*Budget At Completion*) = {_BAC:g}.\n"
    f"- **Gantt del valor ganado (EV)**: el valor ganado de cada actividad, EV = % realizado × coste planificado, se reparte sobre su "
    f"duración real, en el calendario real.\n"
    f"- **Gantt del coste real (AC)**: el coste real se reparte sobre la duración real, en el calendario real.\n\n"
    f"El calendario real (mismo grafo con las duraciones reales) llega hasta el periodo {_t_control}: esa es la **fecha de control**. "
    f"Las actividades sin empezar ({lista_y(datos_proyecto.index[datos_proyecto['duracion_real'] == 0])}) no aparecen en los "
    f"diagramas de EV y AC. La tabla recoge los cálculos de cada fila de los diagramas (PV/periodo = coste plan. / duración plan.; EV = % realizado × coste plan.; EV/periodo = EV / duración real; AC/periodo = AC / duración real):"))
display(limpia_tabla(_reparto))

# %%
pinta_tikz(gantts['gantt_PV'])
_pv_acum = gantts['acumulados']['PV']
display(Markdown(f"**Respuesta (Gantt PV):** la fila *Total* es el PV de cada periodo y la fila *Acumulado* su suma acumulada (la curva PV); "
                 f"el acumulado llega a BAC = {_BAC:g} al final del periodo {_T_plan}."))

# %% [markdown]
# #### Gantt Valor Ganado

# %%
pinta_tikz(gantts['gantt_EV'])
display(Markdown(f"**Respuesta (Gantt EV):** el acumulado de la última columna (periodo {_t_control}, fecha de control) es el valor "
                 f"ganado total, EV = {gantts['acumulados']['EV'].iloc[-1]:g}."))

# %% [markdown]
# #### Gantt Coste Real

# %%
pinta_tikz(gantts['gantt_AC'])
display(Markdown(f"**Respuesta (Gantt AC):** el acumulado del periodo {_t_control} es el coste real total, "
                 f"AC = {gantts['acumulados']['AC'].iloc[-1]:g}."))

# %% [markdown]
# ### Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
acumulados = gantts['acumulados']
acumulados.plot();

# %%
acumulados.tail(1)

# %%
EV = acumulados.tail(1)['EV'].iloc[0]
AC = acumulados.tail(1)['AC'].iloc[0]
PV = acumulados.tail(1)['PV'].iloc[0]
CPI = EV/AC
SPI=EV/PV
_gpv = gantts['Gantt_PV'].data
_cols = [c for c in _gpv.columns if str(c) != 'Total' and int(c) <= _t_control]
_pv_act = _gpv.loc[datos_proyecto.index, _cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
_tabla_v = pd.DataFrame({
    'fin plan.': _cal_plan['fin_mas_temprano'],
    f'PV hasta t = {_t_control}': _pv_act,
    '% realizado': datos_proyecto['realizado'],
    'coste plan.': datos_proyecto['coste_planificado'],
    'EV = % × coste plan.': _frac * datos_proyecto['coste_planificado'],
    'AC': datos_proyecto['coste_real'],
})
_tabla_v.loc['Total'] = [np.nan, _pv_act.sum(), np.nan, _BAC, (_frac * datos_proyecto['coste_planificado']).sum(), datos_proyecto['coste_real'].sum()]
display(Markdown(
    f"Valores de cada actividad en la fecha de control t = {_t_control}. El PV de una actividad es la parte de su coste "
    f"planificado que el plan preveía tener hecha en t = {_t_control} (suma de su fila del Gantt PV hasta esa columna): "
    f"las que terminaban antes reciben todo su coste; las que en el plan estaban a medias, la parte proporcional."))
display(limpia_tabla(_tabla_v))

# %%
Markdown(
    f"- Valor planificado: **PV = {PV:g}** (lo que debía estar hecho en t = {_t_control} según el plan).\n"
    f"- Valor ganado: **EV** = Σ % realizado × coste planificado = **{EV:g}**.\n"
    f"- Coste real: **AC** = Σ coste real = **{AC:g}**.\n"
    f"- Índice de rendimiento del coste: **CPI** = EV / AC = {EV:g} / {AC:g} = **{CPI:.2f}**"
    f"{' < 1: se gasta más de lo presupuestado (por cada unidad gastada se obtienen ' + format(CPI, '.2f') + ' de valor).' if CPI < 1 else ' ≥ 1: no hay sobrecoste.'}\n"
    f"- Índice de rendimiento del plazo: **SPI** = EV / PV = {EV:g} / {PV:g} = **{SPI:.2f}**"
    f"{' < 1: el proyecto va retrasado (se ha ejecutado el ' + format(SPI, '.0%') + ' de lo planificado hasta la fecha).' if SPI < 1 else ' ≥ 1: no hay retraso.'}")

# %% [markdown]
# ### Apartado 3

# %%
display(Markdown(pregunta_3))

# %% [markdown]
# #### Fragmento de proyecto por realizar

# %%
prelaciones_restantes = PredecessorTable.from_dict_of_strings(
    { 'H': '',
      'I': '',
      'J': 'H',
      'K': 'H',
      'L': 'I, J',
      'M': 'K'
    }
    )


# %%
proyecto_restante = prelaciones_restantes.create_project()
proyecto_restante.pert(durations=False)

# %% [markdown]
# #### Cálculo de las duraciones del resto del proyecto

# %%
datos_proyecto_restante = datos_proyecto.copy().drop(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
restantes = datos_proyecto_restante.query('realizado != 100')
duraciones_montecarlo = duraciones_montecarlo.reindex(restantes.index, axis=1)
duraciones_montecarlo = duraciones_montecarlo.eval('H = H * 0.5')
duraciones_montecarlo = duraciones_montecarlo.eval('I = I * 0.4')
_empezadas = restantes.index[restantes['realizado'] > 0]
_neg = [(a, i, duraciones_montecarlo.loc[i, a]) for a in duraciones_montecarlo.columns
        for i in duraciones_montecarlo.index if duraciones_montecarlo.loc[i, a] < 0]
_pm = proyecto_restante.path_matrix(dummies=False)
_dr = _pm @ duraciones_montecarlo.T
_neg_influye = any(_dr[i].max() in [_dr.loc[r, i] for r in _pm.index if _pm.loc[r, a] == 1] for a, i, v in _neg)
display(Markdown(
    "Quedan por completar " + lista_y(restantes.index) + ". Se usan las 10 muestras simuladas de la tabla del enunciado. "
    "Las actividades no empezadas conservan su duración simulada completa; para las empezadas, la duración restante es "
    "proporcional a la fracción que falta: duración restante = (1 − % realizado) × duración simulada:\n\n"
    + "\n".join(f"- {a}: realizado {restantes.loc[a, 'realizado']} % → duración restante = {1 - restantes.loc[a, 'realizado'] / 100:g} × {a} "
                f"(p. ej., iteración 0: {1 - restantes.loc[a, 'realizado'] / 100:g} × "
                f"{duraciones_montecarlo.loc[0, a] / (1 - restantes.loc[a, 'realizado'] / 100):.1f} = {duraciones_montecarlo.loc[0, a]:g})"
                for a in _empezadas)
    + ("\n\n*Nota*: al muestrear una normal con desviación grande respecto a la media aparecen duraciones negativas ("
       + lista_y([f"{a} en la iteración {i}: {v:g}" for a, i, v in _neg])
       + "), que no tienen sentido físico"
       + ("; aquí no influyen en el resultado porque en esas iteraciones su ruta no es la más larga." if not _neg_influye
          else "; conviene tenerlo en cuenta porque afectan a la duración de la ruta más larga.")
       if _neg else "")))
display(duraciones_montecarlo)

# %% [markdown]
# #### Cálculo de la duración del proyecto

# %% [markdown]
# Rutas del fragmento pendiente: cada fila de la matriz de rutas es una ruta y un 1 indica que la actividad pertenece a ella.

# %%
proyecto_restante.display_rutas("pert")

# %% [markdown]
# Multiplicando la matriz de rutas por las duraciones de las actividades, obtenemos las duraciones de las rutas en cada iteración.

# %%
duraciones_caminos =  proyecto_restante.path_matrix(dummies=False) @ duraciones_montecarlo.T
_rutas_acts = {r: ordena_ruta(list(_pm.columns[_pm.loc[r] == 1]), proyecto_restante.calendar(duraciones_montecarlo.loc[0]))
               for r in _pm.index}
display(Markdown("\n".join(
    f"- {r}: {' – '.join(acts)}; iteración 0: {' + '.join(format(duraciones_montecarlo.loc[0, a], 'g') for a in acts)} "
    f"= {duraciones_caminos.loc[r, 0]:.2f}" for r, acts in _rutas_acts.items())))
display(duraciones_caminos)

# %% [markdown]
# Considerando en cada iteración el máximo de las duraciones de las rutas obtenemos los valores de las duraciones del proyecto.

# %%
duraciones_proyecto = pd.DataFrame(duraciones_caminos.max(axis="rows"))
duraciones_proyecto.columns=['Duración']
_mas_larga = duraciones_caminos.idxmax(axis='rows').value_counts()
display(Markdown(
    f"Por ejemplo, en la iteración 0: máx({', '.join(format(v, '.2f') for v in duraciones_caminos[0])}) = "
    f"{duraciones_proyecto.iloc[0, 0]:.2f}. La ruta más larga es "
    + lista_y([f"{r} ({' – '.join(_rutas_acts[r])}) en {plural(n, 'iteración', 'iteraciones')}" for r, n in _mas_larga.items()])
    + " (en caso de empate se cuenta la primera)."))
display(duraciones_proyecto)

# %% [markdown]
# Con estos valores podemos describir la distribución de la duración del proyecto a través de su media y desviación típica.

# %%
duraciones_proyecto.plot(kind='density');

# %%
_x = duraciones_proyecto['Duración']
Markdown(f"Duración media del resto del proyecto: $\\bar x = \\frac{{\\sum x_i}}{{n}}$ = "
         f"({' + '.join(format(v, '.2f') for v in _x)}) / {len(_x)} = {_x.sum():.2f} / {len(_x)} = **{_x.mean():.2f}** periodos.")

# %%
_ss = ((_x - _x.mean()) ** 2).sum()
Markdown(f"Desviación típica muestral: $s = \\sqrt{{\\frac{{\\sum (x_i - \\bar x)^2}}{{n-1}}}}$ = "
         f"√({_ss:.4f} / {len(_x) - 1}) = **{_x.std():.2f}** periodos.\n\n"
         f"**Respuesta:** el resto del proyecto dura en media **{_x.mean():.2f}** periodos, con desviación típica **{_x.std():.2f}** "
         f"periodos (estimación con solo {len(_x)} iteraciones de Montecarlo, por tanto poco precisa).")

# %%

# %%

# %%
