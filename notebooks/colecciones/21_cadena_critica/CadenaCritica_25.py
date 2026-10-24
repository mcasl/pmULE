# ---
# title: CadenaCritica 25
# subtitle: "cadena de 36 días, CP de 9; cuatro informes (amarilla → amarilla → verde → amarilla)"
# coleccion: 21_cadena_critica
# variante: tendencia-fiebre
# nivel: 2
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
import os, sys
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, Markdown, Math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Versión local de la librería: se busca src/pmule.py subiendo desde el directorio actual
_aqui = os.path.abspath('.')
while _aqui != os.path.dirname(_aqui) and not os.path.isfile(os.path.join(_aqui, 'src', 'pmule.py')):
    _aqui = os.path.dirname(_aqui)
sys.path.append(os.path.join(_aqui, 'src'))
from pmule import (PredecessorTable, ProjectGraph, EarnedValue,
                   pinta_tikz, plot_cargas, plot_curvas_valor_ganado,
                   tanto_por_uno, ordena_rutas)

GANTT_PARAMS = dict(arrows=True)   # flechas de prelación en los diagramas de Gantt (rojas las críticas)


# %%
# Funciones auxiliares para mostrar los cálculos intermedios paso a paso
from itertools import combinations as _combinations


def detalle_tiempos_nodos(proyecto, duraciones):
    """Tiempos tempranos y tardíos de los nodos con los candidatos que se comparan en cada uno."""
    G = proyecto.pert_graph
    d = {data['activity']: (0 if data['activity'].startswith('@') else int(duraciones[data['activity']]))
         for _, _, data in G.edges(data=True)}
    nodos = list(proyecto.nodes)
    early, filas = {}, []
    for j in nodos:
        entrantes = [(i, data['activity']) for i, _, data in G.in_edges(j, data=True)]
        if not entrantes:
            early[j] = 0
            filas.append({'nodo': j, 'candidatos t_i + d_ij': 'nodo inicial', 'tiempo temprano = máx': 0})
            continue
        cand = {f'{a}: {early[i]} + {d[a]} = {early[i] + d[a]}': early[i] + d[a] for i, a in entrantes}
        early[j] = max(cand.values())
        filas.append({'nodo': j, 'candidatos t_i + d_ij': '; '.join(cand), 'tiempo temprano = máx': early[j]})
    tempranos = pd.DataFrame(filas).set_index('nodo')
    late, filas = {}, []
    for i in reversed(nodos):
        salientes = [(j, data['activity']) for _, j, data in G.out_edges(i, data=True)]
        if not salientes:
            late[i] = early[i]
            filas.append({'nodo': i, 'candidatos t*_j − d_ij': 'nodo final (= su tiempo temprano)', 'tiempo tardío = mín': late[i]})
            continue
        cand = {f'{a}: {late[j]} − {d[a]} = {late[j] - d[a]}': late[j] - d[a] for j, a in salientes}
        late[i] = min(cand.values())
        filas.append({'nodo': i, 'candidatos t*_j − d_ij': '; '.join(cand), 'tiempo tardío = mín': late[i]})
    tardios = pd.DataFrame(filas).set_index('nodo')
    return tempranos, tardios


def detalle_holguras(calendario, prelaciones):
    """Cómo se obtiene cada fecha y cada holgura a partir de las vecinas."""
    pred = prelaciones.immediate_predecessor
    suc = {a: sorted(s for s, ps in pred.items() if a in ps) for a in calendario.index}
    fin = int(calendario['fin_mas_temprano'].max())
    filas = []
    for a in calendario.index:
        ES, EF = int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'fin_mas_temprano'])
        LS, LF = int(calendario.loc[a, 'inicio_mas_tardio']), int(calendario.loc[a, 'fin_mas_tardio'])
        dur = int(calendario.loc[a, 'duracion'])
        ps, ss = sorted(pred[a]), suc[a]
        ef_pred = {p: int(calendario.loc[p, 'fin_mas_temprano']) for p in ps}
        es_suc = {s: int(calendario.loc[s, 'inicio_mas_temprano']) for s in ss}
        lf_pred = {p: int(calendario.loc[p, 'fin_mas_tardio']) for p in ps}
        ls_suc = {s: int(calendario.loc[s, 'inicio_mas_tardio']) for s in ss}
        min_es_suc = min(es_suc.values()) if es_suc else fin
        max_lf_pred = max(lf_pred.values()) if lf_pred else 0
        filas.append({
            'actividad': a,
            'ES = máx EF pred.': (f"máx({', '.join(f'{p}: {v}' for p, v in ef_pred.items())}) = {ES}" if ps else f'inicial → {ES}'),
            'EF = ES + d': f'{ES} + {dur} = {EF}',
            'LF = mín LS suc.': (f"mín({', '.join(f'{s}: {v}' for s, v in ls_suc.items())}) = {LF}" if ss else f'final → {LF}'),
            'LS = LF − d': f'{LF} − {dur} = {LS}',
            'H_total = LS − ES': f'{LS} − {ES} = {LS - ES}',
            'H_libre = mín ES suc. − EF': f"{min_es_suc}{'' if ss else ' (fin)'} − {EF} = {min_es_suc - EF}",
            'H_indep. = máx(0, mín ES suc. − máx LF pred. − d)': f'máx(0, {min_es_suc} − {max_lf_pred} − {dur}) = {max(0, min_es_suc - max_lf_pred - dur)}',
        })
    return pd.DataFrame(filas).set_index('actividad')


def detalle_rutas(R, valores, nombre='duración', decimales=0):
    """Suma término a término de un valor (duraciones, varianzas…) a lo largo de cada ruta."""
    filas = {}
    for r in R.index:
        acts = list(R.columns[R.loc[r] == 1])
        total = float(sum(valores[a] for a in acts))
        filas[r] = {'actividades': ' - '.join(acts),
                    f'{nombre} = suma': ' + '.join(f'{a}({valores[a]:.{decimales}f})' for a in acts) + f' = {total:.{decimales}f}',
                    nombre: total}
    return pd.DataFrame(filas).T


def detalle_estimaciones(datos, a='a', m='m', b='b'):
    """Media y varianza PERT de cada actividad con los números sustituidos."""
    filas = {}
    for act, fila in datos.iterrows():
        A, M, B = float(fila[a]), float(fila[m]), float(fila[b])
        filas[act] = {'D = (a + 4m + b)/6': f'({A:g} + 4·{M:g} + {B:g})/6 = {(A + 4 * M + B) / 6:.2f}',
                      'σ² = ((b − a)/6)²': f'(({B:g} − {A:g})/6)² = {((B - A) / 6) ** 2:.3f}'}
    return pd.DataFrame(filas).T


def detalle_ackoff(proyecto, duraciones, topes, cur, pasos):
    """En cada paso de Ackoff-Sasieni: rutas críticas, opciones consideradas (actividades comunes y parejas)
    con su coste, y la elegida."""
    R = proyecto.path_matrix(dummies=False)
    d = duraciones.astype(float).copy()
    filas = []
    for paso, elegidas in sorted(pasos.items()):
        L = R @ d[R.columns]
        T = float(L.max())
        criticas = [r for r in L.index if np.isclose(L[r], T)]
        conjuntos = [set(R.columns[R.loc[r] == 1]) for r in criticas]
        con_margen = {x for x in R.columns if d[x] > topes[x] and np.isfinite(cur[x])}
        opciones = {x: float(cur[x]) for x in set.intersection(*conjuntos) & con_margen}
        if len(conjuntos) > 1:
            cand = sorted(set().union(*conjuntos) & con_margen)
            for p, q in _combinations(cand, 2):
                if all(p in s or q in s for s in conjuntos) and p not in opciones and q not in opciones:
                    opciones[f'{p}+{q}'] = float(cur[p] + cur[q])
        orden = sorted(opciones.items(), key=lambda kv: kv[1])
        filas.append({'paso': paso + 1, 'duración': int(T), 'rutas críticas': ', '.join(criticas),
                      'opciones (coste)': '; '.join(f'{k} ({v:.0f})' for k, v in orden[:6]) + (' …' if len(orden) > 6 else ''),
                      'elegida': ', '.join(elegidas), 'coste del paso': float(sum(cur[x] for x in elegidas))})
        for x in elegidas:
            d[x] -= 1
    return pd.DataFrame(filas).set_index('paso')


def detalle_valor_ganado(datos_vg, calendario, semana):
    """PV, EV y AC actividad a actividad en el punto de control."""
    filas = {}
    for a, fila in datos_vg.iterrows():
        es, dur = int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'duracion'])
        frac = min(max(semana - es, 0), dur) / dur
        filas[a] = {'BAC': float(fila['BAC']), 'planificado hasta la semana': f'{min(max(semana - es, 0), dur)} de {dur}',
                    'PV = BAC·fracción': f'{fila["BAC"]:.0f}·{frac:.2f} = {fila["BAC"] * frac:.1f}',
                    'realizado (%)': int(fila['performed']),
                    'EV = BAC·%': f'{fila["BAC"]:.0f}·{fila["performed"] / 100:.2f} = {fila["BAC"] * fila["performed"] / 100:.1f}',
                    'AC': float(fila['AC']), 'PV_i': fila['BAC'] * frac, 'EV_i': fila['BAC'] * fila['performed'] / 100}
    t = pd.DataFrame(filas).T
    return t


def muestra_formula(nombre, expresion, valor, unidades=''):
    """Una línea «nombre = expresión = valor» en Markdown."""
    return Markdown(f'**{nombre}** = {expresion} = **{valor}**{(" " + unidades) if unidades else ""}')

# %%
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp


# %% [markdown]
# # Enunciado

# %% [markdown]
# Un proyecto se gestiona con el método de la cadena crítica. La cadena crítica dura 36 días con las estimaciones agresivas, el colchón del proyecto es de 9 días y la fecha comprometida, el día 45. El cuadro recoge cuatro informes de avance: el día del informe, la fecha prevista de fin de la cadena (día del informe más la duración restante estimada) y los días de cadena completados (duración de la cadena menos su duración restante; de una actividad en curso no se cuenta como restante más que su duración agresiva). Zonas del gráfico de fiebre: verde por debajo de la recta que va del 10 % de consumo con la cadena sin empezar al 70 % con la cadena terminada; roja por encima de la recta que va del 30 % al 90 %; amarilla entre ambas.

# %%
#| tbl-cap: "Informes de avance (días)"
#| label: tbl-datosEnunciado
datos_enunciado = pd.DataFrame([
#  dia, fin_previsto, completado
    (8  , 38          , 6         ),
    (17 , 39          , 14        ),
    (25 , 40          , 21        ),
    (34 , 42          , 28        ),
], columns=['dia', 'fin_previsto', 'completado']).set_index('dia')
datos_enunciado

# %%
pregunta_1 = "1. Para cada informe calcule el porcentaje de colchón consumido, el de cadena completada, el BPI (porcentaje de colchón restante entre porcentaje de cadena restante) y la zona.  "
pregunta_2 = "2. Calcule la velocidad de consumo del colchón entre cada dos informes consecutivos: puntos de porcentaje de colchón consumidos por cada punto de porcentaje de cadena completada. Compárela con la pendiente de las fronteras de las zonas (0,6). ¿Qué indica una velocidad mayor que 0,6? ¿Y una mayor que 1?  "
pregunta_3 = "3. Si se mantuviera la velocidad del último tramo, ¿qué porcentaje del colchón se habría consumido al terminar la cadena y qué día terminaría el proyecto? ¿Se cumpliría la fecha comprometida? ¿Con qué porcentaje de cadena completada entraría el proyecto en la zona roja?  "
pregunta_4 = "4. Demuestre que el BPI de un informe es la velocidad media de consumo que el colchón puede soportar hasta el final de la cadena. Compárelo con la velocidad del último tramo y diga qué debería hacer el jefe de proyecto.  "

for item in sorted(dir()):
    if item.startswith('pregunta_'):
        display(Markdown(eval(item)))

# %% [markdown]
# # Resolución

# %%
from pmule import grafico_fiebre_tikz
T, CP = 36, 9
informes = datos_enunciado.copy()

# %% [markdown]
# ## Apartado 1

# %%
display(Markdown(pregunta_1))

# %%
#| tbl-cap: "Consumo, avance, BPI y zona de cada informe"
#| label: tbl-puntos
x = 100 * informes['completado'] / T
y = 100 * (informes['fin_previsto'] - T) / CP
bpi = (100 - y) / (100 - x)
lim_verde, lim_roja = 10 + 0.6 * x, 30 + 0.6 * x
zona = ['verde' if yy < lv else ('roja' if yy >= lr else 'amarilla') for yy, lv, lr in zip(y, lim_verde, lim_roja)]
pd.DataFrame({f'día {d}': {
    'colchón consumido': f"({informes.loc[d, 'fin_previsto']} − {T})/{CP} = {y[d]:.1f} %",
    'cadena completada': f"{informes.loc[d, 'completado']}/{T} = {x[d]:.1f} %",
    'BPI': f'(100 − {y[d]:.1f})/(100 − {x[d]:.1f}) = {bpi[d]:.2f}',
    'límites verde / rojo': f'{lim_verde[d]:.1f} % / {lim_roja[d]:.1f} %',
    'zona': z} for d, z in zip(informes.index, zona)})

# %% [markdown]
# ## Apartado 2

# %%
display(Markdown(pregunta_2))

# %%
#| tbl-cap: "Velocidad de consumo entre informes consecutivos"
#| label: tbl-velocidad
dias = list(informes.index)
v = {f'{d1} → {d2}': (y[d2] - y[d1]) / (x[d2] - x[d1]) for d1, d2 in zip(dias, dias[1:])}
pd.DataFrame({k: {'velocidad = Δ consumo / Δ avance': f'({y[d2]:.1f} − {y[d1]:.1f})/({x[d2]:.1f} − {x[d1]:.1f}) = {v[k]:.2f}'}
              for (k, (d1, d2)) in zip(v, zip(dias, dias[1:]))})

# %% [markdown]
# Las dos fronteras suben 0,6 puntos de consumo por cada punto de avance. Con una velocidad mayor que 0,6 el punto del gráfico se acerca a la frontera superior: el proyecto empeora respecto a las zonas aunque siga en la misma. Con una velocidad mayor que 1 el colchón se consume, en porcentaje, más deprisa de lo que avanza la cadena: si se mantuviera desde el inicio, el colchón se agotaría antes de terminarla. Con una velocidad negativa el proyecto recupera colchón.

# %% [markdown]
# ## Apartado 3

# %%
display(Markdown(pregunta_3))

# %%
dn, dp = dias[-1], dias[-2]
vn = v[f'{dp} → {dn}']
y100 = y[dn] + vn * (100 - x[dn])
fin_tendencia = T + y100 / 100 * CP
x_roja = (30 - y[dn] + vn * x[dn]) / (vn - 0.6)
texto = (f'Con la velocidad del último tramo ({vn:.2f}), al terminar la cadena se habría consumido {y[dn]:.1f} + {vn:.2f}·(100 − {x[dn]:.1f}) = '
         f'**{y100:.1f} %** del colchón: el proyecto terminaría hacia el día {T} + {y100 / 100:.3f}·{CP} = **{fin_tendencia:.1f}**, '
         + ('dentro de la fecha comprometida' if y100 <= 100 else f'{fin_tendencia - T - CP:.1f} días después de la fecha comprometida') + f' ({T + CP}). ')
texto += (f'La recta de la tendencia, y = {y[dn]:.1f} + {vn:.2f}·(x − {x[dn]:.1f}), corta la frontera roja, y = 30 + 0,6·x, en '
          f'x = (30 − {y[dn]:.1f} + {vn:.2f}·{x[dn]:.1f})/({vn:.2f} − 0,6) = **{x_roja:.1f} %** de cadena completada'
          + (': entraría en la zona roja antes de terminar.' if x_roja < 100 else ', después del final: no llegaría a entrar en la zona roja.'))
Markdown(texto)

# %%
#| fig-cap: "Gráfico de fiebre con los informes y la tendencia del último tramo (discontinua)"
#| label: fig-fiebre
puntos = [(x[d], y[d], f'día {d}') for d in dias]
dibujo = grafico_fiebre_tikz(puntos + [(100.0, y100, 'tendencia')], unir=False)
linea = r'\draw[thick, gray] ' + ' -- '.join(f'({px:.2f},{py:.2f})' for px, py, _ in puntos) + ';'
tendencia = rf'\draw[thick, dashed, blue!70!black] ({x[dn]:.2f},{y[dn]:.2f}) -- (100,{y100:.2f});'
pinta_tikz(dibujo.replace(r'\end{tikzpicture}', linea + '\n' + tendencia + '\n' + r'\end{tikzpicture}'))

# %% [markdown]
# ## Apartado 4

# %%
display(Markdown(pregunta_4))

# %%
Markdown(f'''Si en un informe se han consumido y % del colchón con x % de cadena completada, para terminar justo con el 100 % del colchón 
consumido la velocidad media hasta el final debe ser (100 − y)/(100 − x), que es exactamente el BPI. Un BPI mayor que 1 dice que queda 
proporcionalmente más colchón que cadena; comparado con la velocidad real dice si el ritmo de consumo es sostenible.


En el último informe el BPI es {bpi[dn]:.2f} y la velocidad del último tramo, {vn:.2f}: '''
         + ('el ritmo actual agotaría el colchón antes de terminar la cadena. ' if vn > bpi[dn] else
            'el ritmo actual es sostenible: el colchón alcanzaría hasta el final. ')
         + ({'verde': 'Aunque el informe está en zona verde, conviene preparar ya un plan de recuperación y vigilar el siguiente informe.',
             'amarilla': f'El informe está en zona amarilla, así que el plan de recuperación ya debe estar preparado; con esta tendencia '
                         f'habrá que ejecutarlo hacia el {x_roja:.0f} % de cadena completada, al entrar en la zona roja, o antes si el '
                         'siguiente informe confirma la tendencia.'}[zona[-1]] if vn > bpi[dn] else
            {'verde': 'Con el informe en zona verde basta con seguir vigilando la tendencia.',
             'amarilla': 'Con el informe en zona amarilla se mantiene preparado el plan de recuperación, sin ejecutarlo todavía.'}[zona[-1]])
         + ' La tendencia es una extrapolación: el siguiente informe puede confirmarla o desmentirla.')

# %% [markdown]
# ## Autoevaluación

# %% [markdown]
# Preguntas de respuesta múltiple con los datos de este ejercicio (la correcta, marcada con ✅, es la primera de cada lista). La celda las exporta también en formato GIFT, con las opciones barajadas, a `~/temp/cuestionario_cadena_critica.gift`, para importarlas en Moodle.

# %%
preguntas_test = [
    ('¿Cuál es la velocidad de consumo del colchón en el último tramo (entre los dos últimos informes)?',
     ['1,14', '2,47', '0,86', '1,50'],
     '(66,7 − 44,4)/(77,8 − 58,3) = 1,14 puntos de colchón por punto de cadena.'),
    ('Si se mantiene esa velocidad, ¿qué porcentaje del colchón se habrá consumido al terminar la cadena?',
     ['92,1 %', '85,7 %', '80,0 %', '100,0 %'],
     '66,7 + 1,14·(100 − 77,8) = 92,1 %.'),
    ('Si la velocidad de consumo es mayor que el BPI del último informe, ¿qué ocurrirá si se mantiene?',
     ['el colchón se agotará antes de terminar la cadena', 'el proyecto entrará de inmediato en la zona roja', 'el proyecto terminará antes de la fecha comprometida', 'habrá que recalcular la cadena crítica'],
     'El BPI, (100 − y)/(100 − x), es la velocidad media que el colchón puede soportar hasta el final.'),
    ('¿En qué zona está el informe del día 34?',
     ['amarilla', 'verde', 'roja', 'no se puede saber sin el valor ganado'],
     'Límites 56,7 % y 76,7 % para un 77,8 % de cadena; consumo 66,7 %.'),
]
TITULO_TEST = 'Cadena crítica — tendencia del gráfico de fiebre (cadena de 36 días, CP de 9)'
import os, re


def a_gift(preguntas_test, titulo, semilla=1):
    """Preguntas en formato GIFT (Moodle): caracteres especiales escapados y opciones barajadas."""
    esc = lambda s: re.sub(r'([~=#{}:])', r'\\\1', str(s))
    r = np.random.default_rng(semilla)
    lineas = [f'$CATEGORY: {titulo}', '']
    for i, (q, ops, fb) in enumerate(preguntas_test, 1):
        cuerpo = ' '.join(('=' if j == 0 else '~') + esc(ops[j]) for j in r.permutation(len(ops)))
        lineas.append(f'::{esc(titulo)} P{i}::{esc(q)} {{{cuerpo} ####{esc(fb)}}}')
        lineas.append('')
    return '\n'.join(lineas)


gift = a_gift(preguntas_test, TITULO_TEST)
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_cadena_critica.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)
