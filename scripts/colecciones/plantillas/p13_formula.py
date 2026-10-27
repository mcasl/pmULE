"""Colección 13 — La fórmula del tiempo de compleción τ = ‖R·t‖∞ (arXiv 2410.10252):
matriz de rutas, norma infinito, gradiente, SVD de la matriz de rutas, núcleo de la
incidencia e instancias de referencia (Patterson)."""
import numpy as np
import pandas as pd

import networkx as nx
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX, ESTADO)

META = dict(slug="13_formula_compleccion", prefijo="Formula",
            titulo="La fórmula del tiempo de compleción τ = ‖R·t‖∞",
            descripcion="Matriz de rutas R, τ como norma infinito de R·t, gradiente ∂τ/∂t, descomposición en "
                        "valores singulares de R, núcleo de la matriz de incidencia e instancias de referencia.")

PAT1 = """14      3
2       1       2
0       0       0       0       3       2       3       4
6       1       0       0       2       9       10
4       0       0       0       3       5       6       7
3       0       0       0       2       8       11
1       0       0       0       1       10
6       1       0       1       1       12
2       1       0       0       2       8       11
1       0       0       0       1       13
4       0       1       1       1       14
3       0       0       1       1       12
2       0       0       1       1       12
3       0       1       0       1       13
5       0       0       0       1       14
0       0       0       0       0
"""

PAT20 = """22      3
10      10      10
0       0       0       0       3       2       3       4
3       3       5       2       2       5       6
2       5       4       3       2       5       8
9       5       2       2       2       6       7
7       4       1       4       2       9       10
7       5       5       4       2       11      12
3       3       5       2       2       9       10
2       2       4       4       2       9       10
1       3       2       2       1       13
2       3       2       4       1       14
4       3       3       2       2       13      15
2       4       1       4       2       13      16
2       1       4       4       1       17
3       2       2       2       1       19
5       5       5       4       1       18
1       1       5       4       1       18
1       4       5       4       1       21
8       3       2       3       1       20
6       5       3       3       1       20
4       2       4       6       1       22
2       1       6       2       1       22
0       0       0       0       0
"""

PREP = ("datos_proyecto = datos_enunciado.copy()\n"
        "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
        "proyecto = prelaciones.create_project()\n"
        "R = proyecto.path_matrix(dummies=False)\n"
        "t = datos_proyecto['duration'][R.columns]")


# Celda auxiliar (se define en el cuaderno): el producto R·t fila a fila, con la suma de cada ruta en
# orden de ejecución, y la norma infinito como máximo de esas sumas.
_AUX_FORMULA = r'''
def _num(x, dec=0):
    x = float(x)
    return (f'{x:.{dec}f}' if dec else f'{x:g}').replace('-', '−')


def producto_Rt(proyecto, R, t, base, decimales=0):
    """Cada fila de R por el vector t: (R·t)_r = Σ_i R_ri·t_i, con los términos no nulos (actividades de la
    ruta, en orden de ejecución) y el valor; marca las filas que dan el máximo."""
    cal = proyecto.calendar(base)
    orden = {a: k for k, a in enumerate(ordena_ruta(list(R.columns), cal))}
    filas = {}
    for r in R.index:
        acts = sorted([a for a in R.columns if R.loc[r, a] == 1], key=lambda a: orden.get(a, 0))
        total = float(sum(float(t[a]) for a in acts))
        filas[r] = {'actividades': ' – '.join(acts),
                    'fila de R · t = suma': ' + '.join(f'{a}({_num(t[a], decimales)})' for a in acts) + f' = {_num(total, decimales)}',
                    'R·t': total}
    tabla = pd.DataFrame(filas).T
    tabla['R·t'] = tabla['R·t'].astype(float)
    tabla['máximo'] = np.where(np.isclose(tabla['R·t'], tabla['R·t'].max()), '← τ', '')
    return tabla


def explica_tau(tabla, unidad='', decimales=0, nombre='τ'):
    """«τ = ‖R·t‖∞ = máx(19, 20, 17) = 20, que da la ruta Route_2 (A – C – E)»."""
    v = tabla['R·t']
    T = float(v.max())
    rutas = [f"{r} ({tabla.loc[r, 'actividades']})" for r in v.index if np.isclose(v[r], T)]
    u = f' {unidad}' if unidad else ''
    quien = (f'que corresponde a la ruta {rutas[0]}' if len(rutas) == 1
             else f'alcanzado a la vez por {len(rutas)} rutas: {lista_y(rutas)}')
    return (f"{nombre} = ‖R·t‖∞ = máx({', '.join(_num(x, decimales) for x in v)}) = **{_num(T, decimales)}{u}**, {quien}")
'''


def _tau(rng):
    """τ = ‖R·t‖∞: la fórmula, su equivalencia con el cálculo de tiempos y su uso con varios vectores t."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 4 <= i["n_rutas"] <= 7 and 1 <= i["n_dummies"] <= 3 and i["n_criticas"] == 1 and (i["D"] >= 2).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    k = list(D.index)[int(rng.integers(0, len(D)))]
    celdas = [md("# Enunciado"),
              md("La duración de un proyecto puede escribirse como una fórmula cerrada: si $R$ es la matriz de "
                 "rutas (una fila por cada ruta del nodo inicial al final, una columna por actividad, con un 1 "
                 "donde la actividad pertenece a la ruta) y $t$ el vector de duraciones, entonces\n\n"
                 "$$\\tau = \\lVert R\\,t \\rVert_\\infty = \\max_r \\sum_i R_{ri}\\,t_i .$$\n\n"
                 "Considere el proyecto cuyo cuadro de prelaciones y duraciones (días) se adjunta.")]
    celdas.append(preguntas([
        "Dibuje el grafo PERT, enumere sus rutas y construya la matriz de rutas R. ¿Qué dimensiones tiene?",
        "Calcule el vector R·t y su norma infinito. Compruebe que coincide con el tiempo más temprano del "
        "último nodo obtenido por el método de Zaderenko.",
        "¿Por qué la norma infinito y no otra? ¿Qué daría ‖R·t‖₁? ¿Y qué significaría el mínimo de R·t?",
        f"Escriba τ en función de t_{k} (la duración de {k}) dejando fijas las demás, y dibuje esa función. "
        "¿Es lineal? ¿Dónde cambia de pendiente?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    celdas.append(code(PREP))
    celdas.append(code(_AUX_FORMULA))
    celdas += apartado(1)
    celdas += fig_grafo(durations="False", cap="Grafo PERT", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de rutas R"\n#| label: tbl-R\n'
                       "rutas = proyecto.paths(dummies=False)\n"
                       "display(pd.Series({r: ' - '.join(a) for r, a in rutas.items()}).to_frame('actividades'))\n"
                       f"proyecto.display_rutas('{ESTADO['grafo']}')"))
    celdas.append(code("Markdown(f'**Respuesta:** el grafo tiene {R.shape[0]} rutas del nodo inicial al final (tabla); R tiene {R.shape[0]} filas (una por ruta) '\n"
                       "         f'y {R.shape[1]} columnas (una por actividad real; las ficticias duran 0 y no se incluyen): dimensiones {R.shape[0]} × {R.shape[1]}.')"))
    celdas += apartado(2)
    celdas.append(code("Rt = R @ t\n"
                       "tau = float(Rt.max())\n"
                       "from pmule import display_producto_rutas\n"
                       "display_producto_rutas(R, t, Rt)"))
    celdas.append(md("Cada componente de $R\\,t$ es el producto de una fila de $R$ por $t$: como la fila tiene un 1 en "
                     "las actividades de la ruta y un 0 en las demás, el producto es la suma de las duraciones de esa "
                     "ruta. La norma infinito es el mayor de esos valores."))
    celdas.append(code("producto = producto_Rt(proyecto, R, t, datos_proyecto['duration'])\n"
                       "display(producto)\n"
                       "Markdown(explica_tau(producto, 'días') + '.')"))
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(code("T_zad = proyecto.duration(datos_proyecto['duration'])\n"
                       "Markdown(f'**Respuesta:** R·t = ({\", \".join(_num(x) for x in Rt)}) y ‖R·t‖∞ = {tau:.0f} días; el tiempo más temprano del último nodo por Zaderenko es {T_zad} días '\n"
                       "         f'y la librería (`proyecto.tau`) da {proyecto.tau(t.to_dict()):.0f}. Las tres coinciden: el cálculo de tiempos hacia delante es una forma de calcular el '\n"
                       "         'máximo de las rutas sin enumerarlas.')"))
    celdas += apartado(3)
    celdas.append(code("Markdown(f'''**Respuesta:** ‖R·t‖₁ = {' + '.join(_num(x) for x in Rt)} = {float(Rt.abs().sum()):.0f} sumaría las duraciones de todas las rutas, contando varias veces cada actividad compartida: no tiene sentido como plazo. "
                       "El mínimo de R·t, {float(Rt.min()):.0f} ({Rt.idxmin()}), es la ruta más corta: el tiempo que *como mínimo* ocupa cualquier cadena del proyecto, pero el proyecto no termina hasta que acaba la última ruta. "
                       "El plazo es el máximo porque todas las rutas deben completarse; de ahí la norma infinito.''')"))
    celdas += apartado(4)
    celdas.append(code(f"k = '{k}'\n"
                       "valores = np.arange(0, int(t[k]) + 8)\n"
                       "taus = []\n"
                       "for v in valores:\n"
                       "    t2 = t.copy(); t2[k] = v\n"
                       "    taus.append(float((R @ t2).max()))\n"
                       "pendiente = np.diff(taus)\n"
                       "codo = int(valores[np.argmax(pendiente > 0)]) if (pendiente > 0).any() else None\n"
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(6, 3.5))\n"
                       "ax.plot(valores, taus, marker='o'); ax.axvline(int(t[k]), color='grey', linestyle='--')\n"
                       "ax.set_xlabel(f'duración de {k}'); ax.set_ylabel('τ'); plt.show()"))
    celdas.append(code("con_k = [r for r in R.index if R.loc[r, k] == 1]\n"
                       "sin_k = [r for r in R.index if R.loc[r, k] == 0]\n"
                       "resto = (Rt[con_k] - t[k])                     # lo que suman las demás actividades de cada ruta con k\n"
                       "c1 = float(resto.max())\n"
                       "c0 = float(Rt[sin_k].max()) if sin_k else None\n"
                       "if c0 is None:\n"
                       "    formula = f'τ(t_{k}) = {_num(c1)} + t_{k}: todas las rutas pasan por {k}, así que τ es lineal con pendiente 1 para cualquier duración'\n"
                       "elif c0 - c1 <= 0:\n"
                       "    formula = (f'τ(t_{k}) = máx({_num(c0)}, {_num(c1)} + t_{k}) = {_num(c1)} + t_{k} para cualquier t_{k} ≥ 0, porque la ruta más larga con {k} ({resto.idxmax()}) '\n"
                       "               f'ya supera sin {k} a la más larga sin {k} ({Rt[sin_k].idxmax()}, {_num(c0)}): pendiente 1 en todo el dominio, sin cambio de pendiente')\n"
                       "else:\n"
                       "    codo_f = c0 - c1\n"
                       "    formula = (f'τ(t_{k}) = máx({_num(c0)}, {_num(c1)} + t_{k}), donde {_num(c0)} es la ruta más larga sin {k} ({Rt[sin_k].idxmax()}) y {_num(c1)} lo que suman las demás '\n"
                       "               f'actividades de la ruta más larga con {k} ({resto.idxmax()}). Es lineal a trozos y convexa: pendiente 0 mientras t_{k} < {_num(c0)} − {_num(c1)} = {_num(c0 - c1)} '\n"
                       "               f'(manda la ruta sin {k}) y pendiente 1 a partir de ahí; el cambio de pendiente está en t_{k} = {_num(codo_f)}')\n"
                       "actual = c1 + t[k]\n"
                       "if c0 is None or actual > c0:\n"
                       "    sens = f'con t_{k} = {int(t[k])}, {k} está en la ruta crítica: alargarla o acortarla un día cambia τ un día (pendiente 1)'\n"
                       "elif actual < c0:\n"
                       "    sens = f'con t_{k} = {int(t[k])}, {k} no es crítica: puede alargarse hasta {_num(c0 - c1)} sin cambiar τ (pendiente 0)'\n"
                       "else:\n"
                       "    sens = f'con t_{k} = {int(t[k])} el valor actual está justo en el codo: alargarla aumenta τ, acortarla no lo reduce'\n"
                       "lineal = c0 is None or c0 - c1 <= 0\n"
                       "previo = 'En el rango de duraciones posibles es lineal; ' if lineal else 'No es lineal (es el máximo de funciones lineales); '\n"
                       "Markdown(f'**Respuesta:** {formula}. {previo}{sens}.')"))
    return celdas, f"{n} actividades, {info['n_rutas']} rutas; τ como norma infinito"


def _gradiente(rng):
    """∂τ/∂t con varias rutas críticas: subgradiente, interpretación y comprobación por diferencias."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 9 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 2,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md("Con $\\tau = \\lVert R\\,t \\rVert_\\infty$, la sensibilidad de la duración del proyecto a cada "
                 "actividad es el gradiente $\\partial\\tau/\\partial t_i$. Considere el proyecto cuyo cuadro se "
                 "adjunta, que tiene más de una ruta crítica.")]
    celdas.append(preguntas([
        "Construya R, calcule R·t y determine las rutas críticas.",
        "Calcule ∂τ/∂t_i para cada actividad como la media de los vectores indicadores de las rutas críticas "
        "(lo que da `tau_sensitivity`). ¿Qué valores aparecen y qué significan 1, 0 y los intermedios?",
        "Compruebe por diferencias finitas, para una actividad de cada clase, qué ocurre con τ al aumentar y al "
        "disminuir su duración en un día. ¿Por qué la derivada «no existe» en las actividades intermedias?",
        "Demuestre que la suma de las sensibilidades es la longitud (en número de actividades) media de las "
        "rutas críticas y compruébelo."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(PREP))
    celdas.append(code(_AUX_FORMULA))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y rutas críticas", label="fig-pert")
    celdas.append(code("Rt = R @ t\n"
                       "tau = float(Rt.max())\n"
                       "criticas = Rt.index[np.isclose(Rt, tau)].tolist()\n"
                       "producto = producto_Rt(proyecto, R, t, datos_proyecto['duration'])\n"
                       "display(producto)\n"
                       "Markdown('**Respuesta:** ' + explica_tau(producto) + '. Esas son las rutas críticas.')"))
    celdas += apartado(2)
    celdas.append(code("indicadores = R.loc[criticas]\n"
                       "gradiente = indicadores.mean()\n"
                       "display(indicadores)\n"
                       "pd.DataFrame({'∂τ/∂t (media de indicadores)': gradiente, 'tau_sensitivity': proyecto.tau_sensitivity(t.to_dict()).reindex(R.columns)}).T"))
    celdas.append(code("clases = {v: gradiente.index[np.isclose(gradiente, v)].tolist() for v in sorted(gradiente.unique(), reverse=True)}\n"
                       "k_c = len(criticas)\n"
                       "ejemplos = '; '.join(f\"∂τ/∂t_{acts[0]} = ({' + '.join(str(int(R.loc[r, acts[0]])) for r in criticas)})/{k_c} = {_num(v, 2)}\" for v, acts in clases.items())\n"
                       "clases_txt = '; '.join(f'{_num(v, 2)}: {lista_y(acts)}' for v, acts in clases.items())\n"
                       "Markdown(f'''**Respuesta:** cada valor es la media de la columna de la actividad en las filas críticas, por ejemplo {ejemplos}. Aparecen los valores {clases_txt}. "
                       "Valor 1: la actividad está en todas las rutas críticas (alargarla o acortarla cambia τ uno a uno). Valor 0: no está en ninguna (τ no cambia, al menos para cambios pequeños). "
                       "Valor intermedio: está en algunas de las rutas críticas pero no en todas; alargarla sí alarga el proyecto, pero acortarla no lo acorta, porque las otras rutas críticas "
                       "siguen midiendo lo mismo.''')"))
    celdas += apartado(3)
    celdas.append(code("def tau_con(act, delta):\n"
                       "    t2 = t.copy(); t2[act] += delta\n"
                       "    return float((R @ t2).max())\n"
                       "clases = {v: gradiente.index[np.isclose(gradiente, v)].tolist() for v in sorted(gradiente.unique())}\n"
                       "filas = {}\n"
                       "for v, acts in clases.items():\n"
                       "    a = acts[0]\n"
                       "    filas[a] = {'∂τ/∂t': v, 'τ(t+1) − τ': tau_con(a, 1) - tau, 'τ(t−1) − τ': tau_con(a, -1) - tau}\n"
                       "pd.DataFrame(filas).T"))
    celdas.append(code("frases = []\n"
                       "for a, f in pd.DataFrame(filas).T.iterrows():\n"
                       "    frases.append(f\"{a} (∂τ/∂t = {_num(f['∂τ/∂t'], 2)}): al alargarla τ cambia {_num(f['τ(t+1) − τ'])} y al acortarla {_num(f['τ(t−1) − τ'])}\")\n"
                       "Markdown(f'''**Respuesta:** {'; '.join(frases)}. Para las actividades con valor intermedio la pendiente por la derecha (+1) y por la "
                       "izquierda (0) son distintas: τ es un máximo de funciones lineales y donde dos rutas empatan tiene una esquina, así que no hay derivada. La media de los "
                       "indicadores es un *subgradiente*: el valor que reparte la sensibilidad entre las rutas empatadas.''')"))
    celdas += apartado(4)
    celdas.append(code("suma = float(gradiente.sum())\n"
                       "longitudes = indicadores.sum(axis=1)\n"
                       "longitud_media = float(longitudes.mean())\n"
                       "Markdown(f'''**Respuesta:** Σ_i ∂τ/∂t_i = Σ_i (1/k) Σ_{{r crítica}} R_ri = (1/k) Σ_{{r crítica}} Σ_i R_ri = media del número de actividades de las rutas críticas "
                       "(se intercambia el orden de las sumas). Aquí Σ = {' + '.join(_num(g, 2) for g in gradiente)} = {suma:.2f}, y las {len(criticas)} rutas críticas tienen "
                       "{lista_y([str(int(x)) for x in longitudes])} actividades: media ({' + '.join(str(int(x)) for x in longitudes)})/{len(criticas)} = {longitud_media:.2f}. Coinciden.''')"))
    return celdas, f"{n} actividades, {info['n_criticas']} rutas críticas; gradiente con empates"


def _svd(rng):
    """SVD de R: σ₁/σ₂, u₁ y v₁, aproximación de rango 1 de τ y precisión del ranking frente al conjunto crítico."""
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 6 <= i["n_rutas"] <= 12 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md("La matriz de rutas $R$ describe la estructura del proyecto sin usar duraciones. Su descomposición "
                 "en valores singulares $R = U\\,\\Sigma\\,V^T$ permite resumir esa estructura: $u_1$ pondera las rutas "
                 "y $v_1$ las actividades, y el cociente $\\sigma_1/\\sigma_2$ indica cuánto domina esa primera "
                 "dirección. Considere el proyecto cuyo cuadro se adjunta.")]
    celdas.append(preguntas([
        "Construya R y calcule su SVD. Dé los valores singulares y el cociente σ₁/σ₂.",
        "Ordene las actividades por |v₁| y las rutas por |u₁|. ¿Qué tienen en común las actividades mejor situadas?",
        "Con las duraciones, calcule τ exacto y la aproximación de rango 1, τ₁ = σ₁ · máx(u₁) · (v₁ᵀ t) "
        "(R ≈ σ₁ u₁ v₁ᵀ). ¿Qué error relativo comete?",
        "Compare el ranking por |v₁| con el conjunto de actividades críticas: ¿es crítica la actividad mejor "
        "clasificada? ¿Qué fracción de las k mejor clasificadas son críticas (k = número de críticas)? "
        "¿Por qué la SVD, que no usa duraciones, puede acertar?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    celdas.append(code(PREP))
    celdas.append(code(_AUX_FORMULA))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Matriz de rutas R"\n#| label: tbl-R\n' f"proyecto.display_rutas('{ESTADO['grafo']}')"))
    celdas.append(code("U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)\n"
                       "u1, v1 = U[:, 0], Vt[0, :]\n"
                       "if u1.sum() < 0:\n"
                       "    u1, v1 = -u1, -v1          # R es no negativa: u1 y v1 pueden tomarse no negativos\n"
                       "display(Math(r'\\Sigma = ' + bmatrix(np.diag(S), dec=2)))\n"
                       "Markdown(f'**Respuesta:** valores singulares {\", \".join(f\"{x:.3f}\" for x in S)}; σ₁/σ₂ = {S[0]:.3f}/{S[1]:.3f} = **{S[0]/S[1]:.2f}**.')"))
    celdas += apartado(2)
    celdas.append(code("peso_actividades = pd.Series(v1, index=R.columns).sort_values(ascending=False)\n"
                       "peso_rutas = pd.Series(u1, index=R.index).sort_values(ascending=False)\n"
                       "apariciones = R.sum()\n"
                       "display(pd.DataFrame({'v₁': peso_actividades.round(3), 'rutas en las que aparece': apariciones[peso_actividades.index]}).T)\n"
                       "pd.DataFrame({'u₁': peso_rutas.round(3), 'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in peso_rutas.index], 'nº actividades': R.loc[peso_rutas.index].sum(axis=1)})"))
    celdas.append(code("en_todas = [a for a in R.columns if apariciones[a] == R.shape[0]]\n"
                       "top = list(peso_actividades.index[:3])\n"
                       "t_todas = (f'{lista_y(en_todas)} {\"aparece\" if len(en_todas) == 1 else \"aparecen\"} en las {R.shape[0]} rutas y '\n"
                       "           f'{\"encabeza\" if len(en_todas) == 1 else \"encabezan\"} la lista. ' if en_todas else 'Ninguna actividad está en todas las rutas. ')\n"
                       "Markdown(f'''**Respuesta:** las mejor situadas son {lista_y([f\"{a} (v₁ = {peso_actividades[a]:.3f}, en {int(apariciones[a])} rutas)\" for a in top])}: "
                       "lo que tienen en común es que aparecen en muchas rutas, sobre todo en las rutas largas y con muchas actividades compartidas (las de mayor u₁: "
                       "{peso_rutas.index[0]}, con {int(R.loc[peso_rutas.index[0]].sum())} actividades). {t_todas}v₁ es una medida de centralidad estructural: no depende de las duraciones.''')"))
    celdas += apartado(3)
    celdas.append(md("τ exacto: cada fila de $R$ por $t$ es la duración de una ruta, y τ es la mayor."))
    celdas.append(code("producto = producto_Rt(proyecto, R, t, datos_proyecto['duration'])\n"
                       "display(producto)\n"
                       "tau = float((R @ t).max())\n"
                       "tau_1 = float(S[0] * u1.max() * (v1 @ t.values))\n"
                       "v1t = ' + '.join(f'{x:.3f}·{int(d)}' for x, d in zip(v1, t.values))\n"
                       "Markdown(f'''**Respuesta:** {explica_tau(producto)}. Aproximación de rango 1: v₁ᵀt = {v1t} = {v1 @ t.values:.3f}, y "
                       "τ₁ = σ₁ · máx(u₁) · (v₁ᵀ t) = {S[0]:.3f} × {u1.max():.3f} × {v1 @ t.values:.3f} = **{tau_1:.2f}**. Error relativo = |τ₁ − τ|/τ = "
                       "|{tau_1:.2f} − {tau:.0f}|/{tau:.0f} = **{abs(tau_1 - tau) / tau * 100:.1f} %**.''')"))
    celdas.append(md("La aproximación sustituye cada fila de $R$ por un múltiplo de $v_1$: funciona bien cuando "
                     "las rutas se parecen entre sí (σ₁ ≫ σ₂) y peor cuando el proyecto tiene varias ramas muy "
                     "distintas. El τ exacto es siempre la norma infinito de $R\\,t$; la SVD es un resumen."))
    celdas += apartado(4)
    celdas.append(code("criticas = set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)\n"
                       "k = len(criticas)\n"
                       "top_k = list(peso_actividades.index[:k])\n"
                       "precision = len(set(top_k) & criticas) / k\n"
                       "pd.DataFrame({'ranking por v₁': peso_actividades.index, 'v₁': peso_actividades.round(3).values,\n"
                       "              'crítica': ['sí' if a in criticas else 'no' for a in peso_actividades.index]}).set_index('ranking por v₁').T"))
    celdas.append(code("corte = peso_actividades.iloc[k - 1]\n"
                       "empate_corte = k < len(peso_actividades) and np.isclose(peso_actividades.iloc[k], corte)\n"
                       "nota = (f' Ojo: en el puesto {k} hay un empate de v₁ ({corte:.3f}) con actividades que quedan fuera, así que la precisión depende de cómo se ordenen los empatados.'\n"
                       "        if empate_corte else '')\n"
                       "Markdown(f'''**Respuesta:** la actividad mejor clasificada, {peso_actividades.index[0]}, {\"es\" if peso_actividades.index[0] in criticas else \"no es\"} crítica. "
                       "Hay k = {k} actividades críticas ({lista_y(ordena_ruta(sorted(criticas), proyecto.calendar(datos_proyecto['duration'])))}); entre las {k} primeras del ranking "
                       "({lista_y(top_k)}) hay {len(set(top_k) & criticas)} críticas: precisión {len(set(top_k) & criticas)}/{k} = **{precision:.2f}**.{nota} "
                       "La SVD sólo ve la estructura: una actividad por la que pasan muchas rutas tiene muchas ocasiones de estar en la más larga, "
                       "sea cual sea el vector de duraciones; por eso acierta a menudo, pero no siempre, porque la ruta crítica concreta la fijan las duraciones.''')"))
    return celdas, f"{n} actividades, {info['n_rutas']} rutas; SVD de R y ranking frente al conjunto crítico"


def _nucleo(rng):
    """Matriz de incidencia, núcleo y perturbaciones de las duraciones que no cambian τ."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 4 <= i["n_rutas"] <= 7 and 1 <= i["n_dummies"] <= 3 and (i["D"] >= 3).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    g = info["proyecto"]
    N = g.nullspace()
    y = pd.Series(rng.integers(-2, 3, size=N.shape[1]), index=N.columns)
    while (y == 0).all():
        y = pd.Series(rng.integers(-2, 3, size=N.shape[1]), index=N.columns)
    celdas = [md("# Enunciado"),
              md("La matriz de incidencia $H$ del grafo PERT tiene una fila por arco (actividades y ficticias) y "
                 "una columna por nodo, con −1 en el nodo origen del arco y +1 en el destino. Cada ruta es una "
                 "cadena de arcos del nodo inicial al final. Considere el proyecto cuyo cuadro se adjunta.")]
    celdas.append(preguntas([
        "Dibuje el grafo PERT numerado y escriba la matriz de incidencia H (incluyendo las ficticias).",
        "Construya la matriz de rutas R *con* las ficticias (duración 0) y compruebe que R·H = 0 salvo en las "
        "columnas del nodo inicial y del final. ¿Por qué?",
        "Elija un vector y con un valor entero por nodo interior (0 en el inicial y el final) y defina la "
        "perturbación δ = H·y de las duraciones de los arcos. Compruebe que todas las rutas cambian de duración "
        "en la misma cantidad y que τ no cambia. Interprételo: ¿qué se ha hecho con los «tiempos de los nodos»?",
        "¿Qué dimensión tiene el espacio de perturbaciones que dejan invariantes todas las duraciones de ruta? "
        "¿Qué significa para la identificación de duraciones a partir de observaciones de rutas?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas.append(code(_AUX_FORMULA))
    celdas += apartado(1)
    celdas += fig_grafo(durations="False", cap="Grafo PERT numerado", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de incidencia H (arcos × nodos)"\n#| label: tbl-H\n'
                       "H = proyecto.incidence_matrix()\n"
                       "H"))
    celdas.append(code("Markdown(f'**Respuesta:** H tiene {H.shape[0]} filas (las {len(proyecto.actual_activities)} actividades y {H.shape[0] - len(proyecto.actual_activities)} '\n"
                       "         f'ficticias) y {H.shape[1]} columnas (nodos); cada fila tiene un −1 en el nodo donde empieza el arco y un +1 en el nodo donde termina.')"))
    celdas += apartado(2)
    celdas.append(md("Matriz de rutas con las ficticias (una columna por arco, en el mismo orden que las filas de H):"))
    celdas.append(code("R = proyecto.path_matrix(dummies=True)[H.index]\n"
                       "display(R)\n"
                       "RH = R @ H\n"
                       "display(RH)\n"
                       "interiores = H.columns[1:-1]\n"
                       "r0 = R.index[0]\n"
                       "arcos_r0 = [a for a in R.columns if R.loc[r0, a] == 1]\n"
                       "nodo = interiores[0]\n"
                       "ejemplo = ' + '.join(f'({_num(H.loc[a, nodo])})' for a in arcos_r0 if H.loc[a, nodo] != 0)\n"
                       "cero = bool((RH[interiores] == 0).all().all())\n"
                       "Markdown(f'''Ejemplo: la fila {r0} de R tiene un 1 en los arcos {lista_y(arcos_r0)}; multiplicada por la columna del nodo {nodo} de H solo suman los arcos de la ruta que tocan ese nodo: "
                       "{ejemplo or '0 (la ruta no pasa por el nodo)'} = {_num(RH.loc[r0, nodo])}.\\n\\n**Respuesta:** R·H es {'cero' if cero else 'distinto de cero'} en todas las columnas interiores; en la del nodo inicial vale "
                       "{_num(RH.iloc[0, 0])} y en la del final {_num(RH.iloc[0, -1])} para todas las rutas.''')"))
    celdas.append(md("Una ruta entra y sale de cada nodo interior exactamente una vez, así que la suma de sus "
                     "entradas de incidencia es +1 − 1 = 0; sólo sale del inicial (−1) y sólo entra en el final (+1)."))
    celdas += apartado(3)
    celdas.append(code(literal_dict({str(k): int(v) for k, v in y.items()}, "y_interior") + "\n"
                       "y = pd.Series(0, index=H.columns, dtype=int)\n"
                       "for nodo, valor in y_interior.items():\n"
                       "    y[nodo] = valor\n"
                       "display(y.to_frame('y (0 en el nodo inicial y en el final)').T)\n"
                       "delta = H @ y\n"
                       "t = pd.Series({a: (datos_proyecto.loc[a, 'duration'] if not a.startswith('@') else 0) for a in H.index})\n"
                       "t_perturbado = t + delta\n"
                       "origen = {a: H.columns[H.loc[a] == -1][0] for a in H.index}\n"
                       "destino = {a: H.columns[H.loc[a] == 1][0] for a in H.index}\n"
                       "pd.DataFrame({'t': t, 'δ = y(destino) − y(origen)': [f'{_num(y[destino[a]])} − ({_num(y[origen[a]])}) = {_num(delta[a])}' for a in H.index],\n"
                       "              't + δ': t_perturbado}, index=H.index).T"))
    celdas.append(code("Rt, Rt2 = R @ t, R @ t_perturbado\n"
                       "def _suma(v, r):\n"
                       "    acts = sorted([a for a in R.columns if R.loc[r, a] == 1], key=lambda a: list(H.columns).index(origen[a]))\n"
                       "    return ' + '.join(f'{a}({_num(v[a])})' for a in acts) + f' = {_num(sum(v[a] for a in acts))}'\n"
                       "display(pd.DataFrame({'R·t = suma': [_suma(t, r) for r in R.index], 'R·(t+δ) = suma': [_suma(t_perturbado, r) for r in R.index],\n"
                       "                      'diferencia': (Rt2 - Rt).astype(int)}, index=R.index))\n"
                       "Markdown(f'''**Respuesta:** τ = máx({', '.join(_num(x) for x in Rt)}) = {float(Rt.max()):.0f} ({Rt.idxmax()}) antes y máx({', '.join(_num(x) for x in Rt2)}) = "
                       "{float(Rt2.max()):.0f} después: las duraciones de todas las rutas cambian en {int((Rt2 - Rt).iloc[0])} (= y_final − y_inicial = 0), así que τ es invariante. "
                       "Con los «tiempos de los nodos» se ha hecho lo que explica el párrafo siguiente: desplazar el reloj de cada nodo interior.''')"))
    celdas.append(md("δ = H·y añade a cada arco la diferencia de «potencial» entre su destino y su origen. Es lo "
                     "mismo que mover el reloj de cada nodo: si al nodo j se le suma y_j a su tiempo, los arcos "
                     "que llegan a j parecen y_j más largos y los que salen y_j más cortos, y a lo largo de cualquier "
                     "ruta cerrada entre inicio y final todo se cancela. (Algunas duraciones perturbadas pueden salir "
                     "negativas: es un ejercicio algebraico, no un calendario.)"))
    celdas += apartado(4)
    celdas.append(code("N = proyecto.nullspace()\n"
                       "rango = np.linalg.matrix_rank(N.values)\n"
                       "rango_R = np.linalg.matrix_rank(R.values.astype(float))\n"
                       "nucleo_R = len(t) - rango_R\n"
                       "Markdown(f'''**Respuesta:** las columnas interiores de H generan un espacio de dimensión {rango} (tantas como nodos interiores, {N.shape[1]}) contenido en el núcleo de R. "
                       "Comprobación: R tiene {len(t)} columnas (arcos) y rango {rango_R}, así que su núcleo tiene dimensión {len(t)} − {rango_R} = **{nucleo_R}**"
                       "{' = número de nodos interiores: esas son todas las perturbaciones posibles' if nucleo_R == rango else ''}. "
                       "Hay por tanto {nucleo_R} direcciones independientes en las que pueden moverse las duraciones de los {len(t)} arcos sin que ninguna ruta cambie. "
                       "Consecuencia: observando sólo duraciones de rutas no pueden identificarse las duraciones de los arcos; sólo se identifican salvo esas {nucleo_R} direcciones.''')"))
    return celdas, f"{n} actividades, {info['n_nodos']} nodos; incidencia y núcleo"


def _patterson(rng):
    """Una instancia de referencia (Patterson): lectura, R, τ frente a CPM y ranking SVD."""
    cual = rng.choice(["pat1", "pat20"])
    texto, n_act = (PAT1, 14) if cual == "pat1" else (PAT20, 22)
    celdas = [md("# Enunciado"),
              md(f"El fichero `{cual}.rcp` del conjunto de instancias de Patterson (formato RCP: primera línea, "
                 "número de actividades y de recursos; segunda, disponibilidad de cada recurso; después, una línea "
                 "por actividad con su duración, sus necesidades de recursos, el número de sucesoras y sus "
                 "sucesoras) describe un proyecto con actividades ficticias de inicio y fin de duración cero:")]
    celdas.append(code('#| tbl-cap: "Contenido del fichero"\n#| label: tbl-fichero\n'
                       f"contenido_rcp = '''{texto}'''\n"
                       "print(contenido_rcp)"))
    celdas.append(preguntas([
        "Lea el fichero con `read_rcp` (escribiéndolo antes en disco), construya el cuadro de prelaciones y el "
        "proyecto. ¿Cuántas actividades reales, ficticias, nodos y rutas tiene?",
        "Calcule τ = ‖R·t‖∞ y compárelo con la duración obtenida por el cálculo de tiempos (CPM). Indique las "
        "actividades críticas.",
        "Calcule la SVD de R y el cociente σ₁/σ₂. Ordene las actividades por |v₁| y mida la precisión del ranking: "
        "fracción de las k primeras (k = número de críticas) que son críticas.",
        "Compare el tamaño de R con el del grafo. ¿Cómo crece el número de rutas con el tamaño del proyecto y qué "
        "implica para proyectos de cientos de actividades?"]))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(code("from pmule import read_rcp\n"
                       f"with open('{cual}.rcp', 'w') as f:\n"
                       "    f.write(contenido_rcp)\n"
                       f"datos_rcp = read_rcp('{cual}.rcp')\n"
                       "datos_proyecto = pd.DataFrame({'predecessor': pd.Series({k: (v or '---') for k, v in datos_rcp['predecessors'].items()}),\n"
                       "                               'duration': datos_rcp['duration']['duration']})\n"
                       "datos_proyecto.index.name = 'activity'\n"
                       "datos_proyecto.T"))
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "t = datos_proyecto['duration'][R.columns]\n"
                       "n_fict = len(proyecto.dummies)\n"
                       "Markdown(f'''**Respuesta:** el cuadro tiene {len(proyecto)} actividades: {len(proyecto) - 2} reales y las 2 ficticias de inicio y fin del formato RCP (duración 0, "
                       "{datos_proyecto.index[0]} y {datos_proyecto.index[-1]}). El grafo PERT añade {n_fict} {'ficticia' if n_fict == 1 else 'ficticias'} para representar las prelaciones, "
                       "tiene {len(proyecto.nodes)} nodos y {R.shape[0]} rutas del nodo inicial al final.''')"))
    celdas.append(code(_AUX_FORMULA))
    celdas.append(code('#| fig-cap: "Grafo PERT de la instancia"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=0.6, nodesep=0.3)"))
    celdas += apartado(2)
    celdas.append(code("Rt = R @ t\n"
                       "tau = float(Rt.max())\n"
                       "cpm = proyecto.duration(datos_proyecto['duration'])\n"
                       "criticas = sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts))\n"
                       "producto = producto_Rt(proyecto, R, t, datos_proyecto['duration'])\n"
                       "display(producto)\n"
                       "Markdown(f'''**Respuesta:** {explica_tau(producto)}. El cálculo de tiempos (CPM) da {cpm}: {'coinciden' if np.isclose(tau, cpm) else 'no coinciden'}. "
                       "Actividades críticas: {lista_y(ordena_ruta(criticas, proyecto.calendar(datos_proyecto['duration'])))} ({datos_proyecto.index[0]} y {datos_proyecto.index[-1]} son las ficticias del formato).''')"))
    celdas += apartado(3)
    celdas.append(code("U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)\n"
                       "v1 = np.abs(Vt[0, :])\n"
                       "ranking = pd.Series(v1, index=R.columns).sort_values(ascending=False)\n"
                       "k = len(criticas)\n"
                       "precision = len(set(ranking.index[:k]) & set(criticas)) / k\n"
                       "display(pd.DataFrame({'v₁': ranking.round(3), 'crítica': ['sí' if a in criticas else 'no' for a in ranking.index]}).T)\n"
                       "top_k = list(ranking.index[:k])\n"
                       "aciertos = len(set(top_k) & set(criticas))\n"
                       "empate_corte = k < len(ranking) and np.isclose(ranking.iloc[k], ranking.iloc[k - 1])\n"
                       "nota = f' (en el puesto {k} hay un empate de v₁, así que la precisión depende de cómo se ordenen los empatados)' if empate_corte else ''\n"
                       "Markdown(f'''**Respuesta:** σ₁/σ₂ = {S[0]:.3f}/{S[1]:.3f} = **{S[0]/S[1]:.2f}**. La mejor clasificada, {ranking.index[0]}, {\"es\" if ranking.index[0] in criticas else \"no es\"} crítica. "
                       "Con k = {k} críticas, entre las {k} primeras ({lista_y(top_k)}) hay {aciertos} críticas: precisión {aciertos}/{k} = **{precision:.2f}**{nota}.''')"))
    celdas.append(md("En el conjunto de Patterson (110 instancias) la actividad mejor clasificada por $v_1$ es "
                     "crítica en el 77 % de los casos, y la identificación falla sobre todo cuando σ₁/σ₂ está "
                     "cerca de 1 (dos direcciones dominantes). El ranking es informativo, no determinista."))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'''**Respuesta:** el grafo tiene {len(proyecto.nodes)} nodos y {len(proyecto.activities)} arcos, y R tiene {R.shape[0]} filas × {R.shape[1]} columnas. En un proyecto tan pequeño R es manejable, pero el número de rutas "
                       "crece de forma combinatoria con la anchura del grafo (en las instancias de 51 actividades de Patterson hay hasta varios miles). "
                       "Por eso la fórmula es útil como herramienta analítica (sensibilidad, estructura, simulación vectorizada), mientras que para calcular "
                       "τ en proyectos grandes basta el cálculo de tiempos hacia delante, que es lineal en el número de arcos.''')"))
    return celdas, f"instancia {cual} de Patterson ({n_act} actividades)"


def _svd_vs_mc(rng):
    """Ranking SVD frente a índice de criticidad Monte Carlo."""
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 6 <= i["n_rutas"] <= 12 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1
                    and i["segunda"] is not None and i["T"] - i["segunda"] <= 3,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    sd = P.desviaciones(rng, list(D.index))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    celdas = [md("# Enunciado"),
              md("Dos formas de medir la importancia de una actividad: el índice de criticidad de Monte Carlo "
                 "(fracción de réplicas en que está en la ruta más larga; usa las distribuciones de duración) y el "
                 "peso |v₁| de la SVD de la matriz de rutas (usa sólo la estructura). Considere el proyecto cuyo "
                 "cuadro de prelaciones, duraciones medias y desviaciones típicas (normal) se adjunta.")]
    celdas.append(preguntas([
        "Calcule el índice de criticidad de cada actividad con 20 000 réplicas (`monte_carlo`).",
        "Calcule |v₁| de la SVD de R y ordene las actividades.",
        "Compare los dos rankings: correlación de Spearman y actividades en las que más discrepan. "
        "¿Qué información tiene uno que no tenga el otro?",
        "Repita el Monte Carlo multiplicando todas las desviaciones típicas por 3. ¿Se parece más o menos el "
        "índice de criticidad al ranking SVD? ¿Por qué?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(PREP))
    celdas.append(code(_AUX_FORMULA))
    celdas += apartado(1)
    celdas.append(code("dists = {a: ('normal', float(datos_proyecto.loc[a, 'duration']), float(datos_proyecto.loc[a, 'standard_deviation'])) for a in R.columns}\n"
                       "mc = proyecto.monte_carlo(dists, n_samples=20000, seed=5)\n"
                       "criticidad = mc.sensitivity_mean.reindex(R.columns)\n"
                       "criticidad.sort_values(ascending=False).round(3).to_frame('índice de criticidad').T"))
    celdas.append(code("orden_mc = list(criticidad.sort_values(ascending=False).index)\n"
                       "Markdown(f'''**Respuesta:** el índice de criticidad es la fracción de las 20 000 réplicas en que la actividad está en la ruta más larga. "
                       "Las más críticas son {lista_y([f'{a} ({criticidad[a]:.3f})' for a in orden_mc[:3]])}; las menos, {lista_y([f'{a} ({criticidad[a]:.3f})' for a in orden_mc[-2:]])}.''')"))
    celdas += apartado(2)
    celdas.append(code("U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)\n"
                       "v1 = pd.Series(np.abs(Vt[0, :]), index=R.columns)\n"
                       "v1.sort_values(ascending=False).round(3).to_frame('|v₁|').T"))
    celdas.append(code("Markdown(f'**Respuesta:** ordenadas por |v₁| de mayor a menor: {\", \".join(f\"{a} ({v1[a]:.3f})\" for a in v1.sort_values(ascending=False).index)} '\n"
                       "         '(las que tienen el mismo valor ocupan posiciones equivalentes en la estructura).')"))
    celdas += apartado(3)
    celdas.append(code("from scipy.stats import spearmanr\n"
                       "rho = spearmanr(criticidad, v1).correlation\n"
                       "comparacion = pd.DataFrame({'criticidad MC': criticidad, '|v₁|': v1})\n"
                       "comparacion['rango MC'] = comparacion['criticidad MC'].rank(ascending=False)\n"
                       "comparacion['rango SVD'] = comparacion['|v₁|'].rank(ascending=False)\n"
                       "comparacion['diferencia de rango'] = (comparacion['rango MC'] - comparacion['rango SVD']).abs()\n"
                       "display(comparacion.sort_values('diferencia de rango', ascending=False).round(3))\n"
                       "dif = comparacion.sort_values('diferencia de rango', ascending=False)\n"
                       "maxdif = dif['diferencia de rango'].iloc[0]\n"
                       "peores = dif.index[np.isclose(dif['diferencia de rango'], maxdif)].tolist()\n"
                       "detalle = lista_y([f\"{a} (puesto {_num(dif.loc[a, 'rango MC'], 1)} por Monte Carlo y {_num(dif.loc[a, 'rango SVD'], 1)} por SVD)\" for a in peores])\n"
                       "Markdown(f'**Respuesta:** la correlación de Spearman entre ambos rankings es ρ = **{rho:.2f}**. Donde más discrepan es en {detalle}.')"))
    celdas.append(md("Monte Carlo sabe qué rutas son largas y variables; la SVD sólo sabe qué actividades son "
                     "estructuralmente centrales. Una actividad por la que pasan muchas rutas cortas tiene |v₁| alto "
                     "y criticidad baja; una actividad de una única ruta muy larga, lo contrario. Coinciden cuando "
                     "la estructura y las duraciones apuntan a lo mismo."))
    celdas += apartado(4)
    celdas.append(code("dists3 = {a: ('normal', m, 3 * s) for a, (_, m, s) in dists.items()}\n"
                       "mc3 = proyecto.monte_carlo(dists3, n_samples=20000, seed=5)\n"
                       "criticidad3 = mc3.sensitivity_mean.reindex(R.columns)\n"
                       "rho3 = spearmanr(criticidad3, v1).correlation\n"
                       "pd.DataFrame({'criticidad (σ)': criticidad, 'criticidad (3σ)': criticidad3, '|v₁|': v1}).round(3).T"))
    celdas.append(code("cambio = 'sube' if rho3 > rho + 0.03 else ('baja' if rho3 < rho - 0.03 else 'apenas cambia')\n"
                       "Markdown(f'''**Respuesta:** con σ triplicada ρ pasa de {rho:.2f} a {rho3:.2f} ({cambio}). Lo esperable es que con más incertidumbre más rutas tengan ocasión de ser la más larga "
                       "y el índice de criticidad se reparta según cuántas rutas pasan por cada actividad, que es justo lo que mide la SVD, de modo que los rankings se acerquen; "
                       "con poca incertidumbre la criticidad se concentra en la ruta crítica determinista y la estructura importa menos. "
                       "Si aquí el cambio es pequeño es porque la estructura y las duraciones ya apuntan en la misma dirección: las actividades centrales son también las de la ruta crítica.''')"))
    return celdas, f"{n} actividades, {info['n_rutas']} rutas; criticidad Monte Carlo frente a |v₁|"



def _editar_R(rng):
    """Añadir una prelación: qué filas de R desaparecen, cuáles aparecen y cómo cambia τ."""
    n = int(rng.choice([7, 8]))
    estado = {}

    def acepta(i):
        if not (4 <= i["n_rutas"] <= 7 and 1 <= i["n_dummies"] <= 3 and (i["D"] >= 2).all()):
            return False
        preds = i["preds"]
        G = P.grafo(preds)
        cal = i["calendario"]
        # candidatas (p -> a): sin camino en ningún sentido, p no final, a no inicial, y que la nueva prelación cambie τ
        cands = [(p, a) for p in preds for a in preds if p != a and not nx.has_path(G, p, a) and not nx.has_path(G, a, p)]
        rng.shuffle(cands)
        for p, a in cands[:40]:
            nuevo = {x: set(ps) for x, ps in preds.items()}
            nuevo[a].add(p)
            try:
                j = P.analizar(P.reducir(nuevo), i["D"])
            except Exception:
                continue
            if j["T"] > i["T"] and 3 <= j["n_rutas"] <= 9 and j["n_dummies"] <= 4:
                estado.update(p=p, a=a, nuevo=j)
                return True
        return False

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    p, a, nuevo = estado["p"], estado["a"], estado["nuevo"]
    df = P.tabla_base(info, duration=D)
    df2 = P.tabla_base(nuevo, duration=D)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (días) se adjunta. Durante la "
                 f"revisión del plan se descubre que la actividad {a} no puede empezar hasta que termine {p}: "
                 "hay que añadir esa prelación.")]
    celdas.append(preguntas([
        "Construya la matriz de rutas $R$ del proyecto original y calcule $\\tau = \\lVert R\\,t \\rVert_\\infty$.",
        f"Razone, sin recalcular todo, qué le ocurre a $R$ al añadir la prelación {p} → {a}: ¿qué rutas dejan "
        "de serlo (quedan «absorbidas» por otras más largas) y qué rutas nuevas aparecen?",
        "Construya la matriz de rutas del proyecto modificado y compruebe su razonamiento. Calcule el nuevo "
        "τ y la nueva ruta crítica.",
        "¿Puede añadir una prelación *reducir* τ? ¿Y aumentar el número de rutas? Justifíquelo con $R$."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    celdas.append(code(PREP))
    celdas.append(code(_AUX_FORMULA))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT original", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de rutas original"\n#| label: tbl-R\n'
                       "rutas = {r: ' - '.join(acts) for r, acts in proyecto.paths(dummies=False).items()}\n"
                       "Rt = R @ t\n"
                       "display(R)\n"
                       "producto = producto_Rt(proyecto, R, t, datos_proyecto['duration'])\n"
                       "producto"))
    celdas.append(code("tau = float(Rt.max())\n"
                       "Markdown(f'**Respuesta:** R tiene {R.shape[0]} filas (rutas) y {R.shape[1]} columnas; ' + explica_tau(producto, 'días') + '.')"))
    celdas += apartado(2)
    celdas.append(md(f"Añadir {p} → {a} crea un arco nuevo. Aparecen rutas nuevas: las que llegan hasta {p} por "
                     f"cualquier camino (*prefijos*), saltan a {a} y continúan desde {a} hasta el final "
                     f"(*sufijos*): hasta «número de prefijos × número de sufijos» filas nuevas. Las rutas "
                     f"antiguas siguen existiendo, salvo una excepción: si alguna prelación antigua q → {a} se "
                     f"vuelve redundante (porque q es antecesora de {p}), el cuadro depurado la elimina y las "
                     f"rutas que usaban ese salto directo desaparecen, absorbidas por rutas nuevas que pasan por "
                     f"q, …, {p}, {a} y son más largas. Y si {p} era final (o {a} inicial), las rutas que "
                     f"terminaban en {p} (o empezaban en {a}) dejan de serlo: se prolongan a través del nuevo "
                     f"arco. En la práctica $R$ gana filas y puede perder alguna."))
    celdas.append(code(f"p, a = '{p}', '{a}'\n"
                       "prefijos = [r for r in R.index if R.loc[r, p] == 1]\n"
                       "sufijos = [r for r in R.index if R.loc[r, a] == 1]\n"
                       "Markdown(f'Rutas por {p}: {\", \".join(prefijos)}; rutas por {a}: {\", \".join(sufijos)}. '\n"
                       "         f'Se esperan hasta {len(prefijos)} × {len(sufijos)} = {len(prefijos) * len(sufijos)} combinaciones prefijo-sufijo (menos las repetidas).')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones modificado"\n#| label: tbl-datosModificado\n'
                       + literal_dataframe(df2, nombre="datos_modificados")))
    celdas.append(code("prelaciones2 = PredecessorTable.from_dataframe_of_strings(datos_modificados, activity='activity', predecessor='predecessor')\n"
                       "proyecto2 = prelaciones2.create_project()\n"
                       "R2 = proyecto2.path_matrix(dummies=False)\n"
                       "t2 = datos_modificados['duration'][R2.columns]\n"
                       "rutas2 = {r: ' - '.join(acts) for r, acts in proyecto2.paths(dummies=False).items()}\n"
                       "R2t = R2 @ t2\n"
                       "display(R2)\n"
                       "producto2 = producto_Rt(proyecto2, R2, t2, datos_modificados['duration'])\n"
                       "producto2"))
    celdas.append(code("conj1 = {frozenset(R.columns[R.loc[r] == 1]) for r in R.index}\n"
                       "conj2 = {frozenset(R2.columns[R2.loc[r] == 1]) for r in R2.index}\n"
                       "perdidas = [rutas[r] for r in R.index if frozenset(R.columns[R.loc[r] == 1]) in conj1 - conj2]\n"
                       "nuevas = [rutas2[r] for r in R2.index if frozenset(R2.columns[R2.loc[r] == 1]) in conj2 - conj1]\n"
                       "tau2 = float(R2t.max())\n"
                       "Markdown(f'''**Respuesta:** la matriz modificada tiene {R2.shape[0]} filas (antes {R.shape[0]}).\\n\\n- Rutas que desaparecen: {'; '.join(perdidas) or 'ninguna'}.\\n- Rutas nuevas: {'; '.join(nuevas) or 'ninguna'}.\\n\\n"
                       "Nuevo {explica_tau(producto2, 'días')} (antes τ = {tau:.0f} días).''')"))
    celdas += fig_grafo(durations="datos_modificados['duration']", cap="Grafo PERT modificado", label="fig-pert2", obj="proyecto2")
    celdas += apartado(4)
    celdas.append(md("Añadir una prelación nunca reduce τ: cada ruta nueva contiene como subconjunto de "
                     "actividades a alguna combinación que ya restringía el proyecto, y las rutas que "
                     "desaparecen son las contenidas en alguna ruta nueva (su suma de duraciones es menor o "
                     "igual). Formalmente, para cada fila antigua $r$ existe una fila nueva $r'$ con $R'_{r'} \\ge "
                     "R_r$ componente a componente, luego $\\max R' t \\ge \\max R t$. El número de rutas sí puede "
                     "crecer (producto de prefijos por sufijos) o decrecer (si muchas quedan absorbidas): $R$ no "
                     "es monótona en tamaño, sólo en lo que restringe."))
    celdas.append(code("Markdown(f'**Respuesta:** no puede reducir τ; aquí lo aumenta de {tau:.0f} a {tau2:.0f} días. El número de rutas sí puede cambiar en ambos sentidos: '\n"
                       "         f'aquí pasa de {R.shape[0]} a {R2.shape[0]} ({len(perdidas)} {\"desaparece\" if len(perdidas) == 1 else \"desaparecen\"} y '\n"
                       "         f'{len(nuevas)} {\"aparece\" if len(nuevas) == 1 else \"aparecen\"}).')"))
    return celdas, f"{n} actividades; añadir {p} → {a} cambia τ de {int(info['T'])} a {int(nuevo['T'])}"


def _casi_criticas(rng):
    """Margen de cada ruta, actividades que aparecen en las rutas casi críticas y proyecto más sensible."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 9 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1
                    and i["segunda"] is not None and i["T"] - i["segunda"] <= 2 and (i["D"] >= 2).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    df = P.tabla_base(info, duration=D)
    delta = 2
    # segundo proyecto: mismas prelaciones, duraciones alteradas para que la ruta crítica destaque
    L = info["L"]
    crit = info["rutas_criticas"][0]
    acts_crit = list(info["P"].columns[info["P"].loc[crit] == 1])
    D2 = D.copy()
    D2[acts_crit] = D2[acts_crit] + 2
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (días) se adjunta. El vector "
                 "$R\\,t$ contiene la duración de todas las rutas, no sólo de la crítica: la diferencia "
                 "$\\tau - (R\\,t)_r$ es el *margen* de la ruta $r$ (cuánto puede alargarse sin cambiar τ).")]
    celdas.append(preguntas([
        "Calcule $R\\,t$, τ y el margen de cada ruta. Ordene las rutas de menor a mayor margen.",
        f"Llame *casi críticas* a las rutas con margen ≤ {delta}. ¿Qué actividades pertenecen a alguna ruta "
        "casi crítica sin ser críticas? Obténgalas con $R^\\top$: sume las filas de $R$ correspondientes a las "
        "rutas casi críticas y mire qué columnas son positivas.",
        "¿Cuál es el menor retraso de una sola actividad no crítica que cambiaría la ruta crítica? Compruébelo "
        "recalculando τ.",
        f"Un segundo proyecto tiene las mismas prelaciones pero las actividades de la ruta crítica duran 2 "
        "días más. Calcule sus márgenes. ¿Cuál de los dos proyectos es más *sensible* (más rutas pueden "
        "convertirse en críticas con pequeños retrasos)? ¿Cuál tiene más varianza de τ si todas las "
        "duraciones son inciertas? Razónelo y, si quiere, compruébelo con una simulación breve."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    celdas.append(code(PREP + f"\ndelta = {delta}"))
    celdas.append(code(_AUX_FORMULA))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Duración y margen de cada ruta"\n#| label: tbl-margenes\n'
                       "Rt = R @ t\n"
                       "tau = float(Rt.max())\n"
                       "margen = (tau - Rt)\n"
                       "producto = producto_Rt(proyecto, R, t, datos_proyecto['duration'])\n"
                       "tabla = pd.DataFrame({'actividades': producto['actividades'], 'R·t = suma': producto['fila de R · t = suma'],\n"
                       "                      'margen = τ − R·t': [f'{tau:.0f} − {Rt[r]:.0f} = {margen[r]:.0f}' for r in R.index]}, index=R.index)\n"
                       "tabla = tabla.loc[margen.sort_values(kind='stable').index]\n"
                       "tabla"))
    celdas.append(code("Markdown('**Respuesta:** ' + explica_tau(producto, 'días') + '. Rutas de menor a mayor margen: '\n"
                       "         + ', '.join(f'{r} ({margen[r]:.0f})' for r in margen.sort_values(kind=\"stable\").index) + '.')"))
    celdas += apartado(2)
    celdas.append(code("casi = [r for r in tabla.index if margen[r] <= delta]\n"
                       "criticas = [r for r in tabla.index if margen[r] == 0]\n"
                       "veces = R.loc[casi].sum()               # columnas de R^T · indicador de rutas casi críticas\n"
                       "en_criticas = R.loc[criticas].sum() > 0\n"
                       "casi_sin_ser = veces.index[(veces > 0) & ~en_criticas].tolist()\n"
                       "display(pd.DataFrame({'en rutas casi críticas (nº)': veces, 'crítica': en_criticas.map({True: 'sí', False: 'no'})}).T)\n"
                       "Markdown(f'**Respuesta:** rutas casi críticas (margen ≤ {delta}): {lista_y(casi)}. Sumando sus filas de R, las columnas positivas son '\n"
                       "         f'{lista_y([a for a in R.columns if veces[a] > 0])}; quitando las críticas, quedan {lista_y(casi_sin_ser) if casi_sin_ser else \"ninguna\"}.')"))
    celdas += apartado(3)
    celdas.append(md("Si la actividad $i$ se retrasa $x$ días, cada ruta que la contiene se alarga $x$; la "
                     "primera no crítica en alcanzar τ es la de menor margen entre las que contienen $i$. El "
                     "menor retraso que cambia la ruta crítica es, por tanto, el mínimo sobre las actividades no "
                     "críticas del menor margen de sus rutas (más una unidad si se quiere que *supere* a la "
                     "crítica)."))
    celdas.append(code("no_criticas = [a for a in R.columns if not en_criticas[a]]\n"
                       "umbral = pd.Series({a: float(margen[R[a] == 1].min()) for a in no_criticas}).sort_values()\n"
                       "a_min = umbral.index[0]; x = int(umbral.iloc[0]) + 1\n"
                       "t_retrasada = t.copy(); t_retrasada[a_min] += x\n"
                       "Rt_r = R @ t_retrasada\n"
                       "display(umbral.to_frame('margen mínimo de sus rutas').T)\n"
                       "m_min = int(umbral.iloc[0])\n"
                       "rutas_a = [r for r in R.index if R.loc[r, a_min] == 1 and margen[r] == m_min]\n"
                       "t_empate = t.copy(); t_empate[a_min] += m_min\n"
                       "Rt_e = R @ t_empate\n"
                       "crit_e = [r for r in R.index if np.isclose(Rt_e[r], Rt_e.max())]\n"
                       "crit_r = [r for r in R.index if np.isclose(Rt_r[r], Rt_r.max())]\n"
                       "Markdown(f'''**Respuesta:** la actividad no crítica con menos margen en sus rutas es {a_min}: su ruta {lista_y(rutas_a)} tiene un margen de {plural(m_min, \"día\")}. "
                       "Retrasando {a_min} {plural(m_min, 'día')}, esa ruta empata con la crítica (τ sigue en {float(Rt_e.max()):.0f}, rutas críticas {lista_y(crit_e)}); "
                       "con {plural(x, 'día')}, la supera: τ pasa de {tau:.0f} a {float(Rt_r.max()):.0f} y la ruta crítica es {lista_y(crit_r)} (antes {Rt.idxmax()}). "
                       "Así, el menor retraso que cambia la ruta crítica es {plural(m_min, 'día')} si basta con un empate y {plural(x, 'día')} si debe sustituirla.''')"))
    celdas += apartado(4)
    celdas.append(code("t2 = t.copy()\n"
                       "for act in R.columns[R.loc[Rt.idxmax()] == 1]:\n"
                       "    t2[act] += 2\n"
                       "Rt2 = R @ t2\n"
                       "margen2 = Rt2.max() - Rt2\n"
                       "comparacion = pd.DataFrame({'margen (proyecto 1)': margen, 'margen (proyecto 2)': margen2}).sort_values('margen (proyecto 1)')\n"
                       "display(comparacion)\n"
                       "n_casi1, n_casi2 = int((margen <= delta).sum()), int((margen2 <= delta).sum())\n"
                       "mas_sensible = 'el proyecto 1' if n_casi1 > n_casi2 else ('el proyecto 2' if n_casi2 > n_casi1 else 'ninguno de los dos (empatan)')\n"
                       "Markdown(f'Rutas con margen ≤ {delta}: {n_casi1} en el proyecto 1 y {n_casi2} en el 2. Es más sensible {mas_sensible}: más rutas compiten por ser la crítica.')"))
    celdas.append(code("rng = np.random.default_rng(7)\n"
                       "N = 5000\n"
                       "def simula(tv):\n"
                       "    X = pd.DataFrame(rng.normal(tv.values, 0.25 * tv.values, size=(N, len(tv))), columns=tv.index).clip(lower=0.1)\n"
                       "    return (X @ R.T).max(axis='columns')\n"
                       "T1, T2 = simula(t), simula(t2)\n"
                       "pd.DataFrame({'proyecto 1': [tau, T1.mean(), T1.std(), T1.mean() - tau], 'proyecto 2': [float(Rt2.max()), T2.mean(), T2.std(), T2.mean() - float(Rt2.max())]},\n"
                       "             index=['τ determinista', 'media simulada', 'desv. típica simulada', 'sesgo de fusión (media − τ)']).round(2)"))
    celdas.append(code("mayor_var = 'el proyecto 2' if T2.std() > T1.std() else 'el proyecto 1'\n"
                       "mayor_sesgo = 'el proyecto 1' if T1.mean() - tau > T2.mean() - float(Rt2.max()) else 'el proyecto 2'\n"
                       "Markdown(f'''**Respuesta:** es más sensible {mas_sensible} ({n_casi1} frente a {n_casi2} rutas con margen ≤ {delta}). En la simulación (desviación típica del 25 % de cada duración), "
                       "la desviación típica de τ es {T1.std():.2f} en el proyecto 1 y {T2.std():.2f} en el 2, así que tiene más varianza {mayor_var}; el sesgo de fusión es mayor en {mayor_sesgo} "
                       "({T1.mean() - tau:.2f} frente a {T2.mean() - float(Rt2.max()):.2f}). Ojo: parte de la mayor dispersión del proyecto 2 se debe a que sus actividades críticas son más largas "
                       "y aquí la desviación de cada duración es proporcional a su media.''')"))
    celdas.append(md("Con duraciones inciertas, el proyecto con más rutas casi críticas sufre más *sesgo de "
                     "fusión*: la media simulada supera a τ porque en cada réplica gana la ruta que haya salido "
                     "más larga entre varias candidatas. La desviación típica, en cambio, suele ser algo menor "
                     "en el proyecto sensible (el máximo de varias variables se dispersa menos que una sola) y "
                     "mayor en el proyecto 2, donde τ depende casi siempre de la misma ruta y hereda toda su "
                     "varianza. La sensibilidad se lee directamente en el vector de márgenes $\\tau\\mathbf{1} - "
                     "R\\,t$: cuantos más ceros y valores pequeños tenga, más frágil es el camino crítico."))
    return celdas, f"{n} actividades; {int((T - info['L'] <= delta).sum())} rutas con margen ≤ {delta}"


VARIANTES = [
    Variante("tau-norma-infinito", 1, "R, R·t y τ = ‖R·t‖∞; equivalencia con Zaderenko; τ como función lineal a trozos de una duración", _tau),
    Variante("gradiente", 2, "∂τ/∂t con rutas críticas empatadas: subgradiente, diferencias finitas y suma de sensibilidades", _gradiente),
    Variante("svd-rango-1", 3, "SVD de R: σ₁/σ₂, rankings por u₁ y v₁, aproximación de rango 1 de τ y precisión frente al conjunto crítico", _svd, grafo="pert"),
    Variante("nucleo-incidencia", 3, "matriz de incidencia, R·H = 0 y perturbaciones de las duraciones que no cambian τ", _nucleo, grafo="pert"),
    Variante("patterson", 2, "una instancia de referencia (Patterson, formato RCP): τ frente a CPM y ranking SVD", _patterson, grafo="pert"),
    Variante("svd-vs-montecarlo", 3, "índice de criticidad Monte Carlo frente a |v₁|: correlación y efecto de la incertidumbre", _svd_vs_mc),
    Variante("editar-R", 2, "añadir una prelación: filas de R que aparecen y desaparecen (prefijos × sufijos), nuevo τ y monotonía",
             _editar_R),
    Variante("casi-criticas", 2, "vector de márgenes τ·1 − R·t: rutas casi críticas, actividades que las forman, menor retraso que cambia la crítica y sensibilidad",
             _casi_criticas),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
