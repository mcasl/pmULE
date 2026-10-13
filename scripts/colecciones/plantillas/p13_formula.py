"""Colección 13 — La fórmula del tiempo de compleción τ = ‖R·t‖∞ (arXiv 2410.10252):
matriz de rutas, norma infinito, gradiente, SVD de la matriz de rutas, núcleo de la
incidencia e instancias de referencia (Patterson)."""
import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado,
                        literal_dataframe, literal_dict, BMATRIX)

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
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo PERT"\n#| label: fig-pert\n' "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(code('#| tbl-cap: "Matriz de rutas R"\n#| label: tbl-R\n'
                       "rutas = proyecto.paths(dummies=False)\n"
                       "display(pd.Series({r: ' - '.join(a) for r, a in rutas.items()}).to_frame('actividades'))\n"
                       "proyecto.display_path_matrix(dummies=False)"))
    celdas.append(code("Markdown(f'R tiene {R.shape[0]} filas (rutas) y {R.shape[1]} columnas (actividades).')"))
    celdas += apartado(2)
    celdas.append(code("Rt = R @ t\n"
                       "tau = float(Rt.max())\n"
                       "Math(bmatrix(R) + r' \\times ' + bmatrix(t) + ' = ' + bmatrix(Rt) + r' \\quad\\Rightarrow\\quad \\tau = ' + f'{tau:.0f}')"))
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(code("Markdown(f'‖R·t‖∞ = {tau:.0f}; tiempo más temprano del último nodo = {proyecto.duration(datos_proyecto[\"duration\"])}; '\n"
                       "         f'la librería (`proyecto.tau`) da {proyecto.tau(t.to_dict()):.0f}. Las tres coinciden: el cálculo de tiempos hacia delante es una forma de calcular el máximo sin enumerar las rutas.')"))
    celdas += apartado(3)
    celdas.append(code("Markdown(f'''‖R·t‖₁ = {float(Rt.abs().sum()):.0f} sumaría las duraciones de todas las rutas, contando varias veces cada actividad compartida: no tiene sentido como plazo. "
                       "El mínimo de R·t, {float(Rt.min()):.0f}, es la ruta más corta: el tiempo que *como mínimo* ocupa cualquier cadena del proyecto, pero el proyecto no termina hasta que acaba la última ruta. "
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
    celdas.append(code("Markdown(f'''τ es el máximo de varias funciones lineales de t_{k} (una por ruta): las rutas que no contienen a {k} son constantes y las que la contienen tienen pendiente 1. "
                       "El resultado es lineal a trozos y convexo: pendiente 0 mientras otra ruta manda y pendiente 1 a partir de t_{k} = {codo} (cuando la ruta más larga que pasa por {k} alcanza a la crítica). "
                       "La duración actual de {k} es {int(t[k])}: su sensibilidad es {int(pendiente[list(valores).index(int(t[k]))])}.''')"))
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
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo PERT con tiempos y rutas críticas"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas.append(code("Rt = R @ t\n"
                       "tau = float(Rt.max())\n"
                       "criticas = Rt.index[np.isclose(Rt, tau)].tolist()\n"
                       "display(pd.DataFrame({'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in R.index], 'duración': Rt}))\n"
                       "Markdown(f'τ = {tau:.0f}; rutas críticas: {\", \".join(criticas)}.')"))
    celdas += apartado(2)
    celdas.append(code("indicadores = R.loc[criticas]\n"
                       "gradiente = indicadores.mean()\n"
                       "display(indicadores)\n"
                       "pd.DataFrame({'∂τ/∂t (media de indicadores)': gradiente, 'tau_sensitivity': proyecto.tau_sensitivity(t.to_dict()).reindex(R.columns)}).T"))
    celdas.append(md("Valor 1: la actividad está en todas las rutas críticas (alargarla o acortarla cambia τ uno a "
                     "uno). Valor 0: no está en ninguna (τ no cambia, al menos para cambios pequeños). Valor "
                     "intermedio, por ejemplo 0,5: está en una de las dos rutas críticas; alargarla sí alarga el "
                     "proyecto, pero acortarla no lo acorta, porque la otra ruta sigue midiendo lo mismo."))
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
    celdas.append(md("Para las actividades con valor intermedio la pendiente por la derecha (+1) y por la "
                     "izquierda (0) son distintas: τ es un máximo de funciones lineales y en los puntos donde dos "
                     "rutas empatan tiene una esquina. La media de los indicadores es un *subgradiente*, no una "
                     "derivada: el valor que reparte la sensibilidad entre las rutas empatadas."))
    celdas += apartado(4)
    celdas.append(code("suma = float(gradiente.sum())\n"
                       "longitud_media = float(indicadores.sum(axis=1).mean())\n"
                       "Markdown(f'''Σ_i ∂τ/∂t_i = Σ_i (1/k) Σ_{{r crítica}} R_ri = (1/k) Σ_r Σ_i R_ri = media del número de actividades de las rutas críticas. "
                       "Aquí Σ = {suma:.2f} y la longitud media de las {len(criticas)} rutas críticas es {longitud_media:.2f}.''')"))
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
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Matriz de rutas R"\n#| label: tbl-R\n' "proyecto.display_path_matrix(dummies=False)"))
    celdas.append(code("U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)\n"
                       "u1, v1 = U[:, 0], Vt[0, :]\n"
                       "if u1.sum() < 0:\n"
                       "    u1, v1 = -u1, -v1          # R es no negativa: u1 y v1 pueden tomarse no negativos\n"
                       "display(Math(r'\\Sigma = ' + bmatrix(np.diag(S), dec=2)))\n"
                       "Markdown(f'σ₁ = {S[0]:.3f}, σ₂ = {S[1]:.3f}; σ₁/σ₂ = {S[0]/S[1]:.2f}.')"))
    celdas += apartado(2)
    celdas.append(code("peso_actividades = pd.Series(v1, index=R.columns).sort_values(ascending=False)\n"
                       "peso_rutas = pd.Series(u1, index=R.index).sort_values(ascending=False)\n"
                       "apariciones = R.sum()\n"
                       "display(pd.DataFrame({'v₁': peso_actividades.round(3), 'rutas en las que aparece': apariciones[peso_actividades.index]}).T)\n"
                       "pd.DataFrame({'u₁': peso_rutas.round(3), 'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in peso_rutas.index], 'nº actividades': R.loc[peso_rutas.index].sum(axis=1)})"))
    celdas.append(md("Las actividades con mayor $v_1$ son las que aparecen en más rutas, sobre todo en las rutas "
                     "largas y con muchas actividades compartidas (las que a su vez tienen mayor $u_1$): $v_1$ es "
                     "una medida de centralidad estructural. Las actividades iniciales o finales únicas aparecen "
                     "en todas las rutas y encabezan la lista."))
    celdas += apartado(3)
    celdas.append(code("tau = float((R @ t).max())\n"
                       "tau_1 = float(S[0] * u1.max() * (v1 @ t.values))\n"
                       "Markdown(f'τ = {tau:.0f}; aproximación de rango 1: τ₁ = {S[0]:.3f} × {u1.max():.3f} × {v1 @ t.values:.3f} = {tau_1:.2f}; error relativo {abs(tau_1 - tau) / tau * 100:.1f} %.')"))
    celdas.append(md("La aproximación sustituye cada fila de $R$ por un múltiplo de $v_1$: funciona bien cuando "
                     "las rutas se parecen entre sí (σ₁ ≫ σ₂) y peor cuando el proyecto tiene varias ramas muy "
                     "distintas. El τ exacto es siempre la norma infinito de $R\\,t$; la SVD es un resumen."))
    celdas += apartado(4)
    celdas.append(code("criticas = set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)\n"
                       "k = len(criticas)\n"
                       "top_k = list(peso_actividades.index[:k])\n"
                       "precision = len(set(top_k) & criticas) / k\n"
                       "pd.DataFrame({'ranking por v₁': peso_actividades.index, 'crítica': [a in criticas for a in peso_actividades.index]}).set_index('ranking por v₁').T"))
    celdas.append(code("Markdown(f'''La actividad mejor clasificada, {peso_actividades.index[0]}, {\"es\" if peso_actividades.index[0] in criticas else \"no es\"} crítica. "
                       "Entre las {k} primeras hay {len(set(top_k) & criticas)} críticas (precisión {precision:.2f}). "
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
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo PERT numerado"\n#| label: fig-pert\n' "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(code('#| tbl-cap: "Matriz de incidencia H (arcos × nodos)"\n#| label: tbl-H\n'
                       "H = proyecto.incidence_matrix()\n"
                       "H"))
    celdas += apartado(2)
    celdas.append(code("R = proyecto.path_matrix(dummies=True)[H.index]\n"
                       "RH = R @ H\n"
                       "display(RH)\n"
                       "interiores = H.columns[1:-1]\n"
                       "Markdown(f'R·H es cero en las columnas interiores: {bool((RH[interiores] == 0).all().all())}; en la columna del nodo inicial vale {int(RH.iloc[0, 0])} y en la del final {int(RH.iloc[0, -1])} para todas las rutas.')"))
    celdas.append(md("Una ruta entra y sale de cada nodo interior exactamente una vez, así que la suma de sus "
                     "entradas de incidencia es +1 − 1 = 0; sólo sale del inicial (−1) y sólo entra en el final (+1)."))
    celdas += apartado(3)
    celdas.append(code(literal_dict({str(k): int(v) for k, v in y.items()}, "y_interior") + "\n"
                       "y = pd.Series(0, index=H.columns, dtype=int)\n"
                       "for nodo, valor in y_interior.items():\n"
                       "    y[nodo] = valor\n"
                       "delta = H @ y\n"
                       "t = pd.Series({a: (datos_proyecto.loc[a, 'duration'] if not a.startswith('@') else 0) for a in H.index})\n"
                       "t_perturbado = t + delta\n"
                       "pd.DataFrame({'t': t, 'δ = H·y': delta, 't + δ': t_perturbado}).T"))
    celdas.append(code("Rt, Rt2 = R @ t, R @ t_perturbado\n"
                       "display(pd.DataFrame({'R·t': Rt, 'R·(t+δ)': Rt2, 'diferencia': Rt2 - Rt}))\n"
                       "Markdown(f'τ = {float(Rt.max()):.0f} antes y {float(Rt2.max()):.0f} después: las duraciones de todas las rutas cambian en {int((Rt2 - Rt).iloc[0])} (= y_final − y_inicial = 0), así que τ es invariante.')"))
    celdas.append(md("δ = H·y añade a cada arco la diferencia de «potencial» entre su destino y su origen. Es lo "
                     "mismo que mover el reloj de cada nodo: si al nodo j se le suma y_j a su tiempo, los arcos "
                     "que llegan a j parecen y_j más largos y los que salen y_j más cortos, y a lo largo de cualquier "
                     "ruta cerrada entre inicio y final todo se cancela. (Algunas duraciones perturbadas pueden salir "
                     "negativas: es un ejercicio algebraico, no un calendario.)"))
    celdas += apartado(4)
    celdas.append(code("N = proyecto.nullspace()\n"
                       "rango = np.linalg.matrix_rank(N.values)\n"
                       "Markdown(f'''Las columnas interiores de H generan un espacio de dimensión {rango} (número de nodos interiores, {N.shape[1]}) contenido en el núcleo de R. "
                       "Hay por tanto {rango} direcciones independientes en las que pueden moverse las duraciones de los {len(t)} arcos sin que ninguna ruta cambie. "
                       "Consecuencia: observando sólo duraciones de rutas no pueden identificarse las duraciones de los arcos; sólo se identifican módulo esas {rango} direcciones.''')"))
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
                       "Markdown(f'{len(proyecto)} actividades (incluidas las dos ficticias del formato, de duración 0), {len(proyecto.dummies)} ficticias añadidas por el grafo, {len(proyecto.nodes)} nodos y {R.shape[0]} rutas.')"))
    celdas.append(code('#| fig-cap: "Grafo PERT de la instancia"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=0.6, nodesep=0.3)"))
    celdas += apartado(2)
    celdas.append(code("Rt = R @ t\n"
                       "tau = float(Rt.max())\n"
                       "cpm = proyecto.duration(datos_proyecto['duration'])\n"
                       "criticas = sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts))\n"
                       "Markdown(f'τ = ‖R·t‖∞ = {tau:.0f}; CPM = {cpm}; coinciden: {tau == cpm}. Actividades críticas: {\", \".join(criticas)}.')"))
    celdas += apartado(3)
    celdas.append(code("U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)\n"
                       "v1 = np.abs(Vt[0, :])\n"
                       "ranking = pd.Series(v1, index=R.columns).sort_values(ascending=False)\n"
                       "k = len(criticas)\n"
                       "precision = len(set(ranking.index[:k]) & set(criticas)) / k\n"
                       "display(pd.DataFrame({'v₁': ranking.round(3), 'crítica': [a in criticas for a in ranking.index]}).T)\n"
                       "Markdown(f'σ₁/σ₂ = {S[0]/S[1]:.2f}. La mejor clasificada, {ranking.index[0]}, {\"es\" if ranking.index[0] in criticas else \"no es\"} crítica; precisión en las {k} primeras: {precision:.2f}.')"))
    celdas.append(md("En el conjunto de Patterson (110 instancias) la actividad mejor clasificada por $v_1$ es "
                     "crítica en el 77 % de los casos, y la identificación falla sobre todo cuando σ₁/σ₂ está "
                     "cerca de 1 (dos direcciones dominantes). El ranking es informativo, no determinista."))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'''El grafo tiene {len(proyecto.nodes)} nodos y {len(proyecto.activities)} arcos, pero R tiene {R.shape[0]} filas: el número de rutas "
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
    celdas += apartado(1)
    celdas.append(code("dists = {a: ('normal', float(datos_proyecto.loc[a, 'duration']), float(datos_proyecto.loc[a, 'standard_deviation'])) for a in R.columns}\n"
                       "mc = proyecto.monte_carlo(dists, n_samples=20000, seed=5)\n"
                       "criticidad = mc.sensitivity_mean.reindex(R.columns)\n"
                       "criticidad.sort_values(ascending=False).round(3).to_frame('índice de criticidad').T"))
    celdas += apartado(2)
    celdas.append(code("U, S, Vt = np.linalg.svd(R.values.astype(float), full_matrices=False)\n"
                       "v1 = pd.Series(np.abs(Vt[0, :]), index=R.columns)\n"
                       "v1.sort_values(ascending=False).round(3).to_frame('|v₁|').T"))
    celdas += apartado(3)
    celdas.append(code("from scipy.stats import spearmanr\n"
                       "rho = spearmanr(criticidad, v1).correlation\n"
                       "comparacion = pd.DataFrame({'criticidad MC': criticidad, '|v₁|': v1})\n"
                       "comparacion['rango MC'] = comparacion['criticidad MC'].rank(ascending=False)\n"
                       "comparacion['rango SVD'] = comparacion['|v₁|'].rank(ascending=False)\n"
                       "comparacion['diferencia de rango'] = (comparacion['rango MC'] - comparacion['rango SVD']).abs()\n"
                       "display(comparacion.sort_values('diferencia de rango', ascending=False).round(3))\n"
                       "Markdown(f'Correlación de Spearman entre ambos rankings: ρ = {rho:.2f}.')"))
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
                       "Markdown(f'''Con σ triplicada ρ pasa de {rho:.2f} a {rho3:.2f} ({cambio}). Lo esperable es que con más incertidumbre más rutas tengan ocasión de ser la más larga "
                       "y el índice de criticidad se reparta según cuántas rutas pasan por cada actividad, que es justo lo que mide la SVD, de modo que los rankings se acerquen; "
                       "con poca incertidumbre la criticidad se concentra en la ruta crítica determinista y la estructura importa menos. "
                       "Si aquí el cambio es pequeño es porque la estructura y las duraciones ya apuntan en la misma dirección: las actividades centrales son también las de la ruta crítica.''')"))
    return celdas, f"{n} actividades, {info['n_rutas']} rutas; criticidad Monte Carlo frente a |v₁|"


VARIANTES = [
    Variante("tau-norma-infinito", 1, "R, R·t y τ = ‖R·t‖∞; equivalencia con Zaderenko; τ como función lineal a trozos de una duración", _tau),
    Variante("gradiente", 2, "∂τ/∂t con rutas críticas empatadas: subgradiente, diferencias finitas y suma de sensibilidades", _gradiente),
    Variante("svd-rango-1", 3, "SVD de R: σ₁/σ₂, rankings por u₁ y v₁, aproximación de rango 1 de τ y precisión frente al conjunto crítico", _svd),
    Variante("nucleo-incidencia", 3, "matriz de incidencia, R·H = 0 y perturbaciones de las duraciones que no cambian τ", _nucleo),
    Variante("patterson", 2, "una instancia de referencia (Patterson, formato RCP): τ frente a CPM y ranking SVD", _patterson),
    Variante("svd-vs-montecarlo", 3, "índice de criticidad Monte Carlo frente a |v₁|: correlación y efecto de la incertidumbre", _svd_vs_mc),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
