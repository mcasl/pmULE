"""Colección 09 — Método de Monte Carlo: simulación con la matriz de caminos,
comparación con la aproximación normal, índices de criticidad."""
import numpy as np
import pandas as pd

from .. import proyectos as P
from .p08_probabilidad import _AUX_RUTAS
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX, grafo_nombre, ESTADO)

META = dict(slug="09_montecarlo", prefijo="MonteCarlo",
            titulo="Método de Monte Carlo",
            descripcion="Simular la duración del proyecto con la matriz de caminos, compararla con la "
                        "aproximación normal y calcular índices de criticidad y sensibilidad.",
            n_defecto=3)

DISTRIBUCIONES = {1: "normal", 2: "triangular", 3: "pert"}


def _base(k, rng):
    dist = DISTRIBUCIONES[(k - 1) % 3 + 1]
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 8 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1
                    and i["segunda"] is not None and i["T"] - i["segunda"] <= 2 and i["T"] >= 12
                    and (i["D"] >= 3).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    T = int(info["T"])
    if dist == "normal":
        sd = P.desviaciones(rng, list(D.index))
        df = P.tabla_base(info, duration=D, standard_deviation=sd)
    elif dist == "triangular":
        a, m, b = P.tres_estimaciones_triangular(rng, D)
        df = P.tabla_base(info, a=a, m=m, b=b)
    else:
        a, m, b = P.tres_estimaciones(rng, D)
        df = P.tabla_base(info, a=a, m=m, b=b)
    plazo = T + 1
    prob_obj = 0.90
    N = 10000
    semilla = int(rng.integers(1, 10000))

    nombre_dist = {"normal": "normal", "triangular": "triangular", "pert": "Beta-PERT"}[dist]
    celdas = [md("# Enunciado")]
    if dist == "normal":
        celdas.append(md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones "
                         "típicas (en semanas) se adjunta. Suponga que la duración de cada actividad sigue una "
                         "distribución normal."))
    else:
        celdas.append(md(f"Considere el proyecto cuyo cuadro de prelaciones y estimaciones optimista, más probable y "
                         f"pesimista (en semanas) se adjunta. Suponga que la duración de cada actividad sigue una "
                         f"distribución {nombre_dist} de parámetros $(a, m, b)$."))
    pregs = [f"Dibuje el grafo {grafo_nombre()}, construya la matriz de caminos y calcule la duración del proyecto con las "
             "duraciones medias.",
             f"Genere {N} réplicas de las duraciones de las actividades y, con la matriz de caminos, obtenga la "
             "duración del proyecto en cada una. Calcule su media, desviación típica y percentiles 5, 50 y 95.",
             f"Calcule la probabilidad de terminar en {plazo} semanas o menos y la duración que puede comprometerse "
             f"con un {int(prob_obj * 100)}% de probabilidad. Compare con la aproximación normal por el camino crítico "
             "y explique las diferencias.",
             "Calcule el índice de criticidad de cada ruta y de cada actividad. ¿Qué actividad no crítica "
             "convendría vigilar?"]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    if dist == "normal":
        celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                           "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2"))
    elif dist == "triangular":
        celdas.append(md("Para la distribución triangular de mínimo $a$, moda $m$ y máximo $b$ la media es "
                         "$(a + m + b)/3$ y la varianza $(a^2 + m^2 + b^2 - ab - am - bm)/18$. Sustituyendo:"))
        celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                           "datos_proyecto['duration'] = datos_proyecto.eval('(a + m + b) / 3').round().astype(int)\n"
                           "datos_proyecto['variance'] = datos_proyecto.eval('(a**2 + m**2 + b**2 - a*b - a*m - b*m) / 18')\n"
                           "datos_proyecto['standard_deviation'] = np.sqrt(datos_proyecto['variance'])\n"
                           "pd.DataFrame({'media = (a + m + b)/3': [f\"({r.a:g} + {r.m:g} + {r.b:g})/3 = {(r.a + r.m + r.b) / 3:.2f}\" for r in datos_proyecto.itertuples()],\n"
                           "              'varianza = (a² + m² + b² − ab − am − bm)/18': [f\"({r.a:g}² + {r.m:g}² + {r.b:g}² − {r.a:g}·{r.b:g} − {r.a:g}·{r.m:g} − {r.b:g}·{r.m:g})/18 = {r.variance:.3f}\" for r in datos_proyecto.itertuples()],\n"
                           "              'σ': datos_proyecto['standard_deviation'].round(3).values}, index=datos_proyecto.index)"))
    else:
        celdas.append(md("Con el método PERT la media de cada actividad es $(a + 4m + b)/6$ y su desviación "
                         "típica se aproxima por $(b - a)/6$. Sustituyendo:"))
        celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                           "datos_proyecto['duration'] = datos_proyecto.eval('(a + 4 * m + b) / 6').round().astype(int)\n"
                           "datos_proyecto['standard_deviation'] = datos_proyecto.eval('(b - a) / 6')\n"
                           "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                           "detalle_estimaciones(datos_proyecto)"))
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas.append(code(_AUX_RUTAS))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas.append(md("La matriz de caminos tiene una fila por ruta y una columna por actividad (1 si la actividad "
                     "está en la ruta):"))
    celdas.append(code('#| tbl-cap: "Matriz de caminos"\n#| label: tbl-matrizCaminos\n'
                       "path_matrix = proyecto.path_matrix(dummies=False).drop_duplicates()   # rutas que sólo difieren en ficticias, una vez\n"
                       f"proyecto.display_rutas('{ESTADO['grafo']}')"))
    celdas.append(md("Duración media y varianza de cada ruta (producto de la matriz de caminos por las medias y por las "
                     "varianzas de las actividades):"))
    celdas.append(code("calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duraciones = datos_proyecto['duration'][path_matrix.columns]\n"
                       "duraciones_rutas_medias = path_matrix @ duraciones\n"
                       "duracion_media = int(duraciones_rutas_medias.max())\n"
                       "ruta_critica = duraciones_rutas_medias.idxmax()\n"
                       "desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       "rutas = tabla_rutas(path_matrix, datos_proyecto['duration'], datos_proyecto['variance'], calendario)\n"
                       "limpia_tabla(rutas, decimales=3)"))
    celdas.append(code("acts_critica = ordena_ruta(list(path_matrix.columns[path_matrix.loc[ruta_critica] == 1]), calendario)\n"
                       "Markdown(f'''{explica_sigma_ruta(acts_critica, datos_proyecto[\"variance\"], nombre=\"σ_T\")}.\\n\\n'''\n"
                       "         f'''**Respuesta:** con las duraciones medias el proyecto dura {plural(duracion_media, \"semana\")}; la ruta crítica es '''\n"
                       "         f'''{\" – \".join(acts_critica)}, con desviación típica σ_T = {desviacion_proyecto:.3f} semanas.''')"))
    celdas += apartado(ap); ap += 1
    if dist == "normal":
        celdas.append(md("Cada réplica es un vector de duraciones, una por actividad, generado con su "
                         "distribución normal (truncada en cero por seguridad)."))
        gen = ("duraciones_mc = pd.DataFrame({act: np.clip(rng.normal(fila['duration'], fila['standard_deviation'], N), 0, None)\n"
               "                              for act, fila in datos_proyecto.iterrows()})")
    elif dist == "triangular":
        celdas.append(md("Cada réplica es un vector de duraciones generado con la distribución triangular "
                         "de mínimo $a$, moda $m$ y máximo $b$ (cuando $a = m = b$ la duración es fija)."))
        gen = ("def muestras_triangular(a, m, b, size):\n"
               "    return np.full(size, float(a)) if a == b else rng.triangular(a, m, b, size)\n"
               "duraciones_mc = pd.DataFrame({act: muestras_triangular(fila['a'], fila['m'], fila['b'], N)\n"
               "                              for act, fila in datos_proyecto.iterrows()})")
    else:
        celdas.append(md("La Beta-PERT de parámetros $(a, m, b)$ es $X = a + (b - a)\\,Y$ con "
                         "$Y \\sim \\mathrm{Beta}(\\alpha, \\beta)$, $\\alpha = 1 + 4\\frac{m-a}{b-a}$ y "
                         "$\\beta = 1 + 4\\frac{b-m}{b-a}$; su media es exactamente $(a+4m+b)/6$."))
        gen = ("def muestras_beta_pert(a, m, b, size):\n"
               "    if b == a:\n"
               "        return np.full(size, float(a))\n"
               "    alfa = 1 + 4 * (m - a) / (b - a)\n"
               "    beta = 1 + 4 * (b - m) / (b - a)\n"
               "    return a + (b - a) * rng.beta(alfa, beta, size)\n"
               "duraciones_mc = pd.DataFrame({act: muestras_beta_pert(fila['a'], fila['m'], fila['b'], N)\n"
               "                              for act, fila in datos_proyecto.iterrows()})")
    celdas.append(code(f"N = {N}\n"
                       f"rng = np.random.default_rng({semilla})\n" + gen + "\n"
                       "duraciones_mc = duraciones_mc[path_matrix.columns]"))
    celdas.append(md("Comprobación de que las muestras reproducen la media y la dispersión de cada actividad:"))
    celdas.append(code('#| tbl-cap: "Medias y desviaciones típicas muestrales frente a las teóricas"\n#| label: tbl-comprobacionMuestras\n'
                       "pd.DataFrame({'media teórica': datos_proyecto['duration'], 'media muestral': duraciones_mc.mean(),\n"
                       "              'σ teórica': datos_proyecto['standard_deviation'], 'σ muestral': duraciones_mc.std()}).round(2).T"))
    if dist == "pert":
        celdas.append(md("Las medias muestrales coinciden con las teóricas. Las desviaciones típicas salen algo "
                         "mayores que $(b-a)/6$: esa fórmula es una aproximación; la desviación exacta de la "
                         "Beta-PERT simétrica es $(b-a)/\\sqrt{28} \\approx (b-a)/5.3$."))
    celdas.append(md("La duración de cada ruta en cada réplica es la suma de las duraciones de sus actividades "
                     "(producto de las duraciones por la traspuesta de la matriz de caminos), y la del proyecto "
                     "es la de la ruta más larga: el máximo por filas. Las cinco primeras réplicas:"))
    celdas.append(code("duraciones_rutas_mc = duraciones_mc @ path_matrix.T\n"
                       "duracion_proyecto_mc = duraciones_rutas_mc.max(axis='columns')\n"
                       "pd.concat([duraciones_rutas_mc, duracion_proyecto_mc.rename('Proyecto (máximo)')], axis=1).head().round(2)"))
    celdas.append(md("Los descriptores se calculan sobre las réplicas de la duración del proyecto. El percentil $p$ "
                     "es el valor que deja por debajo la proporción $p$ de las réplicas: se ordenan de menor a mayor y "
                     "se toma la que ocupa la posición $p\\cdot N$ (interpolando entre las dos vecinas)."))
    celdas.append(code('#| tbl-cap: "Descriptores de la duración simulada del proyecto"\n#| label: tbl-descriptores\n'
                       "descriptores = duracion_proyecto_mc.describe(percentiles=[0.05, 0.5, 0.95])\n"
                       "descriptores.index = ['réplicas', 'media', 'desviación típica', 'mínimo', 'P5', 'P50', 'P95', 'máximo']\n"
                       "descriptores.round(2).to_frame('Duración del proyecto (semanas)').T"))
    celdas.append(code("ordenadas = np.sort(duracion_proyecto_mc.values)\n"
                       "Markdown(f'''Por ejemplo, el P95 está entre la réplica {int(0.95 * (N - 1)) + 1} y la {int(0.95 * (N - 1)) + 2} de las {N} ordenadas '''\n"
                       "         f'''({ordenadas[int(0.95 * (N - 1))]:.2f} y {ordenadas[int(0.95 * (N - 1)) + 1]:.2f}).\\n\\n'''\n"
                       "         f'''**Respuesta:** media {descriptores[\"media\"]:.2f} semanas, desviación típica {descriptores[\"desviación típica\"]:.2f}; '''\n"
                       "         f'''percentiles P5 = {descriptores[\"P5\"]:.2f}, P50 = {descriptores[\"P50\"]:.2f} y P95 = {descriptores[\"P95\"]:.2f} semanas.''')"))
    celdas.append(code('#| fig-cap: "Distribución simulada de la duración frente a la aproximación normal"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       f"ax.hist(duracion_proyecto_mc, bins=60, density=True, alpha=0.5, color='skyblue', label='Monte Carlo ({nombre_dist})')\n"
                       "x = np.linspace(duracion_proyecto_mc.min(), duracion_proyecto_mc.max(), 300)\n"
                       "ax.plot(x, norm.pdf(x, duracion_media, desviacion_proyecto), color='navy', label='Aproximación normal (camino crítico)')\n"
                       f"ax.axvline({plazo}, color='red', linestyle='--', label='Plazo ({plazo})')\n"
                       "ax.set_xlabel('Duración del proyecto (semanas)'); ax.legend(); plt.show()"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Con las réplicas, la probabilidad de cumplir el plazo es la proporción de réplicas que no lo "
                     "superan, y la duración que se cumple con una probabilidad dada es el percentil empírico "
                     "correspondiente. Con la aproximación normal, $T \\sim N(\\mu, \\sigma_T)$ con la media y la "
                     "desviación típica de la ruta crítica (apartado 1)."))
    celdas.append(code(f"plazo, probabilidad_objetivo = {plazo}, {prob_obj}\n"
                       "cumplen = int((duracion_proyecto_mc <= plazo).sum())\n"
                       "p_mc = float((duracion_proyecto_mc <= plazo).mean())\n"
                       "t_mc = float(duracion_proyecto_mc.quantile(probabilidad_objetivo))\n"
                       "p_normal = float(norm.cdf(plazo, duracion_media, desviacion_proyecto))\n"
                       "t_normal = float(norm.ppf(probabilidad_objetivo, duracion_media, desviacion_proyecto))\n"
                       "pct = int(probabilidad_objetivo * 100)\n"
                       "Markdown(f'''Monte Carlo: {cumplen} de las {N} réplicas duran {plazo} semanas o menos, luego P(T ≤ {plazo}) ≈ {cumplen}/{N} = **{p_mc:.4f}**; '''\n"
                       "         f'''el percentil {pct} de las réplicas (posición ≈ {int(probabilidad_objetivo * N)} de {N} ordenadas) es **{t_mc:.2f}** semanas.\\n\\n'''\n"
                       "         f'''Aproximación normal:\\n\\n- {explica_probabilidad(duracion_media, desviacion_proyecto, plazo)}\\n'''\n"
                       "         f'''- {explica_percentil(duracion_media, desviacion_proyecto, probabilidad_objetivo, \"semanas\")}''')"))
    celdas.append(code('#| tbl-cap: "Aproximación normal frente a Monte Carlo"\n#| label: tbl-comparacion\n'
                       "pd.DataFrame({'Aproximación normal': [duracion_media, desviacion_proyecto, p_normal * 100, t_normal],\n"
                       "              'Monte Carlo': [duracion_proyecto_mc.mean(), duracion_proyecto_mc.std(), p_mc * 100, t_mc]},\n"
                       "             index=['Duración media', 'Desviación típica', f'P(T ≤ {plazo}) (%)', f'Duración para {pct}%']).round(2)"))
    celdas.append(code("ruta_critica_mc = duraciones_rutas_mc.idxmax(axis='columns')\n"
                       "criticidad_rutas = (ruta_critica_mc.value_counts(normalize=True) * 100).reindex(path_matrix.index, fill_value=0.0)\n"
                       "dm = duracion_proyecto_mc.mean() - duracion_media\n"
                       "ds = duracion_proyecto_mc.std() - desviacion_proyecto\n"
                       "otras = 100 - criticidad_rutas[ruta_critica]\n"
                       "if dm > 0.05:\n"
                       "    t_media = (f'Monte Carlo da una duración media {dm:.2f} semanas mayor. En cada réplica la duración del proyecto es el máximo de todas '\n"
                       "               f'las rutas y en el {otras:.1f} % de las réplicas manda otra ruta distinta de la crítica: las rutas casi críticas empujan la '\n"
                       "               'duración hacia arriba (*sesgo de fusión*), algo que la aproximación normal, que sólo mira la ruta crítica, no recoge.')\n"
                       "else:\n"
                       "    t_media = (f'Las medias casi coinciden (diferencia {dm:+.2f} semanas): la ruta crítica es la más larga en el {100 - otras:.1f} % de las '\n"
                       "               'réplicas, así que el máximo de las rutas apenas se separa de ella y el sesgo de fusión es pequeño.')\n"
                       "if p_mc < p_normal:\n"
                       "    t_prob = (f'La probabilidad de cumplir el plazo es menor con Monte Carlo ({p_mc:.4f} frente a {p_normal:.4f}) y la duración al {pct} % '\n"
                       "              f'mayor ({t_mc:.2f} frente a {t_normal:.2f}): la aproximación normal por el camino crítico es optimista.')\n"
                       "else:\n"
                       "    t_prob = (f'La probabilidad de cumplir el plazo es {\"igual\" if np.isclose(p_mc, p_normal, atol=5e-4) else \"algo mayor\"} con Monte Carlo ({p_mc:.4f} frente a {p_normal:.4f}): '\n"
                       "              'con una sola ruta dominante la diferencia queda dentro del error de muestreo y de la forma de las distribuciones.')\n"
                       f"dist_pert = {dist == 'pert'}\n"
                       "if dist_pert and ds > 0.02:\n"
                       "    t_sd = (f'La desviación típica simulada es mayor ({duracion_proyecto_mc.std():.2f} frente a {desviacion_proyecto:.2f}) porque la aproximación '\n"
                       "            'PERT σ = (b − a)/6 subestima la dispersión real de la Beta-PERT (ver la comprobación de las muestras).')\n"
                       "elif ds < -0.02:\n"
                       "    t_sd = (f'La desviación típica simulada es menor ({duracion_proyecto_mc.std():.2f} frente a {desviacion_proyecto:.2f}): cuando la ruta '\n"
                       "            'crítica sale corta otra ocupa su lugar y la cola izquierda se recorta.')\n"
                       "else:\n"
                       "    t_sd = (f'Las desviaciones típicas son parecidas ({duracion_proyecto_mc.std():.2f} frente a {desviacion_proyecto:.2f}).')\n"
                       "Markdown(f'''{t_media} {t_prob} {t_sd}\\n\\n'''\n"
                       "         f'''**Respuesta:** P(T ≤ {plazo}) ≈ {p_mc:.4f} con Monte Carlo y {p_normal:.4f} con la aproximación normal; la duración que se cumple con un '''\n"
                       "         f'''{pct} % de probabilidad es {t_mc:.2f} semanas con Monte Carlo y {t_normal:.2f} con la normal (en semanas enteras, '''\n"
                       "         f'''{int(np.ceil(round(t_mc, 2)))} y {int(np.ceil(round(t_normal, 2)))}).''')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("El índice de criticidad de una ruta es la fracción de réplicas en que es la más larga; "
                     "el de una actividad, la fracción de réplicas en que pertenece a la ruta más larga."))
    celdas.append(code('#| tbl-cap: "Índice de criticidad de las rutas"\n#| label: tbl-criticidadRutas\n'
                       "pd.DataFrame({'actividades': rutas['actividades'], 'duración media': duraciones_rutas_medias,\n"
                       "              'veces la más larga': ruta_critica_mc.value_counts().reindex(path_matrix.index, fill_value=0),\n"
                       "              'criticidad (%)': criticidad_rutas}).sort_values('criticidad (%)', ascending=False).round(1)"))
    celdas.append(code('#| tbl-cap: "Índice de criticidad de las actividades frente a su holgura total"\n#| label: tbl-criticidadActividades\n'
                       "criticidad_actividades = path_matrix.loc[ruta_critica_mc].mean() * 100\n"
                       "pd.DataFrame({'holgura total': calendario['H_total'], 'criticidad (%)': criticidad_actividades}).round(1).T"))
    celdas.append(md("La librería hace el mismo cálculo de forma vectorizada: la sensibilidad empírica media de "
                     "`monte_carlo` es precisamente el índice de criticidad de cada actividad."))
    if dist == "normal":
        dists = "{act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}"
    elif dist == "triangular":
        dists = "{act: ('triangular', fila['a'], fila['m'], fila['b']) for act, fila in datos_proyecto.iterrows()}"
    else:
        dists = ("{act: (lambda a_, m_, b_: (lambda size=None: muestras_beta_pert(a_, m_, b_, size)))(fila['a'], fila['m'], fila['b'])\n"
                 "                                   for act, fila in datos_proyecto.iterrows()}")
        celdas.append(md("La opción `('pert', a, m, b)` de `monte_carlo` es una normal recortada al intervalo "
                         "$[a, b]$, no la Beta-PERT exacta; por eso se le pasa como función la misma Beta-PERT "
                         "usada arriba (la función recibe `size` y devuelve un vector de muestras)."))
    celdas.append(code(f"resultado = proyecto.monte_carlo({dists}, n_samples=N, seed={semilla})\n"
                       "display(resultado.summary().round(2).rename(columns={'mean': 'media', 'std': 'desv. típica', 'q05': 'P5', 'q50': 'P50', 'q95': 'P95', 'min': 'mínimo', 'max': 'máximo'}))\n"
                       "(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%) según monte_carlo').T"))
    celdas.append(code("vigilar = criticidad_actividades[calendario['H_total'] > 0].sort_values(ascending=False)\n"
                       "rutas_txt = '; '.join(f\"{rutas.loc[r, 'actividades']}: {criticidad_rutas[r]:.1f} %\" for r in criticidad_rutas.sort_values(ascending=False).index if criticidad_rutas[r] > 0)\n"
                       "if len(vigilar) and vigilar.iloc[0] > 0:\n"
                       "    x = vigilar.index[0]\n"
                       "    t_vig = (f'La actividad no crítica que convendría vigilar es **{x}**: está en la ruta más larga en el {vigilar.iloc[0]:.1f} % de las réplicas '\n"
                       "             f'aunque tiene {plural(int(calendario.loc[x, \"H_total\"]), \"semana\")} de holgura total. El análisis determinista la da por holgada, '\n"
                       "             'pero un retraso pequeño la convierte en crítica.')\n"
                       "else:\n"
                       "    t_vig = 'Ninguna actividad no crítica llega a estar en la ruta más larga en las réplicas: no hay ninguna que vigilar especialmente.'\n"
                       "Markdown(f'''**Respuesta:** índices de criticidad de las rutas (las que alguna vez son la más larga): {rutas_txt}. Los de las actividades '''\n"
                       "         f'''son los de la tabla. {t_vig}''')"))
    sub = f"{n} actividades, {info['n_rutas']} rutas, distribución {nombre_dist}"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _a_mano(rng):
    """Cinco réplicas a mano: R·T, máximo por columnas, media, desviación típica y criticidad."""
    n = int(rng.choice([6, 7]))
    info = P.buscar(rng, lambda i: 3 <= i["n_rutas"] <= 5 and 1 <= i["n_dummies"] <= 3 and (i["D"] >= 2).all() and i["T"] >= 8,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, R = info["D"], info["P"]
    N = 5
    T = pd.DataFrame({f"r{j}": (D + rng.integers(-1, 2, size=len(D))).clip(lower=1) for j in range(1, N + 1)}, index=D.index)
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones medias (en días) se adjunta. Para "
                 f"entender el método de Monte Carlo se han generado a mano sólo {N} réplicas de las duraciones "
                 "(cada columna es una réplica: la duración que cada actividad ha tenido en esa simulación).")]
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones y duraciones medias"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(code('#| tbl-cap: "Duraciones simuladas (una columna por réplica)"\n#| label: tbl-replicas\n'
                       + literal_dataframe(T, nombre="replicas")))
    celdas.append(preguntas([
        f"Dibuje el grafo {grafo_nombre()}, enumere las rutas y escriba la matriz de rutas R (una fila por ruta, una columna por actividad).",
        "Escriba la matriz de tiempos T con las réplicas como columnas y calcule a mano el producto R·T. "
        "¿Qué es cada elemento del resultado?",
        "Obtenga la duración del proyecto en cada réplica como el máximo de cada columna (τ = ‖R·t‖∞ para cada "
        "columna t). Calcule la media y la desviación típica muestral de las cinco duraciones.",
        "Para cada réplica, ¿qué ruta ha sido la crítica? Calcule el índice de criticidad de cada ruta y de cada "
        "actividad con estas cinco réplicas. Compare con el camino crítico de las duraciones medias.",
        "¿Qué cambia al pasar de 5 réplicas a 10 000? Compruébelo generando las réplicas con una distribución "
        "normal de media la duración media y desviación típica 1."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas.append(md("Cada fila de la matriz de rutas R es una ruta y cada columna una actividad: 1 si la actividad "
                     "está en la ruta y 0 si no."))
    celdas.append(code('#| tbl-cap: "Matriz de rutas R"\n#| label: tbl-matrizRutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "rutas = proyecto.paths(dummies=False)\n"
                       f"proyecto.display_rutas('{ESTADO['grafo']}')"))
    celdas.append(code('#| tbl-cap: "Rutas del proyecto y su duración con las duraciones medias"\n#| label: tbl-rutas\n'
                       "tabla_rutas = explica_rutas(proyecto, datos_proyecto['duration'], unidad='días')\n"
                       "tabla_rutas"))
    celdas.append(code("Markdown(f'''**Respuesta:** hay {len(R)} rutas: ''' + '; '.join(f\"{r}: {tabla_rutas.loc[r, 'ruta']}\" for r in R.index)\n"
                       "         + '. La matriz R es la de la tabla anterior (filas = rutas, columnas = actividades).')"))
    celdas += apartado(2)
    celdas.append(md("La matriz de tiempos T tiene una fila por actividad y una columna por réplica (es la tabla de "
                     "réplicas del enunciado). Cada elemento de R·T se obtiene multiplicando una fila de R por una "
                     "columna de T, es decir, sumando las duraciones de esa réplica de las actividades marcadas con 1:"))
    celdas.append(code("T = replicas.loc[R.columns]\n"
                       "RT = R @ T\n"
                       "from pmule import display_producto_rutas\n"
                       "display_producto_rutas(R, T, RT)"))
    celdas.append(code('#| tbl-cap: "R·T: duración de cada ruta (filas) en cada réplica (columnas), término a término"\n#| label: tbl-RT\n'
                       "pd.DataFrame({j: [' + '.join(f'{T.loc[a, j]}' for a in R.columns[R.loc[r] == 1]) + f' = {RT.loc[r, j]}' for r in R.index]\n"
                       "              for j in T.columns}, index=R.index)"))
    celdas.append(code("r0, j0 = R.index[0], T.columns[0]\n"
                       "acts0 = list(R.columns[R.loc[r0] == 1])\n"
                       "Markdown(f'''**Respuesta:** el elemento (r, j) de R·T es la duración de la ruta r en la réplica j: la suma de las duraciones, en esa '''\n"
                       "         f'''réplica, de las actividades que forman la ruta. Por ejemplo, {r0} en {j0}: '''\n"
                       "         + ' + '.join(f'{a}({T.loc[a, j0]})' for a in acts0) + f''' = {RT.loc[r0, j0]} días. Una fila de R selecciona las actividades '''\n"
                       "         '''de la ruta; una columna de T es un escenario completo.''')"))
    celdas += apartado(3)
    celdas.append(code("tau = RT.max()\n"
                       "mas_largas = {j: [r for r in RT.index if RT.loc[r, j] == tau[j]] for j in RT.columns}\n"
                       "pd.DataFrame({'τ = máximo de la columna': tau, 'rutas más largas': {j: ', '.join(v) for j, v in mas_largas.items()}}).T"))
    celdas.append(code("media = tau.mean(); desviacion = tau.std(ddof=1)\n"
                       "dif2 = [(x - media) ** 2 for x in tau]\n"
                       "Markdown(f'''Media = ({\" + \".join(str(int(x)) for x in tau)}) / {len(tau)} = {int(tau.sum())} / {len(tau)} = {media:.2f} días.\\n\\n'''\n"
                       "         f'''Desviación típica muestral = √(Σ(τ − media)² / (n − 1)) = √((''' + ' + '.join(f'{d:.2f}' for d in dif2)\n"
                       "         + f''') / {len(tau) - 1}) = √({sum(dif2):.2f} / {len(tau) - 1}) = {desviacion:.2f} días.\\n\\n'''\n"
                       "         f'''**Respuesta:** duraciones del proyecto en las réplicas: {\", \".join(f\"{j} = {int(tau[j])}\" for j in tau.index)}; media {media:.2f} días '''\n"
                       "         f'''y desviación típica muestral {desviacion:.2f} días (con las duraciones medias el proyecto duraría '''\n"
                       "         f'''{proyecto.duration(datos_proyecto[\"duration\"])} días).''')"))
    celdas += apartado(4)
    celdas.append(md("Una ruta es crítica en una réplica si es la más larga en ella (si hay empate, lo son todas las "
                     "empatadas). El índice de criticidad de una ruta es la fracción de réplicas en que es crítica; "
                     "el de una actividad, la fracción de réplicas en que está en alguna ruta crítica."))
    celdas.append(code("es_critica = RT.eq(tau, axis='columns')            # rutas × réplicas: ¿es la más larga?\n"
                       "criticidad_rutas = es_critica.sum(axis='columns')\n"
                       "criticidad_actividades = ((R.T.astype(int) @ es_critica.astype(int)) > 0).sum(axis='columns')\n"
                       "pd.DataFrame({'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in R.index],\n"
                       "              'veces crítica': criticidad_rutas, 'índice de criticidad': criticidad_rutas / len(tau)}, index=R.index)"))
    celdas.append(code("pd.DataFrame({'veces en una ruta crítica': criticidad_actividades, 'índice de criticidad': criticidad_actividades / len(tau)}).T"))
    celdas.append(code("camino_medio = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "rutas_medias = [r for r in R.index if set(R.columns[R.loc[r] == 1]) in [set(a) for a in camino_medio.values()]]\n"
                       "empates = [j for j, v in mas_largas.items() if len(v) > 1]\n"
                       "nota_empate = (f' En {lista_y(empates)} hay empate entre rutas, que cuentan como críticas a la vez; por eso los índices de las rutas pueden sumar más de 1.'\n"
                       "               if empates else '')\n"
                       "Markdown(f'''**Respuesta:** rutas críticas por réplica: {\"; \".join(f\"{j}: {', '.join(v)}\" for j, v in mas_largas.items())}.{nota_empate} '''\n"
                       "         f'''Con las duraciones medias la ruta crítica es {lista_y(rutas_medias)} ({\", \".join(\" – \".join(a) for a in camino_medio.values())}), '''\n"
                       "         f'''que en las réplicas ha sido la más larga {int(criticidad_rutas[rutas_medias].max())} de {len(tau)} veces. Con tan pocas réplicas los '''\n"
                       "         f'''índices sólo pueden valer múltiplos de {1/len(tau):.1f}.''')"))
    celdas += apartado(5)
    celdas.append(md("Se repite exactamente el mismo cálculo (R·T y máximo por columnas) con 10 000 columnas, cada una "
                     "generada con duraciones normales de media la duración media y desviación típica 1."))
    celdas.append(code("rng = np.random.default_rng(1)\n"
                       "N = 10000\n"
                       "T_grande = pd.DataFrame(np.clip(rng.normal(datos_proyecto.loc[R.columns, 'duration'].values[:, None], 1.0, (len(R.columns), N)), 0.5, None), index=R.columns)\n"
                       "tau_grande = (R @ T_grande).max()\n"
                       "criticidad_grande = R.loc[(R @ T_grande).idxmax()].mean()\n"
                       "pd.DataFrame({'5 réplicas': [media, desviacion], '10 000 réplicas': [tau_grande.mean(), tau_grande.std(ddof=1)]}, index=['media', 'desviación típica']).round(2)"))
    celdas.append(code("pd.DataFrame({'5 réplicas': criticidad_actividades / len(tau), '10 000 réplicas': criticidad_grande}).round(3).T"))
    celdas.append(code("Markdown(f'''**Respuesta:** la operación es exactamente la misma (R·T y máximo por columnas); sólo cambia el número de columnas. '''\n"
                       "         f'''Con 10 000 réplicas la media es {tau_grande.mean():.2f} días y la desviación típica {tau_grande.std(ddof=1):.2f} (con 5: {media:.2f} y '''\n"
                       "         f'''{desviacion:.2f}). Con muchas réplicas estos valores se estabilizan (su error estadístico decrece como 1/√N: el error típico de la media '''\n"
                       "         f'''pasa de unos {desviacion / np.sqrt(len(tau)):.2f} días con 5 réplicas a unos {tau_grande.std(ddof=1) / np.sqrt(N):.3f} con 10 000) y los '''\n"
                       "         '''índices de criticidad dejan de ser múltiplos gruesos de 0.2.''')"))
    return celdas, f"{n} actividades, {info['n_rutas']} rutas, {N} réplicas calculadas a mano"


def _mixtas(rng):
    """Distribuciones distintas por actividad, incluidas deterministas; uso de monte_carlo de la librería."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 8 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1
                    and i["segunda"] is not None and i["T"] - i["segunda"] <= 3 and (i["D"] >= 3).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    tipos = rng.choice(["normal", "triangular", "uniforme", "fija"], size=len(D), p=[0.35, 0.3, 0.2, 0.15])
    filas = []
    for a, t in zip(D.index, tipos):
        d = int(D[a])
        if t == "normal":
            filas.append((a, "normal", f"μ = {d}, σ = {float(rng.choice([0.5, 1.0, 1.5]))}"))
        elif t == "triangular":
            p = int(rng.integers(1, 3)); filas.append((a, "triangular", f"a = {max(1, d - p)}, m = {d}, b = {d + p}"))
        elif t == "uniforme":
            p = int(rng.integers(1, 3)); filas.append((a, "uniforme", f"a = {max(1, d - p)}, b = {d + p}"))
        else:
            filas.append((a, "fija", f"{d}"))
    dist = pd.DataFrame(filas, columns=["activity", "distribucion", "parametros"]).set_index("activity")
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones medias (semanas) se adjunta. Cada "
                 "actividad sigue una distribución distinta, según la información disponible sobre ella: normal "
                 "(media y desviación típica), triangular (mínimo, moda, máximo), uniforme (mínimo, máximo) o "
                 "fija (duración cierta).")]
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones y duraciones medias"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(code('#| tbl-cap: "Distribución de la duración de cada actividad"\n#| label: tbl-distribuciones\n'
                       + literal_dataframe(dist, nombre="distribuciones")))
    celdas.append(preguntas([
        "Calcule la media y la varianza de cada actividad según su distribución y compruebe que las medias "
        "coinciden con las duraciones medias del cuadro.",
        "Determine la ruta crítica con las medias y la probabilidad de terminar en la duración media más dos "
        "semanas según la aproximación normal.",
        "Simule el proyecto con 20 000 réplicas usando la distribución de cada actividad (las fijas no varían). "
        "Compare media, desviación típica y la probabilidad anterior.",
        "¿Qué actividad tiene mayor índice de criticidad sin ser crítica en media? ¿Influye en ello que alguna "
        "actividad sea de duración fija?"]))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "import re\n"
                       "def parametros(texto):\n"
                       "    return [float(x) for x in re.findall(r'[-+]?\\d*\\.?\\d+', texto)]"))
    celdas += apartado(1)
    celdas.append(md("Fórmulas de cada distribución: normal, media $\\mu$ y varianza $\\sigma^2$; triangular, media "
                     "$(a + m + b)/3$ y varianza $(a^2 + m^2 + b^2 - ab - am - bm)/18$; uniforme, media $(a + b)/2$ y "
                     "varianza $(b - a)^2/12$; fija, media el valor dado y varianza 0."))
    celdas.append(code("filas = {}\n"
                       "g = lambda x: f'{x:g}'\n"
                       "for act, fila in distribuciones.iterrows():\n"
                       "    p = parametros(fila['parametros'])\n"
                       "    if fila['distribucion'] == 'normal':\n"
                       "        media, var = p[0], p[1] ** 2\n"
                       "        calc_m, calc_v = f'μ = {g(p[0])}', f'σ² = {g(p[1])}² = {var:.3f}'\n"
                       "    elif fila['distribucion'] == 'triangular':\n"
                       "        a, m, b = p; media, var = (a + m + b) / 3, (a*a + m*m + b*b - a*b - a*m - b*m) / 18\n"
                       "        calc_m = f'({g(a)} + {g(m)} + {g(b)})/3 = {media:g}'\n"
                       "        calc_v = f'({g(a)}² + {g(m)}² + {g(b)}² − {g(a)}·{g(b)} − {g(a)}·{g(m)} − {g(b)}·{g(m)})/18 = {var:.3f}'\n"
                       "    elif fila['distribucion'] == 'uniforme':\n"
                       "        a, b = p; media, var = (a + b) / 2, (b - a) ** 2 / 12\n"
                       "        calc_m, calc_v = f'({g(a)} + {g(b)})/2 = {media:g}', f'({g(b)} − {g(a)})²/12 = {var:.3f}'\n"
                       "    else:\n"
                       "        media, var = p[0], 0.0\n"
                       "        calc_m, calc_v = f'{g(p[0])} (fija)', '0'\n"
                       "    filas[act] = {'distribución': fila['distribucion'], 'media': media, 'varianza': var, 'cálculo de la media': calc_m, 'cálculo de la varianza': calc_v}\n"
                       "resumen = pd.DataFrame(filas).T\n"
                       "resumen['media del cuadro'] = datos_proyecto['duration']\n"
                       "datos_proyecto['variance'] = resumen['varianza'].astype(float)\n"
                       "resumen[['distribución', 'cálculo de la media', 'media del cuadro', 'cálculo de la varianza']]"))
    celdas.append(code("coinciden = all(np.isclose(float(resumen.loc[a, 'media']), float(resumen.loc[a, 'media del cuadro'])) for a in resumen.index)\n"
                       "Markdown('**Respuesta:** medias y varianzas en la tabla; ' + ('todas las medias coinciden con las duraciones medias del cuadro.' if coinciden else\n"
                       "         'no coinciden en ' + lista_y([a for a in resumen.index if not np.isclose(float(resumen.loc[a, \"media\"]), float(resumen.loc[a, \"media del cuadro\"]))]) + '.'))"))
    celdas += apartado(2)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con duraciones medias y camino crítico", label="fig-pert")
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['variance'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "ruta_var = max(rutas_criticas, key=lambda r: datos_proyecto.loc[list(rutas_criticas[r]), 'variance'].sum())\n"
                       "acts_var = ordena_ruta(rutas_criticas[ruta_var], calendario)\n"
                       "plazo = duracion_media + 2\n"
                       "p_normal = float(norm.cdf(plazo, duracion_media, desviacion_proyecto)) if desviacion_proyecto > 0 else 1.0\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "Markdown(f'''{explica_sigma_ruta(acts_var, datos_proyecto[\"variance\"], nombre=\"σ_T\")}.\\n\\n'''\n"
                       "         + (f'''{explica_probabilidad(duracion_media, desviacion_proyecto, plazo)}.''' if desviacion_proyecto > 0 else\n"
                       "            'La ruta crítica no tiene varianza: la duración es cierta y P = 1.')\n"
                       "         + f'''\\n\\n**Respuesta:** ruta crítica {\" – \".join(acts_var)}, duración media {plural(duracion_media, \"semana\")}; '''\n"
                       "         f'''P(T ≤ {plazo}) = {p_normal:.4f} según la aproximación normal.''')"))
    celdas += apartado(3)
    celdas.append(md("`monte_carlo` acepta una distribución por actividad: `('normal', μ, σ)`, "
                     "`('triangular', a, m, b)`, `('uniform', a, b)` o una función; una duración fija es una "
                     "función que siempre devuelve el mismo valor. En cada réplica sortea la duración de cada actividad, "
                     "suma las duraciones a lo largo de cada ruta y toma la ruta más larga como duración del proyecto."))
    celdas.append(code("dists = {}\n"
                       "for act, fila in distribuciones.iterrows():\n"
                       "    p = parametros(fila['parametros'])\n"
                       "    if fila['distribucion'] == 'normal':\n"
                       "        dists[act] = ('normal', p[0], p[1])\n"
                       "    elif fila['distribucion'] == 'triangular':\n"
                       "        dists[act] = ('triangular', p[0], p[1], p[2])\n"
                       "    elif fila['distribucion'] == 'uniforme':\n"
                       "        dists[act] = ('uniform', p[0], p[1])\n"
                       "    else:\n"
                       "        dists[act] = (lambda valor: (lambda size=None: np.full(size, valor) if size else valor))(p[0])\n"
                       "resultado = proyecto.monte_carlo(dists, n_samples=20000, seed=11)\n"
                       "p_mc = float((resultado.tau_samples <= plazo).mean())\n"
                       "cumplen = int((resultado.tau_samples <= plazo).sum())\n"
                       "display(resultado.summary().round(2).rename(columns={'mean': 'media', 'std': 'desv. típica', 'q05': 'P5', 'q50': 'P50', 'q95': 'P95', 'min': 'mínimo', 'max': 'máximo'}))\n"
                       "pd.DataFrame({'aproximación normal': [duracion_media, desviacion_proyecto, p_normal], 'Monte Carlo': [resultado.tau_mean, resultado.tau_std, p_mc]},\n"
                       "             index=['media', 'desviación típica', f'P(T ≤ {plazo})']).round(3)"))
    celdas.append(code("dm = resultado.tau_mean - duracion_media\n"
                       "Markdown(f'''Con Monte Carlo, {cumplen} de las 20000 réplicas terminan en {plazo} semanas o menos: P(T ≤ {plazo}) ≈ {cumplen}/20000 = {p_mc:.4f}.\\n\\n'''\n"
                       "         f'''**Respuesta:** la media simulada es {resultado.tau_mean:.3f} semanas ({dm:+.3f} respecto de la media por la ruta crítica, '''\n"
                       "         + ('mayor por el efecto de las rutas casi críticas' if dm > 0.02 else 'prácticamente igual') + '''), '''\n"
                       "         f'''la desviación típica {resultado.tau_std:.3f} (normal: {desviacion_proyecto:.3f}) y P(T ≤ {plazo}) ≈ {p_mc:.4f} '''\n"
                       "         f'''(normal: {p_normal:.4f}, {\"algo menor con Monte Carlo\" if p_mc < p_normal - 0.002 else (\"algo mayor con Monte Carlo: las triangulares y uniformes tienen colas acotadas, más cortas que las de la normal\" if p_mc > p_normal + 0.002 else \"prácticamente igual\")}).''')"))
    celdas += apartado(4)
    celdas.append(md("El índice de criticidad de cada actividad (fracción de réplicas en que está en la ruta más "
                     "larga) es la sensibilidad media que devuelve `monte_carlo`:"))
    celdas.append(code("crit = (resultado.sensitivity_mean * 100).round(1).rename('índice de criticidad (%)')\n"
                       "tabla = pd.concat([crit, calendario['H_total'].rename('holgura'), resumen['distribución'], resumen['varianza'].astype(float).round(2)], axis=1).sort_values('índice de criticidad (%)', ascending=False)\n"
                       "tabla"))
    celdas.append(code("no_criticas = tabla[tabla['holgura'] > 0]\n"
                       "vigilar = no_criticas['índice de criticidad (%)'].idxmax() if len(no_criticas) else None\n"
                       "fijas = list(resumen.index[resumen['distribución'] == 'fija'])\n"
                       "if fijas:\n"
                       "    t_fijas = ('Sí influye: las actividades fijas (' + '; '.join(f\"{a}, con índice {tabla.loc[a, 'índice de criticidad (%)']} %\" for a in fijas)\n"
                       "               + ') no aportan varianza, así que una ruta con actividades fijas varía menos y rara vez «adelanta» a la crítica; '\n"
                       "               'el índice de una actividad fija depende sólo de lo que hagan las demás actividades de sus rutas.')\n"
                       "else:\n"
                       "    t_fijas = 'En este proyecto no hay actividades de duración fija.'\n"
                       "if vigilar is not None and tabla.loc[vigilar, 'índice de criticidad (%)'] > 0:\n"
                       "    t_vig = (f'**Respuesta:** la actividad no crítica en media con mayor índice de criticidad es {vigilar} ({tabla.loc[vigilar, \"índice de criticidad (%)\"]} %, '\n"
                       "             f'holgura {plural(int(tabla.loc[vigilar, \"holgura\"]), \"semana\")}, distribución {tabla.loc[vigilar, \"distribución\"]} con varianza {tabla.loc[vigilar, \"varianza\"]}).')\n"
                       "else:\n"
                       "    t_vig = '**Respuesta:** ninguna actividad no crítica llega a estar en la ruta más larga en las réplicas.'\n"
                       "Markdown(t_vig + ' ' + t_fijas)"))
    return celdas, f"{n} actividades con distribuciones mixtas ({int((tipos == 'fija').sum())} fijas)"


def _convergencia(rng):
    """Cuántas réplicas hacen falta: error de la estimación frente a N."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 8 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1
                    and i["segunda"] is not None and i["T"] - i["segunda"] <= 2 and i["T"] >= 12,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    sd = P.desviaciones(rng, list(D.index))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    plazo = int(info["T"]) + 1
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas "
                 "(semanas, distribución normal) se adjunta. Se quiere saber cuántas réplicas de Monte Carlo hacen "
                 "falta para estimar con precisión la duración del proyecto.")]
    celdas.append(preguntas([
        "Construya la matriz de rutas y simule el proyecto con N = 20, 200, 2 000 y 20 000 réplicas. Anote en cada "
        f"caso la media, la desviación típica, el percentil 90 y la probabilidad de terminar en {plazo} semanas.",
        "Repita cada simulación 10 veces con semillas distintas y calcule la dispersión (desviación típica) de la "
        "media estimada entre repeticiones. ¿Cómo decrece con N?",
        "Compare esa dispersión con la fórmula del error típico de la media, σ/√N. ¿Cuántas réplicas hacen falta "
        "para que el error de la media sea menor de 0,1 semanas?",
        "¿Necesita el percentil 90 más o menos réplicas que la media para la misma precisión? ¿Por qué?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "mu = datos_proyecto.loc[R.columns, 'duration'].values[:, None]\n"
                       "sigma = datos_proyecto.loc[R.columns, 'standard_deviation'].values[:, None]\n"
                       f"plazo = {plazo}\n"
                       "def simular(N, semilla):\n"
                       "    rng = np.random.default_rng(semilla)\n"
                       "    T = np.clip(rng.normal(mu, sigma, (len(R.columns), N)), 0, None)\n"
                       "    return (R.values @ T).max(axis=0)"))
    celdas.append(code(_AUX_RUTAS))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias y el camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de rutas"\n#| label: tbl-matrizRutas\n'
                       f"proyecto.display_rutas('{ESTADO['grafo']}')"))
    celdas.append(code('#| tbl-cap: "Media y varianza de cada ruta (sumas de las de sus actividades)"\n#| label: tbl-rutas\n'
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "limpia_tabla(tabla_rutas(R.drop_duplicates(), datos_proyecto['duration'], datos_proyecto['variance'], calendario, dec_var=2), decimales=2)"))
    celdas.append(md("Cada réplica es un vector de duraciones normales; la duración de cada ruta es R·t y la del proyecto, "
                     "el máximo. Para cada tamaño N se repite el mismo cálculo con más columnas (semilla 0). La "
                     "probabilidad es la proporción de réplicas que no superan el plazo y el percentil 90, el valor que "
                     "deja por debajo el 90 % de las réplicas ordenadas."))
    celdas.append(code("tamanos = [20, 200, 2000, 20000]\n"
                       "filas = {}\n"
                       "for N in tamanos:\n"
                       "    tau = simular(N, 0)\n"
                       "    filas[N] = {'media': tau.mean(), 'desviación típica': tau.std(ddof=1), 'percentil 90': np.quantile(tau, 0.9), f'P(T ≤ {plazo})': (tau <= plazo).mean()}\n"
                       "resultados = pd.DataFrame(filas).T\n"
                       "resultados.index.name = 'N'\n"
                       "resultados.round(3)"))
    celdas.append(code("r = resultados\n"
                       "Markdown(f'''**Respuesta:** con N = 20 000 la media es {r.loc[20000, 'media']:.3f} semanas, la desviación típica {r.loc[20000, 'desviación típica']:.3f}, '''\n"
                       "         f'''el percentil 90 {r.loc[20000, 'percentil 90']:.3f} y P(T ≤ {plazo}) = {r.loc[20000, f'P(T ≤ {plazo})']:.3f}. Con N = 20 las estimaciones son poco fiables '''\n"
                       "         f'''(media {r.loc[20, 'media']:.3f}, σ {r.loc[20, 'desviación típica']:.3f}, P {r.loc[20, f'P(T ≤ {plazo})']:.3f}); entre N = 2 000 y N = 20 000 '''\n"
                       "         f'''la media sólo cambia {abs(r.loc[20000, 'media'] - r.loc[2000, 'media']):.3f} semanas y la probabilidad '''\n"
                       "         f'''{abs(r.loc[20000, f'P(T ≤ {plazo})'] - r.loc[2000, f'P(T ≤ {plazo})']):.3f}.''')"))
    celdas += apartado(2)
    celdas.append(md("Para cada N se repite la simulación con las semillas 1 a 10 y se calcula la desviación típica de "
                     "las 10 medias (y de los 10 percentiles 90) obtenidos: mide cuánto varía la estimación de una "
                     "simulación a otra."))
    celdas.append(code("repeticiones = 10\n"
                       "disp = {}\n"
                       "for N in tamanos:\n"
                       "    medias = [simular(N, s).mean() for s in range(1, repeticiones + 1)]\n"
                       "    p90 = [np.quantile(simular(N, s), 0.9) for s in range(1, repeticiones + 1)]\n"
                       "    disp[N] = {'dispersión de la media': np.std(medias, ddof=1), 'dispersión del percentil 90': np.std(p90, ddof=1)}\n"
                       "dispersion = pd.DataFrame(disp).T\n"
                       "dispersion.index.name = 'N'\n"
                       "dispersion.round(4)"))
    celdas.append(code("cocientes = [dispersion['dispersión de la media'].iloc[i] / dispersion['dispersión de la media'].iloc[i + 1] for i in range(len(tamanos) - 1)]\n"
                       "Markdown(f'''Cociente entre la dispersión de la media con N y con 10·N: ''' + '; '.join(f'{tamanos[i]} → {tamanos[i + 1]}: {c:.2f}' for i, c in enumerate(cocientes))\n"
                       "         + f'''.\\n\\n**Respuesta:** la dispersión de la media pasa de {dispersion[\"dispersión de la media\"].iloc[0]:.4f} (N = 20) a '''\n"
                       "         f'''{dispersion[\"dispersión de la media\"].iloc[-1]:.4f} (N = 20 000): al multiplicar N por 10 se divide aproximadamente por √10 ≈ 3.16 '''\n"
                       "         f'''(cociente medio {np.mean(cocientes):.2f}; con sólo 10 repeticiones cada cociente tiene bastante error), es decir, decrece como 1/√N.''')"))
    celdas += apartado(3)
    celdas.append(md("El error típico de la media de N réplicas independientes es $\\sigma_\\tau/\\sqrt{N}$, donde "
                     "$\\sigma_\\tau$ es la desviación típica de la duración del proyecto (se estima con las 20 000 "
                     "réplicas). Para que sea menor que un error $e$ hace falta $N \\ge (\\sigma_\\tau / e)^2$."))
    celdas.append(code("sigma_tau = simular(20000, 0).std(ddof=1)\n"
                       "dispersion['σ/√N'] = [sigma_tau / np.sqrt(N) for N in tamanos]\n"
                       "N_necesario = int(np.ceil((sigma_tau / 0.1) ** 2))\n"
                       "display(dispersion.round(4))\n"
                       "Markdown(f'''La columna σ/√N ({\", \".join(f\"{x:.4f}\" for x in dispersion[\"σ/√N\"])}) tiene el mismo orden de magnitud que la dispersión observada de la media '''\n"
                       "         f'''({\", \".join(f\"{x:.4f}\" for x in dispersion[\"dispersión de la media\"])}): la fórmula predice bien el error.\\n\\n'''\n"
                       "         f'''Con σ_τ = {sigma_tau:.3f}: N ≥ ({sigma_tau:.3f} / 0.1)² = {(sigma_tau / 0.1) ** 2:.1f}.\\n\\n'''\n"
                       "         f'''**Respuesta:** hacen falta al menos {N_necesario} réplicas para que el error típico de la media sea menor de 0,1 semanas.''')"))
    celdas += apartado(4)
    celdas.append(code("ratio = dispersion['dispersión del percentil 90'] / dispersion['dispersión de la media']\n"
                       "Markdown(f'''Cociente entre la dispersión del percentil 90 y la de la media: {\", \".join(f\"N = {N}: {x:.2f}\" for N, x in ratio.items())}.\\n\\n'''\n"
                       "         f'''**Respuesta:** el percentil 90 necesita más réplicas. Se estima con las réplicas de la cola, que son pocas (el 10 % de N), y su error es '''\n"
                       "         f'''unas {ratio.mean():.1f} veces el de la media para el mismo N; como el error decrece como 1/√N, para igualar la precisión de la media '''\n"
                       "         f'''harían falta unas {ratio.mean() ** 2:.1f} veces más réplicas. Por eso las decisiones basadas en percentiles altos (plazos comprometidos '''\n"
                       "         '''al 90–95 %) necesitan más réplicas que las basadas en la media.''')"))
    return celdas, f"{n} actividades; convergencia de la media y del percentil 90 con N"



def _correlacionadas(rng):
    """Un factor común (meteorología, un mismo equipo) correlaciona las duraciones: efecto sobre la duración del proyecto."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 8 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1
                    and i["T"] >= 12 and (i["D"] >= 2).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    T = int(info["T"])
    sd = P.desviaciones(rng, list(D.index))
    # actividades expuestas al factor común: la mitad, incluyendo al menos dos críticas
    criticas = [a for a in D.index if info["calendario"].loc[a, "H_total"] == 0]
    otras = [a for a in D.index if a not in criticas]
    expuestas = sorted(set(rng.choice(criticas, size=min(2, len(criticas)), replace=False))
                       | set(rng.choice(otras, size=max(1, len(otras) // 2), replace=False)))
    df = P.tabla_base(info, duration=D, standard_deviation=sd,
                      exposed=pd.Series({a: ("sí" if a in expuestas else "no") for a in D.index}))
    N = 10000
    semilla = int(rng.integers(1, 10000))
    sigma_f = 0.15
    plazo = T + 2
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas "
                 "(en semanas) se adjunta. Las actividades marcadas como *expuestas* se realizan al aire libre: "
                 "su duración depende además de un factor meteorológico común $F$ de media 1 y desviación "
                 f"típica {sigma_f} (si la temporada es mala, $F > 1$ y todas ellas se alargan a la vez). La "
                 "duración de una actividad expuesta es $F \\cdot X$, con $X$ normal de la media y desviación del "
                 "cuadro; las no expuestas son normales independientes.")]
    celdas.append(preguntas([
        "Calcule la duración del proyecto y el camino crítico con las duraciones medias.",
        f"Simule {N} réplicas suponiendo todas las duraciones independientes (ignorando $F$) y obtenga la media, "
        f"la desviación típica y la probabilidad de terminar en {plazo} semanas o menos.",
        "Repita la simulación con el factor común: en cada réplica se sortea un único valor de $F$ que "
        "multiplica a todas las actividades expuestas. Compare media, desviación típica, percentil 95 y "
        f"P(T ≤ {plazo}).",
        "¿Por qué la correlación positiva aumenta la dispersión de la duración del proyecto aunque apenas "
        "cambie la media? ¿Qué ocurre con la criticidad de las rutas?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"N, plazo, sigma_F = {N}, {plazo}, {sigma_f}\n"
                       "expuestas = datos_proyecto.index[datos_proyecto['exposed'] == 'sí'].tolist()"))
    celdas.append(code(_AUX_RUTAS))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas.append(code("path_matrix = proyecto.path_matrix(dummies=False).drop_duplicates()   # rutas que sólo difieren en ficticias, una vez\n"
                       "duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "criticas = ordena_ruta(sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)), calendario)\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "Markdown(f'**Respuesta:** duración media del proyecto {plural(duracion_media, \"semana\")}; camino crítico {\" – \".join(criticas)}. '\n"
                       "         f'Actividades expuestas al factor común: {lista_y(expuestas)}.')"))
    celdas.append(code('#| tbl-cap: "Duración media de cada ruta y actividades expuestas que contiene"\n#| label: tbl-rutas\n'
                       "rutas = tabla_rutas(path_matrix, datos_proyecto['duration'], calendario=calendario, nombre='duración media')\n"
                       "rutas['expuestas en la ruta'] = [', '.join(a for a in rutas.loc[r, 'actividades'].split(' – ') if a in expuestas) or 'ninguna' for r in path_matrix.index]\n"
                       "limpia_tabla(rutas)"))
    celdas += apartado(2)
    celdas.append(md("En cada réplica se sortea la duración de cada actividad con su normal (recortada en 0), se "
                     "suman las duraciones a lo largo de cada ruta y la duración del proyecto es la de la ruta más "
                     "larga. La probabilidad es la proporción de réplicas que no superan el plazo y el percentil 95, el "
                     "valor que deja por debajo el 95 % de las réplicas ordenadas."))
    celdas.append(code(f"rng = np.random.default_rng({semilla})\n"
                       "mu, sigma = datos_proyecto['duration'].astype(float), datos_proyecto['standard_deviation'].astype(float)\n"
                       "X = pd.DataFrame(rng.normal(mu.values, sigma.values, size=(N, len(mu))), columns=mu.index).clip(lower=0)\n"
                       "T_indep = (X[path_matrix.columns] @ path_matrix.T).max(axis='columns')\n"
                       "resumen_indep = pd.Series({'media': T_indep.mean(), 'desviación típica': T_indep.std(), 'percentil 95': T_indep.quantile(0.95),\n"
                       "                           f'P(T ≤ {plazo}) (%)': (T_indep <= plazo).mean() * 100})\n"
                       "resumen_indep.round(3).to_frame('independientes').T"))
    celdas.append(code("cumplen_indep = int((T_indep <= plazo).sum())\n"
                       "Markdown(f'''**Respuesta:** suponiendo independencia, media {T_indep.mean():.3f} semanas, desviación típica {T_indep.std():.3f} y '''\n"
                       "         f'''P(T ≤ {plazo}) ≈ {cumplen_indep}/{N} = {cumplen_indep / N:.4f} ({cumplen_indep} de las {N} réplicas terminan a tiempo).''')"))
    celdas += apartado(3)
    celdas.append(md("Se reutilizan las mismas duraciones normales $X$ y se multiplica cada fila (réplica) de las "
                     "actividades expuestas por un valor de $F$ común a toda la fila. Así la única diferencia "
                     "entre las dos simulaciones es la correlación introducida por $F$."))
    celdas.append(code("F = rng.normal(1.0, sigma_F, size=N).clip(min=0.5)\n"
                       "Xc = X.copy()\n"
                       "Xc[expuestas] = Xc[expuestas].mul(F, axis='rows')\n"
                       "duraciones_rutas_c = Xc[path_matrix.columns] @ path_matrix.T\n"
                       "T_corr = duraciones_rutas_c.max(axis='columns')\n"
                       "resumen_corr = pd.Series({'media': T_corr.mean(), 'desviación típica': T_corr.std(), 'percentil 95': T_corr.quantile(0.95),\n"
                       "                          f'P(T ≤ {plazo}) (%)': (T_corr <= plazo).mean() * 100})\n"
                       "pd.DataFrame({'independientes': resumen_indep, 'con factor común': resumen_corr}).round(3)"))
    celdas.append(code('#| tbl-cap: "Correlación entre las duraciones de dos actividades expuestas"\n#| label: tbl-correlacion\n'
                       "a1, a2 = expuestas[:2]\n"
                       "pd.DataFrame({'independientes': [X[a1].corr(X[a2])], 'con factor común': [Xc[a1].corr(Xc[a2])]}, index=[f'corr({a1}, {a2})']).round(3)"))
    celdas.append(code("Markdown(f'''**Respuesta:** con el factor común la media pasa de {resumen_indep[\"media\"]:.3f} a {resumen_corr[\"media\"]:.3f} semanas '''\n"
                       "         f'''(apenas cambia), la desviación típica de {resumen_indep[\"desviación típica\"]:.3f} a {resumen_corr[\"desviación típica\"]:.3f}, '''\n"
                       "         f'''el percentil 95 de {resumen_indep[\"percentil 95\"]:.2f} a {resumen_corr[\"percentil 95\"]:.2f} y P(T ≤ {plazo}) de '''\n"
                       "         f'''{resumen_indep[f\"P(T ≤ {plazo}) (%)\"]:.2f} % a {resumen_corr[f\"P(T ≤ {plazo}) (%)\"]:.2f} %. La correlación entre {a1} y {a2} '''\n"
                       "         f'''pasa de {X[a1].corr(X[a2]):.3f} a {Xc[a1].corr(Xc[a2]):.3f}.''')"))
    celdas.append(code('#| fig-cap: "Distribución de la duración con y sin factor común"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "ax.hist(T_indep, bins=60, density=True, alpha=0.5, label='independientes')\n"
                       "ax.hist(T_corr, bins=60, density=True, alpha=0.5, label='con factor común')\n"
                       "ax.axvline(plazo, color='red', linestyle='--', label=f'plazo ({plazo})')\n"
                       "ax.set_xlabel('Duración del proyecto (semanas)'); ax.legend(); plt.show()"))
    celdas += apartado(4)
    celdas.append(md("La varianza de una suma de variables correlacionadas positivamente es mayor que la suma de "
                     "sus varianzas (se añaden los términos $2\\,\\mathrm{cov}$): cuando $F$ sale alto, todas las "
                     "actividades expuestas se alargan a la vez y no se compensan unas con otras. La media casi "
                     "no cambia porque la de $F$ es 1, pero las colas se ensanchan."))
    celdas.append(code('#| tbl-cap: "Índice de criticidad de las rutas sin y con factor común"\n#| label: tbl-criticidad\n'
                       "ruta_critica_mc = duraciones_rutas_c.idxmax(axis='columns')\n"
                       "criticidad_corr = (ruta_critica_mc.value_counts(normalize=True) * 100).reindex(path_matrix.index, fill_value=0.0)\n"
                       "criticidad_indep = ((X[path_matrix.columns] @ path_matrix.T).idxmax(axis='columns').value_counts(normalize=True) * 100).reindex(path_matrix.index, fill_value=0.0)\n"
                       "pd.DataFrame({'actividades': rutas['actividades'], 'expuestas': rutas['expuestas en la ruta'],\n"
                       "              'criticidad indep. (%)': criticidad_indep, 'criticidad con F (%)': criticidad_corr}).round(1)"))
    celdas.append(code("cambio = (criticidad_corr - criticidad_indep).round(1)\n"
                       "ganan = [r for r in cambio.index if cambio[r] >= 0.1]\n"
                       "pierden = [r for r in cambio.index if cambio[r] <= -0.1]\n"
                       "fmt_r = lambda rs: lista_y([f\"{rutas.loc[r, 'actividades']} ({cambio[r]:+.1f} puntos)\" for r in rs])\n"
                       "if ganan or pierden:\n"
                       "    t_crit = ('Con el factor común ' + (f'ganan criticidad {fmt_r(ganan)}' if ganan else '')\n"
                       "              + ('; ' if ganan and pierden else '') + (f'pierden criticidad {fmt_r(pierden)}' if pierden else '') + '. '\n"
                       "              'Lo que cuenta no es cuántas actividades expuestas tiene cada ruta, sino cuánto de su duración depende de F en '\n"
                       "              'comparación con la ruta crítica: cuando F sale alto se alargan más las rutas con más semanas expuestas, y cuando sale bajo, al revés.')\n"
                       "else:\n"
                       "    t_crit = 'La criticidad de las rutas prácticamente no cambia: la ruta crítica sigue dominando en casi todas las réplicas.'\n"
                       "Markdown(f'''**Respuesta:** la correlación positiva hace que las actividades expuestas se retrasen (o adelanten) a la vez, sin compensarse: '''\n"
                       "         f'''la desviación típica sube de {resumen_indep[\"desviación típica\"]:.3f} a {resumen_corr[\"desviación típica\"]:.3f} mientras la media '''\n"
                       "         f'''apenas cambia ({resumen_indep[\"media\"]:.3f} frente a {resumen_corr[\"media\"]:.3f}), porque F tiene media 1. {t_crit} Suponer independencia, '''\n"
                       "         '''como hace el PERT clásico, es optimista en presencia de causas comunes de retraso.''')"))
    return celdas, f"{n} actividades; {len(expuestas)} expuestas a un factor común (σ_F = {sigma_f})"


def _percentiles(rng):
    """Leer la distribución simulada: percentiles, plazo para una probabilidad y plazo con margen."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 8 and 1 <= i["n_dummies"] <= 4
                    and i["segunda"] is not None and i["T"] - i["segunda"] <= 2 and i["T"] >= 12 and (i["D"] >= 3).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    T = int(info["T"])
    a, m, b = P.tres_estimaciones_triangular(rng, D)
    df = P.tabla_base(info, a=a, m=m, b=b)
    N = 20000
    semilla = int(rng.integers(1, 10000))
    plazo_cliente = T + int(rng.integers(2, 4))
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y estimaciones optimista, más probable y "
                 "pesimista (en semanas) se adjunta; las duraciones siguen distribuciones triangulares "
                 f"independientes. El cliente ha fijado la entrega en la semana {plazo_cliente} con una "
                 "penalización por cada semana de retraso.")]
    celdas.append(preguntas([
        "Calcule la duración del proyecto con las duraciones medias de las triangulares, "
        "$(a + m + b)/3$, y el camino crítico.",
        f"Simule {N} réplicas con la matriz de caminos y obtenga los percentiles 10, 50, 80, 90 y 95 de la "
        "duración. ¿Qué significa cada uno para el jefe de proyecto?",
        f"¿Qué probabilidad hay de pagar penalización? ¿Cuántas semanas de retraso se superan sólo con "
        "probabilidad 0,05? Calcule el retraso medio esperado (media de máx(0, T − plazo)).",
        "¿Qué fecha habría que haber comprometido para que la probabilidad de penalización fuera del 10 %? "
        "Compare con la duración media: ¿cuántas semanas de *margen* supone?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['duration'] = datos_proyecto.eval('(a + m + b) / 3')\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"N, plazo = {N}, {plazo_cliente}"))
    celdas.append(code(_AUX_RUTAS))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Media de cada triangular"\n#| label: tbl-medias\n'
                       "pd.DataFrame({'(a + m + b)/3': [f\"({r['a']:g} + {r['m']:g} + {r['b']:g})/3 = {r['duration']:.2f}\" for _, r in datos_proyecto.iterrows()]}, index=datos_proyecto.index).T"))
    celdas += fig_grafo(durations="datos_proyecto['duration'].round(1)", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Duración media de cada ruta"\n#| label: tbl-rutas\n'
                       "path_matrix = proyecto.path_matrix(dummies=False).drop_duplicates()   # rutas que sólo difieren en ficticias, una vez\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duraciones_rutas_medias = path_matrix @ datos_proyecto['duration'][path_matrix.columns]\n"
                       "duracion_media = float(duraciones_rutas_medias.max())\n"
                       "rutas = tabla_rutas(path_matrix, datos_proyecto['duration'], calendario=calendario, dec=2, nombre='duración media')\n"
                       "limpia_tabla(rutas)"))
    celdas.append(code("criticas_txt = [rutas.loc[r, 'actividades'] for r in rutas.index if np.isclose(rutas.loc[r, 'duración media'], duracion_media)]\n"
                       "Markdown(f'**Respuesta:** con las medias de las triangulares la duración del proyecto es {duracion_media:.2f} semanas y el camino crítico '\n"
                       "         f'{lista_y(criticas_txt)}.')"))
    celdas += apartado(2)
    celdas.append(md("En cada réplica se sortea la duración de cada actividad con su triangular, se suman las "
                     "duraciones a lo largo de cada ruta y la duración del proyecto es la de la ruta más larga. "
                     "El percentil $p$ de las réplicas es el valor que deja por debajo la proporción $p$ de ellas: "
                     "se ordenan de menor a mayor y se toma la que ocupa la posición $p\\cdot N$ (interpolando)."))
    celdas.append(code(f"rng = np.random.default_rng({semilla})\n"
                       "duraciones_mc = pd.DataFrame({act: rng.triangular(fila['a'], fila['m'], fila['b'], N) for act, fila in datos_proyecto.iterrows()})\n"
                       "duraciones_rutas_mc = duraciones_mc[path_matrix.columns] @ path_matrix.T\n"
                       "T_mc = duraciones_rutas_mc.max(axis='columns')\n"
                       "percentiles = T_mc.quantile([0.10, 0.50, 0.80, 0.90, 0.95])\n"
                       "percentiles.index = [f'P{int(round(p * 100))}' for p in percentiles.index]\n"
                       "pd.DataFrame({'posición en la muestra ordenada': [f'≈ {int(round(p * N))} de {N}' for p in (0.10, 0.50, 0.80, 0.90, 0.95)],\n"
                       "              'duración (semanas)': percentiles.round(2).values}, index=percentiles.index).T"))
    celdas.append(code('#| fig-cap: "Función de distribución simulada de la duración"\n#| label: fig-distribucion\n'
                       "from matplotlib import pyplot as plt\n"
                       "x = np.sort(T_mc.values); y = np.arange(1, N + 1) / N\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "ax.plot(x, y)\n"
                       "for nombre, v in percentiles.items():\n"
                       "    ax.axvline(v, color='grey', linestyle=':', alpha=0.7); ax.text(v, 0.02, nombre, rotation=90, va='bottom')\n"
                       "ax.axvline(plazo, color='red', linestyle='--', label=f'plazo del cliente ({plazo})')\n"
                       "ax.set_xlabel('Duración (semanas)'); ax.set_ylabel('P(T ≤ t)'); ax.legend(); plt.show()"))
    celdas.append(code("pc = percentiles\n"
                       "Markdown(f'''**Respuesta:** P10 = {pc['P10']:.2f}, P50 = {pc['P50']:.2f}, P80 = {pc['P80']:.2f}, P90 = {pc['P90']:.2f} y P95 = {pc['P95']:.2f} semanas. '''\n"
                       "         f'''Para el jefe de proyecto, P10 es una fecha optimista (sólo 1 de cada 10 réplicas termina antes); P50, la mediana (tan probable '''\n"
                       "         f'''acabar antes como después); P80, una fecha que se cumple cuatro de cada cinco veces; P90 y P95, fechas «seguras» que sólo se '''\n"
                       "         f'''superan el 10 % y el 5 % de las veces. La diferencia P95 − P50 = {pc['P95'] - pc['P50']:.2f} semanas es el margen que hay que '''\n"
                       "         '''añadir a una previsión central para hacerla fiable.''')"))
    celdas += apartado(3)
    celdas.append(md("Se paga penalización en las réplicas que superan el plazo del cliente; el retraso de cada réplica "
                     "es $\\max(0, T - \\text{plazo})$ y se estudia como cualquier otra variable simulada."))
    celdas.append(code("superan = int((T_mc > plazo).sum())\n"
                       "p_penalizacion = float((T_mc > plazo).mean())\n"
                       "retraso = (T_mc - plazo).clip(lower=0)\n"
                       "retraso_95 = float(retraso.quantile(0.95))\n"
                       "retraso_medio = float(retraso.mean())\n"
                       "Markdown(f'''P(T > {plazo}) ≈ {superan}/{N} = {p_penalizacion:.4f}: se paga penalización en el {p_penalizacion * 100:.1f} % de las réplicas. '''\n"
                       "         f'''El percentil 95 del retraso es {retraso_95:.2f} semanas{\" (= P95 − plazo = \" + f\"{pc['P95']:.2f} − {plazo}\" + \")\" if retraso_95 > 0 else \" (el plazo ya es mayor que el P95)\"}. '''\n"
                       "         f'''El retraso medio es la media de las {N} réplicas de máx(0, T − {plazo}) (las que terminan a tiempo cuentan 0): {retraso.sum():.1f}/{N} = {retraso_medio:.3f} semanas.\\n\\n'''\n"
                       "         f'''**Respuesta:** probabilidad de pagar penalización {p_penalizacion:.4f}; el retraso supera {retraso_95:.2f} semanas sólo con '''\n"
                       "         f'''probabilidad 0,05; retraso medio esperado {retraso_medio:.3f} semanas (multiplicado por la penalización semanal da el coste esperado de la cláusula).''')"))
    celdas += apartado(4)
    celdas.append(code("fecha_10 = float(T_mc.quantile(0.90))\n"
                       "fecha_entera = int(np.ceil(round(fecha_10, 2)))\n"
                       "p_entera = float((T_mc > fecha_entera).mean())\n"
                       "p_media = float((T_mc > duracion_media).mean())\n"
                       "Markdown(f'''Para que P(T > fecha) = 0,10 hay que comprometer el percentil 90 de las réplicas: {fecha_10:.2f} semanas, es decir, semana '''\n"
                       "         f'''{fecha_entera} en semanas enteras (P(T > {fecha_entera}) ≈ {p_entera:.3f}). Comprometer la media ({duracion_media:.2f}) supondría '''\n"
                       "         f'''penalización en el {p_media:.1%} de los casos'''\n"
                       "         + (', algo más de la mitad por el sesgo de fusión de las rutas.' if p_media > 0.5 else '.') + '''\\n\\n'''\n"
                       "         f'''**Respuesta:** habría que haber comprometido {fecha_10:.2f} semanas (semana {fecha_entera}), con un margen de '''\n"
                       "         f'''{fecha_10:.2f} − {duracion_media:.2f} = {fecha_10 - duracion_media:.2f} semanas sobre la duración media.''')"))
    return celdas, f"{n} actividades; plazo del cliente {plazo_cliente}; percentiles y retraso esperado"


def _recursos_limitados(rng):
    """Monte Carlo con un límite de operarios: la duración con recursos es mayor y más dispersa que la libre."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (0 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 24 and (i["D"] >= 2).all()):
            return False
        nombres = list(i["D"].index)
        op = P.recursos(rng, nombres, hi=3).rename("operarios")
        L = int(op.max())
        fa, fb = {}, {}
        for a in nombres:
            d = int(i["D"][a])
            fa[a] = max(1, d - int(rng.integers(1, 3)))
            fb[a] = d + int(rng.integers(1, 4))
        df = P.tabla_base(i, a=pd.Series(fa), m=i["D"], b=pd.Series(fb), operarios=op)
        dists = {a: ("triangular", fa[a], int(i["D"][a]), fb[a]) for a in nombres}
        try:
            r = i["proyecto"].copy().monte_carlo_recursos(dists, df[["operarios"]].copy(),
                                                         ["operarios"], [L], n_samples=300, seed=5)
        except Exception:
            return False
        libre = r.tau_libre_samples
        plazo = int(np.ceil(np.quantile(libre, 0.85)))
        if not (r.prob_superar(plazo, con_recursos=False) <= 0.18 and r.prob_superar(plazo) >= 0.55):
            return False
        estado.update(df=df, L=L, plazo=plazo)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])),
                    max_intentos=8000)
    df, L, plazo = estado["df"], estado["L"], estado["plazo"]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duración de cada actividad (mínima *a*, más probable *m* "
                 "y máxima *b*, en semanas, distribución triangular) y operarios necesarios se adjunta. Cada actividad "
                 f"necesita un número fijo de operarios durante toda su duración y la empresa dispone como máximo de {L} "
                 f"operarios por semana. El contrato exige terminar en **{plazo} semanas**.")]
    celdas.append(preguntas([
        "Calcule con las duraciones más probables la duración del proyecto sin límite de recursos y con el límite de "
        "operarios (algoritmo de asignación).",
        "Simule 2000 réplicas: en cada una sortee las duraciones, calcule la duración sin recursos y **replanifique** "
        "con el límite. Compare media, desviación típica y percentil 90 de ambas distribuciones.",
        f"Estime la probabilidad de cumplir el plazo de {plazo} semanas ignorando los recursos y respetándolos. "
        "¿Qué conclusión sacaría un director que solo mirase el camino crítico?",
        "Calcule el índice de criticidad de cada actividad sin y con recursos, y la frecuencia con que el límite de "
        "operarios retrasa alguna actividad. ¿Qué actividades pasan a mandar al introducir el límite?",
        "¿Cómo cambia la probabilidad de cumplir si se amplía el límite en un operario? Decida si merece la pena."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"limite = [{L}]\n"
                       f"plazo = {plazo}\n"
                       "datos_proyecto['duration'] = datos_proyecto['m']\n"
                       "demanda = datos_proyecto[['predecessor', 'duration', 'operarios']]\n"
                       "dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones más probables y el camino crítico (sin límite)", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario sin límite de recursos"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_libre = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Carga de operarios con los inicios más tempranos frente al límite"\n#| label: fig-cargasInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(demanda.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite)"))
    celdas.append(md("El algoritmo de asignación recorre las semanas: cada actividad disponible (predecesoras terminadas) "
                     "empieza, por orden de prioridad (menor inicio más tardío), si caben sus operarios durante toda su "
                     "duración; si no, espera. Traza semana a semana y Gantt resultante (filas 💤: esperas):"))
    celdas.append(code("res0 = proyecto.copy().asignar_multi(demanda.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt con el límite de operarios y las duraciones más probables"\n#| label: fig-ganttAsignado\n'
                       "pinta_tikz(res0.dibujos['operarios'])"))
    celdas.append(code('#| tbl-cap: "Inicio de cada actividad sin y con límite"\n#| label: tbl-inicios\n'
                       "pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res0.inicios, 'fin con límite': res0.fines,\n"
                       "              'espera (semanas)': res0.esperas}).fillna(0).astype(int).T"))
    celdas.append(code("Markdown(f'**Respuesta:** con las duraciones más probables el proyecto dura {plural(duracion_libre, \"semana\")} sin límite de recursos '\n"
                       "         f'y {plural(res0.duracion, \"semana\")} con {limite[0]} operarios (las esperas de la tabla anterior lo retrasan {res0.duracion - duracion_libre}).')"))
    celdas += apartado(2)
    celdas.append(md("En cada réplica se sortean las duraciones (triangulares, redondeadas a semanas enteras), se calcula "
                     "la duración sin recursos (camino más largo) y se vuelve a aplicar el algoritmo de asignación con el "
                     "límite: dos duraciones por réplica. `monte_carlo_recursos` hace las 2000 réplicas y guarda ambas. "
                     "Los percentiles son los de las réplicas ordenadas (el P90 deja por debajo el 90 % de ellas)."))
    celdas.append(code('#| tbl-cap: "Distribución de la duración con y sin límite de recursos"\n#| label: tbl-mcRecursos\n'
                       "mc = proyecto.copy().monte_carlo_recursos(dists, demanda[['operarios']].copy(), ['operarios'], limite, n_samples=2000, seed=11)\n"
                       "resumen_mc = mc.summary().rename(columns={'q05': 'P5', 'q50': 'P50', 'q95': 'P95'})\n"
                       "resumen_mc.insert(4, 'P90', [np.quantile(mc.tau_libre_samples, 0.9), np.quantile(mc.tau_recursos_samples, 0.9), np.quantile(mc.retraso_samples, 0.9)])\n"
                       "resumen_mc.round(2)"))
    celdas.append(code('#| fig-cap: "Distribución simulada de la duración sin y con el límite de operarios"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "bins = np.arange(min(mc.tau_libre_samples.min(), mc.tau_recursos_samples.min()) - 0.5, max(mc.tau_libre_samples.max(), mc.tau_recursos_samples.max()) + 1.5, 1)\n"
                       "ax.hist(mc.tau_libre_samples, bins=bins, alpha=0.5, label='sin límite')\n"
                       "ax.hist(mc.tau_recursos_samples, bins=bins, alpha=0.5, label=f'con {limite[0]} operarios')\n"
                       "ax.axvline(plazo, color='red', linestyle='--', label=f'plazo ({plazo})')\n"
                       "ax.set_xlabel('duración del proyecto (semanas)'); ax.set_ylabel('réplicas'); ax.legend(); plt.show()"))
    celdas.append(code("rl = resumen_mc\n"
                       "Markdown(f'''**Respuesta:** la media sube de {rl.loc['sin recursos', 'media']:.2f} a {rl.loc['con recursos', 'media']:.2f} semanas, '''\n"
                       "         f'''la desviación típica pasa de {rl.loc['sin recursos', 'desv. típica']:.2f} a {rl.loc['con recursos', 'desv. típica']:.2f} y el percentil 90 de '''\n"
                       "         f'''{rl.loc['sin recursos', 'P90']:.0f} a {rl.loc['con recursos', 'P90']:.0f} semanas. El límite añade esperas (de media '''\n"
                       "         f'''{rl.loc['retraso por recursos', 'media']:.2f} semanas) que dependen de qué actividades coinciden en cada réplica.''')"))
    celdas += apartado(3)
    celdas.append(code("p_libre = 1 - mc.prob_superar(plazo, con_recursos=False)\n"
                       "p_res = 1 - mc.prob_superar(plazo)\n"
                       "n_libre = int((mc.tau_libre_samples <= plazo).sum()); n_res = int((mc.tau_recursos_samples <= plazo).sum())\n"
                       "Markdown(f'''Sin recursos, {n_libre} de las 2000 réplicas terminan en {plazo} semanas o menos: P ≈ {n_libre}/2000 = {p_libre:.3f}. '''\n"
                       "         f'''Con el límite, {n_res} de 2000: P ≈ {p_res:.3f}.\\n\\n'''\n"
                       "         f'''**Respuesta:** P(T ≤ {plazo}) = {p_libre:.3f} ignorando los recursos y {p_res:.3f} respetándolos. Un director que sólo mirase '''\n"
                       "         f'''el camino crítico creería que cumple el plazo {p_libre:.0%} de las veces, cuando con {limite[0]} operarios sólo lo cumple el '''\n"
                       "         f'''{p_res:.1%}: el riesgo real de incumplir es {1 - p_res:.1%} y no {1 - p_libre:.1%}.''')"))
    celdas += apartado(4)
    celdas.append(md("El índice de criticidad de una actividad es el porcentaje de réplicas en que determina la duración: "
                     "sin recursos, si está en la ruta más larga; con recursos, si forma parte de la cadena de actividades "
                     "y esperas por el recurso que fija el final del proyecto."))
    celdas.append(code('#| tbl-cap: "Criticidad sin y con recursos"\n#| label: tbl-criticidadRecursos\n'
                       "criticidad = (mc.criticidad * 100).round(1)\n"
                       "criticidad['cambio'] = criticidad.iloc[:, -1] - criticidad.iloc[:, 0]\n"
                       "criticidad.sort_values(criticidad.columns[1], ascending=False)"))
    celdas.append(code('#| tbl-cap: "Frecuencia con que cada recurso bloquea alguna actividad"\n#| label: tbl-bloqueos\n'
                       "(mc.bloqueos * 100).round(1).to_frame('% de réplicas')"))
    celdas.append(code("pasan = criticidad[criticidad['cambio'] >= 20].sort_values('cambio', ascending=False)\n"
                       "bajan = criticidad[criticidad['cambio'] <= -20].sort_values('cambio')\n"
                       "fmt = lambda t: lista_y([f\"{a} ({t.loc[a].iloc[0]:.1f} % → {t.loc[a].iloc[1]:.1f} %)\" for a in t.index])\n"
                       "Markdown(f'''**Respuesta:** el límite de {limite[0]} operarios retrasa alguna actividad en el {float(mc.bloqueos.iloc[0]) * 100:.1f} % de las réplicas. '''\n"
                       "         + (f'''Pasan a mandar (su índice sube 20 puntos o más) {fmt(pasan)}: en el CPM no siempre estaban en la ruta más larga, pero con el límite forman parte de la cadena de esperas por el recurso que fija el final del proyecto. ''' if len(pasan) else\n"
                       "            'Ninguna actividad gana más de 20 puntos de criticidad. ')\n"
                       "         + (f'''En cambio pierden peso {fmt(bajan)}.''' if len(bajan) else ''))"))
    celdas += apartado(5)
    celdas.append(code("mc2 = proyecto.copy().monte_carlo_recursos(dists, demanda[['operarios']].copy(), ['operarios'], [limite[0] + 1], n_samples=2000, seed=11)\n"
                       "p_res2 = 1 - mc2.prob_superar(plazo)\n"
                       "if p_res2 >= 0.8:\n"
                       "    decision = (f'Con un operario más el plazo pasa a ser razonablemente seguro ({p_res2:.1%}): merece la pena si su coste es menor que '\n"
                       "                'lo que se espera ahorrar en penalizaciones.')\n"
                       "elif p_res2 - p_res >= 0.2:\n"
                       "    decision = (f'Un operario más mejora mucho la probabilidad (+{(p_res2 - p_res) * 100:.1f} puntos), aunque el plazo sigue sin ser seguro '\n"
                       "                f'({p_res2:.1%}): merece la pena si su coste es menor que las penalizaciones evitadas, y conviene combinarlo con otras medidas.')\n"
                       "else:\n"
                       "    decision = (f'Un operario más apenas mejora la probabilidad (+{(p_res2 - p_res) * 100:.1f} puntos, hasta {p_res2:.1%}): por sí solo no resuelve '\n"
                       "                'el problema y sólo merecería la pena si es muy barato; hay que buscar otras medidas (más recursos, acortar actividades críticas o '\n"
                       "                'renegociar el plazo).')\n"
                       "Markdown(f'''Con {limite[0] + 1} operarios P(T ≤ {plazo}) pasa de {p_res:.3f} a {p_res2:.3f}; la media baja de '''\n"
                       "         f'''{mc.tau_recursos_samples.mean():.2f} a {mc2.tau_recursos_samples.mean():.2f} semanas.\\n\\n**Respuesta:** {decision}''')"))
    return celdas, f"{n} actividades triangulares; límite {L} operarios, plazo {plazo}"


VARIANTES = [
    Variante("normal", 1, "Monte Carlo con R·T, comparación con la aproximación normal, criticidad y sensibilidad (normal)",
             lambda rng: _base(1, rng)),
    Variante("triangular", 2, "ídem con duraciones triangulares (a, m, b)", lambda rng: _base(2, rng)),
    Variante("beta-pert", 2, "ídem con duraciones Beta-PERT", lambda rng: _base(3, rng)),
    Variante("a-mano", 1, "cinco réplicas a mano: R·T, máximo por columnas, media, desviación y criticidad", _a_mano),
    Variante("mixtas", 2, "distribuciones distintas por actividad (incluidas fijas) con monte_carlo de la librería", _mixtas),
    Variante("convergencia", 3, "cuántas réplicas hacen falta: error típico de la media y del percentil 90", _convergencia),
    Variante("correlacionadas", 3, "un factor común correlaciona las duraciones expuestas: misma media, más dispersión, menos probabilidad de cumplir",
             _correlacionadas),
    Variante("percentiles", 2, "leer la distribución simulada: percentiles, probabilidad de penalización, retraso esperado y margen a comprometer",
             _percentiles),
    Variante("recursos-limitados", 3, "Monte Carlo con un límite de operarios: la duración con recursos es mayor y más dispersa; criticidad y bloqueos",
             _recursos_limitados),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
