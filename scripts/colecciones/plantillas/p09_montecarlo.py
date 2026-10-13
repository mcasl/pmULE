"""Colección 09 — Método de Monte Carlo: simulación con la matriz de caminos,
comparación con la aproximación normal, índices de criticidad."""
import numpy as np

from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

META = dict(slug="09_montecarlo", prefijo="MonteCarlo",
            titulo="Método de Monte Carlo",
            descripcion="Simular la duración del proyecto con la matriz de caminos, compararla con la "
                        "aproximación normal y calcular índices de criticidad y sensibilidad.",
            n_defecto=3)

DISTRIBUCIONES = {1: "normal", 2: "triangular", 3: "pert"}


def generar(k, rng):
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
    pregs = ["Dibuje el grafo PERT, construya la matriz de caminos y calcule la duración del proyecto con las "
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
        celdas.append(md("Para la distribución triangular la media es $(a + m + b)/3$ y la varianza "
                         "$(a^2 + m^2 + b^2 - ab - am - bm)/18$."))
        celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                           "datos_proyecto['duration'] = datos_proyecto.eval('(a + m + b) / 3').round().astype(int)\n"
                           "datos_proyecto['variance'] = datos_proyecto.eval('(a**2 + m**2 + b**2 - a*b - a*m - b*m) / 18')\n"
                           "datos_proyecto['standard_deviation'] = np.sqrt(datos_proyecto['variance'])\n"
                           "datos_proyecto"))
    else:
        celdas.append(md("Con el método PERT la media de cada actividad es $(a + 4m + b)/6$ y su desviación "
                         "típica se aproxima por $(b - a)/6$."))
        celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                           "datos_proyecto['duration'] = datos_proyecto.eval('(a + 4 * m + b) / 6').round().astype(int)\n"
                           "datos_proyecto['standard_deviation'] = datos_proyecto.eval('(b - a) / 6')\n"
                           "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                           "datos_proyecto"))
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| fig-cap: "Grafo PERT con las duraciones medias"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas.append(code('#| tbl-cap: "Matriz de caminos"\n#| label: tbl-matrizCaminos\n'
                       "path_matrix = proyecto.path_matrix(dummies=False)\n"
                       "proyecto.display_path_matrix(dummies=False)"))
    celdas.append(code("duraciones = datos_proyecto['duration'][path_matrix.columns]\n"
                       "duraciones_rutas_medias = path_matrix @ duraciones\n"
                       "duracion_media = int(duraciones_rutas_medias.max())\n"
                       "ruta_critica = duraciones_rutas_medias.idxmax()\n"
                       "desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       "pd.concat([duraciones_rutas_medias.rename('duración media'),\n"
                       "           (path_matrix @ datos_proyecto['variance'][path_matrix.columns]).rename('varianza')], axis=1)"))
    celdas.append(code("Markdown(f'Con las duraciones medias el proyecto dura {duracion_media} semanas; la ruta crítica es {ruta_critica} y su desviación típica {desviacion_proyecto:.3f}.')"))
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
    celdas.append(code('#| tbl-cap: "Medias y desviaciones típicas muestrales frente a las teóricas"\n#| label: tbl-comprobacionMuestras\n'
                       "pd.DataFrame({'media teórica': datos_proyecto['duration'], 'media muestral': duraciones_mc.mean(),\n"
                       "              'σ teórica': datos_proyecto['standard_deviation'], 'σ muestral': duraciones_mc.std()}).round(2).T"))
    if dist == "pert":
        celdas.append(md("Las medias muestrales coinciden con las teóricas. Las desviaciones típicas salen algo "
                         "mayores que $(b-a)/6$: esa fórmula es una aproximación; la desviación exacta de la "
                         "Beta-PERT simétrica es $(b-a)/\\sqrt{28} \\approx (b-a)/5.3$."))
    celdas.append(md("La duración de cada ruta en cada réplica es el producto de las duraciones por la "
                     "traspuesta de la matriz de caminos; la del proyecto, el máximo por filas."))
    celdas.append(code("duraciones_rutas_mc = duraciones_mc @ path_matrix.T\n"
                       "duracion_proyecto_mc = duraciones_rutas_mc.max(axis='columns')\n"
                       "pd.concat([duraciones_rutas_mc, duracion_proyecto_mc.rename('Proyecto')], axis=1).head().round(2)"))
    celdas.append(code('#| tbl-cap: "Descriptores de la duración simulada del proyecto"\n#| label: tbl-descriptores\n'
                       "duracion_proyecto_mc.describe(percentiles=[0.05, 0.5, 0.95]).round(2).to_frame('Duración del proyecto').T"))
    celdas.append(code('#| fig-cap: "Distribución simulada de la duración frente a la aproximación normal"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       f"ax.hist(duracion_proyecto_mc, bins=60, density=True, alpha=0.5, color='skyblue', label='Monte Carlo ({nombre_dist})')\n"
                       "x = np.linspace(duracion_proyecto_mc.min(), duracion_proyecto_mc.max(), 300)\n"
                       "ax.plot(x, norm.pdf(x, duracion_media, desviacion_proyecto), color='navy', label='Aproximación normal (camino crítico)')\n"
                       f"ax.axvline({plazo}, color='red', linestyle='--', label='Plazo ({plazo})')\n"
                       "ax.set_xlabel('Duración del proyecto (semanas)'); ax.legend(); plt.show()"))
    celdas += apartado(ap); ap += 1
    celdas.append(code(f"plazo, probabilidad_objetivo = {plazo}, {prob_obj}\n"
                       "p_mc = float((duracion_proyecto_mc <= plazo).mean())\n"
                       "t_mc = float(duracion_proyecto_mc.quantile(probabilidad_objetivo))\n"
                       "p_normal = float(norm.cdf((plazo - duracion_media) / desviacion_proyecto))\n"
                       "t_normal = float(duracion_media + norm.ppf(probabilidad_objetivo) * desviacion_proyecto)\n"
                       "pd.DataFrame({'Aproximación normal': [duracion_media, desviacion_proyecto, p_normal * 100, t_normal],\n"
                       "              'Monte Carlo': [duracion_proyecto_mc.mean(), duracion_proyecto_mc.std(), p_mc * 100, t_mc]},\n"
                       "             index=['Duración media', 'Desviación típica', f'P(T ≤ {plazo}) (%)', f'Duración para {int(probabilidad_objetivo*100)}%']).round(2)"))
    celdas.append(md("La aproximación normal sólo mira la ruta crítica. En cada réplica, sin embargo, la "
                     "duración del proyecto es el máximo de todas las rutas, y el máximo de varias variables "
                     "aleatorias tiene una media mayor que cualquiera de ellas: las rutas casi críticas empujan "
                     "la duración hacia arriba (*sesgo de fusión*). Por eso Monte Carlo da una duración media "
                     "algo mayor y una probabilidad de cumplir el plazo algo menor: el método PERT clásico es "
                     "optimista. La desviación típica simulada suele ser menor, porque cuando una ruta sale "
                     "corta otra ocupa su lugar y la cola izquierda se recorta."))
    celdas += apartado(ap); ap += 1
    celdas.append(md("El índice de criticidad de una ruta es la fracción de réplicas en que es la más larga; "
                     "el de una actividad, la fracción de réplicas en que pertenece a la ruta más larga."))
    celdas.append(code('#| tbl-cap: "Índice de criticidad de las rutas"\n#| label: tbl-criticidadRutas\n'
                       "ruta_critica_mc = duraciones_rutas_mc.idxmax(axis='columns')\n"
                       "criticidad_rutas = (ruta_critica_mc.value_counts(normalize=True) * 100).reindex(path_matrix.index, fill_value=0.0)\n"
                       "actividades_ruta = path_matrix.apply(lambda fila: ' - '.join(fila.index[fila == 1]), axis=1)\n"
                       "pd.DataFrame({'actividades': actividades_ruta, 'duración media': duraciones_rutas_medias,\n"
                       "              'criticidad (%)': criticidad_rutas}).sort_values('criticidad (%)', ascending=False).round(1)"))
    celdas.append(code('#| tbl-cap: "Índice de criticidad de las actividades frente a su holgura total"\n#| label: tbl-criticidadActividades\n'
                       "criticidad_actividades = path_matrix.loc[ruta_critica_mc].mean() * 100\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "pd.DataFrame({'holgura total': calendario['H_total'], 'criticidad (%)': criticidad_actividades}).round(1).T"))
    celdas.append(md("La librería hace el mismo cálculo de forma vectorizada: la sensibilidad empírica media de "
                     "`monte_carlo` es precisamente el índice de criticidad de cada actividad."))
    if dist == "normal":
        dists = "{act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}"
    elif dist == "triangular":
        dists = "{act: ('triangular', fila['a'], fila['m'], fila['b']) for act, fila in datos_proyecto.iterrows()}"
    else:
        dists = "{act: ('pert', fila['a'], fila['m'], fila['b']) for act, fila in datos_proyecto.iterrows()}"
    celdas.append(code(f"resultado = proyecto.monte_carlo({dists}, n_samples=N, seed={semilla})\n"
                       "display(resultado.summary().round(2))\n"
                       "(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%) según monte_carlo').T"))
    celdas.append(code("vigilar = criticidad_actividades[calendario['H_total'] > 0].sort_values(ascending=False)\n"
                       "Markdown(f'''La actividad no crítica con mayor índice de criticidad es **{vigilar.index[0]}** "
                       "({vigilar.iloc[0]:.1f} % de las réplicas en la ruta más larga, con holgura {int(calendario.loc[vigilar.index[0], \"H_total\"])}). "
                       "El análisis determinista la da por holgada, pero un retraso pequeño la convierte en crítica: es la que conviene vigilar.''')"))
    sub = f"{n} actividades, {info['n_rutas']} rutas, distribución {nombre_dist}"
    return celdas, sub
