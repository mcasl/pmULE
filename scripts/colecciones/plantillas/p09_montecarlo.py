"""Colección 09 — Método de Monte Carlo: simulación con la matriz de caminos,
comparación con la aproximación normal, índices de criticidad."""
import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

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
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de caminos"\n#| label: tbl-matrizCaminos\n'
                       "path_matrix = proyecto.path_matrix(dummies=False)\n"
                       "proyecto.display_rutas('pert')"))
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
        dists = ("{act: (lambda a_, m_, b_: (lambda size=None: muestras_beta_pert(a_, m_, b_, size)))(fila['a'], fila['m'], fila['b'])\n"
                 "                                   for act, fila in datos_proyecto.iterrows()}")
        celdas.append(md("La opción `('pert', a, m, b)` de `monte_carlo` es una normal recortada al intervalo "
                         "$[a, b]$, no la Beta-PERT exacta; por eso se le pasa como función la misma Beta-PERT "
                         "usada arriba (la función recibe `size` y devuelve un vector de muestras)."))
    celdas.append(code(f"resultado = proyecto.monte_carlo({dists}, n_samples=N, seed={semilla})\n"
                       "display(resultado.summary().round(2))\n"
                       "(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%) según monte_carlo').T"))
    celdas.append(code("vigilar = criticidad_actividades[calendario['H_total'] > 0].sort_values(ascending=False)\n"
                       "Markdown(f'''La actividad no crítica con mayor índice de criticidad es **{vigilar.index[0]}** "
                       "({vigilar.iloc[0]:.1f} % de las réplicas en la ruta más larga, con holgura {int(calendario.loc[vigilar.index[0], \"H_total\"])}). "
                       "El análisis determinista la da por holgada, pero un retraso pequeño la convierte en crítica: es la que conviene vigilar.''')"))
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
        "Dibuje el grafo PERT, enumere las rutas y escriba la matriz de rutas R (una fila por ruta, una columna por actividad).",
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
    celdas.append(code('#| tbl-cap: "Matriz de rutas R"\n#| label: tbl-matrizRutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "rutas = proyecto.paths(dummies=False)\n"
                       "proyecto.display_rutas('pert')"))
    celdas += apartado(2)
    celdas.append(code("T = replicas.loc[R.columns]\n"
                       "RT = R @ T\n"
                       "from pmule import display_producto_rutas\n"
                       "display_producto_rutas(R, T, RT)"))
    celdas.append(code('#| tbl-cap: "R·T: duración de cada ruta (filas) en cada réplica (columnas)"\n#| label: tbl-RT\n'
                       "RT"))
    celdas.append(md("El elemento (r, j) de R·T es la duración de la ruta r en la réplica j: la suma de las "
                     "duraciones, en esa réplica, de las actividades que forman la ruta. Una fila de R selecciona "
                     "las actividades de la ruta; una columna de T es un escenario completo."))
    celdas += apartado(3)
    celdas.append(code("tau = RT.max()\n"
                       "pd.DataFrame({'τ = máximo de la columna': tau, 'ruta más larga': RT.idxmax()}).T"))
    celdas.append(code("media = tau.mean(); desviacion = tau.std(ddof=1)\n"
                       "Markdown(f'''Media = ({\" + \".join(str(int(x)) for x in tau)}) / {len(tau)} = {media:.2f} días. "
                       "Desviación típica muestral = √(Σ(τ − media)² / (n − 1)) = {desviacion:.2f} días. "
                       "Con las duraciones medias el proyecto duraría {proyecto.duration(datos_proyecto[\"duration\"])} días.''')"))
    celdas += apartado(4)
    celdas.append(code("ruta_critica = RT.idxmax()\n"
                       "criticidad_rutas = ruta_critica.value_counts().reindex(R.index, fill_value=0)\n"
                       "criticidad_actividades = R.loc[ruta_critica].sum()\n"
                       "pd.DataFrame({'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in R.index],\n"
                       "              'veces crítica': criticidad_rutas, 'índice de criticidad': criticidad_rutas / len(tau)}, index=R.index)"))
    celdas.append(code("pd.DataFrame({'veces en la ruta crítica': criticidad_actividades, 'índice de criticidad': criticidad_actividades / len(tau)}).T"))
    celdas.append(code("camino_medio = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "Markdown(f'Con las duraciones medias la ruta crítica es {\", \".join(\" - \".join(a) for a in camino_medio.values())}; en las réplicas ha sido la más larga '\n"
                       "         f'{int(criticidad_rutas.max())} de {len(tau)} veces. Con tan pocas réplicas los índices sólo pueden valer múltiplos de {1/len(tau):.1f}.')"))
    celdas += apartado(5)
    celdas.append(code("rng = np.random.default_rng(1)\n"
                       "N = 10000\n"
                       "T_grande = pd.DataFrame(np.clip(rng.normal(datos_proyecto.loc[R.columns, 'duration'].values[:, None], 1.0, (len(R.columns), N)), 0.5, None), index=R.columns)\n"
                       "tau_grande = (R @ T_grande).max()\n"
                       "criticidad_grande = R.loc[(R @ T_grande).idxmax()].mean()\n"
                       "pd.DataFrame({'5 réplicas': [media, desviacion], '10 000 réplicas': [tau_grande.mean(), tau_grande.std(ddof=1)]}, index=['media', 'desviación típica']).round(2)"))
    celdas.append(code("pd.DataFrame({'5 réplicas': criticidad_actividades / len(tau), '10 000 réplicas': criticidad_grande}).round(3).T"))
    celdas.append(md("La operación es exactamente la misma (R·T y máximo por columnas); sólo cambia el número de "
                     "columnas. Con muchas réplicas la media y la desviación típica se estabilizan (su error "
                     "estadístico decrece como 1/√N) y los índices de criticidad dejan de ser múltiplos gruesos."))
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
    celdas.append(code("filas = {}\n"
                       "for act, fila in distribuciones.iterrows():\n"
                       "    p = parametros(fila['parametros'])\n"
                       "    if fila['distribucion'] == 'normal':\n"
                       "        media, var = p[0], p[1] ** 2\n"
                       "    elif fila['distribucion'] == 'triangular':\n"
                       "        a, m, b = p; media, var = (a + m + b) / 3, (a*a + m*m + b*b - a*b - a*m - b*m) / 18\n"
                       "    elif fila['distribucion'] == 'uniforme':\n"
                       "        a, b = p; media, var = (a + b) / 2, (b - a) ** 2 / 12\n"
                       "    else:\n"
                       "        media, var = p[0], 0.0\n"
                       "    filas[act] = {'distribución': fila['distribucion'], 'media': media, 'varianza': var}\n"
                       "resumen = pd.DataFrame(filas).T\n"
                       "resumen['media del cuadro'] = datos_proyecto['duration']\n"
                       "datos_proyecto['variance'] = resumen['varianza'].astype(float)\n"
                       "resumen"))
    celdas += apartado(2)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con duraciones medias y camino crítico", label="fig-pert")
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['variance'])\n"
                       "plazo = duracion_media + 2\n"
                       "p_normal = float(norm.cdf((plazo - duracion_media) / desviacion_proyecto)) if desviacion_proyecto > 0 else 1.0\n"
                       "Markdown(f'Duración media {duracion_media}, σ = {desviacion_proyecto:.3f}; P(T ≤ {plazo}) = {p_normal:.3f} por la aproximación normal.')"))
    celdas += apartado(3)
    celdas.append(md("`monte_carlo` acepta una distribución por actividad: `('normal', μ, σ)`, "
                     "`('triangular', a, m, b)`, `('uniform', a, b)` o una función; una duración fija es una "
                     "función que siempre devuelve el mismo valor."))
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
                       "display(resultado.summary().round(2))\n"
                       "pd.DataFrame({'aproximación normal': [duracion_media, desviacion_proyecto, p_normal], 'Monte Carlo': [resultado.tau_mean, resultado.tau_std, p_mc]},\n"
                       "             index=['media', 'desviación típica', f'P(T ≤ {plazo})']).round(3)"))
    celdas += apartado(4)
    celdas.append(code("calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "crit = (resultado.sensitivity_mean * 100).round(1).rename('índice de criticidad (%)')\n"
                       "tabla = pd.concat([crit, calendario['H_total'].rename('holgura'), resumen['distribución'], resumen['varianza'].astype(float).round(2)], axis=1).sort_values('índice de criticidad (%)', ascending=False)\n"
                       "tabla"))
    celdas.append(code("vigilar = tabla[tabla['holgura'] > 0]['índice de criticidad (%)'].idxmax()\n"
                       "Markdown(f'''La actividad no crítica con mayor índice de criticidad es {vigilar} ({tabla.loc[vigilar, \"índice de criticidad (%)\"]} %). "
                       "Las actividades de duración fija nunca cambian de ruta por sí mismas: su índice depende sólo de lo que hagan las demás; "
                       "una ruta con muchas actividades fijas tiene poca varianza y rara vez «adelanta» a la crítica.''')"))
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
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias y el camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de rutas"\n#| label: tbl-matrizRutas\n'
                       "proyecto.display_rutas('pert')"))
    celdas.append(code('#| tbl-cap: "Media y varianza de cada ruta (sumas de las de sus actividades)"\n#| label: tbl-rutas\n'
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "medias_rutas = detalle_rutas(R, datos_proyecto['duration'][R.columns], nombre='duración media')\n"
                       "varianzas_rutas = detalle_rutas(R, datos_proyecto['variance'][R.columns], nombre='varianza', decimales=2)\n"
                       "pd.concat([medias_rutas, varianzas_rutas[['varianza = suma', 'varianza']]], axis=1)"))
    celdas.append(md("Cada réplica es un vector de duraciones normales; la duración de cada ruta es R·t y la del proyecto, "
                     "el máximo. Para cada tamaño N se repite el mismo cálculo con más columnas."))
    celdas.append(code("tamanos = [20, 200, 2000, 20000]\n"
                       "filas = {}\n"
                       "for N in tamanos:\n"
                       "    tau = simular(N, 0)\n"
                       "    filas[N] = {'media': tau.mean(), 'desviación típica': tau.std(ddof=1), 'percentil 90': np.quantile(tau, 0.9), f'P(T ≤ {plazo})': (tau <= plazo).mean()}\n"
                       "pd.DataFrame(filas).T.round(3)"))
    celdas += apartado(2)
    celdas.append(code("repeticiones = 10\n"
                       "disp = {}\n"
                       "for N in tamanos:\n"
                       "    medias = [simular(N, s).mean() for s in range(1, repeticiones + 1)]\n"
                       "    p90 = [np.quantile(simular(N, s), 0.9) for s in range(1, repeticiones + 1)]\n"
                       "    disp[N] = {'dispersión de la media': np.std(medias, ddof=1), 'dispersión del percentil 90': np.std(p90, ddof=1)}\n"
                       "dispersion = pd.DataFrame(disp).T\n"
                       "dispersion.round(4)"))
    celdas.append(md("Al multiplicar N por 10 la dispersión se divide aproximadamente por √10 ≈ 3,16."))
    celdas += apartado(3)
    celdas.append(code("sigma_tau = simular(20000, 0).std(ddof=1)\n"
                       "dispersion['σ/√N'] = [sigma_tau / np.sqrt(N) for N in tamanos]\n"
                       "N_necesario = int(np.ceil((sigma_tau / 0.1) ** 2))\n"
                       "display(dispersion.round(4))\n"
                       "Markdown(f'Con σ_τ ≈ {sigma_tau:.2f}, para un error típico de 0,1 semanas hacen falta N ≥ (σ/0,1)² ≈ {N_necesario} réplicas.')"))
    celdas += apartado(4)
    celdas.append(md("El percentil 90 se estima con las réplicas de la cola, que son pocas (el 10 % de N), y su "
                     "error es mayor que el de la media para el mismo N; la tabla de dispersiones lo muestra. Por "
                     "eso las decisiones basadas en percentiles altos (plazos comprometidos al 90–95 %) necesitan "
                     "más réplicas que las basadas en la media."))
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
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas.append(code("path_matrix = proyecto.path_matrix(dummies=False)\n"
                       "duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])\n"
                       "Markdown(f'Duración media del proyecto: {duracion_media} semanas. Actividades expuestas al factor común: {\", \".join(expuestas)}.')"))
    celdas.append(code('#| tbl-cap: "Duración media de cada ruta y actividades expuestas que contiene"\n#| label: tbl-rutas\n'
                       "rutas = detalle_rutas(path_matrix, datos_proyecto['duration'][path_matrix.columns], nombre='duración media')\n"
                       "rutas['expuestas en la ruta'] = [', '.join(a for a in path_matrix.columns[path_matrix.loc[r] == 1] if a in expuestas) or 'ninguna' for r in path_matrix.index]\n"
                       "rutas"))
    celdas += apartado(2)
    celdas.append(code(f"rng = np.random.default_rng({semilla})\n"
                       "mu, sigma = datos_proyecto['duration'].astype(float), datos_proyecto['standard_deviation'].astype(float)\n"
                       "X = pd.DataFrame(rng.normal(mu.values, sigma.values, size=(N, len(mu))), columns=mu.index).clip(lower=0)\n"
                       "T_indep = (X[path_matrix.columns] @ path_matrix.T).max(axis='columns')\n"
                       "resumen_indep = pd.Series({'media': T_indep.mean(), 'desviación típica': T_indep.std(), 'percentil 95': T_indep.quantile(0.95),\n"
                       "                           f'P(T ≤ {plazo}) (%)': (T_indep <= plazo).mean() * 100})\n"
                       "resumen_indep.round(3).to_frame('independientes').T"))
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
    celdas.append(code('#| fig-cap: "Distribución de la duración con y sin factor común"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "ax.hist(T_indep, bins=60, density=True, alpha=0.5, label='independientes')\n"
                       "ax.hist(T_corr, bins=60, density=True, alpha=0.5, label='con factor común')\n"
                       "ax.axvline(plazo, color='red', linestyle='--', label=f'plazo ({plazo})')\n"
                       "ax.set_xlabel('Duración del proyecto (semanas)'); ax.legend(); plt.show()"))
    celdas += apartado(4)
    celdas.append(code("ruta_critica_mc = duraciones_rutas_c.idxmax(axis='columns')\n"
                       "criticidad_corr = (ruta_critica_mc.value_counts(normalize=True) * 100).reindex(path_matrix.index, fill_value=0.0)\n"
                       "criticidad_indep = ((X[path_matrix.columns] @ path_matrix.T).idxmax(axis='columns').value_counts(normalize=True) * 100).reindex(path_matrix.index, fill_value=0.0)\n"
                       "pd.DataFrame({'actividades': path_matrix.apply(lambda f: ' - '.join(f.index[f == 1]), axis=1),\n"
                       "              'criticidad indep. (%)': criticidad_indep, 'criticidad con F (%)': criticidad_corr}).round(1)"))
    celdas.append(md("La varianza de una suma de variables correlacionadas positivamente es mayor que la suma de "
                     "sus varianzas (se añaden los términos $2\\,\\mathrm{cov}$): cuando $F$ sale alto, todas las "
                     "actividades expuestas se alargan a la vez y no se compensan unas con otras. La media casi "
                     "no cambia (la de $F$ es 1), pero las colas se ensanchan: el percentil 95 sube y la "
                     "probabilidad de cumplir el plazo baja. Las rutas con más actividades expuestas ganan "
                     "criticidad. Suponer independencia, como hace el PERT clásico, es optimista en presencia "
                     "de causas comunes de retraso."))
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
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Media de cada triangular"\n#| label: tbl-medias\n'
                       "pd.DataFrame({'(a + m + b)/3': [f\"({r['a']:g} + {r['m']:g} + {r['b']:g})/3 = {r['duration']:.2f}\" for _, r in datos_proyecto.iterrows()]}, index=datos_proyecto.index).T"))
    celdas += fig_grafo(durations="datos_proyecto['duration'].round(1)", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas.append(code("path_matrix = proyecto.path_matrix(dummies=False)\n"
                       "duraciones_rutas_medias = path_matrix @ datos_proyecto['duration'][path_matrix.columns]\n"
                       "duracion_media = float(duraciones_rutas_medias.max())\n"
                       "Markdown(f'Con las medias, la ruta más larga es {duraciones_rutas_medias.idxmax()} ({\" - \".join(path_matrix.columns[path_matrix.loc[duraciones_rutas_medias.idxmax()] == 1])}) y la duración {duracion_media:.2f} semanas.')"))
    celdas.append(code('#| tbl-cap: "Duración media de cada ruta"\n#| label: tbl-rutas\n'
                       "detalle_rutas(path_matrix, datos_proyecto['duration'][path_matrix.columns], nombre='duración media', decimales=2)"))
    celdas += apartado(2)
    celdas.append(code(f"rng = np.random.default_rng({semilla})\n"
                       "duraciones_mc = pd.DataFrame({act: rng.triangular(fila['a'], fila['m'], fila['b'], N) for act, fila in datos_proyecto.iterrows()})\n"
                       "duraciones_rutas_mc = duraciones_mc[path_matrix.columns] @ path_matrix.T\n"
                       "T_mc = duraciones_rutas_mc.max(axis='columns')\n"
                       "percentiles = T_mc.quantile([0.10, 0.50, 0.80, 0.90, 0.95])\n"
                       "percentiles.index = [f'P{int(p * 100)}' for p in percentiles.index]\n"
                       "percentiles.round(2).to_frame('duración (semanas)').T"))
    celdas.append(code('#| fig-cap: "Función de distribución simulada de la duración"\n#| label: fig-distribucion\n'
                       "from matplotlib import pyplot as plt\n"
                       "x = np.sort(T_mc.values); y = np.arange(1, N + 1) / N\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "ax.plot(x, y)\n"
                       "for nombre, v in percentiles.items():\n"
                       "    ax.axvline(v, color='grey', linestyle=':', alpha=0.7); ax.text(v, 0.02, nombre, rotation=90, va='bottom')\n"
                       "ax.axvline(plazo, color='red', linestyle='--', label=f'plazo del cliente ({plazo})')\n"
                       "ax.set_xlabel('Duración (semanas)'); ax.set_ylabel('P(T ≤ t)'); ax.legend(); plt.show()"))
    celdas.append(md("El percentil $p$ es la duración que no se supera con probabilidad $p$: P50 es la mediana "
                     "(tan probable acabar antes como después), P80 la fecha que se cumple cuatro de cada cinco "
                     "veces, P95 una fecha «segura». La diferencia P95 − P50 mide cuánto margen hay que añadir a "
                     "una previsión central para hacerla fiable."))
    celdas += apartado(3)
    celdas.append(code("p_penalizacion = float((T_mc > plazo).mean())\n"
                       "retraso = (T_mc - plazo).clip(lower=0)\n"
                       "retraso_95 = float(retraso.quantile(0.95))\n"
                       "retraso_medio = float(retraso.mean())\n"
                       "Markdown(f'''P(T > {plazo}) = {p_penalizacion:.3f}: se paga penalización en el {p_penalizacion * 100:.1f} % de las réplicas. "
                       "El retraso supera {retraso_95:.2f} semanas sólo con probabilidad 0,05 (percentil 95 de máx(0, T − {plazo})). "
                       "El retraso medio esperado es {retraso_medio:.3f} semanas: multiplicado por la penalización semanal da el coste esperado de la cláusula.''')"))
    celdas += apartado(4)
    celdas.append(code("fecha_10 = float(T_mc.quantile(0.90))\n"
                       "Markdown(f'''Para que P(T > fecha) = 0,10 hay que comprometer el percentil 90: {fecha_10:.2f} semanas, "
                       "es decir, {fecha_10 - duracion_media:.2f} semanas de margen sobre la duración media ({duracion_media:.2f}). "
                       "Comprometer la media supone penalización aproximadamente la mitad de las veces (P(T > media) = {float((T_mc > duracion_media).mean()):.3f}, "
                       "algo más de 0,5 por el sesgo de fusión de las rutas).''')"))
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
    celdas.append(code("Markdown(f'Con las duraciones más probables: {duracion_libre} semanas sin límite y {res0.duracion} semanas con {limite[0]} operarios.')"))
    celdas += apartado(2)
    celdas.append(md("En cada réplica se sortean las duraciones (triangulares, redondeadas a semanas enteras), se calcula "
                     "la duración sin recursos (camino más largo) y se vuelve a aplicar el algoritmo de asignación con el "
                     "límite: dos duraciones por réplica. `monte_carlo_recursos` hace las 2000 réplicas y guarda ambas."))
    celdas.append(code('#| tbl-cap: "Distribución de la duración con y sin límite de recursos"\n#| label: tbl-mcRecursos\n'
                       "mc = proyecto.copy().monte_carlo_recursos(dists, demanda[['operarios']].copy(), ['operarios'], limite, n_samples=2000, seed=11)\n"
                       "mc.summary().round(2)"))
    celdas.append(code('#| fig-cap: "Distribución simulada de la duración sin y con el límite de operarios"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "bins = np.arange(min(mc.tau_libre_samples.min(), mc.tau_recursos_samples.min()) - 0.5, max(mc.tau_libre_samples.max(), mc.tau_recursos_samples.max()) + 1.5, 1)\n"
                       "ax.hist(mc.tau_libre_samples, bins=bins, alpha=0.5, label='sin límite')\n"
                       "ax.hist(mc.tau_recursos_samples, bins=bins, alpha=0.5, label=f'con {limite[0]} operarios')\n"
                       "ax.axvline(plazo, color='red', linestyle='--', label=f'plazo ({plazo})')\n"
                       "ax.set_xlabel('duración del proyecto (semanas)'); ax.set_ylabel('réplicas'); ax.legend(); plt.show()"))
    celdas.append(code("Markdown(f'La media sube de {mc.tau_libre_samples.mean():.2f} a {mc.tau_recursos_samples.mean():.2f} semanas '\n"
                       "         f'y la desviación típica pasa de {mc.tau_libre_samples.std():.2f} a {mc.tau_recursos_samples.std():.2f}. '\n"
                       "         'El límite añade esperas que dependen de qué actividades coinciden en cada réplica.')"))
    celdas += apartado(3)
    celdas.append(code("p_libre = 1 - mc.prob_superar(plazo, con_recursos=False)\n"
                       "p_res = 1 - mc.prob_superar(plazo)\n"
                       "Markdown(f'P(T ≤ {plazo}) = {p_libre:.3f} sin recursos y {p_res:.3f} con el límite de operarios. '\n"
                       "         'Quien solo mirase el camino crítico creería que el plazo es seguro cuando en realidad el riesgo real es mucho mayor.')"))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Criticidad sin y con recursos"\n#| label: tbl-criticidadRecursos\n'
                       "(mc.criticidad * 100).round(1).sort_values(mc.criticidad.columns[-1], ascending=False)"))
    celdas.append(code('#| tbl-cap: "Frecuencia con que cada recurso bloquea alguna actividad"\n#| label: tbl-bloqueos\n'
                       "(mc.bloqueos * 100).round(1).to_frame('% de réplicas')"))
    celdas.append(md("La criticidad con recursos incluye actividades que no son críticas en el CPM pero forman parte de la cadena "
                     "de esperas por el recurso: pasan a mandar al introducir el límite."))
    celdas += apartado(5)
    celdas.append(code("mc2 = proyecto.copy().monte_carlo_recursos(dists, demanda[['operarios']].copy(), ['operarios'], [limite[0] + 1], n_samples=2000, seed=11)\n"
                       "p_res2 = 1 - mc2.prob_superar(plazo)\n"
                       "Markdown(f'Con {limite[0] + 1} operarios P(T ≤ {plazo}) pasa de {p_res:.3f} a {p_res2:.3f}; la media baja de '\n"
                       "         f'{mc.tau_recursos_samples.mean():.2f} a {mc2.tau_recursos_samples.mean():.2f} semanas. Merece la pena si el valor de '\n"
                       "         'asegurar el plazo (penalizaciones evitadas × probabilidad ganada) supera el coste del operario adicional.')"))
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
