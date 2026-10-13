"""Colección 08 — Probabilidad de cumplir un plazo con la aproximación normal
por el camino crítico."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="08_probabilidad", prefijo="Probabilidad",
            titulo="Probabilidad de cumplir el plazo (aproximación normal)",
            descripcion="Varianza del proyecto por el camino crítico, probabilidad de terminar antes de un "
                        "plazo o entre dos fechas y duración comprometible para una probabilidad dada.",
            n_defecto=3)


def _base(k, rng):
    varias = (k % 3 != 1)
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 10
                    and (i["n_criticas"] >= 2 if varias else i["n_criticas"] == 1) and i["T"] >= 12,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    sd = P.desviaciones(rng, list(D.index))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    T = int(info["T"])
    # varianza del proyecto = máxima entre rutas críticas
    var_rutas = {r: float((sd[info["P"].columns] ** 2 * info["P"].loc[r]).sum()) for r in info["rutas_criticas"]}
    sigma = float(np.sqrt(max(var_rutas.values())))
    plazo = T + int(rng.integers(1, 3))
    t_a, t_b = T - 1, T + 2
    prob_obj = float(rng.choice([0.90, 0.95]))

    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas "
                 "(en semanas) se adjunta.")]
    pregs = ["Determine la duración media del proyecto y su(s) camino(s) crítico(s).",
             "Calcule la varianza y la desviación típica de la duración del proyecto."
             + (" ¿Qué ruta crítica debe usarse para ello? Razone la respuesta." if varias else ""),
             f"¿Qué probabilidad hay de terminar el proyecto en {plazo} semanas o menos?",
             f"¿Qué probabilidad hay de terminar entre las semanas {t_a} y {t_b}?",
             f"¿Qué duración puede comprometerse con el cliente con una probabilidad del {int(prob_obj * 100)}% "
             "de cumplirla?",
             "¿Qué probabilidad hay de terminar en la duración media o antes? ¿Puede garantizarse esa duración "
             f"con un {int(prob_obj * 100)}% de probabilidad sin modificar el proyecto?"]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| fig-cap: "Grafo PERT con tiempos de los nodos y camino crítico"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas.append(code('#| tbl-cap: "Holgura total de las actividades"\n#| label: tbl-holguras\n'
                       "resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])\n"
                       "resultados_pert['activities'].T"))
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "for nombre, acts in rutas_criticas.items():\n"
                       "    print(f'{nombre}: {\" - \".join(acts)}')\n"
                       "Markdown(f'La duración media del proyecto es {duracion_media} semanas y tiene {len(rutas_criticas)} ruta(s) crítica(s).')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Por el teorema central del límite, la duración de una ruta es aproximadamente normal con "
                     "media la suma de las medias y varianza la suma de las varianzas de sus actividades. "
                     "Para el proyecto se toma la ruta crítica; si hay varias, la de mayor varianza, que es la "
                     "hipótesis más prudente (la verdadera distribución del máximo de varias rutas no es "
                     "normal, y el método PERT la aproxima así)."))
    celdas.append(code('#| tbl-cap: "Varianza de las rutas críticas"\n#| label: tbl-varianzaRutas\n'
                       "desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'],\n"
                       "                                                                   variances=datos_proyecto['variance'])\n"
                       "varianza_rutas"))
    celdas.append(code("Markdown(f'La varianza del proyecto es {desviacion_proyecto**2:.3f} y su desviación típica {desviacion_proyecto:.3f} semanas.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Con $T \\sim N(\\mu, \\sigma)$: $P(T \\le t) = \\Phi\\left(\\frac{t - \\mu}{\\sigma}\\right)$."))
    celdas.append(code(f"plazo = {plazo}\n"
                       "z = (plazo - duracion_media) / desviacion_proyecto\n"
                       "probabilidad_plazo = norm.cdf(z)\n"
                       "Markdown(f'z = ({plazo} − {duracion_media}) / {desviacion_proyecto:.3f} = {z:.3f}, luego P(T ≤ {plazo}) = Φ({z:.3f}) = {probabilidad_plazo:.4f} ({probabilidad_plazo*100:.1f} %).')"))
    celdas += apartado(ap); ap += 1
    celdas.append(code(f"t_a, t_b = {t_a}, {t_b}\n"
                       "p_b = norm.cdf((t_b - duracion_media) / desviacion_proyecto)\n"
                       "p_a = norm.cdf((t_a - duracion_media) / desviacion_proyecto)\n"
                       "Markdown(f'P({t_a} < T ≤ {t_b}) = Φ({(t_b - duracion_media) / desviacion_proyecto:.3f}) − Φ({(t_a - duracion_media) / desviacion_proyecto:.3f}) '\n"
                       "         f'= {p_b:.4f} − {p_a:.4f} = {p_b - p_a:.4f} ({(p_b - p_a)*100:.1f} %).')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Se busca $t$ tal que $P(T \\le t) = p$, es decir, $t = \\mu + z_p\\,\\sigma$ con "
                     "$z_p = \\Phi^{-1}(p)$."))
    celdas.append(code(f"probabilidad_objetivo = {prob_obj}\n"
                       "z_p = norm.ppf(probabilidad_objetivo)\n"
                       "duracion_comprometida = duracion_media + z_p * desviacion_proyecto\n"
                       "Markdown(f'z = {z_p:.3f}; t = {duracion_media} + {z_p:.3f} × {desviacion_proyecto:.3f} = {duracion_comprometida:.2f} semanas. '\n"
                       "         f'Como el calendario se mide en semanas enteras, se comprometerían {int(np.ceil(duracion_comprometida))} semanas.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(code("Markdown(f'''La duración media es la mediana de la normal, así que P(T ≤ {duracion_media}) = 0.5: sólo la mitad de las veces "
                       f"se terminaría en la duración media o antes. Para garantizarla con un {int(prob_obj*100)}% haría falta que la media del proyecto fuera "
                       "{duracion_media} − {z_p:.3f} × {desviacion_proyecto:.3f} = {duracion_media - z_p * desviacion_proyecto:.2f} semanas, es decir, reducir el proyecto "
                       "(acortando actividades críticas) o reducir su varianza. Sin tocar el proyecto no es posible.''')"))
    sub = f"{n} actividades, {info['n_criticas']} ruta(s) crítica(s), σ = {sigma:.2f}"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _casi_critica(rng):
    """La ruta casi crítica es mucho más variable que la crítica: el PERT clásico se equivoca."""
    n = int(rng.choice([9, 10]))

    def acepta(i):
        return (1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1 and i["segunda"] is not None
                and i["T"] - i["segunda"] == 1 and i["T"] >= 14)

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, L, Pm = info["D"], info["L"], info["P"]
    T = int(info["T"])
    critica = L.idxmax()
    segunda = L.drop(critica).idxmax()
    sd = pd.Series(0.3, index=D.index)
    for a in Pm.columns[Pm.loc[critica] == 1]:
        sd[a] = 0.2
    for a in Pm.columns[(Pm.loc[segunda] == 1) & (Pm.loc[critica] == 0)]:
        sd[a] = float(rng.choice([1.5, 1.8, 2.0]))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    plazo = T + 2
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en "
                 "semanas) se adjunta.")]
    celdas.append(preguntas([
        "Determine la duración media del proyecto, la ruta crítica y la ruta que le sigue en duración.",
        f"Calcule la probabilidad de terminar en {plazo} semanas o menos según el método PERT (aproximación normal "
        "por la ruta crítica).",
        f"Calcule la misma probabilidad considerando únicamente la segunda ruta. ¿Qué ruta da la probabilidad más "
        "baja? ¿Cuál es entonces una cota razonable de la probabilidad de cumplir el plazo?",
        "Compruebe con Monte Carlo (10 000 réplicas, duraciones normales) cuál es la probabilidad real y explique "
        "por qué el método PERT se equivoca aquí más de lo habitual."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "medias = R @ datos_proyecto['duration'][R.columns]\n"
                       "varianzas = R @ datos_proyecto['variance'][R.columns]"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo PERT con tiempos y camino crítico"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas.append(code('#| tbl-cap: "Duración media y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "rutas = pd.DataFrame({'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in R.index],\n"
                       "                      'media': medias, 'varianza': varianzas, 'σ': np.sqrt(varianzas)}, index=R.index).sort_values('media', ascending=False)\n"
                       "rutas.round(3)"))
    celdas.append(code("critica, segunda = rutas.index[0], rutas.index[1]\n"
                       "Markdown(f'La ruta crítica es {critica} ({rutas.loc[critica, \"media\"]:.0f} semanas, σ = {rutas.loc[critica, \"σ\"]:.2f}); '\n"
                       "         f'la segunda, {segunda} ({rutas.loc[segunda, \"media\"]:.0f} semanas, σ = {rutas.loc[segunda, \"σ\"]:.2f}), es mucho más variable.')"))
    celdas += apartado(2)
    celdas.append(code(f"plazo = {plazo}\n"
                       "p_critica = norm.cdf((plazo - rutas.loc[critica, 'media']) / rutas.loc[critica, 'σ'])\n"
                       "Markdown(f'Por la ruta crítica: z = ({plazo} − {rutas.loc[critica, \"media\"]:.0f}) / {rutas.loc[critica, \"σ\"]:.2f} = {(plazo - rutas.loc[critica, \"media\"]) / rutas.loc[critica, \"σ\"]:.2f}, P = {p_critica:.3f}.')"))
    celdas += apartado(3)
    celdas.append(code("p_segunda = norm.cdf((plazo - rutas.loc[segunda, 'media']) / rutas.loc[segunda, 'σ'])\n"
                       "Markdown(f'''Por la segunda ruta: P = {p_segunda:.3f}. Para cumplir el plazo deben cumplirlo *todas* las rutas, así que la probabilidad real "
                       "no puede superar la menor de las dos, {min(p_critica, p_segunda):.3f}; y como las rutas comparten actividades y no son independientes, "
                       "tampoco es simplemente el producto. {\"La ruta crítica sigue mandando también en probabilidad.\" if p_critica < p_segunda else \"Es la segunda ruta, y no la crítica, la que manda en probabilidad, aunque no sea la más larga en media.\"}''')"))
    celdas += apartado(4)
    celdas.append(code("rng = np.random.default_rng(7)\n"
                       "N = 10000\n"
                       "muestras = pd.DataFrame({a: np.clip(rng.normal(datos_proyecto.loc[a, 'duration'], datos_proyecto.loc[a, 'standard_deviation'], N), 0, None) for a in R.columns})\n"
                       "duracion_mc = (muestras @ R.T).max(axis='columns')\n"
                       "p_mc = float((duracion_mc <= plazo).mean())\n"
                       "pd.Series({'PERT (ruta crítica)': p_critica, 'segunda ruta': p_segunda, 'Monte Carlo': p_mc}).round(3).to_frame(f'P(T ≤ {plazo})')"))
    celdas.append(md("El método PERT sólo mira la ruta de mayor duración media. Cuando otra ruta casi tan larga es "
                     "mucho más variable, es *ella* la que más veces hace fallar el plazo: en muchas réplicas resulta "
                     "la más larga. La probabilidad real es menor que la de cada ruta por separado, porque basta "
                     "que una de ellas se pase. La regla práctica es examinar todas las rutas casi críticas y "
                     "quedarse con la probabilidad más baja, o directamente simular."))
    return celdas, f"{n} actividades; segunda ruta a {T - int(info['segunda'])} semana(s) de la crítica y mucho más variable"


def _varianza_maxima(rng):
    """Problema inverso: qué desviación típica puede permitirse una actividad crítica para cumplir el plazo
    con una probabilidad dada."""
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1 and i["T"] >= 12
                    and i["segunda"] is not None and i["T"] - i["segunda"] >= 3,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    sd = P.desviaciones(rng, list(D.index))
    criticas = cal.index[cal["H_total"] == 0].tolist()
    X = criticas[int(rng.integers(0, len(criticas)))]
    sd[X] = np.nan
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    T = int(info["T"])
    plazo = T + 2
    prob = 0.90
    celdas = [md("# Enunciado"),
              md(f"Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en "
                 f"semanas) se adjunta. La actividad {X} se va a subcontratar y la desviación típica de su duración "
                 f"depende del contratista elegido. El cliente exige terminar en {plazo} semanas con una probabilidad "
                 f"de al menos el {int(prob * 100)} %.")]
    celdas.append(preguntas([
        "Determine la duración media del proyecto y la ruta crítica. ¿Está en ella la actividad subcontratada?",
        f"Calcule la varianza del proyecto en función de la desviación típica σ de {X}.",
        f"¿Qué desviación típica máxima puede tener {X} para cumplir la exigencia del cliente?",
        f"Si ningún contratista garantiza esa dispersión, ¿cuántas semanas habría que reducir la duración media "
        f"del proyecto (manteniendo las varianzas) para cumplir con el contratista de σ = 1,5?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado (σ de ' + X + ' por determinar)"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df).replace("nan", "np.nan")))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"X, plazo, probabilidad_objetivo = '{X}', {plazo}, {prob}"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo PERT con tiempos y camino crítico"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "criticas = sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts))\n"
                       "Markdown(f'Duración media {duracion_media} semanas; ruta crítica {\" - \".join(criticas)}. {X} {\"sí\" if X in criticas else \"no\"} está en ella.')"))
    celdas += apartado(2)
    celdas.append(code("var_resto = float((datos_proyecto.loc[[a for a in criticas if a != X], 'standard_deviation'] ** 2).sum())\n"
                       "Markdown(f'Varianza del proyecto = varianza del resto de la ruta crítica + σ² = {var_resto:.3f} + σ².')"))
    celdas += apartado(3)
    celdas.append(md("Se necesita $P(T \\le plazo) \\ge p$, es decir, $\\dfrac{plazo - \\mu}{\\sigma_T} \\ge z_p$, "
                     "luego $\\sigma_T \\le \\dfrac{plazo - \\mu}{z_p}$ y $\\sigma_X^2 \\le \\sigma_T^2 - \\sigma_{resto}^2$."))
    celdas.append(code("z_p = norm.ppf(probabilidad_objetivo)\n"
                       "sigma_T_max = (plazo - duracion_media) / z_p\n"
                       "var_X_max = sigma_T_max ** 2 - var_resto\n"
                       "sigma_X_max = np.sqrt(var_X_max) if var_X_max > 0 else 0.0\n"
                       "Markdown(f'''σ_T ≤ ({plazo} − {duracion_media}) / {z_p:.3f} = {sigma_T_max:.3f}; σ_X² ≤ {sigma_T_max:.3f}² − {var_resto:.3f} = {var_X_max:.3f}, "
                       "luego **σ_X ≤ {sigma_X_max:.2f} semanas**{\" (ningún contratista vale: ni con σ = 0 se alcanza la probabilidad)\" if var_X_max <= 0 else \"\"}.''')"))
    celdas += apartado(4)
    celdas.append(code("sigma_contratista = 1.5\n"
                       "sigma_T = np.sqrt(var_resto + sigma_contratista ** 2)\n"
                       "media_necesaria = plazo - z_p * sigma_T\n"
                       "reduccion = int(np.ceil(duracion_media - media_necesaria))\n"
                       "Markdown(f'Con σ_X = 1,5, σ_T = {sigma_T:.3f} y hace falta μ ≤ {plazo} − {z_p:.3f} × {sigma_T:.3f} = {media_necesaria:.2f}: '\n"
                       "         f'reducir la duración media en {max(reduccion, 0)} semana(s) (suponiendo que la ruta crítica no cambia al acortar).')"))
    return celdas, f"{n} actividades; σ máxima admisible de {X} para P ≥ {int(prob*100)} % en {plazo} semanas"


def _diagnostico(rng):
    """Una resolución con errores típicos: varianza de una ruta no crítica, suma de σ en vez de varianzas."""
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1 and i["T"] >= 12 and i["n_rutas"] >= 4,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, L, Pm = info["D"], info["L"], info["P"]
    sd = P.desviaciones(rng, list(D.index))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    T = int(info["T"])
    critica = L.idxmax()
    otra = L.drop(critica).idxmax()
    acts_c = list(Pm.columns[Pm.loc[critica] == 1])
    acts_o = list(Pm.columns[Pm.loc[otra] == 1])
    sigma_ok = float(np.sqrt((sd[acts_c] ** 2).sum()))
    sigma_err1 = float(np.sqrt((sd[acts_o] ** 2).sum()))     # ruta equivocada
    sigma_err2 = float(sd[acts_c].sum())                        # suma de σ
    plazo = T + 2
    p_ok = float(norm.cdf((plazo - T) / sigma_ok))
    p_err = float(norm.cdf((plazo - T) / sigma_err2))
    celdas = [md("# Enunciado"),
              md("Para el proyecto cuyo cuadro se adjunta, un alumno ha entregado la siguiente resolución de la "
                 f"pregunta «¿qué probabilidad hay de terminar en {plazo} semanas o menos?»:\n\n"
                 f"> *La ruta crítica es {' - '.join(acts_c)}, con duración media {T}. La desviación típica del "
                 f"proyecto es la suma de las desviaciones típicas de sus actividades: σ = {' + '.join(f'{sd[a]:.1f}' for a in acts_c)} = "
                 f"{sigma_err2:.1f}. Por tanto z = ({plazo} − {T}) / {sigma_err2:.1f} = {(plazo - T) / sigma_err2:.2f} y "
                 f"P = {p_err:.3f}.*\n\n"
                 f"Otro alumno ha usado la ruta {' - '.join(acts_o)} para calcular la desviación típica, obteniendo σ = {sigma_err1:.2f}.")]
    celdas.append(preguntas([
        "Compruebe la duración media y la ruta crítica.",
        "Localice el error del primer alumno y corrija el cálculo. ¿Sobrestima o subestima la probabilidad su error?",
        "¿Qué error ha cometido el segundo alumno? ¿En qué caso su desviación típica sería la correcta?",
        f"Dé la probabilidad correcta de terminar en {plazo} semanas o menos."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"plazo = {plazo}"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo PERT con tiempos y camino crítico"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])"))
    celdas += apartado(2)
    celdas.append(md("Las varianzas de variables independientes se suman; las desviaciones típicas no. "
                     "$\\sigma_T = \\sqrt{\\sum \\sigma_i^2}$, que es siempre menor que $\\sum \\sigma_i$. Al usar una "
                     "σ demasiado grande el alumno obtiene un z demasiado pequeño y, para un plazo mayor que la "
                     "media, una probabilidad demasiado baja (subestima)."))
    celdas.append(code("desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['variance'])\n"
                       "criticas = sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts))\n"
                       "suma_sigmas = float(datos_proyecto.loc[criticas, 'standard_deviation'].sum())\n"
                       "pd.Series({'suma de σ (incorrecto)': suma_sigmas, 'raíz de la suma de varianzas (correcto)': desviacion_proyecto}).round(3).to_frame('σ del proyecto')"))
    celdas += apartado(3)
    celdas.append(md("El segundo alumno ha sumado varianzas, que es lo correcto, pero de una ruta que no es la "
                     "crítica. La varianza del proyecto se toma de la ruta crítica; sólo si esa otra ruta también "
                     "fuera crítica (misma duración media) habría que considerarla, y entonces se tomaría la de "
                     "mayor varianza."))
    celdas += apartado(4)
    celdas.append(code("z = (plazo - duracion_media) / desviacion_proyecto\n"
                       "Markdown(f'z = ({plazo} − {duracion_media}) / {desviacion_proyecto:.3f} = {z:.3f}; P(T ≤ {plazo}) = {norm.cdf(z):.3f}, frente al {" + f"{p_err:.3f}" + "} del primer alumno.')"))
    return celdas, f"{n} actividades; resolución con σ sumadas y con la ruta equivocada"


VARIANTES = [
    Variante("normal", 1, "varianza por la ruta crítica, P(T ≤ t), P(a < T ≤ b), duración para un percentil",
             lambda rng: _base(1, rng)),
    Variante("dos-rutas-criticas", 2, "ídem con dos rutas críticas de varianzas distintas: cuál usar", lambda rng: _base(2, rng)),
    Variante("dos-rutas-criticas-b", 2, "ídem con otro proyecto", lambda rng: _base(3, rng)),
    Variante("casi-critica-variable", 3, "la segunda ruta es mucho más variable que la crítica: cota de probabilidad y Monte Carlo",
             _casi_critica),
    Variante("varianza-maxima", 3, "inverso: σ máxima admisible de una actividad para cumplir el plazo con probabilidad dada",
             _varianza_maxima),
    Variante("diagnostico", 2, "resolución entregada con σ sumadas y con la ruta equivocada: localizar y corregir",
             _diagnostico),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
