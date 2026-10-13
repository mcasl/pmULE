"""Colección 08 — Probabilidad de cumplir un plazo con la aproximación normal
por el camino crítico."""
import numpy as np

from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

META = dict(slug="08_probabilidad", prefijo="Probabilidad",
            titulo="Probabilidad de cumplir el plazo (aproximación normal)",
            descripcion="Varianza del proyecto por el camino crítico, probabilidad de terminar antes de un "
                        "plazo o entre dos fechas y duración comprometible para una probabilidad dada.",
            n_defecto=3)


def generar(k, rng):
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
