"""Colección 07 — Tres estimaciones: media y varianza PERT de cada actividad,
cálculo inverso, comparación Beta-PERT / triangular / uniforme."""
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="07_estimaciones", prefijo="Estimaciones",
            titulo="Estimaciones de duración (a, m, b)",
            descripcion="Obtener la duración media y la varianza de cada actividad a partir de sus tres "
                        "estimaciones, resolver el problema inverso y comparar distribuciones.",
            n_defecto=3)


def _base(k, rng):
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and i["n_criticas"] == 1 and (i["D"] >= 3).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    a, m, b = P.tres_estimaciones(rng, D)
    asim = [x for x in D.index if m[x] != D[x]]
    # problema inverso: dos actividades de las que se oculta a (o b)
    ocultas = list(rng.choice(list(D.index), size=2, replace=False))
    df = P.tabla_base(info, a=a, m=m, b=b)
    tabla = df.copy().astype(object)
    for x in ocultas:
        tabla.loc[x, "a"] = "?"
    tabla["duration"] = "?"
    tabla.loc[ocultas, "duration"] = [int(D[x]) for x in ocultas]
    comparar = asim[int(rng.integers(0, len(asim)))] if asim else list(D.index)[0]

    celdas = [md("# Enunciado"),
              md("El equipo ha dado tres estimaciones de la duración de cada actividad (optimista $a$, más "
                 f"probable $m$ y pesimista $b$, en semanas). Para las actividades {ocultas[0]} y {ocultas[1]} "
                 "se extravió la estimación optimista, pero se conserva la duración media que se había calculado.")]
    pregs = ["Calcule la duración media y la varianza de cada actividad según el método PERT.",
             f"Recupere la estimación optimista de {ocultas[0]} y {ocultas[1]} a partir de su duración media.",
             "Dibuje el grafo PERT, determine el camino crítico con las duraciones medias y la duración media "
             "del proyecto. ¿Qué papel juegan las estimaciones optimista y pesimista en ese cálculo?",
             f"Para la actividad {comparar}, compare la media y la desviación típica que resultan de suponer "
             "una distribución Beta-PERT, una triangular y una uniforme con los mismos valores extremos. "
             "¿Cuál es más prudente?"]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(tabla)))
    celdas.append(md("# Resolución"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas.append(md("El método PERT supone que la duración sigue una distribución Beta cuya media y varianza "
                     "se aproximan por\n\n$$D = \\frac{a + 4m + b}{6}, \\qquad \\sigma^2 = \\left(\\frac{b - a}{6}\\right)^2.$$"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "conocidas = datos_proyecto['a'] != '?'\n"
                       "numericas = datos_proyecto.loc[conocidas, ['a', 'm', 'b']].astype(float)\n"
                       "datos_proyecto.loc[conocidas, 'duration'] = numericas.eval('(a + 4 * m + b) / 6')\n"
                       "datos_proyecto.loc[conocidas, 'variance'] = numericas.eval('((b - a) / 6) ** 2')\n"
                       "datos_proyecto"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Despejando en la fórmula de la media: $a = 6D - 4m - b$."))
    celdas.append(code("faltan = ~conocidas\n"
                       "datos_proyecto.loc[faltan, 'a'] = (6 * datos_proyecto.loc[faltan, 'duration'].astype(float)\n"
                       "                                   - 4 * datos_proyecto.loc[faltan, 'm'].astype(float)\n"
                       "                                   - datos_proyecto.loc[faltan, 'b'].astype(float))\n"
                       "datos_proyecto[['a', 'm', 'b', 'duration']] = datos_proyecto[['a', 'm', 'b', 'duration']].astype(float)\n"
                       "datos_proyecto['variance'] = datos_proyecto.eval('((b - a) / 6) ** 2')\n"
                       "datos_proyecto['standard_deviation'] = np.sqrt(datos_proyecto['variance'])\n"
                       "datos_proyecto.loc[faltan, ['a', 'm', 'b', 'duration', 'variance']]"))
    celdas += apartado(ap); ap += 1
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duraciones = datos_proyecto['duration'].round().astype(int)"))
    celdas.append(code('#| fig-cap: "Grafo PERT con las duraciones medias y el camino crítico"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=duraciones, ranksep=1)"))
    celdas.append(code("duracion_proyecto = proyecto.duration(duraciones)\n"
                       "camino_critico = proyecto.display_critical_path(duraciones)\n"
                       "Markdown(f'La duración media del proyecto es {duracion_proyecto} semanas. Las estimaciones optimista y pesimista '\n"
                       "         'no intervienen directamente en el camino crítico: sólo a través de la media de cada actividad. '\n"
                       "         'Donde sí cuentan es en la varianza, y por tanto en la probabilidad de cumplir un plazo.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Con los mismos extremos $a$ y $b$ (y la misma moda $m$ en la triangular):\n\n"
                     "| Distribución | Media | Varianza |\n|---|---|---|\n"
                     "| Beta-PERT | $(a + 4m + b)/6$ | $((b-a)/6)^2$ |\n"
                     "| Triangular | $(a + m + b)/3$ | $(a^2 + m^2 + b^2 - ab - am - bm)/18$ |\n"
                     "| Uniforme | $(a + b)/2$ | $(b-a)^2/12$ |"))
    celdas.append(code(f"a_, m_, b_ = datos_proyecto.loc['{comparar}', ['a', 'm', 'b']]\n"
                       "comparacion = pd.DataFrame({\n"
                       "    'Beta-PERT':  [(a_ + 4 * m_ + b_) / 6, ((b_ - a_) / 6) ** 2],\n"
                       "    'Triangular': [(a_ + m_ + b_) / 3, (a_**2 + m_**2 + b_**2 - a_*b_ - a_*m_ - b_*m_) / 18],\n"
                       "    'Uniforme':   [(a_ + b_) / 2, (b_ - a_) ** 2 / 12]},\n"
                       "    index=['media', 'varianza'])\n"
                       "comparacion.loc['desviación típica'] = np.sqrt(comparacion.loc['varianza'])\n"
                       "comparacion.round(3)"))
    celdas.append(code("Markdown(f'''La Beta-PERT concentra la probabilidad cerca de la moda ({m_:.0f}) y por eso tiene la menor varianza; "
                       "la uniforme reparte la probabilidad por igual entre {a_:.0f} y {b_:.0f} y es la más dispersa; la triangular queda en medio. "
                       "Cuando la moda no está centrada, las medias también difieren: la PERT pesa la moda cuatro veces y se queda más cerca de ella. "
                       "La uniforme es la hipótesis más prudente (mayor varianza), pero sólo está justificada si de verdad no se sabe nada entre los extremos.''')"))
    sub = f"{n} actividades, {len(asim)} con estimaciones asimétricas"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _deterministas(rng):
    """Algunas actividades tienen a = m = b: no aportan varianza aunque sean críticas; qué actividades
    determinan la incertidumbre del plazo."""
    import numpy as np
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and i["n_criticas"] == 1 and (i["D"] >= 2).all(),
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    a, m, b = P.tres_estimaciones(rng, D)
    cal = info["calendario"]
    criticas = cal.index[cal["H_total"] == 0].tolist()
    no_crit = cal.index[cal["H_total"] > 0].tolist()
    fijas = [criticas[int(rng.integers(0, len(criticas)))]]
    if no_crit:
        fijas.append(no_crit[int(rng.integers(0, len(no_crit)))])
    for x in fijas:
        a[x] = m[x] = b[x] = int(D[x])
    # una no crítica muy variable
    muy_variable = [x for x in no_crit if x not in fijas]
    if muy_variable:
        v = muy_variable[int(rng.integers(0, len(muy_variable)))]
        a[v], b[v] = max(1, int(D[v]) - 3), int(D[v]) + 3
        m[v] = int(D[v])
    df = P.tabla_base(info, a=a, m=m, b=b)
    celdas = [md("# Enunciado"),
              md("Se dispone de las tres estimaciones (optimista, más probable, pesimista; en semanas) de cada "
                 "actividad. Algunas actividades tienen duración fija (las tres estimaciones coinciden).")]
    celdas.append(preguntas([
        "Calcule la duración media y la varianza de cada actividad. ¿Qué varianza tienen las actividades de duración fija?",
        "Determine el camino crítico y la duración media del proyecto.",
        "Calcule la varianza del proyecto. ¿Qué actividades contribuyen a ella? ¿Contribuye la actividad no crítica "
        "más variable? ¿Y las críticas de duración fija?",
        "Si pudiera reducir a cero la varianza de una sola actividad, ¿cuál elegiría para hacer el plazo más "
        "predecible? ¿Y cuál para acortarlo?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['duration'] = datos_proyecto.eval('(a + 4 * m + b) / 6').round().astype(int)\n"
                       "datos_proyecto['variance'] = datos_proyecto.eval('((b - a) / 6) ** 2')\n"
                       "datos_proyecto"))
    celdas.append(md("Cuando $a = m = b$ la varianza es $(b-a)^2/36 = 0$: la actividad no aporta incertidumbre, "
                     "aunque sí aporta duración."))
    celdas += apartado(2)
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas.append(code('#| fig-cap: "Grafo PERT con duraciones medias y camino crítico"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])\n"
                       "Markdown(f'La duración media del proyecto es {duracion_media} semanas.')"))
    celdas += apartado(3)
    celdas.append(code("desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['variance'])\n"
                       "display(varianza_rutas)\n"
                       "criticas = sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts))\n"
                       "contribuyen = datos_proyecto.loc[criticas, 'variance']\n"
                       "mas_variable_no_critica = datos_proyecto['variance'].drop(criticas).idxmax()\n"
                       "pd.DataFrame({'crítica': datos_proyecto.index.isin(criticas), 'varianza': datos_proyecto['variance']}).T"))
    celdas.append(code("Markdown(f'''La varianza del proyecto, {desviacion_proyecto**2:.3f}, es la suma de las varianzas de las actividades críticas: "
                       "sólo contribuyen {\", \".join(a for a in criticas if datos_proyecto.loc[a, \"variance\"] > 0)}. "
                       "La actividad no crítica más variable, {mas_variable_no_critica} (varianza {datos_proyecto.loc[mas_variable_no_critica, \"variance\"]:.3f}), no cuenta en la aproximación PERT "
                       "porque no está en la ruta crítica, aunque Monte Carlo mostraría que sí influye si su holgura es pequeña. "
                       "Las críticas de duración fija ({\", \".join(a for a in criticas if datos_proyecto.loc[a, \"variance\"] == 0) or \"ninguna\"}) aportan duración pero no varianza.''')"))
    celdas += apartado(4)
    celdas.append(code("mas_variable_critica = contribuyen.idxmax()\n"
                       "mas_larga_critica = datos_proyecto.loc[criticas, 'duration'].idxmax()\n"
                       "Markdown(f'Para hacer el plazo más predecible conviene atacar la varianza de {mas_variable_critica} (la crítica de mayor varianza); '\n"
                       "         f'para acortarlo, la duración de alguna crítica, por ejemplo {mas_larga_critica}, la más larga. Reducir la varianza de una no crítica no cambia nada en el análisis PERT.')"))
    return celdas, f"{n} actividades, {len(fijas)} de duración fija"


VARIANTES = [
    Variante("tres-estimaciones", 1, "media y varianza PERT, cálculo inverso de a, camino crítico, Beta-PERT vs triangular vs uniforme",
             lambda rng: _base(1, rng)),
    Variante("tres-estimaciones-b", 1, "ídem con estimaciones asimétricas", lambda rng: _base(2, rng)),
    Variante("tres-estimaciones-c", 1, "ídem con otro proyecto", lambda rng: _base(3, rng)),
    Variante("deterministas", 2, "actividades de duración fija y qué actividades determinan la varianza del plazo",
             _deterministas),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
