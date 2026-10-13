"""Colección 07 — Tres estimaciones: media y varianza PERT de cada actividad,
cálculo inverso, comparación Beta-PERT / triangular / uniforme."""
from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

META = dict(slug="07_estimaciones", prefijo="Estimaciones",
            titulo="Estimaciones de duración (a, m, b)",
            descripcion="Obtener la duración media y la varianza de cada actividad a partir de sus tres "
                        "estimaciones, resolver el problema inverso y comparar distribuciones.",
            n_defecto=3)


def generar(k, rng):
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
