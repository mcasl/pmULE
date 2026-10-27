"""Colección 07 — Tres estimaciones: media y varianza PERT de cada actividad,
cálculo inverso, comparación Beta-PERT / triangular / uniforme."""
import pandas as pd
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX, grafo_nombre)

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
    tabla["duration"] = pd.Series("?", index=tabla.index, dtype=object)
    tabla.loc[ocultas, "duration"] = [int(D[x]) for x in ocultas]
    comparar = asim[int(rng.integers(0, len(asim)))] if asim else list(D.index)[0]

    celdas = [md("# Enunciado"),
              md("El equipo ha dado tres estimaciones de la duración de cada actividad (optimista $a$, más "
                 f"probable $m$ y pesimista $b$, en semanas). Para las actividades {ocultas[0]} y {ocultas[1]} "
                 "se extravió la estimación optimista, pero se conserva la duración media que se había calculado.")]
    pregs = ["Calcule la duración media y la varianza de cada actividad según el método PERT.",
             f"Recupere la estimación optimista de {ocultas[0]} y {ocultas[1]} a partir de su duración media.",
             f"Dibuje el grafo {grafo_nombre()}, determine el camino crítico con las duraciones medias y la duración media "
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
                     "se aproximan por\n\n$$D = \\frac{a + 4m + b}{6}, \\qquad \\sigma^2 = \\left(\\frac{b - a}{6}\\right)^2.$$\n\n"
                     "Las estimaciones $a$, $m$ y $b$ son las del cuadro del enunciado. Sustituyendo en cada actividad "
                     "con las tres estimaciones conocidas:"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "conocidas = datos_proyecto['a'] != '?'\n"
                       "numericas = datos_proyecto.loc[conocidas, ['a', 'm', 'b']].astype(float)\n"
                       "datos_proyecto.loc[conocidas, 'duration'] = numericas.eval('(a + 4 * m + b) / 6')\n"
                       "datos_proyecto.loc[conocidas, 'variance'] = numericas.eval('((b - a) / 6) ** 2')"))
    celdas.append(code('#| tbl-cap: "Media y varianza de cada actividad con los números sustituidos"\n#| label: tbl-estimacionesPaso\n'
                       "detalle_estimaciones(datos_proyecto[conocidas])"))
    celdas.append(code("sin_a = list(datos_proyecto.index[~conocidas])\n"
                       "Markdown(f'''{lista_y(sin_a)} no tienen estimación optimista: su duración media es la que da el enunciado '''\n"
                       "         f'''({lista_y([f\"{x} = {float(datos_proyecto.loc[x, 'duration']):g}\" for x in sin_a])} semanas) y su varianza se calcula en el apartado 2, '''\n"
                       "         f'''una vez recuperada $a$. La tabla final de ese apartado reúne la media y la varianza de todas las actividades.''')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Despejando $a$ en la fórmula de la media: $6D = a + 4m + b \\Rightarrow a = 6D - 4m - b$. "
                     "Con $a$ ya conocida, la varianza se calcula como en el apartado 1."))
    celdas.append(code("faltan = ~conocidas\n"
                       "datos_proyecto.loc[faltan, 'a'] = (6 * datos_proyecto.loc[faltan, 'duration'].astype(float)\n"
                       "                                   - 4 * datos_proyecto.loc[faltan, 'm'].astype(float)\n"
                       "                                   - datos_proyecto.loc[faltan, 'b'].astype(float))\n"
                       "datos_proyecto[['a', 'm', 'b', 'duration']] = datos_proyecto[['a', 'm', 'b', 'duration']].astype(float)\n"
                       "datos_proyecto['variance'] = datos_proyecto.eval('((b - a) / 6) ** 2')\n"
                       "datos_proyecto['standard_deviation'] = np.sqrt(datos_proyecto['variance'])\n"
                       "lineas = []\n"
                       "for x in sin_a:\n"
                       "    D_, m_x, b_x, a_x = (float(datos_proyecto.loc[x, c]) for c in ('duration', 'm', 'b', 'a'))\n"
                       "    lineas.append(f'- **{x}**: a = 6·{D_:g} − 4·{m_x:g} − {b_x:g} = **{plural(a_x, \"semana\")}**; '\n"
                       "                  f'σ² = (({b_x:g} − {a_x:g})/6)² = **{datos_proyecto.loc[x, \"variance\"]:.3f}**')\n"
                       "Markdown('\\n'.join(lineas))"))
    celdas.append(code('#| tbl-cap: "Estimaciones completas, duración media y varianza de cada actividad"\n#| label: tbl-estimacionesCompletas\n'
                       "limpia_tabla(datos_proyecto[['a', 'm', 'b', 'duration', 'variance']]\n"
                       "             .rename(columns={'duration': 'media D', 'variance': 'varianza σ²'}), decimales=3)"))
    celdas.append(code("Markdown(f'**Respuesta:** ' + '; '.join(f'a({x}) = {plural(datos_proyecto.loc[x, \"a\"], \"semana\")}' for x in sin_a) + '.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duraciones = datos_proyecto['duration'].round().astype(int)"))
    celdas += fig_grafo(durations="duraciones", cap="Grafo PERT con las duraciones medias y el camino crítico", label="fig-pert")
    celdas.append(code("duracion_proyecto = proyecto.duration(duraciones)\n"
                       "display(explica_camino_critico(proyecto, duraciones))\n"
                       "Markdown(f'**Respuesta:** la duración media del proyecto es {plural(duracion_proyecto, \"semana\")}. Las estimaciones optimista y pesimista '\n"
                       "         'sólo intervienen a través de la media D = (a + 4m + b)/6 de cada actividad: el camino crítico y la duración media se calculan '\n"
                       "         'con esas medias, exactamente igual que con duraciones ciertas. Donde $a$ y $b$ cuentan de verdad es en la varianza '\n"
                       "         '((b − a)/6)², y por tanto en la probabilidad de cumplir un plazo.')"))
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
                       "g = lambda x: f'{x:g}'\n"
                       f"Markdown(f'''Para {comparar}, con a = {{g(a_)}}, m = {{g(m_)}} y b = {{g(b_)}}:\n\n"
                       "- Beta-PERT: media = ({g(a_)} + 4·{g(m_)} + {g(b_)})/6 = {comparacion.loc['media', 'Beta-PERT']:.3f}; "
                       "varianza = (({g(b_)} − {g(a_)})/6)² = {comparacion.loc['varianza', 'Beta-PERT']:.3f}; σ = {comparacion.loc['desviación típica', 'Beta-PERT']:.3f}\n"
                       "- Triangular: media = ({g(a_)} + {g(m_)} + {g(b_)})/3 = {comparacion.loc['media', 'Triangular']:.3f}; "
                       "varianza = ({g(a_)}² + {g(m_)}² + {g(b_)}² − {g(a_)}·{g(b_)} − {g(a_)}·{g(m_)} − {g(b_)}·{g(m_)})/18 = {comparacion.loc['varianza', 'Triangular']:.3f}; "
                       "σ = {comparacion.loc['desviación típica', 'Triangular']:.3f}\n"
                       "- Uniforme: media = ({g(a_)} + {g(b_)})/2 = {comparacion.loc['media', 'Uniforme']:.3f}; "
                       "varianza = ({g(b_)} − {g(a_)})²/12 = {comparacion.loc['varianza', 'Uniforme']:.3f}; σ = {comparacion.loc['desviación típica', 'Uniforme']:.3f}''')"))
    celdas.append(code('#| tbl-cap: "Media y dispersión según la distribución supuesta"\n#| label: tbl-comparacion\n'
                       "comparacion.round(3)"))
    celdas.append(code("simetrica = np.isclose(m_ - a_, b_ - m_)\n"
                       "if simetrica:\n"
                       "    medias_txt = (f'Como la moda está centrada ({g(m_)} − {g(a_)} = {g(b_)} − {g(m_)}), las tres distribuciones son simétricas '\n"
                       "                  f'y tienen la misma media, {comparacion.loc[\"media\", \"Beta-PERT\"]:g}; sólo difieren en la dispersión.')\n"
                       "else:\n"
                       "    lado = 'derecha' if (b_ - m_) > (m_ - a_) else 'izquierda'\n"
                       "    medias_txt = (f'La moda no está centrada (cola larga hacia la {lado}), así que las medias también difieren: la Beta-PERT pesa la moda '\n"
                       "                  f'cuatro veces y se queda más cerca de ella ({comparacion.loc[\"media\", \"Beta-PERT\"]:.2f}) que la triangular '\n"
                       "                  f'({comparacion.loc[\"media\", \"Triangular\"]:.2f}) y la uniforme ({comparacion.loc[\"media\", \"Uniforme\"]:.2f}).')\n"
                       "s = comparacion.loc['desviación típica']\n"
                       "Markdown(f'''La Beta-PERT concentra la probabilidad cerca de la moda y es la menos dispersa (σ = {s['Beta-PERT']:.3f}); la uniforme reparte "
                       "la probabilidad por igual entre {g(a_)} y {g(b_)} y es la más dispersa (σ = {s['Uniforme']:.3f}); la triangular queda en medio "
                       "(σ = {s['Triangular']:.3f}). {medias_txt}\n\n"
                       "**Respuesta:** la hipótesis más prudente es la uniforme, porque es la de mayor desviación típica ({s['Uniforme']:.3f} frente a "
                       "{s['Triangular']:.3f} y {s['Beta-PERT']:.3f}) y conduce a plazos con más margen; pero sólo está justificada si de verdad no se sabe "
                       "nada de la duración entre los extremos.''')"))
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
    celdas.append(md("Con las tres estimaciones $a$, $m$, $b$ del cuadro, el método PERT da para cada actividad "
                     "$D = (a + 4m + b)/6$ y $\\sigma^2 = ((b - a)/6)^2$:"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['duration'] = datos_proyecto.eval('(a + 4 * m + b) / 6').round().astype(int)\n"
                       "datos_proyecto['variance'] = datos_proyecto.eval('((b - a) / 6) ** 2')"))
    celdas.append(code('#| tbl-cap: "Media y varianza PERT de cada actividad con los números sustituidos"\n#| label: tbl-estimaciones\n'
                       "detalle_estimaciones(datos_proyecto)"))
    celdas.append(code("fijas = [x for x in datos_proyecto.index if datos_proyecto.loc[x, 'a'] == datos_proyecto.loc[x, 'b']]\n"
                       "Markdown(f'''Cuando $a = m = b$ la varianza es $((b - a)/6)^2 = 0$: la actividad no aporta incertidumbre, aunque sí aporta duración.\n\n"
                       "**Respuesta:** las actividades de duración fija son {lista_y(fijas)} y su varianza es 0 ('''\n"
                       "         + lista_y([f'{x} dura siempre {plural(datos_proyecto.loc[x, \"duration\"], \"semana\")}' for x in fijas])\n"
                       "         + '''). La media y la varianza de las demás actividades son las de la tabla.''')"))
    celdas += apartado(2)
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con duraciones medias y camino crítico", label="fig-pert")
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "criticas = ordena_ruta(sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)), calendario)\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "Markdown(f'**Respuesta:** camino crítico {\" – \".join(criticas)}; duración media del proyecto {plural(duracion_media, \"semana\")}.')"))
    celdas += apartado(3)
    celdas.append(md("La varianza del proyecto se aproxima por la de la ruta crítica: la suma de las varianzas de sus "
                     "actividades (que se suponen independientes)."))
    celdas.append(code('#| tbl-cap: "Varianza de la ruta crítica"\n#| label: tbl-varianzaRuta\n'
                       "desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['variance'])\n"
                       "varianza_rutas.relabel_index(['actividades', 'varianza', 'suma de varianzas'], axis='columns')"))
    celdas.append(code('#| tbl-cap: "Varianza de cada actividad y si es crítica"\n#| label: tbl-varianzas\n'
                       "contribuyen = datos_proyecto.loc[criticas, 'variance']\n"
                       "mas_variable_no_critica = datos_proyecto['variance'].drop(criticas).idxmax()\n"
                       "limpia_tabla(pd.DataFrame({'crítica': datos_proyecto.index.isin(criticas), 'holgura total': calendario['H_total'],\n"
                       "                           'varianza': datos_proyecto['variance']}, index=datos_proyecto.index), decimales=3).T"))
    celdas.append(code("varianza_proyecto = float(contribuyen.sum())\n"
                       "aportan = [a for a in criticas if contribuyen[a] > 0]\n"
                       "fijas_criticas = [a for a in criticas if contribuyen[a] == 0]\n"
                       "h_nc = int(calendario.loc[mas_variable_no_critica, 'H_total'])\n"
                       "if h_nc <= 2:\n"
                       "    matiz = (f'Sin embargo su holgura total es sólo de {plural(h_nc, \"semana\")}: en la realidad (y en una simulación de Monte Carlo) '\n"
                       "             'un retraso de ese tamaño la haría crítica, así que conviene vigilarla.')\n"
                       "else:\n"
                       "    matiz = (f'Con {plural(h_nc, \"semana\")} de holgura total es poco probable que llegue a ser crítica, '\n"
                       "             'así que despreciar su varianza es razonable.')\n"
                       "Markdown(f'''{explica_sigma_ruta(criticas, datos_proyecto[\"variance\"], nombre=\"σ_T\")} semanas, es decir, "
                       "σ²_T = {varianza_proyecto:.3f}.\n\n"
                       "**Respuesta:** la varianza del proyecto es {varianza_proyecto:.3f} (σ = {np.sqrt(varianza_proyecto):.3f} semanas). Contribuyen "
                       "{lista_y(aportan)}, las críticas con varianza no nula. La no crítica más variable, {mas_variable_no_critica} "
                       "(σ² = {datos_proyecto.loc[mas_variable_no_critica, \"variance\"]:.3f}), no contribuye en la aproximación PERT porque no está en "
                       "la ruta crítica. {matiz} "
                       "{('Las críticas de duración fija (' + lista_y(fijas_criticas) + ') aportan duración pero no varianza.') if fijas_criticas else 'Ninguna crítica es de duración fija.'}''')"))
    celdas += apartado(4)
    celdas.append(code("maxima = contribuyen.max()\n"
                       "mas_variables = [a for a in criticas if np.isclose(contribuyen[a], maxima)]\n"
                       "# si la actividad se hiciera de duración fija igual a su valor más probable m\n"
                       "efecto = {}\n"
                       "for x in criticas:\n"
                       "    d2 = datos_proyecto['duration'].copy(); d2[x] = int(datos_proyecto.loc[x, 'm'])\n"
                       "    efecto[x] = int(proyecto.duration(d2))\n"
                       "mejor = min(efecto, key=lambda x: (efecto[x], -contribuyen[x]))\n"
                       "tabla_efecto = limpia_tabla(pd.DataFrame({'m': datos_proyecto.loc[criticas, 'm'], 'media D': datos_proyecto.loc[criticas, 'duration'],\n"
                       "                                          'varianza': contribuyen,\n"
                       "                                          'σ²_T sin su varianza': varianza_proyecto - contribuyen,\n"
                       "                                          'duración si se fija en m': pd.Series(efecto)}), decimales=3)\n"
                       "display(tabla_efecto.T)\n"
                       "if len(mas_variables) == 1:\n"
                       "    quien = f'{mas_variables[0]} (la crítica de mayor varianza, {maxima:.3f}): sin ella'\n"
                       "else:\n"
                       "    quien = f'{lista_y(mas_variables, \"o\")} (empatadas como críticas de mayor varianza, {maxima:.3f} cada una): sin la de cualquiera de ellas'\n"
                       "texto_pred = (f'{quien} σ²_T bajaría de {varianza_proyecto:.3f} '\n"
                       "              f'a {varianza_proyecto - maxima:.3f} (σ_T = {np.sqrt(varianza_proyecto - maxima):.3f} semanas)')\n"
                       "if efecto[mejor] < duracion_media:\n"
                       "    texto_acortar = (f'Anular la varianza manteniendo la media no cambia la duración media ({duracion_media}). Si anularla supone fijar la '\n"
                       "                     f'actividad en su valor más probable m, la que más acorta es {mejor}: m = {int(datos_proyecto.loc[mejor, \"m\"])} < D = '\n"
                       "                     f'{datos_proyecto.loc[mejor, \"duration\"]}, y el proyecto pasaría a durar {efecto[mejor]} semanas.')\n"
                       "else:\n"
                       "    texto_acortar = (f'Anular la varianza manteniendo la media no cambia la duración media ({duracion_media}), y fijar una crítica en su '\n"
                       "                     f'valor más probable tampoco la acorta (ninguna crítica tiene m < D): para acortar el plazo hay que reducir la duración '\n"
                       "                     f'de alguna crítica, no su varianza.')\n"
                       "Markdown(f'''**Respuesta:** para hacer el plazo más predecible elegiría {texto_pred}. {texto_acortar} '''\n"
                       "         'Reducir la varianza de una no crítica no cambia nada en el análisis PERT.')"))
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
