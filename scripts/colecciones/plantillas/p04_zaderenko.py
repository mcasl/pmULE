"""Colección 04 — Método de Zaderenko: tiempos tempranos y tardíos de los
nodos, calendario de las actividades y diagrama de Gantt."""
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="04_zaderenko", prefijo="Zaderenko",
            titulo="Método de Zaderenko y calendario",
            descripcion="Calcular con la matriz de Zaderenko los tiempos tempranos y tardíos, el calendario "
                        "de las actividades y el diagrama de Gantt.",
            n_defecto=3)


def _base(k, rng):
    n = {1: 8, 2: 9, 3: 10}[(k - 1) % 3 + 1]
    info = P.buscar(rng, lambda i: 7 <= i["n_nodos"] <= 9 and 1 <= i["n_dummies"] <= 3 and i["n_rutas"] <= 9,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    cal = info["calendario"]
    no_criticas = [a for a in cal.index if cal.loc[a, "H_total"] > 0]
    criticas = [a for a in cal.index if cal.loc[a, "H_total"] == 0]
    cambio = no_criticas[int(rng.integers(0, len(no_criticas)))] if no_criticas else criticas[0]
    extra = max(1, int(cal.loc[cambio, "H_total"]) + int(rng.integers(-1, 3)))

    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta.")]
    pregs = ["Dibuje el grafo PERT y numere sus nodos.",
             "Construya la matriz de Zaderenko y calcule con ella los tiempos más tempranos y más tardíos "
             "de cada nodo. ¿Cuál es la duración del proyecto?",
             "Calcule el calendario del proyecto: inicio y fin más tempranos y más tardíos de cada actividad.",
             "Dibuje el diagrama de Gantt con las actividades en su inicio más temprano e indique sobre él "
             "las holguras.",
             f"La actividad {cambio} va a durar {extra} días más de lo previsto. Recalcule la duración del "
             f"proyecto y explique el resultado a partir de la holgura de {cambio}."]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas.append(code('#| fig-cap: "Grafo PERT con numeración de nodos"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("En la matriz de Zaderenko la casilla (i, j) contiene la duración de la actividad que va "
                     "del nodo i al nodo j (cero para las ficticias). El tiempo más temprano de un nodo es el "
                     "máximo, sobre su columna, de «tiempo temprano del origen más duración»; el más tardío es "
                     "el mínimo, sobre su fila, de «tiempo tardío del destino menos duración». Los tempranos "
                     "se calculan de izquierda a derecha y los tardíos de derecha a izquierda, partiendo del "
                     "tiempo temprano del último nodo."))
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos y tardíos de los nodos"\n#| label: tbl-tiemposNodos\n'
                       "resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])\n"
                       "resultados_pert['nodes']"))
    celdas.append(code("duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "Markdown(f'La duración del proyecto es el tiempo más temprano del último nodo: {duracion_proyecto} días.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Para una actividad que va del nodo i al nodo j: el inicio más temprano es el tiempo "
                     "temprano de i, el fin más temprano es ese inicio más la duración, el fin más tardío es el "
                     "tiempo tardío de j y el inicio más tardío es ese fin menos la duración."))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Grafo PERT con tiempos de los nodos y camino crítico"\n#| label: fig-pertTiempos\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| fig-cap: "Diagrama de Gantt (inicios más tempranos) con indicación de las holguras"\n#| label: fig-gantt\n'
                       "gantt_df, dibujo = proyecto.gantt(datos_proyecto, 'duration', holguras=True, tikz=True)\n"
                       "pinta_tikz(dibujo)"))
    celdas += apartado(ap); ap += 1
    celdas.append(code(f"holgura = int(calendario.loc['{cambio}', 'H_total'])\n"
                       "duraciones_nuevas = datos_proyecto['duration'].copy()\n"
                       f"duraciones_nuevas['{cambio}'] += {extra}\n"
                       "duracion_nueva = proyecto.duration(duraciones_nuevas)\n"
                       f"Markdown(f'La holgura total de {cambio} es {{holgura}} días. Un retraso de {extra} días '\n"
                       f"         f'{{\"no \" if {extra} <= holgura else \"\"}}supera esa holgura, y la duración del proyecto pasa de '\n"
                       "         f'{duracion_proyecto} a {duracion_nueva} días.')"))
    celdas.append(code('#| tbl-cap: "Calendario con la nueva duración"\n#| label: tbl-calendarioNuevo\n'
                       "proyecto.calendar(duraciones_nuevas)[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    sub = f"{n} actividades, {info['n_nodos']} nodos, {info['n_dummies']} ficticias"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _matriz_dada(rng):
    """Se da la matriz de Zaderenko rellena con las actividades y sus duraciones; hay que
    reconstruir el cuadro de prelaciones, el grafo y los tiempos."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 6 <= i["n_nodos"] <= 8 and 1 <= i["n_dummies"] <= 3,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, g = info["D"], info["proyecto"]
    nodos = list(g.nodes)
    celdas_matriz = {}
    for u, v, d in g.pert_graph.edges(data=True):
        act = d["activity"]
        celdas_matriz[(u, v)] = "ficticia" if act.startswith("@") else f"{act} ({int(D[act])})"
    lineas = ["matriz_enunciado = pd.DataFrame('', index=" + repr(nodos) + ", columns=" + repr(nodos) + ")"]
    for (u, v), txt in sorted(celdas_matriz.items()):
        lineas.append(f"matriz_enunciado.loc['{u}', '{v}'] = {txt!r}")
    lineas.append("matriz_enunciado.index.name = 'nodo'")
    lineas.append("matriz_enunciado")
    celdas = [md("# Enunciado"),
              md("De un proyecto sólo se conserva la matriz de Zaderenko sin calcular: en la casilla (i, j) figura la "
                 "actividad que va del nodo i al nodo j y, entre paréntesis, su duración en días; las ficticias "
                 "aparecen como tales. Los nodos están numerados de modo que todo arco va de un número menor a "
                 "uno mayor.")]
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko del enunciado"\n#| label: tbl-matrizEnunciado\n' + "\n".join(lineas)))
    celdas.append(preguntas([
        "Reconstruya el cuadro de prelaciones inmediatas del proyecto (las ficticias transmiten prelaciones pero no son actividades).",
        "Dibuje el grafo PERT y compruebe que reproduce la matriz.",
        "Calcule los tiempos más tempranos y más tardíos de los nodos y la duración del proyecto.",
        "Obtenga el calendario de las actividades y el camino crítico."]))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(md("Cada casilla no vacía es un arco (i → j). Las predecesoras de una actividad que sale del "
                     "nodo i son las actividades que entran en i, más las que entran en los nodos desde los que "
                     "llega una ficticia a i (siguiendo las cadenas de ficticias hacia atrás)."))
    celdas.append(code("arcos = {}\n"
                       "for i in matriz_enunciado.index:\n"
                       "    for j in matriz_enunciado.columns:\n"
                       "        txt = matriz_enunciado.loc[i, j]\n"
                       "        if txt:\n"
                       "            arcos[(i, j)] = ('ficticia' if txt == 'ficticia' else txt.split(' ')[0], 0 if txt == 'ficticia' else int(txt.split('(')[1].rstrip(')')))\n"
                       "def entran(nodo):\n"
                       "    acts = set()\n"
                       "    for (i, j), (nombre, _) in arcos.items():\n"
                       "        if j == nodo:\n"
                       "            acts |= {nombre} if nombre != 'ficticia' else entran(i)\n"
                       "    return acts\n"
                       "prelaciones_dict, duraciones = {}, {}\n"
                       "for (i, j), (nombre, d) in arcos.items():\n"
                       "    if nombre != 'ficticia':\n"
                       "        prelaciones_dict[nombre] = ','.join(sorted(entran(i))) or '---'\n"
                       "        duraciones[nombre] = d\n"
                       "datos_proyecto = pd.DataFrame({'predecessor': pd.Series(prelaciones_dict), 'duration': pd.Series(duraciones)}).sort_index()\n"
                       "datos_proyecto.index.name = 'activity'\n"
                       "datos_proyecto"))
    celdas += apartado(2)
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas.append(code('#| fig-cap: "Grafo PERT reconstruido"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(md("La numeración de los nodos puede diferir de la del enunciado (hay varias numeraciones "
                     "válidas), pero la matriz tiene las mismas actividades en las mismas posiciones relativas."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko calculada"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(code("resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])\n"
                       "display(resultados_pert['nodes'].T)\n"
                       "Markdown(f'La duración del proyecto es {proyecto.duration(datos_proyecto[\"duration\"])} días.')"))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "proyecto.calendar(datos_proyecto['duration'])[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code("camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])"))
    return celdas, f"{n} actividades, {info['n_nodos']} nodos; reconstrucción desde la matriz"


def _prediccion(rng):
    """Antes de calcular: ¿cambia el plazo si una actividad dura más? Tres casos y comprobación."""
    n = int(rng.choice([8, 9]))

    def acepta(i):
        cal = i["calendario"]
        return (1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1 and (cal["H_total"] >= 3).sum() >= 2
                and ((cal["H_total"] > 0) & (cal["H_total"] <= 2)).sum() >= 1)

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    critica = cal.index[cal["H_total"] == 0].tolist()
    holgada = cal.index[cal["H_total"] >= 3].tolist()
    justa = cal.index[(cal["H_total"] > 0) & (cal["H_total"] <= 2)].tolist()
    c1 = critica[int(rng.integers(0, len(critica)))]
    c2 = holgada[int(rng.integers(0, len(holgada)))]
    c3 = justa[int(rng.integers(0, len(justa)))]
    casos = [(c1, 2), (c2, 2), (c3, int(cal.loc[c3, "H_total"]) + 1)]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta. "
                 "Antes de hacer ningún cálculo numérico, conteste razonando sobre el grafo; después compruébelo.")]
    celdas.append(preguntas([
        "Dibuje el grafo PERT y, mirando sólo su forma, indique qué actividades parecen estar en la ruta más larga.",
        f"Prediga, sin calcular tiempos, si el plazo del proyecto cambia en cada uno de estos casos: "
        f"(a) {casos[0][0]} dura {casos[0][1]} días más; (b) {casos[1][0]} dura {casos[1][1]} días más; "
        f"(c) {casos[2][0]} dura {casos[2][1]} días más. Anote su predicción.",
        "Calcule con Zaderenko los tiempos de los nodos, el calendario y las holguras.",
        "Compruebe las tres predicciones recalculando la duración. ¿Qué dato del calendario permitía acertar sin recalcular?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo PERT con numeración de nodos"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(md("A ojo sólo puede intuirse: las rutas con más actividades o con actividades largas son las "
                     "candidatas. La forma del grafo no basta; hacen falta las duraciones."))
    celdas += apartado(2)
    celdas.append(md("Una predicción razonada: el plazo sólo cambia si la actividad está en la ruta más larga, o si el "
                     "aumento supera la holgura total de la actividad. Sin cálculo sólo es seguro el primer caso si "
                     "se ha identificado bien la ruta crítica; para los otros dos hace falta saber la holgura."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(code('#| tbl-cap: "Calendario y holguras"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas += apartado(4)
    celdas.append(code("casos = " + repr(casos) + "\n"
                       "filas = []\n"
                       "for act, extra in casos:\n"
                       "    d2 = datos_proyecto['duration'].copy(); d2[act] += extra\n"
                       "    nueva = proyecto.duration(d2)\n"
                       "    filas.append({'actividad': act, 'días de más': extra, 'holgura total': int(calendario.loc[act, 'H_total']),\n"
                       "                  'plazo antes': duracion_proyecto, 'plazo después': nueva,\n"
                       "                  'cambia': 'sí' if nueva > duracion_proyecto else 'no'})\n"
                       "pd.DataFrame(filas).set_index('actividad')"))
    celdas.append(md("La regla es: el plazo cambia si y sólo si el retraso supera la holgura total de la actividad "
                     "(para una actividad crítica la holgura es cero y cualquier retraso cuenta). El exceso sobre la "
                     "holgura es exactamente lo que se alarga el proyecto."))
    return celdas, f"{n} actividades; predicción cualitativa sobre {c1}, {c2} y {c3}"


VARIANTES = [
    Variante("zaderenko", 1, "matriz de Zaderenko, tiempos de los nodos, calendario, Gantt y efecto de un retraso",
             lambda rng: _base(1, rng)),
    Variante("zaderenko-b", 1, "ídem con un proyecto mayor", lambda rng: _base(2, rng)),
    Variante("zaderenko-c", 2, "ídem con diez actividades", lambda rng: _base(3, rng)),
    Variante("matriz-dada", 3, "inverso: de la matriz de Zaderenko rellena al cuadro de prelaciones, el grafo y los tiempos",
             _matriz_dada),
    Variante("prediccion", 2, "predecir sin calcular si un retraso cambia el plazo y comprobarlo con las holguras",
             _prediccion),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
