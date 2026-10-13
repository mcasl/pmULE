"""Colección 04 — Método de Zaderenko: tiempos tempranos y tardíos de los
nodos, calendario de las actividades y diagrama de Gantt."""
from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

META = dict(slug="04_zaderenko", prefijo="Zaderenko",
            titulo="Método de Zaderenko y calendario",
            descripcion="Calcular con la matriz de Zaderenko los tiempos tempranos y tardíos, el calendario "
                        "de las actividades y el diagrama de Gantt.",
            n_defecto=3)


def generar(k, rng):
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
