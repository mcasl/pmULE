"""Colección 04 — Método de Zaderenko: tiempos tempranos y tardíos de los
nodos, calendario de las actividades y diagrama de Gantt."""
import networkx as nx
import pandas as pd
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX, ESTADO)

META = dict(slug="04_zaderenko", prefijo="Zaderenko",
            titulo="Método de Zaderenko y calendario",
            descripcion="Calcular con la matriz de Zaderenko los tiempos tempranos y tardíos, el calendario "
                        "de las actividades y el diagrama de Gantt.",
            n_defecto=3)

def _roy_ilustrativo():
    """Grafo Roy equivalente, como ilustración (la matriz de Zaderenko opera sobre el PERT)."""
    if ESTADO["grafo"] == "roy":
        return []
    return [md("A modo ilustrativo, este mismo proyecto como grafo Roy (actividades en los nodos); "
               "la matriz de Zaderenko se construye sobre los sucesos del grafo PERT."),
            *fig_grafo(durations="False", cap="Grafo Roy equivalente (ilustrativo)",
                       label="fig-royIlustrativo", tipo="roy")]



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
             "Dibuje el diagrama de Gantt con las actividades en su inicio más temprano, con las flechas de "
             "prelación entre barras, e indique sobre él las holguras.",
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
    celdas += fig_grafo(durations="False", cap="Grafo PERT con numeración de nodos", label="fig-pert")
    celdas += _roy_ilustrativo()
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
    celdas.append(md("Cada tiempo temprano es el máximo, sobre los arcos que llegan al nodo, de «tiempo temprano del "
                     "origen + duración del arco»; cada tiempo tardío, el mínimo, sobre los arcos que salen, de «tiempo "
                     "tardío del destino − duración». Las ficticias (@) aportan duración 0. Las tablas recogen los "
                     "candidatos que se comparan en cada nodo."))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos: candidatos en cada nodo (de izquierda a derecha)"\n#| label: tbl-tempranosPaso\n'
                       "tiempos_tempranos, tiempos_tardios = detalle_tiempos_nodos(proyecto, datos_proyecto['duration'])\n"
                       "tiempos_tempranos"))
    celdas.append(code('#| tbl-cap: "Tiempos tardíos: candidatos en cada nodo (de derecha a izquierda)"\n#| label: tbl-tardiosPaso\n'
                       "tiempos_tardios"))
    celdas.append(code("duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "Markdown(f'La duración del proyecto es el tiempo más temprano del último nodo: {duracion_proyecto} días.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Para una actividad que va del nodo i al nodo j: el inicio más temprano es el tiempo "
                     "temprano de i, el fin más temprano es ese inicio más la duración, el fin más tardío es el "
                     "tiempo tardío de j y el inicio más tardío es ese fin menos la duración."))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| tbl-cap: "Fechas de cada actividad a partir de los tiempos de sus nodos"\n#| label: tbl-fechasPaso\n'
                       "detalle_holguras(calendario, prelaciones)[['ES = máx EF pred.', 'EF = ES + d', 'LF = mín LS suc.', 'LS = LF − d', 'H_total = LS − ES']]"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos de los nodos y camino crítico", label="fig-pertTiempos")
    celdas += apartado(ap); ap += 1
    celdas.append(md("Las flechas unen el final de cada predecesora con el inicio de su sucesora; en rojo, las "
                     "prelaciones críticas (sin hueco entre barras críticas)."))
    celdas.append(code('#| fig-cap: "Diagrama de Gantt (inicios más tempranos) con flechas de prelación y holguras"\n#| label: fig-gantt\n'
                       "gantt_df, dibujo = proyecto.gantt(datos_proyecto, 'duration', holguras=True, tikz=True, params=GANTT_PARAMS)\n"
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
    celdas += fig_grafo(durations="False", cap="Grafo PERT reconstruido", label="fig-pert")
    celdas += _roy_ilustrativo()
    celdas.append(md("La numeración de los nodos puede diferir de la del enunciado (hay varias numeraciones "
                     "válidas), pero la matriz tiene las mismas actividades en las mismas posiciones relativas."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko calculada"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(code("resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])\n"
                       "display(resultados_pert['nodes'].T)\n"
                       "Markdown(f'La duración del proyecto es {proyecto.duration(datos_proyecto[\"duration\"])} días.')"))
    celdas.append(md("Cada tiempo temprano es el máximo, sobre los arcos que llegan al nodo, de «tiempo temprano del "
                     "origen + duración del arco»; cada tiempo tardío, el mínimo, sobre los arcos que salen, de «tiempo "
                     "tardío del destino − duración». Las ficticias (@) aportan duración 0. Las tablas recogen los "
                     "candidatos que se comparan en cada nodo."))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos: candidatos en cada nodo (de izquierda a derecha)"\n#| label: tbl-tempranosPaso\n'
                       "tiempos_tempranos, tiempos_tardios = detalle_tiempos_nodos(proyecto, datos_proyecto['duration'])\n"
                       "tiempos_tempranos"))
    celdas.append(code('#| tbl-cap: "Tiempos tardíos: candidatos en cada nodo (de derecha a izquierda)"\n#| label: tbl-tardiosPaso\n'
                       "tiempos_tardios"))
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
    celdas += fig_grafo(durations="False", cap="Grafo PERT con numeración de nodos", label="fig-pert")
    celdas += _roy_ilustrativo()
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
    celdas.append(code('#| tbl-cap: "Fechas de cada actividad a partir de los tiempos de sus nodos"\n#| label: tbl-fechasPaso\n'
                       "detalle_holguras(calendario, prelaciones)[['ES = máx EF pred.', 'EF = ES + d', 'LF = mín LS suc.', 'LS = LF − d', 'H_total = LS − ES']]"))
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



def _plazo_impuesto(rng):
    """El cliente impone un plazo menor que la duración: holguras negativas y qué rutas hay que acortar."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and i["segunda"] is not None and i["T"] - i["segunda"] <= 2,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, T, L = info["D"], info["calendario"], int(info["T"]), info["L"]
    k = int(rng.integers(2, 4))
    plazo = T - k
    df = P.tabla_base(info, duration=D)
    n_exceden = int((L > plazo).sum())
    celdas = [md("# Enunciado"),
              md(f"Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta. El "
                 f"cliente exige que esté terminado en **{plazo} semanas**.")]
    celdas.append(preguntas([
        "Calcule con Zaderenko los tiempos de los nodos y el calendario del proyecto. ¿Cuál es su duración? "
        "¿Se cumple el plazo del cliente?",
        f"Recalcule los tiempos tardíos tomando como tiempo tardío del último nodo el plazo impuesto ({plazo}) "
        "en lugar de la duración del proyecto. ¿Qué actividades tienen ahora holgura total negativa y qué "
        "significa?",
        "¿Cuántas rutas del proyecto exceden el plazo y en cuánto cada una? ¿Hay alguna actividad que esté en "
        "todas ellas? ¿Cuántas semanas habría que acortarla para cumplir el plazo si fuera la única que se "
        "acorta?",
        "Si en lugar de acortar actividades se negociara el plazo, ¿cuál es el mínimo plazo que no obliga a "
        "acortar nada?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"plazo = {plazo}"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code("Markdown(f'La duración del proyecto es {duracion_proyecto} semanas, {duracion_proyecto - plazo} más que el plazo: no se cumple.')"))
    celdas += apartado(2)
    celdas.append(md("Los tiempos tempranos no dependen del plazo. Los tardíos sí: si el último nodo debe "
                     "alcanzarse en la semana del plazo, todos los tiempos tardíos (y los inicios y fines más "
                     "tardíos de las actividades) se adelantan en la diferencia entre la duración y el plazo, y "
                     "la holgura total de cada actividad disminuye en esa misma cantidad."))
    celdas.append(code("retraso = duracion_proyecto - plazo\n"
                       "con_plazo = calendario[['inicio_mas_temprano', 'fin_mas_temprano']].copy()\n"
                       "con_plazo['inicio_mas_tardio'] = calendario['inicio_mas_tardio'] - retraso\n"
                       "con_plazo['fin_mas_tardio'] = calendario['fin_mas_tardio'] - retraso\n"
                       "con_plazo['H_total'] = con_plazo['inicio_mas_tardio'] - con_plazo['inicio_mas_temprano']\n"
                       "con_plazo"))
    celdas.append(code("negativas = con_plazo.index[con_plazo['H_total'] < 0].tolist()\n"
                       "Markdown(f'''Con el plazo impuesto, {len(negativas)} actividades tienen holgura total negativa: {\", \".join(negativas)}. "
                       "Una holgura −h significa que, aun empezando lo antes posible, la actividad termina h semanas más tarde de lo que el plazo "
                       "permite: son las actividades de las rutas que exceden el plazo (las críticas tienen −{retraso}), y hay que acortar alguna "
                       "de ellas en cada una de esas rutas.''')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Rutas que exceden el plazo"\n#| label: tbl-rutasExceden\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "duracion_rutas = R @ datos_proyecto['duration'][R.columns]\n"
                       "exceso = (duracion_rutas - plazo)\n"
                       "exceden = exceso[exceso > 0].sort_values(ascending=False)\n"
                       "pd.DataFrame({'actividades': [' - '.join(R.columns[R.loc[r] == 1]) for r in exceden.index],\n"
                       "              'duración': duracion_rutas[exceden.index], 'exceso sobre el plazo': exceden})"))
    celdas.append(code("comunes = [a for a in R.columns if R.loc[exceden.index, a].all()]\n"
                       "if comunes:\n"
                       "    display(Markdown(f'Las actividades {\", \".join(comunes)} están en todas las rutas que exceden el plazo. Acortando una sola de ellas '\n"
                       "                     f'{int(exceden.max())} semanas (el mayor exceso) se cumpliría el plazo, siempre que su duración lo permita.'))\n"
                       "else:\n"
                       "    display(Markdown('Ninguna actividad está en todas las rutas que exceden el plazo: habrá que acortar al menos dos actividades distintas.'))"))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'El mínimo plazo que no obliga a acortar nada es la duración del proyecto, {duracion_proyecto} semanas: con cualquier plazo mayor o igual todas las holguras son no negativas.')"))
    return celdas, f"{n} actividades; duración {T}, plazo impuesto {plazo} ({n_exceden} rutas lo exceden)"


def _gantt_dado(rng):
    """Se da el Gantt (inicio y duración de cada actividad) sin prelaciones: deducir lo que se pueda."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and i["n_criticas"] == 1 and (i["calendario"]["H_total"] > 0).sum() >= 2,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, preds = info["D"], info["calendario"], info["preds"]
    gantt = pd.DataFrame({"start": cal["inicio_mas_temprano"].astype(int), "duration": D})
    gantt.index.name = "activity"
    celdas = [md("# Enunciado"),
              md("De un proyecto sólo se conserva su programación de inicios más tempranos: la semana en que "
                 "empieza cada actividad y su duración (cuadro adjunto). Se sabe que el calendario se calculó "
                 "correctamente a partir de un cuadro de prelaciones que se ha perdido.")]
    celdas.append(preguntas([
        "Dibuje el diagrama de Gantt y calcule el fin más temprano de cada actividad y la duración del proyecto.",
        "¿Qué actividades son iniciales? Para cada una de las demás, ¿qué actividades *pueden* ser "
        "predecesoras inmediatas suyas (terminan no más tarde de su inicio) y cuáles son las «ajustadas» "
        "(terminan exactamente cuando ella empieza)? ¿Por qué toda actividad no inicial tiene al menos una "
        "predecesora ajustada?",
        "Encadenando actividades ajustadas desde la semana 0 hasta el final, identifique una ruta crítica. "
        "¿Puede haber más de una?",
        "Se recupera el cuadro de prelaciones original. Compruebe que es compatible con el Gantt y compare "
        "sus prelaciones con las predecesoras ajustadas deducidas."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(gantt)))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['end'] = datos_proyecto['start'] + datos_proyecto['duration']\n"
                       "duracion_proyecto = int(datos_proyecto['end'].max())\n"
                       "datos_proyecto"))
    celdas.append(code('#| fig-cap: "Diagrama de Gantt de la programación conservada"\n#| label: fig-gantt\n'
                       "import matplotlib.pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 0.4 * len(datos_proyecto) + 1))\n"
                       "for y, (a, fila) in enumerate(datos_proyecto.iterrows()):\n"
                       "    ax.barh(y, fila['duration'], left=fila['start'], color='tab:blue')\n"
                       "    ax.text(fila['start'] + fila['duration'] / 2, y, a, ha='center', va='center', color='white')\n"
                       "ax.set_yticks(range(len(datos_proyecto))); ax.set_yticklabels(datos_proyecto.index); ax.invert_yaxis()\n"
                       "ax.set_xlabel('semana'); ax.set_xticks(range(duracion_proyecto + 1)); ax.grid(axis='x', alpha=0.3)\n"
                       "plt.show()"))
    celdas.append(code("Markdown(f'La duración del proyecto es el mayor fin: {duracion_proyecto} semanas.')"))
    celdas += apartado(2)
    celdas.append(md("Una actividad que empieza en la semana 0 es inicial. Si $P$ terminara después de que "
                     "empiece $A$, $P$ no podría ser predecesora de $A$. En un calendario de inicios más "
                     "tempranos, el inicio de cada actividad no inicial es el *máximo* de los fines de sus "
                     "predecesoras, luego alguna de ellas termina exactamente cuando la actividad empieza."))
    celdas.append(code("posibles, ajustadas = {}, {}\n"
                       "for a, fila in datos_proyecto.iterrows():\n"
                       "    posibles[a] = sorted(p for p, f in datos_proyecto.iterrows() if p != a and f['end'] <= fila['start'])\n"
                       "    ajustadas[a] = sorted(p for p in posibles[a] if datos_proyecto.loc[p, 'end'] == fila['start'])\n"
                       "iniciales = datos_proyecto.index[datos_proyecto['start'] == 0].tolist()\n"
                       "print('Iniciales:', ', '.join(iniciales))\n"
                       "pd.DataFrame({'posibles predecesoras': {a: ', '.join(v) or '—' for a, v in posibles.items()},\n"
                       "              'ajustadas (fin = inicio)': {a: ', '.join(v) or '—' for a, v in ajustadas.items()}})"))
    celdas += apartado(3)
    celdas.append(code("finales = datos_proyecto.index[datos_proyecto['end'] == duracion_proyecto].tolist()\n"
                       "def cadenas(a):\n"
                       "    if datos_proyecto.loc[a, 'start'] == 0:\n"
                       "        return [[a]]\n"
                       "    return [c + [a] for p in ajustadas[a] for c in cadenas(p)]\n"
                       "rutas_criticas_deducidas = [c for f in finales for c in cadenas(f)]\n"
                       "for c in rutas_criticas_deducidas:\n"
                       "    print(' - '.join(c), '=', ' + '.join(str(int(datos_proyecto.loc[x, 'duration'])) for x in c), '=', duracion_proyecto)\n"
                       "Markdown(f'Se deducen {len(rutas_criticas_deducidas)} cadena(s) ajustada(s) de la semana 0 al final: cada una es una ruta crítica '\n"
                       "         '*si* todas sus prelaciones consecutivas existen en el cuadro perdido; puede haber más de una.')"))
    celdas += apartado(4)
    tabla = P.tabla_base(info, duration=D)
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones recuperado"\n#| label: tbl-prelacionesRecuperado\n'
                       + literal_dataframe(tabla, nombre="prelaciones_recuperadas")))
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(prelaciones_recuperadas, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(prelaciones_recuperadas['duration'])\n"
                       "compatible = (calendario['inicio_mas_temprano'] == datos_proyecto['start']).all()\n"
                       "comparacion = pd.DataFrame({'predecesoras reales': {a: ', '.join(sorted(ps)) or '—' for a, ps in prelaciones.immediate_predecessor.items()},\n"
                       "                            'ajustadas deducidas': {a: ', '.join(v) or '—' for a, v in ajustadas.items()}})\n"
                       "G = nx.DiGraph([(p, a) for a, ps in prelaciones.immediate_predecessor.items() for p in ps]); G.add_nodes_from(comparacion.index)\n"
                       "comparacion['ajustadas que son antecesoras'] = [', '.join(sorted(set(ajustadas[a]) & nx.ancestors(G, a))) or '—' for a in comparacion.index]\n"
                       "comparacion['ajustadas por coincidencia'] = [', '.join(sorted(set(ajustadas[a]) - nx.ancestors(G, a))) or '—' for a in comparacion.index]\n"
                       "display(comparacion)\n"
                       "Markdown(f'El cuadro recuperado reproduce el Gantt: {compatible}. Una predecesora ajustada suele ser una antecesora real (inmediata o no), '\n"
                       "         'pero puede ser una coincidencia de fechas; y las predecesoras reales con holgura entre su fin y el inicio de la sucesora no se ven en el Gantt. '\n"
                       "         'Por eso el Gantt no determina el cuadro de prelaciones.')"))
    celdas += fig_grafo(durations="prelaciones_recuperadas['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += _roy_ilustrativo()
    return celdas, f"{n} actividades; sólo se conoce el Gantt de inicios tempranos"


VARIANTES = [
    Variante("zaderenko", 1, "matriz de Zaderenko, tiempos de los nodos, calendario, Gantt y efecto de un retraso",
             lambda rng: _base(1, rng), grafo="pert"),
    Variante("zaderenko-b", 1, "ídem con un proyecto mayor", lambda rng: _base(2, rng), grafo="pert"),
    Variante("zaderenko-c", 2, "ídem con diez actividades", lambda rng: _base(3, rng), grafo="pert"),
    Variante("matriz-dada", 3, "inverso: de la matriz de Zaderenko rellena al cuadro de prelaciones, el grafo y los tiempos",
             _matriz_dada, grafo="pert"),
    Variante("prediccion", 2, "predecir sin calcular si un retraso cambia el plazo y comprobarlo con las holguras",
             _prediccion, grafo="pert"),
    Variante("plazo-impuesto", 2, "plazo menor que la duración: tardíos desde el plazo, holguras negativas, rutas que lo exceden y actividad común",
             _plazo_impuesto, grafo="pert"),
    Variante("gantt-dado", 3, "sólo se conoce el Gantt: deducir iniciales, predecesoras posibles y ajustadas, y una ruta crítica",
             _gantt_dado),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
