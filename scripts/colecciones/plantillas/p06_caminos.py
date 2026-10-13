"""Colección 06 — Matriz de caminos: enumerar rutas, construir la matriz,
calcular la duración como máximo de P·d, sensibilidad, reconstruir el grafo
a partir de las rutas y rutas dominantes vía SVD."""
from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe, BMATRIX

META = dict(slug="06_matriz_caminos", prefijo="Caminos",
            titulo="Matriz de caminos",
            descripcion="Enumerar las rutas, construir la matriz de caminos y usarla para calcular la "
                        "duración del proyecto; reconstruir el grafo a partir de las rutas; rutas dominantes "
                        "mediante la SVD.",
            n_defecto=3)


def generar(k, rng):
    inverso = (k % 3 == 0)
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 8 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    rutas = info["rutas"]

    celdas = [md("# Enunciado")]
    if inverso:
        celdas.append(md("De un proyecto sólo se conocen sus rutas entre el nodo inicial y el final y las "
                         "duraciones de sus actividades (en días)."))
        pregs = ["Construya la matriz de caminos del proyecto.",
                 "Calcule con ella la duración del proyecto e identifique la ruta crítica.",
                 "Reconstruya el cuadro de prelaciones inmediatas y dibuje los grafos PERT y Roy. Compruebe "
                 "que sus rutas son las del enunciado.",
                 "Calcule la sensibilidad de la duración del proyecto a cada actividad: ¿en cuánto aumenta "
                 "la duración si una actividad dura un día más?",
                 "Calcule la descomposición en valores singulares de la matriz de caminos y ordene rutas y "
                 "actividades según el primer vector singular. Compare con la ruta crítica."]
    else:
        celdas.append(md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta."))
        pregs = ["Dibuje el grafo PERT y enumere todas sus rutas entre el nodo inicial y el final.",
                 "Construya la matriz de caminos del proyecto (una fila por ruta, una columna por actividad).",
                 "Calcule la duración de cada ruta como producto de la matriz de caminos por el vector de "
                 "duraciones, y la duración del proyecto como el máximo. Identifique la ruta crítica.",
                 "Calcule la sensibilidad de la duración del proyecto a cada actividad: ¿en cuánto aumenta "
                 "la duración si una actividad dura un día más? ¿Y si dura un día menos?"]
    celdas.append(preguntas(pregs))
    if inverso:
        celdas.append(code('#| tbl-cap: "Rutas del proyecto"\n#| label: tbl-datosRutas\n'
                           + "rutas_enunciado = {\n"
                           + "".join(f"    {r!r}: {acts!r},\n" for r, acts in rutas.items())
                           + "}\n"
                           "pd.Series({r: ' - '.join(a) for r, a in rutas_enunciado.items()}).to_frame('actividades')"))
        celdas.append(code('#| tbl-cap: "Duración de las actividades"\n#| label: tbl-datosDuraciones\n'
                           + literal_dataframe(df[["duration"]])))
    else:
        celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                           + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    ap = 1
    if inverso:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La matriz de caminos tiene un 1 en la fila de una ruta y la columna de una actividad "
                         "cuando la actividad pertenece a la ruta."))
        celdas.append(code("actividades = sorted(set(a for acts in rutas_enunciado.values() for a in acts))\n"
                           "path_matrix = pd.DataFrame(0, index=list(rutas_enunciado), columns=actividades)\n"
                           "for ruta, acts in rutas_enunciado.items():\n"
                           "    path_matrix.loc[ruta, acts] = 1\n"
                           "path_matrix"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("duraciones = datos_enunciado['duration'].reindex(actividades)\n"
                           "duraciones_rutas = path_matrix @ duraciones\n"
                           "Math(bmatrix(path_matrix) + r' \\times ' + bmatrix(duraciones) + ' = ' + bmatrix(duraciones_rutas))"))
        celdas.append(code("duracion_proyecto = int(duraciones_rutas.max())\n"
                           "ruta_critica = duraciones_rutas.idxmax()\n"
                           "Markdown(f'La duración del proyecto es {duracion_proyecto} días, la de la ruta más larga, {ruta_critica}: '\n"
                           "         f'{\" - \".join(rutas_enunciado[ruta_critica])}.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(md("Las predecesoras inmediatas de una actividad son las que la preceden "
                         "inmediatamente en alguna ruta; las rutas son todas las secuencias posibles, así que "
                         "el conjunto de parejas consecutivas de todas las rutas es exactamente el cuadro de "
                         "prelaciones inmediatas. Las actividades que abren alguna ruta son iniciales."))
        celdas.append(code("prelaciones_dict = {a: set() for a in actividades}\n"
                           "for acts in rutas_enunciado.values():\n"
                           "    for anterior, posterior in zip(acts[:-1], acts[1:]):\n"
                           "        prelaciones_dict[posterior].add(anterior)\n"
                           "datos_proyecto = pd.DataFrame({'predecessor': pd.Series({a: ','.join(sorted(p)) or '---' for a, p in prelaciones_dict.items()}),\n"
                           "                               'duration': duraciones})\n"
                           "datos_proyecto.index.name = 'activity'\n"
                           "datos_proyecto"))
        celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                           "proyecto = prelaciones.create_project()"))
        celdas.append(code('#| fig-cap: "Grafo PERT reconstruido, con tiempos y ruta crítica"\n#| label: fig-pert\n'
                           "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
        celdas.append(code('#| fig-cap: "Grafo Roy reconstruido"\n#| label: fig-roy\n'
                           "proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)"))
        celdas.append(code("rutas_grafo = {' - '.join(a) for a in proyecto.paths(dummies=False).values()}\n"
                           "rutas_dadas = {' - '.join(a) for a in rutas_enunciado.values()}\n"
                           "print('Las rutas del grafo coinciden con las del enunciado:', rutas_grafo == rutas_dadas)"))
    else:
        celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                           "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                           "proyecto = prelaciones.create_project()"))
        celdas += apartado(ap); ap += 1
        celdas.append(code('#| fig-cap: "Grafo PERT con numeración de nodos"\n#| label: fig-pert\n'
                           "proyecto.pert(durations=False, ranksep=1)"))
        celdas.append(md("Las rutas se enumeran recorriendo el grafo desde el nodo inicial: en cada nodo se "
                         "abre una rama por cada arco saliente. Las ficticias se recorren pero no se anotan."))
        celdas.append(code('#| tbl-cap: "Rutas del proyecto"\n#| label: tbl-rutas\n'
                           "rutas = proyecto.paths(dummies=False)\n"
                           "pd.Series({r: ' - '.join(a) for r, a in rutas.items()}).to_frame('actividades')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code('#| tbl-cap: "Matriz de caminos"\n#| label: tbl-matrizCaminos\n'
                           "path_matrix = proyecto.path_matrix(dummies=False)\n"
                           "proyecto.display_path_matrix(dummies=False)"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("duraciones = datos_proyecto['duration'][path_matrix.columns]\n"
                           "duraciones_rutas = path_matrix @ duraciones\n"
                           "Math(bmatrix(path_matrix) + r' \\times ' + bmatrix(duraciones) + ' = ' + bmatrix(duraciones_rutas))"))
        celdas.append(code("duracion_proyecto = int(duraciones_rutas.max())\n"
                           "ruta_critica = duraciones_rutas.idxmax()\n"
                           "Markdown(f'La duración del proyecto es {duracion_proyecto} días, la de la ruta más larga, {ruta_critica}: '\n"
                           "         f'{\" - \".join(rutas[ruta_critica])}. Coincide con el tiempo más temprano del último nodo: {proyecto.duration(duraciones)}.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("La duración del proyecto es $\\tau = \\max_r (P\\,d)_r$. Si una actividad crítica dura un "
                     "día más, la ruta crítica se alarga un día y con ella el proyecto: su sensibilidad es 1. "
                     "Si una actividad no crítica dura un día más, la ruta más larga no cambia mientras no se "
                     "agote la holgura: sensibilidad 0. Al acortar una actividad crítica el proyecto sólo se "
                     "acorta hasta que otra ruta pasa a ser la más larga."))
    celdas.append(code("sensibilidad = proyecto.tau_sensitivity(duraciones.to_dict())\n"
                       "holgura = duracion_proyecto - duraciones_rutas\n"
                       "segunda = holgura[holgura > 0].min() if (holgura > 0).any() else None\n"
                       "pd.DataFrame({'sensibilidad (+1 día)': sensibilidad.reindex(duraciones.index)}).T"))
    celdas.append(code("comprobacion = {}\n"
                       "for act in duraciones.index:\n"
                       "    mas = duraciones.copy(); mas[act] += 1\n"
                       "    menos = duraciones.copy(); menos[act] -= 1\n"
                       "    comprobacion[act] = {'+1 día': int((path_matrix @ mas).max()) - duracion_proyecto,\n"
                       "                         '-1 día': int((path_matrix @ menos).max()) - duracion_proyecto}\n"
                       "pd.DataFrame(comprobacion)"))
    celdas.append(code("Markdown(f'Al acortar una actividad crítica el proyecto se reduce un día sólo si ninguna otra ruta alcanza la duración máxima; '\n"
                       "         f'la segunda ruta más larga está a {segunda} día(s) de la crítica, así que esa es la reducción máxima que puede conseguirse '\n"
                       "         'acortando una única actividad crítica que no pertenezca también a esa segunda ruta.')"))
    if inverso:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La descomposición en valores singulares $P = U\\,S\\,V^T$ ordena la información de la "
                         "matriz de caminos por importancia. El primer vector singular por la izquierda, "
                         "$u_1$, pondera las rutas, y el primero por la derecha, $v_1$, las actividades: las "
                         "actividades que aparecen en muchas rutas y las rutas que contienen muchas actividades "
                         "compartidas reciben más peso. Es una ordenación estructural, que no usa las duraciones, "
                         "y por eso no tiene por qué coincidir con la ruta crítica."))
        celdas.append(code("U, S, Vt = np.linalg.svd(path_matrix.values.astype(float))\n"
                           "u1 = np.abs(U[:, 0]); v1 = np.abs(Vt[0, :])\n"
                           "importancia_rutas = pd.Series(u1 / u1.sum(), index=path_matrix.index)\n"
                           "importancia_actividades = pd.Series(v1 / v1.sum(), index=path_matrix.columns)\n"
                           "Math(r'S = ' + bmatrix(S.round(2), dec=2))"))
        celdas.append(code('#| tbl-cap: "Rutas y actividades ordenadas por su peso en el primer vector singular"\n#| label: tbl-svd\n'
                           "ordena_rutas(path_matrix.copy(), importancia_actividades.values, importancia_rutas.values)"))
        celdas.append(code("Markdown(f'La ruta dominante según la SVD es {importancia_rutas.idxmax()} y la ruta crítica es {ruta_critica}. '\n"
                           "         f'La actividad de mayor peso es {importancia_actividades.idxmax()}, que aparece en '\n"
                           "         f'{int(path_matrix[importancia_actividades.idxmax()].sum())} de las {len(path_matrix)} rutas.')"))
    sub = f"{n} actividades, {info['n_rutas']} rutas" + (", reconstrucción desde las rutas y SVD" if inverso else "")
    return celdas, sub
