"""Colección 02 — Grafo PERT: dibujar con el mínimo de ficticias, justificar
cada ficticia, numerar nodos, enumerar rutas y comprobar con los ancestros."""
from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

META = dict(slug="02_pert", prefijo="PERT",
            titulo="Grafo PERT y actividades ficticias",
            descripcion="Dibujar el grafo PERT con el mínimo número de ficticias, justificar cada una, "
                        "numerar los nodos y enumerar las rutas.",
            n_defecto=3)


def _paralelas(preds):
    """Pares de actividades con el mismo conjunto de predecesoras (y alguna predecesora)."""
    vistos = {}
    pares = []
    for a, ps in preds.items():
        clave = frozenset(ps)
        if ps and clave in vistos:
            pares.append((vistos[clave], a))
        vistos.setdefault(clave, a)
    return pares


def generar(k, rng):
    objetivo = {1: (2, 2), 2: (3, 3), 3: (3, 5)}[(k - 1) % 3 + 1]
    n = {1: 8, 2: 9, 3: 10}[(k - 1) % 3 + 1]

    def acepta(i):
        ok = objetivo[0] <= i["n_dummies"] <= objetivo[1] and 5 <= i["n_rutas"] <= 10
        if k % 3 == 2:
            ok = ok and len(_paralelas(i["preds"])) >= 1
        return ok

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])),
                    con_duraciones=False)
    df = P.tabla_base(info)
    paralelas = _paralelas(info["preds"])

    celdas = [md("# Enunciado"),
              md("Considere el proyecto definido por el siguiente cuadro de prelaciones inmediatas.")]
    pregs = ["Dibuje el grafo PERT del proyecto utilizando el mínimo número de actividades ficticias y numere "
             "sus nodos de modo que todo arco vaya de un número menor a uno mayor.",
             "Para cada actividad ficticia, explique qué prelación transmite y por qué no puede evitarse.",
             "Enumere todas las rutas desde el nodo inicial al final e indique cuántas hay.",
             "Construya el cuadro de ancestros y compruebe que el grafo lo respeta."]
    if paralelas:
        a, b = paralelas[0]
        pregs.insert(2, f"Las actividades {a} y {b} tienen las mismas predecesoras. ¿Pueden compartir nodo inicial y "
                        f"nodo final? Razone la respuesta.")
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas.append(md("Partimos del cuadro de prelaciones expandido: las filas dan las predecesoras inmediatas "
                     "y las columnas las sucesoras. Dos actividades pueden compartir nodo final si tienen "
                     "exactamente las mismas sucesoras, y nodo inicial si tienen las mismas predecesoras; "
                     "en los demás casos hacen falta ficticias."))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas.append(code('#| fig-cap: "Grafo PERT con numeración de nodos; las ficticias se dibujan discontinuas"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(code("print(f'El grafo tiene {len(proyecto.nodes)} nodos, {len(proyecto)} actividades reales '\n"
                       "      f'y {len(proyecto.dummies)} ficticias: {\", \".join(proyecto.dummies)}')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Cada ficticia une el nodo final de unas actividades con el nodo inicial de otras. "
                     "La prelación que transmite es «todas las que entran en su nodo origen preceden a "
                     "todas las que salen de su nodo destino»."))
    celdas.append(code("grafo = proyecto.pert_graph\n"
                       "def actividades_que_entran(nodo):\n"
                       "    return sorted(d['activity'] for _, _, d in grafo.in_edges(nodo, data=True) if not d['activity'].startswith('@'))\n"
                       "def actividades_que_salen(nodo):\n"
                       "    return sorted(d['activity'] for _, _, d in grafo.out_edges(nodo, data=True) if not d['activity'].startswith('@'))\n"
                       "\n"
                       "filas = []\n"
                       "for origen, destino, d in grafo.edges(data=True):\n"
                       "    if d['activity'].startswith('@'):\n"
                       "        filas.append({'ficticia': d['activity'],\n"
                       "                      'entran en el nodo origen': ', '.join(actividades_que_entran(origen)),\n"
                       "                      'salen del nodo destino': ', '.join(actividades_que_salen(destino))})\n"
                       "pd.DataFrame(filas).set_index('ficticia')"))
    celdas.append(md("Si en lugar de la ficticia se fusionaran ambos nodos, las actividades que salen del nodo "
                     "destino pasarían a tener como predecesoras a todas las que entran en el origen *y* a las que "
                     "entran en el destino, que no es lo que dice el cuadro; y si se eliminara sin fusionar, se "
                     "perdería la prelación. Por eso ninguna puede evitarse."))
    if paralelas:
        a, b = paralelas[0]
        celdas += apartado(ap); ap += 1
        celdas.append(md(f"{a} y {b} tienen las mismas predecesoras, así que pueden compartir el nodo inicial. "
                         f"Pero dos arcos no pueden unir el mismo par de nodos (el grafo no distinguiría una "
                         f"actividad de la otra), de modo que sólo compartirán también el nodo final si además "
                         f"tienen las mismas sucesoras, y aun en ese caso una de las dos necesita una ficticia."))
        celdas.append(code("sucesoras = {act: sorted(s for s, ps in prelaciones.immediate_predecessor.items() if act in ps)\n"
                           "             for act in datos_proyecto.index}\n"
                           f"print('Sucesoras de {a}:', sucesoras['{a}'])\n"
                           f"print('Sucesoras de {b}:', sucesoras['{b}'])"))
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| tbl-cap: "Rutas del grafo (sin ficticias)"\n#| label: tbl-rutas\n'
                       "rutas = proyecto.paths(dummies=False)\n"
                       "pd.Series({ruta: ' - '.join(acts) for ruta, acts in rutas.items()}).to_frame('actividades')"))
    celdas.append(code("print(f'El grafo tiene {len(rutas)} rutas entre el nodo inicial y el final.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros"\n#| label: tbl-ancestros\n'
                       "proyecto.distant_predecessor(format='DataFrame')"))
    celdas.append(md("Para cada actividad, el conjunto de actividades que aparecen antes que ella en alguna ruta "
                     "debe coincidir con su fila del cuadro de ancestros:"))
    celdas.append(code("ancestros = proyecto.distant_predecessor()\n"
                       "comprobacion = {}\n"
                       "for act in datos_proyecto.index:\n"
                       "    por_rutas = set()\n"
                       "    for acts in rutas.values():\n"
                       "        if act in acts:\n"
                       "            por_rutas |= set(acts[:acts.index(act)])\n"
                       "    comprobacion[act] = por_rutas == set(ancestros[act])\n"
                       "pd.Series(comprobacion).to_frame('coincide')"))
    sub = f"{n} actividades, {info['n_dummies']} ficticias, {info['n_rutas']} rutas"
    return celdas, sub
