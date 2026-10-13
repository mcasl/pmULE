"""Colección 02 — Grafo PERT: dibujar con el mínimo de ficticias, justificar
cada ficticia, numerar nodos, enumerar rutas y comprobar con los ancestros."""
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado,
                        literal_dataframe, literal_dict, BMATRIX)

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


def _base(k, rng):
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


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _compartir_nodos(rng):
    """Qué parejas de actividades pueden compartir nodo inicial, nodo final, ambos o ninguno."""
    n = int(rng.choice([9, 10]))
    estado = {}

    def acepta(i):
        preds = i["preds"]
        suc = P.sucesoras(preds)
        acts = list(preds)
        mismas_pred = [(a, b) for x, a in enumerate(acts) for b in acts[x + 1:] if preds[a] and preds[a] == preds[b]]
        mismas_suc = [(a, b) for x, a in enumerate(acts) for b in acts[x + 1:] if suc[a] and suc[a] == suc[b]]
        ambas = [p for p in mismas_pred if p in mismas_suc]
        if not mismas_pred or not mismas_suc:
            return False
        if not (2 <= i["n_dummies"] <= 5):
            return False
        estado.update(mismas_pred=mismas_pred, mismas_suc=mismas_suc, ambas=ambas)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])), con_duraciones=False)
    df = P.tabla_base(info)
    preds, suc = info["preds"], P.sucesoras(info["preds"])
    acts = list(preds)
    # tres parejas para preguntar: una con mismas predecesoras, una con mismas sucesoras, una sin nada en común
    pares_nada = [(a, b) for x, a in enumerate(acts) for b in acts[x + 1:]
                  if preds[a] != preds[b] and suc[a] != suc[b] and preds[a] and preds[b]]
    pareja_nada = pares_nada[int(rng.integers(0, len(pares_nada)))] if pares_nada else None
    pp, ps = estado["mismas_pred"][0], estado["mismas_suc"][0]
    pregs = [f"¿Pueden las actividades {pp[0]} y {pp[1]} compartir nodo inicial? ¿Y nodo final? Razónelo con el "
             "cuadro de prelaciones expandido.",
             f"Misma pregunta para {ps[0]} y {ps[1]}" + (f", y para {pareja_nada[0]} y {pareja_nada[1]}." if pareja_nada else "."),
             "Enumere todas las parejas de actividades que pueden compartir nodo inicial y todas las que pueden "
             "compartir nodo final. Si una pareja puede compartir ambos, ¿qué ocurre?",
             "Dibuje el grafo PERT y compruebe sobre él las respuestas anteriores."]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto definido por el siguiente cuadro de prelaciones inmediatas. En un grafo "
                 "PERT dos actividades pueden salir del mismo nodo sólo si tienen exactamente las mismas "
                 "predecesoras, y llegar al mismo nodo sólo si tienen exactamente las mismas sucesoras.")]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "predecesoras = prelaciones.immediate_predecessor\n"
                       "sucesoras = {a: {s for s, ps in predecesoras.items() if a in ps} for a in datos_proyecto.index}\n"
                       "def analiza(a, b):\n"
                       "    return pd.Series({'predecesoras de ' + a: ', '.join(sorted(predecesoras[a])) or '—',\n"
                       "                      'predecesoras de ' + b: ', '.join(sorted(predecesoras[b])) or '—',\n"
                       "                      'comparten nodo inicial': predecesoras[a] == predecesoras[b],\n"
                       "                      'sucesoras de ' + a: ', '.join(sorted(sucesoras[a])) or '—',\n"
                       "                      'sucesoras de ' + b: ', '.join(sorted(sucesoras[b])) or '—',\n"
                       "                      'comparten nodo final': sucesoras[a] == sucesoras[b]})"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas.append(code(f"analiza('{pp[0]}', '{pp[1]}').to_frame('').T"))
    celdas.append(md(f"{pp[0]} y {pp[1]} tienen las mismas predecesoras (misma fila en el cuadro expandido), así "
                     f"que salen del mismo nodo. Compartirían también el nodo final sólo si tuvieran las mismas "
                     f"sucesoras (misma columna)."))
    celdas += apartado(2)
    celdas.append(code(f"analiza('{ps[0]}', '{ps[1]}').to_frame('').T"))
    if pareja_nada:
        celdas.append(code(f"analiza('{pareja_nada[0]}', '{pareja_nada[1]}').to_frame('').T"))
    celdas += apartado(3)
    celdas.append(code("acts = list(datos_proyecto.index)\n"
                       "parejas_inicio = [(a, b) for i, a in enumerate(acts) for b in acts[i+1:] if predecesoras[a] == predecesoras[b]]\n"
                       "parejas_fin = [(a, b) for i, a in enumerate(acts) for b in acts[i+1:] if sucesoras[a] == sucesoras[b]]\n"
                       "ambas = [p for p in parejas_inicio if p in parejas_fin]\n"
                       "print('Comparten nodo inicial:', parejas_inicio)\n"
                       "print('Comparten nodo final:  ', parejas_fin)\n"
                       "print('Comparten ambos:       ', ambas or 'ninguna')"))
    celdas.append(md("Si una pareja pudiera compartir los dos nodos tendríamos dos arcos entre el mismo par de "
                     "nodos, y el grafo no podría distinguir una actividad de la otra: una de las dos debe "
                     "terminar en un nodo auxiliar unido al común por una ficticia. Lo mismo ocurre con las "
                     "actividades iniciales (todas comparten el nodo origen) y las finales (todas comparten el "
                     "sumidero): si dos actividades son a la vez iniciales y finales hace falta una ficticia."))
    celdas += apartado(4)
    celdas.append(code('#| fig-cap: "Grafo PERT del proyecto"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
    return celdas, f"{n} actividades, {len(estado['mismas_pred'])} pareja(s) con mismas predecesoras y {len(estado['mismas_suc'])} con mismas sucesoras"


def _ficticias_sobrantes(rng):
    """Grafo construido sin simplificar (una ficticia por prelación que lo necesite) frente al mínimo."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            g0 = i["tabla"].create_project(simplify=False)
        extra = len(g0.dummies) - i["n_dummies"]
        if not (1 <= i["n_dummies"] <= 4 and extra >= 2 and len(g0.dummies) <= 9):
            return False
        estado["n0"] = len(g0.dummies)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])), con_duraciones=False)
    df = P.tabla_base(info)
    celdas = [md("# Enunciado"),
              md("Un método sistemático para dibujar el grafo PERT consiste en dar a cada actividad su propio "
                 "nodo de inicio y de fin y unir con una ficticia cada prelación. El resultado es correcto, pero "
                 "tiene muchas más ficticias de las necesarias. Considere el cuadro de prelaciones adjunto.")]
    celdas.append(preguntas([
        "Dibuje el grafo PERT por el método sistemático (sin fusionar nodos). ¿Cuántas ficticias tiene?",
        "Simplifíquelo: una ficticia puede eliminarse fusionando sus dos nodos cuando al hacerlo no se crea "
        "ninguna prelación nueva ni desaparece ninguna, no se juntan dos arcos entre los mismos nodos y el grafo "
        "sigue teniendo un único origen y un único final. ¿Cuántas ficticias quedan?",
        "Compruebe que ambos grafos tienen el mismo cuadro de ancestros y las mismas rutas (sin ficticias)."]))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo PERT sin simplificar"\n#| label: fig-pertSinSimplificar\n'
                       "proyecto_bruto = prelaciones.create_project(simplify=False)\n"
                       "proyecto_bruto.pert(durations=False, ranksep=1)"))
    celdas.append(code("Markdown(f'El grafo sin simplificar tiene {len(proyecto_bruto.nodes)} nodos y {len(proyecto_bruto.dummies)} ficticias: {\", \".join(proyecto_bruto.dummies)}.')"))
    celdas += apartado(2)
    celdas.append(md("La librería aplica exactamente esas reglas (`simplify`): recorre las ficticias y contrae "
                     "las que pueden contraerse sin alterar las prelaciones distantes ni la estructura."))
    celdas.append(code('#| fig-cap: "Grafo PERT simplificado"\n#| label: fig-pertSimplificado\n'
                       "proyecto = prelaciones.create_project()\n"
                       "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(code("Markdown(f'Tras simplificar quedan {len(proyecto.nodes)} nodos y {len(proyecto.dummies)} ficticias: {\", \".join(proyecto.dummies) or \"ninguna\"}. '\n"
                       "         f'Se han eliminado {len(proyecto_bruto.dummies) - len(proyecto.dummies)}.')"))
    celdas.append(md("Las ficticias que quedan son las que transmiten una prelación que no puede expresarse "
                     "compartiendo nodos: cada una une el final de un grupo de actividades con el inicio de otro "
                     "grupo que depende de ellas *y de otras más*."))
    celdas += apartado(3)
    celdas.append(code("mismo_ancestros = proyecto.distant_predecessor() == proyecto_bruto.distant_predecessor()\n"
                       "rutas = {' - '.join(a) for a in proyecto.paths(dummies=False).values()}\n"
                       "rutas_bruto = {' - '.join(a) for a in proyecto_bruto.paths(dummies=False).values()}\n"
                       "print('Mismo cuadro de ancestros:', mismo_ancestros)\n"
                       "print('Mismas rutas:', rutas == rutas_bruto, f'({len(rutas)} rutas)')"))
    return celdas, f"{n} actividades, {estado['n0']} ficticias sin simplificar frente a {info['n_dummies']}"


VARIANTES = [
    Variante("minimo-ficticias", 1, "dibujar con el mínimo de ficticias, justificar cada una, rutas y ancestros",
             lambda rng: _base(1, rng)),
    Variante("paralelas", 2, "ídem con dos actividades de idénticas predecesoras (arcos paralelos)",
             lambda rng: _base(2, rng)),
    Variante("mas-ficticias", 2, "ídem con un proyecto mayor y más ficticias", lambda rng: _base(3, rng)),
    Variante("compartir-nodos", 2, "qué parejas pueden compartir nodo inicial, nodo final, ambos o ninguno",
             _compartir_nodos),
    Variante("ficticias-sobrantes", 3, "grafo sistemático sin simplificar frente al mínimo: qué ficticias sobran y por qué",
             _ficticias_sobrantes),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
