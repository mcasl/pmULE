"""Colección 02 — Grafo PERT: dibujar con el mínimo de ficticias, justificar
cada ficticia, numerar nodos, enumerar rutas y comprobar con los ancestros."""
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)
from .p03_roy import _AUX_FICTICIAS

def _pl(n, singular, plural=None):
    """«1 semana», «3 semanas»: número y sustantivo concordados (para enunciados y subtítulos)."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


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
                     "en los demás casos hacen falta ficticias. El visor muestra primero el grafo terminado junto al "
                     "cuadro; con los botones se construye desde el principio, fila a fila, y al final recorre las "
                     "columnas del cuadro para comprobar en el grafo las sucesoras de cada actividad."))
    celdas.append(code('#| fig-cap: "Construcción del grafo PERT a partir del cuadro de prelaciones expandido (las ficticias, discontinuas)"\n#| label: fig-pert\n'
                       "proyecto.display_pert_construccion()"))
    celdas.append(code("_arcos = sorted(proyecto.pert_graph.edges(data=True), key=lambda e: (int(e[0]), int(e[1])))\n"
                       "_mal = [d['activity'] for i, j, d in _arcos if int(i) >= int(j)]\n"
                       "Markdown(f'**Respuesta:** el grafo tiene {len(proyecto.nodes)} nodos, numerados del 1 al {len(proyecto.nodes)}, '\n"
                       "         f'{len(proyecto)} actividades reales y **{plural(len(proyecto.dummies), \"ficticia\")}** '\n"
                       "         f'({lista_y(proyecto.dummies)}; el nombre indica los nodos que unen). '\n"
                       "         + ('Todos los arcos van de un número menor a uno mayor: '\n"
                       "            + ', '.join(f\"{d['activity']} ({i} → {j})\" for i, j, d in _arcos) + '.' if not _mal\n"
                       "            else f'Arcos que no cumplen la regla: {lista_y(_mal)}.'))"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Una ficticia que va del nodo *i* al *j* transmite que todas las actividades que terminan en *i* "
                     "preceden a todas las que empiezan en *j*. Para ver por qué no puede evitarse hay que imaginar "
                     "que las actividades que salen de *j* salieran directamente de *i*: si eso añade alguna prelación "
                     "que el cuadro no tiene, o junta dos actividades entre el mismo par de nodos, la ficticia es "
                     "imprescindible."))
    celdas.append(code(_AUX_FICTICIAS))
    celdas.append(code("Markdown('\\n'.join(explica_ficticias(proyecto, prelaciones)))"))
    if paralelas:
        a, b = paralelas[0]
        celdas += apartado(ap); ap += 1
        celdas.append(md(f"{a} y {b} tienen las mismas predecesoras, así que pueden compartir el nodo inicial. "
                         f"Pero dos arcos no pueden unir el mismo par de nodos (el grafo no distinguiría una "
                         f"actividad de la otra): sólo podrían compartir también el nodo final si tuvieran las "
                         f"mismas sucesoras, y aun en ese caso una de las dos tendría que terminar en un nodo propio "
                         f"unido al común por una ficticia."))
        celdas.append(code(f"a, b = '{a}', '{b}'\n"
                           "sucesoras = {act: sorted(s for s, ps in prelaciones.immediate_predecessor.items() if act in ps)\n"
                           "             for act in datos_proyecto.index}\n"
                           "nodos = proyecto.nodes_from_edges\n"
                           "_pred = lista_y(sorted(prelaciones.immediate_predecessor[a])) or 'ninguna'\n"
                           "if sucesoras[a] == sucesoras[b]:\n"
                           "    _txt = (f'{a} y {b} tienen también las mismas sucesoras ({lista_y(sucesoras[a]) or \"ninguna: son finales\"}), '\n"
                           "            f'así que, si compartieran los dos nodos, serían dos arcos entre el mismo par de nodos. Por eso '\n"
                           "            f'salen del mismo nodo ({nodos[a][0]}) pero llegan a nodos distintos: {a} al {nodos[a][1]} y {b} al '\n"
                           "            f'{nodos[b][1]}, y una ficticia lleva el final de una de ellas al nodo de la otra.')\n"
                           "else:\n"
                           "    _txt = (f'Sus sucesoras son distintas ({a}: {lista_y(sucesoras[a]) or \"ninguna\"}; {b}: '\n"
                           "            f'{lista_y(sucesoras[b]) or \"ninguna\"}), así que no pueden compartir el nodo final: salen del '\n"
                           "            f'mismo nodo ({nodos[a][0]}) y llegan a nodos distintos ({a} al {nodos[a][1]}, {b} al {nodos[b][1]}).')\n"
                           "Markdown(f'**Respuesta:** {a} y {b} tienen las mismas predecesoras ({_pred}), luego pueden compartir el nodo '\n"
                           "         f'inicial. {_txt}')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Las rutas se enumeran recorriendo el grafo desde el nodo inicial: en cada nodo se abre una "
                     "rama por cada arco que sale de él; las ficticias se recorren pero no se anotan, porque no son "
                     "actividades."))
    celdas.append(code('#| tbl-cap: "Rutas del grafo (sin ficticias), en orden de ejecución"\n#| label: tbl-rutas\n'
                       "rutas = proyecto.paths(dummies=False)\n"
                       "pd.Series({ruta.replace('Route_', 'Ruta '): ' – '.join(acts) for ruta, acts in rutas.items()}).to_frame('actividades')"))
    celdas.append(code("Markdown(f'**Respuesta:** hay **{plural(len(rutas), \"ruta\")}** entre el nodo inicial y el final.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| fig-cap: "Grafo PERT y cuadro de ancestros"\n#| label: fig-ancestros\n'
                       "proyecto.display_ancestros('pert')"))
    celdas.append(md("Para cada actividad, el conjunto de actividades que aparecen antes que ella en alguna ruta "
                     "debe coincidir con su fila del cuadro de ancestros:"))
    celdas.append(code("ancestros = proyecto.distant_predecessor()\n"
                       "comprobacion = {}\n"
                       "for act in datos_proyecto.index:\n"
                       "    por_rutas = set()\n"
                       "    for acts in rutas.values():\n"
                       "        if act in acts:\n"
                       "            por_rutas |= set(acts[:acts.index(act)])\n"
                       "    comprobacion[act] = {'antes en alguna ruta': ', '.join(sorted(por_rutas)) or '—',\n"
                       "                         'ancestros (cuadro)': ', '.join(sorted(ancestros[act])) or '—',\n"
                       "                         'coincide': 'sí' if por_rutas == set(ancestros[act]) else 'no'}\n"
                       "comprobacion = pd.DataFrame(comprobacion).T\n"
                       "display(comprobacion)\n"
                       "Markdown('**Respuesta:** ' + ('para todas las actividades coinciden, así que el grafo respeta el cuadro de ancestros.'\n"
                       "         if (comprobacion['coincide'] == 'sí').all() else\n"
                       "         f\"no coinciden para {lista_y(comprobacion.index[comprobacion['coincide'] == 'no'])}.\"))"))
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
                       "                      'mismas predecesoras': 'sí' if predecesoras[a] == predecesoras[b] else 'no',\n"
                       "                      'sucesoras de ' + a: ', '.join(sorted(sucesoras[a])) or '—',\n"
                       "                      'sucesoras de ' + b: ', '.join(sorted(sucesoras[b])) or '—',\n"
                       "                      'mismas sucesoras': 'sí' if sucesoras[a] == sucesoras[b] else 'no'})\n"
                       "def concluye(a, b):\n"
                       "    ini, fin = predecesoras[a] == predecesoras[b], sucesoras[a] == sucesoras[b]\n"
                       "    t = (f'**{a} y {b}**: ' + ('tienen las mismas predecesoras (misma fila del cuadro expandido), luego '\n"
                       "         'pueden compartir nodo inicial; ' if ini else 'sus predecesoras son distintas (filas distintas), '\n"
                       "         'luego no pueden compartir nodo inicial; ')\n"
                       "         + ('tienen las mismas sucesoras (misma columna), luego podrían compartir nodo final' if fin else\n"
                       "            'sus sucesoras son distintas (columnas distintas), luego no pueden compartir nodo final'))\n"
                       "    if ini and fin:\n"
                       "        t += ('; pero no pueden compartir los dos a la vez (serían dos arcos entre el mismo par de nodos): '\n"
                       "              'una de las dos necesita una ficticia')\n"
                       "    return Markdown(t + '.')"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas.append(code(f"analiza('{pp[0]}', '{pp[1]}').to_frame('').T"))
    celdas.append(code(f"concluye('{pp[0]}', '{pp[1]}')"))
    celdas += apartado(2)
    celdas.append(code(f"analiza('{ps[0]}', '{ps[1]}').to_frame('').T"))
    celdas.append(code(f"concluye('{ps[0]}', '{ps[1]}')"))
    if pareja_nada:
        celdas.append(code(f"analiza('{pareja_nada[0]}', '{pareja_nada[1]}').to_frame('').T"))
        celdas.append(code(f"concluye('{pareja_nada[0]}', '{pareja_nada[1]}')"))
    celdas += apartado(3)
    celdas.append(code("acts = list(datos_proyecto.index)\n"
                       "parejas_inicio = [(a, b) for i, a in enumerate(acts) for b in acts[i+1:] if predecesoras[a] == predecesoras[b]]\n"
                       "parejas_fin = [(a, b) for i, a in enumerate(acts) for b in acts[i+1:] if sucesoras[a] == sucesoras[b]]\n"
                       "ambas = [p for p in parejas_inicio if p in parejas_fin]\n"
                       "_par = lambda ps: ', '.join(f'{a}–{b}' for a, b in ps) or 'ninguna'\n"
                       "Markdown(f'- Pueden compartir nodo inicial (mismas predecesoras): {_par(parejas_inicio)}.\\n'\n"
                       "         f'- Pueden compartir nodo final (mismas sucesoras): {_par(parejas_fin)}.\\n'\n"
                       "         f'- Cumplen las dos condiciones: {_par(ambas)}.')"))
    celdas.append(md("Las actividades iniciales comparten todas el nodo origen (no tienen predecesoras) y las finales "
                     "el nodo final (no tienen sucesoras), por eso aparecen en las listas aunque tengan `---`."))
    celdas.append(code("Markdown('**Respuesta:** ' + (f'las parejas {_par(ambas)} podrían compartir los dos nodos, pero entonces '\n"
                       "         'serían dos arcos entre el mismo par de nodos y el grafo no distinguiría una actividad de la otra: '\n"
                       "         'en cada una de esas parejas, una actividad termina en un nodo propio unido al común por una ficticia.'\n"
                       "         if ambas else 'ninguna pareja cumple las dos condiciones, así que no hay arcos paralelos que evitar.'))"))
    celdas += apartado(4)
    celdas.append(md("Construcción del grafo con el visor: primero el grafo terminado; con los botones, fila a fila desde "
                     "el principio y, al final, la comprobación de las sucesoras columna a columna."))
    celdas.append(code('#| fig-cap: "Construcción del grafo PERT a partir del cuadro de prelaciones"\n#| label: fig-pert\n'
                       "proyecto.display_pert_construccion()"))
    celdas.append(code(_AUX_FICTICIAS))
    celdas.append(code("_nodos = proyecto.nodes_from_edges\n"
                       "_ini = {}\n"
                       "for a in datos_proyecto.index:\n"
                       "    _ini.setdefault(_nodos[a][0], []).append(a)\n"
                       "_lineas = explica_ficticias(proyecto, prelaciones)\n"
                       "Markdown('En el grafo salen del mismo nodo: ' + '; '.join(f'{lista_y(v)} (nodo {k})' for k, v in sorted(_ini.items(), key=lambda kv: int(kv[0])) if len(v) > 1)\n"
                       "         + f'. Tiene {plural(len(_lineas), \"ficticia\")}' + (':\\n\\n' + '\\n'.join(_lineas) if _lineas else '.'))"))
    return celdas, f"{n} actividades, {_pl(len(estado['mismas_pred']), 'pareja')} con mismas predecesoras y {len(estado['mismas_suc'])} con mismas sucesoras"


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
    celdas.append(code("_orden = lambda fs: sorted(fs, key=lambda f: tuple(int(x) for x in f[1:].split('→')))\n"
                       "Markdown(f'**Respuesta:** el grafo sin simplificar tiene {len(proyecto_bruto.nodes)} nodos y **{plural(len(proyecto_bruto.dummies), \"ficticia\")}**: '\n"
                       "         f'{lista_y(_orden(proyecto_bruto.dummies))}.')"))
    celdas += apartado(2)
    celdas.append(md("La librería aplica exactamente esas reglas (`simplify`): recorre las ficticias y contrae "
                     "las que pueden contraerse sin alterar las prelaciones distantes ni la estructura."))
    celdas.append(code('#| fig-cap: "Grafo PERT simplificado"\n#| label: fig-pertSimplificado\n'
                       "proyecto = prelaciones.create_project()\n"
                       "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(code("Markdown(f'Tras simplificar quedan {len(proyecto.nodes)} nodos y {plural(len(proyecto.dummies), \"ficticia\")}: {lista_y(_orden(proyecto.dummies))}. '\n"
                       "         f'Se han eliminado {len(proyecto_bruto.dummies) - len(proyecto.dummies)}.')"))
    celdas.append(md("Las ficticias que quedan son las que transmiten una prelación que no puede expresarse "
                     "compartiendo nodos:"))
    celdas.append(code(_AUX_FICTICIAS))
    celdas.append(code("_lineas = explica_ficticias(proyecto, prelaciones)\n"
                       "Markdown(('\\n'.join(_lineas) + '\\n\\n' if _lineas else '') + f'**Respuesta:** de {len(proyecto_bruto.dummies)} ficticias '\n"
                       "         f'se pasa a {len(proyecto.dummies)}; las {len(proyecto_bruto.dummies) - len(proyecto.dummies)} eliminadas sólo unían el final de '\n"
                       "         f'una actividad con el inicio de otra sin aportar nada que no pudiera expresarse compartiendo nodos.')"))
    celdas += apartado(3)
    celdas.append(code("mismo_ancestros = proyecto.distant_predecessor() == proyecto_bruto.distant_predecessor()\n"
                       "rutas = {' - '.join(a) for a in proyecto.paths(dummies=False).values()}\n"
                       "rutas_bruto = {' - '.join(a) for a in proyecto_bruto.paths(dummies=False).values()}\n"
                       "Markdown(f'**Respuesta:** los dos grafos tienen {\"el mismo\" if mismo_ancestros else \"distinto\"} cuadro de ancestros y '\n"
                       "         f'{\"las mismas\" if rutas == rutas_bruto else \"distintas\"} rutas ({plural(len(rutas), \"ruta\")}: '\n"
                       "         f'{\"; \".join(sorted(rutas))}). Simplificar sólo cambia la forma del grafo, no las prelaciones que representa.')"))
    return celdas, f"{n} actividades, {estado['n0']} ficticias sin simplificar frente a {info['n_dummies']}"


VARIANTES = [
    Variante("minimo-ficticias", 1, "dibujar con el mínimo de ficticias, justificar cada una, rutas y ancestros",
             lambda rng: _base(1, rng), grafo="pert"),
    Variante("paralelas", 2, "ídem con dos actividades de idénticas predecesoras (arcos paralelos)",
             lambda rng: _base(2, rng), grafo="pert"),
    Variante("mas-ficticias", 2, "ídem con un proyecto mayor y más ficticias", lambda rng: _base(3, rng), grafo="pert"),
    Variante("compartir-nodos", 2, "qué parejas pueden compartir nodo inicial, nodo final, ambos o ninguno",
             _compartir_nodos, grafo="pert"),
    Variante("ficticias-sobrantes", 3, "grafo sistemático sin simplificar frente al mínimo: qué ficticias sobran y por qué",
             _ficticias_sobrantes, grafo="pert"),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
