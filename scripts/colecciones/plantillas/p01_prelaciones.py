"""Colección 01 — Cuadro de prelaciones: depurar redundancias y erratas,
cuadro de prelaciones expandido, cuadro de ancestros, iniciales y finales."""
import networkx as nx
import numpy as np

import pandas as pd
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX, grafo_nombre, ESTADO)
from .p03_roy import _AUX_FICTICIAS

# Iniciales y finales, justificadas con el cuadro expandido (necesita `prelaciones`)
_INI_FIN = (
    "_ini, _fin = sorted(prelaciones.start_activities), sorted(prelaciones.end_activities)\n"
    "_es = lambda l, s, p: s if len(l) == 1 else p\n"
    "Markdown(f'**Respuesta:** {_es(_ini, \"es inicial\", \"son iniciales\")} {lista_y(_ini)}, porque no '\n"
    "         f'{_es(_ini, \"tiene\", \"tienen\")} predecesoras (fila vacía en el cuadro expandido); '\n"
    "         f'{_es(_fin, \"es final\", \"son finales\")} {lista_y(_fin)}, porque no '\n"
    "         f'{_es(_fin, \"es predecesora\", \"son predecesoras\")} de ninguna actividad (columna vacía).')")


def _celdas_grafo(cap_pert="Grafo PERT del proyecto"):
    """Visor de ancestros y, en el PERT, una línea por ficticia con la prelación que transmite y por qué hace falta."""
    celdas = [md(f"Construcción del grafo {grafo_nombre()} a partir del cuadro de prelaciones: el visor muestra primero el "
                 "grafo terminado; con los botones se construye desde el principio, fila a fila, y al final recorre las "
                 "columnas del cuadro comprobando en el grafo las sucesoras de cada actividad."),
              code(f'#| fig-cap: "Construcción del grafo {grafo_nombre()} a partir del cuadro de prelaciones"\n'
                   f'#| label: fig-construccion\n'
                   f"proyecto.display_{ESTADO['grafo']}_construccion()"),
              code(f'#| fig-cap: "Grafo {grafo_nombre()} del proyecto y cuadro de ancestros"\n'
                   f'#| label: fig-{ESTADO["grafo"]}\n'
                   f"proyecto.display_ancestros('{ESTADO['grafo']}')")]
    if ESTADO["grafo"] == "roy":
        celdas.append(code("_n_prel = sum(len(ps) for ps in prelaciones.immediate_predecessor.values())\n"
                           "Markdown(f'El grafo Roy tiene un nodo por actividad ({len(prelaciones.immediate_predecessor)}) más los nodos de inicio y '\n"
                           "         f'fin, y un arco por cada una de las {_n_prel} prelaciones inmediatas (más los que salen del inicio y '\n"
                           "         f'llegan al fin); no necesita ficticias.')"))
    else:
        celdas.append(code(_AUX_FICTICIAS))
        celdas.append(code("_lineas = explica_ficticias(proyecto, prelaciones)\n"
                           "Markdown(f'El grafo PERT tiene {len(proyecto.nodes)} nodos y {plural(len(_lineas), \"ficticia\")}' + "
                           "(':\\n\\n' + '\\n'.join(_lineas) if _lineas else '.'))"))
    return celdas

META = dict(slug="01_prelaciones", prefijo="Prelaciones",
            titulo="Cuadro de prelaciones",
            descripcion="Depurar un cuadro de prelaciones (redundancias, erratas que crean ciclos), "
                        "construir el cuadro expandido y el de ancestros.",
            n_defecto=3)


def _base(k, rng):
    con_ciclo = (k % 3 == 0)
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 2 <= i["n_dummies"] <= 4 and len(P.redundantes_posibles(i["preds"])) >= 3,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=2, con_duraciones=False)
    preds = {a: set(ps) for a, ps in info["preds"].items()}
    # Redundancias: 2 prelaciones implícitas añadidas al cuadro
    pares = P.redundantes_posibles(preds)
    idx = rng.choice(len(pares), size=2, replace=False)
    redundantes = [pares[i] for i in idx]
    dado = {a: set(ps) for a, ps in preds.items()}
    for p, a in redundantes:
        dado[a].add(p)
    # Errata que cierra un ciclo: una actividad tardía figura como predecesora de una temprana
    errata = None
    if con_ciclo:
        G = P.grafo(preds)
        candidatos = [(d, e) for e in preds for d in sorted(nx.descendants(G, e)) if e not in preds[d]]
        d, e = candidatos[int(rng.integers(0, len(candidatos)))]
        errata = (d, e)           # "d precede a e" es la errata
        dado[e].add(d)
    # Orden de presentación barajado para que el alumno no se apoye en el alfabeto
    orden = list(dado)
    rng.shuffle(orden)
    dado_ordenado = {a: dado[a] for a in orden}
    df = P.tabla_base(dict(preds=dado_ordenado))

    celdas = [md("# Enunciado"),
              md("Un compañero ha redactado el cuadro de prelaciones inmediatas del proyecto que se "
                 "muestra a continuación. Antes de dibujar el grafo conviene revisarlo: sospechamos que "
                 "contiene prelaciones redundantes (ya implicadas por otras)"
                 + (" y al menos una errata que hace imposible ejecutar el proyecto" if con_ciclo else "")
                 + ".")]
    pregs = []
    if con_ciclo:
        pregs.append("Compruebe si el cuadro es coherente. Si existe algún ciclo, identifíquelo e indique qué "
                     "prelación debe eliminarse, suponiendo que la errata consiste en una actividad tardía "
                     "anotada por error como predecesora de una temprana.")
    pregs += ["Identifique las prelaciones redundantes y escriba el cuadro de prelaciones inmediatas mínimo.",
              "Construya el cuadro de prelaciones expandido (matriz de prelaciones inmediatas) y, a partir de "
              "él, indique las actividades iniciales y finales del proyecto.",
              "Construya el cuadro de ancestros (prelaciones transitivas) y compruebe que las prelaciones "
              "redundantes del apartado anterior aparecen en él.",
              f"Dibuje el grafo {grafo_nombre()} del proyecto y compruebe sobre él el cuadro de ancestros."]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones tal y como lo recibimos"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    ap = 1
    celdas.append(code("prelaciones_dadas = {act: set(p.strip() for p in fila.split(',') if p.strip() and p.strip() != '---')\n"
                       "                     for act, fila in datos_enunciado['predecessor'].items()}\n"
                       "G = nx.DiGraph()\n"
                       "G.add_nodes_from(prelaciones_dadas)\n"
                       "G.add_edges_from((p, a) for a, ps in prelaciones_dadas.items() for p in ps)"))
    if con_ciclo:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La librería detecta los ciclos al construir la tabla de prelaciones. Un ciclo "
                         "significa que alguna actividad debería terminar antes de empezar ella misma, lo que "
                         "hace el proyecto inejecutable."))
        celdas.append(code("try:\n"
                           "    PredecessorTable.from_dict_of_strings(datos_enunciado['predecessor'].to_dict())\n"
                           "    print('El cuadro no contiene ciclos.')\n"
                           "except ValueError as error:\n"
                           "    print(error)"))
        celdas.append(code("ciclos = list(nx.simple_cycles(G))\n"
                           "Markdown('Ciclos del cuadro: ' + '; '.join(' → '.join(c + [c[0]]) for c in ciclos) + '.')"))
        cadena = " → ".join(nx.shortest_path(P.grafo(preds), errata[1], errata[0]))
        celdas.append(md(f"En el ciclo participa la prelación «{errata[0]} precede a {errata[1]}». Pero el resto del "
                         f"cuadro dice que {errata[0]} es descendiente de {errata[1]} ({cadena}): {errata[0]} es una "
                         f"actividad tardía anotada como predecesora de una temprana. Esa anotación es la errata."))
        celdas.append(code(f"prelaciones_dadas['{errata[1]}'].discard('{errata[0]}')\n"
                           f"G.remove_edge('{errata[0]}', '{errata[1]}')\n"
                           "_quedan = list(nx.simple_cycles(G))\n"
                           f"Markdown('**Respuesta:** el cuadro no es coherente: hay que eliminar la prelación «{errata[0]} precede a "
                           f"{errata[1]}». ' + ('Tras eliminarla no queda ningún ciclo.' if not _quedan else f'Aún quedan ciclos: {{_quedan}}.'))"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Una prelación «p precede a a» es redundante cuando p es antecesora de alguna otra "
                     "predecesora inmediata de a: la relación ya está garantizada por transitividad. "
                     "La reducción transitiva del grafo de prelaciones conserva exactamente las no redundantes."))
    celdas.append(code("reducido = nx.transitive_reduction(G)\n"
                       "redundantes = sorted(set(G.edges) - set(reducido.edges))\n"
                       "_lineas = []\n"
                       "for p, a in redundantes:\n"
                       "    via = sorted(q for q in prelaciones_dadas[a] if q != p and p in nx.ancestors(G, q))\n"
                       "    camino = nx.shortest_path(reducido, p, via[0])\n"
                       "    _lineas.append(f'- **{p} → {a}** es redundante: {a} tiene también como predecesora inmediata a '\n"
                       "                   f'{lista_y(via)}, y {p} ya precede a {via[0]} ({\" → \".join(camino)}); luego {p} precede a {a} '\n"
                       "                   f'por transitividad: {\" → \".join(camino + [a])}.')\n"
                       "_txt = lista_y([f'{p} → {a}' for p, a in redundantes])\n"
                       "Markdown('\\n'.join(_lineas) + f'\\n\\n**Respuesta:** {\"sobran\" if len(redundantes) > 1 else \"sobra\"} '\n"
                       "         f'{_txt}; quitándolas se obtiene el cuadro mínimo siguiente.')"))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones inmediatas mínimo"\n#| label: tbl-prelacionesMinimas\n'
                       "prelaciones_minimas = {a: ','.join(sorted(reducido.predecessors(a))) or '---' for a in sorted(G.nodes)}\n"
                       "datos_proyecto = pd.DataFrame({'predecessor': pd.Series(prelaciones_minimas)})\n"
                       "datos_proyecto.index.name = 'activity'\n"
                       "datos_proyecto"))
    celdas.append(md("La librería hace esta misma depuración al construir la tabla de prelaciones, por lo que "
                     "podemos usarla como comprobación:"))
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "pd.Series({a: ','.join(sorted(ps)) or '---' for a, ps in prelaciones.immediate_predecessor.items()}).to_frame('predecessor')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("En el cuadro expandido cada fila indica las predecesoras inmediatas de una actividad y "
                     "cada columna sus sucesoras inmediatas. Una fila vacía señala una actividad inicial; "
                     "una columna vacía, una actividad final."))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas.append(code(_INI_FIN))
    celdas += apartado(ap); ap += 1
    celdas.append(code("proyecto = prelaciones.create_project()"))
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros (prelaciones transitivas)"\n#| label: tbl-ancestros\n'
                       "proyecto.display_distant_linkage_matrix()"))
    celdas.append(code("ancestros = proyecto.distant_predecessor()\n"
                       "_lineas = [f'- Fila {a}: {p} {\"sí\" if p in ancestros[a] else \"no\"} figura entre los ancestros de {a} '\n"
                       "           f'({\", \".join(sorted(ancestros[a]))}), porque {\" → \".join(nx.shortest_path(reducido, p, a))}.'\n"
                       "           for p, a in redundantes]\n"
                       "Markdown('\\n'.join(_lineas) + '\\n\\n**Respuesta:** ' + ('las prelaciones redundantes aparecen en el cuadro de '\n"
                       "         'ancestros: eliminarlas del cuadro de prelaciones inmediatas no pierde información.'\n"
                       "         if all(p in ancestros[a] for p, a in redundantes) else 'alguna redundante no aparece: revise el cuadro.'))"))
    celdas += apartado(ap); ap += 1
    if ESTADO["grafo"] == "roy":
        celdas.append(md("En el grafo Roy cada actividad es un nodo y cada prelación inmediata un arco, sin "
                         "actividades ficticias. Siguiendo hacia atrás los arcos desde cada actividad se "
                         "recorren todos sus ancestros: al seleccionar una actividad en el visor se resaltan en el "
                         "grafo y en su fila del cuadro de ancestros, y deben coincidir."))
    else:
        celdas.append(md("Las actividades ficticias (discontinuas) del grafo transmiten precisamente las "
                         "prelaciones que no pueden representarse compartiendo nodos. Siguiendo hacia atrás los "
                         "arcos desde el final de cada actividad se recorren todos sus ancestros: al seleccionar una "
                         "actividad en el visor se resaltan en el grafo y en su fila del cuadro de ancestros, y deben "
                         "coincidir."))
    celdas += _celdas_grafo()
    sub = f"{n} actividades, {len(redundantes)} prelaciones redundantes" + (", una errata con ciclo" if con_ciclo else "")
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _texto(rng):
    """El cuadro de prelaciones llega como texto corrido, con un contexto narrativo."""
    from .. import contextos as C
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])), con_duraciones=False)
    preds = info["preds"]
    ctx = C.elegir(rng, n)
    desc = dict(zip(preds, ctx["actividades"]))
    suc = P.sucesoras(preds)
    frases = []
    orden = list(preds)
    rng.shuffle(orden)
    plantillas_frase = [
        "{a} ({A}) no puede empezar hasta que termine{n} {ps}.",
        "Para empezar {a} ({A}) hace falta haber terminado {ps}.",
        "{a} ({A}) sigue a {ps}.",
        "Una vez terminada{s} {ps}, puede empezar {a} ({A}).",
    ]
    for a in orden:
        ps = sorted(preds[a])
        if not ps:
            frases.append(f"{desc[a]} ({a}) puede empezar desde el principio.")
            continue
        nombres_ps = " y ".join(", ".join(f"{desc[p]} ({p})" for p in ps).rsplit(", ", 1)) if len(ps) > 1 else f"{desc[ps[0]]} ({ps[0]})"
        f = plantillas_frase[int(rng.integers(0, len(plantillas_frase)))]
        frases.append(f.format(a=desc[a], A=a, ps=nombres_ps, n="n" if len(ps) > 1 else "", s="s" if len(ps) > 1 else ""))
    texto = " ".join(frases)
    celdas = [md("# Enunciado"),
              md(f"El responsable {C.con_de(ctx['tema'])} ha descrito así las actividades y sus dependencias:\n\n> {texto}")]
    celdas.append(preguntas([
        "Construya el cuadro de prelaciones inmediatas a partir del texto (una fila por actividad con sus predecesoras inmediatas).",
        "Construya el cuadro de prelaciones expandido e indique las actividades iniciales y finales.",
        "Construya el cuadro de ancestros y compruebe que ninguna prelación del texto es redundante.",
        ("Dibuje el grafo PERT con el mínimo número de ficticias." if ESTADO["grafo"] != "roy"
         else "Dibuje el grafo Roy del proyecto.")]))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(md("Se recorre el texto actividad por actividad anotando sólo las predecesoras que el texto "
                     "nombra de forma explícita; las que se deducen por transitividad no se anotan."))
    df = P.tabla_base(info)
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones inmediatas"\n#| label: tbl-prelaciones\n'
                       + literal_dataframe(df)))
    celdas.append(code("descripciones = pd.Series({\n" + "".join(f"    {a!r}: {desc[a]!r},\n" for a in preds)
                       + "}, name='Descripción')\n"
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas.append(code(_INI_FIN))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros"\n#| label: tbl-ancestros\n'
                       "proyecto.display_distant_linkage_matrix()"))
    celdas.append(md("Una prelación «p precede a a» del cuadro sería redundante si p fuera ya ancestro de otra "
                     "predecesora inmediata de a. Se comprueba, para cada actividad con varias predecesoras, si "
                     "alguna de ellas figura en la fila de ancestros de otra:"))
    celdas.append(code("anc = proyecto.distant_predecessor()\n"
                       "_filas = {}\n"
                       "for a, ps in sorted(prelaciones.immediate_predecessor.items()):\n"
                       "    if len(ps) > 1:\n"
                       "        _filas[a] = {'predecesoras inmediatas': ', '.join(sorted(ps)),\n"
                       "                     'ancestros de cada una': '; '.join(f'{q}: {\", \".join(sorted(anc[q])) or \"—\"}' for q in sorted(ps)),\n"
                       "                     'redundantes': ', '.join(f'{p} → {a}' for p in sorted(ps) if any(p in anc[q] for q in ps if q != p)) or 'ninguna'}\n"
                       "redundantes = [r for f in _filas.values() if f['redundantes'] != 'ninguna' for r in f['redundantes'].split(', ')]\n"
                       "display(pd.DataFrame(_filas).T)\n"
                       "Markdown('**Respuesta:** ' + ('ninguna predecesora inmediata es ancestro de otra predecesora de la misma actividad, '\n"
                       "         'así que ninguna prelación del texto es redundante.' if not redundantes else f'son redundantes {lista_y(redundantes)}.'))"))
    celdas += apartado(4)
    celdas += _celdas_grafo()
    return celdas, f"{n} actividades descritas en texto ({ctx['tema']})"


def _grafo_erroneo(rng):
    """Se da un grafo PERT dibujado con errores y el cuadro correcto: hay que encontrarlos."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        preds = {a: set(ps) for a, ps in i["preds"].items()}
        G = P.grafo(preds)
        # quitar una prelación (que no deje a nadie sin predecesora) y añadir otra no redundante
        quitables = [(p, a) for a, ps in preds.items() for p in sorted(ps) if len(ps) > 1]
        if not quitables:
            return False
        p_q, a_q = quitables[int(rng.integers(0, len(quitables)))]
        err = {a: set(ps) for a, ps in preds.items()}
        err[a_q].discard(p_q)
        G2 = P.grafo(err)
        nombres = list(preds)
        cand = [(nombres[i_], nombres[j_]) for i_ in range(n) for j_ in range(i_ + 1, n)
                if nombres[i_] not in err[nombres[j_]] and nombres[i_] not in nx.ancestors(G2, nombres[j_])
                and (p_q, a_q) != (nombres[i_], nombres[j_])]
        if not cand:
            return False
        p_a, a_a = cand[int(rng.integers(0, len(cand)))]
        err[a_a].add(p_a)
        err = P.reducir(err)
        try:
            info_err = P.analizar(err)
        except Exception:
            return False
        if not (1 <= i["n_dummies"] <= 4 and 1 <= info_err["n_dummies"] <= 5):
            return False
        estado.update(err=err, quitada=(p_q, a_q), anadida=(p_a, a_a))
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])), con_duraciones=False)
    err, (p_q, a_q), (p_a, a_a) = estado["err"], estado["quitada"], estado["anadida"]
    df = P.tabla_base(info)
    celdas = [md("# Enunciado"),
              md(f"Un compañero ha dibujado el grafo {grafo_nombre()} que se muestra a continuación a partir del cuadro de "
                 "prelaciones adjunto. Sospechamos que el grafo no refleja exactamente el cuadro.")]
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones (correcto)"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    dibujo = ("proyecto_dibujado.pert(durations=False, ranksep=1)" if ESTADO["grafo"] != "roy"
              else "proyecto_dibujado.roy(durations=False, ranksep=1.2, nodesep=1.2)")
    celdas.append(code(f'#| fig-cap: "Grafo {grafo_nombre()} dibujado por el compañero"\n'
                       f'#| label: fig-{ESTADO["grafo"]}Erroneo\n'
                       + literal_dict(P.a_strings(err), "prelaciones_del_grafo") + "\n"
                       "proyecto_dibujado = ProjectGraph.from_dict_of_strings(prelaciones_del_grafo)\n"
                       + dibujo))
    celdas.append(preguntas([
        "Construya el cuadro de ancestros a partir del cuadro de prelaciones y, por separado, a partir del grafo "
        "dibujado (siguiendo los arcos hacia atrás).",
        "Compare ambos cuadros e identifique qué prelaciones sobran y cuáles faltan en el grafo.",
        ("Dibuje el grafo PERT correcto. ¿Cambia el número de ficticias?" if ESTADO["grafo"] != "roy"
         else "Dibuje el grafo Roy correcto. ¿Cambia el número de arcos?")]))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros según el cuadro de prelaciones"\n#| label: tbl-ancestrosCuadro\n'
                       "proyecto.display_distant_linkage_matrix()"))
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros según el grafo dibujado"\n#| label: tbl-ancestrosGrafo\n'
                       "proyecto_dibujado.display_distant_linkage_matrix()"))
    celdas.append(code("anc_cuadro = proyecto.distant_predecessor()\n"
                       "anc_grafo = proyecto_dibujado.distant_predecessor()\n"
                       "difieren = [a for a in datos_proyecto.index if anc_cuadro[a] != anc_grafo[a]]\n"
                       "Markdown(f'**Respuesta:** los dos cuadros de ancestros difieren en '\n"
                       "         f'{\"la fila\" if len(difieren) == 1 else \"las filas\"} de {lista_y(difieren)}; las demás filas coinciden.'\n"
                       "         if difieren else '**Respuesta:** los dos cuadros de ancestros coinciden fila a fila.')"))
    celdas += apartado(2)
    celdas.append(md("Una prelación que está en el cuadro y no en el grafo se detecta porque alguna actividad "
                     "tiene menos ancestros en el grafo; una que sobra, porque tiene más. Para localizar la "
                     "prelación inmediata responsable se comparan las predecesoras inmediatas de las actividades "
                     "cuyo conjunto de ancestros difiere."))
    celdas.append(code("pd.DataFrame({'ancestros según el cuadro': {a: ', '.join(sorted(anc_cuadro[a])) for a in difieren},\n"
                       "              'ancestros según el grafo': {a: ', '.join(sorted(anc_grafo[a])) for a in difieren}})"))
    celdas.append(code("inm_cuadro = prelaciones.immediate_predecessor\n"
                       "inm_grafo = PredecessorTable.from_dict_of_strings(prelaciones_del_grafo).immediate_predecessor\n"
                       "faltan = sorted((p, a) for a in inm_cuadro for p in inm_cuadro[a] - inm_grafo.get(a, set()))\n"
                       "sobran = sorted((p, a) for a in inm_grafo for p in inm_grafo[a] - inm_cuadro.get(a, set()))\n"
                       "def _por_que(p, a, anc):\n"
                       "    return (f'{p} {\"sí\" if p in anc[a] else \"no\"} es ancestro de {a}')\n"
                       "_l = [f'- Falta **{p} → {a}**: está en el cuadro, pero en el grafo {a} no la tiene como predecesora inmediata '\n"
                       "      f'(según el grafo, {_por_que(p, a, anc_grafo)}).' for p, a in faltan]\n"
                       "_l += [f'- Sobra **{p} → {a}**: el grafo hace que {p} preceda directamente a {a}, y en el cuadro no es así '\n"
                       "       f'(según el cuadro, {_por_que(p, a, anc_cuadro)}).' for p, a in sobran]\n"
                       "Markdown('\\n'.join(_l))"))
    celdas.append(code(f"_errores = {{('{p_q}', '{a_q}'), ('{p_a}', '{a_a}')}}\n"
                       "_otras = [f'{p} → {a}' for p, a in faltan + sobran if (p, a) not in _errores]\n"
                       f"Markdown('**Respuesta:** en el grafo falta la prelación {p_q} → {a_q} y sobra la prelación {p_a} → {a_a}.' + "
                       "(f' Las demás diferencias de predecesoras inmediatas ({lista_y(_otras)}) son consecuencia de estas dos: al '\n"
                       "  'añadir o quitar una prelación cambia cuáles son redundantes.' if _otras else ''))"))
    celdas += apartado(3)
    celdas += fig_grafo(durations="False", cap="Grafo PERT correcto", label="fig-pertCorrecto")
    if ESTADO["grafo"] == "roy":
        celdas.append(code("arcos = sum(len(ps) for ps in prelaciones.immediate_predecessor.values())\n"
                           "arcos_dibujado = sum(len(ps) for ps in PredecessorTable.from_dict_of_strings(prelaciones_del_grafo).immediate_predecessor.values())\n"
                           "Markdown(f'**Respuesta:** el grafo correcto tiene {arcos} arcos de prelación frente a {arcos_dibujado} del dibujado: '\n"
                           "         + ('el número no cambia.' if arcos == arcos_dibujado\n"
                           "            else f'{abs(arcos - arcos_dibujado)} {\"más\" if arcos > arcos_dibujado else \"menos\"}.'))"))
    else:
        celdas.append(code(_AUX_FICTICIAS))
        celdas.append(code("_lineas = explica_ficticias(proyecto, prelaciones)\n"
                           "_n, _m = len(proyecto.dummies), len(proyecto_dibujado.dummies)\n"
                           "Markdown(('\\n'.join(_lineas) + '\\n\\n' if _lineas else '') + f'**Respuesta:** el grafo correcto tiene '\n"
                           "         f'{plural(_n, \"ficticia\")} frente a {_m} del dibujado: '\n"
                           "         + ('el número no cambia.' if _n == _m else f'{\"una más\" if _n - _m == 1 else (f\"{_n - _m} más\" if _n > _m else (\"una menos\" if _m - _n == 1 else f\"{_m - _n} menos\"))}.'))"))
    return celdas, f"{n} actividades, grafo con una prelación de más y otra de menos"



def _numeracion_nodos(rng):
    """Numerar los nodos del PERT respetando i < j en cada arco; contar las numeraciones válidas."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and 6 <= i["n_nodos"] <= 9,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])), con_duraciones=False)
    df = P.tabla_base(info)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones se adjunta. En un grafo PERT los nodos "
                 "(sucesos) se numeran de modo que todo arco vaya de un número menor a uno mayor; esa "
                 "numeración no es única.")]
    celdas.append(preguntas([
        "Dibuje el grafo PERT con el mínimo número de ficticias y numere sus nodos. Compruebe que cada "
        "arco $(i, j)$, ficticias incluidas, cumple $i < j$.",
        "¿Qué nodos pueden intercambiar su número sin violar la regla? Dé una segunda numeración válida.",
        "¿Cuántas numeraciones válidas distintas admite el grafo? (Es el número de ordenaciones topológicas; "
        "cuéntelas con un programa.)",
        "¿Por qué conviene que la numeración sea creciente a lo largo de los arcos? Piense en la matriz de "
        "Zaderenko."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="False", cap="Grafo PERT con los nodos numerados", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Arcos del grafo: origen, destino y comprobación i < j"\n#| label: tbl-arcos\n'
                       "arcos = pd.DataFrame([{'actividad': d['activity'], 'i': int(i), 'j': int(j), 'i < j': 'sí' if int(i) < int(j) else 'no'}\n"
                       "                      for i, j, d in sorted(proyecto.pert_graph.edges(data=True), key=lambda e: (int(e[0]), int(e[1])))]).set_index('actividad')\n"
                       "arcos"))
    celdas.append(code("_n_ok = int((arcos['i < j'] == 'sí').sum())\n"
                       "Markdown(f'**Respuesta:** el grafo tiene {len(proyecto.nodes)} nodos, numerados del 1 al {len(proyecto.nodes)}, '\n"
                       "         f'y {plural(len(proyecto.dummies), \"ficticia\")}. La numeración cumple $i < j$ en {_n_ok} de los {len(arcos)} arcos'\n"
                       "         + (' (todos, ficticias incluidas).' if _n_ok == len(arcos) else '.'))"))
    celdas += apartado(2)
    celdas.append(md("Dos nodos pueden intercambiar su número cuando no hay ningún camino entre ellos (ninguno "
                     "es antecesor del otro). Se comprueba con el grafo: el par $(u, v)$ es intercambiable si "
                     "$v$ no es alcanzable desde $u$ ni $u$ desde $v$."))
    celdas.append(code("G = proyecto.pert_graph\n"
                       "nodos = sorted(G.nodes)\n"
                       "intercambiables = [(u, v) for u in nodos for v in nodos if u < v\n"
                       "                   and not nx.has_path(G, u, v) and not nx.has_path(G, v, u)]\n"
                       "_txt = ', '.join(f'{u} y {v}' for u, v in intercambiables)\n"
                       "Markdown(f'Pares de nodos sin ningún camino entre ellos: {_txt}.' if intercambiables else 'No hay ningún par de nodos sin camino entre ellos.')"))
    celdas.append(code("if intercambiables:\n"
                       "    u, v = intercambiables[0]\n"
                       "    renumerar = {int(u): int(v), int(v): int(u)}\n"
                       "    arcos2 = arcos.copy()\n"
                       "    arcos2['i'] = arcos2['i'].map(lambda x: renumerar.get(x, x)); arcos2['j'] = arcos2['j'].map(lambda x: renumerar.get(x, x))\n"
                       "    arcos2['i < j'] = ['sí' if i < j else 'no' for i, j in zip(arcos2['i'], arcos2['j'])]\n"
                       "    _cambian = [a for a in arcos.index if (arcos.loc[a, 'i'], arcos.loc[a, 'j']) != (arcos2.loc[a, 'i'], arcos2.loc[a, 'j'])]\n"
                       "    display(Markdown(f'**Respuesta:** pueden intercambiar su número {_txt}. Por ejemplo, intercambiando los nodos {u} y {v} '\n"
                       "                     f'cambian los arcos de {lista_y(_cambian)} y todos siguen cumpliendo $i < j$: '\n"
                       "                     f'{\"sí\" if (arcos2[\"i < j\"] == \"sí\").all() else \"no\"}. Segunda numeración válida:'))\n"
                       "    display(arcos2)\n"
                       "else:\n"
                       "    display(Markdown('**Respuesta:** no hay dos nodos intercambiables: cada nodo es antecesor o descendiente de todos los demás y la numeración es única.'))"))
    celdas += apartado(3)
    celdas.append(md("Las numeraciones válidas son las ordenaciones topológicas del grafo de sucesos. Se cuentan "
                     "con programación dinámica sobre subconjuntos: el número de formas de ordenar un conjunto "
                     "$S$ es la suma, sobre cada nodo $v$ de $S$ sin antecesores dentro de $S$, de las formas de "
                     "ordenar $S \\setminus \\{v\\}$."))
    celdas.append(code("from functools import lru_cache\n"
                       "antecesores = {v: {u for u in nodos if nx.has_path(G, u, v) and u != v} for v in nodos}\n"
                       "@lru_cache(maxsize=None)\n"
                       "def ordenaciones(restantes):\n"
                       "    if not restantes:\n"
                       "        return 1\n"
                       "    S = set(restantes)\n"
                       "    return sum(ordenaciones(tuple(sorted(S - {v}))) for v in S if not (antecesores[v] & S))\n"
                       "n_numeraciones = ordenaciones(tuple(nodos))\n"
                       "Markdown(f'**Respuesta:** el grafo admite **{plural(n_numeraciones, \"numeración válida\", \"numeraciones válidas\")}**.')"))
    celdas.append(code("from itertools import islice\n"
                       "_k = min(3, n_numeraciones)\n"
                       "_l = [', '.join(f'{viejo}→{nuevo + 1}' for nuevo, viejo in enumerate(orden)) for orden in islice(nx.all_topological_sorts(G), _k)]\n"
                       "Markdown(('Las dos' if _k == 2 else ('La única' if _k == 1 else 'Tres de ellas')) + ' (nodo de la librería → nuevo número), generadas con networkx:\\n\\n'\n"
                       "         + '\\n'.join(f'- {x}' for x in _l))"))
    celdas += apartado(4)
    celdas.append(md("Con numeración creciente, la matriz de Zaderenko es triangular superior: cada arco está por "
                     "encima de la diagonal, los tiempos tempranos se calculan recorriendo las columnas de "
                     "izquierda a derecha y los tardíos las filas de abajo arriba, sin tener que buscar un orden "
                     "de cálculo. Además, la regla garantiza que no hay ciclos."))
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko sin duraciones: en la celda (i, j) figura la actividad del arco i → j"\n#| label: tbl-zaderenko\n'
                       "Z = pd.DataFrame('', index=nodos, columns=nodos)\n"
                       "for i, j, d in G.edges(data=True):\n"
                       "    Z.loc[i, j] = d['activity']\n"
                       "display(Z)\n"
                       "Markdown('**Respuesta:** con la numeración creciente todos los arcos quedan por encima de la diagonal, así que '\n"
                       "         'los tiempos de cada nodo se pueden calcular en el orden de su número (hacia delante por columnas y '\n"
                       "         'hacia atrás por filas) sabiendo que los nodos de los que depende ya están calculados.')"))
    return celdas, f"{n} actividades, {info['n_nodos']} nodos; numeración de los nodos"


def _datos_sucios(rng):
    """El cuadro llega con nombres en minúscula, separadores distintos, espacios, duplicados y «ninguna»."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3, n=n, n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), con_duraciones=False)
    preds = info["preds"]
    sin_pred = ["---", "ninguna", "-", "", "Ninguna"]
    seps = [", ", ",", "; ", " y ", " / "]
    sucio = {}
    for a, ps in preds.items():
        if not ps:
            sucio[a] = sin_pred[int(rng.integers(0, len(sin_pred)))]
            continue
        lista = sorted(ps)
        if len(lista) >= 1 and rng.random() < 0.35:
            lista = lista + [lista[0]]                 # duplicado
        partes = [(p.lower() if rng.random() < 0.3 else p) + (" " if rng.random() < 0.3 else "") for p in lista]
        sep = seps[int(rng.integers(0, len(seps)))]
        sucio[a] = (" " if rng.random() < 0.3 else "") + sep.join(partes)
    filas = list(preds)
    nombres_sucios = {a: (a.lower() if rng.random() < 0.25 else a) for a in filas}
    tabla = pd.DataFrame({"actividad": [nombres_sucios[a] for a in filas], "predecesoras": [sucio[a] for a in filas]})
    celdas = [md("# Enunciado"),
              md("Un becario ha transcrito el cuadro de prelaciones de un proyecto tal como se adjunta: con "
                 "mayúsculas y minúsculas mezcladas, separadores distintos, espacios de más, alguna predecesora "
                 "repetida y varias formas de decir «ninguna».")]
    celdas.append(preguntas([
        "Enumere los defectos de la tabla que impedirían leerla directamente con la librería.",
        "Normalice la tabla con código (una regla para cada defecto) hasta obtener un cuadro de prelaciones "
        "válido: nombres en mayúscula, predecesoras separadas por comas y `---` para las actividades iniciales.",
        "Construya el proyecto, indique las actividades iniciales y finales y compruebe que no hay "
        "prelaciones redundantes ni ciclos.",
        f"Dibuje el grafo {grafo_nombre()}."]))
    celdas.append(code('#| tbl-cap: "Tabla recibida"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(tabla, indice="actividad")))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(code("t = datos_enunciado.reset_index()\n"
                       "import re\n"
                       "def _partes(s):\n"
                       "    return [p.strip().upper() for p in re.split(r'[,;/]| y ', s) if p.strip()]\n"
                       "def _filas(mascara):\n"
                       "    return lista_y([f'«{a}»' for a in t.loc[mascara, 'actividad']])\n"
                       "_reglas = {\n"
                       "    'nombres de actividad en minúscula': t['actividad'] != t['actividad'].str.upper(),\n"
                       "    'predecesoras en minúscula': t['predecesoras'].map(lambda s: any(p.strip() != p.strip().upper() for p in re.split(r'[,;/]| y ', s)\n"
                       "                                                                 if p.strip().lower() not in ('ninguna', '-', ''))),\n"
                       "    'separadores distintos de la coma (;, y, /)': t['predecesoras'].str.contains(r';| y | /'),\n"
                       "    'espacios sobrantes': (t['predecesoras'] != t['predecesoras'].str.strip()) | t['predecesoras'].str.contains(r'\\s,|,\\s\\s|  '),\n"
                       "    'formas distintas de «---» para indicar que no hay predecesoras': t['predecesoras'].str.strip().str.lower().isin(['ninguna', '-', '']),\n"
                       "    'predecesoras repetidas': t['predecesoras'].map(lambda s: len(_partes(s)) != len(set(_partes(s)))),\n"
                       "}\n"
                       "defectos = [d for d, m in _reglas.items() if m.any()]\n"
                       "Markdown('**Respuesta:** la tabla tiene estos defectos (entre comillas, las filas afectadas):\\n\\n'\n"
                       "         + '\\n'.join(f'- {d}: {_filas(m)}' for d, m in _reglas.items() if m.any()))"))
    celdas += apartado(2)
    celdas.append(md("Cada regla corrige un defecto: pasar a mayúsculas, trocear por cualquiera de los separadores, "
                     "quitar espacios, eliminar repetidos (conservando el orden) y traducir las formas de «ninguna» "
                     "a `---`."))
    celdas.append(code("def normaliza(s):\n"
                       "    s = str(s).strip()\n"
                       "    if s.lower() in ('', '-', '---', 'ninguna'):\n"
                       "        return '---'\n"
                       "    vistas = []\n"
                       "    for p in _partes(s):\n"
                       "        if p not in vistas:\n"
                       "            vistas.append(p)\n"
                       "    return ','.join(vistas)\n"
                       "limpio = pd.DataFrame({'activity': t['actividad'].str.strip().str.upper(),\n"
                       "                       'predecessor': t['predecesoras'].map(normaliza)}).set_index('activity')\n"
                       "pd.concat({'recibido': t.set_index(limpio.index)['predecesoras'], 'normalizado': limpio['predecessor']}, axis=1)"))
    celdas.append(code("desconocidas = {p for s in limpio['predecessor'] if s != '---' for p in s.split(',')} - set(limpio.index)\n"
                       "Markdown('**Respuesta:** la tabla normalizada es un cuadro de prelaciones válido: ' + ('todas las predecesoras son '\n"
                       "         'actividades de la tabla.' if not desconocidas else f'¡ojo!, hay predecesoras que no son actividades: {lista_y(sorted(desconocidas))}.'))"))
    celdas += apartado(3)
    celdas.append(code("datos_proyecto = limpio.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       + _INI_FIN))
    celdas.append(code("G = nx.DiGraph([(p, a) for a, ps in prelaciones.immediate_predecessor.items() for p in ps])\n"
                       "G.add_nodes_from(datos_proyecto.index)\n"
                       "_aciclico = nx.is_directed_acyclic_graph(G)\n"
                       "redundantes = sorted(set(G.edges) - set(nx.transitive_reduction(G).edges))\n"
                       "Markdown(('No hay ciclos (el grafo de prelaciones es acíclico)' if _aciclico else 'Hay ciclos') + ' y '\n"
                       "         + ('no hay prelaciones redundantes: ninguna predecesora inmediata es ancestro de otra predecesora de la misma actividad.'\n"
                       "            if not redundantes else 'son redundantes ' + lista_y([f'{p} → {a}' for p, a in redundantes]) + '.'))"))
    celdas += apartado(4)
    celdas += _celdas_grafo()[:2]             # visor de construcción (sin el de ancestros, que aquí no se pide)
    if ESTADO["grafo"] != "roy":
        celdas += _celdas_grafo()[3:]
    return celdas, f"{n} actividades; tabla con {len(set(sucio.values()))} formatos distintos que normalizar"


VARIANTES = [
    Variante("redundancias", 1, "cuadro con prelaciones redundantes: depurarlo, cuadro expandido y de ancestros",
             lambda rng: _base(1, rng), grafo="alterno"),
    Variante("redundancias-b", 1, "ídem con otro proyecto", lambda rng: _base(2, rng), grafo="alterno"),
    Variante("ciclo", 2, "errata que crea un ciclo: detectarla con la librería y corregirla antes de depurar",
             lambda rng: _base(3, rng), grafo="alterno"),
    Variante("texto", 2, "las prelaciones llegan como texto corrido con contexto; hay que tabularlas", _texto, grafo="alterno"),
    Variante("grafo-erroneo", 3, "se da un grafo PERT con una prelación de más y otra de menos: encontrarlas "
                                 "comparando cuadros de ancestros", _grafo_erroneo, grafo="alterno"),
    Variante("numeracion-nodos", 2, "numerar los nodos del PERT con i < j en cada arco; nodos intercambiables; contar las numeraciones válidas",
             _numeracion_nodos, grafo="pert"),
    Variante("datos-sucios", 1, "cuadro transcrito con minúsculas, separadores y duplicados: normalizarlo con reglas antes de construir el proyecto",
             _datos_sucios, grafo="alterno"),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
