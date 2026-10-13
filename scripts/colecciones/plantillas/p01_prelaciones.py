"""Colección 01 — Cuadro de prelaciones: depurar redundancias y erratas,
cuadro de prelaciones expandido, cuadro de ancestros, iniciales y finales."""
import networkx as nx
import numpy as np

import pandas as pd
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

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
        candidatos = [(d, e) for e in preds for d in nx.descendants(G, e) if e not in preds[d]]
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
              "Dibuje el grafo PERT del proyecto y compruebe sobre él el cuadro de ancestros."]
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
                           "ciclos"))
        celdas.append(md(f"En el ciclo participa la prelación «{errata[0]} precede a {errata[1]}». Como "
                         f"{errata[0]} es, por el resto del cuadro, descendiente de {errata[1]}, esa anotación "
                         f"es la errata: eliminándola el cuadro vuelve a ser acíclico."))
        celdas.append(code(f"prelaciones_dadas['{errata[1]}'].discard('{errata[0]}')\n"
                           f"G.remove_edge('{errata[0]}', '{errata[1]}')\n"
                           "print('Ciclos restantes:', list(nx.simple_cycles(G)))"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Una prelación «p precede a a» es redundante cuando p es antecesora de alguna otra "
                     "predecesora inmediata de a: la relación ya está garantizada por transitividad. "
                     "La reducción transitiva del grafo de prelaciones conserva exactamente las no redundantes."))
    celdas.append(code("reducido = nx.transitive_reduction(G)\n"
                       "redundantes = sorted(set(G.edges) - set(reducido.edges))\n"
                       "for p, a in redundantes:\n"
                       "    via = sorted(q for q in prelaciones_dadas[a] if q != p and p in nx.ancestors(G, q))\n"
                       "    print(f'{p} → {a} es redundante: {p} ya precede a {\", \".join(via)}, que precede(n) a {a}')"))
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
    celdas.append(code("print('Actividades iniciales:', ', '.join(sorted(prelaciones.start_activities)))\n"
                       "print('Actividades finales:  ', ', '.join(sorted(prelaciones.end_activities)))"))
    celdas += apartado(ap); ap += 1
    celdas.append(code("proyecto = prelaciones.create_project()"))
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros (prelaciones transitivas)"\n#| label: tbl-ancestros\n'
                       "proyecto.distant_predecessor(format='DataFrame')"))
    celdas.append(code("ancestros = proyecto.distant_predecessor()\n"
                       "for p, a in redundantes:\n"
                       "    print(f'{p} ∈ ancestros({a}): {p in ancestros[a]}')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Las actividades ficticias (discontinuas) del grafo transmiten precisamente las "
                     "prelaciones que no pueden representarse compartiendo nodos. Siguiendo hacia atrás los "
                     "arcos desde el final de cada actividad se recorren todos sus ancestros."))
    celdas += fig_grafo(durations="False", cap="Grafo PERT del proyecto", label="fig-pert")
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
              md(f"El responsable de {ctx['tema']} ha descrito así las actividades y sus dependencias:\n\n> {texto}")]
    celdas.append(preguntas([
        "Construya el cuadro de prelaciones inmediatas a partir del texto (una fila por actividad con sus predecesoras inmediatas).",
        "Construya el cuadro de prelaciones expandido e indique las actividades iniciales y finales.",
        "Construya el cuadro de ancestros y compruebe que ninguna prelación del texto es redundante.",
        "Dibuje el grafo PERT con el mínimo número de ficticias."]))
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
    celdas.append(code("print('Actividades iniciales:', ', '.join(sorted(prelaciones.start_activities)))\n"
                       "print('Actividades finales:  ', ', '.join(sorted(prelaciones.end_activities)))"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros"\n#| label: tbl-ancestros\n'
                       "proyecto.distant_predecessor(format='DataFrame')"))
    celdas.append(code("G = nx.DiGraph([(p, a) for a, ps in prelaciones.immediate_predecessor.items() for p in ps])\n"
                       "G.add_nodes_from(datos_proyecto.index)\n"
                       "redundantes = sorted(set(G.edges) - set(nx.transitive_reduction(G).edges))\n"
                       "print('Prelaciones redundantes:', redundantes or 'ninguna')"))
    celdas += apartado(4)
    celdas += fig_grafo(durations="False", cap="Grafo PERT del proyecto", label="fig-pert")
    return celdas, f"{n} actividades descritas en texto ({ctx['tema']})"


def _grafo_erroneo(rng):
    """Se da un grafo PERT dibujado con errores y el cuadro correcto: hay que encontrarlos."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        preds = {a: set(ps) for a, ps in i["preds"].items()}
        G = P.grafo(preds)
        # quitar una prelación (que no deje a nadie sin predecesora) y añadir otra no redundante
        quitables = [(p, a) for a, ps in preds.items() for p in ps if len(ps) > 1]
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
              md("Un compañero ha dibujado el grafo PERT que se muestra a continuación a partir del cuadro de "
                 "prelaciones adjunto. Sospechamos que el grafo no refleja exactamente el cuadro.")]
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones (correcto)"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(code('#| fig-cap: "Grafo PERT dibujado por el compañero"\n#| label: fig-pertErroneo\n'
                       + literal_dict(P.a_strings(err), "prelaciones_del_grafo") + "\n"
                       "proyecto_dibujado = ProjectGraph.from_dict_of_strings(prelaciones_del_grafo)\n"
                       "proyecto_dibujado.pert(durations=False, ranksep=1)"))
    celdas.append(preguntas([
        "Construya el cuadro de ancestros a partir del cuadro de prelaciones y, por separado, a partir del grafo "
        "dibujado (siguiendo los arcos hacia atrás).",
        "Compare ambos cuadros e identifique qué prelaciones sobran y cuáles faltan en el grafo.",
        "Dibuje el grafo PERT correcto. ¿Cambia el número de ficticias?"]))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros según el cuadro de prelaciones"\n#| label: tbl-ancestrosCuadro\n'
                       "proyecto.distant_predecessor(format='DataFrame')"))
    celdas.append(code('#| tbl-cap: "Cuadro de ancestros según el grafo dibujado"\n#| label: tbl-ancestrosGrafo\n'
                       "proyecto_dibujado.distant_predecessor(format='DataFrame')"))
    celdas += apartado(2)
    celdas.append(md("Una prelación que está en el cuadro y no en el grafo se detecta porque alguna actividad "
                     "tiene menos ancestros en el grafo; una que sobra, porque tiene más. Para localizar la "
                     "prelación inmediata responsable se comparan las predecesoras inmediatas de las actividades "
                     "cuyo conjunto de ancestros difiere."))
    celdas.append(code("anc_cuadro = proyecto.distant_predecessor()\n"
                       "anc_grafo = proyecto_dibujado.distant_predecessor()\n"
                       "difieren = [a for a in datos_proyecto.index if anc_cuadro[a] != anc_grafo[a]]\n"
                       "pd.DataFrame({'ancestros según el cuadro': {a: ', '.join(sorted(anc_cuadro[a])) for a in difieren},\n"
                       "              'ancestros según el grafo': {a: ', '.join(sorted(anc_grafo[a])) for a in difieren}})"))
    celdas.append(code("inm_cuadro = prelaciones.immediate_predecessor\n"
                       "inm_grafo = PredecessorTable.from_dict_of_strings(prelaciones_del_grafo).immediate_predecessor\n"
                       "faltan = sorted((p, a) for a in inm_cuadro for p in inm_cuadro[a] - inm_grafo.get(a, set()))\n"
                       "sobran = sorted((p, a) for a in inm_grafo for p in inm_grafo[a] - inm_cuadro.get(a, set()))\n"
                       "print('Faltan en el grafo:', faltan)\n"
                       "print('Sobran en el grafo:', sobran)"))
    celdas.append(md(f"Las discrepancias son: falta la prelación {p_q} → {a_q} y sobra la prelación {p_a} → {a_a} "
                     "(si la comparación muestra alguna más, es consecuencia de estas dos: al añadir o quitar una "
                     "prelación cambian las que son redundantes)."))
    celdas += apartado(3)
    celdas += fig_grafo(durations="False", cap="Grafo PERT correcto", label="fig-pertCorrecto")
    celdas.append(code("Markdown(f'El grafo correcto tiene {len(proyecto.dummies)} ficticia(s) frente a {len(proyecto_dibujado.dummies)} del dibujado.')"))
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
                       "arcos = pd.DataFrame([{'actividad': d['activity'], 'i': i, 'j': j, 'i < j': i < j}\n"
                       "                      for i, j, d in proyecto.pert_graph.edges(data=True)]).set_index('actividad')\n"
                       "arcos"))
    celdas.append(code("Markdown(f'La numeración de la librería cumple la regla en {int(arcos[\"i < j\"].sum())} de {len(arcos)} arcos.')"))
    celdas += apartado(2)
    celdas.append(md("Dos nodos pueden intercambiar su número cuando no hay ningún camino entre ellos (ninguno "
                     "es antecesor del otro). Se comprueba con el grafo: el par $(u, v)$ es intercambiable si "
                     "$v$ no es alcanzable desde $u$ ni $u$ desde $v$."))
    celdas.append(code("G = proyecto.pert_graph\n"
                       "nodos = sorted(G.nodes)\n"
                       "intercambiables = [(u, v) for u in nodos for v in nodos if u < v\n"
                       "                   and not nx.has_path(G, u, v) and not nx.has_path(G, v, u)]\n"
                       "print('Pares de nodos intercambiables:', intercambiables or 'ninguno')"))
    celdas.append(code("if intercambiables:\n"
                       "    u, v = intercambiables[0]\n"
                       "    renumerar = {u: v, v: u}\n"
                       "    arcos2 = arcos.copy()\n"
                       "    arcos2['i'] = arcos2['i'].map(lambda x: renumerar.get(x, x)); arcos2['j'] = arcos2['j'].map(lambda x: renumerar.get(x, x))\n"
                       "    arcos2['i < j'] = arcos2['i'] < arcos2['j']\n"
                       "    display(Markdown(f'Intercambiando los nodos {u} y {v} todos los arcos siguen cumpliendo i < j: {bool(arcos2[\"i < j\"].all())}.'))\n"
                       "    display(arcos2)\n"
                       "else:\n"
                       "    display(Markdown('No hay dos nodos intercambiables: la numeración topológica es única.'))"))
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
                       "Markdown(f'El grafo admite **{n_numeraciones}** numeraciones válidas distintas (una sola si cada nodo es antecesor o descendiente de todos los demás).')"))
    celdas.append(code("from itertools import islice\n"
                       "print('Tres de ellas (nodo de la librería → nuevo número), generadas con networkx:')\n"
                       "for orden in islice(nx.all_topological_sorts(G), 3):\n"
                       "    print('  ', {viejo: nuevo + 1 for nuevo, viejo in enumerate(orden)})"))
    celdas += apartado(4)
    celdas.append(md("Con numeración creciente, la matriz de Zaderenko es triangular superior: cada arco está por "
                     "encima de la diagonal, los tiempos tempranos se calculan recorriendo las columnas de "
                     "izquierda a derecha y los tardíos las filas de abajo arriba, sin tener que buscar un orden "
                     "de cálculo. Además, la regla garantiza que no hay ciclos."))
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko (sin duraciones: X marca los arcos)"\n#| label: tbl-zaderenko\n'
                       "Z = pd.DataFrame('', index=nodos, columns=nodos)\n"
                       "for i, j, d in G.edges(data=True):\n"
                       "    Z.loc[i, j] = d['activity']\n"
                       "Z"))
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
        "Dibuje el grafo PERT."]))
    celdas.append(code('#| tbl-cap: "Tabla recibida"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(tabla, indice="actividad")))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(code("defectos = []\n"
                       "t = datos_enunciado.reset_index()\n"
                       "if (t['actividad'] != t['actividad'].str.upper()).any(): defectos.append('nombres de actividad en minúscula')\n"
                       "if t['predecesoras'].str.contains(r'[a-z]').any(): defectos.append('predecesoras en minúscula')\n"
                       "if t['predecesoras'].str.contains(r';| y | /').any(): defectos.append('separadores distintos de la coma (;, y, /)')\n"
                       "if (t['predecesoras'] != t['predecesoras'].str.strip()).any() or t['predecesoras'].str.contains(r'\\s,|,\\s\\s').any(): defectos.append('espacios sobrantes')\n"
                       "if t['predecesoras'].str.lower().isin(['ninguna', '-', '']).any(): defectos.append('varias formas de indicar que no hay predecesoras')\n"
                       "import re\n"
                       "def _partes(s):\n"
                       "    return [p.strip().upper() for p in re.split(r'[,;/]| y ', s) if p.strip()]\n"
                       "if any(len(_partes(s)) != len(set(_partes(s))) for s in t['predecesoras']): defectos.append('predecesoras repetidas')\n"
                       "for d in defectos:\n"
                       "    print('-', d)"))
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
                       "print('Predecesoras que no son actividades de la tabla:', desconocidas or 'ninguna')"))
    celdas += apartado(3)
    celdas.append(code("datos_proyecto = limpio.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "print('Actividades iniciales:', ', '.join(sorted(prelaciones.start_activities)))\n"
                       "print('Actividades finales:  ', ', '.join(sorted(prelaciones.end_activities)))"))
    celdas.append(code("G = nx.DiGraph([(p, a) for a, ps in prelaciones.immediate_predecessor.items() for p in ps])\n"
                       "G.add_nodes_from(datos_proyecto.index)\n"
                       "print('¿Hay ciclos?', not nx.is_directed_acyclic_graph(G))\n"
                       "redundantes = sorted(set(G.edges) - set(nx.transitive_reduction(G).edges))\n"
                       "print('Prelaciones redundantes:', redundantes or 'ninguna')"))
    celdas += apartado(4)
    celdas += fig_grafo(durations="False", cap="Grafo PERT del proyecto", label="fig-pert")
    return celdas, f"{n} actividades; tabla con {len(set(sucio.values()))} formatos distintos que normalizar"


VARIANTES = [
    Variante("redundancias", 1, "cuadro con prelaciones redundantes: depurarlo, cuadro expandido y de ancestros",
             lambda rng: _base(1, rng), grafo="pert"),
    Variante("redundancias-b", 1, "ídem con otro proyecto", lambda rng: _base(2, rng), grafo="pert"),
    Variante("ciclo", 2, "errata que crea un ciclo: detectarla con la librería y corregirla antes de depurar",
             lambda rng: _base(3, rng), grafo="pert"),
    Variante("texto", 2, "las prelaciones llegan como texto corrido con contexto; hay que tabularlas", _texto, grafo="pert"),
    Variante("grafo-erroneo", 3, "se da un grafo PERT con una prelación de más y otra de menos: encontrarlas "
                                 "comparando cuadros de ancestros", _grafo_erroneo, grafo="pert"),
    Variante("numeracion-nodos", 2, "numerar los nodos del PERT con i < j en cada arco; nodos intercambiables; contar las numeraciones válidas",
             _numeracion_nodos, grafo="pert"),
    Variante("datos-sucios", 1, "cuadro transcrito con minúsculas, separadores y duplicados: normalizarlo con reglas antes de construir el proyecto",
             _datos_sucios, grafo="pert"),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
