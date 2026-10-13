"""Colección 01 — Cuadro de prelaciones: depurar redundancias y erratas,
cuadro de prelaciones expandido, cuadro de ancestros, iniciales y finales."""
import networkx as nx
import numpy as np

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado,
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
    celdas.append(code('#| fig-cap: "Grafo PERT del proyecto"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
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
    celdas.append(code('#| fig-cap: "Grafo PERT del proyecto"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
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
    celdas.append(code('#| fig-cap: "Grafo PERT correcto"\n#| label: fig-pertCorrecto\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(code("Markdown(f'El grafo correcto tiene {len(proyecto.dummies)} ficticia(s) frente a {len(proyecto_dibujado.dummies)} del dibujado.')"))
    return celdas, f"{n} actividades, grafo con una prelación de más y otra de menos"


VARIANTES = [
    Variante("redundancias", 1, "cuadro con prelaciones redundantes: depurarlo, cuadro expandido y de ancestros",
             lambda rng: _base(1, rng)),
    Variante("redundancias-b", 1, "ídem con otro proyecto", lambda rng: _base(2, rng)),
    Variante("ciclo", 2, "errata que crea un ciclo: detectarla con la librería y corregirla antes de depurar",
             lambda rng: _base(3, rng)),
    Variante("texto", 2, "las prelaciones llegan como texto corrido con contexto; hay que tabularlas", _texto),
    Variante("grafo-erroneo", 3, "se da un grafo PERT con una prelación de más y otra de menos: encontrarlas "
                                 "comparando cuadros de ancestros", _grafo_erroneo),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
