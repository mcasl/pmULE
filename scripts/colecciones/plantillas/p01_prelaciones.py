"""Colección 01 — Cuadro de prelaciones: depurar redundancias y erratas,
cuadro de prelaciones expandido, cuadro de ancestros, iniciales y finales."""
import networkx as nx
import numpy as np

from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

META = dict(slug="01_prelaciones", prefijo="Prelaciones",
            titulo="Cuadro de prelaciones",
            descripcion="Depurar un cuadro de prelaciones (redundancias, erratas que crean ciclos), "
                        "construir el cuadro expandido y el de ancestros.",
            n_defecto=3)


def generar(k, rng):
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
