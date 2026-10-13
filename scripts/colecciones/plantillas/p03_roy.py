"""Colección 03 — Grafo Roy: dibujar el grafo de actividades en los nodos,
compararlo con el PERT y modelar relaciones con desfase (esperas y solapes)."""
import numpy as np

from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe, literal_dict

META = dict(slug="03_roy", prefijo="Roy",
            titulo="Grafo Roy",
            descripcion="Dibujar el grafo Roy, compararlo con el PERT y modelar esperas y solapes "
                        "entre actividades.",
            n_defecto=3)


def generar(k, rng):
    con_desfases = (k % 3 == 0)
    n = int(rng.choice([8, 9]))

    def acepta(i):
        if not (2 <= i["n_dummies"] <= 5 and 5 <= i["n_rutas"] <= 10):
            return False
        if con_desfases:
            suc = P.sucesoras(i["preds"])
            pares_w = [a for a in i["preds"] if i["D"][a] >= 4 and i["D"][a] % 2 == 0 and suc[a]]
            pares_x = [a for a in i["preds"] if suc[a] and a not in pares_w]
            return bool(pares_w) and bool(pares_x)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    preds = info["preds"]
    suc = P.sucesoras(preds)

    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta.")]
    pregs = ["Dibuje el grafo Roy del proyecto (actividades en los nodos).",
             "Dibuje el grafo PERT equivalente. Compare el número de nodos y de arcos de ambos grafos y "
             "explique a qué se debe la diferencia.",
             "Calcule sobre el grafo Roy los tiempos más tempranos y más tardíos de inicio y fin de cada "
             "actividad, y señale el camino crítico en ambos grafos."]
    mods = None
    if con_desfases:
        W = [a for a in preds if D[a] >= 4 and D[a] % 2 == 0 and suc[a]]
        W = W[int(rng.integers(0, len(W)))]
        Z = sorted(suc[W])[0]
        X_cand = [a for a in preds if suc[a] and a != W and a not in preds[W]]
        X = X_cand[int(rng.integers(0, len(X_cand)))]
        Y = sorted(s for s in suc[X] if s != Z)
        Y = Y[0] if Y else sorted(suc[X])[0]
        espera = int(rng.integers(1, 4))
        mods = dict(W=W, Z=Z, X=X, Y=Y, espera=espera)
        pregs.append(f"Replantee el proyecto si, además, entre el final de {X} y el comienzo de {Y} deben "
                     f"transcurrir {espera} semanas de espera, y {Z} puede comenzar cuando {W} está a la mitad "
                     f"de su ejecución. Dibuje el nuevo grafo Roy y calcule la nueva duración del proyecto.")
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas.append(md("En el grafo Roy cada actividad es un nodo y cada prelación inmediata un arco. Se añaden "
                     "dos nodos auxiliares, inicio y fin, para que el grafo tenga un único origen y un único "
                     "destino. No hacen falta actividades ficticias."))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas.append(code('#| fig-cap: "Grafo Roy del proyecto"\n#| label: fig-roy\n'
                       "proyecto.roy(ranksep=1.2, nodesep=1.2)"))
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| fig-cap: "Grafo PERT equivalente"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=False, ranksep=1)"))
    celdas.append(code("n_act = len(proyecto)\n"
                       "n_prelaciones = sum(len(p) for p in prelaciones.immediate_predecessor.values())\n"
                       "n_arcos_roy = n_prelaciones + len(prelaciones.start_activities) + len(prelaciones.end_activities)\n"
                       "pd.DataFrame({'nodos': [n_act + 2, len(proyecto.nodes)],\n"
                       "              'arcos': [n_arcos_roy, n_act + len(proyecto.dummies)],\n"
                       "              'ficticias': [0, len(proyecto.dummies)]},\n"
                       "             index=['Roy', 'PERT'])"))
    celdas.append(md("En Roy el número de nodos es fijo (actividades más inicio y fin) y los arcos son las "
                     "prelaciones. En PERT los arcos son las actividades y los nodos los sucesos; cuando una "
                     "prelación no puede expresarse compartiendo nodos aparece una ficticia, que es el precio "
                     "que paga el PERT por representar las actividades como arcos."))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Sobre el grafo Roy, el inicio más temprano de una actividad es el máximo de los fines "
                     "más tempranos de sus predecesoras; el fin más tardío, el mínimo de los inicios más "
                     "tardíos de sus sucesoras. Son los mismos números que da el PERT."))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas.append(code("duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])\n"
                       "Markdown(f'La duración del proyecto es {duracion_proyecto} semanas.')"))
    celdas.append(code('#| fig-cap: "Grafo Roy con tiempos y camino crítico"\n#| label: fig-royCritico\n'
                       "proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)"))
    celdas.append(code('#| fig-cap: "Grafo PERT con tiempos y camino crítico"\n#| label: fig-pertCritico\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    if con_desfases:
        W, Z, X, Y, espera = mods["W"], mods["Z"], mods["X"], mods["Y"], mods["espera"]
        mitad = int(D[W]) // 2
        nuevo = {a: set(ps) for a, ps in preds.items()}
        # espera entre X e Y: actividad de espera 💤Y
        nuevo[Y].discard(X)
        nuevo[f"💤{Y}"] = {X}
        nuevo[Y].add(f"💤{Y}")
        # solape: W se parte en W1 y W2; Z arranca tras W1; el resto de sucesoras de W tras W2
        nuevo[f"{W}1"] = set(nuevo.pop(W))
        nuevo[f"{W}2"] = {f"{W}1"}
        for a in list(nuevo):
            if W in nuevo[a]:
                nuevo[a].discard(W)
                nuevo[a].add(f"{W}1" if a == Z else f"{W}2")
        dur_nuevo = {a: int(D[a]) for a in preds if a != W}
        dur_nuevo[f"{W}1"] = mitad
        dur_nuevo[f"{W}2"] = int(D[W]) - mitad
        dur_nuevo[f"💤{Y}"] = espera
        nuevo_str = {a: ",".join(sorted(ps)) if ps else "---" for a, ps in nuevo.items()}
        celdas += apartado(ap); ap += 1
        celdas.append(md(f"Las dos condiciones nuevas son relaciones con desfase. Con actividades en los nodos "
                         f"se modelan con actividades auxiliares:\n\n"
                         f"- La espera entre el final de {X} y el comienzo de {Y} es una actividad de espera, "
                         f"💤{Y}, de duración {espera} y sin consumo de recursos, que sucede a {X} y precede a {Y}.\n"
                         f"- Que {Z} pueda comenzar a mitad de {W} es un solape comienzo-comienzo: se parte {W} "
                         f"en {W}1 y {W}2 (de {mitad} y {int(D[W]) - mitad} semanas) y {Z} pasa a suceder a {W}1, "
                         f"mientras que las demás sucesoras de {W} siguen esperando al final de {W}2."))
        celdas.append(code(literal_dict(nuevo_str, "prelaciones_modificadas") + "\n"
                           + literal_dict(dur_nuevo, "duraciones_modificadas") + "\n"
                           "prelaciones_mod = PredecessorTable.from_dict_of_strings(prelaciones_modificadas)\n"
                           "proyecto_mod = prelaciones_mod.create_project()\n"
                           "duraciones_mod = pd.Series(duraciones_modificadas, name='duration')"))
        celdas.append(code('#| fig-cap: "Grafo Roy del proyecto con la espera y el solape"\n#| label: fig-royModificado\n'
                           "proyecto_mod.roy(durations=duraciones_mod, ranksep=1.2, nodesep=1.2)"))
        celdas.append(code('#| tbl-cap: "Calendario del proyecto modificado"\n#| label: tbl-calendarioModificado\n'
                           "proyecto_mod.calendar(duraciones_mod)"))
        celdas.append(code("duracion_mod = proyecto_mod.duration(duraciones_mod)\n"
                           "Markdown(f'La duración pasa de {duracion_proyecto} a {duracion_mod} semanas: la espera añade tiempo si "
                           f"{X}-{Y} está en una ruta crítica y el solape lo quita si {W}-{Z} lo está. Aquí el efecto neto es "
                           "de {duracion_mod - duracion_proyecto:+d} semanas.')"))
    sub = f"{n} actividades, {info['n_dummies']} ficticias en PERT" + (", con espera y solape" if con_desfases else "")
    return celdas, sub
