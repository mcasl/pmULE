"""Colección 03 — Grafo Roy: dibujar el grafo de actividades en los nodos,
compararlo con el PERT y modelar relaciones con desfase (esperas y solapes)."""
import numpy as np

import pandas as pd
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="03_roy", prefijo="Roy",
            titulo="Grafo Roy",
            descripcion="Dibujar el grafo Roy, compararlo con el PERT y modelar esperas y solapes "
                        "entre actividades.",
            n_defecto=3)


def _base(k, rng):
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
    celdas.append(md("Cómo sale cada fecha: para cada actividad, la tabla muestra los fines más tempranos de sus "
                     "predecesoras que se comparan (el máximo es su inicio más temprano) y los inicios más tardíos "
                     "de sus sucesoras (el mínimo es su fin más tardío)."))
    celdas.append(code('#| tbl-cap: "Cálculo paso a paso de las fechas"\n#| label: tbl-fechasPaso\n'
                       "detalle_holguras(calendario, prelaciones)[['ES = máx EF pred.', 'EF = ES + d', 'LF = mín LS suc.', 'LS = LF − d']]"))
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


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _predecesora_dominante(rng):
    """Cálculo hacia delante y hacia atrás sobre el grafo Roy, actividad a actividad, y la
    predecesora que fija el inicio de cada una."""
    n = int(rng.choice([8, 9]))

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["n_criticas"] == 1):
            return False
        # al menos dos actividades con varias predecesoras cuyo máximo no empata
        cal, preds = i["calendario"], i["preds"]
        buenas = 0
        for a, ps in preds.items():
            if len(ps) >= 2:
                fines = sorted(cal.loc[list(ps), "fin_mas_temprano"])
                if fines[-1] > fines[-2]:
                    buenas += 1
        return buenas >= 2

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, preds = info["D"], info["calendario"], info["preds"]
    df = P.tabla_base(info, duration=D)
    multi = [a for a, ps in preds.items() if len(ps) >= 2]
    objetivo = multi[int(rng.integers(0, len(multi)))]
    dominante = cal.loc[sorted(preds[objetivo]), "fin_mas_temprano"].idxmax()
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta.")]
    celdas.append(preguntas([
        "Dibuje el grafo Roy y calcule sobre él, actividad a actividad y en orden topológico, el inicio y el fin "
        "más tempranos: el inicio más temprano es el mayor de los fines más tempranos de las predecesoras.",
        "Calcule del mismo modo, de atrás hacia delante, los fines e inicios más tardíos.",
        f"Para cada actividad con varias predecesoras indique cuál de ellas fija su inicio más temprano "
        f"(la *predecesora dominante*). En particular, ¿qué predecesora de {objetivo} manda?",
        f"Si se quisiera adelantar el inicio de {objetivo} un día, ¿qué actividad habría que acortar? ¿Serviría de algo "
        f"acortar sus otras predecesoras?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "predecesoras = prelaciones.immediate_predecessor\n"
                       "sucesoras = {a: sorted(s for s, ps in predecesoras.items() if a in ps) for a in datos_proyecto.index}\n"
                       "duracion = datos_proyecto['duration']"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Grafo Roy del proyecto"\n#| label: fig-roy\n'
                       "proyecto.roy(ranksep=1.2, nodesep=1.2)"))
    celdas.append(md("Recorriendo las actividades en un orden en que cada una aparece después de sus "
                     "predecesoras (orden topológico):"))
    celdas.append(code("G = nx.DiGraph([(p, a) for a, ps in predecesoras.items() for p in ps])\n"
                       "G.add_nodes_from(datos_proyecto.index)\n"
                       "orden = list(nx.topological_sort(G))\n"
                       "ES, EF, filas = {}, {}, []\n"
                       "for a in orden:\n"
                       "    ps = sorted(predecesoras[a])\n"
                       "    ES[a] = max((EF[p] for p in ps), default=0)\n"
                       "    EF[a] = ES[a] + int(duracion[a])\n"
                       "    filas.append({'actividad': a, 'predecesoras (EF)': ', '.join(f'{p} ({EF[p]})' for p in ps) or '—',\n"
                       "                  'ES = máx': ES[a], 'duración': int(duracion[a]), 'EF = ES + d': EF[a]})\n"
                       "pd.DataFrame(filas).set_index('actividad')"))
    celdas += apartado(2)
    celdas.append(code("duracion_proyecto = max(EF.values())\n"
                       "LS, LF, filas = {}, {}, []\n"
                       "for a in reversed(orden):\n"
                       "    ss = sucesoras[a]\n"
                       "    LF[a] = min((LS[s] for s in ss), default=duracion_proyecto)\n"
                       "    LS[a] = LF[a] - int(duracion[a])\n"
                       "    filas.append({'actividad': a, 'sucesoras (LS)': ', '.join(f'{s} ({LS[s]})' for s in ss) or '—',\n"
                       "                  'LF = mín': LF[a], 'duración': int(duracion[a]), 'LS = LF − d': LS[a]})\n"
                       "pd.DataFrame(filas).set_index('actividad')"))
    celdas.append(code("calendario = proyecto.calendar(duracion)\n"
                       "coincide = all(calendario.loc[a, 'inicio_mas_temprano'] == ES[a] and calendario.loc[a, 'inicio_mas_tardio'] == LS[a] for a in orden)\n"
                       "Markdown(f'La duración del proyecto es {duracion_proyecto} días. El cálculo coincide con el de la librería: {coincide}.')"))
    celdas += apartado(3)
    celdas.append(code("dominantes = {}\n"
                       "for a in datos_proyecto.index:\n"
                       "    ps = sorted(predecesoras[a])\n"
                       "    if len(ps) >= 2:\n"
                       "        maximo = max(EF[p] for p in ps)\n"
                       "        dominantes[a] = ', '.join(p for p in ps if EF[p] == maximo)\n"
                       "pd.Series(dominantes).to_frame('predecesora dominante')"))
    celdas.append(md(f"La predecesora dominante de {objetivo} es {dominante}: su fin más temprano es el mayor de "
                     f"entre las predecesoras y por tanto el que fija el inicio de {objetivo}. Las demás "
                     f"predecesoras terminan antes y tienen margen (holgura libre) respecto a {objetivo}."))
    celdas += apartado(4)
    celdas.append(code(f"objetivo, dominante = '{objetivo}', '{dominante}'\n"
                       "otras = [p for p in sorted(predecesoras[objetivo]) if p != dominante]\n"
                       "margen = {p: EF[dominante] - EF[p] for p in otras}\n"
                       "d2 = duracion.copy(); d2[dominante] -= 1\n"
                       "ES_nuevo = int(proyecto.calendar(d2).loc[objetivo, 'inicio_mas_temprano'])\n"
                       "Markdown(f'''Para adelantar {objetivo} hay que acortar {dominante} (o alguna de sus antecesoras en la cadena que la "
                       "hace llegar la última): acortándola un día, {objetivo} empieza en {ES_nuevo} en vez de {ES[objetivo]}. "
                       "Acortar {\", \".join(otras)} no sirve: terminan {\", \".join(f\"{p} {m} día(s)\" for p, m in margen.items())} antes que {dominante} y "
                       "su holgura sólo aumentaría. Sólo cuando {dominante} se acorte tanto que otra predecesora pase a ser la última cambiará la dominante.''')"))
    return celdas, f"{n} actividades, cálculo manual sobre Roy; predecesora dominante de {objetivo}"



def _desde_sucesoras(rng):
    """El cuadro da las sucesoras inmediatas de cada actividad; hay que invertirlo antes de dibujar."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and i["n_criticas"] == 1,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, preds = info["D"], info["preds"]
    suc = P.sucesoras(preds)
    tabla = pd.DataFrame({"successor": {a: (",".join(sorted(s)) if s else "---") for a, s in suc.items()},
                          "duration": D})
    tabla.index.name = "activity"
    celdas = [md("# Enunciado"),
              md("En lugar del cuadro de prelaciones habitual, el responsable del proyecto ha anotado para "
                 "cada actividad cuáles pueden empezar *inmediatamente después* de ella (sus sucesoras "
                 "inmediatas) y su duración en semanas.")]
    celdas.append(preguntas([
        "Construya el cuadro de prelaciones (predecesoras inmediatas) a partir del de sucesoras. ¿Qué "
        "actividades son iniciales y cuáles finales? ¿Cómo se reconocen en cada uno de los dos cuadros?",
        "Dibuje el grafo Roy y el grafo PERT. ¿Cuántas ficticias necesita el PERT?",
        "Calcule el calendario y el camino crítico.",
        "Compruebe que el cuadro de sucesoras y el de predecesoras contienen la misma información: "
        "reconstruya el de sucesoras a partir del de predecesoras."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado (sucesoras inmediatas)"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(tabla)))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(md("Si $B$ aparece entre las sucesoras de $A$, entonces $A$ es predecesora de $B$. Se recorre "
                     "el cuadro fila a fila anotando cada actividad como predecesora de las que figuran en su "
                     "casilla. Las iniciales son las que no aparecen en ninguna casilla de sucesoras (ninguna "
                     "las precede); las finales, las que tienen `---` en sucesoras."))
    celdas.append(code("predecesoras = {a: [] for a in datos_enunciado.index}\n"
                       "for a, s in datos_enunciado['successor'].items():\n"
                       "    if s != '---':\n"
                       "        for b in s.split(','):\n"
                       "            predecesoras[b.strip()].append(a)\n"
                       "datos_proyecto = pd.DataFrame({'predecessor': {a: (','.join(sorted(ps)) if ps else '---') for a, ps in predecesoras.items()},\n"
                       "                              'duration': datos_enunciado['duration']})\n"
                       "datos_proyecto.index.name = 'activity'\n"
                       "datos_proyecto"))
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "print('Iniciales (sin predecesoras):', ', '.join(sorted(prelaciones.start_activities)))\n"
                       "print('Finales (sin sucesoras):     ', ', '.join(sorted(prelaciones.end_activities)))"))
    celdas += apartado(2)
    celdas.append(code('#| fig-cap: "Grafo Roy"\n#| label: fig-roy\n'
                       "proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)"))
    celdas.append(code('#| fig-cap: "Grafo PERT"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    celdas.append(code("Markdown(f'El PERT necesita {len(proyecto.dummies)} actividad(es) ficticia(s); el Roy, ninguna.')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas.append(code('#| tbl-cap: "Cálculo paso a paso de las fechas"\n#| label: tbl-fechasPaso\n'
                       "detalle_holguras(calendario, prelaciones)[['ES = máx EF pred.', 'EF = ES + d', 'LF = mín LS suc.', 'LS = LF − d']]"))
    celdas.append(code("duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])\n"
                       "Markdown(f'La duración del proyecto es {duracion_proyecto} semanas.')"))
    celdas += apartado(4)
    celdas.append(code("sucesoras = {a: sorted(b for b, ps in prelaciones.immediate_predecessor.items() if a in ps) for a in datos_proyecto.index}\n"
                       "reconstruido = pd.Series({a: (','.join(s) if s else '---') for a, s in sucesoras.items()}, name='successor')\n"
                       "coincide = (reconstruido == datos_enunciado['successor'].map(lambda s: ','.join(sorted(s.split(','))) if s != '---' else s)).all()\n"
                       "display(pd.concat({'enunciado': datos_enunciado['successor'], 'reconstruido': reconstruido}, axis=1))\n"
                       "Markdown(f'Los dos cuadros coinciden: {coincide}. Son la misma relación leída en los dos sentidos.')"))
    return celdas, f"{n} actividades; cuadro de sucesoras, {info['n_dummies']} ficticias en PERT"


VARIANTES = [
    Variante("roy-vs-pert", 1, "grafo Roy, PERT equivalente, comparación de nodos y arcos, calendario y camino crítico",
             lambda rng: _base(1, rng)),
    Variante("roy-vs-pert-b", 1, "ídem con otro proyecto", lambda rng: _base(2, rng)),
    Variante("esperas-y-solapes", 3, "espera entre dos actividades y solape comienzo-comienzo modelados con actividades auxiliares",
             lambda rng: _base(3, rng)),
    Variante("predecesora-dominante", 2, "cálculo hacia delante y hacia atrás a mano sobre Roy; qué predecesora fija cada inicio",
             _predecesora_dominante),
    Variante("desde-sucesoras", 1, "el cuadro da sucesoras en lugar de predecesoras: invertirlo, Roy, PERT y calendario",
             _desde_sucesoras),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
