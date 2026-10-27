"""Colección 03 — Grafo Roy: dibujar el grafo de actividades en los nodos,
compararlo con el PERT y modelar relaciones con desfase (esperas y solapes)."""
import numpy as np

import pandas as pd
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="03_roy", prefijo="Roy",
            titulo="Grafo Roy",
            descripcion="Dibujar el grafo Roy, compararlo con el PERT y modelar esperas y solapes "
                        "entre actividades.",
            n_defecto=3)

# Celda auxiliar (se define en el cuaderno): una línea por ficticia del PERT con los nodos
# que une, la prelación que transmite y por qué hace falta.
_AUX_FICTICIAS = '''
def explica_ficticias(proyecto, prelaciones):
    """Una línea por ficticia del PERT: nodos que une, prelación que transmite y por qué no
    basta con hacer empezar las sucesoras en el nodo donde terminan las predecesoras."""
    G = proyecto.pert_graph
    pred = {a: set(ps) for a, ps in prelaciones.immediate_predecessor.items()}
    R = nx.DiGraph([(p, a) for a, ps in pred.items() for p in ps])
    R.add_nodes_from(pred)
    anc = {a: nx.ancestors(R, a) for a in pred}
    def entran(nodo, salvo=None):
        acts = set()
        for i, _, d in G.in_edges(nodo, data=True):
            if d['activity'] != salvo:
                acts |= entran(i) if d['activity'].startswith('@') else {d['activity']}
        return acts
    def salen(nodo, salvo=None):
        acts = set()
        for _, j, d in G.out_edges(nodo, data=True):
            if d['activity'] != salvo:
                acts |= salen(j) if d['activity'].startswith('@') else {d['activity']}
        return acts
    def arcos_reales(nodo, sentido):
        aristas = G.in_edges(nodo, data=True) if sentido == 'in' else G.out_edges(nodo, data=True)
        return {(u if sentido == 'in' else v): d['activity'] for u, v, d in aristas if not d['activity'].startswith('@')}
    pl = lambda acts, s, p: s if len(acts) == 1 else p
    lineas = []
    arcos = sorted(proyecto.nodes_from_edges.items(), key=lambda kv: (int(kv[1][0]), int(kv[1][1])))
    for nombre, (i, j) in arcos:
        if not nombre.startswith('@'):
            continue
        antes, despues = sorted(entran(i)), sorted(salen(j))
        # prelaciones inmediatas del cuadro que pasan por la ficticia (las demás ya se deducen de otras)
        pares = [(x, y) for x in antes for y in despues if x in pred[y]] or [(x, y) for x in antes for y in despues]
        despues = sorted({y for _, y in pares})
        grupos = {}
        for y in despues:
            grupos.setdefault(tuple(sorted(x for x, z in pares if z == y)), []).append(y)
        transmite = '; '.join(f'{lista_y(xs)} {pl(xs, "precede", "preceden")} a {lista_y(ys)}' for xs, ys in grupos.items())
        # Si los nodos i y j fueran uno solo, lo que llega a j precedería a lo que sale de i
        llegan_j, salen_i = sorted(entran(j, salvo=nombre)), sorted(salen(i, salvo=nombre))
        falsas = [(x, y) for y in salen_i for x in llegan_j if x not in anc[y]]
        if falsas:
            x_s = sorted({x for x, _ in falsas})
            y_s = sorted({y for _, y in falsas})
            directas_j, directas_i = set(arcos_reales(j, 'in').values()), set(arcos_reales(i, 'out').values())
            llega = (f'{pl(x_s, "llega", "llegan")} al nodo {j}' if set(x_s) <= directas_j else
                     f'{pl(x_s, "llega", "llegan")} al nodo {j} (directamente o por otra ficticia)')
            sale = (f'{pl(y_s, "sale", "salen")} de ese nodo' if set(y_s) <= directas_i else
                    f'{pl(y_s, "empieza", "empiezan")} a partir de ese nodo (directamente o tras otra ficticia)')
            motivo = (f'{lista_y(despues)} {pl(despues, "espera", "esperan")} además a {lista_y(x_s)}, que {llega}; '
                      f'si {lista_y(despues)} {pl(despues, "saliera", "salieran")} del nodo {i}, {lista_y(x_s)} '
                      f'{pl(x_s, "precedería", "precederían")} también a {lista_y(y_s)}, que {sale} y no '
                      f'{pl(y_s, "depende", "dependen")} de {pl(x_s, "ella", "ellas")}')
        else:
            gemelas = None
            ini_i, ini_j = arcos_reales(i, 'in'), arcos_reales(j, 'in')
            fin_i, fin_j = arcos_reales(i, 'out'), arcos_reales(j, 'out')
            for u in sorted(set(ini_i) & set(ini_j)):
                gemelas = gemelas or (ini_i[u], ini_j[u], u, j, f'empiezan en el mismo nodo ({u})')
            for v in sorted(set(fin_i) & set(fin_j)):
                gemelas = gemelas or (fin_i[v], fin_j[v], i, v, f'terminan en el mismo nodo ({v})')
            if gemelas:
                a, b, n1, n2, verbo = gemelas
                a, b = sorted([a, b])
                motivo = (f'{a} y {b} {verbo}; sin la ficticia irían '
                          f'las dos del nodo {n1} al {n2}, dos arcos entre el mismo par de nodos que no se podrían distinguir')
            else:
                motivo = 'sin ella habría que fundir los nodos y se perdería la estructura del grafo'
        lineas.append(f'- Ficticia {i} → {j}: transmite que {transmite}. Hace falta porque {motivo}.')
    return lineas
'''

# Celda que explica, con los números del proyecto, de dónde salen los nodos y arcos de cada grafo
_COMPARA_ROY_PERT = r"""_ini, _fin = sorted(prelaciones.start_activities), sorted(prelaciones.end_activities)
_lineas = explica_ficticias(proyecto, prelaciones)
Markdown(f'''**Roy**: un nodo por actividad más los nodos de inicio y fin: {n_act} + 2 = **{n_act + 2} nodos**. Los arcos son las
{n_prelaciones} prelaciones inmediatas del cuadro, más {plural(len(_ini), 'arco')} desde el inicio hasta
{'la actividad inicial' if len(_ini) == 1 else 'las actividades iniciales'} ({lista_y(_ini)}) y {plural(len(_fin), 'arco')} desde
{'la actividad final' if len(_fin) == 1 else 'las actividades finales'} ({lista_y(_fin)}) hasta el fin:
{n_prelaciones} + {len(_ini)} + {len(_fin)} = **{n_arcos_roy} arcos**.

**PERT**: los arcos son las {n_act} actividades más las ficticias: {n_act} + {len(proyecto.dummies)} = **{n_act + len(proyecto.dummies)} arcos**;
los nodos son los sucesos (inicio y fin de grupos de actividades), **{len(proyecto.nodes)} nodos**, porque las actividades con las
mismas predecesoras salen de un mismo nodo y las que tienen las mismas sucesoras llegan a un mismo nodo.

La diferencia se debe a qué representa cada elemento: en el Roy las prelaciones son arcos y nunca hacen falta ficticias; en el
PERT las prelaciones se expresan compartiendo nodos, y cuando una prelación no puede expresarse así aparece una ficticia.
''' + (f'''Este PERT necesita {plural(len(_lineas), 'ficticia')}:

''' + '\n'.join(_lineas) if _lineas else 'Este PERT no necesita ninguna ficticia.'))"""

_DEF_HOLGURAS = ("Holguras: la *total* (LS − ES) es lo que puede retrasarse una actividad sin alargar el "
                 "proyecto; la *libre* (mín ES de las sucesoras − EF) lo que puede retrasarse sin retrasar el "
                 "inicio más temprano de ninguna sucesora; la *independiente* (máx(0, mín ES suc. − máx LF "
                 "pred. − d)) la que conserva aunque sus predecesoras acaben lo más tarde posible y sus "
                 "sucesoras empiecen lo antes posible.")

# Cálculo paso a paso de las fechas, holguras y camino crítico (tras la celda del calendario)
def _fechas_y_critico(donde='en las dos figuras siguientes'):
    return [
    md("Cómo sale cada fecha (ES/EF: inicio y fin más tempranos; LS/LF: inicio y fin más tardíos): para cada "
       "actividad, la tabla muestra los fines más tempranos de sus predecesoras que se comparan (el máximo es su "
       "ES) y los inicios más tardíos de sus sucesoras (el mínimo es su LF)."),
    code('#| tbl-cap: "Cálculo paso a paso de las fechas"\n#| label: tbl-fechasPaso\n'
         "detalle_holguras(calendario, prelaciones)[['ES = máx EF pred.', 'EF = ES + d', 'LF = mín LS suc.', 'LS = LF − d']]"),
    md("El calendario incluye también las holguras. " + _DEF_HOLGURAS.replace("Holguras: la", "La", 1) + " Las actividades de holgura total nula "
       "son las críticas y forman el camino crítico."),
    code("duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
         "explica_camino_critico(proyecto, datos_proyecto['duration'], 'semanas')"),
    code("_criticas = ordena_ruta(calendario.index[calendario['H_total'] == 0], calendario)\n"
         "_holgadas = ordena_ruta(calendario.index[calendario['H_total'] > 0], calendario)\n"
         "_txt = ', '.join(f'{a} ({int(calendario.loc[a, \"H_total\"])})' for a in _holgadas)\n"
         "Markdown(f'**Respuesta:** el proyecto dura {plural(duracion_proyecto, \"semana\")} y son críticas {lista_y(_criticas)} '\n"
         "         f'(holgura total 0). En el grafo Roy el camino crítico pasa por los nodos de esas actividades y en el '\n"
         "         f'PERT por sus arcos (en rojo " + donde + "). '\n"
         "         + (f'Las demás tienen holgura total positiva (entre paréntesis, en semanas): {_txt}.' if _holgadas else 'No hay actividades con holgura.'))"),
    ]


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
    celdas.append(code(_AUX_FICTICIAS))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas.append(md("En el grafo Roy cada actividad es un nodo y cada prelación inmediata un arco. Se añaden "
                     "dos nodos auxiliares, inicio y fin, para que el grafo tenga un único origen y un único "
                     "destino. No hacen falta actividades ficticias."))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas.append(code('#| fig-cap: "Grafo Roy del proyecto"\n#| label: fig-roy\n'
                       "proyecto.display_roy_construccion(tabla=prelaciones)"))
    celdas += apartado(ap); ap += 1
    celdas += fig_grafo(durations="False", cap="Grafo PERT equivalente", label="fig-pert", tipo="pert")
    celdas.append(code("n_act = len(proyecto)\n"
                       "n_prelaciones = sum(len(p) for p in prelaciones.immediate_predecessor.values())\n"
                       "n_arcos_roy = n_prelaciones + len(prelaciones.start_activities) + len(prelaciones.end_activities)\n"
                       "pd.DataFrame({'nodos': [n_act + 2, len(proyecto.nodes)],\n"
                       "              'arcos': [n_arcos_roy, n_act + len(proyecto.dummies)],\n"
                       "              'ficticias': [0, len(proyecto.dummies)]},\n"
                       "             index=['Roy', 'PERT'])"))
    celdas.append(code(_COMPARA_ROY_PERT))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Sobre el grafo Roy, el inicio más temprano de una actividad es el máximo de los fines "
                     "más tempranos de sus predecesoras; el fin más tardío, el mínimo de los inicios más "
                     "tardíos de sus sucesoras. Son los mismos números que da el PERT."))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas += _fechas_y_critico()
    celdas.append(code('#| fig-cap: "Grafo Roy con tiempos y camino crítico"\n#| label: fig-royCritico\n'
                       "proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pertCritico", tipo="pert")
    if con_desfases:
        W, Z, X, Y, espera = mods["W"], mods["Z"], mods["X"], mods["Y"], mods["espera"]
        mitad = int(D[W]) // 2

        def _modifica(con_espera, con_solape):
            nuevo = {a: set(ps) for a, ps in preds.items()}
            dur_nuevo = {a: int(D[a]) for a in preds}
            if con_espera:      # espera entre X e Y: actividad de espera 💤Y
                nuevo[Y].discard(X)
                nuevo[f"💤{Y}"] = {X}
                nuevo[Y].add(f"💤{Y}")
                dur_nuevo[f"💤{Y}"] = espera
            if con_solape:      # solape: W se parte en W1 y W2; Z arranca tras W1; el resto de sucesoras de W tras W2
                nuevo[f"{W}1"] = set(nuevo.pop(W))
                nuevo[f"{W}2"] = {f"{W}1"}
                for a in list(nuevo):
                    if W in nuevo[a]:
                        nuevo[a].discard(W)
                        nuevo[a].add(f"{W}1" if a == Z else f"{W}2")
                del dur_nuevo[W]
                dur_nuevo[f"{W}1"] = mitad
                dur_nuevo[f"{W}2"] = int(D[W]) - mitad
            return {a: ",".join(sorted(ps)) if ps else "---" for a, ps in nuevo.items()}, dur_nuevo

        nuevo_str, dur_nuevo = _modifica(True, True)
        solo_espera, dur_solo_espera = _modifica(True, False)
        solo_solape, dur_solo_solape = _modifica(False, True)
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
                           "explica_camino_critico(proyecto_mod, duraciones_mod, 'semanas')"))
        celdas.append(md("Para ver qué aporta cada cambio se calcula también el proyecto con sólo uno de los dos."))
        celdas.append(code(literal_dict(solo_espera, "prelaciones_solo_espera") + "\n"
                           + literal_dict(dur_solo_espera, "duraciones_solo_espera") + "\n"
                           + literal_dict(solo_solape, "prelaciones_solo_solape") + "\n"
                           + literal_dict(dur_solo_solape, "duraciones_solo_solape") + "\n"
                           "def _duracion(prel, dur):\n"
                           "    return int(PredecessorTable.from_dict_of_strings(prel).create_project().duration(pd.Series(dur)))\n"
                           "T_espera = _duracion(prelaciones_solo_espera, duraciones_solo_espera)\n"
                           "T_solape = _duracion(prelaciones_solo_solape, duraciones_solo_solape)\n"
                           f"X, Y, W, Z, espera, mitad = {X!r}, {Y!r}, {W!r}, {Z!r}, {espera}, {mitad}\n"
                           "c = calendario\n"
                           "margen = int(c.loc[Y, 'inicio_mas_tardio'] - c.loc[X, 'fin_mas_temprano'])\n"
                           "enlace_critico = (c.loc[W, 'H_total'] == 0 and c.loc[Z, 'H_total'] == 0\n"
                           "                  and c.loc[W, 'fin_mas_temprano'] == c.loc[Z, 'inicio_mas_temprano'])\n"
                           "_e = (f'- **Sólo la espera**: la prelación {X} → {Y} tenía un margen LS({Y}) − EF({X}) = '\n"
                           "      f'{int(c.loc[Y, \"inicio_mas_tardio\"])} − {int(c.loc[X, \"fin_mas_temprano\"])} = {plural(margen, \"semana\")}; '\n"
                           "      f'una espera de {plural(espera, \"semana\")} alarga el proyecto máx(0, {espera} − {margen}) = '\n"
                           "      f'{max(0, espera - margen)}: de {duracion_proyecto} a **{T_espera}** semanas.')\n"
                           "if T_solape < duracion_proyecto:\n"
                           "    _por = (f'la prelación {W} → {Z} está en el camino crítico, así que el proyecto se acorta '\n"
                           "            f'{plural(duracion_proyecto - T_solape, \"semana\")} (hasta que otra ruta pasa a ser la más larga)')\n"
                           "elif enlace_critico:\n"
                           "    _por = f'aunque la prelación {W} → {Z} es crítica, hay otra ruta crítica que no pasa por ella'\n"
                           "else:\n"
                           "    _por = f'la prelación {W} → {Z} no está en el camino crítico'\n"
                           "_s = (f'- **Sólo el solape**: {Z} puede empezar {plural(int(c.loc[W, \"duracion\"]) - mitad, \"semana\")} antes '\n"
                           "      f'de que termine {W} (tras {W}1, de {mitad}); {_por}: el proyecto dura **{T_solape}** semanas.')\n"
                           "Markdown(_e + '\\n' + _s + f'\\n\\n**Respuesta:** con la espera y el solape a la vez el proyecto dura '\n"
                           "         f'**{plural(duracion_mod, \"semana\")}**, ' + ('las mismas que antes.' if duracion_mod == duracion_proyecto else\n"
                           "         f'{plural(abs(duracion_mod - duracion_proyecto), \"semana\")} {\"menos\" if duracion_mod < duracion_proyecto else \"más\"} que las {duracion_proyecto} iniciales.'))"))
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
                       "proyecto.display_roy_construccion(tabla=prelaciones)"))
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
                       "Markdown(f'**Respuesta:** la duración del proyecto es el mayor fin más temprano, {duracion_proyecto} días. '\n"
                       "         + ('El cálculo coincide con el calendario de la librería.' if coincide else 'El cálculo NO coincide con el de la librería.'))"))
    celdas += apartado(3)
    celdas.append(code("dominantes = {}\n"
                       "for a in datos_proyecto.index:\n"
                       "    ps = sorted(predecesoras[a])\n"
                       "    if len(ps) >= 2:\n"
                       "        maximo = max(EF[p] for p in ps)\n"
                       "        dominantes[a] = ', '.join(p for p in ps if EF[p] == maximo)\n"
                       "pd.Series(dominantes).to_frame('predecesora dominante')"))
    celdas.append(code(f"objetivo, dominante = '{objetivo}', '{dominante}'\n"
                       "otras = [p for p in sorted(predecesoras[objetivo]) if p != dominante]\n"
                       "margen = {p: EF[dominante] - EF[p] for p in otras}\n"
                       "_fines = ', '.join(f'EF({p}) = {EF[p]}' for p in sorted(predecesoras[objetivo]))\n"
                       "_margenes = lista_y([f'{p} termina {plural(m, \"día\")} antes' for p, m in margen.items()])\n"
                       "Markdown(f'**Respuesta:** la predecesora dominante de {objetivo} es **{dominante}**: de sus predecesoras '\n"
                       "         f'({_fines}) es la que termina más tarde y fija ES({objetivo}) = {ES[objetivo]}; {_margenes}.')"))
    celdas += apartado(4)
    celdas.append(code("d2 = duracion.copy(); d2[dominante] -= 1\n"
                       "ES_nuevo = int(proyecto.calendar(d2).loc[objetivo, 'inicio_mas_temprano'])\n"
                       "_segunda = max(EF[p] for p in otras)\n"
                       "Markdown(f'**Respuesta:** hay que acortar {dominante} (o alguna de sus antecesoras en la cadena que la hace '\n"
                       "         f'llegar la última). Acortándola un día, EF({dominante}) pasa de {EF[dominante]} a {EF[dominante] - 1} '\n"
                       "         f'y {objetivo} empieza en máx({EF[dominante] - 1}, {_segunda}) = {ES_nuevo} en vez de {ES[objetivo]}. '\n"
                       "         f'Acortar {lista_y(otras)} no sirve, porque {dominante} seguiría llegando la última ({_margenes} que ella): '\n"
                       "         f'sólo aumentaría su margen. La dominante sólo cambia si {dominante} se acorta más de '\n"
                       "         f'{plural(EF[dominante] - _segunda, \"día\")} (lo que le saca la siguiente predecesora que termina más tarde).')"))
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
                       "_ini, _fin = sorted(prelaciones.start_activities), sorted(prelaciones.end_activities)\n"
                       "_v = lambda l, s, p: s if len(l) == 1 else p\n"
                       "Markdown(f'**Respuesta:** {_v(_ini, \"es inicial\", \"son iniciales\")} {lista_y(_ini)}: en el cuadro de sucesoras no '\n"
                       "         f'{_v(_ini, \"aparece\", \"aparecen\")} en la casilla de ninguna actividad, y en el de predecesoras '\n"
                       "         f'{_v(_ini, \"tiene\", \"tienen\")} `---`. {_v(_fin, \"Es final\", \"Son finales\")} {lista_y(_fin)}: en el cuadro de '\n"
                       "         f'sucesoras {_v(_fin, \"tiene\", \"tienen\")} `---`, y en el de predecesoras no {_v(_fin, \"figura\", \"figuran\")} '\n"
                       "         f'como predecesora de ninguna actividad.')"))
    celdas += apartado(2)
    celdas.append(code('#| fig-cap: "Grafo Roy"\n#| label: fig-roy\n'
                       "proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT", label="fig-pert", tipo="pert")
    celdas.append(code(_AUX_FICTICIAS))
    celdas.append(code("_lineas = explica_ficticias(proyecto, prelaciones)\n"
                       "Markdown(f'**Respuesta:** el Roy no necesita ficticias (cada prelación es un arco); el PERT necesita '\n"
                       "         f'**{plural(len(_lineas), \"ficticia\")}**' + (':\\n\\n' + '\\n'.join(_lineas) if _lineas else '.'))"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas += _fechas_y_critico('en las figuras del apartado 2')
    celdas += apartado(4)
    celdas.append(code("sucesoras = {a: sorted(b for b, ps in prelaciones.immediate_predecessor.items() if a in ps) for a in datos_proyecto.index}\n"
                       "reconstruido = pd.Series({a: (','.join(s) if s else '---') for a, s in sucesoras.items()}, name='successor')\n"
                       "coincide = (reconstruido == datos_enunciado['successor'].map(lambda s: ','.join(sorted(s.split(','))) if s != '---' else s)).all()\n"
                       "display(pd.concat({'enunciado': datos_enunciado['successor'], 'reconstruido': reconstruido}, axis=1))\n"
                       "Markdown(('Los dos cuadros coinciden' if coincide else 'Los cuadros NO coinciden') + ': son la misma relación '\n"
                       "         '(«A precede a B») leída en los dos sentidos, por filas en uno y por columnas en el otro.')"))
    return celdas, f"{n} actividades; cuadro de sucesoras, {info['n_dummies']} ficticias en PERT"


VARIANTES = [
    Variante("roy-vs-pert", 1, "grafo Roy, PERT equivalente, comparación de nodos y arcos, calendario y camino crítico",
             lambda rng: _base(1, rng), grafo="roy"),
    Variante("roy-vs-pert-b", 1, "ídem con otro proyecto", lambda rng: _base(2, rng), grafo="roy"),
    Variante("esperas-y-solapes", 3, "espera entre dos actividades y solape comienzo-comienzo modelados con actividades auxiliares",
             lambda rng: _base(3, rng), grafo="roy"),
    Variante("predecesora-dominante", 2, "cálculo hacia delante y hacia atrás a mano sobre Roy; qué predecesora fija cada inicio",
             _predecesora_dominante, grafo="roy"),
    Variante("desde-sucesoras", 1, "el cuadro da sucesoras en lugar de predecesoras: invertirlo, Roy, PERT y calendario",
             _desde_sucesoras, grafo="roy"),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
