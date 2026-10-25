"""Colección 15 — Casos avanzados: relaciones PDM (FC, CC, FF, CF) con desfases, fechas de
calendario reales con festivos y decisión económica completa (indirectos, penalización y prima)."""
import contextlib
import io

import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="15_avanzado", prefijo="Avanzado",
            titulo="Casos avanzados: PDM, calendario real y decisión económica",
            descripcion="Relaciones de precedencia generalizadas (fin-comienzo, comienzo-comienzo, fin-fin, "
                        "comienzo-fin) con desfases, traducción del calendario en días laborables a fechas "
                        "reales con festivos, y elección de la duración que minimiza el coste total con "
                        "indirectos, penalización y prima.")

TIPOS = {"FC": "fin-comienzo", "CC": "comienzo-comienzo", "FF": "fin-fin", "CF": "comienzo-fin"}

PDM = (
    "def calendario_pdm(datos, relaciones):\n"
    "    '''Pasos hacia delante y hacia atrás con relaciones FC/CC/FF/CF y desfases (sin interrumpir actividades).'''\n"
    "    d = datos['duration'].astype(int).to_dict()\n"
    "    G = nx.DiGraph(); G.add_nodes_from(d)\n"
    "    for _, r in relaciones.iterrows():\n"
    "        G.add_edge(r['predecessor'], r['successor'], tipo=r['type'], lag=int(r['lag']))\n"
    "    orden = list(nx.topological_sort(G))\n"
    "    ES, EF, como = {}, {}, {}\n"
    "    for a in orden:\n"
    "        cands = {'inicio del proyecto': 0}\n"
    "        for p, _, e in G.in_edges(a, data=True):\n"
    "            tipo, lag = e['tipo'], e['lag']\n"
    "            if tipo == 'FC': v, f = EF[p] + lag, f'EF({p}) + {lag}'\n"
    "            elif tipo == 'CC': v, f = ES[p] + lag, f'ES({p}) + {lag}'\n"
    "            elif tipo == 'FF': v, f = EF[p] + lag - d[a], f'EF({p}) + {lag} − d({a})'\n"
    "            else: v, f = ES[p] + lag - d[a], f'ES({p}) + {lag} − d({a})'\n"
    "            cands[f'{tipo} desde {p}: {f} = {v}'] = v\n"
    "        ES[a] = max(cands.values()); EF[a] = ES[a] + d[a]\n"
    "        como[a] = '; '.join(k for k in cands if k != 'inicio del proyecto') or 'inicial'\n"
    "    T = max(EF.values())\n"
    "    LS, LF, como_t = {}, {}, {}\n"
    "    for a in reversed(orden):\n"
    "        cands = {'fin del proyecto': T}\n"
    "        for _, s, e in G.out_edges(a, data=True):\n"
    "            tipo, lag = e['tipo'], e['lag']\n"
    "            if tipo == 'FC': v, f = LS[s] - lag, f'LS({s}) − {lag}'\n"
    "            elif tipo == 'CC': v, f = LS[s] - lag + d[a], f'LS({s}) − {lag} + d({a})'\n"
    "            elif tipo == 'FF': v, f = LF[s] - lag, f'LF({s}) − {lag}'\n"
    "            else: v, f = LF[s] - lag + d[a], f'LF({s}) − {lag} + d({a})'\n"
    "            cands[f'{tipo} hacia {s}: {f} = {v}'] = v\n"
    "        LF[a] = min(cands.values()); LS[a] = LF[a] - d[a]\n"
    "        como_t[a] = '; '.join(k for k in cands if k != 'fin del proyecto') or 'final'\n"
    "    cal = pd.DataFrame({'ES': ES, 'EF': EF, 'LS': LS, 'LF': LF}).loc[list(d)]\n"
    "    cal['H_total'] = cal['LS'] - cal['ES']\n"
    "    detalle = pd.DataFrame({'ES = máx(…)': como, 'LF = mín(…)': como_t}).loc[list(d)]\n"
    "    return cal, detalle, T\n")


def _pdm(rng):
    """Relaciones FC/CC/FF/CF con desfases."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and 4 <= i["n_rutas"] <= 7 and (i["D"] >= 3).all(),
                    n=n, n_inicio=2, n_fin=int(rng.choice([1, 2])))
    D, preds = info["D"], info["preds"]
    pares = [(p, a) for a, ps in preds.items() for p in sorted(ps)]
    rel = pd.DataFrame(pares, columns=["predecessor", "successor"])
    rel["type"] = "FC"
    rel["lag"] = 0
    # elegir 3 relaciones no FC+0 distintas
    idx = rng.choice(len(rel), size=min(4, len(rel)), replace=False)
    tipos = ["CC", "FF", "FC", "CF"]
    for j, i in enumerate(idx):
        tipo = tipos[j % 4]
        rel.loc[i, "type"] = tipo
        if tipo == "CC":
            rel.loc[i, "lag"] = int(rng.integers(1, 3))
        elif tipo == "FF":
            rel.loc[i, "lag"] = int(rng.integers(1, 3))
        elif tipo == "FC":
            rel.loc[i, "lag"] = int(rng.integers(1, 4))      # espera (curado, secado)
        else:
            p, a = rel.loc[i, "predecessor"], rel.loc[i, "successor"]
            rel.loc[i, "lag"] = int(D[a]) + int(rng.integers(1, 3))   # CF: a no puede acabar antes de ES(p)+lag
    df = P.tabla_base(info, duration=D)
    n_cc = int((rel["type"] == "CC").sum()); n_ff = int((rel["type"] == "FF").sum())
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyas actividades y duraciones (días) se adjuntan. Sus prelaciones no son "
                 "todas del tipo habitual: el segundo cuadro las describe con el *método de diagramación por "
                 "precedencias* (PDM), donde cada relación tiene un tipo y un desfase (en días):\n\n"
                 "- **FC** (fin-comienzo) + d: la sucesora empieza al menos d días después de que termine la predecesora (FC + 0 es la prelación ordinaria);\n"
                 "- **CC** (comienzo-comienzo) + d: la sucesora empieza al menos d días después de que *empiece* la predecesora (solape);\n"
                 "- **FF** (fin-fin) + d: la sucesora termina al menos d días después de que termine la predecesora;\n"
                 "- **CF** (comienzo-fin) + d: la sucesora termina al menos d días después de que empiece la predecesora.\n\n"
                 "Las actividades no se interrumpen.")]
    celdas.append(preguntas([
        "Escriba, para cada tipo de relación, la condición que impone sobre el inicio más temprano de la "
        "sucesora (ES) en función de ES, EF y la duración. Calcule el paso hacia delante (ES, EF) en orden "
        "topológico, anotando los candidatos de cada actividad.",
        "Escriba las condiciones del paso hacia atrás (LS, LF) para cada tipo y calcúlelo. Obtenga las holguras "
        "totales y la duración del proyecto.",
        "¿Cuál es el camino crítico? ¿Hay alguna actividad crítica que no esté *entera* en él, es decir, cuya "
        "criticidad venga de un solape (CC o FF)? ¿Qué pasaría con el plazo si esa actividad se alargara?",
        "Sustituya todas las relaciones por FC + 0 (las prelaciones clásicas) y calcule la duración con la "
        "librería. ¿Por qué difiere? ¿Qué relaciones acortan y cuáles alargan el proyecto?"]))
    celdas.append(code('#| tbl-cap: "Actividades y duraciones"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    rel_lit = rel.copy()
    rel_lit.index = range(1, len(rel_lit) + 1)
    rel_lit.index.name = "id"
    celdas.append(code('#| tbl-cap: "Relaciones PDM"\n#| label: tbl-relaciones\n'
                       + literal_dataframe(rel_lit, nombre="relaciones", indice="id")))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()"))
    celdas += apartado(1)
    celdas.append(md("Condiciones sobre la sucesora $a$ con predecesora $p$ y desfase $\\ell$:\n\n"
                     "| tipo | condición | sobre ES(a) |\n|---|---|---|\n"
                     "| FC | $ES_a \\ge EF_p + \\ell$ | $EF_p + \\ell$ |\n"
                     "| CC | $ES_a \\ge ES_p + \\ell$ | $ES_p + \\ell$ |\n"
                     "| FF | $EF_a \\ge EF_p + \\ell$ | $EF_p + \\ell - d_a$ |\n"
                     "| CF | $EF_a \\ge ES_p + \\ell$ | $ES_p + \\ell - d_a$ |\n\n"
                     "ES es el máximo de los candidatos (y 0 si no hay ninguno); EF = ES + d."))
    celdas.append(code(PDM))
    celdas.append(code('#| tbl-cap: "Calendario PDM"\n#| label: tbl-calendarioPdm\n'
                       "calendario, detalle, duracion_pdm = calendario_pdm(datos_proyecto, relaciones)\n"
                       "calendario"))
    celdas.append(code('#| tbl-cap: "Candidatos considerados en cada actividad"\n#| label: tbl-candidatos\n'
                       "detalle"))
    celdas += apartado(2)
    celdas.append(md("Hacia atrás, cada relación acota la predecesora: FC: $LF_p \\le LS_a - \\ell$; CC: "
                     "$LS_p \\le LS_a - \\ell$ (luego $LF_p \\le LS_a - \\ell + d_p$); FF: $LF_p \\le LF_a - \\ell$; "
                     "CF: $LS_p \\le LF_a - \\ell$. LF es el mínimo (la duración del proyecto si no hay sucesoras) "
                     "y LS = LF − d. La holgura total es LS − ES."))
    celdas.append(code("Markdown(f'La duración del proyecto es {duracion_pdm} días. Actividades críticas (holgura 0): {\", \".join(calendario.index[calendario[\"H_total\"] == 0])}.')"))
    celdas += apartado(3)
    celdas.append(code("criticas = calendario.index[calendario['H_total'] == 0].tolist()\n"
                       "rel_crit = relaciones[relaciones['predecessor'].isin(criticas) & relaciones['successor'].isin(criticas)].copy()\n"
                       "# una relación es 'activa' si se cumple con igualdad en el calendario temprano\n"
                       "def activa(r):\n"
                       "    p, s, t, l = r['predecessor'], r['successor'], r['type'], int(r['lag'])\n"
                       "    ES, EF = calendario['ES'], calendario['EF']\n"
                       "    return {'FC': ES[s] == EF[p] + l, 'CC': ES[s] == ES[p] + l, 'FF': EF[s] == EF[p] + l, 'CF': EF[s] == ES[p] + l}[t]\n"
                       "rel_crit['activa'] = rel_crit.apply(activa, axis=1)\n"
                       "display(rel_crit)\n"
                       "solapes = rel_crit[rel_crit['activa'] & rel_crit['type'].isin(['CC', 'FF', 'CF'])]\n"
                       "Markdown('Camino crítico: la cadena de relaciones activas entre actividades críticas. ' + (\n"
                       "    f'Las relaciones {\", \".join(f\"{r.predecessor}→{r.successor} ({r.type})\" for r in solapes.itertuples())} no son fin-comienzo: en ellas sólo es crítico el extremo que la relación une. '\n"
                       "    'Con CC, el *inicio* de la predecesora (alargarla no retrasa el proyecto, salvo que otra relación lo impida); con FF, su *fin* (acortarla no adelanta nada); '\n"
                       "    'con CF, el inicio de la predecesora y el fin de la sucesora. Es la paradoja del PDM: una actividad crítica puede alargarse sin retrasar el plazo.'\n"
                       "    if len(solapes) else 'Todas las relaciones activas del camino crítico son FC: el camino crítico se comporta como en el PERT clásico.'))"))
    celdas += apartado(4)
    celdas.append(code("tabla_fc = datos_proyecto.copy()\n"
                       "tabla_fc['predecessor'] = [','.join(sorted(relaciones.loc[relaciones['successor'] == a, 'predecessor'])) or '---' for a in tabla_fc.index]\n"
                       "proyecto_fc = PredecessorTable.from_dataframe_of_strings(tabla_fc, activity='activity', predecessor='predecessor').create_project()\n"
                       "duracion_fc = proyecto_fc.duration(tabla_fc['duration'])\n"
                       "calendario_fc = proyecto_fc.calendar(tabla_fc['duration'])\n"
                       "comparacion = pd.DataFrame({'ES (PDM)': calendario['ES'], 'ES (todo FC+0)': calendario_fc['inicio_mas_temprano'], 'H_total (PDM)': calendario['H_total'], 'H_total (todo FC+0)': calendario_fc['H_total']})\n"
                       "display(comparacion)\n"
                       "Markdown(f'''Con todas las relaciones FC + 0 el proyecto dura {duracion_fc} días frente a {duracion_pdm} con PDM. "
                       "Las relaciones CC y FF permiten solapar actividades (acortan); los desfases positivos en FC son esperas obligadas (alargan); "
                       "una CF con desfase grande obliga a que la sucesora no termine demasiado pronto, lo que puede retrasarla respecto al FC clásico. "
                       "El efecto neto depende de cuáles de ellas están en el camino crítico.''')"))
    celdas += fig_grafo(durations="tabla_fc['duration']", cap="Grafo PERT equivalente con todas las relaciones FC + 0", label="fig-pertFc", obj="proyecto_fc")
    return celdas, f"{n} actividades; {len(idx)} relaciones PDM no ordinarias ({n_cc} CC, {n_ff} FF)"


def _fechas(rng):
    """Del calendario en días laborables a fechas reales con fines de semana y festivos."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and 15 <= i["T"] <= 30 and (i["calendario"]["H_total"] > 0).sum() >= 2,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, T = info["D"], info["calendario"], int(info["T"])
    df = P.tabla_base(info, duration=D)
    # fecha de inicio: un lunes de 2026
    lunes = np.datetime64("2026-01-05") + np.timedelta64(7 * int(rng.integers(0, 40)), "D")
    inicio = str(lunes)
    # festivos: dos días laborables dentro del horizonte (en días naturales ≈ T*7/5)
    horizonte = int(T * 7 / 5) + 10
    cand = [str(lunes + np.timedelta64(k, "D")) for k in range(3, horizonte) if np.is_busday(lunes + np.timedelta64(k, "D"))]
    festivos = sorted(rng.choice(cand, size=2, replace=False).tolist())
    extra_cand = [c for c in cand if c not in festivos]
    festivo_extra = extra_cand[int(rng.integers(0, len(extra_cand)))]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en **días laborables**) se "
                 f"adjunta. El proyecto empieza el lunes {inicio}. Se trabaja de lunes a viernes; son festivos "
                 f"los días {festivos[0]} y {festivos[1]}.")]
    celdas.append(preguntas([
        "Calcule el calendario del proyecto en días laborables (inicio y fin más tempranos y más tardíos, "
        "holgura total) y su duración.",
        "Traduzca el calendario a fechas: fecha de inicio y de fin (último día trabajado) más tempranas y más "
        "tardías de cada actividad. ¿En qué fecha termina el proyecto? ¿Cuántos días naturales dura?",
        "¿Qué actividades críticas atraviesan un festivo o un fin de semana? ¿Cambia eso su duración en días "
        "laborables? ¿Y la fecha de fin del proyecto?",
        f"Se declara un nuevo festivo el {festivo_extra}. ¿Qué fechas cambian y cuánto se retrasa la "
        "entrega? ¿Depende de que el festivo caiga en una actividad crítica?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"inicio = np.datetime64('{inicio}')\n"
                       f"festivos = [np.datetime64('{festivos[0]}'), np.datetime64('{festivos[1]}')]"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos de los nodos y camino crítico",
                        label="fig-grafo")
    celdas.append(code('#| tbl-cap: "Calendario en días laborables"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas += apartado(2)
    celdas.append(md("El día laborable $k$ (contando desde 0) es `busday_offset(inicio, k)`, que salta fines de "
                     "semana y festivos. Una actividad que empieza en el laborable ES y dura d ocupa los "
                     "laborables ES, …, ES + d − 1: su último día trabajado es `busday_offset(inicio, EF − 1)`."))
    celdas.append(code("def fecha(k):\n"
                       "    return np.busday_offset(inicio, int(k), roll='forward', holidays=festivos)\n"
                       "fechas = pd.DataFrame({\n"
                       "    'inicio temprano': [fecha(k) for k in calendario['inicio_mas_temprano']],\n"
                       "    'fin temprano': [fecha(k - 1) for k in calendario['fin_mas_temprano']],\n"
                       "    'inicio tardío': [fecha(k) for k in calendario['inicio_mas_tardio']],\n"
                       "    'fin tardío': [fecha(k - 1) for k in calendario['fin_mas_tardio']]}, index=calendario.index)\n"
                       "fechas"))
    celdas.append(code("DIAS = {'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miércoles', 'Thursday': 'jueves', 'Friday': 'viernes', 'Saturday': 'sábado', 'Sunday': 'domingo'}\n"
                       "fin_proyecto = fecha(duracion - 1)\n"
                       "dias_naturales = int((fin_proyecto - inicio).astype(int)) + 1\n"
                       "Markdown(f'El proyecto termina el {fin_proyecto} ({DIAS[pd.Timestamp(fin_proyecto).day_name()]}): {duracion} días laborables son {dias_naturales} días naturales, '\n"
                       "         f'{dias_naturales - duracion} de ellos no trabajados ({int(np.busday_count(inicio, fin_proyecto + np.timedelta64(1, \"D\"), weekmask=\"Sat Sun\"))} de fin de semana y {len([f for f in festivos if inicio <= f <= fin_proyecto])} festivos).')"))
    celdas += apartado(3)
    celdas.append(code("def dias_no_laborables(ini, fin):\n"
                       "    todos = np.arange(ini, fin + np.timedelta64(1, 'D'), dtype='datetime64[D]')\n"
                       "    return int(sum(1 for x in todos if not np.is_busday(x, holidays=festivos)))\n"
                       "criticas = calendario.index[calendario['H_total'] == 0]\n"
                       "cruce = pd.DataFrame({'inicio': fechas.loc[criticas, 'inicio temprano'], 'fin': fechas.loc[criticas, 'fin temprano'],\n"
                       "                      'días no laborables que atraviesa': [dias_no_laborables(fechas.loc[a, 'inicio temprano'], fechas.loc[a, 'fin temprano']) for a in criticas],\n"
                       "                      'incluye festivo': [any(fechas.loc[a, 'inicio temprano'] <= f <= fechas.loc[a, 'fin temprano'] for f in festivos) for a in criticas]})\n"
                       "display(cruce)\n"
                       "Markdown('La duración en días laborables no cambia: los festivos y fines de semana sólo estiran la actividad en el calendario natural. '\n"
                       "         'La fecha de fin del proyecto sí depende de cuántos días no laborables caen dentro del horizonte, estén donde estén.')"))
    celdas += apartado(4)
    celdas.append(code(f"festivo_extra = np.datetime64('{festivo_extra}')\n"
                       "festivos2 = sorted(festivos + [festivo_extra])\n"
                       "def fecha2(k):\n"
                       "    return np.busday_offset(inicio, int(k), roll='forward', holidays=festivos2)\n"
                       "fechas2 = pd.DataFrame({'fin temprano (antes)': fechas['fin temprano'], 'fin temprano (con el nuevo festivo)': [fecha2(k - 1) for k in calendario['fin_mas_temprano']]})\n"
                       "fechas2['cambia'] = fechas2.iloc[:, 0] != fechas2.iloc[:, 1]\n"
                       "display(fechas2)\n"
                       "fin2 = fecha2(duracion - 1)\n"
                       "Markdown(f'''El nuevo festivo desplaza un día natural todas las fechas posteriores a él ({\", \".join(fechas2.index[fechas2[\"cambia\"]])}), críticas o no: "
                       "la entrega pasa del {fin_proyecto} al {fin2}. No depende de la criticidad, porque el festivo no consume días laborables de ninguna actividad: "
                       "desplaza el calendario entero. Sólo si el festivo cayera después del fin del proyecto no afectaría.''')"))
    return celdas, f"{n} actividades; inicio {inicio}, {len(festivos)} festivos"


def _decision_economica(rng):
    """Duración óptima con costes directos (Ackoff), indirectos, penalización por retraso y prima por adelanto."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 14 and (i["D"] >= 2).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr_ack = ucr.replace(0.0, np.inf)
        if int((i["D"] - tope).sum()) < 5:
            return False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tabla, best, dur, avail = i["proyecto"].ackoff(i["D"], tope, ucr_ack, reduction=100, report=False)
        except Exception:
            return False
        pasos = [(acts, float(sum(ucr_ack[a] for a in acts))) for _, acts in sorted(best.items())]
        if len(pasos) < 4:
            return False
        costes = np.array([c for _, c in pasos])
        indirecto = int(np.median(costes)) + 1
        plazo = int(i["T"]) - 2
        penal = int(max(costes)) + 2
        prima = max(1, int(min(costes)) - 1)
        estado.update(tope=tope, cn=cn, ucr=ucr_ack, pasos=pasos, indirecto=indirecto, plazo=plazo, penal=penal, prima=prima)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, cn, ucr, pasos = estado["tope"], estado["cn"], estado["ucr"], estado["pasos"]
    indirecto, plazo, penal, prima = estado["indirecto"], estado["plazo"], estado["penal"], estado["prima"]
    df = P.tabla_base(info, duration=D, cap_duration=tope, cost_normal=cn, ucr=ucr)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas), coste "
                 "normal (k€) y coste unitario de reducción (k€/semana) se adjunta. Además:\n\n"
                 f"- los costes indirectos (oficina, alquileres, supervisión) son de {indirecto} k€ por semana de duración;\n"
                 f"- el contrato fija la entrega en la semana {plazo}, con una penalización de {penal} k€ por cada semana de retraso;\n"
                 f"- y una prima de {prima} k€ por cada semana de adelanto sobre ese plazo.")]
    celdas.append(preguntas([
        "Calcule la duración normal del proyecto y su coste total con esa duración (directo + indirecto + "
        "penalización o prima).",
        "Obtenga con Ackoff-Sasieni el sobrecoste directo acumulado de cada semana de reducción hasta la "
        "duración tope.",
        "Construya la tabla de coste total frente a duración, con las cuatro componentes, y determine la "
        "duración óptima. Dibuje las curvas.",
        "Analice la decisión paso a paso: ¿qué se compara en cada semana de reducción antes y después del "
        "plazo contractual? ¿A partir de qué prima por semana de adelanto convendría reducir una semana más "
        "que en el óptimo?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"coste_indirecto_semanal, plazo_contrato, penalizacion_semanal, prima_semanal = {indirecto}, {plazo}, {penal}, {prima}"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales y el camino crítico",
                        label="fig-grafo")
    celdas.append(code("duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                       "coste_directo_normal = float(datos_proyecto['cost_normal'].sum())\n"
                       "def coste_plazo(T):\n"
                       "    return penalizacion_semanal * max(0, T - plazo_contrato) - prima_semanal * max(0, plazo_contrato - T)\n"
                       "total_normal = coste_directo_normal + coste_indirecto_semanal * duracion_normal + coste_plazo(duracion_normal)\n"
                       "Markdown(f'''Duración normal {duracion_normal} semanas. Coste total = directo {coste_directo_normal:.0f} + indirecto {coste_indirecto_semanal}·{duracion_normal} = {coste_indirecto_semanal * duracion_normal} "
                       "+ penalización {penalizacion_semanal}·({duracion_normal} − {plazo_contrato}) = {coste_plazo(duracion_normal):.0f} → **{total_normal:.0f} k€**.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni hasta la duración tope"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_tope, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=100, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],\n"
                       "                        'sobrecoste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},\n"
                       "                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))\n"
                       "resumen['duración'] = duracion_normal - resumen.index\n"
                       "resumen['sobrecoste acumulado'] = resumen['sobrecoste del paso'].cumsum()\n"
                       "resumen"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Coste total según la duración"\n#| label: tbl-costeTotal\n'
                       "curva = pd.DataFrame({'duración': [duracion_normal] + resumen['duración'].tolist(),\n"
                       "                      'coste directo': [coste_directo_normal] + (coste_directo_normal + resumen['sobrecoste acumulado']).tolist()}).set_index('duración')\n"
                       "curva['coste indirecto'] = coste_indirecto_semanal * curva.index\n"
                       "curva['penalización'] = [penalizacion_semanal * max(0, T - plazo_contrato) for T in curva.index]\n"
                       "curva['prima (negativa)'] = [-prima_semanal * max(0, plazo_contrato - T) for T in curva.index]\n"
                       "curva['coste total'] = curva.sum(axis=1)\n"
                       "curva"))
    celdas.append(code('#| fig-cap: "Componentes del coste frente a la duración"\n#| label: fig-curvas\n'
                       "ax = curva.plot(marker='o', figsize=(8, 4.5)); ax.axvline(plazo_contrato, color='grey', linestyle=':', label='plazo contractual')\n"
                       "ax.set_ylabel('k€'); ax.set_xlabel('duración (semanas)'); ax.invert_xaxis(); ax.legend(); ax.grid(alpha=0.3);"))
    celdas.append(code("duracion_optima = int(curva['coste total'].idxmin())\n"
                       "Markdown(f'La duración óptima es {duracion_optima} semanas, con un coste total de {curva[\"coste total\"].min():.0f} k€ (frente a {total_normal:.0f} sin reducir): {\"antes del plazo\" if duracion_optima < plazo_contrato else \"en el plazo\" if duracion_optima == plazo_contrato else \"con retraso\"}.')"))
    celdas += apartado(4)
    celdas.append(md("Cada semana de reducción cuesta el sobrecoste del paso de Ackoff-Sasieni y ahorra el coste "
                     "indirecto semanal **más** la penalización semanal mientras el proyecto esté por detrás del "
                     "plazo, o **más** la prima mientras esté por delante. La reducción compensa mientras el "
                     "sobrecoste del paso sea menor que ese ahorro; como la penalización suele ser mayor que la "
                     "prima, el ahorro marginal cae de golpe al cruzar el plazo y ahí suele detenerse la "
                     "reducción."))
    celdas.append(code("decision = resumen[['sobrecoste del paso', 'duración']].copy()\n"
                       "decision['ahorro por la semana'] = [coste_indirecto_semanal + (penalizacion_semanal if T + 1 > plazo_contrato else prima_semanal) for T in decision['duración']]\n"
                       "decision['conviene'] = decision['sobrecoste del paso'] < decision['ahorro por la semana']\n"
                       "display(decision)\n"
                       "paso_siguiente = resumen[resumen['duración'] == duracion_optima - 1]\n"
                       "if len(paso_siguiente):\n"
                       "    sobre = float(paso_siguiente['sobrecoste del paso'].iloc[0])\n"
                       "    prima_necesaria = sobre - coste_indirecto_semanal\n"
                       "    display(Markdown(f'Reducir una semana más (a {duracion_optima - 1}) costaría {sobre:.0f} k€ y ahorraría {coste_indirecto_semanal} de indirectos más la prima: '\n"
                       "                     f'compensaría con una prima de más de {max(prima_necesaria, 0):.0f} k€/semana (ahora es {prima_semanal}).'))\n"
                       "else:\n"
                       "    display(Markdown('El óptimo ya está en la duración tope: ninguna prima permitiría reducir más.'))"))
    return celdas, f"{n} actividades; indirectos {indirecto}, plazo {plazo}, penalización {penal}, prima {prima}"


VARIANTES = [
    Variante("pdm-desfases", 3, "relaciones FC/CC/FF/CF con desfases: pasos adelante y atrás con candidatos, criticidad por solape, comparación con FC+0",
             _pdm, grafo="roy"),
    Variante("fechas-calendario", 2, "del calendario en días laborables a fechas reales: fines de semana, festivos y efecto de un festivo nuevo",
             _fechas),
    Variante("decision-economica", 2, "duración óptima con costes directos (Ackoff), indirectos, penalización y prima; análisis marginal",
             _decision_economica),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
