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

PDM = r"""
def calendario_pdm(datos, relaciones):
    '''Pasos hacia delante y hacia atrás con relaciones FC/CC/FF/CF y desfases (sin interrumpir actividades).'''
    d = datos['duration'].astype(int).to_dict()
    G = nx.DiGraph(); G.add_nodes_from(d)
    for _, r in relaciones.iterrows():
        G.add_edge(r['predecessor'], r['successor'], tipo=r['type'], lag=int(r['lag']))
    orden = list(nx.topological_sort(G))
    ES, EF, como = {}, {}, {}
    for a in orden:
        cands = {}
        for p, _, e in G.in_edges(a, data=True):
            tipo, lag = e['tipo'], e['lag']
            if tipo == 'FC': v, f = EF[p] + lag, f'EF({p}) + {lag} = {EF[p]} + {lag}'
            elif tipo == 'CC': v, f = ES[p] + lag, f'ES({p}) + {lag} = {ES[p]} + {lag}'
            elif tipo == 'FF': v, f = EF[p] + lag - d[a], f'EF({p}) + {lag} − d({a}) = {EF[p]} + {lag} − {d[a]}'
            else: v, f = ES[p] + lag - d[a], f'ES({p}) + {lag} − d({a}) = {ES[p]} + {lag} − {d[a]}'
            cands[f'{tipo}+{lag} desde {p}: {f} = {v}'] = v
        ES[a] = max([0] + list(cands.values())); EF[a] = ES[a] + d[a]
        if not cands:
            como[a] = 'sin predecesoras: 0'
        else:
            como[a] = '; '.join(cands) + (' (negativo: no puede empezar antes del inicio, 0)' if max(cands.values()) < 0 else '')
    T = max(EF.values())
    LS, LF, como_t = {}, {}, {}
    for a in reversed(orden):
        cands = {}
        for _, s, e in G.out_edges(a, data=True):
            tipo, lag = e['tipo'], e['lag']
            if tipo == 'FC': v, f = LS[s] - lag, f'LS({s}) − {lag} = {LS[s]} − {lag}'
            elif tipo == 'CC': v, f = LS[s] - lag + d[a], f'LS({s}) − {lag} + d({a}) = {LS[s]} − {lag} + {d[a]}'
            elif tipo == 'FF': v, f = LF[s] - lag, f'LF({s}) − {lag} = {LF[s]} − {lag}'
            else: v, f = LF[s] - lag + d[a], f'LF({s}) − {lag} + d({a}) = {LF[s]} − {lag} + {d[a]}'
            cands[f'{tipo}+{lag} hacia {s}: {f} = {v}'] = v
        LF[a] = min([T] + list(cands.values())); LS[a] = LF[a] - d[a]
        if not cands:
            como_t[a] = f'sin sucesoras: T = {T}'
        else:
            como_t[a] = '; '.join(cands) + (f' (todos superan T = {T}: no puede acabar después del fin del proyecto)' if min(cands.values()) > T else '')
    cal = pd.DataFrame({'ES': ES, 'EF': EF, 'LS': LS, 'LF': LF}).loc[list(d)]
    cal['H_total'] = cal['LS'] - cal['ES']
    detalle = pd.DataFrame({'ES = máx(…)': como, 'LF = mín(…)': como_t}).loc[list(d)]
    return cal, detalle, T
"""


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
    celdas.append(code(PDM.strip('\n')))
    celdas.append(code("calendario, detalle, duracion_pdm = calendario_pdm(datos_proyecto, relaciones)\n"
                       "_d = datos_proyecto['duration'].astype(int)"))
    celdas.append(md("Paso hacia delante en orden topológico: para cada actividad, un candidato por cada relación "
                     "que llega a ella (tipo + desfase, con los números sustituidos); ES es el mayor y EF = ES + d."))
    celdas.append(code('#| tbl-cap: "Paso hacia delante: candidatos de ES de cada actividad"\n#| label: tbl-candidatos\n'
                       "pd.DataFrame({'candidatos de ES': detalle['ES = máx(…)'], 'ES = máx': calendario['ES'],\n"
                       "              'EF = ES + d': [f\"{calendario.loc[a, 'ES']} + {_d[a]} = {calendario.loc[a, 'EF']}\" for a in calendario.index]})"))
    celdas.append(code("_ult = calendario.index[calendario['EF'] == duracion_pdm].tolist()\n"
                       "Markdown(f'**Respuesta:** ES y EF son los de la tabla. La duración del proyecto es el mayor EF: T = máx(EF) = {duracion_pdm} días (EF de {lista_y(_ult)}).')"))
    celdas += apartado(2)
    celdas.append(md("Hacia atrás, cada relación acota la predecesora: FC: $LF_p \\le LS_a - \\ell$; CC: "
                     "$LS_p \\le LS_a - \\ell$ (luego $LF_p \\le LS_a - \\ell + d_p$); FF: $LF_p \\le LF_a - \\ell$; "
                     "CF: $LS_p \\le LF_a - \\ell$ (luego $LF_p \\le LF_a - \\ell + d_p$). Además ninguna actividad "
                     "puede acabar después del fin del proyecto (LF ≤ T). LF es el menor de los candidatos, "
                     "LS = LF − d y la holgura total es LS − ES."))
    celdas.append(code('#| tbl-cap: "Paso hacia atrás: candidatos de LF y holgura total"\n#| label: tbl-calendarioPdm\n'
                       "pd.DataFrame({'candidatos de LF': detalle['LF = mín(…)'], 'LF = mín': calendario['LF'],\n"
                       "              'LS = LF − d': [f\"{calendario.loc[a, 'LF']} − {_d[a]} = {calendario.loc[a, 'LS']}\" for a in calendario.index],\n"
                       "              'H_total = LS − ES': [f\"{calendario.loc[a, 'LS']} − {calendario.loc[a, 'ES']} = {calendario.loc[a, 'H_total']}\" for a in calendario.index]})"))
    celdas.append(code("criticas = calendario.index[calendario['H_total'] == 0].tolist()\n"
                       "Markdown(f'**Respuesta:** la duración del proyecto es {duracion_pdm} días; actividades críticas (holgura total 0): {lista_y(criticas)}; '\n"
                       "         + 'con holgura: ' + lista_y([f'{a} ({calendario.loc[a, \"H_total\"]})' for a in calendario.index if a not in criticas]) + '.')"))
    celdas += apartado(3)
    celdas.append(md("Una relación es *activa* si se cumple con igualdad en el calendario temprano (es la que "
                     "fija la fecha de la sucesora). El camino crítico es la cadena de relaciones activas entre "
                     "actividades críticas, desde el inicio hasta el fin del proyecto."))
    celdas.append(code("rel_crit = relaciones[relaciones['predecessor'].isin(criticas) & relaciones['successor'].isin(criticas)].copy()\n"
                       "def activa(r):\n"
                       "    p, s, t, l = r['predecessor'], r['successor'], r['type'], int(r['lag'])\n"
                       "    ES, EF = calendario['ES'], calendario['EF']\n"
                       "    return {'FC': ES[s] == EF[p] + l, 'CC': ES[s] == ES[p] + l, 'FF': EF[s] == EF[p] + l, 'CF': EF[s] == ES[p] + l}[t]\n"
                       "rel_crit['activa'] = rel_crit.apply(activa, axis=1).map({True: 'sí', False: 'no'})\n"
                       "display(rel_crit.rename(columns={'predecessor': 'predecesora', 'successor': 'sucesora', 'type': 'tipo', 'lag': 'desfase'}))\n"
                       "Ga = nx.DiGraph(); Ga.add_nodes_from(criticas)\n"
                       "Ga.add_edges_from((r.predecessor, r.successor, {'et': f'{r.type}+{r.lag}'}) for r in rel_crit.itertuples() if r.activa == 'sí')\n"
                       "_ini = [a for a in criticas if calendario.loc[a, 'ES'] == 0 and Ga.in_degree(a) == 0]\n"
                       "_fin = [a for a in criticas if Ga.out_degree(a) == 0]\n"
                       "cadenas = []\n"
                       "for i in _ini:\n"
                       "    for f in _fin:\n"
                       "        for camino in nx.all_simple_paths(Ga, i, f) if i != f else [[i]]:\n"
                       "            cadenas.append(camino[0] + ''.join(f' →({Ga.edges[u, v][\"et\"]}) {v}' for u, v in zip(camino, camino[1:])))\n"
                       "display(Markdown('**Camino crítico:** ' + '; '.join(cadenas) + '.'))"))
    celdas.append(md("Para saber qué pasa con el plazo si una actividad crítica se alarga (o se acorta), se repite "
                     "el cálculo con su duración cambiada en un día. Con relaciones FC el plazo se mueve lo mismo "
                     "que la actividad; con CC, FF o CF puede no moverse, o incluso moverse al revés."))
    celdas.append(code("def _T_con(a, delta):\n"
                       "    dd = datos_proyecto.copy(); dd.loc[a, 'duration'] = int(dd.loc[a, 'duration']) + delta\n"
                       "    return calendario_pdm(dd, relaciones)[2] if dd.loc[a, 'duration'] > 0 else None\n"
                       "sens = pd.DataFrame({'duración': [int(_d[a]) for a in criticas],\n"
                       "                     'T si se alarga 1 día': [_T_con(a, 1) for a in criticas],\n"
                       "                     'T si se acorta 1 día': [_T_con(a, -1) for a in criticas]}, index=criticas)\n"
                       "display(sens)\n"
                       "_no_retrasa = [a for a in criticas if sens.loc[a, 'T si se alarga 1 día'] <= duracion_pdm]\n"
                       "_al_reves = [a for a in criticas if sens.loc[a, 'T si se alarga 1 día'] < duracion_pdm]\n"
                       "_no_adelanta = [a for a in criticas if sens.loc[a, 'T si se acorta 1 día'] is not None and sens.loc[a, 'T si se acorta 1 día'] >= duracion_pdm]\n"
                       "_solapes = rel_crit[(rel_crit['activa'] == 'sí') & rel_crit['type'].isin(['CC', 'FF', 'CF'])]\n"
                       "_txt = ('Relaciones activas del camino crítico que no son fin-comienzo: ' + lista_y([f'{r.predecessor}→{r.successor} ({r.type}+{r.lag})' for r in _solapes.itertuples()]) + '. '\n"
                       "        if len(_solapes) else 'Todas las relaciones activas del camino crítico son FC. ')\n"
                       "if _no_retrasa:\n"
                       "    _txt += (f'**Respuesta:** {lista_y(_no_retrasa)} {\"es crítica\" if len(_no_retrasa) == 1 else \"son críticas\"} sin estar entera{\"s\" if len(_no_retrasa) > 1 else \"\"} en el camino crítico: '\n"
                       "             f'si se alarga{\"n\" if len(_no_retrasa) > 1 else \"\"} un día el plazo no se retrasa (sigue en {duracion_pdm} días o menos)'\n"
                       "             + (f'; alargar {lista_y(_al_reves)} incluso lo acorta, porque su relación CF/FF fija su *fin* y, al ser más larga, puede empezar antes' if _al_reves else '')\n"
                       "             + '. Su criticidad viene sólo de su inicio o sólo de su fin (la paradoja del PDM).')\n"
                       "else:\n"
                       "    _txt += f'**Respuesta:** todas las actividades críticas lo son enteras: alargar cualquiera de ellas un día retrasa el plazo a {duracion_pdm + 1} días.'\n"
                       "if _no_adelanta:\n"
                       "    _txt += f' Acortar {lista_y(_no_adelanta)} un día no adelanta el fin del proyecto.'\n"
                       "Markdown(_txt)"))
    celdas += apartado(4)
    celdas.append(code("tabla_fc = datos_proyecto.copy()\n"
                       "tabla_fc['predecessor'] = [','.join(sorted(relaciones.loc[relaciones['successor'] == a, 'predecessor'])) or '---' for a in tabla_fc.index]\n"
                       "proyecto_fc = PredecessorTable.from_dataframe_of_strings(tabla_fc, activity='activity', predecessor='predecessor').create_project()\n"
                       "duracion_fc = proyecto_fc.duration(tabla_fc['duration'])\n"
                       "calendario_fc = proyecto_fc.calendar(tabla_fc['duration'])\n"
                       "comparacion = pd.DataFrame({'ES (PDM)': calendario['ES'], 'ES (todo FC+0)': calendario_fc['inicio_mas_temprano'], 'H_total (PDM)': calendario['H_total'], 'H_total (todo FC+0)': calendario_fc['H_total']})\n"
                       "display(comparacion)\n"
                       "display(explica_camino_critico(proyecto_fc, tabla_fc['duration'], 'días'))"))
    celdas.append(md("Para ver qué relaciones acortan y cuáles alargan el proyecto, se cambia cada relación que no es "
                     "FC + 0, una a una, por FC + 0 (las demás se dejan como están) y se recalcula la duración PDM: si "
                     "con FC + 0 el proyecto dura más, esa relación lo acortaba; si dura menos, lo alargaba."))
    celdas.append(code("efectos = []\n"
                       "for i, r in relaciones.iterrows():\n"
                       "    if r['type'] == 'FC' and int(r['lag']) == 0:\n"
                       "        continue\n"
                       "    rel2 = relaciones.copy(); rel2.loc[i, 'type'] = 'FC'; rel2.loc[i, 'lag'] = 0\n"
                       "    T2 = calendario_pdm(datos_proyecto, rel2)[2]\n"
                       "    efectos.append({'relación': f\"{r['predecessor']}→{r['successor']} ({r['type']}+{r['lag']})\", 'T con esa relación como FC+0': T2,\n"
                       "                    'efecto de la relación sobre T': duracion_pdm - T2})\n"
                       "efectos = pd.DataFrame(efectos).set_index('relación')\n"
                       "display(efectos)\n"
                       "_acorta = [f'{k}: {plural(-v, \"día\")}' for k, v in efectos['efecto de la relación sobre T'].items() if v < 0]\n"
                       "_alarga = [f'{k}: {plural(v, \"día\")}' for k, v in efectos['efecto de la relación sobre T'].items() if v > 0]\n"
                       "_nada = [k for k, v in efectos['efecto de la relación sobre T'].items() if v == 0]\n"
                       "Markdown(f'**Respuesta:** con todas las relaciones FC + 0 el proyecto dura {duracion_fc} días frente a {duracion_pdm} con PDM '\n"
                       "         + (f'(PDM acorta {plural(duracion_fc - duracion_pdm, \"día\")}). ' if duracion_pdm < duracion_fc else f'(PDM alarga {plural(duracion_pdm - duracion_fc, \"día\")}). ' if duracion_pdm > duracion_fc else '(la misma duración). ')\n"
                       "         + 'Difiere porque CC, FF y CF permiten solapar la sucesora con la predecesora y los desfases añaden esperas. '\n"
                       "         + 'Acortan el proyecto: ' + ('; '.join(_acorta) or 'ninguna') + '. Lo alargan: ' + ('; '.join(_alarga) or 'ninguna') + '.' + (f' No influyen por separado (no fijan la fecha de ninguna actividad crítica, o hay otra relación que fija la misma fecha): {lista_y(_nada)}.' if _nada else '')\n"
                       "         + ' Los efectos de cada relación por separado no tienen por qué sumar la diferencia total: al cambiar varias a la vez cambia el camino crítico.')"))
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
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']].rename(columns={\n"
                       "    'inicio_mas_temprano': 'inicio más temprano (ES)', 'fin_mas_temprano': 'fin más temprano (EF)',\n"
                       "    'inicio_mas_tardio': 'inicio más tardío (LS)', 'fin_mas_tardio': 'fin más tardío (LF)', 'H_total': 'holgura total'})"))
    celdas.append(code("display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'días laborables'))\n"
                       "Markdown(f'**Respuesta:** el calendario es el de la tabla (en días laborables contados desde 0, el inicio del proyecto); '\n"
                       "         f'el proyecto dura {duracion} días laborables.')"))
    celdas += apartado(2)
    celdas.append(md("El calendario en días laborables cuenta los días trabajados desde 0. Para pasarlo a fechas se "
                     "numeran los días laborables reales a partir de la fecha de inicio, saltando fines de semana y "
                     "festivos: el día laborable $k$ es la fecha `busday_offset(inicio, k)`. Una actividad que "
                     "empieza en el laborable ES y dura d ocupa los laborables ES, …, ES + d − 1: su primer día "
                     "trabajado es la fecha del laborable ES y el último, la del laborable EF − 1. La tabla "
                     "siguiente da esa correspondencia semana a semana (F = festivo, — = fin de semana)."))
    celdas.append(code("def fecha(k):\n"
                       "    return np.busday_offset(inicio, int(k), roll='forward', holidays=festivos)\n"
                       "DIAS = {'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miércoles', 'Thursday': 'jueves', 'Friday': 'viernes', 'Saturday': 'sábado', 'Sunday': 'domingo'}\n"
                       "fin_proyecto = fecha(duracion - 1)\n"
                       "_filas, _k = {}, 0\n"
                       "_lunes = inicio\n"
                       "while _lunes <= fin_proyecto:\n"
                       "    fila = {}\n"
                       "    for j, nombre in enumerate(['lun', 'mar', 'mié', 'jue', 'vie', 'sáb', 'dom']):\n"
                       "        dia = _lunes + np.timedelta64(j, 'D')\n"
                       "        if j >= 5:\n"
                       "            fila[nombre] = '—'\n"
                       "        elif dia in festivos:\n"
                       "            fila[nombre] = f'{str(dia)[8:]}: F'\n"
                       "        elif dia <= fin_proyecto:\n"
                       "            fila[nombre] = f'{str(dia)[8:]}: k={_k}'; _k += 1\n"
                       "        else:\n"
                       "            fila[nombre] = ''\n"
                       "    _filas[f'semana del {_lunes}'] = fila\n"
                       "    _lunes = _lunes + np.timedelta64(7, 'D')\n"
                       "pd.DataFrame(_filas).T"))
    celdas.append(code('#| tbl-cap: "Fechas de inicio y fin (último día trabajado) de cada actividad"\n#| label: tbl-fechas\n'
                       "fechas = pd.DataFrame({\n"
                       "    'inicio temprano': [fecha(k) for k in calendario['inicio_mas_temprano']],\n"
                       "    'fin temprano': [fecha(k - 1) for k in calendario['fin_mas_temprano']],\n"
                       "    'inicio tardío': [fecha(k) for k in calendario['inicio_mas_tardio']],\n"
                       "    'fin tardío': [fecha(k - 1) for k in calendario['fin_mas_tardio']]}, index=calendario.index)\n"
                       "fechas"))
    celdas.append(code("dias_naturales = int((fin_proyecto - inicio).astype(int)) + 1\n"
                       "_finde = int(np.busday_count(inicio, fin_proyecto + np.timedelta64(1, 'D'), weekmask='Sat Sun'))\n"
                       "_fest = [f for f in festivos if inicio <= f <= fin_proyecto]\n"
                       "Markdown(f'**Respuesta:** el proyecto termina el {fin_proyecto} ({DIAS[pd.Timestamp(fin_proyecto).day_name()]}), fecha del día laborable T − 1 = {duracion} − 1 = {duracion - 1}. '\n"
                       "         f'Dura {dias_naturales} días naturales (del {inicio} al {fin_proyecto}, ambos incluidos: {fin_proyecto} − {inicio} + 1): '\n"
                       "         f'{duracion} laborables + {plural(_finde, \"día\")} de fin de semana + {plural(len(_fest), \"festivo\")}'\n"
                       "         + (f' ({lista_y([str(f) for f in _fest])})' if _fest else '') + '.')"))
    celdas += apartado(3)
    celdas.append(md("Una actividad atraviesa un día no laborable si alguno cae entre su primer y su último día "
                     "trabajado. Su duración en días laborables no cambia (se interrumpe ese día y sigue al "
                     "siguiente laborable); sólo se estira en días naturales."))
    celdas.append(code("def no_laborables(ini, fin):\n"
                       "    ini, fin = np.datetime64(pd.Timestamp(ini), 'D'), np.datetime64(pd.Timestamp(fin), 'D')\n"
                       "    todos = np.arange(ini, fin + np.timedelta64(1, 'D'), dtype='datetime64[D]')\n"
                       "    return [x for x in todos if not np.is_busday(x, holidays=festivos)]\n"
                       "criticas = calendario.index[calendario['H_total'] == 0]\n"
                       "_nl = {a: no_laborables(fechas.loc[a, 'inicio temprano'], fechas.loc[a, 'fin temprano']) for a in criticas}\n"
                       "cruce = pd.DataFrame({'inicio': fechas.loc[criticas, 'inicio temprano'], 'fin': fechas.loc[criticas, 'fin temprano'],\n"
                       "                      'duración (laborables)': datos_proyecto.loc[criticas, 'duration'],\n"
                       "                      'días naturales': [(pd.Timestamp(fechas.loc[a, 'fin temprano']) - pd.Timestamp(fechas.loc[a, 'inicio temprano'])).days + 1 for a in criticas],\n"
                       "                      'días no laborables que atraviesa': [len(_nl[a]) for a in criticas],\n"
                       "                      'festivos que atraviesa': [', '.join(str(x) for x in _nl[a] if x in festivos) or '—' for a in criticas]})\n"
                       "display(cruce)\n"
                       "_con_finde = [a for a in criticas if any(x not in festivos for x in _nl[a])]\n"
                       "_con_fest = [a for a in criticas if any(x in festivos for x in _nl[a])]\n"
                       "_sin_fest = np.busday_offset(inicio, duracion - 1, roll='forward')\n"
                       "Markdown(f'**Respuesta:** actividades críticas que atraviesan un fin de semana: {lista_y(_con_finde)}; un festivo: {lista_y(_con_fest)}. '\n"
                       "         'Su duración en días laborables no cambia: sólo se estira en días naturales (compárense las dos columnas de duración). '\n"
                       "         + (f'La fecha de fin del proyecto sí cambia: sin festivos terminaría el {_sin_fest}; con ellos, el {fin_proyecto}. '\n"
                       "            'Cuenta cualquier festivo que caiga entre el inicio y el fin del proyecto, esté o no dentro de una actividad crítica, porque ese día no se trabaja en ninguna.'\n"
                       "            if _sin_fest != fin_proyecto else f'La fecha de fin del proyecto ({fin_proyecto}) no cambia por los festivos porque ninguno cae entre el inicio y el fin del proyecto; los fines de semana sí la alejan.'))"))
    celdas += apartado(4)
    celdas.append(md("Con un festivo más, el día laborable $k$ cae en otra fecha para todos los $k$ posteriores "
                     "al festivo: se recalculan las fechas con la nueva lista de festivos."))
    celdas.append(code(f"festivo_extra = np.datetime64('{festivo_extra}')\n"
                       "festivos2 = sorted(festivos + [festivo_extra])\n"
                       "def fecha2(k):\n"
                       "    return np.busday_offset(inicio, int(k), roll='forward', holidays=festivos2)\n"
                       "fechas2 = pd.DataFrame({'inicio temprano (antes)': fechas['inicio temprano'], 'inicio temprano (nuevo)': [fecha2(k) for k in calendario['inicio_mas_temprano']],\n"
                       "                        'fin temprano (antes)': fechas['fin temprano'], 'fin temprano (nuevo)': [fecha2(k - 1) for k in calendario['fin_mas_temprano']]})\n"
                       "fechas2['cambia'] = ((fechas2.iloc[:, 0] != fechas2.iloc[:, 1]) | (fechas2.iloc[:, 2] != fechas2.iloc[:, 3])).map({True: 'sí', False: 'no'})\n"
                       "display(fechas2)\n"
                       "fin2 = fecha2(duracion - 1)\n"
                       "_cambian = fechas2.index[fechas2['cambia'] == 'sí'].tolist()\n"
                       "_igual = fechas2.index[fechas2['cambia'] == 'no'].tolist()\n"
                       "_retraso = int((fin2 - fin_proyecto).astype(int))\n"
                       "_dia = DIAS[pd.Timestamp(festivo_extra).day_name()]\n"
                       "_en = [a for a in calendario.index if fechas.loc[a, 'inicio temprano'] <= festivo_extra <= fechas.loc[a, 'fin temprano']]\n"
                       "Markdown(f'El nuevo festivo ({_dia} {festivo_extra}) retrasa un día laborable todas las fechas iguales o posteriores a él; las anteriores no cambian. '\n"
                       "         f'**Respuesta:** cambian las fechas de {lista_y(_cambian)}' + (f'; no cambian las de {lista_y(_igual)}, que termina{\"n\" if len(_igual) > 1 else \"\"} antes del festivo' if _igual else '') + '. '\n"
                       "         f'La entrega pasa del {fin_proyecto} al {fin2}: {plural(_retraso, \"día natural\", \"días naturales\")} de retraso (un día laborable'\n"
                       "         + (', que en días naturales son más porque por medio hay días no laborables)' if _retraso > 1 else ')') + '. '\n"
                       "         + (f'En el calendario temprano el festivo cae dentro de {lista_y(_en)}, ' if _en else 'En el calendario temprano el festivo no cae dentro de ninguna actividad, ')\n"
                       "         + 'pero eso da igual: no depende de la criticidad, porque el festivo no se trabaja en ninguna actividad y desplaza todo el calendario posterior. Sólo no afectaría a la entrega si cayera después del fin del proyecto.')"))
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
                       "def txt_plazo(T):\n"
                       "    if T > plazo_contrato:\n"
                       "        return f'penalización {penalizacion_semanal}·({T} − {plazo_contrato}) = {penalizacion_semanal * (T - plazo_contrato)}'\n"
                       "    if T < plazo_contrato:\n"
                       "        return f'prima −{prima_semanal}·({plazo_contrato} − {T}) = −{prima_semanal * (plazo_contrato - T)}'\n"
                       "    return 'ni penalización ni prima (entrega en plazo) = 0'\n"
                       "_sg = lambda v: f'+ {v:.0f}' if v >= 0 else f'− {-v:.0f}'\n"
                       "total_normal = coste_directo_normal + coste_indirecto_semanal * duracion_normal + coste_plazo(duracion_normal)\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'semanas'))\n"
                       "display(Markdown('Coste directo normal: ' + explica_suma(datos_proyecto['cost_normal'], 'directo', 'k€')))\n"
                       "Markdown(f'''**Respuesta:** duración normal {duracion_normal} semanas. Coste total = directo {coste_directo_normal:.0f} + indirecto {coste_indirecto_semanal}·{duracion_normal} = {coste_indirecto_semanal * duracion_normal} "
                       "+ {txt_plazo(duracion_normal)} → {coste_directo_normal:.0f} + {coste_indirecto_semanal * duracion_normal} {_sg(coste_plazo(duracion_normal))} = **{total_normal:.0f} k€**.''')"))
    celdas += apartado(2)
    celdas.append(md("Ackoff-Sasieni reduce el proyecto semana a semana: en cada paso hay que acortar una semana "
                     "todas las rutas críticas, y se elige la actividad (o la combinación de actividades) con "
                     "margen de reducción que lo consigue con el menor coste unitario. La primera tabla da la "
                     "duración de cada ruta, para identificar las rutas por su nombre; la segunda, el algoritmo "
                     "completo; la tercera, las opciones que se comparan en cada paso."))
    celdas.append(code("explica_rutas(proyecto, datos_proyecto['duration'])"))
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
                       "resumen['acumulado = suma de pasos'] = [' + '.join(f'{v:.0f}' for v in resumen['sobrecoste del paso'].iloc[:i + 1]) + f' = {resumen[\"sobrecoste acumulado\"].iloc[i]:.0f}' for i in range(len(resumen))]\n"
                       "display(resumen)\n"
                       "Markdown(f'**Respuesta:** la tabla da el sobrecoste directo acumulado de cada semana de reducción; hasta la duración tope '\n"
                       "         f'({int(resumen[\"duración\"].iloc[-1])} semanas, {plural(len(resumen), \"semana\")} de reducción) el sobrecoste acumulado es {resumen[\"sobrecoste acumulado\"].iloc[-1]:.0f} k€.')"))
    celdas += apartado(3)
    celdas.append(md("Para cada duración: coste directo = normal + sobrecoste acumulado; indirecto = coste semanal × "
                     "duración; penalización = penalización semanal × semanas de retraso sobre el plazo; prima = "
                     "−(prima semanal × semanas de adelanto). El coste total es la suma de las cuatro columnas."))
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
                       "_f = curva.loc[duracion_optima]\n"
                       "_k = duracion_normal - duracion_optima\n"
                       "Markdown(f'En la duración óptima, {duracion_optima} semanas ({plural(_k, \"semana\")} de reducción): directo {coste_directo_normal:.0f} + {_f[\"coste directo\"] - coste_directo_normal:.0f} = {_f[\"coste directo\"]:.0f}; '\n"
                       "         f'indirecto {coste_indirecto_semanal}·{duracion_optima} = {_f[\"coste indirecto\"]:.0f}; {txt_plazo(duracion_optima)}; '\n"
                       "         f'total {_f[\"coste directo\"]:.0f} + {_f[\"coste indirecto\"]:.0f} {_sg(_f[\"penalización\"] + _f[\"prima (negativa)\"])} = {_f[\"coste total\"]:.0f} k€. '\n"
                       "         f'**Respuesta:** la duración óptima es {duracion_optima} semanas, con un coste total de {curva[\"coste total\"].min():.0f} k€ (frente a {total_normal:.0f} k€ sin reducir, un ahorro de {total_normal - curva[\"coste total\"].min():.0f} k€), '\n"
                       "         f'{\"antes del plazo contractual\" if duracion_optima < plazo_contrato else \"justo en el plazo contractual\" if duracion_optima == plazo_contrato else \"con retraso sobre el plazo contractual\"} ({plazo_contrato} semanas).')"))
    celdas += apartado(4)
    celdas.append(md("Cada semana de reducción cuesta el sobrecoste del paso de Ackoff-Sasieni y ahorra el coste "
                     "indirecto semanal **más** la penalización semanal mientras el proyecto esté por detrás del "
                     "plazo, o **más** la prima mientras esté por delante. La reducción compensa mientras el "
                     "sobrecoste del paso sea menor que ese ahorro."))
    celdas.append(code("decision = resumen[['duración', 'sobrecoste del paso']].copy()\n"
                       "decision.insert(0, 'de → a', [f'{T + 1} → {T}' for T in decision['duración']])\n"
                       "decision['ahorro por la semana'] = [coste_indirecto_semanal + (penalizacion_semanal if T + 1 > plazo_contrato else prima_semanal) for T in decision['duración']]\n"
                       "decision['ahorro = indirecto + penalización o prima'] = [f'{coste_indirecto_semanal} + {penalizacion_semanal} (penalización) = {coste_indirecto_semanal + penalizacion_semanal}' if T + 1 > plazo_contrato\n"
                       "                                                         else f'{coste_indirecto_semanal} + {prima_semanal} (prima) = {coste_indirecto_semanal + prima_semanal}' for T in decision['duración']]\n"
                       "decision['conviene'] = (decision['sobrecoste del paso'] < decision['ahorro por la semana']).map({True: 'sí', False: 'no'})\n"
                       "display(decision.drop(columns='ahorro por la semana'))\n"
                       "_antes = coste_indirecto_semanal + penalizacion_semanal; _despues = coste_indirecto_semanal + prima_semanal\n"
                       "_txt = (f'Mientras el proyecto va por detrás del plazo ({plazo_contrato} semanas), cada semana reducida ahorra {coste_indirecto_semanal} + {penalizacion_semanal} = {_antes} k€; '\n"
                       "        f'una vez alcanzado, sólo {coste_indirecto_semanal} + {prima_semanal} = {_despues} k€, y se compara con el sobrecoste del paso. ')\n"
                       "paso_siguiente = resumen[resumen['duración'] == duracion_optima - 1]\n"
                       "if len(paso_siguiente) and duracion_optima <= plazo_contrato:\n"
                       "    sobre = float(paso_siguiente['sobrecoste del paso'].iloc[0])\n"
                       "    prima_necesaria = sobre - coste_indirecto_semanal\n"
                       "    _txt += (f'**Respuesta:** se reduce hasta {duracion_optima} semanas; reducir una semana más (a {duracion_optima - 1}) costaría {sobre:.0f} k€ y ahorraría {coste_indirecto_semanal} de indirectos más la prima p: '\n"
                       "             f'compensa si {sobre:.0f} < {coste_indirecto_semanal} + p, es decir, con una prima de más de {sobre:.0f} − {coste_indirecto_semanal} = {max(prima_necesaria, 0):.0f} k€/semana (ahora es {prima_semanal}).')\n"
                       "elif len(paso_siguiente):\n"
                       "    _txt += f'**Respuesta:** se reduce hasta {duracion_optima} semanas; la semana siguiente sigue estando por detrás del plazo, así que la prima no interviene en esa decisión.'\n"
                       "else:\n"
                       "    _txt += f'**Respuesta:** se reduce hasta {duracion_optima} semanas, la duración tope: ninguna prima permitiría reducir más.'\n"
                       "Markdown(_txt)"))
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
