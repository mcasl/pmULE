"""Colección 10 — Reducción de la duración con mínimo coste: CUR, algoritmo
de Ackoff-Sasieni, costes indirectos y duración óptima, penalizaciones."""
import contextlib
import io

import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="10_reduccion", prefijo="Reduccion",
            titulo="Reducción de la duración con mínimo coste",
            descripcion="Coste unitario de reducción, algoritmo de Ackoff-Sasieni paso a paso, costes "
                        "indirectos y duración de coste mínimo, reducción frente a penalización.",
            n_defecto=3)


def _pasos_ackoff(info, D, tope, ucr, reduction):
    with contextlib.redirect_stdout(io.StringIO()):
        tabla, best, dur, avail = info["proyecto"].ackoff(D, tope, ucr, reduction=reduction, report=False)
    pasos = [(acts, float(sum(ucr[a] for a in acts))) for _, acts in sorted(best.items())]
    return pasos, dur


def _base(k, rng):
    variante = (k - 1) % 3 + 1
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 12 and (i["D"] >= 2).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"], n_inf=(2 if variante == 3 else 0))
        ucr_ack = ucr.replace(0.0, np.inf)
        margen = int((i["D"] - tope).sum())
        if margen < 4:
            return False
        try:
            pasos, dur = _pasos_ackoff(i, i["D"], tope, ucr_ack, reduction=100)
        except Exception:
            return False
        if len(pasos) < 4:
            return False
        if variante == 1 and not any(len(acts) >= 2 for acts, _ in pasos[:3]):
            return False
        costes = [c for _, c in pasos]
        if variante == 2 and not (costes[0] < costes[-1]):
            return False
        estado.update(tope=tope, cn=cn, ct=ct, ucr=ucr, ucr_ack=ucr_ack, pasos=pasos, dur_final=dur)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, cn, ct, ucr, pasos = estado["tope"], estado["cn"], estado["ct"], estado["ucr"], estado["pasos"]
    costes = [c for _, c in pasos]
    T_tope = T - len(pasos)

    celdas = [md("# Enunciado")]
    if variante == 1:
        r = 3
        df = P.tabla_base(info, duration=D, cap_duration=tope, cost_normal=cn, cost_cap=ct)
        celdas.append(md("Considere el proyecto cuyo cuadro de prelaciones se adjunta, con la duración normal y la "
                         "duración tope (mínima) de cada actividad, en semanas, y el coste (en k€) de ejecutarla "
                         "en cada una de esas duraciones. Se supone coste lineal entre ambas."))
        pregs = ["Calcule el coste unitario de reducción (CUR) de cada actividad.",
                 "Determine la duración del proyecto, su camino crítico y el coste del proyecto con las "
                 "duraciones normales.",
                 f"Reduzca la duración del proyecto en {r} semanas con el mínimo sobrecoste aplicando el "
                 "algoritmo de Ackoff-Sasieni paso a paso. Indique en cada paso qué actividad(es) se acortan "
                 "y por qué.",
                 "Indique el coste final del proyecto y su(s) nuevo(s) camino(s) crítico(s). ¿Por qué en "
                 "algún paso ha hecho falta acortar más de una actividad a la vez?"]
    elif variante == 2:
        c_ind = int(np.ceil(float(np.median(costes))))
        if c_ind in costes:
            c_ind += 1
        df = P.tabla_base(info, duration=D, cap_duration=tope, cost_normal=cn, cost_cap=ct)
        celdas.append(md(f"Considere el proyecto cuyo cuadro de prelaciones se adjunta, con la duración normal y "
                         f"la duración tope de cada actividad (semanas) y el coste directo (k€) de ejecutarla en "
                         f"cada una. Además, el proyecto tiene unos costes indirectos (oficina, alquileres, "
                         f"seguros) de {c_ind} k€ por semana de duración."))
        pregs = ["Calcule el coste unitario de reducción (CUR) de cada actividad.",
                 "Determine la duración normal del proyecto y su coste total (directo más indirecto).",
                 "Aplique el algoritmo de Ackoff-Sasieni hasta alcanzar la duración tope del proyecto, anotando "
                 "en cada paso el sobrecoste directo.",
                 "Construya la curva de coste total frente a duración y determine la duración de coste mínimo. "
                 "¿Qué regla de parada se deduce comparando el CUR del paso con el coste indirecto semanal?",
                 "¿Por qué no se puede reducir el proyecto por debajo de su duración tope aunque queden "
                 "actividades con margen de reducción?"]
        estado["c_ind"] = c_ind
    else:
        r_lim = min(len(pasos), int(rng.integers(2, 4)))
        T_lim = T - r_lim
        pen = int(np.ceil(float(np.median(costes[:r_lim + 1]))))
        if pen in costes:
            pen += 1
        df = P.tabla_base(info, duration=D, cap_duration=tope, ucr=ucr.replace(0.0, np.inf))
        no_red = sorted(df.index[df["ucr"] == np.inf].tolist())
        celdas.append(md(f"El contrato fija la entrega del proyecto en la semana {T_lim}, con una penalización de "
                         f"{pen} k€ por cada semana de retraso. Se adjuntan las prelaciones, la duración normal y "
                         f"la tope de cada actividad (semanas) y su coste unitario de reducción (k€/semana). "
                         f"Las actividades {', '.join(no_red)} no pueden acelerarse."))
        pregs = ["Determine la duración del proyecto con las duraciones normales y la penalización a la que se "
                 "expone si no se reduce.",
                 f"Aplique el algoritmo de Ackoff-Sasieni para llevar el proyecto a la semana {T_lim}, anotando "
                 "el sobrecoste de cada paso.",
                 "Compare en cada paso el sobrecoste con la penalización evitada y decida cuántas semanas "
                 "conviene reducir. ¿Cuál es el coste total mínimo (sobrecoste más penalización)?",
                 "¿Qué habría cambiado si las actividades no acelerables hubieran sido críticas? Compruébelo "
                 "sobre el camino crítico."]
        estado.update(T_lim=T_lim, pen=pen, r_lim=r_lim)
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    if variante in (1, 2):
        celdas += apartado(ap); ap += 1
        celdas.append(md("$CUR = \\dfrac{\\text{coste tope} - \\text{coste normal}}{\\text{duración normal} - "
                         "\\text{duración tope}}$. Las actividades cuya duración tope coincide con la normal no "
                         "pueden acortarse: su CUR es infinito."))
        celdas.append(code("reducible = datos_proyecto['duration'] > datos_proyecto['cap_duration']\n"
                           "datos_proyecto['ucr'] = np.where(reducible,\n"
                           "                                 (datos_proyecto['cost_cap'] - datos_proyecto['cost_normal']) / (datos_proyecto['duration'] - datos_proyecto['cap_duration']).replace(0, np.nan),\n"
                           "                                 np.inf)\n"
                           "datos_proyecto[['duration', 'cap_duration', 'cost_normal', 'cost_cap', 'ucr']]"))
    celdas += apartado(ap); ap += 1
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales y el camino crítico", label="fig-pert")
    if variante == 1:
        celdas.append(code("duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                           "coste_normal = int(datos_proyecto['cost_normal'].sum())\n"
                           "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])\n"
                           "Markdown(f'Con las duraciones normales el proyecto dura {duracion_normal} semanas y cuesta {coste_normal} k€.')"))
    elif variante == 2:
        celdas.append(code(f"coste_indirecto_semanal = {c_ind}\n"
                           "duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                           "coste_directo_normal = int(datos_proyecto['cost_normal'].sum())\n"
                           "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])\n"
                           "Markdown(f'Duración normal: {duracion_normal} semanas. Coste directo {coste_directo_normal} k€ + indirecto '\n"
                           "         f'{coste_indirecto_semanal} × {duracion_normal} = {coste_indirecto_semanal * duracion_normal} k€; total '\n"
                           "         f'{coste_directo_normal + coste_indirecto_semanal * duracion_normal} k€.')"))
    else:
        celdas.append(code(f"plazo_contrato, penalizacion_semanal = {estado['T_lim']}, {estado['pen']}\n"
                           "duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                           "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])\n"
                           "retraso = max(0, duracion_normal - plazo_contrato)\n"
                           "Markdown(f'Con las duraciones normales el proyecto dura {duracion_normal} semanas, {retraso} más que el plazo: '\n"
                           "         f'penalización de {retraso} × {penalizacion_semanal} = {retraso * penalizacion_semanal} k€.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("En cada paso del algoritmo de Ackoff-Sasieni se acorta una semana la ruta (o rutas) "
                     "crítica(s) eligiendo, entre las actividades críticas con margen, la combinación de menor "
                     "CUR que reduzca *todas* las rutas críticas a la vez. Cuando hay varias rutas críticas "
                     "sin actividades comunes baratas hay que acortar una actividad de cada una, y el coste "
                     "del paso es la suma. La tabla muestra, para cada ruta, el CUR de sus actividades y su "
                     "duración tras cada paso; las últimas filas, el margen de reducción que le queda a cada actividad."))
    if variante == 1:
        red = "reduction=3"
    elif variante == 2:
        red = "reduction=100"
    else:
        red = f"reduction={estado['r_lim']}"
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       f"    costs=datos_proyecto['ucr'], {red}, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(md("Lo que se compara en cada paso: las rutas críticas en ese momento, las opciones que acortan "
                     "todas a la vez (una actividad común con margen, o una pareja con una actividad de cada ruta) "
                     "con su coste, y la elegida (la más barata)."))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso de Ackoff-Sasieni"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],\n"
                       "                        'sobrecoste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},\n"
                       "                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))\n"
                       "resumen['duración'] = duracion_normal - resumen.index\n"
                       "resumen['sobrecoste acumulado'] = resumen['sobrecoste del paso'].cumsum()\n"
                       "resumen"))
    celdas += apartado(ap); ap += 1
    if variante == 1:
        celdas.append(code("duracion_reducida = proyecto.duration(duraciones_reducidas)\n"
                           "coste_final = coste_normal + resumen['sobrecoste del paso'].sum()\n"
                           "camino_critico_reducido = proyecto.display_critical_path(duraciones_reducidas)\n"
                           "pasos_dobles = [p for p, acts in pasos.items() if len(acts) > 1]\n"
                           "Markdown(f'''El proyecto pasa a durar {duracion_reducida} semanas con un coste de {coste_normal} + {resumen[\"sobrecoste del paso\"].sum():.0f} = {coste_final:.0f} k€. "
                           "Al acortar la ruta crítica, otras rutas con holgura pequeña alcanzan su duración y pasan a ser críticas; a partir de ese momento "
                           "reducir una sola actividad no basta, salvo que sea común a todas las rutas críticas, y hay que acortar una actividad de cada ruta "
                           "(paso(s) {\", \".join(str(p + 1) for p in pasos_dobles)}), con el coste sumado.''')"))
        celdas += fig_grafo(durations="duraciones_reducidas", cap="Grafo PERT con las duraciones reducidas", label="fig-pertReducido")
    elif variante == 2:
        celdas.append(code("curva = pd.DataFrame({'duración': [duracion_normal] + resumen['duración'].tolist(),\n"
                           "                      'coste directo': [coste_directo_normal] + (coste_directo_normal + resumen['sobrecoste acumulado']).tolist()})\n"
                           "curva['coste indirecto'] = coste_indirecto_semanal * curva['duración']\n"
                           "curva['coste total'] = curva['coste directo'] + curva['coste indirecto']\n"
                           "curva = curva.set_index('duración')\n"
                           "curva"))
        celdas.append(code('#| fig-cap: "Curva coste-duración del proyecto"\n#| label: fig-curvaCoste\n'
                           "ax = curva.plot(marker='o', figsize=(7, 4))\n"
                           "ax.set_ylabel('k€'); ax.set_xlabel('duración (semanas)'); ax.invert_xaxis();"))
        celdas.append(code("duracion_optima = int(curva['coste total'].idxmin())\n"
                           "Markdown(f'''El coste total es mínimo, {curva[\"coste total\"].min():.0f} k€, para una duración de {duracion_optima} semanas. "
                           "Cada semana de reducción cuesta el CUR del paso y ahorra {coste_indirecto_semanal} k€ de indirectos, así que compensa reducir "
                           "mientras el sobrecoste del paso sea menor que {coste_indirecto_semanal}; como los pasos de Ackoff-Sasieni tienden a ser cada vez más caros, "
                           "en cuanto el CUR del paso supera el coste indirecto semanal se para (conviene comprobar la tabla completa por si un paso posterior volviera a ser barato).''')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("Markdown(f'''La duración tope del proyecto, {duracion_normal - len(pasos)} semanas, se alcanza cuando todas las actividades de alguna ruta crítica "
                           "están en su duración tope: esa ruta ya no puede acortarse y marca la duración mínima, aunque otras actividades (de otras rutas, o críticas "
                           "pero no de esa ruta) conserven margen. Margen que queda por actividad: {{a: int(m) for a, m in margen_restante[margen_restante > 0].items()}}.''')"))
    else:
        celdas.append(code("decision = resumen.copy()\n"
                           "decision['penalización evitada'] = penalizacion_semanal\n"
                           "decision['conviene'] = decision['sobrecoste del paso'] < penalizacion_semanal\n"
                           "decision['coste total (sobrecoste + penalización)'] = decision['sobrecoste acumulado'] + penalizacion_semanal * (decision['duración'] - plazo_contrato).clip(lower=0)\n"
                           "decision"))
        celdas.append(code("sin_reducir = penalizacion_semanal * retraso\n"
                           "mejor = decision['coste total (sobrecoste + penalización)'].idxmin()\n"
                           "coste_mejor = decision.loc[mejor, 'coste total (sobrecoste + penalización)']\n"
                           "Markdown(f'''Sin reducir, la penalización sería de {sin_reducir} k€. Conviene dar cada paso mientras su sobrecoste sea menor que la penalización "
                           "de la semana que evita ({penalizacion_semanal} k€); el coste total mínimo es {coste_mejor:.0f} k€ y se alcanza reduciendo {mejor} semana(s), "
                           "hasta una duración de {duracion_normal - mejor} semanas.''')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("no_acelerables = datos_proyecto.index[np.isinf(datos_proyecto['ucr'])].tolist()\n"
                           "criticas = sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts))\n"
                           "en_critico = [a for a in no_acelerables if a in criticas]\n"
                           "Markdown(f'''Las actividades no acelerables son {\", \".join(no_acelerables)}; de ellas, {\", \".join(en_critico) or \"ninguna\"} está en el camino crítico. "
                           "Una actividad no acelerable en el camino crítico no impide reducir el proyecto mientras haya otras actividades críticas con margen, "
                           "pero cuando todas las rutas críticas pasan por actividades sin margen la duración tope queda fijada y ninguna cantidad de dinero la rebaja.''')"))
    sub = {1: f"{n} actividades, reducción de {3} semanas", 2: f"{n} actividades, costes indirectos de {estado.get('c_ind')} k€/semana",
           3: f"{n} actividades, plazo {estado.get('T_lim')} con penalización"}[variante]
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _barata_no_optima(rng):
    """Con dos rutas críticas, la actividad crítica más barata no sirve si no es común a las dos."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 2 and i["T"] >= 12 and (i["D"] >= 2).all()):
            return False
        Pm, rc = i["P"], i["rutas_criticas"]
        comunes = set(Pm.columns[(Pm.loc[rc[0]] == 1) & (Pm.loc[rc[1]] == 1)])
        solo1 = set(Pm.columns[(Pm.loc[rc[0]] == 1) & (Pm.loc[rc[1]] == 0)])
        solo2 = set(Pm.columns[(Pm.loc[rc[1]] == 1) & (Pm.loc[rc[0]] == 0)])
        if not comunes or not solo1 or not solo2:
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        # forzar: la más barata de todas las críticas está sólo en una ruta; y la pareja barata cuesta menos que la común más barata o más
        barata = sorted(solo1 | solo2, key=lambda a: ucr[a])[0]
        if np.isinf(ucr[barata]):
            return False
        ucr[barata] = 1.0
        tope[barata] = min(tope[barata], int(i["D"][barata]) - 1)
        if tope[barata] < 1:
            return False
        for a in comunes:
            if not np.isinf(ucr[a]):
                ucr[a] = max(ucr[a], 4.0)
        try:
            pasos, _ = _pasos_ackoff(i, i["D"], tope, ucr, reduction=2)
        except Exception:
            return False
        if len(pasos) < 2:
            return False
        estado.update(tope=tope, ucr=ucr, barata=barata, comunes=comunes, pasos=pasos, rc=rc)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, ucr, barata, comunes, pasos = estado["tope"], estado["ucr"], estado["barata"], estado["comunes"], estado["pasos"]
    df = P.tabla_base(info, duration=D, cap_duration=tope, ucr=ucr)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste unitario "
                 "de reducción (k€/semana) se adjunta. Las actividades con CUR infinito no pueden acelerarse.")]
    celdas.append(preguntas([
        "Determine la duración del proyecto y sus rutas críticas. ¿Qué actividades son comunes a todas ellas?",
        f"La actividad crítica más barata de acelerar es {barata}. Un compañero propone acortarla una semana para "
        "reducir el proyecto una semana. Prediga, sin calcular, si funcionará.",
        f"Compruébelo: acorte {barata} una semana y recalcule la duración.",
        "Aplique el algoritmo de Ackoff-Sasieni para reducir el proyecto dos semanas con mínimo coste y explique "
        "por qué en cada paso elige lo que elige (actividad común frente a pareja de actividades)."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales y las rutas críticas", label="fig-pert")
    celdas.append(code("duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "for nombre, acts in rutas_criticas.items():\n"
                       "    print(f'{nombre}: {\" - \".join(acts)}')\n"
                       "comunes = set.intersection(*[set(a) for a in rutas_criticas.values()])\n"
                       "Markdown(f'Duración {duracion_normal} semanas con {len(rutas_criticas)} rutas críticas; actividades comunes a todas: {\", \".join(sorted(comunes)) or \"ninguna\"}.')"))
    celdas += apartado(2)
    celdas.append(md(f"No funcionará: {barata} está en una sola de las rutas críticas. Al acortarla, esa ruta se "
                     "acorta, pero la otra sigue midiendo lo mismo y el proyecto también. Para reducir el proyecto "
                     "hay que acortar *todas* las rutas críticas a la vez: una actividad común, o una actividad de "
                     "cada ruta."))
    celdas += apartado(3)
    celdas.append(code(f"barata = '{barata}'\n"
                       "d1 = datos_proyecto['duration'].copy(); d1[barata] -= 1\n"
                       "Markdown(f'Acortando {barata} una semana la duración del proyecto es {proyecto.duration(d1)} semanas (antes {duracion_normal}): '\n"
                       "         f'se han gastado {datos_proyecto.loc[barata, \"ucr\"]:.0f} k€ para nada; la ruta {[r for r, a in rutas_criticas.items() if barata not in a][0]} sigue siendo crítica.')"))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=2, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(md("Lo que se compara en cada paso: las rutas críticas en ese momento, las opciones que acortan "
                     "todas a la vez (una actividad común con margen, o una pareja con una actividad de cada ruta) "
                     "con su coste, y la elegida (la más barata)."))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso de Ackoff-Sasieni"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],\n"
                       "                        'coste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},\n"
                       "                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))\n"
                       "resumen"))
    celdas.append(code("mas_barata_comun = min(comunes, key=lambda a: datos_proyecto.loc[a, 'ucr']) if comunes else None\n"
                       "pareja = sorted(set(a for r, acts in rutas_criticas.items() for a in acts) - comunes, key=lambda a: datos_proyecto.loc[a, 'ucr'])\n"
                       "Markdown(f'''En cada paso el algoritmo compara el coste de la actividad común más barata ({mas_barata_comun}: {datos_proyecto.loc[mas_barata_comun, \"ucr\"]:.0f} k€) "
                       "con el de la pareja más barata que cubre las dos rutas, y elige la opción de menor coste *que reduzca todas las rutas críticas*. "
                       "{barata} sola nunca es candidata, por barata que sea, salvo emparejada con una actividad de la otra ruta.''')"))
    celdas.append(code("Markdown(f'Duración final {proyecto.duration(duraciones_reducidas)} semanas, sobrecoste total {resumen[\"coste del paso\"].sum():.0f} k€.')"))
    return celdas, f"{n} actividades, dos rutas críticas; {barata} es la más barata pero no es común"


def _desaceleracion(rng):
    """Reducido el proyecto, el plazo se relaja una semana: qué reducción deshacer (la del paso más caro)."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 12 and (i["D"] >= 2).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        try:
            pasos, dur = _pasos_ackoff(i, i["D"], tope, ucr, reduction=4)
        except Exception:
            return False
        if len(pasos) < 4:
            return False
        costes = [c for _, c in pasos]
        if costes.index(max(costes)) == 0 or len(set(costes)) < 3:
            return False
        estado.update(tope=tope, ucr=ucr, pasos=pasos, dur=dur)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, ucr, pasos = estado["tope"], estado["ucr"], estado["pasos"]
    df = P.tabla_base(info, duration=D, cap_duration=tope, ucr=ucr)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste unitario "
                 "de reducción (k€/semana) se adjunta. El proyecto se había reducido cuatro semanas con el algoritmo "
                 "de Ackoff-Sasieni cuando el cliente comunica que acepta la entrega una semana más tarde.")]
    celdas.append(preguntas([
        "Reduzca el proyecto cuatro semanas con el algoritmo de Ackoff-Sasieni y anote el sobrecoste de cada paso.",
        "Hay que devolver una semana. Un compañero propone deshacer el primer paso («fue el primero, es el más fácil "
        "de anular»). Otro propone deshacer el último. ¿Da lo mismo? Razónelo antes de calcular.",
        "Compruebe ambas opciones: duración resultante y dinero recuperado en cada caso. ¿Puede deshacerse un paso "
        "intermedio? ¿Qué regla general se deduce?",
        "¿Por qué el sobrecoste marginal de cada semana de reducción es creciente (o al menos no decreciente) en "
        "Ackoff-Sasieni? ¿Qué forma tiene la curva coste-duración?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion_normal = proyecto.duration(datos_proyecto['duration'])"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=4, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(md("Lo que se compara en cada paso: las rutas críticas en ese momento, las opciones que acortan "
                     "todas a la vez (una actividad común con margen, o una pareja con una actividad de cada ruta) "
                     "con su coste, y la elegida (la más barata)."))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso de Ackoff-Sasieni"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],\n"
                       "                        'coste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},\n"
                       "                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))\n"
                       "sobrecoste = float(resumen['coste del paso'].sum())\n"
                       "display(resumen)\n"
                       "Markdown(f'Duración {duracion_normal} → {proyecto.duration(duraciones_reducidas)} semanas con un sobrecoste de {sobrecoste:.0f} k€.')"))
    celdas += apartado(2)
    celdas.append(md("No da lo mismo. Deshacer un paso es devolver una semana a las actividades que se acortaron en "
                     "él; cualquiera de los pasos devuelve una semana al proyecto (todas las rutas críticas se alargan "
                     "una semana), pero el dinero recuperado es el coste de ese paso. Como los pasos de Ackoff-Sasieni "
                     "son cada vez más caros, deshacer el último recupera más dinero que deshacer el primero."))
    celdas += apartado(3)
    celdas.append(code("filas = {}\n"
                       "for paso, acts in pasos.items():\n"
                       "    d2 = duraciones_reducidas.copy()\n"
                       "    for a in acts:\n"
                       "        d2[a] += 1\n"
                       "    filas[paso + 1] = {'actividades devueltas': ', '.join(acts), 'duración resultante': proyecto.duration(d2),\n"
                       "                       'dinero recuperado': float(sum(datos_proyecto.loc[acts, 'ucr']))}\n"
                       "pd.DataFrame(filas).T"))
    celdas.append(code("costes_pasos = {p: float(sum(datos_proyecto.loc[acts, 'ucr'])) for p, acts in pasos.items()}\n"
                       "mejor = max(costes_pasos, key=costes_pasos.get)\n"
                       "Markdown(f'''Todos los pasos devuelven el proyecto a {proyecto.duration(duraciones_reducidas) + 1} semanas, pero el que más dinero recupera es el paso {mejor + 1} "
                       "({costes_pasos[mejor]:.0f} k€). Un paso intermedio también puede deshacerse: lo que importa no es el orden en que se dieron "
                       "sino su coste. Regla: para relajar el plazo se deshacen las reducciones de mayor coste primero (recorrer la tabla de Ackoff-Sasieni hacia atrás).''')"))
    celdas += apartado(4)
    celdas.append(md("En cada paso el algoritmo elige la opción más barata disponible; las opciones baratas se agotan "
                     "(las actividades llegan a su tope) o dejan de bastar (aparecen nuevas rutas críticas que exigen "
                     "acortar varias actividades a la vez), así que el coste del paso siguiente nunca es menor. La "
                     "curva coste-duración es por tanto convexa: cada semana adicional de reducción cuesta más que la "
                     "anterior, y por eso la decisión de hasta dónde reducir se toma comparando el coste marginal con "
                     "lo que vale cada semana (penalización, coste indirecto, prima)."))
    return celdas, f"{n} actividades; deshacer la reducción más cara al relajar el plazo"



def _lp_coste(R, D, tope, ucr, T_obj):
    """Coste mínimo de reducir el proyecto a T_obj por programación lineal (None si no es factible)."""
    from scipy.optimize import linprog
    acts = list(R.columns)
    c = np.array([0.0 if np.isinf(ucr[a]) else float(ucr[a]) for a in acts])
    ub = np.array([0.0 if np.isinf(ucr[a]) else float(D[a] - tope[a]) for a in acts])
    A = -R.values.astype(float)
    b = T_obj - R.values.astype(float) @ np.array([float(D[a]) for a in acts])
    res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(np.zeros(len(acts)), ub)), method="highs")
    return (float(res.fun), pd.Series(res.x, index=acts)) if res.success else (None, None)


def _programacion_lineal(rng):
    """Ackoff-Sasieni (voraz) frente a la solución exacta por programación lineal con la matriz de caminos."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def _prepara(i, estricto):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 12 and (i["D"] >= 2).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        if int((i["D"] - tope).sum()) < 4:
            return False
        try:
            pasos, _ = _pasos_ackoff(i, i["D"], tope, ucr, reduction=100)
        except Exception:
            return False
        if len(pasos) < 3:
            return False
        for r in (3, 2, 4):
            if r > len(pasos):
                continue
            coste_ackoff = float(sum(c for _, c in pasos[:r]))
            coste_lp, x = _lp_coste(i["P"], i["D"], tope, ucr, int(i["T"]) - r)
            if coste_lp is None:
                continue
            if (not estricto) or coste_lp < coste_ackoff - 1e-6:
                estado.update(tope=tope, ucr=ucr, pasos=pasos, r=r, coste_ackoff=coste_ackoff, coste_lp=coste_lp)
                return True
        return False

    try:
        info = P.buscar(rng, lambda i: _prepara(i, True), n=n, n_inicio=int(rng.choice([2, 3])),
                        n_fin=int(rng.choice([1, 2])), max_intentos=400)
    except RuntimeError:
        info = P.buscar(rng, lambda i: _prepara(i, False), n=n, n_inicio=int(rng.choice([2, 3])),
                        n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, ucr, r = estado["tope"], estado["ucr"], estado["r"]
    mejora = estado["coste_lp"] < estado["coste_ackoff"] - 1e-6
    df = P.tabla_base(info, duration=D, cap_duration=tope, ucr=ucr)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste "
                 "unitario de reducción CUR (k€/semana) se adjunta. Las actividades con CUR infinito no pueden "
                 f"acelerarse. Se quiere reducir la duración del proyecto en {r} semanas con el mínimo sobrecoste.")]
    celdas.append(preguntas([
        "Calcule la duración normal del proyecto, construya la matriz de caminos $R$ y la duración de cada ruta.",
        f"Aplique el algoritmo de Ackoff-Sasieni para reducir {r} semanas y obtenga el sobrecoste total.",
        "Plantee el problema como un programa lineal: variables $x_i$ = semanas que se acorta la actividad "
        "$i$; minimizar $\\sum_i CUR_i\\, x_i$ sujeto a que ninguna ruta supere la duración objetivo, "
        "$R\\,(d - x) \\le T_{obj}$, y $0 \\le x_i \\le d_i - d_i^{tope}$. Resuélvalo con `scipy.optimize.linprog`.",
        "Compare las dos soluciones. ¿Puede el algoritmo voraz ser peor que el óptimo? ¿Por qué el programa "
        "lineal da aquí soluciones enteras?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"reduccion = {r}"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de caminos y duración de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                       "T_objetivo = duracion_normal - reduccion\n"
                       "detalle_rutas(R, datos_proyecto['duration'][R.columns])"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_ackoff, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=reduccion, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("x_ackoff = (datos_proyecto['duration'] - duraciones_ackoff.reindex(datos_proyecto.index)).astype(int)\n"
                       "coste_ackoff = float((x_ackoff * datos_proyecto['ucr'].replace(np.inf, 0)).sum())\n"
                       "Markdown(f'Ackoff-Sasieni acorta {\", \".join(f\"{a} ({int(x)})\" for a, x in x_ackoff[x_ackoff > 0].items())} con un sobrecoste de {coste_ackoff:.0f} k€; duración {proyecto.duration(duraciones_ackoff)}.')"))
    celdas += apartado(3)
    celdas.append(md("Cada fila de $R$ es una ruta; $R\\,d$ son las duraciones de las rutas y $R\\,x$ lo que se "
                     "acorta cada una. La restricción $R\\,(d - x) \\le T_{obj}$ se escribe como "
                     "$-R\\,x \\le T_{obj} - R\\,d$ para `linprog` (que exige la forma $A x \\le b$)."))
    celdas.append(code("from scipy.optimize import linprog\n"
                       "acts = list(R.columns)\n"
                       "d = datos_proyecto.loc[acts, 'duration'].astype(float).values\n"
                       "cur = datos_proyecto.loc[acts, 'ucr'].astype(float)\n"
                       "margen = (datos_proyecto.loc[acts, 'duration'] - datos_proyecto.loc[acts, 'cap_duration']).astype(float)\n"
                       "c = cur.replace(np.inf, 0).values                       # coste por semana acortada\n"
                       "cotas = [(0, 0 if np.isinf(cur[a]) else margen[a]) for a in acts]\n"
                       "A_ub = -R.values.astype(float)\n"
                       "b_ub = T_objetivo - R.values @ d\n"
                       "solucion = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=cotas, method='highs')\n"
                       "x_lp = pd.Series(solucion.x, index=acts).round(6)\n"
                       "coste_lp = float(solucion.fun)\n"
                       "entera = bool(np.allclose(x_lp, x_lp.round()))\n"
                       "pd.DataFrame({'x (Ackoff)': x_ackoff.reindex(acts), 'x (PL)': x_lp, 'CUR': cur}).T"))
    celdas.append(code("duraciones_lp = datos_proyecto['duration'] - x_lp.reindex(datos_proyecto.index).fillna(0)\n"
                       "Markdown(f'La programación lineal acorta {\", \".join(f\"{a} ({x:g})\" for a, x in x_lp[x_lp > 0].items())} con sobrecoste {coste_lp:.0f} k€; '\n"
                       "         f'todas las rutas quedan en como mucho {float((R @ duraciones_lp[acts]).max()):g} semanas (objetivo {T_objetivo}).')"))
    celdas += apartado(4)
    if mejora:
        texto = ("Aquí el programa lineal es **más barato** que Ackoff-Sasieni: el algoritmo voraz decide semana a "
                 "semana y, una vez acortada una actividad, no la «desacorta» aunque más adelante otra "
                 "combinación hubiera resultado mejor para el conjunto. El PL optimiza las {r} semanas a la vez.")
    else:
        texto = ("Aquí las dos soluciones cuestan lo mismo: el algoritmo voraz ha acertado. No siempre es así: "
                 "decide semana a semana y no revisa decisiones anteriores, por lo que puede ser más caro que el "
                 "óptimo cuando una actividad acortada pronto deja de ser útil al cambiar las rutas críticas.")
    celdas.append(code("Markdown(f'''" + texto.replace("{r}", "{reduccion}") + " "
                       "{'Aquí el PL ha dado una solución entera' if entera else 'Aquí el PL ha dado una solución fraccionaria'}: con datos enteros suele ocurrir, pero no está garantizado "
                       "(la matriz de caminos no es en general totalmente unimodular); si hiciera falta, habría que exigir x entera con un programa entero, que `linprog` no resuelve. "
                       "En todo caso el óptimo del PL es una cota inferior del coste de cualquier reducción entera.''')"))
    return celdas, f"{n} actividades; reducir {r} semanas: Ackoff {estado['coste_ackoff']:.0f} k€ frente a PL {estado['coste_lp']:.0f} k€"


def _cur_por_tramos(rng):
    """El CUR crece con cada semana acortada (tramos): hay que usar el coste marginal en cada paso."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 12 and (i["D"] >= 3).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        # margen de exactamente 2 semanas en las actividades reducibles
        tope = pd.Series({a: (int(i["D"][a]) - 2 if ucr[a] > 0 else int(i["D"][a])) for a in i["D"].index}, name="cap_duration")
        cur1 = pd.Series({a: (int(ucr[a]) if ucr[a] > 0 else np.inf) for a in i["D"].index}, name="ucr_1")
        cur2 = pd.Series({a: (int(ucr[a]) + int(rng.integers(2, 7)) if ucr[a] > 0 else np.inf) for a in i["D"].index}, name="ucr_2")
        if (cur1 < np.inf).sum() < 5:
            return False
        criticas = [a for a in i["D"].index if i["calendario"].loc[a, "H_total"] == 0]
        if sum(1 for a in criticas if cur1[a] < np.inf) < 3:
            return False
        estado.update(tope=tope, cur1=cur1, cur2=cur2)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, cur1, cur2 = estado["tope"], estado["cur1"], estado["cur2"]
    r = 3
    df = P.tabla_base(info, duration=D, cap_duration=tope, ucr_1=cur1, ucr_2=cur2)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones normal y tope (semanas) se "
                 "adjunta. Las actividades reducibles pueden acortarse hasta dos semanas, pero el coste no es "
                 "lineal: la primera semana cuesta `ucr_1` k€ y la segunda `ucr_2` k€ (más cara: horas extra, "
                 "turno de noche…). Las actividades con CUR infinito no pueden acelerarse.")]
    celdas.append(preguntas([
        "Calcule la duración normal del proyecto y sus rutas críticas.",
        f"Reduzca el proyecto {r} semanas con mínimo coste aplicando Ackoff-Sasieni paso a paso con el *coste "
        "marginal*: el de la semana que toca acortar a cada actividad. Muestre en cada paso las rutas "
        "críticas, las opciones y su coste.",
        "¿Qué habría resultado de aplicar el algoritmo con un CUR único por actividad igual a la media "
        "$(ucr_1 + ucr_2)/2$? Compare el coste real de esa solución con el del apartado anterior.",
        "¿Por qué con costes crecientes por tramos el procedimiento semana a semana sigue siendo válido, y "
        "qué fallaría si la segunda semana fuera *más barata* que la primera?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       f"reduccion = {r}"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales", label="fig-pert")
    celdas.append(code("duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "for nombre, acts in rutas_criticas.items():\n"
                       "    print(f'{nombre}: {\" - \".join(acts)}')\n"
                       "Markdown(f'Duración normal: {duracion_normal} semanas.')"))
    celdas += apartado(2)
    celdas.append(md("En cada paso el coste marginal de una actividad es `ucr_1` si aún no se ha acortado y "
                     "`ucr_2` si ya se acortó una semana (infinito si no le queda margen). Las opciones son las "
                     "actividades comunes a todas las rutas críticas y las parejas que cubren todas ellas; se "
                     "elige la de menor coste marginal."))
    celdas.append(code("from itertools import combinations\n"
                       "def reducir_por_tramos(d0, tope, ucr_1, ucr_2, semanas):\n"
                       "    d = d0.astype(float).copy()\n"
                       "    filas = []\n"
                       "    for paso in range(1, semanas + 1):\n"
                       "        acortado = d0 - d\n"
                       "        marginal = pd.Series({a: (np.inf if d[a] <= tope[a] else (ucr_1[a] if acortado[a] == 0 else ucr_2[a])) for a in d.index})\n"
                       "        L = R @ d[R.columns]\n"
                       "        T = L.max(); criticas = [r for r in L.index if np.isclose(L[r], T)]\n"
                       "        conjuntos = [set(R.columns[R.loc[r] == 1]) for r in criticas]\n"
                       "        opciones = {}\n"
                       "        for a in set.intersection(*conjuntos):\n"
                       "            if marginal[a] < np.inf: opciones[a] = marginal[a]\n"
                       "        for p, q in combinations(sorted(set().union(*conjuntos)), 2):\n"
                       "            if marginal[p] < np.inf and marginal[q] < np.inf and all(p in s or q in s for s in conjuntos):\n"
                       "                opciones[f'{p}+{q}'] = marginal[p] + marginal[q]\n"
                       "        if not opciones:\n"
                       "            filas.append({'paso': paso, 'duración': int(T), 'rutas críticas': ', '.join(criticas), 'opciones (coste marginal)': 'ninguna', 'elegida': '—', 'coste': np.nan})\n"
                       "            break\n"
                       "        elegida = min(opciones, key=opciones.get)\n"
                       "        filas.append({'paso': paso, 'duración': int(T), 'rutas críticas': ', '.join(criticas),\n"
                       "                      'opciones (coste marginal)': '; '.join(f'{k} ({v:.0f})' for k, v in sorted(opciones.items(), key=lambda kv: kv[1])[:6]),\n"
                       "                      'elegida': elegida, 'coste': opciones[elegida]})\n"
                       "        for a in elegida.split('+'):\n"
                       "            d[a] -= 1\n"
                       "    return pd.DataFrame(filas).set_index('paso'), d\n"
                       "pasos_tramos, d_tramos = reducir_por_tramos(datos_proyecto['duration'], datos_proyecto['cap_duration'],\n"
                       "                                            datos_proyecto['ucr_1'], datos_proyecto['ucr_2'], reduccion)\n"
                       "pasos_tramos"))
    celdas.append(code("coste_tramos = float(pasos_tramos['coste'].sum())\n"
                       "acortadas = (datos_proyecto['duration'] - d_tramos)\n"
                       "Markdown(f'Se acortan {\", \".join(f\"{a} ({int(x)})\" for a, x in acortadas[acortadas > 0].items())}; duración {proyecto.duration(d_tramos.astype(int))} semanas; sobrecoste {coste_tramos:.0f} k€.')"))
    celdas += apartado(3)
    celdas.append(code("ucr_medio = ((datos_proyecto['ucr_1'] + datos_proyecto['ucr_2']) / 2)\n"
                       "tabla_ackoff, pasos, d_medio, _ = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "                                                  costs=ucr_medio, reduction=reduccion, report=False)\n"
                       "acortadas_medio = (datos_proyecto['duration'] - d_medio.reindex(datos_proyecto.index)).astype(int)\n"
                       "def coste_real(acortadas):\n"
                       "    return float(sum((datos_proyecto.loc[a, 'ucr_1'] if x >= 1 else 0) + (datos_proyecto.loc[a, 'ucr_2'] if x >= 2 else 0) for a, x in acortadas.items()))\n"
                       "pd.DataFrame({'con coste marginal': [', '.join(f'{a} ({int(x)})' for a, x in acortadas[acortadas > 0].items()), coste_tramos],\n"
                       "              'con CUR medio': [', '.join(f'{a} ({int(x)})' for a, x in acortadas_medio[acortadas_medio > 0].items()), coste_real(acortadas_medio)]},\n"
                       "             index=['actividades acortadas', 'coste real (k€)'])"))
    celdas.append(md("Con el CUR medio el algoritmo puede preferir acortar dos semanas una actividad que parece "
                     "barata «de media» cuando en realidad su segunda semana es cara, o evitar una cuya primera "
                     "semana era la más barata de todas. El coste real se obtiene valorando cada semana con su "
                     "tramo."))
    celdas += apartado(4)
    celdas.append(md("Con costes marginales *crecientes* la función de coste de cada actividad es convexa, y en "
                     "ese caso acortar semana a semana eligiendo siempre la opción marginal más barata es "
                     "óptimo (es la misma razón por la que el CUR constante funciona). Si la segunda semana "
                     "fuera más barata que la primera (coste cóncavo: un coste fijo de movilización y luego "
                     "barato), el paso a paso fallaría: la primera semana parecería cara y nunca se emprendería, "
                     "aunque acortar dos de golpe fuera lo mejor; habría que evaluar las dos semanas juntas."))
    return celdas, f"{n} actividades; CUR por tramos, reducir {r} semanas"


def _crashing_recursos(rng):
    """Recursos limitados + crashing: Ackoff sin recursos puede elegir una actividad que no acorta el proyecto."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (0 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 22):
            return False
        nombres = list(i["D"].index)
        op = P.recursos(rng, nombres, hi=3).rename("operarios")
        L = int(op.max())
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        if int(np.isfinite(ucr).sum()) < 4:
            return False
        df = P.tabla_base(i, duration=i["D"], operarios=op, cap_duration=tope, ucr=ucr)
        pr = i["proyecto"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = pr.copy().asignar_multi(df[["predecessor", "duration", "operarios"]].copy(), "duration", ["operarios"], [L])
            T_res = int(res.duracion)
            if not (1 <= T_res - int(i["T"]) <= 4):
                return False
            pasos, dur = _pasos_ackoff(i, i["D"], tope, ucr, reduction=2)
            with contextlib.redirect_stdout(io.StringIO()):
                r_ack = pr.copy().asignar_multi(df[["predecessor", "operarios"]].assign(duration=dur), "duration", ["operarios"], [L])
            if int(r_ack.duracion) < T_res:          # Ackoff debe fallar del todo (no acorta nada con recursos)
                return False
            with contextlib.redirect_stdout(io.StringIO()):
                cr = pr.copy().crashing_recursos(df[["predecessor", "duration", "operarios"]].copy(), "duration", ["operarios"], [L],
                                                 tope, ucr, reduccion=2)
            if cr.duracion_final > T_res - 2:
                return False
            with contextlib.redirect_stdout(io.StringIO()):
                libre = pr.copy().asignar_multi(df[["predecessor", "duration", "operarios"]].copy(), "duration", ["operarios"], [L + 1])
            if int(libre.duracion) >= T_res:
                return False
        except Exception:
            return False
        estado.update(df=df, L=L, T_res=T_res)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])),
                    max_intentos=8000)
    df, L, T_res = estado["df"], estado["L"], estado["T_res"]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duración normal y tope (semanas), operarios por "
                 "actividad y coste unitario de reducción (k€/semana) se adjunta. Cada actividad necesita un número fijo "
                 f"de operarios durante toda su duración y la empresa dispone como máximo de {L} operarios en cada semana. "
                 "Las actividades con CUR infinito no pueden acelerarse. Se quiere reducir la duración del proyecto "
                 "**dos semanas** respetando el límite de operarios.")]
    celdas.append(preguntas([
        "Determine la duración del proyecto sin límite de recursos y con el límite de operarios (algoritmo de asignación). "
        "Indique la cadena crítica con recursos: qué actividades fijan la duración y qué esperas por recurso intervienen.",
        "Aplique el algoritmo de Ackoff-Sasieni ignorando los recursos para reducir dos semanas, aplique esas reducciones "
        "y replanifique con el límite de operarios. ¿Se ha conseguido reducir la duración del proyecto con recursos? ¿Por qué?",
        "Reduzca ahora dos semanas teniendo en cuenta el límite: pruebe en cada paso los candidatos (cortes de la cadena "
        "crítica con recursos, actividades sueltas y parejas) y elija el más barato que de verdad acorte el proyecto "
        "replanificado. Compare el sobrecoste con el del apartado anterior.",
        "Suponga que acelerar una actividad exige un operario más por cada semana recortada (modelo con recurso). "
        "¿Cambia la solución?",
        f"Alternativa: contratar un operario más (límite {L + 1}) sin acelerar nada. ¿Qué duración se obtiene? "
        "¿Compensa frente a acelerar?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"limite = [{L}]\n"
                       "base = datos_proyecto[['predecessor', 'duration', 'operarios']]"))
    celdas += apartado(1)
    celdas.append(code("duracion_libre = proyecto.duration(datos_proyecto['duration'])\n"
                       "res0 = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], limite)\n"
                       "cadena = proyecto.copy().cadena_critica_recursos(base.copy(), 'duration', ['operarios'], limite)\n"
                       "Markdown(f'Sin límite: {duracion_libre} semanas. Con {limite[0]} operarios: {res0.duracion} semanas. '\n"
                       "         f'Cadena crítica con recursos: {\", \".join(sorted(cadena[\"actividades\"]))}.')"))
    celdas.append(code('#| tbl-cap: "Arcos de la cadena crítica con recursos"\n#| label: tbl-cadena\n'
                       "cadena['arcos'].rename(columns={'de': 'de', 'a': 'a', 'motivo': 'motivo (precedencia o recurso)'})"))
    celdas.append(md("Los arcos de *recurso* son esperas: una actividad no puede empezar porque otra que usa los mismos "
                     "operarios aún no ha terminado. La duración la fija esa cadena (precedencias más esperas), no el camino crítico."))
    celdas += apartado(2)
    celdas.append(code("tabla_ack, pasos_ack, dur_ack, margen = proyecto.ackoff(durations=datos_proyecto['duration'],\n"
                       "    min_durations=datos_proyecto['cap_duration'], costs=datos_proyecto['ucr'], reduction=2, report=False)\n"
                       "coste_ack = float(sum(datos_proyecto.loc[a, 'ucr'] for acts in pasos_ack.values() for a in acts))\n"
                       "res_ack = proyecto.copy().asignar_multi(base.assign(duration=dur_ack), 'duration', ['operarios'], limite)\n"
                       "Markdown(f'Ackoff sin recursos reduce el proyecto a {proyecto.duration(dur_ack)} semanas (coste {coste_ack:.0f} k€), '\n"
                       "         f'pero con el límite de operarios la duración es {res_ack.duracion} semanas (antes {res0.duracion}). '\n"
                       "         'Ackoff eligió por el camino crítico sin recursos, y eso no acorta la cadena crítica con recursos: '\n"
                       "         'el dinero se gasta sin acortar lo que de verdad manda.')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Crashing con recursos limitados"\n#| label: tbl-crashing\n'
                       "cr = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], limite,\n"
                       "                                       datos_proyecto['cap_duration'], datos_proyecto['ucr'], reduccion=2)\n"
                       "cr.pasos[['actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]"))
    celdas.append(code("Markdown(f'Duración {res0.duracion} → {cr.duracion_final} semanas con un sobrecoste de {cr.coste_total:.0f} k€, '\n"
                       "         f'frente a {coste_ack:.0f} k€ de Ackoff sin recursos, que no consiguió reducir la duración con recursos '\n"
                       "         f'(quedó en {res_ack.duracion}).')"))
    celdas += apartado(4)
    celdas.append(code("extra = {a: {'operarios': 1} for a in datos_proyecto.index if np.isfinite(datos_proyecto.loc[a, 'ucr'])}\n"
                       "cr_b = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], limite,\n"
                       "                                         datos_proyecto['cap_duration'], datos_proyecto['ucr'], reduccion=2,\n"
                       "                                         demanda_extra=extra)\n"
                       "if cr_b.duracion_final <= res0.duracion - 2:\n"
                       "    cierre = 'Aun consumiendo el recurso, el límite todavía permite la reducción pedida, con otras actividades o con un coste distinto.'\n"
                       "else:\n"
                       "    cierre = 'Como acelerar consume el recurso escaso, el límite frena también las reducciones: no se alcanza lo pedido sin ampliar el recurso.'\n"
                       "Markdown(f'Modelo con recurso: parada «{cr_b.motivo_parada}», duración final {cr_b.duracion_final} semanas, '\n"
                       "         f'sobrecoste {cr_b.coste_total:.0f} k€. {cierre}')"))
    celdas += apartado(5)
    celdas.append(code("mas = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], [limite[0] + 1])\n"
                       "Markdown(f'Con {limite[0] + 1} operarios, sin acelerar nada: {mas.duracion} semanas (antes {res0.duracion}). '\n"
                       "         f'Se ahorran {res0.duracion - mas.duracion} semana(s). Reducir {res0.duracion - cr.duracion_final} semanas acelerando cuesta '\n"
                       "         f'{cr.coste_total:.0f} k€. Compensa contratar si el coste del operario adicional durante esas semanas es menor que ese sobrecoste '\n"
                       "         '(y la comparación es justa solo con el mismo número de semanas ahorradas).')"))
    return celdas, f"{n} actividades; límite {L} operarios, reducir dos semanas (Ackoff sin recursos falla)"


def _crashing_incertidumbre(rng):
    """Reducir con duraciones inciertas: la reducción determinista hasta el plazo solo da ~50 % de probabilidad."""
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 10 and 12 <= i["T"] <= 24 and (i["D"] >= 3).all()):
            return False
        a, m, b = P.tres_estimaciones_triangular(rng, i["D"])
        if ((b - a) < 2).sum() > len(i["D"]) // 2:
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        if int(np.isfinite(ucr).sum()) < 4:
            return False
        pg = i["proyecto"]
        dist = {x: ("triangular", float(a[x]), float(m[x]), float(b[x])) for x in i["D"].index}
        df = P.tabla_base(i, duration=i["D"], a=a, m=m, b=b, cap_duration=tope, ucr=ucr)
        T = int(i["T"])
        try:
            t = pg.monte_carlo_crashing(dist, df, "duration", tope, ucr, plazo=T - 2, reduccion_max=8,
                                        n_samples=1500, seed=3)
        except Exception:
            return False
        if t.index.max() < 4:
            return False
        p = t["P(τ ≤ plazo)"]
        if not (0.05 <= p[0] <= 0.35 and p.max() >= 0.9):
            return False
        r90 = int(p[p >= 0.9].index.min())
        if not (3 <= r90 <= t.index.max()):
            return False
        # la reducción determinista justo hasta el plazo (2 meses) debe dar una probabilidad claramente <0,9
        if not (p.get(2, 1.0) <= 0.75 and r90 >= 3):
            return False
        estado.update(df=df, tope=tope, ucr=ucr, plazo=T - 2)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), max_intentos=6000)
    df, plazo = estado["df"], estado["plazo"]
    T = int(info["T"])
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, estimaciones de duración (mínima *a*, más probable *m* y "
                 "máxima *b*, semanas, distribución triangular), duración tope y coste unitario de reducción (k€/semana) se "
                 "adjunta; la columna *duration* es la duración esperada. Las actividades con CUR infinito no se pueden acelerar. "
                 f"El contrato exige entregar en **{plazo} semanas**.")]
    celdas.append(preguntas([
        f"Con las duraciones esperadas el proyecto dura {T} semanas. Simule 5 000 réplicas y estime la probabilidad de entregar "
        f"en {plazo} semanas sin acelerar.",
        f"Aplique Ackoff-Sasieni para reducir el proyecto hasta {plazo} semanas con las duraciones esperadas. "
        "¿Qué probabilidad de cumplir el plazo da ahora esa reducción? Compruébelo simulando.",
        "Construya la tabla de probabilidad de cumplir el plazo, duración media simulada y coste de reducción en función de las "
        "semanas recortadas (de 0 a 8). Use los mismos números aleatorios para todas las filas.",
        "¿Cuántas semanas hay que recortar para tener un 90 % de probabilidad de cumplir? ¿Cuánto cuesta frente a la "
        "reducción determinista? ¿Cuánto cuesta cada punto porcentual de probabilidad ganado en cada tramo?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"plazo = {plazo}\n"
                       "dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}\n"
                       "tope, ucr = datos_proyecto['cap_duration'], datos_proyecto['ucr']"))
    celdas += apartado(1)
    celdas.append(code("tabla = proyecto.monte_carlo_crashing(dists, datos_proyecto, 'duration', tope, ucr, plazo=plazo,\n"
                       "                                       reduccion_max=8, n_samples=5000, seed=7)\n"
                       "Markdown(f'Sin acelerar: duración media simulada {tabla.loc[0, \"τ media\"]:.2f} semanas y P(τ ≤ {plazo}) = {tabla.loc[0, \"P(τ ≤ plazo)\"]:.3f}.')"))
    celdas += apartado(2)
    celdas.append(code("r_det = int(tabla.index[tabla['duración determinista'] <= plazo].min())\n"
                       "Markdown(f'Reducir las duraciones esperadas hasta {plazo} semanas exige {r_det} semana(s) de reducción y {tabla.loc[r_det, \"coste\"]:.0f} k€; '\n"
                       "         f'pero la probabilidad real de cumplir es {tabla.loc[r_det, \"P(τ ≤ plazo)\"]:.3f}: reducir las duraciones esperadas hasta el plazo '\n"
                       "         'deja una probabilidad muy inferior al 90 % que suele exigirse.')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Probabilidad de cumplir el plazo según las semanas recortadas"\n#| label: tbl-crashingMC\n'
                       "tabla.round(3)"))
    celdas += apartado(4)
    celdas.append(code("r90 = int(tabla.index[tabla['P(τ ≤ plazo)'] >= 0.9].min())\n"
                       "marginal = (tabla['coste'].diff() / (100 * tabla['P(τ ≤ plazo)'].diff())).rename('k€ por punto de probabilidad')\n"
                       "display(marginal.round(2).to_frame())\n"
                       "Markdown(f'Para el 90 % hay que recortar {r90} semanas ({tabla.loc[r90, \"coste\"]:.0f} k€) en vez de las {r_det} deterministas '\n"
                       "         f'({tabla.loc[r_det, \"coste\"]:.0f} k€): la diferencia, {tabla.loc[r90, \"coste\"] - tabla.loc[r_det, \"coste\"]:.0f} k€, es lo que cuesta '\n"
                       "         'la seguridad. Los últimos puntos de probabilidad son los más caros.')"))
    return celdas, f"duraciones triangulares; plazo de {plazo} semanas; reducción y probabilidad"


VARIANTES = [
    Variante("cur-y-ackoff", 1, "CUR a partir de costes normal y tope; Ackoff-Sasieni tres semanas con un paso de dos actividades",
             lambda rng: _base(1, rng)),
    Variante("costes-indirectos", 2, "curva coste-duración con costes indirectos y duración de coste mínimo", lambda rng: _base(2, rng)),
    Variante("penalizacion", 2, "reducir frente a pagar penalización; actividades no acelerables", lambda rng: _base(3, rng)),
    Variante("barata-no-optima", 2, "dos rutas críticas: la crítica más barata no sirve si no es común a ambas", _barata_no_optima),
    Variante("desaceleracion", 2, "reducido el proyecto, el plazo se relaja: qué paso deshacer (el más caro) y por qué la curva coste-duración es convexa", _desaceleracion),
    Variante("programacion-lineal", 3, "reducción óptima por programación lineal con la matriz de caminos frente al voraz de Ackoff-Sasieni",
             _programacion_lineal),
    Variante("cur-por-tramos", 3, "CUR creciente por tramos: Ackoff con coste marginal frente a un CUR medio; por qué funciona si el coste es convexo",
             _cur_por_tramos),
    Variante("crashing-recursos", 3, "reducir la duración con un límite de operarios: Ackoff sin recursos falla; cadena crítica con recursos, modelo con recurso y contratar frente a acelerar",
             _crashing_recursos),
    Variante("crashing-incertidumbre", 3, "reducir con duraciones triangulares: la reducción determinista hasta el plazo da poca probabilidad; coste de la seguridad",
             _crashing_incertidumbre),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
