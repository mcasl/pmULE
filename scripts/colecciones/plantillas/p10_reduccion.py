"""Colección 10 — Reducción de la duración con mínimo coste: CUR, algoritmo
de Ackoff-Sasieni, costes indirectos y duración óptima, penalizaciones."""
import contextlib
import io

import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado,
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
    celdas.append(code('#| fig-cap: "Grafo PERT con las duraciones normales y el camino crítico"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
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
        celdas.append(code('#| fig-cap: "Grafo PERT con las duraciones reducidas"\n#| label: fig-pertReducido\n'
                           "proyecto.pert(durations=duraciones_reducidas, ranksep=1)"))
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
    celdas.append(code('#| fig-cap: "Grafo PERT con las duraciones normales y las rutas críticas"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
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


VARIANTES = [
    Variante("cur-y-ackoff", 1, "CUR a partir de costes normal y tope; Ackoff-Sasieni tres semanas con un paso de dos actividades",
             lambda rng: _base(1, rng)),
    Variante("costes-indirectos", 2, "curva coste-duración con costes indirectos y duración de coste mínimo", lambda rng: _base(2, rng)),
    Variante("penalizacion", 2, "reducir frente a pagar penalización; actividades no acelerables", lambda rng: _base(3, rng)),
    Variante("barata-no-optima", 2, "dos rutas críticas: la crítica más barata no sirve si no es común a ambas", _barata_no_optima),
    Variante("desaceleracion", 2, "reducido el proyecto, el plazo se relaja: qué paso deshacer (el más caro) y por qué la curva coste-duración es convexa", _desaceleracion),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
