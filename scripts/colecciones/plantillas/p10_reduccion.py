"""Colección 10 — Reducción de la duración con mínimo coste: CUR, algoritmo
de Ackoff-Sasieni, costes indirectos y duración óptima, penalizaciones."""
import contextlib
import io

import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

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


def generar(k, rng):
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
