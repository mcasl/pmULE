"""Colección 18 — Casos de decisión: varios implicados en un proyecto interactúan (mensajes
con datos nuevos, peticiones, condiciones) y el alumno tiene que cuantificar cada mensaje
con las herramientas del curso, elegir una opción defendible y comunicarla a cada implicado."""
import contextlib
import io

import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe)
from .p17_viabilidad import (FINANZAS, _flujos, _van, _tir, _buscar_caso, _tabla_parametros, PARAMS, MODELO)

META = dict(slug="18_casos", prefijo="Caso",
            titulo="Casos de decisión con varios implicados",
            descripcion="Situaciones en las que proveedor, cliente, contratista, inspector, otros jefes de proyecto, "
                        "gerencia o el comité envían mensajes que cambian los datos o piden algo; hay que cuantificar "
                        "cada uno (calendario, Ackoff, probabilidad, recursos, valor ganado, Monte Carlo, VAN), "
                        "decidir y redactar la respuesta a cada implicado.",
            contexto=False, n_defecto=5)

# Código común que se emite en la resolución de los casos de planificación
UTIL = (
    "def proyecto_de(predecesoras, duraciones, nombres):\n"
    "    '''Tabla de datos y proyecto pmULE a partir de listas de predecesoras y duraciones.'''\n"
    "    df = pd.DataFrame({'predecessor': list(predecesoras), 'duration': list(duraciones)}, index=list(nombres))\n"
    "    df.index.name = 'activity'\n"
    "    p = PredecessorTable.from_dataframe_of_strings(df, activity='activity', predecessor='predecessor').create_project()\n"
    "    return df, p\n"
    "\n"
    "def camino_critico(p, d):\n"
    "    return '; '.join('–'.join(ruta) for ruta in p.critical_path(d).values())\n"
    "\n"
    "def resumen_prob(p, d, sd, plazo):\n"
    "    '''Duración media, desviación típica del camino crítico (la mayor si hay varios), P(T ≤ plazo) y retraso esperado.'''\n"
    "    R = p.path_matrix(dummies=False)\n"
    "    L = R @ d.reindex(R.columns)\n"
    "    V = R @ (sd.reindex(R.columns).fillna(0) ** 2)\n"
    "    T = float(L.max()); sigma = float(np.sqrt(V[L == T].max()))\n"
    "    z = (plazo - T) / sigma\n"
    "    retraso_esp = sigma * (norm.pdf(z) - z * norm.cdf(-z))      # E[máx(0, X − plazo)] con X ~ N(T, σ)\n"
    "    return T, sigma, float(norm.cdf(z)), float(retraso_esp)\n")


def _silencio():
    return contextlib.redirect_stdout(io.StringIO())


def _proy(preds, dur, nombres):
    df = pd.DataFrame({"predecessor": list(preds), "duration": list(dur)}, index=list(nombres))
    df.index.name = "activity"
    with _silencio():
        p = P.PredecessorTable.from_dataframe_of_strings(df, activity="activity", predecessor="predecessor").create_project()
    return df, p


def _T(p, d):
    with _silencio():
        return int(p.duration(d))


def _camino(p, d):
    with _silencio():
        return "; ".join("–".join(r) for r in p.critical_path(d).values())


def _prob(p, d, sd, plazo):
    with _silencio():
        R = p.path_matrix(dummies=False)
    L = R @ d.reindex(R.columns)
    V = R @ (sd.reindex(R.columns).fillna(0) ** 2)
    T = float(L.max()); sigma = float(np.sqrt(V[L == T].max()))
    z = (plazo - T) / sigma
    return T, sigma, float(norm.cdf(z)), float(sigma * (norm.pdf(z) - z * norm.cdf(-z)))


def _ackoff(p, d, tope, cur, r):
    with _silencio():
        _, pasos, dur, _ = p.ackoff(d, tope, cur, reduction=r, report=False)
    coste = float(sum(cur[a] for acts in pasos.values() for a in acts))
    return pasos, dur.reindex(d.index), coste


def _mensaje(de, texto):
    return md(f"> **De: {de}**\n> {texto}")


def _cierre(agentes):
    return md("Responda a las preguntas siguientes **antes** de decidir. En la penúltima se le pide la decisión y "
              "cómo la comunicaría a cada uno de los implicados (" + ", ".join(agentes) + ").")


# ---------------------------------------------------------------------------
# Caso 1 — La reforma del laboratorio: retraso de un proveedor, exigencia de un inspector,
# oferta de reducción del contratista, cliente que paga el retraso, tope de gerencia
# ---------------------------------------------------------------------------
LAB_NOMBRES = list("ABCDEFGHI")
LAB_PREDS = ["---", "---", "A", "B,C", "C", "C", "D,E,F", "D,E", "G,H"]
LAB_DESCR = ["Desmontaje y retirada del laboratorio antiguo", "Proyecto eléctrico y permisos",
             "Obra civil: tabiques, suelo técnico y pintura", "Instalación eléctrica",
             "Climatización y extracción de gases", "Mobiliario de laboratorio (suministro y montaje)",
             "Instalación del equipamiento científico", "Inspección de seguridad",
             "Puesta en marcha y formación del personal"]


def _datos_laboratorio(rng):
    for _ in range(500):
        d = pd.Series({"A": int(rng.choice([2, 3])), "B": int(rng.choice([3, 4])), "C": int(rng.choice([4, 5])),
                       "D": int(rng.choice([2, 3, 4])), "E": int(rng.choice([2, 3])), "F": int(rng.choice([4, 5])),
                       "G": int(rng.choice([2, 3])), "H": 1, "I": int(rng.choice([2, 3]))})
        if d["F"] <= max(d["D"], d["E"]) or d["F"] + d["G"] - max(d["D"], d["E"]) - d["H"] < 1:
            continue
        tope = d.copy(); tope["A"] = 1; tope["C"] = d["C"] - 2; tope["G"] = 1; tope["I"] = 1
        cur = pd.Series(np.inf, index=d.index, dtype=float)
        cur["A"], cur["C"], cur["G"], cur["I"] = (int(rng.choice([2, 3])), int(rng.choice([5, 6, 7])),
                                                  int(rng.choice([8, 9, 10])), int(rng.choice([4, 5])))
        sd = pd.Series({"A": 0.3, "B": 0.5, "C": float(rng.choice([0.7, 0.8, 0.9])), "D": 0.5, "E": 0.6,
                        "F": float(rng.choice([1.0, 1.2])), "G": 0.4, "H": 0.2, "I": 0.3})
        delta = int(rng.choice([2, 3]))
        alquiler = int(rng.choice([4, 5, 6]))
        df0, p0 = _proy(LAB_PREDS, d, LAB_NOMBRES)
        T0 = _T(p0, d)
        with _silencio():
            if len(p0.critical_path(d)) != 1:
                continue
        dF = d.copy(); dF["F"] += delta
        predsH = list(LAB_PREDS); predsH[7] = "D,E,G"
        _, pH = _proy(predsH, d, LAB_NOMBRES)
        df2, p2 = _proy(predsH, dF, LAB_NOMBRES)
        TF, TH, T2 = _T(p0, dF), _T(pH, d), _T(p2, dF)
        retraso = T2 - T0
        topeF = tope.copy(); topeF["F"] = dF["F"]
        opciones, duraciones = {0: 0.0}, {0: dF}
        ok = True
        for r in range(1, retraso + 1):
            pasos, dur, coste = _ackoff(p2, dF, topeF, cur, r)
            if T2 - _T(p2, dur) != r:
                ok = False
                break
            opciones[r], duraciones[r] = coste, dur
        if not ok:
            continue
        total = {r: c + alquiler * (retraso - r) for r, c in opciones.items()}
        r_opt = min(total, key=total.get)
        if not (1 <= r_opt < retraso) or sorted(total.values())[1] - total[r_opt] < 1:
            continue
        bolsa = int(opciones[r_opt]) + int(rng.integers(0, max(1, int(opciones[r_opt + 1] - opciones[r_opt]))))
        if not (opciones[r_opt] <= bolsa < opciones[r_opt + 1]):
            continue
        # plan negociado: mobiliario en dos lotes (F1 mesas en plazo, F2 resto con el retraso), inspección tras D y E
        f1 = d["F"] - 2
        nombres4 = ["A", "B", "C", "D", "E", "F1", "F2", "G", "H", "I"]
        preds4 = ["---", "---", "A", "B,C", "C", "C", "C", "D,E,F1", "D,E", "G,H,F2"]
        dur4 = pd.Series([d["A"], d["B"], d["C"], d["D"], d["E"], f1, dF["F"], d["G"], d["H"], d["I"]], index=nombres4)
        df4, p4 = _proy(preds4, dur4, nombres4)
        T4 = _T(p4, dur4)
        if T4 > T0:
            continue
        sd4 = sd.drop("F"); sd4["F1"], sd4["F2"] = sd["F"] * 0.5, sd["F"]
        with _silencio():
            cal4 = p4.calendar(dur4)
        dur5 = dur4.copy(); dur5["A"] -= 1
        return dict(d=d, tope=tope, cur=cur, sd=sd, delta=delta, alquiler=alquiler, bolsa=bolsa, T0=T0, TF=TF, TH=TH,
                    T2=T2, retraso=retraso, opciones=opciones, total=total, r_opt=r_opt, duraciones=duraciones,
                    p0=p0, p2=p2, dF=dF, predsH=predsH, topeF=topeF, f1=f1, nombres4=nombres4, preds4=preds4,
                    dur4=dur4, dur5=dur5, p4=p4, T4=T4, cal4=cal4, sd4=sd4, df0=df0)
    raise RuntimeError("laboratorio: no se encontraron datos con el dilema buscado")


def _reforma_laboratorio(rng):
    x = _datos_laboratorio(rng)
    d, tope, cur, sd = x["d"], x["tope"], x["cur"], x["sd"]
    T0, retraso, delta, alq, bolsa = x["T0"], x["retraso"], x["delta"], x["alquiler"], x["bolsa"]
    opc, total, r_opt = x["opciones"], x["total"], x["r_opt"]
    with _silencio():
        cal0 = x["p0"].calendar(d)
    semana = 1
    hol_H = int(cal0.loc["H", "H_total"])
    df = pd.DataFrame({"descripcion": LAB_DESCR, "predecessor": LAB_PREDS, "duration": d.values, "tope": tope.values,
                       "CUR": cur.values, "sd": sd.values}, index=LAB_NOMBRES)
    df.index.name = "activity"
    probs = {r: _prob(x["p2"], x["duraciones"][r], sd, T0) for r in opc}
    esp = {r: opc[r] + alq * probs[r][3] for r in opc}
    r_esp = min(esp, key=esp.get)
    T4, p4, dur4, dur5, cal4, sd4 = x["T4"], x["p4"], x["dur4"], x["dur5"], x["cal4"], x["sd4"]
    pr4 = _prob(p4, dur4, sd4, T0)
    pr5 = _prob(p4, dur5, sd4, T0)
    hol_F2 = int(cal4.loc["F2", "H_total"])
    fin_F2 = int(cal4.loc["F2", "fin_mas_temprano"])
    semana_decision = int(cal4.loc["F2", "fin_mas_temprano"]) - 1
    cA = int(cur["A"]); cI = int(cur["I"])
    reductor = "A" if d["A"] > 1 else "I"
    _acts = [a for acts in _ackoff(x["p2"], x["dF"], x["topeF"], cur, r_opt)[0].values() for a in acts]
    actos_opt = ", ".join(f"{a} ({_acts.count(a)} semanas)" if _acts.count(a) > 1 else a for a in dict.fromkeys(_acts))
    agentes = ["proveedor", "inspector", "contratista", "vicerrectora", "gerencia", "técnico de laboratorio"]

    celdas = [md("# Enunciado"),
              md(f"Usted es la persona responsable del proyecto de **reforma del laboratorio de instrumentación** de "
                 f"una escuela de ingeniería. El laboratorio tiene que estar operativo para el inicio del curso, dentro "
                 f"de **{T0} semanas**. El plan aprobado es el que se adjunta (duraciones en semanas; costes en k€). Las "
                 "duraciones tope y los costes unitarios de reducción (CUR) son los que el contratista ha ofrecido por "
                 "contrato para las actividades que ejecuta (A, C, G, I); las demás no pueden acelerarse (CUR infinito). "
                 "La última columna es la desviación típica estimada de cada duración."),
              code('#| tbl-cap: "Plan aprobado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df).rsplit("\n", 1)[0]
                   + "\ndatos_enunciado.replace(np.inf, '—')"),
              md(f"En la semana {semana} (acaba de empezar el desmontaje y todo va según el plan) recibe, el mismo día, los "
                 "siguientes mensajes."),
              _mensaje("Proveedor del mobiliario (Labtec S. L.)",
                       f"Lamentamos comunicarles que, por un problema de suministro de encimeras, la entrega e instalación "
                       f"del mobiliario (actividad F) necesitará **{delta} semanas más** de lo previsto: {d['F'] + delta} en "
                       f"lugar de {d['F']}. Podríamos entregar en la fecha prevista las mesas de trabajo, que son la parte "
                       f"principal del pedido ({x['f1']} semanas de montaje), y el resto {delta} semanas después de lo "
                       "previsto, si eso les sirve de algo. Sentimos las molestias."),
              _mensaje("Inspector de seguridad (Servicio de Prevención)",
                       "He revisado el plan que me envió. No puedo dar por válida una inspección realizada antes de que el "
                       "equipamiento científico esté instalado (actividad G): la inspección (H) debe hacerse **después de "
                       "G**, no sólo después de la instalación eléctrica y la climatización. Mantengo la duración de una semana."),
              _mensaje("Contratista (Obras Lince S. A.)",
                       "Hemos sabido lo del mobiliario. Les recordamos que el contrato contempla acelerar A, C, G e I a los "
                       "precios acordados (CUR). Si nos lo piden esta semana podemos aún acortar A, que está en ejecución, y "
                       "replanificar C. Para G e I habría que planificar turnos de tarde, con un recargo ya incluido en el CUR. "
                       "Esperamos instrucciones."),
              _mensaje("Vicerrectora de Infraestructuras (cliente)",
                       "Me dicen que el laboratorio podría no estar para el inicio del curso. Cada semana que el laboratorio "
                       f"no esté disponible tendremos que alquilar un espacio externo para las prácticas, que cuesta **{alq} k€ "
                       "por semana**. Necesito saber cuánto se va a retrasar y qué proponéis, y lo necesito mañana."),
              _mensaje("Gerencia (control presupuestario)",
                       f"El proyecto tiene una bolsa para imprevistos de **{bolsa} k€**. Cualquier gasto adicional por encima "
                       "de esa cifra requiere una modificación presupuestaria que tardaría en aprobarse al menos un mes. Los "
                       "costes del alquiler de espacios externos saldrían de otra partida (la de docencia), no de la del proyecto."),
              _mensaje("Técnico de laboratorio (usuario final)",
                       "Si hace falta, nosotros podemos ir instalando el equipamiento sobre las mesas de trabajo aunque falten "
                       "los armarios y la campana; lo hemos hecho otras veces. Y la formación (I) podría solaparse con la "
                       "instalación si nos dejan entrar antes. Lo que no podemos es empezar el curso sin la inspección."),
              _cierre(agentes)]
    celdas.append(preguntas([
        "Reconstruya el plan original (calendario, camino crítico, holguras) y después el plan tal como queda con los "
        "dos mensajes que cambian los datos: el retraso del mobiliario y la nueva exigencia del inspector. ¿Cuánto se "
        "retrasa el proyecto si no se hace nada? Separe el efecto de cada mensaje: ¿cuánto retraso aporta el proveedor, "
        "cuánto el inspector y por qué (o por qué no) se suman tal cual?",
        f"Evalúe la oferta del contratista con el algoritmo de Ackoff-Sasieni: ¿qué cuesta recuperar 1, 2, …, {retraso} "
        "semanas y qué actividades habría que acortar en cada caso? Ponga esos costes frente al coste del retraso "
        f"({alq} k€ por semana) y frente al tope de {bolsa} k€ de Gerencia. ¿Qué opción minimiza el coste total para la "
        "universidad? ¿Y cuál respeta el tope del proyecto?",
        f"Con las desviaciones típicas dadas, calcule para cada opción la probabilidad de tener el laboratorio en la "
        f"semana {T0} y el retraso esperado. ¿Cambia la respuesta anterior cuando se tiene en cuenta la incertidumbre? "
        "¿Qué opción es más robusta?",
        "Hay propuestas que no cuestan dinero: la entrega parcial del mobiliario, la inspección en dos fases y el solape "
        "que propone el técnico. Modele las que le parezcan aceptables como cambios en el cuadro de prelaciones y "
        "cuantifique cuánto plazo recuperan. ¿Qué riesgos tienen y a quién habría que preguntar antes de contar con ellas?",
        "Tome una decisión: qué combinación de reducciones y negociaciones propone, cuánto cuesta, qué plazo promete a la "
        "vicerrectora y con qué probabilidad. Redacte en tres o cuatro líneas el mensaje de respuesta a cada uno de los "
        "seis implicados.",
        "Mirando atrás: ¿qué habría tenido que hacerse en la planificación para que estos mensajes no fueran una "
        "sorpresa? Identifique al menos dos fallos de gestión de interesados y uno de gestión de riesgos."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(UTIL))
    celdas.append(code(f"plazo, coste_semana, bolsa = {T0}, {alq}, {bolsa}\n"
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "sd = datos_proyecto['sd']\n"
                       "preds = datos_proyecto['predecessor'].tolist(); nombres = datos_proyecto.index.tolist()\n"
                       "dur0 = datos_proyecto['duration']\n"
                       f"durF = dur0.copy(); durF['F'] += {delta}                                   # retraso del mobiliario\n"
                       "predsH = list(preds); predsH[nombres.index('H')] = 'D,E,G'                 # inspección tras G\n"
                       "df0, p0 = proyecto_de(preds, dur0, nombres)\n"
                       "dfF, pF = proyecto_de(preds, durF, nombres)\n"
                       "dfH, pH = proyecto_de(predsH, dur0, nombres)\n"
                       "df2, p2 = proyecto_de(predsH, durF, nombres)"))
    celdas += apartado(1)
    celdas.append(md(f"Plan original: {T0} semanas justas, con el camino crítico {_camino(x['p0'], d)}. El mobiliario (F) "
                     f"ya era crítico, y la inspección (H) tenía {hol_H} semana{'s' if hol_H != 1 else ''} de holgura."))
    celdas.append(code('#| fig-cap: "Plan original: Gantt de inicios más tempranos con flechas de prelación y holgura total"\n#| label: fig-gantt0\n'
                       "_g, _d = p0.gantt(df0, 'duration', holguras=True, tikz=True, params=dict(inner_text=None, **GANTT_PARAMS))\n"
                       "pinta_tikz(_d)"))
    celdas += fig_grafo(durations="dur0", cap="Grafo PERT del plan original con los tiempos de los nodos", label="fig-grafo0", obj="p0")
    celdas.append(code('#| tbl-cap: "Calendario del plan original"\n#| label: tbl-calendario0\n'
                       "cal0 = p0.calendar(dur0)\n"
                       "cal0[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'H_libre']]"))
    celdas.append(md("Efecto de cada mensaje por separado y de los dos juntos:"))
    celdas.append(code('#| tbl-cap: "Las cuatro situaciones"\n#| label: tbl-situaciones\n'
                       "filas = []\n"
                       f"for nombre, df, p in (('plan original', df0, p0), ('sólo el retraso del mobiliario (F = {d['F'] + delta})', dfF, pF),\n"
                       "                      ('sólo la inspección tras G', dfH, pH), ('los dos cambios', df2, p2)):\n"
                       "    T = int(p.duration(df['duration']))\n"
                       "    filas.append({'situación': nombre, 'duración': T, f'retraso sobre {plazo}': T - plazo, 'camino crítico': camino_critico(p, df['duration'])})\n"
                       "pd.DataFrame(filas).set_index('situación')"))
    celdas.append(md(f"El proveedor aporta {delta} semanas porque F era crítica: su retraso va entero al plazo. El inspector "
                     f"aporta {x['TH'] - T0}: H tenía {hol_H} semana{'s' if hol_H != 1 else ''} de holgura, pero al pasar "
                     "detrás de G deja de aprovecharla y se coloca en el camino crítico (A–C–F–G–H–I). Los efectos se suman "
                     f"({delta} + {x['TH'] - T0} = {retraso}) porque los dos cambios caen sobre la misma ruta; no se "
                     "sumarían si uno de ellos afectara a una ruta con holgura suficiente para absorberlo."))
    celdas.append(code(f'#| fig-cap: "Plan con los dos cambios: {x["T2"]} semanas"\n#| label: fig-gantt2\n'
                       "_g, _d = p2.gantt(df2, 'duration', holguras=True, tikz=True, params=dict(inner_text=None, **GANTT_PARAMS))\n"
                       "pinta_tikz(_d)"))
    celdas += apartado(2)
    celdas.append(md("Ackoff-Sasieni acorta en cada paso, una semana, el conjunto más barato de actividades que corta "
                     "todos los caminos críticos del momento (una por camino). Las actividades sin oferta del contratista "
                     "tienen CUR infinito y nunca se eligen; la duración tope de F es la nueva (el proveedor ya ha dicho lo que tarda)."))
    celdas.append(code('#| tbl-cap: "Opciones de reducción frente al coste del alquiler"\n#| label: tbl-opciones\n'
                       "tope = datos_proyecto['tope'].copy(); tope['F'] = durF['F']\n"
                       "cur = datos_proyecto['CUR'].astype(float)\n"
                       f"retraso = int(p2.duration(durF)) - plazo\n"
                       "opciones, duraciones_opcion = [], {0: durF.copy()}\n"
                       "opciones.append({'semanas recuperadas': 0, 'actividades acortadas': '—', 'sobrecoste (k€)': 0.0, 'duración': plazo + retraso})\n"
                       "for r in range(1, retraso + 1):\n"
                       "    _, pasos, dur, _ = p2.ackoff(durF, tope, cur, reduction=r, report=False)\n"
                       "    duraciones_opcion[r] = dur.reindex(durF.index)\n"
                       "    opciones.append({'semanas recuperadas': r, 'actividades acortadas': ', '.join(a for acts in pasos.values() for a in acts),\n"
                       "                     'sobrecoste (k€)': float(sum(cur[a] for acts in pasos.values() for a in acts)), 'duración': int(p2.duration(dur))})\n"
                       "tabla = pd.DataFrame(opciones).set_index('semanas recuperadas')\n"
                       "tabla['retraso'] = (tabla['duración'] - plazo).clip(lower=0)\n"
                       "tabla['alquiler (k€)'] = coste_semana * tabla['retraso']\n"
                       "tabla['coste total universidad (k€)'] = tabla['sobrecoste (k€)'] + tabla['alquiler (k€)']\n"
                       "tabla[f'dentro de la bolsa de {bolsa} k€'] = tabla['sobrecoste (k€)'] <= bolsa\n"
                       "tabla"))
    lineas = [f"- Sin hacer nada: {retraso} semanas de alquiler, {alq * retraso} k€, que además salen de la partida de docencia."]
    for r in range(1, retraso + 1):
        extra = ""
        if r == r_opt:
            extra = f": **el mínimo para la universidad** y dentro de la bolsa del proyecto"
        elif opc[r] > bolsa:
            extra = f": supera la bolsa en {opc[r] - bolsa:.0f} k€ (modificación presupuestaria, un mes de espera)"
        lineas.append(f"- Recuperar **{r}** semana{'s' if r > 1 else ''} cuesta {opc[r]:.0f} k€ y deja {retraso - r} de alquiler "
                      f"({alq * (retraso - r)} k€): {total[r]:.0f} k€ en total{extra}.")
    celdas.append(md(f"Para recuperar {r_opt} semana{'s' if r_opt != 1 else ''} el algoritmo acorta {actos_opt} (las más baratas "
                     f"por semana); G, con CUR {int(cur['G'])}, sólo entra cuando las demás agotan su margen. Lo que se ve:\n\n"
                     + "\n".join(lineas) + "\n\n"
                     "Obsérvese el conflicto entre perspectivas: para el proyecto (Gerencia) lo barato es no gastar; para la "
                     "universidad el alquiler es un coste real aunque salga de otra partida; para la vicerrectora el plazo "
                     "tiene además un coste reputacional que no está en la tabla."))
    celdas += apartado(3)
    celdas.append(md("Para cada opción se toma la duración media del camino crítico y la varianza de ese camino (suma de "
                     "las varianzas de sus actividades; si hay varios caminos críticos, la mayor), y se supone normalidad "
                     "(teorema central del límite). El retraso esperado es $E[\\max(0, T - \\text{plazo})]$."))
    celdas.append(code('#| tbl-cap: "Probabilidad de cumplir el plazo y coste esperado de cada opción"\n#| label: tbl-probabilidad\n'
                       "filas = []\n"
                       "for r, dur in duraciones_opcion.items():\n"
                       "    T, sigma, prob, retraso_esp = resumen_prob(p2, dur, sd, plazo)\n"
                       "    sobre = float(tabla.loc[r, 'sobrecoste (k€)'])\n"
                       "    filas.append({'semanas recuperadas': r, 'duración media': T, 'σ': round(sigma, 2), f'P(terminar ≤ {plazo})': round(prob, 3),\n"
                       "                  'retraso esperado (sem)': round(retraso_esp, 2), 'alquiler esperado (k€)': round(coste_semana * retraso_esp, 1),\n"
                       "                  'coste total esperado (k€)': round(sobre + coste_semana * retraso_esp, 1)})\n"
                       "pd.DataFrame(filas).set_index('semanas recuperadas')"))
    celdas.append(md(f"La desviación típica del camino crítico es de {probs[0][1]:.1f} semanas, grande comparada con las "
                     f"diferencias que se están discutiendo. Dos consecuencias: la opción «recuperar {retraso} semanas» "
                     f"promete terminar en la semana {T0} *en media*, es decir, con un {probs[retraso][2]:.0%} de probabilidad, "
                     f"no con seguridad; y en coste esperado la mejor opción es "
                     + (f"recuperar {r_esp} semana{'s' if r_esp != 1 else ''}" if r_esp else "no acortar nada")
                     + f" ({esp[r_esp]:.1f} k€ frente a {esp[retraso]:.1f} de la de {retraso})"
                     + (", la misma que sin incertidumbre" if r_esp == r_opt else f", y no la de {r_opt} que salía antes")
                     + ". Con la incertidumbre delante, las reducciones de pago compran menos de lo que parecían, y las "
                     "medidas que recortan plazo sin coste (apartado 4) ganan valor."))
    celdas += apartado(4)
    celdas.append(md("Tres propuestas aparecen en los mensajes. Modeladas como cambios del cuadro de prelaciones:\n\n"
                     "- **Inspección en dos fases.** La parte eléctrica y de climatización (lo que H comprobaba originalmente) "
                     "se inspecciona tras D y E, y tras G queda sólo una comprobación final corta. Si el inspector acepta que "
                     "esa comprobación quepa dentro de la puesta en marcha (I), H vuelve a estar en paralelo y el proyecto "
                     "recupera la semana que había añadido el inspector.\n"
                     f"- **Entrega parcial del mobiliario.** Si G puede empezar sobre las mesas (F1, {x['f1']} semanas, en plazo) y el "
                     f"resto del mobiliario (F2, {d['F'] + delta} semanas desde el fin de la obra civil) sólo tiene que estar "
                     "antes de la puesta en marcha, F deja de retrasar a G.\n"
                     "- **Solapar formación e instalación.** Es una relación comienzo-comienzo entre G e I en vez de fin-comienzo; "
                     "recorta como mucho una semana, pero la formación sobre un equipo a medio instalar es de peor calidad y el "
                     "técnico lo propone como última opción. No se cuenta con ella."))
    celdas.append(code(f'#| fig-cap: "Con inspección en dos fases y entrega parcial del mobiliario (F1 mesas, F2 resto): {T4} semanas sin gastar"\n#| label: fig-gantt4\n'
                       f"nombres4 = {x['nombres4']!r}\n"
                       f"preds4 = {x['preds4']!r}\n"
                       f"dur4 = pd.Series({[int(v) for v in dur4.values]!r}, index=nombres4)\n"
                       "sd4 = sd.drop('F'); sd4['F1'], sd4['F2'] = sd['F'] * 0.5, sd['F']\n"
                       "df4, p4 = proyecto_de(preds4, dur4, nombres4)\n"
                       "_g, _d = p4.gantt(df4, 'duration', holguras=True, tikz=True, params=dict(inner_text=None, **GANTT_PARAMS))\n"
                       "pinta_tikz(_d)"))
    celdas.append(code("cal4 = p4.calendar(dur4)\n"
                       "T4, sigma4, prob4, _ = resumen_prob(p4, dur4, sd4, plazo)\n"
                       "Markdown(f'''Duración: **{int(p4.duration(dur4))} semanas** (P(≤ {plazo}) = {prob4:.2f}); camino crítico {camino_critico(p4, dur4)}. "
                       "F2 (el resto del mobiliario) termina en la semana {int(cal4.loc['F2', 'fin_mas_temprano'])} con holgura {int(cal4.loc['F2', 'H_total'])}.''')"))
    celdas.append(md(f"Con las dos negociaciones el plan vuelve a {T4} semanas en media sin gastar un euro"
                     + (f", pero fíjese en que F2 queda crítica: cualquier semana más del proveedor vuelve a retrasar el proyecto."
                        if hol_F2 == 0 else
                        f"; F2 conserva {hol_F2} semana{'s' if hol_F2 != 1 else ''} de holgura, que es todo el margen que hay frente a un nuevo retraso del proveedor.")
                     + " Los riesgos y a quién preguntar: al inspector, si acepta la inspección en dos fases por escrito; al "
                     f"proveedor, qué garantía da de las {delta} semanas; al técnico, si instalar sobre las mesas sin campana ni "
                     "armarios es seguro y no obliga a desmontar después; y a Gerencia, si la bolsa puede reservarse para una "
                     "reducción *contingente* (acortar I al final sólo si F2 se retrasa)."))
    celdas += apartado(5)
    celdas.append(md("No hay una única respuesta correcta; sí hay respuestas mal justificadas. Una decisión defendible con los "
                     "números anteriores:\n\n"
                     f"1. Negociar hoy la inspección en dos fases y la entrega parcial del mobiliario (coste cero; recuperan "
                     f"{retraso - (T4 - T0)} de las {retraso} semanas"
                     + (" si el proveedor cumple con F2)." if hol_F2 == 0 else ").") + "\n"
                     f"2. Contratar ya la reducción de A ({cA} k€): es la más barata, A está en curso y cada semana ganada "
                     f"ahora es margen contra la incertidumbre del proveedor. Con ello la duración media baja a {pr5[0]:.0f} y la "
                     f"probabilidad de terminar en la {T0} sube del {pr4[2]:.0%} al {pr5[2]:.0%}.\n"
                     f"3. Reservar la reducción de I ({cI} k€) como medida contingente: se decide en la semana {semana_decision}, cuando "
                     f"se sepa si F2 ha llegado. Sobrecoste máximo {cA + cI} k€, "
                     + ("dentro de la bolsa; sin necesidad de modificación presupuestaria." if cA + cI <= bolsa else
                        f"que supera la bolsa de {bolsa} k€ en {cA + cI - bolsa} k€: habría que pedir a Gerencia la ampliación ya, por si acaso.")
                     + "\n"
                     f"4. Prometer a la vicerrectora la semana {T0} con probabilidad alta y la {T0 + 1} como peor caso razonable, y "
                     f"pedirle que no comprometa el alquiler externo hasta la semana {semana_decision}.\n\n"
                     "Mensajes (un ejemplo de tono y contenido):\n\n"
                     f"- *Vicerrectora:* «El retraso sin actuar sería de {retraso} semanas. Con dos cambios de secuencia ya acordados y "
                     f"una reducción de {cA} k€ recuperamos el plazo en media; mantengo una reserva de {cI} k€ para asegurar la semana "
                     f"{T0} si el mobiliario vuelve a retrasarse. Le confirmo la situación en la semana {semana_decision}; no haría falta "
                     "alquilar espacios salvo que entonces se lo indique.»\n"
                     f"- *Proveedor:* «Aceptamos la entrega en dos lotes: mesas en la fecha prevista y resto como muy tarde en la semana "
                     f"{fin_F2}. Necesito confirmación por escrito de esa fecha y un aviso inmediato si peligra: ese lote "
                     + ("está en el camino crítico.»" if hol_F2 == 0 else f"tiene sólo {hol_F2} semana{'s' if hol_F2 != 1 else ''} de margen.»") + "\n"
                     f"- *Inspector:* «Propongo inspección en dos fases: instalación eléctrica y climatización en la semana "
                     f"{int(cal4.loc['H', 'inicio_mas_temprano']) + 1}, y comprobación final del equipamiento dentro de la puesta en marcha. "
                     "¿Le parece aceptable? Si no, dígame qué necesita para poder hacerla así.»\n"
                     f"- *Contratista:* «Acorten A una semana conforme al contrato. Reserven capacidad para acortar I una semana; se lo "
                     f"confirmaré en la semana {semana_decision}.»\n"
                     f"- *Gerencia:* «Gasto comprometido {cA} k€, posible {cA + cI} k€ en total"
                     + (", dentro de la bolsa. No preveo modificación presupuestaria.»" if cA + cI <= bolsa else
                        f". Como el total podría superar la bolsa en {cA + cI - bolsa} k€, pido iniciar ya la modificación por si hiciera falta.»") + "\n"
                     f"- *Técnico:* «Instalaremos sobre las mesas desde la semana {int(cal4.loc['G', 'inicio_mas_temprano']) + 1}; la campana y los "
                     "armarios llegan después. La formación se hará completa, sin solapar. Gracias por la propuesta: es la que nos salva el plazo.»\n\n"
                     f"Otra decisión también defendible: pagar las {retraso} semanas al contratista ({opc[retraso]:.0f} k€) si la vicerrectora "
                     "considera que el inicio de curso es inamovible y acepta tramitar la modificación presupuestaria. Lo que no es "
                     f"defendible es no hacer nada, ni prometer la semana {T0} «con seguridad» con una desviación típica de {probs[0][1]:.1f} semanas."))
    celdas += apartado(6)
    celdas.append(md("- **Interesados.** El inspector no participó en la planificación: su exigencia era previsible y se habría "
                     "incorporado al cuadro de prelaciones desde el principio (identificar a quien puede imponer condiciones y "
                     "consultarle antes de fijar el plan). El técnico de laboratorio, usuario final, tenía la mejor idea del caso y "
                     "nadie se la había pedido.\n"
                     f"- **Riesgos.** F era crítica, dependía de un proveedor externo y tenía la mayor desviación típica del proyecto "
                     f"({sd['F']:.1f}): era el riesgo número uno y no tenía plan de respuesta (pedido anticipado, cláusula de entrega "
                     "parcial, seguimiento semanal con el proveedor). Un plan que termina exactamente en la fecha límite con holgura "
                     "cero en la ruta crítica es, de hecho, un plan sin margen.\n"
                     f"- **Presupuesto.** La bolsa de imprevistos se fijó sin relación con los CUR del contrato: {bolsa} k€ no cubren "
                     f"las {retraso} semanas de reducción ({opc[retraso]:.0f} k€); conocer de antemano «cuánto cuesta cada semana de "
                     "rescate» habría permitido dimensionarla."))
    return celdas, f"reforma del laboratorio: {retraso} semanas de retraso, alquiler {alq} k€/sem, bolsa {bolsa} k€"


# ---------------------------------------------------------------------------
# Caso 2 — Recursos compartidos: otra jefa de proyecto pide prestados operarios
# ---------------------------------------------------------------------------
def _programar(datos, prelaciones, calendario, limite, L):
    """Planificador en paralelo con límite por periodo: la semana s admite limite[s] (o L si no está en limite).
    Una actividad empieza en t si sus predecesoras han terminado y cabe en todas las semanas que ocuparía."""
    D = datos["duration"].astype(int); R = datos["resources"].astype(int)
    preds = prelaciones.immediate_predecessor
    orden = list(calendario["H_total"].sort_values(kind="stable").index)
    carga, inicio, fin, t = {}, {}, {}, 0
    while len(inicio) < len(D):
        for a in orden:
            if a in inicio or any(p not in fin or fin[p] > t for p in preds[a]):
                continue
            semanas = range(t + 1, t + int(D[a]) + 1)
            if all(carga.get(s, 0) + int(R[a]) <= int(limite.get(s, L)) for s in semanas):
                inicio[a], fin[a] = t, t + int(D[a])
                for s in semanas:
                    carga[s] = carga.get(s, 0) + int(R[a])
        t += 1
    inicio, fin = pd.Series(inicio).reindex(D.index), pd.Series(fin).reindex(D.index)
    esperas = {a: int(inicio[a]) - max([int(fin[p]) for p in preds[a]], default=0) for a in D.index}
    return int(fin.max()), {a: v for a, v in esperas.items() if v > 0}


PROGRAMADOR = (
    "def programar(limite, L):\n"
    "    TQPlanificador en paralelo con límite por periodo: la semana s admite limite[s] operarios (L si no está en limite).\n"
    "    Recorre las semanas; en cada una, por orden de holgura total, entra cada actividad disponible (predecesoras terminadas)\n"
    "    si cabe en el límite de TODAS las semanas que ocuparía; si no, espera. Devuelve los inicios y fines de cada actividad.TQ\n"
    "    D = datos_proyecto['duration'].astype(int); R = datos_proyecto['resources'].astype(int)\n"
    "    preds = prelaciones.immediate_predecessor\n"
    "    orden = list(calendario['H_total'].sort_values(kind='stable').index)\n"
    "    carga, inicio, fin, t = {}, {}, {}, 0\n"
    "    while len(inicio) < len(D):\n"
    "        for a in orden:\n"
    "            if a in inicio or any(p not in fin or fin[p] > t for p in preds[a]):\n"
    "                continue\n"
    "            semanas = range(t + 1, t + int(D[a]) + 1)\n"
    "            if all(carga.get(s, 0) + int(R[a]) <= int(limite.get(s, L)) for s in semanas):\n"
    "                inicio[a], fin[a] = t, t + int(D[a])\n"
    "                for s in semanas:\n"
    "                    carga[s] = carga.get(s, 0) + int(R[a])\n"
    "        t += 1\n"
    "    return pd.Series(inicio).reindex(D.index), pd.Series(fin).reindex(D.index)\n"
    "\n"
    "def proyecto_programado(inicio, fin):\n"
    "    TQTabla con actividades de espera 💤 (duración = lo que la actividad espera tras sus predecesoras) para dibujar la programación.TQ\n"
    "    preds = prelaciones.immediate_predecessor\n"
    "    filas = {}\n"
    "    for a in datos_proyecto.index:\n"
    "        ps = sorted(preds[a])\n"
    "        listo = max([int(fin[p]) for p in ps], default=0)\n"
    "        if int(inicio[a]) > listo:\n"
    "            filas['💤' + a] = {'predecessor': ','.join(ps) or '---', 'duration': int(inicio[a]) - listo, 'resources': 0}\n"
    "            ps = ['💤' + a]\n"
    "        filas[a] = {'predecessor': ','.join(ps) or '---', 'duration': int(datos_proyecto.loc[a, 'duration']), 'resources': int(datos_proyecto.loc[a, 'resources'])}\n"
    "    df = pd.DataFrame(filas).T; df.index.name = 'activity'\n"
    "    df['duration'] = df['duration'].astype(int); df['resources'] = df['resources'].astype(int)\n"
    "    return df, PredecessorTable.from_dataframe_of_strings(df, activity='activity', predecessor='predecessor').create_project()\n"
    "\n"
    "def simular_prestamo(ventana, q, L):\n"
    "    TQLímite L fuera de la ventana y L − q dentro. Devuelve (duración, esperas, dibujo del Gantt con la fila Máximo).TQ\n"
    "    limite = {s: L - q for s in range(ventana[0], ventana[1] + 1)}\n"
    "    inicio, fin = programar(limite, L)\n"
    "    df_prog, p_prog = proyecto_programado(inicio, fin)\n"
    "    T_prog = int(fin.max())\n"
    "    maximo = pd.DataFrame({s: [limite.get(s, L)] for s in range(1, T_prog + 1)}, index=['Máximo'])\n"
    "    _, dibujo = p_prog.gantt(df_prog, 'duration', 'resources', total='fila', tikz=True, extra_rows=maximo, params=dict(period_width=1, **GANTT_PARAMS))\n"
    "    esperas = {a[1:]: int(d) for a, d in df_prog['duration'].items() if a.startswith('💤')}\n"
    "    return T_prog, esperas, dibujo\n").replace("TQ", "'" * 3)


def _datos_recursos(rng):
    estado = {}

    def duracion_con(info, df, L, T0, ventana, q):
        limite = {s: L - q for s in range(ventana[0], ventana[1] + 1)}
        return _programar(df, info["tabla"], info["calendario"], limite, L)

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 3 and 4 <= i["n_rutas"] <= 8 and 10 <= i["T"] <= 16 and (i["D"] >= 2).all()):
            return False
        rec = P.recursos(rng, list(i["D"].index), hi=4)
        df = P.tabla_base(i, duration=i["D"], resources=rec)
        try:
            with _silencio():
                g, _ = i["proyecto"].copy().gantt_cargas(df.copy(), "duration", "resources", report=False, tikz=False)
        except Exception:
            return False
        total = g.data.loc["Total"].drop("H_total", errors="ignore").astype(float)
        L, T0 = int(total.max()), int(i["T"])
        q, w = int(rng.choice([1, 2])), int(rng.choice([2, 3]))
        if rec.max() > L - q or (total == L).sum() < 2:
            return False
        tabla = {}
        try:
            for a in range(1, T0 - w + 2):
                tabla[a] = duracion_con(i, df, L, T0, (a, a + w - 1), q)
        except Exception:
            return False
        retrasan = [a for a, (dur, _) in tabla.items() if dur > T0]
        libres = [a for a, (dur, _) in tabla.items() if dur == T0]
        if not retrasan or not libres:
            return False
        a_req = int(rng.choice([a for a in retrasan if a >= 3] or retrasan))   # si se puede, con tiempo para subcontratar
        posteriores = [a for a in libres if a > a_req]
        if not posteriores or not (1 <= tabla[a_req][0] - T0 <= 4):
            return False
        estado.update(rec=rec, df=df, total=total, L=L, q=q, w=w, tabla=tabla, a_req=a_req, a_alt=min(posteriores))
        return True

    for _ in range(5):
        try:
            info = P.buscar(rng, acepta, n=int(rng.choice([7, 8])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
            return info, estado
        except RuntimeError:
            continue
    raise RuntimeError("recursos compartidos: no se encontró un proyecto con el dilema buscado")


def _recursos_compartidos(rng):
    info, e = _datos_recursos(rng)
    D, T0, L, q, w = info["D"], int(info["T"]), e["L"], e["q"], e["w"]
    df, total, tabla, a_req, a_alt = e["df"], e["total"], e["tabla"], e["a_req"], e["a_alt"]
    b_req, b_alt = a_req + w - 1, a_alt + w - 1
    T_req, esperas_req = tabla[a_req]
    retraso = T_req - T0
    pen, pen2 = int(rng.choice([3, 4, 5])), int(rng.choice([6, 8, 10]))
    s = float(rng.choice([2.0, 2.5, 3.0]))
    espera_alt = a_alt - a_req
    sub_posible = a_req >= 2                      # una semana de preaviso: disponibles desde la semana 2
    cal = info["calendario"]
    picos = [int(t) for t in total.index[total == L] if a_req <= int(t) <= b_req] or [int(t) for t in total.index[total == L]]
    criticas = [a for a in D.index if cal.loc[a, "H_total"] == 0]
    en_ventana = [a for a in D.index if cal.loc[a, "inicio_mas_temprano"] < b_req and cal.loc[a, "fin_mas_temprano"] > a_req - 1]
    coste = {"negarse": pen2 * w, "prestar como se pide": pen * retraso, "prestar más tarde": pen2 * espera_alt,
             "subcontratar": s * q * w if sub_posible else np.inf}
    mejor_empresa = min(coste, key=coste.get)
    agentes = ["jefa del proyecto Delta", "director de operaciones", "cliente", "jefe de obra", "recursos humanos"]
    plural = "operarios" if q > 1 else "operario"

    celdas = [md("# Enunciado"),
              md(f"Usted dirige un proyecto de instalaciones cuyo cuadro de prelaciones, duraciones (semanas) y operarios "
                 f"necesarios por actividad se adjunta. La empresa le ha asignado una cuadrilla fija de **{L} operarios**, "
                 f"dimensionada justo para el pico de carga del plan de inicios más tempranos. El contrato con el cliente fija "
                 f"la entrega en la semana {T0} con una penalización de **{pen} k€ por semana** de retraso. El proyecto empieza "
                 "el lunes que viene (semana 1). El viernes recibe estos mensajes."),
              code('#| tbl-cap: "Plan del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Jefa del proyecto Delta",
                       f"Tengo un problema serio: necesito **{q} {plural} de tu cuadrilla durante las semanas {a_req} a {b_req}** "
                       f"(ambas incluidas) para un trabajo que no puedo retrasar. Antes de la semana {a_req} no me sirven. Si no "
                       f"los tengo, mi proyecto se retrasa esas {w} semanas y la penalización de mi cliente es de {pen2} k€ por "
                       "semana; cada semana que tarden en llegar a partir de entonces es una semana de retraso para mí. "
                       "¿Me los puedes dejar?"),
              _mensaje("Director de operaciones",
                       "Lo del proyecto Delta es prioritario para la empresa: es un cliente estratégico y la penalización es alta. "
                       "Entiendo que tu proyecto también tiene plazo. Dime cuánto te cuesta prestarlos y busquemos la opción más "
                       "barata para la empresa en conjunto, no para cada proyecto por separado."),
              _mensaje("Cliente",
                       f"Les recordamos que la fecha de entrega (semana {T0}) está ligada a la apertura de nuestras instalaciones y "
                       f"no es negociable. La penalización de {pen} k€ por semana se aplicará desde el primer día de retraso."),
              _mensaje("Jefe de obra",
                       f"He mirado el plan. En {'las semanas' if len(picos) > 1 else 'la semana'} {', '.join(map(str, picos))} vamos al "
                       f"máximo de gente y la ventana que pide Delta {'las' if len(picos) > 1 else 'la'} pisa; si nos quitan {q} {plural} "
                       f"ahí, algo se retrasa. En cambio, a partir de la semana {a_alt} tenemos más margen: ahí podríamos prestarlos sin "
                       "que el plazo se mueva, si a Delta le vale."),
              _mensaje("Recursos humanos",
                       f"Podemos subcontratar operarios a {s:.1f} k€ por operario y semana, pero necesitamos **una semana de "
                       "preaviso** para tenerlos en obra: como muy pronto, el lunes de la semana 2. Decidid pronto."),
              _cierre(agentes)]
    celdas.append(preguntas([
        f"Dibuje el Gantt de inicios más tempranos con las cargas y el diagrama de cargas. ¿En qué semanas se alcanza el "
        f"máximo de {L} operarios y qué actividades coinciden en ellas? ¿Cuáles son críticas?",
        f"Simule el préstamo de {q} {plural} en las semanas {a_req}–{b_req} con un planificador en paralelo (como Wiest-Levy) con "
        f"un límite de recursos por periodo ({L} fuera de la ventana, {L - q} dentro). ¿Qué actividades esperan, cuánto se "
        "alarga el proyecto y cuánto cuesta en penalizaciones?",
        f"Repita la simulación para todas las ventanas posibles de {w} semanas consecutivas. ¿En cuáles se puede prestar sin "
        "retrasar el proyecto? Explique el resultado a partir de las cargas y las holguras. ¿Tiene razón el jefe de obra?",
        "Construya la tabla de opciones (negarse, prestar como se pide, prestar en la ventana del jefe de obra, subcontratar) "
        "con el coste de cada una para su proyecto, para el proyecto Delta y para la empresa. ¿Qué opción minimiza el coste "
        "de la empresa? ¿Coincide con la que minimiza el de su proyecto?",
        "Tome una decisión y redacte en tres o cuatro líneas el mensaje de respuesta a cada implicado.",
        "¿Qué fallo de organización revela el caso y cómo se evitaría la próxima vez (reparto de recursos entre proyectos, "
        "reservas, prioridades)?"]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"plazo, penalizacion, penalizacion_delta, coste_sub, q, ventana = {T0}, {pen}, {pen2}, {s}, {q}, ({a_req}, {b_req})\n"
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])"))
    celdas.append(md("El préstamo es un límite de recursos *por periodo*. El algoritmo de asignación de la librería (`asignar`, "
                     "Wiest-Levy) no interrumpe actividades en curso, así que cuando el límite baja a mitad de una actividad no "
                     "puede respetarlo; se usa aquí un planificador en paralelo equivalente que sólo deja empezar una actividad si "
                     "cabe en todas las semanas que va a ocupar."))
    celdas.append(code(PROGRAMADOR))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Gantt de inicios más tempranos con los operarios de cada actividad; fila Total, carga por semana"\n#| label: fig-ganttCargas\n'
                       "gantt_cargas, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas con la cuadrilla disponible"\n#| label: fig-cargas\n'
                       "cargas = gantt_cargas.data.loc['Total'].drop('H_total').astype(float)\n"
                       "L = int(cargas.max())\n"
                       "plot_cargas(gantt_df=gantt_cargas, limite_maximo_recursos=[L])"))
    celdas.append(code("picos = [int(t) for t in cargas.index[cargas == L]]\n"
                       "en_pico = {t: [a for a in datos_proyecto.index if calendario.loc[a, 'inicio_mas_temprano'] < t <= calendario.loc[a, 'fin_mas_temprano']] for t in picos}\n"
                       "criticas = list(calendario.index[calendario['H_total'] == 0])\n"
                       "display(calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total', 'H_libre']])\n"
                       "Markdown(f'''Duración {int(proyecto.duration(datos_proyecto[\"duration\"]))} semanas; críticas: {\", \".join(criticas)}. "
                       "El máximo de {L} operarios se alcanza en las semanas {\", \".join(str(t) for t in picos)}; en ellas coinciden "
                       "{\"; \".join(f\"semana {t}: {', '.join(acts)}\" for t, acts in en_pico.items())}.''')"))
    celdas.append(md(f"El plan de inicios más tempranos satura la cuadrilla en las semanas {', '.join(map(str, picos))}. Prestar "
                     f"operarios en una de esas semanas obliga a aplazar alguna actividad; que eso retrase el proyecto depende de "
                     "si la actividad aplazada tiene holgura suficiente (y de si al aplazarla arrastra a otras hasta agotar la suya)."))
    celdas += apartado(2)
    celdas.append(code(f'#| fig-cap: "Programación con {L - q} operarios en las semanas {a_req}–{b_req} y {L} en las demás (fila Máximo); las filas 💤 son las esperas"\n#| label: fig-ganttPrestamo\n'
                       "T_prestamo, esperas, dibujo = simular_prestamo(ventana, q, L)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("Markdown(f'''Con el préstamo el proyecto pasa de {plazo} a **{T_prestamo} semanas** ({T_prestamo - plazo} de retraso, "
                       "{penalizacion * (T_prestamo - plazo)} k€ de penalización). Esperas introducidas por el algoritmo: "
                       "{\", \".join(f\"{a} ({v} sem.)\" for a, v in esperas.items()) or \"ninguna\"}.''')"))
    celdas.append(md(f"El planificador recorre las semanas; en cada una ordena las actividades disponibles por holgura total y "
                     f"deja empezar las que caben en el límite de todas las semanas que ocuparían. En la ventana {a_req}–{b_req} caben "
                     f"{L - q} operarios y coinciden {', '.join(en_ventana)}: espera{'n' if len(esperas_req) != 1 else ''} "
                     f"{', '.join(f'{a} ({v} sem.)' for a, v in esperas_req.items())}, y el retraso de {retraso} semana"
                     f"{'s' if retraso != 1 else ''} es lo que de esa espera no absorben las holguras."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Duración del proyecto según la ventana del préstamo"\n#| label: tbl-ventanas\n'
                       f"w = {w}\n"
                       "filas = []\n"
                       "for a in range(1, plazo - w + 2):\n"
                       "    T_a, esperas_a, _ = simular_prestamo((a, a + w - 1), q, L)\n"
                       "    filas.append({'semanas': f'{a}–{a + w - 1}', 'carga antes del préstamo': ', '.join(str(int(cargas[t])) for t in range(a, a + w)),\n"
                       "                  'duración': T_a, 'retraso': T_a - plazo, 'esperas': ', '.join(f'{x} ({v})' for x, v in esperas_a.items()) or '—'})\n"
                       "ventanas = pd.DataFrame(filas).set_index('semanas')\n"
                       "ventanas"))
    sin_retraso = [f"{a}–{a + w - 1}" for a, (dur, _) in tabla.items() if dur == T0]
    celdas.append(md(f"Se puede prestar sin mover el plazo en las ventanas {', '.join(sin_retraso)}: o la carga de esas semanas "
                     f"queda por debajo de {L - q} sin tocar nada, o lo que hay que aplazar tiene holgura de sobra. El jefe de "
                     f"obra tiene razón: desde la semana {a_alt} el préstamo sale gratis en plazo. La ventana que pide Delta "
                     f"({a_req}–{b_req}) cuesta {retraso} semana{'s' if retraso != 1 else ''} porque cae sobre el pico y lo que se "
                     "aplaza acaba empujando a una actividad crítica."))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Opciones y su coste (k€) para cada parte"\n#| label: tbl-opciones\n'
                       f"T_alt, _, _ = simular_prestamo(({a_alt}, {b_alt}), q, L)\n"
                       f"espera_delta = {a_alt} - ventana[0]\n"
                       f"subcontratar_posible = ventana[0] >= 2     # una semana de preaviso\n"
                       "opciones = pd.DataFrame([\n"
                       "    {'opción': 'negarse', 'mi proyecto': 0, 'proyecto Delta': penalizacion_delta * w, 'detalle': f'Delta se retrasa {w} semanas'},\n"
                       "    {'opción': f'prestar en {ventana[0]}–{ventana[1]}', 'mi proyecto': penalizacion * (T_prestamo - plazo), 'proyecto Delta': 0, 'detalle': f'mi proyecto se retrasa {T_prestamo - plazo}'},\n"
                       f"    {{'opción': 'prestar en {a_alt}–{b_alt}', 'mi proyecto': penalizacion * (T_alt - plazo), 'proyecto Delta': penalizacion_delta * espera_delta, 'detalle': f'Delta espera {{espera_delta}} semana' + ('s' if espera_delta != 1 else '')}},\n"
                       "    {'opción': f'subcontratar {q} para Delta', 'mi proyecto': 0, 'proyecto Delta': coste_sub * q * w if subcontratar_posible else np.inf,\n"
                       "     'detalle': 'preaviso suficiente' if subcontratar_posible else 'no llegan a tiempo (preaviso de 1 semana)'},\n"
                       "]).set_index('opción')\n"
                       "opciones.insert(2, 'empresa', opciones['mi proyecto'] + opciones['proyecto Delta'])\n"
                       "opciones"))
    expl_sub = (f"subcontratar {q} {plural} para Delta cuesta {s * q * w:.1f} k€ y llega a tiempo (la ventana empieza en la semana "
                f"{a_req}, hay una semana de preaviso)" if sub_posible else
                f"subcontratar no llega a tiempo (la ventana empieza en la semana {a_req} y hace falta una de preaviso), así que no es una opción")
    celdas.append(md(f"Para la empresa la opción más barata es **{mejor_empresa}** ({coste[mejor_empresa]:.1f} k€); {expl_sub}. "
                     f"Para mi proyecto lo mejor es negarse o prestar en la ventana del jefe de obra (coste cero), pero la primera carga "
                     f"a Delta con {pen2 * w} k€ y la segunda con {pen2 * espera_alt} k€ de espera. El director de operaciones pide la "
                     "columna «empresa»: ahí está el conflicto entre la optimización local (cada jefe de proyecto defiende su plazo) y la "
                     "global."))
    celdas += apartado(5)
    if mejor_empresa == "prestar más tarde":
        decision = (f"prestar los {q} {plural} desde la semana {a_alt} (gratis para mi plazo) y proponer a Delta que absorba las "
                    f"{espera_alt} semanas de espera, o que pida a RR. HH. la subcontratación para cubrir esas semanas si le sale más barato")
    elif mejor_empresa == "subcontratar":
        decision = (f"no prestar en {a_req}–{b_req}, pedir hoy a RR. HH. la subcontratación de {q} {plural} para Delta "
                    f"({s * q * w:.1f} k€, mucho menos que cualquier penalización) y ofrecer además los míos desde la semana {a_alt} por si fallara")
    elif mejor_empresa == "prestar como se pide":
        decision = (f"prestar en {a_req}–{b_req} asumiendo {retraso} semana{'s' if retraso != 1 else ''} de retraso ({pen * retraso} k€), "
                    "que es lo más barato para la empresa, dejando por escrito al director de operaciones que la penalización se "
                    "imputa a la decisión de empresa y no a la gestión del proyecto")
    else:
        decision = f"negarse ({pen2 * w} k€ para Delta) sólo si ninguna de las otras es viable, que no es el caso"
    sem = lambda n: f"{n} semana{'s' if n != 1 else ''}"
    if mejor_empresa == "prestar como se pide":
        msg_jefa = (f"Te presto {q} {plural} en {a_req}–{b_req}. A mi proyecto le cuesta {sem(retraso)} ({pen * retraso} k€), menos que "
                    "tu penalización; lo dejo por escrito con el director de operaciones.")
        msg_cliente = (f"Les comunicamos un posible retraso de {sem(retraso)} sobre la semana {T0} por una reasignación de recursos "
                       "decidida por la empresa; asumimos la penalización contractual y les confirmaremos la fecha en cuanto se concrete.")
        msg_rrhh = "Por ahora no hace falta subcontratar; os aviso si cambia."
    elif mejor_empresa == "prestar más tarde":
        msg_jefa = (f"En {a_req}–{b_req} prestar {q} {plural} me cuesta {sem(retraso)} de plazo ({pen * retraso} k€). Te los doy sin coste "
                    f"desde la semana {a_alt}: esperar {sem(espera_alt)} te cuesta {pen2 * espera_alt} k€, y sigue siendo lo más barato para la empresa.")
        msg_cliente = f"Mantenemos la entrega en la semana {T0}."
        msg_rrhh = "Por ahora no hace falta subcontratar; os aviso si cambia."
    elif mejor_empresa == "subcontratar":
        msg_jefa = (f"En {a_req}–{b_req} prestar {q} {plural} me cuesta {sem(retraso)} de plazo ({pen * retraso} k€). Lo más barato es que "
                    f"RR. HH. subcontrate {q} para ti ({s * q * w:.1f} k€); lo pido hoy. Si fallara, te presto los míos desde la semana {a_alt}.")
        msg_cliente = f"Mantenemos la entrega en la semana {T0}."
        msg_rrhh = f"Necesito {q} {plural} subcontratados para las semanas {a_req}–{b_req}; confirmad hoy para respetar el preaviso."
    else:
        msg_jefa = f"No puedo prestarlos en {a_req}–{b_req} sin retrasar mi entrega; lo elevamos al director de operaciones."
        msg_cliente = f"Mantenemos la entrega en la semana {T0}."
        msg_rrhh = "Por ahora no hace falta subcontratar; os aviso si cambia."
    celdas.append(md(f"Una decisión defendible: {decision}. Mensajes:\n\n"
                     f"- *Jefa de Delta:* «{msg_jefa}»\n"
                     "- *Director de operaciones:* «Adjunto la tabla con el coste de cada opción para los dos proyectos y para la empresa; "
                     f"la más barata en conjunto es {mejor_empresa} ({coste[mejor_empresa]:.1f} k€). Si decides otra, pido que la "
                     "penalización resultante se registre como decisión de empresa.»\n"
                     f"- *Cliente:* «{msg_cliente}»\n"
                     f"- *Jefe de obra:* «Tenías razón con la ventana {a_alt}–{b_alt}; va en la propuesta como alternativa. Prepara el "
                     "plan de personal con los dos escenarios.»\n"
                     f"- *RR. HH.:* «{msg_rrhh}»"))
    celdas += apartado(6)
    celdas.append(md("El fallo no es de cálculo sino de organización: dos proyectos comparten un recurso sin que nadie mire el "
                     "perfil de carga *conjunto*. Lo que lo evita: un diagrama de cargas de la cartera (suma de los proyectos) "
                     "revisado cada semana por el director de operaciones; reglas explícitas de prioridad entre proyectos antes de que "
                     "surja el conflicto (no durante); una reserva de capacidad o un acuerdo marco de subcontratación con preaviso "
                     "corto; y planes nivelados, no de inicios más tempranos, que rebajan los picos y dejan margen para prestar. "
                     "Y, por supuesto, que la petición llegue con más de un fin de semana de antelación."))
    return celdas, f"recursos compartidos: préstamo de {q} {plural} en {a_req}–{b_req} ({retraso} sem. de retraso) frente a {a_alt}–{b_alt}"


# ---------------------------------------------------------------------------
# Caso 3 — El comité y el valor ganado: un avance declarado que no coincide con el real
# ---------------------------------------------------------------------------
def _datos_comite(rng):
    estado = {}

    def indices(BAC, PV, perf, AC):
        EV = float((BAC * perf / 100).sum()); return EV, EV / float(AC.sum()), EV / float(PV.sum())

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22 and i["n_criticas"] == 1):
            return False
        D, cal, T = i["D"], i["calendario"], int(i["T"])
        BAC = pd.Series(rng.integers(1, 7, size=len(D)) * 10, index=D.index, name="BAC")
        t = int(round(0.5 * T))
        candidatas = [a for a in D.index if cal.loc[a, "H_total"] == 0 and cal.loc[a, "inicio_mas_temprano"] + 1 <= t <= cal.loc[a, "fin_mas_temprano"] - 1 and D[a] >= 3]
        if not candidatas:
            return False
        X = str(rng.choice(candidatas))
        frac = ((t - cal["inicio_mas_temprano"]) / D).clip(0, 1)
        PV = BAC * frac
        perf = (frac * 100).round().astype(int)
        ruido = pd.Series(rng.choice([-5, 0, 0, 5], size=len(D)), index=D.index)
        perf_rep = perf.where((perf == 0) | (perf == 100), (perf + ruido).clip(5, 95))
        g = pd.Series(rng.choice([0.95, 1.0, 1.0, 1.05], size=len(D)), index=D.index)
        AC = (PV * g).round().astype(int)
        f = float(rng.choice([0.4, 0.5, 0.6]))
        perf_real = perf_rep.copy(); perf_real[X] = int(round(perf_rep[X] * f))
        if perf_real[X] < 10 or perf_rep[X] < 50:
            return False
        AC[X] = int(round(PV[X] * float(rng.choice([1.0, 1.1]))))
        EV_rep, CPI_rep, SPI_rep = indices(BAC, PV, perf_rep, AC)
        EV_real, CPI_real, SPI_real = indices(BAC, PV, perf_real, AC)
        if not (CPI_rep >= 0.97 and SPI_rep >= 0.97 and CPI_real <= 0.90 and SPI_real <= 0.90):
            return False
        trans = (t - cal["inicio_mas_temprano"]).clip(lower=0).clip(upper=D).astype(int)
        estado.update(BAC=BAC, t=t, X=X, PV=PV, perf_rep=perf_rep, perf_real=perf_real, AC=AC, trans=trans,
                      EV_rep=EV_rep, CPI_rep=CPI_rep, SPI_rep=SPI_rep, EV_real=EV_real, CPI_real=CPI_real, SPI_real=SPI_real)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    return info, estado


def _comite_valor_ganado(rng):
    info, e = _datos_comite(rng)
    D, T, cal = info["D"], int(info["T"]), info["calendario"]
    BAC, t, X = e["BAC"], e["t"], e["X"]
    BAC_total = float(BAC.sum()); PV_t = float(e["PV"].sum()); AC_t = float(e["AC"].sum())
    pen = int(rng.choice([5, 8, 10]))
    df = P.tabla_base(info, duration=D, BAC=BAC)
    seg = pd.DataFrame({"actual_duration": e["trans"], "AC": e["AC"], "performed": e["perf_rep"]})
    seg.index.name = "activity"
    rep, real = int(e["perf_rep"][X]), int(e["perf_real"][X])
    semanas_X = int(e["trans"][X]); dX = int(D[X])
    dX_est = semanas_X / (real / 100)
    retraso_ritmo = dX_est - dX                       # si se mantiene el ritmo actual
    retraso_X = semanas_X - real / 100 * dX           # si el resto se hace al ritmo planificado
    sem_rojo = lambda v: "verde" if v >= 0.95 else ("ámbar" if v >= 0.85 else "rojo")
    EAC_rep, EAC_real = BAC_total / e["CPI_rep"], BAC_total / e["CPI_real"]
    T_rep, T_real = T / e["SPI_rep"], T / e["SPI_real"]
    sobrepago = float(BAC[X]) * (rep - real) / 100
    certificado = float(BAC[X]) * rep / 100
    sucesoras = sorted(a for a, ps in info["preds"].items() if X in ps)
    agentes = ["contratista", "jefe de equipo", "comité de dirección", "dirección financiera", "cliente"]

    celdas = [md("# Enunciado"),
              md(f"Usted dirige, por parte de la propiedad, un proyecto ejecutado por un contratista principal. El cuadro "
                 "de prelaciones, las duraciones planificadas (semanas) y el presupuesto de cada actividad (BAC, k€) se "
                 f"adjuntan; el coste planificado de cada actividad se reparte uniformemente en su duración. El plazo "
                 f"contractual es de {T} semanas, con una penalización de {pen} k€ por semana de retraso. Al contratista se le "
                 "paga por certificaciones mensuales proporcionales al porcentaje de avance declarado de cada actividad."),
              code('#| tbl-cap: "Planificación del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Al final de la semana {t} el contratista envía su informe de avance (semanas trabajadas, coste incurrido "
                 "en k€ y porcentaje realizado de cada actividad) y, el mismo día, llegan los mensajes siguientes."),
              code(f'#| tbl-cap: "Informe de avance del contratista (final de la semana {t})"\n#| label: tbl-seguimiento\n'
                   + literal_dataframe(seg, nombre="seguimiento")),
              _mensaje("Contratista",
                       f"Adjuntamos el informe de avance de la semana {t}. El proyecto marcha según lo previsto y dentro de "
                       f"presupuesto. Solicitamos la certificación correspondiente, que incluye el {rep} % de la actividad {X} "
                       f"({certificado:.1f} k€)."),
              _mensaje("Jefe de equipo (supervisión de obra)",
                       f"No estoy de acuerdo con el {rep} % de {X}. He estado en obra esta semana: de lo que hay que hacer en {X} "
                       f"yo diría que está hecho **alrededor del {real} %**, aunque las horas y el gasto sí sean los que dicen. "
                       "El resto de actividades me cuadra."),
              _mensaje("Comité de dirección",
                       "Para la reunión del lunes necesitamos un semáforo (verde/ámbar/rojo) de plazo y de coste, la previsión "
                       "de coste final y la fecha de terminación prevista. Un número, no una explicación."),
              _mensaje("Dirección financiera",
                       "Si certificamos avance que no está ejecutado estamos adelantando dinero sin garantía; si retenemos sin "
                       "motivo documentado, el contratista puede reclamar intereses de demora. Necesito un criterio."),
              _mensaje("Cliente",
                       f"Recordamos que el hito de entrega en la semana {T} está vinculado a nuestros compromisos; cualquier "
                       "desviación debe comunicarse por escrito en cuanto se conozca."),
              _cierre(agentes)]
    celdas.append(preguntas([
        "Con los datos del informe del contratista, dibuje los Gantt del valor planificado, del valor ganado y del coste real, "
        "calcule PV, EV, AC, CV, SV, CPI y SPI y asigne el semáforo que pide el comité.",
        f"Repita el cálculo sustituyendo el avance de {X} por la estimación del jefe de equipo. ¿Qué indicadores cambian y "
        "cuánto? ¿Por qué un único porcentaje mueve tanto los índices?",
        "Para las dos versiones, estime el coste final (EAC = BAC / CPI), la variación al final (VAC), el TCPI y la duración "
        f"final (T / SPI). Añada una estimación directa del retraso a partir del ritmo real de {X} (es crítica) y el coste de "
        "la penalización asociada.",
        f"¿Qué certificaría este mes y con qué criterio? Cuantifique el sobrepago que supondría certificar el {rep} % y "
        "proponga cómo verificar el avance físico (hitos, unidades de obra, reglas 0/100 o 50/50).",
        "Decida qué comunica al comité y redacte en tres o cuatro líneas la respuesta a cada implicado.",
        "¿Qué debilidades del sistema de seguimiento revela el caso (quién mide el avance, cómo se liga al pago, qué "
        "verificación independiente hay) y cómo las corregiría en el contrato y en el procedimiento?"]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"semana_control, plazo, penalizacion, actividad_discutida = {t}, {T}, {pen}, {X!r}\n"
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "BAC_total = float(datos_proyecto['BAC'].sum())\n"
                       "calendario_plan = proyecto.calendar(datos_proyecto['duration'])\n"
                       "def indicadores(seguimiento):\n"
                       "    datos_vg = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                       "    detalle = detalle_valor_ganado(datos_vg, calendario_plan, semana_control)\n"
                       "    PV, EV, AC = float(detalle['PV_i'].sum()), float(detalle['EV_i'].sum()), float(detalle['AC'].sum())\n"
                       "    return detalle, pd.Series({'PV': PV, 'EV': EV, 'AC': AC, 'CV': EV - AC, 'SV': EV - PV, 'CPI': EV / AC, 'SPI': EV / PV})\n"
                       "def semaforo(indice):\n"
                       "    return 'verde' if indice >= 0.95 else ('ámbar' if indice >= 0.85 else 'rojo')"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    celdas += apartado(1)
    params = ("params=dict(period_width=1, row_height=0.7, activity_relative_height=0.6, **GANTT_PARAMS,\n"
              "                                       critical_background_color='white!90!green', regular_background_color='white!90!green',\n"
              "                                       regular_text_color='black', critical_text_color='black')")
    celdas.append(code("datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                       "vg = EarnedValue(proyecto)\n"
                       "gantts = vg.calcula_gantts(datos_valor_ganado, planned_durations_label='duration',\n"
                       "                           actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',\n"
                       "                           percentage_complete_label='performed', tikz=True,\n"
                       "                           " + params + ")"))
    celdas.append(code('#| fig-cap: "PV: valor planificado"\n#| label: fig-ganttPV\npinta_tikz(gantts[\'gantt_PV\'])'))
    celdas.append(code('#| fig-cap: "EV: valor ganado según el informe del contratista"\n#| label: fig-ganttEV\npinta_tikz(gantts[\'gantt_EV\'])'))
    celdas.append(code('#| fig-cap: "AC: coste real"\n#| label: fig-ganttAC\npinta_tikz(gantts[\'gantt_AC\'])'))
    celdas.append(code('#| tbl-cap: "PV, EV y AC actividad a actividad con el informe del contratista"\n#| label: tbl-detalleContratista\n'
                       "detalle_rep, ind_rep = indicadores(seguimiento)\n"
                       "display(detalle_rep[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']])\n"
                       "ind_rep.round(3).to_frame('informe del contratista').T"))
    celdas.append(code("Markdown(f'''Semáforo con el informe: plazo **{semaforo(ind_rep[\"SPI\"])}** (SPI = {ind_rep[\"SPI\"]:.2f}), "
                       "coste **{semaforo(ind_rep[\"CPI\"])}** (CPI = {ind_rep[\"CPI\"]:.2f}). Tal como lo cuenta el contratista, el proyecto va bien.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Lo mismo con la estimación del jefe de equipo para la actividad discutida"\n#| label: tbl-detalleReal\n'
                       f"seguimiento_real = seguimiento.copy(); seguimiento_real.loc[actividad_discutida, 'performed'] = {real}\n"
                       "detalle_real, ind_real = indicadores(seguimiento_real)\n"
                       "comparacion = pd.DataFrame({'informe del contratista': ind_rep, 'estimación del jefe de equipo': ind_real})\n"
                       "comparacion['diferencia'] = comparacion.iloc[:, 1] - comparacion.iloc[:, 0]\n"
                       "comparacion.round(3)"))
    celdas.append(code("BAC_X = float(datos_proyecto.loc[actividad_discutida, 'BAC'])\n"
                       f"Markdown(f'''Sólo cambia EV, en {{ind_rep[\"EV\"] - ind_real[\"EV\"]:.1f}} k€ (= {{BAC_X:.0f}} × ({rep} − {real}) %), "
                       "pero como PV y AC no se mueven, los dos índices caen a la vez: SPI = {ind_real[\"SPI\"]:.2f} y CPI = {ind_real[\"CPI\"]:.2f}; "
                       "semáforo de plazo **{semaforo(ind_real[\"SPI\"])}** y de coste **{semaforo(ind_real[\"CPI\"])}**.''')"))
    celdas.append(md(f"Un único porcentaje mueve tanto los índices porque {X} es una actividad grande ({int(BAC[X])} k€ de BAC, "
                     f"el {BAC[X] / BAC_total:.0%} del presupuesto) y porque a mitad de proyecto el valor ganado acumulado es pequeño: "
                     f"{sobrepago:.0f} k€ de valor «declarado y no hecho» pesan mucho sobre {e['EV_real']:.0f} k€ realmente ganados. El valor "
                     "ganado es tan bueno como la medida del avance físico que lo alimenta."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Previsiones de coste y plazo con las dos versiones"\n#| label: tbl-previsiones\n'
                       "duracion_plan = int(proyecto.duration(datos_proyecto['duration']))\n"
                       "prev = pd.DataFrame({nombre: {'EAC = BAC / CPI': BAC_total / ind['CPI'], 'VAC = BAC − EAC': BAC_total - BAC_total / ind['CPI'],\n"
                       "                              'TCPI = (BAC − EV) / (BAC − AC)': (BAC_total - ind['EV']) / (BAC_total - ind['AC']),\n"
                       "                              'duración ≈ T / SPI': duracion_plan / ind['SPI'],\n"
                       "                              'penalización ≈ (T/SPI − T) × pen': max(0.0, duracion_plan / ind['SPI'] - plazo) * penalizacion}\n"
                       "                     for nombre, ind in (('informe del contratista', ind_rep), ('estimación del jefe de equipo', ind_real))})\n"
                       "prev.round(2)"))
    celdas.append(code("semanas_trabajadas = int(seguimiento.loc[actividad_discutida, 'actual_duration'])\n"
                       "d_plan = int(datos_proyecto.loc[actividad_discutida, 'duration'])\n"
                       f"semanas_ganadas = {real} / 100 * d_plan\n"
                       "retraso_directo = semanas_trabajadas - semanas_ganadas                 # el resto, al ritmo planificado\n"
                       f"retraso_ritmo = semanas_trabajadas / ({real} / 100) - d_plan           # el resto, al ritmo actual\n"
                       f"Markdown(f'''Estimación directa: {{actividad_discutida}} lleva {{semanas_trabajadas}} semanas y está al {real} %, es decir, ha ganado "
                       "{semanas_ganadas:.1f} semanas de las {d_plan}: lleva **{retraso_directo:.1f} semanas de retraso** aunque el resto se haga al ritmo "
                       "planificado, y {retraso_ritmo:.1f} si se mantiene el ritmo actual. Al ser crítica (holgura 0), ese retraso va entero al plazo "
                       "si no se corrige: entre {retraso_directo * penalizacion:.0f} y {retraso_ritmo * penalizacion:.0f} k€ de penalización.''')"))
    celdas.append(md(f"Con el informe, el proyecto termina en {T_rep:.1f} semanas y cuesta {EAC_rep:.0f} k€; con la estimación del jefe "
                     f"de equipo, {T_real:.1f} semanas y {EAC_real:.0f} k€ (VAC {BAC_total - EAC_real:.0f}). La estimación directa por el "
                     f"avance de {X} ({retraso_X:.1f} semanas de retraso como mínimo, {retraso_ritmo:.1f} si no cambia el ritmo) es más "
                     "fiable que T/SPI, que mezcla actividades críticas y no críticas y tiende a 1 al final del proyecto aunque se termine "
                     f"tarde. Para el comité: {sem_rojo(e['SPI_real'])} en plazo si se cree al jefe de equipo, y no se puede dar verde con "
                     "una discrepancia así sin resolver."))
    celdas += apartado(4)
    celdas.append(md(f"Certificar el {rep} % de {X} supone pagar {certificado:.1f} k€ por esa actividad, de los que {sobrepago:.1f} k€ "
                     f"corresponderían a trabajo no ejecutado según el jefe de equipo. El criterio defendible frente a Dirección "
                     f"financiera: certificar lo verificado ({real} %, {float(BAC[X]) * real / 100:.1f} k€) y dejar documentada la "
                     "discrepancia con una medición conjunta en obra esta misma semana; si la medición da la razón al contratista se "
                     "certifica el resto sin demora (y sin intereses, porque la retención está motivada). Para que no vuelva a pasar: "
                     "medir el avance por unidades de obra o hitos verificables (no por horas gastadas ni por «sensación»), aplicar la "
                     "regla 0/100 o 50/50 a las actividades cortas, y que la medición la haga o la valide la propiedad."))
    celdas += apartado(5)
    celdas.append(md("Decisión: informar al comité con las dos versiones, el semáforo de la peor y la fecha de la medición conjunta; "
                     f"certificar el {real} % de {X}; exigir al contratista un plan de recuperación de {X}"
                     + (f" (y revisar la planificación de {', '.join(sucesoras)}, que depende{'n' if len(sucesoras) > 1 else ''} de ella)" if sucesoras else "")
                     + f"; avisar al cliente por escrito de un riesgo de retraso de al menos {retraso_X:.0f} semanas. Mensajes:\n\n"
                     f"- *Contratista:* «Certificaremos esta semana el {real} % de {X}, que es lo verificado en obra; el resto en cuanto "
                     "lo comprobemos juntos el jueves. Necesito el viernes un plan de recuperación con hitos semanales medibles.»\n"
                     "- *Jefe de equipo:* «Gracias por avisar. Prepara la medición conjunta del jueves con las unidades de obra que "
                     f"sustentan tu {real} %; a partir de ahora validas tú cada porcentaje antes de la certificación.»\n"
                     f"- *Comité:* «Plazo: {sem_rojo(e['SPI_real'])} (SPI {e['SPI_real']:.2f} con el avance verificado; el "
                     f"informe del contratista daría verde). Coste: {sem_rojo(e['CPI_real'])} (EAC {EAC_real:.0f} k€ frente a "
                     f"{BAC_total:.0f}). Fin previsto: semana {T + retraso_X:.0f} como pronto si no se corrige {X}. El jueves hay medición conjunta.»\n"
                     f"- *Dirección financiera:* «Criterio: se certifica el avance verificado por la propiedad. Esta certificación retiene "
                     f"{sobrepago:.1f} k€ con acta de discrepancia; no hay base para intereses de demora.»\n"
                     f"- *Cliente:* «Les informamos de un riesgo de retraso de al menos {retraso_X:.0f} semanas en el hito, detectado en el "
                     f"control de la semana {t}; el plan de recuperación se les enviará el lunes.»"))
    celdas += apartado(6)
    celdas.append(md("- **Quién mide.** El avance lo declara quien cobra por él: conflicto de interés evidente. La medición debe hacerla "
                     "o validarla la propiedad (dirección facultativa) con un procedimiento escrito.\n"
                     "- **Cómo se mide.** «Porcentaje realizado» sin definición es opinión. Hay que fijar por actividad la unidad de medida "
                     "(m² ejecutados, equipos montados, pruebas superadas) o una regla fija (0/100, 50/50, hitos ponderados).\n"
                     "- **Cómo se paga.** Certificar por avance declarado adelanta caja al contratista y le quita incentivos; ligar el pago a "
                     "hitos verificados y mantener una retención hasta la recepción corrige el incentivo.\n"
                     "- **Verificación independiente.** Un jefe de equipo que pisa la obra detectó en una visita lo que el sistema de "
                     "informes no vio: la verificación física periódica debe estar en el procedimiento, no depender de la iniciativa de alguien."))
    return celdas, f"comité y valor ganado: {X} declarada al {rep} % frente al {real} % real; SPI {e['SPI_rep']:.2f} → {e['SPI_real']:.2f}"


# ---------------------------------------------------------------------------
# Caso 4 — La oferta comercial: qué plazo prometer (PERT frente a Monte Carlo)
# ---------------------------------------------------------------------------
SEMILLA_MC = 2024
N_MC = 20000


def _datos_oferta(rng):
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 14 <= i["T"] <= 24 and i["n_criticas"] == 1
                and i["segunda"] is not None and i["segunda"] >= i["T"] - 2):
            return False
        a, m, b = P.tres_estimaciones(rng, i["D"])
        for act in a.index:                       # las casi críticas, más inciertas (misma media): el sesgo de fusión se nota
            if 0 < i["calendario"].loc[act, "H_total"] <= 2:
                k = 2 if a[act] >= 3 else (1 if a[act] >= 2 else 0)
                a[act] -= k; b[act] += k
        var = ((b - a) / 6) ** 2
        T = float(i["T"])
        with _silencio():
            sigma, _ = i["proyecto"].standard_deviation(i["D"], var.to_dict())
            res = i["proyecto"].monte_carlo({act: ("pert", int(a[act]), int(m[act]), int(b[act])) for act in a.index},
                                            n_samples=6000, seed=SEMILLA_MC)
        tau = res.tau_samples
        p_mc = float((tau <= T).mean()); q90 = float(np.quantile(tau, 0.9)); q90_pert = T + norm.ppf(0.9) * sigma
        no_crit = res.sensitivity_mean[i["calendario"]["H_total"] > 0]
        if not (p_mc <= 0.42 and q90 - q90_pert >= 0.4 and len(no_crit) and no_crit.max() >= 0.25):
            return False
        estado.update(a=a, m=m, b=b, var=var, sigma=float(sigma), q90_pert=q90_pert)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    a, m, b = estado["a"], estado["m"], estado["b"]
    with _silencio():                             # las cifras definitivas, con las réplicas del cuaderno
        res = info["proyecto"].monte_carlo({act: ("pert", int(a[act]), int(m[act]), int(b[act])) for act in a.index},
                                           n_samples=N_MC, seed=SEMILLA_MC)
    tau = res.tau_samples
    crit = res.sensitivity_mean
    no_crit = crit[info["calendario"]["H_total"] > 0]
    estado.update(tau=tau, p_mc=float((tau <= float(info["T"])).mean()), q90=float(np.quantile(tau, 0.9)), crit=crit,
                  vigilar=str(no_crit.idxmax()))
    return info, estado


def _oferta_comercial(rng):
    info, e = _datos_oferta(rng)
    D, T, cal = info["D"], int(info["T"]), info["calendario"]
    a, m, b, sigma, tau = e["a"], e["m"], e["b"], e["sigma"], e["tau"]
    criticas = [x for x in D.index if cal.loc[x, "H_total"] == 0]
    X = max(criticas, key=lambda x: e["var"][x])        # la crítica con mayor varianza: la que discute producción
    pen, bon = int(rng.choice([4, 5, 6])), int(rng.choice([1, 2]))
    desc = int(rng.choice([2, 3]))
    T_comp = T + 1
    df = P.tabla_base(info, a=a, m=m, b=b)
    df = df.rename(columns={"a": "optimista", "m": "mas_probable", "b": "pesimista"})
    p_pert_T = 0.5
    q90_pert, q90_mc = e["q90_pert"], e["q90"]
    # coste esperado de prometer D
    Ds = list(range(T - 1, int(np.ceil(np.quantile(tau, 0.97))) + 2))
    def coste_D(Dp):
        return desc * max(0, Dp - T_comp) + pen * float(np.maximum(tau - Dp, 0).mean()) - bon * float(np.maximum(Dp - tau, 0).mean())
    costes = {Dp: coste_D(Dp) for Dp in Ds}
    D_opt = min(costes, key=costes.get)
    D_legal = int(np.ceil(q90_mc))
    p_opt = float((tau <= D_opt).mean())
    # hipótesis de producción: b_X + 2
    b2 = b.copy(); b2[X] += 2
    m_media2 = ((a + 4 * m + b2) / 6)
    with _silencio():
        T_prod = float(info["proyecto"].duration(m_media2))
    with _silencio():
        res2 = info["proyecto"].monte_carlo({act: ("pert", int(a[act]), int(m[act]), int(b2[act])) for act in a.index},
                                            n_samples=N_MC, seed=SEMILLA_MC)
    tau2 = res2.tau_samples
    agentes = ["director comercial", "jefe de producción", "cliente", "asesoría jurídica", "dirección financiera"]

    celdas = [md("# Enunciado"),
              md("Su empresa prepara la oferta para un proyecto llave en mano. El equipo técnico ha estimado para cada "
                 "actividad tres duraciones (optimista, más probable y pesimista, en semanas) y el cuadro de prelaciones, "
                 "que se adjuntan. Hay que escribir en la oferta **el plazo de entrega comprometido**, y de él dependen "
                 "penalizaciones y bonificaciones. Antes de cerrar la oferta recibe estos mensajes."),
              code('#| tbl-cap: "Cuadro de prelaciones y tres estimaciones de duración"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Director comercial",
                       f"El competidor va a ofertar {T_comp} semanas. Con las estimaciones del equipo nos sale una duración de "
                       f"{T} semanas: propongo escribir **{T} semanas** en la oferta, que es lo que dicen nuestros números y nos "
                       "pone por delante. No veo por qué añadir colchón."),
              _mensaje("Jefe de producción",
                       f"Las estimaciones son razonables salvo la de {X}: la duración pesimista ({int(b[X])}) se queda corta, yo "
                       f"pondría al menos {int(b[X]) + 2}. Y recordad que las medias no se suman alegremente: si varias cosas van "
                       "regular a la vez, el proyecto se va mucho más de lo que dice la ruta crítica."),
              _mensaje("Cliente",
                       f"Las condiciones del contrato: penalización de **{pen} k€ por semana** de retraso sobre el plazo ofertado y "
                       f"bonificación de **{bon} k€ por semana** de adelanto. Valoraremos el plazo ofertado: cada semana por encima "
                       f"de {T_comp} resta puntos en la adjudicación."),
              _mensaje("Asesoría jurídica",
                       "No podemos firmar un plazo que no estemos en condiciones de cumplir con alta probabilidad: pedimos que el "
                       "plazo comprometido se cumpla **al menos en el 90 % de los escenarios**. Si el plazo es menor, que quede "
                       "documentado quién asume el riesgo."),
              _mensaje("Dirección financiera",
                       f"Ofertar más plazo tiene un coste comercial: estimamos que cada semana por encima de las {T_comp} del "
                       f"competidor equivale a {desc} k€ de descuento en precio para seguir siendo competitivos. Pero las "
                       "penalizaciones se pagan en efectivo. Quiero el plazo que minimiza el coste esperado total."),
              _cierre(agentes)]
    celdas.append(preguntas([
        f"Con el método PERT (media y varianza de cada actividad a partir de las tres estimaciones) calcule la duración "
        f"esperada, la desviación típica del camino crítico, la probabilidad de terminar en {T} semanas y el plazo que se "
        "cumpliría con el 90 % de probabilidad.",
        f"Simule el proyecto por Monte Carlo ({N_MC} réplicas, distribución PERT por actividad). Compare con lo anterior: "
        f"probabilidad de terminar en {T} semanas, plazo al 90 % e índice de criticidad de cada actividad. ¿Por qué no "
        "coinciden? ¿Qué actividad no crítica merece vigilancia?",
        f"Incorpore la observación del jefe de producción (pesimista de {X} = {int(b[X]) + 2}). ¿Cuánto cambian la media, la "
        "probabilidad y el plazo al 90 %?",
        "Con las réplicas de Monte Carlo, calcule para cada plazo candidato el coste esperado total: descuento comercial por "
        f"semana ofertada por encima de {T_comp}, penalización esperada y bonificación esperada. ¿Qué plazo minimiza el coste "
        "esperado? ¿Cumple la condición de asesoría jurídica? Si no, ¿cuánto cuesta cumplirla?",
        "Decida el plazo que escribe en la oferta y redacte en tres o cuatro líneas la respuesta a cada implicado.",
        "¿Qué lecciones generales saca sobre prometer plazos a partir de estimaciones (medias, colchones, actividades casi "
        "críticas, seguimiento)?"]))
    celdas.append(md("# Resolución"))
    celdas.append(code(UTIL))
    celdas.append(code(f"plazo_competidor, plazo_comercial, penalizacion, bonificacion, descuento = {T_comp}, {T}, {pen}, {bon}, {desc}\n"
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['duration'] = (datos_proyecto['optimista'] + 4 * datos_proyecto['mas_probable'] + datos_proyecto['pesimista']) / 6\n"
                       "datos_proyecto['varianza'] = ((datos_proyecto['pesimista'] - datos_proyecto['optimista']) / 6) ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Media y varianza PERT de cada actividad; calendario"\n#| label: tbl-pert\n'
                       "display(datos_proyecto[['optimista', 'mas_probable', 'pesimista', 'duration', 'varianza']].round(3))\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']]"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias y el camino crítico", label="fig-pert")
    celdas.append(code("T_media = float(proyecto.duration(datos_proyecto['duration']))\n"
                       "sigma, varianzas_caminos = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['varianza'].to_dict())\n"
                       "display(varianzas_caminos)\n"
                       "p_pert = norm.cdf((plazo_comercial - T_media) / sigma)\n"
                       "plazo90_pert = T_media + norm.ppf(0.9) * sigma\n"
                       "Markdown(f'''Duración esperada {T_media:.1f} semanas, camino crítico {camino_critico(proyecto, datos_proyecto[\"duration\"])}, σ = {sigma:.2f}. "
                       "P(T ≤ {T_media:.0f}) = {p_pert:.2f} (la media se cumple la mitad de las veces); plazo al 90 %: {T_media:.1f} + 1.28 × {sigma:.2f} = **{plazo90_pert:.1f} semanas**.''')"))
    celdas += apartado(2)
    celdas.append(code(f"resultado = proyecto.monte_carlo({{act: ('pert', int(fila['optimista']), int(fila['mas_probable']), int(fila['pesimista'])) for act, fila in datos_proyecto.iterrows()}},\n"
                       f"                                 n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "tau = resultado.tau_samples\n"
                       "display(resultado.summary().round(2))\n"
                       "p_mc = float((tau <= plazo_comercial).mean()); plazo90_mc = float(np.quantile(tau, 0.9))\n"
                       "criticidad = (resultado.sensitivity_mean * 100).round(1)\n"
                       "display(pd.DataFrame({'criticidad (%)': criticidad, 'holgura total': calendario['H_total']}).T)\n"
                       "Markdown(f'''Monte Carlo: media {tau.mean():.2f}, desviación típica {tau.std():.2f}; P(T ≤ {plazo_comercial}) = **{p_mc:.2f}** "
                       "(PERT decía {p_pert:.2f}); plazo al 90 %: **{plazo90_mc:.1f}** semanas (PERT decía {plazo90_pert:.1f}).''')"))
    celdas.append(code('#| fig-cap: "Distribución simulada de la duración y los dos plazos al 90 %"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "ax.hist(tau, bins=40, density=True, alpha=0.6, label='Monte Carlo')\n"
                       "xs = np.linspace(tau.min(), tau.max(), 200); ax.plot(xs, norm.pdf(xs, T_media, sigma), label=f'normal PERT N({T_media:.1f}, {sigma:.2f})')\n"
                       "ax.axvline(plazo90_pert, ls='--', color='C1'); ax.axvline(plazo90_mc, ls='--', color='C0'); ax.axvline(T_media, color='k', lw=0.8)\n"
                       "ax.set_xlabel('duración (semanas)'); ax.legend();"))
    celdas.append(md(f"La media PERT ({T}) subestima la duración y la normal del camino crítico subestima la cola derecha: en las "
                     f"réplicas hay otras rutas (la segunda dura {info['segunda']:.0f} en media) que en bastantes escenarios se "
                     f"convierten en la más larga. Es el *sesgo de fusión* (merge bias): el máximo de varias variables aleatorias tiene "
                     f"media mayor que el mayor de sus medias. Por eso P(T ≤ {T}) baja del 0.50 al {e['p_mc']:.2f} y el plazo al 90 % "
                     f"sube de {q90_pert:.1f} a {q90_mc:.1f}. La actividad no crítica con más criticidad es **{e['vigilar']}** "
                     f"({e['crit'][e['vigilar']]:.0%} de las réplicas en la ruta más larga, holgura {int(cal.loc[e['vigilar'], 'H_total'])}): "
                     "el análisis determinista la da por holgada y conviene vigilarla como si fuera crítica."))
    celdas += apartado(3)
    celdas.append(code(f"datos_prod = datos_proyecto.copy(); datos_prod.loc[{X!r}, 'pesimista'] += 2\n"
                       "datos_prod['duration'] = (datos_prod['optimista'] + 4 * datos_prod['mas_probable'] + datos_prod['pesimista']) / 6\n"
                       "datos_prod['varianza'] = ((datos_prod['pesimista'] - datos_prod['optimista']) / 6) ** 2\n"
                       "T_prod = float(proyecto.duration(datos_prod['duration']))\n"
                       "sigma_prod, _ = proyecto.standard_deviation(datos_prod['duration'], datos_prod['varianza'].to_dict())\n"
                       f"resultado_prod = proyecto.monte_carlo({{act: ('pert', int(fila['optimista']), int(fila['mas_probable']), int(fila['pesimista'])) for act, fila in datos_prod.iterrows()}}, n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "tau_prod = resultado_prod.tau_samples\n"
                       "pd.DataFrame({'estimaciones del equipo': [T_media, sigma, p_pert, plazo90_pert, p_mc, plazo90_mc],\n"
                       f"              'pesimista de {X} + 2': [T_prod, sigma_prod, norm.cdf((plazo_comercial - T_prod) / sigma_prod), T_prod + norm.ppf(0.9) * sigma_prod,\n"
                       "                                        float((tau_prod <= plazo_comercial).mean()), float(np.quantile(tau_prod, 0.9))]},\n"
                       "             index=['media PERT', 'σ PERT', f'P(T ≤ {plazo_comercial}) PERT', 'plazo 90 % PERT', f'P(T ≤ {plazo_comercial}) MC', 'plazo 90 % MC']).round(2)"))
    celdas.append(md(f"Añadir dos semanas a la estimación pesimista de {X} sube su media en {2 / 6:.2f} semanas y su varianza; la "
                     f"duración esperada pasa a {T_prod:.2f} y el plazo al 90 % por Monte Carlo a {np.quantile(tau2, 0.9):.1f}. "
                     "El efecto es moderado pero va en la misma dirección: cuanto más incierta la actividad crítica, más colchón hace falta "
                     "para la misma probabilidad. Y es información que el equipo de producción tenía y la oferta no."))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Coste esperado (k€) según el plazo ofertado, con las réplicas de Monte Carlo"\n#| label: tbl-plazos\n'
                       "candidatos = range(int(T_media) - 1, int(np.ceil(np.quantile(tau, 0.97))) + 2)\n"
                       "filas = []\n"
                       "for Dp in candidatos:\n"
                       "    pen_esp = penalizacion * float(np.maximum(tau - Dp, 0).mean())\n"
                       "    bon_esp = bonificacion * float(np.maximum(Dp - tau, 0).mean())\n"
                       "    com = descuento * max(0, Dp - plazo_competidor)\n"
                       "    filas.append({'plazo ofertado': Dp, 'P(cumplir)': float((tau <= Dp).mean()), 'penalización esperada': pen_esp,\n"
                       "                  'bonificación esperada': bon_esp, 'descuento comercial': com, 'coste esperado total': com + pen_esp - bon_esp})\n"
                       "plazos = pd.DataFrame(filas).set_index('plazo ofertado')\n"
                       "display(plazos.round(2))\n"
                       "D_opt = int(plazos['coste esperado total'].idxmin()); D_legal = int(np.ceil(plazo90_mc))\n"
                       "Markdown(f'''El plazo que minimiza el coste esperado es **{D_opt} semanas** (P(cumplir) = {plazos.loc[D_opt, \"P(cumplir)\"]:.2f}); "
                       "el que exige asesoría jurídica (90 %) es **{D_legal}** semanas, con coste esperado {plazos.loc[D_legal, \"coste esperado total\"]:.1f} k€ "
                       "frente a {plazos.loc[D_opt, \"coste esperado total\"]:.1f}: cumplir la condición cuesta {plazos.loc[D_legal, \"coste esperado total\"] - plazos.loc[D_opt, \"coste esperado total\"]:.1f} k€ esperados.''')"))
    celdas.append(md(f"Prometer {T} semanas (lo que pide el director comercial) tiene un coste esperado de {costes[T]:.1f} k€ y se "
                     f"cumple el {float((tau <= T).mean()):.0%} de las veces; el óptimo económico es {D_opt} ({costes[D_opt]:.1f} k€, "
                     f"P = {p_opt:.2f}) y el plazo «jurídico» {D_legal} ({costes[D_legal]:.1f} k€). La penalización esperada decrece con "
                     f"el plazo ofertado y el descuento comercial crece {desc} k€ por semana: el óptimo está donde la penalización "
                     "marginal evitada iguala el descuento marginal. La condición del 90 % es una preferencia por el riesgo, no un "
                     "óptimo en media: su coste es la diferencia entre las dos filas, y conviene que quien la impone lo sepa."))
    celdas += apartado(5)
    elegido = D_legal if costes[D_legal] - costes[D_opt] <= 2 * desc else D_opt
    celdas.append(md(f"Una decisión defendible: ofertar **{elegido} semanas**"
                     + (f", el plazo al 90 %: la diferencia de coste esperado con el óptimo ({costes[D_legal] - costes[D_opt]:.1f} k€) es pequeña "
                        "frente al riesgo reputacional de incumplir, y evita la discusión interna." if elegido == D_legal else
                        f", el óptimo económico, documentando que se cumple con probabilidad {p_opt:.2f} y que la dirección asume el riesgo "
                        f"de la diferencia hasta el 90 %.")
                     + f" En ningún caso {T}: es la media, y la media se incumple la mitad de las veces en el mejor de los casos "
                     f"(aquí, el {1 - e['p_mc']:.0%}). Mensajes:\n\n"
                     f"- *Director comercial:* «{T} semanas es la duración media, no un plazo: se incumpliría el {1 - e['p_mc']:.0%} de las veces y "
                     f"cada semana cuesta {pen} k€. Propongo {elegido}; el coste esperado total es {costes[elegido]:.1f} k€ frente a {costes[T]:.1f} con {T}.»\n"
                     f"- *Jefe de producción:* «Incorporada tu estimación de {X}; sube el plazo al 90 % en {np.quantile(tau2, 0.9) - q90_mc:.1f} semanas. "
                     f"Durante la ejecución vigilaremos {X} y también {e['vigilar']}, que aparece crítica en el {e['crit'][e['vigilar']]:.0%} de los escenarios.»\n"
                     f"- *Cliente:* «Plazo ofertado: {elegido} semanas, con las penalizaciones y bonificaciones acordadas.»\n"
                     f"- *Asesoría jurídica:* «El plazo ofertado se cumple en el {float((tau <= elegido).mean()):.0%} de los escenarios simulados"
                     + (".»" if elegido == D_legal else f"; el riesgo entre ese valor y el 90 % lo asume la dirección, y queda documentado en el acta de oferta.»") + "\n"
                     f"- *Dirección financiera:* «Coste esperado por plazo adjunto; el mínimo es {D_opt} semanas ({costes[D_opt]:.1f} k€) y el plazo "
                     f"al 90 % cuesta {costes[D_legal] - costes[D_opt]:.1f} k€ más en media. La propuesta es {elegido}.»"))
    celdas += apartado(6)
    celdas.append(md("- La duración media del camino crítico no es un plazo: se cumple la mitad de las veces como mucho, y menos cuando hay "
                     "rutas casi críticas (sesgo de fusión). Un plazo se promete con una probabilidad explícita.\n"
                     "- El colchón tiene un precio (descuento, competitividad) y la falta de colchón también (penalización esperada); la "
                     "decisión es económica y se puede calcular con las réplicas.\n"
                     "- Las actividades casi críticas con mucha varianza son las que convierten el plan en papel mojado: el índice de "
                     "criticidad dice cuáles vigilar, cosa que el análisis determinista no puede.\n"
                     "- La información del jefe de producción debía haber entrado en las estimaciones antes de calcular nada: estimar es un "
                     "proceso de equipo, no una hoja que rellena uno."))
    return celdas, f"oferta comercial: media {T}, P(≤{T}) = {e['p_mc']:.2f} por Monte Carlo, plazo 90 % {q90_mc:.1f}; óptimo {D_opt}"


# ---------------------------------------------------------------------------
# Caso 5 — El comité de inversión: escenarios, un contrato que da seguridad a cambio de margen y financiación
# ---------------------------------------------------------------------------
def _comite_inversion(rng):
    x = float(rng.choice([0.06, 0.08, 0.10]))
    y = float(rng.choice([0.10, 0.15, 0.20]))
    z = float(rng.choice([0.08, 0.10, 0.15]))
    pi = float(rng.choice([0.25, 0.3, 0.4]))

    def van_con(p, **cambios):
        q = dict(p); q.update(cambios)
        return _van(_flujos(q)[1], p["tasa"])

    def acepta(p, r):
        I0 = p["inversion"]
        if not (0.10 * I0 < r["van"] < 0.35 * I0 and p["tasa"] + 0.02 < r["tir"] < 0.30):
            return False
        v_bajo = van_con(p, precio=p["precio"] * (1 - 2 * x))
        v_x = van_con(p, precio=p["precio"] * (1 - x))
        v_vol = van_con(p, unidades_1=p["unidades_1"] * (1 + y))
        v_cv = van_con(p, coste_variable=p["coste_variable"] * (1 + z))
        return (v_x < 0.5 * r["van"] and -0.5 * I0 < v_bajo < -0.05 * I0 and v_vol > 1.5 * r["van"]
                and v_cv < 0.6 * r["van"])

    caso, p, tabla, fcl, res = _buscar_caso(rng, acepta)
    df = _tabla_parametros(p, caso)
    I0, k, t_imp, n = p["inversion"], p["tasa"], p["impuesto"], p["vida"]
    VAN, TIR = res["van"], res["tir"]
    esc = {"caso base": VAN,
           f"precio −{x:.0%}": van_con(p, precio=p["precio"] * (1 - x)),
           f"precio −{2 * x:.0%}": van_con(p, precio=p["precio"] * (1 - 2 * x)),
           f"volumen +{y:.0%}": van_con(p, unidades_1=p["unidades_1"] * (1 + y)),
           f"coste variable +{z:.0%}": van_con(p, coste_variable=p["coste_variable"] * (1 + z))}
    # contrato: 40 % del volumen a precio −x fijo; el resto a mercado
    v_contrato_base = van_con(p, precio=p["precio"] * (1 - 0.4 * x))
    v_contrato_bajo = van_con(p, precio=p["precio"] * (0.4 * (1 - x) + 0.6 * (1 - 2 * x)))
    e_sin = (1 - pi) * VAN + pi * esc[f"precio −{2 * x:.0%}"]
    e_con = (1 - pi) * v_contrato_base + pi * v_contrato_bajo
    # financiación
    alfa, i, n_p = float(rng.choice([0.5, 0.6])), float(rng.choice([0.04, 0.05, 0.06])), int(min(n, rng.choice([4, 5])))
    prestamo = alfa * I0
    C = prestamo * i / (1 - (1 + i) ** -n_p)
    vivo, fa = prestamo, pd.Series(fcl, index=range(n + 1), dtype=float)
    fa[0] += prestamo
    for tt in range(1, n_p + 1):
        inter = vivo * i; fa[tt] += -C + t_imp * inter; vivo -= C - inter
    TIR_acc = _tir(fa.tolist())
    umbral_ce = VAN / I0
    agentes = ["director comercial", "director de producción", "dirección financiera", "cliente clave", "consejero delegado"]

    celdas = [md("# Enunciado"),
              md(f"El comité de inversiones de su empresa estudia invertir en {caso['tema']}. Usted ha preparado el caso base "
                 "(datos adjuntos; mismas hipótesis que en la colección de viabilidad: amortización lineal hasta el valor "
                 "residual, circulante como fracción de las ventas aportado al principio del año y recuperado al final, "
                 "impuesto sobre el beneficio positivo). Antes de la reunión recibe estos mensajes."),
              code('#| tbl-cap: "Datos del caso base"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df, indice="parametro")),
              _mensaje("Director comercial",
                       f"Vuestro volumen es conservador: con la red que tenemos vendemos un **{y:.0%} más** el primer año, seguro. "
                       f"Lo que sí me preocupa es el precio: si entra el competidor que está sondeando el mercado, el precio puede "
                       f"caer un **{2 * x:.0%}** desde el primer año. Le doy a eso una probabilidad del {pi:.0%}."),
              _mensaje("Director de producción",
                       f"Las materias primas están subiendo. Yo no contaría con el coste variable del caso base; pondría al menos un "
                       f"**{z:.0%} más**. Lo demás me parece bien."),
              _mensaje("Dirección financiera",
                       f"El banco ofrece financiar el {alfa:.0%} de la inversión con un préstamo al {i:.0%} a {n_p} años, cuota "
                       "constante. Quiero saber qué hace eso con la rentabilidad del accionista, y que no nos deslumbre: el "
                       "proyecto tiene que ser bueno sin deuda."),
              _mensaje("Cliente clave",
                       f"Estamos dispuestos a firmar un contrato a {n} años por el **40 % del volumen** del caso base a un precio fijo "
                       f"un **{x:.0%} inferior** al previsto, pase lo que pase en el mercado. Es nuestra oferta final."),
              _mensaje("Consejero delegado",
                       f"Regla de la casa: no aprobamos proyectos cuyo VAN esperado no supere el 10 % de la inversión, ni con una "
                       "probabilidad apreciable de VAN negativo sin una cobertura. Quiero una recomendación clara el jueves."),
              _cierre(agentes)]
    celdas.append(preguntas([
        "Calcule los flujos de caja libre del caso base, el VAN, la TIR y los periodos de recuperación simple y descontado. "
        "¿Pasa la regla del consejero delegado?",
        f"Evalúe por separado cada mensaje como escenario: precio −{x:.0%} y −{2 * x:.0%}, volumen +{y:.0%}, coste variable "
        f"+{z:.0%}. Calcule además los umbrales (precio mínimo, volumen mínimo, coste variable máximo con VAN = 0) y diga a qué "
        "variable es más sensible el proyecto.",
        f"Con la probabilidad del {pi:.0%} que da el director comercial a la caída del precio, calcule el VAN esperado sin el "
        "contrato del cliente clave y con él (40 % del volumen a precio −{:.0%} fijo; el resto a precio de mercado en cada "
        "escenario). ¿Cuánto cuesta la seguridad que ofrece el contrato y cuánto riesgo elimina?".format(x),
        "Construya el cuadro del préstamo y el flujo del accionista (con ahorro fiscal de los intereses). Compare la TIR del "
        "proyecto con la del accionista y explique el apalancamiento. ¿Tiene razón Dirección financiera en que el proyecto debe "
        "ser bueno sin deuda?",
        "Haga una recomendación al comité (aprobar, aprobar con contrato, rechazar, pedir más información) y redacte en tres o "
        "cuatro líneas la respuesta a cada implicado.",
        "¿Qué fallos del proceso de evaluación revela el caso (quién aporta las hipótesis, cómo se tratan los riesgos de precio "
        "y coste, cómo se separa la decisión de inversión de la de financiación)?"]))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas += apartado(1)
    celdas.append(code(PARAMS))
    celdas.append(code('#| tbl-cap: "Flujos de caja libre del caso base (k€)"\n#| label: tbl-flujos\n' + MODELO + "\ntabla.round(1).T"))
    celdas.append(code("def modelo(precio=precio, u1=u1, cv=cv, CF=CF, g=g, I0=I0, n=n, VR=VR, w=w, t_imp=t_imp):\n"
                       "    '''Flujos de caja libre del modelo con los parámetros que se indiquen (los demás, del caso base).'''\n"
                       "    unidades = pd.Series([0] + [u1 * (1 + g) ** (t - 1) for t in range(1, n + 1)])\n"
                       "    ventas = unidades * precio; EBITDA = ventas - unidades * cv - pd.Series([0] + [CF] * n)\n"
                       "    amort = pd.Series([0] + [(I0 - VR) / n] * n); BAI = EBITDA - amort\n"
                       "    imp = BAI.clip(lower=0) * t_imp; imp[0] = 0\n"
                       "    circ = pd.Series([w * ventas[1]] + [w * ventas[t] for t in range(1, n + 1)])\n"
                       "    dcirc = circ.diff().fillna(circ[0]); dcirc[n] -= circ[n]\n"
                       "    return (BAI - imp + amort - dcirc - pd.Series([I0] + [0] * n) + pd.Series([0] * n + [VR])).tolist()\n"
                       "VAN, TIR = van(flujos, k), tir(flujos)\n"
                       "display(tabla_descuento(flujos, k).round(2))\n"
                       "Markdown(f'''VAN = {VAN:.1f} k€ al {k:.0%} ({VAN / I0:.1%} de la inversión; la regla pide el 10 %: {\"pasa\" if VAN >= 0.1 * I0 else \"no pasa\"}); "
                       "TIR = {TIR:.2%}; recuperación simple {payback(flujos):.2f} años, descontada {payback(flujos, k):.2f}.''')"))
    celdas.append(md(f"El caso base es {'claramente' if VAN >= 0.15 * I0 else 'moderadamente'} rentable: VAN {VAN:.0f} k€ "
                     f"({umbral_ce:.0%} de la inversión; {'supera' if umbral_ce >= 0.10 else 'no llega a'} el 10 % de la regla), TIR "
                     f"{TIR:.1%} frente al {k:.0%} exigido. El consejero delegado pide además que no haya una probabilidad apreciable de "
                     "VAN negativo, y eso es lo que los mensajes ponen en duda."))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Escenarios de cada mensaje"\n#| label: tbl-escenarios\n'
                       f"x, y, z = {x}, {y}, {z}\n"
                       "escenarios = {'caso base': modelo(), f'precio −{x:.0%}': modelo(precio=precio * (1 - x)), f'precio −{2 * x:.0%}': modelo(precio=precio * (1 - 2 * x)),\n"
                       "              f'volumen +{y:.0%}': modelo(u1=u1 * (1 + y)), f'coste variable +{z:.0%}': modelo(cv=cv * (1 + z))}\n"
                       "tabla_esc = pd.DataFrame({nombre: {'VAN (k€)': van(f, k), 'TIR': tir(f), 'VAN / inversión': van(f, k) / I0} for nombre, f in escenarios.items()}).T\n"
                       "tabla_esc.round(3)"))
    celdas.append(code('#| tbl-cap: "Umbrales (VAN = 0) y variación admisible"\n#| label: tbl-umbrales\n'
                       "from scipy.optimize import brentq\n"
                       "precio_umbral = brentq(lambda v: van(modelo(precio=v), k), precio * 0.2, precio * 2)\n"
                       "u1_umbral = brentq(lambda v: van(modelo(u1=v), k), u1 * 0.2, u1 * 2)\n"
                       "cv_umbral = brentq(lambda v: van(modelo(cv=v), k), cv * 0.2, max(cv * 5, precio * 0.99))\n"
                       "umbrales = pd.DataFrame({'caso base': [precio, u1, cv], 'umbral (VAN = 0)': [precio_umbral, u1_umbral, cv_umbral]},\n"
                       "                        index=['precio', 'unidades año 1', 'coste variable'])\n"
                       "umbrales['variación admisible'] = umbrales['umbral (VAN = 0)'] / umbrales['caso base'] - 1\n"
                       "display(umbrales.round(3))\n"
                       "mas_sensible = umbrales['variación admisible'].abs().idxmin()\n"
                       "Markdown(f'La variable más sensible es **{mas_sensible}**: una variación del {umbrales.loc[mas_sensible, \"variación admisible\"]:+.1%} anula el VAN.')"))
    lineas = "\n".join(f"- {nombre}: VAN {v:.0f} k€ ({v / I0:+.0%} de la inversión)." for nombre, v in esc.items())
    celdas.append(md("Cada mensaje por separado:\n\n" + lineas + "\n\n"
                     f"El mensaje del director comercial tiene dos caras: su volumen convierte el proyecto en muy bueno, pero su "
                     f"escenario de precio lo hunde; el coste variable de producción {'lo deja en pérdidas' if esc[f'coste variable +{z:.0%}'] < 0 else 'se come buena parte del VAN'}. "
                     "El proyecto es más sensible al precio que al volumen (el volumen arrastra costes variables; el precio va "
                     "directo al margen), y el coste variable actúa como un precio negativo."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "VAN esperado sin y con el contrato del cliente clave"\n#| label: tbl-contrato\n'
                       f"pi = {pi}\n"
                       "# sin contrato: todo el volumen a precio de mercado (base, o −2x si entra el competidor)\n"
                       "van_sin = {'mercado estable': van(modelo(), k), 'cae el precio': van(modelo(precio=precio * (1 - 2 * x)), k)}\n"
                       "# con contrato: 40 % a precio fijo (1 − x); el 60 % restante a mercado. Precio medio = 0.4·(1 − x)·p + 0.6·p_mercado\n"
                       "van_con = {'mercado estable': van(modelo(precio=precio * (0.4 * (1 - x) + 0.6)), k),\n"
                       "           'cae el precio': van(modelo(precio=precio * (0.4 * (1 - x) + 0.6 * (1 - 2 * x))), k)}\n"
                       "contrato = pd.DataFrame({'sin contrato': van_sin, 'con contrato': van_con}).T\n"
                       "contrato['VAN esperado'] = (1 - pi) * contrato['mercado estable'] + pi * contrato['cae el precio']\n"
                       "contrato['peor caso'] = contrato['cae el precio']\n"
                       "display(contrato.round(1))\n"
                       "Markdown(f'''Con probabilidad {pi:.0%} de caída del precio: VAN esperado **{contrato.loc[\"sin contrato\", \"VAN esperado\"]:.1f}** k€ sin contrato y "
                       "**{contrato.loc[\"con contrato\", \"VAN esperado\"]:.1f}** con él. El contrato cuesta {van_sin[\"mercado estable\"] - van_con[\"mercado estable\"]:.1f} k€ de VAN "
                       "si el mercado aguanta y ahorra {van_con[\"cae el precio\"] - van_sin[\"cae el precio\"]:.1f} si cae.''')"))
    mejor = "con contrato" if e_con > e_sin else "sin contrato"
    v_bajo = esc[f"precio −{2 * x:.0%}"]
    celdas.append(md(f"El contrato es un seguro: se paga una prima cierta ({VAN - v_contrato_base:.0f} k€ de VAN si el mercado "
                     f"aguanta) a cambio de recortar la pérdida en el escenario malo ({v_contrato_bajo - v_bajo:.0f} k€ "
                     f"menos de pérdida). En valor esperado sale mejor **{mejor}** ({max(e_con, e_sin):.0f} frente a {min(e_con, e_sin):.0f} k€)"
                     + (": como casi todo seguro, la prima vale más que la pérdida esperada que evita (el VAN es casi lineal en el precio y "
                        f"la caída tiene probabilidad {pi:.0%} < 50 %)" if mejor == "sin contrato" else
                        ": el escudo fiscal hace que la pérdida evitada valga más que la prima")
                     + f". Pero la regla del consejero mira también al peor caso: "
                     + ("con el contrato el peor caso sigue siendo negativo" if v_contrato_bajo < 0 else "con el contrato el peor caso deja de ser negativo")
                     + f" ({v_contrato_bajo:.0f} k€ frente a {v_bajo:.0f} sin él). "
                     + ("El 40 % que ofrece el cliente no basta para cubrir el riesgo de precio: cubre menos de la mitad del volumen. Una "
                        "contraoferta razonable es ampliar el porcentaje contratado o acortar el descuento."
                        if v_contrato_bajo < 0 else
                        "El 40 % contratado basta para que el proyecto no destruya valor ni en el escenario malo: es lo que la regla pide.")))
    celdas += apartado(4)
    celdas.append(code(f"alfa, i, n_p = {alfa}, {i}, {n_p}\n"
                       "def cuadro_prestamo(P, i, n_p):\n"
                       "    C = P * i / (1 - (1 + i) ** -n_p)\n"
                       "    vivo, filas = P, []\n"
                       "    for t in range(1, n_p + 1):\n"
                       "        interes = vivo * i; amort = C - interes; vivo -= amort\n"
                       "        filas.append({'año': t, 'cuota': C, 'intereses': interes, 'amortización': amort, 'capital vivo': max(vivo, 0)})\n"
                       "    return pd.DataFrame(filas).set_index('año')\n"
                       "def flujo_accionista(cuadro, prestamo):\n"
                       "    f = pd.Series(flujos, index=range(n + 1)).astype(float)\n"
                       "    f[0] += prestamo\n"
                       "    for t, fila in cuadro.iterrows():\n"
                       "        f[t] += -fila['cuota'] + t_imp * fila['intereses']\n"
                       "    return f\n"
                       "prestamo = alfa * I0\n"
                       "cuadro = cuadro_prestamo(prestamo, i, n_p)\n"
                       "display(cuadro.round(2))\n"
                       "fa = flujo_accionista(cuadro, prestamo)\n"
                       "display(pd.DataFrame({'FCL proyecto': flujos, 'flujo del accionista': fa.tolist()}, index=pd.Index(range(n + 1), name='año')).round(1).T)\n"
                       "TIR_acc = tir(fa.tolist())\n"
                       "Markdown(f'''TIR del proyecto {TIR:.2%}; TIR del accionista con el préstamo {TIR_acc:.2%}. Coste de la deuda después de impuestos "
                       "{i * (1 - t_imp):.2%}, {\"menor\" if i * (1 - t_imp) < TIR else \"mayor\"} que la TIR del proyecto: el apalancamiento {\"amplifica la rentabilidad del accionista\" if i * (1 - t_imp) < TIR else \"la reduce\"}.''')"))
    celdas.append(md(f"La TIR del accionista ({TIR_acc:.1%}) es mayor que la del proyecto ({TIR:.1%}) porque la deuda cuesta "
                     f"{i * (1 - t_imp):.1%} neto, menos de lo que rinde el proyecto: el accionista pone el {1 - alfa:.0%} del capital y "
                     "se queda con la diferencia. Pero el apalancamiento amplifica en los dos sentidos: en el escenario de caída del "
                     "precio las cuotas se pagan igual y la pérdida del accionista es proporcionalmente mayor. Dirección financiera "
                     "tiene razón: la decisión de invertir se toma sobre el flujo de caja libre (sin deuda) y la de financiar, después; "
                     "una TIR del accionista alta no convierte en bueno un proyecto malo."))
    celdas += apartado(5)
    pasa_esperado = max(e_sin, e_con) >= 0.10 * I0
    if not pasa_esperado:
        recomendacion = ("no aprobar tal como está y volver al comité con tres cosas: la probabilidad de caída del precio documentada "
                         "(no una opinión), una cobertura que cubra la mayor parte del volumen (contraoferta al cliente clave) y el coste "
                         "variable revisado con producción")
        corta = "no aprobar tal como está"
    elif v_contrato_bajo < 0:
        recomendacion = ("aprobar condicionado a cerrar una cobertura de precio mayor que la ofrecida (contraoferta al cliente clave: "
                         "más volumen contratado o menos descuento) y a revisar el coste variable con producción")
        corta = "aprobar condicionado a una cobertura mayor"
    else:
        recomendacion = ("aprobar con el contrato del cliente clave, que elimina el VAN negativo del peor caso, y revisar el coste "
                         "variable con producción")
        corta = "aprobar con el contrato"
    celdas.append(md(f"Recomendación defendible: **{recomendacion}**. El caso base pasa la regla del 10 % "
                     f"({'sí' if umbral_ce >= 0.10 else 'no'}: {umbral_ce:.0%}), pero con el riesgo de precio el VAN esperado "
                     f"{'baja a' if pasa_esperado else 'queda en'} {e_sin:.0f} k€ ({e_sin / I0:.0%} de la inversión; "
                     f"{'pasa' if e_sin >= 0.1 * I0 else 'no pasa'}) y hay un {pi:.0%} de probabilidad de VAN {v_bajo:.0f} k€, "
                     "y la regla también mira eso. El volumen del director comercial no se incorpora al caso base (es una opinión sin "
                     "contrato detrás) pero sí como escenario favorable. Mensajes:\n\n"
                     f"- *Director comercial:* «Tu volumen daría VAN {esc[f'volumen +{y:.0%}']:.0f} k€; tu riesgo de precio, {esc[f'precio −{2 * x:.0%}']:.0f}. "
                     "El segundo pesa más que el primero en la decisión: ¿puedes convertir parte del volumen extra en contratos firmados?»\n"
                     f"- *Director de producción:* «Con tu coste variable el VAN baja a {esc[f'coste variable +{z:.0%}']:.0f} k€. Necesito la "
                     "estimación documentada (proveedores, índices) para llevarla al caso base o como escenario.»\n"
                     f"- *Dirección financiera:* «TIR del accionista {TIR_acc:.1%} frente a {TIR:.1%} del proyecto; de acuerdo en decidir "
                     "sobre el FCL sin deuda. Si se aprueba, el préstamo mejora la rentabilidad del capital, con el riesgo que conlleva.»\n"
                     + (f"- *Cliente clave:* «Aceptamos el contrato del 40 % a −{x:.0%}: nos da la estabilidad de precio que el comité exige.»\n"
                        if corta == "aprobar con el contrato" else
                        f"- *Cliente clave:* «Valoramos el contrato. El 40 % a −{x:.0%} cubre poco riesgo para lo que cuesta; "
                        "proponemos un 60 % con el mismo descuento, o el 40 % con un descuento menor.»\n")
                     + f"- *Consejero delegado:* «VAN base {VAN:.0f} k€ ({umbral_ce:.0%} de la inversión); VAN esperado {e_sin:.0f} sin cobertura y "
                     f"{e_con:.0f} con el contrato ofrecido; peor caso {v_bajo:.0f} sin cubrir y {v_contrato_bajo:.0f} con él. Recomiendo {corta}.»"))
    celdas += apartado(6)
    celdas.append(md("- **Hipótesis sin dueño.** El caso base lo preparó una persona; comercial y producción tenían información que lo "
                     "cambia y la aportaron el día antes del comité. Las hipótesis deben recogerse de cada área con su justificación y "
                     "su rango, antes de calcular.\n"
                     "- **Riesgo tratado como opinión.** «Puede caer un X % con probabilidad π» es una hipótesis valiosa si se formaliza "
                     "(escenarios con probabilidades, árbol de decisión, o simulación) en lugar de discutirse en la reunión.\n"
                     "- **Coberturas.** Un contrato a precio fijo es un seguro con prima: se evalúa comparando el valor esperado y el peor "
                     "caso con y sin él, y se negocia (porcentaje, descuento, duración).\n"
                     "- **Inversión y financiación por separado.** La bondad del proyecto se juzga sobre el flujo de caja libre; la "
                     "estructura financiera cambia el reparto entre accionista y banco y el riesgo, no el valor que crea el proyecto."))
    return celdas, f"comité de inversión: {caso['tema']}; VAN {VAN:.0f} k€ ({umbral_ce:.0%}), precio −{2 * x:.0%} con prob. {pi:.0%}"


VARIANTES = [
    Variante("reforma-laboratorio", 3, "retraso de un proveedor crítico, exigencia del inspector, oferta de reducción (Ackoff), coste del retraso para el cliente, tope presupuestario, incertidumbre y negociaciones sin coste",
             _reforma_laboratorio),
    Variante("recursos-compartidos", 2, "otra jefa de proyecto pide operarios prestados: planificador en paralelo con límite por periodo, ventanas con y sin retraso, penalizaciones de dos proyectos, subcontratación y óptimo de empresa frente a óptimo local",
             _recursos_compartidos, grafo="roy"),
    Variante("comite-valor-ganado", 3, "avance declarado por el contratista frente al estimado en obra: PV/EV/AC, índices y semáforo con las dos versiones, EAC, TCPI, previsión de plazo, certificación y verificación del avance",
             _comite_valor_ganado),
    Variante("oferta-comercial", 2, "qué plazo prometer: PERT frente a Monte Carlo (sesgo de fusión, criticidad), coste esperado de penalizaciones y bonificaciones por plazo ofertado, exigencia del 90 % y su precio",
             _oferta_comercial),
    Variante("comite-inversion", 3, "caso de inversión con mensajes contradictorios: escenarios y umbrales, contrato a precio fijo como seguro (VAN esperado y peor caso), apalancamiento y separación inversión/financiación",
             _comite_inversion, grafo="pert"),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
