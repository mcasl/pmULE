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
    "    return T, sigma, float(norm.cdf(z)), float(retraso_esp)\n"
    "\n"
    "def suma_caminos_criticos(p, d, unidad='semanas'):\n"
    "    '''Cada camino crítico, en orden de ejecución, escrito como suma de duraciones.'''\n"
    "    cal = p.calendar(d)\n"
    "    T = float(p.duration(d))\n"
    "    return '; '.join('–'.join(ordena_ruta(r, cal)) + ' = ' + ' + '.join(f'{float(d[a]):g}' for a in ordena_ruta(r, cal)) + f' = {T:g} {unidad}'\n"
    "                     for r in p.critical_path(d).values())\n"
    "\n"
    "def pasos_ackoff(p, d, tope, cur, pasos):\n"
    "    '''Paso a paso de Ackoff-Sasieni: caminos críticos, opciones que los cortan todos (con su CUR) y la elegida.'''\n"
    "    from itertools import combinations\n"
    "    R = p.path_matrix(dummies=False)\n"
    "    d = d.astype(float).copy(); filas = []; acumulado = 0.0\n"
    "    for paso, elegidas in sorted(pasos.items()):\n"
    "        L = R @ d[R.columns]; T = float(L.max())\n"
    "        cal = p.calendar(d)\n"
    "        criticas = [ordena_ruta(list(R.columns[R.loc[r] == 1]), cal) for r in L.index if np.isclose(L[r], T)]\n"
    "        conjuntos = [set(c) for c in criticas]\n"
    "        margen = {x for x in R.columns if d[x] > tope[x] and np.isfinite(cur[x])}\n"
    "        opciones = {x: float(cur[x]) for x in set.intersection(*conjuntos) & margen}\n"
    "        if len(conjuntos) > 1:\n"
    "            for a, b in combinations(sorted(set().union(*conjuntos) & margen), 2):\n"
    "                if all(a in s or b in s for s in conjuntos) and a not in opciones and b not in opciones:\n"
    "                    opciones[f'{a}+{b}'] = float(cur[a] + cur[b])\n"
    "        coste = float(sum(cur[x] for x in elegidas)); acumulado += coste\n"
    "        filas.append({'paso': paso + 1, 'duración antes': int(T), 'caminos críticos': '; '.join('–'.join(c) for c in criticas),\n"
    "                      'opciones con margen (CUR, k€)': '; '.join(f'{k} ({v:g})' for k, v in sorted(opciones.items(), key=lambda kv: kv[1])),\n"
    "                      'se acorta': ' + '.join(elegidas), 'coste del paso (k€)': coste, 'coste acumulado (k€)': acumulado,\n"
    "                      'duración después': int(T) - 1})\n"
    "        for x in elegidas:\n"
    "            d[x] -= 1\n"
    "    return pd.DataFrame(filas).set_index('paso')\n"
    "\n"
    "def explica_retraso_esperado(media, sigma, plazo, unidad='semanas'):\n"
    "    '''E[máx(0, T − plazo)] con T ~ N(media, σ), sin tipificar: (μ − plazo)·P(T > plazo) + σ²·f(plazo).'''\n"
    "    p_mas = 1 - norm.cdf(plazo, media, sigma); f = norm.pdf(plazo, media, sigma)\n"
    "    hoja = lambda x: f'{x:g}'.replace('.', ',')\n"
    "    return (f'E[máx(0, T − {plazo:g})] = (μ − plazo)·P(T > plazo) + σ²·f(plazo) = ({media:g} − {plazo:g})·{p_mas:.4f} + {sigma:.3f}²·{f:.4f} '\n"
    "            f'= **{(media - plazo) * p_mas + sigma ** 2 * f:.2f} {unidad}**, donde f(plazo) es la densidad normal en el plazo: '\n"
    "            f'`=DISTR.NORM.N({hoja(plazo)}; {hoja(media)}; {hoja(round(sigma, 3))}; FALSO)` = {f:.4f}')\n")


def _silencio():
    return contextlib.redirect_stdout(io.StringIO())


def _pl(n, singular, plural_=None):
    """«1 semana», «3 semanas» (como ``plural`` de pmule, para el texto que escribe el generador)."""
    v = float(n)
    txt = f"{int(round(v))}" if v.is_integer() else f"{v:.1f}"
    return f"{txt} {singular if v.is_integer() and abs(int(round(v))) == 1 else (plural_ or singular + 's')}"


def _y(items):
    """«A», «A y B», «A, B y C» (como ``lista_y``)."""
    items = [str(x) for x in items]
    return items[0] if len(items) == 1 else (", ".join(items[:-1]) + " y " + items[-1] if items else "ninguna")


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
    pr6 = _prob(p4, dur5, sd4, T0 + 1)[2]
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
    celdas.append(code("Markdown('Camino crítico de cada situación, escrito como suma de duraciones (en orden de ejecución):\\n\\n' + '\\n'.join(\n"
                       "    f'- {nombre}: {suma_caminos_criticos(p, df[\"duration\"])}' for nombre, df, p in\n"
                       f"    (('plan original', df0, p0), ('sólo el retraso del mobiliario', dfF, pF), ('sólo la inspección tras G', dfH, pH), ('los dos cambios', df2, p2))))"))
    aporta_H = x["TH"] - T0
    suman = retraso == delta + aporta_H
    celdas.append(md(f"El proveedor aporta {_pl(delta, 'semana')} porque F era crítica (holgura total 0): su retraso va entero "
                     f"al plazo ({T0} + {delta} = {x['TF']}). El inspector aporta {_pl(aporta_H, 'semana')}: H tenía "
                     f"{_pl(hol_H, 'semana')} de holgura (la ruta más larga que pasaba por ella, por D o E, sumaba "
                     f"{T0} − {hol_H} = {T0 - hol_H}), pero al ir detrás de G pasa a la ruta "
                     + "–".join(_camino(x['p0'], d).split("–")[:-1] + ["H", "I"])
                     + (f", que es el camino crítico anterior más la semana de H ({T0} + {int(d['H'])} = {x['TH']})"
                        if x["TH"] == T0 + int(d["H"]) else f", que dura {x['TH']}")
                     + ": la holgura que tenía no le sirve, porque estaba en otra ruta. "
                     + (f"Los efectos se suman tal cual ({delta} + {aporta_H} = {retraso}) porque los dos cambios caen sobre la "
                        "misma ruta, la crítica; no se sumarían si uno de ellos afectara a una ruta con holgura suficiente para absorberlo."
                        if suman else
                        f"Los efectos no se suman tal cual ({delta} + {aporta_H} = {delta + aporta_H}, pero juntos dan {retraso}): "
                        "no caen sobre la misma ruta.")
                     + f"\n\n**Respuesta:** si no se hace nada el laboratorio está listo en la semana {x['T2']}, con "
                     f"**{_pl(retraso, 'semana')} de retraso**: {delta} del proveedor y {aporta_H} del inspector."))
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
    celdas.append(md("Cada opción de *r* semanas son los *r* primeros pasos del algoritmo. Paso a paso (caminos críticos del "
                     "momento, actividades o parejas que los cortan todos y aún tienen margen hasta su tope, con su CUR, y la elegida):"))
    celdas.append(code('#| tbl-cap: "Ackoff-Sasieni paso a paso sobre el plan con los dos cambios"\n#| label: tbl-ackoffPasos\n'
                       "_, pasos_todos, _, _ = p2.ackoff(durF, tope, cur, reduction=retraso, report=False)\n"
                       "pasos_ackoff(p2, durF, tope, cur, pasos_todos)"))
    _todas = [a for acts in _ackoff(x["p2"], x["dF"], x["topeF"], cur, retraso)[0].values() for a in acts]
    _no_entran = [a for a in ("A", "C", "G", "I") if a not in _todas]
    _orden_cur = sorted(("A", "C", "G", "I"), key=lambda a: cur[a])
    celdas.append(md(f"Para recuperar {_pl(r_opt, 'semana')} el algoritmo acorta {actos_opt}: en cada paso elige la opción "
                     f"más barata que corta todos los caminos críticos (CUR de menor a mayor: "
                     + ", ".join(f"{a} {int(cur[a])}" for a in _orden_cur) + " k€/semana)"
                     + (f"; {_y(_no_entran)} no {'entra' if len(_no_entran) == 1 else 'entran'} en ninguna opción: en todos los "
                        f"pasos hay otra más barata que corta los mismos caminos críticos" if _no_entran else "")
                     + ". El coste total para la universidad de cada opción es sobrecoste + alquiler × semanas de retraso que quedan:\n\n"
                     + "\n".join(lineas) + "\n\n"
                     f"**Respuesta:** el coste total mínimo para la universidad es recuperar **{_pl(r_opt, 'semana')}** "
                     f"({opc[r_opt]:.0f} + {alq} × {retraso - r_opt} = {total[r_opt]:.0f} k€). Respetan el tope de {bolsa} k€ "
                     f"no acortar nada o recuperar hasta {_pl(r_opt, 'semana')}; la que más plazo recupera dentro del tope es también la de "
                     f"{_pl(r_opt, 'semana')}, así que aquí las dos perspectivas coinciden; recuperar más exige la modificación presupuestaria.\n\n"
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
    celdas.append(code(f"_r = {r_opt}\n"
                       "_dur = duraciones_opcion[0]\n"
                       "_ruta = ordena_ruta(list(p2.critical_path(_dur).values())[0], p2.calendar(_dur))\n"
                       "_T, _s, _, _ = resumen_prob(p2, duraciones_opcion[_r], sd, plazo)\n"
                       "Markdown(f'''Sin reducciones el camino crítico es {\"–\".join(_ruta)} y su desviación típica, con las varianzas "
                       "(σ²) de sus actividades: {explica_sigma_ruta(_ruta, sd ** 2)} semanas. Las reducciones acortan duraciones pero "
                       "no cambian las varianzas, por eso σ cambia poco de una opción a otra (si aparecen varios caminos críticos se toma el de mayor varianza).\\n\\n"
                       "Por ejemplo, para la opción de recuperar {plural(_r, \"semana\")} (duración media {_T:g}):\\n\\n"
                       "- Probabilidad de cumplir: {explica_probabilidad(_T, _s, plazo)}.\\n"
                       "- Retraso esperado: {explica_retraso_esperado(_T, _s, plazo)}.\\n"
                       "- Coste total esperado = sobrecoste + {coste_semana} × retraso esperado = "
                       "{float(tabla.loc[_r, \"sobrecoste (k€)\"]):g} + {coste_semana} × {resumen_prob(p2, duraciones_opcion[_r], sd, plazo)[3]:.2f} = "
                       "**{float(tabla.loc[_r, \"sobrecoste (k€)\"]) + coste_semana * resumen_prob(p2, duraciones_opcion[_r], sd, plazo)[3]:.1f} k€**.''')"))
    celdas.append(md(f"La desviación típica del camino crítico es de {probs[0][1]:.1f} semanas, grande comparada con las "
                     f"diferencias que se están discutiendo. Dos consecuencias: la opción «recuperar {retraso} semanas» "
                     f"promete terminar en la semana {T0} *en media*, es decir, con un {probs[retraso][2]:.0%} de probabilidad, "
                     f"no con seguridad; y en coste esperado la mejor opción es "
                     + (f"recuperar {r_esp} semana{'s' if r_esp != 1 else ''}" if r_esp else "no acortar nada")
                     + f" ({esp[r_esp]:.1f} k€ frente a {esp[retraso]:.1f} de la de {retraso})"
                     + (", la misma que sin incertidumbre" if r_esp == r_opt else f", y no la de {r_opt} que salía antes")
                     + ". Con la incertidumbre delante, las reducciones de pago compran menos de lo que parecían, y las "
                     "medidas que recortan plazo sin coste (apartado 4) ganan valor.\n\n"
                     + ("**Respuesta:** la elección no cambia" if r_esp == r_opt else "**Respuesta:** la elección cambia")
                     + f": en coste esperado la mejor opción es recuperar {_pl(r_esp, 'semana')} ({esp[r_esp]:.1f} k€). Es también la "
                     "más robusta en el sentido de que es la que menos cuesta en promedio cuando la duración real se desvía de la media; "
                     f"ninguna opción de pago asegura el plazo: incluso recuperando las {retraso} semanas la probabilidad de terminar en la "
                     f"semana {T0} es {probs[retraso][2]:.2f}, y con la opción elegida es {probs[r_esp][2]:.2f}."))
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
    preds_par = list(x["preds4"]); preds_par[x["nombres4"].index("H")] = "D,E,G"
    _, p_par = _proy(preds_par, dur4, x["nombres4"])
    T_par = _T(p_par, dur4)
    celdas.append(md("Para cuantificar cuánto recupera cada negociación, se aplica cada una por separado sobre el plan con los dos "
                     "cambios (la inspección en dos fases devuelve H a su posición original, detrás de D y E; la entrega parcial "
                     "parte F en F1 y F2):"))
    celdas.append(code('#| tbl-cap: "Plazo que recupera cada negociación"\n#| label: tbl-negociaciones\n'
                       "preds_parcial = list(preds4); preds_parcial[nombres4.index('H')] = 'D,E,G'      # sólo entrega parcial: H sigue tras G\n"
                       "df_par, p_par = proyecto_de(preds_parcial, dur4, nombres4)\n"
                       "_T2 = int(p2.duration(durF))\n"
                       "filas = []\n"
                       "for nombre, p, d in (('ninguna (los dos cambios)', p2, durF), ('sólo la inspección en dos fases', pF, durF),\n"
                       "                     ('sólo la entrega parcial', p_par, dur4), ('las dos', p4, dur4)):\n"
                       "    T = int(p.duration(d))\n"
                       "    filas.append({'negociación': nombre, 'duración': T, 'semanas recuperadas': _T2 - T, f'retraso sobre {plazo} (−: margen)': T - plazo,\n"
                       "                  'camino crítico (suma)': suma_caminos_criticos(p, d)})\n"
                       "pd.DataFrame(filas).set_index('negociación')"))
    T2 = x["T2"]
    rec_ins, rec_par, rec_dos = T2 - x["TF"], T2 - T_par, T2 - T4
    celdas.append(md(f"La inspección en dos fases sola recupera {_pl(rec_ins, 'semana')} y la entrega parcial sola "
                     f"{_pl(rec_par, 'semana')}; juntas, {_pl(rec_dos, 'semana')}"
                     + (f" ({rec_ins} + {rec_par} = {rec_ins + rec_par}: se suman porque cada una actúa sobre un tramo distinto del camino crítico)"
                        if rec_dos == rec_ins + rec_par else
                        f" ({rec_ins} + {rec_par} = {rec_ins + rec_par} ≠ {rec_dos}: no se suman, porque al aplicar una cambia el camino "
                        "crítico y la otra actúa ya sobre una ruta distinta)")
                     + ". "
                     + (f"Con las dos negociaciones el plan vuelve a {T4} semanas en media sin gastar un euro"
                        if T4 == T0 else
                        f"Con las dos negociaciones el plan baja a {T4} semanas en media, {_pl(T0 - T4, 'semana')} antes del plazo, sin gastar un euro")
                     + (f", pero fíjese en que F2 queda crítica: cualquier semana más del proveedor vuelve a retrasar el proyecto."
                        if hol_F2 == 0 else
                        f"; F2 conserva {hol_F2} semana{'s' if hol_F2 != 1 else ''} de holgura, que es todo el margen que hay frente a un nuevo retraso del proveedor.")
                     + " Los riesgos y a quién preguntar: al inspector, si acepta la inspección en dos fases por escrito; al "
                     f"proveedor, qué garantía da de las {delta} semanas; al técnico, si instalar sobre las mesas sin campana ni "
                     "armarios es seguro y no obliga a desmontar después; y a Gerencia, si la bolsa puede reservarse para una "
                     "reducción *contingente* (acortar I al final sólo si F2 se retrasa). El solape formación–instalación no se "
                     "modela: depende de una relación comienzo-comienzo que el técnico propone sólo como último recurso.\n\n"
                     f"**Respuesta:** aceptando la inspección en dos fases y la entrega parcial se recuperan {_pl(rec_dos, 'semana')}"
                     + (f": todo el retraso" if T4 == T0 else f": todo el retraso ({retraso}) y {_pl(T0 - T4, 'semana')} de margen" if T4 < T0 else
                        f" de las {retraso} de retraso; quedan {_pl(T4 - T0, 'semana')}")
                     + f"; P(terminar ≤ {T0}) = {pr4[2]:.2f}. Antes de contar con ellas hay que confirmarlas con el inspector y con el proveedor."))
    celdas += apartado(5)
    celdas.append(code("dur5 = dur4.copy(); dur5['A'] -= 1                      # plan negociado + reducción de A (una semana)\n"
                       "_, sigma4, _, _ = resumen_prob(p4, dur4, sd4, plazo)\n"
                       "T5, sigma5, _, _ = resumen_prob(p4, dur5, sd4, plazo)\n"
                       "Markdown(f'''Cálculos de apoyo para la decisión:\\n\\n"
                       "- Plan negociado: {suma_caminos_criticos(p4, dur4)}; σ = {sigma4:.3f}; {explica_probabilidad(T4, sigma4, plazo)}.\\n"
                       "- Plan negociado acortando además A una semana: {suma_caminos_criticos(p4, dur5)}; σ = {sigma5:.3f}; "
                       "{explica_probabilidad(T5, sigma5, plazo)}; y {explica_probabilidad(T5, sigma5, plazo + 1)}.''')"))
    celdas.append(md("No hay una única respuesta correcta; sí hay respuestas mal justificadas. Una decisión defendible con los "
                     "números anteriores:\n\n"
                     f"1. Negociar hoy la inspección en dos fases y la entrega parcial del mobiliario (coste cero; recuperan "
                     + (f"las {retraso} semanas de retraso" if T4 == T0 else
                        f"las {retraso} semanas de retraso y dejan {_pl(T0 - T4, 'semana')} de margen" if T4 < T0 else
                        f"{retraso - (T4 - T0)} de las {retraso} semanas")
                     + (" si el proveedor cumple con F2)" if hol_F2 == 0 else ")")
                     + (". La entrega parcial sola ya recupera lo mismo que las dos juntas: la inspección en dos fases no adelanta "
                        "la fecha, pero da margen a la ruta de G y H, así que conviene pedirla sin depender de ella." if rec_par == rec_dos else ".")
                     + "\n"
                     f"2. Contratar ya la reducción de A ({cA} k€): es la más barata, A está en curso y cada semana ganada "
                     f"ahora es margen contra la incertidumbre del proveedor. Con ello la duración media baja a {pr5[0]:.0f} y la "
                     f"probabilidad de terminar en la {T0} sube del {pr4[2]:.0%} al {pr5[2]:.0%}.\n"
                     f"3. Reservar la reducción de I ({cI} k€) como medida contingente: se decide en la semana {semana_decision}, cuando "
                     f"se sepa si F2 ha llegado. Sobrecoste máximo {cA + cI} k€, "
                     + ("dentro de la bolsa; sin necesidad de modificación presupuestaria." if cA + cI <= bolsa else
                        f"que supera la bolsa de {bolsa} k€ en {cA + cI - bolsa} k€: habría que pedir a Gerencia la ampliación ya, por si acaso.")
                     + "\n"
                     f"4. Prometer a la vicerrectora la semana {T0} (probabilidad {pr5[2]:.2f} con la reducción de A, y más si se "
                     f"activa la de I) y la {T0 + 1} como peor caso razonable (P(T ≤ {T0 + 1}) = {pr6:.2f}), y "
                     f"pedirle que no comprometa el alquiler externo hasta la semana {semana_decision}. Sobrecoste: {cA} k€ seguros "
                     f"y {cI} k€ más si hace falta la reducción de I.\n\n"
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
    "def programar(limite, L, registro=None):\n"
    "    TQPlanificador en paralelo con límite por periodo: la semana s admite limite[s] operarios (L si no está en limite).\n"
    "    Recorre las semanas; en cada una, por orden de holgura total, entra cada actividad disponible (predecesoras terminadas)\n"
    "    si cabe en el límite de TODAS las semanas que ocuparía; si no, espera. Devuelve los inicios y fines de cada actividad.\n"
    "    Si se pasa una lista en registro, anota cada espera: la primera semana en la que la actividad no cabe y por qué.TQ\n"
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
    "            elif registro is not None:\n"
    "                s = next(s for s in semanas if carga.get(s, 0) + int(R[a]) > int(limite.get(s, L)))\n"
    "                registro.append({'semana en que podría empezar': t + 1, 'actividad': a, 'operarios': int(R[a]),\n"
    "                                 'semanas que ocuparía': f'{t + 1}–{t + int(D[a])}', 'primera semana sin sitio': s,\n"
    "                                 'carga ya asignada + operarios > límite': f'{carga.get(s, 0)} + {int(R[a])} = {carga.get(s, 0) + int(R[a])} > {int(limite.get(s, L))}'})\n"
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
    # a igual coste para la empresa se prefiere la opción que no retrasa a nadie (y, después, la que no retrasa al cliente)
    preferencia = ["subcontratar", "prestar más tarde", "negarse", "prestar como se pide"]
    mejor_empresa = min(coste, key=lambda k_: (round(coste[k_], 6), preferencia.index(k_)))
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
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total', 'H_libre']]"))
    picos_todos = [int(t) for t in total.index[total == L]]
    en_pico = {t: [a for a in D.index if cal.loc[a, "inicio_mas_temprano"] < t <= cal.loc[a, "fin_mas_temprano"]] for t in picos_todos}
    celdas.append(md(f"**Respuesta:** el plan de inicios más tempranos llega al máximo de {L} operarios en "
                     f"{'las semanas' if len(picos_todos) > 1 else 'la semana'} {_y(picos_todos)}"
                     + ("; en todas coinciden " + _y(en_pico[picos_todos[0]]) if len({tuple(v) for v in en_pico.values()}) == 1 else
                        " (" + "; ".join(f"semana {t}: {_y(v)}" for t, v in en_pico.items()) + ")")
                     + f". Las actividades críticas (holgura total 0) son {_y(criticas)}. "
                     f"La ventana que pide Delta ({a_req}–{b_req}) "
                     + (f"incluye {'las semanas de pico' if len(picos) > 1 else 'la semana de pico'} {_y(picos)}"
                        if set(picos) & set(range(a_req, b_req + 1)) else "no incluye ninguna semana de pico")
                     + f". Prestar {q} {plural} en una semana en la que la carga supera {L - q} obliga a aplazar alguna actividad; "
                     "que eso retrase el proyecto depende de si la actividad aplazada tiene holgura suficiente (y de si al aplazarla "
                     "arrastra a otras hasta agotar la suya)."))
    celdas += apartado(2)
    celdas.append(code(f'#| fig-cap: "Programación con {L - q} operarios en las semanas {a_req}–{b_req} y {L} en las demás (fila Máximo); las filas 💤 son las esperas"\n#| label: fig-ganttPrestamo\n'
                       "T_prestamo, esperas, dibujo = simular_prestamo(ventana, q, L)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(md(f"El planificador recorre las semanas; al principio de cada una ordena las actividades disponibles (predecesoras "
                     "terminadas) por holgura total, de menor a mayor (en caso de empate, por orden alfabético), y deja empezar cada una "
                     f"si cabe en el límite de todas las semanas que ocuparía ({L - q} dentro de la ventana {a_req}–{b_req}, {L} fuera). "
                     "Cada vez que una actividad disponible no cabe, se anota la primera semana en la que se pasaría del límite:"))
    celdas.append(code('#| tbl-cap: "Esperas con el préstamo pedido: por qué no cabe cada actividad"\n#| label: tbl-esperasPrestamo\n'
                       "registro = []\n"
                       "_ = programar({s: L - q for s in range(ventana[0], ventana[1] + 1)}, L, registro)\n"
                       "pd.DataFrame(registro).set_index('semana en que podría empezar') if registro else Markdown('Ninguna actividad tiene que esperar.')"))
    celdas.append(code("_ret = T_prestamo - plazo\n"
                       "_lineas = [f'- {a} espera {plural(v, \"semana\")} y tenía holgura total {int(calendario.loc[a, \"H_total\"])}.' for a, v in esperas.items()]\n"
                       "_criticas = [a for a, v in esperas.items() if calendario.loc[a, 'H_total'] == 0 and v == _ret]\n"
                       "if _ret == 0:\n"
                       "    _porque = 'las holguras absorben todas las esperas'\n"
                       "elif _criticas:\n"
                       "    _porque = lista_y(_criticas) + (' es crítica' if len(_criticas) == 1 else ' son críticas') + ': su espera pasa entera al plazo'\n"
                       "else:\n"
                       "    _porque = f'las esperas superan en {plural(_ret, \"semana\")} la holgura de las rutas que las contienen'\n"
                       "_quien = (('espera ' if len(esperas) == 1 else 'esperan ') + lista_y(list(esperas))) if esperas else 'no espera ninguna actividad'\n"
                       "Markdown('\\n'.join(_lineas) + f'\\n\\n**Respuesta:** con el préstamo en las semanas {ventana[0]}–{ventana[1]} {_quien}; '\n"
                       "         f'el proyecto pasa de {plazo} a **{T_prestamo} semanas** ({_porque}): {plural(_ret, \"semana\")} de retraso y '\n"
                       "         f'{penalizacion} × {_ret} = **{penalizacion * _ret} k€** de penalización.')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Duración del proyecto según la ventana del préstamo"\n#| label: tbl-ventanas\n'
                       f"w = {w}\n"
                       "filas = []\n"
                       "for a in range(1, plazo - w + 2):\n"
                       "    T_a, esperas_a, _ = simular_prestamo((a, a + w - 1), q, L)\n"
                       "    cmax = max(int(cargas[t]) for t in range(a, a + w))\n"
                       "    hol = ', '.join(f'{x} ({v} sem.; holgura {int(calendario.loc[x, \"H_total\"])})' for x, v in esperas_a.items())\n"
                       "    if not esperas_a:\n"
                       "        motivo = f'carga máxima {cmax} ≤ {L - q}: no hay que aplazar nada'\n"
                       "    elif T_a == plazo:\n"
                       "        motivo = f'carga máxima {cmax} > {L - q}; esperan {hol}: lo absorbe la holgura'\n"
                       "    else:\n"
                       "        motivo = f'carga máxima {cmax} > {L - q}; esperan {hol}: no basta la holgura'\n"
                       "    filas.append({'semanas': f'{a}–{a + w - 1}', 'carga antes del préstamo': ', '.join(str(int(cargas[t])) for t in range(a, a + w)),\n"
                       "                  'duración': T_a, 'retraso': T_a - plazo, 'explicación': motivo})\n"
                       "ventanas = pd.DataFrame(filas).set_index('semanas')\n"
                       "ventanas"))
    libres = [a for a, (dur, _) in tabla.items() if dur == T0]
    con_retraso = [a for a, (dur, _) in tabla.items() if dur > T0]
    def _ventanas(inicios):
        if len(inicios) > 2 and inicios == list(range(inicios[0], inicios[-1] + 1)):
            return f"todas las que empiezan entre la semana {inicios[0]} y la {inicios[-1]} ({inicios[0]}–{inicios[0] + w - 1} a {inicios[-1]}–{inicios[-1] + w - 1})"
        return _y([f"{a}–{a + w - 1}" for a in inicios])
    desde_alt = all(a in libres for a in tabla if a >= a_alt)
    _, esperas_req = tabla[a_req]
    crit_req = [a for a, v in esperas_req.items() if cal.loc[a, "H_total"] == 0]
    celdas.append(md(f"Se puede prestar sin mover el plazo en {_ventanas(libres)}; retrasan el proyecto {_ventanas(con_retraso)}. "
                     f"En las ventanas sin retraso, o la carga de esas semanas no pasa de {L - q} y no hay que aplazar nada, o lo que "
                     "se aplaza tiene holgura suficiente (columna «explicación»). "
                     f"La ventana que pide Delta ({a_req}–{b_req}) cuesta {_pl(retraso, 'semana')} porque su carga pasa de {L - q} "
                     + (f"y obliga a aplazar {_y(crit_req)}, {'que es crítica' if len(crit_req) == 1 else 'que son críticas'}."
                        if crit_req else
                        "y " + _y([f"{a_} espera {_pl(v_, 'semana')} con holgura {int(cal.loc[a_, 'H_total'])}" for a_, v_ in esperas_req.items()])
                        + ": lo que la espera excede a la holgura acaba empujando a las actividades críticas.")
                     + "\n\n**Respuesta:** "
                     + (f"el jefe de obra tiene razón: a partir de la semana {a_alt} todas las ventanas de {w} semanas dejan el plazo "
                        f"en {T0} semanas, así que prestar desde entonces no cuesta nada a este proyecto (a Delta le cuesta esperar "
                        f"{_pl(a_alt - a_req, 'semana')})."
                        if desde_alt else
                        f"el jefe de obra tiene razón sólo en parte: la ventana {a_alt}–{b_alt} no retrasa el proyecto, pero no todas "
                        f"las posteriores están libres (retrasan {_ventanas([a for a in con_retraso if a > a_alt])}).")))
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
    etiqueta = {"negarse": "negarse", "prestar como se pide": f"prestar en {a_req}–{b_req}",
                "prestar más tarde": f"prestar en {a_alt}–{b_alt}", "subcontratar": f"subcontratar {q} para Delta"}
    minimos = [k_ for k_, v in coste.items() if np.isclose(v, coste[mejor_empresa])]
    lineas_coste = [
        f"- Negarse: Delta se retrasa las {w} semanas → {pen2} × {w} = **{pen2 * w} k€** (mi proyecto, 0).",
        f"- Prestar en {a_req}–{b_req}: mi proyecto se retrasa {_pl(retraso, 'semana')} → {pen} × {retraso} = **{pen * retraso} k€** (Delta, 0).",
        f"- Prestar en {a_alt}–{b_alt}: mi plazo no se mueve (apartado 3); Delta espera {a_alt} − {a_req} = {_pl(espera_alt, 'semana')} → "
        f"{pen2} × {espera_alt} = **{pen2 * espera_alt} k€**.",
        (f"- Subcontratar: {s:g} k€ por operario y semana × {q} × {w} semanas = **{s * q * w:.1f} k€** (mi proyecto, 0); llega a tiempo "
         f"porque la ventana empieza en la semana {a_req} y RR. HH. puede tenerlos desde la semana 2." if sub_posible else
         f"- Subcontratar: no es posible, porque la ventana empieza en la semana {a_req} y RR. HH. necesita una semana de preaviso "
         "(los subcontratados llegarían, como pronto, en la semana 2)."),
    ]
    coincide = mejor_empresa in ("negarse", "prestar más tarde", "subcontratar")
    celdas.append(md("Coste de cada opción (penalizaciones y subcontratación, en k€):\n\n" + "\n".join(lineas_coste) + "\n\n"
                     "**Respuesta:** para la empresa la opción más barata es "
                     + _y([f"**{etiqueta[k_]}**" for k_ in minimos])
                     + f" ({coste[mejor_empresa]:.1f} k€{', empatadas' if len(minimos) > 1 else ''}). Para mi proyecto cuestan 0 negarse, "
                     f"prestar en {a_alt}–{b_alt}" + (" y que se subcontrate para Delta" if sub_posible else "")
                     + f"; la única que me cuesta es prestar como se pide ({pen * retraso} k€). "
                     + ("Por tanto el óptimo de la empresa es también uno de los óptimos de mi proyecto, pero no el único: si yo eligiera "
                        f"negarme, la empresa pagaría {pen2 * w - coste[mejor_empresa]:.1f} k€ más."
                        if coincide and mejor_empresa != "negarse" else
                        "Por tanto el óptimo de la empresa coincide con el de mi proyecto." if coincide else
                        "Por tanto no coinciden: lo mejor para la empresa es justo lo único que le cuesta dinero a mi proyecto.")
                     + " El director de operaciones pide la columna «empresa»: ahí está el conflicto entre la optimización local "
                     "(cada jefe de proyecto defiende su plazo) y la global."))
    celdas += apartado(5)
    if mejor_empresa == "prestar más tarde":
        hibrido = s * q * espera_alt
        decision = (f"prestar los {q} {plural} desde la semana {a_alt} (gratis para mi plazo) y proponer a Delta que absorba "
                    f"{'la semana' if espera_alt == 1 else f'las {espera_alt} semanas'} de espera ({pen2} × {espera_alt} = {pen2 * espera_alt} k€)"
                    + (f". Mejor aún: cubrir esa espera con {q} {plural} subcontratado{'s' if q > 1 else ''} ({s:g} × {q} × {espera_alt} = "
                       f"{hibrido:.1f} k€ < {pen2 * espera_alt} k€) y pasar después a Delta los míos, en las semanas {a_alt}–{b_req}, "
                       "que no retrasan mi proyecto; la empresa paga sólo la subcontratación"
                       if sub_posible and hibrido < pen2 * espera_alt else ""))
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
    elif mejor_empresa == "prestar más tarde" and sub_posible and s * q * espera_alt < pen2 * espera_alt:
        sem_sub = f"la semana {a_req}" if espera_alt == 1 else f"las semanas {a_req}–{a_alt - 1}"
        msg_jefa = (f"En {a_req}–{b_req} prestar {q} {plural} me cuesta {sem(retraso)} de plazo ({pen * retraso} k€). Te propongo "
                    f"{q} subcontratado{'s' if q > 1 else ''} en {sem_sub} ({s * q * espera_alt:.1f} k€) y los míos desde la semana {a_alt}, "
                    "que no retrasan mi entrega: tú no te retrasas y la empresa paga lo mínimo.")
        msg_cliente = f"Mantenemos la entrega en la semana {T0}."
        msg_rrhh = (f"Necesito {q} {plural} subcontratado{'s' if q > 1 else ''} para {sem_sub} (proyecto Delta); "
                    "confirmad hoy para respetar el preaviso.")
    elif mejor_empresa == "prestar más tarde":
        msg_jefa = (f"En {a_req}–{b_req} prestar {q} {plural} me cuesta {sem(retraso)} de plazo ({pen * retraso} k€). Te los doy sin coste "
                    f"desde la semana {a_alt}: esperar {sem(espera_alt)} te cuesta {pen2 * espera_alt} k€, y sigue siendo lo más barato para la empresa.")
        msg_cliente = f"Mantenemos la entrega en la semana {T0}."
        msg_rrhh = "Por ahora no hace falta subcontratar; os aviso si cambia."
    elif mejor_empresa == "subcontratar":
        msg_jefa = (f"En {a_req}–{b_req} prestar {q} {plural} me cuesta {sem(retraso)} de plazo ({pen * retraso} k€). Lo más barato es que "
                    f"RR. HH. subcontrate {q} para ti ({s * q * w:.1f} k€); lo pido hoy. Si fallara, te presto los míos desde la semana {a_alt}.")
        msg_cliente = f"Mantenemos la entrega en la semana {T0}."
        msg_rrhh = f"Necesito {q} {plural} subcontratado{'s' if q > 1 else ''} para las semanas {a_req}–{b_req}; confirmad hoy para respetar el preaviso."
    else:
        msg_jefa = f"No puedo prestarlos en {a_req}–{b_req} sin retrasar mi entrega; lo elevamos al director de operaciones."
        msg_cliente = f"Mantenemos la entrega en la semana {T0}."
        msg_rrhh = "Por ahora no hace falta subcontratar; os aviso si cambia."
    celdas.append(md(f"Una decisión defendible: {decision}. Mensajes:\n\n"
                     f"- *Jefa de Delta:* «{msg_jefa}»\n"
                     "- *Director de operaciones:* «Adjunto la tabla con el coste de cada opción para los dos proyectos y para la empresa; "
                     f"la más barata en conjunto es {etiqueta[mejor_empresa]} ({coste[mejor_empresa]:.1f} k€). Si decides otra, pido que la "
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
                       "    return 'verde' if indice >= 0.95 else ('ámbar' if indice >= 0.85 else 'rojo')\n"
                       "def explica_indicadores(detalle, ind):\n"
                       "    # PV, EV y AC como suma de las actividades con valor; CV, SV, CPI y SPI con los números sustituidos\n"
                       "    suma = lambda col: ' + '.join(f'{a}({v:.1f})' for a, v in detalle[col].items() if v > 0)\n"
                       "    return (f'- PV = {suma(\"PV_i\")} = **{ind.PV:.1f} k€**\\n'\n"
                       "            f'- EV = {suma(\"EV_i\")} = **{ind.EV:.1f} k€**\\n'\n"
                       "            f'- AC = {suma(\"AC\")} = **{ind.AC:.1f} k€**\\n'\n"
                       "            f'- CV = EV − AC = {ind.EV:.1f} − {ind.AC:.1f} = **{ind.CV:.1f} k€**; SV = EV − PV = {ind.EV:.1f} − {ind.PV:.1f} = **{ind.SV:.1f} k€**\\n'\n"
                       "            f'- CPI = EV / AC = {ind.EV:.1f} / {ind.AC:.1f} = **{ind.CPI:.3f}**; SPI = EV / PV = {ind.EV:.1f} / {ind.PV:.1f} = **{ind.SPI:.3f}**')"))
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
    celdas.append(code("Markdown('Notación: PV, valor planificado (lo que debía estar hecho, a precio de presupuesto); EV, valor ganado '\n"
                       "         '(lo hecho, a precio de presupuesto); AC, coste real; CV y SV, variaciones de coste y de plazo; CPI y SPI, '\n"
                       "         f'índices de rendimiento del coste y del plazo. Con el informe del contratista, al final de la semana {semana_control}:\\n\\n'\n"
                       "         + explica_indicadores(detalle_rep, ind_rep) +\n"
                       "         '\\n\\nCriterio del semáforo (adoptado aquí): verde si el índice es ≥ 0.95, ámbar si está entre 0.85 y 0.95 y rojo si es < 0.85.\\n\\n'\n"
                       "         f'**Respuesta:** semáforo de plazo **{semaforo(ind_rep.SPI)}** (SPI = {ind_rep.SPI:.2f}) y coste **{semaforo(ind_rep.CPI)}** '\n"
                       "         f'(CPI = {ind_rep.CPI:.2f}). Tal como lo cuenta el contratista, el proyecto va bien.')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Lo mismo con la estimación del jefe de equipo para la actividad discutida"\n#| label: tbl-detalleReal\n'
                       f"seguimiento_real = seguimiento.copy(); seguimiento_real.loc[actividad_discutida, 'performed'] = {real}\n"
                       "detalle_real, ind_real = indicadores(seguimiento_real)\n"
                       "comparacion = pd.DataFrame({'informe del contratista': ind_rep, 'estimación del jefe de equipo': ind_real})\n"
                       "comparacion['diferencia'] = comparacion.iloc[:, 1] - comparacion.iloc[:, 0]\n"
                       "comparacion.round(3)"))
    celdas.append(code("BAC_X = float(datos_proyecto.loc[actividad_discutida, 'BAC'])\n"
                       f"Markdown(f'De los tres datos de partida sólo cambia EV, que baja {{ind_rep.EV - ind_real.EV:.1f}} k€ (= {{BAC_X:.0f}} × ({rep} − {real}) %); '\n"
                       "         'PV y AC no se mueven, así que cambian los cuatro indicadores que dependen de EV:\\n\\n'\n"
                       "         + explica_indicadores(detalle_real, ind_real) +\n"
                       "         f'\\n\\n**Respuesta:** CV y SV bajan {ind_rep.EV - ind_real.EV:.1f} k€ cada una; el CPI pasa de {ind_rep.CPI:.3f} a {ind_real.CPI:.3f} '\n"
                       "         f'y el SPI de {ind_rep.SPI:.3f} a {ind_real.SPI:.3f}; semáforo de plazo **{semaforo(ind_real.SPI)}** y de coste **{semaforo(ind_real.CPI)}**.')"))
    celdas.append(md(f"Un único porcentaje mueve tanto los índices porque CPI y SPI son cocientes con EV en el numerador: si EV baja "
                     f"{sobrepago:.1f} k€ con PV y AC fijos, el CPI baja {sobrepago:.1f} / {AC_t:.1f} = {sobrepago / AC_t:.3f} y el SPI "
                     f"{sobrepago:.1f} / {PV_t:.1f} = {sobrepago / PV_t:.3f}. Esos {sobrepago:.1f} k€ (BAC de {X}, {int(BAC[X])} k€, el "
                     f"{BAC[X] / BAC_total:.0%} del presupuesto, por {rep - real} puntos de avance) son el {sobrepago / e['EV_rep']:.0%} "
                     f"del valor ganado declarado ({e['EV_rep']:.1f} k€), que a mitad de proyecto aún es pequeño. El valor ganado es tan bueno "
                     "como la medida del avance físico que lo alimenta."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Previsiones de coste y plazo con las dos versiones"\n#| label: tbl-previsiones\n'
                       "duracion_plan = int(proyecto.duration(datos_proyecto['duration']))\n"
                       "prev = pd.DataFrame({nombre: {'EAC = BAC / CPI': BAC_total / ind['CPI'], 'VAC = BAC − EAC': BAC_total - BAC_total / ind['CPI'],\n"
                       "                              'TCPI = (BAC − EV) / (BAC − AC)': (BAC_total - ind['EV']) / (BAC_total - ind['AC']),\n"
                       "                              'duración ≈ T / SPI': duracion_plan / ind['SPI'],\n"
                       "                              'penalización ≈ (T/SPI − T) × pen': max(0.0, duracion_plan / ind['SPI'] - plazo) * penalizacion}\n"
                       "                     for nombre, ind in (('informe del contratista', ind_rep), ('estimación del jefe de equipo', ind_real))})\n"
                       "prev.round(2)"))
    celdas.append(code("def explica_prevision(ind):\n"
                       "    eac = BAC_total / ind.CPI; t = duracion_plan / ind.SPI\n"
                       "    return (f'EAC = {BAC_total:.0f} / {ind.CPI:.3f} = **{eac:.1f} k€**; VAC = {BAC_total:.0f} − {eac:.1f} = **{BAC_total - eac:.1f} k€**; '\n"
                       "            f'TCPI = ({BAC_total:.0f} − {ind.EV:.1f}) / ({BAC_total:.0f} − {ind.AC:.1f}) = **{(BAC_total - ind.EV) / (BAC_total - ind.AC):.3f}**; '\n"
                       "            f'duración ≈ {duracion_plan} / {ind.SPI:.3f} = **{t:.1f} semanas**; penalización ≈ máx(0, {t:.1f} − {plazo}) × {penalizacion} = '\n"
                       "            f'**{max(0.0, t - plazo) * penalizacion:.1f} k€**')\n"
                       "Markdown('EAC (estimación del coste final), VAC (variación al final), TCPI (índice de rendimiento que haría falta en lo que '\n"
                       "         'queda para acabar con el presupuesto BAC):\\n\\n'\n"
                       "         f'- Informe del contratista: {explica_prevision(ind_rep)}.\\n- Estimación del jefe de equipo: {explica_prevision(ind_real)}.')"))
    celdas.append(code("semanas_trabajadas = int(seguimiento.loc[actividad_discutida, 'actual_duration'])\n"
                       "d_plan = int(datos_proyecto.loc[actividad_discutida, 'duration'])\n"
                       f"semanas_ganadas = {real} / 100 * d_plan\n"
                       "retraso_directo = semanas_trabajadas - semanas_ganadas                 # el resto, al ritmo planificado\n"
                       f"retraso_ritmo = semanas_trabajadas / ({real} / 100) - d_plan           # el resto, al ritmo actual\n"
                       f"Markdown(f'''Estimación directa: {{actividad_discutida}} lleva {{semanas_trabajadas}} semanas y está al {real} %, es decir, ha ganado "
                       f"{real / 100:.2f} × {{d_plan}} = {{semanas_ganadas:.1f}} semanas de las {{d_plan}} planificadas:\\n\\n"
                       "- si el resto se hace al ritmo planificado, retraso = semanas trabajadas − semanas ganadas = {semanas_trabajadas} − {semanas_ganadas:.1f} = "
                       "**{retraso_directo:.1f} semanas**;\\n"
                       f"- si se mantiene el ritmo actual, duración = {{semanas_trabajadas}} / {real / 100:.2f} = {{semanas_trabajadas / ({real} / 100):.1f}} semanas, "
                       "retraso = {semanas_trabajadas / (" + str(real) + " / 100):.1f} − {d_plan} = **{retraso_ritmo:.1f} semanas**.\\n\\n"
                       "Al ser crítica (holgura 0), ese retraso va entero al plazo si no se corrige: entre {retraso_directo:.1f} × {penalizacion} = "
                       "{retraso_directo * penalizacion:.0f} y {retraso_ritmo:.1f} × {penalizacion} = {retraso_ritmo * penalizacion:.0f} k€ de penalización.''')"))
    celdas.append(md(f"La estimación directa por el "
                     f"avance de {X} ({retraso_X:.1f} semanas de retraso como mínimo, {retraso_ritmo:.1f} si no cambia el ritmo) es más "
                     "fiable que T/SPI: el SPI mezcla actividades críticas y no críticas (un adelanto en una con holgura compensa un retraso "
                     "en una crítica) y, al medirse en dinero, tiende a 1 al final del proyecto aunque se termine tarde.\n\n"
                     f"**Respuesta:** con el informe, EAC = {EAC_rep:.1f} k€ y fin en la semana {T_rep:.1f}; con el jefe de equipo, "
                     f"EAC = {EAC_real:.1f} k€ (VAC = {BAC_total - EAC_real:.1f} k€) y fin en la semana {T_real:.1f}. Por el ritmo de {X}, "
                     f"el retraso es de {retraso_X:.1f} a {retraso_ritmo:.1f} semanas, con {retraso_X * pen:.0f} a {retraso_ritmo * pen:.0f} k€ "
                     f"de penalización ({pen} k€/semana). Para el comité: {sem_rojo(e['SPI_real'])} en plazo si se cree al jefe de equipo, "
                     "y no se puede dar verde con una discrepancia así sin resolver."))
    celdas += apartado(4)
    celdas.append(md(f"**Respuesta:** certificar este mes el {real} % de {X} ({int(BAC[X])} × {real / 100:.2f} = "
                     f"{float(BAC[X]) * real / 100:.1f} k€), no el {rep} % que pide el contratista.\n\n"
                     f"Certificar el {rep} % de {X} supone pagar {int(BAC[X])} × {rep / 100:.2f} = {certificado:.1f} k€ por esa actividad, "
                     f"de los que {int(BAC[X])} × ({rep} − {real}) % = {sobrepago:.1f} k€ "
                     f"corresponderían a trabajo no ejecutado según el jefe de equipo. El criterio defendible frente a Dirección "
                     f"financiera: certificar lo verificado ({real} %, {float(BAC[X]) * real / 100:.1f} k€) y dejar documentada la "
                     "discrepancia con una medición conjunta en obra esta misma semana; si la medición da la razón al contratista se "
                     "certifica el resto sin demora (y sin intereses, porque la retención está motivada). Para que no vuelva a pasar: "
                     "medir el avance por unidades de obra o hitos verificables (no por horas gastadas ni por «sensación»), aplicar la "
                     "regla 0/100 o 50/50 a las actividades cortas, y que la medición la haga o la valide la propiedad."))
    celdas += apartado(5)
    celdas.append(md("Decisión: informar al comité con las dos versiones, el semáforo de la peor y la fecha de la medición conjunta; "
                     f"certificar el {real} % de {X}; exigir al contratista un plan de recuperación de {X}"
                     + (f" (y revisar la planificación de {_y(sucesoras)}, que depende{'n' if len(sucesoras) > 1 else ''} de ella)" if sucesoras else "")
                     + f"; avisar al cliente por escrito de un riesgo de retraso de al menos {retraso_X:.1f} semanas. Mensajes:\n\n"
                     f"- *Contratista:* «Certificaremos esta semana el {real} % de {X}, que es lo verificado en obra; el resto en cuanto "
                     "lo comprobemos juntos el jueves. Necesito el viernes un plan de recuperación con hitos semanales medibles.»\n"
                     "- *Jefe de equipo:* «Gracias por avisar. Prepara la medición conjunta del jueves con las unidades de obra que "
                     f"sustentan tu {real} %; a partir de ahora validas tú cada porcentaje antes de la certificación.»\n"
                     f"- *Comité:* «Plazo: {sem_rojo(e['SPI_real'])} (SPI {e['SPI_real']:.2f} con el avance verificado; el "
                     f"informe del contratista daría verde). Coste: {sem_rojo(e['CPI_real'])} (EAC {EAC_real:.0f} k€ frente a "
                     f"{BAC_total:.0f}). Fin previsto: semana {T + retraso_X:.1f} como pronto si no se corrige {X} ({T} + {retraso_X:.1f}). El jueves hay medición conjunta.»\n"
                     f"- *Dirección financiera:* «Criterio: se certifica el avance verificado por la propiedad. Esta certificación retiene "
                     f"{sobrepago:.1f} k€ con acta de discrepancia; no hay base para intereses de demora.»\n"
                     f"- *Cliente:* «Les informamos de un riesgo de retraso de al menos {retraso_X:.1f} semanas en el hito, detectado en el "
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
    var2 = ((b2 - a) / 6) ** 2
    with _silencio():
        sigma2, _ = info["proyecto"].standard_deviation(m_media2, var2.to_dict())
    p_pert2, q90_pert2 = float(norm.cdf(T, T_prod, sigma2)), float(norm.ppf(0.9, T_prod, sigma2))
    p_mc2, q90_mc2 = float((tau2 <= T).mean()), float(np.quantile(tau2, 0.9))
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
                       "display(detalle_estimaciones(datos_proyecto, 'optimista', 'mas_probable', 'pesimista'))\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']]"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias y el camino crítico", label="fig-pert")
    celdas.append(code("T_media = float(proyecto.duration(datos_proyecto['duration']))\n"
                       "sigma, varianzas_caminos = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['varianza'].to_dict())\n"
                       "p_pert = norm.cdf(plazo_comercial, T_media, sigma)\n"
                       "plazo90_pert = norm.ppf(0.9, T_media, sigma)\n"
                       "_ruta = ordena_ruta(list(proyecto.critical_path(datos_proyecto['duration']).values())[0], calendario)\n"
                       "Markdown(f'Duración esperada: {suma_caminos_criticos(proyecto, datos_proyecto[\"duration\"])}. Desviación típica del camino '\n"
                       "         f'crítico (raíz de la suma de las varianzas de sus actividades): {explica_sigma_ruta(_ruta, datos_proyecto[\"varianza\"])} semanas.\\n\\n'\n"
                       "         f'Con T ~ N({T_media:g}; {sigma:.3f}):\\n\\n- {explica_probabilidad(T_media, sigma, plazo_comercial)}: la media se cumple '\n"
                       "         f'la mitad de las veces.\\n- Plazo al 90 %: {explica_percentil(T_media, sigma, 0.9, \"semanas\")}.\\n\\n'\n"
                       "         f'**Respuesta:** duración esperada {T_media:g} semanas, σ = {sigma:.2f} semanas, P(T ≤ {plazo_comercial}) = {p_pert:.2f} '\n"
                       "         f'y plazo al 90 % de {plazo90_pert:.1f} semanas.')"))
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
    celdas.append(md(f"En Monte Carlo cada réplica sortea la duración de cada actividad con su distribución PERT y calcula la duración "
                     f"del proyecto (la ruta más larga de esa réplica); P(T ≤ {T}) es la proporción de las {N_MC} réplicas que terminan "
                     f"en {T} semanas o menos ({int((tau <= T).sum())} / {N_MC} = {e['p_mc']:.2f}), el plazo al 90 % es el percentil 90 de "
                     "las duraciones simuladas y el índice de criticidad de una actividad es el porcentaje de réplicas en las que está "
                     "en la ruta más larga.\n\n"
                     f"La media PERT ({T}) subestima la duración y la normal del camino crítico subestima la cola derecha: en las "
                     f"réplicas hay otras rutas (la segunda dura {info['segunda']:.0f} en media) que en bastantes escenarios se "
                     f"convierten en la más larga. Es el *sesgo de fusión* (merge bias): el máximo de varias variables aleatorias tiene "
                     f"media mayor que el mayor de sus medias. Por eso P(T ≤ {T}) baja del 0.50 al {e['p_mc']:.2f} y el plazo al 90 % "
                     f"sube de {q90_pert:.1f} a {q90_mc:.1f}. La actividad no crítica con más criticidad es **{e['vigilar']}** "
                     f"({e['crit'][e['vigilar']]:.0%} de las réplicas en la ruta más larga, holgura {int(cal.loc[e['vigilar'], 'H_total'])}): "
                     "el análisis determinista la da por holgada y conviene vigilarla como si fuera crítica.\n\n"
                     f"**Respuesta:** por Monte Carlo, P(T ≤ {T}) = {e['p_mc']:.2f} (PERT: 0.50) y plazo al 90 % = {q90_mc:.1f} semanas "
                     f"(PERT: {q90_pert:.1f}); no coinciden por el sesgo de fusión, que el método PERT ignora al mirar sólo el camino "
                     f"crítico. La actividad no crítica que hay que vigilar es {e['vigilar']}."))
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
    celdas.append(md(f"Añadir dos semanas a la estimación pesimista de {X} sube su media en 2/6 = {2 / 6:.2f} semanas "
                     f"(({int(a[X])} + 4·{int(m[X])} + {int(b2[X])})/6 = {(a[X] + 4 * m[X] + b2[X]) / 6:.2f}) y su varianza de "
                     f"{e['var'][X]:.3f} a (({int(b2[X])} − {int(a[X])})/6)² = {((b2[X] - a[X]) / 6) ** 2:.3f}.\n\n"
                     f"**Respuesta:** la duración esperada pasa de {T} a {T_prod:.2f} semanas (+{T_prod - T:.2f}); P(T ≤ {T}) pasa de "
                     f"0.50 a {p_pert2:.2f} con PERT y de {e['p_mc']:.2f} a {p_mc2:.2f} con Monte Carlo; el plazo al 90 % sube de "
                     f"{q90_pert:.2f} a {q90_pert2:.2f} con PERT ({q90_pert2 - q90_pert:+.2f}) y de {q90_mc:.2f} a {q90_mc2:.2f} con Monte "
                     f"Carlo ({q90_mc2 - q90_mc:+.2f} semanas). "
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
                       "_f = plazos.loc[D_opt]\n"
                       "_cumple = _f['P(cumplir)'] >= 0.9\n"
                       "Markdown(f'Para un plazo ofertado D, con las {len(tau)} réplicas τ de la duración (descuento comercial, penalización esperada y bonificación esperada): coste esperado = {descuento} × máx(0, D − '\n"
                       "         f'{plazo_competidor}) + {penalizacion} × media(máx(0, τ − D)) − {bonificacion} × media(máx(0, D − τ)). Para D = {D_opt}:\\n\\n'\n"
                       "         f'- descuento = {descuento} × máx(0, {D_opt} − {plazo_competidor}) = {_f[\"descuento comercial\"]:.2f} k€\\n'\n"
                       "         f'- penalización esperada = {penalizacion} × {_f[\"penalización esperada\"] / penalizacion:.3f} = {_f[\"penalización esperada\"]:.2f} k€\\n'\n"
                       "         f'- bonificación esperada = {bonificacion} × {_f[\"bonificación esperada\"] / bonificacion:.3f} = {_f[\"bonificación esperada\"]:.2f} k€\\n'\n"
                       "         f'- coste esperado = {_f[\"descuento comercial\"]:.2f} + {_f[\"penalización esperada\"]:.2f} − {_f[\"bonificación esperada\"]:.2f} = **{_f[\"coste esperado total\"]:.2f} k€**\\n\\n'\n"
                       "         f'**Respuesta:** el plazo que minimiza el coste esperado es **{D_opt} semanas** (P(cumplir) = {_f[\"P(cumplir)\"]:.2f}), que '\n"
                       "         + ('sí cumple la condición de asesoría jurídica (≥ 0.90).' if _cumple else\n"
                       "            f'no cumple la condición de asesoría jurídica (0.90). El primer plazo que la cumple es **{D_legal} semanas** '\n"
                       "            f'(percentil 90 de las réplicas, {plazo90_mc:.2f}, redondeado hacia arriba; P(cumplir) = {plazos.loc[D_legal, \"P(cumplir)\"]:.2f}), '\n"
                       "            f'con coste esperado {plazos.loc[D_legal, \"coste esperado total\"]:.2f} k€: cumplirla cuesta '\n"
                       "            f'{plazos.loc[D_legal, \"coste esperado total\"]:.2f} − {_f[\"coste esperado total\"]:.2f} = '\n"
                       "            f'**{plazos.loc[D_legal, \"coste esperado total\"] - _f[\"coste esperado total\"]:.2f} k€** esperados.'))"))
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
    from scipy.optimize import brentq
    u = brentq(lambda v: van_con(p, precio=v), p["precio"] * 0.2, p["precio"] * 2) / p["precio"]   # precio umbral / precio
    f_min = (u - 1 + 2 * x) / x                    # fracción contratada a −x para que el peor caso tenga VAN ≥ 0
    x40 = 1 - (u - 0.6 * (1 - 2 * x)) / 0.4        # descuento máximo con el 40 % contratado
    x100 = 1 - u                                   # descuento máximo con todo el volumen contratado
    if f_min <= 1:
        cobertura_txt = (f"Para lograrlo habría que contratar al menos el {f_min:.0%} del volumen con el mismo descuento del {x:.0%} "
                         f"(despejando f en f·(1 − {x:g}) + (1 − f)·(1 − {2 * x:g}) ≥ {u:.3f})"
                         + (f", o el 40 % con un descuento de como mucho el {x40:.1%}" if x40 > 0 else "") + ".")
        cobertura_corta = f"al menos el {f_min:.0%} del volumen a −{x:.0%}" + (f", o el 40 % a −{x40:.1%} como mucho" if x40 > 0 else "")
        msg_cli = (f"El 40 % a −{x:.0%} cubre poco riesgo para lo que cuesta; proponemos el {min(1.0, np.ceil(f_min * 20) / 20):.0%} del volumen con el mismo descuento"
                   + (f", o el 40 % con un descuento del {np.floor(x40 * 200) / 200:.1%}" if x40 > 0 else "") + ".")
    else:
        cobertura_txt = (f"Pero ampliar el porcentaje contratado no basta, ni siquiera al 100 %: con un descuento del {x:.0%} el precio "
                         f"del contrato ya queda por debajo del umbral (−{x100:.1%}). Con todo el volumen haría falta un descuento de "
                         f"como mucho el {x100:.1%}, que deja el VAN en cero también si el mercado aguanta"
                         + (f", y con el 40 % un descuento de como mucho el {x40:.1%}." if x40 > 0 else
                            f"; y con sólo el 40 % el precio fijo tendría que ser un {-x40:.0%} mayor que el previsto. "
                            "Este contrato no sirve para cubrir el riesgo de precio: el problema es que el proyecto sólo aguanta "
                            f"una caída del precio del {x100:.1%}."))
        cobertura_corta = (f"el contrato ofrecido no cubre el riesgo: a −{x:.0%} el precio fijo queda por debajo del umbral (−{x100:.1%}), "
                           "así que hay que buscar otra cobertura o rebajar el riesgo de precio")
        msg_cli = (f"Pero a −{x:.0%} el precio fijo queda por debajo del que el proyecto necesita para no perder dinero "
                   f"(−{x100:.1%}): así no nos sirve como cobertura. Sólo tendría sentido con un descuento claramente menor.")
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
                       "td = tabla_descuento(flujos, k)\n"
                       "display(td.round(2))\n"
                       "def explica_payback(col):\n"
                       "    t = int((td[col] >= 0).idxmax()); prev = float(td.loc[t - 1, col]); f = float(td.loc[t, col] - td.loc[t - 1, col])\n"
                       "    return f'{t - 1} + {-prev:.1f} / {f:.1f} = {t - 1 + (-prev) / f:.2f} años'\n"
                       "display(Markdown('VAN = Σ FCLₜ·(1 + k)⁻ᵗ = ' + ' + '.join(f'({v:.1f})' for v in td['valor actual']) + f' = **{VAN:.1f} k€**. '\n"
                       "                 f'La TIR es la tasa que anula el VAN: con k = {TIR:.4f}, VAN = {van(flujos, TIR):.2f} ≈ 0. '\n"
                       "                 'Recuperación (año en que el acumulado se hace positivo, interpolando dentro del año: años completos + lo que falta / flujo del año): '\n"
                       "                 f'simple {explica_payback(\"acumulado\")}; descontada {explica_payback(\"acumulado descontado\")}.'))\n"
                       "Markdown(f'''VAN = {VAN:.1f} k€ al {k:.0%} ({VAN / I0:.1%} de la inversión; la regla pide el 10 %: {\"pasa\" if VAN >= 0.1 * I0 else \"no pasa\"}); "
                       "TIR = {TIR:.2%}; recuperación simple {payback(flujos):.2f} años, descontada {payback(flujos, k):.2f}.''')"))
    celdas.append(md(f"**Respuesta:** el caso base es {'claramente' if VAN >= 0.15 * I0 else 'moderadamente'} rentable: VAN {VAN:.1f} k€, "
                     f"que es {VAN:.1f} / {I0:.0f} = {umbral_ce:.1%} de la inversión, así que {'pasa' if umbral_ce >= 0.10 else 'no pasa'} "
                     f"la primera parte de la regla (VAN ≥ 10 % de la inversión); TIR {TIR:.1%} frente al {k:.0%} exigido. El consejero "
                     "delegado pide además que no haya una probabilidad apreciable de VAN negativo, y eso es lo que los mensajes ponen en duda."))
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
                       "Markdown(f'**Respuesta:** la variable más sensible es **{mas_sensible}**: una variación del {umbrales.loc[mas_sensible, \"variación admisible\"]:+.1%} anula el VAN.')"))
    lineas = "\n".join(f"- {nombre}: VAN {v:.0f} k€ ({v / I0:+.0%} de la inversión)." for nombre, v in esc.items())
    celdas.append(md("Cada mensaje por separado:\n\n" + lineas + "\n\n"
                     f"El mensaje del director comercial tiene dos caras: su volumen convierte el proyecto en muy bueno, pero su "
                     f"escenario de precio lo hunde; el coste variable de producción {'lo deja en pérdidas' if esc[f'coste variable +{z:.0%}'] < 0 else 'se come buena parte del VAN'}. "
                     f"El proyecto es más sensible al precio que al volumen: un 1 % de precio cambia los ingresos de cada unidad en "
                     f"{p['precio'] / 100:.3g} y todo eso va al margen, mientras que un 1 % de volumen sólo añade el margen de contribución de "
                     f"esas unidades (precio − coste variable = {p['precio']:g} − {p['coste_variable']:g} = {p['precio'] - p['coste_variable']:g} por "
                     "unidad); el coste variable actúa como un precio negativo. Por eso la variación admisible del precio (umbrales) es "
                     "la más pequeña."))
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
                       "display(Markdown(f'Precio medio con el contrato = 0.4·(1 − {x:g})·p + 0.6·p_mercado: {0.4 * (1 - x) + 0.6:.3f}·p si el mercado aguanta y '\n"
                       "                 f'{0.4 * (1 - x) + 0.6 * (1 - 2 * x):.3f}·p si cae un {2 * x:.0%}. VAN esperado = (1 − π)·VAN(estable) + π·VAN(cae):\\n\\n'\n"
                       "                 + '\\n'.join(f'- {nombre}: {1 - pi:g} × {fila[\"mercado estable\"]:.1f} + {pi:g} × ({fila[\"cae el precio\"]:.1f}) = **{fila[\"VAN esperado\"]:.1f} k€**'\n"
                       "                             for nombre, fila in contrato.iterrows())))\n"
                       "Markdown(f'''**Respuesta:** con probabilidad {pi:.0%} de caída del precio, VAN esperado **{contrato.loc[\"sin contrato\", \"VAN esperado\"]:.1f}** k€ sin contrato y "
                       "**{contrato.loc[\"con contrato\", \"VAN esperado\"]:.1f}** con él. El contrato cuesta {van_sin[\"mercado estable\"] - van_con[\"mercado estable\"]:.1f} k€ de VAN "
                       "si el mercado aguanta y ahorra {van_con[\"cae el precio\"] - van_sin[\"cae el precio\"]:.1f} si cae.''')"))
    mejor = "con contrato" if e_con > e_sin else "sin contrato"
    v_bajo = esc[f"precio −{2 * x:.0%}"]
    prima, ahorro = VAN - v_contrato_base, v_contrato_bajo - v_bajo
    celdas.append(md(f"El contrato es un seguro: se paga una prima ({prima:.1f} k€ de VAN si el mercado aguanta) a cambio de recortar "
                     f"la pérdida en el escenario malo ({ahorro:.1f} k€ menos de pérdida). En valor esperado, la prima cuesta "
                     f"{1 - pi:g} × {prima:.1f} = {(1 - pi) * prima:.1f} k€ y el ahorro vale {pi:g} × {ahorro:.1f} = {pi * ahorro:.1f} k€, "
                     f"así que sale mejor **{mejor}** ({max(e_con, e_sin):.1f} frente a {min(e_con, e_sin):.1f} k€)"
                     + (f": como en casi todo seguro, la prima esperada supera a la pérdida esperada que evita (la caída tiene "
                        f"probabilidad {pi:.0%} < 50 % y prima y ahorro son de tamaño parecido)" if mejor == "sin contrato" else
                        ": el ahorro esperado en el escenario malo supera a la prima esperada")
                     + f". Pero la regla del consejero mira también al peor caso: "
                     + ("con el contrato el peor caso sigue siendo negativo" if v_contrato_bajo < 0 else "con el contrato el peor caso deja de ser negativo")
                     + f" ({v_contrato_bajo:.0f} k€ frente a {v_bajo:.0f} sin él). "
                     + (f"El 40 % que ofrece el cliente no basta para cubrir el riesgo de precio. El VAN es nulo con un precio un {x100:.1%} "
                        f"menor que el previsto (precio umbral del apartado 2); el peor caso sólo deja de ser negativo si el precio medio "
                        f"no baja de ese umbral. {cobertura_txt}"
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
                       "display(Markdown(f'Cuota constante C = P·i / (1 − (1 + i)⁻ⁿ) = {prestamo:.1f} × {i:g} / (1 − {1 + i:g}^−{n_p}) = **{cuadro.cuota.iloc[0]:.2f} k€**. '\n"
                       "                 f'Flujo del accionista: en el año 0, FCL + préstamo = {flujos[0]:.1f} + {prestamo:.1f} = {fa[0]:.1f}; en el año 1, '\n"
                       "                 f'FCL − cuota + {t_imp:g} × intereses = {flujos[1]:.1f} − {cuadro.cuota.iloc[0]:.2f} + {t_imp:g} × {cuadro.intereses.iloc[0]:.2f} = {fa[1]:.1f} '\n"
                       "                 '(los intereses se deducen del impuesto: es el ahorro fiscal); igual en los demás años del préstamo.'))\n"
                       "Markdown(f'''TIR del proyecto {TIR:.2%}; TIR del accionista con el préstamo {TIR_acc:.2%}. Coste de la deuda después de impuestos "
                       "{i * (1 - t_imp):.2%}, {\"menor\" if i * (1 - t_imp) < TIR else \"mayor\"} que la TIR del proyecto: el apalancamiento {\"amplifica la rentabilidad del accionista\" if i * (1 - t_imp) < TIR else \"la reduce\"}.''')"))
    celdas.append(md((f"La TIR del accionista ({TIR_acc:.1%}) es mayor que la del proyecto ({TIR:.1%}) porque la deuda cuesta "
                      f"{i:.0%} × (1 − {t_imp:g}) = {i * (1 - t_imp):.2%} neto, menos de lo que rinde el proyecto: el accionista pone sólo el "
                      f"{1 - alfa:.0%} de la inversión y se queda con la diferencia. "
                      if TIR_acc > TIR else
                      f"La TIR del accionista ({TIR_acc:.1%}) no supera a la del proyecto ({TIR:.1%}): la deuda cuesta {i * (1 - t_imp):.2%} "
                      "neto, más de lo que rinde el proyecto. ")
                     + "Pero el apalancamiento amplifica en los dos sentidos: en el escenario de caída del "
                     "precio las cuotas se pagan igual y la pérdida del accionista es proporcionalmente mayor. Dirección financiera "
                     "tiene razón: la decisión de invertir se toma sobre el flujo de caja libre (sin deuda) y la de financiar, después; "
                     "una TIR del accionista alta no convierte en bueno un proyecto malo.\n\n"
                     f"**Respuesta:** TIR del proyecto {TIR:.2%} y del accionista {TIR_acc:.2%}; sí, Dirección financiera tiene razón: el "
                     "proyecto se juzga sin deuda."))
    celdas += apartado(5)
    pasa_esperado = max(e_sin, e_con) >= 0.10 * I0
    if not pasa_esperado:
        recomendacion = ("no aprobar tal como está y volver al comité con tres cosas: la probabilidad de caída del precio documentada "
                         "(no una opinión), una cobertura que de verdad proteja el peor caso (contraoferta al cliente clave: "
                         + cobertura_corta + ") y el coste variable revisado con producción")
        corta = "no aprobar tal como está"
    elif v_contrato_bajo < 0:
        recomendacion = ("aprobar condicionado a cerrar una cobertura de precio mayor que la ofrecida (contraoferta al cliente clave: "
                         + cobertura_corta + ") y a revisar el coste variable con producción")
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
                        (f"- *Cliente clave:* «Valoramos el contrato. " + msg_cli + "»\n"))
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


# ---------------------------------------------------------------------------
# Caso integrador — Plan de contingencia de una planta: recursos limitados, duraciones inciertas,
# aceleración con ayuda del contratista, penalización del cliente y VAN
# ---------------------------------------------------------------------------
def _evaluar_opcion(pg, df, dist_base, L_op, red, pen, c, plazo, F, N, k, L_ref, n, seed):
    """VAN neto (tras penalización y operarios extra) y probabilidad de cumplir de una opción."""
    dist = {a: ("triangular", max(1.0, d[1] - red[a]), max(1.0, d[2] - red[a]), max(1.0, d[3] - red[a]))
            for a, d in dist_base.items()}
    bac = df["BAC"] + df["ucr"].replace(np.inf, 0.0) * red
    mc = pg.monte_carlo_van_recursos(dist, df[["operarios"]], ["operarios"], [L_op], bac, F, N, k,
                                     n_samples=n, seed=seed)
    tau = mc.tau_recursos_samples
    neto = mc.van_recursos_samples - pen * np.maximum(tau - plazo, 0.0) - c * (L_op - L_ref) * tau
    return {"VAN medio": float(neto.mean()), "P(VAN<0)": float((neto < 0).mean()),
            "P(τ≤plazo)": float((tau <= plazo).mean()), "τ medio": float(tau.mean()),
            "coste reducción": float((df["ucr"].replace(np.inf, 0.0) * red).sum())}


def _plan_opciones(pg, df, dist_base, L, pen, c, plazo, F, N, k, n, seed):
    d = df[["predecessor", "duration", "operarios"]]
    tope, ucr = df["cap_duration"], df["ucr"]
    ceros = pd.Series(0.0, index=df.index)
    red_c = pg.crashing_recursos(d, "duration", ["operarios"], [L], tope, ucr, reduccion=2).reducciones
    red_d = pg.crashing_recursos(d, "duration", ["operarios"], [L + 1], tope, ucr, reduccion=2).reducciones
    red_c, red_d = pd.Series(red_c, dtype=float).reindex(df.index).fillna(0.0), pd.Series(red_d, dtype=float).reindex(df.index).fillna(0.0)
    ops = {"A: no hacer nada": (L, ceros), "B: un operario más": (L + 1, ceros),
           "C: acelerar dos meses": (L, red_c), "D: operario más y acelerar": (L + 1, red_d)}
    return pd.DataFrame({n_: _evaluar_opcion(pg, df, dist_base, Lo, r, pen, c, plazo, F, N, k, L, n, seed)
                         for n_, (Lo, r) in ops.items()}).T


def _contingencia_planta(rng):
    estado = {}

    def acepta(i):
        if not (0 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 20):
            return False
        nombres = list(i["D"].index)
        a, m, b = P.tres_estimaciones_triangular(rng, i["D"])
        if ((b - a) < 2).sum() > len(nombres) // 2:
            return False
        op = P.recursos(rng, nombres, hi=3).rename("operarios")
        L = int(op.max())
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        if int(np.isfinite(ucr).sum()) < 4:
            return False
        BAC = pd.Series(rng.integers(2, 9, size=len(nombres)) * 25, index=nombres, name="BAC")
        df = P.tabla_base(i, duration=i["D"], a=a, m=m, b=b, operarios=op, BAC=BAC, cap_duration=tope, ucr=ucr)
        pg = i["proyecto"]
        T0 = int(i["T"])
        try:
            r0 = pg.copy().asignar_multi(df[["predecessor", "duration", "operarios"]].copy(), "duration", ["operarios"], [L],
                                         dibujar=False)
            r1 = pg.copy().asignar_multi(df[["predecessor", "duration", "operarios"]].copy(), "duration", ["operarios"], [L + 1],
                                         dibujar=False)
        except Exception:
            return False
        if not (1 <= r0.duracion - T0 <= 3 and r1.duracion < r0.duracion):
            return False
        k = float(rng.choice([0.08, 0.10, 0.12]))
        N = int(rng.choice([36, 48]))
        dist_base = {x: ("triangular", float(a[x]), float(m[x]), float(b[x])) for x in nombres}
        fijas = {x: ("triangular", float(d), float(d), float(d)) for x, d in i["D"].items()}
        v0 = pg.monte_carlo_van(fijas, BAC, 0.0, N, k, n_samples=1, seed=0).van_samples[0]
        v1 = pg.monte_carlo_van(fijas, BAC, 1.0, N, k, n_samples=1, seed=0).van_samples[0]
        F = float(np.ceil((-v0 / (v1 - v0)) * float(rng.choice([1.05, 1.1, 1.18])) / 5) * 5)
        plazo = T0 + 1
        pen = float(rng.choice([6, 10, 15]))
        c = float(rng.choice([2.0, 3.0, 4.0]))
        try:
            t = _plan_opciones(pg, df, dist_base, L, pen, c, plazo, F, N, k, 300, 5)
        except Exception:
            return False
        best = t["VAN medio"].idxmax()
        if best.startswith(("A", "D")):
            return False
        if not (t.loc[best, "P(τ≤plazo)"] - t.iloc[0]["P(τ≤plazo)"] >= 0.15 and t.iloc[0]["P(τ≤plazo)"] <= 0.45):
            return False
        if t["VAN medio"].sort_values().diff().dropna().min() < 2.0:
            return False
        if not (0.05 <= t["P(VAN<0)"].max() <= 0.8):
            return False
        estado.update(df=df, L=L, k=k, N=N, F=F, plazo=plazo, pen=pen, c=c, T0=T0, Tr=int(r0.duracion), t=t)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), max_intentos=3000)
    df, L, k, N, F, plazo, pen, c = (estado[x] for x in ("df", "L", "k", "N", "F", "plazo", "pen", "c"))
    T0, Tr = estado["T0"], estado["Tr"]
    agentes = ["cliente", "director financiero", "jefe de obra", "contratista"]
    celdas = [md("# Enunciado"),
              md(f"Usted dirige la construcción de una planta cuyo cuadro de prelaciones, duraciones (meses; *a*, *m* y *b* son las "
                 "estimaciones mínima, más probable y máxima de una distribución triangular y *duration* la duración esperada), "
                 f"operarios por actividad, presupuesto (BAC, k€), duración tope y coste unitario de reducción (CUR, k€ por mes) se "
                 f"adjuntan. La cuadrilla fija es de **{L} operarios**. Cuando la planta se entrega, genera un flujo neto de "
                 f"{F:.0f} k€ al mes durante {N} meses; la empresa descuenta al {k:.0%} anual. El viernes recibe estos mensajes."),
              code('#| tbl-cap: "Datos del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Cliente",
                       f"El contrato fija la entrega en el mes {plazo} con una penalización de **{pen:.0f} k€ por mes de retraso**. "
                       "Les pedimos que nos confirmen la probabilidad real de cumplirlo, no la fecha del plan."),
              _mensaje("Director financiero",
                       "No autorizaré ninguna medida cuyo VAN esperado, neto de penalizaciones y de lo que cueste, no supere al de no "
                       "hacer nada. Y quiero saber la probabilidad de que el VAN sea negativo en cada opción."),
              _mensaje("Jefe de obra",
                       f"Podemos subcontratar **un operario más** durante todo el proyecto por {c:.0f} k€ al mes. Con la cuadrilla "
                       "actual, por mucho que acorten las actividades, a veces se quedan esperando por falta de gente."),
              _mensaje("Contratista",
                       "Podemos acelerar actividades con horas extra y refuerzos a los CUR indicados en la tabla (hasta la duración tope). "
                       "Díganos cuántos meses necesita recortar y le decimos qué actividades."),
              _cierre(agentes)]
    celdas.append(preguntas([
        "Calcule la duración con las duraciones esperadas sin límite de recursos y con el límite de operarios. ¿Cuál es la cadena "
        "crítica con recursos y qué actividades esperan?",
        f"Simule 3 000 ejecuciones con duraciones triangulares y con el límite de operarios (replanificando en cada réplica) y "
        f"estime la probabilidad de entregar en el mes {plazo}, el VAN esperado neto de penalización y P(VAN < 0) si no se hace nada.",
        "Evalúe cuatro opciones con los mismos números aleatorios: (A) no hacer nada, (B) contratar un operario más, (C) acelerar dos "
        "meses con el contratista (reducción con recursos limitados) y (D) las dos cosas a la vez. Tabla de VAN esperado neto, "
        "P(VAN < 0), probabilidad de cumplir el plazo, duración media y coste de la aceleración.",
        "¿Qué opción recomienda al director financiero? ¿Y al cliente? ¿Se suman los efectos de B y C en D? Explique por qué.",
        "Redacte en tres o cuatro líneas la respuesta a cada implicado.",
        "¿Qué ha cambiado respecto a un análisis determinista con el camino crítico? Señale al menos tres decisiones que habría "
        "tomado mal sin recursos, sin incertidumbre o sin VAN."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(UTIL))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"L, plazo, penalizacion, coste_operario, F, N, k = {L}, {plazo}, {pen}, {c}, {F}, {N}, {k}\n"
                       "base = datos_proyecto[['predecessor', 'duration', 'operarios']]\n"
                       "tope, ucr = datos_proyecto['cap_duration'], datos_proyecto['ucr']\n"
                       "dist_base = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones esperadas y el camino crítico (sin límite de operarios)", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario sin límite de recursos (meses)"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Carga de operarios con los inicios más tempranos frente al límite"\n#| label: fig-cargasInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(base.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[L])"))
    celdas.append(md("El algoritmo de asignación recorre los meses: cada actividad disponible (predecesoras terminadas) "
                     "empieza, por orden de prioridad (menor inicio más tardío), si caben sus operarios durante toda su "
                     "duración; si no, espera. Traza mes a mes y Gantt resultante (filas 💤: esperas):"))
    celdas.append(code("res0 = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], [L], report=True, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt con el límite de operarios y las duraciones esperadas"\n#| label: fig-ganttAsignado\n'
                       "pinta_tikz(res0.dibujos['operarios'])"))
    celdas.append(code('#| tbl-cap: "Inicio de cada actividad sin y con límite"\n#| label: tbl-inicios\n'
                       "pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res0.inicios, 'fin con límite': res0.fines,\n"
                       "              'espera (meses)': res0.esperas}).fillna(0).astype(int).T"))
    celdas.append(code("duracion_libre = proyecto.duration(datos_proyecto['duration'])\n"
                       "cadena = proyecto.copy().cadena_critica_recursos(base.copy(), 'duration', ['operarios'], [L])\n"
                       "display(cadena['arcos'].rename(columns={'motivo': 'motivo (precedencia o recurso)'}))\n"
                       "_esp = res0.esperas[res0.esperas > 0]\n"
                       "_cadena = sorted(cadena['actividades'], key=lambda a: (res0.inicios[a], a))\n"
                       "Markdown(f'**Respuesta:** sin límite de operarios el proyecto dura {suma_caminos_criticos(proyecto, datos_proyecto[\"duration\"], \"meses\")}; '\n"
                       "         f'con {L} operarios dura **{res0.duracion} meses** ({res0.duracion - duracion_libre:g} más). La cadena crítica con recursos '\n"
                       "         f'(las actividades que fijan esa duración, encadenadas por precedencias o porque una espera a que otra libere '\n"
                       "         f'operarios: arcos «recurso» de la tabla) es {\"–\".join(_cadena)}. Esperan por falta de operarios: '\n"
                       "         + lista_y([f'{a} ({plural(int(v), \"mes\", \"meses\")})' for a, v in _esp.items()]) + '.')"))
    celdas += apartado(2)
    celdas.append(code(EVALUAR_OPCION))
    celdas.append(code("ceros = pd.Series(0.0, index=datos_proyecto.index)\n"
                       "A = evaluar(L, ceros)\n"
                       "Markdown(f'En cada una de las 3000 réplicas se sortean las duraciones (triangulares, redondeadas a meses enteros), se '\n"
                       "         f'replanifica con {L} operarios y se obtiene la duración τ. El VAN de la réplica descuenta, a la tasa mensual '\n"
                       "         f'equivalente (1 + k_m)¹² = 1 + {k:g}, los costes (el BAC de cada actividad repartido uniformemente en su duración) y '\n"
                       "         f'el flujo de {F:g} k€/mes durante {N} meses desde el mes τ; al VAN se le resta la penalización, {penalizacion:g} × máx(0, τ − {plazo}) '\n"
                       "         f'(sin descontar, como aproximación). Contando las réplicas:\\n\\n'\n"
                       "         f'**Respuesta:** si no se hace nada, P(τ ≤ {plazo}) = **{A[\"P(τ≤plazo)\"]:.3f}** (duración media {A[\"τ medio\"]:.2f} meses), '\n"
                       "         f'VAN esperado neto de penalización = **{A[\"VAN medio\"]:.1f} k€** y P(VAN < 0) = **{A[\"P(VAN<0)\"]:.3f}**.')"))
    celdas += apartado(3)
    celdas.append(md("Las reducciones de C y D se calculan con `crashing_recursos`: en cada paso se programa con el límite, "
                     "se halla la cadena crítica con recursos, se prueban como candidatos sus actividades reducibles (sueltas, "
                     "por parejas y el corte de coste mínimo) replanificando con cada uno, y se elige el más barato que de "
                     "verdad acorta el proyecto replanificado. Paso a paso, con el límite actual (C) y con un operario más (D):"))
    celdas.append(code('#| tbl-cap: "Crashing con recursos para la opción C (límite actual)"\n#| label: tbl-crashingC\n'
                       "cr_c = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], [L], tope, ucr, reduccion=2)\n"
                       "cr_c.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]"))
    celdas.append(code('#| tbl-cap: "Crashing con recursos para la opción D (un operario más)"\n#| label: tbl-crashingD\n'
                       "cr_d = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], [L + 1], tope, ucr, reduccion=2)\n"
                       "cr_d.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']]"))
    celdas.append(code("red_c = pd.Series(cr_c.reducciones, dtype=float).reindex(datos_proyecto.index).fillna(0.0)\n"
                       "red_d = pd.Series(cr_d.reducciones, dtype=float).reindex(datos_proyecto.index).fillna(0.0)\n"
                       "opciones = {'A: no hacer nada': (L, ceros), 'B: un operario más': (L + 1, ceros),\n"
                       "            'C: acelerar dos meses': (L, red_c), 'D: operario más y acelerar': (L + 1, red_d)}\n"
                       "tabla = pd.DataFrame({n: evaluar(Lo, r) for n, (Lo, r) in opciones.items()}).T\n"
                       "tabla.rename(columns={'VAN medio': 'VAN esperado neto (k€)', 'P(VAN<0)': 'P(VAN < 0)', 'P(τ≤plazo)': f'P(τ ≤ {plazo})',\n"
                       "                      'τ medio': 'duración media (meses)', 'coste reducción': 'coste de la aceleración (k€)'}).astype(float).round(3)"))
    celdas.append(code("display(pd.DataFrame({'C: reducciones': red_c[red_c > 0], 'D: reducciones': red_d[red_d > 0]}).fillna(0).astype(int).T)\n"
                       "def _coste(red):\n"
                       "    r = red[red > 0]\n"
                       "    return ' + '.join(f'{ucr[a]:g}·{int(v)}' for a, v in r.items()) + f' = {float((ucr[r.index] * r).sum()):g} k€'\n"
                       "def _difiere(cr):\n"
                       "    f = cr.pasos[cr.pasos['ackoff_sin_recursos'].astype(str) != cr.pasos['actividades'].astype(str)]\n"
                       "    return '; '.join(f'en el paso {i} Ackoff sin recursos habría acortado {r.ackoff_sin_recursos}, pero con el límite de operarios '\n"
                       "                     f'eso no acorta el proyecto replanificado y se acorta {r.actividades}' for i, r in f.iterrows())\n"
                       "_dc, _dd = _difiere(cr_c), _difiere(cr_d)\n"
                       "Markdown(f'Coste de la aceleración (Σ CUR × meses recortados): C, {_coste(red_c)}; D, {_coste(red_d)}. '\n"
                       "         + (f'Opción C: {_dc}. ' if _dc else 'Opción C: cada paso acorta lo mismo que Ackoff sin recursos. ')\n"
                       "         + (f'Opción D: {_dd}. ' if _dd else 'Opción D: cada paso acorta lo mismo que Ackoff sin recursos. ')\n"
                       "         + 'El coste de la aceleración se suma al BAC de las actividades acortadas y el del operario extra '\n"
                       "         f'({coste_operario:g} k€/mes × τ, sin descontar) se resta del VAN, junto con la penalización.\\n\\n'\n"
                       "         '**Respuesta:** la tabla de opciones anterior da, para cada una, el VAN esperado neto, P(VAN < 0), la probabilidad de cumplir '\n"
                       "         f'el plazo, la duración media y el coste de la aceleración; se comparan en el apartado siguiente.')"))
    celdas += apartado(4)
    celdas.append(code("mejor = tabla['VAN medio'].idxmax()\n"
                       "cliente = tabla['P(τ≤plazo)'].idxmax()\n"
                       "A0 = tabla.loc['A: no hacer nada']\n"
                       "dB = tabla.loc['B: un operario más', 'VAN medio'] - A0['VAN medio']\n"
                       "dC = tabla.loc['C: acelerar dos meses', 'VAN medio'] - A0['VAN medio']\n"
                       "dD = tabla.loc['D: operario más y acelerar', 'VAN medio'] - A0['VAN medio']\n"
                       "pB = tabla.loc['B: un operario más', 'P(τ≤plazo)'] - A0['P(τ≤plazo)']\n"
                       "pC = tabla.loc['C: acelerar dos meses', 'P(τ≤plazo)'] - A0['P(τ≤plazo)']\n"
                       "pD = tabla.loc['D: operario más y acelerar', 'P(τ≤plazo)'] - A0['P(τ≤plazo)']\n"
                       "mejores_A = [n for n in tabla.index if tabla.loc[n, 'VAN medio'] > A0['VAN medio']]\n"
                       "if np.isclose(dD, dB + dC, atol=0.5):\n"
                       "    _suma = 'aproximadamente sí: los efectos casi se suman'\n"
                       "elif dD < dB + dC:\n"
                       "    _suma = ('no: juntas valen menos que la suma por separado, porque las dos medidas atacan el mismo problema (la '\n"
                       "             'duración con recursos): una vez que el operario extra elimina esperas, acelerar recorta menos meses útiles, '\n"
                       "             'y el coste de cada medida sí se paga entero')\n"
                       "else:\n"
                       "    _suma = ('no: juntas valen más que la suma por separado, porque con un operario más las actividades aceleradas '\n"
                       "             'ya no se quedan esperando por operarios y el recorte llega de verdad al final del proyecto')\n"
                       "Markdown(f'Cambio de cada opción respecto a A (no hacer nada): VAN esperado neto B {dB:+.1f}, C {dC:+.1f}, D {dD:+.1f} k€; '\n"
                       "         f'probabilidad de cumplir el plazo B {pB:+.3f}, C {pC:+.3f}, D {pD:+.3f}.\\n\\n'\n"
                       "         f'**Respuesta:**\\n\\n'\n"
                       "         f'- Al director financiero: **{mejor}**, la de mayor VAN esperado neto ({tabla.loc[mejor, \"VAN medio\"]:.1f} k€ frente a '\n"
                       "         f'{A0[\"VAN medio\"]:.1f} sin hacer nada; P(VAN < 0) = {tabla.loc[mejor, \"P(VAN<0)\"]:.3f}). Superan a no hacer nada: {lista_y(mejores_A)}.\\n'\n"
                       "         f'- Al cliente, que pregunta por la probabilidad de cumplir: con la opción recomendada es {tabla.loc[mejor, \"P(τ≤plazo)\"]:.2f}; '\n"
                       "         f'la más alta la da {cliente} ({tabla.loc[cliente, \"P(τ≤plazo)\"]:.2f})'\n"
                       "         + (', que es la misma.' if cliente == mejor else f', pero cuesta {tabla.loc[mejor, \"VAN medio\"] - tabla.loc[cliente, \"VAN medio\"]:.1f} k€ de VAN esperado.') + '\\n'\n"
                       "         f'- ¿Se suman B y C en D? B + C = ({dB:+.1f}) + ({dC:+.1f}) = {dB + dC:+.1f} k€ frente a D = {dD:+.1f} k€: {_suma}.')"))
    celdas += apartado(5)
    celdas.append(code("_A, _M = tabla.loc['A: no hacer nada'], tabla.loc[mejor]\n"
                       "_red = {'C: acelerar dos meses': red_c, 'D: operario más y acelerar': red_d}.get(mejor)\n"
                       "_acel = lista_y([f'{a} ({plural(int(v), \"mes\", \"meses\")})' for a, v in _red[_red > 0].items()]) if _red is not None else ''\n"
                       "_operario = mejor.startswith(('B', 'D'))\n"
                       "Markdown(f'- *Cliente:* «Con la cuadrilla actual la probabilidad de entregar en el mes {plazo} es de {_A[\"P(τ≤plazo)\"]:.0%} '\n"
                       "         f'(duración media {_A[\"τ medio\"]:.1f} meses). Con las medidas que vamos a tomar sube al {_M[\"P(τ≤plazo)\"]:.0%}, con una '\n"
                       "         f'duración media de {_M[\"τ medio\"]:.1f} meses. Les informaremos cada mes de cómo evoluciona.»\\n'\n"
                       "         f'- *Director financiero:* «Propongo {mejor[3:]}: VAN esperado neto de penalizaciones y costes {_M[\"VAN medio\"]:.1f} k€, '\n"
                       "         f'frente a {_A[\"VAN medio\"]:.1f} sin hacer nada; probabilidad de VAN negativo {_M[\"P(VAN<0)\"]:.1%} (sin hacer nada, '\n"
                       "         f'{_A[\"P(VAN<0)\"]:.1%}). Adjunto la tabla de las cuatro opciones.»\\n'\n"
                       "         + (f'- *Jefe de obra:* «Contratamos el operario adicional ({coste_operario:g} k€/mes): la simulación lo respalda.»\\n' if _operario else\n"
                       "            f'- *Jefe de obra:* «De momento no contratamos el operario adicional: con {coste_operario:g} k€/mes, en la simulación no compensa '\n"
                       "            f'(opción B: {tabla.loc[\"B: un operario más\", \"VAN medio\"]:.1f} k€ frente a {_M[\"VAN medio\"]:.1f}). Vigilemos las esperas de la cadena crítica.»\\n')\n"
                       "         + (f'- *Contratista:* «Necesitamos acortar {_acel}, con un coste de {_M[\"coste reducción\"]:g} k€ a los CUR de la tabla.»' if _red is not None else\n"
                       "            '- *Contratista:* «Por ahora no aceleramos: con el límite de operarios, recortar duraciones no compensa su coste. Les avisaremos si cambia.»'))"))
    celdas += apartado(6)
    celdas.append(code("_ack = [str(x) for x in cr_c.pasos['ackoff_sin_recursos']]; _real = [str(x) for x in cr_c.pasos['actividades']]\n"
                       "_l = [f'- **Sin recursos** se habría dicho que el proyecto dura {duracion_libre:g} meses y que el plazo del mes {plazo} se '\n"
                       "      f'cumple; con {L} operarios dura {res0.duracion}, y sin hacer nada sólo se cumple el {A[\"P(τ≤plazo)\"]:.0%} de las veces.']\n"
                       "if _ack != _real:\n"
                       "    _l.append(f'- **Sin recursos**, Ackoff habría acortado {lista_y(_ack)}; con el límite de operarios lo que de verdad acorta es '\n"
                       "              f'{lista_y(_real)}: se habría pagado una reducción que no adelanta la entrega.')\n"
                       "else:\n"
                       "    _l.append(f'- Aquí Ackoff sin recursos y la reducción con recursos eligen lo mismo ({lista_y(_real)}), pero sólo la segunda '\n"
                       "              'comprueba que el recorte llega al final con el límite de operarios.')\n"
                       "_l.append(f'- **Sin incertidumbre** se habría prometido una fecha, no una probabilidad: la duración media simulada sin hacer nada '\n"
                       "          f'es {A[\"τ medio\"]:.2f} meses (con duraciones esperadas, {res0.duracion}) y P(τ ≤ {plazo}) = {A[\"P(τ≤plazo)\"]:.2f}.')\n"
                       "_l.append(f'- **Sin VAN** se habría elegido la opción que más acorta ({tabla[\"τ medio\"].idxmin()}) y no la que más valor crea '\n"
                       "          f'({mejor})' + ('.' if np.isclose(dD, dB + dC, atol=0.5) else\n"
                       "                         f'; y juzgando cada medida por separado no se ve que B + C ({dB + dC:+.1f} k€) no es D ({dD:+.1f} k€).'))\n"
                       "Markdown('\\n'.join(_l))"))
    return celdas, f"límite de {L} operarios, plazo mes {plazo}, penalización {pen:.0f} k€/mes y VAN"


EVALUAR_OPCION = (
    "def evaluar(L_op, red, n=3000, semilla=21):\n"
    "    '''VAN neto (tras penalización y operarios extra) y probabilidad de cumplir de una opción (mismos números aleatorios).'''\n"
    "    dist = {a: ('triangular', max(1.0, d[1] - red[a]), max(1.0, d[2] - red[a]), max(1.0, d[3] - red[a])) for a, d in dist_base.items()}\n"
    "    bac = datos_proyecto['BAC'] + ucr.replace(np.inf, 0.0) * red\n"
    "    mc = proyecto.monte_carlo_van_recursos(dist, datos_proyecto[['operarios']], ['operarios'], [L_op], bac, F, N, k, n_samples=n, seed=semilla)\n"
    "    tau = mc.tau_recursos_samples\n"
    "    neto = mc.van_recursos_samples - penalizacion * np.maximum(tau - plazo, 0.0) - coste_operario * (L_op - L) * tau\n"
    "    return {'VAN medio': neto.mean(), 'P(VAN<0)': (neto < 0).mean(), 'P(τ≤plazo)': (tau <= plazo).mean(),\n"
    "            'τ medio': tau.mean(), 'coste reducción': float((ucr.replace(np.inf, 0.0) * red).sum())}\n")


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
    Variante("contingencia-planta", 3, "plan de contingencia: límite de operarios, duraciones triangulares, aceleración con recursos, penalización del cliente y VAN; cuatro opciones con Monte Carlo",
             _contingencia_planta),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
