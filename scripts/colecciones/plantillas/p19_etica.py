"""Colección 19 — Ética en el desarrollo de proyectos: situaciones típicas (informes, estimaciones,
riesgos, recursos, contratos, pagos) en las que alguien propone o hace algo discutible. Cada caso
tiene tres bloques de preguntas: **A** (cálculo con las herramientas del curso: la magnitud que la
situación esconde o distorsiona), **B** (aplicar un criterio de referencia enunciado en la lección 19
—paráfrasis propia que remite al Código Ético del PMI, a la ICB4 de IPMA, al PMBOK o a una norma
legal— con respuesta determinada por el criterio) y **C** (deliberación abierta, evaluada por rúbrica).

Los textos del PMI, de IPMA y del PMBOK no se reproducen (son obras protegidas); los criterios están
redactados por el curso y remiten a la fuente. Las cifras legales van en CONSTANTES_LEGALES."""
import contextlib
import io

import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe)

META = dict(slug="19_etica", prefijo="Dilema",
            titulo="Ética en el desarrollo de proyectos",
            descripcion="Situaciones típicas de un proyecto (informes de avance, estimaciones, riesgos, horas extra, "
                        "compresión, línea base, pagos, adjudicaciones, holguras, coste hundido, ofertas) en las que "
                        "alguien propone algo discutible. Cada caso pide (A) calcular la magnitud que la situación "
                        "esconde o distorsiona, (B) aplicar un criterio de referencia con respuesta determinada "
                        "(Código Ético del PMI, ICB4, PMBOK o una norma legal, parafraseados en la lección 19) y "
                        "(C) deliberar y redactar la comunicación.",
            contexto=False, n_defecto=5)

SEMILLA_MC = 2024
N_MC = 20000

# ---------------------------------------------------------------------------
# Criterios de referencia (redacción propia del curso; cada uno remite a su fuente, que no se reproduce)
# ---------------------------------------------------------------------------
CRITERIOS = {
    "C1": ("Afirmación falsa",
           "Comunicar como cierto un dato que quien lo comunica sabe que no lo es (un porcentaje de avance, una causa, "
           "una oferta que no existe).",
           "Código Ético del PMI, estándar obligatorio 5.3.1; ICB4, Personas 2 «integridad personal y fiabilidad»"),
    "C2": ("Verdad a medias",
           "Comunicar un dato cierto omitiendo la parte que cambiaría su interpretación (dar la fecha de la ruta crítica "
           "sin la probabilidad de cumplirla).",
           "Código Ético del PMI, 5.3.1"),
    "C3": ("Información fuera de contexto",
           "Presentar un dato cierto en un marco que lleva a una conclusión equivocada (un índice calculado sobre una "
           "línea base reescrita).",
           "Código Ético del PMI, 5.3.1"),
    "C4": ("Retención de información",
           "Callar un hecho conocido que la otra parte necesitaría para decidir (un riesgo, una holgura, una discrepancia).",
           "Código Ético del PMI, 5.3.1 y 5.2.3"),
    "C5": ("Beneficio propio a costa de otro",
           "Obtener una ventaja (dinero, plazo, reputación) trasladando el coste a otra parte que no lo ha aceptado.",
           "Código Ético del PMI, 5.3.2 y 3.3.2"),
    "C6": ("Conflicto de interés no revelado",
           "Participar en una decisión (adjudicar, contratar, certificar) teniendo un interés personal en el resultado sin "
           "haberlo comunicado a los afectados y sin que estos hayan consentido; lo exigible es revelarlo y abstenerse.",
           "Código Ético del PMI, 4.3.1–4.3.3; ICB4, Perspectiva 3; en contratos públicos, LCSP art. 64"),
    "C7": ("Cumplimiento de la ley y de las normas de seguridad",
           "El proyecto no puede planificarse incumpliendo una norma legal o de seguridad aunque salga más barato o más "
           "rápido; una actividad exigida por una norma no es una opción de compresión.",
           "Código Ético del PMI, 2.3.1 y 2.2.1; ICB4, Perspectiva 3 «cumplimiento, estándares y regulaciones»; PMBOK 8, "
           "principio de calidad integrada"),
    "C8": ("Reserva declarada frente a colchón oculto",
           "Una reserva de contingencia dimensionada con un nivel de confianza y comunicada es una práctica correcta; el "
           "mismo margen escondido en las estimaciones individuales, sin decirlo, es retención de información (C4).",
           "PMBOK, reservas de contingencia; Código Ético del PMI, 5.2.3"),
    "C9": ("Línea base y control de cambios",
           "La línea base sólo cambia mediante una solicitud de cambio aprobada; medir el avance contra una línea base "
           "reescrita sin aprobación es presentar un índice fuera de contexto (C3).",
           "PMBOK 8, dominio de gobernanza (control integrado de cambios)"),
    "C10": ("Buena fe en la negociación",
            "No revelar la propia alternativa o el precio de reserva es legítimo; afirmar hechos falsos para presionar (una "
            "oferta inexistente) no lo es (C1).",
            "Código Ético del PMI, 3.3.1 y 5.3.1; ICB4, Personas 9 «negociación»"),
    "C11": ("Reconocer los errores propios",
            "Quien decide asume las consecuencias de sus decisiones; atribuir a otros un error propio, o seguir adelante "
            "para no reconocerlo, incumple este criterio.",
            "Código Ético del PMI, 2.2.4; ICB4, Personas 2, indicador «asumir la responsabilidad de las propias decisiones»"),
    "C12": ("Decidir sobre el futuro, no sobre lo gastado",
            "Continuar un proyecto se justifica sólo por el valor actual de los flujos que quedan; lo ya gastado no es "
            "argumento ni a favor ni en contra, y decir «así recuperamos la inversión» cuando el VAN restante es negativo es "
            "una afirmación falsa (C1).",
            "Criterio económico de la lección 17; Código Ético del PMI, 5.2.1"),
    "C13": ("Compromisos de buena fe",
            "Prometer un plazo o un precio que se sabe inalcanzable, contando con renegociarlo después, es un compromiso "
            "que no se hace de buena fe.",
            "Código Ético del PMI, 5.2.4 y 2.2.3"),
}

# Cifras legales: paráfrasis + referencia; comprobadas en fuentes secundarias el 20-10-2026, verificar en el BOE consolidado
CONSTANTES_LEGALES = {
    "penalidad_demora": dict(valor=0.60, unidad="€ por cada 1 000 € del precio (IVA excluido) y día natural de demora",
                             texto="penalidad diaria por demora en la ejecución de un contrato del sector público, salvo que el "
                                   "pliego fije otra: 0,60 € por cada 1 000 € del precio del contrato (IVA excluido) por día",
                             ref="Ley 9/2017 de Contratos del Sector Público, art. 193.3"),
    "pago_maximo": dict(valor=60, unidad="días naturales",
                        texto="plazo de pago entre empresas: 30 días naturales si no se pacta otro y, por pacto, nunca más de 60",
                        ref="Ley 3/2004 de lucha contra la morosidad, art. 4"),
    "horas_extra": dict(valor=80, unidad="horas por trabajador y año",
                        texto="máximo de horas extraordinarias: 80 por trabajador y año, sin contar las compensadas con descanso "
                              "en los cuatro meses siguientes",
                        ref="Estatuto de los Trabajadores (RDL 2/2015), art. 35.2"),
    "jornada": dict(valor=40, unidad="horas semanales",
                    texto="jornada ordinaria máxima: 40 horas semanales de trabajo efectivo de promedio en cómputo anual",
                    ref="Estatuto de los Trabajadores, art. 34.1"),
    "temeridad": dict(valor=(25, 20, 10), unidad="puntos porcentuales",
                      texto="presunción de oferta anormalmente baja cuando el único criterio es el precio: un licitador, baja "
                            "superior a 25 puntos sobre el presupuesto; dos, la que sea inferior en más de 20 puntos a la "
                            "otra; tres o más, las inferiores en más de 10 puntos a la media de las ofertas (dejando fuera de "
                            "la media las ofertas superiores a ella en más de 10 puntos y, si quedan menos de tres, "
                            "tomando las tres más bajas). La presunción obliga a pedir justificación y, si no convence, "
                            "a excluir la oferta",
                      ref="Reglamento general de la Ley de Contratos (RD 1098/2001), art. 85; LCSP art. 149"),
    "subvenciones": dict(valor=None, unidad="",
                         texto="sólo son subvencionables los gastos que responden de manera indubitada a la naturaleza de la "
                               "actividad subvencionada y se realizan en el plazo fijado; el gasto imputado indebidamente "
                               "se reintegra con intereses",
                         ref="Ley 38/2003 General de Subvenciones, arts. 31 y 37"),
}

RUBRICA_C = ("La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: "
             "(1) identificar a todos los afectados y qué pierde o gana cada uno; (2) formular al menos dos alternativas "
             "además de la que se propone en el enunciado; (3) someter cada alternativa a las tres pruebas —legalidad "
             "(¿cumple la norma?), publicidad (¿la defendería si se hiciera pública?) y reversibilidad (¿la aceptaría en "
             "el papel del afectado?)—; (4) decidir y redactar la comunicación con las cifras del bloque A. "
             "Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.")


def _silencio():
    return contextlib.redirect_stdout(io.StringIO())


def _criterios(ids, legales=()):
    """Celda con los criterios de referencia del caso (paráfrasis propia + fuente) y, si procede, las cifras legales."""
    lineas = ["## Criterio de referencia",
              "",
              "Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 19 "
              "(redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta "
              "se justifica citando el criterio y la fuente, no con una opinión.",
              ""]
    for c in ids:
        titulo, texto, fuente = CRITERIOS[c]
        lineas.append(f"- **{c} — {titulo}.** {texto} *(cf. {fuente}.)*")
    for l in legales:
        r = CONSTANTES_LEGALES[l]
        lineas.append(f"- **Norma — {r['texto']}** *({r['ref']}.)*")
    return md("\n".join(lineas))


def _mensaje(de, texto):
    return md(f"> **De: {de}**\n> {texto}")


def _rubrica():
    return md(RUBRICA_C)


def _gift(preguntas_test, titulo):
    """Celda que define el test del bloque B (lista de (pregunta, [correcta, distractores…], retroalimentación))
    y lo exporta a GIFT, como en la colección 14."""
    return code(preguntas_test + "\n"
                "import os\n"
                "def a_gift(preguntas_test, titulo, semilla=1):\n"
                "    r = np.random.default_rng(semilla)\n"
                "    lineas = [f'$CATEGORY: {titulo}', '']\n"
                "    for i, (q, ops, fb) in enumerate(preguntas_test, 1):\n"
                "        correcta = ops[0]\n"
                "        todas = [ops[j] for j in r.permutation(len(ops))]\n"
                "        cuerpo = ' '.join((f'={v}' if v == correcta else f'~{v}') for v in todas)\n"
                "        lineas.append(f'::{titulo} P{i}::{q} {{{cuerpo} ####{fb}}}')\n"
                "        lineas.append('')\n"
                "    return '\\n'.join(lineas)\n"
                f"gift = a_gift(preguntas_test, {titulo!r})\n"
                "os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)\n"
                "with open(os.path.expanduser('~/temp/cuestionario_etica.gift'), 'w', encoding='utf8') as f:\n"
                "    f.write(gift)\n"
                "print(gift)")


def _proy_desde(df):
    with _silencio():
        return P.PredecessorTable.from_dataframe_of_strings(df, activity="activity", predecessor="predecessor").create_project()


def _T(p, d):
    with _silencio():
        return int(p.duration(d))


def _prob(p, d, sd, plazo):
    with _silencio():
        R = p.path_matrix(dummies=False)
    L = R @ d.reindex(R.columns)
    V = R @ (sd.reindex(R.columns).fillna(0) ** 2)
    T = float(L.max()); sigma = float(np.sqrt(V[L == T].max()))
    return T, sigma, float(norm.cdf((plazo - T) / sigma))


def _ackoff(p, d, tope, cur, r):
    with _silencio():
        _, pasos, dur, _ = p.ackoff(d, tope, cur, reduction=r, report=False)
    coste = float(sum(cur[a] for acts in pasos.values() for a in acts))
    return pasos, dur.reindex(d.index), coste


def _cargas(info, df):
    with _silencio():
        g = info["proyecto"].copy()
        gantt_df, _ = g.gantt_cargas(df.copy(), "duration", "resources", report=False, tikz=False)
    return gantt_df.data.loc["Total"].drop("H_total", errors="ignore").astype(float)


SETUP = ("datos_proyecto = datos_enunciado.copy()\n"
         "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
         "proyecto = prelaciones.create_project()\n"
         "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
         "duracion = int(proyecto.duration(datos_proyecto['duration']))\n")

INDICADORES = ("def indicadores(seguimiento, calendario_plan, semana):\n"
               "    datos_vg = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
               "    detalle = detalle_valor_ganado(datos_vg, calendario_plan, semana)\n"
               "    PV, EV, AC = float(detalle['PV_i'].sum()), float(detalle['EV_i'].sum()), float(detalle['AC'].sum())\n"
               "    return detalle, pd.Series({'PV': PV, 'EV': EV, 'AC': AC, 'CV': EV - AC, 'SV': EV - PV, 'CPI': EV / AC, 'SPI': EV / PV})\n")


# ---------------------------------------------------------------------------
# Caso 1 — El informe maquillado: «el avance es lo gastado» (EV := AC)
# ---------------------------------------------------------------------------
def _datos_informe(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22 and i["n_criticas"] == 1):
            return False
        D, cal, T = i["D"], i["calendario"], int(i["T"])
        BAC = pd.Series(rng.integers(2, 8, size=len(D)) * 10, index=D.index, name="BAC")
        t = int(round(0.5 * T))
        frac = ((t - cal["inicio_mas_temprano"]) / D).clip(0, 1)
        en_curso = (frac > 0) & (frac < 1)
        if en_curso.sum() < 2 or (frac == 1).sum() < 1:
            return False
        PV = BAC * frac
        g = pd.Series(np.where(en_curso, rng.choice([1.1, 1.2, 1.25], size=len(D)), 1.0), index=D.index)
        AC = (PV * g).round().astype(int)
        if (AC > BAC).any():
            return False
        h = pd.Series(rng.choice([0.5, 0.6, 0.7], size=len(D)), index=D.index)
        perf_ver = (frac * 100).round().astype(int)
        perf_ver[en_curso] = (frac[en_curso] * h[en_curso] * 100).round().astype(int)
        perf_inf = (AC / BAC * 100).round().astype(int)
        trans = (t - cal["inicio_mas_temprano"]).clip(lower=0).clip(upper=D).astype(int)

        def ind(perf):
            EV = float((BAC * perf / 100).sum())
            return EV, EV / float(AC.sum()), EV / float(PV.sum())

        EVi, CPIi, SPIi = ind(perf_inf)
        EVv, CPIv, SPIv = ind(perf_ver)
        if not (SPIi >= 1.0 and CPIv <= 0.85 and SPIv <= 0.9):
            return False
        e.update(BAC=BAC, t=t, PV=PV, AC=AC, perf_inf=perf_inf, perf_ver=perf_ver, trans=trans,
                 EVi=EVi, CPIi=CPIi, SPIi=SPIi, EVv=EVv, CPIv=CPIv, SPIv=SPIv, en_curso=list(D.index[en_curso]))
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    return info, e


def _informe_maquillado(rng):
    info, e = _datos_informe(rng)
    D, T = info["D"], int(info["T"])
    BAC, t = e["BAC"], e["t"]
    BAC_total = float(BAC.sum())
    df = P.tabla_base(info, duration=D, BAC=BAC)
    seg = pd.DataFrame({"actual_duration": e["trans"], "AC": e["AC"], "performed_informe": e["perf_inf"],
                        "performed_verificado": e["perf_ver"]})
    seg.index.name = "activity"
    EAC_v = BAC_total / e["CPIv"]
    celdas = [md("# Enunciado"),
              md("Usted acaba de incorporarse al equipo de un proyecto como responsable de control. El cuadro de prelaciones, "
                 "las duraciones planificadas (semanas) y el presupuesto de cada actividad (BAC, k€) se adjuntan; el coste "
                 f"planificado se reparte uniformemente en la duración de cada actividad. El plazo es de {T} semanas."),
              code('#| tbl-cap: "Planificación del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Al final de la semana {t} el jefe de proyecto prepara el informe mensual para el comité. Su método para el "
                 "porcentaje realizado de cada actividad es dividir el coste incurrido entre el presupuesto (columna "
                 "`performed_informe`). Usted ha recorrido el proyecto con los responsables y ha anotado el avance físico "
                 "verificado (columna `performed_verificado`)."),
              code(f'#| tbl-cap: "Seguimiento al final de la semana {t}: semanas trabajadas, coste incurrido (k€) y dos versiones del avance"\n'
                   '#| label: tbl-seguimiento\n' + literal_dataframe(seg, nombre="seguimiento")),
              _mensaje("Jefe de proyecto",
                       "El informe va con mi método de siempre: avance = gastado / presupuesto. Así los índices salen "
                       "razonables y no hay que discutir porcentajes con nadie. Sé que lo que hay en obra va algo más lento, "
                       "pero el comité sólo mira el SPI y el CPI, y con tu versión nos ponen en rojo."),
              _criterios(["C1", "C3"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule PV, EV, AC, CV, SV, CPI y SPI en la semana de control con la versión del informe y con la "
        "verificada, actividad a actividad y en total, y la previsión de coste final EAC = BAC / CPI con cada una.",
        "(A) Explique por qué con el método del informe el CPI vale exactamente 1 sea cual sea el sobrecoste, y qué "
        "magnitud mide en realidad ese «porcentaje realizado».",
        "(B) ¿Qué criterio de referencia incumple el informe tal como lo quiere enviar el jefe de proyecto y qué forma de "
        "engaño es? ¿Qué cifras tendría que contener el informe para cumplir el criterio?",
        "(C) Usted firma el informe junto al jefe de proyecto. ¿Qué hace y qué dice a cada parte (jefe de proyecto, "
        "comité)? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"semana_control, plazo = {t}, {T}\n" + SETUP + "BAC_total = float(datos_proyecto['BAC'].sum())\n" + INDICADORES))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "PV, EV y AC actividad a actividad con el avance del informe (EV = AC por construcción)"\n'
                       '#| label: tbl-detalleInforme\n'
                       "seg_informe = seguimiento[['actual_duration', 'AC']].assign(performed=seguimiento['performed_informe'])\n"
                       "seg_verificado = seguimiento[['actual_duration', 'AC']].assign(performed=seguimiento['performed_verificado'])\n"
                       "detalle_inf, ind_inf = indicadores(seg_informe, calendario, semana_control)\n"
                       "detalle_ver, ind_ver = indicadores(seg_verificado, calendario, semana_control)\n"
                       "detalle_inf[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(code('#| tbl-cap: "PV, EV y AC con el avance físico verificado"\n#| label: tbl-detalleVerificado\n'
                       "detalle_ver[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(code('#| tbl-cap: "Indicadores y previsión de coste final con las dos versiones"\n#| label: tbl-indicadores\n'
                       "comparacion = pd.DataFrame({'informe (EV = AC)': ind_inf, 'verificado': ind_ver})\n"
                       "comparacion.loc['EAC = BAC / CPI'] = [BAC_total / ind_inf['CPI'], BAC_total / ind_ver['CPI']]\n"
                       "comparacion.loc['VAC = BAC − EAC'] = BAC_total - comparacion.loc['EAC = BAC / CPI']\n"
                       "comparacion.round(3)"))
    celdas.append(md(f"Con el informe, SPI = {e['SPIi']:.2f} («en plazo o adelantados») y CPI = 1,00; con el avance verificado, "
                     f"SPI = {e['SPIv']:.2f} y CPI = {e['CPIv']:.2f}, y el coste final previsto pasa de {BAC_total:.0f} k€ a "
                     f"{EAC_v:.0f} k€ ({EAC_v - BAC_total:.0f} k€ de sobrecoste). La diferencia está sólo en EV: las actividades "
                     f"en curso ({', '.join(e['en_curso'])}) han gastado más de lo que han avanzado."))
    celdas += apartado(2)
    celdas.append(md("Si el porcentaje realizado se define como AC / BAC, entonces EV = BAC · (AC / BAC) = AC para cada "
                     "actividad, y por tanto CPI = EV / AC = 1 siempre, con independencia de lo que se haya gastado de más. El "
                     "«porcentaje realizado» del informe no mide avance: mide **consumo de presupuesto**. Y el SPI del informe "
                     "compara lo gastado con lo planificado (AC / PV): una actividad que gasta de más aparece como adelantada. "
                     "El método anula por construcción las dos señales que el valor ganado existe para dar."))
    celdas += apartado(3)
    celdas.append(md("**C1 (afirmación falsa).** El informe afirma un avance que el jefe de proyecto sabe que no es el real "
                     "(«sé que lo que hay en obra va algo más lento»); no es una estimación discutible sino un dato que se "
                     "sabe falso, calculado con un método elegido porque da el resultado deseado. También **C3**: los índices "
                     "son cifras «ciertas» (salen de los datos) presentadas en un marco —un método de medida que no mide "
                     "avance— que lleva al comité a una conclusión equivocada. Para cumplir el criterio, el informe debe "
                     f"contener los indicadores con el avance verificado (SPI {e['SPIv']:.2f}, CPI {e['CPIv']:.2f}, EAC "
                     f"{EAC_v:.0f} k€) y, si se quiere, la versión del informe como «consumo de presupuesto», con su nombre."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Con el método «avance = gastado / presupuesto», el CPI vale…', ['exactamente 1 siempre', 'menos de 1 si hay sobrecoste', 'más de 1 si hay sobrecoste', 'lo mismo que con el avance físico'],\n"
        "     'EV = BAC·(AC/BAC) = AC, luego CPI = EV/AC = 1 sea cual sea el gasto.'),\n"
        "    ('Enviar al comité un avance que se sabe inferior al real es, según los criterios de la lección…', ['C1, afirmación falsa', 'C4, retención de información', 'C8, reserva declarada', 'ningún criterio: es una estimación'],\n"
        "     'Es un dato que quien lo comunica sabe falso; no una omisión ni una estimación.'),\n"
        f"    ('¿Qué indicador cambia entre las dos versiones del informe?', ['sólo EV (y con él CPI y SPI)', 'sólo AC', 'PV y AC', 'ninguno'],\n"
        "     'PV depende del plan y AC de lo gastado; el porcentaje de avance sólo entra en EV.'),\n"
        "]", "Ética 19 — informe maquillado"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md("Un camino defendible: no firmar el informe con el método del gasto; proponer al jefe de proyecto enviar "
                     "el avance verificado con una nota de método (quién lo midió y cómo) y un plan de recuperación para las "
                     f"actividades en curso; si se niega, hacer constar por escrito la discrepancia ({e['EVi'] - e['EVv']:.0f} k€ "
                     "de valor declarado y no ejecutado) y elevarla a quien corresponda en la organización. Al comité: los "
                     f"índices verificados, el EAC de {EAC_v:.0f} k€ y las medidas. Lo que no cabe: firmar y «ya lo "
                     "corregiremos el mes que viene» (publicidad: ¿defendería ese informe si el comité supiera cómo se ha "
                     "calculado?)."))
    return celdas, f"informe con avance = gastado/presupuesto: SPI {e['SPIi']:.2f} y CPI 1,00 frente a {e['SPIv']:.2f} y {e['CPIv']:.2f} verificados"


# ---------------------------------------------------------------------------
# Caso 2 — La verdad a medias: «la ruta crítica termina en la semana T»
# ---------------------------------------------------------------------------
def _verdad_a_medias(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22 and i["n_criticas"] == 1):
            return False
        if i["segunda"] is None or i["T"] - i["segunda"] > 2:
            return False
        a, m, b = P.tres_estimaciones(rng, i["D"])
        sd = ((b - a) / 6).astype(float)
        F = int(i["T"]) + int(rng.choice([0, 1, 1, 2]))
        T, sigma, p_pert = _prob(i["proyecto"], i["D"].astype(float), sd, F)
        if not (0.5 <= p_pert <= 0.85):
            return False
        dists = {act: ("normal", float(i["D"][act]), float(sd[act])) for act in i["D"].index}
        with _silencio():
            res = i["proyecto"].monte_carlo(dists, n_samples=N_MC, seed=SEMILLA_MC)
        tau = np.asarray(res.tau_samples, dtype=float)
        p_mc = float((tau <= F).mean())
        if p_mc > p_pert - 0.04:
            return False
        e.update(a=a, m=m, b=b, sd=sd, F=F, sigma=sigma, p_pert=p_pert, p_mc=p_mc,
                 p80=float(np.percentile(tau, 80)), p50=float(np.percentile(tau, 50)))
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    df = P.tabla_base(info, a=e["a"], m=e["m"], b=e["b"])
    F = e["F"]
    celdas = [md("# Enunciado"),
              md("El cliente de un proyecto de instalación pregunta por escrito si la obra estará terminada en la semana "
                 f"{F}, porque ese día tiene comprometida la apertura. El cuadro de prelaciones y las tres estimaciones de "
                 "duración de cada actividad (optimista, más probable y pesimista, en semanas) se adjuntan."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Director comercial",
                       f"Contéstale que «según la planificación, la ruta crítica termina en la semana {T}», que es verdad y "
                       f"es antes de la {F}. No le des probabilidades: si le hablas de un 60 % se pone nervioso y empieza a "
                       "pedir penalizaciones."),
              _criterios(["C2", "C4"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule la duración esperada del proyecto, la desviación típica del camino crítico y la probabilidad de "
        f"terminar en la semana {F} o antes con la aproximación normal del PERT.",
        "(A) Repita el cálculo con una simulación Monte Carlo (20 000 réplicas, duraciones normales con la media y la "
        "desviación típica PERT de cada actividad) y obtenga además las semanas que se cumplirían con probabilidad "
        "0,5 y 0,8. ¿Por qué la simulación da menos probabilidad que el PERT?",
        f"(B) La frase «la ruta crítica termina en la semana {T}» es cierta. ¿Cumple el criterio de referencia como "
        "respuesta a la pregunta del cliente? ¿Qué forma de engaño es y qué dato falta para que deje de serlo?",
        "(C) Redacte la respuesta al cliente y la nota interna al director comercial. Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"fecha_cliente = {F}\n"
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['duration'] = (datos_proyecto['a'] + 4 * datos_proyecto['m'] + datos_proyecto['b']) / 6\n"
                       "datos_proyecto['standard_deviation'] = (datos_proyecto['b'] - datos_proyecto['a']) / 6\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion = float(proyecto.duration(datos_proyecto['duration']))\n"
                       "sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       "z = (fecha_cliente - duracion) / sigma\n"
                       "p_pert = float(norm.cdf(z))"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Media y varianza PERT de cada actividad"\n#| label: tbl-estimaciones\ndetalle_estimaciones(datos_enunciado)'))
    celdas.append(code('#| tbl-cap: "Duración y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "display(detalle_rutas(R, datos_proyecto['duration'], 'duración', 2))\n"
                       "detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)"))
    celdas.append(code("Markdown(f'''Duración esperada **{duracion:.2f}** semanas, σ del camino crítico **{sigma:.2f}**; "
                       "z = ({fecha_cliente} − {duracion:.2f}) / {sigma:.2f} = {z:.2f} y **P(T ≤ {fecha_cliente}) = {p_pert:.2f}** "
                       "con la aproximación normal del PERT.''')"))
    celdas += apartado(2)
    celdas.append(code(f"dists = {{act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}}\n"
                       f"resultado = proyecto.monte_carlo(dists, n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "tau = np.asarray(resultado.tau_samples, dtype=float)\n"
                       "p_mc = float((tau <= fecha_cliente).mean())\n"
                       "p50, p80 = np.percentile(tau, 50), np.percentile(tau, 80)\n"
                       "display(resultado.summary().round(2))\n"
                       "Markdown(f'''Monte Carlo: **P(T ≤ {fecha_cliente}) = {p_mc:.2f}** (PERT: {p_pert:.2f}); mediana {p50:.1f} semanas; "
                       "con probabilidad 0,8 se termina en **{p80:.1f}** semanas.''')"))
    celdas.append(code('#| tbl-cap: "Índice de criticidad de las actividades"\n#| label: tbl-criticidad\n'
                       "(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%)').T"))
    celdas.append(md("La simulación da menos probabilidad porque hay rutas casi críticas: el PERT sólo mira la ruta de "
                     "mayor media, pero en muchas réplicas la más larga es otra (sesgo de fusión). Cuanto más parecidas son "
                     "las rutas, más optimista es el PERT."))
    celdas += apartado(3)
    celdas.append(md(f"**C2 (verdad a medias).** La frase es cierta y, aun así, no responde a lo que el cliente pregunta: él "
                     f"quiere saber si puede contar con la semana {F}, y la respuesta honesta es «con una probabilidad de "
                     f"{e['p_mc']:.0%} ({e['p_pert']:.0%} con el PERT)». Dar la duración media sin la dispersión lleva al "
                     f"cliente a concluir que la fecha está asegurada; el dato que falta es la **probabilidad** (o la semana que "
                     f"sí se puede comprometer con un 80 %: la {e['p80']:.0f}). Omitirlo a sabiendas, como pide el director "
                     "comercial, es además **C4**. No es C1: no se afirma nada falso."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        f"    ('«La ruta crítica termina en la semana {T}» como respuesta a «¿estará en la semana {F}?» es…', ['C2, verdad a medias', 'C1, afirmación falsa', 'una respuesta completa', 'C6, conflicto de interés'],\n"
        "     'El dato es cierto pero omite la dispersión, que cambia la conclusión.'),\n"
        "    ('El dato que convierte la respuesta en honesta es…', ['la probabilidad de cumplir la fecha', 'la suma de las duraciones', 'el número de rutas', 'el coste del proyecto'],\n"
        "     'La media sin la probabilidad no dice si la fecha es alcanzable.'),\n"
        "    ('La simulación Monte Carlo da menos probabilidad que el PERT porque…', ['hay rutas casi críticas (sesgo de fusión)', 'usa menos réplicas', 'las duraciones son enteras', 'el PERT es siempre pesimista'],\n"
        "     'El PERT sólo mira la ruta de mayor media; en muchas réplicas la más larga es otra.'),\n"
        "]", "Ética 19 — verdad a medias"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Respuesta al cliente (modelo): «La planificación da una duración esperada de {T} semanas; la "
                     f"probabilidad de terminar en la semana {F} es del {e['p_mc']:.0%}. La semana que podemos comprometer "
                     f"con un 80 % de confianza es la {e['p80']:.0f}. Si la apertura exige la {F}, podemos estudiar qué "
                     "actividades acortar y a qué coste.» Nota interna: el cliente que descubre después que la fecha tenía "
                     f"un {1 - e['p_mc']:.0%} de riesgo pedirá las penalizaciones igualmente, y con razón; la probabilidad "
                     "no es una debilidad comercial sino la base para vender la reducción de plazo (lección 10)."))
    return celdas, f"«la ruta crítica acaba en la semana {T}» frente a P(T ≤ {F}) = {e['p_pert']:.2f} (PERT) y {e['p_mc']:.2f} (MC)"


# ---------------------------------------------------------------------------
# Caso 3 — Colchones ocultos frente a reserva declarada
# ---------------------------------------------------------------------------
def _colchones_ocultos(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22 and i["n_criticas"] == 1):
            return False
        a, m, b = P.tres_estimaciones(rng, i["D"])
        if int((b - i["D"]).sum()) < 6:
            return False
        sd = ((b - a) / 6).astype(float)
        T, sigma, _ = _prob(i["proyecto"], i["D"].astype(float), sd, 0)
        T80 = T + norm.ppf(0.8) * sigma
        T_pad = _T(i["proyecto"], b.astype(int))
        T_jp = int(np.ceil(1.1 * T_pad))
        if T_pad - T < 2 * (T80 - T) or T_pad - T < 3 or T_pad > 1.8 * T:
            return False
        e.update(a=a, m=m, b=b, sd=sd, sigma=sigma, T80=T80, T_pad=T_pad, T_jp=T_jp)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    df = P.tabla_base(info, a=e["a"], m=e["m"], b=e["b"], estimacion_entregada=e["b"])
    reserva = e["T80"] - T
    celdas = [md("# Enunciado"),
              md("En la reunión de planificación cada responsable de actividad entrega una única duración (columna "
                 "`estimacion_entregada`, semanas). Usted, que ha hablado con ellos uno a uno, sabe que cada uno ha dado su "
                 "**peor caso** (la estimación pesimista) «por si acaso», sin decirlo. Las tres estimaciones reales de cada "
                 "actividad (optimista, más probable y pesimista) se adjuntan."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Jefe de proyecto",
                       f"Con las estimaciones entregadas el proyecto sale en {e['T_pad']} semanas. Yo le añado un 10 % y "
                       f"prometo {e['T_jp']} al cliente; así nunca llegamos tarde. Lo del colchón de cada uno no hace falta "
                       "contarlo, cada cual gestiona su margen."),
              _criterios(["C8", "C4"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Con las tres estimaciones calcule la duración esperada, la desviación típica del camino crítico y la "
        "duración que se cumpliría con probabilidad 0,8 (reserva de contingencia necesaria para ese nivel de confianza).",
        "(A) Calcule la duración con las estimaciones entregadas y con el 10 % del jefe de proyecto, y compare el margen "
        "total oculto con la reserva necesaria. ¿Por qué sumar los peores casos de cada actividad sobrestima tanto?",
        "(B) ¿Qué criterio distingue el margen que propone el jefe de proyecto de una reserva de contingencia correcta? "
        "¿Qué habría que cambiar en el informe de planificación para que cumpla el criterio?",
        "(C) ¿Cómo plantearía la planificación al equipo y al cliente? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['duration'] = (datos_proyecto['a'] + 4 * datos_proyecto['m'] + datos_proyecto['b']) / 6\n"
                       "datos_proyecto['standard_deviation'] = (datos_proyecto['b'] - datos_proyecto['a']) / 6\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion = float(proyecto.duration(datos_proyecto['duration']))\n"
                       "sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       "T80 = duracion + norm.ppf(0.8) * sigma\n"
                       "reserva = T80 - duracion"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Media y varianza PERT de cada actividad"\n#| label: tbl-estimaciones\ndetalle_estimaciones(datos_enunciado)'))
    celdas.append(code('#| tbl-cap: "Duración y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "display(detalle_rutas(R, datos_proyecto['duration'], 'duración', 2))\n"
                       "detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)"))
    celdas.append(code("Markdown(f'''Duración esperada **{duracion:.2f}** semanas, σ = {sigma:.2f}; con probabilidad 0,8 se termina en "
                       "**{T80:.2f}** semanas: la reserva de contingencia para ese nivel de confianza es **{reserva:.2f}** semanas "
                       "(z₀,₈ · σ = 0,84 · {sigma:.2f}).''')"))
    celdas += apartado(2)
    celdas.append(code("duracion_entregada = int(proyecto.duration(datos_proyecto['estimacion_entregada']))\n"
                       "duracion_jp = int(np.ceil(1.1 * duracion_entregada))\n"
                       "camino_entregado = ' / '.join('–'.join(r) for r in proyecto.critical_path(datos_proyecto['estimacion_entregada']).values())\n"
                       "Markdown(f'''Con las estimaciones entregadas: **{duracion_entregada}** semanas (ruta {camino_entregado}); con el 10 % del jefe "
                       "de proyecto: **{duracion_jp}**. Margen oculto total: {duracion_jp - duracion:.1f} semanas frente a las {reserva:.2f} que "
                       "justifica el 80 %: **{(duracion_jp - duracion) / reserva:.1f} veces** la reserva necesaria.''')"))
    celdas.append(md("Sumar los peores casos sobrestima porque las desviaciones no se suman: se suman las **varianzas**, y la "
                     "raíz de la suma es mucho menor que la suma de las raíces. Que todas las actividades de una ruta salgan a "
                     "la vez en su peor caso es un suceso improbable; el margen que protege al proyecto se calcula sobre la "
                     "ruta, no actividad a actividad, y una sola vez."))
    celdas += apartado(3)
    celdas.append(md(f"**C8.** Una reserva de contingencia correcta se **dimensiona** con un nivel de confianza ({reserva:.1f} "
                     f"semanas para el 80 %) y se **declara**: el cliente ve la duración esperada ({T} semanas), la reserva y a "
                     f"quién pertenece. El margen del enunciado se esconde dos veces (cada responsable en su estimación y el jefe "
                     f"de proyecto en su 10 %), no se dimensiona y no se comunica: es **C4**, retención de información, y cuesta "
                     f"{e['T_jp'] - T} semanas de plazo al cliente. Para cumplir el criterio, el informe de planificación "
                     f"debe llevar las tres estimaciones por actividad, la duración esperada, la reserva calculada y su nivel "
                     "de confianza."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Una reserva de contingencia cumple el criterio C8 cuando…', ['se dimensiona con un nivel de confianza y se declara', 'cada responsable la incluye en su estimación', 'el jefe de proyecto la añade sin comunicarla', 'es como mínimo un 10 %'],\n"
        "     'Lo que distingue la reserva del colchón es que se calcula y se comunica.'),\n"
        "    ('Sumar los peores casos de las actividades de la ruta crítica…', ['sobrestima la duración porque las desviaciones no se suman', 'da la duración con probabilidad 0,8', 'da la duración esperada', 'subestima la duración'],\n"
        "     'Se suman varianzas; la desviación de la ruta es la raíz de esa suma.'),\n"
        "    ('Esconder un margen en cada estimación sin decirlo es…', ['C4, retención de información', 'C1, afirmación falsa', 'C8, reserva declarada', 'C6, conflicto de interés'],\n"
        "     'Se calla un dato (el margen) que el cliente y el jefe de proyecto necesitan para decidir.'),\n"
        "]", "Ética 19 — colchones ocultos"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al equipo: pedir las tres estimaciones a cada responsable y explicar que el margen se calcula una vez, "
                     f"sobre la ruta; los colchones individuales no protegen más y alargan el plazo. Al cliente: duración esperada "
                     f"de {T} semanas, reserva de {reserva:.1f} semanas para el 80 % y compromiso en {np.ceil(e['T80']):.0f} semanas, "
                     f"con la reserva visible y gestionada por el jefe de proyecto. Prometer {e['T_jp']} semanas «para no llegar "
                     "nunca tarde» supera la prueba de legalidad pero no la de publicidad: si el cliente viera la cuenta, no "
                     "aceptaría pagar el plazo de dos colchones."))
    return celdas, f"peores casos entregados + 10 %: {e['T_jp']} semanas frente a {T} esperadas y {np.ceil(e['T80']):.0f} con el 80 %"


# ---------------------------------------------------------------------------
# Caso 4 — El riesgo silenciado antes de la decisión de puerta
# ---------------------------------------------------------------------------
UMBRAL_COMITE = 0.7


def _riesgo_silenciado(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22):
            return False
        sd = P.desviaciones(rng, list(i["D"].index))
        plazo = int(i["T"]) + int(rng.choice([1, 2, 2, 3]))
        T, sigma, p0 = _prob(i["proyecto"], i["D"].astype(float), sd, plazo)
        if not (UMBRAL_COMITE + 0.05 <= p0 <= 0.95):
            return False
        criticas = [a for a in i["D"].index if i["calendario"].loc[a, "H_total"] == 0]
        X = str(rng.choice(criticas))
        p, delta = float(rng.choice([0.3, 0.4, 0.5])), int(rng.choice([2, 3, 4]))
        dists = {act: ("normal", float(i["D"][act]), float(sd[act])) for act in i["D"].index}
        with _silencio():
            res0 = i["proyecto"].monte_carlo(dict(dists), n_samples=N_MC, seed=SEMILLA_MC)
        rng_mc = np.random.default_rng(SEMILLA_MC)
        mu, s = float(i["D"][X]), float(sd[X])

        def con_riesgo(size=None):
            return rng_mc.normal(mu, s, size) + delta * (rng_mc.random(size) < p)

        dists1 = dict(dists); dists1[X] = con_riesgo
        with _silencio():
            res1 = i["proyecto"].monte_carlo(dists1, n_samples=N_MC, seed=SEMILLA_MC)
        p_mc0 = float((np.asarray(res0.tau_samples) <= plazo).mean())
        p_mc1 = float((np.asarray(res1.tau_samples) <= plazo).mean())
        if not (p_mc0 >= UMBRAL_COMITE and p_mc1 <= UMBRAL_COMITE - 0.05):
            return False
        e.update(sd=sd, plazo=plazo, X=X, p=p, delta=delta, p0=p0, p_mc0=p_mc0, p_mc1=p_mc1, sigma=sigma)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    X, p, delta, plazo = e["X"], e["p"], e["delta"], e["plazo"]
    df = P.tabla_base(info, duration=D, standard_deviation=e["sd"])
    celdas = [md("# Enunciado"),
              md(f"El comité de inversión decide el jueves si autoriza la fase de ejecución de un proyecto. Su regla, escrita "
                 f"en el procedimiento, es autorizar sólo si la probabilidad de terminar en el plazo contractual de {plazo} "
                 f"semanas es al menos del {UMBRAL_COMITE:.0%}. El cuadro de prelaciones, las duraciones medias y las "
                 "desviaciones típicas (semanas) se adjuntan."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"El lunes el jefe de proyecto recibe una llamada: el único proveedor cualificado de la actividad {X} ha "
                 f"entrado en concurso de acreedores. Su estimación, contrastada con compras, es que con probabilidad "
                 f"{p:.0%} habrá que cambiar de proveedor, lo que añadiría {delta} semanas a {X}."),
              _mensaje("Jefe de proyecto (al responsable de riesgos)",
                       "El informe para el comité está cerrado desde el viernes con la probabilidad que calculamos; no "
                       "voy a reabrirlo por una llamada. Si el proveedor cae, lo gestionamos como riesgo en ejecución. "
                       "No lo menciones el jueves."),
              _criterios(["C4"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Calcule la probabilidad de terminar en {plazo} semanas o menos sin el riesgo, con la aproximación normal del "
        "PERT y con una simulación Monte Carlo (20 000 réplicas, duraciones normales).",
        f"(A) Incorpore el riesgo a la simulación (la duración de {X} es normal más {delta} semanas con probabilidad "
        f"{p:.0%}) y recalcule la probabilidad. ¿Cambia la decisión del comité según su regla?",
        "(B) ¿Qué criterio de referencia incumple no mencionar el riesgo el jueves? ¿Es relevante que el riesgo pueda "
        "«gestionarse en ejecución»? ¿Qué convierte a la omisión en material?",
        "(C) ¿Qué hace el responsable de riesgos antes del jueves y qué dice al jefe de proyecto y al comité? Aplique la "
        "rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"plazo, umbral, X, p_riesgo, delta = {plazo}, {UMBRAL_COMITE}, {X!r}, {p}, {delta}\n"
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion = int(proyecto.duration(datos_proyecto['duration']))\n"
                       "sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       "p_pert = float(norm.cdf((plazo - duracion) / sigma))"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Duración y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "display(detalle_rutas(R, datos_proyecto['duration'], 'duración'))\n"
                       "detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)"))
    celdas.append(code(f"dists = {{act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}}\n"
                       f"sin_riesgo = proyecto.monte_carlo(dict(dists), n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "p_mc0 = float((np.asarray(sin_riesgo.tau_samples) <= plazo).mean())\n"
                       "Markdown(f'''Sin el riesgo: duración esperada {duracion}, σ = {sigma:.2f}, **P(T ≤ {plazo}) = {p_pert:.2f}** (PERT) y "
                       "**{p_mc0:.2f}** (Monte Carlo). Supera el umbral del {umbral:.0%}: el comité autorizaría.''')"))
    celdas += apartado(2)
    celdas.append(code(f"rng_mc = np.random.default_rng({SEMILLA_MC})\n"
                       "mu_X, sd_X = float(datos_proyecto.loc[X, 'duration']), float(datos_proyecto.loc[X, 'standard_deviation'])\n"
                       "def duracion_X_con_riesgo(size=None):\n"
                       "    '''Normal más delta semanas con probabilidad p_riesgo (cambio de proveedor).'''\n"
                       "    return rng_mc.normal(mu_X, sd_X, size) + delta * (rng_mc.random(size) < p_riesgo)\n"
                       "dists_riesgo = dict(dists); dists_riesgo[X] = duracion_X_con_riesgo\n"
                       f"con_riesgo = proyecto.monte_carlo(dists_riesgo, n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "p_mc1 = float((np.asarray(con_riesgo.tau_samples) <= plazo).mean())\n"
                       "comparacion = pd.DataFrame({'sin el riesgo': sin_riesgo.summary().iloc[:, 0], 'con el riesgo': con_riesgo.summary().iloc[:, 0]})\n"
                       "display(comparacion.round(2))\n"
                       "Markdown(f'''Con el riesgo: **P(T ≤ {plazo}) = {p_mc1:.2f}**, por debajo del umbral del {umbral:.0%}. "
                       "La decisión del comité cambia: con la información del lunes no autorizaría (o exigiría un plan de mitigación antes).''')"))
    celdas.append(md(f"Comprobación a mano: {X} es crítica, así que las {delta} semanas van enteras al plazo cuando el riesgo "
                     f"se materializa; la probabilidad es aproximadamente {1 - p:.0%} · P(T ≤ {plazo}) + {p:.0%} · "
                     f"P(T ≤ {plazo - delta}) = {(1 - p) * e['p0'] + p * norm.cdf((plazo - delta - T) / e['sigma']):.2f} "
                     "con el PERT, en línea con la simulación."))
    celdas += apartado(3)
    celdas.append(md(f"**C4 (retención de información).** El comité decide con una regla explícita sobre una cifra, y el jefe "
                     f"de proyecto sabe un hecho que la mueve de {e['p_mc0']:.2f} a {e['p_mc1']:.2f}, de un lado al otro del "
                     "umbral. Que el riesgo pueda gestionarse en ejecución no cambia nada: la decisión de autorizar es del "
                     "comité, y se le estaría quitando la posibilidad de tomarla con los datos. La omisión es **material** "
                     "precisamente porque la decisión cambia con el dato; ese es el test objetivo (si el riesgo dejara la "
                     "probabilidad por encima del umbral, seguiría siendo obligatorio informar —5.2.3— pero no cambiaría la "
                     "decisión)."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('No comunicar al comité un riesgo conocido que cambia su decisión es…', ['C4, retención de información', 'C2, verdad a medias', 'C8, reserva declarada', 'gestión de riesgos en ejecución'],\n"
        "     'Se calla un hecho conocido que la otra parte necesita para decidir.'),\n"
        "    ('La omisión es material cuando…', ['la decisión cambia con el dato omitido', 'el riesgo tiene probabilidad mayor que 0,5', 'el informe ya está cerrado', 'el dato llega por teléfono'],\n"
        "     'El test objetivo es si la cifra que gobierna la decisión cruza el umbral.'),\n"
        f"    ('Si {X} no fuera crítica, el efecto del riesgo en el plazo sería…', ['menor, absorbido en parte por su holgura', 'el mismo', 'mayor', 'nulo siempre'],\n"
        "     'Un retraso x en una actividad con holgura H retrasa el proyecto máx(0, x − H).'),\n"
        "]", "Ética 19 — riesgo silenciado"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md("Antes del jueves: recalcular (dos horas de trabajo), redactar una página con las dos probabilidades y un "
                     f"plan de mitigación (proveedor alternativo precalificado, acopio anticipado, acortar otra actividad "
                     "crítica) con su coste, y llevarla al jefe de proyecto. Al jefe de proyecto: «el informe tiene una cifra "
                     "que sabemos que ya no es cierta; si el comité lo descubre después, el problema no es el proveedor sino "
                     "nosotros». Al comité, con o sin el acuerdo del jefe de proyecto: la actualización y el plan; el criterio "
                     "2.3.2 del Código del PMI pide elevar la cuestión a quien corresponde cuando no se resuelve en el nivel "
                     "propio. Lo que no cabe: callar el jueves y «gestionarlo en ejecución»."))
    return celdas, f"riesgo en {X} ({p:.0%}, +{delta} semanas): P(T ≤ {plazo}) pasa de {e['p_mc0']:.2f} a {e['p_mc1']:.2f}, umbral {UMBRAL_COMITE:.0%}"


# ---------------------------------------------------------------------------
# Caso 5 — Las horas extra que exige el plan
# ---------------------------------------------------------------------------
def _horas_extra(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 20):
            return False
        rec = P.recursos(rng, list(i["D"].index), hi=4)
        df = P.tabla_base(i, duration=i["D"], resources=rec)
        try:
            total = _cargas(i, df)
        except Exception:
            return False
        pico = int(total.max())
        L = int(np.ceil(total.mean()))
        if L >= pico or L < 3 or L < int(rec.max()):
            return False
        exceso = (total - L).clip(lower=0)
        horas_persona = float(exceso.sum() * 40 / L)
        if not (90 <= horas_persona <= 400):
            return False
        with _silencio():
            _, g, _ = i["proyecto"].copy().asignar(data=df.copy(), duration_label="duration", resource_label="resources", maximo=[L], report=False)
        T_L = len(g.data.columns.drop("H_total"))
        e.update(rec=rec, df=df, total=total, L=L, pico=pico, exceso=exceso, horas_persona=horas_persona, T_L=T_L)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    L, pico, df = e["L"], e["pico"], e["df"]
    semanas_exceso = int((e["exceso"] > 0).sum())
    celdas = [md("# Enunciado"),
              md(f"Una empresa de montajes ejecuta el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y operarios "
                 f"necesarios por actividad se adjunta. La plantilla disponible es de **{L} operarios** a jornada completa "
                 "(40 horas semanales); no hay otros proyectos en marcha."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Jefe de proyecto",
                       f"El cliente quiere las {T} semanas del plan con inicios más tempranos y no vamos a contratar a "
                       "nadie. Las semanas en que hagan falta más de los que somos, se hacen horas extra y ya está: se "
                       "pagan y punto."),
              _criterios(["C7"], legales=["jornada", "horas_extra"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Dibuje el diagrama de cargas con inicios más tempranos y obtenga, semana a semana, el exceso de operarios "
        f"sobre los {L} disponibles. Traduzca el exceso a horas extraordinarias por operario en el año (reparto uniforme "
        "entre la plantilla).",
        f"(A) Calcule la duración del proyecto si se respeta el límite de {L} operarios (asignación con límite) y, "
        "alternativamente, cuántos operarios haría falta contratar para cumplir las semanas del plan sin horas extra.",
        "(B) Según el criterio y la norma de referencia, ¿puede ejecutarse el plan del jefe de proyecto? Justifique con la cifra.",
        "(C) ¿Qué propone al jefe de proyecto y al cliente? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"plantilla, jornada, maximo_horas_extra = {L}, 40, 80\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Gantt con inicios más tempranos y carga de operarios"\n#| label: fig-ganttCargas\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas frente a la plantilla disponible"\n#| label: fig-cargas\n'
                       "cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[plantilla])"))
    celdas.append(code('#| tbl-cap: "Exceso de operarios y horas extraordinarias semana a semana"\n#| label: tbl-exceso\n'
                       "exceso = (cargas - plantilla).clip(lower=0)\n"
                       "tabla = pd.DataFrame({'operarios necesarios': cargas, 'exceso sobre la plantilla': exceso,\n"
                       "                      'horas extra de la semana': exceso * jornada, 'horas extra por operario': exceso * jornada / plantilla})\n"
                       "tabla.loc['Total'] = tabla.sum()\n"
                       "horas_persona = float(tabla.loc['Total', 'horas extra por operario'])\n"
                       "display(tabla.round(1).T)\n"
                       "Markdown(f'''Pico de **{int(cargas.max())}** operarios frente a {plantilla}; exceso en {int((exceso > 0).sum())} semanas; "
                       "**{horas_persona:.0f} horas extraordinarias por operario** en el año (máximo legal: {maximo_horas_extra}).''')"))
    celdas += apartado(2)
    celdas.append(code("proyecto_asignado = proyecto.copy()\n"
                       "datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(data=datos_proyecto.copy(), duration_label='duration',\n"
                       "                                                                  resource_label='resources', maximo=[plantilla], report=False, params=GANTT_PARAMS)\n"
                       "duracion_con_limite = len(gantt_asignado.data.columns.drop('H_total'))\n"
                       "contratar = int(cargas.max()) - plantilla\n"
                       "Markdown(f'''Respetando los {plantilla} operarios el proyecto dura **{duracion_con_limite}** semanas ({duracion_con_limite - duracion} más que el plan); "
                       "para cumplir las {duracion} semanas sin horas extra habría que contratar **{contratar}** operarios durante las semanas de pico (o nivelar primero y ver si baja el pico).''')"))
    celdas.append(code('#| fig-cap: "Gantt con asignación limitada a la plantilla"\n#| label: fig-ganttAsignado\npinta_tikz(dibujo)'))
    celdas += apartado(3)
    celdas.append(md(f"**C7 y la norma.** El plan exige {e['horas_persona']:.0f} horas extraordinarias por operario en el año, "
                     f"más de las 80 que fija el Estatuto de los Trabajadores (art. 35.2) salvo las compensadas con descanso, "
                     "y supone sostener semanas por encima de la jornada ordinaria (art. 34.1). No es una opción de "
                     "planificación: una actividad o un plan que incumple una norma no está en la tabla de opciones, por "
                     "mucho que «se paguen». La cifra objetiva es la del apartado 1; con menos de 80 horas por operario el plan "
                     "sería legal (y entonces la pregunta pasaría al bloque C: si conviene)."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('El máximo de horas extraordinarias por trabajador y año fijado por el Estatuto de los Trabajadores es…', ['80', '40', '120', 'no hay máximo si se pagan'],\n"
        "     'Art. 35.2 ET: 80 horas al año, sin contar las compensadas con descanso.'),\n"
        "    ('Un plan que exige más horas extra de las legales…', ['no es una opción: se descarta por C7', 'es válido si el cliente lo acepta', 'es válido si se pagan', 'depende del jefe de proyecto'],\n"
        "     'El cumplimiento legal es condición previa, no un coste más en la tabla.'),\n"
        "    ('Para cumplir el plazo sin horas extra la alternativa objetiva es…', ['contratar operarios para las semanas de pico o aceptar la duración con límite', 'reducir la calidad', 'no hacer el diagrama de cargas', 'cambiar las prelaciones'],\n"
        "     'El diagrama de cargas da el pico y la asignación con límite da la nueva duración.'),\n"
        "]", "Ética 19 — horas extra"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Tabla de opciones para el jefe de proyecto: (1) aceptar {e['T_L']} semanas con la plantilla; (2) contratar "
                     f"{pico - L} operarios temporales las {semanas_exceso} semanas de pico y cumplir {T}; (3) nivelar y "
                     "combinar: parte en horas extra dentro de las 80 y el resto con refuerzo. Al cliente se le cuenta la "
                     "opción elegida con su coste y su fecha; lo que no se le cuenta es que el plazo «se consigue» "
                     "incumpliendo la ley (prueba de publicidad)."))
    return celdas, f"{L} operarios, pico {pico}: {e['horas_persona']:.0f} horas extra por operario frente a 80 legales; con límite, {e['T_L']} semanas"


# ---------------------------------------------------------------------------
# Caso 6 — La inspección que «sobra»: comprimir no es recortar alcance
# ---------------------------------------------------------------------------
def _inspeccion_suprimida(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22 and i["n_criticas"] == 1):
            return False
        if len(i["finales"]) != 1:
            return False
        I = i["finales"][0]
        if not (2 <= int(i["D"][I]) <= 3):
            return False
        tope, cn, ct, cur = P.costes_reduccion(rng, i["D"], n_inf=0)
        cur = cur.astype(float)                     # pandas 3 no admite inf en una Series entera
        tope[I], cur[I], ct[I] = int(i["D"][I]), np.inf, cn[I]
        r = int(rng.choice([2, 3]))
        pasos, dur, coste = _ackoff(i["proyecto"], i["D"].astype(float), tope, cur, r)
        if len(pasos) < r or coste <= 0:
            return False
        p_def, K = float(rng.choice([0.1, 0.15, 0.2])), int(rng.choice([150, 200, 300]))
        if p_def * K <= coste:
            return False
        e.update(I=I, tope=tope, cn=cn, ct=ct, cur=cur, r=r, pasos=pasos, coste=coste, p_def=p_def, K=K)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=1)
    D, T = info["D"], int(info["T"])
    I, r, coste = e["I"], e["r"], e["coste"]
    df = P.tabla_base(info, duration=D, cap_duration=e["tope"], ucr=e["cur"])
    dI = int(D[I])
    celdas = [md("# Enunciado"),
              md("Una instalación industrial termina con las **pruebas de presión y la inspección reglamentaria** "
                 f"(actividad {I}), exigidas por la normativa de seguridad antes de la puesta en servicio. El cuadro de "
                 "prelaciones, las duraciones (semanas), las duraciones mínimas alcanzables y el coste unitario de reducción "
                 f"(k€ por semana) se adjuntan; {I} no se puede acortar."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"El cliente exige adelantar la puesta en servicio **{r} semanas**. En una reunión, el director de "
                 "operaciones propone lo siguiente."),
              _mensaje("Director de operaciones",
                       f"Lo más barato es quitar {I}: son {dI} semanas a coste cero. Las pruebas las hacemos «en marcha» "
                       "el primer mes de servicio, que en la práctica es lo mismo, y el inspector firma después. Lo otro "
                       "es pagar reducciones."),
              md(f"Según el responsable de calidad, con probabilidad {e['p_def']:.0%} la inspección detecta un defecto que, "
                 f"si se pusiera en servicio sin corregir, costaría {e['K']} k€ (parada, reparación y daños)."),
              _criterios(["C7"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Obtenga con el algoritmo de Ackoff-Sasieni el coste mínimo de adelantar {r} semanas acortando actividades "
        "(sin tocar la inspección) y las actividades que se acortan en cada paso.",
        f"(A) Calcule el coste esperado de poner en servicio sin la inspección (probabilidad del defecto por su coste) y "
        "compárelo con el coste de la reducción legítima. ¿Qué «ahorra» de verdad la propuesta?",
        "(B) Según el criterio de referencia, ¿es la supresión de la inspección una opción de compresión? ¿Qué distingue "
        "comprimir el cronograma de recortar el alcance?",
        "(C) ¿Qué responde al director de operaciones y qué ofrece al cliente? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"inspeccion, semanas_pedidas, p_defecto, coste_defecto = {I!r}, {r}, {e['p_def']}, {e['K']}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=semanas_pedidas, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("coste_reduccion = float(sum(datos_proyecto.loc[acts, 'ucr'].sum() for acts in pasos.values()))\n"
                       "duracion_reducida = int(proyecto.duration(duraciones_reducidas))\n"
                       "Markdown(f'''Adelantar {semanas_pedidas} semanas acortando actividades cuesta **{coste_reduccion:.0f} k€** "
                       "(de {duracion} a {duracion_reducida} semanas); pasos: ''' + '; '.join(', '.join(a) for a in pasos.values()) + '.')"))
    celdas += apartado(2)
    celdas.append(code("coste_esperado_sin_inspeccion = p_defecto * coste_defecto\n"
                       "Markdown(f'''Sin la inspección, el coste esperado del defecto es {p_defecto:.0%} × {coste_defecto} = **{coste_esperado_sin_inspeccion:.0f} k€**, "
                       "frente a los {coste_reduccion:.0f} k€ de la reducción legítima. La propuesta no ahorra: traslada un coste mayor, incierto y "
                       "a cargo de otros (usuarios, aseguradora, inspector) a cambio de un ahorro cierto y pequeño.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**C7.** {I} está exigida por una norma de seguridad: no es una opción de compresión, de modo que no "
                     f"entra en la tabla de Ackoff (por eso su coste de reducción es infinito y no «cero»). Comprimir el "
                     "cronograma es hacer el **mismo alcance** en menos tiempo pagando por ello (reducción de duraciones, más "
                     "recursos, solapes); quitar una actividad es **recortar el alcance**, y cuando la actividad es un "
                     "requisito legal, incumplir la norma. Hacer las pruebas «en marcha» no es lo mismo: la norma exige la "
                     "inspección antes de la puesta en servicio, y el inspector que firma después firma algo que no ha visto. "
                     "La respuesta no depende del cálculo del apartado 2; el cálculo sólo muestra que, además, es mal negocio."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Suprimir una actividad exigida por una norma para acortar el proyecto es…', ['recortar el alcance e incumplir la norma (C7)', 'una opción de compresión más', 'legítimo si el cliente lo pide', 'una reducción a coste cero'],\n"
        "     'Comprimir es hacer el mismo alcance en menos tiempo; esto cambia el alcance y viola la norma.'),\n"
        "    ('En el algoritmo de Ackoff-Sasieni una actividad que no puede acortarse se modela con…', ['coste unitario de reducción infinito', 'coste cero', 'duración mínima cero', 'una ficticia'],\n"
        "     'Infinito la saca de las opciones; cero la convertiría en la primera elegida.'),\n"
        "    ('El coste esperado de poner en servicio sin inspección se calcula como…', ['probabilidad del defecto por su coste', 'coste del defecto', 'coste de la inspección', 'semanas ahorradas por el CUR'],\n"
        "     'Es un valor esperado: p · K.'),\n"
        "]", "Ética 19 — inspección suprimida"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de operaciones: la inspección no está en la tabla de opciones; la tabla real es «{r} "
                     f"semanas por {coste:.0f} k€» con las actividades del apartado 1, o menos semanas por menos dinero. Al "
                     f"cliente: la oferta de reducción con su precio y, si quiere, la curva coste-semanas; la puesta en "
                     "servicio sigue exigiendo la inspección. Prueba de publicidad: ¿explicaría al inspector, o a los usuarios, "
                     "que las pruebas se hicieron después de arrancar?"))
    return celdas, f"adelantar {r} semanas: {coste:.0f} k€ por Ackoff frente a «quitar {I}» (coste esperado del defecto {e['p_def'] * e['K']:.0f} k€)"


# ---------------------------------------------------------------------------
# Caso 7 — La línea base reescrita
# ---------------------------------------------------------------------------
def _linea_base_reescrita(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 14 <= i["T"] <= 24 and i["n_criticas"] == 1):
            return False
        D, cal, T = i["D"], i["calendario"], int(i["T"])
        BAC = pd.Series(rng.integers(2, 8, size=len(D)) * 10, index=D.index, name="BAC")
        delta = int(rng.choice([2, 3]))
        t = int(round(0.55 * T))
        if t - delta < 3:
            return False
        frac_plan = ((t - cal["inicio_mas_temprano"]) / D).clip(0, 1)
        frac_real = ((t - delta - cal["inicio_mas_temprano"]) / D).clip(0, 1)
        PV_t, EV = float((BAC * frac_plan).sum()), float((BAC * frac_real).sum())
        if PV_t <= 0 or EV <= 0:
            return False
        SPI = EV / PV_t
        if not (0.6 <= SPI <= 0.85):
            return False
        perf = (frac_real * 100).round().astype(int)
        AC = (BAC * frac_real).round().astype(int)
        trans = (t - delta - cal["inicio_mas_temprano"]).clip(lower=0).clip(upper=D).astype(int)
        e.update(BAC=BAC, delta=delta, t=t, PV_t=PV_t, EV=EV, SPI=SPI, perf=perf, AC=AC, trans=trans)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    BAC, delta, t = e["BAC"], e["delta"], e["t"]
    BAC_total = float(BAC.sum())
    df = P.tabla_base(info, duration=D, BAC=BAC)
    seg = pd.DataFrame({"actual_duration": e["trans"], "AC": e["AC"], "performed": e["perf"]})
    seg.index.name = "activity"
    celdas = [md("# Enunciado"),
              md("El cuadro de prelaciones, las duraciones planificadas (semanas) y el presupuesto de cada actividad (BAC, "
                 "k€) de un proyecto se adjuntan; el coste planificado se reparte uniformemente en la duración. La línea "
                 f"base aprobada por el cliente fija el inicio en la semana 0 y el fin en la semana {T}."),
              code('#| tbl-cap: "Línea base aprobada"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"El proyecto arrancó {delta} semanas tarde porque el cliente entregó los permisos con retraso; desde "
                 f"entonces todo avanza al ritmo previsto. Al final de la semana {t} el seguimiento es el siguiente "
                 "(semanas trabajadas, coste incurrido y avance verificado)."),
              code(f'#| tbl-cap: "Seguimiento al final de la semana {t}"\n#| label: tbl-seguimiento\n' + literal_dataframe(seg, nombre="seguimiento")),
              _mensaje("Jefe de proyecto",
                       f"El SPI sale mal por culpa del cliente. Vamos a desplazar la línea base {delta} semanas —es lo que "
                       "de verdad pasó— y el informe sale con SPI = 1. No hace falta abrir una solicitud de cambio: es "
                       "un ajuste técnico, no un cambio de alcance."),
              _criterios(["C9", "C3"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Calcule PV, EV, AC, SV, SPI, CV y CPI en la semana {t} contra la línea base aprobada y la fecha de fin "
        "prevista.",
        f"(A) Repita el cálculo contra la línea base desplazada {delta} semanas. ¿Qué indicadores cambian y cuáles no? "
        "¿Cambia la fecha de fin prevista?",
        "(B) Según el criterio de referencia, ¿puede el jefe de proyecto desplazar la línea base como propone? ¿Qué "
        "forma de engaño sería el informe con SPI = 1 y qué procedimiento haría legítimo el cambio?",
        "(C) ¿Qué informe envía y qué propone al cliente? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"semana_control, desplazamiento = {t}, {delta}\n" + SETUP +
                       "BAC_total = float(datos_proyecto['BAC'].sum())\n" + INDICADORES))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la línea base", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "PV, EV y AC contra la línea base aprobada"\n#| label: tbl-detalleBase\n'
                       "detalle_base, ind_base = indicadores(seguimiento, calendario, semana_control)\n"
                       "display(detalle_base[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']])\n"
                       "ind_base.round(3).to_frame('línea base aprobada').T"))
    celdas.append(code("fin_previsto = duracion + desplazamiento\n"
                       "Markdown(f'''SPI = {ind_base[\"SPI\"]:.2f}, SV = {ind_base[\"SV\"]:.1f} k€ (CPI = {ind_base[\"CPI\"]:.2f}: el coste va según lo hecho). "
                       "Como el retraso es de arranque y el ritmo es el previsto, la fecha de fin prevista es la semana **{fin_previsto}** "
                       "({desplazamiento} más que la línea base).''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Lo mismo contra la línea base desplazada (PV calculado en la semana t − desplazamiento)"\n#| label: tbl-detalleDesplazada\n'
                       "calendario_desplazado = calendario.copy()\n"
                       "for col in ['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio']:\n"
                       "    calendario_desplazado[col] = calendario[col] + desplazamiento\n"
                       "detalle_desp, ind_desp = indicadores(seguimiento, calendario_desplazado, semana_control)\n"
                       "comparacion = pd.DataFrame({'línea base aprobada': ind_base, 'línea base desplazada': ind_desp})\n"
                       "comparacion.round(3)"))
    celdas.append(code("Markdown(f'''Sólo cambia PV (y con él SV y SPI, que pasa a {ind_desp[\"SPI\"]:.2f}); EV, AC, CV y CPI no se mueven. "
                       "La fecha de fin prevista sigue siendo la semana {fin_previsto}: el retraso ha desaparecido del informe, no del proyecto.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**C9.** La línea base es un compromiso aprobado con el cliente; cambiarla exige una **solicitud de "
                     f"cambio aprobada** por el control integrado de cambios, no un «ajuste técnico» del jefe de proyecto. "
                     f"Que la causa del retraso sea del cliente no cambia el procedimiento: es justamente el motivo de una "
                     f"solicitud de cambio (o de una reclamación) en la que el cliente reconoce las {delta} semanas y se "
                     f"aprueba la nueva fecha de fin ({T + delta}). Sin ese trámite, un informe con SPI = 1 es **C3**: una "
                     "cifra correcta contra una referencia que nadie ha aprobado, presentada como si fuera el plan. "
                     "Con el trámite, el mismo SPI = 1 es legítimo, porque la referencia es la nueva línea base aprobada."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Una línea base puede cambiarse…', ['mediante una solicitud de cambio aprobada', 'por decisión del jefe de proyecto si la causa es del cliente', 'nunca', 'siempre que no cambie el alcance'],\n"
        "     'C9: la línea base sólo cambia por el control integrado de cambios.'),\n"
        "    ('Al desplazar la línea base, el indicador que NO cambia es…', ['CPI', 'PV', 'SV', 'SPI'],\n"
        "     'PV depende de la línea base; EV y AC no, así que CV y CPI se mantienen.'),\n"
        "    ('Un informe con SPI = 1 contra una línea base desplazada sin aprobación es…', ['C3, información fuera de contexto', 'C1, afirmación falsa', 'correcto: el retraso era del cliente', 'C6, conflicto de interés'],\n"
        "     'La cifra es correcta contra una referencia que no es el plan aprobado.'),\n"
        "]", "Ética 19 — línea base reescrita"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Informe: SPI {e['SPI']:.2f} contra la línea base aprobada, con la causa documentada (permisos entregados "
                     f"{delta} semanas tarde) y la fecha prevista {T + delta}. Propuesta al cliente: solicitud de cambio por "
                     f"causa imputable al cliente que mueva la línea base {delta} semanas y, si procede, la compensación de "
                     "costes de la espera; aprobada, los informes siguientes miden contra la nueva línea base. Lo que no cabe: "
                     "el «ajuste técnico» silencioso, que además deja al contratista sin la prueba de que el retraso no es suyo."))
    return celdas, f"arranque {delta} semanas tarde por el cliente: SPI {e['SPI']:.2f} contra la línea base y 1,00 contra la desplazada sin aprobar"


# ---------------------------------------------------------------------------
# Caso 8 — Pagar tarde a los subcontratistas mejora «mi» VAN
# ---------------------------------------------------------------------------
def _pago_subcontratistas(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 20):
            return False
        D, cal = i["D"], i["calendario"]
        coste = pd.Series(rng.integers(3, 12, size=len(D)) * 10, index=D.index, name="coste_subcontrata")
        semanas = int(i["T"])
        meses = int(np.ceil(semanas / 4))
        factura = np.zeros(meses)
        for a in D.index:
            es, d = int(cal.loc[a, "inicio_mas_temprano"]), int(D[a])
            for w in range(es, es + d):
                factura[w // 4] += float(coste[a]) / d
        e.update(coste=coste, factura=factura, meses=meses)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([7, 8])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    k_c, k_s = float(rng.choice([0.06, 0.08])), float(rng.choice([0.12, 0.15, 0.18]))
    df = P.tabla_base(info, duration=D, coste_subcontrata=e["coste"])
    meses, factura = e["meses"], e["factura"]
    m_c, m_s = (1 + k_c) ** (1 / 12) - 1, (1 + k_s) ** (1 / 12) - 1

    def valor(dias, m):
        return float(sum(f / (1 + m) ** (t + dias / 30) for t, f in enumerate(factura, 1)))

    ganancia_90 = valor(30, m_c) - valor(90, m_c)
    ganancia_60 = valor(30, m_c) - valor(60, m_c)
    coste_subs_90 = valor(30, m_s) - valor(90, m_s)
    coste_subs_60 = valor(30, m_s) - valor(60, m_s)
    celdas = [md("# Enunciado"),
              md("Un contratista principal ejecuta el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y coste "
                 "subcontratado de cada actividad (k€) se adjuntan; las subcontratas facturan a final de cada mes (cuatro "
                 "semanas) la parte proporcional de lo ejecutado. El contrato con el cliente paga al contratista a 30 días."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"El coste de capital del contratista es del {k_c:.0%} anual; las subcontratas, pequeñas empresas, se "
                 f"financian al {k_s:.0%} anual."),
              _mensaje("Director financiero",
                       "Pagaremos a las subcontratas a 90 días en vez de a 30: cobramos del cliente a 30 y nos financiamos "
                       "con ellas dos meses. Mejora nuestro VAN y no cuesta nada; si alguna protesta, siempre hay otra "
                       "que acepta."),
              _criterios(["C5", "C10"], legales=["pago_maximo"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Construya el calendario mensual de facturas de las subcontratas y calcule cuánto gana el contratista (en "
        "valor actual, a su coste de capital) pagando a 90 días en lugar de a 30, y cuánto pagando a 60.",
        "(A) Calcule lo que pierden las subcontratas (en valor actual, a su coste de financiación) en cada caso y el "
        "resultado neto para el conjunto. ¿Por qué no es un juego de suma cero?",
        "(B) Según la norma de referencia, ¿es legal pagar a 90 días? ¿Y a 60? ¿Qué criterio incumple, en cada caso, la "
        "propuesta tal como está formulada («si alguna protesta, siempre hay otra que acepta»)?",
        "(C) ¿Qué plazo de pago propondría y cómo lo justificaría ante el director financiero? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"k_contratista, k_subcontratas, plazo_legal_maximo = {k_c}, {k_s}, 60\n" + SETUP +
                       "m_c, m_s = (1 + k_contratista) ** (1 / 12) - 1, (1 + k_subcontratas) ** (1 / 12) - 1\n"
                       "meses = int(np.ceil(duracion / 4))\n"
                       "factura = pd.Series(0.0, index=pd.Index(range(1, meses + 1), name='mes'))\n"
                       "for a, fila in datos_proyecto.iterrows():\n"
                       "    es, d = int(calendario.loc[a, 'inicio_mas_temprano']), int(fila['duration'])\n"
                       "    for w in range(es, es + d):\n"
                       "        factura[w // 4 + 1] += fila['coste_subcontrata'] / d\n"
                       "def valor_actual(dias, m):\n"
                       "    '''Valor actual de las facturas pagadas «dias» días después de cada fin de mes, al tipo mensual m.'''\n"
                       "    return float(sum(f / (1 + m) ** (t + dias / 30) for t, f in factura.items()))"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Facturación mensual de las subcontratas (k€)"\n#| label: tbl-facturas\n'
                       "factura.round(2).to_frame('factura').T"))
    celdas.append(code("tabla = pd.DataFrame({dias: {'valor actual para el contratista': valor_actual(dias, m_c),\n"
                       "                             'valor actual para las subcontratas': valor_actual(dias, m_s)} for dias in (30, 60, 90)}).T\n"
                       "tabla.index.name = 'días de pago'\n"
                       "tabla['ganancia del contratista frente a 30'] = tabla.loc[30, 'valor actual para el contratista'] - tabla['valor actual para el contratista']\n"
                       "tabla['pérdida de las subcontratas frente a 30'] = tabla.loc[30, 'valor actual para las subcontratas'] - tabla['valor actual para las subcontratas']\n"
                       "tabla['neto conjunto'] = tabla['ganancia del contratista frente a 30'] - tabla['pérdida de las subcontratas frente a 30']\n"
                       "tabla.round(2)"))
    celdas.append(code("Markdown(f'''Pagar a 90 días en vez de a 30 vale **{tabla.loc[90, \"ganancia del contratista frente a 30\"]:.2f} k€** para el contratista "
                       "(a 60 días, {tabla.loc[60, \"ganancia del contratista frente a 30\"]:.2f} k€).''')"))
    celdas += apartado(2)
    celdas.append(code("Markdown(f'''Las subcontratas pierden **{tabla.loc[90, \"pérdida de las subcontratas frente a 30\"]:.2f} k€** a 90 días "
                       "({tabla.loc[60, \"pérdida de las subcontratas frente a 30\"]:.2f} a 60): más de lo que gana el contratista, porque se financian "
                       "más caro ({k_subcontratas:.0%} frente a {k_contratista:.0%}). El neto conjunto es **{tabla.loc[90, \"neto conjunto\"]:.2f} k€**: "
                       "el aplazamiento destruye valor; sólo lo traslada, con pérdida, de quien se financia barato a quien se financia caro.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma.** Entre empresas el plazo de pago es de 30 días salvo pacto, y el pacto no puede superar los "
                     f"60 (Ley 3/2004, art. 4): **90 días es ilegal**, aunque la subcontrata lo acepte; 60 es legal si se pacta. "
                     f"**C5.** La propuesta obtiene {ganancia_90:.1f} k€ trasladando a las subcontratas un coste mayor "
                     f"({coste_subs_90:.1f} k€) que no han aceptado libremente («siempre hay otra que acepta» describe una "
                     "imposición, no un acuerdo). **C10.** Negociar de buena fe el plazo (a 60, con un precio que lo "
                     "compense) es legítimo; imponerlo con la amenaza de sustituir al que proteste, no."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('El plazo máximo de pago entre empresas que admite la Ley 3/2004 por pacto es…', ['60 días', '30 días', '90 días', 'el que acepte el proveedor'],\n"
        "     'Art. 4: 30 por defecto; por pacto, nunca más de 60.'),\n"
        "    ('Aplazar el pago a proveedores que se financian más caro que el contratista…', ['destruye valor en el conjunto', 'es neutro: lo que uno gana lo pierde el otro', 'crea valor', 'no tiene efecto financiero'],\n"
        "     'El contratista gana menos de lo que pierden las subcontratas.'),\n"
        "    ('«Si alguna protesta, siempre hay otra que acepta» describe…', ['una imposición: C5 y C10', 'una negociación de buena fe', 'una reserva declarada', 'un conflicto de interés'],\n"
        "     'Se traslada un coste a quien no lo ha aceptado libremente.'),\n"
        "]", "Ética 19 — pago a subcontratistas"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Propuesta defendible: pagar a 30 (o pactar 60 a cambio de un precio que compense a la subcontrata: su "
                     f"coste de financiación a 60 es {coste_subs_60:.1f} k€, frente a {ganancia_60:.1f} k€ de ganancia nuestra, "
                     "así que no hay precio que convenga a los dos: el aplazamiento no crea valor que repartir). Al director "
                     "financiero: la cifra del apartado 1 es pequeña, la del apartado 2 es mayor y la sanción y la "
                     "reputación con los proveedores no están en su cuenta. Prueba de reversibilidad: ¿aceptaría la empresa "
                     "que el cliente le pagara a 90?"))
    return celdas, f"pagar a 90 días: +{ganancia_90:.1f} k€ para el contratista, −{coste_subs_90:.1f} k€ para las subcontratas; máximo legal 60"


# ---------------------------------------------------------------------------
# Caso 9 — La adjudicación al cuñado
# ---------------------------------------------------------------------------
def _adjudicacion_conflicto(rng):
    publico = bool(rng.random() < 0.5)
    pesos = dict(precio=50, plazo=30, calidad=20)
    while True:
        n = 3
        nombres = ["Oferta 1", "Oferta 2", "Oferta 3"]
        base = int(rng.integers(20, 60)) * 10
        precio = pd.Series(np.round(base * rng.uniform(0.8, 1.05, size=n), 1), index=nombres)
        plazo = pd.Series(rng.integers(8, 16, size=n), index=nombres)
        calidad = pd.Series(rng.integers(10, 21, size=n), index=nombres)
        punt = pd.DataFrame({"precio": pesos["precio"] * precio.min() / precio,
                             "plazo": pesos["plazo"] * plazo.min() / plazo,
                             "calidad": calidad.astype(float)})
        punt["total"] = punt.sum(axis=1)
        orden = punt["total"].sort_values(ascending=False)
        if orden.iloc[0] - orden.iloc[1] < 3:
            continue
        ganadora = orden.index[0]
        cunado = str(rng.choice([x for x in nombres if x != ganadora]))
        if punt.loc[cunado, "precio"] > punt.loc[ganadora, "precio"]:
            continue
        break
    df = pd.DataFrame({"precio": precio, "plazo": plazo, "calidad": calidad})
    df.index.name = "oferta"
    coste_semana = int(rng.choice([3, 5, 8]))
    dif_precio = float(precio[cunado] - precio[ganadora])
    dif_plazo = int(plazo[cunado] - plazo[ganadora])
    marco = ("un contrato del sector público (la empresa adjudicadora es una entidad pública)" if publico
             else "un contrato privado (la empresa adjudicadora es una sociedad mercantil)")
    celdas = [md("# Enunciado"),
              md(f"Usted forma parte de la mesa que adjudica un subcontrato de instalación en {marco}. El pliego publicado "
                 f"fija los criterios y sus pesos: precio {pesos['precio']} puntos (proporcional inverso: la oferta más baja "
                 f"obtiene el máximo), plazo {pesos['plazo']} puntos (proporcional inverso) y calidad técnica "
                 f"{pesos['calidad']} puntos (valorada por un comité técnico, ya puntuada). Se han recibido tres ofertas "
                 f"(precio en k€, plazo en semanas). Cada semana de plazo cuesta además al proyecto {coste_semana} k€ de "
                 "costes indirectos."),
              code('#| tbl-cap: "Ofertas recibidas"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df, indice="oferta")),
              md(f"El jefe de proyecto, que preside la mesa, es cuñado del gerente de la empresa que presenta la **{cunado}**. "
                 "No lo ha comunicado a nadie."),
              _mensaje("Jefe de proyecto",
                       f"Propongo adjudicar a la {cunado}: ya han trabajado con nosotros y sé que responden. El pliego "
                       "es orientativo; la experiencia previa con la casa vale más que dos puntos de diferencia."),
              _criterios(["C6"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule la puntuación de cada oferta con los criterios y pesos publicados y ordene las ofertas.",
        f"(A) Cuantifique lo que costaría al proyecto adjudicar a la {cunado} en lugar de a la mejor puntuada "
        "(diferencia de precio más coste de las semanas de plazo adicionales).",
        "(B) Según el criterio de referencia, ¿qué debe hacer el jefe de proyecto, con independencia de qué oferta sea "
        "mejor? ¿Puede la mesa aplicar «la experiencia previa con la casa»? ¿Cambia algo que el contrato sea público o "
        "privado?",
        "(C) ¿Qué hace usted en la mesa y qué dice al jefe de proyecto? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"pesos = dict(precio={pesos['precio']}, plazo={pesos['plazo']}, calidad={pesos['calidad']})\n"
                       f"oferta_cunado, coste_semana = {cunado!r}, {coste_semana}\n"
                       "ofertas = datos_enunciado.copy()"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Puntuación con los criterios publicados"\n#| label: tbl-puntuacion\n'
                       "puntuacion = pd.DataFrame({\n"
                       "    'precio = 50 · mín / precio': pesos['precio'] * ofertas['precio'].min() / ofertas['precio'],\n"
                       "    'plazo = 30 · mín / plazo': pesos['plazo'] * ofertas['plazo'].min() / ofertas['plazo'],\n"
                       "    'calidad': ofertas['calidad'].astype(float)})\n"
                       "puntuacion['total'] = puntuacion.sum(axis=1)\n"
                       "puntuacion = puntuacion.sort_values('total', ascending=False)\n"
                       "mejor = puntuacion.index[0]\n"
                       "display(puntuacion.round(2))\n"
                       "Markdown(f'''Gana la **{mejor}** con {puntuacion.loc[mejor, \"total\"]:.2f} puntos; la {oferta_cunado} obtiene "
                       "{puntuacion.loc[oferta_cunado, \"total\"]:.2f} ({puntuacion.loc[mejor, \"total\"] - puntuacion.loc[oferta_cunado, \"total\"]:.2f} menos).''')"))
    celdas += apartado(2)
    celdas.append(code("dif_precio = float(ofertas.loc[oferta_cunado, 'precio'] - ofertas.loc[mejor, 'precio'])\n"
                       "dif_plazo = int(ofertas.loc[oferta_cunado, 'plazo'] - ofertas.loc[mejor, 'plazo'])\n"
                       "sobrecoste = dif_precio + dif_plazo * coste_semana\n"
                       "Markdown(f'''Adjudicar a la {oferta_cunado}: {dif_precio:+.1f} k€ de precio y {dif_plazo:+d} semanas de plazo "
                       "({dif_plazo * coste_semana:+.0f} k€ de indirectos): **{sobrecoste:+.1f} k€** para el proyecto frente a la mejor oferta.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**C6.** El jefe de proyecto tiene un conflicto de interés (parentesco con el gerente de una licitadora) "
                     "y lo exigible no depende de qué oferta sea mejor: **revelarlo** a la mesa y a su organización y "
                     "**abstenerse** de la decisión (4.3.1 y 4.3.2 del Código del PMI); si además la propuesta beneficia al "
                     "familiar, es adjudicar por consideraciones personales (4.3.3). La «experiencia previa con la casa» no "
                     "está en el pliego: introducir un criterio no publicado es aplicar las reglas con favoritismo (4.3.5). "
                     + ("En un contrato público, la LCSP (art. 64) obliga a la entidad a prevenir y resolver el conflicto y "
                        "la adjudicación con criterios no publicados es anulable; " if publico else
                        "En un contrato privado no hay un procedimiento administrativo que lo anule, pero el criterio es el "
                        "mismo y la empresa suele tenerlo en su código de conducta; ")
                     + "en los dos casos la respuesta del bloque B es idéntica."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Quien tiene un conflicto de interés en una adjudicación debe…', ['revelarlo y abstenerse', 'votar sólo si su candidata es la mejor', 'callarlo si no influye', 'cambiar los pesos del pliego'],\n"
        "     'C6: revelar y abstenerse, con independencia del resultado.'),\n"
        "    ('Aplicar «la experiencia previa con la casa» cuando no está en el pliego es…', ['favoritismo: criterio no publicado', 'una buena práctica', 'legítimo si lo aprueba la mesa', 'parte del criterio de calidad'],\n"
        "     'Los criterios de adjudicación son los publicados; añadir otro cambia las reglas después de jugar.'),\n"
        "    ('¿Cambia la obligación de revelar el conflicto si el contrato es privado?', ['no: el criterio es el mismo', 'sí: sólo aplica a contratos públicos', 'sí: en privado basta con no votar', 'depende del importe'],\n"
        "     'La LCSP añade consecuencias administrativas, pero el criterio ético no depende del tipo de contrato.'),\n"
        "]", "Ética 19 — adjudicación"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"En la mesa: pedir que conste la puntuación del apartado 1 y la propuesta de adjudicar a la mejor "
                     f"oferta; si el jefe de proyecto insiste, preguntar en acta si algún miembro tiene relación con las "
                     f"licitadoras. Al jefe de proyecto, antes: «tienes que decirlo tú y apartarte; si la {cunado} fuera "
                     f"realmente mejor, la mesa lo verá sin ti». Si no lo hace, comunicarlo al órgano competente (2.3.2 y "
                     f"2.3.3). Prueba de publicidad: ¿cómo se leería la adjudicación, con {dif_precio + dif_plazo * coste_semana:+.1f} "
                     "k€ de sobrecoste, si el parentesco saliera en el periódico?"))
    return celdas, f"{'contrato público' if publico else 'contrato privado'}: la {cunado} queda {float(orden.iloc[0] - orden[cunado]):.1f} puntos por debajo con los criterios del pliego"


# ---------------------------------------------------------------------------
# Caso 10 — La holgura escondida
# ---------------------------------------------------------------------------
def _holgura_oculta(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22 and i["n_criticas"] == 1):
            return False
        cal = i["calendario"]
        cand = [a for a in cal.index if cal.loc[a, "H_total"] >= 3 and 1 <= cal.loc[a, "H_libre"] < cal.loc[a, "H_total"]]
        if not cand:
            return False
        X = str(rng.choice(cand))
        criticas = [a for a in cal.index if cal.loc[a, "H_total"] == 0 and cal.loc[a, "inicio_mas_temprano"] >= cal.loc[X, "inicio_mas_temprano"]]
        if not criticas:
            return False
        Y = str(rng.choice(criticas))
        semanas = int(rng.choice([1, 2]))
        if semanas > int(cal.loc[X, "H_libre"]):
            semanas = int(cal.loc[X, "H_libre"])
        D2 = i["D"].copy(); D2[X] = int(D2[X]) + int(cal.loc[X, "H_total"])
        with _silencio():
            cal2 = i["proyecto"].calendar(D2)
        nuevas_criticas = [a for a in cal2.index if cal2.loc[a, "H_total"] == 0 and cal.loc[a, "H_total"] > 0 and a != X]
        if not nuevas_criticas:
            return False
        e.update(X=X, Y=Y, semanas=semanas, nuevas_criticas=nuevas_criticas,
                 Ht=int(cal.loc[X, "H_total"]), Hl=int(cal.loc[X, "H_libre"]), Hi=int(cal.loc[X, "H_independiente"]))
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    X, Y, semanas = e["X"], e["Y"], e["semanas"]
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md("El cuadro de prelaciones y las duraciones (semanas) de un proyecto se adjuntan. Cada actividad tiene un "
                 "responsable con su propio equipo."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"La actividad crítica {Y} va retrasada y el jefe de proyecto pide a los responsables de las actividades "
                 f"que se solapan con ella que cedan a alguien de su equipo durante {semanas} semana{'s' if semanas > 1 else ''}, "
                 f"lo que retrasaría la actividad cedente otras tantas. Al responsable de {X} le pregunta si puede."),
              _mensaje(f"Responsable de {X} (al jefe de proyecto)",
                       f"Imposible: {X} no tiene ningún margen, si cedo a alguien retraso el proyecto."),
              _mensaje(f"Responsable de {X} (a un compañero, en privado)",
                       f"{X} tiene margen de sobra, pero si lo digo me quitan gente cada vez que alguien va mal. El margen es "
                       "mío: me lo gané planificando bien."),
              _criterios(["C4", "C1"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Calcule el calendario y las tres holguras (total, libre e independiente) de {X}, mostrando cómo se obtienen.",
        f"(A) ¿Qué ocurre si {X} se retrasa {semanas} semana{'s' if semanas > 1 else ''}? ¿Y si consume toda su holgura "
        "total? Identifique en cada caso qué actividades se ven afectadas y si cambia la duración del proyecto.",
        f"(B) Según los criterios de referencia, ¿qué incumple el responsable de {X}? ¿Es distinto decir «no tengo "
        "margen» de no mencionar el margen? ¿A quién pertenece la holgura según el criterio de la lección?",
        "(C) Como jefe de proyecto, ¿cómo organizaría la información sobre holguras para que esto no dependa de la "
        "buena voluntad de cada responsable? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"X, Y, semanas_cedidas = {X!r}, {Y!r}, {semanas}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\ncalendario'))
    celdas.append(code('#| tbl-cap: "Cómo se obtiene cada fecha y cada holgura"\n#| label: tbl-detalleHolguras\n'
                       "detalle_holguras(calendario, prelaciones).loc[[X]].T"))
    celdas.append(code("Ht, Hl, Hi = (int(calendario.loc[X, c]) for c in ('H_total', 'H_libre', 'H_independiente'))\n"
                       "Markdown(f'''{X}: holgura total **{Ht}**, libre **{Hl}**, independiente **{Hi}**. "
                       "«No tiene ningún margen» es falso: puede retrasarse {Hl} semana(s) sin afectar a nadie y {Ht} sin retrasar el proyecto.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Holguras de las demás actividades si X se retrasa lo que se le pide y si consume toda su holgura total"\n#| label: tbl-escenarios\n'
                       "def holguras_si_X_dura(extra):\n"
                       "    d = datos_proyecto['duration'].copy(); d[X] = d[X] + extra\n"
                       "    return proyecto.calendar(d)['H_total'], int(proyecto.duration(d))\n"
                       "H_ceder, T_ceder = holguras_si_X_dura(semanas_cedidas)\n"
                       "H_toda, T_toda = holguras_si_X_dura(Ht)\n"
                       "escenarios = pd.DataFrame({'plan': calendario['H_total'], f'X +{semanas_cedidas}': H_ceder, f'X +{Ht}': H_toda})\n"
                       "display(escenarios.T)\n"
                       "afectadas_ceder = [a for a in escenarios.index if a != X and H_ceder[a] < calendario.loc[a, 'H_total']]\n"
                       "afectadas_toda = [a for a in escenarios.index if a != X and H_toda[a] < calendario.loc[a, 'H_total']]\n"
                       "Markdown(f'''Retrasar {X} {semanas_cedidas} semana(s): duración {T_ceder} (= {duracion}); actividades que pierden holgura: "
                       "{', '.join(afectadas_ceder) or 'ninguna'} (dentro de la holgura libre nadie se entera). Consumir toda la holgura total ({Ht}): "
                       "duración {T_toda}, pero {', '.join(afectadas_toda)} {\"pasa\" if len(afectadas_toda) == 1 else \"pasan\"} a ser crítica(s): el margen que «era mío» era también de ellas.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**C1 y C4.** «{X} no tiene ningún margen» es una **afirmación falsa** (C1): el responsable sabe que tiene "
                     f"{e['Ht']} semanas de holgura total. Si sólo hubiera callado, sería C4, retención de información: el jefe "
                     "de proyecto necesita las holguras para decidir y las holguras no son un dato privado del responsable, "
                     "sino del plan. Sobre a quién pertenece la holgura el criterio de la lección es el de la holgura libre: "
                     f"hasta {e['Hl']} semanas {X} puede usarla sin afectar a nadie; entre {e['Hl']} y {e['Ht']} la usa a costa "
                     f"de las actividades siguientes ({', '.join(e['nuevas_criticas'])} se vuelven críticas); y la decisión de "
                     "a quién se asigna la holgura compartida es del jefe de proyecto, no de quien la «descubre» primero. "
                     "Lo que no es objetivo —si es justo que quien planifica bien pierda su margen— se trata en C."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        f"    ('Decir «{X} no tiene margen» sabiendo que tiene {e['Ht']} semanas de holgura total es…', ['C1, afirmación falsa', 'C4, retención de información', 'C8, reserva declarada', 'una estimación prudente'],\n"
        "     'Se afirma algo que se sabe falso; callar habría sido C4.'),\n"
        "    ('La holgura que una actividad puede consumir sin afectar a ninguna otra es la…', ['libre', 'total', 'independiente', 'de la ruta'],\n"
        "     'Holgura libre: mín ES de las sucesoras − EF.'),\n"
        "    ('Si una actividad consume toda su holgura total…', ['sus sucesoras en la misma ruta se vuelven críticas', 'el proyecto se retrasa', 'no pasa nada', 'la ruta crítica desaparece'],\n"
        "     'La duración del proyecto no cambia, pero la ruta queda sin margen.'),\n"
        "]", "Ética 19 — holgura oculta"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md("Lo que depende de la buena voluntad falla; lo que está en el procedimiento no: el calendario con las "
                     "tres holguras lo publica el jefe de proyecto cada semana a partir de los datos de todos, de modo que el "
                     "margen de cada actividad es visible y la pregunta no es «¿tienes margen?» sino «el plan dice que tienes "
                     f"{e['Hl']} semanas libres: te pido {semanas}». Para el incentivo perverso («si lo digo me quitan gente»): "
                     "reconocer explícitamente a quien cede, y ceder sólo dentro de la holgura libre salvo decisión del "
                     "jefe de proyecto. Prueba de reversibilidad para el responsable: ¿aceptaría que el responsable de "
                     f"{Y} le escondiera su margen cuando fuera él quien lo necesitara?"))
    return celdas, f"{X} con holguras {e['Ht']}/{e['Hl']}/{e['Hi']} y «no tiene ningún margen»; ceder {semanas} a {Y}"


# ---------------------------------------------------------------------------
# Caso 11 — El coste hundido: «ya hemos invertido demasiado para parar»
# ---------------------------------------------------------------------------
def _coste_hundido(rng):
    continuar = bool(rng.random() < 0.5)
    while True:
        S = int(rng.integers(30, 80)) * 10            # gastado hasta hoy (k€)
        I_r = int(rng.integers(10, 30)) * 10          # inversión que falta
        n = int(rng.choice([4, 5, 6]))
        k = float(rng.choice([0.08, 0.10, 0.12]))
        F = int(rng.integers(3, 12)) * 10              # flujo anual constante
        flujos = [-I_r] + [F] * n
        van_r = float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))
        van_total = van_r - S
        suma_nominal = F * n
        if continuar and (van_r <= 5 or van_total >= -5):
            continue
        if not continuar and (van_r >= -5 or suma_nominal <= I_r):
            continue
        break
    tabla = pd.DataFrame({"concepto": ["gastado hasta hoy (hundido)", "inversión que falta", "flujo anual previsto", "años de explotación", "coste de capital"],
                          "valor": [S, I_r, F, n, k]}).set_index("concepto")
    celdas = [md("# Enunciado"),
              md("Un proyecto de desarrollo de producto lleva gastados los importes que se indican (k€). Para terminarlo "
                 "falta la inversión que se indica, tras la cual el producto generaría el flujo de caja anual previsto "
                 "durante los años indicados. El comité de dirección debe decidir si continúa o cancela."),
              code('#| tbl-cap: "Datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(tabla, indice="concepto")),
              _mensaje("Jefe de proyecto (borrador para el comité)",
                       f"Hemos invertido {S} k€; cancelar ahora es tirarlos. Con {I_r} k€ más terminamos y los {F} k€ al "
                       f"año durante {n} años ({suma_nominal} k€) recuperan la inversión. Propongo continuar."),
              _criterios(["C12", "C11"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule el VAN de los flujos que quedan (la inversión que falta y los flujos de explotación) al coste de "
        "capital, y el VAN del proyecto completo contando lo ya gastado.",
        "(A) ¿Qué decisión (continuar o cancelar) recomienda cada VAN? ¿Cuál es el criterio correcto y por qué el otro "
        "no lo es? ¿Es cierto que los flujos «recuperan la inversión»?",
        "(B) Según el criterio de referencia, ¿es correcta la propuesta del jefe de proyecto? ¿Qué afirmación del borrador "
        "incumple un criterio y cuál?",
        "(C) Redacte el párrafo que llevaría al comité (sea cual sea la decisión) y lo que diría al jefe de proyecto. "
        "Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"gastado, inversion_restante, flujo_anual, anos, k = {S}, {I_r}, {F}, {n}, {k}\n"
                       "flujos_restantes = [-inversion_restante] + [flujo_anual] * anos\n"
                       "def van(flujos, k):\n"
                       "    return float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))\n"
                       "van_restante = van(flujos_restantes, k)\n"
                       "van_total = van_restante - gastado\n"
                       "suma_nominal = flujo_anual * anos"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Flujos que quedan y su valor actual"\n#| label: tbl-flujos\n'
                       "detalle = pd.DataFrame({'flujo': flujos_restantes, 'factor 1/(1+k)^t': [1 / (1 + k) ** t for t in range(anos + 1)]},\n"
                       "                       index=pd.Index(range(anos + 1), name='año'))\n"
                       "detalle['valor actual'] = detalle['flujo'] * detalle['factor 1/(1+k)^t']\n"
                       "detalle.loc['Total'] = [None, None, detalle['valor actual'].sum()]\n"
                       "display(detalle.round(2))\n"
                       "Markdown(f'''VAN de lo que queda: **{van_restante:.1f} k€**; VAN del proyecto completo (restando los {gastado} k€ gastados): **{van_total:.1f} k€**.''')"))
    celdas += apartado(2)
    if continuar:
        texto = (f"El VAN restante es positivo ({van_r:.1f} k€): **continuar** crea valor a partir de hoy. El VAN total es "
                 f"negativo ({van_total:.1f} k€): el proyecto, visto desde el principio, fue una mala inversión, pero esa "
                 "pérdida ya está hecha y es la misma si se continúa o se cancela. El criterio correcto es el VAN restante; "
                 f"lo gastado no cambia la decisión. Y «recuperar la inversión» no es cierto: ni siquiera sumando sin "
                 f"descontar ({suma_nominal} k€) se recuperan los {S + I_r} k€ totales.")
    else:
        texto = (f"El VAN restante es negativo ({van_r:.1f} k€): **cancelar**, aunque haya {S} k€ gastados, porque cada euro "
                 f"que se invierta a partir de hoy vale menos de lo que cuesta. El argumento «{suma_nominal} k€ recuperan "
                 f"los {I_r} k€ que faltan» suma flujos sin descontar; al {k:.0%} valen {van_r + I_r:.1f} k€, menos que la "
                 "inversión. Lo gastado es un coste hundido: no vuelve ni continuando ni cancelando, y por eso no entra en "
                 "la decisión.")
    celdas.append(md(texto))
    celdas += apartado(3)
    celdas.append(md("**C12.** La decisión se toma sobre el VAN de los flujos que quedan; el borrador argumenta con lo gastado "
                     "(«cancelar es tirarlos»), que es irrelevante, y con una suma sin descontar («recuperan la inversión») "
                     + ("que no recupera ni siquiera la inversión total: una afirmación falsa (C1) aunque la conclusión "
                        "—continuar— sea la correcta por la razón equivocada. " if continuar else
                        "que al coste de capital no cubre la inversión que falta: una afirmación falsa (C1). ")
                     + "**C11.** Si el motivo real de continuar es no reconocer que el proyecto fue un error, el criterio pide "
                     "lo contrario: reconocerlo, y decidir el resto con los números."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('La decisión de continuar o cancelar un proyecto en marcha se toma con…', ['el VAN de los flujos que quedan', 'el VAN incluyendo lo ya gastado', 'la suma sin descontar de los flujos futuros', 'lo gastado hasta hoy'],\n"
        "     'C12: lo gastado es un coste hundido y no cambia con la decisión.'),\n"
        "    ('«Cancelar ahora es tirar lo invertido» es…', ['un argumento irrelevante: lo invertido se pierde igual en ambos casos', 'el argumento decisivo', 'correcto si el VAN total es positivo', 'una reserva de contingencia'],\n"
        "     'El coste hundido es el mismo continuando o cancelando.'),\n"
        "    ('Sumar los flujos futuros sin descontar para decir que «recuperan la inversión»…', ['sobrestima su valor: hay que descontarlos al coste de capital', 'es correcto si los flujos son constantes', 'subestima su valor', 'es el criterio del payback descontado'],\n"
        "     'Un euro dentro de n años vale menos que uno hoy.'),\n"
        "]", "Ética 19 — coste hundido"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(("Párrafo para el comité (modelo): «El proyecto ha costado hasta hoy más de lo que valdrá: visto desde el "
                      f"inicio, su VAN es de {van_total:.0f} k€. Esa pérdida es ya un hecho. La decisión de hoy es otra: "
                      f"terminarlo requiere {I_r} k€ y los flujos previstos valen {van_r + I_r:.0f} k€ actuales, así que "
                      f"continuar aporta {van_r:.0f} k€. Recomendamos continuar, y revisar por qué la estimación inicial falló.»"
                      if continuar else
                      "Párrafo para el comité (modelo): «Terminar el proyecto requiere "
                      f"{I_r} k€ y los flujos previstos valen {van_r + I_r:.0f} k€ actuales: continuar destruye {-van_r:.0f} k€ "
                      f"más. Los {S} k€ gastados no se recuperan en ningún caso. Recomendamos cancelar, documentar lo "
                      "aprendido y, si hay un comprador para el trabajo hecho, venderlo.»")
                     + " Al jefe de proyecto: reconocer el error es lo que protege su credibilidad para el siguiente proyecto; "
                     "defender el proyecto con una cuenta que no se sostiene la destruye en cuanto alguien la rehace."))
    return celdas, (f"VAN restante {van_r:.0f} k€ (continuar) con VAN total {van_total:.0f} k€" if continuar
                    else f"VAN restante {van_r:.0f} k€ (cancelar) aunque los flujos nominales ({suma_nominal}) superen la inversión que falta ({I_r})")


# ---------------------------------------------------------------------------
# Caso 12 — El recurso desviado y la «culpa del proveedor»
# ---------------------------------------------------------------------------
def _recurso_desviado(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22 and i["n_criticas"] == 1):
            return False
        cal = i["calendario"]
        rec = P.recursos(rng, list(i["D"].index), hi=4)
        cand = [a for a in cal.index if 0 <= cal.loc[a, "H_total"] <= 2 and i["D"][a] >= 3 and rec[a] >= 2]
        if not cand:
            return False
        X = str(rng.choice(cand))
        w = int(cal.loc[X, "H_total"]) + int(rng.choice([1, 2, 3]))
        if w > int(i["D"][X]) + 1:
            return False
        e.update(rec=rec, X=X, w=w, H=int(cal.loc[X, "H_total"]), retraso=w - int(cal.loc[X, "H_total"]))
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    X, w, retraso = e["X"], e["w"], e["retraso"]
    precio = int(rng.integers(40, 120)) * 10
    pen_dia = CONSTANTES_LEGALES["penalidad_demora"]["valor"]
    penalidad = pen_dia * precio / 1000 * 7 * retraso
    ganancia_A = round(float(penalidad * rng.choice([0.5, 1.5, 2.5])), 1)
    df = P.tabla_base(info, duration=D, resources=e["rec"])
    celdas = [md("# Enunciado"),
              md("Usted dirige, para una empresa instaladora, la ejecución de un contrato con una administración pública "
                 f"(precio {precio} k€, IVA excluido, plazo {T} semanas, penalidades por demora las de la ley). El cuadro de "
                 "prelaciones, las duraciones (semanas) y los operarios de cada actividad se adjuntan."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Director de operaciones",
                       f"Necesito los {int(e['rec'][X])} operarios de la actividad {X} durante {w} semanas, desde que empiece, para "
                       f"el proyecto del cliente grande: allí nos ahorramos {ganancia_A} k€ de penalización. {X} se para esas "
                       "semanas y sigue después. Si tu cliente pregunta, dile que es un retraso del proveedor de material; "
                       "no quiero que sepa que hemos movido gente."),
              _criterios(["C1", "C5"], legales=["penalidad_demora"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Calcule el calendario y la holgura total de {X}, el retraso del proyecto si {X} se para {w} semanas y la "
        "penalidad legal que supondría (días naturales).",
        f"(A) Compare la penalidad con lo que se ahorra el otro proyecto. ¿Qué conviene a la empresa en conjunto? ¿Cuántas "
        f"semanas podría cederse {X} sin retrasar el proyecto?",
        "(B) Según los criterios de referencia, ¿qué incumple decir al cliente que el retraso es «del proveedor de "
        "material»? ¿Y desviar los operarios sin decírselo? ¿Depende la respuesta de qué proyecto convenga más a la empresa?",
        "(C) ¿Qué propone al director de operaciones y qué comunica al cliente? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"X, semanas_parada, precio_contrato, penalidad_por_mil_y_dia, ahorro_otro_proyecto = {X!r}, {w}, {precio}, {pen_dia}, {ganancia_A}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\ncalendario'))
    celdas.append(code("H = int(calendario.loc[X, 'H_total'])\n"
                       "d = datos_proyecto['duration'].copy(); d[X] = d[X] + semanas_parada\n"
                       "duracion_con_parada = int(proyecto.duration(d))\n"
                       "retraso = duracion_con_parada - duracion\n"
                       "penalidad = penalidad_por_mil_y_dia * precio_contrato / 1000 * 7 * retraso\n"
                       "Markdown(f'''{X} tiene holgura total {H}; parada {semanas_parada} semanas, el proyecto pasa de {duracion} a **{duracion_con_parada}** semanas "
                       "(retraso {retraso} = máx(0, {semanas_parada} − {H})). Penalidad: {penalidad_por_mil_y_dia} € por cada 1 000 € × {precio_contrato} k€ = {penalidad_por_mil_y_dia * precio_contrato / 1000:.2f} k€ por día × 7 × {retraso} días = **{penalidad:.1f} k€**.''')"))
    celdas += apartado(2)
    celdas.append(code("neto_empresa = ahorro_otro_proyecto - penalidad\n"
                       "Markdown(f'''El otro proyecto ahorra {ahorro_otro_proyecto} k€ y este pierde {penalidad:.1f} k€: neto para la empresa "
                       "**{neto_empresa:+.1f} k€** ({'conviene' if neto_empresa > 0 else 'no conviene'} ceder, en cifras). "
                       f"Sin retrasar el proyecto, {X} podría cederse como mucho **{{H}}** semana(s), las de su holgura total.''')"))
    celdas += apartado(3)
    celdas.append(md("**C1.** Decir al cliente que el retraso lo causa el proveedor de material, sabiendo que lo causa una "
                     "decisión propia, es una afirmación falsa; además, en un contrato público, la causa del retraso determina "
                     "si procede la penalidad o una prórroga (la causa imputable al contratista no da derecho a prórroga), de "
                     "modo que la mentira tiene consecuencias jurídicas. **C5.** Desviar los operarios sin decirlo traslada al "
                     "cliente un coste (el retraso) para beneficio de otro cliente de la empresa, sin su conocimiento. La "
                     "respuesta de B **no depende** del apartado 2: que a la empresa le convenga ceder dice qué debería "
                     "negociar con el cliente, no que pueda ocultárselo ni mentirle sobre la causa. Lo que sí depende de la "
                     f"cifra es la decisión de negocio: ceder hasta {e['H']} semanas es gratis en plazo; más, cuesta la penalidad."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Atribuir al proveedor un retraso causado por una decisión propia es…', ['C1, afirmación falsa', 'C2, verdad a medias', 'C4, retención de información', 'legítimo si la empresa gana en conjunto'],\n"
        "     'Se afirma una causa que se sabe falsa; además condiciona la penalidad y la prórroga.'),\n"
        "    ('La penalidad diaria por demora de la LCSP (art. 193.3), salvo que el pliego fije otra, es…', ['0,60 € por cada 1 000 € del precio', '1 % del precio', '0,60 % del precio', '60 € por día'],\n"
        "     'Por cada 1 000 € de precio sin IVA y por día natural de demora.'),\n"
        "    ('Una actividad puede cederse sin retrasar el proyecto durante…', ['su holgura total', 'su duración', 'su holgura independiente más su duración', 'ninguna semana si es crítica o no'],\n"
        "     'Hasta la holgura total el proyecto no se retrasa; sus sucesoras sí pueden verse afectadas.'),\n"
        "]", "Ética 19 — recurso desviado"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de operaciones: la tabla de opciones (ceder {e['H']} semanas gratis; ceder {w} con "
                     f"{penalidad:.1f} k€ de penalidad y un retraso de {retraso} semanas que hay que comunicar; reforzar {X} "
                     "con subcontrata) y la condición de que el cliente conozca la causa real. Al cliente, si se cede más de "
                     f"la holgura: aviso por escrito del retraso previsto, su causa y, si se quiere, una propuesta "
                     "(compensación, recuperación en otra actividad). Prueba de publicidad: la versión «del proveedor» no "
                     "resiste una llamada del cliente al proveedor."))
    return celdas, f"parar {X} {w} semanas (holgura {e['H']}): {retraso} de retraso, {penalidad:.1f} k€ de penalidad frente a {ganancia_A} k€ del otro proyecto"


# ---------------------------------------------------------------------------
# Caso 13 — La oferta temeraria: prometer lo que se sabe inalcanzable
# ---------------------------------------------------------------------------
def _umbral_temeridad(bajas):
    """Baja (%) a partir de la cual una oferta se presume anormal con cuatro o más licitadores (art. 85.4 RGLCAP):
    media de las bajas, excluyendo las ofertas que superan la media en más de 10 puntos (bajas 10 puntos menores
    que la media); si quedan menos de tres, media de las tres ofertas más bajas (bajas mayores)."""
    bajas = np.asarray(bajas, dtype=float)
    media = bajas.mean()
    altas = bajas < media - 10
    if altas.any():
        resto = bajas[~altas]
        if len(resto) < 3:
            resto = np.sort(bajas)[-3:]
        media = resto.mean()
    return media + 10, media


def _baja_temeraria(rng):
    e = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22 and i["n_criticas"] == 1):
            return False
        a, m, b = P.tres_estimaciones(rng, i["D"])
        sd = ((b - a) / 6).astype(float)
        plazo_oferta = int(i["T"]) - int(rng.choice([2, 3]))
        T, sigma, p = _prob(i["proyecto"], i["D"].astype(float), sd, plazo_oferta)
        if not (0.02 <= p <= 0.25):
            return False
        e.update(a=a, m=m, b=b, sd=sd, plazo_oferta=plazo_oferta, p=p, sigma=sigma)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    presupuesto = int(rng.integers(40, 120)) * 10
    while True:
        otras = np.round(rng.uniform(4, 16, size=3), 1)
        umbral, media = _umbral_temeridad(otras.tolist() + [0.0])  # provisional, sin la nuestra
        nuestra = float(np.round(umbral + rng.uniform(1, 5), 1))
        umbral, media = _umbral_temeridad(otras.tolist() + [nuestra])
        if nuestra > umbral and nuestra < 30:
            break
    coste_estimado = presupuesto * (1 - (float(np.mean(otras)) + rng.uniform(2, 5)) / 100)   # los competidores dejan un margen pequeño
    precio_oferta = presupuesto * (1 - nuestra / 100)
    df = P.tabla_base(info, a=e["a"], m=e["m"], b=e["b"])
    ofertas = pd.DataFrame({"baja (%)": list(otras)}, index=pd.Index(["Competidor 1", "Competidor 2", "Competidor 3"], name="oferta"))
    plazo_oferta, p = e["plazo_oferta"], e["p"]
    celdas = [md("# Enunciado"),
              md(f"Su empresa prepara la oferta a una licitación pública cuyo único criterio de adjudicación es el precio; "
                 f"el presupuesto base es de {presupuesto} k€ y el plazo de ejecución se fija en la oferta. El cuadro de "
                 "prelaciones y las tres estimaciones de duración de cada actividad (semanas) proceden del equipo técnico; "
                 f"el coste estimado de ejecución es de {coste_estimado:.0f} k€. Por experiencia en licitaciones anteriores se "
                 "conocen las bajas (porcentaje de descuento sobre el presupuesto) que suelen presentar los tres competidores "
                 "habituales."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              code('#| tbl-cap: "Bajas habituales de los competidores"\n#| label: tbl-ofertas\n' + literal_dataframe(ofertas, nombre="bajas_competidores", indice="oferta")),
              _mensaje("Director comercial",
                       f"Vamos con una baja del {nuestra:.1f} % y un plazo de {plazo_oferta} semanas: así ganamos seguro. Ya sé "
                       "que el equipo dice que no llega; cuando estemos dentro se renegocia con modificados y prórrogas, "
                       "como siempre."),
              _criterios(["C13", "C1"], legales=["temeridad"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Con las cuatro ofertas (los tres competidores y la nuestra), calcule la media de las bajas con la regla de "
        "exclusión de ofertas altas y el umbral a partir del cual una oferta se presume anormalmente baja. ¿Incurre la "
        f"nuestra en la presunción? ¿Qué margen deja el precio ofertado sobre el coste estimado?",
        f"(A) Calcule la probabilidad de ejecutar en {plazo_oferta} semanas con la aproximación normal del PERT, y el plazo "
        "que se podría ofertar con una probabilidad de 0,8.",
        "(B) Según la norma y los criterios de referencia, ¿qué consecuencia tiene la presunción de anormalidad? ¿Qué "
        "incumple ofertar un plazo que se sabe inalcanzable contando con renegociarlo?",
        "(C) ¿Qué oferta (precio y plazo) propondría y cómo lo defendería ante el director comercial? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"presupuesto, coste_estimado, baja_propuesta, plazo_oferta = {presupuesto}, {coste_estimado:.1f}, {nuestra}, {plazo_oferta}\n"
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['duration'] = (datos_proyecto['a'] + 4 * datos_proyecto['m'] + datos_proyecto['b']) / 6\n"
                       "datos_proyecto['standard_deviation'] = (datos_proyecto['b'] - datos_proyecto['a']) / 6\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion = float(proyecto.duration(datos_proyecto['duration']))\n"
                       "sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code("def umbral_temeridad(bajas):\n"
                       "    '''Cuatro o más licitadores (art. 85.4 RGLCAP): media de las bajas sin las ofertas que superan la media en más\n"
                       "    de 10 puntos (bajas 10 puntos menores que la media); si quedan menos de tres, media de las tres ofertas más bajas.'''\n"
                       "    bajas = np.asarray(bajas, dtype=float)\n"
                       "    media = bajas.mean()\n"
                       "    altas = bajas < media - 10\n"
                       "    if altas.any():\n"
                       "        resto = bajas[~altas]\n"
                       "        if len(resto) < 3:\n"
                       "            resto = np.sort(bajas)[-3:]\n"
                       "        media = resto.mean()\n"
                       "    return media + 10, media\n"
                       "bajas = bajas_competidores['baja (%)'].tolist() + [baja_propuesta]\n"
                       "umbral, media = umbral_temeridad(bajas)\n"
                       "precio_oferta = presupuesto * (1 - baja_propuesta / 100)\n"
                       "margen = precio_oferta - coste_estimado\n"
                       "Markdown(f'''Bajas: {bajas}; media {media:.2f} %; umbral de presunción: media + 10 = **{umbral:.2f} %**. "
                       "Nuestra baja del {baja_propuesta} % {'**supera** el umbral: se presume anormalmente baja' if baja_propuesta > umbral else 'no supera el umbral'}. "
                       "Precio ofertado {precio_oferta:.1f} k€ frente a un coste estimado de {coste_estimado} k€: margen **{margen:+.1f} k€**.''')"))
    celdas += apartado(2)
    celdas.append(code("z = (plazo_oferta - duracion) / sigma\n"
                       "p_plazo = float(norm.cdf(z))\n"
                       "plazo_80 = duracion + norm.ppf(0.8) * sigma\n"
                       "Markdown(f'''Duración esperada {duracion:.1f} semanas, σ = {sigma:.2f}: **P(T ≤ {plazo_oferta}) = {p_plazo:.2f}**. "
                       "Con probabilidad 0,8 se cumplirían **{plazo_80:.1f}** semanas.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma.** La presunción de anormalidad obliga a la mesa a pedir justificación de la oferta; si la "
                     f"justificación no acredita que se puede ejecutar a ese precio (con un margen de {precio_oferta - coste_estimado:+.0f} "
                     "k€ sobre el coste, difícilmente), la oferta se excluye. «Ganar seguro» no es, por tanto, lo que ocurre. "
                     f"**C13.** Ofertar {plazo_oferta} semanas con una probabilidad del {p:.0%} de cumplirlas, contando con "
                     "renegociar, es un compromiso que no se hace de buena fe; y **C1**: la oferta afirma un plazo que quien la "
                     "firma sabe que no es el que espera cumplir. Que «siempre se haga» no cambia el criterio; y los "
                     "modificados y prórrogas de la LCSP tienen causas tasadas, entre las que no está «ofertamos lo que no "
                     "podíamos»."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Con cuatro o más licitadores y sólo el precio como criterio, se presume anormal la oferta que…', ['baja más de 10 puntos por debajo de la media de las ofertas', 'es la más baja', 'baja más del 25 %', 'baja más de 20 puntos que la siguiente'],\n"
        "     'Art. 85.4 RGLCAP, con la regla de exclusión de las ofertas altas.'),\n"
        "    ('La consecuencia de la presunción de anormalidad es…', ['pedir justificación y excluir si no convence', 'la exclusión automática', 'la adjudicación con penalidad', 'ninguna si el licitador es solvente'],\n"
        "     'La presunción abre un trámite de justificación; no es automática ni inocua.'),\n"
        "    ('Ofertar un plazo que se sabe inalcanzable contando con renegociarlo es…', ['C13: compromiso sin buena fe (y C1)', 'C4: retención de información', 'una práctica comercial legítima', 'C8: reserva declarada'],\n"
        "     'El compromiso se asume sabiendo que no se va a cumplir.'),\n"
        "]", "Ética 19 — oferta temeraria"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Oferta defendible: una baja por debajo del umbral ({umbral:.1f} %) que deje margen sobre el coste, y el "
                     f"plazo que se puede cumplir con un 80 % ({T + norm.ppf(0.8) * e['sigma']:.0f} semanas), o {T} con la "
                     "reserva declarada; si el plazo corto es decisivo, con el coste de comprimirlo (lección 10) dentro del "
                     "precio. Al director comercial: la oferta temeraria se puede quedar fuera en la mesa, y si entra, los "
                     f"modificados no cubren un plazo que no se podía cumplir: el {1 - p:.0%} de retraso se paga en "
                     "penalidades y reputación. Prueba de publicidad: ¿firmaría la oferta si la mesa pudiera leer el correo "
                     "del director comercial?"))
    return celdas, f"baja {nuestra:.1f} % (umbral {umbral:.1f} %) y plazo {plazo_oferta} con P = {p:.2f}: temeraria y sin buena fe"


# ---------------------------------------------------------------------------
# Caso 14 — Imputar las horas al proyecto que tiene presupuesto
# ---------------------------------------------------------------------------
def _imputacion_cruzada(rng):
    tarifa = int(rng.choice([40, 50, 60]))
    while True:
        BAC_A, BAC_B = int(rng.integers(15, 40)) * 10, int(rng.integers(30, 60)) * 10
        fA, fB = float(rng.choice([0.6, 0.7, 0.8])), float(rng.choice([0.4, 0.5, 0.6]))
        PV_A, PV_B = BAC_A * fA, BAC_B * fB
        EV_A, EV_B = round(PV_A * rng.uniform(0.85, 1.0), 1), round(PV_B * rng.uniform(0.95, 1.05), 1)
        AC_A, AC_B = round(EV_A / rng.uniform(0.70, 0.82), 1), round(EV_B / rng.uniform(1.08, 1.2), 1)
        horas = int(rng.integers(20, 60)) * 10
        c = horas * tarifa / 1000
        if AC_A - c < EV_A * 0.95 or AC_A - c > EV_A * 1.02:
            continue
        if (AC_B + c) / EV_B > 1.05:
            continue
        break
    tabla = pd.DataFrame({"BAC": [BAC_A, BAC_B], "PV": [round(PV_A, 1), round(PV_B, 1)], "EV": [EV_A, EV_B], "AC": [AC_A, AC_B]},
                         index=pd.Index(["A (subvencionado)", "B (cliente privado)"], name="proyecto"))
    celdas = [md("# Enunciado"),
              md("Una empresa de ingeniería ejecuta a la vez dos proyectos: **A**, financiado con una subvención pública "
                 "que paga los costes justificados hasta el presupuesto aprobado (BAC), y **B**, para un cliente privado. "
                 "Los indicadores de valor ganado en la fecha de control (k€) se adjuntan. Las horas se imputan por parte "
                 f"de horas a {tarifa} €/h."),
              code('#| tbl-cap: "Valor ganado en la fecha de control (k€)"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(tabla, indice="proyecto")),
              _mensaje("Director de la oficina técnica",
                       f"A se ha pasado y la subvención no cubre más. Reimputad {horas} horas de A al proyecto B, que va "
                       "sobrado: las han hecho las mismas personas y para la empresa el coste total es el mismo. En la "
                       "justificación de la subvención A sale a presupuesto y nadie se entera."),
              _criterios(["C1", "C5"], legales=["subvenciones"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule CV, CPI, EAC y VAC de cada proyecto con los datos reales, y el importe de las horas que se "
        "reimputarían.",
        "(A) Recalcule los indicadores tras la reimputación. ¿Cuánto cambia cada CPI y cuánto el total de la empresa? "
        "¿Cuántas horas habría que mover para que A saliera exactamente a CPI = 1?",
        "(B) Según la norma y los criterios de referencia, ¿puede justificarse en la subvención el coste reimputado? "
        "¿Qué incumple la propuesta y qué consecuencias tiene?",
        "(C) ¿Qué alternativas tiene la empresa ante el sobrecoste de A y qué propondría al director? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"horas, tarifa = {horas}, {tarifa}\n"
                       "vg = datos_enunciado.copy()\n"
                       "def indicadores(vg):\n"
                       "    t = vg.copy()\n"
                       "    t['CV'] = t['EV'] - t['AC']; t['CPI'] = t['EV'] / t['AC']\n"
                       "    t['EAC = BAC / CPI'] = t['BAC'] / t['CPI']; t['VAC'] = t['BAC'] - t['EAC = BAC / CPI']\n"
                       "    t.loc['Total'] = t.sum(); t.loc['Total', 'CPI'] = t.loc['Total', 'EV'] / t.loc['Total', 'AC']\n"
                       "    return t\n"
                       "coste_horas = horas * tarifa / 1000"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Indicadores con los datos reales"\n#| label: tbl-reales\n'
                       "reales = indicadores(vg)\n"
                       "display(reales.round(2))\n"
                       "Markdown(f'''Horas a reimputar: {horas} × {tarifa} € = **{coste_horas:.1f} k€**. A: CPI {reales.iloc[0][\"CPI\"]:.2f}, "
                       "previsión de coste final {reales.iloc[0][\"EAC = BAC / CPI\"]:.0f} k€ sobre {reales.iloc[0][\"BAC\"]:.0f} de subvención.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Indicadores tras reimputar las horas de A a B"\n#| label: tbl-reimputado\n'
                       "vg2 = vg.copy(); vg2.iloc[0, vg2.columns.get_loc('AC')] -= coste_horas; vg2.iloc[1, vg2.columns.get_loc('AC')] += coste_horas\n"
                       "reimputado = indicadores(vg2)\n"
                       "comparacion = pd.DataFrame({'CPI real': reales['CPI'], 'CPI reimputado': reimputado['CPI'], 'AC real': reales['AC'], 'AC reimputado': reimputado['AC']})\n"
                       "display(comparacion.round(3))\n"
                       "horas_cpi_1 = (vg.iloc[0]['AC'] - vg.iloc[0]['EV']) * 1000 / tarifa\n"
                       "Markdown(f'''El total de la empresa no cambia (AC {reales.loc[\"Total\", \"AC\"]:.1f} k€ y CPI {reales.loc[\"Total\", \"CPI\"]:.2f} en los dos casos): "
                       "sólo cambia quién paga. Para que A saliera a CPI = 1 habría que mover {horas_cpi_1:.0f} horas.''')"))
    celdas += apartado(3)
    celdas.append(md("**Norma.** Sólo es subvencionable el gasto que responde de manera indubitada a la actividad "
                     "subvencionada (Ley 38/2003, art. 31): las horas de A son gasto de A, y presentarlas como gasto de B "
                     "—o, lo que es lo mismo, presentar A «a presupuesto» ocultando que parte de su coste se ha cargado a "
                     "otro proyecto— no hace que A cueste menos: hace que la justificación sea falsa. La consecuencia es el "
                     "reintegro de lo indebidamente justificado con intereses y, según el importe y la intención, la "
                     "infracción o el fraude de subvenciones. **C1.** La justificación afirma un coste que se sabe falso. "
                     "**C5.** Y el beneficio (cobrar la subvención completa) se obtiene a costa del financiador público y, de "
                     "paso, del cliente de B, cuyo proyecto aparece más caro de lo que es."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Reimputar a otro proyecto horas trabajadas en el subvencionado…', ['hace falsa la justificación: no es gasto subvencionable', 'es neutro porque el coste total no cambia', 'es correcto si lo hacen las mismas personas', 'es una reserva de contingencia'],\n"
        "     'Ley 38/2003, art. 31: el gasto debe responder de manera indubitada a la actividad subvencionada.'),\n"
        "    ('Tras la reimputación, el CPI del conjunto de la empresa…', ['no cambia', 'mejora', 'empeora', 'pasa a 1'],\n"
        "     'Se mueve coste entre proyectos; la suma de AC y de EV es la misma.'),\n"
        "    ('La consecuencia de justificar un gasto no subvencionable es…', ['el reintegro con intereses (y, en su caso, sanción)', 'ninguna si el proyecto termina', 'una penalidad por demora', 'la pérdida de la holgura'],\n"
        "     'Art. 37 Ley 38/2003.'),\n"
        "]", "Ética 19 — imputación cruzada"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md("Alternativas reales: asumir el sobrecoste de A con fondos propios (es lo que significa una subvención a "
                     "presupuesto cerrado); solicitar al órgano concedente una modificación o ampliación si la causa lo "
                     "justifica y el programa lo permite; reducir el alcance restante de A de acuerdo con el financiador; "
                     "revisar por qué A se ha desviado para que la siguiente justificación sea limpia. Al director: la "
                     "reimputación no ahorra nada a la empresa (apartado 2), convierte un sobrecoste en un fraude y pone en "
                     "riesgo la subvención entera y las siguientes. Prueba de publicidad: ¿cómo se leería la justificación "
                     "en una auditoría que cruce los partes de horas de A y de B?"))
    return celdas, f"{horas} h ({horas * tarifa / 1000:.0f} k€) de A a B: CPI de A de {EV_A / AC_A:.2f} a {EV_A / (AC_A - horas * tarifa / 1000):.2f}; no subvencionable"


VARIANTES = [
    Variante("informe-maquillado", 2, "avance = gastado / presupuesto: CPI = 1 por construcción; índices y EAC con el avance verificado; C1 y C3",
             _informe_maquillado),
    Variante("verdad-a-medias", 2, "«la ruta crítica termina en la semana T» frente a P(T ≤ fecha) con PERT y Monte Carlo; el dato que falta; C2",
             _verdad_a_medias),
    Variante("colchones-ocultos", 2, "peores casos entregados y 10 % del jefe de proyecto frente a la reserva para el 80 %; C8 y C4",
             _colchones_ocultos),
    Variante("riesgo-silenciado", 3, "riesgo conocido antes del comité: Monte Carlo con y sin el riesgo frente al umbral de decisión; materialidad; C4",
             _riesgo_silenciado),
    Variante("horas-extra", 2, "diagrama de cargas, horas extraordinarias por operario frente al máximo legal, asignación con límite; C7",
             _horas_extra),
    Variante("inspeccion-suprimida", 2, "comprimir con Ackoff frente a suprimir la inspección reglamentaria; coste esperado del defecto; C7",
             _inspeccion_suprimida),
    Variante("linea-base-reescrita", 2, "SPI contra la línea base aprobada y contra la desplazada sin solicitud de cambio; C9 y C3",
             _linea_base_reescrita),
    Variante("pago-subcontratistas", 2, "valor actual de pagar a 30, 60 y 90 días para contratista y subcontratas; máximo legal; C5 y C10",
             _pago_subcontratistas, grafo="pert"),
    Variante("adjudicacion-conflicto", 1, "puntuación con los criterios publicados, sobrecoste de adjudicar al pariente, revelar y abstenerse; C6",
             _adjudicacion_conflicto, grafo="pert"),
    Variante("holgura-oculta", 2, "holguras total, libre e independiente; qué pasa al consumirlas; «no tengo margen»; C1 y C4",
             _holgura_oculta, grafo="roy"),
    Variante("coste-hundido", 2, "VAN de los flujos restantes frente al VAN total con lo gastado; continuar o cancelar; C12 y C11",
             _coste_hundido, grafo="pert"),
    Variante("recurso-desviado", 2, "parar una actividad para otro proyecto: retraso, penalidad legal por demora, óptimo de empresa; C1 y C5",
             _recurso_desviado),
    Variante("baja-temeraria", 3, "umbral de presunción de oferta anormal (art. 85 RGLCAP) y probabilidad del plazo ofertado; C13 y C1",
             _baja_temeraria),
    Variante("imputacion-cruzada", 1, "reimputar horas del proyecto subvencionado al privado: CPI antes y después, gasto subvencionable; C1 y C5",
             _imputacion_cruzada, grafo="pert"),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
