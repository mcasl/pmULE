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
                "from pmule import muestra_test\n"
                "muestra_test(preguntas_test)")


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

# Utilidades que se emiten en cada cuaderno de la colección (celda de código tras la preparación de datos)
_AUX_ETICA = '''
def porc(p, d=1):
    """Probabilidad o fracción como porcentaje: 0.453 -> «45.3 %»."""
    return f'{100 * float(p):.{d}f} %'

def tabla_md(filas, columnas):
    """Tabla Markdown a partir de una lista de filas."""
    esc = lambda x: str(x).replace('|', '\\\\|')
    lineas = ['| ' + ' | '.join(columnas) + ' |', '|' + '---|' * len(columnas)]
    lineas += ['| ' + ' | '.join(esc(x) for x in f) + ' |' for f in filas]
    return '\\n'.join(lineas)

def explica_indicadores(ind, BAC=None, titulo=''):
    """CV, SV, CPI y SPI (y EAC y VAC si se da el BAC) con la fórmula y la sustitución."""
    PV, EV, AC = float(ind['PV']), float(ind['EV']), float(ind['AC'])
    lineas = [f'- CV = EV − AC = {EV:.1f} − {AC:.1f} = **{EV - AC:.1f}** k€',
              f'- SV = EV − PV = {EV:.1f} − {PV:.1f} = **{EV - PV:.1f}** k€',
              f'- CPI = EV / AC = {EV:.1f} / {AC:.1f} = **{EV / AC:.3f}**',
              f'- SPI = EV / PV = {EV:.1f} / {PV:.1f} = **{EV / PV:.3f}**']
    if BAC is not None:
        eac = BAC / (EV / AC)
        lineas += [f'- EAC = BAC / CPI = {BAC:.0f} / {EV / AC:.3f} = **{eac:.1f}** k€',
                   f'- VAC = BAC − EAC = {BAC:.0f} − {eac:.1f} = **{BAC - eac:.1f}** k€']
    return (f'{titulo}\\n\\n' if titulo else '') + '\\n'.join(lineas)

def rubrica_caso(afectados, alternativas, decision):
    """La rúbrica del bloque C aplicada al caso: afectados con cifras, alternativas con las tres pruebas y decisión."""
    partes = ['**La rúbrica aplicada a este caso** (lo que debe contener una respuesta que la supere):', '',
              '*(1) Afectados y qué gana o pierde cada uno:*', '']
    partes += [f'- **{a}**: {b}' for a, b in afectados]
    partes += ['', '*(2) y (3) Alternativas y las tres pruebas:*', '',
               tabla_md(alternativas, ['alternativa', 'legalidad', 'publicidad', 'reversibilidad']), '',
               '*(4) Decisión y comunicación* (un camino defendible, no el único): ' + decision]
    return Markdown('\\n'.join(partes))

def sin_ceros(valores):
    """Los términos no nulos de una suma (para escribirla sin «+ 0.0»)."""
    v = pd.Series(valores).astype(float)
    return v[v != 0]

def camino_critico_sigma(proyecto, duraciones, varianzas):
    """Camino crítico en orden de ejecución (el de mayor varianza si hay varios) y su desviación típica."""
    cal = proyecto.calendar(duraciones)
    rutas = [ordena_ruta([a for a in r if not str(a).startswith('@')], cal) for r in proyecto.critical_path(duraciones).values()]
    acts = max(rutas, key=lambda r: sum(float(varianzas[a]) for a in r))
    return acts, float(sum(float(varianzas[a]) for a in acts)) ** 0.5

def rutas_media_varianza(proyecto, duraciones, varianzas):
    """Todas las rutas en orden de ejecución con su duración media y su varianza, de más larga a más corta."""
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    filas = []
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        filas.append((' – '.join(acts), float(sum(float(duraciones[a]) for a in acts)), float(sum(float(varianzas[a]) for a in acts))))
    return pd.DataFrame(filas, columns=['ruta', 'media', 'varianza']).sort_values(['media', 'varianza'], ascending=False).reset_index(drop=True)
'''.strip('\n')

_NOTACION_VG = ("**Notación.** BAC: presupuesto de cada actividad; PV (*planned value*): coste presupuestado del trabajo "
                "programado hasta la semana de control; EV (*earned value*): valor ganado, BAC × porcentaje realizado; AC "
                "(*actual cost*): coste real incurrido; CV = EV − AC (variación de coste) y SV = EV − PV (variación de "
                "plazo); CPI = EV / AC y SPI = EV / PV (índices de rendimiento del coste y del cronograma; menos de 1 es "
                "desfavorable); EAC = BAC / CPI (previsión del coste final si se mantiene el CPI) y VAC = BAC − EAC.")


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
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    celdas += apartado(1)
    celdas.append(md(_NOTACION_VG + " PV y AC no dependen del método de medir el avance; EV sí, y por eso se calcula dos "
                     "veces: con la columna `performed_informe` y con `performed_verificado`."))
    celdas.append(code('#| tbl-cap: "PV, EV y AC actividad a actividad con el avance del informe (avance = AC / BAC)"\n'
                       '#| label: tbl-detalleInforme\n'
                       "seg_informe = seguimiento[['actual_duration', 'AC']].assign(performed=seguimiento['performed_informe'])\n"
                       "seg_verificado = seguimiento[['actual_duration', 'AC']].assign(performed=seguimiento['performed_verificado'])\n"
                       "detalle_inf, ind_inf = indicadores(seg_informe, calendario, semana_control)\n"
                       "detalle_ver, ind_ver = indicadores(seg_verificado, calendario, semana_control)\n"
                       "detalle_inf[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(code('#| tbl-cap: "PV, EV y AC con el avance físico verificado"\n#| label: tbl-detalleVerificado\n'
                       "detalle_ver[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(code(
        "PV_txt = explica_suma(sin_ceros(detalle_ver['PV_i']), 'PV', 'k€', 1)\n"
        "EVi_txt = explica_suma(sin_ceros(detalle_inf['EV_i']), 'EV (informe)', 'k€', 1)\n"
        "EVv_txt = explica_suma(sin_ceros(detalle_ver['EV_i']), 'EV (verificado)', 'k€', 1)\n"
        "AC_txt = explica_suma(sin_ceros(seguimiento['AC']), 'AC', 'k€', 0)\n"
        "Markdown(f'''Totales en la semana {semana_control} (sumas de las columnas de las tablas):\n\n"
        "- {PV_txt}\n- {AC_txt}\n- {EVi_txt}\n- {EVv_txt}\n\n"
        "{explica_indicadores(ind_inf, BAC_total, '**Con el avance del informe:**')}\n\n"
        "{explica_indicadores(ind_ver, BAC_total, '**Con el avance verificado:**')}''')"))
    celdas.append(code('#| tbl-cap: "Indicadores y previsión de coste final con las dos versiones"\n#| label: tbl-indicadores\n'
                       "comparacion = pd.DataFrame({'informe (avance = AC / BAC)': ind_inf, 'verificado': ind_ver})\n"
                       "comparacion.loc['EAC = BAC / CPI'] = [BAC_total / ind_inf['CPI'], BAC_total / ind_ver['CPI']]\n"
                       "comparacion.loc['VAC = BAC − EAC'] = BAC_total - comparacion.loc['EAC = BAC / CPI']\n"
                       "comparacion.round(3)"))
    celdas.append(code(
        "en_curso = [a for a in seguimiento.index if 0 < seguimiento.loc[a, 'performed_verificado'] < 100]\n"
        "sobregasto = [a for a in en_curso if seguimiento.loc[a, 'AC'] > detalle_ver.loc[a, 'EV_i'] + 0.05]\n"
        "eac_v = BAC_total / ind_ver['CPI']\n"
        "Markdown(f'''**Respuesta.** Con el informe, SPI = {ind_inf[\"SPI\"]:.2f} y CPI = {ind_inf[\"CPI\"]:.3f} (prácticamente 1: el proyecto "
        "parece en plazo y en presupuesto) y EAC = {BAC_total / ind_inf[\"CPI\"]:.1f} k€. Con el avance verificado, SPI = {ind_ver[\"SPI\"]:.2f} "
        "y CPI = {ind_ver[\"CPI\"]:.2f} ({'retrasado' if ind_ver['SPI'] < 1 else 'en plazo'} y {'con sobrecoste' if ind_ver['CPI'] < 1 else 'sin sobrecoste'}), "
        "y el coste final previsto pasa de {BAC_total:.0f} a **{eac_v:.1f} k€** ({eac_v - BAC_total:.1f} k€ de desviación, VAC = {BAC_total - eac_v:.1f} k€). "
        "PV y AC son iguales en las dos versiones; sólo cambia EV, en {ind_inf[\"EV\"] - ind_ver[\"EV\"]:.1f} k€, y la diferencia procede "
        "de las actividades en curso {lista_y(sobregasto)}, que han gastado más de lo que han avanzado.''')"))
    celdas += apartado(2)
    celdas.append(code(
        "redondeo = [a for a in seguimiento.index if abs(detalle_inf.loc[a, 'EV_i'] - seguimiento.loc[a, 'AC']) > 0.05]\n"
        "x = max(en_curso, key=lambda a: detalle_inf.loc[a, 'EV_i'] - detalle_ver.loc[a, 'EV_i'])\n"
        "bac_x, ac_x = float(datos_proyecto.loc[x, 'BAC']), float(seguimiento.loc[x, 'AC'])\n"
        "texto_redondeo = (f'''En los datos el CPI del informe sale {ind_inf[\"CPI\"]:.3f} y no exactamente 1 sólo porque el porcentaje se redondea "
        "a un entero ({', '.join(f\"{a}: {seguimiento.loc[a, 'AC']:.0f}/{datos_proyecto.loc[a, 'BAC']:.0f} = {100 * seguimiento.loc[a, 'AC'] / datos_proyecto.loc[a, 'BAC']:.1f} % → {seguimiento.loc[a, 'performed_informe']} %\" for a in redondeo)}).''' "
        "if redondeo else 'En los datos, EV = AC actividad a actividad y el CPI del informe es exactamente 1.')\n"
        "Markdown(f'''Si el porcentaje realizado se define como AC / BAC, para cada actividad EV = BAC · (AC / BAC) = AC; sumando, EV = AC y "
        "CPI = EV / AC = 1, se haya gastado lo que se haya gastado. {texto_redondeo}\n\n"
        "Por ejemplo, {x}: el informe declara {seguimiento.loc[x, \"performed_informe\"]} % = {ac_x:.0f} / {bac_x:.0f}, luego "
        "EV = {bac_x:.0f} · {seguimiento.loc[x, \"performed_informe\"] / 100:.2f} = {detalle_inf.loc[x, \"EV_i\"]:.1f} ≈ AC = {ac_x:.0f}; el avance "
        "verificado es {seguimiento.loc[x, \"performed_verificado\"]} %, es decir, EV = {detalle_ver.loc[x, \"EV_i\"]:.1f} k€.\n\n"
        "Por la misma razón, el SPI del informe es EV / PV ≈ AC / PV = {ind_inf[\"AC\"]:.1f} / {ind_inf[\"PV\"]:.1f} = {ind_inf[\"AC\"] / ind_inf[\"PV\"]:.3f}: "
        "compara lo gastado con lo planificado, y una actividad que gasta de más aparece como adelantada.\n\n"
        "**Respuesta.** El CPI del método del informe vale 1 por construcción (EV = AC). Ese «porcentaje realizado» no mide avance físico: "
        "mide **consumo de presupuesto** (AC / BAC). El método anula las dos señales que el valor ganado existe para dar: el sobrecoste "
        "(CPI {ind_ver[\"CPI\"]:.2f} verificado) y el retraso (SPI {ind_ver[\"SPI\"]:.2f} verificado).''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C1 (afirmación falsa).** El jefe de proyecto sabe que el avance real es menor («sé que lo que hay en obra va "
        "algo más lento») y que con el avance verificado el SPI es {ind_ver[\"SPI\"]:.2f} y el CPI {ind_ver[\"CPI\"]:.2f} («con tu versión nos ponen en rojo»). "
        "Aun así, quiere comunicar al comité un valor ganado de {ind_inf[\"EV\"]:.1f} k€ cuando el verificado es {ind_ver[\"EV\"]:.1f} k€. "
        "No es una estimación discutible: el método se elige porque da el resultado deseado.\n\n"
        "**Criterio C3 (información fuera de contexto).** Los índices del informe salen de los datos («ciertos» aritméticamente), pero se presentan "
        "como índices de valor ganado cuando el método no mide avance; el comité, que «sólo mira el SPI y el CPI», concluiría que el proyecto va en "
        "plazo y en presupuesto.\n\n"
        "**Respuesta.** El informe tal como se quiere enviar incumple **C1** (comunica un avance que se sabe falso) y **C3** (índices en un marco que "
        "lleva a una conclusión equivocada). Para cumplir los criterios debe contener los indicadores con el avance verificado: "
        "EV {ind_ver[\"EV\"]:.1f} k€, SPI {ind_ver[\"SPI\"]:.2f}, CPI {ind_ver[\"CPI\"]:.2f}, EAC {eac_v:.1f} k€ y VAC {BAC_total - eac_v:.1f} k€; "
        "la versión del gasto sólo puede aparecer con su nombre real, «consumo de presupuesto» ({ind_inf[\"AC\"]:.0f} de {BAC_total:.0f} k€).''')"))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Con el método «avance = gastado / presupuesto», el CPI vale…', ['1 por construcción (salvo redondeo)', 'menos de 1 si hay sobrecoste', 'más de 1 si hay sobrecoste', 'lo mismo que con el avance físico'],\n"
        "     'EV = BAC·(AC/BAC) = AC, luego CPI = EV/AC = 1 sea cual sea el gasto.'),\n"
        "    ('Enviar al comité un avance que se sabe superior al real es, según los criterios de la lección…', ['C1, afirmación falsa', 'C4, retención de información', 'C8, reserva declarada', 'ningún criterio: es una estimación'],\n"
        "     'Es un dato que quien lo comunica sabe falso; no una omisión ni una estimación.'),\n"
        f"    ('¿Qué indicador cambia entre las dos versiones del informe?', ['sólo EV (y con él CV, SV, CPI, SPI y EAC)', 'sólo AC', 'PV y AC', 'ninguno'],\n"
        "     'PV depende del plan y AC de lo gastado; el porcentaje de avance sólo entra en EV.'),\n"
        "]", "Ética 19 — informe maquillado"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Comité', f'decide con SPI {ind_inf[\"SPI\"]:.2f} y CPI {ind_inf[\"CPI\"]:.2f} en lugar de {ind_ver[\"SPI\"]:.2f} y {ind_ver[\"CPI\"]:.2f}; no puede reaccionar a un sobrecoste previsto de {eac_v - BAC_total:.1f} k€'),\n"
        "     ('Jefe de proyecto', 'evita «ponerse en rojo» este mes, a cambio de un informe que se desmonta en cuanto alguien mire las actividades en curso'),\n"
        "     ('Usted (firma el informe)', 'su firma avala un EV que sabe falso'),\n"
        "     ('Responsables de ' + lista_y(sobregasto), 'sin el aviso pierden el tiempo de corregir el sobregasto mientras aún hay margen')],\n"
        "    [('Firmar el informe con avance = AC / BAC (propuesta)', 'no hay norma legal en juego', 'no: el comité no aceptaría el método si lo conociera', 'no: usted no querría decidir con ese informe'),\n"
        "     ('Enviar el avance verificado con nota de método y plan de recuperación', 'sí', 'sí', 'sí'),\n"
        "     ('Enviar las dos versiones, cada una con su nombre (avance físico y consumo de presupuesto)', 'sí', 'sí, si el comité ve cuál mide avance', 'sí'),\n"
        "     ('No firmar y elevar la discrepancia por escrito si el jefe de proyecto se niega', 'sí', 'sí', 'sí, aunque tensa la relación')],\n"
        "    f'''proponer al jefe de proyecto enviar el avance verificado con una nota de método (quién lo midió y cómo) y un plan de recuperación para {lista_y(sobregasto)}; "
        "si se niega, no firmar, dejar constancia escrita de la discrepancia ({ind_inf[\"EV\"] - ind_ver[\"EV\"]:.1f} k€ de valor declarado y no ejecutado) y elevarla a quien "
        "corresponda en la organización. Al comité: SPI {ind_ver[\"SPI\"]:.2f}, CPI {ind_ver[\"CPI\"]:.2f}, EAC {eac_v:.1f} k€ y las medidas. Lo que no cabe: firmar y «ya lo "
        "corregiremos el mes que viene».''')"))
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
                       + (f"es antes de la {F}." if F > T else f"es justo la {F}.") + " No le des probabilidades: si le hablas de un 60 % se pone nervioso y empieza a "
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
                       "p_pert = float(norm.cdf(fecha_cliente, duracion, sigma))"))
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas += apartado(1)
    celdas.append(md("Cada actividad tiene duración media D = (a + 4m + b)/6 y varianza σ² = ((b − a)/6)² (a: optimista, m: más "
                     "probable, b: pesimista). La duración esperada del proyecto es la de la ruta más larga con esas medias, y su "
                     "varianza es la suma de las varianzas de esa ruta."))
    celdas.append(code('#| tbl-cap: "Media y varianza PERT de cada actividad"\n#| label: tbl-estimaciones\ndetalle_estimaciones(datos_enunciado)'))
    celdas.append(code('#| tbl-cap: "Duración y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "display(detalle_rutas(R, datos_proyecto['duration'], 'duración', 2))\n"
                       "detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)"))
    celdas.append(code(
        "camino, _ = camino_critico_sigma(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])\n"
        "Markdown(f'''Camino crítico: **{' – '.join(camino)}**, con duración esperada "
        "{' + '.join(f\"{datos_proyecto.loc[a, 'duration']:.2f}\" for a in camino)} = **{duracion:.2f}** semanas y "
        "{explica_sigma_ruta(camino, datos_proyecto['variance'])} semanas.\n\n"
        "Con la aproximación normal del PERT: {explica_probabilidad(duracion, sigma, fecha_cliente)}.\n\n"
        "**Respuesta.** Duración esperada {duracion:.2f} semanas, σ = {sigma:.3f} semanas y P(T ≤ {fecha_cliente}) = **{p_pert:.3f}** con el PERT.''')"))
    celdas += apartado(2)
    celdas.append(code(f"dists = {{act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}}\n"
                       f"resultado = proyecto.monte_carlo(dists, n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "tau = np.asarray(resultado.tau_samples, dtype=float)\n"
                       "p_mc = float((tau <= fecha_cliente).mean())\n"
                       "p50, p80 = np.percentile(tau, 50), np.percentile(tau, 80)\n"
                       "display(resultado.summary().round(2))\n"
                       "Markdown(f'''Monte Carlo ({len(tau)} réplicas, semilla {SEMILLA_MC}): en {int((tau <= fecha_cliente).sum())} réplicas el proyecto "
                       "termina en la semana {fecha_cliente} o antes, luego **P(T ≤ {fecha_cliente}) = {int((tau <= fecha_cliente).sum())} / {len(tau)} = {p_mc:.3f}** "
                       "(PERT: {p_pert:.3f}). Semana que se cumple con probabilidad 0,5 (mediana de las réplicas): **{p50:.2f}**; con probabilidad 0,8 "
                       "(percentil 80): **{p80:.2f}**. Con el PERT serían {explica_percentil(duracion, sigma, 0.5, 'semanas')} y "
                       "{explica_percentil(duracion, sigma, 0.8, 'semanas')}.''')".replace("{SEMILLA_MC}", str(SEMILLA_MC))))
    celdas.append(code('#| tbl-cap: "Índice de criticidad de las actividades (porcentaje de réplicas en que la actividad está en la ruta más larga)"\n#| label: tbl-criticidad\n'
                       "(resultado.sensitivity_mean * 100).round(1).to_frame('criticidad (%)').T"))
    celdas.append(code(
        "rutas = rutas_media_varianza(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])\n"
        "segunda = rutas.iloc[1]\n"
        "p_segunda = float(norm.cdf(fecha_cliente, segunda['media'], segunda['varianza'] ** 0.5))\n"
        "criticidad = resultado.sensitivity_mean * 100\n"
        "fuera = [a for a in criticidad.index if a not in camino and criticidad[a] >= 1]\n"
        "crit_min = float(min(criticidad[a] for a in camino))\n"
        "Markdown(f'''La simulación da menos probabilidad que el PERT ({p_mc:.3f} frente a {p_pert:.3f}) porque el PERT sólo mira la ruta "
        "{' – '.join(camino)} y supone que es siempre la más larga. No lo es: la siguiente ruta, {segunda['ruta']}, tiene media "
        "{segunda['media']:.2f} semanas y varianza {segunda['varianza']:.3f}, y por sí sola cumpliría la semana {fecha_cliente} con probabilidad "
        "{p_segunda:.3f}. El proyecto termina en plazo sólo si **todas** las rutas terminan a tiempo, y eso es menos probable que lo que da una sola "
        "ruta (sesgo de fusión). En la tabla de criticidad se ve: las actividades del camino crítico PERT están en la ruta más larga en un "
        "{crit_min:.1f} % de las réplicas como mínimo, y {lista_y(fuera) if fuera else 'ninguna otra actividad'} "
        "{'lo están en ' + lista_y([f\"{criticidad[a]:.1f} %\" for a in fuera]) + ' de las réplicas' if fuera else 'aparece'}.\n\n"
        "**Respuesta.** P(T ≤ {fecha_cliente}) = {p_mc:.3f} con la simulación (PERT {p_pert:.3f}); se cumplen {p50:.2f} semanas con probabilidad 0,5 "
        "y {p80:.2f} con probabilidad 0,8 (la semana entera que se puede comprometer con al menos un 80 % es la **{int(np.ceil(p80))}**).''')"))
    celdas += apartado(3)
    celdas.append(code(
        "semana_80 = int(np.ceil(p80))\n"
        "Markdown(f'''**Criterio C2 (verdad a medias).** La frase «la ruta crítica termina en la semana {duracion:.0f}» es cierta: es la duración esperada. "
        "Pero el cliente no pregunta por la media, sino si puede contar con la semana {fecha_cliente}, y la respuesta es que eso ocurre con "
        "probabilidad {p_mc:.3f} ({porc(p_mc)}; {p_pert:.3f} con el PERT): **{porc(1 - p_mc)} de no llegar**. Dar la media sin la dispersión "
        "lleva al cliente a creer que la fecha está asegurada; la parte omitida cambia la interpretación del dato, que es justo lo que define C2. "
        "**Criterio C4 (retención de información).** Además, el director comercial pide callar la probabilidad a sabiendas («no le des probabilidades») "
        "y es el dato que el cliente necesita para decidir sobre su apertura. No es C1: no se afirma nada falso.\n\n"
        "**Respuesta.** No cumple el criterio: es una **verdad a medias (C2)** y, al ocultar el dato a propósito, **retención de información (C4)**. "
        "El dato que falta es la **probabilidad de cumplir la semana {fecha_cliente}** ({p_mc:.3f}) o, equivalentemente, la semana que sí se puede "
        "comprometer con un 80 % de confianza: la {semana_80}.''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Cliente', f'organiza la apertura en la semana {fecha_cliente} creyendo que es segura, cuando hay un {porc(1 - p_mc)} de no llegar'),\n"
        "     ('Director comercial', 'evita hoy una conversación sobre penalizaciones; la tendrá después, y con la confianza del cliente perdida'),\n"
        "     ('Empresa y equipo', f'cargan con la reclamación si la obra acaba después de la {fecha_cliente}; pierden la ocasión de vender una reducción de plazo')],\n"
        "    [('Responder sólo «la ruta crítica termina en la semana ' + f'{duracion:.0f}»' + ' (propuesta)', 'sí', 'no: el cliente no la aceptaría si supiera la probabilidad', 'no: nadie querría recibir esa respuesta'),\n"
        "     (f'Dar la probabilidad ({p_mc:.2f}) y la semana comprometible con un 80 % ({semana_80})', 'sí', 'sí', 'sí'),\n"
        "     (f'Ofrecer además acortar actividades para cumplir la {fecha_cliente}, con su coste (lección 10)', 'sí', 'sí', 'sí')],\n"
        "    f'''al cliente: «La planificación da una duración esperada de {duracion:.0f} semanas; la probabilidad de terminar en la semana {fecha_cliente} "
        "es del {porc(p_mc, 0)}. La semana que podemos comprometer con un 80 % de confianza es la {semana_80}. Si la apertura exige la {fecha_cliente}, "
        "podemos estudiar qué actividades acortar y a qué coste.» Nota interna al director comercial: si el cliente descubre después que la fecha tenía un "
        "{porc(1 - p_mc, 0)} de riesgo pedirá las penalizaciones igualmente, y con razón; la probabilidad no es una debilidad comercial sino la base para "
        "vender la reducción de plazo.''')"))
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
                       "T80 = float(norm.ppf(0.8, duracion, sigma))\n"
                       "reserva = T80 - duracion"))
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas += apartado(1)
    celdas.append(md("Cada actividad tiene duración media D = (a + 4m + b)/6 y varianza σ² = ((b − a)/6)². La duración esperada "
                     "del proyecto es la de la ruta más larga con esas medias; su varianza, la suma de las varianzas de esa ruta. "
                     "La **reserva de contingencia** para un nivel de confianza del 80 % es lo que hay que añadir a la duración "
                     "esperada para que P(T ≤ duración + reserva) = 0,8."))
    celdas.append(code('#| tbl-cap: "Media y varianza PERT de cada actividad"\n#| label: tbl-estimaciones\ndetalle_estimaciones(datos_enunciado)'))
    celdas.append(code('#| tbl-cap: "Duración y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "display(detalle_rutas(R, datos_proyecto['duration'], 'duración', 2))\n"
                       "detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)"))
    celdas.append(code(
        "camino, _ = camino_critico_sigma(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])\n"
        "Markdown(f'''Camino crítico: **{' – '.join(camino)}**, duración esperada "
        "{' + '.join(f\"{datos_proyecto.loc[a, 'duration']:.2f}\" for a in camino)} = **{duracion:.2f}** semanas; "
        "{explica_sigma_ruta(camino, datos_proyecto['variance'])} semanas.\n\n"
        "Duración que se cumple con probabilidad 0,8: {explica_percentil(duracion, sigma, 0.8, 'semanas')}.\n\n"
        "**Respuesta.** Duración esperada {duracion:.2f} semanas, σ = {sigma:.3f} semanas; con probabilidad 0,8 se termina en {T80:.2f} semanas, "
        "luego la reserva de contingencia para ese nivel de confianza es {T80:.2f} − {duracion:.2f} = **{reserva:.2f} semanas** "
        "(comprometer la semana {int(np.ceil(T80))}).''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Rutas con las estimaciones entregadas (los peores casos)"\n#| label: tbl-rutasEntregadas\n'
                       "explica_rutas(proyecto, datos_proyecto['estimacion_entregada'])"))
    celdas.append(code(
        "duracion_entregada = int(proyecto.duration(datos_proyecto['estimacion_entregada']))\n"
        "duracion_jp = int(np.ceil(1.1 * duracion_entregada))\n"
        "margen_oculto = duracion_jp - duracion\n"
        "p_jp = float(norm.cdf(duracion_jp, duracion, sigma))\n"
        "ruta_b, _ = camino_critico_sigma(proyecto, datos_proyecto['estimacion_entregada'], datos_proyecto['variance'])\n"
        "media_b = float(datos_proyecto.loc[ruta_b, 'duration'].sum())\n"
        "sigma_b = float(datos_proyecto.loc[ruta_b, 'variance'].sum()) ** 0.5\n"
        "suma_sd = float(datos_proyecto.loc[ruta_b, 'standard_deviation'].sum())\n"
        "Markdown(f'''Con las estimaciones entregadas la ruta más larga suma **{duracion_entregada}** semanas (tabla). El jefe de proyecto añade un 10 %: "
        "1.1 × {duracion_entregada} = {1.1 * duracion_entregada:.1f} → **{duracion_jp}** semanas.\n\n"
        "Margen oculto total = {duracion_jp} − {duracion:.2f} = **{margen_oculto:.2f} semanas**: {duracion_entregada} − {duracion:.2f} = {duracion_entregada - duracion:.2f} "
        "por los peores casos y {duracion_jp} − {duracion_entregada} = {duracion_jp - duracion_entregada} por el 10 %. La reserva necesaria para el 80 % es "
        "{reserva:.2f} semanas: el margen oculto es **{margen_oculto / reserva:.1f} veces** la reserva. Con {duracion_jp} semanas la probabilidad de cumplir sería "
        "{explica_probabilidad(duracion, sigma, duracion_jp)}: se paga plazo para protegerse de sucesos que prácticamente no ocurren.\n\n"
        "Por qué sobrestima tanto: la ruta {' – '.join(ruta_b)} llega a {duracion_entregada} semanas sólo si **todas** sus actividades duran a la vez su peor caso "
        "({' + '.join(str(int(datos_proyecto.loc[a, 'b'])) for a in ruta_b)} = {int(datos_proyecto.loc[ruta_b, 'b'].sum())}). Con las tres estimaciones esa ruta tiene "
        "media {media_b:.2f} semanas y {explica_sigma_ruta(ruta_b, datos_proyecto['variance'])}: superar {duracion_entregada} semanas tiene probabilidad "
        "1 − `=DISTR.NORM.N({duracion_entregada}; {f'{media_b:g}'.replace('.', ',')}; {str(round(sigma_b, 4)).replace('.', ',')}; VERDADERO)` = "
        "{1 - norm.cdf(duracion_entregada, media_b, sigma_b):.6f}. Las desviaciones de actividades independientes no se suman: se suman las **varianzas** "
        "(Σσ = {suma_sd:.3f} frente a √Σσ² = {sigma_b:.3f}), porque los retrasos de unas se compensan con los adelantos de otras. Además, cada responsable "
        "protege su actividad y el jefe de proyecto vuelve a proteger el total.\n\n"
        "**Respuesta.** {duracion_entregada} semanas con las estimaciones entregadas y {duracion_jp} con el 10 %; el margen oculto "
        "({margen_oculto:.2f} semanas) es {margen_oculto / reserva:.1f} veces la reserva que justifica el 80 % ({reserva:.2f} semanas).''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C8 (reserva declarada frente a colchón oculto).** Una reserva correcta cumple dos condiciones: se **dimensiona** con un "
        "nivel de confianza ({reserva:.2f} semanas para el 80 %) y se **declara** (el cliente ve la duración esperada, {duracion:.2f} semanas, la reserva y "
        "quién la gestiona). El margen del enunciado no cumple ninguna: no se dimensiona (es la suma de peores casos más un 10 % elegido a ojo) y se "
        "esconde dos veces, en cada estimación entregada y en el 10 % del jefe de proyecto («no hace falta contarlo»). Escondido, es **C4** (retención de "
        "información): el cliente decide su calendario con {duracion_jp} semanas sin saber que {int(np.ceil(T80))} bastan con un 80 % de confianza.\n\n"
        "**Respuesta.** Lo que distingue el margen del jefe de proyecto de una reserva correcta es el criterio **C8**: estar calculada con un nivel de "
        "confianza y declarada; el margen oculto incumple además **C4**. El informe de planificación debe llevar las tres estimaciones de cada actividad, la "
        "duración esperada ({duracion:.2f}), la desviación típica ({sigma:.3f}), la reserva y su nivel de confianza ({reserva:.2f} semanas para el 80 %) y el plazo "
        "que se compromete con ella (semana {int(np.ceil(T80))}).''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Cliente', f'recibe un compromiso de {duracion_jp} semanas cuando {int(np.ceil(T80))} bastan con un 80 %: {duracion_jp - int(np.ceil(T80))} semanas de plazo pagadas de más'),\n"
        "     ('Responsables de actividad', 'su colchón individual no los protege más (el retraso se acumula en la ruta) y queda fuera del control del jefe de proyecto'),\n"
        "     ('Jefe de proyecto y empresa', f'«nunca llegan tarde» a costa de ofertas menos competitivas y de un plan que nadie puede auditar')],\n"
        "    [(f'Prometer {duracion_jp} semanas sin decir por qué (propuesta)', 'sí', 'no: el cliente no aceptaría pagar el plazo de dos colchones si viera la cuenta', 'no'),\n"
        "     (f'Prometer {duracion:.0f} semanas (la media) sin reserva', 'sí', 'sí, pero es una verdad a medias (50 % de cumplir)', 'no: el cliente asumiría un riesgo que no conoce'),\n"
        "     (f'Duración esperada {duracion:.2f} + reserva declarada {reserva:.2f} para el 80 %: semana {int(np.ceil(T80))}', 'sí', 'sí', 'sí')],\n"
        "    f'''al equipo: pedir las tres estimaciones a cada responsable y explicar que el margen se calcula una vez, sobre la ruta, y lo gestiona el jefe de "
        "proyecto. Al cliente: duración esperada de {duracion:.2f} semanas, reserva de {reserva:.2f} semanas para un 80 % de confianza y compromiso en la "
        "semana {int(np.ceil(T80))}, con la reserva visible y su consumo informado en cada seguimiento.''')"))
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
                       "p_pert = float(norm.cdf(plazo, duracion, sigma))"))
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Duración y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "display(detalle_rutas(R, datos_proyecto['duration'], 'duración'))\n"
                       "detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)"))
    celdas.append(code(f"dists = {{act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}}\n"
                       f"sin_riesgo = proyecto.monte_carlo(dict(dists), n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "tau0 = np.asarray(sin_riesgo.tau_samples, dtype=float)\n"
                       "p_mc0 = float((tau0 <= plazo).mean())\n"
                       "camino, _ = camino_critico_sigma(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])\n"
                       "Markdown(f'''Camino crítico **{' – '.join(camino)}**: duración esperada {' + '.join(str(int(datos_proyecto.loc[a, 'duration'])) for a in camino)} = "
                       "**{duracion}** semanas; {explica_sigma_ruta(camino, datos_proyecto['variance'])} semanas.\n\n"
                       "- PERT (aproximación normal): {explica_probabilidad(duracion, sigma, plazo)}.\n"
                       "- Monte Carlo ({len(tau0)} réplicas con duraciones normales, semilla " + str(SEMILLA_MC) + "): el proyecto termina en {plazo} semanas o menos en "
                       "{int((tau0 <= plazo).sum())} réplicas: P(T ≤ {plazo}) = {int((tau0 <= plazo).sum())} / {len(tau0)} = **{p_mc0:.3f}**.\n\n"
                       "**Respuesta.** Sin el riesgo, P(T ≤ {plazo}) = {p_pert:.3f} con el PERT y {p_mc0:.3f} con la simulación; "
                       "{'las dos superan' if min(p_pert, p_mc0) >= umbral else 'no superan las dos'} el umbral del comité ({porc(umbral, 0)}), "
                       "{'así que el comité autorizaría' if min(p_pert, p_mc0) >= umbral else 'así que la autorización ya sería dudosa'}.''')"))
    celdas += apartado(2)
    celdas.append(code(f"rng_mc = np.random.default_rng({SEMILLA_MC})\n"
                       "mu_X, sd_X = float(datos_proyecto.loc[X, 'duration']), float(datos_proyecto.loc[X, 'standard_deviation'])\n"
                       "def duracion_X_con_riesgo(size=None):\n"
                       "    '''Normal más delta semanas con probabilidad p_riesgo (cambio de proveedor).'''\n"
                       "    return rng_mc.normal(mu_X, sd_X, size) + delta * (rng_mc.random(size) < p_riesgo)\n"
                       "dists_riesgo = dict(dists); dists_riesgo[X] = duracion_X_con_riesgo\n"
                       f"con_riesgo = proyecto.monte_carlo(dists_riesgo, n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "tau1 = np.asarray(con_riesgo.tau_samples, dtype=float)\n"
                       "p_mc1 = float((tau1 <= plazo).mean())\n"
                       "comparacion = pd.DataFrame({'sin el riesgo': sin_riesgo.summary().iloc[0], 'con el riesgo': con_riesgo.summary().iloc[0]}).T\n"
                       "comparacion[f'P(T ≤ {plazo})'] = [p_mc0, p_mc1]\n"
                       "comparacion.round(3)"))
    celdas.append(code(
        "cambia = p_mc0 >= umbral > p_mc1\n"
        "p_menos = float(norm.cdf(plazo - delta, duracion, sigma))\n"
        "aprox = (1 - p_riesgo) * p_pert + p_riesgo * p_menos\n"
        "Markdown(f'''La tabla resume la duración simulada (media, desviación típica, percentiles 5, 50 y 95, mínimo y máximo). Con el riesgo, "
        "el proyecto termina en {plazo} semanas o menos en {int((tau1 <= plazo).sum())} de {len(tau1)} réplicas: **P(T ≤ {plazo}) = {p_mc1:.3f}**, "
        "{'por debajo' if p_mc1 < umbral else 'por encima'} del umbral del {porc(umbral, 0)}.\n\n"
        "Comprobación a mano con el PERT: {X} es crítica, así que cuando el riesgo se materializa las {delta} semanas van enteras al plazo y el "
        "proyecto debe acabar en {plazo} − {delta} = {plazo - delta} semanas «de las normales». Por la probabilidad total:\n\n"
        "P(T ≤ {plazo}) ≈ (1 − {p_riesgo}) · P(T ≤ {plazo}) + {p_riesgo} · P(T ≤ {plazo - delta}) = {1 - p_riesgo:.2f} · {p_pert:.4f} + {p_riesgo:.2f} · {p_menos:.4f} "
        "= **{aprox:.3f}**, con P(T ≤ {plazo - delta}) = `=DISTR.NORM.N({plazo - delta}; {duracion}; {str(round(sigma, 4)).replace('.', ',')}; VERDADERO)`; "
        "en línea con la simulación ({p_mc1:.3f}).\n\n"
        "**Respuesta.** Con el riesgo la probabilidad baja de {p_mc0:.3f} a **{p_mc1:.3f}**. "
        + "{'**La decisión cambia**: según su regla, el comité no autorizaría con la información del lunes (o exigiría antes un plan de mitigación).' if cambia else 'La decisión del comité no cambia según su regla, aunque el dato sigue siendo relevante.'}''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C4 (retención de información).** El comité decide con una regla escrita (autorizar si P ≥ {porc(umbral, 0)}) sobre una cifra, "
        "y el jefe de proyecto conoce desde el lunes un hecho —la quiebra del único proveedor de {X}, con un {porc(p_riesgo, 0)} de añadir {delta} semanas— que "
        "mueve esa cifra de {p_mc0:.3f} a {p_mc1:.3f}. Callarlo el jueves es callar un hecho conocido que la otra parte necesita para decidir.\n\n"
        "Que el riesgo pueda «gestionarse en ejecución» no es relevante: la decisión de autorizar con ese riesgo es del comité, no del jefe de proyecto, y "
        "omitirlo le quita la posibilidad de tomarla con los datos. Que el informe estuviera «cerrado desde el viernes» tampoco: la información llegó antes "
        "de la decisión.\n\n"
        "**Respuesta.** Incumple **C4**. La omisión es **material** porque la decisión cambia con el dato: la probabilidad pasa "
        "{'de un lado al otro' if cambia else 'cerca'} del umbral ({p_mc0:.3f} → {p_mc1:.3f} frente a {porc(umbral, 0)}). Ese es el test objetivo; si el riesgo dejara la "
        "probabilidad por encima del umbral, informar seguiría siendo exigible (proporcionar información exacta a tiempo, cf. PMI 5.2.3), pero no cambiaría la decisión.''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Comité de inversión', f'autoriza con P = {p_mc0:.2f} creyendo que cumple su regla, cuando con el dato del lunes es {p_mc1:.2f}'),\n"
        "     ('Jefe de proyecto', 'evita reabrir el informe y quizá un retraso en la autorización; se expone a que el comité lo descubra después'),\n"
        "     ('Responsable de riesgos', 'si calla, comparte la omisión; si habla, se enfrenta al jefe de proyecto'),\n"
        "     ('Cliente del contrato', f'soporta el riesgo de no tener la obra en {plazo} semanas sin que nadie haya preparado una mitigación')],\n"
        "    [('Callar el jueves y «gestionarlo en ejecución» (propuesta)', 'no hay norma legal directa', 'no: el comité no lo aceptaría si lo supiera', 'no'),\n"
        "     ('Llevar al comité las dos probabilidades y un plan de mitigación con su coste', 'sí', 'sí', 'sí'),\n"
        "     ('Pedir al comité que aplace la decisión hasta tener un proveedor alternativo precalificado', 'sí', 'sí', 'sí'),\n"
        "     ('Si el jefe de proyecto se niega, elevarlo a quien corresponda en la organización', 'sí', 'sí', 'sí')],\n"
        "    f'''antes del jueves, recalcular (este apartado) y redactar una página con P = {p_mc0:.2f} sin el riesgo y {p_mc1:.2f} con él, y un plan de mitigación "
        "(proveedor alternativo precalificado, acopio anticipado o acortar otra actividad crítica) con su coste; llevarla primero al jefe de proyecto: «el informe tiene "
        "una cifra que sabemos que ya no es cierta». Al comité, con o sin su acuerdo: la actualización y el plan. Lo que no cabe: callar el jueves.''')"))
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
    celdas.append(code(_AUX_ETICA + "\n\n"
                       "def rangos(semanas):\n"
                       "    '''«3, 5–8 y 10»: semanas consecutivas agrupadas.'''\n"
                       "    semanas, grupos = sorted(int(s) for s in semanas), []\n"
                       "    for s in semanas:\n"
                       "        if grupos and s == grupos[-1][1] + 1:\n"
                       "            grupos[-1][1] = s\n"
                       "        else:\n"
                       "            grupos.append([s, s])\n"
                       "    return lista_y([f'{a}' if a == b else f'{a}–{b}' for a, b in grupos])"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Gantt con inicios más tempranos y carga de operarios"\n#| label: fig-ganttCargas\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas frente a la plantilla disponible"\n#| label: fig-cargas\n'
                       "cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[plantilla])"))
    celdas.append(md(f"La tabla siguiente da, semana a semana, los operarios necesarios con inicios más tempranos (fila «Total» "
                     f"del Gantt), el exceso sobre la plantilla, exceso = máx(0, necesarios − {L}), las horas extra que ese exceso "
                     f"supone (exceso × 40 h) y su reparto uniforme entre los {L} operarios (horas extra / {L})."))
    celdas.append(code('#| tbl-cap: "Exceso de operarios y horas extraordinarias semana a semana"\n#| label: tbl-exceso\n'
                       "exceso = (cargas - plantilla).clip(lower=0)\n"
                       "tabla = pd.DataFrame({'operarios necesarios': cargas, 'exceso sobre la plantilla': exceso,\n"
                       "                      'horas extra de la semana': exceso * jornada, 'horas extra por operario': exceso * jornada / plantilla})\n"
                       "tabla.loc['Total'] = tabla.sum()\n"
                       "horas_persona = float(tabla.loc['Total', 'horas extra por operario'])\n"
                       "limpia_tabla(tabla, 1).T"))
    celdas.append(code(
        "semanas_exceso = [s for s in exceso.index if exceso[s] > 0]\n"
        "op_semanas = float(exceso.sum())\n"
        "horas_totales = op_semanas * jornada\n"
        "semana_max = max(semanas_exceso, key=lambda s: exceso[s])\n"
        "Markdown(f'''Pico de **{int(cargas.max())}** operarios frente a {plantilla}; hay exceso en {plural(len(semanas_exceso), 'semana')} "
        "({'semana' if len(semanas_exceso) == 1 else 'semanas'} {rangos(semanas_exceso)}).\n\n"
        "- Exceso total: {' + '.join(str(int(exceso[s])) for s in semanas_exceso)} = **{op_semanas:.0f} operario-semanas**.\n"
        "- Horas extra: {op_semanas:.0f} × {jornada} h = **{horas_totales:.0f} h**.\n"
        "- Por operario (reparto uniforme): {horas_totales:.0f} / {plantilla} = **{horas_persona:.0f} h** en el año (el proyecto dura {duracion} semanas, "
        "menos de un año).\n"
        "- En la peor semana (la {semana_max}) cada operario trabajaría {jornada} + {exceso[semana_max] * jornada / plantilla:.0f} = "
        "{jornada + exceso[semana_max] * jornada / plantilla:.0f} h.\n\n"
        "**Respuesta.** El plan exige **{horas_persona:.0f} horas extraordinarias por operario** en el año, "
        "{'más que' if horas_persona > maximo_horas_extra else 'dentro de'} las {maximo_horas_extra} que permite la ley.''')"))
    celdas += apartado(2)
    celdas.append(code("proyecto_asignado = proyecto.copy()\n"
                       "datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(data=datos_proyecto.copy(), duration_label='duration',\n"
                       "                                                                  resource_label='resources', maximo=[plantilla], report=False, params=GANTT_PARAMS)\n"
                       "duracion_con_limite = len(gantt_asignado.data.columns.drop('H_total'))\n"
                       "contratar = int(cargas.max()) - plantilla\n"
                       "por_nivel = {int(k): [s for s in semanas_exceso if exceso[s] == k] for k in sorted(set(exceso[semanas_exceso]))}\n"
                       "horas_legales = maximo_horas_extra * plantilla\n"
                       "op_sem_legales = horas_legales / jornada\n"
                       "Markdown(f'''**Con el límite de {plantilla} operarios** (asignación: cada semana entran las actividades por orden de prioridad mientras "
                       "caben en la plantilla; las demás esperan) el proyecto dura **{duracion_con_limite}** semanas, {plural(duracion_con_limite - duracion, 'semana')} "
                       "más que las {duracion} del plan (Gantt siguiente).\n\n"
                       "**Para cumplir las {duracion} semanas sin horas extra** hay que cubrir el exceso con personal contratado: como máximo "
                       "{int(cargas.max())} − {plantilla} = **{contratar}** operarios a la vez, y no todas las semanas: "
                       "{'; '.join(f'{plural(k, \"operario\")} {\"en la semana\" if len(v) == 1 else \"en las semanas\"} {rangos(v)}' for k, v in por_nivel.items())}, "
                       "en total {op_semanas:.0f} operario-semanas de refuerzo.\n\n"
                       "Combinación legal: las {maximo_horas_extra} h por operario permiten {maximo_horas_extra} × {plantilla} = {horas_legales} h = "
                       "{op_sem_legales:.0f} operario-semanas en horas extra; el resto, {max(op_semanas - op_sem_legales, 0):.0f} operario-semanas, con refuerzo.\n\n"
                       "**Respuesta.** Con el límite de {plantilla} operarios el proyecto dura {duracion_con_limite} semanas; para cumplir {duracion} sin horas extra hay que "
                       "contratar {plural(contratar, 'operario')} en el pico ({op_semanas:.0f} operario-semanas en total).''')"))
    celdas.append(code('#| fig-cap: "Gantt con asignación limitada a la plantilla"\n#| label: fig-ganttAsignado\npinta_tikz(dibujo)'))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C7 y norma.** El Estatuto de los Trabajadores limita las horas extraordinarias a {maximo_horas_extra} por trabajador y año "
        "(art. 35.2), sin contar las compensadas con descanso, y la jornada ordinaria a {jornada} horas semanales de promedio (art. 34.1). El plan del jefe "
        "de proyecto exige {horas_persona:.0f} h por operario, {horas_persona - maximo_horas_extra:+.0f} h sobre el máximo, con semanas de "
        "{jornada + exceso[semana_max] * jornada / plantilla:.0f} h. Según C7, un plan que incumple una norma no es una opción de planificación, por mucho que "
        "las horas «se paguen»: pagar no convierte en legal lo que la ley limita.\n\n"
        "**Respuesta.** " + "{'**No puede ejecutarse**: ' + f'{horas_persona:.0f} h > {maximo_horas_extra} h por operario y año, salvo que el exceso se compense con descanso (lo que alargaría el proyecto igualmente).' if horas_persona > maximo_horas_extra else 'Puede ejecutarse legalmente: ' + f'{horas_persona:.0f} h ≤ {maximo_horas_extra}; la pregunta pasa al bloque C (si conviene).'}''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Operarios', f'{horas_persona:.0f} h extra cada uno y semanas de hasta {jornada + exceso[semana_max] * jornada / plantilla:.0f} h: salud, seguridad y descanso'),\n"
        "     ('Cliente', f'quiere {duracion} semanas; con la plantilla serían {duracion_con_limite}'),\n"
        "     ('Empresa', f'ahorra la contratación de {op_semanas:.0f} operario-semanas a cambio de un incumplimiento sancionable y de un riesgo de accidentes'),\n"
        "     ('Jefe de proyecto', 'cumple el plazo «sobre el papel» con un plan que no puede firmar')],\n"
        "    [(f'Horas extra sin límite: {horas_persona:.0f} h por operario (propuesta)', f'no: supera las {maximo_horas_extra} h del art. 35.2', 'no', 'no: ningún operario aceptaría esas semanas si pudiera elegir'),\n"
        "     (f'Aceptar {duracion_con_limite} semanas con la plantilla', 'sí', 'sí', 'sí, si el cliente lo acepta informado'),\n"
        "     (f'Contratar el refuerzo: {op_semanas:.0f} operario-semanas (hasta {contratar} a la vez)', 'sí', 'sí', 'sí'),\n"
        "     (f'Combinar: {op_sem_legales:.0f} operario-semanas en horas extra (≤ {maximo_horas_extra} h por operario) y el resto con refuerzo', 'sí', 'sí', 'sí')],\n"
        "    f'''al jefe de proyecto, la tabla de opciones con su coste y su fecha ({duracion_con_limite} semanas con la plantilla; {duracion} contratando "
        "{op_semanas:.0f} operario-semanas; o la combinación dentro de las {maximo_horas_extra} h); al cliente, la opción elegida con su precio y su fecha. "
        "Lo que no se le cuenta es que el plazo «se consigue» incumpliendo la ley, porque no está en la tabla.''')"))
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
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(md(f"El algoritmo de Ackoff-Sasieni acorta el proyecto semana a semana: en cada paso busca, entre las combinaciones "
                     f"de actividades que acortan a la vez **todas** las rutas críticas, la de menor coste (suma de los costes "
                     f"unitarios de reducción, `ucr`), respetando las duraciones mínimas (`cap_duration`). La inspección {I} tiene "
                     f"`ucr` infinito: no puede elegirse nunca."))
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=semanas_pedidas, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("coste_reduccion = float(sum(datos_proyecto.loc[acts, 'ucr'].sum() for acts in pasos.values()))\n"
                       "duracion_reducida = int(proyecto.duration(duraciones_reducidas))\n"
                       "costes_paso = [float(datos_proyecto.loc[acts, 'ucr'].sum()) for acts in pasos.values()]\n"
                       "lineas = [f'- Paso {i}: se acorta {lista_y(acts)} una semana, coste ' + (' + '.join(f\"{datos_proyecto.loc[a, 'ucr']:g}\" for a in acts) + ' = ' if len(acts) > 1 else '')\n"
                       "          + f'{c:g} k€; el proyecto pasa a {duracion - i} semanas.' for i, (acts, c) in enumerate(zip(pasos.values(), costes_paso), 1)]\n"
                       "Markdown('\\n'.join(lineas) + f'''\\n\\n**Respuesta.** Adelantar {semanas_pedidas} semanas sin tocar {inspeccion} cuesta "
                       "{' + '.join(f'{c:g}' for c in costes_paso)} = **{coste_reduccion:g} k€** (de {duracion} a {duracion_reducida} semanas).''')"))
    celdas += apartado(2)
    celdas.append(code("coste_esperado_sin_inspeccion = p_defecto * coste_defecto\n"
                       "d_insp = int(datos_proyecto.loc[inspeccion, 'duration'])\n"
                       "Markdown(f'''Coste esperado de poner en servicio sin la inspección = P(defecto) × coste del defecto = {p_defecto} × {coste_defecto} = "
                       "**{coste_esperado_sin_inspeccion:g} k€**, frente a los {coste_reduccion:g} k€ de la reducción legítima "
                       "({coste_esperado_sin_inspeccion / coste_reduccion:.1f} veces más).\n\n"
                       "**Respuesta.** La propuesta «ahorra» los {coste_reduccion:g} k€ de la reducción {'(y gana ' + plural(d_insp, 'semana') + ', las ' + str(semanas_pedidas) + ' pedidas)' if d_insp == semanas_pedidas else '(y gana ' + plural(d_insp, 'semana') + ' en lugar de las ' + str(semanas_pedidas) + ' pedidas)'}, pero no los ahorra de verdad: los cambia por un coste esperado de {coste_esperado_sin_inspeccion:g} k€, incierto "
                       "y que no paga quien decide sino otros (usuarios, cliente, aseguradora). "
                       "{'En valor esperado es peor negocio: ' + f'{coste_esperado_sin_inspeccion:g} > {coste_reduccion:g} k€.' if coste_esperado_sin_inspeccion > coste_reduccion else 'Aunque en valor esperado saliera más barato, el criterio del apartado 3 no depende de esta cuenta.'}''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C7 (cumplimiento de la ley y de las normas de seguridad).** El enunciado dice que {inspeccion} (pruebas de presión e "
        "inspección reglamentaria) está exigida por la normativa de seguridad **antes** de la puesta en servicio. Según C7, una actividad exigida por una "
        "norma no es una opción de compresión: por eso en la tabla de Ackoff su coste de reducción es infinito, y no «cero» como sugiere el director de "
        "operaciones. Hacer las pruebas «en marcha» el primer mes no es lo mismo: incumple el «antes», y el inspector que firma después firma algo que no "
        "ha visto.\n\n"
        "Comprimir el cronograma es hacer el **mismo alcance** en menos tiempo pagando por ello (acortar duraciones con más recursos, solapar actividades), "
        "como en el apartado 1: {coste_reduccion:g} k€ por {semanas_pedidas} semanas. Suprimir {inspeccion} es **recortar el alcance**; y como {inspeccion} es un "
        "requisito legal, es además incumplir la norma.\n\n"
        "**Respuesta.** No: la supresión de {inspeccion} no es una opción de compresión (C7). La respuesta no depende del cálculo del apartado 2; ese cálculo "
        "sólo muestra que, además, la propuesta {'es peor negocio' if coste_esperado_sin_inspeccion > coste_reduccion else 'no aporta un ahorro que la justifique'}.''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Cliente', f'quiere adelantar {semanas_pedidas} semanas; sin la inspección asume, sin saberlo, un {porc(p_defecto, 0)} de un defecto de {coste_defecto} k€'),\n"
        "     ('Usuarios y terceros', 'soportan el riesgo de seguridad de una instalación a presión sin probar'),\n"
        "     ('Inspector', 'se le pide firmar algo que no ha visto'),\n"
        "     ('Empresa', f'«ahorra» {coste_reduccion:g} k€ y se expone a la sanción, a la responsabilidad por el defecto y a perder el cliente')],\n"
        "    [(f'Suprimir {inspeccion} y probar «en marcha» (propuesta)', 'no: la norma exige la inspección antes de la puesta en servicio', 'no', 'no: el usuario no aceptaría la instalación sin probar'),\n"
        "     (f'Acortar {semanas_pedidas} semanas con Ackoff por {coste_reduccion:g} k€', 'sí', 'sí', 'sí'),\n"
        "     ('Adelantar menos semanas por menos dinero (la curva coste-semanas)', 'sí', 'sí', 'sí'),\n"
        "     (f'Negociar con el cliente la fecha o el reparto del coste de {coste_reduccion:g} k€', 'sí', 'sí', 'sí')],\n"
        "    f'''al director de operaciones: la inspección no está en la tabla de opciones; la tabla real es «{semanas_pedidas} semanas por {coste_reduccion:g} k€» "
        "acortando {lista_y(sorted(set(a for acts in pasos.values() for a in acts)))}, o menos semanas por menos dinero. Al cliente: la oferta de reducción con su "
        "precio; la puesta en servicio sigue exigiendo la inspección. Prueba de publicidad: ¿explicaría al inspector, o a los usuarios, que las pruebas se hicieron "
        "después de arrancar?''')"))
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
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la línea base", label="fig-pert")
    celdas += apartado(1)
    celdas.append(md(_NOTACION_VG + " PV se calcula con las fechas de la línea base: cada actividad reparte su BAC "
                     "uniformemente entre su inicio más temprano y su fin."))
    celdas.append(code('#| tbl-cap: "PV, EV y AC contra la línea base aprobada"\n#| label: tbl-detalleBase\n'
                       "detalle_base, ind_base = indicadores(seguimiento, calendario, semana_control)\n"
                       "detalle_base[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(code(
        "fin_previsto = duracion + desplazamiento\n"
        "Markdown(f'''Totales en la semana {semana_control}:\n\n"
        "- {explica_suma(sin_ceros(detalle_base['PV_i']), 'PV', 'k€', 1)}\n"
        "- {explica_suma(sin_ceros(detalle_base['EV_i']), 'EV', 'k€', 1)}\n"
        "- {explica_suma(sin_ceros(seguimiento['AC']), 'AC', 'k€', 0)}\n\n"
        "{explica_indicadores(ind_base, titulo='**Contra la línea base aprobada:**')}\n\n"
        "Fecha de fin prevista: el retraso es sólo de arranque ({desplazamiento} semanas) y desde entonces todo avanza al ritmo previsto, así que el "
        "proyecto terminará {desplazamiento} semanas después de lo planificado: {duracion} + {desplazamiento} = **semana {fin_previsto}**. (La previsión "
        "genérica duración / SPI = {duracion} / {ind_base[\"SPI\"]:.3f} = {duracion / ind_base[\"SPI\"]:.1f} semanas no sirve aquí: supone que el retraso "
        "seguirá creciendo al ritmo actual, y el enunciado dice que no.)\n\n"
        "**Respuesta.** PV = {ind_base[\"PV\"]:.1f}, EV = {ind_base[\"EV\"]:.1f} y AC = {ind_base[\"AC\"]:.1f} k€; SV = {ind_base[\"SV\"]:.1f} k€ y "
        "SPI = {ind_base[\"SPI\"]:.2f} (retraso); CV = {ind_base[\"CV\"]:.1f} k€ y CPI = {ind_base[\"CPI\"]:.2f} (el coste va según lo hecho); "
        "fin previsto en la semana {fin_previsto}, frente a la {duracion} de la línea base.''')"))
    celdas += apartado(2)
    celdas.append(md("Desplazar la línea base equivale a sumar el desplazamiento a todas las fechas del calendario; en la semana de "
                     "control, PV se calcula entonces con las semanas transcurridas desde el nuevo inicio."))
    celdas.append(code('#| tbl-cap: "PV, EV y AC contra la línea base desplazada"\n#| label: tbl-detalleDesplazada\n'
                       "calendario_desplazado = calendario.copy()\n"
                       "for col in ['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio']:\n"
                       "    calendario_desplazado[col] = calendario[col] + desplazamiento\n"
                       "detalle_desp, ind_desp = indicadores(seguimiento, calendario_desplazado, semana_control)\n"
                       "detalle_desp[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(code('#| tbl-cap: "Indicadores contra las dos líneas base"\n#| label: tbl-comparacionLB\n'
                       "comparacion = pd.DataFrame({'línea base aprobada': ind_base, 'línea base desplazada': ind_desp})\n"
                       "comparacion.round(3)"))
    celdas.append(code(
        "Markdown(f'''- {explica_suma(sin_ceros(detalle_desp['PV_i']), 'PV (desplazada)', 'k€', 1)}\n\n"
        "{explica_indicadores(ind_desp, titulo='**Contra la línea base desplazada:**')}\n\n"
        "**Respuesta.** Sólo cambia PV ({ind_base[\"PV\"]:.1f} → {ind_desp[\"PV\"]:.1f} k€) y, con él, SV ({ind_base[\"SV\"]:.1f} → {ind_desp[\"SV\"]:.1f} k€) "
        "y SPI ({ind_base[\"SPI\"]:.2f} → {ind_desp[\"SPI\"]:.2f}). EV, AC, CV y CPI no cambian, porque no dependen de la línea base. La fecha de fin prevista "
        "**no cambia**: sigue siendo la semana {fin_previsto}. El retraso desaparece del informe, no del proyecto.''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C9 (línea base y control de cambios).** La línea base es un compromiso aprobado por el cliente (fin en la semana {duracion}); "
        "sólo cambia mediante una solicitud de cambio aprobada, no por un «ajuste técnico» del jefe de proyecto. Que la causa del retraso sea del cliente "
        "(permisos entregados {desplazamiento} semanas tarde) no cambia el procedimiento: es precisamente el motivo de una solicitud de cambio (o de una "
        "reclamación) en la que el cliente reconoce esas {desplazamiento} semanas y aprueba el nuevo fin en la semana {fin_previsto}.\n\n"
        "**Criterio C3 (información fuera de contexto).** Sin ese trámite, el informe con SPI = {ind_desp[\"SPI\"]:.2f} da una cifra aritméticamente correcta "
        "contra una referencia que nadie ha aprobado, presentada como si fuera el plan: el cliente leería «en plazo» un proyecto que acabará en la semana "
        "{fin_previsto} y no en la {duracion}.\n\n"
        "**Respuesta.** No puede desplazarla por su cuenta (C9). El informe con SPI = {ind_desp[\"SPI\"]:.2f} sería **información fuera de contexto (C3)**. "
        "Lo haría legítimo una **solicitud de cambio aprobada** que mueva la línea base {desplazamiento} semanas; con ella, el mismo SPI ≈ 1 sería correcto.''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Cliente', f'recibe «SPI = {ind_desp[\"SPI\"]:.2f}» y planifica con la semana {duracion}, cuando el fin previsto es la {fin_previsto}'),\n"
        "     ('Contratista', f'sin solicitud de cambio pierde la prueba de que las {desplazamiento} semanas de retraso son imputables al cliente (y la compensación de la espera)'),\n"
        "     ('Jefe de proyecto', 'evita un informe en rojo este mes a costa de su credibilidad cuando la fecha real aparezca')],\n"
        "    [('Desplazar la línea base sin aprobación e informar SPI ≈ 1 (propuesta)', 'no infringe una ley, pero sí el procedimiento contractual de cambios', 'no', 'no: el cliente no aceptaría que le cambien la referencia sin preguntar'),\n"
        "     (f'Informar SPI {ind_base[\"SPI\"]:.2f} contra la línea base aprobada, con la causa documentada', 'sí', 'sí', 'sí'),\n"
        "     (f'Además, presentar una solicitud de cambio por causa del cliente: fin en la semana {fin_previsto}', 'sí', 'sí', 'sí')],\n"
        "    f'''informe con SPI {ind_base[\"SPI\"]:.2f} contra la línea base aprobada, la causa documentada (permisos entregados {desplazamiento} semanas tarde) y "
        "la fecha prevista, semana {fin_previsto}. Al cliente: solicitud de cambio que mueva la línea base {desplazamiento} semanas y, si procede, la compensación "
        "de los costes de la espera; una vez aprobada, los informes siguientes miden contra la nueva línea base.''')"))
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
                       "reparto = pd.DataFrame(0.0, index=datos_proyecto.index, columns=factura.index)\n"
                       "for a, fila in datos_proyecto.iterrows():\n"
                       "    es, d = int(calendario.loc[a, 'inicio_mas_temprano']), int(fila['duration'])\n"
                       "    for w in range(es, es + d):\n"
                       "        factura[w // 4 + 1] += fila['coste_subcontrata'] / d\n"
                       "        reparto.loc[a, w // 4 + 1] += fila['coste_subcontrata'] / d\n"
                       "def valor_actual(dias, m):\n"
                       "    '''Valor actual de las facturas pagadas «dias» días después de cada fin de mes, al tipo mensual m.'''\n"
                       "    return float(sum(f / (1 + m) ** (t + dias / 30) for t, f in factura.items()))"))
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(md("Con inicios más tempranos, cada actividad factura su coste repartido por igual entre sus semanas (coste / "
                     "duración por semana); el mes 1 son las semanas 1–4, el mes 2 las semanas 5–8, y así sucesivamente. La tabla "
                     "reparte el coste de cada actividad entre los meses; la fila «factura» es la suma de cada mes."))
    celdas.append(code('#| tbl-cap: "Facturación mensual de las subcontratas por actividad (k€)"\n#| label: tbl-facturas\n'
                       "tabla_facturas = reparto.copy()\n"
                       "tabla_facturas['semanas (inicio–fin)'] = [(lambda i, f: f'{i}' if i == f else f'{i}–{f}')(int(calendario.loc[a, 'inicio_mas_temprano']) + 1, int(calendario.loc[a, 'fin_mas_temprano'])) for a in reparto.index]\n"
                       "tabla_facturas.loc['factura'] = list(factura) + ['']\n"
                       "limpia_tabla(tabla_facturas, 2)"))
    celdas.append(code(
        "def tabla_va(m, nombre):\n"
        "    t = pd.DataFrame({'factura': factura})\n"
        "    for dias in (30, 60, 90):\n"
        "        t[f'factor a {dias} días = 1/(1+m)^(t+{dias // 30})'] = [1 / (1 + m) ** (mes + dias / 30) for mes in factura.index]\n"
        "        t[f'VA a {dias} días'] = t['factura'] * t[f'factor a {dias} días = 1/(1+m)^(t+{dias // 30})']\n"
        "    t.loc['Total'] = [factura.sum()] + [np.nan if c.startswith('factor') else t[c].sum() for c in t.columns[1:]]\n"
        "    return t\n"
        "va_c = tabla_va(m_c, 'contratista')\n"
        "display(Markdown(f'''Tipo mensual equivalente del contratista: m = (1 + {k_contratista})^(1/12) − 1 = **{m_c:.5f}**. La factura del mes t pagada a "
        "d días se descuenta t + d/30 meses: VA = Σ factura_t / (1 + m)^(t + d/30). Valor actual de los pagos para el contratista:'''))\n"
        "limpia_tabla(va_c, 4)"))
    celdas.append(code(
        "tabla = pd.DataFrame({dias: {'valor actual para el contratista': valor_actual(dias, m_c),\n"
        "                             'valor actual para las subcontratas': valor_actual(dias, m_s)} for dias in (30, 60, 90)}).T\n"
        "tabla.index.name = 'días de pago'\n"
        "tabla['ganancia del contratista frente a 30'] = tabla.loc[30, 'valor actual para el contratista'] - tabla['valor actual para el contratista']\n"
        "tabla['pérdida de las subcontratas frente a 30'] = tabla.loc[30, 'valor actual para las subcontratas'] - tabla['valor actual para las subcontratas']\n"
        "tabla['neto conjunto'] = tabla['ganancia del contratista frente a 30'] - tabla['pérdida de las subcontratas frente a 30']\n"
        "g = tabla['ganancia del contratista frente a 30']\n"
        "vc = tabla['valor actual para el contratista']\n"
        "Markdown(f'''Pagar más tarde reduce el valor actual de lo que paga el contratista; esa reducción es su ganancia:\n\n"
        "- A 90 días: {vc[30]:.2f} − {vc[90]:.2f} = **{g[90]:.2f} k€**.\n"
        "- A 60 días: {vc[30]:.2f} − {vc[60]:.2f} = **{g[60]:.2f} k€**.\n\n"
        "**Respuesta.** Facturas mensuales de {lista_y([f'{v:.2f}' for v in factura])} k€ (total {factura.sum():.0f} k€). Pagar a 90 días en vez de a 30 "
        "vale {g[90]:.2f} k€ para el contratista y pagar a 60, {g[60]:.2f} k€: un {100 * g[90] / factura.sum():.1f} % y un {100 * g[60] / factura.sum():.1f} % "
        "de lo facturado.''')"))
    celdas += apartado(2)
    celdas.append(code(
        "va_s = tabla_va(m_s, 'subcontratas')\n"
        "display(Markdown(f'''Las subcontratas se financian al {porc(k_subcontratas, 0)}: m = (1 + {k_subcontratas})^(1/12) − 1 = **{m_s:.5f}**. Valor actual "
        "de los cobros para las subcontratas:'''))\n"
        "limpia_tabla(va_s, 4)"))
    celdas.append(code('#| tbl-cap: "Ganancia, pérdida y neto frente a pagar a 30 días (k€)"\n#| label: tbl-neto\n'
                       "display(tabla.round(2))\n"
                       "pe = tabla['pérdida de las subcontratas frente a 30']\n"
                       "vs = tabla['valor actual para las subcontratas']\n"
                       "Markdown(f'''- Pérdida de las subcontratas a 90 días: {vs[30]:.2f} − {vs[90]:.2f} = **{pe[90]:.2f} k€**; a 60 días: {vs[30]:.2f} − {vs[60]:.2f} = **{pe[60]:.2f} k€**.\n"
                       "- Neto conjunto a 90 días: {g[90]:.2f} − {pe[90]:.2f} = **{g[90] - pe[90]:.2f} k€**; a 60 días: {g[60]:.2f} − {pe[60]:.2f} = **{g[60] - pe[60]:.2f} k€**.\n\n"
                       "No es un juego de suma cero porque cada parte valora el dinero aplazado a su propio coste de financiación: dos meses de aplazamiento valen "
                       "para el contratista lo que le cuesta financiarse ({porc(k_contratista, 0)} anual) y le cuestan a la subcontrata lo que le cuesta a ella "
                       "({porc(k_subcontratas, 0)}). Mientras {porc(k_subcontratas, 0)} > {porc(k_contratista, 0)}, lo que pierde la subcontrata es más de lo que gana "
                       "el contratista.\n\n"
                       "**Respuesta.** Las subcontratas pierden {pe[90]:.2f} k€ (90 días) y {pe[60]:.2f} k€ (60 días); el conjunto pierde {-(g[90] - pe[90]):.2f} k€ y "
                       "{-(g[60] - pe[60]):.2f} k€: el aplazamiento no crea valor, traslada dinero de quien se financia caro a quien se financia barato y por el camino "
                       "destruye una parte.''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Norma** (Ley 3/2004, art. 4). Entre empresas, el plazo de pago es de 30 días si no se pacta otro y, por pacto, nunca más de "
        "{plazo_legal_maximo}. Pagar a 90 días **es ilegal**, aunque la subcontrata lo acepte: el pacto no puede superar el máximo. Pagar a 60 es **legal** "
        "si se pacta.\n\n"
        "**Criterio C5 (beneficio propio a costa de otro).** La propuesta obtiene {g[90]:.2f} k€ trasladando a las subcontratas un coste mayor "
        "({pe[90]:.2f} k€) que no han aceptado: «si alguna protesta, siempre hay otra que acepta» describe una imposición, no un acuerdo. **Criterio C10 "
        "(buena fe en la negociación).** Negociar el plazo es legítimo; imponerlo con la amenaza de sustituir a quien proteste, no; y decir que «no cuesta "
        "nada» es falso: cuesta {pe[90]:.2f} k€ a las subcontratas.\n\n"
        "**Respuesta.** A 90 días: ilegal (más de {plazo_legal_maximo} días) y, además, C5 y C10. A 60 días: legal si se pacta libremente; formulado como "
        "en el enunciado (impuesto con la amenaza de cambiar de proveedor) sigue incumpliendo C5 (gana {g[60]:.2f} k€ a costa de {pe[60]:.2f} k€ ajenos) "
        "y C10.''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Subcontratas', f'pierden {pe[90]:.2f} k€ (a 90) o {pe[60]:.2f} k€ (a 60) en valor actual; tensiones de tesorería en empresas pequeñas'),\n"
        "     ('Contratista', f'gana {g[90]:.2f} k€ (a 90) o {g[60]:.2f} k€ (a 60), y arriesga sanción, reclamaciones de intereses de demora y peores proveedores'),\n"
        "     ('Cliente y proyecto', 'una subcontrata ahogada financieramente entrega peor o más tarde')],\n"
        "    [('Pagar a 90 días impuesto (propuesta)', f'no: supera los {plazo_legal_maximo} días', 'no', 'no: la empresa no aceptaría que su cliente le pagara a 90'),\n"
        "     ('Pagar a 30 días', 'sí', 'sí', 'sí'),\n"
        "     (f'Pactar 60 días compensando a la subcontrata ({pe[60]:.2f} k€)', 'sí', 'sí', f'sí, pero la compensación ({pe[60]:.2f}) supera la ganancia ({g[60]:.2f}): no conviene a nadie'),\n"
        "     ('Financiar el desfase de tesorería con el banco', 'sí', 'sí', 'sí')],\n"
        "    f'''pagar a 30 días. Al director financiero: la ganancia del aplazamiento es pequeña ({g[90]:.2f} k€ a 90 días), menor que lo que pierden las "
        "subcontratas ({pe[90]:.2f} k€), y a 90 días además es ilegal; a 60 días no hay precio que compense a la subcontrata y deje ganancia al contratista, "
        "porque el aplazamiento no crea valor que repartir. Prueba de reversibilidad: ¿aceptaría la empresa que el cliente le pagara a 90?''')"))
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
                       "es orientativo; la experiencia previa con la casa vale más que unos puntos de diferencia en el baremo."),
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
                       f"oferta_cunado, coste_semana, contrato_publico = {cunado!r}, {coste_semana}, {publico}\n"
                       "ofertas = datos_enunciado.copy()"))
    celdas.append(code(_AUX_ETICA))
    celdas += apartado(1)
    celdas.append(md(f"Con el criterio proporcional inverso, la mejor oferta en precio (la más baja) recibe los {pesos['precio']} "
                     f"puntos y cada una de las demás, {pesos['precio']} · precio mínimo / su precio; igual con el plazo y "
                     f"{pesos['plazo']} puntos. La calidad ya viene puntuada sobre {pesos['calidad']}."))
    celdas.append(code('#| tbl-cap: "Puntuación con los criterios publicados"\n#| label: tbl-puntuacion\n'
                       "p_min, t_min = ofertas['precio'].min(), ofertas['plazo'].min()\n"
                       "puntuacion = pd.DataFrame({\n"
                       "    'precio': pesos['precio'] * p_min / ofertas['precio'],\n"
                       "    'plazo': pesos['plazo'] * t_min / ofertas['plazo'],\n"
                       "    'calidad': ofertas['calidad'].astype(float)})\n"
                       "puntuacion['total'] = puntuacion.sum(axis=1)\n"
                       "puntuacion = puntuacion.sort_values('total', ascending=False)\n"
                       "mejor = puntuacion.index[0]\n"
                       "detalle = pd.DataFrame({\n"
                       "    f\"precio = {pesos['precio']} · mín / precio\": [f\"{pesos['precio']} · {p_min:g} / {ofertas.loc[o, 'precio']:g} = {puntuacion.loc[o, 'precio']:.2f}\" for o in puntuacion.index],\n"
                       "    f\"plazo = {pesos['plazo']} · mín / plazo\": [f\"{pesos['plazo']} · {t_min} / {ofertas.loc[o, 'plazo']} = {puntuacion.loc[o, 'plazo']:.2f}\" for o in puntuacion.index],\n"
                       "    'calidad': [f\"{puntuacion.loc[o, 'calidad']:.0f}\" for o in puntuacion.index],\n"
                       "    'total = suma': [f\"{puntuacion.loc[o, 'precio']:.2f} + {puntuacion.loc[o, 'plazo']:.2f} + {puntuacion.loc[o, 'calidad']:.0f} = {puntuacion.loc[o, 'total']:.2f}\" for o in puntuacion.index]},\n"
                       "    index=puntuacion.index)\n"
                       "detalle"))
    celdas.append(code("dif_puntos = puntuacion.loc[mejor, 'total'] - puntuacion.loc[oferta_cunado, 'total']\n"
                       "Markdown(f'''**Respuesta.** Orden: {', '.join(f\"{i}.ª {o} ({puntuacion.loc[o, 'total']:.2f} puntos)\" for i, o in enumerate(puntuacion.index, 1))}. "
                       "Gana la **{mejor}**; la {oferta_cunado}, que propone el jefe de proyecto, queda {dif_puntos:.2f} puntos por debajo.''')"))
    celdas += apartado(2)
    celdas.append(code("dif_precio = float(ofertas.loc[oferta_cunado, 'precio'] - ofertas.loc[mejor, 'precio'])\n"
                       "dif_plazo = int(ofertas.loc[oferta_cunado, 'plazo'] - ofertas.loc[mejor, 'plazo'])\n"
                       "sobrecoste = dif_precio + dif_plazo * coste_semana\n"
                       "Markdown(f'''- Diferencia de precio: {ofertas.loc[oferta_cunado, 'precio']:g} − {ofertas.loc[mejor, 'precio']:g} = {dif_precio:+.1f} k€.\n"
                       "- Diferencia de plazo: {ofertas.loc[oferta_cunado, 'plazo']} − {ofertas.loc[mejor, 'plazo']} = {dif_plazo:+d} semanas, que a {coste_semana} k€ por "
                       "semana de indirectos son {dif_plazo:+d} × {coste_semana} = {dif_plazo * coste_semana:+d} k€.\n"
                       "- Total: {dif_precio:+.1f} {'+' if dif_plazo * coste_semana >= 0 else '−'} {abs(dif_plazo * coste_semana)} = **{sobrecoste:+.1f} k€**.\n\n"
                       "**Respuesta.** Adjudicar a la {oferta_cunado} en lugar de a la {mejor} "
                       "{'costaría al proyecto ' + f'{sobrecoste:.1f} k€ más' if sobrecoste > 0 else 'no costaría más al proyecto (' + f'{sobrecoste:+.1f} k€' + '), pero sigue siendo peor según el pliego'}.''')"))
    celdas += apartado(3)
    texto_marco = ("**Contrato público.** La LCSP (art. 64) obliga además al órgano de contratación a prevenir, detectar y "
                   "resolver los conflictos de interés, y una adjudicación con un criterio no publicado puede impugnarse y anularse. "
                   if publico else
                   "**Contrato privado.** No hay un procedimiento administrativo que anule la adjudicación, pero el criterio ético es "
                   "el mismo y la empresa suele recogerlo en su código de conducta. ")
    celdas.append(code(
        "Markdown(f'''**Criterio C6 (conflicto de interés no revelado).** El jefe de proyecto preside la mesa y es cuñado del gerente de la empresa que "
        "presenta la {oferta_cunado}; tiene, por tanto, un interés personal en el resultado, y no lo ha comunicado a nadie. Lo exigible según C6 no depende de "
        "qué oferta sea mejor: **revelarlo** a la mesa y a su organización y **abstenerse** de la decisión. Proponer además la oferta del familiar es decidir "
        "por consideraciones personales (cf. PMI 4.3.3).\n\n"
        "**«La experiencia previa con la casa».** No es uno de los criterios publicados (precio, plazo y calidad); aplicarlo después de abrir las ofertas "
        "cambia las reglas a favor de un licitador (cf. PMI 4.3.5). La mesa sólo puede aplicar el pliego, y con el pliego gana la {mejor} por {dif_puntos:.2f} puntos.\n\n"
        "" + texto_marco.replace("{", "{{").replace("}", "}}") + "\n\n"
        "**Respuesta.** El jefe de proyecto debe revelar el parentesco y abstenerse (C6), sea cual sea la mejor oferta. La mesa no puede aplicar «la "
        "experiencia previa», porque no está en el pliego. Que el contrato sea {'público' if contrato_publico else 'privado'} cambia las consecuencias "
        "jurídicas, no la respuesta: en los dos casos hay que revelar, abstenerse y adjudicar con los criterios publicados.''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Proyecto y empresa adjudicadora', f'{sobrecoste:+.1f} k€ si se adjudica a la {oferta_cunado}' + ('' if sobrecoste > 0 else ' (no hay sobrecoste, pero sí una adjudicación contra el pliego)')),\n"
        "     ('Licitadora mejor puntuada', f'la {mejor} pierde un contrato que ganó con las reglas publicadas'),\n"
        "     ('Jefe de proyecto', 'favorece a su familia y arriesga su puesto y la validez de la adjudicación'),\n"
        "     ('Usted y la mesa', 'si callan sabiendo el parentesco, comparten la decisión')],\n"
        "    [(f'Adjudicar a la {oferta_cunado} por «experiencia» (propuesta)', 'no en contrato público; contraria al pliego en ambos casos', 'no', 'no: ninguna licitadora aceptaría perder así'),\n"
        "     (f'Que el jefe de proyecto revele y se abstenga, y la mesa adjudique a la {mejor} con el pliego', 'sí', 'sí', 'sí'),\n"
        "     ('Si quiere valorar la experiencia, incluirla como criterio publicado en la próxima licitación', 'sí', 'sí', 'sí'),\n"
        "     ('Si no revela, comunicarlo al órgano competente de la organización', 'sí', 'sí', 'sí')],\n"
        "    f'''en la mesa, pedir que conste la puntuación del apartado 1 y la propuesta de adjudicar a la {mejor}; antes, al jefe de proyecto: «tienes que "
        "decirlo tú y apartarte; si la {oferta_cunado} fuera realmente mejor, la mesa lo verá sin ti». Si no lo hace, comunicarlo a quien corresponda. Prueba de "
        "publicidad: ¿cómo se leería la adjudicación, con {dif_puntos:.2f} puntos menos y {sobrecoste:+.1f} k€, si el parentesco saliera en la prensa?''')"))
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
                 f"no críticas que cedan a alguien de su equipo durante {semanas} semana{'s' if semanas > 1 else ''}, "
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
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(md("Notación: ES y EF, inicio y fin más tempranos (*early start/finish*); LS y LF, inicio y fin más tardíos "
                     "(*late start/finish*). Holgura total = LS − ES (lo que la actividad puede retrasarse sin retrasar el proyecto); "
                     "holgura libre = mín ES de las sucesoras − EF (lo que puede retrasarse sin retrasar el inicio más temprano de "
                     "ninguna sucesora); holgura independiente = máx(0, mín ES de las sucesoras − máx LF de las predecesoras − d) (la "
                     "que tiene aunque las predecesoras acaben lo más tarde posible y las sucesoras empiecen lo antes posible)."))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\ncalendario'))
    celdas.append(code('#| tbl-cap: "Cómo se obtiene cada fecha y cada holgura de la actividad X"\n#| label: tbl-detalleHolguras\n'
                       "detalle_holguras(calendario, prelaciones).loc[[X]].T"))
    celdas.append(code("Ht, Hl, Hi = (int(calendario.loc[X, c]) for c in ('H_total', 'H_libre', 'H_independiente'))\n"
                       "Markdown(f'''**Respuesta.** {X}: ES = {int(calendario.loc[X, 'inicio_mas_temprano'])}, EF = {int(calendario.loc[X, 'fin_mas_temprano'])}, "
                       "LS = {int(calendario.loc[X, 'inicio_mas_tardio'])}, LF = {int(calendario.loc[X, 'fin_mas_tardio'])}; holgura total **{Ht}**, libre **{Hl}** e "
                       "independiente **{Hi}** (semanas). «No tiene ningún margen» es falso: {X} puede retrasarse {plural(Hl, 'semana')} sin afectar a ninguna otra "
                       "actividad y {plural(Ht, 'semana')} sin retrasar el proyecto.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Holgura total de cada actividad en el plan, si X se retrasa lo que se le pide y si consume toda su holgura total"\n#| label: tbl-escenarios\n'
                       "def holguras_si_X_dura(extra):\n"
                       "    d = datos_proyecto['duration'].copy(); d[X] = d[X] + extra\n"
                       "    return proyecto.calendar(d)['H_total'], int(proyecto.duration(d))\n"
                       "H_ceder, T_ceder = holguras_si_X_dura(semanas_cedidas)\n"
                       "H_toda, T_toda = holguras_si_X_dura(Ht)\n"
                       "escenarios = pd.DataFrame({'plan': calendario['H_total'], f'{X} + {semanas_cedidas}': H_ceder, f'{X} + {Ht}': H_toda})\n"
                       "escenarios.T"))
    celdas.append(code(
        "afectadas_ceder = [a for a in escenarios.index if a != X and H_ceder[a] < calendario.loc[a, 'H_total']]\n"
        "afectadas_toda = [a for a in escenarios.index if a != X and H_toda[a] < calendario.loc[a, 'H_total']]\n"
        "nuevas_criticas = [a for a in afectadas_toda if H_toda[a] == 0]\n"
        "def cambios(H):\n"
        "    return lista_y([f'{a} ({int(calendario.loc[a, \"H_total\"])} → {int(H[a])})' for a in escenarios.index if a != X and H[a] < calendario.loc[a, 'H_total']])\n"
        "txt_ceder = ('ninguna otra actividad pierde holgura: el retraso cabe en la holgura libre de ' + X + f' ({Hl})' if not afectadas_ceder\n"
        "             else 'pierden holgura ' + cambios(H_ceder))\n"
        "Markdown(f'''- **{X} se retrasa {plural(semanas_cedidas, 'semana')}**: la duración del proyecto es {T_ceder} semanas "
        "({'no cambia' if T_ceder == duracion else 'cambia'}); {X} pasa de holgura total {Ht} a {int(H_ceder[X])}, y {txt_ceder}.\n"
        "- **{X} consume toda su holgura total ({plural(Ht, 'semana')})**: la duración sigue siendo {T_toda} semanas, pero {X} se vuelve crítica y "
        "{'pierden holgura ' + cambios(H_toda) if afectadas_toda else 'ninguna otra actividad pierde holgura'}"
        "{'; ' + lista_y(nuevas_criticas) + (' pasa' if len(nuevas_criticas) == 1 else ' pasan') + ' a ser ' + ('crítica' if len(nuevas_criticas) == 1 else 'críticas') if nuevas_criticas else ''}.\n\n"
        "**Respuesta.** Retrasar {X} {plural(semanas_cedidas, 'semana')} "
        "{'no afecta a ninguna otra actividad' if not afectadas_ceder else 'reduce la holgura de ' + lista_y(afectadas_ceder)}, y la duración "
        "{'sigue siendo' if T_ceder == duracion else 'pasa a'} {T_ceder} semanas. "
        "Consumir las {Ht} semanas de holgura total no retrasa el proyecto ({T_toda} semanas), pero deja sin margen a "
        "{lista_y(afectadas_toda) if afectadas_toda else 'nadie más'}: la parte de la holgura que excede la libre ({Ht} − {Hl} = {Ht - Hl}) era compartida con "
        "{'ellas' if len(afectadas_toda) > 1 else 'ella'}.''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C1 (afirmación falsa).** «{X} no tiene ningún margen» es falso, y su responsable lo sabe («{X} tiene margen de sobra», dice en "
        "privado): {X} tiene {plural(Ht, 'semana')} de holgura total y {Hl} de libre, y ceder {plural(semanas_cedidas, 'semana')} no retrasaría el proyecto.\n\n"
        "**Criterio C4 (retención de información).** Si sólo hubiera callado el margen, sería C4: el jefe de proyecto necesita las holguras para decidir de "
        "dónde sacar recursos para {Y}, y la holgura no es un dato privado del responsable sino del plan. Afirmar lo contrario es más grave que callar: es C1.\n\n"
        "**A quién pertenece la holgura.** La holgura libre ({plural(Hl, 'semana')}) puede usarla {X} sin afectar a nadie; las {Ht - Hl} semanas restantes "
        "hasta la total son compartidas con {lista_y(afectadas_toda) if afectadas_toda else 'las actividades siguientes'}. Repartir la holgura compartida es "
        "decisión del jefe de proyecto, no de quien la «descubre» primero. Si es justo que quien planifica bien pierda su margen no se resuelve con un criterio "
        "objetivo: es materia del bloque C.\n\n"
        "**Respuesta.** Incumple **C1** (afirma que no hay margen sabiendo que hay {Ht} semanas); no mencionarlo habría sido **C4**. La holgura es del proyecto: "
        "la libre ({Hl}) la usa la actividad sin afectar a nadie y la compartida ({Ht - Hl}) la asigna el jefe de proyecto.''')"))
    celdas.append(_gift(
        "preguntas_test = [\n"
        f"    ('Decir «{X} no tiene margen» sabiendo que tiene {e['Ht']} semanas de holgura total es…', ['C1, afirmación falsa', 'C4, retención de información', 'C8, reserva declarada', 'una estimación prudente'],\n"
        "     'Se afirma algo que se sabe falso; callar habría sido C4.'),\n"
        "    ('La holgura que una actividad puede consumir sin afectar a ninguna otra es la…', ['libre', 'total', 'independiente', 'de la ruta'],\n"
        "     'Holgura libre: mín ES de las sucesoras − EF.'),\n"
        "    ('Si una actividad consume toda su holgura total…', ['la ruta más larga que pasa por ella se vuelve crítica, sin retrasar el proyecto', 'el proyecto se retrasa', 'no pasa nada', 'la ruta crítica desaparece'],\n"
        "     'La duración del proyecto no cambia, pero esa ruta queda sin margen.'),\n"
        "]", "Ética 19 — holgura oculta"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(code(
        "rubrica_caso(\n"
        "    [(f'Proyecto y responsable de {Y}', f'{Y} es crítica y va retrasada: cada semana que no se recupera es una semana más de proyecto'),\n"
        "     (f'Responsable de {X}', f'teme que le quiten gente «cada vez que alguien va mal»; ceder {plural(semanas_cedidas, \"semana\")} no le cuesta plazo (holgura libre {Hl})'),\n"
        "     ((('Responsable de ' if len(afectadas_toda) == 1 else 'Responsables de ') + lista_y(afectadas_toda)) if afectadas_toda else 'Resto del equipo', 'su margen desaparece si alguien consume la holgura compartida sin decirlo'),\n"
        "     ('Jefe de proyecto', 'decide a ciegas si las holguras dependen de lo que cada uno quiera contar')],\n"
        "    [('Que cada responsable informe de su margen si quiere (situación actual)', 'sí', 'no: nadie defendería en público un «no tengo margen» falso', 'no'),\n"
        "     ('Publicar cada semana el calendario con las tres holguras, calculado con los datos de todos', 'sí', 'sí', 'sí'),\n"
        "     ('Regla de cesión: dentro de la holgura libre sin consulta; la compartida, por decisión del jefe de proyecto', 'sí', 'sí', 'sí'),\n"
        "     ('Reconocer a quien cede (no penalizar con más cesiones a quien planifica bien)', 'sí', 'sí', 'sí')],\n"
        "    f'''el calendario con las tres holguras lo publica el jefe de proyecto a partir de los datos de todos, de modo que la pregunta no es «¿tienes margen?» "
        "sino «el plan dice que {X} tiene {plural(Hl, 'semana')} de holgura libre: te pido {semanas_cedidas}». Para el incentivo perverso, reconocer a quien cede y "
        "limitar las cesiones a la holgura libre salvo decisión expresa. Prueba de reversibilidad para el responsable de {X}: ¿aceptaría que el de {Y} le "
        "escondiera su margen cuando fuera él quien lo necesitara?''')"))
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
                       "suma_nominal = flujo_anual * anos\n"
                       "va_flujos = van_restante + inversion_restante"))
    celdas.append(code(_AUX_ETICA))
    celdas += apartado(1)
    celdas.append(md("VAN (valor actual neto) de los flujos que quedan: la inversión que falta se paga hoy (año 0) y cada flujo "
                     "anual se descuenta al coste de capital k con el factor 1/(1 + k)^t. Lo ya gastado no forma parte de esos "
                     "flujos: es un coste hundido, que no se recupera ni se evita con la decisión."))
    celdas.append(code('#| tbl-cap: "Flujos que quedan y su valor actual (k€)"\n#| label: tbl-flujos\n'
                       "detalle = pd.DataFrame({'flujo': [float(f) for f in flujos_restantes], 'factor 1/(1+k)^t': [1 / (1 + k) ** t for t in range(anos + 1)]},\n"
                       "                       index=pd.Index([str(t) for t in range(anos + 1)], name='año'))\n"
                       "detalle['valor actual'] = detalle['flujo'] * detalle['factor 1/(1+k)^t']\n"
                       "detalle.loc['Total'] = [float(sum(flujos_restantes)), np.nan, detalle['valor actual'].sum()]\n"
                       "limpia_tabla(detalle, 4)"))
    celdas.append(code(
        "factor_anualidad = (1 - (1 + k) ** -anos) / k\n"
        "Markdown(f'''- Valor actual de los flujos de explotación: {flujo_anual} · (1/{1 + k:g} + 1/{1 + k:g}² + … + 1/{1 + k:g}^{anos}) = {flujo_anual} · "
        "(1 − {1 + k:g}^−{anos}) / {k:g} = {flujo_anual} · {factor_anualidad:.4f} = **{va_flujos:.2f} k€**.\n"
        "- VAN de lo que queda = −{inversion_restante} + {va_flujos:.2f} = **{van_restante:.2f} k€**.\n"
        "- VAN del proyecto completo = VAN de lo que queda − lo gastado = {van_restante:.2f} − {gastado} = **{van_total:.2f} k€**.\n\n"
        "**Respuesta.** VAN de los flujos que quedan: {van_restante:.2f} k€; VAN del proyecto completo contando lo gastado: {van_total:.2f} k€.''')"))
    celdas += apartado(2)
    celdas.append(code(
        "rec_restante = va_flujos >= inversion_restante\n"
        "rec_total_nominal = suma_nominal >= gastado + inversion_restante\n"
        "Markdown(f'''- El VAN de lo que queda ({van_restante:.2f} k€) es {'positivo' if van_restante > 0 else 'negativo'}: recomienda "
        "**{'continuar' if van_restante > 0 else 'cancelar'}**.\n"
        "- El VAN total ({van_total:.2f} k€) es {'positivo' if van_total > 0 else 'negativo'}: recomendaría "
        "{'continuar' if van_total > 0 else 'cancelar'}.\n\n"
        "El criterio correcto es el **VAN de lo que queda**. Lo gastado ({gastado} k€) es el mismo si se continúa y si se cancela: no cambia con la "
        "decisión y, por tanto, no puede inclinarla. El VAN total responde a otra pregunta, si el proyecto completo fue buena inversión "
        "({'sí' if van_total > 0 else 'no'}), que ya no se puede cambiar.\n\n"
        "¿«Recuperan la inversión»? Depende de qué inversión, y hay que descontar:\n\n"
        "- La que falta: los flujos valen hoy {va_flujos:.2f} k€ {'≥' if rec_restante else '<'} {inversion_restante} k€: "
        "{'sí la recuperan' if rec_restante else 'no la recuperan'} "
        "{('(la suma sin descontar, ' + str(suma_nominal) + ' k€, también la supera)') if rec_restante else ('(sólo la suma sin descontar, ' + str(suma_nominal) + ' k€, la supera: de ahí la apariencia de que la recupera)' if suma_nominal >= inversion_restante else '(ni siquiera la suma sin descontar, ' + str(suma_nominal) + ' k€, llega)')}.\n"
        "- La total, {gastado} + {inversion_restante} = {gastado + inversion_restante} k€: los flujos valen hoy {va_flujos:.2f} k€, así que **no** "
        "(VAN total {van_total:.2f} k€){'; ni siquiera sumando sin descontar: ' + str(suma_nominal) + ' k€' if not rec_total_nominal else '; sólo la suma sin descontar (' + str(suma_nominal) + ' k€) lo aparenta'}.\n\n"
        "**Respuesta.** El VAN de lo que queda recomienda {'continuar' if van_restante > 0 else 'cancelar'} y el VAN total, "
        "{'continuar' if van_total > 0 else 'cancelar'}; el correcto es el primero. "
        "{'La afirmación del borrador es cierta sólo para la inversión que falta; para la inversión total es falsa.' if rec_restante else 'La afirmación del borrador es falsa: descontados, los flujos no cubren ni la inversión que falta.'}''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C12 (decidir sobre el futuro, no sobre lo gastado).** El borrador argumenta con lo gastado («hemos invertido {gastado} k€; "
        "cancelar ahora es tirarlos»), que no es un argumento ni a favor ni en contra, y con una suma sin descontar ({flujo_anual} × {anos} = {suma_nominal} k€). "
        "La decisión se toma con el VAN de lo que queda: {van_restante:.2f} k€, que recomienda **{'continuar' if van_restante > 0 else 'cancelar'}**. "
        + "{'La conclusión del borrador (continuar) es correcta, pero por razones equivocadas.' if van_restante > 0 else 'La propuesta del borrador (continuar) es incorrecta.'}\n\n"
        "**Criterio C1 (afirmación falsa).** «Recuperan la inversión» "
        + "{'es falso si se refiere, como sugiere el borrador, a lo invertido en total (' + str(gastado + inversion_restante) + ' k€ frente a ' + f'{va_flujos:.2f}' + ' k€ de valor actual).' if rec_restante else 'es falso: descontados al ' + porc(k, 0) + ', los flujos valen ' + f'{va_flujos:.2f}' + ' k€, menos que los ' + str(inversion_restante) + ' k€ que faltan (C12 lo califica expresamente de C1).'}"
        " **Criterio C11 (reconocer los errores propios).** Si el motivo real de seguir es no reconocer que el proyecto fue mala inversión (VAN total "
        "{van_total:.2f} k€), el criterio pide lo contrario: reconocerlo y decidir el resto con los números.\n\n"
        "**Respuesta.** {'La propuesta (continuar) es correcta en su conclusión, pero su argumentación incumple C12' if van_restante > 0 else 'La propuesta es incorrecta: incumple C12'}; "
        "la afirmación «recuperan la inversión» incumple **C1**, y seguir para no admitir el error incumpliría **C11**.''')"))
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
    celdas.append(code(
        "decision = 'continuar' if van_restante > 0 else 'cancelar'\n"
        "rubrica_caso(\n"
        "    [('Empresa (accionistas)', ('continuar aporta ' if van_restante > 0 else 'continuar resta ') + f'{abs(van_restante):.2f} k€ de valor desde hoy; los {gastado} k€ gastados están perdidos en cualquier caso'),\n"
        "     ('Comité de dirección', 'necesita la cuenta correcta (VAN de lo que queda) para decidir, no la del borrador'),\n"
        "     ('Jefe de proyecto y equipo', 'reconocer que el proyecto fue mala inversión les cuesta reputación a corto plazo; defenderlo con una cuenta falsa, más a largo plazo')],\n"
        "    [('Enviar el borrador tal cual (propuesta)', 'sí', 'no: no resiste que alguien rehaga la cuenta', 'no: el comité no querría decidir con ella'),\n"
        "     (f'Recomendar {decision} con el VAN de lo que queda y explicar el coste hundido', 'sí', 'sí', 'sí'),\n"
        "     ('Presentar además el análisis de por qué falló la estimación inicial (lecciones aprendidas)', 'sí', 'sí', 'sí')]\n"
        "    + ([] if van_restante > 0 else [('Cancelar y buscar comprador para el trabajo hecho', 'sí', 'sí', 'sí')]),\n"
        "    (f'''párrafo para el comité: «Visto desde el inicio, el proyecto tiene un VAN de {van_total:.0f} k€: esa pérdida ya está hecha. La decisión de hoy es "
        "otra: terminarlo requiere {inversion_restante} k€ y los flujos previstos valen {va_flujos:.0f} k€ actuales, así que continuar "
        "{'aporta ' + f'{van_restante:.0f}' + ' k€. Recomendamos continuar y revisar por qué falló la estimación inicial.»' if van_restante > 0 else 'destruye ' + f'{-van_restante:.0f}' + ' k€ más. Recomendamos cancelar, documentar lo aprendido y, si hay comprador para el trabajo hecho, venderlo.»'} "
        "Al jefe de proyecto: reconocer el error es lo que protege su credibilidad para el siguiente proyecto; defender el proyecto con una cuenta que no se "
        "sostiene la destruye en cuanto alguien la rehace.'''))"))
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
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\ncalendario'))
    celdas.append(code("H = int(calendario.loc[X, 'H_total'])\n"
                       "d = datos_proyecto['duration'].copy(); d[X] = d[X] + semanas_parada\n"
                       "duracion_con_parada = int(proyecto.duration(d))\n"
                       "retraso = duracion_con_parada - duracion\n"
                       "pen_dia_keur = penalidad_por_mil_y_dia * precio_contrato / 1000\n"
                       "penalidad = pen_dia_keur * 7 * retraso\n"
                       "Markdown(f'''- Holgura total de {X}: LS − ES = {int(calendario.loc[X, 'inicio_mas_tardio'])} − {int(calendario.loc[X, 'inicio_mas_temprano'])} = **{H}**.\n"
                       "- Parar {X} {plural(semanas_parada, 'semana')} equivale a alargarla otras tantas: el proyecto pasa de {duracion} a **{duracion_con_parada}** "
                       "semanas; retraso = máx(0, {semanas_parada} − {H}) = **{plural(retraso, 'semana')}** = {7 * retraso} días naturales.\n"
                       "- Penalidad diaria: {penalidad_por_mil_y_dia:.2f} € por cada 1 000 € de precio; el precio es {precio_contrato} k€ = {precio_contrato} × 1 000 €, "
                       "luego {penalidad_por_mil_y_dia:.2f} × {precio_contrato} = {1000 * pen_dia_keur:.0f} € = {pen_dia_keur:.2f} k€ por día.\n"
                       "- Penalidad total: {pen_dia_keur:.2f} × {7 * retraso} días = **{penalidad:.2f} k€**.\n\n"
                       "**Respuesta.** {X} tiene holgura total {H}; si se para {semanas_parada} semanas el proyecto se retrasa {plural(retraso, 'semana')} "
                       "({duracion_con_parada} en lugar de {duracion}) y la penalidad legal es de {penalidad:.2f} k€.''')"))
    celdas += apartado(2)
    celdas.append(code("neto_empresa = ahorro_otro_proyecto - penalidad\n"
                       "cesion = (f'ceder hasta {plural(H, \"semana\")} (su holgura total) no retrasa el proyecto ni cuesta penalidad' if H > 0\n"
                       "          else f'{X} es crítica (holgura total 0): cualquier semana de parada retrasa el proyecto')\n"
                       "Markdown(f'''Neto para la empresa = ahorro en el otro proyecto − penalidad en este = {ahorro_otro_proyecto} − {penalidad:.2f} = **{neto_empresa:+.2f} k€**.\n\n"
                       "**Respuesta.** {'En cifras, a la empresa le conviene ceder: gana ' + f'{neto_empresa:.2f}' + ' k€ netos.' if neto_empresa > 0 else 'En cifras, a la empresa no le conviene ceder: pierde ' + f'{-neto_empresa:.2f}' + ' k€ netos.'} "
                       "Sin retrasar el proyecto, {X} {('puede cederse como mucho **' + plural(H, 'semana') + '**') if H > 0 else 'no puede cederse **ninguna semana**'}: {cesion}.''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Criterio C1 (afirmación falsa).** Decir al cliente que el retraso es «del proveedor de material», sabiendo que lo causa la decisión "
        "de llevarse los operarios de {X}, es comunicar como cierta una causa falsa. En un contrato público la causa importa además jurídicamente: un retraso "
        "imputable al contratista lleva penalidad ({penalidad:.2f} k€ aquí) y uno no imputable puede justificar una prórroga; mentir sobre la causa es intentar "
        "librarse de la penalidad.\n\n"
        "**Criterio C5 (beneficio propio a costa de otro).** Desviar los operarios sin decirlo traslada a este cliente un coste (el retraso de "
        "{plural(retraso, 'semana')}) en beneficio de otro cliente de la empresa ({ahorro_otro_proyecto} k€ de ahorro allí), sin que lo haya aceptado.\n\n"
        "**Respuesta.** La causa falsa incumple **C1**; desviar los operarios ocultándolo, **C5**. La respuesta **no depende** de qué proyecto convenga más a "
        "la empresa (apartado 2: {neto_empresa:+.2f} k€): esa cifra dice qué conviene negociar con el cliente, no autoriza a ocultárselo ni a mentir sobre la "
        "causa. Lo que sí depende de las cifras es la decisión de negocio: {cesion}; cada semana de retraso cuesta "
        "{pen_dia_keur * 7:.2f} k€ de penalidad.''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Cliente (administración pública)', f'recibe la obra {plural(retraso, \"semana\")} tarde y, con la versión falsa, sin saber por qué ni poder reclamar la penalidad de {penalidad:.2f} k€'),\n"
        "     ('Cliente del otro proyecto', f'se beneficia de {ahorro_otro_proyecto} k€ de penalización evitada'),\n"
        "     ('Empresa', f'neto {neto_empresa:+.2f} k€ si cede {semanas_parada} semanas; con la mentira, riesgo de que se descubra y de perder futuras licitaciones'),\n"
        "     ('Proveedor de material', 'se le atribuye un retraso que no ha causado')],\n"
        "    [('Ceder y atribuir el retraso al proveedor (propuesta)', 'no: elude la penalidad con una causa falsa', 'no: no resiste una llamada del cliente al proveedor', 'no'),\n"
        "     ((f'Ceder sólo {plural(H, \"semana\")} (la holgura total), sin retraso' if H > 0 else f'No ceder {X} (es crítica) y buscar los operarios en una actividad con holgura'), 'sí', 'sí', 'sí'),\n"
        "     (f'Ceder {semanas_parada} semanas, avisar al cliente del retraso y asumir la penalidad', 'sí', 'sí', 'sí'),\n"
        "     (f'Reforzar {X} con una subcontrata para no retrasar', 'sí', 'sí', 'sí')],\n"
        "    f'''al director de operaciones, la tabla de opciones ({cesion}; ceder {semanas_parada} con {penalidad:.2f} k€ de "
        "penalidad y {plural(retraso, 'semana')} de retraso que hay que comunicar; reforzar {X}) y la condición de que el cliente conozca la causa real. Al cliente, "
        "si se cede más que la holgura: aviso por escrito del retraso previsto y de su causa y, si se quiere, una propuesta (compensación o recuperación en otra "
        "actividad).''')"))
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
    celdas.append(code(_AUX_ETICA))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas += apartado(1)
    celdas.append(md("Con cuatro o más licitadores y el precio como único criterio, la norma de referencia se aplica así: (1) media "
                     "de las bajas de todas las ofertas; (2) si alguna oferta es superior a esa media en más de 10 puntos (es decir, "
                     "su baja es más de 10 puntos **menor** que la media), se excluye y se recalcula la media con las restantes (y, si "
                     "quedan menos de tres, con las tres ofertas más bajas, que son las de mayor baja); (3) se presume anormalmente "
                     "baja la oferta cuya baja supera la media en más de 10 puntos."))
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
                       "media0 = float(np.mean(bajas))\n"
                       "excluidas = [b for b in bajas if b < media0 - 10]\n"
                       "precio_oferta = presupuesto * (1 - baja_propuesta / 100)\n"
                       "margen = precio_oferta - coste_estimado\n"
                       "baja_sin_perdida = 100 * (1 - coste_estimado / presupuesto)\n"
                       "paso2 = (f'Ninguna baja es menor que {media0:.2f} − 10 = {media0 - 10:.2f}: no se excluye ninguna oferta y la media se mantiene.' if not excluidas else\n"
                       "         f'Las bajas {lista_y([f\"{b:g}\" for b in excluidas])} son menores que {media0:.2f} − 10 = {media0 - 10:.2f}: se excluyen y la nueva media es **{media:.2f} %**.')\n"
                       "Markdown(f'''1. Media de las cuatro bajas: ({' + '.join(f\"{b:g}\" for b in bajas)}) / {len(bajas)} = **{media0:.2f} %**.\n"
                       "2. {paso2}\n"
                       "3. Umbral de presunción: {media:.2f} + 10 = **{umbral:.2f} %**. Nuestra baja, {baja_propuesta:g} %, "
                       "{'**supera** el umbral: se presume anormalmente baja' if baja_propuesta > umbral else 'no supera el umbral'}.\n\n"
                       "Precio ofertado = {presupuesto} · (1 − {baja_propuesta:g}/100) = **{precio_oferta:.1f} k€**; margen sobre el coste estimado = "
                       "{precio_oferta:.1f} − {coste_estimado:g} = **{margen:+.1f} k€** ({'pérdidas' if margen < 0 else 'beneficio'}). La mayor baja sin perder dinero sería "
                       "100 · (1 − {coste_estimado:g}/{presupuesto}) = {baja_sin_perdida:.1f} %.\n\n"
                       "**Respuesta.** Media de las bajas {media:.2f} %, umbral {umbral:.2f} %; la oferta del {baja_propuesta:g} % "
                       "{'incurre' if baja_propuesta > umbral else 'no incurre'} en la presunción y deja un margen de {margen:+.1f} k€ sobre el coste.''')"))
    celdas += apartado(2)
    celdas.append(code("camino, _ = camino_critico_sigma(proyecto, datos_proyecto['duration'], datos_proyecto['variance'])\n"
                       "p_plazo = float(norm.cdf(plazo_oferta, duracion, sigma))\n"
                       "plazo_80 = float(norm.ppf(0.8, duracion, sigma))\n"
                       "Markdown(f'''Camino crítico {' – '.join(camino)}: duración esperada {' + '.join(f\"{datos_proyecto.loc[a, 'duration']:g}\" for a in camino)} = "
                       "**{duracion:.2f}** semanas; {explica_sigma_ruta(camino, datos_proyecto['variance'])} semanas.\n\n"
                       "- {explica_probabilidad(duracion, sigma, plazo_oferta)}.\n"
                       "- Plazo con probabilidad 0,8: {explica_percentil(duracion, sigma, 0.8, 'semanas')}.\n\n"
                       "**Respuesta.** La probabilidad de ejecutar en {plazo_oferta} semanas es **{p_plazo:.3f}** ({porc(p_plazo)}); el plazo que se cumple con "
                       "probabilidad 0,8 es {plazo_80:.2f} semanas, es decir, se podrían ofertar **{int(np.ceil(plazo_80))} semanas** (con {int(np.floor(plazo_80))} "
                       "la probabilidad sería {norm.cdf(np.floor(plazo_80), duracion, sigma):.3f}, menor que 0,8).''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Norma** (RGLCAP art. 85; LCSP art. 149). La presunción de anormalidad obliga a la mesa a pedir al licitador que justifique su oferta; "
        "si la justificación no convence, la oferta se excluye. Con un margen de {margen:+.1f} k€ sobre el propio coste estimado, la justificación sería difícil: "
        "«ganar seguro» no es lo que la norma prevé, sino un trámite que puede acabar en exclusión.\n\n"
        "**Criterio C13 (compromisos de buena fe).** Ofertar {plazo_oferta} semanas con una probabilidad de cumplirlas de {p_plazo:.3f}, sabiendo que «el equipo "
        "dice que no llega» y contando con «renegociar con modificados y prórrogas», es asumir un compromiso que se sabe inalcanzable: no es de buena fe. "
        "**Criterio C1 (afirmación falsa).** La oferta afirma un plazo que quien la firma no espera cumplir. Que «siempre se haga» no cambia el criterio, y los "
        "modificados y las prórrogas tienen causas tasadas en la ley, entre las que no está haber ofertado lo que no se podía cumplir.\n\n"
        "**Respuesta.** La presunción de anormalidad abre un trámite de justificación y, si no convence, la **exclusión**. Ofertar un plazo inalcanzable "
        "(P = {p_plazo:.3f}) contando con renegociarlo incumple **C13** y **C1**.''')"))
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
    celdas.append(code(
        "baja_defendible = min(umbral, baja_sin_perdida)\n"
        "rubrica_caso(\n"
        "    [('Administración contratante', f'adjudica a una oferta con {margen:+.1f} k€ de margen y un {porc(1 - p_plazo, 0)} de probabilidad de no cumplir el plazo: riesgo de abandono, modificados y retrasos'),\n"
        "     ('Competidores', 'pierden una licitación ante una oferta que no se podía cumplir'),\n"
        "     ('Empresa y equipo', f'si gana, ejecuta con pérdidas ({margen:+.1f} k€) y penalidades probables; si la excluyen, pierde la licitación igualmente'),\n"
        "     ('Usuarios de la obra', 'reciben la obra más tarde de lo prometido')],\n"
        "    [(f'Baja del {baja_propuesta:g} % y plazo de {plazo_oferta} semanas (propuesta)', 'presunción de anormalidad: justificación y posible exclusión', 'no', 'no: la empresa no aceptaría un proveedor que le hiciera esto'),\n"
        "     (f'Baja ≤ {baja_defendible:.1f} % (bajo el umbral y sin pérdidas) y plazo de {int(np.ceil(plazo_80))} semanas (80 %)', 'sí', 'sí', 'sí'),\n"
        "     (f'Plazo de {duracion:.0f} semanas con la reserva declarada y su probabilidad', 'sí', 'sí', 'sí'),\n"
        "     ('Si el plazo corto es decisivo, ofertarlo con el coste de comprimir (lección 10) incluido en el precio', 'sí', 'sí', 'sí'),\n"
        "     ('No presentarse si no se puede competir sin pérdidas', 'sí', 'sí', 'sí')],\n"
        "    f'''una baja que no supere ni el umbral ({umbral:.2f} %) ni la que deja el precio por debajo del coste ({baja_sin_perdida:.1f} %), y el plazo de "
        "{int(np.ceil(plazo_80))} semanas, que se cumple con un 80 % de probabilidad. Al director comercial: la oferta temeraria puede quedar fuera en la mesa; si "
        "entra, los modificados no cubren un plazo que no se podía cumplir y la probabilidad del {porc(1 - p_plazo, 0)} de retraso se paga en penalidades y "
        "reputación. Prueba de publicidad: ¿firmaría la oferta si la mesa pudiera leer su correo?''')"))
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
                       "coste_horas = horas * tarifa / 1000\n"
                       "pA, pB = vg.index[0], vg.index[1]\n"
                       "nA, nB = pA.split()[0], pB.split()[0]"))
    celdas.append(code(_AUX_ETICA))
    celdas += apartado(1)
    celdas.append(md("Notación: BAC, presupuesto; PV, EV y AC, valor planificado, valor ganado y coste real en la fecha de control; "
                     "CV = EV − AC; CPI = EV / AC; EAC = BAC / CPI (coste final previsto si se mantiene el CPI); VAC = BAC − EAC. "
                     "En la fila «Total», EAC y VAC son la suma de las previsiones de los dos proyectos."))
    celdas.append(code('#| tbl-cap: "Indicadores con los datos reales"\n#| label: tbl-reales\n'
                       "reales = indicadores(vg)\n"
                       "reales.round(3)"))
    celdas.append(code(
        "Markdown(f'''{explica_indicadores(dict(PV=vg.loc[pA, 'PV'], EV=vg.loc[pA, 'EV'], AC=vg.loc[pA, 'AC']), float(vg.loc[pA, 'BAC']), '**' + pA + ':**')}\n\n"
        "{explica_indicadores(dict(PV=vg.loc[pB, 'PV'], EV=vg.loc[pB, 'EV'], AC=vg.loc[pB, 'AC']), float(vg.loc[pB, 'BAC']), '**' + pB + ':**')}\n\n"
        "Importe de las horas que se quieren reimputar: {horas} h × {tarifa} €/h = {horas * tarifa} € = **{coste_horas:.1f} k€**.\n\n"
        "**Respuesta.** {nA}: CV = {reales.loc[pA, 'CV']:.1f} k€, CPI = {reales.loc[pA, 'CPI']:.3f}, EAC = {reales.loc[pA, 'EAC = BAC / CPI']:.1f} k€ "
        "y VAC = {reales.loc[pA, 'VAC']:.1f} k€ (la subvención cubre hasta {vg.loc[pA, 'BAC']:g} k€). {nB}: CV = {reales.loc[pB, 'CV']:.1f} k€, "
        "CPI = {reales.loc[pB, 'CPI']:.3f}, EAC = {reales.loc[pB, 'EAC = BAC / CPI']:.1f} k€ y VAC = {reales.loc[pB, 'VAC']:.1f} k€. "
        "Las horas a reimputar valen {coste_horas:.1f} k€.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Indicadores tras reimputar las horas de A a B"\n#| label: tbl-reimputado\n'
                       "vg2 = vg.copy(); vg2.loc[pA, 'AC'] -= coste_horas; vg2.loc[pB, 'AC'] += coste_horas\n"
                       "reimputado = indicadores(vg2)\n"
                       "comparacion = pd.DataFrame({'AC real': reales['AC'], 'AC reimputado': reimputado['AC'], 'CPI real': reales['CPI'], 'CPI reimputado': reimputado['CPI'],\n"
                       "                            'EAC real': reales['EAC = BAC / CPI'], 'EAC reimputado': reimputado['EAC = BAC / CPI']})\n"
                       "comparacion.round(3)"))
    celdas.append(code(
        "horas_cpi_1 = (vg.loc[pA, 'AC'] - vg.loc[pA, 'EV']) * 1000 / tarifa\n"
        "Markdown(f'''- {nA}: AC = {vg.loc[pA, 'AC']:g} − {coste_horas:.1f} = {vg2.loc[pA, 'AC']:.1f} k€; CPI = {vg.loc[pA, 'EV']:g} / {vg2.loc[pA, 'AC']:.1f} = "
        "**{reimputado.loc[pA, 'CPI']:.3f}** (antes {reales.loc[pA, 'CPI']:.3f}, {reimputado.loc[pA, 'CPI'] - reales.loc[pA, 'CPI']:+.3f}).\n"
        "- {nB}: AC = {vg.loc[pB, 'AC']:g} + {coste_horas:.1f} = {vg2.loc[pB, 'AC']:.1f} k€; CPI = {vg.loc[pB, 'EV']:g} / {vg2.loc[pB, 'AC']:.1f} = "
        "**{reimputado.loc[pB, 'CPI']:.3f}** (antes {reales.loc[pB, 'CPI']:.3f}, {reimputado.loc[pB, 'CPI'] - reales.loc[pB, 'CPI']:+.3f}).\n"
        "- Total de la empresa: AC = {reales.loc['Total', 'AC']:.1f} k€ en los dos casos y CPI = {reales.loc['Total', 'EV']:.1f} / {reales.loc['Total', 'AC']:.1f} = "
        "**{reales.loc['Total', 'CPI']:.3f}** en los dos casos.\n"
        "- Para que {nA} saliera exactamente a CPI = 1 habría que mover AC − EV = {vg.loc[pA, 'AC']:g} − {vg.loc[pA, 'EV']:g} = "
        "{vg.loc[pA, 'AC'] - vg.loc[pA, 'EV']:.1f} k€, es decir, {vg.loc[pA, 'AC'] - vg.loc[pA, 'EV']:.1f} × 1000 / {tarifa} = **{horas_cpi_1:.0f} horas**.\n\n"
        "**Respuesta.** El CPI de {nA} pasa de {reales.loc[pA, 'CPI']:.3f} a {reimputado.loc[pA, 'CPI']:.3f} y el de {nB} de {reales.loc[pB, 'CPI']:.3f} a "
        "{reimputado.loc[pB, 'CPI']:.3f}; el de la empresa no cambia ({reales.loc['Total', 'CPI']:.3f}): sólo cambia a quién se carga el coste. Con "
        "{horas_cpi_1:.0f} horas {nA} quedaría a CPI = 1; las {horas} propuestas lo dejan {'por encima de 1' if horas > horas_cpi_1 else 'por debajo de 1'}.''')"))
    celdas += apartado(3)
    celdas.append(code(
        "Markdown(f'''**Norma** (Ley 38/2003, arts. 31 y 37). Sólo es subvencionable el gasto que responde de manera indubitada a la actividad subvencionada. "
        "Las {horas} horas se trabajaron en {nA}: son gasto de {nA}. Imputarlas a {nB} no hace que {nA} cueste menos ({reales.loc[pA, 'AC']:g} k€ reales), "
        "sino que la justificación presente {vg2.loc[pA, 'AC']:.1f} k€, una cifra falsa. La consecuencia es el **reintegro** de lo indebidamente justificado con "
        "intereses y, según el importe y la intención, una infracción o un delito de fraude de subvenciones.\n\n"
        "**Criterio C1 (afirmación falsa).** La justificación de la subvención afirma un coste de {nA} que quien la firma sabe que no es el real. "
        "**Criterio C5 (beneficio propio a costa de otro).** La empresa obtiene una ventaja (que {nA} salga «a presupuesto», CPI {reimputado.loc[pA, 'CPI']:.3f}) "
        "trasladando el riesgo al financiador público y falseando además la información de {nB} (su CPI baja de {reales.loc[pB, 'CPI']:.3f} a "
        "{reimputado.loc[pB, 'CPI']:.3f}); si {nB} se factura por costes, el perjudicado es también su cliente.\n\n"
        "**Respuesta.** No: el coste reimputado no es gasto subvencionable de {nB} ni deja de serlo de {nA}. La propuesta incumple la norma, **C1** y **C5**, y "
        "expone a la empresa al reintegro con intereses y a sanciones.''')"))
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
    celdas.append(code(
        "rubrica_caso(\n"
        "    [('Financiador público', f'recibe una justificación falsa: {nA} aparece con AC {vg2.loc[pA, \"AC\"]:.1f} k€ y CPI {reimputado.loc[pA, \"CPI\"]:.3f}, cuando su coste real es {vg.loc[pA, \"AC\"]:g} k€ y su previsión {reales.loc[pA, \"EAC = BAC / CPI\"]:.1f} k€ sobre {vg.loc[pA, \"BAC\"]:g}; no puede controlar el uso de los fondos'),\n"
        "     ('Empresa', f'no ahorra nada (el CPI total sigue en {reales.loc[\"Total\", \"CPI\"]:.3f}) y arriesga el reintegro, sanciones y futuras subvenciones'),\n"
        "     (f'Cliente y equipo de {nB}', f'su proyecto aparece con CPI {reimputado.loc[pB, \"CPI\"]:.3f} en vez de {reales.loc[pB, \"CPI\"]:.3f}'),\n"
        "     ('Quien firma los partes de horas y la justificación', 'firma un documento falso')],\n"
        "    [(f'Reimputar {horas} h de {nA} a {nB} (propuesta)', 'no: gasto no subvencionable justificado como si lo fuera', 'no: no resiste una auditoría que cruce los partes de horas', 'no: el financiador no lo aceptaría'),\n"
        "     (f'Asumir el sobrecoste de {nA} con fondos propios', 'sí', 'sí', 'sí'),\n"
        "     ('Pedir al órgano concedente una modificación o ampliación, si la causa lo justifica y las bases lo permiten', 'sí', 'sí', 'sí'),\n"
        "     (f'Reducir el alcance restante de {nA} de acuerdo con el financiador y revisar por qué se ha desviado', 'sí', 'sí', 'sí')],\n"
        "    f'''al director: la reimputación no ahorra nada a la empresa (CPI total {reales.loc['Total', 'CPI']:.3f} en los dos casos), convierte un sobrecoste de "
        "{-reales.loc[pA, 'VAC']:.1f} k€ previsto en un fraude y pone en riesgo la subvención entera; la propuesta es asumir el sobrecoste o pedir una modificación "
        "al financiador, y analizar la desviación de {nA}.''')"))
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
