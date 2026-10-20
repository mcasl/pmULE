"""Colección 17 — Análisis de viabilidad financiera: flujos de caja libre de un caso de estudio,
VAN, TIR y periodo de recuperación; vínculo con el proyecto planificado; comparación de
alternativas; sensibilidad, escenarios e inflación; financiación con deuda."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="17_viabilidad", prefijo="Viabilidad",
            titulo="Viabilidad financiera: flujos de caja, VAN, TIR y payback",
            descripcion="Construir los flujos de caja libre de un caso de estudio (inversión, ingresos, costes, "
                        "amortización, impuestos, circulante, valor residual) y evaluar con VAN, TIR y periodo de "
                        "recuperación; desembolsos ligados al Gantt del proyecto; alternativas excluyentes; "
                        "sensibilidad, escenarios e inflación; préstamo y flujo del accionista.")

# Funciones financieras que se emiten en el cuaderno
FINANZAS = (
    "def van(flujos, k):\n"
    "    '''Valor actual neto de una lista de flujos (el primero en t = 0) a la tasa k.'''\n"
    "    return float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))\n"
    "\n"
    "def tir(flujos):\n"
    "    '''Tasa que anula el VAN (por bisección entre −99 % y 1000 %).'''\n"
    "    from scipy.optimize import brentq\n"
    "    return float(brentq(lambda k: van(flujos, k), -0.99, 10.0))\n"
    "\n"
    "def tabla_descuento(flujos, k):\n"
    "    '''Año, flujo, factor de descuento, valor actual y valores acumulados (simple y descontado).'''\n"
    "    t = pd.DataFrame({'flujo': list(flujos)}, index=pd.Index(range(len(flujos)), name='año'))\n"
    "    t['(1 + k)^-t'] = [(1 + k) ** -i for i in t.index]\n"
    "    t['valor actual'] = t['flujo'] * t['(1 + k)^-t']\n"
    "    t['acumulado'] = t['flujo'].cumsum()\n"
    "    t['acumulado descontado'] = t['valor actual'].cumsum()\n"
    "    return t\n"
    "\n"
    "def payback(flujos, k=None):\n"
    "    '''Periodo de recuperación (años, con interpolación lineal dentro del año); descontado si se da k.'''\n"
    "    acum = 0.0\n"
    "    for t, f in enumerate(flujos):\n"
    "        v = f if k is None else f / (1 + k) ** t\n"
    "        if t > 0 and acum < 0 <= acum + v:\n"
    "            return float(t - 1 + (-acum) / v)\n"
    "        acum += v\n"
    "    return float('inf')\n"
    "\n"
    "def anualidad_equivalente(VAN, k, n):\n"
    "    '''Anualidad constante durante n años cuyo valor actual es VAN.'''\n"
    "    return float(VAN * k / (1 - (1 + k) ** -n))\n")

CASOS = [
    dict(tema="una planta fotovoltaica para autoconsumo industrial", unidad="MWh", producto="energía",
         precio=(90, 140), unidades=(900, 1800), cv=(5, 15), cf=(20, 60), inversion=(600, 1400), vida=(6, 8)),
    dict(tema="una línea de envasado automatizada", unidad="miles de unidades", producto="envases",
         precio=(2.0, 4.0), unidades=(150, 400), cv=(0.8, 1.8), cf=(60, 160), inversion=(500, 1200), vida=(5, 7)),
    dict(tema="una nave logística con reparto de última milla", unidad="miles de entregas", producto="entregas",
         precio=(3.0, 5.0), unidades=(200, 500), cv=(1.2, 2.5), cf=(100, 250), inversion=(700, 1500), vida=(5, 8)),
    dict(tema="una aplicación de gestión por suscripción", unidad="suscripciones", producto="suscripciones",
         precio=(0.24, 0.48), unidades=(1500, 4000), cv=(0.05, 0.12), cf=(150, 350), inversion=(400, 900), vida=(5, 6)),
    dict(tema="una flota de vehículos eléctricos de alquiler", unidad="miles de días de alquiler", producto="alquileres",
         precio=(45, 80), unidades=(8, 20), cv=(12, 25), cf=(80, 200), inversion=(800, 1600), vida=(5, 7)),
]


def _redondea(x, paso):
    return float(np.round(x / paso) * paso)


def _parametros(rng, caso):
    """Parámetros de un caso de estudio (k€ salvo precio/cv en €/unidad)."""
    def u(rango, paso):
        return _redondea(rng.uniform(*rango), paso)
    n = int(rng.integers(caso["vida"][0], caso["vida"][1] + 1))
    p = dict(
        inversion=u(caso["inversion"], 10),
        vida=n,
        precio=u(caso["precio"], 0.01 if caso["precio"][1] < 1 else 1),
        unidades_1=u(caso["unidades"], 10 if caso["unidades"][1] > 100 else 1),
        crecimiento=float(rng.choice([0.0, 0.03, 0.05, 0.08])),
        coste_variable=u(caso["cv"], 0.01 if caso["cv"][1] < 1 else 1),
        costes_fijos=u(caso["cf"], 5),
        valor_residual=0.0,
        circulante=float(rng.choice([0.05, 0.08, 0.10])),
        impuesto=0.25,
        tasa=float(rng.choice([0.06, 0.08, 0.10, 0.12])),
    )
    p["valor_residual"] = _redondea(p["inversion"] * float(rng.choice([0.05, 0.10, 0.15])), 5)
    return p


def _flujos(p):
    """Tabla de flujos de caja libre (k€) del modelo completo; devuelve (tabla, lista FCL)."""
    n = p["vida"]
    anos = list(range(0, n + 1))
    unidades = [0] + [p["unidades_1"] * (1 + p["crecimiento"]) ** (t - 1) for t in range(1, n + 1)]
    ventas = [u * p["precio"] for u in unidades]
    cvar = [u * p["coste_variable"] for u in unidades]
    cfij = [0] + [p["costes_fijos"]] * n
    ebitda = [v - a - b for v, a, b in zip(ventas, cvar, cfij)]
    amort = [0] + [(p["inversion"] - p["valor_residual"]) / n] * n
    bai = [e - a for e, a in zip(ebitda, amort)]
    imp = [max(0.0, b) * p["impuesto"] for b in bai]
    imp[0] = 0.0
    bdi = [b - i for b, i in zip(bai, imp)]
    circ = [p["circulante"] * v for v in ventas]
    circ_total = [p["circulante"] * ventas[1]] + circ[1:]       # el circulante del año 1 se aporta en t = 0
    var_circ = [circ_total[0]] + [circ_total[t] - circ_total[t - 1] for t in range(1, n + 1)]
    var_circ[n] = var_circ[n] - circ_total[n]                     # recuperación al final
    inv = [p["inversion"]] + [0.0] * n
    vr = [0.0] * n + [p["valor_residual"]]                         # vendido por su valor contable: sin impuesto
    fcl = [b + a - c - i + v for b, a, c, i, v in zip(bdi, amort, var_circ, inv, vr)]
    tabla = pd.DataFrame({
        "unidades": unidades, "ventas": ventas, "costes variables": cvar, "costes fijos": cfij,
        "EBITDA": ebitda, "amortización": amort, "BAI": bai, "impuesto": imp, "BDI": bdi,
        "+ amortización": amort, "− Δ circulante": [-x for x in var_circ], "− inversión": [-x for x in inv],
        "+ valor residual": vr, "FCL": fcl}, index=pd.Index(anos, name="año"))
    return tabla, fcl


def _van(flujos, k):
    return float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))


def _tir(flujos):
    try:
        return float(brentq(lambda k: _van(flujos, k), -0.99, 10.0))
    except ValueError:
        return float("nan")


def _payback(flujos, k=None):
    acum = 0.0
    for t, f in enumerate(flujos):
        v = f if k is None else f / (1 + k) ** t
        if t > 0 and acum < 0 <= acum + v:
            return t - 1 + (-acum) / v
        acum += v
    return float("inf")


def _buscar_caso(rng, acepta, max_intentos=2000):
    for _ in range(max_intentos):
        caso = CASOS[int(rng.integers(0, len(CASOS)))]
        p = _parametros(rng, caso)
        tabla, fcl = _flujos(p)
        res = dict(van=_van(fcl, p["tasa"]), tir=_tir(fcl), pb=_payback(fcl), pbd=_payback(fcl, p["tasa"]))
        if acepta(p, res):
            return caso, p, tabla, fcl, res
    raise RuntimeError("no se encontró un caso con los rasgos pedidos")


def _tabla_parametros(p, caso):
    filas = [
        ("inversión inicial (k€)", p["inversion"]),
        ("vida del proyecto (años)", p["vida"]),
        (f"precio (€ por {caso['unidad'][:-1] if caso['unidad'].endswith('s') else caso['unidad']})", p["precio"]),
        (f"{caso['unidad']} vendidas el año 1", p["unidades_1"]),
        ("crecimiento anual de las unidades", p["crecimiento"]),
        ("coste variable unitario (€)", p["coste_variable"]),
        ("costes fijos anuales (k€)", p["costes_fijos"]),
        ("valor residual al final (k€)", p["valor_residual"]),
        ("circulante necesario (fracción de las ventas del año)", p["circulante"]),
        ("tipo impositivo", p["impuesto"]),
        ("tasa de descuento", p["tasa"]),
    ]
    df = pd.DataFrame(filas, columns=["parametro", "valor"]).set_index("parametro")
    return df


PARAMS = ("p = datos_enunciado['valor'].to_dict()\n"
          "I0, n, precio = p['inversión inicial (k€)'], int(p['vida del proyecto (años)']), [v for c, v in p.items() if c.startswith('precio')][0]\n"
          "u1 = [v for c, v in p.items() if 'año 1' in c][0]\n"
          "g, cv, CF = p['crecimiento anual de las unidades'], p['coste variable unitario (€)'], p['costes fijos anuales (k€)']\n"
          "VR, w, t_imp, k = p['valor residual al final (k€)'], p['circulante necesario (fracción de las ventas del año)'], p['tipo impositivo'], p['tasa de descuento']")

MODELO = ("anos = list(range(0, n + 1))\n"
          "unidades = pd.Series([0] + [u1 * (1 + g) ** (t - 1) for t in range(1, n + 1)], index=anos)\n"
          "ventas = unidades * precio\n"
          "costes_variables = unidades * cv\n"
          "costes_fijos = pd.Series([0] + [CF] * n, index=anos)\n"
          "EBITDA = ventas - costes_variables - costes_fijos\n"
          "amortizacion = pd.Series([0] + [(I0 - VR) / n] * n, index=anos)\n"
          "BAI = EBITDA - amortizacion\n"
          "impuesto = BAI.clip(lower=0) * t_imp; impuesto[0] = 0\n"
          "BDI = BAI - impuesto\n"
          "circulante = pd.Series([w * ventas[1]] + [w * ventas[t] for t in range(1, n + 1)], index=anos)   # el del año 1 se aporta en t = 0\n"
          "variacion_circulante = circulante.diff().fillna(circulante[0]); variacion_circulante[n] -= circulante[n]   # se recupera al final\n"
          "inversion = pd.Series([I0] + [0] * n, index=anos)\n"
          "valor_residual = pd.Series([0] * n + [VR], index=anos)\n"
          "FCL = BDI + amortizacion - variacion_circulante - inversion + valor_residual\n"
          "tabla = pd.DataFrame({'unidades': unidades, 'ventas': ventas, 'costes variables': costes_variables, 'costes fijos': costes_fijos,\n"
          "                      'EBITDA': EBITDA, 'amortización': amortizacion, 'BAI': BAI, 'impuesto': impuesto, 'BDI': BDI,\n"
          "                      '+ amortización': amortizacion, '− Δ circulante': -variacion_circulante, '− inversión': -inversion,\n"
          "                      '+ valor residual': valor_residual, 'FCL': FCL})\n"
          "tabla.index.name = 'año'\n"
          "flujos = FCL.tolist()")


def _celdas_modelo(celdas, con_explicacion=True):
    if con_explicacion:
        celdas.append(md("Orden de cálculo, año a año (k€): ventas = unidades × precio; EBITDA = ventas − costes "
                         "variables − costes fijos; BAI = EBITDA − amortización (lineal sobre inversión − valor "
                         "residual); impuesto = tipo × BAI si BAI > 0; BDI = BAI − impuesto. El flujo de caja libre "
                         "suma al BDI la amortización (no es un pago) y resta las inversiones en activo fijo y en "
                         "circulante; en el último año se recuperan el circulante y el valor residual."))
    celdas.append(code(PARAMS))
    celdas.append(code('#| tbl-cap: "Flujos de caja libre del proyecto (k€)"\n#| label: tbl-flujos\n'
                       + MODELO + "\n"
                       "tabla.round(1).T"))


def _caso_base(rng):
    def acepta(p, r):
        return 0.03 * p["inversion"] < r["van"] < 0.8 * p["inversion"] and p["tasa"] + 0.02 < r["tir"] < 0.40 \
            and 2 < r["pb"] < p["vida"] - 1 and r["pbd"] < p["vida"]
    caso, p, tabla, fcl, res = _buscar_caso(rng, acepta)
    df = _tabla_parametros(p, caso)
    celdas = [md("# Enunciado"),
              md(f"Una empresa estudia invertir en {caso['tema']}. Los datos del caso se adjuntan (importes en "
                 "k€; precio y coste variable en € por unidad). La inversión se amortiza linealmente durante la "
                 "vida del proyecto hasta su valor residual, que se recupera al final por su valor contable. El "
                 "circulante necesario cada año es una fracción de las ventas de ese año, se aporta al principio "
                 "del año (el del primer año, en el momento 0) y se recupera íntegramente al final.")]
    celdas.append(preguntas([
        "Construya la tabla de flujos de caja libre año a año: ventas, costes, EBITDA, amortización, BAI, "
        "impuesto, BDI y flujo de caja libre (FCL).",
        "Calcule el VAN a la tasa de descuento dada, mostrando el factor de descuento y el valor actual de "
        "cada flujo. ¿Es viable el proyecto?",
        "Calcule la TIR. ¿Cómo se interpreta frente a la tasa de descuento?",
        "Calcule el periodo de recuperación simple y el descontado. ¿Por qué el descontado es mayor? ¿Qué "
        "información da el payback que no dan el VAN ni la TIR, y cuál omite?",
        "¿Qué supuesto del caso le parece más frágil? Compruebe cuánto cambia el VAN si las unidades del año 1 "
        "son un 15 % menores."]))
    celdas.append(code('#| tbl-cap: "Datos del caso"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df, indice="parametro")))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas += apartado(1)
    _celdas_modelo(celdas)
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Descuento de los flujos"\n#| label: tbl-descuento\n'
                       "descuento = tabla_descuento(flujos, k)\n"
                       "descuento.round(2)"))
    celdas.append(code("VAN = van(flujos, k)\n"
                       "Markdown(f'''VAN = Σ FCL_t / (1 + {k})^t = ''' + ' + '.join(f'{f:.1f}/{(1 + k) ** t:.3f}' for t, f in enumerate(flujos)) + f''' = **{VAN:.1f} k€**. "
                       "{(\"Es positivo: el proyecto devuelve la inversión, remunera el capital al \" + f\"{k:.0%}\" + \" y además crea \" + f\"{VAN:.0f}\" + \" k€ de valor; es viable.\") if VAN > 0 else \"Es negativo: no es viable a esa tasa.\"}''')"))
    celdas += apartado(3)
    celdas.append(code("TIR = tir(flujos)\n"
                       "comprobacion = van(flujos, TIR)\n"
                       "Markdown(f'TIR = {TIR:.2%} (VAN a esa tasa: {comprobacion:.4f} ≈ 0). Es la rentabilidad del proyecto; como supera la tasa de descuento ({k:.0%}), '\n"
                       "         'el VAN es positivo: las dos reglas coinciden en un proyecto aislado con flujos convencionales (un desembolso y luego cobros).')"))
    celdas.append(code('#| fig-cap: "Perfil del VAN frente a la tasa de descuento"\n#| label: fig-perfil\n'
                       "from matplotlib import pyplot as plt\n"
                       "ks = np.linspace(0, max(0.4, TIR + 0.1), 200)\n"
                       "fig, ax = plt.subplots(figsize=(7, 4))\n"
                       "ax.plot(ks * 100, [van(flujos, x) for x in ks]); ax.axhline(0, color='grey')\n"
                       "ax.axvline(k * 100, color='tab:green', linestyle='--', label=f'tasa de descuento {k:.0%}: VAN = {VAN:.0f}')\n"
                       "ax.axvline(TIR * 100, color='tab:red', linestyle=':', label=f'TIR = {TIR:.1%}')\n"
                       "ax.set_xlabel('tasa de descuento (%)'); ax.set_ylabel('VAN (k€)'); ax.legend(); plt.show()"))
    celdas += apartado(4)
    celdas.append(code("pb, pbd = payback(flujos), payback(flujos, k)\n"
                       "display(descuento[['flujo', 'acumulado', 'valor actual', 'acumulado descontado']].round(1))\n"
                       "Markdown(f'''Payback simple: el acumulado cambia de signo en el año {int(np.ceil(pb))}; interpolando, {pb:.2f} años. "
                       "Payback descontado: {pbd:.2f} años (mayor, porque los flujos descontados son menores y tardan más en cubrir la inversión). "
                       "El payback mide liquidez y riesgo de plazo (cuánto tiempo está el dinero expuesto), pero ignora los flujos posteriores a la recuperación "
                       "y, en su versión simple, el valor del tiempo; por eso se usa como criterio complementario, nunca único.''')"))
    celdas += apartado(5)
    celdas.append(code("u1_bajo = u1 * 0.85\n"
                       "def modelo(u1, precio, cv, CF, g=g, I0=I0, n=n, VR=VR, w=w, t_imp=t_imp):\n"
                       "    unidades = pd.Series([0] + [u1 * (1 + g) ** (t - 1) for t in range(1, n + 1)])\n"
                       "    ventas = unidades * precio; EBITDA = ventas - unidades * cv - pd.Series([0] + [CF] * n)\n"
                       "    amort = pd.Series([0] + [(I0 - VR) / n] * n); BAI = EBITDA - amort\n"
                       "    imp = BAI.clip(lower=0) * t_imp; imp[0] = 0\n"
                       "    circ = pd.Series([w * ventas[1]] + [w * ventas[t] for t in range(1, n + 1)])\n"
                       "    dcirc = circ.diff().fillna(circ[0]); dcirc[n] -= circ[n]\n"
                       "    return (BAI - imp + amort - dcirc - pd.Series([I0] + [0] * n) + pd.Series([0] * n + [VR])).tolist()\n"
                       "VAN_bajo = van(modelo(u1_bajo, precio, cv, CF), k)\n"
                       "Markdown(f'''Con un 15 % menos de unidades el VAN pasa de {VAN:.1f} a {VAN_bajo:.1f} k€ ({(VAN_bajo - VAN) / VAN:+.0%}): "
                       "{\"sigue siendo viable\" if VAN_bajo > 0 else \"deja de ser viable\"}. Las unidades vendidas (y el precio) son el supuesto más frágil: "
                       "son previsiones de mercado, mientras que la inversión y los costes fijos están casi cerrados por contrato.''')"))
    return celdas, f"{caso['tema']}: I0 = {p['inversion']:.0f} k€, {p['vida']} años, k = {p['tasa']:.0%}; VAN {res['van']:.0f}, TIR {res['tir']:.1%}"


def _vinculo_proyecto(rng):
    """Los desembolsos de la inversión salen del Gantt del proyecto; los ingresos empiezan al terminar."""
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 10 <= i["T"] <= 18 and (i["D"] >= 2).all()):
            return False
        BAC = pd.Series(rng.integers(2, 9, size=len(i["D"])) * 25, index=i["D"].index, name="BAC")
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        try:
            tabla, best, dur, avail = i["proyecto"].ackoff(i["D"], tope, ucr, reduction=3, report=False)
        except Exception:
            return False
        if len(best) < 2:
            return False
        estado.update(BAC=BAC, tope=tope, ucr=ucr, pasos=best)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T, cal = info["D"], int(info["T"]), info["calendario"]
    BAC, tope, ucr = estado["BAC"], estado["tope"], estado["ucr"]
    I0 = float(BAC.sum())
    k = float(rng.choice([0.08, 0.10, 0.12]))
    N = int(rng.choice([48, 60, 72]))
    # flujo mensual de explotación tal que el VAN sea claramente positivo pero sensible al retraso
    km = (1 + k) ** (1 / 12) - 1
    anualidad = I0 / ((1 - (1 + km) ** -N) / km) * float(rng.choice([1.3, 1.5, 1.8]))
    F = float(np.round(anualidad / 5) * 5)
    df = P.tabla_base(info, duration=D, BAC=BAC, cap_duration=tope, ucr=ucr)
    retraso = int(rng.integers(2, 5))
    celdas = [md("# Enunciado"),
              md("Una empresa va a ejecutar el proyecto de inversión cuyo cuadro de prelaciones, duraciones "
                 "(meses), presupuesto por actividad (BAC, k€), duración tope y coste unitario de reducción "
                 "(k€/mes) se adjunta. El coste de cada actividad se desembolsa uniformemente a lo largo de su "
                 f"duración. Cuando el proyecto termina, la instalación genera un flujo de caja neto de {F:.0f} k€ "
                 f"al mes durante {N} meses. La empresa descuenta al {k:.0%} anual.")]
    celdas.append(preguntas([
        "Calcule el calendario del proyecto y, a partir del Gantt de inicios más tempranos, el desembolso de "
        "cada mes de la fase de inversión.",
        "Construya el flujo de caja mensual completo (inversión y explotación) y calcule el VAN con la tasa "
        "mensual equivalente. ¿Es viable?",
        f"El proyecto sufre un retraso de {retraso} meses (los desembolsos se producen igual, pero los "
        "ingresos empiezan más tarde). ¿Cuánto VAN se pierde? ¿Cuánto vale, por tanto, cada mes de adelanto?",
        "Aplique Ackoff-Sasieni para reducir la duración del proyecto mes a mes hasta tres meses. Compare el "
        "sobrecoste de cada mes con la ganancia de VAN por adelantar los ingresos: ¿cuántos meses conviene "
        "acortar?"]))
    celdas.append(code('#| tbl-cap: "Datos del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"F, N, k, retraso = {F}, {N}, {k}, {retraso}\n"
                       "k_mensual = (1 + k) ** (1 / 12) - 1\n"
                       "I0 = float(datos_proyecto['BAC'].sum())"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Calendario del proyecto (meses)"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion = int(proyecto.duration(datos_proyecto['duration']))\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']]"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-grafo")
    celdas.append(md("El coste mensual de cada actividad es su BAC dividido por su duración. Escribiéndolo en cada "
                     "casilla del Gantt de inicios más tempranos, el desembolso de cada mes es la suma de su columna "
                     "(fila *Total*) y la fila *Acumulado* muestra lo invertido hasta ese mes; la columna *Total* de "
                     "cada actividad recupera su BAC."))
    celdas.append(code('#| fig-cap: "Gantt con el coste mensual de cada actividad (k€): la fila Total es el desembolso del mes"\n#| label: fig-ganttCostes\n'
                       "datos_gantt = datos_proyecto.copy()\n"
                       "datos_gantt['coste_mensual'] = datos_gantt['BAC'] / datos_gantt['duration']\n"
                       "gantt_costes, dibujo = proyecto.gantt(datos_gantt, 'duration', 'coste_mensual', total='ambas', acumulado=True, tikz=True,\n"
                       "                                      params=dict(inner_text_digits=1, **GANTT_PARAMS))\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("def desembolsos(cal, dur, duraciones=None):\n"
                       "    '''Desembolso de cada mes (1..T) repartiendo el BAC de cada actividad uniformemente.'''\n"
                       "    duraciones = datos_proyecto['duration'] if duraciones is None else duraciones\n"
                       "    pagos = pd.Series(0.0, index=range(1, int(dur) + 1))\n"
                       "    for a, fila in datos_proyecto.iterrows():\n"
                       "        es, d = int(cal.loc[a, 'inicio_mas_temprano']), int(duraciones[a])\n"
                       "        for m in range(es + 1, es + d + 1):\n"
                       "            pagos[m] += fila['BAC'] / d\n"
                       "    return pagos\n"
                       "pagos = desembolsos(calendario, duracion)\n"
                       "fila_total = gantt_costes.data.loc['Total'].drop('Total', errors='ignore').astype(float)\n"
                       "display(pd.DataFrame({'desembolso (k€)': pagos, 'fila Total del Gantt': fila_total.values[:len(pagos)]}).round(1).T)\n"
                       "Markdown(f'La fase de inversión dura {duracion} meses y suma {pagos.sum():.0f} k€ (= Σ BAC = {I0:.0f}).')"))
    celdas += apartado(2)
    celdas.append(md("Cada mes es un periodo: en los meses 1…T el flujo es −desembolso; en los T+1…T+N, +F. La "
                     "tasa mensual equivalente cumple $(1 + k_m)^{12} = 1 + k$."))
    celdas.append(code("def flujo_mensual(pagos, inicio_ingresos, sobrecoste=0.0):\n"
                       "    T = len(pagos)\n"
                       "    f = [0.0] + [-float(x) for x in pagos] + [0.0] * (inicio_ingresos - T) + [F] * N\n"
                       "    f[1] -= sobrecoste\n"
                       "    return f\n"
                       "flujos = flujo_mensual(pagos, duracion)\n"
                       "VAN = van(flujos, k_mensual)\n"
                       "resumen = pd.DataFrame({'mes': range(len(flujos)), 'flujo (k€)': flujos}).set_index('mes')\n"
                       "display(resumen.T.iloc[:, :duracion + 4].round(1))\n"
                       "Markdown(f'k mensual = (1 + {k})^(1/12) − 1 = {k_mensual:.4%}. VAN = {VAN:.1f} k€ (valor actual de los ingresos {van([0] * (duracion + 1) + [F] * N, k_mensual):.1f} '\n"
                       "         f'menos valor actual de los desembolsos {-van([0] + [-x for x in pagos], k_mensual):.1f}): {\"viable\" if VAN > 0 else \"no viable\"}.')"))
    celdas += apartado(3)
    celdas.append(code("VAN_retraso = van(flujo_mensual(pagos, duracion + retraso), k_mensual)\n"
                       "perdida = VAN - VAN_retraso\n"
                       "valor_mes = [VAN - van(flujo_mensual(pagos, duracion + m), k_mensual) for m in range(1, retraso + 1)]\n"
                       "display(pd.DataFrame({'meses de retraso': range(1, retraso + 1), 'VAN perdido (k€)': valor_mes}).set_index('meses de retraso').T.round(1))\n"
                       "Markdown(f'''Con {retraso} meses de retraso el VAN baja de {VAN:.1f} a {VAN_retraso:.1f} k€: se pierden {perdida:.1f} k€, "
                       "unos {perdida / retraso:.1f} k€ por mes. Cada mes de adelanto vale aproximadamente lo mismo (algo más los primeros, porque los ingresos "
                       "que se desplazan son los más cercanos): es el precio máximo que merece la pena pagar por acortar el proyecto un mes.''')"))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni (hasta tres meses)"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=3, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(code("filas = []\n"
                       "acumulado = 0.0\n"
                       "d = datos_proyecto['duration'].copy()\n"
                       "for paso, acts in sorted(pasos.items()):\n"
                       "    coste = float(sum(datos_proyecto.loc[a, 'ucr'] for a in acts)); acumulado += coste\n"
                       "    for a in acts: d[a] -= 1\n"
                       "    p2 = proyecto.copy(); cal2 = p2.calendar(d); dur2 = int(p2.duration(d))\n"
                       "    VAN2 = van(flujo_mensual(desembolsos(cal2, dur2, d), dur2, sobrecoste=acumulado), k_mensual)\n"
                       "    filas.append({'meses acortados': paso + 1, 'actividades': ', '.join(acts), 'sobrecoste del mes (k€)': coste,\n"
                       "                  'sobrecoste acumulado': acumulado, 'duración': dur2, 'VAN (k€)': VAN2, 'ganancia de VAN frente a no acortar': VAN2 - VAN})\n"
                       "decision = pd.DataFrame(filas).set_index('meses acortados')\n"
                       "display(decision.round(1))\n"
                       "mejor = decision['VAN (k€)'].idxmax() if decision['VAN (k€)'].max() > VAN else 0\n"
                       "Markdown(f'''Conviene acortar {mejor} mes(es): el VAN máximo es {max(VAN, decision[\"VAN (k€)\"].max()):.1f} k€. "
                       "Cada mes de reducción compensa mientras su sobrecoste (pagado al principio) sea menor que el VAN que gana adelantar un mes los ingresos; "
                       "como el sobrecoste de Ackoff-Sasieni crece paso a paso y el valor de un mes de adelanto es casi constante, hay un punto de corte.''')"))
    return celdas, f"{len(D)} actividades, {T} meses de inversión (I0 = {I0:.0f} k€), {F:.0f} k€/mes durante {N} meses, k = {k:.0%}"


def _alternativas(rng):
    """Dos proyectos excluyentes en los que VAN y TIR ordenan distinto; tasa de Fisher; vidas distintas."""
    for _ in range(5000):
        I_A = _redondea(rng.uniform(400, 900), 10)
        I_B = I_A
        n = int(rng.choice([5, 6]))
        # A: flujos crecientes (tarde); B: flujos decrecientes (pronto)
        base_A = I_A * float(rng.uniform(0.12, 0.2)); g_A = float(rng.uniform(0.15, 0.30))
        fl_A = [-I_A] + [_redondea(base_A * (1 + g_A) ** t, 5) for t in range(n)]
        base_B = I_B * float(rng.uniform(0.35, 0.5)); g_B = float(rng.uniform(-0.25, -0.12))
        fl_B = [-I_B] + [_redondea(base_B * (1 + g_B) ** t, 5) for t in range(n)]
        k = float(rng.choice([0.05, 0.06, 0.08]))
        vA, vB, tA, tB = _van(fl_A, k), _van(fl_B, k), _tir(fl_A), _tir(fl_B)
        if vA > vB > 0 and tB > tA > k + 0.01 and (vA - vB) > 0.03 * I_A:
            break
    else:
        raise RuntimeError("sin caso de conflicto VAN/TIR")
    dif = [a - b for a, b in zip(fl_A, fl_B)]
    fisher = _tir(dif)
    m = int(rng.choice([3, 4]))
    I_C = _redondea(I_A * 0.6, 10)
    fl_C = [-I_C] + [_redondea(I_C * float(rng.uniform(0.38, 0.48)), 5)] * m
    df = pd.DataFrame({"A": fl_A, "B": fl_B, "C": fl_C + [0] * (n - m)}, index=pd.Index(range(n + 1), name="year"))
    celdas = [md("# Enunciado"),
              md("Una empresa debe elegir **una sola** de tres inversiones excluyentes (flujos de caja en k€ al "
                 f"final de cada año; el año 0 es la inversión). A y B duran {n} años; C dura {m} y puede "
                 f"repetirse indefinidamente en las mismas condiciones. La tasa de descuento es del {k:.0%}.")]
    celdas.append(preguntas([
        "Calcule el VAN, la TIR y el payback de A y de B. ¿Qué proyecto prefiere cada criterio?",
        "Explique por qué VAN y TIR pueden discrepar entre proyectos excluyentes. Calcule la tasa de Fisher "
        "(la que iguala los VAN) a partir de los flujos diferenciales A − B y dibuje los dos perfiles de VAN.",
        "¿Para qué tasas de descuento se prefiere A y para cuáles B? ¿Cuál es la decisión correcta a la tasa "
        "dada y por qué manda el VAN?",
        "C tiene menor vida. Compare C con la mejor de A y B mediante la anualidad equivalente (o la cadena de "
        "repeticiones). ¿Cambia la decisión?"]))
    celdas.append(code('#| tbl-cap: "Flujos de caja de las alternativas (k€)"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df, indice="year")))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas.append(code(f"k, n, m = {k}, {n}, {m}\n"
                       "A, B = datos_enunciado['A'].tolist(), datos_enunciado['B'].tolist()\n"
                       "C = datos_enunciado['C'].tolist()[:m + 1]"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Criterios para A y B"\n#| label: tbl-criterios\n'
                       "criterios = pd.DataFrame({p: {'VAN (k€)': van(f, k), 'TIR': tir(f), 'payback (años)': payback(f), 'payback descontado': payback(f, k)}\n"
                       "                          for p, f in (('A', A), ('B', B))}).T\n"
                       "criterios.round(3)"))
    celdas.append(code("Markdown(f'El VAN prefiere {criterios[\"VAN (k€)\"].idxmax()} y la TIR prefiere {criterios[\"TIR\"].idxmax()}: los criterios discrepan.')"))
    celdas += apartado(2)
    celdas.append(md("La TIR es una medida *relativa* (rentabilidad por euro y año) y supone la reinversión de los "
                     "flujos a la propia TIR; el VAN es *absoluto* (euros de valor creado a la tasa de descuento). "
                     "Cuando los proyectos tienen distinto perfil temporal (A cobra tarde, B cobra pronto) sus "
                     "perfiles de VAN se cruzan: a la tasa de Fisher ambos valen lo mismo."))
    celdas.append(code("diferencial = [a - b for a, b in zip(A, B)]\n"
                       "fisher = tir(diferencial)\n"
                       "display(pd.DataFrame({'A': A, 'B': B, 'A − B': diferencial}, index=pd.Index(range(n + 1), name='año')).T)\n"
                       "Markdown(f'Tasa de Fisher: TIR de los flujos A − B = {fisher:.2%}; a esa tasa VAN_A = {van(A, fisher):.1f} = VAN_B = {van(B, fisher):.1f}.')"))
    celdas.append(code('#| fig-cap: "Perfiles de VAN de A y B"\n#| label: fig-perfiles\n'
                       "from matplotlib import pyplot as plt\n"
                       "ks = np.linspace(0, 0.4, 200)\n"
                       "fig, ax = plt.subplots(figsize=(7, 4))\n"
                       "for nombre, f in (('A', A), ('B', B)):\n"
                       "    ax.plot(ks * 100, [van(f, x) for x in ks], label=f'{nombre} (TIR {tir(f):.1%})')\n"
                       "ax.axhline(0, color='grey'); ax.axvline(fisher * 100, color='grey', linestyle=':', label=f'Fisher {fisher:.1%}')\n"
                       "ax.axvline(k * 100, color='tab:green', linestyle='--', label=f'tasa de descuento {k:.0%}')\n"
                       "ax.set_xlabel('tasa (%)'); ax.set_ylabel('VAN (k€)'); ax.legend(); plt.show()"))
    celdas += apartado(3)
    celdas.append(code("Markdown(f'''Para tasas por debajo de la de Fisher ({fisher:.1%}) se prefiere A (sus flujos tardíos valen más cuando el descuento es suave); "
                       "por encima, B. A la tasa dada ({k:.0%} < {fisher:.1%}) la decisión correcta es **A**, la de mayor VAN: el objetivo es maximizar el valor "
                       "creado en euros, y la TIR, al ser relativa e implicar reinversión a la propia TIR, no compara bien proyectos de distinto perfil.''')"))
    celdas += apartado(4)
    celdas.append(code("VAN_C = van(C, k)\n"
                       "AE = {'A': anualidad_equivalente(van(A, k), k, n), 'B': anualidad_equivalente(van(B, k), k, n), 'C': anualidad_equivalente(VAN_C, k, m)}\n"
                       "# cadena de repeticiones de C hasta igualar la vida de A (mínimo común múltiplo)\n"
                       "import math\n"
                       "L = math.lcm(n, m)\n"
                       "VAN_C_cadena = sum(VAN_C / (1 + k) ** (j * m) for j in range(L // m))\n"
                       "VAN_A_cadena = sum(van(A, k) / (1 + k) ** (j * n) for j in range(L // n))\n"
                       "display(pd.DataFrame({'VAN (k€)': {'A': van(A, k), 'B': van(B, k), 'C': VAN_C}, 'vida': {'A': n, 'B': n, 'C': m}, 'anualidad equivalente (k€/año)': AE}).round(2))\n"
                       "Markdown(f'''El VAN de C ({VAN_C:.1f}) no es comparable directamente con el de A ({van(A, k):.1f}) porque dura {m} años en vez de {n}. "
                       "La anualidad equivalente convierte cada VAN en una renta anual constante durante su vida: {\", \".join(f\"{p}: {v:.1f}\" for p, v in AE.items())} k€/año. "
                       "Comprobación con cadenas de repeticiones hasta {L} años: A {VAN_A_cadena:.1f} frente a C {VAN_C_cadena:.1f} k€. "
                       "{\"C pasa a ser la mejor opción.\" if AE[\"C\"] > max(AE[\"A\"], AE[\"B\"]) else \"La decisión no cambia: sigue siendo \" + max(AE, key=AE.get) + \".\"}''')"))
    return celdas, f"A y B ({n} años) con conflicto VAN/TIR (Fisher {fisher:.1%}), C de {m} años; k = {k:.0%}"


def _sensibilidad(rng):
    def acepta(p, r):
        return 0.05 * p["inversion"] < r["van"] < 0.6 * p["inversion"] and p["tasa"] + 0.03 < r["tir"] < 0.35
    caso, p, tabla, fcl, res = _buscar_caso(rng, acepta)
    df = _tabla_parametros(p, caso)
    inflacion = float(rng.choice([0.02, 0.03, 0.04]))
    celdas = [md("# Enunciado"),
              md(f"Una empresa estudia invertir en {caso['tema']} (datos adjuntos; mismas hipótesis contables que "
                 "en el caso base: amortización lineal hasta el valor residual, circulante como fracción de las "
                 "ventas aportado al principio de cada año y recuperado al final, valor residual cobrado por su "
                 "valor contable). Antes de decidir, la dirección quiere saber *cuánto pueden equivocarse* las "
                 "previsiones sin que el proyecto deje de ser viable.")]
    celdas.append(preguntas([
        "Calcule los flujos de caja libre y el VAN del caso base.",
        "Umbrales de rentabilidad: ¿qué precio mínimo, qué volumen mínimo del año 1 y qué coste variable "
        "máximo hacen el VAN igual a cero (manteniendo lo demás)? Exprese cada uno como variación porcentual "
        "respecto al caso base. ¿A qué variable es más sensible el proyecto?",
        "Escenarios: pesimista (precio −10 %, unidades −10 %, costes fijos +10 %), base y optimista (precio "
        "+10 %, unidades +10 %, costes fijos −10 %). VAN y TIR de cada uno.",
        f"Los datos están en euros constantes. Si se espera una inflación del {inflacion:.0%} anual que afecta "
        "por igual a precios y costes, ¿cómo deben construirse los flujos en euros corrientes y qué tasa "
        "nominal hay que usar para que el VAN no cambie? Compruébelo (relación de Fisher)."]))
    celdas.append(code('#| tbl-cap: "Datos del caso"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df, indice="parametro")))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas += apartado(1)
    _celdas_modelo(celdas, con_explicacion=False)
    celdas.append(code("def modelo(precio=precio, u1=u1, cv=cv, CF=CF, g=g, I0=I0, n=n, VR=VR, w=w, t_imp=t_imp, inflacion=0.0):\n"
                       "    '''Flujos de caja libre del modelo; con inflación, en euros corrientes (precios y costes crecen cada año).'''\n"
                       "    unidades = pd.Series([0] + [u1 * (1 + g) ** (t - 1) for t in range(1, n + 1)])\n"
                       "    infl = pd.Series([(1 + inflacion) ** t for t in range(n + 1)])\n"
                       "    ventas = unidades * precio * infl; EBITDA = ventas - unidades * cv * infl - pd.Series([0] + [CF] * n) * infl\n"
                       "    amort = pd.Series([0] + [(I0 - VR) / n] * n); BAI = EBITDA - amort\n"
                       "    imp = BAI.clip(lower=0) * t_imp; imp[0] = 0\n"
                       "    circ = pd.Series([w * ventas[1]] + [w * ventas[t] for t in range(1, n + 1)])\n"
                       "    dcirc = circ.diff().fillna(circ[0]); dcirc[n] -= circ[n]\n"
                       "    return (BAI - imp + amort - dcirc - pd.Series([I0] + [0] * n) + pd.Series([0] * n + [VR])).tolist()\n"
                       "VAN = van(modelo(), k)\n"
                       "Markdown(f'VAN del caso base: {VAN:.1f} k€ (TIR {tir(modelo()):.2%}).')"))
    celdas += apartado(2)
    celdas.append(md("Cada umbral se obtiene resolviendo VAN(x) = 0 en la variable x (bisección), con las demás "
                     "fijas. La sensibilidad se lee en la variación porcentual que anula el VAN: cuanto menor, "
                     "más frágil es el proyecto frente a esa variable."))
    celdas.append(code("from scipy.optimize import brentq\n"
                       "precio_umbral = brentq(lambda x: van(modelo(precio=x), k), precio * 0.2, precio * 2)\n"
                       "u1_umbral = brentq(lambda x: van(modelo(u1=x), k), u1 * 0.2, u1 * 2)\n"
                       "cv_umbral = brentq(lambda x: van(modelo(cv=x), k), cv * 0.2, max(cv * 5, precio * 0.99))\n"
                       "umbrales = pd.DataFrame({'caso base': [precio, u1, cv], 'umbral (VAN = 0)': [precio_umbral, u1_umbral, cv_umbral]},\n"
                       "                        index=['precio', 'unidades año 1', 'coste variable'])\n"
                       "umbrales['variación admisible'] = umbrales['umbral (VAN = 0)'] / umbrales['caso base'] - 1\n"
                       "display(umbrales.round(3))\n"
                       "mas_sensible = umbrales['variación admisible'].abs().idxmin()\n"
                       "Markdown(f'La variable más sensible es **{mas_sensible}**: basta una variación del {umbrales.loc[mas_sensible, \"variación admisible\"]:+.1%} para anular el VAN.')"))
    celdas.append(code('#| fig-cap: "VAN frente a la variación de cada variable (diagrama de araña)"\n#| label: fig-arana\n'
                       "from matplotlib import pyplot as plt\n"
                       "xs = np.linspace(-0.3, 0.3, 25)\n"
                       "fig, ax = plt.subplots(figsize=(7, 4))\n"
                       "ax.plot(xs * 100, [van(modelo(precio=precio * (1 + x)), k) for x in xs], label='precio')\n"
                       "ax.plot(xs * 100, [van(modelo(u1=u1 * (1 + x)), k) for x in xs], label='unidades')\n"
                       "ax.plot(xs * 100, [van(modelo(cv=cv * (1 + x)), k) for x in xs], label='coste variable')\n"
                       "ax.plot(xs * 100, [van(modelo(CF=CF * (1 + x)), k) for x in xs], label='costes fijos')\n"
                       "ax.plot(xs * 100, [van(modelo(I0=I0 * (1 + x)), k) for x in xs], label='inversión')\n"
                       "ax.axhline(0, color='grey'); ax.set_xlabel('variación respecto al caso base (%)'); ax.set_ylabel('VAN (k€)'); ax.legend(); plt.show()"))
    celdas += apartado(3)
    celdas.append(code("escenarios = {'pesimista': dict(precio=precio * 0.9, u1=u1 * 0.9, CF=CF * 1.1), 'base': {}, 'optimista': dict(precio=precio * 1.1, u1=u1 * 1.1, CF=CF * 0.9)}\n"
                       "tabla_esc = pd.DataFrame({e: {'VAN (k€)': van(modelo(**kw), k), 'TIR': tir(modelo(**kw))} for e, kw in escenarios.items()}).T\n"
                       "display(tabla_esc.round(3))\n"
                       "Markdown(f'''El escenario pesimista {\"mantiene\" if tabla_esc.loc[\"pesimista\", \"VAN (k€)\"] > 0 else \"destruye\"} la viabilidad (VAN {tabla_esc.loc[\"pesimista\", \"VAN (k€)\"]:.1f}). "
                       "Los escenarios combinan desviaciones simultáneas, que es lo que ocurre en la práctica (un mal año afecta a precio y volumen a la vez); "
                       "el análisis de una variable cada vez las subestima.''')"))
    celdas += apartado(4)
    celdas.append(code(f"inflacion = {inflacion}\n"
                       "k_nominal = (1 + k) * (1 + inflacion) - 1\n"
                       "flujos_corrientes = modelo(inflacion=inflacion)\n"
                       "VAN_corriente = van(flujos_corrientes, k_nominal)\n"
                       "VAN_corriente_mal = van(flujos_corrientes, k)\n"
                       "display(pd.DataFrame({'euros constantes': modelo(), 'euros corrientes': flujos_corrientes}, index=pd.Index(range(n + 1), name='año')).T.round(1))\n"
                       "Markdown(f'''Relación de Fisher: (1 + k_nominal) = (1 + k_real)(1 + π) → k_nominal = (1 + {k})(1 + {inflacion}) − 1 = {k_nominal:.2%}. "
                       "Flujos corrientes descontados a la tasa nominal: VAN = {VAN_corriente:.1f} k€, frente a {VAN:.1f} con flujos constantes y tasa real. "
                       "No coinciden porque no todo crece con la inflación: la amortización (y su ahorro fiscal) y el valor residual son importes nominales fijos, y el "
                       "circulante crece con las ventas nominales; la inflación perjudica al proyecto vía impuestos y circulante. "
                       "El error grave sería mezclar: flujos corrientes con tasa real daría {VAN_corriente_mal:.1f} k€, un VAN inflado.''')"))
    return celdas, f"{caso['tema']}: umbrales de precio, volumen y coste variable; escenarios; inflación {inflacion:.0%}"


def _financiacion(rng):
    def acepta(p, r):
        return 0.05 * p["inversion"] < r["van"] < 0.6 * p["inversion"] and p["tasa"] + 0.03 < r["tir"] < 0.30
    caso, p, tabla, fcl, res = _buscar_caso(rng, acepta)
    df = _tabla_parametros(p, caso)
    alfa = float(rng.choice([0.5, 0.6, 0.7]))
    i = float(rng.choice([0.04, 0.05, 0.06]))
    n_p = int(min(p["vida"], rng.choice([4, 5])))
    ke = float(np.round(p["tasa"] + 0.04, 2))
    i_alto = float(np.round(res["tir"] / (1 - p["impuesto"]) + 0.02, 2))   # coste neto de la deuda por encima de la TIR
    celdas = [md("# Enunciado"),
              md(f"Una empresa estudia invertir en {caso['tema']} (datos adjuntos; mismas hipótesis que en el "
                 f"caso base). Financiaría el {alfa:.0%} de la inversión inicial con un préstamo al {i:.0%} "
                 f"anual, amortizable en {n_p} años por cuotas constantes (sistema francés); el resto lo aportan "
                 f"los socios, que exigen una rentabilidad del {ke:.0%}. Los intereses son deducibles en el "
                 "impuesto de sociedades.")]
    celdas.append(preguntas([
        "Calcule el flujo de caja libre del proyecto (sin deuda), su VAN a la tasa dada y su TIR.",
        "Construya el cuadro de amortización del préstamo: cuota, intereses, amortización de capital y "
        "capital vivo de cada año.",
        "Construya el flujo de caja del accionista: FCL + préstamo recibido − cuota + ahorro fiscal de los "
        "intereses. Calcule su VAN a la rentabilidad exigida por los socios y su TIR.",
        "Compare la TIR del proyecto con la del accionista y explique el efecto apalancamiento. ¿Qué "
        f"ocurriría si el tipo del préstamo fuera del {i_alto:.0%} (hipótesis extrema, para ver el efecto)? Recalcule."]))
    celdas.append(code('#| tbl-cap: "Datos del caso"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df, indice="parametro")))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas += apartado(1)
    _celdas_modelo(celdas, con_explicacion=False)
    celdas.append(code("VAN, TIR = van(flujos, k), tir(flujos)\n"
                       "Markdown(f'VAN del proyecto = {VAN:.1f} k€ al {k:.0%}; TIR del proyecto = {TIR:.2%}.')"))
    celdas += apartado(2)
    celdas.append(md("Cuota constante $C = P \\cdot \\dfrac{i}{1 - (1 + i)^{-n_p}}$; cada año los intereses son "
                     "$i$ por el capital vivo al inicio y la amortización es la cuota menos los intereses."))
    celdas.append(code(f"alfa, i, n_p, ke = {alfa}, {i}, {n_p}, {ke}\n"
                       "def cuadro_prestamo(P, i, n_p):\n"
                       "    C = P * i / (1 - (1 + i) ** -n_p)\n"
                       "    vivo, filas = P, []\n"
                       "    for t in range(1, n_p + 1):\n"
                       "        interes = vivo * i; amort = C - interes; vivo -= amort\n"
                       "        filas.append({'año': t, 'cuota': C, 'intereses': interes, 'amortización': amort, 'capital vivo': max(vivo, 0)})\n"
                       "    return pd.DataFrame(filas).set_index('año')\n"
                       "prestamo = alfa * I0\n"
                       "cuadro = cuadro_prestamo(prestamo, i, n_p)\n"
                       "display(Markdown(f'Préstamo = {alfa:.0%} × {I0:.0f} = {prestamo:.1f} k€; cuota = {prestamo:.1f} × {i} / (1 − (1 + {i})^−{n_p}) = {cuadro[\"cuota\"].iloc[0]:.2f} k€.'))\n"
                       "cuadro.round(2)"))
    celdas += apartado(3)
    celdas.append(code("def flujo_accionista(cuadro, prestamo):\n"
                       "    f = pd.Series(flujos, index=range(n + 1)).astype(float)\n"
                       "    f[0] += prestamo\n"
                       "    for t, fila in cuadro.iterrows():\n"
                       "        f[t] += -fila['cuota'] + t_imp * fila['intereses']\n"
                       "    return f\n"
                       "fa = flujo_accionista(cuadro, prestamo)\n"
                       "detalle = pd.DataFrame({'FCL proyecto': flujos, '+ préstamo': [prestamo] + [0] * n,\n"
                       "                        '− cuota': [0] + [-cuadro['cuota'].get(t, 0) for t in range(1, n + 1)],\n"
                       "                        '+ ahorro fiscal intereses': [0] + [t_imp * cuadro['intereses'].get(t, 0) for t in range(1, n + 1)],\n"
                       "                        'flujo del accionista': fa.tolist()}, index=pd.Index(range(n + 1), name='año'))\n"
                       "display(detalle.round(1).T)\n"
                       "VAN_acc, TIR_acc = van(fa.tolist(), ke), tir(fa.tolist())\n"
                       "Markdown(f'VAN para el accionista (al {ke:.0%}) = {VAN_acc:.1f} k€; TIR del accionista = {TIR_acc:.2%}.')"))
    celdas += apartado(4)
    celdas.append(code(f"i_alto = {i_alto}\n"
                       "cuadro_alto = cuadro_prestamo(prestamo, i_alto, n_p)\n"
                       "fa_alto = flujo_accionista(cuadro_alto, prestamo)\n"
                       "coste_deuda_neto = i * (1 - t_imp); coste_alto_neto = i_alto * (1 - t_imp)\n"
                       "display(pd.DataFrame({'proyecto (sin deuda)': {'TIR': TIR}, f'accionista, préstamo al {i:.0%}': {'TIR': TIR_acc},\n"
                       "                      f'accionista, préstamo al {i_alto:.0%}': {'TIR': tir(fa_alto.tolist())}}).round(4))\n"
                       "Markdown(f'''La deuda cuesta {i:.0%} ({coste_deuda_neto:.2%} tras impuestos), menos que la rentabilidad del proyecto ({TIR:.2%}): la diferencia se la queda el accionista, "
                       "que ha puesto sólo el {1 - alfa:.0%} del capital, y su TIR sube a {TIR_acc:.2%}. Es el apalancamiento financiero positivo. "
                       "Con el préstamo al {i_alto:.0%} ({coste_alto_neto:.2%} neto, por encima de la TIR del proyecto) el efecto se invierte: la TIR del accionista "
                       "{\"cae a\" if tir(fa_alto.tolist()) < TIR else \"baja a\"} {tir(fa_alto.tolist()):.2%}{\", por debajo de la del proyecto\" if tir(fa_alto.tolist()) < TIR else \"\"}. "
                       "La condición es que la rentabilidad del proyecto supere el coste de la deuda después de impuestos. El apalancamiento amplifica en los dos sentidos, "
                       "y además concentra el riesgo: las cuotas se pagan aunque las ventas fallen.''')"))
    return celdas, f"{caso['tema']}: préstamo del {alfa:.0%} al {i:.0%} a {n_p} años; TIR proyecto {res['tir']:.1%}"


def _van_montecarlo(rng):
    """VAN con duraciones triangulares e ingreso incierto: distribución del VAN (monte_carlo_van),
    descomposición de efectos, tornado y valor de reducir cada incertidumbre."""
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 18):
            return False
        nombres = list(i["D"].index)
        BAC = pd.Series(rng.integers(2, 9, size=len(nombres)) * 25, index=nombres, name="BAC")
        a, m, b = P.tres_estimaciones_triangular(rng, i["D"])
        if ((b - a) < 2).sum() > len(nombres) // 2:
            return False
        k = float(rng.choice([0.08, 0.10, 0.12]))
        N = int(rng.choice([36, 48, 60]))
        pg = i["proyecto"]
        fijas = {x: ("uniform", float(d), float(d)) for x, d in i["D"].items()}
        v0 = pg.monte_carlo_van(fijas, BAC, 0.0, N, k, n_samples=1, seed=0).van_samples[0]
        v1 = pg.monte_carlo_van(fijas, BAC, 1.0, N, k, n_samples=1, seed=0).van_samples[0]
        coste, serie = -v0, v1 - v0
        F0 = coste / serie                                  # ingreso mensual que anula el VAN determinista
        F = float(np.ceil(F0 * float(rng.choice([1.10, 1.18, 1.28])) / 5) * 5)
        delta = float(np.round(0.3 * F / 5) * 5)
        dist = {x: ("triangular", float(a[x]), float(m[x]), float(b[x])) for x in nombres}
        res = pg.monte_carlo_van(dist, BAC, ("triangular", F - delta, F, F + delta), N, k, n_samples=3000, seed=0)
        van_det = F * serie - coste
        if van_det <= 0 or not (0.08 <= res.prob_negativo <= 0.45):
            return False
        if res.van_mean > van_det - 0.01 * abs(van_det):    # las duraciones inciertas deben bajar la media
            return False
        estado.update(BAC=BAC, a=a, m=m, b=b, k=k, N=N, F=F, delta=delta)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])),
                    max_intentos=6000)
    D, T, cal = info["D"], int(info["T"]), info["calendario"]
    BAC, a, m, b = estado["BAC"], estado["a"], estado["m"], estado["b"]
    k, N, F, delta = estado["k"], estado["N"], estado["F"], estado["delta"]
    I0 = float(BAC.sum())
    df = P.tabla_base(info, duration=D, a=a, m=m, b=b, BAC=BAC)
    celdas = [md("# Enunciado"),
              md("Una empresa va a ejecutar el proyecto de inversión cuyo cuadro de prelaciones se adjunta, con la duración "
                 "esperada de cada actividad (meses), sus tres estimaciones (optimista $a$, más probable $m$, pesimista $b$, "
                 "con distribución triangular) y su presupuesto (BAC, k€). El coste de cada actividad se desembolsa "
                 "uniformemente a lo largo de su duración. Cuando el proyecto termina, la instalación genera un flujo de caja "
                 f"neto mensual de {F:.0f} k€ durante {N} meses, pero ese flujo es incierto: entre {F - delta:.0f} y "
                 f"{F + delta:.0f} k€ (triangular simétrica). La empresa descuenta al {k:.0%} anual.")]
    celdas.append(preguntas([
        "Con las duraciones esperadas, calcule el calendario, el camino crítico y el VAN determinista del proyecto "
        "(flujo mensual, tasa mensual equivalente).",
        "Simule 5 000 réplicas: duraciones triangulares e ingreso mensual incierto. De cada réplica obtenga la duración "
        "del proyecto y su VAN. Dibuje los histogramas y calcule la media, la desviación típica, P(VAN < 0) y el percentil 5 %.",
        "¿Por qué la media del VAN simulado no coincide con el VAN determinista? Separe los dos efectos simulando "
        "por separado solo la incertidumbre de las duraciones y solo la del ingreso.",
        "¿Qué variables explican la variabilidad del VAN? Calcule la correlación de Spearman del VAN con cada duración y "
        "con el ingreso, y compárela con el índice de criticidad de cada actividad. ¿Hay alguna actividad cuya demora "
        "*mejore* el VAN? ¿Por qué?",
        "La empresa puede (a) encargar un estudio de mercado que fije el ingreso en su valor central, o (b) contratar a "
        "un subcontratista que fije la duración de la actividad más influyente en su valor esperado. Evalúe qué cambia "
        "con cada opción (y con ambas) en la media, la desviación y P(VAN < 0), y qué compra realmente cada una."]))
    celdas.append(code('#| tbl-cap: "Datos del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"F, delta, N, k = {F}, {delta}, {N}, {k}\n"
                       "k_mensual = (1 + k) ** (1 / 12) - 1\n"
                       "I0 = float(datos_proyecto['BAC'].sum())"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Calendario con las duraciones esperadas (meses)"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion = int(proyecto.duration(datos_proyecto['duration']))\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']]"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-grafo")
    celdas.append(code("pagos = pd.Series(0.0, index=range(1, duracion + 1))\n"
                       "for act, fila in datos_proyecto.iterrows():\n"
                       "    es, d = int(calendario.loc[act, 'inicio_mas_temprano']), int(fila['duration'])\n"
                       "    for mes in range(es + 1, es + d + 1):\n"
                       "        pagos[mes] += fila['BAC'] / d\n"
                       "flujos = [0.0] + [-float(x) for x in pagos] + [F] * N\n"
                       "VAN_det = van(flujos, k_mensual)\n"
                       "criticas = calendario.index[calendario['H_total'] == 0].tolist()\n"
                       "Markdown(f'Duración {duracion} meses; actividades críticas: {\", \".join(criticas)}. Inversión Σ BAC = {I0:.0f} k€. '\n"
                       "         f'Con k mensual = {k_mensual:.4%}, el VAN determinista es {VAN_det:.1f} k€.')"))
    celdas += apartado(2)
    celdas.append(code("distribuciones = {act: ('triangular', float(f['a']), float(f['m']), float(f['b'])) for act, f in datos_proyecto.iterrows()}\n"
                       "ingreso = ('triangular', F - delta, F, F + delta)\n"
                       "mc = proyecto.monte_carlo_van(distribuciones, datos_proyecto['BAC'], ingreso, N, k, n_samples=5000, seed=2025)\n"
                       "resumen = mc.summary().round(2)\n"
                       "resumen"))
    celdas.append(code('#| fig-cap: "Distribución de la duración del proyecto y del VAN (5 000 réplicas)"\n#| label: fig-histogramas\n'
                       "import matplotlib.pyplot as plt\n"
                       "fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6))\n"
                       "ejes[0].hist(mc.tau_samples, bins=range(int(mc.tau_samples.min()), int(mc.tau_samples.max()) + 2), color='#4c78a8', edgecolor='white')\n"
                       "ejes[0].axvline(duracion, color='k', ls='--', label=f'determinista ({duracion})'); ejes[0].set_xlabel('duración (meses)'); ejes[0].legend()\n"
                       "ejes[1].hist(mc.van_samples, bins=40, color='#4c78a8', edgecolor='white')\n"
                       "ejes[1].axvline(VAN_det, color='k', ls='--', label='VAN determinista'); ejes[1].axvline(0, color='#c44', label='VAN = 0')\n"
                       "ejes[1].set_xlabel('VAN (k€)'); ejes[1].legend(); plt.tight_layout()"))
    celdas.append(code("Markdown(f'La duración media es {mc.tau_samples.mean():.2f} meses frente a {duracion} del plan; el VAN medio es {mc.van_mean:.1f} k€ '\n"
                       "         f'(determinista {VAN_det:.1f}), con desviación {mc.van_std:.1f}. P(VAN < 0) = {mc.prob_negativo:.1%} y el percentil 5 % es {mc.cuantil(0.05):.1f} k€.')"))
    celdas += apartado(3)
    celdas.append(code("fijas = {act: ('uniform', float(f['duration']), float(f['duration'])) for act, f in datos_proyecto.iterrows()}\n"
                       "solo_duraciones = proyecto.monte_carlo_van(distribuciones, datos_proyecto['BAC'], F, N, k, n_samples=5000, seed=2025)\n"
                       "solo_ingreso = proyecto.monte_carlo_van(fijas, datos_proyecto['BAC'], ingreso, N, k, n_samples=5000, seed=2025)\n"
                       "filas = {'VAN determinista': (VAN_det, 0.0, float(VAN_det < 0))}\n"
                       "for nombre, r in {'solo duraciones inciertas': solo_duraciones, 'solo ingreso incierto': solo_ingreso, 'ambas': mc}.items():\n"
                       "    filas[nombre] = (r.van_mean, r.van_std, r.prob_negativo)\n"
                       "tabla = pd.DataFrame(filas, index=['VAN medio', 'desviación', 'P(VAN < 0)']).T\n"
                       "display(tabla.round(3))\n"
                       "ee_ing, ee_dur = 2 * solo_ingreso.van_std / np.sqrt(5000), 2 * solo_duraciones.van_std / np.sqrt(5000)\n"
                       "Markdown(f'La incertidumbre del ingreso (simétrica y lineal en el VAN) ensancha la distribución pero no mueve la media: '\n"
                       "         f'{solo_ingreso.van_mean:.1f} ± {ee_ing:.1f} (dos errores estándar de simulación) frente a {VAN_det:.1f}. Las duraciones sí la bajan, hasta '\n"
                       "         f'{solo_duraciones.van_mean:.1f} ± {ee_dur:.1f}: el proyecto termina cuando termina el último camino, así que la duración media es '\n"
                       "         f'{solo_duraciones.tau_samples.mean():.2f} > {duracion} (caminos casi críticos que se fusionan) y cada mes de retraso retrasa un flujo descontado. '\n"
                       "         'Es el mismo sesgo de la aproximación normal, ahora medido en k€.')"))
    celdas += apartado(4)
    celdas.append(code("tornado = mc.tornado()\n"
                       "criticidad = proyecto.monte_carlo(distribuciones, n_samples=5000, seed=2025).sensitivity_mean\n"
                       "comparacion = pd.DataFrame({'Spearman con el VAN': tornado['Spearman con el VAN']})\n"
                       "comparacion['índice de criticidad'] = [criticidad.get(v.split()[-1], np.nan) if v.startswith('duración') else np.nan for v in comparacion.index]\n"
                       "display(comparacion.round(3))\n"
                       "positivas = [v for v, r in tornado['Spearman con el VAN'].items() if r > 0.02 and v.startswith('duración')]\n"
                       "Markdown(f'Mandan el ingreso y las actividades con alto índice de criticidad (su retraso retrasa el cobro). '\n"
                       "         f'Con correlación positiva apreciable: {positivas or \"ninguna\"}. Una actividad con holgura que se alarga no retrasa el cobro y '\n"
                       "         'desplaza el pago de su BAC hacia meses más descontados, por lo que puede mejorar ligeramente el VAN; el efecto es pequeño.')"))
    celdas += apartado(5)
    celdas.append(code("influyente = tornado.index[[v.startswith('duración') for v in tornado.index]][0].split()[-1]\n"
                       "centrada = dict(distribuciones); centrada[influyente] = ('uniform', float(datos_proyecto.loc[influyente, 'duration']), float(datos_proyecto.loc[influyente, 'duration']))\n"
                       "opciones = {'sin hacer nada': mc,\n"
                       "            '(a) fijar el ingreso': proyecto.monte_carlo_van(distribuciones, datos_proyecto['BAC'], F, N, k, n_samples=5000, seed=2025),\n"
                       "            f'(b) fijar la duración de {influyente}': proyecto.monte_carlo_van(centrada, datos_proyecto['BAC'], ingreso, N, k, n_samples=5000, seed=2025),\n"
                       "            '(a) y (b)': proyecto.monte_carlo_van(centrada, datos_proyecto['BAC'], F, N, k, n_samples=5000, seed=2025)}\n"
                       "tabla_op = pd.DataFrame({n: (r.van_mean, r.van_std, r.prob_negativo, r.cvar(0.05)) for n, r in opciones.items()},\n"
                       "                        index=['VAN medio', 'desviación', 'P(VAN < 0)', 'CVaR 5 %']).T\n"
                       "display(tabla_op.round(3))\n"
                       "mejor_riesgo = tabla_op.drop(index=['sin hacer nada', '(a) y (b)'])['P(VAN < 0)'].idxmin()\n"
                       "Markdown(f'La actividad más influyente es {influyente}. Fijar el ingreso o una duración cambia poco el VAN medio '\n"
                       "         f'({tabla_op.loc[\"sin hacer nada\", \"VAN medio\"]:.1f} → entre {tabla_op[\"VAN medio\"].drop(\"sin hacer nada\").min():.1f} y {tabla_op[\"VAN medio\"].drop(\"sin hacer nada\").max():.1f}): '\n"
                       "         f'lo que compra es seguridad. La opción que más reduce P(VAN < 0) es {mejor_riesgo}. Si la empresa es neutral al riesgo, solo vale la pena pagar '\n"
                       "         'por lo que sube la media; si le preocupa la cola (P(VAN < 0), CVaR), se paga por reducir dispersión.')"))
    return celdas, f"{len(D)} actividades, {T} meses, I0 = {I0:.0f} k€, ingreso {F:.0f} ± {delta:.0f} k€/mes durante {N} meses, k = {k:.0%}; distribución del VAN"


def _recursos_van(rng):
    """Límite de operarios y VAN: el retraso de los ingresos, ¿compensa ampliar la cuadrilla?"""
    estado = {}

    def acepta(i):
        if not (0 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 24):
            return False
        nombres = list(i["D"].index)
        op = P.recursos(rng, nombres, hi=3).rename("operarios")
        L = int(op.max())
        BAC = pd.Series(rng.integers(2, 9, size=len(nombres)) * 25, index=nombres, name="BAC")
        k = float(rng.choice([0.08, 0.10, 0.12]))
        N = int(rng.choice([36, 48, 60]))
        df = P.tabla_base(i, duration=i["D"], operarios=op, BAC=BAC)
        pg = i["proyecto"]
        try:
            e0 = pg.van_recursos(df, "duration", ["operarios"], [L], BAC, 0.0, N, k, ampliaciones=1)
            e1 = pg.van_recursos(df, "duration", ["operarios"], [L], BAC, 1.0, N, k, ampliaciones=1)
        except Exception:
            return False
        d_lib, d_lim = e0["escenarios"].loc["sin límite", "duración"], e0["escenarios"].loc["con límite", "duración"]
        if not (1 <= d_lim - d_lib <= 3):
            return False
        if e0["ampliaciones"].loc[1, "duración"] >= d_lim:
            return False
        a = e0["escenarios"]["VAN"]
        b = e1["escenarios"]["VAN"] - a
        F = float(np.ceil((-a["sin límite"] / b["sin límite"]) * float(rng.choice([1.15, 1.25, 1.4])) / 5) * 5)
        r = pg.van_recursos(df, "duration", ["operarios"], [L], BAC, F, N, k, ampliaciones=1)
        mejora = float(r["ampliaciones"].loc[1, "VAN"] - r["ampliaciones"].loc[0, "VAN"])
        e = r["escenarios"]["VAN"]
        if mejora <= 15 or e["con límite"] <= 0 or e["sin límite"] - e["con límite"] < 20:
            return False
        c = float(np.round(mejora * float(rng.choice([0.5, 0.8, 1.3])) / d_lim * 2) / 2)
        if c <= 0:
            return False
        estado.update(df=df, L=L, k=k, N=N, F=F, c=c)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), max_intentos=6000)
    df, L, k, N, F, c = estado["df"], estado["L"], estado["k"], estado["N"], estado["F"], estado["c"]
    celdas = [md("# Enunciado"),
              md("Una empresa va a ejecutar el proyecto de inversión cuyo cuadro de prelaciones, duración (meses), operarios "
                 "necesarios y presupuesto (BAC, k€) de cada actividad se adjunta. El coste de cada actividad se desembolsa "
                 "uniformemente a lo largo de su duración. Cuando el proyecto termina, la instalación genera un flujo de caja "
                 f"neto de {F:.0f} k€ al mes durante {N} meses. La empresa descuenta al {k:.0%} anual y dispone como máximo "
                 f"de **{L} operarios** al mes. Contratar un operario más cuesta **{c:.1f} k€ por mes** mientras dure el proyecto.")]
    celdas.append(preguntas([
        "Calcule la duración y el VAN del proyecto sin tener en cuenta el límite de operarios.",
        "Aplique el algoritmo de asignación con el límite de operarios. ¿Cuánto se alarga el proyecto y cuánto cuesta en VAN? "
        "Separe el efecto sobre el VAN de los costes y el de los ingresos y explique cuál domina.",
        f"¿Compensa contratar un operario más (límite {L + 1})? Construya la tabla de duración y VAN al ampliar el límite "
        "en 0, 1, 2 y 3 operarios y compare la mejora del VAN con el coste de contratar (sin descontar, por simplicidad).",
        "Calcule el flujo mensual mínimo (umbral del VAN = 0) con y sin el límite. ¿Cuánto sube el umbral por culpa del límite?"]))
    celdas.append(code('#| tbl-cap: "Datos del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"F, N, k, L, coste_operario = {F}, {N}, {k}, {L}, {c}\n"
                       "BAC = datos_proyecto['BAC']"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico (sin límite de operarios)", label="fig-grafo")
    celdas.append(code('#| tbl-cap: "Calendario sin límite de recursos (meses)"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_libre = int(proyecto.duration(datos_proyecto['duration']))\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(md("El coste mensual de cada actividad es su BAC entre su duración; la fila *Total* del Gantt es el "
                     "desembolso de cada mes. Cada mes es un periodo: en los meses 1…T el flujo es −desembolso y en los "
                     "T+1…T+N, +F; la tasa mensual equivalente cumple $(1 + k_m)^{12} = 1 + k$."))
    celdas.append(code('#| fig-cap: "Gantt de inicios más tempranos con el coste mensual de cada actividad (k€)"\n#| label: fig-ganttCostes\n'
                       "datos_gantt = datos_proyecto.copy()\n"
                       "datos_gantt['coste_mensual'] = datos_gantt['BAC'] / datos_gantt['duration']\n"
                       "gantt_costes, dibujo = proyecto.gantt(datos_gantt, 'duration', 'coste_mensual', total='ambas', acumulado=True, tikz=True,\n"
                       "                                      params=dict(inner_text_digits=1, **GANTT_PARAMS))\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("def desembolsos(inicios, duraciones):\n"
                       "    '''Desembolso de cada mes repartiendo el BAC de cada actividad uniformemente desde su inicio (contado desde 0).'''\n"
                       "    T = int(max(inicios[a] + duraciones[a] for a in datos_proyecto.index))\n"
                       "    pagos = pd.Series(0.0, index=range(1, T + 1))\n"
                       "    for a, fila in datos_proyecto.iterrows():\n"
                       "        es, d = int(inicios[a]), int(duraciones[a])\n"
                       "        for m in range(es + 1, es + d + 1):\n"
                       "            pagos[m] += fila['BAC'] / d\n"
                       "    return pagos\n"
                       "def flujo_mensual(pagos):\n"
                       "    return [0.0] + [-float(x) for x in pagos] + [F] * N\n"
                       "k_mensual = (1 + k) ** (1 / 12) - 1\n"
                       "pagos_libre = desembolsos(calendario['inicio_mas_temprano'], datos_proyecto['duration'])\n"
                       "flujos_libre = flujo_mensual(pagos_libre)\n"
                       "VAN_libre = van(flujos_libre, k_mensual)\n"
                       "display(pd.DataFrame({'desembolso (k€)': pagos_libre}).T.round(1))\n"
                       "Markdown(f'k mensual = (1 + {k})^(1/12) − 1 = {k_mensual:.4%}. Sin límite: {duracion_libre} meses de inversión ({pagos_libre.sum():.0f} k€ = Σ BAC) y '\n"
                       "         f'VAN = {VAN_libre:.1f} k€ (valor actual de los ingresos {van([0.0] * (duracion_libre + 1) + [F] * N, k_mensual):.1f} menos el de los desembolsos '\n"
                       "         f'{-van([0.0] + [-float(x) for x in pagos_libre], k_mensual):.1f}).')"))
    celdas.append(code('#| tbl-cap: "VAN sin y con límite de operarios (van_recursos)"\n#| label: tbl-escenarios\n'
                       "res = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, F, N, k, ampliaciones=3)\n"
                       "escenarios = res['escenarios'].round(2)\n"
                       "escenarios"))
    celdas.append(code("Markdown(f'La librería da lo mismo: sin límite {escenarios.loc[\"sin límite\", \"duración\"]:.0f} meses y VAN {escenarios.loc[\"sin límite\", \"VAN\"]:.1f} k€ '\n"
                       "         f'(VAN de los ingresos {escenarios.loc[\"sin límite\", \"VAN ingresos\"]:.1f}, de los costes {escenarios.loc[\"sin límite\", \"VAN costes\"]:.1f}).')"))
    celdas += apartado(2)
    celdas.append(code('#| fig-cap: "Carga de operarios con los inicios más tempranos frente al límite"\n#| label: fig-cargasInicial\n'
                       "gantt_op, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_op, limite_maximo_recursos=[L])"))
    celdas.append(md("El algoritmo de asignación recorre los meses; cada actividad disponible (predecesoras terminadas) "
                     "empieza, por orden de prioridad (menor inicio más tardío), si caben sus operarios durante toda su "
                     "duración; si no, espera. Traza mes a mes y Gantt resultante (filas 💤: esperas):"))
    celdas.append(code("res_lim = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', ['operarios'], [L], report=True, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt con el límite de operarios"\n#| label: fig-ganttLimite\n'
                       "pinta_tikz(res_lim.dibujos['operarios'])"))
    celdas.append(code('#| tbl-cap: "Inicio de cada actividad sin y con límite"\n#| label: tbl-iniciosLimite\n'
                       "pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res_lim.inicios.reindex(datos_proyecto.index),\n"
                       "              'espera (meses)': res_lim.esperas.reindex(datos_proyecto.index).fillna(0).astype(int)}).T"))
    celdas.append(code('#| tbl-cap: "Desembolso mensual sin y con límite (k€)"\n#| label: tbl-desembolsosLimite\n'
                       "pagos_lim = desembolsos(res_lim.inicios, datos_proyecto['duration'])\n"
                       "flujos_lim = flujo_mensual(pagos_lim)\n"
                       "VAN_lim = van(flujos_lim, k_mensual)\n"
                       "pd.DataFrame({'sin límite': pagos_libre, 'con límite': pagos_lim}).fillna(0).T.round(1)"))
    celdas.append(code("dif = escenarios.loc['con límite'] - escenarios.loc['sin límite']\n"
                       "Markdown(f'Con {L} operarios el proyecto dura {res_lim.duracion} meses (+{res_lim.duracion - duracion_libre}): los ingresos empiezan en el mes {res_lim.duracion + 1} en vez del {duracion_libre + 1}. '\n"
                       "         f'VAN = {VAN_lim:.1f} k€ (la librería: {escenarios.loc[\"con límite\", \"VAN\"]:.1f}) frente a {VAN_libre:.1f} sin límite: {dif[\"VAN\"]:.1f} k€. '\n"
                       "         f'Efecto en los ingresos: {dif[\"VAN ingresos\"]:.1f} k€ (se cobran más tarde); en los costes: {-dif[\"VAN costes\"]:.1f} k€ '\n"
                       "         '(las actividades que esperan se pagan más tarde y el coste se descuenta más). El retraso de los ingresos domina.')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Duración y VAN al ampliar el límite de operarios"\n#| label: tbl-ampliaciones\n'
                       "amp = res['ampliaciones'].copy()\n"
                       "amp['coste de contratar'] = amp.index * coste_operario * amp['duración']\n"
                       "amp['VAN neto de contratar'] = amp['VAN'] - amp['coste de contratar']\n"
                       "amp.round(2)"))
    celdas.append(code("mejor = amp['VAN neto de contratar'].idxmax()\n"
                       "Markdown(f'La mejor opción es ampliar el límite en {mejor} operario(s): VAN neto {amp.loc[mejor, \"VAN neto de contratar\"]:.1f} k€ '\n"
                       "         f'frente a {amp.loc[0, \"VAN neto de contratar\"]:.1f} k€ sin ampliar. Se compara la mejora del VAN por adelantar los ingresos '\n"
                       "         'con lo que cuesta mantener a los operarios todo el proyecto.')"))
    celdas += apartado(4)
    celdas.append(code("e0 = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, 0.0, N, k, ampliaciones=0)['escenarios']['VAN']\n"
                       "e1 = proyecto.van_recursos(datos_proyecto, 'duration', ['operarios'], [L], BAC, 1.0, N, k, ampliaciones=0)['escenarios']['VAN']\n"
                       "umbral = -e0 / (e1 - e0)          # el VAN es lineal en F\n"
                       "Markdown(f'Flujo mensual mínimo: {umbral[\"sin límite\"]:.2f} k€ sin límite y {umbral[\"con límite\"]:.2f} k€ con límite '\n"
                       "         f'(+{umbral[\"con límite\"] - umbral[\"sin límite\"]:.2f} k€ al mes): el límite exige más ingresos para recuperar lo mismo.')"))
    return celdas, f"límite de {L} operarios, ingresos de {F:.0f} k€/mes durante {N} meses"


def _crashing_van(rng):
    """Duración óptima por VAN: cuánto compensa acelerar cuando acelerar adelanta los ingresos."""
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 20 and (i["D"] >= 2).all()):
            return False
        nombres = list(i["D"].index)
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        BAC = pd.Series(rng.integers(2, 9, size=len(nombres)) * 25, index=nombres, name="BAC")
        k = float(rng.choice([0.08, 0.10, 0.12]))
        N = int(rng.choice([36, 48, 60]))
        df = P.tabla_base(i, duration=i["D"], cap_duration=tope, ucr=ucr, BAC=BAC)
        pg = i["proyecto"]
        try:
            t0 = pg.crashing_van(df, "duration", tope, ucr, BAC, 0.0, N, k, reduccion_max=8)
            t1 = pg.crashing_van(df, "duration", tope, ucr, BAC, 1.0, N, k, reduccion_max=8)
        except Exception:
            return False
        if t0.index.max() < 4:
            return False
        a, b = t0["VAN"], t1["VAN"] - t0["VAN"]
        for F in np.arange(10, 400, 5.0):
            r = int((a + F * b).idxmax())
            v = a + F * b
            if 2 <= r <= t0.index.max() - 1 and v[r] > v[0] + 5 and v[0] > 0:
                estado.update(df=df, k=k, N=N, F=float(F))
                return True
        return False

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), max_intentos=6000)
    df, k, N, F = estado["df"], estado["k"], estado["N"], estado["F"]
    celdas = [md("# Enunciado"),
              md("Una empresa va a ejecutar el proyecto de inversión cuyo cuadro de prelaciones, duración normal y tope (meses), "
                 "coste unitario de reducción (CUR, k€ por mes recortado) y presupuesto (BAC, k€) se adjunta. El coste de cada "
                 "actividad se desembolsa uniformemente a lo largo de su duración; el coste de reducción se suma a su presupuesto. "
                 f"Cuando el proyecto termina, la instalación genera {F:.0f} k€ al mes durante {N} meses. La empresa descuenta "
                 f"al {k:.0%} anual. Las actividades con CUR infinito no se pueden acelerar.")]
    celdas.append(preguntas([
        "Calcule la duración y el VAN del proyecto sin reducir.",
        "Aplique el algoritmo de Ackoff-Sasieni mes a mes. Para cada mes recortado, calcule el coste acumulado de reducción, "
        "la duración resultante y el VAN.",
        "¿Cuántos meses conviene recortar? Compare con la reducción a mínimo coste directo (no recortar) y con recortar al máximo.",
        "Calcule el flujo mensual mínimo para que compense recortar uno, dos, … meses. ¿Cómo cambia la decisión si el flujo "
        "real fuera un 30 % menor?"]))
    celdas.append(code('#| tbl-cap: "Datos del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(FINANZAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"F, N, k = {F}, {N}, {k}\n"
                       "BAC = datos_proyecto['BAC']\n"
                       "tope, ucr = datos_proyecto['cap_duration'], datos_proyecto['ucr']"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales y el camino crítico", label="fig-grafo")
    celdas.append(code('#| tbl-cap: "Calendario con las duraciones normales (meses)"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_normal = int(proyecto.duration(datos_proyecto['duration']))\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(md("El coste mensual de cada actividad es su BAC entre su duración; la fila *Total* del Gantt es el "
                     "desembolso de cada mes. En los meses 1…T el flujo es −desembolso y en los T+1…T+N, +F; la tasa "
                     "mensual equivalente cumple $(1 + k_m)^{12} = 1 + k$."))
    celdas.append(code('#| fig-cap: "Gantt de inicios más tempranos con el coste mensual de cada actividad (k€)"\n#| label: fig-ganttCostes\n'
                       "datos_gantt = datos_proyecto.copy()\n"
                       "datos_gantt['coste_mensual'] = datos_gantt['BAC'] / datos_gantt['duration']\n"
                       "gantt_costes, dibujo = proyecto.gantt(datos_gantt, 'duration', 'coste_mensual', total='ambas', acumulado=True, tikz=True,\n"
                       "                                      params=dict(inner_text_digits=1, **GANTT_PARAMS))\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("def desembolsos(duraciones, sobrecoste=None):\n"
                       "    '''Desembolso de cada mes con unas duraciones dadas (inicios más tempranos); el sobrecoste de reducción\n"
                       "    de cada actividad se suma a su BAC y se reparte en su nueva duración.'''\n"
                       "    cal = proyecto.calendar(duraciones)\n"
                       "    T = int(cal['fin_mas_temprano'].max())\n"
                       "    pagos = pd.Series(0.0, index=range(1, T + 1))\n"
                       "    for a in datos_proyecto.index:\n"
                       "        es, d = int(cal.loc[a, 'inicio_mas_temprano']), int(duraciones[a])\n"
                       "        total = float(BAC[a]) + (0.0 if sobrecoste is None else float(sobrecoste[a]))\n"
                       "        for m in range(es + 1, es + d + 1):\n"
                       "            pagos[m] += total / d\n"
                       "    return pagos\n"
                       "def flujo_mensual(pagos):\n"
                       "    return [0.0] + [-float(x) for x in pagos] + [F] * N\n"
                       "k_mensual = (1 + k) ** (1 / 12) - 1\n"
                       "pagos0 = desembolsos(datos_proyecto['duration'])\n"
                       "VAN0 = van(flujo_mensual(pagos0), k_mensual)\n"
                       "display(pd.DataFrame({'desembolso (k€)': pagos0}).T.round(1))\n"
                       "Markdown(f'k mensual = (1 + {k})^(1/12) − 1 = {k_mensual:.4%}. Sin reducir: {duracion_normal} meses de inversión ({pagos0.sum():.0f} k€ = Σ BAC) y '\n"
                       "         f'VAN = {VAN0:.1f} k€ (valor actual de los ingresos {van([0.0] * (duracion_normal + 1) + [F] * N, k_mensual):.1f} menos el de los desembolsos '\n"
                       "         f'{-van([0.0] + [-float(x) for x in pagos0], k_mensual):.1f}).')"))
    celdas.append(code("tabla = proyecto.crashing_van(datos_proyecto, 'duration', tope, ucr, BAC, F, N, k, reduccion_max=8)\n"
                       "Markdown(f'`crashing_van` da lo mismo: sin reducir, {tabla.loc[0, \"duración\"]:.0f} meses y VAN {tabla.loc[0, \"VAN\"]:.1f} k€.')"))
    celdas += apartado(2)
    celdas.append(md("En cada paso del algoritmo de Ackoff-Sasieni se acorta un mes la ruta (o rutas) crítica(s) "
                     "eligiendo, entre las actividades críticas con margen, la combinación de menor CUR que reduce "
                     "*todas* las rutas críticas a la vez. La tabla muestra, para cada ruta, el CUR de sus actividades y "
                     "su duración tras cada paso; las últimas filas, el margen que le queda a cada actividad."))
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni (hasta ocho meses)"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=tope, costs=ucr, reduction=8, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], tope, ucr, pasos)"))
    celdas.append(md("Para cada mes recortado: el coste de reducción se suma al BAC de las actividades acortadas (y se "
                     "reparte en su nueva duración), el proyecto termina antes y los ingresos empiezan antes. La tabla "
                     "repite el cálculo del VAN mes a mes, con las reducciones acumuladas de cada paso (separadas por |):"))
    celdas.append(code('#| tbl-cap: "Reducción mes a mes: actividades acortadas, coste, duración y VAN"\n#| label: tbl-crashingVan\n'
                       "acortadas = {0: '—'}\n"
                       "for r in tabla.index[1:]:\n"
                       "    acortadas[r] = ' | '.join(', '.join(pasos[j]) for j in sorted(pasos) if j < r)\n"
                       "tabla.insert(0, 'actividades acortadas (paso a paso)', pd.Series(acortadas))\n"
                       "tabla.round(2)"))
    celdas.append(code("comprobacion = []\n"
                       "d = datos_proyecto['duration'].astype(float).copy(); sobrecoste = pd.Series(0.0, index=datos_proyecto.index)\n"
                       "for r in tabla.index[1:]:\n"
                       "    for a in pasos[r - 1]:\n"
                       "        d[a] -= 1; sobrecoste[a] += float(ucr[a])\n"
                       "    comprobacion.append({'meses recortados': r, 'duración': int(proyecto.duration(d.astype(int))), 'coste de reducción': float(sobrecoste.sum()),\n"
                       "                         'VAN mes a mes': van(flujo_mensual(desembolsos(d.astype(int), sobrecoste)), k_mensual)})\n"
                       "pd.DataFrame(comprobacion).set_index('meses recortados').round(2)"))
    celdas.append(md("La comprobación mes a mes (desembolsos con el sobrecoste incluido y los ingresos adelantados) "
                     "coincide con la columna VAN de `crashing_van`."))
    celdas += apartado(3)
    celdas.append(code("optimo = int(tabla['VAN'].idxmax())\n"
                       "Markdown(f'Conviene recortar {optimo} mes(es): duración {tabla.loc[optimo, \"duración\"]:.0f} meses, '\n"
                       "         f'VAN {tabla.loc[optimo, \"VAN\"]:.1f} k€ (sin recortar {tabla.loc[0, \"VAN\"]:.1f}; recortando al máximo '\n"
                       "         f'{tabla[\"VAN\"].iloc[-1]:.1f}). El óptimo no es la duración de mínimo coste directo ni la mínima: es donde lo que '\n"
                       "         'cuesta el siguiente mes recortado iguala lo que vale cobrarlo un mes antes.')"))
    celdas += apartado(4)
    celdas.append(code("t0 = proyecto.crashing_van(datos_proyecto, 'duration', tope, ucr, BAC, 0.0, N, k, reduccion_max=8)['VAN']\n"
                       "t1 = proyecto.crashing_van(datos_proyecto, 'duration', tope, ucr, BAC, 1.0, N, k, reduccion_max=8)['VAN']\n"
                       "pend = t1 - t0                      # el VAN es lineal en F\n"
                       "umbral = ((t0.loc[0] - t0) / (pend - pend.loc[0])).drop(0).rename('flujo mínimo (k€/mes)')\n"
                       "display(umbral.round(2).to_frame())\n"
                       "v2 = t0 + 0.7 * F * pend\n"
                       "Markdown(f'Con un flujo un 30 % menor ({0.7 * F:.1f} k€/mes) conviene recortar {int(v2.idxmax())} mes(es): '\n"
                       "         'cuanto menor es el ingreso, menos vale adelantarlo y antes se detiene la reducción.')"))
    return celdas, f"ingresos de {F:.0f} k€/mes durante {N} meses; reducción mes a mes con Ackoff"


VARIANTES = [
    Variante("caso-base", 1, "flujos de caja libre completos (ventas, costes, amortización, impuesto, circulante, residual), VAN con tabla de descuento, TIR, payback simple y descontado",
             _caso_base),
    Variante("vinculo-proyecto", 2, "desembolsos mensuales desde el Gantt, ingresos al terminar; VAN mensual, coste de un retraso y reducción con Ackoff frente al valor de un mes",
             _vinculo_proyecto),
    Variante("alternativas", 2, "proyectos excluyentes con conflicto VAN/TIR, tasa de Fisher, perfiles de VAN y vidas distintas (anualidad equivalente)",
             _alternativas),
    Variante("sensibilidad", 2, "umbrales de precio, volumen y coste (VAN = 0), diagrama de araña, escenarios e inflación (tasa real/nominal)",
             _sensibilidad),
    Variante("financiacion", 3, "cuadro de amortización del préstamo, flujo del accionista con escudo fiscal, TIR del proyecto frente a la del accionista y apalancamiento",
             _financiacion),
    Variante("van-montecarlo", 3, "duraciones triangulares e ingreso incierto: distribución del VAN (Monte Carlo), P(VAN<0), efecto de cada incertidumbre, tornado y valor de reducirla",
             _van_montecarlo),
    Variante("recursos-van", 3, "límite de operarios y VAN: retraso de ingresos, coste de ampliar la cuadrilla y flujo mínimo con y sin límite",
             _recursos_van),
    Variante("crashing-van", 3, "duración óptima por VAN: reducir mes a mes con Ackoff, coste frente a ingresos adelantados y umbral de flujo",
             _crashing_van),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
