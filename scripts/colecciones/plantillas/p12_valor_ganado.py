"""Colección 12 — Valor ganado: Gantts PV/EV/AC, indicadores, previsiones
EAC/VAC/TCPI, programación ganada (Earned Schedule) y seguimiento en el tiempo."""
import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="12_valor_ganado", prefijo="ValorGanado",
            titulo="Método del valor ganado",
            descripcion="Diagramas de Gantt del valor planificado, ganado y coste real; CPI, SPI, EAC, VAC, "
                        "TCPI; programación ganada; evolución de los indicadores en varios puntos de control.",
            n_defecto=3)


def _simular(rng, info, D, BAC):
    """Ejecución real: duraciones y costes reales con sobrecostes y retrasos."""
    preds = info["preds"]
    f = rng.choice([0.8, 1.0, 1.0, 1.25, 1.5, 1.5, 2.0], size=len(D))
    g = rng.choice([0.9, 1.0, 1.1, 1.2, 1.3, 1.5], size=len(D))
    dur_real = pd.Series(np.maximum(1, np.round(D.values * f)).astype(int), index=D.index)
    coste_real = pd.Series(np.round(BAC.values * g).astype(int), index=D.index)
    es, ef = {}, {}
    for a in D.index:                       # D.index está en orden topológico (A, B, C, …)
        es[a] = max([ef[p] for p in preds[a]], default=0)
        ef[a] = es[a] + int(dur_real[a])
    return dur_real, coste_real, pd.Series(es), pd.Series(ef)


def _seguimiento(t, dur_real, coste_real, es):
    transcurrido = (t - es).clip(lower=0).clip(upper=dur_real)
    performed = (100 * transcurrido / dur_real).round().astype(int)
    AC = (coste_real * transcurrido / dur_real).round().astype(int)
    return pd.DataFrame({"actual_duration": transcurrido.astype(int), "AC": AC, "performed": performed})


def _base(k, rng):
    variante = (k - 1) % 3 + 1
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22):
            return False
        BAC = pd.Series(rng.integers(1, 7, size=len(i["D"])) * 10, index=i["D"].index, name="BAC")
        dur_real, coste_real, es, ef = _simular(rng, i, i["D"], BAC)
        T_real = int(ef.max())
        if not (i["T"] < T_real <= 1.6 * i["T"]):
            return False
        if variante == 3:
            ts = [int(round(0.3 * i["T"])), int(round(0.55 * i["T"])), int(round(0.8 * i["T"]))]
        else:
            ts = [int(round(0.5 * i["T"]))]
        segs = [_seguimiento(t, dur_real, coste_real, es) for t in ts]
        for seg in segs:
            if not ((0 < seg["performed"]) & (seg["performed"] < 100)).any() or not (seg["performed"] == 0).any():
                return False
            EV = float((BAC * seg["performed"] / 100).sum()); AC = float(seg["AC"].sum())
            if AC <= EV:                      # queremos un proyecto con sobrecoste
                return False
        estado.update(BAC=BAC, dur_real=dur_real, coste_real=coste_real, es=es, ts=ts, segs=segs, T_real=T_real)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    BAC, ts, segs = estado["BAC"], estado["ts"], estado["segs"]
    df = P.tabla_base(info, duration=D, BAC=BAC)

    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y "
                 "presupuesto por actividad (BAC, en k€) se adjunta. El coste planificado de cada actividad "
                 "se reparte uniformemente a lo largo de su duración.")]
    celdas.append(code('#| tbl-cap: "Planificación del proyecto"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    if variante == 3:
        celdas.append(md(f"Se han realizado tres puntos de control, al final de las semanas {ts[0]}, {ts[1]} y "
                         f"{ts[2]}. En cada uno se anotó, para cada actividad, las semanas trabajadas hasta ese "
                         "momento, el coste real incurrido (k€) y el porcentaje realizado."))
        for j, (t, seg) in enumerate(zip(ts, segs), 1):
            celdas.append(code(f'#| tbl-cap: "Punto de control {j} (final de la semana {t})"\n#| label: tbl-seguimiento{j}\n'
                               + literal_dataframe(seg, nombre=f"seguimiento_{j}")))
        pregs = ["Calcule PV, EV y AC en cada punto de control.",
                 "Calcule CV, SV, CPI y SPI en cada punto de control y represente su evolución.",
                 "Calcule en cada punto la estimación del coste final EAC = BAC / CPI y comente la tendencia.",
                 "A la vista de la evolución, ¿el proyecto se está recuperando o deteriorando? ¿Qué decisiones "
                 "tomaría tras el último punto de control?"]
    else:
        t = ts[0]
        celdas.append(md(f"Al final de la semana {t} se realiza un punto de control. Para cada actividad se "
                         "anotan las semanas trabajadas hasta ese momento, el coste real incurrido (k€) y el "
                         "porcentaje realizado."))
        celdas.append(code(f'#| tbl-cap: "Datos del punto de control (final de la semana {t})"\n#| label: tbl-seguimiento\n'
                           + literal_dataframe(segs[0], nombre="seguimiento")))
        if variante == 1:
            pregs = ["Dibuje los tres diagramas de Gantt del valor ganado: valor planificado (PV), valor ganado "
                     "(EV) y coste real (AC).",
                     "Calcule PV, EV y AC en el punto de control, las variaciones CV y SV y los índices CPI y SPI. "
                     "Interprete la situación del proyecto.",
                     "Estime el coste final del proyecto (EAC) con tres hipótesis: (a) el resto se ejecuta según "
                     "presupuesto, (b) se mantiene la eficiencia en coste actual, (c) se mantienen las "
                     "eficiencias en coste y en plazo. Calcule la variación al final (VAC) en cada caso.",
                     "Calcule el índice de rendimiento necesario para terminar en presupuesto (TCPI). ¿Es "
                     "realista?"]
        else:
            pregs = ["Dibuje los diagramas de Gantt del valor planificado y del valor ganado y calcule PV, EV, "
                     "AC, CPI y SPI en el punto de control.",
                     "Calcule la programación ganada (ES), el retraso en semanas y el índice SPI(t).",
                     "Estime la duración final del proyecto con SPI y con SPI(t). ¿Por qué difieren? ¿Cuál "
                     "es más fiable?",
                     "¿Qué valor tomará el SPI clásico cuando el proyecto termine, aunque acabe tarde? ¿Y el "
                     "SPI(t)?"]
    celdas.append(preguntas(pregs))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "BAC_total = float(datos_proyecto['BAC'].sum())\n"
                       "duracion_planificada = proyecto.duration(datos_proyecto['duration'])"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    ap = 1
    params = ("params=dict(period_width=1, row_height=0.7, activity_relative_height=0.6, **GANTT_PARAMS,\n"
              "                                       critical_background_color='white!90!green', regular_background_color='white!90!green',\n"
              "                                       regular_text_color='black', critical_text_color='black')")
    if variante in (1, 2):
        celdas += apartado(ap); ap += 1
        celdas.append(md("En el Gantt del valor planificado cada casilla lleva el coste planificado de la semana "
                         "(BAC / duración). En el del valor ganado, el BAC multiplicado por el porcentaje realizado se "
                         "reparte entre las semanas trabajadas. En el del coste real, el coste incurrido entre las "
                         "semanas trabajadas."))
        celdas.append(code("datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                           "vg = EarnedValue(proyecto)\n"
                           "gantts = vg.calcula_gantts(datos_valor_ganado, planned_durations_label='duration',\n"
                           "                           actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',\n"
                           "                           percentage_complete_label='performed', tikz=True,\n"
                           "                           " + params + ")"))
        celdas.append(code('#| fig-cap: "PV: valor planificado"\n#| label: fig-ganttPV\npinta_tikz(gantts[\'gantt_PV\'])'))
        celdas.append(code('#| fig-cap: "EV: valor ganado"\n#| label: fig-ganttEV\npinta_tikz(gantts[\'gantt_EV\'])'))
        if variante == 1:
            celdas.append(code('#| fig-cap: "AC: coste real"\n#| label: fig-ganttAC\npinta_tikz(gantts[\'gantt_AC\'])'))
            celdas += apartado(ap); ap += 1
        celdas.append(code(f"semana_control = {t}\n"
                           "acumulados = gantts['acumulados']\n"
                           "PV = float(acumulados.loc[semana_control, 'PV'])\n"
                           "EV = float((datos_valor_ganado['BAC'] * datos_valor_ganado['performed'] / 100).sum())\n"
                           "AC = float(datos_valor_ganado['AC'].sum())\n"
                           "CV, SV = EV - AC, EV - PV\n"
                           "CPI, SPI = EV / AC, EV / PV\n"
                           "pd.Series({'PV': PV, 'EV': EV, 'AC': AC, 'CV': CV, 'SV': SV, 'CPI': CPI, 'SPI': SPI, 'BAC': BAC_total}).round(3).to_frame('valor').T"))
        celdas.append(md("De dónde salen las tres cifras: el valor planificado de cada actividad es su BAC por la "
                         "fracción de su duración que debería haberse ejecutado en la semana de control (según el "
                         "calendario de inicios más tempranos); el valor ganado, su BAC por el porcentaje realizado; "
                         "el coste real, lo gastado en ella."))
        celdas.append(code('#| tbl-cap: "PV, EV y AC actividad a actividad en el punto de control"\n#| label: tbl-valorGanadoPaso\n'
                           "calendario_plan = proyecto.calendar(datos_proyecto['duration'])\n"
                           "detalle_vg = detalle_valor_ganado(datos_valor_ganado, calendario_plan, semana_control)\n"
                           "detalle_vg[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
        celdas.append(code("Markdown(f\'\'\'Sumando las columnas: PV = {detalle_vg[\"PV_i\"].sum():.1f}, EV = {detalle_vg[\"EV_i\"].sum():.1f}, AC = {detalle_vg[\"AC\"].sum():.1f} k€. Y con ellas:\n\n"
                           "- CV = EV − AC = {EV:.1f} − {AC:.1f} = {CV:.1f} k€\n"
                           "- SV = EV − PV = {EV:.1f} − {PV:.1f} = {SV:.1f} k€\n"
                           "- CPI = EV / AC = {EV:.1f} / {AC:.1f} = {CPI:.3f}\n"
                           "- SPI = EV / PV = {EV:.1f} / {PV:.1f} = {SPI:.3f}\'\'\')"))
        celdas.append(code('#| fig-cap: "Curvas S del proyecto"\n#| label: fig-curvasS\n'
                           "plot_curvas_valor_ganado(gantts)"))
        celdas.append(code("Markdown(f'''En la semana {semana_control} se ha ganado valor por {EV:.1f} k€ frente a los {PV:.1f} planificados (SV = {SV:.1f}, SPI = {SPI:.2f}: "
                           "{\"retraso\" if SPI < 1 else \"adelanto\"}) y se han gastado {AC:.1f} k€ (CV = {CV:.1f}, CPI = {CPI:.2f}: por cada euro gastado se obtienen "
                           "{CPI:.2f} euros de valor, {\"sobrecoste\" if CPI < 1 else \"ahorro\"}).''')"))
    if variante == 1:
        celdas += apartado(ap); ap += 1
        celdas.append(md("Las tres previsiones habituales del coste final:\n\n"
                         "- (a) $EAC = AC + (BAC - EV)$: el trabajo restante costará lo presupuestado.\n"
                         "- (b) $EAC = BAC / CPI = AC + (BAC - EV)/CPI$: se mantiene la eficiencia en coste.\n"
                         "- (c) $EAC = AC + (BAC - EV)/(CPI \\cdot SPI)$: el retraso también encarece el resto.\n\n"
                         "En todas, $VAC = BAC - EAC$."))
        celdas.append(code("previsiones = pd.DataFrame({\n"
                           "    '(a) resto según presupuesto': [AC + (BAC_total - EV)],\n"
                           "    '(b) misma eficiencia en coste': [BAC_total / CPI],\n"
                           "    '(c) misma eficiencia en coste y plazo': [AC + (BAC_total - EV) / (CPI * SPI)]},\n"
                           "    index=['EAC']).T\n"
                           "previsiones['VAC'] = BAC_total - previsiones['EAC']\n"
                           "previsiones.round(1)"))
        celdas.append(code("Markdown(f'La hipótesis (a) es la más optimista: supone que los problemas no se repetirán. La (c) es la más pesimista. "
                           "Lo habitual es tomar (b) como previsión central: {BAC_total / CPI:.1f} k€ frente a los {BAC_total:.0f} presupuestados.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(md("$TCPI = \\dfrac{BAC - EV}{BAC - AC}$: eficiencia en coste que habría que mantener en lo que "
                         "queda para terminar exactamente en presupuesto."))
        celdas.append(code("TCPI = (BAC_total - EV) / (BAC_total - AC)\n"
                           "Markdown(f'TCPI = ({BAC_total:.0f} − {EV:.1f}) / ({BAC_total:.0f} − {AC:.1f}) = {TCPI:.2f}. '\n"
                           "         f'Hasta ahora el proyecto rinde CPI = {CPI:.2f}; pasar a {TCPI:.2f} exige una mejora de '\n"
                           "         f'{(TCPI / CPI - 1) * 100:.0f} %, {\"poco realista sin cambios de fondo\" if TCPI > 1.1 else \"alcanzable con medidas correctoras\"}.')"))
    elif variante == 2:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La programación ganada (*Earned Schedule*) busca en qué instante de la planificación se "
                         "había previsto alcanzar el valor que realmente se ha ganado:\n\n"
                         "$$ES = C + \\frac{EV - PV_C}{PV_{C+1} - PV_C}$$\n\n"
                         "donde $C$ es la última semana completa con $PV_C \\le EV$. Entonces $SPI(t) = ES / AT$, siendo "
                         "$AT$ el tiempo real transcurrido."))
        celdas.append(code("pv_acumulado = pd.concat([pd.Series({0: 0.0}), acumulados['PV'].astype(float)])\n"
                           "C = int(pv_acumulado[pv_acumulado <= EV].index.max())\n"
                           "PV_C, PV_C1 = float(pv_acumulado[C]), float(pv_acumulado[C + 1])\n"
                           "AT = semana_control\n"
                           "ES = C + (EV - PV_C) / (PV_C1 - PV_C)\n"
                           "SPI_t = ES / AT\n"
                           "Markdown(f'PV acumulado en la semana {C}: {PV_C:.1f}; en la {C + 1}: {PV_C1:.1f}. '\n"
                           "         f'ES = {C} + ({EV:.1f} − {PV_C:.1f}) / ({PV_C1:.1f} − {PV_C:.1f}) = {ES:.2f} semanas. '\n"
                           "         f'Retraso = AT − ES = {AT - ES:.2f} semanas; SPI(t) = {ES:.2f} / {AT} = {SPI_t:.2f}.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("PD = duracion_planificada\n"
                           "IEAC_spi = PD / SPI\n"
                           "IEAC_t = PD / SPI_t\n"
                           "pd.DataFrame({'con SPI (€)': [SPI, IEAC_spi], 'con SPI(t) (tiempo)': [SPI_t, IEAC_t]},\n"
                           "             index=['índice', 'duración prevista (semanas)']).round(2)"))
        celdas.append(md("El SPI en unidades monetarias compara euros ganados con euros planificados, y depende de "
                         "qué actividades (caras o baratas) estén retrasadas. El SPI(t) compara tiempos: cuántas "
                         "semanas de avance planificado se han conseguido en las semanas transcurridas. Es más fiable "
                         "para prever la duración, sobre todo en la segunda mitad del proyecto."))
        celdas += apartado(ap); ap += 1
        celdas.append(code("Markdown(f'''Al terminar, EV = BAC y PV = BAC, así que el SPI clásico vale 1 aunque el proyecto haya terminado tarde: pierde su capacidad de aviso conforme "
                           "avanza el proyecto. El SPI(t) al final vale PD / duración real = {PD} / duración real, que sigue siendo menor que 1 si hubo retraso.''')"))
    else:
        celdas += apartado(ap); ap += 1
        celdas.append(code("seguimientos = {" + ", ".join(f"{t}: seguimiento_{j}" for j, t in enumerate(ts, 1)) + "}\n"
                           "filas = {}\n"
                           "curvas = {}\n"
                           "for semana, seguimiento in seguimientos.items():\n"
                           "    datos_vg = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                           "    gantts = EarnedValue(proyecto.copy()).calcula_gantts(datos_vg, planned_durations_label='duration',\n"
                           "                                                          actual_durations_label='actual_duration', PV_label='BAC',\n"
                           "                                                          AC_label='AC', percentage_complete_label='performed', tikz=False)\n"
                           "    PV = float(gantts['acumulados'].loc[semana, 'PV'])\n"
                           "    EV = float((datos_vg['BAC'] * datos_vg['performed'] / 100).sum())\n"
                           "    AC = float(datos_vg['AC'].sum())\n"
                           "    filas[semana] = {'PV': PV, 'EV': EV, 'AC': AC}\n"
                           "indicadores = pd.DataFrame(filas).T\n"
                           "indicadores.index.name = 'semana'\n"
                           "indicadores.round(1)"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("indicadores['CV'] = indicadores['EV'] - indicadores['AC']\n"
                           "indicadores['SV'] = indicadores['EV'] - indicadores['PV']\n"
                           "indicadores['CPI'] = indicadores['EV'] / indicadores['AC']\n"
                           "indicadores['SPI'] = indicadores['EV'] / indicadores['PV']\n"
                           "indicadores.round(3)"))
        celdas.append(code('#| fig-cap: "Evolución de CPI y SPI"\n#| label: fig-evolucion\n'
                           "ax = indicadores[['CPI', 'SPI']].plot(marker='o', figsize=(7, 4))\n"
                           "ax.axhline(1, color='grey', linestyle='--'); ax.set_ylabel('índice');"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("indicadores['EAC'] = BAC_total / indicadores['CPI']\n"
                           "indicadores['VAC'] = BAC_total - indicadores['EAC']\n"
                           "indicadores[['CPI', 'EAC', 'VAC']].round({'CPI': 3, 'EAC': 1, 'VAC': 1})"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("cpi = indicadores['CPI']; spi = indicadores['SPI']\n"
                           "tend_c = 'mejora' if cpi.iloc[-1] > cpi.iloc[0] else 'empeora'\n"
                           "tend_s = 'mejora' if spi.iloc[-1] > spi.iloc[0] else 'empeora'\n"
                           "Markdown(f'''Entre el primer y el último punto de control el CPI {tend_c} ({cpi.iloc[0]:.2f} → {cpi.iloc[-1]:.2f}) y el SPI {tend_s} "
                           "({spi.iloc[0]:.2f} → {spi.iloc[-1]:.2f}). La previsión de coste final pasa de {indicadores[\"EAC\"].iloc[0]:.0f} a {indicadores[\"EAC\"].iloc[-1]:.0f} k€ "
                           "frente a un presupuesto de {BAC_total:.0f}. Conviene recordar que el SPI tiende a 1 al final del proyecto aunque haya retraso, "
                           "así que su mejora en los últimos puntos puede ser aparente; para el plazo es preferible la programación ganada. "
                           "Tras el último control procede revisar las actividades pendientes: acelerar las críticas (si el sobrecoste compensa la penalización o los costes "
                           "indirectos) y revisar las causas del sobrecoste en las ya ejecutadas para que no se repitan.''')"))
    if variante == 1:
        sub = f"{n} actividades, un punto de control (semana {ts[0]}), EAC/VAC/TCPI"
    elif variante == 2:
        sub = f"{n} actividades, un punto de control (semana {ts[0]}), programación ganada"
    else:
        sub = f"{n} actividades, tres puntos de control (semanas {', '.join(map(str, ts))})"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _simular_control(rng, info, D, BAC, f_opciones, g_opciones):
    preds = info["preds"]
    f = rng.choice(f_opciones, size=len(D))
    g = rng.choice(g_opciones, size=len(D))
    dur_real = pd.Series(np.maximum(1, np.round(D.values * f)).astype(int), index=D.index)
    coste_real = pd.Series(np.round(BAC.values * g).astype(int), index=D.index)
    es, ef = {}, {}
    for a in D.index:
        es[a] = max([ef[p] for p in preds[a]], default=0)
        ef[a] = es[a] + int(dur_real[a])
    return dur_real, coste_real, pd.Series(es), pd.Series(ef)


def _situacion(rng, modo):
    """modo 'coste-bien-plazo-mal' (CPI > 1, SPI < 1) o 'plazo-bien-coste-mal' (SPI ≥ 1, CPI < 1)."""
    n = int(rng.choice([8, 9]))
    estado = {}
    if modo == "coste-bien-plazo-mal":
        f_op, g_op = [1.0, 1.25, 1.5, 1.5, 2.0], [0.8, 0.85, 0.9, 0.95, 1.0]
    else:
        f_op, g_op = [0.6, 0.75, 0.8, 1.0, 1.0], [1.1, 1.2, 1.3, 1.5]

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22):
            return False
        BAC = pd.Series(rng.integers(1, 7, size=len(i["D"])) * 10, index=i["D"].index, name="BAC")
        dur_real, coste_real, es, ef = _simular_control(rng, i, i["D"], BAC, f_op, g_op)
        t = int(round(0.5 * i["T"]))
        seg = _seguimiento(t, dur_real, coste_real, es)
        if not ((0 < seg["performed"]) & (seg["performed"] < 100)).any() or not (seg["performed"] == 0).any():
            return False
        EV = float((BAC * seg["performed"] / 100).sum()); AC = float(seg["AC"].sum())
        # PV en t: suma de BAC * fracción planificada transcurrida
        cal = i["calendario"]
        frac = ((t - cal["inicio_mas_temprano"]).clip(lower=0).clip(upper=cal["duracion"]) / cal["duracion"])
        PV = float((BAC * frac).sum())
        if modo == "coste-bien-plazo-mal":
            ok = EV > AC * 1.03 and EV < PV * 0.95
        else:
            ok = EV >= PV * 1.0 and EV < AC * 0.95
        if not ok:
            return False
        estado.update(BAC=BAC, seg=seg, t=t)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    BAC, seg, t = estado["BAC"], estado["seg"], estado["t"]
    df = P.tabla_base(info, duration=D, BAC=BAC)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto "
                 "por actividad (BAC, k€) se adjunta; el coste planificado de cada actividad se reparte "
                 f"uniformemente en su duración. Al final de la semana {t} se realiza un punto de control.")]
    celdas.append(code('#| tbl-cap: "Planificación del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(code(f'#| tbl-cap: "Datos del punto de control (final de la semana {t})"\n#| label: tbl-seguimiento\n'
                       + literal_dataframe(seg, nombre="seguimiento")))
    if modo == "coste-bien-plazo-mal":
        pregs = ["Calcule PV, EV y AC en el punto de control, y los índices CPI y SPI.",
                 "El director informa al comité de que «el proyecto va bien: hemos gastado menos de lo previsto». "
                 "¿Es correcta la lectura? ¿Qué parte de la situación oculta?",
                 "¿Qué actividades explican el retraso? Compare, actividad por actividad, el valor ganado con el "
                 "valor planificado.",
                 "Estime el coste final (EAC = BAC/CPI) y la duración final con SPI. ¿Qué previsión se fiaría más y "
                 "cuál menos?"]
    else:
        pregs = ["Calcule PV, EV y AC en el punto de control, y los índices CPI y SPI.",
                 "El director informa al comité de que «vamos adelantados». ¿Es correcta la lectura? ¿Qué parte de la "
                 "situación oculta?",
                 "¿Qué actividades explican el sobrecoste? Compare, actividad por actividad, el coste real con el "
                 "valor ganado.",
                 "Estime el coste final (EAC = BAC/CPI) y la variación al final (VAC). ¿Compensa el adelanto el "
                 "sobrecoste? ¿Con qué dato del contrato lo decidiría?"]
    celdas.append(preguntas(pregs))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "BAC_total = float(datos_proyecto['BAC'].sum())\n"
                       "duracion_planificada = proyecto.duration(datos_proyecto['duration'])\n"
                       f"semana_control = {t}\n"
                       "datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                       "gantts = EarnedValue(proyecto).calcula_gantts(datos_valor_ganado, planned_durations_label='duration',\n"
                       "                                             actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',\n"
                       "                                             percentage_complete_label='performed', tikz=True, params=dict(GANTT_PARAMS))"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "PV: valor planificado"\n#| label: fig-ganttPV\npinta_tikz(gantts[\'gantt_PV\'])'))
    celdas.append(code('#| fig-cap: "EV: valor ganado"\n#| label: fig-ganttEV\npinta_tikz(gantts[\'gantt_EV\'])'))
    celdas.append(code("acumulados = gantts['acumulados']\n"
                       "PV = float(acumulados.loc[semana_control, 'PV'])\n"
                       "EV = float((datos_valor_ganado['BAC'] * datos_valor_ganado['performed'] / 100).sum())\n"
                       "AC = float(datos_valor_ganado['AC'].sum())\n"
                       "CPI, SPI = EV / AC, EV / PV\n"
                       "pd.Series({'PV': PV, 'EV': EV, 'AC': AC, 'CV = EV − AC': EV - AC, 'SV = EV − PV': EV - PV, 'CPI': CPI, 'SPI': SPI}).round(3).to_frame('valor').T"))
    celdas.append(code('#| fig-cap: "Curvas S"\n#| label: fig-curvasS\nplot_curvas_valor_ganado(gantts)'))
    celdas += apartado(2)
    if modo == "coste-bien-plazo-mal":
        celdas.append(code("Markdown(f'''Gastar menos de lo previsto (AC = {AC:.0f} < PV = {PV:.0f}) no dice nada por sí solo: se ha gastado menos porque se ha hecho menos. "
                           "La comparación correcta es entre lo gastado y lo *ganado*: CPI = {CPI:.2f} {\">\" if CPI > 1 else \"<\"} 1, así que el coste sí va bien, "
                           "pero SPI = {SPI:.2f} < 1: el proyecto va retrasado, se ha ganado sólo el {EV/PV*100:.0f} % del valor previsto. La frase del director confunde "
                           "la variación de gasto (AC − PV) con la de coste (EV − AC).''')"))
    else:
        celdas.append(code("Markdown(f'''El adelanto es real (SPI = {SPI:.2f} ≥ 1: se ha ganado valor por {EV:.0f} frente a {PV:.0f} previstos), pero ha costado "
                           "{AC:.0f}: CPI = {CPI:.2f} < 1, cada euro gastado rinde {CPI:.2f}. La lectura del director oculta el sobrecoste, que es lo que "
                           "determinará si el proyecto acaba en presupuesto.''')"))
    celdas += apartado(3)
    celdas.append(code("por_actividad = datos_valor_ganado[['BAC', 'performed', 'AC']].copy()\n"
                       "por_actividad['EV'] = por_actividad['BAC'] * por_actividad['performed'] / 100\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "frac_planificada = ((semana_control - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion'])\n"
                       "por_actividad['PV'] = por_actividad['BAC'] * frac_planificada\n"
                       "por_actividad['SV'] = por_actividad['EV'] - por_actividad['PV']\n"
                       "por_actividad['CV'] = por_actividad['EV'] - por_actividad['AC']\n"
                       "por_actividad.round(1)"))
    if modo == "coste-bien-plazo-mal":
        celdas.append(code("retrasadas = por_actividad[por_actividad['SV'] < -0.5].sort_values('SV')\n"
                           "Markdown(f'Las actividades con SV negativa explican el retraso: {\", \".join(retrasadas.index)}. '\n"
                           "         f'Las de mayor peso son {\", \".join(retrasadas.index[:2])} (SV = {retrasadas[\"SV\"].iloc[0]:.1f} y {retrasadas[\"SV\"].iloc[1]:.1f} k€).' if len(retrasadas) > 1 else 'Una sola actividad explica el retraso.')"))
    else:
        celdas.append(code("caras = por_actividad[por_actividad['CV'] < -0.5].sort_values('CV')\n"
                           "Markdown(f'Las actividades con CV negativa explican el sobrecoste: {\", \".join(caras.index)}; las de mayor peso son {\", \".join(caras.index[:2])}.')"))
    celdas += apartado(4)
    if modo == "coste-bien-plazo-mal":
        celdas.append(code("EAC = BAC_total / CPI\n"
                           "IEAC = duracion_planificada / SPI\n"
                           "Markdown(f'''EAC = {BAC_total:.0f} / {CPI:.2f} = {EAC:.0f} k€ (ahorro de {BAC_total - EAC:.0f}); duración prevista {duracion_planificada} / {SPI:.2f} = {IEAC:.1f} semanas. "
                           "La previsión de coste es la más fiable (el CPI suele ser estable); la de plazo con SPI es la menos: el SPI tiende a 1 al final aunque el retraso persista, "
                           "y además el retraso real depende de si las actividades retrasadas son críticas, cosa que el SPI (en euros) no distingue. Para el plazo conviene la programación ganada o replanificar lo que falta.''')"))
    else:
        celdas.append(code("EAC = BAC_total / CPI\n"
                           "VAC = BAC_total - EAC\n"
                           "Markdown(f'''EAC = {BAC_total:.0f} / {CPI:.2f} = {EAC:.0f} k€, VAC = {VAC:.0f} k€ (sobrecoste previsto). "
                           "Que el adelanto compense depende de lo que valga el tiempo en el contrato: una prima por entrega anticipada o una penalización evitada "
                           "por semana. Si la prima por semana adelantada multiplicada por las semanas que se ganarían supera {-VAC:.0f} k€, compensa; si no, "
                           "se está comprando tiempo demasiado caro.''')"))
    return celdas, f"{n} actividades; punto de control en la semana {t}: {modo.replace('-', ' ')}"


def _diagnostico_vg(rng):
    """Resolución entregada con CPI y SPI intercambiados y la variación de coste con el signo cambiado."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 12 <= i["T"] <= 22):
            return False
        BAC = pd.Series(rng.integers(1, 7, size=len(i["D"])) * 10, index=i["D"].index, name="BAC")
        dur_real, coste_real, es, ef = _simular(rng, i, i["D"], BAC)
        t = int(round(0.5 * i["T"]))
        seg = _seguimiento(t, dur_real, coste_real, es)
        if not ((0 < seg["performed"]) & (seg["performed"] < 100)).any():
            return False
        EV = float((BAC * seg["performed"] / 100).sum()); AC = float(seg["AC"].sum())
        cal = i["calendario"]
        frac = ((t - cal["inicio_mas_temprano"]).clip(lower=0).clip(upper=cal["duracion"]) / cal["duracion"])
        PV = float((BAC * frac).sum())
        if abs(EV / AC - EV / PV) < 0.08 or AC < 1.08 * EV or PV < 1.08 * EV:
            return False
        estado.update(BAC=BAC, seg=seg, t=t, PV=PV, EV=EV, AC=AC)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    BAC, seg, t, PV, EV, AC = estado["BAC"], estado["seg"], estado["t"], estado["PV"], estado["EV"], estado["AC"]
    df = P.tabla_base(info, duration=D, BAC=BAC)
    cpi_mal, spi_mal = EV / PV, EV / AC
    celdas = [md("# Enunciado"),
              md("Para el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y presupuesto por "
                 f"actividad (k€) se adjunta, con el punto de control de la semana {t} que también se da, un alumno "
                 "ha entregado esta resolución:\n\n"
                 f"> *PV = {PV:.0f}, EV = {EV:.0f}, AC = {AC:.0f}. CV = AC − EV = {AC - EV:.0f} (positiva: ahorro). "
                 f"CPI = EV / PV = {cpi_mal:.2f}; SPI = EV / AC = {spi_mal:.2f}. Como CPI < 1 el proyecto va retrasado y como SPI < 1 va con sobrecoste. "
                 f"EAC = BAC × CPI = {float(BAC.sum()) * cpi_mal:.0f} k€.*")]
    celdas.append(code('#| tbl-cap: "Planificación del proyecto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(code(f'#| tbl-cap: "Datos del punto de control (final de la semana {t})"\n#| label: tbl-seguimiento\n'
                       + literal_dataframe(seg, nombre="seguimiento")))
    celdas.append(preguntas([
        "Compruebe PV, EV y AC.",
        "Localice todos los errores de la resolución (hay al menos cuatro) y explique cada uno.",
        "Escriba la resolución correcta: CV, SV, CPI, SPI, EAC y VAC, con su interpretación."]))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "BAC_total = float(datos_proyecto['BAC'].sum())\n"
                       f"semana_control = {t}\n"
                       "datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                       "gantts = EarnedValue(proyecto).calcula_gantts(datos_valor_ganado, planned_durations_label='duration',\n"
                       "                                             actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',\n"
                       "                                             percentage_complete_label='performed', tikz=False)"))
    celdas += apartado(1)
    celdas.append(code("PV = float(gantts['acumulados'].loc[semana_control, 'PV'])\n"
                       "EV = float((datos_valor_ganado['BAC'] * datos_valor_ganado['performed'] / 100).sum())\n"
                       "AC = float(datos_valor_ganado['AC'].sum())\n"
                       "Markdown(f'PV = {PV:.0f}, EV = {EV:.0f}, AC = {AC:.0f}: los tres valores del alumno son correctos.')"))
    celdas += apartado(2)
    celdas.append(md("1. **CV = AC − EV**: la variación de coste es CV = EV − AC. Con el signo cambiado un "
                     "sobrecoste parece un ahorro.\n"
                     "2. **CPI = EV / PV**: eso es el SPI. El CPI compara valor ganado con coste real: CPI = EV / AC.\n"
                     "3. **SPI = EV / AC**: eso es el CPI. SPI = EV / PV.\n"
                     "4. **Interpretación cruzada**: CPI mide coste y SPI mide plazo; al intercambiarlos se "
                     "intercambia también la lectura.\n"
                     "5. **EAC = BAC × CPI**: la previsión es EAC = BAC / CPI (si el rendimiento es malo, CPI < 1, el "
                     "coste final es mayor, no menor)."))
    celdas += apartado(3)
    celdas.append(code("CV, SV = EV - AC, EV - PV\n"
                       "CPI, SPI = EV / AC, EV / PV\n"
                       "EAC = BAC_total / CPI\n"
                       "VAC = BAC_total - EAC\n"
                       "pd.Series({'CV = EV − AC': CV, 'SV = EV − PV': SV, 'CPI = EV / AC': CPI, 'SPI = EV / PV': SPI, 'EAC = BAC / CPI': EAC, 'VAC = BAC − EAC': VAC}).round(3).to_frame('valor').T"))
    celdas.append(code("Markdown(f'''CPI = {CPI:.2f} ({\"sobrecoste\" if CPI < 1 else \"ahorro\"}) y SPI = {SPI:.2f} ({\"retraso\" if SPI < 1 else \"adelanto\"}). "
                       "Previsión de coste final {EAC:.0f} k€ frente a {BAC_total:.0f} presupuestados (VAC = {VAC:.0f}).''')"))
    return celdas, f"{n} actividades; resolución con CPI y SPI intercambiados"



def _inverso(rng):
    """Sólo se conocen las curvas S (PV, EV, AC acumulados por semana): leer el estado del proyecto."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 12 <= i["T"] <= 20):
            return False
        BAC = pd.Series(rng.integers(1, 7, size=len(i["D"])) * 10, index=i["D"].index, name="BAC")
        dur_real, coste_real, es, ef = _simular(rng, i, i["D"], BAC)
        T_real = int(ef.max())
        if not (i["T"] < T_real <= 1.5 * i["T"]):
            return False
        cal = i["calendario"]
        semanas = list(range(1, int(round(0.7 * i["T"])) + 1))
        PV, EV, AC = [], [], []
        for t in semanas:
            frac_plan = ((t - cal["inicio_mas_temprano"]).clip(lower=0).clip(upper=cal["duracion"]) / cal["duracion"])
            PV.append(float((BAC * frac_plan).sum()))
            seg = _seguimiento(t, dur_real, coste_real, es)
            EV.append(float((BAC * seg["performed"] / 100).sum()))
            AC.append(float(seg["AC"].sum()))
        curvas = pd.DataFrame({"PV": PV, "EV": EV, "AC": AC}, index=pd.Index(semanas, name="week")).round(1)
        # queremos: al principio bien, luego retraso y sobrecoste visibles
        spi = curvas["EV"] / curvas["PV"]
        cpi = curvas["EV"] / curvas["AC"].replace(0, np.nan)
        if not (spi.iloc[-1] < 0.9 and cpi.iloc[-1] < 0.95 and (spi.iloc[:3] >= 0.95).all()):
            return False
        estado.update(BAC=BAC, curvas=curvas, T_real=T_real)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    T = int(info["T"])
    BAC, curvas = estado["BAC"], estado["curvas"]
    BAC_total = float(BAC.sum())
    t_fin = int(curvas.index[-1])
    celdas = [md("# Enunciado"),
              md(f"De un proyecto con presupuesto total BAC = {BAC_total:.0f} k€ y duración planificada de {T} "
                 f"semanas sólo se dispone del informe de seguimiento adjunto: los valores acumulados al final de "
                 f"cada semana, hasta la {t_fin}, del valor planificado (PV), el valor ganado (EV) y el coste "
                 "real (AC), en k€. No se conoce el cuadro de actividades.")]
    celdas.append(preguntas([
        "Dibuje las tres curvas S. Calcule CV, SV, CPI y SPI en cada semana y represente la evolución de los "
        "dos índices.",
        "¿En qué semana empezó el proyecto a retrasarse? ¿Y a tener sobrecoste? ¿Fueron simultáneos?",
        f"Con los datos de la semana {t_fin}, estime el coste final (EAC = BAC / CPI) y la duración final por "
        "programación ganada: ES es la semana en que el PV planificado alcanzaba el EV actual; "
        "SPI(t) = ES / semana; duración ≈ duración planificada / SPI(t).",
        "¿Qué no puede saberse sin el cuadro de actividades? Ponga un ejemplo de dos situaciones distintas que "
        "darían las mismas curvas."]))
    celdas.append(code('#| tbl-cap: "Informe de seguimiento (valores acumulados, k€)"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(curvas, indice="week")))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"BAC_total, duracion_planificada = {BAC_total:.0f}, {T}\n"
                       "curvas = datos_enunciado.copy().astype(float)"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Curvas S del proyecto"\n#| label: fig-curvasS\n'
                       "from matplotlib import pyplot as plt\n"
                       "ax = curvas.plot(marker='o', figsize=(8, 4)); ax.set_xlabel('semana'); ax.set_ylabel('k€ acumulados'); ax.grid(alpha=0.3); plt.show()"))
    celdas.append(code('#| tbl-cap: "Indicadores semana a semana"\n#| label: tbl-indicadores\n'
                       "ind = curvas.copy()\n"
                       "ind['CV = EV − AC'] = ind['EV'] - ind['AC']\n"
                       "ind['SV = EV − PV'] = ind['EV'] - ind['PV']\n"
                       "ind['CPI = EV / AC'] = ind['EV'] / ind['AC']\n"
                       "ind['SPI = EV / PV'] = ind['EV'] / ind['PV']\n"
                       "ind.round(3)"))
    celdas.append(code('#| fig-cap: "Evolución de CPI y SPI"\n#| label: fig-indices\n'
                       "ax = ind[['CPI = EV / AC', 'SPI = EV / PV']].plot(marker='o', figsize=(8, 4)); ax.axhline(1, color='grey', linestyle='--'); ax.set_xlabel('semana'); plt.show()"))
    celdas += apartado(2)
    celdas.append(code("umbral = 0.97\n"
                       "semana_retraso = next((int(t) for t, v in ind['SPI = EV / PV'].items() if v < umbral), None)\n"
                       "semana_sobrecoste = next((int(t) for t, v in ind['CPI = EV / AC'].items() if v < umbral), None)\n"
                       "Markdown(f'''Tomando como señal un índice por debajo de {umbral}: el retraso aparece en la semana {semana_retraso} (SPI < {umbral}) y el sobrecoste en la semana {semana_sobrecoste} (CPI < {umbral}). "
                       "{'Fueron simultáneos.' if semana_retraso == semana_sobrecoste else 'No fueron simultáneos: un problema de plazo no implica uno de coste ni al revés (se puede ir despacio gastando poco, o gastar mucho para ir rápido).'}''')"))
    celdas += apartado(3)
    celdas.append(code("t = int(curvas.index[-1])\n"
                       "PV, EV, AC = curvas.loc[t, 'PV'], curvas.loc[t, 'EV'], curvas.loc[t, 'AC']\n"
                       "CPI, SPI = EV / AC, EV / PV\n"
                       "EAC = BAC_total / CPI\n"
                       "# programación ganada: ES = semana en que PV alcanzó el EV actual (interpolando)\n"
                       "pv = curvas['PV']\n"
                       "k = int((pv <= EV).sum())           # última semana con PV ≤ EV\n"
                       "if k == 0:\n"
                       "    ES = EV / pv.iloc[0]\n"
                       "elif k >= len(pv):\n"
                       "    ES = float(pv.index[-1])\n"
                       "else:\n"
                       "    ES = k + (EV - pv.iloc[k - 1]) / (pv.iloc[k] - pv.iloc[k - 1])\n"
                       "SPI_t = ES / t\n"
                       "duracion_estimada = duracion_planificada / SPI_t\n"
                       "Markdown(f'''En la semana {t}: CPI = {EV:.1f} / {AC:.1f} = {CPI:.3f}, EAC = {BAC_total:.0f} / {CPI:.3f} = {EAC:.1f} k€ (VAC = {BAC_total - EAC:.1f}).\n\n"
                       "El PV alcanzaba {EV:.1f} k€ en la semana ES = {ES:.2f} (entre la {k} y la {k + 1}); SPI(t) = {ES:.2f} / {t} = {SPI_t:.3f}; "
                       "duración estimada = {duracion_planificada} / {SPI_t:.3f} = {duracion_estimada:.1f} semanas, es decir, unas {duracion_estimada - duracion_planificada:.1f} semanas de retraso.''')"))
    celdas += apartado(4)
    celdas.append(md("Las curvas agregan todas las actividades: no dicen *qué* actividades van mal ni si son "
                     "críticas. Un SPI = 0,85 puede deberse a una actividad crítica retrasada (el plazo final "
                     "peligra) o a varias no críticas aplazadas dentro de su holgura (el plazo no peligra): las "
                     "mismas curvas, pronósticos opuestos. Igual con el coste: un CPI bajo puede venir de una "
                     "actividad ya terminada (el daño está hecho y no se repetirá) o de una en curso (seguirá "
                     "gastando de más). Por eso el informe debe acompañarse del detalle por actividad."))
    return celdas, f"{n} actividades (ocultas); curvas S hasta la semana {t_fin} de {T}"


def _coste_no_uniforme(rng):
    """El coste planificado de algunas actividades no se reparte uniformemente: PV por perfil de desembolso."""
    n = int(rng.choice([7, 8]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 3 and 10 <= i["T"] <= 18 and (i["D"] >= 2).all()):
            return False
        t = int(round(0.5 * i["T"]))
        cal = i["calendario"]
        en_curso = cal.index[(cal["inicio_mas_temprano"] < t) & (cal["fin_mas_temprano"] > t)].tolist()
        if len(en_curso) < 2:
            return False
        BAC = pd.Series(rng.integers(2, 8, size=len(i["D"])) * 10, index=i["D"].index, name="BAC")
        dur_real, coste_real, es, ef = _simular(rng, i, i["D"], BAC)
        seg = _seguimiento(t, dur_real, coste_real, es)
        if not ((0 < seg["performed"]) & (seg["performed"] < 100)).any():
            return False
        estado.update(t=t, BAC=BAC, seg=seg, en_curso=en_curso)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T, cal = info["D"], int(info["T"]), info["calendario"]
    t, BAC, seg, en_curso = estado["t"], estado["BAC"], estado["seg"], estado["en_curso"]
    # perfil: 'uniforme', 'inicio' (la mitad del coste en la primera semana), 'final' (la mitad en la última)
    perfil = pd.Series("uniforme", index=D.index, name="profile")
    perfil[en_curso[0]] = "inicio"
    perfil[en_curso[1]] = "final"
    otras = [a for a in D.index if a not in en_curso[:2] and D[a] >= 2]
    if otras:
        perfil[otras[int(rng.integers(0, len(otras)))]] = rng.choice(["inicio", "final"])
    df = P.tabla_base(info, duration=D, BAC=BAC, profile=perfil)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas), "
                 "presupuesto (BAC, k€) y *perfil de desembolso* se adjunta. En las actividades de perfil "
                 "`uniforme` el coste se reparte por igual entre sus semanas; en las de perfil `inicio` la mitad "
                 "del BAC se desembolsa en la primera semana (compra de materiales) y el resto por igual entre "
                 "las demás; en las de perfil `final` la mitad se desembolsa en la última semana (pago a la "
                 f"entrega) y el resto por igual entre las anteriores. Al final de la semana {t} se ha anotado el "
                 "seguimiento adjunto (semanas trabajadas, coste real en k€ y porcentaje realizado).")]
    celdas.append(preguntas([
        "Calcule el calendario de inicios más tempranos y, para cada actividad, el coste planificado de cada "
        "semana según su perfil. Compruebe que la suma de cada fila es su BAC.",
        f"Calcule el PV de cada actividad y el total en la semana {t}. Compare con el PV que resultaría "
        "suponiendo todos los perfiles uniformes.",
        "Calcule EV y AC, y los índices CPI y SPI con el PV correcto y con el PV uniforme. ¿Cambia el "
        "diagnóstico?",
        "¿Por qué el EV no depende del perfil de desembolso y el PV sí? ¿Qué perfil hace parecer "
        "«adelantado» un proyecto que va en plazo?"]))
    celdas.append(code('#| tbl-cap: "Planificación del proyecto"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(code('#| tbl-cap: "Seguimiento en la semana de control"\n#| label: tbl-seguimiento\n'
                       + literal_dataframe(seg, nombre="seguimiento")))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       f"semana_control = {t}\n"
                       "BAC_total = float(datos_proyecto['BAC'].sum())"))
    celdas += apartado(1)
    celdas.append(code("def perfil_semanal(BAC, d, perfil):\n"
                       "    d = int(d)\n"
                       "    if perfil == 'uniforme' or d == 1:\n"
                       "        return [BAC / d] * d\n"
                       "    resto = [BAC / 2 / (d - 1)] * (d - 1)\n"
                       "    return [BAC / 2] + resto if perfil == 'inicio' else resto + [BAC / 2]\n"
                       "duracion_planificada = proyecto.duration(datos_proyecto['duration'])\n"
                       "coste_semanal = pd.DataFrame(0.0, index=datos_proyecto.index, columns=range(1, duracion_planificada + 1))\n"
                       "for a, fila in datos_proyecto.iterrows():\n"
                       "    es = int(calendario.loc[a, 'inicio_mas_temprano'])\n"
                       "    for j, c in enumerate(perfil_semanal(float(fila['BAC']), fila['duration'], fila['profile'])):\n"
                       "        coste_semanal.loc[a, es + j + 1] = c\n"
                       "coste_semanal['suma'] = coste_semanal.sum(axis=1)\n"
                       "coste_semanal.round(1)"))
    celdas.append(code("Markdown(f'Cada fila suma su BAC: {bool(np.allclose(coste_semanal[\"suma\"], datos_proyecto[\"BAC\"]))}.')"))
    celdas += apartado(2)
    celdas.append(code("semanas = [c for c in coste_semanal.columns if c != 'suma' and c <= semana_control]\n"
                       "PV_i = coste_semanal[semanas].sum(axis=1)\n"
                       "frac = ((semana_control - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion'])\n"
                       "PV_uniforme_i = datos_proyecto['BAC'] * frac\n"
                       "comparacion = pd.DataFrame({'perfil': datos_proyecto['profile'], 'PV según perfil': PV_i, 'PV uniforme': PV_uniforme_i})\n"
                       "comparacion.loc['Total'] = ['', PV_i.sum(), PV_uniforme_i.sum()]\n"
                       "comparacion.round(1)"))
    celdas.append(code("PV, PV_uniforme = float(PV_i.sum()), float(PV_uniforme_i.sum())\n"
                       "Markdown(f'PV = {PV:.1f} k€ con los perfiles reales frente a {PV_uniforme:.1f} con reparto uniforme; la diferencia está sólo en las actividades en curso con perfil no uniforme.')"))
    celdas += apartado(3)
    celdas.append(code("EV = float((datos_proyecto['BAC'] * seguimiento['performed'] / 100).sum())\n"
                       "AC = float(seguimiento['AC'].sum())\n"
                       "indices = pd.DataFrame({'PV según perfil': [PV, EV, AC, EV / AC, EV / PV], 'PV uniforme': [PV_uniforme, EV, AC, EV / AC, EV / PV_uniforme]},\n"
                       "                       index=['PV', 'EV', 'AC', 'CPI = EV / AC', 'SPI = EV / PV']).round(3)\n"
                       "indices"))
    celdas.append(code("s1, s2 = EV / PV, EV / PV_uniforme\n"
                       "Markdown(f'''SPI = {EV:.1f} / {PV:.1f} = {s1:.3f} con el perfil correcto y {EV:.1f} / {PV_uniforme:.1f} = {s2:.3f} con el uniforme: "
                       "{'el diagnóstico de plazo cambia de signo' if (s1 < 1) != (s2 < 1) else 'el signo del diagnóstico no cambia, pero sí su magnitud'}. "
                       "El CPI no cambia: no depende del PV.''')"))
    celdas += apartado(4)
    celdas.append(md("El EV es el BAC por el porcentaje *físico* realizado: mide trabajo hecho, no dinero "
                     "desembolsado, y por eso no depende de cuándo se planeó pagar. El PV sí es dinero "
                     "planificado hasta la fecha. Si una actividad en plazo tiene perfil `final` (el gran pago "
                     "al terminar), su PV a mitad de camino es pequeño y el EV (proporcional al avance físico) lo "
                     "supera: el proyecto parece adelantado sin estarlo. Con perfil `inicio` ocurre lo contrario. "
                     "Para que el SPI mida plazo, el PV debe seguir el mismo criterio que el EV."))
    return celdas, f"{n} actividades; perfiles de desembolso no uniformes; control en la semana {t}"


def _vigilar(rng):
    """Qué actividades vigilar: índices por actividad cruzados con su criticidad."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 12 <= i["T"] <= 20):
            return False
        BAC = pd.Series(rng.integers(1, 7, size=len(i["D"])) * 10, index=i["D"].index, name="BAC")
        dur_real, coste_real, es, ef = _simular(rng, i, i["D"], BAC)
        t = int(round(0.55 * i["T"]))
        seg = _seguimiento(t, dur_real, coste_real, es)
        cal = i["calendario"]
        activas = seg.index[(seg["actual_duration"] > 0)]
        if len(activas) < 5:
            return False
        frac = ((t - cal["inicio_mas_temprano"]).clip(lower=0).clip(upper=cal["duracion"]) / cal["duracion"])
        PV_i = BAC * frac
        EV_i = BAC * seg["performed"] / 100
        AC_i = seg["AC"].astype(float)
        cpi = (EV_i / AC_i.replace(0, np.nan))
        spi = (EV_i / PV_i.replace(0, np.nan))
        criticas = cal.index[cal["H_total"] == 0]
        # queremos: alguna crítica con SPI<1, alguna no crítica con SPI<1 y alguna con CPI<1 pero SPI>=1
        if not ((spi[criticas] < 0.95).any() and (spi.drop(criticas) < 0.95).any() and ((cpi < 0.95) & (spi >= 1)).any()):
            return False
        estado.update(BAC=BAC, t=t, seg=seg)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    BAC, t, seg = estado["BAC"], estado["t"], estado["seg"]
    df = P.tabla_base(info, duration=D, BAC=BAC)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones planificadas (semanas) y "
                 "presupuesto por actividad (BAC, k€, repartido uniformemente) se adjunta, junto con el "
                 f"seguimiento al final de la semana {t}. El jefe de proyecto quiere una lista corta de "
                 "actividades sobre las que actuar.")]
    celdas.append(preguntas([
        "Calcule PV, EV y AC de cada actividad y los índices CPI y SPI *por actividad*; añada la holgura "
        "total planificada de cada una.",
        "Clasifique las actividades en cuatro grupos: críticas con retraso (SPI < 1), no críticas con "
        "retraso, actividades con sobrecoste (CPI < 1) sin retraso, y el resto. ¿Cuáles amenazan el plazo? "
        "¿Cuáles sólo el presupuesto?",
        "Calcule los índices globales del proyecto. ¿Qué actividades explican la mayor parte de SV y de CV "
        "(ordene por contribución)?",
        "Proponga las tres actividades a vigilar y justifique el criterio: ¿por qué un SPI bajo en una "
        "actividad con mucha holgura preocupa menos que en una crítica, y por qué con el CPI no hay esa "
        "distinción?"]))
    celdas.append(code('#| tbl-cap: "Planificación del proyecto"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(code('#| tbl-cap: "Seguimiento en la semana de control"\n#| label: tbl-seguimiento\n'
                       + literal_dataframe(seg, nombre="seguimiento")))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       f"semana_control = {t}"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Valor ganado por actividad"\n#| label: tbl-porActividad\n'
                       "datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                       "detalle = detalle_valor_ganado(datos_valor_ganado, calendario, semana_control)\n"
                       "por_actividad = pd.DataFrame({'PV': detalle['PV_i'].astype(float), 'EV': detalle['EV_i'].astype(float), 'AC': seguimiento['AC'].astype(float),\n"
                       "                              'H_total': calendario['H_total']})\n"
                       "por_actividad['CPI'] = por_actividad['EV'] / por_actividad['AC'].replace(0, np.nan)\n"
                       "por_actividad['SPI'] = por_actividad['EV'] / por_actividad['PV'].replace(0, np.nan)\n"
                       "por_actividad.round(3)"))
    celdas += apartado(2)
    celdas.append(code("p = por_actividad\n"
                       "grupos = {\n"
                       "    'críticas con retraso (amenazan el plazo)': p.index[(p['H_total'] == 0) & (p['SPI'] < 1)],\n"
                       "    'no críticas con retraso': p.index[(p['H_total'] > 0) & (p['SPI'] < 1)],\n"
                       "    'sobrecoste sin retraso (amenazan el presupuesto)': p.index[(p['CPI'] < 1) & ~(p['SPI'] < 1)],\n"
                       "}\n"
                       "grupos['resto'] = p.index.difference(pd.Index([]).append([pd.Index(v) for v in grupos.values()]))\n"
                       "pd.Series({k: ', '.join(v) or '—' for k, v in grupos.items()}, name='actividades').to_frame()"))
    celdas.append(code("retraso_no_critica = p.index[(p['H_total'] > 0) & (p['SPI'] < 1)]\n"
                       "retraso_semanas = ((1 - p.loc[retraso_no_critica, 'SPI']) * (datos_proyecto.loc[retraso_no_critica, 'duration'])).round(1)\n"
                       "Markdown(f'''Las no críticas retrasadas sólo amenazan el plazo si su retraso supera su holgura. Estimación grosera del retraso (semanas de trabajo que faltan respecto al plan): "
                       "{', '.join(f'{a}: {r} frente a holgura {int(p.loc[a, \"H_total\"])}' for a, r in retraso_semanas.items()) or 'no hay'}.''')"))
    celdas += apartado(3)
    celdas.append(code("PV, EV, AC = p['PV'].sum(), p['EV'].sum(), p['AC'].sum()\n"
                       "contrib = pd.DataFrame({'SV_i = EV − PV': p['EV'] - p['PV'], 'CV_i = EV − AC': p['EV'] - p['AC']})\n"
                       "display(Markdown(f'Proyecto: PV = {PV:.1f}, EV = {EV:.1f}, AC = {AC:.1f}; SV = {EV - PV:.1f}, CV = {EV - AC:.1f}; SPI = {EV / PV:.3f}, CPI = {EV / AC:.3f}.'))\n"
                       "display(contrib.sort_values('SV_i = EV − PV').round(1).T)\n"
                       "Markdown(f'Las variaciones del proyecto son la suma de las de sus actividades: SV = {contrib.iloc[:, 0].sum():.1f}, CV = {contrib.iloc[:, 1].sum():.1f}. '\n"
                       "         f'La mayor contribución negativa a SV es de {contrib.iloc[:, 0].idxmin()} y a CV de {contrib.iloc[:, 1].idxmin()}.')"))
    celdas += apartado(4)
    celdas.append(code("puntuacion = pd.DataFrame({'critica': (p['H_total'] == 0).astype(int), 'SV_i': contrib.iloc[:, 0], 'CV_i': contrib.iloc[:, 1]})\n"
                       "puntuacion['prioridad'] = (-puntuacion['SV_i'].clip(upper=0) * (2 + puntuacion['critica'] * 3) - puntuacion['CV_i'].clip(upper=0))\n"
                       "vigilar = puntuacion.sort_values('prioridad', ascending=False).index[:3].tolist()\n"
                       "display(puntuacion.sort_values('prioridad', ascending=False).round(1))\n"
                       "Markdown(f'''Actividades a vigilar: **{', '.join(vigilar)}** (criterio: retraso ponderado por criticidad más sobrecoste). "
                       "Un SPI bajo en una actividad con holgura puede absorberse sin mover el plazo; en una crítica se traslada entero al final. "
                       "El dinero, en cambio, no tiene holgura: cada euro de sobrecoste, crítico o no, se suma al coste final, por lo que el CPI se vigila en todas.''')"))
    return celdas, f"{n} actividades; qué vigilar cruzando SPI/CPI por actividad con la holgura"


def _prevision_incertidumbre(rng):
    """Valor ganado con duraciones y costes aleatorios: cuánto se puede fiar de SPI y CPI en cada fecha de control."""
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 10 and 12 <= i["T"] <= 24):
            return False
        nombres = list(i["D"].index)
        a, m, b = P.tres_estimaciones_triangular(rng, i["D"])
        if ((b - a) < 2).sum() > len(nombres) // 2:
            return False
        BAC = pd.Series(rng.integers(2, 9, size=len(nombres)) * 25, index=nombres, name="BAC")
        dist = {x: ("triangular", float(a[x]), float(m[x]), float(b[x])) for x in nombres}
        fc = {x: ("triangular", 0.9, 1.0, 1.4) for x in nombres}
        T = int(i["T"])
        puntos = [max(2, round(T * f)) for f in (0.25, 0.5, 0.75)]
        try:
            mae = [i["proyecto"].monte_carlo_valor_ganado(dist, m, BAC, t, n_samples=600, seed=1, factor_coste=fc)
                   .calidad_prevision().loc["duración", "error absoluto medio"] for t in puntos]
        except Exception:
            return False
        if not (mae[2] < mae[0] and mae[0] > 0.3):
            return False
        r = i["proyecto"].monte_carlo_valor_ganado(dist, m, BAC, puntos[1], n_samples=600, seed=1, factor_coste=fc)
        umbral = float(np.round(np.quantile(r.spi, 0.3), 2))
        ret = r.tau_samples > r.T_plan
        al = r.spi < umbral
        if not (0.6 <= umbral <= 1.05 and 0.3 <= ret.mean() <= 0.9 and al.sum() > 30
                and ret[al].mean() - ret[~al].mean() >= 0.1):
            return False
        estado.update(a=a, m=m, b=b, BAC=BAC, puntos=puntos, umbral=umbral)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), max_intentos=6000)
    a, m, b, BAC, puntos, umbral = estado["a"], estado["m"], estado["b"], estado["BAC"], estado["puntos"], estado["umbral"]
    T = int(info["T"])
    df = P.tabla_base(info, duration=info["D"], a=a, m=m, b=b, BAC=BAC)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, estimaciones de duración (mínima *a*, más probable *m* y "
                 "máxima *b*, semanas, distribución triangular) y presupuesto (BAC, k€) se adjunta. Se toma como línea base el "
                 "calendario más temprano con las duraciones más probables *m*; el coste se reparte uniformemente en cada "
                 "actividad. Los costes reales de cada actividad pueden desviarse: factor multiplicativo triangular (0,9; 1,0; 1,4).")]
    celdas.append(preguntas([
        "Calcule la línea base: duración prevista y valor planificado (PV) en cada fecha de control.",
        f"Simule 3 000 ejecuciones y evalúe SPI y CPI en las semanas {puntos[0]}, {puntos[1]} y {puntos[2]}. Describa la "
        "distribución de ambos índices en cada fecha.",
        "En cada fecha, prevea la duración con T/SPI y el coste con BAC/CPI y compárelos con lo que ocurre de verdad en cada "
        "ejecución: sesgo, error absoluto medio y correlación. ¿Cómo cambia la calidad de la previsión con el avance?",
        f"Un jefe de proyecto activa una alarma si SPI < {umbral:.2f}. En la fecha intermedia, ¿qué probabilidad de terminar con retraso "
        "hay con y sin alarma? ¿Qué proporción de proyectos con retraso avisa la alarma (sensibilidad) y qué proporción de "
        "alarmas son falsas?"]))
    celdas.append(code('#| tbl-cap: "Datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"puntos = {puntos}\n"
                       f"umbral = {umbral}\n"
                       "dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}\n"
                       "factor_coste = {a: ('triangular', 0.9, 1.0, 1.4) for a in datos_proyecto.index}\n"
                       "plan = datos_proyecto['m']"))
    celdas += apartado(1)
    celdas.append(code("T_plan = proyecto.duration(plan)\n"
                       "Markdown(f'Línea base: {T_plan} semanas, BAC = {datos_proyecto[\"BAC\"].sum():.0f} k€.')"))
    celdas += apartado(2)
    celdas.append(code("resultados = {t: proyecto.monte_carlo_valor_ganado(dists, plan, datos_proyecto['BAC'], t, n_samples=3000, seed=11, factor_coste=factor_coste)\n"
                       "              for t in puntos}\n"
                       "pd.concat({f'semana {t}': r.summary().round(3) for t, r in resultados.items()})"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Calidad de las previsiones en cada fecha de control"\n#| label: tbl-calidad\n'
                       "pd.concat({f'semana {t}': r.calidad_prevision().round(3) for t, r in resultados.items()})"))
    celdas.append(code("mae = {t: r.calidad_prevision().loc['duración', 'error absoluto medio'] for t, r in resultados.items()}\n"
                       "Markdown('El error absoluto medio de la previsión de duración pasa de ' + ' a '.join(f'{v:.2f}' for v in mae.values()) +\n"
                       "         ' semanas al avanzar el proyecto: cuanto más hecho, menos queda por incertidumbre; y T/SPI es sesgado al principio, '\n"
                       "         'porque el avance temprano incluye la parte aún no afectada por los retrasos de las actividades futuras.')"))
    celdas += apartado(4)
    celdas.append(code("r = resultados[puntos[1]]\n"
                       "retraso = r.tau_samples > r.T_plan\n"
                       "alarma = r.spi < umbral\n"
                       "p_con, p_sin = retraso[alarma].mean(), retraso[~alarma].mean()\n"
                       "sens = alarma[retraso].mean()\n"
                       "falsas = (~retraso[alarma]).mean()\n"
                       "Markdown(f'P(retraso | alarma) = {p_con:.3f}; P(retraso | sin alarma) = {p_sin:.3f}; sensibilidad = {sens:.3f}; '\n"
                       "         f'alarmas falsas = {falsas:.3f}. La alarma discrimina, pero no es infalible: el SPI en una fecha es una señal con ruido.')"))
    return celdas, f"duraciones triangulares y costes inciertos; controles en las semanas {puntos}"


def _ev_recursos(rng):
    """Línea base que ignora los recursos: el SPI cae desde la primera espera aunque nadie trabaje peor."""
    estado = {}

    def acepta(i):
        if not (0 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 24):
            return False
        nombres = list(i["D"].index)
        op = P.recursos(rng, nombres, hi=3).rename("operarios")
        L = int(op.max())
        BAC = pd.Series(rng.integers(2, 9, size=len(nombres)) * 25, index=nombres, name="BAC")
        df = P.tabla_base(i, duration=i["D"], operarios=op, BAC=BAC)
        try:
            t = i["proyecto"].valor_ganado_recursos(df, "duration", ["operarios"], [L], BAC)
        except Exception:
            return False
        Tp, Tr = t.attrs["T_plan"], t.attrs["T_real"]
        if not (2 <= Tr - Tp <= 5):
            return False
        tc = int(round(Tp / 2))
        if not (t.loc[tc, "SPI"] <= 0.93):
            return False
        estado.update(df=df, L=L, tc=tc)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), max_intentos=8000)
    df, L, tc = estado["df"], estado["L"], estado["tc"]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duración (semanas), operarios necesarios y presupuesto "
                 "(BAC, k€) de cada actividad se adjunta; el coste se reparte uniformemente en cada actividad. La línea base se "
                 "preparó con el calendario más temprano, sin mirar los recursos, pero la empresa dispone como máximo de "
                 f"**{L} operarios** por semana: las actividades se ejecutan con el algoritmo de asignación (cada una empieza en "
                 "cuanto puede), sin que nadie trabaje más lento de lo previsto. El coste real coincide con el presupuesto.")]
    celdas.append(preguntas([
        "Calcule la línea base (duración y PV semana a semana) y el calendario real con el límite de operarios.",
        "Calcule EV, SPI y la duración prevista T/SPI semana a semana. ¿Desde qué semana cae el SPI por debajo de 1 y qué "
        "actividad es la primera en esperar?",
        f"En la semana {tc} (mitad de la línea base) ¿qué duración prevé el SPI? Compárela con la duración real. ¿Qué previsión "
        "es la acertada y por qué?",
        "¿Qué debería haber hecho el equipo al preparar la línea base? Calcule el SPI semana a semana con la línea base que "
        "respeta el límite de operarios y explique la diferencia."]))
    celdas.append(code('#| tbl-cap: "Datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"limite = [{L}]\n"
                       f"t_control = {tc}\n"
                       "BAC = datos_proyecto['BAC']"))
    celdas += apartado(1)
    celdas.append(code("tabla = proyecto.valor_ganado_recursos(datos_proyecto, 'duration', ['operarios'], limite, BAC)\n"
                       "res = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', ['operarios'], limite)\n"
                       "T_plan, T_real = tabla.attrs['T_plan'], tabla.attrs['T_real']\n"
                       "Markdown(f'Línea base {T_plan:.0f} semanas; con {limite[0]} operarios el calendario real dura {T_real:.0f}.')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "PV, EV, SPI y duración prevista semana a semana"\n#| label: tbl-valorGanado\n'
                       "tabla.round(3)"))
    celdas.append(code("primera = tabla.index[tabla['SPI'] < 0.999].min()\n"
                       "esperas = res.traza[res.traza.decision == 'espera'].sort_values('periodo')\n"
                       "primera_actividad = esperas.iloc[0]['actividad']\n"
                       "Markdown(f'El SPI cae por debajo de 1 en la semana {primera}; la primera actividad que espera es {primera_actividad} '\n"
                       "         f'(semana {int(esperas.iloc[0][\"periodo\"])}).')"))
    celdas += apartado(3)
    celdas.append(code("fila = tabla.loc[t_control]\n"
                       "Markdown(f'En la semana {t_control}: SPI = {fila[\"SPI\"]:.3f}, duración prevista T/SPI = {fila[\"duración prevista\"]:.1f} semanas; '\n"
                       "         f'duración real {T_real:.0f}: el SPI {\"sobrestima\" if fila[\"duración prevista\"] > T_real else \"subestima\"} el retraso. '\n"
                       "         'Supone que la fracción del plan que se logra cada semana se mantiene hasta el final, cuando el límite de recursos '\n"
                       "         'solo frena algunos tramos del proyecto.')"))
    celdas += apartado(4)
    celdas.append(code("tabla2 = tabla[['PV', 'EV']].copy()\n"
                       "tabla2['PV con recursos'] = tabla['EV']        # línea base replanificada con el límite\n"
                       "tabla2['SPI con línea base realista'] = tabla2['EV'] / tabla2['PV con recursos']\n"
                       "display(tabla2.round(3))\n"
                       "Markdown('Con la línea base calculada respetando el límite de operarios el SPI es 1 en todas las semanas: los problemas de recursos '\n"
                       "         'se detectan al planificar (con la asignación), no al controlar. Un SPI < 1 desde la primera espera es síntoma de una '\n"
                       "         'línea base irrealista, no de un equipo lento.')"))
    return celdas, f"límite de {L} operarios y línea base sin recursos"


VARIANTES = [
    Variante("indicadores-eac", 1, "tres Gantts, PV/EV/AC, CV/SV, CPI/SPI, tres fórmulas de EAC, VAC y TCPI", lambda rng: _base(1, rng)),
    Variante("earned-schedule", 2, "programación ganada: ES, SPI(t), previsión de duración frente a la de SPI", lambda rng: _base(2, rng)),
    Variante("tendencia", 2, "tres puntos de control: evolución de CPI, SPI y EAC", lambda rng: _base(3, rng)),
    Variante("coste-bien-plazo-mal", 2, "gastar menos no es ir bien: CPI > 1 con SPI < 1", lambda rng: _situacion(rng, "coste-bien-plazo-mal")),
    Variante("plazo-bien-coste-mal", 2, "ir adelantado a costa de sobrecoste: SPI ≥ 1 con CPI < 1", lambda rng: _situacion(rng, "plazo-bien-coste-mal")),
    Variante("diagnostico", 2, "resolución entregada con CPI y SPI intercambiados, CV con el signo cambiado y EAC mal", _diagnostico_vg),
    Variante("inverso-desde-curvas", 2, "sólo las curvas S: índices semana a semana, cuándo empezó el retraso y el sobrecoste, EAC y programación ganada",
             _inverso),
    Variante("coste-no-uniforme", 3, "perfiles de desembolso (inicio/final/uniforme): PV correcto frente a uniforme y efecto sobre el SPI",
             _coste_no_uniforme),
    Variante("vigilar", 2, "SPI y CPI por actividad cruzados con la holgura: qué amenaza el plazo, qué el presupuesto y qué vigilar",
             _vigilar),
    Variante("prevision-incertidumbre", 3, "SPI y CPI con duraciones y costes aleatorios: calidad de T/SPI y BAC/CPI en cada fecha de control y fiabilidad de una alarma",
             _prevision_incertidumbre),
    Variante("ev-recursos", 3, "línea base sin recursos y ejecución con límite de operarios: el SPI cae desde la primera espera",
             _ev_recursos),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
