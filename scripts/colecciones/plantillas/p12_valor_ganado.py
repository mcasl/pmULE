"""Colección 12 — Valor ganado: Gantts PV/EV/AC, indicadores, previsiones
EAC/VAC/TCPI, programación ganada (Earned Schedule) y seguimiento en el tiempo."""
import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

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


def generar(k, rng):
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
    celdas.append(code('#| fig-cap: "Grafo PERT de la planificación"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
    ap = 1
    params = ("params=dict(period_width=1, row_height=0.7, activity_relative_height=0.6,\n"
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
