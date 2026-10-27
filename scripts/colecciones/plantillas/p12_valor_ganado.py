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

# Siglas del método (se desarrollan al principio de cada resolución).
_SIGLAS = ("**Notación** (siglas del PMBOK): BAC (*budget at completion*), presupuesto total del proyecto; "
           "PV (*planned value*), valor planificado: presupuesto del trabajo que, según el plan, debía estar hecho "
           "en la fecha de control; EV (*earned value*), valor ganado: presupuesto del trabajo realmente hecho; "
           "AC (*actual cost*), coste real de ese trabajo. Con ellos: CV = EV − AC (*cost variance*), variación del "
           "coste; SV = EV − PV (*schedule variance*), variación del plazo; CPI = EV / AC (*cost performance "
           "index*), índice de rendimiento del coste; SPI = EV / PV (*schedule performance index*), índice de "
           "rendimiento del plazo. Una variación negativa o un índice menor que 1 indica sobrecoste (CV, CPI) o "
           "retraso (SV, SPI).")

# Celda auxiliar (se define en el cuaderno): sumas, indicadores y previsiones con los números sustituidos.
_AUX_VG = r'''
def suma_vg(detalle, columna, nombre):
    """«PV = A(50.0) + B(40.0) + … = 261.0 k€» con las actividades que aportan algo."""
    v = detalle[columna].astype(float)
    nz = v[v.abs() > 1e-9]
    if len(nz) == 0:
        return f'{nombre} = **0** k€'
    otras = ' (las demás aportan 0)' if len(nz) < len(v) else ''
    return f'{nombre} = ' + ' + '.join(f'{a}({x:.1f})' for a, x in nz.items()) + f' = **{v.sum():.1f}** k€{otras}'


def lectura_vg(PV, EV, AC):
    """Lectura del coste y del plazo con los números: (texto coste, texto plazo)."""
    CV, SV, CPI, SPI = EV - AC, EV - PV, EV / AC, EV / PV
    if abs(CV) < 1e-9:
        coste = f'CV = 0 y CPI = 1: el coste va **según presupuesto**'
    elif CV < 0:
        coste = (f'CV = {CV:.1f} k€ < 0 y CPI = {CPI:.3f} < 1: va **por encima del presupuesto** (sobrecoste de '
                 f'{-CV:.1f} k€; cada k€ gastado ha producido solo {CPI:.3f} k€ de valor)')
    else:
        coste = (f'CV = {CV:.1f} k€ > 0 y CPI = {CPI:.3f} > 1: va **por debajo del presupuesto** (ahorro de '
                 f'{CV:.1f} k€; cada k€ gastado ha producido {CPI:.3f} k€ de valor)')
    if abs(SV) < 1e-9:
        plazo = 'SV = 0 y SPI = 1: va **en plazo**'
    elif SV < 0:
        plazo = (f'SV = {SV:.1f} k€ < 0 y SPI = {SPI:.3f} < 1: va **retrasado** (se ha hecho el {100 * SPI:.1f} % '
                 f'del trabajo que el plan preveía para esta fecha)')
    else:
        plazo = (f'SV = {SV:.1f} k€ > 0 y SPI = {SPI:.3f} > 1: va **adelantado** (se ha hecho el {100 * SPI:.1f} % '
                 f'del trabajo que el plan preveía para esta fecha)')
    return coste, plazo


def explica_vg(detalle, PV, EV, AC, respuesta=True):
    """Sumas por actividad, CV, SV, CPI y SPI con los números sustituidos y su lectura."""
    CV, SV, CPI, SPI = EV - AC, EV - PV, EV / AC, EV / PV
    lineas = []
    if detalle is not None:
        lineas += ['Sumando las columnas de la tabla:', '',
                   '- ' + suma_vg(detalle, 'PV_i', 'PV'), '- ' + suma_vg(detalle, 'EV_i', 'EV'),
                   '- ' + suma_vg(detalle, 'AC', 'AC'), '', 'Variaciones e índices:', '']
    lineas += [f'- CV = EV − AC = {EV:.1f} − {AC:.1f} = **{CV:.1f}** k€',
               f'- SV = EV − PV = {EV:.1f} − {PV:.1f} = **{SV:.1f}** k€',
               f'- CPI = EV / AC = {EV:.1f} / {AC:.1f} = **{CPI:.3f}**',
               f'- SPI = EV / PV = {EV:.1f} / {PV:.1f} = **{SPI:.3f}**']
    if respuesta:
        coste, plazo = lectura_vg(PV, EV, AC)
        lineas += ['', f'**Respuesta:** PV = {PV:.1f} k€, EV = {EV:.1f} k€ y AC = {AC:.1f} k€. En coste, {coste}. '
                       f'En plazo, {plazo}.']
    return Markdown('\n'.join(lineas))


def explica_eac(BAC, PV, EV, AC):
    """Las tres previsiones de EAC con ETC y VAC, el orden entre ellas y por qué."""
    CPI, SPI = EV / AC, EV / PV
    resto = BAC - EV
    hip = {'(a)': (1.0, '1'), '(b)': (CPI, f'CPI = {CPI:.3f}'),
           '(c)': (CPI * SPI, f'CPI · SPI = {CPI:.3f} · {SPI:.3f} = {CPI * SPI:.3f}')}
    eac = {h: AC + resto / d for h, (d, _) in hip.items()}
    lin = [f'Trabajo pendiente valorado a precio de presupuesto: BAC − EV = {BAC:.0f} − {EV:.1f} = {resto:.1f} k€. '
           'Cada hipótesis lo divide por un índice distinto para obtener lo que costará de verdad (ETC, *estimate to '
           'complete*); después EAC = AC + ETC (*estimate at completion*) y VAC = BAC − EAC (*variance at completion*):', '']
    for h, (d, txt) in hip.items():
        etc = resto / d
        lin.append(f'- **{h}** índice {txt}: ETC = {resto:.1f} / {d:.3f} = {etc:.1f} k€; EAC = {AC:.1f} + {etc:.1f} = '
                   f'**{eac[h]:.1f}** k€; VAC = {BAC:.0f} − {eac[h]:.1f} = **{BAC - eac[h]:.1f}** k€'
                   + (f' (equivale a BAC / CPI = {BAC:.0f} / {CPI:.3f})' if h == '(b)' else ''))
    orden = sorted(hip, key=lambda h: eac[h])
    lin += ['', f'Cuanto menor es el índice que divide al trabajo pendiente, mayor es la previsión. Aquí los índices son '
                f'{lista_y([f"{h} {hip[h][0]:.3f}" for h in hip])}, así que de menor a mayor EAC el orden es '
                f'{lista_y(orden)}.']
    expl = []
    if CPI < 1:
        expl.append(f'Con CPI = {CPI:.3f} < 1, la (a) supone que el sobrecoste sufrido hasta ahora no se repetirá en lo que queda')
    elif CPI > 1:
        expl.append(f'Con CPI = {CPI:.3f} > 1, la (a) renuncia al ahorro que se viene obteniendo y es más prudente que la (b)')
    if SPI < 1:
        expl.append(f'con SPI = {SPI:.3f} < 1 el retraso añade un factor que encarece el resto, y la (c) queda por encima de la (b)')
    elif SPI > 1:
        expl.append(f'con SPI = {SPI:.3f} > 1 el adelanto abarata el resto en la (c), que queda por debajo de la (b)')
    if expl:
        lin[-1] += ' ' + '; '.join(expl) + '.'
    lin += ['', f'**Respuesta:** EAC (a) = {eac["(a)"]:.1f} k€ (VAC = {BAC - eac["(a)"]:.1f}), (b) = {eac["(b)"]:.1f} k€ '
                f'(VAC = {BAC - eac["(b)"]:.1f}), (c) = {eac["(c)"]:.1f} k€ (VAC = {BAC - eac["(c)"]:.1f}). La más optimista '
                f'es la {orden[0]} y la más pesimista la {orden[-1]}. Como previsión central se suele tomar la (b): '
                f'**{eac["(b)"]:.1f} k€ frente a los {BAC:.0f} k€ presupuestados**.']
    return Markdown('\n'.join(lin))


def explica_tcpi(BAC, EV, AC):
    """TCPI con los números y si es realista frente al CPI."""
    CPI = EV / AC
    resto, fondos = BAC - EV, BAC - AC
    if fondos <= 0:
        return Markdown(f'TCPI = (BAC − EV) / (BAC − AC) = ({BAC:.0f} − {EV:.1f}) / ({BAC:.0f} − {AC:.1f}) = {resto:.1f} / {fondos:.1f}: '
                        f'ya se ha gastado todo el presupuesto y queda trabajo por valor de {resto:.1f} k€. **Respuesta:** terminar en '
                        'presupuesto es imposible; hay que pedir más fondos (rehacer la línea base de coste).')
    TCPI = resto / fondos
    mejora = (TCPI / CPI - 1) * 100
    if TCPI <= CPI:
        juicio = (f'basta con mantener el rendimiento actual (CPI = {CPI:.3f} ≥ TCPI): **es realista**')
    elif mejora <= 10:
        juicio = (f'exige mejorar el rendimiento actual (CPI = {CPI:.3f}) un {mejora:.0f} %: **es alcanzable** con medidas correctoras')
    else:
        juicio = (f'exige mejorar el rendimiento actual (CPI = {CPI:.3f}) un {mejora:.0f} % (TCPI / CPI − 1 = {TCPI:.3f} / {CPI:.3f} − 1): '
                  '**no es realista** sin cambios de fondo (alcance, método o contrato); lo prudente es aceptar un EAC mayor que el BAC')
    return Markdown(f'TCPI = (BAC − EV) / (BAC − AC) = ({BAC:.0f} − {EV:.1f}) / ({BAC:.0f} − {AC:.1f}) = {resto:.1f} / {fondos:.1f} = '
                    f'**{TCPI:.3f}**: queda trabajo por valor de {resto:.1f} k€ y solo quedan {fondos:.1f} k€ de presupuesto, así que '
                    f'cada k€ que se gaste a partir de ahora tendría que producir {TCPI:.3f} k€ de valor.\n\n'
                    f'**Respuesta:** TCPI = {TCPI:.3f}; {juicio}.')


def explica_es(pv_acumulado, EV, AT):
    """Programación ganada: (ES, texto con la interpolación). pv_acumulado indexado por semana (incluye la 0)."""
    pv = pv_acumulado.astype(float)
    C = int(pv[pv <= EV + 1e-9].index.max())
    if C + 1 not in pv.index:
        return float(C), f'El EV = {EV:.1f} k€ alcanza el PV acumulado de la última semana del plan: ES = {C}.'
    PV_C, PV_C1 = float(pv[C]), float(pv[C + 1])
    ES = C + (EV - PV_C) / (PV_C1 - PV_C)
    txt = (f'El EV = {EV:.1f} k€ queda entre el PV acumulado de la semana {C} ({PV_C:.1f}) y el de la semana {C + 1} '
           f'({PV_C1:.1f}), así que C = {C} y ES = {C} + ({EV:.1f} − {PV_C:.1f}) / ({PV_C1:.1f} − {PV_C:.1f}) = '
           f'{C} + {EV - PV_C:.1f} / {PV_C1 - PV_C:.1f} = **{ES:.2f}** semanas: el valor ganado hasta la semana {AT} '
           f'se había previsto ganarlo en la semana {ES:.2f}.')
    return ES, txt
'''


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
    celdas.append(md(_SIGLAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "BAC_total = float(datos_proyecto['BAC'].sum())\n"
                       "duracion_planificada = proyecto.duration(datos_proyecto['duration'])"))
    celdas.append(code(_AUX_VG))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    ap = 1
    params = ("params=dict(period_width=1, row_height=0.7, activity_relative_height=0.6, **GANTT_PARAMS,\n"
              "                                       critical_background_color='white!90!green', regular_background_color='white!90!green',\n"
              "                                       regular_text_color='black', critical_text_color='black')")
    if variante in (1, 2):
        celdas += apartado(ap); ap += 1
        celdas.append(md("En el Gantt del valor planificado (PV) cada casilla lleva el coste planificado de la semana "
                         "(BAC / duración), colocado según el calendario de inicios más tempranos. En el del valor ganado "
                         "(EV), el BAC multiplicado por el porcentaje realizado se reparte entre las semanas trabajadas."
                         + (" En el del coste real (AC), el coste incurrido se reparte entre las semanas trabajadas."
                            if variante == 1 else "")))
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
                           "CPI, SPI = EV / AC, EV / PV"))
        celdas.append(md("De dónde salen las tres cifras: el valor planificado de cada actividad es su BAC por la "
                         "fracción de su duración que debería haberse ejecutado en la semana de control (según el "
                         "calendario de inicios más tempranos); el valor ganado, su BAC por el porcentaje realizado; "
                         "el coste real, lo gastado en ella. Los del proyecto son las sumas."))
        celdas.append(code('#| tbl-cap: "PV, EV y AC actividad a actividad en el punto de control"\n#| label: tbl-valorGanadoPaso\n'
                           "calendario_plan = proyecto.calendar(datos_proyecto['duration'])\n"
                           "detalle_vg = detalle_valor_ganado(datos_valor_ganado, calendario_plan, semana_control)\n"
                           "detalle_vg[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
        celdas.append(code("explica_vg(detalle_vg, PV, EV, AC)"))
        celdas.append(md("En las curvas S (valores acumulados semana a semana) la separación vertical entre EV y PV en la "
                         "semana de control es la SV, y la separación entre EV y AC es la CV."))
        celdas.append(code('#| fig-cap: "Curvas S del proyecto"\n#| label: fig-curvasS\n'
                           "plot_curvas_valor_ganado(gantts)"))
    if variante == 1:
        celdas += apartado(ap); ap += 1
        celdas.append(md("Las tres previsiones habituales del coste final (EAC) suman a lo ya gastado (AC) lo que costará "
                         "el trabajo pendiente (ETC), valorado a precio de presupuesto (BAC − EV) y corregido con un índice:\n\n"
                         "- **(a)** $EAC = AC + (BAC - EV)$: el trabajo restante costará lo presupuestado.\n"
                         "- **(b)** $EAC = AC + (BAC - EV)/CPI = BAC / CPI$: se mantiene la eficiencia en coste.\n"
                         "- **(c)** $EAC = AC + (BAC - EV)/(CPI \\cdot SPI)$: el rendimiento en plazo también influye en "
                         "el coste de lo que queda.\n\n"
                         "En todas, $VAC = BAC - EAC$ (negativa: sobrecoste previsto al terminar)."))
        celdas.append(code("EAC_a = AC + (BAC_total - EV)\n"
                           "EAC_b = BAC_total / CPI\n"
                           "EAC_c = AC + (BAC_total - EV) / (CPI * SPI)\n"
                           "previsiones = pd.DataFrame({\n"
                           "    '(a) resto según presupuesto': [EAC_a],\n"
                           "    '(b) misma eficiencia en coste': [EAC_b],\n"
                           "    '(c) misma eficiencia en coste y plazo': [EAC_c]},\n"
                           "    index=['EAC']).T\n"
                           "previsiones.insert(0, 'ETC', previsiones['EAC'] - AC)\n"
                           "previsiones['VAC'] = BAC_total - previsiones['EAC']\n"
                           "previsiones.round(1)"))
        celdas.append(code("explica_eac(BAC_total, PV, EV, AC)"))
        celdas += apartado(ap); ap += 1
        celdas.append(md("$TCPI = \\dfrac{BAC - EV}{BAC - AC}$ (*to-complete performance index*, índice de rendimiento "
                         "para completar): trabajo que queda dividido por el dinero que queda, es decir, la eficiencia en "
                         "coste que habría que mantener en lo que falta para terminar exactamente en presupuesto. Se "
                         "compara con el CPI conseguido hasta ahora."))
        celdas.append(code("TCPI = (BAC_total - EV) / (BAC_total - AC)\n"
                           "explica_tcpi(BAC_total, EV, AC)"))
    elif variante == 2:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La programación ganada (ES, *earned schedule*) busca en qué instante de la planificación se "
                         "había previsto alcanzar el valor que realmente se ha ganado:\n\n"
                         "$$ES = C + \\frac{EV - PV_C}{PV_{C+1} - PV_C}$$\n\n"
                         "donde $C$ es la última semana completa con $PV_C \\le EV$ (PV acumulado). Entonces el retraso "
                         "en tiempo es $AT - ES$ y $SPI(t) = ES / AT$, siendo $AT$ (*actual time*) el tiempo real "
                         "transcurrido, aquí la semana de control."))
        celdas.append(code('#| tbl-cap: "Valor planificado acumulado al final de cada semana (de la fila Total del Gantt del PV)"\n#| label: tbl-pvAcumulado\n'
                           "acumulados[['PV']].T.round(1)"))
        celdas.append(code("pv_acumulado = pd.concat([pd.Series({0: 0.0}), acumulados['PV'].astype(float)])\n"
                           "AT = semana_control\n"
                           "ES, texto_es = explica_es(pv_acumulado, EV, AT)\n"
                           "SPI_t = ES / AT\n"
                           "Markdown(texto_es + f'''\n\n"
                           "- Retraso = AT − ES = {AT} − {ES:.2f} = **{AT - ES:.2f}** semanas\n"
                           "- SPI(t) = ES / AT = {ES:.2f} / {AT} = **{SPI_t:.3f}**\n\n"
                           "**Respuesta:** ES = {ES:.2f} semanas; el proyecto lleva {abs(AT - ES):.2f} semanas de {\"retraso\" if ES < AT else \"adelanto\"} "
                           "y SPI(t) = {SPI_t:.3f} ({\"< 1: retrasado\" if SPI_t < 1 else \"≥ 1: en plazo o adelantado\"}).''')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("PD = duracion_planificada\n"
                           "IEAC_spi = PD / SPI\n"
                           "IEAC_t = PD / SPI_t\n"
                           "pd.DataFrame({'con SPI (€)': [SPI, IEAC_spi], 'con SPI(t) (tiempo)': [SPI_t, IEAC_t]},\n"
                           "             index=['índice', 'duración prevista (semanas)']).round(2)"))
        celdas.append(code("Markdown(f'''Con la duración planificada PD = {PD} semanas (*planned duration*), la previsión es la duración planificada dividida por el índice de plazo:\n\n"
                           "- con SPI: PD / SPI = {PD} / {SPI:.3f} = **{IEAC_spi:.1f}** semanas ({IEAC_spi - PD:+.1f} respecto al plan)\n"
                           "- con SPI(t): PD / SPI(t) = {PD} / {SPI_t:.3f} = **{IEAC_t:.1f}** semanas ({IEAC_t - PD:+.1f} respecto al plan)\n\n"
                           "Difieren porque miden cosas distintas: SPI = {EV:.1f} / {PV:.1f} compara euros ganados con euros planificados y "
                           "depende de qué actividades (caras o baratas) estén retrasadas; SPI(t) = {ES:.2f} / {AT} compara semanas: el avance "
                           "conseguido en {AT} semanas es el que el plan preveía para la semana {ES:.2f}. "
                           "**Respuesta:** la previsión con SPI(t), {IEAC_t:.1f} semanas, es la más fiable para el plazo, sobre todo en la segunda "
                           "mitad del proyecto, cuando el SPI en euros tiende a 1 aunque el retraso persista (apartado siguiente).''')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("Markdown(f'''Al terminar se ha ganado todo el presupuesto y todo estaba planificado: EV = PV = BAC = {BAC_total:.0f} k€, así que "
                           "el SPI clásico vale {BAC_total:.0f} / {BAC_total:.0f} = 1 aunque el proyecto haya terminado tarde: pierde su capacidad de aviso "
                           "conforme avanza el proyecto. En cambio, al final ES = PD = {PD} semanas y AT = duración real, de modo que "
                           "SPI(t) = {PD} / duración real, menor que 1 si hubo retraso. Por ejemplo, si el proyecto terminara en las {IEAC_t:.1f} semanas "
                           "previstas en el apartado anterior, SPI(t) = {PD} / {IEAC_t:.1f} = {PD / IEAC_t:.3f}. "
                           "**Respuesta:** el SPI clásico acaba valiendo 1; el SPI(t) acaba valiendo PD / duración real (< 1 si se termina tarde).''')"))
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
                           "indicadores.index.name = 'semana'"))
        celdas.append(md("De dónde sale cada cifra en cada punto de control: el valor planificado de cada actividad es su "
                         "BAC por la fracción de su duración que debería haberse ejecutado (según el calendario de inicios "
                         "más tempranos); el valor ganado, su BAC por el porcentaje realizado; el coste real, lo gastado en ella. "
                         "Los del proyecto son las sumas."))
        celdas.append(code("calendario_plan = proyecto.calendar(datos_proyecto['duration'])\n"
                           "for semana, seguimiento in seguimientos.items():\n"
                           "    display(Markdown(f'**Punto de control: final de la semana {semana}**'))\n"
                           "    det = detalle_valor_ganado(pd.concat([datos_proyecto, seguimiento], axis=1), calendario_plan, semana)\n"
                           "    display(det[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']])\n"
                           "    display(Markdown('\\n'.join(['- ' + suma_vg(det, c, n) for c, n in (('PV_i', 'PV'), ('EV_i', 'EV'), ('AC', 'AC'))])))"))
        celdas.append(code('#| tbl-cap: "PV, EV y AC del proyecto en cada punto de control (k€)"\n#| label: tbl-pvevac\n'
                           "indicadores.round(1)"))
        celdas.append(code("Markdown('**Respuesta:** ' + '; '.join(f'semana {s}: PV = {f.PV:.1f}, EV = {f.EV:.1f} y AC = {f.AC:.1f} k€' for s, f in indicadores.iterrows()) + '.')"))
        celdas.append(md("Los tres diagramas de Gantt del valor ganado en el último punto de control:"))
        celdas.append(code("datos_vg_ultimo = pd.concat([datos_proyecto, seguimientos[max(seguimientos)]], axis=1)\n"
                           "gantts_ultimo = EarnedValue(proyecto.copy()).calcula_gantts(datos_vg_ultimo, planned_durations_label='duration',\n"
                           "                                                             actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',\n"
                           "                                                             percentage_complete_label='performed', tikz=True,\n"
                           "                                                             " + params + ")"))
        celdas.append(code('#| fig-cap: "PV: valor planificado (último punto de control)"\n#| label: fig-ganttPV\npinta_tikz(gantts_ultimo[\'gantt_PV\'])'))
        celdas.append(code('#| fig-cap: "EV: valor ganado (último punto de control)"\n#| label: fig-ganttEV\npinta_tikz(gantts_ultimo[\'gantt_EV\'])'))
        celdas.append(code('#| fig-cap: "AC: coste real (último punto de control)"\n#| label: fig-ganttAC\npinta_tikz(gantts_ultimo[\'gantt_AC\'])'))
        celdas.append(code('#| fig-cap: "Curvas S hasta el último punto de control"\n#| label: fig-curvasS\n'
                           "plot_curvas_valor_ganado(gantts_ultimo)"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("indicadores['CV'] = indicadores['EV'] - indicadores['AC']\n"
                           "indicadores['SV'] = indicadores['EV'] - indicadores['PV']\n"
                           "indicadores['CPI'] = indicadores['EV'] / indicadores['AC']\n"
                           "indicadores['SPI'] = indicadores['EV'] / indicadores['PV']\n"
                           "for semana, f in indicadores.iterrows():\n"
                           "    display(Markdown(f'**Semana {semana}**'))\n"
                           "    display(explica_vg(None, f.PV, f.EV, f.AC, respuesta=False))"))
        celdas.append(code('#| tbl-cap: "Variaciones (k€) e índices en cada punto de control"\n#| label: tbl-indicadores\n'
                           "indicadores.round(3)"))
        celdas.append(code('#| fig-cap: "Evolución de CPI y SPI"\n#| label: fig-evolucion\n'
                           "ax = indicadores[['CPI', 'SPI']].plot(marker='o', figsize=(7, 4))\n"
                           "ax.axhline(1, color='grey', linestyle='--'); ax.set_ylabel('índice');"))
        celdas.append(code("serie = lambda x: ' → '.join(f'{v:.3f}' for v in x)\n"
                           "def bajo_uno(x, malo):\n"
                           "    debajo = [str(s) for s, v in x.items() if v < 1]\n"
                           "    if len(debajo) == len(x):\n"
                           "        return f'por debajo de 1 en los tres controles ({malo} en todos)'\n"
                           "    if not debajo:\n"
                           "        return 'igual o por encima de 1 en los tres controles'\n"
                           "    return f'por debajo de 1 ({malo}) solo en la semana {lista_y(debajo)}'\n"
                           "Markdown(f'''**Respuesta:** CPI {serie(indicadores['CPI'])}, {bajo_uno(indicadores['CPI'], 'sobrecoste')}; "
                           "SPI {serie(indicadores['SPI'])}, {bajo_uno(indicadores['SPI'], 'retraso')}. La gráfica muestra esa evolución frente a la referencia 1.''')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("indicadores['EAC'] = BAC_total / indicadores['CPI']\n"
                           "indicadores['VAC'] = BAC_total - indicadores['EAC']\n"
                           "display(Markdown('\\n'.join(f'- Semana {s}: EAC = BAC / CPI = {BAC_total:.0f} / {f.CPI:.3f} = **{f.EAC:.1f}** k€; '\n"
                           "                            f'VAC = BAC − EAC = {BAC_total:.0f} − {f.EAC:.1f} = {f.VAC:.1f} k€' for s, f in indicadores.iterrows())))\n"
                           "indicadores[['CPI', 'EAC', 'VAC']].round({'CPI': 3, 'EAC': 1, 'VAC': 1})"))
        celdas.append(code("eac = indicadores['EAC']\n"
                           "pasos = ['sube' if b > a + 0.05 else ('baja' if b < a - 0.05 else 'se mantiene') for a, b in zip(eac.iloc[:-1], eac.iloc[1:])]\n"
                           "if all(p == 'sube' for p in pasos):\n"
                           "    tendencia = 'la previsión empeora en cada control: el CPI baja y cada vez se prevé un coste final mayor'\n"
                           "elif all(p == 'baja' for p in pasos):\n"
                           "    tendencia = 'la previsión mejora en cada control: el CPI sube y cada vez se prevé un coste final menor'\n"
                           "else:\n"
                           "    tendencia = 'la previsión no evoluciona en un solo sentido (' + ' y luego '.join(pasos) + ')'\n"
                           "Markdown(f'''**Respuesta:** EAC {\" → \".join(f\"{v:.1f}\" for v in eac)} k€ frente a un BAC de {BAC_total:.0f} k€: {tendencia}. "
                           "En el último control se prevé terminar con {abs(indicadores[\"VAC\"].iloc[-1]):.1f} k€ {\"más\" if indicadores[\"VAC\"].iloc[-1] < 0 else \"menos\"} de lo presupuestado.''')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("cpi = indicadores['CPI']; spi = indicadores['SPI']; sem = list(indicadores.index)\n"
                           "def evolucion(x, nombre):\n"
                           "    tramos = [f'{\"mejora\" if b > a else \"empeora\"} de {a:.3f} a {b:.3f} entre las semanas {s0} y {s1}' for (s0, a), (s1, b) in zip(list(x.items())[:-1], list(x.items())[1:])]\n"
                           "    return f'el {nombre} ' + ' y ' .join(tramos)\n"
                           "ultimo = seguimientos[max(seguimientos)]\n"
                           "pendientes = [a for a in ordena_ruta(list(ultimo.index), calendario_plan) if ultimo.loc[a, 'performed'] < 100]\n"
                           "criticas_pend = [a for a in pendientes if calendario_plan.loc[a, 'H_total'] == 0]\n"
                           "coste_mal, plazo_mal = cpi.iloc[-1] < 1, spi.iloc[-1] < 1\n"
                           "sentido_c = 'mejora' if cpi.iloc[-1] > cpi.iloc[-2] else 'empeora'\n"
                           "sentido_s = 'mejora' if spi.iloc[-1] > spi.iloc[-2] else 'empeora'\n"
                           "estado = ('se está deteriorando en coste y en plazo' if sentido_c == sentido_s == 'empeora' else\n"
                           "          'se está recuperando en coste y en plazo' if sentido_c == sentido_s == 'mejora' else\n"
                           "          f'no se recupera del todo: el coste {sentido_c} y el plazo {sentido_s}')\n"
                           "aviso_spi = (' La mejora del SPI en el último tramo puede ser en parte aparente: el SPI tiende a 1 al acercarse el final aunque el retraso persista; '\n"
                           "             'para el plazo es preferible la programación ganada.' if spi.iloc[-1] > spi.iloc[-2] else '')\n"
                           "decisiones = []\n"
                           "if plazo_mal:\n"
                           "    decisiones.append(f'recuperar plazo actuando sobre las actividades críticas pendientes ({lista_y(criticas_pend)}), acelerándolas si el coste de hacerlo '\n"
                           "                      'compensa la penalización o los costes indirectos que se evitan' if criticas_pend else\n"
                           "                      'recuperar plazo en las actividades pendientes que hayan consumido su holgura')\n"
                           "if coste_mal:\n"
                           "    decisiones.append(f'analizar las causas del sobrecoste (CPI = {cpi.iloc[-1]:.3f}) para que no se repita en las actividades pendientes '\n"
                           "                      f'({lista_y(pendientes)}) y revisar el EAC con la dirección ({indicadores[\"EAC\"].iloc[-1]:.1f} k€ frente a {BAC_total:.0f})')\n"
                           "Markdown(f'''Evolución: {evolucion(cpi, \"CPI\")}; {evolucion(spi, \"SPI\")}.{aviso_spi}\n\n"
                           "**Respuesta:** comparando los dos últimos controles el proyecto {estado} (CPI {cpi.iloc[-2]:.3f} → {cpi.iloc[-1]:.3f}, SPI {spi.iloc[-2]:.3f} → {spi.iloc[-1]:.3f}); "
                           "en el último control sigue {\"con sobrecoste\" if coste_mal else \"sin sobrecoste\"} y {\"con retraso\" if plazo_mal else \"sin retraso\"}. "
                           "Decisiones tras el último control: {\"; \".join(decisiones) if decisiones else \"mantener el seguimiento\"}.''')"))
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
    celdas.append(md(_SIGLAS))
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
    celdas.append(code(_AUX_VG))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario planificado (inicios más tempranos): de él sale la fracción planificada de cada actividad en la semana de control"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'duracion']]"))
    celdas.append(md("En el Gantt del valor planificado (PV) cada casilla lleva el coste planificado de la semana (BAC / duración); "
                     "en el del valor ganado (EV), el BAC por el porcentaje realizado repartido entre las semanas trabajadas; en el "
                     "del coste real (AC), el coste incurrido repartido entre las semanas trabajadas."))
    celdas.append(code('#| fig-cap: "PV: valor planificado"\n#| label: fig-ganttPV\npinta_tikz(gantts[\'gantt_PV\'])'))
    celdas.append(code('#| fig-cap: "EV: valor ganado"\n#| label: fig-ganttEV\npinta_tikz(gantts[\'gantt_EV\'])'))
    celdas.append(code('#| fig-cap: "AC: coste real"\n#| label: fig-ganttAC\npinta_tikz(gantts[\'gantt_AC\'])'))
    celdas.append(md("PV de cada actividad = BAC por la fracción planificada transcurrida en la semana de control; EV = BAC por "
                     "el porcentaje realizado; AC = coste real anotado. Los del proyecto son las sumas."))
    celdas.append(code('#| tbl-cap: "PV, EV y AC actividad a actividad en el punto de control"\n#| label: tbl-valorGanadoPaso\n'
                       "detalle_vg = detalle_valor_ganado(datos_valor_ganado, calendario, semana_control)\n"
                       "detalle_vg[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(code("acumulados = gantts['acumulados']\n"
                       "PV = float(acumulados.loc[semana_control, 'PV'])\n"
                       "EV = float((datos_valor_ganado['BAC'] * datos_valor_ganado['performed'] / 100).sum())\n"
                       "AC = float(datos_valor_ganado['AC'].sum())\n"
                       "CPI, SPI = EV / AC, EV / PV\n"
                       "explica_vg(detalle_vg, PV, EV, AC)"))
    celdas.append(code('#| fig-cap: "Curvas S"\n#| label: fig-curvasS\nplot_curvas_valor_ganado(gantts)'))
    celdas += apartado(2)
    if modo == "coste-bien-plazo-mal":
        celdas.append(code("Markdown(f'''Gastar menos de lo previsto (AC = {AC:.1f} < PV = {PV:.1f} k€) no dice nada por sí solo: se ha gastado menos porque se ha hecho menos. "
                           "La comparación correcta es entre lo gastado y lo *ganado*: CPI = EV / AC = {EV:.1f} / {AC:.1f} = {CPI:.3f} {\">\" if CPI > 1 else \"<\"} 1, así que el coste sí va bien "
                           "(CV = {EV - AC:.1f} k€); pero SPI = EV / PV = {EV:.1f} / {PV:.1f} = {SPI:.3f} < 1 (SV = {EV - PV:.1f} k€): se ha hecho solo el {EV/PV*100:.1f} % del trabajo previsto para esta fecha. "
                           "La frase del director confunde la diferencia de gasto (AC − PV = {AC - PV:.1f} k€) con la variación de coste (CV = EV − AC). "
                           "**Respuesta:** la lectura no es correcta; oculta que el proyecto va **retrasado** ({abs(EV - PV):.1f} k€ de trabajo por detrás del plan), aunque en coste vaya bien.''')"))
    else:
        celdas.append(code("Markdown(f'''El adelanto es real: SPI = EV / PV = {EV:.1f} / {PV:.1f} = {SPI:.3f} ≥ 1 (SV = {EV - PV:.1f} k€). Pero ese trabajo ha costado "
                           "AC = {AC:.1f} k€ cuando su presupuesto era EV = {EV:.1f}: CPI = {EV:.1f} / {AC:.1f} = {CPI:.3f} < 1, cada k€ gastado ha producido solo {CPI:.3f} k€ de valor "
                           "(CV = {EV - AC:.1f} k€). **Respuesta:** la lectura es correcta en plazo, pero oculta el **sobrecoste** de {AC - EV:.1f} k€, que es lo que "
                           "determinará si el proyecto acaba en presupuesto.''')"))
    celdas += apartado(3)
    celdas.append(md("Variación de cada actividad: SV = EV − PV y CV = EV − AC (en k€); la holgura total del calendario planificado "
                     "indica si un retraso afecta al plazo final. Las variaciones del proyecto son la suma de las de sus actividades."))
    celdas.append(code("por_actividad = pd.DataFrame({'BAC': datos_valor_ganado['BAC'], 'realizado (%)': datos_valor_ganado['performed'],\n"
                       "                              'PV': detalle_vg['PV_i'].astype(float), 'EV': detalle_vg['EV_i'].astype(float),\n"
                       "                              'AC': datos_valor_ganado['AC'].astype(float), 'H_total': calendario['H_total']})\n"
                       "por_actividad['SV = EV − PV'] = por_actividad['EV'] - por_actividad['PV']\n"
                       "por_actividad['CV = EV − AC'] = por_actividad['EV'] - por_actividad['AC']\n"
                       "por_actividad.round(1)"))
    if modo == "coste-bien-plazo-mal":
        celdas.append(code("sv = por_actividad['SV = EV − PV']\n"
                           "retrasadas = sv[sv < -0.05].sort_values()\n"
                           "SV_total = sv.sum()\n"
                           "lineas = [f'- {a}: SV = {por_actividad.loc[a, \"EV\"]:.1f} − {por_actividad.loc[a, \"PV\"]:.1f} = {v:.1f} k€ ({100 * v / SV_total:.0f} % de la SV total; '\n"
                           "          f'{\"crítica\" if por_actividad.loc[a, \"H_total\"] == 0 else \"holgura total \" + str(int(por_actividad.loc[a, \"H_total\"]))})' for a, v in retrasadas.items()]\n"
                           "criticas = [a for a in retrasadas.index if por_actividad.loc[a, 'H_total'] == 0]\n"
                           "Markdown('Actividades con SV negativa, de mayor a menor peso (SV total = ' + f'{SV_total:.1f} k€):\\n\\n' + '\\n'.join(lineas) + '\\n\\n'\n"
                           "         + f'**Respuesta:** el retraso lo explican {lista_y(list(retrasadas.index))}; las de mayor peso son {lista_y(list(retrasadas.index[:2]))}. '\n"
                           "         + (f'De ellas son críticas {lista_y(criticas)}: su retraso se traslada al plazo final si no se recupera.' if criticas else\n"
                           "            'Ninguna es crítica: mientras su retraso no supere su holgura, el plazo final no se mueve.'))"))
    else:
        celdas.append(code("cv = por_actividad['CV = EV − AC']\n"
                           "caras = cv[cv < -0.05].sort_values()\n"
                           "CV_total = cv.sum()\n"
                           "lineas = [f'- {a}: CV = {por_actividad.loc[a, \"EV\"]:.1f} − {por_actividad.loc[a, \"AC\"]:.1f} = {v:.1f} k€ ({100 * v / CV_total:.0f} % de la CV total)' for a, v in caras.items()]\n"
                           "ahorro = cv[cv > 0.05]\n"
                           "Markdown('Actividades con CV negativa, de mayor a menor peso (CV total = ' + f'{CV_total:.1f} k€):\\n\\n' + '\\n'.join(lineas) + '\\n\\n'\n"
                           "         + (f'Compensan en parte {lista_y(list(ahorro.index))} (CV positiva). ' if len(ahorro) else '')\n"
                           "         + f'**Respuesta:** el sobrecoste lo explican {lista_y(list(caras.index))}; las de mayor peso son {lista_y(list(caras.index[:2]))} '\n"
                           "         + f'({100 * caras.iloc[:2].sum() / CV_total:.0f} % del total entre las dos).')"))
    celdas += apartado(4)
    if modo == "coste-bien-plazo-mal":
        celdas.append(code("EAC = BAC_total / CPI\n"
                           "IEAC = duracion_planificada / SPI\n"
                           "Markdown(f'''- Coste final: EAC = BAC / CPI = {BAC_total:.0f} / {CPI:.3f} = **{EAC:.1f}** k€ (VAC = {BAC_total:.0f} − {EAC:.1f} = {BAC_total - EAC:.1f} k€: ahorro previsto).\n"
                           "- Duración final: duración planificada / SPI = {duracion_planificada} / {SPI:.3f} = **{IEAC:.1f}** semanas ({IEAC - duracion_planificada:.1f} más que las {duracion_planificada} planificadas).\n\n"
                           "**Respuesta:** la previsión de coste es la más fiable (el CPI suele ser estable una vez avanzado el proyecto); la de plazo con SPI, la menos: el SPI en euros "
                           "tiende a 1 al final aunque el retraso persista, y el retraso real depende de si las actividades retrasadas son críticas, cosa que el SPI no distingue. "
                           "Para el plazo conviene la programación ganada o replanificar lo que falta.''')"))
    else:
        celdas.append(code("EAC = BAC_total / CPI\n"
                           "VAC = BAC_total - EAC\n"
                           "IEAC = duracion_planificada / SPI\n"
                           "adelanto = duracion_planificada - IEAC\n"
                           "if adelanto > 0.05:\n"
                           "    corte = (f'Compensa si ese valor semanal multiplicado por las semanas ganadas supera el sobrecoste: '\n"
                           "             f'{-VAC:.1f} k€ / {adelanto:.1f} semanas = {-VAC / adelanto:.1f} k€ por semana.')\n"
                           "    resp = f'el adelanto solo compensa si cada semana ganada vale más de unos {-VAC / adelanto:.1f} k€'\n"
                           "else:\n"
                           "    corte = 'Con SPI = 1 no se prevé ganar tiempo, así que el sobrecoste no compra nada.'\n"
                           "    resp = 'no se prevé terminar antes, así que el sobrecoste no se compensa'\n"
                           "Markdown(f'''- EAC = BAC / CPI = {BAC_total:.0f} / {CPI:.3f} = **{EAC:.1f}** k€\n"
                           "- VAC = BAC − EAC = {BAC_total:.0f} − {EAC:.1f} = **{VAC:.1f}** k€ (sobrecoste previsto)\n"
                           "- Si se mantiene el ritmo, duración ≈ duración planificada / SPI = {duracion_planificada} / {SPI:.3f} = {IEAC:.1f} semanas, unas {adelanto:.1f} semanas de adelanto.\n\n"
                           "Que el adelanto compense depende de lo que valga el tiempo en el contrato: una prima por entrega anticipada o los costes indirectos "
                           "y penalizaciones que se evitan por semana. {corte} "
                           "**Respuesta:** EAC = {EAC:.1f} k€ y VAC = {VAC:.1f} k€; {resp} "
                           "(dato del contrato: prima por semana de adelanto o penalización por semana de retraso); si no, se está comprando tiempo demasiado caro.''')"))
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
    celdas.append(md(_SIGLAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "BAC_total = float(datos_proyecto['BAC'].sum())\n"
                       f"semana_control = {t}\n"
                       "datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                       "gantts = EarnedValue(proyecto).calcula_gantts(datos_valor_ganado, planned_durations_label='duration',\n"
                       "                                             actual_durations_label='actual_duration', PV_label='BAC', AC_label='AC',\n"
                       "                                             percentage_complete_label='performed', tikz=False)"))
    celdas.append(code(_AUX_VG))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario planificado (inicios más tempranos): de él sale la fracción planificada de cada actividad en la semana de control"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'duracion']]"))
    celdas.append(md("PV de cada actividad = BAC por la fracción planificada transcurrida en la semana de control; "
                     "EV = BAC por el porcentaje realizado; AC = coste real anotado. Los totales son las sumas."))
    celdas.append(code('#| tbl-cap: "PV, EV y AC actividad a actividad"\n#| label: tbl-valorGanadoPaso\n'
                       "detalle_vg = detalle_valor_ganado(datos_valor_ganado, calendario, semana_control)\n"
                       "detalle_vg[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(code("PV = float(gantts['acumulados'].loc[semana_control, 'PV'])\n"
                       "EV = float((datos_valor_ganado['BAC'] * datos_valor_ganado['performed'] / 100).sum())\n"
                       "AC = float(datos_valor_ganado['AC'].sum())\n"
                       "Markdown('\\n'.join(['- ' + suma_vg(detalle_vg, c, n) for c, n in (('PV_i', 'PV'), ('EV_i', 'EV'), ('AC', 'AC'))])\n"
                       "         + f'\\n\\n**Respuesta:** PV = {PV:.1f}, EV = {EV:.1f} y AC = {AC:.1f} k€; redondeados, {PV:.0f}, {EV:.0f} y {AC:.0f}: '\n"
                       "         'los tres valores del alumno son correctos. Los errores están en lo que hace con ellos.')"))
    celdas += apartado(2)
    celdas.append(code("CV, SV = EV - AC, EV - PV\n"
                       "CPI, SPI = EV / AC, EV / PV\n"
                       "EAC = BAC_total / CPI\n"
                       "cpi_alumno, spi_alumno = EV / PV, EV / AC\n"
                       "Markdown(f'''1. **CV = AC − EV = {AC - EV:.0f} «positiva: ahorro»**: la variación de coste es CV = EV − AC = {EV:.1f} − {AC:.1f} = {CV:.1f} k€. "
                       "Con el signo cambiado, un sobrecoste de {-CV:.1f} k€ parece un ahorro.\n"
                       "2. **CPI = EV / PV = {cpi_alumno:.2f}**: eso es el SPI. El CPI compara el valor ganado con el coste real: CPI = EV / AC = {EV:.1f} / {AC:.1f} = {CPI:.3f}.\n"
                       "3. **SPI = EV / AC = {spi_alumno:.2f}**: eso es el CPI. SPI = EV / PV = {EV:.1f} / {PV:.1f} = {SPI:.3f}.\n"
                       "4. **Interpretación cruzada** («CPI < 1 → retraso; SPI < 1 → sobrecoste»): el CPI mide coste y el SPI mide plazo. Aquí los dos índices "
                       "correctos son menores que 1 y las conclusiones «retraso» y «sobrecoste» resultan ciertas, pero solo porque dos errores se anulan: el número que el alumno "
                       "llama SPI ({spi_alumno:.2f}) es en realidad el CPI y lo lee como coste; el que llama CPI ({cpi_alumno:.2f}) es el SPI y lo lee como plazo. Quien lea "
                       "su informe con las definiciones correctas sacará conclusiones falsas: creerá que el rendimiento en coste es {cpi_alumno:.2f} cuando es {CPI:.3f}, "
                       "y que el de plazo es {spi_alumno:.2f} cuando es {SPI:.3f}.\n"
                       "5. **EAC = BAC × CPI = {BAC_total * cpi_alumno:.0f} k€**: la previsión es EAC = BAC / CPI. Multiplicar por un índice menor que 1 da un coste final "
                       "menor que el presupuesto ({BAC_total:.0f} k€) justo cuando el proyecto va con sobrecoste; además usa el índice equivocado. Bien hecho: "
                       "EAC = {BAC_total:.0f} / {CPI:.3f} = {EAC:.1f} k€.\n\n"
                       "**Respuesta:** hay cinco errores: el signo de CV, las fórmulas de CPI y de SPI (intercambiadas), la interpretación de los índices (cruzada) y la fórmula del EAC.''')"))
    celdas += apartado(3)
    celdas.append(code("VAC = BAC_total - EAC\n"
                       "pd.Series({'CV = EV − AC': CV, 'SV = EV − PV': SV, 'CPI = EV / AC': CPI, 'SPI = EV / PV': SPI, 'EAC = BAC / CPI': EAC, 'VAC = BAC − EAC': VAC}).round(3).to_frame('valor').T"))
    celdas.append(code("coste, plazo = lectura_vg(PV, EV, AC)\n"
                       "display(explica_vg(None, PV, EV, AC, respuesta=False))\n"
                       "Markdown(f'''- EAC = BAC / CPI = {BAC_total:.0f} / {CPI:.3f} = **{EAC:.1f}** k€\n"
                       "- VAC = BAC − EAC = {BAC_total:.0f} − {EAC:.1f} = **{VAC:.1f}** k€\n\n"
                       "**Respuesta:** en coste, {coste}; en plazo, {plazo}. Si se mantiene la eficiencia en coste, el proyecto costará {EAC:.1f} k€ frente a los "
                       "{BAC_total:.0f} presupuestados: {abs(VAC):.1f} k€ {\"de sobrecoste\" if VAC < 0 else \"de ahorro\"} al terminar.''')"))
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
    celdas.append(md(_SIGLAS))
    celdas.append(code(f"BAC_total, duracion_planificada = {BAC_total:.0f}, {T}\n"
                       "curvas = datos_enunciado.copy().astype(float)"))
    celdas.append(code(_AUX_VG))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Curvas S del proyecto"\n#| label: fig-curvasS\n'
                       "from matplotlib import pyplot as plt\n"
                       "ax = curvas.plot(marker='o', figsize=(8, 4)); ax.set_xlabel('semana'); ax.set_ylabel('k€ acumulados'); ax.grid(alpha=0.3); plt.show()"))
    celdas.append(md("Cada fila de la tabla siguiente aplica las cuatro fórmulas a los valores acumulados de esa semana; debajo, "
                     "el cálculo completo de la última semana como ejemplo."))
    celdas.append(code('#| tbl-cap: "Indicadores semana a semana"\n#| label: tbl-indicadores\n'
                       "ind = curvas.copy()\n"
                       "ind['CV = EV − AC'] = ind['EV'] - ind['AC']\n"
                       "ind['SV = EV − PV'] = ind['EV'] - ind['PV']\n"
                       "ind['CPI = EV / AC'] = ind['EV'] / ind['AC']\n"
                       "ind['SPI = EV / PV'] = ind['EV'] / ind['PV']\n"
                       "ind.round(3)"))
    celdas.append(code("t_ult = int(curvas.index[-1])\n"
                       "display(Markdown(f'**Semana {t_ult}**'))\n"
                       "explica_vg(None, curvas.loc[t_ult, 'PV'], curvas.loc[t_ult, 'EV'], curvas.loc[t_ult, 'AC'])"))
    celdas.append(code('#| fig-cap: "Evolución de CPI y SPI"\n#| label: fig-indices\n'
                       "ax = ind[['CPI = EV / AC', 'SPI = EV / PV']].plot(marker='o', figsize=(8, 4)); ax.axhline(1, color='grey', linestyle='--'); ax.set_xlabel('semana'); plt.show()"))
    celdas += apartado(2)
    celdas.append(code("umbral = 0.97\n"
                       "def inicio(serie, nombre, problema):\n"
                       "    s = next((int(t) for t, v in serie.items() if v < umbral), None)\n"
                       "    if s is None:\n"
                       "        return None, f'el {nombre} no baja de {umbral} en ninguna semana: no hay {problema} apreciable'\n"
                       "    previo = f' (en la semana {s - 1} era {serie[s - 1]:.3f})' if s - 1 in serie.index else ' (desde la primera semana)'\n"
                       "    vuelve = [int(x) for x, v in serie.items() if x > s and v >= umbral]\n"
                       "    nota = (f'; se recupera por encima de {umbral} en la semana {vuelve[0]}, pero termina en {serie.iloc[-1]:.3f}' if vuelve else\n"
                       "            f' y ya no vuelve a superar {umbral} (termina en {serie.iloc[-1]:.3f})')\n"
                       "    return s, f'el {nombre} baja de {umbral} en la semana {s}: {serie[s]:.3f}{previo}{nota}'\n"
                       "semana_retraso, txt_r = inicio(ind['SPI = EV / PV'], 'SPI', 'retraso')\n"
                       "semana_sobrecoste, txt_c = inicio(ind['CPI = EV / AC'], 'CPI', 'sobrecoste')\n"
                       "if semana_retraso == semana_sobrecoste:\n"
                       "    simult = 'Fueron simultáneos.'\n"
                       "elif semana_retraso is None or semana_sobrecoste is None:\n"
                       "    simult = 'No fueron simultáneos: solo aparece uno de los dos problemas.'\n"
                       "else:\n"
                       "    primero = 'el sobrecoste' if semana_sobrecoste < semana_retraso else 'el retraso'\n"
                       "    simult = (f'No fueron simultáneos: {primero} apareció {plural(abs(semana_retraso - semana_sobrecoste), \"semana\")} antes. Un problema de plazo '\n"
                       "              'no implica uno de coste ni al revés (se puede ir despacio gastando poco, o gastar mucho para ir rápido).')\n"
                       "Markdown(f'''Se toma como señal un índice por debajo de {umbral} (unas centésimas por debajo de 1 son ruido de redondeo): {txt_r}; {txt_c}.\n\n"
                       "**Respuesta:** retraso desde la semana {semana_retraso}, sobrecoste desde la semana {semana_sobrecoste}. {simult}''')"))
    celdas += apartado(3)
    celdas.append(code("t = int(curvas.index[-1])\n"
                       "PV, EV, AC = curvas.loc[t, 'PV'], curvas.loc[t, 'EV'], curvas.loc[t, 'AC']\n"
                       "CPI, SPI = EV / AC, EV / PV\n"
                       "EAC = BAC_total / CPI\n"
                       "pv_acumulado = pd.concat([pd.Series({0: 0.0}), curvas['PV']])\n"
                       "ES, texto_es = explica_es(pv_acumulado, EV, t)\n"
                       "SPI_t = ES / t\n"
                       "duracion_estimada = duracion_planificada / SPI_t\n"
                       "Markdown(f'''Coste: CPI = EV / AC = {EV:.1f} / {AC:.1f} = {CPI:.3f}; EAC = BAC / CPI = {BAC_total:.0f} / {CPI:.3f} = **{EAC:.1f}** k€ "
                       "(VAC = {BAC_total:.0f} − {EAC:.1f} = {BAC_total - EAC:.1f} k€).\n\n"
                       "Plazo. {texto_es} SPI(t) = ES / {t} = {ES:.2f} / {t} = {SPI_t:.3f}; duración estimada = {duracion_planificada} / {SPI_t:.3f} = **{duracion_estimada:.1f}** semanas.\n\n"
                       "**Respuesta:** coste final previsto {EAC:.1f} k€ ({abs(BAC_total - EAC):.1f} k€ {\"por encima\" if EAC > BAC_total else \"por debajo\"} del presupuesto) "
                       "y duración prevista {duracion_estimada:.1f} semanas ({abs(duracion_estimada - duracion_planificada):.1f} {\"más\" if duracion_estimada > duracion_planificada else \"menos\"} que las {duracion_planificada} planificadas).''')"))
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
    celdas.append(md(_SIGLAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       f"semana_control = {t}\n"
                       "BAC_total = float(datos_proyecto['BAC'].sum())"))
    celdas.append(code(_AUX_VG))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario de inicios más tempranos"\n#| label: tbl-calendario\n'
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'duracion']]"))
    celdas.append(md("Con el inicio más temprano de cada actividad se reparte su BAC semana a semana según su perfil (una "
                     "actividad que empieza en el instante ES ocupa las semanas ES + 1 a ES + duración); la columna *suma* "
                     "comprueba que se recupera el BAC."))
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
    celdas.append(code("num = lambda x: f'{x:.1f}' if abs(10 * x - round(10 * x)) < 1e-9 else f'{x:.2f}'\n"
                       "rango = lambda a, b: f'la semana {a}' if a == b else f'cada una de las semanas {a} a {b}'\n"
                       "lineas = []\n"
                       "for a, f in datos_proyecto.iterrows():\n"
                       "    es, d, B = int(calendario.loc[a, 'inicio_mas_temprano']), int(f['duration']), float(f['BAC'])\n"
                       "    if f['profile'] == 'uniforme' or d == 1:\n"
                       "        continue\n"
                       "    resto = B / 2 / (d - 1)\n"
                       "    otras = rango(es + 2, es + d) if f['profile'] == 'inicio' else rango(es + 1, es + d - 1)\n"
                       "    clave = f'semana {es + 1}' if f['profile'] == 'inicio' else f'semana {es + d}'\n"
                       "    lineas.append(f'- {a} (perfil {f[\"profile\"]}, BAC = {B:.0f}, {d} semanas, semanas {es + 1} a {es + d}): {B:.0f} / 2 = {B / 2:.1f} en la {clave} '\n"
                       "                  f'y ({B:.0f} − {B / 2:.1f}) / {d - 1} = {num(resto)} en {otras}; suma {B / 2:.1f} + {d - 1}·{num(resto)} = {B:.1f}')\n"
                       "ok = bool(np.allclose(coste_semanal['suma'], datos_proyecto['BAC']))\n"
                       "Markdown('Las de perfil uniforme reciben BAC / duración cada semana. Las de perfil no uniforme:\\n\\n' + '\\n'.join(lineas)\n"
                       "         + f'\\n\\n**Respuesta:** el calendario es el de la tabla anterior y el reparto semanal el de la tabla de costes; '\n"
                       "         + ('todas las filas suman su BAC.' if ok else 'hay filas que no suman su BAC: revisar el reparto.'))"))
    celdas += apartado(2)
    celdas.append(md("El PV de cada actividad en la semana de control es la suma de sus costes planificados hasta esa semana "
                     "(columnas 1 a la semana de control de la tabla anterior); con reparto uniforme sería BAC por la fracción "
                     "de duración transcurrida."))
    celdas.append(code("semanas = [c for c in coste_semanal.columns if c != 'suma' and c <= semana_control]\n"
                       "PV_i = coste_semanal[semanas].sum(axis=1)\n"
                       "frac = ((semana_control - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion'])\n"
                       "PV_uniforme_i = datos_proyecto['BAC'] * frac\n"
                       "comparacion = pd.DataFrame({'perfil': datos_proyecto['profile'], 'PV según perfil': PV_i, 'PV uniforme': PV_uniforme_i})\n"
                       "comparacion.loc['Total'] = ['', PV_i.sum(), PV_uniforme_i.sum()]\n"
                       "comparacion.round(1)"))
    celdas.append(code("PV, PV_uniforme = float(PV_i.sum()), float(PV_uniforme_i.sum())\n"
                       "difieren = [a for a in datos_proyecto.index if abs(PV_i[a] - PV_uniforme_i[a]) > 1e-6]\n"
                       "lineas = []\n"
                       "for a in difieren:\n"
                       "    es, d, B = int(calendario.loc[a, 'inicio_mas_temprano']), int(datos_proyecto.loc[a, 'duration']), float(datos_proyecto.loc[a, 'BAC'])\n"
                       "    k = min(max(semana_control - es, 0), d)\n"
                       "    lineas.append(f'- {a} (perfil {datos_proyecto.loc[a, \"profile\"]}, {k} de {d} semanas transcurridas): según perfil '\n"
                       "                  f'{\" + \".join(num(x) for x in coste_semanal.loc[a, semanas] if x > 0)} = {PV_i[a]:.1f}; uniforme {B:.0f}·{k}/{d} = {PV_uniforme_i[a]:.1f}')\n"
                       "Markdown('Solo cambian las actividades en curso con perfil no uniforme (las terminadas suman su BAC y las no empezadas, 0 con los dos criterios):\\n\\n'\n"
                       "         + '\\n'.join(lineas)\n"
                       "         + f'\\n\\n- PV según perfil = {\" + \".join(f\"{x:.1f}\" for x in PV_i if x > 0)} = **{PV:.1f}** k€\\n'\n"
                       "         + f'- PV uniforme = {\" + \".join(f\"{x:.1f}\" for x in PV_uniforme_i if x > 0)} = **{PV_uniforme:.1f}** k€\\n\\n'\n"
                       "         + f'**Respuesta:** PV = {PV:.1f} k€ con los perfiles reales frente a {PV_uniforme:.1f} k€ con reparto uniforme '\n"
                       "         + f'({abs(PV - PV_uniforme):.1f} k€ {\"más\" if PV > PV_uniforme else \"menos\"}), por {lista_y(difieren)}.')"))
    celdas += apartado(3)
    celdas.append(code("EV_i = datos_proyecto['BAC'] * seguimiento['performed'] / 100\n"
                       "EV = float(EV_i.sum())\n"
                       "AC = float(seguimiento['AC'].sum())\n"
                       "indices = pd.DataFrame({'PV según perfil': [PV, EV, AC, EV / AC, EV / PV], 'PV uniforme': [PV_uniforme, EV, AC, EV / AC, EV / PV_uniforme]},\n"
                       "                       index=['PV', 'EV', 'AC', 'CPI = EV / AC', 'SPI = EV / PV']).round(3)\n"
                       "indices"))
    celdas.append(code("s1, s2 = EV / PV, EV / PV_uniforme\n"
                       "lee = lambda s: 'retraso' if s < 1 else ('adelanto' if s > 1 else 'en plazo')\n"
                       "Markdown(f'''- EV = Σ BAC·% = {\" + \".join(f\"{a}({x:.1f})\" for a, x in EV_i.items() if x > 0)} = **{EV:.1f}** k€\n"
                       "- AC = Σ coste real = {\" + \".join(f\"{a}({x:.0f})\" for a, x in seguimiento['AC'].items() if x > 0)} = **{AC:.1f}** k€\n"
                       "- CPI = EV / AC = {EV:.1f} / {AC:.1f} = **{EV / AC:.3f}** (el mismo con los dos PV: no depende del PV)\n"
                       "- SPI con el PV correcto = {EV:.1f} / {PV:.1f} = **{s1:.3f}** ({lee(s1)}); con el PV uniforme = {EV:.1f} / {PV_uniforme:.1f} = **{s2:.3f}** ({lee(s2)})\n\n"
                       "**Respuesta:** {'el diagnóstico de plazo cambia: ' + lee(s2) + ' con el PV uniforme, ' + lee(s1) + ' con el correcto' if lee(s1) != lee(s2) else 'el diagnóstico de plazo (' + lee(s1) + ') no cambia, pero sí su magnitud: SPI ' + format(s1, '.3f') + ' frente a ' + format(s2, '.3f')}; "
                       "el de coste (CPI = {EV / AC:.3f}, {\"sobrecoste\" if EV < AC else \"ahorro\"}) es el mismo.''')"))
    celdas += apartado(4)
    celdas.append(md("El EV es el BAC por el porcentaje *físico* realizado: mide trabajo hecho, no dinero "
                     "desembolsado, y por eso no depende de cuándo se planeó pagar. El PV sí es dinero "
                     "planificado hasta la fecha. Si una actividad en plazo tiene perfil `final` (el gran pago "
                     "al terminar), su PV a mitad de camino es pequeño y el EV (proporcional al avance físico) lo "
                     "supera: el proyecto parece adelantado sin estarlo. Con perfil `inicio` ocurre lo contrario. "
                     "Para que el SPI mida plazo, el PV debe seguir el mismo criterio que el EV."))
    celdas.append(code("ejemplos = [f'{a} (perfil {datos_proyecto.loc[a, \"profile\"]}): PV según perfil {PV_i[a]:.1f} frente a {PV_uniforme_i[a]:.1f} uniforme' for a in difieren]\n"
                       "Markdown(f'En este proyecto se ve en {\"; \".join(ejemplos)}. **Respuesta:** el EV no depende del perfil porque mide avance físico; '\n"
                       "         'el perfil `final` hace parecer adelantado un proyecto que va en plazo (y el `inicio`, retrasado).' if ejemplos else\n"
                       "         '**Respuesta:** el EV no depende del perfil porque mide avance físico; el perfil `final` hace parecer adelantado un proyecto que va en plazo.')"))
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
    celdas.append(md(_SIGLAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       f"semana_control = {t}"))
    celdas.append(code(_AUX_VG))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación con el camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario planificado (inicios más tempranos y holguras)"\n#| label: tbl-calendario\n'
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'duracion']]"))
    celdas.append(md("PV de cada actividad = BAC por la fracción planificada transcurrida en la semana de control; "
                     "EV = BAC por el porcentaje realizado; AC = coste real anotado:"))
    celdas.append(code('#| tbl-cap: "Cómo se obtiene el PV y el EV de cada actividad"\n#| label: tbl-valorGanadoPaso\n'
                       "datos_valor_ganado = pd.concat([datos_proyecto, seguimiento], axis=1)\n"
                       "detalle = detalle_valor_ganado(datos_valor_ganado, calendario, semana_control)\n"
                       "detalle[['BAC', 'planificado hasta la semana', 'PV = BAC·fracción', 'realizado (%)', 'EV = BAC·%', 'AC']]"))
    celdas.append(md("Con ellos, para cada actividad CPI = EV / AC y SPI = EV / PV; quedan sin definir (—) cuando el "
                     "denominador es 0 (actividad sin coste real o sin trabajo planificado todavía). La holgura total sale "
                     "del calendario planificado."))
    celdas.append(code('#| tbl-cap: "Valor ganado por actividad"\n#| label: tbl-porActividad\n'
                       "por_actividad = pd.DataFrame({'PV': detalle['PV_i'].astype(float), 'EV': detalle['EV_i'].astype(float), 'AC': seguimiento['AC'].astype(float),\n"
                       "                              'H_total': calendario['H_total']})\n"
                       "por_actividad['CPI'] = por_actividad['EV'] / por_actividad['AC'].replace(0, np.nan)\n"
                       "por_actividad['SPI'] = por_actividad['EV'] / por_actividad['PV'].replace(0, np.nan)\n"
                       "limpia_tabla(por_actividad.rename(columns={'H_total': 'holgura total'}), decimales=3)"))
    celdas.append(code("p = por_actividad\n"
                       "lineas = []\n"
                       "for a, f in p.iterrows():\n"
                       "    cpi = f'CPI = {f.EV:.1f} / {f.AC:.1f} = {f.CPI:.3f}' if f.AC > 0 else 'CPI sin definir (AC = 0)'\n"
                       "    spi = f'SPI = {f.EV:.1f} / {f.PV:.1f} = {f.SPI:.3f}' if f.PV > 0 else 'SPI sin definir (PV = 0)'\n"
                       "    lineas.append(f'- {a}: {cpi}; {spi}; holgura total {int(f.H_total)}')\n"
                       "Markdown('\\n'.join(lineas))"))
    celdas += apartado(2)
    celdas.append(code("grupos = {\n"
                       "    'críticas con retraso (amenazan el plazo)': p.index[(p['H_total'] == 0) & (p['SPI'] < 1)],\n"
                       "    'no críticas con retraso': p.index[(p['H_total'] > 0) & (p['SPI'] < 1)],\n"
                       "    'sobrecoste sin retraso (amenazan el presupuesto)': p.index[(p['CPI'] < 1) & ~(p['SPI'] < 1)],\n"
                       "}\n"
                       "grupos['resto'] = p.index.difference(pd.Index([]).append([pd.Index(v) for v in grupos.values()]))\n"
                       "pd.Series({k: ', '.join(v) or '—' for k, v in grupos.items()}, name='actividades').to_frame()"))
    celdas.append(md("Una no crítica retrasada solo amenaza el plazo si su retraso supera su holgura total. El retraso de cada "
                     "actividad, en semanas de trabajo, es la diferencia entre la fracción planificada y la realizada por su "
                     "duración: $(PV_i - EV_i)/BAC_i \\cdot d_i$."))
    celdas.append(code("dur = datos_proyecto['duration']; bac = datos_proyecto['BAC']\n"
                       "retraso = ((p['PV'] - p['EV']) / bac * dur)\n"
                       "no_criticas = list(grupos['no críticas con retraso'])\n"
                       "criticas_r = list(grupos['críticas con retraso (amenazan el plazo)'])\n"
                       "lineas = [f'- {a}: ({p.loc[a, \"PV\"]:.1f} − {p.loc[a, \"EV\"]:.1f}) / {bac[a]:.0f} · {dur[a]} = {retraso[a]:.1f} semanas, '\n"
                       "          f'{\"más\" if retraso[a] > p.loc[a, \"H_total\"] else \"no más\"} que su holgura de {plural(int(p.loc[a, \"H_total\"]), \"semana\")}'\n"
                       "          for a in criticas_r + no_criticas]\n"
                       "amenazan = criticas_r + [a for a in no_criticas if retraso[a] > p.loc[a, 'H_total']]\n"
                       "absorben = [a for a in no_criticas if retraso[a] <= p.loc[a, 'H_total']]\n"
                       "solo_coste = list(grupos['sobrecoste sin retraso (amenazan el presupuesto)'])\n"
                       "tambien_coste = [a for a in criticas_r + no_criticas if p.loc[a, 'CPI'] < 1]\n"
                       "exceden = [a for a in no_criticas if retraso[a] > p.loc[a, 'H_total']]\n"
                       "partes = ([f'{lista_y(criticas_r)}, por ser {\"crítica\" if len(criticas_r) == 1 else \"críticas\"} y estar {\"retrasada\" if len(criticas_r) == 1 else \"retrasadas\"}'] if criticas_r else []) \\\n"
                       "         + ([f'{lista_y(exceden)}, cuyo retraso ya supera su holgura'] if exceden else [])\n"
                       "Markdown('\\n'.join(lineas) + '\\n\\n'\n"
                       "         + f'**Respuesta:** amenazan el plazo {\"; \".join(partes) if partes else \"ninguna actividad\"}'\n"
                       "         + (f'; {lista_y(absorben)} se {\"retrasa\" if len(absorben) == 1 else \"retrasan\"} dentro de su holgura y de momento no lo mueve{\"\" if len(absorben) == 1 else \"n\"}' if absorben else '')\n"
                       "         + f'. Amenazan solo el presupuesto {lista_y(solo_coste) if solo_coste else \"ninguna\"} (CPI < 1 sin retraso)'\n"
                       "         + (f'; además, {lista_y(tambien_coste)} {\"tiene\" if len(tambien_coste) == 1 else \"tienen\"} retraso y sobrecoste a la vez' if tambien_coste else '') + '.')"))
    celdas += apartado(3)
    celdas.append(code("PV, EV, AC = p['PV'].sum(), p['EV'].sum(), p['AC'].sum()\n"
                       "explica_vg(detalle, PV, EV, AC)"))
    celdas.append(md("Las variaciones del proyecto son la suma de las de sus actividades, así que cada actividad contribuye "
                     "con su SV_i = EV_i − PV_i y su CV_i = EV_i − AC_i:"))
    celdas.append(code("contrib = pd.DataFrame({'SV_i = EV − PV': p['EV'] - p['PV'], 'CV_i = EV − AC': p['EV'] - p['AC']})\n"
                       "display(limpia_tabla(contrib.sort_values('SV_i = EV − PV').T, decimales=1))\n"
                       "def ranking(serie, total, nombre):\n"
                       "    neg = serie[serie < -0.05].sort_values()\n"
                       "    if neg.empty:\n"
                       "        return f'ninguna actividad tiene {nombre} negativa'\n"
                       "    partes = [f'{a} ({v:.1f} k€, {100 * v / total:.0f} %)' for a, v in neg.items()]\n"
                       "    empate = [a for a, v in neg.items() if abs(v - neg.iloc[0]) < 0.05]\n"
                       "    nota = f' ({lista_y(empate)} empatan en cabeza)' if len(empate) > 1 else ''\n"
                       "    return ' > '.join(partes) + nota\n"
                       "SVt, CVt = contrib.iloc[:, 0].sum(), contrib.iloc[:, 1].sum()\n"
                       "Markdown(f'''Comprobación: SV = {SVt:.1f} k€ y CV = {CVt:.1f} k€, las mismas que con los totales.\n\n"
                       "**Respuesta:** SPI = {EV / PV:.3f} y CPI = {EV / AC:.3f}. Contribuciones a SV (de mayor a menor peso): {ranking(contrib.iloc[:, 0], SVt, 'SV')}. "
                       "Contribuciones a CV: {ranking(contrib.iloc[:, 1], CVt, 'CV')}.''')"))
    celdas += apartado(4)
    celdas.append(md("Criterio de prioridad (uno razonable entre varios): el retraso de una actividad crítica pesa más que el de "
                     "una con holgura, y el sobrecoste pesa igual en todas. Con las contribuciones negativas en k€:\n\n"
                     "$$\\text{prioridad} = |SV_i^-| \\cdot (2 + 3 \\cdot \\text{crítica}) + |CV_i^-|$$\n\n"
                     "donde $SV_i^-$ y $CV_i^-$ son la parte negativa de cada variación y *crítica* vale 1 si la holgura total "
                     "es 0 (el retraso de una crítica pesa 5 veces su valor, el de una no crítica 2 veces)."))
    celdas.append(code("puntuacion = pd.DataFrame({'crítica': (p['H_total'] == 0).astype(int), 'SV_i': contrib.iloc[:, 0], 'CV_i': contrib.iloc[:, 1]})\n"
                       "puntuacion['prioridad'] = (-puntuacion['SV_i'].clip(upper=0) * (2 + puntuacion['crítica'] * 3) - puntuacion['CV_i'].clip(upper=0))\n"
                       "orden = puntuacion.sort_values('prioridad', ascending=False)\n"
                       "vigilar = [a for a in orden.index if orden.loc[a, 'prioridad'] > 0.05][:3]\n"
                       "display(limpia_tabla(orden.assign(**{'crítica': orden['crítica'].map({1: 'sí', 0: 'no'})}), decimales=1))\n"
                       "calc = [f'{a}: {abs(min(orden.loc[a, \"SV_i\"], 0)):.1f}·{2 + 3 * orden.loc[a, \"crítica\"]} + {abs(min(orden.loc[a, \"CV_i\"], 0)):.1f} = {orden.loc[a, \"prioridad\"]:.1f}' for a in vigilar]\n"
                       "Markdown(f'''{\"; \".join(calc)}.\n\n"
                       "**Respuesta:** actividades a vigilar: **{lista_y(vigilar)}**. Un SPI bajo en una actividad con holgura puede absorberse sin mover el plazo "
                       "(su retraso solo importa cuando supera la holgura); en una crítica se traslada entero al final. El dinero, en cambio, no tiene holgura: "
                       "cada euro de sobrecoste, crítico o no, se suma al coste final, por lo que el CPI se vigila igual en todas.''')"))
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
    celdas.append(md(_SIGLAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"puntos = {puntos}\n"
                       f"umbral = {umbral}\n"
                       "dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}\n"
                       "factor_coste = {a: ('triangular', 0.9, 1.0, 1.4) for a in datos_proyecto.index}\n"
                       "plan = datos_proyecto['m']"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="plan", cap="Grafo PERT de la línea base (duraciones más probables) con el camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario de la línea base"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(plan)\n"
                       "T_plan = proyecto.duration(plan)\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'duracion']]"))
    celdas.append(md("El valor planificado de cada actividad en una fecha es su BAC por la fracción de su duración que, "
                     "según el calendario, debería haberse ejecutado; el PV del proyecto es la suma."))
    celdas.append(code('#| tbl-cap: "Valor planificado de cada actividad en cada fecha de control (k€)"\n#| label: tbl-pvFechas\n'
                       "pv_fechas = pd.DataFrame({f'semana {t}': datos_proyecto['BAC'] * ((t - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion']) for t in puntos})\n"
                       "pv_fechas.loc['Total (PV)'] = pv_fechas.sum()\n"
                       "pv_fechas.round(1)"))
    celdas.append(code("Markdown(f'Línea base: {T_plan} semanas, BAC = {datos_proyecto[\"BAC\"].sum():.0f} k€; PV en las fechas de control: '\n"
                       "         + ', '.join(f'semana {t}: {v:.1f} k€' for t, v in zip(puntos, pv_fechas.loc[\"Total (PV)\"])) + '.')"))
    celdas += apartado(2)
    celdas.append(md("En cada réplica se sortean las duraciones (triangulares) y los factores de coste, se ejecuta el proyecto "
                     "y, en cada fecha de control, se calculan SPI = EV / PV y CPI = EV / AC con la línea base anterior. La tabla "
                     "resume las 3 000 réplicas: media, desviación típica y percentiles 10 y 90 (entre ellos queda el 80 % central)."))
    celdas.append(code("resultados = {t: proyecto.monte_carlo_valor_ganado(dists, plan, datos_proyecto['BAC'], t, n_samples=3000, seed=11, factor_coste=factor_coste)\n"
                       "              for t in puntos}\n"
                       "pd.concat({f'semana {t}': r.summary().round(3) for t, r in resultados.items()})"))
    celdas.append(code("lineas = []\n"
                       "for t, r in resultados.items():\n"
                       "    s = r.summary()\n"
                       "    lineas.append(f'- Semana {t}: SPI medio {s.loc[\"SPI\", \"media\"]:.3f} (σ = {s.loc[\"SPI\", \"desv. típica\"]:.3f}; 80 % central entre '\n"
                       "                  f'{s.loc[\"SPI\", \"P10\"]:.3f} y {s.loc[\"SPI\", \"P90\"]:.3f}); CPI medio {s.loc[\"CPI\", \"media\"]:.3f} '\n"
                       "                  f'(σ = {s.loc[\"CPI\", \"desv. típica\"]:.3f}; entre {s.loc[\"CPI\", \"P10\"]:.3f} y {s.loc[\"CPI\", \"P90\"]:.3f})')\n"
                       "sd_spi = {t: r.summary().loc['SPI', 'desv. típica'] for t, r in resultados.items()}\n"
                       "sd_cpi = {t: r.summary().loc['CPI', 'desv. típica'] for t, r in resultados.items()}\n"
                       "f_medio = (0.9 + 1.0 + 1.4) / 3\n"
                       "sem = lambda x: ('la semana ' if len(x) == 1 else 'las semanas ') + lista_y(x)\n"
                       "mas = [str(t) for t in puntos if sd_spi[t] > sd_cpi[t]]; menos = [str(t) for t in puntos if sd_spi[t] <= sd_cpi[t]]\n"
                       "comp = ('más disperso que el CPI en las tres fechas' if not menos else 'menos disperso que el CPI en las tres fechas' if not mas else\n"
                       "        f'más disperso que el CPI en {sem(mas)} y menos en {sem(menos)}')\n"
                       "cpi_baja = all(b < a for a, b in zip(list(sd_cpi.values())[:-1], list(sd_cpi.values())[1:]))\n"
                       "Markdown('\\n'.join(lineas) + f'''\\n\\n**Respuesta:** el SPI es {comp} (σ del SPI {\" / \".join(f\"{v:.3f}\" for v in sd_spi.values())} "
                       "frente a {\" / \".join(f\"{v:.3f}\" for v in sd_cpi.values())} del CPI en las semanas {lista_y([str(t) for t in puntos])}); el SPI depende de qué "
                       "actividades se han adelantado o retrasado y de su peso en la fecha de control, y su media va de {\" a \".join(f\"{r.summary().loc['SPI', 'media']:.3f}\" for r in resultados.values())}. El CPI se concentra cerca de 1 / {f_medio:.2f} = {1 / f_medio:.3f}, "
                       "porque el factor de coste medio de la triangular (0,9; 1,0; 1,4) es (0,9 + 1,0 + 1,4) / 3 = {f_medio:.2f}{\", y su dispersión baja a medida que se acumulan más actividades en el EV\" if cpi_baja else \"\"}.''')"))
    celdas += apartado(3)
    celdas.append(md("Para cada réplica se compara la previsión con lo que ocurre de verdad en esa misma réplica: error = previsión − "
                     "real. El *sesgo* es la media del error (negativo: se prevé de menos), el *error absoluto medio* la media de "
                     "su valor absoluto y la *correlación* mide si las réplicas que se prevén peores terminan siendo peores."))
    celdas.append(code('#| tbl-cap: "Calidad de las previsiones en cada fecha de control"\n#| label: tbl-calidad\n'
                       "pd.concat({f'semana {t}': r.calidad_prevision().round(3) for t, r in resultados.items()})"))
    celdas.append(code("q = {t: r.calidad_prevision() for t, r in resultados.items()}\n"
                       "lineas = []\n"
                       "for t, c in q.items():\n"
                       "    sd = c.loc['duración', 'sesgo']\n"
                       "    lineas.append(f'- Semana {t}: duración con sesgo {sd:+.2f} semanas ({\"infravalora\" if sd < 0 else \"sobrevalora\"} la duración real en media), '\n"
                       "                  f'error absoluto medio {c.loc[\"duración\", \"error absoluto medio\"]:.2f} semanas y correlación {c.loc[\"duración\", \"correlación\"]:.2f}; '\n"
                       "                  f'coste con sesgo {c.loc[\"coste\", \"sesgo\"]:+.2f} k€, error absoluto medio {c.loc[\"coste\", \"error absoluto medio\"]:.1f} k€ y correlación {c.loc[\"coste\", \"correlación\"]:.2f}')\n"
                       "mae = [c.loc['duración', 'error absoluto medio'] for c in q.values()]\n"
                       "mae_c = [c.loc['coste', 'error absoluto medio'] for c in q.values()]\n"
                       "baja = all(b < a for a, b in zip(mae[:-1], mae[1:]))\n"
                       "s0 = list(q.values())[0].loc['duración', 'sesgo']\n"
                       "sesgo_txt = ('Al principio T/SPI infravalora la duración (sesgo negativo) porque el avance temprano aún no refleja los retrasos de las actividades futuras. ' if s0 < -0.25 else\n"
                       "             'Al principio T/SPI sobrevalora la duración (sesgo positivo): un retraso temprano en euros se extrapola a todo el proyecto aunque luego se absorba en holguras. ' if s0 > 0.25 else\n"
                       "             'T/SPI apenas tiene sesgo en la primera fecha. ')\n"
                       "Markdown('\\n'.join(lineas) + '\\n\\n**Respuesta:** el error absoluto medio de la previsión de duración ' + ('baja' if baja else 'pasa') + ' de '\n"
                       "         + ' a '.join(f'{v:.2f}' for v in mae) + ' semanas y el del coste de ' + ' a '.join(f'{v:.1f}' for v in mae_c) + ' k€ al avanzar el proyecto: '\n"
                       "         'cuanto más hecho, menos queda sujeto a incertidumbre. ' + sesgo_txt + 'BAC/CPI apenas tiene sesgo, porque el factor de sobrecoste es el mismo para todas las actividades.')"))
    celdas += apartado(4)
    celdas.append(code("r = resultados[puntos[1]]\n"
                       "retraso = r.tau_samples > r.T_plan\n"
                       "alarma = r.spi < umbral\n"
                       "n, n_al, n_re = len(retraso), int(alarma.sum()), int(retraso.sum())\n"
                       "n_al_re = int((alarma & retraso).sum()); n_sin_re = int((~alarma & retraso).sum())\n"
                       "p_con, p_sin = retraso[alarma].mean(), retraso[~alarma].mean()\n"
                       "sens = alarma[retraso].mean()\n"
                       "falsas = (~retraso[alarma]).mean()\n"
                       "Markdown(f'''En la semana {puntos[1]}, de las {n} réplicas, {n_al} dan alarma (SPI < {umbral}) y {n_re} terminan después de las {r.T_plan:.0f} semanas planificadas:\n\n"
                       "- P(retraso | alarma) = {n_al_re} / {n_al} = **{p_con:.3f}**\n"
                       "- P(retraso | sin alarma) = {n_sin_re} / {n - n_al} = **{p_sin:.3f}**\n"
                       "- Sensibilidad = P(alarma | retraso) = {n_al_re} / {n_re} = **{sens:.3f}**\n"
                       "- Alarmas falsas = P(sin retraso | alarma) = {n_al - n_al_re} / {n_al} = **{falsas:.3f}**\n\n"
                       "**Respuesta:** con alarma la probabilidad de terminar tarde es {p_con:.3f} y sin ella {p_sin:.3f}; la alarma avisa del {100 * sens:.1f} % de los proyectos que "
                       "terminarán con retraso y el {100 * falsas:.1f} % de sus avisos son falsos. {\"Cuando suena es muy fiable, pero se calla en la mayoría de los retrasos: \" if sens < 0.5 and falsas < 0.1 else \"La alarma discrimina (la probabilidad de retraso cambia mucho con ella), pero no es infalible: \"}"
                       "el SPI en una sola fecha es una señal con ruido.''')"))
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
    celdas.append(md(_SIGLAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"limite = [{L}]\n"
                       f"t_control = {tc}\n"
                       "BAC = datos_proyecto['BAC']"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la línea base (sin recursos) con el camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario de la línea base (inicios más tempranos, sin mirar los operarios)"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Carga de operarios de la línea base frente al límite"\n#| label: fig-cargasInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite)"))
    celdas.append(md("La ejecución real sigue el algoritmo de asignación: semana a semana, cada actividad disponible "
                     "(predecesoras terminadas) empieza si caben sus operarios durante toda su duración; si no, espera. "
                     "Traza semana a semana y Gantt resultante (filas 💤: esperas):"))
    celdas.append(code("res = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt de la ejecución real con el límite de operarios"\n#| label: fig-ganttReal\n'
                       "pinta_tikz(res.dibujos['operarios'])"))
    celdas.append(code('#| tbl-cap: "Inicio de cada actividad en la línea base y en la ejecución real"\n#| label: tbl-inicios\n'
                       "pd.DataFrame({'inicio línea base': calendario['inicio_mas_temprano'].astype(int), 'inicio real': res.inicios, 'fin real': res.fines,\n"
                       "              'espera (semanas)': res.esperas}).fillna(0).astype(int).T"))
    celdas.append(md("El PV de cada semana es la suma, sobre las actividades, del BAC por la fracción planificada "
                     "transcurrida (línea base); el EV, lo mismo con el calendario real (nadie trabaja más despacio, así "
                     "que la fracción ejecutada es la real del calendario con recursos)."))
    celdas.append(code("tabla = proyecto.valor_ganado_recursos(datos_proyecto, 'duration', ['operarios'], limite, BAC)\n"
                       "T_plan, T_real = tabla.attrs['T_plan'], tabla.attrs['T_real']\n"
                       "Markdown(f'**Respuesta:** la línea base dura {T_plan:.0f} semanas (calendario más temprano, tabla del calendario) y su PV semana a semana '\n"
                       "         f'es la columna PV de la tabla del apartado 2; con {limite[0]} operarios el calendario real (inicios de la tabla anterior) dura {T_real:.0f} semanas, '\n"
                       "         f'{T_real - T_plan:.0f} más.')"))
    celdas += apartado(2)
    celdas.append(md("Semana a semana: SPI = EV / PV y duración prevista = T / SPI, con T la duración de la línea base."))
    celdas.append(code('#| tbl-cap: "PV, EV, SPI y duración prevista semana a semana"\n#| label: tbl-valorGanado\n'
                       "tabla.round(3)"))
    celdas.append(code("primera = int(tabla.index[tabla['SPI'] < 0.999].min())\n"
                       "esperas = res.traza[res.traza.decision == 'espera'].sort_values('periodo')\n"
                       "primera_actividad = esperas.iloc[0]['actividad']\n"
                       "periodo_espera = int(esperas.iloc[0]['periodo'])\n"
                       "f1 = tabla.loc[primera]\n"
                       "Markdown(f'''En la semana {primera}: SPI = EV / PV = {f1[\"EV\"]:.1f} / {f1[\"PV\"]:.1f} = {f1[\"SPI\"]:.3f} y T / SPI = {T_plan:.0f} / {f1[\"SPI\"]:.3f} = {f1[\"duración prevista\"]:.1f} semanas. "
                       "La actividad {primera_actividad} debía empezar en la semana {int(calendario.loc[primera_actividad, \"inicio_mas_temprano\"]) + 1} según la línea base, pero no caben "
                       "{\"su\" if int(datos_proyecto.loc[primera_actividad, \"operarios\"]) == 1 else \"sus\"} {plural(int(datos_proyecto.loc[primera_actividad, \"operarios\"]), \"operario\")} y espera (periodo {periodo_espera} de la traza): su parte del PV se planificó y no se gana.\n\n"
                       "**Respuesta:** el SPI cae por debajo de 1 en la semana {primera}; la primera actividad que espera es {primera_actividad} (periodo {periodo_espera}).''')"))
    celdas += apartado(3)
    celdas.append(code("fila = tabla.loc[t_control]\n"
                       "prev = fila['duración prevista']\n"
                       "if prev > T_real:\n"
                       "    porque = ('T/SPI supone que el ritmo relativo de la semana de control se mantiene hasta el final; aquí el SPI está deprimido por esperas que '\n"
                       "              'ya se han producido y que no se repiten en el resto del proyecto, así que sobrestima el retraso.')\n"
                       "else:\n"
                       "    porque = ('T/SPI supone que el ritmo relativo de la semana de control se mantiene hasta el final; aquí quedan esperas por venir que el SPI '\n"
                       "              'de esta semana todavía no refleja, así que subestima el retraso.')\n"
                       "Markdown(f'''En la semana {t_control}: SPI = EV / PV = {fila[\"EV\"]:.1f} / {fila[\"PV\"]:.1f} = {fila[\"SPI\"]:.3f}; duración prevista T / SPI = {T_plan:.0f} / {fila[\"SPI\"]:.3f} = **{prev:.1f}** semanas, "
                       "frente a una duración real de **{T_real:.0f}** semanas (diferencia {prev - T_real:+.1f}). {porque}\n\n"
                       "**Respuesta:** la previsión acertada es la del calendario con recursos ({T_real:.0f} semanas), que conoce exactamente cuándo podrá empezar cada actividad; "
                       "T/SPI ({prev:.1f}) es solo una extrapolación.''')"))
    celdas += apartado(4)
    celdas.append(code("tabla2 = tabla[['PV', 'EV']].copy()\n"
                       "tabla2['PV con recursos'] = tabla['EV']        # línea base replanificada con el límite\n"
                       "tabla2['SPI con línea base realista'] = tabla2['EV'] / tabla2['PV con recursos']\n"
                       "display(tabla2.round(3))\n"
                       "Markdown(f'''Como nadie trabaja más despacio de lo previsto, la línea base que respeta el límite de operarios (calendario de la asignación, {T_real:.0f} semanas) "
                       "planifica exactamente lo que luego se gana: su PV coincide con el EV de cada semana y el SPI = EV / PV = 1 en todas. "
                       "**Respuesta:** el equipo debería haber preparado la línea base con la asignación de recursos ({limite[0]} operarios), comprometiendo {T_real:.0f} semanas en lugar de {T_plan:.0f}. "
                       "Los problemas de recursos se detectan al planificar, no al controlar: un SPI < 1 desde la primera espera es síntoma de una línea base irrealista, no de un equipo lento.''')"))
    return celdas, f"límite de {L} operarios y línea base sin recursos"


def _ev_recursos_incertidumbre(rng):
    """Valor ganado con límite de operarios y duraciones inciertas: qué línea base permite prever mejor."""
    estado = {}

    def acepta(i):
        if not (0 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 22):
            return False
        nombres = list(i["D"].index)
        a, m, b = P.tres_estimaciones_triangular(rng, i["D"])
        if ((b - a) < 2).sum() > len(nombres) // 2:
            return False
        op = P.recursos(rng, nombres, hi=3).rename("operarios")
        L = int(op.max())
        BAC = pd.Series(rng.integers(2, 9, size=len(nombres)) * 25, index=nombres, name="BAC")
        df = P.tabla_base(i, duration=i["D"], a=a, m=m, b=b, operarios=op, BAC=BAC)
        dist = {x: ("triangular", float(a[x]), float(m[x]), float(b[x])) for x in nombres}
        T = int(i["T"])
        tc = max(3, int(round(T * 0.5)))
        try:
            r = i["proyecto"].monte_carlo_valor_ganado_recursos(dist, i["D"], df[["operarios"]], ["operarios"], [L], BAC,
                                                                tc, n_samples=400, seed=1)
        except Exception:
            return False
        if not (2 <= r.T_recursos - r.T_libre <= 5):
            return False
        q = r.calidad_prevision()
        if q.loc["línea base con recursos", "error absoluto medio"] > q.loc["línea base sin recursos", "error absoluto medio"] - 0.8:
            return False
        estado.update(df=df, L=L, tc=tc)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), max_intentos=8000)
    df, L, tc = estado["df"], estado["L"], estado["tc"]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, estimaciones de duración (mínima *a*, más probable *m* y "
                 "máxima *b*, semanas, distribución triangular; *duration* es la duración esperada), operarios necesarios y "
                 "presupuesto (BAC, k€) se adjunta; el coste se reparte uniformemente en cada actividad. La empresa dispone de "
                 f"**{L} operarios** por semana y las actividades se ejecutan en cuanto el límite lo permite. Se barajan dos líneas "
                 "base con las duraciones esperadas: la *ingenua* (calendario más temprano, sin recursos) y la *realista* "
                 "(calendario respetando el límite de operarios).")]
    celdas.append(preguntas([
        "Calcule la duración y el PV de cada línea base y explique en qué se diferencian.",
        f"Simule 3 000 ejecuciones (duraciones triangulares, replanificando con el límite de operarios) y calcule el SPI en "
        f"la semana {tc} con cada línea base. Describa sus distribuciones.",
        "Prevea la duración con T/SPI con cada línea base y compárela con la duración real de cada ejecución: sesgo, error "
        "absoluto medio y correlación. ¿Qué línea base permite prever mejor y por qué?",
        "Repita el análisis en tres fechas de control. ¿Cómo evoluciona el error de cada línea base con el avance del proyecto?"]))
    celdas.append(code('#| tbl-cap: "Datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(md(_SIGLAS))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"limite, t_control = [{L}], {tc}\n"
                       "BAC = datos_proyecto['BAC']\n"
                       "dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}\n"
                       "demanda = datos_proyecto[['operarios']]"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones esperadas (línea base ingenua)", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario de la línea base ingenua (sin recursos)"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(md("La línea base realista aplica el algoritmo de asignación a las duraciones esperadas: traza semana a "
                     "semana y Gantt (filas 💤: esperas)."))
    celdas.append(code("res = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt de la línea base realista (con el límite de operarios)"\n#| label: fig-ganttRealista\n'
                       "pinta_tikz(res.dibujos['operarios'])"))
    celdas.append(md("El PV de cada actividad en la semana de control es su BAC por la fracción de su duración que, según cada "
                     "línea base, debería haberse ejecutado: $\\min(\\max(t - \\text{inicio}, 0), d)/d$."))
    celdas.append(code('#| tbl-cap: "Valor planificado de cada actividad en la semana de control con cada línea base"\n#| label: tbl-pvLineas\n'
                       "frac_ingenua = ((t_control - calendario['inicio_mas_temprano']).clip(lower=0).clip(upper=calendario['duracion']) / calendario['duracion'])\n"
                       "frac_realista = ((t_control - res.inicios.reindex(datos_proyecto.index)).clip(lower=0).clip(upper=datos_proyecto['duration']) / datos_proyecto['duration'])\n"
                       "pv_lineas = pd.DataFrame({'inicio ingenuo': calendario['inicio_mas_temprano'].astype(int), 'inicio realista': res.inicios.reindex(datos_proyecto.index).astype(int),\n"
                       "                          'PV ingenuo': (BAC * frac_ingenua).round(1), 'PV realista': (BAC * frac_realista).round(1)})\n"
                       "pv_lineas.loc['Total'] = ['', '', round((BAC * frac_ingenua).sum(), 1), round((BAC * frac_realista).sum(), 1)]\n"
                       "pv_lineas"))
    celdas.append(code("r = proyecto.monte_carlo_valor_ganado_recursos(dists, datos_proyecto['duration'], demanda, ['operarios'], limite, BAC, t_control, n_samples=3000, seed=11)\n"
                       "pv_i, pv_r = BAC * frac_ingenua, BAC * frac_realista\n"
                       "suma = lambda v: ' + '.join(f'{a}({x:.1f})' for a, x in v.items() if x > 1e-9) + f' = {v.sum():.1f}'\n"
                       "retrasadas = [a for a in datos_proyecto.index if pv_lineas.loc[a, 'inicio realista'] > pv_lineas.loc[a, 'inicio ingenuo']]\n"
                       "Markdown(f'''- PV ingenuo({t_control}) = {suma(pv_i)} k€\n"
                       "- PV realista({t_control}) = {suma(pv_r)} k€\n\n"
                       "**Respuesta:** línea base ingenua: {r.T_libre:.0f} semanas y PV({t_control}) = {r.pv_libre:.1f} k€; realista: {r.T_recursos:.0f} semanas y PV({t_control}) = {r.pv_recursos:.1f} k€. "
                       "La realista retrasa el inicio de {lista_y(retrasadas)} porque no hay operarios para todas a la vez (programa las esperas); la ingenua las ignora y parte de un plan imposible, "
                       "con {r.pv_libre - r.pv_recursos:.1f} k€ más de trabajo previsto para la semana {t_control}.''')"))
    celdas += apartado(2)
    celdas.append(md("En cada réplica se sortean las duraciones, se ejecuta el proyecto con el límite de operarios y se calcula "
                     "SPI = EV / PV en la semana de control con el PV de cada línea base. La tabla resume las 3 000 réplicas."))
    celdas.append(code('#| tbl-cap: "SPI, duración real y previsión T/SPI con cada línea base"\n#| label: tbl-spiRecursos\n'
                       "r.summary().round(3)"))
    celdas.append(code("s = r.summary()\n"
                       "Markdown(f'''**Respuesta:** con la línea base ingenua el SPI tiene media {s.loc[\"SPI sin recursos\", \"media\"]:.3f} (80 % central entre {s.loc[\"SPI sin recursos\", \"P10\"]:.3f} y {s.loc[\"SPI sin recursos\", \"P90\"]:.3f}): "
                       "{\"casi siempre por debajo de 1, aunque nadie trabaje más despacio de lo previsto\" if s.loc[\"SPI sin recursos\", \"P90\"] < 1 else \"desplazado hacia abajo\"}. "
                       "Con la realista tiene media {s.loc[\"SPI con recursos\", \"media\"]:.3f} (entre {s.loc[\"SPI con recursos\", \"P10\"]:.3f} y {s.loc[\"SPI con recursos\", \"P90\"]:.3f}): "
                       "centrado cerca de 1, con la dispersión que causan las duraciones inciertas.''')"))
    celdas += apartado(3)
    celdas.append(md("Error = previsión T/SPI − duración real de la misma réplica; *sesgo* = media del error (positivo: se prevé "
                     "más de lo que dura), *error absoluto medio* = media de su valor absoluto."))
    celdas.append(code('#| tbl-cap: "Calidad de la previsión de duración"\n#| label: tbl-calidadRecursos\n'
                       "calidad = r.calidad_prevision().round(3)\n"
                       "calidad"))
    celdas.append(code("mejor = calidad['error absoluto medio'].idxmin()\n"
                       "si, cr = calidad.loc['línea base sin recursos'], calidad.loc['línea base con recursos']\n"
                       "Markdown(f'''Duración real media {s.loc[\"duración real\", \"media\"]:.1f} semanas. Con la línea base ingenua T/SPI prevé de media {s.loc[\"T/SPI sin recursos\", \"media\"]:.1f} "
                       "(sesgo {si['sesgo']:+.2f}, error absoluto medio {si['error absoluto medio']:.2f} semanas); con la realista {s.loc[\"T/SPI con recursos\", \"media\"]:.1f} "
                       "(sesgo {cr['sesgo']:+.2f}, error absoluto medio {cr['error absoluto medio']:.2f}).\n\n"
                       "**Respuesta:** prevé mejor la {mejor} (error absoluto medio {calidad.loc[mejor, \"error absoluto medio\"]:.2f} semanas). Con la línea base ingenua el SPI cae por debajo de 1 "
                       "aunque el equipo trabaje como se esperaba y T/SPI {\"sobrestima\" if si['sesgo'] > 0 else \"subestima\"} sistemáticamente la duración; la realista ya incluye las esperas "
                       "y el SPI solo mide las desviaciones que no estaban previstas.''')"))
    celdas += apartado(4)
    celdas.append(code("filas = {}\n"
                       "for t in sorted({max(2, round(t_control * 0.5)), t_control, round(t_control * 1.5)}):\n"
                       "    rr = proyecto.monte_carlo_valor_ganado_recursos(dists, datos_proyecto['duration'], demanda, ['operarios'], limite, BAC, t, n_samples=3000, seed=11)\n"
                       "    q = rr.calidad_prevision()['error absoluto medio']\n"
                       "    filas[t] = {'sin recursos': q['línea base sin recursos'], 'con recursos': q['línea base con recursos']}\n"
                       "evolucion = pd.DataFrame(filas).T.rename_axis('semana de control')\n"
                       "display(evolucion.round(3))\n"
                       "peores = [str(t) for t in evolucion.index if evolucion.loc[t, 'sin recursos'] > evolucion.loc[t, 'con recursos']]\n"
                       "serie = lambda c: ' → '.join(f'{v:.2f}' for v in evolucion[c])\n"
                       "Markdown(f'''Error absoluto medio de T/SPI en las semanas {lista_y([str(t) for t in evolucion.index])}: línea base ingenua {serie(\"sin recursos\")} semanas; realista {serie(\"con recursos\")} semanas.\n\n"
                       "**Respuesta:** la línea base ingenua da peor previsión en {\"todas las fechas de control\" if len(peores) == len(evolucion) else (\"la semana \" if len(peores) == 1 else \"las semanas \") + lista_y(peores)}; "
                       "su error es sobre todo sistemático (compara con un plan imposible) y {\"se reduce\" if evolucion[\"sin recursos\"].iloc[-1] < evolucion[\"sin recursos\"].iloc[0] else \"no se reduce\"} al avanzar el proyecto, "
                       "cuando las esperas ya ocurridas pesan menos en lo que queda. El de la realista ({serie(\"con recursos\")} semanas) es ruido de las duraciones inciertas y no desaparece del todo hasta el final.''')"))
    return celdas, f"límite de {L} operarios; dos líneas base; control en la semana {tc}"


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
    Variante("ev-recursos-incertidumbre", 3, "límite de operarios y duraciones triangulares: SPI y T/SPI con línea base sin recursos frente a línea base con recursos",
             _ev_recursos_incertidumbre),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
