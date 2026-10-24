"""Colección 21 — Método de la cadena crítica (CCPM): estimaciones agresivas y seguras,
colchón del proyecto y colchones de alimentación, cadena crítica con recursos, gestión de
los colchones con el gráfico de fiebre (consumo de los colchones de alimentación, tendencia y
BPI frente al SPI del valor ganado), simulación del plan, varios proyectos con un recurso tambor
y comportamientos que consumen los márgenes (multitarea, síndrome del estudiante y ley de
Parkinson).

Las reglas son las de la lección 12 del curso y las de ``ProjectGraph.plan_cadena_critica``:
colchones redondeados al día superior; cada recurso es una unidad (un equipo) y los conflictos
se resuelven con el planificador en paralelo (menor inicio más tardío sin recursos, menor
duración, nombre); cadena de alimentación lo más tarde posible con su colchón delante y, si no
cabe, colchón reducido a la holgura; porcentaje de cadena completada = (duración de la cadena −
duración restante) / duración de la cadena; zonas del gráfico de fiebre 10→70 % y 30→90 %.
Duraciones simuladas lognormales con mediana = agresiva y percentil 90 = segura."""
import contextlib
import io
import math

import numpy as np
import pandas as pd

from .. import proyectos as P
from .. import contextos
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo, ESTADO,
                        literal_dataframe)

META = dict(slug="21_cadena_critica", prefijo="CadenaCritica",
            titulo="Método de la cadena crítica",
            descripcion="Estimaciones agresivas y seguras, colchón del proyecto (cortar y pegar, raíz de la suma "
                        "de cuadrados), colchones de alimentación y programación lo más tarde posible, cadena "
                        "crítica con recursos, seguimiento con el gráfico de fiebre y el BPI (consumo de los "
                        "colchones de alimentación, tendencia, comparación con el SPI), simulación del plan, varios "
                        "proyectos con un recurso tambor (escalonamiento) y comportamientos que consumen los márgenes "
                        "(multitarea, ley de Parkinson).",
            n_defecto=2, contexto=False)

EQUIPOS = ["X", "Y"]
Z90 = 1.2815515655446004

# ---------------------------------------------------------------------------
# Celdas comunes
# ---------------------------------------------------------------------------
AYUDAS = r'''
# Funciones auxiliares de la cadena crítica: cálculos intermedios con los números sustituidos
from math import ceil, sqrt
from pmule import grafico_fiebre_tikz
pd.set_option('display.max_colwidth', None)
GANTT_CC = dict(GANTT_PARAMS, period_width=0.6, activity_inner_text_style=r'\bfseries\small')


def sin_decimales(tabla):
    """Números enteros sin «.0» al mostrar una tabla."""
    return tabla.map(lambda v: f'{v:g}' if isinstance(v, float) else v)


def tabla_margenes(datos, actividades):
    """Margen retirado (segura − agresiva) de cada actividad."""
    t = datos.loc[list(actividades), ['segura', 'agresiva']].copy()
    t['margen = segura − agresiva'] = [f'{s} − {a} = {s - a}' for s, a in zip(t['segura'], t['agresiva'])]
    return t.T


def detalle_colchon(margenes, nombre='colchón'):
    """Colchón por cortar y pegar y por la raíz de la suma de cuadrados (redondeo al día superior)."""
    ms = [int(m) for m in margenes]
    suma, cuad = sum(ms), sum(m * m for m in ms)
    cp, rz = 0.5 * suma, sqrt(cuad)
    return pd.DataFrame({nombre: {
        'cortar y pegar = 0,5·Σ márgenes': f"0,5·({' + '.join(map(str, ms))}) = 0,5·{suma} = {cp:g} → {ceil(cp - 1e-9)}",
        'raíz = √(Σ márgenes²)': f"√({' + '.join(f'{m}²' for m in ms)}) = √{cuad} = {rz:.2f} → {ceil(rz - 1e-9)}"}})


def tabla_alimentacion(plan):
    """Colchones de alimentación: arco en el que desemboca la cadena de alimentación, colchón
    requerido, holgura disponible y colchón adoptado."""
    filas = {}
    for nombre, fila in plan.alimentacion.iterrows():
        ms = [int(x) for x in str(fila['margenes']).split(' + ')]
        if plan.metodo_alimentacion == 'cortar_pegar':
            calc = f"0,5·({' + '.join(map(str, ms))}) = {0.5 * sum(ms):g} → {int(fila['requerido'])}"
        else:
            calc = f"√({' + '.join(f'{m}²' for m in ms)}) = {sqrt(sum(m * m for m in ms)):.2f} → {int(fila['requerido'])}"
        entrada = int(plan.inicio_temprano[fila['antes_de']]) if fila['antes_de'] != 'fin' else int(plan.duracion_cadena)
        fin_p = int(plan.inicio_temprano[fila['desde']] + plan.agresiva[fila['desde']])
        filas[nombre] = {'arco': f"{fila['desde']} → {fila['antes_de']}",
                         'cadena de alimentación': fila['cadena_alimentacion'],
                         'colchón requerido': calc,
                         'holgura disponible = inicio de la cadena − fin más temprano': f"{entrada} − {fin_p} = {int(fila['disponible'])}",
                         'colchón adoptado = mín(requerido, disponible)': int(fila['adoptado'])}
    return pd.DataFrame(filas).T


def detalle_alap(plan):
    """Programación lo más tarde posible de las actividades de alimentación (de la última a la primera)."""
    cadena = set(plan.cadena)
    ca = {(fila['desde'], fila['antes_de']): (n, int(fila['adoptado'])) for n, fila in plan.alimentacion.iterrows()}
    suc = {a: sorted(b for b in plan.predecesoras if a in plan.predecesoras[b]) for a in plan.predecesoras}
    filas = {}
    for a in reversed(plan.orden):
        if a in cadena:
            continue
        cands = []
        for b in suc[a]:
            if b in cadena:
                n, fb = ca[(a, b)]
                cands.append(f'{b} − {n}: {int(plan.inicios[b])} − {fb} = {int(plan.inicios[b]) - fb}')
            else:
                cands.append(f'{b}: {int(plan.inicios[b])}')
        if not suc[a]:
            n, fb = ca[(a, 'fin')]
            cands.append(f'fin − {n}: {plan.duracion_cadena} − {fb} = {plan.duracion_cadena - fb}')
        fin = int(plan.inicios[a] + plan.agresiva[a])
        filas[a] = {'fin = mín(inicio de las sucesoras)': f"mín({'; '.join(cands)}) = {fin}",
                    'inicio = fin − duración': f'{fin} − {int(plan.agresiva[a])} = {int(plan.inicios[a])}',
                    'inicio más temprano': int(plan.inicio_temprano[a])}
    return pd.DataFrame(filas).T


def resumen_informes(plan, informes):
    """Cálculos de cada informe de avance con los números sustituidos."""
    T, CP = plan.duracion_cadena, plan.colchon_proyecto
    columnas = {}
    for inf in informes:
        det = inf['detalle']
        curso = det[det['estado'] == 'en curso']
        pend = det[det['estado'] == 'pendiente']
        partes = [f"{r:g} ({a})" for a, r in curso['restante'].items()] + [f"{int(d)} ({a})" for a, d in pend['agresiva'].items()]
        hecho = float(inf['avance'])
        largas = [f"a {a} le quedan {r:g} días, más que su duración agresiva: se cuentan {int(d)}"
                  for a, r, d in zip(curso.index, curso['restante'], curso['agresiva']) if r > d]
        pa, pc = inf['pct_avance'], inf['pct_consumo']
        columnas[f"día {inf['dia']}"] = {
            'fin previsto = día + restante de la cadena': f"{inf['dia']} + {' + '.join(partes) if partes else '0'} = {inf['fin_previsto']:g}",
            'colchón consumido': f"{inf['fin_previsto']:g} − {T} = {inf['consumo']:g} días → {inf['consumo']:g}/{CP} = {pc:.1f} %",
            'cadena completada': (f"({T} − {T - hecho:g})/{T} = {pa:.1f} %"
                                  + (f" ({'; '.join(largas)})" if largas else '')),
            'BPI = % colchón restante / % cadena restante': (f"(100 − {pc:.1f})/(100 − {pa:.1f}) = {inf['bpi']:.2f}" if pa < 100 else '—'),
            'límite verde = 10 + 0,6·avance': f"10 + 0,6·{pa:.1f} = {inf['limite_verde']:.1f} %",
            'límite rojo = 30 + 0,6·avance': f"30 + 0,6·{pa:.1f} = {inf['limite_roja']:.1f} %",
            'zona': inf['zona']}
    return pd.DataFrame(columnas)
'''.strip("\n")

ARRANQUE = ("datos_proyecto = datos_enunciado.copy()\n"
            "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
            "proyecto = prelaciones.create_project()")

GANTT_CCPM = "GANTT_CC"


def _cal(obj="proyecto", durations="datos_proyecto['duration']", cap="Calendario", label="tbl-calendario"):
    """Celda del calendario con el visor interactivo (grafo activo)."""
    extra = ', grafo="roy"' if ESTADO["grafo"] == "roy" else ""
    return code(f'#| tbl-cap: "{cap}"\n#| label: {label}\n'
                f"calendario = {obj}.calendar({durations})\n"
                f"{obj}.display_calendario({durations}{extra})")


# ---------------------------------------------------------------------------
# Proyectos aleatorios con estimaciones agresiva y segura
# ---------------------------------------------------------------------------
def _estimaciones(rng, nombres, lo=2, hi=9):
    ag = pd.Series(rng.integers(lo, hi + 1, size=len(nombres)), index=nombres, name="agresiva")
    f = rng.choice([0.2, 0.3, 0.4, 0.5, 0.6], size=len(nombres))
    mg = np.maximum(1, np.ceil(ag.values * f - 1e-9)).astype(int)
    return ag, (ag + mg).rename("segura")


def _equipos(rng, nombres, n_rec):
    rec = pd.Series("---", index=nombres, name="recurso", dtype=object)
    libres = list(nombres)
    for r in EQUIPOS[:n_rec]:
        k = int(rng.integers(2, 4))
        for a in sorted(rng.choice(libres, size=min(k, len(libres)), replace=False)):
            rec[a] = r
            libres.remove(a)
    return rec


def _buscar(rng, acepta, n=(8, 9), n_rec=0, metodo="cortar_pegar", max_intentos=6000):
    """Muestrea proyectos (un único final) hasta que ``acepta(info, plan, df)``; la cadena es única."""
    for _ in range(max_intentos):
        nn = int(rng.choice(list(n))) if isinstance(n, (tuple, list)) else int(n)
        preds = P.dag_aleatorio(rng, nn, n_inicio=int(rng.choice([2, 3])), n_fin=1)
        nombres = list(preds)
        ag, se = _estimaciones(rng, nombres)
        rec = _equipos(rng, nombres, n_rec) if n_rec else None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                info = P.analizar(preds, ag)
                cols = dict(agresiva=ag, segura=se)
                if rec is not None:
                    cols["recurso"] = rec
                df = P.tabla_base(info, **cols)
                plan = info["proyecto"].plan_cadena_critica(df, recurso="recurso" if rec is not None else None,
                                                            metodo=metodo)
        except Exception:
            continue
        if plan.cadenas_empatadas != 1:
            continue
        if acepta(info, plan, df):
            return info, plan, df
    raise RuntimeError("No se encontró un proyecto con los rasgos pedidos")


def _rutas_criticas(info, durac):
    L = info["P"] @ durac[info["P"].columns]
    T = float(L.max())
    return [r for r in L.index if np.isclose(L[r], T)], T


def _con_contexto(celdas, rng):
    """La mitad de las veces, descripción de las actividades según un contexto narrativo (en días)."""
    if rng.random() >= 0.5:
        return celdas
    i, nombres = contextos._actividades_de_la_tabla(celdas)
    if i < 0 or not nombres or len(nombres) > 14:
        return celdas
    ctx = contextos.elegir(rng, len(nombres))
    lineas = ["descripciones = pd.Series({"] + [f"    {a!r}: {d!r}," for a, d in zip(nombres, ctx["actividades"])]
    lineas += ["}, name='Descripción')", "descripciones.to_frame()"]
    return (celdas[:i + 1]
            + [md(f"El proyecto corresponde {contextos.con_a(ctx['tema'])}; las actividades son las siguientes "
                  "(las duraciones del cuadro están en días)."),
               code('#| tbl-cap: "Descripción de las actividades"\n#| label: tbl-descripciones\n' + "\n".join(lineas))]
            + celdas[i + 1:])


def _datos(df):
    return code('#| tbl-cap: "Cuadro de datos del enunciado (días)"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df))


ESTIMACIONES = ("Para cada actividad, su responsable ha dado una estimación **segura** (con la que cumpliría "
                "aproximadamente nueve de cada diez veces, el percentil 90) y una estimación **agresiva pero "
                "alcanzable** (la mediana, el percentil 50), ambas en días.")


# ---------------------------------------------------------------------------
# Variante 1 — Multitarea
# ---------------------------------------------------------------------------
MULTITAREA = r'''
def multitarea(duraciones, bloque=None, cambio=0):
    """Turnos de una persona con varias tareas. Sin multitarea (bloque=None) las hace una detrás de
    otra en el orden dado; con multitarea trabaja `bloque` días en cada tarea pendiente, rotando en
    ese orden. Cada vez que pasa de una tarea a otra pierde `cambio` días.
    Devuelve (día en que termina cada tarea, tabla de turnos)."""
    restante = dict(duraciones)
    t, turnos, fin, anterior = 0, [], {}, None
    while restante:
        for tarea in list(duraciones):
            if tarea not in restante:
                continue
            if anterior is not None and tarea != anterior and cambio:
                turnos.append(('cambio', t, cambio))
                t += cambio
            trabajo = restante[tarea] if bloque is None else min(bloque, restante[tarea])
            turnos.append((tarea, t, trabajo))
            t += trabajo
            restante[tarea] -= trabajo
            anterior = tarea
            if restante[tarea] == 0:
                fin[tarea] = t
                del restante[tarea]
    return pd.Series(fin).reindex(list(duraciones)), pd.DataFrame(turnos, columns=['tarea', 'inicio', 'días'])


def dibuja_turnos(turnos, tareas, titulo, ax):
    """Diagrama de barras de los turnos (los cambios, en gris)."""
    colores = dict(zip(tareas, ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']))
    for i, tarea in enumerate(tareas):
        tramos = [(r.inicio, r.días) for r in turnos.itertuples() if r.tarea == tarea]
        ax.broken_barh(tramos, (i - 0.35, 0.7), color=colores[tarea])
    cambios = [(r.inicio, r.días) for r in turnos.itertuples() if r.tarea == 'cambio']
    if cambios:
        ax.broken_barh(cambios, (len(tareas) - 0.35, 0.7), color='lightgrey')
    ax.set_yticks(range(len(tareas) + (1 if cambios else 0)))
    ax.set_yticklabels(list(tareas) + (['cambios'] if cambios else []))
    ax.set_xlim(0, float((turnos['inicio'] + turnos['días']).max()) + 0.5)
    ax.invert_yaxis(); ax.set_title(titulo, fontsize=10); ax.grid(axis='x', alpha=0.3)


def sin_decimales(tabla):
    """Números enteros sin «.0» al mostrar una tabla."""
    return tabla.map(lambda v: f'{v:g}' if isinstance(v, float) else v)
'''.strip("\n")


def _turnos(dur, bloque):
    """Fines de cada tarea con multitarea en turnos de ``bloque`` días (sin pérdida al cambiar)."""
    resto, t, fin = list(dur), 0, [0] * len(dur)
    while any(resto):
        for i, r in enumerate(resto):
            if r:
                w = min(bloque, r)
                t += w
                resto[i] -= w
                if resto[i] == 0:
                    fin[i] = t
    return fin


def _multitarea(rng):
    k = int(rng.choice([3, 4]))
    while True:
        dur = [int(x) for x in rng.integers(4, 13, size=k)]
        bloque = int(rng.choice([2, 3, 4]))
        seguidas = list(np.cumsum(dur))
        multi = _turnos(dur, bloque)
        if (sum(1 for d in dur if d > bloque) >= k - 1 and len(set(dur)) >= 2 and sorted(dur) != dur
                and sum(multi) > sum(seguidas) and sum(1 for a, b in zip(multi, seguidas) if a > b) >= k - 1):
            break
    cambio = int(rng.choice([1, 1, 2]))
    tareas = [f"T{i}" for i in range(1, k + 1)]
    df = pd.DataFrame({"duration": dur}, index=pd.Index(tareas, name="tarea"))
    persona = str(rng.choice(["Una ingeniera", "Un técnico", "Una diseñadora", "Un analista"]))
    celdas = [md("# Enunciado"),
              md(f"{persona} de la oficina técnica tiene asignadas {k} tareas, cada una de un proyecto distinto, "
                 "con las duraciones (días de trabajo efectivo) del cuadro. Los {k} jefes de proyecto le presionan "
                 "para que «avance un poco en lo de todos».".replace("{k}", str(k))),
              preguntas([
                  "Si hace las tareas una detrás de otra en el orden del cuadro, sin interrupciones, ¿qué día termina "
                  "cada una? ¿Cuál es el tiempo medio de terminación?",
                  f"Si reparte su tiempo en turnos de {bloque} días, rotando entre las tareas pendientes en el orden del "
                  "cuadro (multitarea), y cambiar de tarea no le cuesta nada, ¿qué día termina cada una? ¿Cuánto se "
                  "retrasa cada tarea y cuánto el tiempo medio respecto al apartado 1?",
                  f"Repita los dos cálculos suponiendo que cada vez que pasa de una tarea a otra pierde {cambio} "
                  f"día{'s' if cambio > 1 else ''} en retomarla (preparación, recuperar el hilo). ¿Cuándo termina el "
                  "último trabajo en cada caso?",
                  "¿En qué orden habría que hacerlas, sin multitarea, para minimizar el tiempo medio de terminación? "
                  "¿Qué recomienda el método de la cadena crítica sobre la multitarea y por qué?"]),
              code('#| tbl-cap: "Tareas y duraciones (días)"\n#| label: tbl-datosEnunciado\n'
                   + literal_dataframe(df, indice="tarea")),
              md("# Resolución"),
              code(MULTITAREA),
              code(f"duraciones = datos_enunciado['duration'].to_dict()\ntareas = list(duraciones)\n"
                   f"bloque, cambio = {bloque}, {cambio}")]
    celdas += apartado(1)
    celdas.append(md("Sin multitarea cada tarea termina cuando han terminado ella y todas las anteriores: los fines son "
                     "las sumas acumuladas de las duraciones."))
    celdas.append(code('#| tbl-cap: "Sin multitarea: día de terminación de cada tarea"\n#| label: tbl-secuencial\n'
                       "fin_seguidas, turnos_seguidas = multitarea(duraciones)\n"
                       "pd.DataFrame({'fin = suma acumulada': [' + '.join(str(duraciones[x]) for x in tareas[:i + 1]) + f' = {fin_seguidas[t]}' for i, t in enumerate(tareas)]}, index=tareas).T"))
    celdas.append(code("Markdown(f'Tiempo medio de terminación sin multitarea: ({\" + \".join(str(v) for v in fin_seguidas)})/{len(tareas)} = **{fin_seguidas.mean():.2f} días**; el último trabajo termina el día {fin_seguidas.max()}.')"))
    celdas += apartado(2)
    celdas.append(md(f"Con multitarea se trabaja {bloque} días en cada tarea pendiente, por turno, en el orden del cuadro; "
                     "una tarea sale de la rotación cuando termina. La tabla de turnos recoge cada tramo de trabajo."))
    celdas.append(code('#| tbl-cap: "Turnos con multitarea (sin pérdida al cambiar)"\n#| label: tbl-turnos\n'
                       "fin_multi, turnos_multi = multitarea(duraciones, bloque)\n"
                       "turnos_multi.assign(fin=turnos_multi['inicio'] + turnos_multi['días']).T"))
    celdas.append(code('#| tbl-cap: "Día de terminación con y sin multitarea"\n#| label: tbl-comparacion\n'
                       "comparacion = pd.DataFrame({'sin multitarea': fin_seguidas, 'con multitarea': fin_multi})\n"
                       "comparacion['retraso'] = comparacion['con multitarea'] - comparacion['sin multitarea']\n"
                       "comparacion.loc['media'] = comparacion.mean()\n"
                       "sin_decimales(comparacion.astype(float).round(2))"))
    celdas.append(code("retrasadas = [t for t in tareas if comparacion.loc[t, 'retraso'] > 0]\n"
                       "Markdown(f'El último trabajo termina igual (día {fin_multi.max()}: el trabajo total es el mismo), pero '\n"
                       "         f'{\", \".join(retrasadas)} terminan más tarde: el tiempo medio pasa de {fin_seguidas.mean():.2f} a {fin_multi.mean():.2f} días '\n"
                       "         f'(+{fin_multi.mean() - fin_seguidas.mean():.2f}). Cada proyecto recibe su resultado más tarde y, si la tarea es crítica, '\n"
                       "         'el proyecto entero se retrasa.')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Con pérdida en cada cambio de tarea"\n#| label: tbl-cambio\n'
                       "fin_seguidas_c, turnos_seguidas_c = multitarea(duraciones, None, cambio)\n"
                       "fin_multi_c, turnos_multi_c = multitarea(duraciones, bloque, cambio)\n"
                       "n_cambios = pd.Series({'sin multitarea': int((turnos_seguidas_c['tarea'] == 'cambio').sum()),\n"
                       "                       'con multitarea': int((turnos_multi_c['tarea'] == 'cambio').sum())})\n"
                       "tabla_c = pd.DataFrame({'sin multitarea': fin_seguidas_c, 'con multitarea': fin_multi_c})\n"
                       "tabla_c.loc['media'] = tabla_c.mean()\n"
                       "tabla_c.loc['cambios de tarea'] = n_cambios\n"
                       "tabla_c.loc['días perdidos'] = n_cambios * cambio\n"
                       "sin_decimales(tabla_c.astype(float).round(2))"))
    celdas.append(code('#| fig-cap: "Turnos de trabajo: sin multitarea y con multitarea (con pérdida en cada cambio)"\n#| label: fig-turnos\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, axs = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)\n"
                       "dibuja_turnos(turnos_seguidas_c, tareas, 'Sin multitarea', axs[0])\n"
                       "dibuja_turnos(turnos_multi_c, tareas, f'Multitarea en turnos de {bloque} días', axs[1])\n"
                       "axs[1].set_xlabel('días'); plt.tight_layout(); plt.show()"))
    celdas.append(code("Markdown(f'Con la pérdida por cambio, la multitarea ya no sólo retrasa las tareas intermedias: el último trabajo pasa del día '\n"
                       "         f'{fin_seguidas_c.max()} al {fin_multi_c.max()} ({n_cambios[\"con multitarea\"]} cambios frente a {n_cambios[\"sin multitarea\"]}), '\n"
                       "         'es decir, la persona produce menos. Es el efecto que Goldratt atribuye a la multitarea mal entendida.')"))
    celdas += apartado(4)
    celdas.append(code("orden_spt = sorted(tareas, key=lambda t: (duraciones[t], t))\n"
                       "fin_spt, _ = multitarea({t: duraciones[t] for t in orden_spt})\n"
                       "Markdown(f'Ordenando de la más corta a la más larga ({\" → \".join(orden_spt)}) el tiempo medio de terminación es '\n"
                       "         f'{fin_spt.mean():.2f} días, el mínimo posible sin multitarea (frente a {fin_seguidas.mean():.2f} en el orden del cuadro): '\n"
                       "         'es la regla de la duración más corta primero: si dos tareas seguidas están en orden larga–corta, intercambiarlas '\n"
                       "         'adelanta la corta lo que dura la larga y retrasa la larga sólo lo que dura la corta.')"))
    celdas.append(md("El método de la cadena crítica pide **evitar la multitarea** en las actividades de la cadena: quien "
                     "trabaja en una actividad crítica se dedica a ella hasta terminarla y la entrega en cuanto acaba, "
                     "como el corredor de relevos. La multitarea alarga todas las actividades que se reparten (cada una "
                     "termina cuando termina su último turno), retrasa la entrega a las sucesoras y, con la pérdida de "
                     "cada cambio, reduce la capacidad real del recurso. Para decidir qué tarea va primero, la prioridad "
                     "la marca el estado de los colchones de cada proyecto, no quien más presiona."))
    celdas += _autoevaluacion(_p_multitarea(dur, bloque, cambio), f"Cadena crítica — multitarea ({k} tareas, turnos de {bloque} días)")
    return celdas, f"{k} tareas en turnos de {bloque} días; cambio de tarea de {cambio} día{'s' if cambio > 1 else ''}"


# ---------------------------------------------------------------------------
# Variante 2 — Colchón del proyecto
# ---------------------------------------------------------------------------
def _colchon_proyecto(rng):
    def acepta(info, plan, df):
        crit_s, _ = _rutas_criticas(info, df["segura"])
        return len(crit_s) == 1 and 4 <= len(plan.cadena) and plan.duracion_tradicional > plan.compromiso
    info, plan, df = _buscar(rng, acepta)
    crit_s, T_s = _rutas_criticas(info, df["segura"])
    distinta = set(plan.cadena) != set(info["P"].columns[info["P"].loc[crit_s[0]] == 1])
    celdas = [md("# Enunciado"),
              md("Se va a programar un proyecto con el método de la cadena crítica. " + ESTIMACIONES +
                 " Las actividades no comparten recursos. Los colchones se redondean al día superior."),
              preguntas([
                  "Calcule la duración del proyecto con las estimaciones seguras (cronograma tradicional) e indique su "
                  "ruta crítica.",
                  "Con las estimaciones agresivas, determine la cadena crítica y su duración, y el margen de seguridad "
                  "retirado de cada una de sus actividades.",
                  "Dimensione el colchón del proyecto por el método de cortar y pegar y por el de la raíz de la suma de "
                  "cuadrados. ¿Qué fecha se compromete con el cliente con el de cortar y pegar? ¿Cuántos días se ganan "
                  "respecto al cronograma tradicional?",
                  "¿Por qué el colchón del proyecto es menor que el margen total retirado de la cadena? ¿Por qué no debe "
                  "penalizarse a quien supere su estimación agresiva?"]),
              _datos(df),
              md("# Resolución"),
              code(ARRANQUE + "\nR = proyecto.path_matrix(dummies=False)"),
              code(AYUDAS)]
    celdas += apartado(1)
    celdas.append(code("datos_proyecto['duration'] = datos_proyecto['segura']"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones seguras y su camino crítico", label="fig-pert")
    celdas.append(_cal(cap="Calendario del cronograma tradicional (estimaciones seguras)"))
    celdas.append(code('#| tbl-cap: "Duración de cada ruta con las estimaciones seguras"\n#| label: tbl-rutasSeguras\n'
                       "rutas_seguras = detalle_rutas(R, datos_proyecto['segura'][R.columns])\n"
                       "rutas_seguras"))
    celdas.append(code("T_tradicional = int(rutas_seguras['duración'].max())\n"
                       "ruta_tradicional = rutas_seguras['duración'].astype(float).idxmax()\n"
                       "Markdown(f'El cronograma tradicional dura **{T_tradicional} días**; ruta crítica: {rutas_seguras.loc[ruta_tradicional, \"actividades\"]}.')"))
    celdas += apartado(2)
    celdas.append(md("La cadena crítica es la secuencia más larga de actividades dependientes con las estimaciones "
                     "agresivas. Sin recursos compartidos, es la ruta crítica calculada con esas estimaciones."))
    celdas += fig_grafo(durations="datos_proyecto['agresiva']", cap="Grafo PERT con las estimaciones agresivas: su camino crítico es la cadena crítica", label="fig-pertAgresiva")
    celdas.append(code('#| tbl-cap: "Duración de cada ruta con las estimaciones agresivas"\n#| label: tbl-rutasAgresivas\n'
                       "rutas_agresivas = detalle_rutas(R, datos_proyecto['agresiva'][R.columns])\n"
                       "rutas_agresivas"))
    celdas.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura', metodo='cortar_pegar')\n"
                       "cadena = plan.cadena\n"
                       "Markdown(f'Cadena crítica: **{\" - \".join(cadena)}**, con {\" + \".join(str(int(datos_proyecto.loc[a, \"agresiva\"])) for a in cadena)} = '\n"
                       "         f'**{plan.duracion_cadena} días**.')"))
    celdas.append(code('#| tbl-cap: "Margen retirado de las actividades de la cadena"\n#| label: tbl-margenes\n'
                       "tabla_margenes(datos_proyecto, cadena)"))
    celdas.append(code("margen_cadena = int(plan.margen[cadena].sum())\n"
                       "Markdown(f'El margen retirado de la cadena suma {\" + \".join(str(int(plan.margen[a])) for a in cadena)} = **{margen_cadena} días**.')"))
    if distinta:
        celdas.append(md("La cadena crítica no coincide con la ruta crítica del cronograma tradicional: al retirar los "
                         "márgenes, que no son proporcionales a las duraciones, puede pasar a ser más larga otra ruta. "
                         "La cadena se identifica siempre con las estimaciones agresivas."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Colchón del proyecto por los dos métodos"\n#| label: tbl-colchon\n'
                       "detalle_colchon(plan.margen[cadena], 'colchón del proyecto')"))
    celdas.append(code("CP = plan.colchon_proyecto\n"
                       "Markdown(f'Con cortar y pegar el colchón es de {CP} días y la fecha comprometida, {plan.duracion_cadena} + {CP} = '\n"
                       "         f'**{plan.compromiso} días**, frente a los {T_tradicional} del cronograma tradicional: '\n"
                       "         f'**{T_tradicional - plan.compromiso} días menos**. Con la raíz de la suma de cuadrados el colchón sería de '\n"
                       "         f'{ceil(plan.calculo_colchon_proyecto[\"raiz\"] - 1e-9)} días.')"))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'''Los márgenes individuales no se suman: es muy improbable que todas las actividades de la cadena consuman a la vez "
                       "todo su margen. Si las duraciones son independientes, lo que se suma son las varianzas, y la desviación de la cadena crece como "
                       "la raíz de la suma de cuadrados ({plan.calculo_colchon_proyecto[\"raiz\"]:.2f} días frente a {margen_cadena}). Concentrar el margen al "
                       "final protege la fecha de entrega con menos días que repartido por las actividades, donde además se perdería por el síndrome del "
                       "estudiante y la ley de Parkinson.\n\n"
                       "La estimación agresiva es la mediana: se espera que la mitad de las actividades la superen. Penalizarlo llevaría a los "
                       "responsables a volver a inflar sus estimaciones, y el margen oculto reaparecería. Lo que se controla es el consumo del "
                       "colchón del proyecto, no el cumplimiento de cada duración.''')"))
    celdas += _autoevaluacion(_p_colchon(plan), f"Cadena crítica — colchón del proyecto (cadena de {plan.duracion_cadena} días)")
    return _con_contexto(celdas, rng), (f"{len(df)} actividades; cadena de {len(plan.cadena)} actividades; "
                                       f"compromiso {plan.compromiso} frente a {plan.duracion_tradicional} días")


# ---------------------------------------------------------------------------
# Variante 3 — Dimensionado del colchón: cortar y pegar frente a raíz
# ---------------------------------------------------------------------------
def _dimensionado(rng):
    def acepta(info, plan, df):
        return len(plan.cadena) >= 5 and plan.calculo_colchon_proyecto["cortar_pegar"] > plan.calculo_colchon_proyecto["raiz"] + 1
    info, plan, df = _buscar(rng, acepta, n=(9, 10))
    celdas = [md("# Enunciado"),
              md("Un proyecto se programa con el método de la cadena crítica. " + ESTIMACIONES +
                 " Las actividades no comparten recursos y los colchones se redondean al día superior."),
              preguntas([
                  "Determine la cadena crítica con las estimaciones agresivas, su duración y el margen retirado de cada "
                  "una de sus actividades.",
                  "Calcule el colchón del proyecto por el método de cortar y pegar y por el de la raíz de la suma de "
                  "cuadrados.",
                  "Suponga que la duración de cada actividad de la cadena es normal, independiente de las demás, con "
                  "media su estimación agresiva y con la estimación segura como percentil 90, de modo que "
                  "σ = margen/1,2816. ¿Con qué probabilidad termina la cadena en su duración más el colchón, con cada "
                  "uno de los dos métodos?",
                  "Si una cadena tuviera n actividades con el mismo margen m (tome el margen medio de esta cadena), "
                  "¿cómo crecen los dos colchones con n? ¿A partir de qué n es mayor el de cortar y pegar? ¿Qué método "
                  "sobreprotege las cadenas largas y cuál protege mejor las cortas?"]),
              _datos(df),
              md("# Resolución"),
              code(ARRANQUE + "\nR = proyecto.path_matrix(dummies=False)"),
              code(AYUDAS)]
    celdas += apartado(1)
    celdas.append(code("datos_proyecto['duration'] = datos_proyecto['agresiva']"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones agresivas y la cadena crítica", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Duración de cada ruta con las estimaciones agresivas"\n#| label: tbl-rutas\n'
                       "detalle_rutas(R, datos_proyecto['agresiva'][R.columns])"))
    celdas.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')\n"
                       "cadena, T = plan.cadena, plan.duracion_cadena\n"
                       "margenes = plan.margen[cadena].astype(int)\n"
                       "Markdown(f'Cadena crítica: **{\" - \".join(cadena)}**, {T} días, con {len(cadena)} actividades.')"))
    celdas.append(code('#| tbl-cap: "Margen retirado de las actividades de la cadena"\n#| label: tbl-margenes\n'
                       "tabla_margenes(datos_proyecto, cadena)"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Colchón del proyecto por los dos métodos"\n#| label: tbl-colchon\n'
                       "detalle_colchon(margenes, 'colchón del proyecto')"))
    celdas.append(code("CP_cp = ceil(0.5 * margenes.sum() - 1e-9)\n"
                       "CP_rz = ceil(sqrt((margenes ** 2).sum()) - 1e-9)\n"
                       "Markdown(f'Cortar y pegar: **{CP_cp} días**; raíz de la suma de cuadrados: **{CP_rz} días**.')"))
    celdas += apartado(3)
    celdas.append(md("Con duraciones normales, la de la cadena es normal con media la suma de las medias (la duración "
                     "agresiva de la cadena, $T$) y varianza la suma de las varianzas, $\\sigma_C^2 = \\sum_i "
                     "(m_i/1{,}2816)^2$. La probabilidad de cumplir $T + CP$ es $\\Phi(CP/\\sigma_C)$."))
    celdas.append(code("sigma = margenes / 1.2816\n"
                       "sigma_cadena = float(np.sqrt((sigma ** 2).sum()))\n"
                       "display(pd.DataFrame({'σ = margen/1,2816': sigma.round(3), 'σ²': (sigma ** 2).round(3)}).T)\n"
                       "Markdown(f'σ de la cadena = √{(sigma ** 2).sum():.3f} = **{sigma_cadena:.3f} días** '\n"
                       "         f'(que es la raíz de la suma de cuadrados de los márgenes dividida por 1,2816: {sqrt((margenes ** 2).sum()):.3f}/1,2816).')"))
    celdas.append(code('#| tbl-cap: "Probabilidad de terminar la cadena en T + CP"\n#| label: tbl-probabilidad\n'
                       "pd.DataFrame({'colchón': [CP_cp, CP_rz], 'fecha T + CP': [T + CP_cp, T + CP_rz],\n"
                       "              'z = CP/σ': [f'{CP_cp}/{sigma_cadena:.3f} = {CP_cp / sigma_cadena:.3f}', f'{CP_rz}/{sigma_cadena:.3f} = {CP_rz / sigma_cadena:.3f}'],\n"
                       "              'P = Φ(z)': [round(norm.cdf(CP_cp / sigma_cadena), 4), round(norm.cdf(CP_rz / sigma_cadena), 4)]},\n"
                       "             index=['cortar y pegar', 'raíz']).T"))
    celdas.append(md("Sin redondear, el colchón de la raíz es exactamente $1{,}2816\\,\\sigma_C$: da la probabilidad 0,90 "
                     "que se buscaba en cada actividad, ahora para la cadena completa (el redondeo al día superior la sube "
                     "algo). El de cortar y pegar no tiene un nivel de confianza fijo: depende del número de actividades. "
                     "El cálculo ignora las cadenas de alimentación, que se protegen con sus propios colchones."))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Colchones y probabilidad de cumplir para n actividades de margen medio m"\n#| label: tbl-n\n'
                       "m = float(margenes.mean())\n"
                       "n = np.arange(1, 13)\n"
                       "tabla_n = pd.DataFrame({'cortar y pegar = n·m/2': n * m / 2, 'raíz = √n·m': np.sqrt(n) * m,\n"
                       "                        'P con cortar y pegar = Φ(1,2816·√n/2)': norm.cdf(1.2816 * np.sqrt(n) / 2),\n"
                       "                        'P con raíz': norm.cdf(1.2816) + 0 * n}, index=pd.Index(n, name='n'))\n"
                       "tabla_n.round(3).T"))
    celdas.append(code("Markdown(f'Con m = {m:.2f} días, cortar y pegar crece linealmente (n·m/2) y la raíz como √n·m. Son iguales cuando '\n"
                       "         'n/2 = √n, es decir, n = 4: con más de cuatro actividades cortar y pegar da un colchón mayor, y su probabilidad '\n"
                       "         'de cumplir se acerca a 1 (sobreprotege las cadenas largas, a costa de alargar el compromiso); con menos de cuatro da un '\n"
                       "         f'colchón menor que la raíz y protege menos del 90 % (con n = 2, P = {norm.cdf(1.2816 * np.sqrt(2) / 2):.2f}). '\n"
                       "         f'Esta cadena tiene {len(cadena)} actividades: cortar y pegar le asigna {CP_cp - CP_rz} días más que la raíz.')"))
    celdas += _autoevaluacion(_p_dimensionado(plan), f"Cadena crítica — dimensionado del colchón (cadena de {len(plan.cadena)} actividades)")
    return _con_contexto(celdas, rng), (f"cadena de {len(plan.cadena)} actividades; colchón {plan.colchon_proyecto} (cortar y pegar) "
                                       f"frente a {math.ceil(plan.calculo_colchon_proyecto['raiz'] - 1e-9)} (raíz)")


# ---------------------------------------------------------------------------
# Variante 4 — Colchones de alimentación y programación lo más tarde posible
# ---------------------------------------------------------------------------
def _alimentacion(rng):
    def acepta(info, plan, df):
        a = plan.alimentacion
        if len(a) < 2 or len(a) > 4:
            return False
        largas = sum(1 for c in a["cadena_alimentacion"] if " - " in c)
        retrasadas = int((plan.inicios - plan.inicio_temprano).gt(0).sum())
        return largas >= 1 and retrasadas >= 2 and len(plan.cadena) >= 4
    info, plan, df = _buscar(rng, acepta)
    no_cabe = plan.alimentacion[plan.alimentacion["requerido"] > plan.alimentacion["disponible"]]
    pregs = ["Determine la cadena crítica con las estimaciones agresivas, su duración y el colchón del proyecto.",
             "Identifique los puntos en los que una actividad que no es de la cadena desemboca en ella. Para cada uno, "
             "indique la cadena de alimentación (la más larga que llega a ese punto), dimensione su colchón de "
             "alimentación y compruebe si cabe en la holgura disponible.",
             "Programe las actividades de alimentación lo más tarde posible, con su colchón por delante de la cadena "
             "crítica. Indique el inicio de cada actividad y dibuje el diagrama de Gantt con los colchones.",
             "¿Por qué se programan las cadenas de alimentación lo más tarde posible y no en su inicio más temprano? "
             "¿Qué se gana y qué se arriesga?"]
    if len(no_cabe):
        pregs[3] += (" Para el colchón que no cabe, compare las dos salidas posibles: reducirlo a la holgura disponible "
                     "o mantenerlo completo retrasando la cadena crítica.")
    celdas = [md("# Enunciado"),
              md("Un proyecto se programa con el método de la cadena crítica. " + ESTIMACIONES +
                 " Las actividades no comparten recursos. Todos los colchones se dimensionan por el método de cortar y "
                 "pegar (la mitad de la suma de los márgenes de la cadena que protegen) y se redondean al día superior. "
                 "La cadena crítica no se retrasa para hacer sitio a un colchón de alimentación: si no cabe, se reduce a "
                 "la holgura disponible."),
              preguntas(pregs), _datos(df),
              md("# Resolución"),
              code(ARRANQUE + "\nR = proyecto.path_matrix(dummies=False)"),
              code(AYUDAS)]
    celdas += apartado(1)
    celdas.append(code("datos_proyecto['duration'] = datos_proyecto['agresiva']"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones agresivas y la cadena crítica", label="fig-pert")
    celdas.append(_cal(cap="Calendario con las estimaciones agresivas (inicios más tempranos y holguras)"))
    celdas.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura', metodo='cortar_pegar')\n"
                       "cadena = plan.cadena\n"
                       "Markdown(f'Cadena crítica: **{\" - \".join(cadena)}** ({plan.duracion_cadena} días).')"))
    celdas.append(code('#| tbl-cap: "Colchón del proyecto"\n#| label: tbl-colchon\n'
                       "detalle_colchon(plan.margen[cadena], 'colchón del proyecto').loc[['cortar y pegar = 0,5·Σ márgenes']]"))
    celdas.append(code("Markdown(f'Colchón del proyecto: **{plan.colchon_proyecto} días**; fecha comprometida {plan.duracion_cadena} + {plan.colchon_proyecto} = **{plan.compromiso} días**.')"))
    celdas += apartado(2)
    celdas.append(md("Un colchón de alimentación protege a la cadena crítica de los retrasos de un camino que desemboca en "
                     "ella: se coloca en el arco por el que ese camino entra en la cadena. Se dimensiona con los márgenes "
                     "de la cadena de alimentación más larga que llega a ese arco (sólo con actividades que no son de la "
                     "cadena crítica). La holgura disponible es lo que puede separarse el fin de la última actividad de "
                     "alimentación, en su fecha más temprana, del inicio de la actividad de la cadena."))
    celdas.append(code('#| tbl-cap: "Colchones de alimentación"\n#| label: tbl-alimentacion\n'
                       "tabla_alimentacion(plan)"))
    if len(no_cabe):
        celdas.append(code("no_cabe = plan.alimentacion[plan.alimentacion['requerido'] > plan.alimentacion['disponible']]\n"
                           "Markdown('; '.join(f'{n} no cabe: requiere {int(f.requerido)} días y sólo hay {int(f.disponible)} de holgura, así que se reduce a {int(f.adoptado)}' "
                           "for n, f in no_cabe.iterrows()) + '.')"))
    celdas += apartado(3)
    celdas.append(md("La cadena crítica se programa en sus inicios más tempranos. Las actividades de alimentación se "
                     "programan hacia atrás, de la última a la primera: cada una termina cuando empieza la primera de sus "
                     "sucesoras o, si la sucesora es de la cadena, cuando empieza su colchón de alimentación."))
    celdas.append(code('#| tbl-cap: "Programación lo más tarde posible de las actividades de alimentación"\n#| label: tbl-alap\n'
                       "detalle_alap(plan)"))
    celdas.append(code('#| tbl-cap: "Plan de cadena crítica"\n#| label: tbl-plan\n'
                       "plan.tabla()[['agresiva', 'cadena', 'inicio', 'fin']].assign(**{'inicio más temprano': plan.inicio_temprano}).T"))
    celdas.append(code('#| fig-cap: "Diagrama de Gantt del plan de cadena crítica: cadena en rojo, alimentación en verde, colchones en naranja"\n#| label: fig-ganttCCPM\n'
                       f"pinta_tikz(plan.gantt_tikz(params={GANTT_CCPM}))"))
    celdas += apartado(4)
    celdas.append(md("Programar la alimentación lo más tarde posible evita comprometer recursos y dinero antes de tiempo "
                     "(el gasto se retrasa, el trabajo en curso es menor y se pueden incorporar cambios con menos "
                     "pérdida) y deja el margen donde protege: en el colchón, a la vista de todos, en lugar de en una "
                     "holgura que se consumiría por el síndrome del estudiante. A cambio, el camino de alimentación queda "
                     "sin más protección que su colchón: si se retrasa más que él, el retraso pasa a la cadena crítica y "
                     "consume colchón del proyecto. Por eso se vigila el consumo de cada colchón de alimentación."))
    if len(no_cabe):
        celdas.append(code("exceso = int((no_cabe['requerido'] - no_cabe['disponible']).max())\n"
                           "Markdown(f'Si el colchón que no cabe se mantuviera completo habría que retrasar la cadena crítica {exceso} día(s): '\n"
                           "         f'la fecha comprometida pasaría de {plan.compromiso} a {plan.compromiso + exceso}. Reducirlo a la holgura '\n"
                           "         'mantiene el compromiso, pero deja ese camino peor protegido: su cadena de alimentación debe empezar el primer '\n"
                           "         'día posible y conviene vigilarla de cerca.')"))
    celdas += _autoevaluacion(_p_alimentacion(plan), f"Cadena crítica — colchones de alimentación (compromiso el día {plan.compromiso})")
    sub = (f"{len(plan.alimentacion)} colchones de alimentación" + (f" ({len(no_cabe)} no cabe)" if len(no_cabe) else "")
           + f"; compromiso {plan.compromiso} días")
    return _con_contexto(celdas, rng), sub


# ---------------------------------------------------------------------------
# Variante 5 — Cadena crítica con recursos
# ---------------------------------------------------------------------------
PRIORIDAD = ("Los conflictos de recursos se resuelven programando día a día: cuando varias actividades del mismo "
             "equipo pueden empezar, tiene prioridad la de menor inicio más tardío en el cronograma sin recursos "
             "(calculado con las estimaciones agresivas); a igualdad, la de menor duración y, después, la primera "
             "por orden alfabético.")

RECURSOS_INI = ("equipos = sorted(set(datos_proyecto['recurso']) - {'---'})\n"
                "datos_proyecto['equipo'] = datos_proyecto['recurso'].replace('---', '')\n"
                "datos_proyecto['duration'] = datos_proyecto['agresiva']")

CONFLICTOS = r'''
def solapes(calendario, datos):
    """Pares de actividades del mismo equipo que se solapan en el cronograma sin recursos."""
    filas = []
    for equipo, grupo in datos[datos['recurso'] != '---'].groupby('recurso'):
        acts = sorted(grupo.index)
        for i, a in enumerate(acts):
            for b in acts[i + 1:]:
                ia = (int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'fin_mas_temprano']))
                ib = (int(calendario.loc[b, 'inicio_mas_temprano']), int(calendario.loc[b, 'fin_mas_temprano']))
                solape = min(ia[1], ib[1]) - max(ia[0], ib[0])
                if solape > 0:
                    filas.append({'equipo': equipo, 'actividades': f'{a} y {b}',
                                  'intervalos': f'{a} [{ia[0]}, {ia[1]}) y {b} [{ib[0]}, {ib[1]})', 'días de solape': solape})
    return pd.DataFrame(filas, columns=['equipo', 'actividades', 'intervalos', 'días de solape']).rename(index=lambda i: i + 1)


def esperas_por_recurso(asignacion):
    """Actividades que esperan a un equipo: desde cuándo están listas, cuándo empiezan y quién las bloquea."""
    t = asignacion.traza
    esp = t[t['decision'] == 'espera']
    filas = {}
    for a, g in esp.groupby('actividad'):
        filas[a] = {'lista desde el día': int(g['periodo'].min()) - 1, 'empieza el día': int(asignacion.inicios[a]),
                    'espera (días)': int(asignacion.inicios[a]) - int(g['periodo'].min()) + 1,
                    'equipo que la bloquea': ', '.join(sorted(set(g['bloqueada_por'])))}
    return pd.DataFrame(filas).T
'''.strip("\n")


def _nivelacion_celdas(ap_conflictos, ap_nivelacion):
    """Celdas comunes de las variantes con recursos: conflictos sin recursos y red nivelada."""
    c = []
    c += apartado(ap_conflictos)
    c += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones agresivas, sin tener en cuenta los equipos", label="fig-pert")
    c.append(_cal(cap="Calendario sin tener en cuenta los equipos (estimaciones agresivas)"))
    c.append(code('#| fig-cap: "Gantt de inicios más tempranos sin tener en cuenta los equipos (en cada barra, el equipo)"\n#| label: fig-ganttLibre\n'
                  "_, dibujo = proyecto.gantt(datos_proyecto, 'duration', 'equipo', tikz=True, params=GANTT_CC)\n"
                  "pinta_tikz(dibujo)"))
    c.append(code('#| tbl-cap: "Actividades de un mismo equipo que se solapan"\n#| label: tbl-solapes\n'
                  "solapes(calendario, datos_proyecto)"))
    c.append(code("T_libre = int(calendario['fin_mas_temprano'].max())\n"
                  "ruta_libre = [a for a in calendario.index if calendario.loc[a, 'H_total'] == 0]\n"
                  "Markdown(f'Sin tener en cuenta los equipos el proyecto dura {T_libre} días y su ruta crítica es {\" - \".join(sorted(ruta_libre, key=lambda a: calendario.loc[a, \"inicio_mas_temprano\"]))}; '\n"
                  "         'pero el cronograma no es realizable: un equipo no puede hacer dos actividades a la vez.')"))
    if ap_nivelacion != ap_conflictos:
        c += apartado(ap_nivelacion)
    c.append(md("Se programa día a día con un equipo de cada tipo (límite 1). La prioridad usa el inicio más tardío del "
                "calendario anterior. La tabla recoge las actividades que tienen que esperar a su equipo."))
    c.append(code("demanda = pd.DataFrame({e: (datos_proyecto['recurso'] == e).astype(int) for e in equipos})\n"
                  "demanda['duration'] = datos_proyecto['agresiva']\n"
                  "asignacion = proyecto.asignar_multi(demanda, 'duration', equipos, [1] * len(equipos), dibujar=False)\n"
                  "Markdown(f'Duración con los equipos: **{asignacion.duracion} días** (sin ellos, {T_libre}).')"))
    c.append(code('#| tbl-cap: "Esperas por falta de equipo"\n#| label: tbl-esperas\n'
                  "esperas_por_recurso(asignacion)"))
    c.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura', recurso='recurso', metodo=METODO)\n"
                  "pd.DataFrame({'orden en que atiende a sus actividades': {e: ' → '.join(plan.secuencias[e]) for e in equipos}})"))
    c.append(code('#| tbl-cap: "Enlaces de recurso que se añaden a la red"\n#| label: tbl-enlaces\n'
                  "plan.enlaces_recurso.rename(index=lambda i: i + 1)"))
    c.append(md("Cada enlace de recurso obliga a que una actividad empiece después de que termine la anterior del mismo "
                "equipo. Con ellos la red queda *nivelada*: cualquier programación que respete sus prelaciones respeta "
                "los equipos. (Los enlaces que ya están implícitos en las prelaciones no se añaden.)"))
    c.append(code("red = plan.red_nivelada()\n"
                  "proyecto_nivelado = PredecessorTable.from_dataframe_of_strings(red, activity='activity', predecessor='predecessor').create_project()\n"
                  "red[['predecessor']].T"))
    c += fig_grafo(durations="red['agresiva']", obj="proyecto_nivelado",
                   cap="Red nivelada (prelaciones y enlaces de recurso) con las estimaciones agresivas: su camino crítico es la cadena crítica",
                   label="fig-pertNivelado")
    c.append(code('#| fig-cap: "Gantt de la red nivelada (inicios más tempranos)"\n#| label: fig-ganttNivelado\n'
                  "_, dibujo = proyecto_nivelado.gantt(datos_proyecto.loc[red.index], 'duration', 'equipo', tikz=True, params=GANTT_CC)\n"
                  "pinta_tikz(dibujo)"))
    return c


def _recursos_acepta(min_alim=0):
    def acepta(info, plan, df):
        if not (plan.enlaces_cadena["motivo"] == "recurso").any():
            return False
        if info["n_criticas"] != 1 or plan.duracion_cadena <= info["T"]:
            return False
        ruta = set(info["P"].columns[info["P"].loc[info["rutas_criticas"][0]] == 1])
        if ruta == set(plan.cadena):
            return False
        return len(plan.alimentacion) >= min_alim and len(plan.cadena) >= 4 and len(plan.colchones_recursos) >= 1
    return acepta


def _con_recursos(rng):
    n_rec = int(rng.choice([1, 2]))
    info, plan, df = _buscar(rng, _recursos_acepta(), n_rec=n_rec)
    celdas = [md("# Enunciado"),
              md("Un proyecto se programa con el método de la cadena crítica. " + ESTIMACIONES +
                 " Algunas actividades las ejecuta el mismo equipo (columna *recurso*); cada equipo sólo puede atender "
                 "una actividad a la vez y, una vez empezada, no la interrumpe. " + PRIORIDAD +
                 " Los colchones se dimensionan por el método de cortar y pegar y se redondean al día superior."),
              preguntas([
                  "Sin tener en cuenta los equipos, calcule con las estimaciones agresivas la duración del proyecto y su "
                  "ruta crítica. ¿Qué actividades de un mismo equipo se solapan?",
                  "Resuelva los conflictos con la regla de prioridad indicada. ¿En qué orden atiende cada equipo a sus "
                  "actividades? Añada esos enlaces de recurso a la red y calcule la duración del proyecto con los equipos.",
                  "Identifique la cadena crítica: sus actividades y si cada enlace entre ellas es de precedencia o de "
                  "recurso. ¿Coincide con la ruta crítica del apartado 1? ¿Por qué?",
                  "Dimensione el colchón del proyecto y dé la fecha comprometida. ¿Antes de qué actividades de la cadena "
                  "colocaría colchones de recursos y en qué consisten?"]),
              _datos(df),
              md("# Resolución"),
              code(ARRANQUE + "\nMETODO = 'cortar_pegar'\n" + RECURSOS_INI),
              code(AYUDAS + "\n\n\n" + CONFLICTOS)]
    celdas += _nivelacion_celdas(1, 2)
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Enlaces de la cadena crítica"\n#| label: tbl-cadena\n'
                       "plan.enlaces_cadena.rename(index=lambda i: i + 1)"))
    celdas.append(code("Markdown(f'Cadena crítica: **{\" - \".join(plan.cadena)}**, de {plan.duracion_cadena} días '\n"
                       "         f'({\" + \".join(str(int(plan.agresiva[a])) for a in plan.cadena)}). No coincide con la ruta crítica sin recursos '\n"
                       "         f'({\" - \".join(sorted(ruta_libre, key=lambda a: calendario.loc[a, \"inicio_mas_temprano\"]))}, {T_libre} días): '\n"
                       "         'la secuencia más larga pasa por un enlace de recurso, una dependencia que no está en el cuadro de prelaciones '\n"
                       "         'pero que existe porque el mismo equipo no puede hacer dos cosas a la vez. Por eso la cadena crítica se busca '\n"
                       "         'después de resolver los conflictos de recursos.')"))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Colchón del proyecto"\n#| label: tbl-colchon\n'
                       "detalle_colchon(plan.margen[plan.cadena], 'colchón del proyecto').loc[['cortar y pegar = 0,5·Σ márgenes']]"))
    celdas.append(code("Markdown(f'Colchón del proyecto: **{plan.colchon_proyecto} días**; fecha comprometida {plan.duracion_cadena} + '\n"
                       "         f'{plan.colchon_proyecto} = **{plan.compromiso} días**.')"))
    celdas.append(code('#| tbl-cap: "Colchones de recursos (avisos)"\n#| label: tbl-avisos\n'
                       "plan.colchones_recursos.rename(columns={'antes_de': 'antes de', 'recursos': 'equipo al que se avisa', "
                       "'despues_de': 'mientras termina', 'inicio_previsto': 'inicio previsto'}).rename(index=lambda i: i + 1)"))
    celdas.append(md("Un colchón de recursos no es tiempo en el cronograma: es un **aviso** al equipo que va a entrar en la "
                     "cadena crítica, unos días antes de que termine la actividad anterior, para que esté libre y "
                     "preparado en cuanto reciba el testigo. Se pone antes de cada actividad de la cadena que usa un equipo "
                     "que no usaba su predecesora en la cadena; si la cadena sigue con el mismo equipo, no hace falta."))
    celdas += _autoevaluacion(_p_recursos(plan, info), f"Cadena crítica — cadena con recursos (cadena de {plan.duracion_cadena} días)")
    sub = (f"{n_rec} equipo{'s' if n_rec > 1 else ''}; la cadena pasa por {int((plan.enlaces_cadena['motivo'] == 'recurso').sum())} "
           f"enlace(s) de recurso; {plan.duracion_cadena} frente a {int(info['T'])} días sin recursos")
    return _con_contexto(celdas, rng), sub


# ---------------------------------------------------------------------------
# Variante 6 — Plan completo: recursos, colchones y comparación con el cronograma tradicional
# ---------------------------------------------------------------------------
def _plan_completo(rng):
    n_rec = int(rng.choice([1, 2]))
    metodo = str(rng.choice(["cortar_pegar", "raiz"]))
    info, plan, df = _buscar(rng, _recursos_acepta(min_alim=1), n_rec=n_rec, metodo=metodo)
    nombre_metodo = "cortar y pegar" if metodo == "cortar_pegar" else "la raíz de la suma de cuadrados"
    fila_metodo = "cortar y pegar = 0,5·Σ márgenes" if metodo == "cortar_pegar" else "raíz = √(Σ márgenes²)"
    celdas = [md("# Enunciado"),
              md("Se quiere preparar el plan de cadena crítica de un proyecto. " + ESTIMACIONES +
                 " Algunas actividades las ejecuta el mismo equipo (columna *recurso*); cada equipo sólo puede atender "
                 "una actividad a la vez y no la interrumpe. " + PRIORIDAD +
                 f" Todos los colchones se dimensionan por el método de {nombre_metodo} y se redondean al día superior; "
                 "si un colchón de alimentación no cabe en la holgura disponible, se reduce a ella (la cadena crítica no "
                 "se retrasa)."),
              preguntas([
                  "Resuelva los conflictos de recursos con la regla de prioridad, obtenga la red nivelada (prelaciones "
                  "más enlaces de recurso) y la cadena crítica con las estimaciones agresivas.",
                  "Dimensione el colchón del proyecto y los colchones de alimentación.",
                  "Programe el proyecto (la cadena lo antes posible y las actividades de alimentación lo más tarde "
                  "posible, con su colchón delante) y dibuje el diagrama de Gantt con los colchones. ¿Qué fecha se "
                  "compromete con el cliente?",
                  "Compare con el cronograma tradicional: estimaciones seguras con la misma secuencia de los equipos. "
                  "¿Cuántos días se ganan con el plan de cadena crítica?",
                  "¿Antes de qué actividades de la cadena hay que avisar a un equipo (colchones de recursos)? ¿Por qué el "
                  "aviso no ocupa tiempo en el cronograma?"]),
              _datos(df),
              md("# Resolución"),
              code(ARRANQUE + f"\nMETODO = {metodo!r}\n" + RECURSOS_INI),
              code(AYUDAS + "\n\n\n" + CONFLICTOS)]
    celdas += _nivelacion_celdas(1, 1)
    celdas.append(code('#| tbl-cap: "Enlaces de la cadena crítica"\n#| label: tbl-cadena\n'
                       "plan.enlaces_cadena.rename(index=lambda i: i + 1)"))
    celdas.append(code("Markdown(f'Cadena crítica: **{\" - \".join(plan.cadena)}**, {plan.duracion_cadena} días.')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Colchón del proyecto"\n#| label: tbl-colchon\n'
                       f"detalle_colchon(plan.margen[plan.cadena], 'colchón del proyecto').loc[[{fila_metodo!r}]]"))
    celdas.append(code('#| tbl-cap: "Colchones de alimentación"\n#| label: tbl-alimentacion\n'
                       "tabla_alimentacion(plan)"))
    celdas.append(code("Markdown(f'Colchón del proyecto: **{plan.colchon_proyecto} días**. Colchones de alimentación: '\n"
                       "         + '; '.join(f'{n} = {int(f.adoptado)} día(s) antes de {f.antes_de}' for n, f in plan.alimentacion.iterrows()) + '.')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Programación lo más tarde posible de las actividades de alimentación"\n#| label: tbl-alap\n'
                       "detalle_alap(plan)"))
    celdas.append(code('#| tbl-cap: "Plan de cadena crítica"\n#| label: tbl-plan\n'
                       "plan.tabla().T"))
    celdas.append(code('#| fig-cap: "Gantt del plan de cadena crítica: cadena en rojo, alimentación en verde, colchones en naranja; enlaces de recurso discontinuos"\n#| label: fig-ganttCCPM\n'
                       f"pinta_tikz(plan.gantt_tikz(params={GANTT_CCPM}))"))
    celdas.append(code("Markdown(f'Fecha comprometida: {plan.duracion_cadena} + {plan.colchon_proyecto} = **{plan.compromiso} días**.')"))
    celdas += apartado(4)
    celdas.append(code("datos_proyecto['duration'] = datos_proyecto['segura']\n"
                       "calendario_tradicional = proyecto_nivelado.calendar(datos_proyecto.loc[red.index, 'segura'])\n"
                       "ruta_tradicional = [a for a in calendario_tradicional.sort_values('inicio_mas_temprano').index if calendario_tradicional.loc[a, 'H_total'] == 0]\n"
                       "Markdown(f'Con las estimaciones seguras en la red nivelada el proyecto dura **{plan.duracion_tradicional} días** '\n"
                       "         f'(ruta crítica {\" - \".join(ruta_tradicional)}). El plan de cadena crítica compromete {plan.compromiso}: '\n"
                       "         f'**{plan.duracion_tradicional - plan.compromiso} días menos**. El margen que el cronograma tradicional reparte entre las '\n"
                       "         f'actividades ({int(plan.margen.sum())} días en total) se sustituye por {plan.colchon_proyecto} días de colchón del proyecto '\n"
                       "         f'y {int(plan.alimentacion[\"adoptado\"].sum())} de colchones de alimentación, situados donde protegen la entrega.')"))
    celdas += apartado(5)
    celdas.append(code('#| tbl-cap: "Colchones de recursos (avisos)"\n#| label: tbl-avisos\n'
                       "plan.colchones_recursos.rename(columns={'antes_de': 'antes de', 'recursos': 'equipo al que se avisa', "
                       "'despues_de': 'mientras termina', 'inicio_previsto': 'inicio previsto'}).rename(index=lambda i: i + 1)"))
    celdas.append(md("El colchón de recursos es un aviso, no una espera: el equipo se compromete a estar libre cuando termine "
                     "la actividad anterior de la cadena (que, por la regla del corredor de relevos, puede terminar antes de "
                     "lo previsto). Añadir tiempo alargaría la cadena sin protegerla; lo que se protege es la disponibilidad "
                     "del recurso, que es lo que haría perder días a la cadena."))
    celdas += _autoevaluacion(_p_plan_completo(plan), f"Cadena crítica — plan completo (compromiso el día {plan.compromiso})")
    sub = (f"{n_rec} equipo{'s' if n_rec > 1 else ''}; colchones por {'cortar y pegar' if metodo == 'cortar_pegar' else 'raíz'}; "
           f"compromiso {plan.compromiso} frente a {plan.duracion_tradicional} días")
    return _con_contexto(celdas, rng), sub


# ---------------------------------------------------------------------------
# Variante 7 — Seguimiento: consumo del colchón, BPI y gráfico de fiebre
# ---------------------------------------------------------------------------
def _ejecucion(plan, rng):
    """Una ejecución (duraciones reales enteras, lognormales) por relevos: (inicios, fines) reales."""
    real = {}
    for a in sorted(plan.agresiva.index):
        med, p90 = float(plan.agresiva[a]), float(plan.segura[a])
        sigma = math.log(p90 / med) / Z90 if p90 > med else 0.0
        real[a] = min(max(1, int(round(med * math.exp(sigma * rng.standard_normal())))), math.ceil(1.5 * p90))
    ini, fin = {}, {}
    for a in plan.orden:
        st = max([0] + [fin[p] for p in plan.predecesoras[a]])
        if a not in plan.cadena:
            st = max(st, int(plan.inicios[a]))
        ini[a], fin[a] = st, st + real[a]
    return real, ini, fin


def _informe(plan, ini, fin, dia):
    terminadas = {a: fin[a] for a in plan.orden if fin[a] <= dia}
    en_curso = {a: fin[a] - dia for a in plan.orden if ini[a] < dia < fin[a]}
    return dict(dia=dia, terminadas=terminadas, en_curso=en_curso)


def _seguimiento(rng):
    def acepta(info, plan, df):
        return 4 <= len(plan.cadena) <= 6 and plan.colchon_proyecto >= 4 and len(plan.alimentacion) >= 1
    encontrado = False
    for _ in range(200):
        info, plan, df = _buscar(rng, acepta)
        for _ in range(60):
            real, ini, fin = _ejecucion(plan, rng)
            T_real = max(fin.values())
            dias = []
            for frac in (0.3, 0.6, 0.85):
                objetivo = int(round(frac * T_real))
                validos = [t for t in range(1, T_real) if any(ini[c] < t < fin[c] for c in plan.cadena)]
                if not validos:
                    break
                dias.append(min(validos, key=lambda t: (abs(t - objetivo), t)))
            if len(set(dias)) < 3:
                continue
            informes = [_informe(plan, ini, fin, t) for t in dias]
            segs = [plan.seguimiento(i["dia"], i["terminadas"], i["en_curso"]) for i in informes]
            ok = True
            for i, s in zip(informes, segs):
                det = s["detalle"]
                simple = i["dia"] + float(det.loc[det["estado"] != "terminada", "restante"].sum())
                if abs(simple - s["fin_previsto"]) > 1e-9 or s["consumo"] < 0:
                    ok = False
            zonas = [s["zona"] for s in segs]
            if not ok or len(set(zonas)) < 2 or zonas[0] == "roja":
                continue
            if max(s["pct_consumo"] for s in segs) > 160:
                continue
            encontrado = True
            break
        if encontrado:
            break
    if not encontrado:
        raise RuntimeError("No se encontró una ejecución con informes en zonas distintas")
    df = df.assign(inicio_plan=plan.inicios.reindex(df.index).astype(int))
    texto_inf = ["informes = ["]
    for i in informes:
        texto_inf.append(f"    dict(dia={i['dia']}, terminadas={i['terminadas']!r}, en_curso={i['en_curso']!r}),")
    texto_inf.append("]")
    texto_inf.append("tabla_informes = pd.DataFrame({f\"día {i['dia']}\": {a: (f\"terminó el día {i['terminadas'][a]}\" if a in i['terminadas'] else "
                     "(f\"en curso, le quedan {i['en_curso'][a]}\" if a in i['en_curso'] else '—')) for a in datos_enunciado.index} for i in informes})")
    texto_inf.append("tabla_informes")
    cad = " – ".join(plan.cadena)
    celdas = [md("# Enunciado"),
              md("Un proyecto se ha programado con el método de la cadena crítica (estimaciones agresivas, colchones por "
                 "cortar y pegar redondeados al día superior, actividades de alimentación lo más tarde posible). La cadena "
                 f"crítica es {cad}, de {plan.duracion_cadena} días; el colchón del proyecto es de {plan.colchon_proyecto} días "
                 f"y la fecha comprometida, el día {plan.compromiso}. El cuadro recoge las estimaciones y el inicio de cada "
                 "actividad en el plan (días contados desde el inicio del proyecto, que es el instante 0)."),
              _datos(df),
              md("Durante la ejecución se reciben tres informes de avance. En cada uno se indica qué actividades han "
                 "terminado (y en qué día), cuántos días estima el responsable que le quedan a cada actividad en curso, y "
                 "qué actividades no han empezado. Las actividades de la cadena empiezan en cuanto terminan sus "
                 "predecesoras (corredor de relevos)."),
              code('#| tbl-cap: "Informes de avance"\n#| label: tbl-informes\n' + "\n".join(texto_inf)),
              preguntas([
                  "Para cada informe, calcule la fecha prevista de fin de la cadena y el colchón consumido, en días y en "
                  "porcentaje del colchón del proyecto.",
                  "Calcule el porcentaje de cadena crítica completada en cada informe (duración de la cadena menos su "
                  "duración restante, entre la duración de la cadena; de una actividad en curso no se cuenta como "
                  "restante más que su duración agresiva) y el índice de protección del colchón, BPI "
                  "(porcentaje de colchón restante entre porcentaje de cadena restante).",
                  "Sitúe los informes en el gráfico de fiebre (zona verde por debajo de la recta que va del 10 % de consumo "
                  "con la cadena sin empezar al 70 % con la cadena terminada; zona roja por encima de la recta que va del "
                  "30 % al 90 %; amarilla entre ambas) e indique la zona de cada uno.",
                  "¿Qué debe hacer el jefe de proyecto tras cada informe? ¿Por qué se pide a los responsables la duración "
                  "restante y no se controla si cada actividad cumple su fecha de fin?"]),
              md("# Resolución"),
              code(ARRANQUE + "\ndatos_proyecto['duration'] = datos_proyecto['agresiva']"),
              code(AYUDAS)]
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones agresivas y la cadena crítica", label="fig-pert")
    celdas.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')\n"
                       "assert plan.cadena == " + repr(plan.cadena) + " and plan.colchon_proyecto == " + str(plan.colchon_proyecto) + "\n"
                       "T, CP = plan.duracion_cadena, plan.colchon_proyecto\n"
                       "Markdown(f'Plan: cadena {\" - \".join(plan.cadena)} ({T} días), colchón del proyecto {CP} días, compromiso el día {plan.compromiso}.')"))
    celdas.append(code('#| fig-cap: "Gantt del plan de cadena crítica"\n#| label: fig-ganttCCPM\n'
                       f"pinta_tikz(plan.gantt_tikz(params={GANTT_CCPM}))"))
    celdas += apartado(1)
    celdas.append(md("La fecha prevista de fin de la cadena es la del informe más lo que le queda a la cadena: la duración "
                     "restante estimada de la actividad en curso más las duraciones agresivas de las que no han empezado. "
                     "Todo lo que pase de la duración de la cadena es colchón consumido."))
    celdas.append(code("seguimientos = [plan.seguimiento(i['dia'], i['terminadas'], i['en_curso']) for i in informes]\n"
                       "for s in seguimientos:\n"
                       "    display(Markdown(f\"**Día {s['dia']}** — estado de la cadena crítica:\"))\n"
                       "    display(sin_decimales(s['detalle'][['estado', 'agresiva', 'completado', 'restante', 'fin previsto']]))\n"
                       "resumen = resumen_informes(plan, seguimientos)"))
    celdas.append(code('#| tbl-cap: "Fecha prevista de fin y colchón consumido"\n#| label: tbl-consumo\n'
                       "resumen.loc[['fin previsto = día + restante de la cadena', 'colchón consumido']]"))
    celdas += apartado(2)
    celdas.append(md("Lo completado de la cadena es su duración menos lo que le queda: cuenta entera cada actividad "
                     "terminada (por su duración agresiva, aunque haya tardado más o menos) y, de la que está en curso, "
                     "su duración agresiva menos lo que le queda (cero si le queda más que eso)."))
    celdas.append(code('#| tbl-cap: "Cadena completada y BPI"\n#| label: tbl-bpi\n'
                       "resumen.loc[['cadena completada', 'BPI = % colchón restante / % cadena restante']]"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Límites de las zonas para el avance de cada informe y zona"\n#| label: tbl-zonas\n'
                       "resumen.loc[['límite verde = 10 + 0,6·avance', 'límite rojo = 30 + 0,6·avance', 'zona']]"))
    celdas.append(code('#| fig-cap: "Gráfico de fiebre con los tres informes"\n#| label: fig-fiebre\n'
                       "pinta_tikz(grafico_fiebre_tikz(seguimientos))"))
    celdas += apartado(4)
    celdas.append(code("acciones = {'verde': 'no hace falta actuar: el colchón se consume a un ritmo menor que el avance de la cadena',\n"
                       "            'amarilla': 'preparar un plan de recuperación (refuerzos, solapes, reasignar a quien termine), sin ejecutarlo todavía',\n"
                       "            'roja': 'ejecutar el plan de recuperación: si no, la fecha comprometida está en peligro'}\n"
                       "Markdown('\\n'.join(f\"- Día {s['dia']} (zona {s['zona']}, BPI {s['bpi']:.2f}): {acciones[s['zona']]}.\" for s in seguimientos))"))
    celdas.append(md("Con estimaciones agresivas, la mitad de las actividades terminarán después de su duración prevista: "
                     "controlar el cumplimiento de cada fecha de fin generaría alarmas continuas y presiones para volver a "
                     "inflar las estimaciones. Lo que importa es si queda colchón suficiente para lo que queda de cadena, y "
                     "para eso hace falta la **duración restante** de lo que está en curso: con ella se calcula el fin "
                     "previsto, el consumo del colchón y el BPI, una única señal que todo el equipo entiende."))
    celdas += _autoevaluacion(_p_seguimiento(plan, segs), f"Cadena crítica — seguimiento (cadena de {plan.duracion_cadena} días, CP de {plan.colchon_proyecto})")
    zonas = " → ".join(s["zona"] for s in segs)
    return _con_contexto(celdas, rng), f"cadena de {len(plan.cadena)} actividades, CP de {plan.colchon_proyecto} días; informes en zona {zonas}"


# ---------------------------------------------------------------------------
# Variante 8 — ¿Es suficiente el colchón? Monte Carlo del plan
# ---------------------------------------------------------------------------
SIMULACION = ("las duraciones son independientes y lognormales, con mediana la estimación agresiva y percentil 90 la "
              "segura (σ del logaritmo = ln(segura/agresiva)/1,2816; constante si coinciden)")


def _monte_carlo(rng):
    n_rec = int(rng.choice([0, 1]))
    acepta = (lambda info, plan, df: len(plan.alimentacion) >= 2 and len(plan.cadena) >= 4) if n_rec == 0 else \
        (lambda info, plan, df: len(plan.alimentacion) >= 1 and (plan.enlaces_cadena["motivo"] == "recurso").any())
    info, plan, df = _buscar(rng, acepta, n_rec=n_rec)
    N, semilla = 20000, int(rng.integers(1, 10000))
    rec = n_rec > 0
    texto = ("Un proyecto se programa con el método de la cadena crítica. " + ESTIMACIONES +
             (" Las actividades marcadas en la columna *recurso* las ejecuta un mismo equipo, que sólo puede atender "
              "una a la vez. " + PRIORIDAD
              if rec else " Las actividades no comparten recursos.") +
             " Los colchones se dimensionan por cortar y pegar, redondeados al día superior.")
    celdas = [md("# Enunciado"), md(texto),
              preguntas([
                  "Obtenga el plan de cadena crítica: cadena, colchón del proyecto, colchones de alimentación y fecha "
                  "comprometida, y la duración del cronograma tradicional (estimaciones seguras" +
                  (", con la misma secuencia del equipo)." if rec else ")."),
                  f"Simule {N} ejecuciones (semilla {semilla}) en las que {SIMULACION}; las actividades de la cadena "
                  "empiezan en cuanto terminan sus predecesoras y las de alimentación no antes de su inicio del plan. "
                  "Calcule la probabilidad de terminar en la duración de la cadena, en la fecha comprometida y en la "
                  "duración del cronograma tradicional.",
                  "¿Qué colchón del proyecto haría falta para cumplir con probabilidad 0,90? ¿Y 0,95? Compare con los "
                  "colchones de cortar y pegar y de la raíz de la suma de cuadrados.",
                  "Simule también la cadena crítica aislada (la suma de las duraciones de sus actividades). ¿Qué "
                  "probabilidad tendría de cumplir la fecha comprometida? Explique la diferencia con el proyecto completo. "
                  "¿Estaba sobreprotegido el cronograma tradicional?"]),
              _datos(df),
              md("# Resolución"),
              code(ARRANQUE + "\ndatos_proyecto['duration'] = datos_proyecto['agresiva']\n"
                   + (f"RECURSO = 'recurso'\n" if rec else "RECURSO = None\n") + f"N, semilla = {N}, {semilla}"),
              code(AYUDAS)]
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones agresivas" +
                        (" (sin los enlaces de recurso)" if rec else " y la cadena crítica"), label="fig-pert")
    celdas.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura', recurso=RECURSO)\n"
                       "cadena, T, CP = plan.cadena, plan.duracion_cadena, plan.colchon_proyecto\n"
                       "plan.enlaces_cadena.rename(index=lambda i: i + 1)"))
    celdas.append(code('#| tbl-cap: "Colchón del proyecto"\n#| label: tbl-colchon\n'
                       "detalle_colchon(plan.margen[cadena], 'colchón del proyecto')"))
    celdas.append(code('#| tbl-cap: "Colchones de alimentación"\n#| label: tbl-alimentacion\n'
                       "tabla_alimentacion(plan)"))
    celdas.append(code('#| fig-cap: "Gantt del plan de cadena crítica"\n#| label: fig-ganttCCPM\n'
                       f"pinta_tikz(plan.gantt_tikz(params={GANTT_CCPM}))"))
    celdas.append(code("Markdown(f'Cadena {\" - \".join(cadena)} de {T} días; colchón del proyecto {CP}; compromiso **{plan.compromiso} días**. '\n"
                       "         f'Cronograma tradicional (estimaciones seguras): **{plan.duracion_tradicional} días**.')"))
    celdas += apartado(2)
    celdas.append(md("En cada réplica la duración de cada actividad es $a\\,e^{\\sigma Z}$, con $a$ la estimación "
                     "agresiva, $Z$ normal tipificada y $\\sigma = \\ln(s/a)/1{,}2816$, de modo que su mediana es $a$ y su "
                     "percentil 90 es $s$, la estimación segura. Con ellas se recorre la red (nivelada) hacia delante."))
    celdas.append(code("T_sim = plan.simular(N, semilla=semilla, regimen='relevos')\n"
                       "fechas = {'duración de la cadena': T, 'fecha comprometida': plan.compromiso, 'cronograma tradicional': plan.duracion_tradicional}\n"
                       "sin_decimales(pd.DataFrame({'días': fechas, 'P(duración ≤ días)': {k: round(float((T_sim <= v).mean()), 4) for k, v in fechas.items()}}).T)"))
    celdas.append(code('#| fig-cap: "Distribución simulada de la duración del proyecto"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
                       "ax.hist(T_sim, bins=50, density=True, color='lightsteelblue', edgecolor='white')\n"
                       "for (nombre, v), color in zip(fechas.items(), ['grey', 'tab:orange', 'tab:red']):\n"
                       "    ax.axvline(v, color=color, linestyle='--', label=f'{nombre} ({v})')\n"
                       "ax.set_xlabel('duración (días)'); ax.set_ylabel('densidad'); ax.legend(); plt.show()"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Colchón necesario para cada nivel de confianza"\n#| label: tbl-necesario\n'
                       "percentiles = {p: float(np.percentile(T_sim, 100 * p)) for p in (0.5, 0.8, 0.9, 0.95)}\n"
                       "pd.DataFrame({f'P{int(100 * p)}': {'duración': round(v, 2), 'colchón = ⌈percentil − T⌉': f'⌈{v:.2f} − {T}⌉ = {ceil(v - T - 1e-9)}'}\n"
                       "              for p, v in percentiles.items()})"))
    celdas.append(code("CP_rz = ceil(plan.calculo_colchon_proyecto['raiz'] - 1e-9)\n"
                       "Markdown(f\"Para el 90 % harían falta {ceil(percentiles[0.9] - T - 1e-9)} días de colchón y para el 95 %, {ceil(percentiles[0.95] - T - 1e-9)}. \"\n"
                       "         f\"Cortar y pegar da {CP} (probabilidad {float((T_sim <= T + CP).mean()):.3f}) y la raíz de la suma de cuadrados {CP_rz} \"\n"
                       "         f\"(probabilidad {float((T_sim <= T + CP_rz).mean()):.3f}).\")"))
    celdas += apartado(4)
    celdas.append(code("rng = np.random.default_rng(semilla + 1)\n"
                       "a = plan.agresiva[cadena].astype(float).values\n"
                       "s = plan.segura[cadena].astype(float).values\n"
                       "sigma = np.log(s / a) / 1.2816\n"
                       "T_cadena = (a * np.exp(sigma * rng.standard_normal((N, len(cadena))))).sum(axis=1)\n"
                       "pd.DataFrame({'cadena aislada': [float(np.median(T_cadena)), float((T_cadena <= plan.compromiso).mean())],\n"
                       "              'proyecto completo': [float(np.median(T_sim)), float((T_sim <= plan.compromiso).mean())]},\n"
                       "             index=['mediana', 'P(≤ fecha comprometida)']).round(3)"))
    celdas.append(code("Markdown(f\"La cadena aislada cumpliría la fecha comprometida con probabilidad {float((T_cadena <= plan.compromiso).mean()):.3f}; \"\n"
                       "         f\"el proyecto completo, con {float((T_sim <= plan.compromiso).mean()):.3f}. La diferencia es el **sesgo de convergencia**: \"\n"
                       "         \"en los puntos donde una cadena de alimentación desemboca en la cadena crítica manda la que llega más tarde, y a veces \"\n"
                       "         \"el retraso de la alimentación supera su colchón y pasa a la cadena. Los colchones de alimentación existen para que esa \"\n"
                       "         \"diferencia sea pequeña. \"\n"
                       "         f\"El cronograma tradicional ({plan.duracion_tradicional} días) se cumpliría con probabilidad {float((T_sim <= plan.duracion_tradicional).mean()):.3f} \"\n"
                       "         f\"si se ejecutara por relevos: sus {plan.duracion_tradicional - plan.compromiso} días de más sobre el compromiso suben la \"\n"
                       "         f\"probabilidad de {float((T_sim <= plan.compromiso).mean()):.3f} a {float((T_sim <= plan.duracion_tradicional).mean()):.3f}: \"\n"
                       "         \"está sobreprotegido, y además, por la ley de Parkinson, esos días tenderían a consumirse igualmente.\")"))
    celdas += _autoevaluacion(_p_monte_carlo(plan, N, semilla), f"Cadena crítica — Monte Carlo del colchón (compromiso el día {plan.compromiso})")
    sub = (("con un equipo compartido; " if rec else "") +
           f"compromiso {plan.compromiso} frente a {plan.duracion_tradicional} días; Monte Carlo de {N} réplicas")
    return _con_contexto(celdas, rng), sub


# ---------------------------------------------------------------------------
# Variante 9 — Cronograma tradicional con la ley de Parkinson frente a relevos
# ---------------------------------------------------------------------------
def _parkinson(rng):
    def acepta(info, plan, df):
        crit_s, _ = _rutas_criticas(info, df["segura"])
        return len(crit_s) == 1 and len(plan.alimentacion) >= 1 and len(plan.cadena) >= 4
    info, plan, df = _buscar(rng, acepta)
    N, semilla = 20000, int(rng.integers(1, 10000))
    celdas = [md("# Enunciado"),
              md("Un proyecto puede programarse de dos maneras. " + ESTIMACIONES +
                 " Las actividades no comparten recursos.\n\n"
                 "- **Cronograma tradicional**: estimaciones seguras, cada actividad con su fecha de inicio (la más "
                 "temprana); el responsable empieza en esa fecha (no antes, aunque pueda) y, por la ley de Parkinson, "
                 "nunca entrega antes de su estimación segura: si termina antes, no lo comunica.\n"
                 "- **Cadena crítica**: estimaciones agresivas, colchones por cortar y pegar (redondeados al día "
                 "superior); las actividades de la cadena empiezan en cuanto terminan sus predecesoras (relevos) y las de "
                 "alimentación no antes de su inicio del plan."),
              preguntas([
                  "Calcule la duración del cronograma tradicional y su ruta crítica. ¿Cuántas actividades tiene?",
                  "Obtenga el plan de cadena crítica: cadena, colchones y fecha comprometida.",
                  f"Simule {N} ejecuciones (semilla {semilla}) en las que {SIMULACION}, en los dos regímenes. ¿Con qué "
                  "probabilidad se cumple la fecha prometida en cada caso? Compare también las duraciones medias.",
                  "Si cada actividad cumple su estimación segura con probabilidad 0,9 de forma independiente, ¿qué "
                  "probabilidad hay de que ninguna actividad de la ruta crítica tradicional la supere? Compárela con la "
                  "simulada y explique por qué el cronograma tradicional, con un margen en cada actividad, protege tan "
                  "poco la fecha de entrega."]),
              _datos(df),
              md("# Resolución"),
              code(ARRANQUE + f"\nR = proyecto.path_matrix(dummies=False)\nN, semilla = {N}, {semilla}"),
              code(AYUDAS)]
    celdas += apartado(1)
    celdas.append(code("datos_proyecto['duration'] = datos_proyecto['segura']"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones seguras (cronograma tradicional)", label="fig-pert")
    celdas.append(_cal(cap="Calendario del cronograma tradicional"))
    celdas.append(code("ruta_tradicional = [a for a in calendario.sort_values('inicio_mas_temprano').index if calendario.loc[a, 'H_total'] == 0]\n"
                       "T_tradicional = int(calendario['fin_mas_temprano'].max())\n"
                       "k = len(ruta_tradicional)\n"
                       "Markdown(f'El cronograma tradicional dura **{T_tradicional} días**; ruta crítica {\" - \".join(ruta_tradicional)} ({k} actividades).')"))
    celdas += apartado(2)
    celdas.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')\n"
                       "cadena, T, CP = plan.cadena, plan.duracion_cadena, plan.colchon_proyecto\n"
                       "display(detalle_colchon(plan.margen[cadena], 'colchón del proyecto').loc[['cortar y pegar = 0,5·Σ márgenes']])\n"
                       "display(tabla_alimentacion(plan))\n"
                       "Markdown(f'Cadena {\" - \".join(cadena)} ({T} días), colchón del proyecto {CP} días: compromiso **{plan.compromiso} días**.')"))
    celdas.append(code('#| fig-cap: "Gantt del plan de cadena crítica"\n#| label: fig-ganttCCPM\n'
                       f"pinta_tikz(plan.gantt_tikz(params={GANTT_CCPM}))"))
    celdas += apartado(3)
    celdas.append(md("En el régimen tradicional cada actividad empieza en $\\max$(su fecha del cronograma, fin de sus "
                     "predecesoras) y dura $\\max$(duración simulada, estimación segura): los adelantos se pierden y los "
                     "retrasos se transmiten. En el de relevos, cada actividad de la cadena empieza al terminar sus "
                     "predecesoras y dura lo que dura."))
    celdas.append(code("T_trad = plan.simular(N, semilla=semilla, regimen='tradicional')\n"
                       "T_rel = plan.simular(N, semilla=semilla, regimen='relevos')\n"
                       "pd.DataFrame({'tradicional (Parkinson)': {'fecha prometida': T_tradicional, 'P(cumplir)': float((T_trad <= T_tradicional).mean()),\n"
                       "                                          'duración media': float(T_trad.mean()), 'percentil 90': float(np.percentile(T_trad, 90))},\n"
                       "              'cadena crítica (relevos)': {'fecha prometida': plan.compromiso, 'P(cumplir)': float((T_rel <= plan.compromiso).mean()),\n"
                       "                                           'duración media': float(T_rel.mean()), 'percentil 90': float(np.percentile(T_rel, 90))},\n"
                       "              'relevos con la fecha tradicional': {'fecha prometida': T_tradicional, 'P(cumplir)': float((T_rel <= T_tradicional).mean()),\n"
                       "                                                   'duración media': float(T_rel.mean()), 'percentil 90': float(np.percentile(T_rel, 90))}}).astype(float).round(3).pipe(sin_decimales)"))
    celdas.append(code('#| fig-cap: "Duración simulada del proyecto en los dos regímenes"\n#| label: fig-regimenes\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
                       "bins = np.linspace(min(T_rel.min(), T_trad.min()), max(T_rel.max(), T_trad.max()), 60)\n"
                       "ax.hist(T_trad, bins=bins, alpha=0.6, density=True, label='tradicional (Parkinson)')\n"
                       "ax.hist(T_rel, bins=bins, alpha=0.6, density=True, label='cadena crítica (relevos)')\n"
                       "ax.axvline(T_tradicional, color='tab:blue', linestyle='--'); ax.axvline(plan.compromiso, color='tab:orange', linestyle='--')\n"
                       "ax.set_xlabel('duración (días)'); ax.set_ylabel('densidad'); ax.legend(); plt.show()"))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'''La probabilidad de que ninguna de las {k} actividades de la ruta crítica supere su estimación segura es \n"
                       "0,9^{k} = **{0.9 ** k:.3f}**. Con la ley de Parkinson y las fechas fijas, cualquier exceso de una actividad crítica retrasa el proyecto "
                       "y ningún adelanto lo compensa, así que la probabilidad de cumplir no puede ser mayor que esa (la simulada es "
                       "{float((T_trad <= T_tradicional).mean()):.3f}: también fallan las no críticas que se pasan de su holgura).\n\n"
                       "Cada estimación segura protege su actividad, no la fecha de entrega: el margen se pierde porque los adelantos no se "
                       "transmiten (ley de Parkinson), porque quien tiene holgura empieza tarde (síndrome del estudiante) y porque en las "
                       "convergencias manda el camino que llega más tarde. La cadena crítica retira esos márgenes y los concentra en "
                       "colchones: promete {T_tradicional - plan.compromiso} días antes y cumple con probabilidad "
                       "{float((T_rel <= plan.compromiso).mean()):.3f}. La última columna de la tabla muestra que el problema no estaba en las "
                       "estimaciones: con las mismas duraciones, ejecutando por relevos, la fecha tradicional se cumpliría con probabilidad "
                       "{float((T_rel <= T_tradicional).mean()):.3f}.''')"))
    k_trad = int(info["P"].loc[_rutas_criticas(info, df["segura"])[0][0]].sum())
    celdas += _autoevaluacion(_p_parkinson(plan, N, semilla, k_trad), f"Cadena crítica — Parkinson frente a relevos (ruta de {k_trad} actividades)")
    return _con_contexto(celdas, rng), (f"ruta tradicional de {k_trad} actividades; "
                                       f"compromiso {plan.compromiso} frente a {plan.duracion_tradicional} días")


# ---------------------------------------------------------------------------
# Variante 10 — Varios proyectos y un recurso tambor: escalonamiento
# ---------------------------------------------------------------------------
TAMBOR = r'''
# Funciones auxiliares: varios proyectos que comparten el recurso tambor
from math import ceil

COLORES_TRAMO = {'antes': 'green!25', 'tambor': 'red!30', 'despues': 'green!25', 'CP': 'orange!45',
                 'DB': 'orange!15', 'CB': 'black!12'}


def sin_decimales(tabla):
    """Números enteros sin «.0» al mostrar una tabla."""
    return tabla.map(lambda v: f'{v:g}' if isinstance(v, float) else v)


def tambor_sin_escalonar(datos, turno=None):
    """Todos los proyectos se lanzan el día 0 y llegan al tambor al terminar su parte «antes».
    turno=None: cuando el tambor queda libre atiende, hasta terminarla, la tarea del proyecto de mayor
    prioridad que le espera. turno=q: multitarea, turnos de q días rotando en el orden de prioridad entre
    los proyectos que le esperan. Devuelve (fin de la tarea del tambor de cada proyecto, tramos)."""
    orden = list(datos.index)
    llegada = {p: int(datos.loc[p, 'antes']) for p in orden}
    resto = {p: int(datos.loc[p, 'tambor']) for p in orden}
    t, tramos, fin, ultimo = 0, [], {}, -1
    while resto:
        esperan = [i for i, p in enumerate(orden) if p in resto and llegada[p] <= t]
        if not esperan:
            t = min(llegada[p] for p in resto)
            continue
        if turno is None:
            i = esperan[0]
        else:
            siguientes = [i for i in esperan if i > ultimo]
            i = siguientes[0] if siguientes else esperan[0]
        p = orden[i]
        w = resto[p] if turno is None else min(turno, resto[p])
        tramos.append((p, t, w))
        t += w
        resto[p] -= w
        ultimo = i
        if resto[p] == 0:
            fin[p] = t
            del resto[p]
    return pd.Series(fin).reindex(orden), tramos


def escalonar(datos, colchon_tambor, fraccion):
    """Programa del tambor en el orden de prioridad, con colchón de capacidad tras cada tarea, y
    lanzamiento de cada proyecto para que llegue al tambor con su colchón del tambor por delante."""
    filas, libre, libre_txt = [], None, ''
    for p, f in datos.iterrows():
        a, t, b, cp = (int(f[c]) for c in ('antes', 'tambor', 'despues', 'CP'))
        llega = a + colchon_tambor
        if libre is None:
            s = llega
            calc = f'{a} + {colchon_tambor} = {s}'
        else:
            s = max(llega, libre)
            calc = f'máx({a} + {colchon_tambor}; {libre_txt}) = máx({llega}; {libre}) = {s}'
        cb = ceil(fraccion * t - 1e-9)
        filas.append(dict(proyecto=p, a=a, t=t, b=b, cp=cp, inicio_tambor=s, calc_inicio=calc,
                          lanzamiento=s - colchon_tambor - a, fin_tambor=s + t, fin=s + t + b,
                          compromiso=s + t + b + cp, colchon_capacidad=cb, libre_despues=s + t + cb))
        libre = s + t + cb
        libre_txt = f'{s} + {t} + {cb} = {libre}'
    return pd.DataFrame(filas).set_index('proyecto')


def coma(x):
    return f'{x:g}'.replace('.', ',')


def tabla_escalonado(esc, colchon_tambor, fraccion):
    """Cálculos del escalonamiento con los números sustituidos."""
    filas = {}
    for p, f in esc.iterrows():
        filas[p] = {'inicio en el tambor = máx(antes + colchón del tambor; tambor libre)': f['calc_inicio'],
                    'lanzamiento = inicio en el tambor − colchón del tambor − antes':
                        f"{f['inicio_tambor']} − {colchon_tambor} − {f['a']} = {f['lanzamiento']}",
                    'fin previsto = inicio en el tambor + tambor + después':
                        f"{f['inicio_tambor']} + {f['t']} + {f['b']} = {f['fin']}",
                    'fecha comprometida = fin previsto + CP': f"{f['fin']} + {f['cp']} = {f['compromiso']}",
                    f'colchón de capacidad = ⌈{coma(fraccion)}·tambor⌉': f"⌈{coma(fraccion)}·{f['t']}⌉ = {f['colchon_capacidad']}",
                    'tambor libre para el siguiente = inicio + tambor + colchón de capacidad':
                        f"{f['inicio_tambor']} + {f['t']} + {f['colchon_capacidad']} = {f['libre_despues']}"}
    return pd.DataFrame(filas)


def tabla_sin_escalonar(datos, fin_tambor):
    """Fin de cada proyecto lanzado el día 0 y comparación con el compromiso calculado con el tambor libre."""
    filas = {}
    for p, f in datos.iterrows():
        a, t, b, cp = (int(f[c]) for c in ('antes', 'tambor', 'despues', 'CP'))
        fin = int(fin_tambor[p]) + b
        libre = a + t + b + cp
        filas[p] = {'llega al tambor (= antes)': a, 'termina en el tambor': int(fin_tambor[p]),
                    'fin = termina en el tambor + después': f'{int(fin_tambor[p])} + {b} = {fin}',
                    'compromiso con el tambor libre = antes + tambor + después + CP': f'{a} + {t} + {b} + {cp} = {libre}',
                    'retraso sobre el compromiso': max(0, fin - libre)}
    return pd.DataFrame(filas)


def en_curso_a_la_vez(inicios, fines):
    """Máximo de proyectos abiertos a la vez (intervalos [inicio, fin))."""
    eventos = sorted([(int(s), 1) for s in inicios] + [(int(f), -1) for f in fines])
    n = m = 0
    for _, d in eventos:
        n += d
        m = max(m, n)
    return m


def gantt_tramos_tikz(filas, horizonte):
    """Diagrama de barras por tramos (TikZ): filas = [(etiqueta, [(inicio, días, tipo, texto), ...]), ...]."""
    u = min(0.34, 15.0 / max(horizonte, 1))
    n = len(filas)
    paso = 5 if horizonte <= 60 else 10
    L = [rf'\begin{{tikzpicture}}[x={u:.4f}cm, y=-0.62cm, font=\small]']
    for x in range(0, horizonte + 1, paso):
        L.append(rf'\draw[black!15] ({x},-0.55) -- ({x},{n - 0.45});')
        L.append(rf'\node[font=\scriptsize, black!60] at ({x},-0.85) {{{x}}};')
    for i, (etiqueta, tramos) in enumerate(filas):
        L.append(rf'\node[anchor=east, font=\small\bfseries] at (-0.6,{i}) {{{etiqueta}}};')
        for s, d, tipo, texto in tramos:
            if d <= 0:
                continue
            borde = ', dashed' if tipo in ('DB', 'CB') else ''
            L.append(rf'\draw[fill={COLORES_TRAMO[tipo]}, draw=black!55{borde}] ({s},{i - 0.32}) rectangle ({s + d},{i + 0.32});')
            if texto != '':
                L.append(rf'\node[font=\scriptsize] at ({s + d / 2:.2f},{i}) {{{texto}}};')
    L.append(r'\end{tikzpicture}')
    return '\n'.join(L)


def filas_sin_escalonar(datos, fin_tambor, tramos):
    filas = []
    for p, f in datos.iterrows():
        a, b = int(f['antes']), int(f['despues'])
        mios = [(s, w, 'tambor', '') for q, s, w in tramos if q == p]
        filas.append((p, [(0, a, 'antes', a)] + mios + [(int(fin_tambor[p]), b, 'despues', b)]))
    filas.append(('Tambor', [(s, w, 'tambor', q) for q, s, w in tramos]))
    return filas


def filas_escalonado(esc, colchon_tambor):
    filas = []
    for p, f in esc.iterrows():
        lz, a, s, t, b, cp = (int(f[c]) for c in ('lanzamiento', 'a', 'inicio_tambor', 't', 'b', 'cp'))
        filas.append((p, [(lz, a, 'antes', a), (lz + a, colchon_tambor, 'DB', colchon_tambor), (s, t, 'tambor', t),
                          (s + t, b, 'despues', b), (s + t + b, cp, 'CP', cp)]))
    filas.append(('Tambor', [x for p, f in esc.iterrows() for x in
                             ((int(f['inicio_tambor']), int(f['t']), 'tambor', p),
                              (int(f['fin_tambor']), int(f['colchon_capacidad']), 'CB', ''))]))
    return filas
'''.strip("\n")

NOMBRES_TAMBOR = ["la oficina de cálculo de estructuras", "el laboratorio de ensayos", "el equipo de puesta en marcha",
                  "el equipo de la grúa de gran tonelaje", "la unidad de validación de software"]


def _tambor(rng):
    ns = {"pd": pd}
    exec(TAMBOR, ns)
    for _ in range(20000):
        n = int(rng.choice([3, 4]))
        a = [int(x) for x in rng.integers(2, 11, size=n)]
        t = [int(x) for x in rng.integers(4, 10, size=n)]
        b = [int(x) for x in rng.integers(2, 9, size=n)]
        fr = rng.choice([0.25, 0.3, 0.35, 0.4], size=n)
        cp = [int(math.ceil((x + y + z) * f - 1e-9)) for x, y, z, f in zip(a, t, b, fr)]
        q, DB = int(rng.choice([2, 3])), int(rng.choice([2, 3]))
        nombres = [f"P{i}" for i in range(1, n + 1)]
        df = pd.DataFrame({"antes": a, "tambor": t, "despues": b, "CP": cp}, index=pd.Index(nombres, name="proyecto"))
        if q >= min(t):
            continue
        fA, _ = ns["tambor_sin_escalonar"](df, q)
        fB, _ = ns["tambor_sin_escalonar"](df, None)
        esc = ns["escalonar"](df, DB, 0.25)
        espera_B = [int(fB[p]) - df.loc[p, "tambor"] - df.loc[p, "antes"] for p in nombres]
        if sum(1 for e in espera_B if e > 0) < 2:
            continue
        if sum(1 for p in nombres if fA[p] > fB[p]) < n - 1:
            continue
        finA = fA + df["despues"]
        if not any(finA[p] > df.loc[p, ["antes", "tambor", "despues", "CP"]].sum() for p in nombres):
            continue
        llega = df["antes"] + DB
        if not any(esc.loc[p, "inicio_tambor"] > llega[p] for p in nombres):
            continue
        if int((esc["lanzamiento"] > 0).sum()) < 2:
            continue
        durB = float((fB + df["despues"]).mean())
        durC = float((esc["fin"] - esc["lanzamiento"]).mean())
        if not durC < durB - 1:
            continue
        if int(esc["fin"].max()) - int((fB + df["despues"]).max()) > 6:
            continue
        horizonte = int(max(esc["compromiso"].max(), (finA + df["CP"]).max()))
        if horizonte > 75:
            continue
        an, tn, bn = int(rng.integers(2, 9)), int(rng.integers(4, 9)), int(rng.integers(2, 8))
        cpn = int(math.ceil((an + tn + bn) * 0.35 - 1e-9))
        libre = int(esc["libre_despues"].iloc[-1])
        if an + DB >= libre:
            continue
        break
    else:
        raise RuntimeError("No se encontró una cartera con los rasgos pedidos")
    tambor = str(rng.choice(NOMBRES_TAMBOR))
    nuevo = f"P{n + 1}"
    celdas = [md("# Enunciado"),
              md(f"Una empresa tiene en cartera {n} proyectos que dependen de un mismo equipo escaso, {tambor}: es el "
                 "**tambor**, el recurso que limita cuántos proyectos puede sacar adelante la organización, y sólo "
                 "atiende una tarea a la vez. Cada proyecto se ha planificado con el método de la cadena crítica y su "
                 "cadena pasa una vez por el tambor. El cuadro resume, en días y con estimaciones agresivas, lo que dura "
                 "la cadena de cada proyecto antes de su tarea en el tambor (*antes*), la tarea del tambor (*tambor*), lo "
                 "que queda después (*despues*) y su colchón del proyecto (*CP*). Los proyectos están en el orden de "
                 "prioridad fijado por la dirección. Las demás actividades de cada proyecto no compiten por el tambor."),
              preguntas([
                  f"Si se lanzan los {n} proyectos el día 0 y el tambor reparte su tiempo en turnos de {q} días entre los "
                  "proyectos que le esperan (multitarea, rotando en el orden de prioridad; cada proyecto entra en la "
                  "rotación cuando termina su parte *antes*), ¿qué día termina cada proyecto? Compárelo con la fecha que "
                  "se le comprometería si tuviera el tambor libre (antes + tambor + después + CP).",
                  "Repita el cálculo si, con todos los proyectos lanzados también el día 0, el tambor no hace "
                  "multitarea: cuando queda libre atiende, hasta terminarla, la tarea del proyecto de mayor prioridad "
                  "que le espera.",
                  "Escalone los proyectos por el tambor: sus tareas se programan en el orden de prioridad, una detrás "
                  "de otra, con un **colchón de capacidad** tras cada una igual al 25 % de su duración (redondeado al "
                  f"día superior); cada proyecto se lanza de modo que su parte *antes* termine {DB} días (**colchón del "
                  "tambor**) antes de su turno en el tambor; el primero se lanza el día 0. Dé el día de lanzamiento, el "
                  "inicio en el tambor, el fin previsto y la fecha comprometida de cada proyecto, y dibuje el programa.",
                  "Compare los tres regímenes: fin de cada proyecto, duración media de un proyecto (de su lanzamiento a "
                  "su fin) y máximo de proyectos abiertos a la vez. ¿Qué gana la organización con el escalonamiento? "
                  "¿Para qué sirven el colchón de capacidad y el colchón del tambor?",
                  f"Llega un proyecto nuevo, {nuevo}, con antes = {an}, tambor = {tn}, después = {bn} y CP = {cpn} días, de "
                  "prioridad menor que los anteriores. Con el programa escalonado, ¿qué día puede lanzarse y qué fecha "
                  "puede comprometerse con su cliente?"]),
              code('#| tbl-cap: "Cartera de proyectos (días, estimaciones agresivas; en orden de prioridad)"\n'
                   '#| label: tbl-datosEnunciado\n' + literal_dataframe(df, indice="proyecto")),
              md("# Resolución"),
              code(TAMBOR),
              code(f"datos = datos_enunciado.copy()\nturno, colchon_tambor, fraccion = {q}, {DB}, 0.25\n"
                   f"nuevo = dict(nombre={nuevo!r}, antes={an}, tambor={tn}, despues={bn}, CP={cpn})")]
    celdas += apartado(1)
    celdas.append(md(f"Cada proyecto llega al tambor el día *antes* (se lanzó el día 0). El tambor trabaja {q} días "
                     "seguidos en el primer proyecto que espera, pasa al siguiente en el orden de prioridad, y así "
                     "sucesivamente; un proyecto sale de la rotación cuando termina su tarea. La tabla recoge los turnos."))
    celdas.append(code('#| tbl-cap: "Turnos del tambor con multitarea"\n#| label: tbl-turnosMulti\n'
                       "fin_multi, tramos_multi = tambor_sin_escalonar(datos, turno)\n"
                       "pd.DataFrame(tramos_multi, columns=['proyecto', 'inicio', 'días']).assign(fin=lambda t: t['inicio'] + t['días']).rename(index=lambda i: i + 1).T"))
    celdas.append(code('#| tbl-cap: "Fin de cada proyecto con multitarea en el tambor"\n#| label: tbl-finMulti\n'
                       "tabla_sin_escalonar(datos, fin_multi)"))
    celdas.append(code('#| fig-cap: "Todos los proyectos lanzados el día 0 con multitarea en el tambor (tareas del tambor en rojo)"\n#| label: fig-multi\n'
                       "fin_proy_multi = fin_multi + datos['despues']\n"
                       "pinta_tikz(gantt_tramos_tikz(filas_sin_escalonar(datos, fin_multi, tramos_multi), int((fin_proy_multi).max())))"))
    celdas.append(code("compromiso_libre = datos[['antes', 'tambor', 'despues', 'CP']].sum(axis=1)\n"
                       "fuera = [p for p in datos.index if fin_proy_multi[p] > compromiso_libre[p]]\n"
                       "Markdown(f'Con multitarea, {len(fuera)} de los {len(datos)} proyectos ({\", \".join(fuera)}) terminan después de la fecha '\n"
                       "         'que se les comprometería con el tambor libre: ni siquiera su colchón del proyecto absorbe la espera.' if fuera else\n"
                       "         'Todos los proyectos cumplirían la fecha calculada con el tambor libre, pero consumiendo su colchón.')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Fin de cada proyecto sin multitarea en el tambor (todos lanzados el día 0)"\n#| label: tbl-finSecuencial\n'
                       "fin_sec, tramos_sec = tambor_sin_escalonar(datos, None)\n"
                       "fin_proy_sec = fin_sec + datos['despues']\n"
                       "display(pd.DataFrame(tramos_sec, columns=['proyecto', 'inicio', 'días']).assign(fin=lambda t: t['inicio'] + t['días']).rename(index=lambda i: i + 1).T)\n"
                       "tabla_sin_escalonar(datos, fin_sec)"))
    celdas.append(code("antes_ = [p for p in datos.index if fin_proy_sec[p] < fin_proy_multi[p]]\n"
                       "ult_s, ult_m = int(fin_proy_sec.max()), int(fin_proy_multi.max())\n"
                       "frase = (f'el último proyecto termina el día {ult_s} frente al {ult_m}' if ult_s != ult_m else\n"
                       "         f'el último proyecto termina el mismo día ({ult_s}), porque el tambor hace el mismo trabajo')\n"
                       "Markdown(f'Sin multitarea terminan antes {\", \".join(antes_)} '\n"
                       "         f'(media de los fines: {fin_proy_sec.mean():.2f} frente a {fin_proy_multi.mean():.2f} días) y ' + frase + ': repartir el '\n"
                       "         'tambor no adelanta a nadie y retrasa a casi todos, porque cada tarea del tambor termina en su último turno y está '\n"
                       "         'en la cadena crítica de su proyecto.')"))
    celdas += apartado(3)
    celdas.append(md("El tambor marca el ritmo de la organización: sus tareas se programan primero, en el orden de "
                     "prioridad y sin interrupción, dejando tras cada una el colchón de capacidad. Después cada proyecto "
                     "se lanza hacia atrás desde su turno (la *cuerda* que tira del lanzamiento): su parte *antes* debe "
                     "terminar el colchón del tambor antes de que le toque."))
    celdas.append(code('#| tbl-cap: "Escalonamiento por el tambor"\n#| label: tbl-escalonado\n'
                       "esc = escalonar(datos, colchon_tambor, fraccion)\n"
                       "tabla_escalonado(esc, colchon_tambor, fraccion)"))
    celdas.append(code('#| fig-cap: "Programa escalonado: parte antes y después en verde, colchón del tambor y colchón del proyecto en naranja, tarea del tambor en rojo y colchones de capacidad en gris"\n#| label: fig-escalonado\n'
                       "pinta_tikz(gantt_tramos_tikz(filas_escalonado(esc, colchon_tambor), int(esc['compromiso'].max())))"))
    celdas.append(code("Markdown('Lanzamientos: ' + '; '.join(f'{p} el día {int(f.lanzamiento)}, compromiso el día {int(f.compromiso)}' for p, f in esc.iterrows()) + '.')"))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Comparación de los tres regímenes"\n#| label: tbl-regimenes\n'
                       "ceros = pd.Series(0, index=datos.index)\n"
                       "regimenes = {'multitarea, todos el día 0': (ceros, fin_proy_multi), 'sin multitarea, todos el día 0': (ceros, fin_proy_sec),\n"
                       "             'escalonados por el tambor': (esc['lanzamiento'], esc['fin'])}\n"
                       "comparacion = pd.DataFrame({r: {**{f'fin de {p}': int(f[p]) for p in datos.index},\n"
                       "                                'duración media (fin − lanzamiento)': round(float((f - l).mean()), 2),\n"
                       "                                'máximo de proyectos abiertos a la vez': en_curso_a_la_vez(l, f),\n"
                       "                                'último fin': int(f.max())} for r, (l, f) in regimenes.items()})\n"
                       "sin_decimales(comparacion.astype(float))"))
    celdas.append(code("dur = comparacion.loc['duración media (fin − lanzamiento)']\n"
                       "ultimo = comparacion.loc['último fin']\n"
                       "frase_ultimo = (f'El último proyecto termina el día {int(ultimo.iloc[2])}, '\n"
                       "                + (f'{int(ultimo.iloc[2] - ultimo.iloc[1])} días después que sin escalonar: es el precio de los colchones de capacidad y del tambor, la reserva que hace fiables las fechas comprometidas. '\n"
                       "                   if ultimo.iloc[2] > ultimo.iloc[1] else 'no más tarde que sin escalonar. '))\n"
                       "Markdown(f'Con todos los proyectos lanzados a la vez hay {len(datos)} abiertos desde el día 0; escalonados, como máximo '\n"
                       "         f\"{int(comparacion.loc['máximo de proyectos abiertos a la vez'].iloc[2])}. Cada proyecto pasa abierto, de media, {dur.iloc[2]:.2f} días frente a {dur.iloc[1]:.2f} sin multitarea \"\n"
                       "         f'y {dur.iloc[0]:.2f} con multitarea: inmoviliza menos tiempo a su equipo y menos dinero, y su fecha no depende de cómo se '\n"
                       "         'reparta el tambor. ' + frase_ultimo)"))
    celdas.append(md("La organización no puede terminar más proyectos de los que permite el tambor: lanzar más trabajo "
                     "del que puede atender sólo alarga las colas y empuja a la multitarea. Escalonando, el tambor trabaja "
                     "sin interrupciones y sin repartirse, y cada proyecto entra cuando el tambor podrá atenderlo. El "
                     "**colchón de capacidad** deja al tambor un margen tras cada tarea: si una se alarga, no arrastra a "
                     "los proyectos siguientes, cuyas fechas se han comprometido contando con su turno. El **colchón del "
                     "tambor** protege el turno: el trabajo previo llega con unos días de adelanto, de modo que un retraso "
                     "en la parte *antes* no deja al tambor parado esperando."))
    celdas += apartado(5)
    celdas.append(code("libre = int(esc['libre_despues'].iloc[-1])\n"
                       "s_n = max(nuevo['antes'] + colchon_tambor, libre)\n"
                       "lanz_n = s_n - colchon_tambor - nuevo['antes']\n"
                       "fin_n = s_n + nuevo['tambor'] + nuevo['despues']\n"
                       "Markdown(f\"El tambor queda libre, tras la última tarea y su colchón de capacidad, el día {libre}. \"\n"
                       "         f\"Inicio de {nuevo['nombre']} en el tambor: máx({nuevo['antes']} + {colchon_tambor}; {libre}) = **{s_n}**; \"\n"
                       "         f\"lanzamiento: {s_n} − {colchon_tambor} − {nuevo['antes']} = **día {lanz_n}**; fin previsto {s_n} + {nuevo['tambor']} + \"\n"
                       "         f\"{nuevo['despues']} = {fin_n}; fecha comprometida {fin_n} + {nuevo['CP']} = **día {fin_n + nuevo['CP']}**. \"\n"
                       "         \"Lanzarlo antes no lo adelantaría: sólo lo tendría abierto esperando al tambor.\")"))
    celdas += _autoevaluacion(_p_tambor(ns, df, q, DB, an, tn, bn, cpn, nuevo), f"Cadena crítica — varios proyectos y un tambor ({n} proyectos)")
    sub = f"{n} proyectos y un tambor; turnos de {q} días; colchón del tambor de {DB} días"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variante 11 — Consumo de los colchones de alimentación y prioridades
# ---------------------------------------------------------------------------
PREVISION = r'''
def detalle_prevision(plan, dia, terminadas, en_curso):
    """Recorrido hacia delante desde el informe: fin previsto de cada actividad con sus candidatos.
    La cadena empieza en cuanto terminan sus predecesoras; la alimentación, no antes de su inicio del plan."""
    fin, filas = {}, {}
    for a in plan.orden:
        d = int(plan.agresiva[a])
        if a in terminadas:
            fin[a] = terminadas[a]
            filas[a] = {'estado': 'terminada', 'fin previsto': f'terminó el día {fin[a]}'}
        elif a in en_curso:
            fin[a] = dia + en_curso[a]
            filas[a] = {'estado': 'en curso', 'fin previsto': f'{dia} + {en_curso[a]} (restante) = {fin[a]}'}
        else:
            cands = [f'informe {dia}'] + [f'{p} {fin[p]}' for p in sorted(plan.predecesoras[a])]
            ini = max([dia] + [fin[p] for p in plan.predecesoras[a]])
            if a not in plan.cadena:
                cands.append(f'plan {int(plan.inicios[a])}')
                ini = max(ini, int(plan.inicios[a]))
            fin[a] = ini + d
            filas[a] = {'estado': 'pendiente', 'fin previsto': f"máx({'; '.join(cands)}) + {d} = {ini} + {d} = {fin[a]}"}
        filas[a]['de la cadena'] = 'sí' if a in plan.cadena else ''
    return pd.DataFrame(filas).T[['de la cadena', 'estado', 'fin previsto']], fin


def tabla_consumo_alimentacion(plan, fin):
    """Consumo de cada colchón de alimentación: fin previsto de la actividad que desemboca en él frente al
    inicio del colchón en el plan."""
    filas = {}
    for n, f in plan.alimentacion.iterrows():
        entrada = int(plan.inicios[f['antes_de']]) if f['antes_de'] != 'fin' else int(plan.duracion_cadena)
        ini = entrada - int(f['adoptado'])
        cons = max(0, fin[f['desde']] - ini)
        exceso = max(0, cons - int(f['adoptado']))
        filas[n] = {'arco': f"{f['desde']} → {f['antes_de']}", 'colchón (días)': int(f['adoptado']),
                    'el colchón empieza': f'{entrada} − {int(f["adoptado"])} = {ini}',
                    'consumido = máx(0, fin previsto − inicio del colchón)': f'máx(0, {fin[f["desde"]]} − {ini}) = {cons}',
                    '% consumido': round(100 * cons / int(f['adoptado']), 1) if f['adoptado'] else float('nan'),
                    'pasa a la cadena': exceso}
    return pd.DataFrame(filas).T


def fin_solo_cadena(plan, dia, terminadas, en_curso):
    """Fin previsto de la cadena si las cadenas de alimentación no la retrasaran."""
    fin = {}
    for a in plan.cadena:
        if a in terminadas:
            fin[a] = terminadas[a]
        elif a in en_curso:
            fin[a] = dia + en_curso[a]
        else:
            fin[a] = max([dia] + [fin[p] for p in plan.predecesoras[a] if p in fin]) + int(plan.agresiva[a])
    return fin[plan.cadena[-1]]


def colchones_que_protegen(plan, actividad):
    """Colchón que protege a una actividad: el del proyecto si es de la cadena; si no, los colchones de
    alimentación de los arcos por los que llega a la cadena (o al final)."""
    if actividad in plan.cadena:
        return ['CP']
    cadena = set(plan.cadena)
    suc = {x: [y for y in plan.predecesoras if x in plan.predecesoras[y]] for x in plan.predecesoras}
    vistos, pila = set(), [actividad]
    while pila:
        x = pila.pop()
        if x not in vistos:
            vistos.add(x)
            pila += [y for y in suc[x] if y not in cadena]
    return [n for n, f in plan.alimentacion.iterrows() if f['desde'] in vistos]
'''.strip("\n")


def _penetracion(plan, dia, terminadas, en_curso, ns):
    """Rasgos de un informe para la variante 11 (con las mismas funciones que el cuaderno)."""
    s = plan.seguimiento(dia, terminadas, en_curso)
    _, fin = ns["detalle_prevision"](plan, dia, terminadas, en_curso)
    cons = ns["tabla_consumo_alimentacion"](plan, fin)
    return s, fin, cons


def _consumo_alimentacion(rng):
    ns = {"pd": pd}
    exec(PREVISION, ns)

    def acepta(info, plan, df):
        a = plan.alimentacion
        return (2 <= len(a) <= 4 and int((a["adoptado"] >= 2).sum()) >= 2 and bool((a["adoptado"] >= 1).all())
                and len(plan.cadena) >= 4)
    hallado = None
    for _ in range(300):
        info, plan, df = _buscar(rng, acepta)
        for _ in range(60):
            real, ini, fin = _ejecucion(plan, rng)
            T_real = max(fin.values())
            for dia in range(max(2, int(0.25 * plan.duracion_cadena)), min(T_real, int(0.8 * plan.duracion_cadena) + 1)):
                terminadas = {a: fin[a] for a in plan.orden if fin[a] <= dia}
                en_curso = {a: fin[a] - dia for a in plan.orden if ini[a] < dia < fin[a]}
                if len(en_curso) < 2 or not any(a in plan.cadena for a in en_curso):
                    continue
                s, fprev, cons = _penetracion(plan, dia, terminadas, en_curso, ns)
                pct = cons["% consumido"].astype(float)
                parcial = cons[(pct > 0) & (pct < 100)]
                penetra = cons[cons["pasa a la cadena"] > 0]
                if len(parcial) == 0 or len(penetra) == 0:
                    continue
                desde = {n: plan.alimentacion.loc[n, "desde"] for n in penetra.index}
                if all(d in terminadas for d in desde.values()):
                    continue
                solo = ns["fin_solo_cadena"](plan, dia, terminadas, en_curso)
                if not (s["fin_previsto"] > solo and solo >= plan.duracion_cadena) or s["zona"] == "roja":
                    continue
                # prioridad: máximo % consumido único entre las actividades en curso
                pcts = {}
                for a in en_curso:
                    cs = ns["colchones_que_protegen"](plan, a)
                    if not cs:
                        break
                    pcts[a] = max(s["pct_consumo"] if c == "CP" else float(pct[c]) for c in cs)
                else:
                    orden_p = sorted(pcts.values(), reverse=True)
                    if len(set(ns["colchones_que_protegen"](plan, a)[0] for a in en_curso)) >= 2 and orden_p[0] > orden_p[1] + 1:
                        hallado = (info, plan, df, dia, terminadas, en_curso)
                        break
            if hallado:
                break
        if hallado:
            break
    if not hallado:
        raise RuntimeError("No se encontró un informe con un colchón de alimentación desbordado")
    info, plan, df, dia, terminadas, en_curso = hallado
    df = df.assign(inicio_plan=plan.inicios.reindex(df.index).astype(int))
    tabla_ca = pd.DataFrame({"arco": [f"{f['desde']} → {f['antes_de']}" for _, f in plan.alimentacion.iterrows()],
                             "días": [int(x) for x in plan.alimentacion["adoptado"]]},
                            index=pd.Index(list(plan.alimentacion.index), name="colchón"))
    estado = {a: (f"terminó el día {terminadas[a]}" if a in terminadas else
                  (f"en curso, le quedan {en_curso[a]}" if a in en_curso else "—")) for a in df.index}
    cad = " – ".join(plan.cadena)
    celdas = [md("# Enunciado"),
              md("Un proyecto se ha programado con el método de la cadena crítica (estimaciones agresivas, colchones por "
                 "cortar y pegar redondeados al día superior, actividades de alimentación lo más tarde posible). La cadena "
                 f"crítica es {cad}, de {plan.duracion_cadena} días; el colchón del proyecto es de "
                 f"{plan.colchon_proyecto} días y la fecha comprometida, el día {plan.compromiso}. El cuadro recoge las "
                 "estimaciones y el inicio de cada actividad en el plan (días contados desde el instante 0); el segundo "
                 "cuadro, los colchones de alimentación del plan, cada uno en el arco por el que su cadena de alimentación "
                 "desemboca en la cadena crítica (o en el final del proyecto)."),
              _datos(df),
              code('#| tbl-cap: "Colchones de alimentación del plan (días)"\n#| label: tbl-colchonesPlan\n'
                   + literal_dataframe(tabla_ca, nombre="colchones_plan", indice="colchón")),
              md(f"El día {dia} se recibe un informe de avance: qué actividades han terminado (y cuándo), cuántos días "
                 "estima su responsable que le quedan a cada actividad en curso y qué actividades no han empezado. Las "
                 "actividades de la cadena empiezan en cuanto terminan sus predecesoras y las de alimentación no antes de "
                 "su inicio en el plan."),
              code('#| tbl-cap: "Informe de avance"\n#| label: tbl-informe\n'
                   f"informe = dict(dia={dia}, terminadas={terminadas!r}, en_curso={en_curso!r})\n"
                   f"pd.DataFrame({{'día {dia}': {estado!r}}})"),
              preguntas([
                  "Calcule la fecha prevista de fin de cada actividad que no ha terminado (recorrido hacia delante desde "
                  "el informe con las duraciones agresivas de las que no han empezado).",
                  "¿Cuánto ha consumido cada colchón de alimentación, en días y en porcentaje? ¿Alguno queda desbordado? "
                  "¿Cuántos días pasan entonces a la cadena crítica?",
                  "Calcule el fin previsto del proyecto y el colchón del proyecto consumido. ¿Qué parte de ese consumo se "
                  "debe a la propia cadena y qué parte al colchón de alimentación desbordado? Calcule la cadena "
                  "completada y sitúe el informe en el gráfico de fiebre (zonas 10 → 70 % y 30 → 90 %).",
                  "Un técnico puede reforzar una de las actividades en curso. Con la regla de la gestión por colchones "
                  "—prioridad a la actividad cuyo colchón (el del proyecto para las de la cadena; el de alimentación "
                  "al que llega, para las demás) está más consumido, en porcentaje—, ¿a cuál debe ir? ¿Por qué no "
                  "basta con mirar las holguras del calendario?"]),
              md("# Resolución"),
              code(ARRANQUE + "\ndatos_proyecto['duration'] = datos_proyecto['agresiva']"),
              code(AYUDAS + "\n\n\n" + PREVISION)]
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones agresivas y la cadena crítica", label="fig-pert")
    celdas.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')\n"
                       f"assert plan.cadena == {plan.cadena!r} and plan.colchon_proyecto == {plan.colchon_proyecto}\n"
                       "T, CP = plan.duracion_cadena, plan.colchon_proyecto\n"
                       "tabla_alimentacion(plan)"))
    celdas.append(code('#| fig-cap: "Gantt del plan de cadena crítica: cadena en rojo, alimentación en verde, colchones en naranja"\n#| label: fig-ganttCCPM\n'
                       f"pinta_tikz(plan.gantt_tikz(params={GANTT_CCPM}))"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Fin previsto de cada actividad"\n#| label: tbl-prevision\n'
                       "dia, terminadas, en_curso = informe['dia'], informe['terminadas'], informe['en_curso']\n"
                       "prevision, fin_previsto = detalle_prevision(plan, dia, terminadas, en_curso)\n"
                       "prevision"))
    celdas += apartado(2)
    celdas.append(md("Un colchón de alimentación empieza en el plan cuando termina la última actividad de su cadena de "
                     "alimentación (su inicio es el de la actividad de la cadena a la que protege menos su tamaño). Todo "
                     "lo que esa actividad termine más tarde lo consume; si consume más que su tamaño, el exceso retrasa "
                     "a la actividad de la cadena crítica y pasa a consumir colchón del proyecto."))
    celdas.append(code('#| tbl-cap: "Consumo de los colchones de alimentación"\n#| label: tbl-consumoCA\n'
                       "consumo_ca = tabla_consumo_alimentacion(plan, fin_previsto)\n"
                       "sin_decimales(consumo_ca)"))
    celdas.append(code("desbordados = consumo_ca[consumo_ca['pasa a la cadena'] > 0]\n"
                       "Markdown('; '.join(f'{n} ({f[\"arco\"]}) se desborda: consume {f[\"consumido = máx(0, fin previsto − inicio del colchón)\"].split(\" = \")[-1]} días de {f[\"colchón (días)\"]} '\n"
                       "                   f'y pasan {f[\"pasa a la cadena\"]} día(s) a la cadena crítica' for n, f in desbordados.iterrows()) + '.')"))
    celdas += apartado(3)
    celdas.append(code("s = plan.seguimiento(dia, terminadas, en_curso)\n"
                       "solo = fin_solo_cadena(plan, dia, terminadas, en_curso)\n"
                       "Markdown(f\"Fin previsto del proyecto: **día {s['fin_previsto']:g}** (el máximo de los fines previstos); colchón consumido \"\n"
                       "         f\"{s['fin_previsto']:g} − {T} = {s['consumo']:g} día(s), {s['consumo']:g}/{CP} = **{s['pct_consumo']:.1f} %**. Si las cadenas \"\n"
                       "         f\"de alimentación no la retrasaran, la cadena terminaría el día {solo}: {solo - T} día(s) de consumo se deben a la propia \"\n"
                       "         f\"cadena y {s['fin_previsto'] - solo:g} al colchón de alimentación desbordado.\")"))
    celdas.append(code('#| tbl-cap: "Cadena completada, BPI y zona"\n#| label: tbl-zona\n'
                       "resumen_informes(plan, [s]).loc[['cadena completada', 'BPI = % colchón restante / % cadena restante', 'límite verde = 10 + 0,6·avance', 'límite rojo = 30 + 0,6·avance', 'zona']]"))
    celdas.append(code('#| fig-cap: "Gráfico de fiebre del informe"\n#| label: fig-fiebre\n'
                       "pinta_tikz(grafico_fiebre_tikz([s]))"))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Prioridad de las actividades en curso según el estado de su colchón"\n#| label: tbl-prioridad\n'
                       "pct_ca = consumo_ca['% consumido'].astype(float)\n"
                       "filas = {}\n"
                       "for a in sorted(en_curso):\n"
                       "    cs = colchones_que_protegen(plan, a)\n"
                       "    pcts = {c: (s['pct_consumo'] if c == 'CP' else float(pct_ca[c])) for c in cs}\n"
                       "    filas[a] = {'colchón que la protege': ', '.join(cs), '% consumido': round(max(pcts.values()), 1)}\n"
                       "prioridad = pd.DataFrame(filas).T.sort_values('% consumido', ascending=False)\n"
                       "prioridad['prioridad'] = range(1, len(prioridad) + 1)\n"
                       "sin_decimales(prioridad)"))
    celdas.append(code("primera = prioridad.index[0]\n"
                       "Markdown(f'El técnico debe reforzar **{primera}**, cuyo colchón ({prioridad.loc[primera, \"colchón que la protege\"]}) está '\n"
                       "         f'consumido en un {prioridad.loc[primera, \"% consumido\"]:.1f} %: es la que más amenaza la fecha comprometida. '\n"
                       "         'Las holguras del calendario no sirven para decidirlo: en el plan de cadena crítica las actividades de alimentación '\n"
                       "         'están programadas lo más tarde posible y su margen no es una holgura, sino su colchón; y la cadena crítica no tiene '\n"
                       "         'holgura, pero su protección es el colchón del proyecto. El porcentaje de colchón consumido da una medida común para '\n"
                       "         'comparar actividades de cadenas distintas, y la misma señal para todo el equipo.')"))
    celdas += _autoevaluacion(_p_consumo(plan, dia, terminadas, en_curso, ns), f"Cadena crítica — consumo de los colchones de alimentación (día {dia})")
    sub = (f"día {dia}: {len(desde)} colchón de alimentación desbordado; CP consumido "
           f"{plan.seguimiento(dia, terminadas, en_curso)['pct_consumo']:.0f} %")
    return _con_contexto(celdas, rng), sub


# ---------------------------------------------------------------------------
# Variante 12 — Tendencia del gráfico de fiebre: velocidad de consumo y previsión
# ---------------------------------------------------------------------------
def _tendencia(rng):
    def acepta(info, plan, df):
        return 5 <= len(plan.cadena) <= 7 and plan.colchon_proyecto >= 5
    hallado = None
    for _ in range(300):
        info, plan, df = _buscar(rng, acepta)
        T, CP = plan.duracion_cadena, plan.colchon_proyecto
        for _ in range(80):
            real, ini, fin = _ejecucion(plan, rng)
            T_real = max(fin.values())
            dias = []
            for frac in (0.2, 0.4, 0.6, 0.8):
                objetivo = int(round(frac * T_real))
                validos = [t for t in range(1, T_real) if any(ini[c] < t < fin[c] for c in plan.cadena)]
                if not validos:
                    break
                dias.append(min(validos, key=lambda t: (abs(t - objetivo), t)))
            if len(dias) < 4 or len(set(dias)) < 4:
                continue
            segs = []
            for d in dias:
                term = {a: fin[a] for a in plan.orden if fin[a] <= d}
                curso = {a: fin[a] - d for a in plan.orden if ini[a] < d < fin[a]}
                segs.append(plan.seguimiento(d, term, curso))
            xs = [s["pct_avance"] for s in segs]
            ys = [s["pct_consumo"] for s in segs]
            if min(ys) < 0 or any(x2 - x1 < 8 for x1, x2 in zip(xs, xs[1:])):
                continue
            v = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            if not (0.75 < v < 4) or segs[-1]["zona"] == "roja" or max(ys) > 100:
                continue
            hallado = (plan, dias, segs)
            break
        if hallado:
            break
    if not hallado:
        raise RuntimeError("No se encontraron informes con la tendencia buscada")
    plan, dias, segs = hallado
    T, CP = plan.duracion_cadena, plan.colchon_proyecto
    tabla = pd.DataFrame({"fin_previsto": [int(round(s["fin_previsto"])) for s in segs],
                          "completado": [int(round(s["avance"])) for s in segs]},
                         index=pd.Index(dias, name="dia"))
    celdas = [md("# Enunciado"),
              md(f"Un proyecto se gestiona con el método de la cadena crítica. La cadena crítica dura {T} días con las "
                 f"estimaciones agresivas, el colchón del proyecto es de {CP} días y la fecha comprometida, el día "
                 f"{T + CP}. El cuadro recoge cuatro informes de avance: el día del informe, la fecha prevista de fin de "
                 "la cadena (día del informe más la duración restante estimada) y los días de cadena completados "
                 "(duración de la cadena menos su duración restante; de una actividad en curso no se cuenta como "
                 "restante más que su duración agresiva). Zonas del gráfico de fiebre: verde por debajo de la recta "
                 "que va del 10 % de consumo con la cadena sin empezar al 70 % con la cadena terminada; roja por encima "
                 "de la recta que va del 30 % al 90 %; amarilla entre ambas."),
              code('#| tbl-cap: "Informes de avance (días)"\n#| label: tbl-datosEnunciado\n'
                   + literal_dataframe(tabla, indice="dia")),
              preguntas([
                  "Para cada informe calcule el porcentaje de colchón consumido, el de cadena completada, el BPI "
                  "(porcentaje de colchón restante entre porcentaje de cadena restante) y la zona.",
                  "Calcule la velocidad de consumo del colchón entre cada dos informes consecutivos: puntos de "
                  "porcentaje de colchón consumidos por cada punto de porcentaje de cadena completada. Compárela con la "
                  "pendiente de las fronteras de las zonas (0,6). ¿Qué indica una velocidad mayor que 0,6? ¿Y una "
                  "mayor que 1?",
                  "Si se mantuviera la velocidad del último tramo, ¿qué porcentaje del colchón se habría consumido al "
                  "terminar la cadena y qué día terminaría el proyecto? ¿Se cumpliría la fecha comprometida? ¿Con qué "
                  "porcentaje de cadena completada entraría el proyecto en la zona roja?",
                  "Demuestre que el BPI de un informe es la velocidad media de consumo que el colchón puede soportar "
                  "hasta el final de la cadena. Compárelo con la velocidad del último tramo y diga qué debería hacer el "
                  "jefe de proyecto."]),
              md("# Resolución"),
              code(f"from pmule import grafico_fiebre_tikz\nT, CP = {T}, {CP}\ninformes = datos_enunciado.copy()")]
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Consumo, avance, BPI y zona de cada informe"\n#| label: tbl-puntos\n'
                       "x = 100 * informes['completado'] / T\n"
                       "y = 100 * (informes['fin_previsto'] - T) / CP\n"
                       "bpi = (100 - y) / (100 - x)\n"
                       "lim_verde, lim_roja = 10 + 0.6 * x, 30 + 0.6 * x\n"
                       "zona = ['verde' if yy < lv else ('roja' if yy >= lr else 'amarilla') for yy, lv, lr in zip(y, lim_verde, lim_roja)]\n"
                       "pd.DataFrame({f'día {d}': {\n"
                       "    'colchón consumido': f\"({informes.loc[d, 'fin_previsto']} − {T})/{CP} = {y[d]:.1f} %\",\n"
                       "    'cadena completada': f\"{informes.loc[d, 'completado']}/{T} = {x[d]:.1f} %\",\n"
                       "    'BPI': f'(100 − {y[d]:.1f})/(100 − {x[d]:.1f}) = {bpi[d]:.2f}',\n"
                       "    'límites verde / rojo': f'{lim_verde[d]:.1f} % / {lim_roja[d]:.1f} %',\n"
                       "    'zona': z} for d, z in zip(informes.index, zona)})"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Velocidad de consumo entre informes consecutivos"\n#| label: tbl-velocidad\n'
                       "dias = list(informes.index)\n"
                       "v = {f'{d1} → {d2}': (y[d2] - y[d1]) / (x[d2] - x[d1]) for d1, d2 in zip(dias, dias[1:])}\n"
                       "pd.DataFrame({k: {'velocidad = Δ consumo / Δ avance': f'({y[d2]:.1f} − {y[d1]:.1f})/({x[d2]:.1f} − {x[d1]:.1f}) = {v[k]:.2f}'}\n"
                       "              for (k, (d1, d2)) in zip(v, zip(dias, dias[1:]))})"))
    celdas.append(md("Las dos fronteras suben 0,6 puntos de consumo por cada punto de avance. Con una velocidad mayor que "
                     "0,6 el punto del gráfico se acerca a la frontera superior: el proyecto empeora respecto a las "
                     "zonas aunque siga en la misma. Con una velocidad mayor que 1 el colchón se consume, en porcentaje, "
                     "más deprisa de lo que avanza la cadena: si se mantuviera desde el inicio, el colchón se agotaría "
                     "antes de terminarla. Con una velocidad negativa el proyecto recupera colchón."))
    celdas += apartado(3)
    celdas.append(code("dn, dp = dias[-1], dias[-2]\n"
                       "vn = v[f'{dp} → {dn}']\n"
                       "y100 = y[dn] + vn * (100 - x[dn])\n"
                       "fin_tendencia = T + y100 / 100 * CP\n"
                       "x_roja = (30 - y[dn] + vn * x[dn]) / (vn - 0.6)\n"
                       "texto = (f'Con la velocidad del último tramo ({vn:.2f}), al terminar la cadena se habría consumido {y[dn]:.1f} + {vn:.2f}·(100 − {x[dn]:.1f}) = '\n"
                       "         f'**{y100:.1f} %** del colchón: el proyecto terminaría hacia el día {T} + {y100 / 100:.3f}·{CP} = **{fin_tendencia:.1f}**, '\n"
                       "         + ('dentro de la fecha comprometida' if y100 <= 100 else f'{fin_tendencia - T - CP:.1f} días después de la fecha comprometida') + f' ({T + CP}). ')\n"
                       "texto += (f'La recta de la tendencia, y = {y[dn]:.1f} + {vn:.2f}·(x − {x[dn]:.1f}), corta la frontera roja, y = 30 + 0,6·x, en '\n"
                       "          f'x = (30 − {y[dn]:.1f} + {vn:.2f}·{x[dn]:.1f})/({vn:.2f} − 0,6) = **{x_roja:.1f} %** de cadena completada'\n"
                       "          + (': entraría en la zona roja antes de terminar.' if x_roja < 100 else ', después del final: no llegaría a entrar en la zona roja.'))\n"
                       "Markdown(texto)"))
    celdas.append(code('#| fig-cap: "Gráfico de fiebre con los informes y la tendencia del último tramo (discontinua)"\n#| label: fig-fiebre\n'
                       "puntos = [(x[d], y[d], f'día {d}') for d in dias]\n"
                       "dibujo = grafico_fiebre_tikz(puntos + [(100.0, y100, 'tendencia')], unir=False)\n"
                       "linea = r'\\draw[thick, gray] ' + ' -- '.join(f'({px:.2f},{py:.2f})' for px, py, _ in puntos) + ';'\n"
                       "tendencia = rf'\\draw[thick, dashed, blue!70!black] ({x[dn]:.2f},{y[dn]:.2f}) -- (100,{y100:.2f});'\n"
                       "pinta_tikz(dibujo.replace(r'\\end{tikzpicture}', linea + '\\n' + tendencia + '\\n' + r'\\end{tikzpicture}'))"))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'''Si en un informe se han consumido y % del colchón con x % de cadena completada, para terminar justo con el 100 % del colchón \n"
                       "consumido la velocidad media hasta el final debe ser (100 − y)/(100 − x), que es exactamente el BPI. Un BPI mayor que 1 dice que queda \n"
                       "proporcionalmente más colchón que cadena; comparado con la velocidad real dice si el ritmo de consumo es sostenible.\n\n\n"
                       "En el último informe el BPI es {bpi[dn]:.2f} y la velocidad del último tramo, {vn:.2f}: '''\n"
                       "         + ('el ritmo actual agotaría el colchón antes de terminar la cadena. ' if vn > bpi[dn] else\n"
                       "            'el ritmo actual es sostenible: el colchón alcanzaría hasta el final. ')\n"
                       "         + ({'verde': 'Aunque el informe está en zona verde, conviene preparar ya un plan de recuperación y vigilar el siguiente informe.',\n"
                       "             'amarilla': f'El informe está en zona amarilla, así que el plan de recuperación ya debe estar preparado; con esta tendencia '\n"
                       "                         f'habrá que ejecutarlo hacia el {x_roja:.0f} % de cadena completada, al entrar en la zona roja, o antes si el '\n"
                       "                         'siguiente informe confirma la tendencia.'}[zona[-1]] if vn > bpi[dn] else\n"
                       "            {'verde': 'Con el informe en zona verde basta con seguir vigilando la tendencia.',\n"
                       "             'amarilla': 'Con el informe en zona amarilla se mantiene preparado el plan de recuperación, sin ejecutarlo todavía.'}[zona[-1]])\n"
                       "         + ' La tendencia es una extrapolación: el siguiente informe puede confirmarla o desmentirla.')"))
    zonas = [s["zona"] for s in segs]
    celdas += _autoevaluacion(_p_tendencia(plan, segs), f"Cadena crítica — tendencia del gráfico de fiebre (cadena de {T} días, CP de {CP})")
    sub = f"cadena de {T} días, CP de {CP}; cuatro informes ({' → '.join(zonas)})"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variante 13 — Valor ganado (SPI) frente a gestión por colchones (BPI)
# ---------------------------------------------------------------------------
VALOR_CC = r'''
def detalle_pv_ev(plan, datos, dia, terminadas, trabajados, en_curso):
    """PV con el plan de cadena crítica como línea base (reparto uniforme en la duración agresiva desde el inicio
    del plan) y EV con el avance físico (días trabajados / (trabajados + restantes) para las que están en curso)."""
    filas = {}
    for a in sorted(plan.agresiva.index):
        bac, ini, d = float(datos.loc[a, 'presupuesto']), int(plan.inicios[a]), int(plan.agresiva[a])
        hecho_plan = min(max(dia - ini, 0), d)
        pv = bac * hecho_plan / d
        if a in terminadas:
            pct, txt = 1.0, '100 %'
        elif a in en_curso:
            w, r = trabajados[a], en_curso[a]
            pct, txt = w / (w + r), f'{w}/({w} + {r}) = {100 * w / (w + r):.1f} %'
        else:
            pct, txt = 0.0, '0 %'
        filas[a] = {'cadena': 'sí' if a in plan.cadena else '', 'presupuesto': f'{bac:g}',
                    'plan': f'[{ini}, {ini + d})', 'PV = presupuesto·días previstos/duración': f'{bac:g}·{hecho_plan}/{d} = {pv:.2f}',
                    'avance físico': txt, 'EV = presupuesto·avance': f'{bac:g}·{pct:.3f} = {bac * pct:.2f}',
                    'PV_i': pv, 'EV_i': bac * pct}
    return pd.DataFrame(filas).T


def indices(tabla, filtro=None):
    t = tabla if filtro is None else tabla[filtro]
    pv, ev = float(t['PV_i'].sum()), float(t['EV_i'].sum())
    return pv, ev, (ev / pv if pv else float('nan'))
'''.strip("\n")


def _spi(rng):
    ns = {"pd": pd}
    exec(VALOR_CC, ns)
    escenario = str(rng.choice(["adelanto", "retraso"]))

    def acepta(info, plan, df):
        a = plan.alimentacion
        retrasadas = int((plan.inicios - plan.inicio_temprano).ge(2).sum())
        return len(a) >= 2 and len(plan.cadena) >= 4 and retrasadas >= 2
    hallado = None
    for _ in range(300):
        info, plan, df = _buscar(rng, acepta)
        coste = pd.Series(rng.integers(1, 5, size=len(df)), index=df.index)
        df = df.assign(presupuesto=(df["agresiva"] * coste).astype(int))
        for _ in range(60):
            real = {}
            for a in sorted(plan.agresiva.index):
                med, p90 = float(plan.agresiva[a]), float(plan.segura[a])
                sigma = math.log(p90 / med) / Z90 if p90 > med else 0.0
                real[a] = min(max(1, int(round(med * math.exp(sigma * rng.standard_normal())))), math.ceil(1.5 * p90))
            extra = {a: int(rng.integers(0, 3)) for a in sorted(plan.agresiva.index)}
            ini, fin = {}, {}
            for a in plan.orden:
                st = max([0] + [fin[p] for p in plan.predecesoras[a]])
                if a not in plan.cadena and escenario == "retraso":
                    st = max(st, int(plan.inicios[a]) + extra[a])
                ini[a], fin[a] = st, st + real[a]
            T_real = max(fin.values())
            for dia in range(max(3, int(0.3 * plan.duracion_cadena)), min(T_real, int(0.75 * plan.duracion_cadena) + 1)):
                if not any(ini[c] < dia < fin[c] for c in plan.cadena):
                    continue
                term = {a: fin[a] for a in plan.orden if fin[a] <= dia}
                curso = {a: fin[a] - dia for a in plan.orden if ini[a] < dia < fin[a]}
                trab = {a: dia - ini[a] for a in curso}
                s = plan.seguimiento(dia, term, curso)
                tabla = ns["detalle_pv_ev"](plan, df, dia, term, trab, curso)
                pv, ev, spi = ns["indices"](tabla)
                _, _, spi_c = ns["indices"](tabla, tabla["cadena"] == "sí")
                _, _, spi_a = ns["indices"](tabla, tabla["cadena"] != "sí")
                if not (pv > 0) or not np.isfinite(spi_a):
                    continue
                if escenario == "adelanto":
                    ok = (1.0 <= spi <= 1.4 and s["zona"] != "verde" and spi_c < 1.0 and spi_a > 1.0
                          and s["pct_consumo"] <= 100)
                else:
                    ok = (0.7 <= spi <= 0.95 and s["zona"] == "verde" and spi_c >= 0.98 and spi_a < 0.9
                          and s["pct_consumo"] >= 0 and all(x["consumido"] <= x["adoptado"] for _, x in s["alimentacion"].iterrows()))
                if ok:
                    hallado = (info, plan, df, dia, term, trab, curso, spi, s)
                    break
            if hallado:
                break
        if hallado:
            break
    if not hallado:
        raise RuntimeError("No se encontró un informe en el que SPI y BPI discrepen")
    info, plan, df, dia, term, trab, curso, spi, s = hallado
    df = df.assign(inicio_plan=plan.inicios.reindex(df.index).astype(int))
    estado = {a: (f"terminó el día {term[a]}" if a in term else
                  (f"lleva {trab[a]} días, le quedan {curso[a]}" if a in curso else "—")) for a in df.index}
    cad = " – ".join(plan.cadena)
    celdas = [md("# Enunciado"),
              md("Un proyecto se ha programado con el método de la cadena crítica (estimaciones agresivas, colchones por "
                 "cortar y pegar redondeados al día superior, actividades de alimentación lo más tarde posible). La cadena "
                 f"crítica es {cad}, de {plan.duracion_cadena} días; el colchón del proyecto es de "
                 f"{plan.colchon_proyecto} días y la fecha comprometida, el día {plan.compromiso}. El cuadro recoge las "
                 "estimaciones, el inicio de cada actividad en el plan (días desde el instante 0) y su presupuesto (miles "
                 "de euros). La oficina de proyectos sigue el proyecto con el valor ganado, tomando como línea base el "
                 "plan de cadena crítica (el presupuesto de cada actividad se reparte uniformemente a lo largo de su "
                 "duración agresiva, desde su inicio en el plan) y midiendo el avance físico de una actividad en curso "
                 "como días trabajados / (días trabajados + días restantes)."),
              _datos(df),
              md(f"El día {dia} se recibe el informe de avance del cuadro. Las actividades de la cadena empiezan en cuanto "
                 "terminan sus predecesoras; para prever las que no han empezado se usan las duraciones agresivas y, para "
                 "las de alimentación, no antes de su inicio en el plan."),
              code('#| tbl-cap: "Informe de avance"\n#| label: tbl-informe\n'
                   f"informe = dict(dia={dia}, terminadas={term!r}, trabajados={trab!r}, en_curso={curso!r})\n"
                   f"pd.DataFrame({{'día {dia}': {estado!r}}})"),
              preguntas([
                  "Calcule el valor planificado (PV) y el valor ganado (EV) en el día del informe, la variación del "
                  "cronograma (SV = EV − PV) y el índice de desempeño del cronograma (SPI = EV/PV). ¿Qué concluiría un "
                  "informe basado sólo en el valor ganado?",
                  "Calcule el fin previsto de la cadena, el colchón del proyecto consumido, la cadena completada, el BPI "
                  "y la zona del gráfico de fiebre (zonas 10 → 70 % y 30 → 90 %). ¿Qué concluye la gestión por colchones?",
                  "Calcule el SPI de las actividades de la cadena crítica y el de las actividades de alimentación por "
                  "separado y explique por qué discrepan los dos diagnósticos.",
                  "¿Qué indicador debe guiar las decisiones del jefe de proyecto para proteger la fecha comprometida? "
                  "¿Qué limitaciones tiene aquí el SPI?"]),
              md("# Resolución"),
              code(ARRANQUE + "\ndatos_proyecto['duration'] = datos_proyecto['agresiva']"),
              code(AYUDAS + "\n\n\n" + VALOR_CC)]
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las estimaciones agresivas y la cadena crítica", label="fig-pert")
    celdas.append(code("plan = proyecto.plan_cadena_critica(datos_proyecto, agresiva='agresiva', segura='segura')\n"
                       f"assert plan.cadena == {plan.cadena!r} and plan.colchon_proyecto == {plan.colchon_proyecto}\n"
                       "T, CP = plan.duracion_cadena, plan.colchon_proyecto\n"
                       "dia, terminadas, trabajados, en_curso = informe['dia'], informe['terminadas'], informe['trabajados'], informe['en_curso']"))
    celdas.append(code('#| fig-cap: "Gantt del plan de cadena crítica (línea base del valor ganado)"\n#| label: fig-ganttCCPM\n'
                       f"pinta_tikz(plan.gantt_tikz(params={GANTT_CCPM}))"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Valor planificado y valor ganado de cada actividad (miles de euros)"\n#| label: tbl-pvev\n'
                       "tabla_vg = detalle_pv_ev(plan, datos_proyecto, dia, terminadas, trabajados, en_curso)\n"
                       "tabla_vg.drop(columns=['PV_i', 'EV_i'])"))
    celdas.append(code("PV, EV, SPI = indices(tabla_vg)\n"
                       "Markdown(f'PV = {PV:.2f}; EV = {EV:.2f}; SV = EV − PV = {EV - PV:+.2f} miles de euros; **SPI = {EV:.2f}/{PV:.2f} = {SPI:.3f}**. '\n"
                       "         + ('Según el valor ganado el proyecto va adelantado: se ha hecho más trabajo del previsto a estas alturas.' if SPI >= 1 else\n"
                       "            f'Según el valor ganado el proyecto va retrasado: se ha hecho sólo el {100 * SPI:.1f} % del trabajo previsto a estas alturas.'))"))
    celdas += apartado(2)
    celdas.append(code("s = plan.seguimiento(dia, terminadas, en_curso)\n"
                       "display(sin_decimales(s['detalle'][['estado', 'agresiva', 'completado', 'restante', 'fin previsto']]))\n"
                       "resumen = resumen_informes(plan, [s])\n"
                       "resumen"))
    celdas.append(code('#| fig-cap: "Gráfico de fiebre del informe"\n#| label: fig-fiebre\n'
                       "pinta_tikz(grafico_fiebre_tikz([s]))"))
    celdas.append(code("Markdown(f\"Según la gestión por colchones el proyecto está en zona **{s['zona']}**: se ha consumido el {s['pct_consumo']:.1f} % del \"\n"
                       "         f\"colchón con el {s['pct_avance']:.1f} % de la cadena completada (BPI {s['bpi']:.2f}). \"\n"
                       "         + ('La fecha comprometida está amenazada.' if s['zona'] != 'verde' else 'La fecha comprometida no corre peligro.'))"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "SPI de la cadena crítica y de la alimentación"\n#| label: tbl-spiPartes\n'
                       "partes = {'cadena crítica': tabla_vg['cadena'] == 'sí', 'alimentación': tabla_vg['cadena'] != 'sí', 'todo el proyecto': None}\n"
                       "spis = {k: indices(tabla_vg, f) for k, f in partes.items()}\n"
                       "pd.DataFrame({k: {'PV': round(pv, 2), 'EV': round(ev, 2), 'SPI = EV/PV': f'{ev:.2f}/{pv:.2f} = {spi:.3f}'} for k, (pv, ev, spi) in spis.items()})"))
    if escenario == "adelanto":
        celdas.append(code("Markdown(f\"El SPI global ({spis['todo el proyecto'][2]:.3f}) mezcla dos cosas: la alimentación va por delante del plan \"\n"
                           "         f\"(SPI {spis['alimentación'][2]:.3f}): sus actividades se están haciendo antes de lo que prevé el plan, que las programa \"\n"
                           "         \"lo más tarde posible; la \"\n"
                           "         f\"cadena crítica va por detrás (SPI {spis['cadena crítica'][2]:.3f}). El trabajo adelantado en la alimentación suma valor \"\n"
                           "         \"ganado, pero no acerca la fecha de entrega, que sólo depende de la cadena y de sus colchones; el retraso de la cadena, \"\n"
                           "         \"en cambio, consume colchón del proyecto día a día.\")"))
    else:
        celdas.append(code("Markdown(f\"El SPI global ({spis['todo el proyecto'][2]:.3f}) mezcla dos cosas: la cadena crítica va al ritmo del plan \"\n"
                           "         f\"(SPI {spis['cadena crítica'][2]:.3f}) y la alimentación va retrasada (SPI {spis['alimentación'][2]:.3f}), pero su \"\n"
                           "         \"retraso cabe en sus colchones de alimentación: por eso no llega a la cadena ni consume colchón del proyecto. El valor \"\n"
                           "         \"ganado pesa igual un día de retraso en cualquier actividad; la gestión por colchones sólo cuenta el que amenaza la entrega.\")"))
        celdas.append(code('#| tbl-cap: "Consumo de los colchones de alimentación"\n#| label: tbl-consumoCA\n'
                           "ca = s['alimentacion'].set_index('colchon')[['desde', 'antes_de', 'fin_previsto', 'adoptado', 'consumido']]\n"
                           "ca.index.name = None\n"
                           "sin_decimales(ca.rename(columns={'antes_de': 'antes de', 'fin_previsto': 'fin previsto', 'adoptado': 'colchón (días)'}).astype(object))"))
    celdas += apartado(4)
    celdas.append(md("Para proteger la fecha comprometida manda la gestión por colchones: el BPI y el gráfico de fiebre "
                     "miran sólo lo que fija la entrega (la cadena crítica, y la alimentación cuando desborda su colchón) y "
                     "comparan lo que queda de colchón con lo que queda de cadena. La Guía del PMBOK presenta el BPI "
                     "precisamente como una señal pensada para proteger la fecha de entrega y para decidir las prioridades "
                     "del trabajo diario, a diferencia del SPI.\n\n"
                     "Limitaciones del SPI en este caso: (1) suma en dinero todas las actividades, críticas o no, de modo que "
                     "el trabajo hecho fuera de la cadena compensa el retraso de la cadena (o el retraso de la alimentación "
                     "dentro de su colchón parece un problema); (2) depende de la línea base: con las actividades de "
                     "alimentación lo más tarde posible, empezarlas antes eleva el EV sin cambiar la fecha de fin; (3) "
                     "tiende a 1 al final del proyecto aunque se termine tarde, porque al terminar EV = PV = presupuesto. "
                     "El valor ganado sigue siendo útil para controlar el coste (CPI) y el gasto, pero el plazo se gobierna "
                     "con los colchones."))
    celdas += _autoevaluacion(_p_spi(plan, df, dia, term, trab, curso, s, escenario, ns), f"Cadena crítica — BPI frente a SPI (día {dia})")
    sub = (f"día {dia}: SPI {spi:.2f} y zona {s['zona']} "
           + ("(alimentación adelantada, cadena retrasada)" if escenario == "adelanto" else "(alimentación retrasada dentro de su colchón)"))
    return _con_contexto(celdas, rng), sub


# ---------------------------------------------------------------------------
# Autoevaluación: preguntas de opción múltiple con los datos de cada ejercicio, exportables a GIFT
# (las calcula el generador sin consumir rng, para no alterar los datos de los cuadernos)
# ---------------------------------------------------------------------------
TEST_CELDA = r'''
import os, re


def a_gift(preguntas_test, titulo, semilla=1):
    """Preguntas en formato GIFT (Moodle): caracteres especiales escapados y opciones barajadas."""
    esc = lambda s: re.sub(r'([~=#{}:])', r'\\\1', str(s))
    r = np.random.default_rng(semilla)
    lineas = [f'$CATEGORY: {titulo}', '']
    for i, (q, ops, fb) in enumerate(preguntas_test, 1):
        cuerpo = ' '.join(('=' if j == 0 else '~') + esc(ops[j]) for j in r.permutation(len(ops)))
        lineas.append(f'::{esc(titulo)} P{i}::{esc(q)} {{{cuerpo} ####{esc(fb)}}}')
        lineas.append('')
    return '\n'.join(lineas)


gift = a_gift(preguntas_test, TITULO_TEST)
os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
with open(os.path.expanduser('~/temp/cuestionario_cadena_critica.gift'), 'w', encoding='utf8') as f:
    f.write(gift)
from pmule import muestra_test
muestra_test(preguntas_test)
'''.strip("\n")


def _autoevaluacion(preguntas_test, titulo):
    """Apartado final con el test: se muestra legible y se exporta a GIFT."""
    lineas = ["preguntas_test = ["]
    for q, ops, fb in preguntas_test:
        assert len(ops) == len(set(ops)) >= 3, (q, ops)
        lineas.append(f"    ({q!r},\n     {ops!r},\n     {fb!r}),")
    lineas += ["]", f"TITULO_TEST = {titulo!r}"]
    return [md("## Autoevaluación"),
            md("Preguntas de respuesta múltiple con los datos de este ejercicio (la correcta, marcada con ✅, es la "
               "primera de cada lista). La celda las exporta también en formato GIFT, con las opciones barajadas, a "
               "`~/temp/cuestionario_cadena_critica.gift`, para importarlas en Moodle."),
            code("\n".join(lineas) + "\n" + TEST_CELDA)]


def _coma(x, dec=0):
    return f"{x:.{dec}f}".replace(".", ",")


def _ops(*opciones):
    """Opciones sin repetir, la correcta primero, como mucho cuatro."""
    vistas = []
    for o in opciones:
        o = str(o)
        if o not in vistas:
            vistas.append(o)
        if len(vistas) == 4:
            break
    return vistas


def _ops_num(correcta, candidatas, dec=0, pre="", unidad="", paso=None):
    """Opciones numéricas: la correcta, los distractores y, si coinciden, valores vecinos."""
    paso = paso or (1 if dec == 0 else 5 * 10 ** -dec)
    f = lambda v: f"{pre}{_coma(v, dec)}{unidad}"
    vecinos = [correcta + paso, correcta - paso, correcta + 2 * paso, correcta - 2 * paso]
    return _ops(f(correcta), *[f(c) for c in candidatas], *[f(v) for v in vecinos])


def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _zona_ops(zona):
    return _ops(zona, *[z for z in ("verde", "amarilla", "roja") if z != zona], "no se puede saber sin el valor ganado")


def _p_multitarea(dur, bloque, cambio):
    ns = {"pd": pd}
    exec(MULTITAREA, ns)
    d = {f"T{i}": int(v) for i, v in enumerate(dur, 1)}
    fin_s, _ = ns["multitarea"](d)
    fin_m, _ = ns["multitarea"](d, bloque)
    fin_mc, t_mc = ns["multitarea"](d, bloque, cambio)
    fin_sc, _ = ns["multitarea"](d, None, cambio)
    k, total = len(dur), sum(d.values())
    n_cambios = int((t_mc["tarea"] == "cambio").sum())
    return [
        ("Sin multitarea y en el orden del cuadro, ¿cuál es el tiempo medio de terminación de las tareas?",
         _ops_num(float(fin_s.mean()), [float(fin_m.mean()), total / k, float(fin_s.max())], dec=2, unidad=" días"),
         f"Las tareas terminan en las sumas acumuladas de las duraciones ({', '.join(str(int(v)) for v in fin_s)}); "
         f"su media es {_coma(float(fin_s.mean()), 2)} días."),
        (f"Con multitarea en turnos de {bloque} días y sin pérdida al cambiar de tarea, ¿qué día termina la última tarea?",
         _ops_num(total, [total + bloque, int(fin_mc.max()), max(d.values()) * k], pre="día "),
         "El trabajo total es el mismo, así que la última tarea termina el mismo día que sin multitarea; lo que se "
         "retrasa son las demás."),
        (f"Con multitarea en turnos de {bloque} días y una pérdida de {cambio} día(s) en cada cambio de tarea, ¿qué día "
         "termina la última tarea?",
         _ops_num(int(fin_mc.max()), [total, int(fin_sc.max()), int(fin_mc.max()) + bloque], pre="día "),
         f"Hay {n_cambios} cambios de tarea, que añaden {n_cambios * cambio} días al trabajo: la persona produce menos."),
        ("¿Qué orden minimiza, sin multitarea, el tiempo medio de terminación?",
         ["de la tarea más corta a la más larga", "de la tarea más larga a la más corta",
          "el orden en que se asignaron las tareas", "repartir el tiempo en turnos iguales"],
         "Es la regla de la duración más corta primero: adelantar una tarea corta retrasa a la larga menos de lo "
         "que adelanta a la corta."),
    ]


def _margenes(plan):
    ms = [int(plan.margen[a]) for a in plan.cadena]
    return ms, math.ceil(0.5 * sum(ms) - 1e-9), math.sqrt(sum(m * m for m in ms))


def _p_colchon(plan):
    T = plan.duracion_cadena
    ms, cp, rz = _margenes(plan)
    rz_d = math.ceil(rz - 1e-9)
    return [
        ("Con el colchón del proyecto calculado por cortar y pegar, ¿qué fecha se compromete con el cliente?",
         _ops_num(T + cp, [T + sum(ms), plan.duracion_tradicional, T + rz_d], pre="día "),
         f"Cadena de {T} días más un colchón de 0,5·{sum(ms)} → {cp} días: día {T + cp}. Sumar todo el margen "
         f"retirado ({T + sum(ms)}) devolvería la protección repartida que el método quiere evitar."),
        ("¿Cuántos días tiene el colchón del proyecto por el método de la raíz de la suma de cuadrados?",
         _ops_num(rz_d, [sum(ms), cp, math.floor(rz)], unidad=" días"),
         f"√({' + '.join(f'{m}²' for m in ms)}) = {_coma(rz, 2)}, redondeado al día superior: {rz_d}."),
        ("Una actividad de la cadena tarda más que su estimación agresiva. ¿Qué debe hacer el jefe de proyecto?",
         ["nada especial: es lo esperado la mitad de las veces y se vigila el consumo del colchón",
          "pedir explicaciones al responsable por incumplir su estimación",
          "retrasar la fecha comprometida en lo que se haya pasado",
          "volver a las estimaciones seguras para el resto de actividades"],
         "La estimación agresiva es la mediana: se espera que la mitad de las actividades la superen. Se controla el "
         "colchón, no el cumplimiento de cada duración."),
    ]


def _p_dimensionado(plan):
    ms, cp, rz = _margenes(plan)
    rz_d = math.ceil(rz - 1e-9)
    s = rz / 1.2816
    return [
        ("Si cada actividad de la cadena es normal con la estimación segura como percentil 90, ¿cuánto vale la "
         "desviación típica de la cadena?",
         _ops_num(s, [sum(ms) / 1.2816, rz, sum(ms) / len(ms) / 1.2816], dec=2, unidad=" días"),
         f"σ = √(Σ márgenes²)/1,2816 = {_coma(rz, 3)}/1,2816 = {_coma(s, 2)} días: se suman las varianzas, no las desviaciones."),
        ("¿Con qué probabilidad se termina la cadena en su duración más el colchón de cortar y pegar?",
         _ops_num(_Phi(cp / s), [_Phi(rz_d / s), 0.9, 0.5], dec=3),
         f"P = Φ(CP/σ) = Φ({cp}/{_coma(s, 3)}) = {_coma(_Phi(cp / s), 3)}."),
        ("Para una cadena de n actividades con el mismo margen, ¿cuándo da cortar y pegar un colchón mayor que la "
         "raíz de la suma de cuadrados?",
         ["con más de 4 actividades", "con más de 2 actividades", "con más de 9 actividades",
          "nunca: siempre da un colchón menor"],
         "n·m/2 = √n·m cuando √n = 2, es decir, n = 4; a partir de ahí cortar y pegar crece más deprisa."),
    ]


def _fila_ca(plan, preferir_no_cabe=True):
    al = plan.alimentacion
    if preferir_no_cabe and (al["requerido"] > al["disponible"]).any():
        return al[al["requerido"] > al["disponible"]].iloc[0], al[al["requerido"] > al["disponible"]].index[0]
    al = al[al["adoptado"] > 0] if (al["adoptado"] > 0).any() else al
    return al.iloc[0], al.index[0]


def _p_ca(plan):
    f, n = _fila_ca(plan)
    ms = [int(x) for x in str(f["margenes"]).split(" + ")]
    return (f"¿Cuántos días de colchón de alimentación se colocan en el arco {f['desde']} → {f['antes_de']} ({n})?",
            _ops_num(int(f["adoptado"]), [int(f["requerido"]), int(f["disponible"]), sum(ms)], unidad=" días"),
            f"Colchón requerido {int(f['requerido'])} (cadena de alimentación {f['cadena_alimentacion']}); holgura "
            f"disponible {int(f['disponible'])}; se adopta el menor: {int(f['adoptado'])}.")


def _p_alimentacion(plan):
    preguntas = [_p_ca(plan)]
    for n, f in plan.alimentacion.iterrows():
        x, adopt = f["desde"], int(f["adoptado"])
        entrada = int(plan.inicios[f["antes_de"]]) if f["antes_de"] != "fin" else int(plan.duracion_cadena)
        d = int(plan.agresiva[x])
        if adopt > 0 and int(plan.inicios[x]) + d == entrada - adopt and int(plan.inicios[x]) > int(plan.inicio_temprano[x]):
            preguntas.append((
                f"En el plan de cadena crítica, ¿qué día empieza la actividad de alimentación {x}?",
                _ops_num(int(plan.inicios[x]), [int(plan.inicio_temprano[x]), entrada - d, int(plan.inicios[x]) - adopt], pre="día "),
                f"Lo más tarde posible con su colchón delante: termina cuando empieza {n}, {entrada} − {adopt} = "
                f"{entrada - adopt}, y empieza el día {entrada - adopt} − {d} = {int(plan.inicios[x])} (su inicio más "
                f"temprano sería el {int(plan.inicio_temprano[x])})."))
            break
    preguntas.append((
        "¿Por qué se programan las actividades de alimentación lo más tarde posible?",
        ["para no comprometer recursos ni dinero antes de tiempo, dejando el margen en su colchón",
         "para que pasen a formar parte de la cadena crítica",
         "porque así la cadena crítica es más corta",
         "porque el colchón de alimentación se suma al colchón del proyecto"],
        "Se retrasa el gasto y el trabajo en curso, y el margen queda a la vista en el colchón en lugar de "
        "perderse en una holgura."))
    return preguntas


def _p_recursos(plan, info):
    T = plan.duracion_cadena
    preguntas = [
        ("¿Cuánto dura la cadena crítica teniendo en cuenta los equipos (estimaciones agresivas)?",
         _ops_num(T, [int(info["T"]), plan.duracion_tradicional, T + 1], unidad=" días"),
         f"Sin los equipos la red dura {int(info['T'])} días; los enlaces de recurso la alargan hasta {T}.")]
    rec = plan.enlaces_cadena[plan.enlaces_cadena["motivo"] == "recurso"]
    if len(rec):
        e = rec.iloc[0]
        preguntas.append((
            f"En la cadena crítica, ¿qué tipo de enlace une {e['de']} con {e['a']}?",
            [f"de recurso: las dos las hace el equipo {e['recurso']}", "de precedencia: está en el cuadro de prelaciones",
             "una actividad ficticia del grafo PERT", "un colchón de alimentación"],
            "El cuadro de prelaciones no los relaciona: los encadena el equipo, que no puede hacer las dos a la vez."))
    av = plan.colchones_recursos.iloc[0]
    otras = [a for a in plan.cadena if a not in set(plan.colchones_recursos["antes_de"])]
    otras += [a for a in plan.agresiva.index if a not in plan.cadena]
    preguntas.append((
        "¿Antes de qué actividad de la cadena se coloca un colchón de recursos?",
        _ops(av["antes_de"], *otras),
        f"{av['antes_de']} usa el equipo {av['recursos']}, que no usaba {av['despues_de']}, su predecesora en la "
        "cadena: hay que avisar al equipo antes de que esta termine."))
    preguntas.append((
        "¿Qué es un colchón de recursos?",
        ["un aviso al equipo para que esté preparado, sin tiempo en el cronograma",
         "unos días que se añaden antes de la actividad", "un equipo de reserva contratado por si acaso",
         "la holgura del equipo entre dos actividades"],
        "Protege la disponibilidad del recurso; añadir tiempo alargaría la cadena sin protegerla."))
    return preguntas


def _p_plan_completo(plan):
    T = plan.duracion_cadena
    ms = [int(plan.margen[a]) for a in plan.cadena]
    otro = "raiz" if plan.metodo == "cortar_pegar" else "cortar_pegar"
    cp_otro = math.ceil(plan.calculo_colchon_proyecto[otro] - 1e-9)
    trad = plan.duracion_tradicional
    return [
        ("¿Qué fecha se compromete con el cliente?",
         _ops_num(plan.compromiso, [trad, T + sum(ms), T + cp_otro], pre="día "),
         f"Duración de la cadena con los equipos ({T}) más el colchón del proyecto ({plan.colchon_proyecto}): día "
         f"{plan.compromiso}."),
        ("¿Cuántos días se ganan respecto al cronograma tradicional (estimaciones seguras, misma secuencia de los equipos)?",
         _ops_num(trad - plan.compromiso, [trad - T, int(plan.margen.sum()) - plan.colchon_proyecto, 0], unidad=" días"),
         f"{trad} − {plan.compromiso} = {trad - plan.compromiso} días."),
        _p_ca(plan),
    ]


def _p_seguimiento(plan, segs):
    T = plan.duracion_cadena
    s1, s = segs[0], segs[-1]
    det1 = s1["detalle"]
    sin_restante = s1["dia"] + float(det1.loc[det1["estado"] != "terminada", "agresiva"].sum())
    det = s["detalle"]
    pa, pc = s["pct_avance"], s["pct_consumo"]
    bruto = 100 * (T - float(det["restante"].sum())) / T
    return [
        (f"En el informe del día {s1['dia']}, ¿cuál es la fecha prevista de fin de la cadena?",
         _ops_num(float(s1["fin_previsto"]), [T, plan.compromiso, sin_restante], pre="día "),
         "Fecha del informe más la duración restante de la actividad en curso más las duraciones agresivas de las "
         f"pendientes: día {_coma(float(s1['fin_previsto']))}."),
        (f"En el informe del día {s['dia']}, ¿qué porcentaje de la cadena crítica está completado?",
         _ops_num(pa, [bruto, 100 * s["dia"] / T, pc], dec=1, unidad=" %"),
         f"(duración de la cadena − duración restante)/duración = {_coma(pa, 1)} %; de una actividad en curso no se "
         "cuenta como restante más que su duración agresiva."),
        (f"¿Cuál es el BPI en el informe del día {s['dia']}?",
         _ops_num((100 - pc) / (100 - pa), [pc / pa if pa else 0.0, (100 - pa) / (100 - pc) if pc < 100 else 0.0, 1 - pc / 100],
                  dec=2),
         f"BPI = % de colchón restante / % de cadena restante = (100 − {_coma(pc, 1)})/(100 − {_coma(pa, 1)})."),
        (f"¿En qué zona del gráfico de fiebre está el informe del día {s['dia']}?",
         _zona_ops(s["zona"]),
         f"Límite verde 10 + 0,6·{_coma(pa, 1)} = {_coma(s['limite_verde'], 1)} %; límite rojo "
         f"30 + 0,6·{_coma(pa, 1)} = {_coma(s['limite_roja'], 1)} %; consumo {_coma(pc, 1)} %."),
    ]


def _p_monte_carlo(plan, N, semilla):
    T, CP = plan.duracion_cadena, plan.colchon_proyecto
    t = plan.simular(N, semilla=semilla, regimen="relevos")
    p = {x: float((t <= x).mean()) for x in (T, plan.compromiso, plan.duracion_tradicional)}
    c90, c95, c50 = (math.ceil(float(np.percentile(t, q)) - T - 1e-9) for q in (90, 95, 50))
    return [
        ("Según la simulación, ¿con qué probabilidad se cumple la fecha comprometida?",
         _ops_num(p[plan.compromiso], [p[T], p[plan.duracion_tradicional], 0.9], dec=2),
         f"Proporción de réplicas que terminan en {plan.compromiso} días o menos: {_coma(p[plan.compromiso], 3)}."),
        ("¿Qué colchón del proyecto haría falta para cumplir con probabilidad 0,90?",
         _ops_num(c90, [CP, c95, c50], unidad=" días"),
         f"Percentil 90 de la duración simulada menos la duración de la cadena ({T}), redondeado al día superior: {c90}."),
        ("¿Por qué el proyecto completo cumple la fecha con menos probabilidad que la cadena crítica aislada?",
         ["por el sesgo de convergencia: en las uniones manda el camino que llega más tarde",
          "porque el colchón de cortar y pegar está mal calculado",
          "porque la simulación usa duraciones lognormales",
          "porque las actividades de alimentación empiezan antes de lo previsto"],
         "Los colchones de alimentación existen para que esa diferencia sea pequeña."),
    ]


def _p_parkinson(plan, N, semilla, k):
    t_trad = plan.simular(N, semilla=semilla, regimen="tradicional")
    t_rel = plan.simular(N, semilla=semilla, regimen="relevos")
    p_trad = float((t_trad <= plan.duracion_tradicional).mean())
    p_rel = float((t_rel <= plan.compromiso).mean())
    return [
        (f"Si cada una de las {k} actividades de la ruta crítica tradicional cumple su estimación segura con "
         "probabilidad 0,9, de forma independiente, ¿qué probabilidad hay de que ninguna la supere?",
         _ops_num(0.9 ** k, [0.9, max(0.0, 1 - 0.1 * k), 0.9 ** (k - 1)], dec=3),
         f"0,9^{k} = {_coma(0.9 ** k, 3)}."),
        ("En la simulación del régimen tradicional con la ley de Parkinson, ¿con qué probabilidad se cumple la fecha "
         "prometida?",
         _ops_num(p_trad, [p_rel, 0.9, 1.0], dec=2),
         f"{_coma(p_trad, 3)}: como mucho 0,9^{k}, porque cualquier exceso en la ruta crítica se transmite y ningún "
         "adelanto lo compensa."),
        ("¿Qué describe la ley de Parkinson?",
         ["el trabajo se alarga hasta ocupar el tiempo disponible: quien termina antes no lo comunica",
          "con holgura, las actividades se empiezan en el último momento",
          "en las uniones de caminos manda el que llega más tarde",
          "repartir a una persona entre varias tareas las retrasa todas"],
         "Las otras opciones son el síndrome del estudiante, el sesgo de convergencia y la multitarea."),
    ]


def _p_tambor(ns, df, q, DB, an, tn, bn, cpn, nuevo):
    esc = ns["escalonar"](df, DB, 0.25)
    u = esc.iloc[-1]
    p_u = esc.index[-1]
    libre = int(u["libre_despues"])
    s_n = max(an + DB, libre)
    comp = s_n + tn + bn + cpn
    fA, _ = ns["tambor_sin_escalonar"](df, q)
    fB, _ = ns["tambor_sin_escalonar"](df, None)
    p1 = df.index[0]
    a1, t1, b1, cp1 = (int(df.loc[p1, c]) for c in ("antes", "tambor", "despues", "CP"))
    return [
        (f"En el programa escalonado, ¿qué día se lanza {p_u}?",
         _ops_num(int(u["lanzamiento"]), [int(u["inicio_tambor"]) - int(u["a"]), 0, int(u["inicio_tambor"])], pre="día "),
         f"Entra en el tambor el día {int(u['inicio_tambor'])}; se lanza {int(u['inicio_tambor'])} − {DB} (colchón del "
         f"tambor) − {int(u['a'])} (antes) = {int(u['lanzamiento'])}."),
        (f"¿Qué fecha puede comprometerse para el proyecto nuevo {nuevo}?",
         _ops_num(comp, [an + tn + bn + cpn, comp - int(u["colchon_capacidad"]), s_n + tn + bn], pre="día "),
         f"El tambor queda libre el día {libre}; {nuevo} entra el día máx({an} + {DB}; {libre}) = {s_n} y se compromete "
         f"para {s_n} + {tn} + {bn} + {cpn} = {comp}."),
        (f"Con todos los proyectos lanzados el día 0 y multitarea en el tambor, ¿qué día termina {p1}?",
         _ops_num(int(fA[p1]) + b1, [a1 + t1 + b1, int(fB[p1]) + b1, int(fA[p1])], pre="día "),
         f"Su tarea del tambor termina el día {int(fA[p1])}, en su último turno; después quedan {b1} días."),
        ("¿Para qué sirve el colchón de capacidad?",
         ["para que el retraso de una tarea del tambor no arrastre a los proyectos siguientes",
          "para proteger la fecha de entrega de cada proyecto", "para que el tambor pueda hacer multitarea",
          "para lanzar antes los proyectos"],
         "Las fechas de los proyectos siguientes se comprometen contando con su turno en el tambor; la fecha de "
         "entrega la protege el colchón del proyecto."),
    ]


def _p_consumo(plan, dia, terminadas, en_curso, ns):
    s, fin, cons = _penetracion(plan, dia, terminadas, en_curso, ns)
    solo = ns["fin_solo_cadena"](plan, dia, terminadas, en_curso)
    T = plan.duracion_cadena
    desb = cons[cons["pasa a la cadena"] > 0].iloc[0]
    n = cons[cons["pasa a la cadena"] > 0].index[0]
    consumido = int(str(desb["consumido = máx(0, fin previsto − inicio del colchón)"]).split(" = ")[-1])
    pct_ca = cons["% consumido"].astype(float)
    prioridad = {}
    for a in sorted(en_curso):
        cs = ns["colchones_que_protegen"](plan, a)
        prioridad[a] = max(s["pct_consumo"] if c == "CP" else float(pct_ca[c]) for c in cs)
    orden = sorted(prioridad, key=lambda a: -prioridad[a])
    resto = [a for a in orden[1:]] + [a for a in plan.cadena if a not in prioridad] + [a for a in sorted(plan.agresiva.index) if a not in prioridad]
    return [
        (f"¿Cuántos días pasan a la cadena crítica desde el colchón de alimentación {n} ({desb['arco']})?",
         _ops_num(int(desb["pasa a la cadena"]), [consumido, int(desb["colchón (días)"]), 0], unidad=" días"),
         f"Consume {consumido} días de un colchón de {int(desb['colchón (días)'])}: el exceso, "
         f"{int(desb['pasa a la cadena'])}, retrasa a la actividad de la cadena en la que desemboca."),
        ("¿Cuántos días del colchón del proyecto se consumen por culpa de las cadenas de alimentación?",
         _ops_num(float(s["fin_previsto"]) - solo, [float(s["consumo"]), solo - T, int(desb["pasa a la cadena"]) + 1], unidad=" días"),
         f"Fin previsto {_coma(float(s['fin_previsto']))}; si la alimentación no la retrasara, la cadena terminaría el "
         f"día {solo}: {_coma(float(s['fin_previsto']) - solo)} día(s) se deben a la alimentación."),
        ("Según la gestión por colchones, ¿qué actividad en curso debe reforzarse primero?",
         _ops(orden[0], *resto),
         f"Es la de colchón más consumido en porcentaje ({_coma(prioridad[orden[0]], 1)} %)."),
    ]


def _p_tendencia(plan, segs):
    T, CP = plan.duracion_cadena, plan.colchon_proyecto
    xs = [100 * int(round(s["avance"])) / T for s in segs]
    ys = [100 * (int(round(s["fin_previsto"])) - T) / CP for s in segs]
    dias = [s["dia"] for s in segs]
    v = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y100 = ys[-1] + v * (100 - xs[-1])
    bpi = (100 - ys[-1]) / (100 - xs[-1])
    lv, lr = 10 + 0.6 * xs[-1], 30 + 0.6 * xs[-1]
    zona = "verde" if ys[-1] < lv else ("roja" if ys[-1] >= lr else "amarilla")
    return [
        ("¿Cuál es la velocidad de consumo del colchón en el último tramo (entre los dos últimos informes)?",
         _ops_num(v, [(ys[-1] - ys[-2]) / (dias[-1] - dias[-2]), ys[-1] / xs[-1], bpi], dec=2),
         f"({_coma(ys[-1], 1)} − {_coma(ys[-2], 1)})/({_coma(xs[-1], 1)} − {_coma(xs[-2], 1)}) = {_coma(v, 2)} puntos de "
         "colchón por punto de cadena."),
        ("Si se mantiene esa velocidad, ¿qué porcentaje del colchón se habrá consumido al terminar la cadena?",
         _ops_num(y100, [100 * ys[-1] / xs[-1], ys[-1] + 0.6 * (100 - xs[-1]), 100.0], dec=1, unidad=" %"),
         f"{_coma(ys[-1], 1)} + {_coma(v, 2)}·(100 − {_coma(xs[-1], 1)}) = {_coma(y100, 1)} %."),
        ("Si la velocidad de consumo es mayor que el BPI del último informe, ¿qué ocurrirá si se mantiene?",
         ["el colchón se agotará antes de terminar la cadena", "el proyecto entrará de inmediato en la zona roja",
          "el proyecto terminará antes de la fecha comprometida", "habrá que recalcular la cadena crítica"],
         "El BPI, (100 − y)/(100 − x), es la velocidad media que el colchón puede soportar hasta el final."),
        (f"¿En qué zona está el informe del día {dias[-1]}?",
         _zona_ops(zona),
         f"Límites {_coma(lv, 1)} % y {_coma(lr, 1)} % para un {_coma(xs[-1], 1)} % de cadena; consumo {_coma(ys[-1], 1)} %."),
    ]


def _p_spi(plan, df, dia, term, trab, curso, s, escenario, ns):
    tabla = ns["detalle_pv_ev"](plan, df, dia, term, trab, curso)
    pv, ev, spi = ns["indices"](tabla)
    _, _, spi_c = ns["indices"](tabla, tabla["cadena"] == "sí")
    _, _, spi_a = ns["indices"](tabla, tabla["cadena"] != "sí")
    if escenario == "adelanto":
        motivo = (["el trabajo adelantado en la alimentación compensa en el SPI el retraso de la cadena crítica",
                   "la cadena crítica va adelantada", "el colchón del proyecto está mal dimensionado",
                   "el SPI no incluye las actividades de la cadena crítica"],
                  "El SPI suma en dinero todas las actividades; la fecha sólo depende de la cadena y de sus colchones.")
    else:
        motivo = (["la alimentación va retrasada, pero su retraso cabe en sus colchones de alimentación",
                   "la cadena crítica va retrasada", "el colchón del proyecto ya está agotado",
                   "el SPI sólo mide la cadena crítica"],
                  "Un retraso de la alimentación que no desborda su colchón no llega a la cadena ni consume colchón del proyecto.")
    return [
        ("¿Cuánto vale el SPI de todo el proyecto en el día del informe?",
         _ops_num(spi, [spi_c, spi_a, pv / ev if ev else 0.0], dec=2),
         f"SPI = EV/PV = {_coma(ev, 2)}/{_coma(pv, 2)} = {_coma(spi, 3)}."),
        ("¿En qué zona del gráfico de fiebre está el informe?",
         _zona_ops(s["zona"]),
         f"Consumo {_coma(s['pct_consumo'], 1)} % con un {_coma(s['pct_avance'], 1)} % de cadena completada."),
        ("¿Por qué discrepan el SPI y la gestión por colchones en este informe?", motivo[0], motivo[1]),
        ("¿Qué indicador debe guiar las decisiones para proteger la fecha comprometida?",
         ["el BPI y el gráfico de fiebre", "el SPI de todo el proyecto", "el SPI de las actividades de alimentación",
          "el CPI del valor ganado"],
         "El BPI mira sólo lo que fija la entrega; el valor ganado sigue siendo útil para controlar el coste."),
    ]


VARIANTES = [
    Variante("multitarea", 1, "turnos de una persona con varias tareas: fines con y sin multitarea, pérdida en cada cambio y orden que minimiza el tiempo medio",
             _multitarea, grafo="pert"),
    Variante("colchon-proyecto", 1, "estimaciones segura y agresiva: cronograma tradicional, cadena crítica, colchón del proyecto por cortar y pegar y raíz, fecha comprometida",
             _colchon_proyecto),
    Variante("dimensionado-colchon", 2, "cortar y pegar frente a raíz de la suma de cuadrados: probabilidad de cumplir con duraciones normales y crecimiento con el número de actividades",
             _dimensionado),
    Variante("colchones-alimentacion", 2, "colchones de alimentación (cadena de alimentación más larga, holgura disponible, colchón que no cabe) y programación lo más tarde posible; Gantt",
             _alimentacion),
    Variante("cadena-con-recursos", 2, "equipos compartidos: solapes, regla de prioridad, enlaces de recurso, red nivelada, cadena crítica distinta de la ruta crítica, colchones de recursos",
             _con_recursos),
    Variante("plan-completo", 3, "plan completo con equipos: red nivelada, cadena, colchón del proyecto y de alimentación (cortar y pegar o raíz), Gantt y comparación con el cronograma tradicional",
             _plan_completo),
    Variante("seguimiento-fiebre", 2, "tres informes de avance: fin previsto, colchón consumido, cadena completada, BPI, zona y gráfico de fiebre",
             _seguimiento),
    Variante("monte-carlo-colchon", 3, "Monte Carlo del plan por relevos: probabilidad de cada fecha, colchón para el 90 y el 95 %, cadena aislada y sesgo de convergencia",
             _monte_carlo),
    Variante("parkinson-relevos", 3, "cronograma tradicional con ley de Parkinson y fechas fijas frente a relevos: probabilidad de cumplir, 0,9^k y por qué se pierden los márgenes",
             _parkinson),
    Variante("multiproyecto-tambor", 3, "varios proyectos y un recurso tambor: multitarea, sin multitarea y escalonamiento con colchón de capacidad y colchón del tambor; lanzamiento de un proyecto nuevo",
             _tambor, grafo="pert"),
    Variante("consumo-alimentacion", 2, "informe con un colchón de alimentación desbordado: previsión, consumo de cada colchón, parte del consumo del CP que viene de la alimentación y prioridad por el estado de los colchones",
             _consumo_alimentacion),
    Variante("tendencia-fiebre", 2, "cuatro informes: consumo, avance, BPI y zona; velocidad de consumo, previsión con la tendencia, entrada en la zona roja y el BPI como velocidad sostenible",
             _tendencia, grafo="pert"),
    Variante("bpi-frente-a-spi", 3, "el mismo informe con valor ganado (PV, EV, SPI con el plan de cadena crítica como línea base) y con el colchón (BPI, zona): por qué discrepan",
             _spi),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
