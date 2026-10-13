"""Colección 11 — Recursos: diagrama de cargas, nivelación, asignación con
límite, evaluación de desplazamientos y varios tipos de recurso."""
import contextlib
import io

import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="11_recursos", prefijo="Recursos",
            titulo="Nivelación y asignación de recursos",
            descripcion="Diagrama de cargas, nivelación por mínimos cuadrados, asignación con límite de "
                        "recursos (Wiest-Levy), evaluación de desplazamientos y varios tipos de recurso.",
            n_defecto=3)


def _cargas(info, df):
    with contextlib.redirect_stdout(io.StringIO()):
        g = info["proyecto"].copy()
        gantt_df, _ = g.gantt_cargas(df.copy(), "duration", "resources", report=False, tikz=False)
    total = gantt_df.data.loc["Total"].drop("H_total", errors="ignore").astype(float)
    return total


def _nivelar(info, df):
    with contextlib.redirect_stdout(io.StringIO()):
        g = info["proyecto"].copy()
        desp, cuad, _, gantt_df, _ = g.nivelar(df.copy(), "duration", "resources", report=False)
    return desp, gantt_df


def _base(k, rng):
    variante = (k - 1) % 3 + 1
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 10 <= i["T"] <= 20):
            return False
        rec = P.recursos(rng, list(i["D"].index), hi=4)
        df = P.tabla_base(i, duration=i["D"], resources=rec)
        if (i["calendario"]["H_total"] > 0).sum() < 3:
            return False
        try:
            total = _cargas(i, df)
            desp, gantt_niv = _nivelar(i, df)
        except Exception:
            return False
        if (desp["desplazamientos"] > 0).sum() < 2:
            return False
        if variante == 2:
            pico = float(total.max())
            lim = int(np.ceil(total.mean()))
            if lim >= pico or lim < 3:
                return False
            estado["limite"] = lim
        estado.update(rec=rec, df=df, total=total)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    df, total = estado["df"], estado["total"]
    cal = info["calendario"]

    celdas = [md("# Enunciado")]
    if variante == 1:
        celdas.append(md("Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y número de "
                         "operarios necesarios por actividad se adjunta. Todas las actividades se han programado "
                         "en su inicio más temprano."))
        pregs = ["Dibuje el diagrama de Gantt y el diagrama de cargas. ¿Cuál es el número máximo de operarios "
                 "necesario y en qué semanas?",
                 "¿Está nivelado el consumo de recursos? Calcule la suma de cuadrados de las cargas como medida "
                 "de desnivel.",
                 "Nivele los recursos desplazando actividades no críticas dentro de su holgura, sin alargar el "
                 "proyecto, con el algoritmo de mínimos cuadrados. Indique qué actividades desplaza y cuánto.",
                 "Compare el diagrama de cargas antes y después: máximo de operarios y suma de cuadrados."]
    elif variante == 2:
        lim = estado["limite"]
        tramo = int(rng.integers(3, 6))
        lim2 = lim + int(rng.integers(2, 4))
        estado.update(tramo=tramo, lim2=lim2)
        celdas.append(md(f"Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y número de "
                         f"operarios necesarios por actividad se adjunta. La empresa sólo dispone de {lim} operarios "
                         f"durante las {tramo} primeras semanas y de {lim2} a partir de entonces."))
        pregs = ["Dibuje el diagrama de cargas con las actividades en su inicio más temprano y compruebe si se "
                 "respeta el límite de operarios.",
                 "Reprograme el proyecto con el algoritmo de asignación de Wiest-Levy para respetar el límite. "
                 "¿Qué actividades se desplazan y cuánto se alarga el proyecto?",
                 f"¿Y si se dispusiera de {lim} operarios durante todo el proyecto? Compare las dos duraciones.",
                 "Explique por qué la asignación puede alargar el proyecto y la nivelación nunca."]
    else:
        maq = pd.Series(rng.integers(0, 3, size=len(D)), index=D.index, name="machines")
        if (maq > 0).sum() < 3:
            maq.iloc[:3] = 1
        df = df.assign(machines=maq)
        no_crit = cal.index[cal["H_total"] > 0].tolist()
        X = no_crit[int(rng.integers(0, len(no_crit)))]
        resto = [a for a in no_crit if a != X]
        Y = resto[int(rng.integers(0, len(resto)))]
        sX = int(min(cal.loc[X, "H_total"], 2)); sY = int(min(cal.loc[Y, "H_total"], 1))
        estado.update(X=X, Y=Y, sX=sX, sY=sY)
        celdas.append(md(f"Considere el proyecto cuyo cuadro de prelaciones y duraciones (semanas) se adjunta. Cada "
                         f"actividad necesita un número de operarios y, algunas, una máquina excavadora (columna "
                         f"*machines*). El jefe de obra propone retrasar {X} {sX} semana(s) y {Y} {sY} semana(s) para "
                         f"suavizar la demanda de operarios."))
        pregs = ["Dibuje los diagramas de cargas de operarios y de máquinas con las actividades en su inicio más "
                 "temprano. ¿Cuántas máquinas hacen falta como máximo?",
                 "Evalúe la propuesta del jefe de obra: ¿respeta las holguras? ¿Reduce la suma de cuadrados de "
                 "las cargas de operarios?",
                 "Compare la propuesta con el resultado del algoritmo de nivelación. ¿Cuál es mejor?",
                 "¿Qué ocurre con la carga de máquinas tras la nivelación de operarios? ¿Puede una nivelación "
                 "empeorar el otro recurso?"]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code('#| fig-cap: "Diagrama de Gantt con las cargas de operarios (inicios más tempranos)"\n#| label: fig-ganttInicial\n'
                       "proyecto_inicial = proyecto.copy()\n"
                       "gantt_inicial, dibujo = proyecto_inicial.gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    if variante == 2:
        lim, tramo, lim2 = estado["limite"], estado["tramo"], estado["lim2"]
        celdas.append(code(f"limite_maximo_recursos = [{lim}] * {tramo} + [{lim2}]"))
        celdas.append(code('#| fig-cap: "Diagrama de cargas con el límite de operarios"\n#| label: fig-cargasInicial\n'
                           "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite_maximo_recursos)"))
        celdas.append(code("cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                           "limite = pd.Series([limite_maximo_recursos[min(t - 1, len(limite_maximo_recursos) - 1)] for t in range(1, len(cargas) + 1)], index=cargas.index)\n"
                           "exceso = cargas[cargas > limite]\n"
                           "Markdown(f'El límite se supera en las semanas {\", \".join(str(t) for t in exceso.index)}: hacen falta hasta {int(cargas.max())} operarios.')"))
    else:
        celdas.append(code('#| fig-cap: "Diagrama de cargas de operarios"\n#| label: fig-cargasInicial\n'
                           "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=None)"))
        celdas.append(code("cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                           "semanas_pico = cargas.index[cargas == cargas.max()].tolist()\n"
                           "Markdown(f'El máximo de operarios es {int(cargas.max())}, en las semanas {\", \".join(map(str, semanas_pico))}. '\n"
                           "         f'La media es {cargas.mean():.2f} operarios/semana.')"))
    if variante == 3:
        celdas.append(code('#| fig-cap: "Diagrama de Gantt con las cargas de máquinas"\n#| label: fig-ganttMaquinas\n'
                           "gantt_maquinas, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'machines', report=False, tikz=True, params=GANTT_PARAMS)\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code("cargas_maquinas = gantt_maquinas.data.loc['Total'].drop('H_total').astype(float)\n"
                           "Markdown(f'Hacen falta como máximo {int(cargas_maquinas.max())} máquina(s).')"))
    if variante == 1:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La suma de cuadrados de las cargas, $\\sum_t c_t^2$, penaliza los picos: para una misma "
                         "carga total es mínima cuando el perfil es plano. Es la función objetivo de la nivelación."))
        celdas.append(code("suma_cuadrados_inicial = float((cargas ** 2).sum())\n"
                           "Markdown(f'Suma de cuadrados inicial: {suma_cuadrados_inicial:.0f}. La carga total es {cargas.sum():.0f} operarios·semana; '\n"
                           "         f'si se repartiera uniformemente en {len(cargas)} semanas la suma de cuadrados sería {len(cargas) * cargas.mean() ** 2:.0f}.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(md("El algoritmo recorre las actividades no críticas (por orden de fin más temprano) y, para "
                         "cada una, prueba todos los desplazamientos dentro de su holgura eligiendo el que minimiza "
                         "la suma de cuadrados. La tabla muestra la suma de cuadrados para cada desplazamiento "
                         "posible, con el mínimo resaltado."))
        celdas.append(code('#| tbl-cap: "Suma de cuadrados para cada desplazamiento posible"\n#| label: tbl-cuadrados\n'
                           "proyecto_nivelado = proyecto.copy()\n"
                           "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(\n"
                           "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')\n"
                           "cuadrados"))
        celdas.append(code('#| tbl-cap: "Desplazamientos elegidos"\n#| label: tbl-desplazamientos\n'
                           "desplazamientos[desplazamientos['desplazamientos'] > 0]"))
        celdas += apartado(ap); ap += 1
        celdas.append(code('#| fig-cap: "Diagrama de Gantt tras la nivelación"\n#| label: fig-ganttNivelado\n'
                           "gantt_nivelado, dibujo = proyecto_nivelado.gantt_cargas(datos_nivelados, 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code('#| fig-cap: "Diagrama de cargas tras la nivelación"\n#| label: fig-cargasNivelado\n'
                           "plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=None)"))
        celdas.append(code("cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)\n"
                           "pd.DataFrame({'antes': [cargas.max(), (cargas ** 2).sum(), len(cargas)],\n"
                           "              'después': [cargas_niveladas.max(), (cargas_niveladas ** 2).sum(), len(cargas_niveladas)]},\n"
                           "             index=['máximo de operarios', 'suma de cuadrados', 'duración']).astype(int)"))
        celdas.append(md("La duración no cambia porque sólo se han movido actividades dentro de su holgura; el "
                         "perfil es más plano y el máximo de operarios, menor o igual."))
    elif variante == 2:
        celdas += apartado(ap); ap += 1
        celdas.append(md("El algoritmo de Wiest-Levy recorre el calendario periodo a periodo: cuando la demanda supera "
                         "el límite, aplaza las actividades con más holgura (y, si no basta, también críticas, lo que "
                         "alarga el proyecto). Las filas 💤 marcan las esperas introducidas."))
        celdas.append(code('#| fig-cap: "Diagrama de Gantt tras la asignación"\n#| label: fig-ganttAsignado\n'
                           "proyecto_asignado = proyecto.copy()\n"
                           "datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(\n"
                           "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                           "    maximo=limite_maximo_recursos, report=False, params=dict(period_width=1, **GANTT_PARAMS))\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code('#| fig-cap: "Diagrama de cargas tras la asignación"\n#| label: fig-cargasAsignado\n'
                           "plot_cargas(gantt_df=gantt_asignado, limite_maximo_recursos=limite_maximo_recursos)"))
        celdas.append(code("duracion_inicial = proyecto.duration(datos_proyecto['duration'])\n"
                           "duracion_asignada = len(gantt_asignado.data.columns.drop('H_total'))\n"
                           "esperas = datos_asignados[datos_asignados.index.str.startswith('💤')]\n"
                           "Markdown(f'El proyecto pasa de {duracion_inicial} a {duracion_asignada} semanas. Esperas introducidas: '\n"
                           "         f'{\", \".join(f\"{i[1:]} ({int(fila.duration)} sem.)\" for i, fila in esperas.iterrows()) or \"ninguna\"}.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code('#| fig-cap: "Diagrama de Gantt con el límite constante"\n#| label: fig-ganttLimiteFijo\n'
                           f"limite_fijo = [{lim}]\n"
                           "proyecto_fijo = proyecto.copy()\n"
                           "datos_fijo, gantt_fijo, dibujo = proyecto_fijo.asignar(\n"
                           "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                           "    maximo=limite_fijo, report=False, params=dict(period_width=1, **GANTT_PARAMS))\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code("duracion_fija = len(gantt_fijo.data.columns.drop('H_total'))\n"
                           f"Markdown(f'Con {lim} operarios durante todo el proyecto la duración sería de {{duracion_fija}} semanas, frente a {{duracion_asignada}} con el refuerzo a partir de la semana {tramo}.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(md("La nivelación sólo usa las holguras: desplaza actividades no críticas sin tocar el "
                         "plazo. La asignación impone una restricción dura; si en algún periodo ni siquiera "
                         "aplazando todas las actividades con holgura cabe la demanda, tiene que aplazar "
                         "actividades críticas y el proyecto se alarga."))
    else:
        X, Y, sX, sY = estado["X"], estado["Y"], estado["sX"], estado["sY"]
        celdas += apartado(ap); ap += 1
        celdas.append(code(f"holguras = proyecto.calendar(datos_proyecto['duration'])['H_total']\n"
                           f"propuesta = {{'{X}': {sX}, '{Y}': {sY}}}\n"
                           "pd.DataFrame({'desplazamiento propuesto': pd.Series(propuesta), 'holgura total': holguras[list(propuesta)]})"))
        celdas.append(code('#| fig-cap: "Diagrama de Gantt con la propuesta del jefe de obra"\n#| label: fig-ganttPropuesta\n'
                           "suma_cuadrados_inicial = float((cargas ** 2).sum())\n"
                           "proyecto_propuesta = proyecto.copy()\n"
                           "suma_cuadrados_propuesta, _ = proyecto_propuesta.evaluar_desplazamiento(\n"
                           "    datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=False, **propuesta)\n"
                           "_, _, dibujo = proyecto.copy().desplazar(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True,\n"
                           "                                         params=GANTT_PARAMS, **propuesta)\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code("Markdown(f'La propuesta respeta las holguras (cada desplazamiento es menor o igual que la holgura total) y la suma de cuadrados '\n"
                           "         f'pasa de {suma_cuadrados_inicial:.0f} a {suma_cuadrados_propuesta:.0f}: '\n"
                           "         f'{\"mejora\" if suma_cuadrados_propuesta < suma_cuadrados_inicial else \"no mejora\"} el nivelado.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code('#| tbl-cap: "Desplazamientos del algoritmo de nivelación"\n#| label: tbl-desplazamientos\n'
                           "proyecto_nivelado = proyecto.copy()\n"
                           "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(\n"
                           "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')\n"
                           "desplazamientos[desplazamientos['desplazamientos'] > 0]"))
        celdas.append(code('#| fig-cap: "Diagrama de cargas de operarios tras la nivelación"\n#| label: fig-cargasNivelado\n'
                           "plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=None)"))
        celdas.append(code("cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)\n"
                           "suma_cuadrados_nivelado = float((cargas_niveladas ** 2).sum())\n"
                           "pd.DataFrame({'suma de cuadrados': [suma_cuadrados_inicial, suma_cuadrados_propuesta, suma_cuadrados_nivelado],\n"
                           "              'máximo de operarios': [cargas.max(), None, cargas_niveladas.max()]},\n"
                           "             index=['inicial', 'propuesta del jefe de obra', 'nivelación'])"))
        celdas.append(code("Markdown(f'El algoritmo de nivelación {\"mejora\" if suma_cuadrados_nivelado < suma_cuadrados_propuesta else \"no mejora\"} la propuesta: '\n"
                           "         f'prueba sistemáticamente todos los desplazamientos de cada actividad no crítica, aunque, al hacerlo actividad a actividad, tampoco garantiza el óptimo global.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code('#| fig-cap: "Cargas de máquinas tras la nivelación de operarios"\n#| label: fig-maquinasNivelado\n'
                           "datos_maquinas = datos_nivelados.copy()\n"
                           "datos_maquinas['machines'] = datos_proyecto['machines'].reindex(datos_maquinas.index).fillna(0).astype(int)\n"
                           "gantt_maquinas_niv, dibujo = proyecto_nivelado.copy().gantt_cargas(datos_maquinas, 'duration', 'machines', report=False, tikz=True, params=GANTT_PARAMS)\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code("cargas_maquinas_niv = gantt_maquinas_niv.data.loc['Total'].drop('H_total').astype(float)\n"
                           "pd.DataFrame({'antes': [cargas_maquinas.max(), (cargas_maquinas ** 2).sum()],\n"
                           "              'después': [cargas_maquinas_niv.max(), (cargas_maquinas_niv ** 2).sum()]},\n"
                           "             index=['máximo de máquinas', 'suma de cuadrados (máquinas)']).astype(int)"))
        celdas.append(md("La nivelación optimiza un único recurso; los desplazamientos que allanan la demanda de "
                         "operarios pueden concentrar la de máquinas. Con varios recursos hay que nivelar con una "
                         "función objetivo combinada o comprobar cada recurso tras nivelar el principal."))
    sub = {1: f"{n} actividades, nivelación", 2: f"{n} actividades, asignación con límite {estado.get('limite')}→{estado.get('lim2')}",
           3: f"{n} actividades, dos recursos y propuesta de desplazamiento"}[variante]
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _consumible(rng):
    """Recurso consumible (combustible, material) no almacenable frente a almacenable."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 10 <= i["T"] <= 20):
            return False
        rec = pd.Series(rng.integers(1, 6, size=len(i["D"])) * 10, index=i["D"].index, name="resources")
        df = P.tabla_base(i, duration=i["D"], resources=rec)
        try:
            total = _cargas(i, df)
        except Exception:
            return False
        acumulado = total.cumsum().values
        lim = int(np.ceil(max(acumulado[k] / (k + 1) for k in range(len(acumulado))) / 10) * 10)
        if lim >= total.max() or lim < rec.max():
            return False
        estado.update(rec=rec, df=df, total=total, lim=lim)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    df, total, lim = estado["df"], estado["total"], estado["lim"]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (semanas) se adjunta. Cada actividad "
                 "consume cada semana la cantidad de combustible indicada (litros/semana). El suministrador entrega "
                 f"como máximo {lim} litros por semana.")]
    celdas.append(preguntas([
        "Dibuje el diagrama de consumo semanal con las actividades en su inicio más temprano. ¿En qué semanas se "
        "supera el suministro?",
        "Si no se puede almacenar combustible de una semana para la siguiente, reprograme el proyecto para "
        "respetar el suministro (algoritmo de asignación). ¿Cuánto se alarga?",
        "Si en cambio se dispone de un depósito donde guardar lo no consumido, la restricción pasa a ser "
        "acumulada: el consumo acumulado hasta cada semana no puede superar el suministro acumulado. Compruebe si la "
        "programación original la cumple.",
        "Compare ambas situaciones. ¿Qué tipo de recurso son los operarios de los ejercicios anteriores?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"suministro = {lim}"))
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Diagrama de Gantt con el consumo semanal"\n#| label: fig-ganttInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Consumo semanal frente al suministro"\n#| label: fig-cargasInicial\n'
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[suministro])"))
    celdas.append(code("consumo = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "exceso = consumo[consumo > suministro]\n"
                       "Markdown(f'Se supera el suministro en las semanas {\", \".join(str(t) for t in exceso.index)} (máximo {int(consumo.max())} litros).')"))
    celdas += apartado(2)
    celdas.append(code('#| fig-cap: "Diagrama de Gantt tras la asignación"\n#| label: fig-ganttAsignado\n'
                       "proyecto_asignado = proyecto.copy()\n"
                       "datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                       "    maximo=[suministro], report=False, params=dict(period_width=1, **GANTT_PARAMS))\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("duracion_inicial = proyecto.duration(datos_proyecto['duration'])\n"
                       "duracion_asignada = len(gantt_asignado.data.columns.drop('H_total'))\n"
                       "Markdown(f'Sin almacenamiento el proyecto pasa de {duracion_inicial} a {duracion_asignada} semanas.')"))
    celdas += apartado(3)
    celdas.append(code("acumulado = pd.DataFrame({'consumo acumulado': consumo.cumsum(), 'suministro acumulado': suministro * np.arange(1, len(consumo) + 1)})\n"
                       "acumulado['cumple'] = acumulado['consumo acumulado'] <= acumulado['suministro acumulado']\n"
                       "acumulado.T"))
    celdas.append(code('#| fig-cap: "Consumo y suministro acumulados"\n#| label: fig-acumulado\n'
                       "ax = acumulado[['consumo acumulado', 'suministro acumulado']].plot(figsize=(7, 4)); ax.set_xlabel('semana'); ax.set_ylabel('litros');"))
    celdas.append(code("Markdown('Con depósito la programación original ' + ('es viable: el depósito absorbe los picos con lo sobrante de las semanas de menos consumo.' if acumulado['cumple'].all() else f'tampoco es viable: en la semana {acumulado.index[~acumulado[\"cumple\"]][0]} el consumo acumulado supera lo entregado hasta entonces.'))"))
    celdas += apartado(4)
    celdas.append(md("Un recurso no almacenable (mano de obra, maquinaria, un suministro que no cabe en depósito) "
                     "impone una restricción semana a semana; uno almacenable impone una restricción acumulada, "
                     "mucho menos exigente. Los operarios son del primer tipo: las horas de un operario que no se "
                     "usan esta semana no se guardan para la siguiente. Por eso los recursos renovables se tratan con "
                     "asignación y los consumibles con balances acumulados (y, si hace falta, un depósito dimensionado "
                     "con la diferencia máxima entre las dos curvas)."))
    return celdas, f"{n} actividades; combustible con suministro de {lim} l/semana"


def _limite_insuficiente(rng):
    """Factibilidad: límite inferior a lo que necesita una actividad; duración en función del límite."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 10 <= i["T"] <= 20):
            return False
        rec = P.recursos(rng, list(i["D"].index), hi=6)
        if rec.max() < 5 or (rec == rec.max()).sum() > 2:
            return False
        df = P.tabla_base(i, duration=i["D"], resources=rec)
        try:
            total = _cargas(i, df)
        except Exception:
            return False
        if total.max() <= rec.max() + 1:
            return False
        estado.update(rec=rec, df=df, total=total)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    df, total, rec = estado["df"], estado["total"], estado["rec"]
    lim_malo = int(rec.max()) - 1
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y operarios necesarios "
                 f"se adjunta. La empresa se plantea cuántos operarios contratar para el proyecto.")]
    celdas.append(preguntas([
        f"¿Es posible ejecutar el proyecto con {lim_malo} operarios? Razónelo mirando sólo el cuadro de datos.",
        "¿Cuál es el número máximo de operarios que haría falta para ejecutar el proyecto en su duración mínima "
        "con las actividades en su inicio más temprano? ¿Y tras nivelar?",
        "Construya la tabla «operarios disponibles → duración del proyecto» aplicando el algoritmo de asignación "
        "para cada número de operarios entre el mínimo factible y el máximo necesario.",
        "Si cada operario cuesta 2 k€ por semana contratado durante todo el proyecto y cada semana de duración "
        "tiene un coste fijo de 5 k€, ¿cuántos operarios conviene contratar?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion_minima = proyecto.duration(datos_proyecto['duration'])"))
    celdas += apartado(1)
    celdas.append(code(f"limite = {lim_malo}\n"
                       "imposibles = datos_proyecto.index[datos_proyecto['resources'] > limite].tolist()\n"
                       "Markdown(f'No: {\", \".join(imposibles)} necesita(n) {int(datos_proyecto[\"resources\"].max())} operarios a la vez, más que los {limite} disponibles, '\n"
                       "         'y una actividad no puede ejecutarse con menos operarios de los que requiere (no se admite alargarla). El mínimo factible es el máximo requerido por una actividad.')"))
    celdas += apartado(2)
    celdas.append(code('#| fig-cap: "Cargas con las actividades en su inicio más temprano"\n#| label: fig-cargasInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=None)"))
    celdas.append(code("cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "proyecto_nivelado = proyecto.copy()\n"
                       "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')\n"
                       "cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)\n"
                       "Markdown(f'Máximo de operarios: {int(cargas.max())} con inicios más tempranos y {int(cargas_niveladas.max())} tras nivelar (duración {duracion_minima} en ambos casos).')"))
    celdas += apartado(3)
    celdas.append(code("minimo = int(datos_proyecto['resources'].max())\n"
                       "maximo = int(cargas_niveladas.max())\n"
                       "filas = {}\n"
                       "for L in range(minimo, maximo + 1):\n"
                       "    p = proyecto.copy()\n"
                       "    _, g, _ = p.asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources', maximo=[L], report=False)\n"
                       "    filas[L] = len(g.data.columns.drop('H_total'))\n"
                       "tabla = pd.Series(filas, name='duración (semanas)').rename_axis('operarios').to_frame()\n"
                       "tabla.T"))
    celdas.append(code('#| fig-cap: "Duración del proyecto según los operarios disponibles"\n#| label: fig-duracionOperarios\n'
                       "ax = tabla.plot(marker='o', legend=False, figsize=(7, 4)); ax.set_ylabel('semanas');"))
    celdas += apartado(4)
    celdas.append(code("tabla['coste operarios'] = 2 * tabla.index * tabla['duración (semanas)']\n"
                       "tabla['coste fijo'] = 5 * tabla['duración (semanas)']\n"
                       "tabla['coste total'] = tabla['coste operarios'] + tabla['coste fijo']\n"
                       "display(tabla)\n"
                       "Markdown(f'El coste total es mínimo con {tabla[\"coste total\"].idxmin()} operarios ({tabla[\"coste total\"].min():.0f} k€). '\n"
                       "         'Más operarios acortan el proyecto pero cuestan cada semana; la tabla duración-operarios tiene escalones porque la asignación es heurística y el plazo sólo baja cuando el límite libera un cuello de botella.')"))
    return celdas, f"{n} actividades; factibilidad y duración en función de los operarios"



PLANIFICADOR = (
    "def planificar(datos, prelaciones, limite, prioridad, interrumpible=False, max_periodos=200):\n"
    "    '''Programación en paralelo periodo a periodo con un recurso renovable limitado.\n"
    "    prioridad(a, t) -> número: menor = antes. Devuelve (semanas trabajadas por actividad, inicio, fin).'''\n"
    "    pred = prelaciones.immediate_predecessor\n"
    "    restante = datos['duration'].astype(int).to_dict()\n"
    "    rec = datos['resources'].astype(int).to_dict()\n"
    "    trabajo = {a: [] for a in restante}\n"
    "    en_curso = set()\n"
    "    for t in range(1, max_periodos + 1):\n"
    "        terminadas = {a for a, r in restante.items() if r == 0}\n"
    "        if len(terminadas) == len(restante):\n"
    "            break\n"
    "        elegibles = [a for a in restante if restante[a] > 0 and set(pred[a]) <= terminadas]\n"
    "        if not interrumpible:\n"
    "            # las ya empezadas siguen; las nuevas se ordenan por prioridad\n"
    "            fijas = [a for a in elegibles if a in en_curso]\n"
    "            nuevas = sorted([a for a in elegibles if a not in en_curso], key=lambda a: prioridad(a, t))\n"
    "            orden = fijas + nuevas\n"
    "        else:\n"
    "            orden = sorted(elegibles, key=lambda a: prioridad(a, t))\n"
    "        disponible = limite\n"
    "        activas = []\n"
    "        for a in orden:\n"
    "            if rec[a] <= disponible:\n"
    "                activas.append(a); disponible -= rec[a]\n"
    "        for a in activas:\n"
    "            restante[a] -= 1; trabajo[a].append(t)\n"
    "        en_curso = {a for a in activas if restante[a] > 0}\n"
    "    inicio = pd.Series({a: (min(s) if s else np.nan) for a, s in trabajo.items()})\n"
    "    fin = pd.Series({a: (max(s) if s else np.nan) for a, s in trabajo.items()})\n"
    "    return trabajo, inicio, fin\n"
    "\n"
    "def gantt_texto(trabajo, limite):\n"
    "    '''Tabla actividad × semana con el recurso usado y fila de carga total.'''\n"
    "    T = max(max(s) for s in trabajo.values() if s)\n"
    "    g = pd.DataFrame('', index=list(trabajo), columns=range(1, T + 1))\n"
    "    for a, semanas in trabajo.items():\n"
    "        for t in semanas:\n"
    "            g.loc[a, t] = int(datos_proyecto.loc[a, 'resources'])\n"
    "    carga = pd.Series({t: sum(int(datos_proyecto.loc[a, 'resources']) for a, s in trabajo.items() if t in s) for t in g.columns})\n"
    "    g.loc['Total'] = carga\n"
    "    g.loc['≤ límite'] = ['sí' if c <= limite else 'NO' for c in carga]\n"
    "    return g\n")


def _proyecto_recursos(rng, n):
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and 10 <= i["T"] <= 18 and (i["D"] >= 2).all()):
            return False
        rec = P.recursos(rng, list(i["D"].index), hi=4)
        df = P.tabla_base(i, duration=i["D"], resources=rec)
        try:
            total = _cargas(i, df)
        except Exception:
            return False
        if total.max() < rec.max() + 2 or (i["calendario"]["H_total"] > 0).sum() < 3:
            return False
        estado.update(rec=rec, df=df, total=total)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    return info, estado


def _reglas_prioridad(rng):
    """Programación con recurso limitado según tres reglas de prioridad; comparación con la librería."""
    n = int(rng.choice([8, 9]))
    info, estado = _proyecto_recursos(rng, n)
    D, T = info["D"], int(info["T"])
    df, total, rec = estado["df"], estado["total"], estado["rec"]
    limite = int(max(rec.max(), int(np.ceil(total.max() * 0.6))))
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y operarios necesarios "
                 f"se adjunta. Sólo se dispone de {limite} operarios y las actividades no pueden interrumpirse "
                 "una vez empezadas.")]
    celdas.append(preguntas([
        "Calcule el calendario sin restricción de recursos y dibuje las cargas con los inicios más tempranos. "
        f"¿En qué semanas se superan los {limite} operarios?",
        "Programe el proyecto periodo a periodo con un planificador en paralelo: en cada semana se ordenan las "
        "actividades disponibles según una regla de prioridad y se van admitiendo mientras quepan en el "
        "límite. Aplique tres reglas: (a) menor holgura total, (b) menor duración (SPT) y (c) mayor número de "
        "operarios. Dé la duración con cada una.",
        "Compare con el resultado del algoritmo de Wiest-Levy de la librería (`asignar`).",
        "¿Alguna regla garantiza la duración mínima? ¿Qué información usa la regla de la holgura que no usan "
        "las otras?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"limite = {limite}"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Calendario sin restricción de recursos"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_minima = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Cargas con los inicios más tempranos"\n#| label: fig-cargasInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[limite])"))
    celdas.append(code("cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "exceso = cargas[cargas > limite]\n"
                       "Markdown(f'Se superan los {limite} operarios en las semanas {\", \".join(str(t) for t in exceso.index)} (cargas {\", \".join(f\"{int(c)}\" for c in exceso)}).')"))
    celdas += apartado(2)
    celdas.append(md("El planificador recorre las semanas: una actividad es *disponible* cuando han terminado sus "
                     "predecesoras; las que ya están en curso siguen (no se interrumpen) y las nuevas entran por "
                     "orden de prioridad mientras quepan en el límite."))
    celdas.append(code(PLANIFICADOR))
    celdas.append(code("reglas = {\n"
                       "    '(a) menor holgura total': lambda a, t: (calendario.loc[a, 'H_total'], a),\n"
                       "    '(b) menor duración (SPT)': lambda a, t: (datos_proyecto.loc[a, 'duration'], a),\n"
                       "    '(c) más operarios primero': lambda a, t: (-datos_proyecto.loc[a, 'resources'], a),\n"
                       "}\n"
                       "resultados = {}\n"
                       "for nombre, regla in reglas.items():\n"
                       "    trabajo, inicio, fin = planificar(datos_proyecto, prelaciones, limite, regla)\n"
                       "    resultados[nombre] = dict(trabajo=trabajo, inicio=inicio, fin=fin, duracion=int(fin.max()))\n"
                       "pd.DataFrame({nombre: r['inicio'].astype(int).rename('inicio') for nombre, r in resultados.items()}).assign(**{'sin límite': calendario['inicio_mas_temprano'] + 1}).T"))
    for etiqueta, nombre in (("a", "(a) menor holgura total"), ("b", "(b) menor duración (SPT)"), ("c", "(c) más operarios primero")):
        celdas.append(code(f'#| tbl-cap: "Programación con la regla {nombre}"\n#| label: tbl-regla{etiqueta}\n'
                           f"gantt_texto(resultados['{nombre}']['trabajo'], limite)"))
    celdas.append(code("duraciones = pd.Series({nombre: r['duracion'] for nombre, r in resultados.items()}, name='duración (semanas)')\n"
                       "duraciones['sin límite de recursos'] = duracion_minima\n"
                       "duraciones.to_frame().T"))
    celdas += apartado(3)
    celdas.append(code('#| fig-cap: "Asignación de la librería (Wiest-Levy)"\n#| label: fig-ganttAsignado\n'
                       "proyecto_asignado = proyecto.copy()\n"
                       "datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                       "    maximo=[limite], report=False, params=dict(period_width=1, **GANTT_PARAMS))\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("duracion_libreria = len(gantt_asignado.data.columns.drop('H_total'))\n"
                       "Markdown(f'La librería obtiene {duracion_libreria} semanas; las reglas dan {\", \".join(f\"{k}: {v}\" for k, v in duraciones.drop(\"sin límite de recursos\").items())}. '\n"
                       "         'Wiest-Levy también es una heurística (aplaza las actividades con más holgura), de modo que ninguna de las cuatro cifras es necesariamente el óptimo.')"))
    celdas += apartado(4)
    celdas.append(md("Ninguna regla garantiza el óptimo: el problema de programación con recursos limitados es "
                     "NP-duro y las reglas de prioridad son heurísticas; para proyectos pequeños se puede "
                     "comprobar por enumeración o programación entera. La regla de la holgura mínima es la única "
                     "que usa la *estructura* del proyecto (qué actividades tienen margen y cuáles arrastrarían el "
                     "plazo); SPT y «más operarios» sólo miran la actividad aislada. Por eso suele funcionar mejor, "
                     "aunque no siempre: la holgura se calculó sin recursos y deja de ser exacta en cuanto se "
                     "aplaza algo."))
    return celdas, f"{n} actividades; {limite} operarios; tres reglas de prioridad frente a Wiest-Levy"


def _interrumpibles(rng):
    """Actividades interrumpibles frente a no interrumpibles con el mismo límite de recursos."""
    n = int(rng.choice([8, 9]))
    info, estado = _proyecto_recursos(rng, n)
    D, T = info["D"], int(info["T"])
    df, total, rec = estado["df"], estado["total"], estado["rec"]
    limite = int(max(rec.max(), int(np.ceil(total.max() * 0.6))))
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y operarios necesarios "
                 f"se adjunta, con {limite} operarios disponibles. Se discute si merece la pena organizar el "
                 "trabajo de modo que una actividad pueda interrumpirse y reanudarse más tarde (sin coste "
                 "añadido) cuando los operarios hagan falta en otra más urgente.")]
    celdas.append(preguntas([
        "Programe el proyecto con un planificador en paralelo y la regla «menor holgura total primero» "
        "suponiendo que las actividades **no** pueden interrumpirse. ¿Cuánto dura?",
        "Repita suponiendo que sí pueden interrumpirse: en cada semana se vuelven a ordenar *todas* las "
        "actividades disponibles, también las empezadas, y una actividad en curso puede quedar parada si "
        "entra otra más prioritaria. ¿Cuánto dura? ¿Qué actividades se interrumpen y cuántas veces?",
        "Compare las dos programaciones semana a semana. ¿De dónde sale la ganancia (si la hay)?",
        "¿Qué costes ocultos tiene interrumpir una actividad en la práctica? ¿Cuándo compensa?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_minima = proyecto.duration(datos_proyecto['duration'])\n"
                       f"limite = {limite}"))
    celdas.append(code(PLANIFICADOR))
    celdas.append(code("regla = lambda a, t: (calendario.loc[a, 'H_total'], datos_proyecto.loc[a, 'duration'], a)"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Programación sin interrupciones"\n#| label: tbl-sinInterrupcion\n'
                       "trabajo_no, inicio_no, fin_no = planificar(datos_proyecto, prelaciones, limite, regla, interrumpible=False)\n"
                       "gantt_texto(trabajo_no, limite)"))
    celdas.append(code("Markdown(f'Sin interrupciones el proyecto dura {int(fin_no.max())} semanas (mínimo sin límite de recursos: {duracion_minima}).')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Programación con interrupciones permitidas"\n#| label: tbl-conInterrupcion\n'
                       "trabajo_si, inicio_si, fin_si = planificar(datos_proyecto, prelaciones, limite, regla, interrumpible=True)\n"
                       "gantt_texto(trabajo_si, limite)"))
    celdas.append(code("def interrupciones(semanas):\n"
                       "    return sum(1 for s1, s2 in zip(semanas, semanas[1:]) if s2 > s1 + 1)\n"
                       "inter = pd.Series({a: interrupciones(s) for a, s in trabajo_si.items()})\n"
                       "inter = inter[inter > 0]\n"
                       "Markdown(f'''Con interrupciones el proyecto dura {int(fin_si.max())} semanas. "
                       "{('Se interrumpen ' + ', '.join(f'{a} ({v} vez/veces)' for a, v in inter.items()) + '.') if len(inter) else 'Ninguna actividad llega a interrumpirse.'}''')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Inicio y fin de cada actividad en las dos programaciones"\n#| label: tbl-comparacion\n'
                       "pd.DataFrame({'inicio (no interr.)': inicio_no.astype(int), 'fin (no interr.)': fin_no.astype(int),\n"
                       "              'inicio (interr.)': inicio_si.astype(int), 'fin (interr.)': fin_si.astype(int),\n"
                       "              'semanas trabajadas (interr.)': {a: len(s) for a, s in trabajo_si.items()}})"))
    celdas.append(md("Si hay ganancia, viene de que al permitir interrumpir no hay que esperar a que termine una "
                     "actividad poco urgente para liberar operarios: una actividad crítica puede entrar en cuanto "
                     "está disponible. Si no la hay, es que la regla de prioridad ya había evitado bloquear el "
                     "recurso con actividades holgadas. Interrumpir nunca empeora la duración con este "
                     "planificador (el no interrumpible es un caso particular), pero puede fragmentar mucho el "
                     "trabajo."))
    celdas += apartado(4)
    celdas.append(md("Interrumpir tiene costes que el modelo ignora: volver a montar y desmontar (tiempos de "
                     "preparación), pérdida de ritmo y de aprendizaje, el material en curso inmovilizado, la "
                     "coordinación de los operarios y la dificultad de seguimiento. Compensa cuando el ahorro de "
                     "plazo vale más que esos costes: actividades largas, modulares (pueden partirse en lotes) y "
                     "sin costes de preparación apreciables; no compensa en tareas con fraguados, curados o "
                     "procesos continuos."))
    return celdas, f"{n} actividades; {limite} operarios; interrumpir o no interrumpir"


VARIANTES = [
    Variante("nivelacion", 1, "cargas, suma de cuadrados y nivelación por mínimos cuadrados", lambda rng: _base(1, rng)),
    Variante("asignacion", 2, "asignación Wiest-Levy con límite por tramos y con límite fijo", lambda rng: _base(2, rng)),
    Variante("dos-recursos", 3, "dos tipos de recurso; evaluación de un desplazamiento propuesto frente a la nivelación",
             lambda rng: _base(3, rng)),
    Variante("consumible", 2, "recurso consumible no almacenable frente a almacenable (restricción semanal frente a acumulada)",
             _consumible),
    Variante("limite-insuficiente", 2, "factibilidad del límite; tabla operarios → duración y número óptimo de operarios",
             _limite_insuficiente),
    Variante("reglas-prioridad", 3, "planificador en paralelo con tres reglas de prioridad (holgura, SPT, más recurso) frente a Wiest-Levy",
             _reglas_prioridad),
    Variante("interrumpibles", 3, "misma regla con y sin interrupciones permitidas: duración, actividades interrumpidas y costes ocultos",
             _interrumpibles),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
