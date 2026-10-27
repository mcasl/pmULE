"""Colección 11 — Recursos: diagrama de cargas, nivelación, asignación con
límite, evaluación de desplazamientos y varios tipos de recurso."""
import contextlib
import io

import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

def _pl(n, singular, plural=None):
    """«1 semana», «3 semanas»: número y sustantivo concordados (para enunciados y subtítulos)."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


META = dict(slug="11_recursos", prefijo="Recursos",
            titulo="Nivelación y asignación de recursos",
            descripcion="Diagrama de cargas, nivelación por mínimos cuadrados, asignación con límite de "
                        "recursos (Wiest-Levy), evaluación de desplazamientos y varios tipos de recurso.",
            n_defecto=3)

# Celda auxiliar (se define en el cuaderno): explicaciones con los números de cada caso.
_AUX_REC = r'''
def semanas_txt(lista):
    """«la semana 6», «las semanas 1, 2 y 5» o «ninguna semana»."""
    lista = [str(int(x)) for x in lista]
    if not lista:
        return 'ninguna semana'
    return ('la semana ' if len(lista) == 1 else 'las semanas ') + lista_y(lista)


def explica_cuadrados(cargas):
    """Σ c² agrupando las semanas de igual carga: «6·3² + 4·5² = 54 + 100 = 154»."""
    c = pd.Series(cargas).astype(int)
    cuenta = c.value_counts().sort_index()
    terminos = [f'{n}·{v}²' if n > 1 else f'{v}²' for v, n in cuenta.items()]
    parciales = [f'{n * v * v}' for v, n in cuenta.items()]
    return f'Σ c² = {" + ".join(terminos)} = {" + ".join(parciales)} = **{int((c ** 2).sum())}**'


def explica_nivelacion(cuadrados, calendario):
    """Una línea por actividad recorrida: desplazamientos probados, suma mínima y elección."""
    tabla = getattr(cuadrados, 'data', cuadrados)
    lineas, elegidos = [], {}
    for a, fila in tabla.iterrows():
        v = fila.dropna()
        k = int(v.index.max())
        H = int(calendario.loc[a, 'H_total'])
        minimo = v.min()
        mejor = int(v.idxmin())
        empates = [int(x) for x in v.index[v == minimo]]
        if k == 0:
            probado = 'solo puede probarse 0'
        else:
            probado = f'se prueban 0 a {k}'
        if k < H:
            probado += (f' (no {H}: los desplazamientos ya aplicados a actividades que comparten su holgura '
                        f'le han dejado {plural(k, "semana")})')
        empate = (f'; empatan {lista_y([str(x) for x in empates])} y se toma el menor' if len(empates) > 1 else '')
        lineas.append(f'- {a} (holgura total {H}): {probado}; suma mínima {int(minimo)} con {mejor}{empate} → '
                      + (f'se desplaza **{plural(mejor, "semana")}**' if mejor > 0 else 'no se desplaza'))
        elegidos[a] = mejor
    return '\n'.join(lineas), {a: d for a, d in elegidos.items() if d > 0}


def tabla_asignacion(proyecto, datos, maximo, recurso='resources', duracion='duration'):
    """Semanas en que la demanda supera el límite: orden de prioridad, qué entra y qué se aplaza y por qué."""
    pasos, finales, _, _ = proyecto.copy().asignar_pasos(datos.copy(), duracion, recurso, maximo)
    filas = []
    for ps in pasos:
        if ps['total'] <= ps['limite']:
            continue
        t, L = ps['periodo'], int(ps['limite'])
        orden, aplazadas, acum = [], [], 0
        for a, f in ps['candidatas'].iterrows():
            r = int(f[t])
            orden.append(f'{a} (en curso, {r})' if f['Empezadas'] else f'{a} (H={int(f["H_total"])}, d={int(f[duracion])}, {r})')
            if a in ps['se_programan']:
                acum += r
            else:
                aplazadas.append(f'{a}: {acum} + {r} = {acum + r} > {L}')
        filas.append({'semana': t, 'límite': L, 'demanda': int(ps['total']),
                      'orden de prioridad (holgura, duración, recurso)': ' › '.join(orden),
                      'entran': lista_y(sorted(ps['se_programan'])) if ps['se_programan'] else '—',
                      'se aplazan una semana (no caben)': '; '.join(aplazadas)})
    return pd.DataFrame(filas).set_index('semana') if filas else Markdown('En ninguna semana la demanda supera el límite: no hay que aplazar nada.')


def primera_critica_aplazada(proyecto, datos, maximo, recurso='resources', duracion='duration'):
    """(semana, actividad) de la primera actividad sin holgura que la asignación tiene que aplazar, o None."""
    pasos, _, _, _ = proyecto.copy().asignar_pasos(datos.copy(), duracion, recurso, maximo)
    for ps in pasos:
        for a in ps['candidatas'].index:
            if a in ps['se_retrasan'] and float(ps['candidatas'].loc[a, 'H_total']) == 0:
                return ps['periodo'], a
    return None


def esperas_asignacion(datos_asignados):
    """{actividad: semanas de espera} a partir de las filas 💤 que añade la asignación."""
    e = datos_asignados[datos_asignados.index.str.startswith('💤')]
    return {i[1:]: int(f.duration) for i, f in e.iterrows()}
'''

# Explicación de las esperas de ``asignar_multi`` a partir de su traza.
_AUX_MULTI = r'''
def explica_esperas_multi(traza, datos, recursos, limites):
    """Una línea por espera: el recurso que se satura, la semana y la suma que supera el límite."""
    carga = {r: {} for r in recursos}
    lineas = []
    for _, f in traza.iterrows():
        a, t = f['actividad'], int(f['periodo'])
        semanas = range(t, t + int(datos.loc[a, 'duration']))
        if f['decision'] == 'empieza':
            for r in recursos:
                for w in semanas:
                    carga[r][w] = carga[r].get(w, 0) + int(datos.loc[a, r])
            continue
        partes = []
        for r, L in zip(recursos, limites):
            q = int(datos.loc[a, r])
            for w in semanas:
                c = carga[r].get(w, 0)
                if c + q > L:
                    partes.append(f'{r} en la semana {w}: ocupación {c} + {q} pedidos = {c + q} > {L}')
                    break
        lineas.append(f'- Semana {t}, {a}: ' + '; '.join(partes))
    return lineas
'''

# Explicación del orden de prioridad de ``asignar`` (Wiest-Levy de la librería).
_REGLA_ASIGNAR = ("Regla del algoritmo: en cada semana se toman las actividades que el calendario vigente sitúa en ella "
                  "y se ordenan así: primero las ya empezadas (no se interrumpen); después, menor holgura total "
                  "(recalculada tras los aplazamientos anteriores) y, a igualdad, menor duración. Se admiten en ese orden "
                  "mientras la suma de recursos no supere el límite; las que no caben se aplazan una semana (y con ellas "
                  "sus sucesoras, si no tienen holgura). En la traza, *Total* es la demanda de esa semana antes de aplazar.")


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
                         f"*machines*). El jefe de obra propone retrasar {X} {_pl(sX, 'semana')} y {Y} {_pl(sY, 'semana')} para "
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
    celdas.append(code(_AUX_REC))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario: fechas y holgura total de cada actividad"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(md("En el Gantt cada actividad empieza en su inicio más temprano y la casilla lleva los operarios que "
                     "ocupa; la fila *Total* suma la columna (carga de la semana) y la fila *Cuadrados* eleva cada carga "
                     "al cuadrado (su suma, en la última columna, es la medida de desnivel que usa la nivelación)."))
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
                           "detalle = '; '.join(f'semana {int(t)}: {int(c)} > {int(limite[t])}' for t, c in exceso.items())\n"
                           "if len(exceso):\n"
                           "    resp = (f'no se respeta: la carga supera el límite en {semanas_txt(exceso.index)} ({detalle}); '\n"
                           "            f'con todo en su inicio más temprano harían falta hasta {int(cargas.max())} operarios.')\n"
                           "else:\n"
                           "    pico = int(cargas.idxmax())\n"
                           "    resp = (f'sí se respeta: la carga no supera el límite en ninguna semana. El pico, {int(cargas.max())} operarios en la semana {pico}, '\n"
                           "            f'cae ya en el tramo con límite {int(limite[pico])}, y en las semanas con límite {int(limite.min())} la carga no pasa de '\n"
                           "            f'{int(cargas[limite == limite.min()].max())}.')\n"
                           "Markdown('**Respuesta:** ' + resp)"))
    else:
        celdas.append(code('#| fig-cap: "Diagrama de cargas de operarios"\n#| label: fig-cargasInicial\n'
                           "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=None)"))
        celdas.append(code("cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                           "semanas_pico = cargas.index[cargas == cargas.max()].tolist()\n"
                           "Markdown(f'Cargas semana a semana: {\", \".join(str(int(c)) for c in cargas)} (total {int(cargas.sum())} operarios·semana en {len(cargas)} semanas, '\n"
                           "         f'media {cargas.sum():.0f} / {len(cargas)} = {cargas.mean():.2f} operarios/semana).\\n\\n'\n"
                           "         f'**Respuesta:** el máximo es de **{int(cargas.max())} operarios**, en {semanas_txt(semanas_pico)}.')"))
    if variante == 3:
        celdas.append(code('#| fig-cap: "Diagrama de Gantt con las cargas de máquinas"\n#| label: fig-ganttMaquinas\n'
                           "gantt_maquinas, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'machines', report=False, tikz=True, params=GANTT_PARAMS)\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code("cargas_maquinas = gantt_maquinas.data.loc['Total'].drop('H_total').astype(float)\n"
                           "pico_maq = cargas_maquinas.index[cargas_maquinas == cargas_maquinas.max()].tolist()\n"
                           "Markdown(f'**Respuesta:** hacen falta como máximo **{plural(int(cargas_maquinas.max()), \"máquina\")}**, en {semanas_txt(pico_maq)}.')"))
    if variante == 1:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La suma de cuadrados de las cargas, $\\sum_t c_t^2$, penaliza los picos: para una misma "
                         "carga total es mínima cuando el perfil es plano (todas las semanas con la carga media). Es la "
                         "función objetivo de la nivelación."))
        celdas.append(code("suma_cuadrados_inicial = float((cargas ** 2).sum())\n"
                         "plano = len(cargas) * cargas.mean() ** 2\n"
                         "Markdown(f'''{explica_cuadrados(cargas)} (agrupando las semanas de igual carga).\\n\\n"
                         "Si los {int(cargas.sum())} operarios·semana se repartieran por igual en las {len(cargas)} semanas, cada una tendría {cargas.mean():.2f} y la suma sería "
                         "{len(cargas)} · {cargas.mean():.2f}² = {plano:.0f}.\\n\\n"
                         "**Respuesta:** no está nivelado: la carga va de {int(cargas.min())} a {int(cargas.max())} operarios y su suma de cuadrados es "
                         "{suma_cuadrados_inicial:.0f}, {suma_cuadrados_inicial - plano:.0f} por encima de la de un perfil plano ({plano:.0f}).''')"))
        celdas += apartado(ap); ap += 1
        celdas.append(md("El algoritmo recorre las actividades no críticas (por orden de fin más temprano) y, para "
                         "cada una, prueba todos los desplazamientos dentro de la holgura que le queda, eligiendo el que "
                         "minimiza la suma de cuadrados (a igualdad, el menor desplazamiento)."))
        celdas.append(code('#| tbl-cap: "Actividades no críticas en el orden en que las recorre el algoritmo (fin más temprano creciente; a igualdad, más holgura primero)"\n#| label: tbl-ordenNivelacion\n'
                           "no_criticas = calendario[calendario['H_total'] > 0].sort_values(['fin_mas_temprano', 'H_total'], ascending=[True, False])\n"
                           "no_criticas[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']].T"))
        celdas.append(md("Para cada una de ellas se prueba cada desplazamiento posible (las columnas de la tabla siguiente), "
                         "se recalculan las cargas y su suma de cuadrados, y se aplica el desplazamiento de suma mínima "
                         "(resaltado) antes de pasar a la siguiente actividad; por eso el resultado depende del orden y la "
                         "holgura disponible puede ser menor que la del calendario inicial."))
        celdas.append(code('#| tbl-cap: "Suma de cuadrados para cada desplazamiento posible"\n#| label: tbl-cuadrados\n'
                           "proyecto_nivelado = proyecto.copy()\n"
                           "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(\n"
                           "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')\n"
                           "cuadrados"))
        celdas.append(code("texto_niv, elegidos = explica_nivelacion(cuadrados, calendario)\n"
                           "Markdown(texto_niv + '\\n\\n**Respuesta:** ' + (('se desplazan ' + lista_y([f'{a} {plural(d, \"semana\")}' for a, d in elegidos.items()])\n"
                           "         + ', sin alargar el proyecto.') if elegidos else 'ningún desplazamiento mejora la suma de cuadrados.'))"))
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
        celdas.append(code("m0, m1 = int(cargas.max()), int(cargas_niveladas.max())\n"
                           "s0, s1 = int((cargas ** 2).sum()), int((cargas_niveladas ** 2).sum())\n"
                           "Markdown(f'''Después de nivelar: {explica_cuadrados(cargas_niveladas)}.\\n\\n"
                           "**Respuesta:** la suma de cuadrados baja de {s0} a {s1} ({s0 - s1} menos) y el máximo de operarios "
                           "{f\"baja de {m0} a {m1}\" if m1 < m0 else f\"se mantiene en {m0}\"}; la duración sigue siendo de {len(cargas_niveladas)} semanas porque solo se "
                           "han movido actividades dentro de su holgura.{\" El pico no baja: la nivelación minimiza la suma de cuadrados, no el máximo, y no ha encontrado desplazamientos que vacíen las semanas de carga máxima.\" if m1 == m0 else \"\"}''')"))
    elif variante == 2:
        celdas += apartado(ap); ap += 1
        celdas.append(md("El algoritmo de asignación (Wiest-Levy en la librería) recorre el calendario semana a semana y "
                         "aplaza lo que no cabe; si no basta con consumir holguras, aplaza también actividades críticas y el "
                         "proyecto se alarga. Las filas 💤 del Gantt marcan las esperas introducidas.\n\n" + _REGLA_ASIGNAR))
        celdas.append(md("Traza semana a semana (qué se programa y qué se aplaza una semana):"))
        celdas.append(code("_ = proyecto.copy().asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                           "                           maximo=limite_maximo_recursos, report=True)"))
        celdas.append(md("Por qué se aplaza cada actividad: en las semanas en que la demanda supera el límite, la tabla da el "
                         "orden de prioridad (con su holgura, duración y operarios), las que entran y, para cada aplazada, la "
                         "suma que no cabe."))
        celdas.append(code('#| tbl-cap: "Semanas con conflicto en la asignación"\n#| label: tbl-conflictos\n'
                           "tabla_asignacion(proyecto, datos_proyecto, limite_maximo_recursos)"))
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
                           "esperas = esperas_asignacion(datos_asignados)\n"
                           "cargas_asig = gantt_asignado.data.loc['Total'].drop('H_total').astype(float)\n"
                           "txt_esperas = lista_y([f\"{a} {plural(e, 'semana')}\" for a, e in esperas.items()]) if esperas else 'ninguna actividad'\n"
                           "if not esperas:\n"
                           "    resp = f'no hace falta desplazar ninguna actividad: el proyecto mantiene sus {duracion_inicial} semanas.'\n"
                           "elif duracion_asignada > duracion_inicial:\n"
                           "    resp = (f'se desplazan {txt_esperas}; el proyecto pasa de {duracion_inicial} a {duracion_asignada} semanas '\n"
                           "            f'(se alarga {plural(duracion_asignada - duracion_inicial, \"semana\")}) y el pico de operarios queda en {int(cargas_asig.max())}.')\n"
                           "else:\n"
                           "    resp = (f'se desplazan {txt_esperas}, dentro de su holgura: el proyecto mantiene sus {duracion_inicial} semanas '\n"
                           "            f'y el pico de operarios queda en {int(cargas_asig.max())}.')\n"
                           "Markdown('**Respuesta:** ' + resp)"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("_ = proyecto.copy().asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                           f"                           maximo=[{lim}], report=True)"))
        celdas.append(code('#| tbl-cap: "Semanas con conflicto con el límite constante"\n#| label: tbl-conflictosFijo\n'
                           f"tabla_asignacion(proyecto, datos_proyecto, [{lim}])"))
        celdas.append(code('#| fig-cap: "Diagrama de Gantt con el límite constante"\n#| label: fig-ganttLimiteFijo\n'
                           f"limite_fijo = [{lim}]\n"
                           "proyecto_fijo = proyecto.copy()\n"
                           "datos_fijo, gantt_fijo, dibujo = proyecto_fijo.asignar(\n"
                           "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                           "    maximo=limite_fijo, report=False, params=dict(period_width=1, **GANTT_PARAMS))\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code("duracion_fija = len(gantt_fijo.data.columns.drop('H_total'))\n"
                           "esperas_fijo = esperas_asignacion(datos_fijo)\n"
                           "txt_esperas_fijo = lista_y([f\"{a} {plural(e, 'semana')}\" for a, e in esperas_fijo.items()]) if esperas_fijo else 'ninguna'\n"
                           "Markdown(f'Esperas con el límite constante: {txt_esperas_fijo}.\\n\\n'\n"
                           f"         f'**Respuesta:** con {lim} operarios durante todo el proyecto la duración sería de {{duracion_fija}} semanas, frente a {{duracion_asignada}} con el refuerzo a {lim2} operarios a partir de la semana {tramo + 1}; '\n"
                           "         + (f'{plural(duracion_fija - duracion_asignada, \"semana\")} más ({duracion_fija - duracion_inicial} más que sin límite).' if duracion_fija > duracion_asignada else\n"
                           "            'la misma duración; en este proyecto el refuerzo no acorta el plazo.'))"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("critica = primera_critica_aplazada(proyecto, datos_proyecto, limite_maximo_recursos)\n"
                           "motivo = (f'en la semana {critica[0]} la actividad {critica[1]}, sin holgura, no cabía y hubo que aplazarla' if critica else\n"
                           "          'hubo que aplazar actividades más allá de su holgura')\n"
                           "Markdown(f'''La nivelación sólo usa las holguras: desplaza actividades no críticas sin tocar el plazo, y si un desplazamiento "
                           "alargara el proyecto no se prueba. La asignación impone una restricción dura (el límite de cada semana); si ni siquiera "
                           "aplazando todas las actividades con holgura cabe la demanda, tiene que aplazar actividades críticas y el proyecto se alarga. "
                           "**Respuesta:** {f'aquí la asignación ha alargado el proyecto de {duracion_inicial} a {duracion_asignada} semanas porque {motivo}' if duracion_asignada > duracion_inicial else ('aquí la asignación no ha necesitado alargar el proyecto: le ha bastado con consumir holguras' if esperas else 'aquí, con los límites del enunciado, no ha hecho falta aplazar nada')}; "
                           "la nivelación, por construcción, nunca cambia la duración.''')"))
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
                           "p_prop = proyecto.copy()\n"
                           "datos_prop, gantt_prop, dibujo = p_prop.desplazar(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True,\n"
                           "                                                  params=GANTT_PARAMS, **propuesta)\n"
                           "pinta_tikz(dibujo)"))
        celdas.append(code("cargas_prop = gantt_prop.data.loc['Total'].drop('H_total').astype(float)\n"
                           "duracion_inicial = proyecto.duration(datos_proyecto['duration'])\n"
                           "duracion_prop = len(cargas_prop)\n"
                           "cada_una = all(propuesta[a] <= holguras[a] for a in propuesta)\n"
                           "respeta = cada_una and duracion_prop == duracion_inicial\n"
                           "comprobacion = '; '.join(f'{a}: {d} ≤ {int(holguras[a])}' if d <= holguras[a] else f'{a}: {d} > {int(holguras[a])}' for a, d in propuesta.items())\n"
                           "Markdown(f'''Cada desplazamiento frente a su holgura total: {comprobacion}. Aplicados los dos a la vez, el proyecto dura {duracion_prop} semanas "
                           "(sin desplazar, {duracion_inicial}){\": no se alarga\" if duracion_prop == duracion_inicial else \": las dos actividades comparten holgura y juntas la superan\"}.\\n\\n"
                           "Cargas con la propuesta: {explica_cuadrados(cargas_prop)} (antes {suma_cuadrados_inicial:.0f}); máximo {int(cargas_prop.max())} operarios (antes {int(cargas.max())}).\\n\\n"
                           "**Respuesta:** la propuesta {\"respeta\" if respeta else \"no respeta\"} las holguras y la suma de cuadrados pasa de {suma_cuadrados_inicial:.0f} a "
                           "{suma_cuadrados_propuesta:.0f}: {\"mejora\" if suma_cuadrados_propuesta < suma_cuadrados_inicial else \"no mejora\"} el nivelado.''')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code('#| tbl-cap: "Suma de cuadrados para cada desplazamiento posible en la nivelación"\n#| label: tbl-cuadrados\n'
                           "proyecto_nivelado = proyecto.copy()\n"
                           "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(\n"
                           "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')\n"
                           "cuadrados"))
        celdas.append(code("calendario_inicial = proyecto.calendar(datos_proyecto['duration'])\n"
                           "texto_niv, elegidos = explica_nivelacion(cuadrados, calendario_inicial)\n"
                           "Markdown('El algoritmo recorre las no críticas por fin más temprano y aplica a cada una el desplazamiento de menor suma de cuadrados:\\n\\n' + texto_niv)"))
        celdas.append(code('#| fig-cap: "Diagrama de cargas de operarios tras la nivelación"\n#| label: fig-cargasNivelado\n'
                           "plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=None)"))
        celdas.append(code("cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)\n"
                           "suma_cuadrados_nivelado = float((cargas_niveladas ** 2).sum())\n"
                           "pd.DataFrame({'suma de cuadrados': [suma_cuadrados_inicial, suma_cuadrados_propuesta, suma_cuadrados_nivelado],\n"
                           "              'máximo de operarios': [cargas.max(), cargas_prop.max(), cargas_niveladas.max()]},\n"
                           "             index=['inicial', 'propuesta del jefe de obra', 'nivelación']).astype(int)"))
        celdas.append(code("mejor = 'la nivelación' if suma_cuadrados_nivelado < suma_cuadrados_propuesta else ('la propuesta' if suma_cuadrados_propuesta < suma_cuadrados_nivelado else 'ninguna (empatan)')\n"
                           "Markdown(f'''**Respuesta:** es mejor {mejor}: suma de cuadrados {suma_cuadrados_nivelado:.0f} con la nivelación "
                           "({lista_y([f'{a} {plural(d, \"semana\")}' for a, d in elegidos.items()]) if elegidos else 'sin desplazamientos'}) frente a {suma_cuadrados_propuesta:.0f} con la propuesta "
                           "y máximo de {int(cargas_niveladas.max())} frente a {int(cargas_prop.max())} operarios. El algoritmo prueba sistemáticamente todos los desplazamientos de cada "
                           "actividad no crítica; aun así, al decidir actividad a actividad, tampoco garantiza el óptimo global.''')"))
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
        celdas.append(code("q0, q1 = int((cargas_maquinas ** 2).sum()), int((cargas_maquinas_niv ** 2).sum())\n"
                           "M0, M1 = int(cargas_maquinas.max()), int(cargas_maquinas_niv.max())\n"
                           "efecto = ('empeora' if (q1 > q0 or M1 > M0) else ('mejora' if (q1 < q0 or M1 < M0) else 'no cambia'))\n"
                           "Markdown(f'''**Respuesta:** en este proyecto la carga de máquinas {efecto} con la nivelación de operarios (máximo {M0} → {M1}, suma de cuadrados {q0} → {q1}), "
                           "{\"pero no está garantizado\" if efecto != \"empeora\" else \"como efecto secundario\"}: la nivelación optimiza un único recurso y los desplazamientos que allanan la demanda de "
                           "operarios pueden concentrar la de máquinas. Sí, una nivelación puede empeorar el otro recurso; con varios recursos hay que nivelar con una función objetivo "
                           "combinada o comprobar cada recurso tras nivelar el principal.''')"))
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
    celdas.append(code(_AUX_REC))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario: fechas y holgura total de cada actividad"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(md("En el Gantt cada actividad ocupa sus semanas desde su inicio más temprano con su consumo semanal; la fila "
                     "*Total* es el consumo de cada semana."))
    celdas.append(code('#| fig-cap: "Diagrama de Gantt con el consumo semanal"\n#| label: fig-ganttInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Consumo semanal frente al suministro"\n#| label: fig-cargasInicial\n'
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[suministro])"))
    celdas.append(code("consumo = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "exceso = consumo[consumo > suministro]\n"
                       "detalle = '; '.join(f'semana {int(t)}: {int(c)} > {suministro}' for t, c in exceso.items())\n"
                       "Markdown(f'**Respuesta:** se supera el suministro en {semanas_txt(exceso.index)} ({detalle}); el consumo máximo es de {int(consumo.max())} litros.')"))
    celdas += apartado(2)
    celdas.append(md("Sin depósito, el suministro es un límite semanal como el de los operarios, y se aplica el algoritmo de "
                     "asignación.\n\n" + _REGLA_ASIGNAR + "\n\nTraza semana a semana:"))
    celdas.append(code("_ = proyecto.copy().asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                       "                           maximo=[suministro], report=True)"))
    celdas.append(code('#| tbl-cap: "Semanas con conflicto: orden de prioridad, qué entra y qué se aplaza"\n#| label: tbl-conflictos\n'
                       "tabla_asignacion(proyecto, datos_proyecto, [suministro])"))
    celdas.append(code('#| fig-cap: "Diagrama de Gantt tras la asignación"\n#| label: fig-ganttAsignado\n'
                       "proyecto_asignado = proyecto.copy()\n"
                       "datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                       "    maximo=[suministro], report=False, params=dict(period_width=1, **GANTT_PARAMS))\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("duracion_inicial = proyecto.duration(datos_proyecto['duration'])\n"
                       "duracion_asignada = len(gantt_asignado.data.columns.drop('H_total'))\n"
                       "esperas = esperas_asignacion(datos_asignados)\n"
                       "txt_esperas = lista_y([f\"{a} {plural(e, 'semana')}\" for a, e in esperas.items()]) if esperas else 'ninguna'\n"
                       "Markdown(f'**Respuesta:** sin almacenamiento el proyecto pasa de {duracion_inicial} a {duracion_asignada} semanas '\n"
                       "         f'(se alarga {plural(duracion_asignada - duracion_inicial, \"semana\")}); esperas introducidas: {txt_esperas}.')"))
    celdas.append(code('#| fig-cap: "Consumo semanal tras la asignación frente al suministro"\n#| label: fig-cargasAsignado\n'
                       "plot_cargas(gantt_df=gantt_asignado, limite_maximo_recursos=[suministro])"))
    celdas += apartado(3)
    celdas.append(md("Con depósito, lo no consumido una semana queda disponible para las siguientes: basta con que el consumo "
                     "acumulado hasta cada semana no supere lo entregado hasta entonces ($\\sum_{s \\le t} c_s \\le t \\cdot S$). "
                     "La fila *margen* es la diferencia, es decir, lo que hay en el depósito al final de cada semana."))
    celdas.append(code("acumulado = pd.DataFrame({'consumo acumulado': consumo.cumsum(), 'suministro acumulado': suministro * np.arange(1, len(consumo) + 1)})\n"
                       "acumulado['margen'] = acumulado['suministro acumulado'] - acumulado['consumo acumulado']\n"
                       "acumulado['cumple'] = (acumulado['margen'] >= 0).map({True: 'sí', False: 'no'})\n"
                       "limpia_tabla(acumulado).T"))
    celdas.append(code('#| fig-cap: "Consumo y suministro acumulados"\n#| label: fig-acumulado\n'
                       "ax = acumulado[['consumo acumulado', 'suministro acumulado']].plot(figsize=(7, 4)); ax.set_xlabel('semana'); ax.set_ylabel('litros');"))
    celdas.append(code("t_min = int(acumulado['margen'].idxmin())\n"
                       "fila = acumulado.loc[t_min]\n"
                       "if (acumulado['margen'] >= 0).all():\n"
                       "    texto = (f'es viable: en todas las semanas lo entregado cubre lo consumido. La semana más ajustada es la {t_min}: '\n"
                       "             f'{int(fila[\"suministro acumulado\"])} − {int(fila[\"consumo acumulado\"])} = {int(fila[\"margen\"])} litros de margen. '\n"
                       "             'El depósito absorbe los picos con lo sobrante de las semanas de menos consumo.')\n"
                       "else:\n"
                       "    t0 = int(acumulado.index[acumulado['margen'] < 0][0])\n"
                       "    f0 = acumulado.loc[t0]\n"
                       "    texto = (f'tampoco es viable: en la semana {t0} el consumo acumulado ({int(f0[\"consumo acumulado\"])}) supera lo entregado hasta entonces '\n"
                       "             f'({int(f0[\"suministro acumulado\"])}).')\n"
                       "Markdown('**Respuesta:** con depósito la programación original ' + texto)"))
    celdas += apartado(4)
    celdas.append(code("viable = bool((acumulado['margen'] >= 0).all())\n"
                       "Markdown(f'''Sin depósito el proyecto dura {duracion_asignada} semanas; con depósito, {str(duracion_inicial) + ' (la programación original es viable)' if viable else 'tampoco basta la programación original'}. "
                       "Un recurso no almacenable (mano de obra, maquinaria, un suministro que no cabe en depósito) impone una restricción semana a semana; uno almacenable "
                       "impone una restricción acumulada, mucho menos exigente. **Respuesta:** los operarios son recursos no almacenables (renovables): las horas de un operario "
                       "que no se usan esta semana no se guardan para la siguiente. Por eso se tratan con asignación, y los consumibles almacenables con balances acumulados "
                       "(y un depósito con capacidad para el mayor margen acumulado que haya que guardar).''')"))
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
    celdas.append(code(_AUX_REC))
    celdas += apartado(1)
    celdas.append(code(f"limite = {lim_malo}\n"
                       "imposibles = datos_proyecto.index[datos_proyecto['resources'] > limite].tolist()\n"
                       "necesita = ', '.join(f'{a} necesita {int(datos_proyecto.loc[a, \"resources\"])}' for a in imposibles)\n"
                       "Markdown(f'**Respuesta:** no. {necesita} operarios a la vez, más que los {limite} disponibles, y una actividad no puede ejecutarse '\n"
                       "         f'con menos operarios de los que requiere (no se admite alargarla). El mínimo factible es el mayor requerimiento de una actividad: '\n"
                       "         f'{int(datos_proyecto[\"resources\"].max())} operarios.')"))
    celdas += apartado(2)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario: fechas y holgura total de cada actividad"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Cargas con las actividades en su inicio más temprano"\n#| label: fig-cargasInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=None)"))
    celdas.append(md("La nivelación por mínimos cuadrados desplaza las actividades no críticas dentro de su holgura (sin cambiar "
                     "la duración); la tabla da la suma de cuadrados de cada desplazamiento probado (resaltado el mínimo):"))
    celdas.append(code('#| tbl-cap: "Suma de cuadrados para cada desplazamiento posible"\n#| label: tbl-cuadrados\n'
                       "cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "proyecto_nivelado = proyecto.copy()\n"
                       "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo = proyecto_nivelado.nivelar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')\n"
                       "cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)\n"
                       "cuadrados"))
    celdas.append(code("texto_niv, elegidos = explica_nivelacion(cuadrados, calendario)\n"
                       "pico0 = cargas.index[cargas == cargas.max()].tolist(); pico1 = cargas_niveladas.index[cargas_niveladas == cargas_niveladas.max()].tolist()\n"
                       "Markdown(texto_niv + f'''\\n\\nCargas con inicios más tempranos: {\", \".join(str(int(c)) for c in cargas)} (máximo {int(cargas.max())} en {semanas_txt(pico0)}); "
                       "tras nivelar: {\", \".join(str(int(c)) for c in cargas_niveladas)} (máximo {int(cargas_niveladas.max())} en {semanas_txt(pico1)}). "
                       "Sumas de cuadrados: {(cargas ** 2).sum():.0f} antes y {(cargas_niveladas ** 2).sum():.0f} después.\\n\\n"
                       "**Respuesta:** con las actividades en su inicio más temprano hacen falta **{int(cargas.max())} operarios** y tras nivelar **{int(cargas_niveladas.max())}**, "
                       "en ambos casos con la duración mínima de {duracion_minima} semanas.''')"))
    celdas.append(code('#| fig-cap: "Cargas tras la nivelación"\n#| label: fig-cargasNivelado\n'
                       "plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=None)"))
    celdas += apartado(3)
    celdas.append(md("Para cada número de operarios se aplica el algoritmo de asignación.\n\n" + _REGLA_ASIGNAR +
                     "\n\nCon el mínimo factible se muestra la traza completa, el porqué de cada aplazamiento y el Gantt:"))
    celdas.append(code("minimo = int(datos_proyecto['resources'].max())\n"
                       "maximo = int(cargas_niveladas.max())\n"
                       "p_min = proyecto.copy()\n"
                       "datos_min, gantt_min, dibujo_min = p_min.asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                       "                                                 maximo=[minimo], report=True, params=dict(period_width=1, **GANTT_PARAMS))"))
    celdas.append(code('#| tbl-cap: "Semanas con conflicto con el mínimo de operarios"\n#| label: tbl-conflictos\n'
                       "tabla_asignacion(proyecto, datos_proyecto, [minimo])"))
    celdas.append(code('#| fig-cap: "Gantt con el mínimo de operarios factible"\n#| label: fig-ganttMinimo\n'
                       "pinta_tikz(dibujo_min)"))
    celdas.append(code("filas = {}\n"
                       "for L in range(minimo, maximo + 1):\n"
                       "    p = proyecto.copy()\n"
                       "    d_L, g, _ = p.asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources', maximo=[L], report=False)\n"
                       "    esperas_L = esperas_asignacion(d_L)\n"
                       "    filas[L] = {'duración (semanas)': len(g.data.columns.drop('H_total')),\n"
                       "                'esperas introducidas': ', '.join(f\"{a} ({plural(e, 'semana')})\" for a, e in esperas_L.items()) or 'ninguna'}\n"
                       "tabla = pd.DataFrame(filas).T.rename_axis('operarios')\n"
                       "tabla.index = tabla.index.astype(int)\n"
                       "tabla['duración (semanas)'] = tabla['duración (semanas)'].astype(int)\n"
                       "tabla"))
    celdas.append(code("Markdown('**Respuesta:** ' + '; '.join(f'con {L} operarios, {plural(int(f[\"duración (semanas)\"]), \"semana\")}' for L, f in tabla.iterrows())\n"
                       "         + (f'. Con {maximo} operarios (el máximo del calendario nivelado) se alcanza ya la duración mínima de {duracion_minima} semanas.'\n"
                       "            if tabla.loc[maximo, 'duración (semanas)'] == duracion_minima else\n"
                       "            f'. Ni con {maximo} operarios la asignación alcanza la duración mínima de {duracion_minima} semanas: es una heurística y no reproduce el calendario nivelado.'))"))
    celdas.append(code('#| fig-cap: "Duración del proyecto según los operarios disponibles"\n#| label: fig-duracionOperarios\n'
                       "ax = tabla['duración (semanas)'].plot(marker='o', legend=False, figsize=(7, 4)); ax.set_ylabel('semanas'); ax.set_xlabel('operarios');"))
    celdas += apartado(4)
    celdas.append(md("Coste de cada opción: operarios × duración × 2 k€ (cada operario se paga todas las semanas del "
                     "proyecto) más duración × 5 k€ de coste fijo."))
    celdas.append(code("tabla['coste operarios'] = 2 * tabla.index * tabla['duración (semanas)']\n"
                       "tabla['coste fijo'] = 5 * tabla['duración (semanas)']\n"
                       "tabla['coste total'] = tabla['coste operarios'] + tabla['coste fijo']\n"
                       "display(tabla)\n"
                       "lineas = [f'- {L} operarios: 2 · {L} · {int(f[\"duración (semanas)\"])} + 5 · {int(f[\"duración (semanas)\"])} = {int(f[\"coste operarios\"])} + {int(f[\"coste fijo\"])} = {int(f[\"coste total\"])} k€' for L, f in tabla.iterrows()]\n"
                       "mejor = int(tabla['coste total'].idxmin())\n"
                       "Markdown('\\n'.join(lineas) + f'''\\n\\n**Respuesta:** conviene contratar **{mejor} operarios** ({int(tabla.loc[mejor, \"coste total\"])} k€, {tabla.loc[mejor, \"duración (semanas)\"]} semanas). "
                       "Más operarios acortan el proyecto pero cuestan cada semana; la tabla operarios-duración tiene escalones porque el plazo solo baja cuando el límite libera "
                       "un cuello de botella, y añadir un operario que no acorta el proyecto solo encarece.''')"))
    return celdas, f"{n} actividades; factibilidad y duración en función de los operarios"



PLANIFICADOR = (
    "def planificar(datos, prelaciones, limite, prioridad, interrumpible=False, max_periodos=200, registro=None):\n"
    "    '''Programación en paralelo periodo a periodo con un recurso renovable limitado.\n"
    "    prioridad(a, t) -> número: menor = antes. Devuelve (semanas trabajadas por actividad, inicio, fin).\n"
    "    Si se pasa una lista en registro, añade una entrada por semana con el orden, las que entran y las que esperan.'''\n"
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
    "        activas, esperan = [], []\n"
    "        for a in orden:\n"
    "            if rec[a] <= disponible:\n"
    "                activas.append(a); disponible -= rec[a]\n"
    "            else:\n"
    "                esperan.append((a, rec[a], limite - disponible))\n"
    "        if registro is not None:\n"
    "            registro.append(dict(semana=t, orden=list(orden), activas=list(activas), esperan=esperan,\n"
    "                                 en_curso=set(en_curso)))\n"
    "        for a in activas:\n"
    "            restante[a] -= 1; trabajo[a].append(t)\n"
    "        en_curso = {a for a in activas if restante[a] > 0}\n"
    "    inicio = pd.Series({a: (min(s) if s else np.nan) for a, s in trabajo.items()})\n"
    "    fin = pd.Series({a: (max(s) if s else np.nan) for a, s in trabajo.items()})\n"
    "    return trabajo, inicio, fin\n"
    "\n"
    "def tabla_decisiones(registro, clave, limite):\n"
    "    '''Semanas en que alguna actividad disponible tiene que esperar: orden, las que trabajan y por qué esperan las demás.'''\n"
    "    filas = []\n"
    "    for r in registro:\n"
    "        if not r['esperan']:\n"
    "            continue\n"
    "        etiqueta = lambda a: ', '.join(x for x in ('en curso' if a in r['en_curso'] else clave(a), f\"{int(datos_proyecto.loc[a, 'resources'])} op.\") if x)\n"
    "        orden = ' › '.join(f'{a} ({etiqueta(a)})' for a in r['orden'])\n"
    "        filas.append({'semana': r['semana'], 'orden de prioridad': orden,\n"
    "                      'trabajan': ', '.join(r['activas']) or '—',\n"
    "                      'esperan (no caben)': '; '.join(f'{a}: {usado} + {q} = {usado + q} > {limite}' for a, q, usado in r['esperan'])})\n"
    "    return pd.DataFrame(filas).set_index('semana') if filas else Markdown('Ninguna actividad disponible tiene que esperar.')\n"
    "\n"
    "def gantt_texto(trabajo, limite):\n"
    "    '''Gantt de pmule: semanas trabajadas por cada actividad (en tramos si se interrumpe) con los operarios\n"
    "    que usa en cada semana, y la carga total frente al límite.'''\n"
    "    T = max(max(s) for s in trabajo.values() if s)\n"
    "    tramos = {}\n"
    "    for a, semanas in trabajo.items():\n"
    "        q = int(datos_proyecto.loc[a, 'resources'])\n"
    "        seg = []\n"
    "        for t in sorted(semanas):\n"
    "            if seg and seg[-1][0] + seg[-1][1] == t - 1:\n"
    "                seg[-1] = (seg[-1][0], seg[-1][1] + 1, q)\n"
    "            else:\n"
    "                seg.append((t - 1, 1, q))\n"
    "        tramos[a] = seg\n"
    "    carga = [sum(int(datos_proyecto.loc[a, 'resources']) for a, s in trabajo.items() if t in s) for t in range(1, T + 1)]\n"
    "    extra = pd.DataFrame([carga, ['sí' if c <= limite else 'NO' for c in carga]], index=['Total', r'$\\le$ límite'], columns=range(1, T + 1))\n"
    "    return pinta_tikz(gantt_tramos_tikz(tramos, horizonte=T, extra_rows=extra, params={'segments_text_per_period': True}))\n")


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
    celdas.append(code(_AUX_REC))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario sin restricción de recursos"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_minima = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Cargas con los inicios más tempranos"\n#| label: fig-cargasInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[limite])"))
    celdas.append(code("cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "exceso = cargas[cargas > limite]\n"
                       "Markdown(f'**Respuesta:** sin límite el proyecto dura {duracion_minima} semanas; con los inicios más tempranos se superan los {limite} operarios en '\n"
                       "         f'{semanas_txt(exceso.index)} (cargas {\", \".join(f\"{int(c)}\" for c in exceso)}).')"))
    celdas += apartado(2)
    celdas.append(md("El planificador recorre las semanas: una actividad es *disponible* cuando han terminado sus "
                     "predecesoras; las que ya están en curso siguen (no se interrumpen) y las nuevas entran por "
                     "orden de prioridad mientras quepan en el límite (si una no cabe, se prueba la siguiente). A igualdad "
                     "de prioridad decide el orden alfabético."))
    celdas.append(code(PLANIFICADOR))
    celdas.append(code("reglas = {\n"
                       "    '(a) menor holgura total': lambda a, t: (calendario.loc[a, 'H_total'], a),\n"
                       "    '(b) menor duración (SPT)': lambda a, t: (datos_proyecto.loc[a, 'duration'], a),\n"
                       "    '(c) más operarios primero': lambda a, t: (-datos_proyecto.loc[a, 'resources'], a),\n"
                       "}\n"
                       "claves = {\n"
                       "    '(a) menor holgura total': lambda a: f\"H={int(calendario.loc[a, 'H_total'])}\",\n"
                       "    '(b) menor duración (SPT)': lambda a: f\"d={int(datos_proyecto.loc[a, 'duration'])}\",\n"
                       "    '(c) más operarios primero': lambda a: '',\n"
                       "}\n"
                       "resultados = {}\n"
                       "for nombre, regla in reglas.items():\n"
                       "    registro = []\n"
                       "    trabajo, inicio, fin = planificar(datos_proyecto, prelaciones, limite, regla, registro=registro)\n"
                       "    resultados[nombre] = dict(trabajo=trabajo, inicio=inicio, fin=fin, duracion=int(fin.max()), registro=registro)\n"
                       "pd.DataFrame({nombre: r['inicio'].astype(int).rename('inicio') for nombre, r in resultados.items()}).assign(**{'sin límite': calendario['inicio_mas_temprano'] + 1}).T"))
    celdas.append(md("La tabla anterior da la semana de inicio de cada actividad con cada regla. Para cada regla, primero las "
                     "semanas en que alguna actividad disponible tiene que esperar (con el orden de prioridad, las que trabajan "
                     "y la suma que no cabe) y después el Gantt de operarios:"))
    for etiqueta, nombre in (("a", "(a) menor holgura total"), ("b", "(b) menor duración (SPT)"), ("c", "(c) más operarios primero")):
        celdas.append(code(f'#| tbl-cap: "Esperas con la regla {nombre}"\n#| label: tbl-decisiones{etiqueta}\n'
                           f"tabla_decisiones(resultados['{nombre}']['registro'], claves['{nombre}'], limite)"))
        celdas.append(code(f'#| tbl-cap: "Programación con la regla {nombre}"\n#| label: tbl-regla{etiqueta}\n'
                           f"gantt_texto(resultados['{nombre}']['trabajo'], limite)"))
    celdas.append(code("duraciones = pd.Series({nombre: r['duracion'] for nombre, r in resultados.items()}, name='duración (semanas)')\n"
                       "duraciones['sin límite de recursos'] = duracion_minima\n"
                       "display(duraciones.to_frame().T)\n"
                       "d = duraciones.drop('sin límite de recursos')\n"
                       "mejores = [k for k, v in d.items() if v == d.min()]\n"
                       "Markdown('**Respuesta:** ' + '; '.join(f'{k}: {v} semanas' for k, v in d.items()) + f' (sin límite, {duracion_minima}). '\n"
                       "         + ('La mejor es la ' if len(mejores) == 1 else 'Las mejores son ') + lista_y(mejores) + f', con {d.min()} semanas.')"))
    celdas += apartado(3)
    celdas.append(md("La librería (`asignar`) también recorre las semanas, pero parte del calendario sin límite y lo va "
                     "corrigiendo: solo considera en cada semana las actividades que el calendario vigente sitúa en ella "
                     "y aplaza una semana las que no caben.\n\n" + _REGLA_ASIGNAR +
                     "\n\nEs, por tanto, parecida a la regla (a), pero con la holgura recalculada tras cada aplazamiento y "
                     "desempate por duración. Traza semana a semana:"))
    celdas.append(code("_ = proyecto.copy().asignar(data=datos_proyecto.copy(), duration_label='duration', resource_label='resources', maximo=[limite], report=True)"))
    celdas.append(code('#| tbl-cap: "Semanas con conflicto en la asignación de la librería"\n#| label: tbl-conflictos\n'
                       "tabla_asignacion(proyecto, datos_proyecto, [limite])"))
    celdas.append(code('#| fig-cap: "Asignación de la librería (Wiest-Levy)"\n#| label: fig-ganttAsignado\n'
                       "proyecto_asignado = proyecto.copy()\n"
                       "datos_asignados, gantt_asignado, dibujo = proyecto_asignado.asignar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources',\n"
                       "    maximo=[limite], report=False, params=dict(period_width=1, **GANTT_PARAMS))\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("duracion_libreria = len(gantt_asignado.data.columns.drop('H_total'))\n"
                       "comparacion = 'igual que' if duracion_libreria == d.min() else ('mejor que' if duracion_libreria < d.min() else 'peor que')\n"
                       "Markdown(f'**Respuesta:** la librería obtiene {duracion_libreria} semanas, {comparacion} la mejor regla ({d.min()}); las reglas dan '\n"
                       "         f'{\"; \".join(f\"{k}: {v}\" for k, v in d.items())}. Wiest-Levy también es una heurística, de modo que ninguna de las cuatro cifras es necesariamente el óptimo.')"))
    celdas += apartado(4)
    celdas.append(code("da = d['(a) menor holgura total']\n"
                       "nota = ('Aquí la regla de la holgura da el mejor resultado' if da == d.min() else\n"
                       "        f'Aquí, de hecho, la regla de la holgura ({da} semanas) no es la mejor: {lista_y(mejores)} {\"da\" if len(mejores) == 1 else \"dan\"} {d.min()} semanas')\n"
                       "Markdown(f'''**Respuesta:** ninguna regla garantiza el óptimo: el problema de programación con recursos limitados es NP-duro y las reglas de "
                       "prioridad son heurísticas; para proyectos pequeños se puede comprobar por enumeración o programación entera. La regla de la holgura mínima es la "
                       "única que usa la *estructura* del proyecto (qué actividades tienen margen y cuáles arrastrarían el plazo); SPT y «más operarios» solo miran la "
                       "actividad aislada. Por eso suele funcionar mejor, aunque no siempre: la holgura se calculó sin recursos y deja de ser exacta en cuanto se aplaza "
                       "algo. {nota}.''')"))
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
    celdas.append(code(_AUX_REC))
    celdas.append(code(PLANIFICADOR))
    celdas.append(code("regla = lambda a, t: (calendario.loc[a, 'H_total'], datos_proyecto.loc[a, 'duration'], a)\n"
                       "clave = lambda a: f\"H={int(calendario.loc[a, 'H_total'])}, d={int(datos_proyecto.loc[a, 'duration'])}\""))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico (sin límite de operarios)", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario sin límite de recursos (la regla de prioridad usa la holgura total)"\n#| label: tbl-calendario\n'
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Cargas con los inicios más tempranos frente al límite"\n#| label: fig-cargasInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[limite])"))
    celdas.append(md("El planificador recorre las semanas: una actividad es *disponible* cuando han terminado sus "
                     "predecesoras; las disponibles se ordenan por la regla (menor holgura total, luego menor duración, luego "
                     "orden alfabético) y entran mientras quepan en el límite. Sin interrupciones, las ya empezadas van "
                     "primero. La primera tabla recoge las semanas en que alguna actividad disponible tiene que esperar y por qué; "
                     "la segunda, el Gantt de operarios."))
    celdas.append(code('#| tbl-cap: "Esperas sin interrupciones"\n#| label: tbl-decisionesNo\n'
                       "registro_no = []\n"
                       "trabajo_no, inicio_no, fin_no = planificar(datos_proyecto, prelaciones, limite, regla, interrumpible=False, registro=registro_no)\n"
                       "tabla_decisiones(registro_no, clave, limite)"))
    celdas.append(code('#| tbl-cap: "Programación sin interrupciones"\n#| label: tbl-sinInterrupcion\n'
                       "gantt_texto(trabajo_no, limite)"))
    celdas.append(code("Markdown(f'**Respuesta:** sin interrupciones el proyecto dura **{int(fin_no.max())} semanas**, '\n"
                       "         f'{int(fin_no.max()) - duracion_minima} más que sin límite de operarios ({duracion_minima}).')"))
    celdas += apartado(2)
    celdas.append(md("Ahora, cada semana se ordenan por la misma regla *todas* las disponibles, también las empezadas: una "
                     "actividad en curso deja de trabajar esa semana si otras de más prioridad ocupan los operarios."))
    celdas.append(code('#| tbl-cap: "Esperas con interrupciones permitidas"\n#| label: tbl-decisionesSi\n'
                       "registro_si = []\n"
                       "trabajo_si, inicio_si, fin_si = planificar(datos_proyecto, prelaciones, limite, regla, interrumpible=True, registro=registro_si)\n"
                       "tabla_decisiones(registro_si, clave, limite)"))
    celdas.append(code('#| tbl-cap: "Programación con interrupciones permitidas"\n#| label: tbl-conInterrupcion\n'
                       "gantt_texto(trabajo_si, limite)"))
    celdas.append(code("def interrupciones(semanas):\n"
                       "    return sum(1 for s1, s2 in zip(semanas, semanas[1:]) if s2 > s1 + 1)\n"
                       "inter = pd.Series({a: interrupciones(s) for a, s in trabajo_si.items()})\n"
                       "inter = inter[inter > 0]\n"
                       "detalle_inter = lista_y([f\"{a} ({'una vez' if v == 1 else str(v) + ' veces'}: trabaja en las semanas {', '.join(map(str, trabajo_si[a]))})\" for a, v in inter.items()])\n"
                       "Markdown(f'''**Respuesta:** con interrupciones el proyecto dura **{int(fin_si.max())} semanas**. "
                       "{('Se interrumpe' + ('n ' if len(inter) > 1 else ' ') + detalle_inter + '.') if len(inter) else 'Ninguna actividad llega a interrumpirse: en ninguna semana aparece una actividad disponible más prioritaria que otra ya en curso y que necesite sus operarios.'}''')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Inicio y fin de cada actividad en las dos programaciones"\n#| label: tbl-comparacion\n'
                       "comparacion = pd.DataFrame({'inicio (no interr.)': inicio_no.astype(int), 'fin (no interr.)': fin_no.astype(int),\n"
                       "              'inicio (interr.)': inicio_si.astype(int), 'fin (interr.)': fin_si.astype(int),\n"
                       "              'semanas trabajadas (interr.)': {a: len(s) for a, s in trabajo_si.items()}})\n"
                       "comparacion"))
    celdas.append(code("cambian = [a for a in comparacion.index if (comparacion.loc[a, 'inicio (no interr.)'], comparacion.loc[a, 'fin (no interr.)']) != (comparacion.loc[a, 'inicio (interr.)'], comparacion.loc[a, 'fin (interr.)'])]\n"
                       "ganancia = int(fin_no.max()) - int(fin_si.max())\n"
                       "if ganancia > 0:\n"
                       "    texto = (f'Hay ganancia de {plural(ganancia, \"semana\")}: al permitir interrumpir no hay que esperar a que termine una actividad poco urgente para '\n"
                       "             f'liberar operarios, y las más prioritarias entran en cuanto están disponibles. Cambian las fechas de {lista_y(cambian)}.')\n"
                       "elif ganancia == 0:\n"
                       "    texto = ('No hay ganancia: ' + (f'aunque cambian las fechas de {lista_y(cambian)}, ' if cambian else 'las dos programaciones coinciden semana a semana; ')\n"
                       "             + 'la regla de prioridad ya evitaba bloquear los operarios con actividades holgadas.')\n"
                       "else:\n"
                       "    texto = (f'Interrumpir empeora el resultado en {plural(-ganancia, \"semana\")}: el planificador decide semana a semana sin mirar adelante, y '\n"
                       "             f'parar actividades a medias retrasa sus sucesoras. Cambian las fechas de {lista_y(cambian)}.')\n"
                       "Markdown(f'**Respuesta:** {int(fin_no.max())} semanas sin interrupciones frente a {int(fin_si.max())} con ellas. ' + texto +\n"
                       "         ' Ojo: el planificador es una heurística, así que permitir interrupciones no garantiza mejorar (puede incluso empeorar) y, cuando mejora, fragmenta el trabajo.')"))
    celdas += apartado(4)
    celdas.append(md("**Respuesta:** interrumpir tiene costes que el modelo ignora: volver a montar y desmontar (tiempos de "
                     "preparación), pérdida de ritmo y de aprendizaje, el material en curso inmovilizado, la "
                     "coordinación de los operarios y la dificultad de seguimiento. Compensa cuando el ahorro de "
                     "plazo vale más que esos costes: actividades largas, modulares (pueden partirse en lotes) y "
                     "sin costes de preparación apreciables; no compensa en tareas con fraguados, curados o "
                     "procesos continuos."))
    return celdas, f"{n} actividades; {limite} operarios; interrumpir o no interrumpir"


def _limites_simultaneos(rng):
    """Dos recursos con límite a la vez: asignación simultánea (asignar_multi) frente a aplicar
    el algoritmo de un solo recurso y comprobar el otro."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (0 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 9 <= i["T"] <= 20):
            return False
        nombres = list(i["D"].index)
        op = P.recursos(rng, nombres, hi=4).rename("operarios")
        eq = pd.Series(rng.integers(0, 3, size=len(nombres)), index=nombres, name="equipos")
        if eq.max() < 2 or (eq > 0).sum() < 4:
            return False
        L1 = int(op.max()) + int(rng.integers(0, 2))
        L2 = int(eq.max())
        df = P.tabla_base(i, duration=i["D"], operarios=op, equipos=eq)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = i["proyecto"].copy().asignar_multi(df.copy(), "duration", ["operarios", "equipos"], [L1, L2])
        except Exception:
            return False
        if not (1 <= res.duracion - int(i["T"]) <= 4):
            return False
        espera = res.traza[res.traza.decision == "espera"]
        bloqueos = set(", ".join(espera.bloqueada_por).replace(" ", "").split(","))
        if not {"operarios", "equipos"} <= bloqueos:
            return False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                p1 = i["proyecto"].copy()
                d1, _, _ = p1.asignar(data=df.copy(), duration_label="duration", resource_label="operarios",
                                      maximo=[L1], report=False)
                g2, _ = p1.gantt_cargas(d1, "duration", "equipos", report=False, tikz=False)
        except Exception:
            return False
        c2 = g2.data.loc["Total"].drop("H_total", errors="ignore").astype(float)
        if c2.max() <= L2:           # aplicar solo operarios debe violar el límite de equipos
            return False
        estado.update(df=df, L1=L1, L2=L2)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])),
                    max_intentos=8000)
    df, L1, L2 = estado["df"], estado["L1"], estado["L2"]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones (semanas) y recursos necesarios "
                 "se adjunta. Cada actividad necesita un número fijo de operarios y de equipos (grúas, andamios…) "
                 f"durante toda su duración. La empresa dispone como máximo de {L1} operarios y {L2} equipos "
                 "en cada semana, **a la vez**.")]
    celdas.append(preguntas([
        "Dibuje las cargas de operarios y de equipos con todas las actividades en su inicio más temprano. "
        "¿En qué semanas se supera cada límite?",
        "Aplique el algoritmo de asignación comprobando los dos recursos a la vez (una actividad solo empieza si cabe "
        "en ambos en todas las semanas que ocuparía). Indique el calendario resultante, la duración del proyecto "
        "y, para cada actividad que espera, qué recurso se lo impide.",
        "Aplique ahora el algoritmo de asignación solo a los operarios, ignorando los equipos, y compruebe qué "
        "ocurre con los equipos. ¿Qué enseña esto sobre resolver los recursos de uno en uno?",
        "¿Qué recurso es el cuello de botella? Construya la tabla de duración del proyecto al ampliar cada límite "
        "hasta dos unidades y recomiende qué ampliar primero."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion_minima = proyecto.duration(datos_proyecto['duration'])\n"
                       "recursos = ['operarios', 'equipos']\n"
                       f"limites = [{L1}, {L2}]"))
    celdas.append(code(_AUX_REC + "\n\n" + _AUX_MULTI))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico (sin límites)", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario sin límite de recursos"\n#| label: tbl-calendario\n'
                       "calendario_libre = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario_libre[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Cargas de operarios con los inicios más tempranos"\n#| label: fig-cargasOperarios\n'
                       "gantt_op, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_op, limite_maximo_recursos=limites[0])"))
    celdas.append(code('#| fig-cap: "Cargas de equipos con los inicios más tempranos"\n#| label: fig-cargasEquipos\n'
                       "gantt_eq, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'equipos', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "plot_cargas(gantt_df=gantt_eq, limite_maximo_recursos=limites[1])"))
    celdas.append(code("cargas_op = gantt_op.data.loc['Total'].drop('H_total').astype(float)\n"
                       "cargas_eq = gantt_eq.data.loc['Total'].drop('H_total').astype(float)\n"
                       "exceso_op = [int(t) for t, c in cargas_op.items() if c > limites[0]]\n"
                       "exceso_eq = [int(t) for t, c in cargas_eq.items() if c > limites[1]]\n"
                       "det = lambda cargas, ex, L: ' (' + '; '.join(f'{int(cargas[t])} > {L}' for t in ex) + ')' if ex else ''\n"
                       "Markdown(f'**Respuesta:** los operarios (límite {limites[0]}) se superan en {semanas_txt(exceso_op)}{det(cargas_op, exceso_op, limites[0])}; '\n"
                       "         f'los equipos (límite {limites[1]}), en {semanas_txt(exceso_eq)}{det(cargas_eq, exceso_eq, limites[1])}.')"))
    celdas += apartado(2)
    celdas.append(md("El algoritmo recorre las semanas; en cada una, las actividades disponibles (predecesoras terminadas) "
                     "se ordenan por prioridad (menor inicio más tardío, luego menor duración) y cada una empieza sólo si "
                     "cabe en **los dos** recursos durante todas las semanas que ocuparía; si no, espera. La traza indica "
                     "entre paréntesis el recurso que bloquea a cada actividad que espera."))
    celdas.append(code("resultado = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', recursos, limites, report=True, tikz=True, params=GANTT_PARAMS)\n"
                       "calendario = pd.DataFrame({'inicio (semana)': resultado.inicios + 1, 'fin (semana)': resultado.fines,\n"
                       "                          'espera (semanas)': resultado.esperas}).fillna(0).astype(int)\n"
                       "calendario"))
    celdas.append(code("esperas = resultado.traza[resultado.traza.decision == 'espera']\n"
                       "esperas = esperas.rename(columns={'bloqueada_por': 'recurso que se satura'})\n"
                       "esperas[['periodo', 'actividad', 'recurso que se satura']].reset_index(drop=True)"))
    celdas.append(code("Markdown('Por qué espera cada actividad (ocupación ya comprometida en esa semana + lo que pide la actividad, frente al límite):\\n\\n'\n"
                       "         + '\\n'.join(explica_esperas_multi(resultado.traza, datos_proyecto, recursos, limites)))"))
    celdas.append(code('#| fig-cap: "Gantt de operarios tras la asignación simultánea (filas 💤: esperas)"\n#| label: fig-ganttOperarios\n'
                       "pinta_tikz(resultado.dibujos['operarios'])"))
    celdas.append(code('#| fig-cap: "Gantt de equipos tras la asignación simultánea"\n#| label: fig-ganttEquipos\n'
                       "pinta_tikz(resultado.dibujos['equipos'])"))
    celdas.append(code('#| tbl-cap: "Carga de cada recurso semana a semana frente a su límite"\n#| label: tbl-cargasLimites\n'
                       "limpia_tabla(pd.concat([resultado.cargas.add_suffix(' ocupados'), resultado.limites.add_suffix(' (límite)')], axis=1)).T"))
    celdas.append(code('#| fig-cap: "Cargas de operarios tras la asignación simultánea"\n#| label: fig-asignacionOperarios\n'
                       "plot_cargas(gantt_df=resultado.gantts['operarios'], limite_maximo_recursos=list(resultado.limites['operarios']))"))
    celdas.append(code('#| fig-cap: "Cargas de equipos tras la asignación simultánea"\n#| label: fig-asignacionEquipos\n'
                       "plot_cargas(gantt_df=resultado.gantts['equipos'], limite_maximo_recursos=list(resultado.limites['equipos']))"))
    celdas.append(code("bloq = esperas.groupby('actividad')['recurso que se satura'].agg(lambda s: lista_y(sorted(set(', '.join(s).split(', ')))))\n"
                       "txt_bloq = lista_y([f'{a} ({r})' for a, r in bloq.items()]) if len(bloq) else 'ninguna'\n"
                       "Markdown(f'**Respuesta:** el calendario es el de la tabla de inicios y fines; la duración pasa de {duracion_minima} a {resultado.duracion} semanas. '\n"
                       "         f'Esperan, por el recurso indicado: {txt_bloq}.')"))
    celdas += apartado(3)
    celdas.append(code("p_solo = proyecto.copy()\n"
                       "datos_solo, gantt_solo, dibujo_solo = p_solo.asignar(data=datos_proyecto.copy(), duration_label='duration',\n"
                       "                                                     resource_label='operarios', maximo=[limites[0]], report=True, params=dict(period_width=1, **GANTT_PARAMS))\n"
                       "gantt_eq_solo, _ = p_solo.gantt_cargas(datos_solo, 'duration', 'equipos', report=False, tikz=False)\n"
                       "cargas_eq_solo = gantt_eq_solo.data.loc['Total'].drop('H_total').astype(float)\n"
                       "exceso = [int(t) for t, c in cargas_eq_solo.items() if c > limites[1]]"))
    celdas.append(code('#| fig-cap: "Gantt de operarios aplicando el algoritmo sólo a los operarios"\n#| label: fig-ganttSoloOperarios\n'
                       "pinta_tikz(dibujo_solo)"))
    celdas.append(code('#| fig-cap: "Cargas de equipos que resultan de ese calendario"\n#| label: fig-cargasEquiposSolo\n'
                       "plot_cargas(gantt_df=gantt_eq_solo, limite_maximo_recursos=limites[1])"))
    celdas.append(code("duracion_solo = len(gantt_solo.data.columns.drop('H_total'))\n"
                       "det_solo = '; '.join(f'semana {t}: {int(cargas_eq_solo[t])} > {limites[1]}' for t in exceso)\n"
                       "Markdown(f'**Respuesta:** atendiendo solo a los operarios el proyecto duraría {duracion_solo} semanas (frente a {resultado.duracion} con los dos límites), '\n"
                       "         f'pero el calendario no es factible: los equipos se superan en {semanas_txt(exceso)} ({det_solo}). '\n"
                       "         'Resolver un recurso desplaza actividades y puede crear (o dejar) picos en el otro: hay que comprobar todos a la vez, '\n"
                       "         'y corregir uno a uno no garantiza converger a un calendario factible.')"))
    celdas += apartado(4)
    celdas.append(code("filas = {}\n"
                       "for a in range(3):\n"
                       "    for b in range(3):\n"
                       "        r = proyecto.copy().asignar_multi(datos_proyecto.copy(), 'duration', recursos, [limites[0] + a, limites[1] + b])\n"
                       "        filas[(limites[0] + a, limites[1] + b)] = r.duracion\n"
                       "tabla = pd.Series(filas).unstack()\n"
                       "tabla.index.name, tabla.columns.name = 'operarios', 'equipos'\n"
                       "tabla"))
    celdas.append(code("base = int(tabla.iloc[0, 0])\n"
                       "mejora_op = base - int(tabla.iloc[1, 0])\n"
                       "mejora_eq = base - int(tabla.iloc[0, 1])\n"
                       "mejora_ambos = base - int(tabla.iloc[1, 1])\n"
                       "if mejora_op > mejora_eq:\n"
                       "    texto = 'los operarios son el cuello de botella: un operario más ahorra más semanas que un equipo más, así que conviene ampliarlos primero'\n"
                       "elif mejora_eq > mejora_op:\n"
                       "    texto = 'los equipos son el cuello de botella: un equipo más ahorra más semanas que un operario más, así que conviene ampliarlos primero'\n"
                       "elif mejora_op > 0:\n"
                       "    texto = 'los dos recursos limitan por igual: un operario o un equipo más ahorran lo mismo; decidiría el coste de cada unidad'\n"
                       "else:\n"
                       "    texto = f'ampliar uno solo de los recursos no acorta el proyecto; ampliando los dos se ganan {plural(mejora_ambos, \"semana\")}'\n"
                       "Markdown(f'''Con los límites actuales: {base} semanas. Con un operario más: {int(tabla.iloc[1, 0])} ({plural(mejora_op, \"semana\")} menos); con un equipo más: "
                       "{int(tabla.iloc[0, 1])} ({plural(mejora_eq, \"semana\")} menos); con uno de cada: {int(tabla.iloc[1, 1])}.\\n\\n"
                       "**Respuesta:** {texto}.{' Ampliando los dos se gana más que con cualquiera por separado: cuando ambos están saturados, ampliar uno solo mueve el cuello de botella al otro.' if mejora_ambos > max(mejora_op, mejora_eq) else ''}''')"))
    return celdas, f"{n} actividades; {L1} operarios y {L2} equipos a la vez"


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
    Variante("limites-simultaneos", 3, "dos recursos con límite a la vez: asignación simultánea (calendario, recurso que bloquea) frente a resolver uno solo",
             _limites_simultaneos),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
