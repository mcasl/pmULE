"""Colección 14 — Comprensión conceptual: diseñar una red con rasgos dados, razonar con datos
incompletos, verdadero/falso con contraejemplo, detectar programaciones inválidas y preguntas
tipo test con exportación a Moodle (GIFT)."""
import networkx as nx
import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="14_comprension", prefijo="Comprension",
            titulo="Comprensión conceptual",
            descripcion="Diseñar redes con rasgos dados, acotar la duración con datos incompletos, afirmaciones "
                        "verdadero/falso con contraejemplo, programaciones que violan las prelaciones y "
                        "cuestionarios tipo test exportables a Moodle (GIFT).")


def _disena(rng):
    """Diseñar un proyecto que cumpla unos rasgos: el generador busca uno y lo verifica."""
    n = int(rng.choice([6, 7]))
    # se busca un proyecto con varias rutas críticas y alguna ficticia; las condiciones del enunciado
    # se toman de él (así siempre existe solución)
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 2 and 2 <= i["n_criticas"] <= 3
                    and 3 <= i["n_rutas"] <= 6 and (i["calendario"]["H_total"] > 0).sum() >= 1,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])), max_intentos=6000)
    D = info["D"]
    n_crit, n_dum, n_ini = int(info["n_criticas"]), int(info["n_dummies"]), len(info["iniciales"])
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md(f"Diseñe un proyecto de **{n} actividades** (A, B, …) con duraciones enteras que cumpla "
                 "*todas* estas condiciones:\n\n"
                 f"- exactamente {n_ini} actividades iniciales;\n"
                 f"- su grafo PERT con el mínimo número de ficticias tiene exactamente {n_dum} ficticia(s);\n"
                 f"- tiene exactamente {n_crit} rutas críticas;\n"
                 "- al menos una actividad con holgura total positiva;\n"
                 "- ninguna prelación redundante.")]
    celdas.append(preguntas([
        "Escriba el cuadro de prelaciones y duraciones de su proyecto.",
        "Compruebe con la librería cada una de las condiciones.",
        "¿Qué condición le ha resultado más difícil de conseguir y por qué? ¿Qué cambio mínimo en las "
        "duraciones haría que sólo hubiera una ruta crítica?",
        "¿Es único el proyecto que cumple las condiciones? Dé otro distinto (basta cambiar una prelación o "
        "una duración sin violar ninguna condición) o argumente que no existe."]))
    celdas.append(md("# Resolución"))
    celdas.append(md("Hay muchas soluciones; la que sigue la encontró un programa que genera proyectos al azar "
                     "y comprueba las condiciones (el mismo método que puede usar el lector: proponer, "
                     "comprobar, corregir). Para tener varias rutas críticas conviene construir rutas en "
                     "paralelo y ajustar después las duraciones para que sumen lo mismo; las ficticias "
                     "aparecen cuando dos actividades comparten algunas predecesoras pero no todas."))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Una solución"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas += apartado(2)
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "G = nx.DiGraph([(p, a) for a, ps in prelaciones.immediate_predecessor.items() for p in ps]); G.add_nodes_from(datos_proyecto.index)\n"
                       "redundantes = sorted(set(G.edges) - set(nx.transitive_reduction(G).edges))\n"
                       "comprobacion = pd.DataFrame({\n"
                       f"    'exigido': [{n}, {n_ini}, {n_dum}, {n_crit}, '≥ 1', 0],\n"
                       "    'obtenido': [len(datos_proyecto), len(prelaciones.start_activities), len(proyecto.dummies), len(rutas_criticas),\n"
                       "                 int((calendario['H_total'] > 0).sum()), len(redundantes)]},\n"
                       "    index=['actividades', 'iniciales', 'ficticias', 'rutas críticas', 'con holgura positiva', 'prelaciones redundantes'])\n"
                       "comprobacion['cumple'] = [str(e) == str(o) if e != '≥ 1' else o >= 1 for e, o in zip(comprobacion['exigido'], comprobacion['obtenido'])]\n"
                       "comprobacion"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las rutas críticas", label="fig-pert")
    celdas.append(code("for nombre, acts in rutas_criticas.items():\n"
                       "    print(f'{nombre}: {\" - \".join(acts)}')\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']]"))
    celdas += apartado(3)
    celdas.append(code("R = proyecto.path_matrix(dummies=False)\n"
                       "Rt = R @ datos_proyecto['duration'][R.columns]\n"
                       "criticas = Rt.index[Rt == Rt.max()].tolist()\n"
                       "# actividad que está en una ruta crítica pero no en todas: alargarla 1 deja una sola crítica\n"
                       "solo_en_una = [a for a in R.columns if 0 < R.loc[criticas, a].sum() < len(criticas)]\n"
                       "a_cambiar = solo_en_una[0] if solo_en_una else None\n"
                       "if a_cambiar:\n"
                       "    d2 = datos_proyecto['duration'].copy(); d2[a_cambiar] += 1\n"
                       "    n_crit2 = len(proyecto.critical_path(d2))\n"
                       "    display(Markdown(f'La condición más exigente suele ser el número exacto de rutas críticas (hay que igualar sumas). Alargando {a_cambiar} un día quedan {n_crit2} ruta(s) crítica(s).'))\n"
                       "else:\n"
                       "    display(Markdown('Todas las actividades críticas están en todas las rutas críticas; habría que acortar una actividad común y alargar otra.'))"))
    celdas += apartado(4)
    celdas.append(md("No es único: en general hay muchísimos proyectos que cumplen las condiciones. Por "
                     "ejemplo, multiplicar todas las duraciones por 2 conserva las rutas críticas, las "
                     "holguras positivas, las ficticias y las iniciales; o permutar los nombres de las "
                     "actividades. Una solución *estructuralmente* distinta se obtiene cambiando una prelación "
                     "y recomprobando."))
    celdas.append(code("datos_otro = datos_proyecto.copy()\n"
                       "datos_otro['duration'] = datos_otro['duration'] * 2\n"
                       "p2 = PredecessorTable.from_dataframe_of_strings(datos_otro, activity='activity', predecessor='predecessor').create_project()\n"
                       "Markdown(f'Con las duraciones dobladas: {len(p2.critical_path(datos_otro[\"duration\"]))} rutas críticas, {len(p2.dummies)} ficticias, '\n"
                       "         f'{int((p2.calendar(datos_otro[\"duration\"])[\"H_total\"] > 0).sum())} actividades con holgura: las mismas condiciones.')"))
    return celdas, f"{n} actividades, {n_ini} iniciales, {n_dum} ficticias, {n_crit} rutas críticas"


def _datos_insuficientes(rng):
    """Faltan duraciones: qué puede y qué no puede concluirse; cotas de la duración."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9):
            return False
        cal = i["calendario"]
        # dos actividades con duración desconocida: una crítica y una con holgura ≥ 2
        crit = [a for a in cal.index if cal.loc[a, "H_total"] == 0]
        holg = [a for a in cal.index if cal.loc[a, "H_total"] >= 2]
        if not crit or not holg:
            return False
        estado["X"] = crit[int(rng.integers(0, len(crit)))]
        estado["Y"] = holg[int(rng.integers(0, len(holg)))]
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, T = info["D"], info["calendario"], int(info["T"])
    X, Y = estado["X"], estado["Y"]
    lo_X, hi_X = max(1, int(D[X]) - 2), int(D[X]) + 3
    lo_Y, hi_Y = max(1, int(D[Y]) - 1), int(D[Y]) + int(cal.loc[Y, "H_total"]) + 2
    dur_txt = pd.Series({a: (f"{lo_X}-{hi_X}" if a == X else f"{lo_Y}-{hi_Y}" if a == Y else str(int(D[a]))) for a in D.index}, name="duration")
    df = P.tabla_base(info, duration=dur_txt)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones se adjunta. De dos actividades sólo se conoce "
                 "un intervalo de duraciones posibles (en semanas), anotado como `mín-máx`.")]
    celdas.append(preguntas([
        "Sin conocer las dos duraciones, ¿qué puede afirmarse con seguridad sobre la duración del proyecto? "
        "Dé la mejor cota inferior y la mejor cota superior y diga con qué combinación de valores se "
        "alcanza cada una.",
        f"¿Hay actividades que son críticas *para cualquier* valor de las duraciones desconocidas? ¿Y "
        "actividades que no lo son nunca? Use la matriz de rutas.",
        f"Para qué valores de la duración de {Y} deja de tener holgura (se vuelve crítica), suponiendo "
        f"{X} en su valor mínimo? ¿Y suponiéndola en su valor máximo?",
        "¿Qué dato adicional reduciría más la incertidumbre sobre la duración del proyecto: conocer la "
        f"duración de {X} o la de {Y}? Justifíquelo con las cotas."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "def rango(s):\n"
                       "    s = str(s)\n"
                       "    return (int(s.split('-')[0]), int(s.split('-')[1])) if '-' in s else (int(s), int(s))\n"
                       "minimos = datos_proyecto['duration'].map(lambda s: rango(s)[0])\n"
                       "maximos = datos_proyecto['duration'].map(lambda s: rango(s)[1])\n"
                       f"X, Y = '{X}', '{Y}'"))
    celdas += apartado(1)
    celdas.append(md("τ = ‖R·t‖∞ es creciente en cada duración, luego la cota inferior se alcanza con todas las "
                     "desconocidas en su mínimo y la superior con todas en su máximo: no hace falta probar "
                     "combinaciones."))
    celdas.append(code("Rt_min = R @ minimos[R.columns]; Rt_max = R @ maximos[R.columns]\n"
                       "cota_inf, cota_sup = int(Rt_min.max()), int(Rt_max.max())\n"
                       "display(pd.DataFrame({'actividades': {r: ' - '.join(R.columns[R.loc[r] == 1]) for r in R.index}, 'duración mínima': Rt_min, 'duración máxima': Rt_max}))\n"
                       "Markdown(f'{cota_inf} ≤ τ ≤ {cota_sup}. La cota inferior se alcanza con {X} = {minimos[X]} y {Y} = {minimos[Y]} (ruta {Rt_min.idxmax()}); la superior con {X} = {maximos[X]} y {Y} = {maximos[Y]} (ruta {Rt_max.idxmax()}).')"))
    celdas += apartado(2)
    celdas.append(md("Una actividad es crítica siempre si pertenece a *toda* ruta que pueda ser la más larga "
                     "para alguna combinación de valores. Como sólo hay dos duraciones inciertas, basta probar "
                     "las cuatro combinaciones extremas (τ es lineal a trozos y el máximo de R·t cambia de "
                     "ruta sólo en esos vértices o entre ellos) y, para estar seguros, una rejilla de valores "
                     "intermedios."))
    celdas.append(code("from itertools import product\n"
                       "siempre, nunca = set(R.columns), set(R.columns)\n"
                       "for dX, dY in product(range(minimos[X], maximos[X] + 1), range(minimos[Y], maximos[Y] + 1)):\n"
                       "    t = minimos.copy(); t[X], t[Y] = dX, dY\n"
                       "    Rt = R @ t[R.columns]\n"
                       "    criticas_ahora = set(R.columns[(R.loc[Rt == Rt.max()].sum() > 0)])\n"
                       "    siempre &= criticas_ahora; nunca -= criticas_ahora\n"
                       "Markdown(f'Críticas para cualquier valor: {\", \".join(sorted(siempre)) or \"ninguna\"}. Nunca críticas: {\", \".join(sorted(nunca)) or \"ninguna\"}. '\n"
                       "         f'Las demás ({\", \".join(sorted(set(R.columns) - siempre - nunca))}) lo son o no según los valores.')"))
    celdas += apartado(3)
    celdas.append(code("def holgura_Y(dX, dY):\n"
                       "    t = minimos.copy(); t[X], t[Y] = dX, dY\n"
                       "    return int(proyecto.calendar(t).loc[Y, 'H_total'])\n"
                       "tabla = pd.DataFrame({f'{X} = {dX}': {dY: holgura_Y(dX, dY) for dY in range(minimos[Y], maximos[Y] + 1)} for dX in (minimos[X], maximos[X])})\n"
                       "tabla.index.name = f'duración de {Y}'\n"
                       "display(tabla.T)\n"
                       "umbrales = {col: next((dY for dY, h in tabla[col].items() if h == 0), None) for col in tabla.columns}\n"
                       "Markdown(f'{Y} se vuelve crítica a partir de ' + '; '.join(f'{Y} = {u} cuando {col}' if u is not None else f'nunca cuando {col}' for col, u in umbrales.items()) + '. '\n"
                       "         f'Cuanto más dura {X} (crítica), más holgura tiene {Y} y más tarde se vuelve crítica.')"))
    celdas += apartado(4)
    celdas.append(code("def amplitud(fija):\n"
                       "    # anchura del intervalo de τ cuando se conoce la actividad `fija` (promedio sobre sus valores posibles)\n"
                       "    anchuras = []\n"
                       "    for v in range(minimos[fija], maximos[fija] + 1):\n"
                       "        lo, hi = minimos.copy(), maximos.copy(); lo[fija] = hi[fija] = v\n"
                       "        anchuras.append(int((R @ hi[R.columns]).max() - (R @ lo[R.columns]).max()))\n"
                       "    return float(np.mean(anchuras))\n"
                       "amp = pd.Series({f'conociendo {X}': amplitud(X), f'conociendo {Y}': amplitud(Y), 'sin conocer ninguna': float(cota_sup - cota_inf)})\n"
                       "display(amp.to_frame('anchura del intervalo de τ').T)\n"
                       "Markdown(f'Conviene averiguar primero {X if amp.iloc[0] < amp.iloc[1] else Y}: es la que más estrecha el intervalo de la duración. '\n"
                       "         f'En general, la incertidumbre de una actividad crítica se traslada entera a τ; la de una actividad con holgura sólo cuenta cuando agota la holgura.')"))
    return celdas, f"{n} actividades; duraciones de {X} y {Y} en intervalos; cotas de τ"


AFIRMACIONES = [
    # (texto, verdadera?, clave para el contraejemplo o la justificación)
    ("Una actividad con holgura libre cero es crítica.", False, "libre0_no_critica"),
    ("Si una actividad crítica se retrasa una semana, el proyecto se retrasa una semana.", True, "critica_retraso"),
    ("Si una actividad no crítica se retrasa más que su holgura total, el proyecto se retrasa.", True, "holgura_total"),
    ("Acortar una actividad crítica siempre acorta el proyecto.", False, "varias_criticas"),
    ("La holgura total de una actividad nunca es menor que su holgura libre.", True, "orden_holguras"),
    ("Todo proyecto tiene al menos una ruta crítica.", True, "al_menos_una"),
    ("Un proyecto con dos rutas críticas tiene al menos dos actividades con holgura cero en cada ruta, distintas entre sí.", False, "criticas_comunes"),
    ("La suma de las holguras libres de las actividades de una ruta es como mucho el margen de esa ruta (τ menos su duración).", True, "suma_libres"),
    ("Un grafo PERT sin actividades ficticias representa un proyecto sin prelaciones compartidas.", False, "ficticias"),
    ("La holgura independiente de una actividad puede ser mayor que su holgura libre.", False, "orden_holguras"),
]


def _verdadero_falso(rng):
    """Afirmaciones verdadero/falso; las falsas se refutan con un contraejemplo calculado sobre un proyecto."""
    n = int(rng.choice([8, 9]))

    def acepta(i):
        cal = i["calendario"]
        return (1 <= i["n_dummies"] <= 4 and i["n_criticas"] >= 2
                and ((cal["H_libre"] == 0) & (cal["H_total"] > 0)).sum() >= 1
                and (cal["H_total"] > cal["H_libre"]).sum() >= 1)

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    idx = rng.choice(len(AFIRMACIONES), size=6, replace=False)
    elegidas = [AFIRMACIONES[int(i)] for i in idx]
    celdas = [md("# Enunciado"),
              md("Para cada afirmación diga si es verdadera o falsa. Si es falsa, dé un contraejemplo concreto "
                 "(puede apoyarse en el proyecto cuyo cuadro se adjunta, o en otro); si es verdadera, "
                 "justifíquela brevemente.")]
    celdas.append(preguntas([f"«{texto}»" for texto, _, _ in elegidas]))
    celdas.append(code('#| tbl-cap: "Proyecto de apoyo"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion = proyecto.duration(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "calendario"))
    for k, (texto, verdadera, clave) in enumerate(elegidas, 1):
        celdas += apartado(k)
        celdas.append(md(f"«{texto}» — **{'Verdadera' if verdadera else 'Falsa'}**."))
        if clave == "libre0_no_critica":
            celdas.append(code("ejemplo = calendario.index[(calendario['H_libre'] == 0) & (calendario['H_total'] > 0)].tolist()\n"
                               "Markdown(f'Contraejemplo: {\", \".join(ejemplo)} tiene(n) holgura libre 0 y holgura total {\", \".join(str(int(calendario.loc[a, \"H_total\"])) for a in ejemplo)} > 0: '\n"
                               "         'no es crítica, sólo ocurre que su sucesora empieza justo cuando ella termina (en el calendario de inicios tempranos).')"))
        elif clave == "critica_retraso":
            celdas.append(code("a = calendario.index[calendario['H_total'] == 0][0]\n"
                               "d2 = datos_proyecto['duration'].copy(); d2[a] += 1\n"
                               "Markdown(f'Justificación: una actividad crítica está en una ruta de duración τ; alargarla x alarga esa ruta a τ + x, y ninguna ruta puede ser más corta que la más larga. Comprobación con {a}: {duracion} → {proyecto.duration(d2)}.')"))
        elif clave == "holgura_total":
            celdas.append(code("a = calendario.index[calendario['H_total'] > 0][0]\n"
                               "h = int(calendario.loc[a, 'H_total'])\n"
                               "d2 = datos_proyecto['duration'].copy(); d2[a] += h + 1\n"
                               "Markdown(f'Justificación: la holgura total es exactamente el retraso que admite sin mover el fin del proyecto. Comprobación con {a} (holgura {h}): retrasada {h + 1} semanas el proyecto pasa de {duracion} a {proyecto.duration(d2)}.')"))
        elif clave == "varias_criticas":
            celdas.append(code("solo_en_una = [a for a in R.columns if 0 < R.loc[list(rutas_criticas), a].sum() < len(rutas_criticas)] if len(rutas_criticas) > 1 else []\n"
                               "if solo_en_una:\n"
                               "    a = solo_en_una[0]; d2 = datos_proyecto['duration'].copy(); d2[a] -= 1\n"
                               "    display(Markdown(f'Contraejemplo: el proyecto tiene {len(rutas_criticas)} rutas críticas y {a} sólo está en alguna de ellas; acortándola una semana la duración sigue siendo {proyecto.duration(d2)} (antes {duracion}), porque la otra ruta crítica no cambia.'))\n"
                               "else:\n"
                               "    display(Markdown('Contraejemplo: con dos rutas críticas paralelas (A→B y A→C, B y C de la misma duración), acortar B no acorta el proyecto porque C sigue durando lo mismo.'))"))
        elif clave == "orden_holguras":
            celdas.append(code("Markdown(f'Justificación: H_indep ≤ H_libre ≤ H_total siempre. La libre es la parte de la total que no afecta a las sucesoras (H_libre = mín ES suc − EF ≤ LS − ES = H_total, porque mín ES suc ≤ mín LS suc = LF); la independiente, además, supone las predecesoras acabando lo más tarde posible, lo que sólo puede reducir el margen. En el proyecto: ' + ', '.join(f\"{a}: {int(r.H_independiente)} ≤ {int(r.H_libre)} ≤ {int(r.H_total)}\" for a, r in calendario.iterrows()) + '.')"))
        elif clave == "al_menos_una":
            celdas.append(code("Markdown('Justificación: la duración del proyecto es el máximo de las duraciones de las rutas, y ese máximo lo alcanza al menos una ruta (el conjunto de rutas es finito y no vacío). Esa ruta tiene margen cero: es crítica.')"))
        elif clave == "criticas_comunes":
            celdas.append(code("comunes = [a for a in R.columns if R.loc[list(rutas_criticas), a].all()]\n"
                               "Markdown(f'Contraejemplo: dos rutas críticas pueden compartir actividades. En este proyecto las rutas críticas son {\", \".join(\" - \".join(v) for v in rutas_criticas.values())} y comparten {\", \".join(comunes) or \"(ninguna; pero basta imaginar un tramo común al inicio)\"}: no hace falta que cada ruta tenga dos actividades críticas propias.')"))
        elif clave == "suma_libres":
            celdas.append(code("filas = []\n"
                               "for r in R.index:\n"
                               "    acts = list(R.columns[R.loc[r] == 1])\n"
                               "    filas.append({'ruta': r, 'suma de holguras libres': int(calendario.loc[acts, 'H_libre'].sum()), 'margen de la ruta': int(duracion - (R.loc[r] * datos_proyecto['duration'][R.columns]).sum())})\n"
                               "display(pd.DataFrame(filas).set_index('ruta'))\n"
                               "Markdown('Justificación: las holguras libres de una ruta son huecos disjuntos entre el fin de una actividad y el inicio de la siguiente (o el fin del proyecto); su suma es como mucho el margen total de la ruta, y es menor si alguna actividad de la ruta empieza después del fin de su predecesora en la ruta por culpa de otra predecesora.')"))
        elif clave == "ficticias":
            celdas.append(code("Markdown(f'Contraejemplo: un proyecto en cadena A→B→C no tiene ficticias y sí prelaciones; y el proyecto en el que D tiene predecesoras A y B y E sólo B comparte prelaciones (B) y necesita ficticia. Las ficticias aparecen cuando dos actividades comparten *parte* de sus predecesoras o cuando dos actividades irían entre los mismos dos nodos; no por el mero hecho de compartir predecesoras. Este proyecto tiene {len(proyecto.dummies)} ficticia(s).')"))
    return celdas, f"6 afirmaciones; proyecto de apoyo con {n} actividades"


def _comparar_planificaciones(rng):
    """Dos programaciones propuestas: detectar violaciones de prelaciones y cuál respeta el plazo."""
    n = int(rng.choice([8, 9]))
    suc_de = {}

    def acepta(i):
        cal = i["calendario"]
        suc = P.sucesoras(i["preds"])
        suc_de.clear(); suc_de.update(suc)
        return (1 <= i["n_dummies"] <= 4 and (cal["H_total"] > 0).sum() >= 5
                and any(suc[a] for a in cal.index if cal.loc[a, "H_total"] > 0))

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, preds, T = info["D"], info["calendario"], info["preds"], int(info["T"])
    df = P.tabla_base(info, duration=D)
    ES = cal["inicio_mas_temprano"].astype(int)
    LS = cal["inicio_mas_tardio"].astype(int)
    # Plan 1: inicios tardíos con dos violaciones (dos actividades adelantadas antes del fin de una predecesora)
    plan1 = LS.copy()
    con_pred = [a for a in D.index if preds[a]]
    v = list(rng.choice(con_pred, size=2, replace=False))
    for a in v:
        p = sorted(preds[a])[0]
        plan1[a] = int(plan1[p]) + int(D[p]) - 1          # empieza una semana antes de que acabe p
    # Plan 2: inicios tempranos con una actividad holgada retrasada dentro de su holgura y otra (con
    # sucesoras) retrasada una semana más de lo que su holgura permite
    plan2 = ES.copy()
    holgadas = [a for a in D.index if cal.loc[a, "H_total"] > 0]
    con_suc = [a for a in holgadas if suc_de[a]]
    fuera = con_suc[int(rng.integers(0, len(con_suc)))]
    otras = [a for a in holgadas if a != fuera]
    dentro = otras[int(rng.integers(0, len(otras)))]
    plan2[dentro] = int(ES[dentro]) + int(cal.loc[dentro, "H_total"])
    plan2[fuera] = int(ES[fuera]) + int(cal.loc[fuera, "H_total"]) + 1
    planes = pd.DataFrame({"start_plan_1": plan1, "start_plan_2": plan2})
    planes.index.name = "activity"
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (semanas) se adjunta. Dos "
                 "jefes de obra han propuesto sendas programaciones, dadas por la semana de inicio de cada "
                 "actividad (segundo cuadro). Las actividades no se interrumpen.")]
    celdas.append(preguntas([
        "Para cada plan, compruebe actividad por actividad si respeta las prelaciones: una actividad no "
        "puede empezar antes de que terminen todas sus predecesoras. Liste las violaciones.",
        "Calcule la duración del proyecto que resulta de cada plan (fin de la última actividad) y compárela "
        "con la duración mínima del proyecto.",
        "Corrija cada plan con el mínimo cambio: adelante o retrase sólo las actividades necesarias (y sus "
        "sucesoras si hace falta) y recalcule la duración.",
        "¿Qué plan es «más tardío» (deja menos margen) y qué riesgo supone? ¿Qué plan deja más holgura para "
        "absorber imprevistos?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(code('#| tbl-cap: "Programaciones propuestas (semana de inicio)"\n#| label: tbl-planes\n'
                       + literal_dataframe(planes, nombre="planes")))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_minima = proyecto.duration(datos_proyecto['duration'])\n"
                       "pred = prelaciones.immediate_predecessor"))
    celdas += apartado(1)
    celdas.append(code("def violaciones(inicio):\n"
                       "    fin = inicio + datos_proyecto['duration']\n"
                       "    filas = []\n"
                       "    for a in inicio.index:\n"
                       "        for p in sorted(pred[a]):\n"
                       "            if fin[p] > inicio[a]:\n"
                       "                filas.append({'actividad': a, 'empieza en': int(inicio[a]), 'predecesora': p, 'termina en': int(fin[p]), 'conflicto': int(fin[p] - inicio[a])})\n"
                       "    return pd.DataFrame(filas)\n"
                       "for plan in ('start_plan_1', 'start_plan_2'):\n"
                       "    v = violaciones(planes[plan])\n"
                       "    display(Markdown(f'**{plan}**: {len(v)} violación(es)'))\n"
                       "    if len(v): display(v.set_index('actividad'))"))
    celdas += apartado(2)
    celdas.append(code("fines = planes.add(datos_proyecto['duration'], axis='rows')\n"
                       "duraciones = fines.max().rename('fin del proyecto')\n"
                       "display(duraciones.to_frame().T)\n"
                       "Markdown(f'Duración mínima del proyecto: {duracion_minima}. Un plan con violaciones no es ejecutable tal cual: su duración aparente es engañosa.')"))
    celdas += apartado(3)
    celdas.append(md("Corrección mínima: se recorren las actividades en orden topológico y cada una empieza en el "
                     "máximo entre la fecha propuesta y el fin de sus predecesoras (ya corregidas). Así sólo se "
                     "mueven las actividades que lo necesitan y sus sucesoras."))
    celdas.append(code("G = nx.DiGraph(); G.add_nodes_from(datos_proyecto.index); G.add_edges_from((p, a) for a, ps in pred.items() for p in ps)\n"
                       "orden = list(nx.topological_sort(G))\n"
                       "def corregir(inicio):\n"
                       "    nuevo = inicio.copy()\n"
                       "    for a in orden:\n"
                       "        fin_pred = max([nuevo[p] + datos_proyecto.loc[p, 'duration'] for p in pred[a]], default=0)\n"
                       "        nuevo[a] = max(int(inicio[a]), int(fin_pred))\n"
                       "    return nuevo\n"
                       "corregidos = pd.DataFrame({plan: corregir(planes[plan]) for plan in planes.columns})\n"
                       "cambios = (corregidos != planes)\n"
                       "display(pd.concat({'propuesto': planes, 'corregido': corregidos}, axis=1))\n"
                       "fines_c = corregidos.add(datos_proyecto['duration'], axis='rows').max()\n"
                       "Markdown('Actividades movidas: ' + '; '.join(f'{plan}: {\", \".join(cambios.index[cambios[plan]]) or \"ninguna\"}' for plan in planes.columns) + '. '\n"
                       "         + 'Duración tras corregir: ' + ', '.join(f'{plan}: {int(v)}' for plan, v in fines_c.items()) + '.')"))
    celdas += apartado(4)
    celdas.append(md("El margen de cada actividad en un plan es lo que puede retrasarse sin mover el fin *de ese "
                     "plan*: su inicio más tardío (recalculado para la duración del plan corregido) menos el "
                     "inicio programado."))
    celdas.append(code("margen = pd.DataFrame({plan: (calendario['inicio_mas_tardio'] + (fines_c[plan] - duracion_minima) - corregidos[plan]) for plan in corregidos.columns})\n"
                       "display(margen.rename(columns=lambda c: f'margen restante ({c})').T)\n"
                       "sin_margen = (margen <= 0).sum()\n"
                       "mas_tardio = sin_margen.idxmax()\n"
                       "Markdown(f'Actividades sin margen (empiezan en su inicio más tardío): ' + ', '.join(f'{p}: {int(v)}' for p, v in sin_margen.items()) + f'. '\n"
                       "         f'El plan más tardío es {mas_tardio}: en él casi cualquier retraso se traslada al fin del proyecto, porque las actividades han consumido su holgura antes de empezar. '\n"
                       "         'El otro plan conserva holgura para absorber imprevistos, a cambio de necesitar antes los recursos y de inmovilizar antes el dinero.')"))
    return celdas, f"{n} actividades; dos planes con violaciones ({', '.join(v)} y {fuera})"


def _tipo_test(rng):
    """Cuestionario de respuesta múltiple generado sobre un proyecto, con exportación a GIFT (Moodle)."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 9 and i["n_criticas"] == 1
                    and (i["calendario"]["H_total"] > 0).sum() >= 3,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, T = info["D"], info["calendario"], int(info["T"])
    sd = P.desviaciones(rng, list(D.index))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    holgadas = cal.index[cal["H_total"] > 0].tolist()
    X = holgadas[int(rng.integers(0, len(holgadas)))]
    plazo = T + 2
    celdas = [md("# Enunciado"),
              md("Sobre el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas "
                 "(semanas) se adjunta, responda al cuestionario. Cada pregunta tiene una sola respuesta "
                 "correcta. Al final se pide generar el cuestionario en formato GIFT para importarlo en "
                 "Moodle.")]
    celdas.append(preguntas([
        "¿Cuál es la duración del proyecto?",
        f"¿Cuál es la holgura total de {X}?",
        "¿Cuántas rutas tiene el grafo (del nodo inicial al final, sin contar ficticias como actividades)?",
        "¿Cuántas actividades ficticias necesita el grafo PERT con el mínimo número de ellas?",
        f"¿Cuál es, aproximadamente, la probabilidad de terminar en {plazo} semanas o menos (aproximación "
        "normal por el camino crítico)?",
        "Genere el fichero GIFT con las cinco preguntas, sus opciones y la retroalimentación."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion = int(proyecto.duration(datos_proyecto['duration']))\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       f"X, plazo = '{X}', {plazo}\n"
                       "p_plazo = float(norm.cdf((plazo - duracion) / sigma))"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas.append(md("Cada pregunta se construye con la respuesta correcta calculada y tres distractores "
                     "*plausibles*: errores típicos (sumar todas las duraciones, confundir holgura libre y "
                     "total, contar las ficticias como rutas, usar la varianza en lugar de la desviación…)."))
    celdas.append(code("h_total, h_libre = int(calendario.loc[X, 'H_total']), int(calendario.loc[X, 'H_libre'])\n"
                       "n_rutas, n_fict = len(R), len(proyecto.dummies)\n"
                       "p_mal = float(norm.cdf((plazo - duracion) / sigma ** 2))\n"
                       "def opciones(correcta, distractores):\n"
                       "    vistos, lista = set(), []\n"
                       "    for v in [correcta] + distractores:\n"
                       "        if v not in vistos:\n"
                       "            vistos.add(v); lista.append(v)\n"
                       "    return lista[:4]\n"
                       "preguntas_test = [\n"
                       "    ('¿Cuál es la duración del proyecto (semanas)?', opciones(duracion, [int(datos_proyecto['duration'].sum()), duracion - 1, duracion + 2, duracion + 1, duracion - 2]),\n"
                       "     'La duración es la de la ruta más larga, no la suma de todas las actividades.'),\n"
                       "    (f'¿Cuál es la holgura total de {X} (semanas)?', opciones(h_total, [h_libre, 0, h_total + 2, h_total + 1, h_total + 3]),\n"
                       "     'H_total = inicio más tardío − inicio más temprano; la holgura libre puede ser menor.'),\n"
                       "    ('¿Cuántas rutas tiene el grafo del nodo inicial al final?', opciones(n_rutas, [n_rutas + n_fict, n_rutas - 1, n_rutas + 2, n_rutas + 1, n_rutas + 3]),\n"
                       "     'Las ficticias no añaden rutas distintas en términos de actividades.'),\n"
                       "    ('¿Cuántas actividades ficticias necesita el PERT con el mínimo número de ellas?', opciones(n_fict, [0, n_fict + 1, n_fict + 2, n_fict + 3]),\n"
                       "     'Hacen falta ficticias cuando dos actividades comparten parte de sus predecesoras o irían entre los mismos nodos.'),\n"
                       f"    (f'¿Probabilidad aproximada de terminar en {{plazo}} semanas o menos?', opciones(round(p_plazo, 2), [round(p_mal, 2), 0.5, round(1 - p_plazo, 2), 0.95, round(p_plazo / 2, 2)]),\n"
                       "     'z = (plazo − duración) / σ, con σ la desviación típica de la ruta crítica (raíz de la suma de varianzas).'),\n"
                       "]\n"
                       "for i, (q, ops, fb) in enumerate(preguntas_test, 1):\n"
                       "    print(f'{i}. {q}  →  correcta: {ops[0]}   (opciones: {ops})')"))
    celdas += apartado(6)
    celdas.append(md("El formato GIFT de Moodle escribe cada pregunta como `::título::texto {=correcta ~incorrecta "
                     "~incorrecta #retroalimentación}`. Las opciones se barajan al generar el fichero."))
    celdas.append(code("import os\n"
                       "def a_gift(preguntas_test, titulo, semilla=1):\n"
                       "    r = np.random.default_rng(semilla)\n"
                       "    lineas = [f'$CATEGORY: {titulo}', '']\n"
                       "    for i, (q, ops, fb) in enumerate(preguntas_test, 1):\n"
                       "        correcta, resto = ops[0], ops[1:]\n"
                       "        orden = list(r.permutation(len(ops)))\n"
                       "        todas = [ops[j] for j in orden]\n"
                       "        cuerpo = ' '.join((f'={v}' if v == correcta else f'~{v}') for v in todas)\n"
                       "        lineas.append(f'::{titulo} P{i}::{q} {{{cuerpo} ####{fb}}}')\n"
                       "        lineas.append('')\n"
                       "    return '\\n'.join(lineas)\n"
                       "gift = a_gift(preguntas_test, 'Proyecto test')\n"
                       "os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)\n"
                       "with open(os.path.expanduser('~/temp/cuestionario.gift'), 'w', encoding='utf8') as f:\n"
                       "    f.write(gift)\n"
                       "from pmule import muestra_test\n"
                "muestra_test(preguntas_test)"))
    return celdas, f"{n} actividades; 5 preguntas de respuesta múltiple con exportación GIFT"


VARIANTES = [
    Variante("disena-una-red", 2, "diseñar un proyecto con rasgos dados (iniciales, ficticias, rutas críticas) y verificarlos", _disena),
    Variante("datos-insuficientes", 2, "dos duraciones en intervalos: cotas de τ, actividades siempre/nunca críticas y qué dato averiguar primero",
             _datos_insuficientes),
    Variante("verdadero-falso", 2, "seis afirmaciones sobre holguras, rutas críticas y ficticias; contraejemplos calculados", _verdadero_falso),
    Variante("comparar-planificaciones", 1, "dos programaciones propuestas: violaciones de prelaciones, corrección mínima y margen restante",
             _comparar_planificaciones),
    Variante("tipo-test", 1, "cuestionario de cinco preguntas con distractores razonados y exportación a GIFT (Moodle)", _tipo_test),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
