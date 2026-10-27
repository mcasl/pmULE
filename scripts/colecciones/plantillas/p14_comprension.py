"""Colección 14 — Comprensión conceptual: diseñar una red con rasgos dados, razonar con datos
incompletos, verdadero/falso con contraejemplo, detectar programaciones inválidas y preguntas
tipo test con exportación a Moodle (GIFT)."""
import networkx as nx
import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

def _pl(n, singular, plural=None):
    """«1 semana», «3 semanas»: número y sustantivo concordados (para enunciados y subtítulos)."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


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
                 f"- su grafo PERT con el mínimo número de ficticias tiene exactamente {_pl(n_dum, 'ficticia')};\n"
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
    celdas.append(code(r'''
        display(explica_camino_critico(proyecto, datos_proyecto['duration'], 'días'))
        display(Markdown('Duración de cada ruta del grafo (suma de las duraciones de sus actividades):'))
        display(explica_rutas(proyecto, datos_proyecto['duration'], 'días'))
        calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']]'''))
    celdas.append(code(r'''
        _holgadas = calendario.index[calendario['H_total'] > 0].tolist()
        _ini = sorted(prelaciones.start_activities)
        Markdown(f'**Respuesta:** el proyecto del apartado 1 cumple todas las condiciones: '
                 f'{plural(len(datos_proyecto), "actividad", "actividades")}; '
                 f'{plural(len(_ini), "actividad inicial", "actividades iniciales")} ({lista_y(_ini)}); '
                 f'{plural(len(proyecto.dummies), "ficticia")} en el grafo PERT; '
                 f'{plural(len(rutas_criticas), "ruta crítica", "rutas críticas")} (las de duración {proyecto.duration(datos_proyecto["duration"])} días de la tabla de rutas); '
                 'holgura total positiva en ' + lista_y([f'{a} ({int(calendario.loc[a, "H_total"])})' for a in _holgadas]) + '; '
                 'y ninguna prelación redundante (ninguna predecesora inmediata lo es también de forma indirecta, a través de otra).')'''))
    celdas += apartado(3)
    celdas.append(code(r'''
        R = proyecto.path_matrix(dummies=False)
        Rt = R @ datos_proyecto['duration'][R.columns]
        criticas = Rt.index[Rt == Rt.max()].tolist()
        _nombre = lambda r: ' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario))
        # actividad que está en una sola ruta crítica: alargarla un día deja esa ruta como única crítica
        en_una = [a for a in R.columns if R.loc[criticas, a].sum() == 1]
        a_cambiar = en_una[0] if en_una else None
        display(Markdown('La condición más exigente suele ser el número exacto de rutas críticas: hay que conseguir que '
                         f'{len(criticas)} rutas sumen exactamente lo mismo y que ninguna otra las supere; '
                         'un cambio de una unidad en una duración basta para romper el empate.'))
        if a_cambiar:
            d2 = datos_proyecto['duration'].copy(); d2[a_cambiar] += 1
            Rt2 = R @ d2[R.columns]
            n_crit2 = int((Rt2 == Rt2.max()).sum())
            display(pd.DataFrame({'ruta crítica': [_nombre(r) for r in criticas],
                                  f'contiene {a_cambiar}': ['sí' if R.loc[r, a_cambiar] else 'no' for r in criticas],
                                  'duración actual': [int(Rt[r]) for r in criticas],
                                  f'duración con {a_cambiar} = {int(d2[a_cambiar])}': [int(Rt2[r]) for r in criticas]}).set_index('ruta crítica'))
            _sin = [_nombre(r) for r in criticas if not R.loc[r, a_cambiar]]
            display(Markdown(f'**Respuesta:** basta alargar {a_cambiar} un día (de {int(datos_proyecto.loc[a_cambiar, "duration"])} a {int(d2[a_cambiar])}). '
                             f'{a_cambiar} sólo está en la ruta crítica {_nombre([r for r in criticas if R.loc[r, a_cambiar]][0])}, que pasa a durar {int(Rt2.max())} días, '
                             f'mientras {lista_y(_sin)} se {"queda" if len(_sin) == 1 else "quedan"} en {int(Rt.max())}: '
                             f'queda {plural(n_crit2, "ruta crítica", "rutas críticas")}.'))
        else:
            display(Markdown('**Respuesta:** ninguna actividad está en una sola de las rutas críticas; habría que alargar una '
                             'actividad de una de ellas y acortar a la vez otra común a las demás.'))'''))
    celdas += apartado(4)
    celdas.append(md("No es único: en general hay muchísimos proyectos que cumplen las condiciones. Por "
                     "ejemplo, multiplicar todas las duraciones por 2 multiplica por 2 la duración de todas las "
                     "rutas, así que las mismas rutas siguen empatadas como críticas y las mismas actividades "
                     "conservan holgura positiva (doblada); las prelaciones, y con ellas las iniciales y las "
                     "ficticias, no cambian. También vale permutar los nombres de las actividades. Una solución "
                     "*estructuralmente* distinta se obtiene cambiando una prelación y recomprobando."))
    celdas.append(code(r'''
        datos_otro = datos_proyecto.copy()
        datos_otro['duration'] = datos_otro['duration'] * 2
        p2 = PredecessorTable.from_dataframe_of_strings(datos_otro, activity='activity', predecessor='predecessor').create_project()
        _n_crit, _n_fic = len(p2.critical_path(datos_otro['duration'])), len(p2.dummies)
        _n_hol = int((p2.calendar(datos_otro['duration'])['H_total'] > 0).sum())
        Markdown('**Respuesta:** no es único. Otro proyecto válido: las mismas prelaciones con las duraciones '
                 + ', '.join(f'{a} = {int(d)}' for a, d in datos_otro['duration'].items())
                 + f' (duración {p2.duration(datos_otro["duration"])} días). Comprobación: {plural(_n_crit, "ruta crítica", "rutas críticas")}, '
                 f'{plural(_n_fic, "ficticia")} y {plural(_n_hol, "actividad", "actividades")} con holgura positiva: cumple las mismas condiciones.')'''))
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
    celdas.append(md("La duración del proyecto es la de la ruta más larga: τ = ‖R·t‖∞, el máximo de las "
                     "duraciones de las rutas (R·t, una fila de la matriz de rutas R por ruta). Alargar una "
                     "actividad nunca acorta ninguna ruta, así que τ crece con cada duración: la cota inferior se "
                     "alcanza con las dos desconocidas en su mínimo y la superior con las dos en su máximo, sin "
                     "necesidad de probar otras combinaciones. La tabla da la duración de cada ruta en esos dos "
                     "casos."))
    celdas.append(code(r'''
        Rt_min = R @ minimos[R.columns]; Rt_max = R @ maximos[R.columns]
        cota_inf, cota_sup = int(Rt_min.max()), int(Rt_max.max())
        _acts = {r: list(R.columns[R.loc[r] == 1]) for r in R.index}
        _suma = lambda r, t: ' + '.join(str(int(t[a])) for a in _acts[r]) + f' = {int(sum(t[a] for a in _acts[r]))}'
        display(pd.DataFrame({'actividades': {r: ' – '.join(_acts[r]) for r in R.index},
                              f'duración con {X} = {minimos[X]}, {Y} = {minimos[Y]}': {r: _suma(r, minimos) for r in R.index},
                              f'duración con {X} = {maximos[X]}, {Y} = {maximos[Y]}': {r: _suma(r, maximos) for r in R.index}}))
        _r_inf = [' – '.join(_acts[r]) for r in R.index if Rt_min[r] == cota_inf]
        _r_sup = [' – '.join(_acts[r]) for r in R.index if Rt_max[r] == cota_sup]
        Markdown(f'**Respuesta:** {cota_inf} ≤ τ ≤ {cota_sup} semanas. La cota inferior, {cota_inf}, se alcanza con {X} = {minimos[X]} y {Y} = {minimos[Y]} '
                 f'(la ruta más larga es entonces {lista_y(_r_inf, "o")}); la superior, {cota_sup}, con {X} = {maximos[X]} y {Y} = {maximos[Y]} '
                 f'(ruta {lista_y(_r_sup, "o")}). Ningún valor intermedio puede dar menos de {cota_inf} ni más de {cota_sup}.')'''))
    celdas += apartado(2)
    celdas.append(md("Una actividad es crítica para un juego de duraciones si está en alguna ruta de duración "
                     "máxima. Se prueban todas las combinaciones enteras de las dos duraciones desconocidas; la "
                     "tabla muestra las cuatro combinaciones extremas como ejemplo:"))
    celdas.append(code(r'''
        from itertools import product
        def _criticas(dX, dY):
            t = minimos.copy(); t[X], t[Y] = dX, dY
            Rt = R @ t[R.columns]
            return int(Rt.max()), set(R.columns[(R.loc[Rt == Rt.max()].sum() > 0)])
        display(pd.DataFrame([{X: dX, Y: dY, 'τ': _criticas(dX, dY)[0], 'actividades críticas': ', '.join(sorted(_criticas(dX, dY)[1]))}
                              for dX in (minimos[X], maximos[X]) for dY in (minimos[Y], maximos[Y])]).set_index([X, Y]))
        siempre, nunca = set(R.columns), set(R.columns)
        _n = 0
        for dX, dY in product(range(minimos[X], maximos[X] + 1), range(minimos[Y], maximos[Y] + 1)):
            criticas_ahora = _criticas(dX, dY)[1]; _n += 1
            siempre &= criticas_ahora; nunca -= criticas_ahora
        _depende = sorted(set(R.columns) - siempre - nunca)
        Markdown(f'Recorriendo las {_n} combinaciones posibles: **críticas para cualquier valor:** {lista_y(sorted(siempre))}; '
                 f'**nunca críticas:** {lista_y(sorted(nunca))}' + (f'; las demás ({lista_y(_depende)}) lo son o no según los valores.' if _depende else '.'))'''))
    celdas += apartado(3)
    celdas.append(md(f"La holgura total de {Y} es lo que le falta a la ruta más larga que pasa por {Y} para "
                     f"igualar la duración del proyecto: H_total({Y}) = τ − (duración de la ruta más larga que "
                     f"contiene {Y}). La tabla da esa holgura para cada duración posible de {Y}, con {X} en su "
                     "mínimo y en su máximo (las demás actividades con su duración conocida)."))
    celdas.append(code(r'''
        def holgura_Y(dX, dY):
            t = minimos.copy(); t[X], t[Y] = dX, dY
            return int(proyecto.calendar(t).loc[Y, 'H_total'])
        tabla = pd.DataFrame({f'{X} = {dX}': {dY: holgura_Y(dX, dY) for dY in range(minimos[Y], maximos[Y] + 1)} for dX in (minimos[X], maximos[X])})
        tabla.index.name = f'duración de {Y}'
        display(tabla.T.rename_axis(f'holgura total de {Y}'))
        umbrales = {col: next((dY for dY, h in tabla[col].items() if h == 0), None) for col in tabla.columns}
        _u = list(umbrales.values())
        _txt = '; '.join(f'con {col}, a partir de {Y} = {u}' if u is not None else f'con {col}, nunca (su holgura mínima es {int(tabla[col].min())})'
                         for col, u in umbrales.items())
        _misma_ruta = bool(((R[X] == 1) & (R[Y] == 1)).any())
        if _u[0] == _u[1]:
            _por = (f'El umbral no depende de {X}: {X} y {Y} están en una misma ruta, de modo que alargar {X} alarga a la vez la ruta de {Y} y la duración del proyecto.'
                    if _misma_ruta and X in siempre else f'El umbral no depende de la duración de {X}.')
        elif _u[1] is None or (_u[0] is not None and _u[1] > _u[0]):
            _por = f'Cuanto más dura {X}, más larga es la ruta crítica sin {Y} y más holgura le queda a {Y}: se vuelve crítica más tarde.'
        else:
            _por = f'Cuanto más dura {X}, antes se vuelve crítica {Y}.'
        Markdown(f'**Respuesta:** {Y} pasa a tener holgura total 0 (se vuelve crítica) {_txt}. {_por}')'''))
    celdas += apartado(4)
    celdas.append(md("Se compara el intervalo [cota inferior, cota superior] de τ que queda al conocer cada dato: "
                     "fijada una de las dos duraciones en cada uno de sus valores posibles, la otra sigue variando "
                     "entre su mínimo y su máximo."))
    celdas.append(code(r'''
        filas = []
        for fija, otra in ((X, Y), (Y, X)):
            for v in range(minimos[fija], maximos[fija] + 1):
                lo, hi = minimos.copy(), maximos.copy(); lo[fija] = hi[fija] = v
                a, b = int((R @ lo[R.columns]).max()), int((R @ hi[R.columns]).max())
                filas.append({'dato conocido': f'{fija} = {v}', 'actividad conocida': fija, 'cota inferior': a, 'cota superior': b, 'anchura': b - a})
        intervalos = pd.DataFrame(filas).set_index('dato conocido')
        display(intervalos.drop(columns='actividad conocida'))
        amp = intervalos.groupby('actividad conocida')['anchura'].agg(['mean', 'max'])
        mejor = X if amp.loc[X, 'mean'] < amp.loc[Y, 'mean'] else Y if amp.loc[Y, 'mean'] < amp.loc[X, 'mean'] else None
        _txt = (f'Sin conocer ninguna, τ está entre {cota_inf} y {cota_sup} (anchura {cota_sup - cota_inf} semanas). '
                f'Conociendo {X}, la anchura media es {amp.loc[X, "mean"]:.1f} semanas (como mucho {int(amp.loc[X, "max"])}); '
                f'conociendo {Y}, {amp.loc[Y, "mean"]:.1f} semanas (como mucho {int(amp.loc[Y, "max"])}). ')
        if mejor:
            otra = Y if mejor == X else X
            display(Markdown(_txt + f'**Respuesta:** conviene averiguar primero la duración de {mejor}: es la que más estrecha el intervalo de la duración del proyecto. '
                     f'Fijado {mejor}, queda la incertidumbre de {otra} (rango de {maximos[otra] - minimos[otra]} semanas), que deja una anchura media de {amp.loc[mejor, "mean"]:.1f}; '
                     f'fijado {otra}, la de {mejor} (rango de {maximos[mejor] - minimos[mejor]} semanas) deja {amp.loc[otra, "mean"]:.1f}. '
                     'Una actividad sólo traslada a τ la parte de su rango en la que está en la ruta más larga: mientras tiene holgura, alargarla no alarga el proyecto.'))
        else:
            display(Markdown(_txt + '**Respuesta:** los dos datos reducen la incertidumbre por igual.'))'''))
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
                       "pred = prelaciones.immediate_predecessor\n"
                       "calendario"))
    for k, (texto, verdadera, clave) in enumerate(elegidas, 1):
        celdas += apartado(k)
        celdas.append(md(f"«{texto}» — **{'Verdadera' if verdadera else 'Falsa'}**."))
        if clave == "libre0_no_critica":
            celdas.append(code(r"""
                ejemplo = calendario.index[(calendario['H_libre'] == 0) & (calendario['H_total'] > 0)].tolist()
                a = ejemplo[0]
                _suc = sorted(s for s, ps in pred.items() if a in ps)
                _es = {s: int(calendario.loc[s, 'inicio_mas_temprano']) for s in _suc}
                _r = calendario.loc[a]
                Markdown(f'Contraejemplo: {a}. Su holgura libre es H_libre = mín(ES de sus sucesoras) − EF = mín(' + ', '.join(f'{s}: {v}' for s, v in _es.items())
                         + f') − {int(_r.fin_mas_temprano)} = 0, pero su holgura total es H_total = LS − ES = {int(_r.inicio_mas_tardio)} − {int(_r.inicio_mas_temprano)} = {int(_r.H_total)} > 0: '
                         f'no es crítica (ES/EF: inicio/fin más temprano; LS: inicio más tardío). Holgura libre 0 sólo significa que, en el calendario de inicios tempranos, una sucesora empieza justo cuando ella termina.'
                         + (f' Lo mismo ocurre con {lista_y(ejemplo[1:])}.' if len(ejemplo) > 1 else ''))"""))
        elif clave == "critica_retraso":
            celdas.append(code("a = calendario.index[calendario['H_total'] == 0][0]\n"
                               "d2 = datos_proyecto['duration'].copy(); d2[a] += 1\n"
                               "Markdown(f'Justificación: una actividad crítica está en una ruta de duración τ (la duración del proyecto); alargarla x semanas alarga esa ruta a τ + x, y la duración del proyecto es la de la ruta más larga. Comprobación con {a}: la duración pasa de {duracion} a {proyecto.duration(d2)} semanas.')"))
        elif clave == "holgura_total":
            celdas.append(code("a = calendario.index[calendario['H_total'] > 0][0]\n"
                               "h = int(calendario.loc[a, 'H_total'])\n"
                               "d2 = datos_proyecto['duration'].copy(); d2[a] += h + 1\n"
                               "Markdown(f'Justificación: la holgura total es exactamente el retraso que admite una actividad sin mover el fin del proyecto; un retraso mayor alarga el proyecto en lo que exceda de la holgura. Comprobación con {a} (holgura total {h}): retrasada {h + 1} semanas, el proyecto pasa de {duracion} a {proyecto.duration(d2)} semanas.')"))
        elif clave == "varias_criticas":
            celdas.append(code(r"""
                solo_en_una = [a for a in R.columns if 0 < R.loc[list(rutas_criticas), a].sum() < len(rutas_criticas)] if len(rutas_criticas) > 1 else []
                if solo_en_una:
                    a = solo_en_una[0]; d2 = datos_proyecto['duration'].copy(); d2[a] -= 1
                    _sin = [' – '.join(ordena_ruta(v, calendario)) for v in rutas_criticas.values() if a not in v]
                    display(Markdown(f'Contraejemplo: el proyecto tiene {len(rutas_criticas)} rutas críticas y {a} no está en todas ellas; acortándola una semana la duración sigue siendo {proyecto.duration(d2)} semanas (antes {duracion}), porque ' + (f'la ruta crítica {_sin[0]} no contiene {a} y sigue' if len(_sin) == 1 else f'las rutas críticas {lista_y(_sin)} no contienen {a} y siguen') + f' durando {duracion}.'))
                else:
                    display(Markdown('Contraejemplo: con dos rutas críticas paralelas (A→B y A→C, B y C de la misma duración), acortar B no acorta el proyecto porque la ruta A–C sigue durando lo mismo.'))"""))
        elif clave == "orden_holguras":
            celdas.append(code("Markdown(f'Justificación: H_indep ≤ H_libre ≤ H_total siempre. La libre es la parte de la total que no afecta a las sucesoras (H_libre = mín ES suc − EF ≤ LF − EF = H_total, porque mín ES suc ≤ mín LS suc = LF; ES/EF: inicio/fin más temprano, LS/LF: inicio/fin más tardío); la independiente, además, supone las predecesoras acabando lo más tarde posible, lo que sólo puede reducir el margen. En el proyecto (H_indep ≤ H_libre ≤ H_total): ' + ', '.join(f\"{a}: {int(r.H_independiente)} ≤ {int(r.H_libre)} ≤ {int(r.H_total)}\" for a, r in calendario.iterrows()) + '.')"))
        elif clave == "al_menos_una":
            celdas.append(code("Markdown(f'Justificación: la duración del proyecto es el máximo de las duraciones de las rutas, y ese máximo lo alcanza al menos una ruta (el conjunto de rutas es finito y no vacío). Esa ruta tiene margen cero: es crítica. En este proyecto la duración es {duracion} semanas y la alcanza{\"n\" if len(rutas_criticas) > 1 else \"\"} ' + lista_y([\" – \".join(ordena_ruta(v, calendario)) for v in rutas_criticas.values()]) + '.')"))
        elif clave == "criticas_comunes":
            celdas.append(code(r"""
                comunes = [a for a in R.columns if R.loc[list(rutas_criticas), a].all()]
                _rutas = [' – '.join(ordena_ruta(v, calendario)) for v in rutas_criticas.values()]
                if comunes:
                    display(Markdown(f'Contraejemplo: dos rutas críticas pueden compartir actividades. En este proyecto las rutas críticas son {lista_y(_rutas)} y comparten {lista_y(comunes)}: '
                             'las actividades con holgura cero no tienen por qué ser distintas en cada ruta.'))
                else:
                    display(Markdown(f'Contraejemplo: dos rutas críticas pueden compartir actividades (en este proyecto, {lista_y(_rutas)}, no comparten ninguna, pero basta un proyecto '
                             'con A→B y A→C, B y C de igual duración: las rutas críticas A–B y A–C comparten A, y cada una tiene una sola actividad propia).'))"""))
        elif clave == "suma_libres":
            celdas.append(code(r"""
                filas = []
                for r in R.index:
                    acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario)
                    dur_r = int(sum(datos_proyecto.loc[a, 'duration'] for a in acts))
                    filas.append({'ruta': ' – '.join(acts),
                                  'suma de holguras libres': ' + '.join(str(int(calendario.loc[a, 'H_libre'])) for a in acts) + f" = {int(calendario.loc[acts, 'H_libre'].sum())}",
                                  'margen de la ruta = τ − duración': f'{duracion} − {dur_r} = {duracion - dur_r}'})
                display(pd.DataFrame(filas).set_index('ruta'))
                Markdown('Justificación: las holguras libres de una ruta son huecos disjuntos entre el fin de una actividad y el inicio de la siguiente (o el fin del proyecto); su suma es como mucho el margen total de la ruta, y es menor si alguna actividad de la ruta empieza después del fin de su predecesora en la ruta por culpa de otra predecesora. La tabla lo comprueba en todas las rutas del proyecto.')"""))
        elif clave == "ficticias":
            celdas.append(code(r"""
                _suc = {a: frozenset(s for s, ps in pred.items() if a in ps) for a in pred}
                _pares = [(a, b) for a, b in _combinations(sorted(pred), 2) if pred[a] and set(pred[a]) == set(pred[b]) and _suc[a] != _suc[b]]
                _ej = (f' En este proyecto ocurre lo mismo con {_pares[0][0]} y {_pares[0][1]}: tienen las mismas predecesoras ({lista_y(sorted(pred[_pares[0][0]]))}), salen del mismo nodo y esa prelación compartida no requiere ninguna ficticia.'
                       if _pares else '')
                Markdown('Contraejemplo: en el proyecto A; C y D precedidas por A; E precedida por C, el grafo PERT (A: 1→2, C: 2→3, D: 2→4, E: 3→4) no tiene ficticias '
                         'y, sin embargo, C y D comparten la prelación de A (salen las dos del nodo final de A).' + _ej + ' Las ficticias sólo aparecen cuando dos actividades comparten *parte* de sus predecesoras (no todas) o cuando dos actividades irían entre los mismos dos nodos; '
                         f'no por el mero hecho de compartir predecesoras. Este proyecto necesita {plural(len(proyecto.dummies), "ficticia")}.')"""))
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
                       "pred = prelaciones.immediate_predecessor\n"
                       "NOMBRE = {'start_plan_1': 'Plan 1', 'start_plan_2': 'Plan 2'}"))
    celdas += apartado(1)
    celdas.append(code(r"""
        def violaciones(inicio):
            fin = inicio + datos_proyecto['duration']
            filas = []
            for a in inicio.index:
                for p in sorted(pred[a]):
                    if fin[p] > inicio[a]:
                        filas.append({'actividad': a, 'empieza en': int(inicio[a]), 'predecesora': p,
                                      'la predecesora termina en': f"{int(inicio[p])} + {int(datos_proyecto.loc[p, 'duration'])} = {int(fin[p])}",
                                      'semanas de solape': int(fin[p] - inicio[a])})
            return pd.DataFrame(filas)
        _resp = []
        for plan in ('start_plan_1', 'start_plan_2'):
            v = violaciones(planes[plan])
            display(Markdown(f'**{NOMBRE[plan]}**: {plural(len(v), "violación", "violaciones")}'))
            if len(v):
                display(v.set_index('actividad'))
            _resp.append(f'{NOMBRE[plan]}: ' + (', '.join([f"{r['actividad']} empieza en {r['empieza en']} y su predecesora {r['predecesora']} no termina hasta {int(planes.loc[r['predecesora'], plan] + datos_proyecto.loc[r['predecesora'], 'duration'])}" for _, r in v.iterrows()])
                                                if len(v) else 'ninguna violación'))
        Markdown('Para cada actividad se compara su inicio con el fin (inicio + duración) de cada predecesora. **Respuesta:** ' + '. '.join(_resp) + '.')"""))
    celdas += apartado(2)
    celdas.append(md("El proyecto termina cuando termina su última actividad: fin del plan = máx(inicio + duración) "
                     "sobre todas las actividades. La tabla da el fin de cada actividad en cada plan."))
    celdas.append(code(r"""
        fines = planes.add(datos_proyecto['duration'], axis='rows')
        display(pd.DataFrame({f'fin en {NOMBRE[p]}': [f"{int(planes.loc[a, p])} + {int(datos_proyecto.loc[a, 'duration'])} = {int(fines.loc[a, p])}" for a in planes.index]
                              for p in planes.columns}, index=planes.index))
        duraciones = fines.max()
        _txt = []
        for p in planes.columns:
            ult = fines.index[fines[p] == duraciones[p]].tolist()
            comp = 'igual que la' if duraciones[p] == duracion_minima else (f'{int(duraciones[p] - duracion_minima)} más que la' if duraciones[p] > duracion_minima else f'{int(duracion_minima - duraciones[p])} menos que la')
            _txt.append(f'{NOMBRE[p]} terminaría en la semana {int(duraciones[p])} (fin de {lista_y(ult)}), {comp} duración mínima')
        Markdown(f'La duración mínima del proyecto (camino crítico) es {duracion_minima} semanas. **Respuesta:** ' + '; '.join(_txt) + '. '
                 'Pero un plan con violaciones no es ejecutable tal cual: su duración aparente es engañosa y hay que corregirlo (apartado 3).')"""))
    celdas += apartado(3)
    celdas.append(md("Corrección mínima: se recorren las actividades en orden topológico (cada una después de sus "
                     "predecesoras) y cada una empieza en el máximo entre la fecha propuesta y el fin de sus "
                     "predecesoras (ya corregidas). Así sólo se retrasan las actividades que lo necesitan y, en "
                     "cadena, sus sucesoras."))
    celdas.append(code(r"""
        G = nx.DiGraph(); G.add_nodes_from(datos_proyecto.index); G.add_edges_from((p, a) for a, ps in pred.items() for p in ps)
        orden = list(nx.topological_sort(G))
        def corregir(inicio):
            nuevo = inicio.copy()
            for a in orden:
                fin_pred = max([nuevo[p] + datos_proyecto.loc[p, 'duration'] for p in pred[a]], default=0)
                nuevo[a] = max(int(inicio[a]), int(fin_pred))
            return nuevo
        corregidos = pd.DataFrame({plan: corregir(planes[plan]) for plan in planes.columns})
        cambios = (corregidos != planes)
        display(pd.concat({'inicio propuesto': planes.rename(columns=NOMBRE), 'inicio corregido': corregidos.rename(columns=NOMBRE)}, axis=1))
        fines_c = corregidos.add(datos_proyecto['duration'], axis='rows').max()
        _txt = []
        for p in planes.columns:
            mov = [f'{a} pasa de {int(planes.loc[a, p])} a {int(corregidos.loc[a, p])}' for a in cambios.index[cambios[p]]]
            dif = int(fines_c[p] - duracion_minima)
            _txt.append(f'{NOMBRE[p]}: ' + (lista_y(mov) if mov else 'no hay que mover nada') + f'; termina en la semana {int(fines_c[p])}'
                        + (' (la duración mínima)' if dif == 0 else f' ({plural(dif, "semana")} más que la duración mínima)'))
        Markdown('**Respuesta:** ' + '; '.join(_txt) + '.')"""))
    celdas += apartado(4)
    celdas.append(md("El margen de cada actividad en un plan es lo que puede retrasarse sin mover el fin *de ese "
                     "plan*: su inicio más tardío (recalculado para la duración del plan corregido) menos el "
                     "inicio programado. Un margen 0 significa que la actividad empieza en su inicio más tardío."))
    celdas.append(code(r"""
        margen = pd.DataFrame({plan: (calendario['inicio_mas_tardio'] + (fines_c[plan] - duracion_minima) - corregidos[plan]) for plan in corregidos.columns})
        display(margen.rename(columns=lambda c: f'margen restante ({NOMBRE[c]})').T)
        sin_margen = {p: margen.index[margen[p] <= 0].tolist() for p in margen.columns}
        mas_tardio = max(margen.columns, key=lambda p: len(sin_margen[p]))
        otro = [p for p in margen.columns if p != mas_tardio][0]
        _extra = '; '.join(f'{NOMBRE[p]} termina además {plural(int(fines_c[p] - duracion_minima), "semana")} después de la duración mínima ({duracion_minima})'
                           for p in margen.columns if fines_c[p] > duracion_minima)
        Markdown('Actividades sin margen: ' + '; '.join(f'{NOMBRE[p]}: {len(sin_margen[p])} ({lista_y(sin_margen[p])})' for p in margen.columns)
                 + '. Suma de márgenes: ' + ', '.join(f'{NOMBRE[p]}: {int(margen[p].sum())}' for p in margen.columns) + ' semanas. '
                 f'**Respuesta:** el plan más tardío es el {NOMBRE[mas_tardio]}: {len(sin_margen[mas_tardio])} de sus {len(margen)} actividades empiezan en su inicio más tardío, '
                 'de modo que casi cualquier retraso se traslada al fin del proyecto (han consumido su holgura antes de empezar). '
                 + (f'El {NOMBRE[otro]} deja más holgura para absorber imprevistos ({int(margen[otro].sum())} semanas de margen sumado frente a {int(margen[mas_tardio].sum())}), '
                    'a cambio de necesitar antes los recursos y de adelantar los pagos.' if margen[otro].sum() > margen[mas_tardio].sum() else
                    f'Aun así, el margen sumado es mayor o igual en el {NOMBRE[mas_tardio]} ({int(margen[mas_tardio].sum())} frente a {int(margen[otro].sum())} semanas), concentrado en pocas actividades.') + (f' Ojo: {_extra}.' if _extra else ''))"""))
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
    celdas.append(md("Cada pregunta del test lleva la respuesta correcta calculada y tres distractores "
                     "*plausibles*: errores típicos (sumar todas las duraciones, confundir holgura libre y "
                     "total, contar las ficticias como rutas, usar la varianza en lugar de la desviación típica…). "
                     "En cada apartado se justifica la respuesta correcta con los números del proyecto y se "
                     "explica de dónde sale cada opción incorrecta."))
    celdas.append(code(r"""
        dur = datos_proyecto['duration']
        h_total, h_libre = int(calendario.loc[X, 'H_total']), int(calendario.loc[X, 'H_libre'])
        n_rutas, n_fict = len(R), len(proyecto.dummies)
        camino = ordena_ruta(list(proyecto.critical_path(dur).values())[0], calendario)
        Rt = R @ dur[R.columns]
        _rutas = {r: ' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario)) for r in R.index}
        p_mal = float(norm.cdf(plazo, duracion, sigma ** 2))

        def opciones(correcta, distractores, motivo_otro):
            # distractores: lista de (valor, motivo); se descartan los repetidos y se dejan 4 opciones
            vistos, lista, motivos = {correcta}, [correcta], {}
            for v, m in distractores:
                if v not in vistos and len(lista) < 4:
                    vistos.add(v); lista.append(v); motivos[v] = m or motivo_otro
            return lista, motivos

        def ruta_de(v):
            rs = [_rutas[r] for r in R.index if int(Rt[r]) == v]
            return f'es la duración de {lista_y(rs, "o")}, que no es la ruta más larga' if rs else None

        explicaciones = {}
        ops1, mot1 = opciones(duracion, [(int(dur.sum()), f'suma de las duraciones de todas las actividades ({" + ".join(str(int(d)) for d in dur)} = {int(dur.sum())}): ignora que las actividades en paralelo se solapan'),
                                         (duracion - 1, ruta_de(duracion - 1)), (duracion + 2, None), (duracion + 1, None), (duracion - 2, ruta_de(duracion - 2))],
                              'no es la duración de ninguna ruta del grafo')
        explicaciones[1] = (f'La duración del proyecto es la de la ruta más larga (camino crítico) {" – ".join(camino)}: '
                            f'{" + ".join(str(int(dur[a])) for a in camino)} = **{duracion} semanas**.', mot1)
        _r = calendario.loc[X]
        _suc = sorted(s for s, ps in prelaciones.immediate_predecessor.items() if X in ps)
        ops2, mot2 = opciones(h_total, [(h_libre, f'es la holgura libre de {X}, no la total' + (f' (mín ES de sus sucesoras − EF = mín(' + ', '.join(str(int(calendario.loc[s, "inicio_mas_temprano"])) for s in _suc) + f') − {int(_r.fin_mas_temprano)} = {h_libre})' if _suc else '')),
                                        (0, f'sería la holgura de una actividad crítica, y {X} no lo es'), (h_total + 2, None), (h_total + 1, None), (h_total + 3, None)],
                              f'no corresponde a ninguna holgura de {X}')
        explicaciones[2] = (f'Holgura total = inicio más tardío − inicio más temprano: H_total({X}) = LS − ES = {int(_r.inicio_mas_tardio)} − {int(_r.inicio_mas_temprano)} = **{h_total} semanas** '
                            f'(equivale a LF − EF = {int(_r.fin_mas_tardio)} − {int(_r.fin_mas_temprano)}).', mot2)
        ops3, mot3 = opciones(n_rutas, [(n_rutas + n_fict, f'resulta de sumar las {n_fict} ficticias a las {n_rutas} rutas, como si cada ficticia fuera una ruta más'),
                                        (n_rutas - 1, None), (n_rutas + 2, None), (n_rutas + 1, None), (n_rutas + 3, None)],
                              'no coincide con la enumeración de rutas')
        explicaciones[3] = (f'Enumerando los caminos del nodo inicial al final (las ficticias forman parte de los caminos, pero no son actividades) hay **{n_rutas} rutas**: '
                            + '; '.join(_rutas.values()) + '.', mot3)
        _G = proyecto.pert_graph
        _fic = []
        for u, v, d in _G.edges(data=True):
            if str(d['activity']).startswith('@'):
                llegan = sorted(dd['activity'] for _, _, dd in _G.in_edges(u, data=True) if not str(dd['activity']).startswith('@'))
                salen = sorted(dd['activity'] for _, _, dd in _G.out_edges(v, data=True) if not str(dd['activity']).startswith('@'))
                _fic.append(f'{d["activity"]} (del nodo {u} al {v}: ' + (f'{lista_y(salen)} espera{"n" if len(salen) > 1 else ""} también a {lista_y(llegan)}' if llegan and salen else 'separa dos actividades que irían entre los mismos nodos') + ')')
        ops4, mot4 = opciones(n_fict, [(0, 'sin ficticias no pueden representarse las prelaciones en las que dos actividades comparten sólo parte de sus predecesoras'),
                                       (n_fict + 1, None), (n_fict + 2, None), (n_fict + 3, None)],
                              'sobran ficticias: con menos ya se representan todas las prelaciones')
        explicaciones[4] = (f'El grafo PERT con el mínimo número de ficticias tiene **{n_fict}**: ' + '; '.join(_fic) + '.', mot4)
        ops5, mot5 = opciones(round(p_plazo, 2), [(round(p_mal, 2), f'usa la varianza σ² = {sigma ** 2:.3f} en lugar de la desviación típica σ = {sigma:.3f}'),
                                                  (0.5, f'sería la probabilidad de terminar en la duración media, {duracion} semanas, no en {plazo}'),
                                                  (round(1 - p_plazo, 2), f'es la probabilidad de *no* cumplir el plazo, P(T > {plazo}) = 1 − {p_plazo:.4f}'),
                                                  (0.95, None), (round(p_plazo / 2, 2), None)],
                              'no sale de ningún cálculo con la normal de la ruta crítica')
        _var = datos_proyecto['variance']
        explicaciones[5] = (f'La duración del proyecto se aproxima por una normal de media la del camino crítico ({duracion} semanas) y desviación típica la raíz de la suma de las varianzas de sus actividades: '
                            f'{explica_sigma_ruta(camino, _var)} semanas. Entonces ' + explica_probabilidad(duracion, sigma, plazo) + f', que redondeada es **{round(p_plazo, 2)}**.', mot5)
        preguntas_test = [
            ('¿Cuál es la duración del proyecto (semanas)?', ops1),
            (f'¿Cuál es la holgura total de {X} (semanas)?', ops2),
            ('¿Cuántas rutas tiene el grafo del nodo inicial al final?', ops3),
            ('¿Cuántas actividades ficticias necesita el PERT con el mínimo número de ellas?', ops4),
            (f'¿Probabilidad aproximada de terminar en {plazo} semanas o menos?', ops5),
        ]

        def resuelve(k):
            q, ops = preguntas_test[k - 1]
            texto, motivos = explicaciones[k]
            return Markdown(f'**{q}**\n\n' + '\n'.join(f'- {"✅" if j == 0 else "◻️"} {o}' for j, o in enumerate(ops)) + '\n\n'
                            + f'**Respuesta: {ops[0]}.** {texto}\n\n' + 'Opciones incorrectas:\n\n' + '\n'.join(f'- {v}: {m}.' for v, m in motivos.items()))"""))
    for k in range(1, 6):
        celdas += apartado(k)
        celdas.append(code(f"resuelve({k})"))
    celdas += apartado(6)
    celdas.append(md("El formato GIFT de Moodle escribe cada pregunta como `::título::texto {=correcta ~incorrecta "
                     "~incorrecta ####retroalimentación general}`; cada opción puede llevar su propia "
                     "retroalimentación tras `#`. Los caracteres especiales de GIFT (`~ = # { } :`) se escapan con "
                     "`\\` y las opciones se barajan al generar el fichero. Este es el fichero generado:"))
    celdas.append(code(r"""
        import os, re
        def _gift_esc(t):
            t = re.sub(r'\*\*|`', '', str(t)).replace('*', '')
            return re.sub(r'([~=#{}:])', r'\\\1', t)
        def a_gift(preguntas_test, explicaciones, titulo, semilla=1):
            r = np.random.default_rng(semilla)
            lineas = [f'$CATEGORY: {titulo}', '']
            for i, (q, ops) in enumerate(preguntas_test, 1):
                texto, motivos = explicaciones[i]
                orden = list(r.permutation(len(ops)))
                cuerpo = ' '.join(f'={_gift_esc(ops[j])}' if j == 0 else f'~{_gift_esc(ops[j])}#{_gift_esc(motivos[ops[j]])}' for j in orden)
                lineas.append(f'::{_gift_esc(titulo)} P{i}::{_gift_esc(q)} {{{cuerpo} ####{_gift_esc(texto)}}}')
                lineas.append('')
            return '\n'.join(lineas)
        gift = a_gift(preguntas_test, explicaciones, 'Proyecto test')
        os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)
        with open(os.path.expanduser('~/temp/cuestionario.gift'), 'w', encoding='utf8') as f:
            f.write(gift)
        Markdown('```text\n' + gift + '\n```')"""))
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
