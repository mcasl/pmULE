"""Colección 05 — Holguras total, libre e independiente, caminos críticos y
su interpretación."""
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="05_holguras", prefijo="Holguras",
            titulo="Holguras y camino crítico",
            descripcion="Calcular e interpretar las holguras total, libre e independiente; identificar "
                        "todos los caminos críticos.",
            n_defecto=3)


def _rasgos(cal):
    tl = (cal["H_total"] > cal["H_libre"]).sum()
    li = (cal["H_libre"] > cal["H_independiente"]).sum()
    lp = (cal["H_libre"] > 0).sum()
    return tl, li, lp


def _base(k, rng):
    nivel = (k - 1) % 3 + 1
    n = int(rng.choice([9, 10]))

    def acepta(i):
        tl, li, lp = _rasgos(i["calendario"])
        if not (1 <= i["n_dummies"] <= 4 and tl >= 2 and lp >= 1):
            return False
        if nivel >= 2:
            suc_i = P.sucesoras(i["preds"])
            cal_i = i["calendario"]
            if not any(suc_i[a] for a in cal_i.index[cal_i["H_libre"] > cal_i["H_independiente"]]):
                return False
        if nivel == 3 and i["n_criticas"] < 2:
            return False
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    cand_x = cal[(cal["H_total"] > cal["H_libre"])].index.tolist()
    X = cand_x[int(rng.integers(0, len(cand_x)))]
    hX = int(cal.loc[X, "H_total"])
    suc = P.sucesoras(info["preds"])
    cand_v = [a for a in cal[(cal["H_libre"] > cal["H_independiente"])].index if suc[a]]
    V = cand_v[int(rng.integers(0, len(cand_v)))] if cand_v else None

    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta.")]
    pregs = ["Calcule para cada actividad sus fechas más tempranas y más tardías de inicio y fin, y sus "
             "holguras total, libre e independiente.",
             "Determine el camino crítico. ¿Cuántas rutas críticas tiene el proyecto?",
             "¿Qué actividades pueden retrasarse sin afectar ni al plazo del proyecto ni al inicio más temprano "
             "de ninguna otra actividad? ¿Cuáles pueden retrasarse sin afectar al plazo pero obligando a "
             "retrasar a alguna sucesora? Justifíquelo con las holguras.",
             f"La actividad {X} se retrasa tanto como permite su holgura total ({hX} semanas). Compruebe que el "
             f"plazo no cambia y determine qué ocurre con las fechas y holguras de sus sucesoras."]
    if V is not None:
        pregs.append(f"Suponga que las predecesoras de {V} terminan en su fecha más tardía. ¿Cuánto puede "
                     f"retrasarse entonces {V} sin afectar a nadie? Relaciónelo con su holgura independiente.")
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos de los nodos y camino crítico", label="fig-pert")
    celdas.append(md("Con las fechas de cada actividad (ES, EF, LS, LF) y las de sus vecinas se definen las "
                     "tres holguras:\n\n"
                     "- **Total**: $H_t = LS - ES = LF - EF$. Retraso máximo sin alargar el proyecto, "
                     "suponiendo que las predecesoras acaban lo antes posible y las sucesoras pueden empezar lo "
                     "más tarde posible.\n"
                     "- **Libre**: $H_l = \\min_{s \\in suc} ES_s - EF$. Retraso máximo sin mover el inicio más "
                     "temprano de ninguna sucesora.\n"
                     "- **Independiente**: $H_i = \\max\\bigl(0,\\ \\min_{s \\in suc} ES_s - \\max_{p \\in pred} LF_p - D\\bigr)$. "
                     "Retraso disponible incluso si las predecesoras acaban lo más tarde posible y las sucesoras "
                     "deben empezar lo antes posible.\n\n"
                     "Siempre $H_t \\ge H_l \\ge H_i$."))
    celdas.append(code('#| tbl-cap: "Calendario y holguras del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas.append(md("Con los números sustituidos, actividad a actividad (ES, EF, LS y LF son inicio y fin más "
                     "tempranos y más tardíos; «pred.» y «suc.» las predecesoras y sucesoras inmediatas):"))
    celdas.append(code('#| tbl-cap: "Cálculo paso a paso de fechas y holguras"\n#| label: tbl-holgurasPaso\n'
                       "detalle_holguras(calendario, prelaciones)"))
    celdas += apartado(ap); ap += 1
    celdas.append(code("duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "for nombre, acts in rutas_criticas.items():\n"
                       "    print(f'{nombre}: {\" - \".join(acts)}')\n"
                       "Markdown(f'La duración del proyecto es {duracion_proyecto} semanas y hay {len(rutas_criticas)} ruta(s) crítica(s). '\n"
                       "         'Las actividades críticas son las de holgura total nula; cuando varias rutas comparten la duración máxima, '\n"
                       "         'todas ellas son críticas y un retraso en cualquiera alarga el proyecto.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(code("sin_afectar_a_nadie = calendario.index[calendario['H_libre'] > 0].tolist()\n"
                       "solo_sucesoras = calendario.index[(calendario['H_total'] > calendario['H_libre'])].tolist()\n"
                       "Markdown(f'''\n"
                       "- Pueden retrasarse (hasta su holgura libre) sin mover a ninguna otra actividad: **{\", \".join(sin_afectar_a_nadie)}**.\n"
                       "- Tienen holgura total mayor que la libre, así que agotar la total obliga a retrasar sucesoras aunque el plazo se mantenga: **{\", \".join(solo_sucesoras)}**.\n"
                       "''')"))
    celdas.append(code('#| tbl-cap: "Actividades cuyo retraso repercute en sus sucesoras"\n#| label: tbl-holgurasDistintas\n'
                       "calendario.loc[solo_sucesoras, ['H_total', 'H_libre', 'H_independiente']]"))
    celdas += apartado(ap); ap += 1
    celdas.append(md(f"Un retraso de {X} igual a su holgura total equivale a alargar su duración en {hX} semanas. "
                     f"Recalculamos el calendario con esa duración."))
    celdas.append(code("duraciones_retraso = datos_proyecto['duration'].copy()\n"
                       f"duraciones_retraso['{X}'] += {hX}\n"
                       "calendario_retraso = proyecto.calendar(duraciones_retraso)\n"
                       f"sucesoras = sorted(s for s, ps in prelaciones.immediate_predecessor.items() if '{X}' in ps)\n"
                       "comparacion = pd.concat({'antes': calendario.loc[sucesoras, ['inicio_mas_temprano', 'H_total', 'H_libre']],\n"
                       "                         'después': calendario_retraso.loc[sucesoras, ['inicio_mas_temprano', 'H_total', 'H_libre']]}, axis=1)\n"
                       "comparacion"))
    celdas.append(code("Markdown(f'La duración del proyecto sigue siendo {proyecto.duration(duraciones_retraso)} semanas, '\n"
                       f"         f'pero {X} ha consumido holgura que compartía con sus sucesoras: las que tenían inicio más temprano '\n"
                       f"         f'anterior al nuevo fin de {X} se retrasan y su holgura total se reduce; alguna pasa a ser crítica.')"))
    if V is not None:
        hi = int(cal.loc[V, "H_independiente"]); hl = int(cal.loc[V, "H_libre"])
        celdas += apartado(ap); ap += 1
        celdas.append(code(f"fila = calendario.loc['{V}']\n"
                           f"predecesoras = sorted(prelaciones.immediate_predecessor['{V}'])\n"
                           f"sucesoras_V = sorted(s for s, ps in prelaciones.immediate_predecessor.items() if '{V}' in ps)\n"
                           "fin_tardio_pred = int(calendario.loc[predecesoras, 'fin_mas_tardio'].max())\n"
                           "inicio_temprano_suc = int(calendario.loc[sucesoras_V, 'inicio_mas_temprano'].min())\n"
                           "margen = inicio_temprano_suc - fin_tardio_pred - int(fila['duracion'])\n"
                           f"Markdown(f'''Si las predecesoras de {V} ({{\", \".join(predecesoras)}}) acaban en su fecha más tardía, {V} no puede empezar antes de la semana {{fin_tardio_pred}}. "
                           f"Para no mover a sus sucesoras ({{\", \".join(sucesoras_V)}}) debe acabar antes de la semana {{inicio_temprano_suc}}. "
                           f"Le quedan {{inicio_temprano_suc}} − {{fin_tardio_pred}} − {{int(fila[\"duracion\"])}} = {{margen}} semanas, "
                           f"es decir, su holgura independiente, {{int(fila[\"H_independiente\"])}} (si el cálculo sale negativo la holgura es cero). "
                           f"Su holgura libre, {{int(fila[\"H_libre\"])}}, supone en cambio que las predecesoras acaban lo antes posible.''')"))
    sub = f"{n} actividades, {info['n_criticas']} ruta(s) crítica(s)"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _compartida(rng):
    """Dos actividades en cadena con la misma holgura total: no pueden gastarla las dos."""
    n = int(rng.choice([9, 10]))
    estado = {}

    def acepta(i):
        cal, preds = i["calendario"], i["preds"]
        if not (1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1):
            return False
        pares = [(p, a) for a, ps in preds.items() for p in ps
                 if cal.loc[p, "H_total"] == cal.loc[a, "H_total"] > 0 and cal.loc[p, "H_libre"] == 0]
        if not pares:
            return False
        estado["par"] = pares[int(rng.integers(0, len(pares)))]
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    X, Y = estado["par"]
    h = int(cal.loc[X, "H_total"])
    conj = "e" if Y.startswith("I") else "y"
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta.")]
    celdas.append(preguntas([
        "Calcule el calendario del proyecto con las holguras total, libre e independiente.",
        f"{X} {conj} {Y} son consecutivas y tienen la misma holgura total. El responsable de {X} pide retrasarla "
        f"{h} semanas y el de {Y} también. ¿Pueden concederse las dos peticiones sin alargar el proyecto? Razónelo "
        "antes de calcular.",
        f"Compruébelo: recalcule el calendario con {X} retrasada {h} semanas y después con las dos retrasadas.",
        "¿Qué holgura expresa cuánto puede retrasarse cada una sin quitarle margen a la otra? ¿Cómo repartiría "
        f"las {h} semanas entre {X} {conj} {Y}?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion_proyecto = proyecto.duration(datos_proyecto['duration'])"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario y holguras"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas.append(md("Con los números sustituidos, actividad a actividad (ES, EF, LS y LF son inicio y fin más "
                     "tempranos y más tardíos; «pred.» y «suc.» las predecesoras y sucesoras inmediatas):"))
    celdas.append(code('#| tbl-cap: "Cálculo paso a paso de fechas y holguras"\n#| label: tbl-holgurasPaso\n'
                       "detalle_holguras(calendario, prelaciones)"))
    celdas += apartado(2)
    celdas.append(md(f"No. La holgura total de {X} y la de {Y} son la *misma* holgura: la de la cadena en la que "
                     f"están. {X} tiene holgura libre cero, así que cualquier retraso suyo mueve el inicio de {Y} y le "
                     f"consume holgura. Si {X} gasta las {h} semanas, {Y} se queda sin ninguna; si además {Y} se "
                     f"retrasa, el proyecto se alarga."))
    celdas += apartado(3)
    celdas.append(code(f"X, Y, h = '{X}', '{Y}', {h}\n"
                       "d1 = datos_proyecto['duration'].copy(); d1[X] += h\n"
                       "d2 = d1.copy(); d2[Y] += h\n"
                       "cal1, cal2 = proyecto.calendar(d1), proyecto.calendar(d2)\n"
                       "pd.DataFrame({'original': calendario.loc[[X, Y], 'H_total'], f'{X} retrasada {h}': cal1.loc[[X, Y], 'H_total'],\n"
                       "              'las dos retrasadas': cal2.loc[[X, Y], 'H_total']})"))
    celdas.append(code("Markdown(f'Duración del proyecto: {duracion_proyecto} semanas en la planificación original, {proyecto.duration(d1)} con {X} retrasada '\n"
                       "         f'y **{proyecto.duration(d2)}** con las dos retrasadas: el segundo retraso ya no cabe en la holgura.')"))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'''La holgura libre de {X} es {int(calendario.loc[X, \"H_libre\"])}: es lo que puede retrasarse sin mover a {Y}. "
                       "La holgura independiente de {Y} es {int(calendario.loc[Y, \"H_independiente\"])}: lo que puede retrasarse {Y} aunque {X} acabe lo más tarde posible. "
                       "Las {h} semanas son un recurso compartido de la cadena: cualquier reparto x + y ≤ {h} entre {X} " + conj + " {Y} respeta el plazo; "
                       "lo que no es posible es que cada una disponga de {h} por su cuenta.''')"))
    return celdas, f"{n} actividades; {X} {conj} {Y} comparten {h} semanas de holgura"


def _vigilancia(rng):
    """Ordenar las actividades por riesgo de retraso y la pregunta en negativo."""
    n = int(rng.choice([9, 10]))

    def acepta(i):
        cal = i["calendario"]
        return (1 <= i["n_dummies"] <= 4 and i["n_criticas"] >= 1
                and ((cal["H_total"] > 0) & (cal["H_libre"] == 0)).sum() >= 1
                and (cal["H_libre"] > 0).sum() >= 2 and cal["H_total"].nunique() >= 4)

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta. El jefe "
                 "de proyecto sólo puede hacer seguimiento estrecho de cuatro actividades.")]
    celdas.append(preguntas([
        "Calcule el calendario del proyecto con sus holguras.",
        "Ordene las actividades de mayor a menor necesidad de vigilancia (primero las críticas; entre las demás, "
        "menor holgura total primero y, a igualdad, menor holgura libre). ¿Cuáles son las cuatro elegidas?",
        "¿Qué actividades *no críticas* no pueden retrasarse ni una semana sin obligar a mover alguna sucesora? "
        "¿Por qué no son críticas si no admiten retraso?",
        "¿Qué actividad puede retrasarse más semanas sin que nadie lo note? ¿Cuántas?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario y holguras"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas.append(md("Con los números sustituidos, actividad a actividad (ES, EF, LS y LF son inicio y fin más "
                     "tempranos y más tardíos; «pred.» y «suc.» las predecesoras y sucesoras inmediatas):"))
    celdas.append(code('#| tbl-cap: "Cálculo paso a paso de fechas y holguras"\n#| label: tbl-holgurasPaso\n'
                       "detalle_holguras(calendario, prelaciones)"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Actividades ordenadas por necesidad de vigilancia"\n#| label: tbl-orden\n'
                       "orden = calendario.sort_values(['H_total', 'H_libre', 'duracion'], ascending=[True, True, False])\n"
                       "orden['prioridad'] = range(1, len(orden) + 1)\n"
                       "orden[['prioridad', 'H_total', 'H_libre', 'H_independiente', 'duracion']]"))
    celdas.append(code("Markdown(f'Las cuatro actividades a vigilar son **{\", \".join(orden.index[:4])}**. '\n"
                       "         'Si hay más de cuatro críticas, el criterio de desempate es la duración (una actividad larga tiene más ocasiones de retrasarse).')"))
    celdas += apartado(3)
    celdas.append(code("sin_margen = calendario.index[(calendario['H_total'] > 0) & (calendario['H_libre'] == 0)].tolist()\n"
                       "Markdown(f'''Las actividades {\", \".join(sin_margen)} tienen holgura libre cero pero holgura total positiva: cualquier retraso suyo "
                       "mueve el inicio más temprano de una sucesora, pero esa sucesora tiene margen suficiente para absorberlo sin que el proyecto se alargue. "
                       "No son críticas porque el plazo no depende de ellas; lo que no admiten es retrasarse sin molestar.''')"))
    celdas += apartado(4)
    celdas.append(code("mas_libre = calendario['H_libre'].idxmax()\n"
                       "Markdown(f'{mas_libre} puede retrasarse hasta {int(calendario.loc[mas_libre, \"H_libre\"])} semanas sin mover a ninguna otra actividad ni al plazo (su holgura libre).')"))
    return celdas, f"{n} actividades; ordenación por vigilancia y actividades sin margen"


def _diagnostico(rng):
    """Se da un calendario resuelto con dos errores; hay que detectarlos y corregirlos."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and (i["calendario"]["H_total"] > 0).sum() >= 3,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    tabla = cal[["inicio_mas_temprano", "fin_mas_temprano", "inicio_mas_tardio", "fin_mas_tardio", "H_total", "H_libre"]].copy()
    no_crit = cal.index[cal["H_total"] > 0].tolist()
    crit = cal.index[cal["H_total"] == 0].tolist()
    e1 = no_crit[int(rng.integers(0, len(no_crit)))]
    tabla.loc[e1, "H_libre"] = int(tabla.loc[e1, "H_total"]) + 1          # H_libre > H_total: imposible
    resto = [a for a in crit if a != e1]
    e2 = resto[int(rng.integers(0, len(resto)))]
    tabla.loc[e2, "inicio_mas_tardio"] = int(tabla.loc[e2, "inicio_mas_tardio"]) + 2   # LS - ES ≠ H_total
    celdas = [md("# Enunciado"),
              md("Un alumno ha entregado el calendario que se adjunta para el proyecto cuyo cuadro de prelaciones y "
                 "duraciones (semanas) también se da. Contiene dos errores.")]
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(code('#| tbl-cap: "Calendario entregado (con errores)"\n#| label: tbl-calendarioEntregado\n'
                       + literal_dataframe(tabla.astype(int), nombre="calendario_entregado")))
    celdas.append(preguntas([
        "Sin recalcular todo el proyecto, localice los errores comprobando las relaciones que debe cumplir "
        "cualquier calendario: EF = ES + d, LF = LS + d, H_total = LS − ES, 0 ≤ H_libre ≤ H_total, y ES de cada "
        "actividad igual al máximo EF de sus predecesoras.",
        "Explique qué significa cada error (qué se ha calculado mal).",
        "Calcule el calendario correcto y compárelo."]))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "c = calendario_entregado.copy()\n"
                       "d = datos_proyecto['duration']"))
    celdas += apartado(1)
    celdas.append(code("comprobaciones = pd.DataFrame({\n"
                       "    'EF = ES + d': c['fin_mas_temprano'] == c['inicio_mas_temprano'] + d,\n"
                       "    'LF = LS + d': c['fin_mas_tardio'] == c['inicio_mas_tardio'] + d,\n"
                       "    'H_total = LS − ES': c['H_total'] == c['inicio_mas_tardio'] - c['inicio_mas_temprano'],\n"
                       "    'H_libre ≤ H_total': c['H_libre'] <= c['H_total'],\n"
                       "    'ES = máx EF pred.': [c.loc[a, 'inicio_mas_temprano'] == max((c.loc[p, 'fin_mas_temprano'] for p in prelaciones.immediate_predecessor[a]), default=0) for a in c.index]})\n"
                       "comprobaciones[~comprobaciones.all(axis=1)]"))
    celdas += apartado(2)
    celdas.append(md(f"- En {e1} la holgura libre es mayor que la total, lo que es imposible: la holgura libre es "
                     f"una parte de la total (la que no afecta a las sucesoras).\n"
                     f"- En {e2} el inicio más tardío no cuadra con el fin más tardío menos la duración (ni con "
                     f"H_total = LS − ES): se ha copiado mal o se ha calculado el tardío sin restar la duración."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Calendario correcto"\n#| label: tbl-calendarioCorrecto\n'
                       "calendario = proyecto.calendar(d)\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'H_libre']]"))
    celdas.append(code("difieren = (calendario[c.columns] != c).any(axis=1)\n"
                       "pd.concat({'entregado': c[difieren], 'correcto': calendario.loc[difieren, c.columns]}, axis=1)"))
    return celdas, f"{n} actividades; calendario con errores en {e1} y {e2}"



def _mini(rng):
    """Cinco o seis actividades: todo el cálculo a mano, paso a paso, sin ficticias."""
    n = int(rng.choice([5, 6]))

    def acepta(i):
        cal = i["calendario"]
        return (i["n_dummies"] == 0 and i["n_criticas"] == 1 and (cal["H_total"] > 0).sum() >= 2
                and (cal["H_total"] > cal["H_libre"]).sum() >= 1 and 2 <= i["n_rutas"] <= 4)

    info = P.buscar(rng, acepta, n=n, n_inicio=2, n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    X = cal.index[(cal["H_total"] > cal["H_libre"])][0]
    celdas = [md("# Enunciado"),
              md("Considere el pequeño proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta. "
                 "Resuélvalo completamente a mano, anotando cada operación, antes de comprobarlo con la librería.")]
    celdas.append(preguntas([
        "Dibuje el grafo PERT y enumere todas las rutas desde el inicio hasta el final con su duración. "
        "¿Cuál es la duración del proyecto y cuál la ruta crítica?",
        "Calcule, por el paso hacia adelante, el inicio y fin más tempranos de cada actividad; por el paso "
        "hacia atrás, el fin e inicio más tardíos.",
        "Calcule las holguras total, libre e independiente de cada actividad escribiendo la fórmula con los "
        "números sustituidos.",
        f"La actividad {X} tiene holgura total mayor que la libre. Explique con las fechas de sus sucesoras "
        "por qué."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Rutas del proyecto y su duración"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "detalle_rutas(R, datos_proyecto['duration'][R.columns])"))
    celdas.append(code("duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "camino_critico = proyecto.display_critical_path(datos_proyecto['duration'])\n"
                       "Markdown(f'La duración del proyecto es la de la ruta más larga: {duracion_proyecto} días.')"))
    celdas += apartado(2)
    celdas.append(md("Paso hacia adelante (de izquierda a derecha): ES = máximo de los EF de las predecesoras "
                     "(0 si es inicial); EF = ES + duración. Paso hacia atrás (de derecha a izquierda): LF = "
                     "mínimo de los LS de las sucesoras (la duración del proyecto si es final); LS = LF − duración."))
    celdas.append(code('#| tbl-cap: "Paso hacia adelante y hacia atrás"\n#| label: tbl-pasos\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "detalle = detalle_holguras(calendario, prelaciones)\n"
                       "detalle[['ES = máx EF pred.', 'EF = ES + d', 'LF = mín LS suc.', 'LS = LF − d']]"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Holguras con los números sustituidos"\n#| label: tbl-holgurasPaso\n'
                       "detalle[['H_total = LS − ES', 'H_libre = mín ES suc. − EF', 'H_indep. = máx(0, mín ES suc. − máx LF pred. − d)']]"))
    celdas.append(code('#| tbl-cap: "Calendario y holguras según la librería"\n#| label: tbl-calendario\n'
                       "calendario"))
    celdas += apartado(4)
    celdas.append(code(f"X = '{X}'\n"
                       "sucesoras_X = [a for a, ps in prelaciones.immediate_predecessor.items() if X in ps]\n"
                       "Markdown(f'''{X} termina lo más pronto en {int(calendario.loc[X, \"fin_mas_temprano\"])} y sus sucesoras ({\", \".join(sucesoras_X)}) "
                       "empiezan lo más pronto en {int(calendario.loc[sucesoras_X, \"inicio_mas_temprano\"].min())}: eso limita su holgura libre a "
                       "{int(calendario.loc[X, \"H_libre\"])}. Pero las sucesoras pueden empezar lo más tarde en {int(calendario.loc[sucesoras_X, \"inicio_mas_tardio\"].min())}, "
                       "así que {X} puede retrasarse hasta {int(calendario.loc[X, \"H_total\"])} si arrastra a sus sucesoras (holgura total).''')"))
    return celdas, f"{n} actividades, {info['n_rutas']} rutas, sin ficticias; cálculo completo a mano"


VARIANTES = [
    Variante("tres-holguras", 1, "holguras total, libre e independiente; caminos críticos; retraso de una actividad",
             lambda rng: _base(1, rng), grafo="roy"),
    Variante("tres-holguras-b", 2, "ídem con holgura independiente distinta de la libre", lambda rng: _base(2, rng), grafo="roy"),
    Variante("varias-criticas", 2, "ídem con dos o más rutas críticas", lambda rng: _base(3, rng), grafo="roy"),
    Variante("holgura-compartida", 2, "dos actividades consecutivas no pueden gastar las dos la misma holgura total",
             _compartida, grafo="roy"),
    Variante("vigilancia", 2, "ordenar por riesgo; no críticas sin margen; la que más puede retrasarse sin que se note",
             _vigilancia, grafo="roy"),
    Variante("diagnostico", 3, "calendario entregado con dos errores: localizarlos por las relaciones entre fechas y holguras",
             _diagnostico, grafo="roy"),
    Variante("mini-a-mano", 1, "cinco o seis actividades sin ficticias: rutas, pasos adelante y atrás y holguras con los números sustituidos",
             _mini, grafo="roy"),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
