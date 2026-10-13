"""Colección 05 — Holguras total, libre e independiente, caminos críticos y
su interpretación."""
from .. import proyectos as P
from ..cuaderno import md, code, preguntas, apartado, literal_dataframe

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


def generar(k, rng):
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
    celdas.append(code('#| fig-cap: "Grafo PERT con tiempos de los nodos y camino crítico"\n#| label: fig-pert\n'
                       "proyecto.pert(durations=datos_proyecto['duration'], ranksep=1)"))
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
