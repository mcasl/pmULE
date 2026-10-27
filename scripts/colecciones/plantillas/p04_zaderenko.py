"""Colección 04 — Método de Zaderenko: tiempos tempranos y tardíos de los
nodos, calendario de las actividades y diagrama de Gantt."""
import networkx as nx
import pandas as pd
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX, ESTADO)

META = dict(slug="04_zaderenko", prefijo="Zaderenko",
            titulo="Método de Zaderenko y calendario",
            descripcion="Calcular con la matriz de Zaderenko los tiempos tempranos y tardíos, el calendario "
                        "de las actividades y el diagrama de Gantt.",
            n_defecto=3)

# Celda auxiliar (se define en cada cuaderno, sin salida): tiempos de los nodos y fechas de las
# actividades leídas de los nodos del grafo PERT, con los números sustituidos.
_AUX_ZADERENKO = r'''
_COLS_CAL = {'inicio_mas_temprano': 'ES', 'fin_mas_temprano': 'EF', 'inicio_mas_tardio': 'LS',
             'fin_mas_tardio': 'LF', 'H_total': 'H total'}


def tiempos_nodos(proyecto, duraciones):
    """Tiempos temprano y tardío de cada nodo (sucesos del grafo PERT)."""
    n = proyecto.calculate_pert(durations=duraciones)['nodes']
    t = pd.DataFrame({'tiempo temprano': n['early'].astype(int).values, 'tiempo tardío': n['late'].astype(int).values},
                     index=pd.Index([str(i) for i in n.index], name='nodo'))
    return t


def arcos_pert(proyecto):
    """Actividad → arco (nodo origen → nodo destino); las ficticias al final."""
    arcos = sorted(proyecto.nodes_from_edges.items(), key=lambda kv: (kv[0].startswith('@'), int(kv[1][0]), int(kv[1][1]), kv[0]))
    t = pd.DataFrame({'nodo origen i': [i for _, (i, j) in arcos], 'nodo destino j': [j for _, (i, j) in arcos]},
                     index=pd.Index([f'ficticia {a}' if a.startswith('@') else a for a, _ in arcos], name='actividad'))
    return t


def respuesta_duracion(proyecto, duraciones, unidad='días'):
    t = tiempos_nodos(proyecto, duraciones)
    finales = [str(i) for i in proyecto.pert_graph.nodes if proyecto.pert_graph.out_degree(i) == 0]
    f = finales[0]
    return Markdown(f"**Respuesta:** la duración del proyecto es el tiempo temprano del último nodo, el {f}: "
                    f"**{int(t.loc[f, 'tiempo temprano'])} {unidad}** (y su tiempo tardío se iguala a ese valor para "
                    f"empezar el cálculo hacia atrás).")


def fechas_desde_nodos(proyecto, duraciones):
    """Fechas y holgura total de cada actividad i → j a partir de los tiempos de sus nodos:
    ES = t_i, EF = t_i + d, LF = t*_j, LS = t*_j − d, H total = t*_j − t_i − d."""
    t = tiempos_nodos(proyecto, duraciones)
    filas = []
    for a, (i, j) in sorted(proyecto.nodes_from_edges.items()):
        if a.startswith('@'):
            continue
        d, ti, tj = int(duraciones[a]), int(t.loc[i, 'tiempo temprano']), int(t.loc[j, 'tiempo tardío'])
        filas.append({'actividad': a, 'arco i → j': f'{i} → {j}', 'ES = t_i': f'{ti}', 'EF = t_i + d': f'{ti} + {d} = {ti + d}',
                      'LF = t*_j': f'{tj}', 'LS = t*_j − d': f'{tj} − {d} = {tj - d}',
                      'H total = t*_j − t_i − d': f'{tj} − {ti} − {d} = {tj - ti - d}'})
    return pd.DataFrame(filas).set_index('actividad')


def respuesta_calendario(calendario, unidad='días'):
    crit = calendario.index[calendario['H_total'] == 0].tolist()
    holg = calendario.index[calendario['H_total'] > 0].tolist()
    return Markdown(f"**Respuesta:** el calendario es el de la tabla. Las actividades críticas (holgura total 0) son "
                    f"{lista_y(crit)}; las demás pueden retrasarse sin alargar el proyecto hasta su holgura total: "
                    + lista_y(f"{a} {plural(int(calendario.loc[a, 'H_total']), unidad[:-1] if unidad.endswith('s') else unidad)}" for a in holg) + '.')

def respuesta_critico(proyecto, duraciones, unidad='días'):
    """Camino(s) crítico(s) como suma y respuesta."""
    display(explica_camino_critico(proyecto, duraciones, unidad=unidad))
    cal = proyecto.calendar(duraciones)
    rutas = [' – '.join(ordena_ruta(r, cal)) for r in proyecto.critical_path(duraciones).values()]
    return Markdown(f"**Respuesta:** {plural(len(rutas), 'camino crítico', 'caminos críticos')}: {lista_y(rutas)}; "
                    f"son las actividades con holgura total 0 en el calendario.")


def deduce_prelaciones(matriz):
    """Cuadro de prelaciones a partir de la matriz de Zaderenko con nombres: (datos, tabla del razonamiento)."""
    arcos = {}
    for i in matriz.index:
        for j in matriz.columns:
            txt = matriz.loc[i, j]
            if txt:
                arcos[(str(i), str(j))] = ('ficticia', 0) if txt == 'ficticia' else (txt.split(' ')[0], int(txt.split('(')[1].rstrip(')')))

    def entran(nodo):
        acts, via = set(), []
        for (i, j), (nombre, _) in sorted(arcos.items()):
            if j == nodo:
                if nombre == 'ficticia':
                    sub, _ = entran(i)
                    acts |= sub
                    via.append(f"ficticia {i} → {j}, que trae {lista_y(sorted(sub))} (lo que llega al nodo {i})")
                else:
                    acts.add(nombre)
                    via.append(f"{nombre} (arco {i} → {j})")
        return acts, via

    filas, prel, dur = [], {}, {}
    for (i, j), (nombre, d) in sorted(arcos.items(), key=lambda kv: kv[1][0]):
        if nombre == 'ficticia':
            continue
        acts, via = entran(i)
        prel[nombre], dur[nombre] = ','.join(sorted(acts)) or '---', d
        filas.append({'actividad': nombre, 'arco': f'{i} → {j}',
                      'llega al nodo origen': '; '.join(via) or f'nada: el nodo {i} es el inicial',
                      'predecesoras inmediatas': ', '.join(sorted(acts)) or '— (inicial)'})
    datos = pd.DataFrame({'predecessor': pd.Series(prel), 'duration': pd.Series(dur)}).sort_index()
    datos.index.name = 'activity'
    return datos, pd.DataFrame(filas).set_index('actividad')


def compara_matriz(matriz, proyecto):
    """¿Reproduce el grafo reconstruido los arcos de la matriz del enunciado?"""
    enun = {}
    for i in matriz.index:
        for j in matriz.columns:
            txt = matriz.loc[i, j]
            if txt and txt != 'ficticia':
                enun[txt.split(' ')[0]] = (str(i), str(j))
    rec = {a: ij for a, ij in proyecto.nodes_from_edges.items() if not a.startswith('@')}
    n_fict_e = int((matriz == 'ficticia').sum().sum())
    n_fict_r = sum(a.startswith('@') for a in proyecto.nodes_from_edges)
    distintos = [a for a in sorted(enun) if enun[a] != rec.get(a)]
    if not distintos:
        txt = (f"**Respuesta:** sí la reproduce: las {len(enun)} actividades están en los mismos arcos (i → j) que en la "
               f"matriz del enunciado, con la misma numeración de nodos, y hay {n_fict_r} ficticias "
               f"({'las mismas' if n_fict_r == n_fict_e else f'{n_fict_e} en el enunciado'}).")
    else:
        txt = (f"**Respuesta:** el grafo tiene la misma estructura, pero la numeración de algunos nodos difiere (hay varias "
               f"numeraciones válidas): " + '; '.join(f"{a}: {enun[a][0]} → {enun[a][1]} en el enunciado, "
                                                       f"{rec[a][0]} → {rec[a][1]} en el grafo" for a in distintos) + '.')
    return Markdown(txt)


def rutas_por_actividad(proyecto, duraciones):
    """Para cada actividad, la ruta más larga que pasa por ella y su duración."""
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    d = pd.Series(duraciones)
    largo = {r: int(sum(d[a] for a in R.columns[R.loc[r] == 1])) for r in R.index}
    res = {}
    for a in R.columns:
        rs = [r for r in R.index if R.loc[r, a] == 1]
        r = max(rs, key=lambda r: largo[r])
        res[a] = (' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)), largo[r])
    return res, max(largo.values())


def respuesta_forma(proyecto, duraciones, unidad='días'):
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    n_acts = R.sum(axis=1)
    mas = n_acts[n_acts == n_acts.max()].index
    nombres = [' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)) for r in mas]
    criticas = [' – '.join(ordena_ruta(r, cal)) for r in proyecto.critical_path(duraciones).values()]
    coincide = set(nombres) & set(criticas)
    return Markdown(f"**Respuesta:** mirando solo la forma, las candidatas son las rutas con más actividades: "
                    f"{lista_y(nombres)} ({int(n_acts.max())} actividades). Con las duraciones (apartado 3) la ruta más "
                    f"larga resulta ser {lista_y(criticas)}"
                    + (', que sí está entre ellas.' if coincide else
                       ', que no está entre ellas: la forma del grafo no basta, hacen falta las duraciones.'))


def respuesta_prediccion(proyecto, duraciones, casos, unidad='días'):
    """Predicción de cada caso sumando las duraciones de las rutas (sin tiempos de los nodos)."""
    mayor, T = rutas_por_actividad(proyecto, duraciones)
    L = [f"**Respuesta** (la ruta más larga del proyecto suma {T} {unidad}):", '']
    for letra, (a, extra) in zip('abc', casos):
        ruta, l = mayor[a]
        margen = T - l
        if margen == 0:
            pred = f"está en la ruta más larga, {ruta} ({l}): cualquier aumento alarga el plazo, **sí cambia** (+{extra})"
        elif extra <= margen:
            pred = (f"la ruta más larga que pasa por {a} es {ruta}, con {l}, {T} − {l} = {plural(margen, unidad[:-1])} menos que la "
                    f"más larga: el aumento de {extra} cabe ({extra} ≤ {margen}), **no cambia**")
        else:
            pred = (f"la ruta más larga que pasa por {a} es {ruta}, con {l}, {T} − {l} = {plural(margen, unidad[:-1])} menos que la "
                    f"más larga: el aumento de {extra} no cabe ({extra} > {margen}), **sí cambia** (+{extra - margen})")
        L.append(f"- ({letra}) {a} dura {extra} {unidad} más: {pred}.")
    return Markdown('\n'.join(L))


def respuesta_comprobacion(calendario, T0, casos, unidad='días'):
    L = []
    for letra, (a, extra) in zip('abc', casos):
        h = int(calendario.loc[a, 'H_total'])
        if extra <= h:
            L.append(f"- ({letra}) {a}: holgura total {h}, aumento {extra} ≤ {h}: el plazo sigue en {T0} {unidad}.")
        else:
            L.append(f"- ({letra}) {a}: holgura total {h}, aumento {extra} > {h}: el plazo pasa a {T0} + ({extra} − {h}) = {T0 + extra - h} {unidad}.")
    return Markdown('\n'.join(L) + "\n\n**Respuesta:** las tres predicciones del apartado 2 se confirman. El dato que permitía acertar es la "
                    "**holgura total** de cada actividad: el plazo cambia si y sólo si el aumento la supera (en una crítica es "
                    "0 y cualquier aumento cuenta), y el proyecto se alarga exactamente el exceso.")

def nodos_con_plazo(proyecto, duraciones, plazo):
    """Tiempos de los nodos con el tiempo tardío del último nodo igual al plazo impuesto."""
    t = tiempos_nodos(proyecto, duraciones)
    k = int(proyecto.duration(duraciones)) - int(plazo)
    t[f'tardío con plazo = tardío − {k}'] = t['tiempo tardío'] - k
    return t.T


def respuesta_negativas(con_plazo, k, unidad='semanas'):
    neg = con_plazo.index[con_plazo['H_total'] < 0].tolist()
    return Markdown(f"**Respuesta:** {plural(len(neg), 'actividad tiene', 'actividades tienen')} holgura total negativa: "
                    + lista_y(f"{a} ({int(con_plazo.loc[a, 'H_total'])})".replace('(-', '(−') for a in neg)
                    + f". Una holgura −h significa que, aun empezando lo antes posible, la actividad acaba h {unidad} después de "
                    f"lo que permite el plazo (su fin más temprano supera en h su nuevo fin más tardío): la ruta más larga que "
                    f"pasa por ella excede el plazo en h {unidad}. Las críticas tienen −{k}, todo el exceso del proyecto; para "
                    f"cumplir el plazo hay que recuperar ese tiempo acortando actividades de esas rutas.")


def rutas_exceden(proyecto, duraciones, plazo):
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(duraciones)
    d = pd.Series(duraciones)
    filas = []
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        L = int(sum(d[a] for a in acts))
        if L > plazo:
            filas.append({'ruta': ' – '.join(acts), 'duración = suma': ' + '.join(str(int(d[a])) for a in acts) + f' = {L}',
                          'exceso = duración − plazo': f'{L} − {plazo} = {L - plazo}', '_exceso': L - plazo, '_acts': acts})
    return pd.DataFrame(filas).sort_values('_exceso', ascending=False, kind='stable').reset_index(drop=True)


def respuesta_exceden(tabla, duraciones, unidad='semanas'):
    d = pd.Series(duraciones)
    n = len(tabla)
    m = int(tabla['_exceso'].max())
    comunes = [a for a in tabla['_acts'].iloc[0] if all(a in acts for acts in tabla['_acts'])]
    txt = (f"**Respuesta:** {plural(n, 'ruta excede', 'rutas exceden')} el plazo: "
           + lista_y(f"{r} en {e}" for r, e in zip(tabla['ruta'], tabla['_exceso'])) + '. ')
    if comunes:
        txt += ((f"{lista_y(comunes)} {'está' if len(comunes) == 1 else 'están'} en todas ellas. " if n > 1 else
                 f"Al ser la única, cualquiera de sus actividades ({lista_y(comunes)}) está en todas. ")
                + f"Si se acorta una sola, hay "
                f"que acortarla {m} {unidad}, el mayor exceso (con menos, la ruta que más excede seguiría por encima del "
                f"plazo; las rutas que no exceden no importan): ")
        txt += '; '.join((f"{a} pasaría de {int(d[a])} a {int(d[a]) - m}" if int(d[a]) > m else
                          f"{a} no puede (solo dura {int(d[a])})") for a in comunes) + '.'
    else:
        txt += 'Ninguna actividad está en todas ellas: habrá que acortar al menos dos actividades distintas.'
    return Markdown(txt)


def fines_gantt(datos):
    t = pd.DataFrame({'inicio': datos['start'].astype(int), 'duración': datos['duration'].astype(int),
                      'fin = inicio + duración': [f"{int(s)} + {int(d)} = {int(s + d)}" for s, d in zip(datos['start'], datos['duration'])]})
    t.index.name = 'actividad'
    return t


def respuesta_cadenas(cadenas):
    nombres = [' – '.join(c) for c in cadenas]
    if len(cadenas) == 1:
        return Markdown(f"**Respuesta:** hay una sola cadena de actividades ajustadas de la semana 0 al final: **{nombres[0]}**. "
                        "Toda ruta crítica es una cadena ajustada (en ella cada actividad empieza justo cuando acaba la "
                        "anterior) y el proyecto tiene al menos una, así que esta cadena es la ruta crítica, y es única.")
    return Markdown(f"**Respuesta:** hay {len(cadenas)} cadenas ajustadas de la semana 0 al final: {lista_y(nombres)}. "
                    "Al menos una es ruta crítica (toda ruta crítica es una cadena ajustada), pero con solo el Gantt no se "
                    "sabe cuáles: un eslabón puede ser una coincidencia de fechas entre actividades que no son "
                    "predecesoras. Sí puede haber más de una ruta crítica, si existen las prelaciones de varias cadenas.")


def respuesta_recuperado(datos, ajustadas, prelaciones, proyecto, calendario, cadenas):
    pred = prelaciones.immediate_predecessor
    G = nx.DiGraph([(p, a) for a, ps in pred.items() for p in ps])
    G.add_nodes_from(datos.index)
    iguales = int((calendario['inicio_mas_temprano'].astype(int) == datos['start'].astype(int)).sum())
    L = [f"Los inicios más tempranos calculados con el cuadro recuperado coinciden con los del Gantt en "
         f"{iguales} de {len(datos)} actividades: " + ('**es compatible**.' if iguales == len(datos) else '**no es compatible**.')]
    ocultas = [(p, a) for a in datos.index for p in sorted(pred[a]) if p not in ajustadas[a]]
    if ocultas:
        L.append('- Predecesoras reales que no son ajustadas (acaban antes de que empiece la sucesora; el Gantt no las '
                 'delata): ' + '; '.join(f"{p} → {a} ({p} acaba en {int(datos.loc[p, 'end'])}, {a} empieza en {int(datos.loc[a, 'start'])})" for p, a in ocultas) + '.')
    no_inm = [(p, a) for a in datos.index for p in ajustadas[a] if p not in pred[a] and p in nx.ancestors(G, a)]
    if no_inm:
        L.append('- Ajustadas que son antecesoras pero no predecesoras inmediatas: ' + '; '.join(f'{p} de {a}' for p, a in no_inm) + '.')
    coinc = [(p, a) for a in datos.index for p in ajustadas[a] if p not in nx.ancestors(G, a)]
    if coinc:
        L.append('- Ajustadas por coincidencia de fechas (no preceden a la actividad): ' + '; '.join(f'{p} de {a}' for p, a in coinc) + '.')
    if not (ocultas or no_inm or coinc):
        L.append('- Aquí las predecesoras ajustadas coinciden exactamente con las reales.')
    criticas = [' – '.join(ordena_ruta(r, calendario)) for r in proyecto.critical_path(calendario['duracion']).values()]
    nombres = [' – '.join(c) for c in cadenas]
    L.append(f"\n**Respuesta:** el cuadro recuperado {'reproduce' if iguales == len(datos) else 'no reproduce'} el Gantt. "
             f"{'La ruta crítica real es' if len(criticas) == 1 else 'Las rutas críticas reales son'} {lista_y(criticas)}"
             + (f"; de las cadenas deducidas, {lista_y([c for c in nombres if c not in criticas])} no lo "
                f"{'es' if len([c for c in nombres if c not in criticas]) == 1 else 'son'}." if any(c not in criticas for c in nombres) else
                ', la misma que se dedujo del Gantt.')
             + " El Gantt no determina el cuadro de prelaciones: oculta las prelaciones con holgura y puede sugerir "
             "otras por coincidencia de fechas.")
    return Markdown(L[0] + '\n\n' + '\n'.join(L[1:]))
'''


def _roy_ilustrativo(forzar=False):
    """Grafo Roy equivalente, como ilustración (la matriz de Zaderenko opera sobre el PERT)."""
    if ESTADO["grafo"] == "roy" and not forzar:
        return []
    return [md("A modo ilustrativo, este mismo proyecto como grafo Roy (actividades en los nodos); "
               "la matriz de Zaderenko se construye sobre los sucesos del grafo PERT."),
            *fig_grafo(durations="False", cap="Grafo Roy equivalente (ilustrativo)",
                       label="fig-royIlustrativo", tipo="roy")]



_FECHAS_NODOS = ("Para una actividad que va del nodo i al nodo j (con $t_i$ el tiempo temprano de i y $t^*_j$ el "
                 "tardío de j): el inicio más temprano es $ES = t_i$, el fin más temprano $EF = t_i + d$, el fin más "
                 "tardío $LF = t^*_j$ y el inicio más tardío $LS = t^*_j - d$; la holgura total es "
                 "$H_t = t^*_j - t_i - d$. Con los números de los nodos:")


def _base(k, rng):
    n = {1: 8, 2: 9, 3: 10}[(k - 1) % 3 + 1]
    info = P.buscar(rng, lambda i: 7 <= i["n_nodos"] <= 9 and 1 <= i["n_dummies"] <= 3 and i["n_rutas"] <= 9,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    cal = info["calendario"]
    no_criticas = [a for a in cal.index if cal.loc[a, "H_total"] > 0]
    criticas = [a for a in cal.index if cal.loc[a, "H_total"] == 0]
    cambio = no_criticas[int(rng.integers(0, len(no_criticas)))] if no_criticas else criticas[0]
    extra = max(1, int(cal.loc[cambio, "H_total"]) + int(rng.integers(-1, 3)))

    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta.")]
    pregs = ["Dibuje el grafo PERT y numere sus nodos.",
             "Construya la matriz de Zaderenko y calcule con ella los tiempos más tempranos y más tardíos "
             "de cada nodo. ¿Cuál es la duración del proyecto?",
             "Calcule el calendario del proyecto: inicio y fin más tempranos y más tardíos de cada actividad.",
             "Dibuje el diagrama de Gantt con las actividades en su inicio más temprano, con las flechas de "
             "prelación entre barras, e indique sobre él las holguras.",
             f"La actividad {cambio} va a durar {extra} días más de lo previsto. Recalcule la duración del "
             f"proyecto y explique el resultado a partir de la holgura de {cambio}."]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas.append(code(_AUX_ZADERENKO))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones expandido"\n#| label: tbl-prelacionesExpandido\n'
                       "prelaciones.display_immediate_linkage_matrix()"))
    celdas += fig_grafo(durations="False", cap="Grafo PERT con numeración de nodos", label="fig-pert")
    celdas.append(md("**Respuesta:** el grafo PERT de la figura. Los nodos se numeran de modo que cada arco va de un número "
                     "menor a uno mayor; cada actividad (y cada ficticia) es el arco entre estos nodos:"))
    celdas.append(code('#| tbl-cap: "Arco del grafo PERT de cada actividad"\n#| label: tbl-arcos\n'
                       "arcos_pert(proyecto).T"))
    celdas += _roy_ilustrativo()
    celdas += apartado(ap); ap += 1
    celdas.append(md("En la matriz de Zaderenko la casilla (i, j) contiene la duración de la actividad que va "
                     "del nodo i al nodo j (cero para las ficticias). El tiempo más temprano de un nodo es el "
                     "máximo, sobre su columna, de «tiempo temprano del origen más duración»; el más tardío es "
                     "el mínimo, sobre su fila, de «tiempo tardío del destino menos duración». Los tempranos "
                     "se calculan de izquierda a derecha y los tardíos de derecha a izquierda, partiendo del "
                     "tiempo temprano del último nodo."))
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(md("Cada tiempo temprano es el máximo, sobre los arcos que llegan al nodo, de «tiempo temprano del "
                     "origen + duración del arco»; cada tiempo tardío, el mínimo, sobre los arcos que salen, de «tiempo "
                     "tardío del destino − duración». Las ficticias (@) aportan duración 0. Las tablas recogen los "
                     "candidatos que se comparan en cada nodo."))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos: candidatos en cada nodo (de izquierda a derecha)"\n#| label: tbl-tempranosPaso\n'
                       "tiempos_tempranos, tiempos_tardios = detalle_tiempos_nodos(proyecto, datos_proyecto['duration'])\n"
                       "tiempos_tempranos"))
    celdas.append(code('#| tbl-cap: "Tiempos tardíos: candidatos en cada nodo (de derecha a izquierda)"\n#| label: tbl-tardiosPaso\n'
                       "tiempos_tardios"))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos y tardíos de los nodos"\n#| label: tbl-tiemposNodos\n'
                       "duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "tiempos_nodos(proyecto, datos_proyecto['duration']).T"))
    celdas.append(code("respuesta_duracion(proyecto, datos_proyecto['duration'])"))
    celdas += apartado(ap); ap += 1
    celdas.append(md(_FECHAS_NODOS))
    celdas.append(code('#| tbl-cap: "Fechas de cada actividad a partir de los tiempos de sus nodos"\n#| label: tbl-fechasPaso\n'
                       "fechas_desde_nodos(proyecto, datos_proyecto['duration'])"))
    celdas.append(md("Calendario resultante (ES/EF: inicio y fin más tempranos; LS/LF: inicio y fin más tardíos):"))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code("respuesta_calendario(calendario)"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos de los nodos y camino crítico", label="fig-pertTiempos")
    celdas += apartado(ap); ap += 1
    celdas.append(md("Las flechas unen el final de cada predecesora con el inicio de su sucesora; en rojo, las "
                     "prelaciones críticas (sin hueco entre barras críticas)."))
    celdas.append(code('#| fig-cap: "Diagrama de Gantt (inicios más tempranos) con flechas de prelación y holguras"\n#| label: fig-gantt\n'
                       "gantt_df, dibujo = proyecto.gantt(datos_proyecto, 'duration', holguras=True, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code("holg = calendario.index[calendario['H_total'] > 0]\n"
                       "Markdown('**Respuesta:** cada barra empieza en el inicio más temprano de su actividad y la columna de la derecha '\n"
                       "         'da su holgura total, lo que la barra puede desplazarse a la derecha sin alargar el proyecto: '\n"
                       "         + lista_y(f'{a}: {plural(int(calendario.loc[a, \"H_total\"]), \"día\")}' for a in holg)\n"
                       "         + f'. Las críticas ({lista_y(calendario.index[calendario[\"H_total\"] == 0])}) tienen holgura 0: sus barras van encadenadas, unidas por flechas rojas.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(code(f"holgura = int(calendario.loc['{cambio}', 'H_total'])\n"
                       "duraciones_nuevas = datos_proyecto['duration'].copy()\n"
                       f"duraciones_nuevas['{cambio}'] += {extra}\n"
                       "duracion_nueva = int(proyecto.duration(duraciones_nuevas))\n"
                       f"Markdown(f'**Respuesta:** la holgura total de {cambio} es H total = LS − ES = {{int(calendario.loc[\"{cambio}\", \"inicio_mas_tardio\"])}} − {{int(calendario.loc[\"{cambio}\", \"inicio_mas_temprano\"])}} = {{plural(holgura, \"día\")}}. '\n"
                       f"         + (f'El aumento de {extra} días no la supera ({extra} ≤ {{holgura}}): se absorbe en la holgura y **la duración del proyecto no cambia: {{duracion_nueva}} días**.'\n"
                       f"            if {extra} <= holgura else\n"
                       f"            f'El aumento de {extra} días la supera en {extra} − {{holgura}} = {{plural({extra} - holgura, \"día\")}}, que es lo que se alarga el proyecto: '\n"
                       f"            f'**{{duracion_proyecto}} + {{{extra} - holgura}} = {{duracion_nueva}} días**. {cambio} pasa a ser crítica.'))"))
    celdas.append(md("Calendario recalculado con la nueva duración:"))
    celdas.append(code('#| tbl-cap: "Calendario con la nueva duración"\n#| label: tbl-calendarioNuevo\n'
                       "proyecto.calendar(duraciones_nuevas)[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    sub = f"{n} actividades, {info['n_nodos']} nodos, {info['n_dummies']} ficticias"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _matriz_dada(rng):
    """Se da la matriz de Zaderenko rellena con las actividades y sus duraciones; hay que
    reconstruir el cuadro de prelaciones, el grafo y los tiempos."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 6 <= i["n_nodos"] <= 8 and 1 <= i["n_dummies"] <= 3,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, g = info["D"], info["proyecto"]
    nodos = list(g.nodes)
    celdas_matriz = {}
    for u, v, d in g.pert_graph.edges(data=True):
        act = d["activity"]
        celdas_matriz[(u, v)] = "ficticia" if act.startswith("@") else f"{act} ({int(D[act])})"
    lineas = ["matriz_enunciado = pd.DataFrame('', index=" + repr(nodos) + ", columns=" + repr(nodos) + ")"]
    for (u, v), txt in sorted(celdas_matriz.items()):
        lineas.append(f"matriz_enunciado.loc['{u}', '{v}'] = {txt!r}")
    lineas.append("matriz_enunciado.index.name = 'nodo'")
    lineas.append("matriz_enunciado")
    celdas = [md("# Enunciado"),
              md("De un proyecto sólo se conserva la matriz de Zaderenko sin calcular: en la casilla (i, j) figura la "
                 "actividad que va del nodo i al nodo j y, entre paréntesis, su duración en días; las ficticias "
                 "aparecen como tales. Los nodos están numerados de modo que todo arco va de un número menor a "
                 "uno mayor.")]
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko del enunciado"\n#| label: tbl-matrizEnunciado\n' + "\n".join(lineas)))
    celdas.append(preguntas([
        "Reconstruya el cuadro de prelaciones inmediatas del proyecto (las ficticias transmiten prelaciones pero no son actividades).",
        "Dibuje el grafo PERT y compruebe que reproduce la matriz.",
        "Calcule los tiempos más tempranos y más tardíos de los nodos y la duración del proyecto.",
        "Obtenga el calendario de las actividades y el camino crítico."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(_AUX_ZADERENKO))
    celdas += apartado(1)
    celdas.append(md("Cada casilla no vacía es un arco (i → j). Las predecesoras de una actividad que sale del "
                     "nodo i son las actividades que entran en i, más las que entran en los nodos desde los que "
                     "llega una ficticia a i (siguiendo las cadenas de ficticias hacia atrás). Actividad a actividad:"))
    celdas.append(code('#| tbl-cap: "De la matriz a las prelaciones: qué llega al nodo origen de cada actividad"\n#| label: tbl-deduccion\n'
                       "datos_proyecto, deduccion = deduce_prelaciones(matriz_enunciado)\n"
                       "deduccion"))
    celdas.append(md("**Respuesta:** el cuadro de prelaciones inmediatas reconstruido es:"))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones reconstruido"\n#| label: tbl-prelacionesReconstruido\n'
                       "datos_proyecto"))
    celdas += apartado(2)
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += fig_grafo(durations="False", cap="Grafo PERT reconstruido", label="fig-pert")
    celdas.append(code("compara_matriz(matriz_enunciado, proyecto)"))
    celdas += _roy_ilustrativo()
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko calculada"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(md("Cada tiempo temprano es el máximo, sobre los arcos que llegan al nodo, de «tiempo temprano del "
                     "origen + duración del arco»; cada tiempo tardío, el mínimo, sobre los arcos que salen, de «tiempo "
                     "tardío del destino − duración». Las ficticias (@) aportan duración 0. Las tablas recogen los "
                     "candidatos que se comparan en cada nodo."))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos: candidatos en cada nodo (de izquierda a derecha)"\n#| label: tbl-tempranosPaso\n'
                       "tiempos_tempranos, tiempos_tardios = detalle_tiempos_nodos(proyecto, datos_proyecto['duration'])\n"
                       "tiempos_tempranos"))
    celdas.append(code('#| tbl-cap: "Tiempos tardíos: candidatos en cada nodo (de derecha a izquierda)"\n#| label: tbl-tardiosPaso\n'
                       "tiempos_tardios"))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos y tardíos de los nodos"\n#| label: tbl-tiemposNodos\n'
                       "tiempos_nodos(proyecto, datos_proyecto['duration']).T"))
    celdas.append(code("respuesta_duracion(proyecto, datos_proyecto['duration'])"))
    celdas += apartado(4)
    celdas.append(md(_FECHAS_NODOS))
    celdas.append(code('#| tbl-cap: "Fechas de cada actividad a partir de los tiempos de sus nodos"\n#| label: tbl-fechasPaso\n'
                       "fechas_desde_nodos(proyecto, datos_proyecto['duration'])"))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "proyecto.calendar(datos_proyecto['duration'])[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code("respuesta_critico(proyecto, datos_proyecto['duration'])"))
    return celdas, f"{n} actividades, {info['n_nodos']} nodos; reconstrucción desde la matriz"


def _prediccion(rng):
    """Antes de calcular: ¿cambia el plazo si una actividad dura más? Tres casos y comprobación."""
    n = int(rng.choice([8, 9]))

    def acepta(i):
        cal = i["calendario"]
        return (1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1 and (cal["H_total"] >= 3).sum() >= 2
                and ((cal["H_total"] > 0) & (cal["H_total"] <= 2)).sum() >= 1)

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    critica = cal.index[cal["H_total"] == 0].tolist()
    holgada = cal.index[cal["H_total"] >= 3].tolist()
    justa = cal.index[(cal["H_total"] > 0) & (cal["H_total"] <= 2)].tolist()
    c1 = critica[int(rng.integers(0, len(critica)))]
    c2 = holgada[int(rng.integers(0, len(holgada)))]
    c3 = justa[int(rng.integers(0, len(justa)))]
    casos = [(c1, 2), (c2, 2), (c3, int(cal.loc[c3, "H_total"]) + 1)]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta. "
                 "Antes de hacer ningún cálculo numérico, conteste razonando sobre el grafo; después compruébelo.")]
    celdas.append(preguntas([
        "Dibuje el grafo PERT y, mirando sólo su forma, indique qué actividades parecen estar en la ruta más larga.",
        f"Prediga, sin calcular tiempos, si el plazo del proyecto cambia en cada uno de estos casos: "
        f"(a) {casos[0][0]} dura {casos[0][1]} días más; (b) {casos[1][0]} dura {casos[1][1]} días más; "
        f"(c) {casos[2][0]} dura {casos[2][1]} días más. Anote su predicción.",
        "Calcule con Zaderenko los tiempos de los nodos, el calendario y las holguras.",
        "Compruebe las tres predicciones recalculando la duración. ¿Qué dato del calendario permitía acertar sin recalcular?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas.append(code(_AUX_ZADERENKO))
    celdas += apartado(1)
    celdas += fig_grafo(durations="False", cap="Grafo PERT con numeración de nodos", label="fig-pert")
    celdas += _roy_ilustrativo()
    celdas.append(md("A ojo sólo puede intuirse: las rutas con más actividades o con actividades largas son las "
                     "candidatas. La forma del grafo no basta; hacen falta las duraciones."))
    celdas.append(code("respuesta_forma(proyecto, datos_proyecto['duration'])"))
    celdas += apartado(2)
    celdas.append(md("El plazo sólo cambia si la actividad está en la ruta más larga, o si el aumento es mayor que lo que "
                     "le falta a la ruta más larga que pasa por ella para igualar a la del proyecto (que es su holgura "
                     "total). Sin calcular los tiempos de los nodos puede razonarse con las sumas de las rutas:"))
    celdas.append(code('#| tbl-cap: "Rutas del grafo y su duración"\n#| label: tbl-rutas\n'
                       "explica_rutas(proyecto, datos_proyecto['duration'], unidad='días').drop(columns='duración')"))
    celdas.append(code("casos = " + repr(casos) + "\n"
                       "respuesta_prediccion(proyecto, datos_proyecto['duration'], casos)"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(md("Tiempos de los nodos con los candidatos que se comparan en cada uno (tempranos: máximo de «tiempo "
                     "temprano del origen + duración»; tardíos: mínimo de «tiempo tardío del destino − duración»):"))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos: candidatos en cada nodo (de izquierda a derecha)"\n#| label: tbl-tempranosPaso\n'
                       "tiempos_tempranos, tiempos_tardios = detalle_tiempos_nodos(proyecto, datos_proyecto['duration'])\n"
                       "tiempos_tempranos"))
    celdas.append(code('#| tbl-cap: "Tiempos tardíos: candidatos en cada nodo (de derecha a izquierda)"\n#| label: tbl-tardiosPaso\n'
                       "tiempos_tardios"))
    celdas.append(md(_FECHAS_NODOS))
    celdas.append(code('#| tbl-cap: "Fechas de cada actividad a partir de los tiempos de sus nodos"\n#| label: tbl-fechasPaso\n'
                       "fechas_desde_nodos(proyecto, datos_proyecto['duration'])"))
    celdas.append(code('#| tbl-cap: "Calendario y holguras"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code("respuesta_calendario(calendario)"))
    celdas += apartado(4)
    celdas.append(code("casos = " + repr(casos) + "\n"
                       "filas = []\n"
                       "for act, extra in casos:\n"
                       "    d2 = datos_proyecto['duration'].copy(); d2[act] += extra\n"
                       "    nueva = proyecto.duration(d2)\n"
                       "    filas.append({'actividad': act, 'días de más': extra, 'holgura total': int(calendario.loc[act, 'H_total']),\n"
                       "                  'plazo antes': duracion_proyecto, 'plazo después': nueva,\n"
                       "                  'cambia': 'sí' if nueva > duracion_proyecto else 'no'})\n"
                       "pd.DataFrame(filas).set_index('actividad')"))
    celdas.append(code("respuesta_comprobacion(calendario, int(duracion_proyecto), casos)"))
    return celdas, f"{n} actividades; predicción cualitativa sobre {c1}, {c2} y {c3}"



def _plazo_impuesto(rng):
    """El cliente impone un plazo menor que la duración: holguras negativas y qué rutas hay que acortar."""
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and i["segunda"] is not None and i["T"] - i["segunda"] <= 2,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, T, L = info["D"], info["calendario"], int(info["T"]), info["L"]
    k = int(rng.integers(2, 4))
    plazo = T - k
    df = P.tabla_base(info, duration=D)
    n_exceden = int((L > plazo).sum())
    celdas = [md("# Enunciado"),
              md(f"Considere el proyecto cuyo cuadro de prelaciones y duraciones (en semanas) se adjunta. El "
                 f"cliente exige que esté terminado en **{plazo} semanas**.")]
    celdas.append(preguntas([
        "Calcule con Zaderenko los tiempos de los nodos y el calendario del proyecto. ¿Cuál es su duración? "
        "¿Se cumple el plazo del cliente?",
        f"Recalcule los tiempos tardíos tomando como tiempo tardío del último nodo el plazo impuesto ({plazo}) "
        "en lugar de la duración del proyecto. ¿Qué actividades tienen ahora holgura total negativa y qué "
        "significa?",
        "¿Cuántas rutas del proyecto exceden el plazo y en cuánto cada una? ¿Hay alguna actividad que esté en "
        "todas ellas? ¿Cuántas semanas habría que acortarla para cumplir el plazo si fuera la única que se "
        "acorta?",
        "Si en lugar de acortar actividades se negociara el plazo, ¿cuál es el mínimo plazo que no obliga a "
        "acortar nada?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"plazo = {plazo}"))
    celdas.append(code(_AUX_ZADERENKO))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Matriz de Zaderenko"\n#| label: tbl-zaderenko\n'
                       "proyecto.zaderenko(durations=datos_proyecto['duration'])"))
    celdas.append(md("Tiempos de los nodos con los candidatos que se comparan en cada uno (tempranos: máximo de «tiempo "
                     "temprano del origen + duración»; tardíos: mínimo de «tiempo tardío del destino − duración»):"))
    celdas.append(code('#| tbl-cap: "Tiempos tempranos: candidatos en cada nodo (de izquierda a derecha)"\n#| label: tbl-tempranosPaso\n'
                       "tiempos_tempranos, tiempos_tardios = detalle_tiempos_nodos(proyecto, datos_proyecto['duration'])\n"
                       "tiempos_tempranos"))
    celdas.append(code('#| tbl-cap: "Tiempos tardíos: candidatos en cada nodo (de derecha a izquierda)"\n#| label: tbl-tardiosPaso\n'
                       "tiempos_tardios"))
    celdas.append(md(_FECHAS_NODOS))
    celdas.append(code('#| tbl-cap: "Fechas de cada actividad a partir de los tiempos de sus nodos"\n#| label: tbl-fechasPaso\n'
                       "fechas_desde_nodos(proyecto, datos_proyecto['duration'])"))
    celdas.append(code('#| tbl-cap: "Calendario del proyecto"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_proyecto = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code("Markdown(f'**Respuesta:** la duración del proyecto es el tiempo temprano del último nodo, **{duracion_proyecto} semanas**; '\n"
                       "         f'supera el plazo del cliente ({plazo}) en {duracion_proyecto} − {plazo} = {plural(duracion_proyecto - plazo, \"semana\")}: **no se cumple**.')"))
    celdas += apartado(2)
    celdas.append(md("Los tiempos tempranos no dependen del plazo. Los tardíos sí: si el último nodo debe "
                     "alcanzarse en la semana del plazo, todos los tiempos tardíos (y los inicios y fines más "
                     "tardíos de las actividades) se adelantan en la diferencia entre la duración y el plazo, y "
                     "la holgura total de cada actividad disminuye en esa misma cantidad. Tiempos de los nodos:"))
    celdas.append(code('#| tbl-cap: "Tiempos de los nodos con el plazo impuesto"\n#| label: tbl-nodosPlazo\n'
                       "retraso = int(duracion_proyecto - plazo)\n"
                       "nodos_con_plazo(proyecto, datos_proyecto['duration'], plazo)"))
    celdas.append(md("Fechas de las actividades con esos tiempos tardíos (LS y LF bajan en la diferencia; H total = LS − ES):"))
    celdas.append(code('#| tbl-cap: "Calendario con el plazo impuesto"\n#| label: tbl-calendarioPlazo\n'
                       "con_plazo = calendario[['inicio_mas_temprano', 'fin_mas_temprano']].copy()\n"
                       "con_plazo['inicio_mas_tardio'] = calendario['inicio_mas_tardio'] - retraso\n"
                       "con_plazo['fin_mas_tardio'] = calendario['fin_mas_tardio'] - retraso\n"
                       "con_plazo['H_total'] = con_plazo['inicio_mas_tardio'] - con_plazo['inicio_mas_temprano']\n"
                       "con_plazo.astype(int).rename(columns=_COLS_CAL).rename_axis('actividad')"))
    celdas.append(code("respuesta_negativas(con_plazo, retraso)"))
    celdas += apartado(3)
    celdas.append(md("Rutas cuya duración (suma de las duraciones de sus actividades, en orden de ejecución) supera el plazo:"))
    celdas.append(code('#| tbl-cap: "Rutas que exceden el plazo"\n#| label: tbl-rutasExceden\n'
                       "exceden = rutas_exceden(proyecto, datos_proyecto['duration'], plazo)\n"
                       "exceden.drop(columns=['_exceso', '_acts'])"))
    celdas.append(code("respuesta_exceden(exceden, datos_proyecto['duration'])"))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'**Respuesta:** {duracion_proyecto} semanas, la duración del proyecto. Con ese plazo el tiempo tardío del último '\n"
                       "         f'nodo coincide con su tiempo temprano, las críticas tienen holgura total 0 y las demás positiva; con cualquier '\n"
                       "         f'plazo menor habría holguras negativas y habría que acortar alguna actividad.')"))
    return celdas, f"{n} actividades; duración {T}, plazo impuesto {plazo} ({n_exceden} rutas lo exceden)"


def _gantt_dado(rng):
    """Se da el Gantt (inicio y duración de cada actividad) sin prelaciones: deducir lo que se pueda."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and i["n_criticas"] == 1 and (i["calendario"]["H_total"] > 0).sum() >= 2,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal, preds = info["D"], info["calendario"], info["preds"]
    gantt = pd.DataFrame({"start": cal["inicio_mas_temprano"].astype(int), "duration": D})
    gantt.index.name = "activity"
    celdas = [md("# Enunciado"),
              md("De un proyecto sólo se conserva su programación de inicios más tempranos: la semana en que "
                 "empieza cada actividad y su duración (cuadro adjunto). Se sabe que el calendario se calculó "
                 "correctamente a partir de un cuadro de prelaciones que se ha perdido.")]
    celdas.append(preguntas([
        "Dibuje el diagrama de Gantt y calcule el fin más temprano de cada actividad y la duración del proyecto.",
        "¿Qué actividades son iniciales? Para cada una de las demás, ¿qué actividades *pueden* ser "
        "predecesoras inmediatas suyas (terminan no más tarde de su inicio) y cuáles son las «ajustadas» "
        "(terminan exactamente cuando ella empieza)? ¿Por qué toda actividad no inicial tiene al menos una "
        "predecesora ajustada?",
        "Encadenando actividades ajustadas desde la semana 0 hasta el final, identifique una ruta crítica. "
        "¿Puede haber más de una?",
        "Se recupera el cuadro de prelaciones original. Compruebe que es compatible con el Gantt y compare "
        "sus prelaciones con las predecesoras ajustadas deducidas."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(gantt)))
    celdas.append(md("# Resolución"))
    celdas.append(code(_AUX_ZADERENKO))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Fin más temprano de cada actividad"\n#| label: tbl-fines\n'
                       "datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['end'] = datos_proyecto['start'] + datos_proyecto['duration']\n"
                       "duracion_proyecto = int(datos_proyecto['end'].max())\n"
                       "fines_gantt(datos_proyecto)"))
    celdas.append(code('#| fig-cap: "Diagrama de Gantt de la programación conservada"\n#| label: fig-gantt\n'
                       "# Gantt de pmule con las barras del enunciado (sin flechas: las prelaciones aún no se conocen)\n"
                       "pinta_tikz(gantt_tramos_tikz({a: [(int(f['start']), int(f['duration']), int(f['duration']))] for a, f in datos_proyecto.iterrows()},\n"
                       "                             horizonte=duracion_proyecto))"))
    celdas.append(code("Markdown(f'**Respuesta:** los fines más tempranos son los de la tabla; la duración del proyecto es el mayor de ellos: '\n"
                       "         f'máx({\", \".join(f\"{a}: {int(e)}\" for a, e in datos_proyecto[\"end\"].items())}) = **{duracion_proyecto} semanas**.')"))
    celdas += apartado(2)
    celdas.append(md("Una actividad que empieza en la semana 0 es inicial. Si $P$ terminara después de que "
                     "empiece $A$, $P$ no podría ser predecesora de $A$. En un calendario de inicios más "
                     "tempranos, el inicio de cada actividad no inicial es el *máximo* de los fines de sus "
                     "predecesoras, luego alguna de ellas termina exactamente cuando la actividad empieza."))
    celdas.append(code("posibles, ajustadas = {}, {}\n"
                       "for a, fila in datos_proyecto.iterrows():\n"
                       "    posibles[a] = sorted(p for p, f in datos_proyecto.iterrows() if p != a and f['end'] <= fila['start'])\n"
                       "    ajustadas[a] = sorted(p for p in posibles[a] if datos_proyecto.loc[p, 'end'] == fila['start'])\n"
                       "iniciales = datos_proyecto.index[datos_proyecto['start'] == 0].tolist()\n"
                       "pd.DataFrame({'inicio': datos_proyecto['start'].astype(int),\n"
                       "              'posibles predecesoras (fin ≤ inicio)': {a: ', '.join(f\"{p} ({int(datos_proyecto.loc[p, 'end'])})\" for p in v) or '—' for a, v in posibles.items()},\n"
                       "              'ajustadas (fin = inicio)': {a: ', '.join(v) or '—' for a, v in ajustadas.items()}}).rename_axis('actividad')"))
    celdas.append(code("Markdown(f'**Respuesta:** son iniciales {lista_y(iniciales)} (empiezan en la semana 0). Las posibles predecesoras y las '\n"
                       "         'ajustadas de cada una de las demás son las de la tabla (entre paréntesis, la semana en que acaba cada candidata). '\n"
                       "         'Toda actividad no inicial tiene al menos una ajustada porque su inicio más temprano es el máximo de los fines '\n"
                       "         'de sus predecesoras: la que da ese máximo acaba justo cuando ella empieza.')"))
    celdas += apartado(3)
    celdas.append(code("finales = datos_proyecto.index[datos_proyecto['end'] == duracion_proyecto].tolist()\n"
                       "def cadenas(a):\n"
                       "    if datos_proyecto.loc[a, 'start'] == 0:\n"
                       "        return [[a]]\n"
                       "    return [c + [a] for p in ajustadas[a] for c in cadenas(p)]\n"
                       "rutas_criticas_deducidas = [c for f in finales for c in cadenas(f)]\n"
                       "display(Markdown('Cadenas de actividades ajustadas desde la semana 0 hasta el final (cada una suma la duración del proyecto):\\n\\n'\n"
                       "                 + '\\n'.join(f\"- {' – '.join(c)}: {' + '.join(str(int(datos_proyecto.loc[x, 'duration'])) for x in c)} = {duracion_proyecto}\" for c in rutas_criticas_deducidas)))\n"
                       "respuesta_cadenas(rutas_criticas_deducidas)"))
    celdas += apartado(4)
    tabla = P.tabla_base(info, duration=D)
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones recuperado"\n#| label: tbl-prelacionesRecuperado\n'
                       + literal_dataframe(tabla, nombre="prelaciones_recuperadas")))
    celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(prelaciones_recuperadas, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "calendario = proyecto.calendar(prelaciones_recuperadas['duration'])\n"
                       "compatible = (calendario['inicio_mas_temprano'] == datos_proyecto['start']).all()\n"
                       "comparacion = pd.DataFrame({'predecesoras reales': {a: ', '.join(sorted(ps)) or '—' for a, ps in prelaciones.immediate_predecessor.items()},\n"
                       "                            'ajustadas deducidas': {a: ', '.join(v) or '—' for a, v in ajustadas.items()}})\n"
                       "G = nx.DiGraph([(p, a) for a, ps in prelaciones.immediate_predecessor.items() for p in ps]); G.add_nodes_from(comparacion.index)\n"
                       "comparacion['ajustadas que son antecesoras'] = [', '.join(sorted(set(ajustadas[a]) & nx.ancestors(G, a))) or '—' for a in comparacion.index]\n"
                       "comparacion['ajustadas por coincidencia'] = [', '.join(sorted(set(ajustadas[a]) - nx.ancestors(G, a))) or '—' for a in comparacion.index]\n"
                       "display(comparacion.rename_axis('actividad'))\n"
                       "respuesta_recuperado(datos_proyecto, ajustadas, prelaciones, proyecto, calendario, rutas_criticas_deducidas)"))
    celdas += fig_grafo(durations="prelaciones_recuperadas['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert", tipo="pert")
    celdas += _roy_ilustrativo(forzar=True)
    return celdas, f"{n} actividades; sólo se conoce el Gantt de inicios tempranos"


VARIANTES = [
    Variante("zaderenko", 1, "matriz de Zaderenko, tiempos de los nodos, calendario, Gantt y efecto de un retraso",
             lambda rng: _base(1, rng), grafo="pert"),
    Variante("zaderenko-b", 1, "ídem con un proyecto mayor", lambda rng: _base(2, rng), grafo="pert"),
    Variante("zaderenko-c", 2, "ídem con diez actividades", lambda rng: _base(3, rng), grafo="pert"),
    Variante("matriz-dada", 3, "inverso: de la matriz de Zaderenko rellena al cuadro de prelaciones, el grafo y los tiempos",
             _matriz_dada, grafo="pert"),
    Variante("prediccion", 2, "predecir sin calcular si un retraso cambia el plazo y comprobarlo con las holguras",
             _prediccion, grafo="pert"),
    Variante("plazo-impuesto", 2, "plazo menor que la duración: tardíos desde el plazo, holguras negativas, rutas que lo exceden y actividad común",
             _plazo_impuesto, grafo="pert"),
    Variante("gantt-dado", 3, "sólo se conoce el Gantt: deducir iniciales, predecesoras posibles y ajustadas, y una ruta crítica",
             _gantt_dado),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
