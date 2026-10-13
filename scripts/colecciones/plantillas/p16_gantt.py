"""Colección 16 — Gantt y grafos: del PERT o del Roy al diagrama de Gantt y, al revés, del
Gantt (con flechas de prelación) al cuadro de prelaciones, al Roy y al PERT."""
import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="16_gantt_grafos", prefijo="Gantt",
            titulo="Gantt y grafos: ida y vuelta",
            descripcion="Construir el diagrama de Gantt a partir del grafo PERT o del Roy (leyendo el cuadro "
                        "de prelaciones del dibujo) y reconstruir el cuadro, el Roy y el PERT a partir de un "
                        "Gantt con flechas de prelación.")


def _proyecto(rng, n=None):
    n = n or int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 3 and 4 <= i["n_rutas"] <= 8 and i["n_criticas"] == 1
                    and (i["calendario"]["H_total"] > 0).sum() >= 2 and (i["calendario"]["H_libre"] > 0).sum() >= 1,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    return info


def _datos_ocultos(df, mostrar):
    """Celda con la tabla completa (predecesoras incluidas) pero que sólo muestra las columnas `mostrar`."""
    lineas = literal_dataframe(df).splitlines()
    lineas[-1] = f"datos_enunciado[{mostrar!r}]" if mostrar else "# (el cuadro completo no se muestra: el enunciado es el diagrama)"
    return "\n".join(lineas)


def _celda_gantt(nombre="proyecto", datos="datos_enunciado", holguras=False, label="fig-gantt", cap="Diagrama de Gantt"):
    return code(f'#| fig-cap: "{cap}"\n#| label: {label}\n'
                f"_g, _dibujo = {nombre}.gantt({datos}, 'duration', holguras={holguras}, tikz=True, params=dict(inner_text=None, **GANTT_PARAMS))\n"
                "pinta_tikz(_dibujo)")


def _grafo_a_gantt(rng, tipo):
    info = _proyecto(rng)
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    nombre = "PERT" if tipo == "pert" else "Roy"
    celdas = [md("# Enunciado"),
              md(f"De un proyecto se dispone del grafo {nombre} (sin tiempos) y de la duración de cada actividad "
                 "(en semanas), que se adjuntan.")]
    celdas.append(preguntas([
        f"Lea del grafo el cuadro de prelaciones inmediatas." + (" ¿Qué prelaciones aporta cada actividad ficticia?" if tipo == "pert" else ""),
        "Calcule el calendario (inicios y fines más tempranos y más tardíos, holguras) y construya el diagrama de "
        "Gantt de inicios más tempranos, dibujando una flecha por cada prelación inmediata.",
        "Marque el camino crítico sobre el Gantt. ¿Qué actividades tienen hueco entre su fin y el inicio de "
        "todas sus sucesoras, y cuánto? Relaciónelo con la holgura libre.",
        "Dibuje el Gantt de inicios más tardíos. ¿Cuánto se ha desplazado cada actividad respecto al anterior? "
        "Relaciónelo con la holgura total."]))
    celdas.append(code('#| tbl-cap: "Duración de las actividades"\n#| label: tbl-datosEnunciado\n'
                       + _datos_ocultos(df, ["duration"])))
    celdas.append(code("_proyecto = PredecessorTable.from_dataframe_of_strings(datos_enunciado, activity='activity', predecessor='predecessor').create_project()"))
    celdas += fig_grafo(durations="False", cap=f"Grafo {nombre} del proyecto", label="fig-grafoEnunciado", obj="_proyecto")
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    if tipo == "pert":
        celdas.append(md("Cada arco es una actividad: sus predecesoras inmediatas son los arcos que llegan a su "
                         "nodo inicial. Si llega una ficticia, se sigue hacia atrás hasta las actividades reales "
                         "que la originan: la ficticia «transporta» esas prelaciones sin ser una actividad."))
    else:
        celdas.append(md("En el Roy cada nodo es una actividad y cada arco una prelación inmediata: las "
                         "predecesoras de una actividad son los nodos de los que recibe arcos. No hay ficticias "
                         "que interpretar."))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "datos_proyecto[['predecessor', 'duration']]"))
    if tipo == "pert":
        celdas.append(code("for d in proyecto.dummies:\n"
                           "    i, j = [(u, v) for u, v, dd in proyecto.pert_graph.edges(data=True) if dd['activity'] == d][0]\n"
                           "    origen = [dd['activity'] for _, _, dd in proyecto.pert_graph.in_edges(i, data=True)]\n"
                           "    destino = [dd['activity'] for _, _, dd in proyecto.pert_graph.out_edges(j, data=True)]\n"
                           "    print(f'{d}: lleva la prelación de {\", \".join(origen)} hacia {\", \".join(destino)}')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Calendario y holguras"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas.append(code('#| tbl-cap: "Cálculo paso a paso de las fechas"\n#| label: tbl-fechasPaso\n'
                       "detalle_holguras(calendario, prelaciones)[['ES = máx EF pred.', 'EF = ES + d', 'LF = mín LS suc.', 'LS = LF − d']]"))
    celdas.append(md("Cada barra empieza en el inicio más temprano y dura lo que la actividad; las flechas unen el "
                     "fin de cada predecesora con el inicio de su sucesora (en rojo, entre actividades críticas)."))
    celdas.append(_celda_gantt("proyecto", "datos_proyecto", holguras=True, label="fig-ganttTemprano", cap="Gantt de inicios más tempranos (última columna: holgura total)"))
    celdas += apartado(3)
    celdas.append(code("criticas = calendario.index[calendario['H_total'] == 0].tolist()\n"
                       "con_hueco = calendario[calendario['H_libre'] > 0]\n"
                       "Markdown(f'''Camino crítico (flechas rojas, barras sin hueco): {\", \".join(criticas)}. "
                       "Tienen hueco hasta todas sus sucesoras: {\", \".join(f\"{a} ({int(h)} semanas)\" for a, h in con_hueco[\"H_libre\"].items())}; "
                       "ese hueco es exactamente la holgura libre (mín ES de las sucesoras − EF).''')"))
    celdas += apartado(4)
    celdas.append(md("En el Gantt tardío cada barra empieza en su inicio más tardío. No basta con retrasar "
                     "cada actividad su holgura total (la holgura de una cadena es compartida: si la gasta la "
                     "predecesora ya no la tiene la sucesora); lo que se hace es insertar delante de cada "
                     "actividad una espera 💤 igual a su inicio más tardío menos el fin más tardío de sus "
                     "predecesoras, y volver a calcular."))
    celdas.append(code("filas = {}\n"
                       "for a, fila in datos_proyecto.iterrows():\n"
                       "    ps = sorted(prelaciones.immediate_predecessor[a])\n"
                       "    espera = int(calendario.loc[a, 'inicio_mas_tardio'] - (calendario.loc[ps, 'fin_mas_tardio'].max() if ps else 0))\n"
                       "    if espera > 0:\n"
                       "        filas['💤' + a] = {'predecessor': fila['predecessor'], 'duration': espera}\n"
                       "        filas[a] = {'predecessor': '💤' + a, 'duration': int(fila['duration'])}\n"
                       "    else:\n"
                       "        filas[a] = {'predecessor': fila['predecessor'], 'duration': int(fila['duration'])}\n"
                       "datos_tardio = pd.DataFrame(filas).T; datos_tardio.index.name = 'activity'; datos_tardio['duration'] = datos_tardio['duration'].astype(int)\n"
                       "proyecto_tardio = PredecessorTable.from_dataframe_of_strings(datos_tardio, activity='activity', predecessor='predecessor').create_project()\n"
                       "esperas = datos_tardio.loc[datos_tardio.index.str.startswith('💤'), 'duration']\n"
                       "desplazamiento = (proyecto_tardio.calendar(datos_tardio['duration']).loc[datos_proyecto.index, 'inicio_mas_temprano'] - calendario['inicio_mas_temprano'])\n"
                       "pd.DataFrame({'espera insertada': esperas.rename(index=lambda x: x[1:]).reindex(datos_proyecto.index).fillna(0).astype(int),\n"
                       "              'desplazamiento respecto al Gantt temprano': desplazamiento.astype(int), 'H_total': calendario['H_total']}).T"))
    celdas.append(_celda_gantt("proyecto_tardio", "datos_tardio", holguras=False, label="fig-ganttTardio", cap="Gantt de inicios más tardíos (las filas 💤 son las esperas)"))
    celdas.append(md("El desplazamiento de cada barra respecto al Gantt temprano es su holgura total (LS − ES), "
                     "pero la espera insertada es menor cuando una predecesora ya ha consumido parte de esa "
                     "holgura. Las críticas no se mueven y el fin del proyecto no cambia; en el Gantt tardío no "
                     "queda hueco entre ninguna actividad y sus sucesoras: toda la holgura está gastada."))
    return celdas, f"{len(D)} actividades, {info['n_dummies']} ficticias; del {nombre} al Gantt"


def _gantt_a_grafo(rng, tipo):
    info = _proyecto(rng)
    D, cal = info["D"], info["calendario"]
    df = P.tabla_base(info, duration=D)
    nombre = "PERT" if tipo == "pert" else "Roy"
    celdas = [md("# Enunciado"),
              md("De un proyecto sólo se conserva el diagrama de Gantt de inicios más tempranos que se adjunta "
                 "(semanas). Cada flecha une el fin de una actividad con el inicio de otra que no puede empezar "
                 "antes (prelación inmediata); las flechas rojas unen actividades críticas.")]
    celdas.append(preguntas([
        "Reconstruya el cuadro de prelaciones inmediatas y las duraciones a partir del diagrama. ¿Qué "
        "actividades son iniciales y cuáles finales? ¿Cómo se reconocen?",
        f"Dibuje el grafo {nombre}" + (" con el mínimo número de actividades ficticias. ¿Por qué necesita ficticias "
                                        "el PERT si el Gantt no las necesita?" if tipo == "pert" else
                                        ". ¿Qué información del Gantt no aparece en el Roy y cuál aparece en los dos?"),
        "Calcule el calendario con la librería y compruebe que reproduce el Gantt (inicios más tempranos) y "
        "el camino crítico marcado en rojo.",
        "Lea sobre el Gantt la holgura libre de cada actividad no crítica (hueco hasta su primera sucesora) y "
        "compárela con la calculada. ¿Se puede leer también la holgura total? ¿Qué haría falta?"]))
    celdas.append(code(_datos_ocultos(df, None)))
    celdas.append(code("_proyecto = PredecessorTable.from_dataframe_of_strings(datos_enunciado, activity='activity', predecessor='predecessor').create_project()"))
    celdas.append(_celda_gantt("_proyecto", "datos_enunciado", holguras=False, label="fig-ganttEnunciado", cap="Diagrama de Gantt del proyecto"))
    celdas.append(md("# Resolución"))
    celdas += apartado(1)
    celdas.append(md("La duración de cada actividad es la longitud de su barra. Sus predecesoras inmediatas son "
                     "las actividades de las que recibe una flecha. Las iniciales no reciben ninguna flecha (y "
                     "empiezan en la semana 0); las finales no emiten ninguna."))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "print('Iniciales:', ', '.join(sorted(prelaciones.start_activities)), '  Finales:', ', '.join(sorted(prelaciones.end_activities)))\n"
                       "datos_proyecto[['predecessor', 'duration']]"))
    celdas += apartado(2)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap=f"Grafo {nombre} reconstruido, con tiempos y camino crítico", label="fig-grafo")
    if tipo == "pert":
        celdas.append(code("Markdown(f'''El PERT representa las actividades como arcos, y un arco sólo puede salir de un nodo: cuando dos actividades comparten "
                           "*algunas* predecesoras pero no todas, hace falta una ficticia para no inventar prelaciones. El Gantt no tiene ese problema porque "
                           "dibuja cada prelación como una flecha independiente. Este grafo necesita {len(proyecto.dummies)} ficticia(s).''')"))
    else:
        celdas.append(md("El Roy y el Gantt contienen la misma información estructural (actividades y prelaciones "
                         "inmediatas) y, si se anotan los tiempos, también las fechas. El Gantt añade la escala "
                         "temporal (se *ve* la duración y los huecos); el Roy hace visibles las rutas y permite el "
                         "cálculo nodo a nodo sin dibujar a escala."))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Calendario calculado"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario"))
    celdas.append(_celda_gantt("proyecto", "datos_proyecto", holguras=True, label="fig-ganttCalculado", cap="Gantt calculado (coincide con el del enunciado; última columna: holgura total)"))
    celdas += apartado(4)
    celdas.append(code("suc = {a: [b for b, ps in prelaciones.immediate_predecessor.items() if a in ps] for a in calendario.index}\n"
                       "fin = int(calendario['fin_mas_temprano'].max())\n"
                       "lectura = pd.DataFrame({\n"
                       "    'EF': calendario['fin_mas_temprano'],\n"
                       "    'mín ES de sus sucesoras (o fin)': [int(calendario.loc[suc[a], 'inicio_mas_temprano'].min()) if suc[a] else fin for a in calendario.index],\n"
                       "    'H_libre leída = hueco': [int((calendario.loc[suc[a], 'inicio_mas_temprano'].min() if suc[a] else fin) - calendario.loc[a, 'fin_mas_temprano']) for a in calendario.index],\n"
                       "    'H_libre calculada': calendario['H_libre'], 'H_total calculada': calendario['H_total']})\n"
                       "lectura"))
    celdas.append(md("La holgura libre es el hueco visible entre el fin de la barra y el inicio de la sucesora más "
                     "próxima. La holgura total no se ve en un Gantt de inicios tempranos: requiere saber cuánto "
                     "podrían retrasarse *también* las sucesoras, es decir, el paso hacia atrás (o el Gantt de "
                     "inicios tardíos). Sólo coinciden cuando la sucesora es crítica."))
    return celdas, f"{len(D)} actividades, {info['n_dummies']} ficticias en PERT; del Gantt al {nombre}"


VARIANTES = [
    Variante("pert-a-gantt", 1, "del PERT sin tiempos (más duraciones) al cuadro, el calendario y los Gantt temprano y tardío",
             lambda rng: _grafo_a_gantt(rng, "pert"), grafo="pert"),
    Variante("roy-a-gantt", 1, "del Roy al cuadro, el calendario y los Gantt temprano y tardío; holgura libre como hueco, total como desplazamiento",
             lambda rng: _grafo_a_gantt(rng, "roy"), grafo="roy"),
    Variante("gantt-a-roy", 2, "del Gantt con flechas al cuadro de prelaciones y al Roy; qué holguras se leen en el Gantt",
             lambda rng: _gantt_a_grafo(rng, "roy"), grafo="roy"),
    Variante("gantt-a-pert", 2, "del Gantt con flechas al cuadro y al PERT con mínimo de ficticias; por qué el Gantt no las necesita",
             lambda rng: _gantt_a_grafo(rng, "pert"), grafo="pert"),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
