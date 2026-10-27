"""Colección 06 — Matriz de caminos: enumerar rutas, construir la matriz,
calcular la duración como máximo de P·d, sensibilidad, reconstruir el grafo
a partir de las rutas y rutas dominantes vía SVD."""
import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX, grafo_nombre, ESTADO)

META = dict(slug="06_matriz_caminos", prefijo="Caminos",
            titulo="Matriz de caminos",
            descripcion="Enumerar las rutas, construir la matriz de caminos y usarla para calcular la "
                        "duración del proyecto; reconstruir el grafo a partir de las rutas; rutas dominantes "
                        "mediante la SVD.",
            n_defecto=3)


def _base(k, rng):
    inverso = (k % 3 == 0)
    n = int(rng.choice([8, 9]))
    info = P.buscar(rng, lambda i: 5 <= i["n_rutas"] <= 8 and 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    df = P.tabla_base(info, duration=D)
    rutas = info["rutas"]

    celdas = [md("# Enunciado")]
    if inverso:
        celdas.append(md("De un proyecto sólo se conocen sus rutas entre el nodo inicial y el final y las "
                         "duraciones de sus actividades (en días)."))
        pregs = ["Construya la matriz de caminos del proyecto.",
                 "Calcule con ella la duración del proyecto e identifique la ruta crítica.",
                 "Reconstruya el cuadro de prelaciones inmediatas y dibuje los grafos PERT y Roy. Compruebe "
                 "que sus rutas son las del enunciado.",
                 "Calcule la sensibilidad de la duración del proyecto a cada actividad: ¿en cuánto aumenta "
                 "la duración si una actividad dura un día más?",
                 "Calcule la descomposición en valores singulares de la matriz de caminos y ordene rutas y "
                 "actividades según el primer vector singular. Compare con la ruta crítica."]
    else:
        celdas.append(md("Considere el proyecto cuyo cuadro de prelaciones y duraciones (en días) se adjunta."))
        pregs = [f"Dibuje el grafo {grafo_nombre()} y enumere todas sus rutas entre el nodo inicial y el final.",
                 "Construya la matriz de caminos del proyecto (una fila por ruta, una columna por actividad).",
                 "Calcule la duración de cada ruta como producto de la matriz de caminos por el vector de "
                 "duraciones, y la duración del proyecto como el máximo. Identifique la ruta crítica.",
                 "Calcule la sensibilidad de la duración del proyecto a cada actividad: ¿en cuánto aumenta "
                 "la duración si una actividad dura un día más? ¿Y si dura un día menos?"]
    celdas.append(preguntas(pregs))
    if inverso:
        celdas.append(code('#| tbl-cap: "Rutas del proyecto"\n#| label: tbl-datosRutas\n'
                           + "rutas_enunciado = {\n"
                           + "".join(f"    {r!r}: {acts!r},\n" for r, acts in rutas.items())
                           + "}\n"
                           "pd.Series({r: ' - '.join(a) for r, a in rutas_enunciado.items()}).to_frame('actividades')"))
        celdas.append(code('#| tbl-cap: "Duración de las actividades"\n#| label: tbl-datosDuraciones\n'
                           + literal_dataframe(df[["duration"]])))
    else:
        celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                           + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    ap = 1
    if inverso:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La matriz de caminos tiene un 1 en la fila de una ruta y la columna de una actividad "
                         "cuando la actividad pertenece a la ruta."))
        celdas.append(code("actividades = sorted(set(a for acts in rutas_enunciado.values() for a in acts))\n"
                           "path_matrix = pd.DataFrame(0, index=list(rutas_enunciado), columns=actividades)\n"
                           "for ruta, acts in rutas_enunciado.items():\n"
                           "    path_matrix.loc[ruta, acts] = 1\n"
                           "from pmule import matriz_caminos_tikz\n"
                           "matriz_caminos_tikz(path_matrix)"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("duraciones = datos_enunciado['duration'].reindex(actividades)\n"
                           "duraciones_rutas = path_matrix @ duraciones\n"
                           "from pmule import display_producto_rutas\n"
                           "display_producto_rutas(path_matrix, duraciones, duraciones_rutas)"))
        celdas.append(md("Cada fila del producto es la suma de las duraciones de las actividades de esa ruta:"))
        celdas.append(code('#| tbl-cap: "Duración de cada ruta, sumando sus actividades"\n#| label: tbl-rutasPaso-a\n'
                           "pd.DataFrame({r: {'ruta': ' – '.join(acts),\n"
                           "                  'duración (días) = suma': ' + '.join(str(int(duraciones[a])) for a in acts) + f' = {int(duraciones_rutas[r])}'}\n"
                           "              for r, acts in rutas_enunciado.items()}).T"))
        celdas.append(code("duracion_proyecto = int(duraciones_rutas.max())\n"
                           "criticas = [r for r in duraciones_rutas.index if duraciones_rutas[r] == duracion_proyecto]\n"
                           "ruta_critica = criticas[0]\n"
                           "Markdown(f'La duración del proyecto es la mayor de las duraciones de las rutas: T = máx('\n"
                           "         f'{\", \".join(str(int(x)) for x in duraciones_rutas)}) = **{duracion_proyecto} días**.\\n\\n'\n"
                           "         f'**Respuesta:** el proyecto dura {duracion_proyecto} días; '\n"
                           "         + ('; '.join(f'{r}: {\" – \".join(rutas_enunciado[r])}' for r in criticas))\n"
                           "         + (' es la ruta crítica.' if len(criticas) == 1 else ' son rutas críticas.'))"))
        celdas += apartado(ap); ap += 1
        celdas.append(md("Las predecesoras inmediatas de una actividad son las que la preceden "
                         "inmediatamente en alguna ruta; las rutas son todas las secuencias posibles, así que "
                         "el conjunto de parejas consecutivas de todas las rutas es exactamente el cuadro de "
                         "prelaciones inmediatas. Las actividades que abren alguna ruta son iniciales."))
        celdas.append(code("prelaciones_dict = {a: set() for a in actividades}\n"
                           "for acts in rutas_enunciado.values():\n"
                           "    for anterior, posterior in zip(acts[:-1], acts[1:]):\n"
                           "        prelaciones_dict[posterior].add(anterior)\n"
                           "datos_proyecto = pd.DataFrame({'predecessor': pd.Series({a: ','.join(sorted(p)) or '---' for a, p in prelaciones_dict.items()}),\n"
                           "                               'duration': duraciones})\n"
                           "datos_proyecto.index.name = 'activity'\n"
                           "datos_proyecto"))
        celdas.append(code("prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                           "proyecto = prelaciones.create_project()"))
        celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT reconstruido, con tiempos y ruta crítica", label="fig-pert", tipo="pert")
        celdas.append(code('#| fig-cap: "Grafo Roy reconstruido"\n#| label: fig-roy\n'
                           "proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)"))
        celdas.append(code("rutas_grafo = {' - '.join(a) for a in proyecto.paths(dummies=False).values()}\n"
                           "rutas_dadas = {' - '.join(a) for a in rutas_enunciado.values()}\n"
                           "Markdown('**Respuesta:** el cuadro de prelaciones reconstruido es el de la tabla anterior, y las rutas de los grafos '\n"
                           "         + ('coinciden con las del enunciado.' if rutas_grafo == rutas_dadas else 'NO coinciden con las del enunciado.'))"))
    else:
        celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                           "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                           "proyecto = prelaciones.create_project()"))
        celdas += apartado(ap); ap += 1
        celdas += fig_grafo(durations="False",
                            cap=("Grafo PERT con numeración de nodos" if ESTADO["grafo"] != "roy" else "Grafo Roy"),
                            label="fig-pert")
        if ESTADO["grafo"] == "roy":
            celdas.append(md("Las rutas se enumeran recorriendo el grafo desde el nodo de inicio: en cada nodo se "
                             "abre una rama por cada arco saliente."))
        else:
            celdas.append(md("Las rutas se enumeran recorriendo el grafo desde el nodo inicial: en cada nodo se "
                             "abre una rama por cada arco saliente. Las ficticias se recorren pero no se anotan."))
        celdas.append(code('#| tbl-cap: "Rutas del proyecto, en orden de ejecución"\n#| label: tbl-rutas\n'
                           "rutas = proyecto.paths(dummies=False)\n"
                           "pd.Series({r: ' – '.join(a) for r, a in rutas.items()}).to_frame('actividades')"))
        celdas.append(code("Markdown(f'**Respuesta:** el grafo tiene **{plural(len(rutas), \"ruta\")}** entre el nodo inicial y el final.')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code('#| tbl-cap: "Matriz de caminos"\n#| label: tbl-matrizCaminos\n'
                           "path_matrix = proyecto.path_matrix(dummies=False)\n"
                           f"proyecto.display_rutas('{ESTADO['grafo']}')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("duraciones = datos_proyecto['duration'][path_matrix.columns]\n"
                           "duraciones_rutas = path_matrix @ duraciones\n"
                           "from pmule import display_producto_rutas\n"
                           "display_producto_rutas(path_matrix, duraciones, duraciones_rutas)"))
        celdas.append(md("Cada fila del producto es la suma de las duraciones de las actividades de esa ruta:"))
        celdas.append(code('#| tbl-cap: "Duración de cada ruta, sumando sus actividades"\n#| label: tbl-rutasPaso-b\n'
                           "explica_rutas(proyecto, duraciones, 'días')[['ruta', 'duración (días) = suma', 'crítica']]"))
        celdas.append(code("duracion_proyecto = int(duraciones_rutas.max())\n"
                           "criticas = [r for r in duraciones_rutas.index if duraciones_rutas[r] == duracion_proyecto]\n"
                           "ruta_critica = criticas[0]\n"
                           "Markdown(f'La duración del proyecto es la mayor de las duraciones de las rutas: T = máx('\n"
                           "         f'{\", \".join(str(int(x)) for x in duraciones_rutas)}) = **{duracion_proyecto} días**, que coincide con el '\n"
                           "         f'tiempo más temprano del último nodo del grafo ({int(proyecto.duration(duraciones))}).\\n\\n'\n"
                           "         f'**Respuesta:** el proyecto dura {duracion_proyecto} días; '\n"
                           "         + ('; '.join(f'{r}: {\" – \".join(rutas[r])}' for r in criticas))\n"
                           "         + (' es la ruta crítica.' if len(criticas) == 1 else ' son rutas críticas.'))"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("La duración del proyecto es $\\tau = \\max_r (P\\,d)_r$. Si una actividad crítica dura un "
                     "día más, la ruta crítica se alarga un día y con ella el proyecto: su sensibilidad es 1. "
                     "Si una actividad no crítica dura un día más, la ruta más larga no cambia mientras no se "
                     "agote la holgura: sensibilidad 0. Al acortar una actividad crítica el proyecto sólo se "
                     "acorta hasta que otra ruta pasa a ser la más larga."))
    celdas.append(md("Para cada actividad basta mirar la ruta más larga que la contiene: su *margen* hasta T (que es la "
                     "holgura total de la actividad) dice si un día más alarga el proyecto, y la ruta más larga que *no* "
                     "la contiene dice si un día menos lo acorta."))
    celdas.append(code("T = duracion_proyecto\n"
                       "L = duraciones_rutas.astype(int)\n"
                       "sensibilidad = proyecto.tau_sensitivity(duraciones.to_dict())\n"
                       "filas = {}\n"
                       "for act in duraciones.index:\n"
                       "    con = [r for r in L.index if path_matrix.loc[r, act] == 1]\n"
                       "    sin = [r for r in L.index if path_matrix.loc[r, act] == 0]\n"
                       "    r_con = max(con, key=lambda r: L[r])\n"
                       "    L_sin = int(L[sin].max()) if sin else 0\n"
                       "    margen = T - int(L[r_con])\n"
                       "    mas = 1 if margen == 0 else 0\n"
                       "    menos = (max(T - 1, L_sin) - T) if margen == 0 else 0\n"
                       "    filas[act] = {'ruta más larga con ella': f'{r_con} ({int(L[r_con])})',\n"
                       "                  'margen = T − esa ruta': f'{T} − {int(L[r_con])} = {margen}',\n"
                       "                  'ruta más larga sin ella': L_sin if sin else '—',\n"
                       "                  '+1 día': f'{mas:+d}' if mas else '0', '−1 día': f'{menos:+d}' if menos else '0',\n"
                       "                  'tau_sensitivity': int(round(float(sensibilidad[act])))}\n"
                       "sens = pd.DataFrame(filas).T\n"
                       "sens"))
    celdas.append(code("crit = [a for a in sens.index if sens.loc[a, '+1 día'] == '+1']\n"
                       "resto = [a for a in sens.index if a not in crit]\n"
                       "baja = [a for a in crit if sens.loc[a, '−1 día'] == '-1']\n"
                       "_resto = lista_y([f\"{a} (margen {sens.loc[a, 'margen = T − esa ruta'].split(' = ')[-1]})\" for a in resto])\n"
                       "Markdown(f'La columna de la librería (`tau_sensitivity`) coincide con la de +1 día.\\n\\n'\n"
                       "         f'**Respuesta:** si una actividad dura un día más, el proyecto pasa de {T} a {T + 1} días cuando es crítica '\n"
                       "         f'({lista_y(crit)}: margen 0) y no cambia en las demás' + (f', {_resto}, porque su margen absorbe el día.' if resto else '.')\n"
                       "         + f' Si dura un día menos, el proyecto se acorta un día con '\n"
                       "         + (f'{lista_y(baja)}, porque la ruta más larga que no las contiene dura menos de {T}' if baja else 'ninguna')\n"
                       "         + (f'; con {lista_y([a for a in crit if a not in baja])} no cambia, porque otra ruta de {T} días no la contiene' if len(baja) < len(crit) else '')\n"
                       "         + '; acortar una actividad no crítica no cambia nada.')"))
    if inverso:
        celdas += apartado(ap); ap += 1
        celdas.append(md("La descomposición en valores singulares $P = U\\,S\\,V^T$ ordena la información de la "
                         "matriz de caminos por importancia. El primer vector singular por la izquierda, "
                         "$u_1$, pondera las rutas, y el primero por la derecha, $v_1$, las actividades: las "
                         "actividades que aparecen en muchas rutas y las rutas que contienen muchas actividades "
                         "compartidas reciben más peso. Es una ordenación estructural, que no usa las duraciones, "
                         "y por eso no tiene por qué coincidir con la ruta crítica."))
        celdas.append(code("U, S, Vt = np.linalg.svd(path_matrix.values.astype(float))\n"
                           "u1 = np.abs(U[:, 0]); v1 = np.abs(Vt[0, :])\n"
                           "importancia_rutas = pd.Series(u1 / u1.sum(), index=path_matrix.index)\n"
                           "importancia_actividades = pd.Series(v1 / v1.sum(), index=path_matrix.columns)\n"
                           "Math(r'S = ' + bmatrix(S.round(2), dec=2))"))
        celdas.append(code('#| tbl-cap: "Rutas y actividades ordenadas por su peso en el primer vector singular"\n#| label: tbl-svd\n'
                           "ordena_rutas(path_matrix.copy(), importancia_actividades.values, importancia_rutas.values)"))
        celdas.append(md("Cómo sale cada importancia: como $P\\,v_1 = s_1\\,u_1$, el peso de una ruta es la suma de los pesos "
                         "$v_1$ de sus actividades dividida por $s_1$; la columna «Importancia» de la tabla anterior es ese "
                         "peso normalizado para que las rutas sumen 1 ($u_1/\\sum u_1$), y cada celda de actividad es "
                         "$v_1/\\sum v_1$ si la actividad está en la ruta."))
        celdas.append(code("_v = pd.Series(v1, index=path_matrix.columns)\n"
                           "_u = pd.Series(u1, index=path_matrix.index)\n"
                           "display(Markdown(f'$s_1$ = {S[0]:.3f}; pesos $v_1$ de las actividades: '\n"
                           "                 + ', '.join(f'{a} {x:.3f}' for a, x in _v.sort_values(ascending=False).items()) + '.'))\n"
                           "_tabla = pd.DataFrame({r: {'ruta': ' – '.join(rutas_enunciado[r]),\n"
                           "                           'u₁ = (suma de v₁)/s₁': '(' + ' + '.join(f'{_v[a]:.3f}' for a in rutas_enunciado[r]) + f')/{S[0]:.3f} = {_u[r]:.3f}',\n"
                           "                           'importancia = u₁/Σu₁': round(float(importancia_rutas[r]), 3),\n"
                           "                           'duración': int(duraciones_rutas[r])} for r in path_matrix.index}).T\n"
                           "_tabla.sort_values('importancia = u₁/Σu₁', ascending=False)"))
        celdas.append(code("_dom = [r for r in importancia_rutas.index if np.isclose(importancia_rutas[r], importancia_rutas.max())]\n"
                           "_act = [a for a in importancia_actividades.index if np.isclose(importancia_actividades[a], importancia_actividades.max())]\n"
                           "_veces = int(path_matrix[_act[0]].sum())\n"
                           "Markdown('**Respuesta:** ' + (f'la ruta dominante según la SVD es {_dom[0]}' if len(_dom) == 1 else\n"
                           "         f'empatan como dominantes según la SVD las rutas {lista_y(_dom)}')\n"
                           "         + f' (importancia {importancia_rutas.max():.3f}); la ruta crítica, {ruta_critica} ({duracion_proyecto} días), '\n"
                           "         + ('es una de ellas' if ruta_critica in _dom and len(_dom) > 1 else ('coincide con ella' if ruta_critica in _dom else 'no coincide'))\n"
                           "         + '. ' + (f'La actividad de mayor peso es {_act[0]}' if len(_act) == 1 else f'Las actividades de mayor peso son {lista_y(_act)}')\n"
                           "         + f', que aparece{\"\" if len(_act) == 1 else \"n\"} en {_veces} de las {len(path_matrix)} rutas. La SVD sólo mira la '\n"
                           "         'estructura (en cuántas rutas está cada actividad); la ruta crítica depende además de las duraciones.')"))
    sub = f"{n} actividades, {info['n_rutas']} rutas" + (", reconstrucción desde las rutas y SVD" if inverso else "")
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _escenarios(rng):
    """Varios escenarios de duraciones a la vez: R·T con T de varias columnas."""
    n = int(rng.choice([7, 8]))
    info = P.buscar(rng, lambda i: 4 <= i["n_rutas"] <= 7 and 1 <= i["n_dummies"] <= 3 and i["n_criticas"] == 1,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    nombres = ["optimista", "normal", "pesimista"]
    T = pd.DataFrame({"optimista": (D - rng.integers(0, 3, size=len(D))).clip(lower=1),
                      "normal": D,
                      "pesimista": D + rng.integers(0, 4, size=len(D))}, index=D.index)
    df = P.tabla_base(info)
    celdas = [md("# Enunciado"),
              md("Para el proyecto cuyo cuadro de prelaciones se adjunta se manejan tres escenarios de duración "
                 "de las actividades (en días): optimista, normal y pesimista. Se quiere conocer la duración del "
                 "proyecto en los tres sin repetir tres veces el cálculo de tiempos.")]
    celdas.append(preguntas([
        f"Dibuje el grafo {grafo_nombre()} y construya la matriz de caminos R.",
        "Disponga los tres escenarios como columnas de una matriz T (una fila por actividad) y calcule el "
        "producto R·T. ¿Qué representa cada columna del resultado?",
        "Obtenga la duración del proyecto en cada escenario como el máximo de cada columna, e indique la ruta "
        "crítica de cada uno. ¿Es siempre la misma?",
        "¿Por qué este procedimiento es el mismo que se usa en el método de Monte Carlo?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de prelaciones"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(code('#| tbl-cap: "Escenarios de duración (días)"\n#| label: tbl-escenarios\n'
                       + literal_dataframe(T, nombre="escenarios")))
    celdas.append(md("# Resolución"))
    celdas.append(code(BMATRIX))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas += apartado(1)
    celdas += fig_grafo(durations="False", cap="Grafo PERT", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de caminos"\n#| label: tbl-matrizCaminos\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       f"proyecto.display_rutas('{ESTADO['grafo']}')"))
    celdas += apartado(2)
    celdas.append(code("T = escenarios.loc[R.columns]\n"
                       "duraciones_rutas = R @ T\n"
                       "from pmule import display_producto_rutas\n"
                       "display_producto_rutas(R, T, duraciones_rutas)"))
    celdas.append(md("Cada elemento de R·T es la suma de las duraciones, en ese escenario, de las actividades de esa ruta:"))
    celdas.append(code('#| tbl-cap: "Duración de cada ruta en cada escenario, sumando sus actividades"\n#| label: tbl-duracionesRutas\n'
                       "calendario = proyecto.calendar(escenarios['normal'])\n"
                       "acts_ruta = {r: ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario) for r in R.index}\n"
                       "pd.DataFrame({r: {'ruta': ' – '.join(acts_ruta[r]),\n"
                       "                  **{e: ' + '.join(str(int(T.loc[a, e])) for a in acts_ruta[r]) + f' = {int(duraciones_rutas.loc[r, e])}'\n"
                       "                     for e in T.columns}} for r in R.index}).T"))
    celdas.append(md("**Respuesta:** cada columna de R·T es el vector de duraciones de las rutas en un escenario: la fila r, "
                     "columna s, es la duración de la ruta r cuando las actividades duran lo que indica el escenario s."))
    celdas += apartado(3)
    celdas.append(code("_crit = {e: [r for r in R.index if duraciones_rutas.loc[r, e] == duraciones_rutas[e].max()] for e in T.columns}\n"
                       "resumen = pd.DataFrame({e: {'duración del proyecto = máx de la columna': f'máx({\", \".join(str(int(x)) for x in duraciones_rutas[e])}) = {int(duraciones_rutas[e].max())}',\n"
                       "                            'ruta crítica': '; '.join(f'{r}: {\" – \".join(acts_ruta[r])}' for r in _crit[e])}\n"
                       "                        for e in T.columns}).T\n"
                       "resumen"))
    celdas.append(code("_iguales = len({tuple(v) for v in _crit.values()}) == 1\n"
                       "Markdown('**Respuesta:** el proyecto dura ' + lista_y([f'{int(duraciones_rutas[e].max())} días en el escenario {e}' for e in T.columns])\n"
                       "         + '. ' + ('La ruta crítica es la misma en los tres escenarios.' if _iguales else\n"
                       "         'La ruta crítica cambia de un escenario a otro: la ruta más larga depende de las duraciones, no sólo de la forma del grafo.'))"))
    celdas += apartado(4)
    celdas.append(md("En Monte Carlo cada réplica es un escenario: la matriz T tiene una columna por réplica (miles "
                     "de columnas) y R·T da de una vez la duración de todas las rutas en todas las réplicas; el "
                     "máximo de cada columna es la duración del proyecto en esa réplica. Con tres columnas el "
                     "cálculo se hace a mano; con diez mil lo hace el ordenador, pero la operación es idéntica."))
    return celdas, f"{n} actividades, {info['n_rutas']} rutas, tres escenarios"


VARIANTES = [
    Variante("matriz-y-duracion", 1, "enumerar rutas, matriz de caminos, duración como máximo de R·d, sensibilidad",
             lambda rng: _base(1, rng)),
    Variante("matriz-y-duracion-b", 1, "ídem con otro proyecto", lambda rng: _base(2, rng)),
    Variante("rutas-a-grafo-svd", 3, "reconstruir el grafo a partir de las rutas; rutas dominantes por SVD",
             lambda rng: _base(3, rng), grafo="pert"),
    Variante("escenarios", 2, "tres escenarios de duración a la vez con R·T; puente con Monte Carlo", _escenarios),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
