"""Colección 08 — Probabilidad de cumplir un plazo con la aproximación normal
por el camino crítico."""
import numpy as np
import pandas as pd
from scipy.stats import norm

import networkx as nx
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

def _pl(n, singular, plural=None):
    """«1 semana», «3 semanas»: número y sustantivo concordados (para enunciados y subtítulos)."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


META = dict(slug="08_probabilidad", prefijo="Probabilidad",
            titulo="Probabilidad de cumplir el plazo (aproximación normal)",
            descripcion="Varianza del proyecto por el camino crítico, probabilidad de terminar antes de un "
                        "plazo o entre dos fechas y duración comprometible para una probabilidad dada.",
            n_defecto=3)


_AUX_RUTAS = '''
def tabla_rutas(R, medias, varianzas=None, calendario=None, dec=0, dec_var=3, nombre='media'):
    """Media (y varianza) de cada ruta con la suma término a término; actividades en orden de ejecución."""
    filas = {}
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), calendario)
        m = float(sum(float(medias[a]) for a in acts))
        fila = {'actividades': ' – '.join(acts),
                f'{nombre} = suma': ' + '.join(f'{a}({float(medias[a]):.{dec}f})' for a in acts) + f' = {m:.{dec}f}',
                nombre: m}
        if varianzas is not None:
            v = float(sum(float(varianzas[a]) for a in acts))
            fila['varianza = suma'] = ' + '.join(f'{a}({float(varianzas[a]):.{dec_var}f})' for a in acts) + f' = {v:.{dec_var}f}'
            fila['varianza'] = v
            fila['σ = √varianza'] = v ** 0.5
        filas[r] = fila
    t = pd.DataFrame(filas).T
    for c in (nombre, 'varianza', 'σ = √varianza'):
        if c in t:
            t[c] = t[c].astype(float)
    return t
'''


def _base(k, rng):
    varias = (k % 3 != 1)
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 10
                    and (i["n_criticas"] >= 2 if varias else i["n_criticas"] == 1) and i["T"] >= 12,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D = info["D"]
    sd = P.desviaciones(rng, list(D.index))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    T = int(info["T"])
    # varianza del proyecto = máxima entre rutas críticas
    var_rutas = {r: float((sd[info["P"].columns] ** 2 * info["P"].loc[r]).sum()) for r in info["rutas_criticas"]}
    sigma = float(np.sqrt(max(var_rutas.values())))
    plazo = T + int(rng.integers(1, 3))
    t_a, t_b = T - 1, T + 2
    prob_obj = float(rng.choice([0.90, 0.95]))

    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas "
                 "(en semanas) se adjunta.")]
    pregs = ["Determine la duración media del proyecto y su(s) camino(s) crítico(s).",
             "Calcule la varianza y la desviación típica de la duración del proyecto."
             + (" ¿Qué ruta crítica debe usarse para ello? Razone la respuesta." if varias else ""),
             f"¿Qué probabilidad hay de terminar el proyecto en {plazo} semanas o menos?",
             f"¿Qué probabilidad hay de terminar entre las semanas {t_a} y {t_b}?",
             f"¿Qué duración puede comprometerse con el cliente con una probabilidad del {int(prob_obj * 100)}% "
             "de cumplirla?",
             "¿Qué probabilidad hay de terminar en la duración media o antes? ¿Puede garantizarse esa duración "
             f"con un {int(prob_obj * 100)}% de probabilidad sin modificar el proyecto?"]
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    ap = 1
    celdas += apartado(ap); ap += 1
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos de los nodos y camino crítico", label="fig-pert")
    celdas.append(md("Holgura total de cada actividad (sin las ficticias), leída del grafo resuelto:"))
    celdas.append(code('#| tbl-cap: "Holgura total de las actividades"\n#| label: tbl-holguras\n'
                       "resultados_pert = proyecto.calculate_pert(durations=datos_proyecto['duration'])\n"
                       "holguras = resultados_pert['activities'].T\n"
                       "holguras = holguras[[c for c in holguras.columns if not str(c).startswith('@')]]   # sin las ficticias @i→j\n"
                       "holguras.loc[['H_total']].rename(index={'H_total': 'holgura total'})"))
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "actividades_criticas = ordena_ruta(holguras.columns[holguras.loc['H_total'] == 0], calendario)\n"
                       "rutas_criticas_txt = [' – '.join(ordena_ruta(acts, calendario)) for acts in rutas_criticas.values()]\n"
                       "display(Markdown(f'Las actividades con holgura total nula son {lista_y(actividades_criticas)}; el resto puede retrasarse '\n"
                       "                 'hasta su holgura total sin alargar el proyecto.'))\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "Markdown(f'**Respuesta:** la duración media del proyecto es {plural(duracion_media, \"semana\")}; '\n"
                       "         + (f'el camino crítico es {rutas_criticas_txt[0]}.' if len(rutas_criticas_txt) == 1 else\n"
                       "            f'hay {len(rutas_criticas_txt)} caminos críticos: {lista_y(rutas_criticas_txt)}.'))"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Por el teorema central del límite, la duración de una ruta es aproximadamente normal con "
                     "media la suma de las medias y varianza la suma de las varianzas de sus actividades "
                     "($\\sigma_i^2$ = desviación típica al cuadrado). Para el proyecto se toma la ruta crítica; si hay varias, la de mayor varianza, que es la "
                     "hipótesis más prudente (la verdadera distribución del máximo de varias rutas no es "
                     "normal, y el método PERT la aproxima así)."))
    celdas.append(md("Varianza de cada ruta crítica, suma de las varianzas de sus actividades (resaltada la mayor):"))
    celdas.append(code('#| tbl-cap: "Varianza de las rutas críticas"\n#| label: tbl-varianzaRutas\n'
                       "desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'],\n"
                       "                                                                   variances=datos_proyecto['variance'])\n"
                       "varianza_rutas.relabel_index(['actividades', 'varianza', 'suma de varianzas'], axis='columns')   # Styler: resalta la mayor varianza"))
    celdas.append(code("var_criticas = {r: float(datos_proyecto.loc[list(acts), 'variance'].sum()) for r, acts in rutas_criticas.items()}\n"
                       "ruta_var = max(var_criticas, key=var_criticas.get)\n"
                       "acts_var = ordena_ruta(rutas_criticas[ruta_var], calendario)\n"
                       "txt_var = ' – '.join(acts_var)\n"
                       "varianza_proyecto = var_criticas[ruta_var]\n"
                       "if len(var_criticas) == 1:\n"
                       "    eleccion = f'Sólo hay una ruta crítica, {txt_var}; su varianza es la del proyecto.'\n"
                       "else:\n"
                       "    otras = [r for r in var_criticas if r != ruta_var]\n"
                       "    comparacion_txt = lista_y([f\"{var_criticas[r]:.3f} de {' – '.join(ordena_ruta(rutas_criticas[r], calendario))}\" for r in otras])\n"
                       "    if all(np.isclose(var_criticas[r], varianza_proyecto) for r in otras):\n"
                       "        eleccion = (f'Las rutas críticas tienen la misma varianza ({varianza_proyecto:.3f}), así que da igual cuál se use: '\n"
                       "                    f'por ejemplo {txt_var}.')\n"
                       "    else:\n"
                       "        eleccion = (f'Hay {len(var_criticas)} rutas críticas con la misma media ({duracion_media} semanas) pero distinta varianza: '\n"
                       "                    f'{txt_var} tiene varianza {varianza_proyecto:.3f}, frente a {comparacion_txt}. Se usa la de mayor varianza, {txt_var}, '\n"
                       "                    'porque es la hipótesis más prudente: da la normal más dispersa y, para plazos por encima de la media, '\n"
                       "                    'la menor probabilidad de cumplirlos.')\n"
                       "Markdown(f'''{eleccion}\\n\\n{explica_sigma_ruta(acts_var, datos_proyecto[\"variance\"], nombre=\"σ_T\")} semanas.\\n\\n'''\n"
                       "         f'''**Respuesta:** varianza del proyecto σ²_T = {varianza_proyecto:.3f} y desviación típica σ_T = {desviacion_proyecto:.3f} semanas'''\n"
                       "         + (f''' (ruta {txt_var}).''' if len(var_criticas) > 1 else '.'))"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("La duración del proyecto se modela como $T \\sim N(\\mu, \\sigma_T)$, con la media y la "
                     "desviación típica de los apartados 1 y 2. La probabilidad se obtiene directamente con la "
                     "función de distribución normal de la hoja de cálculo (o de la calculadora), sin tipificar."))
    celdas.append(code(f"plazo = {plazo}\n"
                       "probabilidad_plazo = norm.cdf(plazo, duracion_media, desviacion_proyecto)\n"
                       "Markdown(f'''{explica_probabilidad(duracion_media, desviacion_proyecto, plazo)}.\\n\\n'''\n"
                       "         f'''**Respuesta:** la probabilidad de terminar en {plazo} semanas o menos es {probabilidad_plazo:.4f} ({probabilidad_plazo:.1%}).''')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Terminar entre las dos semanas es terminar después de la primera y no más tarde de la segunda: "
                     "la probabilidad es la diferencia de las dos funciones de distribución."))
    celdas.append(code(f"t_a, t_b = {t_a}, {t_b}\n"
                       "p_b = norm.cdf(t_b, duracion_media, desviacion_proyecto)\n"
                       "p_a = norm.cdf(t_a, duracion_media, desviacion_proyecto)\n"
                       "Markdown(f'''{explica_intervalo(duracion_media, desviacion_proyecto, t_a, t_b)}.\\n\\n'''\n"
                       "         f'''**Respuesta:** la probabilidad de terminar entre las semanas {t_a} y {t_b} es {p_b - p_a:.4f} ({p_b - p_a:.1%}).''')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Se busca el plazo $t$ que se cumple con la probabilidad pedida, $P(T \\le t) = p$: es el "
                     "percentil $p$ de la normal, que da la inversa de la función de distribución."))
    celdas.append(code(f"probabilidad_objetivo = {prob_obj}\n"
                       "duracion_comprometida = norm.ppf(probabilidad_objetivo, duracion_media, desviacion_proyecto)\n"
                       "semanas_enteras = int(np.ceil(duracion_comprometida))\n"
                       "Markdown(f'''{explica_percentil(duracion_media, desviacion_proyecto, probabilidad_objetivo, 'semanas')}.\\n\\n'''\n"
                       "         f'''Como el calendario se mide en semanas enteras, se redondea hacia arriba a {semanas_enteras} semanas: '''\n"
                       "         f'''{explica_probabilidad(duracion_media, desviacion_proyecto, semanas_enteras)} ≥ {probabilidad_objetivo:g}.\\n\\n'''\n"
                       "         f'''**Respuesta:** puede comprometerse una duración de {plural(semanas_enteras, \"semana\")} '''\n"
                       "         f'''({duracion_comprometida:.2f} semanas sin redondear).''')"))
    celdas += apartado(ap); ap += 1
    celdas.append(code("p_media = norm.cdf(duracion_media, duracion_media, desviacion_proyecto)\n"
                       "margen = duracion_comprometida - duracion_media\n"
                       "media_necesaria = duracion_media - margen\n"
                       f"Markdown(f'''{{explica_probabilidad(duracion_media, desviacion_proyecto, duracion_media)}}: la media de una normal es también su mediana, "
                       "así que sólo la mitad de las veces se terminaría en la duración media o antes. Esto no depende de σ_T: con la media en "
                       "{duracion_media} semanas, P(T ≤ {duracion_media}) = 0.5 por pequeña que sea la varianza, así que reducir la varianza no basta.\\n\\n"
                       "Para que {duracion_media} semanas se cumplan con un {probabilidad_objetivo:.0%} tienen que ser el percentil "
                       "{probabilidad_objetivo * 100:.0f} de la duración. Según el apartado anterior, ese percentil queda {duracion_comprometida:.2f} − "
                       "{duracion_media} = {margen:.2f} semanas por encima de la media; con la misma σ_T, la media debería bajar a "
                       "{duracion_media} − {margen:.2f} = {media_necesaria:.2f} semanas.\\n\\n"
                       "**Respuesta:** P(T ≤ {duracion_media}) = {p_media:.1f}. No puede garantizarse esa duración con un {probabilidad_objetivo:.0%} "
                       "sin modificar el proyecto: habría que acortar la duración media al menos {margen:.2f} semanas (hasta {media_necesaria:.2f}, "
                       "acortando actividades críticas) manteniendo la varianza.''')"))
    sub = f"{n} actividades, {_pl(info['n_criticas'], 'ruta crítica', 'rutas críticas')}, σ = {sigma:.2f}"
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _casi_critica(rng):
    """La ruta casi crítica es mucho más variable que la crítica: el PERT clásico se equivoca."""
    n = int(rng.choice([9, 10]))

    def acepta(i):
        return (1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1 and i["segunda"] is not None
                and i["T"] - i["segunda"] == 1 and i["T"] >= 14)

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, L, Pm = info["D"], info["L"], info["P"]
    T = int(info["T"])
    critica = L.idxmax()
    segunda = L.drop(critica).idxmax()
    sd = pd.Series(0.3, index=D.index)
    for a in Pm.columns[Pm.loc[critica] == 1]:
        sd[a] = 0.2
    for a in Pm.columns[(Pm.loc[segunda] == 1) & (Pm.loc[critica] == 0)]:
        sd[a] = float(rng.choice([1.5, 1.8, 2.0]))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    plazo = T + 2
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en "
                 "semanas) se adjunta.")]
    celdas.append(preguntas([
        "Determine la duración media del proyecto, la ruta crítica y la ruta que le sigue en duración.",
        f"Calcule la probabilidad de terminar en {plazo} semanas o menos según el método PERT (aproximación normal "
        "por la ruta crítica).",
        f"Calcule la misma probabilidad considerando únicamente la segunda ruta. ¿Qué ruta da la probabilidad más "
        "baja? ¿Cuál es entonces una cota razonable de la probabilidad de cumplir el plazo?",
        "Compruebe con Monte Carlo (10 000 réplicas, duraciones normales) cuál es la probabilidad real y explique "
        "por qué el método PERT se equivoca aquí más de lo habitual."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])"))
    celdas.append(code(_AUX_RUTAS))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(md("Media y varianza de cada ruta (sumas de las de sus actividades), ordenadas de mayor a menor duración media:"))
    celdas.append(code('#| tbl-cap: "Duración media y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "rutas = tabla_rutas(R, datos_proyecto['duration'], datos_proyecto['variance'], calendario).sort_values('media', ascending=False)\n"
                       "limpia_tabla(rutas, decimales=3)"))
    celdas.append(code("critica, segunda = rutas.index[0], rutas.index[1]\n"
                       "txt_c, txt_s = rutas.loc[critica, 'actividades'], rutas.loc[segunda, 'actividades']\n"
                       "mu_c, mu_s = float(rutas.loc[critica, 'media']), float(rutas.loc[segunda, 'media'])\n"
                       "s_c, s_s = float(rutas.loc[critica, 'σ = √varianza']), float(rutas.loc[segunda, 'σ = √varianza'])\n"
                       "Markdown(f'''**Respuesta:** la duración media del proyecto es {plural(mu_c, \"semana\")}; la ruta crítica es {txt_c} "
                       "(σ = {s_c:.3f}) y la que le sigue es {txt_s}, con {plural(mu_s, \"semana\")} de media (sólo {plural(mu_c - mu_s, \"semana\")} menos) "
                       "pero una desviación típica mucho mayor (σ = {s_s:.3f}).''')"))
    celdas += apartado(2)
    celdas.append(code(f"plazo = {plazo}\n"
                       "p_critica = norm.cdf(plazo, mu_c, s_c)\n"
                       "Markdown(f'''Por la ruta crítica {txt_c}: {explica_probabilidad(mu_c, s_c, plazo)}.\\n\\n'''\n"
                       "         f'''**Respuesta:** según el método PERT, P(T ≤ {plazo}) = {p_critica:.4f} ({p_critica:.1%}).''')"))
    celdas += apartado(3)
    celdas.append(code("p_segunda = norm.cdf(plazo, mu_s, s_s)\n"
                       "cota = min(p_critica, p_segunda)\n"
                       "manda = ('la segunda ruta, y no la crítica, es la que da la probabilidad más baja: manda en probabilidad aunque no sea la más larga en media'\n"
                       "         if p_segunda < p_critica else 'la ruta crítica sigue dando la probabilidad más baja')\n"
                       "Markdown(f'''Por la segunda ruta {txt_s} (duración L₂): {explica_probabilidad(mu_s, s_s, plazo, nombre=\"L₂\")}.\\n\\n'''\n"
                       "         f'''Para cumplir el plazo deben cumplirlo *todas* las rutas a la vez, así que la probabilidad real no puede superar la menor de las dos; '''\n"
                       "         f'''y como las rutas comparten actividades y no son independientes, tampoco es simplemente el producto.\\n\\n'''\n"
                       "         f'''**Respuesta:** {manda} ({p_segunda:.4f} frente a {p_critica:.4f}). Una cota razonable de la probabilidad de cumplir el plazo es '''\n"
                       "         f'''P(T ≤ {plazo}) ≤ {cota:.4f}.''')"))
    celdas += apartado(4)
    celdas.append(md("En cada réplica se sortea la duración de cada actividad con su normal (recortada en 0), se "
                     "suman las duraciones a lo largo de cada ruta y la duración del proyecto es la de la ruta "
                     "más larga en esa réplica. La probabilidad simulada es la proporción de réplicas que "
                     "terminan dentro del plazo."))
    celdas.append(code("rng = np.random.default_rng(7)\n"
                       "N = 10000\n"
                       "muestras = pd.DataFrame({a: np.clip(rng.normal(datos_proyecto.loc[a, 'duration'], datos_proyecto.loc[a, 'standard_deviation'], N), 0, None) for a in R.columns})\n"
                       "duraciones_rutas_mc = muestras @ R.T\n"
                       "duracion_mc = duraciones_rutas_mc.max(axis='columns')\n"
                       "p_mc = float((duracion_mc <= plazo).mean())\n"
                       "cumplen = int((duracion_mc <= plazo).sum())\n"
                       "mas_larga = duraciones_rutas_mc.idxmax(axis='columns')\n"
                       "frac_segunda = float((mas_larga == segunda).mean())\n"
                       "frac_critica = float((mas_larga == critica).mean())\n"
                       "pd.Series({'PERT (ruta crítica)': p_critica, 'segunda ruta': p_segunda, 'Monte Carlo': p_mc}).round(4).to_frame(f'P(T ≤ {plazo})')"))
    celdas.append(code("error_tipico = np.sqrt(cota * (1 - cota) / N)\n"
                       "if p_mc <= cota:\n"
                       "    frase_cota = f'queda por debajo de la cota del apartado 3, {cota:.4f}, como debe ser'\n"
                       "else:\n"
                       "    frase_cota = (f'supera en {p_mc - cota:.4f} la cota del apartado 3, {cota:.4f}: es error de muestreo, porque con {N} réplicas '\n"
                       "                  f'una proporción tiene un error típico de √(p(1 − p)/N) = √({cota:.4f} · {1 - cota:.4f} / {N}) = {error_tipico:.4f}')\n"
                       "culpa = ('es *ella* la que más veces hace fallar el plazo' if p_segunda < p_critica else\n"
                       "         'en muchas réplicas resulta la más larga y alarga el proyecto')\n"
                       "Markdown(f'''Monte Carlo: {cumplen} de las {N} réplicas terminan en {plazo} semanas o menos, luego P(T ≤ {plazo}) ≈ {cumplen}/{N} = **{p_mc:.4f}**, "
                       "que {frase_cota}. La segunda ruta {txt_s} resulta la más larga en el {frac_segunda:.1%} de las réplicas y la crítica {txt_c} en el {frac_critica:.1%}.\\n\\n"
                       "**Respuesta:** la probabilidad real es ≈ {p_mc:.4f}, frente al {p_critica:.4f} del método PERT, que la sobrestima en {p_critica - p_mc:.4f}. "
                       "El método PERT sólo mira la ruta de mayor duración media ({txt_c}, σ = {s_c:.3f}); aquí otra ruta casi tan larga ({plural(mu_c - mu_s, \"semana\")} "
                       "menos) es mucho más variable (σ = {s_s:.3f}) y {culpa}. La regla práctica es examinar todas las rutas "
                       "casi críticas y quedarse con la probabilidad más baja, o directamente simular.''')"))
    return celdas, f"{n} actividades; segunda ruta a {_pl(T - int(info['segunda']), 'semana')} de la crítica y mucho más variable"


def _varianza_maxima(rng):
    """Problema inverso: qué desviación típica puede permitirse una actividad crítica para cumplir el plazo
    con una probabilidad dada."""
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1 and i["T"] >= 12
                    and i["segunda"] is not None and i["T"] - i["segunda"] >= 3,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    sd = P.desviaciones(rng, list(D.index))
    criticas = cal.index[cal["H_total"] == 0].tolist()
    X = criticas[int(rng.integers(0, len(criticas)))]
    sd[X] = np.nan
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    T = int(info["T"])
    plazo = T + 2
    prob = 0.90
    celdas = [md("# Enunciado"),
              md(f"Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas (en "
                 f"semanas) se adjunta. La actividad {X} se va a subcontratar y la desviación típica de su duración "
                 f"depende del contratista elegido. El cliente exige terminar en {plazo} semanas con una probabilidad "
                 f"de al menos el {int(prob * 100)} %.")]
    celdas.append(preguntas([
        "Determine la duración media del proyecto y la ruta crítica. ¿Está en ella la actividad subcontratada?",
        f"Calcule la varianza del proyecto en función de la desviación típica σ de {X}.",
        f"¿Qué desviación típica máxima puede tener {X} para cumplir la exigencia del cliente?",
        f"Si ningún contratista garantiza esa dispersión, ¿cuántas semanas habría que reducir la duración media "
        f"del proyecto (manteniendo las varianzas) para cumplir con el contratista de σ = 1,5?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado (σ de ' + X + ' por determinar)"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df).replace("nan", "np.nan")))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"X, plazo, probabilidad_objetivo = '{X}', {plazo}, {prob}"))
    celdas.append(code(_AUX_RUTAS))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "criticas = ordena_ruta(sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)), calendario)\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "Markdown(f'**Respuesta:** duración media {plural(duracion_media, \"semana\")}; ruta crítica {\" – \".join(criticas)}. '\n"
                       "         f'La actividad subcontratada, {X}, {\"sí\" if X in criticas else \"no\"} está en ella.')"))
    celdas.append(md("Media de cada ruta y varianza de sus actividades de desviación conocida (la de " + X + " se cuenta como 0 "
                     "hasta conocerla):"))
    celdas.append(code('#| tbl-cap: "Media de cada ruta y varianza de las actividades de desviación conocida"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "sd_conocida = pd.to_numeric(datos_proyecto['standard_deviation'], errors='coerce').fillna(0.0)\n"
                       "rutas = tabla_rutas(R, datos_proyecto['duration'], sd_conocida ** 2, calendario).sort_values('media', ascending=False)\n"
                       "limpia_tabla(rutas.rename(columns={'varianza = suma': f'varianza sin {X} = suma', 'varianza': f'varianza sin {X}'}).drop(columns='σ = √varianza'), decimales=3)"))
    celdas += apartado(2)
    celdas.append(code("resto = [a for a in criticas if a != X]\n"
                       "var_resto = float((datos_proyecto.loc[resto, 'standard_deviation'] ** 2).sum())\n"
                       "Markdown(f'''La varianza del proyecto es la de la ruta crítica, la suma de las varianzas de sus actividades. Sin {X}: '''\n"
                       "         f'''{explica_suma((datos_proyecto.loc[resto, \"standard_deviation\"] ** 2).round(3), \"σ²_resto\", decimales=3)}; '''\n"
                       "         f'''a ella se suma la varianza de {X}, σ² (σ_X en lo que sigue).\\n\\n**Respuesta:** σ²_T = {var_resto:.3f} + σ², es decir, σ_T = √({var_resto:.3f} + σ²).''')"))
    celdas += apartado(3)
    celdas.append(md("Exigir $P(T \\le \\text{plazo}) \\ge p$ equivale a que el plazo sea al menos el percentil $p$ de "
                     "$T$. En cualquier normal ese percentil está a un número fijo de desviaciones típicas por encima "
                     "de la media, $k_p$, el percentil de la normal de media 0 y desviación 1 "
                     "(`=INV.NORM.ESTAND(p)` o `=INV.NORM(p; 0; 1)` en la hoja de cálculo). Por tanto hace falta "
                     "$\\mu + k_p\\,\\sigma_T \\le \\text{plazo}$, es decir, $\\sigma_T \\le (\\text{plazo} - \\mu)/k_p$, "
                     "y como $\\sigma_T^2 = \\sigma^2_{resto} + \\sigma_X^2$, $\\sigma_X^2 \\le \\sigma_T^2 - \\sigma^2_{resto}$."))
    celdas.append(code("z_p = norm.ppf(probabilidad_objetivo)\n"
                       "sigma_T_max = (plazo - duracion_media) / z_p\n"
                       "var_X_max = sigma_T_max ** 2 - var_resto\n"
                       "sigma_X_max = np.sqrt(var_X_max) if var_X_max > 0 else 0.0\n"
                       "p_txt = f'{probabilidad_objetivo:g}'.replace('.', ',')\n"
                       "if var_X_max > 0:\n"
                       "    comprobacion = (f'Comprobación con σ_X = {sigma_X_max:.3f}: σ_T = √({var_resto:.3f} + {sigma_X_max:.3f}²) = {sigma_T_max:.3f} y '\n"
                       "                    f'{explica_probabilidad(duracion_media, sigma_T_max, plazo)}.')\n"
                       "    respuesta = f'la desviación típica de {X} no puede superar **{sigma_X_max:.2f} semanas**'\n"
                       "else:\n"
                       "    comprobacion = 'Ni con σ_X = 0 se alcanza la probabilidad exigida: el resto de la ruta crítica ya es demasiado variable.'\n"
                       "    respuesta = 'ningún contratista permite cumplir la exigencia, ni siquiera uno sin variabilidad'\n"
                       "Markdown(f'''k_p = `=INV.NORM.ESTAND({p_txt})` = {z_p:.4f}.\\n\\n'''\n"
                       "         f'''σ_T ≤ ({plazo} − {duracion_media}) / {z_p:.4f} = {sigma_T_max:.3f}; σ_X² ≤ {sigma_T_max:.3f}² − {var_resto:.3f} = {var_X_max:.3f}'''\n"
                       "         + (f''', luego σ_X ≤ √{var_X_max:.3f} = {sigma_X_max:.3f}.''' if var_X_max > 0 else '.') + '\\n\\n' + comprobacion\n"
                       "         + f'''\\n\\n**Respuesta:** {respuesta}.''')"))
    celdas += apartado(4)
    celdas.append(md("Con la desviación típica del contratista, $\\sigma_T$ queda fijada y lo que se ajusta es la media: "
                     "hace falta $\\mu \\le \\text{plazo} - k_p\\,\\sigma_T$."))
    celdas.append(code("sigma_contratista = 1.5\n"
                       "sigma_T = np.sqrt(var_resto + sigma_contratista ** 2)\n"
                       "media_necesaria = plazo - z_p * sigma_T\n"
                       "reduccion = max(int(np.ceil(duracion_media - media_necesaria)), 0)\n"
                       "nueva_media = duracion_media - reduccion\n"
                       "segunda = float(rutas['media'].iloc[1]) if len(rutas) > 1 else 0.0\n"
                       "if segunda <= nueva_media:\n"
                       "    nota_ruta = f'La ruta crítica no cambia: la siguiente ruta dura {segunda:g} semanas de media, no más que {nueva_media}.'\n"
                       "else:\n"
                       "    nota_ruta = (f'Ojo: la siguiente ruta dura {segunda:g} semanas de media, más que {nueva_media}; habría que acortarla también '\n"
                       "                 'y recalcular la varianza con la nueva ruta crítica.')\n"
                       "Markdown(f'''σ_T = √({var_resto:.3f} + 1.5²) = √{var_resto + 2.25:.3f} = {sigma_T:.3f}; hace falta μ ≤ {plazo} − {z_p:.4f} × {sigma_T:.3f} = {media_necesaria:.2f} semanas. '''\n"
                       "         f'''Como las duraciones son enteras, la media debe bajar de {duracion_media} a {nueva_media} semanas. Comprobación: '''\n"
                       "         f'''{explica_probabilidad(nueva_media, sigma_T, plazo)}. {nota_ruta}\\n\\n'''\n"
                       "         f'''**Respuesta:** habría que reducir la duración media del proyecto en {plural(reduccion, \"semana\")} (acortando actividades críticas).''')"))
    return celdas, f"{n} actividades; σ máxima admisible de {X} para P ≥ {int(prob*100)} % en {plazo} semanas"


def _diagnostico(rng):
    """Una resolución con errores típicos: varianza de una ruta no crítica, suma de σ en vez de varianzas."""
    n = int(rng.choice([9, 10]))
    info = P.buscar(rng, lambda i: 1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 1 and i["T"] >= 12 and i["n_rutas"] >= 4,
                    n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, L, Pm = info["D"], info["L"], info["P"]
    sd = P.desviaciones(rng, list(D.index))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    T = int(info["T"])
    critica = L.idxmax()
    otra = L.drop(critica).idxmax()
    acts_c = list(Pm.columns[Pm.loc[critica] == 1])
    acts_o = list(Pm.columns[Pm.loc[otra] == 1])
    sigma_ok = float(np.sqrt((sd[acts_c] ** 2).sum()))
    sigma_err1 = float(np.sqrt((sd[acts_o] ** 2).sum()))     # ruta equivocada
    sigma_err2 = float(sd[acts_c].sum())                        # suma de σ
    plazo = T + 2
    p_ok = float(norm.cdf((plazo - T) / sigma_ok))
    p_err = float(norm.cdf((plazo - T) / sigma_err2))
    celdas = [md("# Enunciado"),
              md("Para el proyecto cuyo cuadro se adjunta, un alumno ha entregado la siguiente resolución de la "
                 f"pregunta «¿qué probabilidad hay de terminar en {plazo} semanas o menos?»:\n\n"
                 f"> *La ruta crítica es {' - '.join(acts_c)}, con duración media {T}. La desviación típica del "
                 f"proyecto es la suma de las desviaciones típicas de sus actividades: σ = {' + '.join(f'{sd[a]:.1f}' for a in acts_c)} = "
                 f"{sigma_err2:.1f}. Por tanto z = ({plazo} − {T}) / {sigma_err2:.1f} = {(plazo - T) / sigma_err2:.2f} y "
                 f"P = {p_err:.3f}.*\n\n"
                 f"Otro alumno ha usado la ruta {' - '.join(acts_o)} para calcular la desviación típica, obteniendo σ = {sigma_err1:.2f}.")]
    celdas.append(preguntas([
        "Compruebe la duración media y la ruta crítica.",
        "Localice el error del primer alumno y corrija el cálculo. ¿Sobrestima o subestima la probabilidad su error?",
        "¿Qué error ha cometido el segundo alumno? ¿En qué caso su desviación típica sería la correcta?",
        f"Dé la probabilidad correcta de terminar en {plazo} semanas o menos."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"plazo = {plazo}\n"
                       f"ruta_alumno_1 = {acts_c!r}   # ruta crítica del primer alumno\n"
                       f"ruta_alumno_2 = {acts_o!r}   # ruta usada por el segundo alumno\n"
                       f"sigma_alumno_1 = {sigma_err2!r}"))
    celdas.append(code(_AUX_RUTAS))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con tiempos y camino crítico", label="fig-pert")
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "criticas = ordena_ruta(sorted(set(a for acts in proyecto.critical_path(datos_proyecto['duration']).values() for a in acts)), calendario)\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "coincide = set(criticas) == set(ruta_alumno_1)\n"
                       "Markdown(f'**Respuesta:** la duración media es {plural(duracion_media, \"semana\")} y la ruta crítica {\" – \".join(criticas)}: '\n"
                       "         + ('ambas coinciden con las del primer alumno.' if coincide else 'no coinciden con las del primer alumno.'))"))
    celdas += apartado(2)
    celdas.append(md("Las varianzas de actividades independientes se suman; las desviaciones típicas no: "
                     "$\\sigma_T = \\sqrt{\\sum \\sigma_i^2}$, que es siempre menor que $\\sum \\sigma_i$. El primer "
                     "alumno ha sumado desviaciones típicas."))
    celdas.append(code("desviacion_proyecto, varianza_rutas = proyecto.standard_deviation(datos_proyecto['duration'], datos_proyecto['variance'])\n"
                       "suma_sigmas = float(datos_proyecto.loc[criticas, 'standard_deviation'].sum())\n"
                       "p_correcta = norm.cdf(plazo, duracion_media, desviacion_proyecto)\n"
                       "p_alumno = norm.cdf(plazo, duracion_media, sigma_alumno_1)\n"
                       "sentido = 'subestima' if p_alumno < p_correcta else 'sobrestima'\n"
                       "Markdown(f'''Cálculo correcto: {explica_sigma_ruta(criticas, datos_proyecto[\"variance\"], nombre=\"σ_T\")} semanas, frente a la suma de desviaciones '''\n"
                       "         f'''del alumno, {suma_sigmas:.1f}.\\n\\n'''\n"
                       "         f'''- Con la σ del alumno: {explica_probabilidad(duracion_media, sigma_alumno_1, plazo)}.\\n'''\n"
                       "         f'''- Con la σ correcta: {explica_probabilidad(duracion_media, desviacion_proyecto, plazo)}.\\n\\n'''\n"
                       "         f'''**Respuesta:** el error es sumar desviaciones típicas en lugar de varianzas; la σ correcta es {desviacion_proyecto:.3f} y no {suma_sigmas:.1f}. '''\n"
                       "         f'''Una σ demasiado grande reparte la probabilidad hacia las colas y, para un plazo por encima de la media ({plazo} > {duracion_media}), '''\n"
                       "         f'''el alumno **{sentido}** la probabilidad: {p_alumno:.4f} en lugar de {p_correcta:.4f}.''')"))
    celdas += apartado(3)
    celdas.append(md("Media y varianza de cada ruta (sumas de las de sus actividades):"))
    celdas.append(code('#| tbl-cap: "Media y varianza de cada ruta (sumas de las de sus actividades)"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "rutas = tabla_rutas(R, datos_proyecto['duration'], datos_proyecto['variance'], calendario)\n"
                       "limpia_tabla(rutas, decimales=3)"))
    celdas.append(code("r2 = next(r for r in R.index if set(R.columns[R.loc[r] == 1]) == set(ruta_alumno_2))\n"
                       "media_2, var_2, sigma_2 = (float(rutas.loc[r2, c]) for c in ('media', 'varianza', 'σ = √varianza'))\n"
                       "var_c = desviacion_proyecto ** 2\n"
                       "if var_2 > var_c:\n"
                       "    caso = (f'Su σ sería la correcta si esa ruta fuera también crítica (media {duracion_media} en lugar de {media_2:g}): habría entonces dos rutas '\n"
                       "            f'críticas y, como la suya tiene más varianza ({var_2:.3f} > {var_c:.3f}), sería la que hay que usar.')\n"
                       "else:\n"
                       "    caso = (f'Su σ sólo sería la correcta si esa ruta fuera la única crítica; aunque empatara en media con la crítica actual, se usaría la de '\n"
                       "            f'mayor varianza, y la suya tiene menos ({var_2:.3f} ≤ {var_c:.3f}).')\n"
                       "Markdown(f'''**Respuesta:** el segundo alumno ha sumado varianzas, que es lo correcto, pero de la ruta {rutas.loc[r2, \"actividades\"]}, que no es la crítica: '''\n"
                       "         f'''dura {media_2:g} semanas de media frente a {duracion_media}. La varianza del proyecto se toma de la ruta crítica ({var_c:.3f}), no de otra. {caso}''')"))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'''{explica_probabilidad(duracion_media, desviacion_proyecto, plazo)}.\\n\\n'''\n"
                       "         f'''**Respuesta:** la probabilidad correcta de terminar en {plazo} semanas o menos es {p_correcta:.4f} ({p_correcta:.1%}), '''\n"
                       "         f'''frente al {p_alumno:.3f} del primer alumno.''')"))
    return celdas, f"{n} actividades; resolución con σ sumadas y con la ruta equivocada"



def _hito(rng):
    """Probabilidad de alcanzar un hito intermedio (el fin de una actividad) en una fecha dada."""
    n = int(rng.choice([9, 10]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 10 and i["n_criticas"] == 1):
            return False
        cal = i["calendario"]
        suc = P.sucesoras(i["preds"])
        # hito: actividad con sucesoras, que empieza al menos en la semana 4 y a la que llegan ≥ 2 cadenas
        G = P.grafo(i["preds"])
        cand = [a for a in cal.index if suc[a] and cal.loc[a, "inicio_mas_temprano"] >= 4
                and sum(len(list(nx.all_simple_paths(G, x, a))) for x in i["iniciales"]) >= 2]
        if not cand:
            return False
        estado["X"] = cand[int(rng.integers(0, len(cand)))]
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, cal = info["D"], info["calendario"]
    X = estado["X"]
    sd = P.desviaciones(rng, list(D.index))
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    EF_X = int(cal.loc[X, "fin_mas_temprano"])
    fecha = EF_X + int(rng.integers(1, 3))
    T = int(info["T"])
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas "
                 f"(en semanas) se adjunta. El fin de la actividad {X} es un hito del proyecto (una entrega "
                 f"parcial) comprometido para la semana {fecha}.")]
    celdas.append(preguntas([
        f"Enumere todas las cadenas de actividades que van desde una actividad inicial hasta {X} (incluida) "
        "y calcule la media y la varianza de la duración de cada una.",
        f"¿Cuál es la fecha media del hito? ¿Qué cadena determina su varianza? Calcule la probabilidad de "
        f"alcanzar el hito en la semana {fecha} o antes.",
        f"Calcule la probabilidad de terminar *todo el proyecto* en la semana {T + 1} o antes. ¿Por qué son "
        "cuestiones distintas, aunque se resuelvan igual?",
        f"¿Es {X} crítica? ¿Cambiaría la probabilidad del hito si se retrasara una actividad que no esté en "
        f"ninguna cadena que llegue a {X}?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"X, fecha_hito = '{X}', {fecha}"))
    celdas.append(code(_AUX_RUTAS))
    celdas += apartado(1)
    celdas.append(md("Las cadenas que llegan al hito son los caminos del grafo de actividades (Roy) desde las "
                     "iniciales hasta $X$. Cada una tiene media la suma de las medias y varianza la suma de las "
                     "varianzas de sus actividades."))
    celdas.append(code('#| fig-cap: "Grafo Roy con el hito marcado"\n#| label: fig-roy\n'
                       "proyecto.roy(durations=datos_proyecto['duration'], ranksep=1.2, nodesep=1.2)"))
    celdas.append(code('#| tbl-cap: "Cadenas hasta el hito"\n#| label: tbl-cadenas\n'
                       "G = nx.DiGraph([(p, a) for a, ps in prelaciones.immediate_predecessor.items() for p in ps])\n"
                       "cadenas = [c for ini in sorted(prelaciones.start_activities) for c in nx.all_simple_paths(G, ini, X)]\n"
                       "if not cadenas:\n"
                       "    cadenas = [[X]]\n"
                       "R_hito = pd.DataFrame(0, index=[f'cadena_{i + 1}' for i in range(len(cadenas))], columns=list(datos_proyecto.index))\n"
                       "for i, c in enumerate(cadenas):\n"
                       "    R_hito.loc[f'cadena_{i + 1}', c] = 1\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "tabla_hito = tabla_rutas(R_hito, datos_proyecto['duration'], datos_proyecto['variance'], calendario)\n"
                       "limpia_tabla(tabla_hito, decimales=3)"))
    celdas.append(code("Markdown('**Respuesta:** ' + '; '.join(f\"{tabla_hito.loc[c, 'actividades']}: media {tabla_hito.loc[c, 'media']:g}, varianza {tabla_hito.loc[c, 'varianza']:.3f}\" for c in tabla_hito.index) + '.')"))
    celdas += apartado(2)
    celdas.append(code("fecha_media = float(tabla_hito['media'].max())\n"
                       "criticas_hito = tabla_hito.index[np.isclose(tabla_hito['media'].astype(float), fecha_media)]\n"
                       "cadena_hito = max(criticas_hito, key=lambda c: tabla_hito.loc[c, 'varianza'])\n"
                       "varianza_hito = float(tabla_hito.loc[cadena_hito, 'varianza'])\n"
                       "sigma_hito = float(np.sqrt(varianza_hito))\n"
                       "p_hito = norm.cdf(fecha_hito, fecha_media, sigma_hito)\n"
                       "txt_cadena = tabla_hito.loc[cadena_hito, 'actividades']\n"
                       "empate = (f' Hay {len(criticas_hito)} cadenas con esa media; se toma la de mayor varianza.' if len(criticas_hito) > 1 else '')\n"
                       "Markdown(f'''La fecha media del hito es la de la cadena más larga, {txt_cadena}: {fecha_media:g} semanas (coincide con el fin más temprano de {X} '''\n"
                       "         f'''en el calendario, {int(calendario.loc[X, \"fin_mas_temprano\"])}).{empate} Esa cadena determina la varianza del hito: '''\n"
                       "         f'''{explica_sigma_ruta(ordena_ruta(cadenas[list(R_hito.index).index(cadena_hito)], calendario), datos_proyecto[\"variance\"], nombre=\"σ_hito\")}.\\n\\n'''\n"
                       "         f'''{explica_probabilidad(fecha_media, sigma_hito, fecha_hito, nombre=\"T_hito\")}.\\n\\n'''\n"
                       "         f'''**Respuesta:** fecha media del hito: semana {fecha_media:g}; determina su varianza la cadena {txt_cadena} '''\n"
                       "         f'''(σ² = {varianza_hito:.3f}); la probabilidad de alcanzar el hito en la semana {fecha_hito} o antes es {p_hito:.4f} ({p_hito:.1%}).''')"))
    celdas += apartado(3)
    celdas.append(code(f"plazo = {T + 1}\n"
                       "duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "sigma_proyecto, varianza_rutas = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "ruta_var = max(rutas_criticas, key=lambda r: datos_proyecto.loc[list(rutas_criticas[r]), 'variance'].sum())\n"
                       "acts_var = ordena_ruta(rutas_criticas[ruta_var], calendario)\n"
                       "p_proyecto = norm.cdf(plazo, duracion_media, sigma_proyecto)\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "Markdown(f'''{explica_sigma_ruta(acts_var, datos_proyecto[\"variance\"], nombre=\"σ_T\")}.\\n\\n'''\n"
                       "         f'''{explica_probabilidad(duracion_media, sigma_proyecto, plazo)}.\\n\\n'''\n"
                       "         f'''**Respuesta:** P(T ≤ {plazo}) = {p_proyecto:.4f} ({p_proyecto:.1%}). Es la misma técnica aplicada a otro suceso: el hito depende sólo '''\n"
                       "         f'''de las cadenas que llegan a {X} (media {fecha_media:g}, σ = {sigma_hito:.3f}); el fin del proyecto, de todas las rutas completas '''\n"
                       "         f'''(media {duracion_media}, σ = {sigma_proyecto:.3f}). Una probabilidad alta del hito no garantiza una probabilidad alta del plazo, ni al revés.''')"))
    celdas += apartado(4)
    celdas.append(code("es_critica = int(calendario.loc[X, 'H_total']) == 0\n"
                       "antecesoras = nx.ancestors(G, X) | {X}\n"
                       "ajenas = ordena_ruta(sorted(set(datos_proyecto.index) - antecesoras), calendario)\n"
                       "Markdown(f'''**Respuesta:** {X} {\"es\" if es_critica else \"no es\"} crítica (holgura total {plural(int(calendario.loc[X, \"H_total\"]), \"semana\")}). '''\n"
                       "         f'''La probabilidad del hito sólo depende de las actividades de sus cadenas ({lista_y(ordena_ruta(sorted(antecesoras), calendario))}): '''\n"
                       "         f'''un retraso en {lista_y(ajenas, \"o\")} no la altera, aunque sí pueda alterar la del proyecto.''')"))
    return celdas, f"{n} actividades; hito al final de {X} (fecha media {EF_X}, comprometida {fecha})"


def _ser_critica(rng):
    """Dos cadenas paralelas: probabilidad de que cada una sea la crítica y probabilidad exacta del plazo."""
    n1, n2 = int(rng.choice([2, 3])), int(rng.choice([2, 3]))
    nombres1 = list("BCD")[:n1]
    nombres2 = list("EFG")[:n2]
    preds = {"A": set()}
    prev = "A"
    for a in nombres1:
        preds[a] = {prev}; prev = a
    fin1 = prev
    prev = "A"
    for a in nombres2:
        preds[a] = {prev}; prev = a
    fin2 = prev
    preds["H"] = {fin1, fin2}
    # medias: cadena 1 algo más larga que la 2
    while True:
        d1 = rng.integers(3, 8, size=n1)
        d2 = rng.integers(3, 8, size=n2)
        if 1 <= d1.sum() - d2.sum() <= 2:
            break
    D = pd.Series({"A": int(rng.integers(2, 4)), **{a: int(x) for a, x in zip(nombres1, d1)},
                   **{a: int(x) for a, x in zip(nombres2, d2)}, "H": int(rng.integers(1, 3))}, name="duration")
    sd = pd.Series({"A": 0.0, **{a: float(rng.choice([0.5, 0.8, 1.0])) for a in nombres1},
                    **{a: float(rng.choice([1.0, 1.2, 1.5])) for a in nombres2}, "H": 0.0}, name="standard_deviation")
    info = P.analizar(preds, D)
    df = P.tabla_base(info, duration=D, standard_deviation=sd)
    T = int(info["T"])
    plazo = T + 1
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones medias y desviaciones típicas "
                 "(en semanas) se adjunta. Las actividades A y H tienen duración fija; las demás son "
                 "independientes y aproximadamente normales.")]
    celdas.append(preguntas([
        "Dibuje el grafo y enumere las rutas con su duración media y su varianza. ¿Cuál es la ruta crítica "
        "en media?",
        "Las dos rutas comparten A y H. Calcule la media y la varianza de la *diferencia* entre la duración "
        "de la ruta que pasa por " + nombres1[0] + " y la que pasa por " + nombres2[0] + ". ¿Qué probabilidad "
        "hay de que la segunda resulte más larga (es decir, de que sea ella la crítica)?",
        f"Calcule la probabilidad de terminar el proyecto en {plazo} semanas o menos (a) con el método PERT "
        "(sólo la ruta crítica) y (b) de forma exacta, usando que las dos cadenas son independientes: el "
        "proyecto termina a tiempo si *las dos* cadenas terminan a tiempo.",
        "¿Cuál de las dos probabilidades es mayor? ¿Es general ese sentido del error del método PERT?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"plazo = {plazo}"))
    celdas.append(code(_AUX_RUTAS))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Media y varianza de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "rutas = tabla_rutas(R, datos_proyecto['duration'], datos_proyecto['variance'], calendario)\n"
                       "limpia_tabla(rutas, decimales=3)"))
    celdas.append(code(f"ruta1 = R.index[R['{nombres1[0]}'] == 1][0]; ruta2 = R.index[R['{nombres2[0]}'] == 1][0]\n"
                       "mu1, mu2 = float(rutas.loc[ruta1, 'media']), float(rutas.loc[ruta2, 'media'])\n"
                       "v1, v2 = float(rutas.loc[ruta1, 'varianza']), float(rutas.loc[ruta2, 'varianza'])\n"
                       "txt1, txt2 = rutas.loc[ruta1, 'actividades'], rutas.loc[ruta2, 'actividades']\n"
                       "Markdown(f'**Respuesta:** en media la ruta crítica es {txt1 if mu1 >= mu2 else txt2} ({max(mu1, mu2):g} semanas frente a {min(mu1, mu2):g} '\n"
                       "         f'de {txt2 if mu1 >= mu2 else txt1}).')"))
    celdas += apartado(2)
    celdas.append(md("A y H están en las dos rutas y son fijas: en la diferencia $L_2 - L_1$ se cancelan. La "
                     "diferencia de dos normales independientes es normal con media la diferencia de medias y "
                     "varianza la *suma* de varianzas."))
    celdas.append(code("mu_dif, var_dif = mu2 - mu1, v1 + v2\n"
                       "signo = lambda x: f'{x:g}'.replace('-', '−')\n"
                       "sigma_dif = np.sqrt(var_dif)\n"
                       "p_ruta2_critica = 1 - norm.cdf(0, mu_dif, sigma_dif)\n"
                       "Markdown(f'''Con L₁ = duración de {txt1} y L₂ = duración de {txt2}: media de L₂ − L₁ = {mu2:g} − {mu1:g} = {signo(mu_dif)}; '''\n"
                       "         f'''varianza = {v1:.3f} + {v2:.3f} = {var_dif:.3f}; σ = √{var_dif:.3f} = {sigma_dif:.3f}.\\n\\n'''\n"
                       "         f'''La segunda ruta es más larga si L₂ − L₁ > 0: {explica_probabilidad(mu_dif, sigma_dif, 0, mayor=True, nombre=\"L₂ − L₁\")}.\\n\\n'''\n"
                       "         f'''**Respuesta:** L₂ − L₁ tiene media {signo(mu_dif)} y varianza {var_dif:.3f}; la ruta «no crítica» {txt2} resulta la más larga '''\n"
                       "         f'''(y por tanto la crítica) con probabilidad {p_ruta2_critica:.4f}, el {p_ruta2_critica:.1%} de las veces.''')"))
    celdas += apartado(3)
    celdas.append(code("duracion_media = proyecto.duration(datos_proyecto['duration'])\n"
                       "sigma_pert = np.sqrt(v1 if mu1 >= mu2 else v2)\n"
                       "p_pert = norm.cdf(plazo, duracion_media, sigma_pert)\n"
                       "dA, dH = float(datos_proyecto.loc['A', 'duration']), float(datos_proyecto.loc['H', 'duration'])\n"
                       "limite = plazo - dA - dH   # lo que puede durar cada cadena intermedia\n"
                       "p1 = norm.cdf(limite, mu1 - dA - dH, np.sqrt(v1))\n"
                       "p2 = norm.cdf(limite, mu2 - dA - dH, np.sqrt(v2))\n"
                       "p_exacta = p1 * p2\n"
                       f"cad1, cad2 = ' – '.join({nombres1!r}), ' – '.join({nombres2!r})\n"
                       "Markdown(f'''(a) PERT, con la ruta crítica en media, T ~ N({duracion_media}; √{v1 if mu1 >= mu2 else v2:.3f} = {sigma_pert:.3f}): '''\n"
                       "         f'''{explica_probabilidad(duracion_media, sigma_pert, plazo)}.\\n\\n'''\n"
                       "         f'''(b) Exacta: A y H son fijas, así que el proyecto termina a tiempo si cada cadena intermedia dura como mucho '''\n"
                       "         f'''{plazo} − {dA:g} − {dH:g} = {limite:g} semanas. La cadena {cad1} (Y₁) tiene media {mu1 - dA - dH:g} y σ = {np.sqrt(v1):.3f}; '''\n"
                       "         f'''la cadena {cad2} (Y₂), media {mu2 - dA - dH:g} y σ = {np.sqrt(v2):.3f}:\\n\\n'''\n"
                       "         f'''- {explica_probabilidad(mu1 - dA - dH, np.sqrt(v1), limite, nombre=\"Y₁\")}\\n'''\n"
                       "         f'''- {explica_probabilidad(mu2 - dA - dH, np.sqrt(v2), limite, nombre=\"Y₂\")}\\n\\n'''\n"
                       "         f'''Como las dos cadenas son independientes, P(T ≤ {plazo}) = P(Y₁ ≤ {limite:g}) · P(Y₂ ≤ {limite:g}) = {p1:.4f} × {p2:.4f} = **{p_exacta:.4f}**.\\n\\n'''\n"
                       "         f'''**Respuesta:** (a) {p_pert:.4f} con el método PERT; (b) {p_exacta:.4f} de forma exacta.''')"))
    celdas += apartado(4)
    celdas.append(code("Markdown(f'''**Respuesta:** es mayor la del método PERT ({p_pert:.4f} frente a {p_exacta:.4f}, {p_pert - p_exacta:.4f} más): PERT es optimista. '''\n"
                       "         'Ese sentido del error es general: la duración del proyecto es el máximo de todas las rutas, que en cada escenario es mayor o igual '\n"
                       "         'que la duración de la ruta crítica sola, luego P(máx ≤ t) ≤ P(ruta crítica ≤ t). El error crece cuando hay rutas casi críticas '\n"
                       "         f'''con varianza apreciable, como aquí: {txt2} dura sólo {plural(mu1 - mu2, \"semana\")} menos en media y tiene más varianza ({v2:.3f} frente a {v1:.3f}).''')"))
    return celdas, f"dos cadenas paralelas de {n1} y {n2} actividades; P(ruta no crítica resulta la más larga)"


VARIANTES = [
    Variante("normal", 1, "varianza por la ruta crítica, P(T ≤ t), P(a < T ≤ b), duración para un percentil",
             lambda rng: _base(1, rng)),
    Variante("dos-rutas-criticas", 2, "ídem con dos rutas críticas de varianzas distintas: cuál usar", lambda rng: _base(2, rng)),
    Variante("dos-rutas-criticas-b", 2, "ídem con otro proyecto", lambda rng: _base(3, rng)),
    Variante("casi-critica-variable", 3, "la segunda ruta es mucho más variable que la crítica: cota de probabilidad y Monte Carlo",
             _casi_critica),
    Variante("varianza-maxima", 3, "inverso: σ máxima admisible de una actividad para cumplir el plazo con probabilidad dada",
             _varianza_maxima),
    Variante("diagnostico", 2, "resolución entregada con σ sumadas y con la ruta equivocada: localizar y corregir",
             _diagnostico),
    Variante("hito", 2, "probabilidad de alcanzar un hito intermedio: cadenas hasta la actividad, media y varianza del hito frente a las del proyecto",
             _hito),
    Variante("ser-critica", 3, "dos cadenas paralelas: probabilidad de que la no crítica resulte la más larga y probabilidad exacta del plazo frente a PERT",
             _ser_critica),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
