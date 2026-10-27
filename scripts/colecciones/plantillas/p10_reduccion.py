"""Colección 10 — Reducción de la duración con mínimo coste: CUR, algoritmo
de Ackoff-Sasieni, costes indirectos y duración óptima, penalizaciones."""
import contextlib
import io

import numpy as np
import pandas as pd

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX)

META = dict(slug="10_reduccion", prefijo="Reduccion",
            titulo="Reducción de la duración con mínimo coste",
            descripcion="Coste unitario de reducción, algoritmo de Ackoff-Sasieni paso a paso, costes "
                        "indirectos y duración de coste mínimo, reducción frente a penalización.",
            n_defecto=3)


# Celda auxiliar (se define en el cuaderno): rutas en orden de ejecución con su suma y narración de
# cada paso de Ackoff-Sasieni (rutas críticas, opciones con su coste, elección, cuánto se reduce y por
# qué se detiene, coste acumulado).
_AUX_REDUCCION = r'''
from itertools import product as _product


def _num(x):
    return f'{float(x):g}'.replace('-', '−')


def rangos(numeros):
    """[1, 2, 3, 5] → «1 a 3 y 5»."""
    ns, partes = sorted(int(x) for x in numeros), []
    for x in ns:
        if partes and x == partes[-1][1] + 1:
            partes[-1][1] = x
        else:
            partes.append([x, x])
    return lista_y([f'{a}' if a == b else (f'{a} y {b}' if b == a + 1 else f'{a} a {b}') for a, b in partes])


def tabla_rutas(proyecto, valores, nombre='duración', decimales=0, base=None):
    """Cada ruta con sus actividades en orden de ejecución y su duración como suma término a término."""
    R = proyecto.path_matrix(dummies=False)
    cal = proyecto.calendar(base if base is not None else valores)
    filas = {}
    for r in R.index:
        acts = ordena_ruta(list(R.columns[R.loc[r] == 1]), cal)
        total = float(sum(float(valores[a]) for a in acts))
        filas[r] = {'actividades': ' – '.join(acts),
                    f'{nombre} = suma': ' + '.join(f'{a}({float(valores[a]):.{decimales}f})' for a in acts)
                                        + f' = {total:.{decimales}f}',
                    nombre: total}
    t = pd.DataFrame(filas).T
    t[nombre] = t[nombre].astype(float)
    return t


def opciones_ackoff(conjuntos, d, topes, cur):
    """Combinaciones mínimas (una actividad con margen por ruta crítica, sin sobrantes) y su coste."""
    elegibles = [[a for a in sorted(s) if d[a] > topes[a] and np.isfinite(cur[a])] for s in conjuntos]
    if any(not e for e in elegibles):
        return []
    combos = {frozenset(c) for c in _product(*elegibles)}
    minimos = [c for c in combos if not any(o < c for o in combos)]
    return sorted(((tuple(sorted(c)), float(sum(cur[a] for a in c))) for c in minimos),
                  key=lambda x: (x[1], len(x[0]), x[0]))


def info_ackoff(proyecto, duraciones, topes, cur, pasos):
    """Estado de cada paso: duración, rutas críticas, opciones, elegidas y duraciones tras el paso."""
    R = proyecto.path_matrix(dummies=False)
    d = pd.Series(duraciones, dtype=float).copy()
    topes, cur = pd.Series(topes, dtype=float), pd.Series(cur, dtype=float)
    info = []
    for k in sorted(pasos):
        L = R @ d[R.columns]
        T = float(L.max())
        crit = [r for r in L.index if np.isclose(L[r], T)]
        conj = [set(R.columns[R.loc[r] == 1]) for r in crit]
        info.append(dict(T=T, crit=crit, opciones=opciones_ackoff(conj, d, topes, cur),
                         elegidas=tuple(sorted(pasos[k])), d_antes=d.copy()))
        for a in pasos[k]:
            d[a] -= 1
        info[-1]['d_despues'] = d.copy()
    return info


def describe_ruta(proyecto, r, base):
    """«Route_3 (A – C – F)»: actividades de la ruta en orden de ejecución."""
    R = proyecto.path_matrix(dummies=False)
    return f"{r} ({' – '.join(ordena_ruta(list(R.columns[R.loc[r] == 1]), proyecto.calendar(base)))})"


def rutas_sin_margen(proyecto, d, topes, cur):
    """Rutas críticas (con las duraciones d) en las que ninguna actividad puede acortarse más."""
    R = proyecto.path_matrix(dummies=False)
    d = pd.Series(d, dtype=float)
    L = R @ d[R.columns]
    crit = [r for r in L.index if np.isclose(L[r], L.max())]
    return [r for r in crit if not any(d[a] > topes[a] and np.isfinite(cur[a]) for a in R.columns[R.loc[r] == 1])]


def narra_ackoff(proyecto, duraciones, topes, cur, pasos, unidad='semana', moneda='k€', max_opciones=5):
    """Markdown con el razonamiento de cada paso (o grupo de pasos iguales) de Ackoff-Sasieni."""
    R = proyecto.path_matrix(dummies=False)
    topes, cur = pd.Series(topes, dtype=float), pd.Series(cur, dtype=float)
    orden = ordena_ruta(list(R.columns), proyecto.calendar(pd.Series(duraciones)))
    pos = {a: i for i, a in enumerate(orden)}
    ruta = lambda r: f"{r} ({' – '.join(sorted(R.columns[R.loc[r] == 1], key=pos.get))})"

    def criticas(dd):
        L = R @ dd[R.columns]
        return float(L.max()), [r for r in L.index if np.isclose(L[r], L.max())]

    def opc(acts, coste):
        if len(acts) == 1:
            return f'{acts[0]} ({_num(coste)})'
        return f"{' + '.join(acts)} ({' + '.join(_num(cur[a]) for a in acts)} = {_num(coste)})"

    info = info_ackoff(proyecto, duraciones, topes, cur, pasos)
    if not info:
        return Markdown('No se ha dado ningún paso: ninguna combinación de actividades con margen acorta todas las rutas críticas.')
    grupos = []
    for x in info:
        if grupos and grupos[-1][-1]['elegidas'] == x['elegidas'] and grupos[-1][-1]['crit'] == x['crit']:
            grupos[-1].append(x)
        else:
            grupos.append([x])
    lineas, acum, paso = [], 0.0, 1
    for g in grupos:
        x0, n = g[0], len(g)
        el = x0['elegidas']
        c = float(sum(cur[a] for a in el))
        acum += n * c
        crit, ops = x0['crit'], x0['opciones']
        cab = (f'**Paso {paso}**' if n == 1 else f'**Pasos {paso} y {paso + 1}**' if n == 2
               else f'**Pasos {paso} a {paso + n - 1}**')
        cab += f" ({plural(x0['T'], unidad)} → {plural(x0['T'] - n, unidad)})."
        t_rutas = (f'Ruta crítica: {ruta(crit[0])}.' if len(crit) == 1
                   else f'Rutas críticas: {lista_y([ruta(r) for r in crit])}.')
        t_ops = '; '.join(opc(a, cc) for a, cc in ops[:max_opciones])
        if len(ops) > max_opciones:
            t_ops += f' y {len(ops) - max_opciones} más, de coste igual o mayor'
        quien = 'la ruta crítica' if len(crit) == 1 else 'todas las rutas críticas a la vez'
        t_ops = f'Opciones que acortan {quien} (CUR en {moneda}/{unidad}): {t_ops}.'
        if len(crit) == 1:
            tipo = 'la actividad de la ruta crítica con margen y menor CUR'
        elif len(el) == 1:
            tipo = 'actividad común a todas las rutas críticas'
        else:
            tipo = 'una combinación que acorta todas las rutas críticas'
        t_el = f"Se elige **{' + '.join(el)}** ({tipo}), la opción más barata: {_num(c)} {moneda} por {unidad}"
        empates = [' + '.join(a) for a, cc in ops if np.isclose(cc, c) and a != el]
        if empates:
            t_el += (f" (empata con {lista_y(empates)}: cuestan lo mismo en este paso y el algoritmo "
                     "toma la primera que encuentra)")
        if len(crit) > 1 and len(el) > 1:
            comunes = [(a, cc) for a, cc in ops if len(a) == 1]
            if comunes:
                t_el += f"; la actividad común más barata, {comunes[0][0][0]}, cuesta {_num(comunes[0][1])}"
            else:
                t_el += '; no hay ninguna actividad común a todas las rutas críticas que tenga margen'
        d_fin = g[-1]['d_despues']
        motivos = []
        llegan = [a for a in el if d_fin[a] <= topes[a]]
        if llegan:
            motivos.append(f"{lista_y(f'{a} llega a su duración tope ({plural(topes[a], unidad)})' for a in llegan)}")
        T_n, crit_n = criticas(d_fin)
        nuevas = [r for r in crit_n if r not in crit]
        if nuevas:
            motivos.append(f"{lista_y([ruta(r) for r in nuevas])} {'alcanza' if len(nuevas) == 1 else 'alcanzan'} "
                           f"la duración del proyecto ({plural(T_n, unidad)}) y {'pasa' if len(nuevas) == 1 else 'pasan'} "
                           "a ser crítica" + ('' if len(nuevas) == 1 else 's'))
        dejan = [r for r in crit if r not in crit_n]
        if dejan:
            motivos.append(f"{lista_y(dejan)} se acorta{'' if len(dejan) == 1 else 'n'} dos {unidad}s en el paso "
                           f"y deja{'' if len(dejan) == 1 else 'n'} de ser crítica" + ('' if len(dejan) == 1 else 's'))
        if g is grupos[-1] and not motivos:
            motivos.append('se alcanza la reducción pedida')
        t_red = f"Se {'reduce' if n == 1 else 'reducen'} {plural(n, unidad)}" + (f": {'; '.join(motivos)}." if motivos else '.')
        t_coste = f"Sobrecoste: {n} × {_num(c)} = {_num(n * c)} {moneda}; acumulado: **{_num(acum)} {moneda}**."
        lineas.append(f'- {cab} {t_rutas} {t_ops} {t_el}. {t_red} {t_coste}')
        paso += n
    T_ini, T_fin = info[0]['T'], info[-1]['T'] - 1
    costes = [_num(sum(cur[a] for a in x['elegidas'])) for x in info]
    cierre = (f"**Resultado:** la duración pasa de {plural(T_ini, unidad)} a {plural(T_fin, unidad)} con un sobrecoste "
              f"de {' + '.join(costes)} = **{_num(acum)} {moneda}**.")
    T_n, crit_n = criticas(info[-1]['d_despues'])
    conj = [set(R.columns[R.loc[r] == 1]) for r in crit_n]
    if not opciones_ackoff(conj, info[-1]['d_despues'], topes, cur):
        bloq = [r for r, s in zip(crit_n, conj)
                if not any(info[-1]['d_despues'][a] > topes[a] and np.isfinite(cur[a]) for a in s)]
        cierre += (f" No puede reducirse más: en {lista_y([ruta(r) for r in bloq])} ninguna actividad tiene margen "
                   "(todas están en su duración tope o no son acelerables), así que esa ruta fija la duración mínima.")
    return Markdown('\n'.join(lineas) + '\n\n' + cierre)
'''


# Celda auxiliar de la variante con recursos: cadena crítica con recursos explicada y candidatos de
# cada paso de crashing_recursos con la duración replanificada de cada uno.
_AUX_RECURSOS = r'''
from itertools import combinations as _comb


def _predecesoras(datos):
    return {a: [p.strip() for p in str(datos.loc[a, 'predecessor']).split(',') if p.strip() not in ('', '---')]
            for a in datos.index}


def explica_cadena(cadena, res, datos, unidad='semana'):
    """Cadena crítica con recursos en orden de ejecución (→ precedencia, ⇒ recurso), su suma y las esperas."""
    preds = _predecesoras(datos)
    ini, fin = res.inicios, res.fines
    acts = sorted(cadena['actividades'], key=lambda a: (ini[a], fin[a], a))
    arcos = {(f['de'], f['a']): f['motivo'] for _, f in cadena['arcos'].iterrows()}
    texto = f'{acts[0]} ({int(ini[acts[0]])}–{int(fin[acts[0]])})'
    for x, y in zip(acts, acts[1:]):
        texto += (' ⇒ ' if arcos.get((x, y)) == 'recurso' else ' → ') + f'{y} ({int(ini[y])}–{int(fin[y])})'
    duraciones = [int(fin[a] - ini[a]) for a in acts]
    simple = (all((x, y) in arcos for x, y in zip(acts, acts[1:])) and int(ini[acts[0]]) == 0
              and int(fin[acts[-1]]) == int(res.duracion))
    suma = f" Duración = {' + '.join(str(d) for d in duraciones)} = {int(res.duracion)} {unidad}s." if simple else ''
    esperas = []
    for (x, y), motivo in sorted(arcos.items(), key=lambda kv: (ini[kv[0][1]], kv[0])):
        if motivo != 'recurso':
            continue
        listo = max([int(fin[p]) for p in preds[y]], default=0)
        espera = int(ini[y]) - listo
        if espera > 0:
            esperas.append(f'{y} podría empezar en {listo} (cuando terminan sus predecesoras), pero espera '
                           f'{plural(espera, unidad)}, hasta {int(ini[y])}, a que {x} libere los operarios')
        else:
            esperas.append(f'{y} empieza en {int(ini[y])}, justo cuando {x} le libera los operarios')
    t_esp = ('Esperas por recurso: ' + '; '.join(esperas) + '.') if esperas else \
        'No hay arcos de recurso: la cadena es un camino del grafo.'
    return f'{texto} (→ precedencia, ⇒ espera por recurso).{suma} {t_esp}'


def candidatos_crashing(proyecto, base, limite, topes, cur, elegidos, extra=0, unidad='semana'):
    """Para cada paso: cadena crítica con recursos, cada candidato (actividades reducibles de la cadena,
    sueltas y por parejas) con su coste y la duración que da al replanificar, y el elegido."""
    red = {a: 0 for a in base.index}
    for paso, el in enumerate(elegidos, 1):
        el = tuple(sorted(el))
        datos = base.copy()
        datos['duration'] = [int(base.loc[a, 'duration']) - red[a] for a in base.index]
        datos['operarios'] = [base.loc[a, 'operarios'] + extra * red[a] for a in base.index]
        cad = proyecto.cadena_critica_recursos(datos, 'duration', ['operarios'], limite)
        T = int(cad['duracion'])
        reducibles = [a for a in sorted(cad['actividades']) if datos.loc[a, 'duration'] > topes[a] and np.isfinite(cur[a])]
        cands = [(a,) for a in reducibles] + list(_comb(reducibles, 2))
        if el not in cands:
            cands.append(el)
        filas = []
        for c in cands:
            d2 = datos.copy()
            for a in c:
                d2.loc[a, 'duration'] -= 1
                d2.loc[a, 'operarios'] += extra
            coste = float(sum(cur[a] for a in c))
            if (d2['operarios'] > limite[0]).any():
                T2, txt = np.nan, f'necesita más de {limite[0]} operarios: no cabe'
            else:
                T2 = int(proyecto.asignar_multi(d2, 'duration', ['operarios'], limite, dibujar=False).duracion)
                txt = f'acorta a {T2}' if T2 < T else 'no acorta'
            filas.append({'candidato': ' + '.join(c),
                          'coste (k€)': _num(coste) if len(c) == 1 else f"{' + '.join(_num(cur[a]) for a in c)} = {_num(coste)}",
                          'duración replanificada': '—' if np.isnan(T2) else T2, 'resultado': txt,
                          'elegido': '✔' if c == el else '', '_c': coste})
        tabla = pd.DataFrame(filas).sort_values(['_c', 'candidato']).set_index('candidato')
        baratos = tabla[tabla['_c'] < tabla.loc[' + '.join(el), '_c'] - 1e-9]
        display(Markdown(f"**Paso {paso}** (duración con recursos {plural(T, unidad)}). Cadena crítica con recursos: "
                         f"{lista_y(sorted(cad['actividades']))}; reducibles en ella: {lista_y(reducibles)}. Candidatos, de más "
                         "barato a más caro:"))
        display(tabla.drop(columns='_c'))
        t = f"Se elige **{' + '.join(el)}** ({_num(tabla.loc[' + '.join(el), '_c'])} k€), el candidato más barato que de verdad acorta el proyecto replanificado"
        if len(baratos):
            t += f"; los más baratos ({lista_y(list(baratos.index))}) no lo acortan o no caben en el límite"
        display(Markdown(t + '.'))
        for a in el:
            red[a] += 1
'''


def _pasos_ackoff(info, D, tope, ucr, reduction):
    with contextlib.redirect_stdout(io.StringIO()):
        tabla, best, dur, avail = info["proyecto"].ackoff(D, tope, ucr, reduction=reduction, report=False)
    pasos = [(acts, float(sum(ucr[a] for a in acts))) for _, acts in sorted(best.items())]
    return pasos, dur


def _base(k, rng):
    variante = (k - 1) % 3 + 1
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 12 and (i["D"] >= 2).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"], n_inf=(2 if variante == 3 else 0))
        ucr_ack = ucr.replace(0.0, np.inf)
        margen = int((i["D"] - tope).sum())
        if margen < 4:
            return False
        try:
            pasos, dur = _pasos_ackoff(i, i["D"], tope, ucr_ack, reduction=100)
        except Exception:
            return False
        if len(pasos) < 4:
            return False
        if variante == 1 and not any(len(acts) >= 2 for acts, _ in pasos[:3]):
            return False
        costes = [c for _, c in pasos]
        if variante == 2 and not (costes[0] < costes[-1]):
            return False
        estado.update(tope=tope, cn=cn, ct=ct, ucr=ucr, ucr_ack=ucr_ack, pasos=pasos, dur_final=dur)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, cn, ct, ucr, pasos = estado["tope"], estado["cn"], estado["ct"], estado["ucr"], estado["pasos"]
    costes = [c for _, c in pasos]
    T_tope = T - len(pasos)

    celdas = [md("# Enunciado")]
    if variante == 1:
        r = 3
        df = P.tabla_base(info, duration=D, cap_duration=tope, cost_normal=cn, cost_cap=ct)
        celdas.append(md("Considere el proyecto cuyo cuadro de prelaciones se adjunta, con la duración normal y la "
                         "duración tope (mínima) de cada actividad, en semanas, y el coste (en k€) de ejecutarla "
                         "en cada una de esas duraciones. Se supone coste lineal entre ambas."))
        pregs = ["Calcule el coste unitario de reducción (CUR) de cada actividad.",
                 "Determine la duración del proyecto, su camino crítico y el coste del proyecto con las "
                 "duraciones normales.",
                 f"Reduzca la duración del proyecto en {r} semanas con el mínimo sobrecoste aplicando el "
                 "algoritmo de Ackoff-Sasieni paso a paso. Indique en cada paso qué actividad(es) se acortan "
                 "y por qué.",
                 "Indique el coste final del proyecto y su(s) nuevo(s) camino(s) crítico(s). ¿Por qué en "
                 "algún paso ha hecho falta acortar más de una actividad a la vez?"]
    elif variante == 2:
        c_ind = int(np.ceil(float(np.median(costes))))
        if c_ind in costes:
            c_ind += 1
        df = P.tabla_base(info, duration=D, cap_duration=tope, cost_normal=cn, cost_cap=ct)
        celdas.append(md(f"Considere el proyecto cuyo cuadro de prelaciones se adjunta, con la duración normal y "
                         f"la duración tope de cada actividad (semanas) y el coste directo (k€) de ejecutarla en "
                         f"cada una. Además, el proyecto tiene unos costes indirectos (oficina, alquileres, "
                         f"seguros) de {c_ind} k€ por semana de duración."))
        pregs = ["Calcule el coste unitario de reducción (CUR) de cada actividad.",
                 "Determine la duración normal del proyecto y su coste total (directo más indirecto).",
                 "Aplique el algoritmo de Ackoff-Sasieni hasta alcanzar la duración tope del proyecto, anotando "
                 "en cada paso el sobrecoste directo.",
                 "Construya la curva de coste total frente a duración y determine la duración de coste mínimo. "
                 "¿Qué regla de parada se deduce comparando el CUR del paso con el coste indirecto semanal?",
                 "¿Por qué no se puede reducir el proyecto por debajo de su duración tope aunque queden "
                 "actividades con margen de reducción?"]
        estado["c_ind"] = c_ind
    else:
        r_lim = min(len(pasos), int(rng.integers(2, 4)))
        T_lim = T - r_lim
        pen = int(np.ceil(float(np.median(costes[:r_lim + 1]))))
        if pen in costes:
            pen += 1
        df = P.tabla_base(info, duration=D, cap_duration=tope, ucr=ucr.replace(0.0, np.inf))
        no_red = sorted(df.index[df["ucr"] == np.inf].tolist())
        celdas.append(md(f"El contrato fija la entrega del proyecto en la semana {T_lim}, con una penalización de "
                         f"{pen} k€ por cada semana de retraso. Se adjuntan las prelaciones, la duración normal y "
                         f"la tope de cada actividad (semanas) y su coste unitario de reducción (k€/semana). "
                         f"Las actividades {', '.join(no_red)} no pueden acelerarse."))
        pregs = ["Determine la duración del proyecto con las duraciones normales y la penalización a la que se "
                 "expone si no se reduce.",
                 f"Aplique el algoritmo de Ackoff-Sasieni para llevar el proyecto a la semana {T_lim}, anotando "
                 "el sobrecoste de cada paso.",
                 "Compare en cada paso el sobrecoste con la penalización evitada y decida cuántas semanas "
                 "conviene reducir. ¿Cuál es el coste total mínimo (sobrecoste más penalización)?",
                 "¿Qué habría cambiado si las actividades no acelerables hubieran sido críticas? Compruébelo "
                 "sobre el camino crítico."]
        estado.update(T_lim=T_lim, pen=pen, r_lim=r_lim)
    celdas.append(preguntas(pregs))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()"))
    celdas.append(code(_AUX_REDUCCION))
    ap = 1
    if variante in (1, 2):
        celdas += apartado(ap); ap += 1
        celdas.append(md("$CUR = \\dfrac{\\text{coste tope} - \\text{coste normal}}{\\text{duración normal} - "
                         "\\text{duración tope}}$. Las actividades cuya duración tope coincide con la normal no "
                         "pueden acortarse: su CUR es infinito."))
        celdas.append(code("reducible = datos_proyecto['duration'] > datos_proyecto['cap_duration']\n"
                           "datos_proyecto['ucr'] = np.where(reducible,\n"
                           "                                 (datos_proyecto['cost_cap'] - datos_proyecto['cost_normal']) / (datos_proyecto['duration'] - datos_proyecto['cap_duration']).replace(0, np.nan),\n"
                           "                                 np.inf)\n"
                           "def _cur_txt(f):\n"
                           "    if f['duration'] == f['cap_duration']:\n"
                           "        return 'duración tope = normal: no se puede acortar → ∞'\n"
                           "    return (f\"({f['cost_cap']:g} − {f['cost_normal']:g}) / ({f['duration']:g} − {f['cap_duration']:g}) = \"\n"
                           "            f\"{f['cost_cap'] - f['cost_normal']:g} / {f['duration'] - f['cap_duration']:g} = {(f['cost_cap'] - f['cost_normal']) / (f['duration'] - f['cap_duration']):g}\")\n"
                           "pd.DataFrame({'duración normal': datos_proyecto['duration'], 'duración tope': datos_proyecto['cap_duration'],\n"
                           "              'coste normal (k€)': datos_proyecto['cost_normal'], 'coste tope (k€)': datos_proyecto['cost_cap'],\n"
                           "              'CUR = Δcoste / Δduración (k€/semana)': datos_proyecto.apply(_cur_txt, axis=1)})"))
        celdas.append(code("acelerables = datos_proyecto.index[np.isfinite(datos_proyecto['ucr'])].tolist()\n"
                           "no_acelerables = [a for a in datos_proyecto.index if a not in acelerables]\n"
                           "Markdown('**Respuesta:** CUR (k€/semana): ' + '; '.join(f'{a} = {datos_proyecto.loc[a, \"ucr\"]:g}' for a in acelerables)\n"
                           "         + (f'; {lista_y(no_acelerables)}: ∞ (no acelerable{\"\" if len(no_acelerables) == 1 else \"s\"}).' if no_acelerables else '.'))"))
    celdas += apartado(ap); ap += 1
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales y el camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario con las duraciones normales"\n#| label: tbl-calendario\n'
                       "calendario_normal = proyecto.calendar(datos_proyecto['duration'])\n"
                       "calendario_normal[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(md("La duración del proyecto es la de su ruta más larga. Conviene tener a la vista la duración de "
                     "*todas* las rutas: al acortar la crítica, las que están a una o dos semanas de ella serán las "
                     "primeras en volverse críticas y obligarán a acortar varias actividades a la vez."))
    celdas.append(code('#| tbl-cap: "Duración de cada ruta con las duraciones normales"\n#| label: tbl-rutasNormal\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "rutas_normal = tabla_rutas(proyecto, datos_proyecto['duration'])\n"
                       "rutas_normal['margen hasta la crítica'] = (rutas_normal['duración'].max() - rutas_normal['duración']).astype(int)\n"
                       "rutas_normal"))
    if variante == 1:
        celdas.append(code("duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                           "coste_normal = int(datos_proyecto['cost_normal'].sum())\n"
                           "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                           "Markdown(f'**Respuesta:** con las duraciones normales el proyecto dura {duracion_normal} semanas (camino crítico arriba) y su coste es la suma '\n"
                           "         f'de los costes normales: {explica_suma(datos_proyecto[\"cost_normal\"], \"coste normal\", \"k€\")}.')"))
    elif variante == 2:
        celdas.append(code(f"coste_indirecto_semanal = {c_ind}\n"
                           "duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                           "coste_directo_normal = int(datos_proyecto['cost_normal'].sum())\n"
                           "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                           "display(Markdown(f'Coste directo con las duraciones normales: {explica_suma(datos_proyecto[\"cost_normal\"], \"coste directo\", \"k€\")}.'))\n"
                           "Markdown(f'**Respuesta:** duración normal {duracion_normal} semanas. Coste total = directo + indirecto = {coste_directo_normal} + '\n"
                           "         f'{coste_indirecto_semanal} × {duracion_normal} = {coste_directo_normal} + {coste_indirecto_semanal * duracion_normal} = '\n"
                           "         f'**{coste_directo_normal + coste_indirecto_semanal * duracion_normal} k€**.')"))
    else:
        celdas.append(code(f"plazo_contrato, penalizacion_semanal = {estado['T_lim']}, {estado['pen']}\n"
                           "duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                           "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                           "retraso = max(0, duracion_normal - plazo_contrato)\n"
                           "Markdown(f'**Respuesta:** con las duraciones normales el proyecto dura {duracion_normal} semanas, {duracion_normal} − {plazo_contrato} = '\n"
                           "         f'{plural(retraso, \"semana\")} más que el plazo: penalización de {retraso} × {penalizacion_semanal} = **{retraso * penalizacion_semanal} k€**.')"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("En cada paso del algoritmo de Ackoff-Sasieni se acorta una semana la ruta crítica "
                     "(o las rutas críticas) eligiendo, entre las actividades críticas con margen, la combinación de menor "
                     "CUR que reduzca *todas* las rutas críticas a la vez. Cuando hay varias rutas críticas "
                     "sin actividades comunes baratas hay que acortar una actividad de cada una, y el coste "
                     "del paso es la suma. La tabla muestra, para cada ruta, el CUR de sus actividades y su "
                     "duración tras cada paso; las últimas filas, el margen de reducción que le queda a cada actividad."))
    if variante == 1:
        red = "reduction=3"
    elif variante == 2:
        red = "reduction=100"
    else:
        red = f"reduction={estado['r_lim']}"
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       f"    costs=datos_proyecto['ucr'], {red}, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(md("Lo que se compara en cada paso: las rutas críticas en ese momento, las opciones que acortan "
                     "todas a la vez (una actividad común con margen, o una pareja con una actividad de cada ruta) "
                     "con su coste, y la elegida (la más barata)."))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso de Ackoff-Sasieni"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(md("Razonamiento de cada paso (los pasos consecutivos que acortan lo mismo se agrupan): rutas "
                     "críticas, opciones con su coste, cuál se elige y por qué, cuánto se reduce y hasta dónde, y el "
                     "sobrecoste acumulado."))
    celdas.append(code("narra_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],\n"
                       "                        'sobrecoste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},\n"
                       "                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))\n"
                       "resumen['duración'] = duracion_normal - resumen.index\n"
                       "resumen['sobrecoste acumulado'] = resumen['sobrecoste del paso'].cumsum()\n"
                       "resumen"))
    celdas.append(md("Comprobación paso a paso: la duración de cada ruta tras cada paso es la que figura en las "
                     "columnas numeradas de la tabla de Ackoff-Sasieni; tras el último paso, las rutas que empatan "
                     "en la duración máxima son las críticas del proyecto reducido."))
    celdas.append(code('#| tbl-cap: "Duración de cada ruta tras la reducción"\n#| label: tbl-rutasReducidas\n'
                       "rutas_reducidas = tabla_rutas(proyecto, duraciones_reducidas, base=datos_proyecto['duration'])\n"
                       "rutas_reducidas['crítica'] = np.where(rutas_reducidas['duración'] == rutas_reducidas['duración'].max(), 'sí', '')\n"
                       "rutas_reducidas"))
    celdas += apartado(ap); ap += 1
    if variante == 1:
        celdas.append(code("duracion_reducida = proyecto.duration(duraciones_reducidas)\n"
                           "sobrecoste = float(resumen['sobrecoste del paso'].sum())\n"
                           "coste_final = coste_normal + sobrecoste\n"
                           "display(explica_camino_critico(proyecto, duraciones_reducidas))\n"
                           "info = info_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)\n"
                           "motivos = []\n"
                           "for i, x in enumerate(info):\n"
                           "    if len(x['elegidas']) < 2:\n"
                           "        continue\n"
                           "    comunes = [(a, c) for a, c in x['opciones'] if len(a) == 1]\n"
                           "    coste = sum(datos_proyecto.loc[a, 'ucr'] for a in x['elegidas'])\n"
                           "    por_que = (f'la actividad común más barata con margen, {comunes[0][0][0]}, costaba {_num(comunes[0][1])} k€, más que '\n"
                           "               if comunes else 'ninguna actividad común a todas ellas tenía margen, así que hubo que acortar ')\n"
                           "    motivos.append(f\"en el paso {i + 1} las rutas críticas eran {lista_y(x['crit'])} y {por_que}{' + '.join(x['elegidas'])} \"\n"
                           "                   f\"({' + '.join(_num(datos_proyecto.loc[a, 'ucr']) for a in x['elegidas'])} = {_num(coste)} k€)\")\n"
                           "Markdown(f'''**Respuesta:** el proyecto pasa a durar {duracion_reducida} semanas con un coste de {coste_normal} + {sobrecoste:g} = **{coste_final:g} k€**; "
                           "sus caminos críticos son los de la lista anterior. Hizo falta acortar varias actividades a la vez porque, al acortar la ruta crítica, otras rutas con poco margen "
                           "alcanzaron su duración y pasaron a ser críticas, y para reducir el proyecto hay que acortar *todas* las rutas críticas en el mismo paso: "
                           "{'; '.join(motivos)}. Una sola actividad basta únicamente si es común a todas las rutas críticas.''')"))
        celdas += fig_grafo(durations="duraciones_reducidas", cap="Grafo PERT con las duraciones reducidas", label="fig-pertReducido")
    elif variante == 2:
        celdas.append(md("La tabla recorre la curva paso a paso: cada paso suma su CUR al coste directo y ahorra una "
                         "semana de costes indirectos, así que el coste total varía en CUR − coste indirecto semanal."))
        celdas.append(code("curva = pd.DataFrame({'duración': [duracion_normal] + resumen['duración'].tolist(),\n"
                           "                      'CUR del paso': [np.nan] + resumen['sobrecoste del paso'].tolist(),\n"
                           "                      'coste directo': [coste_directo_normal] + (coste_directo_normal + resumen['sobrecoste acumulado']).tolist()})\n"
                           "curva['coste indirecto'] = coste_indirecto_semanal * curva['duración']\n"
                           "curva['coste total'] = curva['coste directo'] + curva['coste indirecto']\n"
                           "curva['variación del total = CUR − indirecto'] = ['—'] + [f'{c:g} − {coste_indirecto_semanal} = {_num(c - coste_indirecto_semanal)}' for c in resumen['sobrecoste del paso']]\n"
                           "curva = curva.set_index('duración')\n"
                           "limpia_tabla(curva, decimales=1)"))
        celdas.append(code('#| fig-cap: "Curva coste-duración del proyecto"\n#| label: fig-curvaCoste\n'
                           "ax = curva[['coste directo', 'coste indirecto', 'coste total']].plot(marker='o', figsize=(7, 4))\n"
                           "ax.set_ylabel('k€'); ax.set_xlabel('duración (semanas)'); ax.invert_xaxis();"))
        celdas.append(code("minimo = float(curva['coste total'].min())\n"
                           "optimas = [int(x) for x in curva.index[np.isclose(curva['coste total'], minimo)]]\n"
                           "duracion_optima = max(optimas)\n"
                           "cur_pasos = resumen['sobrecoste del paso']\n"
                           "baratos = [str(p) for p, c in cur_pasos.items() if c < coste_indirecto_semanal and not np.isclose(c, coste_indirecto_semanal)]\n"
                           "iguales = [str(p) for p, c in cur_pasos.items() if np.isclose(c, coste_indirecto_semanal)]\n"
                           "caros = [str(p) for p, c in cur_pasos.items() if c > coste_indirecto_semanal and not np.isclose(c, coste_indirecto_semanal)]\n"
                           "def _clase(ps, verbo_1, verbo_n, resto_1, resto_n):\n"
                           "    uno = len(ps) == 1\n"
                           "    return f\"{'el paso' if uno else 'los pasos'} {rangos(ps)} {verbo_1 if uno else verbo_n} {resto_1 if uno else resto_n}\"\n"
                           "clases = []\n"
                           "if baratos:\n"
                           "    clases.append(_clase(baratos, 'cuesta', 'cuestan', f'menos de {coste_indirecto_semanal} k€ y baja el total', f'menos de {coste_indirecto_semanal} k€ y bajan el total'))\n"
                           "if iguales:\n"
                           "    clases.append(_clase(iguales, 'cuesta', 'cuestan', f'exactamente {coste_indirecto_semanal} k€ y lo deja igual', f'exactamente {coste_indirecto_semanal} k€ y lo dejan igual'))\n"
                           "if caros:\n"
                           "    clases.append(_clase(caros, 'cuesta', 'cuestan', f'más de {coste_indirecto_semanal} k€ y lo sube', f'más de {coste_indirecto_semanal} k€ y lo suben'))\n"
                           "vuelve = bool(caros) and any(c < coste_indirecto_semanal for c in cur_pasos[cur_pasos.index > int(caros[0])])\n"
                           "if len(optimas) == 1:\n"
                           "    resp = f'el coste total es mínimo, **{minimo:g} k€**, con una duración de **{optimas[0]} semanas**'\n"
                           "else:\n"
                           "    resp = (f'el coste total mínimo, **{minimo:g} k€**, se alcanza con {lista_y(sorted(optimas, reverse=True), \"o\")} semanas (empate: esos pasos cuestan '\n"
                           "            f'lo mismo que el indirecto que ahorran); la más larga, **{duracion_optima} semanas**, consigue ese coste con menos aceleración')\n"
                           "regla = (f'Regla de parada: compensa dar un paso mientras su CUR sea menor que el coste indirecto semanal ({coste_indirecto_semanal} k€), da igual si es igual y no compensa si es mayor. '\n"
                           "         f'Aquí {lista_y(clases)}. ')\n"
                           "regla += ('Ojo: tras un paso más caro que el indirecto vuelve a haber uno más barato, así que no basta con parar en el primer paso caro: hay que mirar la curva completa.' if vuelve\n"
                           "          else 'Como los CUR de los pasos no decrecen, basta con parar en el primer paso cuyo CUR supera el coste indirecto semanal.')\n"
                           "Markdown(f'**Respuesta:** {resp}. {regla}')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("bloqueadas = rutas_sin_margen(proyecto, duraciones_reducidas, datos_proyecto['cap_duration'], datos_proyecto['ucr'])\n"
                           "con_margen = {a: int(m) for a, m in margen_restante.items() if m > 0 and np.isfinite(datos_proyecto.loc[a, 'ucr'])}\n"
                           "cal_normal = proyecto.calendar(datos_proyecto['duration'])\n"
                           "def _estado(a):\n"
                           "    if not np.isfinite(datos_proyecto.loc[a, 'ucr']):\n"
                           "        return f'{a} no acelerable'\n"
                           "    return f\"{a} en su tope, {int(duraciones_reducidas[a])}\"\n"
                           "detalle_bloq = '; '.join(describe_ruta(proyecto, r, datos_proyecto['duration']) + ' no tiene ninguna actividad con margen (' + '; '.join(\n"
                           "    _estado(a) for a in ordena_ruta(list(R.columns[R.loc[r] == 1]), cal_normal)) + ')' for r in bloqueadas)\n"
                           "if con_margen:\n"
                           "    uno = len(con_margen) == 1\n"
                           "    margen_txt = (('Sigue' if uno else 'Siguen') + ' teniendo margen ' + lista_y([f'{a} ({plural(m, \"semana\")})' for a, m in con_margen.items()])\n"
                           "                  + (', pero no está en esa ruta: acortarla' if uno else ', pero ninguna de ellas está en esa ruta: acortarlas') + ' no acorta la ruta más larga.')\n"
                           "else:\n"
                           "    margen_txt = 'Ya no le queda margen a ninguna actividad.'\n"
                           "Markdown(f'''**Respuesta:** la duración tope del proyecto es {proyecto.duration(duraciones_reducidas)} semanas ({duracion_normal} − {len(pasos)} pasos). "
                           "En ella, la ruta crítica {detalle_bloq}, así que no puede acortarse y fija la duración mínima del proyecto. "
                           "{margen_txt}''')"))
    else:
        celdas.append(md("Para cada número de semanas reducidas se compara el sobrecoste del paso con la penalización "
                         "de la semana de retraso que evita, y se calcula el coste total = sobrecoste acumulado + "
                         "penalización por las semanas que aún exceden el plazo."))
        celdas.append(code("sin_reducir = penalizacion_semanal * retraso\n"
                           "filas = {0: {'duración': duracion_normal, 'actividades acortadas': '—', 'sobrecoste del paso': '—', 'penalización evitada': '—',\n"
                           "             '¿conviene el paso?': '—', 'coste total = sobrecoste acumulado + penalización': f'0 + {penalizacion_semanal} × {retraso} = {sin_reducir}',\n"
                           "             'coste total': float(sin_reducir)}}\n"
                           "for p, f in resumen.iterrows():\n"
                           "    exceso = max(0, int(f['duración']) - plazo_contrato)\n"
                           "    evita = penalizacion_semanal if f['duración'] + 1 > plazo_contrato else 0\n"
                           "    total = float(f['sobrecoste acumulado']) + penalizacion_semanal * exceso\n"
                           "    c = float(f['sobrecoste del paso'])\n"
                           "    filas[p] = {'duración': int(f['duración']), 'actividades acortadas': f['actividades acortadas'], 'sobrecoste del paso': f'{c:g}',\n"
                           "                'penalización evitada': evita, '¿conviene el paso?': 'sí' if c < evita else ('indiferente' if c == evita else 'no'),\n"
                           "                'coste total = sobrecoste acumulado + penalización': f\"{f['sobrecoste acumulado']:g} + {penalizacion_semanal} × {exceso} = {total:g}\",\n"
                           "                'coste total': total}\n"
                           "decision = pd.DataFrame(filas).T\n"
                           "decision.index.name = 'semanas reducidas'\n"
                           "decision.drop(columns='coste total')"))
        celdas.append(code("totales = decision['coste total'].astype(float)\n"
                           "coste_mejor = float(totales.min())\n"
                           "optimos = [int(r) for r in totales.index[np.isclose(totales, coste_mejor)]]\n"
                           "mejor = optimos[0]\n"
                           "pasos_txt = '; '.join(f\"paso {p}: {f['sobrecoste del paso']} k€ frente a {f['penalización evitada']} k€ evitados ({f['¿conviene el paso?']})\"\n"
                           "                      for p, f in decision.iloc[1:].iterrows())\n"
                           "if len(optimos) == 1:\n"
                           "    concl = f'conviene reducir **{plural(mejor, \"semana\")}**, hasta {duracion_normal - mejor} semanas, con un coste total mínimo de **{coste_mejor:g} k€**'\n"
                           "else:\n"
                           "    concl = (f'el coste total mínimo, **{coste_mejor:g} k€**, se obtiene reduciendo {lista_y(optimos, \"o\")} semanas (empate); lo razonable es reducir '\n"
                           "             f'**{plural(mejor, \"semana\")}**, que alcanza ese coste con menos aceleración')\n"
                           "Markdown(f'**Respuesta:** sin reducir se pagarían {sin_reducir} k€ de penalización. Comparando cada paso con la penalización que evita ({pasos_txt}), '\n"
                           "         f'{concl} (frente a {sin_reducir} k€ sin reducir).')"))
        celdas += apartado(ap); ap += 1
        celdas.append(code("no_acelerables = datos_proyecto.index[np.isinf(datos_proyecto['ucr'])].tolist()\n"
                           "rutas_crit = proyecto.critical_path(datos_proyecto['duration'])\n"
                           "criticas = sorted(set(a for acts in rutas_crit.values() for a in acts))\n"
                           "en_critico = [a for a in no_acelerables if a in criticas]\n"
                           "acelerables_criticas = [a for a in criticas if a not in no_acelerables]\n"
                           "_, _, d_min, _ = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                           "                                 costs=datos_proyecto['ucr'], reduction=100, report=False)\n"
                           "T_min = proyecto.duration(d_min)\n"
                           "bloqueadas = rutas_sin_margen(proyecto, d_min, datos_proyecto['cap_duration'], datos_proyecto['ucr'])\n"
                           "if en_critico:\n"
                           "    t1 = (f'de las actividades no acelerables ({lista_y(no_acelerables)}), {lista_y(en_critico)} {\"está\" if len(en_critico) == 1 else \"están\"} en el camino crítico. '\n"
                           "          f'Aun así se puede reducir, porque el camino crítico tiene otras actividades acelerables ({lista_y(acelerables_criticas)}) y el algoritmo elige entre ellas.')\n"
                           "else:\n"
                           "    t1 = f'ninguna de las actividades no acelerables ({lista_y(no_acelerables)}) está en el camino crítico, así que no han limitado la reducción.'\n"
                           "rutas_bloq = lista_y([describe_ruta(proyecto, r, datos_proyecto['duration']) for r in bloqueadas])\n"
                           "t2 = (f'Lo que cambia una actividad no acelerable en el camino crítico es el límite: no aporta margen. Si se aplica Ackoff-Sasieni hasta el final, el proyecto no baja de '\n"
                           "      f'**{T_min} semanas**, porque en {rutas_bloq} ya no queda ninguna actividad con margen (las acelerables están en su tope y las demás no se pueden acelerar). '\n"
                           "      f'Si todas las actividades críticas fueran no acelerables, el proyecto no podría reducirse nada y habría que pagar la penalización completa, {retraso * penalizacion_semanal} k€.')\n"
                           "Markdown(f'**Respuesta:** {t1} {t2}')"))
    sub = {1: f"{n} actividades, reducción de {3} semanas", 2: f"{n} actividades, costes indirectos de {estado.get('c_ind')} k€/semana",
           3: f"{n} actividades, plazo {estado.get('T_lim')} con penalización"}[variante]
    return celdas, sub


# ---------------------------------------------------------------------------
# Variantes adicionales
# ---------------------------------------------------------------------------
def _barata_no_optima(rng):
    """Con dos rutas críticas, la actividad crítica más barata no sirve si no es común a las dos."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and i["n_criticas"] == 2 and i["T"] >= 12 and (i["D"] >= 2).all()):
            return False
        Pm, rc = i["P"], i["rutas_criticas"]
        comunes = set(Pm.columns[(Pm.loc[rc[0]] == 1) & (Pm.loc[rc[1]] == 1)])
        solo1 = set(Pm.columns[(Pm.loc[rc[0]] == 1) & (Pm.loc[rc[1]] == 0)])
        solo2 = set(Pm.columns[(Pm.loc[rc[1]] == 1) & (Pm.loc[rc[0]] == 0)])
        if not comunes or not solo1 or not solo2:
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        # forzar: la más barata de todas las críticas está sólo en una ruta; y la pareja barata cuesta menos que la común más barata o más
        barata = sorted(solo1 | solo2, key=lambda a: ucr[a])[0]
        if np.isinf(ucr[barata]):
            return False
        ucr[barata] = 1.0
        tope[barata] = min(tope[barata], int(i["D"][barata]) - 1)
        if tope[barata] < 1:
            return False
        for a in comunes:
            if not np.isinf(ucr[a]):
                ucr[a] = max(ucr[a], 4.0)
        try:
            pasos, _ = _pasos_ackoff(i, i["D"], tope, ucr, reduction=2)
        except Exception:
            return False
        if len(pasos) < 2:
            return False
        estado.update(tope=tope, ucr=ucr, barata=barata, comunes=comunes, pasos=pasos, rc=rc)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, ucr, barata, comunes, pasos = estado["tope"], estado["ucr"], estado["barata"], estado["comunes"], estado["pasos"]
    df = P.tabla_base(info, duration=D, cap_duration=tope, ucr=ucr)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste unitario "
                 "de reducción (k€/semana) se adjunta. Las actividades con CUR infinito no pueden acelerarse.")]
    celdas.append(preguntas([
        "Determine la duración del proyecto y sus rutas críticas. ¿Qué actividades son comunes a todas ellas?",
        f"La actividad crítica más barata de acelerar es {barata}. Un compañero propone acortarla una semana para "
        "reducir el proyecto una semana. Prediga, sin calcular, si funcionará.",
        f"Compruébelo: acorte {barata} una semana y recalcule la duración.",
        "Aplique el algoritmo de Ackoff-Sasieni para reducir el proyecto dos semanas con mínimo coste y explique "
        "por qué en cada paso elige lo que elige (actividad común frente a pareja de actividades)."]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)"))
    celdas.append(code(_AUX_REDUCCION))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales y las rutas críticas", label="fig-pert")
    celdas.append(code("duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "comunes = set.intersection(*[set(a) for a in rutas_criticas.values()])\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "orden_comunes = ordena_ruta(sorted(comunes), proyecto.calendar(datos_proyecto['duration']))\n"
                       "Markdown(f'**Respuesta:** el proyecto dura {duracion_normal} semanas y tiene {len(rutas_criticas)} rutas críticas, '\n"
                       "         f'{lista_y([describe_ruta(proyecto, r, datos_proyecto[\"duration\"]) for r in rutas_criticas])}; '\n"
                       "         f'las actividades comunes a todas ellas son {lista_y(orden_comunes)}.')"))
    celdas.append(code('#| tbl-cap: "Duración de cada ruta (suma de sus actividades)"\n#| label: tbl-rutas\n'
                       "tabla_rutas(proyecto, datos_proyecto['duration'])"))
    celdas += apartado(2)
    celdas.append(code(f"barata = '{barata}'\n"
                       "con_barata = [r for r, a in rutas_criticas.items() if barata in a]\n"
                       "sin_barata = [r for r, a in rutas_criticas.items() if barata not in a]\n"
                       "comunes_con_margen = [a for a in orden_comunes if np.isfinite(datos_proyecto.loc[a, 'ucr']) and datos_proyecto.loc[a, 'duration'] > datos_proyecto.loc[a, 'cap_duration']]\n"
                       "Markdown(f'''**Respuesta:** no funcionará. {barata} solo está en {lista_y(con_barata)}; {lista_y([describe_ruta(proyecto, r, datos_proyecto[\"duration\"]) for r in sin_barata])} "
                       "no la contiene y seguiría midiendo {duracion_normal} semanas, así que el proyecto no se acorta. Para reducir el proyecto hay que acortar *todas* las rutas críticas "
                       "a la vez: una actividad común con margen ({lista_y(comunes_con_margen)}) o una actividad de cada ruta.''')"))
    celdas += apartado(3)
    celdas.append(code("d1 = datos_proyecto['duration'].copy(); d1[barata] -= 1\n"
                       "Markdown(f'**Respuesta:** acortando {barata} una semana la duración del proyecto sigue siendo {proyecto.duration(d1)} semanas (antes {duracion_normal}): '\n"
                       "         f'se han gastado {_num(datos_proyecto.loc[barata, \"ucr\"])} k€ para nada, porque {lista_y(sin_barata)} sigue midiendo {duracion_normal} semanas (tabla).')"))
    celdas.append(code('#| tbl-cap: "Duración de cada ruta tras acortar sólo la actividad más barata"\n#| label: tbl-rutasBarata\n'
                       "tabla_rutas(proyecto, d1, base=datos_proyecto['duration'])"))
    celdas += apartado(4)
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=2, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(md("Lo que se compara en cada paso: las rutas críticas en ese momento, las opciones que acortan "
                     "todas a la vez (una actividad común con margen, o una pareja con una actividad de cada ruta) "
                     "con su coste, y la elegida (la más barata)."))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso de Ackoff-Sasieni"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(md("Razonamiento de cada paso: en cada uno se comparan las actividades comunes a todas las rutas "
                     "críticas (acortan todas a la vez) con las parejas que toman una actividad de cada ruta."))
    celdas.append(code("narra_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],\n"
                       "                        'coste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},\n"
                       "                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))\n"
                       "resumen['sobrecoste acumulado'] = resumen['coste del paso'].cumsum()\n"
                       "resumen"))
    celdas.append(code("info = info_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)\n"
                       "frases = []\n"
                       "for i, x in enumerate(info):\n"
                       "    el = ' + '.join(x['elegidas'])\n"
                       "    comunes_paso = [(a[0], c) for a, c in x['opciones'] if len(a) == 1]\n"
                       "    parejas = [(' + '.join(a), c) for a, c in x['opciones'] if len(a) > 1]\n"
                       "    if len(x['elegidas']) == 1:\n"
                       "        alt = f'la pareja más barata, {parejas[0][0]}, costaría {_num(parejas[0][1])} k€' if parejas else 'no hay parejas posibles'\n"
                       "        frases.append(f'en el paso {i + 1} se acorta la actividad común {el} ({_num(datos_proyecto.loc[el, \"ucr\"])} k€) porque {alt}')\n"
                       "    else:\n"
                       "        alt = f'la actividad común más barata con margen, {comunes_paso[0][0]}, costaría {_num(comunes_paso[0][1])} k€' if comunes_paso else 'no queda ninguna actividad común con margen'\n"
                       "        frases.append(f'en el paso {i + 1} se acorta la pareja {el} ({_num(sum(datos_proyecto.loc[a, \"ucr\"] for a in x[\"elegidas\"]))} k€) porque {alt}')\n"
                       "usa_barata = [i + 1 for i, x in enumerate(info) if barata in x['elegidas']]\n"
                       "nota = (f'{barata} solo entra emparejada con una actividad de la otra ruta ({\"paso\" if len(usa_barata) == 1 else \"pasos\"} {rangos(usa_barata)}).' if usa_barata\n"
                       "        else f'{barata} no se acorta en ningún paso: sola no sirve, y emparejada no compensa.')\n"
                       "Markdown(f'''**Respuesta:** {'; '.join(frases)}. {nota} Duración final {proyecto.duration(duraciones_reducidas)} semanas con un sobrecoste total de "
                       "{' + '.join(_num(c) for c in resumen['coste del paso'])} = **{_num(resumen['coste del paso'].sum())} k€**.''')"))
    return celdas, f"{n} actividades, dos rutas críticas; {barata} es la más barata pero no es común"


def _desaceleracion(rng):
    """Reducido el proyecto, el plazo se relaja una semana: qué reducción deshacer (la del paso más caro)."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 12 and (i["D"] >= 2).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        try:
            pasos, dur = _pasos_ackoff(i, i["D"], tope, ucr, reduction=4)
        except Exception:
            return False
        if len(pasos) < 4:
            return False
        costes = [c for _, c in pasos]
        if costes.index(max(costes)) == 0 or len(set(costes)) < 3:
            return False
        estado.update(tope=tope, ucr=ucr, pasos=pasos, dur=dur)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, ucr, pasos = estado["tope"], estado["ucr"], estado["pasos"]
    df = P.tabla_base(info, duration=D, cap_duration=tope, ucr=ucr)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste unitario "
                 "de reducción (k€/semana) se adjunta. El proyecto se había reducido cuatro semanas con el algoritmo "
                 "de Ackoff-Sasieni cuando el cliente comunica que acepta la entrega una semana más tarde.")]
    celdas.append(preguntas([
        "Reduzca el proyecto cuatro semanas con el algoritmo de Ackoff-Sasieni y anote el sobrecoste de cada paso.",
        "Hay que devolver una semana. Un compañero propone deshacer el primer paso («fue el primero, es el más fácil "
        "de anular»). Otro propone deshacer el último. ¿Da lo mismo? Razónelo antes de calcular.",
        "Compruebe ambas opciones: duración resultante y dinero recuperado en cada caso. ¿Puede deshacerse un paso "
        "intermedio? ¿Qué regla general se deduce?",
        "¿Por qué el sobrecoste marginal de cada semana de reducción es creciente (o al menos no decreciente) en "
        "Ackoff-Sasieni? ¿Qué forma tiene la curva coste-duración?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion_normal = proyecto.duration(datos_proyecto['duration'])"))
    celdas.append(code(_AUX_REDUCCION))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales y el camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Duración de cada ruta con las duraciones normales"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "tabla_rutas(proyecto, datos_proyecto['duration'])"))
    celdas.append(code("explica_camino_critico(proyecto, datos_proyecto['duration'])"))
    celdas.append(md("En cada paso del algoritmo se acorta una semana la ruta crítica (o las rutas críticas) eligiendo, entre las "
                     "actividades críticas con margen, la combinación de menor CUR que reduce *todas* las rutas críticas a "
                     "la vez. La tabla muestra, para cada ruta, el CUR de sus actividades y su duración tras cada paso."))
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=4, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(md("Lo que se compara en cada paso: las rutas críticas en ese momento, las opciones que acortan "
                     "todas a la vez (una actividad común con margen, o una pareja con una actividad de cada ruta) "
                     "con su coste, y la elegida (la más barata)."))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso de Ackoff-Sasieni"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(md("Razonamiento de cada paso (los pasos consecutivos que acortan lo mismo se agrupan):"))
    celdas.append(code("narra_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],\n"
                       "                        'coste del paso': [sum(datos_proyecto.loc[acts, 'ucr']) for acts in pasos.values()]},\n"
                       "                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))\n"
                       "resumen['sobrecoste acumulado'] = resumen['coste del paso'].cumsum()\n"
                       "sobrecoste = float(resumen['coste del paso'].sum())\n"
                       "display(resumen)\n"
                       "Markdown(f'**Respuesta:** duración {duracion_normal} → {proyecto.duration(duraciones_reducidas)} semanas; sobrecoste de cada paso: '\n"
                       "         f'{\"; \".join(f\"paso {p}: {_num(c)} k€\" for p, c in resumen[\"coste del paso\"].items())}; total '\n"
                       "         f'{\" + \".join(_num(c) for c in resumen[\"coste del paso\"])} = **{_num(sobrecoste)} k€**.')"))
    celdas += apartado(2)
    celdas.append(md("**Respuesta:** no da lo mismo. Deshacer un paso es devolver una semana a las actividades que se "
                     "acortaron en él: el proyecto vuelve a alargarse una semana (las rutas críticas recuperan esa "
                     "semana), pero el dinero que se recupera es el coste de *ese* paso. Como en Ackoff-Sasieni los "
                     "pasos suelen ser cada vez más caros (apartado 4), deshacer el último recupera más dinero que "
                     "deshacer el primero; el apartado siguiente lo comprueba con los números."))
    celdas += apartado(3)
    celdas.append(code("filas = {}\n"
                       "for paso, acts in pasos.items():\n"
                       "    d2 = duraciones_reducidas.copy()\n"
                       "    for a in acts:\n"
                       "        d2[a] += 1\n"
                       "    filas[paso + 1] = {'actividades devueltas': ', '.join(acts), 'duración resultante': proyecto.duration(d2),\n"
                       "                       'dinero recuperado': float(sum(datos_proyecto.loc[acts, 'ucr']))}\n"
                       "pd.DataFrame(filas).T"))
    celdas.append(code("costes_pasos = {p + 1: float(sum(datos_proyecto.loc[acts, 'ucr'])) for p, acts in pasos.items()}\n"
                       "acts_pasos = {p + 1: ' + '.join(acts) for p, acts in pasos.items()}\n"
                       "T_red = proyecto.duration(duraciones_reducidas)\n"
                       "durs = {p: int(f['duración resultante']) for p, f in pd.DataFrame(filas).T.iterrows()}\n"
                       "ultimo = max(costes_pasos)\n"
                       "maximo = max(costes_pasos.values())\n"
                       "mejores = [p for p, c in costes_pasos.items() if np.isclose(c, maximo)]\n"
                       "iguales = [p for p, t in durs.items() if t == T_red + 1]\n"
                       "if len(iguales) == len(durs):\n"
                       "    t_dur = f'Todos los pasos devuelven el proyecto a {T_red + 1} semanas, así que cualquiera cumple el nuevo plazo.'\n"
                       "else:\n"
                       "    otros = [f'el paso {p} lo deja en {t} semanas' for p, t in durs.items() if t != T_red + 1]\n"
                       "    t_dur = f'Los pasos {rangos(iguales)} devuelven el proyecto a {T_red + 1} semanas; {lista_y(otros)}.'\n"
                       "t_mejor = (f'el que más recupera es el paso {mejores[0]} ({acts_pasos[mejores[0]]}, {_num(maximo)} k€)' if len(mejores) == 1\n"
                       "           else f'los que más recuperan son los pasos {rangos(mejores)} ({_num(maximo)} k€ cada uno; da igual cuál se deshaga)')\n"
                       "Markdown(f'''**Respuesta:** deshacer el primer paso ({acts_pasos[1]}) recupera {_num(costes_pasos[1])} k€; deshacer el último ({acts_pasos[ultimo]}) recupera "
                       "{_num(costes_pasos[ultimo])} k€. {t_dur} De todos, {t_mejor}. Un paso intermedio también puede deshacerse: lo que importa no es el orden en que se dieron "
                       "sino su coste. Regla general: para relajar el plazo se deshacen primero las reducciones más caras (se recorre la tabla de Ackoff-Sasieni hacia atrás, "
                       "comprobando que la duración vuelve a la deseada).''')"))
    celdas += apartado(4)
    celdas.append(md("En cada paso el algoritmo elige la opción más barata disponible; las opciones baratas se agotan "
                     "(las actividades llegan a su tope) o dejan de bastar (aparecen nuevas rutas críticas que exigen "
                     "acortar varias actividades a la vez), así que normalmente el paso siguiente no es más barato. "
                     "La curva coste-duración es entonces convexa: cada semana adicional de reducción cuesta al menos "
                     "lo que la anterior, y por eso la decisión de hasta dónde reducir se toma comparando el coste "
                     "marginal con lo que vale cada semana (penalización, coste indirecto, prima). No es un teorema: "
                     "el algoritmo es voraz y, si un paso acorta dos actividades de una misma ruta, esa ruta puede "
                     "dejar de ser crítica y el paso siguiente volver a ser barato."))
    celdas.append(code("costes = list(resumen['coste del paso'])\n"
                       "if all(b >= a for a, b in zip(costes, costes[1:])):\n"
                       "    t = f'aquí se cumple: los costes de los pasos son {\" ≤ \".join(_num(c) for c in costes)} k€, y la curva coste-duración es convexa.'\n"
                       "else:\n"
                       "    bajan = [i + 2 for i, (a, b) in enumerate(zip(costes, costes[1:])) if b < a]\n"
                       "    t = (f'aquí no se cumple del todo: los costes de los pasos son {\", \".join(_num(c) for c in costes)} k€ y el paso {rangos(bajan)} es más barato que el anterior; '\n"
                       "         'por eso hay que mirar la tabla completa antes de decidir.')\n"
                       "Markdown(f'**Respuesta:** {t}')"))
    return celdas, f"{n} actividades; deshacer la reducción más cara al relajar el plazo"



def _lp_coste(R, D, tope, ucr, T_obj):
    """Coste mínimo de reducir el proyecto a T_obj por programación lineal (None si no es factible)."""
    from scipy.optimize import linprog
    acts = list(R.columns)
    c = np.array([0.0 if np.isinf(ucr[a]) else float(ucr[a]) for a in acts])
    ub = np.array([0.0 if np.isinf(ucr[a]) else float(D[a] - tope[a]) for a in acts])
    A = -R.values.astype(float)
    b = T_obj - R.values.astype(float) @ np.array([float(D[a]) for a in acts])
    res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(np.zeros(len(acts)), ub)), method="highs")
    return (float(res.fun), pd.Series(res.x, index=acts)) if res.success else (None, None)


def _programacion_lineal(rng):
    """Ackoff-Sasieni (voraz) frente a la solución exacta por programación lineal con la matriz de caminos."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def _prepara(i, estricto):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 12 and (i["D"] >= 2).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        if int((i["D"] - tope).sum()) < 4:
            return False
        try:
            pasos, _ = _pasos_ackoff(i, i["D"], tope, ucr, reduction=100)
        except Exception:
            return False
        if len(pasos) < 3:
            return False
        for r in (3, 2, 4):
            if r > len(pasos):
                continue
            coste_ackoff = float(sum(c for _, c in pasos[:r]))
            coste_lp, x = _lp_coste(i["P"], i["D"], tope, ucr, int(i["T"]) - r)
            if coste_lp is None:
                continue
            if (not estricto) or coste_lp < coste_ackoff - 1e-6:
                estado.update(tope=tope, ucr=ucr, pasos=pasos, r=r, coste_ackoff=coste_ackoff, coste_lp=coste_lp)
                return True
        return False

    try:
        info = P.buscar(rng, lambda i: _prepara(i, True), n=n, n_inicio=int(rng.choice([2, 3])),
                        n_fin=int(rng.choice([1, 2])), max_intentos=400)
    except RuntimeError:
        info = P.buscar(rng, lambda i: _prepara(i, False), n=n, n_inicio=int(rng.choice([2, 3])),
                        n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, ucr, r = estado["tope"], estado["ucr"], estado["r"]
    mejora = estado["coste_lp"] < estado["coste_ackoff"] - 1e-6
    df = P.tabla_base(info, duration=D, cap_duration=tope, ucr=ucr)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duraciones normal y tope (semanas) y coste "
                 "unitario de reducción CUR (k€/semana) se adjunta. Las actividades con CUR infinito no pueden "
                 f"acelerarse. Se quiere reducir la duración del proyecto en {r} semanas con el mínimo sobrecoste.")]
    celdas.append(preguntas([
        "Calcule la duración normal del proyecto, construya la matriz de caminos $R$ y la duración de cada ruta.",
        f"Aplique el algoritmo de Ackoff-Sasieni para reducir {r} semanas y obtenga el sobrecoste total.",
        "Plantee el problema como un programa lineal: variables $x_i$ = semanas que se acorta la actividad "
        "$i$; minimizar $\\sum_i CUR_i\\, x_i$ sujeto a que ninguna ruta supere la duración objetivo, "
        "$R\\,(d - x) \\le T_{obj}$, y $0 \\le x_i \\le d_i - d_i^{tope}$. Resuélvalo con `scipy.optimize.linprog`.",
        "Compare las dos soluciones. ¿Puede el algoritmo voraz ser peor que el óptimo? ¿Por qué el programa "
        "lineal da aquí soluciones enteras?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"reduccion = {r}"))
    celdas.append(code(_AUX_REDUCCION))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Matriz de caminos y duración de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                       "T_objetivo = duracion_normal - reduccion\n"
                       "tabla_rutas(proyecto, datos_proyecto['duration'])"))
    celdas.append(md("La matriz de caminos $R$ tiene una fila por ruta y una columna por actividad, con un 1 si la "
                     "actividad está en la ruta; el producto $R\\,d$ da la duración de cada ruta (columna «duración» "
                     "de la tabla anterior)."))
    celdas.append(code("display(R)\n"
                       "rutas_dur = R @ datos_proyecto['duration'][R.columns]\n"
                       "criticas = [r for r in rutas_dur.index if rutas_dur[r] == rutas_dur.max()]\n"
                       "Markdown(f'**Respuesta:** $R$ tiene {R.shape[0]} filas (rutas) y {R.shape[1]} columnas (actividades). La duración normal es la mayor de las '\n"
                       "         f'duraciones de ruta, máx(R·d) = máx({\", \".join(str(int(v)) for v in rutas_dur)}) = **{duracion_normal} semanas**, que da '\n"
                       "         f'{lista_y([describe_ruta(proyecto, r, datos_proyecto[\"duration\"]) for r in criticas])}. El objetivo es {duracion_normal} − {reduccion} = {T_objetivo} semanas.')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_ackoff, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=reduccion, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("narra_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos)"))
    celdas.append(code("x_ackoff = (datos_proyecto['duration'] - duraciones_ackoff.reindex(datos_proyecto.index)).astype(int)\n"
                       "coste_ackoff = float((x_ackoff * datos_proyecto['ucr'].replace(np.inf, 0)).sum())\n"
                       "suma_ackoff = ' + '.join(f'{_num(datos_proyecto.loc[a, \"ucr\"])}·{int(x)}' for a, x in x_ackoff[x_ackoff > 0].items())\n"
                       "Markdown(f'**Respuesta:** Ackoff-Sasieni acorta {lista_y([f\"{a} {plural(x, 'semana')}\" for a, x in x_ackoff[x_ackoff > 0].items()])}; '\n"
                       "         f'sobrecoste = Σ CUR·x = {suma_ackoff} = **{_num(coste_ackoff)} k€**; duración {proyecto.duration(duraciones_ackoff)} semanas.')"))
    celdas += apartado(3)
    celdas.append(md("Cada fila de $R$ es una ruta; $R\\,d$ son las duraciones de las rutas y $R\\,x$ lo que se "
                     "acorta cada una. La restricción $R\\,(d - x) \\le T_{obj}$ se escribe como "
                     "$-R\\,x \\le T_{obj} - R\\,d$ para `linprog` (que exige la forma $A x \\le b$)."))
    celdas.append(code("from scipy.optimize import linprog\n"
                       "acts = list(R.columns)\n"
                       "d = datos_proyecto.loc[acts, 'duration'].astype(float).values\n"
                       "cur = datos_proyecto.loc[acts, 'ucr'].astype(float)\n"
                       "margen = (datos_proyecto.loc[acts, 'duration'] - datos_proyecto.loc[acts, 'cap_duration']).astype(float)\n"
                       "c = cur.replace(np.inf, 0).values                       # coste por semana acortada\n"
                       "cotas = [(0, 0 if np.isinf(cur[a]) else margen[a]) for a in acts]\n"
                       "A_ub = -R.values.astype(float)\n"
                       "b_ub = T_objetivo - R.values @ d\n"
                       "solucion = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=cotas, method='highs')\n"
                       "x_lp = pd.Series(solucion.x, index=acts).round(6)\n"
                       "coste_lp = float(solucion.fun)\n"
                       "entera = bool(np.allclose(x_lp, x_lp.round()))\n"
                       "pd.DataFrame({'x (Ackoff)': x_ackoff.reindex(acts), 'x (PL)': x_lp, 'CUR': cur}).T"))
    celdas.append(code("duraciones_lp = datos_proyecto['duration'] - x_lp.reindex(datos_proyecto.index).fillna(0)\n"
                       "suma_lp = ' + '.join(f'{_num(cur[a])}·{x:g}' for a, x in x_lp[x_lp > 0].items())\n"
                       "Markdown(f'**Respuesta:** la programación lineal acorta {lista_y([f\"{a} {plural(x, 'semana', decimales=2)}\" for a, x in x_lp[x_lp > 0].items()])}, con sobrecoste '\n"
                       "         f'Σ CUR·x = {suma_lp} = **{_num(coste_lp)} k€**; todas las rutas quedan en como mucho {float((R @ duraciones_lp[acts]).max()):g} semanas (objetivo {T_objetivo}).')"))
    celdas.append(code('#| tbl-cap: "Duración de cada ruta con la solución del programa lineal"\n#| label: tbl-rutasLP\n'
                       "tabla_rutas(proyecto, duraciones_lp, decimales=1, base=datos_proyecto['duration'])"))
    celdas += apartado(4)
    celdas.append(code("misma = bool(np.allclose(x_ackoff.reindex(acts).astype(float), x_lp))\n"
                       "if coste_lp < coste_ackoff - 1e-6:\n"
                       "    comp = (f'el programa lineal es **más barato** ({_num(coste_lp)} k€ frente a {_num(coste_ackoff)} k€ de Ackoff-Sasieni, {_num(coste_ackoff - coste_lp)} k€ menos). '\n"
                       "            'Sí, el voraz puede ser peor: decide semana a semana y, una vez acortada una actividad, no la «desacorta» aunque más adelante otra '\n"
                       "            f'combinación hubiera resultado mejor para el conjunto; el PL optimiza las {reduccion} semanas a la vez.')\n"
                       "elif misma:\n"
                       "    comp = (f'las dos soluciones coinciden (mismas actividades y {_num(coste_lp)} k€): aquí el voraz ha acertado. No siempre es así: decide semana a semana y '\n"
                       "            'no revisa decisiones anteriores, por lo que puede ser más caro que el óptimo cuando una actividad acortada pronto deja de ser útil al cambiar las rutas críticas.')\n"
                       "else:\n"
                       "    comp = (f'las dos soluciones cuestan lo mismo ({_num(coste_lp)} k€) aunque acortan actividades distintas: son óptimos alternativos (combinaciones distintas con el mismo coste), '\n"
                       "            'y aquí el voraz ha acertado. No siempre es así: decide semana a semana y no revisa decisiones anteriores, por lo que puede ser más caro que el óptimo '\n"
                       "            'cuando una actividad acortada pronto deja de ser útil al cambiar las rutas críticas.')\n"
                       "ent = ('El PL ha dado aquí una solución entera' if entera else 'El PL ha dado aquí una solución fraccionaria')\n"
                       "Markdown(f'''**Respuesta:** {comp} {ent}: con datos enteros suele ocurrir, pero no está garantizado "
                       "(la matriz de caminos no es en general totalmente unimodular); si hiciera falta, habría que exigir x entera con un programa entero, que `linprog` no resuelve. "
                       "En todo caso el óptimo del PL es una cota inferior del coste de cualquier reducción entera.''')"))
    return celdas, f"{n} actividades; reducir {r} semanas: Ackoff {estado['coste_ackoff']:.0f} k€ frente a PL {estado['coste_lp']:.0f} k€"


def _cur_por_tramos(rng):
    """El CUR crece con cada semana acortada (tramos): hay que usar el coste marginal en cada paso."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 9 and i["T"] >= 12 and (i["D"] >= 3).all()):
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        # margen de exactamente 2 semanas en las actividades reducibles
        tope = pd.Series({a: (int(i["D"][a]) - 2 if ucr[a] > 0 else int(i["D"][a])) for a in i["D"].index}, name="cap_duration")
        cur1 = pd.Series({a: (int(ucr[a]) if ucr[a] > 0 else np.inf) for a in i["D"].index}, name="ucr_1")
        cur2 = pd.Series({a: (int(ucr[a]) + int(rng.integers(2, 7)) if ucr[a] > 0 else np.inf) for a in i["D"].index}, name="ucr_2")
        if (cur1 < np.inf).sum() < 5:
            return False
        criticas = [a for a in i["D"].index if i["calendario"].loc[a, "H_total"] == 0]
        if sum(1 for a in criticas if cur1[a] < np.inf) < 3:
            return False
        estado.update(tope=tope, cur1=cur1, cur2=cur2)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    tope, cur1, cur2 = estado["tope"], estado["cur1"], estado["cur2"]
    r = 3
    df = P.tabla_base(info, duration=D, cap_duration=tope, ucr_1=cur1, ucr_2=cur2)
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones y duraciones normal y tope (semanas) se "
                 "adjunta. Las actividades reducibles pueden acortarse hasta dos semanas, pero el coste no es "
                 "lineal: la primera semana cuesta `ucr_1` k€ y la segunda `ucr_2` k€ (más cara: horas extra, "
                 "turno de noche…). Las actividades con CUR infinito no pueden acelerarse.")]
    celdas.append(preguntas([
        "Calcule la duración normal del proyecto y sus rutas críticas.",
        f"Reduzca el proyecto {r} semanas con mínimo coste aplicando Ackoff-Sasieni paso a paso con el *coste "
        "marginal*: el de la semana que toca acortar a cada actividad. Muestre en cada paso las rutas "
        "críticas, las opciones y su coste.",
        "¿Qué habría resultado de aplicar el algoritmo con un CUR único por actividad igual a la media "
        "$(ucr_1 + ucr_2)/2$? Compare el coste real de esa solución con el del apartado anterior.",
        "¿Por qué con costes crecientes por tramos el procedimiento semana a semana sigue siendo válido, y "
        "qué fallaría si la segunda semana fuera *más barata* que la primera?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       f"reduccion = {r}"))
    celdas.append(code(_AUX_REDUCCION))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales", label="fig-pert")
    celdas.append(code("duracion_normal = proyecto.duration(datos_proyecto['duration'])\n"
                       "rutas_criticas = proyecto.critical_path(datos_proyecto['duration'])\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "Markdown(f'**Respuesta:** duración normal {duracion_normal} semanas; '\n"
                       "         f'{\"ruta crítica\" if len(rutas_criticas) == 1 else \"rutas críticas\"}: {lista_y([describe_ruta(proyecto, r, datos_proyecto[\"duration\"]) for r in rutas_criticas])}.')"))
    celdas.append(code('#| tbl-cap: "Duración de cada ruta con las duraciones normales"\n#| label: tbl-rutas\n'
                       "tabla_rutas(proyecto, datos_proyecto['duration'])"))
    celdas += apartado(2)
    celdas.append(md("En cada paso el coste marginal de una actividad es `ucr_1` si aún no se ha acortado y "
                     "`ucr_2` si ya se acortó una semana (infinito si no le queda margen). Las opciones son las "
                     "combinaciones mínimas que acortan todas las rutas críticas a la vez (una actividad común, o "
                     "una actividad de cada ruta), valoradas con su coste marginal; se elige la más barata. La "
                     "tabla indica, para la elegida, qué semana de cada actividad se acorta y a qué precio."))
    celdas.append(code("def reducir_por_tramos(d0, tope, ucr_1, ucr_2, semanas):\n"
                       "    d = d0.astype(float).copy()\n"
                       "    filas, acumulado = [], 0.0\n"
                       "    for paso in range(1, semanas + 1):\n"
                       "        acortado = d0 - d\n"
                       "        marginal = pd.Series({a: (np.inf if d[a] <= tope[a] else (ucr_1[a] if acortado[a] == 0 else ucr_2[a])) for a in d.index})\n"
                       "        L = R @ d[R.columns]\n"
                       "        T = L.max(); criticas = [r for r in L.index if np.isclose(L[r], T)]\n"
                       "        conjuntos = [set(R.columns[R.loc[r] == 1]) for r in criticas]\n"
                       "        opciones = opciones_ackoff(conjuntos, d, tope, marginal)\n"
                       "        if not opciones:\n"
                       "            filas.append({'paso': paso, 'duración': int(T), 'rutas críticas': ', '.join(criticas), 'opciones (coste marginal)': 'ninguna',\n"
                       "                          'elegida': '—', 'semana que se acorta (precio)': '—', 'coste': np.nan, 'acumulado': acumulado})\n"
                       "            break\n"
                       "        elegida, coste = opciones[0]\n"
                       "        acumulado += coste\n"
                       "        txt = lambda acts, c: acts[0] + f' ({_num(c)})' if len(acts) == 1 else ' + '.join(acts) + f\" ({' + '.join(_num(marginal[a]) for a in acts)} = {_num(c)})\"\n"
                       "        semana = '; '.join(f\"{a}: {int(acortado[a]) + 1}.ª semana ({'ucr_1' if acortado[a] == 0 else 'ucr_2'} = {_num(marginal[a])})\" for a in elegida)\n"
                       "        filas.append({'paso': paso, 'duración': int(T), 'rutas críticas': ', '.join(criticas),\n"
                       "                      'opciones (coste marginal)': '; '.join(txt(a, c) for a, c in opciones[:6]) + (' …' if len(opciones) > 6 else ''),\n"
                       "                      'elegida': ' + '.join(elegida), 'semana que se acorta (precio)': semana, 'coste': coste, 'acumulado': acumulado})\n"
                       "        for a in elegida:\n"
                       "            d[a] -= 1\n"
                       "    return pd.DataFrame(filas).set_index('paso'), d\n"
                       "pasos_tramos, d_tramos = reducir_por_tramos(datos_proyecto['duration'], datos_proyecto['cap_duration'],\n"
                       "                                            datos_proyecto['ucr_1'], datos_proyecto['ucr_2'], reduccion)\n"
                       "pasos_tramos"))
    celdas.append(code("coste_tramos = float(pasos_tramos['coste'].sum())\n"
                       "acortadas = (datos_proyecto['duration'] - d_tramos)\n"
                       "lista_acortadas = lista_y([a + ' ' + plural(x, 'semana') for a, x in acortadas[acortadas > 0].items()])\n"
                       "Markdown(f\"**Respuesta:** se acortan {lista_acortadas}; \"\n"
                       "         f\"la duración pasa de {duracion_normal} a {proyecto.duration(d_tramos.astype(int))} semanas con un sobrecoste de \"\n"
                       "         f\"{' + '.join(_num(c) for c in pasos_tramos['coste'])} = **{_num(coste_tramos)} k€**.\")"))
    celdas += apartado(3)
    celdas.append(code("ucr_medio = ((datos_proyecto['ucr_1'] + datos_proyecto['ucr_2']) / 2)\n"
                       "tabla_ackoff, pasos, d_medio, _ = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "                                                  costs=ucr_medio, reduction=reduccion, report=False)\n"
                       "acortadas_medio = (datos_proyecto['duration'] - d_medio.reindex(datos_proyecto.index)).astype(int)\n"
                       "def coste_real(acortadas):\n"
                       "    return float(sum((datos_proyecto.loc[a, 'ucr_1'] if x >= 1 else 0) + (datos_proyecto.loc[a, 'ucr_2'] if x >= 2 else 0) for a, x in acortadas.items()))\n"
                       "def coste_real_txt(acortadas):\n"
                       "    partes = [f\"{a}: {_num(datos_proyecto.loc[a, 'ucr_1'])}\" + (f\" + {_num(datos_proyecto.loc[a, 'ucr_2'])}\" if x >= 2 else '')\n"
                       "              for a, x in acortadas.items() if x >= 1]\n"
                       "    return '; '.join(partes) + f' → total {_num(coste_real(acortadas))}'\n"
                       "pd.DataFrame({'con coste marginal': [', '.join(f'{a} ({int(x)})' for a, x in acortadas[acortadas > 0].items()), coste_real_txt(acortadas)],\n"
                       "              'con CUR medio': [', '.join(f'{a} ({int(x)})' for a, x in acortadas_medio[acortadas_medio > 0].items()), coste_real_txt(acortadas_medio)]},\n"
                       "             index=['actividades acortadas (semanas)', 'coste real, cada semana con su tramo (k€)'])"))
    celdas.append(md("Con el CUR medio el algoritmo puede preferir acortar dos semanas una actividad que parece "
                     "barata «de media» cuando en realidad su segunda semana es cara, o evitar una cuya primera "
                     "semana era la más barata de todas. El coste real se obtiene valorando cada semana con su "
                     "tramo."))
    celdas.append(code("c_medio = coste_real(acortadas_medio)\n"
                       "misma = bool((acortadas_medio.reindex(acortadas.index).fillna(0).astype(int) == acortadas.astype(int)).all())\n"
                       "if misma:\n"
                       "    t = f'con el CUR medio se acortan las mismas actividades y el coste real es el mismo, {_num(c_medio)} k€: aquí el atajo no ha hecho daño, pero no es fiable.'\n"
                       "elif c_medio > coste_tramos + 1e-9:\n"
                       "    t = f'con el CUR medio se acortan otras semanas y el coste real sube a {_num(c_medio)} k€, {_num(c_medio - coste_tramos)} k€ más que con el coste marginal ({_num(coste_tramos)} k€).'\n"
                       "elif np.isclose(c_medio, coste_tramos):\n"
                       "    t = f'con el CUR medio se acortan otras semanas, pero el coste real coincide ({_num(c_medio)} k€): aquí hay dos soluciones igual de baratas, lo que es casualidad.'\n"
                       "else:\n"
                       "    t = (f'con el CUR medio sale incluso más barato ({_num(c_medio)} k€ frente a {_num(coste_tramos)} k€): el procedimiento semana a semana es voraz y, '\n"
                       "         'con varias rutas críticas, no garantiza el óptimo.')\n"
                       "Markdown(f'**Respuesta:** {t}')"))
    celdas += apartado(4)
    celdas.append(md("**Respuesta:** con costes marginales *crecientes* el coste de cada actividad es convexo: cada "
                     "tramo se comporta como una «actividad» distinta con su propio CUR, y el tramo barato siempre "
                     "se usa antes que el caro. Por eso el procedimiento semana a semana valora bien cada semana y "
                     "es tan válido como Ackoff-Sasieni con CUR constante (con la misma salvedad: es voraz y, cuando "
                     "hay varias rutas críticas, no garantiza el óptimo; la programación lineal sí). Si la segunda "
                     "semana fuera *más barata* que la primera (coste cóncavo: un coste fijo de movilización y "
                     "después barato), el paso a paso fallaría: la primera semana parecería cara y no se emprendería "
                     "aunque acortar dos semanas de golpe fuera lo mejor; habría que evaluar las dos semanas juntas."))
    return celdas, f"{n} actividades; CUR por tramos, reducir {r} semanas"


def _crashing_recursos(rng):
    """Recursos limitados + crashing: Ackoff sin recursos puede elegir una actividad que no acorta el proyecto."""
    n = int(rng.choice([8, 9]))
    estado = {}

    def acepta(i):
        if not (0 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 10 and 10 <= i["T"] <= 22):
            return False
        nombres = list(i["D"].index)
        op = P.recursos(rng, nombres, hi=3).rename("operarios")
        L = int(op.max())
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        if int(np.isfinite(ucr).sum()) < 4:
            return False
        df = P.tabla_base(i, duration=i["D"], operarios=op, cap_duration=tope, ucr=ucr)
        pr = i["proyecto"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = pr.copy().asignar_multi(df[["predecessor", "duration", "operarios"]].copy(), "duration", ["operarios"], [L])
            T_res = int(res.duracion)
            if not (1 <= T_res - int(i["T"]) <= 4):
                return False
            pasos, dur = _pasos_ackoff(i, i["D"], tope, ucr, reduction=2)
            with contextlib.redirect_stdout(io.StringIO()):
                r_ack = pr.copy().asignar_multi(df[["predecessor", "operarios"]].assign(duration=dur), "duration", ["operarios"], [L])
            if int(r_ack.duracion) < T_res:          # Ackoff debe fallar del todo (no acorta nada con recursos)
                return False
            with contextlib.redirect_stdout(io.StringIO()):
                cr = pr.copy().crashing_recursos(df[["predecessor", "duration", "operarios"]].copy(), "duration", ["operarios"], [L],
                                                 tope, ucr, reduccion=2)
            if cr.duracion_final > T_res - 2:
                return False
            with contextlib.redirect_stdout(io.StringIO()):
                libre = pr.copy().asignar_multi(df[["predecessor", "duration", "operarios"]].copy(), "duration", ["operarios"], [L + 1])
            if int(libre.duracion) >= T_res:
                return False
        except Exception:
            return False
        estado.update(df=df, L=L, T_res=T_res)
        return True

    info = P.buscar(rng, acepta, n=n, n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])),
                    max_intentos=8000)
    df, L, T_res = estado["df"], estado["L"], estado["T_res"]
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, duración normal y tope (semanas), operarios por "
                 "actividad y coste unitario de reducción (k€/semana) se adjunta. Cada actividad necesita un número fijo "
                 f"de operarios durante toda su duración y la empresa dispone como máximo de {L} operarios en cada semana. "
                 "Las actividades con CUR infinito no pueden acelerarse. Se quiere reducir la duración del proyecto "
                 "**dos semanas** respetando el límite de operarios.")]
    celdas.append(preguntas([
        "Determine la duración del proyecto sin límite de recursos y con el límite de operarios (algoritmo de asignación). "
        "Indique la cadena crítica con recursos: qué actividades fijan la duración y qué esperas por recurso intervienen.",
        "Aplique el algoritmo de Ackoff-Sasieni ignorando los recursos para reducir dos semanas, aplique esas reducciones "
        "y replanifique con el límite de operarios. ¿Se ha conseguido reducir la duración del proyecto con recursos? ¿Por qué?",
        "Reduzca ahora dos semanas teniendo en cuenta el límite: pruebe en cada paso los candidatos (cortes de la cadena "
        "crítica con recursos, actividades sueltas y parejas) y elija el más barato que de verdad acorte el proyecto "
        "replanificado. Compare el sobrecoste con el del apartado anterior.",
        "Suponga que acelerar una actividad exige un operario más por cada semana recortada (modelo con recurso). "
        "¿Cambia la solución?",
        f"Alternativa: contratar un operario más (límite {L + 1}) sin acelerar nada. ¿Qué duración se obtiene? "
        "¿Compensa frente a acelerar?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"limite = [{L}]\n"
                       "base = datos_proyecto[['predecessor', 'duration', 'operarios']]"))
    celdas.append(code(_AUX_REDUCCION))
    celdas.append(code(_AUX_RECURSOS))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones normales y el camino crítico (sin límite de operarios)", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Calendario sin límite de recursos"\n#| label: tbl-calendario\n'
                       "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
                       "duracion_libre = proyecto.duration(datos_proyecto['duration'])\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]"))
    celdas.append(code('#| fig-cap: "Gantt de inicios más tempranos con los operarios de cada actividad (fila Total: carga semanal)"\n#| label: fig-ganttInicial\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'operarios', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Carga de operarios frente al límite"\n#| label: fig-cargasInicial\n'
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=limite)"))
    celdas.append(code("cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "exceso = cargas[cargas > limite[0]]\n"
                       "def _activas(p):\n"
                       "    return [a for a in datos_proyecto.index if calendario.loc[a, 'inicio_mas_temprano'] < p <= calendario.loc[a, 'fin_mas_temprano']]\n"
                       "detalle_exceso = '; '.join(f'semana {int(t)}: ' + ' + '.join(f\"{a}({int(datos_proyecto.loc[a, 'operarios'])})\" for a in _activas(int(t))) + f' = {int(c)}'\n"
                       "                           for t, c in exceso.items())\n"
                       "display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "Markdown(f'Sin límite el proyecto dura {duracion_libre} semanas, pero con los inicios más tempranos se superan los {limite[0]} operarios disponibles '\n"
                       "         f'({detalle_exceso} operarios): hay que reprogramar.')"))
    celdas.append(md("El algoritmo de asignación recorre las semanas una a una. En cada semana son *candidatas* las "
                     "actividades no empezadas cuyas predecesoras han terminado; se ordenan por prioridad (menor inicio "
                     "más tardío en el calendario sin recursos, después menor duración) y cada una empieza si, en todas las "
                     "semanas que ocuparía, los operarios ya comprometidos más los suyos no superan el límite; si no, "
                     "espera. La traza muestra la decisión de cada semana y, entre paréntesis, el recurso que bloquea a las que esperan."))
    celdas.append(code("res0 = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt con el límite de operarios (filas 💤: esperas por falta de operarios)"\n#| label: fig-ganttAsignado\n'
                       "pinta_tikz(res0.dibujos['operarios'])"))
    celdas.append(code('#| tbl-cap: "Calendario con recursos: inicio, fin y espera de cada actividad"\n#| label: tbl-calendarioRecursos\n'
                       "calendario_recursos = pd.DataFrame({'inicio sin límite': calendario['inicio_mas_temprano'].astype(int), 'inicio con límite': res0.inicios,\n"
                       "                                    'fin con límite': res0.fines, 'espera (semanas)': res0.esperas}).fillna(0).astype(int)\n"
                       "calendario_recursos"))
    celdas.append(code('#| tbl-cap: "Carga de operarios semana a semana frente al límite"\n#| label: tbl-cargasRecursos\n'
                       "pd.concat([res0.cargas['operarios'].rename('operarios ocupados'), res0.limites['operarios'].rename('límite')], axis=1).T"))
    celdas.append(code("cadena = proyecto.copy().cadena_critica_recursos(base.copy(), 'duration', ['operarios'], limite)\n"
                       "Markdown(f'Sin límite: {duracion_libre} semanas. Con {limite[0]} operarios: {res0.duracion} semanas ({res0.duracion - duracion_libre} más). '\n"
                       "         f'Actividades de la cadena crítica con recursos: {lista_y(sorted(cadena[\"actividades\"]))}; sus arcos:')"))
    celdas.append(code('#| tbl-cap: "Arcos de la cadena crítica con recursos"\n#| label: tbl-cadena\n'
                       "cadena['arcos'].rename(columns={'de': 'de', 'a': 'a', 'motivo': 'motivo (precedencia o recurso)'})"))
    celdas.append(md("La cadena crítica con recursos se lee hacia atrás desde la última actividad: cada arco une una "
                     "actividad con otra que empieza justo cuando ella termina, bien porque es su predecesora "
                     "(*precedencia*), bien porque le libera los operarios que esperaba (*recurso*). Los arcos de "
                     "recurso son esperas, no prelaciones: la duración la fija esa cadena (precedencias más esperas), "
                     "no el camino crítico del grafo."))
    celdas.append(code("Markdown(f'**Respuesta:** sin límite de recursos el proyecto dura {duracion_libre} semanas; con {limite[0]} operarios, {res0.duracion} semanas '\n"
                       "         f'({plural(res0.duracion - duracion_libre, \"semana\")} más). La cadena crítica con recursos es '\n"
                       "         f'{explica_cadena(cadena, res0, datos_proyecto)}')"))
    celdas += apartado(2)
    celdas.append(md("Ackoff-Sasieni sólo mira el grafo: acorta en cada paso, una semana, la combinación más barata de "
                     "actividades que reduce todas las rutas críticas *sin recursos*. La tabla muestra, para cada ruta, "
                     "el CUR de sus actividades y su duración tras cada paso."))
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni sin tener en cuenta los recursos"\n#| label: tbl-ackoff\n'
                       "tabla_ack, pasos_ack, dur_ack, margen = proyecto.ackoff(durations=datos_proyecto['duration'],\n"
                       "    min_durations=datos_proyecto['cap_duration'], costs=datos_proyecto['ucr'], reduction=2, report=False)\n"
                       "tabla_ack"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos_ack)"))
    celdas.append(code("resumen_ack = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos_ack.values()],\n"
                       "                            'coste del paso': [float(sum(datos_proyecto.loc[acts, 'ucr'])) for acts in pasos_ack.values()]},\n"
                       "                           index=pd.Index(range(1, len(pasos_ack) + 1), name='paso'))\n"
                       "coste_ack = float(resumen_ack['coste del paso'].sum())\n"
                       "resumen_ack"))
    celdas.append(md("Aplicadas esas reducciones, se vuelve a programar con el límite de operarios:"))
    celdas.append(code("res_ack = proyecto.copy().asignar_multi(base.assign(duration=dur_ack.reindex(base.index).astype(int)), 'duration', ['operarios'], limite, report=True, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt con el límite de operarios tras las reducciones de Ackoff-Sasieni"\n#| label: fig-ganttAckoff\n'
                       "pinta_tikz(res_ack.dibujos['operarios'])"))
    celdas.append(code('#| tbl-cap: "Duración e inicio de cada actividad antes y después de aplicar Ackoff-Sasieni (con el límite)"\n#| label: tbl-inicioAckoff\n'
                       "pd.DataFrame({'duración normal': datos_proyecto['duration'], 'duración Ackoff': dur_ack.reindex(datos_proyecto.index).astype(int),\n"
                       "              'inicio con límite (antes)': res0.inicios, 'inicio con límite (después)': res_ack.inicios,\n"
                       "              'fin con límite (antes)': res0.fines, 'fin con límite (después)': res_ack.fines}).T"))
    celdas.append(code("datos_ack = base.assign(duration=dur_ack.reindex(base.index).astype(int))\n"
                       "cadena_ack = proyecto.copy().cadena_critica_recursos(datos_ack.copy(), 'duration', ['operarios'], limite)\n"
                       "acortadas_ack = sorted(set(a for acts in pasos_ack.values() for a in acts))\n"
                       "fuera = [a for a in acortadas_ack if a not in cadena['actividades']]\n"
                       "dentro = [a for a in acortadas_ack if a in cadena['actividades']]\n"
                       "if fuera and not dentro:\n"
                       "    por_que = (f'{lista_y(fuera)} no {\"está\" if len(fuera) == 1 else \"están\"} en la cadena crítica con recursos del apartado 1, así que acortar'\n"
                       "               f'{\"la\" if len(fuera) == 1 else \"las\"} adelanta otras actividades, pero no lo que fija el plazo con recursos')\n"
                       "else:\n"
                       "    por_que = (f'aunque {lista_y(dentro)} {\"está\" if len(dentro) == 1 else \"están\"} en la cadena crítica con recursos, al replanificar cambian las '\n"
                       "               'esperas por operarios y otra cadena pasa a fijar el plazo')\n"
                       "costes_ack = ' + '.join(_num(c) for c in resumen_ack['coste del paso'])\n"
                       "logrado = 'no' if res_ack.duracion >= res0.duracion else 'solo en parte'\n"
                       "Markdown(f'''**Respuesta:** {logrado}. Ackoff-Sasieni sin recursos acorta {lista_y(acortadas_ack)} con un coste de {costes_ack} = {_num(coste_ack)} k€ y el grafo baja a "
                       "{proyecto.duration(dur_ack)} semanas sobre el papel, pero al replanificar con {limite[0]} operarios el proyecto dura {res_ack.duracion} semanas (antes {res0.duracion}): "
                       "{por_que}. Con las duraciones de Ackoff la cadena que fija el plazo es {explica_cadena(cadena_ack, res_ack, datos_proyecto)} "
                       "El dinero se gasta sin acortar lo que de verdad fija el plazo.''')"))
    celdas += apartado(3)
    celdas.append(md("`crashing_recursos` repite en cada paso cuatro operaciones: (1) programa con el límite, (2) halla la "
                     "cadena crítica con recursos, (3) evalúa como candidatos las actividades reducibles de esa cadena "
                     "una a una, por parejas y el corte de coste mínimo de la cadena, **replanificando** con cada "
                     "candidato, y (4) se queda con el más barato que de verdad acorta la duración replanificada. La "
                     "tabla muestra, paso a paso, la cadena crítica, lo que habría elegido Ackoff-Sasieni sin recursos, "
                     "cuántos candidatos se evaluaron y cuántas reducciones de una sola actividad no acortaron nada."))
    celdas.append(code('#| tbl-cap: "Crashing con recursos limitados, paso a paso"\n#| label: tbl-crashing\n'
                       "cr = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], limite,\n"
                       "                                       datos_proyecto['cap_duration'], datos_proyecto['ucr'], reduccion=2)\n"
                       "COLUMNAS_CR = {'cadena_critica': 'cadena crítica con recursos', 'ackoff_sin_recursos': 'Ackoff sin recursos elegiría',\n"
                       "               'candidatos_evaluados': 'candidatos evaluados', 'reducciones_sin_efecto': 'actividades sueltas que no acortan',\n"
                       "               'actividades': 'se acorta', 'coste_paso': 'coste del paso', 'coste_acumulado': 'coste acumulado',\n"
                       "               'duracion_antes': 'duración antes', 'duracion_despues': 'duración después'}\n"
                       "cr.pasos[['cadena_critica', 'ackoff_sin_recursos', 'candidatos_evaluados', 'reducciones_sin_efecto', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']].rename(columns=COLUMNAS_CR)"))
    celdas.append(md("Detalle de cada paso: todos los candidatos evaluados (actividades reducibles de la cadena crítica "
                     "con recursos, sueltas y por parejas), su coste y la duración que resulta al replanificar con el "
                     "límite de operarios."))
    celdas.append(code("candidatos_crashing(proyecto, base, limite, datos_proyecto['cap_duration'], datos_proyecto['ucr'],\n"
                       "                    [acts.split(', ') for acts in cr.pasos['actividades']])"))
    celdas.append(code('#| tbl-cap: "Duraciones finales y semanas recortadas a cada actividad"\n#| label: tbl-duracionesCrashing\n'
                       "pd.DataFrame({'duración normal': datos_proyecto['duration'], 'duración tope': datos_proyecto['cap_duration'],\n"
                       "              'duración final': cr.duraciones.reindex(datos_proyecto.index).astype(int), 'recortado': cr.reducciones.reindex(datos_proyecto.index).astype(int),\n"
                       "              'CUR': datos_proyecto['ucr']}).T"))
    celdas.append(code("datos_cr = base.assign(duration=cr.duraciones.reindex(base.index).astype(int))\n"
                       "res_cr = proyecto.copy().asignar_multi(datos_cr, 'duration', ['operarios'], limite, report=False, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt con el límite de operarios tras el crashing con recursos"\n#| label: fig-ganttCrashing\n'
                       "pinta_tikz(res_cr.dibujos['operarios'])"))
    celdas.append(code("recortes = cr.reducciones[cr.reducciones > 0]\n"
                       "lista_cr = lista_y([a + ' ' + plural(x, 'semana') for a, x in recortes.items()])\n"
                       "costes_cr = ' + '.join(_num(c) for c in cr.pasos['coste_paso'])\n"
                       "logro = ('se consigue la reducción pedida' if cr.duracion_final <= res0.duracion - 2\n"
                       "         else f'solo se consigue {plural(res0.duracion - cr.duracion_final, \"semana\")} (parada: «{cr.motivo_parada}»)')\n"
                       "Markdown(f'**Respuesta:** se acortan {lista_cr}: la duración con recursos pasa de {res0.duracion} a {cr.duracion_final} semanas ({logro}) con un sobrecoste de '\n"
                       "         f'{costes_cr} = **{_num(cr.coste_total)} k€**, frente a los {_num(coste_ack)} k€ de Ackoff sin recursos, que no acortaban nada (el proyecto quedaba en {res_ack.duracion}). '\n"
                       "         'Cuesta más, pero es dinero que sí acorta el proyecto: se acortan actividades de la cadena crítica con recursos y se comprueba en cada paso, replanificando, que el proyecto termina antes.')"))
    celdas += apartado(4)
    celdas.append(md("En el modelo *con recurso*, acelerar una actividad consume operarios: cada semana recortada añade un "
                     "operario a su demanda, de modo que la reducción puede chocar con el mismo límite que intenta esquivar."))
    celdas.append(code('#| tbl-cap: "Crashing con recursos cuando acelerar consume un operario por semana recortada"\n#| label: tbl-crashingRecurso\n'
                       "extra = {a: {'operarios': 1} for a in datos_proyecto.index if np.isfinite(datos_proyecto.loc[a, 'ucr'])}\n"
                       "cr_b = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], limite,\n"
                       "                                         datos_proyecto['cap_duration'], datos_proyecto['ucr'], reduccion=2,\n"
                       "                                         demanda_extra=extra)\n"
                       "cr_b.pasos[['cadena_critica', 'candidatos_evaluados', 'actividades', 'coste_paso', 'coste_acumulado', 'duracion_antes', 'duracion_despues']].rename(columns=COLUMNAS_CR)"))
    celdas.append(code("candidatos_crashing(proyecto, base, limite, datos_proyecto['cap_duration'], datos_proyecto['ucr'],\n"
                       "                    [acts.split(', ') for acts in cr_b.pasos['actividades']], extra=1)"))
    celdas.append(code('#| tbl-cap: "Demanda de operarios de cada actividad tras las reducciones (modelo con recurso)"\n#| label: tbl-demandaRecurso\n'
                       "pd.DataFrame({'operarios (normal)': datos_proyecto['operarios'], 'operarios (tras recortar)': cr_b.demanda['operarios'].reindex(datos_proyecto.index),\n"
                       "              'recortado': cr_b.reducciones.reindex(datos_proyecto.index).astype(int)}).T"))
    celdas.append(code("recortes_b = cr_b.reducciones[cr_b.reducciones > 0]\n"
                       "lista_b = lista_y([a + ' ' + plural(x, 'semana') for a, x in recortes_b.items()]) if len(recortes_b) else 'ninguna actividad'\n"
                       "mismas = bool((cr_b.reducciones == cr.reducciones).all())\n"
                       "if cr_b.duracion_final <= res0.duracion - 2 and mismas:\n"
                       "    t = (f'no cambia: se acortan las mismas actividades ({lista_b}), con el mismo coste ({_num(cr_b.coste_total)} k€) y la misma duración final '\n"
                       "         f'({cr_b.duracion_final} semanas); el operario extra cabe en las semanas en que trabajan las actividades aceleradas.')\n"
                       "elif cr_b.duracion_final <= res0.duracion - 2:\n"
                       "    t = (f'sí cambia: se alcanza igualmente la reducción pedida ({cr_b.duracion_final} semanas), pero acortando {lista_b} en lugar de {lista_cr}, '\n"
                       "         f'con un sobrecoste de {_num(cr_b.coste_total)} k€ (modelo con dinero: {_num(cr.coste_total)} k€). Los candidatos que antes servían dejan de '\n"
                       "         'hacerlo porque el operario adicional no cabe en el límite o retrasa a otra actividad (tablas anteriores).')\n"
                       "else:\n"
                       "    t = (f'sí cambia: acortando {lista_b} solo se llega a {cr_b.duracion_final} semanas, {plural(res0.duracion - cr_b.duracion_final, \"semana\")} de las 2 pedidas, '\n"
                       "         f'con {_num(cr_b.coste_total)} k€; después ninguna reducción de una semana acorta el proyecto, porque el operario extra que exige acelerar no cabe '\n"
                       "         'en el límite o retrasa a otra actividad. Sin ampliar el recurso no se alcanza lo pedido.')\n"
                       "Markdown(f'**Respuesta:** {t}')"))
    celdas += apartado(5)
    celdas.append(code("mas = proyecto.copy().asignar_multi(base.copy(), 'duration', ['operarios'], [limite[0] + 1], report=True, tikz=True, params=GANTT_PARAMS)"))
    celdas.append(code('#| fig-cap: "Gantt con un operario más, sin acelerar ninguna actividad"\n#| label: fig-ganttMasOperarios\n'
                       "pinta_tikz(mas.dibujos['operarios'])"))
    celdas.append(code("objetivo = res0.duracion - 2\n"
                       "if mas.duracion <= objetivo:\n"
                       "    t = (f'con {limite[0] + 1} operarios y sin acelerar nada el proyecto dura {mas.duracion} semanas (antes {res0.duracion}): ahorra '\n"
                       "         f'{plural(res0.duracion - mas.duracion, \"semana\")} y ya cumple el objetivo de {objetivo} semanas. Acelerar con {limite[0]} operarios cuesta '\n"
                       "         f'{_num(cr.coste_total)} k€, así que compensa contratar si el operario adicional cuesta menos de {_num(cr.coste_total)} k€ en las {mas.duracion} semanas '\n"
                       "         f'del proyecto, es decir, menos de {_num(cr.coste_total)}/{mas.duracion} = {cr.coste_total / mas.duracion:.2f} k€ por semana.')\n"
                       "else:\n"
                       "    cr_mas = proyecto.copy().crashing_recursos(base.copy(), 'duration', ['operarios'], [limite[0] + 1],\n"
                       "                                               datos_proyecto['cap_duration'], datos_proyecto['ucr'], objetivo=objetivo)\n"
                       "    rec_mas = cr_mas.reducciones[cr_mas.reducciones > 0]\n"
                       "    t = (f'con {limite[0] + 1} operarios y sin acelerar nada el proyecto dura {mas.duracion} semanas (antes {res0.duracion}): ahorra '\n"
                       "         f'{plural(res0.duracion - mas.duracion, \"semana\")}, pero no llega al objetivo de {objetivo} semanas, así que por sí solo no basta. ')\n"
                       "    if cr_mas.duracion_final <= objetivo:\n"
                       "        t += (f'Para llegar habría que acelerar además {lista_y([a + \" \" + plural(x, \"semana\") for a, x in rec_mas.items()])} ({_num(cr_mas.coste_total)} k€, '\n"
                       "              f'frente a {_num(cr.coste_total)} k€ acelerando con {limite[0]} operarios): contratar compensa si el operario adicional durante las {objetivo} '\n"
                       "              f'semanas cuesta menos que la diferencia, {_num(cr.coste_total - cr_mas.coste_total)} k€.')\n"
                       "    else:\n"
                       "        t += f'Ni acelerando con {limite[0] + 1} operarios se llega al objetivo (se queda en {cr_mas.duracion_final} semanas).'\n"
                       "Markdown(f'**Respuesta:** {t}')"))
    return celdas, f"{n} actividades; límite {L} operarios, reducir dos semanas (Ackoff sin recursos falla)"


def _crashing_incertidumbre(rng):
    """Reducir con duraciones inciertas: la reducción determinista hasta el plazo solo da ~50 % de probabilidad."""
    estado = {}

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 5 <= i["n_rutas"] <= 10 and 12 <= i["T"] <= 24 and (i["D"] >= 3).all()):
            return False
        a, m, b = P.tres_estimaciones_triangular(rng, i["D"])
        if ((b - a) < 2).sum() > len(i["D"]) // 2:
            return False
        tope, cn, ct, ucr = P.costes_reduccion(rng, i["D"])
        ucr = ucr.replace(0.0, np.inf)
        if int(np.isfinite(ucr).sum()) < 4:
            return False
        pg = i["proyecto"]
        dist = {x: ("triangular", float(a[x]), float(m[x]), float(b[x])) for x in i["D"].index}
        df = P.tabla_base(i, duration=i["D"], a=a, m=m, b=b, cap_duration=tope, ucr=ucr)
        T = int(i["T"])
        try:
            t = pg.monte_carlo_crashing(dist, df, "duration", tope, ucr, plazo=T - 2, reduccion_max=8,
                                        n_samples=1500, seed=3)
        except Exception:
            return False
        if t.index.max() < 4:
            return False
        p = t["P(τ ≤ plazo)"]
        if not (0.05 <= p[0] <= 0.35 and p.max() >= 0.9):
            return False
        r90 = int(p[p >= 0.9].index.min())
        if not (3 <= r90 <= t.index.max()):
            return False
        # la reducción determinista justo hasta el plazo (2 meses) debe dar una probabilidad claramente <0,9
        if not (p.get(2, 1.0) <= 0.75 and r90 >= 3):
            return False
        estado.update(df=df, tope=tope, ucr=ucr, plazo=T - 2)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])),
                    n_fin=int(rng.choice([1, 2])), max_intentos=6000)
    df, plazo = estado["df"], estado["plazo"]
    T = int(info["T"])
    celdas = [md("# Enunciado"),
              md("Considere el proyecto cuyo cuadro de prelaciones, estimaciones de duración (mínima *a*, más probable *m* y "
                 "máxima *b*, semanas, distribución triangular), duración tope y coste unitario de reducción (k€/semana) se "
                 "adjunta; la columna *duration* es la duración esperada. Las actividades con CUR infinito no se pueden acelerar. "
                 f"El contrato exige entregar en **{plazo} semanas**.")]
    celdas.append(preguntas([
        f"Con las duraciones esperadas el proyecto dura {T} semanas. Simule 5 000 réplicas y estime la probabilidad de entregar "
        f"en {plazo} semanas sin acelerar.",
        f"Aplique Ackoff-Sasieni para reducir el proyecto hasta {plazo} semanas con las duraciones esperadas. "
        "¿Qué probabilidad de cumplir el plazo da ahora esa reducción? Compruébelo simulando.",
        "Construya la tabla de probabilidad de cumplir el plazo, duración media simulada y coste de reducción en función de las "
        "semanas recortadas (de 0 a 8). Use los mismos números aleatorios para todas las filas.",
        "¿Cuántas semanas hay que recortar para tener un 90 % de probabilidad de cumplir? ¿Cuánto cuesta frente a la "
        "reducción determinista? ¿Cuánto cuesta cada punto porcentual de probabilidad ganado en cada tramo?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       f"plazo = {plazo}\n"
                       "dists = {a: ('triangular', f.a, f.m, f.b) for a, f in datos_proyecto.iterrows()}\n"
                       "tope, ucr = datos_proyecto['cap_duration'], datos_proyecto['ucr']"))
    celdas.append(code(_AUX_REDUCCION))
    celdas.append(md("La duración esperada de una triangular es $(a + m + b)/3$; el cuadro la da ya redondeada en "
                     "*duration*. Con esas duraciones esperadas se calcula el calendario determinista."))
    celdas.append(code('#| tbl-cap: "Media de cada triangular frente a la duración esperada del cuadro"\n#| label: tbl-medias\n'
                       "pd.DataFrame({'(a + m + b)/3': [f\"({r['a']:g} + {r['m']:g} + {r['b']:g})/3 = {(r['a'] + r['m'] + r['b']) / 3:.2f}\" for _, r in datos_proyecto.iterrows()],\n"
                       "              'duration': datos_proyecto['duration']}, index=datos_proyecto.index).T"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones esperadas y el camino crítico", label="fig-pert")
    celdas.append(code('#| tbl-cap: "Duración esperada de cada ruta"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "tabla_rutas(proyecto, datos_proyecto['duration'])"))
    celdas += apartado(1)
    celdas.append(md("Cada réplica sortea una duración por actividad con su triangular; la duración del proyecto es el "
                     "máximo de las rutas (τ = ‖R·t‖∞). La probabilidad de cumplir es la fracción de réplicas con τ ≤ plazo."))
    celdas.append(code("mc0 = proyecto.monte_carlo(dists, n_samples=5000, seed=7)\n"
                       "display(mc0.summary().round(2))\n"
                       "p0 = float((mc0.tau_samples <= plazo).mean())\n"
                       "Markdown(f'Sin acelerar: duración media simulada {mc0.tau_mean:.2f} semanas (determinista: {proyecto.duration(datos_proyecto[\"duration\"])}) y P(τ ≤ {plazo}) = {p0:.3f}.')"))
    celdas.append(code('#| fig-cap: "Distribución simulada de la duración sin acelerar"\n#| label: fig-histograma\n'
                       "from matplotlib import pyplot as plt\n"
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "ax.hist(mc0.tau_samples, bins=40, density=True, alpha=0.6)\n"
                       "ax.axvline(plazo, color='red', linestyle='--', label=f'plazo ({plazo})')\n"
                       "ax.set_xlabel('duración del proyecto (semanas)'); ax.legend(); plt.show()"))
    celdas.append(code("tabla = proyecto.monte_carlo_crashing(dists, datos_proyecto, 'duration', tope, ucr, plazo=plazo,\n"
                       "                                       reduccion_max=8, n_samples=5000, seed=7)\n"
                       "p0_ent = float(tabla.loc[0, 'P(τ ≤ plazo)'])\n"
                       "Markdown(f'''**Respuesta:** sin acelerar, la probabilidad de entregar en {plazo} semanas es P(τ ≤ {plazo}) ≈ **{p0:.3f}** ({100 * p0:.1f} % de las 5 000 réplicas), "
                       "con una duración media simulada de {mc0.tau_mean:.2f} semanas frente a las {proyecto.duration(datos_proyecto[\"duration\"])} deterministas. "
                       "Para los apartados siguientes se usa `monte_carlo_crashing`, que repite la simulación para cada número de semanas recortadas con los mismos números aleatorios "
                       "y **redondea cada duración sorteada a semanas enteras** (mínimo 1), como el resto del curso; con ese redondeo, sin recortar sale una media de "
                       "{tabla.loc[0, \"τ media\"]:.2f} semanas y P(τ ≤ {plazo}) = {p0_ent:.3f}. La diferencia con {p0:.3f} se debe al redondeo: una duración simulada de "
                       "{plazo}.4 semanas, por ejemplo, cuenta como {plazo} y cumple el plazo.''')"))
    celdas += apartado(2)
    celdas.append(code("r_det = int(tabla.index[tabla['duración determinista'] <= plazo].min())\n"
                       "Markdown(f'Con las duraciones esperadas el proyecto dura {proyecto.duration(datos_proyecto[\"duration\"])} semanas: hay que recortar '\n"
                       "         f'{proyecto.duration(datos_proyecto[\"duration\"])} − {plazo} = {plural(r_det, \"semana\")} para llegar a {plazo}.')"))
    celdas.append(md("En cada paso del algoritmo de Ackoff-Sasieni se acorta una semana la ruta crítica (o las rutas críticas) "
                     "eligiendo, entre las actividades críticas con margen, la combinación de menor CUR que reduce "
                     "*todas* las rutas críticas a la vez."))
    celdas.append(code('#| tbl-cap: "Algoritmo de Ackoff-Sasieni con las duraciones esperadas hasta el plazo"\n#| label: tbl-ackoff\n'
                       "tabla_ackoff, pasos, duraciones_reducidas, margen_restante = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=tope, costs=ucr, reduction=r_det, report=False)\n"
                       "tabla_ackoff"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPaso\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], tope, ucr, pasos)"))
    celdas.append(code("narra_ackoff(proyecto, datos_proyecto['duration'], tope, ucr, pasos)"))
    celdas.append(code("resumen = pd.DataFrame({'actividades acortadas': [', '.join(acts) for acts in pasos.values()],\n"
                       "                        'coste del paso': [float(sum(ucr[acts])) for acts in pasos.values()]},\n"
                       "                       index=pd.Index(range(1, len(pasos) + 1), name='paso'))\n"
                       "resumen['sobrecoste acumulado'] = resumen['coste del paso'].cumsum()\n"
                       "resumen"))
    celdas.append(code("p_det = float(tabla.loc[r_det, 'P(τ ≤ plazo)'])\n"
                       "if p_det < 0.5:\n"
                       "    matiz = 'menos de la mitad de las veces: con la duración esperada justo en el plazo, el sesgo de fusión (en cada réplica manda la ruta que más se alarga) empuja la media por encima del plazo'\n"
                       "elif p_det < 0.75:\n"
                       "    matiz = 'poco más de la mitad de las veces: la duración esperada queda justo en el plazo y, por el sesgo de fusión de las rutas, muchas réplicas lo superan'\n"
                       "else:\n"
                       "    matiz = 'algo más de lo esperable al poner la media justo en el plazo, por el redondeo a semanas enteras, pero aún por debajo de lo que suele exigirse'\n"
                       "Markdown(f'''**Respuesta:** reducir las duraciones esperadas hasta {plazo} semanas exige {plural(r_det, \"semana\")} de reducción y "
                       "{' + '.join(_num(c) for c in resumen['coste del paso'])} = {tabla.loc[r_det, \"coste\"]:.0f} k€ (duración determinista {proyecto.duration(duraciones_reducidas)} semanas); "
                       "pero, simulando, la probabilidad de cumplir es solo P(τ ≤ {plazo}) = **{p_det:.3f}**: se cumple {matiz}. Está lejos del 90 % que suele exigirse.''')"))
    celdas += apartado(3)
    celdas.append(md("Para cada número de semanas recortadas $r$, la tabla da las reducciones que aplica Ackoff-Sasieni "
                     "(una por paso, separadas por «;»), el coste acumulado, la duración determinista, la media y el "
                     "percentil 90 simulados y la probabilidad de cumplir, siempre con las mismas 5 000 réplicas de "
                     "duraciones redondeadas a semanas enteras (a cada réplica se le restan las semanas recortadas a "
                     "cada actividad, sin bajar de 1 semana)."))
    celdas.append(code('#| tbl-cap: "Probabilidad de cumplir el plazo según las semanas recortadas"\n#| label: tbl-crashingMC\n'
                       "acortadas = {0: '—'}\n"
                       "for r in tabla.index[1:]:\n"
                       "    _, p_r, _, _ = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=tope, costs=ucr, reduction=int(r), report=False)\n"
                       "    acortadas[r] = '; '.join(' + '.join(p_r[j]) for j in sorted(p_r))\n"
                       "tabla.insert(0, 'actividades acortadas (paso a paso)', pd.Series(acortadas))\n"
                       "tabla.round(3)"))
    celdas.append(code('#| fig-cap: "Probabilidad de cumplir y coste de reducción según las semanas recortadas"\n#| label: fig-probCoste\n'
                       "fig, ax = plt.subplots(figsize=(8, 4))\n"
                       "ax.plot(tabla.index, tabla['P(τ ≤ plazo)'], marker='o', label='P(τ ≤ plazo)'); ax.axhline(0.9, color='grey', linestyle=':')\n"
                       "ax.set_xlabel('semanas recortadas'); ax.set_ylabel('probabilidad')\n"
                       "ax2 = ax.twinx(); ax2.plot(tabla.index, tabla['coste'], marker='s', color='tab:red', label='coste (k€)'); ax2.set_ylabel('k€')\n"
                       "ax.legend(loc='upper left'); ax2.legend(loc='lower right'); plt.show()"))
    celdas.append(code("r_max = int(tabla.index.max())\n"
                       "tope_txt = '' if r_max >= 8 else (f' No se puede recortar más de {plural(r_max, \"semana\")}: después no queda ninguna combinación con margen que acorte '\n"
                       "                                   'todas las rutas críticas (Ackoff-Sasieni se detiene).')\n"
                       "P_ = tabla['P(τ ≤ plazo)']; C_ = tabla['coste']; M_ = tabla['τ media']\n"
                       "Markdown(f'**Respuesta:** la tabla anterior. Al recortar, la probabilidad de cumplir sube de {P_[0]:.3f} (sin recortar) a {P_[r_max]:.3f} '\n"
                       "         f'({plural(r_max, \"semana\")} recortadas, {C_[r_max]:.0f} k€) y la duración media simulada baja de {M_[0]:.2f} a {M_[r_max]:.2f} semanas,'\n"
                       "         + (' siempre algo por encima de la determinista (sesgo de fusión).' if (M_ >= tabla['duración determinista'] - 1e-9).all() else '.') + tope_txt)"))
    celdas += apartado(4)
    celdas.append(md("Coste de cada punto porcentual de probabilidad en cada tramo: lo que cuesta recortar esa semana "
                     "dividido entre los puntos que gana la probabilidad, $\\Delta\\text{coste} / (100\\,\\Delta P)$."))
    celdas.append(code("r90 = int(tabla.index[tabla['P(τ ≤ plazo)'] >= 0.9].min())\n"
                       "P_ = tabla['P(τ ≤ plazo)']; C_ = tabla['coste']\n"
                       "filas = {}\n"
                       "for r in tabla.index[1:]:\n"
                       "    dc, dp = C_[r] - C_[r - 1], 100 * (P_[r] - P_[r - 1])\n"
                       "    filas[f'{r - 1} → {r}'] = {'Δcoste (k€)': f'{_num(C_[r])} − {_num(C_[r - 1])} = {_num(dc)}',\n"
                       "                               'Δprobabilidad (puntos)': f'{100 * P_[r]:.1f} − {100 * P_[r - 1]:.1f} = {dp:.1f}',\n"
                       "                               'k€ por punto': (f'{_num(dc)} / {dp:.1f} = {dc / dp:.2f}' if dp > 0 else 'no gana probabilidad')}\n"
                       "marginal = pd.DataFrame(filas).T\n"
                       "marginal.index.name = 'semanas recortadas'\n"
                       "display(marginal)\n"
                       "por_punto = {k: (C_[int(k.split(' → ')[1])] - C_[int(k.split(' → ')[0])]) / (100 * (P_[int(k.split(' → ')[1])] - P_[int(k.split(' → ')[0])]))\n"
                       "             for k in filas if P_[int(k.split(' → ')[1])] > P_[int(k.split(' → ')[0])]}\n"
                       "caro = max(por_punto, key=por_punto.get)\n"
                       "fin_txt = (': los últimos puntos, cuando la probabilidad ya está cerca de 1, son los más caros.' if caro == list(por_punto)[-1]\n"
                       "           else '. En general los últimos puntos, cuando la probabilidad ya está cerca de 1, son los más caros.')\n"
                       "barato = min(por_punto, key=por_punto.get)\n"
                       "Markdown(f'''**Respuesta:** para tener al menos un 90 % de probabilidad hay que recortar **{plural(r90, \"semana\")}** (P = {P_[r90]:.3f}), con un coste de "
                       "**{C_[r90]:.0f} k€**, frente a {plural(r_det, \"semana\")} y {C_[r_det]:.0f} k€ de la reducción determinista hasta el plazo (P = {P_[r_det]:.3f}): la diferencia, "
                       "{C_[r90] - C_[r_det]:.0f} k€, es lo que cuesta la seguridad. El punto de probabilidad más barato se compra en el tramo {barato} ({por_punto[barato]:.2f} k€) y el más caro "
                       "en el tramo {caro} ({por_punto[caro]:.2f} k€){fin_txt}''')"))
    return celdas, f"duraciones triangulares; plazo de {plazo} semanas; reducción y probabilidad"


VARIANTES = [
    Variante("cur-y-ackoff", 1, "CUR a partir de costes normal y tope; Ackoff-Sasieni tres semanas con un paso de dos actividades",
             lambda rng: _base(1, rng)),
    Variante("costes-indirectos", 2, "curva coste-duración con costes indirectos y duración de coste mínimo", lambda rng: _base(2, rng)),
    Variante("penalizacion", 2, "reducir frente a pagar penalización; actividades no acelerables", lambda rng: _base(3, rng)),
    Variante("barata-no-optima", 2, "dos rutas críticas: la crítica más barata no sirve si no es común a ambas", _barata_no_optima),
    Variante("desaceleracion", 2, "reducido el proyecto, el plazo se relaja: qué paso deshacer (el más caro) y por qué la curva coste-duración es convexa", _desaceleracion),
    Variante("programacion-lineal", 3, "reducción óptima por programación lineal con la matriz de caminos frente al voraz de Ackoff-Sasieni",
             _programacion_lineal),
    Variante("cur-por-tramos", 3, "CUR creciente por tramos: Ackoff con coste marginal frente a un CUR medio; por qué funciona si el coste es convexo",
             _cur_por_tramos),
    Variante("crashing-recursos", 3, "reducir la duración con un límite de operarios: Ackoff sin recursos falla; cadena crítica con recursos, modelo con recurso y contratar frente a acelerar",
             _crashing_recursos),
    Variante("crashing-incertidumbre", 3, "reducir con duraciones triangulares: la reducción determinista hasta el plazo da poca probabilidad; coste de la seguridad",
             _crashing_incertidumbre),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
