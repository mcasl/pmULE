"""Colección 05 — Holguras total, libre e independiente, caminos críticos y
su interpretación."""
from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe, literal_dict, BMATRIX, grafo_nombre)

def _pl(n, singular, plural=None):
    """«1 semana», «3 semanas»: número y sustantivo concordados (para enunciados y subtítulos)."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


META = dict(slug="05_holguras", prefijo="Holguras",
            titulo="Holguras y camino crítico",
            descripcion="Calcular e interpretar las holguras total, libre e independiente; identificar "
                        "todos los caminos críticos.",
            n_defecto=3)


# Celda auxiliar (se define en cada cuaderno, sin salida): textos de respuesta calculados con
# los números del calendario, para que cada conclusión sea la de *ese* proyecto.
_AUX_HOLGURAS = r'''
_COLS_CAL = {'inicio_mas_temprano': 'ES', 'fin_mas_temprano': 'EF', 'inicio_mas_tardio': 'LS',
             'fin_mas_tardio': 'LF', 'H_total': 'H total', 'H_libre': 'H libre', 'H_independiente': 'H independiente',
             'duracion': 'duración'}


def _sucesoras(prelaciones):
    pred = prelaciones.immediate_predecessor
    return {a: sorted(s for s, ps in pred.items() if a in ps) for a in pred}


def _v(cal, a, col):
    return int(cal.loc[a, col])


def _verbo(n, singular, plural_):
    return singular if n == 1 else plural_


def _min_es_suc(cal, suc, a):
    """(mín ES de las sucesoras, expresión «mín(E: 12, G: 12)», «9 (ES de G)» o «19 (fin del proyecto)»)."""
    if not suc[a]:
        fin = int(cal['fin_mas_temprano'].max())
        return fin, f'{fin} (fin del proyecto)'
    es = {s: _v(cal, s, 'inicio_mas_temprano') for s in suc[a]}
    m = min(es.values())
    if len(es) == 1:
        return m, f'{m} (ES de {suc[a][0]})'
    return m, 'mín(' + ', '.join(f'{s}: {v}' for s, v in es.items()) + ')'


def respuesta_rutas(tabla_rutas, unidad='semana'):
    """Respuesta al «¿cuántas rutas críticas?» a partir de la tabla de explica_rutas."""
    dur = tabla_rutas['duración'].astype(float)
    T = int(round(dur.max()))
    criticas = tabla_rutas.loc[tabla_rutas['crítica'] == 'sí', 'ruta'].tolist()
    resto = tabla_rutas[tabla_rutas['crítica'] == 'no']
    acts = []
    for r in criticas:
        for a in r.split(' – '):
            if a not in acts:
                acts.append(a)
    n = len(criticas)
    txt = (f"**Respuesta:** el proyecto tiene **{plural(n, 'ruta crítica', 'rutas críticas')}**, "
           f"{lista_y('**' + r + '**' for r in criticas)}, "
           + (f"que suma {plural(T, unidad)}" if n == 1 else f"que suman {'ambas' if n == 2 else 'todas'} {plural(T, unidad)}")
           + f", la duración del proyecto. Actividades críticas (holgura total nula): {lista_y(sorted(acts))}.")
    if len(resto) == 1:
        segunda = int(round(float(resto['duración'].iloc[0])))
        txt += (f" La otra ruta, {resto['ruta'].iloc[0]}, suma {segunda} y queda a "
                f"{plural(T - segunda, unidad)} de ser crítica.")
    elif len(resto):
        d2 = resto['duración'].astype(float)
        segunda = int(round(d2.max()))
        sig = resto.loc[np.isclose(d2, d2.max()), 'ruta'].tolist()
        txt += (f" Las otras {plural(len(resto), 'ruta suma', 'rutas suman')} menos; "
                + (f"la más larga de ellas, {sig[0]} ({segunda}), queda" if len(sig) == 1
                   else f"las más largas de ellas, {lista_y(sig)} ({segunda}), quedan")
                + f" a {plural(T - segunda, unidad)} de ser crítica{'s' if len(sig) > 1 else ''}.")
    else:
        txt += ' Todas las rutas del grafo son críticas.'
    return Markdown(txt)


def respuesta_retrasables(calendario, prelaciones, unidad='semana'):
    """Qué actividades pueden retrasarse sin afectar a nadie, cuáles solo arrastrando a sus
    sucesoras y cuáles no admiten retraso, con las holguras sustituidas."""
    cal, suc = calendario, _sucesoras(prelaciones)
    criticas = [a for a in cal.index if _v(cal, a, 'H_total') == 0]
    libres = [a for a in cal.index if _v(cal, a, 'H_libre') > 0]
    arrastran = [a for a in cal.index if _v(cal, a, 'H_total') > _v(cal, a, 'H_libre')]
    L = ['**Respuesta:**', '']
    L.append(f"- **No admiten ningún retraso** sin alargar el plazo las críticas (holgura total 0): {lista_y(criticas)}.")
    L.append('- **Pueden retrasarse sin afectar ni al plazo ni al inicio más temprano de ninguna otra actividad** '
             'las de holgura libre positiva, como mucho su holgura libre (H libre = mín ES de las sucesoras − EF): '
             + ('; '.join(f"**{a}**, hasta {plural(_v(cal, a, 'H_libre'), unidad)} "
                          f"(H libre = {_min_es_suc(cal, suc, a)[1]} − {_v(cal, a, 'fin_mas_temprano')} = {_v(cal, a, 'H_libre')})"
                          for a in libres) if libres else 'ninguna') + '.')
    L.append('- **Pueden retrasarse sin afectar al plazo pero obligando a retrasar a alguna sucesora** las de holgura '
             'total mayor que la libre: ' + (lista_y(f'**{a}**' for a in arrastran) if arrastran else 'ninguna') + '.')
    for a in arrastran:
        ht, hl, ef = _v(cal, a, 'H_total'), _v(cal, a, 'H_libre'), _v(cal, a, 'fin_mas_temprano')
        mueve = []
        for s in suc[a]:
            es = _v(cal, s, 'inicio_mas_temprano')
            if es < ef + ht:
                t = es - ef
                mueve.append(f'cualquier retraso suyo mueve a {s} (ES de {s} = {es} = EF de {a})' if t == 0 else
                             f'un retraso de más de {plural(t, unidad)} mueve a {s} (ES de {s} = {es} = EF de {a} + {t})')
        L.append(f"    - {a}: H total = {ht} > H libre = {hl}. Puede retrasarse hasta {plural(ht, unidad)} sin alargar el plazo"
                 + (f' (y hasta {plural(hl, unidad)} sin mover a nadie)' if hl else '') + ', pero ' + '; '.join(mueve) + '.')
    return Markdown('\n'.join(L))


def analiza_retraso(proyecto, prelaciones, calendario, duraciones, X, h, unidad='semana'):
    """Retrasa X h unidades (alarga su duración) y devuelve (tabla antes/después de las actividades
    que cambian, texto de la respuesta, duraciones con el retraso)."""
    cal0, suc, pred = calendario, _sucesoras(prelaciones), prelaciones.immediate_predecessor
    d1 = duraciones.copy()
    d1[X] += h
    cal1 = proyecto.calendar(d1)
    cols = list(_COLS_CAL)[:6]
    cambian = [a for a in cal0.index if any(_v(cal0, a, c) != _v(cal1, a, c) for c in cols)]
    tabla = pd.concat({'antes': cal0.loc[cambian, cols].astype(int).rename(columns=_COLS_CAL),
                       'después': cal1.loc[cambian, cols].astype(int).rename(columns=_COLS_CAL)}, axis=1)
    tabla.index.name = 'actividad'
    us = plural(2, unidad).split(' ', 1)[1]
    T0, T1 = int(proyecto.duration(duraciones)), int(proyecto.duration(d1))
    finales = [a for a in cal1.index if not suc[a]]
    fines = ', '.join(f"{a}: {_v(cal1, a, 'fin_mas_temprano')}" for a in finales)
    L = [f"**Respuesta.** **El plazo no cambia**: el fin del proyecto es el mayor fin más temprano de las actividades "
         f"finales, máx({fines}) = **{T1}** {us}, igual que antes ({T0})."]
    hl = _v(cal0, X, 'H_libre')
    tx = (f"- **{X}** acaba en {_v(cal1, X, 'fin_mas_temprano')} en lugar de {_v(cal0, X, 'fin_mas_temprano')} "
          f"y su holgura total pasa de {h} a {_v(cal1, X, 'H_total')}: la ha consumido entera. ")
    if hl == 0:
        tx += 'Como su holgura libre era 0, todo el retraso empuja a sus sucesoras.'
    else:
        tx += (f"De las {plural(h, unidad)} de retraso, {hl} {_verbo(hl, 'cabe', 'caben')} en su holgura libre "
               f"y {h - hl} {_verbo(h - hl, 'empuja', 'empujan')} a sus sucesoras.")
    L.append(tx)
    movidas = [a for a in cambian if a != X and _v(cal1, a, 'inicio_mas_temprano') != _v(cal0, a, 'inicio_mas_temprano')]

    def cadena(a):
        camino = [a]
        while camino[-1] != X:
            b = camino[-1]
            es_b = _v(cal1, b, 'inicio_mas_temprano')
            p = [p for p in sorted(pred[b]) if (p == X or p in movidas) and _v(cal1, p, 'fin_mas_temprano') == es_b][0]
            camino.append(p)
        return ' → '.join(reversed(camino))

    def cambios(a, columnas):
        return ', '.join(f'{_COLS_CAL[c]} {_v(cal0, a, c)} → {_v(cal1, a, c)}' for c in columnas
                         if _v(cal0, a, c) != _v(cal1, a, c))

    if movidas:
        partes = []
        for a in movidas:
            p = f"{a} ({cambios(a, ['inicio_mas_temprano', 'H_total', 'H_libre'])}"
            p += ')' if a in suc[X] else f'; le llega en cadena: {cadena(a)})'
            partes.append(p)
        L.append(f"- **Se retrasa{'n' if len(movidas) > 1 else ''}** (cambia su inicio más temprano): " + '; '.join(partes) + '.')
    quietas = [s for s in suc[X] if s not in movidas]
    for s in quietas:
        es, ef_x = _v(cal1, s, 'inicio_mas_temprano'), _v(cal1, X, 'fin_mas_temprano')
        fija = [p for p in sorted(pred[s]) if p != X and _v(cal1, p, 'fin_mas_temprano') == es][0]
        L.append(f"- {s}, sucesora de {X}, no se mueve: su inicio más temprano ({es}) lo fija {fija}, que acaba en {es}, "
                 + (f"y {X} acaba ahora en {ef_x}, antes." if ef_x < es else
                    f"y {X} acaba ahora justo entonces, en {ef_x}: por eso {s} no se retrasa, pero {X} ya no tiene ningún margen."))
    pierden = [a for a in cambian if a != X and a not in movidas and _v(cal1, a, 'H_total') < _v(cal0, a, 'H_total')]
    if pierden:
        L.append(f"- **{_verbo(len(pierden), 'Pierde', 'Pierden')} holgura sin moverse** "
                 + '; '.join(f"{a} ({cambios(a, ['inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'H_libre'])})" for a in pierden) + ': '
                 f"{_verbo(len(pierden), 'va', 'van')} antes de {X} en la misma cadena y, como {X} dura ahora más, "
                 f"su fin más tardío se adelanta.")
    ganan = [a for a in cambian if a != X and a not in movidas and a not in pierden]
    if ganan:
        L.append(f"- **{_verbo(len(ganan), 'Gana', 'Ganan')} holgura libre** "
                 + '; '.join(f"{a} ({cambios(a, ['H_libre'])})" for a in ganan) + ': alguna de sus sucesoras empieza ahora más tarde.')
    nuevas = [a for a in cal0.index if _v(cal0, a, 'H_total') > 0 and _v(cal1, a, 'H_total') == 0]
    conservan = [a for a in cambian if _v(cal1, a, 'H_total') > 0]
    tx = (f"- **{_verbo(len(nuevas), 'Pasa', 'Pasan')} a ser crítica{'s' if len(nuevas) > 1 else ''}** (holgura total 0): "
          f"{lista_y(nuevas)}.")
    if conservan:
        tx += (f" {_verbo(len(conservan), 'Conserva', 'Conservan')} parte de su holgura: "
               + lista_y(f"{a} (H total {_v(cal1, a, 'H_total')})" for a in conservan) + '.')
    L.append(tx)
    r0 = {frozenset(r) for r in proyecto.critical_path(duraciones).values()}
    r1 = [r for r in proyecto.critical_path(d1).values()]
    rn = [' – '.join(ordena_ruta(r, cal1)) for r in r1 if frozenset(r) not in r0]
    L.append(f"- **{_verbo(len(rn), 'Aparece', 'Aparecen')} {plural(len(rn), 'ruta crítica nueva', 'rutas críticas nuevas')}** "
             f"({lista_y(rn)}): el proyecto pasa de {plural(len(r0), 'ruta crítica', 'rutas críticas')} a {len(r1)}:")
    return tabla, L[0] + '\n\n' + '\n'.join(L[1:]), d1


def respuesta_independiente(calendario, prelaciones, V, unidad='semana'):
    """Retraso disponible para V si sus predecesoras acaban en su fecha más tardía (holgura independiente)."""
    cal, suc = calendario, _sucesoras(prelaciones)
    ps = sorted(prelaciones.immediate_predecessor[V])
    lf = {p: _v(cal, p, 'fin_mas_tardio') for p in ps}
    inicio = max(lf.values())
    fin, fin_txt = _min_es_suc(cal, suc, V)
    if fin_txt.startswith('mín'):
        fin_txt += f' = {fin}'
    d, ef, es = _v(cal, V, 'duracion'), _v(cal, V, 'fin_mas_temprano'), _v(cal, V, 'inicio_mas_temprano')
    hi, hl = _v(cal, V, 'H_independiente'), _v(cal, V, 'H_libre')
    margen = fin - inicio - d
    lf_txt = (f'máx(' + ', '.join(f'LF de {p}: {v}' for p, v in lf.items()) + f') = {inicio}') if len(lf) > 1 else f'LF de {ps[0]} = {inicio}'
    txt = (f"Si {_verbo(len(ps), 'la predecesora', 'las predecesoras')} de {V} ({lista_y(ps)}) {_verbo(len(ps), 'acaba', 'acaban')} en su fecha más tardía, {V} no puede empezar hasta "
           f"{lf_txt}. Para no mover el inicio más temprano de ninguna sucesora debe acabar, como tarde, en {fin_txt}. "
           f"Le queda {fin} − {inicio} − {d} = {margen}")
    if margen < 0:
        txt = txt[:-len(str(margen))] + f"−{-margen} < 0: ningún margen, y la holgura independiente es máx(0, −{-margen}) = 0."
    else:
        txt += f", que es su holgura independiente."
    txt += (f" Su holgura libre, {fin} − {ef} = {hl}, supone en cambio que las predecesoras acaban lo antes posible "
            f"(entonces {V} empieza en {es} y acaba en {ef}).")
    txt += ((f"\n\n**Respuesta:** {V} puede retrasarse **{plural(hi, unidad)}** sin afectar a nadie aunque sus predecesoras "
             f"agoten su holgura" if hi else f"\n\n**Respuesta:** {V} **no puede retrasarse nada** (0 {plural(2, unidad).split()[1]}) sin "
             f"afectar a alguna sucesora si sus predecesoras agotan su holgura")
            + f": es su holgura independiente ({hi}), menor que su holgura libre ({hl}).")
    return Markdown(txt)


def respuesta_compartida(calendario, prelaciones, X, Y, h, unidad='semana'):
    """Razonamiento previo (solo con el calendario original) de si caben los dos retrasos."""
    cal, suc = calendario, _sucesoras(prelaciones)
    ef_x, es_y, ht_y = _v(cal, X, 'fin_mas_temprano'), _v(cal, Y, 'inicio_mas_temprano'), _v(cal, Y, 'H_total')
    g = es_y - ef_x
    empuje = max(0, h - g)
    queda = ht_y - empuje
    alarga = max(0, h - queda)
    txt = (f"**Respuesta:** {'no' if alarga else 'sí'}. {X} {'e' if Y.startswith('I') else 'y'} {Y} tienen la misma holgura "
           f"total ({h}) porque es la holgura de la cadena en la que están. {X} tiene holgura libre 0 "
           f"(H libre = {_min_es_suc(cal, suc, X)[1]} − {ef_x} = {_v(cal, X, 'H_libre')})")
    if g == 0:
        txt += (f" y {Y} empieza en cuanto acaba {X} (ES de {Y} = {es_y} = EF de {X}): cada {unidad} de retraso de {X} "
                f"retrasa el inicio de {Y} y le quita una {unidad} de holgura.")
    else:
        txt += (f"; {Y} empieza {plural(g, unidad)} después de que acabe {X} (ES de {Y} = {es_y} = EF de {X} + {g}): "
                f"solo el retraso de {X} que pase de {plural(g, unidad)} retrasa a {Y}.")
    txt += (f" Si {X} gasta {'su' if h == 1 else 'sus'} {plural(h, unidad)}, {Y} empieza {plural(empuje, unidad)} más tarde y le "
            f"{_verbo(queda, 'queda', 'quedan')} {ht_y} − {empuje} = {plural(queda, unidad)} de holgura total")
    if alarga:
        txt += (f"; si además {Y} se retrasa {plural(h, unidad)}, el proyecto se alarga {h} − {queda} = {plural(alarga, unidad)}.")
    else:
        txt += f", suficientes para su retraso de {plural(h, unidad)}."
    return Markdown(txt)


def respuesta_reparto(proyecto, calendario, duraciones, X, Y, h, unidad='semana'):
    """Holgura libre de X, independiente de Y y repartos (x, y) que respetan el plazo."""
    cal = calendario
    hl_x, hi_y = _v(cal, X, 'H_libre'), _v(cal, Y, 'H_independiente')
    T = int(proyecto.duration(duraciones))
    ymax = {}
    for x in range(h + 1):
        d = duraciones.copy()
        d[X] += x
        ymax[x] = _v(proyecto.calendar(d), Y, 'H_total') if int(proyecto.duration(d)) == T else None
    conj = 'e' if Y.startswith('I') else 'y'
    txt = (f"La holgura **libre** de {X} ({hl_x}) es lo que {X} puede retrasarse sin mover el inicio de {Y} ni quitarle "
           f"margen; la holgura **independiente** de {Y} ({hi_y}) es lo que {Y} puede retrasarse aunque {X} agote su "
           f"holgura y acabe lo más tarde posible.")
    if hl_x == 0 and hi_y == 0:
        txt += ' Las dos son 0: ninguna de las dos puede retrasarse sin gastar holgura que también necesita la otra.'
    if all(v is not None and x + v == h for x, v in ymax.items()):
        reparto = (f"si {X} se retrasa x y {Y} se retrasa y, el plazo se respeta mientras x + y ≤ {h}"
                   + (f" (por ejemplo, {(h + 1) // 2} para {X} {conj} {h // 2} para {Y}, o más para la que tenga más riesgo de retrasarse)"
                      if h > 1 else f": la única {unidad} se concede a una de las dos, la que más la necesite"))
    else:
        reparto = (f"con {X} retrasada x, a {Y} le queda de holgura total: "
                   + ', '.join(f'x = {x} → {v}' for x, v in ymax.items() if v is not None))
    txt += (f"\n\n**Respuesta:** la holgura libre (de {X}) y la independiente (de {Y}). Las {plural(h, unidad)} son un "
            f"recurso compartido de la cadena: {reparto}; lo que no cabe es que cada una disponga de {plural(h, unidad)} "
            f"por su cuenta.") if h > 1 else (f"\n\n**Respuesta:** la holgura libre (de {X}) y la independiente (de {Y}). "
            f"Hay una sola {unidad} de holgura compartida: {reparto}.")
    return Markdown(txt)


def respuesta_vigilancia(orden, n=4):
    """Las n actividades a vigilar según la ordenación, explicando los desempates."""
    crit = [a for a in orden.index if int(orden.loc[a, 'H_total']) == 0]
    elegidas = list(orden.index[:n])
    nc = len(crit)
    dur = lambda a: int(orden.loc[a, 'duracion'])
    if nc > n:
        txt = (f"Hay {nc} actividades críticas (holgura total 0), más de las {n} que se pueden vigilar; entre ellas se "
               f"desempata por duración (una actividad larga tiene más ocasiones de retrasarse): "
               + ', '.join(f'{a} ({dur(a)})' for a in crit) + '.')
        ult = dur(elegidas[-1])
        emp = [a for a in crit if dur(a) == ult]
        if any(a not in elegidas for a in emp):
            txt += (f" {lista_y(emp)} empatan con {plural(ult, 'semana')}: la última plaza puede ser para cualquiera "
                    f"de ellas (la tabla pone primero {elegidas[-1]}).")
    elif nc == n:
        txt = f"Las {n} críticas ({lista_y(crit)}) ocupan justo las {n} plazas."
    else:
        resto = [a for a in orden.index if a not in crit]
        txt = (f"Las {nc} críticas ({lista_y(crit)}) van primero; las {n - nc} plazas restantes son para las no críticas "
               f"de menor holgura total (a igualdad, menor holgura libre): "
               + ', '.join(f"{a} (H total {int(orden.loc[a, 'H_total'])}, H libre {int(orden.loc[a, 'H_libre'])})" for a in resto[:n - nc + 1])
               + '.')
        ult = elegidas[-1]
        clave = lambda a: (int(orden.loc[a, 'H_total']), int(orden.loc[a, 'H_libre']))
        emp = [a for a in resto if clave(a) == clave(ult)]
        if any(a not in elegidas for a in emp):
            txt += (f" {lista_y(emp)} empatan en las dos holguras: se desempata por duración (la más larga primero).")
    txt += f"\n\n**Respuesta:** las cuatro actividades a vigilar son **{lista_y(elegidas)}**."
    return Markdown(txt)


def respuesta_sin_margen(calendario, prelaciones, unidad='semana'):
    """No críticas con holgura libre nula: a quién mueven y por qué no son críticas."""
    cal, suc = calendario, _sucesoras(prelaciones)
    sin = [a for a in cal.index if _v(cal, a, 'H_total') > 0 and _v(cal, a, 'H_libre') == 0]
    L = []
    for a in sin:
        ef = _v(cal, a, 'fin_mas_temprano')
        pegadas = [s for s in suc[a] if _v(cal, s, 'inicio_mas_temprano') == ef]
        L.append(f"- **{a}**: H libre = {_min_es_suc(cal, suc, a)[1]} − {ef} = 0, pero H total = LS − ES = "
                 f"{_v(cal, a, 'inicio_mas_tardio')} − {_v(cal, a, 'inicio_mas_temprano')} = {_v(cal, a, 'H_total')}. "
                 f"Cualquier retraso suyo mueve a {lista_y(pegadas)} (empieza{'n' if len(pegadas) > 1 else ''} en {ef}, "
                 f"cuando acaba {a}), que lo absorbe{'n' if len(pegadas) > 1 else ''} con su holgura total ("
                 + ', '.join(f"{s}: {_v(cal, s, 'H_total')}" for s in pegadas) + ').')
    L.append(f"\n**Respuesta:** {lista_y(sin)} no {'admite' if len(sin) == 1 else 'admiten'} ni una {unidad} de retraso "
             f"sin mover a una sucesora (holgura libre 0), pero no {'es crítica' if len(sin) == 1 else 'son críticas'} porque "
             f"su holgura total es positiva: mientras el retraso no la supere, el plazo no cambia. Crítica significa que "
             f"el plazo depende de la actividad (holgura total 0), no que su retraso mueva a otra actividad.")
    return Markdown('\n'.join(L))


def respuesta_mas_libre(calendario, prelaciones, unidad='semana'):
    """Actividad (o actividades) con mayor holgura libre."""
    cal, suc = calendario, _sucesoras(prelaciones)
    m = int(cal['H_libre'].max())
    quien = [a for a in cal.index if _v(cal, a, 'H_libre') == m]
    txt = (f"**Respuesta:** {lista_y(quien)}, hasta **{plural(m, unidad)}**: {'es la' if len(quien) == 1 else 'son las'} "
           f"de mayor holgura libre (" + '; '.join(f"{a}: H libre = {_min_es_suc(cal, suc, a)[1]} − {_v(cal, a, 'fin_mas_temprano')} = {m}" for a in quien)
           + "). Ese retraso no mueve el inicio más temprano de ninguna otra actividad ni el plazo.")
    for a in quien:
        hi = _v(cal, a, 'H_independiente')
        if hi < m:
            txt += (f" Eso supone que las predecesoras de {a} acaban lo antes posible; si acabaran lo más tarde posible, "
                    f"solo le quedaría su holgura independiente ({hi}).")
    return Markdown(txt)


def respuesta_comprobaciones(c, d, prelaciones):
    """Relaciones que no cumple el calendario entregado, con los números."""
    L = []
    for a in c.index:
        ES, EF, LS, LF = (int(c.loc[a, k]) for k in ['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio'])
        Ht, Hl, da = int(c.loc[a, 'H_total']), int(c.loc[a, 'H_libre']), int(d[a])
        if EF != ES + da:
            L.append(f"- {a}: ES + d = {ES} + {da} = {ES + da} ≠ EF = {EF}")
        if LF != LS + da:
            L.append(f"- {a}: LS + d = {LS} + {da} = {LS + da} ≠ LF = {LF}")
        if Ht != LS - ES:
            L.append(f"- {a}: LS − ES = {LS} − {ES} = {LS - ES} ≠ H total = {Ht}")
        if not 0 <= Hl <= Ht:
            L.append(f"- {a}: H libre = {Hl} > H total = {Ht}")
        ps = sorted(prelaciones.immediate_predecessor[a])
        es_ok = max((int(c.loc[p, 'fin_mas_temprano']) for p in ps), default=0)
        if ES != es_ok:
            L.append(f"- {a}: máx EF de las predecesoras = {es_ok} ≠ ES = {ES}")
    malas = sorted({l.split(':')[0][2:] for l in L})
    return Markdown(f"**Respuesta:** los errores están en {lista_y(malas)}. Relaciones que fallan:\n\n" + '\n'.join(L))


def respuesta_errores(c, calendario, prelaciones, e1, e2):
    """Qué se ha calculado mal en cada actividad errónea y cuál es el valor correcto."""
    cal, suc = calendario, _sucesoras(prelaciones)
    hl_ok = _v(cal, e1, 'H_libre')
    t1 = (f"- **{e1}**: la holgura libre entregada ({int(c.loc[e1, 'H_libre'])}) es mayor que la total "
          f"({int(c.loc[e1, 'H_total'])}), lo que es imposible: la libre es la parte de la total que no afecta a las "
          f"sucesoras (H libre ≤ H total). Bien calculada, H libre = mín ES de las sucesoras − EF = "
          f"{_min_es_suc(cal, suc, e1)[1]} − {_v(cal, e1, 'fin_mas_temprano')} = **{hl_ok}**.")
    LS, LF, ES, EF = (int(c.loc[e2, k]) for k in ['inicio_mas_tardio', 'fin_mas_tardio', 'inicio_mas_temprano', 'fin_mas_temprano'])
    d, Ht = _v(cal, e2, 'duracion'), int(c.loc[e2, 'H_total'])
    t2 = (f"- **{e2}**: el inicio más tardío entregado ({LS}) no cumple LS = LF − d ({LF} − {d} = {LF - d}) ni "
          f"H total = LS − ES ({LS} − {ES} = {LS - ES}, pero la tabla dice {Ht}). En cambio LF, EF y H total sí son "
          f"coherentes entre sí (LF − EF = {LF} − {EF} = {LF - EF} = H total), así que el dato erróneo es LS: debe ser "
          f"LF − d = **{LF - d}**"
          + (f" ({e2} es crítica: su inicio más tardío coincide con el más temprano)." if Ht == 0 else '.'))
    return Markdown(t1 + '\n' + t2)


def respuesta_total_vs_libre(calendario, prelaciones, X, unidad='día'):
    """Por qué la holgura total de X es mayor que la libre, con las fechas de sus sucesoras."""
    cal, suc = calendario, _sucesoras(prelaciones)
    ef, ht, hl = _v(cal, X, 'fin_mas_temprano'), _v(cal, X, 'H_total'), _v(cal, X, 'H_libre')
    ls = {s: _v(cal, s, 'inicio_mas_tardio') for s in suc[X]}
    lf = min(ls.values())
    ls_txt = (f'mín(' + ', '.join(f'{s}: {v}' for s, v in ls.items()) + ')') if len(ls) > 1 else f'{lf} (LS de {suc[X][0]})'
    txt = (f"{X} acaba lo más pronto en {ef}. La holgura libre se mide contra el inicio **más temprano** de sus sucesoras: "
           f"H libre = {_min_es_suc(cal, suc, X)[1]} − {ef} = {hl}. La holgura total, contra su inicio **más tardío** "
           f"(el fin más tardío de {X} es el mínimo LS de sus sucesoras): H total = LF − EF = {ls_txt} − {ef} = {ht}.")
    txt += (f"\n\n**Respuesta:** {X} puede retrasarse hasta {plural(ht, unidad)} sin alargar el proyecto, pero "
            + (f"cualquier retraso suyo" if hl == 0 else f"pasados {plural(hl, unidad)}, el retraso")
            + f" retrasa el inicio más temprano de {lista_y(s for s in suc[X] if _v(cal, s, 'inicio_mas_temprano') < ef + ht)}, "
            f"que se come así parte de su propia holgura: la diferencia, {ht} − {hl} = {plural(ht - hl, unidad)}, es "
            f"holgura que {X} comparte con sus sucesoras.")
    return Markdown(txt)
'''


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
    celdas.append(code(_AUX_HOLGURAS))
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
    celdas.append(md("Duración de cada ruta del grafo, sumando las duraciones de sus actividades en orden de ejecución; "
                     "las críticas son las de mayor duración:"))
    celdas.append(code('#| tbl-cap: "Todas las rutas del grafo, con su duración como suma"\n#| label: tbl-rutas\n'
                       "tabla_rutas = explica_rutas(proyecto, datos_proyecto['duration'])\n"
                       "tabla_rutas.drop(columns='duración')"))
    celdas.append(code("display(explica_camino_critico(proyecto, datos_proyecto['duration']))\n"
                       "respuesta_rutas(tabla_rutas)"))
    celdas += apartado(ap); ap += 1
    celdas.append(md("Holguras de las actividades no críticas (las críticas tienen las tres holguras nulas):"))
    celdas.append(code('#| tbl-cap: "Holguras de las actividades no críticas"\n#| label: tbl-holgurasDistintas\n'
                       "con_holgura = calendario.index[calendario['H_total'] > 0].tolist()\n"
                       "calendario.loc[con_holgura, ['H_total', 'H_libre', 'H_independiente']].astype(int).rename(columns=_COLS_CAL).rename_axis('actividad')"))
    celdas.append(code("respuesta_retrasables(calendario, prelaciones)"))
    celdas += apartado(ap); ap += 1
    celdas.append(md(f"Un retraso de {X} igual a su holgura total equivale a alargar su duración en {hX} semanas. "
                     f"Recalculamos el calendario con esa duración y comparamos *todas* las actividades cuyas fechas u "
                     f"holguras cambian (no sólo las sucesoras inmediatas de {X}: el retraso se propaga en cadena, y "
                     f"también cambian las fechas más tardías de las que van antes)."))
    celdas.append(code('#| tbl-cap: "Actividades cuyas fechas u holguras cambian al retrasar la actividad"\n#| label: tbl-comparacionRetraso\n'
                       f"comparacion, texto_retraso, duraciones_retraso = analiza_retraso(proyecto, prelaciones, calendario, datos_proyecto['duration'], '{X}', {hX})\n"
                       "comparacion"))
    celdas.append(code("display(Markdown(texto_retraso))\n"
                       "explica_camino_critico(proyecto, duraciones_retraso)"))
    if V is not None:
        celdas += apartado(ap); ap += 1
        celdas.append(code(f"respuesta_independiente(calendario, prelaciones, '{V}')"))
    sub = f"{n} actividades, {_pl(info['n_criticas'], 'ruta crítica', 'rutas críticas')}"
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
        f"{h} semana{'s' if h != 1 else ''} y el de {Y} también. ¿Pueden concederse las dos peticiones sin alargar el proyecto? Razónelo "
        "antes de calcular.",
        f"Compruébelo: recalcule el calendario con {X} retrasada {h} semana{'s' if h != 1 else ''} y después con las dos retrasadas.",
        "¿Qué holgura expresa cuánto puede retrasarse cada una sin quitarle margen a la otra? ¿Cómo repartiría "
        + (f"las {h} semanas" if h != 1 else "la semana") + f" entre {X} {conj} {Y}?"]))
    celdas.append(code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n'
                       + literal_dataframe(df)))
    celdas.append(md("# Resolución"))
    celdas.append(code("datos_proyecto = datos_enunciado.copy()\n"
                       "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
                       "proyecto = prelaciones.create_project()\n"
                       "duracion_proyecto = proyecto.duration(datos_proyecto['duration'])"))
    celdas.append(code(_AUX_HOLGURAS))
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
    celdas.append(code(f"respuesta_compartida(calendario, prelaciones, '{X}', '{Y}', {h})"))
    celdas += apartado(3)
    celdas.append(md(f"Holgura total de {X} {conj} {Y} en la planificación original, con {X} retrasada y con las dos retrasadas:"))
    celdas.append(code('#| tbl-cap: "Holgura total de las dos actividades en cada caso"\n#| label: tbl-compartida\n'
                       f"X, Y, h = '{X}', '{Y}', {h}\n"
                       "d1 = datos_proyecto['duration'].copy(); d1[X] += h\n"
                       "d2 = d1.copy(); d2[Y] += h\n"
                       "cal1, cal2 = proyecto.calendar(d1), proyecto.calendar(d2)\n"
                       "pd.DataFrame({'original': calendario.loc[[X, Y], 'H_total'], f'{X} retrasada {h}': cal1.loc[[X, Y], 'H_total'],\n"
                       "              'las dos retrasadas': cal2.loc[[X, Y], 'H_total']}).astype(int).rename_axis('H total de')"))
    celdas.append(code("T0, T1, T2 = int(duracion_proyecto), int(proyecto.duration(d1)), int(proyecto.duration(d2))\n"
                       "Markdown(f'Con {X} retrasada, {Y} empieza en {int(cal1.loc[Y, \"inicio_mas_temprano\"])} en lugar de {int(calendario.loc[Y, \"inicio_mas_temprano\"])} '\n"
                       "         f'y su holgura total baja de {int(calendario.loc[Y, \"H_total\"])} a {int(cal1.loc[Y, \"H_total\"])}.\\n\\n'\n"
                       "         f'**Respuesta:** duración del proyecto: {plural(T0, \"semana\")} en la planificación original, {T1} con {X} retrasada '\n"
                       "         + (f'y **{T2}** con las dos retrasadas ({T0} + {T2 - T0}): el segundo retraso ya no cabe en la holgura, así que no pueden concederse las dos peticiones.'\n"
                       "            if T2 > T0 else f'y {T2} con las dos retrasadas: caben las dos peticiones.'))"))
    celdas += apartado(4)
    celdas.append(code(f"respuesta_reparto(proyecto, calendario, datos_proyecto['duration'], '{X}', '{Y}', {h})"))
    return celdas, f"{n} actividades; {X} {conj} {Y} comparten {h} semana{'s' if h != 1 else ''} de holgura"


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
    celdas.append(code(_AUX_HOLGURAS))
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
    celdas.append(md("Orden de vigilancia: primero las críticas (holgura total 0, desempatadas por duración, la más larga "
                     "primero); después, menor holgura total y, a igualdad, menor holgura libre:"))
    celdas.append(code('#| tbl-cap: "Actividades ordenadas por necesidad de vigilancia"\n#| label: tbl-orden\n'
                       "orden = calendario.sort_values(['H_total', 'H_libre', 'duracion'], ascending=[True, True, False])\n"
                       "orden['prioridad'] = range(1, len(orden) + 1)\n"
                       "orden[['prioridad', 'H_total', 'H_libre', 'H_independiente', 'duracion']].astype(int).rename(columns=_COLS_CAL).rename_axis('actividad')"))
    celdas.append(code("respuesta_vigilancia(orden)"))
    celdas += apartado(3)
    celdas.append(code("respuesta_sin_margen(calendario, prelaciones)"))
    celdas += apartado(4)
    celdas.append(code("respuesta_mas_libre(calendario, prelaciones)"))
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
    celdas.append(code(_AUX_HOLGURAS))
    celdas += apartado(1)
    celdas.append(md("Comprobamos, fila a fila, las relaciones del enunciado en el calendario entregado; la tabla muestra "
                     "solo las filas en las que alguna falla («NO»):"))
    celdas.append(code("comprobaciones = pd.DataFrame({\n"
                       "    'EF = ES + d': c['fin_mas_temprano'] == c['inicio_mas_temprano'] + d,\n"
                       "    'LF = LS + d': c['fin_mas_tardio'] == c['inicio_mas_tardio'] + d,\n"
                       "    'H_total = LS − ES': c['H_total'] == c['inicio_mas_tardio'] - c['inicio_mas_temprano'],\n"
                       "    'H_libre ≤ H_total': c['H_libre'] <= c['H_total'],\n"
                       "    'ES = máx EF pred.': [c.loc[a, 'inicio_mas_temprano'] == max((c.loc[p, 'fin_mas_temprano'] for p in prelaciones.immediate_predecessor[a]), default=0) for a in c.index]})\n"
                       "comprobaciones = comprobaciones.replace({True: 'sí', False: '**NO**'}).rename_axis('actividad')\n"
                       "comprobaciones[(comprobaciones != 'sí').any(axis=1)]"))
    celdas.append(code("respuesta_comprobaciones(c, d, prelaciones)"))
    celdas += apartado(2)
    celdas.append(code(f"calendario = proyecto.calendar(d)\nrespuesta_errores(c, calendario, prelaciones, '{e1}', '{e2}')"))
    celdas += apartado(3)
    celdas.append(code('#| tbl-cap: "Calendario correcto"\n#| label: tbl-calendarioCorrecto\n'
                       "calendario = proyecto.calendar(d)\n"
                       "calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total', 'H_libre']]"))
    celdas.append(md("Filas en las que el calendario entregado y el correcto difieren:"))
    celdas.append(code('#| tbl-cap: "Entregado frente a correcto"\n#| label: tbl-diferencias\n'
                       "difieren = (calendario[c.columns] != c).any(axis=1)\n"
                       "pd.concat({'entregado': c[difieren].rename(columns=_COLS_CAL), 'correcto': calendario.loc[difieren, c.columns].astype(int).rename(columns=_COLS_CAL)}, axis=1).rename_axis('actividad')"))
    celdas.append(code("Markdown('**Respuesta:** el calendario correcto solo difiere del entregado en ' + lista_y(\n"
                       "    f'{a} (' + ', '.join(f'{_COLS_CAL[k]} {int(c.loc[a, k])} → {int(calendario.loc[a, k])}' for k in c.columns if int(c.loc[a, k]) != int(calendario.loc[a, k])) + ')'\n"
                       "    for a in c.index[difieren]) + '; el resto de fechas y holguras eran correctas.')"))
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
        f"Dibuje el grafo {grafo_nombre()} y enumere todas las rutas desde el inicio hasta el final con su duración. "
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
    celdas.append(code(_AUX_HOLGURAS))
    celdas += apartado(1)
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT", label="fig-pert")
    celdas.append(md("Rutas desde el inicio hasta el final, con sus actividades en orden de ejecución y su duración como suma; "
                     "la duración del proyecto es la de la ruta más larga:"))
    celdas.append(code('#| tbl-cap: "Rutas del proyecto y su duración"\n#| label: tbl-rutas\n'
                       "tabla_rutas = explica_rutas(proyecto, datos_proyecto['duration'], unidad='días')\n"
                       "tabla_rutas.drop(columns='duración')"))
    celdas.append(code("display(explica_camino_critico(proyecto, datos_proyecto['duration'], unidad='días'))\n"
                       "respuesta_rutas(tabla_rutas, unidad='día')"))
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
    celdas.append(code(f"respuesta_total_vs_libre(calendario, prelaciones, '{X}', unidad='día')"))
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
