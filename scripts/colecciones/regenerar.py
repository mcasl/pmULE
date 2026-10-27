"""
regenerar.py — vuelve a generar cuadernos ya existentes de las colecciones
conservando su variante, su semilla y su grafo (PERT o Roy), de modo que el
enunciado y los datos no cambian y sólo se propagan a la resolución los
cambios hechos en las plantillas.

A diferencia de ``generador --forzar`` (que asigna la variante por la posición
k en la lista *actual* de variantes y renumeraría los cuadernos cuando se han
añadido variantes), aquí la variante se lee de la cabecera del cuaderno y el
grafo se deduce del cuaderno existente.

Uso (desde la raíz del repositorio):

    python -m scripts.colecciones.regenerar                    # todos los del catálogo
    python -m scripts.colecciones.regenerar --solo 10 12       # sólo esas colecciones
    python -m scripts.colecciones.regenerar Reduccion_15 Recursos_03
    python -m scripts.colecciones.regenerar --comprobar        # no escribe: informa de si el enunciado se reproduce
    python -m scripts.colecciones.regenerar --sin-index        # no tocar README, índice ni catálogo

Al terminar informa de los cuadernos cuyo enunciado no coincide con el
anterior (por ejemplo, por una plantilla no determinista) para revisarlos.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import numpy as np

from . import contextos                                                   # noqa: E402
from .cuaderno import render, ESTADO                                      # noqa: E402
from .generador import (REPO, COLECCIONES, PROB_CONTEXTO, semilla_de, _leer_cabecera,   # noqa: E402
                        _emparejar, _entrada_desde_fichero, escribir_readme,
                        actualizar_index, escribir_catalogo)
from .plantillas import PLANTILLAS                                        # noqa: E402


def _enunciado(texto: str) -> str:
    i = texto.find("# # Resolución")
    texto = texto[:i] if i >= 0 else texto
    texto = re.sub(r"^# grafo: .*\n", "", texto, flags=re.M)   # la cabecera grafo no es enunciado
    # np.str_('x') (numpy 2 con pandas 2) y 'x' (pandas 3) son el mismo dato
    return re.sub(r"np\.str_\('([^']*)'\)", r"'\1'", texto)


def _cuenta(texto: str, metodo: str) -> int:
    """Apariciones del grafo ``metodo`` en el cuaderno: llamadas directas y visores."""
    extra = texto.count(f"display_{metodo}_construccion(")
    return (texto.count(f".{metodo}(") + texto.count(f"display_{metodo}_pasos(") + extra
            + texto.count(f"display_ancestros('{metodo}'") + texto.count(f'display_ancestros("{metodo}"')
            + texto.count(f"display_rutas('{metodo}'") + texto.count(f'display_rutas("{metodo}"'))


def _generar(plantilla, variante, k, grafo, semilla):
    meta = plantilla.META
    ESTADO["grafo"] = grafo
    rng = np.random.default_rng(semilla_de(semilla, PLANTILLAS.index(plantilla) + 1, k))
    celdas, subtitulo = variante.fn(rng)
    if meta.get("contexto", True) and rng.random() < PROB_CONTEXTO:
        celdas = contextos.inyectar(celdas, rng)
    return render(celdas, titulo=f"{meta['prefijo']} {k:02d}", subtitulo=subtitulo,
                  coleccion=meta["slug"], variante=variante.nombre, nivel=variante.nivel)


def regenerar_cuaderno(plantilla, ruta_py: Path, semilla: int, escribir: bool) -> dict:
    viejo = ruta_py.read_text(encoding="utf8")
    cab = _leer_cabecera(ruta_py)
    nombre_variante = cab.get("variante", "")
    k = int(ruta_py.stem.rsplit("_", 1)[1])
    candidatas = [v for v in plantilla.VARIANTES if v.nombre == nombre_variante]
    if not candidatas:
        return dict(ruta=ruta_py, estado="variante desconocida", variante=nombre_variante)
    variante = candidatas[0]
    g_cab = cab.get("grafo")
    if g_cab in ("pert", "roy", "ambos"):
        grafos = [g_cab]                        # la cabecera del cuaderno manda
    elif variante.grafo != "alterno":
        grafos = [variante.grafo]
    elif _cuenta(viejo, "roy") == 0 and _cuenta(viejo, "pert") == 0:
        # el cuaderno antiguo no dibujaba grafo: se aplica la regla de las vueltas del generador
        grafos = ["pert" if ((k - 1) // len(plantilla.VARIANTES)) % 2 == 0 else "roy"]
    else:
        grafos = ["pert", "roy"]
    mejor, mejor_puntos = None, None
    enunciado_viejo = _enunciado(viejo)
    for grafo in grafos:
        try:
            texto = _generar(plantilla, variante, k, grafo, semilla)
        except Exception as exc:                 # noqa: BLE001
            return dict(ruta=ruta_py, estado=f"error: {exc!r}", variante=nombre_variante)
        if len(grafos) == 1:
            puntos = (0, 0.0)
        else:
            # el grafo correcto es el que reproduce las llamadas .pert( / .roy( (o sus visores
            # de pasos) del cuaderno existente; a igualdad, el texto más parecido
            desajuste = abs(_cuenta(viejo, "roy") - _cuenta(texto, "roy")) + abs(_cuenta(viejo, "pert") - _cuenta(texto, "pert"))
            puntos = (-desajuste, difflib.SequenceMatcher(None, viejo, texto, autojunk=False).quick_ratio())
        if mejor_puntos is None or puntos > mejor_puntos:
            mejor, mejor_puntos = (grafo, texto), puntos
    grafo, texto = mejor
    coincide = re.sub(r"[ \t]+", " ", _enunciado(texto)) == re.sub(r"[ \t]+", " ", enunciado_viejo)
    if escribir:
        ruta_py.write_text(texto, encoding="utf8")
        _emparejar(ruta_py)
    return dict(ruta=ruta_py, estado="ok" if coincide else "enunciado distinto", variante=nombre_variante,
                grafo=grafo, cambia=(texto != viejo))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cuadernos", nargs="*", help="nombres de cuadernos (Reduccion_15 …); por defecto, todos")
    p.add_argument("--solo", nargs="*", default=None, help="prefijos numéricos de las colecciones (05 09 …)")
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--comprobar", action="store_true", help="no escribir nada; informar de si el enunciado se reproduce")
    p.add_argument("--sin-index", action="store_true", help="no tocar docs/index.md, el README ni el catálogo")
    args = p.parse_args(argv)

    pedidos = set(args.cuadernos)
    resultados = []
    for plantilla in PLANTILLAS:
        meta = plantilla.META
        if args.solo and not any(meta["slug"].startswith(s) for s in args.solo):
            continue
        for ruta_py in sorted((COLECCIONES / meta["slug"]).glob(f"{meta['prefijo']}_*.py")):
            if pedidos and ruta_py.stem not in pedidos:
                continue
            r = regenerar_cuaderno(plantilla, ruta_py, args.semilla, escribir=not args.comprobar)
            resultados.append(r)
            marca = {"ok": "=" if not r.get("cambia") else "+", "enunciado distinto": "!"}.get(r["estado"], "E")
            print(f"  {marca} {ruta_py.relative_to(REPO)}  [{r['variante']}, {r.get('grafo', '-')}]  {r['estado']}")
    if not args.comprobar and not args.sin_index and resultados:
        resumen = []
        grafos = {r["ruta"]: r.get("grafo") for r in resultados}
        for indice, plantilla in enumerate(PLANTILLAS, 1):
            meta = plantilla.META
            existentes = sorted((COLECCIONES / meta["slug"]).glob(f"{meta['prefijo']}_*.py"))
            if existentes:
                entradas = []
                for r in existentes:
                    e = _entrada_desde_fichero(meta, r.stem, r)
                    k = int(r.stem.rsplit("_", 1)[1])
                    e["semilla"] = semilla_de(args.semilla, indice, k)
                    if grafos.get(r):
                        e["grafo"] = grafos[r]
                    entradas.append(e)
                resumen.append((plantilla, entradas))
        escribir_readme(resumen)
        actualizar_index(resumen)
        escribir_catalogo(resumen)
    malos = [r for r in resultados if r["estado"] != "ok"]
    print(f"{len(resultados)} cuaderno(s); {len(malos)} con enunciado distinto o error"
          + (":" if malos else "."))
    for r in malos:
        print(f"   {r['ruta'].relative_to(REPO)}: {r['estado']}")
    return 1 if any(r["estado"].startswith("error") or r["estado"] == "variante desconocida" for r in resultados) else 0


if __name__ == "__main__":
    sys.exit(main())
