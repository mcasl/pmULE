"""
pasos.py — añade a un cuaderno jupytext (py:percent) el cálculo paso a paso de
los tiempos, usando las funciones de pmULE:

Primero se muestra el resultado y a continuación, en un visor con botones, el
cálculo paso a paso (como con la matriz de Zaderenko):

- tras el primer grafo PERT con tiempos (``obj.pert(..., durations=X)``):
  ``obj.display_pert_pasos(X)`` (tempranos hacia delante, tardíos hacia atrás);
- tras el primer grafo Roy con tiempos (``obj.roy(..., durations=X)``):
  ``obj.display_roy_pasos(X)`` (mínimos y máximos);
- tras la primera matriz de Zaderenko (``obj.zaderenko(X)``):
  ``obj.display_zaderenko_pasos(X)``.

Es idempotente: si el cuaderno ya llama a alguna de esas funciones, no la
repite; y si la llamada del PERT o del Roy está *antes* del grafo (versión
anterior, con la lista de cálculos delante), la mueve detrás del grafo. Lo usa ``cuaderno.render`` (colecciones y exámenes) y sirve también
para los ``Ejercicio_NN``:

    python -m scripts.colecciones.pasos notebooks/Ejercicio_*.py notebooks/colecciones/*/*.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CABECERA_CELDA = re.compile(r"^# %%.*$", re.M)


def _argumento(llamada: str, inicio: int) -> str:
    """Expresión que empieza en ``inicio`` hasta la coma o el paréntesis de cierre
    del nivel superior."""
    nivel, i = 0, inicio
    while i < len(llamada):
        c = llamada[i]
        if c in "([{":
            nivel += 1
        elif c in ")]}":
            if nivel == 0:
                break
            nivel -= 1
        elif c == "," and nivel == 0:
            break
        elif c in "'\"":
            fin = llamada.index(c, i + 1)
            i = fin
        i += 1
    return llamada[inicio:i].strip()


def _durations(codigo: str, metodo: str):
    """(objeto, expresión de las duraciones) de la primera llamada ``obj.metodo(...)``
    con duraciones, o None."""
    for m in re.finditer(rf"(\w+)\.{metodo}\(", codigo):
        args = _argumento_completo(codigo[m.end():])
        k = re.search(r"\bdurations\s*=\s*", args)
        if k:
            expr = _argumento(args, k.end())
        elif metodo == "zaderenko":
            expr = _argumento(args, 0)          # primer argumento posicional
        else:
            continue
        if expr and expr not in ("False", "None"):
            return m.group(1), expr
    return None


def _argumento_completo(resto: str) -> str:
    """Texto de los argumentos de una llamada hasta su paréntesis de cierre."""
    nivel = 0
    for i, c in enumerate(resto):
        if c in "([{":
            nivel += 1
        elif c in ")]}":
            if nivel == 0:
                return resto[:i]
            nivel -= 1
    return resto


INTRO = {
    "pert": "Cálculo de los tiempos tempranos y tardíos de los sucesos paso a paso "
            "(avance con los botones):",
    "roy": "Cálculo de los tiempos mínimos y máximos de los nodos del grafo Roy paso a paso "
           "(avance con los botones):",
    "zaderenko": "Algoritmo de Zaderenko paso a paso (avance con los botones):",
}
FUNCION = {"pert": "display_pert_pasos", "roy": "display_roy_pasos",
           "zaderenko": "display_zaderenko_pasos"}
LLAMADA = re.compile(r"^# %%[^\n]*\n(\w+)\.display_(pert|roy)_pasos\(.*\)\s*$", re.S)


def _es_codigo(celda: str) -> bool:
    return celda.startswith("# %%") and "[markdown]" not in celda.splitlines()[0]


def _celdas(texto: str):
    """(texto previo a la primera celda, lista de celdas tal como están escritas)."""
    cabeceras = list(CABECERA_CELDA.finditer(texto))
    if not cabeceras:
        return None, []
    celdas = []
    for k, m in enumerate(cabeceras):
        fin = cabeceras[k + 1].start() if k + 1 < len(cabeceras) else len(texto)
        celdas.append(texto[m.start():fin])
    return texto[:cabeceras[0].start()], celdas


def _nueva(celda: str) -> str:
    """Celda escrita por este script: acaba en una línea en blanco."""
    return celda.rstrip("\n") + "\n\n"


def _celda_intro(metodo: str) -> str:
    return _nueva(f"# %% [markdown]\n# {INTRO[metodo]}")


def anade_pasos(texto: str) -> str:
    cabecera, original = _celdas(texto)
    if not original:
        return texto
    celdas = list(original)
    # 1. Versión anterior: la lista de cálculos estaba delante del grafo; se
    #    pasa detrás del grafo (y se le añade la frase de presentación).
    for metodo in ("pert", "roy"):
        for k, celda in enumerate(celdas):
            m = LLAMADA.match(celda) if _es_codigo(celda) else None
            if not (m and m.group(2) == metodo):
                continue
            destino = next((j for j in range(k + 1, len(celdas))
                            if _es_codigo(celdas[j]) and _durations(celdas[j], metodo)), None)
            if destino is None:
                break
            celdas[destino] = _nueva(celdas[destino])
            celdas[destino + 1:destino + 1] = [_celda_intro(metodo), _nueva(celda)]
            del celdas[k]
            break
    # 2. Lo que falte se añade detrás del primer grafo (o matriz) con tiempos.
    hecho = {m: any(_es_codigo(c) and f"{FUNCION[m]}(" in c for c in celdas)
             for m in FUNCION}
    salida = []
    for celda in celdas:
        anadidas = []
        if _es_codigo(celda):
            for metodo in ("pert", "roy", "zaderenko"):
                if hecho[metodo]:
                    continue
                r = _durations(celda, metodo)
                if r:
                    obj, expr = r
                    anadidas += [_celda_intro(metodo),
                                 _nueva(f"# %%\n{obj}.{FUNCION[metodo]}({expr})")]
                    hecho[metodo] = True
        salida.append(_nueva(celda) if anadidas else celda)
        salida += anadidas
    if salida == original:
        return texto
    cola = texto[len(texto.rstrip("\n")):]
    return (cabecera + "".join(salida)).rstrip("\n") + cola


if __name__ == "__main__":
    cambiados = 0
    for p in sys.argv[1:]:
        ruta = Path(p)
        viejo = ruta.read_text(encoding="utf8")
        nuevo = anade_pasos(viejo)
        if nuevo != viejo:
            ruta.write_text(nuevo, encoding="utf8")
            cambiados += 1
            print("añadidos pasos:", ruta)
    print(f"{cambiados} cuaderno(s) modificados")
