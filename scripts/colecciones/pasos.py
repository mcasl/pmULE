"""
pasos.py — añade a un cuaderno jupytext (py:percent) el cálculo paso a paso de
los tiempos, usando las funciones de pmULE:

- antes del primer grafo PERT con tiempos (``obj.pert(..., durations=X)``):
  ``obj.display_pert_pasos(X)`` (tempranos hacia delante, tardíos hacia atrás);
- antes del primer grafo Roy con tiempos (``obj.roy(..., durations=X)``):
  ``obj.display_roy_pasos(X)`` (mínimos y máximos);
- tras la primera matriz de Zaderenko (``obj.zaderenko(X)``):
  ``obj.display_zaderenko_pasos(X)`` (visor con colores y botones).

Es idempotente: si el cuaderno ya llama a alguna de esas funciones, no repite
esa parte. Lo usa ``cuaderno.render`` (colecciones y exámenes) y sirve también
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


def anade_pasos(texto: str) -> str:
    cabeceras = list(CABECERA_CELDA.finditer(texto))
    if not cabeceras:
        return texto
    trozos = [texto[:cabeceras[0].start()]]
    celdas = []
    for k, m in enumerate(cabeceras):
        fin = cabeceras[k + 1].start() if k + 1 < len(cabeceras) else len(texto)
        celdas.append(texto[m.start():fin])
    hecho = {"pert": "display_pert_pasos(" in texto,
             "roy": "display_roy_pasos(" in texto,
             "zaderenko": "display_zaderenko_pasos(" in texto}
    salida = []
    for celda in celdas:
        es_codigo = celda.startswith("# %%") and "[markdown]" not in celda.splitlines()[0]
        despues = []
        if es_codigo:
            for metodo, funcion in (("pert", "display_pert_pasos"), ("roy", "display_roy_pasos")):
                if hecho[metodo]:
                    continue
                r = _durations(celda, metodo)
                if r:
                    obj, expr = r
                    salida.append(f"# %%\n{obj}.{funcion}({expr})\n\n")
                    hecho[metodo] = True
            if not hecho["zaderenko"]:
                r = _durations(celda, "zaderenko")
                if r:
                    obj, expr = r
                    despues.append("# %% [markdown]\n# Algoritmo de Zaderenko paso a paso "
                                   "(avance con los botones):\n\n")
                    despues.append(f"# %%\n{obj}.display_zaderenko_pasos({expr})\n\n")
                    hecho["zaderenko"] = True
        if despues and not celda.endswith("\n\n"):
            celda = celda.rstrip("\n") + "\n\n"
        salida.append(celda)
        salida.extend(despues)
    resultado = "".join(trozos + salida)
    if not texto.endswith("\n\n") and resultado.endswith("\n\n"):
        resultado = resultado[:-1]
    return resultado


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
