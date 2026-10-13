"""
cuaderno.py — utilidades para emitir cuadernos jupytext (py:percent) con el
mismo guion que los Ejercicio_NN de pmULE: cabecera quarto, imports, enunciado
con `pregunta_N`, tabla de datos literal y resolución por apartados.
"""
from __future__ import annotations

import math
from textwrap import dedent

import numpy as np
import pandas as pd

CABECERA = """\
# ---
# title: {titulo}
# subtitle: "{subtitulo}"
# coleccion: {coleccion}
# variante: {variante}
# nivel: {nivel}
# lang: es
# documentclass: scrreprt
# papersize: A4
# classoption: []
# execute:
#     echo: False
#     enabled: true
#     freeze: false
# toc: true
# format:
#     html:
#         theme: cosmo
#         embed-resources: true
#         fig-responsive: true
# warning: false
# message: false
# jupyter: python3
# ---
"""

PREAMBULO = """\
import os, sys
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, Markdown, Math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Versión local de la librería: se busca src/pmule.py subiendo desde el directorio actual
_aqui = os.path.abspath('.')
while _aqui != os.path.dirname(_aqui) and not os.path.isfile(os.path.join(_aqui, 'src', 'pmule.py')):
    _aqui = os.path.dirname(_aqui)
sys.path.append(os.path.join(_aqui, 'src'))
from pmule import (PredecessorTable, ProjectGraph, EarnedValue,
                   pinta_tikz, plot_cargas, plot_curvas_valor_ganado,
                   tanto_por_uno, ordena_rutas)
"""

TEMP = """\
# %%capture
# %mkdir -p ~/temp
# %cd ~/temp
"""

BMATRIX = r'''
def bmatrix(a, dec=0):
    """Matriz LaTeX (bmatrix) a partir de un DataFrame o array."""
    a = np.asarray(getattr(a, 'values', a), dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    filas = [' & '.join(f'{x:.{dec}f}' for x in fila) + r'\\' for fila in a]
    return r'\begin{bmatrix}' + '\n' + '\n'.join(filas) + '\n' + r'\end{bmatrix}'
'''.strip("\n")


def md(texto: str) -> tuple:
    return ("md", dedent(texto).strip("\n"))


def code(texto: str) -> tuple:
    return ("code", dedent(texto).strip("\n"))


def preguntas(lista: list[str]) -> tuple:
    """Celda que define pregunta_1..N y las muestra."""
    lineas = [f'pregunta_{i} = "{i}. {p}  "' for i, p in enumerate(lista, 1)]
    lineas.append("")
    lineas.append("for item in sorted(dir()):")
    lineas.append("    if item.startswith('pregunta_'):")
    lineas.append("        display(Markdown(eval(item)))")
    return ("code", "\n".join(lineas))


def apartado(n: int) -> list[tuple]:
    return [md(f"## Apartado {n}"), code(f"display(Markdown(pregunta_{n}))")]


def _lit(v) -> str:
    if isinstance(v, str):
        return repr(v)
    if v is None:
        return "None"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isinf(v):
            return "np.inf"
        if float(v).is_integer():
            return str(int(v))
        return repr(round(float(v), 4))
    return repr(v)


def literal_dataframe(df: pd.DataFrame, nombre: str = "datos_enunciado",
                      indice: str = "activity") -> str:
    """Tabla de datos como literal Python alineado, al estilo de los Ejercicio_NN."""
    df = df.copy()
    if indice not in df.columns:
        df.index.name = indice
        df = df.reset_index()
    cols = list(df.columns)
    filas = []
    for _, fila in df.iterrows():
        filas.append([_lit(fila[c]) for c in cols])
    anchos = [max(len(f[j]) for f in filas + [[c for c in cols]]) for j in range(len(cols))]
    cab = "#  " + ", ".join(c.ljust(anchos[j]) for j, c in enumerate(cols))
    lineas = [f"{nombre} = pd.DataFrame([", cab]
    for f in filas:
        lineas.append("    (" + ", ".join(v.ljust(anchos[j]) for j, v in enumerate(f)) + "),")
    lineas.append("], columns=[" + ", ".join(repr(c) for c in cols) + f"]).set_index({indice!r})")
    lineas.append(nombre)
    return "\n".join(lineas)


def literal_dict(d: dict, nombre: str) -> str:
    lineas = [f"{nombre} = {{"]
    for k, v in d.items():
        lineas.append(f"    {k!r}: {v!r},")
    lineas.append("}")
    return "\n".join(lineas)


def _celda_md(texto: str) -> str:
    cuerpo = "\n".join(("# " + l) if l.strip() else "#" for l in texto.split("\n"))
    return "# %% [markdown]\n" + cuerpo


def _celda_code(texto: str) -> str:
    return "# %%\n" + texto


class Variante:
    """Una variante de una plantilla: nombre corto (slug), nivel 1-3, descripción y
    función generadora ``fn(rng) -> (celdas, subtitulo)``."""

    def __init__(self, nombre: str, nivel: int, descripcion: str, fn):
        self.nombre, self.nivel, self.descripcion, self.fn = nombre, nivel, descripcion, fn


def elegir(variantes: list, k: int):
    return variantes[(k - 1) % len(variantes)]


def partir_enunciado(celdas: list[tuple]) -> tuple[list, list]:
    """Separa (enunciado, resolución) en la celda markdown '# Resolución'."""
    for i, (tipo, texto) in enumerate(celdas):
        if tipo == "md" and texto.strip().startswith("# Resolución"):
            return celdas[:i], celdas[i:]
    return celdas, []


def render(celdas: list[tuple], titulo: str, subtitulo: str, coleccion: str = "",
           variante: str = "", nivel: int = 0) -> str:
    """Devuelve el texto completo del cuaderno en formato py:percent."""
    partes = [CABECERA.format(titulo=titulo, subtitulo=subtitulo, coleccion=coleccion,
                              variante=variante, nivel=nivel).rstrip("\n")]
    partes.append(_celda_code(PREAMBULO))
    partes.append(_celda_code(TEMP))
    for tipo, texto in celdas:
        partes.append(_celda_md(texto) if tipo == "md" else _celda_code(texto))
    return "\n\n".join(partes) + "\n"


def despachar(variantes: list, k: int, rng):
    """Ejecuta la variante k (cíclica) y devuelve (celdas, subtitulo, variante)."""
    v = elegir(variantes, k)
    celdas, sub = v.fn(rng)
    return celdas, sub, v
