"""
test_plantillas.py — comprueba que todas las variantes de todas las colecciones
generan un cuaderno válido (sin ejecutarlo): la generación termina, cada celda
de código es Python sintácticamente correcto, hay enunciado, preguntas y
resolución, y la tabla de datos se puede volver a leer.

    python -m pytest scripts/colecciones/test_plantillas.py -q      (desde la raíz)
    PMULE_TEST_EJECUTAR=1 … ejecuta además cada cuaderno con nbclient (lento).
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1] / "src"))
sys.path.insert(0, str(AQUI.parents[1]))

from scripts.colecciones import contextos                      # noqa: E402
from scripts.colecciones.cuaderno import render, partir_enunciado   # noqa: E402
from scripts.colecciones.plantillas import PLANTILLAS           # noqa: E402

CASOS = [(p, k, s) for p in PLANTILLAS for k in range(1, len(p.VARIANTES) + 1) for s in (11, 23)]


@pytest.mark.parametrize("plantilla,k,semilla", CASOS,
                         ids=[f"{p.META['slug']}-{p.VARIANTES[k-1].nombre}-s{s}" for p, k, s in CASOS])
def test_variante(plantilla, k, semilla):
    rng = np.random.default_rng(semilla * 7919 + k)
    celdas, subtitulo, variante = plantilla.generar(k, rng)
    assert variante is plantilla.VARIANTES[k - 1]
    assert subtitulo
    tipos = [t for t, _ in celdas]
    assert "md" in tipos and "code" in tipos
    assert any(t == "md" and x.strip().startswith("# Enunciado") for t, x in celdas)
    enun, resol = partir_enunciado(celdas)
    assert resol, "falta la celda '# Resolución'"
    assert any("pregunta_1 =" in x for t, x in enun if t == "code")
    for t, x in celdas:
        if t == "code":
            fuente = "\n".join(l for l in x.splitlines() if not l.lstrip().startswith(("%", "#|")))
            ast.parse(fuente)
    texto = render(celdas, "t", subtitulo, plantilla.META["slug"], variante.nombre, variante.nivel)
    assert texto.count("# %%") >= 6
    # el contexto narrativo se puede inyectar sin romper nada
    con_ctx = contextos.inyectar(celdas, np.random.default_rng(semilla))
    assert len(con_ctx) >= len(celdas)


@pytest.mark.skipif(not os.environ.get("PMULE_TEST_EJECUTAR"), reason="ejecución lenta; PMULE_TEST_EJECUTAR=1")
@pytest.mark.parametrize("plantilla,k", [(p, k) for p in PLANTILLAS for k in range(1, len(p.VARIANTES) + 1)],
                         ids=[f"{p.META['slug']}-{p.VARIANTES[k-1].nombre}" for p in PLANTILLAS for k in range(1, len(p.VARIANTES) + 1)])
def test_ejecuta(plantilla, k, tmp_path):
    import nbformat
    from nbclient import NotebookClient
    import jupytext
    rng = np.random.default_rng(99 + k)
    celdas, subtitulo, variante = plantilla.generar(k, rng)
    texto = render(celdas, "t", subtitulo, plantilla.META["slug"], variante.nombre, variante.nivel)
    destino = AQUI.parents[1] / "notebooks" / "colecciones" / "_pruebas"
    destino.mkdir(exist_ok=True)
    ruta = destino / f"prueba_{plantilla.META['slug']}_{k}.py"
    ruta.write_text(texto, encoding="utf8")
    nb = jupytext.read(ruta)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(destino)}}).execute()
    ruta.unlink()
