#!/usr/bin/env bash
# Renderiza a HTML los cuadernos de notebooks/colecciones/ (todos, o los que se
# indiquen) y deja el resultado en docs/colecciones/<coleccion>/<nombre>.html.
#
# Uso:  ./render_colecciones.sh                                   # todos
#       ./render_colecciones.sh notebooks/colecciones/05_holguras/Holguras_01.ipynb ...
#
# Mismo flujo que render_ejercicio.sh: quarto ejecuta el cuaderno con el Python
# del proyecto (.venv), el HTML va a docs/ y se sincroniza el par jupytext.
# (Sin bashismos: funciona también con `sh render_colecciones.sh`.)
set -eu
cd "$(dirname "$0")"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi
QUARTO_PYTHON="${QUARTO_PYTHON:-$(command -v python3)}"
export QUARTO_PYTHON

if [ $# -gt 0 ]; then
    cuadernos="$*"
else
    cuadernos="$(find notebooks/colecciones -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' | sort)"
fi

n=0
for nb in $cuadernos; do
    [ -f "$nb" ] || { echo "No existe: $nb" >&2; exit 1; }
    coleccion="$(basename "$(dirname "$nb")")"
    nombre="$(basename "$nb" .ipynb)"
    destino="docs/colecciones/${coleccion}"
    mkdir -p "$destino"
    echo "→ Rendering ${nb}..."
    quarto render "$nb"
    # Según cómo detecte quarto el proyecto, el HTML aparece junto al cuaderno
    # o directamente en docs/colecciones/<coleccion>/ (output-dir del proyecto).
    if [ -f "$(dirname "$nb")/${nombre}.html" ]; then
        mv "$(dirname "$nb")/${nombre}.html" "${destino}/${nombre}.html"
    elif [ ! -f "${destino}/${nombre}.html" ]; then
        echo "No encuentro el HTML generado para ${nb}" >&2; exit 1
    fi
    if command -v jupytext >/dev/null 2>&1; then
        jupytext --sync --quiet "$nb"
    fi
    git add "$nb" "${nb%.ipynb}.py" "${destino}/${nombre}.html" 2>/dev/null || true
    n=$((n + 1))
done
echo "Hecho: ${n} cuaderno(s) renderizado(s) en docs/colecciones/"
