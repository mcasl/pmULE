#!/usr/bin/env sh
# Renderiza las lecciones (notebooks/colecciones/lecciones/*.qmd, una por colección, con la
# teoría: conceptos, algoritmos, ejemplo guía y uso de la librería) a
# docs/colecciones/lecciones/<slug>.html. Las que llevan celdas {python} se ejecutan (figuras de pmULE).
#
# Uso:  ./render_lecciones.sh                                   # todas
#       ./render_lecciones.sh notebooks/colecciones/lecciones/05_holguras.qmd
set -eu
cd "$(dirname "$0")"

# Algunas lecciones ejecutan código (figuras de la librería): mismo Python que render_colecciones.sh
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi
QUARTO_PYTHON="${QUARTO_PYTHON:-$(command -v python3)}"
export QUARTO_PYTHON

if [ $# -gt 0 ]; then
    lecciones="$*"
else
    lecciones="$(find notebooks/colecciones/lecciones -name '*.qmd' | sort)"
fi
destino="docs/colecciones/lecciones"
mkdir -p "$destino"
n=0
for q in $lecciones; do
    [ -f "$q" ] || { echo "No existe: $q" >&2; exit 1; }
    nombre="$(basename "$q" .qmd)"
    echo "→ Rendering ${q}..."
    quarto render "$q" --to html
    if [ -f "$(dirname "$q")/${nombre}.html" ]; then
        mv "$(dirname "$q")/${nombre}.html" "${destino}/${nombre}.html"
    elif [ ! -f "${destino}/${nombre}.html" ]; then
        echo "No encuentro el HTML generado para ${q}" >&2; exit 1
    fi
    git add "$q" "${destino}/${nombre}.html" 2>/dev/null || true
    n=$((n + 1))
done
echo "Hecho: ${n} lección(es) renderizada(s) en ${destino}/"
