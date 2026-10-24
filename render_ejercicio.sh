#!/usr/bin/env bash
# File: render-ejercicio.sh
# Usage: ./render-ejercicio.sh 10
#        ./render-ejercicio.sh 7

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 NUMBER"
    echo "Example: $0 10"
    exit 1
fi

NUM="$1"
NOTEBOOK="notebooks/Ejercicio_${NUM}.ipynb"
JUPYTEXT_PY="notebooks/Ejercicio_${NUM}.py"
HTML="notebooks/Ejercicio_${NUM}.html"
TARGET="docs/Ejercicio_${NUM}.html"

# Quarto ejecuta el cuaderno (execute.enabled) y necesita el Python del
# proyecto, no el del sistema: activar .venv si no hay otro entorno activo.
if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
export QUARTO_PYTHON="${QUARTO_PYTHON:-$(command -v python3)}"

if [[ ! -f "$NOTEBOOK" ]]; then
    echo "Error: Notebook not found → $NOTEBOOK"
    exit 1
fi

echo "→ Rendering ${NOTEBOOK}..."
quarto render "$NOTEBOOK" --metadata-file scripts/html_extra.yml

echo "→ Moving HTML..."
mv "$HTML" "$TARGET"
"${PYTHON:-python3}" scripts/normaliza_ids.py "$TARGET"   # ids de tablas deterministas

echo "→ Sincronizando par jupytext..."
if command -v jupytext >/dev/null 2>&1; then
    jupytext --sync --quiet "$NOTEBOOK"
else
    echo "  (jupytext no disponible; se omite)"
fi

echo "→ Git add..."
git add "$NOTEBOOK" "$TARGET"
[[ -f "$JUPYTEXT_PY" ]] && git add "$JUPYTEXT_PY"

echo ""
echo "Done! 🎉"
echo "Files staged:"
echo "  • $NOTEBOOK"
[[ -f "$JUPYTEXT_PY" ]] && echo "  • $JUPYTEXT_PY"
echo "  • $TARGET"
