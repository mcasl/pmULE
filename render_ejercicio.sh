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
HTML="notebooks/Ejercicio_${NUM}.html"
TARGET="docs/Ejercicio_${NUM}.html"

if [[ ! -f "$NOTEBOOK" ]]; then
    echo "Error: Notebook not found → $NOTEBOOK"
    exit 1
fi

echo "→ Rendering ${NOTEBOOK}..."
quarto render "$NOTEBOOK"

echo "→ Moving HTML..."
mv "$HTML" "$TARGET"

echo "→ Git add..."
git add "$NOTEBOOK" "$TARGET"

echo ""
echo "Done! 🎉"
echo "Files staged:"
echo "  • $NOTEBOOK"
echo "  • $TARGET"
