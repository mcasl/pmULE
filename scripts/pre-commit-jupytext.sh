#!/usr/bin/env bash
# Hook pre-commit: mantiene sincronizados los pares jupytext (ipynb <-> py:percent)
# de notebooks/ que entren en el commit, y añade al stage el gemelo actualizado.
# Instalación:  make hooks   (crea el enlace .git/hooks/pre-commit -> este fichero)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

JT=".venv/bin/jupytext"
[ -x "$JT" ] || JT="$(command -v jupytext || true)"
if [ -z "$JT" ]; then
    echo "pre-commit: jupytext no encontrado (ejecuta 'make sync'); se omite la sincronización" >&2
    exit 0
fi

staged=$(git diff --cached --name-only --diff-filter=ACMR | grep -E '^notebooks/.*\.(ipynb|py)$' || true)
[ -z "$staged" ] && exit 0

# Una sola llamada a jupytext para todos los pares (cada arranque de Python
# cuesta ~2 s: con un proceso por fichero, un commit grande tardaba minutos)
pares=()
for f in $staged; do
    base="${f%.*}"
    # Solo cuadernos emparejados: deben existir ambos ficheros
    [ -f "$base.ipynb" ] && [ -f "$base.py" ] || continue
    case " ${pares[*]-} " in *" $base.ipynb "*) continue ;; esac
    pares+=("$base.ipynb")
done
[ ${#pares[@]} -eq 0 ] && exit 0
"$JT" --sync --quiet "${pares[@]}" >/dev/null 2>&1
for nb in "${pares[@]}"; do
    git add "$nb" "${nb%.ipynb}.py"
done
echo "pre-commit: ${#pares[@]} par(es) jupytext sincronizado(s)"
exit 0
