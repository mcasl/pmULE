#!/usr/bin/env bash
# Publica docs/ (HTML renderizados) en la rama gh-pages de origin con UN SOLO commit,
# de modo que GitHub conserva solo la última versión de los HTML (los resultados),
# mientras que master guarda el historial del código. docs/ no está en master.
#
# Uso (desde cualquier carpeta del repositorio, en el Mac):
#   scripts/publica_docs.sh            # sube docs/ a gh-pages (push forzado)
#   scripts/publica_docs.sh --dry-run  # prepara el commit local y no sube nada
#
# Mecánica: un repositorio auxiliar fuera del repo (~/.cache/pmule-pages.git) que usa
# docs/ como árbol de trabajo (no se duplica nada en disco). Se recrea en cada
# ejecución, así que gh-pages siempre es una historia de un solo commit.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
REMOTO="$(git remote get-url origin)"
NOMBRE="$(git config user.name)"; CORREO="$(git config user.email)"
GD="${PMULE_PAGES_GIT:-$HOME/.cache/pmule-pages.git}"
[ -d docs ] || { echo "No existe docs/"; exit 1; }

rm -rf "$GD"
g() { git --git-dir="$GD" --work-tree=docs "$@"; }
g init -q
g symbolic-ref HEAD refs/heads/gh-pages
mkdir -p "$GD/info"
printf '%s\n' '.DS_Store' '.ipynb_checkpoints/' '/index.html' > "$GD/info/exclude"
rm -f docs/.nojekyll                         # Pages (Jekyll) convierte docs/index.md en la página principal
g config user.name  "$NOMBRE"
g config user.email "$CORREO"
g add -A
g commit -q -m "Documentación renderizada ($(date '+%Y-%m-%d %H:%M'))"
echo "→ $(g ls-files | wc -l | tr -d ' ') ficheros en el commit $(g rev-parse --short HEAD)"
if [ "$DRY" = 1 ]; then echo "(dry-run: no se sube nada; repositorio auxiliar en $GD)"; exit 0; fi
g push --force "$REMOTO" gh-pages:gh-pages
echo "Publicado en la rama gh-pages de $REMOTO"
