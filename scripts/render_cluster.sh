#!/usr/bin/env bash
# Renderiza cuadernos de pmule en el clúster GVIS (legio2) con SLURM y un
# contenedor Apptainer, y trae los HTML a docs/ (mismo destino que
# render_ejercicio.sh y render_colecciones.sh).
#
# Uso (desde la raíz del repositorio, en el Mac):
#   scripts/render_cluster.sh notebooks/Ejercicio_30.ipynb notebooks/colecciones/03_roy/*.ipynb
#   scripts/render_cluster.sh --grep '\.roy('     # cuadernos cuyo .py contiene el patrón
#   scripts/render_cluster.sh --all               # ejercicios, colecciones y exámenes
#   scripts/render_cluster.sh --build             # (re)construye el contenedor
# Variable: PMULE_MEM=12G sube la memoria de cada tarea (por defecto 4G, para cuadernos pesados)
# Opciones: -m N  limita las tareas simultáneas (por defecto sin límite: el array
#           corre al máximo que permita el gestor SLURM)
#
# Pasos: comprobar conexión (si falla: VPN) → jupytext --sync local → rsync de
# ida → sbatch --wait (array, una tarea por cuaderno) → rsync de vuelta →
# git add de .ipynb y .py. Los HTML de docs/ no se versionan en master: se publican
# con scripts/publica_docs.sh (rama gh-pages, un solo commit).
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${PMULE_CLUSTER:-legio2}"
REMOTO="pmule_render"           # ~/pmule_render en el clúster
MAX=
BUILD=0
cuadernos=()

todos() { find notebooks -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' \
            \( -path 'notebooks/Ejercicio_*' -o -path 'notebooks/colecciones/*' -o -path 'notebooks/examenes/*' \) \
            -not -path 'notebooks/colecciones/lecciones/*' | sort; }

while [ $# -gt 0 ]; do
    case "$1" in
        --build) BUILD=1 ;;
        -m) MAX="$2"; shift ;;
        --all) while IFS= read -r nb; do cuadernos+=("$nb"); done < <(todos) ;;
        --grep)
            while IFS= read -r nb; do
                py="${nb%.ipynb}.py"
                [ -f "$py" ] && grep -q -- "$2" "$py" && cuadernos+=("$nb")
            done < <(todos)
            shift ;;
        *.ipynb) cuadernos+=("$1") ;;
        *) echo "Argumento no reconocido: $1" >&2; exit 1 ;;
    esac
    shift
done

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true 2>/dev/null; then
    echo "No hay conexión con $HOST: conéctate a la VPN de la Universidad de León y repite." >&2
    exit 2
fi

ssh "$HOST" "mkdir -p ~/$REMOTO/repo ~/$REMOTO/logs ~/$REMOTO/out"
echo "→ rsync de ida"
rsync -az --delete --exclude '__pycache__/' --exclude '.ipynb_checkpoints/' \
      --exclude '_freeze/' --exclude '.quarto/' --exclude '_cache/' \
      notebooks src scripts pyproject.toml uv.lock "$HOST:$REMOTO/repo/"

if [ "$BUILD" = 1 ] || ! ssh "$HOST" "test -f ~/$REMOTO/pmule-render.sif"; then
    echo "→ Construyendo el contenedor (solo la primera vez o con --build)"
    ssh "$HOST" "cd ~/$REMOTO/repo && apptainer build --force --fakeroot ../pmule-render.sif scripts/cluster/pmule-render.def"
fi
[ ${#cuadernos[@]} -gt 0 ] || { echo "Sin cuadernos que renderizar."; exit 0; }

echo "→ jupytext --sync de ${#cuadernos[@]} cuaderno(s)"
if command -v jupytext >/dev/null 2>&1 || [ -x .venv/bin/jupytext ]; then
    JT="$(command -v jupytext || echo .venv/bin/jupytext)"
    "$JT" --sync --quiet "${cuadernos[@]}"
    rsync -az notebooks "$HOST:$REMOTO/repo/"
fi

lista="lista_$(date +%Y%m%d_%H%M%S).txt"
printf '%s\n' "${cuadernos[@]}" | ssh "$HOST" "cat > ~/$REMOTO/$lista"
ssh "$HOST" "rm -rf ~/$REMOTO/out/docs"
n=${#cuadernos[@]}
echo "→ sbatch: $n tarea(s), ${MAX:+máx. $MAX a la vez}${MAX:-sin límite de simultáneas}"
t0=$(date +%s)
if ssh "$HOST" "cd ~/$REMOTO && sbatch --wait --parsable${PMULE_MEM:+ --mem=$PMULE_MEM} --array=1-$n${MAX:+%$MAX} repo/scripts/cluster/render_array.sbatch $lista"; then
    estado=0
else
    estado=1
fi
echo "→ Tiempo en el clúster: $(( $(date +%s) - t0 )) s"

echo "→ rsync de vuelta"
# Quarto genera un docs/index.html que redirige a un cuaderno: no se trae (el índice es docs/index.md)
rsync -az --exclude '/index.html' "$HOST:$REMOTO/out/docs/" docs/
generados="$(ssh "$HOST" "cd ~/$REMOTO/out/docs 2>/dev/null && find . -name '*.html' | sed 's|^\./||'" || true)"
ok=0; fallos=(); hechos=(); htmls=()
for nb in "${cuadernos[@]}"; do
    nombre="$(basename "$nb" .ipynb)"
    rel="$(dirname "${nb#notebooks/}")"
    if [ "$rel" = "." ]; then relhtml="${nombre}.html"; else relhtml="${rel}/${nombre}.html"; fi
    html="docs/$relhtml"
    if grep -qxF -- "$relhtml" <<<"$generados"; then
        hechos+=("$nb"); htmls+=("$html")
        ok=$((ok + 1))
    else
        fallos+=("$nb")
    fi
done
if [ ${#htmls[@]} -gt 0 ]; then
    # Los nodos no tienen Internet: incrustar los scripts que quedaron en el CDN
    PY="$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )"
    "$PY" scripts/cluster/incrusta_cdn.py "${htmls[@]}"
    "$PY" scripts/normaliza_ids.py "${htmls[@]}"      # ids de tablas deterministas (git no acumula ruido)
    for i in "${!hechos[@]}"; do
        git add "${hechos[$i]}" "${hechos[$i]%.ipynb}.py" 2>/dev/null || true
    done
fi
echo "Hecho: $ok de $n cuaderno(s) renderizados (publica con scripts/publica_docs.sh)."
if [ ${#fallos[@]} -gt 0 ]; then
    echo "Fallaron (ver ~/$REMOTO/logs en $HOST):"; printf '  %s\n' "${fallos[@]}"
    exit 1
fi
exit $estado
