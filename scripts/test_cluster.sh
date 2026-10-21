#!/usr/bin/env bash
# Ejecuta la batería de tests de pmule en el clúster GVIS (legio2) con SLURM
# (job array) y el mismo contenedor Apptainer que render_cluster.sh.
#
# Uso (desde la raíz del repositorio, en el Mac):
#   scripts/test_cluster.sh                 # toda la suite (src/)
#   scripts/test_cluster.sh test_core.py    # solo esos ficheros (relativos a src/)
# Variables: PMULE_TROZO=8  (tests de tests_ejercicios por tarea), PMULE_MEM=4G
#
# Pasos: conexión → rsync de ida (directorio ~/pmule_test, independiente del de
# render_cluster.sh, así que pueden convivir) → recoge los tests con pytest en
# local y los reparte (un fichero = una tarea; tests_ejercicios en trozos) →
# sbatch (array) → resumen y logs de las tareas fallidas.
# Requiere que exista ~/pmule_render/pmule-render.sif (ver render_cluster.sh --build).
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${PMULE_CLUSTER:-legio2}"
REMOTO="pmule_test"
TROZO="${PMULE_TROZO:-8}"
PY="$( [ -x .venv/bin/python ] && echo "$PWD/.venv/bin/python" || echo python3 )"

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true 2>/dev/null; then
    echo "No hay conexión con $HOST: conéctate a la VPN de la Universidad de León y repite." >&2
    exit 2
fi
ssh "$HOST" "test -f ~/pmule_render/pmule-render.sif" \
    || { echo "Falta el contenedor: ejecuta scripts/render_cluster.sh --build" >&2; exit 2; }

ssh "$HOST" "mkdir -p ~/$REMOTO/repo ~/$REMOTO/logs"
echo "→ rsync de ida"
rsync -az --delete --exclude '__pycache__/' --exclude '.ipynb_checkpoints/' \
      --exclude '_freeze/' --exclude '.quarto/' --exclude '_cache/' \
      notebooks src scripts pyproject.toml uv.lock "$HOST:$REMOTO/repo/"

echo "→ recogiendo los tests"
tareas="tareas_$(date +%Y%m%d_%H%M%S)"
dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT
( cd src && "$PY" -m pytest --collect-only -q -p no:cacheprovider "$@" 2>/dev/null ) \
    | grep '::' > "$dir/ids.txt" || true
n="$("$PY" - "$dir" "$TROZO" <<'PYEOF'
import sys, collections
d, trozo = sys.argv[1], int(sys.argv[2])
por = collections.OrderedDict()
for l in open(f"{d}/ids.txt", encoding="utf-8").read().splitlines():
    l = l.removeprefix("src/")      # pytest los numera desde la raíz; las tareas corren en src/
    por.setdefault(l.split("::")[0], []).append(l)
grupos = []
for fich, ids in por.items():
    if fich.startswith("tests_ejercicios/"):
        grupos += [ids[i:i + trozo] for i in range(0, len(ids), trozo)]
    else:
        grupos.append(ids)
for k, g in enumerate(grupos, 1):
    open(f"{d}/{k:03d}.txt", "w", encoding="utf-8").write("\n".join(g) + "\n")
print(len(grupos))
PYEOF
)"
[ "$n" -gt 0 ] || { echo "No se ha recogido ningún test."; exit 1; }
total="$(wc -l < "$dir/ids.txt" | tr -d ' ')"
ssh "$HOST" "rm -rf ~/$REMOTO/$tareas ~/$REMOTO/out && mkdir -p ~/$REMOTO/$tareas ~/$REMOTO/out"
rsync -az --include='[0-9][0-9][0-9].txt' --exclude='*' "$dir/" "$HOST:$REMOTO/$tareas/"

echo "→ sbatch: $total test(s) en $n tarea(s)"
t0=$(date +%s)
estado=0
ssh "$HOST" "cd ~/$REMOTO && sbatch --wait --parsable${PMULE_MEM:+ --mem=$PMULE_MEM} --array=1-$n repo/scripts/cluster/test_array.sbatch $tareas" || estado=1
echo "→ Tiempo en el clúster: $(( $(date +%s) - t0 )) s"

ok="$(ssh "$HOST" "ls ~/$REMOTO/out/*.ok 2>/dev/null | wc -l" | tr -d ' ')"
fallos="$(ssh "$HOST" "ls ~/$REMOTO/out/*.fail 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.fail//'" || true)"
echo "Tareas correctas: $ok de $n"
if [ -n "$fallos" ] || [ "$ok" -ne "$n" ]; then
    echo "Con fallos (o sin terminar): ${fallos:-ninguna registrada}"
    for t in $fallos; do
        echo "------ tarea $t ------"
        ssh "$HOST" "grep -l \"$t.txt\\|^\\[\" ~/$REMOTO/logs/*_$((10#$t)).log 2>/dev/null | tail -1 | xargs -r tail -n 25"
    done
    exit 1
fi
echo "Suite completa: OK ($total tests)."
exit $estado
