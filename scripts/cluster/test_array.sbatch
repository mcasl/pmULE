#!/bin/bash
#SBATCH --job-name=pmule-test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%A_%a.log
# Una tarea del array = un grupo de tests (un fichero test_*.py, o un trozo de
# tests_ejercicios). Lo lanza scripts/test_cluster.sh:
#   cd ~/pmule_test && sbatch --array=1-N repo/scripts/cluster/test_array.sbatch tareas_X
# tareas_X/NNN.txt: identificadores de pytest (relativos a src/), uno por línea.
# Cada tarea trabaja en una copia privada y deja el resultado en out/<NNN>.{ok,fail}.
set -euo pipefail
BASE="${SLURM_SUBMIT_DIR:-$HOME/pmule_test}"
SIF="${PMULE_SIF:-$HOME/pmule_render/pmule-render.sif}"
LISTA="$BASE/$1/$(printf '%03d' "$SLURM_ARRAY_TASK_ID").txt"
[ -f "$LISTA" ] || { echo "Tarea ${SLURM_ARRAY_TASK_ID} sin tests"; exit 0; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pmulet_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXX")"
trap 'rm -rf "$WORK"' EXIT
rsync -a --exclude '_freeze/' --exclude '.quarto/' --exclude '_cache/' \
      "$BASE/repo/notebooks" "$BASE/repo/src" "$BASE/repo/scripts" "$BASE/repo/pyproject.toml" "$WORK/"
mkdir -p "$WORK/home"
cp "$LISTA" "$WORK/ids.txt"
echo "[$(hostname)] $(wc -l < "$LISTA") test(s) de $(head -1 "$LISTA")"
mkdir -p "$BASE/out"
if apptainer exec --cleanenv --containall --net --network none --bind "$WORK:$WORK" --pwd "$WORK/src" \
    --env HOME="$WORK/home",MPLCONFIGDIR="$WORK/home/.mpl",PYTHONDONTWRITEBYTECODE=1 \
    "$SIF" bash -c 'mapfile -t ids < ../ids.txt; python -m pytest -q -p no:cacheprovider "${ids[@]}"'; then
    touch "$BASE/out/$(basename "$LISTA" .txt).ok"
else
    touch "$BASE/out/$(basename "$LISTA" .txt).fail"
    exit 1
fi
