#!/bin/bash
#SBATCH --job-name=pmule-render
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%A_%a.log
# Una tarea del array = un cuaderno. Uso (lo lanza scripts/render_cluster.sh):
#   cd ~/pmule_render && sbatch --array=1-N[%M] repo/scripts/cluster/render_array.sbatch lista.txt
# lista.txt: rutas relativas al repositorio (notebooks/.../X.ipynb), una por línea.
# Cada tarea trabaja en una copia privada (sin choques entre los quarto
# simultáneos) y deja el HTML en out/docs/<ruta relativa a notebooks/>.
set -euo pipefail
BASE="${SLURM_SUBMIT_DIR:-$HOME/pmule_render}"
LISTA="$BASE/$1"
SIF="$BASE/pmule-render.sif"
nb="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LISTA")"
[ -n "$nb" ] || { echo "Tarea ${SLURM_ARRAY_TASK_ID} sin cuaderno"; exit 0; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pmule_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXX")"
trap 'rm -rf "$WORK"' EXIT
rsync -a --exclude '_freeze/' --exclude '.quarto/' --exclude '_cache/' \
      "$BASE/repo/notebooks" "$BASE/repo/src" "$WORK/"
mkdir -p "$WORK/home" "$WORK/docs"
cp "$BASE/repo/scripts/html_extra.yml" "$WORK/html_extra.yml"

nombre="$(basename "$nb" .ipynb)"
rel="$(dirname "${nb#notebooks/}")"           # p. ej. colecciones/03_roy  o  .
echo "[$(hostname)] $nb"
# Cada tarea corre en su propio espacio de red (--net --network none: solo un
# loopback privado), así los núcleos de Jupyter de tareas que coinciden en un
# nodo no pueden chocar por los puertos TCP de localhost. Se reintenta por si
# acaso falla algo transitorio.
for intento in 1 2 3; do
    if apptainer exec --cleanenv --containall --net --network none --bind "$WORK:$WORK" --pwd "$WORK" \
        --env HOME="$WORK/home",MPLCONFIGDIR="$WORK/home/.mpl" \
        "$SIF" quarto render "$nb" --metadata-file "$WORK/html_extra.yml"; then
        break
    fi
    [ "$intento" = 3 ] && { echo "FALLO $nb"; exit 1; }
    echo "Reintento $((intento + 1)) de $nb"; sleep $(( 2 + RANDOM % 6 ))
done

# Quarto deja el HTML junto al cuaderno o en el output-dir del proyecto (../docs)
html=""
for c in "$WORK/$(dirname "$nb")/${nombre}.html" "$WORK/docs/${rel}/${nombre}.html"; do
    [ -f "$c" ] && { html="$c"; break; }
done
[ -n "$html" ] || { echo "No encuentro el HTML de $nb"; exit 1; }
mkdir -p "$BASE/out/docs/$rel"
cp "$html" "$BASE/out/docs/$rel/${nombre}.html"
echo "OK $nb"
