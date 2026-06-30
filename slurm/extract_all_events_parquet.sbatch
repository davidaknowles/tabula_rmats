#!/usr/bin/env bash
#SBATCH --job-name=tm_all_events_extract
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-199%20
#SBATCH --output=.rmats_run/logs/slurm/all_events_extract_%A_%a.out
#SBATCH --error=.rmats_run/logs/slurm/all_events_extract_%A_%a.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/Tabula_muris_nic
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$HOME/venv/torch/lib/python3.12/site-packages"

MAIN_DIR=/gpfs/commons/home/nkeung/tabula_muris_data
MANIFEST=.rmats_run/full_cells_manifest.csv
PSI_DIR="$MAIN_DIR/psi_data/cells_all_events_parquet"

mkdir -p "$PSI_DIR" .rmats_run/logs/slurm

python rmats_results.py \
  --mode cell \
  --main_dir "$MAIN_DIR" \
  --manifest "$MANIFEST" \
  --workers "${SLURM_CPUS_PER_TASK:-2}" \
  --event-types all \
  --count-type JCEC \
  --output-format parquet \
  --skip-existing \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$SLURM_ARRAY_TASK_COUNT" \
  --psi-dir "$PSI_DIR"
