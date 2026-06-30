#!/usr/bin/env bash
#SBATCH --job-name=tm_all_events_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=.rmats_run/logs/slurm/all_events_build_%j.out
#SBATCH --error=.rmats_run/logs/slurm/all_events_build_%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/Tabula_muris_nic
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$HOME/venv/torch/lib/python3.12/site-packages"

MAIN_DIR=/gpfs/commons/home/nkeung/tabula_muris_data
MANIFEST=.rmats_run/full_cells_manifest.csv
PSI_DIR="$MAIN_DIR/psi_data/cells_all_events_parquet"
FINAL_DIR="$MAIN_DIR/psi_data/final_data_all_events_parquet"

mkdir -p "$FINAL_DIR" .rmats_run/logs/slurm

python build_sparse_matrix.py \
  --main_dir "$MAIN_DIR" \
  --manifest "$MANIFEST" \
  --psi-dir "$PSI_DIR" \
  --input-format parquet \
  --output-dir "$FINAL_DIR" \
  --write-h5ad \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --chunk-size 20
