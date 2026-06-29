#!/bin/bash

set -e
source ~/.bashrc
# Initialize conda for non-interactive shell
source /gpfs/commons/home/nkeung/miniconda3/etc/profile.d/conda.sh

# Conda environment: Pandas, Numpy, PySam, 
conda activate tabula_sapiens
cd "$(dirname "$0")"

python verify_rmats.py --mode cell                                                  # Ensure all cells successfully ran
python rmats_results.py --mode cell --main_dir /gpfs/commons/home/nkeung/tabula_muris_data      # Extract per-cell PSI tables
python build_sparse_matrix.py --main_dir /gpfs/commons/home/nkeung/tabula_muris_data             # Build sparse cells x events matrix
