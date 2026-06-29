#!/bin/bash

set -e
source ~/.bashrc
# Initialize conda for non-interactive shell
source /gpfs/commons/home/nkeung/miniconda3/etc/profile.d/conda.sh

# Conda environment: Pandas, Numpy, PySam, 
conda activate tabula_sapiens
cd "$(dirname "$0")"

python get_bam_paths.py                 # Basic metadata checks. Gets and saves paths to all BAM files
python check_cell_types.py              # Filters cell types with >= 30 observations

# Load samtools module (only for Tabula Sapiens)
# module load SAMtools/
python check_read_len.py                # Double check that all BAM paths have read lengths of 100
python submit_jobs.py --mode cell        # Submit all single-cell rMATS jobs through Slurm

# All rMATS jobs submitted. Double check that these are completed. 
# Then run tm_psi_pipeline.sh
