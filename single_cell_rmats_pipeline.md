# Single-Cell rMATS Pipeline

This repository originally ran rMATS-Turbo once per cell type and produced one PSI table per class.
That was convenient for aggregation, but it hid within-class heterogeneity and forced the downstream data
model to stay dense.

The updated workflow runs rMATS once per individual cell and then merges the results through a shared
event key so the final output is a sparse cells x events matrix.

## What Changed

- Jobs are now submitted as a Slurm array, one task per cell.
- Each task creates a single-cell `b1.txt` containing one BAM path.
- `rMATS` output is post-processed into one PSI table per cell.
- The final aggregation step builds:
  - a sparse PSI matrix
  - a sparse observation mask
  - cell metadata
  - event metadata

## Event Consistency

Events are made consistent across cells by reducing each rMATS output to a genomic event key built from:

- `gene_id`
- `exon_location`
- `exon_strand`
- `chromosome`

Within each cell, duplicate rows that map to the same exon coordinates are collapsed before export.
That makes the event universe stable enough to union across cells and assemble a single matrix.

## Run Order

1. Submit the per-cell jobs:

```bash
python submit_jobs.py --mode cell
```

2. When the Slurm array finishes, verify completion:

```bash
python verify_rmats.py --mode cell
```

3. Extract one PSI table per cell from the rMATS archives:

```bash
python rmats_results.py --mode cell --main_dir /gpfs/commons/home/nkeung/tabula_muris_data
```

4. Build the sparse matrix:

```bash
python build_sparse_matrix.py --main_dir /gpfs/commons/home/nkeung/tabula_muris_data
```

## Output Layout

- `rmats/cells_manifest.csv`
  - job manifest used by the Slurm array
- `rmats/<cell_id>/output_archive.zip`
  - archived rMATS output for one cell
- `psi_data/cells/<cell_id>.csv`
  - per-cell PSI table
- `psi_data/final_data/cell_event_psi.npz`
  - sparse PSI matrix
- `psi_data/final_data/cell_event_observed.npz`
  - sparse binary mask indicating which matrix entries were actually observed
- `psi_data/final_data/cell_metadata.csv`
  - row metadata for the matrix
- `psi_data/final_data/event_metadata.csv`
  - column metadata for the matrix
- `psi_data/final_data/matrix_info.json`
  - matrix dimensions and file names

## Sparse Matrix Interpretation

The PSI matrix stores observed values only.
The companion observation mask is required because a true PSI of `0` and a missing event would both
look like zero in a sparse numeric matrix.

Use the observation mask to distinguish:

- observed PSI value, including true zero
- missing event for that cell

## Notes

- The pipeline still filters to cell classes present in `cell_counts.csv` so it matches the original
  quality-control criteria.
- The legacy cell-type mode is still available in the scripts, but the default pipeline uses single cells.
