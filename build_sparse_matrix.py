"""Build a sparse cells x events matrix from per-cell PSI tables."""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad


MERGE_COLS = ["gene_id", "exon_location", "exon_strand", "chromosome"]


def event_key(row: pd.Series) -> str:
    return "|".join([str(row[col]) for col in MERGE_COLS])


def prepare_event_df(psi_file: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(psi_file)
    required_cols = {"psi", "inclusion_count", "exclusion_count"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {psi_file}: {sorted(missing_cols)}")

    df = df.drop_duplicates(subset=["event_id"] if "event_id" in df.columns else MERGE_COLS).copy()
    df["psi"] = pd.to_numeric(df["psi"], errors="coerce")
    df["inclusion_count"] = pd.to_numeric(df["inclusion_count"], errors="coerce")
    df["exclusion_count"] = pd.to_numeric(df["exclusion_count"], errors="coerce")
    if "event_id" in df.columns:
        keys = df["event_id"].astype(str)
    else:
        keys = df.apply(event_key, axis=1)
    return df, keys


def load_manifest(main_dir: str, manifest_path: str = None) -> pd.DataFrame:
    if manifest_path is not None:
        return pd.read_csv(manifest_path)
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".rmats_run", "cells_manifest.csv"),
        os.path.join(main_dir, "jobs", "cells_manifest.csv"),
        os.path.join(main_dir, "rmats", "cells_manifest.csv"),
    ]
    manifest = next((path for path in candidates if os.path.exists(path)), None)
    if manifest is None:
        raise FileNotFoundError(f"Missing manifest: {candidates[0]}")
    return pd.read_csv(manifest)


def load_completed(main_dir: str) -> set:
    json_path = os.path.join(main_dir, "rmats", "completed_cells.json")
    if not os.path.exists(json_path):
        return set()
    with open(json_path, "r") as handle:
        return set(json.load(handle))


def build_matrix(main_dir: str, limit: int = None, manifest_path: str = None, write_h5ad: bool = False) -> None:
    manifest = load_manifest(main_dir, manifest_path)
    completed = load_completed(main_dir)
    manifest = manifest[manifest["safe_cell_id"].isin(completed)].copy()
    if limit is not None:
        manifest = manifest.head(limit)
    manifest = manifest.reset_index(drop=True)

    psi_dir = os.path.join(main_dir, "psi_data", "cells")
    output_dir = os.path.join(main_dir, "psi_data", "final_data")
    os.makedirs(output_dir, exist_ok=True)

    event_to_col: Dict[str, int] = {}
    event_meta: Dict[str, dict] = {}
    cell_rows: List[dict] = []
    psi_nnz = 0
    count_nnz = 0

    for cell_row_idx, row in manifest.iterrows():
        cell_id = str(row["cell_id"])
        safe_cell_id = str(row["safe_cell_id"])
        cell_type = str(row["cell_ontology_class"])
        psi_file = os.path.join(psi_dir, f"{safe_cell_id}.csv")
        missing_psi = not os.path.exists(psi_file)
        observed_events = 0
        if missing_psi:
            print(f"⚠️ Missing PSI table for {cell_id}: {psi_file}")
        else:
            df, keys = prepare_event_df(psi_file)
            observed_events = len(df)
            psi_nnz += int(df["psi"].notna().sum())
            count_mask = df["inclusion_count"].notna() & df["exclusion_count"].notna()
            count_nnz += int(count_mask.sum())

            for key, event_row in zip(keys, df.itertuples(index=False)):
                if key not in event_to_col:
                    event_to_col[key] = len(event_to_col)
                    row_dict = event_row._asdict()
                    event_meta[key] = {
                        "event_id": key,
                        "gene_id": row_dict.get("gene_id", ""),
                        "gene_symbol": row_dict.get("gene_symbol", ""),
                        "chromosome": row_dict.get("chromosome", ""),
                        "exon_strand": row_dict.get("exon_strand", ""),
                        "exon_location": row_dict.get("exon_location", ""),
                        "exon_length": row_dict.get("exon_length", ""),
                    }

        cell_rows.append(
            {
                "cell_index": cell_row_idx,
                "cell_id": cell_id,
                "safe_cell_id": safe_cell_id,
                "cell_type": cell_type,
                "bam_path": row["bam_path"],
                "n_events_observed": int(observed_events),
                "missing_psi_table": missing_psi,
            }
        )
        if (cell_row_idx + 1) % 1000 == 0:
            print(
                f"Indexed {cell_row_idx + 1} cells; "
                f"{len(event_to_col)} events; {psi_nnz} PSI entries; {count_nnz} count entries",
                flush=True,
            )

    n_cells = len(manifest)
    n_events = len(event_to_col)

    data_rows = np.empty(psi_nnz, dtype=np.int32)
    data_cols = np.empty(psi_nnz, dtype=np.int32)
    data_vals = np.empty(psi_nnz, dtype=np.float32)
    count_rows = np.empty(count_nnz, dtype=np.int32)
    count_cols = np.empty(count_nnz, dtype=np.int32)
    inclusion_vals = np.empty(count_nnz, dtype=np.float32)
    exclusion_vals = np.empty(count_nnz, dtype=np.float32)

    psi_pos = 0
    count_pos = 0
    for cell_row_idx, row in manifest.iterrows():
        safe_cell_id = str(row["safe_cell_id"])
        psi_file = os.path.join(psi_dir, f"{safe_cell_id}.csv")
        if not os.path.exists(psi_file):
            continue

        df, keys = prepare_event_df(psi_file)
        col_idx = np.fromiter((event_to_col[key] for key in keys), dtype=np.int32, count=len(df))

        psi_mask = df["psi"].notna().to_numpy()
        n_psi = int(psi_mask.sum())
        if n_psi:
            entry_slice = slice(psi_pos, psi_pos + n_psi)
            data_rows[entry_slice] = cell_row_idx
            data_cols[entry_slice] = col_idx[psi_mask]
            data_vals[entry_slice] = df.loc[psi_mask, "psi"].to_numpy(dtype=np.float32)
            psi_pos += n_psi

        count_mask = (df["inclusion_count"].notna() & df["exclusion_count"].notna()).to_numpy()
        n_count = int(count_mask.sum())
        if n_count:
            entry_slice = slice(count_pos, count_pos + n_count)
            count_rows[entry_slice] = cell_row_idx
            count_cols[entry_slice] = col_idx[count_mask]
            inclusion_vals[entry_slice] = df.loc[count_mask, "inclusion_count"].to_numpy(dtype=np.float32)
            exclusion_vals[entry_slice] = df.loc[count_mask, "exclusion_count"].to_numpy(dtype=np.float32)
            count_pos += n_count

        if (cell_row_idx + 1) % 1000 == 0:
            print(
                f"Filled {cell_row_idx + 1} cells; "
                f"{psi_pos}/{psi_nnz} PSI entries; {count_pos}/{count_nnz} count entries",
                flush=True,
            )

    if psi_pos != psi_nnz or count_pos != count_nnz:
        raise RuntimeError(
            f"Sparse entry count mismatch: psi {psi_pos}/{psi_nnz}, count {count_pos}/{count_nnz}"
        )

    psi_matrix = sparse.coo_matrix(
        (data_vals, (data_rows, data_cols)),
        shape=(n_cells, n_events),
    ).tocsr()
    observed_matrix = sparse.coo_matrix(
        (np.ones(psi_nnz, dtype=np.int8), (data_rows, data_cols)),
        shape=(n_cells, n_events),
    ).tocsr()
    inclusion_count_matrix = sparse.coo_matrix(
        (
            inclusion_vals,
            (count_rows, count_cols),
        ),
        shape=(n_cells, n_events),
    ).tocsr()
    exclusion_count_matrix = sparse.coo_matrix(
        (
            exclusion_vals,
            (count_rows, count_cols),
        ),
        shape=(n_cells, n_events),
    ).tocsr()
    count_observed_matrix = sparse.coo_matrix(
        (
            np.ones(count_nnz, dtype=np.int8),
            (count_rows, count_cols),
        ),
        shape=(n_cells, n_events),
    ).tocsr()

    sparse.save_npz(os.path.join(output_dir, "cell_event_psi.npz"), psi_matrix)
    sparse.save_npz(os.path.join(output_dir, "cell_event_observed.npz"), observed_matrix)
    sparse.save_npz(os.path.join(output_dir, "cell_event_inclusion_count.npz"), inclusion_count_matrix)
    sparse.save_npz(os.path.join(output_dir, "cell_event_exclusion_count.npz"), exclusion_count_matrix)
    sparse.save_npz(os.path.join(output_dir, "cell_event_count_observed.npz"), count_observed_matrix)

    cell_meta_df = pd.DataFrame(cell_rows)
    cell_meta_df.to_csv(os.path.join(output_dir, "cell_metadata.csv"), index=False)

    event_meta_df = pd.DataFrame(
        [event_meta[key] for key in event_to_col.keys()]
    )
    event_meta_df.insert(0, "event_index", range(len(event_meta_df)))
    event_meta_df.to_csv(os.path.join(output_dir, "event_metadata.csv"), index=False)

    if write_h5ad:
        adata = ad.AnnData(
            X=inclusion_count_matrix,
            obs=cell_meta_df.set_index("safe_cell_id", drop=False),
            var=event_meta_df.set_index("event_id", drop=False),
        )
        adata.layers["exclusion_count"] = exclusion_count_matrix
        adata.layers["psi"] = psi_matrix
        adata.layers["psi_observed"] = observed_matrix
        adata.layers["count_observed"] = count_observed_matrix
        adata.uns["X_name"] = "inclusion_count"
        adata.write_h5ad(os.path.join(output_dir, "cell_event_counts.h5ad"), compression="gzip")

    with open(os.path.join(output_dir, "matrix_info.json"), "w") as handle:
        json.dump(
            {
                "n_cells": n_cells,
                "n_events": n_events,
                "psi_matrix": "cell_event_psi.npz",
                "observed_matrix": "cell_event_observed.npz",
                "inclusion_count_matrix": "cell_event_inclusion_count.npz",
                "exclusion_count_matrix": "cell_event_exclusion_count.npz",
                "count_observed_matrix": "cell_event_count_observed.npz",
                "anndata": "cell_event_counts.h5ad" if write_h5ad else None,
            },
            handle,
            indent=2,
        )

    print(f"Saved sparse matrix with shape {psi_matrix.shape}")
    print(f"Saved outputs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build sparse cell x event PSI matrix")
    parser.add_argument("--main_dir", required=True, help="Top-level data directory")
    parser.add_argument("--manifest", help="Manifest CSV to include")
    parser.add_argument("--limit", type=int, default=None, help="Only include the first N completed cells")
    parser.add_argument("--write-h5ad", action="store_true", help="Also write AnnData with inclusion counts in X")
    args = parser.parse_args()
    build_matrix(args.main_dir, args.limit, args.manifest, args.write_h5ad)


if __name__ == "__main__":
    main()
