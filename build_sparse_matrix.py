"""Build a sparse cells x events matrix from per-cell rMATS PSI tables."""

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from typing import Dict, List, Tuple

import pandas.errors
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad


MERGE_COLS = ["gene_id", "exon_location", "exon_strand", "chromosome"]
VALUE_COLS = {"psi", "inclusion_count", "exclusion_count", "sample_label"}
_FILL_CONTEXT = {}


def event_key(row: pd.Series) -> str:
    return "|".join([str(row[col]) for col in MERGE_COLS])


def find_psi_file(psi_dir: str, safe_cell_id: str, input_format: str = "auto") -> str:
    if input_format == "auto":
        candidates = [
            os.path.join(psi_dir, f"{safe_cell_id}.parquet"),
            os.path.join(psi_dir, f"{safe_cell_id}.csv"),
        ]
    else:
        extension = "parquet" if input_format == "parquet" else "csv"
        candidates = [os.path.join(psi_dir, f"{safe_cell_id}.{extension}")]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


def prepare_event_df(psi_file: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(psi_file) or os.path.getsize(psi_file) == 0:
        raise ValueError(f"Missing or empty PSI table: {psi_file}")
    try:
        if psi_file.endswith(".parquet"):
            df = pd.read_parquet(psi_file)
        else:
            df = pd.read_csv(psi_file)
    except pandas.errors.EmptyDataError as exc:
        raise ValueError(f"Empty PSI table: {psi_file}") from exc
    required_cols = {"event_id", "psi", "inclusion_count", "exclusion_count"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {psi_file}: {sorted(missing_cols)}")

    df = df.drop_duplicates(subset=["event_id"]).copy()
    df["psi"] = pd.to_numeric(df["psi"], errors="coerce")
    df["inclusion_count"] = pd.to_numeric(df["inclusion_count"], errors="coerce")
    df["exclusion_count"] = pd.to_numeric(df["exclusion_count"], errors="coerce")
    for col in ("incformlen", "skipformlen"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keys = df["event_id"].astype(str)
    return df, keys


def event_metadata_from_row(row_dict: dict) -> dict:
    metadata = {}
    for col, value in row_dict.items():
        if col in VALUE_COLS:
            continue
        metadata[col] = value
    return metadata


def sanitize_metadata_for_anndata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            values = np.array(["" if pd.isna(value) else str(value) for value in df[col].to_numpy()], dtype=object)
            categories = pd.Index(pd.unique(values), dtype=object)
            df[col] = pd.Categorical(values, categories=categories)
    return df


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


def chunked(items: List[dict], chunk_size: int) -> List[List[dict]]:
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def index_chunk(task: Tuple[int, List[dict], str, str]) -> Tuple[int, List[dict], Dict[str, dict], int, int, int, int]:
    chunk_id, rows, psi_dir, input_format = task
    cell_rows = []
    event_meta = {}
    psi_nnz = 0
    inclusion_nnz = 0
    exclusion_nnz = 0
    count_observed_nnz = 0

    for row in rows:
        cell_row_idx = int(row["cell_index"])
        cell_id = str(row["cell_id"])
        safe_cell_id = str(row["safe_cell_id"])
        cell_type = str(row["cell_ontology_class"])
        psi_file = find_psi_file(psi_dir, safe_cell_id, input_format)
        missing_psi = not os.path.exists(psi_file)
        observed_events = 0
        if missing_psi:
            print(f"⚠️ Missing PSI table for {cell_id}: {psi_file}", flush=True)
        else:
            df, keys = prepare_event_df(psi_file)
            observed_events = len(df)
            psi_nnz += int(df["psi"].notna().sum())
            count_mask = df["inclusion_count"].notna() & df["exclusion_count"].notna()
            inc_nonzero = count_mask & (df["inclusion_count"] > 0)
            exc_nonzero = count_mask & (df["exclusion_count"] > 0)
            any_nonzero = inc_nonzero | exc_nonzero
            inclusion_nnz += int(inc_nonzero.sum())
            exclusion_nnz += int(exc_nonzero.sum())
            count_observed_nnz += int(any_nonzero.sum())

            for key, row_dict in zip(keys, df.to_dict("records")):
                event_meta.setdefault(key, event_metadata_from_row(row_dict))

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

    return chunk_id, cell_rows, event_meta, psi_nnz, inclusion_nnz, exclusion_nnz, count_observed_nnz


def init_fill_worker(psi_dir: str, event_to_col: Dict[str, int], input_format: str) -> None:
    _FILL_CONTEXT["psi_dir"] = psi_dir
    _FILL_CONTEXT["event_to_col"] = event_to_col
    _FILL_CONTEXT["input_format"] = input_format


def fill_chunk(task: Tuple[int, List[dict], str]) -> Tuple[int, str, int, int, int, int]:
    chunk_id, rows, tmp_dir = task
    psi_dir = _FILL_CONTEXT["psi_dir"]
    event_to_col = _FILL_CONTEXT["event_to_col"]
    input_format = _FILL_CONTEXT["input_format"]

    data_rows = []
    data_cols = []
    data_vals = []
    inclusion_rows = []
    inclusion_cols = []
    inclusion_vals = []
    exclusion_rows = []
    exclusion_cols = []
    exclusion_vals = []
    count_observed_rows = []
    count_observed_cols = []

    for row in rows:
        cell_row_idx = int(row["cell_index"])
        safe_cell_id = str(row["safe_cell_id"])
        psi_file = find_psi_file(psi_dir, safe_cell_id, input_format)
        if not os.path.exists(psi_file):
            continue

        df, keys = prepare_event_df(psi_file)
        col_idx = np.fromiter((event_to_col[key] for key in keys), dtype=np.int32, count=len(df))

        psi_mask = df["psi"].notna().to_numpy()
        n_psi = int(psi_mask.sum())
        if n_psi:
            data_rows.append(np.full(n_psi, cell_row_idx, dtype=np.int32))
            data_cols.append(col_idx[psi_mask])
            data_vals.append(df.loc[psi_mask, "psi"].to_numpy(dtype=np.float32))

        count_mask = df["inclusion_count"].notna() & df["exclusion_count"].notna()
        inc_nonzero = (count_mask & (df["inclusion_count"] > 0)).to_numpy()
        exc_nonzero = (count_mask & (df["exclusion_count"] > 0)).to_numpy()
        any_nonzero = inc_nonzero | exc_nonzero
        n_inc = int(inc_nonzero.sum())
        if n_inc:
            inclusion_rows.append(np.full(n_inc, cell_row_idx, dtype=np.int32))
            inclusion_cols.append(col_idx[inc_nonzero])
            inclusion_vals.append(df.loc[inc_nonzero, "inclusion_count"].to_numpy(dtype=np.float32))
        n_exc = int(exc_nonzero.sum())
        if n_exc:
            exclusion_rows.append(np.full(n_exc, cell_row_idx, dtype=np.int32))
            exclusion_cols.append(col_idx[exc_nonzero])
            exclusion_vals.append(df.loc[exc_nonzero, "exclusion_count"].to_numpy(dtype=np.float32))
        n_obs = int(any_nonzero.sum())
        if n_obs:
            count_observed_rows.append(np.full(n_obs, cell_row_idx, dtype=np.int32))
            count_observed_cols.append(col_idx[any_nonzero])

    data_rows_arr = np.concatenate(data_rows) if data_rows else np.empty(0, dtype=np.int32)
    data_cols_arr = np.concatenate(data_cols) if data_cols else np.empty(0, dtype=np.int32)
    data_vals_arr = np.concatenate(data_vals) if data_vals else np.empty(0, dtype=np.float32)
    inclusion_rows_arr = np.concatenate(inclusion_rows) if inclusion_rows else np.empty(0, dtype=np.int32)
    inclusion_cols_arr = np.concatenate(inclusion_cols) if inclusion_cols else np.empty(0, dtype=np.int32)
    inclusion_vals_arr = np.concatenate(inclusion_vals) if inclusion_vals else np.empty(0, dtype=np.float32)
    exclusion_rows_arr = np.concatenate(exclusion_rows) if exclusion_rows else np.empty(0, dtype=np.int32)
    exclusion_cols_arr = np.concatenate(exclusion_cols) if exclusion_cols else np.empty(0, dtype=np.int32)
    exclusion_vals_arr = np.concatenate(exclusion_vals) if exclusion_vals else np.empty(0, dtype=np.float32)
    count_observed_rows_arr = (
        np.concatenate(count_observed_rows) if count_observed_rows else np.empty(0, dtype=np.int32)
    )
    count_observed_cols_arr = (
        np.concatenate(count_observed_cols) if count_observed_cols else np.empty(0, dtype=np.int32)
    )

    out_path = os.path.join(tmp_dir, f"chunk_{chunk_id:06d}.npz")
    np.savez(
        out_path,
        data_rows=data_rows_arr,
        data_cols=data_cols_arr,
        data_vals=data_vals_arr,
        inclusion_rows=inclusion_rows_arr,
        inclusion_cols=inclusion_cols_arr,
        inclusion_vals=inclusion_vals_arr,
        exclusion_rows=exclusion_rows_arr,
        exclusion_cols=exclusion_cols_arr,
        exclusion_vals=exclusion_vals_arr,
        count_observed_rows=count_observed_rows_arr,
        count_observed_cols=count_observed_cols_arr,
    )
    return (
        chunk_id,
        out_path,
        len(data_vals_arr),
        len(inclusion_vals_arr),
        len(exclusion_vals_arr),
        len(count_observed_rows_arr),
    )


def build_matrix(
    main_dir: str,
    limit: int = None,
    manifest_path: str = None,
    write_h5ad: bool = False,
    psi_dir: str = None,
    output_dir: str = None,
    workers: int = 1,
    chunk_size: int = 25,
    input_format: str = "auto",
) -> None:
    manifest = load_manifest(main_dir, manifest_path)
    completed = load_completed(main_dir)
    manifest = manifest[manifest["safe_cell_id"].isin(completed)].copy()
    if limit is not None:
        manifest = manifest.head(limit)
    manifest = manifest.reset_index(drop=True)

    psi_dir = psi_dir or os.path.join(main_dir, "psi_data", "cells")
    output_dir = output_dir or os.path.join(main_dir, "psi_data", "final_data")
    os.makedirs(output_dir, exist_ok=True)

    event_to_col: Dict[str, int] = {}
    event_meta: Dict[str, dict] = {}
    cell_rows: List[dict] = []
    psi_nnz = 0
    inclusion_nnz = 0
    exclusion_nnz = 0
    count_observed_nnz = 0
    manifest_records = []
    for cell_row_idx, row in manifest.iterrows():
        row_dict = row.to_dict()
        row_dict["cell_index"] = cell_row_idx
        manifest_records.append(row_dict)

    index_tasks = [
        (chunk_id, rows, psi_dir, input_format)
        for chunk_id, rows in enumerate(chunked(manifest_records, chunk_size))
    ]
    if workers > 1 and len(index_tasks) > 1:
        with mp.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(index_chunk, index_tasks)
            for chunks_done, (
                _,
                chunk_cell_rows,
                chunk_event_meta,
                chunk_psi_nnz,
                chunk_inclusion_nnz,
                chunk_exclusion_nnz,
                chunk_count_observed_nnz,
            ) in enumerate(iterator, 1):
                cell_rows.extend(chunk_cell_rows)
                psi_nnz += chunk_psi_nnz
                inclusion_nnz += chunk_inclusion_nnz
                exclusion_nnz += chunk_exclusion_nnz
                count_observed_nnz += chunk_count_observed_nnz
                for key, metadata in chunk_event_meta.items():
                    if key not in event_to_col:
                        event_to_col[key] = len(event_to_col)
                        event_meta[key] = metadata
                if chunks_done % max(1, 1000 // chunk_size) == 0:
                    print(
                        f"Indexed ~{chunks_done * chunk_size} cells; "
                        f"{len(event_to_col)} events; {psi_nnz} PSI entries; "
                        f"{inclusion_nnz} inclusion entries; {exclusion_nnz} exclusion entries",
                        flush=True,
                    )
    else:
        for chunks_done, task in enumerate(index_tasks, 1):
            (
                _,
                chunk_cell_rows,
                chunk_event_meta,
                chunk_psi_nnz,
                chunk_inclusion_nnz,
                chunk_exclusion_nnz,
                chunk_count_observed_nnz,
            ) = index_chunk(task)
            cell_rows.extend(chunk_cell_rows)
            psi_nnz += chunk_psi_nnz
            inclusion_nnz += chunk_inclusion_nnz
            exclusion_nnz += chunk_exclusion_nnz
            count_observed_nnz += chunk_count_observed_nnz
            for key, metadata in chunk_event_meta.items():
                if key not in event_to_col:
                    event_to_col[key] = len(event_to_col)
                    event_meta[key] = metadata
            if chunks_done % max(1, 1000 // chunk_size) == 0:
                print(
                    f"Indexed ~{chunks_done * chunk_size} cells; "
                    f"{len(event_to_col)} events; {psi_nnz} PSI entries; "
                    f"{inclusion_nnz} inclusion entries; {exclusion_nnz} exclusion entries",
                    flush=True,
                )

    cell_rows = sorted(cell_rows, key=lambda row: row["cell_index"])

    n_cells = len(manifest)
    n_events = len(event_to_col)

    data_rows = np.empty(psi_nnz, dtype=np.int32)
    data_cols = np.empty(psi_nnz, dtype=np.int32)
    data_vals = np.empty(psi_nnz, dtype=np.float32)
    inclusion_rows = np.empty(inclusion_nnz, dtype=np.int32)
    inclusion_cols = np.empty(inclusion_nnz, dtype=np.int32)
    inclusion_vals = np.empty(inclusion_nnz, dtype=np.float32)
    exclusion_rows = np.empty(exclusion_nnz, dtype=np.int32)
    exclusion_cols = np.empty(exclusion_nnz, dtype=np.int32)
    exclusion_vals = np.empty(exclusion_nnz, dtype=np.float32)
    count_observed_rows = np.empty(count_observed_nnz, dtype=np.int32)
    count_observed_cols = np.empty(count_observed_nnz, dtype=np.int32)

    psi_pos = 0
    inclusion_pos = 0
    exclusion_pos = 0
    count_observed_pos = 0
    fill_tasks = [(chunk_id, rows, "") for chunk_id, rows in enumerate(chunked(manifest_records, chunk_size))]
    with tempfile.TemporaryDirectory(prefix="sparse_chunks_", dir=output_dir) as tmp_dir:
        fill_tasks = [(chunk_id, rows, tmp_dir) for chunk_id, rows, _ in fill_tasks]
        if workers > 1 and len(fill_tasks) > 1:
            with mp.Pool(
                processes=workers,
                initializer=init_fill_worker,
                initargs=(psi_dir, event_to_col, input_format),
            ) as pool:
                iterator = pool.imap_unordered(fill_chunk, fill_tasks)
                for chunks_done, (
                    _,
                    chunk_path,
                    chunk_psi_nnz,
                    chunk_inclusion_nnz,
                    chunk_exclusion_nnz,
                    chunk_count_observed_nnz,
                ) in enumerate(iterator, 1):
                    with np.load(chunk_path) as chunk:
                        psi_slice = slice(psi_pos, psi_pos + chunk_psi_nnz)
                        data_rows[psi_slice] = chunk["data_rows"]
                        data_cols[psi_slice] = chunk["data_cols"]
                        data_vals[psi_slice] = chunk["data_vals"]
                        psi_pos += chunk_psi_nnz

                        inclusion_slice = slice(inclusion_pos, inclusion_pos + chunk_inclusion_nnz)
                        inclusion_rows[inclusion_slice] = chunk["inclusion_rows"]
                        inclusion_cols[inclusion_slice] = chunk["inclusion_cols"]
                        inclusion_vals[inclusion_slice] = chunk["inclusion_vals"]
                        inclusion_pos += chunk_inclusion_nnz

                        exclusion_slice = slice(exclusion_pos, exclusion_pos + chunk_exclusion_nnz)
                        exclusion_rows[exclusion_slice] = chunk["exclusion_rows"]
                        exclusion_cols[exclusion_slice] = chunk["exclusion_cols"]
                        exclusion_vals[exclusion_slice] = chunk["exclusion_vals"]
                        exclusion_pos += chunk_exclusion_nnz

                        observed_slice = slice(
                            count_observed_pos,
                            count_observed_pos + chunk_count_observed_nnz,
                        )
                        count_observed_rows[observed_slice] = chunk["count_observed_rows"]
                        count_observed_cols[observed_slice] = chunk["count_observed_cols"]
                        count_observed_pos += chunk_count_observed_nnz
                    os.remove(chunk_path)
                    if chunks_done % max(1, 1000 // chunk_size) == 0:
                        print(
                            f"Filled ~{chunks_done * chunk_size} cells; "
                            f"{psi_pos}/{psi_nnz} PSI entries; "
                            f"{inclusion_pos}/{inclusion_nnz} inclusion entries; "
                            f"{exclusion_pos}/{exclusion_nnz} exclusion entries",
                            flush=True,
                        )
        else:
            init_fill_worker(psi_dir, event_to_col, input_format)
            for chunks_done, task in enumerate(fill_tasks, 1):
                (
                    _,
                    chunk_path,
                    chunk_psi_nnz,
                    chunk_inclusion_nnz,
                    chunk_exclusion_nnz,
                    chunk_count_observed_nnz,
                ) = fill_chunk(task)
                with np.load(chunk_path) as chunk:
                    psi_slice = slice(psi_pos, psi_pos + chunk_psi_nnz)
                    data_rows[psi_slice] = chunk["data_rows"]
                    data_cols[psi_slice] = chunk["data_cols"]
                    data_vals[psi_slice] = chunk["data_vals"]
                    psi_pos += chunk_psi_nnz

                    inclusion_slice = slice(inclusion_pos, inclusion_pos + chunk_inclusion_nnz)
                    inclusion_rows[inclusion_slice] = chunk["inclusion_rows"]
                    inclusion_cols[inclusion_slice] = chunk["inclusion_cols"]
                    inclusion_vals[inclusion_slice] = chunk["inclusion_vals"]
                    inclusion_pos += chunk_inclusion_nnz

                    exclusion_slice = slice(exclusion_pos, exclusion_pos + chunk_exclusion_nnz)
                    exclusion_rows[exclusion_slice] = chunk["exclusion_rows"]
                    exclusion_cols[exclusion_slice] = chunk["exclusion_cols"]
                    exclusion_vals[exclusion_slice] = chunk["exclusion_vals"]
                    exclusion_pos += chunk_exclusion_nnz

                    observed_slice = slice(count_observed_pos, count_observed_pos + chunk_count_observed_nnz)
                    count_observed_rows[observed_slice] = chunk["count_observed_rows"]
                    count_observed_cols[observed_slice] = chunk["count_observed_cols"]
                    count_observed_pos += chunk_count_observed_nnz
                os.remove(chunk_path)
                if chunks_done % max(1, 1000 // chunk_size) == 0:
                    print(
                        f"Filled ~{chunks_done * chunk_size} cells; "
                        f"{psi_pos}/{psi_nnz} PSI entries; "
                        f"{inclusion_pos}/{inclusion_nnz} inclusion entries; "
                        f"{exclusion_pos}/{exclusion_nnz} exclusion entries",
                        flush=True,
                    )

    if (
        psi_pos != psi_nnz
        or inclusion_pos != inclusion_nnz
        or exclusion_pos != exclusion_nnz
        or count_observed_pos != count_observed_nnz
    ):
        raise RuntimeError(
            "Sparse entry count mismatch: "
            f"psi {psi_pos}/{psi_nnz}, "
            f"inclusion {inclusion_pos}/{inclusion_nnz}, "
            f"exclusion {exclusion_pos}/{exclusion_nnz}, "
            f"count_observed {count_observed_pos}/{count_observed_nnz}"
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
            (inclusion_rows, inclusion_cols),
        ),
        shape=(n_cells, n_events),
    ).tocsr()
    exclusion_count_matrix = sparse.coo_matrix(
        (
            exclusion_vals,
            (exclusion_rows, exclusion_cols),
        ),
        shape=(n_cells, n_events),
    ).tocsr()
    count_observed_matrix = sparse.coo_matrix(
        (
            np.ones(count_observed_nnz, dtype=np.int8),
            (count_observed_rows, count_observed_cols),
        ),
        shape=(n_cells, n_events),
    ).tocsr()

    sparse.save_npz(os.path.join(output_dir, "cell_event_psi.npz"), psi_matrix)
    sparse.save_npz(os.path.join(output_dir, "cell_event_observed.npz"), observed_matrix)
    sparse.save_npz(os.path.join(output_dir, "cell_event_inclusion_count.npz"), inclusion_count_matrix)
    sparse.save_npz(os.path.join(output_dir, "cell_event_exclusion_count.npz"), exclusion_count_matrix)
    sparse.save_npz(os.path.join(output_dir, "cell_event_count_observed.npz"), count_observed_matrix)

    cell_meta_df = pd.DataFrame(cell_rows)
    cell_meta_df = sanitize_metadata_for_anndata(cell_meta_df)
    cell_meta_df.to_csv(os.path.join(output_dir, "cell_metadata.csv"), index=False)

    event_meta_df = pd.DataFrame(
        [event_meta[key] for key in event_to_col.keys()]
    )
    event_meta_df.insert(0, "event_index", range(len(event_meta_df)))
    event_meta_df = sanitize_metadata_for_anndata(event_meta_df)
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
                "count_observed_semantics": "cell-event pairs with at least one positive inclusion or exclusion count",
                "stored_count_semantics": "inclusion_count and exclusion_count matrices store positive counts only; absent entries are zero",
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
    parser.add_argument("--psi-dir", help="Directory containing per-cell PSI CSVs")
    parser.add_argument("--output-dir", help="Directory for final sparse matrix outputs")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel CSV reader workers")
    parser.add_argument("--chunk-size", type=int, default=25, help="Cells per parallel collation chunk")
    parser.add_argument("--input-format", choices=["auto", "csv", "parquet"], default="auto", help="Per-cell table format")
    args = parser.parse_args()
    build_matrix(
        args.main_dir,
        args.limit,
        args.manifest,
        args.write_h5ad,
        args.psi_dir,
        args.output_dir,
        args.workers,
        args.chunk_size,
        args.input_format,
    )


if __name__ == "__main__":
    main()
