"""
Extract per-sample PSI tables from rMATS outputs.

Default mode is single-cell:
  - each completed cell gets its own PSI table in psi_data/cells/
  - rows are keyed by shared exon coordinates so they can be merged later

Legacy cell-type mode is still available for compatibility, but the new
pipeline uses per-cell tables and a downstream sparse matrix builder.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import zipfile
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


MERGE_COLS = ["gene_id", "exon_location", "exon_strand", "chromosome"]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())


def read_completed(main_dir: str, filename: str) -> set:
    json_path = os.path.join(main_dir, "rmats", filename)
    if not os.path.exists(json_path):
        return set()
    with open(json_path, "r") as handle:
        return set(json.load(handle))


def load_manifest(main_dir: str, manifest_path: Optional[str] = None) -> pd.DataFrame:
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


def extract_psi_table(rmats_df: pd.DataFrame, sample_label: str) -> pd.DataFrame:
    ijc_df = (
        rmats_df["IJC_SAMPLE_1"].fillna("NA").astype(str).str.split(",", expand=True).replace("NA", np.nan).astype(float)
    )
    sjc_df = (
        rmats_df["SJC_SAMPLE_1"].fillna("NA").astype(str).str.split(",", expand=True).replace("NA", np.nan).astype(float)
    )

    mask = ijc_df.isna() | sjc_df.isna()
    ijc_df[mask] = np.nan
    sjc_df[mask] = np.nan

    rmats_df = rmats_df.copy()
    rmats_df["total_inc"] = np.nansum(ijc_df.values, axis=1)
    rmats_df["total_skip"] = np.nansum(sjc_df.values, axis=1)

    group_cols = ["chr", "strand", "exonStart_0base", "exonEnd"]
    agg_counts = (
        rmats_df.groupby(group_cols, as_index=False)
        .agg(
            {
                "total_inc": "sum",
                "total_skip": "sum",
                "ID": "first",
                "GeneID": "first",
                "geneSymbol": "first",
                "upstreamES": "first",
                "upstreamEE": "first",
                "downstreamES": "first",
                "downstreamEE": "first",
            }
        )
    )

    exon_len = agg_counts["exonEnd"] - agg_counts["exonStart_0base"]
    len_i = 99 + exon_len.clip(upper=99) + (exon_len - 100 + 1).clip(lower=0)
    len_s = 99
    agg_counts["i_norm"] = agg_counts["total_inc"] / len_i
    agg_counts["s_norm"] = agg_counts["total_skip"] / len_s
    denom = agg_counts["i_norm"] + agg_counts["s_norm"]
    agg_counts["psi"] = np.where(denom == 0, np.nan, 100 * (agg_counts["i_norm"] / denom))

    output = pd.DataFrame(
        {
            "cassette_exon": "Yes",
            "alternative_splice_site_group": "No",
            "linked_exons": "No",
            "mutually_exclusive_exons": "No",
            "exon_strand": agg_counts["strand"],
            "exon_length": exon_len,
            "gene_type": "NA",
            "gene_id": agg_counts["GeneID"].str.strip('"'),
            "gene_symbol": agg_counts["geneSymbol"].str.strip('"'),
            "exon_location": agg_counts["chr"]
            + ":"
            + (agg_counts["exonStart_0base"] + 1).astype(str)
            + "-"
            + agg_counts["exonEnd"].astype(str),
            "exon_boundary": "",
            "chromosome": agg_counts["chr"],
            "inclusion_count": agg_counts["total_inc"],
            "exclusion_count": agg_counts["total_skip"],
            "psi": agg_counts["psi"],
            "event_id": agg_counts["GeneID"].str.strip('"')
            + "|"
            + agg_counts["chr"]
            + "|"
            + agg_counts["strand"]
            + "|"
            + (agg_counts["exonStart_0base"] + 1).astype(str)
            + "-"
            + agg_counts["exonEnd"].astype(str),
            "sample_label": sample_label,
        }
    )
    return output


def process_sample(sample_name: str, main_dir: str, psi_dir: str, sample_label: str) -> int:
    sample_dir = os.path.join(main_dir, "rmats", sample_name)
    zip_path = os.path.join(sample_dir, "output_archive.zip")
    output_dir = os.path.join(sample_dir, "output")

    if not os.path.exists(zip_path) and not os.path.exists(output_dir):
        print(f"\t⚠️ Missing rMATS archive for {sample_name}")
        return 1

    if not os.path.exists(output_dir):
        print("\tUnzipping...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(sample_dir)
        print("\tUnzipped!")

    try:
        rmats_path = os.path.join(output_dir, "SE.MATS.JCEC.txt")
        rmats_df = pd.read_csv(rmats_path, sep="\t")
        out_df = extract_psi_table(rmats_df, sample_label)
        os.makedirs(psi_dir, exist_ok=True)
        csv_name = os.path.join(psi_dir, f"{sample_name}.csv")
        out_df.to_csv(csv_name, index=False)
        if os.path.exists(csv_name):
            print(f"\t✅ Successfully saved {csv_name}")
            return 0
        print(f"\t⚠️ Failed to save {csv_name}")
        return 1
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print("\tRemoved unzipped files")


def process_cell_task(task: Tuple[str, str, str, str, str]) -> int:
    sample_name, main_dir, psi_dir, sample_label, cell_id = task
    print(f"🔷 Processing {cell_id}...")
    return process_sample(sample_name, main_dir, psi_dir, sample_label)


def main():
    parser = argparse.ArgumentParser(description="Extract PSI tables from rMATS outputs")
    parser.add_argument("--mode", choices=["cell", "cell_type"], default="cell")
    parser.add_argument("--cell_id", help="Process a single cell")
    parser.add_argument("--cell_type", help="Process a single cell type")
    parser.add_argument("--main_dir", required=True, help="Top-level data directory")
    parser.add_argument("--manifest", help="Manifest CSV to process")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N completed samples")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel extraction workers")
    args = parser.parse_args()

    main_dir = args.main_dir
    if args.mode == "cell":
        manifest = load_manifest(main_dir, args.manifest)
        completed = read_completed(main_dir, "completed_cells.json")
        psi_dir = os.path.join(main_dir, "psi_data", "cells")
        if args.cell_id:
            manifest = manifest[manifest["cell_id"].astype(str) == args.cell_id]
        else:
            manifest = manifest[manifest["safe_cell_id"].isin(completed)]
        if args.limit is not None:
            manifest = manifest.head(args.limit)

        tasks = [
            (
                safe_name(row["safe_cell_id"]),
                main_dir,
                psi_dir,
                row["cell_ontology_class"],
                row["cell_id"],
            )
            for _, row in manifest.iterrows()
        ]
        files_saved = 0
        if args.workers > 1 and len(tasks) > 1:
            with mp.Pool(processes=args.workers) as pool:
                for exit_code in pool.imap_unordered(process_cell_task, tasks, chunksize=10):
                    if exit_code == 0:
                        files_saved += 1
        else:
            for task in tasks:
                exit_code = process_cell_task(task)
                if exit_code == 0:
                    files_saved += 1
        print(f"\nSuccessfully saved {files_saved} files")
        return

    # Legacy cell-type path
    if not args.cell_type:
        raise ValueError("--cell_type is required in cell_type mode")
    completed = read_completed(main_dir, "completed.json")
    if args.cell_type.replace(" ", "_") not in completed:
        print(f"\t⚠️ {args.cell_type} is not marked complete, continuing anyway")

    sample_name = safe_name(args.cell_type)
    psi_dir = os.path.join(main_dir, "psi_data")
    exit_code = process_sample(sample_name, main_dir, psi_dir, args.cell_type)
    if exit_code == 0:
        print(f"\nSuccessfully saved 1 file")


if __name__ == "__main__":
    main()
