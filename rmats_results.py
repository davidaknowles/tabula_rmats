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
import zipfile
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


EVENT_TYPES = ("SE", "A3SS", "A5SS", "MXE", "RI")
COORD_COLS = {
    "SE": ("exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
    "A3SS": ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"),
    "A5SS": ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"),
    "MXE": (
        "1stExonStart_0base",
        "1stExonEnd",
        "2ndExonStart_0base",
        "2ndExonEnd",
        "upstreamES",
        "upstreamEE",
        "downstreamES",
        "downstreamEE",
    ),
    "RI": ("riExonStart_0base", "riExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
}


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


def parse_event_types(value: str) -> Tuple[str, ...]:
    if value.lower() == "all":
        return EVENT_TYPES
    event_types = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    invalid = sorted(set(event_types).difference(EVENT_TYPES))
    if invalid:
        raise ValueError(f"Invalid event type(s): {invalid}; valid choices are {EVENT_TYPES}")
    return event_types


def strip_quotes(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip('"')


def raw_coordinate_key(row: pd.Series, event_type: str) -> str:
    parts = [f"{col}={row[col]}" for col in COORD_COLS[event_type]]
    return ";".join(parts)


def primary_location(row: pd.Series, event_type: str) -> str:
    if event_type == "SE":
        start = int(row["exonStart_0base"]) + 1
        end = int(row["exonEnd"])
    elif event_type in {"A3SS", "A5SS"}:
        start = int(row["longExonStart_0base"]) + 1
        end = int(row["longExonEnd"])
    elif event_type == "MXE":
        start = int(row["1stExonStart_0base"]) + 1
        end = int(row["1stExonEnd"])
    elif event_type == "RI":
        start = int(row["riExonStart_0base"]) + 1
        end = int(row["riExonEnd"])
    else:
        raise ValueError(f"Unsupported event type: {event_type}")
    return f"{row['chr']}:{start}-{end}"


def extract_psi_table(rmats_df: pd.DataFrame, sample_label: str, event_type: str) -> pd.DataFrame:
    event_type = event_type.upper()
    coord_cols = COORD_COLS[event_type]
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
    rmats_df["IncFormLen"] = pd.to_numeric(rmats_df["IncFormLen"], errors="coerce")
    rmats_df["SkipFormLen"] = pd.to_numeric(rmats_df["SkipFormLen"], errors="coerce")

    group_cols = ["chr", "strand", *coord_cols]
    first_cols = {
        "ID": "first",
        "GeneID": "first",
        "geneSymbol": "first",
        "IncFormLen": "first",
        "SkipFormLen": "first",
    }
    agg_counts = (
        rmats_df.groupby(group_cols, as_index=False)
        .agg(
            {
                "total_inc": "sum",
                "total_skip": "sum",
                **first_cols,
            }
        )
    )

    agg_counts["i_norm"] = agg_counts["total_inc"] / agg_counts["IncFormLen"]
    agg_counts["s_norm"] = agg_counts["total_skip"] / agg_counts["SkipFormLen"]
    denom = agg_counts["i_norm"] + agg_counts["s_norm"]
    agg_counts["psi"] = np.where(denom == 0, np.nan, 100 * (agg_counts["i_norm"] / denom))
    gene_id = strip_quotes(agg_counts["GeneID"])
    gene_symbol = strip_quotes(agg_counts["geneSymbol"])
    coord_key = agg_counts.apply(lambda row: raw_coordinate_key(row, event_type), axis=1)
    event_id = (
        event_type
        + "|"
        + gene_id
        + "|"
        + agg_counts["chr"].astype(str)
        + "|"
        + agg_counts["strand"].astype(str)
        + "|"
        + coord_key
    )

    output = pd.DataFrame(
        {
            "event_type": event_type,
            "exon_strand": agg_counts["strand"],
            "event_coordinates": coord_key,
            "gene_type": "NA",
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "exon_location": agg_counts.apply(lambda row: primary_location(row, event_type), axis=1),
            "exon_boundary": "",
            "chromosome": agg_counts["chr"],
            "incformlen": agg_counts["IncFormLen"],
            "skipformlen": agg_counts["SkipFormLen"],
            "inclusion_count": agg_counts["total_inc"],
            "exclusion_count": agg_counts["total_skip"],
            "psi": agg_counts["psi"],
            "event_id": event_id,
            "sample_label": sample_label,
        }
    )
    for col in sorted({coord for coords in COORD_COLS.values() for coord in coords}):
        output[col] = agg_counts[col] if col in agg_counts.columns else np.nan
    return output


def read_rmats_table(sample_dir: str, event_type: str, use_jcec: bool) -> pd.DataFrame:
    suffix = "JCEC" if use_jcec else "JC"
    relative_path = f"output/{event_type}.MATS.{suffix}.txt"
    zip_path = os.path.join(sample_dir, "output_archive.zip")
    output_path = os.path.join(sample_dir, relative_path)

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(relative_path) as handle:
                return pd.read_csv(handle, sep="\t")
    if os.path.exists(output_path):
        return pd.read_csv(output_path, sep="\t")
    raise FileNotFoundError(f"Missing rMATS table: {zip_path}:{relative_path} or {output_path}")


def process_sample(
    sample_name: str,
    main_dir: str,
    psi_dir: str,
    sample_label: str,
    event_types: Iterable[str],
    use_jcec: bool,
) -> int:
    sample_dir = os.path.join(main_dir, "rmats", sample_name)
    zip_path = os.path.join(sample_dir, "output_archive.zip")
    output_dir = os.path.join(sample_dir, "output")

    if not os.path.exists(zip_path) and not os.path.exists(output_dir):
        print(f"\t⚠️ Missing rMATS archive for {sample_name}")
        return 1

    tables = []
    for event_type in event_types:
        rmats_df = read_rmats_table(sample_dir, event_type, use_jcec)
        tables.append(extract_psi_table(rmats_df, sample_label, event_type))

    out_df = pd.concat(tables, ignore_index=True)
    os.makedirs(psi_dir, exist_ok=True)
    csv_name = os.path.join(psi_dir, f"{sample_name}.csv")
    out_df.to_csv(csv_name, index=False)
    if os.path.exists(csv_name):
        print(f"\t✅ Successfully saved {csv_name} ({len(out_df):,} events)")
        return 0
    print(f"\t⚠️ Failed to save {csv_name}")
    return 1


def process_cell_task(task: Tuple[str, str, str, str, str, Tuple[str, ...], bool]) -> int:
    sample_name, main_dir, psi_dir, sample_label, cell_id, event_types, use_jcec = task
    print(f"🔷 Processing {cell_id}...")
    return process_sample(sample_name, main_dir, psi_dir, sample_label, event_types, use_jcec)


def main():
    parser = argparse.ArgumentParser(description="Extract PSI tables from rMATS outputs")
    parser.add_argument("--mode", choices=["cell", "cell_type"], default="cell")
    parser.add_argument("--cell_id", help="Process a single cell")
    parser.add_argument("--cell_type", help="Process a single cell type")
    parser.add_argument("--main_dir", required=True, help="Top-level data directory")
    parser.add_argument("--manifest", help="Manifest CSV to process")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N completed samples")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel extraction workers")
    parser.add_argument("--event-types", default="all", help="Comma-separated event types to extract, or 'all'")
    parser.add_argument("--count-type", choices=["JCEC", "JC"], default="JCEC", help="rMATS count table type to use")
    parser.add_argument("--psi-dir", help="Output directory for per-cell PSI CSVs")
    args = parser.parse_args()

    main_dir = args.main_dir
    event_types = parse_event_types(args.event_types)
    use_jcec = args.count_type == "JCEC"
    if args.mode == "cell":
        manifest = load_manifest(main_dir, args.manifest)
        completed = read_completed(main_dir, "completed_cells.json")
        psi_dir = args.psi_dir or os.path.join(main_dir, "psi_data", "cells")
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
                event_types,
                use_jcec,
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
    psi_dir = args.psi_dir or os.path.join(main_dir, "psi_data")
    exit_code = process_sample(sample_name, main_dir, psi_dir, args.cell_type, event_types, use_jcec)
    if exit_code == 0:
        print(f"\nSuccessfully saved 1 file")


if __name__ == "__main__":
    main()
