"""Reconcile single-cell rMATS completion state with output archives."""

import argparse
import json
import os
import re
from typing import Tuple

import pandas as pd


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())


def build_full_manifest(data_dir: str) -> pd.DataFrame:
    metadata_file = os.path.join(data_dir, "bam_paths.csv")
    cell_counts_file = os.path.join(data_dir, "cell_counts.csv")
    ts = pd.read_csv(metadata_file)
    counts = pd.read_csv(cell_counts_file)
    allowed_cell_types = set(counts["cell_ontology_class"].astype(str))
    ts = ts[ts["cell_ontology_class"].isin(allowed_cell_types)].copy()
    ts = ts.dropna(subset=["bam_path", "clean_cell_id"])
    ts = ts.drop_duplicates(subset=["clean_cell_id"])
    ts = ts.rename(columns={"clean_cell_id": "cell_id"})
    ts = ts[["cell_id", "cell_ontology_class", "bam_path"]].sort_values(
        ["cell_ontology_class", "cell_id"]
    )
    ts["safe_cell_id"] = ts["cell_id"].map(safe_name)
    return ts.reset_index(drop=True)


def archive_exists(rmats_dir: str, safe_cell_id: str) -> bool:
    archive = os.path.join(rmats_dir, safe_cell_id, "output_archive.zip")
    return os.path.exists(archive) and os.path.getsize(archive) > 0


def reconcile(data_dir: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rmats_dir = os.path.join(data_dir, "rmats")
    completed_path = os.path.join(rmats_dir, "completed_cells.json")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(rmats_dir, exist_ok=True)

    full_manifest = build_full_manifest(data_dir)
    archived_mask = full_manifest["safe_cell_id"].map(lambda sample: archive_exists(rmats_dir, sample))
    archived = set(full_manifest.loc[archived_mask, "safe_cell_id"].astype(str))
    missing_manifest = full_manifest.loc[~archived_mask].copy()

    if os.path.exists(completed_path):
        with open(completed_path) as handle:
            completed = set(json.load(handle))
    else:
        completed = set()
    completed.update(archived)

    tmp_path = f"{completed_path}.reconcile.tmp"
    with open(tmp_path, "w") as handle:
        json.dump(sorted(completed), handle)
    os.replace(tmp_path, completed_path)

    full_path = os.path.join(output_dir, "full_cells_manifest.csv")
    missing_path = os.path.join(output_dir, "missing_cells_manifest.csv")
    full_manifest.to_csv(full_path, index=False)
    missing_manifest.to_csv(missing_path, index=False)

    print(f"Full eligible cells: {len(full_manifest)}")
    print(f"Archive-backed completed cells in manifest: {len(archived)}")
    print(f"Missing archives: {len(missing_manifest)}")
    print(f"Completed JSON entries after reconcile: {len(completed)}")
    print(f"Full manifest: {full_path}")
    print(f"Missing manifest: {missing_path}")
    if not missing_manifest.empty:
        print("Missing examples:")
        print(missing_manifest.head(20).to_string(index=False))

    return full_manifest, missing_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile rMATS completion JSON with output archives")
    parser.add_argument("--data_dir", default="/gpfs/commons/home/nkeung/tabula_muris_data")
    parser.add_argument("--output_dir", default=".rmats_run")
    args = parser.parse_args()
    reconcile(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
