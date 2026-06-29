import pandas as pd
import json
import os
import argparse
import re


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check that the expected rMATS jobs have completed")
    parser.add_argument("--mode", choices=["cell", "cell_type"], default="cell")
    parser.add_argument("--main_dir", default="/gpfs/commons/home/nkeung/tabula_muris_data", help="Top-level data directory")
    parser.add_argument("--manifest", help="Optional manifest CSV for cell mode")
    args = parser.parse_args()

    dir_name = args.main_dir
    if args.mode == "cell":
        manifest_candidates = [
            args.manifest,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".rmats_run", "cells_manifest.csv"),
            os.path.join(dir_name, "jobs", "cells_manifest.csv"),
            os.path.join(dir_name, "rmats", "cells_manifest.csv"),
        ]
        manifest_path = next((path for path in manifest_candidates if path and os.path.exists(path)), None)
        if manifest_path is None:
            raise FileNotFoundError(f"Missing manifest: {manifest_candidates[1]}")
        expected_df = pd.read_csv(manifest_path)
        id_column = "cell_id" if "cell_id" in expected_df.columns else "clean_cell_id"
        expected_names = [safe_name(x) for x in expected_df[id_column].astype(str).tolist()]
        json_file = os.path.join(dir_name, "rmats", "completed_cells.json")
        label = "cells"
    else:
        tm_metadata = os.path.join(dir_name, "cell_counts.csv")
        expected_df = pd.read_csv(tm_metadata)
        expected_names = [safe_name(x) for x in expected_df["cell_ontology_class"].astype(str).tolist()]
        json_file = os.path.join(dir_name, "rmats", "completed.json")
        label = "cell classes"

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Completion file not found: {json_file}")

    with open(json_file) as f:
        completed_cells = set(json.load(f))

    missing = [name for name in expected_names if name not in completed_cells]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} {label}: {missing[:20]}")

    print(f"✅ All {label} have rMATS results")
