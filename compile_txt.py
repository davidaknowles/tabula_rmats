import argparse
import os
import re

import pandas as pd


def safe_name(value: str) -> str:
    """Make a filesystem-safe identifier."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())


def write_b1_file(output_dir: str, bam_paths):
    b1_path = os.path.join(output_dir, "b1.txt")
    with open(b1_path, "w") as handle:
        handle.write(",".join(bam_paths))
    return b1_path


def ensure_cell_layout(base_dir: str, sample_name: str) -> None:
    print("Creating intermediate directories...")
    for subdir in ["", "temp", "output", "data"]:
        dir_path = os.path.join(base_dir, sample_name, subdir)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate rMATS input files for a cell or cell type")
    parser.add_argument("--mode", choices=["cell", "cell_type"], default="cell",
                        help="Operate on individual cells or grouped cell types")
    parser.add_argument("--cell_id", help="Single cell identifier")
    parser.add_argument("--cell_type", help="Cell ontology class")
    parser.add_argument("--bam_path", help="BAM path for a single cell")
    parser.add_argument("--metadata", help="Path to metadata CSV")
    parser.add_argument("--main_dir", required=True, help="Main directory for rMATS working files")
    args = parser.parse_args()

    cwd = args.main_dir

    if args.mode == "cell":
        if not args.cell_id or not args.bam_path:
            raise ValueError("--cell_id and --bam_path are required in cell mode")

        sample_name = safe_name(args.cell_id)
        bam_paths = [args.bam_path]
        label = args.cell_type or args.cell_id
        print(f"Preparing single-cell job for {args.cell_id} ({label})")
    else:
        if not args.cell_type or not args.metadata:
            raise ValueError("--cell_type and --metadata are required in cell_type mode")

        metadata_file = args.metadata
        ts = pd.read_csv(metadata_file)
        ts = ts[ts["cell_ontology_class"] == args.cell_type]
        bam_paths = ts["bam_path"].dropna().tolist()
        sample_name = safe_name(args.cell_type)
        label = args.cell_type
        print(f"Preparing grouped job for {args.cell_type}")

    if not bam_paths:
        raise ValueError(f"No BAM paths found for {label}")

    print(f"{len(bam_paths)} BAM file(s) to process for {label}\n")
    ensure_cell_layout(cwd, sample_name)

    print("Creating input .txt file...")
    b1_path = write_b1_file(os.path.join(cwd, sample_name), bam_paths)
    if os.path.exists(b1_path):
        print(f"✅ Successfully saved {b1_path}\n")


if __name__ == "__main__":
    main()
