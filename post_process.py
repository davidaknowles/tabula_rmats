import os
import shutil
import json
import argparse
import re
import fcntl
import tempfile

def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())


def main():
    parser = argparse.ArgumentParser(description="Delete temp files saved in --main_dir and log rMATS completion.")
    parser.add_argument("--mode", choices=["cell", "cell_type"], default="cell",
                        help="Match the job granularity used to create the rMATS output")
    parser.add_argument("--cell_id", help="Single cell identifier")
    parser.add_argument("--cell_type", help="Cell type name")
    parser.add_argument("--main_dir", required=True, help="Main directory for txt files")
    args = parser.parse_args()

    main_dir = args.main_dir
    if args.mode == "cell":
        if not args.cell_id:
            raise ValueError("--cell_id is required in cell mode")
        sample_name = safe_name(args.cell_id)
        completed_filename = "completed_cells.json"
    else:
        if not args.cell_type:
            raise ValueError("--cell_type is required in cell_type mode")
        sample_name = safe_name(args.cell_type)
        completed_filename = "completed.json"

    cell_dir = os.path.join(main_dir, sample_name)

    # Delete temp files
    dirs_to_delete = [
        os.path.join(cell_dir, "temp"),
        os.path.join(cell_dir, "output", "tmp"),
        os.path.join(cell_dir, "data")
    ]
    for d in dirs_to_delete:
        if os.path.exists(d):
            shutil.rmtree(d)  # Recursively delete all contents
            print(f"Deleted {d}")
        else:
            print(f"Skipped {d} (does not exist)")
    print()

    # Zip output files
    output_dir = os.path.join(cell_dir, "output")
    if os.path.exists(output_dir):
        zip_path = os.path.join(cell_dir, "output_archive")
        shutil.make_archive(base_name=zip_path, format="zip", root_dir=cell_dir, base_dir="output")
        print(f"Zipped output directory to {zip_path}.zip")
    else:
        print(f"Output directory {output_dir} not found.")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed unzipped output folder {output_dir}")

    # Add cell/cell type to completed JSON. Array tasks finish concurrently, so
    # guard the read-modify-write sequence to avoid lost updates.
    json_file = os.path.join(main_dir, completed_filename)
    lock_file = f"{json_file}.lock"
    with open(lock_file, "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                completed_cells = set(json.load(f))
        else:
            completed_cells = set()

        completed_cells.add(sample_name)

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{completed_filename}.",
            suffix=".tmp",
            dir=os.path.dirname(json_file),
            text=True,
        )
        with os.fdopen(fd, "w") as f:
            json.dump(sorted(completed_cells), f)
        os.replace(tmp_path, json_file)


if __name__ == "__main__":
    main()
