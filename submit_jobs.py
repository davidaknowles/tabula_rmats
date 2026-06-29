"""Submit rMATS jobs for either cell types or individual cells."""

import argparse
import json
import os
import re
import subprocess
import time
from typing import Optional, Tuple

import pandas as pd


# -------------------- CONFIG --------------------

hour = 2
memory = 128
nthread = 8
max_concurrent = 50

data_dir = "/gpfs/commons/home/nkeung/tabula_muris_data"
metadata_file = os.path.join(data_dir, "bam_paths.csv")
cell_counts_file = os.path.join(data_dir, "cell_counts.csv")

code_dir = os.path.dirname(os.path.abspath(__file__))
slurm_dir = os.path.join(code_dir, ".rmats_run")
rmats_results_dir = os.path.join(data_dir, "rmats")
conda_sh = "/gpfs/commons/home/nkeung/miniconda3/etc/profile.d/conda.sh"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())


def ensure_dirs() -> None:
    os.makedirs(slurm_dir, exist_ok=True)
    os.makedirs(os.path.join(slurm_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(slurm_dir, "output"), exist_ok=True)
    os.makedirs(rmats_results_dir, exist_ok=True)


def load_completed(mode: str) -> set[str]:
    completed_file = os.path.join(
        rmats_results_dir,
        "completed_cells.json" if mode == "cell" else "completed.json",
    )
    if not os.path.exists(completed_file):
        return set()
    with open(completed_file, "r") as handle:
        return set(json.load(handle))


def build_manifest(mode: str, limit: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    if mode == "cell":
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
        manifest_path = os.path.join(slurm_dir, "cells_manifest.csv")
    else:
        ts = pd.read_csv(metadata_file)
        counts = pd.read_csv(cell_counts_file)
        allowed_cell_types = set(counts["cell_ontology_class"].astype(str))
        ts = ts[ts["cell_ontology_class"].isin(allowed_cell_types)].copy()
        ts = ts.dropna(subset=["bam_path", "cell_ontology_class"])
        ts = ts.drop_duplicates(subset=["cell_ontology_class"])
        ts = ts[["cell_ontology_class", "bam_path"]].sort_values(
            "cell_ontology_class"
        )
        ts["safe_cell_id"] = ts["cell_ontology_class"].map(safe_name)
        manifest_path = os.path.join(slurm_dir, "cell_type_manifest.csv")

    completed = load_completed(mode)
    ts = ts[~ts["safe_cell_id"].isin(completed)].copy()
    if limit is not None:
        ts = ts.head(limit)

    ts.to_csv(manifest_path, index=False)
    return ts.reset_index(drop=True), manifest_path


def create_cell_array_script(manifest_path: str, num_jobs: int) -> str:
    timestamp = time.strftime("_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    job_name = f"ts_cells{timestamp}"
    slurm_file_path = os.path.join(
        slurm_dir, "scripts", f"ts_cells_array{timestamp}.sh"
    )
    array_end = max(0, num_jobs - 1)
    concurrency = min(max_concurrent, num_jobs)

    script = f"""#!/bin/bash
set -eo pipefail
trap 'echo "[$(date)] ERROR on line $LINENO" >&2' ERR
source ~/.bashrc
set +u
source "{conda_sh}"

LOG_DIR="{slurm_dir}/output/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/{job_name}_$SLURM_ARRAY_TASK_ID.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date)] Starting single-cell job"
echo "[$(date)] Manifest: {manifest_path}"

conda activate tabula_sapiens
set -u
MANIFEST="{manifest_path}"
eval "$(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import pandas as pd
import re
import shlex
import sys

manifest = sys.argv[1]
index = int(sys.argv[2])
row = pd.read_csv(manifest).iloc[index]

def safe_name(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())

print(f"CELL_ID={{shlex.quote(str(row['cell_id']))}}")
print(f"CELL_TYPE={{shlex.quote(str(row['cell_ontology_class']))}}")
print(f"BAM_PATH={{shlex.quote(str(row['bam_path']))}}")
print(f"SAFE_CELL_ID={{shlex.quote(safe_name(row['safe_cell_id']))}}")
PY
)"

cd "{code_dir}"
python compile_txt.py --mode cell --cell_id "$CELL_ID" --cell_type "$CELL_TYPE" --bam_path "$BAM_PATH" --main_dir "{rmats_results_dir}"
conda deactivate

set +u
source "{conda_sh}"
conda activate rmats_testing
set -u
WORKDIR="{rmats_results_dir}/$SAFE_CELL_ID"
cd "$WORKDIR"
echo "Starting rMATS for $CELL_ID at $(date)"
rmats.py --gtf "/gpfs/commons/home/nkeung/gene_annotations/gencode.vM25.primary_assembly.annotation.gtf" \\
    --b1 b1.txt \\
    --od output \\
    --tmp data \\
    -t single \\
    --readLength 100 \\
    --nthread {nthread} \\
    --statoff
echo "rMATS finished with exit code $?"
conda deactivate

cd "{code_dir}"
python3 post_process.py --mode cell --cell_id "$CELL_ID" --main_dir "{rmats_results_dir}"
echo "[$(date)] Finished single-cell job"
echo "Job done for $CELL_ID"
"""

    with open(slurm_file_path, "w") as handle:
        handle.write(
            "#!/bin/bash\n"
            "##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION\n"
            "##NECESSARY JOB SPECIFICATIONS\n"
            f"#SBATCH --job-name={job_name}\n"
            "#SBATCH --partition=cpu\n"
            f"#SBATCH --time={hour}:00:00\n"
            f"#SBATCH --mem={memory}G\n"
            f"#SBATCH --cpus-per-task={nthread}\n"
            "#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --output={slurm_dir}/output/out_{job_name}.%A_%a\n"
            "#SBATCH --mail-user=nkeung@nygenome.org\n"
            f"#SBATCH --array=0-{array_end}%{concurrency}\n\n"
            f"{script}"
        )
    return slurm_file_path


def create_celltype_script(manifest_path: str, num_jobs: int) -> str:
    timestamp = time.strftime("_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    job_name = f"ts_celltypes{timestamp}"
    slurm_file_path = os.path.join(
        slurm_dir, "scripts", f"ts_celltypes{timestamp}.sh"
    )

    script = f"""#!/bin/bash
set -eo pipefail
trap 'echo "[$(date)] ERROR on line $LINENO" >&2' ERR
source ~/.bashrc
set +u
source "{conda_sh}"

LOG_DIR="{slurm_dir}/output/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/{job_name}_$SLURM_ARRAY_TASK_ID.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date)] Starting cell-type job"
echo "[$(date)] Manifest: {manifest_path}"

conda activate tabula_sapiens
set -u
MANIFEST="{manifest_path}"
eval "$(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import pandas as pd
import re
import shlex
import sys

manifest = sys.argv[1]
index = int(sys.argv[2])
row = pd.read_csv(manifest).iloc[index]

def safe_name(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())

print(f"CELL_TYPE={{shlex.quote(str(row['cell_ontology_class']))}}")
print(f"BAM_PATH={{shlex.quote(str(row['bam_path']))}}")
print(f"SAFE_CELL_ID={{shlex.quote(safe_name(row['safe_cell_id']))}}")
PY
)"

cd "{code_dir}"
python compile_txt.py --mode cell_type --cell_type "$CELL_TYPE" --metadata "{metadata_file}" --main_dir "{rmats_results_dir}"
conda deactivate

set +u
source "{conda_sh}"
conda activate rmats_testing
set -u
WORKDIR="{rmats_results_dir}/$SAFE_CELL_ID"
cd "$WORKDIR"
echo "Starting rMATS for $CELL_TYPE at $(date)"
rmats.py --gtf "/gpfs/commons/home/nkeung/gene_annotations/gencode.vM25.primary_assembly.annotation.gtf" \\
    --b1 b1.txt \\
    --od output \\
    --tmp data \\
    -t single \\
    --readLength 100 \\
    --nthread {nthread} \\
    --statoff
echo "rMATS finished with exit code $?"
conda deactivate

cd "{code_dir}"
python3 post_process.py --mode cell_type --cell_type "$CELL_TYPE" --main_dir "{rmats_results_dir}"
echo "[$(date)] Finished cell-type job"
echo "Job done for $CELL_TYPE"
"""

    with open(slurm_file_path, "w") as handle:
        handle.write(
            "#!/bin/bash\n"
            "##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION\n"
            "##NECESSARY JOB SPECIFICATIONS\n"
            f"#SBATCH --job-name={job_name}\n"
            "#SBATCH --partition=cpu\n"
            f"#SBATCH --time={hour}:00:00\n"
            f"#SBATCH --mem={memory}G\n"
            f"#SBATCH --cpus-per-task={nthread}\n"
            "#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --output={slurm_dir}/output/out_{job_name}.%A_%a\n"
            "#SBATCH --mail-user=nkeung@nygenome.org\n"
            f"#SBATCH --array=0-{max(0, num_jobs - 1)}%{min(max_concurrent, num_jobs)}\n\n"
            f"{script}"
        )
    return slurm_file_path


def submit_job(
    script_path: str, dry_run: bool = False, dependency_job_id: Optional[str] = None
) -> Optional[str]:
    os.system(f"chmod u+x {script_path}")
    command = ["sbatch"]
    if dependency_job_id:
        command.append(f"--dependency=afterany:{dependency_job_id}")
    command.append(script_path)
    if dry_run:
        print(f"Dry run: {' '.join(command)}")
        return None

    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout.strip())
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    return match.group(1) if match else None


def main():
    global data_dir, metadata_file, cell_counts_file, slurm_dir, rmats_results_dir

    parser = argparse.ArgumentParser(description="Submit rMATS jobs through Slurm")
    parser.add_argument("--mode", choices=["cell", "cell_type"], default="cell")
    parser.add_argument("--limit", type=int, default=None, help="Submit only the first N jobs")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Split jobs into multiple array submissions with at most this many tasks each",
    )
    parser.add_argument(
        "--chain-chunks",
        action="store_true",
        help="Submit chunked arrays with afterany dependencies so chunks run sequentially",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write scripts without submitting them")
    parser.add_argument("--data_dir", default=data_dir, help="Top-level data directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    metadata_file = os.path.join(data_dir, "bam_paths.csv")
    cell_counts_file = os.path.join(data_dir, "cell_counts.csv")
    slurm_dir = os.path.join(code_dir, ".rmats_run")
    rmats_results_dir = os.path.join(data_dir, "rmats")

    ensure_dirs()
    manifest_df, manifest_path = build_manifest(args.mode, args.limit)
    if manifest_df.empty:
        print("No jobs to submit after filtering completed samples.")
        return

    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    if args.chunk_size and len(manifest_df) > args.chunk_size:
        chunk_dir = os.path.join(slurm_dir, "manifests")
        os.makedirs(chunk_dir, exist_ok=True)
        dependency_job_id = None
        submitted = 0
        for chunk_start in range(0, len(manifest_df), args.chunk_size):
            chunk_df = manifest_df.iloc[chunk_start : chunk_start + args.chunk_size].copy()
            chunk_index = chunk_start // args.chunk_size
            chunk_manifest_path = os.path.join(
                chunk_dir,
                f"{args.mode}_manifest_chunk_{chunk_index:04d}.csv",
            )
            chunk_df.to_csv(chunk_manifest_path, index=False)
            if args.mode == "cell":
                script_path = create_cell_array_script(chunk_manifest_path, len(chunk_df))
            else:
                script_path = create_celltype_script(chunk_manifest_path, len(chunk_df))
            job_id = submit_job(
                script_path,
                dry_run=args.dry_run,
                dependency_job_id=dependency_job_id if args.chain_chunks else None,
            )
            if args.chain_chunks and job_id:
                dependency_job_id = job_id
            submitted += len(chunk_df)
        print(
            f"Prepared {submitted} job(s) in {chunk_index + 1} chunk(s) using {manifest_path}"
        )
        return

    if args.mode == "cell":
        script_path = create_cell_array_script(manifest_path, len(manifest_df))
    else:
        script_path = create_celltype_script(manifest_path, len(manifest_df))

    submit_job(script_path, dry_run=args.dry_run)
    print(f"Prepared {len(manifest_df)} job(s) using {manifest_path}")


if __name__ == "__main__":
    main()
