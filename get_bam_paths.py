import pandas as pd
import os


def print_basic_metadata(tm):
    """
    Checks and prints basic information about mouse dataset
    """
    print("----- TABULA MURIS SENIS METADATA -----\n")

    print(f"Total cells: {len(tm)}")
    print(f"Total specimens: {len(tm['mouse.id'].unique())}")
    print(f"Specimen IDs: {tm['mouse.id'].unique()}\n")

    print(f"Cell Types with over 30 cells:")
    counts = tm['cell_ontology_class'].value_counts()
    common_cell_types = counts[counts > 30]
    print(f"{len(common_cell_types)} out of {len(counts)} classes have at least 30 cells observed\n")
    print(common_cell_types)


    print(f"\n----------------------------------------\n")

def map_path_names(path):
    """
    Searches a directory for all BAM files in a given path. Constructs and returns a Pandas dataframe with these paths
    """
    bam_paths = []

    for dirname, _, filenames in os.walk(path):
        for f in filenames:
            # Check all BAM files. Exclude duplicates (called "Aligned.sortedByCoord.out.bam.CB.bam")
            if f.endswith(".bam") and "Aligned.sortedByCoord.out.bam.CB.bam" not in f:
                full_path = os.path.join(dirname, f)
                cell_id = f.split(".")[0]
                bam_paths.append((cell_id, full_path))
    
    bam_df = pd.DataFrame(bam_paths, columns=["clean_cell_id", "bam_path"])
    return bam_df

def pick_latest_date(bam_df):
    """
    Function to remove re-runs of the same cell. Specifically only works on experiments 180813_A00111_0188_AH7G2FDSXX and 
    180831_A00111_0201_BH7WGCDSXX. Will always pick 180831_A00111_0201_BH7WGCDSXX because it is the re-run, and BAM sizes
    are generally larger
    """

    # Sanity checks
    # print("Original BAMs:", len(bam_df))
    # print("Unique cells:", bam_df["clean_cell_id"].nunique())

    bam_df["run_date"] = bam_df["bam_path"].str.extract(r"/(\d{6})_A00111")
    bam_df["run_date"] = bam_df["run_date"].astype(int)

    # Sort cells by their cell IDs and their run dates. Then drop the later one
    bam_df = (
    bam_df.sort_values(["clean_cell_id", "run_date"], ascending=[True, False])
          .drop_duplicates("clean_cell_id", keep="first")
    )

    # Sanity checks
    # print("Cells retained:", len(bam_df))
    # print("Date used:", bam_df["run_date"].value_counts())
    
    return bam_df
    

    
if __name__ == "__main__":

    DATA_DIR = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/"
    tm = pd.read_csv(os.path.join(DATA_DIR, "metadata", "tabula-muris-senis-facs-official-raw-obj__cell-metadata__cleaned_ids.csv"),
                     dtype={5:str})
    # Read "cell_ontology_id" as a string and save it like that

    # Initial metadata about dataset
    print_basic_metadata(tm)

    print("Searching all BAM files in directory...\n")
    bam_df = map_path_names(os.path.join(DATA_DIR, "Plate_seq"))
    print(f"Total files found: {len(bam_df)}\n")

    # ---------- Check for duplicate cell IDs ----------

    # Eliminate re-runs from 180813 and 180831
    bam_df = pick_latest_date(bam_df)

    dupes = bam_df["clean_cell_id"].value_counts()
    dupes = dupes[dupes > 1]

    if len(dupes) > 0:
        print("Duplicate BAMs found!")
        print(dupes.head(20))
        raise ValueError("Fix duplicates before continuing")
    print(f"\nSuccessfully handled duplicates\n")

    # Merge tables and save new dataframe
    tm = tm.merge(
        bam_df,
        on="clean_cell_id",
        how="left"
    )

    print(f"Length of new dataframe after merge: {len(tm)}")

    # Account for missing BAM files
    # Check for bias
    tm["has_bam"] = tm["bam_path"].notna()

    # Check cell ontology classes
    print("Cell Ontology:", tm.groupby("cell_ontology_class")["has_bam"].mean().sort_values())
    # Check age
    print("Age:", tm.groupby("age")["has_bam"].mean())

    
    # Drop rows with missing BAM paths
    print("\nDropping rows with missing BAM")
    tm_clean = tm[tm["bam_path"].notna()].copy()

    # Drop rows with missing cell ontology classes
    print("\nDropping rows with missing cell ontology classes")
    tm_clean = tm_clean[tm_clean["cell_ontology_class"].notna()]

    # Final sanity checks
    dup_check = tm_clean["clean_cell_id"].duplicated().sum()
    print("Duplicate cells in metadata:", dup_check)

    bam_dupes = bam_df["clean_cell_id"].duplicated().sum()
    print("Duplicate BAM mappings:", bam_dupes)

    print("Cells with BAM:", tm_clean["bam_path"].notna().sum())
    print("Total cells:", len(tm_clean))

    
    # Metadata After Dropping
    
    print("\n\n\n")
    print_basic_metadata(tm_clean)

    tm_clean = tm_clean.drop(columns=["run_date", "has_bam"])
    tm_clean.to_csv("/gpfs/commons/home/nkeung/tabula_muris_data/bam_paths.csv", index=False)
