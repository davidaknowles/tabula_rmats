import pandas as pd
import numpy as np


def save_csv(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to CSV.
    """
    df.to_csv(output_path, index=False)


def remove_allOther_species_multiz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load exon coordinate file (species-based).
    """
    # return pd.read_csv(file_path)
    df = df[df["Species Name"] == "mm10"].reset_index(drop=True)
    print(f"Loaded {len(df)} exons for mm10 from Multiz data.")
    return df

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load exon metadata file (with PSI/tissue expression).
    """
    return pd.read_csv(file_path)



def load_exon_metadata_from_ASCOT(file_path: str) -> pd.DataFrame:
    """
    Load exon metadata file (with PSI/tissue expression).
    """
    return pd.read_csv(file_path)


def parse_location(location: str):
    """
    Parse genomic location like 'chr19:58352098-58352184' → (chrom, start, end).
    """
    chrom, coords = location.split(":")
    start, end = map(int, coords.split("-"))
    return chrom, start, end


def add_parsed_coordinates(df: pd.DataFrame, location_col: str = "exon_location") -> pd.DataFrame:
    """
    Add parsed chromosome, start, end columns to metadata DataFrame.
    """
    parsed = df[location_col].apply(parse_location)
    df["chromosome_parsed"] = parsed.apply(lambda x: x[0])
    df["exon_start_parsed"] = parsed.apply(lambda x: x[1])
    df["exon_end_parsed"] = parsed.apply(lambda x: x[2])
    return df


def find_overlaps(df_multiz, df_ascot, whichPrime = 3) -> pd.DataFrame:
    """
    Find overlapping exons between two datasets.
    df1: coordinates (first file)
    df2: metadata with parsed locations (second file)
    """
    # if whichPrime == 3:
    plus_matches_3p = pd.merge(
        df_ascot[df_ascot['exon_strand'] == '+'],
        df_multiz,
        left_on=['chromosome_parsed', 'exon_start_parsed'],
        right_on=['Chromosome', 'Exon Start']
    ).drop_duplicates(subset=['Chromosome', 'Exon Start'])


    minus_matches_3p = pd.merge(
        df_ascot[df_ascot['exon_strand'] == '-'],
        df_multiz,
        left_on=['chromosome_parsed', 'exon_end_parsed'],
        right_on=['Chromosome', 'Exon End']
    ).drop_duplicates(subset=['Chromosome', 'Exon End'])
    
    # elif whichPrime == 5:
    plus_matches_5p = pd.merge(
        df_ascot[df_ascot['exon_strand'] == '+'],
        df_multiz,
        left_on=['chromosome_parsed', 'exon_end_parsed'],
        right_on=['Chromosome', 'Exon End']
    ).drop_duplicates(subset=['Chromosome', 'Exon End'])

    minus_matches_5p = pd.merge(
        df_ascot[df_ascot['exon_strand'] == '-'],
        df_multiz,
        left_on=['chromosome_parsed', 'exon_start_parsed'],
        right_on=['Chromosome', 'Exon Start']
    ).drop_duplicates(subset=['Chromosome', 'Exon Start'])

    # Concatenate all matches
    matches = pd.concat([plus_matches_3p, minus_matches_3p, plus_matches_5p, minus_matches_5p], ignore_index=True)

    # Drop duplicates based on Chromosome + coordinates
    matches = matches.drop_duplicates(subset=['Chromosome', 'Exon Start', 'Exon End'])
    total_unique_boundaries = len(df_ascot)

    match_percentage = (len(matches) / total_unique_boundaries) * 100
    print(f"Match Percentage: {match_percentage}")

    return matches


    
def get_tissue_PSI_ASCOT(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the tissue expression values from exon metadata DataFrame.
    Assumes tissue columns start after 'exon_boundary' and before the last 'chromosome' column.
    """
    # Find column positions dynamically
    start_idx = df.columns.get_loc("exon_boundary") + 1
    end_idx = df.columns.get_loc("chromosome")
    tissue_cols = df.columns[start_idx:end_idx]
    
    # Extract and convert to float
    expr_matrix = df[tissue_cols].astype(float)
    
    # Replace -1.0 values with NaN
    expr_matrix = expr_matrix.replace(-1.0, np.nan)
    
    return expr_matrix


def compute_spearman_corr(expr_matrix: pd.DataFrame, exon_ids: pd.Series) -> pd.DataFrame:
    """
    Compute Spearman correlation between exons based on tissue profiles.
    
    Args:
        expr_matrix: DataFrame (N_exons × N_tissues)
    
    Returns:
        corr_df: Spearman correlation matrix (N_exons × N_exons)
    """
    corr_df = expr_matrix.T.corr(method="spearman")
    
    # Rename rows and columns to exon_id
    corr_df.index = exon_ids
    corr_df.columns = exon_ids
    
    return corr_df


def _compute_mad_block(X_block: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    import warnings
    """
    Compute mean absolute distance (MAD) between a block of exons and all exons.

    Args:
        X_block: (B × T) array for block of exons
        X_all:   (N × T) array for all exons

    Returns:
        mad_block: (B × N) array of MAD values
    """
    diffs = np.abs(X_block[:, None, :] - X_all[None, :, :])/100
    val = np.nanmean(diffs, axis=2)
    # diffs: shape (B, N, T)
    valid_counts = np.sum(~np.isnan(diffs), axis=2)  # (B, N)
    empty_mask = valid_counts == 0                   # True where nanmean warns/returns NaN

    # 1) Total empty pairs (i,j)
    total_empty_pairs = int(empty_mask.sum())

    # 2) Rows (block exons) that have ≥1 empty pair
    rows_with_any = int(empty_mask.any(axis=1).sum())

    # 3) Columns (all exons) that have ≥1 empty pair
    cols_with_any = int(empty_mask.any(axis=0).sum())

    # 4) Rows/cols that are completely empty (every pair empty)
    rows_all_empty = int(empty_mask.all(axis=1).sum())
    cols_all_empty = int(empty_mask.all(axis=0).sum())

    # # 5) Per-exon counts (how many partners are empty for each exon)
    # empty_per_row = empty_mask.sum(axis=1)  # length B
    # empty_per_col = empty_mask.sum(axis=0)  # length N

    print(f"Total empty (i,j) pairs: {total_empty_pairs}")
    print(f"Rows with ≥1 empty pair (block exons): {rows_with_any} / {diffs.shape[0]}")
    print(f"Cols with ≥1 empty pair (all exons):   {cols_with_any} / {diffs.shape[1]}")
    print(f"Rows completely empty: {rows_all_empty}")
    print(f"Cols completely empty: {cols_all_empty}")
    
    return val
    

def format_exon_ids(mad_matrix: pd.DataFrame, exon_ids: pd.Series) -> pd.DataFrame:
    """
    Format exon IDs by replacing '.' with '_'.
    """
    mad_df = pd.DataFrame(mad_matrix, index=exon_ids, columns=exon_ids)
    mad_df.index.name = None
    # mad_df.index.name = "exon_id"
    return mad_df

def compute_meanAbsoluteDistance(expr_matrix: pd.DataFrame, exon_ids: pd.Series) -> pd.DataFrame:
    """
    Compute full mean absolute distance (MAD) matrix (not memory efficient).

    Args:
        expr_matrix: DataFrame (N_exons × N_tissues)
        exon_ids: Series of exon IDs (length N_exons)

    Returns:
        mad_df: DataFrame (N_exons × N_exons)
    """
    X = expr_matrix.to_numpy().astype(float)
    mad_matrix = _compute_mad_block(X, X)
    if exon_ids is None:
        return pd.DataFrame(mad_matrix)
    mad_df = format_exon_ids(mad_matrix, exon_ids)
    return mad_df



def compute_meanAbsoluteDistance_blockwise(expr_matrix: pd.DataFrame, exon_ids: pd.Series, block_size: int = 1000) -> pd.DataFrame:
    """
    Compute full mean absolute distance (MAD) matrix in memory-efficient blocks.

    Args:
        expr_matrix: DataFrame (N_exons × N_tissues)
        exon_ids: Series of exon IDs (length N_exons)
        block_size: Block size for memory efficiency

    Returns:
        mad_df: DataFrame (N_exons × N_exons)
    """
   
    X = expr_matrix.to_numpy().astype(float)
    N, _ = X.shape
    mad_matrix = np.empty((N, N), dtype=float)
    mad_matrix.fill(np.nan)

    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        mad_matrix[i_start:i_end, :] = _compute_mad_block(X[i_start:i_end], X)

    # mad_df = pd.DataFrame(mad_matrix, index=exon_ids, columns=exon_ids)
    # mad_df.index.name = "exon_id"
    if exon_ids is None:
        return pd.DataFrame(mad_matrix)
    mad_df = format_exon_ids(mad_matrix, exon_ids)
    return mad_df


def compress_corr_matrix(corr_df: pd.DataFrame):
    """
    Convert correlation matrix to long-form (pairs + corr).
    Only keeps upper triangle (i<j).
    """
    exon_ids = corr_df.index
    tril_mask = np.tril(np.ones(corr_df.shape), k=0).astype(bool)
    
    corr_df = corr_df.mask(tril_mask)  # mask lower triangle & diagonal
    
    # Convert to long format
    corr_long = corr_df.stack().reset_index()
    corr_long.columns = ["exon1", "exon2", "spearman_corr"]
    
    return corr_long


def save_matrix(corr_df: pd.DataFrame, output_path: str):
    """
    Save correlation matrix to CSV.
    """
    # corr_df.to_csv(output_path, index=True)
    with open(output_path, 'wb') as f:
        pickle.dump(corr_df, f)



def load_spearmanCorrFile(file_path: str) -> pd.DataFrame:
    """
    Load exon metadata file (with PSI/tissue expression).
    """
    return pd.read_pickle(file_path)


def expected_high_corr_per_exon(corr_df: pd.DataFrame, threshold: float = 0.8) -> float:
    """
    Compute the expected (average) number of exons per exon 
    that have correlation above threshold.
    
    Args:
        corr_df: Correlation matrix (exon × exon)
        threshold: cutoff value for "high correlation"
    
    Returns:
        avg_corr_partners: average number of correlated exons per exon
    """
    # Mask diagonal so self-correlations don't count
    corr = corr_df.copy()
    np.fill_diagonal(corr.values, np.nan)
    
    # Count how many exons each exon is correlated with above threshold
    counts = (corr >= threshold).sum(axis=1)
    
    # Average across all exons
    avg_corr_partners = counts.mean()
    
    print(f"Average number of exons per exon with corr ≥ {threshold}: {avg_corr_partners:.2f}")
    return avg_corr_partners



def high_corr_exons(corr_df: pd.DataFrame, sp_threshold: float = 0.8, exon_similarity_threshold: int = 500) -> set:
    
    """
    Load correlation data and return set of exons with corr >= threshold.
    Assumes correlation file is long-form: [exon1, exon2, spearman_corr].
    sp_threshold: Correlation threshold to consider "highly correlated".
    exon_similarity_threshold: Number of highly correlated exons to consider an exon "hyper-correlated". As there are a lot of -1 (NaN) values in the correlation matrix, we first remove exons that have more than this number of 1s (self-correlations) before applying the sp_threshold filter.
    values, exons with a lot of NaN values can end up being correlated with almost all exons just by chance. thus filter out those exons first. the number can be tuned. Eg, out of 1000 exons, if an exon has more than 500 exons with correlation 1, we consider it hyper-correlated and remove it first.
    """

    print(f"Original correlation matrix shape: {corr_df.shape}")

    # --- Step 1: Pre-filter to remove hyper-correlated exons ---
    # Count the number of '1s' in each row
    ones_per_exon = (corr_df == 1).sum(axis=1)

    # Identify exons to remove where the count of 1s exceeds the removal threshold
    exons_to_remove = ones_per_exon[ones_per_exon > exon_similarity_threshold].index
    num_removed = len(exons_to_remove)

    # Drop the identified rows and their corresponding columns
    corr_df = corr_df.drop(index=exons_to_remove, columns=exons_to_remove)

    print(f"Removed {num_removed} hyper-correlated exons (had > {exon_similarity_threshold} ones).")
    print(f"Filtered matrix shape: {corr_df.shape}")

    # --- Average correlation (off-diagonal only) ---
    off_diag = corr_df.where(~np.eye(len(corr_df), dtype=bool))
    avg_corr = off_diag.stack().mean()
    print(f"Average correlation (off-diagonal): {avg_corr:.4f}")

    # Create a boolean matrix where True indicates a correlation > threshold
    low_corr_mask = corr_df.values > sp_threshold

    # Set the diagonal to False to ignore self-correlations
    np.fill_diagonal(low_corr_mask, False)
    
    # Find the row and column indices where the condition is True
    involved_rows, involved_cols = np.where(low_corr_mask)
    
    # Combine all indices and find the unique ones
    all_involved_indices = np.concatenate([involved_rows, involved_cols])
    unique_involved_indices = np.unique(all_involved_indices)
    
    # Get the exon names from the DataFrame's index
    exon_names = corr_df.index.to_numpy()
    
    # Select the names of the involved exons using their indices
    involved_exon_names = exon_names[unique_involved_indices]

    print(f"Found {len(set(involved_exon_names))} high-corr exons to remove")

    # Return the names as a set for automatic de-duplication
    return set(involved_exon_names)


    
    # high_corr = corr_df[corr_df["spearman_corr"] >= threshold]
    
    # # Collect all exons that appear in high correlation pairs
    # exon_set = set(high_corr["exon1"]).union(set(high_corr["exon2"]))
    # print(f"Found {len(exon_set)} high-corr exons to remove")
    # return exon_set


def filter_exons_ASCOT(ascot_df: pd.DataFrame, high_corr_exonset: set) -> pd.DataFrame:

    """
    Remove exons that are highly correlated (>= threshold) from metadata CSV.
    """
    
    # Filter metadata
    filtered_df = ascot_df[~ascot_df["exon_id"].isin(high_corr_exonset)].reset_index(drop=True)
    return filtered_df
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time


trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")

def load_pkl(file_path: str) -> pd.DataFrame:
    """
    Load pickled DataFrame.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    # mad_df = pd.read_pickle(file_path)
    return mad_df


def extract_mad_values(mad_df: pd.DataFrame) -> np.ndarray:
    """
    Extract upper-triangular (excluding diagonal) MAD values from matrix.
    """
    mad_values = mad_df.to_numpy()
    # Take only upper triangle (i<j) to avoid duplicates and self-distances
    vals = mad_values[np.triu_indices_from(mad_values, k=1)]
    return vals[~np.isnan(vals)]


def sample_values(values: np.ndarray, n: int = 1_000_000) -> np.ndarray:
    """
    Optionally sample values for memory efficiency.
    """
    if len(values) > n:
        return np.random.choice(values, size=int(n), replace=False)
    return values


def plot_histogram(division: str, values: np.ndarray, bins: int = 100) -> None:
    """
    Plot histogram of MAD values.
    """
    plt.hist(values, bins=bins, edgecolor="black")
    plt.xlabel("Mean Absolute Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Pairwise Exon MADs")
    plt.savefig(f"/gpfs/commons/home/atalukder/Contrastive_Learning/code/ASCOT_DataWhomologs/figures/{division}_mad_histogram{trimester}.png")

    # plt.show()


def plot_kde(values: np.ndarray) -> None:
    """
    Plot KDE (smoothed distribution) of MAD values.
    """
    sns.kdeplot(values, fill=True)
    plt.xlabel("Mean Absolute Distance")
    plt.ylabel("Density")
    plt.title("MAD Distribution (KDE)")
    plt.savefig(f"/gpfs/commons/home/atalukder/Contrastive_Learning/code/ASCOT_DataWhomologs/figures/mad_kde{trimester}.png")
    # plt.show()


def summarize_distribution(values: np.ndarray) -> dict:
    """
    Compute summary statistics for MAD values.
    """
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "5th_percentile": float(np.percentile(values, 5)),
        "95th_percentile": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }