import pandas as pd
import numpy as np
import json
from scipy.special import logit
import matplotlib.pyplot as plt
import os

def sanity_checks(full_df, psi_cols):
    problems = 0
    # 1. Uniqueness of exon_ids
    duplicates = full_df[full_df['exon_id'].duplicated(keep=False)]
    if not duplicates.empty:
        print("❌ Duplicate exon_ids found:")
        print(duplicates)
        problems += 1
    else:
        print("✅ exon_id is unique")


    # 2. Exon Boundary should be empty (type is float64)
    if (full_df["exon_boundary"].isna()).all():
        print("✅ All exon_boundary values are empty")
    else:
        print(f"❌ Some exon_boundary values are not empty")
        problems += 1


    # 3. Ensure exon_location has a value
    missing_location = full_df["exon_location"].isna().sum() + (full_df["exon_location"] == "").sum()
    if missing_location > 0:
        print(f"❌ Found {missing_location} rows with missing exon_location")
        problems += 1
    else:
        print("✅ exon_location is filled")


    # 4. PSI columns: either numeric or empty string
    for col in psi_cols:
        invalid_vals = full_df[~full_df[col].apply(lambda x: isinstance(x, (int, float)) or x == "")][col]
        if len(invalid_vals) > 0:
            print(f"❌ Column {col} has {len(invalid_vals)} invalid values")
            problems += 1

    if len(psi_cols) > 0:
        print("✅ Checked PSI columns for valid values")


    # 5.1 All rows must have at least one PSI value
    psi_numeric = full_df[psi_cols].apply(pd.to_numeric, errors='coerce')

    # Mask: True if row has **all NaN** in numeric form (i.e., no numeric PSI at all)
    mask_all_nan = psi_numeric.isna().all(axis=1)

    num_invalid = mask_all_nan.sum()
    if num_invalid > 0:
        print(f"❌ Found {num_invalid} rows with no numeric PSI values")
        # Print row indices and PSI values for inspection
        print(full_df.loc[mask_all_nan, psi_cols])
        problems += 1
    else:
        print(f"✅ All rows have at least one numeric PSI value")


    # 5.2 All PSI values must be between 0 and 100 inclusive
    psi_numeric = full_df[psi_cols].apply(pd.to_numeric, errors='coerce')

    # Mask: True if PSI < 0 or > 100
    invalid_rows = (psi_numeric < 0) | (psi_numeric > 100)

    num_invalid = invalid_rows.any(axis=1).sum()
    if num_invalid > 0:
        print(f"❌ Found {num_invalid} rows with PSI values below 0 or above 100")
        # Print row indices and PSI values for inspection
        print(full_df.loc[mask_all_nan, psi_cols])
        problems += 1
    else:
        print(f"✅ All PSI values are between 0 and 100 inclusive")
    

    # 5.3 For CSVs with no mean, no NAN values allowed. Must be -1
    psi_numeric = full_df[psi_cols].apply(pd.to_numeric, errors='coerce')

    # Mask: All rows where PSI is not NA and non-negative OR PSI is -1
    mask_invalid = ~( (psi_numeric.notna()) & ((psi_numeric >= 0) | (psi_numeric == -1)) ).any(axis=1)

    num_invalid = mask_invalid.sum()
    if num_invalid > 0:
        print(f"❌ Found {num_invalid} rows where all PSI values are invalid (NaN or < -1)")
        # Optionally print these rows for inspection
        print(full_df.loc[mask_invalid, psi_cols])
    else:
        print(f"✅ All rows have at least one valid PSI value (-1 or >=0)")


    # 6. Mean PSI and Logit Mean PSI should not be NAN or empty
    for col in ["mean_psi", "logit_mean_psi"]:
        nan_count = full_df[col].isna().sum()
        empty_count = (full_df[col] == "").sum()
        if nan_count + empty_count > 0:
            print(f"❌ Column {col} has {nan_count} NaN values and {empty_count} \"\" values")
            problems += 1
        else:
            print(f"✅ Column {col} has no NaN values")
    

    # 7 Ensure PSI is the true average of the row
    psi_numeric = full_df[psi_cols].apply(pd.to_numeric, errors='coerce')
    computed_mean = psi_numeric.mean(axis=1, skipna=True)

    mismatches = full_df[
        (full_df["mean_psi"] - computed_mean).abs() > 1e-6
    ]

    if len(mismatches) > 0:
        print(f"❌ Mean PSI does not match in {len(mismatches)} rows")
        problems += 1
    else:
        print(f"✅ Mean PSI matches in all rows")


    # 8. Ensure logit mean PSI also matches
    eps = 1e-6
    computed_logit = logit(np.clip(computed_mean / 100, eps, 1-eps))
    mismatch_logit = full_df[
        (full_df["logit_mean_psi"] - computed_logit).abs() > eps
    ]
    if len(mismatch_logit) > 0:
        print(f"❌ Logit mean PSI does not match in {len(mismatch_logit)} rows")
        problems += 1
    else:
        print(f"✅ Logit mean PSI matches in all rows")


    if problems == 0:
        print(f"\n✅ All checks passed!")
    else:
        print(f"\n{problems} problems left to address!")



def generate_plots(train_df, val_df, test_df, output_dir):
    plt.figure(figsize=(8,5))
    plt.hist(train_df["mean_psi"], bins=50, alpha=0.5, label="Train")
    plt.hist(val_df["mean_psi"], bins=50, alpha=0.5, label="Val")
    plt.hist(test_df["mean_psi"], bins=50, alpha=0.5, label="Test")
    plt.xlabel("Mean PSI (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("PSI distribution across splits")
    plt.savefig(os.path.join(output_dir, "mean_psi_dist.png"), dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8,5))
    plt.hist(train_df["logit_mean_psi"], bins=50, alpha=0.5, label="Train")
    plt.hist(val_df["logit_mean_psi"], bins=50, alpha=0.5, label="Val")
    plt.hist(test_df["logit_mean_psi"], bins=50, alpha=0.5, label="Test")
    plt.xlabel("Logit Mean PSI (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Logit PSI distribution across splits")
    plt.savefig(os.path.join(output_dir, "logit_mean_psi_dist.png"), dpi=300, bbox_inches="tight")


def main():
    # full_ascot_file = "/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/full_cassette_exons_with_mean_psi.csv"
    # final_train_data = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/TSCelltype_finetuning/train_cassette_exons_with_logit_mean_psi.csv"
    # final_validation_data = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/TSCelltype_finetuning/val_cassette_exons_with_logit_mean_psi.csv"
    # final_test_data = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/TSCelltype_finetuning/test_cassette_exons_with_logit_mean_psi.csv"
    final_train_data = "/gpfs/commons/home/nkeung/tabula_muris_data/psi_data/final_data/train_cassette_exons_with_logit_mean_psi.csv"
    final_validation_data = "/gpfs/commons/home/nkeung/tabula_muris_data/psi_data/final_data/val_cassette_exons_with_logit_mean_psi.csv"
    final_test_data = "/gpfs/commons/home/nkeung/tabula_muris_data/psi_data/final_data/test_cassette_exons_with_logit_mean_psi.csv"

    train_df = pd.read_csv(final_train_data)
    val_df = pd.read_csv(final_validation_data)
    test_df = pd.read_csv(final_test_data)

    with open("/gpfs/commons/home/nkeung/tabula_muris_data/rmats/completed.json", "r") as f:
        cells = list(json.load(f))
        psi_cols = [x.replace("_", " ") for x in cells]
    # psi_cols = ["pericyte", "mesenchymal stem cell of adipose tissue", "ltf+ epithelial cell"]    # For subset exons

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{name} dtypes:")
        print(df.dtypes)

    
    print(f"===== Training Data =====")
    sanity_checks(train_df, psi_cols)
    print()

    print(f"===== Validation Data =====")
    sanity_checks(val_df, psi_cols)
    print()

    print(f"===== Test Data =====")
    sanity_checks(test_df, psi_cols)
    print()

    output_dir = "/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generate_plots(train_df, val_df, test_df, output_dir)
    
if __name__ == "__main__":
    main()
