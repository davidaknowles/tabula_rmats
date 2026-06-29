"""
Removes all exons that appear in less than 10 cell types. Splits test, training, and validation sets by chromosome number
as described in MTSplice paper
"""

import pandas as pd
import numpy as np
import os

main_dir = '/gpfs/commons/home/nkeung/tabula_muris_data/psi_data/final_data/'
output_path = "/gpfs/commons/home/nkeung/tabula_muris_data/filtered_psi/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load full dataset
full_df = pd.read_csv(main_dir+"full_cassette_exons_with_mean_psi.csv")
# Only keep cassette exons
full_df = full_df[full_df["cassette_exon"] == "Yes"].copy()
n_total = len(full_df)
print(f"Total exons (unfiltered): {n_total}\n")

# Filter out exons expressed in only 10 cell types
all_cells = full_df.columns[12:-3].tolist()
full_df["Counts"] = full_df[all_cells].notna().sum(axis=1)

filtered_df = full_df[full_df["Counts"] >= 10].copy()
print(filtered_df)
print(f"\nExons after filtering: {len(filtered_df)}")
print(f"Fraction of exons remaining: {len(filtered_df) / n_total}\n")
# print(filtered_df.describe())
filtered_df.drop("Counts", axis=1).to_csv(os.path.join(output_path, "full_cassette_exons_with_mean_psi.csv"), index=False)


# Split by exons
# Construct new column with chromosome numbers as integers
filtered_df["chr_num"] = filtered_df["chromosome"].str.replace("chr", "")
filtered_df["chr_num"] = filtered_df["chr_num"].replace({"X": "23", "Y": "23"})
filtered_df["chr_num"] = pd.to_numeric(filtered_df["chr_num"], errors="coerce")

# === Training Set ===

print("\nSplitting training exons...")
train_mask = ((filtered_df["chr_num"].isin([4, 6, 8])) |
              ((filtered_df["chr_num"] >= 10) & (filtered_df["chr_num"] <= 23)))
train_df = filtered_df[train_mask].copy()
print("Chromosomes saved:")
print(train_df["chr_num"].unique())

train_df.drop(["chr_num", "Counts"], axis=1, inplace=True)
train_df.to_csv(os.path.join(output_path, "train_cassette_exons_with_logit_mean_psi.csv"), index=False)
train_df.drop(["mean_psi", "logit_mean_psi"], axis=1, inplace=True)
train_df.to_csv(os.path.join(output_path, "train_cassette_exons.csv"), index=False)

print(f"✅ Saved training split")
print(f"Training split: {len(train_df)}, ({len(train_df) / len(filtered_df)})\n")


# === Validation Set ===

print("Splitting validation exons...")
val_mask = ((filtered_df["chr_num"].isin([1, 7, 9])))
val_df = filtered_df[val_mask].copy()
print("Chromosomes saved:")
print(val_df["chr_num"].unique())

val_df.drop(["chr_num", "Counts"], axis=1, inplace=True)
val_df.to_csv(os.path.join(output_path, "val_cassette_exons_with_logit_mean_psi.csv"), index=False)
val_df.drop(["mean_psi", "logit_mean_psi"], axis=1, inplace=True)
val_df.to_csv(os.path.join(output_path, "val_cassette_exons.csv"), index=False)

print(f"✅ Saved validation split")
print(f"Validation split: {len(val_df)}, ({len(val_df) / len(filtered_df)})\n")


# === Test Set === 

print("Splitting test exons...")
test_mask = ~(train_mask | val_mask)
test_df = filtered_df[test_mask].copy()
print("Chromosomes saved:")
print(test_df["chr_num"].unique())

test_df.drop(["chr_num", "Counts"], axis=1, inplace=True)
test_df.to_csv(os.path.join(output_path, "test_cassette_exons_with_logit_mean_psi.csv"), index=False)
test_df.drop(["mean_psi", "logit_mean_psi"], axis=1, inplace=True)
test_df.to_csv(os.path.join(output_path, "test_cassette_exons.csv"), index=False)

print(f"✅ Saved test split")
print(f"Test split: {len(test_df)}, ({len(test_df) / len(filtered_df)})")
