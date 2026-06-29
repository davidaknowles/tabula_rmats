import pandas as pd
import numpy as np
import json

main_dir = '/gpfs/commons/home/nkeung/tabula_muris_data/psi_data/final_data/'
RANDOM_SEED = 124
rng = np.random.default_rng(RANDOM_SEED)

# Load full dataset
full_df = pd.read_csv(main_dir+"full_cassette_exons_with_mean_psi.csv")
# Only keep cassette exons
full_df = full_df[full_df["cassette_exon"] == "Yes"].copy()

n_total = len(full_df)
n_test = int(0.15 * n_total)
n_val = int(0.15 * n_total)
n_train = n_total - n_test - n_val

# TEST SET
allowed_list = pd.read_csv(main_dir+"full_cassette_exons_with_mean_psi_NO_MULTIZ_OVERLAPS.csv")
allowed_list = allowed_list.drop_duplicates(subset=["ascot_exon_id"])

# Filtering Tabula Muris data
# Theta join full_df and allowed_list where full_df["exon_id"] == allowed_list["ascot_exon_id"]
safe_df = full_df.merge(
    allowed_list[["ascot_exon_id", "Exon Name"]],
    left_on="exon_id",
    right_on="ascot_exon_id",
    how="inner"
).drop(columns=["ascot_exon_id"])
test_df = safe_df.sample(n=n_test, random_state=rng)

remaining_df = full_df[~ full_df["exon_id"].isin(test_df["exon_id"])].reset_index(drop=True)

# TRAINING AND VALIDATION SET
train_df = remaining_df.sample(n=n_train, random_state=rng)
val_df = remaining_df[~ remaining_df["exon_id"].isin(train_df["exon_id"])].reset_index(drop=True).sample(n=n_val, random_state=rng)


# Sanity Checks
used_exons = pd.read_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/train_exon_list.csv")
leaked_data = test_df[test_df["Exon Name"].isin(used_exons["exon_id"])]
assert len(leaked_data) == 0
test_df = test_df.drop("Exon Name", axis="columns")
print(f"✅ No exons used in pre-training found in test set. No data leakage\n")

assert len(set(train_df["exon_id"]) & set(val_df["exon_id"])) == 0
assert len(set(train_df["exon_id"]) & set(test_df["exon_id"])) == 0
assert len(set(val_df["exon_id"]) & set(test_df["exon_id"])) == 0
print("✅ Splits complete and non-overlapping!")
print(f"Train: {len(train_df)} ({100* len(train_df)/n_total}%), Val: {len(val_df)} ({100* len(val_df)/n_total}%), Test: {len(test_df)} ({100* len(test_df)/n_total}%)")

# Save splits
train_df.to_csv(main_dir+"train_cassette_exons_with_logit_mean_psi.csv", index=False)
val_df.to_csv(main_dir+"val_cassette_exons_with_logit_mean_psi.csv", index=False)
test_df.to_csv(main_dir+"test_cassette_exons_with_logit_mean_psi.csv", index=False)

# Remove mean PSI and logit mean PSI for later files
train_df = train_df.drop(columns=["mean_psi", "logit_mean_psi"])
val_df = val_df.drop(columns=["mean_psi", "logit_mean_psi"])
test_df = test_df.drop(columns=["mean_psi", "logit_mean_psi"])

# Set NAN to -1
with open("/gpfs/commons/home/nkeung/tabula_muris_data/rmats/completed.json", "r") as f:
    cell_types = list(json.load(f))

cols = [s.replace("_", " ") for s in cell_types]
train_df[cols] = train_df[cols].replace("", np.nan).fillna(-1)
val_df[cols] = val_df[cols].replace("", np.nan).fillna(-1)
test_df[cols] = test_df[cols].replace("", np.nan).fillna(-1)

train_df.to_csv(main_dir+"train_cassette_exons.csv", index=False)
val_df.to_csv(main_dir+"val_cassette_exons.csv", index=False)
test_df.to_csv(main_dir+"test_cassette_exons.csv", index=False)