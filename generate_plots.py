import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logit

data_dir = "/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data"
spearman_df = pd.read_csv(os.path.join(data_dir, "TS_Spearman_file.tsv"), sep="\t")

cell_types = ["retina - microglia", "nampt neutrophil", "lung ciliated cell", "ltf+ epithelial cell"]
with open("/gpfs/commons/home/nkeung/tabula_sapiens/completed.json", "r") as f:
    raw = json.load(f)
all_cell_types = [s.replace("_", " ") for s in raw]
data_splits = ["train", "test"]

output_dir = "/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/figures"

for split in data_splits:
    print(f"\n===== Data Split: {split} =====\n")
    split_path = os.path.join(data_dir, f"{split}_cassette_exons_with_logit_mean_psi.csv")
    split_df = pd.read_csv(split_path)

    split_path = os.path.join(output_dir, f"{split}")
    os.makedirs(split_path, exist_ok=True)
    filtered = split_df[cell_types]
    psi_counts = filtered.apply(lambda col: col.dropna().replace("", np.nan).count())
    
    # plt.figure(figsize=(8,5))
    # plt.bar(psi_counts.index, psi_counts.values)
    # plt.xlabel("Cell Type")
    # plt.ylabel("Exon Count")
    # plt.title(f"Exon Counts for {split} data")
    # plt.savefig(os.path.join(split_path, "exon_counts.png"), dpi=300, bbox_inches="tight")
    # print(f"✅ Generated bar plot with the number of exons")

    if split == "test":
        # Exon Count vs Spearman Correlation
        x = spearman_df["n_valid_psi"]
        y = spearman_df["spearman_delta"]
        labels = spearman_df["cell_type"]
        plt.figure(figsize=(8,5))
        plt.scatter(x, y)

        # Least Squares Trend line
        m, b = np.polyfit(x, y, 1)
        x_sorted = np.sort(x)
        plt.plot(x_sorted, m*x_sorted + b, color="red", linewidth=2, label=f"y = {m:.3f}x + {b:.3f}", linestyle="--")

        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Label outliers
        for i, (x_val, y_val, label) in enumerate(zip(x, y, labels)):
            if y_val < lower_bound or y_val > upper_bound:
                plt.text(x_val, y_val, label, fontsize=8, ha='left', va='bottom')
            if label in ["lung ciliated cell", "ltf+ epithelial cell", "acinar cell of salivary gland", "fast muscle cell"]:
                plt.text(x_val, y_val, label, fontsize=8, ha='left', va='bottom', color="black")
            elif label in ["retina - microglia", "nampt neutrophil", "enteroglial cell", "natural killer cell"]:
                plt.text(x_val, y_val, label, fontsize=8, ha='left', va='bottom', color="red")

        plt.xlabel("Exon Count")
        plt.ylabel("Spearman Correlation for Delta Psi")
        plt.title(f"Spearman Correlation and Exon Count for {split} data")
        plt.savefig(os.path.join(split_path, "count_spearman.png"), dpi=300, bbox_inches="tight")
        print(f"✅ Generated Spearman and exon count graph")

        # Get Top 50 and Bottom 50 Spearman correlation cell types
        spearman_df.sort_values(by=["spearman_delta"], ascending=False, inplace=True)
        top_50 = spearman_df["cell_type"].head(50).tolist()
        bottom_50 = spearman_df["cell_type"].head(50).tolist()
        
        # Bin Delta Logit PSI Counts
        def compute_delta_logit_psi(psi_col):
            psi_col = pd.to_numeric(psi_col, errors="coerce")
            valid_entries = psi_col.dropna()
            eps = 1e-6
            computed_logit = logit(np.clip(valid_entries / 100, eps, 1-eps))
            return computed_logit - split_df.loc[valid_entries.index, "logit_mean_psi"]
        
        delta_logit_df = split_df[all_cell_types].apply(compute_delta_logit_psi, axis=0)

        logit_bins = [-np.inf, -5, 5, np.inf]
        bin_labels = [0, -1, 1]

        def get_logit_bin_counts(series):
            binned = pd.cut(series.dropna(), bins=logit_bins, labels=bin_labels, include_lowest=True)
            counts = binned.value_counts().reindex(bin_labels, fill_value=0)
            total = counts.sum()
            fraction = counts / total if total > 0 else counts
            result = pd.concat([counts.rename(lambda x: x),
                                fraction.rename(lambda x: f"{x}_norm")])
            return result

        logit_bin_stats_df = delta_logit_df.apply(get_logit_bin_counts, axis=0).T.reset_index()
        logit_bin_stats_df.rename(columns={"index": "cell_type"}, inplace=True)
        logit_bin_stats_df = pd.merge(
            logit_bin_stats_df,
            spearman_df[["cell_type", "spearman_delta"]],
            how="inner",
            on="cell_type"
        )
        logit_bin_stats_df.to_csv(os.path.join(split_path, "delta_logit_bin_counts.csv"), index=False)
        print(f"✅ Saved binned delta logit PSI as CSV")
        
        # Bin PSI and Save Counts
        bins = [0, 10, 90, 100]
        bin_labels = ["Low", "Mid", "High"]

        psi_vals = split_df[all_cell_types].apply(pd.to_numeric, errors="coerce")
        def get_psi_bin_counts(series):
            binned = pd.cut(series.dropna(), bins=bins, labels=bin_labels, include_lowest=True)
            counts = binned.value_counts().reindex(bin_labels, fill_value=0)
            total = counts.sum()
            fraction = counts / total if total > 0 else counts
            result = pd.concat([counts.rename(lambda x: f"{x}_count"),
                                fraction.rename(lambda x: f"{x}_norm")])
            return result
        
        psi_bin_stats_df = psi_vals.apply(get_psi_bin_counts, axis=0).T.reset_index()
        psi_bin_stats_df.rename(columns={"index": "cell_type"}, inplace=True)
        psi_bin_stats_df = pd.merge(
            psi_bin_stats_df,
            spearman_df[["cell_type", "spearman_delta"]],
            how="inner",
            on="cell_type"
        )
        psi_bin_stats_df.to_csv(os.path.join(split_path, "psi_bin_counts.csv"), index=False)
        print(f"✅ Saved binned PSI in CSV")


    # Generate Cell Type-Specific Plots
    for cell in cell_types:
        c_name = cell.replace(" ", "_")
        print(f"\tGenerating plots for {cell}")

        # Psi Distribution
        psi_values = split_df[cell].dropna()
        psi_values = psi_values[psi_values != ""]
        plt.figure(figsize=(8,5))
        plt.hist(psi_values, density=True, bins=50, alpha=0.5)
        plt.xlabel("Mean PSI (%)")
        plt.ylabel("Frequency (Out of 1)")
        plt.title(f"PSI Dist For {cell} Exons")
        plt.savefig(os.path.join(split_path, f"{c_name}_psi_dist.png"), dpi=300, bbox_inches="tight")
        print(f"\t\t✅ Generated PSI distribution histogram for {cell}")

        # Delta Logit Distribution
        cols = [cell, "logit_mean_psi"]
        logit_df = split_df[cols].dropna()
        logit_df = logit_df[(logit_df[cell] != "") & (logit_df["logit_mean_psi"] != "")]
        eps = 1e-6
        computed_logit = logit(np.clip(logit_df[cell] / 100, eps, 1-eps))
        delta_logit = computed_logit - logit_df["logit_mean_psi"]

        plt.figure(figsize=(8,5))
        plt.hist(delta_logit, density=True, bins=50, alpha=0.5)
        plt.xlabel("Delta Logit PSI")
        plt.ylabel("Frequency (Out of 1)")
        plt.title(f"Delta Logit PSI For {cell} Exons")
        plt.savefig(os.path.join(split_path, f"{c_name}_delta_logit_dist.png"), dpi=300, bbox_inches="tight")
        print(f"\t\t✅ Generated delta logit PSI histogram for {cell}")

        # Psi Bins
        counts, edges = np.histogram(psi_values, bins=[0, 10, 90, 100])
        proportions = counts / counts.sum()
        plt.figure(figsize=(8,5))
        plt.bar(["0-10", "10-90", "90-100"], proportions)
        plt.xlabel("PSI (%)")
        plt.ylabel("Frequency (Out of 1)")
        plt.title("PSI Distribution (0–10, 10–90, 90–100 bins)")
        plt.savefig(os.path.join(split_path, f"{c_name}_psi_bins_dist.png"), dpi=300, bbox_inches="tight")
        print(f"\t\t✅ Generated psi bin plot for {cell}")
        
        plt.close("all")
        