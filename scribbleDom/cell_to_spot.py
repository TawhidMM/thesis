import pandas as pd
import numpy as np
import os


def label_percentage_bins(df_cell_spot, df_cell_clusters, results_folder):
    df_merged = df_cell_spot.merge(
        df_cell_clusters,
        left_on="Object ID",
        right_on="Cell_ID",
        how="left"
    )

    bin_edges = np.arange(0, 110, 10)  # 0, 10, ..., 100

    # Count each label's frequency per spot
    spot_label_counts = df_merged.groupby(["assigned_spot", "Cluster_Label"]).size().reset_index(name="count")

    # Get total cells per spot
    spot_totals = df_merged.groupby("assigned_spot").size().reset_index(name="total")

    # Merge totals
    spot_label_counts = spot_label_counts.merge(spot_totals, on="assigned_spot", how="left")

    # Calculate percentage
    spot_label_counts["percentage"] = (spot_label_counts["count"] / spot_label_counts["total"]) * 100

    # Bin percentages
    spot_label_counts["bin"] = pd.cut(spot_label_counts["percentage"], bins=bin_edges, include_lowest=True)

    # Count per label per bin
    bin_counts = spot_label_counts.groupby(["Cluster_Label", "bin"], observed=False).size().reset_index(name="count")
    bin_counts = bin_counts.pivot(index="Cluster_Label", columns="bin", values="count").fillna(0).astype(int)

    # Save CSV
    bin_counts.to_csv(f"../input-processing/{results_folder}/spot_labels/label_stat.csv", index=True)


def assign_spot_labels(df_cell_spot, df_cell_clusters, results_folder, threshold=0.5):
    # Merge cell spot info with cluster labels
    df_merged = df_cell_spot.merge(
        df_cell_clusters,
        left_on="Object ID",
        right_on="Cell_ID",
        how="left"
    )

    # Majority voting per spot
    spot_label_info = []
    for spot, group in df_merged.groupby("assigned_spot"):
        counts = group["Cluster_Label"].dropna().value_counts()
        if counts.empty:
            spot_label_info.append({
                "spot_id": spot,
                "majority_label": None,
                "majority_share": 0,
                "total_cells": 0
            })
        else:
            majority_label = counts.idxmax()
            majority_count = counts.max()
            total_count = counts.sum()
            majority_share = majority_count / total_count

            if majority_share < threshold:
                majority_label = None

            spot_label_info.append({
                "spot_id": spot,
                "majority_label": majority_label,
                "majority_share": majority_share,
                "total_cells": total_count
            })

    df_spot_labels = pd.DataFrame(spot_label_info)

    # Save results
    output_path = f"../input-processing/{results_folder}/spot_labels/{threshold}"
    os.makedirs(output_path, exist_ok=True)

    # Overall bin stats
    df_spot_labels["majority_percent"] = (df_spot_labels["majority_share"] * 100).round(2)
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    df_spot_labels["bin"] = pd.cut(df_spot_labels["majority_percent"], bins=bins, labels=labels, include_lowest=True)

    # Save outputs
    df_spot_labels.to_csv(f"{output_path}/spot_labels_from_cell.csv", index=False)

    # Save value counts for majority labels
    label_counts = df_spot_labels["majority_label"].value_counts(dropna=False)
    label_counts.to_csv(f"{output_path}/majority_label_counts.csv")


# Example usage:
results_folder = 'raw'
cell_spot_path = '../morphology_preprocessing/morphology_with_spot.csv'
cell_cluster_path = f'../input-processing/{results_folder}/cell_mclust_result.csv'

cell_spot_df = pd.read_csv(cell_spot_path)
cell_cluster_df = pd.read_csv(cell_cluster_path)

thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for threshold in thresholds:
    print(f"Threshold: {threshold}")
    assign_spot_labels(cell_spot_df, cell_cluster_df, results_folder, threshold)


# assign_spot_labels(cell_spot_df, cell_cluster_df, results_folder, 0.5)
label_percentage_bins(cell_spot_df, cell_cluster_df, results_folder)
