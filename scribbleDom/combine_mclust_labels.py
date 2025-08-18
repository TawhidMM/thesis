import os

import pandas as pd




def combine_mclust_labels(folder_name):
    # Load files
    gene_mclust_labels_df = pd.read_csv("/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/mclust_result.csv")
    morphology_mclust_labels_df = pd.read_csv(f"{folder_name}/spot_labels_from_cell.csv")
    gene_mclust_backbone = pd.read_csv("/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/mclust_backbone.csv")

    gene_mclust_labels_df.columns = ['spot_id', 'label']

    # Merge keeping all spots from gene_mclust_labels_df
    merged = gene_mclust_labels_df.merge(morphology_mclust_labels_df, on="spot_id", how="left")

    final_labels = []
    for _, row in merged.iterrows():
        if row["label"] == row["majority_label"]:
            final_labels.append(row["majority_label"])
        else:
            final_labels.append(None)

    # Final DataFrame with only spot and label/null
    final_df = pd.DataFrame({
        "spot_id": merged["spot_id"],
        "label": final_labels
    })

    # Save to CSV
    final_df.to_csv(f"{folder_name}/combined_mclust.csv", index=False)

    label_counts = final_df["label"].value_counts()
    label_counts.to_csv(f"{folder_name}/combined_mclust_value_counts.csv")


def read_spot_label_csvs(folder_name):
    base_path = os.path.join(f"../input-processing/{folder_name}", "spot_labels")
    data = {}

    # loop through threshold subfolders
    for threshold in os.listdir(base_path):
        threshold_path = os.path.join(base_path, threshold)

        if os.path.isdir(threshold_path):
            combine_mclust_labels(threshold_path)


results_folder = 'raw'
read_spot_label_csvs(results_folder)



# df = pd.read_csv("/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/cancers/bcdc_ffpe/combined_mclust_backbone.csv")
# print(df.iloc[:, -1].value_counts())
#
#
# df = pd.read_csv("/home/tawhid-mubashwir/Storage/morphlink/input-processing/without_od/spot_labels_from_cell.csv")
# print(df.iloc[:, 1].value_counts())

"""
manual scribble
2.0    33
1.0    21
Name: count, dtype: int64

gene mclust label
1    1937
2     581
Name: count, dtype: int64

gene mclust backbone label
1.0    1345
2.0     135
Name: count, dtype: int64

---------------------------------------------------
"""
