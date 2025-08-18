import pandas as pd
import numpy as np
import torch
from collections import defaultdict

def make_graph_backbone(mclust_result, edge_index, node_names, threshold=1.0, output_path='./mclust_backbone.csv'):
    """
    Refines mclust cluster assignment using graph neighborhood agreement.

    Args:
        mclust_result (pd.DataFrame): must have node names as index, and last column is the cluster
        edge_index (torch.LongTensor): shape [2, num_edges], PyG format
        node_names (list): list of node names (barcodes), matching PyG graph node index
        threshold (float): min % of neighbors that must match to keep assignment
        output_path (str): CSV file to save refined clusters
    """

    # Get cluster assignments as list aligned with node indices
    cluster_labels = mclust_result.iloc[:, -1]
    cluster_map = dict(cluster_labels)
    cluster_array = np.array([cluster_map[name] for name in node_names])

    num_nodes = len(node_names)
    refined_clusters = np.full(num_nodes, None)

    # Build adjacency list from edge_index
    adjacency = defaultdict(list)
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adjacency[s].append(d)
        adjacency[d].append(s)

    # Loop through all nodes
    for i in range(num_nodes):
        my_cluster = cluster_array[i]
        neighbors = adjacency[i]

        if len(neighbors) == 0:
            continue

        neighbor_clusters = cluster_array[neighbors]
        match_count = np.sum(neighbor_clusters == my_cluster)
        match_ratio = match_count / len(neighbors)

        if match_ratio >= threshold:
            refined_clusters[i] = my_cluster

    # Save result
    df_backbone = pd.DataFrame({
        "Cell ID": node_names,
        "mclust_label": refined_clusters
    })
    df_backbone.set_index("Cell ID", inplace=True)
    df_backbone.to_csv(output_path)
    print(f"Refined clusters saved to {output_path}")


folder_name = 'yeo-johnson'

mclust_result_path = f"../input-processing/{folder_name}/cell_mclust_result.csv"
mclust_backbone_path = f"../input-processing/{folder_name}/mclust_backbone.csv"


mclust_result_df = pd.read_csv(mclust_result_path, index_col=0)
node_names = mclust_result_df.index
edge_index = torch.load("preprocessed/edge_index.pt", weights_only=True).cpu()

make_graph_backbone(
    mclust_result=mclust_result_df,
    edge_index=edge_index,
    node_names=node_names,
    output_path=mclust_backbone_path
)

mclust_backbone_result=pd.read_csv(mclust_backbone_path, index_col=0)

counts_backbone = mclust_backbone_result.iloc[:, -1].value_counts()
counts_mclust   = mclust_result_df.iloc[:, -1].value_counts()

print(counts_backbone)
print(counts_mclust)

counts_df = pd.DataFrame({
    "backbone_counts": counts_backbone,
    "mclust_counts": counts_mclust
}).fillna(-1).astype(int)

# Save to CSV
counts_df.to_csv(f"../input-processing/{folder_name}/mclust_counts.csv")

print(f"Saved counts")