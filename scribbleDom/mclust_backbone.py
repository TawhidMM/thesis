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
    cluster_labels = mclust_result.iloc[:, -1]  # assumes last column is cluster
    cluster_map = dict(cluster_labels)  # barcode -> cluster
    cluster_array = np.array([cluster_map[name] for name in node_names])  # index aligned

    num_nodes = len(node_names)
    refined_clusters = np.full(num_nodes, None)  # initialize with None

    # Build adjacency list from edge_index
    adjacency = defaultdict(list)
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adjacency[s].append(d)
        adjacency[d].append(s)  # undirected

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
            refined_clusters[i] = my_cluster  # keep it

    # Save result
    df_backbone = pd.DataFrame({
        "Cell ID": node_names,
        "mclust_label": refined_clusters
    })
    df_backbone.set_index("Cell ID", inplace=True)
    df_backbone.to_csv(output_path)
    print(f"Refined clusters saved to {output_path}")


mclust_result=pd.read_csv("../input/cell_mclust_result.csv", index_col=0)
# node_names = mclust_result_df.index
# edge_index = torch.load("preprocessed/edge_index.power_transformation", weights_only=True).cpu()
#
# make_graph_backbone(
#     mclust_result_df=mclust_result_df,
#     edge_index=edge_index,
#     node_names=node_names,
#     output_path="../input/cell_mclust_backbone.csv"
# )

mclust_backbone_result=pd.read_csv("../input/cell_mclust_backbone.csv", index_col=0)

print(mclust_backbone_result.iloc[:,-1].value_counts())
print(mclust_result.iloc[:,-1].value_counts())
