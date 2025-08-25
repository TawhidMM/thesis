import os

import pandas as pd
import torch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from argument_parser import *


def radius_graph_torch(coords: torch.Tensor, radius: float):
    """
    PyTorch version of radius graph using torch_geometric.
    Returns: edge_index [2, num_edges]
    """
    edge_index = radius_graph(x=coords, r=radius, max_num_neighbors=1000)
    return edge_index


def weight_edges_torch(coords: torch.Tensor, edge_index: torch.Tensor, sigma: float = 1.0):
    """
    Compute Gaussian weights from Euclidean distances.
    Returns: edge_weights [num_edges]
    """
    row, col = edge_index
    diffs = coords[row] - coords[col]
    dists = torch.norm(diffs, dim=1)
    weights = torch.exp(-(dists ** 2) / (2 * sigma ** 2))
    return weights


def add_self_loops_torch(edge_index, edge_weight=None, fill_value=1.0, num_nodes=None):
    """
    Adds self loops and optionally a fixed weight for each.
    """
    if edge_weight is None:
        return add_self_loops(edge_index, fill_value=fill_value, num_nodes=num_nodes)
    else:
        return add_self_loops(edge_index, edge_weight=edge_weight, fill_value=fill_value, num_nodes=num_nodes)

def normalize_adj_torch(edge_index, edge_weight, num_nodes):
    """
    Symmetric GCN normalization for PyTorch geometric.
    """
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm_weight


def create_scribble_mask(df, label_col):
    df[label_col] = df[label_col].fillna(-1)
    scribble_mask = df[label_col] != -1

    torch.save(torch.tensor(scribble_mask.to_numpy()), f"{saving_path}/scribble_mask.pt")

def extract_labels(df, label_col, class_map=None):
    labels = df[label_col].fillna(-1)
    if class_map is None:
        unique_classes = sorted(labels[labels != -1].unique())
        class_map = {cls: i for i, cls in enumerate(unique_classes)}
    mapped_labels = labels.map(class_map).fillna(-1).astype(int)

    mapped_labels = torch.tensor(mapped_labels.values)
    torch.save(mapped_labels, f"{saving_path}/labels.pt")

def create_feature_matrix(feature_csv_path):
    """
    Extracts features from the DataFrame based on specified columns.
    Returns: tensor of features
    """
    features = pd.read_csv(feature_csv_path)
    features = features.drop('Cell ID', axis=1)

    features = torch.tensor(features.values, dtype=torch.float32)
    torch.save(features, f"{saving_path}/features.pt")

# --- Main Function ---
def build_graph_from_csv(csv_path, radius=0.02):
    df = pd.read_csv(csv_path)
    print("number of nodes in graph construction: ", len(df))

    # Step 1: Extract coordinates and convert to tensor
    coords = torch.tensor(df.iloc[:, -2:].values, dtype=torch.float32)

    # Step 2: Build radius graph
    edge_index = radius_graph_torch(coords, radius)
    torch.save(edge_index, f"{saving_path}/edge_index.pt")


if __name__ == "__main__":
    saving_path = f"graph_representation/{dataset}/{samples[0]}"
    os.makedirs(saving_path, exist_ok=True)

    if scheme == 'expert':
        cell_scribble = f"preprocessed/{dataset}/{samples[0]}/cell_level_scribble.csv"
    elif scheme == 'mclust':
        cell_scribble = "../input/cell_mclust_backbone.csv"
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Supported schemes are 'expert' and 'mclust'.")

    coords_path = f"preprocessed/{dataset}/{samples[0]}/cell_coords.csv"
    morphology_pc_path = f"/home/tawhid-mubashwir/Storage/morphlink/input-processing/{dataset}/{samples[0]}/yeo-johnson/morphology_pca.csv"

    graph_radius = 96.40082438014726 / 2.0

    build_graph_from_csv(coords_path, graph_radius)
    print("Graph construction completed.")

    df = pd.read_csv(cell_scribble)
    print("Number of nodes in label :", len(df))

    create_scribble_mask(df, df.columns[-1])
    print("Scribble mask created.")

    extract_labels(df, df.columns[-1])
    print("Labels extracted and saved.")

    create_feature_matrix(morphology_pc_path)
    print("Feature matrix created and saved.")