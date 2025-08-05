import torch
from torch_scatter import scatter_add
from torch_geometric.nn import radius_graph
from torch_geometric.utils import add_self_loops
import pandas as pd


def radius_graph_torch(coords: torch.Tensor, radius: float, loop: bool = False):
    """
    PyTorch version of radius graph using torch_geometric.
    Returns: edge_index [2, num_edges]
    """
    edge_index = radius_graph(x=coords, r=radius, loop=loop)
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
    scribble_mask = df[label_col] != -1

    torch.save(torch.tensor(scribble_mask.to_numpy()), "preprocessed/scribble_mask.pt")

def extract_labels(df, label_col, class_map=None):
    labels = df[label_col].fillna(-1)
    if class_map is None:
        unique_classes = sorted(labels[labels != -1].unique())
        class_map = {cls: i for i, cls in enumerate(unique_classes)}
    mapped_labels = labels.map(class_map).fillna(-1).astype(int)

    mapped_labels = torch.tensor(mapped_labels.values)
    torch.save(mapped_labels, "preprocessed/labels.pt")

def create_feature_matrix(feature_csv_path):
    """
    Extracts features from the DataFrame based on specified columns.
    Returns: tensor of features
    """
    features = pd.read_csv(feature_csv_path)
    features = features.drop('Cell ID', axis=1)

    features = torch.tensor(features.values, dtype=torch.float32)
    torch.save(features, "preprocessed/features.pt")

# --- Main Function ---
def build_graph_from_csv(csv_path, radius=0.02, sigma=0.01):
    df = pd.read_csv(csv_path)
    print("number of nodes in graph construction: ", len(df))

    # Step 1: Extract coordinates and convert to tensor
    coords = torch.tensor(df[['centroid_x_px', 'centroid_y_px']].values, dtype=torch.float32)

    # Step 2: Build radius graph
    edge_index = radius_graph_torch(coords, radius)
    torch.save(edge_index, "preprocessed/edge_index.pt")

    # Step 3: Compute edge weights
    # edge_weight = weight_edges_torch(coords, edge_index, sigma=sigma)

    # Step 4: Add self-loops
    # edge_index, edge_weight = add_self_loops_torch(edge_index, edge_weight=None, fill_value=1.0, num_nodes=coords.size(0))

    # Step 5: Normalize adjacency
    # edge_weight = torch.ones(edge_index.size(1))
    # edge_index, edge_weight = normalize_adj_torch(edge_index, edge_weight, num_nodes=coords.size(0))

    # return coords, features, edge_index, edge_weight



if __name__ == "__main__":
    # Example usage
    cell_scribble = "/home/tawhid-mubashwir/Storage/morphlink/input/cell_level_scribble.csv"
    file_path = f"/home/tawhid-mubashwir/Storage/morphlink/morphology_preprocessing/morphology_with_spot.csv"
    morphology_pc_path = f"/home/tawhid-mubashwir/Storage/morphlink/input/morphology_pca_15.csv"

    spot_radius = 188.56998854645946 / 2.0

    build_graph_from_csv(file_path, spot_radius)
    print("Graph construction completed.")

    LABEL_COLUMN = 'scribble_label'

    df = pd.read_csv(cell_scribble)
    print("Number of nodes in label :", len(df))

    create_scribble_mask(df, LABEL_COLUMN)
    print("Scribble mask created.")

    extract_labels(df, LABEL_COLUMN)
    print("Labels extracted and saved.")

    create_feature_matrix(morphology_pc_path)
    print("Feature matrix created and saved.")