import os

import pandas as pd
import torch
from torch_geometric.nn import radius_graph

from argument_parser import *


def radius_graph_torch(coords: torch.Tensor, radius: float):
    """
    PyTorch version of radius graph using torch_geometric.
    Returns: edge_index [2, num_edges]
    """
    edge_index = radius_graph(x=coords, r=radius, max_num_neighbors=1000)
    return edge_index


def create_cell_to_spot_mapping(cell_df, spot_df):
    spot_id_map = {sid: i for i, sid in enumerate(spot_df.iloc[:, 0].unique())}

    print("Number of unique spots in cell: ", len(cell_df['assigned_spot'].unique()))
    print("Number of unique spots in scribble: ", len(spot_df.iloc[:, 0].unique()))

    spot_col_name = 'assigned_spot'
    cell_to_spot = torch.tensor(
        [spot_id_map[sid] for sid in cell_df[spot_col_name].values],
        dtype=torch.long
    )
    torch.save(cell_to_spot, f"{saving_path}/cell_to_spot.pt")


def create_spot_scribble_mask(spot_scribble_df):
    label_col_name = spot_scribble_df.columns[-1]

    spot_scribble_df[label_col_name] = spot_scribble_df[label_col_name].fillna(-1)
    scribble_mask = spot_scribble_df[label_col_name] != -1

    torch.save(torch.tensor(scribble_mask.to_numpy()), f"{saving_path}/scribble_mask.pt")


def extract_spot_labels(spot_scribble_df):
    label_col_name = spot_scribble_df.columns[-1]
    labels = spot_scribble_df[label_col_name].fillna(-1)

    unique_classes = sorted(labels[labels != -1].unique())
    class_map = {cls: i for i, cls in enumerate(unique_classes)}
    mapped_labels = labels.map(class_map).fillna(-1).astype(int)

    mapped_labels = torch.tensor(mapped_labels.values)
    torch.save(mapped_labels, f"{saving_path}/scribble_labels.pt")


def create_cell_scribble_mask(df, label_col):
    df[label_col] = df[label_col].fillna(-1)
    scribble_mask = df[label_col] != -1

    torch.save(torch.tensor(scribble_mask.to_numpy()), f"{saving_path}/scribble_mask.pt")


def extract_cell_labels(df, label_col, class_map=None):
    labels = df[label_col].fillna(-1)
    if class_map is None:
        unique_classes = sorted(labels[labels != -1].unique())
        class_map = {cls: i for i, cls in enumerate(unique_classes)}
    mapped_labels = labels.map(class_map).fillna(-1).astype(int)

    mapped_labels = torch.tensor(mapped_labels.values)
    torch.save(mapped_labels, f"{saving_path}/scribble_labels.pt")


def create_feature_matrix(feature_csv_path):
    """
    Extracts features from the DataFrame based on specified columns.
    Returns: tensor of features
    """
    features = pd.read_csv(feature_csv_path)
    features = features.drop('Cell ID', axis=1)

    features = torch.tensor(features.values, dtype=torch.float32)
    torch.save(features, f"{saving_path}/features.pt")



def build_graph_from_csv(csv_path, radius=0.02):
    df = pd.read_csv(csv_path)
    print("number of nodes in graph construction: ", len(df))

    # Step 1: Extract coordinates and convert to tensor
    coords = torch.tensor(df.iloc[:, -2:].values, dtype=torch.float32)

    # Step 2: Build radius graph
    edge_index = radius_graph_torch(coords, radius)
    torch.save(edge_index, f"{saving_path}/edge_index.pt")

# --- Main Function ---
if __name__ == "__main__":
    saving_path = f"graph_representation/{dataset}/{samples[0]}"
    os.makedirs(saving_path, exist_ok=True)

    if scheme == 'expert':
        cell_scribble = f"preprocessed/{dataset}/{samples[0]}/cell_level_scribble.csv"
        spot_scribble = f"/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/{dataset}/{samples[0]}/manual_scribble.csv"
    elif scheme == 'mclust':
        cell_scribble = "../input/cell_mclust_backbone.csv"
        spot_scribble = f"/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/{dataset}/{samples[0]}/mclust_backbone.csv"
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Supported schemes are 'expert' and 'mclust'.")

    coords_path = f"preprocessed/{dataset}/{samples[0]}/cell_coords.csv"
    morphology_with_spots_path = f"preprocessed/{dataset}/{samples[0]}/morphology_with_spot.csv"
    morphology_pc_path = f"/home/tawhid-mubashwir/Storage/morphlink/input-processing/{dataset}/{samples[0]}/yeo-johnson/morphology_pca.csv"

    graph_radius = 96.40082438014726 / 2.0

    build_graph_from_csv(coords_path, graph_radius)
    print("Graph construction completed.")

    scribble_df = pd.read_csv(spot_scribble)
    scribble_df = scribble_df.sort_values(by=scribble_df.columns[0])
    cell_with_spot_df = pd.read_csv(morphology_with_spots_path)

    assigned_spots = cell_with_spot_df['assigned_spot'].values
    mask = scribble_df.iloc[:, 0].isin(assigned_spots)
    torch.save(torch.tensor(mask.to_numpy()), f"{saving_path}/mask.pt")

    scribble_df = scribble_df[mask].reset_index(drop=True)
    print("Number of nodes in label :", len(scribble_df))

    create_cell_to_spot_mapping(cell_with_spot_df, scribble_df)
    print("Cell to spot mapping created.")

    create_spot_scribble_mask(scribble_df)
    print("Scribble mask created.")

    extract_spot_labels(scribble_df)
    print("Labels extracted and saved.")

    create_feature_matrix(morphology_pc_path)
    print("Feature matrix created and saved.")