import glob
import os
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal

from argument_parser import *


def build_neighbors_from_edge_index(edge_index, num_nodes, pad=True, pad_value=-1):
    edge_index = edge_index.cpu().numpy()

    # Build undirected graph
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.T)

    # Collect neighbors
    neighbors = [list(graph.neighbors(i)) for i in range(num_nodes)]

    if pad:
        max_k = max(len(nbrs) for nbrs in neighbors)
        neighbor_array = np.full((num_nodes, max_k), pad_value, dtype=int)
        for i, nbrs in enumerate(neighbors):
            neighbor_array[i, :len(nbrs)] = nbrs
        return neighbor_array
    else:
        return neighbors  # return as list-of-lists


def compute_label_probability(labels, neighbors, gamma=3):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    num_nodes = len(labels)
    num_classes = int(labels.max() + 1)

    counts = np.zeros((num_nodes, num_classes), dtype=np.float32)

    # Count labels among neighbors
    for i in range(num_nodes):
        for nbr in neighbors[i]:
            lbl = labels[nbr]
            counts[i, lbl] += 1

    # Softmax with temperature scaling
    exp_scores = np.exp(gamma * counts / (counts.sum(axis=1, keepdims=True) + 1e-8))
    probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-8)

    # Return probability for the ground-truth class
    return probs[np.arange(num_nodes), labels]



def compute_p_y_given_D(features, labels):
    features = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    ll = 0.0
    variances = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2: continue
        x_c = features[idx]
        mu = x_c.mean(axis=0)
        cov = np.cov(x_c, rowvar=False, bias=True) + np.eye(x_c.shape[1]) * 1e-5

        ll += multivariate_normal.logpdf(x_c, mean=mu, cov=cov).sum()
        variances[c] = np.trace(cov)

    return ll, variances


def evaluate_graph_run(features, labels, edge_index, gamma=3):
    num_nodes = features.shape[0]
    neighbors = build_neighbors_from_edge_index(edge_index, num_nodes)
    p_D = compute_label_probability(labels, neighbors, gamma)
    p_y_given_D_ll, var_dict = compute_p_y_given_D(features, labels)

    joint_ll = np.log(p_D + 1e-9).sum() + p_y_given_D_ll
    return {
        "log_likelihood_joint": joint_ll,
        "log_likelihood_conditional": p_y_given_D_ll,
        "variance": sum(var_dict.values())
    }


def pick_best_graph_run(base_folder):
    output_directories = sorted(glob.glob(os.path.join(base_folder, "*")))

    results = []

    features = torch.load(f"graph_representation/{dataset}/{sample}/features.pt", weights_only=True)
    edge_index = torch.load(f"graph_representation/{dataset}/{sample}/edge_index.pt", weights_only=True)

    for output_dir in output_directories:
        print(f"Evaluating {output_dir}...")

        try:
            labels = pd.read_csv(f"{output_dir}/final_cell_labels.csv", index_col=0)
            labels = labels['predicted_label'].values
            labels = torch.tensor(labels, dtype=torch.int8)

            metrics = evaluate_graph_run(features, labels, edge_index)
            metrics["output_dir"] = output_dir
            results.append(metrics)
        except Exception as e:
            print(f"Skipping {output_dir}: {e}")

    df = pd.DataFrame(results)
    df["score"] = df["log_likelihood_joint"] - df["variance"] * 100
    best_idx = df["score"].idxmax()

    return df.iloc[best_idx]["output_dir"]



for sample in samples:

    model_output_directory = f"{output_data_path}/{dataset}/{sample}/morphology/{scheme}"

    best_model_path = pick_best_graph_run(model_output_directory)


    src_final_meta = f"{best_model_path}/meta_data.csv"
    src_final_label = f"{best_model_path}/final_cell_labels.csv"

    final_output_dir = f"{final_output_folder}/{dataset}/{sample}/morphology/{scheme}"
    os.makedirs(final_output_dir, exist_ok=True)

    dest_final_meta = f"{final_output_dir}/meta_data.csv"
    dest_final_label = f"{final_output_dir}/final_cell_labels.csv"

    shutil.copyfile(src_final_meta,dest_final_meta)
    shutil.copyfile(src_final_label,dest_final_label)