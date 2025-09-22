import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from argument_parser import *
from model import GraphClassifier
from new_model import GCNPipeline


def compute_loss(logits, spot_labels, labeled_mask):
    unlabeled_mask = ~labeled_mask

    # Supervised loss (only on labeled spots)
    supervised_loss = F.cross_entropy(
        logits[labeled_mask],
        spot_labels[labeled_mask]
    )

    target = torch.argmax(logits, dim=1)

    similarity_loss = F.cross_entropy(
        logits[unlabeled_mask],
        target[unlabeled_mask]
    )

    total_loss = alpha * similarity_loss + (1 - alpha) * supervised_loss
    return total_loss, supervised_loss.item(), similarity_loss.item()


def train(model, data, optimizer, cell_to_spot, num_spots, spot_labels, labeled_mask, device):
    model.train()
    optimizer.zero_grad()

    # Forward pass → spot-level logits
    out = model(
        data.morph_pcs.to(device),
        data.edge_index.to(device),
        cell_to_spot.to(device),
        num_spots
    )

    loss, sup_loss, unsup_loss = compute_loss(
        out,
        spot_labels.to(device),
        labeled_mask.to(device)
    )

    loss.backward()
    optimizer.step()
    return loss.item(), sup_loss, unsup_loss


def partial_scribble(labels, scribble_mask, label_fraction=0.2):
    """
    Randomly select a subset of labeled nodes to keep. The rest will be masked out.
    Returns a new scribble_mask and optionally sparse scribble_labels.
    """
    # Get indices of labeled nodes
    labeled_indices = torch.where(scribble_mask)[0]

    num_keep = int(label_fraction * len(labeled_indices))
    selected_indices = labeled_indices[torch.randperm(len(labeled_indices))[:num_keep]]

    # Create new mask
    new_mask = torch.zeros_like(scribble_mask, dtype=torch.bool)
    new_mask[selected_indices] = True

    updated_labels = labels.clone()
    updated_labels[~new_mask] = -1

    return updated_labels, new_mask



def scribble_dom_hyperparams():
    model_params = []
    for sample in samples:
        for seed in seed_options:
            for lr in lr_options:
                for alpha in alpha_options:
                    model_params.append(
                        {
                            'sample': sample,
                            'seed': seed,
                            'lr': lr,
                            'alpha': alpha
                        }
                    )

    return model_params


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # random.seed(12345)




device = torch.device('cpu')

if torch.cuda.is_available():
    print("GPU available")
    device = torch.device('cuda')



model_params = scribble_dom_hyperparams()
last_sample = ''

for parameter in tqdm(model_params):
    print(parameter)
    sample = parameter['sample']
    seed = parameter['seed']
    lr = parameter['lr']
    alpha = parameter['alpha']

    graph_representation_path = f"graph_representation/{dataset}/{sample}"

    morphology_pcs = torch.load(f"{graph_representation_path}/features.pt", weights_only=True)
    edge_index = torch.load(f"{graph_representation_path}/edge_index.pt", weights_only=True)
    labels = torch.load(f"{graph_representation_path}/scribble_labels.pt", weights_only=True)
    scribble_mask = torch.load(f"{graph_representation_path}/scribble_mask.pt", weights_only=True)
    cell_to_spot = torch.load(f"{graph_representation_path}/cell_to_spot.pt", weights_only=True)
    mask = torch.load(f"{graph_representation_path}/mask.pt", weights_only=True)

    # cell_ids = pd.read_csv(f"preprocessed/{dataset}/{sample}/cell_level_annotation.csv")
    cell_ids = pd.read_csv(
        f"/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/{dataset}/{sample}/manual_scribble.csv",
        index_col=0)

    cell_ids = cell_ids.sort_index().index
    cell_ids = cell_ids[mask.numpy()]

    # print("Num labeled spots:", scribble_mask.sum().item())
    # print("Num unlabeled spots:", (~scribble_mask).sum().item())
    # print("Unique scribble_labels in supervised set:", torch.unique(scribble_labels[scribble_mask]))
    # print("cell to spot shape:", torch.unique(cell_to_spot))

    print("************************************************")
    set_seed(seed)

    # Create a PyG Data object
    data = Data(morph_pcs=morphology_pcs, edge_index=edge_index)

    NUM_CLASSES = labels.max().item() + 1
    print(f'Number of classes: {NUM_CLASSES}')

    # morphology_encoder = GraphClassifier(in_dim=morph_pcs.size(1), hidden_dims=[64], num_classes=n_cluster)
    model = GCNPipeline(input_dim=morphology_pcs.size(1), gcn_hidden_dims=[64, 64], proj_dim=128, out_dim=NUM_CLASSES)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    current_labels = labels.clone()
    current_masks = scribble_mask.clone()

    for epoch in range(max_iter):
        train_loss, scr_loss, sim_loss = (
            train(model, data, optimizer, cell_to_spot, len(current_labels), current_labels, current_masks, device))

        if epoch % 100 == 0 or epoch == max_iter - 1:
            print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, '
                  f'Supervised Loss: {scr_loss:.4f}, Unsupervised Loss: {sim_loss:.4f}')

    # final prediction
    model.eval()

    with torch.no_grad():
        logits = model(
            data.morph_pcs.to(device), data.edge_index.to(device),
            cell_to_spot.to(device), len(current_labels)
        ).cpu()
        predictions = logits.argmax(dim=1).numpy()


    output_folder_path = f'./{output_data_path}/{dataset}/{sample}/morphology-new'
    leaf_output_folder_path = f'{output_folder_path}/{scheme}/Hyper_{alpha}'

    os.makedirs(leaf_output_folder_path, exist_ok=True)

    logits_df = pd.DataFrame(
        logits.numpy(),
        index=cell_ids,
        columns=[f"class_{i}" for i in range(logits.shape[1])]
    )
    logits_df.to_csv(f"{leaf_output_folder_path}/final_cell_logits.csv")

    predictions_df = pd.DataFrame({
        "cell_id": cell_ids,
        "predicted_label": predictions
    })

    predictions_df.to_csv(f"{leaf_output_folder_path}/final_cell_labels.csv", index=False)

    meta_data_df = pd.DataFrame(
        [{
            "dataset": dataset,
            "sample": sample,
            "npcs": n_pcs,
            "nconv": nConv,
            "seed": seed,
            "learning_rate": lr,
            "alpha": alpha
        }]
    )
