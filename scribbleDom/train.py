import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from argument_parser import *
from model import GraphClassifier


def compute_loss(logits, labels, scribble_mask, alpha=0.5):
    """
    Compute the combined supervised and unsupervised (cross-entropy based) loss.
    Args:
        logits: (N, C) output from the model
        labels: (N,) ground-truth labels (with -1 for unlabeled)
        scribble_mask: (N,) boolean tensor indicating labeled nodes
        alpha: weight for unsupervised loss
    """
    non_scribble_mask = ~scribble_mask

    # supervised loss (L_scr)
    supervised_loss = F.cross_entropy(logits[scribble_mask], labels[scribble_mask])

    # Unsupervised similarity loss (L_sim)
    target = torch.argmax(logits, dim=1)

    similarity_loss = F.cross_entropy(logits[non_scribble_mask], target[non_scribble_mask])

    return alpha * similarity_loss + (1 - alpha) * supervised_loss, supervised_loss.item(), similarity_loss.item()



def train(model, data, optimizer, scribble_mask, labels, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))  # logits
    loss, sup_loss, unsup_loss = compute_loss(out, labels.to(device), scribble_mask.to(device))
    loss.backward()
    optimizer.step()
    return loss.item(), sup_loss, unsup_loss



def partial_scribble(labels, scribble_mask, label_fraction=0.2):
    """
    Randomly select a subset of labeled nodes to keep. The rest will be masked out.
    Returns a new scribble_mask and optionally sparse labels.
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
                            'seed': seed,
                            'lr': lr,
                            'sample': sample,
                            'alpha': alpha
                        }
                    )

    return model_params


def auto_scribble_dom_hyperparams():
    model_params = scribble_dom_hyperparams()
    new_params = []

    for beta in beta_options:
        for params in model_params:
            param_copy = params.copy()
            param_copy['beta'] = beta
            new_params.append(param_copy)

    return new_params




torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cpu')

if torch.cuda.is_available():
    print("GPU available")
    device = torch.device('cuda')

# Load saved components
print("loading preprocessed data......")

x = torch.load("preprocessed/features.pt", weights_only=True)
edge_index = torch.load("preprocessed/edge_index.pt", weights_only=True)
labels = torch.load("preprocessed/labels.pt", weights_only=True)
scribble_mask = torch.load("preprocessed/scribble_mask.pt", weights_only=True)
cell_ids = pd.read_csv("/home/tawhid-mubashwir/Storage/morphlink/input/cell_level_annotation.csv")
cell_ids = cell_ids['Cell ID'].values


if scheme == 'expert':
    model_params = scribble_dom_hyperparams()
elif scheme == 'mclust':
    model_params = auto_scribble_dom_hyperparams()
else:
    raise ValueError(f"Unknown scheme: {scheme}. Supported schemes are 'expert' and 'mclust'.")


for parameter in tqdm(model_params):
    print(parameter)
    seed = parameter['seed']
    lr = parameter['lr']
    sample = parameter['sample']
    alpha = parameter['alpha']

    if scheme == 'mclust':
        beta = parameter['beta']

    print("************************************************")
    print('Model description:')
    print(f'sample: {sample}')
    print(f'seed: {seed}')
    print(f'lr: {lr}')
    print(f'alpha: {alpha}')

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index)

    NUM_CLASSES = labels.max().item() + 1
    print(f'Number of classes: {NUM_CLASSES}')
    # why hidden layer is a list ?????
    model = GraphClassifier(in_dim=x.size(1), hidden_dims=[64], num_classes=n_cluster)
    model = model.to(device)
    # why not SDG ??? main pipeline uses SDG
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    current_labels = labels.clone()
    current_masks = scribble_mask.clone()

    if scheme == 'mclust':
        current_labels, current_masks = partial_scribble(current_labels, current_masks, beta)

    for epoch in range(max_iter):
        train_loss, scr_loss, sim_loss = train(model, data, optimizer, current_masks, current_labels, device)

        if epoch % 100 == 0 or epoch == max_iter - 1:
            print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, '
                  f'Supervised Loss: {scr_loss:.4f}, Unsupervised Loss: {sim_loss:.4f}')

    # final prediction
    model.eval()

    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()


    output_folder_path = f'./{output_data_path}/{dataset}/{sample}/morphology'
    leaf_output_folder_path = f'{output_folder_path}/{scheme}/Hyper_{alpha}'
    if scheme == 'mclust':
        leaf_output_folder_path += f'_{beta}'
    os.makedirs(leaf_output_folder_path, exist_ok=True)

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
    if scheme == 'mclust':
        meta_data_df['beta'] = beta
    meta_data_df.to_csv(f'{leaf_output_folder_path}/meta_data.csv')
