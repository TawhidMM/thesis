import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from argument_parser import *

# ----------------------------
# Meta-learner model
# ----------------------------
class MetaLearner(nn.Module):
    def __init__(self, num_classes, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(num_classes * 2, hidden_dim)  # concat gene+morph logits
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, gene_logits, morph_logits):
        x = torch.cat([gene_logits, morph_logits], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)  # [N, num_classes]


# ----------------------------
# Loss functions
# ----------------------------
def meta_loss(meta_logits, gene_logits, morph_logits, scribble_mask, scribbles):

    non_scribble_mask = ~scribble_mask
    # 1. Supervised CE loss
    sup_loss = F.cross_entropy(meta_logits[scribble_mask], scribbles[scribble_mask])

    # 2. KL divergence w.r.t ensemble of base pipelines
    p_gene = F.softmax(gene_logits[non_scribble_mask], dim=-1)
    p_morph = F.softmax(morph_logits[non_scribble_mask], dim=-1)
    p_ens = 0.5 * (p_gene + p_morph)

    p_meta = F.log_softmax(meta_logits[non_scribble_mask], dim=-1)
    kl_loss = F.kl_div(p_meta, p_ens, reduction='batchmean')

    # 3. Confidence regularization (negative entropy)
    p_hat = F.softmax(meta_logits[non_scribble_mask], dim=-1)
    conf_loss = -(p_hat * p_hat.log()).sum(dim=1).mean()

    total_loss = sup_loss + lambda_kl * kl_loss + lambda_conf * conf_loss
    return total_loss, {"sup": sup_loss.item(), "kl": kl_loss.item(), "conf": conf_loss.item()}


# ----------------------------
# Example training loop
# ----------------------------
def train_meta_learner(gene_logits, morph_logits, scribble_mask, scribble_labels, num_classes,
                       device, val_split=0.25, epochs=400, lr=1e-2):

    # Split train/val
    idx = np.arange(len(scribble_labels))
    train_idx, val_idx = train_test_split(idx, test_size=val_split, stratify=scribble_labels)

    gene_train, morph_train = gene_logits[train_idx], morph_logits[train_idx]
    mask_train, labels_train = scribble_mask[train_idx], scribble_labels[train_idx]

    gene_val, morph_val = gene_logits[val_idx], morph_logits[val_idx]
    mask_val, labels_val = scribble_mask[val_idx], scribble_labels[val_idx]

    # Model
    model = MetaLearner(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(gene_train.to(device), morph_train.to(device))
        loss, comps = meta_loss(out, gene_train.to(device), morph_train.to(device),
                                mask_train.to(device), labels_train.to(device))
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(gene_val.to(device), morph_val.to(device))
            val_loss, _ = meta_loss(val_out, gene_val.to(device), morph_val.to(device),
                                    mask_val.to(device), labels_val.to(device))
            val_acc = (val_out.argmax(1).cpu() == labels_val).float().mean().item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss {loss.item():.4f} "
                  f"(sup={comps['sup']:.3f}, kl={comps['kl']:.3f}, conf={comps['conf']:.3f}) | "
                  f"Val Loss {val_loss.item():.4f} | Val Acc {val_acc:.3f}")

    return model


morphology_logits = pd.read_csv(f'model_outputs/{dataset}/{samples[0]}/morphology/{scheme}/Hyper_0.6/final_cell_logits.csv', index_col=0)
gene_expression_logits = pd.read_csv(f'/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/final_outputs/{dataset}/{samples[0]}/{scheme}/final_barcode_logits.csv', index_col=0)
spot_scribble_df = pd.read_csv(f"/mnt/Drive E/Class Notes/L-4 T-1/Thesis/ScribbleDom/preprocessed_data/{dataset}/{samples[0]}/manual_scribble.csv", index_col=0)

# Align indices
common_indices = morphology_logits.index.intersection(gene_expression_logits.index)
morphology_logits = morphology_logits.loc[common_indices]
gene_expression_logits = gene_expression_logits.loc[common_indices]
spot_scribble_df = spot_scribble_df.loc[common_indices]

morphology_logits.sort_index(inplace=True)
gene_expression_logits.sort_index(inplace=True)
spot_scribble_df.sort_index(inplace=True)

label_col_name = spot_scribble_df.columns[-1]

spot_scribble_df[label_col_name] = spot_scribble_df[label_col_name].fillna(-1)
scribble_mask = spot_scribble_df[label_col_name] != -1

labels = spot_scribble_df[label_col_name]

unique_classes = sorted(labels[labels != -1].unique())
class_map = {cls: i for i, cls in enumerate(unique_classes)}
mapped_labels = labels.map(class_map).fillna(-1).astype(int)

morphology_logits = torch.tensor(morphology_logits.values, dtype=torch.float32)
gene_expression_logits = torch.tensor(gene_expression_logits.values, dtype=torch.float32)
scribble_mask = torch.tensor(scribble_mask.values)
mapped_labels = torch.tensor(mapped_labels.values)

num_classes = n_cluster

lambda_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
mu_list     = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_params = []
for lambda_kl in lambda_list:
    for lambda_conf in mu_list:
        for seed in seed_options:
            model_params.append(
                {
                    "lamda_kl": lambda_kl,
                    "lamda_conf": lambda_conf,
                    "seed": seed
                }
            )


for params in tqdm(model_params):

    lambda_kl = params["lamda_kl"]
    lambda_conf = params["lamda_conf"]
    seed = params["seed"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    meta_learner_model = (
        train_meta_learner(gene_expression_logits, morphology_logits,
               scribble_mask, mapped_labels, num_classes, device))

    # final results
    meta_learner_model.eval()
    with torch.no_grad():
        final_logits = meta_learner_model(gene_expression_logits.to(device), morphology_logits.to(device))
        final_predictions = final_logits.argmax(1).cpu().numpy()
        final_df = pd.DataFrame({
            'Cell ID': common_indices,
            'Predicted Label': final_predictions
        })

        output_folder = f"{output_data_path}/{dataset}/{samples[0]}/meta_learner/{scheme}/kl_{lambda_kl}_cof_{lambda_conf}"
        os.makedirs(output_folder, exist_ok=True)

        final_df.to_csv(f'{output_folder}/final_meta_learner_predictions.csv', index=False)

        meta_data_df = pd.DataFrame(
            [{
                "lamda_kl": lambda_kl,
                "lamda_conf": lambda_conf
            }]
        )
        meta_data_df.to_csv(f'{output_folder}/meta_data.csv')

