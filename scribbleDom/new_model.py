import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax
from torch_geometric.nn import GraphNorm

class SpotAttentionAggregator(Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.attn_fc = Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)   # scalar score per cell
        )

    def forward(self, cell_embs, cell_to_spot, num_spots):
        # raw attention scores (shape [N_cells, 1]) -> squeeze -> [N_cells]
        scores = self.attn_fc(cell_embs).squeeze(-1)

        if torch.isnan(scores).any():
            print("NaN in attention scores")
            raise RuntimeError

        attn_weights = softmax(scores, cell_to_spot)  # [N_cells]

        # weighted sum per spot: use index_add_ (attn_weights.unsqueeze(1) broadcasts)
        spot_embs = torch.zeros((num_spots, cell_embs.size(1)), device=cell_embs.device, dtype=cell_embs.dtype)
        spot_embs.index_add_(0, cell_to_spot, cell_embs * attn_weights.unsqueeze(-1))

        return spot_embs


class GCNPipeline(nn.Module):
    def __init__(self, input_dim, gcn_hidden_dims, proj_dim, out_dim, dropout=0.1):
        """
        Args:
            input_dim (int): dimension of input features (after PCA).
            gcn_hidden_dims (list[int]): list of hidden dimensions for GCN layers.
                                         Example: [64, 32] for 2 GCN layers.
            proj_dim (int): dimension after initial linear projection (before GCN).
            out_dim (int): final output dimension (for clustering).
            dropout (float): dropout rate.
        """
        super(GCNPipeline, self).__init__()

        self.dropout = dropout

        # (1) Linear projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            GraphNorm(proj_dim),
            nn.ReLU(),
            # nn.Dropout(dropout)
        )

        # (2) GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms_layers = nn.ModuleList()

        last_dim = proj_dim
        for hidden_dim in gcn_hidden_dims:
            self.gcn_layers.append(GCNConv(last_dim, hidden_dim))
            self.gcn_norms_layers.append(GraphNorm(hidden_dim))
            last_dim = hidden_dim

        self.cell_to_spot_aggregator = SpotAttentionAggregator(last_dim)

        # (3) MLP head for clustering
        self.mlp_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            GraphNorm(last_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(last_dim, out_dim)  # cluster logits / embedding
        )

    def forward(self, x, edge_index, cell_to_spot, num_spots):
        # (1) project features
        x = self.proj(x)

        # (2) pass through GCN layers
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, edge_index)
            x = self.gcn_norms_layers[i](x)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

        cell_embeddings = x  # final per-cell embeddings

        spot_embeddings = self.cell_to_spot_aggregator(cell_embeddings, cell_to_spot, num_spots)

        # (4) clustering head (on spot embeddings)
        cluster_logits = self.mlp_head(spot_embeddings)

        return cluster_logits

# Example usage:
# gcn_hidden_dims = [64, 32] → two GCN layers: proj_dim → 64 → 32
# model = GCNPipeline(input_dim=15, gcn_hidden_dims=[64, 32], proj_dim=64, out_dim=10)
