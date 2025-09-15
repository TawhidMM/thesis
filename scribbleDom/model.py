import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, ModuleList, Module
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax


class GraphConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm1d(out_channels)
        self.relu = ReLU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        return self.relu(x)


class GraphInceptionBlock(Module):
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        assert len(out_channels_list) >= 2, "Need at least 2 branches for inception"
        self.branches = ModuleList([
            GraphConvBlock(in_channels, out_c) for out_c in out_channels_list
        ])

    def forward(self, x, edge_index):
        out = [branch(x, edge_index) for branch in self.branches]
        return torch.cat(out, dim=1)




class SpotAttentionAggregator(Module):
    def __init__(self, in_dim, hidden_dim=64):
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



class GraphClassifier(Module):
    def __init__(self, in_dim, hidden_dims, num_classes, attn_hidden=64):
        super().__init__()

        self.initial_proj = Linear(in_dim, hidden_dims[0])
        self.bn1 = BatchNorm1d(hidden_dims[0])

        self.inception1 = GraphInceptionBlock(hidden_dims[0], [64, 64, 32])
        self.inception2 = GraphInceptionBlock(160, [64, 64, 32])  # 160 = 64+64+32

        self.aggregator = SpotAttentionAggregator(160, hidden_dim=attn_hidden)
        self.final_proj = Linear(160, num_classes)

    def forward(self, x, edge_index, cell_to_spot, num_spots):
        # x: cell-level features aligned with cell_to_spot
        x = self.initial_proj(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.inception1(x, edge_index)
        x = self.inception2(x, edge_index)   # [N_cells, 160]

        # pool cells -> spots (aggregator will filter invalid cell_to_spot if any)
        spot_embs = self.aggregator(x, cell_to_spot, num_spots)  # [N_spots, 160]

        out = self.final_proj(spot_embs)  # [N_spots, num_classes]
        return out
