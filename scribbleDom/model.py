import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, ModuleList, Module
from torch_geometric.nn import GCNConv, global_mean_pool


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


class GraphClassifier(Module):
    def __init__(self, in_dim, hidden_dims, num_classes):
        super().__init__()

        self.initial_proj = Linear(in_dim, hidden_dims[0])
        self.bn1 = BatchNorm1d(hidden_dims[0])

        self.inception1 = GraphInceptionBlock(hidden_dims[0], [64, 64, 32])
        self.inception2 = GraphInceptionBlock(160, [64, 64, 32])  # 160 = 64+64+32

        self.final_proj = Linear(160, num_classes)

    def forward(self, x, edge_index):
        x = self.initial_proj(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.inception1(x, edge_index)
        x = self.inception2(x, edge_index)

        out = self.final_proj(x)
        return out
