import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch_geometric.data import Batch


class GNNLayer(nn.Module):
    """
    A simple GNN layer updating node features with an MLP and neighborhood aggregation.
    """

    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, node_dim)
        )

    def forward(self, h, edge_index):
        row, col = edge_index
        node_feat = h[row]
        delta_h = self.node_mlp(node_feat)

        # Aggregate updates using scatter add
        h = h + torch.zeros_like(h).scatter_add(
            0, row.unsqueeze(1).expand_as(delta_h), delta_h
        )
        return h


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over node features.
    """

    def __init__(self, node_dim, embed_dim=256, num_heads=4):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, embed_dim)
        self.attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, h, batch=None):
        h_proj = self.node_proj(h)

        if batch is not None:
            out = []
            for i in range(batch.max().item() + 1):
                idx = batch == i
                h_i = h_proj[idx].unsqueeze(0)  # [1, num_nodes, embed_dim]
                attn_out, _ = self.attn(self.query.unsqueeze(0), h_i, h_i)
                out.append(attn_out.squeeze(0))
            return torch.stack(out)
        else:
            h_i = h_proj.unsqueeze(0)
            attn_out, _ = self.attn(self.query.unsqueeze(0), h_i, h_i)
            return attn_out.squeeze(0)


class GNNFingerprint2D(nn.Module):
    """
    GNN encoder with stacked message passing layers and attention pooling.
    """

    def __init__(self, node_input_dim, hidden_dim=256, num_layers=6, out_dim=1024):
        super().__init__()
        self.node_embed = nn.Linear(node_input_dim, 64)

        self.gnn_layers = nn.ModuleList(
            [GNNLayer(64, hidden_dim) for _ in range(num_layers)]
        )

        self.attn_pool = AttentionPooling(node_dim=64, embed_dim=256, num_heads=4)

        self.projection_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        if isinstance(data, list):
            for d in data:
                if d.edge_index.shape[0] != 2:
                    d.edge_index = d.edge_index.t()
            data = Batch.from_data_list(data)
        elif data.edge_index.shape[0] != 2:
            data.edge_index = data.edge_index.t()

        h = self.node_embed(data.x)

        for gnn in self.gnn_layers:
            h = gnn(h, data.edge_index)

        pooled = self.attn_pool(h, batch=getattr(data, "batch", None))
        return self.projection_head(pooled)