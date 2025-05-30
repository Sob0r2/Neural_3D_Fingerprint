import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class GNNLayer(nn.Module):
    """
    A simple GNN layer updating node and edge features with MLPs.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, node_dim)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        
        node_feat = torch.cat([h[row], edge_attr], dim=1)
        delta_h = self.node_mlp(node_feat)

        edge_feat = torch.cat([h[row], h[col], edge_attr], dim=1)
        delta_edge = self.edge_mlp(edge_feat)

        h = h + torch.zeros_like(h).scatter_add(
            0, row.unsqueeze(1).expand_as(delta_h), delta_h
        )
        edge_attr = edge_attr + delta_edge
        return h, edge_attr


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over nodes with optional edge context.
    """

    def __init__(self, node_dim, edge_dim, embed_dim=256, num_heads=4):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, embed_dim)
        self.edge_proj = nn.Linear(edge_dim, embed_dim)
        self.attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, h, edge_index, edge_attr, batch=None):
        h_proj = self.node_proj(h)
        if edge_index is not None and edge_attr is not None:
            row, _ = edge_index
            agg_edge = scatter(
                self.edge_proj(edge_attr), row, dim=0, dim_size=h.size(0), reduce="mean"
            )
            h_proj += agg_edge

        if batch is not None:
            out = []
            for i in range(batch.max().item() + 1):
                idx = batch == i
                h_i = h_proj[idx].unsqueeze(0)
                attn_out, _ = self.attn(self.query.unsqueeze(0), h_i, h_i)
                out.append(attn_out.squeeze(0))
            return torch.stack(out)
        else:
            h_i = h_proj.unsqueeze(0)
            attn_out, _ = self.attn(self.query.unsqueeze(0), h_i, h_i)
            return attn_out.squeeze(0)


class GNNFingerprint3D(nn.Module):
    """
    Full GNN encoder model with multiple layers and attention pooling.
    """

    def __init__(
        self, node_input_dim, edge_input_dim, hidden_dim=256, num_layers=6, out_dim=1024
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_input_dim, 64)
        self.edge_embed = nn.Linear(edge_input_dim, 32)

        self.gnn_layers = nn.ModuleList(
            [GNNLayer(64, 32, hidden_dim) for _ in range(num_layers)]
        )

        self.attn_pool = AttentionPooling(64, 32, embed_dim=256, num_heads=4)
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
        edge_attr = self.edge_embed(data.edge_attr)

        for gnn in self.gnn_layers:
            h, edge_attr = gnn(h, data.edge_index, edge_attr)

        pooled = self.attn_pool(
            h, data.edge_index, edge_attr, batch=getattr(data, "batch", None)
        )
        return self.projection_head(pooled)
