"""GNN models for link prediction.

A single encoder/decoder class where the only thing that changes between
experiments is the convolution type ('gcn' or 'gat'). That keeps the
GCN-vs-GAT comparison clean and controlled.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GNNLinkPredictor(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        conv_type: str = "gcn",
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.dropout = dropout

        if conv_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif conv_type == "gat":
            self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
            self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1)
        elif conv_type == "sage":
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)
        else:
            raise ValueError("conv_type must be 'gcn', 'gat', or 'sage'")

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
