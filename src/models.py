"""GNN models for link prediction.

A single encoder/decoder class where the only thing that changes between
experiments is the convolution type ('gcn' or 'gat'). That keeps the
GCN-vs-GAT comparison clean and controlled.

Three optional modifications, all off by default so previously logged runs
stay comparable. Each applies to BOTH conv types: applying a fix only to GAT
would confound the central GCN-vs-GAT claim, since a difference could then
come from the fix rather than from attention.

match_capacity
    PyG's GATConv concatenates its heads, so GATConv(in, 64, heads=4) emits
    256 channels while GCNConv(in, 64) emits 64. The default configuration
    therefore gives GAT roughly 4x the parameters of GCN, and "GAT loses"
    invites the reply that the two were never the same size. With this on,
    GAT's per-head width becomes hidden_dim // heads so both encoders emit
    hidden_dim channels. This is a validity fix, not a tuning knob: it belongs
    in the report whether or not it changes the numbers.

layer_norm
    GCNConv normalises by node degree internally; GATConv does not. On a graph
    whose degrees span one to several hundred, that leaves GAT's activations
    scaled by neighbourhood size, which is a plausible cause of the training
    instability observed here (GAT early-stopped at 351 / 90 / 167 epochs
    across three seeds of one configuration, against GCN's 52 / 32 / 65).
    LayerNorm after each conv removes that scale dependence.

dropout
    Exposed on the CLI. The 0.5 default is aggressive for attention models,
    and under the default (unmatched) capacity it drops 256 GAT channels
    against GCN's 64.
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
        match_capacity: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.dropout = dropout
        self.match_capacity = match_capacity
        self.use_layer_norm = layer_norm

        if conv_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
            h1 = hidden_dim
        elif conv_type == "gat":
            if match_capacity:
                if hidden_dim % heads != 0:
                    raise ValueError(
                        f"--match-capacity needs hidden_dim ({hidden_dim}) "
                        f"divisible by heads ({heads})."
                    )
                per_head = hidden_dim // heads
            else:
                per_head = hidden_dim
            self.conv1 = GATConv(in_dim, per_head, heads=heads)
            h1 = per_head * heads
            self.conv2 = GATConv(h1, out_dim, heads=1)
        elif conv_type == "sage":
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)
            h1 = hidden_dim
        else:
            raise ValueError("conv_type must be 'gcn', 'gat', or 'sage'")

        self.hidden_out_dim = h1
        if layer_norm:
            self.norm1 = torch.nn.LayerNorm(h1)
            self.norm2 = torch.nn.LayerNorm(out_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.use_layer_norm:
            x = self.norm1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.use_layer_norm:
            x = self.norm2(x)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)