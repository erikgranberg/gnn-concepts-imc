"""
Graph Neural Network modules.
"""

from typing import List

import torch.nn as nn
import torch_geometric.nn as geom_nn


class ModuleGCN(nn.Module):
    """Graph Convolutional Network layers."""

    def __init__(
        self,
        in_channels: int,
        units: List[int],
        dropout_rate: float = 0.0,
        activation: nn.Module = None,
        **kwargs
    ):
        super().__init__()
        if activation is None:
            activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        layers = []
        prev_dim = in_channels
        for dim in units:
            layers += [
                geom_nn.GCNConv(in_channels=prev_dim, out_channels=dim),
                activation,
                nn.Dropout(dropout_rate),
            ]
            prev_dim = dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
