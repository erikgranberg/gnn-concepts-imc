"""
Multi-Layer Perceptron module.
"""

import torch.nn as nn


class ModuleMLP(nn.Sequential):
    """Linear layers with an activation between them (none after the last)."""

    def __init__(self, in_channels: int, units: list, activation: nn.Module = None):
        if activation is None:
            activation = nn.LeakyReLU(0.1)

        layers = []
        prev_dim = in_channels
        for i, dim in enumerate(units):
            layers.append(nn.Linear(prev_dim, dim))
            if i < len(units) - 1:
                layers.append(activation)
            prev_dim = dim

        super().__init__(*layers)
