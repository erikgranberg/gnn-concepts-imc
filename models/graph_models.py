"""
GCN survival model with a concept bottleneck.

Node features -> MLP feature embedding -> GCN layers -> global mean pool.
The pooled graph embedding is concatenated with the NND + Ripley's K features
and (optionally) clinical features before an MLP head that outputs a
log-hazard, trained with the Cox partial likelihood (see train.py).
"""

from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .consts import (
    BATCH_KEY_NODE_FEATURES,
    BATCH_KEY_NODE_EMBEDDING,
    BATCH_KEY_GRAPH_EMBEDDING,
    BATCH_KEY_GRAPH_LABELS,
    BATCH_KEY_GRAPH_YHAT,
)
from .modules import ModuleGCN, ModuleMLP


class GraphEmbedding(nn.Module):
    """
    GCN-based graph embedding model for survival prediction.

    Architecture:
        1. Feature embedding MLP
        2. GCN layers for node embedding
        3. Global mean pooling to graph embedding
        4. Concatenation with NND/Ripley and clinical features (bottleneck)
        5. MLP head outputting a log-hazard
    """

    def __init__(
        self,
        dim_features: int = 4,  # x, y + one-hot cell type
        dim_output: int = 1,  # log-hazard
        dim_node_embedding: List[int] = [32, 32],
        dim_mlp: List[int] = [16],
        dropout_rate: float = 0.1,
        dim_nnd_features: int = 0,
        dim_clinical_features: int = 0,
    ):
        super().__init__()
        self.dim_latent = dim_node_embedding[-1]
        self.dim_nnd_features = dim_nnd_features
        self.dim_clinical_features = dim_clinical_features

        # Feature embedding
        self.feature_embedding = ModuleMLP(
            in_channels=dim_features,
            units=dim_node_embedding,
            activation=nn.LeakyReLU(0.1),
        )

        # GCN layers
        self.node_embedding = ModuleGCN(
            in_channels=self.dim_latent,
            units=dim_node_embedding,
            dropout_rate=dropout_rate,
        )

        # Output head (includes NND + clinical features at bottleneck)
        self.output_head = ModuleMLP(
            in_channels=self.dim_latent + dim_nnd_features + dim_clinical_features,
            units=dim_mlp + [dim_output],
            activation=nn.LeakyReLU(0.1),
        )

    def forward(self, data):
        x = data.x[BATCH_KEY_NODE_FEATURES]
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") else None

        # Feature embedding
        x = self.feature_embedding(x)

        # Node embedding via GCN
        node_emb = self.node_embedding(x, edge_index)

        # Global pooling
        graph_emb = global_mean_pool(node_emb, batch)

        # Concatenate auxiliary features at bottleneck
        bottleneck_features = [graph_emb]

        if self.dim_nnd_features > 0 and hasattr(data, "nnd_features"):
            nnd = data.nnd_features.view(-1, self.dim_nnd_features)
            bottleneck_features.append(nnd)

        if self.dim_clinical_features > 0 and hasattr(data, "clinical_features"):
            clinical = data.clinical_features.view(-1, self.dim_clinical_features)
            bottleneck_features.append(clinical)

        graph_emb_aug = torch.cat(bottleneck_features, dim=1)

        # Prediction
        y_hat = self.output_head(graph_emb_aug)

        return {
            BATCH_KEY_NODE_EMBEDDING: node_emb,
            BATCH_KEY_GRAPH_EMBEDDING: graph_emb,
            BATCH_KEY_GRAPH_YHAT: y_hat,
            BATCH_KEY_GRAPH_LABELS: data.y[BATCH_KEY_GRAPH_LABELS],
        }
