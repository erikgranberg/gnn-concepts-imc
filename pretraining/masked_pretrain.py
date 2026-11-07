"""
Masked cell reconstruction pretraining for GNN encoders.

Pretext task: randomly mask cells in each graph, predict their
positions and types from spatial context. This learns useful
representations without requiring survival labels.

Pretraining uses every core in the dataset, including those of patients
without survival data. The defaults reproduce the encoder used in the thesis
(3-layer GCN, hidden dim 32, k = 12, 15 % masking, 100 epochs).

Usage:
    uv run python -m pretraining.masked_pretrain --output pretrained_metacluster_nocap.pt
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import mlflow
from torch_geometric.loader import DataLoader

from data_utils import create_pytorch_dataset
from models.modules import ModuleMLP, ModuleGCN
from models.consts import BATCH_KEY_NODE_FEATURES


class MaskedPretrainEncoder(nn.Module):
    """
    Encoder with reconstruction heads for masked pretraining.

    Architecture:
        1. Learnable [MASK] token replaces masked node features
        2. Feature embedding MLP
        3. GNN layers for node embedding
        4. Reconstruction heads predict masked positions and types

    The feature_embedding and node_embedding can be transferred to
    survival models after pretraining.
    """

    def __init__(
        self,
        dim_features: int = 4,  # [x, y, ...cell_type_onehot]
        n_cell_types: int = 2,  # number of cell type classes
        dim_hidden: int = 32,
        n_layers: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dim_features = dim_features
        self.n_cell_types = n_cell_types
        self.dim_hidden = dim_hidden

        # Learnable mask token (replaces masked node features)
        self.mask_token = nn.Parameter(torch.randn(dim_features))

        # Feature embedding (same as survival models)
        self.feature_embedding = ModuleMLP(
            in_channels=dim_features,
            units=[dim_hidden] * n_layers,
            activation=nn.LeakyReLU(0.1),
        )

        # GCN encoder (same architecture as the survival model)
        self.node_embedding = ModuleGCN(
            in_channels=dim_hidden,
            units=[dim_hidden] * n_layers,
            dropout_rate=dropout_rate,
        )

        # Reconstruction heads
        # Position head: predict (x, y) coordinates
        self.position_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(dim_hidden, 2),  # x, y
        )

        # Type head: predict cell type
        self.type_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(dim_hidden, n_cell_types),  # logits for each cell type
        )

    def forward(self, data, mask_indices):
        """
        Forward pass with masked reconstruction.

        Args:
            data: PyG Data object
            mask_indices: Boolean tensor indicating which nodes to mask

        Returns:
            dict with predicted positions and types for masked nodes
        """
        x = data.x[BATCH_KEY_NODE_FEATURES].clone()
        edge_index = data.edge_index

        # Replace masked nodes with learnable mask token
        x[mask_indices] = self.mask_token

        # Feature embedding
        x = self.feature_embedding(x)

        # GCN encoding
        node_emb = self.node_embedding(x, edge_index)

        # Reconstruct only masked nodes
        masked_emb = node_emb[mask_indices]
        pred_pos = self.position_head(masked_emb)
        pred_type = self.type_head(masked_emb)

        return {
            "pred_pos": pred_pos,
            "pred_type": pred_type,
        }

    def get_encoder_state_dict(self):
        """
        Get state dict for just the encoder parts (for transfer to survival model).
        """
        return {
            "feature_embedding": self.feature_embedding.state_dict(),
            "node_embedding": self.node_embedding.state_dict(),
        }


def create_mask(num_nodes: int, mask_ratio: float = 0.15, device: torch.device = None):
    """Create random mask for nodes."""
    n_mask = max(1, int(num_nodes * mask_ratio))
    mask_indices = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    perm = torch.randperm(num_nodes, device=device)[:n_mask]
    mask_indices[perm] = True
    return mask_indices


def pretrain_epoch(model, loader, optimizer, mask_ratio, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_pos_loss = 0
    total_type_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Create mask
        mask = create_mask(batch.num_nodes, mask_ratio, device)

        # Get ground truth for masked nodes
        x_true = batch.x[BATCH_KEY_NODE_FEATURES]
        true_pos = x_true[mask, :2]  # x, y coordinates
        true_type = x_true[mask, 2:].argmax(dim=1)  # cell type (one-hot -> index)

        # Forward pass
        output = model(batch, mask)

        # Reconstruction losses
        pos_loss = F.mse_loss(output["pred_pos"], true_pos)
        type_loss = F.cross_entropy(output["pred_type"], true_type)

        # Combined loss (weight position more since it's continuous)
        loss = pos_loss + 0.5 * type_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pos_loss += pos_loss.item()
        total_type_loss += type_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "pos_loss": total_pos_loss / n_batches,
        "type_loss": total_type_loss / n_batches,
    }


@torch.no_grad()
def evaluate_pretrain(model, loader, mask_ratio, device):
    """Evaluate pretraining reconstruction quality."""
    model.eval()
    total_pos_loss = 0
    total_type_correct = 0
    total_masked = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        # Fixed mask for evaluation consistency
        torch.manual_seed(42)
        mask = create_mask(batch.num_nodes, mask_ratio, device)

        x_true = batch.x[BATCH_KEY_NODE_FEATURES]
        true_pos = x_true[mask, :2]
        true_type = x_true[mask, 2:].argmax(dim=1)

        output = model(batch, mask)

        pos_loss = F.mse_loss(output["pred_pos"], true_pos)
        pred_type = output["pred_type"].argmax(dim=1)
        type_correct = (pred_type == true_type).sum().item()

        total_pos_loss += pos_loss.item()
        total_type_correct += type_correct
        total_masked += mask.sum().item()
        n_batches += 1

    return {
        "pos_loss": total_pos_loss / n_batches,
        "type_acc": total_type_correct / total_masked if total_masked > 0 else 0,
    }


def pretrain_encoder(
    data_path: str,
    hidden_dim: int = 32,
    n_layers: int = 3,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    mask_ratio: float = 0.15,
    k: int = 12,
    max_cells: Optional[int] = None,
    device: str = "auto",
    output_path: str = "pretrained_encoder.pt",
    seed: int = 42,
):
    """
    Run masked pretraining and save encoder weights.

    Args:
        data_path: Path to parquet data
        hidden_dim: Hidden dimension
        n_layers: Number of GNN layers
        epochs: Number of pretraining epochs
        batch_size: Batch size
        lr: Learning rate
        mask_ratio: Fraction of nodes to mask
        k: k for k-NN graph
        max_cells: Max cells per sample (None for no cap)
        device: Device string
        output_path: Where to save pretrained weights
        seed: Random seed

    Returns:
        Path to saved pretrained weights
    """
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Load data (no survival labels needed!)
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} cells from {df['core'].nunique()} cores")

    # Create dataset - use any endpoint since we don't use survival labels,
    # and keep cores without survival data (they are valid unlabelled graphs)
    dataset = create_pytorch_dataset(
        df,
        endpoint="OS",  # doesn't matter for pretraining
        k=k,
        max_cells=max_cells,
        normalize_coords=True,
        require_survival=False,
    )
    print(f"Dataset: {len(dataset)} graphs")
    print(f"Cell types: {dataset.cell_types} ({dataset.n_cell_types} types)")
    print(f"Node features: {dataset.dim_features} dims")

    # Use all data for pretraining (no train/val/test split needed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model - use dim_features and n_cell_types from dataset
    model = MaskedPretrainEncoder(
        dim_features=dataset.dim_features,
        n_cell_types=dataset.n_cell_types,
        dim_hidden=hidden_dim,
        n_layers=n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel: GCN (pretrain)")
    print(f"Parameters: {n_params:,}")
    print(f"Mask ratio: {mask_ratio:.0%}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    print(f"\nPretraining for {epochs} epochs...")
    print("-" * 60)

    mlflow.set_experiment("pretrain-masked")
    with mlflow.start_run():
        mlflow.log_params({
            "model_type": "gcn",
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "mask_ratio": mask_ratio,
            "k": k,
            "max_cells": max_cells,
            "seed": seed,
        })

        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            metrics = pretrain_epoch(model, loader, optimizer, mask_ratio, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": metrics["loss"],
                "train_pos_loss": metrics["pos_loss"],
                "train_type_loss": metrics["type_loss"],
            }, step=epoch)

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                # Save full model for potential resumption
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.get_encoder_state_dict(),
                    "config": {
                        "model_type": "gcn",
                        "hidden_dim": hidden_dim,
                        "n_layers": n_layers,
                        "dim_features": dataset.dim_features,
                        "n_cell_types": dataset.n_cell_types,
                    },
                }, output_path)

            if epoch % 10 == 0 or epoch == 1:
                eval_metrics = evaluate_pretrain(model, loader, mask_ratio, device)
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Pos MSE: {metrics['pos_loss']:.4f} | "
                    f"Type Acc: {eval_metrics['type_acc']:.1%}"
                )
                mlflow.log_metrics({
                    "eval_pos_loss": eval_metrics["pos_loss"],
                    "eval_type_acc": eval_metrics["type_acc"],
                }, step=epoch)

        print("-" * 60)
        print(f"Best loss: {best_loss:.4f}")
        print(f"Saved pretrained encoder to {output_path}")

        mlflow.log_artifact(output_path)

    return output_path


def load_pretrained_weights(model: nn.Module, pretrained_path: str, strict: bool = False):
    """
    Load pretrained encoder weights into a survival model.

    Args:
        model: Survival model (GraphEmbedding)
        pretrained_path: Path to pretrained weights file
        strict: Whether to require exact match

    Returns:
        model with loaded weights
    """
    checkpoint = torch.load(pretrained_path, weights_only=False, map_location="cpu")
    encoder_state = checkpoint["encoder_state_dict"]

    # Load feature embedding
    if hasattr(model, "feature_embedding"):
        model.feature_embedding.load_state_dict(encoder_state["feature_embedding"])
        print(f"Loaded feature_embedding from {pretrained_path}")

    # Load node embedding
    if hasattr(model, "node_embedding"):
        model.node_embedding.load_state_dict(encoder_state["node_embedding"])
        print(f"Loaded node_embedding from {pretrained_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Masked pretraining for GNN encoder")
    parser.add_argument("--data", type=str, default="datasets/point_patterns_metaclusters.parquet")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="pretrained_encoder.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pretrain_encoder(
        data_path=args.data,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        k=args.k,
        max_cells=args.max_cells,
        device=args.device,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
