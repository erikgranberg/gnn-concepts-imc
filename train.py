"""
GNN training for survival prediction using Cox partial likelihood loss.

Predicts risk scores from spatial cell arrangements for survival analysis.
Higher risk score = higher hazard = shorter expected survival.

The defaults are the configuration used in the thesis (3-layer GCN, hidden
dim 32, k = 12, lr 5e-4, batch size 64, at most 100 epochs, patience 15).

Usage:
    # Full model on the IMC data (clinical + NND + Ripley's K at the bottleneck)
    uv run python train.py --endpoint DFS --clinical

    # Same, starting from the masked-pretraining encoder
    uv run python train.py --endpoint DFS --clinical \
        --pretrained pretrained_metacluster_nocap.pt --encoder_lr 1e-4
"""

import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
from torch_geometric.loader import DataLoader

from data_utils import (
    create_pytorch_dataset,
    patient_split,
    NND_FEATURE_NAMES,
    SPATIAL_FEATURE_NAMES,
    StratifiedBatchSampler,
    get_event_labels,
)
from models import GraphEmbedding
from models.consts import BATCH_KEY_GRAPH_YHAT, BATCH_KEY_GRAPH_LABELS
from pretraining import load_pretrained_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN for survival prediction")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/point_patterns_metaclusters.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="DFS",
        choices=["DFS", "OS"],
        help="Survival endpoint",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("--k", type=int, default=12, help="k for k-NN graph")
    parser.add_argument(
        "--max_cells", type=int, default=None,
        help="Max cells per sample (default: no cap)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained encoder weights (from pretraining/masked_pretrain.py)",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze pretrained encoder weights during fine-tuning",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=None,
        help="Learning rate for the pretrained encoder during fine-tuning. "
             "Should be lower than --lr, e.g. --lr 5e-4 --encoder_lr 5e-5. "
             "Only used when --pretrained is set and --freeze_encoder is not.",
    )
    parser.add_argument(
        "--clinical",
        action="store_true",
        help="Include clinical features (age, grade, tumor_size, ER/PR/HER2) at bottleneck",
    )
    parser.add_argument(
        "--val_pids_file",
        type=str,
        default=None,
        help="Path to text file with one patient ID per line to use as val set. "
             "Overrides the random seed-based split.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="best_model.pt",
        help="Path to save the best model checkpoint (default: best_model.pt)",
    )
    parser.add_argument(
        "--no_spatial_features",
        action="store_true",
        help="Exclude the NND + Ripley's K features from the bottleneck, so the "
             "Cox head sees only the GCN graph embedding (plus clinical features "
             "if --clinical is also set). Use to isolate what the GNN learns on "
             "its own, without the classical spatial-statistic bottleneck.",
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get the appropriate device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards partial likelihood loss.

    For each event (uncensored observation), computes:
        loss = -[risk_i - log(sum_j: t_j >= t_i of exp(risk_j))]

    The model predicts risk scores where higher = higher hazard.
    """

    def forward(
        self, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            risk: Predicted risk scores (batch_size,)
            time: Survival times (batch_size,)
            event: Event indicators, 1=event, 0=censored (batch_size,)

        Returns:
            Negative partial log-likelihood (scalar)
        """
        # Sort by time (descending) for efficient risk set computation
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        event = event[order]

        # Compute log-sum-exp of risk scores for cumulative risk set
        # For numerical stability, subtract max
        risk_max = risk.max()
        log_cumsum_exp = torch.logcumsumexp(risk - risk_max, dim=0) + risk_max

        # Partial likelihood: sum over events of [risk_i - log(cumulative_risk)]
        log_likelihood = risk - log_cumsum_exp

        # Only count uncensored observations
        n_events = event.sum()
        if n_events == 0:
            return torch.tensor(0.0, device=risk.device, requires_grad=True)

        return -torch.sum(log_likelihood * event) / n_events


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """
    Compute Harrell's concordance index (C-index).

    C-index measures the probability that for a random pair of subjects,
    the subject with higher risk score has shorter survival time.

    Args:
        risk: Predicted risk scores (higher = higher risk)
        time: Observed survival times
        event: Event indicators (1=event, 0=censored)

    Returns:
        C-index in [0, 1], where 0.5 = random, 1.0 = perfect
    """
    n = len(risk)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only compare if at least one is uncensored
            # and we can determine ordering
            if time[i] == time[j]:
                continue

            # Determine which has shorter time
            if time[i] < time[j]:
                shorter, longer = i, j
            else:
                shorter, longer = j, i

            # Can only use pair if the shorter time is an event
            if event[shorter] == 0:
                continue

            # Check concordance: higher risk should have shorter time
            if risk[shorter] > risk[longer]:
                concordant += 1
            elif risk[shorter] < risk[longer]:
                discordant += 1
            else:
                # Tie in risk - count as 0.5
                concordant += 0.5
                discordant += 0.5

    total = concordant + discordant
    if total == 0:
        return 0.5

    return concordant / total


def create_model(
    hidden_dim: int,
    dim_nnd_features: int = 0,
    dim_clinical_features: int = 0,
    n_layers: int = 2,
    dim_features: int = 4,
) -> nn.Module:
    """Create the GCN survival model."""
    kwargs = {
        "dim_features": dim_features,  # 2 + n_cell_types
        "dim_output": 1,  # risk score
        "dim_node_embedding": [hidden_dim] * n_layers,
        "dim_mlp": [hidden_dim // 2],
        "dim_nnd_features": dim_nnd_features,
        "dim_clinical_features": dim_clinical_features,
    }
    return GraphEmbedding(**kwargs, dropout_rate=0.1)


def infer_dim_nnd_features(state_dict: dict, dim_latent: int, dim_clinical_features: int) -> int:
    """
    Recover how many NND/Ripley features a saved checkpoint's bottleneck expects,
    by reading the output_head's input width. Lets loaders (e.g.
    run_explainability.py) handle checkpoints trained with or without
    --no_spatial_features without needing to know which was used ahead of time.
    """
    in_channels = state_dict["output_head.0.weight"].shape[1]
    return in_channels - dim_latent - dim_clinical_features


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        output = model(batch)
        risk = output[BATCH_KEY_GRAPH_YHAT].squeeze(-1)
        y_true = output[BATCH_KEY_GRAPH_LABELS]

        # PyG batching concatenates graph labels, reshape to (batch_size, 2)
        y_true = y_true.view(batch.num_graphs, 2)
        time = y_true[:, 0]
        event = y_true[:, 1]

        loss = criterion(risk, time, event)

        # Skip if no events in batch
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, pids=None):
    """Evaluate model and compute C-index.

    Args:
        pids: Optional list of patient IDs in loader order (loader must have
              shuffle=False). When provided, risk scores are averaged across
              cores per patient before computing C-index, matching the
              patient-level evaluation used by the Cox PH baselines.
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    all_risks = []
    all_times = []
    all_events = []

    for batch in loader:
        batch = batch.to(device)
        output = model(batch)

        risk = output[BATCH_KEY_GRAPH_YHAT].squeeze(-1)
        y_true = output[BATCH_KEY_GRAPH_LABELS]

        y_true = y_true.view(batch.num_graphs, 2)
        time = y_true[:, 0]
        event = y_true[:, 1]

        loss = criterion(risk, time, event)
        total_loss += loss.item()
        n_batches += 1

        all_risks.extend(risk.cpu().numpy())
        all_times.extend(time.cpu().numpy())
        all_events.extend(event.cpu().numpy())

    avg_loss = total_loss / n_batches

    risks = np.array(all_risks)
    times = np.array(all_times)
    events = np.array(all_events)

    if pids is not None:
        # Average risk scores across cores per patient, then compute C-index
        # at patient level — comparable to Cox PH baselines.
        pid_risks: dict = {}
        pid_labels: dict = {}
        for r, t, e, pid in zip(risks, times, events, pids):
            pid_risks.setdefault(pid, []).append(r)
            pid_labels[pid] = (t, e)
        unique_pids = list(pid_risks)
        risks  = np.array([np.mean(pid_risks[p])  for p in unique_pids])
        times  = np.array([pid_labels[p][0]        for p in unique_pids])
        events = np.array([pid_labels[p][1]        for p in unique_pids])

    c_index = concordance_index(risks, times, events)

    return avg_loss, c_index, risks, times, events


def freeze_encoder(model) -> None:
    """Freeze the (pretrained) feature embedding and GCN layers."""
    for module in (model.feature_embedding, model.node_embedding):
        for param in module.parameters():
            param.requires_grad = False


def make_optimizer(model, lr: float, encoder_lr: float | None = None):
    """Adam with weight decay 1e-4. With encoder_lr, the encoder (feature
    embedding + GCN layers) and the head get separate learning rates."""
    if encoder_lr is None:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    encoder_params = (
        list(model.feature_embedding.parameters()) +
        list(model.node_embedding.parameters())
    )
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.Adam([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": head_params,    "lr": lr},
    ], weight_decay=1e-4)


def fit(model, train_loader, val_loader, val_pids, optimizer, device,
        epochs: int, patience: int, checkpoint_path: str | None = None,
        verbose: bool = False) -> dict:
    """
    Train with the Cox partial likelihood, select the epoch with the best
    patient-level validation C-index, and stop early after `patience` epochs
    without improvement (0 disables early stopping). The learning rate is
    halved when the validation C-index plateaus for 20 epochs.

    Per-epoch metrics are logged to the active MLflow run. The best weights are
    saved to checkpoint_path (if given) whenever the validation C-index improves.

    Returns {"best_val_cindex", "best_epoch", "best_state"}.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20,  # maximise C-index
    )
    criterion = CoxPHLoss()

    if verbose:
        print(f"\nTraining for {epochs} epochs (Cox PH loss)...")
        if patience > 0:
            print(f"Early stopping patience: {patience} epochs")
        print("-" * 70)

    best_val_cindex = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        _, train_cindex, _, _, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_cindex, _, _, _ = evaluate(
            model, val_loader, criterion, device, pids=val_pids
        )
        scheduler.step(val_cindex)

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_cindex": train_cindex,
                "val_loss": val_loss,
                "val_cindex": val_cindex,
            },
            step=epoch,
        )

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:7.4f} | "
                f"Val Loss: {val_loss:7.4f} | "
                f"Val C-index: {val_cindex:.3f}"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
            break

    if verbose:
        print("-" * 70)
        print(f"Best validation C-index: {best_val_cindex:.3f} at epoch {best_epoch}")

    return {"best_val_cindex": best_val_cindex, "best_epoch": best_epoch,
            "best_state": best_state}


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df):,} cells from {df['core'].nunique()} cores")

    # Create dataset
    dataset = create_pytorch_dataset(
        df,
        endpoint=args.endpoint,
        k=args.k,
        max_cells=args.max_cells,
        normalize_coords=True,
    )
    print(f"Dataset: {len(dataset)} graphs ({args.endpoint} endpoint)")

    # Count events
    n_events = sum(
        1 for i in range(len(dataset)) if dataset[i].y[BATCH_KEY_GRAPH_LABELS][1] == 1
    )
    print(f"Events: {n_events} / {len(dataset)} ({n_events / len(dataset):.1%})")

    # Train/val split (by patient to prevent data leakage)
    if args.val_pids_file is not None:
        with open(args.val_pids_file) as f:
            val_pids = set(line.strip() for line in f if line.strip())
        # Compare as strings — PIDs may be int in the dataset but str in the file
        train_idx = [i for i in range(len(dataset))
                     if str(dataset.data[dataset.cores[i]]["PID"]) not in val_pids]
        val_idx   = [i for i in range(len(dataset))
                     if str(dataset.data[dataset.cores[i]]["PID"]) in val_pids]
        print(f"Using val_pids_file: {len(val_pids)} val patients specified")
    else:
        train_idx, val_idx = patient_split(
            dataset, val_size=0.2, random_state=args.seed
        )

    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]

    # PIDs in val loader order (shuffle=False) for patient-level C-index
    val_pids_ordered = [str(dataset.data[dataset.cores[i]]["PID"]) for i in val_idx]

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Count events per split
    train_events = sum(1 for d in train_dataset if d.y[BATCH_KEY_GRAPH_LABELS][1] == 1)
    val_events = sum(1 for d in val_dataset if d.y[BATCH_KEY_GRAPH_LABELS][1] == 1)
    print(f"Events - Train: {train_events}, Val: {val_events}")

    # Data loaders with stratified sampling for stable Cox loss
    train_labels = get_event_labels(train_dataset)
    train_sampler = StratifiedBatchSampler(
        train_labels, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Stratified batching: ~{train_sampler.events_per_batch} events per batch")
    print(f"Cell types: {dataset.cell_types} ({dataset.n_cell_types} types)")
    print(f"Node features: {dataset.dim_features} dims (2 coords + {dataset.n_cell_types} cell types)")

    # Model — NND bottleneck normally includes Ripley's K (5 NND + 4 Ripley = 9),
    # unless --no_spatial_features asks for a bottleneck with just the GCN
    # embedding (plus clinical features, if --clinical is also set).
    dim_nnd = 0 if args.no_spatial_features else len(SPATIAL_FEATURE_NAMES)
    dim_clinical = dataset.n_clinical_features if args.clinical else 0

    if args.no_spatial_features:
        print("Spatial features excluded from bottleneck (--no_spatial_features)")
    if args.clinical:
        print(f"Clinical features: {dataset.clinical_feature_names}")

    model = create_model(
        args.hidden_dim,
        dim_nnd_features=dim_nnd,
        dim_clinical_features=dim_clinical,
        n_layers=args.n_layers,
        dim_features=dataset.dim_features,
    ).to(device)

    # Load pretrained weights if specified
    if args.pretrained:
        print(f"\nLoading pretrained weights from {args.pretrained}")
        model = load_pretrained_weights(model, args.pretrained)

        if args.freeze_encoder:
            print("Freezing encoder weights")
            freeze_encoder(model)

    print("\nModel: GCN")
    if args.pretrained:
        print(f"Pretrained: {args.pretrained}")
        print(f"Frozen encoder: {args.freeze_encoder}")
    if args.clinical:
        print(f"Clinical features: {dim_clinical} dims")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Training setup (separate encoder lr only when fine-tuning a pretrained encoder)
    encoder_lr = (args.encoder_lr
                  if args.pretrained and not args.freeze_encoder else None)
    optimizer = make_optimizer(model, args.lr, encoder_lr)
    if encoder_lr is not None:
        print(f"Fine-tuning: encoder lr={encoder_lr}, head lr={args.lr}")

    mlflow.set_experiment("survival-gnn")
    with mlflow.start_run():
        # Log all args plus derived params for easy filtering
        params = vars(args).copy()
        params["pretrained_used"] = args.pretrained is not None
        params["finetune_mode"] = (
            "frozen" if args.freeze_encoder else
            "finetune" if (args.pretrained and args.encoder_lr is not None) else
            "scratch" if not args.pretrained else
            "finetune_same_lr"
        )
        mlflow.log_params(params)

        result = fit(
            model, train_loader, val_loader, val_pids_ordered, optimizer, device,
            epochs=args.epochs, patience=args.patience,
            checkpoint_path=args.model_out, verbose=True,
        )
        mlflow.log_metrics({"best_epoch": result["best_epoch"],
                            "best_val_cindex": result["best_val_cindex"]})

if __name__ == "__main__":
    main()
