"""
Hyperparameter grid search for the GCN survival model.

Runs all combinations (or a random subset) of hidden dimension, number of GCN
layers, learning rate and k (3 x 3 x 3 x 3 = 81 configurations) on the single
stratified 80/20 patient split, logging each trial as a separate MLflow run
under the 'survival-gnn-and-clinical-sweep' experiment and writing a ranked
CSV to results/.

Usage:
    uv run python sweep.py --clinical --endpoint DFS
    uv run python sweep.py --clinical --endpoint DFS \
        --pretrained pretrained_metacluster_nocap.pt --encoder_lr 5e-5 1e-4

Batch size is fixed (--batch_size, default 64) rather than swept: the Cox
partial likelihood builds risk sets within each batch, so batch size changes
the loss itself, and lr should be tuned with it held constant.
"""

import argparse
import itertools
import os
import random

import torch
import numpy as np
import pandas as pd
import mlflow
from torch_geometric.loader import DataLoader

from data_utils import (
    create_pytorch_dataset,
    patient_split,
    SPATIAL_FEATURE_NAMES,
    StratifiedBatchSampler,
    get_event_labels,
)
from train import (
    CoxPHLoss,
    evaluate,
    create_model,
    get_device,
    fit,
    freeze_encoder,
    make_optimizer,
)
from pretraining import load_pretrained_weights


SEARCH_SPACE = {
    "hidden_dim": [8, 16, 32],
    "lr": [1e-3, 5e-4, 1e-4],
    "n_layers": [2, 3, 4],
    "k": [8, 10, 12],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for GNN survival models"
    )
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
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--max_cells", type=int, default=None,
        help="Max cells per sample (default: no cap)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Fixed batch size for all trials (not swept)"
    )
    parser.add_argument(
        "--max_trials", type=int, default=0, help="Max trials (0 = full grid)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained encoder weights. When specified, sweep includes freeze_encoder=[True,False]",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        nargs="+",
        default=None,
        help="Encoder learning rates to sweep for non-frozen pretrained trials "
             "(e.g. 5e-5 1e-4). Omitted: encoder and head share lr.",
    )
    parser.add_argument(
        "--clinical",
        action="store_true",
        help="Include clinical features (age, grade, tumor_size, ER/PR/HER2) at bottleneck",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Where to save the ranked results table (rewritten after every trial). "
             "Default: results/sweep_{endpoint}_{scratch|pretrained}.csv",
    )
    return parser.parse_args()


def save_results(results: list, path: str, args) -> None:
    """Write the trials so far to CSV, ranked by val C-index."""
    df = pd.DataFrame(results).sort_values("best_val_cindex", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["endpoint"] = args.endpoint
    df["clinical"] = args.clinical
    df["max_cells"] = args.max_cells
    df["val_cindex_level"] = "patient"
    df["epochs"] = args.epochs
    df["patience"] = args.patience
    df["seed"] = args.seed
    df["pretrained_path"] = args.pretrained
    df["data"] = args.data
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def build_grid(include_pretrain_options: bool = False, pretrained_config: dict = None,
               encoder_lrs: list = None):
    """Build full grid of hyperparameter combinations.

    When pretrained_config is provided, fix hidden_dim and n_layers to match
    the pretrained weights (can't transfer between architectures).
    """
    space = SEARCH_SPACE.copy()

    if pretrained_config:
        # Fix architecture to match pretrained weights
        space["hidden_dim"] = [pretrained_config["hidden_dim"]]
        space["n_layers"] = [pretrained_config["n_layers"]]

    if include_pretrain_options:
        space["freeze_encoder"] = [True, False]

    keys = list(space.keys())
    values = list(space.values())
    combos = []
    for vals in itertools.product(*values):
        combo = dict(zip(keys, vals))
        # encoder_lr only matters when the pretrained encoder is being trained
        if include_pretrain_options and not combo["freeze_encoder"] and encoder_lrs:
            for enc_lr in encoder_lrs:
                combos.append({**combo, "encoder_lr": enc_lr})
        else:
            combos.append({**combo, "encoder_lr": None})
    return combos


def run_trial(
    params,
    train_loader,
    val_loader,
    args,
    device,
    pretrained_path=None,
    dim_features=4,
    dim_clinical_features=0,
    val_pids=None,
):
    """Run a single training trial and return metrics.

    val_pids: patient IDs in val loader order; when given, val C-index is
        computed at patient level (risk averaged over each patient's cores).
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dim_nnd = len(SPATIAL_FEATURE_NAMES)
    model = create_model(
        params["hidden_dim"],
        dim_nnd_features=dim_nnd,
        dim_clinical_features=dim_clinical_features,
        n_layers=params["n_layers"],
        dim_features=dim_features,
    ).to(device)

    if pretrained_path:
        model = load_pretrained_weights(model, pretrained_path)
        if params.get("freeze_encoder", False):
            freeze_encoder(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # encoder_lr is None for frozen and from-scratch trials (see build_grid)
    optimizer = make_optimizer(
        model, params["lr"], params["encoder_lr"] if pretrained_path else None
    )
    result = fit(model, train_loader, val_loader, val_pids, optimizer, device,
                 epochs=args.epochs, patience=args.patience)

    # Evaluate best checkpoint on train
    if result["best_state"] is not None:
        model.load_state_dict(result["best_state"])
        model.to(device)

    _, train_cindex, _, _, _ = evaluate(model, train_loader, CoxPHLoss(), device)

    return {
        "best_val_cindex": result["best_val_cindex"],
        "train_cindex": train_cindex,
        "best_epoch": result["best_epoch"],
        "n_params": n_params,
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df):,} cells from {df['core'].nunique()} cores")

    # Build grid
    use_pretrained = args.pretrained is not None
    pretrained_config = None

    if use_pretrained:
        # Load pretrained config to fix architecture params
        checkpoint = torch.load(args.pretrained, weights_only=False)
        pretrained_config = checkpoint["config"]
        print(f"\nUsing pretrained weights: {args.pretrained}")
        print(
            f"  Architecture: hidden_dim={pretrained_config['hidden_dim']} "
            f"n_layers={pretrained_config['n_layers']}"
        )
        print("  Grid varies: lr, k, freeze_encoder, encoder_lr")

    grid = build_grid(
        include_pretrain_options=use_pretrained, pretrained_config=pretrained_config,
        encoder_lrs=args.encoder_lr,
    )
    for params in grid:
        params["batch_size"] = args.batch_size
    total = len(grid)
    print(f"\nFull grid: {total} combinations")

    if args.max_trials > 0 and args.max_trials < total:
        random.seed(args.seed)
        grid = random.sample(grid, args.max_trials)
        print(f"Sampling {args.max_trials} trials")

    n_trials = len(grid)

    # Cache datasets by k value to avoid recomputation
    dataset_cache = {}
    split_cache = {}

    mlflow.set_experiment("survival-gnn-and-clinical-sweep")
    results = []
    output_csv = args.output_csv or os.path.join(
        "results",
        f"sweep_{args.endpoint}_{'pretrained' if use_pretrained else 'scratch'}.csv",
    )
    print(f"Results table: {output_csv}")

    for i, params in enumerate(grid):
        k = params["k"]

        # Create or reuse dataset for this k
        if k not in dataset_cache:
            print(f"\nBuilding dataset for k={k}...")
            dataset = create_pytorch_dataset(
                df,
                endpoint=args.endpoint,
                k=k,
                max_cells=args.max_cells,
                normalize_coords=True,
            )
            dataset_cache[k] = dataset

            train_idx, val_idx = patient_split(
                dataset,
                val_size=0.2,
                random_state=args.seed,
            )
            # Materialise once per k: __getitem__ rebuilds the k-NN graph on
            # every call, which is slow on uncapped cores.
            # val PIDs in val loader order (shuffle=False) for the patient-level
            # C-index, matching train.py / run_cv.py and the Cox baselines.
            split_cache[k] = (
                [dataset[j] for j in train_idx],
                [dataset[j] for j in val_idx],
                [str(dataset.data[dataset.cores[j]]["PID"]) for j in val_idx],
            )

        dataset = dataset_cache[k]
        train_dataset, val_dataset, val_pids = split_cache[k]

        batch_size = params["batch_size"]
        train_labels = get_event_labels(train_dataset)
        train_sampler = StratifiedBatchSampler(
            train_labels, batch_size=batch_size, shuffle=True, seed=args.seed
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        freeze_str = ""
        if use_pretrained:
            freeze_str = (f" freeze={params.get('freeze_encoder', False)}"
                          f" encoder_lr={params['encoder_lr']}")
        print(
            f"\n[{i + 1}/{n_trials}] hidden_dim={params['hidden_dim']} "
            f"lr={params['lr']} n_layers={params['n_layers']} k={params['k']} batch={params['batch_size']}{freeze_str}"
        )

        with mlflow.start_run() as run:
            log_params = {
                "hidden_dim": params["hidden_dim"],
                "lr": params["lr"],
                "n_layers": params["n_layers"],
                "k": params["k"],
                "batch_size": params["batch_size"],
                "endpoint": args.endpoint,
                "epochs": args.epochs,
                "patience": args.patience,
                "seed": args.seed,
                "max_cells": args.max_cells,
                "val_cindex_level": "patient",
                "data": args.data,
                "pretrained_used": use_pretrained,
            }
            if use_pretrained:
                log_params["freeze_encoder"] = params.get("freeze_encoder", False)
                log_params["pretrained_path"] = args.pretrained
                log_params["encoder_lr"] = params["encoder_lr"]
            log_params["clinical"] = args.clinical
            mlflow.log_params(log_params)

            dim_clinical = dataset.n_clinical_features if args.clinical else 0
            metrics = run_trial(
                params,
                train_loader,
                val_loader,
                args,
                device,
                pretrained_path=args.pretrained,
                dim_features=dataset.dim_features,
                dim_clinical_features=dim_clinical,
                val_pids=val_pids,
            )

            mlflow.log_metrics(
                {
                    "best_val_cindex": metrics["best_val_cindex"],
                    "train_cindex": metrics["train_cindex"],
                    "best_epoch": metrics["best_epoch"],
                    "n_params": metrics["n_params"],
                }
            )

        print(
            f"  -> val_cindex={metrics['best_val_cindex']:.3f} "
            f"epoch={metrics['best_epoch']} "
            f"params={metrics['n_params']:,}"
        )

        results.append({**params, **metrics, "mlflow_run_id": run.info.run_id})
        save_results(results, output_csv, args)

    # Summary table sorted by validation C-index
    width = 98 if use_pretrained else 79
    print("\n" + "=" * width)
    print("SWEEP RESULTS (sorted by val C-index)")
    print("=" * width)

    results.sort(key=lambda x: x["best_val_cindex"], reverse=True)

    if use_pretrained:
        print(
            f"{'#':>3} {'hdim':>5} {'lr':>8} {'layers':>6} {'k':>3} {'bs':>3} {'freeze':>6} {'enc_lr':>8} "
            f"{'val_ci':>7} {'train_ci':>9} {'epoch':>5} {'params':>8}"
        )
        print("-" * width)
        for rank, r in enumerate(results, 1):
            freeze = "Y" if r.get("freeze_encoder", False) else "N"
            enc_lr = f"{r['encoder_lr']:8.1e}" if r["encoder_lr"] is not None else f"{'-':>8}"
            print(
                f"{rank:3d} {r['hidden_dim']:5d} {r['lr']:8.1e} "
                f"{r['n_layers']:6d} {r['k']:3d} {r['batch_size']:3d} {freeze:>6} {enc_lr} "
                f"{r['best_val_cindex']:7.3f} "
                f"{r['train_cindex']:9.3f} {r['best_epoch']:5d} {r['n_params']:8,}"
            )
    else:
        print(
            f"{'#':>3} {'hdim':>5} {'lr':>8} {'layers':>6} {'k':>3} {'bs':>3} "
            f"{'val_ci':>7} {'train_ci':>9} {'epoch':>5} {'params':>8}"
        )
        print("-" * width)
        for rank, r in enumerate(results, 1):
            print(
                f"{rank:3d} {r['hidden_dim']:5d} {r['lr']:8.1e} "
                f"{r['n_layers']:6d} {r['k']:3d} {r['batch_size']:3d} "
                f"{r['best_val_cindex']:7.3f} "
                f"{r['train_cindex']:9.3f} {r['best_epoch']:5d} {r['n_params']:8,}"
            )

    print(f"\nBest trial: {results[0]}")
    print(f"Results table saved to {output_csv}")


if __name__ == "__main__":
    main()
