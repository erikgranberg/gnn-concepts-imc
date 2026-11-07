"""
Entry point for post-hoc explainability analysis (thesis Chapter 6).

Runs on a trained checkpoint and the held-out patients of the single
stratified 80/20 split (seed 42):
  1. Block-permutation ablation (GCN embedding / NND / Ripley's K)  block_ablation.py
  2. Per-feature permutation importance (9 NND + Ripley features)   block_ablation.py
  3. Graph-level concept discovery (K-means on graph embeddings)     concept_discovery.py
  4. Node-level concept discovery (K-means on node embeddings)       concept_discovery.py
  5. Embedding probe, synthetic data only                            embedding_probe.py

Saves all figures to figures/output/explainability/ and logs summary
metrics to MLflow under the experiment 'explainability'.

Usage:
    # Real data (model trained with --clinical)
    uv run python analysis/explainability/run_explainability.py \\
        --model_path best_model.pt --endpoint DFS --clinical

    # Synthetic validation, regime 3
    uv run python analysis/explainability/run_explainability.py \\
        --model_path models/synthetic_regime3_noise0.0.pt \\
        --data datasets/synthetic_regime3_noise0.0.parquet \\
        --endpoint DFS --synthetic --regime 3
"""

import argparse
import os
import sys

import pandas as pd
import torch
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_utils import create_pytorch_dataset, SPATIAL_FEATURE_NAMES
from train import create_model, get_device, infer_dim_nnd_features
from data_utils import patient_split

from embedding_probe import run_embedding_probe
from block_ablation     import run_block_ablation, run_feature_permutation
from concept_discovery  import (run_graph_concept_discovery,
                                run_node_concept_discovery)


DEFAULT_OUTPUT_DIR = "figures/output/explainability"
MLFLOW_EXPERIMENT  = "explainability"


def parse_args():
    p = argparse.ArgumentParser(description="Explainability analysis for survival GNN")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to trained model checkpoint (.pt)")
    p.add_argument("--data", type=str,
                   default="datasets/point_patterns_metaclusters.parquet",
                   help="Path to parquet dataset")
    p.add_argument("--endpoint", type=str, default="DFS", choices=["DFS", "OS"])
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--n_layers",   type=int, default=3)
    p.add_argument("--k",          type=int, default=12)
    p.add_argument("--max_cells",  type=int, default=None)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--clinical",   action="store_true",
                   help="Model was trained with --clinical")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    # Analysis selection
    p.add_argument("--no_embedding_probe", action="store_true",
                   help="Skip the embedding probe (synthetic only)")
    p.add_argument("--no_block_ablation", action="store_true",
                   help="Skip block-permutation ablation")
    p.add_argument("--no_feature_perm", action="store_true",
                   help="Skip per-feature permutation importance")
    p.add_argument("--n_perm", type=int, default=50,
                   help="Number of permutation replicates for block/feature ablation")
    p.add_argument("--no_graph_concepts", action="store_true",
                   help="Skip graph-level concept discovery")
    p.add_argument("--no_node_concepts", action="store_true",
                   help="Skip node-level concept discovery")
    p.add_argument("--graph_k_min", type=int, default=2,
                   help="Minimum K for graph-level concept search")
    p.add_argument("--graph_k_max", type=int, default=8,
                   help="Maximum K for graph-level concept search")
    p.add_argument("--node_k_min", type=int, default=2,
                   help="Minimum K for node-level concept search")
    p.add_argument("--node_k_max", type=int, default=8,
                   help="Maximum K for node-level concept search")

    # Synthetic options
    p.add_argument("--synthetic", action="store_true",
                   help="Dataset is synthetic — load ground-truth signal for validation")
    p.add_argument("--regime", type=int, default=None,
                   help="Synthetic regime number (1-4), used in output filenames")

    return p.parse_args()


def load_model(args, dataset, device):
    dim_clinical = dataset.n_clinical_features if args.clinical else 0

    state = torch.load(args.model_path, map_location=device)

    # Infer the bottleneck's NND/Ripley width from the checkpoint itself, so
    # checkpoints trained with --no_spatial_features (dim_nnd=0) load correctly
    # without the caller needing to know how the model was trained.
    dim_nnd = infer_dim_nnd_features(state, args.hidden_dim, dim_clinical)
    if dim_nnd not in (0, len(SPATIAL_FEATURE_NAMES)):
        print(f"  Warning: inferred dim_nnd_features={dim_nnd}, expected 0 or "
              f"{len(SPATIAL_FEATURE_NAMES)} — checkpoint/args may be mismatched.")
    print(f"  Inferred bottleneck: dim_nnd_features={dim_nnd}, "
          f"dim_clinical_features={dim_clinical}")

    model = create_model(
        args.hidden_dim,
        dim_nnd_features=dim_nnd,
        dim_clinical_features=dim_clinical,
        n_layers=args.n_layers,
        dim_features=dataset.dim_features,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"  {len(df):,} cells, {df['core'].nunique()} cores, {df['PID'].nunique()} patients")

    if args.synthetic and "signal" in df.columns:
        sig = df.groupby("PID")["signal"].first()
        print(f"  Synthetic signal: [{sig.min():.2f}, {sig.max():.2f}]")

    # ── Build dataset ──────────────────────────────────────────────────────────
    print("Building PyG dataset...")
    dataset = create_pytorch_dataset(
        df,
        endpoint=args.endpoint,
        k=args.k,
        max_cells=args.max_cells,
        normalize_coords=True,
    )
    print(f"  {len(dataset)} graphs")

    # Train/val split (same as training); the analyses evaluate on val patients
    train_idx, val_idx = patient_split(dataset, val_size=0.2, random_state=args.seed)
    train_pids = {dataset.data[dataset.cores[i]]["PID"] for i in train_idx}
    val_pids   = {dataset.data[dataset.cores[i]]["PID"] for i in val_idx}
    print(f"  Val set: {len(val_pids)} patients, {len(val_idx)} cores")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args, dataset, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    # Suffix for output filenames
    suffix_parts = [args.endpoint.lower()]
    if args.synthetic:
        suffix_parts.append(f"synthetic_r{args.regime}" if args.regime else "synthetic")
    suffix = "_".join(suffix_parts)

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_params({
            "model_path": args.model_path,
            "data":       args.data,
            "endpoint":   args.endpoint,
            "synthetic":  args.synthetic,
            "regime":     args.regime,
            "seed":       args.seed,
        })

        # ── 1. Block-permutation ablation ─────────────────────────────────────
        if not args.no_block_ablation:
            print("\n── Block-permutation ablation ──────────────────────────────────")
            abl_summary = run_block_ablation(
                model=model,
                dataset=dataset,
                device=device,
                val_pids=val_pids,
                output_dir=args.output_dir,
                suffix=suffix,
                n_perm=args.n_perm,
            )
            if abl_summary:
                for cond, vals in abl_summary.items():
                    if not cond.endswith("_zeroed"):
                        mlflow.log_metric(f"block_abl_{cond}_mean", vals["mean"])
                        if vals["sd"] > 0:
                            mlflow.log_metric(f"block_abl_{cond}_sd", vals["sd"])

        # ── 2. Per-feature permutation importance ─────────────────────────────
        if not args.no_feature_perm:
            print("\n── Per-feature permutation importance ──────────────────────────")
            fp_summary = run_feature_permutation(
                model=model,
                dataset=dataset,
                device=device,
                val_pids=val_pids,
                output_dir=args.output_dir,
                suffix=suffix,
                n_perm=args.n_perm,
            )
            if fp_summary:
                mlflow.log_metric("feat_perm_baseline", fp_summary["baseline"])
                for feat, vals in fp_summary["features"].items():
                    mlflow.log_metric(f"feat_perm_drop_{feat}", vals["drop"])

        # ── 3. Graph-level concept discovery ──────────────────────────────────
        if not args.no_graph_concepts:
            print("\n── Graph-level concept discovery ───────────────────────────────")
            k_range = (args.graph_k_min, args.graph_k_max)
            gc_summary = run_graph_concept_discovery(
                model=model,
                dataset=dataset,
                device=device,
                raw_df=df,
                train_pids=train_pids,
                val_pids=val_pids,
                output_dir=args.output_dir,
                suffix=suffix,
                k_range=k_range,
            )
            if gc_summary:
                mlflow.log_metric("graph_concepts_best_k",      gc_summary["best_k"])
                mlflow.log_metric("graph_concepts_completeness", gc_summary["completeness"])
                if gc_summary["completeness_signal"] is not None:
                    mlflow.log_metric("graph_concepts_completeness_signal",
                                       gc_summary["completeness_signal"])

        # ── 4. Node-level concept discovery ───────────────────────────────────
        if not args.no_node_concepts:
            print("\n── Node-level concept discovery ────────────────────────────────")
            k_range = (args.node_k_min, args.node_k_max)
            nc_summary = run_node_concept_discovery(
                model=model,
                dataset=dataset,
                device=device,
                train_pids=train_pids,
                val_pids=val_pids,
                output_dir=args.output_dir,
                suffix=suffix,
                k_range=k_range,
            )
            if nc_summary:
                mlflow.log_metric("node_concepts_best_k",      nc_summary["best_k"])
                mlflow.log_metric("node_concepts_completeness", nc_summary["completeness"])

        # ── 5. Embedding probe (synthetic only) ───────────────────────
        if not args.no_embedding_probe and args.synthetic:
            print("\n── Embedding residual probe ────────────────────────────────────")
            probe_summary = run_embedding_probe(
                model=model,
                dataset=dataset,
                device=device,
                raw_df=df,
                train_pids=train_pids,
                val_pids=val_pids,
                output_dir=args.output_dir,
                suffix=suffix,
                title=f"Embedding probe, Regime {args.regime}" if args.regime else "Embedding probe",
            )
            if probe_summary:
                mlflow.log_metrics({
                    "probe_r2_classical":  probe_summary["r2_classical"],
                    "probe_r2_raw_emb":    probe_summary["r2_raw_emb"],
                    "probe_r2_residual":   probe_summary["r2_residual"],
                    "probe_pc1r_classical":probe_summary["pc1_r_classical"],
                    "probe_pc1r_raw_emb":  probe_summary["pc1_r_raw_emb"],
                    "probe_pc1r_residual": probe_summary["pc1_r_residual"],
                })

    print(f"\nDone. Figures saved to: {args.output_dir}/")
    print("View MLflow: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
