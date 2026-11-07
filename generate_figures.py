"""
Generate the dataset, model and baseline figures used in the thesis.

    Fig 1  — example tissue core and its k-NN graph   (fig1_core_knn_graph)
    Fig 2  — dataset overview                         (fig2_dataset_overview)
    Fig 3  — train/validation split balance           (fig3_split_balance)
    Fig 5  — GNN model architecture schematic         (fig5_model_architecture)
    Fig 6b — clinical Cox PH forest plot              (fig6b_cox_forest_<endpoint>)

The model comparison (fig7_model_comparison_<endpoint>) is produced by
run_cv.py, the pretraining figure (fig8_pretraining_simple) by
analysis/pretrain_eval.py, and the explainability figures by the scripts in
analysis/explainability/.

Usage:
    uv run python generate_figures.py
    uv run python generate_figures.py --figs 1 2   # only Fig 1 and 2
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "figure.dpi":         150,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "output")
ALL_FIGS = ["1", "2", "3", "5", "6b"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument(
        "--data",
        default="datasets/point_patterns_metaclusters.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--endpoint",
        default="DFS",
        choices=["DFS", "OS"],
        help="Survival endpoint for Figure 6b (default: DFS)",
    )
    parser.add_argument(
        "--figs",
        nargs="+",
        default=ALL_FIGS,
        choices=ALL_FIGS,
        help="Which figures to generate (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the 80/20 patient split (must match training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    figs = set(args.figs)

    print(f"Loading {args.data}...")
    df = pd.read_parquet(args.data)
    print(
        f"  {len(df):,} cells  ·  {df['core'].nunique()} cores  ·  "
        f"{df['PID'].nunique()} patients"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if "1" in figs:
        print("\n── Figure 1: Example tissue core and k-NN graph ─────────────────")
        from figures.dataset_figs import plot_core_and_knn_graph
        plot_core_and_knn_graph(df, OUTPUT_DIR)

    if "2" in figs:
        print("\n── Figure 2: Dataset overview ───────────────────────────────────")
        from figures.dataset_figs import plot_dataset_overview
        plot_dataset_overview(df, OUTPUT_DIR)

    if "3" in figs:
        print("\n── Figure 3: Split balance ──────────────────────────────────────")
        from figures.dataset_figs import plot_split_balance
        plot_split_balance(df, OUTPUT_DIR, seed=args.seed)

    if "5" in figs:
        print("\n── Figure 5: Model architecture ─────────────────────────────────")
        from figures.dataset_figs import plot_model_architecture
        plot_model_architecture(OUTPUT_DIR)

    if "6b" in figs:
        print("\n── Figure 6b: Cox forest plot ───────────────────────────────────")
        from figures.baseline_figs import plot_cox_forest
        plot_cox_forest(df, OUTPUT_DIR, endpoint=args.endpoint, seed=args.seed)

    print(f"\nDone. Figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
