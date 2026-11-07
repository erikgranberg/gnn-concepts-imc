"""
Combined graph-level K-search figure (Appendix D).

Standalone: recomputes the graph-level K search exactly as
concept_discovery.run_graph_concept_discovery does (same embeddings, same
train/val split, same search_k call with the same completeness targets), for
the four synthetic regimes and the IMC data, saves the scores to CSV and draws
all five panels in one figure with a single shared legend.

    uv run python analysis/explainability/k_search_combined.py
    uv run python analysis/explainability/k_search_combined.py --plot_only
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_utils import create_pytorch_dataset, patient_split
from run_explainability import load_model, get_device
from concept_discovery import extract_all_embeddings, search_k

OUTPUT_DIR = "figures/output/explainability"
CSV_PATH   = "results/graph_k_search_dfs.csv"

# (panel title, model checkpoint, data, clinical features at the bottleneck)
CONFIGS = [
    ("R1, Composition",            "models/synthetic_regime1_noise0.0.pt",
     "datasets/synthetic_regime1_noise0.0.parquet", False),
    ("R2, Proximity",              "models/synthetic_regime2_noise0.0.pt",
     "datasets/synthetic_regime2_noise0.0.parquet", False),
    ("R3, Anchored clustering",    "models/synthetic_regime3_noise0.0.pt",
     "datasets/synthetic_regime3_noise0.0.parquet", False),
    ("R4, Independent clustering", "models/synthetic_regime4_noise0.0.pt",
     "datasets/synthetic_regime4_noise0.0.parquet", False),
    ("Real IMC data, DFS",         "best_model.pt",
     "datasets/point_patterns_metaclusters.parquet", True),
]

SIL_COLOR  = "#4575b4"
COMP_COLOR = {"risk": "#7fbc41", "signal": "#9e3d8c"}   # as in _plot_k_search


def compute(device, k_range=(2, 8), seed=42):
    rows = []
    for title, model_path, data, clinical in CONFIGS:
        print(f"\n── {title} ──")
        df = pd.read_parquet(data)
        dataset = create_pytorch_dataset(df, endpoint="DFS", k=12,
                                         max_cells=None, normalize_coords=True)
        train_idx, _ = patient_split(dataset, val_size=0.2, random_state=seed)
        train_pids = {dataset.data[dataset.cores[i]]["PID"] for i in train_idx}
        args = SimpleNamespace(model_path=model_path, hidden_dim=32,
                               n_layers=3, clinical=clinical)
        model = load_model(args, dataset, device)

        # Mirrors run_graph_concept_discovery
        emb_data = extract_all_embeddings(model, dataset, device)
        cores = list(emb_data["graph_emb"])
        G     = np.array([emb_data["graph_emb"][c] for c in cores])
        risks = np.array([emb_data["risk"][c] for c in cores])
        pids  = [emb_data["pid"][c] for c in cores]
        train_mask = np.array([p in train_pids for p in pids])
        val_mask   = ~train_mask

        targets = {"risk": (risks, train_mask, val_mask)}
        if "signal" in df.columns:
            sig_map = df.groupby("PID")["signal"].first().to_dict()
            sigs = np.array([sig_map.get(p, np.nan) for p in pids])
            sig_mask = ~np.isnan(sigs)
            if (train_mask & sig_mask).sum() >= 10 and (val_mask & sig_mask).sum() >= 5:
                targets["signal"] = (sigs, train_mask & sig_mask, val_mask & sig_mask)

        best_k, sil, comp = search_k(G, k_range=k_range, n_init=10,
                                     completeness_targets=targets)
        print(f"  best K={best_k}  completeness→risk={comp['risk'][best_k]:.3f}"
              + (f"  →s={comp['signal'][best_k]:.3f}" if "signal" in comp else ""))
        for k in sorted(sil):
            rows.append({"panel": title, "k": k, "best_k": best_k,
                         "silhouette": sil[k],
                         "completeness_risk": comp["risk"][k],
                         "completeness_signal": comp.get("signal", {}).get(k, np.nan)})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    out.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {CSV_PATH}")
    return out


def plot(res: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for ax, (title, *_ ) in zip(axes.flat, CONFIGS):
        d = res[res["panel"] == title].sort_values("k")
        best_k = int(d["best_k"].iloc[0])
        ax.plot(d["k"], d["silhouette"], marker="o", ms=4, color=SIL_COLOR)
        ax.axvline(best_k, color="grey", linestyle=":", linewidth=1.5)
        ax.set_title(f"{title} (chosen $K = {best_k}$)", fontsize=10)
        ax.set_xticks(d["k"])
        ax.tick_params(axis="y", labelcolor=SIL_COLOR, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax2 = ax.twinx()
        for name in ("risk", "signal"):
            col = f"completeness_{name}"
            if d[col].notna().any():
                ax2.plot(d["k"], d[col], marker="s", ms=4, linestyle="--",
                         color=COMP_COLOR[name])
        ax2.tick_params(labelsize=8)
        ax2.spines["top"].set_visible(False)
    for ax in axes[1, :2]:
        ax.set_xlabel("K (number of concepts)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Silhouette score", color=SIL_COLOR, fontsize=9)
    axes[0, 2].set_xlabel("K (number of concepts)", fontsize=9)

    # Sixth cell: shared legend
    leg_ax = axes[1, 2]
    leg_ax.axis("off")
    handles = [
        Line2D([], [], color=SIL_COLOR, marker="o", ms=4, label="Silhouette (left axis)"),
        Line2D([], [], color=COMP_COLOR["risk"], marker="s", ms=4, linestyle="--",
               label="Completeness $R^2$ → risk\n(right axis, held-out)"),
        Line2D([], [], color=COMP_COLOR["signal"], marker="s", ms=4, linestyle="--",
               label="Completeness $R^2$ → $s$\n(right axis, held-out)"),
        Line2D([], [], color="grey", linestyle=":", linewidth=1.5, label="Chosen $K$"),
    ]
    leg_ax.legend(handles=handles, loc="center", fontsize=9, frameon=False)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"fig_graph_concepts_k_search_combined_dfs.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot_only", action="store_true",
                   help=f"Replot from {CSV_PATH} without recomputing")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    res = pd.read_csv(CSV_PATH) if args.plot_only else compute(get_device(args.device))
    plot(res)


if __name__ == "__main__":
    main()
