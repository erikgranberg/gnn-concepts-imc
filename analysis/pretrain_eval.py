"""
Evaluate the masked-pretraining encoder on its own pretraining task → Figure 8.

Masks 15% of cells (as in pretraining) and compares the pretrained model's
reconstruction of the masked cells against training-free baselines:

    cell type   majority class | untrained (random-init) GNN |
                neighbour majority vote | pretrained GNN
    position    core mean      | untrained GNN |
                neighbour mean | pretrained GNN

Neighbour baselines only use unmasked neighbours in the same k-NN graph the
GNN sees, so they show what is available from local context without learning.
Pretraining used every core (no held-out split), so this is in-sample; with
~6k parameters and ~1.2M cells, overfitting is not a concern.

Panel A plots masked cell-type accuracy over pretraining epochs, read from the
matching MLflow run in the `pretrain-masked` experiment.

Usage:
    uv run python -m analysis.pretrain_eval --pretrained pretrained_metacluster_nocap.pt --k 12
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import mlflow
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from data_utils import create_pytorch_dataset
from models.consts import BATCH_KEY_NODE_FEATURES
from pretraining.masked_pretrain import MaskedPretrainEncoder
from figures import save_fig

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

METHOD_COLORS = {
    "Majority class":  "#cccccc",
    "Core mean":       "#cccccc",
    "Untrained GNN":   "#aaaaaa",
    "Neighbour vote":  "#4878CF",
    "Neighbour mean":  "#4878CF",
    "Pretrained GNN":  "#D65F5F",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate masked pretraining → Figure 8")
    p.add_argument("--pretrained", default="pretrained_metacluster_nocap.pt")
    p.add_argument("--data", default="datasets/point_patterns_metaclusters.parquet")
    p.add_argument("--k", type=int, default=12, help="k-NN graph k used in pretraining")
    p.add_argument("--max_cells", type=int, default=None)
    p.add_argument("--mask_ratio", type=float, default=0.15)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Random masks to average over")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="figures/output")
    p.add_argument("--results_csv", default="results/pretrain_eval.csv")
    p.add_argument("--simple_only", action="store_true",
                   help="Skip evaluation; draw only the one-panel learning curve "
                        "(fig8_pretraining_simple) from --results_csv and MLflow")
    p.add_argument("--no_neighbour_line", action="store_true",
                   help="Leave the neighbour-vote reference line out of the simple figure")
    return p.parse_args()


def plot_simple(curve, majority_acc, majority_name, neighbour_acc, output_dir):
    """One-panel figure: masked cell-type accuracy over pretraining epochs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve[0], curve[1], "o-", color=METHOD_COLORS["Pretrained GNN"],
            linewidth=2, label="Pretrained GNN")
    if neighbour_acc is not None:
        ax.axhline(neighbour_acc, color=METHOD_COLORS["Neighbour vote"], linestyle="--",
                   linewidth=1, label="Neighbour majority vote (no training)")
    ax.axhline(majority_acc, color="gray", linestyle=":", linewidth=1.2,
               label=f"Always predict {majority_name} (most common)")
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Accuracy on masked cells")
    ax.set_title("Masked cell-type prediction during pretraining")
    ax.set_ylim(0.45, 0.75)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, loc="center right")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_fig(fig, os.path.join(output_dir, "fig8_pretraining_simple.pdf"))
    plt.close(fig)


def load_models(path, device):
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]

    def build():
        return MaskedPretrainEncoder(
            dim_features=cfg["dim_features"],
            n_cell_types=cfg["n_cell_types"],
            dim_hidden=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        ).to(device).eval()

    pretrained = build()
    pretrained.load_state_dict(ckpt["model_state_dict"])
    torch.manual_seed(0)
    untrained = build()
    return pretrained, untrained, cfg


def scatter_sum(values, index, n):
    out = torch.zeros((n,) + values.shape[1:], dtype=values.dtype, device=values.device)
    return out.index_add_(0, index, values)


@torch.no_grad()
def evaluate_seed(loader, pretrained, untrained, n_types, majority, mask_ratio, seed, device):
    """Return per-masked-cell true/predicted types and squared position errors."""
    gen = torch.Generator().manual_seed(seed)
    rows = {"true": [], "pred": {m: [] for m in
                                 ("Majority class", "Untrained GNN", "Neighbour vote", "Pretrained GNN")},
            "sqerr": {m: [] for m in
                      ("Core mean", "Untrained GNN", "Neighbour mean", "Pretrained GNN")}}

    for batch in loader:
        n = batch.num_nodes
        n_mask = max(1, int(n * mask_ratio))
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.randperm(n, generator=gen)[:n_mask]] = True
        batch, mask = batch.to(device), mask.to(device)

        x = batch.x[BATCH_KEY_NODE_FEATURES]
        pos, types = x[:, :2], x[:, 2:].argmax(dim=1)
        true_type, true_pos = types[mask], pos[mask]
        rows["true"].append(true_type.cpu())

        # GNNs
        for name, model in (("Pretrained GNN", pretrained), ("Untrained GNN", untrained)):
            out = model(batch, mask)
            rows["pred"][name].append(out["pred_type"].argmax(dim=1).cpu())
            rows["sqerr"][name].append(((out["pred_pos"] - true_pos) ** 2).mean(dim=1).cpu())

        # Majority class
        rows["pred"]["Majority class"].append(torch.full_like(true_type, majority).cpu())

        # Unmasked neighbours of each node (edges src → tgt, as the GNN aggregates)
        src, tgt = batch.edge_index
        keep = ~mask[src]
        src, tgt = src[keep], tgt[keep]
        onehot = torch.nn.functional.one_hot(types[src], n_types).float()
        votes = scatter_sum(onehot, tgt, n)
        n_nb = votes.sum(dim=1)
        vote = votes.argmax(dim=1)
        vote[n_nb == 0] = majority
        rows["pred"]["Neighbour vote"].append(vote[mask].cpu())

        # Core mean of unmasked positions (per graph), and neighbour mean
        g = batch.batch
        um = ~mask
        g_sum = scatter_sum(pos[um], g[um], batch.num_graphs)
        g_cnt = scatter_sum(torch.ones(int(um.sum()), device=device), g[um], batch.num_graphs)
        core_mean = (g_sum / g_cnt.clamp(min=1).unsqueeze(1))[g]
        rows["sqerr"]["Core mean"].append(((core_mean[mask] - true_pos) ** 2).mean(dim=1).cpu())

        nb_sum = scatter_sum(pos[src], tgt, n)
        nb_mean = nb_sum / n_nb.clamp(min=1).unsqueeze(1)
        nb_mean[n_nb == 0] = core_mean[n_nb == 0]
        rows["sqerr"]["Neighbour mean"].append(((nb_mean[mask] - true_pos) ** 2).mean(dim=1).cpu())

    true = torch.cat(rows["true"]).numpy()
    preds = {m: torch.cat(v).numpy() for m, v in rows["pred"].items()}
    sqerr = {m: torch.cat(v).numpy() for m, v in rows["sqerr"].items()}
    return true, preds, sqerr


def fetch_curve(cfg, k, max_cells):
    """Masked type accuracy per epoch from the matching pretraining MLflow run."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("pretrain-masked")
    if exp is None:
        return None
    filt = (f"params.model_type = '{cfg['model_type']}' and "
            f"params.hidden_dim = '{cfg['hidden_dim']}' and "
            f"params.n_layers = '{cfg['n_layers']}' and params.k = '{k}' and "
            f"params.max_cells = '{max_cells}'")
    runs = client.search_runs([exp.experiment_id], filter_string=filt,
                              order_by=["start_time DESC"], max_results=1)
    if not runs:
        return None
    hist = client.get_metric_history(runs[0].info.run_id, "eval_type_acc")
    hist = sorted(hist, key=lambda m: m.step)
    return [m.step for m in hist], [m.value for m in hist]


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    pretrained, untrained, cfg = load_models(args.pretrained, device)

    if args.simple_only:
        res = pd.read_csv(args.results_csv).set_index("method")
        curve = fetch_curve(cfg, args.k, args.max_cells)
        if curve is None:
            raise SystemExit("No matching pretraining run in MLflow for the learning curve")
        # The majority class is always tumour here: its recall is 1 for that class
        majority_name = next(c[len("recall_"):] for c in res.columns
                             if c.startswith("recall_") and res.loc["Majority class", c] == 1)
        plot_simple(curve, res.loc["Majority class", "accuracy"], majority_name,
                    None if args.no_neighbour_line else res.loc["Neighbour vote", "accuracy"],
                    args.output_dir)
        return

    print(f"Loading {args.data} ...")
    df = pd.read_parquet(args.data)
    dataset = create_pytorch_dataset(
        df, endpoint="OS", k=args.k, max_cells=args.max_cells,
        normalize_coords=True, require_survival=False,
    )
    cell_types = list(dataset.cell_types)
    graphs = [dataset[i] for i in range(len(dataset))]
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    print(f"{len(graphs)} cores, cell types: {cell_types}")

    all_types = torch.cat([g.x[BATCH_KEY_NODE_FEATURES][:, 2:].argmax(dim=1) for g in graphs])
    freq = np.bincount(all_types.numpy(), minlength=len(cell_types)) / len(all_types)
    majority = int(freq.argmax())

    # ── Evaluate over several random masks ──────────────────────────────────
    acc, bal_acc, recall, mse = {}, {}, {}, {}
    for seed in args.seeds:
        true, preds, sqerr = evaluate_seed(loader, pretrained, untrained, len(cell_types),
                                           majority, args.mask_ratio, seed, device)
        for m, p in preds.items():
            rec = np.array([(p[true == c] == c).mean() for c in range(len(cell_types))])
            acc.setdefault(m, []).append((p == true).mean())
            bal_acc.setdefault(m, []).append(rec.mean())
            recall.setdefault(m, []).append(rec)
        for m, e in sqerr.items():
            mse.setdefault(m, []).append(e.mean())
        print(f"  seed {seed}: " + ", ".join(f"{m}={acc[m][-1]:.3f}" for m in preds))

    type_methods = ["Majority class", "Untrained GNN", "Neighbour vote", "Pretrained GNN"]
    pos_methods = ["Core mean", "Untrained GNN", "Neighbour mean", "Pretrained GNN"]

    rows = []
    for m in type_methods:
        rows.append({"task": "cell_type", "method": m,
                     "accuracy": np.mean(acc[m]), "accuracy_sd": np.std(acc[m]),
                     "balanced_accuracy": np.mean(bal_acc[m]),
                     "balanced_accuracy_sd": np.std(bal_acc[m]),
                     **{f"recall_{t}": np.mean([r[i] for r in recall[m]])
                        for i, t in enumerate(cell_types)}})
    for m in pos_methods:
        rows.append({"task": "position", "method": m,
                     "mse": np.mean(mse[m]), "mse_sd": np.std(mse[m])})
    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    res.to_csv(args.results_csv, index=False)
    print(res.round(4).to_string(index=False))
    print(f"Saved {args.results_csv}")

    # ── Figure 8 ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # A: accuracy over pretraining epochs
    curve = fetch_curve(cfg, args.k, args.max_cells)
    if curve is not None:
        ax_a.plot(curve[0], curve[1], "o-", color=METHOD_COLORS["Pretrained GNN"],
                  label="Pretrained GNN")
    else:
        ax_a.text(0.5, 0.5, "No matching MLflow run", ha="center", transform=ax_a.transAxes)
    ax_a.axhline(np.mean(acc["Neighbour vote"]), color=METHOD_COLORS["Neighbour vote"],
                 linestyle="--", label="Neighbour vote")
    ax_a.axhline(freq[majority], color="gray", linestyle=":",
                 label=f"Majority class ({cell_types[majority]})")
    ax_a.set_xlabel("Pretraining epoch")
    ax_a.set_ylabel("Masked cell-type accuracy")
    ax_a.set_title("A  Learning curve", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, loc="lower right")

    # B: accuracy and balanced accuracy
    xs = np.arange(len(type_methods))
    w = 0.38
    for off, metric, label, alpha in ((-w / 2, acc, "Accuracy", 0.9),
                                      (w / 2, bal_acc, "Balanced accuracy", 0.5)):
        vals = [np.mean(metric[m]) for m in type_methods]
        sds = [np.std(metric[m]) for m in type_methods]
        bars = ax_b.bar(xs + off, vals, w, yerr=sds, capsize=3, label=label,
                        color=[METHOD_COLORS[m] for m in type_methods], alpha=alpha,
                        edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax_b.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=8)
    ax_b.axhline(1 / len(cell_types), color="gray", linestyle=":", linewidth=1)
    ax_b.set_xticks(xs, [m.replace(" ", "\n") for m in type_methods])
    ax_b.set_ylim(0, 1)
    ax_b.set_ylabel("Masked cell-type prediction")
    ax_b.set_title("B  Cell-type reconstruction", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="upper left")

    # C: per-class recall
    xs = np.arange(len(cell_types))
    cmp = ["Neighbour vote", "Pretrained GNN"]
    for j, m in enumerate(cmp):
        vals = np.mean(recall[m], axis=0)
        ax_c.bar(xs + (j - 0.5) * w, vals, w, color=METHOD_COLORS[m], label=m,
                 edgecolor="black", linewidth=0.5)
    ax_c.set_xticks(xs, [f"{t}\n({f:.0%})" for t, f in zip(cell_types, freq)])
    ax_c.set_ylim(0, 1)
    ax_c.set_ylabel("Recall (masked cells)")
    ax_c.set_title("C  Per cell type (share of cells)", loc="left", fontweight="bold")
    ax_c.legend(frameon=False)

    # D: position reconstruction
    xs = np.arange(len(pos_methods))
    vals = [np.mean(mse[m]) for m in pos_methods]
    bars = ax_d.bar(xs, vals, 0.6, color=[METHOD_COLORS[m] for m in pos_methods],
                    edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax_d.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.4f}",
                  ha="center", va="bottom", fontsize=8)
    ax_d.set_yscale("log")
    ax_d.set_xticks(xs, [m.replace(" ", "\n") for m in pos_methods])
    ax_d.set_ylabel("Position MSE (normalised coords, log)")
    ax_d.set_title("D  Position reconstruction", loc="left", fontweight="bold")

    fig.suptitle(f"Masked pretraining task ({int(args.mask_ratio * 100)}% of cells masked, "
                 f"mean ± SD over {len(args.seeds)} masks)", fontsize=12)
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    save_fig(fig, os.path.join(args.output_dir, "fig8_pretraining.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
