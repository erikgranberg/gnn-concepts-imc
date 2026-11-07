"""
Task 3: Block-permutation ablation at the GNN bottleneck.

Measures the contribution of each feature block to survival prediction
(C-index) by permuting blocks across patients. Permutation preserves each
block's marginal distribution while destroying its correlation with survival,
so the C-index drop measures that block's predictive contribution.

Blocks:
  GCN embedding   — graph_emb (dim_latent dims from global pooling)
  NND features    — first 5 of nnd_features (median/mean NND, frac_adj/comm, n_immune)
  Ripley features — last 4 of nnd_features (K at 20/50/100/200 µm)

Conditions evaluated:
  full          — no ablation (baseline)
  emb_ablated   — permute GCN embedding  (model relies on NND+Ripley only)
  nnd_ablated   — permute NND features   (model relies on embedding+Ripley)
  ripley_ablated— permute Ripley features (model relies on embedding+NND)
  stats_ablated — permute NND+Ripley     (model relies on GCN embedding only)

Per-regime expected ordering (synthetic validation):
  R1 composition : emb_ablated ≈ full  (composition visible in NND/Ripley too)
  R2 proximity   : nnd_ablated drops most
  R3/R4 clustering: stats_ablated drops least if GNN captures clustering signal
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.consts import BATCH_KEY_GRAPH_EMBEDDING
from data_utils import NND_FEATURE_NAMES, RIPLEY_FEATURE_NAMES

N_NND    = len(NND_FEATURE_NAMES)     # 5
N_RIPLEY = len(RIPLEY_FEATURE_NAMES)  # 4


# ── Bottleneck extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_bottleneck(model, dataset, device) -> dict:
    """
    One forward pass per graph. Returns pid → dict with:
      emb      : (D_emb,)     GCN graph embedding
      nnd      : (5,)         NND spatial features (normalized)
      ripley   : (4,)         Ripley spatial features (normalized)
      clinical : (D_clin,)    clinical features (normalized); always extracted
                              from the dataset, whether or not the model was
                              trained with --clinical (see _predict_from_bottleneck)
      time     : float
      event    : float
    Multiple cores per patient are averaged.
    """
    model.eval()
    pid_emb:      dict = {}
    pid_nnd:      dict = {}
    pid_ripley:   dict = {}
    pid_clinical: dict = {}
    pid_labels:   dict = {}

    for idx in range(len(dataset)):
        g   = dataset[idx].to(device)
        pid = dataset.data[dataset.cores[idx]]["PID"]

        out = model(g)
        emb = out[BATCH_KEY_GRAPH_EMBEDDING].squeeze(0).cpu().numpy()

        sf = g.nnd_features.squeeze(0).cpu().numpy()  # length N_NND + N_RIPLEY
        nnd    = sf[:N_NND]
        ripley = sf[N_NND:]

        clinical = g.clinical_features.squeeze(0).cpu().numpy()

        label = g.y["graph_labels"].view(-1, 2)[0]
        t, e  = float(label[0]), float(label[1])

        pid_emb.setdefault(pid,      []).append(emb)
        pid_nnd.setdefault(pid,      []).append(nnd)
        pid_ripley.setdefault(pid,   []).append(ripley)
        pid_clinical.setdefault(pid, []).append(clinical)
        pid_labels[pid] = (t, e)

    result = {}
    for pid in pid_emb:
        t, e = pid_labels[pid]
        result[pid] = {
            "emb":      np.mean(pid_emb[pid],      axis=0),
            "nnd":      np.mean(pid_nnd[pid],      axis=0),
            "ripley":   np.mean(pid_ripley[pid],   axis=0),
            "clinical": np.mean(pid_clinical[pid], axis=0),
            "time":     t,
            "event":    e,
        }
    return result


# ── Prediction from bottleneck ────────────────────────────────────────────────

@torch.no_grad()
def _predict_from_bottleneck(model, emb: np.ndarray, nnd: np.ndarray,
                              ripley: np.ndarray, device,
                              clinical: np.ndarray | None = None) -> np.ndarray:
    """
    Run model.output_head on pre-assembled bottleneck rows.

    emb      : (N, D_emb)
    nnd      : (N, N_NND)
    ripley   : (N, N_RIPLEY)
    clinical : (N, D_clin) or None. Passed through unmodified (never permuted
               or ablated) whenever the model's output_head expects it — this
               ablation only targets the GCN embedding / NND / Ripley blocks,
               clinical features are outside its scope and held fixed at their
               true values in every condition, matching how Figure 7 treats
               "Clinical" as a separate additive baseline rather than one of
               the ablated spatial blocks.
    Returns risk scores (N,).
    """
    parts = [emb, nnd, ripley]
    expected_dim = model.output_head[0].in_features
    if clinical is not None and emb.shape[1] + nnd.shape[1] + ripley.shape[1] + clinical.shape[1] == expected_dim:
        parts.append(clinical)
    bottleneck = np.concatenate(parts, axis=1)
    t = torch.tensor(bottleneck, dtype=torch.float32, device=device)
    risk = model.output_head(t).squeeze(-1).cpu().numpy()
    return risk


# ── Permutation conditions ────────────────────────────────────────────────────

def _apply_condition(emb, nnd, ripley, condition: str,
                     rng: np.random.Generator) -> tuple:
    """
    Return (emb, nnd, ripley) with the appropriate block(s) permuted.
    Permutation shuffles patient-order for that block (preserves marginals).
    """
    n = len(emb)

    def _perm(X):
        return X[rng.permutation(n)]

    if condition == "full":
        return emb, nnd, ripley
    elif condition == "emb_ablated":
        return _perm(emb), nnd, ripley
    elif condition == "nnd_ablated":
        return emb, _perm(nnd), ripley
    elif condition == "ripley_ablated":
        return emb, nnd, _perm(ripley)
    elif condition == "stats_ablated":
        return emb, _perm(nnd), _perm(ripley)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ── Mean-imputation cross-check ───────────────────────────────────────────────

def _apply_mean_imputation(emb, nnd, ripley, condition: str) -> tuple:
    """
    Deterministic ablation: replace a block with its column means (≈ zeros
    for normalized features). Cross-check for permutation results.
    """
    def _zero(X):
        return np.zeros_like(X)

    if condition == "full":
        return emb, nnd, ripley
    elif condition == "emb_ablated":
        return _zero(emb), nnd, ripley
    elif condition == "nnd_ablated":
        return emb, _zero(nnd), ripley
    elif condition == "ripley_ablated":
        return emb, nnd, _zero(ripley)
    elif condition == "stats_ablated":
        return emb, _zero(nnd), _zero(ripley)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ── Main analysis ─────────────────────────────────────────────────────────────

CONDITIONS = ["full", "emb_ablated", "nnd_ablated", "ripley_ablated", "stats_ablated"]

CONDITION_LABELS = {
    "full":           "Full model",
    "emb_ablated":    "Emb ablated\n(stats only)",
    "nnd_ablated":    "NND ablated",
    "ripley_ablated": "Ripley ablated",
    "stats_ablated":  "Stats ablated\n(emb only)",
}


def run_block_ablation(
    model,
    dataset,
    device,
    val_pids,
    output_dir: str,
    suffix:     str  = "",
    n_perm:     int  = 50,
    cross_check: bool = True,
) -> dict | None:
    """
    Full block-permutation ablation pipeline.

    val_pids: patient IDs to evaluate on (held-out set only).
    Returns summary dict with mean/SD C-index per condition.
    """
    if getattr(model, "dim_nnd_features", 0) == 0:
        print("  Model has no NND/Ripley features at the bottleneck "
              "(trained with --no_spatial_features) — skipping block ablation.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    print(f"  Extracting bottleneck components ({len(dataset)} graphs)...")
    pid_data = extract_bottleneck(model, dataset, device)

    # Keep only val patients
    val_data = {p: d for p, d in pid_data.items() if p in val_pids}
    if len(val_data) < 5:
        print("  Too few val patients — skipping block ablation.")
        return None

    pids     = list(val_data)
    emb      = np.array([val_data[p]["emb"]      for p in pids])
    nnd      = np.array([val_data[p]["nnd"]      for p in pids])
    ripley   = np.array([val_data[p]["ripley"]   for p in pids])
    clinical = np.array([val_data[p]["clinical"] for p in pids])
    times    = np.array([val_data[p]["time"]     for p in pids])
    events   = np.array([val_data[p]["event"]    for p in pids])

    if events.sum() < 2:
        print("  Fewer than 2 events in val — skipping.")
        return None

    print(f"  Val patients: {len(pids)} | events: {int(events.sum())}")
    print(f"  Embedding dim: {emb.shape[1]} | NND: {nnd.shape[1]} | Ripley: {ripley.shape[1]}")

    rng = np.random.default_rng(0)
    summary = {}

    for cond in CONDITIONS:
        if cond == "full":
            # Single deterministic evaluation
            risk = _predict_from_bottleneck(model, emb, nnd, ripley, device, clinical)
            ci   = concordance_index(times, -risk, events)
            summary[cond] = {"mean": ci, "sd": 0.0, "all": [ci]}
            print(f"  {cond:<18}  C-index = {ci:.4f}")
        else:
            # Permutation replicates
            cis = []
            for _ in range(n_perm):
                e_p, n_p, r_p = _apply_condition(emb, nnd, ripley, cond, rng)
                risk = _predict_from_bottleneck(model, e_p, n_p, r_p, device, clinical)
                cis.append(concordance_index(times, -risk, events))
            mean_ci, sd_ci = float(np.mean(cis)), float(np.std(cis))
            summary[cond] = {"mean": mean_ci, "sd": sd_ci, "all": cis}
            print(f"  {cond:<18}  C-index = {mean_ci:.4f} ± {sd_ci:.4f}")

    # ── Mean-imputation cross-check ────────────────────────────────────────────
    if cross_check:
        print("\n  Cross-check (mean-imputation, deterministic):")
        for cond in CONDITIONS[1:]:   # skip "full"
            e_p, n_p, r_p = _apply_mean_imputation(emb, nnd, ripley, cond)
            risk = _predict_from_bottleneck(model, e_p, n_p, r_p, device, clinical)
            ci   = concordance_index(times, -risk, events)
            summary[f"{cond}_zeroed"] = {"mean": ci, "sd": 0.0, "all": [ci]}
            print(f"  {cond:<18}  C-index = {ci:.4f}  (zeroed)")

    # ── Figure ─────────────────────────────────────────────────────────────────
    _save_ablation_figure(summary, output_dir, suffix)

    return summary


def _save_ablation_figure(summary: dict, output_dir: str, suffix: str):
    conditions = CONDITIONS
    labels = [CONDITION_LABELS[c] for c in conditions]
    means  = [summary[c]["mean"] for c in conditions]
    sds    = [summary[c]["sd"]   for c in conditions]

    colors = ["#2c7bb6", "#d7191c", "#fdae61", "#abd9e9", "#1a9641"]
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, means, yerr=sds, color=colors, width=0.6, alpha=0.85,
                  capsize=4, error_kw={"linewidth": 1.2})

    # Reference line at 0.5 (random)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="Random (0.5)")

    # Annotate full-model value with a horizontal dashed line
    full_ci = summary["full"]["mean"]
    ax.axhline(full_ci, color=colors[0], linewidth=0.8, linestyle=":",
               label=f"Full model ({full_ci:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("C-index (val set)")
    ax.set_ylim(max(0, min(means) - 0.15), min(1, max(means) + 0.1))
    ax.set_title("Block-permutation ablation\nmean ± SD C-index over permutation replicates")
    ax.legend(fontsize=8, framealpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fname = f"fig_block_ablation{('_' + suffix) if suffix else ''}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {fname}.pdf/png")


# ── Per-feature permutation importance ────────────────────────────────────────

def run_feature_permutation(
    model,
    dataset,
    device,
    val_pids,
    output_dir: str,
    suffix:     str = "",
    n_perm:     int = 50,
) -> dict | None:
    """
    Per-feature permutation importance for the 9 spatial features at the
    GNN bottleneck (5 NND + 4 Ripley).

    Each feature is permuted independently across patients while all other
    features and the GCN embedding are held fixed. The C-index drop relative
    to the full model measures that feature's contribution.

    Returns a dict: {feature_name: {"mean": float, "sd": float, "drop": float}}
    """
    if getattr(model, "dim_nnd_features", 0) == 0:
        print("  Model has no NND/Ripley features at the bottleneck "
              "(trained with --no_spatial_features) — skipping feature permutation.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    print(f"  Extracting bottleneck ({len(dataset)} graphs)...")
    pid_data = extract_bottleneck(model, dataset, device)

    val_data = {p: d for p, d in pid_data.items() if p in val_pids}
    if len(val_data) < 5:
        print("  Too few val patients — skipping feature permutation.")
        return None

    pids     = list(val_data)
    emb      = np.array([val_data[p]["emb"]      for p in pids])
    nnd      = np.array([val_data[p]["nnd"]      for p in pids])
    ripley   = np.array([val_data[p]["ripley"]   for p in pids])
    clinical = np.array([val_data[p]["clinical"] for p in pids])
    times    = np.array([val_data[p]["time"]     for p in pids])
    events   = np.array([val_data[p]["event"]    for p in pids])

    if events.sum() < 2:
        print("  Fewer than 2 events — skipping.")
        return None

    # Baseline C-index (full model, no permutation)
    risk_full = _predict_from_bottleneck(model, emb, nnd, ripley, device, clinical)
    baseline  = concordance_index(times, -risk_full, events)
    print(f"  Baseline C-index: {baseline:.4f}")
    print(f"  Val patients: {len(pids)} | events: {int(events.sum())}\n")

    feature_names = NND_FEATURE_NAMES + RIPLEY_FEATURE_NAMES   # 9 total
    rng = np.random.default_rng(0)
    results = {}

    for feat_idx, feat_name in enumerate(feature_names):
        cis = []
        for _ in range(n_perm):
            nnd_p    = nnd.copy()
            ripley_p = ripley.copy()
            perm     = rng.permutation(len(pids))

            if feat_idx < N_NND:
                nnd_p[:, feat_idx] = nnd[perm, feat_idx]
            else:
                ripley_p[:, feat_idx - N_NND] = ripley[perm, feat_idx - N_NND]

            risk_p = _predict_from_bottleneck(model, emb, nnd_p, ripley_p, device, clinical)
            cis.append(concordance_index(times, -risk_p, events))

        mean_ci = float(np.mean(cis))
        sd_ci   = float(np.std(cis))
        drop    = baseline - mean_ci
        results[feat_name] = {"mean": mean_ci, "sd": sd_ci, "drop": drop}
        print(f"  {feat_name:<26}  drop = {drop:+.4f}  "
              f"(C = {mean_ci:.4f} ± {sd_ci:.4f})")

    _save_feature_perm_figure(results, baseline, output_dir, suffix)
    return {"baseline": baseline, "features": results}


def _save_feature_perm_figure(results: dict, baseline: float,
                               output_dir: str, suffix: str):
    # Sort by descending importance (largest drop first)
    sorted_feats = sorted(results, key=lambda f: results[f]["drop"], reverse=True)
    drops = np.array([results[f]["drop"] for f in sorted_feats])
    sds   = np.array([results[f]["sd"]   for f in sorted_feats])

    colors = [
        "#d7191c" if f in NND_FEATURE_NAMES else "#2c7bb6"
        for f in sorted_feats
    ]

    n = len(sorted_feats)
    fig, ax = plt.subplots(figsize=(6, max(3.5, n * 0.45)))
    y = np.arange(n)

    ax.barh(y, drops, xerr=sds, color=colors, alpha=0.85, height=0.6,
            capsize=3, error_kw={"linewidth": 1.1})
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(baseline - 0.5, color="white", linewidth=0)  # invisible; sets x-margin

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_feats, fontsize=9)
    ax.set_xlabel("C-index drop  (baseline − permuted)", fontsize=10)
    ax.set_title(
        f"Per-feature permutation importance\n"
        f"baseline C = {baseline:.3f}  |  mean ± SD over {50} permutations",
        fontsize=9,
    )

    nnd_patch = mpatches.Patch(color="#d7191c", alpha=0.85, label="NND features")
    rip_patch = mpatches.Patch(color="#2c7bb6", alpha=0.85, label="Ripley features")
    ax.legend(handles=[nnd_patch, rip_patch], fontsize=8, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fname = f"fig_feature_importance{('_' + suffix) if suffix else ''}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {fname}.pdf/png")
