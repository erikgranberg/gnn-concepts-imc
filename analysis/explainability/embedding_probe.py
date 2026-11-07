"""
Task 4: Embedding residual characterization.

Tests whether the GCN embedding contains information about the ground-truth
spatial signal (σ_thomas in R3) that is orthogonal to classical spatial
statistics (NND + Ripley's K features).

Steps:
1. Extract 32-dim GCN graph embeddings for all patients.
2. Build classical feature matrix from each graph's stored spatial features
   (5 NND + 4 Ripley's K = 9 features, already normalized by the dataset).
3. Residualize: regress each embedding dimension on classical features
   (RidgeCV, fit on train patients), take residuals on val patients.
4. Probe: fit RidgeCV on train, evaluate R² on val for three feature sets:
     - classical features alone
     - raw embedding
     - residual embedding (orthogonal to classical)
5. Report comparison with 95% bootstrap CIs and save figure.

On R3 (clustering regime) the expected result is:
    classical R² ≈ 0  (NND + Ripley's K are blind to σ_thomas by design)
    raw embedding R² >> 0  (GNN captures clustering scale)
    residual R²  >> 0  (the clustering signal is orthogonal to classical features)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.consts import BATCH_KEY_GRAPH_EMBEDDING
from data_utils import SPATIAL_FEATURE_NAMES, NND_FEATURE_NAMES


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_patient_embeddings(model, dataset, device) -> dict:
    """
    Return pid → {"embedding": (D,), "spatial_features": (F,)}.

    embedding: GCN graph embedding (before NND concat), averaged over cores.
    spatial_features: normalized NND+Ripley features from the graph object,
                      averaged over cores.
    """
    model.eval()
    pid_emb: dict = {}
    pid_sf: dict = {}

    for idx in range(len(dataset)):
        g = dataset[idx].to(device)
        pid = dataset.data[dataset.cores[idx]]["PID"]

        out = model(g)
        emb = out[BATCH_KEY_GRAPH_EMBEDDING].squeeze(0).cpu().numpy()
        sf = g.nnd_features.squeeze(0).cpu().numpy()

        pid_emb.setdefault(pid, []).append(emb)
        pid_sf.setdefault(pid, []).append(sf)

    return {
        pid: {
            "embedding":        np.mean(pid_emb[pid], axis=0),
            "spatial_features": np.mean(pid_sf[pid],  axis=0),
        }
        for pid in pid_emb
    }


# ── Residualization ────────────────────────────────────────────────────────────

_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def residualize_embedding(
    train_emb: np.ndarray,     # (n_train, D)
    train_sf:  np.ndarray,     # (n_train, F)
    target_emb: np.ndarray,    # (n_target, D)
    target_sf:  np.ndarray,    # (n_target, F)
) -> np.ndarray:
    """
    For each embedding dimension, fit RidgeCV(spatial_features → embedding_dim)
    on train patients, then return target_emb - predicted(target_sf).

    When called with train as target, returns in-sample residuals (for probe
    fitting). When called with val as target, returns out-of-sample residuals.
    """
    residuals = np.zeros_like(target_emb)
    for dim in range(train_emb.shape[1]):
        reg = RidgeCV(alphas=_RIDGE_ALPHAS)
        reg.fit(train_sf, train_emb[:, dim])
        residuals[:, dim] = target_emb[:, dim] - reg.predict(target_sf)
    return residuals


# ── Linear probe utilities ────────────────────────────────────────────────────

def _probe_r2(X_train, y_train, X_val, y_val) -> float:
    """R² of RidgeCV probe fit on train, evaluated on val."""
    reg = RidgeCV(alphas=_RIDGE_ALPHAS)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_val)
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _bootstrap_r2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    n_boot:  int = 500,
    rng:     np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Bootstrap CI for R² by resampling val patients.

    Returns (point_estimate, ci_lo_2.5%, ci_hi_97.5%).
    The probe is fit once on train; bootstrap only resamples the val evaluation.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    reg = RidgeCV(alphas=_RIDGE_ALPHAS)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_val)

    point = _r2_from_pred(y_val, y_pred)

    boot = []
    n = len(y_val)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(_r2_from_pred(y_val[idx], y_pred[idx]))

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def _r2_from_pred(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _pc1_pearson(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r between the first PC of X and y (sign-flipped to positive r)."""
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X)[:, 0]
    r, p = stats.pearsonr(pc1, y)
    return (abs(r), p)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_embedding_probe(
    model,
    dataset,
    device,
    raw_df:     pd.DataFrame,
    train_pids,
    val_pids,
    output_dir: str,
    suffix:     str = "",
    n_boot:     int = 500,
    title:      str = "",
) -> dict | None:
    """
    Full embedding probe pipeline. Requires a synthetic dataset with a
    'signal' column in raw_df.

    Returns a summary dict of R² values and CI bounds, or None if skipped.
    """
    if "signal" not in raw_df.columns:
        print("  No 'signal' column — skipping (synthetic data only).")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Extract embeddings for all patients ─────────────────────────────────
    print("  Extracting GCN embeddings for all patients...")
    pid_data = extract_patient_embeddings(model, dataset, device)

    sig_map = raw_df.groupby("PID")["signal"].first().to_dict()

    train_rows, val_rows = [], []
    for pid, d in pid_data.items():
        sig = sig_map.get(pid)
        if sig is None:
            continue
        row = {
            "pid":    pid,
            "signal": float(sig),
            **{f"emb_{i}": float(d["embedding"][i])       for i in range(len(d["embedding"]))},
            **{f"sf_{i}":  float(d["spatial_features"][i]) for i in range(len(d["spatial_features"]))},
        }
        if pid in train_pids:
            train_rows.append(row)
        elif pid in val_pids:
            val_rows.append(row)

    if len(train_rows) < 10 or len(val_rows) < 5:
        print("  Insufficient patients — skipping embedding probe.")
        return None

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)

    # Explicit prefixes: a bare "s" prefix would also match the "signal" target
    emb_cols = [c for c in train_df.columns if c.startswith("emb_")]
    sf_cols  = [c for c in train_df.columns if c.startswith("sf_")]

    train_emb    = train_df[emb_cols].values.astype(float)
    train_sf     = train_df[sf_cols].values.astype(float)
    train_signal = train_df["signal"].values.astype(float)

    val_emb    = val_df[emb_cols].values.astype(float)
    val_sf     = val_df[sf_cols].values.astype(float)
    val_signal = val_df["signal"].values.astype(float)

    n_emb = train_emb.shape[1]
    n_sf  = train_sf.shape[1]
    print(f"  Train: {len(train_df)} patients | Val: {len(val_df)} patients")
    print(f"  Embedding dim: {n_emb} | Classical features: {n_sf} ({', '.join(SPATIAL_FEATURE_NAMES)})")

    # ── 2. Design check: verify NND ⊥ signal on val ───────────────────────────
    print("\n  Design check — Pearson r(classical feature, signal) on val:")
    for i, name in enumerate(SPATIAL_FEATURE_NAMES):
        r_check, p_check = stats.pearsonr(val_sf[:, i], val_signal)
        print(f"    {name:<25}  r={r_check:+.3f}  p={p_check:.3f}")

    # ── 3. Residualize embedding against classical features ───────────────────
    print("\n  Residualizing embedding (fit on train)...")
    train_residual = residualize_embedding(train_emb, train_sf, train_emb, train_sf)
    val_residual   = residualize_embedding(train_emb, train_sf, val_emb,   val_sf)

    # ── 4. Linear probes with bootstrap CIs ──────────────────────────────────
    print("  Running linear probes (RidgeCV, fit on train, R² on val)...")
    rng = np.random.default_rng(0)

    r2_sf,  ci_lo_sf,  ci_hi_sf  = _bootstrap_r2(train_sf,       train_signal, val_sf,       val_signal, n_boot, rng)
    r2_emb, ci_lo_emb, ci_hi_emb = _bootstrap_r2(train_emb,      train_signal, val_emb,      val_signal, n_boot, rng)
    r2_res, ci_lo_res, ci_hi_res = _bootstrap_r2(train_residual,  train_signal, val_residual, val_signal, n_boot, rng)

    print(f"\n  Ridge probe R² on val (95% bootstrap CI, n_val={len(val_df)}):")
    print(f"    Classical (NND+Ripley):  {r2_sf:+.3f}  [{ci_lo_sf:+.3f}, {ci_hi_sf:+.3f}]")
    print(f"    Raw embedding:           {r2_emb:+.3f}  [{ci_lo_emb:+.3f}, {ci_hi_emb:+.3f}]")
    print(f"    Residual embedding:      {r2_res:+.3f}  [{ci_lo_res:+.3f}, {ci_hi_res:+.3f}]")

    # PC1 Pearson as scalar sanity check
    r_pc1_sf,  p_pc1_sf  = _pc1_pearson(val_sf,       val_signal)
    r_pc1_emb, p_pc1_emb = _pc1_pearson(val_emb,      val_signal)
    r_pc1_res, p_pc1_res = _pc1_pearson(val_residual,  val_signal)

    print(f"\n  PC1 Pearson |r| on val:")
    print(f"    Classical:   |r|={r_pc1_sf:.3f}   p={p_pc1_sf:.4f}")
    print(f"    Raw emb:     |r|={r_pc1_emb:.3f}  p={p_pc1_emb:.4f}")
    print(f"    Residual:    |r|={r_pc1_res:.3f}  p={p_pc1_res:.4f}")

    # ── 5. Figure ──────────────────────────────────────────────────────────────
    _save_probe_figure(
        r2_values  = [r2_sf,      r2_emb,      r2_res],
        ci_lo      = [ci_lo_sf,   ci_lo_emb,   ci_lo_res],
        ci_hi      = [ci_hi_sf,   ci_hi_emb,   ci_hi_res],
        pc1_r      = [r_pc1_sf,   r_pc1_emb,   r_pc1_res],
        pc1_p      = [p_pc1_sf,   p_pc1_emb,   p_pc1_res],
        n_val      = len(val_df),
        output_dir = output_dir,
        suffix     = suffix,
        title      = title,
    )

    return {
        "r2_classical":    r2_sf,   "r2_classical_lo":  ci_lo_sf,  "r2_classical_hi":  ci_hi_sf,
        "r2_raw_emb":      r2_emb,  "r2_raw_emb_lo":    ci_lo_emb, "r2_raw_emb_hi":    ci_hi_emb,
        "r2_residual":     r2_res,  "r2_residual_lo":   ci_lo_res, "r2_residual_hi":   ci_hi_res,
        "pc1_r_classical": r_pc1_sf,
        "pc1_r_raw_emb":   r_pc1_emb,
        "pc1_r_residual":  r_pc1_res,
        "n_train":         len(train_df),
        "n_val":           len(val_df),
    }


def _save_probe_figure(
    r2_values:  list[float],
    ci_lo:      list[float],
    ci_hi:      list[float],
    pc1_r:      list[float],
    pc1_p:      list[float],
    n_val:      int,
    output_dir: str,
    suffix:     str,
    title:      str = "",
):
    labels = ["Classical\n(NND+Ripley)", "Raw\nembedding", "Residual\nembedding"]
    colors = ["#7fbc41", "#4575b4", "#d73027"]
    x = np.arange(len(labels))

    err_lo = [max(r2 - lo, 0) for r2, lo in zip(r2_values, ci_lo)]
    err_hi = [max(hi - r2, 0) for r2, hi in zip(r2_values, ci_hi)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: R² bar chart
    ax = axes[0]
    ax.bar(x, r2_values, color=colors, width=0.55, alpha=0.85)
    ax.errorbar(x, r2_values, yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Ridge probe R²")
    ax.set_title(f"Signal recovery (R²)\n95% bootstrap CI, n_val={n_val}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: PC1 Pearson |r| bar chart
    ax = axes[1]
    ax.bar(x, pc1_r, color=colors, width=0.55, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("PC1 Pearson |r|")
    ax.set_ylim(0, 1)
    for xi, r, p in zip(x, pc1_r, pc1_p):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(xi, r + 0.02, star, ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Signal recovery (PC1 Pearson |r|)\nval set, n={n_val}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    prefix = f"{title}: " if title else ""
    fig.suptitle(f"{prefix}how much does each feature set encode the true signal s?",
                 fontsize=10, y=1.01)
    fig.tight_layout()

    fname = f"fig_embedding_probe{('_' + suffix) if suffix else ''}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {fname}.pdf/png")
