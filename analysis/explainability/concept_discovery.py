"""
GCExplainer-style concept discovery for the GNN survival model.

Two complementary levels of analysis:

  Graph-level concepts
    Cluster the 32-dim GCN graph embeddings (one per core) into K groups.
    Each concept = a prototypical whole-core spatial arrangement.
    Output: UMAP coloured by concept/risk, per-concept representative cores,
    feature profiles, and completeness score (concept → risk R²).

  Node-level concepts
    Cluster the 32-dim node embeddings (one per cell) into K groups.
    Each concept = a prototypical local cell neighbourhood.
    Output: UMAP coloured by concept/cell-type, per-concept cell-type
    composition, completeness score (per-core concept distribution → risk R²),
    and cropped local-neighbourhood galleries (several small physical-radius
    crops per concept, from different patients) rather than a single whole
    representative core.

K is chosen automatically over a search range using silhouette score.

References:
  Magister et al. (2021) "GCExplainer: Human-in-the-Loop Concept-based
  Explanations for Graph Neural Networks"
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib.transforms import blended_transform_factory, offset_copy
from lifelines.statistics import multivariate_logrank_test

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.consts import (
    BATCH_KEY_NODE_FEATURES,
    BATCH_KEY_NODE_EMBEDDING,
    BATCH_KEY_GRAPH_EMBEDDING,
    BATCH_KEY_GRAPH_LABELS,
    BATCH_KEY_GRAPH_YHAT,
)
from data_utils import NND_FEATURE_NAMES, RIPLEY_FEATURE_NAMES, SPATIAL_FEATURE_NAMES

N_NND    = len(NND_FEATURE_NAMES)
N_RIPLEY = len(RIPLEY_FEATURE_NAMES)

_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


# ── Shared: embedding extraction ──────────────────────────────────────────────

@torch.no_grad()
def extract_all_embeddings(model, dataset, device) -> dict:
    """
    Single forward pass over the full dataset.

    Returns dict with keys:
      graph_emb   : {core: (32,)}
      node_emb    : {core: (N_nodes, 32)}
      node_types  : {core: (N_nodes,) int — index into dataset.cell_types}
      node_xy     : {core: (N_nodes, 2) — normalised coordinates}
      risk        : {core: float}
      pid         : {core: str}
      time_event  : {core: (time, event)}
      spatial_feat: {core: (9,) — normalised NND+Ripley}
      edge_index  : {core: (2, E) int — the actual k-NN edges used in this
                     forward pass, for tracing the model's true receptive field}
    """
    model.eval()
    out = {k: {} for k in
           ("graph_emb", "node_emb", "node_types", "node_xy",
            "risk", "pid", "time_event", "spatial_feat", "edge_index")}

    for idx in range(len(dataset)):
        g    = dataset[idx].to(device)
        core = dataset.cores[idx]
        pid  = dataset.data[core]["PID"]

        fwd  = model(g)
        risk = fwd[BATCH_KEY_GRAPH_YHAT].squeeze().item()
        label = fwd[BATCH_KEY_GRAPH_LABELS].view(-1, 2)[0]

        graph_emb = fwd[BATCH_KEY_GRAPH_EMBEDDING].squeeze(0).cpu().numpy()
        node_emb  = fwd[BATCH_KEY_NODE_EMBEDDING].cpu().numpy()       # (N, 32)
        node_feat = g.x[BATCH_KEY_NODE_FEATURES].cpu().numpy()        # (N, 4)
        sf        = g.nnd_features.squeeze(0).cpu().numpy()           # (9,)

        # Cell type: one-hot starts at column 2
        node_types = np.argmax(node_feat[:, 2:], axis=1)              # (N,)
        node_xy    = node_feat[:, :2]                                  # (N, 2)

        out["graph_emb"][core]    = graph_emb
        out["node_emb"][core]     = node_emb
        out["node_types"][core]   = node_types
        out["node_xy"][core]      = node_xy
        out["risk"][core]         = risk
        out["pid"][core]          = pid
        out["time_event"][core]   = (label[0].item(), label[1].item())
        out["spatial_feat"][core] = sf
        out["edge_index"][core]   = g.edge_index.cpu().numpy()

    return out


def _infer_n_layers(model) -> int:
    """
    Introspect the loaded model to recover the number of GCN/GIN/GAT
    message-passing layers, instead of trusting a separately-passed value
    that could silently mismatch the checkpoint's actual architecture.
    """
    from torch_geometric.nn import MessagePassing
    n = sum(1 for m in model.node_embedding.modules()
            if isinstance(m, MessagePassing))
    if n == 0:
        raise ValueError("Could not infer n_layers from model.node_embedding "
                         "— no MessagePassing layers found.")
    return n


# ── Shared: K search ──────────────────────────────────────────────────────────

def search_k(
    X: np.ndarray,
    k_range: tuple[int, int] = (2, 10),
    n_init: int = 10,
    random_state: int = 0,
    completeness_targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[int, dict, dict]:
    """
    Search for optimal K by silhouette score on X.

    completeness_targets: optional {name: (y, train_mask, val_mask)}. For
    each K, besides silhouette, also fits a one-hot(cluster id) → y RidgeCV
    probe on train/val (same split, same `_completeness_r2` used for the
    final reported completeness) and records its R². This reuses the exact
    clustering already fit for the silhouette curve at that K, so the two
    curves describe the same clusterings and are directly comparable on one
    plot — purely diagnostic, it does not change how best_k is chosen below.

    Returns (best_k, silhouette_scores: {k: score}, completeness: {name: {k: r2}}).
    """
    scores = {}
    completeness = {name: {} for name in (completeness_targets or {})}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores[k] = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        if completeness_targets:
            X_oh = OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1))
            for name, (y, train_mask, val_mask) in completeness_targets.items():
                completeness[name][k] = _completeness_r2(
                    X_oh[train_mask], y[train_mask],
                    X_oh[val_mask],   y[val_mask],
                )
    best_k = max(scores, key=scores.__getitem__)
    return best_k, scores, completeness


def _plot_k_search(scores: dict, best_k: int, title: str,
                   output_dir: str, fname: str,
                   completeness_curves: dict[str, dict] | None = None):
    ks = sorted(scores)
    vals = [scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ln_sil, = ax.plot(ks, vals, marker="o", color="#4575b4", label="Silhouette")
    ln_best = ax.axvline(best_k, color="grey", linestyle=":", linewidth=1.5)
    ax.set_xlabel("K (number of concepts)", fontsize=11)
    ax.set_ylabel("Silhouette score", color="#4575b4", fontsize=11)
    ax.tick_params(axis="y", labelcolor="#4575b4")
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)

    handles = [ln_sil, ln_best]
    labels_ = ["Silhouette", f"Chosen K = {best_k}"]
    if completeness_curves:
        ax2 = ax.twinx()
        ax2.set_ylabel("Completeness R² (held-out)", fontsize=11)
        # Distinct from the silhouette blue and the grey chosen-K line
        comp_colors = ["#7fbc41", "#9e3d8c", "#fdae61"]
        for (name, curve), color in zip(completeness_curves.items(), comp_colors):
            cks = sorted(curve)
            ln, = ax2.plot(cks, [curve[k] for k in cks], marker="s",
                           linestyle="--", color=color, label=f"Completeness → {name}")
            handles.append(ln)
            labels_.append(f"Completeness → {name}")
        ax2.spines["top"].set_visible(False)
    else:
        ax.spines["right"].set_visible(False)

    # Below the axes so it can never cover a curve
    ax.legend(handles, labels_, fontsize=9, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


# ── Shared: completeness ──────────────────────────────────────────────────────

def _completeness_r2(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> float:
    reg = RidgeCV(alphas=_RIDGE_ALPHAS)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_val)
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH-LEVEL CONCEPT DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def run_graph_concept_discovery(
    model,
    dataset,
    device,
    raw_df:     pd.DataFrame,
    train_pids,
    val_pids,
    output_dir: str,
    suffix:     str = "",
    k_range:    tuple[int, int] = (2, 10),
    n_init:     int = 10,
    umap_seed:  int = 0,
) -> dict | None:
    """
    Graph-level concept discovery: cluster 32-dim graph embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("  Extracting embeddings...")
    emb_data = extract_all_embeddings(model, dataset, device)

    cores = list(emb_data["graph_emb"])
    G     = np.array([emb_data["graph_emb"][c] for c in cores])   # (n_cores, 32)
    risks = np.array([emb_data["risk"][c]       for c in cores])
    pids  = [emb_data["pid"][c] for c in cores]
    sfs   = np.array([emb_data["spatial_feat"][c] for c in cores])

    train_mask = np.array([pids[i] in train_pids for i in range(len(cores))])
    val_mask   = ~train_mask

    sig_map = (raw_df.groupby("PID")["signal"].first().to_dict()
               if "signal" in raw_df.columns else {})

    # ── K search ──────────────────────────────────────────────────────────────
    completeness_targets = {"risk": (risks, train_mask, val_mask)}
    if sig_map:
        sigs = np.array([sig_map.get(pid, np.nan) for pid in pids])
        sig_mask = ~np.isnan(sigs)
        tr_sig = train_mask & sig_mask
        va_sig = val_mask & sig_mask
        if tr_sig.sum() >= 10 and va_sig.sum() >= 5:
            completeness_targets["signal"] = (sigs, tr_sig, va_sig)

    print(f"  Searching K in {k_range} on {len(cores)} graph embeddings...")
    best_k, k_scores, k_completeness = search_k(
        G, k_range=k_range, n_init=n_init,
        completeness_targets=completeness_targets,
    )
    print(f"  Best K={best_k} (silhouette={k_scores[best_k]:.3f})")
    _plot_k_search(k_scores, best_k,
                   title="Graph-level K search (silhouette vs. completeness)",
                   output_dir=output_dir,
                   fname=f"fig_graph_concepts_k_search{_sfx(suffix)}",
                   completeness_curves=k_completeness)

    # ── Final K-means ─────────────────────────────────────────────────────────
    km = KMeans(n_clusters=best_k, n_init=n_init, random_state=0)
    labels = km.fit_predict(G)

    # ── Kaplan-Meier curves per concept ───────────────────────────────────────
    print("  Plotting KM curves per concept...")
    _plot_graph_concept_km(
        labels=labels,
        pids=pids,
        cores=cores,
        emb_data=emb_data,
        best_k=best_k,
        val_pids=val_pids,
        output_dir=output_dir,
        suffix=suffix,
    )

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("  Computing UMAP...")
    reducer = UMAP(n_components=2, random_state=umap_seed, n_neighbors=15,
                   min_dist=0.1, verbose=False)
    G_2d = reducer.fit_transform(G)

    _plot_graph_umap(G_2d, labels, risks, best_k, output_dir, suffix)
    _plot_graph_concepts_combined(G_2d, labels, risks, best_k, pids, cores,
                                  emb_data, val_pids, output_dir, suffix)

    # ── Per-concept profiles ───────────────────────────────────────────────────
    concept_stats = _graph_concept_profiles(
        labels, risks, sfs, sig_map, pids, best_k, output_dir, suffix
    )

    # ── Representative cores ───────────────────────────────────────────────────
    print("  Plotting representative cores...")
    centroids = km.cluster_centers_
    for k in range(best_k):
        mask_k = labels == k
        idx_k  = np.where(mask_k)[0]
        dists  = np.linalg.norm(G[idx_k] - centroids[k], axis=1)
        rep_idx = idx_k[np.argmin(dists)]
        rep_core = cores[rep_idx]
        rep_pid  = pids[rep_idx]
        _plot_representative_core(
            raw_df=raw_df,
            core=rep_core,
            pid=rep_pid,
            concept_id=k,
            n_concepts=best_k,
            risk=float(risks[rep_idx]),
            cell_types=dataset.cell_types,
            output_dir=output_dir,
            suffix=suffix,
            level="graph",
        )

    # ── Completeness ──────────────────────────────────────────────────────────
    # Reuses the same (y, train_mask, val_mask) targets already swept over K
    # above, now evaluated on the final chosen clustering.
    X_oh = OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1))

    r2_completeness = _completeness_r2(
        X_oh[train_mask], risks[train_mask],
        X_oh[val_mask],   risks[val_mask],
    )
    print(f"  Completeness R² (concept → risk, val): {r2_completeness:.3f}")

    # Signal completeness (synthetic only): does the discrete concept
    # assignment itself recover the ground-truth generating signal, as
    # opposed to merely predicting the model's own risk output?
    r2_completeness_signal = None
    if "signal" in completeness_targets:
        sigs, tr, va = completeness_targets["signal"]
        r2_completeness_signal = _completeness_r2(
            X_oh[tr], sigs[tr],
            X_oh[va], sigs[va],
        )
        print(f"  Completeness R² (concept → signal, val): {r2_completeness_signal:.3f}")

    return {
        "best_k":                  best_k,
        "k_scores":                k_scores,
        "completeness":            r2_completeness,
        "completeness_signal":     r2_completeness_signal,
        "concept_stats":           concept_stats,
    }


def _plot_graph_concept_km(
    labels:   np.ndarray,
    pids:     list,
    cores:    list,
    emb_data: dict,
    best_k:   int,
    val_pids,
    output_dir: str,
    suffix:   str,
    ax=None,
) -> None:
    """
    Kaplan-Meier survival curves, one per graph-level concept.

    Each patient is assigned the concept that appears most often across their
    cores (majority vote). Survival time/event come from the dataset labels.
    Plotted for val patients only, with at-risk counts, log-rank p and Cox HRs.
    """
    # ── Patient-level concept assignment (majority vote across cores) ─────────
    pid_concept_votes: dict = {}
    pid_survival:      dict = {}

    for i, core in enumerate(cores):
        pid = pids[i]
        pid_concept_votes.setdefault(pid, []).append(int(labels[i]))
        pid_survival[pid] = emb_data["time_event"][core]   # (time, event)

    patient_pids     = list(pid_survival)
    patient_concepts = np.array([
        int(np.bincount(pid_concept_votes[p]).argmax()) for p in patient_pids
    ])
    patient_times  = np.array([pid_survival[p][0] for p in patient_pids])
    patient_events = np.array([pid_survival[p][1] for p in patient_pids])

    # Concepts are fit on all cores (unsupervised), but survival separation is
    # only evaluated on val patients: the model was trained with a Cox loss on
    # the train patients, so separation there is partly by construction.
    is_val = np.array([p in val_pids for p in patient_pids])

    _plot_concept_km(
        patient_concepts=patient_concepts[is_val],
        patient_times=patient_times[is_val],
        patient_events=patient_events[is_val],
        group_labels={k: f"Concept {k}" for k in range(best_k)},
        fname=f"fig_graph_concepts_km{_sfx(suffix)}",
        output_dir=output_dir,
        ax=ax,
    )


def _plot_concept_km(
    patient_concepts: np.ndarray,
    patient_times:    np.ndarray,
    patient_events:   np.ndarray,
    group_labels:     dict,
    fname:            str,
    output_dir:       str,
    ax=None,
) -> None:
    """
    KM curves per concept group with a number-at-risk table, log-rank p, and a
    univariable Cox HR table (each group vs the largest group as reference).

    Saves {fname}.pdf/png and the HR table as {fname}_hr.csv. If ax is given,
    draws into it instead and saves nothing (used by the combined figure).
    """
    # Groups with <3 patients are dropped from the plot (too few for a
    # meaningful KM curve) — exclude them from the log-rank test and Cox model
    # too, so the reported statistics reflect exactly the curves that are drawn.
    groups = [k for k in group_labels if (patient_concepts == k).sum() >= 3]
    keep   = np.isin(patient_concepts, groups)
    c, t, e = patient_concepts[keep], patient_times[keep], patient_events[keep]

    # ── Log-rank test ─────────────────────────────────────────────────────────
    if len(groups) > 1 and e.sum() >= 2:
        lr = multivariate_logrank_test(t, c, e)
        p_str = _fmt_p(lr.p_value)
    else:
        p_str = "n/a"

    # ── Cox HR per group vs reference (largest group) ─────────────────────────
    ref = max(groups, key=lambda k: (c == k).sum()) if groups else None
    hr_rows = []
    cox = None
    if len(groups) > 1 and e.sum() >= 2:
        cox_df = pd.DataFrame({"T": t, "E": e})
        for k in groups:
            if k != ref:
                cox_df[f"g{k}"] = (c == k).astype(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cox = CoxPHFitter().fit(cox_df, duration_col="T", event_col="E")
        except Exception as exc:   # e.g. convergence failure with tiny groups
            print(f"  Cox fit failed ({exc.__class__.__name__}); HRs shown as n/a")
            cox = None

    for k in groups:
        row = {"group": group_labels[k], "n": int((c == k).sum()),
               "events": int(e[c == k].sum())}
        if k == ref:
            row.update(hr=1.0, ci_low=np.nan, ci_high=np.nan, p=np.nan)
        elif cox is not None:
            s = cox.summary.loc[f"g{k}"]
            row.update(hr=s["exp(coef)"],
                       ci_low=s["exp(coef) lower 95%"],
                       ci_high=s["exp(coef) upper 95%"],
                       p=s["p"])
        else:
            row.update(hr=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
        hr_rows.append(row)
    hr_df = pd.DataFrame(hr_rows)

    # ── Plot: KM with HRs in the legend, at-risk counts underneath ───────────
    # The full HR table (n, events, p) goes to the CSV / report rather than the
    # figure; KM CI bands are omitted since the HR CIs carry the uncertainty.
    def _legend_label(r) -> str:
        if r.group == group_labels[ref]:
            return f"{r.group}: reference"
        if np.isnan(r.hr):
            return f"{r.group}: HR n/a"
        return f"{r.group}: HR {r.hr:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f})"

    legend_labels = {r.group: _legend_label(r) for r in hr_df.itertuples()}

    cmap = plt.cm.get_cmap("tab10", max(len(group_labels), 1))
    # Fixed axes (no tight_layout): the legend sits outside the axes, and
    # tight_layout would shrink the plot to make room for it. bbox_inches=
    # "tight" on save expands the canvas to whatever falls outside instead.
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(4.5, 3))
        ax  = fig.add_axes([0, 0, 1, 1])

    for k in groups:
        mask = c == k
        kmf = KaplanMeierFitter()
        kmf.fit(t[mask], e[mask],
                label=legend_labels.get(group_labels[k], group_labels[k]))
        kmf.plot_survival_function(
            ax=ax, color=cmap(k), linewidth=2, ci_show=False,
        )

    # Font sizes are large because the figure is shown at half text width in
    # the report. Log-rank p and the HRs share one box anchored above the
    # top-right corner of the axes, so it can never cover a curve; it takes
    # the place of the title, which the report's subcaptions already give.
    ax.set_xlabel("Time (months)", fontsize=16)
    ax.set_ylabel("Survival probability", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=14, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2,
              title=(f"Log-rank {p_str}, n = {len(c)}, {int(e.sum())} events\n"
                     "Cox HR (95% CI)"),
              title_fontsize=14, alignment="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _add_at_risk_table(ax, [(group_labels[k], t[c == k], cmap(k)) for k in groups])
    if not standalone:
        return

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    hr_df.to_csv(os.path.join(output_dir, f"{fname}_hr.csv"), index=False)
    print(f"  Saved {fname}.pdf/png + _hr.csv  (val only, log-rank {p_str})")
    if not hr_df.empty:
        print(hr_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _add_at_risk_table(ax, groups: list) -> None:
    """
    Number-at-risk rows under a KM axis, one per (label, durations, colour),
    at the axis' x ticks. Drawn by hand because lifelines'
    add_at_risk_counts is broken on the installed numpy/pandas.
    """
    ax.set_xlim(left=0)
    x0, x1 = ax.get_xlim()
    ticks  = [x for x in ax.get_xticks() if x0 <= x <= x1]
    ax.set_xticks(ticks)
    # Rows are offset from the bottom of the axes in points, not axes
    # fractions, so their spacing doesn't depend on the plot's height.
    fig    = ax.figure
    trans  = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0, 0, "Number at risk", fontsize=14, fontweight="bold",
            ha="left", va="center",
            transform=offset_copy(ax.transAxes, fig, y=-64, units="points"))
    for i, (label, durations, color) in enumerate(groups):
        dy = -86 - 19 * i
        ax.text(0, 0, label, fontsize=14, color=color, ha="right",
                va="center", transform=offset_copy(ax.transAxes, fig, x=-18,
                                                   y=dy, units="points"))
        row = offset_copy(trans, fig, y=dy, units="points")
        for x in ticks:
            ax.text(x, 0, f"{int((durations >= x).sum())}", transform=row,
                    fontsize=14, ha="center", va="center")


def _fmt_p(p: float, prefix: bool = True) -> str:
    if p < 0.001:
        return "p < 0.001" if prefix else "<0.001"
    return f"p = {p:.3f}" if prefix else f"{p:.3f}"


def _plot_graph_umap(G_2d, labels, risks, best_k, output_dir, suffix,
                     axes=None, fontsize=None):
    """
    Two UMAP panels: coloured by concept and by predicted risk. Saves
    fig_graph_concepts_umap{suffix} unless axes (concept, risk, colorbar) are
    given, in which case it draws into them and saves nothing.
    """
    cmap_concepts = plt.cm.get_cmap("tab10", best_k)

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        cax = None
    else:
        axes, cax = axes[:2], axes[2]
    fs = {} if fontsize is None else {"fontsize": fontsize}

    # Panel 1: concept assignment
    ax = axes[0]
    for k in range(best_k):
        m = labels == k
        ax.scatter(G_2d[m, 0], G_2d[m, 1], c=[cmap_concepts(k)],
                   label=f"Concept {k}", s=25, alpha=0.8, linewidths=0)
    ax.set_title("UMAP, concept", **fs)
    ax.legend(fontsize=fontsize - 2 if fontsize else 7, markerscale=1.5,
              framealpha=0.5)
    ax.set_xlabel("UMAP 1", **fs); ax.set_ylabel("UMAP 2", **fs)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel 2: predicted risk
    ax = axes[1]
    sc = ax.scatter(G_2d[:, 0], G_2d[:, 1], c=risks,
                    cmap="RdBu_r", s=25, alpha=0.8, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, cax=cax)
    cb.set_label("Predicted risk", **fs)
    if fontsize:
        cb.ax.tick_params(labelsize=fontsize - 2)
    ax.set_title("UMAP, predicted risk", **fs)
    ax.set_xlabel("UMAP 1", **fs); ax.set_ylabel("UMAP 2", **fs)
    ax.set_xticks([]); ax.set_yticks([])
    if not standalone:
        return

    fig.tight_layout()
    fname = f"fig_graph_concepts_umap{_sfx(suffix)}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.pdf/png")


def _plot_graph_concepts_combined(G_2d, labels, risks, best_k, pids, cores,
                                  emb_data, val_pids, output_dir, suffix):
    """
    One figure with the two UMAP panels and the KM curves side by side, for
    use at full text width in the report. Axes are placed by hand (no
    tight_layout) because the KM legend and at-risk table sit outside its
    axes, and bbox_inches="tight" expands the canvas to include them.
    """
    # The KM block (legend above, at-risk table below) is taller than its
    # axes, so the square UMAP panels are centred on the whole block.
    fig = plt.figure(figsize=(15, 4.2))
    ax_concept = fig.add_axes([0.00, -0.04, 0.23, 0.83])
    ax_risk    = fig.add_axes([0.29, -0.04, 0.23, 0.83])
    cax        = fig.add_axes([0.53, -0.04, 0.011, 0.83])
    ax_km      = fig.add_axes([0.70, 0.0, 0.30, 0.80])

    _plot_graph_umap(G_2d, labels, risks, best_k, output_dir, suffix,
                     axes=[ax_concept, ax_risk, cax], fontsize=16)
    _plot_graph_concept_km(labels=labels, pids=pids, cores=cores,
                           emb_data=emb_data, best_k=best_k,
                           val_pids=val_pids, output_dir=output_dir,
                           suffix=suffix, ax=ax_km)

    fname = f"fig_graph_concepts{_sfx(suffix)}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.pdf/png")


def _graph_concept_profiles(labels, risks, sfs, sig_map, pids,
                              best_k, output_dir, suffix) -> dict:
    """Per-concept mean risk, NND, and Ripley features. Returns summary dict."""
    feature_names = NND_FEATURE_NAMES + RIPLEY_FEATURE_NAMES

    concept_stats = {}
    for k in range(best_k):
        m = labels == k
        concept_stats[k] = {
            "n":          int(m.sum()),
            "mean_risk":  float(risks[m].mean()),
            "sd_risk":    float(risks[m].std()),
            "mean_sf":    sfs[m].mean(axis=0),
        }
        if sig_map:
            sigs = [sig_map.get(p, np.nan) for p in np.array(pids)[m]]
            concept_stats[k]["mean_signal"] = float(np.nanmean(sigs))

    # Figure: mean risk per concept + selected spatial features
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cmap = plt.cm.get_cmap("tab10", best_k)
    ks   = list(range(best_k))

    ax = axes[0]
    means = [concept_stats[k]["mean_risk"] for k in ks]
    sds   = [concept_stats[k]["sd_risk"]   for k in ks]
    colors = [cmap(k) for k in ks]
    ax.bar(ks, means, yerr=sds, color=colors, capsize=4, width=0.6, alpha=0.85)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"C{k}\n(n={concept_stats[k]['n']})" for k in ks],
                       fontsize=8)
    ax.set_ylabel("Mean predicted risk ± SD")
    ax.set_title("Risk profile per graph concept")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Heatmap: normalised spatial features per concept
    ax = axes[1]
    sf_matrix = np.array([concept_stats[k]["mean_sf"] for k in ks])
    # Normalise each feature to [0,1] for display
    sf_norm = (sf_matrix - sf_matrix.min(0)) / (
        sf_matrix.max(0) - sf_matrix.min(0) + 1e-8)
    im = ax.imshow(sf_norm.T, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"C{k}" for k in ks], fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title("Spatial feature profile\n(row-normalised)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.tight_layout()
    fname = f"fig_graph_concepts_profiles{_sfx(suffix)}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.pdf/png")

    return concept_stats


def _plot_representative_core(raw_df, core, pid, concept_id, n_concepts,
                               risk, cell_types, output_dir, suffix, level):
    """Scatter plot of one representative core coloured by cell type."""
    sub = raw_df[raw_df["core"] == core]
    if sub.empty:
        return

    cmap  = plt.cm.get_cmap("tab10", len(cell_types))
    ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}

    fig, ax = plt.subplots(figsize=(4, 4))
    for ct in cell_types:
        cells = sub[sub["cell_type"] == ct]
        if cells.empty:
            continue
        i = ct_to_idx[ct]
        ax.scatter(cells["x"], cells["y"], c=[cmap(i)], s=4,
                   alpha=0.7, linewidths=0, label=ct)

    ax.set_title(f"Concept {concept_id} representative core\n"
                 f"PID={pid}  risk={risk:.3f}", fontsize=9)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.legend(fontsize=7, markerscale=2, loc="upper right", framealpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fname = f"fig_{level}_concept_{concept_id}_core{_sfx(suffix)}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE-LEVEL CONCEPT DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def run_node_concept_discovery(
    model,
    dataset,
    device,
    train_pids,
    val_pids,
    output_dir:      str,
    suffix:          str = "",
    k_range:         tuple[int, int] = (2, 10),
    n_init:          int = 10,
    max_nodes_total: int = 50_000,
    umap_seed:       int = 0,
) -> dict | None:
    """
    Node-level concept discovery: cluster 32-dim node embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("  Extracting node embeddings...")
    emb_data = extract_all_embeddings(model, dataset, device)
    n_layers = _infer_n_layers(model)
    print(f"  Inferred {n_layers}-layer receptive field for gallery crops")

    cores  = list(emb_data["node_emb"])
    pids   = [emb_data["pid"][c] for c in cores]
    risks  = {c: emb_data["risk"][c] for c in cores}

    # Per-core bounding box (full, un-subsampled cell coordinates) used to map
    # each node's normalised [0,1] coordinates back to approximate microns,
    # purely for plotting node positions in real units. Using the dataset's
    # own stored coords (rather than re-deriving from a forward pass)
    # sidesteps PointPatternDataset's random per-core cell subsampling above
    # `max_cells` (data_utils.py), which would otherwise make positional
    # alignment with any external per-cell table unreliable.
    core_bbox = {
        c: (dataset.data[c]["coords"].min(axis=0), dataset.data[c]["coords"].max(axis=0))
        for c in cores
    }

    # Pool all node embeddings with core/cell-type metadata
    all_node_emb   = []
    all_node_types = []
    all_node_core  = []
    all_node_xy    = []

    for c in cores:
        ne = emb_data["node_emb"][c]         # (N, 32)
        nt = emb_data["node_types"][c]       # (N,)
        nx = emb_data["node_xy"][c]          # (N, 2)
        all_node_emb.append(ne)
        all_node_types.append(nt)
        all_node_core.extend([c] * len(ne))
        all_node_xy.append(nx)

    all_node_emb   = np.concatenate(all_node_emb,   axis=0)   # (N_total, 32)
    all_node_types = np.concatenate(all_node_types, axis=0)   # (N_total,)
    all_node_xy    = np.concatenate(all_node_xy,    axis=0)   # (N_total, 2)
    all_node_core  = np.array(all_node_core)

    print(f"  Total nodes: {len(all_node_emb):,}")

    # Subsample for K-means and UMAP if needed
    if len(all_node_emb) > max_nodes_total:
        rng  = np.random.default_rng(0)
        idx  = rng.choice(len(all_node_emb), max_nodes_total, replace=False)
        idx  = np.sort(idx)
        X_km = all_node_emb[idx]
        types_km = all_node_types[idx]
        core_km  = all_node_core[idx]
        print(f"  Subsampled to {max_nodes_total:,} nodes for K-means / UMAP")
    else:
        X_km     = all_node_emb
        types_km = all_node_types
        core_km  = all_node_core
        idx      = np.arange(len(all_node_emb))

    # ── K search ──────────────────────────────────────────────────────────────
    print(f"  Searching K in {k_range} on {len(X_km):,} node embeddings...")
    best_k, k_scores, _ = search_k(X_km, k_range=k_range, n_init=n_init)
    print(f"  Best K={best_k} (silhouette={k_scores[best_k]:.3f})")
    _plot_k_search(k_scores, best_k,
                   title="Node-level K search (silhouette)",
                   output_dir=output_dir,
                   fname=f"fig_node_concepts_k_search{_sfx(suffix)}")

    # ── Final K-means (fit on subsample, predict all) ────────────────────────
    km = KMeans(n_clusters=best_k, n_init=n_init, random_state=0)
    km.fit(X_km)
    labels_all = km.predict(all_node_emb)      # assign every node
    labels_km  = labels_all[idx]               # labels for subsample

    # ── KM curves per node concept ────────────────────────────────────────────
    print("  Plotting KM curves per node concept...")
    _plot_node_concept_km(
        labels_all=labels_all,
        all_node_core=all_node_core,
        cores=cores,
        pids=pids,
        emb_data=emb_data,
        best_k=best_k,
        val_pids=val_pids,
        output_dir=output_dir,
        suffix=suffix,
    )

    # ── UMAP (on subsample) ───────────────────────────────────────────────────
    print("  Computing UMAP on node subsample...")
    reducer = UMAP(n_components=2, random_state=umap_seed, n_neighbors=30,
                   min_dist=0.1, verbose=False)
    X_2d = reducer.fit_transform(X_km)

    _plot_node_umap(X_2d, labels_km, types_km, best_k, dataset.cell_types,
                    output_dir, suffix)

    # ── Per-concept cell-type composition ─────────────────────────────────────
    cell_types = dataset.cell_types
    _plot_node_composition(labels_all, all_node_types, best_k, cell_types,
                           output_dir, suffix)

    # ── Per-concept cropped neighbourhood galleries ───────────────────────────
    # For each concept, show several cropped local windows around diverse
    # exemplar cells (different patients) rather than one whole cherry-picked
    # core — a node-level concept is a local neighbourhood, not a core-scale
    # pattern, so the illustration should match that scale.
    print("  Plotting node concept galleries...")
    _plot_node_concept_galleries(
        labels_all=labels_all,
        all_node_core=all_node_core,
        all_node_types=all_node_types,
        best_k=best_k,
        emb_data=emb_data,
        core_bbox=core_bbox,
        cell_types=cell_types,
        output_dir=output_dir,
        suffix=suffix,
        centroids=km.cluster_centers_,
        all_node_emb=all_node_emb,
        n_hops=n_layers,
    )

    # ── Completeness: per-core concept distribution → risk ────────────────────
    core_concept_dist = {}
    for c in cores:
        mask = all_node_core == c
        counts = np.bincount(labels_all[mask], minlength=best_k).astype(float)
        core_concept_dist[c] = counts / counts.sum()

    train_cores = [c for c, p in zip(cores, pids) if p in train_pids]
    val_cores   = [c for c, p in zip(cores, pids) if p in val_pids]

    if train_cores and val_cores:
        X_train = np.array([core_concept_dist[c] for c in train_cores])
        y_train = np.array([risks[c] for c in train_cores])
        X_val   = np.array([core_concept_dist[c] for c in val_cores])
        y_val   = np.array([risks[c] for c in val_cores])

        r2_completeness = _completeness_r2(X_train, y_train, X_val, y_val)
        print(f"  Node completeness R² (concept dist → risk, val): {r2_completeness:.3f}")
    else:
        r2_completeness = float("nan")

    return {
        "best_k":       best_k,
        "k_scores":     k_scores,
        "completeness": r2_completeness,
    }


def _plot_node_concept_km(
    labels_all:    np.ndarray,
    all_node_core: np.ndarray,
    cores:         list,
    pids:          list,
    emb_data:      dict,
    best_k:        int,
    val_pids,
    output_dir:    str,
    suffix:        str,
) -> None:
    """
    Kaplan-Meier survival curves, one per node-level concept.

    Each patient is assigned the concept that dominates across all their nodes
    (majority vote across all cores). This ties node-level concepts back to
    patient-level survival outcomes. Plotted for val patients only, with
    at-risk counts, log-rank p and Cox HRs.
    """
    pid_concept_votes: dict = {}
    pid_survival:      dict = {}

    for i, core in enumerate(cores):
        pid = pids[i]
        mask = all_node_core == core
        node_concepts = labels_all[mask]
        pid_concept_votes.setdefault(pid, []).extend(node_concepts.tolist())
        pid_survival[pid] = emb_data["time_event"][core]

    patient_pids = list(pid_survival)

    # Compute per-patient concept fraction vectors
    concept_fracs = np.array([
        np.bincount(np.array(pid_concept_votes[p], dtype=int), minlength=best_k).astype(float)
        / max(len(pid_concept_votes[p]), 1)
        for p in patient_pids
    ])  # shape (n_patients, best_k)

    if best_k == 2:
        # Median split on concept-0 fraction for balanced groups regardless of
        # global class imbalance (majority-vote degenerates when one concept is rare).
        frac0 = concept_fracs[:, 0]
        median_frac = np.median(frac0)
        # 0 = concept-0 enriched (above median), 1 = concept-0 depleted (below)
        patient_concepts = (frac0 < median_frac).astype(int)
        km_labels = {
            0: "Concept-0 enriched",
            1: "Concept-0 depleted",
        }
    else:
        patient_concepts = concept_fracs.argmax(axis=1).astype(int)
        km_labels = {k: f"Concept {k}" for k in range(best_k)}

    patient_times  = np.array([pid_survival[p][0] for p in patient_pids])
    patient_events = np.array([pid_survival[p][1] for p in patient_pids])

    # Group assignment (incl. the K=2 median threshold) uses all patients —
    # it never touches outcomes — but survival is evaluated on val patients
    # only; see _plot_graph_concept_km.
    is_val = np.array([p in val_pids for p in patient_pids])

    _plot_concept_km(
        patient_concepts=patient_concepts[is_val],
        patient_times=patient_times[is_val],
        patient_events=patient_events[is_val],
        group_labels=km_labels,
        fname=f"fig_node_concepts_km{_sfx(suffix)}",
        output_dir=output_dir,
    )


def _plot_node_umap(X_2d, labels, types, best_k, cell_types, output_dir, suffix):
    cmap_concepts  = plt.cm.get_cmap("tab10", best_k)
    # NB: resampling Set2 to len(cell_types) colours (e.g. via
    # get_cmap("Set2", N)) interpolates across the full 8-colour palette, so
    # for small N (e.g. 2, the synthetic tumor/immune case) it lands on the
    # first *and last* entries — teal and a low-contrast grey. Indexing the
    # raw 8-colour palette directly keeps every class a vivid, distinct hue.
    cmap_celltypes = plt.cm.get_cmap("Set2")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: coloured by concept
    ax = axes[0]
    for k in range(best_k):
        m = labels == k
        ax.scatter(X_2d[m, 0], X_2d[m, 1], c=[cmap_concepts(k)],
                   s=3, alpha=0.5, linewidths=0, label=f"C{k}", rasterized=True)
    ax.set_title("Node embeddings — UMAP\ncoloured by concept")
    ax.legend(fontsize=7, markerscale=3, framealpha=0.5,
              loc="best", ncol=2)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])

    # Right: coloured by cell type
    ax = axes[1]
    for i, ct in enumerate(cell_types):
        m = types == i
        ax.scatter(X_2d[m, 0], X_2d[m, 1], c=[cmap_celltypes(i)],
                   s=3, alpha=0.5, linewidths=0, label=ct, rasterized=True)
    ax.set_title("Node embeddings — UMAP\ncoloured by cell type")
    ax.legend(fontsize=7, markerscale=3, framealpha=0.5, loc="best")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fname = f"fig_node_concepts_umap{_sfx(suffix)}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}.pdf/png")


def _plot_node_composition(labels_all, all_node_types, best_k, cell_types,
                            output_dir, suffix):
    """Stacked bar: fraction of each cell type per concept."""
    n_ct = len(cell_types)
    # See _plot_node_umap for why the raw (unresampled) palette is used.
    cmap = plt.cm.get_cmap("Set2")

    fracs = np.zeros((best_k, n_ct))
    for k in range(best_k):
        mask = labels_all == k
        counts = np.bincount(all_node_types[mask], minlength=n_ct).astype(float)
        fracs[k] = counts / counts.sum() if counts.sum() > 0 else 0

    fig, ax = plt.subplots(figsize=(max(5, best_k * 0.9), 4))
    bottom = np.zeros(best_k)
    x = np.arange(best_k)
    for i, ct in enumerate(cell_types):
        ax.bar(x, fracs[:, i], bottom=bottom, color=cmap(i),
               label=ct, width=0.6, alpha=0.9)
        bottom += fracs[:, i]

    total_per_concept = np.bincount(labels_all, minlength=best_k)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"C{k}\n(n={total_per_concept[k]:,})" for k in range(best_k)],
        fontsize=8
    )
    ax.set_ylabel("Fraction of nodes")
    ax.set_ylim(0, 1)
    ax.set_title("Node concept — cell-type composition")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fname = f"fig_node_concepts_composition{_sfx(suffix)}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.pdf/png")


def _receptive_field_nodes(edge_index: np.ndarray, seed: int, n_hops: int) -> set:
    """
    Backward BFS on the directed k-NN graph: which nodes actually contributed
    to `seed`'s embedding after `n_hops` GCN layers.

    `knn_graph` edges point from each node's k-NN *sources* to that node as
    *destination* (edge_index[1]) — verified empirically: a node's incoming
    edges exactly match its own k-nearest neighbours. GCNConv aggregates a
    destination node's representation from its incoming-edge sources, so the
    receptive field of `seed` is the set of nodes reachable by repeatedly
    following edges backward (dst -> its sources) `n_hops` times — not a
    physical distance, but the literal graph neighbourhood the model used.
    """
    src, dst = edge_index[0], edge_index[1]
    predecessors: dict = {}
    for s, d in zip(src, dst):
        predecessors.setdefault(int(d), []).append(int(s))

    visited = {seed}
    frontier = {seed}
    for _ in range(n_hops):
        next_frontier = set()
        for node in frontier:
            for p in predecessors.get(node, []):
                if p not in visited:
                    visited.add(p)
                    next_frontier.add(p)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def _plot_node_concept_galleries(labels_all, all_node_core, all_node_types,
                                  best_k, emb_data, core_bbox, cell_types,
                                  output_dir, suffix, centroids, all_node_emb,
                                  n_hops: int,
                                  n_gallery: int = 6):
    """
    For each concept, show a small gallery of cropped local neighbourhoods —
    each crop is exactly the set of cells within `n_hops` of the model's own
    k-NN graph (i.e. the literal receptive field of that GCN embedding, not
    an arbitrary physical radius) — around several diverse exemplar cells
    (the nodes closest to that concept's centroid, at most one per patient),
    rather than one whole cherry-picked core. Each crop is coloured by cell
    type, with the cells assigned to the concept outlined in black. Panel
    extent varies with local cell density (that's honest — the model's own
    receptive field genuinely covers more physical area in sparser tissue),
    so each panel carries its own scale bar for cross-panel comparability.
    """
    # See _plot_node_umap for why the raw (unresampled) palette is used.
    cmap_celltypes = plt.cm.get_cmap("Set2")

    for k in range(best_k):
        idx_k = np.where(labels_all == k)[0]
        if len(idx_k) == 0:
            continue
        dists = np.linalg.norm(all_node_emb[idx_k] - centroids[k], axis=1)
        order = idx_k[np.argsort(dists)]

        # Greedily take the closest-to-centroid node from each distinct
        # patient, so the gallery shows independent instances of the concept
        # rather than several crops from the same patient's cores.
        exemplars = []
        seen_pids = set()
        for node_idx in order:
            core = all_node_core[node_idx]
            pid  = emb_data["pid"][core]
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            exemplars.append((node_idx, core, pid))
            if len(exemplars) == n_gallery:
                break
        if not exemplars:
            continue

        n_cols = min(3, len(exemplars))
        n_rows = -(-len(exemplars) // n_cols)  # ceil
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3.2 * n_cols, 3.2 * n_rows),
                                  squeeze=False)

        # Pass 1: compute each exemplar's true receptive-field crop and its
        # physical extent, so a sensible common scale-bar length can be
        # chosen before drawing (panel extents vary with local density).
        panels = []
        for node_idx, core, pid in exemplars:
            mask_core = all_node_core == core
            node_xy_norm = emb_data["node_xy"][core]          # (n_core, 2), [0,1]
            node_types_c = all_node_types[mask_core]
            node_label_c = labels_all[mask_core]
            edge_index_c = emb_data["edge_index"][core]

            # Map normalised coords back to approximate microns using this
            # core's own bounding box, purely for plotting in real units.
            bbox_min, bbox_max = core_bbox[core]
            span = bbox_max - bbox_min
            span = np.where(span == 0, 1.0, span)
            node_xy_um = node_xy_norm * span + bbox_min

            # `node_idx` is a global index into the concatenated node arrays;
            # since all_node_core/all_node_types/labels_all are built one core
            # at a time (in the same order as emb_data["node_xy"][core]), its
            # position within this core's own block also locates it in
            # node_xy_norm/node_xy_um/edge_index_c (all local to that core).
            local_idx = int(node_idx - np.flatnonzero(mask_core)[0])
            center_um = node_xy_um[local_idx]

            crop_idx = np.array(sorted(_receptive_field_nodes(
                edge_index_c, local_idx, n_hops)))

            xy_crop    = node_xy_um[crop_idx]
            types_crop = node_types_c[crop_idx]
            label_crop = node_label_c[crop_idx]
            extent_um  = float(np.linalg.norm(xy_crop - center_um, axis=1).max()) \
                         if len(crop_idx) > 1 else 1.0

            panels.append(dict(pid=pid, xy=xy_crop, types=types_crop,
                                labels=label_crop, center=center_um,
                                extent=extent_um))

        # Pick a "nice" round scale-bar length from the panels' own extents,
        # rather than a dataset-wide constant, so it stays legible whether
        # this is a sparse synthetic core or a dense real one.
        median_extent = float(np.median([p["extent"] for p in panels]))
        candidates = [10, 20, 25, 50, 100, 150, 200, 250, 500, 1000]
        scale_bar_um = min(candidates, key=lambda c: abs(c - median_extent / 3))

        for i, p in enumerate(panels):
            ax = axes[i // n_cols][i % n_cols]

            colors = [cmap_celltypes(t) for t in p["types"]]
            ax.scatter(p["xy"][:, 0], p["xy"][:, 1], c=colors, s=16,
                       alpha=0.85, linewidths=0)
            member = p["labels"] == k
            if member.any():
                ax.scatter(p["xy"][member, 0], p["xy"][member, 1],
                           facecolors=[cmap_celltypes(t) for t in p["types"][member]],
                           edgecolors="black", linewidths=0.8, s=22)

            local_frac = member.mean() if len(member) else 0.0
            ax.set_title(f"{p['pid']}\nn={len(p['labels'])}, frac={local_frac:.2f}",
                        fontsize=7)
            # Each panel is scaled to its own crop (receptive-field extent
            # varies with local density — that's real, not an artefact), with
            # a small margin; a shared scale bar keeps panels comparable.
            pad = 0.15 * p["extent"] + 1.0
            ax.set_xlim(p["center"][0] - p["extent"] - pad, p["center"][0] + p["extent"] + pad)
            ax.set_ylim(p["center"][1] - p["extent"] - pad, p["center"][1] + p["extent"] + pad)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

            x0 = p["center"][0] - p["extent"] - pad * 0.5
            y0 = p["center"][1] - p["extent"] - pad * 0.5
            ax.plot([x0, x0 + scale_bar_um], [y0, y0], color="black", linewidth=1.5)
            ax.text(x0 + scale_bar_um / 2, y0, f"{scale_bar_um} µm",
                    fontsize=6, ha="center", va="bottom")

        for j in range(len(exemplars), n_rows * n_cols):
            axes[j // n_cols][j % n_cols].axis("off")

        type_handles = [
            mpatches.Patch(color=cmap_celltypes(i), label=ct)
            for i, ct in enumerate(cell_types)
        ]
        member_handle = plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor="none",
                                    markeredgecolor="black",
                                    markersize=7, label=f"Concept {k} member")
        fig.legend(handles=type_handles + [member_handle],
                  fontsize=7, ncol=len(cell_types) + 1,
                  loc="lower center", bbox_to_anchor=(0.5, -0.02),
                  framealpha=0.5)
        fig.suptitle(
            f"Concept {k} — {n_hops}-hop receptive-field neighbourhoods "
            f"from {len(exemplars)} patients", fontsize=10
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))

        fname = f"fig_node_concept_{k}_gallery{_sfx(suffix)}"
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                        bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved node concept galleries (0–{best_k-1})")


# ── Utility ───────────────────────────────────────────────────────────────────

def _sfx(suffix: str) -> str:
    return f"_{suffix}" if suffix else ""
