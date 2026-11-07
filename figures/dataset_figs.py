"""
Dataset and model figures.

Figure 1: Example tissue core and its k-NN graph
Figure 2: Dataset overview — cells per core, cell type composition, KM curves
Figure 3: Train / validation split balance
Figure 5: GNN model architecture schematic
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split

from data_utils import prepare_survival_data, make_stratify_labels
from figures import save_fig


# ── Colour palette ─────────────────────────────────────────────────────────────

CELL_COLORS = {
    "tumor": "#4878CF",
    "T_cell": "#6ACC65",
    "B_cell": "#B47CC7",
    "macrophage": "#D65F5F",
    "stromal": "#B8B8B8",
    "endothelial": "#4EAEC9",
    "immune": "#D65F5F",  # binary (synthetic) alias
}

SPLIT_COLORS = {"Train": "#4878CF", "Val": "#D65F5F"}


# Preferred display order for cell types
CELL_TYPE_ORDER = [
    "tumor",
    "T_cell",
    "B_cell",
    "macrophage",
    "stromal",
    "endothelial",
    "immune",
]


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _patient_split_pids(patient_df, val_size=0.2, seed=42):
    """Return (train_pids, val_pids) sets using the same joint stratification
    as data_utils.patient_split and clinical_baseline.split_patients.

    Stratifies on clinical subtype + grade so the split is stable across
    endpoints and balanced on the strongest prognostic variables.
    Rare combinations (< 2 patients) are collapsed into 'Other'.
    """
    pids = sorted(patient_df["PID"].unique())
    subtype = patient_df.groupby("PID")["clinical_type"].first().fillna("Unknown")
    grade = patient_df.groupby("PID")["grade"].first()
    grade_str = grade.apply(lambda g: str(int(g)) if pd.notna(g) else "Unknown")
    label_s = make_stratify_labels(subtype, grade_str)
    stratify = [label_s[p] for p in pids]
    train_pids, val_pids = train_test_split(
        pids, test_size=val_size, random_state=seed, stratify=stratify
    )
    return set(train_pids), set(val_pids)


def _patient_level(df):
    """
    Aggregate to one row per patient.

    Returns a DataFrame with:
    - PID, grade, tumor_size, age, ERStatus, PRStatus, HER2Status, clinical_type
    - frac_<cell_type> columns (fraction of all cells for that patient)
    - n_cores (number of TMA cores)
    """
    meta_cols = [
        c
        for c in [
            "PID",
            "grade",
            "tumor_size",
            "age",
            "ERStatus",
            "PRStatus",
            "HER2Status",
            "clinical_type",
            "Patientstatus",
            "DFSmonth",
            "OSmonth",
            "cohort",
        ]
        if c in df.columns
    ]
    pat = df.groupby("PID")[meta_cols].first().reset_index(drop=True)

    # Cell type fractions (across all cores for that patient)
    ct_counts = df.groupby(["PID", "cell_type"]).size().unstack(fill_value=0)
    ct_frac = ct_counts.div(ct_counts.sum(axis=1), axis=0)
    ct_frac.columns = [f"frac_{c}" for c in ct_frac.columns]
    ct_frac = ct_frac.reset_index()

    pat = pat.merge(ct_frac, on="PID", how="left")

    # Number of cores per patient
    cores_pp = df.groupby("PID")["core"].nunique().rename("n_cores")
    pat = pat.merge(cores_pp, on="PID", how="left")

    return pat


# ── Figure 1: Example tissue cores ────────────────────────────────────────────


def _pick_diverse_cores(df, n=3, min_cells=400, max_cells=5000):
    """
    Pick n cores spanning low / medium / high tumour fraction.
    Prefers moderate-size cores for visual clarity.
    """
    stats = []
    for core, grp in df.groupby("core"):
        n_total = len(grp)
        if not (min_cells <= n_total <= max_cells):
            continue
        tumor_frac = (grp["cell_type"] == "tumor").mean()
        stats.append(
            {
                "core": core,
                "n_cells": n_total,
                "tumor_frac": tumor_frac,
                "PID": grp["PID"].iloc[0],
                "subtype": grp["clinical_type"].iloc[0]
                if "clinical_type" in grp.columns
                else "?",
                "OSmonth": grp["OSmonth"].iloc[0] if "OSmonth" in grp.columns else None,
                "Patientstatus": grp["Patientstatus"].iloc[0]
                if "Patientstatus" in grp.columns
                else "",
            }
        )

    stats_df = pd.DataFrame(stats).sort_values("tumor_frac").reset_index(drop=True)

    # Divide into n equal buckets, pick core nearest each bucket's median
    bucket_size = len(stats_df) // n
    selected = []
    for b in range(n):
        lo = b * bucket_size
        hi = (b + 1) * bucket_size if b < n - 1 else len(stats_df)
        bucket = stats_df.iloc[lo:hi]
        median_frac = bucket["tumor_frac"].median()
        idx = (bucket["tumor_frac"] - median_frac).abs().idxmin()
        selected.append(stats_df.loc[idx])

    return selected  # list of Series


def plot_dataset_overview(df, output_dir):
    """
    Figure 2: Three-panel overview.
      A  Cells per core — histogram
      B  Cell-type composition — waterfall (patients sorted by tumour fraction)
      C  Kaplan-Meier OS and DFS for the full cohort
    """
    os.makedirs(output_dir, exist_ok=True)

    df_os = prepare_survival_data(df, endpoint="OS")
    df_pfs = prepare_survival_data(df, endpoint="DFS")

    pat = _patient_level(df)

    # Add OS / DFS survival columns at patient level
    os_ev = (
        df_os.groupby("PID")[["time", "event"]]
        .first()
        .rename(columns={"time": "os_time", "event": "os_event"})
    )
    pfs_ev = (
        df_pfs.groupby("PID")[["time", "event"]]
        .first()
        .rename(columns={"time": "pfs_time", "event": "pfs_event"})
    )
    pat = pat.join(os_ev, on="PID").join(pfs_ev, on="PID")

    cells_per_core = df.groupby("core").size()

    cell_types_present = [ct for ct in CELL_TYPE_ORDER if f"frac_{ct}" in pat.columns]

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(
        2,
        3,
        hspace=0.45,
        wspace=0.35,
        height_ratios=[1, 1.1],
    )
    ax_hist = fig.add_subplot(gs[0, 0])  # A
    ax_comp = fig.add_subplot(gs[0, 1:])  # B  (wider)
    ax_km = fig.add_subplot(gs[1, :])  # C  (full width)

    # ── Panel A: cells per core ─────────────────────────────────────────────────
    ax_hist.hist(
        cells_per_core.values,
        bins=35,
        color="#4878CF",
        edgecolor="white",
        linewidth=0.4,
    )
    med = cells_per_core.median()
    ax_hist.axvline(
        med, color="#D65F5F", linestyle="--", linewidth=1.3, label=f"Median {med:.0f}"
    )
    ax_hist.set_xlabel("Cells per core", fontsize=14)
    ax_hist.set_ylabel("Number of cores", fontsize=14)
    ax_hist.set_title("A   Cells per core", fontsize=14, fontweight="bold", loc="left")
    ax_hist.legend(fontsize=12, frameon=False)
    ax_hist.tick_params(labelsize=12)

    # ── Panel B: composition waterfall ──────────────────────────────────────────
    tumor_col = "frac_tumor"
    if tumor_col in pat.columns:
        pat_sorted = pat.sort_values(tumor_col, ascending=False).reset_index(drop=True)
    else:
        pat_sorted = pat.reset_index(drop=True)

    bottom = np.zeros(len(pat_sorted))
    for ct in cell_types_present:
        vals = pat_sorted[f"frac_{ct}"].fillna(0).values
        ax_comp.bar(
            range(len(pat_sorted)),
            vals,
            bottom=bottom,
            color=CELL_COLORS.get(ct, "#999"),
            width=1.0,
            linewidth=0,
            label=ct.replace("_", " "),
        )
        bottom += vals

    ax_comp.set_xlim(-0.5, len(pat_sorted) - 0.5)
    ax_comp.set_ylim(0, 1)
    ax_comp.set_xlabel(
        f"Patients  (n={len(pat_sorted)}, sorted by tumour fraction)", fontsize=14
    )
    ax_comp.set_ylabel("Cell-type fraction", fontsize=14)
    ax_comp.set_title(
        "B   Cell-type composition per patient",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax_comp.set_xticks([])
    ax_comp.tick_params(labelsize=12)
    handles = [
        mpatches.Patch(color=CELL_COLORS.get(ct, "#999"), label=ct.replace("_", " "))
        for ct in cell_types_present
    ]
    ax_comp.legend(
        handles=handles, fontsize=12, frameon=False, loc="upper right", ncol=2
    )

    # ── Panel C: KM OS + DFS ────────────────────────────────────────────────────
    pat_os = pat.dropna(subset=["os_time", "os_event"])
    pat_pfs = pat.dropna(subset=["pfs_time", "pfs_event"])

    kmf_os = KaplanMeierFitter()
    kmf_os.fit(
        pat_os["os_time"],
        event_observed=pat_os["os_event"],
        label=f"OS  (events {int(pat_os['os_event'].sum())}/{len(pat_os)})",
    )
    kmf_os.plot_survival_function(ax=ax_km, ci_show=True, color="#4878CF", linewidth=2)

    kmf_pfs = KaplanMeierFitter()
    kmf_pfs.fit(
        pat_pfs["pfs_time"],
        event_observed=pat_pfs["pfs_event"],
        label=f"DFS  (events {int(pat_pfs['pfs_event'].sum())}/{len(pat_pfs)})",
    )
    kmf_pfs.plot_survival_function(ax=ax_km, ci_show=True, color="#D65F5F", linewidth=2)

    ax_km.set_xlabel("Time (months)", fontsize=14)
    ax_km.set_ylabel("Survival probability", fontsize=14)
    ax_km.set_ylim(0, 1.05)
    ax_km.set_title(
        "C   Kaplan–Meier survival curves", fontsize=14, fontweight="bold", loc="left"
    )
    ax_km.tick_params(labelsize=12)
    ax_km.legend(fontsize=13, frameon=False)

    fig.suptitle(
        "Dataset overview — Jackson et al. (2020) IMC cohort",
        fontsize=17,
        fontweight="bold",
    )

    out = os.path.join(output_dir, "fig2_dataset_overview.pdf")
    save_fig(fig, out)
    plt.close(fig)


# ── Figure 3: Split balance ────────────────────────────────────────────────────


def plot_split_balance(df, output_dir, val_size=0.2, seed=42):
    """
    Figure 3: Four-panel comparison of train vs validation on
    grade, clinical subtype, event rates, and age.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_os = prepare_survival_data(df, endpoint="OS")
    df_pfs = prepare_survival_data(df, endpoint="DFS")

    pat = _patient_level(df)
    os_ev = (
        df_os.groupby("PID")[["time", "event"]]
        .first()
        .rename(columns={"time": "os_time", "event": "os_event"})
    )
    pfs_ev = (
        df_pfs.groupby("PID")[["time", "event"]]
        .first()
        .rename(columns={"time": "pfs_time", "event": "pfs_event"})
    )
    pat = pat.join(os_ev, on="PID").join(pfs_ev, on="PID")

    train_pids, val_pids = _patient_split_pids(pat, val_size=val_size, seed=seed)
    pat["split"] = pat["PID"].apply(lambda p: "Train" if p in train_pids else "Val")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    w = 0.32  # bar width

    # ── A: Grade ───────────────────────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(3)
    for offset, (split, color) in zip([-w / 2, w / 2], SPLIT_COLORS.items()):
        sub = pat[pat["split"] == split]
        n = len(sub)
        pcts = [100 * (sub["grade"] == g).mean() for g in [1, 2, 3]]
        ax.bar(
            x + offset,
            pcts,
            width=w,
            color=color,
            label=f"{split} (n={n})",
            alpha=0.85,
            linewidth=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Grade 1", "Grade 2", "Grade 3"], fontsize=13)
    ax.set_ylabel("Patients (%)", fontsize=14)
    ax.set_title("A   Tumour grade", fontsize=14, fontweight="bold", loc="left")
    ax.legend(fontsize=12, frameon=False)
    ax.set_ylim(0, 65)
    ax.tick_params(labelsize=12)

    # ── B: Clinical subtype ────────────────────────────────────────────────────
    ax = axes[1]
    subtypes = ["HR+HER2-", "TripleNeg", "HR+HER2+", "HR-HER2+"]
    x = np.arange(len(subtypes))
    for offset, (split, color) in zip([-w / 2, w / 2], SPLIT_COLORS.items()):
        sub = pat[pat["split"] == split]
        pcts = [100 * (sub["clinical_type"] == s).mean() for s in subtypes]
        ax.bar(x + offset, pcts, width=w, color=color, alpha=0.85, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(subtypes, fontsize=12, rotation=15, ha="right")
    ax.set_ylabel("Patients (%)", fontsize=14)
    ax.set_title("B   Clinical subtype", fontsize=14, fontweight="bold", loc="left")
    ax.tick_params(labelsize=12)

    # ── C: Event rates ─────────────────────────────────────────────────────────
    ax = axes[2]
    x = np.arange(2)
    for offset, (split, color) in zip([-w / 2, w / 2], SPLIT_COLORS.items()):
        sub = pat[pat["split"] == split]
        pcts = [100 * sub["os_event"].mean(), 100 * sub["pfs_event"].mean()]
        ax.bar(x + offset, pcts, width=w, color=color, alpha=0.85, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["OS events", "DFS events"], fontsize=13)
    ax.set_ylabel("Patients (%)", fontsize=14)
    ax.set_title("C   Event rates", fontsize=14, fontweight="bold", loc="left")
    ax.set_ylim(0, 55)
    ax.tick_params(labelsize=12)

    # ── D: Age distribution ────────────────────────────────────────────────────
    ax = axes[3]
    for split, color in SPLIT_COLORS.items():
        ages = pat[pat["split"] == split]["age"].dropna()
        ax.hist(
            ages.values,
            bins=14,
            color=color,
            alpha=0.55,
            label=split,
            density=True,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("Age (years)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("D   Age distribution", fontsize=14, fontweight="bold", loc="left")
    ax.legend(fontsize=12, frameon=False)
    ax.tick_params(labelsize=12)

    fig.suptitle("Train / validation split balance", fontsize=17, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(output_dir, "fig3_split_balance.pdf")
    save_fig(fig, out)
    plt.close(fig)


def plot_core_and_knn_graph(df, output_dir, k=12, max_cells=300, seed=42):
    """
    Figure 1: the example core (all cells) and the k-NN
    graph of that core subsampled to max_cells cells, with one shared legend.

    Cell types use the Set2 palette indexed in CELL_TYPES_METACLUSTER order,
    matching the IMC explainability figures (concept_discovery.py). Edges are
    built with the same _knn_graph call and min-max coordinate normalisation
    as create_pytorch_dataset, so the edge count in the title is the number
    of directed edges the GCN would see for this subsample.
    """
    import torch
    from data_utils import CELL_TYPES_METACLUSTER, _knn_graph

    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.get_cmap("Set2")
    colors = {ct: cmap(i) for i, ct in enumerate(CELL_TYPES_METACLUSTER)}

    # Second of three diverse example cores
    selected = _pick_diverse_cores(df, n=3, min_cells=400, max_cells=4000)
    core_df = df[df["core"] == selected[1]["core"]].reset_index(drop=True)
    present = [ct for ct in CELL_TYPES_METACLUSTER if ct in set(core_df["cell_type"])]

    rng = np.random.default_rng(seed)
    if len(core_df) > max_cells:
        sub = core_df.iloc[rng.choice(len(core_df), size=max_cells, replace=False)]
        sub = sub.reset_index(drop=True)
    else:
        sub = core_df

    coords = sub[["x", "y"]].values.astype(float)
    rng_xy = coords.max(axis=0) - coords.min(axis=0)
    rng_xy[rng_xy == 0] = 1
    norm = (coords - coords.min(axis=0)) / rng_xy
    edge_index = _knn_graph(torch.tensor(norm, dtype=torch.float32), k=k, loop=False).numpy()
    n_edges = edge_index.shape[1]

    fig, (ax_core, ax_graph) = plt.subplots(1, 2, figsize=(10.5, 6.5), sharey=True)

    for ct in present:
        cells = core_df[core_df["cell_type"] == ct]
        ax_core.scatter(cells["x"], cells["y"], color=colors[ct], s=7,
                        alpha=0.85, linewidths=0, rasterized=True)
    ax_core.set_title(f"Tissue core ({len(core_df):,} cells)", fontsize=14)

    segs = np.stack([coords[edge_index[0]], coords[edge_index[1]]], axis=1)
    from matplotlib.collections import LineCollection
    ax_graph.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=0.4,
                                           alpha=0.6, zorder=1))
    types = sub["cell_type"].values
    for ct in present:
        m = types == ct
        ax_graph.scatter(coords[m, 0], coords[m, 1], color=colors[ct], s=18,
                         linewidths=0, zorder=2)
    ax_graph.set_title(f"k-NN graph (k={k}, {len(sub)} cells, {n_edges:,} edges)",
                       fontsize=14)

    # Pad by 2% so markers on the core boundary are not clipped by the axes
    pad = 0.02 * max(np.ptp(core_df["x"]), np.ptp(core_df["y"]))
    xlim = (core_df["x"].min() - pad, core_df["x"].max() + pad)
    ylim = (core_df["y"].min() - pad, core_df["y"].max() + pad)
    for ax in (ax_core, ax_graph):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_core.set_ylabel("y (µm)", fontsize=13)

    handles = [mpatches.Patch(color=colors[ct], label=ct.replace("_", " ")) for ct in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, -0.035))
    plt.tight_layout(rect=[0, 0.07, 1, 1], w_pad=0.5)

    print(f"  core {selected[1]['core']}: {len(core_df)} cells, subsample {len(sub)}, "
          f"{n_edges} directed edges")
    save_fig(fig, os.path.join(output_dir, "fig1_core_knn_graph.pdf"))
    plt.close(fig)


def plot_model_architecture(output_dir):
    """
    Figure 5: Schematic of the GNN survival model pipeline.

    Layout (all boxes on y=2, clinical features drops in from above at the
    concat junction):

      Tissue Core → Node Feats → k-NN Graph → GCN Layers → Global Pool ─┐
                                                                          ⊕ → Cox Risk
                                            Clinical Features ────────────┘
    """
    os.makedirs(output_dir, exist_ok=True)

    # x-centres of main-flow boxes
    XS = [0.9, 2.4, 3.9, 5.5, 7.1, 8.7, 10.3]
    # XS[5] = concat junction (circle), XS[6] = Cox risk score
    Y = 2.0  # main flow y
    Y_CLIN = 3.4  # clinical features box y (above)

    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    ax.set_xlim(-0.1, 11.3)
    ax.set_ylim(0.1, 4.3)
    ax.axis("off")

    def box(cx, cy, w, h, label, sublabel=None, color="#4878CF"):
        rect = plt.Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            facecolor=color,
            edgecolor="white",
            linewidth=1.8,
            alpha=0.88,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            cy + (0.16 if sublabel else 0),
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
            zorder=3,
        )
        if sublabel:
            ax.text(
                cx,
                cy - 0.22,
                sublabel,
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                alpha=0.9,
                zorder=3,
            )

    def harrow(x0, x1, y=Y):
        """Horizontal arrow from x0 to x1 at height y."""
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.3),
            zorder=1,
        )

    def varrow(x, y0, y1):
        """Vertical arrow from y0 to y1 at x."""
        ax.annotate(
            "",
            xy=(x, y1),
            xytext=(x, y0),
            arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.3),
            zorder=1,
        )

    BW = 1.2  # box width
    BH = 1.15  # box height

    # 1. Tissue core
    box(XS[0], Y, BW, BH, "Tissue Core", "cells (x,y,type)", color="#888888")
    harrow(XS[0] + BW / 2, XS[1] - BW / 2)

    # 2. Node features
    box(XS[1], Y, BW, BH, "Node Features", "x, y + 6-dim one-hot", color="#6688BB")
    harrow(XS[1] + BW / 2, XS[2] - BW / 2)

    # 3. k-NN graph
    box(XS[2], Y, BW, BH, "k-NN Graph", "k = 12", color="#6688BB")
    harrow(XS[2] + BW / 2, XS[3] - BW / 2)

    # 4. GCN layers
    box(XS[3], Y, BW, BH, "GCN Layers", "×3, ReLU + BN", color="#4878CF")
    harrow(XS[3] + BW / 2, XS[4] - BW / 2)

    # 5. Global mean pool
    box(XS[4], Y, BW, BH, "Global\nMean Pool", "graph → vector", color="#4878CF")
    harrow(XS[4] + BW / 2, XS[5] - 0.22)  # → concat circle

    # 6. Concat circle
    circle = plt.Circle((XS[5], Y), 0.22, color="#555555", zorder=2)
    ax.add_patch(circle)
    ax.text(
        XS[5], Y, "⊕", ha="center", va="center", fontsize=17, color="white", zorder=3
    )
    harrow(XS[5] + 0.22, XS[6] - BW / 2)  # concat → Cox

    # Clinical features box above the concat junction
    box(
        XS[5],
        Y_CLIN,
        BW,
        0.85,
        "Clinical Features",
        "grade, ER, HER2…",
        color="#D65F5F",
    )
    varrow(XS[5], Y_CLIN - 0.85 / 2, Y + 0.22)  # drop down into concat

    # NND + Ripley features box below the concat junction
    Y_NND = 0.72
    box(
        XS[5],
        Y_NND,
        BW,
        0.75,
        "NND + Ripley",
        "spatial summary",
        color="#5E9080",
    )
    varrow(XS[5], Y_NND + 0.75 / 2, Y - 0.22)  # rise up into concat

    # 7. Cox risk score
    box(XS[6], Y, BW, BH, "Cox Risk Score", "scalar θ", color="#C75454")

    # Loss label below
    ax.text(
        XS[6],
        Y - BH / 2 - 0.15,
        "Cox PH loss (partial likelihood)",
        ha="center",
        va="top",
        fontsize=11,
        color="#888888",
    )

    ax.set_title(
        "GNN Survival Model Architecture", fontsize=17, fontweight="bold", pad=6
    )
    plt.tight_layout()

    out = os.path.join(output_dir, "fig5_model_architecture.pdf")
    save_fig(fig, out)
    plt.close(fig)
