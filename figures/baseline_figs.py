"""
Cox PH baseline figures.

Figure 6b: Forest plot of the clinical Cox PH model (single 80/20 split)
Figure 7:  Model comparison bar chart (5-fold CV, produced by run_cv.py)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clinical_baseline import (
    load_patient_data,
    split_patients,
    CLINICAL_FEATURES,
)
from lifelines import CoxPHFitter

from figures import save_fig


def plot_cox_forest(
    df, output_dir, endpoint="OS", seed=42, penalizer=0.01
):
    """
    Figure 6b: Forest plot of hazard ratios from the clinical Cox PH model.

    Fitted on the training set. Features sorted by absolute log-HR.
    Highlights grade and subtype features used for stratification.
    """
    os.makedirs(output_dir, exist_ok=True)

    patient_df = load_patient_data(data_path=None, endpoint=endpoint, df=df)
    patient_df = patient_df.dropna(subset=["grade", "tumor_size", "age"])
    train, _ = split_patients(patient_df, seed=seed)

    cols = ["time", "event"] + CLINICAL_FEATURES
    tr = train[cols].dropna()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(tr, duration_col="time", event_col="event")

    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["hr", "lower", "upper", "p"]

    # Human-readable labels
    label_map = {
        "grade":      "Grade",
        "tumor_size": "Tumour size",
        "age":        "Age",
        "ER_pos":     "ER positive",
        "PR_pos":     "PR positive",
        "HER2_pos":   "HER2 positive",
    }
    summary.index = [label_map.get(f, f) for f in summary.index]

    # Sort by absolute log-HR descending
    summary["abs_log_hr"] = np.abs(np.log(summary["hr"]))
    summary = summary.sort_values("abs_log_hr")

    # Features used for stratification (grade directly; subtype is encoded via ER/PR/HER2)
    strat_features = {"Grade", "ER positive", "PR positive", "HER2 positive"}

    fig, ax = plt.subplots(figsize=(5.2, 0.6 * len(summary) + 2.0))

    y = np.arange(len(summary))
    for i, (feat, row) in enumerate(summary.iterrows()):
        color = "#4878CF" if feat in strat_features else "#555555"
        sig = row["p"] < 0.05

        ax.plot([row["lower"], row["upper"]], [i, i],
                color=color, linewidth=1.5, zorder=1)
        ax.scatter(row["hr"], i,
                   color=color, s=60 if sig else 35,
                   zorder=2, marker="D" if sig else "o")

        # p-value annotation to the right of CI
        if row["p"] < 0.001:
            p_label = "p<0.001"
        elif row["p"] < 0.01:
            p_label = f"p={row['p']:.3f}"
        elif sig:
            p_label = f"p={row['p']:.2f}"
        else:
            p_label = f"p={row['p']:.2f}"
        ax.text(row["upper"] + 0.03, i, p_label,
                va="center", ha="left", fontsize=14, color=color)

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index, fontsize=15)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=15)
    ax.set_xscale("log")
    ax.set_title(
        f"Clinical Cox PH, {endpoint}",
        fontsize=15, fontweight="bold",
    )

    # Legend: colour marks the stratification variables, marker shape marks
    # significance, so the two are not conflated in one entry.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#4878CF", linewidth=2,
               label="Stratification variable"),
        Line2D([0], [0], color="#555555", linewidth=2,
               label="Other covariate"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
               markersize=8, label="p < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=7, label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements, fontsize=13, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.35, -0.18))
    ax.tick_params(labelsize=14)

    plt.tight_layout()
    out = os.path.join(output_dir, f"fig6b_cox_forest_{endpoint.lower()}.pdf")
    save_fig(fig, out)
    plt.close(fig)


# Figure 7 bars, top to bottom: (label, source, key, colour). source "cox"
# keys are run_cv.COX_MODELS, "gnn" keys are run_cv conditions.
MODEL_COMPARISON_BARS = [
    ("NND (Cox PH)",                                     "cox", "nnd",                 "#aaaaaa"),
    ("Ripley (Cox PH)",                                  "cox", "ripley",              "#aaaaaa"),
    ("NND + Ripley (Cox PH)",                            "cox", "nnd_ripley",          "#888888"),
    ("Clinical (Cox PH)",                                "cox", "clinical",            "#4878CF"),
    ("Clinical + NND (Cox PH)",                          "cox", "clinical_nnd",        "#3C6AB2"),
    ("Clinical + Ripley (Cox PH)",                       "cox", "clinical_ripley",     "#3C6AB2"),
    ("Clinical + NND + Ripley (Cox PH)",                 "cox", "clinical_nnd_ripley", "#2F5C94"),
    ("GNN only\n(graph embedding, no pretrain)",         "gnn", "gnn_only",            "#E8A04C"),
    ("GNN only\n(graph embedding, pretrain fine-tune)",  "gnn", "gnn_only_finetune",   "#E8A04C"),
    ("GNN only\n(graph embedding, pretrain frozen)",     "gnn", "gnn_only_frozen",     "#E8A04C"),
    ("GNN\n(NND+Ripley+clinical, no pretrain)",          "gnn", "no_pretrain",         "#D65F5F"),
    ("GNN\n(NND+Ripley+clinical, pretrain fine-tune)",   "gnn", "pretrain_finetune",   "#D65F5F"),
    ("GNN\n(NND+Ripley+clinical, pretrain frozen)",      "gnn", "pretrain_frozen",     "#D65F5F"),
]


def plot_model_comparison(output_dir: str, cox: dict, gnn: dict, endpoint: str = "DFS"):
    """
    Figure 7: Horizontal bar chart comparing CV C-index (mean ± SD) across models.

    cox: {key: (mean, sd)} for the Cox PH baselines (grey: spatial only,
         blue: with clinical features). Clinical + NND + Ripley is the fairest
         classical comparison for the full GNN, whose bottleneck has the same
         features.
    gnn: {condition: (mean, sd)} for the GNN variants (orange: GCN embedding
         only, red: full bottleneck). Missing or None entries are skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    sources = {"cox": cox, "gnn": gnn}

    # (label, mean, sd, color)
    models = [("Random baseline", 0.500, None, "#cccccc")]
    for label, source, key, color in MODEL_COMPARISON_BARS:
        mean, sd = sources[source].get(key, (None, None))
        models.append((label, mean, sd, color))

    models = [(l, v, s, c) for l, v, s, c in models if v is not None]

    labels = [m[0] for m in models]
    values = [m[1] for m in models]
    sds    = [m[2] for m in models]
    colors = [m[3] for m in models]

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(models) + 2.0))

    bars = ax.barh(labels, values, color=colors, alpha=0.85, linewidth=0)

    for bar, val, sd in zip(bars, values, sds):
        y = bar.get_y() + bar.get_height() / 2
        if sd is not None:
            ax.errorbar(val, y, xerr=sd, fmt="none", color="black",
                        capsize=3, linewidth=1.2)
        ax.text(
            val + (sd if sd else 0) + 0.007, y,
            f"{val:.3f}" + (f" ±{sd:.3f}" if sd else ""),
            va="center", ha="left", fontsize=12,
        )

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Random (0.5)")
    clinical_val = cox["clinical"][0]
    clinical_nnd_ripley_val = cox["clinical_nnd_ripley"][0]
    ax.axvline(clinical_val, color="#4878CF", linestyle=":", linewidth=1.2,
               label=f"Clinical (Cox PH) ({clinical_val:.3f})")
    ax.axvline(clinical_nnd_ripley_val, color="#2F5C94", linestyle=":", linewidth=1.2,
               label=f"Clinical + NND + Ripley (Cox PH) ({clinical_nnd_ripley_val:.3f})")

    ax.set_xlabel("Validation C-index (Harrell)", fontsize=14)
    ax.set_xlim(0.38, max(values) + 0.14)
    ax.set_title(f"Model comparison — {endpoint} endpoint", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.tick_params(labelsize=13)
    ax.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(output_dir, f"fig7_model_comparison_{endpoint.lower()}.pdf")
    save_fig(fig, out)
    plt.close(fig)
