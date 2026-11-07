"""
Ground-truth high/low signal example cores for the synthetic regimes.

The synthetic `signal` column is a per-patient scalar (e.g. immune fraction,
clustering tightness) — it describes a whole-core spatial pattern, not any
single cell. So unlike the GCExplainer node-level galleries (which crop small
receptive-field neighbourhoods around individual cells, since that's what the
GCN embedding actually sees), the natural "ground truth" comparison is the
whole core: one example with the highest signal (best prognosis by
construction) and one with the lowest, per regime.

These are meant to sit next to the graph-level concept representative-core
plots (fig_graph_concept_{k}_core_*) for a direct visual check of whether a
discovered concept resembles the true high- or low-signal pattern. Colours
match the GCExplainer galleries' Set2 colormap (same cell-type -> colour
mapping), rather than the tab10 palette used elsewhere in this file, so the
two can be compared side by side without a colour-mapping mismatch.

Usage:
    uv run python -m analysis.explainability.signal_examples
    uv run python -m analysis.explainability.signal_examples --regimes 3 4
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_OUTPUT_DIR = "figures/output/explainability"
CELL_TYPES = ["tumor", "immune"]  # matches data_utils.CELL_TYPES_BINARY ordering

REGIME_NAMES = {
    1: "composition",
    2: "proximity",
    3: "clustering",
    4: "ripley",
}


def _plot_signal_example(df: pd.DataFrame, pid: int, signal: float, which: str,
                          regime: int, output_dir: str):
    """One whole-core scatter plot, styled to match the GCExplainer galleries."""
    sub = df[df["PID"] == pid]
    core = sub["core"].iloc[0]

    # NB: plt.cm.get_cmap("Set2", 2) resamples the 8-colour qualitative
    # palette down to 2 by taking the first *and last* entries (teal, then
    # grey) rather than the first two — grey at small marker size/alpha is
    # low-contrast on white. Index the raw 8-colour palette directly instead,
    # so both classes get a vivid, well-separated Set2 hue (teal vs orange).
    set2 = plt.cm.get_cmap("Set2")
    ct_to_idx = {ct: i for i, ct in enumerate(CELL_TYPES)}

    fig, ax = plt.subplots(figsize=(4, 4))
    for ct in CELL_TYPES:
        cells = sub[sub["cell_type"] == ct]
        if cells.empty:
            continue
        ax.scatter(cells["x"], cells["y"], c=[set2(ct_to_idx[ct])], s=4,
                   alpha=0.7, linewidths=0, label=ct)

    label = "High" if which == "high" else "Low"
    ax.set_title(f"{label} signal example core\nPID={pid}  signal={signal:.3f}",
                fontsize=9)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.legend(fontsize=7, markerscale=2, loc="upper right", framealpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fname = f"fig_signal_example_{which}_r{regime}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.pdf/png  (core={core})")


def plot_signal_examples(regime: int, output_dir: str, data_dir: str = "datasets",
                          noise: float = 0.0):
    """High- and low-signal example cores for one synthetic regime."""
    path = os.path.join(data_dir, f"synthetic_regime{regime}_noise{noise:.1f}.parquet")
    df = pd.read_parquet(path)

    patient_signal = df.groupby("PID")["signal"].first()
    pid_high = patient_signal.idxmax()
    pid_low  = patient_signal.idxmin()

    print(f"Regime {regime} ({REGIME_NAMES.get(regime, '?')}): "
          f"high PID={pid_high} signal={patient_signal[pid_high]:.3f}, "
          f"low PID={pid_low} signal={patient_signal[pid_low]:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    _plot_signal_example(df, pid_high, patient_signal[pid_high], "high", regime, output_dir)
    _plot_signal_example(df, pid_low,  patient_signal[pid_low],  "low",  regime, output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Ground-truth high/low signal example cores")
    p.add_argument("--regimes", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--noise", type=float, default=0.0,
                   help="Noise level to load (signal and point patterns are "
                        "identical across noise levels for a given regime — "
                        "only the simulated survival time/event differ — so "
                        "this just selects which generated file to read)")
    p.add_argument("--data_dir", default="datasets")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    for regime in args.regimes:
        plot_signal_examples(regime, args.output_dir, args.data_dir, args.noise)


if __name__ == "__main__":
    main()
