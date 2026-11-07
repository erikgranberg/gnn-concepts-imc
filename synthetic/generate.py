"""
Synthetic point pattern dataset generation.

Generates a parquet file in the same format as the real metaclusters dataset
so it can be fed directly into the training and evaluation pipeline. Each of
300 patients gets one core of tumour and immune points; survival is drawn
from a Cox exponential model driven by a per-patient spatial signal s:

  1  composition            — s = immune cell fraction
  2  proximity              — s = fraction of immune cells placed near tumour cells
  3  anchored clustering    — s = tightness of immune Thomas clusters around tumour clusters
  4  independent clustering — s = tightness of immune Thomas clusters at random locations

The defaults (beta = 5, noise tau = 0, seed 42 + regime) reproduce the
datasets used in the thesis exactly.

Usage:
    uv run python synthetic/generate.py                 # regimes 1-4, tau = 0
    uv run python synthetic/generate.py --regime 3

Output:
    datasets/synthetic_regime{N}_noise{tau}.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make project root importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthetic.processes import (
    regime1_composition,
    regime2_proximity,
    regime3_clustering,
    regime4_ripley,
)
from synthetic.survival import simulate_survival


REGIME_FNS = {
    1: regime1_composition,
    2: regime2_proximity,
    3: regime3_clustering,
    4: regime4_ripley,
}

REGIME_NAMES = {
    1: "composition",
    2: "proximity",
    3: "clustering",
    4: "ripley",
}


def generate_patient(
    pid: int,
    regime_fn,
    rng: np.random.Generator,
    n_cells: int,
    regime_kwargs: dict,
) -> tuple[pd.DataFrame, float]:
    """
    Generate one patient's point pattern.

    Returns:
        df:     Cell-level DataFrame (one row per cell).
        signal: The latent spatial signal value for this patient.
    """
    tumor_coords, immune_coords, config = regime_fn(rng=rng, n_cells=n_cells,
                                                    **regime_kwargs)

    # Scale coordinates to micron range matching real IMC data (~800 µm cores).
    # This ensures NND thresholds in nnd_analysis.py (12 µm adjacent,
    # 250 µm communicating) produce meaningful variance across patients.
    scale = 800.0
    rows = []
    for i, (x, y) in enumerate(tumor_coords):
        rows.append({"x": x * scale, "y": y * scale, "cell_type": "tumor"})
    for i, (x, y) in enumerate(immune_coords):
        rows.append({"x": x * scale, "y": y * scale, "cell_type": "immune"})

    df = pd.DataFrame(rows)
    df["CellId"] = range(len(df))
    df["PID"]    = pid
    df["core"]   = f"P{pid:04d}_core1"

    return df, config.signal


def generate_dataset(
    regime: int,
    n_patients: int,
    n_cells: int,
    beta: float,
    noise: float,
    censoring_rate: float,
    seed: int,
    regime_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Generate a full synthetic dataset for one regime and noise level.

    Returns a cell-level DataFrame matching the real data column schema.
    """
    rng = np.random.default_rng(seed)
    regime_fn = REGIME_FNS[regime]
    regime_kwargs = regime_kwargs or {}

    cell_dfs = []
    signals  = []

    for pid in range(n_patients):
        df, signal = generate_patient(pid, regime_fn, rng, n_cells, regime_kwargs)
        cell_dfs.append(df)
        signals.append(signal)

    signals = np.array(signals)
    times, events = simulate_survival(
        signals,
        beta=beta,
        noise=noise,
        censoring_rate=censoring_rate,
        rng=rng,
    )

    # Build patient-level metadata.
    # Clinical features are intentionally omitted — they are noise in the
    # synthetic setting and would confound validation of spatial signal recovery.
    # The pipeline handles missing clinical columns by imputing 0; always run
    # synthetic experiments without --clinical.
    patient_meta = pd.DataFrame({
        "PID":          range(n_patients),
        "OSmonth":      times,
        "DFSmonth":     times,           # same for synthetic (no PFS distinction)
        "signal":       signals,         # ground-truth latent variable
        "Patientstatus": ["death" if e else "alive" for e in events],
        "cohort":        "synthetic",
        "FileName_FullStack": [f"synthetic_{pid}" for pid in range(n_patients)],
        "metacluster":   np.nan,
    })

    # Merge cell data with patient metadata
    all_cells = pd.concat(cell_dfs, ignore_index=True)
    all_cells = all_cells.merge(patient_meta, on="PID", how="left")

    event_rate = events.mean()
    print(f"  Regime {regime} ({REGIME_NAMES[regime]})  noise={noise:.2f}  "
          f"n={n_patients}  events={event_rate:.1%}  "
          f"signal range=[{signals.min():.2f}, {signals.max():.2f}]")

    return all_cells


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic point pattern dataset")
    parser.add_argument(
        "--regime", nargs="+", default=["all"],
        help="Which regime(s) to generate: 1-4 or all (default: all)",
    )
    parser.add_argument("--n_patients", type=int, default=300,
                        help="Number of patients per dataset (default: 300)")
    parser.add_argument("--n_cells", type=int, default=300,
                        help="Approximate cells per core (default: 300)")
    parser.add_argument(
        "--noise", nargs="+", type=float, default=[0.0],
        help="Noise levels tau to generate (default: 0.0, as in the thesis)",
    )
    parser.add_argument("--beta", type=float, default=5.0,
                        help="Effect size beta (default: 5.0, as in the thesis)")
    parser.add_argument("--censoring_rate", type=float, default=0.4,
                        help="Target censoring rate (default: 0.4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output_dir", default="datasets",
                        help="Output directory (default: datasets/)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    regimes = [1, 2, 3, 4] if "all" in args.regime else [int(r) for r in args.regime]
    noises  = args.noise

    for regime in regimes:
        for noise in noises:
            print(f"\nGenerating regime={regime} noise={noise}...")
            df = generate_dataset(
                regime=regime,
                n_patients=args.n_patients,
                n_cells=args.n_cells,
                beta=args.beta,
                noise=noise,
                censoring_rate=args.censoring_rate,
                seed=args.seed + regime,  # different seed per regime, same per noise
            )

            fname = f"synthetic_regime{regime}_noise{noise:.1f}.parquet"
            out   = os.path.join(args.output_dir, fname)
            df.to_parquet(out, index=False)
            print(f"  Saved {len(df):,} cells → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
