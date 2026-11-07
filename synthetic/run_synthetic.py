"""
Synthetic validation (thesis Table "Synthetic validation C-index").

For each regime × noise level, and each seed (one random 80/20 patient split
per seed):
  1. Generates the dataset if it does not already exist (beta = 5).
  2. Fits Cox PH models on NND, Ripley's K, and NND + Ripley's K features,
     and scores the oracle (ranking patients by the true signal s).
  3. Trains the GNN (no pretraining, no clinical features); with --gnn_only
     the bottleneck is the GCN embedding alone.
  4. Logs regime, noise, model and val C-index to MLflow
     (experiment "synthetic-validation").

A summary table (mean ± SD over seeds) is printed at the end.

Usage (the thesis table is the union of these two runs):
    uv run python synthetic/run_synthetic.py              # NND, Ripley, NND+Ripley, GNN full, oracle
    uv run python synthetic/run_synthetic.py --gnn_only   # GNN only column
    uv run python synthetic/run_synthetic.py --no_train   # Cox PH baselines only
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nnd_analysis import compute_nnd_metrics
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from data_utils import prepare_survival_data, RIPLEY_RADII_UM
from scipy.spatial.distance import cdist

MLFLOW_EXPERIMENT = "synthetic-validation"

NND_FEATURES = [
    "median_nnd", "mean_nnd", "frac_adjacent", "frac_communicating", "n_immune",
]

RIPLEY_FEATURES = [f"ripley_k_{int(r)}" for r in RIPLEY_RADII_UM]

# ── GNN training hyperparameters (match real-data best config) ─────────────────
GNN_ARGS = [
    "--hidden_dim",  "32",
    "--lr",          "0.0005",
    "--n_layers",    "3",
    "--k",           "12",
    "--batch_size",  "64",
    "--epochs",      "100",
    "--patience",    "15",
    # no --clinical: clinical columns are absent in synthetic data
]


# ── Ripley's K helper ─────────────────────────────────────────────────────────

def _ripley_k(coords: np.ndarray, area: float) -> np.ndarray:
    """Compute Ripley's K at RIPLEY_RADII_UM for a set of 2-D points (µm)."""
    n = len(coords)
    if n < 2:
        return np.zeros(len(RIPLEY_RADII_UM))
    D = cdist(coords, coords)
    intensity = n / area
    return np.array(
        [(np.sum(D < r) - n) / (intensity * n) for r in RIPLEY_RADII_UM]
    )


# ── Patient split ──────────────────────────────────────────────────────────────

def split_pids(patient_df: pd.DataFrame, seed: int, val_size: float = 0.2):
    """Simple random 80/20 patient split (no subtype stratification needed)."""
    pids = sorted(patient_df["PID"].unique())
    train_pids, val_pids = train_test_split(pids, test_size=val_size,
                                            random_state=seed, shuffle=True)
    train = patient_df[patient_df["PID"].isin(train_pids)]
    val   = patient_df[patient_df["PID"].isin(val_pids)]
    return train, val, val_pids


# ── NND Cox PH baseline ────────────────────────────────────────────────────────

def compute_nnd_patient_df(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Build patient-level NND + Ripley's K feature table from cell-level data."""
    df = prepare_survival_data(df, endpoint=endpoint)
    df = df.dropna(subset=["time", "event"])

    nnd_rows = []
    for core in df["core"].unique():
        subset = df[df["core"] == core].copy()
        metrics = compute_nnd_metrics(subset)
        if metrics is None:
            continue

        row = {
            "core":               metrics.core,
            "median_nnd":         metrics.median_nnd,
            "mean_nnd":           metrics.mean_nnd,
            "frac_adjacent":      metrics.frac_adjacent,
            "frac_communicating": metrics.frac_communicating,
            "n_immune":           metrics.n_immune,
        }

        # Ripley's K for immune cells
        immune_mask = subset["cell_type"] == "immune"
        immune_coords = subset.loc[immune_mask, ["x", "y"]].values.astype(float)
        x_range = subset["x"].max() - subset["x"].min()
        y_range = subset["y"].max() - subset["y"].min()
        area = float(max(x_range * y_range, 1.0))
        k_vals = _ripley_k(immune_coords, area)
        for fname, kv in zip(RIPLEY_FEATURES, k_vals):
            row[fname] = kv

        nnd_rows.append(row)

    nnd_df = pd.DataFrame(nnd_rows)
    meta = df.groupby("core")[["PID", "time", "event"]].first().reset_index()
    patient_df = meta.merge(nnd_df, on="core", how="left")

    all_feats = NND_FEATURES + RIPLEY_FEATURES
    agg = {f: "mean" for f in all_feats}
    agg.update({"time": "first", "event": "first"})
    patient_df = patient_df.groupby("PID").agg(agg).reset_index()
    return patient_df


def _run_cox(tr: pd.DataFrame, vl: pd.DataFrame,
             feature_pool: list[str], label: str) -> float | None:
    """Fit a Cox PH model on tr and evaluate on vl. Returns val C-index."""
    active = [f for f in feature_pool if tr[f].var() > 1e-6]
    if not active:
        print(f"    {label} Cox PH skipped: no features with variance")
        return None
    try:
        cph = CoxPHFitter(penalizer=1.0)
        cph.fit(tr[["time", "event"] + active],
                duration_col="time", event_col="event")
        pred = cph.predict_partial_hazard(vl[active])
        return concordance_index(vl["time"], -pred.values.ravel(), vl["event"])
    except Exception as e:
        print(f"    {label} Cox PH failed: {e}")
        return None


def run_baselines(data_path: str, endpoint: str, seed: int) -> dict:
    """
    Fit NND-only, Ripley-only and NND + Ripley Cox PH, and score the oracle.
    Returns {"nnd_cox", "ripley_cox", "nnd_ripley_cox", "oracle"} -> val C-index
    (None if a model could not be fitted).

    The oracle ranks val patients by the ground-truth signal itself (higher
    signal = lower hazard), giving the C-index ceiling for that split.
    """
    df = pd.read_parquet(data_path)
    patient_df = compute_nnd_patient_df(df, endpoint)
    signal = df.groupby("PID")["signal"].first()

    train, val, _ = split_pids(patient_df, seed=seed)
    oracle_ci = concordance_index(val["time"], val["PID"].map(signal), val["event"])

    all_feats = NND_FEATURES + RIPLEY_FEATURES
    cols = ["time", "event"] + all_feats
    tr = train[cols].dropna()
    vl = val[cols].dropna()

    out = {"nnd_cox": None, "ripley_cox": None, "nnd_ripley_cox": None,
           "oracle": oracle_ci}
    if len(tr) < 10 or vl["event"].sum() < 2:
        return out

    out["nnd_cox"]        = _run_cox(tr, vl, NND_FEATURES,    "NND")
    out["ripley_cox"]     = _run_cox(tr, vl, RIPLEY_FEATURES, "Ripley")
    out["nnd_ripley_cox"] = _run_cox(tr, vl, all_feats,       "NND + Ripley")
    return out


# ── GNN training ───────────────────────────────────────────────────────────────

def run_gnn(data_path: str, endpoint: str, seed: int, val_pids: list,
            gnn_only: bool = False) -> bool:
    """Train GNN for one dataset/seed, writing val PIDs explicitly.

    gnn_only drops NND + Ripley's K from the bottleneck (--no_spatial_features),
    so the Cox head sees the GCN embedding alone.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(str(p) for p in val_pids))
        pids_file = f.name
    # Throwaway checkpoint path, so sweep runs never overwrite best_model.pt
    model_out = pids_file.replace(".txt", ".pt")

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "..", "train.py"),
        *GNN_ARGS,
        "--data",          data_path,
        "--endpoint",      endpoint,
        "--seed",          str(seed),
        "--val_pids_file", pids_file,
        "--model_out",     model_out,
        *(["--no_spatial_features"] if gnn_only else []),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.join(os.path.dirname(__file__), ".."))
    os.unlink(pids_file)
    if os.path.exists(model_out):
        os.unlink(model_out)

    if result.returncode != 0:
        print(f"    GNN training failed:\n{result.stderr[-400:]}")
        return False
    return True


def fetch_gnn_cindex(endpoint: str, seed: int, data_path: str,
                     gnn_only: bool = False) -> float | None:
    """Fetch best_val_cindex from the most recent matching MLflow run."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("survival-gnn")
    if exp is None:
        return None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f"params.endpoint = '{endpoint}' and "
            f"params.seed = '{seed}' and "
            f"params.data = '{data_path}' and "
            f"params.val_pids_file != 'None' and "
            f"params.no_spatial_features = '{gnn_only}'"
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None

    ci = runs[0].data.metrics.get("best_val_cindex")
    if ci is None:
        history = client.get_metric_history(runs[0].info.run_id, "val_cindex")
        if history:
            ci = max(m.value for m in history)
    return ci


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic validation sweep")
    parser.add_argument("--regimes", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--noise",   nargs="+", type=float, default=[0.0],
                        help="Noise levels tau (the thesis reports tau = 0)")
    parser.add_argument("--seeds",   nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="One random 80/20 split (and GNN init) per seed")
    parser.add_argument("--endpoint", default="DFS", choices=["DFS", "OS"])
    parser.add_argument("--n_patients", type=int, default=300)
    parser.add_argument("--beta", type=float, default=5.0,
                        help="Effect size β for dataset generation (default: 5.0)")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip GNN training (Cox PH only)")
    parser.add_argument("--gnn_only", action="store_true",
                        help="Train the GNN without NND/Ripley's K at the "
                             "bottleneck (GCN embedding only)")
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    records = []   # for summary table

    for regime in args.regimes:
        for noise in args.noise:
            data_path = f"datasets/synthetic_regime{regime}_noise{noise:.1f}.parquet"

            # Generate dataset if missing
            if not os.path.exists(data_path):
                print(f"\nGenerating {data_path}...")
                subprocess.run([
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "generate.py"),
                    "--regime", str(regime),
                    "--noise",  str(noise),
                    "--n_patients", str(args.n_patients),
                    "--beta",   str(args.beta),
                ], cwd=os.path.join(os.path.dirname(__file__), ".."))

            print(f"\n── Regime {regime}  noise={noise:.1f} ─────────────────────────────")

            BASELINES = [("nnd_cox", "NND Cox PH"), ("ripley_cox", "Ripley Cox"),
                         ("nnd_ripley_cox", "NND+Ripley"), ("oracle", "Oracle")]
            cis = {name: [] for name, _ in BASELINES}
            gnn_cis = []
            gnn_label = "gnn_only" if args.gnn_only else "gnn"

            for seed in args.seeds:
                # ── Cox PH baselines and oracle ───────────────────────────────
                baseline_cis = run_baselines(data_path, args.endpoint, seed)
                for name, label in BASELINES:
                    ci = baseline_cis[name]
                    if ci is None:
                        continue
                    cis[name].append(ci)
                    with mlflow.start_run():
                        mlflow.log_params({
                            "regime":    regime,
                            "noise":     noise,
                            "seed":      seed,
                            "model":     name,
                            "endpoint":  args.endpoint,
                            "data_path": data_path,
                        })
                        mlflow.log_metric("val_cindex", ci)
                    print(f"  {label:<11} seed={seed}  val_cindex={ci:.3f}")

                # ── GNN ───────────────────────────────────────────────────────
                if not args.no_train:
                    # Need val PIDs to match Cox PH split
                    df = pd.read_parquet(data_path)
                    patient_df = prepare_survival_data(df, endpoint=args.endpoint)
                    patient_df = patient_df.dropna(subset=["time", "event"])
                    pids_df = patient_df.groupby("PID").first().reset_index()
                    _, _, val_pids = split_pids(pids_df, seed=seed)

                    success = run_gnn(data_path, args.endpoint, seed, val_pids,
                                      gnn_only=args.gnn_only)
                    if success:
                        gnn_ci = fetch_gnn_cindex(args.endpoint, seed, data_path,
                                                  gnn_only=args.gnn_only)
                        if gnn_ci is not None:
                            gnn_cis.append(gnn_ci)
                            with mlflow.start_run():
                                mlflow.log_params({
                                    "regime":    regime,
                                    "noise":     noise,
                                    "seed":      seed,
                                    "model":     gnn_label,
                                    "endpoint":  args.endpoint,
                                    "data_path": data_path,
                                })
                                mlflow.log_metric("val_cindex", gnn_ci)
                            print(f"  {'GNN only' if args.gnn_only else 'GNN':<11} "
                                  f"seed={seed}  val_cindex={gnn_ci:.3f}")

            # Summary for this regime/noise
            def _ms(vals):
                return (np.mean(vals), np.std(vals)) if vals else (None, None)
            record = {"regime": regime, "noise": noise}
            for name, _ in BASELINES:
                record[name] = _ms(cis[name])
            record["gnn"] = _ms(gnn_cis)
            records.append(record)

    # ── Summary table ──────────────────────────────────────────────────────────
    def _fmt(ms):
        mean, sd = ms
        return f"{mean:.3f}±{sd:.3f}" if mean is not None else "    —    "

    gnn_col = "GNN only" if args.gnn_only else "GNN full"
    cols = [("nnd_cox", "NND"), ("ripley_cox", "Ripley"),
            ("nnd_ripley_cox", "NND+Ripley"), ("gnn", gnn_col), ("oracle", "Oracle")]
    W = 16 + 14 * len(cols)
    print("\n" + "=" * W)
    print(f"{'Regime':<8} {'Noise':<7}" + "".join(f"{label:>14}" for _, label in cols))
    print("=" * W)
    for r in records:
        print(f"  {r['regime']:<6} {r['noise']:<7.1f}"
              + "".join(f"{_fmt(r[name]):>14}" for name, _ in cols))
    print("=" * W)
    print("\nDone. View results: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
