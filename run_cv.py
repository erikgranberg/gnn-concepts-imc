"""
Patient-level 5-fold cross-validation of all models → Figure 7.

Uses the patient-level folds saved in cv_folds.json (the folds used in the
thesis, stratified on subtype x grade; --new_folds generates fresh folds
stratified on subtype x grade x event status), fits the Cox PH baselines on each fold and runs train.py for each fold x GNN
condition, collects the GNN results from MLflow, and saves Figure 7 plus a
CSV with the mean +- SD C-index per model (results/cv_<endpoint>.csv).

GNN conditions:
    no_pretrain        full bottleneck (GCN + NND + Ripley + clinical), from scratch
    pretrain_finetune  same, encoder initialised from masked pretraining (--encoder_lr)
    pretrain_frozen    same, pretrained encoder frozen
    gnn_only*          the three above with the bottleneck reduced to the GCN embedding

The same folds are used for the GNN and the Cox PH baselines, so the
comparison is on the same patients in each fold.

Usage:
    uv run python run_cv.py --endpoint DFS --encoder_lr 1e-4
    uv run python run_cv.py --endpoint DFS --encoder_lr 1e-4 --plot_only   # re-fetch & replot
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import mlflow

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "output")
FOLDS_FILE = "cv_folds.json"   # saved so folds can be inspected / reused

# ── Shared GNN hyperparameters ─────────────────────────────────────────────────
# Defaults for the architecture / optimisation flags below (overridable on the
# command line, e.g. with the winners of sweep.py).
DATA_PATH = "datasets/point_patterns_metaclusters.parquet"

ALL_CONDITIONS = ["no_pretrain", "pretrain_finetune", "pretrain_frozen",
                  "gnn_only", "gnn_only_finetune", "gnn_only_frozen"]
# Conditions that load the pretrained encoder
PRETRAINED_CONDITIONS = {"pretrain_finetune", "pretrain_frozen",
                         "gnn_only_finetune", "gnn_only_frozen"}
PRETRAINED_PATH = "pretrained_metacluster_nocap.pt"


# ── Fold generation ────────────────────────────────────────────────────────────

def generate_folds(endpoint: str, n_folds: int, seed: int) -> list[list[str]]:
    """
    Generate patient-level stratified k-folds.

    Stratifies on subtype × grade × event status (data_utils.make_cv_folds)
    so each fold is representative and holds a similar number of events.
    Returns a list of val-PID lists, one per fold.
    """
    from data_utils import make_cv_folds
    from clinical_baseline import load_patient_data

    print("  Loading patient data to generate folds...")
    df = pd.read_parquet("datasets/point_patterns_metaclusters.parquet")
    patient_df = load_patient_data(data_path=None, endpoint=endpoint, df=df)
    patient_df = patient_df.dropna(subset=["grade", "tumor_size", "age"])

    folds = make_cv_folds(patient_df, n_folds, seed)

    print(f"  Generated {n_folds} folds: "
          + ", ".join(f"fold {i+1}: {len(f)} val patients" for i, f in enumerate(folds)))
    return folds


# ── Clinical baseline per fold ─────────────────────────────────────────────────

# Cox PH baselines: key -> (label, feature groups)
COX_MODELS = {
    "nnd":                 ("NND",                     ["nnd"]),
    "ripley":              ("Ripley",                  ["ripley"]),
    "nnd_ripley":          ("NND + Ripley",            ["nnd", "ripley"]),
    "clinical":            ("Clinical",                ["clinical"]),
    "clinical_nnd":        ("Clinical + NND",          ["clinical", "nnd"]),
    "clinical_ripley":     ("Clinical + Ripley",       ["clinical", "ripley"]),
    "clinical_nnd_ripley": ("Clinical + NND + Ripley", ["clinical", "nnd", "ripley"]),
}


def clinical_cv(folds: list[list[str]], endpoint: str) -> dict:
    """
    Fit each Cox PH baseline on the training folds and evaluate on the held-out
    fold. Returns {model key: (mean C-index, SD)} for the keys in COX_MODELS.
    """
    from clinical_baseline import (
        load_patient_data, fit_cox_val,
        CLINICAL_FEATURES, NND_FEATURES, RIPLEY_FEATURES,
    )
    groups = {"clinical": CLINICAL_FEATURES, "nnd": NND_FEATURES,
              "ripley": RIPLEY_FEATURES}

    print("  Computing clinical Cox PH baseline across folds...")
    df = pd.read_parquet("datasets/point_patterns_metaclusters.parquet")
    patient_df = load_patient_data(data_path=None, endpoint=endpoint, df=df)
    patient_df = patient_df.dropna(subset=["grade", "tumor_size", "age"])

    cis = {key: [] for key in COX_MODELS}
    for i, val_pids in enumerate(folds):
        val_pids_set = set(val_pids)
        train = patient_df[~patient_df["PID"].isin(val_pids_set)]
        val   = patient_df[ patient_df["PID"].isin(val_pids_set)]
        for key, (_, feature_groups) in COX_MODELS.items():
            features = [f for g in feature_groups for f in groups[g]]
            ci = fit_cox_val(train, val, features)
            if ci is not None:
                cis[key].append(ci)
        print(f"    Fold {i+1}: " + "  ".join(
            f"{COX_MODELS[k][0].lower()}={cis[k][-1]:.3f}"
            for k in ("clinical", "nnd", "nnd_ripley", "clinical_nnd_ripley")))

    return {key: (float(np.mean(v)), float(np.std(v))) for key, v in cis.items()}


# ── GNN training ───────────────────────────────────────────────────────────────

def run_fold(condition_extra_args: list, fold_idx: int, val_pids: list[str],
             endpoint: str, base_args: list) -> bool:
    """Write val PIDs to a temp file and run train.py for one fold."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix=f"fold{fold_idx}_", delete=False
    ) as f:
        f.write("\n".join(str(p) for p in val_pids))
        pids_file = f.name
    # Fold checkpoints are throwaway (results are read back from MLflow); keep
    # them out of best_model.pt, which train.py would otherwise overwrite.
    fd, model_out = tempfile.mkstemp(suffix=".pt", prefix=f"fold{fold_idx}_model_")
    os.close(fd)

    cmd = [
        sys.executable, "train.py",
        *base_args,
        "--endpoint", endpoint,
        "--seed", str(fold_idx),       # controls weight init only (split is explicit)
        "--val_pids_file", pids_file,
        "--model_out", model_out,
        *condition_extra_args,
    ]
    print(f"    Running fold {fold_idx} — extra={condition_extra_args or '(none)'}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(pids_file)
    os.unlink(model_out)

    if result.returncode != 0:
        print(f"    ERROR:\n{result.stderr[-500:]}")
        return False
    return True


# ── MLflow result collection ───────────────────────────────────────────────────

def fetch_gnn_results(condition_name: str, n_folds: int,
                      endpoint: str,
                      finetune_mode: str | None = None,
                      no_spatial_features: bool | None = None,
                      clinical: bool | None = None,
                      extra_params: dict | None = None,
                      ) -> tuple[float | None, float | None]:
    """
    Collect best_val_cindex from the most recent n_folds runs matching
    this condition, ordered by start time descending.

    finetune_mode: if provided, filters on params.finetune_mode (e.g. 'finetune',
        'finetune_same_lr', 'frozen', 'scratch') to distinguish proper fine-tune
        runs from old single-lr runs.
    no_spatial_features: if provided, filters on params.no_spatial_features, so
        a bottleneck-ablation sweep doesn't pick up runs from a normal sweep
        (or vice versa) when both share endpoint/finetune_mode.
    clinical: if provided, filters on params.clinical, for the same reason.
    extra_params: exact-match filters on further logged params (e.g.
        batch_size, max_cells), so runs with other hyperparameters — such as
        other batch sizes — are never mixed in.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("survival-gnn")
    if experiment is None:
        raise RuntimeError("MLflow experiment 'survival-gnn' not found.")
    exp_id = experiment.experiment_id

    pretrained_used = condition_name in PRETRAINED_CONDITIONS
    freeze = "frozen" in condition_name

    filter_parts = [
        f"params.endpoint = '{endpoint}'",
        f"params.pretrained_used = '{pretrained_used}'",
    ]
    if pretrained_used:
        filter_parts.append(f"params.freeze_encoder = '{freeze}'")
    # Only runs that used a val_pids_file (i.e. CV runs, not ad-hoc runs)
    filter_parts.append("params.val_pids_file != 'None'")
    if finetune_mode is not None:
        filter_parts.append(f"params.finetune_mode = '{finetune_mode}'")
    if no_spatial_features is not None:
        filter_parts.append(f"params.no_spatial_features = '{no_spatial_features}'")
    if clinical is not None:
        filter_parts.append(f"params.clinical = '{clinical}'")
    for key, value in (extra_params or {}).items():
        filter_parts.append(f"params.{key} = '{value}'")

    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=" and ".join(filter_parts),
        order_by=["start_time DESC"],
        max_results=n_folds,
    )

    if not runs:
        print(f"  WARNING: no MLflow runs found for condition={condition_name}")
        return None, None

    cindices = []
    for run in runs:
        ci = run.data.metrics.get("best_val_cindex")
        if ci is None:
            # Fall back to max per-epoch val_cindex
            history = client.get_metric_history(run.info.run_id, "val_cindex")
            if history:
                ci = max(m.value for m in history)
        if ci is not None:
            cindices.append(ci)

    if not cindices:
        return None, None

    print(f"  {condition_name:<22}  folds={[f'{c:.3f}' for c in cindices]}"
          f"  mean={np.mean(cindices):.3f}  sd={np.std(cindices):.3f}")
    return float(np.mean(cindices)), float(np.std(cindices))


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="k-fold CV ablation → Figure 7")
    parser.add_argument("--endpoint", default="DFS", choices=["DFS", "OS"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for fold generation (not model training)")
    parser.add_argument("--new_folds", action="store_true",
                        help="Generate new folds (stratified on subtype x grade x event, "
                             "--seed) instead of reusing --folds_file. By default the "
                             "saved folds are reused, so results match the thesis.")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip training; re-fetch MLflow results and regenerate figure")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (head lr when --encoder_lr is set)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max_cells", type=int, default=None,
                        help="Max cells per core (default: no cap)")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--pretrained", default=PRETRAINED_PATH,
                        help="Pretrained encoder for the pretrain_* conditions; its "
                             "architecture must match --hidden_dim/--n_layers.")
    parser.add_argument("--encoder_lr", type=float, default=None,
                        help="Separate learning rate for the pretrained encoder in the "
                             "*_finetune conditions (the thesis uses 1e-4). If omitted, "
                             "encoder and head share --lr.")
    parser.add_argument("--no_spatial_features", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Exclude NND+Ripley features from the bottleneck of the "
                             "non-gnn_only conditions (passed to train.py). Omitted: "
                             "don't pass the flag and don't filter the MLflow fetch by "
                             "it. Explicit --no_spatial_features / --no-no_spatial_features "
                             "also filters the fetch to that value.")
    parser.add_argument("--clinical", action=argparse.BooleanOptionalAction, default=True,
                        help="Include clinical features in the bottleneck of the "
                             "non-gnn_only conditions (default: yes). Also filters the "
                             "MLflow fetch.")
    parser.add_argument("--conditions", nargs="+",
                        default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help="Restrict which GNN conditions to run/fetch (default: all six). "
                             "gnn_only* are no_pretrain / pretrain_finetune / pretrain_frozen "
                             "with the bottleneck reduced to the GCN embedding alone (no "
                             "clinical, NND or Ripley features).")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Where to save Figure 7 (default: figures/output).")
    parser.add_argument("--folds_file", default=FOLDS_FILE,
                        help="Where to save/load the fold assignments "
                             "(default: cv_folds.json).")
    parser.add_argument("--results_csv", default=None,
                        help="Where to write mean/SD C-index per model "
                             "(default: results/cv_<endpoint>.csv).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build run-time BASE_ARGS and CONDITIONS ───────────────────────────────
    base_args = [
        "--hidden_dim", str(args.hidden_dim),
        "--lr",         str(args.lr),
        "--n_layers",   str(args.n_layers),
        "--k",          str(args.k),
        "--batch_size", str(args.batch_size),
        "--epochs",     str(args.epochs),
        "--patience",   str(args.patience),
        "--data",       args.data,
    ]
    if args.max_cells is not None:
        base_args += ["--max_cells", str(args.max_cells)]
    # Bottleneck flags are per condition: gnn_only always drops the clinical
    # and NND/Ripley features, the other conditions follow the CLI flags.
    bottleneck_args = []
    if args.no_spatial_features:
        bottleneck_args += ["--no_spatial_features"]
    if args.clinical:
        bottleneck_args += ["--clinical"]

    finetune_extra = ["--pretrained", args.pretrained]
    if args.encoder_lr is not None:
        finetune_extra += ["--encoder_lr", str(args.encoder_lr)]
    frozen_extra = ["--pretrained", args.pretrained, "--freeze_encoder"]

    all_conditions = {
        "no_pretrain":       bottleneck_args,
        "pretrain_finetune": bottleneck_args + finetune_extra,
        "pretrain_frozen":   bottleneck_args + frozen_extra,
        "gnn_only":          ["--no_spatial_features"],
        "gnn_only_finetune": ["--no_spatial_features"] + finetune_extra,
        "gnn_only_frozen":   ["--no_spatial_features"] + frozen_extra,
    }
    conditions = {c: all_conditions[c] for c in args.conditions}
    if not args.plot_only and any(c in PRETRAINED_CONDITIONS for c in conditions) \
            and not os.path.exists(args.pretrained):
        sys.exit(f"Pretrained encoder not found: {args.pretrained} "
                 f"(pass --pretrained or use --conditions no_pretrain)")
    if not args.plot_only and any(c in PRETRAINED_CONDITIONS for c in conditions):
        # A mismatch makes every pretrain fold crash inside train.py, which
        # run_fold only reports — the fetch would then silently pick up old runs.
        import torch
        cfg = torch.load(args.pretrained, weights_only=False, map_location="cpu")["config"]
        wanted = {"hidden_dim": args.hidden_dim, "n_layers": args.n_layers}
        mismatch = {k: (cfg[k], v) for k, v in wanted.items() if cfg[k] != v}
        if mismatch:
            sys.exit(f"{args.pretrained} architecture does not match the requested "
                     f"model (checkpoint, requested): {mismatch}")

    # Map each condition to the finetune_mode string logged by train.py
    # so we can filter MLflow to only pick up runs from this invocation style.
    finetune_mode_map = {
        "no_pretrain":       "scratch",
        "pretrain_finetune": "finetune" if args.encoder_lr is not None else "finetune_same_lr",
        "pretrain_frozen":   "frozen",
        "gnn_only":          "scratch",
        "gnn_only_finetune": "finetune" if args.encoder_lr is not None else "finetune_same_lr",
        "gnn_only_frozen":   "frozen",
    }
    # (no_spatial_features, clinical) filters per condition. Global flags:
    # no_spatial_features None/True/False passes straight through; clinical is
    # always explicit — pre-existing runs all logged clinical='True'.
    bottleneck_filter = {c: (args.no_spatial_features, args.clinical) for c in all_conditions}
    for c in ("gnn_only", "gnn_only_finetune", "gnn_only_frozen"):
        bottleneck_filter[c] = (True, False)

    # ── Folds ─────────────────────────────────────────────────────────────────
    # Reuse the saved folds (cv_folds.json holds the folds used in the thesis)
    # unless --new_folds is given or no matching file exists.
    reuse = False
    if not args.new_folds and os.path.exists(args.folds_file):
        with open(args.folds_file) as f:
            saved = json.load(f)
        reuse = (saved.get("endpoint") == args.endpoint
                 and len(saved.get("folds", [])) == args.folds)
        if not reuse and args.plot_only:
            sys.exit(f"{args.folds_file} does not match --endpoint/--folds")
    elif args.plot_only:
        sys.exit(f"--plot_only needs an existing {args.folds_file}")

    if not reuse:
        print(f"\n── Generating {args.folds} folds ({args.endpoint}) ──────────────────────")
        folds = generate_folds(args.endpoint, args.folds, args.seed)
        from data_utils import CV_STRATIFY_SCHEME
        with open(args.folds_file, "w") as f:
            json.dump({"endpoint": args.endpoint, "stratify": CV_STRATIFY_SCHEME,
                       "folds": folds}, f, indent=2)
        print(f"  Folds saved to {args.folds_file}")
    else:
        print(f"\n── Loading folds from {args.folds_file} ────────────────────────────────")
        folds = saved["folds"]
        print(f"  Loaded {len(folds)} folds")

    # ── Clinical baseline ─────────────────────────────────────────────────────
    print(f"\n── Clinical Cox PH baseline ────────────────────────────────────────")
    cox = clinical_cv(folds, args.endpoint)
    for key, (label, _) in COX_MODELS.items():
        mean, sd = cox[key]
        print(f"  {label:<24} mean={mean:.3f}    sd={sd:.3f}")

    # ── GNN training ──────────────────────────────────────────────────────────
    if not args.plot_only:
        total = len(conditions) * len(folds)
        done = 0
        for cname, extra in conditions.items():
            print(f"\n── Condition: {cname} ──────────────────────────────────────────────")
            for fold_idx, val_pids in enumerate(folds, start=1):
                run_fold(extra, fold_idx, val_pids, args.endpoint, base_args)
                done += 1
                print(f"  [{done}/{total}] complete")

    # ── Collect GNN results ───────────────────────────────────────────────────
    print(f"\n── Collecting GNN results from MLflow ──────────────────────────────")
    gnn = {}
    hparam_filter = {
        "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
        "k": args.k, "lr": args.lr, "batch_size": args.batch_size,
        "max_cells": args.max_cells, "data": args.data,
    }
    for cname in conditions:
        extra = dict(hparam_filter)
        if cname in PRETRAINED_CONDITIONS:
            extra["pretrained"] = args.pretrained
        no_spatial_filter, clinical_filter = bottleneck_filter[cname]
        m, s = fetch_gnn_results(
            cname, len(folds), args.endpoint,
            finetune_mode=finetune_mode_map[cname],
            no_spatial_features=no_spatial_filter,
            clinical=clinical_filter,
            extra_params=extra,
        )
        gnn[cname] = (m, s)

    # ── Results table ─────────────────────────────────────────────────────────
    rows = ([(f"{label} (Cox PH)", *cox[key]) for key, (label, _) in COX_MODELS.items()]
            + [(f"GNN: {c}", *gnn[c]) for c in conditions])
    results_csv = args.results_csv or os.path.join("results", f"cv_{args.endpoint}.csv")
    os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=["model", "cindex_mean", "cindex_sd"]).to_csv(
        results_csv, index=False, float_format="%.4f")
    print(f"  Results table saved to {results_csv}")

    # ── Figure 7 ─────────────────────────────────────────────────────────────
    print(f"\n── Generating Figure 7 ─────────────────────────────────────────────")
    from figures.baseline_figs import plot_model_comparison
    plot_model_comparison(args.output_dir, cox=cox, gnn=gnn, endpoint=args.endpoint)
    print("Done.")


if __name__ == "__main__":
    main()
