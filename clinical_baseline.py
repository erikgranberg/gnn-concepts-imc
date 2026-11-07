"""
Cox PH baselines on clinical, NND and Ripley's K features.

load_patient_data() builds the patient-level feature table, split_patients()
gives the single stratified 80/20 split (same patients as the GNN), and
fit_cox_val() fits a Cox PH model on one set of patients and scores another.
The 5-fold CV of the baselines is in run_cv.py.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.spatial.distance import cdist

from data_utils import prepare_survival_data, make_stratify_labels, RIPLEY_RADII_UM
from nnd_analysis import compute_nnd_metrics


CLINICAL_FEATURES = ["grade", "tumor_size", "age", "ER_pos", "PR_pos", "HER2_pos"]
NND_FEATURES = [
    "median_nnd",
    "mean_nnd",
    "frac_adjacent",
    "frac_communicating",
    "n_immune",
]
RIPLEY_FEATURES = [f"ripley_k_{int(r)}" for r in RIPLEY_RADII_UM]

# True immune cell types in the metacluster representation.
# Stromal and endothelial cells are excluded — they are structural,
# not part of the immune response, and including them makes NND
# features near-constant and biologically uninterpretable.
IMMUNE_METACLUSTERS = {"T_cell", "B_cell", "macrophage"}


def load_patient_data(data_path: str, endpoint: str, df: pd.DataFrame = None) -> pd.DataFrame:
    """Load parquet (or use supplied df), compute NND metrics, deduplicate to patient level."""
    if df is None:
        df = pd.read_parquet(data_path)
    df = prepare_survival_data(df, endpoint=endpoint)
    df = df.dropna(subset=["time", "event"])

    # Compute NND metrics per core.
    # For metacluster data, map only true immune cells (T cell, B cell,
    # macrophage) to "immune" and drop structural cells (stromal, endothelial).
    # This keeps the biological meaning of Maisel et al.: proximity of tumour
    # cells to immune cells specifically.
    unique_types = set(df["cell_type"].unique())
    nnd_rows = []
    for core in df["core"].unique():
        subset = df[df["core"] == core]
        if "immune" not in unique_types:
            subset = subset.copy()
            subset["cell_type"] = subset["cell_type"].apply(
                lambda x: "tumor" if x == "tumor"
                          else "immune" if x in IMMUNE_METACLUSTERS
                          else None
            )
            subset = subset[subset["cell_type"].notna()]
        metrics = compute_nnd_metrics(subset)
        if metrics is not None:
            row = {
                "core": metrics.core,
                "median_nnd": metrics.median_nnd,
                "mean_nnd": metrics.mean_nnd,
                "frac_adjacent": metrics.frac_adjacent,
                "frac_communicating": metrics.frac_communicating,
                "n_immune": metrics.n_immune,
            }
            # Ripley's K for immune cells at multiple radii
            immune_mask = subset["cell_type"] == "immune"
            immune_coords = subset.loc[immune_mask, ["x", "y"]].values.astype(float)
            n_imm = len(immune_coords)
            if n_imm >= 2:
                x_range = subset["x"].max() - subset["x"].min()
                y_range = subset["y"].max() - subset["y"].min()
                area = float(max(x_range * y_range, 1.0))
                D = cdist(immune_coords, immune_coords)
                intensity = n_imm / area
                for r, fname in zip(RIPLEY_RADII_UM, RIPLEY_FEATURES):
                    row[fname] = (np.sum(D < r) - n_imm) / (intensity * n_imm)
            else:
                for fname in RIPLEY_FEATURES:
                    row[fname] = 0.0
            nnd_rows.append(row)
    nnd_df = pd.DataFrame(nnd_rows)

    # Patient-level metadata (one row per core, then deduplicate by PID)
    meta_cols = [
        "PID",
        "time",
        "event",
        "grade",
        "tumor_size",
        "age",
        "ERStatus",
        "PRStatus",
        "HER2Status",
        "clinical_type",
    ]
    patient_df = df.groupby("core")[meta_cols].first().reset_index()
    patient_df = patient_df.merge(nnd_df, on="core", how="left")

    # Average NND + Ripley features across cores for same patient
    agg = {feat: "mean" for feat in NND_FEATURES + RIPLEY_FEATURES}
    agg.update(
        {
            "time": "first",
            "event": "first",
            "grade": "first",
            "tumor_size": "first",
            "age": "first",
            "ERStatus": "first",
            "PRStatus": "first",
            "HER2Status": "first",
            "clinical_type": "first",
        }
    )
    patient_df = patient_df.groupby("PID").agg(agg).reset_index()

    # Encode binary clinical variables
    patient_df["ER_pos"] = (patient_df["ERStatus"] == "positive").astype(int)
    patient_df["PR_pos"] = (patient_df["PRStatus"] == "positive").astype(int)
    patient_df["HER2_pos"] = (patient_df["HER2Status"] == "positive").astype(int)

    return patient_df


def split_patients(
    patient_df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.2,
):
    """Split patients stratified by clinical subtype + grade.

    Joint stratification on subtype (HR+HER2-, TripleNeg, etc.) and grade
    ensures the val set is representative on both the strongest clinical
    predictor (grade) and the variable most correlated with spatial patterns
    the GNN encodes. Using stable clinical variables (not endpoint-specific
    event status) means OS and PFS experiments use the same val patients,
    making results directly comparable.

    Rare combinations with fewer than 2 patients are collapsed into 'Other'
    to satisfy train_test_split's minimum stratum size requirement.
    """
    pids = sorted(patient_df["PID"].unique())
    subtype_s = patient_df.groupby("PID")["clinical_type"].first().fillna("Unknown")
    grade_s = patient_df.groupby("PID")["grade"].first().apply(
        lambda g: str(int(g)) if pd.notna(g) else "Unknown"
    )
    label_s = make_stratify_labels(subtype_s, grade_s)
    stratify = [label_s[p] for p in pids]

    train_pids, val_pids = train_test_split(
        pids, test_size=val_size, random_state=seed, stratify=stratify
    )

    train = patient_df[patient_df["PID"].isin(train_pids)]
    val = patient_df[patient_df["PID"].isin(val_pids)]
    return train, val


def fit_cox_val(train: pd.DataFrame, val: pd.DataFrame, features: list[str],
                penalizer: float = 0.01) -> float | None:
    """Fit Cox PH on train and return the C-index on val (None if < 10 train rows)."""
    cols = ["time", "event"] + features
    tr = train[cols].dropna()
    vl = val[cols].dropna()
    if len(tr) < 10:
        return None
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(tr, duration_col="time", event_col="event")
    pred = cph.predict_partial_hazard(vl[features])
    return concordance_index(vl["time"], -pred.values.ravel(), vl["event"])
