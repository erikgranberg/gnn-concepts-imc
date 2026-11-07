"""
PyTorch Geometric Dataset for breast cancer point patterns.

Builds a k-NN graph per tissue core from cell coordinates, and computes the
per-core NND + Ripley's K features and per-patient clinical features that are
concatenated at the model bottleneck.

Event definitions (derived at dataset creation time):
    DFS (Disease-Free Survival):
        - Event: death, death by primary disease, alive w metastases, or
          DFSmonth < OSmonth (relapse in a patient now alive without metastases)
        - Time: DFSmonth

    OS (Overall Survival):
        - Event: death, death by primary disease
        - Time: OSmonth
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
try:
    from torch_geometric.nn import knn_graph as _pyg_knn_graph
    _TORCH_CLUSTER_OK = True
except Exception:
    _TORCH_CLUSTER_OK = False

import hashlib
import pandas as pd
import numpy as np
from typing import Optional
from scipy.spatial.distance import cdist

from nnd_analysis import compute_nnd_metrics


def _stable_seed(key: str) -> int:
    """Deterministic 32-bit seed from a string, stable across processes/runs
    (unlike Python's built-in hash(), which is randomised per-process)."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def _knn_graph(x: torch.Tensor, k: int, loop: bool = False) -> torch.Tensor:
    """
    Build a k-NN graph edge index. Uses torch-cluster via PyG when available;
    falls back to a sklearn-based implementation when torch-cluster is not
    installed or incompatible with the current PyTorch version.
    """
    if _TORCH_CLUSTER_OK:
        try:
            return _pyg_knn_graph(x, k=k, loop=loop)
        except Exception:
            pass  # fall through to sklearn fallback

    from sklearn.neighbors import NearestNeighbors
    x_np = x.detach().cpu().numpy()
    n = len(x_np)
    k_eff = min(k + (0 if loop else 1), n)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(x_np)
    _, indices = nbrs.kneighbors(x_np)          # (n, k_eff)
    if not loop:
        indices = indices[:, 1:]                 # drop self (column 0)
    k_actual = indices.shape[1]
    dst = np.repeat(np.arange(n), k_actual)     # target nodes
    src = indices.ravel()                        # source nodes (neighbours)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index


# NND feature names extracted from SpatialMetrics
NND_FEATURE_NAMES = ["median_nnd", "mean_nnd", "frac_adjacent", "frac_communicating", "n_immune"]

# Ripley's K feature names (immune self-clustering at four spatial scales)
RIPLEY_RADII_UM = [20.0, 50.0, 100.0, 200.0]
RIPLEY_FEATURE_NAMES = [f"ripley_k_{int(r)}" for r in RIPLEY_RADII_UM]

# Combined spatial features used at the GNN bottleneck
SPATIAL_FEATURE_NAMES = NND_FEATURE_NAMES + RIPLEY_FEATURE_NAMES


def _compute_ripley_k(coords: np.ndarray, area: float) -> np.ndarray:
    """
    Compute Ripley's K at RIPLEY_RADII_UM for a set of 2-D points (µm).

    Uses simple (uncorrected) estimator:
        K(r) = (A / n²) * Σ_i Σ_{j≠i} 1[d_ij < r]

    Returns array of length len(RIPLEY_RADII_UM). Returns zeros if n < 2.
    """
    n = len(coords)
    if n < 2:
        return np.zeros(len(RIPLEY_RADII_UM), dtype=np.float32)

    D = cdist(coords, coords)
    intensity = n / area
    k_vals = []
    for r in RIPLEY_RADII_UM:
        count = np.sum(D < r) - n   # exclude diagonal (self-distance = 0)
        k_vals.append(count / (intensity * n))
    return np.array(k_vals, dtype=np.float32)

# Clinical feature configuration
CLINICAL_CONTINUOUS = ["age", "tumor_size", "grade"]
CLINICAL_CATEGORICAL = ["ERStatus", "PRStatus", "HER2Status"]

def extract_clinical_features(row: pd.Series) -> dict:
    """
    Extract clinical features from a row of metadata.

    Returns dict with continuous values and categorical indicators.
    Missing values are set to NaN (will be imputed later).
    """
    features = {}

    # Continuous features (will be z-scored)
    for col in CLINICAL_CONTINUOUS:
        if col in row.index and pd.notna(row[col]):
            features[col] = float(row[col])
        else:
            features[col] = np.nan

    # Categorical features (will be one-hot encoded)
    # ERStatus, PRStatus, HER2Status: typically "positive" or "negative"
    for col in CLINICAL_CATEGORICAL:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).lower().strip()
            features[f"{col}_positive"] = 1.0 if val in ["positive", "pos", "1", "+"] else 0.0
        else:
            features[f"{col}_positive"] = np.nan

    return features


def get_clinical_feature_names() -> list[str]:
    """Get ordered list of clinical feature names."""
    names = list(CLINICAL_CONTINUOUS)
    names += [f"{col}_positive" for col in CLINICAL_CATEGORICAL]
    return names

# Batch keys (must match models/consts.py)
BATCH_KEY_NODE_FEATURES = "node_features"
BATCH_KEY_NODE_LABELS = "node_labels"
BATCH_KEY_GRAPH_LABELS = "graph_labels"

# Cell type vocabularies
CELL_TYPES_BINARY = ["tumor", "immune"]  # synthetic data
CELL_TYPES_METACLUSTER = ["tumor", "T_cell", "B_cell", "macrophage", "stromal", "endothelial"]


def prepare_survival_data(df: pd.DataFrame, endpoint: str = "OS") -> pd.DataFrame:
    """
    Prepare survival data by defining events based on endpoint.

    Args:
        df: DataFrame with raw metadata (Patientstatus, DFSmonth, OSmonth)
        endpoint: "DFS" or "OS"

    Returns:
        DataFrame with 'event' and 'time' columns added
    """
    df = df.copy()

    # Normalize patient status for matching
    df["_status_lower"] = df["Patientstatus"].str.lower().str.strip()

    if endpoint == "DFS":
        # Disease-free survival: death OR metastatic progression OR relapse.
        # Patientstatus has no relapse category: patients who relapsed and are
        # now alive without metastases are "alive", but their DFS clock stops
        # before follow-up ends (DFSmonth < OSmonth).
        event_statuses = ["death", "death by primary disease", "alive w metastases"]
        relapsed = df["DFSmonth"] < df["OSmonth"]
        df["event"] = (df["_status_lower"].isin(event_statuses) | relapsed).astype(int)
        df["time"] = df["DFSmonth"]

    elif endpoint == "OS":
        # Overall survival: death only
        event_statuses = ["death", "death by primary disease"]
        df["event"] = df["_status_lower"].isin(event_statuses).astype(int)
        df["time"] = df["OSmonth"]

    else:
        raise ValueError(f"Unknown endpoint: {endpoint}. Use 'DFS' or 'OS'.")

    # Clean up
    df = df.drop(columns=["_status_lower"])

    return df


class PointPatternDataset(Dataset):
    """
    PyTorch Geometric Dataset for spatial point patterns.

    Each sample is a tissue core represented as a graph where:
        - Nodes are cells with features (coordinates, cell type)
        - Edges connect spatially proximate cells
        - Graph-level labels are survival outcomes

    Args:
        df: DataFrame with columns [core, CellId, x, y, cell_type, Patientstatus, DFSmonth, OSmonth, ...]
        endpoint: "DFS" or "OS" for survival endpoint
        k: Number of nearest neighbors for graph construction
        max_cells: Maximum cells per sample (subsampling if exceeded)
        normalize_coords: Normalize coordinates to [0, 1]
        cell_types: List of cell type names for vocabulary. If None, auto-detects.
        require_survival: Drop cores with missing survival data (default). Set
            False for self-supervised pretraining to keep unlabelled cores.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        endpoint: str = "OS",
        k: int = 10,
        max_cells: Optional[int] = None,
        normalize_coords: bool = True,
        cell_types: Optional[list[str]] = None,
        require_survival: bool = True,
    ):
        self.k = k
        self.max_cells = max_cells
        self.normalize_coords = normalize_coords
        self.endpoint = endpoint

        # Derive survival events based on endpoint
        df = prepare_survival_data(df, endpoint=endpoint)

        # Filter out samples with missing survival data. Self-supervised
        # pretraining passes require_survival=False to keep unlabelled cores;
        # their time is then NaN and must not be used as a label.
        if require_survival:
            df = df.dropna(subset=["time", "event"])

        # Auto-detect or validate cell types
        unique_types = df["cell_type"].unique().tolist()
        if cell_types is None:
            # Auto-detect: check if it's binary or metacluster format
            if set(unique_types) <= {"tumor", "immune", "stroma"}:
                self.cell_types = CELL_TYPES_BINARY
            else:
                self.cell_types = CELL_TYPES_METACLUSTER
        else:
            self.cell_types = cell_types

        self.cell_type_to_idx = {ct: i for i, ct in enumerate(self.cell_types)}
        self.n_cell_types = len(self.cell_types)
        self.dim_features = 2 + self.n_cell_types  # x, y + one-hot cell type

        # Get unique cores
        self.cores = df["core"].unique().tolist()

        # Build per-core data and compute NND metrics
        self.data = {}
        nnd_raw = {}
        for core in self.cores:
            subset = df[df["core"] == core]

            # Map cell type strings to indices
            cell_type_indices = subset["cell_type"].map(self.cell_type_to_idx)
            # Handle unknown cell types (map to -1, will be filtered or ignored)
            cell_type_indices = cell_type_indices.fillna(-1).astype(int).values

            self.data[core] = {
                "coords": subset[["x", "y"]].values.astype(np.float32),
                "cell_types": cell_type_indices.astype(np.int64),
                "time": float(subset["time"].iloc[0]),
                "event": float(subset["event"].iloc[0]),
                "PID": subset["PID"].iloc[0] if "PID" in subset.columns else None,
                "clinical": extract_clinical_features(subset.iloc[0]),
                "clinical_type": subset["clinical_type"].iloc[0] if "clinical_type" in subset.columns else "Unknown",
            }

            # Compute NND metrics for this core (uses tumor/immune convention).
            # For metaclusters, map only true immune cells (T cell, B cell,
            # macrophage) to "immune" and drop structural cells (stromal,
            # endothelial). This preserves the biological meaning of Maisel
            # et al.: tumour-to-immune proximity, not tumour-to-any-cell.
            _IMMUNE_METACLUSTERS = {"T_cell", "B_cell", "macrophage"}
            if "immune" not in unique_types:
                subset_nnd = subset.copy()
                subset_nnd["cell_type"] = subset_nnd["cell_type"].apply(
                    lambda x: "tumor" if x == "tumor"
                              else "immune" if x in _IMMUNE_METACLUSTERS
                              else None
                )
                subset_nnd = subset_nnd[subset_nnd["cell_type"].notna()]
                metrics = compute_nnd_metrics(subset_nnd)
            else:
                metrics = compute_nnd_metrics(subset)

            if metrics is not None:
                nnd_raw[core] = np.array([
                    metrics.median_nnd,
                    metrics.mean_nnd,
                    metrics.frac_adjacent,
                    metrics.frac_communicating,
                    metrics.n_immune,
                ], dtype=np.float32)
            else:
                nnd_raw[core] = np.zeros(len(NND_FEATURE_NAMES), dtype=np.float32)

            # Ripley's K for immune cells in this core
            _IMMUNE_FOR_RIPLEY = {"T_cell", "B_cell", "macrophage", "immune"}
            immune_mask = subset["cell_type"].isin(_IMMUNE_FOR_RIPLEY)
            immune_coords = subset.loc[immune_mask, ["x", "y"]].values.astype(np.float32)
            x_range = subset["x"].max() - subset["x"].min()
            y_range = subset["y"].max() - subset["y"].min()
            area = float(max(x_range * y_range, 1.0))
            nnd_raw[core] = np.concatenate([
                nnd_raw[core],
                _compute_ripley_k(immune_coords, area),
            ])

        # Z-score normalize all spatial features (NND + Ripley's K) across all cores
        all_nnd = np.stack(list(nnd_raw.values()))
        self.nnd_mean = all_nnd.mean(axis=0)
        self.nnd_std = all_nnd.std(axis=0)
        self.nnd_std[self.nnd_std == 0] = 1.0  # avoid division by zero

        self.nnd_features = {}
        for core in self.cores:
            self.nnd_features[core] = (nnd_raw[core] - self.nnd_mean) / self.nnd_std

        # Extract and normalize clinical features
        self.clinical_feature_names = get_clinical_feature_names()
        self.n_clinical_features = len(self.clinical_feature_names)

        # Build raw clinical feature matrix
        clinical_raw = {}
        for core in self.cores:
            clinical_dict = self.data[core]["clinical"]
            clinical_raw[core] = np.array(
                [clinical_dict.get(name, np.nan) for name in self.clinical_feature_names],
                dtype=np.float32
            )

        # Compute mean/std for continuous features (first 3), ignoring NaN
        all_clinical = np.stack(list(clinical_raw.values()))
        self.clinical_mean = np.nanmean(all_clinical, axis=0)
        self.clinical_std = np.nanstd(all_clinical, axis=0)
        # Don't normalize binary features (last 3) - set their std to 1
        n_continuous = len(CLINICAL_CONTINUOUS)
        self.clinical_std[n_continuous:] = 1.0
        self.clinical_mean[n_continuous:] = 0.0
        self.clinical_std[self.clinical_std == 0] = 1.0  # avoid division by zero

        # Normalize and impute missing with 0 (mean after z-scoring)
        self.clinical_features = {}
        for core in self.cores:
            normalized = (clinical_raw[core] - self.clinical_mean) / self.clinical_std
            # Impute NaN with 0 (which is the mean after z-scoring for continuous,
            # and indicates "unknown" for binary)
            normalized = np.nan_to_num(normalized, nan=0.0)
            self.clinical_features[core] = normalized

    def __len__(self) -> int:
        return len(self.cores)

    def __getitem__(self, idx: int) -> Data:
        core = self.cores[idx]
        data = self.data[core]

        coords = data["coords"].copy()
        cell_types = data["cell_types"].copy()
        n_cells = len(coords)

        # Subsample if needed. Seeded per-core (not the global RNG state) so
        # which cells get kept is stable across reruns — otherwise any core
        # above max_cells silently gets a different random subset every time
        # the dataset is built, making downstream analysis non-reproducible.
        if self.max_cells is not None and n_cells > self.max_cells:
            rng = np.random.default_rng(_stable_seed(core))
            idx_sub = rng.choice(n_cells, self.max_cells, replace=False)
            coords = coords[idx_sub]
            cell_types = cell_types[idx_sub]
            n_cells = self.max_cells

        # Normalize coordinates to [0, 1]
        if self.normalize_coords and n_cells > 0:
            coords_min = coords.min(axis=0)
            coords_max = coords.max(axis=0)
            coords_range = coords_max - coords_min
            coords_range[coords_range == 0] = 1
            coords = (coords - coords_min) / coords_range

        # Convert to tensors
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        cell_types_tensor = torch.tensor(cell_types, dtype=torch.long)

        # Build spatial k-NN graph
        if n_cells > 1:
            edge_index = _knn_graph(coords_tensor, k=min(self.k, n_cells - 1), loop=False)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Node features: coordinates + one-hot cell type
        # Shape: (n_cells, 2 + n_cell_types)
        cell_type_onehot = torch.zeros(n_cells, self.n_cell_types, dtype=torch.float32)
        # Handle valid cell types only (ignore -1 for unknown types)
        valid_mask = cell_types_tensor >= 0
        valid_indices = torch.arange(n_cells)[valid_mask]
        valid_types = cell_types_tensor[valid_mask]
        cell_type_onehot[valid_indices, valid_types] = 1.0
        node_features = torch.cat([coords_tensor, cell_type_onehot], dim=1)

        # Graph labels: [time, event]
        graph_labels = torch.tensor(
            [data["time"], data["event"]], dtype=torch.float32
        )

        # PyG Data object with dict-based x / y attributes (keys in models/consts.py)
        pyg_data = Data(
            edge_index=edge_index,
            num_nodes=n_cells,
        )

        # Use dict-based x and y as expected by the models
        pyg_data.x = {
            BATCH_KEY_NODE_FEATURES: node_features,
            BATCH_KEY_NODE_LABELS: cell_type_onehot,
        }
        pyg_data.y = {
            BATCH_KEY_GRAPH_LABELS: graph_labels,
        }

        # Attach NND features
        pyg_data.nnd_features = torch.tensor(
            self.nnd_features[core], dtype=torch.float32
        )

        # Attach clinical features
        pyg_data.clinical_features = torch.tensor(
            self.clinical_features[core], dtype=torch.float32
        )

        # Store metadata for reference
        pyg_data.core = core
        pyg_data.n_cells = n_cells

        return pyg_data


def create_pytorch_dataset(
    df: pd.DataFrame,
    endpoint: str = "OS",
    k: int = 10,
    max_cells: Optional[int] = None,
    normalize_coords: bool = True,
    cell_types: Optional[list[str]] = None,
    require_survival: bool = True,
) -> PointPatternDataset:
    """
    Create a PyTorch Geometric Dataset from the point pattern DataFrame.

    Args:
        df: DataFrame with point pattern data (from data/create_dataset_metaclusters.py
            or synthetic/generate.py)
        endpoint: "DFS" or "OS" for survival endpoint
        k: Number of nearest neighbors for graph construction
        max_cells: Maximum cells per sample (None for no limit)
        normalize_coords: Whether to normalize coordinates to [0, 1]
        cell_types: List of cell type names. If None, auto-detects from data.
        require_survival: Drop cores with missing survival data. Set False for
            self-supervised pretraining.

    Returns:
        PointPatternDataset compatible with PyTorch Geometric DataLoader
    """
    return PointPatternDataset(
        df,
        endpoint=endpoint,
        k=k,
        max_cells=max_cells,
        normalize_coords=normalize_coords,
        cell_types=cell_types,
        require_survival=require_survival,
    )


def make_stratify_labels(subtype: "pd.Series", grade: "pd.Series") -> "pd.Series":
    """
    Build stratification labels for train/val splitting.

    Starts with subtype+grade joint labels, then iteratively collapses any
    stratum with fewer than 2 members: first by dropping grade (subtype-only),
    then by assigning to the most common stratum. Guarantees all strata have
    >= 2 members so sklearn's train_test_split never raises.

    Args:
        subtype: Series indexed by PID with clinical subtype strings.
        grade:   Series indexed by PID with grade strings.

    Returns:
        Series indexed by PID with final stratification labels.
    """
    labels = subtype.astype(str) + "_G" + grade.astype(str)
    for _ in range(3):  # at most 3 passes: fine -> coarse -> most common
        counts = labels.value_counts()
        rare = set(counts[counts < 2].index)
        if not rare:
            break
        most_common = counts.index[0]
        def _collapse(label, rare=rare, most_common=most_common):
            if label not in rare:
                return label
            if "_G" in label:
                return label.rsplit("_G", 1)[0]  # drop grade
            return most_common  # already at subtype-only, assign to most common
        labels = labels.apply(_collapse)
    return labels


# Version tag stored alongside saved CV folds, so loaders can tell folds made
# with an older stratification scheme apart and regenerate instead of reusing.
CV_STRATIFY_SCHEME = "subtype_grade_event"


def make_cv_folds(patient_df: "pd.DataFrame", n_folds: int, seed: int) -> list:
    """
    Patient-level stratified k-folds for the CV experiments.

    Stratifies on subtype × grade × event status. The single 80/20 split
    (patient_split) keeps subtype × grade only, so the explainability and
    sweep splits are unchanged. Event status is included because the C-index
    of a fold depends on how many events it holds. Strata with fewer than
    n_folds members are collapsed: first by dropping grade, then subtype
    (keeping event status), then into the most common stratum.

    Args:
        patient_df: one or more rows per patient with PID, clinical_type,
                    grade and event columns.
        n_folds:    number of folds.
        seed:       shuffle seed for StratifiedKFold.

    Returns:
        List of n_folds lists of validation PIDs (ints).
    """
    from sklearn.model_selection import StratifiedKFold

    per_pid = patient_df.groupby("PID").first()
    pids    = sorted(per_pid.index)
    subtype = per_pid["clinical_type"].fillna("Unknown").astype(str)
    grade   = per_pid["grade"].apply(lambda g: str(int(g)) if pd.notna(g) else "Unknown")
    event   = per_pid["event"].astype(int).astype(str)

    labels = subtype + "_G" + grade + "_E" + event
    for _ in range(4):  # fine -> drop grade -> drop subtype -> most common
        counts = labels.value_counts()
        rare = set(counts[counts < n_folds].index)
        if not rare:
            break
        most_common = counts.index[0]
        def _collapse(label, rare=rare, most_common=most_common):
            if label not in rare:
                return label
            if "_G" in label:
                sub, rest = label.split("_G", 1)
                return sub + "_E" + rest.rsplit("_E", 1)[1]   # drop grade
            if "_E" in label:
                return "E" + label.rsplit("_E", 1)[1]         # drop subtype
            return most_common
        labels = labels.apply(_collapse)

    stratify = [labels[p] for p in pids]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [[int(pids[i]) for i in val_idx] for _, val_idx in skf.split(pids, stratify)]


def patient_split(
    dataset: PointPatternDataset,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split dataset indices by patient ID to prevent data leakage.

    Multiple cores from the same patient are always kept in the same split.

    Args:
        dataset: PointPatternDataset instance
        val_size: Fraction of patients for validation set
        random_state: Random seed for reproducibility

    Returns:
        (train_indices, val_indices) as lists of dataset indices
    """
    from sklearn.model_selection import train_test_split

    # Build patient -> core indices mapping
    pid_to_indices: dict[int, list[int]] = {}
    for idx, core in enumerate(dataset.cores):
        pid = dataset.data[core]["PID"]
        pid_to_indices.setdefault(pid, []).append(idx)

    pids = sorted(pid_to_indices.keys())

    # Stratify jointly on clinical subtype + grade via shared helper.
    # Using stable clinical variables (not endpoint-specific event status)
    # means OS and PFS runs use the same val patients.
    subtype_s = pd.Series(
        {pid: dataset.data[dataset.cores[pid_to_indices[pid][0]]]["clinical_type"] or "Unknown"
         for pid in pids}
    )
    grade_s = pd.Series(
        {pid: (lambda v: str(int(v)) if v == v else "Unknown")(
            dataset.data[dataset.cores[pid_to_indices[pid][0]]]["clinical"].get("grade", float("nan"))
        ) for pid in pids}
    )
    label_s = make_stratify_labels(subtype_s, grade_s)
    stratify = [label_s[p] for p in pids]

    train_pids, val_pids = train_test_split(
        pids, test_size=val_size, random_state=random_state, stratify=stratify
    )

    # Map back to core indices
    train_idx = [i for pid in train_pids for i in pid_to_indices[pid]]
    val_idx = [i for pid in val_pids for i in pid_to_indices[pid]]

    return train_idx, val_idx


class StratifiedBatchSampler:
    """
    Batch sampler that ensures each batch has a balanced proportion of events.

    This stabilizes Cox PH loss by ensuring every batch has both events and
    censored observations to compare.
    """

    def __init__(self, labels: list[int], batch_size: int, shuffle: bool = True, seed: int = 42):
        """
        Args:
            labels: List of event indicators (0=censored, 1=event) for each sample
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle within strata each epoch
            seed: Random seed
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        # Split indices by event status
        self.event_indices = [i for i, y in enumerate(labels) if y == 1]
        self.censored_indices = [i for i, y in enumerate(labels) if y == 0]

        self.n_events = len(self.event_indices)
        self.n_censored = len(self.censored_indices)
        self.n_total = len(labels)

        # Target ratio of events per batch
        self.event_ratio = self.n_events / self.n_total
        self.events_per_batch = max(1, int(batch_size * self.event_ratio))
        self.censored_per_batch = batch_size - self.events_per_batch

    def __iter__(self):
        # Shuffle indices within each stratum
        if self.shuffle:
            self.rng.shuffle(self.event_indices)
            self.rng.shuffle(self.censored_indices)

        event_idx = 0
        censored_idx = 0

        batches = []
        while event_idx < self.n_events or censored_idx < self.n_censored:
            batch = []

            # Add events
            for _ in range(self.events_per_batch):
                if event_idx < self.n_events:
                    batch.append(self.event_indices[event_idx])
                    event_idx += 1

            # Add censored
            for _ in range(self.censored_per_batch):
                if censored_idx < self.n_censored:
                    batch.append(self.censored_indices[censored_idx])
                    censored_idx += 1

            if batch:
                if self.shuffle:
                    self.rng.shuffle(batch)
                batches.append(batch)

        # Shuffle batch order
        if self.shuffle:
            self.rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        # Approximate number of batches
        return (self.n_total + self.batch_size - 1) // self.batch_size


def get_event_labels(dataset_list: list) -> list[int]:
    """Extract event labels from a list of PyG Data objects."""
    return [int(d.y[BATCH_KEY_GRAPH_LABELS][1].item()) for d in dataset_list]
