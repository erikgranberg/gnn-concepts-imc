"""
Nearest-neighbour distance (NND) metrics following Maisel et al. (2022).

For each core, computes distances from every tumour cell to its nearest
immune cell and summarises them as:
  - median / mean NND
  - fraction of tumour cells with an immune cell within 12 um (adjacent)
  - fraction of tumour cells with an immune cell within 250 um (communicating)
  - number of immune cells

Reference: Maisel et al. (2022) "Spatial Metrics of Interaction between
CD163-Positive Macrophages and Cancer Cells" Cancers 14(2):308
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SpatialMetrics:
    """Spatial metrics for a single core."""

    core: str
    n_tumor: int
    n_immune: int
    median_nnd: float
    mean_nnd: float
    std_nnd: float
    n_adjacent: int  # tumor cells with immune cell within 12 μm
    n_communicating: int  # tumor cells with immune cell within 250 μm
    frac_adjacent: float
    frac_communicating: float
    # Survival data
    time: float
    event: int


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================


def compute_nnd_metrics(
    df_core: pd.DataFrame,
    adjacent_threshold: float = 12.0,
    communicating_threshold: float = 250.0,
) -> SpatialMetrics | None:
    """
    Compute NND metrics for a single core.

    Coordinates are assumed to be in pixels (~1 μm/pixel for IMC).
    """
    tumor = df_core[df_core["cell_type"] == "tumor"]
    immune = df_core[df_core["cell_type"] == "immune"]

    n_tumor = len(tumor)
    n_immune = len(immune)

    if n_tumor == 0 or n_immune == 0:
        return None

    # Build KD-tree for immune cells
    immune_coords = immune[["x", "y"]].values
    tumor_coords = tumor[["x", "y"]].values
    tree = cKDTree(immune_coords)

    # Query NND for each tumor cell
    distances, _ = tree.query(tumor_coords, k=1)

    # Compute metrics
    median_nnd = np.median(distances)
    mean_nnd = np.mean(distances)
    std_nnd = np.std(distances)

    # Count tumor cells with immune cell within thresholds
    n_adjacent = np.sum(distances <= adjacent_threshold)
    n_communicating = np.sum(distances <= communicating_threshold)

    return SpatialMetrics(
        core=df_core["core"].iloc[0],
        n_tumor=n_tumor,
        n_immune=n_immune,
        median_nnd=median_nnd,
        mean_nnd=mean_nnd,
        std_nnd=std_nnd,
        n_adjacent=n_adjacent,
        n_communicating=n_communicating,
        frac_adjacent=n_adjacent / n_tumor,
        frac_communicating=n_communicating / n_tumor,
        time=df_core["time"].iloc[0],
        event=int(df_core["event"].iloc[0]),
    )
