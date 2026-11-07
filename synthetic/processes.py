"""
Spatial point process implementations for synthetic dataset generation.

Four regimes of increasing spatial complexity:

  Regime 1 — Composition
    Survival driven by immune cell fraction. Positions are uniform Poisson.
    Any model that counts cell types should solve this.

  Regime 2 — Proximity (NND)
    Immune fraction is fixed. Survival is driven by how close immune cells
    are to tumor cells (median NND). A proximity parameter p ∈ [0, 1]
    controls each patient: p=1 means immune cells are placed directly near
    tumor cells, p=0 means they are placed uniformly at random.
    NND-based features (and the GNN) should both capture this signal.

  Regime 3 — Higher-order clustering (Thomas process)
    Immune fraction and mean NND are approximately fixed. Survival is driven
    by the degree of clustering of immune cells around tumor nests, controlled
    by the Thomas process cluster scale σ_thomas. Small σ means tight immune
    clusters; large σ means dispersed. Only models sensitive to higher-order
    spatial structure (GNN, Ripley's K) should capture this reliably.

  Regime 4 — Independent immune clustering (Ripley's K-sensitive, NND-blind)
    Immune cells form Thomas-process clusters at *random* locations, not
    anchored to tumor nests. Tumor cells are uniform Poisson. Median NND
    (tumour→immune) is approximately constant across patients because the
    expected distance to the nearest cluster centre does not depend on σ.
    Ripley's K of immune cells varies strongly with σ_thomas. Only models
    that measure immune self-clustering (Ripley's K, GNN) should detect
    the signal; NND-based features should be near-random.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PatternConfig:
    """Per-patient spatial pattern parameters."""
    n_tumor:     int
    n_immune:    int
    signal:      float    # the latent variable that drives survival


def _uniform_poisson(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly in [0, 1]^2. Returns (n, 2) array."""
    return rng.uniform(0.0, 1.0, size=(n, 2))


def _near_anchor(anchors: np.ndarray, n: int, sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Place n points near randomly chosen anchor points.
    Each point is drawn from N(anchor, sigma^2 I), clipped to [0,1]^2.
    """
    idx = rng.integers(0, len(anchors), size=n)
    offsets = rng.normal(0, sigma, size=(n, 2))
    pts = anchors[idx] + offsets
    return np.clip(pts, 0.0, 1.0)


# ── Regime 1: Composition ──────────────────────────────────────────────────────

def regime1_composition(
    rng: np.random.Generator,
    n_cells: int = 300,
    frac_immune_range: tuple[float, float] = (0.05, 0.60),
) -> tuple[np.ndarray, np.ndarray, PatternConfig]:
    """
    Generate a pattern where survival signal = immune cell fraction.

    Returns:
        tumor_coords: (n_tumor, 2)
        immune_coords: (n_immune, 2)
        config: PatternConfig with signal = frac_immune
    """
    frac = rng.uniform(*frac_immune_range)
    n_immune = max(1, int(n_cells * frac))
    n_tumor  = n_cells - n_immune

    tumor_coords  = _uniform_poisson(n_tumor,  rng)
    immune_coords = _uniform_poisson(n_immune, rng)

    return tumor_coords, immune_coords, PatternConfig(
        n_tumor=n_tumor, n_immune=n_immune, signal=frac
    )


# ── Regime 2: Proximity (NND) ──────────────────────────────────────────────────

def regime2_proximity(
    rng: np.random.Generator,
    n_cells: int = 300,
    frac_immune: float = 0.25,
    proximity_sigma: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, PatternConfig]:
    """
    Generate a pattern where survival signal = proximity of immune to tumor.

    A proximity parameter p ∈ [0, 1] is drawn per patient:
      - p=1: all immune cells placed near (within proximity_sigma of) tumor cells
      - p=0: all immune cells placed uniformly at random
    Intermediate values linearly interpolate the two placement modes.

    Returns signal = p (higher = better survival in downstream model).
    """
    p = rng.uniform(0.0, 1.0)
    n_immune = max(1, int(n_cells * frac_immune))
    n_tumor  = n_cells - n_immune

    tumor_coords = _uniform_poisson(n_tumor, rng)

    n_near   = int(n_immune * p)
    n_random = n_immune - n_near

    near_coords   = (_near_anchor(tumor_coords, n_near,   proximity_sigma, rng)
                     if n_near   > 0 else np.empty((0, 2)))
    random_coords = (_uniform_poisson(n_random, rng)
                     if n_random > 0 else np.empty((0, 2)))

    immune_coords = np.vstack([c for c in [near_coords, random_coords] if len(c)])

    return tumor_coords, immune_coords, PatternConfig(
        n_tumor=n_tumor, n_immune=n_immune, signal=float(p)
    )


# ── Regime 3: Higher-order clustering (Thomas process) ────────────────────────

def regime3_clustering(
    rng: np.random.Generator,
    n_cells: int = 300,
    frac_immune: float = 0.25,
    n_tumor_clusters: int = 8,
    sigma_range: tuple[float, float] = (0.01, 0.15),
) -> tuple[np.ndarray, np.ndarray, PatternConfig]:
    """
    Generate a pattern where survival signal = immune clustering scale (σ_thomas).

    Tumor cells form n_tumor_clusters Gaussian clusters (fixed).
    Immune cells are placed via a Thomas process centred on tumor cluster
    parents, with scale σ_thomas drawn per patient.

    Small σ_thomas → tight immune clusters near tumor nests → better survival.
    Large σ_thomas → dispersed immune cells → worse survival.

    Mean NND and immune fraction are approximately equal across patients,
    so NND-only models should struggle, while the GNN can learn clustering.

    Returns signal = 1 - σ_thomas_normalised  (higher = better survival).
    """
    n_immune = max(1, int(n_cells * frac_immune))
    n_tumor  = n_cells - n_immune

    # Tumor cells: fixed cluster structure
    cluster_centers = rng.uniform(0.1, 0.9, size=(n_tumor_clusters, 2))
    tumor_coords    = _near_anchor(cluster_centers, n_tumor, sigma=0.06, rng=rng)

    # Immune cells: Thomas process around tumor cluster centres
    sigma_thomas = rng.uniform(*sigma_range)
    immune_coords = _near_anchor(cluster_centers, n_immune,
                                 sigma=sigma_thomas, rng=rng)

    # Normalise signal so 1 = tight clusters, 0 = dispersed
    lo, hi = sigma_range
    signal = 1.0 - (sigma_thomas - lo) / (hi - lo)

    return tumor_coords, immune_coords, PatternConfig(
        n_tumor=n_tumor, n_immune=n_immune, signal=float(signal)
    )


# ── Regime 4: Independent immune clustering (Ripley's K) ──────────────────────

def regime4_ripley(
    rng: np.random.Generator,
    n_cells: int = 300,
    frac_immune: float = 0.25,
    n_immune_clusters: int = 6,
    sigma_range: tuple[float, float] = (0.01, 0.15),
) -> tuple[np.ndarray, np.ndarray, PatternConfig]:
    """
    Generate a pattern where survival signal = immune self-clustering scale.

    Tumor cells are placed uniformly at random (no cluster structure).
    Immune cells form a Thomas process with n_immune_clusters parents placed
    uniformly at random — independent of the tumour positions.

    Because cluster centres are uniform random and the number of immune cells
    is fixed, the expected tumour→immune NND does not depend on σ_thomas.
    Ripley's K of immune cells, however, varies strongly: small σ produces
    tight immune clusters (large K at small radii), large σ gives dispersed
    placement (K ≈ πr²).

    Small σ_thomas → tight immune self-clusters → better survival.
    Large σ_thomas → dispersed immune cells → worse survival.

    Returns signal = 1 - σ_thomas_normalised (higher = better survival).
    """
    n_immune = max(1, int(n_cells * frac_immune))
    n_tumor  = n_cells - n_immune

    tumor_coords = _uniform_poisson(n_tumor, rng)

    # Immune cells: Thomas process with random cluster centres (not near tumor)
    cluster_centers = rng.uniform(0.05, 0.95, size=(n_immune_clusters, 2))
    sigma_thomas    = rng.uniform(*sigma_range)
    immune_coords   = _near_anchor(cluster_centers, n_immune,
                                   sigma=sigma_thomas, rng=rng)

    lo, hi = sigma_range
    signal = 1.0 - (sigma_thomas - lo) / (hi - lo)

    return tumor_coords, immune_coords, PatternConfig(
        n_tumor=n_tumor, n_immune=n_immune, signal=float(signal)
    )
