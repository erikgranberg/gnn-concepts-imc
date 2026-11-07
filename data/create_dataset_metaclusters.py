"""
Breast Cancer IMC Point Pattern Dataset - Metacluster Version
==============================================================

Creates point patterns using the 27 metaclusters from Jackson et al. (2020)
grouped into biologically meaningful categories.

This replaces the binary tumor/immune classification (Maisel thresholding)
with richer cell type information derived from PhenoGraph clustering.

Metacluster groups:
    - T_cell: T cells (clusters 3, 5)
    - B_cell: B cells (clusters 1, 2)
    - macrophage: Macrophages (clusters 4, 6)
    - endothelial: Endothelial cells (cluster 7)
    - stromal: Stromal subtypes (clusters 8-13)
    - tumor: All tumor subtypes (clusters 14-27)
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from skimage import io
from scipy import ndimage
from typing import Optional


# =============================================================================
# METACLUSTER CONFIGURATION
# =============================================================================

# Map 27 metaclusters to grouped categories
METACLUSTER_TO_GROUP = {
    # Immune - T cells
    3: "T_cell",
    5: "T_cell",
    # Immune - B cells
    1: "B_cell",
    2: "B_cell",  # T and B cells - assign to B for now
    # Immune - Macrophages
    4: "macrophage",
    6: "macrophage",
    # Endothelial
    7: "endothelial",
    # Stromal subtypes
    8: "stromal",   # vimentin hi
    9: "stromal",   # small circular
    10: "stromal",  # small elongated
    11: "stromal",  # fibronectin hi
    12: "stromal",  # large elongated
    13: "stromal",  # SMA hi vimentin hi
    # Tumor subtypes
    14: "tumor",  # hypoxic
    15: "tumor",  # apoptotic
    16: "tumor",  # proliferative
    17: "tumor",  # p53+ EGFR+
    18: "tumor",  # basal CK
    19: "tumor",  # CK7+ CK hi cadherin hi
    20: "tumor",  # CK7+ CK+
    21: "tumor",  # epithelial low
    22: "tumor",  # CK low HR low
    23: "tumor",  # CK+ HR hi
    24: "tumor",  # CK+ HR+
    25: "tumor",  # CK+ HR low
    26: "tumor",  # CK low HR hi p53+
    27: "tumor",  # myoepithelial
}

# Cell type categories for the model
CELL_TYPES = ["tumor", "T_cell", "B_cell", "macrophage", "stromal", "endothelial"]


@dataclass
class DataPaths:
    """Configure paths to data files."""

    data_publication_dir: Path = Path("data/raw/Data_publication")
    masks_dir: Path = Path("data/raw/OMEnMasks/Basel_Zuri_masks")
    cluster_labels_dir: Path = Path("data/raw/Cluster_labels")

    def cohort_dir(self, cohort: str) -> Path:
        if cohort == "Basel":
            return self.data_publication_dir / "BaselTMA"
        elif cohort == "Zurich":
            return self.data_publication_dir / "ZurichTMA"
        else:
            raise ValueError(f"Unknown cohort: {cohort}")

    def patient_metadata(self, cohort: str) -> Path:
        if cohort == "Basel":
            return self.cohort_dir(cohort) / "Basel_PatientMetadata.csv"
        else:
            return self.cohort_dir(cohort) / "Zuri_PatientMetadata.csv"

    def metacluster_file(self, cohort: str) -> Path:
        if cohort == "Basel":
            return self.cluster_labels_dir / "Basel_metaclusters.csv"
        else:
            return self.cluster_labels_dir / "Zurich_matched_metaclusters.csv"


# =============================================================================
# MODULE 1: CELL TYPE FROM METACLUSTERS
# =============================================================================


def load_metaclusters(path: Path) -> pd.DataFrame:
    """
    Load metacluster assignments and map to grouped cell types.

    The metacluster CSV has columns: id, cluster
    where id is "{core}_{cell_number}" format.
    """
    df = pd.read_csv(path)

    # Parse the id column to extract core and CellId
    # Format: "corename_cellnumber"
    df["core"] = df["id"].str.rsplit("_", n=1).str[0]
    df["CellId"] = df["id"].str.rsplit("_", n=1).str[1].astype(int)

    # Map metacluster to grouped cell type
    df["cell_type"] = df["cluster"].map(METACLUSTER_TO_GROUP)

    # Keep original cluster for potential fine-grained analysis
    df["metacluster"] = df["cluster"]

    return df[["core", "CellId", "cell_type", "metacluster"]]


# =============================================================================
# MODULE 2: COORDINATE EXTRACTION FROM MASKS
# =============================================================================


def find_mask_path(fullstack_filename: str, masks_path: Path) -> Optional[str]:
    """Find mask file, trying both naming conventions."""
    for suffix in ["_full_maks.tiff", "_full_mask.tiff"]:
        mask_name = fullstack_filename.replace("_full.tiff", suffix)
        mask_path = masks_path / mask_name
        if mask_path.exists():
            return str(mask_path)
    return None


def extract_centroids_from_mask(mask_path: Path) -> pd.DataFrame:
    """
    Extract cell centroids from a segmentation mask.

    Mask is a labeled image where pixel value = cell ID (0 = background).
    Returns DataFrame with CellId, x, y columns.
    """
    mask = io.imread(mask_path)

    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids != 0]

    if len(cell_ids) == 0:
        return pd.DataFrame(columns=["CellId", "x", "y"])

    cell_ids_sorted = np.sort(cell_ids)

    centroids = ndimage.center_of_mass(
        np.ones_like(mask), labels=mask, index=cell_ids_sorted
    )

    return pd.DataFrame(
        {
            "CellId": np.arange(1, len(cell_ids_sorted) + 1),
            "x": [c[1] for c in centroids],
            "y": [c[0] for c in centroids],
        }
    )


# =============================================================================
# MODULE 3: PATIENT METADATA
# =============================================================================


def load_patient_metadata(patient_data_path: Path, masks_path: Path) -> pd.DataFrame:
    """Load patient metadata with raw survival information."""
    meta = pd.read_csv(patient_data_path)

    meta["mask_filename"] = meta["FileName_FullStack"].apply(
        lambda x: find_mask_path(x, masks_path)
    )

    cols = [
        "core",
        "PID",
        "FileName_FullStack",
        "mask_filename",
        "Patientstatus",
        "DFSmonth",
        "OSmonth",
        "clinical_type",
        "ERStatus",
        "PRStatus",
        "HER2Status",
        "grade",
        "tumor_size",
        "age",
    ]
    cols = [c for c in cols if c in meta.columns]

    return meta[cols]


# =============================================================================
# MODULE 4: DATA ASSEMBLY
# =============================================================================


def build_point_patterns(
    paths: DataPaths,
    cohort: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build complete point pattern dataset using metaclusters.

    Args:
        paths: Data paths configuration
        cohort: "Basel", "Zurich", or None for both

    Returns:
        DataFrame with columns: core, CellId, cell_type, metacluster, x, y, + metadata
    """
    cohorts = [cohort] if cohort else ["Basel", "Zurich"]
    all_patterns = []

    for coh in cohorts:
        print(f"\n{'='*60}")
        print(f"Processing {coh} cohort")
        print("=" * 60)

        # Load metaclusters
        print("Loading metaclusters...")
        metaclusters = load_metaclusters(paths.metacluster_file(coh))
        print(f"  Loaded {len(metaclusters):,} cell annotations")

        # Cell type distribution
        print("  Cell type distribution:")
        for ct in CELL_TYPES:
            count = (metaclusters["cell_type"] == ct).sum()
            pct = count / len(metaclusters) * 100
            print(f"    {ct}: {count:,} ({pct:.1f}%)")

        # Load patient metadata
        print("Loading patient metadata...")
        metadata = load_patient_metadata(
            paths.patient_metadata(coh), paths.masks_dir
        )
        metadata["cohort"] = coh
        print(f"  Found {len(metadata)} cores with metadata")

        # Process each core
        print("Extracting coordinates from masks...")
        patterns = []
        cores_processed = 0
        cores_skipped = 0

        for _, row in metadata.iterrows():
            core = row["core"]

            if pd.isna(row["mask_filename"]):
                cores_skipped += 1
                continue

            # Extract coordinates
            coords = extract_centroids_from_mask(Path(row["mask_filename"]))
            if len(coords) == 0:
                cores_skipped += 1
                continue

            coords["core"] = core

            # Merge with metaclusters
            core_metaclusters = metaclusters[metaclusters["core"] == core]
            coords = coords.merge(
                core_metaclusters[["CellId", "cell_type", "metacluster"]],
                on="CellId",
                how="left",
            )

            # Drop cells without metacluster assignment
            n_before = len(coords)
            coords = coords.dropna(subset=["cell_type"])
            n_dropped = n_before - len(coords)
            if n_dropped > 0:
                print(f"  Warning: {core} - {n_dropped} cells without metacluster")

            # Add metadata
            for col in metadata.columns:
                if col not in ["core", "mask_filename"]:
                    coords[col] = row[col]

            patterns.append(coords)
            cores_processed += 1

        print(f"  Processed {cores_processed} cores, skipped {cores_skipped}")
        all_patterns.extend(patterns)

    # Combine all
    result = pd.concat(all_patterns, ignore_index=True)

    # Fix column types for parquet compatibility
    for col in ["grade", "tumor_size", "age"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    print(f"\n{'='*60}")
    print("FINAL DATASET")
    print("=" * 60)
    print(f"Total cells: {len(result):,}")
    print(f"Total cores: {result['core'].nunique()}")
    print(f"Total patients: {result['PID'].nunique()}")
    print("\nCell type distribution:")
    for ct in CELL_TYPES:
        count = (result["cell_type"] == ct).sum()
        pct = count / len(result) * 100
        print(f"  {ct}: {count:,} ({pct:.1f}%)")

    return result


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create point pattern dataset using metaclusters"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/point_patterns_metaclusters.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        choices=["Basel", "Zurich"],
        help="Process single cohort (default: both)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory holding the extracted Zenodo data (Data_publication/, "
             "OMEnMasks/) and the Cluster_labels/ written by create_metaclusters.py",
    )
    args = parser.parse_args()

    # Configure paths
    base = Path(args.data_dir)
    paths = DataPaths(
        data_publication_dir=base / "Data_publication",
        masks_dir=base / "OMEnMasks" / "Basel_Zuri_masks",
        cluster_labels_dir=base / "Cluster_labels",
    )

    # Build dataset
    df = build_point_patterns(paths, cohort=args.cohort)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
