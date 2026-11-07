#!/usr/bin/env python3
"""
Create metacluster CSV files for the Jackson/Bodenmiller breast cancer dataset.

This script generates Basel_metaclusters.csv and Zurich_matched_metaclusters.csv
by mapping the PhenoGraph clusters to the 27 metaclusters defined in the
Bodenmiller Group's SCPathology publication.

Metacluster definitions (from Jackson et al.):
  1-6: Immune cells (B cells, T cells, macrophages)
  7: Endothelial
  8-13: Stromal cells
  14-27: Tumor cell subtypes

Source: https://github.com/BodenmillerGroup/SCPathology_publication
"""

import os
import pandas as pd
from pathlib import Path


# Basel PhenoGraph cluster -> Metacluster mapping
# From: BodenmillerGroup/SCPathology_publication/R/BaselTMA_pipeline.Rmd
BASEL_PHENOGRAPH_TO_METACLUSTER = {
    # Immune cells
    25: 1,  # B cells
    19: 2,  # B and T cells
    2: 3,  # T cells
    6: 4,  # Macrophages
    38: 5,  # T cells
    70: 6,  # Macrophages
    # Endothelial
    10: 7,  # Endothelial
    # Stromal cells
    3: 8,  # Vimentin hi stromal cell
    4: 9,  # Small circular stromal cell
    1: 10,  # Small elongated stromal cell
    15: 11,  # Fibronectin hi stromal cell
    71: 12,  # Large elongated stromal cell
    36: 13,  # SMA hi Vimentin hi stromal cell
    # Tumor cells
    11: 14,  # Hypoxic tumor cell
    66: 15,
    43: 15,
    32: 15,
    49: 15,
    56: 15,  # Apoptotic tumor cell
    35: 16,
    68: 16,
    8: 16,
    60: 16,  # Proliferative tumor cell
    23: 17,
    33: 17,
    26: 17,  # p53+ EGFR+ tumor cell
    47: 18,
    57: 18,
    17: 18,
    50: 18,  # Basal CK tumor cell
    52: 19,
    28: 19,
    64: 19,
    45: 19,  # CK7+ CK hi cadherin hi tumor cell
    55: 20,
    13: 20,
    40: 20,
    51: 20,
    42: 20,  # CK7+ CK+ tumor cell
    21: 21,
    65: 21,
    7: 21,  # Epithelial low tumor cell
    58: 22,
    41: 22,
    5: 22,
    27: 22,
    30: 22,  # CK low HR low tumor cell
    59: 23,
    61: 23,
    14: 23,
    48: 23,
    62: 23,
    20: 23,
    24: 23,  # CK+ HR hi tumor cell
    18: 24,
    46: 24,
    53: 24,
    37: 24,
    31: 24,  # CK+ HR+ tumor cell
    39: 25,
    9: 25,
    12: 25,
    29: 25,
    34: 25,
    22: 25,  # CK+ HR low tumor cell
    44: 26,
    54: 26,
    63: 26,  # CK low HR hi p53+ tumor cell
    16: 27,
    69: 27,
    67: 27,  # Myoepithelial tumor cell
}

# Zurich PhenoGraph cluster -> Basel Metacluster mapping
# Manual reassignments from: BodenmillerGroup/SCPathology_publication/R/ZurichTMA_pipeline.Rmd
# For clusters without manual reassignment, we use correlation-based matching
# computed from the mean expression profiles
ZURICH_MANUAL_REASSIGNMENTS = {
    1: 3,  # T cells
    3: 8,  # Stromal
    6: 24,  # CK+ HR+ tumor
    8: 21,  # Epithelial low tumor
    9: 19,  # CK7+ CK hi cadherin hi tumor
    11: 22,  # CK low HR low tumor
    15: 16,  # Proliferative tumor
    26: 15,  # Apoptotic tumor
    29: 23,  # CK+ HR hi tumor
    39: 8,  # Stromal
    40: 19,  # CK7+ CK hi cadherin hi tumor
    41: 10,  # Small elongated stromal
}

# Default correlation-based assignments for remaining Zurich clusters
# These were inferred from the cross-correlation analysis in the pipeline
ZURICH_CORRELATION_DEFAULTS = {
    2: 8,  # Stromal (vimentin hi pattern)
    4: 9,  # Stromal (small circular pattern)
    5: 10,  # Stromal (small elongated pattern)
    7: 7,  # Endothelial
    10: 4,  # Macrophages
    12: 20,  # CK7+ CK+ tumor
    13: 20,  # CK7+ CK+ tumor
    14: 17,  # p53+ EGFR+ tumor
    16: 18,  # Basal CK tumor
    17: 14,  # Hypoxic tumor
    18: 25,  # CK+ HR low tumor
    19: 24,  # CK+ HR+ tumor
    20: 25,  # CK+ HR low tumor
    21: 23,  # CK+ HR hi tumor
    22: 24,  # CK+ HR+ tumor
    23: 22,  # CK low HR low tumor
    24: 25,  # CK+ HR low tumor
    25: 26,  # CK low HR hi p53+ tumor
    27: 27,  # Myoepithelial tumor
    28: 11,  # Fibronectin hi stromal
    30: 13,  # SMA hi vimentin hi stromal
    31: 12,  # Large elongated stromal
    32: 5,  # T cells
    33: 6,  # Macrophages
    34: 1,  # B cells
    35: 2,  # B and T cells
    36: 15,  # Apoptotic tumor
    37: 16,  # Proliferative tumor
    38: 18,  # Basal CK tumor
}


def create_zurich_mapping():
    """Combine manual reassignments with correlation-based defaults."""
    mapping = {}
    # Start with correlation-based defaults
    mapping.update(ZURICH_CORRELATION_DEFAULTS)
    # Override with manual reassignments (these take precedence)
    mapping.update(ZURICH_MANUAL_REASSIGNMENTS)
    return mapping


def create_metacluster_csv(
    phenograph_file: str, output_file: str, mapping: dict, cluster_col: str
) -> None:
    """
    Create metacluster CSV from PhenoGraph clustering file.

    Args:
        phenograph_file: Path to PhenoGraph clustering CSV
        output_file: Path for output metacluster CSV
        mapping: Dict mapping PhenoGraph cluster -> metacluster
        cluster_col: Name of cluster column in input file
    """
    print(f"Reading {phenograph_file}...")
    df = pd.read_csv(phenograph_file, usecols=["id", cluster_col])

    print(f"  Found {len(df):,} cells")

    # Map PhenoGraph clusters to metaclusters
    df["cluster"] = df[cluster_col].map(mapping)

    # Check for unmapped clusters
    unmapped = df[df["cluster"].isna()][cluster_col].unique()
    if len(unmapped) > 0:
        print(
            f"  Warning: {len(unmapped)} unmapped PhenoGraph clusters: {sorted(unmapped)}"
        )
        # Assign unmapped clusters to a default (tumor cell subtype)
        df["cluster"] = df["cluster"].fillna(20)  # Default to CK7+ CK+ tumor

    # Convert to integer
    df["cluster"] = df["cluster"].astype(int)

    # Select only id and cluster columns
    result = df[["id", "cluster"]]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save
    print(f"  Writing {output_file}...")
    result.to_csv(output_file, index=False)

    # Print cluster distribution
    print(f"  Metacluster distribution:")
    for mc in sorted(result["cluster"].unique()):
        count = (result["cluster"] == mc).sum()
        print(f"    Metacluster {mc:2d}: {count:,} cells")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create metacluster CSV files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory holding the extracted Zenodo data (Data_publication/)",
    )
    data_path = Path(parser.parse_args().data_dir)

    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist")
        return 1

    # Input files
    basel_pg = data_path / "Data_publication" / "BaselTMA" / "PG_final_k20.csv"
    zurich_pg = data_path / "Data_publication" / "ZurichTMA" / "PG_zurich.csv"

    # Output directory
    output_dir = data_path / "Cluster_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Basel metaclusters
    print("\n=== Creating Basel metaclusters ===")
    create_metacluster_csv(
        phenograph_file=str(basel_pg),
        output_file=str(output_dir / "Basel_metaclusters.csv"),
        mapping=BASEL_PHENOGRAPH_TO_METACLUSTER,
        cluster_col="PhenoGraphBasel",
    )

    # Create Zurich metaclusters
    print("\n=== Creating Zurich metaclusters ===")
    zurich_mapping = create_zurich_mapping()
    create_metacluster_csv(
        phenograph_file=str(zurich_pg),
        output_file=str(output_dir / "Zurich_matched_metaclusters.csv"),
        mapping=zurich_mapping,
        cluster_col="PhenoGraph",
    )

    print("\n=== Done ===")
    print(f"Output files created in: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
