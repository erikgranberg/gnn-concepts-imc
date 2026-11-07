# Interpretable GNN survival models for spatial point patterns

Code for the master's thesis "Explainability for Spatial Graph Neural Networks in Breast
Cancer Survival Prediction: A Synthetic Validation Approach". It covers
spatial survival analysis of breast cancer Imaging Mass Cytometry (IMC) data
from [Jackson et al. (2020)](https://doi.org/10.1038/s41586-019-1876-x).

Each tissue core becomes a k-NN graph over cells (k = 12). A 3-layer GCN
encodes the graph, and the pooled embedding is concatenated with
nearest-neighbour-distance (NND) and Ripley's K features and the clinical
features at a bottleneck. An MLP head outputs a log-hazard, trained with the
Cox partial likelihood. The model is compared against clinical and
spatial-statistic Cox PH baselines with 5-fold cross-validation. It is checked
on four synthetic regimes with a known spatial signal, then analysed with
permutation ablation, an embedding probe and GCExplainer-style concept
discovery.

## Repository structure

```
├── data/                        raw IMC data → point-pattern parquet
│   ├── create_metaclusters.py       PhenoGraph clusters → 27 metaclusters
│   └── create_dataset_metaclusters.py  cell centroids + 6 cell-type groups + clinical data
├── synthetic/                   synthetic regimes 1–4 and the synthetic validation table
├── models/                      GCN survival model (graph_models.py, modules/)
├── pretraining/                 self-supervised masked cell-type/position pretraining
├── figures/                     figure implementations (dataset, baselines, Fig 7)
├── analysis/
│   ├── pretrain_eval.py             pretraining evaluation → Fig 8
│   └── explainability/              ablation, embedding probe, concept discovery
├── data_utils.py                PyG dataset, k-NN graphs, NND/Ripley/clinical features, splits
├── nnd_analysis.py              NND metrics (Maisel et al. 2022)
├── clinical_baseline.py         Cox PH baselines (clinical, NND, Ripley's K)
├── train.py                     GNN training (Cox loss, early stopping, MLflow logging)
├── sweep.py                     hyperparameter grid search
├── run_cv.py                    5-fold CV of all models → Fig 7
├── generate_figures.py          Figs 1, 2, 3, 5, 6b
├── cv_folds.json                the 5 CV folds (patient IDs) used in the thesis
├── best_model.pt                IMC model used for the explainability analyses
├── pretrained_metacluster_nocap.pt   pretrained GCN encoder
├── models/synthetic_regime{1-4}_noise0.0.pt   synthetic models used for explainability
└── results/                     CSVs behind the tables and figures
```

## Installation

Install with [uv](https://github.com/astral-sh/uv) (Python ≥ 3.12):

```bash
uv sync
```

Every command below runs from the repository root with `uv run`. Runs are
logged to MLflow in a local `mlflow.db`, which you can browse with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Data

### IMC data (Jackson et al. 2020)

The raw data is the Zenodo record
[10.5281/zenodo.3518284](https://zenodo.org/records/3518284), licensed CC BY.
You need `SingleCell_and_Metadata.zip` (patient metadata and PhenoGraph
clusters) and `OMEandSingleCellMasks.zip` (single-cell segmentation masks).
Arrange them under `data/raw/` as follows:

```
data/raw/
├── Data_publication/
│   ├── BaselTMA/   Basel_PatientMetadata.csv, PG_final_k20.csv
│   └── ZurichTMA/  Zuri_PatientMetadata.csv,  PG_zurich.csv
└── OMEnMasks/Basel_Zuri_masks/   *_full_mask.tiff
```

Then build the dataset (285 patients with survival data, 723 cores, ~1.2 M
cells, six cell types: tumour, T cell, B cell, macrophage, stromal,
endothelial):

```bash
uv run python data/create_metaclusters.py            # writes data/raw/Cluster_labels/
uv run python data/create_dataset_metaclusters.py    # writes datasets/point_patterns_metaclusters.parquet
```

### Synthetic data

```bash
uv run python synthetic/generate.py     # datasets/synthetic_regime{1-4}_noise0.0.parquet
```

The generator is deterministic and reproduces the thesis datasets exactly
(β = 5, τ = 0). See [`synthetic/README.md`](synthetic/README.md).

## Reproducing the thesis results

The defaults of every script are the configuration used in the thesis: GCN
with 3 layers, hidden dimension 32, k = 12, learning rate 5e-4, batch size 64,
at most 100 epochs, early-stopping patience 15, endpoint DFS, seed 42.
Section, table and appendix numbers refer to the thesis. "Fig N" refers to the
output file names the scripts write to `figures/output/` (e.g.
`fig7_model_comparison_dfs.pdf`).

### Real-data model comparison (Section 5.2, Fig 6b, Fig 7)

```bash
# Hyperparameter sweep (81 configurations, single 80/20 split)
uv run python sweep.py --clinical                     # → results/sweep_DFS_scratch.csv
uv run python sweep.py --clinical \
  --pretrained pretrained_metacluster_nocap.pt \
  --encoder_lr 5e-5 1e-4                              # → results/sweep_DFS_pretrained.csv

# 5-fold CV of the Cox PH baselines and all six GNN variants → Fig 7
uv run python run_cv.py --encoder_lr 1e-4             # → results/cv_DFS.csv

# Clinical Cox PH forest plot → Fig 6b
uv run python generate_figures.py --figs 6b
```

`run_cv.py` reuses the patient folds in `cv_folds.json`, which were stratified
on subtype × grade (seed 42) and are the folds used in the thesis.
`--new_folds` generates fresh folds stratified on subtype × grade × event
instead. `--plot_only` re-reads the GNN results from MLflow rather than
retraining.

### Self-supervised pretraining (Section 4.4, Fig 8)

```bash
uv run python -m pretraining.masked_pretrain --output pretrained_metacluster_nocap.pt
uv run python -m analysis.pretrain_eval --pretrained pretrained_metacluster_nocap.pt
# → figures/output/fig8_pretraining_simple.pdf, results/pretrain_eval.csv
```

### Synthetic validation (Section 5.1, Table 5.1)

```bash
uv run python synthetic/run_synthetic.py              # NND, Ripley, NND+Ripley, GNN full, oracle
uv run python synthetic/run_synthetic.py --gnn_only   # GNN only
```

Both report the mean ± SD over five random 80/20 splits (seeds 1–5).
`results/synthetic_nnd_ripley_combined_cox.csv` holds the per-seed values of
the NND, Ripley's K and combined Cox PH models.

### Explainability (Chapter 6)

The models analysed are the checkpoints in this repository. To retrain them:

```bash
# IMC model (best_model.pt)
uv run python train.py --clinical \
  --pretrained pretrained_metacluster_nocap.pt --encoder_lr 1e-4

# Synthetic models
for r in 1 2 3 4; do
  uv run python train.py --data datasets/synthetic_regime${r}_noise0.0.parquet \
    --model_out models/synthetic_regime${r}_noise0.0.pt
done
```

Block and feature permutation ablation, graph- and node-level concepts and
the embedding probe:

```bash
uv run python analysis/explainability/run_explainability.py \
  --model_path best_model.pt --clinical

for r in 1 2 3 4; do
  uv run python analysis/explainability/run_explainability.py \
    --model_path models/synthetic_regime${r}_noise0.0.pt \
    --data datasets/synthetic_regime${r}_noise0.0.parquet --synthetic --regime $r
  # node-concept correlation with the true signal (Table 6.3, |r| column)
  uv run python analysis/explainability/node_concept_signal.py \
    --model_path models/synthetic_regime${r}_noise0.0.pt \
    --data datasets/synthetic_regime${r}_noise0.0.parquet --synthetic --regime $r
done

# Silhouette / completeness vs K for all five models → Appendix B
uv run python analysis/explainability/k_search_combined.py   # → results/graph_k_search_dfs.csv

# Lowest/highest-signal example cores per regime (Fig 3.2)
uv run python -m analysis.explainability.signal_examples
```

Outputs go to `figures/output/explainability/`. The analyses use the
held-out patients of the single stratified 80/20 split (seed 42).

### Dataset figures (Figs 1, 2, 3, 5)

```bash
uv run python generate_figures.py --figs 1 2 3 5
```

## Citation

If you use this code, please cite the thesis.

## License

The code is released under the [MIT License](LICENSE). The IMC data from
Jackson et al. (2020) is not included, it is distributed under its own
license (CC BY) via Zenodo.
