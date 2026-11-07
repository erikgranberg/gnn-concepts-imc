# Synthetic point-pattern datasets

Synthetic survival datasets with a known ground-truth spatial signal, used to
check which spatial signal each model (NND, Ripley's K, GNN) captures and to
validate the explainability methods (thesis Sections 3.2, 5.1 and 6.3).

## Design

Each dataset has 300 simulated patients with one core each: a 2D point pattern
of tumour and immune points in an ~800 µm square. Each regime produces a
per-patient signal `s ∈ [0, 1]` (higher = better prognosis), and survival is
drawn from a Cox exponential model

```
λ_i = exp(−β · s_i + τ · ε_i),   ε_i ~ N(0, 1)
```

with `β = 5` and `τ = 0`. Event times are exponential with mean `60 / λ_i`
months; censoring times are drawn independently, targeting 40 % censoring.

| Regime | Name | Signal `s` |
|--------|------|------------|
| 1 | Composition | immune fraction, `Uniform(0.05, 0.60)`; uniform (Poisson) placement |
| 2 | Proximity | fraction `p` of immune cells placed near tumour cells; immune fraction fixed at 25 % |
| 3 | Anchored clustering | `1 − σ_thomas`: immune Thomas clusters around the 8 tumour cluster centres |
| 4 | Independent clustering | `1 − σ_thomas`: immune Thomas clusters at random locations |

The generators are in `processes.py`, and the survival model is in
`survival.py`.

## Output

`datasets/synthetic_regime{N}_noise{τ}.parquet`, with the same schema as the
metaclusters dataset (`x`, `y`, `cell_type` ∈ {`tumor`, `immune`}, `core`,
`PID`, `OSmonth`, `DFSmonth`, `Patientstatus`), plus the ground-truth `signal`
column, which the models never see. There are no clinical columns, so
synthetic runs never use `--clinical`.

## Usage

```bash
# Generate regimes 1–4 at τ = 0 (byte-identical to the thesis datasets)
uv run python synthetic/generate.py

# Synthetic validation table: NND, Ripley, NND+Ripley, GNN full, oracle
uv run python synthetic/run_synthetic.py
# ...and the GNN-only column
uv run python synthetic/run_synthetic.py --gnn_only
```

`run_synthetic.py` uses seeds 1–5. Each seed is one random 80/20 patient
split, and the script reports mean ± SD over the seeds. It generates any
missing datasets itself. Results are logged to the `synthetic-validation`
MLflow experiment.
