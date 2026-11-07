"""
Node-concept correlation with the true synthetic signal s.

Standalone: rebuilds the node-level concept assignments exactly as
concept_discovery.run_node_concept_discovery does (same embeddings, node
pooling order, subsample, K search and K-means seed), without modifying the
pipeline, then for each held-out patient computes the fraction of its nodes
assigned to concept 0 and correlates it with s.

Reports |Spearman rho| with a 95% bootstrap CI over the held-out patients
(same split, n_boot=500 and rng seed 0 as the embedding probe). The node
completeness R² is recomputed as an identity check: it should match the
`node_concepts_completeness` value logged by run_explainability.py.

Takes the same CLI arguments as run_explainability.py, e.g.
    uv run python analysis/explainability/node_concept_signal.py \\
        --model_path models/synthetic_regime3_noise0.0.pt \\
        --data datasets/synthetic_regime3_noise0.0.parquet \\
        --endpoint DFS --synthetic --regime 3
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_utils import create_pytorch_dataset, patient_split
from run_explainability import parse_args, load_model, get_device
from concept_discovery import extract_all_embeddings, search_k, _completeness_r2


def main():
    args = parse_args()
    device = get_device(args.device)

    df = pd.read_parquet(args.data)
    if "signal" not in df.columns:
        sys.exit("No 'signal' column: synthetic data only.")
    sig_map = df.groupby("PID")["signal"].first().to_dict()

    dataset = create_pytorch_dataset(df, endpoint=args.endpoint, k=args.k,
                                     max_cells=args.max_cells, normalize_coords=True)
    train_idx, val_idx = patient_split(dataset, val_size=0.2, random_state=args.seed)
    train_pids = {dataset.data[dataset.cores[i]]["PID"] for i in train_idx}
    val_pids   = {dataset.data[dataset.cores[i]]["PID"] for i in val_idx}
    model = load_model(args, dataset, device)

    # ── Node assignments, mirroring run_node_concept_discovery ────────────────
    emb_data = extract_all_embeddings(model, dataset, device)
    cores = list(emb_data["node_emb"])
    pids  = [emb_data["pid"][c] for c in cores]
    risks = {c: emb_data["risk"][c] for c in cores}

    all_node_emb  = np.concatenate([emb_data["node_emb"][c] for c in cores], axis=0)
    all_node_core = np.array([c for c in cores for _ in range(len(emb_data["node_emb"][c]))])

    max_nodes_total, n_init = 50_000, 10
    if len(all_node_emb) > max_nodes_total:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(all_node_emb), max_nodes_total, replace=False))
    else:
        idx = np.arange(len(all_node_emb))
    X_km = all_node_emb[idx]

    best_k, _, _ = search_k(X_km, k_range=(args.node_k_min, args.node_k_max), n_init=n_init)
    km = KMeans(n_clusters=best_k, n_init=n_init, random_state=0).fit(X_km)
    labels_all = km.predict(all_node_emb)

    # ── Identity check: node completeness (concept dist → risk, val cores) ────
    dist = {}
    for c in cores:
        counts = np.bincount(labels_all[all_node_core == c], minlength=best_k).astype(float)
        dist[c] = counts / counts.sum()
    tr = [c for c, p in zip(cores, pids) if p in train_pids]
    va = [c for c, p in zip(cores, pids) if p in val_pids]
    r2 = _completeness_r2(np.array([dist[c] for c in tr]), np.array([risks[c] for c in tr]),
                          np.array([dist[c] for c in va]), np.array([risks[c] for c in va]))

    # ── Per held-out patient: fraction of nodes in concept 0 vs s ─────────────
    core_pid = dict(zip(cores, pids))
    node_pid = np.array([core_pid[c] for c in all_node_core])
    val_list = sorted(p for p in set(pids) if p in val_pids)
    frac0 = np.array([(labels_all[node_pid == p] == 0).mean() for p in val_list])
    s     = np.array([sig_map[p] for p in val_list], dtype=float)

    rho, p_val = stats.spearmanr(frac0, s)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(500):
        b = rng.integers(0, len(s), size=len(s))
        boot.append(abs(stats.spearmanr(frac0[b], s[b])[0]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    print(f"\nRegime {args.regime}: K={best_k}, n_val={len(s)}")
    print(f"  Node completeness R² (identity check): {r2:.4f}")
    print(f"  Spearman rho(frac concept 0, s) = {rho:+.3f}  (p = {p_val:.2g})")
    print(f"  |rho| = {abs(rho):.3f}  95% bootstrap CI [{lo:.3f}, {hi:.3f}]")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"node_concept_signal_{args.endpoint.lower()}"
                                        f"_synthetic_r{args.regime}.csv")
    pd.DataFrame([{"regime": args.regime, "k": best_k, "n_val": len(s),
                   "node_completeness": r2, "spearman_rho": rho, "p": p_val,
                   "abs_rho": abs(rho), "ci_low": lo, "ci_high": hi}]).to_csv(out, index=False)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
