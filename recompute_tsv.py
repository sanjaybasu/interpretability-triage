#!/usr/bin/env python3
"""Recompute TSV with corrected TP/FN labels.

The original tsv_analysis.json used the wrong field name for model predictions
('detection' instead of 'gemma2_detection'), causing n_TP=0 and degenerate TSV.
This script recomputes with correct labels (65 TP, 79 FN).
"""

import json
import math
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

OUTPUT_DIR = Path(__file__).parent / "output"
SEED = 42
np.random.seed(SEED)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (center, max(0, center - margin), min(1, center + margin))


def main():
    # Load base results with correct field name
    with open(OUTPUT_DIR / "gemma2_base_results.json") as f:
        base_results = json.load(f)

    # Load hidden states
    hidden_states = torch.load(OUTPUT_DIR / "gemma2_hidden_states.pt",
                                map_location="cpu", weights_only=True)
    print(f"Hidden states shape: {hidden_states.shape}")
    # Expected: [400, 28, 3584]

    n_cases, n_layers, hidden_dim = hidden_states.shape
    assert n_cases == 400

    # Extract correct labels
    ground_truth = np.array([r.get("detection_truth", 0) for r in base_results])
    predictions = np.array([r.get("gemma2_detection", 0) for r in base_results])

    tp_mask = (ground_truth == 1) & (predictions == 1)
    fn_mask = (ground_truth == 1) & (predictions == 0)
    hazard_mask = ground_truth == 1

    n_tp = tp_mask.sum()
    n_fn = fn_mask.sum()
    print(f"TP: {n_tp}, FN: {n_fn}, Total hazard: {hazard_mask.sum()}")

    # Use probe best layer (23)
    best_layer = 23

    # Extract hidden states at best layer
    H = hidden_states[:, best_layer, :].float().numpy()
    print(f"Hidden states at layer {best_layer}: shape {H.shape}")

    # Re-verify probe AUROC at best layer
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_preds = np.zeros(n_cases)
    for train_idx, test_idx in skf.split(H, ground_truth):
        clf = LogisticRegression(C=1.0, solver="saga", max_iter=2000,
                                  random_state=SEED)
        clf.fit(H[train_idx], ground_truth[train_idx])
        cv_preds[test_idx] = clf.predict_proba(H[test_idx])[:, 1]
    probe_auroc = roc_auc_score(ground_truth, cv_preds)
    print(f"Probe AUROC at layer {best_layer}: {probe_auroc:.4f}")

    # Compute TSV: mean_TP - mean_FN (among hazard cases only)
    H_tp = H[tp_mask]
    H_fn = H[fn_mask]
    print(f"H_tp shape: {H_tp.shape}, H_fn shape: {H_fn.shape}")

    mean_tp = H_tp.mean(axis=0)
    mean_fn = H_fn.mean(axis=0)
    tsv_raw = mean_tp - mean_fn
    tsv_norm = np.linalg.norm(tsv_raw)
    tsv_unit = tsv_raw / tsv_norm if tsv_norm > 0 else tsv_raw

    print(f"TSV raw norm: {tsv_norm:.4f}")

    # Also compute detection direction (det): mean_hazard - mean_benign
    H_hazard = H[hazard_mask]
    H_benign = H[~hazard_mask]
    det_raw = H_hazard.mean(axis=0) - H_benign.mean(axis=0)
    det_norm = np.linalg.norm(det_raw)
    det_unit = det_raw / det_norm if det_norm > 0 else det_raw

    # Cosine similarity between TSV and detection direction
    cosine_tsv_det = float(np.dot(tsv_unit, det_unit))
    print(f"Cosine(TSV, detection): {cosine_tsv_det:.4f}")

    # TSV AUROC: can TSV projection separate TP from FN?
    H_hazard_only = H[hazard_mask]
    labels_hazard = predictions[hazard_mask]  # 1=TP, 0=FN
    tsv_projections = H_hazard_only @ tsv_unit
    tsv_auroc = roc_auc_score(labels_hazard, tsv_projections)
    print(f"TSV AUROC (TP vs FN): {tsv_auroc:.4f}")

    # Bootstrap CI for TSV AUROC
    rng = np.random.RandomState(SEED)
    n_boot = 1000
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(len(labels_hazard), size=len(labels_hazard), replace=True)
        if len(np.unique(labels_hazard[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(labels_hazard[idx], tsv_projections[idx]))
    tsv_auroc_ci = (float(np.percentile(boot_aurocs, 2.5)),
                     float(np.percentile(boot_aurocs, 97.5)))
    print(f"TSV AUROC 95% CI: ({tsv_auroc_ci[0]:.4f}, {tsv_auroc_ci[1]:.4f})")

    # Test TSV steering effectiveness (simulation: project FN hidden states
    # along TSV direction and check if they cross into TP-like space)
    fn_projections = H_fn @ tsv_unit
    tp_projections = H_tp @ tsv_unit
    mean_fn_proj = fn_projections.mean()
    mean_tp_proj = tp_projections.mean()
    std_fn_proj = fn_projections.std()
    std_tp_proj = tp_projections.std()
    tsv_cohens_d = (mean_tp_proj - mean_fn_proj) / math.sqrt(
        ((n_tp - 1) * std_tp_proj**2 + (n_fn - 1) * std_fn_proj**2) /
        (n_tp + n_fn - 2)
    ) if (n_tp + n_fn - 2) > 0 else 0

    print(f"TSV projection: TP mean={mean_tp_proj:.4f}, FN mean={mean_fn_proj:.4f}")
    print(f"TSV Cohen's d (TP vs FN projection): {tsv_cohens_d:.4f}")

    # For each alpha, simulate how many FN would cross the TP decision boundary
    # (defined as the midpoint between TP and FN means in TSV projection space)
    boundary = (mean_tp_proj + mean_fn_proj) / 2
    print(f"\nSimulated TSV steering (boundary={boundary:.4f}):")
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0]:
        shifted_fn = fn_projections + alpha * tsv_norm
        n_cross = (shifted_fn >= boundary).sum()
        print(f"  alpha={alpha:5.1f}: {n_cross}/{n_fn} FN cross boundary "
              f"({n_cross/n_fn*100:.1f}%)")

    # Save corrected results
    results = {
        "best_layer": best_layer,
        "n_tp": int(n_tp),
        "n_fn": int(n_fn),
        "n_hazard": int(hazard_mask.sum()),
        "n_benign": int((~hazard_mask).sum()),
        "probe_auroc_recomputed": float(probe_auroc),
        "tsv_raw_norm": float(tsv_norm),
        "det_raw_norm": float(det_norm),
        "cosine_similarity_tsv_det": float(cosine_tsv_det),
        "tsv_auroc_tp_vs_fn": float(tsv_auroc),
        "tsv_auroc_ci_lower": float(tsv_auroc_ci[0]),
        "tsv_auroc_ci_upper": float(tsv_auroc_ci[1]),
        "tsv_cohens_d": float(tsv_cohens_d),
        "tp_mean_projection": float(mean_tp_proj),
        "fn_mean_projection": float(mean_fn_proj),
        "steering_simulation": {
            f"alpha_{alpha}": {
                "n_fn_cross": int((fn_projections + alpha * tsv_norm >= boundary).sum()),
                "n_fn_total": int(n_fn),
                "cross_rate": float(
                    (fn_projections + alpha * tsv_norm >= boundary).sum() / n_fn
                ),
            }
            for alpha in [1.0, 2.0, 5.0, 10.0, 20.0]
        },
    }

    out_path = OUTPUT_DIR / "tsv_analysis_corrected.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
