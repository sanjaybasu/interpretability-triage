#!/usr/bin/env python3
"""Step 12: Concept activation distribution analysis.

Computes summary statistics about concept activations in TP vs FN cases
to establish that:
  (a) Natural concept activations are extremely sparse (most near zero)
  (b) Steered concept activations for TP vs FN cases are nearly identical
  (c) Alpha=1.0 is far outside the observed distribution
  (d) TP-mean correction targets are trivially close to FN baselines

This analysis supports the argument that concept activations encode hazard
information observationally but that the TP-FN gap is too small for
concept-level correction to be meaningful.
"""

import json
import sys
from pathlib import Path

import numpy as np

from config import OUTPUT_DIR, SEED

np.random.seed(SEED)


def main():
    print("Loading data...")
    with open(OUTPUT_DIR / "steerling_base_results.json") as f:
        results = json.load(f)
    weights = np.load(OUTPUT_DIR / "base_concept_weights.npy")
    with open(OUTPUT_DIR / "concept_correction_targets.json") as f:
        targets = json.load(f)

    n_cases, n_concepts = weights.shape
    print(f"  {n_cases} cases, {n_concepts} concepts")

    # A. Global activation distribution
    print("\n--- A. Global Concept Activation Distribution ---")
    all_vals = weights.ravel()
    print(f"  Total values: {len(all_vals):,}")
    print(f"  Min: {all_vals.min():.6f}")
    print(f"  Max: {all_vals.max():.6f}")
    print(f"  Mean: {all_vals.mean():.6f}")
    print(f"  Median: {np.median(all_vals):.6f}")
    print(f"  Std: {all_vals.std():.6f}")
    print(f"  Fraction = 0: {(all_vals == 0).mean():.4f}")
    print(f"  Fraction < 0.01: {(all_vals < 0.01).mean():.4f}")
    print(f"  Fraction < 0.05: {(all_vals < 0.05).mean():.4f}")
    print(f"  Fraction < 0.10: {(all_vals < 0.10).mean():.4f}")
    for p in [50, 75, 90, 95, 99, 99.9]:
        val = np.percentile(all_vals, p)
        print(f"  P{p}: {val:.6f}")

    # B. TP vs FN activation comparison
    print("\n--- B. TP vs FN Concept Activations ---")
    tp_idx = [i for i, r in enumerate(results) if r["detection_truth"] == 1 and r["steerling_detection"] == 1]
    fn_idx = [i for i, r in enumerate(results) if r["detection_truth"] == 1 and r["steerling_detection"] == 0]
    print(f"  TP cases: {len(tp_idx)}, FN cases: {len(fn_idx)}")

    tp_w = weights[tp_idx]
    fn_w = weights[fn_idx]

    tp_means = tp_w.mean(axis=0)
    fn_means = fn_w.mean(axis=0)

    print(f"\n  Per-concept TP mean: overall mean={tp_means.mean():.6f}, max={tp_means.max():.6f}")
    print(f"  Per-concept FN mean: overall mean={fn_means.mean():.6f}, max={fn_means.max():.6f}")

    diff = tp_means - fn_means
    abs_diff = np.abs(diff)
    print(f"\n  TP-FN difference: mean={diff.mean():.6f}, median={np.median(diff):.6f}")
    print(f"  |TP-FN| difference: mean={abs_diff.mean():.6f}, max={abs_diff.max():.6f}")
    print(f"  Fraction |diff| < 0.001: {(abs_diff < 0.001).mean():.4f}")
    print(f"  Fraction |diff| < 0.01: {(abs_diff < 0.01).mean():.4f}")
    print(f"  Fraction |diff| < 0.05: {(abs_diff < 0.05).mean():.4f}")

    # C. Steered concept activations (the top-20 LOO-selected concepts per case)
    print("\n--- C. Steered Concept Activations (LOO top-20) ---")
    steered_tp_vals = []
    steered_fn_vals = []
    steered_tp_mean_targets = []
    steered_obs_max_targets = []

    for t in targets:
        idx = t["global_index"] if "global_index" in t else t["case_index"]
        loo_concepts = t.get("loo_concepts", {})
        concept_ids = [int(c) for c in list(loo_concepts.keys())[:20]]

        if not concept_ids:
            continue

        case_vals = weights[idx, concept_ids]

        if t["is_tp"]:
            steered_tp_vals.extend(case_vals.tolist())
        elif t["is_fn"]:
            steered_fn_vals.extend(case_vals.tolist())

        tp_mean = t.get("tp_mean_concepts", {})
        obs_max = t.get("observed_max_concepts", {})
        steered_tp_mean_targets.extend([float(v) for v in tp_mean.values()])
        steered_obs_max_targets.extend([float(v) for v in obs_max.values()])

    steered_tp_vals = np.array(steered_tp_vals) if steered_tp_vals else np.array([])
    steered_fn_vals = np.array(steered_fn_vals) if steered_fn_vals else np.array([])

    if len(steered_tp_vals) > 0 and len(steered_fn_vals) > 0:
        print(f"  TP steered concept values (n={len(steered_tp_vals)}): mean={steered_tp_vals.mean():.6f}, median={np.median(steered_tp_vals):.6f}, max={steered_tp_vals.max():.6f}")
        print(f"  FN steered concept values (n={len(steered_fn_vals)}): mean={steered_fn_vals.mean():.6f}, median={np.median(steered_fn_vals):.6f}, max={steered_fn_vals.max():.6f}")
        print(f"  TP-FN gap in steered concepts: {steered_tp_vals.mean() - steered_fn_vals.mean():.6f}")

    # D. TP-mean correction targets
    print("\n--- D. TP-Mean Correction Targets ---")
    if steered_tp_mean_targets:
        tp_mean_arr = np.array(steered_tp_mean_targets)
        print(f"  TP-mean target values (n={len(tp_mean_arr)}): mean={tp_mean_arr.mean():.6f}, median={np.median(tp_mean_arr):.6f}, max={tp_mean_arr.max():.6f}")
    if steered_obs_max_targets:
        obs_max_arr = np.array(steered_obs_max_targets)
        print(f"  Observed-max (P95) target values (n={len(obs_max_arr)}): mean={obs_max_arr.mean():.6f}, median={np.median(obs_max_arr):.6f}, max={obs_max_arr.max():.6f}")

    # E. Key comparisons for manuscript
    print("\n--- E. Key Comparisons ---")
    obs_max_global = weights.max()
    print(f"  Max observed activation (any concept, any case): {obs_max_global:.6f}")
    print(f"  Alpha=1.0 is {1.0 / obs_max_global:.1f}x the max observed activation")

    if len(steered_tp_vals) > 0:
        steered_max = max(steered_tp_vals.max(), steered_fn_vals.max()) if len(steered_fn_vals) > 0 else steered_tp_vals.max()
        print(f"  Max observed activation in steered concepts: {steered_max:.6f}")
        print(f"  Alpha=1.0 is {1.0 / steered_max:.1f}x the max steered concept activation")

    if len(steered_tp_vals) > 0 and len(steered_fn_vals) > 0:
        gap = steered_tp_vals.mean() - steered_fn_vals.mean()
        print(f"\n  Mean TP-FN gap in steered concepts: {gap:.6f}")
        print(f"  This gap is {abs(gap) / 1.0 * 100:.2f}% of alpha=1.0")
        if steered_tp_mean_targets:
            tp_mean_arr = np.array(steered_tp_mean_targets)
            print(f"  TP-mean correction would shift FN concepts by ~{gap:.6f} on average")
            print(f"  (equivalent to setting to {steered_fn_vals.mean() + gap:.6f} from {steered_fn_vals.mean():.6f})")

    # Save summary
    summary = {
        "global_activation": {
            "n_cases": int(n_cases),
            "n_concepts": int(n_concepts),
            "min": float(all_vals.min()),
            "max": float(all_vals.max()),
            "mean": float(all_vals.mean()),
            "median": float(np.median(all_vals)),
            "std": float(all_vals.std()),
            "frac_zero": float((all_vals == 0).mean()),
            "frac_lt_001": float((all_vals < 0.01).mean()),
            "p99": float(np.percentile(all_vals, 99)),
            "p999": float(np.percentile(all_vals, 99.9)),
        },
        "tp_fn_comparison": {
            "n_tp": len(tp_idx),
            "n_fn": len(fn_idx),
            "tp_concept_mean": float(tp_means.mean()),
            "fn_concept_mean": float(fn_means.mean()),
            "mean_abs_diff": float(abs_diff.mean()),
            "max_abs_diff": float(abs_diff.max()),
            "frac_diff_lt_001": float((abs_diff < 0.001).mean()),
        },
        "steered_concepts": {
            "tp_mean": float(steered_tp_vals.mean()) if len(steered_tp_vals) > 0 else None,
            "fn_mean": float(steered_fn_vals.mean()) if len(steered_fn_vals) > 0 else None,
            "tp_fn_gap": float(steered_tp_vals.mean() - steered_fn_vals.mean()) if len(steered_tp_vals) > 0 and len(steered_fn_vals) > 0 else None,
        },
        "alpha_comparison": {
            "max_observed": float(obs_max_global),
            "alpha_1_over_max_ratio": float(1.0 / obs_max_global),
        },
    }

    out_path = OUTPUT_DIR / "concept_distribution_analysis.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
