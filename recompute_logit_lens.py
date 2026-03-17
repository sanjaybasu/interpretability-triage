#!/usr/bin/env python3
"""Recompute logit lens summary with corrected TP/FN labels.

The original logit_lens_summary.json used the wrong field name for model
predictions ('detection' instead of 'gemma2_detection'), causing all cases
to be classified as FN (n_TP=0). This script recomputes the summary using
the correct field, producing TP and FN mean ranks per layer, Cohen's d,
and critical layer identification.
"""

import json
import math
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"


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

    # Load per-case logit lens data
    with open(OUTPUT_DIR / "logit_lens_results.json") as f:
        logit_data = json.load(f)

    assert len(base_results) == len(logit_data) == 400

    # Classify cases using correct field
    tp_indices = []
    fn_indices = []
    fp_indices = []
    tn_indices = []

    for i, r in enumerate(base_results):
        gt = r.get("detection_truth", 0)
        pred = r.get("gemma2_detection", 0)
        if gt == 1 and pred == 1:
            tp_indices.append(i)
        elif gt == 1 and pred == 0:
            fn_indices.append(i)
        elif gt == 0 and pred == 1:
            fp_indices.append(i)
        else:
            tn_indices.append(i)

    print(f"TP: {len(tp_indices)}, FN: {len(fn_indices)}, "
          f"FP: {len(fp_indices)}, TN: {len(tn_indices)}")
    print(f"Total hazard: {len(tp_indices) + len(fn_indices)}")
    print(f"Sensitivity: {len(tp_indices)/(len(tp_indices)+len(fn_indices)):.3f}")

    n_layers = 28
    n_vocab = 152064  # Qwen 2.5 7B vocab size

    # Compute per-layer mean ranks for TP and FN
    def compute_group_stats(indices, label):
        if not indices:
            return {}
        layer_stats = {}
        for layer in range(n_layers):
            ranks = []
            top100 = 0
            top50 = 0
            top10 = 0
            for idx in indices:
                case_data = logit_data[idx]
                layer_key = str(layer)
                if layer_key not in case_data:
                    continue
                layer_ranks = case_data[layer_key]
                # Mean rank across all hazard token IDs
                token_ranks = list(layer_ranks.values())
                if token_ranks:
                    min_rank = min(token_ranks)
                    mean_rank = np.mean(token_ranks)
                    ranks.append(mean_rank)
                    if min_rank <= 100:
                        top100 += 1
                    if min_rank <= 50:
                        top50 += 1
                    if min_rank <= 10:
                        top10 += 1

            if ranks:
                layer_stats[layer] = {
                    "mean_rank": float(np.mean(ranks)),
                    "std_rank": float(np.std(ranks)),
                    "median_rank": float(np.median(ranks)),
                    "n_cases": len(ranks),
                    "top100_count": top100,
                    "top50_count": top50,
                    "top10_count": top10,
                }
        return layer_stats

    tp_stats = compute_group_stats(tp_indices, "TP")
    fn_stats = compute_group_stats(fn_indices, "FN")

    # Compute Cohen's d at each layer (TP vs FN mean rank)
    cohens_d = {}
    for layer in range(n_layers):
        if layer in tp_stats and layer in fn_stats:
            n_tp = tp_stats[layer]["n_cases"]
            n_fn = fn_stats[layer]["n_cases"]
            mean_tp = tp_stats[layer]["mean_rank"]
            mean_fn = fn_stats[layer]["mean_rank"]
            std_tp = tp_stats[layer]["std_rank"]
            std_fn = fn_stats[layer]["std_rank"]
            # Pooled SD
            pooled_sd = math.sqrt(
                ((n_tp - 1) * std_tp**2 + (n_fn - 1) * std_fn**2) /
                (n_tp + n_fn - 2)
            ) if (n_tp + n_fn - 2) > 0 else 1.0
            d = (mean_fn - mean_tp) / pooled_sd if pooled_sd > 0 else 0
            cohens_d[layer] = {
                "d": float(d),
                "mean_tp": float(mean_tp),
                "mean_fn": float(mean_fn),
                "pooled_sd": float(pooled_sd),
            }

    # Identify critical layers (largest |Cohen's d|)
    if cohens_d:
        sorted_layers = sorted(cohens_d.items(), key=lambda x: abs(x[1]["d"]),
                                reverse=True)
        critical_layers = [
            {"layer": layer, **stats}
            for layer, stats in sorted_layers[:5]
        ]
    else:
        critical_layers = []

    # Build summary
    summary = {
        "n_tp": len(tp_indices),
        "n_fn": len(fn_indices),
        "n_fp": len(fp_indices),
        "n_tn": len(tn_indices),
        "n_hazard": len(tp_indices) + len(fn_indices),
        "n_benign": len(fp_indices) + len(tn_indices),
        "sensitivity": len(tp_indices) / (len(tp_indices) + len(fn_indices)),
        "tp": {},
        "fn": {},
        "cohens_d": {},
        "critical_layers": critical_layers,
        "first_top100_layer": {},
        "first_top50_layer": {},
        "first_top10_layer": {},
    }

    # Format TP and FN stats
    for layer in range(n_layers):
        if layer in tp_stats:
            summary["tp"][str(layer)] = tp_stats[layer]
        if layer in fn_stats:
            summary["fn"][str(layer)] = fn_stats[layer]
        if layer in cohens_d:
            summary["cohens_d"][str(layer)] = cohens_d[layer]

    # Find first layer where any case reaches top-K
    for group_name, group_stats, group_indices in [
        ("tp", tp_stats, tp_indices), ("fn", fn_stats, fn_indices)
    ]:
        for threshold, key in [(100, "first_top100_layer"),
                                (50, "first_top50_layer"),
                                (10, "first_top10_layer")]:
            count_key = f"top{threshold}_count"
            entered = 0
            first_layer = None
            for layer in range(n_layers):
                if layer in group_stats and group_stats[layer][count_key] > 0:
                    entered = group_stats[layer][count_key]
                    if first_layer is None:
                        first_layer = layer
            summary[key][group_name] = {
                "first_layer": first_layer,
                "n_entered": entered,
                "n_total": len(group_indices),
            }

    # Write corrected summary
    out_path = OUTPUT_DIR / "logit_lens_summary_corrected.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")

    # Print key results
    print(f"\n{'='*60}")
    print("CORRECTED LOGIT LENS SUMMARY")
    print(f"{'='*60}")
    print(f"TP cases: {len(tp_indices)}, FN cases: {len(fn_indices)}")
    print()

    print("Layer | TP mean rank | FN mean rank | Cohen's d | TP top-100 | FN top-100")
    print("-" * 85)
    for layer in range(n_layers):
        tp_mr = tp_stats.get(layer, {}).get("mean_rank", float("nan"))
        fn_mr = fn_stats.get(layer, {}).get("mean_rank", float("nan"))
        d = cohens_d.get(layer, {}).get("d", float("nan"))
        tp_t100 = tp_stats.get(layer, {}).get("top100_count", 0)
        fn_t100 = fn_stats.get(layer, {}).get("top100_count", 0)
        print(f"  {layer:2d}  | {tp_mr:12.1f} | {fn_mr:12.1f} | {d:9.3f} | "
              f"{tp_t100:10d} | {fn_t100:10d}")

    print(f"\nCritical layers (by |Cohen's d|):")
    for cl in critical_layers:
        print(f"  Layer {cl['layer']}: d={cl['d']:.3f} "
              f"(TP mean={cl['mean_tp']:.0f}, FN mean={cl['mean_fn']:.0f})")

    # Also write corrected critical_layers.json
    cl_path = OUTPUT_DIR / "critical_layers_corrected.json"
    with open(cl_path, "w") as f:
        json.dump(critical_layers, f, indent=2)
    print(f"\nWrote {cl_path}")

    # Compute corrected TSV partition stats (for documentation)
    print(f"\nCorrected TSV partition: n_TP={len(tp_indices)}, n_FN={len(fn_indices)}")
    print("TSV can now be computed with corrected code (requires hidden states on Modal)")


if __name__ == "__main__":
    main()
