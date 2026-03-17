#!/usr/bin/env python3
"""Step 9: Concept-safety alignment analysis with leave-one-out concept selection.

Identifies which of the 33,732 Atlas concepts are associated with:
  (a) each hazard category (vs. all other cases)
  (b) correct vs. incorrect hazard detection (TP vs. FN)
  (c) correct vs. incorrect action recommendation

Uses leave-one-out cross-validation for concept selection: for each
case i, the top-K concepts are identified using all cases EXCEPT case i,
preventing circular concept-outcome correlations.

No GPU required; operates on pre-extracted concept activations.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import FDR_ALPHA, OUTPUT_DIR, SEED
from src.utils import benjamini_hochberg

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)


def load_base_data():
    """Load base results and concept weights, matched by index."""
    with open(OUTPUT_DIR / "steerling_base_results.json") as f:
        results = json.load(f)

    weights = np.load(OUTPUT_DIR / "base_concept_weights.npy")
    assert len(results) == weights.shape[0], (
        f"Mismatch: {len(results)} results vs {weights.shape[0]} weight rows"
    )
    return results, weights


def concept_hazard_association(results, weights, min_category_n=3):
    """For each hazard category, identify concepts with significantly
    higher activations vs. all other cases (Mann-Whitney U, BH-FDR).

    Uses ALL cases (no LOO here; LOO is applied in the steering target
    selection step). This provides the summary statistics for Table 3.
    """
    n_concepts = weights.shape[1]

    cat_indices = defaultdict(list)
    for i, r in enumerate(results):
        cat = r.get("hazard_category", "benign")
        if cat and cat != "benign" and cat != "Unknown":
            cat_indices[cat].append(i)

    hazard_idx = [i for i, r in enumerate(results) if r["detection_truth"] == 1]
    benign_idx = [i for i, r in enumerate(results) if r["detection_truth"] == 0]
    cat_indices["_any_hazard"] = hazard_idx

    alignment = {}
    all_rows = []

    for cat, pos_idx in cat_indices.items():
        if len(pos_idx) < min_category_n:
            continue

        neg_idx = benign_idx if cat == "_any_hazard" else [
            i for i in range(len(results)) if i not in pos_idx
        ]

        pos_w = weights[pos_idx]
        neg_w = weights[neg_idx]

        p_values = np.ones(n_concepts)
        effect_sizes = np.zeros(n_concepts)

        for c in range(n_concepts):
            pos_vals = pos_w[:, c]
            neg_vals = neg_w[:, c]
            if pos_vals.std() == 0 and neg_vals.std() == 0:
                continue
            pooled_n = len(pos_vals) + len(neg_vals) - 2
            if pooled_n > 0:
                s_pool = np.sqrt(
                    ((len(pos_vals) - 1) * pos_vals.std(ddof=1) ** 2
                     + (len(neg_vals) - 1) * neg_vals.std(ddof=1) ** 2)
                    / pooled_n
                )
                if s_pool > 0:
                    effect_sizes[c] = (pos_vals.mean() - neg_vals.mean()) / s_pool

            try:
                _, p = stats.mannwhitneyu(
                    pos_vals, neg_vals, alternative="two-sided"
                )
                p_values[c] = p
            except ValueError:
                pass

        fdr = benjamini_hochberg(p_values, alpha=FDR_ALPHA)
        n_sig = fdr["n_rejected"]

        sig_mask = fdr["rejected"]
        sig_concepts = []
        for c in range(n_concepts):
            all_rows.append({
                "hazard_category": cat,
                "concept_index": c,
                "effect_size": float(effect_sizes[c]),
                "p_value": float(p_values[c]),
                "q_value": float(fdr["q_values"][c]),
                "significant": bool(sig_mask[c]),
            })
            if sig_mask[c]:
                sig_concepts.append((c, effect_sizes[c], fdr["q_values"][c]))

        sig_concepts.sort(key=lambda x: abs(x[1]), reverse=True)
        alignment[cat] = {
            "n_positive": len(pos_idx),
            "n_negative": len(neg_idx),
            "n_significant": n_sig,
            "top_concepts": [
                {"index": c, "effect_size": float(d), "q_value": float(q)}
                for c, d, q in sig_concepts[:100]
            ],
        }

        print(f"  {cat}: {len(pos_idx)} cases, {n_sig} FDR-sig concepts")

    return alignment, pd.DataFrame(all_rows)


def detection_discrimination(results, weights):
    """Among true hazard cases, identify concepts that discriminate
    correct detection (TP) from missed detection (FN).
    """
    n_concepts = weights.shape[1]

    tp_idx = [
        i for i, r in enumerate(results)
        if r["detection_truth"] == 1 and r["steerling_detection"] == 1
    ]
    fn_idx = [
        i for i, r in enumerate(results)
        if r["detection_truth"] == 1 and r["steerling_detection"] == 0
    ]

    print(f"\n  TP cases: {len(tp_idx)}, FN cases: {len(fn_idx)}")

    if len(tp_idx) < 3 or len(fn_idx) < 3:
        print("  Too few cases for detection discrimination analysis")
        return None

    tp_w = weights[tp_idx]
    fn_w = weights[fn_idx]

    p_values = np.ones(n_concepts)
    effect_sizes = np.zeros(n_concepts)
    tp_means = np.zeros(n_concepts)
    fn_means = np.zeros(n_concepts)

    for c in range(n_concepts):
        tp_vals = tp_w[:, c]
        fn_vals = fn_w[:, c]
        tp_means[c] = tp_vals.mean()
        fn_means[c] = fn_vals.mean()

        if tp_vals.std() == 0 and fn_vals.std() == 0:
            continue
        pooled_n = len(tp_vals) + len(fn_vals) - 2
        if pooled_n > 0:
            s_pool = np.sqrt(
                ((len(tp_vals) - 1) * tp_vals.std(ddof=1) ** 2
                 + (len(fn_vals) - 1) * fn_vals.std(ddof=1) ** 2)
                / pooled_n
            )
            if s_pool > 0:
                effect_sizes[c] = (tp_vals.mean() - fn_vals.mean()) / s_pool

        try:
            _, p = stats.mannwhitneyu(tp_vals, fn_vals, alternative="two-sided")
            p_values[c] = p
        except ValueError:
            pass

    fdr = benjamini_hochberg(p_values, alpha=FDR_ALPHA)

    sig_concepts = []
    for c in range(n_concepts):
        if fdr["rejected"][c]:
            sig_concepts.append({
                "index": c,
                "effect_size": float(effect_sizes[c]),
                "q_value": float(fdr["q_values"][c]),
                "tp_mean_activation": float(tp_means[c]),
                "fn_mean_activation": float(fn_means[c]),
            })
    sig_concepts.sort(key=lambda x: abs(x["effect_size"]), reverse=True)

    print(f"  Detection-discriminating concepts (FDR): {len(sig_concepts)}")

    return {
        "n_tp": len(tp_idx),
        "n_fn": len(fn_idx),
        "n_significant": len(sig_concepts),
        "top_concepts": sig_concepts[:100],
    }


def compute_loo_steering_targets(results, weights, k=20):
    """Leave-one-out concept selection for each case.

    For case i, concept-hazard associations are computed using all
    cases EXCEPT case i. The top-K concepts from this LOO analysis
    are used as steering targets for case i.

    This prevents circular concept selection where the same case
    contributes to both concept identification and correction testing.
    """
    n_cases = len(results)
    n_concepts = weights.shape[1]

    hazard_idx_set = set(
        i for i, r in enumerate(results) if r["detection_truth"] == 1
    )
    benign_idx_set = set(
        i for i, r in enumerate(results) if r["detection_truth"] == 0
    )

    # Precompute per-category case indices
    cat_case_map = defaultdict(set)
    for i, r in enumerate(results):
        cat = r.get("hazard_category", "benign")
        if cat and cat != "benign" and cat != "Unknown":
            cat_case_map[cat].add(i)

    # Precompute global concept sums for efficient LOO
    # For the global "any hazard" analysis:
    hazard_sum = weights[list(hazard_idx_set)].sum(axis=0)
    hazard_sq_sum = (weights[list(hazard_idx_set)] ** 2).sum(axis=0)
    benign_sum = weights[list(benign_idx_set)].sum(axis=0)
    benign_sq_sum = (weights[list(benign_idx_set)] ** 2).sum(axis=0)

    targets = []
    random_concepts = sorted(
        np.random.choice(n_concepts, size=k, replace=False).tolist()
    )

    print(f"  Computing LOO targets for {n_cases} cases (K={k})...")

    for i in range(n_cases):
        r = results[i]
        cat = r.get("hazard_category", "benign")
        is_hazard = r["detection_truth"] == 1

        # LOO: remove case i from the appropriate group
        loo_hazard = hazard_idx_set - {i}
        loo_benign = benign_idx_set - {i}

        # Find category-specific concepts using LOO
        if cat in cat_case_map and len(cat_case_map[cat] - {i}) >= 3:
            loo_cat = cat_case_map[cat] - {i}
            loo_neg = set(range(n_cases)) - loo_cat - {i}

            pos_w = weights[list(loo_cat)]
            neg_w = weights[list(loo_neg)]

            # Fast ranking: use mean difference as proxy for effect size
            pos_mean = pos_w.mean(axis=0)
            neg_mean = neg_w.mean(axis=0)
            diff = np.abs(pos_mean - neg_mean)
            top_k_idx = np.argsort(diff)[-k:][::-1].tolist()
        else:
            # Fallback to global hazard concepts using LOO
            loo_h = list(loo_hazard)
            loo_b = list(loo_benign)
            if len(loo_h) >= 3 and len(loo_b) >= 3:
                h_mean = weights[loo_h].mean(axis=0)
                b_mean = weights[loo_b].mean(axis=0)
                diff = np.abs(h_mean - b_mean)
                top_k_idx = np.argsort(diff)[-k:][::-1].tolist()
            else:
                top_k_idx = random_concepts

        target = {
            "case_index": i,
            "case_id": r.get("case_id", r.get("name", "")),
            "dataset": r.get("dataset", "unknown"),
            "hazard_category": cat,
            "detection_truth": r["detection_truth"],
            "action_truth": r.get(
                "action_truth", r.get("ground_truth_action", "None")
            ),
            "baseline_detection": r["steerling_detection"],
            "baseline_action": r.get("steerling_action", "None"),
            "is_fn": is_hazard and r["steerling_detection"] == 0,
            "is_tp": is_hazard and r["steerling_detection"] == 1,
            "is_fp": not is_hazard and r["steerling_detection"] == 1,
            "is_tn": not is_hazard and r["steerling_detection"] == 0,
            "loo_concepts": {int(c): 1.0 for c in top_k_idx},
            "random_concepts": {int(c): 1.0 for c in random_concepts},
        }
        targets.append(target)

    n_fn = sum(1 for t in targets if t["is_fn"])
    n_tp = sum(1 for t in targets if t["is_tp"])
    n_fp = sum(1 for t in targets if t["is_fp"])
    n_tn = sum(1 for t in targets if t["is_tn"])
    print(f"  Steering targets: FN={n_fn}, TP={n_tp}, FP={n_fp}, TN={n_tn}")

    return targets


def main():
    print("Loading base data...")
    results, weights = load_base_data()
    n_cases, n_concepts = weights.shape
    print(f"  {n_cases} cases, {n_concepts} concepts")

    phys_idx = [i for i, r in enumerate(results) if r.get("dataset") == "physician"]
    rw_idx = [i for i, r in enumerate(results) if r.get("dataset") != "physician"]
    print(f"  Physician: {len(phys_idx)}, Real-world: {len(rw_idx)}")

    # A. Concept-hazard alignment (physician cases, for Table 3 reporting)
    print("\n--- Concept-Hazard Alignment (physician cases) ---")
    phys_results = [results[i] for i in phys_idx]
    phys_weights = weights[phys_idx]
    alignment, detail_df = concept_hazard_association(phys_results, phys_weights)

    with open(OUTPUT_DIR / "concept_hazard_alignment.json", "w") as f:
        json.dump(alignment, f, indent=2, default=str)
    detail_df.to_csv(OUTPUT_DIR / "concept_hazard_detail.csv", index=False)

    # B. Detection discrimination (physician cases, true hazards only)
    print("\n--- Detection Discrimination (physician hazard cases) ---")
    detection_disc = detection_discrimination(phys_results, phys_weights)
    if detection_disc:
        with open(OUTPUT_DIR / "detection_discrimination.json", "w") as f:
            json.dump(detection_disc, f, indent=2, default=str)

    # C. LOO steering targets for all 400 cases
    print("\n--- Computing LOO Steering Targets ---")
    targets = compute_loo_steering_targets(results, weights)
    with open(OUTPUT_DIR / "concept_correction_targets.json", "w") as f:
        json.dump(targets, f, indent=2, default=str)

    # Summary table
    print("\n--- Alignment Summary ---")
    summary_rows = []
    for cat in sorted(alignment.keys()):
        data = alignment[cat]
        cat_label = "Any hazard (global)" if cat.startswith("_") else cat
        top_d = data["top_concepts"][0]["effect_size"] if data["top_concepts"] else 0
        summary_rows.append({
            "Category": cat_label,
            "N cases": data["n_positive"],
            "FDR-sig concepts": data["n_significant"],
            "Top effect size (d)": f"{top_d:.3f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "concept_hazard_summary.csv", index=False)

    print(f"\nAll alignment results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
