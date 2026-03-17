#!/usr/bin/env python3
"""Step 3b: Reanalysis on proper 33,732-d concept activation weights.

Uses the sigmoid concept weights computed from hidden states in step 2b,
operating in the correct concept index space (0-33,731) rather than the
4,096-d hidden feature space used in step 03.

Analyses:
  (A) Effective number of independent tests (eigenvalue-based)
  (B) Differential activation by demographic group (paired t-tests, BH-FDR)
  (C) Permutation test with corrected framing (excess over null)
  (D) Cross-validated L1 logistic regression on concept weights
  (E) PCA of concept weight space
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from config import (
    C_VALUES,
    CI_LEVEL,
    CV_FOLDS,
    FDR_ALPHA,
    OUTPUT_DIR,
    PERMUTATION_N,
    SEED,
)
from src.utils import benjamini_hochberg, detection_metrics

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

N_REAL_CONCEPTS = 33732


def load_concept_weights():
    """Load proper concept weight arrays from step 2b."""
    base = np.load(OUTPUT_DIR / "base_concept_weights.npy")
    demo = np.load(OUTPUT_DIR / "demo_concept_weights.npy")
    with open(OUTPUT_DIR / "demo_concept_weights_meta.json") as f:
        meta = json.load(f)
    return base, demo, meta


def effective_n_tests(data, threshold=0.95):
    """Compute effective number of independent tests via eigenvalue analysis.

    Uses the method of Li & Ji (2005, Heredity): effective N equals the
    number of eigenvalues needed to explain `threshold` fraction of variance,
    plus a fractional correction for the partial eigenvalue.
    """
    # Center the data
    centered = data - data.mean(axis=0)
    # Compute covariance eigenvalues (use SVD for efficiency)
    # For n_samples < n_features, compute n_samples eigenvalues
    n_samples, n_features = centered.shape
    if n_samples < n_features:
        gram = centered @ centered.T / (n_samples - 1)
        eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    else:
        cov = centered.T @ centered / (n_samples - 1)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]

    eigenvalues = eigenvalues[eigenvalues > 0]
    total_var = eigenvalues.sum()
    cumvar = np.cumsum(eigenvalues) / total_var

    # Number of eigenvalues for threshold
    n_eff_threshold = np.searchsorted(cumvar, threshold) + 1

    # Li & Ji method: sum of floor(eigenvalue/max_eigenvalue) indicators
    # Modified: count eigenvalues > 1 (after dividing by mean)
    scaled = eigenvalues / eigenvalues.mean()
    n_eff_lj = sum(1 for e in scaled if e > 1.0)

    return {
        "n_eigenvalues_positive": len(eigenvalues),
        f"n_eff_{int(threshold*100)}pct_variance": int(n_eff_threshold),
        "n_eff_li_ji": int(n_eff_lj),
        "top_eigenvalue_pct": float(eigenvalues[0] / total_var),
        "top5_eigenvalues_pct": float(cumvar[min(4, len(cumvar)-1)]),
    }


def differential_activation_analysis(demo_weights, meta):
    """Paired t-tests for each concept × variation against White reference."""
    variations = sorted(set(m["variation"] for m in meta if m["variation"] != "race_white"))
    case_ids = sorted(set(m["case_id"] for m in meta))

    # Build lookup: (case_id, variation) → row index
    lookup = {}
    for i, m in enumerate(meta):
        lookup[(m["case_id"], m["variation"])] = i

    results = []
    for variation in variations:
        paired_diffs = []
        paired_case_ids = []
        for cid in case_ids:
            white_idx = lookup.get((cid, "race_white"))
            var_idx = lookup.get((cid, variation))
            if white_idx is not None and var_idx is not None:
                diff = demo_weights[var_idx] - demo_weights[white_idx]  # (33732,)
                paired_diffs.append(diff)
                paired_case_ids.append(cid)

        if not paired_diffs:
            continue

        diffs = np.stack(paired_diffs)  # (N_pairs, 33732)
        n_pairs = diffs.shape[0]
        print(f"  {variation}: {n_pairs} paired vignettes")

        p_values = []
        for j in range(N_REAL_CONCEPTS):
            d = diffs[:, j]
            if np.std(d) < 1e-12:
                p_values.append(1.0)
            else:
                _, p = stats.ttest_rel(
                    [demo_weights[lookup[(cid, variation)]][j] for cid in paired_case_ids],
                    [demo_weights[lookup[(cid, "race_white")]][j] for cid in paired_case_ids],
                )
                p_values.append(p)

        p_arr = np.array(p_values)
        bh = benjamini_hochberg(p_arr)
        q_arr = bh["q_values"]

        # Cohen's d for each concept
        for j in range(N_REAL_CONCEPTS):
            d_col = diffs[:, j]
            mean_diff = d_col.mean()
            sd_diff = d_col.std(ddof=1) if d_col.std() > 0 else 1e-12
            cohens_d = mean_diff / sd_diff

            results.append({
                "concept_index": j,
                "dimension": "race",
                "variation": variation,
                "mean_diff": float(mean_diff),
                "cohens_d": float(cohens_d),
                "t_stat": float(mean_diff / (sd_diff / np.sqrt(n_pairs))) if sd_diff > 0 else 0,
                "p_value": float(p_arr[j]),
                "q_value_fdr": float(q_arr[j]),
                "p_bonferroni": float(min(p_arr[j] * N_REAL_CONCEPTS * len(variations), 1.0)),
                "significant_fdr": bool(q_arr[j] < FDR_ALPHA),
                "significant_bonf": bool(min(p_arr[j] * N_REAL_CONCEPTS * len(variations), 1.0) < 0.05),
            })

    return pd.DataFrame(results)


def _vectorized_fdr_count(diffs, alpha=FDR_ALPHA):
    """Vectorized: count FDR-significant concepts from paired differences.

    diffs: (n_pairs, n_concepts)
    Returns: number of FDR-significant concepts.
    """
    n = diffs.shape[0]
    mean_d = diffs.mean(axis=0)
    std_d = diffs.std(axis=0, ddof=1)
    # Avoid division by zero
    valid = std_d > 1e-12
    t_stat = np.zeros_like(mean_d)
    t_stat[valid] = mean_d[valid] / (std_d[valid] / np.sqrt(n))
    # Two-sided p-value from t-distribution
    p_values = np.ones_like(mean_d)
    p_values[valid] = 2.0 * stats.t.sf(np.abs(t_stat[valid]), df=n-1)
    bh = benjamini_hochberg(p_values, alpha)
    return int(bh["n_rejected"])


def permutation_test(demo_weights, meta, n_perms=PERMUTATION_N):
    """Permutation test: shuffle demographic labels within each vignette.
    Uses vectorized t-tests for speed.
    """
    variations = sorted(set(m["variation"] for m in meta if m["variation"] != "race_white"))
    case_ids = sorted(set(m["case_id"] for m in meta))

    lookup = {}
    for i, m in enumerate(meta):
        lookup[(m["case_id"], m["variation"])] = i

    # Build paired diff arrays for observed data
    n_obs = 0
    for variation in variations:
        paired_diffs = []
        for cid in case_ids:
            w_idx = lookup.get((cid, "race_white"))
            v_idx = lookup.get((cid, variation))
            if w_idx is not None and v_idx is not None:
                paired_diffs.append(demo_weights[v_idx] - demo_weights[w_idx])
        if paired_diffs:
            n_obs += _vectorized_fdr_count(np.stack(paired_diffs))
    print(f"  Observed FDR-significant: {n_obs}")

    # Pre-compute case-variation index structure
    case_var_indices = {}
    for cid in case_ids:
        indices = []
        vars_list = []
        for i, m in enumerate(meta):
            if m["case_id"] == cid:
                indices.append(i)
                vars_list.append(m["variation"])
        case_var_indices[cid] = (indices, vars_list)

    null_counts = []
    for perm_i in tqdm(range(n_perms), desc="Permutation test"):
        # Create shuffled label mapping
        shuffled_var = [None] * len(meta)
        for cid, (indices, vars_list) in case_var_indices.items():
            shuffled = list(vars_list)
            np.random.shuffle(shuffled)
            for idx, var in zip(indices, shuffled):
                shuffled_var[idx] = var

        n_sig = 0
        for variation in variations:
            paired_diffs = []
            for cid in case_ids:
                w_idx = None
                v_idx = None
                for i, m in enumerate(meta):
                    if m["case_id"] == cid:
                        if shuffled_var[i] == "race_white":
                            w_idx = i
                        elif shuffled_var[i] == variation:
                            v_idx = i
                if w_idx is not None and v_idx is not None:
                    paired_diffs.append(demo_weights[v_idx] - demo_weights[w_idx])
            if paired_diffs:
                n_sig += _vectorized_fdr_count(np.stack(paired_diffs))

        null_counts.append(int(n_sig))

    null_arr = np.array(null_counts)
    n_exceed = (null_arr >= n_obs).sum()
    p_perm = (n_exceed + 1) / (n_perms + 1)

    return {
        "observed_n_significant": int(n_obs),
        "null_mean": float(null_arr.mean()),
        "null_sd": float(null_arr.std()),
        "null_min": int(null_arr.min()),
        "null_max": int(null_arr.max()),
        "excess_over_null_mean": int(n_obs - null_arr.mean()),
        "excess_over_null_max": int(n_obs - null_arr.max()),
        "n_permutations": n_perms,
        "n_exceed": int(n_exceed),
        "p_value": float(p_perm),
    }


def l1_analysis(base_weights, base_results):
    """Cross-validated L1 logistic regression on 33,732-d concept weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    y = np.array([r["detection_truth"] for r in base_results])
    X = base_weights  # (400, 33732)

    print(f"  L1 analysis: X={X.shape}, y={y.shape}, prevalence={y.mean():.3f}")

    results = []
    for C in C_VALUES:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        fold_accs = []
        fold_coefs = []

        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(
                C=C, penalty="l1", solver="saga", max_iter=5000,
                random_state=SEED, class_weight="balanced",
            )
            clf.fit(X[train_idx], y[train_idx])
            fold_accs.append(clf.score(X[test_idx], y[test_idx]))
            fold_coefs.append(clf.coef_[0])

        mean_coef = np.mean(fold_coefs, axis=0)
        n_nonzero = (np.abs(mean_coef) > 1e-8).sum()

        results.append({
            "C": C,
            "n_nonzero_concepts": int(n_nonzero),
            "cv_accuracy": float(np.mean(fold_accs)),
            "cv_accuracy_std": float(np.std(fold_accs)),
        })
        print(f"    C={C}: {n_nonzero} non-zero concepts, "
              f"accuracy={np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")

    return results


def pca_analysis(demo_weights, meta):
    """PCA of concept weight space across demographic conditions."""
    from sklearn.decomposition import PCA

    # Filter to complete paired data
    case_ids = sorted(set(m["case_id"] for m in meta))
    variations = sorted(set(m["variation"] for m in meta))

    lookup = {}
    for i, m in enumerate(meta):
        lookup[(m["case_id"], m["variation"])] = i

    complete_indices = []
    complete_labels = []
    for cid in case_ids:
        indices = [lookup.get((cid, v)) for v in variations]
        if all(idx is not None for idx in indices):
            for v, idx in zip(variations, indices):
                complete_indices.append(idx)
                complete_labels.append(v)

    X = demo_weights[complete_indices]  # (N_complete, 33732)
    labels = complete_labels

    print(f"  PCA: {X.shape[0]} complete observations ({X.shape[0]//len(variations)} vignettes)")

    pca = PCA(n_components=min(50, X.shape[0]-1))
    scores = pca.fit_transform(X)

    # ANOVA on first 5 PCs
    anova_results = []
    for pc in range(min(5, scores.shape[1])):
        groups = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append(scores[i, pc])
        f_stat, p_val = stats.f_oneway(*groups.values())
        anova_results.append({
            "pc": pc + 1,
            "variance_explained": float(pca.explained_variance_ratio_[pc]),
            "f_stat": float(f_stat),
            "p_value": float(p_val),
        })

    return {
        "n_complete_observations": X.shape[0],
        "n_vignettes": X.shape[0] // len(variations),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist()[:50],
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "top5_variance": float(sum(pca.explained_variance_ratio_[:5])),
        "n_components_90pct": int(np.searchsorted(
            np.cumsum(pca.explained_variance_ratio_), 0.9) + 1),
        "anova_results": anova_results,
    }


def main():
    print("Loading concept weight data...")
    base_weights, demo_weights, meta = load_concept_weights()
    print(f"  Base: {base_weights.shape}, Demo: {demo_weights.shape}")

    with open(OUTPUT_DIR / "steerling_base_results.json") as f:
        base_results = json.load(f)

    # --- (A) Effective number of independent tests ---
    print("\n=== Effective Number of Independent Tests ===")
    eff_tests = effective_n_tests(demo_weights, threshold=0.95)
    print(f"  Positive eigenvalues: {eff_tests['n_eigenvalues_positive']}")
    print(f"  N_eff (95% variance): {eff_tests['n_eff_95pct_variance']}")
    print(f"  N_eff (Li & Ji): {eff_tests['n_eff_li_ji']}")
    print(f"  PC1 variance: {eff_tests['top_eigenvalue_pct']:.1%}")
    print(f"  Top 5 PCs: {eff_tests['top5_eigenvalues_pct']:.1%}")

    with open(OUTPUT_DIR / "effective_n_tests.json", "w") as f:
        json.dump(eff_tests, f, indent=2)

    # --- (B) Differential activation analysis ---
    print("\n=== Differential Activation Analysis ===")
    diff_df = differential_activation_analysis(demo_weights, meta)
    diff_df.to_csv(OUTPUT_DIR / "concept_differential_activation.csv", index=False)

    n_fdr = diff_df["significant_fdr"].sum()
    n_bonf = diff_df["significant_bonf"].sum()
    n_total = len(diff_df)
    print(f"  Total tests: {n_total}")
    print(f"  FDR-significant: {n_fdr} ({n_fdr/n_total:.1%})")
    print(f"  Bonferroni-significant: {n_bonf} ({n_bonf/n_total:.1%})")

    # Top concepts by Cohen's d
    top_d = diff_df.sort_values("cohens_d", key=abs, ascending=False).head(20)
    print(f"\n  Top 20 by |Cohen's d|:")
    for _, row in top_d.iterrows():
        print(f"    concept {row['concept_index']} ({row['variation']}): "
              f"d={row['cohens_d']:.4f}, q={row['q_value_fdr']:.2e}")

    # --- (C) Permutation test ---
    print("\n=== Permutation Test ===")
    perm = permutation_test(demo_weights, meta, n_perms=min(PERMUTATION_N, 200))
    print(f"  Observed significant: {perm['observed_n_significant']}")
    print(f"  Null mean: {perm['null_mean']:.1f} ± {perm['null_sd']:.1f}")
    print(f"  Null max: {perm['null_max']}")
    print(f"  Excess over null mean: {perm['excess_over_null_mean']}")
    print(f"  Excess over null max: {perm['excess_over_null_max']}")
    print(f"  Permutation p-value: {perm['p_value']}")

    with open(OUTPUT_DIR / "concept_permutation_test.json", "w") as f:
        json.dump(perm, f, indent=2)

    # --- (D) L1 analysis ---
    print("\n=== Cross-Validated L1 Analysis ===")
    l1_results = l1_analysis(base_weights, base_results)
    with open(OUTPUT_DIR / "concept_l1_analysis.json", "w") as f:
        json.dump(l1_results, f, indent=2)

    # --- (E) PCA ---
    print("\n=== PCA of Concept Weight Space ===")
    pca_results = pca_analysis(demo_weights, meta)
    print(f"  PC1 variance: {pca_results['pc1_variance']:.1%}")
    print(f"  Top 5 PCs: {pca_results['top5_variance']:.1%}")
    print(f"  Components for 90%: {pca_results['n_components_90pct']}")
    for r in pca_results["anova_results"]:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"    PC{r['pc']}: F={r['f_stat']:.2f}, p={r['p_value']:.4f} {sig}")

    with open(OUTPUT_DIR / "concept_pca_summary.json", "w") as f:
        json.dump(pca_results, f, indent=2)

    # --- Summary ---
    summary = {
        "n_real_concepts": N_REAL_CONCEPTS,
        "n_base_cases": base_weights.shape[0],
        "n_demo_inferences": demo_weights.shape[0],
        "effective_n_tests": eff_tests,
        "n_fdr_significant": int(n_fdr),
        "n_bonf_significant": int(n_bonf),
        "n_total_tests": int(n_total),
        "pct_fdr_significant": float(n_fdr / n_total),
        "permutation_p": perm["p_value"],
        "excess_over_null_mean": perm["excess_over_null_mean"],
        "l1_results": l1_results,
        "pca_summary": {
            "pc1_variance": pca_results["pc1_variance"],
            "n_components_90pct": pca_results["n_components_90pct"],
        },
    }
    with open(OUTPUT_DIR / "concept_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
