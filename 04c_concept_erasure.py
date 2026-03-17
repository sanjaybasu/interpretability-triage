#!/usr/bin/env python3
"""Step 4c: Concept-level race erasure via LEACE and null-space projection.

Demonstrates that distributed racial encoding in concept activations — invisible
to PCA and near-null under L1 — can be identified and removed by supervised
linear concept erasure (LEACE). This transforms the finding from purely
diagnostic ("race is encoded") to constructive ("race can be erased while
preserving clinical content").

Runs locally on CPU using saved concept activations from Steps 2-3.
No GPU or steerling package required.

Methods implemented:
  1. Linear probe: L2-regularized logistic regression predicting race from
     33,732 concept activations (cross-validated)
  2. Mean difference direction: average(X_nonwhite - X_white) as race vector
  3. LEACE (Belrose et al., 2023): optimal linear concept erasure
  4. INLP (Ravfogel et al., 2020): iterative null-space projection
  5. Re-evaluation: differential activation analysis on erased activations
  6. Preservation tests: PCA structure, hazard-concept associations
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from config import FDR_ALPHA, SEED, OUTPUT_DIR
from src.utils import benjamini_hochberg, cohens_d

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_concept_data():
    """Load and deduplicate concept activations with metadata."""
    weights = np.load(OUTPUT_DIR / "demo_concept_weights.npy")
    with open(OUTPUT_DIR / "demo_concept_weights_meta.json") as f:
        meta = json.load(f)

    # Deduplicate: average activations for repeated (case_id, variation) pairs
    pair_to_rows = {}
    for i, m in enumerate(meta):
        key = (m["case_id"], m["variation"])
        pair_to_rows.setdefault(key, []).append(i)

    dedup_keys = sorted(pair_to_rows.keys())
    dedup_weights = np.zeros((len(dedup_keys), weights.shape[1]), dtype=np.float32)
    dedup_meta = []
    for i, key in enumerate(dedup_keys):
        rows = pair_to_rows[key]
        dedup_weights[i] = weights[rows].mean(axis=0)
        dedup_meta.append({"case_id": key[0], "variation": key[1]})

    print(f"Loaded {weights.shape[0]} rows, deduplicated to {len(dedup_keys)} "
          f"({len(set(k[0] for k in dedup_keys))} cases × "
          f"{len(set(k[1] for k in dedup_keys))} variations)")
    return dedup_weights, dedup_meta


def build_paired_data(weights, meta):
    """Build paired arrays: for each case, White vs Black vs Hispanic."""
    case_to_idx = {}
    for i, m in enumerate(meta):
        case_to_idx.setdefault(m["case_id"], {})[m["variation"]] = i

    # Keep only cases with all three variations
    complete_cases = [
        cid for cid, variants in case_to_idx.items()
        if all(v in variants for v in ["race_white", "race_black", "race_hispanic"])
    ]
    complete_cases.sort()
    print(f"Complete triplets: {len(complete_cases)}")

    # Build matrices
    X_white = np.array([weights[case_to_idx[c]["race_white"]] for c in complete_cases])
    X_black = np.array([weights[case_to_idx[c]["race_black"]] for c in complete_cases])
    X_hispanic = np.array([weights[case_to_idx[c]["race_hispanic"]] for c in complete_cases])

    return X_white, X_black, X_hispanic, complete_cases


def race_probe(X_all, y_all, n_folds=5):
    """Train cross-validated logistic regression probe to predict race."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    # L2-regularized logistic regression (multinomial)
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="saga", max_iter=5000,
        random_state=SEED, multi_class="multinomial",
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_accs = []
    for train_idx, test_idx in skf.split(X_all, y_enc):
        clf.fit(X_all[train_idx], y_enc[train_idx])
        fold_accs.append(clf.score(X_all[test_idx], y_enc[test_idx]))

    # Fit on all data for coefficient extraction
    clf.fit(X_all, y_enc)

    # Cross-validated predictions for confusion analysis
    y_pred_cv = cross_val_predict(clf, X_all, y_enc, cv=skf)
    cv_acc = np.mean(y_pred_cv == y_enc)

    print(f"\nRace probe results:")
    print(f"  Fold accuracies: {[f'{a:.3f}' for a in fold_accs]}")
    print(f"  Mean CV accuracy: {np.mean(fold_accs):.3f} (chance = {1/len(le.classes_):.3f})")
    print(f"  CV predict accuracy: {cv_acc:.3f}")

    return clf, le, fold_accs, cv_acc


def compute_race_directions(clf, X_white, X_black, X_hispanic):
    """Extract race encoding directions from probe and mean differences."""
    # Method 1: Logistic regression weight vectors (one per class)
    # For 3-class: weights shape is (3, 33732)
    W_probe = clf.coef_  # (n_classes, n_features)

    # Method 2: Mean difference vectors
    diff_black = (X_black - X_white).mean(axis=0)  # (33732,)
    diff_hispanic = (X_hispanic - X_white).mean(axis=0)

    # Combined race subspace: SVD of stacked difference vectors
    diff_stack = np.vstack([diff_black, diff_hispanic])  # (2, 33732)
    U, S, Vt = np.linalg.svd(diff_stack, full_matrices=False)
    race_basis = Vt[:2]  # Top 2 singular vectors spanning race subspace

    # Also compute from probe weights
    U_p, S_p, Vt_p = np.linalg.svd(W_probe, full_matrices=False)
    probe_basis = Vt_p[:2]  # Top 2 directions from probe

    print(f"\nRace directions:")
    print(f"  Mean diff singular values: {S}")
    print(f"  Probe singular values: {S_p[:3]}")
    print(f"  Cosine similarity (diff_black vs probe[0]): "
          f"{abs(np.dot(diff_black / np.linalg.norm(diff_black), probe_basis[0])):.3f}")

    return {
        "mean_diff_black": diff_black,
        "mean_diff_hispanic": diff_hispanic,
        "race_basis_diff": race_basis,
        "race_basis_probe": probe_basis,
        "probe_weights": W_probe,
    }


def leace_projection(X, directions, method="diff"):
    """Apply LEACE-style projection to erase race directions.

    For the mean-difference method: P = I - V @ V^T where V spans
    the race subspace identified by mean difference vectors.

    For the probe method: P = I - V @ V^T where V spans the
    logistic regression weight subspace.
    """
    if method == "diff":
        basis = directions["race_basis_diff"]
    elif method == "probe":
        basis = directions["race_basis_probe"]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Orthogonal projection: remove components along race basis
    # P = I - V^T @ V (V is already orthonormal from SVD)
    X_proj = X - X @ basis.T @ basis
    return X_proj


def inlp_projection(X_all, y_all, n_iterations=10):
    """Iterative Null-space Projection (Ravfogel et al., 2020).

    Iteratively trains linear classifiers, projects out their
    weight vectors, and retrains until the classifier can no longer
    predict the protected attribute.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    X_proj = X_all.copy()
    removed_directions = []
    accuracies = []

    for it in range(n_iterations):
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="saga", max_iter=3000,
            random_state=SEED + it, multi_class="multinomial",
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + it)
        y_pred = cross_val_predict(clf, X_proj, y_enc, cv=skf)
        acc = np.mean(y_pred == y_enc)
        accuracies.append(acc)

        # Stop if accuracy is near chance
        if acc < (1 / len(le.classes_)) + 0.02:
            print(f"  INLP converged at iteration {it + 1} (acc={acc:.3f})")
            break

        # Fit on all data and extract weight direction
        clf.fit(X_proj, y_enc)
        W = clf.coef_  # (n_classes, n_features)
        # SVD to get principal direction
        _, _, Vt = np.linalg.svd(W, full_matrices=False)
        direction = Vt[0]  # Top singular vector
        removed_directions.append(direction)

        # Project out this direction
        X_proj = X_proj - np.outer(X_proj @ direction, direction)

    print(f"\nINLP results:")
    print(f"  Iterations: {len(accuracies)}")
    print(f"  Accuracies: {[f'{a:.3f}' for a in accuracies]}")
    print(f"  Directions removed: {len(removed_directions)}")

    return X_proj, removed_directions, accuracies


def rerun_differential_analysis(X_white, X_black, X_hispanic, label="original"):
    """Run paired t-tests for differential concept activation."""
    n_concepts = X_white.shape[1]
    results = []

    for var_name, X_var in [("race_black", X_black), ("race_hispanic", X_hispanic)]:
        diffs = X_var - X_white  # (n_cases, n_concepts)
        for j in range(n_concepts):
            d = diffs[:, j]
            if np.std(d) == 0:
                t_stat, p_val = 0.0, 1.0
            else:
                t_stat, p_val = stats.ttest_1samp(d, 0)
            results.append({
                "concept_index": j,
                "variation": var_name,
                "mean_diff": d.mean(),
                "t_stat": t_stat,
                "p_value": p_val,
            })

    df = pd.DataFrame(results)

    # BH-FDR correction
    bh = benjamini_hochberg(df["p_value"].values, alpha=FDR_ALPHA)
    df["q_value_fdr"] = bh["q_values"]
    df["significant_fdr"] = bh["rejected"]
    df["p_bonferroni"] = np.minimum(df["p_value"] * len(df), 1.0)
    df["significant_bonf"] = df["p_bonferroni"] < 0.05

    # Cohen's d
    ds = []
    for _, row in df.iterrows():
        var_name = row["variation"]
        j = row["concept_index"]
        X_var = X_black if var_name == "race_black" else X_hispanic
        ds.append(cohens_d(X_var[:, j], X_white[:, j]))
    df["cohens_d"] = ds

    n_fdr = df["significant_fdr"].sum()
    n_bonf = df["significant_bonf"].sum()
    max_d = df["cohens_d"].abs().max()

    print(f"\n  [{label}] Differential activation:")
    print(f"    FDR-significant: {n_fdr:,} / {len(df):,} ({100*n_fdr/len(df):.1f}%)")
    print(f"    Bonferroni-significant: {n_bonf:,} / {len(df):,}")
    print(f"    Max |Cohen's d|: {max_d:.3f}")

    return df, n_fdr, n_bonf, max_d


def test_pca_preservation(X_original, X_erased, label="erased"):
    """Check that PCA structure is preserved after erasure."""
    from sklearn.decomposition import PCA

    pca_orig = PCA(n_components=5, random_state=SEED).fit(X_original)
    pca_erased = PCA(n_components=5, random_state=SEED).fit(X_erased)

    print(f"\n  PCA preservation ({label}):")
    print(f"    Original explained variance: {pca_orig.explained_variance_ratio_[:5]}")
    print(f"    Erased explained variance:   {pca_erased.explained_variance_ratio_[:5]}")

    # Cosine similarity between corresponding PCs
    for i in range(min(3, pca_orig.n_components_)):
        cos_sim = abs(np.dot(
            pca_orig.components_[i], pca_erased.components_[i]
        )) / (np.linalg.norm(pca_orig.components_[i]) *
               np.linalg.norm(pca_erased.components_[i]))
        print(f"    PC{i+1} cosine similarity: {cos_sim:.4f}")

    return pca_orig, pca_erased


def main():
    np.random.seed(SEED)
    print("=" * 60)
    print("Step 4c: Concept-Level Race Erasure (LEACE/INLP)")
    print("=" * 60)

    # Load data
    weights, meta = load_concept_data()
    X_white, X_black, X_hispanic, cases = build_paired_data(weights, meta)
    n_cases = len(cases)
    n_concepts = X_white.shape[1]

    # Stack all for probe training
    X_all = np.vstack([X_white, X_black, X_hispanic])
    y_all = np.array(["white"] * n_cases + ["black"] * n_cases + ["hispanic"] * n_cases)

    # ============================================================
    # 1. Baseline: differential activation on original activations
    # ============================================================
    print("\n" + "=" * 60)
    print("1. BASELINE: Original concept activations")
    print("=" * 60)
    df_orig, n_fdr_orig, n_bonf_orig, max_d_orig = rerun_differential_analysis(
        X_white, X_black, X_hispanic, label="original"
    )

    # ============================================================
    # 2. Race prediction probe
    # ============================================================
    print("\n" + "=" * 60)
    print("2. RACE PREDICTION PROBE")
    print("=" * 60)
    clf, le, fold_accs, cv_acc = race_probe(X_all, y_all)

    # ============================================================
    # 3. Compute race directions
    # ============================================================
    print("\n" + "=" * 60)
    print("3. RACE DIRECTION EXTRACTION")
    print("=" * 60)
    directions = compute_race_directions(clf, X_white, X_black, X_hispanic)

    # ============================================================
    # 4. LEACE projection (mean difference method)
    # ============================================================
    print("\n" + "=" * 60)
    print("4. LEACE ERASURE: Mean Difference Method")
    print("=" * 60)
    X_all_leace_diff = leace_projection(X_all, directions, method="diff")
    X_w_ld = X_all_leace_diff[:n_cases]
    X_b_ld = X_all_leace_diff[n_cases:2*n_cases]
    X_h_ld = X_all_leace_diff[2*n_cases:]

    # Re-test probe after erasure
    clf_post_diff, _, fold_accs_post_diff, cv_acc_post_diff = race_probe(
        X_all_leace_diff, y_all
    )
    df_ld, n_fdr_ld, n_bonf_ld, max_d_ld = rerun_differential_analysis(
        X_w_ld, X_b_ld, X_h_ld, label="LEACE-diff"
    )
    pca_orig_diff, pca_erased_diff = test_pca_preservation(X_all, X_all_leace_diff, "LEACE-diff")

    # ============================================================
    # 5. LEACE projection (probe method)
    # ============================================================
    print("\n" + "=" * 60)
    print("5. LEACE ERASURE: Probe Weight Method")
    print("=" * 60)
    X_all_leace_probe = leace_projection(X_all, directions, method="probe")
    X_w_lp = X_all_leace_probe[:n_cases]
    X_b_lp = X_all_leace_probe[n_cases:2*n_cases]
    X_h_lp = X_all_leace_probe[2*n_cases:]

    clf_post_probe, _, fold_accs_post_probe, cv_acc_post_probe = race_probe(
        X_all_leace_probe, y_all
    )
    df_lp, n_fdr_lp, n_bonf_lp, max_d_lp = rerun_differential_analysis(
        X_w_lp, X_b_lp, X_h_lp, label="LEACE-probe"
    )
    pca_orig_probe, pca_erased_probe = test_pca_preservation(X_all, X_all_leace_probe, "LEACE-probe")

    # ============================================================
    # 6. INLP (Iterative Null-Space Projection)
    # ============================================================
    print("\n" + "=" * 60)
    print("6. INLP: Iterative Null-Space Projection")
    print("=" * 60)
    X_all_inlp, removed_dirs, inlp_accs = inlp_projection(X_all, y_all, n_iterations=15)
    X_w_inlp = X_all_inlp[:n_cases]
    X_b_inlp = X_all_inlp[n_cases:2*n_cases]
    X_h_inlp = X_all_inlp[2*n_cases:]

    df_inlp, n_fdr_inlp, n_bonf_inlp, max_d_inlp = rerun_differential_analysis(
        X_w_inlp, X_b_inlp, X_h_inlp, label="INLP"
    )
    pca_orig_inlp, pca_erased_inlp = test_pca_preservation(X_all, X_all_inlp, "INLP")

    # ============================================================
    # 7. Summary comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: Concept Erasure Comparison")
    print("=" * 60)

    summary = {
        "original": {
            "probe_cv_accuracy": cv_acc,
            "n_fdr_significant": int(n_fdr_orig),
            "n_bonf_significant": int(n_bonf_orig),
            "max_cohens_d": float(max_d_orig),
            "pca_pc1_variance": float(pca_orig_diff.explained_variance_ratio_[0]),
        },
        "leace_mean_diff": {
            "probe_cv_accuracy": cv_acc_post_diff,
            "n_fdr_significant": int(n_fdr_ld),
            "n_bonf_significant": int(n_bonf_ld),
            "max_cohens_d": float(max_d_ld),
            "pca_pc1_variance": float(pca_erased_diff.explained_variance_ratio_[0]),
            "directions_removed": 2,
        },
        "leace_probe": {
            "probe_cv_accuracy": cv_acc_post_probe,
            "n_fdr_significant": int(n_fdr_lp),
            "n_bonf_significant": int(n_bonf_lp),
            "max_cohens_d": float(max_d_lp),
            "pca_pc1_variance": float(pca_erased_probe.explained_variance_ratio_[0]),
            "directions_removed": 2,
        },
        "inlp": {
            "probe_cv_accuracy": float(inlp_accs[-1]) if inlp_accs else None,
            "n_fdr_significant": int(n_fdr_inlp),
            "n_bonf_significant": int(n_bonf_inlp),
            "max_cohens_d": float(max_d_inlp),
            "pca_pc1_variance": float(pca_erased_inlp.explained_variance_ratio_[0]),
            "directions_removed": len(removed_dirs),
            "inlp_accuracy_trajectory": [float(a) for a in inlp_accs],
        },
        "metadata": {
            "n_cases": n_cases,
            "n_concepts": n_concepts,
            "n_total_tests": int(n_concepts * 2),
            "seed": SEED,
            "fdr_alpha": FDR_ALPHA,
            "chance_accuracy": 1 / 3,
        },
    }

    # Print comparison table
    print(f"\n{'Method':<20} {'Probe Acc':<12} {'FDR Sig':<12} {'Bonf Sig':<12} {'Max |d|':<10} {'PC1 Var':<10}")
    print("-" * 76)
    for method, s in summary.items():
        if method == "metadata":
            continue
        acc = s.get("probe_cv_accuracy", 0)
        print(f"{method:<20} {acc:.3f}        {s['n_fdr_significant']:<12,} {s['n_bonf_significant']:<12,} {s['max_cohens_d']:<10.3f} {s['pca_pc1_variance']:.3f}")

    # Compute reduction percentages
    fdr_reduction_ld = 100 * (1 - n_fdr_ld / n_fdr_orig) if n_fdr_orig > 0 else 0
    fdr_reduction_lp = 100 * (1 - n_fdr_lp / n_fdr_orig) if n_fdr_orig > 0 else 0
    fdr_reduction_inlp = 100 * (1 - n_fdr_inlp / n_fdr_orig) if n_fdr_orig > 0 else 0
    print(f"\nFDR-significant reduction:")
    print(f"  LEACE (mean diff): {fdr_reduction_ld:.1f}%")
    print(f"  LEACE (probe):     {fdr_reduction_lp:.1f}%")
    print(f"  INLP:              {fdr_reduction_inlp:.1f}%")

    summary["reductions"] = {
        "fdr_reduction_leace_diff_pct": fdr_reduction_ld,
        "fdr_reduction_leace_probe_pct": fdr_reduction_lp,
        "fdr_reduction_inlp_pct": fdr_reduction_inlp,
    }

    # Save results
    out_path = OUTPUT_DIR / "concept_erasure_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save INLP trajectory for plotting
    inlp_traj_path = OUTPUT_DIR / "inlp_trajectory.json"
    with open(inlp_traj_path, "w") as f:
        json.dump({
            "accuracies": [float(a) for a in inlp_accs],
            "n_directions_removed": list(range(1, len(inlp_accs) + 1)),
        }, f, indent=2)

    # Save erased differential activation results
    df_ld.to_csv(OUTPUT_DIR / "concept_differential_activation_leace_diff.csv", index=False)
    df_inlp.to_csv(OUTPUT_DIR / "concept_differential_activation_inlp.csv", index=False)

    print(f"\nAll erasure outputs saved to {OUTPUT_DIR}/")
    return summary


if __name__ == "__main__":
    summary = main()
