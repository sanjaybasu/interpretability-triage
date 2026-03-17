#!/usr/bin/env python3
"""Step 8: Cross-model comparative analysis.

Generates the key comparison tables and figures for the paper:
  1. Method comparison matrix: probe paradox across architectures
  2. Erasure comparison: LEACE effectiveness by architecture
  3. NLP interpretability comparison
  4. Summary figure data for manuscript

Compares:
  - Steerling-8B (concept bottleneck, 33,732 concepts)
  - Comparison model (standard LLM, hidden states)
  - NLP response analysis (natural language interpretability)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_steerling_results():
    """Load all Steerling-8B analysis results."""
    results = {}

    # Core analysis
    path = OUTPUT_DIR / "concept_analysis_summary.json"
    if path.exists():
        with open(path) as f:
            results["analysis"] = json.load(f)

    # Erasure
    path = OUTPUT_DIR / "concept_erasure_results.json"
    if path.exists():
        with open(path) as f:
            results["erasure"] = json.load(f)

    # PCA
    path = OUTPUT_DIR / "concept_pca_summary.json"
    if path.exists():
        with open(path) as f:
            results["pca"] = json.load(f)

    return results


def load_comparison_results():
    """Load comparison model analysis and erasure results."""
    results = {}

    # Find comparison model files
    for path in sorted(Path(OUTPUT_DIR).glob("comparison_*_analysis.json")):
        model_short = path.stem.replace("comparison_", "").replace("_analysis", "")
        with open(path) as f:
            results[model_short] = {"analysis": json.load(f)}

        erasure_path = OUTPUT_DIR / f"comparison_{model_short}_erasure.json"
        if erasure_path.exists():
            with open(erasure_path) as f:
                results[model_short]["erasure"] = json.load(f)

        info_path = OUTPUT_DIR / f"comparison_{model_short}_info.json"
        if info_path.exists():
            with open(info_path) as f:
                results[model_short]["info"] = json.load(f)

    return results


def load_nlp_results():
    """Load NLP response analysis results."""
    path = OUTPUT_DIR / "nlp_response_analysis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_steering_results():
    """Load Steerling steering experiment results."""
    path = OUTPUT_DIR / "corrected_steering_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def generate_method_comparison_table(steerling, comparison, nlp):
    """Generate the main comparison table: 5 methods × N models."""
    rows = []

    # Steerling-8B
    if steerling:
        analysis = steerling.get("analysis", {})
        erasure = steerling.get("erasure", {})

        rows.append({
            "Model": "Steerling-8B",
            "Architecture": "Concept Bottleneck",
            "Representation": "33,732 concepts",
            "Probe Accuracy": erasure.get("original", {}).get("probe_cv_accuracy", None),
            "INLP Directions": erasure.get("inlp", {}).get("directions_removed", None),
            "PCA Separation": "None (all P > 0.05)",
            "Paired FDR-sig": analysis.get("n_fdr_significant", None),
            "Paired FDR %": f"{analysis.get('pct_fdr_significant', 0) * 100:.1f}%",
            "Max Cohen's d": analysis.get("n_fdr_significant", None),
            "Permutation P": analysis.get("permutation_p", None),
            "LEACE FDR Reduction": f"{erasure.get('reductions', {}).get('fdr_reduction_leace_diff_pct', 0):.1f}%",
            "LEACE Directions": erasure.get("leace_mean_diff", {}).get("directions_removed", None),
            "PC1 Preserved": f"{erasure.get('leace_mean_diff', {}).get('pca_pc1_variance', 0):.1%}",
        })

    # Comparison model(s)
    for model_short, data in comparison.items():
        analysis = data.get("analysis", {})
        erasure = data.get("erasure", {})
        info = data.get("info", {})

        paired = analysis.get("paired_tests", {})
        perm = analysis.get("permutation", {})

        # PCA separation
        pca = analysis.get("pca", {})
        anova = pca.get("anova_results", [])
        pca_sig = [a for a in anova if a.get("p_value", 1) < 0.05]
        pca_str = f"{len(pca_sig)} PCs sig" if pca_sig else "None (all P > 0.05)"

        rows.append({
            "Model": info.get("model_id", model_short),
            "Architecture": "Standard Transformer",
            "Representation": f"{analysis.get('n_features', '?')} hidden dims",
            "Probe Accuracy": analysis.get("probe", {}).get("cv_accuracy", None),
            "INLP Directions": analysis.get("inlp", {}).get("directions_removed", None),
            "PCA Separation": pca_str,
            "Paired FDR-sig": paired.get("n_fdr_significant", None),
            "Paired FDR %": f"{paired.get('pct_fdr_significant', 0) * 100:.1f}%",
            "Max Cohen's d": paired.get("max_cohens_d", None),
            "Permutation P": perm.get("p_value", None),
            "LEACE FDR Reduction": f"{erasure.get('reductions', {}).get('fdr_reduction_leace_diff_pct', 0):.1f}%",
            "LEACE Directions": erasure.get("leace_mean_diff", {}).get("directions_removed", None),
            "PC1 Preserved": f"{erasure.get('leace_mean_diff', {}).get('pca_pc1_variance', 0):.1%}",
        })

    df = pd.DataFrame(rows)
    return df


def generate_nlp_comparison_table(nlp):
    """Generate NLP interpretability comparison across models."""
    rows = []
    for model_name, data in nlp.items():
        dec = data.get("decision_consistency", {})
        sem = data.get("semantic_divergence", {})
        length = data.get("response_length", {})

        rows.append({
            "Model": model_name,
            "Decision Consistency": f"{dec.get('consistency_rate', 0):.1%}",
            "W-B Similarity": f"{sem.get('white_black_similarity', {}).get('mean', 0):.3f}",
            "W-H Similarity": f"{sem.get('white_hispanic_similarity', {}).get('mean', 0):.3f}",
            "Length Diff (W-B)": f"{length.get('white_vs_black', {}).get('mean_diff', 0):.1f}",
            "Length Diff (W-H)": f"{length.get('white_vs_hispanic', {}).get('mean_diff', 0):.1f}",
            "Detection Rate W": f"{dec.get('detection_rate_white', 0):.3f}",
            "Detection Rate B": f"{dec.get('detection_rate_black', 0):.3f}",
            "Detection Rate H": f"{dec.get('detection_rate_hispanic', 0):.3f}",
        })

    return pd.DataFrame(rows)


def generate_interpretability_strategy_table(steerling, comparison, nlp):
    """Generate the key paper table: interpretability strategies ranked by detection power."""
    rows = []

    # Strategy 1: Output-level auditing (decision comparison)
    for model_name, data in nlp.items():
        dec = data.get("decision_consistency", {})
        det_w = dec.get("detection_rate_white", 0)
        det_b = dec.get("detection_rate_black", 0)
        det_h = dec.get("detection_rate_hispanic", 0)
        max_gap = max(abs(det_w - det_b), abs(det_w - det_h))
        rows.append({
            "Strategy": "Output-level auditing",
            "Model": model_name,
            "Access Required": "API output only",
            "Detection Power": "Low" if max_gap < 0.05 else "Moderate",
            "Max Disparity": f"{max_gap:.3f}",
            "Can Detect Hidden Bias": "No",
            "Can Remediate": "No (prompt engineering only)",
        })

    # Strategy 2: NLP response analysis
    for model_name, data in nlp.items():
        sem = data.get("semantic_divergence", {})
        within_sim = sem.get("mean_within_vignette_similarity", 1.0)
        rows.append({
            "Strategy": "NLP response analysis",
            "Model": model_name,
            "Access Required": "API output only",
            "Detection Power": "Moderate" if within_sim < 0.95 else "Low",
            "Max Disparity": f"sim={within_sim:.3f}",
            "Can Detect Hidden Bias": "Partial (text-level only)",
            "Can Remediate": "No",
        })

    # Strategy 3: Cross-sectional probe on hidden states
    if steerling:
        erasure = steerling.get("erasure", {})
        probe_acc = erasure.get("original", {}).get("probe_cv_accuracy", 0.333)
        rows.append({
            "Strategy": "Cross-sectional probe",
            "Model": "Steerling-8B (concepts)",
            "Access Required": "Internal representations",
            "Detection Power": "None (chance accuracy)",
            "Max Disparity": f"acc={probe_acc:.3f} vs 0.333",
            "Can Detect Hidden Bias": "No (false reassurance)",
            "Can Remediate": "No",
        })

    for model_short, data in comparison.items():
        probe_acc = data.get("analysis", {}).get("probe", {}).get("cv_accuracy", 0.333)
        above_chance = probe_acc > 0.333 + 0.05
        rows.append({
            "Strategy": "Cross-sectional probe",
            "Model": f"{model_short} (hidden states)",
            "Access Required": "Internal representations",
            "Detection Power": "Moderate" if above_chance else "None",
            "Max Disparity": f"acc={probe_acc:.3f} vs 0.333",
            "Can Detect Hidden Bias": "Yes" if above_chance else "No",
            "Can Remediate": "Via LEACE" if above_chance else "No",
        })

    # Strategy 4: Paired within-vignette analysis
    if steerling:
        analysis = steerling.get("analysis", {})
        n_fdr = analysis.get("n_fdr_significant", 0)
        rows.append({
            "Strategy": "Paired within-vignette",
            "Model": "Steerling-8B (concepts)",
            "Access Required": "Internal representations + paired design",
            "Detection Power": "High",
            "Max Disparity": f"{n_fdr:,} FDR-sig features",
            "Can Detect Hidden Bias": "Yes",
            "Can Remediate": "Via LEACE (2 directions, 100%)",
        })

    for model_short, data in comparison.items():
        paired = data.get("analysis", {}).get("paired_tests", {})
        n_fdr = paired.get("n_fdr_significant", 0)
        erasure = data.get("erasure", {})
        leace_red = erasure.get("reductions", {}).get("fdr_reduction_leace_diff_pct", 0)
        rows.append({
            "Strategy": "Paired within-vignette",
            "Model": f"{model_short} (hidden states)",
            "Access Required": "Internal representations + paired design",
            "Detection Power": "High" if n_fdr > 100 else "Low",
            "Max Disparity": f"{n_fdr:,} FDR-sig features",
            "Can Detect Hidden Bias": "Yes" if n_fdr > 0 else "No",
            "Can Remediate": f"Via LEACE ({leace_red:.0f}%)" if leace_red > 50 else "Limited",
        })

    return pd.DataFrame(rows)


def print_key_findings(steerling, comparison, nlp):
    """Print the key findings for the manuscript."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR MANUSCRIPT")
    print("=" * 70)

    # Finding 1: Probe paradox comparison
    print("\n1. PROBE PARADOX: CBM-specific or general?")
    if steerling:
        erasure = steerling.get("erasure", {})
        s_probe = erasure.get("original", {}).get("probe_cv_accuracy", 0)
        s_fdr = steerling.get("analysis", {}).get("n_fdr_significant", 0)
        print(f"   Steerling-8B: probe={s_probe:.3f} (chance=0.333), "
              f"paired FDR-sig={s_fdr:,}")

    for model_short, data in comparison.items():
        c_probe = data.get("analysis", {}).get("probe", {}).get("cv_accuracy", 0)
        c_fdr = data.get("analysis", {}).get("paired_tests", {}).get("n_fdr_significant", 0)
        c_inlp = data.get("analysis", {}).get("inlp", {}).get("directions_removed", 0)
        print(f"   {model_short}: probe={c_probe:.3f}, INLP dirs={c_inlp}, "
              f"paired FDR-sig={c_fdr:,}")

        if c_probe > 0.333 + 0.05:
            print("   >> PROBE PARADOX IS CBM-SPECIFIC: standard probe detects bias in non-CBM model")
        else:
            print("   >> PROBE PARADOX IS GENERAL: standard probe fails for both architectures")

    # Finding 2: LEACE comparison
    print("\n2. ERASURE EFFECTIVENESS:")
    if steerling:
        erasure = steerling.get("erasure", {})
        s_red = erasure.get("reductions", {}).get("fdr_reduction_leace_diff_pct", 0)
        s_pc1 = erasure.get("leace_mean_diff", {}).get("pca_pc1_variance", 0)
        print(f"   Steerling-8B: {s_red:.1f}% FDR reduction, PC1 preserved={s_pc1:.1%}")

    for model_short, data in comparison.items():
        erasure = data.get("erasure", {})
        if erasure:
            c_red = erasure.get("reductions", {}).get("fdr_reduction_leace_diff_pct", 0)
            c_pc1 = erasure.get("leace_mean_diff", {}).get("pca_pc1_variance", 0)
            print(f"   {model_short}: {c_red:.1f}% FDR reduction, PC1 preserved={c_pc1:.1%}")

    # Finding 3: NLP detection
    print("\n3. NATURAL LANGUAGE INTERPRETABILITY:")
    for model_name, data in nlp.items():
        dec = data.get("decision_consistency", {})
        sem = data.get("semantic_divergence", {})
        cons = dec.get("consistency_rate", 0)
        sim = sem.get("mean_within_vignette_similarity", 0)
        print(f"   {model_name}: consistency={cons:.1%}, within-vignette sim={sim:.3f}")


def main():
    print("Loading all results...")
    steerling = load_steerling_results()
    comparison = load_comparison_results()
    nlp = load_nlp_results()
    steering = load_steering_results()

    if not steerling and not comparison:
        print("No results found. Run steps 03b/04c (Steerling) and/or 06 (comparison) first.")
        sys.exit(1)

    # Generate tables
    print("\n--- Method Comparison Table ---")
    method_table = generate_method_comparison_table(steerling, comparison, nlp)
    print(method_table.to_string(index=False))
    method_table.to_csv(OUTPUT_DIR / "table_method_comparison.csv", index=False)

    if nlp:
        print("\n--- NLP Comparison Table ---")
        nlp_table = generate_nlp_comparison_table(nlp)
        print(nlp_table.to_string(index=False))
        nlp_table.to_csv(OUTPUT_DIR / "table_nlp_comparison.csv", index=False)

    print("\n--- Interpretability Strategy Table ---")
    strategy_table = generate_interpretability_strategy_table(steerling, comparison, nlp)
    print(strategy_table.to_string(index=False))
    strategy_table.to_csv(OUTPUT_DIR / "table_strategy_comparison.csv", index=False)

    # Key findings
    print_key_findings(steerling, comparison, nlp)

    # Save combined summary
    summary = {
        "steerling_available": bool(steerling),
        "comparison_models": list(comparison.keys()),
        "nlp_models": list(nlp.keys()),
        "steering_available": steering is not None,
    }
    with open(OUTPUT_DIR / "comparative_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll comparative tables saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
