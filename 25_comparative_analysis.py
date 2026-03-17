#!/usr/bin/env python3
"""Step 25: Comparative analysis across 4 interpretability methods.

Generates head-to-head comparison tables and figures for:
  Arm 1: Concept Bottleneck Steering (Steerling-8B)
  Arm 2: SAE Feature Steering (Qwen 2.5 7B, trained from scratch)
  Arm 3: Activation Patching (Qwen 2.5 7B logit lens)
  Arm 4: Probing / TSV Steering (Qwen 2.5 7B)

Outputs:
  - tables/comparative_correction_rates.csv
  - tables/comparative_disruption_rates.csv
  - tables/comparative_summary.csv
  - figures/figure2_comparative_correction.pdf
  - figures/figure3_method_comparison.pdf
  - output/comparative_analysis.json
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, SEED
from src.utils import wilson_ci

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_arm_results():
    """Load results from all 4 arms."""
    arms = {}

    # Arm 1: Steerling concept steering
    steerling_path = OUTPUT_DIR / "causal_correction_results.json"
    if steerling_path.exists():
        with open(steerling_path) as f:
            arms["concept_steering"] = json.load(f)
    tp_path = OUTPUT_DIR / "tp_correction_results.json"
    if tp_path.exists():
        with open(tp_path) as f:
            arms["concept_tp_correction"] = json.load(f)

    # Arm 2: SAE steering
    sae_path = OUTPUT_DIR / "sae_steering_results.json"
    if sae_path.exists():
        with open(sae_path) as f:
            arms["sae_steering"] = json.load(f)
    sae_summary_path = OUTPUT_DIR / "sae_steering_summary.json"
    if sae_summary_path.exists():
        with open(sae_summary_path) as f:
            arms["sae_summary"] = json.load(f)

    # Arm 3: Activation patching
    patch_path = OUTPUT_DIR / "activation_patching_results.json"
    if patch_path.exists():
        with open(patch_path) as f:
            arms["activation_patching"] = json.load(f)
    patch_summary_path = OUTPUT_DIR / "activation_patching_summary.json"
    if patch_summary_path.exists():
        with open(patch_summary_path) as f:
            arms["patching_summary"] = json.load(f)

    # Arm 4: TSV steering
    tsv_path = OUTPUT_DIR / "tsv_steering_results.json"
    if tsv_path.exists():
        with open(tsv_path) as f:
            arms["tsv_steering"] = json.load(f)
    tsv_summary_path = OUTPUT_DIR / "tsv_steering_summary.json"
    if tsv_summary_path.exists():
        with open(tsv_summary_path) as f:
            arms["tsv_summary"] = json.load(f)

    # Probe results
    probe_path = OUTPUT_DIR / "probe_results.json"
    if probe_path.exists():
        with open(probe_path) as f:
            arms["probes"] = json.load(f)

    # Logit lens
    logit_path = OUTPUT_DIR / "logit_lens_summary.json"
    if logit_path.exists():
        with open(logit_path) as f:
            arms["logit_lens"] = json.load(f)

    return arms


def extract_correction_metrics(arms):
    """Extract standardized correction metrics from each arm.

    Returns list of dicts with: method, dataset, fn_corrected, fn_total,
    tp_disrupted, tp_total, fp_induced, fp_total, alpha/condition.
    """
    rows = []

    # Arm 1: Steerling concept steering — best condition (TP-mean or alpha=1.0)
    if "concept_tp_correction" in arms:
        data = arms["concept_tp_correction"]
        for case in data:
            if isinstance(case, dict) and "dataset" in case:
                pass  # Parse per-case results
        # Use summary from existing Table 4 instead
    if "concept_steering" in arms:
        data = arms["concept_steering"]
        # Parse the causal correction results format
        for entry in data if isinstance(data, list) else []:
            if isinstance(entry, dict):
                rows.append({
                    "method": "Concept Steering (Steerling-8B)",
                    "dataset": entry.get("dataset", "physician"),
                    "condition": entry.get("condition", ""),
                    "alpha": entry.get("alpha", ""),
                    "case_id": entry.get("case_id", ""),
                    "original_detection": entry.get("original_detection", 0),
                    "steered_detection": entry.get("steered_detection", 0),
                    "ground_truth": entry.get("ground_truth", 0),
                })

    # Arm 2: SAE steering
    if "sae_steering" in arms:
        data = arms["sae_steering"]
        for entry in data if isinstance(data, list) else []:
            if isinstance(entry, dict):
                rows.append({
                    "method": "SAE Feature Steering (Qwen 2.5 7B)",
                    "dataset": entry.get("dataset", "physician"),
                    "condition": entry.get("condition", ""),
                    "alpha": entry.get("alpha", ""),
                    "case_id": entry.get("case_id", ""),
                    "original_detection": entry.get("original_detection", 0),
                    "steered_detection": entry.get("steered_detection", 0),
                    "ground_truth": entry.get("ground_truth", 0),
                })

    # Arm 3: Activation patching
    if "activation_patching" in arms:
        data = arms["activation_patching"]
        for entry in data if isinstance(data, list) else []:
            if isinstance(entry, dict):
                rows.append({
                    "method": "Activation Patching (Logit Lens)",
                    "dataset": entry.get("dataset", "physician"),
                    "condition": entry.get("condition", ""),
                    "alpha": entry.get("alpha", ""),
                    "case_id": entry.get("case_id", ""),
                    "original_detection": entry.get("original_detection", 0),
                    "steered_detection": entry.get("steered_detection", 0),
                    "ground_truth": entry.get("ground_truth", 0),
                })

    # Arm 4: TSV steering
    if "tsv_steering" in arms:
        data = arms["tsv_steering"]
        for entry in data if isinstance(data, list) else []:
            if isinstance(entry, dict):
                rows.append({
                    "method": "TSV Steering (Probing)",
                    "dataset": entry.get("dataset", "physician"),
                    "condition": entry.get("condition", ""),
                    "alpha": entry.get("alpha", ""),
                    "case_id": entry.get("case_id", ""),
                    "original_detection": entry.get("original_detection", 0),
                    "steered_detection": entry.get("steered_detection", 0),
                    "ground_truth": entry.get("ground_truth", 0),
                })

    return rows


def compute_summary_table(rows):
    """Compute aggregate correction/disruption metrics per method and condition."""
    if not rows:
        print("No per-case results available. Falling back to summary files.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary_rows = []

    for (method, dataset, condition), group in df.groupby(["method", "dataset", "condition"]):
        # False negatives: ground_truth=1, original_detection=0
        fn_mask = (group["ground_truth"] == 1) & (group["original_detection"] == 0)
        fn_cases = group[fn_mask]
        fn_corrected = int(fn_cases["steered_detection"].sum()) if len(fn_cases) > 0 else 0
        fn_total = len(fn_cases)

        # True positives: ground_truth=1, original_detection=1
        tp_mask = (group["ground_truth"] == 1) & (group["original_detection"] == 1)
        tp_cases = group[tp_mask]
        tp_disrupted = int((tp_cases["steered_detection"] == 0).sum()) if len(tp_cases) > 0 else 0
        tp_total = len(tp_cases)

        # True negatives: ground_truth=0, original_detection=0
        tn_mask = (group["ground_truth"] == 0) & (group["original_detection"] == 0)
        tn_cases = group[tn_mask]
        fp_induced = int(tn_cases["steered_detection"].sum()) if len(tn_cases) > 0 else 0
        tn_total = len(tn_cases)

        fn_rate = wilson_ci(fn_corrected, fn_total)
        tp_rate = wilson_ci(tp_disrupted, tp_total)
        fp_rate = wilson_ci(fp_induced, tn_total)

        summary_rows.append({
            "Method": method,
            "Dataset": dataset,
            "Condition": condition,
            "FN corrected": f"{fn_corrected}/{fn_total}" if fn_total > 0 else "-",
            "FN correction rate (95% CI)": f"{fn_rate[0]:.3f} ({fn_rate[1]:.3f}-{fn_rate[2]:.3f})" if fn_total > 0 else "-",
            "TP disrupted": f"{tp_disrupted}/{tp_total}" if tp_total > 0 else "-",
            "TP disruption rate (95% CI)": f"{tp_rate[0]:.3f} ({tp_rate[1]:.3f}-{tp_rate[2]:.3f})" if tp_total > 0 else "-",
            "FP induced": f"{fp_induced}/{tn_total}" if tn_total > 0 else "-",
            "FP induction rate (95% CI)": f"{fp_rate[0]:.3f} ({fp_rate[1]:.3f}-{fp_rate[2]:.3f})" if tn_total > 0 else "-",
            # Raw values for plotting
            "_fn_rate": fn_rate[0] if fn_total > 0 else 0,
            "_fn_lo": fn_rate[1] if fn_total > 0 else 0,
            "_fn_hi": fn_rate[2] if fn_total > 0 else 0,
            "_tp_rate": tp_rate[0] if tp_total > 0 else 0,
            "_tp_lo": tp_rate[1] if tp_total > 0 else 0,
            "_tp_hi": tp_rate[2] if tp_total > 0 else 0,
        })

    return pd.DataFrame(summary_rows)


def build_summary_from_files(arms):
    """Build summary table from per-arm summary JSON files when per-case data isn't available."""
    rows = []

    # Extract from summary files that each arm produces
    method_map = {
        "sae_summary": "SAE Feature Steering (Qwen 2.5 7B)",
        "patching_summary": "Activation Patching (Logit Lens)",
        "tsv_summary": "TSV Steering (Probing)",
    }

    for key, method_name in method_map.items():
        if key in arms:
            summary = arms[key]
            if isinstance(summary, dict):
                for condition, metrics in summary.items():
                    if isinstance(metrics, dict):
                        rows.append({
                            "Method": method_name,
                            "Condition": condition,
                            **metrics,
                        })

    return pd.DataFrame(rows) if rows else None


def plot_comparative_correction(summary_df):
    """Figure 2: Head-to-head FN correction rates across methods."""
    if summary_df.empty or "_fn_rate" not in summary_df.columns:
        print("No data for comparative correction figure.")
        return

    # Filter to physician dataset, best condition per method
    df = summary_df[summary_df["Dataset"] == "physician"].copy()
    if df.empty:
        df = summary_df.copy()

    # Get best correction rate per method
    best = df.loc[df.groupby("Method")["_fn_rate"].idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: FN correction rate
    ax = axes[0]
    methods = best["Method"].values
    y_pos = np.arange(len(methods))
    fn_rates = best["_fn_rate"].values
    fn_lo = best["_fn_lo"].values
    fn_hi = best["_fn_hi"].values
    xerr = np.array([fn_rates - fn_lo, fn_hi - fn_rates])

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(methods)]
    ax.barh(y_pos, fn_rates, xerr=xerr, color=colors, capsize=4, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.split("(")[0].strip() for m in methods], fontsize=10)
    ax.set_xlabel("FN correction rate", fontsize=12)
    ax.set_title("A. Correcting missed hazards", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Panel B: TP disruption rate
    ax = axes[1]
    tp_rates = best["_tp_rate"].values
    tp_lo = best["_tp_lo"].values
    tp_hi = best["_tp_hi"].values
    xerr_tp = np.array([tp_rates - tp_lo, tp_hi - tp_rates])

    ax.barh(y_pos, tp_rates, xerr=xerr_tp, color=colors, capsize=4, height=0.6, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.split("(")[0].strip() for m in methods], fontsize=10)
    ax.set_xlabel("TP disruption rate", fontsize=12)
    ax.set_title("B. Disrupting correct detections", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"figure2_comparative_correction.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure2_comparative_correction")


def plot_method_comparison(arms):
    """Figure 3: Multi-panel comparison across interpretability methods.

    Panel A: Encoding fidelity (probe AUROC across layers)
    Panel B: Feature isolation (SAE sparsity vs concept sparsity)
    Panel C: Interventional fidelity (correction rate vs disruption rate)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Probe AUROC across layers
    ax = axes[0]
    if "probes" in arms:
        probes = arms["probes"]
        if isinstance(probes, dict) and "per_layer" in probes:
            layers = []
            aurocs = []
            for layer_data in probes["per_layer"]:
                layers.append(layer_data.get("layer", 0))
                aurocs.append(layer_data.get("auroc", 0.5))
            ax.plot(layers, aurocs, "b-", linewidth=2, label="Qwen 2.5 7B")
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Cross-validated AUROC", fontsize=12)
            ax.legend(fontsize=10)
    ax.set_title("A. Hazard encoding by layer", fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, 1.0)

    # Panel B: Feature sparsity comparison
    ax = axes[1]
    # Steerling concepts vs SAE features
    sparsity_data = {}
    if "sae_summary" in arms and isinstance(arms["sae_summary"], dict):
        sae_info = arms["sae_summary"]
        if "feature_sparsity" in sae_info:
            sparsity_data["SAE features"] = sae_info["feature_sparsity"]
    # Add concept sparsity from existing analysis
    concept_sparsity_path = OUTPUT_DIR / "concept_pca_summary.json"
    if concept_sparsity_path.exists():
        with open(concept_sparsity_path) as f:
            pca = json.load(f)
            if "sparsity" in pca:
                sparsity_data["CBM concepts"] = pca["sparsity"]

    if sparsity_data:
        names = list(sparsity_data.keys())
        values = [sparsity_data[n] for n in names]
        colors = ["#4CAF50", "#2196F3"][:len(names)]
        ax.bar(names, values, color=colors, width=0.5)
        ax.set_ylabel("Active features (%)", fontsize=12)
    ax.set_title("B. Feature sparsity", fontsize=13, fontweight="bold")

    # Panel C: Correction vs disruption trade-off
    ax = axes[2]
    method_labels = {
        "Concept Steering": ("#2196F3", "o"),
        "SAE Steering": ("#4CAF50", "s"),
        "Activation Patching": ("#FF9800", "^"),
        "TSV Steering": ("#9C27B0", "D"),
        "Prompt Engineering": ("#607D8B", "x"),
    }
    # Plot each method's correction vs disruption at best alpha
    # This will be populated from the summary data
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Break-even")
    ax.set_xlabel("FN correction rate", fontsize=12)
    ax.set_ylabel("TP disruption rate", fontsize=12)
    ax.set_title("C. Correction-disruption trade-off", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"figure3_method_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure3_method_comparison")


def main():
    print("Loading results from all arms...")
    arms = load_arm_results()
    print(f"  Loaded arms: {list(arms.keys())}")

    if not arms:
        print("ERROR: No results found. Run the Qwen pipeline first.")
        return

    # Extract per-case metrics and compute summary
    rows = extract_correction_metrics(arms)
    print(f"  Total per-case rows: {len(rows)}")

    if rows:
        summary_df = compute_summary_table(rows)
    else:
        summary_df = build_summary_from_files(arms)
        if summary_df is None:
            summary_df = pd.DataFrame()

    if not summary_df.empty:
        # Save summary table
        csv_path = TABLES_DIR / "comparative_summary.csv"
        # Drop internal columns for CSV
        export_cols = [c for c in summary_df.columns if not c.startswith("_")]
        summary_df[export_cols].to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

        # Generate figures
        plot_comparative_correction(summary_df)

    # Method comparison figure (uses probe/logit lens data)
    plot_method_comparison(arms)

    # Save comprehensive analysis JSON
    analysis = {
        "arms_loaded": list(arms.keys()),
        "n_per_case_rows": len(rows),
    }

    # Add key findings
    if "probes" in arms:
        analysis["best_probe_layer"] = arms["probes"].get("best_layer", {})
    if "logit_lens" in arms:
        analysis["logit_lens"] = arms["logit_lens"]
    if "sae_summary" in arms:
        analysis["sae_summary"] = arms["sae_summary"]
    if "tsv_summary" in arms:
        analysis["tsv_summary"] = arms["tsv_summary"]

    analysis_path = OUTPUT_DIR / "comparative_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Saved {analysis_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 70)
    if not summary_df.empty:
        for method in summary_df["Method"].unique():
            method_df = summary_df[summary_df["Method"] == method]
            best_idx = method_df["_fn_rate"].idxmax() if "_fn_rate" in method_df.columns else 0
            best = method_df.loc[best_idx] if best_idx in method_df.index else method_df.iloc[0]
            print(f"\n{method}:")
            if "_fn_rate" in method_df.columns:
                print(f"  Best FN correction: {best['_fn_rate']:.1%}")
                print(f"  TP disruption:      {best['_tp_rate']:.1%}")
                print(f"  Condition:          {best.get('Condition', 'N/A')}")
    else:
        print("No correction metrics available yet.")

    print("\nComparative analysis complete.")


if __name__ == "__main__":
    main()
