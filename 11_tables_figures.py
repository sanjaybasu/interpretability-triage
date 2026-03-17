#!/usr/bin/env python3
"""Step 11: Generate tables and figures for manuscript.

Produces all exhibits for the causal faithfulness paper:
  Table 1: Study population and vignette characteristics
  Table 2: Baseline triage performance (Steerling-8B)
  Table 3: Concept-hazard alignment summary
  Table 4: Causal correction results (primary analysis)
  Figure 1: Dose-response curve (sensitivity vs. alpha)
  Figure 2: Correction and disruption rates by intervention type
  Figure 3: Category-level correction heatmap

Appendix:
  eTable 1: Full concept-hazard alignment
  eTable 2: Sensitivity analysis (K=5,10,50)
  eFigure 1: Concept activation distributions (TP vs. FN)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import FIGURES_DIR, OUTPUT_DIR, TABLES_DIR, SEED
from src.utils import (
    bca_bootstrap_ci,
    detection_metrics,
    format_ci,
    wilson_ci,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def load_all_data():
    """Load all results needed for tables and figures."""
    data = {}

    with open(OUTPUT_DIR / "steerling_base_results.json") as f:
        data["base"] = json.load(f)

    with open(OUTPUT_DIR / "concept_hazard_alignment.json") as f:
        data["alignment"] = json.load(f)

    path = OUTPUT_DIR / "concept_analysis_summary.json"
    if path.exists():
        with open(path) as f:
            data["concept_summary"] = json.load(f)

    path = OUTPUT_DIR / "causal_correction_results.json"
    if path.exists():
        with open(path) as f:
            data["correction"] = json.load(f)

    path = OUTPUT_DIR / "causal_correction_summary.csv"
    if path.exists():
        data["correction_summary"] = pd.read_csv(path)

    path = OUTPUT_DIR / "tp_correction_results.json"
    if path.exists():
        with open(path) as f:
            data["tp_correction"] = json.load(f)

    return data


def table1_study_population(data):
    """Table 1: Characteristics of physician-created and real-world datasets."""
    base = data["base"]
    phys = [r for r in base if r.get("dataset") == "physician"]
    rw = [r for r in base if r.get("dataset") != "physician"]

    rows = []

    # Physician dataset
    p_hazards = sum(1 for r in phys if r["detection_truth"] == 1)
    p_benign = sum(1 for r in phys if r["detection_truth"] == 0)
    p_911 = sum(1 for r in phys if r.get("action_truth") == "Call 911/988")
    p_none = sum(1 for r in phys if r.get("action_truth") == "None")

    from collections import Counter
    p_cats = Counter(r.get("hazard_category", "unknown") for r in phys
                     if r["detection_truth"] == 1)

    rows.append({
        "Characteristic": "Total vignettes, N",
        "Physician-created (N=200)": str(len(phys)),
        "Real-world (N=200)": str(len(rw)),
    })
    rows.append({
        "Characteristic": "Hazard-containing vignettes, N (%)",
        "Physician-created (N=200)": f"{p_hazards} ({p_hazards/len(phys)*100:.0f})",
        "Real-world (N=200)": f"{sum(1 for r in rw if r['detection_truth']==1)} "
                              f"({sum(1 for r in rw if r['detection_truth']==1)/len(rw)*100:.0f})",
    })
    rows.append({
        "Characteristic": "Benign vignettes, N (%)",
        "Physician-created (N=200)": f"{p_benign} ({p_benign/len(phys)*100:.0f})",
        "Real-world (N=200)": f"{sum(1 for r in rw if r['detection_truth']==0)} "
                              f"({sum(1 for r in rw if r['detection_truth']==0)/len(rw)*100:.0f})",
    })
    rows.append({
        "Characteristic": "Action: Call 911/988, N (%)",
        "Physician-created (N=200)": f"{p_911} ({p_911/len(phys)*100:.0f})",
        "Real-world (N=200)": f"{sum(1 for r in rw if r.get('action_truth', r.get('ground_truth_action'))=='Call 911/988')} "
                              f"({sum(1 for r in rw if r.get('action_truth', r.get('ground_truth_action'))=='Call 911/988')/len(rw)*100:.0f})",
    })
    rows.append({
        "Characteristic": "Action: None, N (%)",
        "Physician-created (N=200)": f"{p_none} ({p_none/len(phys)*100:.0f})",
        "Real-world (N=200)": f"{sum(1 for r in rw if r.get('action_truth', r.get('ground_truth_action'))=='None')} "
                              f"({sum(1 for r in rw if r.get('action_truth', r.get('ground_truth_action'))=='None')/len(rw)*100:.0f})",
    })

    # Hazard categories
    rows.append({
        "Characteristic": "Hazard categories",
        "Physician-created (N=200)": "",
        "Real-world (N=200)": "",
    })
    for cat, n in p_cats.most_common():
        rw_n = sum(1 for r in rw if r.get("hazard_category", r.get("ground_truth_hazard_category")) == cat
                   and r["detection_truth"] == 1)
        rows.append({
            "Characteristic": f"  {cat.replace('_', ' ').title()}, N",
            "Physician-created (N=200)": str(n),
            "Real-world (N=200)": str(rw_n) if rw_n > 0 else "-",
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table1_study_population.csv", index=False)
    return df


def table2_baseline_performance(data):
    """Table 2: Baseline triage performance by dataset."""
    base = data["base"]
    rows = []

    for dataset_name, filter_fn in [
        ("Physician-created", lambda r: r.get("dataset") == "physician"),
        ("Real-world", lambda r: r.get("dataset") in ("real-world", "realworld") or r.get("dataset") not in ("physician", None)),
        ("Combined", lambda r: True),
    ]:
        subset = [r for r in base if filter_fn(r)]
        y_true = [r["detection_truth"] for r in subset]
        y_pred = [r["steerling_detection"] for r in subset]
        metrics = detection_metrics(y_true, y_pred, seed=SEED)

        # Action accuracy among detected cases
        detected = [r for r in subset if r["steerling_detection"] == 1]
        action_correct = sum(
            1 for r in detected
            if r.get("steerling_action") == r.get(
                "action_truth", r.get("ground_truth_action", "None")
            )
        )
        action_n = len(detected) if detected else 1
        action_ci = wilson_ci(action_correct, action_n)

        rows.append({
            "Dataset": dataset_name,
            "N": len(subset),
            "Prevalence": f"{sum(y_true)}/{len(subset)} ({sum(y_true)/len(subset)*100:.0f}%)",
            "Sensitivity (95% CI)": format_ci(metrics["sensitivity"]),
            "Specificity (95% CI)": format_ci(metrics["specificity"]),
            "PPV (95% CI)": format_ci(metrics["ppv"]),
            "NPV (95% CI)": format_ci(metrics["npv"]),
            "MCC (95% CI)": format_ci(metrics["mcc"]),
            "Action accuracy (95% CI)": format_ci(action_ci),
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table2_baseline_performance.csv", index=False)
    return df


def table3_concept_alignment(data):
    """Table 3: Concept-hazard alignment summary."""
    alignment = data["alignment"]
    rows = []

    for cat in sorted(alignment.keys()):
        info = alignment[cat]
        if cat == "_any_hazard":
            label = "Any hazard (global)"
        else:
            label = cat.replace("_", " ").title()

        top_d = info["top_concepts"][0]["effect_size"] if info["top_concepts"] else 0
        top_q = info["top_concepts"][0]["q_value"] if info["top_concepts"] else 1

        rows.append({
            "Hazard category": label,
            "N cases": info["n_positive"],
            "FDR-significant concepts": f"{info['n_significant']:,}",
            "% of 33,732 concepts": f"{info['n_significant']/33732*100:.1f}",
            "Top concept effect size (d)": f"{top_d:.2f}",
            "Top concept q-value": f"{top_q:.1e}" if top_q > 0 else "<1e-10",
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table3_concept_alignment.csv", index=False)
    return df


def table4_causal_correction(data):
    """Table 4: Causal correction results (primary outcome)."""
    if "correction_summary" not in data:
        print("  Skipping Table 4: no correction results yet")
        return None

    df = data["correction_summary"]
    rows = []

    # Add baseline row from base results
    for dataset in ["physician", "real-world"]:
        dataset_label = "Physician" if dataset == "physician" else "Real-world"
        base_subset = [
            r for r in data["base"]
            if (r.get("dataset") == dataset if dataset != "real-world"
                else r.get("dataset") != "physician")
        ]
        if not base_subset:
            continue
        y_true = [r["detection_truth"] for r in base_subset]
        y_pred = [r["steerling_detection"] for r in base_subset]
        metrics = detection_metrics(y_true, y_pred, seed=SEED)

        rows.append({
            "Dataset": dataset_label,
            "Intervention": "None (baseline)",
            "Concept type": "-",
            "Alpha": "-",
            "Sensitivity (95% CI)": format_ci(metrics["sensitivity"]),
            "Specificity (95% CI)": format_ci(metrics["specificity"]),
            "MCC (95% CI)": format_ci(metrics["mcc"]),
            "FN correction rate": "-",
            "TP disruption rate": "-",
            "FP induction rate": "-",
        })

    # TP-mean correction rows (from tp_correction_results.json)
    if "tp_correction" in data:
        tp_results = data["tp_correction"]
        for steering_cfg, label in [
            ("tp_correction", "TP-mean correction"),
            ("observed_max", "TP P95 correction"),
        ]:
            for dataset, dataset_label in [
                ("physician", "Physician"),
                ("real-world", "Real-world"),
            ]:
                subset = [
                    r for r in tp_results
                    if r["steering_config"] == steering_cfg
                    and r["dataset"] == dataset
                ]
                if not subset:
                    continue
                y_true = [r["detection_truth"] for r in subset]
                y_steered = [r["steered_detection"] for r in subset]
                metrics = detection_metrics(y_true, y_steered, seed=SEED)

                # Compute correction/disruption counts
                fn_cases = [r for r in subset if r["is_fn"]]
                tp_cases = [r for r in subset if r["is_tp"]]
                tn_cases = [r for r in subset if r["is_tn"]]
                fn_corrected = sum(1 for r in fn_cases if r["steered_detection"] == 1)
                tp_disrupted = sum(1 for r in tp_cases if r["steered_detection"] == 0)
                fp_induced = sum(1 for r in tn_cases if r["steered_detection"] == 1)

                fn_total = len(fn_cases)
                tp_total = len(tp_cases)
                tn_total = len(tn_cases)

                fn_str = f"{fn_corrected}/{fn_total} ({fn_corrected/fn_total:.1%})" if fn_total > 0 else "-"
                tp_str = f"{tp_disrupted}/{tp_total} ({tp_disrupted/tp_total:.1%})" if tp_total > 0 else "-"
                fp_str = f"{fp_induced}/{tn_total} ({fp_induced/tn_total:.1%})" if tn_total > 0 else "-"

                rows.append({
                    "Dataset": dataset_label,
                    "Intervention": label,
                    "Concept type": steering_cfg,
                    "Alpha": "TP-mean*" if steering_cfg == "tp_correction" else "TP-P95*",
                    "Sensitivity (95% CI)": format_ci(metrics["sensitivity"]),
                    "Specificity (95% CI)": format_ci(metrics["specificity"]),
                    "MCC (95% CI)": format_ci(metrics["mcc"]),
                    "FN correction rate": fn_str,
                    "TP disruption rate": tp_str,
                    "FP induction rate": fp_str,
                })

    # OOD intervention rows (from causal_correction_summary.csv)
    for _, row in df.iterrows():
        ds = "Physician" if row["dataset"] == "physician" else "Real-world"
        alpha_str = f"{row['alpha']:.2f}" if pd.notna(row["alpha"]) else "-"

        sens_ci = format_ci((row["sensitivity"], row["sensitivity_lo"], row["sensitivity_hi"]))
        spec_ci = format_ci((row["specificity"], row["specificity_lo"], row["specificity_hi"]))
        mcc_ci = format_ci((row["mcc"], row["mcc_lo"], row["mcc_hi"]))

        fn_str = f"{row['fn_corrected']}/{row['fn_total']} ({row['fn_correction_rate']:.1%})" if row["fn_total"] > 0 else "-"
        tp_str = f"{row['tp_disrupted']}/{row['tp_total']} ({row['tp_disruption_rate']:.1%})" if row["tp_total"] > 0 else "-"
        fp_str = f"{row['fp_induced']}/{row['tn_total']} ({row['fp_induction_rate']:.1%})" if row["tn_total"] > 0 else "-"

        rows.append({
            "Dataset": ds,
            "Intervention": row["steering_config"],
            "Concept type": row["concept_type"],
            "Alpha": alpha_str,
            "Sensitivity (95% CI)": sens_ci,
            "Specificity (95% CI)": spec_ci,
            "MCC (95% CI)": mcc_ci,
            "FN correction rate": fn_str,
            "TP disruption rate": tp_str,
            "FP induction rate": fp_str,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "table4_causal_correction.csv", index=False)
    return result


def figure1_dose_response(data):
    """Figure 1: Sensitivity as a function of concept activation level."""
    if "correction_summary" not in data:
        print("  Skipping Figure 1: no correction results yet")
        return

    df = data["correction_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

    for ax_i, (dataset, label) in enumerate([
        ("physician", "Physician-created"),
        ("real-world", "Real-world"),
    ]):
        ax = axes[ax_i]
        ds = df[df["dataset"] == dataset]
        if ds.empty:
            ds = df[df["dataset"] != "physician"] if dataset == "real-world" else ds

        # Hazard concepts dose-response
        hazard = ds[ds["concept_type"] == "hazard"].sort_values("alpha")
        if not hazard.empty:
            ax.errorbar(
                hazard["alpha"], hazard["sensitivity"],
                yerr=[
                    hazard["sensitivity"] - hazard["sensitivity_lo"],
                    hazard["sensitivity_hi"] - hazard["sensitivity"],
                ],
                marker="o", capsize=3, label="Hazard concepts",
                color="#2166ac", linewidth=1.5,
            )

        # Random concepts
        rand = ds[ds["concept_type"] == "random"]
        for _, row in rand.iterrows():
            ax.errorbar(
                row["alpha"], row["sensitivity"],
                yerr=[[row["sensitivity"] - row["sensitivity_lo"]],
                      [row["sensitivity_hi"] - row["sensitivity"]]],
                marker="s", capsize=3, color="#b2182b",
                label="Random concepts" if _ == rand.index[0] else None,
            )

        # Prompt hint
        prompt = ds[ds["concept_type"] == "prompt"]
        if not prompt.empty:
            row = prompt.iloc[0]
            ax.axhline(
                row["sensitivity"], color="#4daf4a", linestyle="--",
                linewidth=1, label="Prompt engineering",
            )
            ax.axhspan(
                row["sensitivity_lo"], row["sensitivity_hi"],
                alpha=0.15, color="#4daf4a",
            )

        # Baseline
        base_subset = [
            r for r in data["base"]
            if (r.get("dataset") == dataset if dataset != "real-world"
                else r.get("dataset") != "physician")
        ]
        y_true = [r["detection_truth"] for r in base_subset]
        y_pred = [r["steerling_detection"] for r in base_subset]
        base_metrics = detection_metrics(y_true, y_pred, seed=SEED)
        ax.axhline(
            base_metrics["sensitivity"][0], color="gray",
            linestyle=":", linewidth=1, label="Baseline (no steering)",
        )

        ax.set_xlabel("Concept activation level (alpha)")
        ax.set_title(label)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1)
        if ax_i == 0:
            ax.set_ylabel("Sensitivity")
            ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_dose_response.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure1_dose_response.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figure1_dose_response")


def figure2_correction_disruption(data):
    """Figure 2: Correction and disruption rates by intervention type."""
    if "correction_summary" not in data:
        print("  Skipping Figure 2: no correction results yet")
        return

    df = data["correction_summary"]
    phys = df[df["dataset"] == "physician"]
    if phys.empty:
        phys = df

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Panel A: FN correction rates
    ax = axes[0]
    configs = ["hazard_alpha1.00", "random_alpha1.00", "prompt_hint"]
    labels = ["Hazard concepts\n(alpha=1.0)", "Random concepts\n(alpha=1.0)", "Prompt\nengineering"]
    colors = ["#2166ac", "#b2182b", "#4daf4a"]

    for i, (cfg, lbl, col) in enumerate(zip(configs, labels, colors)):
        row = phys[phys["steering_config"] == cfg]
        if not row.empty:
            row = row.iloc[0]
            ci = wilson_ci(int(row["fn_corrected"]), int(row["fn_total"]))
            ax.bar(i, ci[0], color=col, width=0.6, alpha=0.8)
            ax.errorbar(
                i, ci[0],
                yerr=[[ci[0] - ci[1]], [ci[2] - ci[0]]],
                fmt="none", capsize=5, color="black",
            )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("FN correction rate")
    ax.set_title("A. Correcting missed hazards")
    ax.set_ylim(0, 1)

    # Panel B: TP disruption rates
    ax = axes[1]
    configs_supp = ["hazard_alpha0.00", "random_alpha0.00"]
    labels_supp = ["Hazard concepts\n(alpha=0.0)", "Random concepts\n(alpha=0.0)"]
    colors_supp = ["#2166ac", "#b2182b"]

    for i, (cfg, lbl, col) in enumerate(zip(configs_supp, labels_supp, colors_supp)):
        row = phys[phys["steering_config"] == cfg]
        if not row.empty:
            row = row.iloc[0]
            ci = wilson_ci(int(row["tp_disrupted"]), int(row["tp_total"]))
            ax.bar(i, ci[0], color=col, width=0.6, alpha=0.8)
            ax.errorbar(
                i, ci[0],
                yerr=[[ci[0] - ci[1]], [ci[2] - ci[0]]],
                fmt="none", capsize=5, color="black",
            )

    ax.set_xticks(range(len(labels_supp)))
    ax.set_xticklabels(labels_supp, fontsize=7)
    ax.set_ylabel("TP disruption rate")
    ax.set_title("B. Disrupting correct detections")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_correction_disruption.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure2_correction_disruption.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figure2_correction_disruption")


def figure3_category_heatmap(data):
    """Figure 3: Category-level correction rates (heatmap)."""
    if "correction" not in data:
        print("  Skipping Figure 3: no correction results yet")
        return

    results = data["correction"]
    phys = [r for r in results if r["dataset"] == "physician"]

    # Compute per-category correction rate for hazard_alpha1.00
    from collections import defaultdict
    cat_fn = defaultdict(lambda: {"total": 0, "corrected": 0})
    for r in phys:
        if r["steering_config"] == "hazard_alpha1.00" and r["is_fn"]:
            cat = r["hazard_category"]
            cat_fn[cat]["total"] += 1
            if r["steered_detection"] == 1:
                cat_fn[cat]["corrected"] += 1

    if not cat_fn:
        print("  No FN cases to plot for Figure 3")
        return

    cats = sorted(cat_fn.keys())
    rates = [cat_fn[c]["corrected"] / cat_fn[c]["total"] if cat_fn[c]["total"] > 0 else 0 for c in cats]
    totals = [cat_fn[c]["total"] for c in cats]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = range(len(cats))
    bars = ax.barh(
        y_pos, rates, color="#2166ac", alpha=0.8, height=0.6
    )

    for i, (r, t) in enumerate(zip(rates, totals)):
        ax.text(r + 0.02, i, f"{r:.0%} (n={t})", va="center", fontsize=7)

    labels = [c.replace("_", " ").title() for c in cats]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("FN correction rate (hazard concepts, alpha=1.0)")
    ax.set_xlim(0, 1.15)
    ax.set_title("Correction rate by hazard category")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_category_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure3_category_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved figure3_category_heatmap")


def main():
    print("Loading all data...")
    data = load_all_data()

    print("\n--- Table 1: Study Population ---")
    t1 = table1_study_population(data)
    print(t1.to_string(index=False))

    print("\n--- Table 2: Baseline Performance ---")
    t2 = table2_baseline_performance(data)
    print(t2.to_string(index=False))

    print("\n--- Table 3: Concept-Hazard Alignment ---")
    t3 = table3_concept_alignment(data)
    print(t3.to_string(index=False))

    print("\n--- Table 4: Causal Correction ---")
    t4 = table4_causal_correction(data)
    if t4 is not None:
        print(t4.to_string(index=False))

    print("\n--- Figures ---")
    figure1_dose_response(data)
    figure2_correction_disruption(data)
    figure3_category_heatmap(data)

    print(f"\nAll tables saved to {TABLES_DIR}/")
    print(f"All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
