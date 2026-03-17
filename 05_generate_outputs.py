#!/usr/bin/env python3
"""Step 5: Generate all tables and figures for the manuscript.

Reads outputs from steps 01-04 and produces publication-ready tables (CSV)
and figures (PDF/PNG) following the manuscript structure.

Tables:
  Table 1: Study population characteristics
  Table 2: Triage detection performance across models
  Table 3: Top contributing concepts by hazard category
  Table 4: Concepts with differential activation by demographic group
  Table 5: Triage performance before and after concept-level debiasing

Figures:
  Figure 1: Study design flow diagram
  Figure 2: Concept attribution heatmap
  Figure 3: Triage sensitivity and specificity by demographic group
  Figure 4: Effect of concept steering on triage equity
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    EXISTING_RESULTS_DIR,
    FIGURES_DIR,
    OUTPUT_DIR,
    PHYSICIAN_TEST,
    REALWORLD_TEST,
    TABLES_DIR,
)
from src.utils import detection_metrics, format_ci, wilson_ci

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


# ===== TABLE 1: Study Population Characteristics =====

def generate_table1():
    """Demographic and clinical characteristics of the study population."""
    with open(PHYSICIAN_TEST) as f:
        phys = json.load(f)
    with open(REALWORLD_TEST) as f:
        rw = json.load(f)

    rows = []

    # Physician set
    n_phys = len(phys)
    n_phys_hazard = sum(1 for c in phys if c["detection_truth"] == 1)
    n_phys_benign = n_phys - n_phys_hazard

    hazard_cats = {}
    for c in phys:
        cat = c.get("hazard_category", "Unknown")
        hazard_cats[cat] = hazard_cats.get(cat, 0) + 1

    rows.append(["", "Physician-created (N=200)", "Real-world (N=200)"])
    rows.append(["Hazard cases, n (%)",
                 f"{n_phys_hazard} ({100*n_phys_hazard/n_phys:.1f})",
                 f"{sum(1 for c in rw if c.get('ground_truth_detection',0)==1)} "
                 f"({100*sum(1 for c in rw if c.get('ground_truth_detection',0)==1)/len(rw):.1f})"])
    rows.append(["Benign cases, n (%)",
                 f"{n_phys_benign} ({100*n_phys_benign/n_phys:.1f})",
                 f"{sum(1 for c in rw if c.get('ground_truth_detection',0)==0)} "
                 f"({100*sum(1 for c in rw if c.get('ground_truth_detection',0)==0)/len(rw):.1f})"])

    # Real-world demographics
    ages = [c.get("patient_age") for c in rw if c.get("patient_age")]
    sexes = [c.get("patient_sex") for c in rw if c.get("patient_sex")]
    races = [c.get("patient_race") for c in rw if c.get("patient_race")]

    rows.append(["Age, mean (SD)", "N/A (vignettes)",
                 f"{np.mean(ages):.1f} ({np.std(ages):.1f})" if ages else "N/A"])

    sex_counts = {}
    for s in sexes:
        sex_counts[s] = sex_counts.get(s, 0) + 1
    for sex_label in ["F", "M"]:
        n_sex = sex_counts.get(sex_label, 0)
        rows.append([f"Sex: {sex_label}, n (%)", "N/A",
                     f"{n_sex} ({100*n_sex/len(rw):.1f})" if n_sex else "0 (0.0)"])

    race_counts = {}
    for r in races:
        race_counts[r] = race_counts.get(r, 0) + 1
    for race_label in ["White", "Black or African American",
                       "Hispanic or Latino", "Asian", "Other"]:
        n_race = race_counts.get(race_label, 0)
        rows.append([f"Race: {race_label}, n (%)", "N/A",
                     f"{n_race} ({100*n_race/len(rw):.1f})" if n_race else "0 (0.0)"])

    # Hazard categories
    rows.append(["", "", ""])
    rows.append(["Hazard category distribution", "", ""])
    for cat in sorted(hazard_cats.keys()):
        if cat == "benign":
            continue
        rw_cat = sum(1 for c in rw
                     if c.get("ground_truth_hazard_category") == cat)
        rows.append([f"  {cat}", str(hazard_cats[cat]),
                     str(rw_cat) if rw_cat else "0"])

    # Word count
    phys_wc = [len(c["message"].split()) for c in phys]
    rw_wc = [c.get("word_count", len(c["message"].split())) for c in rw]
    rows.append(["Message word count, median (IQR)",
                 f"{np.median(phys_wc):.0f} ({np.percentile(phys_wc,25):.0f}-{np.percentile(phys_wc,75):.0f})",
                 f"{np.median(rw_wc):.0f} ({np.percentile(rw_wc,25):.0f}-{np.percentile(rw_wc,75):.0f})"])

    df = pd.DataFrame(rows, columns=["Characteristic", "Physician-created", "Real-world"])
    df.to_csv(TABLES_DIR / "table1_study_population.csv", index=False)
    print(f"Table 1 saved: {TABLES_DIR / 'table1_study_population.csv'}")
    return df


# ===== TABLE 2: Triage Detection Performance =====

def generate_table2():
    """Triage detection performance for Steerling-8B.

    Reports sensitivity, specificity, and MCC on physician and real-world
    test sets under baseline and concept-level intervention conditions.
    """
    # Load Steerling results
    steerling_path = OUTPUT_DIR / "steerling_base_results.json"
    if not steerling_path.exists():
        print("WARNING: Steerling results not found. Run step 01 first.")
        return pd.DataFrame()

    with open(steerling_path) as f:
        steerling = json.load(f)

    # Load existing baselines
    phys_metrics_path = EXISTING_RESULTS_DIR / "physician_metrics.csv"
    rw_metrics_path = EXISTING_RESULTS_DIR / "realworld_metrics.csv"

    rows = []
    for dataset_label, dataset_name in [("Physician", "physician"),
                                        ("Real-world", "real-world")]:
        subset = [r for r in steerling if r["dataset"] == dataset_name]
        if not subset:
            continue
        y_true = [r["detection_truth"] for r in subset]
        y_pred = [r["steerling_detection"] for r in subset]
        metrics = detection_metrics(y_true, y_pred)

        rows.append({
            "Test set": dataset_label,
            "Model": "Steerling-8B",
            "N": len(subset),
            "Sensitivity": format_ci(metrics["sensitivity"]),
            "Specificity": format_ci(metrics["specificity"]),
            "PPV": format_ci(metrics["ppv"]),
            "NPV": format_ci(metrics["npv"]),
            "MCC": format_ci(metrics["mcc"]),
        })

    # Add existing baselines from CSV if available
    for path, ds_label in [(phys_metrics_path, "Physician"),
                           (rw_metrics_path, "Real-world")]:
        if path.exists():
            baseline_df = pd.read_csv(path)
            for _, brow in baseline_df.iterrows():
                model_name = brow.get("system", brow.get("model", "Unknown"))
                rows.append({
                    "Test set": ds_label,
                    "Model": model_name,
                    "N": int(brow.get("n", brow.get("N", 0))),
                    "Sensitivity": brow.get("sensitivity", ""),
                    "Specificity": brow.get("specificity", ""),
                    "PPV": brow.get("precision", brow.get("ppv", "")),
                    "NPV": brow.get("npv", ""),
                    "MCC": brow.get("mcc", ""),
                })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table2_triage_performance.csv", index=False)
    print(f"Table 2 saved: {TABLES_DIR / 'table2_triage_performance.csv'}")
    return df


# ===== TABLE 3: Top Contributing Concepts =====

def generate_table3():
    """Top contributing concepts by hazard category."""
    path = OUTPUT_DIR / "top_concepts_by_category.csv"
    if not path.exists():
        print("WARNING: Concept analysis not found. Run step 03 first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Pivot: top 5 concepts per category
    top5 = df[df["rank"] <= 5]
    top5.to_csv(TABLES_DIR / "table3_top_concepts.csv", index=False)
    print(f"Table 3 saved: {TABLES_DIR / 'table3_top_concepts.csv'}")
    return top5


# ===== TABLE 4: Differentially Activated Concepts =====

def generate_table4():
    """Concepts with differential activation by demographic group."""
    path = OUTPUT_DIR / "differential_concepts.csv"
    if not path.exists():
        print("WARNING: Differential analysis not found. Run step 03 first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Top 20 by effect size
    top20 = df.nlargest(20, "cohens_d")
    top20.to_csv(TABLES_DIR / "table4_differential_concepts.csv", index=False)
    print(f"Table 4 saved: {TABLES_DIR / 'table4_differential_concepts.csv'}")
    return top20


# ===== TABLE 5: Before/After Concept Steering =====

def generate_table5():
    """Triage performance before and after concept-level debiasing."""
    path = OUTPUT_DIR / "steering_summary.csv"
    if not path.exists():
        print("WARNING: Steering results not found. Run step 04 first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.to_csv(TABLES_DIR / "table5_steering_results.csv", index=False)
    print(f"Table 5 saved: {TABLES_DIR / 'table5_steering_results.csv'}")
    return df


# ===== FIGURE 2: Concept Attribution Heatmap =====

def generate_figure2():
    """Heatmap of top concept activations by hazard category."""
    path = OUTPUT_DIR / "top_concepts_by_category.csv"
    if not path.exists():
        print("WARNING: Concept data not found.")
        return

    df = pd.read_csv(path)
    top5 = df[df["rank"] <= 5]

    categories = sorted(top5["hazard_category"].unique())
    concepts = sorted(top5["concept_index"].unique())

    matrix = np.zeros((len(categories), len(concepts)))
    cat_idx = {c: i for i, c in enumerate(categories)}
    con_idx = {c: i for i, c in enumerate(concepts)}

    for _, row in top5.iterrows():
        ci = cat_idx.get(row["hazard_category"])
        cj = con_idx.get(row["concept_index"])
        if ci is not None and cj is not None:
            matrix[ci, cj] = row["mean_activation"]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        matrix, xticklabels=[f"C{c}" for c in concepts],
        yticklabels=categories, cmap="YlOrRd", ax=ax,
        cbar_kws={"label": "Mean concept activation (arbitrary units)"},
        linewidths=0.5,
    )
    ax.set_xlabel("Concept index")
    ax.set_ylabel("Hazard category")
    ax.set_title("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_concept_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure2_concept_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved: {FIGURES_DIR / 'figure2_concept_heatmap.pdf'}")


# ===== FIGURE 3: Demographic Bias Detection =====

def generate_figure3():
    """Forest plot of demographic disparities in triage detection."""
    path = OUTPUT_DIR / "triage_disparities.csv"
    if not path.exists():
        print("WARNING: Disparity data not found.")
        return

    df = pd.read_csv(path)
    df = df.dropna(subset=["sensitivity"])
    df = df.sort_values("sensitivity")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    # Panel A: Sensitivity by demographic group
    y_pos = range(len(df))
    ax1.errorbar(
        df["sensitivity"], y_pos,
        xerr=[
            df["sensitivity"] - df["sensitivity_lo"],
            df["sensitivity_hi"] - df["sensitivity"],
        ],
        fmt="o", color="steelblue", capsize=3, markersize=5,
    )
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(df["variation"].str.replace("race_", "").str.title())
    ax1.set_xlabel("Sensitivity (95% CI)")
    ax1.set_title("A. Triage sensitivity by demographic group")
    ref_row = df.loc[df["variation"] == "race_white", "sensitivity"]
    if len(ref_row) > 0:
        ax1.axvline(x=ref_row.values[0], color="gray", linestyle="--", alpha=0.5)

    # Panel B: Specificity by demographic group
    df2 = df.dropna(subset=["specificity"])
    y_pos2 = range(len(df2))
    ax2.errorbar(
        df2["specificity"], y_pos2,
        xerr=[
            df2["specificity"] - df2["specificity_lo"],
            df2["specificity_hi"] - df2["specificity"],
        ],
        fmt="s", color="coral", capsize=3, markersize=5,
    )
    ax2.set_xlabel("Specificity (95% CI)")
    ax2.set_title("B. Triage specificity by demographic group")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_demographic_bias.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure3_demographic_bias.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 3 saved: {FIGURES_DIR / 'figure3_demographic_bias.pdf'}")


# ===== FIGURE 4: Steering Effect on Equity =====

def generate_figure4():
    """Before/after steering comparison of triage equity."""
    path = OUTPUT_DIR / "steering_summary.csv"
    if not path.exists():
        print("WARNING: Steering data not found.")
        return

    df = pd.read_csv(path)

    # Filter to key comparisons
    race_vars = ["race_white", "race_black", "race_hispanic"]
    steer_types = ["unsteered", "race_suppressed", "all_bias_suppressed"]

    subset = df[
        df["variation"].isin(race_vars) &
        df["steering"].isin(steer_types)
    ].copy()

    if subset.empty:
        print("WARNING: No data for steering figure.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Sensitivity by race, before/after steering
    colors = {"race_white": "#4477AA", "race_black": "#EE6677",
              "race_hispanic": "#228833"}
    width = 0.25

    for i, steer in enumerate(steer_types):
        steer_data = subset[subset["steering"] == steer]
        x = np.arange(len(race_vars))
        vals = []
        errs_lo = []
        errs_hi = []
        for rv in race_vars:
            row = steer_data[steer_data["variation"] == rv]
            if len(row) == 1:
                vals.append(row.iloc[0]["sensitivity"])
                errs_lo.append(
                    row.iloc[0]["sensitivity"] - row.iloc[0]["sensitivity_lo"]
                )
                errs_hi.append(
                    row.iloc[0]["sensitivity_hi"] - row.iloc[0]["sensitivity"]
                )
            else:
                vals.append(0)
                errs_lo.append(0)
                errs_hi.append(0)

        ax1.bar(
            x + i * width, vals, width, yerr=[errs_lo, errs_hi],
            label=steer.replace("_", " ").title(), capsize=3, alpha=0.8,
        )

    ax1.set_xticks(np.arange(len(race_vars)) + width)
    ax1.set_xticklabels([v.replace("race_", "").title() for v in race_vars])
    ax1.set_ylabel("Sensitivity (95% CI)")
    ax1.set_title("A. Triage sensitivity by race")
    ax1.legend(title="Steering condition")
    ax1.set_ylim(0, 1.05)

    # Panel B: White-Black sensitivity gap
    gaps = []
    gap_labels = []
    for steer in steer_types:
        white_row = subset[
            (subset["steering"] == steer) & (subset["variation"] == "race_white")
        ]
        black_row = subset[
            (subset["steering"] == steer) & (subset["variation"] == "race_black")
        ]
        if len(white_row) == 1 and len(black_row) == 1:
            gap = (
                white_row.iloc[0]["sensitivity"] -
                black_row.iloc[0]["sensitivity"]
            )
            gaps.append(gap)
            gap_labels.append(steer.replace("_", " ").title())

    if gaps:
        bars = ax2.barh(range(len(gaps)), gaps, color=["#BBBBBB", "#EE6677", "#4477AA"])
        ax2.set_yticks(range(len(gaps)))
        ax2.set_yticklabels(gap_labels)
        ax2.set_xlabel("White-Black sensitivity gap (percentage points)")
        ax2.set_title("B. Racial disparity reduction")
        ax2.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_steering_equity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure4_steering_equity.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 4 saved: {FIGURES_DIR / 'figure4_steering_equity.pdf'}")


# ===== FIGURE 1: Study Design Flow Diagram =====

def generate_figure1():
    """Study design flow diagram showing four pipeline stages."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box_style = dict(
        boxstyle="round,pad=0.4", facecolor="#E8F4FD",
        edgecolor="#2C5F8A", linewidth=1.5,
    )
    arrow_props = dict(
        arrowstyle="-|>", color="#2C5F8A", linewidth=1.5,
        connectionstyle="arc3,rad=0",
    )

    # Stage A
    ax.annotate(
        "Stage A\nBase Inference\n400 cases × Steerling-8B\nConcept extraction\n(4,096 concepts)",
        xy=(2.5, 5.5), fontsize=8, ha="center", va="center",
        bbox=box_style,
    )
    # Stage B
    ax.annotate(
        "Stage B\nDemographic Variation\n200 vignettes × 3 race groups\n= 600 inferences\nWhite (ref), Black, Hispanic",
        xy=(7.5, 5.5), fontsize=8, ha="center", va="center",
        bbox=box_style,
    )
    # Stage C
    ax.annotate(
        "Stage C\nBias Detection\nPaired t-tests + Wilcoxon\nBH-FDR correction\nPermutation validation",
        xy=(2.5, 2.0), fontsize=8, ha="center", va="center",
        bbox=box_style,
    )
    # Stage D
    ax.annotate(
        "Stage D\nConcept Steering\nSuppress bias concepts\nPrompt debiasing baseline\nCompensatory encoding check",
        xy=(7.5, 2.0), fontsize=8, ha="center", va="center",
        bbox=box_style,
    )

    # Arrows: A→B, A→C, B→C, C→D
    ax.annotate("", xy=(5.5, 5.5), xytext=(4.0, 5.5), arrowprops=arrow_props)
    ax.annotate("", xy=(2.5, 3.5), xytext=(2.5, 4.3), arrowprops=arrow_props)
    ax.annotate("", xy=(2.5, 3.5), xytext=(7.5, 4.3), arrowprops=arrow_props)
    ax.annotate("", xy=(5.5, 2.0), xytext=(4.0, 2.0), arrowprops=arrow_props)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_study_design.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure1_study_design.png", bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved: {FIGURES_DIR / 'figure1_study_design.pdf'}")


def main():
    print("Generating tables...")
    generate_table1()
    generate_table2()
    generate_table3()
    generate_table4()
    generate_table5()

    print("\nGenerating figures...")
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()

    print("\nAll outputs generated.")


if __name__ == "__main__":
    main()
