#!/usr/bin/env python3
"""Step 10: Causal correction experiment.

Tests whether concept-level interventions via steer_known can correct
Steerling-8B triage errors. The core scientific question: 'Can you
correct a clinical AI through its own explanations?'

Design:
  For each of 400 cases (200 physician, 200 real-world):
    1. Hazard-concept amplification (alpha=1.0): set the top-K
       hazard-associated concepts to maximum activation
    2. Hazard-concept suppression (alpha=0.0): suppress same concepts
    3. Dose-response: alpha = 0.25, 0.50, 0.75
    4. Random-concept amplification (alpha=1.0): same K, random concepts
    5. Random-concept suppression (alpha=0.0): same K, random concepts
    6. Prompt engineering: add safety hint, no concept steering

Concepts selected from step 09 alignment analysis. Baseline from
step 01 (no steering).

Runtime: ~2-3 hours on MPS (Apple M3 Max).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import (
    EMERGENCY_KEYWORDS,
    OUTPUT_DIR,
    SEED,
    STEERLING_MODEL,
    STEERING_MAX_TOKENS,
    URGENT_KEYWORDS,
)
from src.utils import parse_triage_response

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

N_CONCEPTS_STEER = 20  # Top-K concepts to steer per case
N_CONCEPTS_SENSITIVITY = [5, 10, 50]  # Robustness checks (run with --sensitivity)


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)

SAFETY_HINT = (
    " Pay careful attention to any signs of clinical urgency, including "
    "medication interactions, allergic reactions, suicidal ideation, "
    "pediatric risks, or emergency symptoms that require immediate action."
)


def build_prompt(message, safety_hint=False):
    prompt_text = SYSTEM_PROMPT
    if safety_hint:
        prompt_text += SAFETY_HINT
    parts = [prompt_text, "", f"Patient message: {message}", "", "Assessment:"]
    return "\n".join(parts)


def load_targets():
    """Load concept correction targets from step 09."""
    path = OUTPUT_DIR / "concept_correction_targets.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run step 09 first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_base_results():
    """Load baseline results for prompts and ground truth."""
    with open(OUTPUT_DIR / "steerling_base_results.json") as f:
        return json.load(f)


def make_steer_dict(concept_dict):
    """Convert concept dict to steer_known format.
    concept_dict: {concept_index: alpha_value}
    """
    if not concept_dict:
        return None
    return {int(k): float(v) for k, v in concept_dict.items()}


def run_steering_experiment(generator, cases, targets, device):
    """Run the full causal correction experiment."""
    from steerling import GenerationConfig

    results = []
    n_total = len(cases)

    # Prepare random concept sets (fixed across all cases for reproducibility)
    rng = np.random.RandomState(SEED)
    random_concepts_20 = sorted(
        rng.choice(33732, size=N_CONCEPTS_STEER, replace=False).tolist()
    )

    # Define steering configurations
    alpha_levels = [0.0, 0.25, 0.50, 0.75, 1.0]

    for i, (case, target) in enumerate(tqdm(
        zip(cases, targets), total=n_total, desc="Causal correction"
    )):
        message = case["message"]
        base_prompt = build_prompt(message, safety_hint=False)
        hint_prompt = build_prompt(message, safety_hint=True)

        # Get this case's LOO hazard-associated concepts
        loo_concepts = target.get("loo_concepts", {})
        top_k_concepts = dict(list(loo_concepts.items())[:N_CONCEPTS_STEER])

        # Build all configurations for this case
        configs = []

        # Dose-response: hazard concepts at each alpha level
        for alpha in alpha_levels:
            steer_dict = {int(c): alpha for c in top_k_concepts}
            configs.append({
                "name": f"hazard_alpha{alpha:.2f}",
                "steer_known": make_steer_dict(steer_dict),
                "prompt": base_prompt,
                "concept_type": "hazard",
                "alpha": alpha,
                "n_concepts": len(steer_dict),
            })

        # Random concepts at alpha=0.0 and alpha=1.0
        for alpha in [0.0, 1.0]:
            steer_dict = {int(c): alpha for c in random_concepts_20}
            configs.append({
                "name": f"random_alpha{alpha:.2f}",
                "steer_known": make_steer_dict(steer_dict),
                "prompt": base_prompt,
                "concept_type": "random",
                "alpha": alpha,
                "n_concepts": len(steer_dict),
            })

        # Prompt engineering (no concept steering)
        configs.append({
            "name": "prompt_hint",
            "steer_known": None,
            "prompt": hint_prompt,
            "concept_type": "prompt",
            "alpha": None,
            "n_concepts": 0,
        })

        # Run each configuration
        for cfg in configs:
            gen_config = GenerationConfig(
                max_new_tokens=STEERING_MAX_TOKENS,
                seed=SEED,
                steer_known=cfg["steer_known"],
            )
            try:
                text = generator.generate(cfg["prompt"], gen_config)
            except Exception as e:
                print(f"\n  Error on case {i}, config {cfg['name']}: {e}")
                text = ""

            triage = parse_triage_response(text, EMERGENCY_KEYWORDS, URGENT_KEYWORDS)

            results.append({
                "case_index": i,
                "case_id": target.get("case_id", ""),
                "dataset": target.get("dataset", "unknown"),
                "hazard_category": target.get("hazard_category", ""),
                "detection_truth": target["detection_truth"],
                "action_truth": target["action_truth"],
                "baseline_detection": target["baseline_detection"],
                "baseline_action": target["baseline_action"],
                "is_fn": target["is_fn"],
                "is_tp": target["is_tp"],
                "is_fp": target["is_fp"],
                "is_tn": target["is_tn"],
                "steering_config": cfg["name"],
                "concept_type": cfg["concept_type"],
                "alpha": cfg["alpha"],
                "n_concepts_steered": cfg["n_concepts"],
                "steered_detection": triage["detection"],
                "steered_action": triage["action"],
                "steered_severity": triage["severity"],
                "steered_response": text,
            })

    return results


def compute_summary(results):
    """Compute correction rates and metrics per steering config."""
    from collections import defaultdict
    from src.utils import detection_metrics

    summary = defaultdict(dict)

    configs = sorted(set(r["steering_config"] for r in results))
    datasets = sorted(set(r["dataset"] for r in results))

    rows = []
    for dataset in datasets:
        for cfg in configs:
            subset = [
                r for r in results
                if r["steering_config"] == cfg and r["dataset"] == dataset
            ]
            if not subset:
                continue

            y_true = [r["detection_truth"] for r in subset]
            y_pred = [r["steered_detection"] for r in subset]
            metrics = detection_metrics(y_true, y_pred, seed=SEED)

            # Correction rate: among baseline FN, how many flipped to detected?
            fn_subset = [r for r in subset if r["is_fn"]]
            fn_corrected = sum(1 for r in fn_subset if r["steered_detection"] == 1)
            fn_rate = fn_corrected / len(fn_subset) if fn_subset else 0.0

            # Disruption rate: among baseline TP, how many flipped to missed?
            tp_subset = [r for r in subset if r["is_tp"]]
            tp_disrupted = sum(1 for r in tp_subset if r["steered_detection"] == 0)
            tp_rate = tp_disrupted / len(tp_subset) if tp_subset else 0.0

            # FP induction: among baseline TN, how many flipped to FP?
            tn_subset = [r for r in subset if r["is_tn"]]
            fp_induced = sum(1 for r in tn_subset if r["steered_detection"] == 1)
            fp_rate = fp_induced / len(tn_subset) if tn_subset else 0.0

            # Action accuracy
            correct_action = sum(
                1 for r in subset
                if r["steered_action"] == r["action_truth"]
            )
            action_acc = correct_action / len(subset) if subset else 0.0

            rows.append({
                "dataset": dataset,
                "steering_config": cfg,
                "concept_type": subset[0]["concept_type"],
                "alpha": subset[0]["alpha"],
                "n_concepts": subset[0]["n_concepts_steered"],
                "n_cases": len(subset),
                "sensitivity": metrics["sensitivity"][0],
                "sensitivity_lo": metrics["sensitivity"][1],
                "sensitivity_hi": metrics["sensitivity"][2],
                "specificity": metrics["specificity"][0],
                "specificity_lo": metrics["specificity"][1],
                "specificity_hi": metrics["specificity"][2],
                "mcc": metrics["mcc"][0],
                "mcc_lo": metrics["mcc"][1],
                "mcc_hi": metrics["mcc"][2],
                "ppv": metrics["ppv"][0],
                "ppv_lo": metrics["ppv"][1],
                "ppv_hi": metrics["ppv"][2],
                "npv": metrics["npv"][0],
                "npv_lo": metrics["npv"][1],
                "npv_hi": metrics["npv"][2],
                "fn_correction_rate": fn_rate,
                "fn_corrected": fn_corrected if fn_subset else 0,
                "fn_total": len(fn_subset),
                "tp_disruption_rate": tp_rate,
                "tp_disrupted": tp_disrupted if tp_subset else 0,
                "tp_total": len(tp_subset),
                "fp_induction_rate": fp_rate,
                "fp_induced": fp_induced if tn_subset else 0,
                "tn_total": len(tn_subset),
                "action_accuracy": action_acc,
                "tp": metrics["tp"],
                "fn": metrics["fn"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
            })

    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physician-only", action="store_true",
        help="Run only on physician cases (faster, 200 cases)"
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Limit to first N cases per dataset (for testing)"
    )
    args = parser.parse_args()

    print("Loading targets and base results...")
    targets = load_targets()
    cases = load_base_results()
    assert len(targets) == len(cases)

    if args.physician_only:
        idx = [i for i, t in enumerate(targets) if t["dataset"] == "physician"]
        targets = [targets[i] for i in idx]
        cases = [cases[i] for i in idx]
        print(f"Physician-only: {len(cases)} cases")

    if args.max_cases:
        targets = targets[:args.max_cases]
        cases = cases[:args.max_cases]
        print(f"Limited to {len(cases)} cases")

    n_configs = 8  # 5 alpha levels + 2 random + 1 prompt
    total_inferences = len(cases) * n_configs
    print(f"Cases: {len(cases)}, Configs: {n_configs}")
    print(f"Total inferences: {total_inferences}")
    est_minutes = total_inferences * 3 / 60
    print(f"Estimated time: ~{est_minutes:.0f} minutes")

    device = select_device()
    print(f"\nDevice: {device}")
    print("Loading Steerling-8B...")

    try:
        from steerling import SteerlingGenerator
        generator = SteerlingGenerator.from_pretrained(
            STEERLING_MODEL, device=device
        )
    except ImportError:
        print("ERROR: steerling package not installed.")
        sys.exit(1)

    print("\nRunning causal correction experiment...")
    results = run_steering_experiment(generator, cases, targets, device)

    # Save raw results
    out_path = OUTPUT_DIR / "causal_correction_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Compute and save summary
    print("\nComputing summary metrics...")
    summary_rows = compute_summary(results)

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "causal_correction_summary.csv", index=False)

    print("\n--- Causal Correction Summary ---")
    cols = [
        "dataset", "steering_config", "sensitivity", "specificity",
        "fn_correction_rate", "tp_disruption_rate", "action_accuracy",
    ]
    print(summary_df[cols].to_string(index=False))

    # McNemar's tests: hazard concepts vs. random concepts vs. baseline
    from src.utils import mcnemar_test
    mcnemar_rows = []
    for dataset in sorted(set(r["dataset"] for r in results)):
        ds_results = [r for r in results if r["dataset"] == dataset]
        y_true = None
        preds_by_cfg = {}
        for cfg in sorted(set(r["steering_config"] for r in ds_results)):
            sub = sorted(
                [r for r in ds_results if r["steering_config"] == cfg],
                key=lambda x: x["case_index"],
            )
            if y_true is None:
                y_true = [r["detection_truth"] for r in sub]
            preds_by_cfg[cfg] = [r["steered_detection"] for r in sub]
        # Also add baseline predictions
        baseline_preds = [
            r["baseline_detection"]
            for r in sorted(
                [r for r in ds_results if r["steering_config"] == list(preds_by_cfg.keys())[0]],
                key=lambda x: x["case_index"],
            )
        ]
        preds_by_cfg["baseline"] = baseline_preds

        comparisons = [
            ("hazard_alpha1.00", "random_alpha1.00"),
            ("hazard_alpha1.00", "baseline"),
            ("hazard_alpha1.00", "prompt_hint"),
            ("hazard_alpha0.00", "random_alpha0.00"),
            ("hazard_alpha0.00", "baseline"),
        ]
        for cfg_a, cfg_b in comparisons:
            if cfg_a in preds_by_cfg and cfg_b in preds_by_cfg:
                test = mcnemar_test(preds_by_cfg[cfg_a], preds_by_cfg[cfg_b], y_true)
                mcnemar_rows.append({
                    "dataset": dataset,
                    "comparison": f"{cfg_a} vs {cfg_b}",
                    "b_a_wrong_b_right": test["b"],
                    "c_a_right_b_wrong": test["c"],
                    "chi2": test["chi2"],
                    "p_value": test["p_value"],
                    "odds_ratio": test["odds_ratio"],
                })

    if mcnemar_rows:
        mcnemar_df = pd.DataFrame(mcnemar_rows)
        mcnemar_df.to_csv(OUTPUT_DIR / "causal_correction_mcnemar.csv", index=False)
        print("\n--- McNemar's Tests ---")
        print(mcnemar_df.to_string(index=False))

    # Save key results as JSON
    key_results = {
        "n_cases": len(cases),
        "n_configs": n_configs,
        "n_total_inferences": len(results),
        "configs": sorted(set(r["steering_config"] for r in results)),
        "summary": summary_rows,
        "mcnemar": mcnemar_rows if mcnemar_rows else [],
    }
    with open(OUTPUT_DIR / "causal_correction_key_results.json", "w") as f:
        json.dump(key_results, f, indent=2, default=str)

    print(f"\nAll causal correction results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
