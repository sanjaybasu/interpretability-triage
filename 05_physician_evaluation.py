#!/usr/bin/env python3
"""Step 5: Physician evaluation of concept-level explanations.

Generates evaluation materials for 3 blinded independent physicians to assess
whether concept bottleneck explanations add clinical value to triage decisions.

Protocol:
  1. Select 30 stratified cases (hazard × category × model correctness)
  2. For each case, present:
     a. Clinical vignette (blinded to demographics)
     b. Model triage response (text)
     c. Concept "explanation" (top-5 activated concepts with proxy labels)
  3. Physicians rate on 5-point Likert scales
  4. Compute inter-rater reliability (Krippendorff's alpha)

Uses outputs from steps 01-03 plus concept_proxy_labels.json.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, SEED

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR = OUTPUT_DIR / "physician_evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

N_EVAL_CASES = 30


def load_data():
    """Load base results and hidden dimension proxy labels."""
    with open(OUTPUT_DIR / "steerling_base_results.json") as f:
        results = json.load(f)

    # Load hidden dimension proxy labels (from token embedding projection)
    proxy_path = OUTPUT_DIR / "hidden_dim_proxy_labels.json"
    if proxy_path.exists():
        with open(proxy_path) as f:
            proxy_labels = json.load(f)
    else:
        proxy_labels = {}

    return results, proxy_labels


def get_top_k_concepts(activations, k=5):
    """Get top-k most activated hidden dimensions (as concept proxies).

    Note: These are hidden feature dimensions (0-4095), NOT Atlas concept
    indices. This distinction is central to the interpretability critique.
    """
    arr = np.array(activations)
    top_k_idx = np.argsort(np.abs(arr))[::-1][:k]
    return [(int(idx), float(arr[idx])) for idx in top_k_idx]


def select_cases(results):
    """Stratified selection: balance hazard/benign, categories, correct/incorrect."""
    hazard = [r for r in results if r["detection_truth"] == 1]
    benign = [r for r in results if r["detection_truth"] == 0]

    # Correct vs incorrect
    hazard_correct = [r for r in hazard if r["steerling_detection"] == 1]
    hazard_incorrect = [r for r in hazard if r["steerling_detection"] == 0]
    benign_correct = [r for r in benign if r["steerling_detection"] == 0]
    benign_incorrect = [r for r in benign if r["steerling_detection"] == 1]

    selected = []

    # 10 true hazard, correct detection (diverse categories)
    cats = list({r["hazard_category"] for r in hazard_correct})
    random.shuffle(cats)
    for cat in cats[:10]:
        pool = [r for r in hazard_correct if r["hazard_category"] == cat]
        if pool:
            selected.append(random.choice(pool))
    while len(selected) < 10 and hazard_correct:
        r = random.choice(hazard_correct)
        if r not in selected:
            selected.append(r)

    # 5 true hazard, missed by model (false negatives)
    random.shuffle(hazard_incorrect)
    for r in hazard_incorrect[:5]:
        selected.append(r)

    # 10 benign, correct (true negatives)
    random.shuffle(benign_correct)
    for r in benign_correct[:10]:
        selected.append(r)

    # 5 benign, false alarms (false positives)
    random.shuffle(benign_incorrect)
    for r in benign_incorrect[:5]:
        selected.append(r)

    random.shuffle(selected)
    return selected[:N_EVAL_CASES]


def build_concept_explanation(case, proxy_labels):
    """Build concept-level explanation card for a case.

    Uses hidden dimension proxy labels derived from token embedding projection.
    These are deliberately presented as-is to test whether physicians can
    derive clinical meaning from concept bottleneck representations.
    """
    activations = case.get("known_concept_activations", [])
    if not activations:
        return {"top_concepts": [], "explanation_text": "No concept data available."}

    top5 = get_top_k_concepts(activations, k=5)

    explanation_items = []
    for dim_idx, value in top5:
        proxy = proxy_labels.get(str(dim_idx), {})
        proxy_tokens = proxy.get("top3_tokens", [])
        label = proxy_tokens[0] if proxy_tokens else f"dim_{dim_idx}"

        explanation_items.append({
            "hidden_dimension": dim_idx,
            "activation_value": round(value, 4),
            "proxy_label": label,
            "proxy_alternatives": proxy_tokens[1:3] if len(proxy_tokens) > 1 else [],
        })

    # Build human-readable explanation text
    lines = ["Top-5 activated internal concept features:"]
    for i, item in enumerate(explanation_items):
        direction = "+" if item["activation_value"] > 0 else "-"
        alts = ", ".join(f"'{a}'" for a in item["proxy_alternatives"])
        lines.append(
            f"  {i+1}. Feature #{item['hidden_dimension']} "
            f"(nearest tokens: '{item['proxy_label']}'"
            f"{', ' + alts if alts else ''}): "
            f"{direction}{abs(item['activation_value']):.3f}"
        )

    return {
        "top_concepts": explanation_items,
        "explanation_text": "\n".join(lines),
    }


def generate_evaluation_form(cases, proxy_labels):
    """Generate JSON evaluation form for physicians."""
    eval_cases = []

    for i, case in enumerate(cases):
        explanation = build_concept_explanation(case, proxy_labels)

        eval_case = {
            "case_number": i + 1,
            "case_id": case["case_id"],
            # Clinical vignette (no demographics)
            "clinical_vignette": case["message"],
            "hazard_category": case["hazard_category"],
            # Model output
            "model_response": case["steerling_response"],
            "model_detection": case["steerling_detection"],
            # Concept explanation
            "concept_explanation": explanation["explanation_text"],
            "concept_details": explanation["top_concepts"],
            # Ground truth (hidden from evaluators)
            "_ground_truth": {
                "detection_truth": case["detection_truth"],
                "action_truth": case.get("action_truth", "None"),
                "model_correct": (
                    case["steerling_detection"] == case["detection_truth"]
                ),
            },
            # Rating fields (to be filled by physicians)
            "ratings": {
                "q1_response_quality": {
                    "prompt": (
                        "How appropriate is the model's triage response "
                        "for this clinical scenario?"
                    ),
                    "scale": "1=Completely inappropriate, 2=Mostly inappropriate, "
                             "3=Neutral, 4=Mostly appropriate, 5=Completely appropriate",
                    "value": None,
                },
                "q2_explanation_clarity": {
                    "prompt": (
                        "How clear and understandable is the concept-level "
                        "explanation provided?"
                    ),
                    "scale": "1=Completely unclear, 2=Mostly unclear, "
                             "3=Neutral, 4=Mostly clear, 5=Completely clear",
                    "value": None,
                },
                "q3_explanation_helpfulness": {
                    "prompt": (
                        "Does the concept explanation help you understand "
                        "WHY the model reached this triage conclusion?"
                    ),
                    "scale": "1=Not at all, 2=Slightly, 3=Moderately, "
                             "4=Considerably, 5=Completely",
                    "value": None,
                },
                "q4_trust_increase": {
                    "prompt": (
                        "Does seeing the concept explanation increase your "
                        "trust in the model's triage decision?"
                    ),
                    "scale": "1=Decreases trust, 2=No change, 3=Slightly increases, "
                             "4=Moderately increases, 5=Greatly increases",
                    "value": None,
                },
                "q5_clinical_meaningfulness": {
                    "prompt": (
                        "Are the concept labels clinically meaningful? "
                        "Do they correspond to recognizable clinical features?"
                    ),
                    "scale": "Yes / No / Unclear",
                    "value": None,
                },
                "q6_would_change_decision": {
                    "prompt": (
                        "Would seeing this concept explanation change your "
                        "own clinical decision about this patient?"
                    ),
                    "scale": "Yes / No",
                    "value": None,
                },
                "q7_free_text": {
                    "prompt": "Any additional comments on this case?",
                    "value": None,
                },
            },
        }
        eval_cases.append(eval_case)

    return eval_cases


def generate_evaluation_protocol():
    """Generate the full evaluation protocol document."""
    protocol = {
        "title": (
            "Physician Evaluation of Concept Bottleneck Explanations "
            "for Clinical Triage"
        ),
        "version": "1.0",
        "date": "2026-03-05",
        "investigators": "Basu S, Berkowitz SA",
        "overview": (
            "This evaluation assesses whether concept-level explanations from "
            "a concept bottleneck language model (Steerling-8B) provide "
            "clinically useful interpretability for triage decisions in a "
            "Medicaid population health context.\n\n"
            "IMPORTANT CONTEXT: Steerling-8B uses a concept bottleneck "
            "architecture with 33,732 supervised concepts. However, the "
            "concept vocabulary (Atlas) is proprietary and labels are not "
            "publicly available. The 'proxy labels' shown were generated by "
            "projecting concept embeddings onto the nearest vocabulary tokens "
            "via cosine similarity (mean similarity = 0.31, indicating weak "
            "correspondence). This is the best-available interpretation."
        ),
        "evaluator_instructions": (
            "1. You will review 30 clinical triage cases.\n"
            "2. For each case, you will see:\n"
            "   a. A patient message/clinical vignette\n"
            "   b. The AI model's triage response\n"
            "   c. A 'concept explanation' showing the top-5 internal "
            "concepts that most strongly activated for this input\n"
            "3. Rate each case on the provided scales.\n"
            "4. You are blinded to:\n"
            "   - The correct triage answer\n"
            "   - Whether the model got the answer right or wrong\n"
            "   - Patient demographics\n"
            "5. Complete the evaluation independently; do not discuss "
            "cases with other evaluators.\n"
            "6. There are no right or wrong answers for the ratings; "
            "we want your honest clinical assessment."
        ),
        "analysis_plan": {
            "primary_outcome": (
                "Mean rating for Q3 (explanation helpfulness), compared "
                "to neutral (3.0) via one-sample t-test."
            ),
            "secondary_outcomes": [
                "Inter-rater reliability (Krippendorff's alpha) for Q1-Q4",
                "Proportion answering 'No' to Q5 (clinical meaningfulness)",
                "Proportion answering 'No' to Q6 (would change decision)",
                "Correlation between Q2 (clarity) and Q3 (helpfulness)",
            ],
            "sample_size_justification": (
                "30 cases × 3 raters = 90 observations. Detectable effect "
                "for one-sample t-test (H0: μ=3.0 vs H1: μ≠3.0) with "
                "α=0.05, power=0.80: δ ≥ 0.53 (medium effect, Cohen's d). "
                "For inter-rater reliability, 30 items × 3 raters is "
                "sufficient for Krippendorff's alpha estimation."
            ),
        },
    }
    return protocol


def main():
    results, proxy_labels = load_data()
    print(f"Loaded {len(results)} base results, {len(proxy_labels)} proxy labels")

    # Select stratified cases
    cases = select_cases(results)
    print(f"Selected {len(cases)} evaluation cases:")

    # Summarize selection
    truth_counts = Counter(c["detection_truth"] for c in cases)
    correct_counts = Counter(
        c["steerling_detection"] == c["detection_truth"] for c in cases
    )
    cat_counts = Counter(c["hazard_category"] for c in cases)
    print(f"  Hazard: {truth_counts.get(1, 0)}, Benign: {truth_counts.get(0, 0)}")
    print(f"  Correct: {correct_counts.get(True, 0)}, "
          f"Incorrect: {correct_counts.get(False, 0)}")
    print(f"  Categories: {len(cat_counts)} unique")

    # Generate evaluation form
    eval_cases = generate_evaluation_form(cases, proxy_labels)

    # Save evaluation materials
    with open(EVAL_DIR / "evaluation_cases.json", "w") as f:
        json.dump(eval_cases, f, indent=2)
    print(f"\nSaved {len(eval_cases)} cases to {EVAL_DIR}/evaluation_cases.json")

    # Generate protocol
    protocol = generate_evaluation_protocol()
    with open(EVAL_DIR / "evaluation_protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"Saved protocol to {EVAL_DIR}/evaluation_protocol.json")

    # Generate answer key (for analysis after evaluation)
    answer_key = []
    for ec in eval_cases:
        answer_key.append({
            "case_number": ec["case_number"],
            "case_id": ec["case_id"],
            "detection_truth": ec["_ground_truth"]["detection_truth"],
            "action_truth": ec["_ground_truth"]["action_truth"],
            "model_correct": ec["_ground_truth"]["model_correct"],
            "hazard_category": ec["hazard_category"],
        })
    with open(EVAL_DIR / "answer_key.json", "w") as f:
        json.dump(answer_key, f, indent=2)

    # Generate readable evaluation form (Markdown)
    lines = [
        "# Physician Evaluation: Concept Bottleneck Explanations for Triage",
        "",
        f"**Evaluator ID**: _______________",
        f"**Date**: _______________",
        "",
        "## Instructions",
        "",
        protocol["evaluator_instructions"],
        "",
        "---",
        "",
    ]

    for ec in eval_cases:
        lines.append(f"## Case {ec['case_number']}")
        lines.append("")
        lines.append("### Clinical Vignette")
        lines.append(f"> {ec['clinical_vignette']}")
        lines.append("")
        lines.append("### Model Triage Response")
        resp = ec["model_response"][:500]
        if len(ec["model_response"]) > 500:
            resp += "..."
        lines.append(f"> {resp}")
        lines.append("")
        lines.append("### Concept Explanation")
        lines.append("```")
        lines.append(ec["concept_explanation"])
        lines.append("```")
        lines.append("")

        for qkey, qdata in ec["ratings"].items():
            if qkey == "q7_free_text":
                lines.append(f"**{qdata['prompt']}**")
                lines.append("")
                lines.append("_______________________________________________")
            elif qkey in ("q5_clinical_meaningfulness", "q6_would_change_decision"):
                lines.append(f"**{qdata['prompt']}**")
                lines.append(f"  {qdata['scale']}")
                lines.append("")
            else:
                lines.append(f"**{qdata['prompt']}**")
                lines.append(f"  {qdata['scale']}")
                lines.append("  Rating: ____")
            lines.append("")

        lines.append("---")
        lines.append("")

    md_path = EVAL_DIR / "evaluation_form.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved readable form to {md_path}")

    # Print sample case
    print(f"\n--- Sample Case ---")
    sample = eval_cases[0]
    print(f"Vignette: {sample['clinical_vignette'][:200]}...")
    print(f"Response: {sample['model_response'][:200]}...")
    print(f"Explanation:\n{sample['concept_explanation']}")


if __name__ == "__main__":
    main()
