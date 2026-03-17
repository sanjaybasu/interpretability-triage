#!/usr/bin/env python3
"""Step 1: Run Steerling-8B inference with concept extraction on triage cases.

Loads physician-created and real-world test cases, runs each through
Steerling-8B, and extracts concept activations from the concept bottleneck
layer using the native get_embeddings API.

Requires: ~18GB memory (Apple Silicon MPS with unified memory, or CUDA GPU).
Runtime: ~2-4 hours for 400 base cases.
"""

import json
import sys

import torch
from tqdm import tqdm

from config import (
    EMERGENCY_KEYWORDS,
    OUTPUT_DIR,
    PHYSICIAN_TEST,
    REALWORLD_TEST,
    SEED,
    STEERLING_MODEL,
    URGENT_KEYWORDS,
)
from src.utils import parse_triage_response

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def select_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_test_cases():
    """Load physician and real-world test cases."""
    with open(PHYSICIAN_TEST) as f:
        physician = json.load(f)
    with open(REALWORLD_TEST) as f:
        realworld = json.load(f)
    cases = []
    for c in physician:
        cases.append({
            "case_id": c.get("name", c.get("study_id", "")),
            "message": c["message"],
            "detection_truth": c["detection_truth"],
            "action_truth": c.get("action_truth", "None"),
            "hazard_category": c.get("hazard_category", "Unknown"),
            "dataset": "physician",
            "patient_race": None,
            "patient_sex": None,
            "patient_age": None,
        })
    for c in realworld:
        cases.append({
            "case_id": c.get("case_id", ""),
            "message": c["message"],
            "detection_truth": c["ground_truth_detection"],
            "action_truth": c.get("ground_truth_action", "None"),
            "hazard_category": c.get("ground_truth_hazard_category", "Unknown"),
            "dataset": "real-world",
            "patient_race": c.get("patient_race"),
            "patient_sex": c.get("patient_sex"),
            "patient_age": c.get("patient_age"),
        })
    return cases


SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)


def build_prompt(message, demographic_prefix=None):
    """Construct prompt for Steerling inference."""
    parts = [SYSTEM_PROMPT, ""]
    if demographic_prefix:
        parts.append(demographic_prefix)
    parts.append(f"Patient message: {message}")
    parts.append("")
    parts.append("Assessment:")
    return "\n".join(parts)


def extract_concept_activations(generator, prompt):
    """Extract concept activations using the native get_embeddings API.

    Returns mean-pooled and per-token known concept activations.
    """
    result = {}
    try:
        # Mean-pooled concept vector (primary analysis)
        known_mean = generator.get_embeddings(
            prompt, pooling="mean", embedding_type="known"
        )
        result["known_concept_activations"] = known_mean.cpu().float().numpy().tolist()

        # Max-pooled (sensitivity analysis)
        known_none = generator.get_embeddings(
            prompt, pooling="none", embedding_type="known"
        )
        known_np = known_none.cpu().float().numpy()
        result["known_concept_activations_max"] = known_np.max(axis=0).tolist()

        # Unknown pathway (for completeness)
        try:
            unk_mean = generator.get_embeddings(
                prompt, pooling="mean", embedding_type="unknown"
            )
            result["unknown_concept_activations"] = unk_mean.cpu().float().numpy().tolist()
        except Exception:
            pass

    except Exception as e:
        print(f"  Concept extraction error: {e}")

    return result


def main():
    print("Loading test cases...")
    cases = load_test_cases()
    print(f"  Physician cases: {sum(1 for c in cases if c['dataset']=='physician')}")
    print(f"  Real-world cases: {sum(1 for c in cases if c['dataset']=='real-world')}")

    device = select_device()
    print(f"\nDevice: {device}")
    print(f"Loading Steerling-8B from {STEERLING_MODEL}...")

    try:
        from steerling import SteerlingGenerator, GenerationConfig

        generator = SteerlingGenerator.from_pretrained(
            STEERLING_MODEL, device=device
        )
        print("  Loaded via SteerlingGenerator")
    except ImportError:
        print("  ERROR: steerling package not installed. Run: pip install steerling")
        sys.exit(1)

    gen_config = GenerationConfig(max_new_tokens=300, seed=SEED)
    results = []

    for case in tqdm(cases, desc="Running inference"):
        prompt = build_prompt(case["message"])

        # Generate response
        generated_text = generator.generate(prompt, gen_config)

        # Extract concept activations
        concept_result = extract_concept_activations(generator, prompt)

        # Parse triage decision
        triage = parse_triage_response(
            generated_text, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
        )

        results.append({
            **case,
            "prompt": prompt,
            "steerling_response": generated_text,
            "steerling_detection": triage["detection"],
            "steerling_severity": triage["severity"],
            "steerling_action": triage["action"],
            **concept_result,
        })

    out_path = OUTPUT_DIR / "steerling_base_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Summary
    phys = [r for r in results if r["dataset"] == "physician"]
    rw = [r for r in results if r["dataset"] == "real-world"]
    for label, subset in [("Physician", phys), ("Real-world", rw)]:
        y_true = [r["detection_truth"] for r in subset]
        y_pred = [r["steerling_detection"] for r in subset]
        tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
        fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
        tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
        fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\n{label} set (N={len(subset)}):")
        print(f"  Sensitivity: {sens:.3f}")
        print(f"  Specificity: {spec:.3f}")


if __name__ == "__main__":
    main()
