#!/usr/bin/env python3
"""Step 2: Run Steerling inference on demographic-varied vignettes.

Following Giorgi et al. (Nature Medicine 2025), each physician-created
vignette is presented with 34 demographic prefixes (33 sociodemographic
groups + 1 control). Concept activations are extracted for each variation
to identify differentially activated concepts.

Runtime: ~12-20 hours for 200 cases x 34 variations = 6,800 inferences.
"""

import json
import sys

import torch
from tqdm import tqdm

from config import (
    DEMOGRAPHIC_VARIATIONS,
    EMERGENCY_KEYWORDS,
    OUTPUT_DIR,
    PHYSICIAN_TEST,
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


def load_physician_cases():
    """Load physician-created test cases."""
    with open(PHYSICIAN_TEST) as f:
        data = json.load(f)
    cases = []
    for c in data:
        cases.append({
            "case_id": c.get("name", c.get("study_id", "")),
            "message": c["message"],
            "detection_truth": c["detection_truth"],
            "action_truth": c.get("action_truth", "None"),
            "hazard_category": c.get("hazard_category", "Unknown"),
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
    """Construct prompt with optional demographic prefix."""
    parts = [SYSTEM_PROMPT, ""]
    if demographic_prefix:
        parts.append(demographic_prefix)
    parts.append(f"Patient message: {message}")
    parts.append("")
    parts.append("Assessment:")
    return "\n".join(parts)


def main():
    cases = load_physician_cases()
    n_variations = len(DEMOGRAPHIC_VARIATIONS)
    total = len(cases) * n_variations
    print(f"Cases: {len(cases)}, Variations: {n_variations}, Total: {total}")

    device = select_device()
    print(f"\nDevice: {device}")
    print("Loading Steerling-8B...")
    try:
        from steerling import SteerlingGenerator, GenerationConfig

        generator = SteerlingGenerator.from_pretrained(
            STEERLING_MODEL, device=device
        )
    except ImportError:
        print("ERROR: steerling package not installed.")
        sys.exit(1)

    gen_config = GenerationConfig(max_new_tokens=300, seed=SEED)
    results = []

    for case in tqdm(cases, desc="Cases"):
        for var_name, var_prefix in DEMOGRAPHIC_VARIATIONS.items():
            prompt = build_prompt(case["message"], var_prefix)

            # Generate response
            text = generator.generate(prompt, gen_config)

            # Extract concept activations via native API
            concept_data = {}
            try:
                known_mean = generator.get_embeddings(
                    prompt, pooling="mean", embedding_type="known"
                )
                concept_data["known_mean"] = known_mean.cpu().float().numpy().tolist()

                known_none = generator.get_embeddings(
                    prompt, pooling="none", embedding_type="known"
                )
                known_np = known_none.cpu().float().numpy()
                concept_data["known_max"] = known_np.max(axis=0).tolist()
            except Exception as e:
                print(f"  Concept error for {var_name}: {e}")

            triage = parse_triage_response(
                text, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
            )

            results.append({
                "case_id": case["case_id"],
                "variation": var_name,
                "demographic_prefix": var_prefix or "",
                "detection_truth": case["detection_truth"],
                "action_truth": case["action_truth"],
                "hazard_category": case["hazard_category"],
                "steerling_response": text,
                "steerling_detection": triage["detection"],
                "steerling_action": triage["action"],
                "known_concept_mean": concept_data.get("known_mean"),
                "known_concept_max": concept_data.get("known_max"),
            })

    out_path = OUTPUT_DIR / "demographic_variation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Summary: detection rate by demographic group
    from collections import defaultdict
    import numpy as np

    by_var = defaultdict(list)
    for r in results:
        by_var[r["variation"]].append(r["steerling_detection"])

    print("\nDetection rate by demographic variation:")
    ref_key = "race_white" if "race_white" in by_var else list(by_var.keys())[0]
    ref_rate = np.mean(by_var[ref_key])
    for var_name in sorted(by_var.keys()):
        rate = np.mean(by_var[var_name])
        diff = rate - ref_rate
        marker = " (ref)" if var_name == ref_key else f" (delta={diff:+.3f})"
        print(f"  {var_name}: {rate:.3f}{marker}")


if __name__ == "__main__":
    main()
