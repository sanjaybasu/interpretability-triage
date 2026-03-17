#!/usr/bin/env python3
"""Step 2b: Extract proper 33,732-d concept activation weights.

The original step 02 extracted 4,096-d composed known features via
get_embeddings(embedding_type="known"). These are hidden-space features,
NOT individual concept activations.

This script:
  1. Loads the Steerling model
  2. Re-extracts hidden states via get_embeddings(embedding_type="hidden")
  3. Computes concept logits = hidden @ concept_predictor^T → 33,732-d (after slicing 12 padding rows)
  4. Applies sigmoid → per-concept activation weights in [0,1]
  5. Saves proper concept activation vectors for all inferences

Also extracts base inference hidden states for the L1 analysis.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from config import (
    DEMOGRAPHIC_VARIATIONS,
    OUTPUT_DIR,
    PHYSICIAN_TEST,
    STEERLING_MODEL,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CACHE = (
    Path.home()
    / ".cache/huggingface/hub/models--guidelabs--steerling-8b"
    / "snapshots/337e00164c67b3e458de12430246bd9e633568f7"
)

N_REAL_CONCEPTS = 33732  # Actual Atlas concepts (rest is padding)

SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)


def build_prompt(message, demographic_prefix=None):
    parts = [SYSTEM_PROMPT, ""]
    if demographic_prefix:
        parts.append(demographic_prefix)
    parts.append(f"Patient message: {message}")
    parts.append("")
    parts.append("Assessment:")
    return "\n".join(parts)


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = select_device()
    print(f"Device: {device}")

    # Load concept predictor matrix from cached weights
    print("Loading concept predictor matrix...")
    shard4 = load_file(str(MODEL_CACHE / "model-00004-of-00004.safetensors"))
    predictor = shard4["known_head.concept_predictor.weight"].float().cpu()
    # Only use real concepts (not padding)
    predictor = predictor[:N_REAL_CONCEPTS]  # (33732, 4096)
    print(f"  Predictor shape: {predictor.shape}")

    # Load model
    print("Loading Steerling-8B...")
    from steerling import SteerlingGenerator
    generator = SteerlingGenerator.from_pretrained(STEERLING_MODEL, device=device)

    # --- Phase 1: Base inference concept weights ---
    print("\n=== Phase 1: Base inference (400 cases) ===")
    with open(OUTPUT_DIR / "steerling_base_results.json") as f:
        base_results = json.load(f)

    base_concept_weights = []
    timings = []
    for i, case in enumerate(tqdm(base_results, desc="Base cases")):
        prompt = case["prompt"]
        t0 = time.time()
        hidden = generator.get_embeddings(
            prompt, pooling="mean", embedding_type="hidden"
        ).cpu().float()  # (4096,)
        dt = time.time() - t0
        timings.append(dt)

        # Compute concept weights: sigmoid(hidden @ predictor^T)
        logits = torch.mv(predictor, hidden)  # (33732,)
        weights = torch.sigmoid(logits).numpy()  # (33732,) in [0,1]
        base_concept_weights.append(weights)

        if i == 0:
            n_active = (weights > 0.5).sum()
            print(f"  First case: {n_active} concepts > 0.5, "
                  f"extraction time: {dt:.2f}s")

    base_weights_array = np.stack(base_concept_weights)  # (400, 33732)
    np.save(OUTPUT_DIR / "base_concept_weights.npy", base_weights_array)
    print(f"  Saved base concept weights: {base_weights_array.shape}")
    print(f"  Mean extraction time: {np.mean(timings):.2f}s")

    # --- Phase 2: Demographic variation concept weights ---
    print("\n=== Phase 2: Demographic variation (600 inferences) ===")
    with open(PHYSICIAN_TEST) as f:
        physician_cases = json.load(f)
    case_map = {c["name"]: c for c in physician_cases}

    with open(OUTPUT_DIR / "demographic_variation_results.json") as f:
        demo_results = json.load(f)

    demo_concept_weights = []
    demo_metadata = []
    timings = []

    for i, record in enumerate(tqdm(demo_results, desc="Demo variation")):
        case_id = record["case_id"]
        variation = record["variation"]
        prefix = record["demographic_prefix"]

        case = case_map.get(case_id)
        if case is None:
            print(f"  WARNING: case_id {case_id} not found in physician test")
            demo_concept_weights.append(np.zeros(N_REAL_CONCEPTS))
            demo_metadata.append(record)
            continue

        prompt = build_prompt(case["message"], prefix)
        t0 = time.time()
        hidden = generator.get_embeddings(
            prompt, pooling="mean", embedding_type="hidden"
        ).cpu().float()
        dt = time.time() - t0
        timings.append(dt)

        logits = torch.mv(predictor, hidden)
        weights = torch.sigmoid(logits).numpy()
        demo_concept_weights.append(weights)
        demo_metadata.append(record)

        if i == 0:
            n_active = (weights > 0.5).sum()
            print(f"  First case: {n_active} concepts > 0.5, "
                  f"extraction time: {dt:.2f}s")

    demo_weights_array = np.stack(demo_concept_weights)  # (600, 33732)
    np.save(OUTPUT_DIR / "demo_concept_weights.npy", demo_weights_array)
    # Save metadata for matching
    with open(OUTPUT_DIR / "demo_concept_weights_meta.json", "w") as f:
        json.dump([{"case_id": m["case_id"], "variation": m["variation"]}
                   for m in demo_metadata], f)
    print(f"  Saved demo concept weights: {demo_weights_array.shape}")
    print(f"  Mean extraction time: {np.mean(timings):.2f}s")

    # --- Summary statistics ---
    print("\n=== Summary ===")
    print(f"Base concept weights: shape {base_weights_array.shape}")
    print(f"  Mean weight: {base_weights_array.mean():.6f}")
    print(f"  Max weight: {base_weights_array.max():.4f}")
    print(f"  Concepts with mean weight > 0.1: {(base_weights_array.mean(axis=0) > 0.1).sum()}")
    print(f"  Concepts with mean weight > 0.5: {(base_weights_array.mean(axis=0) > 0.5).sum()}")

    print(f"\nDemo concept weights: shape {demo_weights_array.shape}")
    print(f"  Mean weight: {demo_weights_array.mean():.6f}")
    print(f"  Max weight: {demo_weights_array.max():.4f}")
    print(f"  Concepts with mean weight > 0.1: {(demo_weights_array.mean(axis=0) > 0.1).sum()}")
    print(f"  Concepts with mean weight > 0.5: {(demo_weights_array.mean(axis=0) > 0.5).sum()}")

    print(f"\nTotal extraction time: {sum(timings)/60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
