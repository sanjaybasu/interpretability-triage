#!/usr/bin/env python3
"""Run TP-mean concept correction experiment on Modal cloud GPU.

Tests the concept correction hypothesis directly: for each false-negative case,
set the top-K hazard-associated concepts to the mean activation observed in
true-positive cases of the same category. This uses leave-one-out to prevent
circular concept selection.

This is the strongest possible test of concept-level correction because:
  (a) Target values are in-distribution (observed TP means, not arbitrary alpha)
  (b) It directly asks: "If concepts matched a correctly-detected case, would
      the model detect the hazard?"
"""

import json
import os
import sys
from pathlib import Path

import modal

app = modal.App("concept-tp-correction-v1")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch>=2.8.0",
        "triton>=3.0.0",
        "numpy>=2.3.0",
        "safetensors>=0.4.0",
        "tiktoken>=0.8.0",
        "pydantic>=2.10.0",
        "huggingface_hub>=0.20.0",
        "steerling>=0.1.2",
        "tqdm",
    )
)

model_volume = modal.Volume.from_name("steerling-cache", create_if_missing=True)

SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/model-cache": model_volume},
)
def run_batch(batch_json: str) -> str:
    """Run a batch of cases with TP-mean correction. Returns JSON results."""
    from steerling import SteerlingGenerator, GenerationConfig

    batch = json.loads(batch_json)
    batch_id = batch["batch_id"]
    cases = batch["cases"]
    targets = batch["targets"]
    emergency_kw = batch["emergency_kw"]
    urgent_kw = batch["urgent_kw"]
    seed = batch["seed"]
    max_tokens = batch["max_tokens"]

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    print(f"Batch {batch_id}: Loading model...")
    generator = SteerlingGenerator.from_pretrained(
        "guidelabs/steerling-8b", device="cuda"
    )
    model_volume.commit()
    print(f"Batch {batch_id}: Processing {len(cases)} cases x 2 configs.")

    def build_prompt(message):
        return "\n".join([SYSTEM_PROMPT, "", f"Patient message: {message}", "", "Assessment:"])

    def parse_response(text):
        text_lower = text.lower()
        if any(kw in text_lower for kw in emergency_kw):
            return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
        if any(kw in text_lower for kw in urgent_kw):
            return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
        return {"detection": 0, "severity": "Benign", "action": "None"}

    results = []

    for i, (case, target) in enumerate(zip(cases, targets)):
        case_idx = target["global_index"]
        message = case["message"]
        prompt = build_prompt(message)

        # Two conditions: tp_correction and observed_max
        configs = []

        # TP-mean correction: set concepts to mean activation from TP cases
        tp_steer = target.get("tp_mean_concepts")
        if tp_steer:
            configs.append({
                "name": "tp_correction",
                "steer_known": {int(c): float(v) for c, v in tp_steer.items()},
                "concept_type": "tp_correction",
                "alpha": None,
                "n_concepts": len(tp_steer),
            })

        # Observed maximum: set concepts to 95th percentile of observed activation
        obs_max_steer = target.get("observed_max_concepts")
        if obs_max_steer:
            configs.append({
                "name": "observed_max",
                "steer_known": {int(c): float(v) for c, v in obs_max_steer.items()},
                "concept_type": "observed_max",
                "alpha": None,
                "n_concepts": len(obs_max_steer),
            })

        for cfg in configs:
            gen_config = GenerationConfig(
                max_new_tokens=max_tokens,
                seed=seed,
                steer_known=cfg["steer_known"],
            )
            try:
                text = generator.generate(prompt, gen_config)
            except Exception as e:
                print(f"  Error case {case_idx}, {cfg['name']}: {e}")
                text = ""

            triage = parse_response(text)
            results.append({
                "case_index": case_idx,
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

        print(f"  Batch {batch_id}: Case {case_idx} done ({i+1}/{len(cases)})")

    print(f"Batch {batch_id}: Complete. {len(results)} results.")
    return json.dumps(results)


@app.local_entrypoint()
def main():
    """Compute TP-mean targets, dispatch batches, collect results."""
    import numpy as np

    from config import (
        EMERGENCY_KEYWORDS,
        OUTPUT_DIR,
        SEED,
        STEERING_MAX_TOKENS,
        URGENT_KEYWORDS,
    )

    print("Loading local data...")
    with open(OUTPUT_DIR / "steerling_base_results.json") as f:
        all_cases = json.load(f)
    with open(OUTPUT_DIR / "concept_correction_targets.json") as f:
        all_targets = json.load(f)
    weights = np.load(OUTPUT_DIR / "base_concept_weights.npy")

    n_cases = len(all_cases)
    print(f"Total cases: {n_cases}")

    # Compute TP-mean concept targets using LOO
    print("Computing TP-mean correction targets...")
    from collections import defaultdict

    # Index cases by category and detection status
    cat_tp_idx = defaultdict(list)  # category -> list of TP case indices
    all_tp_idx = []
    for i, r in enumerate(all_cases):
        if r["detection_truth"] == 1 and r["steerling_detection"] == 1:
            cat = r.get("hazard_category", "unknown")
            cat_tp_idx[cat].append(i)
            all_tp_idx.append(i)

    print(f"  Total TP cases: {len(all_tp_idx)}")
    for cat, indices in sorted(cat_tp_idx.items()):
        print(f"    {cat}: {len(indices)} TP")

    # For each case, compute TP-mean targets for the LOO-selected concepts
    for i, t in enumerate(all_targets):
        t["global_index"] = i
        loo_concepts = t.get("loo_concepts", {})
        top_k = list(loo_concepts.keys())[:20]
        concept_ids = [int(c) for c in top_k]

        cat = t.get("hazard_category", "unknown")

        # LOO: use TP cases from same category, excluding case i
        cat_tp = [j for j in cat_tp_idx.get(cat, []) if j != i]
        if len(cat_tp) < 2:
            # Fallback to all TP cases (excluding case i)
            cat_tp = [j for j in all_tp_idx if j != i]

        if cat_tp and concept_ids:
            tp_mean_vals = weights[cat_tp][:, concept_ids].mean(axis=0)
            t["tp_mean_concepts"] = {
                str(c): float(v) for c, v in zip(concept_ids, tp_mean_vals)
            }
            # Also compute 95th percentile of observed activation
            obs_p95 = np.percentile(weights[cat_tp][:, concept_ids], 95, axis=0)
            t["observed_max_concepts"] = {
                str(c): float(v) for c, v in zip(concept_ids, obs_p95)
            }
        else:
            t["tp_mean_concepts"] = {}
            t["observed_max_concepts"] = {}

    # Summarize TP-mean target values
    all_tp_vals = []
    for t in all_targets:
        all_tp_vals.extend(t.get("tp_mean_concepts", {}).values())
    if all_tp_vals:
        all_tp_vals = np.array(all_tp_vals)
        print(f"\nTP-mean target values: mean={all_tp_vals.mean():.6f}, "
              f"median={np.median(all_tp_vals):.6f}, max={all_tp_vals.max():.6f}")

    # Batch and dispatch
    BATCH_SIZE = 10
    batches = []
    for batch_id, start in enumerate(range(0, n_cases, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, n_cases)
        batch = {
            "batch_id": batch_id,
            "cases": all_cases[start:end],
            "targets": all_targets[start:end],
            "emergency_kw": EMERGENCY_KEYWORDS,
            "urgent_kw": URGENT_KEYWORDS,
            "seed": SEED,
            "max_tokens": STEERING_MAX_TOKENS,
        }
        batches.append(json.dumps(batch, default=str))

    print(f"\nDispatching {len(batches)} batches to Modal GPUs...")

    all_results = []
    completed = 0
    partial_path = OUTPUT_DIR / "tp_correction_partial.json"
    for result_json in run_batch.map(batches, order_outputs=False):
        batch_results = json.loads(result_json)
        all_results.extend(batch_results)
        completed += 1
        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        n_done = len(set(r["case_index"] for r in all_results))
        print(f"  Batch {completed}/{len(batches)} done. Cases: {n_done}/{n_cases}")

    all_results.sort(key=lambda r: (r["case_index"], r["steering_config"]))
    out_path = OUTPUT_DIR / "tp_correction_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")

    if partial_path.exists():
        partial_path.unlink()

    # Quick summary
    for cfg_name in ["tp_correction", "observed_max"]:
        sub = [r for r in all_results if r["steering_config"] == cfg_name]
        if not sub:
            continue
        fn_sub = [r for r in sub if r["is_fn"]]
        tp_sub = [r for r in sub if r["is_tp"]]
        fn_corr = sum(1 for r in fn_sub if r["steered_detection"] == 1)
        tp_disr = sum(1 for r in tp_sub if r["steered_detection"] == 0)
        print(f"\n{cfg_name}: FN corrected={fn_corr}/{len(fn_sub)}, "
              f"TP disrupted={tp_disr}/{len(tp_sub)}")
