#!/usr/bin/env python3
"""Modal deployment: TSV steering + activation patching (v2: greedy, with responses + FP testing).

Fixes from v1:
  - Greedy decoding (do_sample=False) for reproducibility
  - Saves full text responses for LLM judge re-scoring
  - Tests TN cases for false-positive induction
  - Wilson score CIs and McNemar's test computed in-script

Runs generation-time steering experiments on Qwen 2.5 7B using:
  1. TSV (truthfulness separator vector) at layer 23
  2. Activation patching at critical layer 22
  3. Random-direction control
"""

import json
import modal
import os

app = modal.App("tsv-steering-v2")

volume = modal.Volume.from_name("concept-triage-results", create_if_missing=True)
model_cache = modal.Volume.from_name("gemma2-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "huggingface-hub>=0.26.0",
        "numpy",
        "scipy",
        "scikit-learn",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours
    volumes={"/results": volume, "/model-cache": model_cache},
)
def run_steering():
    import numpy as np
    import torch
    from scipy.stats import chi2
    from transformers import AutoModelForCausalLM, AutoTokenizer

    RESULTS_DIR = "/results/output"
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    SEED = 42
    BEST_LAYER = 23
    CRITICAL_LAYER = 22
    TSV_ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]
    PATCH_ALPHAS = [0.5, 1.0, 2.0, 5.0]
    RANDOM_ALPHAS = [1.0, 5.0, 10.0]

    SYSTEM_PROMPT = (
        "You are a clinical triage assistant for a Medicaid population health "
        "program. A care coordinator has shared the following patient message or "
        "encounter note. Assess whether the message contains any clinical hazard "
        "requiring action. If it does, specify the recommended action (e.g., call "
        "911, contact doctor, routine follow-up). If not, reassure."
    )

    EMERGENCY_KEYWORDS = [
        "call 911", "call 988", "emergency room", "emergency department",
        "ambulance", "go to the er", "go to the ed", "call emergency",
        "seek emergency", "life-threatening", "immediate danger",
        "immediately go to", "rush to the hospital",
    ]
    URGENT_KEYWORDS = [
        "urgent care", "call doctor", "call your doctor", "contact prescriber",
        "contact your provider", "seen urgently", "seen today", "same day",
        "same-day", "within 24 hours",
    ]

    def parse_detection(text):
        t = text.lower()
        for kw in EMERGENCY_KEYWORDS:
            if kw in t:
                return 1
        for kw in URGENT_KEYWORDS:
            if kw in t:
                return 1
        return 0

    def build_prompt(tokenizer, message):
        messages = [{"role": "user", "content": (
            f"{SYSTEM_PROMPT}\n\nPatient message: {message}\n\nAssessment:"
        )}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0, 0.0)
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
        return (round(p, 4), round(max(0, centre - margin), 4),
                round(min(1, centre + margin), 4))

    def mcnemar_p(b, c):
        """McNemar's test with continuity correction."""
        if b + c == 0:
            return 1.0
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        return float(1 - chi2.cdf(stat, df=1))

    # Load data
    print("Loading data...")
    with open(f"{RESULTS_DIR}/gemma2_base_results.json") as f:
        base_results = json.load(f)

    hidden_states = torch.load(
        f"{RESULTS_DIR}/gemma2_hidden_states.pt",
        map_location="cpu", weights_only=True
    ).float()
    print(f"Hidden states: {hidden_states.shape}")

    # Correct labels
    gt = np.array([r["detection_truth"] for r in base_results])
    pred = np.array([r.get("gemma2_detection", 0) for r in base_results])

    tp_mask = (gt == 1) & (pred == 1)
    fn_mask = (gt == 1) & (pred == 0)
    tn_mask = (gt == 0) & (pred == 0)
    tp_idx = np.where(tp_mask)[0]
    fn_idx = np.where(fn_mask)[0]
    tn_idx = np.where(tn_mask)[0]
    # Sample 50 TN cases for FP induction testing
    rng_tn = np.random.RandomState(SEED)
    tn_sample = rng_tn.choice(tn_idx, size=min(50, len(tn_idx)), replace=False).tolist()
    print(f"TP: {len(tp_idx)}, FN: {len(fn_idx)}, TN: {len(tn_idx)}, TN sample: {len(tn_sample)}")

    # Compute TSV at best_layer
    H = hidden_states[:, BEST_LAYER, :].numpy()
    mean_tp = H[tp_mask].mean(axis=0)
    mean_fn = H[fn_mask].mean(axis=0)
    tsv_raw = mean_tp - mean_fn
    tsv_norm = np.linalg.norm(tsv_raw)
    tsv_unit = tsv_raw / tsv_norm
    print(f"TSV norm: {tsv_norm:.4f}")

    # Compute patch direction at critical_layer
    H_crit = hidden_states[:, CRITICAL_LAYER, :].numpy()
    mean_tp_crit = H_crit[tp_mask].mean(axis=0)
    mean_fn_crit = H_crit[fn_mask].mean(axis=0)
    patch_raw = mean_tp_crit - mean_fn_crit
    patch_norm = np.linalg.norm(patch_raw)
    patch_unit = patch_raw / patch_norm
    print(f"Patch norm at layer {CRITICAL_LAYER}: {patch_norm:.4f}")

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    os.environ["HF_HOME"] = "/model-cache"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Convert directions to tensors
    tsv_tensor = torch.tensor(tsv_unit, dtype=torch.float16).to(device)
    patch_tensor = torch.tensor(patch_unit, dtype=torch.float16).to(device)

    def run_steered_inference(case_idx, layer_idx, direction, alpha, magnitude):
        """Run one steered inference and return (detection, response_text)."""
        msg = base_results[case_idx]["message"]
        prompt = build_prompt(tokenizer, msg)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patch = (alpha * magnitude * direction).to(hidden.dtype)
            hidden[:, -1, :] = hidden[:, -1, :] + patch
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        target_layer = model.model.layers[layer_idx]
        handle = target_layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=300,
                    do_sample=False,  # GREEDY decoding
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            detection = parse_detection(response)
        except Exception as e:
            print(f"  ERROR on case {case_idx}: {e}")
            response = ""
            detection = 0
        finally:
            handle.remove()

        return detection, response

    def run_baseline(case_idx):
        """Run unsteered inference with greedy decoding."""
        msg = base_results[case_idx]["message"]
        prompt = build_prompt(tokenizer, msg)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=300,
                do_sample=False,  # GREEDY decoding
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return parse_detection(response), response

    # Test cases
    test_fn = list(fn_idx)
    test_tp = list(tp_idx)
    test_tn = tn_sample

    def run_experiment(method_name, layer_idx, direction, magnitude, alphas):
        """Run a complete steering experiment across all alpha levels."""
        experiment_results = []

        for alpha in alphas:
            print(f"\n  {method_name} alpha={alpha}")
            per_case = []

            # FN cases
            fn_corrected = 0
            for i, idx in enumerate(test_fn):
                det, resp = run_steered_inference(int(idx), layer_idx, direction, alpha, magnitude)
                corrected = (det == 1)
                if corrected:
                    fn_corrected += 1
                per_case.append({
                    "case_idx": int(idx),
                    "case_id": base_results[idx].get("case_id", ""),
                    "case_type": "fn",
                    "detection_truth": int(gt[idx]),
                    "base_detection": int(pred[idx]),
                    "steered_detection": det,
                    "corrected": corrected,
                    "steered_response": resp[:500],
                })
                if (i + 1) % 20 == 0:
                    print(f"    FN: {i+1}/{len(test_fn)}, corrected: {fn_corrected}")

            # TP cases
            tp_disrupted = 0
            for i, idx in enumerate(test_tp):
                det, resp = run_steered_inference(int(idx), layer_idx, direction, alpha, magnitude)
                disrupted = (det == 0)
                if disrupted:
                    tp_disrupted += 1
                per_case.append({
                    "case_idx": int(idx),
                    "case_id": base_results[idx].get("case_id", ""),
                    "case_type": "tp",
                    "detection_truth": int(gt[idx]),
                    "base_detection": int(pred[idx]),
                    "steered_detection": det,
                    "disrupted": disrupted,
                    "steered_response": resp[:500],
                })
                if (i + 1) % 20 == 0:
                    print(f"    TP: {i+1}/{len(test_tp)}, disrupted: {tp_disrupted}")

            # TN cases (FP induction)
            fp_induced = 0
            for i, idx in enumerate(test_tn):
                det, resp = run_steered_inference(int(idx), layer_idx, direction, alpha, magnitude)
                induced = (det == 1)
                if induced:
                    fp_induced += 1
                per_case.append({
                    "case_idx": int(idx),
                    "case_id": base_results[idx].get("case_id", ""),
                    "case_type": "tn",
                    "detection_truth": int(gt[idx]),
                    "base_detection": int(pred[idx]),
                    "steered_detection": det,
                    "fp_induced": induced,
                    "steered_response": resp[:500],
                })

            corr_ci = wilson_ci(fn_corrected, len(test_fn))
            disr_ci = wilson_ci(tp_disrupted, len(test_tp))
            fp_ci = wilson_ci(fp_induced, len(test_tn))

            result = {
                "method": method_name,
                "layer": layer_idx,
                "alpha": alpha,
                "fn_corrected": fn_corrected,
                "fn_total": len(test_fn),
                "fn_correction_rate": corr_ci[0],
                "fn_correction_ci": [corr_ci[1], corr_ci[2]],
                "tp_disrupted": tp_disrupted,
                "tp_total": len(test_tp),
                "tp_disruption_rate": disr_ci[0],
                "tp_disruption_ci": [disr_ci[1], disr_ci[2]],
                "fp_induced": fp_induced,
                "tn_tested": len(test_tn),
                "fp_induction_rate": fp_ci[0],
                "fp_induction_ci": [fp_ci[1], fp_ci[2]],
                "net_corrected": fn_corrected - tp_disrupted,
                "per_case": per_case,
            }

            print(f"    FN corr: {fn_corrected}/{len(test_fn)}, "
                  f"TP disr: {tp_disrupted}/{len(test_tp)}, "
                  f"FP ind: {fp_induced}/{len(test_tn)}, "
                  f"net: {fn_corrected - tp_disrupted}")

            experiment_results.append(result)

        return experiment_results

    # --- Baseline verification (greedy) ---
    print(f"\n{'='*70}\nBASELINE VERIFICATION (greedy, {len(test_fn)+len(test_tp)} cases)\n{'='*70}")
    baseline_per_case = []
    for i, idx in enumerate(list(fn_idx) + list(tp_idx)):
        det, resp = run_baseline(int(idx))
        baseline_per_case.append({
            "case_idx": int(idx),
            "case_type": "fn" if idx in fn_idx else "tp",
            "base_detection_stored": int(pred[idx]),
            "base_detection_greedy": det,
            "response": resp[:500],
        })
        if (i + 1) % 20 == 0:
            print(f"  Baseline: {i+1}/{len(test_fn)+len(test_tp)}")

    # Check baseline consistency
    greedy_fn_det = sum(1 for r in baseline_per_case if r["case_type"] == "fn" and r["base_detection_greedy"] == 1)
    greedy_tp_det = sum(1 for r in baseline_per_case if r["case_type"] == "tp" and r["base_detection_greedy"] == 1)
    print(f"  Greedy baseline: FN->detected={greedy_fn_det}/{len(test_fn)}, "
          f"TP->detected={greedy_tp_det}/{len(test_tp)}")

    all_results = {"baseline_verification": baseline_per_case, "experiments": []}

    # --- TSV Steering ---
    print(f"\n{'='*70}\nTSV STEERING at layer {BEST_LAYER}\n{'='*70}")
    tsv_results = run_experiment("tsv_steering", BEST_LAYER, tsv_tensor, tsv_norm, TSV_ALPHAS)
    all_results["experiments"].extend(tsv_results)

    # Intermediate save
    out_path = f"{RESULTS_DIR}/tsv_patching_steering_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    volume.commit()
    print(f"\nSaved intermediate (TSV complete)")

    # --- Activation Patching ---
    print(f"\n{'='*70}\nACTIVATION PATCHING at layer {CRITICAL_LAYER}\n{'='*70}")
    patch_results = run_experiment("activation_patching", CRITICAL_LAYER, patch_tensor, patch_norm, PATCH_ALPHAS)
    all_results["experiments"].extend(patch_results)

    # Intermediate save
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    volume.commit()
    print(f"\nSaved intermediate (patching complete)")

    # --- Random Direction Control ---
    print(f"\n{'='*70}\nRANDOM DIRECTION CONTROL at layer {BEST_LAYER}\n{'='*70}")
    rng = np.random.RandomState(SEED)
    random_dir = rng.randn(hidden_states.shape[2]).astype(np.float32)
    random_dir = random_dir / np.linalg.norm(random_dir)
    random_tensor = torch.tensor(random_dir, dtype=torch.float16).to(device)

    random_results = run_experiment("random_control", BEST_LAYER, random_tensor, tsv_norm, RANDOM_ALPHAS)
    all_results["experiments"].extend(random_results)

    # --- McNemar tests: TSV vs random at matching alphas ---
    print(f"\n{'='*70}\nMcNEMAR TESTS\n{'='*70}")
    mcnemar_results = []
    for tsv_r in tsv_results:
        alpha = tsv_r["alpha"]
        rand_r = next((r for r in random_results if r["alpha"] == alpha), None)
        if rand_r is None:
            continue

        # For FN correction: compare paired outcomes
        tsv_fn_cases = {c["case_idx"]: c.get("corrected", False) for c in tsv_r["per_case"] if c["case_type"] == "fn"}
        rand_fn_cases = {c["case_idx"]: c.get("corrected", False) for c in rand_r["per_case"] if c["case_type"] == "fn"}
        common = set(tsv_fn_cases.keys()) & set(rand_fn_cases.keys())

        b = sum(1 for idx in common if tsv_fn_cases[idx] and not rand_fn_cases[idx])
        c = sum(1 for idx in common if not tsv_fn_cases[idx] and rand_fn_cases[idx])

        p_val = mcnemar_p(b, c)
        mcnemar_results.append({
            "comparison": f"tsv_alpha{alpha}_vs_random_alpha{alpha}",
            "alpha": alpha,
            "tsv_only_corrected": b,
            "random_only_corrected": c,
            "mcnemar_p": round(p_val, 4),
        })
        print(f"  TSV vs random at alpha={alpha}: b={b}, c={c}, p={p_val:.4f}")

    all_results["mcnemar_tests"] = mcnemar_results

    # --- Summary table ---
    summary = []
    for r in all_results["experiments"]:
        summary.append({
            "method": r["method"],
            "layer": r["layer"],
            "alpha": r["alpha"],
            "fn_corrected": r["fn_corrected"],
            "fn_total": r["fn_total"],
            "tp_disrupted": r["tp_disrupted"],
            "tp_total": r["tp_total"],
            "fp_induced": r["fp_induced"],
            "tn_tested": r["tn_tested"],
            "net_corrected": r["net_corrected"],
        })
    all_results["summary_table"] = summary

    # Final save
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    volume.commit()

    # Print summary
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for s in summary:
        net = s["fn_corrected"] - s["tp_disrupted"]
        print(f"  {s['method']} L{s['layer']} a={s['alpha']}: "
              f"FN corr={s['fn_corrected']}/{s['fn_total']}, "
              f"TP disr={s['tp_disrupted']}/{s['tp_total']}, "
              f"FP ind={s['fp_induced']}/{s['tn_tested']}, "
              f"net={net:+d}")

    return all_results


@app.local_entrypoint()
def main():
    results = run_steering.remote()

    os.makedirs("output", exist_ok=True)
    local_path = "output/tsv_patching_steering_v2.json"
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved locally to {local_path}")
