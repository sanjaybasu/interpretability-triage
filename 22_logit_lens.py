#!/usr/bin/env python3
"""Step 22: Logit Lens analysis and Activation Patching on Qwen 2.5 7B Instruct.

Traces where hazard detection fails in the transformer network and tests
whether patching activations at critical layers can correct false negatives.

Design:
  Part A: Logit Lens -- project each layer's residual stream through unembedding
          to track hazard token logit rank across all 28 layers.
  Part B: Critical Layer Identification -- find layers where TP and FN
          trajectories maximally diverge in hazard token ranking.
  Part C: Activation Patching -- inject TP-FN correction direction at critical
          layers to correct false negatives; dose-response over alpha values.
  Part D: Output -- per-case and aggregate results with Wilson CIs.

Inputs:
  output/gemma2_base_results.json   -- 400 case inference results
  output/gemma2_hidden_states.pt    -- [400, 42, 3584] residual streams

Runtime: ~4-6 hours on A100 (inference for patching); analysis-only ~30 min.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import (
    DATA_DIR,
    EMERGENCY_KEYWORDS,
    GEMMA_MODEL,
    GEMMA_HIDDEN_DIM,
    GEMMA_N_LAYERS,
    OUTPUT_DIR,
    SEED,
    URGENT_KEYWORDS,
)
from src.utils import parse_triage_response, wilson_ci

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMMA_MODEL_ID = GEMMA_MODEL
N_LAYERS = GEMMA_N_LAYERS
HIDDEN_DIM = GEMMA_HIDDEN_DIM
HAZARD_TOKENS_RAW = ["911", "emergency", "ambulance", "urgent", "danger", "hospital"]
ALPHA_VALUES = [0.5, 1.0, 2.0, 5.0]
TOP_K_THRESHOLDS = [100, 50, 10]
MAX_NEW_TOKENS = 150

SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_base_results():
    """Load Qwen 2.5 7B base inference results (400 cases)."""
    path = OUTPUT_DIR / "gemma2_base_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run Qwen 2.5 7B base inference first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_hidden_states():
    """Load pre-extracted hidden states [400, 42, 3584]."""
    path = OUTPUT_DIR / "gemma2_hidden_states.pt"
    if not path.exists():
        print(f"ERROR: {path} not found. Run Qwen 2.5 7B hidden state extraction first.")
        sys.exit(1)
    return torch.load(path, map_location="cpu", weights_only=True)


def classify_cases(results):
    """Partition cases into TP, FN, TN, FP indices."""
    tp, fn, tn, fp = [], [], [], []
    for i, r in enumerate(results):
        truth = r["detection_truth"]
        pred = r.get("gemma2_detection", r.get("detection_pred", 0))
        if truth == 1 and pred == 1:
            tp.append(i)
        elif truth == 1 and pred == 0:
            fn.append(i)
        elif truth == 0 and pred == 0:
            tn.append(i)
        else:
            fp.append(i)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


# ---------------------------------------------------------------------------
# Part A: Logit Lens Analysis
# ---------------------------------------------------------------------------
def get_hazard_token_ids(tokenizer):
    """Tokenize hazard words and collect all resulting token IDs.

    Some words produce multiple sub-tokens; we track all of them.
    Returns dict mapping word -> list of token IDs.
    """
    hazard_map = {}
    for word in HAZARD_TOKENS_RAW:
        ids = tokenizer.encode(word, add_special_tokens=False)
        hazard_map[word] = ids
    # Flatten to unique set for rank computation
    all_ids = set()
    for ids in hazard_map.values():
        all_ids.update(ids)
    return hazard_map, sorted(all_ids)


def compute_logit_lens_single(hidden_states_case, unembed_weight, hazard_ids):
    """Compute hazard token logit ranks at each layer for one case.

    Args:
        hidden_states_case: [n_layers, hidden_dim] -- last-token residual stream.
        unembed_weight: [vocab_size, hidden_dim] -- lm_head weight matrix.
        hazard_ids: list of token IDs to track.

    Returns:
        Dict mapping layer_idx -> {token_id: rank}.
    """
    # logits at each layer: hidden @ W^T -> [n_layers, vocab_size]
    logits = hidden_states_case.float() @ unembed_weight.float().T

    per_layer = {}
    for layer_idx in range(logits.shape[0]):
        layer_logits = logits[layer_idx]  # [vocab_size]
        # Rank: number of tokens with higher logit + 1 (1-indexed)
        ranks = {}
        for tid in hazard_ids:
            rank = int((layer_logits > layer_logits[tid]).sum().item()) + 1
            ranks[tid] = rank
        per_layer[layer_idx] = ranks
    return per_layer


def run_logit_lens(hidden_states, unembed_weight, hazard_ids, results):
    """Run logit lens across all 400 cases.

    Returns:
        per_case: list of dicts, each with layer -> {token_id: rank}
        summary: aggregated TP vs FN trajectories
    """
    n_cases = hidden_states.shape[0]
    groups = classify_cases(results)
    per_case = []

    print(f"Running logit lens on {n_cases} cases, {N_LAYERS} layers each...")
    for i in tqdm(range(n_cases), desc="Logit lens"):
        layer_ranks = compute_logit_lens_single(
            hidden_states[i], unembed_weight, hazard_ids
        )
        per_case.append(layer_ranks)

    # Compute summary: mean rank per layer, per group (TP vs FN)
    summary = {"tp": {}, "fn": {}}
    for group_name in ["tp", "fn"]:
        indices = groups[group_name]
        if not indices:
            continue
        for layer_idx in range(N_LAYERS):
            all_ranks = []
            for i in indices:
                ranks = per_case[i][layer_idx]
                all_ranks.extend(ranks.values())
            summary[group_name][layer_idx] = {
                "mean_rank": float(np.mean(all_ranks)),
                "median_rank": float(np.median(all_ranks)),
                "std_rank": float(np.std(all_ranks)),
                "n_cases": len(indices),
            }

    # Compute first-entry thresholds for TP vs FN
    for group_name in ["tp", "fn"]:
        indices = groups[group_name]
        if not indices:
            continue
        for k in TOP_K_THRESHOLDS:
            entry_layers = []
            for i in indices:
                first_layer = None
                for layer_idx in range(N_LAYERS):
                    min_rank = min(per_case[i][layer_idx].values())
                    if min_rank <= k:
                        first_layer = layer_idx
                        break
                entry_layers.append(first_layer)
            valid = [l for l in entry_layers if l is not None]
            summary[group_name][f"first_top{k}_layer"] = {
                "mean": float(np.mean(valid)) if valid else None,
                "median": float(np.median(valid)) if valid else None,
                "n_entered": len(valid),
                "n_total": len(indices),
            }

    return per_case, summary


# ---------------------------------------------------------------------------
# Part B: Critical Layer Identification
# ---------------------------------------------------------------------------
def identify_critical_layers(per_case, results):
    """Find layers where TP and FN hazard rank trajectories maximally diverge.

    Uses Cohen's d effect size at each layer between TP and FN mean hazard ranks.

    Returns:
        List of dicts sorted by |effect_size| descending.
    """
    groups = classify_cases(results)
    tp_indices = groups["tp"]
    fn_indices = groups["fn"]

    if not tp_indices or not fn_indices:
        print("WARNING: Empty TP or FN group, cannot identify critical layers.")
        return []

    layer_divergence = []
    for layer_idx in range(N_LAYERS):
        tp_ranks = []
        for i in tp_indices:
            tp_ranks.append(np.mean(list(per_case[i][layer_idx].values())))
        fn_ranks = []
        for i in fn_indices:
            fn_ranks.append(np.mean(list(per_case[i][layer_idx].values())))

        tp_arr = np.array(tp_ranks)
        fn_arr = np.array(fn_ranks)

        # Cohen's d: (mean_fn - mean_tp) / pooled_sd
        # Positive d means FN has higher ranks (worse detection)
        n_tp, n_fn = len(tp_arr), len(fn_arr)
        mean_diff = float(fn_arr.mean() - tp_arr.mean())
        pooled_sd = float(np.sqrt(
            ((n_tp - 1) * tp_arr.std(ddof=1) ** 2 +
             (n_fn - 1) * fn_arr.std(ddof=1) ** 2)
            / (n_tp + n_fn - 2)
        ))
        effect_size = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

        layer_divergence.append({
            "layer": layer_idx,
            "mean_rank_tp": float(tp_arr.mean()),
            "mean_rank_fn": float(fn_arr.mean()),
            "rank_diff": mean_diff,
            "cohens_d": effect_size,
            "abs_cohens_d": abs(effect_size),
        })

    layer_divergence.sort(key=lambda x: x["abs_cohens_d"], reverse=True)
    return layer_divergence


# ---------------------------------------------------------------------------
# Part C: Activation Patching
# ---------------------------------------------------------------------------
def compute_correction_direction(hidden_states, groups, layer_idx):
    """Compute TP - FN mean direction at a specific layer.

    Returns:
        direction: [hidden_dim] unit vector
        magnitude: scalar norm of the raw difference
    """
    tp_states = hidden_states[groups["tp"], layer_idx, :]  # [n_tp, hidden_dim]
    fn_states = hidden_states[groups["fn"], layer_idx, :]  # [n_fn, hidden_dim]

    tp_mean = tp_states.float().mean(dim=0)
    fn_mean = fn_states.float().mean(dim=0)

    raw_direction = tp_mean - fn_mean  # Points from FN toward TP
    magnitude = raw_direction.norm().item()
    if magnitude > 0:
        unit_direction = raw_direction / magnitude
    else:
        unit_direction = raw_direction
    return unit_direction, magnitude


def make_patch_hook(layer_idx, direction, alpha, magnitude):
    """Create a forward hook that patches the residual stream at layer_idx.

    The hook adds alpha * magnitude * direction to the last token's hidden state.
    """
    def hook_fn(module, input, output):
        # Decoder layers may return a tuple: (hidden_states, ..., ...)
        hidden = output[0]  # [batch, seq_len, hidden_dim]
        # Patch last token position
        patch = (alpha * magnitude * direction).to(hidden.device, hidden.dtype)
        hidden[:, -1, :] = hidden[:, -1, :] + patch
        # Reconstruct output tuple with patched hidden states
        return (hidden,) + output[1:]
    return hook_fn


def make_random_direction(hidden_dim, rng):
    """Generate a random unit vector of the given dimension."""
    vec = torch.from_numpy(rng.randn(hidden_dim).astype(np.float32))
    return vec / vec.norm()


def format_chat_messages(message):
    """Format case as chat messages for Qwen 2.5 7B Instruct."""
    user_content = (
        f"Patient message: {message}\n\n"
        "Assess whether this contains a clinical hazard."
    )
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"},
    ]


def _run_patched_generation(model, tokenizer, inputs, layer_idx, direction,
                            alpha, magnitude):
    """Run generation with a patched layer and return decoded response."""
    hook = model.model.layers[layer_idx].register_forward_hook(
        make_patch_hook(layer_idx, direction, alpha, magnitude)
    )
    try:
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
            )
    finally:
        hook.remove()

    new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_activation_patching(model, tokenizer, results, hidden_states,
                            critical_layers, groups, device):
    """Run activation patching experiments on FN, TP, and TN cases.

    For each FN case:
      - Patch at primary critical layer with correction direction at each alpha
      - Patch with random direction as control
    For each TP case:
      - Same patching to measure disruption
    For each TN case:
      - Correction direction to measure FP induction

    Returns:
        per_case_results: list of dicts with patching outcomes
    """
    rng = np.random.RandomState(SEED)
    top_critical = critical_layers[:3]
    primary_layer = top_critical[0]["layer"]

    # Compute correction direction at primary critical layer
    correction_dir, correction_mag = compute_correction_direction(
        hidden_states, groups, primary_layer
    )
    correction_dir = correction_dir.to(device)

    # Random control direction (same magnitude)
    random_dir = make_random_direction(HIDDEN_DIM, rng).to(device)

    fn_indices = groups["fn"]
    tp_indices = groups["tp"]
    tn_indices = groups["tn"]

    all_results = []

    # --- Patch FN cases (correction) ---
    print(f"\nPatching {len(fn_indices)} FN cases at layer {primary_layer}...")
    for i in tqdm(fn_indices, desc="FN patching"):
        case = results[i]
        messages = format_chat_messages(case["message"])
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        case_result = {
            "case_idx": i,
            "case_id": case.get("case_id", f"case_{i}"),
            "group": "fn",
            "detection_truth": case["detection_truth"],
            "base_detection": case.get("detection_pred", case.get("detection", 0)),
            "critical_layer": primary_layer,
            "patched": {},
        }

        for alpha in ALPHA_VALUES:
            # Correction direction patching
            response = _run_patched_generation(
                model, tokenizer, inputs, primary_layer,
                correction_dir, alpha, correction_mag
            )
            parsed = parse_triage_response(
                response, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
            )
            case_result["patched"][f"correction_alpha{alpha}"] = {
                "response": response,
                "detection": parsed["detection"],
                "severity": parsed["severity"],
                "corrected": parsed["detection"] == 1,
            }

            # Random direction control
            response_rand = _run_patched_generation(
                model, tokenizer, inputs, primary_layer,
                random_dir, alpha, correction_mag
            )
            parsed_rand = parse_triage_response(
                response_rand, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
            )
            case_result["patched"][f"random_alpha{alpha}"] = {
                "response": response_rand,
                "detection": parsed_rand["detection"],
                "severity": parsed_rand["severity"],
                "corrected": parsed_rand["detection"] == 1,
            }

        all_results.append(case_result)

    # --- Patch TP cases (disruption test) ---
    print(f"\nPatching {len(tp_indices)} TP cases at layer {primary_layer}...")
    for i in tqdm(tp_indices, desc="TP patching"):
        case = results[i]
        messages = format_chat_messages(case["message"])
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        case_result = {
            "case_idx": i,
            "case_id": case.get("case_id", f"case_{i}"),
            "group": "tp",
            "detection_truth": case["detection_truth"],
            "base_detection": case.get("detection_pred", case.get("detection", 0)),
            "critical_layer": primary_layer,
            "patched": {},
        }

        for alpha in ALPHA_VALUES:
            # Correction direction on TP (should ideally not disrupt)
            response = _run_patched_generation(
                model, tokenizer, inputs, primary_layer,
                correction_dir, alpha, correction_mag
            )
            parsed = parse_triage_response(
                response, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
            )
            case_result["patched"][f"correction_alpha{alpha}"] = {
                "response": response,
                "detection": parsed["detection"],
                "severity": parsed["severity"],
                "disrupted": parsed["detection"] == 0,
            }

            # Random direction on TP
            response_rand = _run_patched_generation(
                model, tokenizer, inputs, primary_layer,
                random_dir, alpha, correction_mag
            )
            parsed_rand = parse_triage_response(
                response_rand, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
            )
            case_result["patched"][f"random_alpha{alpha}"] = {
                "response": response_rand,
                "detection": parsed_rand["detection"],
                "severity": parsed_rand["severity"],
                "disrupted": parsed_rand["detection"] == 0,
            }

        all_results.append(case_result)

    # --- Patch TN cases (FP induction test) ---
    print(f"\nPatching {len(tn_indices)} TN cases at layer {primary_layer}...")
    for i in tqdm(tn_indices, desc="TN patching"):
        case = results[i]
        messages = format_chat_messages(case["message"])
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        case_result = {
            "case_idx": i,
            "case_id": case.get("case_id", f"case_{i}"),
            "group": "tn",
            "detection_truth": case["detection_truth"],
            "base_detection": case.get("detection_pred", case.get("detection", 0)),
            "critical_layer": primary_layer,
            "patched": {},
        }

        for alpha in ALPHA_VALUES:
            response = _run_patched_generation(
                model, tokenizer, inputs, primary_layer,
                correction_dir, alpha, correction_mag
            )
            parsed = parse_triage_response(
                response, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
            )
            case_result["patched"][f"correction_alpha{alpha}"] = {
                "response": response,
                "detection": parsed["detection"],
                "severity": parsed["severity"],
                "fp_induced": parsed["detection"] == 1,
            }

        all_results.append(case_result)

    return all_results


# ---------------------------------------------------------------------------
# Part D: Summary Computation
# ---------------------------------------------------------------------------
def compute_patching_summary(patching_results):
    """Compute aggregate metrics with Wilson CIs across alpha values."""
    summary = {}

    for alpha in ALPHA_VALUES:
        alpha_key = f"alpha{alpha}"
        summary[alpha_key] = {}

        # FN correction rate (correction direction)
        fn_cases = [r for r in patching_results if r["group"] == "fn"]
        corr_key = f"correction_alpha{alpha}"
        fn_corrected = sum(
            1 for r in fn_cases
            if r["patched"].get(corr_key, {}).get("corrected", False)
        )
        n_fn = len(fn_cases)
        fn_corr_pt, fn_corr_lo, fn_corr_hi = wilson_ci(fn_corrected, n_fn)
        summary[alpha_key]["fn_correction_rate"] = {
            "k": fn_corrected, "n": n_fn,
            "rate": fn_corr_pt,
            "ci_lower": fn_corr_lo, "ci_upper": fn_corr_hi,
        }

        # FN correction rate (random direction -- control)
        rand_key = f"random_alpha{alpha}"
        fn_rand_corrected = sum(
            1 for r in fn_cases
            if r["patched"].get(rand_key, {}).get("corrected", False)
        )
        fn_rand_pt, fn_rand_lo, fn_rand_hi = wilson_ci(fn_rand_corrected, n_fn)
        summary[alpha_key]["fn_random_correction_rate"] = {
            "k": fn_rand_corrected, "n": n_fn,
            "rate": fn_rand_pt,
            "ci_lower": fn_rand_lo, "ci_upper": fn_rand_hi,
        }

        # TP disruption rate (correction direction)
        tp_cases = [r for r in patching_results if r["group"] == "tp"]
        tp_disrupted = sum(
            1 for r in tp_cases
            if r["patched"].get(corr_key, {}).get("disrupted", False)
        )
        n_tp = len(tp_cases)
        tp_dis_pt, tp_dis_lo, tp_dis_hi = wilson_ci(tp_disrupted, n_tp)
        summary[alpha_key]["tp_disruption_rate"] = {
            "k": tp_disrupted, "n": n_tp,
            "rate": tp_dis_pt,
            "ci_lower": tp_dis_lo, "ci_upper": tp_dis_hi,
        }

        # TP disruption rate (random direction)
        tp_rand_disrupted = sum(
            1 for r in tp_cases
            if r["patched"].get(rand_key, {}).get("disrupted", False)
        )
        tp_rand_pt, tp_rand_lo, tp_rand_hi = wilson_ci(tp_rand_disrupted, n_tp)
        summary[alpha_key]["tp_random_disruption_rate"] = {
            "k": tp_rand_disrupted, "n": n_tp,
            "rate": tp_rand_pt,
            "ci_lower": tp_rand_lo, "ci_upper": tp_rand_hi,
        }

        # FP induction rate on TN cases
        tn_cases = [r for r in patching_results if r["group"] == "tn"]
        fp_induced = sum(
            1 for r in tn_cases
            if r["patched"].get(corr_key, {}).get("fp_induced", False)
        )
        n_tn = len(tn_cases)
        fp_pt, fp_lo, fp_hi = wilson_ci(fp_induced, n_tn)
        summary[alpha_key]["fp_induction_rate"] = {
            "k": fp_induced, "n": n_tn,
            "rate": fp_pt,
            "ci_lower": fp_lo, "ci_upper": fp_hi,
        }

    return summary


def print_summary_tables(logit_summary, critical_layers, patching_summary):
    """Print formatted summary tables to stdout."""
    print("\n" + "=" * 72)
    print("LOGIT LENS SUMMARY: Hazard Token Entry Layers (TP vs FN)")
    print("=" * 72)
    for k in TOP_K_THRESHOLDS:
        key = f"first_top{k}_layer"
        tp_entry = logit_summary.get("tp", {}).get(key, {})
        fn_entry = logit_summary.get("fn", {}).get(key, {})
        tp_mean = tp_entry.get("mean", "N/A")
        fn_mean = fn_entry.get("mean", "N/A")
        tp_frac = f"{tp_entry.get('n_entered', 0)}/{tp_entry.get('n_total', 0)}"
        fn_frac = f"{fn_entry.get('n_entered', 0)}/{fn_entry.get('n_total', 0)}"
        tp_str = f"{tp_mean:.1f}" if isinstance(tp_mean, float) else tp_mean
        fn_str = f"{fn_mean:.1f}" if isinstance(fn_mean, float) else fn_mean
        print(f"  Top-{k:>3d}:  TP entry layer = {tp_str:>6s} ({tp_frac})"
              f"   FN entry layer = {fn_str:>6s} ({fn_frac})")

    print("\n" + "=" * 72)
    print("CRITICAL LAYERS (top 10 by |Cohen's d|)")
    print("=" * 72)
    print(f"  {'Layer':>5s}  {'TP rank':>10s}  {'FN rank':>10s}"
          f"  {'Diff':>8s}  {'d':>8s}")
    print("  " + "-" * 45)
    for entry in critical_layers[:10]:
        print(f"  {entry['layer']:5d}  {entry['mean_rank_tp']:10.1f}"
              f"  {entry['mean_rank_fn']:10.1f}  {entry['rank_diff']:8.1f}"
              f"  {entry['cohens_d']:8.3f}")

    if patching_summary:
        print("\n" + "=" * 72)
        print("ACTIVATION PATCHING RESULTS")
        print("=" * 72)
        print(f"  {'Alpha':>6s}  {'FN corr':>10s}  {'FN rand':>10s}"
              f"  {'TP disrupt':>12s}  {'FP induced':>12s}")
        print("  " + "-" * 55)
        for alpha in ALPHA_VALUES:
            ak = f"alpha{alpha}"
            s = patching_summary.get(ak, {})
            fn_c = s.get("fn_correction_rate", {})
            fn_r = s.get("fn_random_correction_rate", {})
            tp_d = s.get("tp_disruption_rate", {})
            fp_i = s.get("fp_induction_rate", {})

            def fmt(d):
                if not d:
                    return "N/A"
                return f"{d.get('rate', 0):.3f}"

            print(f"  {alpha:6.1f}  {fmt(fn_c):>10s}  {fmt(fn_r):>10s}"
                  f"  {fmt(tp_d):>12s}  {fmt(fp_i):>12s}")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _convert_keys(obj):
    """Recursively convert int keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _convert_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_keys(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data, path):
    """Save data to JSON with key conversion."""
    with open(path, "w") as f:
        json.dump(_convert_keys(data), f, indent=2, default=str)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    device = select_device()
    print(f"Device: {device}")

    # Load data
    print("\nLoading base results and hidden states...")
    results = load_base_results()
    hidden_states = load_hidden_states()
    # Validate shape dynamically
    if hidden_states.shape[1] != N_LAYERS or hidden_states.shape[2] != HIDDEN_DIM:
        print(f"WARNING: Hidden state shape {list(hidden_states.shape)} does not match "
              f"expected (*, {N_LAYERS}, {HIDDEN_DIM}). Adjusting constants.")
    N_LAYERS_ACTUAL = hidden_states.shape[1]
    HIDDEN_DIM_ACTUAL = hidden_states.shape[2]
    groups = classify_cases(results)
    print(f"  Cases: {len(results)} total | "
          f"TP={len(groups['tp'])} FN={len(groups['fn'])} "
          f"TN={len(groups['tn'])} FP={len(groups['fp'])}")

    # Load model + tokenizer for logit lens unembedding and patching
    print(f"\nLoading {GEMMA_MODEL_ID}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Extract unembedding matrix
    unembed_weight = model.lm_head.weight.detach().cpu()  # [vocab_size, hidden_dim]
    print(f"  Unembedding matrix: {unembed_weight.shape}")

    # Get hazard token IDs
    hazard_map, hazard_ids = get_hazard_token_ids(tokenizer)
    print(f"  Hazard tokens: {hazard_map}")
    print(f"  Unique hazard token IDs: {hazard_ids}")

    # -------------------------------------------------------------------
    # Part A: Logit Lens
    # -------------------------------------------------------------------
    print("\n--- Part A: Logit Lens Analysis ---")
    per_case_lens, logit_summary = run_logit_lens(
        hidden_states, unembed_weight, hazard_ids, results
    )
    save_json(per_case_lens, OUTPUT_DIR / "logit_lens_results.json")
    save_json(logit_summary, OUTPUT_DIR / "logit_lens_summary.json")

    # -------------------------------------------------------------------
    # Part B: Critical Layer Identification
    # -------------------------------------------------------------------
    print("\n--- Part B: Critical Layer Identification ---")
    critical_layers = identify_critical_layers(per_case_lens, results)
    save_json(critical_layers, OUTPUT_DIR / "critical_layers.json")
    if critical_layers:
        print(f"  Most critical layer: {critical_layers[0]['layer']} "
              f"(d={critical_layers[0]['cohens_d']:.3f})")

    # -------------------------------------------------------------------
    # Part C: Activation Patching
    # -------------------------------------------------------------------
    print("\n--- Part C: Activation Patching ---")
    if not critical_layers:
        print("  Skipping: no critical layers identified.")
        patching_results = []
        patching_summary = {}
    else:
        patching_results = run_activation_patching(
            model, tokenizer, results, hidden_states,
            critical_layers, groups, device
        )
        patching_summary = compute_patching_summary(patching_results)

    # -------------------------------------------------------------------
    # Part D: Save and print
    # -------------------------------------------------------------------
    print("\n--- Part D: Output ---")
    save_json(patching_results, OUTPUT_DIR / "activation_patching_results.json")
    save_json(patching_summary, OUTPUT_DIR / "activation_patching_summary.json")

    print_summary_tables(logit_summary, critical_layers, patching_summary)

    elapsed = time.time() - t0
    print(f"\nDone. Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
