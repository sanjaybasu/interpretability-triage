#!/usr/bin/env python3
"""Step 23: Probing Classifiers and Truthfulness Separator Vectors (TSV).

Learns a linear direction in Qwen 2.5 7B hidden-state space that separates
correct hazard detection (TP) from missed hazards (FN), then uses that
direction to steer the model via activation addition at inference time.

Parts:
  A. Linear probing: per-layer logistic regression on ground-truth detection
  B. TSV computation: TP-vs-FN separator vector + detection direction
  C. Single-layer steering: add alpha * TSV to residual stream at critical layer
  D. Multi-layer steering: top-3 layers simultaneously
  E. Output: per-case results, aggregate summaries with Wilson CIs

Inputs:
  - output/gemma2_base_results.json (400-case predictions)
  - output/gemma2_hidden_states.pt ([400, 42, 3584] mean-pooled hidden states)

Runtime: ~3-5 hours on A100 (steering requires full model load).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from config import (
    CV_FOLDS,
    EMERGENCY_KEYWORDS,
    GEMMA_MODEL,
    GEMMA_HIDDEN_DIM,
    GEMMA_N_LAYERS,
    OUTPUT_DIR,
    SEED,
    STEERING_MAX_TOKENS,
    URGENT_KEYWORDS,
)
from src.utils import parse_triage_response, wilson_ci

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMMA2_MODEL_ID = GEMMA_MODEL
HIDDEN_DIM = GEMMA_HIDDEN_DIM
N_LAYERS = GEMMA_N_LAYERS
C_VALUES = [0.01, 0.1, 1.0, 10.0]
ALPHA_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]

SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)


def select_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_base_results():
    """Load Qwen 2.5 7B base results (400 cases with predictions and ground truth)."""
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
        print(f"ERROR: {path} not found. Run hidden-state extraction first.")
        sys.exit(1)
    states = torch.load(path, map_location="cpu", weights_only=True)
    if states.shape[1] != N_LAYERS:
        print(f"WARNING: Expected {N_LAYERS} layers, got {states.shape[1]}. "
              f"Using actual shape.")
    if states.shape[2] != HIDDEN_DIM:
        print(f"WARNING: Expected hidden dim {HIDDEN_DIM}, got {states.shape[2]}. "
              f"Using actual shape.")
    return states


def extract_labels(results):
    """Extract ground truth and prediction arrays from base results.

    Returns:
        ground_truth: np.array of shape (N,), binary
        predictions: np.array of shape (N,), binary
        case_types: list of str ('TP', 'FP', 'TN', 'FN')
    """
    ground_truth = np.array([r["detection_truth"] for r in results], dtype=int)
    predictions = np.array([r.get("gemma2_detection", r.get("detection", 0))
                            for r in results], dtype=int)
    case_types = []
    for gt, pred in zip(ground_truth, predictions):
        if gt == 1 and pred == 1:
            case_types.append("TP")
        elif gt == 1 and pred == 0:
            case_types.append("FN")
        elif gt == 0 and pred == 1:
            case_types.append("FP")
        else:
            case_types.append("TN")
    return ground_truth, predictions, case_types


# =========================================================================
# Part A: Linear Probing
# =========================================================================

def run_probing(hidden_states, ground_truth):
    """Train logistic regression probes at every layer.

    Args:
        hidden_states: torch.Tensor [N, 42, 3584]
        ground_truth: np.array [N] binary labels

    Returns:
        list of dicts with per-layer metrics, and index of best layer.
    """
    print("\n" + "=" * 70)
    print("PART A: Linear Probing (per-layer logistic regression)")
    print("=" * 70)

    n_cases = hidden_states.shape[0]
    n_layers = hidden_states.shape[1]
    layer_results = []

    for layer_idx in tqdm(range(n_layers), desc="Probing layers"):
        X = hidden_states[:, layer_idx, :].numpy().astype(np.float32)
        y = ground_truth

        best_auroc = -1.0
        best_c = None
        best_acc = None

        for c_val in C_VALUES:
            skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                  random_state=SEED)
            fold_aurocs = []
            fold_accs = []

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                clf = LogisticRegression(
                    C=c_val,
                    penalty="l2",
                    solver="saga",
                    max_iter=2000,
                    random_state=SEED,
                    n_jobs=1,
                )
                clf.fit(X_train, y_train)

                y_prob = clf.predict_proba(X_test)[:, 1]
                y_pred = clf.predict(X_test)

                fold_accs.append(np.mean(y_pred == y_test))
                # Guard against single-class folds
                if len(np.unique(y_test)) > 1:
                    fold_aurocs.append(roc_auc_score(y_test, y_prob))
                else:
                    fold_aurocs.append(0.5)

            mean_auroc = float(np.mean(fold_aurocs))
            mean_acc = float(np.mean(fold_accs))

            if mean_auroc > best_auroc:
                best_auroc = mean_auroc
                best_c = c_val
                best_acc = mean_acc

        layer_results.append({
            "layer": layer_idx,
            "accuracy": round(best_acc, 4),
            "auroc": round(best_auroc, 4),
            "best_C": best_c,
        })

        if (layer_idx + 1) % 10 == 0:
            print(f"  Layer {layer_idx:2d}: AUROC={best_auroc:.4f}, "
                  f"Acc={best_acc:.4f}, C={best_c}")

    # Identify best layer
    best_layer_idx = int(np.argmax([r["auroc"] for r in layer_results]))
    best_layer = layer_results[best_layer_idx]
    print(f"\nBest probing layer: {best_layer['layer']} "
          f"(AUROC={best_layer['auroc']:.4f})")

    # Save
    probe_output = {
        "per_layer": layer_results,
        "best_layer": best_layer["layer"],
        "best_auroc": best_layer["auroc"],
        "n_cases": n_cases,
        "cv_folds": CV_FOLDS,
        "c_values_tested": C_VALUES,
    }
    with open(OUTPUT_DIR / "probe_results.json", "w") as f:
        json.dump(probe_output, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'probe_results.json'}")

    return layer_results, best_layer["layer"]


# =========================================================================
# Part B: Truthfulness Separator Vector (TSV)
# =========================================================================

def compute_tsv(hidden_states, ground_truth, predictions, case_types,
                best_layer):
    """Compute TSV and detection direction at the most informative layer.

    Returns:
        tsv_dir: torch.Tensor [3584] unit vector (TP - FN direction)
        det_dir: torch.Tensor [3584] unit vector (hazard - benign direction)
        tsv_analysis: dict with metrics
    """
    print("\n" + "=" * 70)
    print("PART B: Truthfulness Separator Vector (TSV)")
    print("=" * 70)

    states_at_layer = hidden_states[:, best_layer, :]  # [N, 3584]
    case_types_arr = np.array(case_types)

    # Indices by case type
    tp_mask = case_types_arr == "TP"
    fn_mask = case_types_arr == "FN"
    hazard_mask = ground_truth == 1
    benign_mask = ground_truth == 0

    n_tp = int(tp_mask.sum())
    n_fn = int(fn_mask.sum())
    n_hazard = int(hazard_mask.sum())
    n_benign = int(benign_mask.sum())

    print(f"  TP cases: {n_tp}, FN cases: {n_fn}")
    print(f"  Hazard cases: {n_hazard}, Benign cases: {n_benign}")

    if n_tp < 2 or n_fn < 2:
        print("WARNING: Too few TP or FN cases for reliable TSV computation.")

    # TSV direction: normalize(mean_TP - mean_FN)
    mean_tp = states_at_layer[tp_mask].float().mean(dim=0)
    mean_fn = states_at_layer[fn_mask].float().mean(dim=0)
    tsv_raw = mean_tp - mean_fn
    tsv_dir = tsv_raw / tsv_raw.norm()

    # Detection direction: normalize(mean_hazard - mean_benign)
    mean_hazard = states_at_layer[hazard_mask].float().mean(dim=0)
    mean_benign = states_at_layer[benign_mask].float().mean(dim=0)
    det_raw = mean_hazard - mean_benign
    det_dir = det_raw / det_raw.norm()

    # Cosine similarity between TSV and detection direction
    cos_sim = float(torch.dot(tsv_dir, det_dir))
    print(f"  Cosine similarity (TSV vs detection): {cos_sim:.4f}")

    # Project hazard-positive cases onto TSV, compute AUROC for TP vs FN
    hazard_states = states_at_layer[hazard_mask].float()
    hazard_projections = (hazard_states @ tsv_dir).numpy()
    hazard_labels = (case_types_arr[hazard_mask] == "TP").astype(int)

    if len(np.unique(hazard_labels)) > 1:
        tsv_auroc = float(roc_auc_score(hazard_labels, hazard_projections))
    else:
        tsv_auroc = float("nan")
    print(f"  TSV AUROC (TP vs FN separation): {tsv_auroc:.4f}")

    # Save vectors
    torch.save({
        "tsv_direction": tsv_dir,
        "detection_direction": det_dir,
        "best_layer": best_layer,
        "tsv_raw_norm": float(tsv_raw.norm()),
        "det_raw_norm": float(det_raw.norm()),
    }, OUTPUT_DIR / "tsv_vectors.pt")
    print(f"Saved: {OUTPUT_DIR / 'tsv_vectors.pt'}")

    # Save analysis
    tsv_analysis = {
        "best_layer": best_layer,
        "cosine_similarity_tsv_det": round(cos_sim, 4),
        "tsv_auroc_tp_vs_fn": round(tsv_auroc, 4),
        "n_tp": n_tp,
        "n_fn": n_fn,
        "n_hazard": n_hazard,
        "n_benign": n_benign,
        "tsv_raw_norm": round(float(tsv_raw.norm()), 4),
        "det_raw_norm": round(float(det_raw.norm()), 4),
    }
    with open(OUTPUT_DIR / "tsv_analysis.json", "w") as f:
        json.dump(tsv_analysis, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'tsv_analysis.json'}")

    return tsv_dir, det_dir, tsv_analysis


def compute_loo_tsv(hidden_states, case_types_arr, best_layer, exclude_idx):
    """Compute leave-one-out TSV excluding the case at exclude_idx.

    Prevents circularity: the case being steered is not included in the
    mean computation for the TSV direction.

    Returns:
        tsv_dir: torch.Tensor [3584] unit vector, or None if insufficient data.
    """
    states_at_layer = hidden_states[:, best_layer, :].float()
    tp_mask = np.array(case_types_arr) == "TP"
    fn_mask = np.array(case_types_arr) == "FN"

    # Remove the excluded case from the appropriate group
    tp_indices = np.where(tp_mask)[0]
    fn_indices = np.where(fn_mask)[0]

    tp_indices = tp_indices[tp_indices != exclude_idx]
    fn_indices = fn_indices[fn_indices != exclude_idx]

    if len(tp_indices) < 1 or len(fn_indices) < 1:
        return None

    mean_tp = states_at_layer[tp_indices].mean(dim=0)
    mean_fn = states_at_layer[fn_indices].mean(dim=0)
    tsv_raw = mean_tp - mean_fn
    norm = tsv_raw.norm()
    if norm < 1e-8:
        return None
    return tsv_raw / norm


# =========================================================================
# Part C: Representation Steering with TSV (single layer)
# =========================================================================

def build_chat_prompt(tokenizer, message):
    """Build a chat-formatted prompt for Qwen 2.5 7B Instruct."""
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nPatient message: {message}\n\nAssessment:"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def make_steering_hook(layer_idx, direction, alpha, device):
    """Create a forward hook that adds alpha * direction to the residual stream.

    The hook modifies output[0] (hidden_states) at all token positions.
    Direction must be a unit vector of shape [hidden_dim].
    """
    dir_tensor = direction.to(device=device, dtype=torch.float16)

    def hook_fn(module, input, output):
        # output is a tuple: (hidden_states, ...) for transformer layers
        hidden = output[0]
        # Add steering vector to all token positions
        # hidden shape: [batch, seq_len, hidden_dim]
        hidden = hidden + alpha * dir_tensor.unsqueeze(0).unsqueeze(0)
        # Return modified output tuple
        return (hidden,) + output[1:]

    return hook_fn


def run_single_case_steered(model, tokenizer, message, hooks, device):
    """Run a single steered generation and parse the response.

    Args:
        model: AutoModelForCausalLM
        tokenizer: tokenizer
        message: patient message string
        hooks: list of registered hook handles (already attached)
        device: torch device string

    Returns:
        dict with 'response', 'detection', 'severity', 'action'
    """
    prompt = build_chat_prompt(tokenizer, message)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=STEERING_MAX_TOKENS,
            do_sample=False,
            temperature=1.0,
        )

    # Decode only the generated tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    parsed = parse_triage_response(response, EMERGENCY_KEYWORDS, URGENT_KEYWORDS)
    return {
        "response": response,
        "detection": parsed["detection"],
        "severity": parsed["severity"],
        "action": parsed["action"],
    }


def run_steering_experiment(model, tokenizer, results, hidden_states,
                            case_types, tsv_dir, det_dir, best_layer,
                            device):
    """Run Part C: single-layer TSV steering.

    Tests TSV steering on FN, TP, and TN cases with multiple alpha values.
    Includes random-direction and detection-direction negative controls.

    Returns:
        list of per-case result dicts
    """
    print("\n" + "=" * 70)
    print("PART C: Single-Layer Representation Steering")
    print("=" * 70)

    case_types_arr = np.array(case_types)
    fn_indices = np.where(case_types_arr == "FN")[0]
    tp_indices = np.where(case_types_arr == "TP")[0]
    tn_indices = np.where(case_types_arr == "TN")[0]

    print(f"  FN cases to correct: {len(fn_indices)}")
    print(f"  TP cases (disruption test): {len(tp_indices)}")
    print(f"  TN cases (FP induction test): {len(tn_indices)}")

    # Random direction control (fixed seed, same norm as TSV = unit vector)
    rng = np.random.RandomState(SEED)
    random_dir = torch.tensor(rng.randn(HIDDEN_DIM), dtype=torch.float32)
    random_dir = random_dir / random_dir.norm()

    # Target layer module path: model.model.layers[best_layer]
    target_layer = model.model.layers[best_layer]

    all_steering_results = []

    # Define steering conditions
    conditions = [
        ("tsv", tsv_dir),
        ("random", random_dir),
        ("detection", det_dir),
    ]

    for cond_name, direction in conditions:
        for alpha in ALPHA_VALUES:
            print(f"\n  Condition: {cond_name}, alpha={alpha}")

            # --- FN cases (correction target) ---
            for i, case_idx in enumerate(tqdm(fn_indices,
                                              desc=f"  FN {cond_name} a={alpha}")):
                # Leave-one-out TSV for the TSV condition
                if cond_name == "tsv":
                    loo_dir = compute_loo_tsv(
                        hidden_states, case_types, best_layer, case_idx,
                    )
                    if loo_dir is None:
                        continue
                    steer_dir = loo_dir
                else:
                    steer_dir = direction

                hook = target_layer.register_forward_hook(
                    make_steering_hook(best_layer, steer_dir, alpha, device)
                )
                try:
                    result = run_single_case_steered(
                        model, tokenizer, results[case_idx]["message"],
                        [hook], device,
                    )
                finally:
                    hook.remove()

                gt = results[case_idx]["detection_truth"]
                base_det = int(case_types_arr[case_idx] != "TN"
                               and case_types_arr[case_idx] != "FN")
                # For FN: base detection = 0, ground truth = 1
                all_steering_results.append({
                    "case_idx": int(case_idx),
                    "case_id": results[case_idx].get("case_id", ""),
                    "case_type": "FN",
                    "ground_truth": int(gt),
                    "base_detection": 0,
                    "steered_detection": result["detection"],
                    "corrected": int(result["detection"] == 1),
                    "condition": cond_name,
                    "alpha": alpha,
                    "n_layers_steered": 1,
                    "steered_layers": [best_layer],
                    "response_snippet": result["response"][:200],
                })

            # --- TP cases (disruption test) ---
            for case_idx in tqdm(tp_indices,
                                 desc=f"  TP {cond_name} a={alpha}"):
                if cond_name == "tsv":
                    loo_dir = compute_loo_tsv(
                        hidden_states, case_types, best_layer, case_idx,
                    )
                    if loo_dir is None:
                        continue
                    steer_dir = loo_dir
                else:
                    steer_dir = direction

                hook = target_layer.register_forward_hook(
                    make_steering_hook(best_layer, steer_dir, alpha, device)
                )
                try:
                    result = run_single_case_steered(
                        model, tokenizer, results[case_idx]["message"],
                        [hook], device,
                    )
                finally:
                    hook.remove()

                all_steering_results.append({
                    "case_idx": int(case_idx),
                    "case_id": results[case_idx].get("case_id", ""),
                    "case_type": "TP",
                    "ground_truth": 1,
                    "base_detection": 1,
                    "steered_detection": result["detection"],
                    "disrupted": int(result["detection"] == 0),
                    "condition": cond_name,
                    "alpha": alpha,
                    "n_layers_steered": 1,
                    "steered_layers": [best_layer],
                    "response_snippet": result["response"][:200],
                })

            # --- TN cases (FP induction test) ---
            for case_idx in tqdm(tn_indices,
                                 desc=f"  TN {cond_name} a={alpha}"):
                # No LOO needed for TN cases (they are not in TP/FN groups)
                steer_dir = direction

                hook = target_layer.register_forward_hook(
                    make_steering_hook(best_layer, steer_dir, alpha, device)
                )
                try:
                    result = run_single_case_steered(
                        model, tokenizer, results[case_idx]["message"],
                        [hook], device,
                    )
                finally:
                    hook.remove()

                all_steering_results.append({
                    "case_idx": int(case_idx),
                    "case_id": results[case_idx].get("case_id", ""),
                    "case_type": "TN",
                    "ground_truth": 0,
                    "base_detection": 0,
                    "steered_detection": result["detection"],
                    "fp_induced": int(result["detection"] == 1),
                    "condition": cond_name,
                    "alpha": alpha,
                    "n_layers_steered": 1,
                    "steered_layers": [best_layer],
                    "response_snippet": result["response"][:200],
                })

    return all_steering_results


# =========================================================================
# Part D: Multi-Layer Steering
# =========================================================================

def run_multilayer_steering(model, tokenizer, results, hidden_states,
                            case_types, tsv_dir, det_dir, layer_results,
                            device):
    """Run Part D: steering at top-3 most informative layers simultaneously.

    Returns:
        list of per-case result dicts (same schema as single-layer)
    """
    print("\n" + "=" * 70)
    print("PART D: Multi-Layer Steering (top-3 layers)")
    print("=" * 70)

    # Identify top-3 layers by probing AUROC
    sorted_layers = sorted(layer_results, key=lambda x: x["auroc"],
                           reverse=True)
    top3_layers = [r["layer"] for r in sorted_layers[:3]]
    print(f"  Top-3 layers: {top3_layers}")
    print(f"  AUROCs: {[r['auroc'] for r in sorted_layers[:3]]}")

    case_types_arr = np.array(case_types)
    fn_indices = np.where(case_types_arr == "FN")[0]
    tp_indices = np.where(case_types_arr == "TP")[0]
    tn_indices = np.where(case_types_arr == "TN")[0]

    # Precompute per-layer TSV directions (for LOO, we use the best_layer TSV
    # direction for all layers, since the TSV is defined as the TP-FN contrast)
    # For multi-layer steering, we compute a TSV at each of the top-3 layers.
    def compute_loo_tsv_at_layer(layer_idx, exclude_idx):
        """LOO TSV at a specific layer."""
        states = hidden_states[:, layer_idx, :].float()
        tp_mask = case_types_arr == "TP"
        fn_mask = case_types_arr == "FN"
        tp_idx = np.where(tp_mask)[0]
        fn_idx = np.where(fn_mask)[0]
        tp_idx = tp_idx[tp_idx != exclude_idx]
        fn_idx = fn_idx[fn_idx != exclude_idx]
        if len(tp_idx) < 1 or len(fn_idx) < 1:
            return None
        mean_tp = states[tp_idx].mean(dim=0)
        mean_fn = states[fn_idx].mean(dim=0)
        raw = mean_tp - mean_fn
        norm = raw.norm()
        if norm < 1e-8:
            return None
        return raw / norm

    # Precompute global (non-LOO) TSV at each top-3 layer for controls
    global_tsv_by_layer = {}
    global_det_by_layer = {}
    for layer_idx in top3_layers:
        states = hidden_states[:, layer_idx, :].float()
        tp_mask = case_types_arr == "TP"
        fn_mask = case_types_arr == "FN"
        mean_tp = states[tp_mask].mean(dim=0)
        mean_fn = states[fn_mask].mean(dim=0)
        raw = mean_tp - mean_fn
        global_tsv_by_layer[layer_idx] = raw / raw.norm()

        hazard_mask = np.array([r["detection_truth"] for r in results]) == 1
        benign_mask = ~hazard_mask
        mean_h = states[hazard_mask].mean(dim=0)
        mean_b = states[benign_mask].mean(dim=0)
        det_raw = mean_h - mean_b
        global_det_by_layer[layer_idx] = det_raw / det_raw.norm()

    # Random directions per layer (fixed seed)
    rng = np.random.RandomState(SEED + 1)
    random_dirs_by_layer = {}
    for layer_idx in top3_layers:
        rd = torch.tensor(rng.randn(HIDDEN_DIM), dtype=torch.float32)
        random_dirs_by_layer[layer_idx] = rd / rd.norm()

    target_layers = {idx: model.model.layers[idx] for idx in top3_layers}

    all_multi_results = []

    conditions = [
        ("tsv_multi", "tsv"),
        ("random_multi", "random"),
        ("detection_multi", "detection"),
    ]

    for cond_name, dir_type in conditions:
        for alpha in ALPHA_VALUES:
            print(f"\n  Multi-layer condition: {cond_name}, alpha={alpha}")

            # --- FN cases ---
            for case_idx in tqdm(fn_indices,
                                 desc=f"  FN {cond_name} a={alpha}"):
                hooks = []
                try:
                    for layer_idx in top3_layers:
                        if dir_type == "tsv":
                            d = compute_loo_tsv_at_layer(layer_idx, case_idx)
                            if d is None:
                                d = global_tsv_by_layer[layer_idx]
                        elif dir_type == "random":
                            d = random_dirs_by_layer[layer_idx]
                        else:
                            d = global_det_by_layer[layer_idx]

                        h = target_layers[layer_idx].register_forward_hook(
                            make_steering_hook(layer_idx, d, alpha, device)
                        )
                        hooks.append(h)

                    result = run_single_case_steered(
                        model, tokenizer, results[case_idx]["message"],
                        hooks, device,
                    )
                finally:
                    for h in hooks:
                        h.remove()

                all_multi_results.append({
                    "case_idx": int(case_idx),
                    "case_id": results[case_idx].get("case_id", ""),
                    "case_type": "FN",
                    "ground_truth": 1,
                    "base_detection": 0,
                    "steered_detection": result["detection"],
                    "corrected": int(result["detection"] == 1),
                    "condition": cond_name,
                    "alpha": alpha,
                    "n_layers_steered": 3,
                    "steered_layers": top3_layers,
                    "response_snippet": result["response"][:200],
                })

            # --- TP cases ---
            for case_idx in tqdm(tp_indices,
                                 desc=f"  TP {cond_name} a={alpha}"):
                hooks = []
                try:
                    for layer_idx in top3_layers:
                        if dir_type == "tsv":
                            d = compute_loo_tsv_at_layer(layer_idx, case_idx)
                            if d is None:
                                d = global_tsv_by_layer[layer_idx]
                        elif dir_type == "random":
                            d = random_dirs_by_layer[layer_idx]
                        else:
                            d = global_det_by_layer[layer_idx]

                        h = target_layers[layer_idx].register_forward_hook(
                            make_steering_hook(layer_idx, d, alpha, device)
                        )
                        hooks.append(h)

                    result = run_single_case_steered(
                        model, tokenizer, results[case_idx]["message"],
                        hooks, device,
                    )
                finally:
                    for h in hooks:
                        h.remove()

                all_multi_results.append({
                    "case_idx": int(case_idx),
                    "case_id": results[case_idx].get("case_id", ""),
                    "case_type": "TP",
                    "ground_truth": 1,
                    "base_detection": 1,
                    "steered_detection": result["detection"],
                    "disrupted": int(result["detection"] == 0),
                    "condition": cond_name,
                    "alpha": alpha,
                    "n_layers_steered": 3,
                    "steered_layers": top3_layers,
                    "response_snippet": result["response"][:200],
                })

            # --- TN cases ---
            for case_idx in tqdm(tn_indices,
                                 desc=f"  TN {cond_name} a={alpha}"):
                hooks = []
                try:
                    for layer_idx in top3_layers:
                        if dir_type == "random":
                            d = random_dirs_by_layer[layer_idx]
                        elif dir_type == "detection":
                            d = global_det_by_layer[layer_idx]
                        else:
                            d = global_tsv_by_layer[layer_idx]

                        h = target_layers[layer_idx].register_forward_hook(
                            make_steering_hook(layer_idx, d, alpha, device)
                        )
                        hooks.append(h)

                    result = run_single_case_steered(
                        model, tokenizer, results[case_idx]["message"],
                        hooks, device,
                    )
                finally:
                    for h in hooks:
                        h.remove()

                all_multi_results.append({
                    "case_idx": int(case_idx),
                    "case_id": results[case_idx].get("case_id", ""),
                    "case_type": "TN",
                    "ground_truth": 0,
                    "base_detection": 0,
                    "steered_detection": result["detection"],
                    "fp_induced": int(result["detection"] == 1),
                    "condition": cond_name,
                    "alpha": alpha,
                    "n_layers_steered": 3,
                    "steered_layers": top3_layers,
                    "response_snippet": result["response"][:200],
                })

    return all_multi_results


# =========================================================================
# Part E: Output and Summary
# =========================================================================

def compute_summary(all_results):
    """Compute aggregate metrics with Wilson CIs for each condition x alpha.

    Returns:
        list of summary dicts
    """
    # Group by (condition, alpha, n_layers_steered)
    groups = {}
    for r in all_results:
        key = (r["condition"], r["alpha"], r["n_layers_steered"])
        if key not in groups:
            groups[key] = {"FN": [], "TP": [], "TN": []}
        groups[key][r["case_type"]].append(r)

    summaries = []
    for (condition, alpha, n_layers), case_groups in sorted(groups.items()):
        fn_cases = case_groups["FN"]
        tp_cases = case_groups["TP"]
        tn_cases = case_groups["TN"]

        # FN correction rate
        n_fn = len(fn_cases)
        k_corrected = sum(1 for c in fn_cases if c.get("corrected", 0))
        fn_rate, fn_lo, fn_hi = wilson_ci(k_corrected, n_fn)

        # TP disruption rate
        n_tp = len(tp_cases)
        k_disrupted = sum(1 for c in tp_cases if c.get("disrupted", 0))
        tp_rate, tp_lo, tp_hi = wilson_ci(k_disrupted, n_tp)

        # FP induction rate
        n_tn = len(tn_cases)
        k_fp = sum(1 for c in tn_cases if c.get("fp_induced", 0))
        fp_rate, fp_lo, fp_hi = wilson_ci(k_fp, n_tn)

        summaries.append({
            "condition": condition,
            "alpha": alpha,
            "n_layers_steered": n_layers,
            "fn_correction_rate": round(fn_rate, 4),
            "fn_correction_ci": [round(fn_lo, 4), round(fn_hi, 4)],
            "fn_n": n_fn,
            "fn_k_corrected": k_corrected,
            "tp_disruption_rate": round(tp_rate, 4),
            "tp_disruption_ci": [round(tp_lo, 4), round(tp_hi, 4)],
            "tp_n": n_tp,
            "tp_k_disrupted": k_disrupted,
            "fp_induction_rate": round(fp_rate, 4),
            "fp_induction_ci": [round(fp_lo, 4), round(fp_hi, 4)],
            "tn_n": n_tn,
            "tn_k_fp_induced": k_fp,
        })

    return summaries


def print_summary_table(summaries):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("STEERING RESULTS SUMMARY")
    print("=" * 100)
    header = (f"{'Condition':<20} {'Alpha':>6} {'Layers':>6} "
              f"{'FN Corrected':>14} {'TP Disrupted':>14} {'FP Induced':>14}")
    print(header)
    print("-" * 100)

    for s in summaries:
        fn_str = (f"{s['fn_k_corrected']}/{s['fn_n']} "
                  f"({s['fn_correction_rate']:.1%})")
        tp_str = (f"{s['tp_k_disrupted']}/{s['tp_n']} "
                  f"({s['tp_disruption_rate']:.1%})")
        fp_str = (f"{s['tn_k_fp_induced']}/{s['tn_n']} "
                  f"({s['fp_induction_rate']:.1%})")
        print(f"{s['condition']:<20} {s['alpha']:>6.1f} "
              f"{s['n_layers_steered']:>6d} "
              f"{fn_str:>14} {tp_str:>14} {fp_str:>14}")


def save_outputs(single_results, multi_results, summaries):
    """Save all outputs."""
    all_results = single_results + multi_results

    with open(OUTPUT_DIR / "tsv_steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'tsv_steering_results.json'} "
          f"({len(all_results)} results)")

    with open(OUTPUT_DIR / "tsv_steering_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'tsv_steering_summary.json'} "
          f"({len(summaries)} conditions)")


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time.time()
    device = select_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading base results and hidden states...")
    results = load_base_results()
    hidden_states = load_hidden_states()
    ground_truth, predictions, case_types = extract_labels(results)

    n_total = len(results)
    print(f"  Cases: {n_total}")
    print(f"  TP={sum(1 for c in case_types if c == 'TP')}, "
          f"FN={sum(1 for c in case_types if c == 'FN')}, "
          f"FP={sum(1 for c in case_types if c == 'FP')}, "
          f"TN={sum(1 for c in case_types if c == 'TN')}")

    # ------------------------------------------------------------------
    # Part A: Linear Probing
    # ------------------------------------------------------------------
    layer_results, best_layer = run_probing(hidden_states, ground_truth)

    # ------------------------------------------------------------------
    # Part B: TSV Computation
    # ------------------------------------------------------------------
    tsv_dir, det_dir, tsv_analysis = compute_tsv(
        hidden_states, ground_truth, predictions, case_types, best_layer,
    )

    # ------------------------------------------------------------------
    # Parts C & D: Steering (requires model load)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Loading {GEMMA2_MODEL_ID} for steering experiments...")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(GEMMA2_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        GEMMA2_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    actual_n_layers = len(model.model.layers)
    print(f"  Model loaded. Layers: {actual_n_layers}")
    assert actual_n_layers == hidden_states.shape[1], (
        f"Model has {actual_n_layers} layers but hidden states have {hidden_states.shape[1]}"
    )

    # Part C: Single-layer steering
    single_results = run_steering_experiment(
        model, tokenizer, results, hidden_states, case_types,
        tsv_dir, det_dir, best_layer, device,
    )

    # Part D: Multi-layer steering
    multi_results = run_multilayer_steering(
        model, tokenizer, results, hidden_states, case_types,
        tsv_dir, det_dir, layer_results, device,
    )

    # ------------------------------------------------------------------
    # Part E: Summarize and save
    # ------------------------------------------------------------------
    all_results = single_results + multi_results
    summaries = compute_summary(all_results)
    print_summary_table(summaries)
    save_outputs(single_results, multi_results, summaries)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
