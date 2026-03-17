#!/usr/bin/env python3
"""Step 21: SAE feature extraction and SAE-targeted steering.

Trains sparse autoencoders from scratch on extracted hidden states, then:
  A) Train SAEs on cached hidden states at target layers
  B) Identify hazard-associated SAE features via Mann-Whitney U + BH-FDR
  C) Steer false-negative cases by clamping hazard features in the SAE
     latent space, then decoding back to the residual stream
  D) Evaluate correction rate, disruption rate, and net benefit

Requires:
  - output/gemma2_base_results.json (base predictions)
  - output/gemma2_hidden_states.pt (optional; extracted on the fly if absent)
  - ~16GB GPU VRAM for comparative model (float16)

Usage:
  python 21_sae_steering.py                  # Run all parts
  python 21_sae_steering.py --part A         # SAE training + feature extraction only
  python 21_sae_steering.py --part B         # Hazard feature identification only
  python 21_sae_steering.py --part C         # Steering only
  python 21_sae_steering.py --part D         # Summary only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from config import (
    DATA_DIR,
    EMERGENCY_KEYWORDS,
    GEMMA_MODEL,
    GEMMA_HIDDEN_DIM,
    GEMMA_N_LAYERS,
    OUTPUT_DIR,
    SAE_L1_COEFF,
    SAE_LAYERS,
    SAE_LR,
    SAE_TRAIN_EPOCHS,
    SAE_WIDTH,
    SEED,
    TOP_K_CONCEPTS,
    URGENT_KEYWORDS,
)
from src.utils import benjamini_hochberg, parse_triage_response, wilson_ci

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMMA2_MODEL_ID = GEMMA_MODEL
N_FEATURES = SAE_WIDTH  # 16K features
TARGET_LAYERS = SAE_LAYERS  # Middle and late layers from config
TOP_K = TOP_K_CONCEPTS  # 20, matching existing pipeline

# Steering multipliers
STEERING_MODES = {
    "tp_mean": 1.0,       # Clamp to TP mean (in-distribution)
    "amplified": 2.0,     # 2x TP mean
    "random_control": 1.0,  # Random features, same magnitude (negative control)
}

SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)

# File paths
HIDDEN_STATES_PATH = OUTPUT_DIR / "gemma2_hidden_states.pt"
BASE_RESULTS_PATH = OUTPUT_DIR / "gemma2_base_results.json"
SAE_FEATURES_PATHS = {layer: OUTPUT_DIR / f"sae_features_L{layer}.npy" for layer in TARGET_LAYERS}
HAZARD_FEATURES_PATH = OUTPUT_DIR / "sae_hazard_features.json"
STEERING_RESULTS_PATH = OUTPUT_DIR / "sae_steering_results.json"
STEERING_SUMMARY_PATH = OUTPUT_DIR / "sae_steering_summary.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_test_cases():
    """Load physician and real-world test cases (same as 01_run_steerling_inference.py)."""
    physician_path = DATA_DIR / "physician_test_clean_n200.json"
    realworld_path = DATA_DIR / "realworld_test_n200_complete.json"

    with open(physician_path) as f:
        physician = json.load(f)
    with open(realworld_path) as f:
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
        })
    for c in realworld:
        cases.append({
            "case_id": c.get("case_id", ""),
            "message": c["message"],
            "detection_truth": c["ground_truth_detection"],
            "action_truth": c.get("ground_truth_action", "None"),
            "hazard_category": c.get("ground_truth_hazard_category", "Unknown"),
            "dataset": "real-world",
        })
    return cases


def build_chat_messages(message):
    """Build chat-formatted messages for Qwen 2.5 7B Instruct."""
    return [
        {"role": "user", "content": (
            f"{SYSTEM_PROMPT}\n\n"
            f"Patient message: {message}\n\n"
            "Assessment:"
        )},
    ]


def load_base_results():
    """Load Qwen 2.5 7B base inference results."""
    if not BASE_RESULTS_PATH.exists():
        print(f"ERROR: {BASE_RESULTS_PATH} not found. Run Qwen 2.5 7B base inference first.")
        sys.exit(1)
    with open(BASE_RESULTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SAE training (from scratch on extracted hidden states)
# ---------------------------------------------------------------------------

class SparseAutoencoder(torch.nn.Module):
    """L1-regularised sparse autoencoder for residual stream decomposition.

    Architecture: hidden -> ReLU(hidden @ W_enc + b_enc) -> features @ W_dec + b_dec
    Trained with MSE reconstruction loss + L1 sparsity penalty on features.
    """

    def __init__(self, hidden_dim, n_features):
        super().__init__()
        self.encoder = torch.nn.Linear(hidden_dim, n_features)
        self.decoder = torch.nn.Linear(n_features, hidden_dim, bias=True)
        # Initialize decoder columns as unit vectors (Anthropic convention)
        with torch.no_grad():
            self.decoder.weight.data = torch.nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, features):
        return self.decoder(features)

    def forward(self, x):
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features


def train_sae(hidden_states, layer_idx, device="cpu"):
    """Train a sparse autoencoder on hidden states for a given layer.

    Args:
        hidden_states: Tensor [N, hidden_dim] float32
        layer_idx: int, layer index (for logging)
        device: torch device string

    Returns:
        dict with keys W_enc, b_enc, W_dec, b_dec as torch tensors (float32, CPU).
    """
    hidden_dim = hidden_states.shape[1]
    n_features = N_FEATURES

    print(f"  Training SAE for layer {layer_idx}: "
          f"hidden_dim={hidden_dim}, n_features={n_features}")
    print(f"  L1 coeff={SAE_L1_COEFF}, epochs={SAE_TRAIN_EPOCHS}, lr={SAE_LR}")

    sae_model = SparseAutoencoder(hidden_dim, n_features).to(device)
    optimizer = torch.optim.Adam(sae_model.parameters(), lr=SAE_LR)

    # DataLoader from hidden states
    dataset = torch.utils.data.TensorDataset(hidden_states.to(device))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, drop_last=False
    )

    for epoch in range(SAE_TRAIN_EPOCHS):
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            reconstructed, features = sae_model(batch)

            recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)
            l1_loss = features.abs().mean()
            loss = recon_loss + SAE_L1_COEFF * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Re-normalize decoder columns to unit norm (Anthropic convention)
            with torch.no_grad():
                sae_model.decoder.weight.data = torch.nn.functional.normalize(
                    sae_model.decoder.weight.data, dim=0
                )

            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()
            n_batches += 1

        avg_recon = total_recon_loss / n_batches
        avg_l1 = total_l1_loss / n_batches
        print(f"    Epoch {epoch+1}/{SAE_TRAIN_EPOCHS}: "
              f"recon={avg_recon:.6f}, L1={avg_l1:.6f}")

    # Extract weights for downstream analysis
    sae_model.eval()
    sae_model.cpu()

    sae = {
        "W_enc": sae_model.encoder.weight.data.T.clone(),  # [hidden_dim, n_features]
        "b_enc": sae_model.encoder.bias.data.clone(),       # [n_features]
        "W_dec": sae_model.decoder.weight.data.T.clone(),   # [n_features, hidden_dim]
        "b_dec": sae_model.decoder.bias.data.clone(),       # [hidden_dim]
    }

    # Report sparsity
    with torch.no_grad():
        all_features = sae_model.encode(hidden_states.cpu())
        avg_l0 = (all_features > 0).float().sum(dim=1).mean().item()
        nonzero_frac = (all_features > 0).float().mean().item()

    print(f"  SAE layer {layer_idx}: W_enc {list(sae['W_enc'].shape)}, "
          f"W_dec {list(sae['W_dec'].shape)}, "
          f"n_features={n_features}")
    print(f"  Final sparsity: avg L0={avg_l0:.1f}, "
          f"nonzero frac={nonzero_frac:.4f}")

    # Save trained SAE weights
    sae_path = OUTPUT_DIR / f"sae_weights_L{layer_idx}.pt"
    torch.save(sae, sae_path)
    print(f"  Saved SAE weights: {sae_path}")

    del sae_model
    return sae


def load_or_train_sae(hidden_states, layer_idx, device="cpu"):
    """Load cached SAE weights or train from scratch."""
    sae_path = OUTPUT_DIR / f"sae_weights_L{layer_idx}.pt"
    if sae_path.exists():
        print(f"  Loading cached SAE weights: {sae_path}")
        return torch.load(sae_path, map_location="cpu", weights_only=True)
    return train_sae(hidden_states, layer_idx, device)


def sae_encode(hidden, sae):
    """Run SAE encoder: features = ReLU(hidden @ W_enc + b_enc).

    Args:
        hidden: Tensor of shape [..., hidden_dim] (float32).
        sae: Dict with W_enc [hidden_dim, n_features] and b_enc [n_features].

    Returns:
        Sparse feature activations, shape [..., n_features].
    """
    return torch.relu(hidden @ sae["W_enc"] + sae["b_enc"])


def sae_decode(features, sae):
    """Run SAE decoder: reconstructed = features @ W_dec + b_dec.

    Args:
        features: Tensor of shape [..., n_features] (float32).
        sae: Dict with W_dec [n_features, hidden_dim] and b_dec [hidden_dim].

    Returns:
        Reconstructed hidden states, shape [..., hidden_dim].
    """
    return features @ sae["W_dec"] + sae["b_dec"]


# ---------------------------------------------------------------------------
# Part A: SAE Feature Extraction
# ---------------------------------------------------------------------------

def extract_hidden_states_on_the_fly(cases, device):
    """Extract hidden states from Qwen 2.5 7B Instruct for all layers of interest.

    Called only when gemma2_hidden_states.pt does not exist.

    Returns:
        Tensor of shape [N, len(TARGET_LAYERS), hidden_dim] with mean-pooled
        hidden states per target layer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading Qwen 2.5 7B Instruct for hidden state extraction...")
    tokenizer = AutoTokenizer.from_pretrained(GEMMA2_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA2_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Hidden dim: {hidden_dim}, Layers: {n_layers}")

    # Validate target layers
    for layer_idx in TARGET_LAYERS:
        if layer_idx >= n_layers:
            print(f"  WARNING: Layer {layer_idx} >= num_layers {n_layers}, clamping.")

    all_hidden = torch.zeros(len(cases), len(TARGET_LAYERS), hidden_dim, dtype=torch.float32)

    for i, case in enumerate(tqdm(cases, desc="  Extracting hidden states")):
        messages = build_chat_messages(case["message"])
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = f"{SYSTEM_PROMPT}\n\nPatient message: {case['message']}\n\nAssessment:"

        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # outputs.hidden_states is a tuple of (n_layers+1) tensors,
            # each [1, seq_len, hidden_dim]. Index 0 = embedding output.
            for j, layer_idx in enumerate(TARGET_LAYERS):
                layer_hidden = outputs.hidden_states[layer_idx + 1]
                mean_pooled = layer_hidden.mean(dim=1).squeeze(0)
                all_hidden[i, j] = mean_pooled.cpu().float()

        # Free memory periodically
        if (i + 1) % 50 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    torch.save(all_hidden, HIDDEN_STATES_PATH)
    print(f"  Saved hidden states: {HIDDEN_STATES_PATH} {list(all_hidden.shape)}")

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return all_hidden


def part_a_extract_sae_features():
    """Part A: Train SAEs from scratch on hidden states and extract features."""
    print("\n" + "=" * 70)
    print("PART A: SAE Training + Feature Extraction")
    print("=" * 70)

    cases = load_test_cases()
    device = select_device()
    print(f"Device: {device}")
    print(f"Cases: {len(cases)}")
    print(f"Target layers: {TARGET_LAYERS}")

    # Load or extract hidden states
    if HIDDEN_STATES_PATH.exists():
        print(f"\nLoading cached hidden states: {HIDDEN_STATES_PATH}")
        all_hidden = torch.load(HIDDEN_STATES_PATH, map_location="cpu", weights_only=True)
        print(f"  Shape: {list(all_hidden.shape)}")

        # Handle different storage formats
        if all_hidden.ndim == 3 and all_hidden.shape[1] > len(TARGET_LAYERS):
            print(f"  Extracting target layers {TARGET_LAYERS} from {all_hidden.shape[1]} layers")
            all_hidden_target = all_hidden[:, TARGET_LAYERS, :]
        elif all_hidden.ndim == 3 and all_hidden.shape[1] == len(TARGET_LAYERS):
            all_hidden_target = all_hidden
        else:
            print(f"  WARNING: Unexpected shape {list(all_hidden.shape)}, re-extracting.")
            all_hidden_target = extract_hidden_states_on_the_fly(cases, device)
    else:
        print(f"\n{HIDDEN_STATES_PATH} not found. Extracting hidden states on the fly.")
        all_hidden_target = extract_hidden_states_on_the_fly(cases, device)

    n_cases = all_hidden_target.shape[0]
    hidden_dim = all_hidden_target.shape[2]
    print(f"\nHidden states: {n_cases} cases, {len(TARGET_LAYERS)} layers, dim={hidden_dim}")

    # Train SAE and extract features for each target layer
    for j, layer_idx in enumerate(TARGET_LAYERS):
        out_path = OUTPUT_DIR / f"sae_features_L{layer_idx}.npy"
        if out_path.exists():
            print(f"\n  {out_path} already exists, skipping layer {layer_idx}.")
            continue

        print(f"\n  Processing layer {layer_idx}...")
        layer_hidden = all_hidden_target[:, j, :].float()  # [N, hidden_dim]

        # Train or load SAE
        sae = load_or_train_sae(layer_hidden, layer_idx, device)

        # Validate dimension compatibility
        if sae["W_enc"].shape[0] != hidden_dim:
            print(f"  ERROR: SAE W_enc dim {sae['W_enc'].shape[0]} != hidden dim {hidden_dim}")
            print(f"  Skipping layer {layer_idx}.")
            continue

        # Encode in batches to manage memory
        batch_size = 64
        n_features = sae["W_enc"].shape[1]
        all_features = np.zeros((n_cases, n_features), dtype=np.float32)

        for start in range(0, n_cases, batch_size):
            end = min(start + batch_size, n_cases)
            batch_hidden = all_hidden_target[start:end, j, :]  # [batch, hidden_dim]
            with torch.no_grad():
                batch_features = sae_encode(batch_hidden, sae)
            all_features[start:end] = batch_features.numpy()

        # Report sparsity
        nonzero_frac = (all_features > 0).mean()
        avg_l0 = (all_features > 0).sum(axis=1).mean()
        print(f"  Layer {layer_idx}: nonzero fraction = {nonzero_frac:.4f}, "
              f"avg L0 = {avg_l0:.1f}")

        np.save(out_path, all_features)
        print(f"  Saved: {out_path} {all_features.shape}")

        del sae

    print("\nPart A complete.")


# ---------------------------------------------------------------------------
# Part B: Hazard-Associated Feature Identification
# ---------------------------------------------------------------------------

def part_b_identify_hazard_features():
    """Part B: Find SAE features associated with hazard presence via Mann-Whitney U."""
    print("\n" + "=" * 70)
    print("PART B: Hazard-Associated Feature Identification")
    print("=" * 70)

    base_results = load_base_results()
    y_true = np.array([r["detection_truth"] for r in base_results])
    y_pred = np.array([r.get("gemma2_detection", r.get("detection", 0)) for r in base_results])

    hazard_mask = y_true == 1
    no_hazard_mask = y_true == 0
    n_hazard = int(hazard_mask.sum())
    n_no_hazard = int(no_hazard_mask.sum())
    print(f"Hazard-present: {n_hazard}, Hazard-absent: {n_no_hazard}")

    tp_mask = (y_true == 1) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    print(f"True positives: {tp_mask.sum()}, False negatives: {fn_mask.sum()}")

    hazard_features_all = {}

    for layer_idx in TARGET_LAYERS:
        feat_path = OUTPUT_DIR / f"sae_features_L{layer_idx}.npy"
        if not feat_path.exists():
            print(f"\n  {feat_path} not found, skipping layer {layer_idx}.")
            continue

        print(f"\n  Layer {layer_idx}:")
        features = np.load(feat_path)  # [N, n_features]
        n_features = features.shape[1]
        print(f"  Features shape: {features.shape}")

        # Mann-Whitney U for each feature: hazard-present vs hazard-absent
        p_values = np.ones(n_features)
        effect_sizes = np.zeros(n_features)
        mean_hazard = np.zeros(n_features)
        mean_no_hazard = np.zeros(n_features)

        # Pre-filter: only test features active in >= 5% of cases
        active_frac = (features > 0).mean(axis=0)
        testable = active_frac >= 0.05
        n_testable = int(testable.sum())
        print(f"  Testable features (>= 5% active): {n_testable} / {n_features}")

        for k in tqdm(range(n_features), desc=f"  Mann-Whitney L{layer_idx}",
                       disable=n_features > 20000):
            if not testable[k]:
                continue
            h_vals = features[hazard_mask, k]
            nh_vals = features[no_hazard_mask, k]
            mean_hazard[k] = h_vals.mean()
            mean_no_hazard[k] = nh_vals.mean()

            if h_vals.max() == 0 and nh_vals.max() == 0:
                continue

            try:
                stat, p = stats.mannwhitneyu(h_vals, nh_vals, alternative="two-sided")
                p_values[k] = p
                # Rank-biserial effect size: r = 1 - 2U / (n1 * n2)
                effect_sizes[k] = 1 - 2 * stat / (n_hazard * n_no_hazard)
            except ValueError:
                pass

        # BH-FDR correction on testable features only
        testable_idx = np.where(testable)[0]
        testable_pvals = p_values[testable_idx]
        bh_result = benjamini_hochberg(testable_pvals, alpha=0.05)
        q_values = np.ones(n_features)
        q_values[testable_idx] = bh_result["q_values"]

        significant = q_values < 0.05
        n_significant = int(significant.sum())
        print(f"  Significant after BH-FDR: {n_significant}")

        # Select top-K by mean difference among significant features
        mean_diff = mean_hazard - mean_no_hazard
        scores = np.where(significant, mean_diff, -np.inf)
        top_k_idx = np.argsort(scores)[::-1][:TOP_K]

        # TP mean activations for steering targets
        tp_means = features[tp_mask].mean(axis=0) if tp_mask.sum() > 0 else np.zeros(n_features)

        layer_result = {
            "layer": layer_idx,
            "n_testable": n_testable,
            "n_significant": n_significant,
            "top_k_features": [int(x) for x in top_k_idx],
            "top_k_effect_sizes": [float(effect_sizes[x]) for x in top_k_idx],
            "top_k_mean_diff": [float(mean_diff[x]) for x in top_k_idx],
            "top_k_q_values": [float(q_values[x]) for x in top_k_idx],
            "top_k_tp_mean_activation": [float(tp_means[x]) for x in top_k_idx],
        }
        hazard_features_all[f"layer_{layer_idx}"] = layer_result

        print(f"  Top {TOP_K} hazard features: {top_k_idx[:5].tolist()} ...")
        print(f"  Top effect sizes: {[f'{effect_sizes[x]:.3f}' for x in top_k_idx[:5]]}")
        print(f"  Top mean diffs: {[f'{mean_diff[x]:.4f}' for x in top_k_idx[:5]]}")

    with open(HAZARD_FEATURES_PATH, "w") as f:
        json.dump(hazard_features_all, f, indent=2)
    print(f"\nSaved: {HAZARD_FEATURES_PATH}")
    print("Part B complete.")

    return hazard_features_all


# ---------------------------------------------------------------------------
# Part C: SAE-Targeted Steering
# ---------------------------------------------------------------------------

def part_c_sae_steering():
    """Part C: Steer false-negative cases by clamping SAE hazard features."""
    print("\n" + "=" * 70)
    print("PART C: SAE-Targeted Steering")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load hazard features
    if not HAZARD_FEATURES_PATH.exists():
        print(f"ERROR: {HAZARD_FEATURES_PATH} not found. Run Part B first.")
        sys.exit(1)
    with open(HAZARD_FEATURES_PATH) as f:
        hazard_features = json.load(f)

    # Load base results
    base_results = load_base_results()
    y_true = np.array([r["detection_truth"] for r in base_results])
    y_pred = np.array([r.get("gemma2_detection", r.get("detection", 0)) for r in base_results])

    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]

    print(f"TP: {len(tp_indices)}, FN: {len(fn_indices)}, "
          f"TN: {len(tn_indices)}, FP: {len(fp_indices)}")

    if len(fn_indices) == 0:
        print("No false negatives to correct. Exiting Part C.")
        return

    # Select the best layer (most significant features)
    best_layer_key = max(
        hazard_features.keys(),
        key=lambda k: hazard_features[k]["n_significant"]
    )
    best_layer = hazard_features[best_layer_key]
    layer_idx = best_layer["layer"]
    print(f"\nSteering layer: {layer_idx} ({best_layer['n_significant']} significant features)")

    top_k_features = best_layer["top_k_features"]
    tp_mean_activations = np.array(best_layer["top_k_tp_mean_activation"])

    # Load SAE weights for the steering layer (already trained in Part A)
    sae_path = OUTPUT_DIR / f"sae_weights_L{layer_idx}.pt"
    if not sae_path.exists():
        print(f"ERROR: {sae_path} not found. Run Part A first.")
        sys.exit(1)
    sae = torch.load(sae_path, map_location="cpu", weights_only=True)

    # Load model
    device = select_device()
    print(f"\nLoading Qwen 2.5 7B Instruct on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(GEMMA2_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA2_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  Model loaded. Hidden dim: {model.config.hidden_size}")

    cases = load_test_cases()

    # Prepare random features for negative control
    rng = np.random.RandomState(SEED)
    n_features_total = N_FEATURES
    random_features = rng.choice(
        [i for i in range(n_features_total) if i not in top_k_features],
        size=TOP_K, replace=False
    ).tolist()

    # Precompute random-feature TP means if feature file exists
    random_tp_means = tp_mean_activations.copy()  # fallback
    feat_path = OUTPUT_DIR / f"sae_features_L{layer_idx}.npy"
    if feat_path.exists() and len(tp_indices) > 0:
        all_feats = np.load(feat_path)
        tp_feats = all_feats[tp_indices]
        random_tp_means = tp_feats[:, random_features].mean(axis=0)

    # Build steering plan
    all_results = []
    steer_plan = []
    for idx in fn_indices:
        for mode in STEERING_MODES:
            steer_plan.append((int(idx), mode, "fn"))
    tp_sample = tp_indices.tolist()
    tn_sample = tn_indices[:50].tolist() if len(tn_indices) > 50 else tn_indices.tolist()
    for idx in tp_sample:
        steer_plan.append((idx, "tp_mean", "tp"))
    for idx in tn_sample:
        steer_plan.append((idx, "tp_mean", "tn"))

    print(f"\nSteering plan: {len(steer_plan)} runs "
          f"({len(fn_indices)} FN x {len(STEERING_MODES)} modes + "
          f"{len(tp_sample)} TP + {len(tn_sample)} TN)")

    def make_steering_hook(sae_weights, feature_indices, target_activations, multiplier):
        """Create a forward hook that modifies residual stream via SAE round-trip.

        The hook:
          1. Encodes the residual stream through the SAE encoder
          2. Clamps specified feature activations to target values * multiplier
          3. Decodes back to the residual stream
          4. Replaces the original hidden states with the reconstructed version
        """
        def hook_fn(module, input, output):
            hidden = output[0]
            original_dtype = hidden.dtype

            # Process in float32 for SAE numerical stability
            h_float = hidden.float().cpu()

            # SAE encode
            features = sae_encode(h_float, sae_weights)

            # Clamp target features at every token position
            for i, feat_idx in enumerate(feature_indices):
                target_val = target_activations[i] * multiplier
                features[..., feat_idx] = target_val

            # SAE decode
            reconstructed = sae_decode(features, sae_weights)

            # Move back to original device and dtype
            reconstructed = reconstructed.to(dtype=original_dtype, device=hidden.device)

            return (reconstructed,) + output[1:]

        return hook_fn

    # Access the target layer module (model.model.layers[i])
    target_module = model.model.layers[layer_idx]

    for case_idx, mode, case_type in tqdm(steer_plan, desc="  Steering"):
        case = cases[case_idx]

        # Determine feature indices and target activations per mode
        if mode == "random_control":
            feat_indices = random_features
            target_acts = random_tp_means
        else:
            feat_indices = top_k_features
            target_acts = tp_mean_activations

        multiplier = STEERING_MODES[mode]

        # Register hook
        hook_handle = target_module.register_forward_hook(
            make_steering_hook(sae, feat_indices, target_acts, multiplier)
        )

        # Build prompt
        messages = build_chat_messages(case["message"])
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = f"{SYSTEM_PROMPT}\n\nPatient message: {case['message']}\n\nAssessment:"

        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=1.0,
            )

        # Remove hook immediately
        hook_handle.remove()

        # Decode response (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse triage decision
        triage = parse_triage_response(response_text, EMERGENCY_KEYWORDS, URGENT_KEYWORDS)

        result = {
            "case_idx": int(case_idx),
            "case_id": case["case_id"],
            "case_type": case_type,
            "steering_mode": mode,
            "steering_layer": layer_idx,
            "detection_truth": int(case["detection_truth"]),
            "base_detection": int(y_pred[case_idx]),
            "steered_detection": triage["detection"],
            "steered_severity": triage["severity"],
            "steered_response": response_text[:500],
        }

        if case_type == "fn":
            result["corrected"] = (triage["detection"] == 1)
        elif case_type == "tp":
            result["disrupted"] = (triage["detection"] == 0)
        elif case_type == "tn":
            result["fp_induced"] = (triage["detection"] == 1)

        all_results.append(result)

        # Periodic memory cleanup
        if len(all_results) % 20 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    with open(STEERING_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {STEERING_RESULTS_PATH} ({len(all_results)} results)")

    del model, tokenizer, sae
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Part C complete.")
    return all_results


# ---------------------------------------------------------------------------
# Part D: Summary
# ---------------------------------------------------------------------------

def part_d_summary():
    """Part D: Compute and print summary statistics from steering results."""
    print("\n" + "=" * 70)
    print("PART D: Summary")
    print("=" * 70)

    if not STEERING_RESULTS_PATH.exists():
        print(f"ERROR: {STEERING_RESULTS_PATH} not found. Run Part C first.")
        sys.exit(1)

    with open(STEERING_RESULTS_PATH) as f:
        results = json.load(f)

    base_results = load_base_results()
    y_true = np.array([r["detection_truth"] for r in base_results])
    y_pred = np.array([r.get("gemma2_detection", r.get("detection", 0)) for r in base_results])
    n_total = len(y_true)
    base_tp = int(((y_true == 1) & (y_pred == 1)).sum())
    base_fn = int(((y_true == 1) & (y_pred == 0)).sum())
    base_tn = int(((y_true == 0) & (y_pred == 0)).sum())
    base_fp = int(((y_true == 0) & (y_pred == 1)).sum())
    base_sens = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0
    base_spec = base_tn / (base_tn + base_fp) if (base_tn + base_fp) > 0 else 0

    print(f"\nBaseline (Qwen 2.5 7B Instruct, N={n_total}):")
    print(f"  Sensitivity: {base_sens:.3f} (TP={base_tp}, FN={base_fn})")
    print(f"  Specificity: {base_spec:.3f} (TN={base_tn}, FP={base_fp})")

    summary = {
        "baseline": {
            "n_total": n_total,
            "tp": base_tp, "fn": base_fn, "tn": base_tn, "fp": base_fp,
            "sensitivity": round(base_sens, 4),
            "specificity": round(base_spec, 4),
        },
        "steering_modes": {},
    }

    for mode in STEERING_MODES:
        mode_results = [r for r in results if r["steering_mode"] == mode]

        fn_results = [r for r in mode_results if r["case_type"] == "fn"]
        tp_results = [r for r in mode_results if r["case_type"] == "tp"]
        tn_results = [r for r in mode_results if r["case_type"] == "tn"]

        n_fn = len(fn_results)
        n_corrected = sum(1 for r in fn_results if r.get("corrected", False))
        correction_rate = n_corrected / n_fn if n_fn > 0 else 0
        correction_ci = wilson_ci(n_corrected, n_fn)

        n_tp = len(tp_results)
        n_disrupted = sum(1 for r in tp_results if r.get("disrupted", False))
        disruption_rate = n_disrupted / n_tp if n_tp > 0 else 0
        disruption_ci = wilson_ci(n_disrupted, n_tp)

        n_tn = len(tn_results)
        n_fp_induced = sum(1 for r in tn_results if r.get("fp_induced", False))
        fp_induction_rate = n_fp_induced / n_tn if n_tn > 0 else 0
        fp_induction_ci = wilson_ci(n_fp_induced, n_tn)

        # Net benefit
        net_corrected = n_corrected - n_disrupted - n_fp_induced
        new_tp = base_tp - n_disrupted + n_corrected
        new_fn = base_fn - n_corrected + n_disrupted
        new_fp = base_fp + n_fp_induced
        new_tn = base_tn - n_fp_induced
        new_sens = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
        new_spec = new_tn / (new_tn + new_fp) if (new_tn + new_fp) > 0 else 0

        mode_summary = {
            "fn_tested": n_fn,
            "fn_corrected": n_corrected,
            "correction_rate": round(correction_rate, 4),
            "correction_ci_95": [round(correction_ci[1], 4), round(correction_ci[2], 4)],
            "tp_tested": n_tp,
            "tp_disrupted": n_disrupted,
            "disruption_rate": round(disruption_rate, 4),
            "disruption_ci_95": [round(disruption_ci[1], 4), round(disruption_ci[2], 4)],
            "tn_tested": n_tn,
            "fp_induced": n_fp_induced,
            "fp_induction_rate": round(fp_induction_rate, 4),
            "fp_induction_ci_95": [round(fp_induction_ci[1], 4), round(fp_induction_ci[2], 4)],
            "net_corrected": net_corrected,
            "adjusted_sensitivity": round(new_sens, 4),
            "adjusted_specificity": round(new_spec, 4),
            "sensitivity_delta": round(new_sens - base_sens, 4),
            "specificity_delta": round(new_spec - base_spec, 4),
        }
        summary["steering_modes"][mode] = mode_summary

        print(f"\n  Mode: {mode} (multiplier={STEERING_MODES[mode]})")
        print(f"    FN correction: {n_corrected}/{n_fn} = {correction_rate:.1%} "
              f"(95% CI {correction_ci[1]:.1%}-{correction_ci[2]:.1%})")
        print(f"    TP disruption: {n_disrupted}/{n_tp} = {disruption_rate:.1%} "
              f"(95% CI {disruption_ci[1]:.1%}-{disruption_ci[2]:.1%})")
        print(f"    FP induction:  {n_fp_induced}/{n_tn} = {fp_induction_rate:.1%} "
              f"(95% CI {fp_induction_ci[1]:.1%}-{fp_induction_ci[2]:.1%})")
        print(f"    Net corrected: {net_corrected}")
        print(f"    Adjusted sensitivity: {new_sens:.3f} (delta {new_sens - base_sens:+.3f})")
        print(f"    Adjusted specificity: {new_spec:.3f} (delta {new_spec - base_spec:+.3f})")

    # Identify best mode
    best_mode = max(
        summary["steering_modes"].keys(),
        key=lambda m: summary["steering_modes"][m]["net_corrected"]
    )
    summary["best_mode"] = best_mode
    summary["best_net_corrected"] = summary["steering_modes"][best_mode]["net_corrected"]

    print(f"\n  Best mode: {best_mode} "
          f"(net corrected = {summary['best_net_corrected']})")

    with open(STEERING_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {STEERING_SUMMARY_PATH}")
    print("Part D complete.")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAE feature extraction and steering for Qwen 2.5 7B Instruct"
    )
    parser.add_argument(
        "--part", type=str, default=None,
        choices=["A", "B", "C", "D"],
        help="Run a specific part only (default: run all)."
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.part is None or args.part == "A":
        part_a_extract_sae_features()

    if args.part is None or args.part == "B":
        part_b_identify_hazard_features()

    if args.part is None or args.part == "C":
        part_c_sae_steering()

    if args.part is None or args.part == "D":
        part_d_summary()

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
