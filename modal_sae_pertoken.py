#!/usr/bin/env python3
"""Modal deployment: Per-token SAE training and steering for Arm 2 (corrected).

Corrects a methodological limitation in the original SAE arm: the original
trained on only 400 mean-pooled hidden-state vectors, whereas standard SAE
practice requires training on per-token activations (typically 50K-200K tokens).

Pipeline:
  1. Load Qwen 2.5 7B Instruct on A100
  2. Extract per-token hidden states at layer 14 for all 400 cases
  3. Train a 16,384-width SAE on the full per-token activation corpus
  4. Identify hazard-associated features via Mann-Whitney U + BH-FDR
  5. Steer false-negative cases by clamping top-K hazard features
  6. Run random-feature negative control
  7. Save all results

Usage:
  modal run modal_sae_pertoken.py
"""

import json
import modal
import os

app = modal.App("sae-pertoken-steering")

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


# ---------------------------------------------------------------------------
# SAE module (defined at top level so it can be instantiated inside the remote)
# ---------------------------------------------------------------------------

SAE_HIDDEN_DIM = 3584
SAE_N_FEATURES = 16384
SAE_L1_COEFF = 5e-3
SAE_LR = 1e-3
SAE_BATCH_SIZE = 256
SAE_EPOCHS = 8
TARGET_LAYER = 14
TOP_K = 20
SEED = 42

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

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
    """Parse triage detection from model response text."""
    t = text.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in t:
            return 1
    for kw in URGENT_KEYWORDS:
        if kw in t:
            return 1
    return 0


def build_prompt(tokenizer, message):
    """Build chat-formatted prompt for Qwen 2.5 7B Instruct."""
    messages = [{"role": "user", "content": (
        f"{SYSTEM_PROMPT}\n\nPatient message: {message}\n\nAssessment:"
    )}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Main remote function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=10800,  # 3 hours
    volumes={"/results": volume, "/model-cache": model_cache},
)
def run_sae_pertoken():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import stats
    from transformers import AutoModelForCausalLM, AutoTokenizer

    RESULTS_DIR = "/results/output"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["HF_HOME"] = "/model-cache"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # Step 0: Load base results and partition cases
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 0: Loading base results")
    print("=" * 70)

    with open(f"{RESULTS_DIR}/gemma2_base_results.json") as f:
        base_results = json.load(f)

    n_cases = len(base_results)
    gt = np.array([r["detection_truth"] for r in base_results])
    pred = np.array([r.get("gemma2_detection", 0) for r in base_results])

    tp_mask = (gt == 1) & (pred == 1)
    fn_mask = (gt == 1) & (pred == 0)
    tn_mask = (gt == 0) & (pred == 0)
    hazard_mask = gt == 1
    benign_mask = gt == 0

    tp_idx = np.where(tp_mask)[0]
    fn_idx = np.where(fn_mask)[0]
    tn_idx = np.where(tn_mask)[0]

    print(f"Total cases: {n_cases}")
    print(f"TP: {len(tp_idx)}, FN: {len(fn_idx)}, TN: {len(tn_idx)}")
    print(f"Hazard: {hazard_mask.sum()}, Benign: {benign_mask.sum()}")

    # ------------------------------------------------------------------
    # Step 1: Load model and extract per-token hidden states at layer 14
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Extracting per-token hidden states at layer 14")
    print("=" * 70)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract per-token activations and track which case each token belongs to
    all_token_activations = []  # list of tensors, each [seq_len, hidden_dim]
    case_token_indices = []     # (case_idx, start_token_idx, end_token_idx)
    total_tokens = 0

    for i, case in enumerate(base_results):
        prompt = build_prompt(tokenizer, case["message"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states[layer_idx+1] has shape [1, seq_len, hidden_dim]
            layer_hidden = outputs.hidden_states[TARGET_LAYER + 1]
            # Move to CPU as float32 for SAE training
            token_acts = layer_hidden.squeeze(0).cpu().float()  # [seq_len, hidden_dim]

        case_token_indices.append((i, total_tokens, total_tokens + seq_len))
        all_token_activations.append(token_acts)
        total_tokens += seq_len

        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            print(f"  Processed {i+1}/{n_cases} cases, {total_tokens} tokens so far")

    # Concatenate all per-token activations into one large matrix
    all_activations = torch.cat(all_token_activations, dim=0)  # [total_tokens, 3584]
    print(f"\nTotal per-token activations: {all_activations.shape}")
    print(f"  {total_tokens} tokens from {n_cases} cases")

    # Build per-token label array (1=hazard case, 0=benign case)
    token_labels = np.zeros(total_tokens, dtype=np.int32)
    token_case_ids = np.zeros(total_tokens, dtype=np.int32)
    for case_idx, start, end in case_token_indices:
        token_labels[start:end] = gt[case_idx]
        token_case_ids[start:end] = case_idx

    # Build per-token TP/FN masks for later feature analysis
    token_tp_mask = np.zeros(total_tokens, dtype=bool)
    token_fn_mask = np.zeros(total_tokens, dtype=bool)
    for case_idx, start, end in case_token_indices:
        if tp_mask[case_idx]:
            token_tp_mask[start:end] = True
        elif fn_mask[case_idx]:
            token_fn_mask[start:end] = True

    print(f"  Tokens from TP cases: {token_tp_mask.sum()}")
    print(f"  Tokens from FN cases: {token_fn_mask.sum()}")
    print(f"  Tokens from hazard cases: {(token_labels == 1).sum()}")
    print(f"  Tokens from benign cases: {(token_labels == 0).sum()}")

    # Free model memory before SAE training
    del model
    torch.cuda.empty_cache()
    print("  Model unloaded to free GPU memory for SAE training")

    # ------------------------------------------------------------------
    # Step 2: Train SAE on per-token activations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Training 16,384-width SAE on per-token activations")
    print("=" * 70)

    class SparseAutoencoder(nn.Module):
        """L1-regularized sparse autoencoder for residual stream decomposition."""

        def __init__(self, hidden_dim, n_features):
            super().__init__()
            self.encoder = nn.Linear(hidden_dim, n_features)
            self.decoder = nn.Linear(n_features, hidden_dim, bias=True)
            # Initialize decoder columns as unit vectors (Anthropic convention)
            with torch.no_grad():
                self.decoder.weight.data = F.normalize(
                    self.decoder.weight.data, dim=0
                )

        def encode(self, x):
            return torch.relu(self.encoder(x))

        def forward(self, x):
            features = self.encode(x)
            reconstructed = self.decoder(features)
            return reconstructed, features

    sae = SparseAutoencoder(SAE_HIDDEN_DIM, SAE_N_FEATURES).cuda()
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

    # Create DataLoader from per-token activations
    dataset = torch.utils.data.TensorDataset(all_activations)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=SAE_BATCH_SIZE, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=True
    )

    print(f"  SAE: {SAE_HIDDEN_DIM} -> {SAE_N_FEATURES} -> {SAE_HIDDEN_DIM}")
    print(f"  Training tokens: {total_tokens}")
    print(f"  Batch size: {SAE_BATCH_SIZE}, Epochs: {SAE_EPOCHS}")
    print(f"  L1 coeff: {SAE_L1_COEFF}, LR: {SAE_LR}")

    for epoch in range(SAE_EPOCHS):
        total_recon = 0.0
        total_l1 = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.cuda()
            reconstructed, features = sae(batch)

            recon_loss = F.mse_loss(reconstructed, batch)
            l1_loss = features.abs().mean()
            loss = recon_loss + SAE_L1_COEFF * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder columns to unit norm after each step
            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(
                    sae.decoder.weight.data, dim=0
                )

            total_recon += recon_loss.item()
            total_l1 += l1_loss.item()
            n_batches += 1

        avg_recon = total_recon / n_batches
        avg_l1 = total_l1 / n_batches
        print(f"  Epoch {epoch+1}/{SAE_EPOCHS}: recon={avg_recon:.6f}, L1={avg_l1:.6f}")

    # Report final sparsity
    sae.eval()
    with torch.no_grad():
        sample_idx = torch.randperm(total_tokens)[:5000]
        sample_acts = all_activations[sample_idx].cuda()
        sample_features = sae.encode(sample_acts)
        avg_l0 = (sample_features > 0).float().sum(dim=1).mean().item()
        nonzero_frac = (sample_features > 0).float().mean().item()

    print(f"\n  Final sparsity: avg L0 = {avg_l0:.1f} / {SAE_N_FEATURES}")
    print(f"  Nonzero fraction: {nonzero_frac:.4f}")

    # Save SAE weights
    sae_weights = {
        "W_enc": sae.encoder.weight.data.T.cpu().clone(),   # [hidden_dim, n_features]
        "b_enc": sae.encoder.bias.data.cpu().clone(),        # [n_features]
        "W_dec": sae.decoder.weight.data.T.cpu().clone(),    # [n_features, hidden_dim]
        "b_dec": sae.decoder.bias.data.cpu().clone(),        # [hidden_dim]
    }
    sae_path = f"{RESULTS_DIR}/sae_pertoken_weights_L{TARGET_LAYER}.pt"
    torch.save(sae_weights, sae_path)
    print(f"  Saved SAE weights: {sae_path}")

    # ------------------------------------------------------------------
    # Step 3: Encode all tokens and compute per-case mean SAE features
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Encoding all tokens through SAE")
    print("=" * 70)

    # Encode all tokens in batches
    all_sae_features = torch.zeros(total_tokens, SAE_N_FEATURES)
    encode_batch = 1024
    for start in range(0, total_tokens, encode_batch):
        end = min(start + encode_batch, total_tokens)
        with torch.no_grad():
            batch_feats = sae.encode(all_activations[start:end].cuda())
            all_sae_features[start:end] = batch_feats.cpu()

    all_sae_features_np = all_sae_features.numpy()
    print(f"  Encoded features shape: {all_sae_features_np.shape}")

    # Compute per-case mean SAE features (for Mann-Whitney testing)
    case_mean_features = np.zeros((n_cases, SAE_N_FEATURES), dtype=np.float32)
    for case_idx, start, end in case_token_indices:
        case_mean_features[case_idx] = all_sae_features_np[start:end].mean(axis=0)

    print(f"  Per-case mean features shape: {case_mean_features.shape}")

    # Free SAE from GPU (will reload model for steering)
    sae_cpu = SparseAutoencoder(SAE_HIDDEN_DIM, SAE_N_FEATURES)
    sae_cpu.load_state_dict(sae.state_dict())
    sae_cpu.eval()
    del sae
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: Identify hazard-associated features
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Identifying hazard-associated SAE features")
    print("=" * 70)

    # Use per-case mean features for statistical testing
    hazard_features = case_mean_features[hazard_mask]
    benign_features = case_mean_features[benign_mask]
    tp_features = case_mean_features[tp_mask]
    fn_features = case_mean_features[fn_mask]

    n_hazard = hazard_features.shape[0]
    n_benign = benign_features.shape[0]
    print(f"  Hazard cases: {n_hazard}, Benign cases: {n_benign}")

    # Pre-filter: only test features active in >= 5% of cases
    active_frac = (case_mean_features > 0).mean(axis=0)
    testable = active_frac >= 0.05
    n_testable = int(testable.sum())
    print(f"  Testable features (>= 5% active): {n_testable} / {SAE_N_FEATURES}")

    p_values = np.ones(SAE_N_FEATURES)
    effect_sizes = np.zeros(SAE_N_FEATURES)
    mean_hazard_act = np.zeros(SAE_N_FEATURES)
    mean_benign_act = np.zeros(SAE_N_FEATURES)

    for k in range(SAE_N_FEATURES):
        if not testable[k]:
            continue
        h_vals = hazard_features[:, k]
        b_vals = benign_features[:, k]
        mean_hazard_act[k] = h_vals.mean()
        mean_benign_act[k] = b_vals.mean()
        if h_vals.max() == 0 and b_vals.max() == 0:
            continue
        try:
            stat, p = stats.mannwhitneyu(h_vals, b_vals, alternative="two-sided")
            p_values[k] = p
            effect_sizes[k] = 1 - 2 * stat / (n_hazard * n_benign)
        except ValueError:
            pass

    # BH-FDR correction
    testable_idx = np.where(testable)[0]
    testable_pvals = p_values[testable_idx]
    m = len(testable_pvals)
    sorted_order = np.argsort(testable_pvals)
    sorted_pvals = testable_pvals[sorted_order]
    q_values_sorted = np.minimum(1.0, sorted_pvals * m / (np.arange(1, m + 1)))
    # Enforce monotonicity
    for i in range(m - 2, -1, -1):
        q_values_sorted[i] = min(q_values_sorted[i], q_values_sorted[i + 1])
    q_values_all = np.ones(SAE_N_FEATURES)
    q_values_all[testable_idx[sorted_order]] = q_values_sorted

    significant = q_values_all < 0.05
    n_significant = int(significant.sum())
    print(f"  Significant after BH-FDR (q < 0.05): {n_significant}")

    # Select top-K by mean difference (hazard > benign) among significant
    mean_diff = mean_hazard_act - mean_benign_act
    scores = np.where(significant & (mean_diff > 0), mean_diff, -np.inf)
    top_k_idx = np.argsort(scores)[::-1][:TOP_K]

    # TP mean activations for steering targets
    tp_means = tp_features.mean(axis=0) if len(tp_idx) > 0 else np.zeros(SAE_N_FEATURES)

    hazard_feature_info = {
        "layer": TARGET_LAYER,
        "total_tokens": total_tokens,
        "n_testable": n_testable,
        "n_significant": n_significant,
        "top_k_features": [int(x) for x in top_k_idx],
        "top_k_effect_sizes": [float(effect_sizes[x]) for x in top_k_idx],
        "top_k_mean_diff": [float(mean_diff[x]) for x in top_k_idx],
        "top_k_q_values": [float(q_values_all[x]) for x in top_k_idx],
        "top_k_tp_mean_activation": [float(tp_means[x]) for x in top_k_idx],
    }

    print(f"  Top {TOP_K} hazard features: {top_k_idx[:5].tolist()} ...")
    print(f"  Top effect sizes: {[f'{effect_sizes[x]:.3f}' for x in top_k_idx[:5]]}")
    print(f"  Top mean diffs: {[f'{mean_diff[x]:.4f}' for x in top_k_idx[:5]]}")

    hazard_path = f"{RESULTS_DIR}/sae_pertoken_hazard_features.json"
    with open(hazard_path, "w") as f:
        json.dump(hazard_feature_info, f, indent=2)
    print(f"  Saved: {hazard_path}")

    # ------------------------------------------------------------------
    # Step 5 & 6: Steering (hazard features + random control)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5-6: Steering false-negative cases")
    print("=" * 70)

    # Reload model for generation
    print(f"Reloading {MODEL_ID} for steering...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Prepare SAE encode/decode functions using saved weights
    W_enc = sae_weights["W_enc"].to(device).half()  # [hidden_dim, n_features]
    b_enc = sae_weights["b_enc"].to(device).half()   # [n_features]
    W_dec = sae_weights["W_dec"].to(device).half()   # [n_features, hidden_dim]
    b_dec = sae_weights["b_dec"].to(device).half()   # [hidden_dim]

    def sae_encode_gpu(x):
        """Encode hidden states through SAE on GPU. x: [..., hidden_dim]"""
        return torch.relu(x @ W_enc + b_enc)

    def sae_decode_gpu(features):
        """Decode SAE features back to hidden dim on GPU. features: [..., n_features]"""
        return features @ W_dec + b_dec

    # Prepare random control features
    rng = np.random.RandomState(SEED)
    random_features = rng.choice(
        [i for i in range(SAE_N_FEATURES) if i not in top_k_idx],
        size=TOP_K, replace=False
    ).tolist()
    random_tp_means = tp_means[random_features] if len(tp_idx) > 0 else np.ones(TOP_K)

    # Steering modes
    STEERING_CONFIGS = [
        {"name": "sae_tp_mean", "features": [int(x) for x in top_k_idx],
         "target_acts": [float(tp_means[x]) for x in top_k_idx], "multiplier": 1.0},
        {"name": "sae_amplified_2x", "features": [int(x) for x in top_k_idx],
         "target_acts": [float(tp_means[x]) for x in top_k_idx], "multiplier": 2.0},
        {"name": "sae_random_control", "features": random_features,
         "target_acts": [float(x) for x in random_tp_means], "multiplier": 1.0},
    ]

    def make_sae_steering_hook(feature_indices, target_activations, multiplier):
        """Create a forward hook that clamps SAE features during generation.

        The hook:
          1. Encodes residual stream through SAE encoder
          2. Clamps specified features to target * multiplier at all token positions
          3. Decodes back to residual stream
        """
        feat_idx_tensor = torch.tensor(feature_indices, dtype=torch.long, device=device)
        target_tensor = torch.tensor(
            [t * multiplier for t in target_activations],
            dtype=torch.float16, device=device
        )

        def hook_fn(module, input, output):
            hidden = output[0]
            original_dtype = hidden.dtype

            # SAE round-trip with feature clamping
            features = sae_encode_gpu(hidden)
            features[..., feat_idx_tensor] = target_tensor
            reconstructed = sae_decode_gpu(features)

            return (reconstructed.to(original_dtype),) + output[1:]

        return hook_fn

    target_module = model.model.layers[TARGET_LAYER]

    # Test cases: all FN (for correction) + TP sample (for disruption) + TN sample (for FP induction)
    tn_sample = tn_idx[:50].tolist() if len(tn_idx) > 50 else tn_idx.tolist()

    all_steering_results = []

    for config in STEERING_CONFIGS:
        mode_name = config["name"]
        feat_indices = config["features"]
        target_acts = config["target_acts"]
        multiplier = config["multiplier"]

        print(f"\n  --- Mode: {mode_name} (multiplier={multiplier}) ---")

        fn_corrected = 0
        tp_disrupted = 0
        fp_induced = 0
        mode_results = []

        # Steer FN cases
        for i, idx in enumerate(fn_idx):
            torch.manual_seed(SEED + int(idx))
            case = base_results[idx]
            prompt = build_prompt(tokenizer, case["message"])
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)

            hook = make_sae_steering_hook(feat_indices, target_acts, multiplier)
            handle = target_module.register_forward_hook(hook)

            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=300,
                        do_sample=False, temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                input_len = inputs["input_ids"].shape[1]
                response = tokenizer.decode(
                    output_ids[0][input_len:], skip_special_tokens=True
                )
                detection = parse_detection(response)
            except Exception as e:
                print(f"    ERROR on FN case {idx}: {e}")
                response = ""
                detection = 0
            finally:
                handle.remove()

            corrected = (detection == 1)
            if corrected:
                fn_corrected += 1

            mode_results.append({
                "case_idx": int(idx),
                "case_id": case.get("case_id", ""),
                "case_type": "fn",
                "steering_mode": mode_name,
                "detection_truth": int(gt[idx]),
                "base_detection": int(pred[idx]),
                "steered_detection": detection,
                "corrected": corrected,
                "steered_response": response[:500],
            })

            if (i + 1) % 20 == 0:
                print(f"    FN: {i+1}/{len(fn_idx)}, corrected so far: {fn_corrected}")
                torch.cuda.empty_cache()

        # Steer TP cases (disruption check)
        for i, idx in enumerate(tp_idx):
            torch.manual_seed(SEED + int(idx))
            case = base_results[idx]
            prompt = build_prompt(tokenizer, case["message"])
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)

            hook = make_sae_steering_hook(feat_indices, target_acts, multiplier)
            handle = target_module.register_forward_hook(hook)

            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=300,
                        do_sample=False, temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                input_len = inputs["input_ids"].shape[1]
                response = tokenizer.decode(
                    output_ids[0][input_len:], skip_special_tokens=True
                )
                detection = parse_detection(response)
            except Exception as e:
                print(f"    ERROR on TP case {idx}: {e}")
                response = ""
                detection = 1
            finally:
                handle.remove()

            disrupted = (detection == 0)
            if disrupted:
                tp_disrupted += 1

            mode_results.append({
                "case_idx": int(idx),
                "case_id": case.get("case_id", ""),
                "case_type": "tp",
                "steering_mode": mode_name,
                "detection_truth": int(gt[idx]),
                "base_detection": int(pred[idx]),
                "steered_detection": detection,
                "disrupted": disrupted,
                "steered_response": response[:500],
            })

            if (i + 1) % 20 == 0:
                print(f"    TP: {i+1}/{len(tp_idx)}, disrupted so far: {tp_disrupted}")
                torch.cuda.empty_cache()

        # Steer TN cases (FP induction check)
        for i, idx in enumerate(tn_sample):
            torch.manual_seed(SEED + int(idx))
            case = base_results[idx]
            prompt = build_prompt(tokenizer, case["message"])
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)

            hook = make_sae_steering_hook(feat_indices, target_acts, multiplier)
            handle = target_module.register_forward_hook(hook)

            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=300,
                        do_sample=False, temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                input_len = inputs["input_ids"].shape[1]
                response = tokenizer.decode(
                    output_ids[0][input_len:], skip_special_tokens=True
                )
                detection = parse_detection(response)
            except Exception as e:
                print(f"    ERROR on TN case {idx}: {e}")
                response = ""
                detection = 0
            finally:
                handle.remove()

            induced = (detection == 1)
            if induced:
                fp_induced += 1

            mode_results.append({
                "case_idx": int(idx),
                "case_id": case.get("case_id", ""),
                "case_type": "tn",
                "steering_mode": mode_name,
                "detection_truth": int(gt[idx]),
                "base_detection": int(pred[idx]),
                "steered_detection": detection,
                "fp_induced": induced,
                "steered_response": response[:500],
            })

        # Print mode summary
        correction_rate = fn_corrected / len(fn_idx) if len(fn_idx) > 0 else 0
        disruption_rate = tp_disrupted / len(tp_idx) if len(tp_idx) > 0 else 0
        fp_rate = fp_induced / len(tn_sample) if len(tn_sample) > 0 else 0

        print(f"\n  {mode_name} summary:")
        print(f"    FN corrected: {fn_corrected}/{len(fn_idx)} = {correction_rate:.1%}")
        print(f"    TP disrupted: {tp_disrupted}/{len(tp_idx)} = {disruption_rate:.1%}")
        print(f"    FP induced:   {fp_induced}/{len(tn_sample)} = {fp_rate:.1%}")
        print(f"    Net corrected: {fn_corrected - tp_disrupted - fp_induced}")

        all_steering_results.extend(mode_results)

    # ------------------------------------------------------------------
    # Step 7: Compute summary and save everything
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: Computing summary statistics")
    print("=" * 70)

    base_tp = int(tp_mask.sum())
    base_fn = int(fn_mask.sum())
    base_tn = int(tn_mask.sum())
    base_fp = int(((gt == 0) & (pred == 1)).sum())
    base_sens = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0
    base_spec = base_tn / (base_tn + base_fp) if (base_tn + base_fp) > 0 else 0

    print(f"\nBaseline (N={n_cases}):")
    print(f"  Sensitivity: {base_sens:.3f} (TP={base_tp}, FN={base_fn})")
    print(f"  Specificity: {base_spec:.3f} (TN={base_tn}, FP={base_fp})")

    def wilson_ci(k, n, z=1.96):
        """Wilson score interval for binomial proportion."""
        if n == 0:
            return (0.0, 0.0, 0.0)
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
        return (round(p, 4), round(max(0, centre - margin), 4),
                round(min(1, centre + margin), 4))

    summary = {
        "method": "sae_pertoken",
        "sae_architecture": {
            "hidden_dim": SAE_HIDDEN_DIM,
            "n_features": SAE_N_FEATURES,
            "target_layer": TARGET_LAYER,
            "total_training_tokens": total_tokens,
            "l1_coeff": SAE_L1_COEFF,
            "epochs": SAE_EPOCHS,
            "batch_size": SAE_BATCH_SIZE,
            "lr": SAE_LR,
            "avg_l0": round(avg_l0, 1),
            "nonzero_frac": round(nonzero_frac, 4),
        },
        "hazard_features": {
            "n_testable": n_testable,
            "n_significant": n_significant,
            "top_k": TOP_K,
        },
        "baseline": {
            "n_total": n_cases,
            "tp": base_tp, "fn": base_fn, "tn": base_tn, "fp": base_fp,
            "sensitivity": round(base_sens, 4),
            "specificity": round(base_spec, 4),
        },
        "steering_modes": {},
    }

    for config in STEERING_CONFIGS:
        mode_name = config["name"]
        mode_results = [r for r in all_steering_results if r["steering_mode"] == mode_name]

        fn_results = [r for r in mode_results if r["case_type"] == "fn"]
        tp_results = [r for r in mode_results if r["case_type"] == "tp"]
        tn_results = [r for r in mode_results if r["case_type"] == "tn"]

        n_fn = len(fn_results)
        n_corrected = sum(1 for r in fn_results if r.get("corrected", False))
        correction = wilson_ci(n_corrected, n_fn)

        n_tp = len(tp_results)
        n_disrupted = sum(1 for r in tp_results if r.get("disrupted", False))
        disruption = wilson_ci(n_disrupted, n_tp)

        n_tn = len(tn_results)
        n_fp_ind = sum(1 for r in tn_results if r.get("fp_induced", False))
        fp_ind = wilson_ci(n_fp_ind, n_tn)

        net_corrected = n_corrected - n_disrupted - n_fp_ind
        new_tp = base_tp - n_disrupted + n_corrected
        new_fn = base_fn - n_corrected + n_disrupted
        new_fp = base_fp + n_fp_ind
        new_tn = base_tn - n_fp_ind
        new_sens = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
        new_spec = new_tn / (new_tn + new_fp) if (new_tn + new_fp) > 0 else 0

        mode_summary = {
            "fn_tested": n_fn,
            "fn_corrected": n_corrected,
            "correction_rate": correction[0],
            "correction_ci_95": [correction[1], correction[2]],
            "tp_tested": n_tp,
            "tp_disrupted": n_disrupted,
            "disruption_rate": disruption[0],
            "disruption_ci_95": [disruption[1], disruption[2]],
            "tn_tested": n_tn,
            "fp_induced": n_fp_ind,
            "fp_induction_rate": fp_ind[0],
            "fp_induction_ci_95": [fp_ind[1], fp_ind[2]],
            "net_corrected": net_corrected,
            "adjusted_sensitivity": round(new_sens, 4),
            "adjusted_specificity": round(new_spec, 4),
            "sensitivity_delta": round(new_sens - base_sens, 4),
            "specificity_delta": round(new_spec - base_spec, 4),
        }
        summary["steering_modes"][mode_name] = mode_summary

        print(f"\n  {mode_name}:")
        print(f"    FN correction: {n_corrected}/{n_fn} = {correction[0]:.1%} "
              f"(95% CI {correction[1]:.1%}-{correction[2]:.1%})")
        print(f"    TP disruption: {n_disrupted}/{n_tp} = {disruption[0]:.1%} "
              f"(95% CI {disruption[1]:.1%}-{disruption[2]:.1%})")
        print(f"    FP induction:  {n_fp_ind}/{n_tn} = {fp_ind[0]:.1%} "
              f"(95% CI {fp_ind[1]:.1%}-{fp_ind[2]:.1%})")
        print(f"    Net corrected: {net_corrected}")
        print(f"    Sensitivity: {base_sens:.3f} -> {new_sens:.3f} ({new_sens - base_sens:+.3f})")
        print(f"    Specificity: {base_spec:.3f} -> {new_spec:.3f} ({new_spec - base_spec:+.3f})")

    # Identify best mode
    best_mode = max(
        summary["steering_modes"].keys(),
        key=lambda m: summary["steering_modes"][m]["net_corrected"]
    )
    summary["best_mode"] = best_mode
    summary["best_net_corrected"] = summary["steering_modes"][best_mode]["net_corrected"]

    print(f"\n  Best mode: {best_mode} "
          f"(net corrected = {summary['best_net_corrected']})")

    # Save all outputs
    steering_path = f"{RESULTS_DIR}/sae_pertoken_steering_results.json"
    with open(steering_path, "w") as f:
        json.dump(all_steering_results, f, indent=2)
    print(f"\nSaved steering results: {steering_path}")

    summary_path = f"{RESULTS_DIR}/sae_pertoken_steering_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # Save per-token SAE features (for downstream analysis)
    features_path = f"{RESULTS_DIR}/sae_pertoken_features_L{TARGET_LAYER}.npy"
    np.save(features_path, case_mean_features)
    print(f"Saved per-case mean features: {features_path}")

    volume.commit()
    print("\nDone. All results committed to volume.")

    return summary


@app.local_entrypoint()
def main():
    results = run_sae_pertoken.remote()

    # Save summary locally
    os.makedirs("output", exist_ok=True)
    local_path = "output/sae_pertoken_steering_summary.json"
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved locally to {local_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("PER-TOKEN SAE STEERING SUMMARY")
    print("=" * 70)
    print(f"Training tokens: {results['sae_architecture']['total_training_tokens']}")
    print(f"SAE L0: {results['sae_architecture']['avg_l0']}")
    print(f"Significant features: {results['hazard_features']['n_significant']}")
    print(f"Best mode: {results['best_mode']} "
          f"(net corrected = {results['best_net_corrected']})")
    for mode, stats in results["steering_modes"].items():
        print(f"\n  {mode}:")
        print(f"    Correction: {stats['fn_corrected']}/{stats['fn_tested']} "
              f"= {stats['correction_rate']:.1%}")
        print(f"    Disruption: {stats['tp_disrupted']}/{stats['tp_tested']} "
              f"= {stats['disruption_rate']:.1%}")
        print(f"    Sensitivity: {stats['adjusted_sensitivity']:.3f} "
              f"(delta {stats['sensitivity_delta']:+.3f})")
