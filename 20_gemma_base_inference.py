#!/usr/bin/env python3
"""Step 20: Run Qwen 2.5 7B Instruct inference with hidden state extraction on triage cases.

Loads physician-created and real-world test cases, runs each through
Qwen/Qwen2.5-7B-Instruct using HuggingFace transformers, generates triage
responses, and extracts hidden states at all 28 layers for downstream
interpretability analysis (probing, SAE, logit lens, TSV).

Hidden states are mean-pooled across the input token sequence at each layer
and saved as a [N_cases, 28, hidden_dim] float16 tensor.

Requires: ~18GB VRAM in float16 (A10G or better on Modal; Apple Silicon MPS
with 24GB+ unified memory for local testing).
Runtime: ~1-2 hours for 400 cases on A10G.
"""

import json
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    EMERGENCY_KEYWORDS,
    GEMMA_MODEL,
    OUTPUT_DIR,
    PHYSICIAN_TEST,
    REALWORLD_TEST,
    SEED,
    URGENT_KEYWORDS,
)
from src.utils import parse_triage_response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMMA_MODEL_ID = GEMMA_MODEL
MAX_NEW_TOKENS = 300

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "gemma2_base_results.json"
HIDDEN_STATES_PATH = OUTPUT_DIR / "gemma2_hidden_states.pt"


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data loading (same as 01_run_steerling_inference.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt construction (same system prompt as 01_run_steerling_inference.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)


def build_chat_messages(message):
    """Build chat messages list for Qwen 2.5 7B Instruct chat template.

    Uses the same system prompt as Steerling inference but formatted
    as a user turn (the system instruction is prepended to the user message).
    """
    user_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Patient message: {message}\n\n"
        f"Assessment:"
    )
    return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Step 20: Qwen 2.5 7B Instruct Inference + Hidden State Extraction")
    print("=" * 70)

    # Load test cases
    print("\nLoading test cases...")
    cases = load_test_cases()
    n_physician = sum(1 for c in cases if c["dataset"] == "physician")
    n_realworld = sum(1 for c in cases if c["dataset"] == "real-world")
    print(f"  Physician cases: {n_physician}")
    print(f"  Real-world cases: {n_realworld}")
    print(f"  Total: {len(cases)}")

    # Device and model
    device = select_device()
    print(f"\nDevice: {device}")
    print(f"Loading {GEMMA_MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device if device == "cuda" else None,
        output_hidden_states=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Confirm layer count from model config
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Loaded: {n_layers} layers, hidden_dim={hidden_dim}")
    if device == "cuda":
        print(f"  Memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # Preallocate hidden state accumulator (float32 for numerical stability
    # during accumulation; cast to float16 on save)
    all_hidden_states = torch.zeros(
        len(cases), n_layers, hidden_dim, dtype=torch.float32
    )

    results = []
    t0 = time.time()

    for idx, case in enumerate(tqdm(cases, desc="Running inference")):
        messages = build_chat_messages(case["message"])

        # Tokenize using chat template
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            # Step 1: Forward pass on input tokens to extract hidden states.
            # This is a single forward pass (no generation) so we get clean
            # hidden states for the prompt only.
            input_outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_states is a tuple of (n_layers + 1) tensors, each
            # [batch=1, seq_len, hidden_dim]. Index [0] is the embedding
            # layer output; [1:] are the transformer layer outputs.
            hidden_states = input_outputs.hidden_states[1:]  # skip embedding layer

            # Mean-pool across input sequence length per layer
            for layer_idx in range(n_layers):
                layer_hs = hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
                all_hidden_states[idx, layer_idx] = layer_hs.float().mean(dim=0).cpu()

            # Step 2: Generate the response (separate call to avoid holding
            # all intermediate hidden states in memory during generation).
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        # Decode only the generated tokens (strip input prefix)
        generated_ids = gen_outputs[0][input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse triage decision using same logic as Steerling pipeline
        triage = parse_triage_response(
            generated_text, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
        )

        results.append({
            **case,
            "gemma2_response": generated_text,
            "gemma2_detection": triage["detection"],
            "gemma2_severity": triage["severity"],
            "gemma2_action": triage["action"],
            "input_tokens": input_len,
            "generated_tokens": len(generated_ids),
        })

        # Periodic memory reporting on CUDA
        if device == "cuda" and (idx + 1) % 50 == 0:
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            tqdm.write(f"  [{idx + 1}/{len(cases)}] Peak CUDA memory: {mem_gb:.1f} GB")

    elapsed = time.time() - t0
    print(f"\nInference complete in {elapsed / 60:.1f} minutes")
    print(f"  Average: {elapsed / len(cases):.1f} sec/case")

    # -----------------------------------------------------------------------
    # Save results (no hidden states -- too large for JSON)
    # -----------------------------------------------------------------------
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {RESULTS_PATH}")

    # Save hidden states as float16 tensor
    all_hidden_states_f16 = all_hidden_states.half()
    torch.save(all_hidden_states_f16, HIDDEN_STATES_PATH)
    size_mb = HIDDEN_STATES_PATH.stat().st_size / 1e6
    print(f"Saved hidden states {list(all_hidden_states_f16.shape)} to {HIDDEN_STATES_PATH} ({size_mb:.0f} MB)")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    phys = [r for r in results if r["dataset"] == "physician"]
    rw = [r for r in results if r["dataset"] == "real-world"]

    for label, subset in [("Physician", phys), ("Real-world", rw), ("Overall", results)]:
        y_true = [r["detection_truth"] for r in subset]
        y_pred = [r["gemma2_detection"] for r in subset]
        tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
        fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
        tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
        fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tp + tn) / len(subset) if len(subset) > 0 else 0
        print(f"\n{label} set (N={len(subset)}):")
        print(f"  Sensitivity: {sens:.3f}  (TP={tp}, FN={fn})")
        print(f"  Specificity: {spec:.3f}  (TN={tn}, FP={fp})")
        print(f"  Accuracy:    {acc:.3f}")

    # Token length statistics
    input_lens = [r["input_tokens"] for r in results]
    gen_lens = [r["generated_tokens"] for r in results]
    print(f"\nToken statistics:")
    print(f"  Input tokens:  mean={sum(input_lens)/len(input_lens):.0f}, "
          f"min={min(input_lens)}, max={max(input_lens)}")
    print(f"  Output tokens: mean={sum(gen_lens)/len(gen_lens):.0f}, "
          f"min={min(gen_lens)}, max={max(gen_lens)}")


if __name__ == "__main__":
    main()
