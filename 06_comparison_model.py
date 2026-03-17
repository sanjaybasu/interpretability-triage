#!/usr/bin/env python3
"""Step 6: Non-CBM comparison model — inference, analysis, and erasure.

Tests whether the probe paradox (standard bias detection failing despite
pervasive encoding) is specific to concept bottleneck models or general
to standard LLMs.

Runs the same 5 bias detection methods on hidden states from a standard
(non-interpretable) LLM to enable direct comparison with Steerling-8B
concept activations.

Phases (each saves to disk; phases 2-3 can re-run independently):
  Phase 1: Inference + hidden state extraction (GPU, ~3-6h on MPS)
  Phase 2: Bias detection analysis (CPU, ~20min)
  Phase 3: LEACE/INLP erasure (CPU, ~10min)

Usage:
  python 06_comparison_model.py              # Run all phases
  python 06_comparison_model.py --phase 2    # Re-run analysis only
  python 06_comparison_model.py --phase 3    # Re-run erasure only
  python 06_comparison_model.py --model Qwen/Qwen2.5-7B-Instruct  # Alt model
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import (
    DEMOGRAPHIC_VARIATIONS,
    EMERGENCY_KEYWORDS,
    FDR_ALPHA,
    OUTPUT_DIR,
    PERMUTATION_N,
    PHYSICIAN_TEST,
    SEED,
    URGENT_KEYWORDS,
)
from src.utils import benjamini_hochberg, cohens_d, parse_triage_response

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default comparison model
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_physician_cases():
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


def build_prompt(message, demographic_prefix=None):
    parts = [SYSTEM_PROMPT, ""]
    if demographic_prefix:
        parts.append(demographic_prefix)
    parts.append(f"Patient message: {message}")
    parts.append("")
    parts.append("Assessment:")
    return "\n".join(parts)


def format_chat_messages(message, demographic_prefix=None):
    """Format as chat messages for models with chat templates."""
    user_content = ""
    if demographic_prefix:
        user_content += demographic_prefix + "\n"
    user_content += f"Patient message: {message}\n\nAssess whether this contains a clinical hazard."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ============================================================
# PHASE 1: Inference + hidden state extraction
# ============================================================

def phase1_inference(model_id):
    """Run comparison model on 600 demographic-varied inferences."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = select_device()
    print(f"Device: {device}")
    print(f"Loading model: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device if device == "cuda" else None,
        )
        if device == "mps":
            model = model.to(device)
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        if model_id != FALLBACK_MODEL:
            print(f"Trying fallback: {FALLBACK_MODEL}")
            return phase1_inference(FALLBACK_MODEL)
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model_short = model_id.split("/")[-1].lower().replace("-", "_")
    hidden_dim = model.config.hidden_size
    print(f"Hidden dim: {hidden_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    cases = load_physician_cases()
    variations = list(DEMOGRAPHIC_VARIATIONS.items())
    total = len(cases) * len(variations)
    print(f"Cases: {len(cases)}, Variations: {len(variations)}, Total: {total}")

    results = []
    hidden_states_list = []
    meta_list = []

    # Check for checkpoint
    ckpt_path = OUTPUT_DIR / f"comparison_{model_short}_checkpoint.json"
    start_idx = 0
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt["completed"]
        results = ckpt["results"]
        hidden_states_list = [np.array(h) for h in ckpt.get("hidden_states", [])]
        meta_list = ckpt.get("meta", [])
        print(f"Resuming from checkpoint: {start_idx}/{total}")

    idx = 0
    for case in tqdm(cases, desc="Cases"):
        for var_name, var_prefix in variations:
            if idx < start_idx:
                idx += 1
                continue

            # Build prompt using chat template if available
            try:
                messages = format_chat_messages(case["message"], var_prefix)
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = build_prompt(case["message"], var_prefix)

            inputs = tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)

            # Extract hidden states (single forward pass)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
                # Mean pool across token positions
                hidden_vec = last_hidden.mean(dim=1).squeeze(0)  # (hidden_dim,)
                hidden_np = hidden_vec.cpu().float().numpy()

            # Generate triage response
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # Decode only new tokens
                new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
                response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            triage = parse_triage_response(
                response_text, EMERGENCY_KEYWORDS, URGENT_KEYWORDS
            )

            results.append({
                "case_id": case["case_id"],
                "variation": var_name,
                "demographic_prefix": var_prefix or "",
                "detection_truth": case["detection_truth"],
                "action_truth": case["action_truth"],
                "hazard_category": case["hazard_category"],
                "response": response_text,
                "detection": triage["detection"],
                "action": triage["action"],
            })

            hidden_states_list.append(hidden_np)
            meta_list.append({"case_id": case["case_id"], "variation": var_name})
            idx += 1

            # Checkpoint every 50 inferences
            if idx % 50 == 0:
                print(f"  Checkpoint at {idx}/{total}")
                ckpt_data = {
                    "completed": idx,
                    "results": results,
                    "meta": meta_list,
                    # Don't save hidden states in JSON (too large); save separately
                }
                with open(ckpt_path, "w") as f:
                    json.dump(ckpt_data, f)
                np.save(
                    OUTPUT_DIR / f"comparison_{model_short}_hidden_states_partial.npy",
                    np.array(hidden_states_list),
                )

    # Save final outputs
    hidden_states = np.array(hidden_states_list)  # (600, hidden_dim)

    out_results = OUTPUT_DIR / f"comparison_{model_short}_results.json"
    with open(out_results, "w") as f:
        json.dump(results, f, indent=2, default=str)

    out_hidden = OUTPUT_DIR / f"comparison_{model_short}_hidden_states.npy"
    np.save(out_hidden, hidden_states)

    out_meta = OUTPUT_DIR / f"comparison_{model_short}_meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta_list, f, indent=2)

    # Save model info
    model_info = {
        "model_id": model_id,
        "model_short": model_short,
        "hidden_dim": hidden_dim,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_inferences": len(results),
        "device": device,
    }
    with open(OUTPUT_DIR / f"comparison_{model_short}_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()
    partial_path = OUTPUT_DIR / f"comparison_{model_short}_hidden_states_partial.npy"
    if partial_path.exists():
        partial_path.unlink()

    print(f"\nPhase 1 complete: {len(results)} inferences saved")
    print(f"  Hidden states: {hidden_states.shape}")
    print(f"  Results: {out_results}")

    # Print detection rate summary
    from collections import defaultdict
    by_var = defaultdict(list)
    for r in results:
        by_var[r["variation"]].append(r["detection"])

    print("\nDetection rate by demographic group:")
    ref_rate = np.mean(by_var.get("race_white", [0]))
    for var_name in sorted(by_var.keys()):
        rate = np.mean(by_var[var_name])
        diff = rate - ref_rate
        marker = " (ref)" if var_name == "race_white" else f" (delta={diff:+.3f})"
        print(f"  {var_name}: {rate:.3f}{marker}")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return model_short


# ============================================================
# PHASE 2: Bias detection analysis
# ============================================================

def phase2_analysis(model_short):
    """Run 5 bias detection methods on comparison model hidden states."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Bias Detection Analysis ({model_short})")
    print(f"{'='*60}")

    # Load data
    hidden_states = np.load(OUTPUT_DIR / f"comparison_{model_short}_hidden_states.npy")
    with open(OUTPUT_DIR / f"comparison_{model_short}_meta.json") as f:
        meta = json.load(f)

    n_features = hidden_states.shape[1]
    print(f"Hidden states: {hidden_states.shape}")

    # Deduplicate (average repeated case_id/variation pairs)
    pair_to_rows = {}
    for i, m in enumerate(meta):
        key = (m["case_id"], m["variation"])
        pair_to_rows.setdefault(key, []).append(i)

    dedup_keys = sorted(pair_to_rows.keys())
    dedup_weights = np.zeros((len(dedup_keys), n_features), dtype=np.float32)
    dedup_meta = []
    for i, key in enumerate(dedup_keys):
        rows = pair_to_rows[key]
        dedup_weights[i] = hidden_states[rows].mean(axis=0)
        dedup_meta.append({"case_id": key[0], "variation": key[1]})

    print(f"Deduplicated: {len(dedup_keys)} ({len(set(k[0] for k in dedup_keys))} cases)")

    # Build paired data
    case_to_idx = {}
    for i, m in enumerate(dedup_meta):
        case_to_idx.setdefault(m["case_id"], {})[m["variation"]] = i

    complete_cases = sorted([
        cid for cid, variants in case_to_idx.items()
        if all(v in variants for v in ["race_white", "race_black", "race_hispanic"])
    ])
    print(f"Complete triplets: {len(complete_cases)}")

    X_white = np.array([dedup_weights[case_to_idx[c]["race_white"]] for c in complete_cases])
    X_black = np.array([dedup_weights[case_to_idx[c]["race_black"]] for c in complete_cases])
    X_hispanic = np.array([dedup_weights[case_to_idx[c]["race_hispanic"]] for c in complete_cases])
    n_cases = len(complete_cases)

    X_all = np.vstack([X_white, X_black, X_hispanic])
    y_all = np.array(["white"] * n_cases + ["black"] * n_cases + ["hispanic"] * n_cases)

    # --- Method 1: Cross-sectional logistic probe ---
    print(f"\n--- Method 1: Cross-sectional Logistic Probe ---")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="saga", max_iter=5000,
        random_state=SEED, multi_class="multinomial",
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_accs = []
    for train_idx, test_idx in skf.split(X_all, y_enc):
        clf.fit(X_all[train_idx], y_enc[train_idx])
        fold_accs.append(clf.score(X_all[test_idx], y_enc[test_idx]))
    probe_acc = np.mean(fold_accs)
    print(f"  Fold accuracies: {[f'{a:.3f}' for a in fold_accs]}")
    print(f"  Mean CV accuracy: {probe_acc:.3f} (chance = 0.333)")

    # Fit on all data for direction extraction
    clf.fit(X_all, y_enc)

    # --- Method 2: INLP ---
    print(f"\n--- Method 2: INLP ---")
    X_inlp = X_all.copy()
    inlp_dirs = []
    inlp_accs = []
    for it in range(15):
        clf_inlp = LogisticRegression(
            penalty="l2", C=1.0, solver="saga", max_iter=3000,
            random_state=SEED + it, multi_class="multinomial",
        )
        skf_inlp = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + it)
        y_pred = cross_val_predict(clf_inlp, X_inlp, y_enc, cv=skf_inlp)
        acc = np.mean(y_pred == y_enc)
        inlp_accs.append(float(acc))
        if acc < 0.333 + 0.02:
            print(f"  INLP converged at iteration {it + 1} (acc={acc:.3f})")
            break
        clf_inlp.fit(X_inlp, y_enc)
        _, _, Vt = np.linalg.svd(clf_inlp.coef_, full_matrices=False)
        direction = Vt[0]
        inlp_dirs.append(direction)
        X_inlp = X_inlp - np.outer(X_inlp @ direction, direction)
        print(f"  Iteration {it + 1}: acc={acc:.3f}, removed direction")
    print(f"  Total directions removed: {len(inlp_dirs)}")

    # --- Method 3: PCA + ANOVA ---
    print(f"\n--- Method 3: PCA + ANOVA ---")
    pca = PCA(n_components=min(20, X_all.shape[0] - 1), random_state=SEED)
    scores = pca.fit_transform(X_all)
    labels = y_all

    anova_results = []
    for pc in range(min(5, scores.shape[1])):
        groups = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append(scores[i, pc])
        f_stat, p_val = stats.f_oneway(*groups.values())
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        anova_results.append({
            "pc": pc + 1,
            "variance_explained": float(pca.explained_variance_ratio_[pc]),
            "f_stat": float(f_stat),
            "p_value": float(p_val),
        })
        print(f"  PC{pc+1}: var={pca.explained_variance_ratio_[pc]:.3f}, "
              f"F={f_stat:.2f}, p={p_val:.4f} {sig}")

    # --- Method 4: Paired within-vignette t-tests ---
    print(f"\n--- Method 4: Paired t-tests with FDR correction ---")
    diff_results = []
    for var_name, X_var in [("race_black", X_black), ("race_hispanic", X_hispanic)]:
        diffs = X_var - X_white
        for j in range(n_features):
            d = diffs[:, j]
            if np.std(d) < 1e-12:
                t_stat, p_val = 0.0, 1.0
            else:
                t_stat, p_val = stats.ttest_1samp(d, 0)
            sd_diff = d.std(ddof=1) if d.std() > 0 else 1e-12
            diff_results.append({
                "feature_index": j,
                "variation": var_name,
                "mean_diff": float(d.mean()),
                "cohens_d": float(d.mean() / sd_diff),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            })

    df_diff = pd.DataFrame(diff_results)
    bh = benjamini_hochberg(df_diff["p_value"].values, alpha=FDR_ALPHA)
    df_diff["q_value_fdr"] = bh["q_values"]
    df_diff["significant_fdr"] = bh["rejected"]
    df_diff["p_bonferroni"] = np.minimum(df_diff["p_value"] * len(df_diff), 1.0)
    df_diff["significant_bonf"] = df_diff["p_bonferroni"] < 0.05

    n_fdr = df_diff["significant_fdr"].sum()
    n_bonf = df_diff["significant_bonf"].sum()
    n_total = len(df_diff)
    max_d = df_diff["cohens_d"].abs().max()
    print(f"  Total tests: {n_total}")
    print(f"  FDR-significant: {n_fdr} ({100*n_fdr/n_total:.1f}%)")
    print(f"  Bonferroni-significant: {n_bonf}")
    print(f"  Max |Cohen's d|: {max_d:.3f}")

    # --- Method 5: Permutation test ---
    print(f"\n--- Method 5: Permutation test ---")
    np.random.seed(SEED)

    def vectorized_fdr_count(diffs_arr, alpha=FDR_ALPHA):
        n = diffs_arr.shape[0]
        mean_d = diffs_arr.mean(axis=0)
        std_d = diffs_arr.std(axis=0, ddof=1)
        valid = std_d > 1e-12
        t_stat = np.zeros_like(mean_d)
        t_stat[valid] = mean_d[valid] / (std_d[valid] / np.sqrt(n))
        p_values = np.ones_like(mean_d)
        p_values[valid] = 2.0 * stats.t.sf(np.abs(t_stat[valid]), df=n - 1)
        return benjamini_hochberg(p_values, alpha)["n_rejected"]

    null_counts = []
    n_perms = min(PERMUTATION_N, 200)
    for perm_i in tqdm(range(n_perms), desc="Permutation test"):
        n_sig = 0
        for var_name, X_var in [("race_black", X_black), ("race_hispanic", X_hispanic)]:
            stacked = np.stack([X_white, X_var])  # (2, n_cases, n_features)
            perm_diffs = np.zeros_like(X_white)
            for c in range(n_cases):
                if np.random.random() < 0.5:
                    perm_diffs[c] = stacked[1, c] - stacked[0, c]
                else:
                    perm_diffs[c] = stacked[0, c] - stacked[1, c]
            n_sig += vectorized_fdr_count(perm_diffs)
        null_counts.append(n_sig)

    null_arr = np.array(null_counts)
    n_exceed = (null_arr >= n_fdr).sum()
    p_perm = (n_exceed + 1) / (n_perms + 1)
    print(f"  Observed FDR-sig: {n_fdr}")
    print(f"  Null mean: {null_arr.mean():.1f} +/- {null_arr.std():.1f}")
    print(f"  Permutation p-value: {p_perm}")

    # Save all results
    df_diff.to_csv(OUTPUT_DIR / f"comparison_{model_short}_differential.csv", index=False)

    summary = {
        "model_id": model_short,
        "n_features": n_features,
        "n_cases": n_cases,
        "probe": {
            "cv_accuracy": float(probe_acc),
            "fold_accuracies": [float(a) for a in fold_accs],
            "chance": 1 / 3,
        },
        "inlp": {
            "directions_removed": len(inlp_dirs),
            "accuracy_trajectory": inlp_accs,
        },
        "pca": {
            "pc1_variance": float(pca.explained_variance_ratio_[0]),
            "top5_variance": float(sum(pca.explained_variance_ratio_[:5])),
            "anova_results": anova_results,
        },
        "paired_tests": {
            "n_total_tests": n_total,
            "n_fdr_significant": int(n_fdr),
            "n_bonf_significant": int(n_bonf),
            "pct_fdr_significant": float(n_fdr / n_total) if n_total > 0 else 0,
            "max_cohens_d": float(max_d),
        },
        "permutation": {
            "observed_n_significant": int(n_fdr),
            "null_mean": float(null_arr.mean()),
            "null_sd": float(null_arr.std()),
            "null_max": int(null_arr.max()),
            "p_value": float(p_perm),
        },
    }

    with open(OUTPUT_DIR / f"comparison_{model_short}_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 2 complete. Analysis saved.")
    return summary


# ============================================================
# PHASE 3: LEACE/INLP erasure
# ============================================================

def phase3_erasure(model_short):
    """Run LEACE and INLP concept erasure on comparison model hidden states."""
    print(f"\n{'='*60}")
    print(f"Phase 3: LEACE/INLP Erasure ({model_short})")
    print(f"{'='*60}")

    # Load data
    hidden_states = np.load(OUTPUT_DIR / f"comparison_{model_short}_hidden_states.npy")
    with open(OUTPUT_DIR / f"comparison_{model_short}_meta.json") as f:
        meta = json.load(f)

    n_features = hidden_states.shape[1]

    # Deduplicate
    pair_to_rows = {}
    for i, m in enumerate(meta):
        key = (m["case_id"], m["variation"])
        pair_to_rows.setdefault(key, []).append(i)

    dedup_keys = sorted(pair_to_rows.keys())
    dedup_weights = np.zeros((len(dedup_keys), n_features), dtype=np.float32)
    dedup_meta = []
    for i, key in enumerate(dedup_keys):
        rows = pair_to_rows[key]
        dedup_weights[i] = hidden_states[rows].mean(axis=0)
        dedup_meta.append({"case_id": key[0], "variation": key[1]})

    # Build paired data
    case_to_idx = {}
    for i, m in enumerate(dedup_meta):
        case_to_idx.setdefault(m["case_id"], {})[m["variation"]] = i

    complete_cases = sorted([
        cid for cid, variants in case_to_idx.items()
        if all(v in variants for v in ["race_white", "race_black", "race_hispanic"])
    ])

    X_white = np.array([dedup_weights[case_to_idx[c]["race_white"]] for c in complete_cases])
    X_black = np.array([dedup_weights[case_to_idx[c]["race_black"]] for c in complete_cases])
    X_hispanic = np.array([dedup_weights[case_to_idx[c]["race_hispanic"]] for c in complete_cases])
    n_cases = len(complete_cases)

    X_all = np.vstack([X_white, X_black, X_hispanic])
    y_all = np.array(["white"] * n_cases + ["black"] * n_cases + ["hispanic"] * n_cases)

    # --- Baseline probe ---
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="saga", max_iter=5000,
        random_state=SEED, multi_class="multinomial",
    )
    clf.fit(X_all, y_enc)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_pred_cv = cross_val_predict(clf, X_all, y_enc, cv=skf)
    baseline_acc = np.mean(y_pred_cv == y_enc)

    def run_diff_analysis(Xw, Xb, Xh, label):
        results = []
        for var_name, X_var in [("race_black", Xb), ("race_hispanic", Xh)]:
            diffs = X_var - Xw
            for j in range(n_features):
                d = diffs[:, j]
                if np.std(d) < 1e-12:
                    t_stat, p_val = 0.0, 1.0
                else:
                    t_stat, p_val = stats.ttest_1samp(d, 0)
                sd = d.std(ddof=1) if d.std() > 0 else 1e-12
                results.append({
                    "feature_index": j, "variation": var_name,
                    "mean_diff": float(d.mean()), "cohens_d": float(d.mean() / sd),
                    "t_stat": float(t_stat), "p_value": float(p_val),
                })
        df = pd.DataFrame(results)
        bh = benjamini_hochberg(df["p_value"].values, alpha=FDR_ALPHA)
        df["q_value_fdr"] = bh["q_values"]
        df["significant_fdr"] = bh["rejected"]
        n_sig = df["significant_fdr"].sum()
        max_d = df["cohens_d"].abs().max()
        print(f"  [{label}] FDR-sig: {n_sig}/{len(df)}, max |d|: {max_d:.4f}")
        return df, int(n_sig), float(max_d)

    # Baseline differential activation
    df_orig, n_fdr_orig, max_d_orig = run_diff_analysis(
        X_white, X_black, X_hispanic, "original"
    )

    # --- LEACE (mean difference) ---
    print("\n--- LEACE (mean difference directions) ---")
    diff_black = (X_black - X_white).mean(axis=0)
    diff_hispanic = (X_hispanic - X_white).mean(axis=0)
    diff_stack = np.vstack([diff_black, diff_hispanic])
    _, S_diff, Vt_diff = np.linalg.svd(diff_stack, full_matrices=False)
    race_basis_diff = Vt_diff[:2]
    print(f"  Singular values: {S_diff}")

    X_all_ld = X_all - X_all @ race_basis_diff.T @ race_basis_diff
    X_w_ld = X_all_ld[:n_cases]
    X_b_ld = X_all_ld[n_cases:2*n_cases]
    X_h_ld = X_all_ld[2*n_cases:]

    y_pred_ld = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="saga", max_iter=5000,
                          random_state=SEED, multi_class="multinomial"),
        X_all_ld, y_enc, cv=skf,
    )
    acc_ld = np.mean(y_pred_ld == y_enc)
    df_ld, n_fdr_ld, max_d_ld = run_diff_analysis(X_w_ld, X_b_ld, X_h_ld, "LEACE-diff")

    # PCA preservation
    pca_orig = PCA(n_components=5, random_state=SEED).fit(X_all)
    pca_ld = PCA(n_components=5, random_state=SEED).fit(X_all_ld)
    pc1_var_orig = float(pca_orig.explained_variance_ratio_[0])
    pc1_var_ld = float(pca_ld.explained_variance_ratio_[0])
    print(f"  PC1 variance: {pc1_var_orig:.3f} -> {pc1_var_ld:.3f}")

    # --- LEACE (probe directions) ---
    print("\n--- LEACE (probe weight directions) ---")
    _, _, Vt_probe = np.linalg.svd(clf.coef_, full_matrices=False)
    race_basis_probe = Vt_probe[:2]

    X_all_lp = X_all - X_all @ race_basis_probe.T @ race_basis_probe
    X_w_lp = X_all_lp[:n_cases]
    X_b_lp = X_all_lp[n_cases:2*n_cases]
    X_h_lp = X_all_lp[2*n_cases:]

    y_pred_lp = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="saga", max_iter=5000,
                          random_state=SEED, multi_class="multinomial"),
        X_all_lp, y_enc, cv=skf,
    )
    acc_lp = np.mean(y_pred_lp == y_enc)
    df_lp, n_fdr_lp, max_d_lp = run_diff_analysis(X_w_lp, X_b_lp, X_h_lp, "LEACE-probe")

    pca_lp = PCA(n_components=5, random_state=SEED).fit(X_all_lp)
    pc1_var_lp = float(pca_lp.explained_variance_ratio_[0])

    # --- INLP ---
    print("\n--- INLP ---")
    X_inlp = X_all.copy()
    inlp_dirs = []
    inlp_accs = []
    for it in range(15):
        clf_it = LogisticRegression(
            penalty="l2", C=1.0, solver="saga", max_iter=3000,
            random_state=SEED + it, multi_class="multinomial",
        )
        skf_it = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + it)
        y_pred_it = cross_val_predict(clf_it, X_inlp, y_enc, cv=skf_it)
        acc_it = np.mean(y_pred_it == y_enc)
        inlp_accs.append(float(acc_it))
        if acc_it < 0.333 + 0.02:
            print(f"  INLP converged at iteration {it + 1} (acc={acc_it:.3f})")
            break
        clf_it.fit(X_inlp, y_enc)
        _, _, Vt_it = np.linalg.svd(clf_it.coef_, full_matrices=False)
        direction = Vt_it[0]
        inlp_dirs.append(direction)
        X_inlp = X_inlp - np.outer(X_inlp @ direction, direction)
        print(f"  Iteration {it + 1}: acc={acc_it:.3f}")

    X_w_inlp = X_inlp[:n_cases]
    X_b_inlp = X_inlp[n_cases:2*n_cases]
    X_h_inlp = X_inlp[2*n_cases:]
    df_inlp, n_fdr_inlp, max_d_inlp = run_diff_analysis(
        X_w_inlp, X_b_inlp, X_h_inlp, "INLP"
    )
    pca_inlp = PCA(n_components=5, random_state=SEED).fit(X_inlp)
    pc1_var_inlp = float(pca_inlp.explained_variance_ratio_[0])

    # --- Summary ---
    fdr_red_ld = 100 * (1 - n_fdr_ld / n_fdr_orig) if n_fdr_orig > 0 else 0
    fdr_red_lp = 100 * (1 - n_fdr_lp / n_fdr_orig) if n_fdr_orig > 0 else 0
    fdr_red_inlp = 100 * (1 - n_fdr_inlp / n_fdr_orig) if n_fdr_orig > 0 else 0

    summary = {
        "original": {
            "probe_cv_accuracy": float(baseline_acc),
            "n_fdr_significant": n_fdr_orig,
            "max_cohens_d": max_d_orig,
            "pca_pc1_variance": pc1_var_orig,
        },
        "leace_mean_diff": {
            "probe_cv_accuracy": float(acc_ld),
            "n_fdr_significant": n_fdr_ld,
            "max_cohens_d": max_d_ld,
            "pca_pc1_variance": pc1_var_ld,
            "directions_removed": 2,
        },
        "leace_probe": {
            "probe_cv_accuracy": float(acc_lp),
            "n_fdr_significant": n_fdr_lp,
            "max_cohens_d": max_d_lp,
            "pca_pc1_variance": pc1_var_lp,
            "directions_removed": 2,
        },
        "inlp": {
            "n_fdr_significant": n_fdr_inlp,
            "max_cohens_d": max_d_inlp,
            "pca_pc1_variance": pc1_var_inlp,
            "directions_removed": len(inlp_dirs),
            "accuracy_trajectory": inlp_accs,
        },
        "reductions": {
            "fdr_reduction_leace_diff_pct": fdr_red_ld,
            "fdr_reduction_leace_probe_pct": fdr_red_lp,
            "fdr_reduction_inlp_pct": fdr_red_inlp,
        },
        "metadata": {
            "n_cases": n_cases,
            "n_features": n_features,
            "seed": SEED,
        },
    }

    with open(OUTPUT_DIR / f"comparison_{model_short}_erasure.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save differential activation CSVs
    df_ld.to_csv(OUTPUT_DIR / f"comparison_{model_short}_diff_leace.csv", index=False)
    df_inlp.to_csv(OUTPUT_DIR / f"comparison_{model_short}_diff_inlp.csv", index=False)

    print(f"\n{'Method':<20} {'Probe':<8} {'FDR Sig':<10} {'Max |d|':<10} {'PC1 Var':<8}")
    print("-" * 56)
    for method, s in summary.items():
        if method in ("reductions", "metadata"):
            continue
        acc = s.get("probe_cv_accuracy", "—")
        acc_str = f"{acc:.3f}" if isinstance(acc, float) else acc
        print(f"{method:<20} {acc_str:<8} {s['n_fdr_significant']:<10} "
              f"{s['max_cohens_d']:<10.4f} {s['pca_pc1_variance']:.3f}")

    print(f"\nFDR reduction: LEACE-diff {fdr_red_ld:.1f}%, "
          f"LEACE-probe {fdr_red_lp:.1f}%, INLP {fdr_red_inlp:.1f}%")
    print("Phase 3 complete.")
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Non-CBM comparison model analysis")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1/2/3) or 0 for all")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    if args.phase == 0 or args.phase == 1:
        model_short = phase1_inference(args.model)

    if args.phase == 0 or args.phase == 2:
        phase2_analysis(model_short)

    if args.phase == 0 or args.phase == 3:
        phase3_erasure(model_short)

    print(f"\nAll phases complete for {model_short}.")


if __name__ == "__main__":
    main()
