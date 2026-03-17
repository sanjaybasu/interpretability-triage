#!/usr/bin/env python3
"""Step 7: NLP analysis of triage response text across demographic groups.

Tests whether natural language explanations — the most accessible form of
interpretability — reveal demographic bias in model responses. Analyzes
response text from both Steerling-8B and the comparison model without
requiring access to internal representations.

Analyses:
  1. Response length differences by demographic group
  2. Urgency keyword frequency differences
  3. Semantic divergence (cosine similarity between same-vignette responses)
  4. Lexical divergence (TF-IDF for demographic-specific language)
  5. Triage decision consistency across demographic variants

This represents "natural language interpretability" — testing whether a
clinician reading the model's output could detect demographic bias.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DEMOGRAPHIC_VARIATIONS,
    EMERGENCY_KEYWORDS,
    FDR_ALPHA,
    OUTPUT_DIR,
    SEED,
    URGENT_KEYWORDS,
)
from src.utils import benjamini_hochberg

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_steerling_responses():
    """Load Steerling-8B demographic variation responses."""
    path = OUTPUT_DIR / "demographic_variation_results.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def load_comparison_responses(model_short):
    """Load comparison model responses."""
    path = OUTPUT_DIR / f"comparison_{model_short}_results.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def build_triplets(responses, response_key="steerling_response", detection_key="steerling_detection"):
    """Build case triplets (white, black, hispanic) from response data."""
    case_map = {}
    for r in responses:
        cid = r["case_id"]
        var = r["variation"]
        case_map.setdefault(cid, {})[var] = r

    triplets = []
    for cid in sorted(case_map.keys()):
        variants = case_map[cid]
        if all(v in variants for v in ["race_white", "race_black", "race_hispanic"]):
            triplets.append({
                "case_id": cid,
                "white": variants["race_white"],
                "black": variants["race_black"],
                "hispanic": variants["race_hispanic"],
            })
    return triplets


def analyze_response_length(triplets, response_key):
    """Test whether response length differs by demographic group."""
    lengths = {"white": [], "black": [], "hispanic": []}
    for t in triplets:
        for race in lengths:
            text = t[race].get(response_key, "")
            lengths[race].append(len(text))

    # Paired tests
    w = np.array(lengths["white"])
    b = np.array(lengths["black"])
    h = np.array(lengths["hispanic"])

    results = {
        "mean_length": {race: float(np.mean(v)) for race, v in lengths.items()},
        "std_length": {race: float(np.std(v)) for race, v in lengths.items()},
    }

    # Paired t-tests
    t_wb, p_wb = stats.ttest_rel(w, b)
    t_wh, p_wh = stats.ttest_rel(w, h)
    d_wb = (w - b).mean() / (w - b).std(ddof=1) if (w - b).std() > 0 else 0
    d_wh = (w - h).mean() / (w - h).std(ddof=1) if (w - h).std() > 0 else 0

    results["white_vs_black"] = {
        "mean_diff": float((w - b).mean()),
        "cohens_d": float(d_wb),
        "t_stat": float(t_wb),
        "p_value": float(p_wb),
    }
    results["white_vs_hispanic"] = {
        "mean_diff": float((w - h).mean()),
        "cohens_d": float(d_wh),
        "t_stat": float(t_wh),
        "p_value": float(p_wh),
    }

    return results


def analyze_urgency_keywords(triplets, response_key):
    """Test whether urgency keyword usage differs by demographic group."""
    all_keywords = EMERGENCY_KEYWORDS + URGENT_KEYWORDS

    counts = {"white": [], "black": [], "hispanic": []}
    emergency_counts = {"white": [], "black": [], "hispanic": []}
    urgent_counts = {"white": [], "black": [], "hispanic": []}

    for t in triplets:
        for race in counts:
            text = t[race].get(response_key, "").lower()
            n_emerg = sum(1 for kw in EMERGENCY_KEYWORDS if kw in text)
            n_urgent = sum(1 for kw in URGENT_KEYWORDS if kw in text)
            counts[race].append(n_emerg + n_urgent)
            emergency_counts[race].append(n_emerg)
            urgent_counts[race].append(n_urgent)

    results = {}
    for label, data in [("any_keyword", counts), ("emergency", emergency_counts),
                        ("urgent", urgent_counts)]:
        w = np.array(data["white"])
        b = np.array(data["black"])
        h = np.array(data["hispanic"])

        t_wb, p_wb = stats.ttest_rel(w, b) if w.std() + b.std() > 0 else (0, 1)
        t_wh, p_wh = stats.ttest_rel(w, h) if w.std() + h.std() > 0 else (0, 1)

        results[label] = {
            "mean": {race: float(np.mean(v)) for race, v in data.items()},
            "white_vs_black": {"mean_diff": float((w - b).mean()), "p_value": float(p_wb)},
            "white_vs_hispanic": {"mean_diff": float((w - h).mean()), "p_value": float(p_wh)},
        }

    return results


def analyze_semantic_divergence(triplets, response_key):
    """Measure semantic similarity between same-vignette responses across races.

    Uses TF-IDF vectors as a lightweight proxy for semantic similarity.
    Lower within-vignette similarity = more demographic influence on response.
    """
    wb_sims = []
    wh_sims = []
    bh_sims = []

    # Collect all texts for TF-IDF fitting
    all_texts = []
    text_map = []  # (triplet_idx, race)
    for i, t in enumerate(triplets):
        for race in ["white", "black", "hispanic"]:
            text = t[race].get(response_key, "")
            all_texts.append(text)
            text_map.append((i, race))

    if not all_texts:
        return {}

    # Fit TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(all_texts)

    # Compute within-vignette similarities
    for i, t in enumerate(triplets):
        idx_w = i * 3 + 0
        idx_b = i * 3 + 1
        idx_h = i * 3 + 2

        sim_wb = cosine_similarity(tfidf_matrix[idx_w], tfidf_matrix[idx_b])[0, 0]
        sim_wh = cosine_similarity(tfidf_matrix[idx_w], tfidf_matrix[idx_h])[0, 0]
        sim_bh = cosine_similarity(tfidf_matrix[idx_b], tfidf_matrix[idx_h])[0, 0]

        wb_sims.append(sim_wb)
        wh_sims.append(sim_wh)
        bh_sims.append(sim_bh)

    results = {
        "white_black_similarity": {
            "mean": float(np.mean(wb_sims)),
            "std": float(np.std(wb_sims)),
            "median": float(np.median(wb_sims)),
        },
        "white_hispanic_similarity": {
            "mean": float(np.mean(wh_sims)),
            "std": float(np.std(wh_sims)),
            "median": float(np.median(wh_sims)),
        },
        "black_hispanic_similarity": {
            "mean": float(np.mean(bh_sims)),
            "std": float(np.std(bh_sims)),
            "median": float(np.median(bh_sims)),
        },
        "mean_within_vignette_similarity": float(np.mean(wb_sims + wh_sims + bh_sims)),
    }

    # Compute between-vignette similarity for comparison (baseline noise)
    n = len(triplets)
    if n > 10:
        between_sims = []
        np.random.seed(SEED)
        for _ in range(min(1000, n * 10)):
            i, j = np.random.choice(n, 2, replace=False)
            race_i = np.random.choice(["white", "black", "hispanic"])
            race_j = np.random.choice(["white", "black", "hispanic"])
            idx_i = i * 3 + ["white", "black", "hispanic"].index(race_i)
            idx_j = j * 3 + ["white", "black", "hispanic"].index(race_j)
            between_sims.append(
                cosine_similarity(tfidf_matrix[idx_i], tfidf_matrix[idx_j])[0, 0]
            )
        results["between_vignette_similarity"] = {
            "mean": float(np.mean(between_sims)),
            "std": float(np.std(between_sims)),
        }

    return results


def analyze_lexical_differences(triplets, response_key):
    """Find words/phrases that appear differentially by demographic group."""
    texts_by_race = {"white": [], "black": [], "hispanic": []}
    for t in triplets:
        for race in texts_by_race:
            texts_by_race[race].append(t[race].get(response_key, ""))

    # Concatenate per race
    race_docs = {race: " ".join(texts) for race, texts in texts_by_race.items()}

    # TF-IDF on per-race documents
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(list(race_docs.values()))
    feature_names = tfidf.get_feature_names_out()

    # Find terms that are most distinctive per race
    results = {}
    race_names = list(race_docs.keys())
    for i, race in enumerate(race_names):
        scores = matrix[i].toarray().flatten()
        mean_others = np.mean([matrix[j].toarray().flatten() for j in range(3) if j != i], axis=0)
        diff = scores - mean_others
        top_idx = np.argsort(diff)[-10:][::-1]
        results[f"distinctive_{race}"] = [
            {"term": feature_names[j], "score": float(diff[j])}
            for j in top_idx if diff[j] > 0
        ]

    return results


def analyze_decision_consistency(triplets, detection_key):
    """Measure triage decision consistency across demographic variants."""
    consistent = 0
    inconsistent = 0
    flip_patterns = Counter()

    for t in triplets:
        dets = {
            race: t[race].get(detection_key, 0)
            for race in ["white", "black", "hispanic"]
        }
        vals = list(dets.values())
        if len(set(vals)) == 1:
            consistent += 1
        else:
            inconsistent += 1
            pattern = f"W={dets['white']},B={dets['black']},H={dets['hispanic']}"
            flip_patterns[pattern] += 1

    total = consistent + inconsistent
    results = {
        "n_total": total,
        "n_consistent": consistent,
        "n_inconsistent": inconsistent,
        "consistency_rate": float(consistent / total) if total > 0 else 0,
        "flip_patterns": dict(flip_patterns.most_common(10)),
    }

    # Detection rates by race
    for race in ["white", "black", "hispanic"]:
        detections = [t[race].get(detection_key, 0) for t in triplets]
        results[f"detection_rate_{race}"] = float(np.mean(detections))

    return results


def analyze_one_model(responses, model_name, response_key, detection_key):
    """Run all NLP analyses for one model."""
    print(f"\n{'='*60}")
    print(f"NLP Analysis: {model_name}")
    print(f"{'='*60}")

    triplets = build_triplets(responses, response_key, detection_key)
    print(f"Complete triplets: {len(triplets)}")

    if not triplets:
        print("No complete triplets found!")
        return None

    results = {"model": model_name, "n_triplets": len(triplets)}

    print("\n--- Response Length ---")
    length_results = analyze_response_length(triplets, response_key)
    results["response_length"] = length_results
    for race, mean in length_results["mean_length"].items():
        print(f"  {race}: {mean:.0f} chars (sd={length_results['std_length'][race]:.0f})")
    for comp in ["white_vs_black", "white_vs_hispanic"]:
        r = length_results[comp]
        print(f"  {comp}: diff={r['mean_diff']:.1f}, d={r['cohens_d']:.3f}, p={r['p_value']:.4f}")

    print("\n--- Urgency Keywords ---")
    kw_results = analyze_urgency_keywords(triplets, response_key)
    results["urgency_keywords"] = kw_results
    for label, data in kw_results.items():
        means = data["mean"]
        print(f"  {label}: W={means['white']:.2f}, B={means['black']:.2f}, H={means['hispanic']:.2f}")

    print("\n--- Semantic Divergence ---")
    sem_results = analyze_semantic_divergence(triplets, response_key)
    results["semantic_divergence"] = sem_results
    if sem_results:
        print(f"  Within-vignette similarity (W-B): {sem_results['white_black_similarity']['mean']:.3f}")
        print(f"  Within-vignette similarity (W-H): {sem_results['white_hispanic_similarity']['mean']:.3f}")
        if "between_vignette_similarity" in sem_results:
            print(f"  Between-vignette similarity: {sem_results['between_vignette_similarity']['mean']:.3f}")

    print("\n--- Lexical Differences ---")
    lex_results = analyze_lexical_differences(triplets, response_key)
    results["lexical_differences"] = lex_results
    for race in ["white", "black", "hispanic"]:
        key = f"distinctive_{race}"
        if key in lex_results and lex_results[key]:
            terms = [d["term"] for d in lex_results[key][:5]]
            print(f"  Distinctive for {race}: {', '.join(terms)}")

    print("\n--- Decision Consistency ---")
    dec_results = analyze_decision_consistency(triplets, detection_key)
    results["decision_consistency"] = dec_results
    print(f"  Consistent: {dec_results['n_consistent']}/{dec_results['n_total']} "
          f"({dec_results['consistency_rate']:.1%})")
    print(f"  Detection rates: W={dec_results['detection_rate_white']:.3f}, "
          f"B={dec_results['detection_rate_black']:.3f}, "
          f"H={dec_results['detection_rate_hispanic']:.3f}")
    if dec_results["flip_patterns"]:
        print(f"  Top flip patterns: {dec_results['flip_patterns']}")

    return results


def main():
    np.random.seed(SEED)
    all_results = {}

    # Steerling-8B
    steerling_data = load_steerling_responses()
    if steerling_data:
        steerling_results = analyze_one_model(
            steerling_data, "Steerling-8B",
            response_key="steerling_response",
            detection_key="steerling_detection",
        )
        if steerling_results:
            all_results["steerling"] = steerling_results

    # Comparison model(s)
    import glob
    for path in sorted(Path(OUTPUT_DIR).glob("comparison_*_results.json")):
        model_short = path.stem.replace("comparison_", "").replace("_results", "")
        print(f"\nFound comparison model: {model_short}")
        comp_data = load_comparison_responses(model_short)
        if comp_data:
            comp_results = analyze_one_model(
                comp_data, model_short,
                response_key="response",
                detection_key="detection",
            )
            if comp_results:
                all_results[model_short] = comp_results

    # Save combined results
    out_path = OUTPUT_DIR / "nlp_response_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll NLP results saved to {out_path}")

    # Print comparison table if multiple models
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'Consistency':<15} {'W-B sim':<10} {'W-H sim':<10}")
        print("-" * 60)
        for name, r in all_results.items():
            cons = r.get("decision_consistency", {}).get("consistency_rate", 0)
            wb_sim = r.get("semantic_divergence", {}).get("white_black_similarity", {}).get("mean", 0)
            wh_sim = r.get("semantic_divergence", {}).get("white_hispanic_similarity", {}).get("mean", 0)
            print(f"{name:<25} {cons:<15.1%} {wb_sim:<10.3f} {wh_sim:<10.3f}")


if __name__ == "__main__":
    main()
