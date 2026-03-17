# Extended Methods

This document provides a brief overview of the analysis pipeline. For complete methodological detail, see the Supplementary Information accompanying the manuscript.

## Pipeline Overview

| Step | Script | Input | Output | Runtime |
|------|--------|-------|--------|---------|
| 1 | `01_run_steerling_inference.py` | Test cases (400) | `output/steerling_base_results.json` | ~2-4 hrs (GPU) |
| 2 | `02_demographic_variation.py` | Physician cases (200) | `output/demographic_variation_results.json` | ~4-6 hrs (GPU) |
| 3 | `03_analyze_concepts.py` | Steps 1-2 outputs | `output/{concept_hazard_summary,concept_analysis_summary}.csv` | ~10 min (CPU) |
| 4 | `04_concept_steering.py` | Steps 1-3 outputs | `output/{causal_correction_results,tp_correction_results}.json` | ~8-12 hrs (GPU) |
| 5 | `05_generate_outputs.py` | All outputs | `tables/*.csv`, `figures/*.{pdf,png}` | ~2 min (CPU) |

## Statistical Methods

- **Confidence intervals:** Wilson score (proportions), BCa bootstrap with 1,000 resamples (MCC)
- **Multiple testing:** Benjamini-Hochberg FDR correction (q < 0.05) across 33,732 concepts per hazard category
- **Effect sizes:** Cohen's d (standardized mean difference)
- **Comparison tests:** McNemar's test with continuity correction (paired intervention comparisons)
- **Concept selection:** Leave-one-out cross-validation, top 20 concepts by absolute effect size

## Reproducibility

All stochastic operations use seed = 42. The pipeline is deterministic given fixed model weights and input data.
