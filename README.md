# Interpretability without actionability: mechanistic intervention methods fail to correct clinical triage errors in language models

Analysis pipeline for the manuscript submitted to *The Lancet Digital Health* (Research Article).

**Authors:** Basu S, Patel SY, Sheth P, Muralidharan B, Elamaran N, Kinra A, Morgan J, Batniji R

## Overview

Four-arm systematic comparison of mechanistic interpretability methods for correcting false-negative triage errors in two language models, evaluated on 400 physician-adjudicated clinical vignettes (144 hazards, 256 benign).

### Arm 1: Concept Bottleneck Steering (Steerling-8B)
33,732 supervised medical concepts; test-time intervention via `steer_known` at five alpha levels, random-concept controls, prompt engineering, and in-distribution TP-mean correction.

### Arm 2: SAE Feature Steering (Qwen 2.5 7B)
Sparse autoencoder trained from scratch (16,384-width, layer 14). Training succeeded; downstream steering failed.

### Arm 3: Logit Lens + Activation Patching (Qwen 2.5 7B)
Hazard token rank tracking across 28 layers. Best FN rank: 15,020/152,064. No patchable direction identified.

### Arm 4: Linear Probing + TSV Steering (Qwen 2.5 7B)
Per-layer logistic regression probes (AUROC 0.982 at layer 23). TSV degenerate (n_TP=0).

## Requirements

### Hardware
- Apple M3 Max (Arm 1 local analyses)
- NVIDIA A100 80 GB (Arms 2-4 via Modal cloud computing)
- NVIDIA A10 GPUs (Arm 1 concept intervention experiments via Modal)

### Software
- Python 3.13 (Arm 1), Python 3.11 (Arms 2-4)
- CUDA 12.1+ (for GPU steps)

## Pipeline Scripts

### Arm 1 (Steerling-8B, local + Modal)
1. `01_steerling_base.py` -- Baseline inference (400 cases) with concept activation extraction
2. `02_demographic_variation.py` -- Inference with demographic prefixes (600 inferences)
3. `03b_analyze_concept_weights.py` -- Statistical analysis of concept activations
4. `04c_concept_erasure.py` -- LEACE/INLP concept erasure
5. `09_concept_safety_alignment.py` -- Concept-hazard association with leave-one-out concept selection
6. `10_modal_run.py` -- Out-of-distribution causal correction (Modal A10 GPUs)
7. `10c_tp_correction_modal.py` -- In-distribution TP-mean correction (Modal A10 GPUs)
8. `11_tables_figures.py` -- Table and figure generation
9. `12_concept_distribution_analysis.py` -- Concept activation distribution analysis

### Arms 2-4 (Qwen 2.5 7B, Modal A100)
10. `20_gemma_base_inference.py` -- Qwen 2.5 7B base inference + hidden state extraction (28 layers)
11. `21_sae_steering.py` -- SAE training + feature extraction + steering attempt
12. `22_logit_lens.py` -- Logit lens + activation patching
13. `23_probing_tsv.py` -- Per-layer probing + TSV computation + steering attempt
14. `25_comparative_analysis.py` -- Head-to-head comparison across all 4 arms
15. `modal_gemma_pipeline.py` -- Modal cloud orchestration for Arms 2-4

### Infrastructure
- `config.py` -- Centralized parameters (model paths, layer counts, SAE width, seeds)
- `src/utils.py` -- Statistical utilities (Wilson CI, BCa bootstrap, McNemar)
- `launch_pipeline.py`, `check_status.py` -- Pipeline management

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GEMMA_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Arms 2-4 model |
| `GEMMA_N_LAYERS` | 28 | Transformer layers |
| `SAE_WIDTH` | 16,384 | SAE bottleneck width |
| `SEED` | 42 | Random seed for all operations |
| `N_BOOTSTRAP` | 1,000 | Bootstrap resamples |
| `TOP_K_CONCEPTS` | 20 | Concepts per hazard category |

## Outputs

### JSON (in `output/`)
- `probe_results.json` -- Per-layer probe accuracy and AUROC (28 layers)
- `logit_lens_summary.json` -- Hazard token ranks by layer
- `tsv_analysis.json` -- TSV computation results
- `activation_patching_summary.json` -- Patching results
- `causal_correction_results.json` -- Arm 1 intervention results
- `comparative_analysis.json` -- Cross-arm summary

### Data
- **Physician-created vignettes** (N=200): 18 hazard categories, adjudicated by a board-certified physician (JM)
- **Real-world encounter notes** (N=200): De-identified Medicaid patient encounter messages

## Citation

```bibtex
@article{basu2026interpretability,
  title={Interpretability without actionability: mechanistic intervention
         methods fail to correct clinical triage errors in language models},
  author={Basu, Sanjay and Patel, Sadiq Y. and Sheth, Parth and
          Muralidharan, Bhairavi and Elamaran, Namrata and
          Kinra, Aakriti and Morgan, John and Batniji, Rajaie},
  journal={The Lancet Digital Health},
  year={2026},
  note={Under review}
}
```

## License

Apache License 2.0.
