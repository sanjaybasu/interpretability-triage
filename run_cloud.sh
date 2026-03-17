#!/usr/bin/env bash
# Cloud GPU execution script for concept attribution triage pipeline.
# Designed for A100/H100 instances (Lambda, RunPod, Vast.ai, etc.).
#
# Usage:
#   1. Upload this repo to the cloud instance
#   2. chmod +x run_cloud.sh
#   3. ./run_cloud.sh 2>&1 | tee pipeline.log
#
# Prerequisites: NVIDIA GPU with >=24GB VRAM, CUDA 12+, Python 3.10+

set -euo pipefail

echo "=== Concept Attribution Triage Pipeline ==="
echo "Started: $(date)"
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. This script requires an NVIDIA GPU."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Setup Python environment
PYTHON=${PYTHON:-python3}
echo "Python: $($PYTHON --version)"
echo "Setting up environment..."

$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet -e ".[dev]"

# Verify steerling is importable
$PYTHON -c "import steerling; print(f'steerling {steerling.__version__}')"
$PYTHON -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run tests first
echo ""
echo "=== Running unit tests ==="
$PYTHON -m pytest tests/ -v --tb=short
echo ""

# Step 1: Base inference (~2-4 hrs on A100)
echo "=== Step 1: Steerling-8B base inference (400 cases) ==="
echo "Started: $(date)"
$PYTHON 01_run_steerling_inference.py
echo "Finished: $(date)"
echo ""

# Step 2: Demographic variation (~12-20 hrs on A100)
echo "=== Step 2: Demographic variation (6,800 inferences) ==="
echo "Started: $(date)"
$PYTHON 02_demographic_variation.py
echo "Finished: $(date)"
echo ""

# Step 3: Concept analysis (~10 min, CPU)
echo "=== Step 3: Concept analysis ==="
echo "Started: $(date)"
$PYTHON 03_analyze_concepts.py
echo "Finished: $(date)"
echo ""

# Step 4: Concept steering (~8-12 hrs on A100)
echo "=== Step 4: Concept steering experiments ==="
echo "Started: $(date)"
$PYTHON 04_concept_steering.py
echo "Finished: $(date)"
echo ""

# Step 5: Generate outputs (~2 min, CPU)
echo "=== Step 5: Generate tables and figures ==="
echo "Started: $(date)"
$PYTHON 05_generate_outputs.py
echo "Finished: $(date)"
echo ""

echo "=== Pipeline complete ==="
echo "Finished: $(date)"
echo ""
echo "Output files:"
ls -la output/
echo ""
echo "Tables:"
ls -la tables/ 2>/dev/null || echo "(none)"
echo ""
echo "Figures:"
ls -la figures/ 2>/dev/null || echo "(none)"
echo ""
echo "To download results: tar czf results.tar.gz output/ tables/ figures/ pipeline.log"
