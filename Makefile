.PHONY: all setup lint test clean pipeline step1 step2 step3 step4 step5

PYTHON ?= python3

# Run the full pipeline end-to-end
all: pipeline

# Install dependencies
setup:
	$(PYTHON) -m pip install -e ".[dev]"

# Run linting
lint:
	ruff check src/ tests/ *.py
	mypy src/ --ignore-missing-imports

# Run tests
test:
	pytest tests/ -v --tb=short

# Run the full 5-step pipeline
pipeline: step1 step2 step3 step4 step5

# Step 1: Base Steerling-8B inference (~2-4 hrs, GPU required)
step1:
	$(PYTHON) 01_run_steerling_inference.py

# Step 2: Demographic variation inference (~12-20 hrs, GPU required)
step2:
	$(PYTHON) 02_demographic_variation.py

# Step 3: Concept analysis (~10 min, CPU only)
step3:
	$(PYTHON) 03_analyze_concepts.py

# Step 4: Concept steering experiments (~8-12 hrs, GPU required)
step4:
	$(PYTHON) 04_concept_steering.py

# Step 5: Generate tables and figures (~2 min, CPU only)
step5:
	$(PYTHON) 05_generate_outputs.py

# Remove generated outputs (keeps data)
clean:
	rm -f output/*.json output/*.csv
	rm -f tables/*.csv
	rm -f figures/*.pdf figures/*.png
