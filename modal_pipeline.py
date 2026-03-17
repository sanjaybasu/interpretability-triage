"""Modal cloud execution for concept attribution triage pipeline.

Runs the full 5-step pipeline on an A100 GPU via Modal.

Usage:
    cd packaging/concept_attribution_triage

    # Deploy the app (one-time setup):
    .venv/bin/python -m modal deploy modal_pipeline.py

    # Launch the pipeline (fire-and-forget, no persistent connection needed):
    .venv/bin/python -m modal run modal_pipeline.py::launch

    # Check status / list results in volume:
    .venv/bin/python -m modal run modal_pipeline.py::status

    # Download results to local directories:
    .venv/bin/python -m modal run modal_pipeline.py::download

Results are saved to a Modal Volume after each step for resilience.
"""

import modal

app = modal.App("concept-attribution-triage")

# Persistent volume for results
volume = modal.Volume.from_name("concept-triage-results", create_if_missing=True)

# Build the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "transformers>=4.38.0",
        "tiktoken",
        "huggingface-hub",
        "safetensors",
        "pydantic",
    )
    .pip_install("steerling>=0.1.0")
    .add_local_dir(".", "/app", ignore=[".venv", "__pycache__", "*.pyc", ".git"], copy=True)
)


def _commit_to_volume(vol):
    """Copy output/tables/figures to volume and commit."""
    import os
    import shutil

    for dirname in ["output", "tables", "figures"]:
        if os.path.exists(dirname):
            dest = f"/results/{dirname}"
            shutil.copytree(dirname, dest, dirs_exist_ok=True)
            files = os.listdir(dirname)
            print(f"  Saved {dirname}/: {len(files)} files -> volume")
    vol.commit()


@app.function(
    image=image,
    gpu="L4",
    timeout=86400,  # 24 hours
    volumes={"/results": volume},
)
def run_pipeline():
    """Run the full 5-step pipeline on an A100 GPU."""
    import subprocess
    import sys
    import os
    import json
    import shutil
    import time

    os.chdir("/app")

    # Verify GPU
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"VRAM: {vram / 1e9:.1f} GB")

    # Ensure output directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Write a status file to track progress
    def update_status(step, state, msg=""):
        status = {"step": step, "state": state, "message": msg, "timestamp": time.time()}
        with open("/results/pipeline_status.json", "w") as f:
            json.dump(status, f)
        volume.commit()

    steps = [
        ("01_run_steerling_inference.py", "Step 1: Base inference (400 cases)"),
        ("02_demographic_variation.py", "Step 2: Demographic variation (600 inferences)"),
        ("02b_extract_concept_weights.py", "Step 3: Extract concept weights"),
        ("03b_analyze_concept_weights.py", "Step 4: Concept analysis (FDR, PCA, L1)"),
        ("04b_corrected_steering.py", "Step 5: Concept steering experiments"),
        ("04c_concept_erasure.py", "Step 6: Concept erasure (LEACE/INLP)"),
        ("05_generate_outputs.py", "Step 7: Generate tables and figures"),
    ]

    pipeline_start = time.time()

    for i, (script, desc) in enumerate(steps, 1):
        print(f"\n{'='*60}")
        print(f"{desc}")
        print(f"{'='*60}")
        step_start = time.time()
        update_status(i, "running", desc)

        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True,
        )

        step_elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"ERROR: {script} failed with return code {result.returncode}")
            print(f"Step took {step_elapsed/3600:.2f} hours")
            update_status(i, "failed", f"{desc} failed after {step_elapsed/3600:.2f}h")
            # Save partial results to volume
            print("Saving partial results to volume...")
            _commit_to_volume(volume)
            return {"status": "failed", "failed_step": i, "script": script}

        print(f"Completed: {script} ({step_elapsed/3600:.2f} hours)")

        # Commit results to volume after EACH step
        print(f"Committing step {i} results to volume...")
        _commit_to_volume(volume)

    total_elapsed = time.time() - pipeline_start
    update_status(5, "completed", f"All steps completed in {total_elapsed/3600:.1f}h")

    print(f"\nPipeline completed in {total_elapsed/3600:.1f} hours")

    # Return a summary of what was produced
    summary = {"status": "completed", "elapsed_hours": total_elapsed / 3600}
    for dirname in ["output", "tables", "figures"]:
        if os.path.exists(dirname):
            summary[dirname] = os.listdir(dirname)
    return summary


@app.function(
    image=modal.Image.debian_slim(python_version="3.13"),
    volumes={"/results": volume},
)
def list_results():
    """List all results in the volume."""
    import os
    import json

    volume.reload()
    results = {}

    # Check status
    status_path = "/results/pipeline_status.json"
    if os.path.exists(status_path):
        with open(status_path) as f:
            results["status"] = json.load(f)

    for dirname in ["output", "tables", "figures"]:
        path = f"/results/{dirname}"
        if os.path.exists(path):
            files = []
            for fn in os.listdir(path):
                fpath = os.path.join(path, fn)
                size = os.path.getsize(fpath)
                files.append({"name": fn, "size_bytes": size})
            results[dirname] = sorted(files, key=lambda x: x["name"])
    return results


@app.local_entrypoint()
def launch():
    """Launch pipeline as a detached spawn (fire-and-forget)."""
    print("Spawning pipeline on Modal A100 (detached)...")
    fc = run_pipeline.spawn()
    print(f"Function call ID: {fc.object_id}")
    print("Pipeline is running on Modal. Your local machine can disconnect safely.")
    print("")
    print("To check status:  .venv/bin/python -m modal run modal_pipeline.py::status")
    print("To download:      .venv/bin/python -m modal run modal_pipeline.py::download")


@app.local_entrypoint()
def status():
    """Check pipeline status and list results."""
    import json
    results = list_results.remote()
    if "status" in results:
        s = results["status"]
        print(f"Pipeline status: step {s['step']}, state: {s['state']}")
        print(f"  {s.get('message', '')}")
    else:
        print("No status found (pipeline may not have started yet)")

    for dirname in ["output", "tables", "figures"]:
        if dirname in results:
            files = results[dirname]
            total = sum(f["size_bytes"] for f in files)
            print(f"\n{dirname}/ ({len(files)} files, {total/1024:.0f} KB):")
            for f in files:
                print(f"  {f['name']} ({f['size_bytes']/1024:.1f} KB)")


@app.local_entrypoint()
def download():
    """Download all results from Modal volume to local directories."""
    import os

    local_base = os.path.dirname(os.path.abspath(__file__))

    # First check what's available
    results = list_results.remote()
    if "status" in results:
        s = results["status"]
        print(f"Pipeline status: step {s['step']}, state: {s['state']}")

    for dirname in ["output", "tables", "figures"]:
        if dirname not in results or not results[dirname]:
            print(f"No files in {dirname}/")
            continue

        local_dir = os.path.join(local_base, dirname)
        os.makedirs(local_dir, exist_ok=True)

        for entry in volume.listdir(dirname):
            remote_path = entry.path
            local_path = os.path.join(local_base, remote_path)
            print(f"  Downloading {remote_path}...")
            with open(local_path, "wb") as f:
                for chunk in volume.read_file(remote_path):
                    f.write(chunk)

    print("\nAll results downloaded to local directories.")
