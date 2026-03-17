"""Modal cloud execution for the 4-arm interpretability pipeline.

Runs Arms 2-4 on A100 GPUs via Modal (Arm 1 / Steerling already complete):
  Arm 2: SAE feature isolation + steering
  Arm 3: Logit lens + activation patching
  Arm 4: Probing classifiers + TSV steering

Usage:
    cd packaging/concept_attribution_triage

    # Deploy (builds image, one-time or after code changes):
    .venv/bin/python -m modal deploy modal_gemma_pipeline.py

    # Launch pipeline (detached — safe to close terminal):
    .venv/bin/python launch_pipeline.py

    # Check status:
    .venv/bin/python check_status.py

    # Download results:
    .venv/bin/python -m modal run modal_gemma_pipeline.py::download
"""

import modal

app = modal.App("concept-triage-gemma")

# Persistent volume for results
volume = modal.Volume.from_name("concept-triage-results", create_if_missing=True)

# Model cache volume (avoid re-downloading model each run)
model_cache = modal.Volume.from_name("gemma2-model-cache", create_if_missing=True)

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "huggingface-hub>=0.26.0",
        "safetensors",
        "pydantic",
    )
    .add_local_dir(
        ".",
        "/app",
        ignore=[".venv", "__pycache__", "*.pyc", ".git",
                "output/*.pt", "output/*.npy"],
        copy=True,
    )
)


def _commit_to_volume(vol):
    """Copy output files to volume and commit."""
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
    gpu="A100",
    timeout=86400,
    volumes={"/results": volume, "/model-cache": model_cache},
    memory=65536,
)
def run_gemma_pipeline():
    """Run the 4-step interpretability pipeline on A100."""
    import subprocess
    import sys
    import os
    import json
    import time
    import traceback

    os.chdir("/app")

    # Set HuggingFace cache to persistent volume
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    # Verify GPU
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"VRAM: {vram / 1e9:.1f} GB")

    os.makedirs("output", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Copy existing Steerling results from volume if available
    for fname in ["steerling_base_results.json", "causal_correction_results.json",
                   "tp_correction_results.json"]:
        src = f"/results/output/{fname}"
        dst = f"output/{fname}"
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
            print(f"  Restored {fname} from volume")

    def update_status(step, state, msg=""):
        status = {"step": step, "state": state, "message": msg, "timestamp": time.time()}
        with open("/results/gemma_pipeline_status.json", "w") as f:
            json.dump(status, f)
        volume.commit()

    steps = [
        ("20_gemma_base_inference.py", "Step 1: Qwen 2.5 7B base inference (400 cases + hidden states)"),
        ("21_sae_steering.py", "Step 2: SAE feature extraction + SAE-targeted steering"),
        ("22_logit_lens.py", "Step 3: Logit lens + activation patching"),
        ("23_probing_tsv.py", "Step 4: Probing classifiers + TSV steering"),
    ]

    pipeline_start = time.time()
    step_results = []

    for i, (script, desc) in enumerate(steps, 1):
        print(f"\n{'='*60}")
        print(f"{desc}")
        print(f"{'='*60}")
        step_start = time.time()
        update_status(i, "running", desc)

        try:
            result = subprocess.run(
                [sys.executable, "-u", script],
                capture_output=True,
                text=True,
            )
            # Always print output
            if result.stdout:
                print(result.stdout[-5000:])
            if result.stderr:
                print("STDERR:", result.stderr[-3000:])

            step_elapsed = time.time() - step_start

            if result.returncode != 0:
                print(f"ERROR: {script} failed (rc={result.returncode}, {step_elapsed/3600:.2f}h)")
                update_status(i, "failed", f"{desc} failed after {step_elapsed/3600:.2f}h")
                step_results.append({"step": i, "script": script, "status": "failed"})
            else:
                print(f"OK: {script} ({step_elapsed/3600:.2f}h)")
                step_results.append({"step": i, "script": script, "status": "ok",
                                     "elapsed_h": step_elapsed / 3600})
        except Exception as e:
            step_elapsed = time.time() - step_start
            print(f"EXCEPTION in {script}: {e}")
            traceback.print_exc()
            update_status(i, "exception", f"{desc}: {e}")
            step_results.append({"step": i, "script": script, "status": "exception", "error": str(e)})

        # Always commit after each step
        print(f"Committing step {i} results to volume...")
        _commit_to_volume(volume)

    total_elapsed = time.time() - pipeline_start
    update_status(4, "completed", f"All steps completed in {total_elapsed/3600:.1f}h")

    print(f"\nPipeline completed in {total_elapsed/3600:.1f} hours")

    summary = {
        "status": "completed",
        "elapsed_hours": total_elapsed / 3600,
        "steps": step_results,
    }
    for dirname in ["output", "tables", "figures"]:
        if os.path.exists(dirname):
            summary[dirname] = [f for f in os.listdir(dirname) if not f.endswith('.pt')]

    # Save summary to volume
    with open("/results/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()

    return summary


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/results": volume},
)
def list_results():
    """List all results in the volume."""
    import os
    import json

    volume.reload()
    results = {}

    status_path = "/results/gemma_pipeline_status.json"
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
def status():
    """Check pipeline status and list results."""
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
            if remote_path.endswith(('.pt', '.npy')) and dirname == "output":
                print(f"  Skipping large file {remote_path} (use download_all for binaries)")
                continue
            local_path = os.path.join(local_base, remote_path)
            # Skip directories
            if entry.stat().st_size == 0 and not remote_path.endswith(('.json', '.csv', '.pdf', '.png')):
                print(f"  Skipping {remote_path} (likely a directory)")
                continue
            try:
                print(f"  Downloading {remote_path}...")
                with open(local_path, "wb") as f:
                    for chunk in volume.read_file(remote_path):
                        f.write(chunk)
            except (IsADirectoryError, OSError) as e:
                print(f"  Skipping {remote_path}: {e}")
                continue

    print("\nAll results downloaded to local directories.")


@app.local_entrypoint()
def download_all():
    """Download ALL results including large binary files."""
    import os

    local_base = os.path.dirname(os.path.abspath(__file__))

    for dirname in ["output", "tables", "figures"]:
        local_dir = os.path.join(local_base, dirname)
        os.makedirs(local_dir, exist_ok=True)

        for entry in volume.listdir(dirname):
            remote_path = entry.path
            local_path = os.path.join(local_base, remote_path)
            size_mb = entry.stat().st_size / 1e6
            print(f"  Downloading {remote_path} ({size_mb:.1f} MB)...")
            with open(local_path, "wb") as f:
                for chunk in volume.read_file(remote_path):
                    f.write(chunk)

    print("\nAll results (including binaries) downloaded.")
