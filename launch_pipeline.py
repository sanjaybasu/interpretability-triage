#!/usr/bin/env python3
"""Launch the deployed pipeline function (fire-and-forget).

This calls the already-deployed function via Function.from_name(),
avoiding ephemeral app heartbeat issues. The function runs on Modal
infrastructure independently of your local machine.

Prerequisites: modal deploy modal_gemma_pipeline.py
"""
import modal

fn = modal.Function.from_name("concept-triage-gemma", "run_gemma_pipeline")
fc = fn.spawn()
print(f"Pipeline spawned on Modal A100.")
print(f"Function call ID: {fc.object_id}")
print()
print("Check status:  .venv/bin/python check_status.py")
print("Download:      .venv/bin/python -m modal run modal_gemma_pipeline.py::download")
