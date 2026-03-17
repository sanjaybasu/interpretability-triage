"""Check Modal volume for TSV/SAE steering results."""
import subprocess
import sys

def check_volume():
    result = subprocess.run(
        ["modal", "volume", "ls", "concept-triage-results", "/output/"],
        capture_output=True, text=True
    )
    files = result.stdout.strip().split("\n") if result.stdout.strip() else []
    tsv_found = any("tsv_patching_steering" in f for f in files)
    sae_found = any("sae_pertoken_steering" in f for f in files)
    print(f"TSV steering results: {'FOUND' if tsv_found else 'not yet'}")
    print(f"SAE pertoken results: {'FOUND' if sae_found else 'not yet'}")
    if tsv_found:
        subprocess.run([
            "modal", "volume", "get", "concept-triage-results",
            "output/tsv_patching_steering_results.json",
            "/Users/sanjaybasu/waymark-local/packaging/concept_attribution_triage/output/tsv_patching_steering_results.json"
        ])
        print("Downloaded TSV results.")
    if sae_found:
        for fname in ["sae_pertoken_steering_summary.json", "sae_pertoken_steering_results.json"]:
            subprocess.run([
                "modal", "volume", "get", "concept-triage-results",
                f"output/{fname}",
                f"/Users/sanjaybasu/waymark-local/packaging/concept_attribution_triage/output/{fname}"
            ])
        print("Downloaded SAE results.")
    result = subprocess.run(["modal", "app", "list"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "tsv-steeri" in line or "sae-pertok" in line:
            print(f"  {line.strip()}")

if __name__ == "__main__":
    check_volume()
