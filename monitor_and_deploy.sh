#!/bin/bash
# Monitor Modal GPU availability and deploy when slots open.
# Checks every 2 minutes. Deploys TSV v2 and SAE pertoken when GPU is free.

cd /Users/sanjaybasu/waymark-local/packaging/concept_attribution_triage

LOG=/tmp/modal_monitor.log
TSV_DEPLOYED=0
SAE_DEPLOYED=0

echo "$(date): Monitor started" >> $LOG

while true; do
    # Count active GPU tasks
    ACTIVE=$(modal app list 2>/dev/null | grep "ephemeral" | awk '{print $8}' | grep -v "^0$" | paste -sd+ - | bc 2>/dev/null || echo "0")

    echo "$(date): Active tasks: $ACTIVE, TSV deployed: $TSV_DEPLOYED, SAE deployed: $SAE_DEPLOYED" >> $LOG

    if [ "$ACTIVE" -le 2 ] 2>/dev/null; then
        if [ "$TSV_DEPLOYED" -eq 0 ]; then
            echo "$(date): Deploying TSV v2..." >> $LOG
            modal run --detach modal_tsv_steering_v2.py >> $LOG 2>&1
            TSV_DEPLOYED=1
            sleep 30
        fi

        if [ "$SAE_DEPLOYED" -eq 0 ]; then
            echo "$(date): Deploying SAE pertoken..." >> $LOG
            modal run --detach modal_sae_pertoken.py >> $LOG 2>&1
            SAE_DEPLOYED=1
        fi
    fi

    # Check if both are deployed
    if [ "$TSV_DEPLOYED" -eq 1 ] && [ "$SAE_DEPLOYED" -eq 1 ]; then
        echo "$(date): Both deployed. Monitoring for results..." >> $LOG

        # Check for results every 5 minutes
        while true; do
            sleep 300
            TSV_DONE=$(modal volume ls concept-triage-results /output/ 2>/dev/null | grep "tsv_patching_steering_v2" | wc -l)
            SAE_DONE=$(modal volume ls concept-triage-results /output/ 2>/dev/null | grep "sae_pertoken_steering_summary" | wc -l)
            echo "$(date): TSV result files: $TSV_DONE, SAE result files: $SAE_DONE" >> $LOG

            if [ "$TSV_DONE" -gt 0 ] && [ "$SAE_DONE" -gt 0 ]; then
                echo "$(date): Both experiments complete! Downloading results..." >> $LOG
                modal volume get concept-triage-results /output/tsv_patching_steering_v2.json output/tsv_patching_steering_v2.json 2>> $LOG
                modal volume get concept-triage-results /output/sae_pertoken_steering_summary.json output/sae_pertoken_steering_summary.json 2>> $LOG
                modal volume get concept-triage-results /output/sae_pertoken_steering_results.json output/sae_pertoken_steering_results.json 2>> $LOG
                echo "$(date): Results downloaded. Monitor complete." >> $LOG
                exit 0
            fi
        done
    fi

    sleep 120
done
