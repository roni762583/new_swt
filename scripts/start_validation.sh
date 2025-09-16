#!/bin/bash

echo "🔍 SWT Validation Container Starting..."
echo "📁 Monitoring checkpoints directory for new best models..."

# Create marker file to track last validated checkpoint
mkdir -p /workspace/validation_state 2>/dev/null || true

# Create marker with old timestamp if it doesn't exist
if [ ! -f /workspace/validation_state/last_validated.txt ]; then
    touch -t 202301010000 /workspace/validation_state/last_validated.txt 2>/dev/null || \
    touch /workspace/validation_state/last_validated.txt 2>/dev/null || \
    echo "Warning: Could not create marker file" >&2
fi

while true; do
    # ONLY look for NEW BEST checkpoints
    BEST_CHECKPOINT=$(find /workspace/checkpoints -name '*best*.pth' -newer /workspace/validation_state/last_validated.txt 2>/dev/null | head -1)

    if [ ! -z "${BEST_CHECKPOINT}" ]; then
        echo "🏆 Found NEW BEST checkpoint: ${BEST_CHECKPOINT}"
        echo "🧪 Running validation on best performer..."

        # Run validation and capture results
        VALIDATION_OUTPUT=$(python swt_validation/validate_with_precomputed_wst.py \
            --checkpoints "${BEST_CHECKPOINT}" \
            --wst-file precomputed_wst/GBPJPY_WST_CLEAN_2022-2025.h5 \
            --csv-file data/GBPJPY_M1_REAL_2022-2025.csv \
            --runs 100 2>&1) || true

        echo "${VALIDATION_OUTPUT}"

        # Extract key metrics from validation output
        SHARPE=$(echo "${VALIDATION_OUTPUT}" | grep "Sharpe Ratio:" | tail -1 | sed 's/.*Sharpe Ratio: //' | cut -d' ' -f1)
        RETURN=$(echo "${VALIDATION_OUTPUT}" | grep "Total Return:" | tail -1 | sed 's/.*Total Return: //' | cut -d'%' -f1)

        # ALWAYS preserve checkpoint and generate report for manual review
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        CHECKPOINT_NAME=$(basename "${BEST_CHECKPOINT}" .pth)
        PRESERVED_NAME="validated_${TIMESTAMP}_${CHECKPOINT_NAME}_sharpe${SHARPE}_return${RETURN}.pth"

        echo "💾 Preserving checkpoint as ${PRESERVED_NAME}"
        cp "${BEST_CHECKPOINT}" "/workspace/validation_results/${PRESERVED_NAME}"

        # ALWAYS generate PDF report with box-whisker plots
        echo "📊 Generating comprehensive PDF report with distribution plots..."

        # Save validation output to file for PDF generator
        echo "${VALIDATION_OUTPUT}" > "/workspace/validation_results/validation_${TIMESTAMP}.txt"

        # Run comprehensive validation with distribution plots
        python validation_with_plots.py \
            --checkpoint "${BEST_CHECKPOINT}" \
            --wst-file precomputed_wst/GBPJPY_WST_CLEAN_2022-2025.h5 \
            --csv-file data/GBPJPY_M1_REAL_2022-2025.csv \
            --runs 50 \
            --output-dir /workspace/validation_results || echo "Validation with plots failed"

        echo "📈 Results saved for manual review: ${PRESERVED_NAME}"

        # Update last validated marker
        touch /workspace/validation_state/last_validated.txt
        echo "✅ Validation complete for BEST checkpoint: ${BEST_CHECKPOINT}"
    else
        echo "⏳ Waiting for new best checkpoint... (checking every 30s)"
    fi

    sleep 30
done