#!/bin/bash
set -e

echo "🔍 SWT Validation Container Starting..."
echo "📁 Monitoring checkpoints directory for new best models..."

# Create marker file to track last validated checkpoint
mkdir -p /workspace/validation_state
touch /workspace/validation_state/last_validated.txt

while true; do
    # ONLY look for NEW BEST checkpoints
    BEST_CHECKPOINT=$(find /workspace/checkpoints -name '*best*.pth' -newer /workspace/validation_state/last_validated.txt 2>/dev/null | head -1)

    if [ ! -z "${BEST_CHECKPOINT}" ]; then
        echo "🏆 Found NEW BEST checkpoint: ${BEST_CHECKPOINT}"
        echo "🧪 Running validation on best performer..."

        python swt_validation/validate_with_precomputed_wst.py \
            --checkpoint "${BEST_CHECKPOINT}" \
            --wst-file precomputed_wst/GBPJPY_WST_3.5years_streaming.h5 \
            --csv-file data/GBPJPY_M1_3.5years_20250912.csv \
            --num-simulations 100 || true

        # Update last validated marker
        touch /workspace/validation_state/last_validated.txt
        echo "✅ Validation complete for BEST checkpoint: ${BEST_CHECKPOINT}"
    else
        echo "⏳ Waiting for new best checkpoint... (checking every 30s)"
    fi

    sleep 30
done