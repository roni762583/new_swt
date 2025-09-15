#!/bin/bash
set -e

echo "🚀 SWT Training Container Starting..."
echo "📊 Target: 1,000,000 episodes"

# Check for resume state
mkdir -p /workspace/training_state

if [ -f /workspace/training_state/last_episode.txt ]; then
    LAST_EP=$(cat /workspace/training_state/last_episode.txt)
    echo "📂 Resuming from episode ${LAST_EP}"
    RESUME_FLAG="--resume-from ${LAST_EP}"
else
    echo "🆕 Starting fresh training run"
    RESUME_FLAG=""
fi

# Start training with resume capability
python training_main.py \
    --config config/training.yaml \
    --max-episodes 1000000 \
    --enable-validation \
    ${RESUME_FLAG} || true

echo "✅ Training loop ended. Container staying alive for restart..."
tail -f /dev/null