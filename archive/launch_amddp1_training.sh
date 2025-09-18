#!/bin/bash
# Launch AMDDP1 training with proper setup

echo "============================================"
echo "LAUNCHING MICRO MUZERO WITH AMDDP1 REWARDS"
echo "============================================"
echo ""
echo "Configuration:"
echo "  - AMDDP1 reward system with retroactive assignment"
echo "  - 4 parallel workers for session collection"
echo "  - Pre-validated session queue (100 sessions)"
echo "  - Single position enforcement"
echo "  - Fixed LR: 0.002"
echo "  - Temperature: 4.0 -> 2.0 (1% decay per 1000 episodes)"
echo "  - MCTS simulations: 15"
echo "  - Session length: 360 minutes"
echo ""

# Clear old checkpoints if requested
if [ "$1" == "--clear" ]; then
    echo "Clearing old checkpoints and buffers..."
    rm -f /workspace/micro/checkpoints/*.pth
    echo "Cleared!"
    echo ""
fi

# Launch training
echo "Starting training..."
cd /workspace
python3 /workspace/micro/training/train_micro_muzero_amddp1.py

echo ""
echo "Training complete or interrupted"