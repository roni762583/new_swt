#!/bin/bash
set -e

echo "💹 SWT Live Trading Container Starting..."
echo "🔌 OANDA Environment: ${OANDA_ENVIRONMENT:-practice}"
echo "📊 Instrument: ${INSTRUMENT:-GBP_JPY}"
echo "⏸️  WAITING FOR INSTRUCTION TO START TRADING"
echo ""
echo "To start live trading, run:"
echo "  docker exec swt_live_trading touch /workspace/live_state/START_TRADING"
echo ""
echo "To specify a checkpoint:"
echo '  docker exec swt_live_trading bash -c "echo /workspace/checkpoints/your_checkpoint.pth > /workspace/live_state/checkpoint_path.txt"'
echo ""

mkdir -p /workspace/live_state

while true; do
    if [ -f /workspace/live_state/START_TRADING ]; then
        echo "🚀 Trading signal received! Starting live trading..."

        # Check for specified checkpoint
        if [ -f /workspace/live_state/checkpoint_path.txt ]; then
            CHECKPOINT=$(cat /workspace/live_state/checkpoint_path.txt)
            echo "📦 Using specified checkpoint: ${CHECKPOINT}"
        else
            # Find best checkpoint
            CHECKPOINT=$(find /workspace/checkpoints -name '*best*.pth' | head -1)
            if [ -z "${CHECKPOINT}" ]; then
                CHECKPOINT=$(find /workspace/checkpoints -name '*.pth' | sort -V | tail -1)
            fi
            echo "📦 Using checkpoint: ${CHECKPOINT}"
        fi

        if [ ! -z "${CHECKPOINT}" ] && [ -f "${CHECKPOINT}" ]; then
            echo "🔥 Starting LIVE TRADING..."
            python live_trading_main.py \
                --checkpoint "${CHECKPOINT}" \
                --config config/live.yaml || true
            echo "⚠️  Trading stopped. Removing start signal..."
            rm -f /workspace/live_state/START_TRADING
        else
            echo "❌ No valid checkpoint found. Please train a model first."
        fi
    else
        echo -n "."
        sleep 10
    fi
done