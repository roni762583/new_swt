#!/usr/bin/env python3
"""
Backtest Pretrained Model on Validation Data.

Simulates actual trading to measure P&L, win rate, expectancy.
Uses validation time period (last 30% of data chronologically).
"""

import os
import sys
import duckdb
import numpy as np
import torch
import logging
from pathlib import Path
from collections import defaultdict
from ppo_agent import PolicyNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 26 ML features
ML_FEATURES = [
    'log_return_1m', 'log_return_5m', 'log_return_60m', 'efficiency_ratio_h1',
    'momentum_strength_10_zsarctan_w20',
    'atr_14', 'atr_14_zsarctan_w20', 'vol_ratio_deviation', 'realized_vol_60_zsarctan_w20',
    'h1_swing_range_position', 'swing_point_range',
    'high_swing_slope_h1', 'low_swing_slope_h1', 'h1_trend_slope_zsarctan',
    'h1_swing_range_position_zsarctan_w20', 'swing_point_range_zsarctan_w20',
    'high_swing_slope_h1_zsarctan', 'low_swing_slope_h1_zsarctan',
    'high_swing_slope_m1_zsarctan_w20', 'low_swing_slope_m1_zsarctan_w20',
    'combo_geometric',
    'bb_position',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]


class BacktestEngine:
    """Simple backtest engine for pretrained model."""

    def __init__(self, spread_pips: float = 4.0, pip_value: float = 0.01):
        """Initialize backtest engine.

        Args:
            spread_pips: Trading spread in pips (4 for GBPJPY)
            pip_value: Pip value (0.01 for JPY pairs)
        """
        self.spread_pips = spread_pips
        self.pip_value = pip_value
        self.pip_multiplier = 100  # To convert price to pips

        # Trading state
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.entry_bar = 0

        # Performance tracking
        self.trades = []
        self.balance = 10000.0
        self.equity = 10000.0
        self.peak_equity = 10000.0

    def reset(self):
        """Reset backtest state."""
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.trades = []
        self.balance = 10000.0
        self.equity = 10000.0
        self.peak_equity = 10000.0

    def get_position_state(self, current_price: float) -> tuple:
        """Calculate position state features.

        Returns:
            (position_side, position_pips, bars_since_entry,
             pips_from_peak, max_drawdown_pips, accumulated_dd)
        """
        if self.position == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Position pips (unrealized P&L in pips)
        if self.position == 1:  # Long
            position_pips = (current_price - self.entry_price) * self.pip_multiplier - self.spread_pips
        else:  # Short
            position_pips = (self.entry_price - current_price) * self.pip_multiplier - self.spread_pips

        # Bars since entry
        bars_since_entry = float(len(self.trades))  # Approximate

        # Drawdown metrics (simplified)
        pips_from_peak = max(0, self.peak_equity - self.equity) * self.pip_multiplier
        max_drawdown_pips = pips_from_peak
        accumulated_dd = pips_from_peak

        return (
            float(self.position),
            position_pips,
            bars_since_entry,
            pips_from_peak,
            max_drawdown_pips,
            accumulated_dd
        )

    def execute_action(self, action: int, current_price: float, bar_idx: int) -> dict:
        """Execute trading action.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
            current_price: Current market price
            bar_idx: Current bar index

        Returns:
            dict with trade info (if any)
        """
        trade_info = None

        # HOLD (0)
        if action == 0:
            pass  # Do nothing

        # BUY (1) - Only if flat
        elif action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.entry_bar = bar_idx

        # SELL (2) - Only if flat
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
            self.entry_bar = bar_idx

        # CLOSE (3) - Only if in position
        elif action == 3 and self.position != 0:
            # Calculate P&L
            if self.position == 1:  # Closing long
                pips = (current_price - self.entry_price) * self.pip_multiplier - self.spread_pips
            else:  # Closing short
                pips = (self.entry_price - current_price) * self.pip_multiplier - self.spread_pips

            # Update balance
            self.balance += pips
            self.equity = self.balance
            self.peak_equity = max(self.peak_equity, self.equity)

            # Record trade
            trade_info = {
                'entry_bar': self.entry_bar,
                'exit_bar': bar_idx,
                'direction': 'LONG' if self.position == 1 else 'SHORT',
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'pips': pips,
                'balance': self.balance
            }
            self.trades.append(trade_info)

            # Close position
            self.position = 0
            self.entry_price = 0.0
            self.entry_bar = 0

        return trade_info


def load_validation_data(db_path: str, train_ratio: float = 0.7):
    """Load validation data with prices.

    Args:
        db_path: Path to master.duckdb
        train_ratio: Train/val split ratio

    Returns:
        features (26 ML), prices, bar_indices
    """
    logger.info(f"Loading validation data from {db_path}")
    conn = duckdb.connect(db_path, read_only=True)

    # Build query
    feature_cols = ', '.join(ML_FEATURES)

    query = f"""
        SELECT
            bar_index,
            close,
            {feature_cols}
        FROM master
        WHERE pretrain_action IS NOT NULL
        ORDER BY bar_index
    """

    df = conn.execute(query).df()
    conn.close()

    # Extract data
    bar_indices = df['bar_index'].values
    prices = df['close'].values
    features = df[ML_FEATURES].values.astype(np.float32)

    # Handle NaN
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values, filling with 0")
        features = np.nan_to_num(features, nan=0.0)

    # Split to get validation set (30%)
    split_idx = int(len(features) * train_ratio)
    features_val = features[split_idx:]
    prices_val = prices[split_idx:]
    bar_indices_val = bar_indices[split_idx:]

    logger.info(f"Validation set: {len(features_val):,} bars")

    return features_val, prices_val, bar_indices_val


def backtest_model(checkpoint_path: str, db_path: str = "master.duckdb"):
    """Run backtest on pretrained model.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        db_path: Path to database
    """
    logger.info("="*70)
    logger.info("BACKTEST - PRETRAINED ZIGZAG MODEL")
    logger.info("="*70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PolicyNetwork(input_dim=32, action_dim=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded")

    # Load validation data
    features, prices, bar_indices = load_validation_data(db_path)

    # Initialize backtest engine
    engine = BacktestEngine(spread_pips=4.0, pip_value=0.01)

    logger.info("\nRunning backtest...")
    logger.info(f"Bars to process: {len(features):,}")
    logger.info(f"Initial balance: {engine.balance:.2f}")

    # Run backtest
    action_counts = defaultdict(int)

    with torch.no_grad():
        for i, (feat, price, bar_idx) in enumerate(zip(features, prices, bar_indices)):
            # Get position state
            pos_state = engine.get_position_state(price)

            # Combine features: 26 ML + 6 position
            full_features = np.concatenate([feat, pos_state])
            state_tensor = torch.FloatTensor(full_features).unsqueeze(0).to(device)

            # Get action from model
            logits, _ = model(state_tensor)
            action = logits.argmax(dim=1).item()

            # Execute action
            trade_info = engine.execute_action(action, price, bar_idx)

            # Track actions
            action_counts[action] += 1

            # Log trades
            if trade_info:
                logger.info(
                    f"Trade #{len(engine.trades)}: {trade_info['direction']} "
                    f"{trade_info['pips']:+.1f} pips | Balance: {trade_info['balance']:.2f}"
                )

            # Progress
            if (i + 1) % 50000 == 0:
                logger.info(f"Processed {i+1:,} / {len(features):,} bars...")

    # Calculate metrics
    logger.info("\n" + "="*70)
    logger.info("BACKTEST RESULTS")
    logger.info("="*70)

    total_trades = len(engine.trades)
    if total_trades == 0:
        logger.info("No trades executed!")
        return

    # Extract trade P&Ls
    trade_pips = [t['pips'] for t in engine.trades]
    winners = [p for p in trade_pips if p > 0]
    losers = [p for p in trade_pips if p <= 0]

    win_count = len(winners)
    loss_count = len(losers)
    win_rate = win_count / total_trades * 100

    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    avg_trade = np.mean(trade_pips)

    profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf')
    expectancy = avg_trade

    total_pips = sum(trade_pips)
    net_profit = engine.balance - 10000.0

    logger.info(f"\nTrading Performance:")
    logger.info(f"  Total Trades: {total_trades}")
    logger.info(f"  Winners: {win_count} ({win_rate:.1f}%)")
    logger.info(f"  Losers: {loss_count} ({100-win_rate:.1f}%)")
    logger.info(f"  Average Win: {avg_win:+.2f} pips")
    logger.info(f"  Average Loss: {avg_loss:+.2f} pips")
    logger.info(f"  Average Trade: {avg_trade:+.2f} pips (Expectancy)")
    logger.info(f"  Profit Factor: {profit_factor:.2f}")
    logger.info(f"  Total P&L: {total_pips:+.2f} pips")
    logger.info(f"  Net Profit: ${net_profit:+.2f}")
    logger.info(f"  Final Balance: ${engine.balance:.2f}")

    logger.info(f"\nAction Distribution:")
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
    total_actions = sum(action_counts.values())
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        pct = count / total_actions * 100
        logger.info(f"  {action_names[action]:5}: {count:,} ({pct:.1f}%)")

    # Trade duration analysis
    durations = [t['exit_bar'] - t['entry_bar'] for t in engine.trades]
    logger.info(f"\nTrade Duration:")
    logger.info(f"  Min: {min(durations)} bars")
    logger.info(f"  Max: {max(durations)} bars")
    logger.info(f"  Average: {np.mean(durations):.1f} bars")

    logger.info("\n" + "="*70)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*70)

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'total_pips': total_pips,
        'net_profit': net_profit,
        'final_balance': engine.balance
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Backtest pretrained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pretrain_zigzag_best.pth',
                        help='Path to pretrained checkpoint')
    parser.add_argument('--db', type=str, default='master.duckdb',
                        help='Path to database')

    args = parser.parse_args()

    # Check files exist
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path(args.db).exists():
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Run backtest
    backtest_model(args.checkpoint, args.db)


if __name__ == "__main__":
    main()
