#!/usr/bin/env python3
"""
Supervised Pretraining on ZigZag Labels.

Trains the custom PolicyNetwork (32→64→128→32 residual architecture)
on ZigZag pivot labels before PPO fine-tuning.

Label mapping (4-action space matching PPO):
- pretrain_action = 0 → HOLD
- pretrain_action = 1 → BUY (open long at swing lows)
- pretrain_action = 2 → SELL (open short at swing highs)
- pretrain_action = 3 → CLOSE (exit position)
"""

import os
import duckdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
from datetime import datetime
from ppo_agent import PolicyNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 26 ML features from db-state.txt (optimized set, zero high-correlation)
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

# Position features (will be zeros during pretraining)
POSITION_FEATURES = [
    'position_side', 'position_pips', 'bars_since_entry',
    'pips_from_peak', 'max_drawdown_pips', 'accumulated_dd'
]


class ZigZagDataset(Dataset):
    """Dataset for ZigZag supervised learning."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data_from_db(db_path: str, train_ratio: float = 0.7):
    """Load and split data from DuckDB.

    Args:
        db_path: Path to master.duckdb
        train_ratio: Fraction for training (0.7 = 70% train, 30% val)

    Returns:
        X_train, X_val, y_train, y_val
    """
    logger.info(f"Loading data from {db_path}")
    conn = duckdb.connect(db_path, read_only=True)

    # Get data statistics
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pretrain_action = 0 THEN 1 ELSE 0 END) as hold_count,
            SUM(CASE WHEN pretrain_action = 1 THEN 1 ELSE 0 END) as buy_count,
            SUM(CASE WHEN pretrain_action = 2 THEN 1 ELSE 0 END) as sell_count,
            SUM(CASE WHEN pretrain_action = 3 THEN 1 ELSE 0 END) as close_count
        FROM master
        WHERE pretrain_action IS NOT NULL
    """).fetchone()

    logger.info(f"Dataset stats:")
    logger.info(f"  Total: {stats[0]:,} rows")
    logger.info(f"  HOLD: {stats[1]:,} ({stats[1]/stats[0]*100:.2f}%)")
    logger.info(f"  BUY: {stats[2]:,} ({stats[2]/stats[0]*100:.2f}%)")
    logger.info(f"  SELL: {stats[3]:,} ({stats[3]/stats[0]*100:.2f}%)")
    logger.info(f"  CLOSE: {stats[4]:,} ({stats[4]/stats[0]*100:.2f}%)")

    # Build query for 26 ML features
    feature_cols = ', '.join(ML_FEATURES)

    query = f"""
        SELECT
            {feature_cols},
            pretrain_action
        FROM master
        WHERE pretrain_action IS NOT NULL
        ORDER BY bar_index
    """

    logger.info("Loading features...")
    df = conn.execute(query).df()
    conn.close()

    # Separate features and labels
    X = df[ML_FEATURES].values.astype(np.float32)

    # Labels are already correct: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
    # No remapping needed!
    y = df['pretrain_action'].values.astype(np.int64)

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")

    # Handle NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    # Add 6 position features (all zeros during pretraining)
    position_zeros = np.zeros((X.shape[0], 6), dtype=np.float32)
    X_full = np.concatenate([X, position_zeros], axis=1)

    logger.info(f"Final input shape with position features: {X_full.shape}")

    # Split train/validation
    split_idx = int(len(X_full) * train_ratio)
    X_train = X_full[:split_idx]
    X_val = X_full[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]

    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Val set: {len(X_val):,} samples")

    return X_train, X_val, y_train, y_val


def train_epoch(model, dataloader, criterion, optimizer, device, class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass - only use policy logits, ignore value
        logits, _ = model(features)

        # Weighted loss
        if class_weights is not None:
            weights = class_weights[labels].to(device)
            loss = (criterion(logits, labels) * weights).mean()
        else:
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Per-class metrics
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits, _ = model(features)
            loss = criterion(logits, labels).mean()

            total_loss += loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(4):
                class_mask = labels == i
                class_total[i] += class_mask.sum().item()
                class_correct[i] += (predicted[class_mask] == i).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    # Class accuracies
    class_acc = [100. * class_correct[i] / max(class_total[i], 1) for i in range(4)]

    return avg_loss, accuracy, class_acc


def main():
    # Configuration
    DB_PATH = "master.duckdb"
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    TRAIN_RATIO = 0.7
    BATCH_SIZE = 256
    EPOCHS = 1  # Single epoch test - see if it's learning before committing
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_EVERY_EPOCH = True  # Save checkpoint after each epoch for monitoring

    logger.info("="*60)
    logger.info("ZIGZAG SUPERVISED PRETRAINING")
    logger.info("="*60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Input features: 32 (26 ML + 6 position)")
    logger.info(f"Output classes: 4 (HOLD=0, BUY=1, SELL=2, CLOSE=3)")
    logger.info(f"Train ratio: {TRAIN_RATIO*100:.0f}%")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")

    # Load data
    X_train, X_val, y_train, y_val = load_data_from_db(DB_PATH, TRAIN_RATIO)

    # Compute class weights for imbalanced dataset (manual calculation)
    # All 4 classes should exist: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
    class_counts = np.array([np.sum(y_train == c) for c in range(4)])
    class_weights_np = len(y_train) / (4 * class_counts)
    class_weights = torch.FloatTensor(class_weights_np).to(DEVICE)

    logger.info(f"\nClass weights (for imbalanced data):")
    label_names = ['HOLD', 'BUY', 'SELL', 'CLOSE']
    for cls in range(4):
        logger.info(f"  {label_names[cls]:5} ({cls}): weight={class_weights_np[cls]:6.2f}, count={class_counts[cls]:,}")

    # Create datasets
    train_dataset = ZigZagDataset(X_train, y_train)
    val_dataset = ZigZagDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create model matching PPO architecture: 4 actions
    logger.info(f"\nModel architecture: 32 input features → 4 output classes")
    model = PolicyNetwork(input_dim=32, action_dim=4).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')  # Will apply weights manually
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    best_val_acc = 0
    best_epoch = 0

    logger.info("\n" + "="*60)
    logger.info("TRAINING START")
    logger.info("="*60)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, class_weights)
        val_loss, val_acc, class_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        logger.info(f"  Val Acc by class: HOLD={class_acc[0]:.1f}% BUY={class_acc[1]:.1f}% SELL={class_acc[2]:.1f}% CLOSE={class_acc[3]:.1f}%")
        logger.info(f"  LR: {current_lr:.6f}")

        # Save checkpoint every epoch for monitoring
        if SAVE_EVERY_EPOCH:
            epoch_checkpoint_path = CHECKPOINT_DIR / f"pretrain_zigzag_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_acc': class_acc,
                'architecture': '32→64→128→32 residual',
                'input_features': ML_FEATURES + POSITION_FEATURES,
                'label_mapping': {'HOLD': 0, 'BUY': 1, 'SELL': 2, 'CLOSE': 3},
                'num_actions': 4
            }, epoch_checkpoint_path)
            logger.info(f"  💾 Saved epoch checkpoint: {epoch_checkpoint_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            checkpoint_path = CHECKPOINT_DIR / "pretrain_zigzag_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_acc': class_acc,
                'architecture': '32→64→128→32 residual',
                'input_features': ML_FEATURES + POSITION_FEATURES,
                'label_mapping': {'HOLD': 0, 'BUY': 1, 'SELL': 2, 'CLOSE': 3},
                'num_actions': 4
            }, checkpoint_path)

            logger.info(f"  ✅ Saved best checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_path = CHECKPOINT_DIR / f"pretrain_zigzag_final_epoch{EPOCHS}.pth"
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'architecture': '32→64→128→32 residual',
        'input_features': ML_FEATURES + POSITION_FEATURES,
    }, final_path)

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    logger.info(f"Best checkpoint: {CHECKPOINT_DIR / 'pretrain_zigzag_best.pth'}")
    logger.info(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
