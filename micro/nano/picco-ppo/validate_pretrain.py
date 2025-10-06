#!/usr/bin/env python3
"""
Baseline Validation for Pretrained ZigZag Model.

Tests the supervised pretrained model on validation data BEFORE PPO training.
Provides baseline metrics: accuracy, precision, recall, F1, confusion matrix.
"""

import os
import sys
import duckdb
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from collections import defaultdict
from ppo_agent import PolicyNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 26 ML features (same as pretrain_zigzag.py)
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


def load_validation_data(db_path: str, train_ratio: float = 0.7):
    """Load validation data from DuckDB (same split as pretraining).

    Args:
        db_path: Path to master.duckdb
        train_ratio: Same ratio used in pretraining (0.7)

    Returns:
        X_val, y_val (numpy arrays)
    """
    logger.info(f"Loading validation data from {db_path}")
    conn = duckdb.connect(db_path, read_only=True)

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

    df = conn.execute(query).df()
    conn.close()

    # Separate features and labels
    X = df[ML_FEATURES].values.astype(np.float32)
    y = df['pretrain_action'].values.astype(np.int64)

    # Handle NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    # Add 6 position features (all zeros for validation, matching pretraining)
    position_zeros = np.zeros((X.shape[0], 6), dtype=np.float32)
    X_full = np.concatenate([X, position_zeros], axis=1)

    # Split to get validation set (30%)
    split_idx = int(len(X_full) * train_ratio)
    X_val = X_full[split_idx:]
    y_val = y[split_idx:]

    logger.info(f"Validation set: {len(X_val):,} samples")

    return X_val, y_val


def compute_metrics(y_true, y_pred, class_names=['HOLD', 'BUY', 'SELL', 'CLOSE']):
    """Compute comprehensive classification metrics (without sklearn).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class

    Returns:
        dict with metrics
    """
    n_classes = 4

    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    # Per-class metrics
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    support = np.zeros(n_classes, dtype=np.int64)

    for i in range(n_classes):
        # Support (number of true instances)
        support[i] = cm[i, :].sum()

        # True positives
        tp = cm[i, i]

        # False positives (predicted as i but not actually i)
        fp = cm[:, i].sum() - tp

        # False negatives (actually i but not predicted as i)
        fn = cm[i, :].sum() - tp

        # Precision: TP / (TP + FP)
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0.0

        # Recall: TP / (TP + FN)
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        else:
            recall[i] = 0.0

        # F1: 2 * (precision * recall) / (precision + recall)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0.0

    # Build classification report string
    report_lines = []
    report_lines.append(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    report_lines.append("-" * 62)

    for i, name in enumerate(class_names):
        report_lines.append(
            f"{name:<12} "
            f"{precision[i]:>10.4f} "
            f"{recall[i]:>10.4f} "
            f"{f1[i]:>10.4f} "
            f"{support[i]:>10}"
        )

    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    report_lines.append("-" * 62)
    report_lines.append(
        f"{'Macro Avg':<12} "
        f"{macro_precision:>10.4f} "
        f"{macro_recall:>10.4f} "
        f"{macro_f1:>10.4f} "
        f"{support.sum():>10}"
    )

    # Weighted averages
    weighted_precision = (precision * support).sum() / support.sum()
    weighted_recall = (recall * support).sum() / support.sum()
    weighted_f1 = (f1 * support).sum() / support.sum()

    report_lines.append(
        f"{'Weighted Avg':<12} "
        f"{weighted_precision:>10.4f} "
        f"{weighted_recall:>10.4f} "
        f"{weighted_f1:>10.4f} "
        f"{support.sum():>10}"
    )

    report = "\n".join(report_lines)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'classification_report': report
    }


def validate_model(checkpoint_path: str, db_path: str = "master.duckdb"):
    """Run baseline validation on pretrained model.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        db_path: Path to database
    """
    logger.info("="*70)
    logger.info("BASELINE VALIDATION - PRETRAINED ZIGZAG MODEL")
    logger.info("="*70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Log checkpoint info
    logger.info(f"Checkpoint info:")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  Val Accuracy (from training): {checkpoint.get('val_acc', 'N/A'):.2f}%")
    logger.info(f"  Val Loss (from training): {checkpoint.get('val_loss', 'N/A'):.4f}")
    logger.info(f"  Architecture: {checkpoint.get('architecture', 'N/A')}")

    # Create model
    model = PolicyNetwork(input_dim=32, action_dim=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully")

    # Load validation data
    X_val, y_val = load_validation_data(db_path)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_val).to(device)

    logger.info("\nRunning inference on validation set...")

    # Batch inference (to handle large dataset)
    batch_size = 1024
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            logits, _ = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)

    predictions = np.array(predictions)
    logger.info(f"Inference complete: {len(predictions):,} predictions")

    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics = compute_metrics(y_val, predictions)

    # Display results
    logger.info("\n" + "="*70)
    logger.info("VALIDATION RESULTS")
    logger.info("="*70)

    logger.info(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")

    logger.info("\nPer-Class Metrics:")
    class_names = ['HOLD', 'BUY', 'SELL', 'CLOSE']
    logger.info(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    logger.info("-" * 60)

    for i, name in enumerate(class_names):
        logger.info(
            f"{name:<8} "
            f"{metrics['precision'][i]:>10.4f} "
            f"{metrics['recall'][i]:>10.4f} "
            f"{metrics['f1'][i]:>10.4f} "
            f"{metrics['support'][i]:>10,}"
        )

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info(f"{'':>8} " + "".join([f"{name:>10}" for name in class_names]))
    logger.info("-" * 60)

    cm = metrics['confusion_matrix']
    for i, name in enumerate(class_names):
        row_str = f"{name:<8} " + "".join([f"{cm[i,j]:>10,}" for j in range(4)])
        logger.info(row_str)

    # Detailed classification report
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*70)
    logger.info(metrics['classification_report'])

    # Action distribution analysis
    logger.info("\n" + "="*70)
    logger.info("PREDICTION DISTRIBUTION ANALYSIS")
    logger.info("="*70)

    true_counts = [(y_val == i).sum() for i in range(4)]
    pred_counts = [(predictions == i).sum() for i in range(4)]

    logger.info(f"{'Class':<8} {'True Count':>12} {'True %':>10} {'Pred Count':>12} {'Pred %':>10}")
    logger.info("-" * 70)

    for i, name in enumerate(class_names):
        logger.info(
            f"{name:<8} "
            f"{true_counts[i]:>12,} "
            f"{true_counts[i]/len(y_val)*100:>9.2f}% "
            f"{pred_counts[i]:>12,} "
            f"{pred_counts[i]/len(predictions)*100:>9.2f}%"
        )

    logger.info("\n" + "="*70)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*70)

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate pretrained ZigZag model')
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

    # Run validation
    validate_model(args.checkpoint, args.db)


if __name__ == "__main__":
    main()
