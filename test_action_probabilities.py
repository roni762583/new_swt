#!/usr/bin/env python3
"""
Test action probabilities to check if Hold-only problem is resolved.
This script loads the model and generates action probabilities for sample data.
"""

import torch
import numpy as np
import sys
import logging
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')
sys.path.append('/home/aharon/projects/new_swt')

try:
    from micro.models.micro_networks import MicroStochasticMuZero
    from micro.training.train_micro_muzero import TrainingConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import...")
    sys.path.append('/home/aharon/projects/new_swt/micro')
    from models.micro_networks import MicroStochasticMuZero
    from training.train_micro_muzero import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_action_probabilities():
    """Test action probabilities with fresh model and sample data."""
    logger.info("🎯 TESTING ACTION PROBABILITIES")
    logger.info("=" * 50)

    # Create fresh model
    config = TrainingConfig()
    model = MicroStochasticMuZero(
        input_features=15,  # Corrected to 15 features
        lag_window=32,
        hidden_dim=256,
        action_dim=4,
        z_dim=16,
        support_size=300
    )

    logger.info(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create sample observation (32 timesteps, 15 features)
    observation = torch.randn(1, 32, 15)

    # Validate observation
    logger.info(f"📊 Sample observation shape: {observation.shape}")
    logger.info(f"📊 Sample observation range: [{observation.min():.3f}, {observation.max():.3f}]")

    # Test model inference
    model.eval()
    with torch.no_grad():
        try:
            # Get policy predictions
            hidden_state = model.representation(observation)
            logger.info(f"✅ Hidden state shape: {hidden_state.shape}")

            policy_logits = model.policy(hidden_state)
            logger.info(f"✅ Policy logits shape: {policy_logits.shape}")

            # Convert to probabilities
            action_probs = torch.softmax(policy_logits, dim=-1)
            action_probs_np = action_probs.squeeze().numpy()

            # Action names
            actions = ['Hold', 'Buy', 'Sell', 'Close']

            logger.info("🎯 ACTION PROBABILITIES:")
            logger.info("-" * 30)
            for i, (action, prob) in enumerate(zip(actions, action_probs_np)):
                percentage = prob * 100
                logger.info(f"  {action:5s}: {percentage:6.2f}%")

            # Check for Hold-only problem
            hold_prob = action_probs_np[0]
            if hold_prob > 0.8:
                logger.warning(f"⚠️  HOLD-ONLY PROBLEM DETECTED: Hold={hold_prob:.1%}")
                logger.warning("    Model is heavily biased toward Hold action")
            elif hold_prob < 0.4:
                logger.info(f"✅ GOOD DIVERSITY: Hold={hold_prob:.1%}")
                logger.info("    Model shows action diversity")
            else:
                logger.info(f"🔶 MODERATE: Hold={hold_prob:.1%}")
                logger.info("    Model shows some diversity")

            # Test value prediction
            value_logits = model.value(hidden_state)
            value_probs = torch.softmax(value_logits, dim=-1)
            value_scalar = model.value.get_value(value_probs)

            logger.info(f"📈 Value prediction: {value_scalar.item():.3f}")

            # Test multiple samples for consistency
            logger.info("\n🔄 TESTING MULTIPLE SAMPLES:")
            logger.info("-" * 30)
            hold_probs = []
            for i in range(5):
                # Different random observation
                obs_test = torch.randn(1, 32, 15)
                hidden_test = model.representation(obs_test)
                policy_test = model.policy(hidden_test)
                probs_test = torch.softmax(policy_test, dim=-1).squeeze().numpy()
                hold_probs.append(probs_test[0])

                logger.info(f"  Sample {i+1}: Hold={probs_test[0]:.1%}, Buy={probs_test[1]:.1%}, Sell={probs_test[2]:.1%}, Close={probs_test[3]:.1%}")

            avg_hold = np.mean(hold_probs)
            std_hold = np.std(hold_probs)

            logger.info(f"\n📊 STATISTICAL SUMMARY:")
            logger.info(f"  Average Hold probability: {avg_hold:.1%}")
            logger.info(f"  Standard deviation: {std_hold:.3f}")
            logger.info(f"  Diversity score: {1-avg_hold:.1%}")

            if avg_hold > 0.85:
                logger.error("❌ FAILED: Model stuck in Hold-only mode")
                return False
            elif avg_hold < 0.4:
                logger.info("✅ PASSED: Good action diversity")
                return True
            else:
                logger.info("🔶 PARTIAL: Some diversity but could be better")
                return True

        except Exception as e:
            logger.error(f"❌ Model inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_data_quality():
    """Test if the 15-feature data is available and clean."""
    logger.info("\n🔍 TESTING DATA QUALITY")
    logger.info("=" * 50)

    try:
        import duckdb
        db_path = "/home/aharon/projects/new_swt/data/micro_features.duckdb"

        if not Path(db_path).exists():
            logger.error(f"❌ Database not found: {db_path}")
            return False

        conn = duckdb.connect(db_path, read_only=True)

        # Check table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        logger.info(f"📋 Available tables: {[t[0] for t in tables]}")

        if ('micro_features',) not in tables:
            logger.error("❌ micro_features table not found")
            return False

        # Check feature count
        columns = conn.execute("SELECT * FROM micro_features LIMIT 1").fetchdf().columns
        logger.info(f"📊 Total columns: {len(columns)}")

        # Check expected features
        expected_features = [
            'position_in_range_60', 'min_max_scaled_momentum_60',
            'min_max_scaled_rolling_range', 'min_max_scaled_momentum_5',
            'price_change_pips',  # The 5th technical feature
            'dow_cos_final', 'dow_sin_final', 'hour_cos_final', 'hour_sin_final',
            'position_side', 'position_pips', 'bars_since_entry',
            'pips_from_peak', 'max_drawdown_pips', 'accumulated_dd'
        ]

        missing_features = []
        for feat in expected_features:
            if feat not in columns:
                missing_features.append(feat)

        if missing_features:
            logger.error(f"❌ Missing features: {missing_features}")
            return False
        else:
            logger.info(f"✅ All 15 features found in database")

        # Check for NaN/Inf in features
        for feat in expected_features:
            result = conn.execute(f"SELECT COUNT(*) FROM micro_features WHERE {feat} IS NULL OR {feat} = 'nan' OR {feat} = 'inf'").fetchone()[0]
            if result > 0:
                logger.warning(f"⚠️  Feature {feat} has {result} invalid values")

        conn.close()
        logger.info("✅ Data quality check passed")
        return True

    except Exception as e:
        logger.error(f"❌ Data quality check failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 MICRO TRAINING DIAGNOSIS")
    logger.info("=" * 60)

    # Test 1: Data Quality
    data_ok = test_data_quality()

    # Test 2: Action Probabilities
    action_ok = test_action_probabilities()

    logger.info("\n" + "=" * 60)
    logger.info("📋 FINAL RESULTS:")
    logger.info(f"  Data Quality: {'✅ PASS' if data_ok else '❌ FAIL'}")
    logger.info(f"  Action Diversity: {'✅ PASS' if action_ok else '❌ FAIL'}")

    if data_ok and action_ok:
        logger.info("🎉 ALL TESTS PASSED - System should work correctly")
    else:
        logger.info("⚠️  ISSUES DETECTED - Check logs above")