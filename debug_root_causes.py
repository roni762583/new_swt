#!/usr/bin/env python3
"""Debug root causes of training issues."""

import torch
import numpy as np
import sys
import json
sys.path.append('/workspace')

from micro.training.train_micro_muzero import MicroMuZeroTrainer, TrainingConfig
from micro.training.mcts_micro import MCTS
from micro.data.micro_dataloader import MicroDataLoader

def debug_policy_outputs():
    """Debug why model always outputs Hold action."""
    print("🔍 DEBUGGING POLICY OUTPUTS")
    print("=" * 50)

    # Load model from latest checkpoint
    config = TrainingConfig()
    trainer = MicroMuZeroTrainer(config)

    try:
        # Try to load latest checkpoint
        checkpoint_path = "/workspace/micro/checkpoints/latest.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        trainer.model.set_weights(checkpoint['model_state'])
        print(f"✅ Loaded checkpoint from episode {checkpoint['episode']}")
    except Exception as e:
        print(f"⚠️  Using random weights: {e}")

    # Get a sample observation
    window = trainer.data_loader.get_random_window(split='train')
    observation = torch.tensor(window['observation'], dtype=torch.float32).unsqueeze(0)

    print(f"📊 Observation shape: {observation.shape}")
    print(f"📊 Observation range: [{observation.min():.3f}, {observation.max():.3f}]")

    # Check if observation has NaN/Inf
    has_nan = torch.isnan(observation).any()
    has_inf = torch.isinf(observation).any()
    print(f"📊 Observation has NaN: {has_nan}, Inf: {has_inf}")

    # Test model inference
    trainer.model.eval()
    with torch.no_grad():
        try:
            hidden, policy_logits, value_probs = trainer.model.initial_inference(observation)

            print(f"🧠 Policy logits: {policy_logits.squeeze()}")
            print(f"🧠 Policy logits range: [{policy_logits.min():.3f}, {policy_logits.max():.3f}]")

            # Check for NaN in outputs
            policy_has_nan = torch.isnan(policy_logits).any()
            value_has_nan = torch.isnan(value_probs).any()
            print(f"🧠 Policy has NaN: {policy_has_nan}, Value has NaN: {value_has_nan}")

            # Apply softmax
            policy_probs = torch.softmax(policy_logits, dim=1)
            print(f"🎯 Policy probabilities: {policy_probs.squeeze()}")

            # Check if extremely biased towards action 0
            max_prob = policy_probs.max().item()
            argmax = policy_probs.argmax().item()
            print(f"🎯 Max probability: {max_prob:.4f} for action {argmax}")

            if max_prob > 0.99:
                print("⚠️  ISSUE: Policy is extremely biased towards one action!")

            # Check value output
            if hasattr(trainer.model.value, 'get_value'):
                value = trainer.model.value.get_value(value_probs)
                print(f"💰 Value estimate: {value.item():.3f}")

        except Exception as e:
            print(f"❌ Model inference failed: {e}")
            import traceback
            traceback.print_exc()

def debug_mcts_exploration():
    """Debug MCTS exploration parameters."""
    print("\n🎲 DEBUGGING MCTS EXPLORATION")
    print("=" * 50)

    config = TrainingConfig()
    trainer = MicroMuZeroTrainer(config)

    # Check MCTS parameters
    mcts = trainer.mcts
    print(f"🎲 Dirichlet alpha: {mcts.dirichlet_alpha}")
    print(f"🎲 Exploration fraction: {mcts.exploration_fraction}")
    print(f"🎲 Temperature: {config.temperature}")
    print(f"🎲 Number of simulations: {mcts.num_simulations}")

    # Test MCTS action selection
    window = trainer.data_loader.get_random_window(split='train')
    observation = torch.tensor(window['observation'], dtype=torch.float32).unsqueeze(0)

    try:
        result = mcts.run(observation, add_exploration_noise=True, temperature=config.temperature)

        print(f"🎯 MCTS action: {result['action']}")
        print(f"🎯 Action probabilities: {result['action_probs']}")
        print(f"🎯 Root value: {result['value']:.3f}")

        # Check if MCTS is exploring
        entropy = -np.sum(result['action_probs'] * np.log(result['action_probs'] + 1e-8))
        print(f"🌡️  Policy entropy: {entropy:.3f} (max={np.log(4):.3f})")

        if entropy < 0.1:
            print("⚠️  ISSUE: Very low entropy - not exploring!")

    except Exception as e:
        print(f"❌ MCTS failed: {e}")

def debug_training_losses():
    """Debug NaN losses in training."""
    print("\n💥 DEBUGGING TRAINING LOSSES")
    print("=" * 50)

    config = TrainingConfig()
    trainer = MicroMuZeroTrainer(config)

    # Try to get a batch and compute losses
    if len(trainer.buffer) >= config.batch_size:
        try:
            batch = trainer.buffer.sample(config.batch_size)
            print(f"📦 Sampled batch of {len(batch)} experiences")

            # Check batch for NaN
            for i, exp in enumerate(batch[:3]):  # Check first 3
                has_nan_obs = np.isnan(exp.observation).any()
                has_nan_reward = np.isnan(exp.reward)
                has_nan_value = np.isnan(exp.value)

                if has_nan_obs or has_nan_reward or has_nan_value:
                    print(f"⚠️  Experience {i} has NaN: obs={has_nan_obs}, reward={has_nan_reward}, value={has_nan_value}")

            # Try training step
            trainer.model.train()
            losses = trainer.train_batch(batch)

            print(f"📈 Losses: {losses}")

        except Exception as e:
            print(f"❌ Training batch failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("📦 Buffer too small for batch sampling")

def debug_validation_issues():
    """Debug NaN values in validation."""
    print("\n🔬 DEBUGGING VALIDATION ISSUES")
    print("=" * 50)

    # Check recent validation file
    validation_files = [
        "/workspace/micro/validation_results/validation_micro_checkpoint_ep000750.pth_20250917_153249.json"
    ]

    for file_path in validation_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            print(f"📄 File: {file_path.split('/')[-1]}")
            print(f"📊 Mean reward: {data.get('mean_reward')}")
            print(f"📊 Mean value: {data.get('mean_value')}")
            print(f"📊 Actions: {data.get('action_distribution')}")

            if str(data.get('mean_value')).lower() == 'nan':
                print("⚠️  ISSUE: mean_value is NaN in validation")

        except Exception as e:
            print(f"❌ Failed to read {file_path}: {e}")

def main():
    """Run all debugging functions."""
    print("🐛 ROOT CAUSE ANALYSIS")
    print("=" * 60)

    debug_policy_outputs()
    debug_mcts_exploration()
    debug_training_losses()
    debug_validation_issues()

    print("\n" + "=" * 60)
    print("🔍 ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()