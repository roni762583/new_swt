#!/usr/bin/env python3
"""
Minimal test of checkpoint saving logic
"""
import sys
import os
import torch
import pickle
from datetime import datetime

sys.path.append('/app')

print('='*60)
print('MINIMAL CHECKPOINT SAVE TEST')
print('='*60)

# Import necessary components
from ppo_agent import PPOLearningPolicy
from checkpoint_manager import CheckpointManager

# Initialize policy
print("\n1. Initializing policy...")
policy = PPOLearningPolicy(state_dim=17)

# Load from parashat_vayelech.pth
checkpoint_path = "/app/checkpoints/parashat_vayelech.pth"
if os.path.exists(checkpoint_path):
    print(f"2. Loading weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'policy_state_dict' in ckpt:
        policy.agent.policy.load_state_dict(ckpt['policy_state_dict'])
        print("   ✅ Model weights loaded")

        # Count parameters
        total_params = sum(p.numel() for p in policy.agent.policy.parameters())
        print(f"   Total parameters: {total_params:,}")
else:
    print("   ❌ Checkpoint not found")

# Initialize checkpoint manager
print("\n3. Initializing checkpoint manager...")
ckpt_manager = CheckpointManager("/app/checkpoints")

# Save immediate checkpoint with ACTUAL weights
print("\n4. Saving checkpoint with actual model weights...")

state = {
    'policy_state': policy.agent.policy.state_dict(),
    'optimizer_state': policy.agent.optimizer.state_dict(),
    'total_steps': 1000,
    'update_count': 50
}

metrics = {
    'total_trades': 100,
    'avg_pips': 0.36,
    'win_rate': 66.7,
    'test_save': True,
    'timestamp': datetime.now().isoformat()
}

saved_path = ckpt_manager.save_checkpoint(
    state=state,
    episode=5,
    expectancy_R=0.250,  # Better than parashat to trigger best detection
    metrics=metrics
)

print(f"   ✅ Checkpoint saved: {saved_path}")

# Verify the saved checkpoint
if saved_path and os.path.exists(saved_path):
    print("\n5. Verifying saved checkpoint...")

    with open(saved_path, 'rb') as f:
        saved_data = pickle.load(f)

    print(f"   Episode: {saved_data.get('episode')}")
    print(f"   Expectancy: {saved_data.get('expectancy_R')}")

    if 'state' in saved_data and 'policy_state' in saved_data['state']:
        policy_state = saved_data['state']['policy_state']
        if policy_state is not None:
            param_count = sum(p.numel() for p in policy_state.values())
            print(f"   ✅ Policy weights saved: {param_count:,} parameters")
        else:
            print(f"   ❌ Policy state is None!")
    else:
        print(f"   ❌ No policy state in checkpoint!")

print("\n" + "="*60)
print("✅ TEST COMPLETE")
print("="*60)
