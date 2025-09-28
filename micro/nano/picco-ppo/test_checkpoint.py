#!/usr/bin/env python3
"""
Test checkpoint saving with immediate save
"""
import sys
sys.path.append('/app')

from train_optimized import train_optimized

print('='*60)
print('TESTING CHECKPOINT SAVING')
print('='*60)
print('Will run episodes 5-6 (resuming from parashat_vayelech.pth)')
print('Checkpoint should be saved immediately with actual weights')
print('='*60)

# Run for 2 episodes (5-6) with 1 env for quick test
train_optimized(episodes=6, n_envs=1)