#!/usr/bin/env python3
"""
Unified runner for PPO training, validation, and evaluation.
Single container entry point for all operations.
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime
import json

def run_training(args):
    """Run training with specified parameters."""
    print("="*60)
    print("🚀 STARTING PPO TRAINING")
    print("="*60)

    if args.full:
        # Full PPO with neural network
        cmd = [
            "python", "train.py",
            "--timesteps", str(args.timesteps),
            "--n_envs", str(args.n_envs)
        ]
        if args.load_checkpoint:
            cmd.extend(["--load_model", args.load_checkpoint])
    else:
        # Minimal rule-based training
        cmd = ["python", "train_minimal.py"]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode

def run_validation(args):
    """Run validation on specified split."""
    print("="*60)
    print("🔬 RUNNING VALIDATION")
    print("="*60)

    cmd = ["python", "validate_minimal.py"]
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode

def run_evaluation(args):
    """Evaluate a trained model."""
    print("="*60)
    print("📊 EVALUATING TRAINED MODEL")
    print("="*60)

    if not args.model_path:
        # Find latest model
        models_dir = "models"
        if os.path.exists(models_dir):
            models = sorted([f for f in os.listdir(models_dir) if f.endswith('.zip')])
            if models:
                args.model_path = os.path.join(models_dir, models[-1])
                print(f"Using latest model: {args.model_path}")
            else:
                print("❌ No models found in models directory")
                return 1
        else:
            print("❌ Models directory not found")
            return 1

    cmd = [
        "python", "evaluate.py",
        "--model", args.model_path,
        "--n_episodes", str(args.n_episodes)
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode

def run_session(args):
    """Run a complete training session with validation."""
    print("="*60)
    print("🎯 RUNNING COMPLETE SESSION")
    print("="*60)

    session_results = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'full' if args.full else 'minimal'
    }

    # 1. Training
    print("\n📚 Phase 1: Training")
    print("-"*40)
    train_result = run_training(args)
    session_results['training_complete'] = (train_result == 0)

    if train_result != 0:
        print("❌ Training failed")
        return 1

    # 2. Validation
    print("\n🔍 Phase 2: Validation")
    print("-"*40)
    val_result = run_validation(args)
    session_results['validation_complete'] = (val_result == 0)

    # 3. Save session summary
    os.makedirs("results", exist_ok=True)
    session_file = f"results/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Load validation results if they exist
    val_files = sorted([f for f in os.listdir("results") if f.startswith("validation_")])
    if val_files:
        with open(f"results/{val_files[-1]}", 'r') as f:
            val_data = json.load(f)
            session_results['validation_expectancy_R'] = val_data.get('expectancy_R', 0)
            session_results['test_expectancy_R'] = val_data.get('expectancy_R', 0)

    with open(session_file, 'w') as f:
        json.dump(session_results, f, indent=2)

    # 4. Performance summary
    print("\n" + "="*60)
    print("📈 SESSION PERFORMANCE SUMMARY")
    print("="*60)

    if 'validation_expectancy_R' in session_results:
        exp_R = session_results['validation_expectancy_R']
        print(f"Validation Expectancy: {exp_R:.3f}R")

        if exp_R > 0.5:
            print("🏆 EXCELLENT SYSTEM - Ready for production!")
        elif exp_R > 0.25:
            print("✅ GOOD SYSTEM - Profitable with room to improve")
        elif exp_R > 0:
            print("⚠️ ACCEPTABLE - Marginally profitable")
        else:
            print("🔴 NEEDS IMPROVEMENT - Not yet profitable")

    print(f"\nResults saved to: {session_file}")
    return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified PPO Training/Validation/Evaluation Runner"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Training parser
    train_parser = subparsers.add_parser('train', help='Run training')
    train_parser.add_argument('--full', action='store_true',
                             help='Use full PPO with neural network')
    train_parser.add_argument('--timesteps', type=int, default=1000000,
                             help='Number of timesteps for full PPO')
    train_parser.add_argument('--n_envs', type=int, default=4,
                             help='Number of parallel environments')
    train_parser.add_argument('--load_checkpoint', type=str,
                             help='Load from checkpoint')

    # Validation parser
    val_parser = subparsers.add_parser('validate', help='Run validation')

    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model_path', type=str,
                            help='Path to model (uses latest if not specified)')
    eval_parser.add_argument('--n_episodes', type=int, default=10,
                            help='Number of episodes to evaluate')

    # Session parser (train + validate)
    session_parser = subparsers.add_parser('session',
                                          help='Run complete session (train + validate)')
    session_parser.add_argument('--full', action='store_true',
                               help='Use full PPO with neural network')
    session_parser.add_argument('--timesteps', type=int, default=1000000,
                               help='Number of timesteps for full PPO')
    session_parser.add_argument('--n_envs', type=int, default=4,
                               help='Number of parallel environments')

    args = parser.parse_args()

    if not args.command:
        print("Please specify a command: train, validate, evaluate, or session")
        parser.print_help()
        return 1

    # Route to appropriate function
    if args.command == 'train':
        return run_training(args)
    elif args.command == 'validate':
        return run_validation(args)
    elif args.command == 'evaluate':
        return run_evaluation(args)
    elif args.command == 'session':
        return run_session(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())