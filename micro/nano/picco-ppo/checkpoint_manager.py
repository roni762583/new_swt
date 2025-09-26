#!/usr/bin/env python3
"""
Checkpoint management system for PPO training.
Keeps only 2 recent checkpoints + best performer.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import pickle


class CheckpointManager:
    """Manages checkpoint saving and loading with automatic cleanup."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.best_checkpoint_file = self.checkpoint_dir / "best_checkpoint.pkl"

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'checkpoints': [],
            'best': None,
            'best_expectancy_R': float('-inf')
        }

    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def save_checkpoint(
        self,
        state: Dict,
        episode: int,
        expectancy_R: float,
        metrics: Optional[Dict] = None
    ) -> str:
        """
        Save checkpoint with automatic management.

        Args:
            state: Model/training state to save
            episode: Current episode number
            expectancy_R: Current expectancy in R-multiples
            metrics: Additional metrics to save

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_ep{episode:06d}_{timestamp}_R{expectancy_R:+.3f}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint_data = {
            'state': state,
            'episode': episode,
            'expectancy_R': expectancy_R,
            'timestamp': timestamp,
            'metrics': metrics or {}
        }

        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        # Update metadata
        checkpoint_info = {
            'name': checkpoint_name,
            'path': str(checkpoint_path),
            'episode': episode,
            'expectancy_R': expectancy_R,
            'timestamp': timestamp,
            'metrics': metrics or {}
        }

        self.metadata['checkpoints'].append(checkpoint_info)

        # Check if this is the best checkpoint
        if expectancy_R > self.metadata['best_expectancy_R']:
            print(f"🏆 New best checkpoint! Expectancy: {expectancy_R:.3f}R")
            self._save_best_checkpoint(checkpoint_data, checkpoint_info)
            self.metadata['best'] = checkpoint_info
            self.metadata['best_expectancy_R'] = expectancy_R

        # Clean up old checkpoints (keep only 2 recent + best)
        self._cleanup_checkpoints()

        # Save updated metadata
        self._save_metadata()

        print(f"💾 Saved checkpoint: {checkpoint_name}")
        return str(checkpoint_path)

    def _save_best_checkpoint(self, checkpoint_data: Dict, checkpoint_info: Dict):
        """Save a copy as the best checkpoint."""
        # Save best checkpoint data
        with open(self.best_checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        # Also save a readable JSON with best metrics
        best_metrics_file = self.checkpoint_dir / "best_metrics.json"
        with open(best_metrics_file, 'w') as f:
            json.dump({
                'episode': checkpoint_info['episode'],
                'expectancy_R': checkpoint_info['expectancy_R'],
                'timestamp': checkpoint_info['timestamp'],
                'metrics': checkpoint_info['metrics']
            }, f, indent=2)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only 2 recent + best."""
        # Sort checkpoints by episode (most recent first)
        checkpoints = sorted(
            self.metadata['checkpoints'],
            key=lambda x: x['episode'],
            reverse=True
        )

        # Identify checkpoints to keep
        keep_indices = set()

        # Keep the 2 most recent
        for i in range(min(2, len(checkpoints))):
            keep_indices.add(i)

        # Keep the best checkpoint
        if self.metadata['best']:
            for i, ckpt in enumerate(checkpoints):
                if ckpt['name'] == self.metadata['best']['name']:
                    keep_indices.add(i)
                    break

        # Delete checkpoints not in keep list
        checkpoints_to_delete = []
        for i, ckpt in enumerate(checkpoints):
            if i not in keep_indices:
                checkpoint_path = Path(ckpt['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"🗑️  Removed old checkpoint: {ckpt['name']}")
                checkpoints_to_delete.append(ckpt)

        # Update metadata to reflect only kept checkpoints
        for ckpt in checkpoints_to_delete:
            self.metadata['checkpoints'].remove(ckpt)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to specific checkpoint, or None for best

        Returns:
            Checkpoint data or None if not found
        """
        if checkpoint_path is None:
            # Load best checkpoint
            if self.best_checkpoint_file.exists():
                with open(self.best_checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                print(f"✅ Loaded best checkpoint (Expectancy: {checkpoint_data['expectancy_R']:.3f}R)")
                return checkpoint_data
            else:
                print("❌ No best checkpoint found")
                return None
        else:
            # Load specific checkpoint
            path = Path(checkpoint_path)
            if path.exists():
                with open(path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                print(f"✅ Loaded checkpoint from {path.name}")
                return checkpoint_data
            else:
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                return None

    def get_latest_checkpoint(self) -> Optional[Dict]:
        """Get the most recent checkpoint."""
        if not self.metadata['checkpoints']:
            return None

        latest = max(self.metadata['checkpoints'], key=lambda x: x['episode'])
        return self.load_checkpoint(latest['path'])

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        checkpoints = []
        for ckpt in self.metadata['checkpoints']:
            checkpoints.append({
                'name': ckpt['name'],
                'episode': ckpt['episode'],
                'expectancy_R': ckpt['expectancy_R'],
                'timestamp': ckpt['timestamp']
            })

        # Sort by episode
        checkpoints.sort(key=lambda x: x['episode'], reverse=True)
        return checkpoints

    def get_best_metrics(self) -> Optional[Dict]:
        """Get metrics for the best checkpoint."""
        if self.metadata['best']:
            return {
                'episode': self.metadata['best']['episode'],
                'expectancy_R': self.metadata['best']['expectancy_R'],
                'metrics': self.metadata['best']['metrics']
            }
        return None


def test_checkpoint_manager():
    """Test the checkpoint manager functionality."""
    print("Testing Checkpoint Manager")
    print("="*50)

    # Create test manager
    manager = CheckpointManager("test_checkpoints")

    # Simulate saving checkpoints
    for episode in [100, 200, 300, 400, 500]:
        expectancy_R = (episode - 250) / 500  # Simulate improving performance

        # Mock state
        state = {
            'model_weights': f"weights_{episode}",
            'optimizer_state': f"opt_{episode}",
            'episode': episode
        }

        # Mock metrics
        metrics = {
            'win_rate': 50 + episode / 20,
            'total_pips': episode * 2.5,
            'trades': episode // 2
        }

        manager.save_checkpoint(state, episode, expectancy_R, metrics)
        print()

    # List checkpoints
    print("\nAvailable Checkpoints:")
    print("-"*30)
    for ckpt in manager.list_checkpoints():
        print(f"Episode {ckpt['episode']:6d}: {ckpt['expectancy_R']:+.3f}R - {ckpt['name']}")

    # Load best checkpoint
    print("\nLoading best checkpoint:")
    best = manager.load_checkpoint()
    if best:
        print(f"  Episode: {best['episode']}")
        print(f"  Expectancy: {best['expectancy_R']:.3f}R")

    # Get best metrics
    print("\nBest checkpoint metrics:")
    best_metrics = manager.get_best_metrics()
    if best_metrics:
        print(f"  Episode: {best_metrics['episode']}")
        print(f"  Expectancy: {best_metrics['expectancy_R']:.3f}R")
        print(f"  Win Rate: {best_metrics['metrics']['win_rate']:.1f}%")

    # Clean up test directory
    import shutil
    if Path("test_checkpoints").exists():
        shutil.rmtree("test_checkpoints")
        print("\n✅ Test cleanup complete")


if __name__ == "__main__":
    test_checkpoint_manager()