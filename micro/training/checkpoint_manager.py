#!/usr/bin/env python3
"""
Checkpoint management utilities for keeping only recent checkpoints.
"""

import os
import glob
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def cleanup_old_checkpoints(checkpoint_dir: str, keep_recent: int = 2):
    """
    Keep only the most recent N episode checkpoints plus best.pth and latest.pth.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_recent: Number of recent episode checkpoints to keep
    """
    # Get all episode checkpoints
    pattern = os.path.join(checkpoint_dir, "micro_checkpoint_ep*.pth")
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_recent:
        return  # Nothing to clean

    # Sort by episode number (extract from filename)
    def get_episode(path):
        filename = os.path.basename(path)
        # micro_checkpoint_ep000100.pth -> 100
        ep_str = filename.replace("micro_checkpoint_ep", "").replace(".pth", "")
        return int(ep_str)

    checkpoints.sort(key=get_episode)

    # Remove old checkpoints, keeping only the most recent
    to_remove = checkpoints[:-keep_recent]

    for checkpoint in to_remove:
        try:
            os.remove(checkpoint)
            logger.info(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            logger.error(f"Failed to remove {checkpoint}: {e}")

    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} old checkpoints, kept {keep_recent} recent ones")


def get_checkpoint_info(checkpoint_path: str) -> dict:
    """Get information about a checkpoint without loading the full model."""
    import torch

    try:
        # Load with map_location to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        info = {
            'episode': checkpoint.get('episode', 0),
            'total_steps': checkpoint.get('total_steps', 0),
            'temperature': checkpoint.get('temperature', 0),
        }

        # Get training stats if available
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            if 'expectancies' in stats and stats['expectancies']:
                recent = stats['expectancies'][-100:]
                info['recent_expectancy'] = sum(recent) / len(recent) if recent else 0
            if 'win_rates' in stats and stats['win_rates']:
                recent = stats['win_rates'][-100:]
                info['recent_win_rate'] = sum(recent) / len(recent) if recent else 0

        return info
    except Exception as e:
        logger.error(f"Failed to load checkpoint info: {e}")
        return {}