#!/usr/bin/env python3
"""
SWT Checkpoint Manager - Production-grade checkpointing system
Enhanced from V7/V8 for WST-Enhanced Stochastic MuZero

Features:
- Frequent checkpoints with smart retention
- Complete WST + 5-Network Stochastic MuZero state
- Experience buffer persistence  
- Training resumption from any checkpoint
- Best model tracking and automatic cleanup
- Emergency backup system
"""

import torch
import numpy as np
import pandas as pd
import logging
import json
import shutil
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import heapq

logger = logging.getLogger(__name__)


@dataclass
class SWTCheckpointMetadata:
    """Enhanced metadata for SWT checkpoint tracking with comprehensive session data"""
    episode: int
    timestamp: str
    total_reward: float
    total_trades: int
    win_rate: float
    avg_pnl_pips: float
    avg_amddp5_reward: float
    avg_amddp1_reward: float  # Average AMDDP1 reward for comparison
    model_loss: float
    wst_backend: str = "kymatio"
    session_id: Optional[int] = None
    checkpoint_type: str = "regular"  # regular, best, emergency, milestone
    file_size_mb: float = 0.0
    network_architecture: str = "SWT-5-network"
    curriculum_stage: str = "full_reward"
    expectancy: float = 0.0  # Trading expectancy (avg_profit / avg_loss)
    avg_loss_pips: float = 0.0  # Average loss for expectancy calculation
    
    # Enhanced session data
    session_data: Optional[Dict[str, Any]] = None  # Complete session analytics
    trading_summary: Optional[Dict[str, Any]] = None  # Detailed trading stats
    exploration_rate: Optional[float] = None  # Epsilon exploration rate used
    session_duration: Optional[float] = None  # Session execution time in seconds
    bars_processed: Optional[int] = None  # Number of market bars processed
    quality_score: Optional[float] = None  # Calculated trading quality score
    
    # Market condition context
    market_session_id: Optional[str] = None  # Actual market session identifier
    market_timeframe: Optional[str] = None  # Market timeframe (e.g., "2022-09-22 12:23 - 18:23")
    price_volatility: Optional[float] = None  # Session price volatility (std)
    market_trend: Optional[str] = None  # Detected trend direction
    
    # Performance analytics
    max_drawdown_pips: Optional[float] = None  # Maximum drawdown during session
    sharpe_ratio: Optional[float] = None  # Risk-adjusted returns
    profit_factor: Optional[float] = None  # Gross profit / Gross loss
    consecutive_wins: Optional[int] = None  # Longest winning streak
    consecutive_losses: Optional[int] = None  # Longest losing streak


@dataclass 
class SWTCheckpointConfig:
    """Configuration for SWT checkpoint management"""
    checkpoint_dir: str = "swt_checkpoints"
    save_frequency: int = 25  # Save every 25 episodes
    keep_best_count: int = 1  # Keep 1 best checkpoint
    keep_recent_count: int = 2  # Keep 2 most recent checkpoints
    emergency_backup_frequency: int = 100  # Emergency backup every 100 episodes
    auto_cleanup: bool = False
    compress_checkpoints: bool = False


class SWTCheckpointManager:
    """Production-grade checkpoint management for SWT system"""
    
    def __init__(self, config: SWTCheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoints: List[SWTCheckpointMetadata] = []
        self.best_reward = -float('inf')
        self.best_checkpoint_path: Optional[Path] = None
        
        # Load existing checkpoints
        self._load_checkpoint_registry()
        
        logger.info(f"🗃️ SWT Checkpoint Manager initialized")
        logger.info(f"   Directory: {self.checkpoint_dir}")
        logger.info(f"   Save frequency: {config.save_frequency} episodes")
        logger.info(f"   Keep best: {config.keep_best_count}, recent: {config.keep_recent_count}")
        
    def save_checkpoint(self, 
                       state: Dict[str, Any],
                       metadata: SWTCheckpointMetadata,
                       is_best: bool = False,
                       force_save: bool = False) -> Path:
        """
        Save complete SWT system checkpoint
        
        Args:
            state: Complete training state (models, buffers, optimizers, etc.)
            metadata: Checkpoint metadata
            is_best: Whether this is the best model so far
            force_save: Force save regardless of frequency
            
        Returns:
            Path to saved checkpoint
        """
        
        # Determine checkpoint type
        if is_best:
            metadata.checkpoint_type = "best"
        elif metadata.episode % self.config.emergency_backup_frequency == 0:
            metadata.checkpoint_type = "emergency"
        elif metadata.episode % (self.config.save_frequency * 4) == 0:
            metadata.checkpoint_type = "milestone" 
        
        # Create checkpoint filename with PNL suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pnl_pips = int(round(metadata.avg_pnl_pips * 100))  # Convert to pips and round
        filename = f"swt_checkpoint_ep{metadata.episode:06d}_{timestamp}_{pnl_pips:+04d}.pth"
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare complete checkpoint state
        full_checkpoint = {
            # Core models
            'market_encoder_state': state.get('market_encoder_state'),
            'muzero_network_state': state.get('muzero_network_state'),
            'mcts_state': state.get('mcts_state', {}),
            
            # Training state
            'optimizer_state': state.get('optimizer_state'),
            'scheduler_state': state.get('scheduler_state'),
            'episode': metadata.episode,
            'best_reward': metadata.total_reward,
            
            # Experience buffer
            'experience_buffer': state.get('experience_buffer', []),
            'buffer_size': len(state.get('experience_buffer', [])),
            
            # Training metrics
            'training_metrics': state.get('training_metrics', {}),
            'loss_history': state.get('loss_history', []),
            
            # SWT-specific state
            'wst_backend': metadata.wst_backend,
            'curriculum_stage': metadata.curriculum_stage,
            
            # Metadata
            'metadata': asdict(metadata),
            'config': state.get('config', {}),
            'timestamp': timestamp
        }
        
        try:
            # Save checkpoint
            torch.save(full_checkpoint, checkpoint_path)
            
            # Update metadata with file size
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            metadata.file_size_mb = file_size_mb
            
            # Update tracking
            self.checkpoints.append(metadata)
            
            # Generate PNG chart for this checkpoint
            self._generate_checkpoint_chart(checkpoint_path)
            
            # Update best tracking using trading quality score
            quality_score = self._calculate_trading_quality_score(metadata)
            if is_best or quality_score > self.best_reward:
                self.best_reward = quality_score
                self.best_checkpoint_path = checkpoint_path
                
                # Create symlink to best checkpoint
                best_link = self.checkpoint_dir / "best_checkpoint.pth"
                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                try:
                    best_link.symlink_to(checkpoint_path.name)
                except FileExistsError:
                    pass  # Symlink already exists
                
                # Copy best checkpoint and its PNG with same name
                self._update_best_checkpoint_files(checkpoint_path)
                
            logger.info(f"💾 Saved {metadata.checkpoint_type} checkpoint:")
            logger.info(f"   Episode {metadata.episode}: {checkpoint_path.name}")
            logger.info(f"   Reward: {metadata.total_reward:.2f}, Size: {file_size_mb:.1f}MB")
            
            # Auto-cleanup old checkpoints
            if self.config.auto_cleanup:
                self._cleanup_old_checkpoints()
                
            # Save checkpoint registry
            self._save_checkpoint_registry()
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            raise
    
    def _generate_checkpoint_chart(self, checkpoint_path: Path) -> None:
        """Generate PNG chart for checkpoint using visualization script"""
        try:
            # Run the visualization script
            script_path = Path(__file__).parent.parent / "swt_visualizations" / "checkpoint_trade_visualizer.py"
            
            if script_path.exists():
                result = subprocess.run([
                    sys.executable, str(script_path), str(checkpoint_path)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    logger.debug(f"📊 Generated chart for {checkpoint_path.name}")
                else:
                    logger.warning(f"Chart generation failed: {result.stderr}")
            else:
                logger.debug(f"Visualization script not found: {script_path}")
                
        except Exception as e:
            logger.debug(f"Chart generation error: {e}")
    
    def _update_best_checkpoint_files(self, checkpoint_path: Path) -> None:
        """Copy best checkpoint and its PNG with consistent naming"""
        try:
            # Copy checkpoint as best_checkpoint.pth
            best_checkpoint_file = self.checkpoint_dir / "best_checkpoint.pth"
            shutil.copy2(checkpoint_path, best_checkpoint_file)
            
            # Copy PNG as best_checkpoint.png if it exists
            png_path = checkpoint_path.with_suffix('.png')
            if png_path.exists():
                best_png_file = self.checkpoint_dir / "best_checkpoint.png"
                shutil.copy2(png_path, best_png_file)
                logger.info(f"📊 Updated best_checkpoint.png")
            
            logger.debug(f"Updated best checkpoint files")
            
        except Exception as e:
            logger.warning(f"Failed to update best checkpoint files: {e}")
            
    def load_checkpoint(self, checkpoint_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Load SWT checkpoint for training resumption
        
        Args:
            checkpoint_path: Path to checkpoint, or None for best checkpoint
            
        Returns:
            Complete checkpoint state
        """
        
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pth"
            
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            logger.info(f"📂 Loading checkpoint: {checkpoint_path.name}")
            
            # Fix PyTorch 2.6 weights_only issue with numpy serialization
            import torch.serialization
            try:
                # Try to get numpy scalar function for safe loading
                numpy_scalar = getattr(np.core.multiarray, 'scalar', None)
                if numpy_scalar:
                    torch.serialization.add_safe_globals([numpy_scalar])
            except AttributeError:
                pass
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Validate checkpoint structure
            required_keys = ['market_encoder_state', 'muzero_network_state', 'episode']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                logger.warning(f"⚠️ Missing checkpoint keys: {missing_keys}")
                
            metadata = checkpoint.get('metadata', {})
            episode = checkpoint.get('episode', 0)
            
            logger.info(f"✅ Checkpoint loaded successfully:")
            logger.info(f"   Episode: {episode}")
            logger.info(f"   Reward: {metadata.get('total_reward', 'Unknown')}")
            logger.info(f"   WST Backend: {metadata.get('wst_backend', 'Unknown')}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
            
    def get_resumption_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints for resumption"""
        
        if not self.checkpoints:
            return {
                'can_resume': False,
                'latest_episode': 0,
                'best_reward': 0.0,
                'available_checkpoints': 0
            }
            
        latest_checkpoint = max(self.checkpoints, key=lambda x: x.episode)
        best_checkpoint = max(self.checkpoints, key=lambda x: x.total_reward)
        
        return {
            'can_resume': True,
            'latest_episode': latest_checkpoint.episode,
            'latest_reward': latest_checkpoint.total_reward,
            'best_episode': best_checkpoint.episode, 
            'best_reward': best_checkpoint.total_reward,
            'available_checkpoints': len(self.checkpoints),
            'checkpoint_types': list(set(cp.checkpoint_type for cp in self.checkpoints))
        }
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints based on strict retention policy"""
        
        total_allowed = self.config.keep_best_count + self.config.keep_recent_count
        
        # CRITICAL: Only cleanup when we have MORE than allowed checkpoints
        # This prevents deleting the best checkpoint during exploration phases
        if len(self.checkpoints) <= total_allowed:
            logger.debug(f"Skipping cleanup: {len(self.checkpoints)} checkpoints <= {total_allowed} allowed")
            return
            
        # Additional safety: Never delete if we have fewer than 5 checkpoints
        # This allows exploration episodes to accumulate before aggressive cleanup
        if len(self.checkpoints) < 5:
            logger.info(f"Exploration phase: keeping all {len(self.checkpoints)} checkpoints")
            return
            
        # Get best checkpoint by trading quality score
        best_checkpoints = heapq.nlargest(self.config.keep_best_count, 
                                        self.checkpoints, 
                                        key=self._calculate_trading_quality_score)
        
        # Get last N checkpoints by episode number (most recent episodes)
        recent_checkpoints = heapq.nlargest(self.config.keep_recent_count,
                                         self.checkpoints,
                                         key=lambda cp: cp.episode)
        
        # Combine and deduplicate checkpoints to keep
        keep_checkpoints = set()
        for cp_list in [best_checkpoints, recent_checkpoints]:
            for cp in cp_list:
                keep_checkpoints.add((cp.episode, cp.timestamp))
                
        # Remove checkpoints not in keep list
        removed_count = 0
        for checkpoint in self.checkpoints[:]:
            cp_key = (checkpoint.episode, checkpoint.timestamp)
            if cp_key not in keep_checkpoints:
                # Find and remove file
                timestamp_clean = checkpoint.timestamp.replace('-', '').replace('T', '_').replace(':', '').split('.')[0]
                for pattern in [
                    f"*ep{checkpoint.episode:06d}*{timestamp_clean}*.pth",
                    f"*ep{checkpoint.episode:06d}*.pth"
                ]:
                    matching_files = list(self.checkpoint_dir.glob(pattern))
                    for file_path in matching_files:
                        try:
                            file_path.unlink()
                            removed_count += 1
                            logger.debug(f"Removed checkpoint: {file_path.name}")
                        except FileNotFoundError:
                            pass  # Already removed
                        
                self.checkpoints.remove(checkpoint)
                
        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} old checkpoints (keeping {len(keep_checkpoints)}/{total_allowed} total)")
            
    def _calculate_trading_quality_score(self, checkpoint: SWTCheckpointMetadata) -> float:
        """
        Calculate enhanced trading quality score with expectancy
        
        Prioritizes:
        1. Trading expectancy (risk-adjusted performance)
        2. Positive P&L in pips (absolute performance)
        3. Consistency (win_rate * total_trades)
        4. Penalizes zero-trade sessions heavily
        
        Args:
            checkpoint: Checkpoint metadata
            
        Returns:
            Quality score (higher = better)
        """
        # Zero trades = heavily penalized
        if checkpoint.total_trades == 0:
            return -1000.0
            
        # Primary: Expectancy score (risk-adjusted performance)
        expectancy_score = checkpoint.expectancy * 5.0  # Weight expectancy heavily
        
        # Secondary: Absolute P&L performance
        pnl_score = checkpoint.avg_pnl_pips * 0.5
        
        # Consistency bonus: favor multiple profitable trades
        consistency_factor = (checkpoint.win_rate / 100.0) * min(checkpoint.total_trades, 10)
        trade_bonus = consistency_factor * 2.0  # Up to +20 bonus for 10 trades at 100% win rate
            
        # Volume bonus: slight preference for more trading activity
        volume_bonus = min(checkpoint.total_trades * 0.1, 1.0)
        
        # Combine scores with expectancy as primary factor
        total_score = expectancy_score + pnl_score + trade_bonus + volume_bonus
        
        logger.debug(f"📊 Quality Score Breakdown:")
        logger.debug(f"   Expectancy: {checkpoint.expectancy:.4f} → Score: {expectancy_score:.2f}")
        logger.debug(f"   P&L: {checkpoint.avg_pnl_pips:.4f} → Score: {pnl_score:.2f}")
        logger.debug(f"   Consistency: {trade_bonus:.2f}, Volume: {volume_bonus:.2f}")
        logger.debug(f"   Total Score: {total_score:.2f}")
        
        return total_score
    
    def _calculate_expectancy(self, trades_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calculate trading expectancy using industry-standard formula
        
        Formula: Expectancy = Average Profit per Trade ÷ Average Loss
        
        Args:
            trades_data: List of trade dictionaries with pnl_pips
            
        Returns:
            Tuple of (expectancy, avg_loss_pips)
        """
        if not trades_data:
            return 0.0, 0.0
            
        # Extract P&L values
        pnl_values = [trade.get('pnl_pips', 0.0) for trade in trades_data]
        
        if not pnl_values:
            return 0.0, 0.0
            
        # Calculate average profit per trade
        avg_profit = sum(pnl_values) / len(pnl_values)
        
        # Get losing trades for risk calculation
        losses = [pnl for pnl in pnl_values if pnl < 0]
        
        if not losses:
            # No losses means infinite expectancy, but return high finite value
            return 999.0, 0.0
            
        # Calculate average loss (risk factor R)
        avg_loss = abs(sum(losses) / len(losses))
        
        if avg_loss == 0:
            return 999.0, 0.0
            
        # Calculate expectancy
        expectancy = avg_profit / avg_loss
        
        logger.debug(f"📊 Expectancy calculation:")
        logger.debug(f"   Trades: {len(trades_data)}, Losses: {len(losses)}")
        logger.debug(f"   Avg Profit: {avg_profit:.4f} pips")
        logger.debug(f"   Avg Loss: {avg_loss:.4f} pips")
        logger.debug(f"   Expectancy: {expectancy:.4f}")
        
        return expectancy, avg_loss
            
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk"""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        registry = {
            'checkpoints': [asdict(cp) for cp in self.checkpoints],
            'best_quality_score': self.best_reward,
            'config': asdict(self.config),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
            
    def _load_checkpoint_registry(self):
        """Load existing checkpoint registry"""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        if not registry_path.exists():
            return
            
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
                
            # Filter checkpoints to only include existing files
            existing_checkpoints = []
            for cp_data in registry.get('checkpoints', []):
                cp = SWTCheckpointMetadata(**cp_data)
                # Check if checkpoint file actually exists
                timestamp_clean = cp.timestamp.replace('-', '').replace('T', '_').replace(':', '').split('.')[0]
                pnl_pips = int(round(cp.avg_pnl_pips * 100))
                expected_filename = f"swt_checkpoint_ep{cp.episode:06d}_{timestamp_clean}_{pnl_pips:+04d}.pth"
                checkpoint_file = self.checkpoint_dir / expected_filename
                
                if checkpoint_file.exists():
                    existing_checkpoints.append(cp)
                else:
                    logger.debug(f"Skipping missing checkpoint: {expected_filename}")
            
            self.checkpoints = existing_checkpoints
            
            # Reset best_reward to start fresh comparison
            self.best_reward = -float('inf')
            
            # If we have existing checkpoints, find and set the best one
            if self.checkpoints:
                best_checkpoint = max(self.checkpoints, key=self._calculate_trading_quality_score)
                self.best_reward = self._calculate_trading_quality_score(best_checkpoint)
                
                # Create symlink to the actual best checkpoint from current run
                timestamp_clean = best_checkpoint.timestamp.replace('-', '').replace('T', '_').replace(':', '').split('.')[0]
                pnl_pips = int(round(best_checkpoint.avg_pnl_pips * 100))
                best_filename = f"swt_checkpoint_ep{best_checkpoint.episode:06d}_{timestamp_clean}_{pnl_pips:+04d}.pth"
                best_file_path = self.checkpoint_dir / best_filename
                
                if best_file_path.exists():
                    self.best_checkpoint_path = best_file_path
                    best_link = self.checkpoint_dir / "best_checkpoint.pth"
                    if best_link.exists() or best_link.is_symlink():
                        best_link.unlink()
                    try:
                        best_link.symlink_to(best_filename)
                        logger.info(f"Created best_checkpoint.pth -> {best_filename}")
                    except FileExistsError:
                        pass
                
                logger.info(f"Best checkpoint: Episode {best_checkpoint.episode} (score: {self.best_reward:.2f})")
            else:
                logger.info("No existing checkpoints found, starting fresh")
            
            logger.info(f"📋 Loaded {len(self.checkpoints)} existing checkpoints")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load checkpoint registry: {e}")


def create_swt_checkpoint_manager(config_dict: dict = None) -> SWTCheckpointManager:
    """Factory function to create SWT checkpoint manager"""
    
    if config_dict is None:
        config = SWTCheckpointConfig()
    else:
        config = SWTCheckpointConfig(**config_dict)
        
    return SWTCheckpointManager(config)