#!/usr/bin/env python3
"""
SWT Session Manager - 6-hour sessions with 1-hour walk-forward
Adapted from V7/V8 Enhanced Session Manager for WST system
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass
# duckdb removed - using CSV only

logger = logging.getLogger(__name__)

DataSplit = Literal['train', 'validate', 'test']

@dataclass
class SWTSessionConfig:
    """Configuration for SWT session management"""
    session_hours: float = 12.0  # 12-hour sessions (720 M1 bars)
    walk_forward_hours: float = 1.0  # 1-hour walk forward (60 M1 bars)
    max_gap_minutes: int = 10  # Skip sessions with gaps > 10 minutes
    min_session_bars: int = 300  # Minimum bars for valid session (5 hours)
    max_sessions: int = 1000  # Maximum sessions to retain
    
    # Data splits
    train_ratio: float = 0.95  # 95% training data
    test_ratio: float = 0.05  # 5% test data
    max_session_retries: int = 3  # Retries to find good session


@dataclass
class SWTDataSplitInfo:
    """Information about data splits"""
    name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    total_bars: int
    percentage: float
    sessions_available: int


class SWTSessionManager:
    """
    SWT Session Manager for WST-Enhanced Stochastic MuZero
    
    Features:
    - 6-hour trading sessions with 1-hour walk-forward
    - Train/Test data splits (95/5)
    - Gap detection and session skipping (>10 minutes)
    - Random session sampling for diverse training
    - Session data extraction for WST processing
    """
    
    def __init__(self, 
                 db_path: str = "swt_data/trading_db_muzero_pipeline.duckdb",
                 config: Optional[SWTSessionConfig] = None):
        self.db_path = Path(db_path)
        self.config = config or SWTSessionConfig()
        self.current_session_id = None
        self.current_session_data = None
        
        # Data splits
        self.data_splits: Dict[DataSplit, SWTDataSplitInfo] = {}
        self.current_split: DataSplit = 'train'
        
        # Initialize data splits
        self._initialize_data_splits()
        
        logger.info(f"✅ SWT Session Manager initialized")
        logger.info(f"   Database: {self.db_path}")
        logger.info(f"   Session: {self.config.session_hours}h + {self.config.walk_forward_hours}h walk-forward")
        logger.info(f"   Gap tolerance: {self.config.max_gap_minutes} minutes")
        logger.info(f"   Data splits: Train={self.config.train_ratio:.0%}, Test={self.config.test_ratio:.0%}")
    
    def _initialize_data_splits(self):
        """Initialize train/test data splits"""
        try:
            # Handle CSV files directly
            if self.db_path.suffix == '.csv':
                # Load CSV data
                import pandas as pd
                df = pd.read_csv(self.db_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                min_time = df['timestamp'].min()
                max_time = df['timestamp'].max()
                total_bars = len(df)
                result = [(min_time, max_time, total_bars)]
            else:
                # Connect to database
                conn = duckdb.connect(str(self.db_path))
                
                # Get full data range
                result = conn.execute("""
                    SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time, COUNT(*) as total_bars
                    FROM v7_best_tcnae_trained_precalculated_latents
                """).fetchall()
            
            if not result:
                raise RuntimeError("No data found in database")
            
            min_time, max_time, total_bars = result[0]
            min_time = pd.to_datetime(min_time)
            max_time = pd.to_datetime(max_time)
            
            logger.info(f"📊 Full dataset: {total_bars:,} bars from {min_time} to {max_time}")
            
            # Calculate split boundaries (chronological)
            total_duration = max_time - min_time
            train_duration = total_duration * self.config.train_ratio
            
            train_start = min_time
            train_end = train_start + train_duration
            test_start = train_end
            test_end = max_time
            
            # Calculate sessions available for each split
            train_hours = train_duration.total_seconds() / 3600
            test_hours = (test_end - test_start).total_seconds() / 3600
            
            # Sessions = (total_hours - session_hours) / walk_forward_hours
            train_sessions = max(0, int((train_hours - self.config.session_hours) / self.config.walk_forward_hours))
            test_sessions = max(0, int((test_hours - self.config.session_hours) / self.config.walk_forward_hours))
            
            # Get actual bar counts for splits
            if self.db_path.suffix == '.csv':
                # Count bars in CSV data splits
                train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)
                test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)
                train_bars = train_mask.sum()
                test_bars = test_mask.sum()
            else:
                train_bars = conn.execute("""
                    SELECT COUNT(*) FROM v7_best_tcnae_trained_precalculated_latents
                    WHERE timestamp >= ? AND timestamp <= ?
                """, (train_start, train_end)).fetchone()[0]
                
                test_bars = conn.execute("""
                    SELECT COUNT(*) FROM v7_best_tcnae_trained_precalculated_latents
                    WHERE timestamp >= ? AND timestamp <= ?
                """, (test_start, test_end)).fetchone()[0]
            
            # Store split information
            self.data_splits['train'] = SWTDataSplitInfo(
                name='train',
                start_time=train_start,
                end_time=train_end,
                total_bars=train_bars,
                percentage=self.config.train_ratio,
                sessions_available=train_sessions
            )
            
            self.data_splits['test'] = SWTDataSplitInfo(
                name='test',
                start_time=test_start,
                end_time=test_end,
                total_bars=test_bars,
                percentage=self.config.test_ratio,
                sessions_available=test_sessions
            )
            
            conn.close()
            
            logger.info(f"📈 Training split: {train_bars:,} bars, {train_sessions:,} sessions")
            logger.info(f"📉 Test split: {test_bars:,} bars, {test_sessions:,} sessions")
            
        except Exception as e:
            logger.error(f"Failed to initialize data splits: {e}")
            raise
    
    def get_next_session_data(self, episode: int, split: Optional[DataSplit] = None) -> Optional[Dict[str, Any]]:
        """
        Get next 6-hour session data for episode
        
        Args:
            episode: Episode number for walk-forward calculation
            split: Data split to use ('train' or 'test') or None for current
            
        Returns:
            Dictionary with session data or None if no more sessions
        """
        # Use specified split or current split
        data_split = split or self.current_split
        if data_split not in self.data_splits:
            logger.error(f"Invalid data split: {data_split}")
            return None
        
        split_info = self.data_splits[data_split]
        
        # Check if we have sessions available
        if episode >= split_info.sessions_available:
            logger.info(f"No more sessions available in {data_split} split (episode {episode} >= {split_info.sessions_available})")
            return None
        
        try:
            # Connect to database
            conn = duckdb.connect(str(self.db_path))
            
            # Calculate session offset within the split
            hours_offset = episode * self.config.walk_forward_hours
            session_start = split_info.start_time + timedelta(hours=hours_offset)
            session_end = session_start + timedelta(hours=self.config.session_hours)
            
            # Ensure we don't exceed split boundaries
            if session_end > split_info.end_time:
                logger.info(f"Session would exceed {data_split} split boundary")
                conn.close()
                return None
            
            # Extract session data with gap detection
            session_data = self._extract_session_data(conn, session_start, session_end, data_split, episode)
            
            conn.close()
            return session_data
            
        except Exception as e:
            logger.error(f"Error getting session data for episode {episode}, split {data_split}: {e}")
            return None
    
    def _extract_session_data(self, conn, start_time: pd.Timestamp, end_time: pd.Timestamp, 
                            data_split: DataSplit, episode: int) -> Optional[Dict[str, Any]]:
        """Extract session data with adaptive gap skipping"""
        
        split_info = self.data_splits[data_split]
        max_search_attempts = 50  # Aggressive search for valid sessions
        
        for attempt in range(max_search_attempts):
            try:
                # Ensure we don't exceed split boundaries
                if end_time > split_info.end_time:
                    logger.info(f"Reached end of {data_split} split at episode {episode}")
                    return None
                
                # Try to get data
                try:
                    # Try raw data table first
                    latents_df = conn.execute("""
                        SELECT timestamp, close, volume
                        FROM raw_forex_data
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp
                    """, (start_time, end_time)).fetchdf()
                except:
                    # Fallback to latents table
                    latents_df = conn.execute("""
                        SELECT timestamp, latent_0 as close, 100 as volume
                        FROM v7_best_tcnae_trained_precalculated_latents
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp
                    """, (start_time, end_time)).fetchdf()
                    
                    # Rescale latent_0 to reasonable price range
                    latents_df['close'] = (latents_df['close'] * 0.1) + 1.8
                
                # Check if session has sufficient data
                if len(latents_df) < self.config.min_session_bars:
                    logger.info(f"Session too short ({len(latents_df)} bars), skipping to next data window")
                    # Skip forward by 6 hours to find next potential session
                    start_time += timedelta(hours=6)
                    end_time += timedelta(hours=6)
                    continue
                
                # Gap detection
                timestamps = pd.to_datetime(latents_df['timestamp'])
                time_diffs = timestamps.diff().dt.total_seconds() / 60
                large_gaps = time_diffs > self.config.max_gap_minutes
                
                if large_gaps.any():
                    gap_idx = large_gaps.idxmax()
                    gap_size = time_diffs.iloc[gap_idx]
                    logger.info(f"Gap detected ({gap_size:.1f} min), skipping to after gap")
                    
                    # Skip to after the gap - find next valid data point
                    gap_time = timestamps.iloc[gap_idx]
                    next_data = conn.execute("""
                        SELECT MIN(timestamp) as next_start
                        FROM v7_best_tcnae_trained_precalculated_latents
                        WHERE timestamp > ?
                    """, (gap_time,)).fetchone()
                    
                    if next_data and next_data[0]:
                        next_start = pd.to_datetime(next_data[0])
                        start_time = next_start
                        end_time = start_time + timedelta(hours=self.config.session_hours)
                        logger.info(f"Skipping to {start_time} after gap")
                        continue
                    else:
                        logger.warning("No more data after gap")
                        return None
                
                # Session is valid
                session_data = {
                    'session_id': f"{data_split}_ep{episode:06d}_{start_time.strftime('%Y%m%d_%H%M')}",
                    'episode': episode,
                    'split': data_split,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_hours': self.config.session_hours,
                    'total_bars': len(latents_df),
                    'gap_detected': False,
                    'max_gap_minutes': time_diffs.max() if len(time_diffs) > 1 else 0,
                    'data': latents_df,
                    'attempt': attempt + 1
                }
                
                logger.info(f"✅ Session {session_data['session_id']}: {len(latents_df)} bars")
                return session_data
                
            except Exception as e:
                logger.error(f"Session extraction error: {e}")
                # Skip forward by 1 hour and try again
                start_time += timedelta(hours=1)
                end_time += timedelta(hours=1)
                continue
        
        logger.error(f"Failed to find valid session after {max_search_attempts} attempts")
        return None
    
    def set_split(self, split: DataSplit):
        """Set the current data split"""
        if split in self.data_splits:
            self.current_split = split
            logger.info(f"🎯 Switched to {split} split")
        else:
            logger.error(f"Invalid split: {split}")
    
    def get_split_info(self, split: Optional[DataSplit] = None) -> Optional[SWTDataSplitInfo]:
        """Get information about a data split"""
        data_split = split or self.current_split
        return self.data_splits.get(data_split)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        stats = {
            'current_split': self.current_split,
            'session_config': {
                'session_hours': self.config.session_hours,
                'walk_forward_hours': self.config.walk_forward_hours,
                'max_gap_minutes': self.config.max_gap_minutes,
                'min_session_bars': self.config.min_session_bars
            }
        }
        
        for split_name, split_info in self.data_splits.items():
            stats[f'{split_name}_split'] = {
                'start_time': str(split_info.start_time),
                'end_time': str(split_info.end_time),
                'total_bars': split_info.total_bars,
                'sessions_available': split_info.sessions_available,
                'percentage': split_info.percentage
            }
        
        return stats


def create_swt_session_manager(config: Optional[SWTSessionConfig] = None) -> SWTSessionManager:
    """Factory function to create SWT session manager"""
    return SWTSessionManager(config=config)