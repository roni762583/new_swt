#!/usr/bin/env python3
"""
Simple CSV Session Manager for SWT Training
CSV-only implementation without duckdb dependencies
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DataSplit = Literal['train', 'validate', 'test']

@dataclass
class SWTSessionConfig:
    """Configuration for SWT session management"""
    session_hours: float = 6.0  # 6-hour sessions (360 M1 bars)
    walk_forward_hours: float = 1.0  # 1-hour walk forward (60 M1 bars)
    max_gap_minutes: int = 10  # Skip sessions with gaps > 10 minutes
    min_session_bars: int = 300  # Minimum bars for valid session (5 hours)
    max_sessions: int = 1000  # Maximum sessions to retain
    
    # Data splits
    train_ratio: float = 0.8  # 80% training data
    validation_ratio: float = 0.1  # 10% validation data  
    test_ratio: float = 0.1  # 10% test data
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


class SimpleCsvSessionManager:
    """
    Simple CSV-based Session Manager for SWT
    
    Features:
    - 6-hour trading sessions with 1-hour walk-forward
    - Train/Validation/Test data splits (80/10/10)
    - Gap detection and session skipping
    - Direct CSV loading without duckdb
    """
    
    def __init__(self, 
                 data_path: str = "data/GBPJPY_M1_REAL_2022-2025.csv",
                 config: Optional[SWTSessionConfig] = None):
        self.data_path = Path(data_path)
        self.config = config or SWTSessionConfig()
        self.current_session_id = None
        self.current_session_data = None
        
        # Data splits
        self.data_splits: Dict[DataSplit, SWTDataSplitInfo] = {}
        self.current_split: DataSplit = 'train'
        
        # Load CSV data once
        self.df = None
        self._load_csv_data()
        
        # Initialize data splits
        self._initialize_data_splits()
        
        logger.info(f"✅ Simple CSV Session Manager initialized")
        logger.info(f"   Data file: {self.data_path}")
        logger.info(f"   Session: {self.config.session_hours}h + {self.config.walk_forward_hours}h walk-forward")
        logger.info(f"   Data splits: Train={self.config.train_ratio:.0%}, Val={self.config.validation_ratio:.0%}, Test={self.config.test_ratio:.0%}")
    
    def _load_csv_data(self):
        """Load CSV data once at initialization"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if self.data_path.suffix != '.csv':
            raise ValueError(f"Only CSV files are supported, got: {self.data_path}")
        
        logger.info(f"Loading CSV data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Handle different column name formats
        if 'time' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['time'])
        elif 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        else:
            raise ValueError("CSV must have 'time' or 'timestamp' column")
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.df):,} bars from {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
    
    def _initialize_data_splits(self):
        """Initialize train/validation/test data splits"""
        min_time = self.df['timestamp'].min()
        max_time = self.df['timestamp'].max()
        total_bars = len(self.df)
        
        logger.info(f"📊 Full dataset: {total_bars:,} bars from {min_time} to {max_time}")
        
        # Calculate split boundaries (chronological)
        total_duration = max_time - min_time
        train_duration = total_duration * self.config.train_ratio
        val_duration = total_duration * self.config.validation_ratio
        
        train_start = min_time
        train_end = train_start + train_duration
        val_start = train_end
        val_end = val_start + val_duration
        test_start = val_end
        test_end = max_time
        
        # Calculate sessions available for each split
        train_hours = train_duration.total_seconds() / 3600
        val_hours = val_duration.total_seconds() / 3600
        test_hours = (test_end - test_start).total_seconds() / 3600
        
        # Sessions = (total_hours - session_hours) / walk_forward_hours
        train_sessions = max(0, int((train_hours - self.config.session_hours) / self.config.walk_forward_hours))
        val_sessions = max(0, int((val_hours - self.config.session_hours) / self.config.walk_forward_hours))
        test_sessions = max(0, int((test_hours - self.config.session_hours) / self.config.walk_forward_hours))
        
        # Get actual bar counts for splits
        train_mask = (self.df['timestamp'] >= train_start) & (self.df['timestamp'] <= train_end)
        val_mask = (self.df['timestamp'] >= val_start) & (self.df['timestamp'] <= val_end)
        test_mask = (self.df['timestamp'] >= test_start) & (self.df['timestamp'] <= test_end)
        
        train_bars = train_mask.sum()
        val_bars = val_mask.sum()
        test_bars = test_mask.sum()
        
        # Store split information
        self.data_splits['train'] = SWTDataSplitInfo(
            name='train',
            start_time=train_start,
            end_time=train_end,
            total_bars=train_bars,
            percentage=self.config.train_ratio,
            sessions_available=train_sessions
        )
        
        self.data_splits['validate'] = SWTDataSplitInfo(
            name='validate',
            start_time=val_start,
            end_time=val_end,
            total_bars=val_bars,
            percentage=self.config.validation_ratio,
            sessions_available=val_sessions
        )
        
        self.data_splits['test'] = SWTDataSplitInfo(
            name='test',
            start_time=test_start,
            end_time=test_end,
            total_bars=test_bars,
            percentage=self.config.test_ratio,
            sessions_available=test_sessions
        )
        
        logger.info(f"📈 Training split: {train_bars:,} bars, {train_sessions:,} sessions")
        logger.info(f"📊 Validation split: {val_bars:,} bars, {val_sessions:,} sessions")
        logger.info(f"📉 Test split: {test_bars:,} bars, {test_sessions:,} sessions")
    
    def get_next_session_data(self, episode: int, split: Optional[DataSplit] = None) -> Optional[Dict[str, Any]]:
        """
        Get next 6-hour session data for episode
        
        Args:
            episode: Episode number for walk-forward calculation
            split: Data split to use ('train', 'validate', or 'test')
            
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
        
        # Calculate session offset within the split
        hours_offset = episode * self.config.walk_forward_hours
        session_start = split_info.start_time + timedelta(hours=hours_offset)
        session_end = session_start + timedelta(hours=self.config.session_hours)
        
        # Ensure we don't exceed split boundaries
        if session_end > split_info.end_time:
            logger.info(f"Session would exceed {data_split} split boundary")
            return None
        
        # Extract session data
        session_data = self._extract_session_data(session_start, session_end, data_split, episode)
        
        if session_data:
            self.current_session_id = episode
            self.current_session_data = session_data
        
        return session_data
    
    def _extract_session_data(self, start_time: pd.Timestamp, end_time: pd.Timestamp, 
                            data_split: DataSplit, episode: int) -> Optional[Dict[str, Any]]:
        """Extract session data from CSV"""
        
        # Filter data for time range
        mask = (self.df['timestamp'] >= start_time) & (self.df['timestamp'] <= end_time)
        session_df = self.df[mask].copy()
        
        # Check if session has sufficient data
        if len(session_df) < self.config.min_session_bars:
            logger.warning(f"Session too short ({len(session_df)} bars < {self.config.min_session_bars})")
            return None
        
        # Gap detection
        time_diffs = session_df['timestamp'].diff().dt.total_seconds() / 60
        large_gaps = time_diffs > self.config.max_gap_minutes
        
        if large_gaps.any():
            gap_count = large_gaps.sum()
            max_gap = time_diffs.max()
            logger.warning(f"Session rejected: {gap_count} gaps (max: {max_gap:.1f} minutes) - exceeds {self.config.max_gap_minutes}min limit")
            return None  # Reject sessions with gaps

        # Weekend detection (Friday 21:00 GMT to Sunday 21:00 GMT)
        weekend_present = False
        for _, row in session_df.iterrows():
            timestamp = row['timestamp']
            weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
            hour = timestamp.hour
            # Friday after 21:00 or Saturday or Sunday before 21:00
            if (weekday == 4 and hour >= 21) or weekday == 5 or (weekday == 6 and hour < 21):
                weekend_present = True
                break

        if weekend_present:
            logger.warning(f"Session rejected: weekend data detected (Friday 21:00+ or Saturday/Sunday before 21:00)")
            return None  # Reject sessions with weekend data
        
        # Prepare session data
        session_data = {
            'episode': episode,
            'split': data_split,
            'start_time': start_time,
            'end_time': end_time,
            'bars': len(session_df),
            'data': session_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values,
            'has_gaps': large_gaps.any()
        }
        
        logger.info(f"📊 Episode {episode} ({data_split}): {len(session_df)} bars from {start_time} to {end_time}")
        
        return session_data
    
    def get_random_session_data(self, split: DataSplit = 'train') -> Optional[Dict[str, Any]]:
        """Get a random session from the specified split"""
        split_info = self.data_splits.get(split)
        if not split_info or split_info.sessions_available == 0:
            return None
        
        # Pick a random episode
        random_episode = np.random.randint(0, split_info.sessions_available)
        return self.get_next_session_data(random_episode, split)
    
    def set_split(self, split: DataSplit):
        """Set the current data split"""
        if split not in self.data_splits:
            raise ValueError(f"Invalid split: {split}")
        self.current_split = split
        logger.info(f"Switched to {split} split")