#!/usr/bin/env python3
"""
SWT Session Sampler
Randomly samples 24-hour continuous M1 data sets for training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class SessionWindow:
    """24-hour training session window"""
    start_index: int
    end_index: int
    start_timestamp: str
    end_timestamp: str
    session_id: str
    bar_count: int
    quality_score: float


class SWTSessionSampler:
    """
    Production-grade session sampler for SWT training
    Randomly samples 24-hour continuous M1 data windows
    """
    
    def __init__(self, data_path: str, session_hours: int = 6):
        self.data_path = Path(data_path)
        self.session_hours = session_hours
        self.session_bars = session_hours * 60  # M1 bars per session
        
        # Load and prepare data
        self._load_data()
        self._validate_data()
        self._index_sessions()
        
        logger.info(f"📊 SWT Session Sampler initialized")
        logger.info(f"   Data file: {self.data_path}")
        logger.info(f"   Total bars: {len(self.df):,}")
        logger.info(f"   Session length: {session_hours}h ({self.session_bars} bars)")
        logger.info(f"   Available sessions: {len(self.valid_sessions):,}")
        
    def _load_data(self) -> None:
        """Load and preprocess CSV data"""
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        logger.info(f"📥 Loading data from {self.data_path}")
        
        # Load CSV with proper data types
        dtype_dict = {
            'open': 'float32',
            'high': 'float32', 
            'low': 'float32',
            'close': 'float32',
            'volume': 'int32'
        }
        
        self.df = pd.read_csv(self.data_path, dtype=dtype_dict)
        
        # Convert timestamp and sort
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived features for session quality assessment (close-only data)
        self.df['price_change'] = self.df['close'].pct_change()
        self.df['volatility'] = self.df['close'].diff().abs().rolling(10).std()  # Price change volatility
        self.df['spread'] = 0.0004  # Fixed 4-pip spread for OANDA data
        
        logger.info(f"✅ Data loaded: {len(self.df):,} bars")
        
    def _validate_data(self) -> None:
        """Validate data integrity for session sampling"""
        
        # Check for critical issues
        null_prices = self.df[['open', 'high', 'low', 'close']].isnull().any(axis=1).sum()
        zero_prices = (self.df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        ohlc_errors = (
            (self.df['high'] < self.df['low']).sum() +
            (self.df['high'] < self.df['open']).sum() +
            (self.df['high'] < self.df['close']).sum() +
            (self.df['low'] > self.df['open']).sum() +
            (self.df['low'] > self.df['close']).sum()
        )
        
        if null_prices > 0:
            logger.warning(f"⚠️ Found {null_prices} bars with null prices")
        if zero_prices > 0:
            logger.warning(f"⚠️ Found {zero_prices} bars with zero/negative prices")  
        if ohlc_errors > 0:
            logger.warning(f"⚠️ Found {ohlc_errors} OHLC consistency errors")
            
        # Calculate time gaps
        time_diffs = self.df['timestamp'].diff()
        expected_interval = pd.Timedelta(minutes=1)
        large_gaps = (time_diffs > expected_interval * 5).sum()  # >5min gaps
        
        if large_gaps > 0:
            logger.warning(f"⚠️ Found {large_gaps} large time gaps (>5min)")
            
        logger.info(f"✅ Data validation complete - Ready for session sampling")
        
    def _index_sessions(self) -> None:
        """Index all valid 6-hour continuous sessions with weekend boundary enforcement"""
        
        self.valid_sessions = []
        total_bars = len(self.df)
        
        # Scan for continuous 6-hour windows
        for start_idx in range(0, total_bars - self.session_bars, 60):  # Check every hour
            end_idx = start_idx + self.session_bars
            
            if end_idx >= total_bars:
                break
                
            # Check session continuity
            session_data = self.df.iloc[start_idx:end_idx]
            time_diffs = session_data['timestamp'].diff().dropna()
            
            # Skip sessions with gaps >10 minutes
            max_gap_minutes = time_diffs.max().total_seconds() / 60
            
            if max_gap_minutes <= 10.0:  # Continuous session
                # CRITICAL: Check weekend boundary enforcement
                if not self._validate_weekend_boundaries(session_data):
                    continue  # Skip sessions that cross weekend
                    
                # Calculate session quality
                quality_score = self._calculate_session_quality(session_data)
                
                session = SessionWindow(
                    start_index=start_idx,
                    end_index=end_idx,
                    start_timestamp=session_data['timestamp'].iloc[0].isoformat(),
                    end_timestamp=session_data['timestamp'].iloc[-1].isoformat(),
                    session_id=f"session_{start_idx:08d}",
                    bar_count=len(session_data),
                    quality_score=quality_score
                )
                
                self.valid_sessions.append(session)
                
        logger.info(f"🔍 Session indexing complete: {len(self.valid_sessions):,} valid sessions found")
        logger.info(f"   Weekend boundary enforcement: ✅ Active")
        
    def _calculate_session_quality(self, session_data: pd.DataFrame) -> float:
        """Calculate quality score for a session (0-1)"""
        
        # Quality factors
        # 1. Data completeness
        completeness = 1.0 - (session_data.isnull().sum().sum() / (len(session_data) * len(session_data.columns)))
        
        # 2. Price movement (avoid flat/inactive periods)
        price_range = (session_data['high'].max() - session_data['low'].min()) / session_data['close'].mean()
        movement_score = min(price_range * 1000, 1.0)  # Scale to 0-1
        
        # 3. Volume activity 
        if 'volume' in session_data.columns:
            avg_volume = session_data['volume'].mean()
            volume_score = min(avg_volume / 100, 1.0)  # Basic volume scoring
        else:
            volume_score = 0.8  # Default if no volume data
            
        # 4. Volatility (market activity)
        volatility = session_data['volatility'].mean() / session_data['close'].mean()
        volatility_score = min(volatility * 10000, 1.0)  # Scale volatility
        
        # Weighted combination
        quality_score = (
            completeness * 0.4 +
            movement_score * 0.3 +
            volume_score * 0.1 +
            volatility_score * 0.2
        )
        
        return max(0.0, min(1.0, quality_score))
        
    def _validate_weekend_boundaries(self, session_data: pd.DataFrame) -> bool:
        """
        Validate that session doesn't cross weekend boundaries
        CRITICAL: Sessions must end before Friday market close to force position closure
        
        Args:
            session_data: 24-hour session DataFrame
            
        Returns:
            True if session is valid (no weekend crossing), False otherwise
        """
        
        start_time = pd.to_datetime(session_data['timestamp'].iloc[0])
        end_time = pd.to_datetime(session_data['timestamp'].iloc[-1])
        
        # Get day of week (0=Monday, 6=Sunday)
        start_weekday = start_time.weekday()
        end_weekday = end_time.weekday()
        
        # RULE 1: Session cannot start on weekend (Saturday=5, Sunday=6)
        if start_weekday >= 5:
            return False
            
        # RULE 2: Session cannot end on weekend  
        if end_weekday >= 5:
            return False
            
        # RULE 3: Session cannot span from Friday to Monday (weekend gap)
        if start_weekday == 4 and end_weekday == 0:  # Friday to Monday
            return False
            
        # RULE 4: For Friday sessions, must end before market close (21:00 GMT Friday)
        if start_weekday == 4:  # Friday session
            # Convert to GMT if needed (assuming UTC timestamps)
            friday_close_hour = 21  # 21:00 GMT Friday
            
            if end_time.weekday() == 4 and end_time.hour >= friday_close_hour:
                return False
                
        # RULE 5: Check for weekend gaps within session
        session_weekdays = pd.to_datetime(session_data['timestamp']).dt.weekday
        has_weekend_days = (session_weekdays >= 5).any()
        
        if has_weekend_days:
            return False
            
        return True
        
    def sample_random_session(self, min_quality: float = 0.3) -> Optional[SessionWindow]:
        """
        Sample a random 6-hour session above quality threshold
        
        Args:
            min_quality: Minimum quality score required
            
        Returns:
            SessionWindow or None if no qualifying sessions
        """
        
        # Filter sessions by quality
        qualified_sessions = [
            session for session in self.valid_sessions 
            if session.quality_score >= min_quality
        ]
        
        if not qualified_sessions:
            logger.warning(f"⚠️ No sessions meet quality threshold {min_quality}")
            return None
            
        # Random selection
        session = random.choice(qualified_sessions)
        
        logger.debug(f"🎲 Sampled session {session.session_id}: "
                    f"Quality={session.quality_score:.3f}, "
                    f"Range={session.start_timestamp[:10]} to {session.end_timestamp[:10]}")
        
        return session
        
    def get_session_data(self, session: SessionWindow) -> pd.DataFrame:
        """
        Extract session data for training
        
        Args:
            session: SessionWindow to extract
            
        Returns:
            DataFrame with session data
        """
        
        session_df = self.df.iloc[session.start_index:session.end_index].copy()
        
        # Reset index for training
        session_df = session_df.reset_index(drop=True)
        
        # Add session metadata
        session_df.attrs['session_id'] = session.session_id
        session_df.attrs['quality_score'] = session.quality_score
        session_df.attrs['start_timestamp'] = session.start_timestamp
        session_df.attrs['end_timestamp'] = session.end_timestamp
        
        return session_df
        
    def sample_training_batch(
        self, 
        batch_size: int, 
        min_quality: float = 0.3,
        ensure_diversity: bool = True
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        Sample a batch of training sessions
        
        Args:
            batch_size: Number of sessions to sample
            min_quality: Minimum quality threshold
            ensure_diversity: Avoid temporal clustering
            
        Returns:
            List of (session_id, session_data) tuples
        """
        
        qualified_sessions = [
            session for session in self.valid_sessions 
            if session.quality_score >= min_quality
        ]
        
        if len(qualified_sessions) < batch_size:
            logger.warning(f"⚠️ Requested {batch_size} sessions, only {len(qualified_sessions)} available")
            batch_size = len(qualified_sessions)
            
        if ensure_diversity:
            # Sort by start time and sample with temporal spacing
            qualified_sessions.sort(key=lambda x: x.start_timestamp)
            
            # Ensure minimum temporal separation (1 week)
            min_separation_hours = 24 * 7  # 1 week
            selected_sessions = []
            
            for session in qualified_sessions:
                # Check temporal separation from already selected
                too_close = False
                for selected in selected_sessions:
                    start_time_1 = pd.to_datetime(session.start_timestamp)
                    start_time_2 = pd.to_datetime(selected.start_timestamp)
                    time_diff_hours = abs((start_time_1 - start_time_2).total_seconds() / 3600)
                    
                    if time_diff_hours < min_separation_hours:
                        too_close = True
                        break
                        
                if not too_close:
                    selected_sessions.append(session)
                    
                if len(selected_sessions) >= batch_size:
                    break
                    
            # If diversity requirement too strict, fall back to random
            if len(selected_sessions) < batch_size:
                selected_sessions = random.sample(qualified_sessions, batch_size)
                
        else:
            # Pure random sampling
            selected_sessions = random.sample(qualified_sessions, batch_size)
            
        # Extract session data
        batch_data = []
        for session in selected_sessions:
            session_data = self.get_session_data(session)
            batch_data.append((session.session_id, session_data))
            
        logger.info(f"📦 Sampled training batch: {len(batch_data)} sessions")
        if batch_data:
            avg_quality = np.mean([session.quality_score for session in selected_sessions])
            logger.info(f"   Average quality: {avg_quality:.3f}")
            
        return batch_data
        
    def get_session_statistics(self) -> Dict[str, any]:
        """Get comprehensive session sampling statistics"""
        
        if not self.valid_sessions:
            return {'error': 'No valid sessions indexed'}
            
        quality_scores = [session.quality_score for session in self.valid_sessions]
        
        # Temporal distribution
        start_dates = [pd.to_datetime(session.start_timestamp).date() for session in self.valid_sessions]
        date_range = (min(start_dates), max(start_dates))
        
        # Quality distribution
        quality_stats = {
            'mean': np.mean(quality_scores),
            'median': np.median(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores),
            'q75': np.percentile(quality_scores, 75),
            'q90': np.percentile(quality_scores, 90)
        }
        
        # Quality thresholds
        high_quality = sum(1 for q in quality_scores if q >= 0.7)
        medium_quality = sum(1 for q in quality_scores if 0.3 <= q < 0.7)
        low_quality = sum(1 for q in quality_scores if q < 0.3)
        
        return {
            'total_sessions': len(self.valid_sessions),
            'total_data_bars': len(self.df),
            'session_length_hours': self.session_hours,
            'session_length_bars': self.session_bars,
            'date_range': date_range,
            'quality_distribution': {
                'high_quality_sessions': high_quality,
                'medium_quality_sessions': medium_quality, 
                'low_quality_sessions': low_quality,
                'statistics': quality_stats
            },
            'sampling_efficiency': len(self.valid_sessions) / max(1, len(self.df) // self.session_bars)
        }


def create_session_sampler(data_path: str, session_hours: int = 24) -> SWTSessionSampler:
    """Factory function to create session sampler"""
    return SWTSessionSampler(data_path, session_hours)


if __name__ == "__main__":
    # Example usage
    sampler = create_session_sampler("swt_data/oanda_full_3year_data.csv")
    
    # Get statistics
    stats = sampler.get_session_statistics()
    print(f"Sessions available: {stats['total_sessions']:,}")
    print(f"Quality distribution: High={stats['quality_distribution']['high_quality_sessions']}, "
          f"Medium={stats['quality_distribution']['medium_quality_sessions']}, "
          f"Low={stats['quality_distribution']['low_quality_sessions']}")
    
    # Sample a batch
    batch = sampler.sample_training_batch(batch_size=4, min_quality=0.3)
    print(f"Sampled {len(batch)} sessions for training")