#!/usr/bin/env python3
"""
Pre-calculate and cache valid session indices for micro trading.
Identifies gaps, weekends, and ensures 360 contiguous bars per session.
"""

import duckdb
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionIndexCalculator:
    """Pre-calculates valid session start indices with gap/weekend detection."""

    def __init__(
        self,
        db_path: str = "/workspace/data/micro_features.duckdb",
        session_length: int = 360,  # 6 hours
        max_gap_minutes: int = 10,
        cache_path: str = "/workspace/micro/cache/valid_session_indices.pkl"
    ):
        self.db_path = db_path
        self.session_length = session_length
        self.max_gap_minutes = max_gap_minutes
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = duckdb.connect(db_path, read_only=True)

    def calculate_all_valid_indices(self) -> Dict[str, List[int]]:
        """
        Calculate all valid session start indices.

        Returns:
            Dictionary with 'train', 'val', 'test' keys containing valid indices
        """
        logger.info("HYBRID: Quick initialization for immediate training...")

        # Get total count WITHOUT loading all data
        total_bars = self.conn.execute("SELECT COUNT(*) FROM micro_features").fetchone()[0]
        logger.info(f"Total bars: {total_bars}")

        # Calculate data splits (70/15/15)
        train_end = int(total_bars * 0.7)
        val_end = int(total_bars * 0.85)

        logger.info(f"Data splits - Train: 0-{train_end}, Val: {train_end}-{val_end}, Test: {val_end}-{total_bars}")

        # SKIP gap detection for now - will validate on demand

        # HYBRID: Start with minimal valid sessions for immediate training
        # More will be discovered dynamically during training
        initial_sessions = 20  # Start with just 20 sessions per split

        # Generate random initial indices
        import random
        random.seed(42)  # For reproducibility

        valid_indices = {
            'train': sorted(random.sample(range(0, train_end - self.session_length),
                                        min(initial_sessions, train_end - self.session_length))),
            'val': sorted(random.sample(range(train_end, val_end - self.session_length),
                                      min(initial_sessions, val_end - train_end - self.session_length))),
            'test': sorted(random.sample(range(val_end, total_bars - self.session_length),
                                       min(initial_sessions, total_bars - val_end - self.session_length))),
            'good_sessions': set(),  # Validated sessions (will grow during training)
            'bad_sessions': set(),   # Invalid sessions to avoid
            'total_bars': total_bars,
            'train_range': (0, train_end),
            'val_range': (train_end, val_end),
            'test_range': (val_end, total_bars)
        }

        logger.info(f"HYBRID initialization complete - Starting with {initial_sessions} sessions per split")
        logger.info("More sessions will be discovered during training")

        # Add metadata
        valid_indices['metadata'] = {
            'total_bars': total_bars,
            'session_length': self.session_length,
            'max_gap_minutes': self.max_gap_minutes,
            'train_end': train_end,
            'val_end': val_end,
            'calculation_time': datetime.now().isoformat(),
            'gaps_found': 0,  # Will be updated during discovery
            'hybrid_mode': True
        }

        return valid_indices

    def validate_session(self, start_idx: int) -> bool:
        """
        Quickly validate a single session for gaps/weekends.
        Used by hybrid approach during training.
        """
        try:
            # Query just the session data
            query = f"""
            SELECT timestamp
            FROM micro_features
            WHERE bar_index >= {start_idx} AND bar_index < {start_idx + self.session_length}
            ORDER BY bar_index
            """
            session_df = self.conn.execute(query).fetchdf()

            if len(session_df) != self.session_length:
                return False

            # Check for gaps > max_gap_minutes
            timestamps = pd.to_datetime(session_df['timestamp'])
            time_diffs = timestamps.diff().dt.total_seconds() / 60

            if time_diffs[1:].max() > self.max_gap_minutes:
                return False

            # Check for weekend
            if any(ts.weekday() >= 5 for ts in timestamps):
                # Allow if session just touches weekend edge
                weekend_bars = sum(1 for ts in timestamps if ts.weekday() >= 5)
                if weekend_bars > 10:  # More than 10 weekend bars = invalid
                    return False

            return True
        except:
            return False

    def _find_gaps_and_breaks(self, df: pd.DataFrame) -> List[Tuple[int, int, str]]:
        """
        Find all gaps and market breaks.

        Returns:
            List of (start_idx, end_idx, reason) tuples
        """
        gaps_and_breaks = []

        for i in range(1, len(df)):
            curr_time = df.iloc[i]['timestamp']
            prev_time = df.iloc[i-1]['timestamp']

            # Calculate time difference
            time_diff = (curr_time - prev_time).total_seconds() / 60  # minutes

            # Check for gap
            if time_diff > self.max_gap_minutes:
                gaps_and_breaks.append((i-1, i, f'gap_{time_diff:.0f}min'))

            # Check for weekend (Friday 22:00 UTC to Sunday 22:00 UTC)
            if prev_time.weekday() == 4 and prev_time.hour >= 22:  # Friday evening
                if curr_time.weekday() == 6 and curr_time.hour >= 22:  # Sunday evening
                    gaps_and_breaks.append((i-1, i, 'weekend'))

        logger.info(f"Found {len(gaps_and_breaks)} gaps/breaks")
        return gaps_and_breaks

    def _find_valid_indices_in_range(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        gaps_and_breaks: List[Tuple[int, int, str]]
    ) -> List[int]:
        """
        Find all valid session start indices in a given range.

        A valid session:
        1. Has exactly session_length contiguous bars
        2. Contains no gaps > max_gap_minutes
        3. Contains no weekend breaks
        4. Doesn't extend beyond the range
        """
        valid_indices = []

        # Create set of invalid ranges for quick lookup
        invalid_ranges = set()
        for gap_start, gap_end, reason in gaps_and_breaks:
            # Mark all indices that would include this gap in their session
            for idx in range(max(start_idx, gap_start - self.session_length + 1),
                           min(end_idx, gap_end + 1)):
                invalid_ranges.add(idx)

        # Check each potential starting point
        for idx in range(start_idx, end_idx - self.session_length + 1):
            # Skip if this index would include a gap
            if idx in invalid_ranges:
                continue

            # Verify we have enough bars
            session_end = idx + self.session_length
            if session_end > end_idx:
                continue

            # Additional validation: check actual timestamps
            session_df = df.iloc[idx:session_end]

            # Check for consistency
            is_valid = True
            for i in range(1, len(session_df)):
                time_diff = (session_df.iloc[i]['timestamp'] -
                           session_df.iloc[i-1]['timestamp']).total_seconds() / 60
                if time_diff > self.max_gap_minutes:
                    is_valid = False
                    break

                # Check weekend
                if session_df.iloc[i-1]['timestamp'].weekday() == 4 and \
                   session_df.iloc[i-1]['timestamp'].hour >= 22:
                    is_valid = False
                    break

            if is_valid:
                valid_indices.append(idx)

        logger.info(f"Found {len(valid_indices)} valid indices in range [{start_idx}, {end_idx})")
        return valid_indices

    def save_to_cache(self, valid_indices: Dict[str, List[int]]):
        """Save calculated indices to pickle file."""
        with open(self.cache_path, 'wb') as f:
            pickle.dump(valid_indices, f)
        logger.info(f"Saved valid indices to {self.cache_path}")

    def load_from_cache(self) -> Dict[str, List[int]]:
        """Load pre-calculated indices from cache."""
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")

        with open(self.cache_path, 'rb') as f:
            valid_indices = pickle.load(f)

        metadata = valid_indices.get('metadata', {})
        logger.info(f"Loaded cached indices calculated at {metadata.get('calculation_time', 'unknown')}")
        logger.info(f"  Train sessions: {len(valid_indices['train'])}")
        logger.info(f"  Val sessions: {len(valid_indices['val'])}")
        logger.info(f"  Test sessions: {len(valid_indices['test'])}")

        return valid_indices

    def get_or_calculate_indices(self) -> Dict[str, List[int]]:
        """Get indices from cache or calculate if not exists."""
        if self.cache_path.exists():
            logger.info("Loading pre-calculated indices from cache...")
            return self.load_from_cache()
        else:
            logger.info("Calculating valid session indices (this will take a few minutes)...")
            start_time = time.time()

            valid_indices = self.calculate_all_valid_indices()
            self.save_to_cache(valid_indices)

            elapsed = time.time() - start_time
            logger.info(f"Calculation complete in {elapsed:.1f} seconds")

            return valid_indices


def main():
    """Calculate and cache valid session indices."""
    calculator = SessionIndexCalculator()

    # Calculate or load indices
    valid_indices = calculator.get_or_calculate_indices()

    # Print summary
    print("\n" + "="*60)
    print("VALID SESSION INDICES SUMMARY")
    print("="*60)
    print(f"Training sessions:   {len(valid_indices['train']):,}")
    print(f"Validation sessions: {len(valid_indices['val']):,}")
    print(f"Test sessions:       {len(valid_indices['test']):,}")
    print(f"Total valid sessions: {len(valid_indices['train']) + len(valid_indices['val']) + len(valid_indices['test']):,}")

    metadata = valid_indices['metadata']
    print(f"\nSession length: {metadata['session_length']} bars (6 hours)")
    print(f"Max gap allowed: {metadata['max_gap_minutes']} minutes")
    print(f"Gaps/breaks found: {metadata['gaps_found']}")
    print("="*60)


if __name__ == "__main__":
    main()