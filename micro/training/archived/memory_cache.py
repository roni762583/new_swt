#!/usr/bin/env python3
"""
In-memory cache for session data to eliminate database I/O during training.
Loads all session data into memory at startup for fast access.
"""

import numpy as np
import pandas as pd
import duckdb
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class MemorySessionCache:
    """Cache all session data in memory for fast access."""

    def __init__(
        self,
        db_path: str = "/workspace/data/micro_features.duckdb",
        session_indices_path: str = "/workspace/micro/cache/valid_session_indices.pkl",
        cache_path: str = "/workspace/micro/cache/session_data_cache.pkl"
    ):
        self.db_path = db_path
        self.session_indices_path = session_indices_path
        self.cache_path = cache_path

        # Session indices by split
        self.session_indices = {}

        # Cached session data
        self.session_cache = {}

        # Load everything into memory
        self._load_cache()

    def _load_cache(self):
        """Load or build the session data cache."""
        cache_file = Path(self.cache_path)

        # Try to load existing cache
        if cache_file.exists():
            try:
                logger.info(f"Loading session cache from {self.cache_path}")
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.session_indices = cache_data['indices']
                    self.session_cache = cache_data['sessions']
                logger.info(f"Loaded {len(self.session_cache)} cached sessions")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, rebuilding...")

        # Build new cache
        logger.info("Building session data cache...")
        self._build_cache()

        # Save cache
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'indices': self.session_indices,
                    'sessions': self.session_cache
                }, f)
            logger.info(f"Saved cache to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _build_cache(self):
        """Build the session cache from database."""
        start_time = time.time()

        # Load session indices
        with open(self.session_indices_path, 'rb') as f:
            self.session_indices = pickle.load(f)

        logger.info(f"Loading sessions - Train: {len(self.session_indices['train'])}, "
                   f"Val: {len(self.session_indices['val'])}, "
                   f"Test: {len(self.session_indices['test'])}")

        # Connect to database
        conn = duckdb.connect(self.db_path, read_only=True)

        # Cache all sessions
        total_sessions = 0
        for split, indices in self.session_indices.items():
            for session_info in indices:
                start_idx = session_info['start_idx']
                end_idx = session_info['end_idx']

                # Load session data with lookback
                lookback_start = max(0, start_idx - 32)

                # Build query for all features
                feature_queries = []

                # Technical features with lags
                tech_features = [
                    'ema_diff', 'bb_position', 'rsi_norm', 'macd_norm',
                    'atr_norm', 'volume_ratio', 'price_change', 'high_low_ratio', 'close_open_ratio'
                ]
                for feat in tech_features:
                    lag_cols = [f"{feat}_lag_{i}" for i in range(1, 33)]
                    feature_queries.extend(lag_cols)

                # Market features
                market_features = [
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                    'is_london', 'is_newyork', 'is_tokyo', 'is_sydney'
                ]
                feature_queries.extend(market_features)

                # Construct query
                columns = ", ".join(feature_queries)
                query = f"""
                SELECT idx, timestamp, {columns}
                FROM micro_features
                WHERE idx >= {lookback_start} AND idx <= {end_idx}
                ORDER BY idx
                """

                # Load data
                df = conn.execute(query).fetchdf()

                # Cache the session
                cache_key = f"{split}_{start_idx}_{end_idx}"
                self.session_cache[cache_key] = {
                    'data': df,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'lookback_start': lookback_start
                }

                total_sessions += 1
                if total_sessions % 10 == 0:
                    logger.info(f"Cached {total_sessions} sessions...")

        conn.close()

        elapsed = time.time() - start_time
        logger.info(f"Cached {total_sessions} sessions in {elapsed:.1f}s")
        logger.info(f"Total cache size: {self._get_cache_size_mb():.1f} MB")

    def get_session_data(self, split: str, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Get cached session data."""
        cache_key = f"{split}_{start_idx}_{end_idx}"

        if cache_key in self.session_cache:
            return self.session_cache[cache_key]['data'].copy()

        # Fallback to database if not cached
        logger.warning(f"Session {cache_key} not in cache, loading from database")
        return self._load_from_database(start_idx, end_idx)

    def _load_from_database(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Fallback to load from database."""
        conn = duckdb.connect(self.db_path, read_only=True)

        lookback_start = max(0, start_idx - 32)

        # Build query
        feature_queries = []

        # Technical features with lags
        tech_features = [
            'ema_diff', 'bb_position', 'rsi_norm', 'macd_norm',
            'atr_norm', 'volume_ratio', 'price_change', 'high_low_ratio', 'close_open_ratio'
        ]
        for feat in tech_features:
            lag_cols = [f"{feat}_lag_{i}" for i in range(1, 33)]
            feature_queries.extend(lag_cols)

        # Market features
        market_features = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_london', 'is_newyork', 'is_tokyo', 'is_sydney'
        ]
        feature_queries.extend(market_features)

        columns = ", ".join(feature_queries)
        query = f"""
        SELECT idx, timestamp, {columns}
        FROM micro_features
        WHERE idx >= {lookback_start} AND idx <= {end_idx}
        ORDER BY idx
        """

        df = conn.execute(query).fetchdf()
        conn.close()

        return df

    def get_random_session(self, split: str) -> Tuple[int, int]:
        """Get random session indices for a split."""
        if split not in self.session_indices:
            raise ValueError(f"Invalid split: {split}")

        indices = self.session_indices[split]
        if not indices:
            raise ValueError(f"No sessions available for split: {split}")

        session = np.random.choice(indices)
        return session['start_idx'], session['end_idx']

    def _get_cache_size_mb(self) -> float:
        """Estimate cache size in MB."""
        import sys
        total_size = 0

        # Size of indices
        total_size += sys.getsizeof(self.session_indices)
        for v in self.session_indices.values():
            total_size += sys.getsizeof(v)

        # Size of cached data
        for cache_data in self.session_cache.values():
            if 'data' in cache_data:
                # Estimate DataFrame size
                df = cache_data['data']
                total_size += df.memory_usage(deep=True).sum()

        return total_size / (1024 * 1024)


# Global cache instance
_cache_instance = None


def get_memory_cache() -> MemorySessionCache:
    """Get or create the global memory cache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MemorySessionCache()
    return _cache_instance


if __name__ == "__main__":
    # Test the cache
    logging.basicConfig(level=logging.INFO)

    cache = MemorySessionCache(
        db_path="/home/aharon/projects/new_swt/data/micro_features.duckdb"
    )

    # Test getting a random session
    start_idx, end_idx = cache.get_random_session('train')
    logger.info(f"Random train session: {start_idx} to {end_idx}")

    # Test getting session data
    data = cache.get_session_data('train', start_idx, end_idx)
    logger.info(f"Session data shape: {data.shape}")