#!/usr/bin/env python3
"""
On-demand cache that loads ALL columns but only for requested rows.
This fixes the column mismatch issue while keeping memory usage low.
"""

import numpy as np
import pandas as pd
import duckdb
import logging
from typing import Dict, Optional, Tuple
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class OptimizedMemoryCache:
    """Cache that loads complete rows (all 297 columns) on demand."""

    def __init__(
        self,
        db_path: str = "/workspace/data/micro_features.duckdb",
        max_cached_sessions: int = 20,
        session_length: int = 360,
        lookback: int = 32
    ):
        self.db_path = db_path
        self.session_length = session_length
        self.lookback = lookback
        self.max_cached_sessions = max_cached_sessions

        # LRU cache for recent sessions (stores complete DataFrames with all columns)
        self.session_cache = OrderedDict()

        # Get column information at startup
        self._initialize()

    def _initialize(self):
        """Initialize - just get column info, no data loading."""
        logger.info("Initializing on-demand cache (all columns, minimal rows)...")

        conn = duckdb.connect(self.db_path, read_only=True,
                             config={'temp_directory': '/dev/shm'})

        # Get ALL columns - we need every single one for the features to work
        columns_df = conn.execute("DESCRIBE micro_features").fetchdf()
        self.all_columns = columns_df['column_name'].tolist()
        logger.info(f"Database has {len(self.all_columns)} columns (loading ALL)")

        # Get row count for info
        total_rows = conn.execute("SELECT COUNT(*) FROM micro_features").fetchone()[0]
        logger.info(f"Total rows: {total_rows:,}")

        conn.close()

        logger.info(f"Cache initialized - will load sessions on demand")
        logger.info(f"Max cached sessions: {self.max_cached_sessions}")
        logger.info(f"Estimated memory per session: ~3MB (393 rows x 297 cols x 8 bytes)")

    def get_session_data(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Get session data with ALL columns for the specific row range.
        Uses LRU caching to keep recent sessions in memory.
        Note: Caller is responsible for including lookback in start_idx.
        """
        # Don't add lookback - caller already handles it
        lookback_start = start_idx

        # Create cache key
        cache_key = f"{lookback_start}_{end_idx}"

        # Check if in cache (and move to end for LRU)
        if cache_key in self.session_cache:
            self.session_cache.move_to_end(cache_key)
            return self.session_cache[cache_key].copy()

        # Load from database with ALL columns
        conn = duckdb.connect(self.db_path, read_only=True,
                             config={'temp_directory': '/dev/shm'})

        # Build query for ALL columns
        columns_str = ", ".join(self.all_columns)
        query = f"""
        SELECT {columns_str}
        FROM micro_features
        WHERE bar_index >= {lookback_start} AND bar_index <= {end_idx}
        ORDER BY bar_index
        """

        # Load the data
        df = conn.execute(query).fetchdf()
        conn.close()

        # Set index
        if 'bar_index' in df.columns:
            df.set_index('bar_index', inplace=True)

        # Add to cache
        self.session_cache[cache_key] = df

        # Evict oldest if cache is full
        if len(self.session_cache) > self.max_cached_sessions:
            self.session_cache.popitem(last=False)  # Remove oldest (first) item

        # Log cache status periodically
        if len(self.session_cache) % 10 == 0:
            logger.debug(f"Cache has {len(self.session_cache)} sessions")

        return df.copy()

    def _estimate_memory_usage(self) -> float:
        """Estimate current cache memory usage in MB."""
        if not self.session_cache:
            return 0.0

        # Estimate: each session is ~393 rows x 297 columns x 8 bytes
        # Plus DataFrame overhead
        per_session_mb = (393 * 297 * 8) / (1024 * 1024)  # ~0.9 MB per session
        overhead = 1.5  # DataFrame overhead factor

        return len(self.session_cache) * per_session_mb * overhead

    def clear_cache(self):
        """Clear the session cache to free memory."""
        self.session_cache.clear()
        logger.info("Cache cleared")


# Global cache instance
_optimized_cache = None


def get_optimized_cache() -> OptimizedMemoryCache:
    """Get or create the global optimized cache instance."""
    global _optimized_cache
    if _optimized_cache is None:
        _optimized_cache = OptimizedMemoryCache()
    return _optimized_cache


if __name__ == "__main__":
    # Test the optimized cache
    logging.basicConfig(level=logging.INFO)

    cache = OptimizedMemoryCache(
        db_path="/home/aharon/projects/new_swt/data/micro_features.duckdb",
        max_cached_sessions=20
    )

    logger.info(f"Cache initialized")
    logger.info(f"Estimated memory usage: {cache._estimate_memory_usage():.1f} MB")

    # Test data retrieval
    test_start = 50000
    test_end = test_start + 360

    # First fetch (from database)
    start_time = time.time()
    data1 = cache.get_session_data(test_start, test_end)
    elapsed1 = time.time() - start_time
    logger.info(f"First fetch: {elapsed1:.3f}s, shape: {data1.shape}")

    # Second fetch (from cache)
    start_time = time.time()
    data2 = cache.get_session_data(test_start, test_end)
    elapsed2 = time.time() - start_time
    logger.info(f"Second fetch (cached): {elapsed2:.3f}s, shape: {data2.shape}")

    logger.info(f"Cache speedup: {elapsed1/elapsed2:.1f}x")
    logger.info(f"Memory usage: {cache._estimate_memory_usage():.1f} MB")