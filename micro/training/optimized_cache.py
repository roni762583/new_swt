#!/usr/bin/env python3
"""
Optimized memory cache that only loads essential columns and uses chunking.
Reduces memory footprint from 2.6GB to ~300MB per worker.
"""

import numpy as np
import pandas as pd
import duckdb
import logging
from typing import Dict, Optional, Tuple
import time
from pathlib import Path
import gc

logger = logging.getLogger(__name__)


class OptimizedMemoryCache:
    """Optimized cache with selective column loading and chunking."""

    def __init__(
        self,
        db_path: str = "/workspace/data/micro_features.duckdb",
        chunk_size: int = 100000,  # Load 100k rows at a time
        max_cache_size: int = 500000,  # Max 500k rows in memory
        session_length: int = 360,
        lookback: int = 32
    ):
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.max_cache_size = max_cache_size
        self.session_length = session_length
        self.lookback = lookback

        # Cache structure: dict of chunks
        self.cache_chunks = {}
        self.chunk_ranges = []

        # Essential columns - just take first 100 columns to reduce memory
        # The database has 297 columns, we'll use a subset
        self.essential_columns = None  # Will be set dynamically

        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize cache with metadata and first chunk."""
        logger.info("Initializing optimized memory cache...")
        start_time = time.time()

        conn = duckdb.connect(self.db_path, read_only=True,
                             config={'temp_directory': '/tmp'})

        # Get column names - use first 100 columns to save memory
        all_columns = conn.execute("DESCRIBE micro_features").fetchdf()['column_name'].tolist()
        # Always include bar_index and timestamp, then take next 98 columns
        self.essential_columns = ['bar_index', 'timestamp'] + [
            col for col in all_columns[2:100] if col not in ['bar_index', 'timestamp']
        ]
        logger.info(f"Using {len(self.essential_columns)} columns out of {len(all_columns)} total")

        # Get total row count
        total_rows = conn.execute(
            "SELECT COUNT(*) FROM micro_features"
        ).fetchone()[0]

        logger.info(f"Total rows in database: {total_rows:,}")

        # Create chunk ranges
        for i in range(0, total_rows, self.chunk_size):
            end = min(i + self.chunk_size, total_rows)
            self.chunk_ranges.append((i, end))

        logger.info(f"Created {len(self.chunk_ranges)} chunks of {self.chunk_size:,} rows")

        # Load first few chunks (up to max_cache_size)
        rows_loaded = 0
        chunks_to_load = []

        for idx, (start, end) in enumerate(self.chunk_ranges):
            if rows_loaded + (end - start) > self.max_cache_size:
                break
            chunks_to_load.append(idx)
            rows_loaded += (end - start)

        logger.info(f"Loading {len(chunks_to_load)} initial chunks...")

        for chunk_idx in chunks_to_load:
            self._load_chunk(conn, chunk_idx)

        conn.close()

        elapsed = time.time() - start_time
        memory_mb = self._estimate_memory_usage()
        logger.info(f"Cache initialized in {elapsed:.1f}s")
        logger.info(f"Memory usage: {memory_mb:.1f} MB (vs 2600 MB full cache)")

    def _load_chunk(self, conn, chunk_idx: int):
        """Load a specific chunk into memory."""
        start, end = self.chunk_ranges[chunk_idx]

        columns = ", ".join(self.essential_columns)
        query = f"""
        SELECT {columns}
        FROM micro_features
        WHERE bar_index >= {start} AND bar_index < {end}
        ORDER BY bar_index
        """

        df = conn.execute(query).fetchdf()
        df.set_index('bar_index', inplace=True)

        # Separate timestamp column if present
        timestamp_col = None
        if 'timestamp' in df.columns:
            timestamp_col = df['timestamp'].values
            df = df.drop('timestamp', axis=1)

        # Store as numpy arrays to save memory
        self.cache_chunks[chunk_idx] = {
            'data': df.values.astype(np.float32),  # Use float32 to save memory
            'columns': df.columns.tolist(),
            'index': df.index.values,
            'timestamp': timestamp_col,
            'range': (start, end)
        }

    def get_session_data(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Get session data from cache or load from database."""
        lookback_start = max(0, start_idx - self.lookback)

        # Check which chunks we need
        needed_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(self.chunk_ranges):
            if chunk_start <= end_idx and chunk_end > lookback_start:
                needed_chunks.append(chunk_idx)

        # Collect data from cached chunks
        dfs = []
        missing_ranges = []

        for chunk_idx in needed_chunks:
            if chunk_idx in self.cache_chunks:
                chunk = self.cache_chunks[chunk_idx]
                # Extract relevant rows
                mask = (chunk['index'] >= lookback_start) & (chunk['index'] <= end_idx)
                if mask.any():
                    df_chunk = pd.DataFrame(
                        chunk['data'][mask],
                        index=chunk['index'][mask],
                        columns=chunk['columns']
                    )
                    dfs.append(df_chunk)
            else:
                chunk_range = self.chunk_ranges[chunk_idx]
                missing_ranges.append((
                    max(chunk_range[0], lookback_start),
                    min(chunk_range[1], end_idx + 1)
                ))

        # Load missing data from database if needed
        if missing_ranges:
            df_missing = self._load_from_database(missing_ranges)
            if df_missing is not None and not df_missing.empty:
                dfs.append(df_missing)

        # Combine all data
        if dfs:
            result = pd.concat(dfs).sort_index()
            # Filter to exact range needed
            return result.loc[lookback_start:end_idx].copy()
        else:
            # Fallback to database
            return self._load_from_database([(lookback_start, end_idx + 1)])

    def _load_from_database(self, ranges: list) -> pd.DataFrame:
        """Load specific ranges from database."""
        conn = duckdb.connect(self.db_path, read_only=True,
                             config={'temp_directory': '/tmp'})

        dfs = []
        for start, end in ranges:
            columns = ", ".join(self.essential_columns)
            query = f"""
            SELECT {columns}
            FROM micro_features
            WHERE bar_index >= {start} AND bar_index < {end}
            ORDER BY bar_index
            """
            df = conn.execute(query).fetchdf()
            if not df.empty:
                df.set_index('bar_index', inplace=True)
                dfs.append(df)

        conn.close()

        if dfs:
            return pd.concat(dfs).sort_index()
        return pd.DataFrame()

    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB."""
        total_bytes = 0
        for chunk in self.cache_chunks.values():
            # Data array size
            total_bytes += chunk['data'].nbytes
            # Index array size
            total_bytes += chunk['index'].nbytes * 8
            # Overhead estimate
            total_bytes += 1024

        return total_bytes / (1024 * 1024)

    def evict_chunk(self, chunk_idx: int):
        """Remove a chunk from cache to free memory."""
        if chunk_idx in self.cache_chunks:
            del self.cache_chunks[chunk_idx]
            gc.collect()

    def ensure_chunk_loaded(self, chunk_idx: int):
        """Ensure a specific chunk is in memory."""
        if chunk_idx not in self.cache_chunks:
            # Evict oldest chunk if at capacity
            if len(self.cache_chunks) * self.chunk_size >= self.max_cache_size:
                oldest = min(self.cache_chunks.keys())
                self.evict_chunk(oldest)

            # Load the chunk
            conn = duckdb.connect(self.db_path, read_only=True,
                                 config={'temp_directory': '/tmp'})
            self._load_chunk(conn, chunk_idx)
            conn.close()


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
        db_path="/home/aharon/projects/new_swt/data/micro_features.duckdb"
    )

    logger.info(f"Cache initialized with {len(cache.cache_chunks)} chunks")
    logger.info(f"Memory usage: {cache._estimate_memory_usage():.1f} MB")

    # Test data retrieval
    test_start = 50000
    test_end = test_start + 360

    start_time = time.time()
    data = cache.get_session_data(test_start, test_end)
    elapsed = time.time() - start_time

    logger.info(f"Retrieved session data in {elapsed:.3f}s")
    logger.info(f"Data shape: {data.shape}")