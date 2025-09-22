#!/usr/bin/env python3
"""
Simple in-memory cache that preloads all data for faster episode collection.
"""

import numpy as np
import pandas as pd
import duckdb
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleMemoryCache:
    """Simple cache that loads entire database into memory at startup."""

    def __init__(
        self,
        db_path: str = "/workspace/data/micro_features.duckdb",
        session_length: int = 360,
        lookback: int = 32
    ):
        self.db_path = db_path
        self.session_length = session_length
        self.lookback = lookback
        self.data = None
        self._load_all_data()

    def _load_all_data(self):
        """Load entire micro_features table into memory."""
        logger.info("Loading entire database into memory...")
        start_time = time.time()

        # Use read_only mode and disable temp directory
        conn = duckdb.connect(self.db_path, read_only=True, config={'temp_directory': '/tmp'})

        # Load all data at once
        query = """
        SELECT *
        FROM micro_features
        ORDER BY bar_index
        """

        self.data = conn.execute(query).fetchdf()
        conn.close()

        elapsed = time.time() - start_time
        memory_mb = self.data.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Loaded {len(self.data)} rows in {elapsed:.1f}s")
        logger.info(f"Memory usage: {memory_mb:.1f} MB")

        # Create index for fast lookups
        self.data.set_index('bar_index', inplace=True)

    def get_session_data(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Get session data from cache."""
        # Include lookback period
        lookback_start = max(0, start_idx - self.lookback)

        # Use iloc for fast integer position-based indexing
        # Convert index positions to iloc positions
        try:
            start_pos = self.data.index.get_loc(lookback_start)
            end_pos = self.data.index.get_loc(end_idx)
            return self.data.iloc[start_pos:end_pos+1].copy()
        except KeyError:
            # Fallback to range query
            return self.data.loc[lookback_start:end_idx].copy()


# Global cache instance
_cache = None


def get_simple_cache() -> SimpleMemoryCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = SimpleMemoryCache()
    return _cache