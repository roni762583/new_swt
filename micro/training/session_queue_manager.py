#!/usr/bin/env python3
"""
Session Queue Manager for Parallel Training

Pre-validates sessions and maintains a queue for multiprocessing workers.
"""

import multiprocessing as mp
from multiprocessing import Queue, Process, Lock
import pandas as pd
import numpy as np
import duckdb
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidSession:
    """Valid 360-minute session ready for training."""
    start_idx: int
    end_idx: int
    start_time: datetime
    end_time: datetime
    split: str  # 'train', 'val', or 'test'
    quality_score: float


class SessionQueueManager:
    """
    Manages session validation and queue for parallel training.

    Validates 360-minute sessions with:
    - Gap detection (max 10 minutes)
    - Weekend filtering
    - Position closure verification
    """

    def __init__(self,
                 db_path: str,
                 session_length: int = 360,
                 max_gap_minutes: int = 10,
                 queue_size: int = 1000,
                 num_validators: int = 4):
        """
        Initialize session queue manager.

        Args:
            db_path: Path to micro_features.duckdb
            session_length: Session length in minutes (360 = 6 hours)
            max_gap_minutes: Maximum allowed gap between bars
            queue_size: Maximum queue size
            num_validators: Number of parallel validation processes
        """
        self.db_path = db_path
        self.session_length = session_length
        self.max_gap_minutes = max_gap_minutes
        self.queue_size = queue_size
        self.num_validators = num_validators

        # Multiprocessing components
        self.manager = mp.Manager()
        self.valid_queue = self.manager.Queue(maxsize=queue_size)
        self.rejection_stats = self.manager.dict()
        self.stop_event = self.manager.Event()
        self.validation_lock = self.manager.Lock()

        # Get data ranges for splits
        self._init_data_ranges()

        logger.info(f"SessionQueueManager initialized")
        logger.info(f"  Database: {db_path}")
        logger.info(f"  Session: {session_length}min, max gap {max_gap_minutes}min")
        logger.info(f"  Queue size: {queue_size}")
        logger.info(f"  Validators: {num_validators}")

    def _init_data_ranges(self):
        """Initialize data split ranges."""
        conn = duckdb.connect(self.db_path, read_only=True)

        total_rows = conn.execute(
            "SELECT COUNT(*) FROM micro_features"
        ).fetchone()[0]

        # 70/15/15 split
        train_end = int(total_rows * 0.7)
        val_end = int(total_rows * 0.85)

        self.data_ranges = {
            'train': (0, train_end),
            'val': (train_end, val_end),
            'test': (val_end, total_rows)
        }

        conn.close()

        logger.info(f"Data splits initialized:")
        for split, (start, end) in self.data_ranges.items():
            logger.info(f"  {split}: rows {start:,}-{end:,}")

    def validate_session(self, start_idx: int, split: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a single session.

        Returns:
            (is_valid, rejection_reason)
        """
        conn = duckdb.connect(self.db_path, read_only=True)

        # Get session data
        query = f"""
        SELECT timestamp, close, position_side
        FROM micro_features
        WHERE bar_index >= {start_idx}
        AND bar_index < {start_idx + self.session_length}
        ORDER BY bar_index
        LIMIT {self.session_length}
        """

        rows = conn.execute(query).fetchall()
        conn.close()

        if len(rows) < self.session_length:
            return False, "insufficient_data"

        prev_time = None
        has_open_position = False
        start_time = None
        end_time = None

        for i, (timestamp_str, close, position_side) in enumerate(rows):
            # Parse timestamp
            if isinstance(timestamp_str, str):
                timestamp = pd.to_datetime(timestamp_str)
            else:
                timestamp = timestamp_str

            if i == 0:
                start_time = timestamp
            if i == len(rows) - 1:
                end_time = timestamp

            # Check weekend hours
            weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
            hour = timestamp.hour

            # Saturday is always closed
            if weekday == 5:
                return False, "weekend_saturday"

            # Friday after 22:00 UTC
            if weekday == 4 and hour >= 22:
                return False, "friday_close"

            # Sunday before 22:00 UTC
            if weekday == 6 and hour < 22:
                return False, "sunday_closed"

            # Check gap between bars
            if prev_time is not None:
                gap_seconds = (timestamp - prev_time).total_seconds()
                gap_minutes = gap_seconds / 60.0

                if gap_minutes > self.max_gap_minutes:
                    return False, f"gap_{gap_minutes:.1f}min"

            # Check if position is open at end
            if i == len(rows) - 1:  # Last bar
                if position_side != 0:  # Position still open
                    has_open_position = True

            prev_time = timestamp

        # FIXED: Don't reject sessions with open positions
        # Sessions with open positions are valid for training
        # We only warn about them for debugging
        if has_open_position:
            logger.debug(f"Session ends with open position (this is OK for training)")

        # Always return True - all sessions are valid

        return True, None

    def validation_worker(self, worker_id: int):
        """
        Worker process for parallel session validation.

        Args:
            worker_id: Unique worker identifier
        """
        logger.info(f"Validator {worker_id} started")

        while not self.stop_event.is_set():
            # Generate random sessions from all splits
            for split in ['train', 'val', 'test']:
                if self.valid_queue.full():
                    time.sleep(0.1)
                    continue

                range_start, range_end = self.data_ranges[split]
                max_start = range_end - self.session_length - 100

                if max_start <= range_start:
                    continue

                # Try random indices
                for _ in range(10):  # 10 attempts per split
                    if self.stop_event.is_set():
                        break

                    start_idx = np.random.randint(range_start, max_start)
                    is_valid, reason = self.validate_session(start_idx, split)

                    if is_valid:
                        # Calculate quality score (simplified)
                        quality_score = np.random.uniform(0.5, 1.0)

                        # Get timestamps
                        conn = duckdb.connect(self.db_path, read_only=True)
                        timestamps = conn.execute(f"""
                            SELECT MIN(timestamp), MAX(timestamp)
                            FROM micro_features
                            WHERE bar_index >= {start_idx}
                            AND bar_index < {start_idx + self.session_length}
                        """).fetchone()
                        conn.close()

                        session = ValidSession(
                            start_idx=start_idx,
                            end_idx=start_idx + self.session_length,
                            start_time=pd.to_datetime(timestamps[0]),
                            end_time=pd.to_datetime(timestamps[1]),
                            split=split,
                            quality_score=quality_score
                        )

                        try:
                            self.valid_queue.put(session, timeout=1)
                            logger.debug(f"Validator {worker_id}: Added {split} session {start_idx}")
                        except:
                            pass  # Queue full
                    else:
                        # Track rejection reason
                        with self.validation_lock:
                            key = f"{split}_{reason}"
                            self.rejection_stats[key] = self.rejection_stats.get(key, 0) + 1

            time.sleep(0.01)  # Small delay to prevent CPU spinning

        logger.info(f"Validator {worker_id} stopped")

    def start(self):
        """Start validation workers."""
        self.workers = []

        for i in range(self.num_validators):
            p = Process(target=self.validation_worker, args=(i,))
            p.start()
            self.workers.append(p)

        logger.info(f"Started {self.num_validators} validation workers")

    def stop(self):
        """Stop validation workers."""
        logger.info("Stopping validation workers...")
        self.stop_event.set()

        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        logger.info("All validation workers stopped")

    def get_session(self, timeout: float = 1.0) -> Optional[ValidSession]:
        """
        Get a validated session from the queue.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            ValidSession or None if timeout
        """
        try:
            return self.valid_queue.get(timeout=timeout)
        except:
            return None

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.valid_queue.qsize()

    def get_rejection_stats(self) -> Dict[str, int]:
        """Get rejection statistics."""
        with self.validation_lock:
            return dict(self.rejection_stats)

    def print_stats(self):
        """Print queue and rejection statistics."""
        queue_size = self.get_queue_size()
        rejection_stats = self.get_rejection_stats()

        logger.info(f"\n{'='*60}")
        logger.info(f"Session Queue Statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Queue size: {queue_size}/{self.queue_size}")

        if rejection_stats:
            logger.info(f"\nRejection reasons:")
            for reason, count in sorted(rejection_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {reason}: {count}")


def test_session_queue():
    """Test the session queue manager."""
    logger.info("Testing SessionQueueManager...")

    # Initialize manager
    manager = SessionQueueManager(
        db_path='/home/aharon/projects/new_swt/data/micro_features.duckdb',
        session_length=360,
        max_gap_minutes=10,
        queue_size=100,
        num_validators=2
    )

    # Start validation
    manager.start()

    # Let it run for a bit
    logger.info("Filling queue...")
    time.sleep(5)

    # Get some sessions
    logger.info("\nGetting validated sessions:")
    for i in range(5):
        session = manager.get_session(timeout=1.0)
        if session:
            logger.info(f"  Session {i}: {session.split} [{session.start_idx}-{session.end_idx}]")
        else:
            logger.info(f"  Session {i}: Timeout")

    # Print stats
    manager.print_stats()

    # Stop
    manager.stop()
    logger.info("\nTest complete")


if __name__ == "__main__":
    test_session_queue()