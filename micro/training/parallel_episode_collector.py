#!/usr/bin/env python3
"""
Parallel episode collector using multiprocessing for efficient data generation.
Workers run episodes in parallel and send experiences back to main process.
"""

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from queue import Empty
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path

from episode_runner import EpisodeRunner, Episode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CollectionJob:
    """Job for episode collection."""
    job_id: int
    split: str  # 'train', 'val', or 'test'
    session_idx: Optional[int]  # None for random
    temperature: float
    add_noise: bool


@dataclass
class CollectionResult:
    """Result from episode collection."""
    job_id: int
    worker_id: int
    episode: Episode
    collection_time: float
    success: bool
    error_message: Optional[str] = None


class EpisodeWorker(Process):
    """Worker process that collects episodes."""

    def __init__(
        self,
        worker_id: int,
        job_queue: Queue,
        result_queue: Queue,
        model_path: str,
        model_config: Dict,
        mcts_config: Dict,
        db_path: str,
        session_indices_path: str,
        device: str = 'cpu'
    ):
        super().__init__()
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.model_path = model_path
        self.model_config = model_config
        self.mcts_config = mcts_config
        self.db_path = db_path
        self.session_indices_path = session_indices_path
        self.device = device

        # These will be initialized in run()
        self.model = None
        self.mcts = None
        self.runner = None

    def run(self):
        """Main worker loop."""
        try:
            # Initialize in worker process (can't pickle CUDA models)
            self._initialize_worker()

            logger.info(f"Worker {self.worker_id} started on {self.device}")

            jobs_processed = 0
            total_time = 0

            while True:
                try:
                    # Get job (timeout after 5 seconds)
                    job = self.job_queue.get(timeout=5.0)

                    if job is None:  # Poison pill
                        break

                    # Collect episode
                    start_time = time.time()
                    result = self._collect_episode(job)
                    elapsed = time.time() - start_time

                    result.collection_time = elapsed
                    result.worker_id = self.worker_id

                    # Send result
                    self.result_queue.put(result)

                    jobs_processed += 1
                    total_time += elapsed

                    if jobs_processed % 10 == 0:
                        avg_time = total_time / jobs_processed
                        logger.info(
                            f"Worker {self.worker_id}: {jobs_processed} episodes, "
                            f"avg {avg_time:.1f}s/episode"
                        )

                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error: {e}")
                    if 'job' in locals():
                        error_result = CollectionResult(
                            job_id=job.job_id,
                            worker_id=self.worker_id,
                            episode=None,
                            collection_time=0.0,
                            success=False,
                            error_message=str(e)
                        )
                        self.result_queue.put(error_result)

        except Exception as e:
            logger.error(f"Worker {self.worker_id} initialization failed: {e}")
        finally:
            logger.info(f"Worker {self.worker_id} shutting down")

    def _initialize_worker(self):
        """Initialize model, MCTS, and episode runner."""
        # Import here to avoid pickling issues
        import sys
        sys.path.append('/workspace')

        from micro.models.micro_networks import MicroStochasticMuZero
        from micro.training.stochastic_mcts import StochasticMCTS

        # Create model
        self.model = MicroStochasticMuZero(**self.model_config)
        self.model.to(self.device)
        self.model.eval()

        # Load weights if path provided
        if Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state' in checkpoint:
                self.model.set_weights(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])

        # Create MCTS
        self.mcts = StochasticMCTS(
            model=self.model,
            **self.mcts_config
        )

        # Create episode runner with AMDDP10 rewards
        self.runner = EpisodeRunner(
            model=self.model,
            mcts=self.mcts,
            db_path=self.db_path,
            session_indices_path=self.session_indices_path,
            device=self.device,
            use_amddp10=True  # Use enhanced reward system
        )

    def _collect_episode(self, job: CollectionJob) -> CollectionResult:
        """Collect a single episode."""
        try:
            episode = self.runner.run_episode(
                split=job.split,
                session_idx=job.session_idx,
                temperature=job.temperature,
                add_noise=job.add_noise
            )

            return CollectionResult(
                job_id=job.job_id,
                worker_id=self.worker_id,
                episode=episode,
                collection_time=0.0,  # Will be set by caller
                success=True
            )

        except Exception as e:
            logger.error(f"Episode collection failed: {e}")
            return CollectionResult(
                job_id=job.job_id,
                worker_id=self.worker_id,
                episode=None,
                collection_time=0.0,
                success=False,
                error_message=str(e)
            )


class ParallelEpisodeCollector:
    """Manages parallel episode collection using multiple workers."""

    def __init__(
        self,
        model_path: str,
        model_config: Dict,
        mcts_config: Dict,
        num_workers: int = 4,
        db_path: str = "/workspace/data/micro_features.duckdb",
        session_indices_path: str = "/workspace/micro/cache/valid_session_indices.pkl",
        device_list: Optional[List[str]] = None
    ):
        self.model_path = model_path
        self.model_config = model_config
        self.mcts_config = mcts_config
        self.num_workers = num_workers
        self.db_path = db_path
        self.session_indices_path = session_indices_path

        # Device assignment
        if device_list is None:
            device_list = ['cpu'] * num_workers
        self.device_list = device_list

        # Queues
        self.job_queue = Queue(maxsize=num_workers * 2)
        self.result_queue = Queue()

        # Workers
        self.workers = []

        # Statistics
        self.stats = {
            'episodes_collected': 0,
            'total_experiences': 0,
            'total_trades': 0,
            'avg_expectancy': 0.0,
            'collection_times': []
        }

    def start(self):
        """Start worker processes."""
        logger.info(f"Starting {self.num_workers} workers...")

        for i in range(self.num_workers):
            device = self.device_list[i % len(self.device_list)]
            worker = EpisodeWorker(
                worker_id=i,
                job_queue=self.job_queue,
                result_queue=self.result_queue,
                model_path=self.model_path,
                model_config=self.model_config,
                mcts_config=self.mcts_config,
                db_path=self.db_path,
                session_indices_path=self.session_indices_path,
                device=device
            )
            worker.start()
            self.workers.append(worker)

        logger.info("All workers started")

    def stop(self):
        """Stop all workers."""
        logger.info("Stopping workers...")

        # Send poison pills
        for _ in range(self.num_workers):
            self.job_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Worker {worker.worker_id} did not stop gracefully")
                worker.terminate()

        self.workers.clear()
        logger.info("All workers stopped")

    def collect_episodes(
        self,
        num_episodes: int,
        split: str = 'train',
        temperature: float = 1.0,
        add_noise: bool = True,
        timeout: float = 300.0
    ) -> List[Episode]:
        """
        Collect multiple episodes in parallel.

        Args:
            num_episodes: Number of episodes to collect
            split: Data split to use
            temperature: MCTS temperature
            add_noise: Whether to add exploration noise
            timeout: Maximum time to wait for collection

        Returns:
            List of collected episodes
        """
        episodes = []
        jobs_submitted = 0
        results_received = 0

        start_time = time.time()

        # Submit jobs
        for i in range(num_episodes):
            job = CollectionJob(
                job_id=i,
                split=split,
                session_idx=None,  # Random selection
                temperature=temperature,
                add_noise=add_noise
            )
            self.job_queue.put(job)
            jobs_submitted += 1

        # Collect results
        while results_received < jobs_submitted:
            if time.time() - start_time > timeout:
                logger.warning(f"Collection timeout after {timeout}s")
                break

            try:
                result = self.result_queue.get(timeout=1.0)
                results_received += 1

                if result.success and result.episode is not None:
                    episodes.append(result.episode)
                    self._update_stats(result.episode, result.collection_time)

                    # Log progress
                    if len(episodes) % 10 == 0:
                        logger.info(
                            f"Collected {len(episodes)}/{num_episodes} episodes "
                            f"({results_received}/{jobs_submitted} jobs)"
                        )
                else:
                    logger.warning(
                        f"Job {result.job_id} failed: {result.error_message}"
                    )

            except Empty:
                continue

        elapsed = time.time() - start_time
        logger.info(
            f"Collected {len(episodes)} episodes in {elapsed:.1f}s "
            f"({elapsed/max(len(episodes), 1):.1f}s/episode)"
        )

        return episodes

    def _update_stats(self, episode: Episode, collection_time: float):
        """Update collection statistics."""
        self.stats['episodes_collected'] += 1
        self.stats['total_experiences'] += len(episode.experiences)
        self.stats['total_trades'] += episode.num_trades
        self.stats['collection_times'].append(collection_time)

        # Update running average expectancy
        n = self.stats['episodes_collected']
        prev_avg = self.stats['avg_expectancy']
        self.stats['avg_expectancy'] = (
            (prev_avg * (n - 1) + episode.expectancy) / n
        )

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        stats = self.stats.copy()
        if stats['collection_times']:
            stats['avg_collection_time'] = np.mean(stats['collection_times'])
        return stats


def test_parallel_collector():
    """Test the parallel collector setup."""
    model_config = {
        'input_features': 15,
        'lag_window': 32,
        'hidden_dim': 256,
        'action_dim': 4,
        'num_outcomes': 3,
        'support_size': 300
    }

    mcts_config = {
        'num_simulations': 25,
        'discount': 0.997,
        'depth_limit': 3,
        'dirichlet_alpha': 1.0,
        'exploration_fraction': 0.5
    }

    collector = ParallelEpisodeCollector(
        model_path="/workspace/micro/checkpoints/latest.pth",
        model_config=model_config,
        mcts_config=mcts_config,
        num_workers=4
    )

    logger.info("Parallel collector created successfully")
    logger.info("To test with actual collection:")
    logger.info("  1. Ensure model checkpoint exists")
    logger.info("  2. Run: collector.start()")
    logger.info("  3. Collect: episodes = collector.collect_episodes(100)")
    logger.info("  4. Stop: collector.stop()")


if __name__ == "__main__":
    test_parallel_collector()