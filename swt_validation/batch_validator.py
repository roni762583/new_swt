#!/usr/bin/env python3
"""
Batch Processing Validator with Optimized Memory Management
Processes multiple checkpoints efficiently with parallel inference
"""

import gc
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config
from swt_core.types import SWTAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_memory_gb: float = 4.0  # Maximum memory per process
    inference_batch_size: int = 256


class BatchDataLoader:
    """
    Efficient batch data loader with prefetching and memory management
    """

    def __init__(self, data_path: str, config: BatchConfig):
        """Initialize batch data loader"""
        self.config = config
        self.data_path = data_path

        # Load data info without loading full dataset
        self.data_info = self._get_data_info()
        self.total_samples = self.data_info['total_samples']

        logger.info(f"📊 BatchDataLoader initialized: {self.total_samples} samples")
        logger.info(f"   Batch size: {config.batch_size}, Workers: {config.num_workers}")

    def _get_data_info(self) -> Dict[str, Any]:
        """Get dataset info without loading all data"""
        # Read just header and count rows efficiently
        with open(self.data_path, 'r') as f:
            header = f.readline()
            row_count = sum(1 for _ in f)

        return {
            'total_samples': row_count,
            'columns': header.strip().split(','),
            'file_size_mb': Path(self.data_path).stat().st_size / (1024 * 1024)
        }

    def iter_batches(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """
        Iterate over data in efficient batches
        Yields preprocessed batches ready for inference
        """
        if end_idx is None:
            end_idx = self.total_samples

        # Use chunked reading for memory efficiency
        chunk_size = self.config.batch_size * 10  # Read 10 batches at a time

        for chunk_start in range(start_idx, end_idx, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_idx)

            # Read chunk from CSV efficiently
            chunk_df = pd.read_csv(
                self.data_path,
                skiprows=chunk_start + 1,  # +1 for header
                nrows=chunk_end - chunk_start,
                header=None
            )

            # Process chunk into batches
            for batch_start in range(0, len(chunk_df), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(chunk_df))
                batch_data = chunk_df.iloc[batch_start:batch_end]

                # Convert to tensors (placeholder - adapt to your feature extraction)
                batch_tensor = self._prepare_batch(batch_data)

                yield batch_tensor

            # Clean up chunk
            del chunk_df
            gc.collect()

    def _prepare_batch(self, batch_data: pd.DataFrame) -> torch.Tensor:
        """Prepare batch for inference (placeholder - implement your preprocessing)"""
        # This should match your actual feature extraction
        # For now, returning random features of correct dimension
        batch_size = len(batch_data)
        return torch.randn(batch_size, 137, dtype=torch.float32)


class BatchValidator:
    """
    Optimized batch validator with parallel processing
    """

    def __init__(self, config: BatchConfig = None):
        """Initialize batch validator"""
        self.config = config or BatchConfig()
        self.device = torch.device(self.config.device)

        # Setup multiprocessing
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)

        logger.info(f"🚀 BatchValidator initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Workers: {self.config.num_workers}")

    def validate_checkpoint(
        self,
        checkpoint_path: str,
        data_loader: BatchDataLoader,
        num_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Validate a single checkpoint with batch processing

        Args:
            checkpoint_path: Path to checkpoint
            data_loader: Batch data loader
            num_samples: Number of samples to validate

        Returns:
            Validation results
        """
        results = {}
        checkpoint_name = Path(checkpoint_path).stem

        logger.info(f"📦 Loading checkpoint: {checkpoint_name}")

        try:
            # Load checkpoint
            start_time = time.time()
            checkpoint_data = load_checkpoint_with_proper_config(checkpoint_path)
            load_time = time.time() - start_time

            networks = checkpoint_data['networks']
            networks.eval()

            if self.config.device == 'cuda':
                networks = networks.cuda()

            results['checkpoint'] = checkpoint_name
            results['load_time'] = load_time

            # Run batch inference
            logger.info(f"⚡ Running batch inference on {num_samples} samples...")
            inference_results = self._run_batch_inference(
                networks,
                data_loader,
                num_samples
            )
            results['inference'] = inference_results

            # Calculate metrics
            logger.info("📊 Calculating validation metrics...")
            metrics = self._calculate_metrics(inference_results)
            results['metrics'] = metrics

            # Clean up
            del checkpoint_data
            del networks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results['status'] = 'success'

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)

        return results

    @torch.no_grad()
    def _run_batch_inference(
        self,
        networks: torch.nn.Module,
        data_loader: BatchDataLoader,
        num_samples: int
    ) -> Dict[str, Any]:
        """
        Run efficient batch inference
        """
        predictions = []
        inference_times = []
        samples_processed = 0

        # Process in batches
        for batch in data_loader.iter_batches(end_idx=num_samples):
            if samples_processed >= num_samples:
                break

            # Move batch to device
            batch = batch.to(self.device, non_blocking=True)

            # Time inference
            start_time = time.perf_counter()

            # Forward pass
            hidden = networks.representation_network(batch)
            latent = networks.chance_encoder(batch)
            policy_logits = networks.policy_network(hidden, latent)
            value = networks.value_network(hidden, latent)

            # Get predictions
            actions = torch.argmax(policy_logits, dim=-1)

            inference_time = (time.perf_counter() - start_time) * 1000  # ms

            # Store results
            predictions.extend(actions.cpu().numpy().tolist())
            inference_times.append(inference_time / len(batch))  # ms per sample

            samples_processed += len(batch)

            # Log progress
            if samples_processed % 1000 == 0:
                logger.info(f"   Processed {samples_processed}/{num_samples} samples")

        # Calculate statistics
        inference_times_flat = np.array(inference_times)

        return {
            'num_samples': samples_processed,
            'predictions': predictions[:num_samples],  # Trim to exact count
            'timing': {
                'mean_ms': float(np.mean(inference_times_flat)),
                'median_ms': float(np.median(inference_times_flat)),
                'p95_ms': float(np.percentile(inference_times_flat, 95)),
                'p99_ms': float(np.percentile(inference_times_flat, 99)),
                'throughput_per_sec': float(1000 / np.mean(inference_times_flat))
            }
        }

    def _calculate_metrics(self, inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate validation metrics from predictions"""
        predictions = np.array(inference_results['predictions'])

        # Calculate action distribution
        action_counts = np.bincount(predictions, minlength=4)
        action_probs = action_counts / len(predictions)

        # Calculate trading metrics
        buy_ratio = action_probs[SWTAction.BUY.value]
        sell_ratio = action_probs[SWTAction.SELL.value]
        hold_ratio = action_probs[SWTAction.HOLD.value]
        close_ratio = action_probs[SWTAction.CLOSE.value]

        trade_frequency = buy_ratio + sell_ratio + close_ratio

        return {
            'action_distribution': {
                'hold': float(hold_ratio),
                'buy': float(buy_ratio),
                'sell': float(sell_ratio),
                'close': float(close_ratio)
            },
            'trade_frequency': float(trade_frequency),
            'position_bias': float(buy_ratio - sell_ratio),  # Positive = long bias
            'inference_throughput': inference_results['timing']['throughput_per_sec']
        }

    def validate_multiple_checkpoints(
        self,
        checkpoint_paths: List[str],
        data_path: str,
        samples_per_checkpoint: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple checkpoints in parallel

        Args:
            checkpoint_paths: List of checkpoint paths
            data_path: Path to validation data
            samples_per_checkpoint: Samples to validate per checkpoint

        Returns:
            List of validation results
        """
        logger.info(f"🎯 Validating {len(checkpoint_paths)} checkpoints")

        # Create shared data loader
        data_loader = BatchDataLoader(data_path, self.config)

        # Validate checkpoints in parallel using process pool
        with ProcessPoolExecutor(max_workers=2) as process_executor:
            # Create partial function with fixed arguments
            validate_fn = partial(
                self._validate_checkpoint_worker,
                data_path=data_path,
                config=self.config,
                samples=samples_per_checkpoint
            )

            # Run validation in parallel
            results = list(process_executor.map(validate_fn, checkpoint_paths))

        # Summarize results
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"✅ Validation complete: {successful}/{len(checkpoint_paths)} successful")

        return results

    @staticmethod
    def _validate_checkpoint_worker(
        checkpoint_path: str,
        data_path: str,
        config: BatchConfig,
        samples: int
    ) -> Dict[str, Any]:
        """Worker function for parallel checkpoint validation"""
        # Create fresh validator and data loader in worker process
        validator = BatchValidator(config)
        data_loader = BatchDataLoader(data_path, config)

        return validator.validate_checkpoint(
            checkpoint_path,
            data_loader,
            samples
        )

    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main entry point for batch validation"""
    import argparse

    parser = argparse.ArgumentParser(description='Batch checkpoint validation')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='Checkpoint paths')
    parser.add_argument('--data', default='data/GBPJPY_M1_3.5years_20250912.csv', help='Data path')
    parser.add_argument('--samples', type=int, default=10000, help='Samples per checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Configure batch processing
    config = BatchConfig(
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=args.device
    )

    # Create validator
    validator = BatchValidator(config)

    # Validate checkpoints
    if len(args.checkpoints) == 1:
        # Single checkpoint
        data_loader = BatchDataLoader(args.data, config)
        results = validator.validate_checkpoint(
            args.checkpoints[0],
            data_loader,
            args.samples
        )

        # Print results
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)

        if results['status'] == 'success':
            print(f"Checkpoint: {results['checkpoint']}")
            print(f"\n⚡ Inference Performance:")
            timing = results['inference']['timing']
            print(f"  Mean: {timing['mean_ms']:.2f}ms")
            print(f"  P95: {timing['p95_ms']:.2f}ms")
            print(f"  Throughput: {timing['throughput_per_sec']:.1f} samples/sec")

            print(f"\n📊 Action Distribution:")
            dist = results['metrics']['action_distribution']
            print(f"  Hold: {dist['hold']*100:.1f}%")
            print(f"  Buy: {dist['buy']*100:.1f}%")
            print(f"  Sell: {dist['sell']*100:.1f}%")
            print(f"  Close: {dist['close']*100:.1f}%")

            print(f"\n📈 Trading Metrics:")
            print(f"  Trade Frequency: {results['metrics']['trade_frequency']*100:.1f}%")
            print(f"  Position Bias: {results['metrics']['position_bias']*100:+.1f}%")
        else:
            print(f"❌ Validation failed: {results.get('error', 'Unknown error')}")
    else:
        # Multiple checkpoints
        results = validator.validate_multiple_checkpoints(
            args.checkpoints,
            args.data,
            args.samples
        )

        # Print summary table
        print("\n" + "="*80)
        print("BATCH VALIDATION SUMMARY")
        print("="*80)
        print(f"{'Checkpoint':<40} {'Status':<10} {'Throughput':<15} {'Trade Freq':<10}")
        print("-"*80)

        for result in results:
            if result['status'] == 'success':
                throughput = result['inference']['timing']['throughput_per_sec']
                trade_freq = result['metrics']['trade_frequency'] * 100
                print(f"{result['checkpoint'][:40]:<40} {'✅ OK':<10} {throughput:>8.1f} /sec   {trade_freq:>6.1f}%")
            else:
                print(f"{result['checkpoint'][:40]:<40} {'❌ FAIL':<10} {'N/A':<15} {'N/A':<10}")

    # Cleanup
    validator.cleanup()
    print("="*80)


if __name__ == "__main__":
    main()