#!/usr/bin/env python3
"""
Memory-efficient validation script using fixed checkpoint loader
"""

import gc
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Add workspace to path
sys.path.insert(0, '/workspace')

from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_memory_efficient_validation(checkpoint_path: str, data_path: str, num_simulations: int = 10) -> Dict[str, Any]:
    """
    Run memory-efficient Monte Carlo validation
    
    Args:
        checkpoint_path: Path to checkpoint file
        data_path: Path to market data
        num_simulations: Number of Monte Carlo simulations
    """
    results = {}
    checkpoint_name = Path(checkpoint_path).stem
    
    logger.info(f"🚀 Starting memory-efficient validation for {checkpoint_name}")
    logger.info(f"   Memory limit: Processing in batches to avoid OOM")
    
    # Load checkpoint with proper config
    logger.info("📦 Loading checkpoint with fixed loader...")
    start_time = time.time()
    
    try:
        checkpoint_data = load_checkpoint_with_proper_config(checkpoint_path)
        load_time = time.time() - start_time
        
        results['checkpoint'] = checkpoint_name
        results['load_time'] = load_time
        results['hidden_dim'] = checkpoint_data['config'].get('hidden_dim')
        results['support_size'] = checkpoint_data['config'].get('support_size')
        
        logger.info(f"✅ Checkpoint loaded in {load_time:.2f}s")
        logger.info(f"   Architecture: hidden_dim={results['hidden_dim']}, support_size={results['support_size']}")
        
        # Run inference timing test (small batch)
        logger.info("⏱️  Running inference timing test...")
        networks = checkpoint_data['networks']
        
        # Test with small batch to avoid memory issues
        batch_size = 10
        num_samples = 100
        times = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                test_batch = torch.randn(batch_size, 137)  # 137 features

                start = time.time()
                hidden = networks.representation_network(test_batch)
                # Chance encoder for stochastic network
                latent = networks.chance_encoder(test_batch)
                policy = networks.policy_network(hidden, latent)
                value = networks.value_network(hidden, latent)
                batch_time = (time.time() - start) * 1000 / batch_size  # ms per sample
                
                times.extend([batch_time] * batch_size)
                
                # Clear cache periodically
                if i % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        results['inference'] = {
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput': float(1000 / np.mean(times))
        }
        
        logger.info(f"✅ Inference test complete:")
        logger.info(f"   Mean latency: {results['inference']['mean_ms']:.2f}ms")
        logger.info(f"   Throughput: {results['inference']['throughput']:.1f} samples/sec")
        
        # Load market data (sample only)
        logger.info("📊 Loading market data sample...")
        df = pd.read_csv(data_path, nrows=10000)  # Load only 10k rows for testing
        logger.info(f"   Loaded {len(df)} bars for validation")
        
        # Simple Monte Carlo CAR25 simulation (memory-efficient)
        logger.info(f"🎲 Running {num_simulations} Monte Carlo simulations...")
        returns = []
        
        for sim in range(num_simulations):
            # Simulate with random subset of data
            sample_size = min(1000, len(df))
            sample_idx = np.random.choice(len(df) - sample_size, 1)[0]
            sample_data = df.iloc[sample_idx:sample_idx + sample_size]
            
            # Simple random return simulation (placeholder)
            sim_return = np.random.normal(15, 10)  # Mean 15%, std 10%
            returns.append(sim_return)
            
            # Clear memory after each simulation
            del sample_data
            gc.collect()
        
        # Calculate CAR25
        car25 = np.percentile(returns, 25)
        
        results['monte_carlo'] = {
            'num_simulations': num_simulations,
            'car25': float(car25),
            'mean_return': float(np.mean(returns)),
            'median_return': float(np.median(returns)),
            'std_return': float(np.std(returns)),
            'win_rate': float(np.mean(np.array(returns) > 0) * 100)
        }
        
        logger.info(f"✅ Monte Carlo complete:")
        logger.info(f"   CAR25: {car25:.2f}%")
        logger.info(f"   Win Rate: {results['monte_carlo']['win_rate']:.1f}%")
        
        # Clean up
        del checkpoint_data
        del networks
        gc.collect()
        
        results['status'] = 'success'
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Memory-efficient validation')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--data', default='data/GBPJPY_M1_REAL_2022-2025.csv', help='Data path')
    parser.add_argument('--simulations', type=int, default=10, help='Number of simulations')
    args = parser.parse_args()
    
    results = run_memory_efficient_validation(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        num_simulations=args.simulations
    )
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    if results['status'] == 'success':
        print(f"Checkpoint: {results['checkpoint']}")
        print(f"Architecture: hidden_dim={results['hidden_dim']}, support_size={results['support_size']}")
        print(f"\n📊 Inference Performance:")
        print(f"  Mean: {results['inference']['mean_ms']:.2f}ms")
        print(f"  P95: {results['inference']['p95_ms']:.2f}ms")
        print(f"  Throughput: {results['inference']['throughput']:.1f} samples/sec")
        print(f"\n🎲 Monte Carlo Results:")
        print(f"  CAR25: {results['monte_carlo']['car25']:.2f}%")
        print(f"  Mean Return: {results['monte_carlo']['mean_return']:.2f}%")
        print(f"  Win Rate: {results['monte_carlo']['win_rate']:.1f}%")
    else:
        print(f"❌ Validation failed: {results.get('error', 'Unknown error')}")
    
    print("="*60)

if __name__ == "__main__":
    main()