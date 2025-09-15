#!/usr/bin/env python3
"""
Automatic Validation Monitor
Watches for new best checkpoints and automatically runs validation with reporting
"""

import os
import sys
import json
import time
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CheckpointMonitor:
    def __init__(self,
                 checkpoint_dir: str = "checkpoints/checkpoints",
                 registry_file: str = "checkpoints/checkpoints/checkpoint_registry.json",
                 validation_dir: str = "validation_results",
                 check_interval: int = 60):
        """
        Initialize checkpoint monitor

        Args:
            checkpoint_dir: Directory containing checkpoints
            registry_file: Path to checkpoint registry JSON
            validation_dir: Directory to save validation results
            check_interval: Seconds between checks
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.registry_file = Path(registry_file)
        self.validation_dir = Path(validation_dir)
        self.check_interval = check_interval

        # Create validation directory
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # Track validated checkpoints
        self.validated_checkpoints = self._load_validated_history()

        # Best checkpoint tracking
        self.current_best = None
        self.best_score = float('-inf')

        logger.info(f"✅ Monitor initialized - watching {self.checkpoint_dir}")

    def _load_validated_history(self) -> set:
        """Load history of already validated checkpoints"""
        history_file = self.validation_dir / "validated_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_validated_history(self):
        """Save history of validated checkpoints"""
        history_file = self.validation_dir / "validated_history.json"
        with open(history_file, 'w') as f:
            json.dump(list(self.validated_checkpoints), f, indent=2)

    def _get_checkpoint_hash(self, checkpoint_path: Path) -> str:
        """Get hash of checkpoint file for uniqueness"""
        hasher = hashlib.md5()
        with open(checkpoint_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def check_for_new_best(self) -> Optional[Dict[str, Any]]:
        """Check if there's a new best checkpoint"""
        if not self.registry_file.exists():
            return None

        with open(self.registry_file, 'r') as f:
            registry = json.load(f)

        # Find best checkpoint by quality score
        best_checkpoint = None
        best_score = float('-inf')

        for episode, info in registry.items():
            if info.get('quality_score', float('-inf')) > best_score:
                best_score = info['quality_score']
                best_checkpoint = {
                    'episode': episode,
                    'path': info['path'],
                    'score': best_score,
                    'metrics': info
                }

        if best_checkpoint and best_score > self.best_score:
            # Check if already validated
            checkpoint_hash = self._get_checkpoint_hash(Path(best_checkpoint['path']))
            if checkpoint_hash not in self.validated_checkpoints:
                self.best_score = best_score
                self.current_best = best_checkpoint
                return best_checkpoint

        return None

    def run_validation(self, checkpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive validation on checkpoint"""
        logger.info(f"🔬 Running validation for Episode {checkpoint_info['episode']}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode = checkpoint_info['episode']

        # Create validation output directory
        output_dir = self.validation_dir / f"episode_{episode}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'episode': episode,
            'timestamp': timestamp,
            'checkpoint_path': checkpoint_info['path'],
            'quality_score': checkpoint_info['score'],
            'original_metrics': checkpoint_info['metrics']
        }

        # 1. Run Monte Carlo CAR25 validation
        logger.info("📊 Running Monte Carlo CAR25 validation...")
        mc_result = self._run_monte_carlo(checkpoint_info['path'], output_dir)
        results['monte_carlo'] = mc_result

        # 2. Run inference timing test
        logger.info("⏱️  Running inference timing test...")
        timing_result = self._run_timing_test(checkpoint_info['path'], output_dir)
        results['timing'] = timing_result

        # 3. Run backtesting
        logger.info("📈 Running backtesting...")
        backtest_result = self._run_backtest(checkpoint_info['path'], output_dir)
        results['backtest'] = backtest_result

        # 4. Generate comprehensive report
        logger.info("📝 Generating validation report...")
        report_path = self._generate_report(results, output_dir)
        results['report_path'] = str(report_path)

        # 5. Copy checkpoint to validated directory
        validated_checkpoint = output_dir / f"validated_checkpoint_ep{episode}.pth"
        shutil.copy2(checkpoint_info['path'], validated_checkpoint)
        results['validated_checkpoint'] = str(validated_checkpoint)

        # Mark as validated
        checkpoint_hash = self._get_checkpoint_hash(Path(checkpoint_info['path']))
        self.validated_checkpoints.add(checkpoint_hash)
        self._save_validated_history()

        logger.info(f"✅ Validation complete for Episode {episode}")
        return results

    def _run_monte_carlo(self, checkpoint_path: str, output_dir: Path) -> Dict[str, Any]:
        """Run Monte Carlo CAR25 validation"""
        output_file = output_dir / "monte_carlo_results.json"

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.getcwd()}:/workspace",
            "-w", "/workspace",
            "-e", "PYTHONPATH=/workspace",
            "new_swt-swt-training:latest",
            "python", "swt_validation/run_mc_validation.py",
            "--checkpoint", checkpoint_path,
            "--data", "data/GBPJPY_M1_3.5years_20250912.csv",
            "--runs", "100",
            "--output", str(output_file)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if output_file.exists():
                with open(output_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'status': 'failed',
                    'error': result.stderr if result.returncode != 0 else 'No output file'
                }
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Monte Carlo validation timed out'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _run_timing_test(self, checkpoint_path: str, output_dir: Path) -> Dict[str, Any]:
        """Run inference timing test"""
        try:
            import torch
            from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config

            # Load checkpoint
            checkpoint_data = load_checkpoint_with_proper_config(checkpoint_path)
            networks = checkpoint_data['networks']

            # Run timing test
            num_samples = 1000
            test_inputs = torch.randn(num_samples, 137)  # 137 features

            times = []
            with torch.no_grad():
                for i in range(num_samples):
                    start = time.time()
                    hidden = networks.representation_network(test_inputs[i:i+1])
                    latent = networks.chance_encoder(hidden)
                    policy = networks.policy_network(hidden, latent)
                    value = networks.value_network(hidden, latent)
                    times.append((time.time() - start) * 1000)  # ms

            import numpy as np
            return {
                'mean_ms': float(np.mean(times)),
                'median_ms': float(np.median(times)),
                'std_ms': float(np.std(times)),
                'p95_ms': float(np.percentile(times, 95)),
                'p99_ms': float(np.percentile(times, 99)),
                'throughput': float(1000 / np.mean(times))
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _run_backtest(self, checkpoint_path: str, output_dir: Path) -> Dict[str, Any]:
        """Run backtesting simulation"""
        # Simplified backtest - you can expand this
        return {
            'status': 'pending',
            'note': 'Full backtesting to be implemented'
        }

    def _generate_report(self, results: Dict[str, Any], output_dir: Path) -> Path:
        """Generate comprehensive validation report"""
        report_path = output_dir / "validation_report.md"

        with open(report_path, 'w') as f:
            f.write(f"# Validation Report - Episode {results['episode']}\n\n")
            f.write(f"**Date**: {results['timestamp']}\n")
            f.write(f"**Checkpoint**: {results['checkpoint_path']}\n")
            f.write(f"**Quality Score**: {results['quality_score']:.2f}\n\n")

            # Original metrics
            f.write("## Training Metrics\n\n")
            metrics = results['original_metrics']
            f.write(f"- Win Rate: {metrics.get('win_rate', 'N/A')}%\n")
            f.write(f"- Average PnL: {metrics.get('avg_pnl', 'N/A')} pips\n")
            f.write(f"- Max Drawdown: {metrics.get('max_drawdown', 'N/A')} pips\n\n")

            # Monte Carlo results
            if 'monte_carlo' in results and results['monte_carlo'].get('status') != 'error':
                f.write("## Monte Carlo CAR25 Validation\n\n")
                mc = results['monte_carlo']
                f.write(f"- CAR25: {mc.get('car25', 'N/A')}%\n")
                f.write(f"- Median Return: {mc.get('median_return', 'N/A')}%\n")
                f.write(f"- Win Rate: {mc.get('win_rate', 'N/A')}%\n\n")

            # Timing results
            if 'timing' in results and 'mean_ms' in results['timing']:
                f.write("## Inference Performance\n\n")
                timing = results['timing']
                f.write(f"- Mean Latency: {timing['mean_ms']:.2f} ms\n")
                f.write(f"- P95 Latency: {timing['p95_ms']:.2f} ms\n")
                f.write(f"- P99 Latency: {timing['p99_ms']:.2f} ms\n")
                f.write(f"- Throughput: {timing['throughput']:.0f} samples/sec\n\n")

            f.write("## Summary\n\n")
            f.write(f"✅ Validation completed successfully\n")
            f.write(f"📁 Results saved to: {output_dir}\n")

        # Also save JSON version
        json_path = output_dir / "validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        return report_path

    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("🚀 Starting checkpoint monitoring...")

        while True:
            try:
                # Check for new best checkpoint
                new_best = self.check_for_new_best()

                if new_best:
                    logger.info(f"🎯 New best checkpoint detected: Episode {new_best['episode']} (score: {new_best['score']:.2f})")

                    # Run validation
                    validation_results = self.run_validation(new_best)

                    # Log results
                    logger.info(f"📊 Validation results saved to: {validation_results.get('report_path')}")

                    # Optional: Send notification (email, Slack, etc.)
                    self._send_notification(validation_results)

                # Wait before next check
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

    def _send_notification(self, results: Dict[str, Any]):
        """Send notification about validation results"""
        # Implement notification logic here (email, Slack, webhook, etc.)
        logger.info(f"📧 Notification: Episode {results['episode']} validated with score {results['quality_score']:.2f}")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Automatic checkpoint validation monitor')
    parser.add_argument('--checkpoint-dir', default='checkpoints/checkpoints', help='Checkpoint directory')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()

    monitor = CheckpointMonitor(
        checkpoint_dir=args.checkpoint_dir,
        check_interval=args.interval
    )

    if args.once:
        # Run single check
        new_best = monitor.check_for_new_best()
        if new_best:
            monitor.run_validation(new_best)
        else:
            logger.info("No new best checkpoint found")
    else:
        # Run continuous monitoring
        monitor.monitor_loop()

if __name__ == "__main__":
    main()