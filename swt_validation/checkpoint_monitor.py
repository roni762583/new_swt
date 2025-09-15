#!/usr/bin/env python3
"""
Checkpoint Monitor for Continuous Validation
Monitors checkpoint directory for new models and automatically validates them
"""

import os
import sys
import time
import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from swt_validation.monte_carlo_car25 import MonteCarloCAR25Validator
from swt_validation.automated_validator import AutomatedValidator
from swt_validation.pdf_report_generator import PDFReportGenerator
from swt_core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class CheckpointMonitor:
    """Monitor checkpoint directory and automatically validate new checkpoints"""

    def __init__(
        self,
        checkpoint_dir: str,
        output_dir: str,
        watch_interval: int = 30,
        auto_validate: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.watch_interval = watch_interval
        self.auto_validate = auto_validate

        # Track processed checkpoints
        self.processed_checkpoints: Set[str] = set()
        self.checkpoint_hashes: Dict[str, str] = {}
        self.validation_results: Dict[str, Dict] = {}

        # Load previous state if exists
        self.state_file = self.output_dir / "monitor_state.json"
        self._load_state()

        # Initialize validator
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config("config/validation.yaml")
        self.validator = AutomatedValidator(
            data_path="data/GBPJPY_M1_3.5years_20250912.csv",
            output_dir=str(self.output_dir)
        )

        logger.info(f"🔍 Checkpoint Monitor initialized")
        logger.info(f"   Watching: {self.checkpoint_dir}")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Interval: {self.watch_interval}s")

    def _load_state(self):
        """Load previous monitoring state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_checkpoints = set(state.get('processed', []))
                    self.checkpoint_hashes = state.get('hashes', {})
                    self.validation_results = state.get('results', {})
                logger.info(f"📂 Loaded state: {len(self.processed_checkpoints)} processed checkpoints")
            except Exception as e:
                logger.warning(f"⚠️ Could not load state: {e}")

    def _save_state(self):
        """Save monitoring state"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'processed': list(self.processed_checkpoints),
                'hashes': self.checkpoint_hashes,
                'results': self.validation_results,
                'last_update': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Could not save state: {e}")

    def _get_file_hash(self, filepath: Path) -> str:
        """Get hash of file for change detection"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _find_new_checkpoints(self) -> List[Path]:
        """Find new or modified checkpoints"""
        new_checkpoints = []

        if not self.checkpoint_dir.exists():
            logger.warning(f"⚠️ Checkpoint directory does not exist: {self.checkpoint_dir}")
            return new_checkpoints

        # Find all .pth files
        for checkpoint_path in self.checkpoint_dir.glob("*.pth"):
            checkpoint_name = checkpoint_path.name

            # Skip if already processed
            if checkpoint_name in self.processed_checkpoints:
                # Check if file has changed
                current_hash = self._get_file_hash(checkpoint_path)
                if self.checkpoint_hashes.get(checkpoint_name) != current_hash:
                    logger.info(f"🔄 Checkpoint modified: {checkpoint_name}")
                    new_checkpoints.append(checkpoint_path)
                    self.checkpoint_hashes[checkpoint_name] = current_hash
            else:
                # New checkpoint found
                logger.info(f"🆕 New checkpoint found: {checkpoint_name}")
                new_checkpoints.append(checkpoint_path)
                self.checkpoint_hashes[checkpoint_name] = self._get_file_hash(checkpoint_path)

        return new_checkpoints

    def _validate_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Validate a single checkpoint"""
        logger.info(f"🔍 Validating checkpoint: {checkpoint_path.name}")

        result = {
            'checkpoint': checkpoint_path.name,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }

        try:
            # Quick validation first
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            episode = checkpoint.get('episode', 'unknown')

            logger.info(f"   Episode: {episode}")
            logger.info(f"   Networks: {list(checkpoint.get('networks', {}).keys())}")

            # Run Monte Carlo validation
            if self.auto_validate:
                logger.info("   Running Monte Carlo validation (10 runs)...")
                validator = MonteCarloCAR25Validator(
                    checkpoint_path=str(checkpoint_path),
                    data_path="data/GBPJPY_M1_3.5years_20250912.csv",
                    num_runs=10,  # Quick validation with 10 runs
                    output_dir=str(self.output_dir)
                )
                mc_results = validator.run_monte_carlo_validation()

                result.update({
                    'status': 'completed',
                    'episode': episode,
                    'avg_return': mc_results.get('avg_return', 0),
                    'win_rate': mc_results.get('win_rate', 0),
                    'sharpe_ratio': mc_results.get('sharpe_ratio', 0),
                    'max_drawdown': mc_results.get('max_drawdown', 0)
                })

                # Generate PDF report
                try:
                    pdf_generator = PDFReportGenerator(output_dir=str(self.output_dir))
                    pdf_path = pdf_generator.generate_validation_report(
                        checkpoint_name=checkpoint_path.stem,
                        validation_results=mc_results
                    )
                    if pdf_path:
                        logger.info(f"   📄 PDF report generated: {pdf_path}")
                        result['pdf_report'] = pdf_path
                except Exception as pdf_e:
                    logger.warning(f"   ⚠️ PDF generation failed: {pdf_e}")

                # Mark as best if better than previous
                if self._is_best_checkpoint(result):
                    self._update_best_checkpoint(checkpoint_path, result)

                logger.info(f"   ✅ Validation complete: Return={result['avg_return']:.2f}%, Win Rate={result['win_rate']:.2f}%")
            else:
                result['status'] = 'skipped'
                logger.info("   ⏭️ Auto-validation disabled")

        except Exception as e:
            logger.error(f"   ❌ Validation failed: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    def _is_best_checkpoint(self, result: Dict) -> bool:
        """Check if this is the best checkpoint so far"""
        best_file = self.output_dir / "best_checkpoint.json"

        if not best_file.exists():
            return True

        try:
            with open(best_file, 'r') as f:
                best = json.load(f)
                # Compare by Sharpe ratio primarily
                return result.get('sharpe_ratio', 0) > best.get('sharpe_ratio', 0)
        except:
            return True

    def _update_best_checkpoint(self, checkpoint_path: Path, result: Dict):
        """Update the best checkpoint symlink and metadata"""
        logger.info(f"🏆 New best checkpoint: {checkpoint_path.name}")

        # Save best checkpoint metadata
        best_file = self.output_dir / "best_checkpoint.json"
        with open(best_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Create symlink to best checkpoint
        best_link = self.checkpoint_dir / "best_validated.pth"
        if best_link.exists():
            best_link.unlink()
        best_link.symlink_to(checkpoint_path)

        # Also copy to shared volume if configured
        shared_dir = Path("/shared/checkpoints")
        if shared_dir.exists():
            shared_best = shared_dir / "best_validated.pth"
            if shared_best.exists():
                shared_best.unlink()
            shared_best.symlink_to(checkpoint_path)

    def run(self):
        """Main monitoring loop"""
        logger.info("🚀 Starting checkpoint monitoring...")

        try:
            while True:
                # Find new checkpoints
                new_checkpoints = self._find_new_checkpoints()

                if new_checkpoints:
                    logger.info(f"📦 Found {len(new_checkpoints)} new/modified checkpoints")

                    for checkpoint_path in new_checkpoints:
                        # Validate checkpoint
                        result = self._validate_checkpoint(checkpoint_path)

                        # Store results
                        self.validation_results[checkpoint_path.name] = result
                        self.processed_checkpoints.add(checkpoint_path.name)

                        # Save results to file
                        result_file = self.output_dir / f"validation_{checkpoint_path.stem}.json"
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2)

                        # Save state after each validation
                        self._save_state()

                # Sleep before next check
                time.sleep(self.watch_interval)

        except KeyboardInterrupt:
            logger.info("⏹️ Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
            raise
        finally:
            self._save_state()
            logger.info("💾 State saved")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Checkpoint Monitor for Continuous Validation")
    parser.add_argument("--checkpoint-dir", default="/shared/checkpoints",
                       help="Directory to monitor for checkpoints")
    parser.add_argument("--output-dir", default="validation_results",
                       help="Directory for validation results")
    parser.add_argument("--watch-interval", type=int, default=30,
                       help="Check interval in seconds")
    parser.add_argument("--auto-validate", action="store_true", default=True,
                       help="Automatically validate new checkpoints")
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run monitor
    monitor = CheckpointMonitor(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        watch_interval=args.watch_interval,
        auto_validate=args.auto_validate
    )

    monitor.run()

if __name__ == "__main__":
    main()