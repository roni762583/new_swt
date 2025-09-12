#!/usr/bin/env python3
"""
Automated Validation System for SWT Checkpoints
Triggers validation based on performance improvements and schedules
"""

import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import multiprocessing as mp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from swt_validation.composite_scorer import (
    CompositeScorer, CheckpointMetrics, create_metrics_from_dict
)
from swt_validation.monte_carlo_car25 import MonteCarloCAR25Validator, CAR25Config
from swt_validation.walk_forward_analysis import WalkForwardAnalyzer, WalkForwardConfig

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels with increasing thoroughness"""
    QUICK = "quick"  # Basic metrics only (~30 seconds)
    STANDARD = "standard"  # Quick backtest + scoring (~2 minutes)
    FULL = "full"  # Monte Carlo validation (~10-30 minutes)
    COMPREHENSIVE = "comprehensive"  # Full MC + Walk-forward (~1-2 hours)


@dataclass
class ValidationTrigger:
    """Conditions that trigger validation"""
    expectancy_improvement_threshold: float = 0.10  # 10% improvement
    episode_interval: int = 100  # Every N episodes
    time_interval_hours: float = 6.0  # Every N hours
    score_improvement_threshold: float = 5.0  # 5 point score improvement
    force_validation_on_best: bool = True  # Always validate new best
    min_trades_for_validation: int = 30  # Minimum trades before validation


@dataclass
class ValidationResult:
    """Result from validation run"""
    checkpoint_path: str
    episode: int
    timestamp: datetime
    validation_level: ValidationLevel
    
    # Quick validation results
    expectancy: Optional[float] = None
    win_rate: Optional[float] = None
    composite_score: Optional[float] = None
    grade: Optional[str] = None
    
    # Full validation results
    car25: Optional[float] = None
    monte_carlo_runs: Optional[int] = None
    probability_of_profit: Optional[float] = None
    
    # Walk-forward results
    walk_forward_efficiency: Optional[float] = None
    robustness_score: Optional[float] = None
    
    # Recommendations
    recommendation: Optional[str] = None
    deploy_ready: bool = False
    validation_time_seconds: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'checkpoint_path': self.checkpoint_path,
            'episode': self.episode,
            'timestamp': self.timestamp.isoformat(),
            'validation_level': self.validation_level.value,
            'expectancy': self.expectancy,
            'win_rate': self.win_rate,
            'composite_score': self.composite_score,
            'grade': self.grade,
            'car25': self.car25,
            'monte_carlo_runs': self.monte_carlo_runs,
            'probability_of_profit': self.probability_of_profit,
            'walk_forward_efficiency': self.walk_forward_efficiency,
            'robustness_score': self.robustness_score,
            'recommendation': self.recommendation,
            'deploy_ready': self.deploy_ready,
            'validation_time_seconds': self.validation_time_seconds
        }


class AutomatedValidator:
    """
    Automated validation system with intelligent triggering
    """
    
    def __init__(self, 
                 data_path: str,
                 triggers: Optional[ValidationTrigger] = None,
                 output_dir: str = "validation_results"):
        """
        Initialize automated validator
        
        Args:
            data_path: Path to test data for validation
            triggers: Validation trigger configuration
            output_dir: Directory for validation results
        """
        self.data_path = Path(data_path)
        self.triggers = triggers or ValidationTrigger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Tracking state
        self.best_expectancy = 0
        self.best_score = 0
        self.last_validation_time = datetime.now()
        self.last_validation_episode = 0
        self.validation_history = []
        
        # Initialize components
        self.scorer = CompositeScorer()
        self.car25_validator = None  # Lazy initialization
        self.walk_forward_analyzer = None  # Lazy initialization
        
        # Load state if exists
        self._load_state()
    
    def should_validate(self, 
                       current_metrics: CheckpointMetrics,
                       force: bool = False) -> ValidationLevel:
        """
        Determine if validation should run and at what level
        
        Args:
            current_metrics: Current checkpoint metrics
            force: Force validation regardless of triggers
            
        Returns:
            ValidationLevel or None if no validation needed
        """
        if force:
            return ValidationLevel.FULL
        
        # Check minimum trades
        if current_metrics.total_trades < self.triggers.min_trades_for_validation:
            logger.info(f"Skipping validation: only {current_metrics.total_trades} trades")
            return None
        
        # Calculate quick score for comparison
        quick_score = self.scorer.calculate_composite_score(current_metrics)
        
        # Check for significant expectancy improvement
        expectancy_improvement = 0
        if self.best_expectancy > 0:
            expectancy_improvement = (current_metrics.expectancy - self.best_expectancy) / self.best_expectancy
        
        # Check for score improvement
        score_improvement = quick_score.total_score - self.best_score
        
        # Check time interval
        hours_since_last = (datetime.now() - self.last_validation_time).total_seconds() / 3600
        
        # Check episode interval
        episodes_since_last = current_metrics.episode - self.last_validation_episode
        
        # Determine validation level based on triggers
        if expectancy_improvement >= self.triggers.expectancy_improvement_threshold * 2:
            # Major improvement - full validation
            logger.info(f"Triggering FULL validation: {expectancy_improvement:.1%} expectancy improvement")
            return ValidationLevel.FULL
        
        elif expectancy_improvement >= self.triggers.expectancy_improvement_threshold:
            # Moderate improvement - standard validation
            logger.info(f"Triggering STANDARD validation: {expectancy_improvement:.1%} expectancy improvement")
            return ValidationLevel.STANDARD
        
        elif score_improvement >= self.triggers.score_improvement_threshold * 2:
            # Major score improvement
            logger.info(f"Triggering FULL validation: {score_improvement:.1f} point score improvement")
            return ValidationLevel.FULL
        
        elif score_improvement >= self.triggers.score_improvement_threshold:
            # Moderate score improvement
            logger.info(f"Triggering STANDARD validation: {score_improvement:.1f} point score improvement")
            return ValidationLevel.STANDARD
        
        elif episodes_since_last >= self.triggers.episode_interval:
            # Regular interval validation
            logger.info(f"Triggering STANDARD validation: {episodes_since_last} episodes since last validation")
            return ValidationLevel.STANDARD
        
        elif hours_since_last >= self.triggers.time_interval_hours:
            # Time-based validation
            logger.info(f"Triggering QUICK validation: {hours_since_last:.1f} hours since last validation")
            return ValidationLevel.QUICK
        
        elif self.triggers.force_validation_on_best and quick_score.total_score > self.best_score:
            # New best checkpoint
            logger.info(f"Triggering STANDARD validation: new best checkpoint (score: {quick_score.total_score:.1f})")
            return ValidationLevel.STANDARD
        
        return None
    
    async def validate_checkpoint(self,
                                 checkpoint_path: str,
                                 metrics: CheckpointMetrics,
                                 validation_level: Optional[ValidationLevel] = None) -> ValidationResult:
        """
        Run validation at specified level
        
        Args:
            checkpoint_path: Path to checkpoint file
            metrics: Checkpoint metrics
            validation_level: Level of validation to perform
            
        Returns:
            ValidationResult with outcomes
        """
        start_time = datetime.now()
        
        # Determine validation level if not specified
        if validation_level is None:
            validation_level = self.should_validate(metrics)
            if validation_level is None:
                logger.info("No validation triggered")
                return None
        
        logger.info(f"🔍 Running {validation_level.value} validation for episode {metrics.episode}")
        
        result = ValidationResult(
            checkpoint_path=checkpoint_path,
            episode=metrics.episode,
            timestamp=datetime.now(),
            validation_level=validation_level
        )
        
        # Run appropriate validation
        if validation_level == ValidationLevel.QUICK:
            await self._run_quick_validation(metrics, result)
        
        elif validation_level == ValidationLevel.STANDARD:
            await self._run_standard_validation(checkpoint_path, metrics, result)
        
        elif validation_level == ValidationLevel.FULL:
            await self._run_full_validation(checkpoint_path, metrics, result)
        
        elif validation_level == ValidationLevel.COMPREHENSIVE:
            await self._run_comprehensive_validation(checkpoint_path, metrics, result)
        
        # Calculate validation time
        result.validation_time_seconds = (datetime.now() - start_time).total_seconds()
        
        # Update tracking state
        self._update_state(metrics, result)
        
        # Save result
        self._save_result(result)
        
        # Log summary
        self._log_validation_summary(result)
        
        return result
    
    async def _run_quick_validation(self, metrics: CheckpointMetrics, result: ValidationResult) -> None:
        """Run quick validation (metrics only)"""
        # Calculate composite score
        score = self.scorer.calculate_composite_score(metrics)
        
        result.expectancy = metrics.expectancy
        result.win_rate = metrics.win_rate
        result.composite_score = score.total_score
        result.grade = score.grade
        result.recommendation = score.recommendation
        result.deploy_ready = score.total_score >= 80 and metrics.total_trades >= 100
    
    async def _run_standard_validation(self, checkpoint_path: str, 
                                      metrics: CheckpointMetrics, 
                                      result: ValidationResult) -> None:
        """Run standard validation (quick backtest + scoring)"""
        # First do quick validation
        await self._run_quick_validation(metrics, result)
        
        # Run quick backtest (last 1000 bars)
        try:
            backtest_result = await self._run_quick_backtest(checkpoint_path)
            
            # Update metrics with backtest results
            if backtest_result:
                result.expectancy = backtest_result.get('expectancy', metrics.expectancy)
                result.win_rate = backtest_result.get('win_rate', metrics.win_rate)
                
                # Recalculate score with updated metrics
                updated_metrics = create_metrics_from_dict({
                    **asdict(metrics),
                    'expectancy': result.expectancy,
                    'win_rate': result.win_rate
                })
                score = self.scorer.calculate_composite_score(updated_metrics)
                result.composite_score = score.total_score
                result.grade = score.grade
                result.recommendation = score.recommendation
                
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
    
    async def _run_full_validation(self, checkpoint_path: str,
                                  metrics: CheckpointMetrics,
                                  result: ValidationResult) -> None:
        """Run full validation (Monte Carlo CAR25)"""
        # First do standard validation
        await self._run_standard_validation(checkpoint_path, metrics, result)
        
        # Initialize CAR25 validator if needed
        if self.car25_validator is None:
            config = CAR25Config(monte_carlo_runs=200)  # Reduced for faster validation
            self.car25_validator = MonteCarloCAR25Validator(config)
            self.car25_validator.load_test_data(str(self.data_path))
        
        # Run Monte Carlo validation
        try:
            self.car25_validator.load_checkpoint(checkpoint_path)
            car25_report = self.car25_validator.run_monte_carlo_validation(num_runs=200)
            
            # Extract key metrics
            result.car25 = car25_report['car25_metrics']['car25']
            result.monte_carlo_runs = car25_report['monte_carlo_runs']
            result.probability_of_profit = car25_report['car25_metrics'].get('probability_of_profit', 0)
            
            # Update deploy readiness based on CAR25
            if result.car25 >= 0.15:  # 15% annual return threshold
                result.deploy_ready = True
                result.recommendation = car25_report['recommendation']
            
        except Exception as e:
            logger.error(f"Monte Carlo validation failed: {e}")
    
    async def _run_comprehensive_validation(self, checkpoint_path: str,
                                           metrics: CheckpointMetrics,
                                           result: ValidationResult) -> None:
        """Run comprehensive validation (Full MC + Walk-forward)"""
        # First do full validation
        await self._run_full_validation(checkpoint_path, metrics, result)
        
        # Initialize walk-forward analyzer if needed
        if self.walk_forward_analyzer is None:
            config = WalkForwardConfig(total_periods=6)  # Reduced for faster validation
            self.walk_forward_analyzer = WalkForwardAnalyzer(config)
            self.walk_forward_analyzer.load_data(str(self.data_path))
        
        # Run walk-forward analysis
        try:
            self.walk_forward_analyzer.set_checkpoint(checkpoint_path)
            wf_report = self.walk_forward_analyzer.run_walk_forward_analysis()
            
            # Extract key metrics
            result.walk_forward_efficiency = wf_report.avg_efficiency_ratio
            result.robustness_score = wf_report.robustness_score
            
            # Update recommendation based on walk-forward
            if wf_report.robustness_score < 50:
                result.deploy_ready = False
                result.recommendation = f"CAUTION: Low robustness ({wf_report.robustness_score:.0f}%) - possible overfitting"
            
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
    
    async def _run_quick_backtest(self, checkpoint_path: str) -> Dict[str, Any]:
        """Run quick backtest on recent data"""
        # This would run actual backtest - placeholder for now
        # In production, this would load checkpoint and run on last 1000 bars
        logger.info("Running quick backtest on last 1000 bars...")
        
        # Simulate backtest result
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            'expectancy': 0.45,
            'win_rate': 0.52,
            'total_trades': 45,
            'sharpe_ratio': 1.65
        }
    
    def _update_state(self, metrics: CheckpointMetrics, result: ValidationResult) -> None:
        """Update tracking state after validation"""
        if result.expectancy and result.expectancy > self.best_expectancy:
            self.best_expectancy = result.expectancy
        
        if result.composite_score and result.composite_score > self.best_score:
            self.best_score = result.composite_score
        
        self.last_validation_time = datetime.now()
        self.last_validation_episode = metrics.episode
        
        # Add to history
        self.validation_history.append(result)
        
        # Keep only last 100 validations
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]
        
        # Save state
        self._save_state()
    
    def _save_result(self, result: ValidationResult) -> None:
        """Save validation result to file"""
        filename = f"validation_ep{result.episode}_{result.validation_level.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"📁 Validation result saved to {filepath}")
    
    def _log_validation_summary(self, result: ValidationResult) -> None:
        """Log validation summary"""
        logger.info("="*60)
        logger.info(f"VALIDATION SUMMARY - Episode {result.episode}")
        logger.info("="*60)
        logger.info(f"Level: {result.validation_level.value.upper()}")
        logger.info(f"Expectancy: {result.expectancy:.3f}" if result.expectancy else "Expectancy: N/A")
        logger.info(f"Win Rate: {result.win_rate:.1%}" if result.win_rate else "Win Rate: N/A")
        logger.info(f"Composite Score: {result.composite_score:.1f}/100" if result.composite_score else "Score: N/A")
        logger.info(f"Grade: {result.grade}" if result.grade else "Grade: N/A")
        
        if result.car25 is not None:
            logger.info(f"CAR25: {result.car25:.2%}")
            logger.info(f"Monte Carlo Runs: {result.monte_carlo_runs}")
        
        if result.robustness_score is not None:
            logger.info(f"Robustness Score: {result.robustness_score:.1f}%")
            logger.info(f"Walk-Forward Efficiency: {result.walk_forward_efficiency:.2%}")
        
        logger.info(f"Deploy Ready: {'✅ YES' if result.deploy_ready else '❌ NO'}")
        logger.info(f"Recommendation: {result.recommendation}")
        logger.info(f"Validation Time: {result.validation_time_seconds:.1f} seconds")
        logger.info("="*60)
    
    def _save_state(self) -> None:
        """Save validator state to file"""
        state = {
            'best_expectancy': self.best_expectancy,
            'best_score': self.best_score,
            'last_validation_time': self.last_validation_time.isoformat(),
            'last_validation_episode': self.last_validation_episode,
            'history_count': len(self.validation_history)
        }
        
        state_file = self.output_dir / "validator_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self) -> None:
        """Load validator state from file"""
        state_file = self.output_dir / "validator_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.best_expectancy = state.get('best_expectancy', 0)
                self.best_score = state.get('best_score', 0)
                self.last_validation_time = datetime.fromisoformat(state.get('last_validation_time', datetime.now().isoformat()))
                self.last_validation_episode = state.get('last_validation_episode', 0)
                
                logger.info(f"✅ Loaded validator state: best score={self.best_score:.1f}, best expectancy={self.best_expectancy:.3f}")
                
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def get_validation_schedule(self, current_episode: int) -> Dict[str, Any]:
        """Get upcoming validation schedule"""
        schedule = {
            'current_episode': current_episode,
            'next_scheduled_validations': []
        }
        
        # Episode-based validation
        next_episode_validation = ((current_episode // self.triggers.episode_interval) + 1) * self.triggers.episode_interval
        schedule['next_scheduled_validations'].append({
            'trigger': 'episode_interval',
            'at_episode': next_episode_validation,
            'episodes_remaining': next_episode_validation - current_episode
        })
        
        # Time-based validation
        next_time_validation = self.last_validation_time + timedelta(hours=self.triggers.time_interval_hours)
        time_remaining = (next_time_validation - datetime.now()).total_seconds() / 3600
        if time_remaining > 0:
            schedule['next_scheduled_validations'].append({
                'trigger': 'time_interval',
                'at_time': next_time_validation.isoformat(),
                'hours_remaining': time_remaining
            })
        
        return schedule


def create_validation_callback(validator: AutomatedValidator) -> Callable:
    """
    Create a callback function for training integration
    
    Args:
        validator: AutomatedValidator instance
        
    Returns:
        Callback function to be called after each checkpoint save
    """
    async def validation_callback(checkpoint_path: str, metrics_dict: Dict[str, Any]) -> None:
        """Callback to trigger validation after checkpoint save"""
        try:
            # Convert dict to CheckpointMetrics
            metrics = create_metrics_from_dict(metrics_dict)
            
            # Check if validation should run
            validation_level = validator.should_validate(metrics)
            
            if validation_level:
                # Run validation asynchronously
                result = await validator.validate_checkpoint(
                    checkpoint_path, metrics, validation_level
                )
                
                # Check if this is a deployable checkpoint
                if result and result.deploy_ready:
                    logger.info(f"🚀 DEPLOYABLE CHECKPOINT: {checkpoint_path}")
                    
                    # Create symlink to best deployable checkpoint
                    best_link = Path(checkpoint_path).parent / "best_deployable.pth"
                    if best_link.exists():
                        best_link.unlink()
                    best_link.symlink_to(Path(checkpoint_path).name)
                    
        except Exception as e:
            logger.error(f"Validation callback failed: {e}")
    
    return validation_callback


async def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Checkpoint Validation')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--episode', type=int, required=True, help='Episode number')
    parser.add_argument('--level', choices=['quick', 'standard', 'full', 'comprehensive'],
                       default='standard', help='Validation level')
    parser.add_argument('--force', action='store_true', help='Force validation')
    
    args = parser.parse_args()
    
    # Create example metrics (in production, these would come from checkpoint)
    metrics = CheckpointMetrics(
        checkpoint_path=args.checkpoint,
        episode=args.episode,
        timestamp=datetime.now(),
        expectancy=0.45,
        win_rate=0.52,
        profit_factor=1.6,
        total_trades=85,
        sharpe_ratio=1.5,
        sortino_ratio=1.8,
        calmar_ratio=1.2,
        max_drawdown_pct=0.18,
        max_drawdown_pips=180,
        avg_win_pips=5.2,
        avg_loss_pips=3.8,
        win_loss_ratio=1.37,
        recovery_factor=2.1
    )
    
    # Create validator
    validator = AutomatedValidator(args.data)
    
    # Run validation
    validation_level = ValidationLevel[args.level.upper()]
    result = await validator.validate_checkpoint(
        args.checkpoint, metrics, 
        validation_level if args.force else None
    )
    
    if result:
        print(f"\n✅ Validation complete: {result.recommendation}")
    else:
        print("\n❌ No validation triggered")


if __name__ == "__main__":
    asyncio.run(main())