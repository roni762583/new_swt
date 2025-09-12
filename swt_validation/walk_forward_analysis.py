#!/usr/bin/env python3
"""
Walk-Forward Analysis Framework for New SWT
Implementation of Dr. Howard Bandy's walk-forward optimization and validation

Walk-forward analysis tests the robustness of a trading system by:
1. Training on in-sample data
2. Testing on out-of-sample data
3. Rolling forward through time
4. Measuring consistency of performance
"""

import sys
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from swt_validation.monte_carlo_car25 import MonteCarloCAR25Validator, CAR25Config, TradeResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    total_periods: int = 12  # Total walk-forward periods
    in_sample_months: int = 6  # Months for in-sample training
    out_sample_months: int = 2  # Months for out-of-sample testing
    optimization_metric: str = 'sharpe_ratio'  # Metric to optimize
    min_trades_per_period: int = 30  # Minimum trades for valid period
    anchored_mode: bool = False  # If True, in-sample always starts from beginning
    
    # Performance degradation thresholds
    max_acceptable_degradation: float = 0.30  # 30% performance drop out-of-sample
    min_efficiency_ratio: float = 0.50  # Out-sample must be at least 50% of in-sample
    

@dataclass 
class WalkForwardPeriod:
    """Single walk-forward period result"""
    period_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    
    # In-sample metrics
    in_sample_trades: int
    in_sample_return: float
    in_sample_sharpe: float
    in_sample_max_dd: float
    in_sample_win_rate: float
    in_sample_profit_factor: float
    
    # Out-of-sample metrics
    out_sample_trades: int
    out_sample_return: float
    out_sample_sharpe: float
    out_sample_max_dd: float
    out_sample_win_rate: float
    out_sample_profit_factor: float
    
    # Efficiency metrics
    return_efficiency: float  # Out-sample return / In-sample return
    sharpe_efficiency: float  # Out-sample Sharpe / In-sample Sharpe
    consistency_score: float  # Overall consistency metric
    

@dataclass
class WalkForwardReport:
    """Complete walk-forward analysis report"""
    analysis_timestamp: datetime
    checkpoint_path: str
    config: WalkForwardConfig
    periods: List[WalkForwardPeriod]
    
    # Aggregate metrics
    avg_in_sample_return: float
    avg_out_sample_return: float
    avg_efficiency_ratio: float
    consistency_score: float
    robustness_score: float
    
    # Statistical tests
    degradation_test_passed: bool
    consistency_test_passed: bool
    robustness_test_passed: bool
    
    # Recommendations
    recommendation: str
    confidence_level: str  # HIGH, MEDIUM, LOW
    

class WalkForwardAnalyzer:
    """
    Walk-forward analysis for trading system validation
    Based on Dr. Howard Bandy's methodology
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.test_data = None
        self.checkpoint_path = None
        self.validator = None
        self.periods = []
        
    def load_data(self, data_path: str) -> None:
        """Load and prepare data for walk-forward analysis"""
        logger.info(f"📊 Loading data for walk-forward analysis: {data_path}")
        
        self.test_data = pd.read_csv(data_path)
        self.test_data['timestamp'] = pd.to_datetime(self.test_data['timestamp'])
        self.test_data = self.test_data.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate available periods
        total_months = (self.test_data['timestamp'].max() - self.test_data['timestamp'].min()).days / 30
        required_months = self.config.in_sample_months + self.config.out_sample_months
        available_periods = int(total_months / self.config.out_sample_months)
        
        if available_periods < self.config.total_periods:
            logger.warning(f"⚠️ Only {available_periods} periods available, requested {self.config.total_periods}")
            self.config.total_periods = available_periods
            
        logger.info(f"✅ Data loaded: {len(self.test_data):,} bars, {available_periods} walk-forward periods")
        
    def set_checkpoint(self, checkpoint_path: str) -> None:
        """Set checkpoint for analysis"""
        self.checkpoint_path = checkpoint_path
        
        # Initialize validator
        car25_config = CAR25Config()
        self.validator = MonteCarloCAR25Validator(car25_config)
        self.validator.load_checkpoint(checkpoint_path)
        
    def run_walk_forward_analysis(self) -> WalkForwardReport:
        """
        Execute complete walk-forward analysis
        
        Returns:
            WalkForwardReport with detailed results
        """
        logger.info(f"🚀 Starting walk-forward analysis with {self.config.total_periods} periods")
        
        # Define walk-forward periods
        periods = self._define_periods()
        
        # Run analysis for each period
        period_results = []
        for i, (in_start, in_end, out_start, out_end) in enumerate(periods):
            logger.info(f"📈 Processing period {i+1}/{len(periods)}")
            
            period_result = self._analyze_period(
                period_id=i,
                in_sample_start=in_start,
                in_sample_end=in_end,
                out_sample_start=out_start,
                out_sample_end=out_end
            )
            
            period_results.append(period_result)
            
            # Log progress
            self._log_period_result(period_result)
            
        # Generate report
        report = self._generate_report(period_results)
        
        # Create visualizations
        self._create_visualizations(report)
        
        return report
        
    def _define_periods(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Define walk-forward period boundaries"""
        periods = []
        
        data_start = self.test_data['timestamp'].min()
        data_end = self.test_data['timestamp'].max()
        
        if self.config.anchored_mode:
            # Anchored walk-forward: in-sample always starts from beginning
            for i in range(self.config.total_periods):
                in_sample_start = data_start
                in_sample_end = data_start + pd.DateOffset(months=self.config.in_sample_months * (i + 1))
                out_sample_start = in_sample_end
                out_sample_end = out_sample_start + pd.DateOffset(months=self.config.out_sample_months)
                
                if out_sample_end > data_end:
                    break
                    
                periods.append((in_sample_start, in_sample_end, out_sample_start, out_sample_end))
        else:
            # Rolling walk-forward: fixed window size
            for i in range(self.config.total_periods):
                period_start = data_start + pd.DateOffset(months=self.config.out_sample_months * i)
                in_sample_start = period_start
                in_sample_end = in_sample_start + pd.DateOffset(months=self.config.in_sample_months)
                out_sample_start = in_sample_end
                out_sample_end = out_sample_start + pd.DateOffset(months=self.config.out_sample_months)
                
                if out_sample_end > data_end:
                    break
                    
                periods.append((in_sample_start, in_sample_end, out_sample_start, out_sample_end))
                
        return periods
        
    def _analyze_period(self, period_id: int, in_sample_start: datetime, 
                       in_sample_end: datetime, out_sample_start: datetime,
                       out_sample_end: datetime) -> WalkForwardPeriod:
        """Analyze single walk-forward period"""
        
        # Extract in-sample data
        in_sample_data = self.test_data[
            (self.test_data['timestamp'] >= in_sample_start) &
            (self.test_data['timestamp'] < in_sample_end)
        ].copy()
        
        # Extract out-of-sample data
        out_sample_data = self.test_data[
            (self.test_data['timestamp'] >= out_sample_start) &
            (self.test_data['timestamp'] < out_sample_end)
        ].copy()
        
        # Run trading simulation for in-sample
        in_sample_trades = self.validator._simulate_trading(in_sample_data)
        in_sample_metrics = self.validator._calculate_performance_metrics(in_sample_trades, in_sample_data)
        
        # Run trading simulation for out-of-sample
        out_sample_trades = self.validator._simulate_trading(out_sample_data)
        out_sample_metrics = self.validator._calculate_performance_metrics(out_sample_trades, out_sample_data)
        
        # Calculate efficiency ratios
        return_efficiency = self._safe_divide(
            out_sample_metrics['annualized_return'],
            in_sample_metrics['annualized_return']
        )
        
        sharpe_efficiency = self._safe_divide(
            out_sample_metrics['sharpe_ratio'],
            in_sample_metrics['sharpe_ratio']
        )
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(
            in_sample_metrics, out_sample_metrics
        )
        
        return WalkForwardPeriod(
            period_id=period_id,
            in_sample_start=in_sample_start,
            in_sample_end=in_sample_end,
            out_sample_start=out_sample_start,
            out_sample_end=out_sample_end,
            
            # In-sample metrics
            in_sample_trades=in_sample_metrics['total_trades'],
            in_sample_return=in_sample_metrics['annualized_return'],
            in_sample_sharpe=in_sample_metrics['sharpe_ratio'],
            in_sample_max_dd=in_sample_metrics['max_drawdown_pct'],
            in_sample_win_rate=in_sample_metrics['win_rate'],
            in_sample_profit_factor=in_sample_metrics['profit_factor'],
            
            # Out-of-sample metrics
            out_sample_trades=out_sample_metrics['total_trades'],
            out_sample_return=out_sample_metrics['annualized_return'],
            out_sample_sharpe=out_sample_metrics['sharpe_ratio'],
            out_sample_max_dd=out_sample_metrics['max_drawdown_pct'],
            out_sample_win_rate=out_sample_metrics['win_rate'],
            out_sample_profit_factor=out_sample_metrics['profit_factor'],
            
            # Efficiency metrics
            return_efficiency=return_efficiency,
            sharpe_efficiency=sharpe_efficiency,
            consistency_score=consistency_score
        )
        
    def _calculate_consistency_score(self, in_sample: Dict, out_sample: Dict) -> float:
        """Calculate consistency score between in-sample and out-of-sample"""
        
        # Compare key metrics
        metrics_to_compare = ['win_rate', 'profit_factor', 'sharpe_ratio']
        scores = []
        
        for metric in metrics_to_compare:
            in_val = in_sample.get(metric, 0)
            out_val = out_sample.get(metric, 0)
            
            if in_val != 0:
                ratio = min(out_val / in_val, in_val / out_val) if out_val != 0 else 0
                scores.append(ratio)
            else:
                scores.append(0)
                
        return np.mean(scores) * 100  # Convert to percentage
        
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safe division handling zero denominator"""
        if denominator == 0:
            return 0 if numerator == 0 else float('inf')
        return numerator / denominator
        
    def _generate_report(self, periods: List[WalkForwardPeriod]) -> WalkForwardReport:
        """Generate comprehensive walk-forward report"""
        
        # Calculate aggregate metrics
        avg_in_sample_return = np.mean([p.in_sample_return for p in periods])
        avg_out_sample_return = np.mean([p.out_sample_return for p in periods])
        avg_efficiency_ratio = np.mean([p.return_efficiency for p in periods])
        
        # Calculate consistency score
        consistency_scores = [p.consistency_score for p in periods]
        overall_consistency = np.mean(consistency_scores)
        
        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(periods)
        
        # Run statistical tests
        degradation_test = self._test_degradation(periods)
        consistency_test = self._test_consistency(periods)
        robustness_test = robustness_score >= 60  # 60% threshold
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            degradation_test, consistency_test, robustness_test, robustness_score
        )
        
        # Determine confidence level
        if degradation_test and consistency_test and robustness_test:
            confidence_level = "HIGH"
        elif (degradation_test and consistency_test) or (consistency_test and robustness_test):
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
            
        return WalkForwardReport(
            analysis_timestamp=datetime.now(),
            checkpoint_path=self.checkpoint_path,
            config=self.config,
            periods=periods,
            avg_in_sample_return=avg_in_sample_return,
            avg_out_sample_return=avg_out_sample_return,
            avg_efficiency_ratio=avg_efficiency_ratio,
            consistency_score=overall_consistency,
            robustness_score=robustness_score,
            degradation_test_passed=degradation_test,
            consistency_test_passed=consistency_test,
            robustness_test_passed=robustness_test,
            recommendation=recommendation,
            confidence_level=confidence_level
        )
        
    def _calculate_robustness_score(self, periods: List[WalkForwardPeriod]) -> float:
        """Calculate overall robustness score (0-100)"""
        scores = []
        
        for period in periods:
            score = 0
            
            # Positive out-of-sample return (30 points)
            if period.out_sample_return > 0:
                score += 30
                
            # Efficiency ratio (30 points)
            if period.return_efficiency >= 0.8:
                score += 30
            elif period.return_efficiency >= 0.6:
                score += 20
            elif period.return_efficiency >= 0.4:
                score += 10
                
            # Consistency (20 points)
            if period.consistency_score >= 80:
                score += 20
            elif period.consistency_score >= 60:
                score += 15
            elif period.consistency_score >= 40:
                score += 10
                
            # Low drawdown (20 points)
            if period.out_sample_max_dd <= 0.10:
                score += 20
            elif period.out_sample_max_dd <= 0.15:
                score += 15
            elif period.out_sample_max_dd <= 0.20:
                score += 10
                
            scores.append(score)
            
        return np.mean(scores)
        
    def _test_degradation(self, periods: List[WalkForwardPeriod]) -> bool:
        """Test if out-of-sample performance degradation is acceptable"""
        degradations = []
        
        for period in periods:
            if period.in_sample_return > 0:
                degradation = 1 - (period.out_sample_return / period.in_sample_return)
                degradations.append(degradation)
                
        if not degradations:
            return False
            
        avg_degradation = np.mean(degradations)
        return avg_degradation <= self.config.max_acceptable_degradation
        
    def _test_consistency(self, periods: List[WalkForwardPeriod]) -> bool:
        """Test if performance is consistent across periods"""
        # Check if majority of periods have positive out-of-sample returns
        positive_periods = sum(1 for p in periods if p.out_sample_return > 0)
        positive_ratio = positive_periods / len(periods)
        
        # Check if efficiency ratios are acceptable
        good_efficiency = sum(1 for p in periods if p.return_efficiency >= self.config.min_efficiency_ratio)
        efficiency_ratio = good_efficiency / len(periods)
        
        return positive_ratio >= 0.6 and efficiency_ratio >= 0.5
        
    def _generate_recommendation(self, degradation_test: bool, consistency_test: bool,
                                robustness_test: bool, robustness_score: float) -> str:
        """Generate recommendation based on test results"""
        
        if all([degradation_test, consistency_test, robustness_test]):
            if robustness_score >= 80:
                return "HIGHLY RECOMMENDED: Excellent walk-forward performance with strong consistency"
            else:
                return "RECOMMENDED: Good walk-forward performance with acceptable consistency"
        elif degradation_test and consistency_test:
            return "CAUTIOUSLY RECOMMENDED: Acceptable performance but monitor robustness"
        elif consistency_test:
            return "USE WITH CAUTION: Consistent but shows performance degradation"
        else:
            return "NOT RECOMMENDED: Poor walk-forward performance indicates overfitting"
            
    def _log_period_result(self, period: WalkForwardPeriod) -> None:
        """Log results for a single period"""
        logger.info(f"  Period {period.period_id + 1}:")
        logger.info(f"    In-sample: Return={period.in_sample_return:.1%}, Sharpe={period.in_sample_sharpe:.2f}")
        logger.info(f"    Out-sample: Return={period.out_sample_return:.1%}, Sharpe={period.out_sample_sharpe:.2f}")
        logger.info(f"    Efficiency: {period.return_efficiency:.1%}, Consistency: {period.consistency_score:.1f}%")
        
    def _create_visualizations(self, report: WalkForwardReport) -> None:
        """Create walk-forward analysis visualizations"""
        try:
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            periods = report.periods
            x = range(1, len(periods) + 1)
            
            # Plot 1: Returns comparison
            ax1 = axes[0, 0]
            ax1.plot(x, [p.in_sample_return for p in periods], 'b-', label='In-Sample', marker='o')
            ax1.plot(x, [p.out_sample_return for p in periods], 'r-', label='Out-of-Sample', marker='s')
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.set_xlabel('Period')
            ax1.set_ylabel('Annualized Return (%)')
            ax1.set_title('Walk-Forward Returns')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Efficiency Ratio
            ax2 = axes[0, 1]
            efficiency_ratios = [p.return_efficiency for p in periods]
            colors = ['green' if e >= 0.5 else 'orange' if e >= 0.3 else 'red' for e in efficiency_ratios]
            ax2.bar(x, efficiency_ratios, color=colors)
            ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Target (50%)')
            ax2.set_xlabel('Period')
            ax2.set_ylabel('Efficiency Ratio')
            ax2.set_title('Out-of-Sample Efficiency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Consistency Score
            ax3 = axes[1, 0]
            consistency_scores = [p.consistency_score for p in periods]
            ax3.plot(x, consistency_scores, 'g-', marker='D')
            ax3.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Acceptable (60%)')
            ax3.fill_between(x, consistency_scores, alpha=0.3)
            ax3.set_xlabel('Period')
            ax3.set_ylabel('Consistency Score (%)')
            ax3.set_title('Period Consistency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Drawdown Comparison
            ax4 = axes[1, 1]
            width = 0.35
            x_pos = np.arange(len(periods))
            ax4.bar(x_pos - width/2, [p.in_sample_max_dd for p in periods], width, label='In-Sample', color='blue', alpha=0.7)
            ax4.bar(x_pos + width/2, [p.out_sample_max_dd for p in periods], width, label='Out-of-Sample', color='red', alpha=0.7)
            ax4.set_xlabel('Period')
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.set_title('Maximum Drawdown Comparison')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([str(i+1) for i in range(len(periods))])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Walk-Forward Analysis - Robustness Score: {report.robustness_score:.1f}%', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            output_path = Path('walk_forward_analysis.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            logger.info(f"📊 Visualizations saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")


def main():
    """Main entry point for walk-forward analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Walk-Forward Analysis for SWT Trading System')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--data', required=True, help='Path to test data CSV')
    parser.add_argument('--periods', type=int, default=12, help='Number of walk-forward periods')
    parser.add_argument('--in-sample-months', type=int, default=6, help='Months for in-sample')
    parser.add_argument('--out-sample-months', type=int, default=2, help='Months for out-of-sample')
    parser.add_argument('--anchored', action='store_true', help='Use anchored walk-forward')
    parser.add_argument('--output', default='walk_forward_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    # Configure walk-forward
    config = WalkForwardConfig(
        total_periods=args.periods,
        in_sample_months=args.in_sample_months,
        out_sample_months=args.out_sample_months,
        anchored_mode=args.anchored
    )
    
    # Create analyzer
    analyzer = WalkForwardAnalyzer(config)
    
    # Load data and checkpoint
    analyzer.load_data(args.data)
    analyzer.set_checkpoint(args.checkpoint)
    
    # Run analysis
    logger.info("🚀 Starting walk-forward analysis...")
    report = analyzer.run_walk_forward_analysis()
    
    # Save report
    output_path = Path(args.output)
    report_dict = asdict(report)
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
        
    logger.info(f"📊 Analysis complete! Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS SUMMARY")
    print("="*60)
    print(f"Periods Analyzed: {len(report.periods)}")
    print(f"Avg In-Sample Return: {report.avg_in_sample_return:.2%}")
    print(f"Avg Out-Sample Return: {report.avg_out_sample_return:.2%}")
    print(f"Avg Efficiency Ratio: {report.avg_efficiency_ratio:.2%}")
    print(f"Consistency Score: {report.consistency_score:.1f}%")
    print(f"Robustness Score: {report.robustness_score:.1f}%")
    print(f"\nValidation Tests:")
    print(f"  Degradation Test: {'✅ PASSED' if report.degradation_test_passed else '❌ FAILED'}")
    print(f"  Consistency Test: {'✅ PASSED' if report.consistency_test_passed else '❌ FAILED'}")
    print(f"  Robustness Test: {'✅ PASSED' if report.robustness_test_passed else '❌ FAILED'}")
    print(f"\nConfidence Level: {report.confidence_level}")
    print(f"Recommendation: {report.recommendation}")
    print("="*60)
    

if __name__ == "__main__":
    main()