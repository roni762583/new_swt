#!/usr/bin/env python3
"""
Enhanced Monte Carlo Validation with Stress Testing
Implements Dr. Bandy's full stress testing methodology:
- Random trade order shuffling
- Random trade dropping (10%)
- Random trade repetition
- Dropping last 20% of trades
- Multiple confidence levels (CAR25, CAR50, CAR75)
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import argparse
from datetime import datetime, timedelta
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    num_runs: int = 1000
    drop_rate: float = 0.1  # Drop 10% of trades randomly
    repeat_rate: float = 0.1  # Repeat 10% of trades randomly
    drop_last_pct: float = 0.2  # Drop last 20% of trades
    shuffle_trades: bool = True  # Randomize trade order
    bootstrap_samples: int = 100  # Bootstrap resampling count

    # Confidence levels for CAR calculation
    car_percentiles: List[int] = None

    def __post_init__(self):
        if self.car_percentiles is None:
            self.car_percentiles = [5, 10, 25, 50, 75, 90, 95]

@dataclass
class TradeResult:
    """Individual trade result"""
    entry_time: datetime
    exit_time: datetime
    pnl_pips: float
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    position_size: float
    hold_time: timedelta

class MonteCarloStressValidator:
    """Enhanced Monte Carlo validator with comprehensive stress testing"""

    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()
        self.original_trades: List[TradeResult] = []
        self.stress_results = []

    def extract_trades_from_backtest(self, backtest_results: pd.DataFrame) -> List[TradeResult]:
        """Extract individual trades from backtest results"""
        trades = []

        position_open = False
        entry_data = {}

        for idx, row in backtest_results.iterrows():
            if row['action'] in ['buy', 'sell'] and not position_open:
                # Position opened
                entry_data = {
                    'entry_time': row['timestamp'],
                    'entry_price': row['price'],
                    'direction': 'long' if row['action'] == 'buy' else 'short',
                    'position_size': row.get('position_size', 1.0)
                }
                position_open = True

            elif row['action'] == 'close' and position_open:
                # Position closed
                trade = TradeResult(
                    entry_time=entry_data['entry_time'],
                    exit_time=row['timestamp'],
                    pnl_pips=row['pnl'],
                    direction=entry_data['direction'],
                    entry_price=entry_data['entry_price'],
                    exit_price=row['price'],
                    position_size=entry_data['position_size'],
                    hold_time=row['timestamp'] - entry_data['entry_time']
                )
                trades.append(trade)
                position_open = False

        return trades

    def stress_test_trades(self, trades: List[TradeResult], run_id: int) -> Dict[str, Any]:
        """Apply stress testing to trades"""
        np.random.seed(run_id)  # Reproducible randomness
        random.seed(run_id)

        stressed_trades = trades.copy()

        # 1. Shuffle trade order (maintains causality within each trade)
        if self.config.shuffle_trades:
            random.shuffle(stressed_trades)

        # 2. Randomly drop trades
        num_to_drop = int(len(stressed_trades) * self.config.drop_rate)
        if num_to_drop > 0:
            indices_to_drop = random.sample(range(len(stressed_trades)), num_to_drop)
            stressed_trades = [t for i, t in enumerate(stressed_trades) if i not in indices_to_drop]

        # 3. Randomly repeat trades (simulate lucky streaks)
        num_to_repeat = int(len(trades) * self.config.repeat_rate)
        if num_to_repeat > 0:
            trades_to_repeat = random.sample(stressed_trades, min(num_to_repeat, len(stressed_trades)))
            stressed_trades.extend(trades_to_repeat)

        # 4. Drop last X% of trades (test early stopping)
        if self.config.drop_last_pct > 0:
            cutoff_idx = int(len(stressed_trades) * (1 - self.config.drop_last_pct))
            trades_without_last = stressed_trades[:cutoff_idx]
        else:
            trades_without_last = stressed_trades

        # Calculate metrics for different stress scenarios
        results = {
            'run_id': run_id,
            'original_count': len(trades),
            'stressed_count': len(stressed_trades),
            'without_last_count': len(trades_without_last),

            # Full stressed results
            'total_pnl': sum(t.pnl_pips for t in stressed_trades),
            'num_wins': sum(1 for t in stressed_trades if t.pnl_pips > 0),
            'num_losses': sum(1 for t in stressed_trades if t.pnl_pips < 0),
            'win_rate': sum(1 for t in stressed_trades if t.pnl_pips > 0) / len(stressed_trades) * 100 if stressed_trades else 0,
            'avg_win': np.mean([t.pnl_pips for t in stressed_trades if t.pnl_pips > 0]) if any(t.pnl_pips > 0 for t in stressed_trades) else 0,
            'avg_loss': np.mean([t.pnl_pips for t in stressed_trades if t.pnl_pips < 0]) if any(t.pnl_pips < 0 for t in stressed_trades) else 0,

            # Without last 20% results
            'total_pnl_without_last': sum(t.pnl_pips for t in trades_without_last) if trades_without_last else 0,
            'win_rate_without_last': sum(1 for t in trades_without_last if t.pnl_pips > 0) / len(trades_without_last) * 100 if trades_without_last else 0,
        }

        # Calculate drawdown
        results['max_drawdown'] = self._calculate_max_drawdown(stressed_trades)
        results['max_drawdown_without_last'] = self._calculate_max_drawdown(trades_without_last)

        # Calculate Sharpe ratio
        if stressed_trades:
            returns = [t.pnl_pips for t in stressed_trades]
            results['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            results['sharpe_ratio'] = 0

        # Calculate annual return estimate (assuming 252 trading days)
        if stressed_trades:
            avg_trades_per_day = len(stressed_trades) / 252  # Rough estimate
            daily_pnl = results['total_pnl'] / 252
            results['annual_return_pips'] = daily_pnl * 252
            results['annual_return_pct'] = (results['annual_return_pips'] / 10000) * 100  # Assuming 10000 pip account
        else:
            results['annual_return_pips'] = 0
            results['annual_return_pct'] = 0

        return results

    def _calculate_max_drawdown(self, trades: List[TradeResult]) -> float:
        """Calculate maximum drawdown from trades"""
        if not trades:
            return 0

        cumulative_pnl = []
        running_total = 0

        for trade in trades:
            running_total += trade.pnl_pips
            cumulative_pnl.append(running_total)

        # Calculate drawdown
        peak = cumulative_pnl[0]
        max_dd = 0

        for value in cumulative_pnl:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def bootstrap_confidence_intervals(self, trades: List[TradeResult]) -> Dict[str, Any]:
        """Calculate confidence intervals using bootstrap resampling"""
        bootstrap_results = []

        for i in range(self.config.bootstrap_samples):
            # Resample with replacement
            resampled = random.choices(trades, k=len(trades))

            total_pnl = sum(t.pnl_pips for t in resampled)
            win_rate = sum(1 for t in resampled if t.pnl_pips > 0) / len(resampled) * 100

            # Estimate annual return
            annual_return = (total_pnl / len(trades)) * 252 * 10  # Rough estimate

            bootstrap_results.append({
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'annual_return': annual_return
            })

        # Calculate confidence intervals
        pnl_values = [r['total_pnl'] for r in bootstrap_results]
        wr_values = [r['win_rate'] for r in bootstrap_results]
        ar_values = [r['annual_return'] for r in bootstrap_results]

        return {
            'pnl_ci_95': (np.percentile(pnl_values, 2.5), np.percentile(pnl_values, 97.5)),
            'pnl_ci_68': (np.percentile(pnl_values, 16), np.percentile(pnl_values, 84)),
            'win_rate_ci_95': (np.percentile(wr_values, 2.5), np.percentile(wr_values, 97.5)),
            'annual_return_ci_95': (np.percentile(ar_values, 2.5), np.percentile(ar_values, 97.5)),
        }

    def run_monte_carlo_stress_test(self, trades: List[TradeResult]) -> Dict[str, Any]:
        """Run full Monte Carlo stress test"""
        logger.info(f"🎲 Running {self.config.num_runs} Monte Carlo stress test iterations...")

        self.original_trades = trades
        stress_results = []

        # Run stress tests in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.stress_test_trades, trades, i)
                for i in range(self.config.num_runs)
            ]

            for future in as_completed(futures):
                result = future.result()
                stress_results.append(result)

                if len(stress_results) % 100 == 0:
                    logger.info(f"  Completed {len(stress_results)}/{self.config.num_runs} iterations")

        self.stress_results = stress_results

        # Calculate CAR values at different percentiles
        annual_returns = [r['annual_return_pct'] for r in stress_results]
        car_results = {}

        for percentile in self.config.car_percentiles:
            car_value = np.percentile(annual_returns, percentile)
            car_results[f'CAR{percentile}'] = car_value

        # Calculate comprehensive statistics
        results = {
            'num_runs': self.config.num_runs,
            'stress_config': {
                'drop_rate': self.config.drop_rate,
                'repeat_rate': self.config.repeat_rate,
                'drop_last_pct': self.config.drop_last_pct,
                'shuffle_trades': self.config.shuffle_trades
            },

            # CAR values
            'car_values': car_results,
            'car25': car_results.get('CAR25', 0),  # Primary metric

            # Overall statistics
            'mean_annual_return': np.mean(annual_returns),
            'median_annual_return': np.median(annual_returns),
            'std_annual_return': np.std(annual_returns),

            # Win rate statistics
            'mean_win_rate': np.mean([r['win_rate'] for r in stress_results]),
            'std_win_rate': np.std([r['win_rate'] for r in stress_results]),

            # Drawdown statistics
            'mean_max_drawdown': np.mean([r['max_drawdown'] for r in stress_results]),
            'worst_drawdown': np.max([r['max_drawdown'] for r in stress_results]),

            # Sharpe ratio statistics
            'mean_sharpe': np.mean([r['sharpe_ratio'] for r in stress_results]),
            'median_sharpe': np.median([r['sharpe_ratio'] for r in stress_results]),

            # Without last 20% statistics (early stopping test)
            'mean_return_without_last': np.mean([r['total_pnl_without_last'] for r in stress_results]),
            'mean_win_rate_without_last': np.mean([r['win_rate_without_last'] for r in stress_results]),

            # Risk metrics
            'probability_positive': sum(1 for r in annual_returns if r > 0) / len(annual_returns) * 100,
            'probability_double_digit': sum(1 for r in annual_returns if r > 10) / len(annual_returns) * 100,

            # Bootstrap confidence intervals
            'confidence_intervals': self.bootstrap_confidence_intervals(trades)
        }

        # Add robustness score (0-100)
        results['robustness_score'] = self._calculate_robustness_score(results)

        return results

    def _calculate_robustness_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall robustness score (0-100)"""
        score = 0

        # CAR25 > 10% = 25 points
        if results['car25'] > 10:
            score += 25
        elif results['car25'] > 5:
            score += 15
        elif results['car25'] > 0:
            score += 5

        # Probability of positive return > 80% = 25 points
        if results['probability_positive'] > 80:
            score += 25
        elif results['probability_positive'] > 60:
            score += 15
        elif results['probability_positive'] > 40:
            score += 5

        # Mean Sharpe > 1.5 = 25 points
        if results['mean_sharpe'] > 1.5:
            score += 25
        elif results['mean_sharpe'] > 1.0:
            score += 15
        elif results['mean_sharpe'] > 0.5:
            score += 5

        # Low drawdown volatility = 25 points
        dd_coefficient_of_variation = results['mean_max_drawdown'] / results['worst_drawdown'] if results['worst_drawdown'] > 0 else 0
        if dd_coefficient_of_variation < 0.5:
            score += 25
        elif dd_coefficient_of_variation < 0.75:
            score += 15
        elif dd_coefficient_of_variation < 1.0:
            score += 5

        return min(100, score)

    def generate_report(self, results: Dict[str, Any], output_path: Optional[Path] = None) -> str:
        """Generate comprehensive stress test report"""
        report = []
        report.append("=" * 60)
        report.append("MONTE CARLO STRESS TEST VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of runs: {results['num_runs']}")
        report.append("")

        # Stress test configuration
        report.append("STRESS TEST CONFIGURATION:")
        report.append(f"  • Trade order shuffling: {'Yes' if results['stress_config']['shuffle_trades'] else 'No'}")
        report.append(f"  • Random trade dropping: {results['stress_config']['drop_rate']*100:.0f}%")
        report.append(f"  • Random trade repetition: {results['stress_config']['repeat_rate']*100:.0f}%")
        report.append(f"  • Drop last trades: {results['stress_config']['drop_last_pct']*100:.0f}%")
        report.append("")

        # CAR Analysis
        report.append("COMPOUND ANNUAL RETURN (CAR) ANALYSIS:")
        for percentile, value in results['car_values'].items():
            stars = "⭐" if percentile == 'CAR25' else ""
            report.append(f"  • {percentile}: {value:.2f}% {stars}")
        report.append("")

        # Key metrics
        report.append("KEY PERFORMANCE METRICS:")
        report.append(f"  📊 Robustness Score: {results['robustness_score']:.0f}/100")
        report.append(f"  📈 Mean Annual Return: {results['mean_annual_return']:.2f}%")
        report.append(f"  📉 Worst Drawdown: {results['worst_drawdown']:.2f} pips")
        report.append(f"  ⚖️ Mean Sharpe Ratio: {results['mean_sharpe']:.2f}")
        report.append(f"  🎯 Mean Win Rate: {results['mean_win_rate']:.1f}%")
        report.append("")

        # Risk assessment
        report.append("RISK ASSESSMENT:")
        report.append(f"  • Probability of Profit: {results['probability_positive']:.1f}%")
        report.append(f"  • Probability >10% Return: {results['probability_double_digit']:.1f}%")

        ci = results['confidence_intervals']
        report.append(f"  • 95% CI Annual Return: [{ci['annual_return_ci_95'][0]:.1f}, {ci['annual_return_ci_95'][1]:.1f}]")
        report.append("")

        # Early stopping test
        report.append("EARLY STOPPING TEST (without last 20%):")
        report.append(f"  • Mean Return: {results['mean_return_without_last']:.2f} pips")
        report.append(f"  • Mean Win Rate: {results['mean_win_rate_without_last']:.1f}%")
        report.append("")

        # Interpretation
        report.append("INTERPRETATION:")
        if results['robustness_score'] >= 75:
            report.append("  ✅ EXCELLENT: Strategy shows strong robustness to stress testing")
        elif results['robustness_score'] >= 50:
            report.append("  ⚠️ GOOD: Strategy is reasonably robust but has some vulnerabilities")
        elif results['robustness_score'] >= 25:
            report.append("  ⚠️ FAIR: Strategy shows moderate robustness, consider improvements")
        else:
            report.append("  ❌ POOR: Strategy fails robustness testing, not recommended for live trading")

        report.append("")
        report.append("=" * 60)

        report_text = "\n".join(report)

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

            # Also save JSON results
            json_path = output_path.parent / f"{output_path.stem}_data.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

        return report_text

def main():
    """Main entry point for stress testing"""
    parser = argparse.ArgumentParser(description='Monte Carlo Stress Test Validation')
    parser.add_argument('--trades-file', required=True, help='Path to trades CSV or backtest results')
    parser.add_argument('--num-runs', type=int, default=1000, help='Number of Monte Carlo runs')
    parser.add_argument('--drop-rate', type=float, default=0.1, help='Random trade drop rate')
    parser.add_argument('--repeat-rate', type=float, default=0.1, help='Random trade repeat rate')
    parser.add_argument('--drop-last', type=float, default=0.2, help='Drop last X% of trades')
    parser.add_argument('--output', help='Output report path')
    args = parser.parse_args()

    # Load trades
    trades_df = pd.read_csv(args.trades_file)

    # Configure stress test
    config = StressTestConfig(
        num_runs=args.num_runs,
        drop_rate=args.drop_rate,
        repeat_rate=args.repeat_rate,
        drop_last_pct=args.drop_last,
        shuffle_trades=True
    )

    # Run validation
    validator = MonteCarloStressValidator(config)

    # Extract trades (adapt based on your format)
    trades = validator.extract_trades_from_backtest(trades_df)

    # Run stress test
    results = validator.run_monte_carlo_stress_test(trades)

    # Generate report
    output_path = Path(args.output) if args.output else None
    report = validator.generate_report(results, output_path)

    print(report)

    # Return code based on robustness
    return 0 if results['robustness_score'] >= 50 else 1

if __name__ == "__main__":
    exit(main())