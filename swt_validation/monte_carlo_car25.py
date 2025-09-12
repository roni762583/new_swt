#!/usr/bin/env python3
"""
Monte Carlo CAR25 Validation Framework for New SWT
Based on Dr. Howard Bandy's methodology for robust trading system validation

CAR25 = Compound Annual Return at 25th percentile (conservative estimate)
Bootstrap validation with Monte Carlo simulation for statistical confidence
"""

import sys
import numpy as np
import pandas as pd
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import argparse
import warnings
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from swt_core.config_manager import ConfigManager
from swt_features.feature_processor import FeatureProcessor
# InferenceEngine not needed - using networks directly
from swt_inference.checkpoint_loader import CheckpointLoader
from swt_environments.swt_forex_env import SWTForexEnvironment, SWTAction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CAR25Config:
    """Configuration for CAR25 validation"""
    monte_carlo_runs: int = 1000  # Dr. Bandy recommends 1000+ runs
    bootstrap_sample_size: int = 252  # Trading days per year
    confidence_level: float = 0.25  # 25th percentile
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_position_size: int = 10000  # Units
    spread_cost_pips: float = 1.5
    commission_per_trade: float = 0.0
    slippage_pips: float = 0.5
    
    # Walk-forward parameters
    in_sample_ratio: float = 0.7  # 70% for training
    out_sample_ratio: float = 0.3  # 30% for testing
    walk_forward_windows: int = 10  # Number of walk-forward periods
    
    # Performance thresholds (Dr. Bandy's recommendations)
    min_acceptable_car25: float = 0.15  # 15% annual return
    max_acceptable_drawdown: float = 0.25  # 25% max drawdown
    min_profit_factor: float = 1.5
    min_win_rate: float = 0.40
    

@dataclass
class TradeResult:
    """Individual trade result"""
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_dollars: float
    position_size: int
    spread_cost: float
    slippage_cost: float
    

@dataclass
class MonteCarloResult:
    """Single Monte Carlo run result"""
    run_id: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_pips: float
    total_pnl_dollars: float
    avg_win_pips: float
    avg_loss_pips: float
    profit_factor: float
    max_drawdown_pips: float
    max_drawdown_pct: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float
    expectancy: float
    kelly_criterion: float
    trades_per_day: float
    execution_time_seconds: float
    

@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""
    window_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    in_sample_return: float
    out_sample_return: float
    in_sample_sharpe: float
    out_sample_sharpe: float
    in_sample_max_dd: float
    out_sample_max_dd: float
    efficiency_ratio: float  # Out-sample / In-sample performance
    

class MonteCarloCAR25Validator:
    """
    Monte Carlo validator implementing Dr. Howard Bandy's CAR25 methodology
    with walk-forward analysis and robustness testing
    """
    
    def __init__(self, config: CAR25Config):
        self.config = config  # CAR25 validation config
        self.swt_config = None  # SWT system config (loaded from checkpoint)
        self.test_data = None
        self.checkpoint_path = None
        self.networks = None
        self.feature_processor = None
        self.results_cache = {}
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint and initialize inference engine"""
        logger.info(f"💾 Loading checkpoint: {checkpoint_path}")
        
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(strict_validation=False)  # Allow non-Episode 13475
        
        # Add agent_system attribute for checkpoint loader compatibility
        from swt_core.types import AgentType
        config.agent_system = AgentType.STOCHASTIC_MUZERO
        
        # Initialize components
        self.feature_processor = FeatureProcessor(config)
        
        # Load checkpoint
        loader = CheckpointLoader(config)
        checkpoint_data = loader.load_checkpoint(checkpoint_path)
        
        # Store networks directly from checkpoint (it's a single network object)
        self.networks = checkpoint_data['networks']
        self.checkpoint_info = checkpoint_data.get('checkpoint_info', {})
        
        # Store SWT config for later use
        self.swt_config = config
        
        logger.info(f"✅ Checkpoint loaded successfully")
        
    def run_inference(self, market_features: np.ndarray, position_features: np.ndarray) -> Tuple[int, float]:
        """Run inference using loaded networks"""
        with torch.no_grad():
            # Convert to tensors
            market_tensor = torch.FloatTensor(market_features).unsqueeze(0)
            position_tensor = torch.FloatTensor(position_features).unsqueeze(0)
            
            # Get initial hidden state from representation network (only needs market features)
            hidden_state = self.networks.representation_network(market_tensor)
            
            # Generate latent variable for stochastic network
            latent_z = self.networks.chance_encoder(hidden_state)
            
            # Combine hidden state with latent for policy/value networks
            combined = torch.cat([hidden_state, latent_z], dim=-1)
            
            # Get policy and value
            policy_logits = self.networks.policy_network(hidden_state, latent_z)
            value = self.networks.value_network(hidden_state, latent_z)
            
            # Get action probabilities
            action_probs = torch.softmax(policy_logits, dim=-1).squeeze().numpy()
            
            # Select action (argmax for deterministic)
            action = int(np.argmax(action_probs))
            confidence = float(action_probs[action])
            
            return action, confidence
        
    def load_test_data(self, data_path: str) -> None:
        """Load and prepare test data"""
        logger.info(f"📊 Loading test data: {data_path}")
        
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        self.test_data = pd.read_csv(data_path)
        
        # Handle both 'time' and 'timestamp' columns
        if 'time' in self.test_data.columns and 'timestamp' not in self.test_data.columns:
            self.test_data['timestamp'] = self.test_data['time']
        
        # Ensure required columns (volume is optional)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        missing_cols = set(required_cols) - set(self.test_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert timestamp to datetime
        self.test_data['timestamp'] = pd.to_datetime(self.test_data['timestamp'])
        self.test_data = self.test_data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✅ Loaded {len(self.test_data):,} bars")
        logger.info(f"📅 Date range: {self.test_data['timestamp'].min()} to {self.test_data['timestamp'].max()}")
        
    def run_monte_carlo_validation(self, num_runs: Optional[int] = None) -> Dict[str, Any]:
        """
        Run full Monte Carlo validation with CAR25 calculation
        
        Returns:
            Dictionary with validation results and statistics
        """
        num_runs = num_runs or self.config.monte_carlo_runs
        logger.info(f"🎲 Starting Monte Carlo validation with {num_runs} runs")
        
        # Run Monte Carlo simulations in parallel
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for run_id in range(num_runs):
                future = executor.submit(self._run_single_monte_carlo, run_id)
                futures.append(future)
                
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if len(results) % 100 == 0:
                        logger.info(f"Progress: {len(results)}/{num_runs} runs completed")
                except Exception as e:
                    logger.error(f"Monte Carlo run failed: {e}")
                    
        # Calculate CAR25 and other statistics
        car25_metrics = self._calculate_car25_metrics(results)
        
        # Generate validation report
        validation_report = self._generate_validation_report(results, car25_metrics)
        
        return validation_report
        
    def _run_single_monte_carlo(self, run_id: int) -> MonteCarloResult:
        """Execute single Monte Carlo run with bootstrap sampling"""
        start_time = datetime.now()
        
        # Set seed for reproducibility
        np.random.seed(run_id)
        
        # Bootstrap sample from available data
        total_bars = len(self.test_data)
        sample_size = min(self.config.bootstrap_sample_size * 390, total_bars)  # 390 bars per day
        
        # Random starting points for sessions
        max_start = total_bars - sample_size
        if max_start <= 0:
            start_idx = 0
            end_idx = total_bars
        else:
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + sample_size
            
        # Extract sample data
        sample_data = self.test_data.iloc[start_idx:end_idx].copy()
        
        # Run trading simulation
        trades = self._simulate_trading(sample_data)
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(trades, sample_data)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return MonteCarloResult(
            run_id=run_id,
            execution_time_seconds=execution_time,
            **metrics
        )
        
    def _simulate_trading(self, data: pd.DataFrame) -> List[TradeResult]:
        """Simulate trading on given data"""
        trades = []
        current_position = None
        
        # Create environment with correct parameters
        env = SWTForexEnvironment(
            session_data=data,  # Pass data as session_data
            config_dict={
                'spread_pips': self.config.spread_cost_pips,
                'price_series_length': 256,
                'reward': {
                    'type': 'pure_pips',
                    'drawdown_penalty': 0.05,
                    'profit_protection': True,
                    'min_protected_reward': 0.01
                }
            }
        )
        
        obs, info = env.reset()  # reset returns tuple (obs, info)
        done = False
        
        while not done:
            # Process market prices through WST to get 128-dim features
            # The environment gives us 256 prices, we need to process them
            market_prices = obs['market_prices']
            
            # Process through feature processor to get WST features
            # Create a dummy state dict for the feature processor
            state_dict = {
                'market_prices': market_prices,
                'position_features': obs['position_features']
            }
            processed_features = self.feature_processor.extract_features(state_dict)
            
            # Get action from networks using WST features
            action, confidence = self.run_inference(
                market_features=processed_features['market_features'],  # 128-dim WST features
                position_features=obs['position_features']
            )
            
            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Record trade if position changed
            if 'trade' in info and info['trade']:
                trade = TradeResult(
                    entry_time=info['trade']['entry_time'],
                    exit_time=info['trade']['exit_time'],
                    direction=info['trade']['direction'],
                    entry_price=info['trade']['entry_price'],
                    exit_price=info['trade']['exit_price'],
                    pnl_pips=info['trade']['pnl_pips'],
                    pnl_dollars=info['trade']['pnl_dollars'],
                    position_size=self.config.max_position_size,
                    spread_cost=self.config.spread_cost_pips,
                    slippage_cost=self.config.slippage_pips
                )
                trades.append(trade)
                
        return trades
        
    def _calculate_performance_metrics(self, trades: List[TradeResult], data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self._empty_metrics()
            
        # Basic statistics
        total_trades = len(trades)
        pnl_list = [t.pnl_pips for t in trades]
        winning_trades = [p for p in pnl_list if p > 0]
        losing_trades = [p for p in pnl_list if p <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl_pips = sum(pnl_list)
        avg_win_pips = np.mean(winning_trades) if winning_trades else 0
        avg_loss_pips = abs(np.mean(losing_trades)) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        # Drawdown calculation
        cumulative_pnl = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown_pips = np.max(drawdown) if len(drawdown) > 0 else 0
        max_drawdown_pct = max_drawdown_pips / self.config.initial_capital if self.config.initial_capital > 0 else 0
        
        # Return calculations
        total_pnl_dollars = sum(t.pnl_dollars for t in trades)
        total_return_pct = (total_pnl_dollars / self.config.initial_capital) * 100
        
        # Annualized return (assuming 252 trading days)
        days_traded = (data['timestamp'].max() - data['timestamp'].min()).days
        if days_traded > 0:
            years_traded = days_traded / 365.25
            annualized_return = ((1 + total_return_pct/100) ** (1/years_traded) - 1) * 100
        else:
            annualized_return = 0
            
        # Risk metrics
        returns = np.array(pnl_list) / self.config.initial_capital
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Recovery factor
        recovery_factor = total_pnl_pips / max_drawdown_pips if max_drawdown_pips > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win_pips) - ((1 - win_rate) * avg_loss_pips)
        
        # Kelly Criterion
        if avg_win_pips > 0 and avg_loss_pips > 0:
            kelly_criterion = (win_rate * avg_loss_pips - (1 - win_rate) * avg_win_pips) / (avg_win_pips * avg_loss_pips)
            kelly_criterion = max(0, min(kelly_criterion, 0.25))  # Cap at 25%
        else:
            kelly_criterion = 0
            
        # Trading frequency
        trades_per_day = total_trades / max(1, days_traded)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl_pips': total_pnl_pips,
            'total_pnl_dollars': total_pnl_dollars,
            'avg_win_pips': avg_win_pips,
            'avg_loss_pips': avg_loss_pips,
            'profit_factor': profit_factor,
            'max_drawdown_pips': max_drawdown_pips,
            'max_drawdown_pct': max_drawdown_pct,
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'expectancy': expectancy,
            'kelly_criterion': kelly_criterion,
            'trades_per_day': trades_per_day
        }
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0) -> float:
        """Calculate Sortino ratio (downside risk only)"""
        if len(returns) < 2:
            return 0
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0
        return np.sqrt(252) * (np.mean(returns) - target_return) / downside_std
        
    def _calculate_car25_metrics(self, results: List[MonteCarloResult]) -> Dict[str, Any]:
        """Calculate CAR25 and related metrics from Monte Carlo results"""
        
        # Extract annualized returns
        returns = [r.annualized_return for r in results]
        
        # Handle empty results
        if not returns or len(returns) == 0:
            return {
                'car25': 0.0,
                'car50': 0.0,
                'car75': 0.0,
                'quality_score': 0.0,
                'passes_thresholds': False,
                'aggregate_metrics': {}
            }
        
        # Calculate percentiles
        car25 = np.percentile(returns, 25)  # 25th percentile (conservative)
        car50 = np.percentile(returns, 50)  # Median
        car75 = np.percentile(returns, 75)  # 75th percentile (optimistic)
        
        # Other aggregate metrics
        avg_win_rate = np.mean([r.win_rate for r in results])
        avg_profit_factor = np.mean([r.profit_factor for r in results])
        avg_max_dd = np.mean([r.max_drawdown_pct for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_sortino = np.mean([r.sortino_ratio for r in results])
        
        # Robustness metrics
        return_std = np.std(returns)
        return_skew = stats.skew(returns)
        return_kurtosis = stats.kurtosis(returns)
        
        # System quality score (custom metric)
        quality_score = self._calculate_quality_score(results)
        
        return {
            'car25': car25,
            'car50': car50,
            'car75': car75,
            'mean_return': np.mean(returns),
            'std_return': return_std,
            'skew_return': return_skew,
            'kurtosis_return': return_kurtosis,
            'avg_win_rate': avg_win_rate,
            'avg_profit_factor': avg_profit_factor,
            'avg_max_drawdown': avg_max_dd,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_sortino_ratio': avg_sortino,
            'quality_score': quality_score,
            'passes_thresholds': self._check_thresholds(car25, avg_max_dd, avg_profit_factor, avg_win_rate)
        }
        
    def _calculate_quality_score(self, results: List[MonteCarloResult]) -> float:
        """Calculate overall system quality score (0-100)"""
        scores = []
        
        for r in results:
            score = 0
            
            # Return component (40 points)
            if r.annualized_return >= 0.30:
                score += 40
            elif r.annualized_return >= 0.20:
                score += 30
            elif r.annualized_return >= 0.15:
                score += 20
            elif r.annualized_return >= 0.10:
                score += 10
                
            # Risk component (30 points)
            if r.max_drawdown_pct <= 0.10:
                score += 30
            elif r.max_drawdown_pct <= 0.15:
                score += 25
            elif r.max_drawdown_pct <= 0.20:
                score += 20
            elif r.max_drawdown_pct <= 0.25:
                score += 10
                
            # Consistency component (30 points)
            if r.profit_factor >= 2.0:
                score += 15
            elif r.profit_factor >= 1.5:
                score += 10
            elif r.profit_factor >= 1.2:
                score += 5
                
            if r.win_rate >= 0.60:
                score += 15
            elif r.win_rate >= 0.50:
                score += 10
            elif r.win_rate >= 0.40:
                score += 5
                
            scores.append(score)
            
        return np.mean(scores)
        
    def _check_thresholds(self, car25: float, avg_dd: float, profit_factor: float, win_rate: float) -> bool:
        """Check if system passes Dr. Bandy's recommended thresholds"""
        return (
            car25 >= self.config.min_acceptable_car25 and
            avg_dd <= self.config.max_acceptable_drawdown and
            profit_factor >= self.config.min_profit_factor and
            win_rate >= self.config.min_win_rate
        )
        
    def _generate_validation_report(self, results: List[MonteCarloResult], car25_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'checkpoint_path': str(self.checkpoint_path),
            'monte_carlo_runs': len(results),
            'car25_metrics': car25_metrics,
            'threshold_analysis': {
                'min_car25_threshold': self.config.min_acceptable_car25,
                'actual_car25': car25_metrics['car25'],
                'car25_passed': car25_metrics['car25'] >= self.config.min_acceptable_car25,
                'max_dd_threshold': self.config.max_acceptable_drawdown,
                'actual_avg_dd': car25_metrics['avg_max_drawdown'],
                'dd_passed': car25_metrics['avg_max_drawdown'] <= self.config.max_acceptable_drawdown,
                'min_profit_factor': self.config.min_profit_factor,
                'actual_profit_factor': car25_metrics['avg_profit_factor'],
                'profit_factor_passed': car25_metrics['avg_profit_factor'] >= self.config.min_profit_factor,
                'overall_passed': car25_metrics['passes_thresholds']
            },
            'recommendation': self._generate_recommendation(car25_metrics),
            'detailed_results': [asdict(r) for r in results[:10]]  # Sample of results
        }
        
        return report
        
    def _generate_recommendation(self, metrics: Dict[str, Any]) -> str:
        """Generate trading recommendation based on validation results"""
        
        if metrics['passes_thresholds']:
            if metrics['quality_score'] >= 80:
                return "STRONGLY RECOMMENDED: System shows excellent risk-adjusted returns with high reliability"
            elif metrics['quality_score'] >= 60:
                return "RECOMMENDED: System meets all thresholds with good performance characteristics"
            else:
                return "ACCEPTABLE: System passes minimum thresholds but shows room for improvement"
        else:
            failures = []
            if metrics['car25'] < self.config.min_acceptable_car25:
                failures.append(f"CAR25 ({metrics['car25']:.2%}) below threshold")
            if metrics['avg_max_drawdown'] > self.config.max_acceptable_drawdown:
                failures.append(f"Drawdown ({metrics['avg_max_drawdown']:.2%}) exceeds limit")
            if metrics['avg_profit_factor'] < self.config.min_profit_factor:
                failures.append(f"Profit factor ({metrics['avg_profit_factor']:.2f}) too low")
                
            return f"NOT RECOMMENDED: System fails validation - {', '.join(failures)}"
            
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl_pips': 0,
            'total_pnl_dollars': 0,
            'avg_win_pips': 0,
            'avg_loss_pips': 0,
            'profit_factor': 0,
            'max_drawdown_pips': 0,
            'max_drawdown_pct': 0,
            'total_return_pct': 0,
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'recovery_factor': 0,
            'expectancy': 0,
            'kelly_criterion': 0,
            'trades_per_day': 0
        }


def main():
    """Main entry point for CAR25 validation"""
    parser = argparse.ArgumentParser(description='Monte Carlo CAR25 Validation for SWT Trading System')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--data', required=True, help='Path to test data CSV')
    parser.add_argument('--runs', type=int, default=1000, help='Number of Monte Carlo runs')
    parser.add_argument('--output', default='car25_validation_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    # Configure validation
    config = CAR25Config(monte_carlo_runs=args.runs)
    
    # Create validator
    validator = MonteCarloCAR25Validator(config)
    
    # Load checkpoint and data
    validator.load_checkpoint(args.checkpoint)
    validator.load_test_data(args.data)
    
    # Run validation
    logger.info("🚀 Starting CAR25 Monte Carlo validation...")
    report = validator.run_monte_carlo_validation()
    
    # Save report
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    logger.info(f"📊 Validation complete! Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CAR25 VALIDATION SUMMARY")
    print("="*60)
    print(f"CAR25 (Conservative): {report['car25_metrics']['car25']:.2%}")
    print(f"CAR50 (Median): {report['car25_metrics']['car50']:.2%}")
    print(f"CAR75 (Optimistic): {report['car25_metrics']['car75']:.2%}")
    print(f"Quality Score: {report['car25_metrics']['quality_score']:.1f}/100")
    print(f"Validation: {'✅ PASSED' if report['car25_metrics']['passes_thresholds'] else '❌ FAILED'}")
    print(f"\nRecommendation: {report['recommendation']}")
    print("="*60)
    

if __name__ == "__main__":
    main()