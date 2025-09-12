#!/usr/bin/env python3
"""
SWT Checkpoint Performance Testing Framework
Validates any checkpoint (Episode 13475 or future ones) against unseen test data
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path for both development and Docker environments
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try importing with proper error handling for Docker environment
try:
    from swt_core.config_manager import ConfigManager
    from swt_core.types import AgentType, ProcessState
    from swt_core.exceptions import ConfigurationError, CheckpointError, InferenceError
    from swt_features.feature_processor import FeatureProcessor
    from swt_inference.checkpoint_loader import CheckpointLoader
    from swt_inference.inference_engine import InferenceEngine as SWTInferenceEngine
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Checking available modules...")
    
    # List available modules for debugging
    for path in sys.path:
        if os.path.exists(path):
            logger.info(f"Path exists: {path}")
            if 'swt' in str(path) or path == str(project_root):
                logger.info(f"Contents of {path}:")
                try:
                    contents = os.listdir(path)
                    for item in contents:
                        if 'swt' in item:
                            logger.info(f"  - {item}")
                except:
                    pass
    
    # Try alternative import approach
    try:
        import swt_core.config_manager as config_manager_module
        import swt_features.feature_processor as feature_processor_module
        import swt_inference.checkpoint_loader as checkpoint_loader_module
        import swt_inference.inference_engine as inference_engine_module
        
        ConfigManager = config_manager_module.ConfigManager
        FeatureProcessor = feature_processor_module.FeatureProcessor
        CheckpointLoader = checkpoint_loader_module.CheckpointLoader
        SWTInferenceEngine = inference_engine_module.InferenceEngine
        
        logger.info("Successfully imported using alternative method")
    except ImportError as e2:
        logger.error(f"Alternative import also failed: {e2}")
        logger.error("Unable to import required SWT modules")
        sys.exit(1)

class CheckpointPerformanceTester:
    """
    Comprehensive checkpoint testing framework for SWT system
    Tests any checkpoint against unseen forex data and generates performance reports
    """
    
    def __init__(self, config_path: str = "config/trading.yaml"):
        """
        Initialize the checkpoint tester
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = None
        self.config = None
        self.feature_processor = None
        self.checkpoint_loader = None
        self.inference_engine = None
        
        # Performance tracking
        self.test_results = {
            'checkpoint_info': {},
            'test_data_info': {},
            'performance_metrics': {},
            'trading_simulation': {},
            'episode_13475_compatibility': False,
            'test_timestamp': datetime.now().isoformat()
        }
        
        logger.info("🧪 Checkpoint Performance Tester initialized")
    
    def setup_testing_environment(self):
        """Initialize all testing components"""
        try:
            logger.info("🔧 Setting up testing environment...")
            
            # Load configuration
            self.config_manager = ConfigManager()
            self.config = self.config_manager.load_config()
            self.config.force_episode_13475_mode()
            
            # Initialize feature processor
            self.feature_processor = FeatureProcessor(self.config)
            logger.info("✅ Feature processor initialized")
            
            # Initialize checkpoint loader
            self.checkpoint_loader = CheckpointLoader(self.config)
            logger.info("✅ Checkpoint loader initialized")
            
            logger.info("✅ Testing environment ready")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise ConfigurationError(f"Testing setup failed: {str(e)}")
    
    def load_test_data(self, csv_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load and validate test data from CSV
        
        Args:
            csv_path: Path to CSV file with OHLCV data
            max_rows: Maximum number of rows to load (for memory management)
            
        Returns:
            Validated DataFrame with proper formatting
        """
        try:
            logger.info(f"📊 Loading test data from: {csv_path}")
            
            # Check if file exists
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"Test data file not found: {csv_path}")
            
            # Load CSV with proper parsing
            df = pd.read_csv(csv_path, nrows=max_rows)
            
            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Check if we have at least timestamp and close (minimal requirement)
                if 'timestamp' not in df.columns or 'close' not in df.columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                logger.warning(f"⚠️  Missing columns {missing_columns}, will use close price for OHLC")
                
                # Fill missing OHLC with close price
                for col in ['open', 'high', 'low']:
                    if col not in df.columns:
                        df[col] = df['close']
                
                # Fill missing volume with default
                if 'volume' not in df.columns:
                    df['volume'] = 100
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Remove any NaN values
            df = df.dropna()
            
            # Data validation
            if len(df) == 0:
                raise ValueError("No valid data after cleaning")
            
            # Store test data info
            self.test_results['test_data_info'] = {
                'file_path': csv_path,
                'total_rows': len(df),
                'date_range_start': df.index.min().isoformat(),
                'date_range_end': df.index.max().isoformat(),
                'duration_days': (df.index.max() - df.index.min()).days,
                'columns': list(df.columns),
                'price_range': {
                    'min': float(df['close'].min()),
                    'max': float(df['close'].max()),
                    'mean': float(df['close'].mean())
                }
            }
            
            logger.info(f"✅ Loaded {len(df):,} data points")
            logger.info(f"📅 Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"💰 Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {e}")
            raise
    
    def load_and_validate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint and validate its compatibility
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data with validation info
        """
        try:
            logger.info(f"📦 Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint_data = self.checkpoint_loader.load_checkpoint(checkpoint_path)
            
            # Extract checkpoint metadata
            checkpoint_info = {
                'file_path': checkpoint_path,
                'episode': checkpoint_data.get('episode', 'unknown'),
                'model_config': checkpoint_data.get('config', {}),
                'timestamp': checkpoint_data.get('timestamp', 'unknown'),
                'file_size_mb': Path(checkpoint_path).stat().st_size / (1024 * 1024)
            }
            
            # Validate Episode 13475 compatibility
            episode_13475_compatible = self._validate_episode_13475_parameters(checkpoint_data)
            checkpoint_info['episode_13475_compatible'] = episode_13475_compatible
            
            # Initialize inference engine with this checkpoint
            self.inference_engine = SWTInferenceEngine(
                networks=checkpoint_data['networks'],
                config=self.config
            )
            
            # Store checkpoint info
            self.test_results['checkpoint_info'] = checkpoint_info
            self.test_results['episode_13475_compatibility'] = episode_13475_compatible
            
            if episode_13475_compatible:
                logger.info("✅ Checkpoint is Episode 13475 compatible")
            else:
                logger.warning("⚠️  Checkpoint may not be Episode 13475 compatible")
            
            logger.info(f"✅ Checkpoint loaded successfully (Episode: {checkpoint_info['episode']})")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise CheckpointError(f"Checkpoint loading failed: {str(e)}")
    
    def _validate_episode_13475_parameters(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Validate if checkpoint matches Episode 13475 parameters"""
        try:
            config = checkpoint_data.get('config', {})
            
            expected_params = {
                'mcts_simulations': 15,
                'c_puct': 1.25,
                'wst_J': 2,
                'wst_Q': 6,
                'position_features_dim': 9,
                'market_features_dim': 128
            }
            
            compatible = True
            for param, expected_value in expected_params.items():
                actual_value = config.get(param)
                if actual_value != expected_value:
                    logger.warning(f"⚠️  Parameter mismatch: {param}={actual_value}, expected={expected_value}")
                    compatible = False
            
            return compatible
            
        except Exception as e:
            logger.warning(f"⚠️  Could not validate Episode 13475 compatibility: {e}")
            return False
    
    def run_inference_benchmark(self, test_data: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Run inference benchmark on test data
        
        Args:
            test_data: Test DataFrame with OHLCV data
            sample_size: Number of samples to test (for performance)
            
        Returns:
            Performance metrics dictionary
        """
        try:
            logger.info(f"🚀 Running inference benchmark on {sample_size} samples...")
            
            # Sample test data points
            if len(test_data) > sample_size:
                sample_indices = np.random.choice(len(test_data), sample_size, replace=False)
                sample_data = test_data.iloc[sample_indices].copy()
            else:
                sample_data = test_data.copy()
            
            logger.info(f"📊 Testing {len(sample_data)} data points")
            
            # Performance tracking
            inference_times = []
            feature_processing_times = []
            successful_inferences = 0
            failed_inferences = 0
            action_distribution = {'hold': 0, 'buy': 0, 'sell': 0}
            confidence_scores = []
            
            # Mock position state for testing
            mock_position = {
                'position_size': 0.0,
                'unrealized_pnl': 0.0,
                'entry_price': 0.0,
                'holding_time': 0,
                'daily_pnl': 0.0
            }
            
            logger.info("🔄 Processing samples...")
            
            for i, (timestamp, row) in enumerate(sample_data.iterrows()):
                try:
                    # Prepare market data
                    market_data = {
                        'timestamp': timestamp,
                        'price': row['close'],
                        'volume': row['volume'],
                        'bid': row['close'] - 0.0002,  # Mock spread
                        'ask': row['close'] + 0.0002,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low']
                    }
                    
                    # Feature processing timing
                    feature_start = time.time()
                    observation = self.feature_processor.process_observation(
                        market_data=market_data,
                        position_info=mock_position
                    )
                    feature_time = time.time() - feature_start
                    feature_processing_times.append(feature_time)
                    
                    # Inference timing
                    inference_start = time.time()
                    result = self.inference_engine.run_inference(observation)
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)
                    
                    # Track results
                    action = result.get('action', 0)
                    confidence = result.get('confidence', 0.0)
                    
                    # Map action to string
                    action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
                    action_str = action_map.get(action, 'hold')
                    action_distribution[action_str] += 1
                    
                    confidence_scores.append(confidence)
                    successful_inferences += 1
                    
                    # Progress logging
                    if (i + 1) % 100 == 0:
                        logger.info(f"   Progress: {i+1}/{len(sample_data)} ({((i+1)/len(sample_data)*100):.1f}%)")
                    
                except Exception as e:
                    failed_inferences += 1
                    logger.debug(f"   Inference failed for sample {i}: {e}")
                    continue
            
            # Calculate metrics
            metrics = {
                'total_samples': len(sample_data),
                'successful_inferences': successful_inferences,
                'failed_inferences': failed_inferences,
                'success_rate': successful_inferences / len(sample_data) * 100,
                'performance': {
                    'avg_inference_time_ms': np.mean(inference_times) * 1000,
                    'p95_inference_time_ms': np.percentile(inference_times, 95) * 1000,
                    'p99_inference_time_ms': np.percentile(inference_times, 99) * 1000,
                    'avg_feature_time_ms': np.mean(feature_processing_times) * 1000,
                    'total_time_seconds': sum(inference_times) + sum(feature_processing_times)
                },
                'action_distribution': action_distribution,
                'action_distribution_pct': {
                    k: (v / successful_inferences * 100) for k, v in action_distribution.items()
                },
                'confidence_stats': {
                    'mean': float(np.mean(confidence_scores)),
                    'std': float(np.std(confidence_scores)),
                    'min': float(np.min(confidence_scores)),
                    'max': float(np.max(confidence_scores)),
                    'p25': float(np.percentile(confidence_scores, 25)),
                    'p50': float(np.percentile(confidence_scores, 50)),
                    'p75': float(np.percentile(confidence_scores, 75))
                }
            }
            
            # Performance evaluation
            performance_grade = self._evaluate_performance(metrics)
            metrics['performance_grade'] = performance_grade
            
            # Store results
            self.test_results['performance_metrics'] = metrics
            
            logger.info("✅ Inference benchmark completed")
            logger.info(f"📊 Results summary:")
            logger.info(f"   Success rate: {metrics['success_rate']:.1f}%")
            logger.info(f"   Avg inference time: {metrics['performance']['avg_inference_time_ms']:.2f}ms")
            logger.info(f"   Action distribution: {metrics['action_distribution']}")
            logger.info(f"   Performance grade: {performance_grade}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Inference benchmark failed: {e}")
            raise
    
    def _evaluate_performance(self, metrics: Dict[str, Any]) -> str:
        """Evaluate performance and assign grade"""
        score = 0
        
        # Success rate (40 points)
        success_rate = metrics['success_rate']
        if success_rate >= 95:
            score += 40
        elif success_rate >= 90:
            score += 35
        elif success_rate >= 85:
            score += 30
        elif success_rate >= 80:
            score += 20
        
        # Inference speed (30 points)
        avg_inference_ms = metrics['performance']['avg_inference_time_ms']
        if avg_inference_ms <= 100:
            score += 30
        elif avg_inference_ms <= 200:
            score += 25
        elif avg_inference_ms <= 300:
            score += 20
        elif avg_inference_ms <= 500:
            score += 15
        
        # Action diversity (20 points) - model should make decisions, not just hold
        action_dist = metrics['action_distribution_pct']
        hold_pct = action_dist.get('hold', 100)
        if hold_pct <= 70:
            score += 20
        elif hold_pct <= 80:
            score += 15
        elif hold_pct <= 90:
            score += 10
        elif hold_pct <= 95:
            score += 5
        
        # Confidence distribution (10 points)
        conf_stats = metrics['confidence_stats']
        if conf_stats['std'] > 0.1 and conf_stats['mean'] > 0.3:
            score += 10
        elif conf_stats['mean'] > 0.3:
            score += 7
        elif conf_stats['std'] > 0.1:
            score += 5
        
        # Assign grade
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B (Good)"
        elif score >= 60:
            return "C (Acceptable)"
        elif score >= 50:
            return "D (Poor)"
        else:
            return "F (Failed)"
    
    def run_trading_simulation(self, test_data: pd.DataFrame, initial_balance: float = 10000.0) -> Dict[str, Any]:
        """
        Run simplified trading simulation
        
        Args:
            test_data: Test data for simulation
            initial_balance: Starting balance
            
        Returns:
            Trading simulation results
        """
        try:
            logger.info(f"📈 Running trading simulation (initial balance: ${initial_balance:,.2f})")
            
            # Simulation parameters
            balance = initial_balance
            position_size = 0.0
            position_entry_price = 0.0
            position_entry_time = None
            total_trades = 0
            winning_trades = 0
            trade_pnl_history = []
            
            # Risk management
            max_position_size = 0.02  # 2% per trade
            daily_loss_limit = -200.0  # $200 daily loss limit
            daily_pnl = 0.0
            
            # Mock position for inference
            position_info = {
                'position_size': 0.0,
                'unrealized_pnl': 0.0,
                'entry_price': 0.0,
                'holding_time': 0,
                'daily_pnl': 0.0
            }
            
            simulation_data = []
            
            logger.info(f"🔄 Simulating {len(test_data)} time steps...")
            
            for i, (timestamp, row) in enumerate(test_data.iterrows()):
                try:
                    current_price = row['close']
                    
                    # Update position info
                    position_info['position_size'] = position_size
                    if position_size != 0:
                        unrealized_pnl = (current_price - position_entry_price) * position_size * 10000  # Mock pip calculation
                        position_info['unrealized_pnl'] = unrealized_pnl
                        position_info['entry_price'] = position_entry_price
                        position_info['holding_time'] = (timestamp - position_entry_time).total_seconds() / 60 if position_entry_time else 0
                    else:
                        position_info['unrealized_pnl'] = 0.0
                        position_info['entry_price'] = 0.0
                        position_info['holding_time'] = 0
                    
                    position_info['daily_pnl'] = daily_pnl
                    
                    # Prepare market data
                    market_data = {
                        'timestamp': timestamp,
                        'price': current_price,
                        'volume': row['volume'],
                        'bid': current_price - 0.0002,
                        'ask': current_price + 0.0002
                    }
                    
                    # Get inference
                    observation = self.feature_processor.process_observation(
                        market_data=market_data,
                        position_info=position_info
                    )
                    
                    result = self.inference_engine.run_inference(observation)
                    action = result.get('action', 0)
                    confidence = result.get('confidence', 0.0)
                    
                    # Trading logic
                    trade_executed = False
                    
                    # Risk management check
                    if daily_pnl < daily_loss_limit:
                        # Skip trading for the day
                        action = 0  # Force hold
                    
                    if action == 1 and position_size == 0 and confidence > 0.35:  # Buy signal
                        position_size = max_position_size
                        position_entry_price = current_price
                        position_entry_time = timestamp
                        total_trades += 1
                        trade_executed = True
                        
                    elif action == 2 and position_size != 0:  # Sell signal (close position)
                        # Calculate P&L
                        pnl = (current_price - position_entry_price) * position_size * 10000
                        balance += pnl
                        daily_pnl += pnl
                        trade_pnl_history.append(pnl)
                        
                        if pnl > 0:
                            winning_trades += 1
                        
                        # Close position
                        position_size = 0.0
                        position_entry_price = 0.0
                        position_entry_time = None
                        trade_executed = True
                    
                    # Record simulation step
                    simulation_data.append({
                        'timestamp': timestamp,
                        'price': current_price,
                        'action': action,
                        'confidence': confidence,
                        'position_size': position_size,
                        'balance': balance,
                        'daily_pnl': daily_pnl,
                        'trade_executed': trade_executed
                    })
                    
                    # Progress logging
                    if (i + 1) % 1000 == 0:
                        logger.info(f"   Progress: {i+1}/{len(test_data)} ({((i+1)/len(test_data)*100):.1f}%) | Balance: ${balance:,.2f}")
                    
                except Exception as e:
                    logger.debug(f"   Simulation step {i} failed: {e}")
                    continue
            
            # Final calculations
            final_return = (balance - initial_balance) / initial_balance * 100
            win_rate = (winning_trades / max(total_trades, 1)) * 100
            avg_trade_pnl = np.mean(trade_pnl_history) if trade_pnl_history else 0
            
            simulation_results = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return_pct': final_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate_pct': win_rate,
                'avg_trade_pnl': avg_trade_pnl,
                'max_drawdown': min(trade_pnl_history) if trade_pnl_history else 0,
                'max_profit': max(trade_pnl_history) if trade_pnl_history else 0,
                'trade_pnl_history': trade_pnl_history[:100],  # Store first 100 trades
                'simulation_duration_days': (test_data.index.max() - test_data.index.min()).days
            }
            
            # Store results
            self.test_results['trading_simulation'] = simulation_results
            
            logger.info("✅ Trading simulation completed")
            logger.info(f"📊 Simulation results:")
            logger.info(f"   Total return: {final_return:.2f}%")
            logger.info(f"   Total trades: {total_trades}")
            logger.info(f"   Win rate: {win_rate:.1f}%")
            logger.info(f"   Final balance: ${balance:,.2f}")
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"❌ Trading simulation failed: {e}")
            raise
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive test report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        try:
            logger.info("📋 Generating test report...")
            
            checkpoint_info = self.test_results['checkpoint_info']
            test_data_info = self.test_results['test_data_info']
            performance_metrics = self.test_results['performance_metrics']
            trading_simulation = self.test_results['trading_simulation']
            
            report = f"""
# SWT Checkpoint Performance Test Report

## Test Summary
- **Test Date**: {self.test_results['test_timestamp']}
- **Episode 13475 Compatible**: {'✅ Yes' if self.test_results['episode_13475_compatibility'] else '❌ No'}
- **Performance Grade**: {performance_metrics.get('performance_grade', 'N/A')}

## Checkpoint Information
- **File**: {checkpoint_info['file_path']}
- **Episode**: {checkpoint_info['episode']}
- **File Size**: {checkpoint_info['file_size_mb']:.2f} MB
- **Episode 13475 Compatible**: {'✅ Yes' if checkpoint_info['episode_13475_compatible'] else '❌ No'}

## Test Data Information
- **File**: {test_data_info['file_path']}
- **Total Records**: {test_data_info['total_rows']:,}
- **Date Range**: {test_data_info['date_range_start']} to {test_data_info['date_range_end']}
- **Duration**: {test_data_info['duration_days']} days
- **Price Range**: {test_data_info['price_range']['min']:.4f} - {test_data_info['price_range']['max']:.4f}

## Performance Metrics
- **Success Rate**: {performance_metrics['success_rate']:.1f}%
- **Average Inference Time**: {performance_metrics['performance']['avg_inference_time_ms']:.2f}ms
- **P95 Inference Time**: {performance_metrics['performance']['p95_inference_time_ms']:.2f}ms
- **Average Feature Processing**: {performance_metrics['performance']['avg_feature_time_ms']:.2f}ms

## Action Distribution
- **Hold**: {performance_metrics['action_distribution']['hold']} ({performance_metrics['action_distribution_pct']['hold']:.1f}%)
- **Buy**: {performance_metrics['action_distribution']['buy']} ({performance_metrics['action_distribution_pct']['buy']:.1f}%)
- **Sell**: {performance_metrics['action_distribution']['sell']} ({performance_metrics['action_distribution_pct']['sell']:.1f}%)

## Confidence Statistics
- **Mean**: {performance_metrics['confidence_stats']['mean']:.3f}
- **Std Dev**: {performance_metrics['confidence_stats']['std']:.3f}
- **Range**: {performance_metrics['confidence_stats']['min']:.3f} - {performance_metrics['confidence_stats']['max']:.3f}

## Trading Simulation Results
- **Initial Balance**: ${trading_simulation['initial_balance']:,.2f}
- **Final Balance**: ${trading_simulation['final_balance']:,.2f}
- **Total Return**: {trading_simulation['total_return_pct']:.2f}%
- **Total Trades**: {trading_simulation['total_trades']}
- **Win Rate**: {trading_simulation['win_rate_pct']:.1f}%
- **Average Trade P&L**: ${trading_simulation['avg_trade_pnl']:.2f}
- **Max Profit**: ${trading_simulation['max_profit']:.2f}
- **Max Drawdown**: ${trading_simulation['max_drawdown']:.2f}

## Performance Assessment

### Speed Performance
{'✅ Excellent' if performance_metrics['performance']['avg_inference_time_ms'] <= 200 else '⚠️ Needs Improvement' if performance_metrics['performance']['avg_inference_time_ms'] <= 500 else '❌ Poor'} - Average inference time: {performance_metrics['performance']['avg_inference_time_ms']:.2f}ms

### Decision Making
{'✅ Good Diversity' if performance_metrics['action_distribution_pct']['hold'] <= 80 else '⚠️ Too Conservative' if performance_metrics['action_distribution_pct']['hold'] <= 90 else '❌ No Decisions'} - Hold rate: {performance_metrics['action_distribution_pct']['hold']:.1f}%

### Trading Performance  
{'✅ Profitable' if trading_simulation['total_return_pct'] > 0 else '❌ Unprofitable'} - Return: {trading_simulation['total_return_pct']:.2f}%

## Recommendations

{self._generate_recommendations(performance_metrics, trading_simulation)}

---
*Generated by SWT Checkpoint Performance Tester*
"""
            
            # Save report if path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report)
                logger.info(f"📄 Report saved to: {output_path}")
            
            # Also save raw results as JSON
            json_path = output_path.replace('.md', '.json') if output_path else 'test_results.json'
            with open(json_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            
            logger.info(f"📊 Raw results saved to: {json_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            raise
    
    def _generate_recommendations(self, performance_metrics: Dict, trading_simulation: Dict) -> str:
        """Generate performance recommendations"""
        recommendations = []
        
        # Performance recommendations
        avg_inference_ms = performance_metrics['performance']['avg_inference_time_ms']
        if avg_inference_ms > 200:
            recommendations.append("• **Optimize inference speed** - Consider model pruning or hardware acceleration")
        
        # Action diversity recommendations
        hold_pct = performance_metrics['action_distribution_pct']['hold']
        if hold_pct > 90:
            recommendations.append("• **Increase decision frequency** - Model may be too conservative")
        elif hold_pct < 50:
            recommendations.append("• **Review risk management** - Model may be over-trading")
        
        # Trading performance recommendations
        total_return = trading_simulation['total_return_pct']
        win_rate = trading_simulation['win_rate_pct']
        
        if total_return < 0:
            recommendations.append("• **Review trading strategy** - Negative returns indicate poor performance")
        
        if win_rate < 45:
            recommendations.append("• **Improve trade selection** - Low win rate suggests poor entry signals")
        
        # Episode 13475 compatibility
        if not self.test_results['episode_13475_compatibility']:
            recommendations.append("• **Verify Episode 13475 compatibility** - Parameters may not match proven configuration")
        
        if not recommendations:
            recommendations.append("• **Performance looks good** - Consider testing with more diverse data")
        
        return '\n'.join(recommendations)

def test_checkpoint_against_csv(checkpoint_path: str, csv_path: str, output_dir: str = "test_results"):
    """
    Convenience function to test a checkpoint against CSV data
    
    Args:
        checkpoint_path: Path to checkpoint file
        csv_path: Path to test CSV data
        output_dir: Directory for output files
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize tester
        tester = CheckpointPerformanceTester()
        tester.setup_testing_environment()
        
        # Load data and checkpoint
        test_data = tester.load_test_data(csv_path, max_rows=10000)  # Limit for testing
        tester.load_and_validate_checkpoint(checkpoint_path)
        
        # Run tests
        tester.run_inference_benchmark(test_data, sample_size=1000)
        tester.run_trading_simulation(test_data)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = Path(checkpoint_path).stem
        report_path = f"{output_dir}/test_report_{checkpoint_name}_{timestamp}.md"
        
        report = tester.generate_report(report_path)
        
        logger.info(f"✅ Testing complete! Report: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        return False

def main():
    """Main entry point with command line support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SWT Checkpoint Performance Tester")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--data", required=True, help="Path to test CSV data")
    parser.add_argument("--output", default="test_results", help="Output directory")
    parser.add_argument("--config", default="config/trading.yaml", help="Configuration file")
    parser.add_argument("--max-rows", type=int, help="Maximum rows to load from CSV")
    parser.add_argument("--sample-size", type=int, default=1000, help="Inference benchmark sample size")
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting checkpoint performance testing...")
        
        # Initialize tester
        tester = CheckpointPerformanceTester(args.config)
        tester.setup_testing_environment()
        
        # Load data
        test_data = tester.load_test_data(args.data, args.max_rows)
        tester.load_and_validate_checkpoint(args.checkpoint)
        
        # Run benchmarks
        tester.run_inference_benchmark(test_data, args.sample_size)
        tester.run_trading_simulation(test_data)
        
        # Generate report
        Path(args.output).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = Path(args.checkpoint).stem
        report_path = f"{args.output}/test_report_{checkpoint_name}_{timestamp}.md"
        
        tester.generate_report(report_path)
        
        logger.info("✅ All tests completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())