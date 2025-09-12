#!/usr/bin/env python3
"""
REAL Episode 13475 Trading Performance Test
EXACT training settings - NO artificial indicators, NO stops, NO targets
Last 3 months of data testing with REAL checkpoint
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple
import time

# Add to Python path
sys.path.append(str(Path(__file__).parent))

from swt_core.config_manager import ConfigManager
from swt_features.feature_processor import FeatureProcessor
from swt_inference.checkpoint_loader import CheckpointLoader
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Episode13475RealPerformanceTest:
    """
    REAL Episode 13475 performance test with EXACT training conditions
    NO artificial indicators, NO stops, NO fake shit
    """
    
    def __init__(self, checkpoint_path: str, data_path: str):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        
        # Load REAL Episode 13475 configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # Initialize components
        self.feature_processor = FeatureProcessor(self.config)
        self.checkpoint_loader = CheckpointLoader(self.config)
        
        # Load REAL Episode 13475 networks
        logger.info("🔥 Loading REAL Episode 13475 checkpoint...")
        self.checkpoint_data = self.checkpoint_loader.load_checkpoint(checkpoint_path)
        self.networks = self.checkpoint_data['networks']
        self.networks.eval()
        
        # Trading state
        self.trades = []
        self.current_position = None
        self.balance = 10000.0
        self.total_pips = 0.0
        
        logger.info("✅ REAL Episode 13475 test system initialized")
    
    def load_data(self) -> pd.DataFrame:
        """Load last 3 months of GBPJPY data"""
        logger.info(f"📊 Loading data from: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime')
        
        # Get last 3 months
        end_date = df['datetime'].max()
        start_date = end_date - timedelta(days=90)
        
        test_data = df[df['datetime'] >= start_date].copy()
        logger.info(f"📈 Test data: {len(test_data)} candles from {start_date} to {end_date}")
        
        return test_data
    
    def run_inference(self, market_data: np.ndarray, position_info: Dict) -> Tuple[int, float]:
        """
        Run REAL Episode 13475 inference with EXACT training setup
        NO artificial indicators - just WST + MCTS + Neural Networks
        """
        with torch.no_grad():
            # Process with WST (EXACT Episode 13475 setup: J=2, Q=6)
            observation = self.feature_processor.process_observation(
                market_data=market_data,
                position_info=position_info
            )
            
            if observation is None:
                return 0, 0.0  # Hold
            
            # Convert to tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            
            # REAL Episode 13475 initial inference
            result = self.networks.initial_inference(obs_tensor)
            
            # Get policy and value
            policy_logits = result['policy_logits']
            value_dist = result['value_distribution']
            
            # Convert to action probabilities
            policy_probs = F.softmax(policy_logits, dim=1)
            
            # Get action (0=Hold, 1=Buy, 2=Sell, 3=Close)
            action = policy_probs.argmax(dim=1).item()
            confidence = policy_probs.max().item()
            
            return action, confidence
    
    def execute_trade(self, action: int, confidence: float, price: float, timestamp: datetime) -> None:
        """
        Execute trade with EXACT Episode 13475 logic
        NO artificial stops or targets - pure signal-based
        """
        min_confidence = 0.35  # Episode 13475 exact threshold
        
        if confidence < min_confidence:
            return  # Skip low confidence
        
        if action == 1 and self.current_position is None:  # Buy
            self.current_position = {
                'type': 'BUY',
                'entry_price': price,
                'entry_time': timestamp,
                'size': 1  # Episode 13475 exact size
            }
            logger.info(f"🟢 BUY at {price:.5f} (confidence: {confidence:.3f})")
            
        elif action == 2 and self.current_position is None:  # Sell
            self.current_position = {
                'type': 'SELL',
                'entry_price': price,
                'entry_time': timestamp,
                'size': 1  # Episode 13475 exact size
            }
            logger.info(f"🔴 SELL at {price:.5f} (confidence: {confidence:.3f})")
            
        elif action == 3 and self.current_position is not None:  # Close
            self.close_position(price, timestamp, confidence)
    
    def close_position(self, price: float, timestamp: datetime, confidence: float) -> None:
        """Close position and record trade results"""
        if self.current_position is None:
            return
        
        entry_price = self.current_position['entry_price']
        position_type = self.current_position['type']
        
        if position_type == 'BUY':
            pips = (price - entry_price) * 10000  # GBPJPY pip calculation
        else:  # SELL
            pips = (entry_price - price) * 10000
        
        # Record trade
        trade = {
            'entry_time': self.current_position['entry_time'],
            'exit_time': timestamp,
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': price,
            'pips': pips,
            'confidence': confidence,
            'duration_minutes': (timestamp - self.current_position['entry_time']).total_seconds() / 60
        }
        
        self.trades.append(trade)
        self.total_pips += pips
        
        logger.info(f"💰 CLOSE {position_type} at {price:.5f}: {pips:+.1f} pips (total: {self.total_pips:+.1f})")
        
        self.current_position = None
    
    def run_backtest(self) -> Dict:
        """
        Run REAL Episode 13475 backtest on last 3 months
        EXACT training conditions - NO shortcuts
        """
        logger.info("🚀 Starting REAL Episode 13475 backtest...")
        
        data = self.load_data()
        
        # Process each candle with REAL Episode 13475
        market_buffer = []
        processed_candles = 0
        
        for idx, row in data.iterrows():
            # Build market data buffer (256 window for WST)
            market_buffer.append([
                row['open'], row['high'], row['low'], row['close'], row['volume']
            ])
            
            if len(market_buffer) > 256:
                market_buffer.pop(0)
            
            # Need full window for WST processing
            if len(market_buffer) < 256:
                continue
            
            # Prepare market data
            market_data = np.array(market_buffer)
            
            # Position info
            position_info = {
                'has_position': self.current_position is not None,
                'position_type': self.current_position['type'] if self.current_position else 'NONE',
                'position_size': self.current_position['size'] if self.current_position else 0,
                'entry_price': self.current_position['entry_price'] if self.current_position else 0.0,
                'current_price': row['close'],
                'unrealized_pnl': 0.0,
                'bars_in_position': 0,
                'max_drawdown': 0.0,
                'accumulated_drawdown': 0.0,
                'bars_since_last_dd': 0
            }
            
            # Run REAL Episode 13475 inference
            action, confidence = self.run_inference(market_data, position_info)
            
            # Execute trade
            self.execute_trade(action, confidence, row['close'], row['datetime'])
            
            processed_candles += 1
            
            if processed_candles % 1000 == 0:
                logger.info(f"📈 Processed {processed_candles} candles, trades: {len(self.trades)}, pips: {self.total_pips:+.1f}")
        
        # Close any remaining position
        if self.current_position is not None:
            final_price = data.iloc[-1]['close']
            final_time = data.iloc[-1]['datetime']
            self.close_position(final_price, final_time, 1.0)
        
        # Calculate results
        results = self.calculate_results()
        
        logger.info("🏁 REAL Episode 13475 backtest completed!")
        return results
    
    def calculate_results(self) -> Dict:
        """Calculate REAL trading performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pips': 0.0,
                'total_pips': 0.0,
                'max_drawdown': 0.0,
                'profitable_trades': 0,
                'losing_trades': 0
            }
        
        winning_trades = [t for t in self.trades if t['pips'] > 0]
        losing_trades = [t for t in self.trades if t['pips'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100
        avg_pips = sum(t['pips'] for t in self.trades) / len(self.trades)
        
        # Calculate drawdown
        running_pips = 0
        peak_pips = 0
        max_dd = 0
        
        for trade in self.trades:
            running_pips += trade['pips']
            peak_pips = max(peak_pips, running_pips)
            drawdown = peak_pips - running_pips
            max_dd = max(max_dd, drawdown)
        
        results = {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_pips': avg_pips,
            'total_pips': self.total_pips,
            'max_drawdown': max_dd,
            'profitable_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': sum(t['pips'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t['pips'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'largest_win': max((t['pips'] for t in self.trades), default=0),
            'largest_loss': min((t['pips'] for t in self.trades), default=0)
        }
        
        return results


def main():
    checkpoint_path = "checkpoints/episode_13475.pth"
    data_path = "data/GBPJPY_M1_202201-202508.csv"
    
    if not Path(checkpoint_path).exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not Path(data_path).exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return
    
    # Run REAL Episode 13475 test
    tester = Episode13475RealPerformanceTest(checkpoint_path, data_path)
    results = tester.run_backtest()
    
    # Display REAL results
    print("\n" + "="*60)
    print("🏆 REAL EPISODE 13475 TRADING RESULTS")
    print("="*60)
    print(f"📊 Total Trades: {results['total_trades']}")
    print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
    print(f"📈 Average Pips: {results['avg_pips']:+.1f}")
    print(f"💰 Total Pips: {results['total_pips']:+.1f}")
    print(f"📉 Max Drawdown: {results['max_drawdown']:.1f} pips")
    print(f"✅ Profitable: {results['profitable_trades']}")
    print(f"❌ Losing: {results['losing_trades']}")
    print(f"🟢 Avg Win: {results['avg_win']:+.1f} pips")
    print(f"🔴 Avg Loss: {results['avg_loss']:+.1f} pips")
    print(f"🚀 Largest Win: {results['largest_win']:+.1f} pips")
    print(f"💥 Largest Loss: {results['largest_loss']:+.1f} pips")
    print("="*60)
    
    # Save results
    with open('episode13475_real_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("💾 Results saved to episode13475_real_results.json")
    
    return results


if __name__ == "__main__":
    main()