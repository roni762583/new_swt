#!/usr/bin/env python3
"""
SWT Forex Trading Environment
WST-Enhanced forex environment for Stochastic MuZero training
"""

import numpy as np
import pandas as pd
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_VERSION = 'gymnasium'
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_VERSION = 'gym'
    except ImportError:
        # Fallback minimal implementation
        class MockSpace:
            def __init__(self, **kwargs):
                pass
        
        class MockSpaces:
            @staticmethod
            def Discrete(n):
                return MockSpace()
            
            @staticmethod  
            def Box(**kwargs):
                return MockSpace()
                
            @staticmethod
            def Dict(spaces_dict):
                return MockSpace()
        
        class MockGym:
            Env = object
        
        gym = MockGym()
        spaces = MockSpaces()
        GYM_VERSION = 'mock'
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import IntEnum
import logging
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class SWTAction(IntEnum):
    """Trading actions for SWT environment"""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


@dataclass
class SWTTradeRecord:
    """Record of a completed trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    pnl_pips: float
    duration_bars: int
    max_drawdown_pips: float
    accumulated_drawdown_pips: float
    reward_amddp5: float
    reward_amddp1: float  # AMDDP1 reward with 1% drawdown penalty
    exit_reason: str


@dataclass
class SWTPositionState:
    """Current position state"""
    is_long: bool = False
    is_short: bool = False
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    duration_bars: int = 0
    unrealized_pnl_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    accumulated_drawdown_pips: float = 0.0
    bars_since_max_drawdown: int = 0
    near_stop_loss: bool = False
    near_take_profit: bool = False
    high_drawdown: bool = False


class SWTForexEnvironment(gym.Env):
    """
    SWT Forex Trading Environment
    
    Provides:
    - 256 M5 closing prices for WST processing
    - 9D position features
    - AMDDP5 reward with profit protection
    - Gymnasium-compatible interface
    """
    
    def __init__(self, 
                 data_path: str = None,
                 config_dict: dict = None,
                 start_idx: int = None,
                 end_idx: int = None,
                 session_data: pd.DataFrame = None):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_dict)
        
        # Use session data if provided, otherwise load from file
        if session_data is not None:
            self.df = session_data.copy()
            self.data_path = "session_data"
        else:
            # Load forex data from file
            self.data_path = Path(data_path)
            self.df = self._load_data()
            
            # Set data range
            if start_idx is not None and end_idx is not None:
                self.df = self.df.iloc[start_idx:end_idx].copy()
        
        self.total_steps = len(self.df) - self.config['price_series_length'] - 1
        
        # Environment parameters
        self.price_series_length = self.config['price_series_length']
        self.spread_pips = self.config['spread_pips']
        self.pip_value = self.config['pip_value']
        
        # Reward parameters
        self.reward_type = self.config['reward']['type']
        self.drawdown_penalty = self.config['reward']['drawdown_penalty']
        self.profit_protection = self.config['reward']['profit_protection']
        self.min_protected_reward = self.config['reward']['min_protected_reward']
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        
        # Observation space: market prices + position features
        self.observation_space = spaces.Dict({
            'market_prices': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.price_series_length,), dtype=np.float32
            ),
            'position_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(9,), dtype=np.float32
            )
        })
        
        # State variables
        self.current_step = 0
        self.position = SWTPositionState()
        self.completed_trades: List[SWTTradeRecord] = []
        self.total_pnl_pips = 0.0
        self.total_reward = 0.0
        
        # Price normalization parameters
        self._price_mean = None
        self._price_std = None
        self._initialize_price_normalization()
        
        logger.info(f"🏦 SWT Forex Environment initialized")
        logger.info(f"   Gym version: {GYM_VERSION}")
        logger.info(f"   Data: {len(self.df)} bars")
        logger.info(f"   Price series length: {self.price_series_length}")
        logger.info(f"   Spread: {self.spread_pips} pips")
        logger.info(f"   Reward: {self.reward_type}")
        
    def _load_config(self, config_dict: dict = None) -> dict:
        """Load environment configuration"""
        default_config = {
            'price_series_length': 256,
            'spread_pips': 4.0,
            'pip_value': 0.01,
            'reward': {
                'type': 'pure_pips',
                'drawdown_penalty': 0.05,
                'profit_protection': True,
                'min_protected_reward': 0.01
            }
        }
        
        if config_dict:
            default_config.update(config_dict)
            
        return default_config
        
    def _load_data(self) -> pd.DataFrame:
        """Load forex data from file"""
        if self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.duckdb':
            import duckdb
            conn = duckdb.connect(str(self.data_path))
            # Use the OHLC training data table
            try:
                df = conn.execute("SELECT timestamp, open, high, low, close, volume FROM m1_ohlcv_oanda_hist_training_data ORDER BY timestamp").df()
            except Exception as e:
                # Show available tables if primary table not found
                tables = conn.execute("SHOW TABLES").df()
                raise ValueError(f"Could not load OHLC data. Available tables: {tables['name'].tolist()}. Error: {e}")
            conn.close()
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")
            
        # Ensure required columns exist
        required_columns = ['timestamp', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Convert timestamp if string
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    def _initialize_price_normalization(self):
        """Initialize price normalization parameters"""
        prices = self.df['close'].values
        self._price_mean = np.mean(prices)
        self._price_std = np.std(prices)
        
        if self._price_std == 0:
            self._price_std = 1.0
            
        logger.info(f"   Price normalization: mean={self._price_mean:.5f}, std={self._price_std:.5f}")
        
    def _normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """Normalize price series to zero mean, unit variance"""
        return (prices - self._price_mean) / self._price_std
        
    def _get_market_observation(self) -> np.ndarray:
        """Get 256-length price series for WST processing"""
        start_idx = self.current_step
        end_idx = self.current_step + self.price_series_length
        
        prices = self.df['close'].iloc[start_idx:end_idx].values
        normalized_prices = self._normalize_prices(prices)
        
        return normalized_prices.astype(np.float32)
        
    def _find_valid_session_starts(self, session_length: int, max_start: int) -> List[int]:
        """Find valid session starting points (no weekends, no gaps)"""
        valid_starts = []

        # Check each potential starting point
        for start_idx in range(0, max_start, 60):  # Check every hour for efficiency
            # Get session data
            end_idx = start_idx + session_length + self.price_series_length

            if end_idx >= len(self.df):
                continue

            session_data = self.df.iloc[start_idx:end_idx]

            # Check for weekend (Friday 21:00 to Sunday 21:00 GMT)
            has_weekend = False
            if hasattr(session_data.index[0], 'weekday'):
                # Index is datetime
                for timestamp in session_data.index:
                    weekday = timestamp.weekday()
                    hour = timestamp.hour
                    # Friday after 21:00 or Saturday or Sunday before 21:00
                    if (weekday == 4 and hour >= 21) or weekday == 5 or (weekday == 6 and hour < 21):
                        has_weekend = True
                        break

            if has_weekend:
                continue

            # Check for time gaps (more than 10 minutes between consecutive bars)
            if hasattr(session_data.index[0], 'to_pydatetime'):
                # Index is datetime - check for gaps
                time_diffs = session_data.index.to_series().diff()
                max_gap = time_diffs.max()
                if max_gap > pd.Timedelta(minutes=10):
                    continue

            # This is a valid session start
            valid_starts.append(start_idx)

        return valid_starts

    def _get_position_features(self) -> np.ndarray:
        """Get 9D position feature vector"""
        features = np.zeros(9, dtype=np.float32)
        
        # 1. Position side
        if self.position.is_long:
            features[0] = 1.0
        elif self.position.is_short:
            features[0] = -1.0
        else:
            features[0] = 0.0
            
        # 2. Duration in trade (normalized)
        features[1] = min(self.position.duration_bars / 720.0, 1.0)
        
        # 3. Unrealized PnL (normalized)
        features[2] = self.position.unrealized_pnl_pips / 100.0
        
        # 4. Entry price relative to current
        current_price = self.df['close'].iloc[self.current_step + self.price_series_length]
        if self.position.entry_price > 0:
            features[3] = (self.position.entry_price - current_price) / current_price
        else:
            features[3] = 0.0
            
        # 5. Recent price change
        if self.current_step > 0:
            prev_price = self.df['close'].iloc[self.current_step + self.price_series_length - 1]
            price_change_pips = (current_price - prev_price) / self.pip_value
            features[4] = price_change_pips / 50.0
        else:
            features[4] = 0.0
            
        # 6. Max drawdown
        features[5] = self.position.max_drawdown_pips / 50.0
        
        # 7. Accumulated drawdown
        features[6] = self.position.accumulated_drawdown_pips / 100.0
        
        # 8. Bars since max drawdown
        features[7] = min(self.position.bars_since_max_drawdown / 60.0, 1.0)
        
        # 9. Risk flags
        risk_score = 0.0
        if self.position.near_stop_loss:
            risk_score += 0.3
        if self.position.near_take_profit:
            risk_score += 0.3
        if self.position.high_drawdown:
            risk_score += 0.4
        features[8] = min(risk_score, 1.0)
        
        return features
        
    def _update_position_state(self):
        """Update position state with current market data"""
        if not (self.position.is_long or self.position.is_short):
            return
            
        current_price = self.df['close'].iloc[self.current_step + self.price_series_length]
        
        # Update duration
        self.position.duration_bars += 1
        
        # Calculate unrealized PnL (spread was applied at entry as initial negative PnL)
        if self.position.is_long:
            market_pnl_pips = (current_price - self.position.entry_price) / self.pip_value
        else:  # short
            market_pnl_pips = (self.position.entry_price - current_price) / self.pip_value
            
        # Total PnL = market movement - spread cost (applied once)
        self.position.unrealized_pnl_pips = market_pnl_pips - self.spread_pips
        
        # Update drawdown tracking
        if self.position.unrealized_pnl_pips < 0:
            drawdown_pips = abs(self.position.unrealized_pnl_pips)
            self.position.accumulated_drawdown_pips += drawdown_pips
            
            if drawdown_pips > self.position.max_drawdown_pips:
                self.position.max_drawdown_pips = drawdown_pips
                self.position.bars_since_max_drawdown = 0
            else:
                self.position.bars_since_max_drawdown += 1
        else:
            self.position.bars_since_max_drawdown += 1
            
        # Update risk flags
        self.position.high_drawdown = self.position.max_drawdown_pips > 20.0
        self.position.near_stop_loss = self.position.unrealized_pnl_pips < -15.0
        self.position.near_take_profit = self.position.unrealized_pnl_pips > 15.0
        
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate step reward using FIXED reward structure
        
        FIXED Delta AMDDP1 Structure (proper credit assignment):
        - All actions: Delta AMDDP1 (change in trade value)
        - No more +1.0 decision rewards (prevents learning deception)
        - Proper credit flows to all trade-contributing actions
        """
        
        if self.reward_type == 'pure_pips':
            # Only reward on trade completion
            if action == SWTAction.CLOSE and (self.position.is_long or self.position.is_short):
                return self.position.unrealized_pnl_pips
            else:
                return 0.0
                
        elif self.reward_type in ['amddp5', 'amddp1']:
            # FIXED: Delta AMDDP1 reward structure
            
            # Calculate current AMDDP1
            current_amddp1 = self._get_current_amddp1()
            
            # Calculate delta reward (change from previous step)
            if not hasattr(self, '_prev_amddp1'):
                self._prev_amddp1 = 0.0  # Initialize
            
            delta_reward = current_amddp1 - self._prev_amddp1
            self._prev_amddp1 = current_amddp1
            
            # Add action costs for realism
            if action == SWTAction.BUY or action == SWTAction.SELL:
                delta_reward -= 0.1  # Entry spread cost
            elif action == SWTAction.CLOSE:
                delta_reward -= 0.1  # Exit spread cost
            
            return delta_reward
            
        else:
            return 0.0  # Unknown reward type
    
    def _get_current_amddp1(self) -> float:
        """Get current AMDDP1 value for delta calculation"""
        if not (self.position.is_long or self.position.is_short):
            return 0.0  # Flat position
        
        current_pnl = self.position.unrealized_pnl_pips
        penalty = 0.01 if self.reward_type == 'amddp1' else self.drawdown_penalty
        amddp1_value = current_pnl - (penalty * self.position.accumulated_drawdown_pips)
        
        # Apply profit protection
        if self.profit_protection and current_pnl > 0 and amddp1_value < 0:
            amddp1_value = self.min_protected_reward
            
        return amddp1_value
    
    def force_close_position(self) -> Tuple[float, Dict[str, Any]]:
        """Force close open position at episode end for proper reward calculation"""
        if not (self.position.is_long or self.position.is_short):
            return 0.0, {'completed_trades': []}
        
        # Execute forced close
        reward, trade_completed, action_info = self._execute_action(SWTAction.CLOSE)
        
        info = {
            'action_info': 'forced_close_episode_end',
            'completed_trades': self.completed_trades[-1:] if trade_completed else [],
            'forced_close': True
        }
        
        return reward, info
            
    def _execute_action(self, action: int) -> Tuple[float, bool, str]:
        """Execute trading action"""
        
        current_price = self.df['close'].iloc[self.current_step + self.price_series_length]
        reward = 0.0
        trade_completed = False
        info = ""
        
        if action == SWTAction.HOLD:
            # Do nothing, just update position state
            info = "hold"
            
        elif action == SWTAction.BUY:
            if not (self.position.is_long or self.position.is_short):
                # Open long position
                self.position.is_long = True
                self.position.entry_price = current_price  # Use market price
                self.position.entry_time = self.df['timestamp'].iloc[self.current_step + self.price_series_length]
                self.position.duration_bars = 0
                # Initial PnL will be calculated in _update_position_state()
                self.position.max_drawdown_pips = 0.0
                self.position.accumulated_drawdown_pips = 0.0
                self.position.bars_since_max_drawdown = 0
                info = "opened_long"
            else:
                info = "buy_ignored_position_open"
                
        elif action == SWTAction.SELL:
            if not (self.position.is_long or self.position.is_short):
                # Open short position
                self.position.is_short = True
                self.position.entry_price = current_price  # Use market price
                self.position.entry_time = self.df['timestamp'].iloc[self.current_step + self.price_series_length]
                self.position.duration_bars = 0
                # Initial PnL will be calculated in _update_position_state()
                self.position.max_drawdown_pips = 0.0
                self.position.accumulated_drawdown_pips = 0.0
                self.position.bars_since_max_drawdown = 0
                info = "opened_short"
            else:
                info = "sell_ignored_position_open"
                
        elif action == SWTAction.CLOSE:
            if self.position.is_long or self.position.is_short:
                # Close position
                exit_time = self.df['timestamp'].iloc[self.current_step + self.price_series_length]
                
                # Calculate final PnL (spread already applied at entry, no additional spread needed)
                if self.position.is_long:
                    exit_price = current_price  # Use market price
                    pnl_pips = (exit_price - self.position.entry_price) / self.pip_value
                    direction = 1
                else:  # short
                    exit_price = current_price  # Use market price
                    pnl_pips = (self.position.entry_price - exit_price) / self.pip_value
                    direction = -1
                    
                # Calculate AMDDP5 reward for completed trade
                if self.reward_type == 'amddp5':
                    reward_amddp5 = pnl_pips - self.drawdown_penalty * self.position.accumulated_drawdown_pips
                    if self.profit_protection and pnl_pips > 0 and reward_amddp5 < 0:
                        reward_amddp5 = self.min_protected_reward
                else:
                    reward_amddp5 = pnl_pips
                    
                # Calculate AMDDP1 reward (always calculated for comparison)
                # AMDDP1 uses 1% (0.01) penalty regardless of current reward_type
                reward_amddp1 = pnl_pips - (0.01 * self.position.accumulated_drawdown_pips)
                if self.profit_protection and pnl_pips > 0 and reward_amddp1 < 0:
                    reward_amddp1 = self.min_protected_reward
                    
                # Create trade record
                trade_record = SWTTradeRecord(
                    entry_time=self.position.entry_time,
                    exit_time=exit_time,
                    entry_price=self.position.entry_price,
                    exit_price=exit_price,
                    direction=direction,
                    pnl_pips=pnl_pips,
                    duration_bars=self.position.duration_bars,
                    max_drawdown_pips=self.position.max_drawdown_pips,
                    accumulated_drawdown_pips=self.position.accumulated_drawdown_pips,
                    reward_amddp5=reward_amddp5,
                    reward_amddp1=reward_amddp1,
                    exit_reason="manual_close"
                )
                
                self.completed_trades.append(trade_record)
                self.total_pnl_pips += pnl_pips
                trade_completed = True
                
                # Calculate rolling expectancy and log trade completion
                self._log_trade_completion(trade_record)
                
                # Reset position
                self.position = SWTPositionState()
                
                info = f"closed_position_pnl_{pnl_pips:.2f}"
                
            else:
                info = "close_ignored_no_position"
                
        # Calculate step reward
        reward = self._calculate_reward(action)
        self.total_reward += reward
        
        return reward, trade_completed, info
        
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute environment step"""
        
        # Execute action
        reward, trade_completed, action_info = self._execute_action(action)
        
        # Update position state if position is open
        self._update_position_state()
        
        # Move to next timestep
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.total_steps
        
        # Get next observation
        if not done:
            obs = {
                'market_prices': self._get_market_observation(),
                'position_features': self._get_position_features()
            }
        else:
            # Return zeros for terminal state
            obs = {
                'market_prices': np.zeros(self.price_series_length, dtype=np.float32),
                'position_features': np.zeros(9, dtype=np.float32)
            }
        
        # Info dictionary
        info = {
            'action_info': action_info,
            'trade_completed': trade_completed,
            'current_step': self.current_step,
            'total_trades': len(self.completed_trades),
            'total_pnl_pips': self.total_pnl_pips,
            'position_open': self.position.is_long or self.position.is_short,
            'position_pnl': self.position.unrealized_pnl_pips if (self.position.is_long or self.position.is_short) else 0.0
        }
        
        return obs, reward, done, False, info
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to random 6-hour session"""

        if seed is not None:
            np.random.seed(seed)

        # Select random 6-hour session (360 minutes)
        session_length = 360  # 6 hours in minutes
        max_start = self.total_steps - session_length - self.price_series_length

        if max_start <= 0:
            # Not enough data for random selection
            self.current_step = 0
        else:
            # Find valid sessions (skip weekends and gaps)
            valid_starts = self._find_valid_session_starts(session_length, max_start)
            if valid_starts:
                # Randomly select from valid sessions
                self.current_step = np.random.choice(valid_starts)
            else:
                # Fallback to beginning if no valid sessions
                self.current_step = 0
        self.position = SWTPositionState()
        self.completed_trades = []
        self.total_pnl_pips = 0.0
        self.total_reward = 0.0
        
        # FIXED: Initialize delta AMDDP1 tracking
        self._prev_amddp1 = 0.0
        
        # Get initial observation
        obs = {
            'market_prices': self._get_market_observation(),
            'position_features': self._get_position_features()
        }
        
        info = {
            'total_steps': self.total_steps,
            'price_series_length': self.price_series_length
        }
        
        return obs, info
        
    def get_trade_statistics(self) -> Dict[str, float]:
        """Get trading performance statistics"""
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl_pips': 0.0,
                'total_pnl_pips': 0.0,
                'max_trade_pips': 0.0,
                'min_trade_pips': 0.0
            }
            
        pnls = [trade.pnl_pips for trade in self.completed_trades]
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        
        return {
            'total_trades': len(self.completed_trades),
            'win_rate': winning_trades / len(self.completed_trades) * 100,
            'avg_pnl_pips': np.mean(pnls),
            'total_pnl_pips': sum(pnls),
            'max_trade_pips': max(pnls) if pnls else 0.0,
            'min_trade_pips': min(pnls) if pnls else 0.0
        }
    
    def _log_trade_completion(self, trade_record: SWTTradeRecord) -> None:
        """Log trade completion with rolling expectancy and AMDDP1 vs PNL comparison"""
        import logging
        logger = logging.getLogger(__name__)
        
        total_trades = len(self.completed_trades)
        pnl = trade_record.pnl_pips
        amddp1_reward = trade_record.reward_amddp1
        duration = trade_record.duration_bars
        
        # Calculate rolling statistics
        all_pnls = [t.pnl_pips for t in self.completed_trades]
        all_durations = [t.duration_bars for t in self.completed_trades]
        winning_trades = [p for p in all_pnls if p > 0]
        losing_trades = [p for p in all_pnls if p < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss) if total_trades > 0 else 0
        avg_duration = np.mean(all_durations) if all_durations else 0
        max_duration = max(all_durations) if all_durations else 0
        
        # Log every 10th trade or significant trades
        if total_trades % 10 == 0 or abs(pnl) > 20 or duration > 100:
            reward_diff = amddp1_reward - pnl
            logger.info(f"💼 TRADE #{total_trades}: PNL {pnl:+.1f} → AMDDP1 {amddp1_reward:+.2f} (Δ{reward_diff:+.2f}) | "
                       f"⏱️ {duration}bars | 📊 E[{expectancy:+.2f}], Avg/Max Duration: {avg_duration:.0f}/{max_duration}bars")


def create_swt_forex_environment(data_path: str = None, config_dict: dict = None, session_data: pd.DataFrame = None, **kwargs) -> SWTForexEnvironment:
    """Factory function to create SWT forex environment"""
    return SWTForexEnvironment(data_path=data_path, config_dict=config_dict, session_data=session_data, **kwargs)


def test_swt_forex_environment():
    """Test SWT forex environment"""
    
    logger.info("🧪 Testing SWT Forex Environment")
    
    # Create dummy data for testing
    dates = pd.date_range('2023-01-01', periods=1000, freq='5T')
    prices = 1.5 + np.cumsum(np.random.randn(1000) * 0.0001)
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })
    
    test_path = Path("/tmp/test_forex_data.csv")
    test_data.to_csv(test_path, index=False)
    
    try:
        # Create environment
        env = create_swt_forex_environment(str(test_path))
        
        # Test reset
        obs, info = env.reset()
        logger.info(f"   Initial observation shapes:")
        logger.info(f"     market_prices: {obs['market_prices'].shape}")
        logger.info(f"     position_features: {obs['position_features'].shape}")
        
        # Test steps
        total_reward = 0.0
        for i in range(20):
            action = np.random.randint(0, 4)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if info.get('trade_completed', False):
                logger.info(f"   Trade completed: {info['action_info']}")
                
            if done:
                break
                
        # Test statistics
        stats = env.get_trade_statistics()
        logger.info(f"   Final statistics: {stats}")
        
        logger.info("✅ SWT Forex Environment tests passed!")
        
    finally:
        # Cleanup
        if test_path.exists():
            test_path.unlink()
            
    return env


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_swt_forex_environment()