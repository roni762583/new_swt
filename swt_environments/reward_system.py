"""
AMDDP Reward System
Accumulated Maximum Drawdown Penalty reward calculation for trading environments

Implements the AMDDP1 reward system with proper credit assignment for all actions
that contribute to profitable trades while penalizing excessive risk exposure.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from swt_core.types import PositionType
from swt_core.exceptions import RewardCalculationError

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Reward calculation types"""
    AMDDP1 = "amddp1"  # 1% drawdown penalty
    AMDDP5 = "amddp5"  # 5% drawdown penalty
    PNL_ONLY = "pnl_only"  # Pure PnL without penalty


@dataclass
class RewardResult:
    """Result from reward calculation"""
    total_reward: float
    profit_reward: float
    amddp_penalty: float
    cost_penalty: float
    position_value: float
    current_drawdown: float
    reward_type: RewardType
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate reward result"""
        if not np.isfinite(self.total_reward):
            raise ValueError(f"Invalid total_reward: {self.total_reward}")
        if not np.isfinite(self.profit_reward):
            raise ValueError(f"Invalid profit_reward: {self.profit_reward}")


@dataclass
class AMDDPState:
    """Internal state for AMDDP calculation"""
    current_value: float = 0.0
    previous_value: float = 0.0
    max_value: float = 0.0
    accumulated_drawdown: float = 0.0
    bars_since_max_drawdown: int = 0
    total_trades: int = 0
    profitable_trades: int = 0


class AMDDPRewardSystem:
    """
    AMDDP (Accumulated Maximum Drawdown Penalty) Reward System
    
    Calculates delta rewards based on change in position value with drawdown penalty.
    Provides proper credit assignment for all actions contributing to trades.
    """
    
    def __init__(self, 
                 reward_type: RewardType = RewardType.AMDDP1,
                 lookback_periods: int = 100,
                 penalty_factor: float = 0.01,
                 profit_protection: bool = True,
                 min_protected_reward: float = 0.01):
        """
        Initialize AMDDP reward system
        
        Args:
            reward_type: Type of reward calculation
            lookback_periods: Number of periods for drawdown calculation
            penalty_factor: Drawdown penalty factor (0.01 for AMDDP1, 0.05 for AMDDP5)
            profit_protection: Whether to protect profitable trades from negative rewards
            min_protected_reward: Minimum reward for protected profitable trades
        """
        self.reward_type = reward_type
        self.lookback_periods = lookback_periods
        self.penalty_factor = penalty_factor
        self.profit_protection = profit_protection
        self.min_protected_reward = min_protected_reward
        
        # Internal state
        self.state = AMDDPState()
        self.equity_history: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        logger.info(f"💰 AMDDPRewardSystem initialized: {reward_type.value}, penalty={penalty_factor}")
    
    def calculate_reward(self, 
                        current_equity: float,
                        realized_pnl: float = 0.0,
                        unrealized_pnl: float = 0.0,
                        transaction_cost: float = 0.0,
                        current_drawdown: Optional[float] = None) -> RewardResult:
        """
        Calculate AMDDP reward for current step
        
        Args:
            current_equity: Current total equity
            realized_pnl: Realized profit/loss for this step
            unrealized_pnl: Current unrealized profit/loss
            transaction_cost: Transaction costs incurred
            current_drawdown: Pre-calculated drawdown (optional)
            
        Returns:
            RewardResult with detailed reward breakdown
        """
        try:
            # Update equity history
            self.equity_history.append(current_equity)
            if len(self.equity_history) > self.lookback_periods:
                self.equity_history.pop(0)
            
            # Calculate current position value
            position_value = realized_pnl + unrealized_pnl
            
            # Calculate or use provided drawdown
            if current_drawdown is not None:
                calculated_drawdown = current_drawdown
            else:
                calculated_drawdown = self._calculate_drawdown()
            
            # Calculate reward components
            profit_reward = self._calculate_profit_reward(position_value)
            amddp_penalty = self._calculate_amddp_penalty(calculated_drawdown)
            cost_penalty = transaction_cost
            
            # Calculate total reward based on type
            if self.reward_type == RewardType.PNL_ONLY:
                total_reward = profit_reward - cost_penalty
            else:
                # AMDDP1 or AMDDP5
                total_reward = profit_reward - amddp_penalty - cost_penalty
                
                # Apply profit protection if enabled
                if (self.profit_protection and 
                    realized_pnl > 0 and 
                    total_reward < 0):
                    total_reward = self.min_protected_reward
                    logger.debug(f"💼 Profit protection applied: {realized_pnl:.2f} PnL protected")
            
            # Update internal state
            self._update_state(current_equity, total_reward, realized_pnl)
            
            # Create result
            result = RewardResult(
                total_reward=float(total_reward),
                profit_reward=float(profit_reward),
                amddp_penalty=float(amddp_penalty),
                cost_penalty=float(cost_penalty),
                position_value=float(position_value),
                current_drawdown=float(calculated_drawdown),
                reward_type=self.reward_type,
                metadata={
                    "equity": current_equity,
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "accumulated_drawdown": self.state.accumulated_drawdown,
                    "bars_since_max_dd": self.state.bars_since_max_drawdown,
                    "total_trades": self.state.total_trades,
                    "profitable_trades": self.state.profitable_trades,
                    "win_rate": self.state.profitable_trades / max(1, self.state.total_trades)
                }
            )
            
            return result
            
        except Exception as e:
            raise RewardCalculationError(
                f"AMDDP reward calculation failed: {str(e)}",
                context={
                    "current_equity": current_equity,
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "reward_type": self.reward_type.value
                },
                original_error=e
            )
    
    def _calculate_profit_reward(self, position_value: float) -> float:
        """Calculate profit component of reward (delta AMDDP)"""
        current_amddp = position_value - (self.penalty_factor * self.state.accumulated_drawdown)
        delta_reward = current_amddp - self.state.previous_value
        
        # Update for next calculation
        self.state.previous_value = current_amddp
        
        return delta_reward
    
    def _calculate_amddp_penalty(self, current_drawdown: float) -> float:
        """Calculate AMDDP penalty component"""
        # Update accumulated drawdown
        if current_drawdown > 0:
            self.state.accumulated_drawdown += current_drawdown
            self.state.bars_since_max_drawdown = 0
        else:
            self.state.bars_since_max_drawdown += 1
        
        # Calculate penalty (already incorporated in delta calculation)
        return self.penalty_factor * current_drawdown
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from equity history"""
        if len(self.equity_history) < 2:
            return 0.0
        
        # Find maximum equity in recent history
        max_equity = max(self.equity_history)
        current_equity = self.equity_history[-1]
        
        # Calculate drawdown as percentage
        if max_equity > 0:
            drawdown = max(0, (max_equity - current_equity) / max_equity)
        else:
            drawdown = 0.0
        
        # Update internal state
        if current_equity > self.state.max_value:
            self.state.max_value = current_equity
            self.state.bars_since_max_drawdown = 0
        
        return drawdown
    
    def _update_state(self, current_equity: float, total_reward: float, realized_pnl: float) -> None:
        """Update internal reward system state"""
        self.state.current_value = current_equity
        
        # Track trade completion
        if realized_pnl != 0.0:
            self.state.total_trades += 1
            if realized_pnl > 0:
                self.state.profitable_trades += 1
            
            # Record trade
            trade_record = {
                "pnl": realized_pnl,
                "reward": total_reward,
                "drawdown": self.state.accumulated_drawdown,
                "bars_held": self.state.bars_since_max_drawdown,
                "equity": current_equity
            }
            self.trade_history.append(trade_record)
            
            # Maintain trade history size
            if len(self.trade_history) > 1000:
                self.trade_history.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward system statistics"""
        if not self.trade_history:
            return {"status": "no_trades"}
        
        trades_data = np.array([trade["pnl"] for trade in self.trade_history])
        rewards_data = np.array([trade["reward"] for trade in self.trade_history])
        
        return {
            "reward_type": self.reward_type.value,
            "penalty_factor": self.penalty_factor,
            "total_trades": self.state.total_trades,
            "profitable_trades": self.state.profitable_trades,
            "win_rate": self.state.profitable_trades / max(1, self.state.total_trades),
            "current_equity": self.state.current_value,
            "max_equity": self.state.max_value,
            "accumulated_drawdown": self.state.accumulated_drawdown,
            "bars_since_max_dd": self.state.bars_since_max_drawdown,
            "trade_statistics": {
                "total_pnl": float(np.sum(trades_data)),
                "average_pnl": float(np.mean(trades_data)),
                "total_reward": float(np.sum(rewards_data)),
                "average_reward": float(np.mean(rewards_data)),
                "pnl_std": float(np.std(trades_data)),
                "reward_std": float(np.std(rewards_data)),
                "max_pnl": float(np.max(trades_data)),
                "min_pnl": float(np.min(trades_data)),
                "reward_efficiency": float(np.sum(rewards_data) / max(1, abs(np.sum(trades_data))))
            },
            "equity_curve": {
                "current": self.state.current_value,
                "max": self.state.max_value,
                "history_length": len(self.equity_history),
                "recent_high": float(max(self.equity_history[-20:])) if len(self.equity_history) >= 20 else self.state.current_value
            }
        }
    
    def reset(self) -> None:
        """Reset reward system state"""
        self.state = AMDDPState()
        self.equity_history.clear()
        # Keep trade history for analysis but reset active state
        logger.info("🔄 AMDDP reward system reset")
    
    def update_config(self, 
                     reward_type: Optional[RewardType] = None,
                     penalty_factor: Optional[float] = None,
                     profit_protection: Optional[bool] = None) -> None:
        """Update reward system configuration"""
        if reward_type is not None:
            old_type = self.reward_type
            self.reward_type = reward_type
            logger.info(f"🔧 Reward type changed: {old_type.value} → {reward_type.value}")
        
        if penalty_factor is not None:
            old_factor = self.penalty_factor
            self.penalty_factor = penalty_factor
            logger.info(f"🔧 Penalty factor changed: {old_factor} → {penalty_factor}")
        
        if profit_protection is not None:
            self.profit_protection = profit_protection
            logger.info(f"🔧 Profit protection: {'enabled' if profit_protection else 'disabled'}")
    
    def get_expectancy(self) -> Dict[str, float]:
        """Calculate trading expectancy metrics"""
        if not self.trade_history:
            return {"expectancy": 0.0, "samples": 0}
        
        pnl_values = [trade["pnl"] for trade in self.trade_history]
        reward_values = [trade["reward"] for trade in self.trade_history]
        
        # Calculate expectancy (expected value per trade)
        pnl_expectancy = np.mean(pnl_values)
        reward_expectancy = np.mean(reward_values)
        
        # Calculate win rate and average win/loss
        wins = [pnl for pnl in pnl_values if pnl > 0]
        losses = [abs(pnl) for pnl in pnl_values if pnl < 0]
        
        win_rate = len(wins) / len(pnl_values)
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Expectancy formula: (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            "expectancy": float(expectancy),
            "reward_expectancy": float(reward_expectancy),
            "win_rate": float(win_rate),
            "average_win": float(avg_win),
            "average_loss": float(avg_loss),
            "profit_factor": float(avg_win / max(0.01, avg_loss)),
            "samples": len(pnl_values),
            "total_pnl": float(sum(pnl_values)),
            "total_reward": float(sum(reward_values))
        }
    
    def analyze_drawdown_penalty_impact(self) -> Dict[str, Any]:
        """Analyze impact of drawdown penalty on rewards"""
        if not self.trade_history:
            return {"status": "no_data"}
        
        pnl_sum = sum(trade["pnl"] for trade in self.trade_history)
        reward_sum = sum(trade["reward"] for trade in self.trade_history)
        penalty_impact = pnl_sum - reward_sum
        
        return {
            "total_pnl": pnl_sum,
            "total_reward": reward_sum,
            "penalty_impact": penalty_impact,
            "penalty_percentage": (penalty_impact / max(abs(pnl_sum), 0.01)) * 100,
            "reward_efficiency": reward_sum / max(abs(pnl_sum), 0.01),
            "current_accumulated_drawdown": self.state.accumulated_drawdown,
            "penalty_factor": self.penalty_factor,
            "analysis": {
                "penalty_reduces_reward": penalty_impact > 0,
                "significant_impact": abs(penalty_impact) > abs(pnl_sum) * 0.1,
                "drawdown_controlled": self.state.accumulated_drawdown < 50.0  # 50 pips threshold
            }
        }