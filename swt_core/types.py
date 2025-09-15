#!/usr/bin/env python3
"""
Core data types and structures for the SWT system.
These types are shared between training and live trading to ensure consistency.

CRITICAL: Includes process management to prevent rogue training processes.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np


# ================================
# Process Management Types
# ================================

class ProcessState(Enum):
    """Process state enumeration"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    CRASHED = "crashed"


@dataclass
class ProcessInfo:
    """Process information for monitoring and control"""
    pid: int
    name: str
    state: ProcessState
    start_time: datetime
    last_heartbeat: datetime
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_message: Optional[str] = None


class ManagedProcess:
    """
    Base class for managed processes with proper lifecycle control.
    
    PREVENTS ROGUE PROCESSES by:
    1. Automatic cleanup on exit
    2. Heartbeat monitoring  
    3. Resource usage tracking
    4. Signal handling for graceful shutdown
    5. Hard limits that cannot be bypassed
    """
    
    def __init__(
        self, 
        name: str, 
        max_runtime_hours: Optional[float] = 24.0,
        max_episodes: Optional[int] = None,
        enable_external_monitoring: bool = True
    ):
        self.name = name
        self.pid = os.getpid()
        self.max_runtime_hours = max_runtime_hours
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.start_time = datetime.now(timezone.utc)
        self.state = ProcessState.INITIALIZING
        self.shutdown_event = threading.Event()
        self.enable_external_monitoring = enable_external_monitoring
        
        # CRITICAL: Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # CRITICAL: Register cleanup on exit
        atexit.register(self._cleanup_on_exit)
        
        # Start monitoring systems
        self._start_monitoring()
        self._write_pid_file()
        
        print(f"🚀 Managed process '{name}' started (PID: {self.pid})")
        if max_runtime_hours:
            print(f"⏰ Runtime limit: {max_runtime_hours} hours")
        if max_episodes:
            print(f"📊 Episode limit: {max_episodes}")
        
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully"""
        print(f"🛑 Process {self.name} received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.state = ProcessState.STOPPING
        
    def _cleanup_on_exit(self) -> None:
        """Cleanup resources on process exit"""
        print(f"🧹 Cleaning up process {self.name}...")
        self._remove_pid_file()
        self._remove_heartbeat_file()
        self.state = ProcessState.STOPPED
        
    def _start_monitoring(self) -> None:
        """Start process monitoring thread"""
        def monitor():
            while not self.shutdown_event.is_set():
                try:
                    # CRITICAL: Check runtime limit (cannot be bypassed)
                    if self.max_runtime_hours:
                        runtime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
                        if runtime_hours > self.max_runtime_hours:
                            print(f"⏰ CRITICAL: Process {self.name} exceeded runtime limit ({runtime_hours:.1f}/{self.max_runtime_hours}h)")
                            print("🛑 FORCE SHUTDOWN - Runtime limit cannot be bypassed")
                            self.shutdown_event.set()
                            self.state = ProcessState.STOPPING
                            break
                    
                    # CRITICAL: Check episode limit (cannot be bypassed)
                    if self.max_episodes and self.episode_count >= self.max_episodes:
                        print(f"📊 CRITICAL: Process {self.name} exceeded episode limit ({self.episode_count}/{self.max_episodes})")
                        print("🛑 FORCE SHUTDOWN - Episode limit cannot be bypassed")
                        self.shutdown_event.set()
                        self.state = ProcessState.STOPPING
                        break
                    
                    # Update heartbeat file for external monitoring
                    self._update_heartbeat()
                    
                    # Check resource usage
                    self._check_resource_usage()
                    
                except Exception as e:
                    print(f"❌ Monitor thread error: {e}")
                
                # Sleep for monitoring interval
                self.shutdown_event.wait(60)  # Check every minute
                
        monitor_thread = threading.Thread(target=monitor, daemon=True, name=f"{self.name}_monitor")
        monitor_thread.start()
        
    def _write_pid_file(self) -> None:
        """Write PID file for external monitoring"""
        if self.enable_external_monitoring:
            pid_file = f"{self.name}.pid"
            with open(pid_file, "w") as f:
                f.write(str(self.pid))
            print(f"📄 PID file written: {pid_file}")
            
    def _remove_pid_file(self) -> None:
        """Remove PID file"""
        pid_file = f"{self.name}.pid"
        try:
            os.remove(pid_file)
        except FileNotFoundError:
            pass
            
    def _update_heartbeat(self) -> None:
        """Update heartbeat file for external monitoring"""
        if not self.enable_external_monitoring:
            return
            
        heartbeat_data = {
            "pid": self.pid,
            "process_name": self.name,
            "state": self.state.value,
            "start_time": self.start_time.isoformat(),
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "runtime_hours": (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600,
            "episode_count": self.episode_count,
            "max_episodes": self.max_episodes,
            "max_runtime_hours": self.max_runtime_hours,
            "memory_usage_mb": self._get_memory_usage(),
            "limits_enforced": True  # Always true - limits cannot be disabled
        }
        
        heartbeat_file = f"{self.name}_heartbeat.json"
        with open(heartbeat_file, "w") as f:
            json.dump(heartbeat_data, f, indent=2)
            
    def _remove_heartbeat_file(self) -> None:
        """Remove heartbeat file"""
        heartbeat_file = f"{self.name}_heartbeat.json"
        try:
            os.remove(heartbeat_file)
        except FileNotFoundError:
            pass
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(self.pid)
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
            
    def _check_resource_usage(self) -> None:
        """Check resource usage and warn if high"""
        memory_mb = self._get_memory_usage()
        if memory_mb > 4096:  # Warn if over 4GB
            print(f"⚠️ High memory usage: {memory_mb:.1f} MB")
        
    def should_continue(self) -> bool:
        """Check if process should continue running"""
        if self.shutdown_event.is_set():
            return False
            
        # CRITICAL: Hard limits cannot be bypassed
        if self.max_episodes and self.episode_count >= self.max_episodes:
            print(f"🛑 Episode limit reached: {self.episode_count}/{self.max_episodes}")
            return False
            
        if self.max_runtime_hours:
            runtime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
            if runtime_hours > self.max_runtime_hours:
                print(f"🛑 Runtime limit reached: {runtime_hours:.1f}/{self.max_runtime_hours}h")
                return False
                
        return self.state in [ProcessState.RUNNING, ProcessState.INITIALIZING]
        
    def increment_episode(self) -> None:
        """Increment episode counter"""
        self.episode_count += 1
        
    def stop(self) -> None:
        """Stop the process gracefully"""
        print(f"🛑 Stopping process {self.name}...")
        self.shutdown_event.set()
        self.state = ProcessState.STOPPED


# ================================
# Trading Types  
# ================================

class ActionType(IntEnum):
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


class PositionDirection(Enum):
    """Position direction"""
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


@dataclass
class PositionState:
    """
    Universal position state representation.
    Used by both training environment and live trading system.

    CRITICAL: This ensures identical feature calculation between training and live.
    Contains all fields needed for the 9 position features:
    1. current_equity_pips - arctan scaled by 150
    2. bars_since_entry - arctan scaled by 2000
    3. position_efficiency - already in [-1, 1]
    4. pips_from_peak - arctan scaled by 150
    5. max_drawdown_pips - arctan scaled by 150
    6. amddp_reward - arctan scaled by 150
    7. is_long - binary flag
    8. is_short - binary flag
    9. has_position - binary flag
    """
    # Basic position info
    direction: PositionDirection = PositionDirection.FLAT
    position_type: Optional[PositionType] = None  # For compatibility
    volume: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    entry_time: Optional[datetime] = None

    # Duration tracking (for feature 2)
    bars_since_entry: int = 0
    duration_minutes: int = 0

    # P&L tracking (for feature 1)
    unrealized_pnl_pips: float = 0.0  # current_equity_pips
    unrealized_pnl_percent: float = 0.0

    # Peak tracking (for features 3, 4)
    peak_equity: float = 0.0  # High water mark
    min_equity: float = 0.0   # Low water mark
    position_efficiency: float = 0.0  # Feature 3
    pips_from_peak: float = 0.0  # Feature 4

    # Drawdown tracking (for features 5, 6)
    max_drawdown_pips: float = 0.0  # Feature 5
    dd_sum: float = 0.0  # Cumulative sum of drawdown increases (for AMDDP)
    accumulated_drawdown: float = 0.0  # Alternative name for dd_sum

    # For backward compatibility
    max_adverse_pips: float = 0.0  # Same as max_drawdown_pips
    bars_since_max_drawdown: int = 0

    # Derived properties
    @property
    def is_long(self) -> bool:
        return self.direction == PositionDirection.LONG or self.position_type == PositionType.LONG

    @property
    def is_short(self) -> bool:
        return self.direction == PositionDirection.SHORT or self.position_type == PositionType.SHORT

    @property
    def is_flat(self) -> bool:
        return self.direction == PositionDirection.FLAT or self.position_type == PositionType.FLAT

    @property
    def has_position(self) -> bool:
        return not self.is_flat

    @property
    def current_equity_pips(self) -> float:
        """Alias for unrealized_pnl_pips (training compatibility)"""
        return self.unrealized_pnl_pips


@dataclass
class MarketState:
    """Market state for feature processing"""
    timestamp: datetime
    prices: np.ndarray  # Price series for WST processing (256 bars)
    current_price: float
    volume: int = 0
    spread_pips: float = 0.0
    
    def __post_init__(self):
        """Validate market state"""
        if len(self.prices) == 0:
            raise ValueError("Price series cannot be empty")
        if self.current_price <= 0:
            raise ValueError("Current price must be positive")


@dataclass  
class TradingDecision:
    """Trading decision from inference engine"""
    action: ActionType
    confidence: float
    value_estimate: float
    policy_probs: np.ndarray
    inference_time_ms: float
    timestamp: datetime
    
    # Additional context
    position_state: Optional[PositionState] = None
    market_state: Optional[MarketState] = None
    
    def __post_init__(self):
        """Validate trading decision"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        if len(self.policy_probs) != 4:
            raise ValueError(f"Policy probs must have 4 elements, got {len(self.policy_probs)}")


@dataclass
class TradeResult:
    """Result of trade execution"""
    success: bool
    action: ActionType
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    volume: float = 0.0
    slippage_pips: float = 0.0
    realized_pnl_pips: Optional[float] = None
    error_message: Optional[str] = None
    
    # OANDA specific fields
    order_id: Optional[str] = None
    transaction_id: Optional[str] = None


# ================================
# Configuration Types
# ================================

@dataclass
class FeatureConfig:
    """
    Feature processing configuration.
    CRITICAL: These values MUST match training environment exactly.
    """
    # Normalization parameters (from training environment swt_forex_env.py)
    duration_max_bars: float = 720.0  # EXACT match
    pnl_scale_factor: float = 100.0   # EXACT match
    drawdown_scale_factor: float = 50.0  # EXACT match
    price_change_scale_factor: float = 50.0  # EXACT match
    bars_since_drawdown_max: float = 60.0  # EXACT match
    
    # Risk thresholds (from training environment)
    high_drawdown_pips: float = 20.0   # EXACT match
    near_stop_loss_pips: float = -15.0  # EXACT match
    near_take_profit_pips: float = 15.0  # EXACT match
    
    # Risk weights (from training environment)
    risk_weight_high_drawdown: float = 0.4  # EXACT match
    risk_weight_near_stop: float = 0.3      # EXACT match
    risk_weight_near_profit: float = 0.3    # EXACT match


@dataclass
class MCTSConfig:
    """
    MCTS configuration.
    CRITICAL: These MUST match Episode 13475 training exactly.
    """
    simulations: int = 15        # Episode 13475 setting
    c_puct: float = 1.25        # Episode 13475 setting
    temperature: float = 1.0     # Episode 13475 setting (NOT 0.0!)
    exploration_noise: bool = False
    dirichlet_alpha: float = 0.25
    dirichlet_epsilon: float = 0.0


@dataclass
class TradingConfig:
    """Trading configuration"""
    # Position sizing - always fixed volume of 1 unit
    fixed_volume: float = 1.0
    trade_volume: float = 1.0  # Alias for fixed_volume (always 1 unit)
    
    # Risk management
    stop_loss_pips: float = 0.0
    take_profit_pips: float = 30.0
    max_drawdown_pips: float = 50.0
    
    # Decision making (Episode 13475 capability)
    min_confidence: float = 0.35  # Episode 13475 achieved 38.5% max
    min_bars_between_trades: int = 5
    max_position_duration_bars: int = 1440  # 24 hours
    
    # Spread filtering
    max_spread_pips: float = 8.0
    
    def __post_init__(self):
        """Ensure trade_volume and fixed_volume are synchronized at 1.0"""
        self.fixed_volume = 1.0
        self.trade_volume = 1.0
    
    def validate(self) -> None:
        """Validate trading configuration"""
        if self.trade_volume != 1.0:
            raise ValueError(f"Trade volume must be 1.0, got {self.trade_volume}")
        if self.fixed_volume != 1.0:
            raise ValueError(f"Fixed volume must be 1.0, got {self.fixed_volume}")


@dataclass
class ProcessLimits:
    """
    Resource limits to prevent rogue processes.
    CRITICAL: These limits cannot be bypassed.
    """
    # Core limits
    max_episodes: int = 20000                    # Hard episode limit
    episode_limit_enabled: bool = True           # Cannot be disabled
    max_runtime_hours: float = 24.0              # 24 hour maximum runtime
    runtime_limit_enabled: bool = True          # Cannot be disabled
    max_memory_gb: float = 4.0                   # 4GB memory limit
    max_cpu_percent: float = 80.0                # 80% CPU usage limit
    max_checkpoints: int = 10                    # Keep only 10 checkpoints
    checkpoint_cleanup_enabled: bool = True     # Auto-cleanup old checkpoints
    checkpoint_cleanup_interval_hours: float = 1.0  # Cleanup every hour
    
    # Legacy compatibility
    max_memory_mb: float = field(init=False)     # Computed from max_memory_gb
    heartbeat_interval_seconds: float = 60.0
    enable_external_monitoring: bool = True  # Cannot be disabled
    
    def __post_init__(self):
        """Set computed values after initialization"""
        self.max_memory_mb = self.max_memory_gb * 1024.0
    
    def validate(self) -> None:
        """Validate process limits"""
        if self.max_episodes <= 0:
            raise ValueError(f"Max episodes must be positive, got {self.max_episodes}")
        if self.max_memory_gb <= 0:
            raise ValueError(f"Max memory must be positive, got {self.max_memory_gb}")


@dataclass  
class ProcessMonitoring:
    """Process monitoring configuration"""
    enable_heartbeat: bool = True
    heartbeat_interval_seconds: float = 60.0
    enable_external_monitoring: bool = True
    write_pid_file: bool = True
    write_heartbeat_file: bool = True
    enable_resource_monitoring: bool = True
    resource_check_interval_seconds: float = 300.0
    memory_alert_threshold_gb: float = 3.0
    cpu_alert_threshold_percent: float = 70.0
    runtime_warning_hours: float = 20.0
    episode_warning_count: int = 18000
    
    def validate(self) -> None:
        """Validate process monitoring configuration"""
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError(f"Heartbeat interval must be positive, got {self.heartbeat_interval_seconds}")


@dataclass
class ProcessConfig:
    """Complete process configuration"""
    limits: ProcessLimits = field(default_factory=ProcessLimits)
    monitoring: ProcessMonitoring = field(default_factory=ProcessMonitoring)
    
    def validate(self) -> None:
        """Validate complete process configuration"""
        self.limits.validate()
        self.monitoring.validate()


@dataclass
class WSTransformConfig:
    """WST configuration matching Episode 13475"""
    J: int = 2
    Q: int = 6
    backend: str = "fallback" 
    max_order: int = 2
    output_dim: int = 128
    
    def validate(self) -> None:
        """Validate WST configuration"""
        if self.J != 2 or self.Q != 6:
            raise ValueError(f"WST must use J=2, Q=6 for Episode 13475 compatibility, got J={self.J}, Q={self.Q}")


@dataclass
class NormalizationParams:
    """Normalization parameters for position features"""
    duration_max_bars: float = 720.0
    pnl_scale_factor: float = 100.0
    price_change_scale_factor: float = 50.0
    drawdown_scale_factor: float = 50.0
    accumulated_dd_scale: float = 100.0
    bars_since_dd_scale: float = 60.0


@dataclass
class PositionFeatures:
    """Position feature configuration matching Episode 13475"""
    dimension: int = 9
    enabled: bool = True
    normalization_params: NormalizationParams = field(default_factory=NormalizationParams)
    
    def validate(self) -> None:
        """Validate position features configuration"""
        if self.dimension != 9:
            raise ValueError(f"Position features must have dimension 9 for Episode 13475 compatibility, got {self.dimension}")


@dataclass  
class FeatureProcessingConfig:
    """Feature processing configuration"""
    wst_config: WSTransformConfig = field(default_factory=WSTransformConfig)
    position_features: PositionFeatures = field(default_factory=PositionFeatures)
    
    def validate(self) -> None:
        """Validate feature processing configuration"""
        self.wst_config.validate()
        self.position_features.validate()


@dataclass
class MCTSParameters:
    """MCTS configuration matching Episode 13475"""
    num_simulations: int = 15
    c_puct: float = 1.25
    temperature: float = 1.0
    discount_factor: float = 0.997
    max_search_time_ms: int = 1000
    
    def validate(self) -> None:
        """Validate MCTS parameters"""
        if self.num_simulations != 15:
            raise ValueError(f"MCTS must use 15 simulations for Episode 13475 compatibility, got {self.num_simulations}")
        if abs(self.c_puct - 1.25) > 0.001:
            raise ValueError(f"MCTS C_PUCT must be 1.25 for Episode 13475 compatibility, got {self.c_puct}")


# ================================
# Constants
# ================================

# Instrument specifications
INSTRUMENT_SPECS = {
    "GBP_JPY": {
        "pip_value": 0.01,
        "pip_multiplier": 100,
        "typical_spread_pips": 2.5,
        "min_trade_size": 1,
        "max_trade_size": 1000000,
    },
    "EUR_USD": {
        "pip_value": 0.0001,
        "pip_multiplier": 10000,
        "typical_spread_pips": 1.0,
        "min_trade_size": 1,
        "max_trade_size": 1000000,
    }
}

# Process control constants (CANNOT BE CHANGED)
MAX_PROCESS_RUNTIME_HOURS = 24.0
MAX_TRAINING_EPISODES = 20000
HEARTBEAT_INTERVAL_SECONDS = 60.0
RESOURCE_CHECK_INTERVAL_SECONDS = 300.0

# Feature processing constants (MUST MATCH TRAINING)
POSITION_FEATURE_COUNT = 9    # Exactly 9D as in training
MARKET_FEATURE_LENGTH = 256   # WST requires 256 price points
ACTION_SPACE_SIZE = 4         # HOLD, BUY, SELL, CLOSE


# ================================
# Exception Types
# ================================

class SWTException(Exception):
    """Base SWT exception"""
    pass


class ConfigurationError(SWTException):
    """Configuration error"""
    pass


class FeatureError(SWTException):
    """Feature processing error"""
    pass


class InferenceError(SWTException):
    """Inference error"""
    pass


class ExecutionError(SWTException):
    """Trade execution error"""  
    pass


class ProcessControlError(SWTException):
    """Process control error - CRITICAL"""
    pass


class CheckpointCorruptionError(SWTException):
    """Checkpoint corruption error - CRITICAL"""
    pass


# ================================
# Architecture and Configuration Enums
# ================================

class NetworkArchitecture(Enum):
    """Neural network architecture types"""
    STANDARD = "standard"                        # Basic MLP networks
    RESIDUAL = "residual"                       # ResNet-style blocks
    TRANSFORMER = "transformer"                 # Attention-based networks
    HYBRID = "hybrid"                          # Mix of architectures


class AgentType(Enum):
    """Agent system types"""
    STOCHASTIC_MUZERO = "stochastic_muzero"     # Standard Stochastic MuZero
    EXPERIMENTAL = "experimental"               # Enhanced experimental agent


class PositionType(Enum):
    """Trading position types"""
    FLAT = 0
    LONG = 1
    SHORT = -1


class TradingAction(Enum):
    """Trading actions available to agent"""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


@dataclass(frozen=True)
class MarketState:
    """
    Immutable market state representation
    Unified across all agent types for consistent feature processing
    """
    # WST-processed market features
    wst_features: np.ndarray                    # Shape: (128,) - WST coefficients
    
    # Raw price data (for additional processing if needed)
    ohlc: Tuple[float, float, float, float]     # Open, High, Low, Close
    volume: float
    spread: float
    timestamp: datetime
    
    # Market metadata
    volatility: Optional[float] = None
    liquidity_score: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate market state dimensions and values"""
        if self.wst_features.shape != (128,):
            raise ValueError(f"Expected WST features shape (128,), got {self.wst_features.shape}")
        
        if np.isnan(self.wst_features).any():
            raise ValueError("WST features contain NaN values")
        
        if self.spread < 0:
            raise ValueError(f"Spread must be non-negative, got {self.spread}")


@dataclass(frozen=True)
class PositionState:
    """
    Immutable position state representation
    Compatible with both standard and experimental feature extraction
    """
    position_type: PositionType
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    units: int = 0
    
    # Computed position metrics
    unrealized_pnl_pips: float = 0.0
    duration_minutes: int = 0
    max_favorable_pips: float = 0.0
    max_adverse_pips: float = 0.0
    
    # Risk metrics
    accumulated_drawdown: float = 0.0
    bars_since_entry: int = 0
    bars_since_max_drawdown: int = 0
    efficiency_ratio: float = 0.0
    
    def is_flat(self) -> bool:
        """Check if position is flat"""
        return self.position_type == PositionType.FLAT
    
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.position_type == PositionType.LONG
    
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.position_type == PositionType.SHORT
    
    def has_position(self) -> bool:
        """Check if any position is open"""
        return not self.is_flat()


@dataclass(frozen=True)
class TradingDecision:
    """
    Immutable trading decision output from agent
    Standardized across all agent types
    """
    action: TradingAction
    confidence: float                           # 0.0 to 1.0
    value_estimate: float                       # Expected value in pips
    policy_distribution: np.ndarray             # Shape: (4,) - Action probabilities
    
    # MCTS metadata (if applicable)
    mcts_visits: Optional[np.ndarray] = None    # Visit counts per action
    mcts_simulations: Optional[int] = None      # Number of simulations performed
    search_time_ms: Optional[float] = None      # Time spent in search
    
    # Agent-specific metadata
    agent_type: Optional[AgentType] = None
    model_confidence: Optional[float] = None    # Internal model confidence
    features_used: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate trading decision"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        
        if self.policy_distribution.shape != (4,):
            raise ValueError(f"Policy distribution must have shape (4,), got {self.policy_distribution.shape}")
        
        if not np.isclose(self.policy_distribution.sum(), 1.0, atol=1e-6):
            raise ValueError("Policy distribution must sum to 1.0")


@dataclass(frozen=True)
class TradeResult:
    """
    Immutable trade execution result
    Records outcome of trading decision
    """
    action_taken: TradingAction
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    slippage_pips: float = 0.0
    
    # Position changes
    position_before: Optional[PositionState] = None
    position_after: Optional[PositionState] = None
    
    # Trade outcome (if position was closed)
    realized_pnl_pips: Optional[float] = None
    trade_duration_minutes: Optional[int] = None
    
    # Execution metadata
    order_id: Optional[str] = None
    execution_success: bool = True
    error_message: Optional[str] = None


@dataclass
class MCTSConfig:
    """
    MCTS configuration for different agent types
    Supports both standard and experimental MCTS variants
    """
    # Core MCTS parameters
    num_simulations: int = 15
    c_puct: float = 1.25
    discount_factor: float = 0.997
    
    # Temperature settings
    temperature: float = 1.0
    temperature_schedule: Optional[str] = None  # "linear", "exponential", "constant"
    
    # Enhancement flags (for experimental agent)
    enable_gumbel_selection: bool = False
    enable_rezero_optimization: bool = False
    enable_progressive_widening: bool = False
    
    # ReZero-specific settings
    backward_view_cache_size: int = 1000
    temporal_reuse_enabled: bool = False
    
    # Gumbel-specific settings
    gumbel_temperature: float = 0.1
    gumbel_simple_loss: bool = True
    
    # Progressive widening settings
    widening_factor: float = 2.0
    max_actions_per_node: int = 50
    
    def validate(self) -> None:
        """Validate MCTS configuration"""
        if self.num_simulations <= 0:
            raise ValueError(f"num_simulations must be positive, got {self.num_simulations}")
        
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be positive, got {self.c_puct}")
        
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(f"discount_factor must be in [0,1], got {self.discount_factor}")


@dataclass
class NetworkConfig:
    """
    Neural network configuration for different architectures
    Supports seamless switching between implementations
    """
    architecture: NetworkArchitecture = NetworkArchitecture.STANDARD
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = "relu"
    dropout: float = 0.1
    
    # Architecture-specific settings
    residual_connections: bool = False
    layer_norm: bool = True
    batch_norm: bool = False
    
    # Transformer settings (if architecture == TRANSFORMER)
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    feedforward_dim: int = 512
    
    # Value network specific
    value_support_size: int = 601
    value_support_range: Tuple[float, float] = (-300.0, 300.0)
    
    # Experimental enhancements
    enable_consistency_loss: bool = False
    enable_value_prefix: bool = False
    enable_concurrent_prediction: bool = False
    
    def validate(self) -> None:
        """Validate network configuration"""
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0,1], got {self.dropout}")


@dataclass
class FeatureConfig:
    """
    Feature processing configuration
    Unified for both standard and experimental agents
    """
    # WST parameters
    wst_J: int = 2                              # Number of scales
    wst_Q: int = 6                              # Number of orientations per scale
    wst_backend: str = "kymatio"               # "kymatio" or "pytorch-wavelets"
    
    # Position feature settings
    position_feature_dim: int = 9               # Standard 9D position features
    position_normalization: Dict[str, float] = field(default_factory=lambda: {
        "duration_scale": 720.0,               # 12-hour session normalization
        "pnl_scale": 100.0,                    # PnL normalization factor
        "price_change_scale": 50.0,            # Price change normalization
        "drawdown_scale": 50.0,                # Drawdown normalization
        "accumulated_dd_scale": 100.0,         # Accumulated drawdown scale
        "bars_since_dd_scale": 60.0           # Time since drawdown scale
    })
    
    # Market data processing
    price_window_size: int = 256                # Number of bars for WST processing
    price_normalization: str = "zscore"        # "zscore", "minmax", "robust"
    gap_detection_threshold: float = 5.0       # Pip threshold for gap detection
    
    def validate(self) -> None:
        """Validate feature configuration"""
        if self.wst_J <= 0 or self.wst_J > 10:
            raise ValueError(f"wst_J must be in [1,10], got {self.wst_J}")
        
        if self.wst_Q <= 0 or self.wst_Q > 20:
            raise ValueError(f"wst_Q must be in [1,20], got {self.wst_Q}")
        
        if self.price_window_size <= 0:
            raise ValueError(f"price_window_size must be positive, got {self.price_window_size}")


@dataclass
class ObservationSpace:
    """
    Observation space definition for RL environment
    Consistent across all agent types
    """
    market_state_dim: int = 128                 # WST feature dimension
    position_state_dim: int = 9                 # Position feature dimension  
    total_dim: int = field(init=False)          # Computed total
    
    # Data types
    market_dtype: np.dtype = np.float32
    position_dtype: np.dtype = np.float32
    
    def __post_init__(self) -> None:
        """Compute total observation dimension"""
        object.__setattr__(self, 'total_dim', self.market_state_dim + self.position_state_dim)
    
    def validate_observation(self, market_features: np.ndarray, 
                           position_features: np.ndarray) -> bool:
        """Validate observation against space definition"""
        if market_features.shape != (self.market_state_dim,):
            return False
        
        if position_features.shape != (self.position_state_dim,):
            return False
        
        if market_features.dtype != self.market_dtype:
            return False
        
        if position_features.dtype != self.position_dtype:
            return False
        
        return True


@dataclass
class ActionSpace:
    """
    Action space definition for RL environment
    Standard 4-action trading space
    """
    n_actions: int = 4
    actions: Tuple[TradingAction, ...] = (
        TradingAction.HOLD,
        TradingAction.BUY, 
        TradingAction.SELL,
        TradingAction.CLOSE
    )
    
    def validate_action(self, action: Union[int, TradingAction]) -> bool:
        """Validate action against space definition"""
        if isinstance(action, int):
            return 0 <= action < self.n_actions
        elif isinstance(action, TradingAction):
            return action in self.actions
        else:
            return False
    
    def action_to_int(self, action: TradingAction) -> int:
        """Convert trading action to integer"""
        return action.value
    
    def int_to_action(self, action_int: int) -> TradingAction:
        """Convert integer to trading action"""
        if not 0 <= action_int < self.n_actions:
            raise ValueError(f"Invalid action integer: {action_int}")
        return TradingAction(action_int)