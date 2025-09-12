"""
SWT Live Trading System
Event-driven live trading implementation using shared inference components

This module provides production-ready live trading functionality with:
- Real-time OANDA data streaming
- Event-driven trading decisions  
- Order execution with error handling
- Position reconciliation and monitoring
- Performance tracking and alerting

Key Components:
- DataFeed: Real-time market data streaming
- EventTrader: Main trading orchestration
- TradeExecutor: Order execution system
- PositionReconciler: Position management
- LiveMonitor: Performance tracking
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_feed import OANDADataFeed
    from .event_trader import SWTEventTrader  
    from .trade_executor import SWTTradeExecutor
    from .position_reconciler import SWTPositionReconciler
    from .monitoring import SWTLiveMonitor

__all__ = [
    "OANDADataFeed",
    "SWTEventTrader", 
    "SWTTradeExecutor",
    "SWTPositionReconciler", 
    "SWTLiveMonitor"
]

__version__ = "1.0.0"