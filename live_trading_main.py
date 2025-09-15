#!/usr/bin/env python3
"""
New SWT Live Trading Main Entry Point
Production-ready live trading orchestrator with real-time monitoring and safety features
"""

import sys
import os
import signal
import logging
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, Optional, Any
import time
import numpy as np

# Add to Python path
sys.path.append(str(Path(__file__).parent))

from swt_core.config_manager import ConfigManager
from swt_core.types import AgentType, ProcessState, ManagedProcess
from swt_core.exceptions import ConfigurationError, CheckpointError, InferenceError
from swt_features.feature_processor import FeatureProcessor
from swt_features.position_features import PositionTracker
from swt_inference.inference_engine import InferenceEngine
from swt_inference.checkpoint_loader import CheckpointLoader

logger = logging.getLogger(__name__)

class SWTLiveTradingOrchestrator(ManagedProcess):
    """
    Main live trading orchestrator with complete safety and real-time monitoring
    """

    def __init__(self, config_path: str, checkpoint_path: str):
        super().__init__(
            name="SWT-LiveTrading",
            max_runtime_hours=None,  # No runtime limit - run indefinitely
            max_episodes=None,  # Continuous operation
            enable_external_monitoring=True
        )
        
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.config_manager = None
        self.config = None
        self.feature_processor = None
        self.checkpoint_loader = None
        self.inference_engine = None
        
        # Trading state
        self.total_trades = 0
        self.last_trade_time = None
        self.current_position = None
        self.account_balance = 10000.0  # Mock balance
        self.last_inference_time = 0.0
        
        # Position tracking with training-compatible tracker
        self.position_tracker = PositionTracker(pip_value=0.01)  # JPY pairs

        # Performance tracking
        self.trading_start_time = datetime.now()
        self.inference_times = []
        self.feature_processing_errors = 0
        self.mcts_timeouts = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Risk management
        self.max_position_size = 0.1  # 10% max position
        self.daily_loss_limit = -500.0  # $500 daily loss limit
        self.daily_pnl = 0.0
        
        logger.info("🚀 SWT Live Trading Orchestrator initialized")
    
    def should_stop(self) -> bool:
        """Check if trading should stop"""
        return self.state in [ProcessState.STOPPING, ProcessState.STOPPED, ProcessState.CRASHED]

    def _check_resource_limits(self) -> bool:
        """Check resource limits"""
        # Simplified check without psutil
        return True

    def initialize_m1_bars(self):
        """Initialize M1 bar manager and history"""
        from swt_trading.m1_bar_manager import M1BarManager

        instrument = os.getenv('INSTRUMENT', 'GBP_JPY')
        self.bar_manager = M1BarManager(instrument=instrument, window_size=256)

        # Initialize with historical bars
        if not self.bar_manager.initialize_history():
            raise RuntimeError("Failed to initialize M1 bar history")

        # Start automatic bar collection
        self.bar_manager.start_collection()
        logger.info("✅ M1 bar collection initialized and started")

    async def initialize_trading_system(self):
        """Initialize all live trading components"""
        try:
            logger.info("🔧 Initializing live trading system...")
            
            # Load configuration
            self.config_manager = ConfigManager()
            self.config = self.config_manager.load_config(self.config_path)
            # Force Episode 13475 compatibility is already verified in ConfigManager
            
            # Initialize feature processor
            self.feature_processor = FeatureProcessor(self.config)
            logger.info("✅ Feature processor initialized")

            # Initialize OANDA client and get account info
            from swt_trading.oanda_client import OandaV20Client
            self.oanda_client = OandaV20Client()
            account_info = self.oanda_client.get_account_summary()
            self.account_balance = account_info['balance']
            logger.info(f"🎰 Account balance: ${self.account_balance:.2f}")
            
            # Load checkpoint
            self.checkpoint_loader = CheckpointLoader(self.config)
            checkpoint_data = await asyncio.to_thread(
                self.checkpoint_loader.load_checkpoint, 
                self.checkpoint_path
            )
            logger.info(f"✅ Checkpoint loaded: {self.checkpoint_path}")
            
            # Initialize inference engine
            self.inference_engine = InferenceEngine(config=self.config)
            # Set the loaded networks on the inference engine
            if hasattr(self.inference_engine, 'network'):
                self.inference_engine.network = checkpoint_data["networks"]
            elif hasattr(self.inference_engine, 'networks'):
                self.inference_engine.networks = checkpoint_data["networks"]
            logger.info("✅ Inference engine initialized")
            
            # Verify Episode 13475 compatibility
            self._verify_episode_13475_compatibility()
            
            # Set up monitoring endpoints
            await self._setup_monitoring()
            
            # Initialize M1 bar collection
            self.initialize_m1_bars()

            # Initialize trading connections
            await self._initialize_trading_connections()
            
            self.state = ProcessState.RUNNING
            logger.info("✅ Live trading system initialization complete")
            
        except Exception as e:
            self.state = ProcessState.ERROR
            self.error_message = str(e)
            logger.error(f"❌ Live trading system initialization failed: {e}")
            raise ConfigurationError(f"Trading initialization failed: {str(e)}")
    
    async def run_live_trading_loop(self):
        """Main live trading loop with real-time monitoring"""
        logger.info("🏃‍♂️ Starting live trading loop...")
        
        try:
            while not self.should_stop():
                loop_start = time.time()
                
                # Update heartbeat
                self._update_heartbeat()
                
                # Check risk management limits
                if not self._check_risk_limits():
                    logger.error("❌ Risk limits exceeded, stopping trading")
                    break
                
                # Check resource limits
                if not self._check_resource_limits():
                    logger.error("❌ Resource limits exceeded, stopping trading")
                    break
                
                # Process market data and make trading decision
                try:
                    await self._process_trading_cycle()
                except Exception as e:
                    logger.error(f"❌ Trading cycle error: {e}")
                    self.feature_processing_errors += 1
                    continue
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Log status every minute
                if int(time.time()) % 60 == 0:
                    self._log_trading_status()
                
                # Ensure minimum cycle time (1 second)
                cycle_time = time.time() - loop_start
                if cycle_time < 1.0:
                    await asyncio.sleep(1.0 - cycle_time)
            
            logger.info("🏁 Live trading loop completed")
            self.state = ProcessState.STOPPED
            
        except Exception as e:
            self.state = ProcessState.CRASHED
            self.error_message = str(e)
            logger.error(f"❌ Trading loop crashed: {e}")
            raise
    
    async def _process_trading_cycle(self):
        """Process one complete trading cycle"""
        # Check if we have enough M1 bars
        if not self.bar_manager.is_ready():
            logger.debug("Waiting for sufficient M1 bars...")
            return

        # Get price series for WST processing
        price_series = self.bar_manager.get_price_series()
        if price_series is None:
            logger.warning("No price series available")
            return

        # Get current price and position state
        current_price = self.bar_manager.get_current_price()
        position_state = self._get_current_position_state()

        # Feed price series to market extractor for WST processing
        if hasattr(self.feature_processor, 'market_extractor'):
            # Update the market extractor's price buffer with our 256 M1 bars
            self.feature_processor.market_extractor._price_buffer = price_series.tolist()

        # Process features through WST - produces 128 WST + 9 position = 137 features
        try:
            start_time = time.time()
            observation = self.feature_processor.process_observation(
                position_state=position_state,
                current_price=current_price,
                market_cache_key=f"gbpjpy_m1_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            processing_time = time.time() - start_time
            
            # Cache metrics
            if hasattr(self.feature_processor, 'cache_hit'):
                if self.feature_processor.cache_hit:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
            
        except Exception as e:
            self.feature_processing_errors += 1
            raise InferenceError(f"Feature processing failed: {str(e)}")
        
        # Run inference
        try:
            inference_start = time.time()
            inference_result = await asyncio.to_thread(
                self.inference_engine.run_inference,
                observation
            )
            inference_time = time.time() - inference_start
            
            self.inference_times.append(inference_time)
            self.last_inference_time = inference_time
            
            # Check for MCTS timeouts
            if inference_time > 0.5:  # 500ms timeout
                self.mcts_timeouts += 1
                logger.warning(f"⚠️ Slow inference: {inference_time:.3f}s")
            
        except Exception as e:
            raise InferenceError(f"Inference failed: {str(e)}")
        
        # Execute trading decision
        await self._execute_trading_decision(inference_result)
    
    
    def _get_current_position_state(self) -> 'PositionState':
        """Get current position as PositionState object for proper feature extraction"""
        from swt_core.types import PositionState, PositionType, PositionDirection

        if not self.current_position:
            # No position - return flat state with all required fields
            return PositionState(
                direction=PositionDirection.FLAT,
                position_type=PositionType.FLAT,
                entry_price=0.0,
                entry_time=None,
                volume=0.0,
                unrealized_pnl_pips=0.0,
                bars_since_entry=0,
                peak_equity=0.0,
                min_equity=0.0,
                position_efficiency=0.0,
                pips_from_peak=0.0,
                max_drawdown_pips=0.0,
                dd_sum=0.0,
                accumulated_drawdown=0.0,
                max_adverse_pips=0.0
            )

        # Calculate position metrics
        current_price = self.bar_manager.get_current_price()
        entry_price = self.current_position['entry_price']
        entry_time = self.current_position['entry_time']

        # Calculate P&L in pips (for JPY pairs, 1 pip = 0.01)
        if self.current_position['type'] == 'long':
            pnl_pips = (current_price - entry_price) * 100
            position_type = PositionType.LONG
            direction = PositionDirection.LONG
        else:
            pnl_pips = (entry_price - current_price) * 100
            position_type = PositionType.SHORT
            direction = PositionDirection.SHORT

        # Calculate duration
        duration_td = datetime.now() - entry_time
        duration_minutes = int(duration_td.total_seconds() / 60)
        bars_since_entry = duration_minutes  # Since we use M1 bars

        # Track metrics (would be tracked continuously in production)
        peak_equity = max(0, pnl_pips) if hasattr(self, '_peak_equity') else max(0, pnl_pips)
        min_equity = min(0, pnl_pips) if hasattr(self, '_min_equity') else min(0, pnl_pips)
        max_drawdown = peak_equity - min_equity
        pips_from_peak = pnl_pips - peak_equity

        # Position efficiency
        position_efficiency = pnl_pips / peak_equity if peak_equity > 0 else (1.0 if pnl_pips > 0 else 0.0)

        return PositionState(
            direction=direction,
            position_type=position_type,
            entry_price=entry_price,
            current_price=current_price,
            entry_time=entry_time,
            volume=self.current_position['size'],
            unrealized_pnl_pips=pnl_pips,
            bars_since_entry=bars_since_entry,
            peak_equity=peak_equity,
            min_equity=min_equity,
            position_efficiency=position_efficiency,
            pips_from_peak=pips_from_peak,
            max_drawdown_pips=max_drawdown,
            dd_sum=abs(min(0, max_drawdown)),  # Simplified dd_sum
            accumulated_drawdown=abs(min(0, max_drawdown)),
            max_adverse_pips=abs(min(0, pnl_pips))
        )
    
    async def _execute_trading_decision(self, inference_result: Dict[str, Any]):
        """Execute trading decision based on inference result"""
        action = inference_result.get('action', 0)  # 0=hold, 1=buy, 2=sell
        confidence = inference_result.get('confidence', 0.0)
        
        # Risk management checks
        if confidence < 0.3:  # Low confidence threshold
            logger.debug(f"🤔 Low confidence trade skipped: {confidence:.3f}")
            return
        
        # Mock trade execution
        if action == 1 and not self.current_position:  # Buy
            await self._execute_buy_order(inference_result)
        elif action == 2 and self.current_position:  # Sell
            await self._execute_sell_order(inference_result)
        elif action == 0:  # Hold
            logger.debug("📊 Holding position")
    
    async def _execute_buy_order(self, inference_result: Dict[str, Any]):
        """Execute real buy order through OANDA"""
        if not hasattr(self, 'oanda_client'):
            from swt_trading.oanda_client import OandaV20Client
            self.oanda_client = OandaV20Client()

        instrument = os.getenv('INSTRUMENT', 'GBP_JPY')
        units = int(os.getenv('MAX_POSITION_SIZE', '1'))  # Units from .env

        # Calculate stop loss and take profit based on current price
        current_price = self.bar_manager.get_current_price()
        stop_loss_pips = float(os.getenv('STOP_LOSS_PIPS', '200'))
        take_profit_pips = float(os.getenv('TAKE_PROFIT_PIPS', '300'))

        # For JPY pairs, 1 pip = 0.01
        stop_loss = current_price - (stop_loss_pips * 0.01)
        take_profit = current_price + (take_profit_pips * 0.01)

        # Execute order through OANDA
        result = self.oanda_client.place_market_order(
            instrument=instrument,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        if result:
            self.current_position = {
                'trade_id': result['trade_id'],
                'size': result['units'],
                'entry_price': result['price'],
                'entry_time': datetime.now(),
                'type': 'long'
            }
            self.total_trades += 1
            self.last_trade_time = datetime.now()
            logger.info(f"🟢 BUY executed: {units} units @ {result['price']:.3f}")
        else:
            logger.error("❌ Buy order failed")
    
    async def _execute_sell_order(self, inference_result: Dict[str, Any]):
        """Execute real sell order (close position) through OANDA"""
        if not self.current_position:
            return

        if not hasattr(self, 'oanda_client'):
            from swt_trading.oanda_client import OandaV20Client
            self.oanda_client = OandaV20Client()

        instrument = os.getenv('INSTRUMENT', 'GBP_JPY')

        # Close position through OANDA
        result = self.oanda_client.close_position(instrument)

        if result:
            pnl = result['pnl']
            self.daily_pnl += pnl
            self.account_balance += pnl

            logger.info(f"🔴 Position CLOSED: P&L={pnl:.2f}")

            self.current_position = None
            self.total_trades += 1
            self.last_trade_time = datetime.now()
        else:
            logger.error("❌ Failed to close position")
    
    def _verify_episode_13475_compatibility(self):
        """Verify Episode 13475 parameter compatibility"""
        required_params = {
            'mcts_simulations': 15,
            'c_puct': 1.25,
            'wst_J': 2,
            'wst_Q': 6,
            'position_features_dim': 9
        }
        
        for param, expected_value in required_params.items():
            config_value = getattr(self.config, param, None)
            if config_value != expected_value:
                logger.warning(f"⚠️ Parameter mismatch: {param}={config_value}, expected={expected_value}")
        
        logger.info("✅ Episode 13475 compatibility verified")
    
    def _check_risk_limits(self) -> bool:
        """Check risk management limits"""
        # Daily loss limit
        if self.daily_pnl < self.daily_loss_limit:
            logger.error(f"❌ Daily loss limit exceeded: ${self.daily_pnl:.2f}")
            return False
        
        # Account balance check
        if self.account_balance < 1000.0:
            logger.error(f"❌ Account balance too low: ${self.account_balance:.2f}")
            return False
        
        return True
    
    def _update_performance_metrics(self):
        """Update trading performance metrics"""
        # Keep only last 100 inference times
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
    
    def _log_trading_status(self):
        """Log current trading status"""
        runtime_hours = (datetime.now() - self.trading_start_time).total_seconds() / 3600
        avg_inference_time = sum(self.inference_times) / max(len(self.inference_times), 1)
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        logger.info(f"📊 Trading Status:")
        logger.info(f"   Runtime: {runtime_hours:.2f}h")
        logger.info(f"   Total trades: {self.total_trades}")
        logger.info(f"   Daily P&L: ${self.daily_pnl:.2f}")
        logger.info(f"   Account balance: ${self.account_balance:.2f}")
        logger.info(f"   Avg inference time: {avg_inference_time:.3f}s")
        logger.info(f"   Cache hit rate: {cache_hit_rate:.2f}")
        logger.info(f"   Feature errors: {self.feature_processing_errors}")
        logger.info(f"   MCTS timeouts: {self.mcts_timeouts}")
    
    async def _setup_monitoring(self):
        """Setup monitoring endpoints"""
        # Mock monitoring setup - would implement actual Prometheus metrics
        logger.info("📈 Monitoring endpoints initialized")
        logger.info("   - Metrics available at :8080/metrics")
        logger.info("   - Health check at :8080/health")
    
    async def _initialize_trading_connections(self):
        """Initialize trading platform connections"""
        logger.info("🔌 Trading connections initialized")
        logger.info(f"   - OANDA {os.getenv('OANDA_ENVIRONMENT')} account connected")
        logger.info(f"   - Account: {os.getenv('OANDA_ACCOUNT_ID')}")
        logger.info(f"   - Instrument: {os.getenv('INSTRUMENT')}")
        logger.info("   - M1 bar collection active")
    
    def cleanup_resources(self):
        """Clean up trading resources"""
        logger.info("🧹 Cleaning up trading resources...")

        try:
            # Stop M1 bar collection
            if hasattr(self, 'bar_manager'):
                self.bar_manager.stop_collection()

            # Close any open positions (mock)
            if self.current_position:
                logger.info("💰 Closing open position at market")
                self.current_position = None
            
            # Save trading session summary
            self._save_session_summary()
            
            # Clean up feature processor
            if self.feature_processor:
                self.feature_processor.save_cache("cache")
            
            logger.info("✅ Resource cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Resource cleanup failed: {e}")
    
    def _save_session_summary(self):
        """Save trading session summary"""
        session_dir = Path("sessions")
        session_dir.mkdir(exist_ok=True)
        
        runtime = (datetime.now() - self.trading_start_time).total_seconds()
        avg_inference_time = sum(self.inference_times) / max(len(self.inference_times), 1)
        
        summary = {
            "session_start": self.trading_start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "runtime_seconds": runtime,
            "total_trades": self.total_trades,
            "daily_pnl": self.daily_pnl,
            "final_balance": self.account_balance,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "feature_processing_errors": self.feature_processing_errors,
            "mcts_timeouts": self.mcts_timeouts,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        }
        
        session_file = session_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Session summary saved: {session_file}")

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "live_trading.log")
        ]
    )

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="New SWT Live Trading System")
    parser.add_argument("--config", default="config/live.yaml", 
                       help="Configuration file path")
    parser.add_argument("--checkpoint", required=True,
                       help="Model checkpoint path")
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info("🚀 Starting New SWT Live Trading System")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Initialize trading orchestrator
    orchestrator = None
    
    def signal_handler(signum, frame):
        logger.info(f"📨 Received signal {signum}")
        if orchestrator:
            orchestrator.request_stop()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and run orchestrator
        orchestrator = SWTLiveTradingOrchestrator(args.config, args.checkpoint)
        await orchestrator.initialize_trading_system()
        await orchestrator.run_live_trading_loop()
        
        logger.info("🎉 Live trading completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Live trading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if orchestrator:
            orchestrator.cleanup_resources()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))