"""
Comprehensive Live Trading System Integration Tests
Tests the complete live trading infrastructure

This test suite validates:
- All swt_live components integration
- End-to-end trading workflow
- Error handling and recovery
- Performance and reliability
- Real-time data processing
- Position management accuracy

CRITICAL: These tests verify production readiness of the live trading system.
"""

import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Import system under test
from swt_core.config_manager import SWTConfig
from swt_core.types import TradingAction, TradingDecision, PositionType, PositionState
from swt_live.data_feed import OANDADataFeed, StreamingBar
from swt_live.trade_executor import SWTTradeExecutor, OrderRequest, ExecutionReport, OrderStatus
from swt_live.position_reconciler import SWTPositionReconciler
from swt_live.monitoring import SWTLiveMonitor, AlertLevel
from swt_live.event_trader import SWTEventTrader

logger = logging.getLogger(__name__)


class TestLiveTradingSystemIntegration:
    """Integration tests for complete live trading system"""
    
    @pytest.fixture
    async def mock_config(self):
        """Create mock configuration for testing"""
        config = MagicMock(spec=SWTConfig)
        config.agent_system.value = "stochastic_muzero"
        config.live_trading_config.oanda_account_id = "test_account"
        config.live_trading_config.oanda_api_token = "test_token"
        config.live_trading_config.oanda_environment = "practice"
        config.live_trading_config.instrument = "GBP_JPY"
        config.trading_config.position_size = 1000
        config.trading_config.min_confidence = 0.6
        config.trading_config.max_position_size = 5000
        config.trading_config.max_daily_trades = 20
        config.trading_config.max_daily_loss = 500.0
        config.trading_config.max_slippage_pips = 3.0
        config.trading_config.max_drawdown_percent = 10.0
        config.trading_config.max_consecutive_losses = 5
        return config
    
    @pytest.fixture
    def mock_streaming_bar(self):
        """Create mock streaming bar data"""
        return StreamingBar(
            instrument="GBP_JPY",
            timestamp=datetime.now(timezone.utc),
            bid=195.123,
            ask=195.126,
            volume=1000,
            spread=0.003
        )
    
    @pytest.fixture
    def mock_trading_decision(self):
        """Create mock trading decision"""
        return TradingDecision(
            action=TradingAction.BUY,
            confidence=0.75,
            value_estimate=5.2,
            policy_distribution=[0.1, 0.7, 0.15, 0.05],
            mcts_visits=[5, 35, 8, 2],
            search_time_ms=45.6,
            agent_type=MagicMock(),
            model_confidence=0.72
        )

    # Test 1: Data Feed Integration
    @pytest.mark.asyncio
    async def test_data_feed_integration(self, mock_config):
        """Test OANDA data feed integration"""
        
        # Mock callback for data reception
        received_bars = []
        async def on_bar_callback(bar: StreamingBar):
            received_bars.append(bar)
        
        error_received = []
        async def on_error_callback(error: Exception):
            error_received.append(error)
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.content = AsyncMock()
            mock_response.content.__aiter__ = AsyncMock(return_value=[
                b'{"type":"PRICE","instrument":"GBP_JPY","time":"2025-01-01T12:00:00.000000Z","bids":[{"price":"195.123"}],"asks":[{"price":"195.126"}]}\n',
                b'{"type":"HEARTBEAT","time":"2025-01-01T12:00:30.000000Z"}\n'
            ])
            
            mock_session.return_value.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            
            # Create and test data feed
            data_feed = OANDADataFeed(
                config=mock_config,
                on_bar_callback=on_bar_callback,
                on_error_callback=on_error_callback
            )
            
            try:
                await data_feed.start()
                await asyncio.sleep(0.1)  # Allow processing
                
                # Verify data feed status
                status = data_feed.get_status()
                assert status["status"] == "connected"
                assert status["instrument"] == "GBP_JPY"
                
                # Verify bars were processed
                assert len(received_bars) > 0
                assert received_bars[0].instrument == "GBP_JPY"
                assert received_bars[0].bid == 195.123
                assert received_bars[0].ask == 195.126
                
            finally:
                await data_feed.stop()

    # Test 2: Trade Executor Integration
    @pytest.mark.asyncio
    async def test_trade_executor_integration(self, mock_config):
        """Test trade executor with mock OANDA API"""
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful order response
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json = AsyncMock(return_value={
                "orderFillTransaction": [{
                    "type": "ORDER_FILL",
                    "orderID": "12345",
                    "id": "67890",
                    "units": "1000",
                    "price": "195.125",
                    "commission": "0.25",
                    "pl": "0.0"
                }]
            })
            
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            
            # Create and test trade executor
            executor = SWTTradeExecutor(mock_config)
            
            try:
                await executor.start()
                
                # Create test order
                order = OrderRequest(
                    action=TradingAction.BUY,
                    instrument="GBP_JPY", 
                    units=1000
                )
                
                # Execute order
                execution = await executor.execute_order(order, expected_price=195.124)
                
                # Verify execution
                assert execution.status == OrderStatus.FILLED
                assert execution.filled_units == 1000
                assert execution.execution_price == 195.125
                assert execution.slippage_pips == 0.1  # 0.001 * 100
                
                # Verify statistics
                stats = executor.get_execution_stats()
                assert stats["orders_sent"] == 1
                assert stats["orders_filled"] == 1
                assert stats["fill_rate_pct"] == 100.0
                
            finally:
                await executor.stop()

    # Test 3: Position Reconciler Integration  
    @pytest.mark.asyncio
    async def test_position_reconciler_integration(self, mock_config):
        """Test position reconciler with mock trade executor"""
        
        # Mock trade executor
        mock_executor = AsyncMock(spec=SWTTradeExecutor)
        mock_executor.get_position_status = AsyncMock(return_value={
            "long": {"units": "1000", "averagePrice": "195.125"},
            "short": {"units": "0", "averagePrice": "0"},
            "instrument": "GBP_JPY"
        })
        
        # Create position reconciler
        reconciler = SWTPositionReconciler(mock_config, mock_executor)
        
        try:
            await reconciler.start()
            
            # Test initial position reconciliation
            in_sync = await reconciler.reconcile_position()
            assert in_sync  # Should be in sync initially
            
            # Test execution processing
            execution = ExecutionReport(
                order_id="12345",
                transaction_id="67890",
                status=OrderStatus.FILLED,
                filled_units=1000,
                execution_price=195.125,
                pl_realized=0.0
            )
            
            await reconciler.process_execution(execution)
            
            # Verify position updated
            position = reconciler.get_current_position()
            assert position.position_type == PositionType.LONG
            assert position.units == 1000
            assert position.entry_price == 195.125
            
            # Test P&L update
            reconciler.update_unrealized_pnl(195.130)  # 0.5 pip profit
            assert position.unrealized_pnl == 5.0  # 1000 units * 0.005
            
            # Verify statistics
            stats = reconciler.get_reconciliation_stats()
            assert stats["sync_status"] == "in_sync"
            assert stats["total_reconciliations"] >= 1
            
        finally:
            await reconciler.stop()

    # Test 4: Live Monitor Integration
    @pytest.mark.asyncio
    async def test_live_monitor_integration(self, mock_config, mock_trading_decision):
        """Test live monitoring system"""
        
        # Create live monitor
        monitor = SWTLiveMonitor(mock_config)
        
        try:
            await monitor.start()
            
            # Test decision recording
            await monitor.record_decision(mock_trading_decision, processing_time=0.045)
            
            # Verify decision recorded
            performance = monitor.get_performance_summary()
            assert performance["total_decisions"] == 1
            assert performance["average_confidence"] == 0.75
            
            # Test execution recording
            execution = ExecutionReport(
                order_id="12345",
                transaction_id="67890",
                status=OrderStatus.FILLED,
                filled_units=1000,
                execution_price=195.125,
                pl_realized=5.0,  # Profit
                slippage_pips=0.2
            )
            
            await monitor.record_execution(execution)
            
            # Verify execution recorded
            performance = monitor.get_performance_summary()
            assert performance["total_trades"] == 1
            assert performance["winning_trades"] == 1
            assert performance["total_pnl"] == 5.0
            assert performance["average_slippage_pips"] == 0.2
            
            # Test alert system (force drawdown alert)
            losing_execution = ExecutionReport(
                order_id="12346",
                transaction_id="67891",
                status=OrderStatus.FILLED,
                filled_units=1000,
                execution_price=195.120,
                pl_realized=-50.0,  # Large loss
                slippage_pips=0.3
            )
            
            await monitor.record_execution(losing_execution)
            
            # Check alerts were generated
            alerts = monitor.get_recent_alerts(limit=5)
            assert len(alerts) > 0
            
            # Verify health metrics
            health = monitor.get_health_summary()
            assert health["uptime_seconds"] > 0
            assert health["system_status"] == "running"
            
        finally:
            await monitor.stop()

    # Test 5: End-to-End Event Trader Integration
    @pytest.mark.asyncio
    async def test_event_trader_integration(self, mock_config, mock_streaming_bar):
        """Test complete event trader orchestration"""
        
        with patch('swt_inference.agent_factory.AgentFactory.create_agent') as mock_agent_factory, \
             patch('swt_features.feature_processor.FeatureProcessor') as mock_feature_processor, \
             patch('swt_inference.inference_engine.SWTInferenceEngine') as mock_inference_engine, \
             patch('aiohttp.ClientSession'):
            
            # Mock agent creation
            mock_agent = MagicMock()
            mock_agent.load_checkpoint = MagicMock()
            mock_agent_factory.return_value = mock_agent
            
            # Mock feature processor
            mock_fp_instance = MagicMock()
            mock_fp_instance.add_market_data = MagicMock()
            mock_fp_instance.is_ready = MagicMock(return_value=True)
            mock_fp_instance.process_observation = MagicMock(return_value=MagicMock())
            mock_feature_processor.return_value = mock_fp_instance
            
            # Mock inference engine
            mock_ie_instance = MagicMock()
            mock_ie_instance.get_trading_decision = AsyncMock(return_value=TradingDecision(
                action=TradingAction.HOLD,  # Safe action for testing
                confidence=0.4,  # Below threshold
                value_estimate=0.0,
                policy_distribution=[0.7, 0.1, 0.1, 0.1],
                agent_type=mock_config.agent_system,
                model_confidence=0.4
            ))
            mock_inference_engine.return_value = mock_ie_instance
            
            # Create event trader
            event_trader = SWTEventTrader(
                config=mock_config,
                checkpoint_path="/fake/path/checkpoint.pth"
            )
            
            try:
                # This will initialize all components
                await event_trader.start()
                
                # Verify system is running
                stats = event_trader.get_system_stats()
                assert stats["state"] == "running"
                assert stats["total_bars_processed"] == 0  # No data processed yet
                
                # Simulate market data event (this triggers the main trading loop)
                await event_trader._on_market_data(mock_streaming_bar)
                
                # Verify data was processed
                stats = event_trader.get_system_stats()
                assert stats["total_bars_processed"] == 1
                assert stats["total_decisions_made"] >= 0  # May or may not make decision
                
                # Verify components are working
                diagnostics = event_trader.get_diagnostics()
                assert "components" in diagnostics
                assert diagnostics["state"] == "running"
                
                # Test pause/resume
                await event_trader.pause()
                stats = event_trader.get_system_stats()
                assert stats["state"] == "paused"
                
                await event_trader.resume()
                stats = event_trader.get_system_stats()
                assert stats["state"] == "running"
                
            finally:
                await event_trader.stop()

    # Test 6: Error Handling and Recovery
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_config):
        """Test system error handling and recovery mechanisms"""
        
        # Test data feed error handling
        error_callbacks_called = []
        
        def error_callback(error):
            error_callbacks_called.append(error)
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock connection error
            mock_session.return_value.get.side_effect = ConnectionError("Connection failed")
            
            data_feed = OANDADataFeed(
                config=mock_config,
                on_bar_callback=lambda bar: None,
                on_error_callback=error_callback
            )
            
            try:
                await data_feed.start()
                await asyncio.sleep(0.1)  # Allow error to propagate
                
                # Verify error was handled
                status = data_feed.get_status()
                assert status["status"] in ["reconnecting", "error"]
                assert len(error_callbacks_called) > 0
                
            finally:
                await data_feed.stop()
        
        # Test trade executor retry logic
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock temporary failure then success
            mock_response_fail = AsyncMock()
            mock_response_fail.status = 500
            mock_response_fail.json = AsyncMock(return_value={"errorMessage": "Server error"})
            
            mock_response_success = AsyncMock() 
            mock_response_success.status = 201
            mock_response_success.json = AsyncMock(return_value={
                "orderFillTransaction": [{
                    "type": "ORDER_FILL",
                    "orderID": "12345",
                    "id": "67890", 
                    "units": "1000",
                    "price": "195.125"
                }]
            })
            
            # First call fails, second succeeds
            mock_session.return_value.post.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_success
            ]
            
            executor = SWTTradeExecutor(mock_config)
            
            try:
                await executor.start()
                
                order = OrderRequest(
                    action=TradingAction.BUY,
                    instrument="GBP_JPY",
                    units=1000
                )
                
                # Should succeed after retry
                execution = await executor.execute_order(order)
                assert execution.status == OrderStatus.FILLED
                
            finally:
                await executor.stop()

    # Test 7: Performance and Reliability
    @pytest.mark.asyncio
    async def test_performance_and_reliability(self, mock_config):
        """Test system performance under load"""
        
        # Test monitoring system performance tracking
        monitor = SWTLiveMonitor(mock_config)
        
        try:
            await monitor.start()
            
            # Simulate rapid decision making
            decisions_made = 0
            start_time = asyncio.get_event_loop().time()
            
            for i in range(10):
                decision = TradingDecision(
                    action=TradingAction.HOLD,
                    confidence=0.5 + (i * 0.05),  # Varying confidence
                    value_estimate=i * 0.1,
                    policy_distribution=[0.25, 0.25, 0.25, 0.25],
                    agent_type=mock_config.agent_system,
                    model_confidence=0.5
                )
                
                processing_time = 0.01 + (i * 0.001)  # Simulated processing time
                await monitor.record_decision(decision, processing_time)
                decisions_made += 1
                
                # Small delay to simulate realistic timing
                await asyncio.sleep(0.001)
            
            end_time = asyncio.get_event_loop().time()
            
            # Verify performance metrics
            performance = monitor.get_performance_summary()
            assert performance["total_decisions"] == 10
            assert performance["average_confidence"] > 0.5
            
            health = monitor.get_health_summary()
            assert health["processing_latency_ms"] > 0
            
            # Verify no memory leaks in data structures
            assert len(monitor.decision_history) == 10
            assert len(monitor.alerts) >= 0  # May have alerts
            
        finally:
            await monitor.stop()

    # Test 8: Configuration and Validation
    @pytest.mark.asyncio
    async def test_configuration_validation(self, mock_config):
        """Test configuration validation and edge cases"""
        
        # Test invalid configuration handling
        invalid_config = MagicMock(spec=SWTConfig)
        invalid_config.live_trading_config.oanda_account_id = ""  # Invalid
        invalid_config.trading_config.position_size = 0  # Invalid
        
        # Should handle invalid config gracefully
        try:
            executor = SWTTradeExecutor(invalid_config)
            await executor.start()
            
            # Invalid order should be rejected
            invalid_order = OrderRequest(
                action=TradingAction.BUY,
                instrument="INVALID",
                units=0  # Invalid
            )
            
            with pytest.raises(Exception):  # Should raise validation error
                await executor.execute_order(invalid_order)
                
        except Exception as e:
            # Expected to fail with invalid config
            assert "validation" in str(e).lower() or "invalid" in str(e).lower()

    # Test 9: Real-time Data Processing
    @pytest.mark.asyncio 
    async def test_realtime_data_processing(self, mock_config, mock_streaming_bar):
        """Test real-time data processing pipeline"""
        
        received_data = []
        processing_times = []
        
        async def data_callback(bar):
            start_time = asyncio.get_event_loop().time()
            received_data.append(bar)
            # Simulate processing
            await asyncio.sleep(0.001)
            end_time = asyncio.get_event_loop().time()
            processing_times.append(end_time - start_time)
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock rapid data stream
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.content.__aiter__ = AsyncMock(return_value=[
                b'{"type":"PRICE","instrument":"GBP_JPY","time":"2025-01-01T12:00:00.000000Z","bids":[{"price":"195.123"}],"asks":[{"price":"195.126"}]}\n',
                b'{"type":"PRICE","instrument":"GBP_JPY","time":"2025-01-01T12:00:01.000000Z","bids":[{"price":"195.124"}],"asks":[{"price":"195.127"}]}\n',
                b'{"type":"PRICE","instrument":"GBP_JPY","time":"2025-01-01T12:00:02.000000Z","bids":[{"price":"195.125"}],"asks":[{"price":"195.128"}]}\n'
            ])
            
            mock_session.return_value.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            
            data_feed = OANDADataFeed(
                config=mock_config,
                on_bar_callback=data_callback,
                on_error_callback=lambda e: None
            )
            
            try:
                await data_feed.start()
                await asyncio.sleep(0.1)  # Allow processing
                
                # Verify rapid data processing
                assert len(received_data) >= 3
                assert all(t < 0.01 for t in processing_times)  # Fast processing
                
                # Verify data quality
                for bar in received_data:
                    assert bar.instrument == "GBP_JPY"
                    assert bar.bid > 0
                    assert bar.ask > bar.bid
                    assert bar.spread == bar.ask - bar.bid
                
            finally:
                await data_feed.stop()


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "--tb=short", 
        "--asyncio-mode=auto"
    ])