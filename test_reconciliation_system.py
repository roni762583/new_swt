#!/usr/bin/env python3
"""
Comprehensive Test Framework for Position Reconciliation System
Tests all phases of the reconciliation system integration
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import unittest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional

# Add to Python path
sys.path.append(str(Path(__file__).parent))

from position_reconciliation import (
    BrokerPositionReconciler, 
    BrokerPosition, 
    InternalPosition, 
    PositionDiscrepancy,
    ReconciliationEvent
)
from live_trading_episode_13475 import Episode13475LiveAgent

logger = logging.getLogger(__name__)

class ReconciliationSystemTestSuite(unittest.IsolatedAsyncioTestCase):
    """Comprehensive test suite for position reconciliation system"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_oanda_executor = Mock()
        self.test_config_path = "/tmp/test_config.yaml"
        
        # Set up logging for tests
        logging.basicConfig(level=logging.INFO)
        
    async def test_phase_1_architecture(self):
        """Test Phase 1: Position reconciliation architecture"""
        logger.info("🧪 Testing Phase 1: Architecture Design")
        
        # Test BrokerPositionReconciler initialization
        reconciler = BrokerPositionReconciler(
            oanda_executor=self.mock_oanda_executor,
            instrument="GBP_JPY"
        )
        
        self.assertIsNotNone(reconciler)
        self.assertEqual(reconciler.instrument, "GBP_JPY")
        self.assertIsNotNone(reconciler.reconciliation_events)
        
        logger.info("✅ Phase 1: Architecture design validated")
    
    async def test_phase_2_broker_position_query(self):
        """Test Phase 2: Broker position query system"""
        logger.info("🧪 Testing Phase 2: Broker Position Query")
        
        # Mock broker position response
        self.mock_oanda_executor.api.position.get.return_value = Mock(
            body={'position': {
                'instrument': 'GBP_JPY',
                'long': {'units': '100'},
                'short': {'units': '0'},
                'unrealizedPL': '5.25'
            }}
        )
        
        reconciler = BrokerPositionReconciler(
            oanda_executor=self.mock_oanda_executor,
            instrument="GBP_JPY"
        )
        
        broker_position = await reconciler.get_broker_position("GBP_JPY")
        
        self.assertIsNotNone(broker_position)
        self.assertEqual(broker_position.units, 100)
        self.assertEqual(broker_position.instrument, "GBP_JPY")
        
        logger.info("✅ Phase 2: Broker position query validated")
    
    async def test_phase_3_startup_reconciliation(self):
        """Test Phase 3: Startup reconciliation logic"""
        logger.info("🧪 Testing Phase 3: Startup Reconciliation")
        
        with patch.object(Episode13475LiveAgent, '_initialize_oanda_connection'):
            agent = Episode13475LiveAgent(self.test_config_path)
            agent.oanda_executor = self.mock_oanda_executor
            
            # Mock reconciler
            mock_reconciler = AsyncMock()
            mock_reconciler.startup_reconciliation.return_value = ReconciliationEvent(
                timestamp=datetime.now(),
                event_type="startup",
                status="completed",
                broker_position=BrokerPosition(
                    instrument="GBP_JPY",
                    units=100,
                    average_price=195.123,
                    unrealized_pnl=5.25,
                    margin_used=100.0,
                    timestamp=datetime.now()
                ),
                internal_position=None,
                discrepancies=[],
                notes="Startup reconciliation successful"
            )
            
            agent.position_reconciler = mock_reconciler
            
            await agent._startup_reconciliation()
            
            # Verify position was synchronized
            self.assertIsNotNone(agent.current_position)
            self.assertEqual(agent.current_position['type'], 'long')
            self.assertEqual(agent.current_position['size'], 100)
            
            logger.info("✅ Phase 3: Startup reconciliation validated")
    
    async def test_phase_4_real_time_verification(self):
        """Test Phase 4: Real-time position verification"""
        logger.info("🧪 Testing Phase 4: Real-time Verification")
        
        with patch.object(Episode13475LiveAgent, '_initialize_oanda_connection'):
            agent = Episode13475LiveAgent(self.test_config_path)
            agent.oanda_executor = self.mock_oanda_executor
            
            # Mock reconciler with post-trade check
            mock_reconciler = AsyncMock()
            mock_reconciler.perform_reconciliation.return_value = ReconciliationEvent(
                timestamp=datetime.now(),
                event_type="post_buy",
                status="completed",
                broker_position=None,
                internal_position=None,
                discrepancies=[],
                notes="Post-trade verification successful"
            )
            
            agent.position_reconciler = mock_reconciler
            
            await agent._post_trade_reconciliation_check("buy")
            
            # Verify reconciliation was called
            mock_reconciler.perform_reconciliation.assert_called_once_with(
                instrument="GBP_JPY",
                event_type="post_buy"
            )
            
            logger.info("✅ Phase 4: Real-time verification validated")
    
    async def test_phase_5_position_feature_validation(self):
        """Test Phase 5: Position feature validation"""
        logger.info("🧪 Testing Phase 5: Position Feature Validation")
        
        with patch.object(Episode13475LiveAgent, '_initialize_oanda_connection'):
            agent = Episode13475LiveAgent(self.test_config_path)
            
            # Test valid position
            valid_position = {
                'type': 'long',
                'size': 100,
                'entry_price': 195.123,
                'entry_time': datetime.now(),
                'confidence': 0.85
            }
            
            validated = agent._validate_position_features(valid_position)
            self.assertIsNotNone(validated)
            self.assertIn('position_age_minutes', validated)
            
            # Test invalid position
            invalid_position = {
                'type': 'invalid_type',
                'size': -100,  # Invalid size
                'entry_price': 0,  # Invalid price
            }
            
            validated_invalid = agent._validate_position_features(invalid_position)
            self.assertIsNone(validated_invalid)
            
            logger.info("✅ Phase 5: Position feature validation validated")
    
    async def test_phase_6_periodic_reconciliation(self):
        """Test Phase 6: Periodic reconciliation checks"""
        logger.info("🧪 Testing Phase 6: Periodic Reconciliation")
        
        with patch.object(Episode13475LiveAgent, '_initialize_oanda_connection'):
            agent = Episode13475LiveAgent(self.test_config_path)
            agent.oanda_executor = self.mock_oanda_executor
            
            # Mock reconciler
            mock_reconciler = AsyncMock()
            mock_reconciler.perform_reconciliation.return_value = ReconciliationEvent(
                timestamp=datetime.now(),
                event_type="periodic_health_check",
                status="completed",
                broker_position=None,
                internal_position=None,
                discrepancies=[],
                notes="Periodic check successful"
            )
            mock_reconciler.get_reconciliation_stats.return_value = Mock(
                total_reconciliations=10,
                successful_reconciliations=10,
                failed_reconciliations=0,
                success_rate=100.0,
                total_discrepancies=0,
                critical_discrepancies=0,
                last_reconciliation_time=datetime.now(),
                recent_discrepancies=[]
            )
            
            agent.position_reconciler = mock_reconciler
            agent.last_reconciliation_time = datetime.now() - timedelta(minutes=6)  # Force periodic check
            
            await agent._periodic_reconciliation_check()
            
            # Verify periodic reconciliation was called
            mock_reconciler.perform_reconciliation.assert_called_once()
            
            logger.info("✅ Phase 6: Periodic reconciliation validated")
    
    async def test_phase_7_edge_case_recovery(self):
        """Test Phase 7: Edge cases and recovery"""
        logger.info("🧪 Testing Phase 7: Edge Cases and Recovery")
        
        with patch.object(Episode13475LiveAgent, '_initialize_oanda_connection'):
            agent = Episode13475LiveAgent(self.test_config_path)
            agent.oanda_executor = self.mock_oanda_executor
            
            # Test container restart recovery
            # Create mock session file
            session_dir = Path("sessions")
            session_dir.mkdir(exist_ok=True)
            
            mock_session = {
                "session_end": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "total_trades": 5,
                "daily_pnl": 25.50,
                "final_balance": 10025.50,
                "max_drawdown": 5.0
            }
            
            session_file = session_dir / f"episode_13475_live_test.json"
            with open(session_file, 'w') as f:
                json.dump(mock_session, f)
            
            await agent._handle_container_restart_recovery()
            
            # Verify state was restored
            self.assertEqual(agent.account_balance, 10025.50)
            self.assertEqual(agent.daily_pnl, 25.50)
            
            # Cleanup
            session_file.unlink()
            
            # Test network recovery
            self.mock_oanda_executor._get_current_price.return_value = 195.123
            network_ok = await agent._handle_network_recovery()
            self.assertTrue(network_ok)
            
            logger.info("✅ Phase 7: Edge cases and recovery validated")
    
    async def test_discrepancy_handling(self):
        """Test discrepancy detection and handling"""
        logger.info("🧪 Testing Discrepancy Detection and Handling")
        
        reconciler = BrokerPositionReconciler(
            oanda_executor=self.mock_oanda_executor,
            instrument="GBP_JPY"
        )
        
        # Create test positions with discrepancies
        broker_position = BrokerPosition(
            instrument="GBP_JPY",
            units=100,
            average_price=195.123,
            unrealized_pnl=5.25,
            margin_used=100.0,
            timestamp=datetime.now()
        )
        
        internal_position = InternalPosition(
            instrument="GBP_JPY",
            position_type="long",
            size=150,  # Size mismatch
            entry_price=195.456,  # Price mismatch
            entry_time=datetime.now() - timedelta(minutes=5),
            confidence=0.85
        )
        
        discrepancies = reconciler.compare_positions(broker_position, internal_position)
        
        self.assertTrue(len(discrepancies) > 0)
        self.assertTrue(any(d.discrepancy_type == "SIZE_MISMATCH" for d in discrepancies))
        self.assertTrue(any(d.discrepancy_type == "PRICE_MISMATCH" for d in discrepancies))
        
        logger.info("✅ Discrepancy detection validated")
    
    async def test_reconciliation_statistics(self):
        """Test reconciliation statistics tracking"""
        logger.info("🧪 Testing Reconciliation Statistics")
        
        reconciler = BrokerPositionReconciler(
            oanda_executor=self.mock_oanda_executor,
            instrument="GBP_JPY"
        )
        
        # Simulate some reconciliation events
        reconciler.reconciliation_events.extend([
            ReconciliationEvent(
                timestamp=datetime.now(),
                event_type="startup",
                status="completed",
                broker_position=None,
                internal_position=None,
                discrepancies=[],
                notes="Test event 1"
            ),
            ReconciliationEvent(
                timestamp=datetime.now(),
                event_type="post_trade",
                status="completed",
                broker_position=None,
                internal_position=None,
                discrepancies=[
                    PositionDiscrepancy(
                        discrepancy_type="SIZE_MISMATCH",
                        severity="WARNING",
                        description="Test discrepancy",
                        broker_value="100",
                        internal_value="150",
                        timestamp=datetime.now()
                    )
                ],
                notes="Test event 2"
            )
        ])
        
        stats = reconciler.get_reconciliation_stats()
        
        self.assertEqual(stats.total_reconciliations, 2)
        self.assertEqual(stats.successful_reconciliations, 2)
        self.assertEqual(stats.total_discrepancies, 1)
        self.assertEqual(stats.success_rate, 100.0)
        
        logger.info("✅ Reconciliation statistics validated")
    
    async def test_integration_workflow(self):
        """Test complete integration workflow"""
        logger.info("🧪 Testing Complete Integration Workflow")
        
        with patch.object(Episode13475LiveAgent, '_initialize_oanda_connection'):
            # Create agent with mocked components
            agent = Episode13475LiveAgent(self.test_config_path)
            agent.oanda_executor = self.mock_oanda_executor
            
            # Mock complete reconciler
            mock_reconciler = AsyncMock()
            
            # Mock startup reconciliation
            mock_reconciler.startup_reconciliation.return_value = ReconciliationEvent(
                timestamp=datetime.now(),
                event_type="startup",
                status="completed",
                broker_position=None,
                internal_position=None,
                discrepancies=[],
                notes="Clean startup"
            )
            
            # Mock post-trade reconciliation
            mock_reconciler.perform_reconciliation.return_value = ReconciliationEvent(
                timestamp=datetime.now(),
                event_type="post_trade",
                status="completed",
                broker_position=None,
                internal_position=None,
                discrepancies=[],
                notes="Trade verified"
            )
            
            # Mock statistics
            mock_reconciler.get_reconciliation_stats.return_value = Mock(
                total_reconciliations=3,
                successful_reconciliations=3,
                failed_reconciliations=0,
                success_rate=100.0,
                total_discrepancies=0,
                critical_discrepancies=0,
                last_reconciliation_time=datetime.now(),
                recent_discrepancies=[]
            )
            
            agent.position_reconciler = mock_reconciler
            
            # Test complete workflow
            await agent._startup_reconciliation()
            await agent._post_trade_reconciliation_check("buy")
            await agent._periodic_reconciliation_check()
            
            # Verify all reconciliation methods were called
            mock_reconciler.startup_reconciliation.assert_called_once()
            mock_reconciler.perform_reconciliation.assert_called()
            
            logger.info("✅ Complete integration workflow validated")

class ReconciliationPerformanceTests(unittest.IsolatedAsyncioTestCase):
    """Performance and stress tests for reconciliation system"""
    
    async def test_reconciliation_performance(self):
        """Test reconciliation system performance under load"""
        logger.info("🧪 Testing Reconciliation Performance")
        
        mock_oanda = Mock()
        mock_oanda.api.position.get.return_value = Mock(
            body={'position': {
                'instrument': 'GBP_JPY',
                'long': {'units': '0'},
                'short': {'units': '0'},
                'unrealizedPL': '0'
            }}
        )
        
        reconciler = BrokerPositionReconciler(
            oanda_executor=mock_oanda,
            instrument="GBP_JPY"
        )
        
        # Measure performance of multiple reconciliations
        start_time = datetime.now()
        
        for i in range(100):
            await reconciler.perform_reconciliation("GBP_JPY", f"performance_test_{i}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.assertLess(duration, 10.0)  # Should complete 100 reconciliations in under 10 seconds
        logger.info(f"✅ Performance test: 100 reconciliations in {duration:.2f}s")

async def run_all_tests():
    """Run all reconciliation system tests"""
    logger.info("🚀 Starting Comprehensive Reconciliation System Tests")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    test_loader = unittest.TestLoader()
    suite.addTests(test_loader.loadTestsFromTestCase(ReconciliationSystemTestSuite))
    suite.addTests(test_loader.loadTestsFromTestCase(ReconciliationPerformanceTests))
    
    # Run tests with custom runner
    class AsyncTestRunner:
        def __init__(self):
            self.results = []
            
        async def run_suite(self, test_suite):
            for test_group in test_suite:
                if hasattr(test_group, '_tests'):
                    for test in test_group._tests:
                        try:
                            logger.info(f"Running: {test._testMethodName}")
                            await test.debug()
                            self.results.append((test._testMethodName, "PASS", None))
                            logger.info(f"✅ {test._testMethodName} PASSED")
                        except Exception as e:
                            self.results.append((test._testMethodName, "FAIL", str(e)))
                            logger.error(f"❌ {test._testMethodName} FAILED: {e}")
            
            return self.results
    
    # Run all tests
    runner = AsyncTestRunner()
    results = await runner.run_suite(suite)
    
    # Summary
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    
    logger.info(f"📊 Test Results: {passed} PASSED, {failed} FAILED")
    
    if failed > 0:
        logger.error("❌ Some tests failed:")
        for test_name, status, error in results:
            if status == "FAIL":
                logger.error(f"   {test_name}: {error}")
    else:
        logger.info("🎉 All tests passed!")
    
    return failed == 0

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)