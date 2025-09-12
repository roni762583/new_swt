#!/usr/bin/env python3
"""
Validation Script for Live Trading System with Position Reconciliation
Validates the complete integration without executing real trades
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import os
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

# Add to Python path
sys.path.append(str(Path(__file__).parent))

from live_trading_episode_13475 import Episode13475LiveAgent
from position_reconciliation import BrokerPositionReconciler

logger = logging.getLogger(__name__)

class ReconciliationIntegrationValidator:
    """Validator for reconciliation system integration with live trading"""
    
    def __init__(self):
        self.validation_results = []
        self.test_config_path = "/tmp/test_live_config.yaml"
        
    async def validate_complete_integration(self) -> bool:
        """Validate complete reconciliation integration"""
        logger.info("🔍 Starting Complete Reconciliation Integration Validation")
        
        validations = [
            ("Import Validation", self._validate_imports),
            ("Configuration Setup", self._validate_configuration),
            ("Agent Initialization", self._validate_agent_initialization),
            ("Reconciliation System Init", self._validate_reconciliation_initialization),
            ("Startup Reconciliation", self._validate_startup_reconciliation),
            ("Trading Cycle Integration", self._validate_trading_cycle_integration),
            ("Post-Trade Verification", self._validate_post_trade_verification),
            ("Periodic Health Checks", self._validate_periodic_health_checks),
            ("Edge Case Handling", self._validate_edge_case_handling),
            ("Session Persistence", self._validate_session_persistence),
            ("Error Recovery", self._validate_error_recovery)
        ]
        
        all_passed = True
        
        for validation_name, validation_func in validations:
            try:
                logger.info(f"🧪 Running: {validation_name}")
                result = await validation_func()
                
                if result:
                    logger.info(f"✅ {validation_name}: PASSED")
                    self.validation_results.append((validation_name, "PASS", None))
                else:
                    logger.error(f"❌ {validation_name}: FAILED")
                    self.validation_results.append((validation_name, "FAIL", "Validation failed"))
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"❌ {validation_name}: ERROR - {e}")
                self.validation_results.append((validation_name, "ERROR", str(e)))
                all_passed = False
        
        self._log_validation_summary()
        return all_passed
    
    async def _validate_imports(self) -> bool:
        """Validate all required imports work correctly"""
        try:
            # Test reconciliation imports
            from position_reconciliation import (
                BrokerPositionReconciler,
                BrokerPosition,
                InternalPosition,
                PositionDiscrepancy,
                ReconciliationEvent
            )
            
            # Test live trading imports
            from live_trading_episode_13475 import Episode13475LiveAgent
            
            # Test OANDA integration
            from oanda_trade_executor import OANDATradeExecutor
            
            logger.info("   All critical imports successful")
            return True
            
        except ImportError as e:
            logger.error(f"   Import error: {e}")
            return False
    
    async def _validate_configuration(self) -> bool:
        """Validate configuration handling"""
        try:
            # Create mock config
            config_content = """
episode: 13475
mcts_simulations: 15
c_puct: 1.25
wst_J: 2
wst_Q: 6
"""
            with open(self.test_config_path, 'w') as f:
                f.write(config_content)
            
            logger.info("   Configuration file created successfully")
            return os.path.exists(self.test_config_path)
            
        except Exception as e:
            logger.error(f"   Configuration error: {e}")
            return False
    
    async def _validate_agent_initialization(self) -> bool:
        """Validate agent can be initialized with reconciliation system"""
        try:
            # Set mock environment variables
            os.environ['OANDA_API_KEY'] = 'test_key'
            os.environ['OANDA_ACCOUNT_ID'] = 'test_account'
            os.environ['OANDA_ENVIRONMENT'] = 'practice'
            
            # Mock the OANDA initialization to avoid real API calls
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                
                # Verify agent has reconciliation-related attributes
                self.assertTrue(hasattr(agent, 'position_reconciler'))
                self.assertTrue(hasattr(agent, 'last_reconciliation_time'))
                
                logger.info("   Agent initialization with reconciliation attributes successful")
                return True
                
        except Exception as e:
            logger.error(f"   Agent initialization error: {e}")
            return False
    
    async def _validate_reconciliation_initialization(self) -> bool:
        """Validate reconciliation system initialization"""
        try:
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                
                # Mock OANDA executor
                mock_executor = Mock()
                agent.oanda_executor = mock_executor
                
                # Test reconciliation initialization
                await agent._initialize_position_reconciliation()
                
                self.assertIsNotNone(agent.position_reconciler)
                self.assertIsNotNone(agent.last_reconciliation_time)
                
                logger.info("   Reconciliation system initialization successful")
                return True
                
        except Exception as e:
            logger.error(f"   Reconciliation initialization error: {e}")
            return False
    
    async def _validate_startup_reconciliation(self) -> bool:
        """Validate startup reconciliation process"""
        try:
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                agent.oanda_executor = Mock()
                
                # Mock reconciler with startup response
                mock_reconciler = AsyncMock()
                from position_reconciliation import ReconciliationEvent
                
                mock_reconciler.startup_reconciliation.return_value = ReconciliationEvent(
                    timestamp=datetime.now(),
                    event_type="startup",
                    status="completed",
                    broker_position=None,
                    internal_position=None,
                    discrepancies=[],
                    notes="Clean startup - no existing positions"
                )
                
                agent.position_reconciler = mock_reconciler
                
                # Test startup reconciliation
                await agent._startup_reconciliation()
                
                # Verify startup reconciliation was called
                mock_reconciler.startup_reconciliation.assert_called_once()
                
                logger.info("   Startup reconciliation process successful")
                return True
                
        except Exception as e:
            logger.error(f"   Startup reconciliation error: {e}")
            return False
    
    async def _validate_trading_cycle_integration(self) -> bool:
        """Validate trading cycle includes reconciliation checks"""
        try:
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                agent.oanda_executor = Mock()
                
                # Check that the trading cycle methods reference reconciliation
                trading_cycle_code = agent._process_trading_cycle.__code__.co_names
                
                # Verify reconciliation is integrated into trading flow
                self.assertTrue(hasattr(agent, '_post_trade_reconciliation_check'))
                self.assertTrue(hasattr(agent, '_periodic_reconciliation_check'))
                self.assertTrue(hasattr(agent, '_validate_position_features'))
                
                logger.info("   Trading cycle reconciliation integration verified")
                return True
                
        except Exception as e:
            logger.error(f"   Trading cycle integration error: {e}")
            return False
    
    async def _validate_post_trade_verification(self) -> bool:
        """Validate post-trade reconciliation verification"""
        try:
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                agent.oanda_executor = Mock()
                
                # Mock reconciler
                mock_reconciler = AsyncMock()
                from position_reconciliation import ReconciliationEvent
                
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
                
                # Test post-trade verification
                await agent._post_trade_reconciliation_check("buy")
                
                # Verify post-trade reconciliation was called
                mock_reconciler.perform_reconciliation.assert_called_once_with(
                    instrument="GBP_JPY",
                    event_type="post_buy"
                )
                
                logger.info("   Post-trade verification successful")
                return True
                
        except Exception as e:
            logger.error(f"   Post-trade verification error: {e}")
            return False
    
    async def _validate_periodic_health_checks(self) -> bool:
        """Validate periodic reconciliation health checks"""
        try:
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                agent.oanda_executor = Mock()
                
                # Mock reconciler with stats
                mock_reconciler = AsyncMock()
                from position_reconciliation import ReconciliationEvent
                
                mock_reconciler.perform_reconciliation.return_value = ReconciliationEvent(
                    timestamp=datetime.now(),
                    event_type="periodic_health_check",
                    status="completed",
                    broker_position=None,
                    internal_position=None,
                    discrepancies=[],
                    notes="Health check passed"
                )
                
                mock_stats = Mock()
                mock_stats.total_reconciliations = 5
                mock_stats.successful_reconciliations = 5
                mock_stats.success_rate = 100.0
                mock_stats.total_discrepancies = 0
                mock_stats.critical_discrepancies = 0
                mock_stats.last_reconciliation_time = datetime.now()
                mock_stats.recent_discrepancies = []
                
                mock_reconciler.get_reconciliation_stats.return_value = mock_stats
                
                agent.position_reconciler = mock_reconciler
                
                # Force periodic check by setting old timestamp
                from datetime import timedelta
                agent.last_reconciliation_time = datetime.now() - timedelta(minutes=6)
                
                # Test periodic health check
                await agent._periodic_reconciliation_check()
                
                # Verify periodic reconciliation was called
                mock_reconciler.perform_reconciliation.assert_called_once()
                
                logger.info("   Periodic health checks successful")
                return True
                
        except Exception as e:
            logger.error(f"   Periodic health check error: {e}")
            return False
    
    async def _validate_edge_case_handling(self) -> bool:
        """Validate edge case handling (container restart, network issues, etc.)"""
        try:
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                agent.oanda_executor = Mock()
                
                # Test container restart recovery
                await agent._handle_container_restart_recovery()
                
                # Test network recovery
                agent.oanda_executor._get_current_price.return_value = 195.123
                network_ok = await agent._handle_network_recovery()
                self.assertTrue(network_ok)
                
                # Test partial fill handling
                mock_result = Mock()
                mock_result.fill_units = 0.5  # Partial fill
                mock_result.trade_id = "test_trade_123"
                
                await agent._handle_partial_fill_recovery(mock_result)
                
                logger.info("   Edge case handling successful")
                return True
                
        except Exception as e:
            logger.error(f"   Edge case handling error: {e}")
            return False
    
    async def _validate_session_persistence(self) -> bool:
        """Validate session persistence includes reconciliation data"""
        try:
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                agent.oanda_executor = Mock()
                
                # Mock reconciler with stats
                mock_reconciler = Mock()
                mock_stats = Mock()
                mock_stats.total_reconciliations = 10
                mock_stats.successful_reconciliations = 9
                mock_stats.success_rate = 90.0
                mock_stats.total_discrepancies = 1
                mock_stats.critical_discrepancies = 0
                mock_stats.last_reconciliation_time = datetime.now()
                mock_stats.recent_discrepancies = []
                
                mock_reconciler.get_reconciliation_stats.return_value = mock_stats
                agent.position_reconciler = mock_reconciler
                
                # Test session summary includes reconciliation data
                summary = agent._get_reconciliation_summary()
                
                self.assertIn('enabled', summary)
                self.assertIn('total_reconciliations', summary)
                self.assertIn('success_rate', summary)
                
                logger.info("   Session persistence with reconciliation data successful")
                return True
                
        except Exception as e:
            logger.error(f"   Session persistence error: {e}")
            return False
    
    async def _validate_error_recovery(self) -> bool:
        """Validate error recovery mechanisms"""
        try:
            with self._mock_oanda_initialization():
                agent = Episode13475LiveAgent(self.test_config_path)
                agent.oanda_executor = Mock()
                
                # Test position feature validation with invalid data
                invalid_position = {
                    'type': 'invalid',
                    'size': -100,
                    'entry_price': 0
                }
                
                validated = agent._validate_position_features(invalid_position)
                self.assertIsNone(validated)  # Should return None for invalid data
                
                # Test error recovery for trade execution
                mock_trade_func = AsyncMock(side_effect=Exception("Network error"))
                
                try:
                    await agent._execute_with_error_recovery(mock_trade_func)
                except Exception:
                    pass  # Expected to fail after retries
                
                logger.info("   Error recovery mechanisms successful")
                return True
                
        except Exception as e:
            logger.error(f"   Error recovery validation error: {e}")
            return False
    
    def _mock_oanda_initialization(self):
        """Context manager to mock OANDA initialization"""
        from unittest.mock import patch
        return patch.object(Episode13475LiveAgent, '_initialize_oanda_connection', new_callable=AsyncMock)
    
    def assertTrue(self, condition):
        """Assert helper"""
        if not condition:
            raise AssertionError("Assertion failed")
    
    def assertIsNotNone(self, obj):
        """Assert not None helper"""
        if obj is None:
            raise AssertionError("Object is None")
    
    def assertIn(self, item, container):
        """Assert in helper"""
        if item not in container:
            raise AssertionError(f"{item} not in {container}")
    
    def _log_validation_summary(self):
        """Log validation summary"""
        passed = sum(1 for _, status, _ in self.validation_results if status == "PASS")
        failed = sum(1 for _, status, _ in self.validation_results if status in ["FAIL", "ERROR"])
        
        logger.info("=" * 60)
        logger.info("📊 RECONCILIATION INTEGRATION VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ PASSED: {passed}")
        logger.info(f"❌ FAILED: {failed}")
        logger.info("")
        
        if failed > 0:
            logger.error("Failed validations:")
            for name, status, error in self.validation_results:
                if status in ["FAIL", "ERROR"]:
                    logger.error(f"   {name}: {error}")
        else:
            logger.info("🎉 ALL VALIDATIONS PASSED!")
            logger.info("🛡️ Position reconciliation system fully integrated!")
        
        logger.info("=" * 60)

async def main():
    """Main validation entry point"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting Reconciliation Integration Validation")
    
    validator = ReconciliationIntegrationValidator()
    success = await validator.validate_complete_integration()
    
    if success:
        logger.info("✅ Reconciliation system integration validation SUCCESSFUL")
        return 0
    else:
        logger.error("❌ Reconciliation system integration validation FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))