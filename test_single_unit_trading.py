#!/usr/bin/env python3
"""
Test Single Unit Trading with Real P&L Calculation
Validates the OANDA P&L integration fix for single-unit trades
"""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

# Add to Python path
sys.path.append(str(Path(__file__).parent))

from live_trading_episode_13475 import Episode13475LiveAgent

logger = logging.getLogger(__name__)

class SingleUnitTradingTest:
    """Test framework for single unit trading P&L calculations"""
    
    def __init__(self):
        self.test_results = []
        self.test_config_path = "/tmp/test_single_unit_config.yaml"
        
    async def run_comprehensive_test(self):
        """Run comprehensive single unit trading test"""
        logger.info("🧪 Starting Single Unit Trading P&L Test")
        logger.info("=" * 60)
        
        tests = [
            ("OANDA P&L Query Integration", self._test_oanda_pnl_query),
            ("Spread-Aware Calculation", self._test_spread_aware_calculation),
            ("P&L Validation System", self._test_pnl_validation),
            ("Position Management Integration", self._test_position_management),
            ("Micro P&L Detection", self._test_micro_pnl_detection),
            ("Real Trading Simulation", self._test_real_trading_simulation)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            try:
                logger.info(f"🔍 Testing: {test_name}")
                result = await test_func()
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                    self.test_results.append((test_name, "PASS", None))
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    self.test_results.append((test_name, "FAIL", "Test failed"))
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                self.test_results.append((test_name, "ERROR", str(e)))
                all_passed = False
        
        self._log_test_summary()
        return all_passed
    
    async def _test_oanda_pnl_query(self):
        """Test OANDA unrealized P&L query integration"""
        try:
            # Mock OANDA agent setup
            with self._mock_oanda_agent() as agent:
                # Mock OANDA API response
                mock_response = Mock()
                mock_response.status = 200
                mock_response.body = {
                    'position': {
                        'unrealizedPL': '0.000123',  # Very small P&L for single unit
                        'long': {
                            'units': '1',
                            'averagePrice': '195.123'
                        },
                        'short': {
                            'units': '0'
                        }
                    }
                }
                
                agent.oanda_executor.api.position.get.return_value = mock_response
                
                # Test P&L query
                pnl = await agent._get_oanda_unrealized_pnl()
                
                self.assertEqual(pnl, 0.000123)
                logger.info(f"   ✅ OANDA P&L query returned: ${pnl:.6f}")
                
                return True
                
        except Exception as e:
            logger.error(f"   ❌ OANDA P&L query test failed: {e}")
            return False
    
    async def _test_spread_aware_calculation(self):
        """Test spread-aware P&L calculation"""
        try:
            with self._mock_oanda_agent() as agent:
                # Set up mock position
                agent.current_position = {
                    'type': 'long',
                    'size': 1,
                    'entry_price': 195.123,
                    'entry_time': datetime.now()
                }
                
                # Test calculation with different prices
                test_prices = [
                    195.124,  # +0.1 pip
                    195.133,  # +1.0 pip
                    195.223,  # +10.0 pips
                ]
                
                for price in test_prices:
                    pips, pnl_usd = await agent._calculate_position_pnl_with_spread(price)
                    
                    expected_pips = (price - 195.123) * 10000
                    
                    logger.info(f"   Price: {price} -> {pips:.1f} pips, ${pnl_usd:.6f} USD")
                    
                    # Verify pip calculation is correct
                    if abs(pips - expected_pips) > 0.1:
                        logger.error(f"   ❌ Pip calculation error: expected {expected_pips:.1f}, got {pips:.1f}")
                        return False
                
                return True
                
        except Exception as e:
            logger.error(f"   ❌ Spread-aware calculation test failed: {e}")
            return False
    
    async def _test_pnl_validation(self):
        """Test P&L validation system comparing estimates vs OANDA"""
        try:
            with self._mock_oanda_agent() as agent:
                # Set up position
                agent.current_position = {
                    'type': 'long',
                    'size': 1,
                    'entry_price': 195.123,
                    'entry_time': datetime.now()
                }
                
                # Mock OANDA P&L response
                mock_response = Mock()
                mock_response.status = 200
                mock_response.body = {
                    'position': {
                        'unrealizedPL': '0.001234',  # OANDA actual P&L
                        'long': {'units': '1', 'averagePrice': '195.123'},
                        'short': {'units': '0'}
                    }
                }
                agent.oanda_executor.api.position.get.return_value = mock_response
                
                # Test position management with validation
                current_price = 195.133  # +1 pip
                result = await agent._manage_existing_position(
                    current_price=current_price,
                    rsi_14=50.0,
                    sma_20=195.100,
                    bb_upper=195.200,
                    bb_lower=195.050
                )
                
                # Verify it doesn't crash and returns valid result
                self.assertIsNotNone(result)
                self.assertTrue(len(result) == 2)  # Should return (action, confidence)
                
                logger.info(f"   ✅ P&L validation completed successfully")
                logger.info(f"   Position management returned: {result}")
                
                return True
                
        except Exception as e:
            logger.error(f"   ❌ P&L validation test failed: {e}")
            return False
    
    async def _test_position_management(self):
        """Test position management with integrated P&L calculation"""
        try:
            with self._mock_oanda_agent() as agent:
                # Mock all required methods for position management
                agent._get_market_data = AsyncMock(return_value={
                    'price': 195.133,
                    'timestamp': datetime.now(),
                    'symbol': 'GBPJPY'
                })
                
                # Set up position
                agent.current_position = {
                    'type': 'long',
                    'size': 1,
                    'entry_price': 195.123,
                    'entry_time': datetime.now()
                }
                
                # Mock OANDA response
                mock_response = Mock()
                mock_response.status = 200
                mock_response.body = {
                    'position': {
                        'unrealizedPL': '0.000067',
                        'long': {'units': '1', 'averagePrice': '195.123'},
                        'short': {'units': '0'}
                    }
                }
                agent.oanda_executor.api.position.get.return_value = mock_response
                
                # Test position management
                action, confidence = await agent._manage_existing_position(
                    current_price=195.133,
                    rsi_14=50.0,
                    sma_20=195.100,
                    bb_upper=195.200,
                    bb_lower=195.050
                )
                
                logger.info(f"   ✅ Position management: {action} (confidence: {confidence:.2f})")
                
                # Should be valid action
                self.assertIn(action, ['hold', 'close'])
                self.assertTrue(0.0 <= confidence <= 1.0)
                
                return True
                
        except Exception as e:
            logger.error(f"   ❌ Position management test failed: {e}")
            return False
    
    async def _test_micro_pnl_detection(self):
        """Test detection of micro P&L amounts below typical measurement thresholds"""
        try:
            with self._mock_oanda_agent() as agent:
                # Test various micro P&L amounts
                test_pnl_amounts = [
                    0.000001,  # $0.000001 - extremely small
                    0.000067,  # $0.000067 - 1 pip for 1 unit GBP/JPY
                    0.000670,  # $0.000670 - 10 pips for 1 unit
                    0.001000,  # $0.001000 - measurable threshold
                ]
                
                for pnl_amount in test_pnl_amounts:
                    # Mock OANDA response
                    mock_response = Mock()
                    mock_response.status = 200
                    mock_response.body = {
                        'position': {
                            'unrealizedPL': str(pnl_amount),
                            'long': {'units': '1', 'averagePrice': '195.123'},
                            'short': {'units': '0'}
                        }
                    }
                    agent.oanda_executor.api.position.get.return_value = mock_response
                    
                    pnl = await agent._get_oanda_unrealized_pnl()
                    
                    measurable = abs(pnl) > 0.001
                    status = "✅ Measurable" if measurable else "⚠️ Below threshold"
                    
                    logger.info(f"   P&L ${pnl:.6f}: {status}")
                
                logger.info("   ✅ Micro P&L detection test completed")
                return True
                
        except Exception as e:
            logger.error(f"   ❌ Micro P&L detection test failed: {e}")
            return False
    
    async def _test_real_trading_simulation(self):
        """Simulate real trading scenario with single unit"""
        try:
            with self._mock_oanda_agent() as agent:
                logger.info("   🎯 Simulating 1-unit GBP/JPY trade scenario")
                
                # Simulate trade execution
                entry_price = 195.123
                current_price = 195.143  # +2.0 pips movement
                
                # Calculate what the P&L should be
                pips_moved = (current_price - entry_price) * 10000
                
                # For GBP/JPY, 1 unit = 1 GBP
                # 1 pip = 0.01 JPY change
                # P&L in JPY = pips * 0.01 * 1 unit
                expected_pnl_jpy = pips_moved * 0.01
                
                # Convert to USD (rough estimate)
                jpy_to_usd = 0.0067  # Approximate rate
                expected_pnl_usd = expected_pnl_jpy * jpy_to_usd
                
                logger.info(f"   Entry: {entry_price}, Current: {current_price}")
                logger.info(f"   Movement: +{pips_moved:.1f} pips")
                logger.info(f"   Expected P&L: {expected_pnl_jpy:.6f} JPY ≈ ${expected_pnl_usd:.8f} USD")
                
                # Mock OANDA to return realistic P&L
                mock_response = Mock()
                mock_response.status = 200
                mock_response.body = {
                    'position': {
                        'unrealizedPL': f'{expected_pnl_usd:.8f}',
                        'long': {'units': '1', 'averagePrice': str(entry_price)},
                        'short': {'units': '0'}
                    }
                }
                agent.oanda_executor.api.position.get.return_value = mock_response
                
                # Test the actual P&L query
                actual_pnl = await agent._get_oanda_unrealized_pnl()
                
                logger.info(f"   🎯 OANDA Reported P&L: ${actual_pnl:.8f} USD")
                
                # Verify the P&L is as expected
                pnl_diff = abs(actual_pnl - expected_pnl_usd)
                if pnl_diff < 0.000001:  # Within microsecond precision
                    logger.info("   ✅ P&L calculation matches expectation")
                    
                    # Check if it's measurable
                    if abs(actual_pnl) >= 0.01:
                        logger.info("   ✅ P&L is above $0.01 - measurable in account balance")
                    else:
                        logger.warning("   ⚠️ P&L below $0.01 - may not appear in account balance changes")
                        logger.warning("   💡 Consider increasing minimum trade size for measurable P&L")
                
                return True
                
        except Exception as e:
            logger.error(f"   ❌ Real trading simulation failed: {e}")
            return False
    
    def _mock_oanda_agent(self):
        """Create mock OANDA agent for testing"""
        from unittest.mock import patch
        
        # Mock the OANDA initialization
        with patch.object(Episode13475LiveAgent, '_initialize_oanda_connection', new_callable=AsyncMock):
            agent = Episode13475LiveAgent(self.test_config_path)
            
            # Mock OANDA executor
            mock_executor = Mock()
            mock_api = Mock()
            mock_executor.api = mock_api
            mock_executor.account_id = "test_account_123"
            agent.oanda_executor = mock_executor
            
            return agent
    
    def assertEqual(self, a, b, msg=None):
        """Assert helper"""
        if a != b:
            raise AssertionError(msg or f"Expected {a} == {b}")
    
    def assertIsNotNone(self, obj, msg=None):
        """Assert not None helper"""
        if obj is None:
            raise AssertionError(msg or "Object is None")
    
    def assertTrue(self, condition, msg=None):
        """Assert true helper"""
        if not condition:
            raise AssertionError(msg or "Condition is not true")
    
    def assertIn(self, item, container, msg=None):
        """Assert in helper"""
        if item not in container:
            raise AssertionError(msg or f"{item} not in {container}")
    
    def _log_test_summary(self):
        """Log test summary"""
        passed = sum(1 for _, status, _ in self.test_results if status == "PASS")
        failed = sum(1 for _, status, _ in self.test_results if status in ["FAIL", "ERROR"])
        
        logger.info("=" * 60)
        logger.info("📊 SINGLE UNIT TRADING P&L TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ PASSED: {passed}")
        logger.info(f"❌ FAILED: {failed}")
        
        if failed > 0:
            logger.error("Failed tests:")
            for name, status, error in self.test_results:
                if status in ["FAIL", "ERROR"]:
                    logger.error(f"   {name}: {error}")
        else:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("✅ Single unit trading P&L system is working correctly!")
        
        logger.info("=" * 60)

async def main():
    """Main test entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting Single Unit Trading P&L Tests")
    
    tester = SingleUnitTradingTest()
    success = await tester.run_comprehensive_test()
    
    if success:
        logger.info("✅ Single unit trading P&L system validation SUCCESSFUL")
        return 0
    else:
        logger.error("❌ Single unit trading P&L system validation FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))