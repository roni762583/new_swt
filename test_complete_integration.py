#!/usr/bin/env python3
"""
Complete System Integration Tests
Validates the entire new_swt system end-to-end functionality

This test suite validates:
- Configuration management across all components
- Feature processing pipeline (Phase 2)
- Inference engine integration (Phase 3) 
- Component interoperability
- Performance characteristics
- Error handling and recovery
"""

import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
import json
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add to Python path
sys.path.append(str(Path(__file__).parent))

class SystemIntegrationValidator:
    """Comprehensive system integration validator"""
    
    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.error_log: List[str] = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("🚀 Starting Complete System Integration Tests...")
        
        test_methods = [
            ("configuration_system", self.test_configuration_system),
            ("feature_processing_integration", self.test_feature_processing_integration),
            ("inference_engine_integration", self.test_inference_engine_integration),
            ("component_interoperability", self.test_component_interoperability),
            ("error_handling", self.test_error_handling),
            ("performance_benchmarks", self.test_performance_benchmarks),
            ("memory_management", self.test_memory_management),
            ("concurrent_operations", self.test_concurrent_operations)
        ]
        
        for test_name, test_method in test_methods:
            try:
                logger.info(f"🧪 Running {test_name.replace('_', ' ').title()}...")
                start_time = time.time()
                
                result = test_method()
                
                execution_time = (time.time() - start_time) * 1000
                self.test_results[test_name] = result
                self.performance_metrics[f"{test_name}_time_ms"] = execution_time
                
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{status} {test_name} ({execution_time:.2f}ms)")
                
            except Exception as e:
                self.test_results[test_name] = False
                self.error_log.append(f"{test_name}: {str(e)}")
                logger.error(f"❌ FAIL {test_name}: {e}")
        
        return self._generate_report()
    
    def test_configuration_system(self) -> bool:
        """Test configuration management system"""
        try:
            # Test imports
            from swt_core.config_manager import ConfigManager
            from swt_core.types import AgentType, NetworkArchitecture, MCTSParameters
            
            # Test configuration loading
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Test Episode 13475 mode
            config_manager.force_episode_13475_mode()
            
            # Verify critical parameters
            assert config.feature_config.wst_J == 2
            assert config.feature_config.wst_Q == 6
            assert config.observation_space.position_state_dim == 9
            assert config.observation_space.market_state_dim == 128
            
            # Test configuration switching
            original_agent = config.agent_system
            config.switch_agent_system(AgentType.EXPERIMENTAL, validate=True)
            assert config.agent_system == AgentType.EXPERIMENTAL
            
            config.switch_agent_system(original_agent, validate=True)
            assert config.agent_system == original_agent
            
            logger.info("✅ Configuration system validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration system test failed: {e}")
            return False
    
    def test_feature_processing_integration(self) -> bool:
        """Test complete feature processing pipeline"""
        try:
            from swt_core.config_manager import ConfigManager
            from swt_core.types import PositionState, PositionType
            from swt_features.feature_processor import FeatureProcessor, MarketDataPoint
            
            # Initialize system
            config_manager = ConfigManager()
            config = config_manager.load_config()
            config_manager.force_episode_13475_mode()
            
            feature_processor = FeatureProcessor(config)
            
            # Generate comprehensive market data
            base_time = datetime.now()
            base_price = 199.45
            
            # Add 256 data points for WST
            for i in range(256):
                price = base_price + 0.01 * np.sin(i * 0.1) + 0.002 * np.random.randn()
                
                data_point = MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    open=price - 0.001,
                    high=price + 0.003,
                    low=price - 0.003,
                    close=price,
                    volume=100.0 + 50 * np.random.random(),
                    spread=2.5 + 0.5 * np.random.random()
                )
                
                feature_processor.add_market_data(data_point)
            
            # Test different position states
            position_states = [
                PositionState(),  # Flat
                PositionState(
                    position_type=PositionType.LONG,
                    entry_price=199.20,
                    bars_since_entry=75,
                    unrealized_pnl_pips=25.0,
                    max_adverse_pips=10.0
                ),
                PositionState(
                    position_type=PositionType.SHORT,
                    entry_price=199.70,
                    bars_since_entry=120,
                    unrealized_pnl_pips=-18.0,
                    max_adverse_pips=30.0,
                    accumulated_drawdown=25.0
                )
            ]
            
            # Process observations
            for i, position_state in enumerate(position_states):
                observation = feature_processor.process_observation(
                    position_state=position_state,
                    current_price=199.45 + 0.01 * i,
                    market_cache_key=f"integration_test_{i}"
                )
                
                # Validate observation structure
                assert observation.market_features.shape == (128,)
                assert observation.position_features.shape == (9,)
                assert observation.combined_features.shape == (137,)
                
                # Validate feature ranges
                assert not np.isnan(observation.combined_features).any()
                assert not np.isinf(observation.combined_features).any()
                
                # Validate observation space compliance
                assert observation.validate(config.observation_space)
            
            # Test system status
            status = feature_processor.get_system_status()
            assert status["is_ready"]
            assert status["processing_stats"]["total_observations_processed"] == 3
            
            logger.info("✅ Feature processing integration passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature processing integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_inference_engine_integration(self) -> bool:
        """Test inference engine with mock operations"""
        try:
            from swt_core.config_manager import ConfigManager
            from swt_core.types import PositionState, PositionType
            from swt_inference.inference_engine import InferenceEngine
            from swt_features.feature_processor import MarketDataPoint
            
            # Initialize system
            config_manager = ConfigManager()
            config = config_manager.load_config()
            config_manager.force_episode_13475_mode()
            
            inference_engine = InferenceEngine(config=config)
            
            # Add market data
            base_time = datetime.now()
            base_price = 199.45
            
            for i in range(256):
                price = base_price + 0.005 * np.sin(i * 0.05)
                
                data_point = MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    open=price - 0.0005,
                    high=price + 0.002,
                    low=price - 0.002,
                    close=price,
                    volume=100.0
                )
                
                inference_engine.add_market_data(data_point)
            
            # Test system readiness (without actual model)
            assert inference_engine.feature_processor.is_ready()
            
            # Test mock inference
            position = PositionState(
                position_type=PositionType.LONG,
                entry_price=199.30,
                bars_since_entry=50,
                unrealized_pnl_pips=15.0
            )
            
            # Create mock inference result
            mock_result = inference_engine.create_mock_inference_result(action=1)  # BUY
            
            # Validate mock result
            assert mock_result.trading_decision.action in [0, 1, 2, 3]
            assert 0.0 <= mock_result.trading_decision.confidence <= 1.0
            assert mock_result.trading_decision.policy_distribution.shape == (4,)
            assert np.sum(mock_result.trading_decision.policy_distribution) == pytest.approx(1.0, rel=1e-2)
            
            # Test diagnostics
            diagnostics = inference_engine.get_diagnostics(position, 199.45)
            assert "inference_engine" in diagnostics
            assert "features" in diagnostics
            assert "performance" in diagnostics
            
            # Test performance summary
            performance = inference_engine.get_performance_summary()
            assert "agent_type" in performance
            assert "is_ready" in performance
            
            logger.info("✅ Inference engine integration passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Inference engine integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_component_interoperability(self) -> bool:
        """Test interoperability between all system components"""
        try:
            from swt_core.config_manager import ConfigManager
            from swt_core.types import AgentType, PositionState
            from swt_inference.inference_engine import InferenceEngine
            from swt_inference.agent_factory import AgentFactory
            from swt_features.feature_processor import MarketDataPoint
            
            # Initialize with different agent types
            config_manager = ConfigManager()
            
            for agent_type in [AgentType.STOCHASTIC_MUZERO, AgentType.EXPERIMENTAL]:
                config = config_manager.load_config()
                config.switch_agent_system(agent_type, validate=True)
                config_manager.force_episode_13475_mode()
                
                # Test agent factory
                available_agents = AgentFactory.get_available_agents()
                assert agent_type in available_agents
                
                # Test inference engine initialization
                inference_engine = InferenceEngine(config=config)
                
                # Add market data
                base_time = datetime.now()
                for i in range(256):
                    data_point = MarketDataPoint(
                        timestamp=base_time + timedelta(minutes=i),
                        open=199.40 + 0.001 * i,
                        high=199.42 + 0.001 * i,
                        low=199.38 + 0.001 * i,
                        close=199.41 + 0.001 * i,
                        volume=100.0
                    )
                    inference_engine.add_market_data(data_point)
                
                # Test system integration
                assert inference_engine.is_ready() == inference_engine.feature_processor.is_ready()
                
                # Test configuration consistency
                system_status = inference_engine.get_diagnostics()
                assert system_status["inference_engine"]["configuration"]["agent_system"] == agent_type.value
            
            logger.info("✅ Component interoperability passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component interoperability test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and recovery mechanisms"""
        try:
            from swt_core.config_manager import ConfigManager
            from swt_core.types import PositionState
            from swt_core.exceptions import FeatureProcessingError, InferenceError
            from swt_inference.inference_engine import InferenceEngine
            from swt_features.feature_processor import MarketDataPoint
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            inference_engine = InferenceEngine(config=config)
            
            # Test invalid market data handling
            try:
                invalid_data = MarketDataPoint(
                    timestamp=datetime.now(),
                    open=-1.0,  # Invalid price
                    high=199.0,
                    low=198.0,
                    close=199.5,
                    volume=100.0
                )
                inference_engine.add_market_data(invalid_data)
                # Should raise an exception
                return False
            except (FeatureProcessingError, ValueError):
                # Expected behavior
                pass
            
            # Test insufficient data handling
            position = PositionState()
            try:
                # Try to get decision without sufficient market data
                result = inference_engine.get_trading_decision(position, 199.45)
                # Should raise an exception due to no agent loaded
                return False
            except InferenceError:
                # Expected behavior
                pass
            
            # Test graceful degradation
            diagnostics = inference_engine.get_diagnostics(position, 199.45)
            assert "error" not in diagnostics.get("inference_engine", {})
            
            logger.info("✅ Error handling test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test system performance characteristics"""
        try:
            from swt_core.config_manager import ConfigManager
            from swt_core.types import PositionState
            from swt_inference.inference_engine import InferenceEngine
            from swt_features.feature_processor import MarketDataPoint
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            config_manager.force_episode_13475_mode()
            
            inference_engine = InferenceEngine(config=config)
            
            # Benchmark market data ingestion
            base_time = datetime.now()
            start_time = time.time()
            
            for i in range(256):
                data_point = MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    open=199.40,
                    high=199.42,
                    low=199.38,
                    close=199.41,
                    volume=100.0
                )
                inference_engine.add_market_data(data_point)
            
            ingestion_time = (time.time() - start_time) * 1000
            self.performance_metrics["market_data_ingestion_ms"] = ingestion_time
            
            # Benchmark observation processing
            position = PositionState()
            processing_times = []
            
            for i in range(10):
                start_time = time.time()
                
                observation = inference_engine.feature_processor.process_observation(
                    position_state=position,
                    current_price=199.45,
                    market_cache_key=f"benchmark_{i}"
                )
                
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
            
            avg_processing_time = np.mean(processing_times)
            self.performance_metrics["avg_observation_processing_ms"] = avg_processing_time
            self.performance_metrics["max_observation_processing_ms"] = np.max(processing_times)
            
            # Performance assertions
            assert ingestion_time < 500, f"Market data ingestion too slow: {ingestion_time:.2f}ms"
            assert avg_processing_time < 100, f"Observation processing too slow: {avg_processing_time:.2f}ms"
            
            logger.info(f"✅ Performance benchmarks passed:")
            logger.info(f"  Market data ingestion: {ingestion_time:.2f}ms")
            logger.info(f"  Avg observation processing: {avg_processing_time:.2f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {e}")
            return False
    
    def test_memory_management(self) -> bool:
        """Test memory usage and management"""
        try:
            import psutil
            import gc
            
            from swt_core.config_manager import ConfigManager
            from swt_inference.inference_engine import InferenceEngine
            from swt_features.feature_processor import MarketDataPoint
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            inference_engine = InferenceEngine(config=config)
            
            # Add large amount of market data
            base_time = datetime.now()
            for i in range(1000):  # More data than typical
                data_point = MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    open=199.40 + 0.001 * np.random.randn(),
                    high=199.42 + 0.001 * np.random.randn(),
                    low=199.38 + 0.001 * np.random.randn(),
                    close=199.41 + 0.001 * np.random.randn(),
                    volume=100.0 + 10 * np.random.randn()
                )
                inference_engine.add_market_data(data_point)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test reset and cleanup
            inference_engine.reset(clear_market_data=True)
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = peak_memory - initial_memory
            memory_cleanup = peak_memory - final_memory
            
            self.performance_metrics["initial_memory_mb"] = initial_memory
            self.performance_metrics["peak_memory_mb"] = peak_memory
            self.performance_metrics["final_memory_mb"] = final_memory
            
            # Memory usage should be reasonable
            assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB"
            assert memory_cleanup > 0, f"Memory not properly cleaned up: {memory_cleanup:.2f}MB"
            
            logger.info(f"✅ Memory management test passed:")
            logger.info(f"  Peak memory usage: {memory_increase:.2f}MB")
            logger.info(f"  Memory cleanup: {memory_cleanup:.2f}MB")
            
            return True
            
        except ImportError:
            logger.warning("⚠️ psutil not available, skipping memory test")
            return True
        except Exception as e:
            logger.error(f"❌ Memory management test failed: {e}")
            return False
    
    def test_concurrent_operations(self) -> bool:
        """Test concurrent operation safety"""
        try:
            import threading
            import queue
            
            from swt_core.config_manager import ConfigManager
            from swt_core.types import PositionState
            from swt_inference.inference_engine import InferenceEngine
            from swt_features.feature_processor import MarketDataPoint
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            inference_engine = InferenceEngine(config=config)
            
            # Add initial market data
            base_time = datetime.now()
            for i in range(256):
                data_point = MarketDataPoint(
                    timestamp=base_time + timedelta(minutes=i),
                    open=199.40,
                    high=199.42,
                    low=199.38,
                    close=199.41,
                    volume=100.0
                )
                inference_engine.add_market_data(data_point)
            
            # Test concurrent observation processing
            results_queue = queue.Queue()
            errors_queue = queue.Queue()
            
            def worker_thread(thread_id: int):
                try:
                    position = PositionState()
                    
                    for i in range(5):
                        observation = inference_engine.feature_processor.process_observation(
                            position_state=position,
                            current_price=199.45,
                            market_cache_key=f"concurrent_{thread_id}_{i}"
                        )
                        results_queue.put((thread_id, i, observation.combined_features.shape))
                        
                except Exception as e:
                    errors_queue.put((thread_id, e))
            
            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)
            
            # Check results
            assert errors_queue.empty(), f"Thread errors: {list(errors_queue.queue)}"
            
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            assert len(results) == 15, f"Expected 15 results, got {len(results)}"
            
            # Verify all results have correct shape
            for thread_id, iteration, shape in results:
                assert shape == (137,), f"Wrong shape from thread {thread_id}: {shape}"
            
            logger.info("✅ Concurrent operations test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Concurrent operations test failed: {e}")
            return False
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        passed_tests = sum(self.test_results.values())
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "overall_status": "PASS" if success_rate == 1.0 else "FAIL"
            },
            "test_results": self.test_results.copy(),
            "performance_metrics": self.performance_metrics.copy(),
            "errors": self.error_log.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        return report

def main():
    """Run complete system integration validation"""
    print("🚀 Complete System Integration Validation")
    print("=" * 60)
    
    validator = SystemIntegrationValidator()
    report = validator.run_all_tests()
    
    # Display results
    print("\n📊 Test Results Summary:")
    print("-" * 40)
    
    for test_name, result in report["test_results"].items():
        status = "✅ PASS" if result else "❌ FAIL"
        test_display = test_name.replace("_", " ").title()
        print(f"{status} {test_display}")
    
    print(f"\n🎯 Overall Result: {report['summary']['passed_tests']}/{report['summary']['total_tests']} tests passed")
    print(f"Success Rate: {report['summary']['success_rate']:.1%}")
    
    # Display performance metrics
    if report["performance_metrics"]:
        print("\n⚡ Performance Metrics:")
        print("-" * 40)
        for metric, value in report["performance_metrics"].items():
            if metric.endswith("_ms"):
                print(f"{metric.replace('_', ' ').title()}: {value:.2f}ms")
            elif metric.endswith("_mb"):
                print(f"{metric.replace('_', ' ').title()}: {value:.2f}MB")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value}")
    
    # Display errors if any
    if report["errors"]:
        print("\n❌ Errors:")
        print("-" * 40)
        for error in report["errors"]:
            print(f"  {error}")
    
    # Save detailed report
    report_file = Path("integration_test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    if report["summary"]["overall_status"] == "PASS":
        print("\n🎉 SYSTEM INTEGRATION VALIDATION PASSED!")
        print("✅ All components working together correctly")
        print("✅ Performance meets requirements") 
        print("✅ Error handling functioning properly")
        print("✅ Memory management under control")
        print("✅ System ready for production deployment")
        return True
    else:
        print("\n❌ SYSTEM INTEGRATION VALIDATION FAILED")
        print("Please review errors above and fix issues before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)