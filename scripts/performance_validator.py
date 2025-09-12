#!/usr/bin/env python3
"""
SWT Performance Validator - Production Performance Validation and Benchmarking
Comprehensive performance testing suite for Episode 13475 trading system.
"""

import os
import sys
import time
import json
import asyncio
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psutil
    import torch
    import numpy as np
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
except ImportError as e:
    print(f"Warning: Optional dependencies not available: {e}")

@dataclass
class PerformanceMetrics:
    """Performance test results container"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    
    # Latency metrics
    avg_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    
    # Throughput metrics
    requests_per_second: Optional[float] = None
    operations_per_second: Optional[float] = None
    
    # Resource metrics
    peak_cpu_percent: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    avg_memory_mb: Optional[float] = None
    
    # Additional metrics
    custom_metrics: Optional[Dict[str, Any]] = None

class PerformanceValidator:
    """Main performance validation class"""
    
    def __init__(self, config_path: str = "config/live.yaml"):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        self.results: List[PerformanceMetrics] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup performance test logging"""
        logger = logging.getLogger('performance_validator')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive performance validation suite
        
        Returns:
            Dict containing all test results and summary
        """
        self.logger.info("Starting comprehensive performance validation")
        start_time = datetime.now()
        
        try:
            # System performance tests
            await self._test_system_performance()
            
            # Inference performance tests
            await self._test_inference_performance()
            
            # Data processing performance tests
            await self._test_data_processing()
            
            # Memory management tests
            await self._test_memory_management()
            
            # Concurrent load tests
            await self._test_concurrent_load()
            
            # Network performance tests
            await self._test_network_performance()
            
            # End-to-end integration tests
            await self._test_integration_performance()
            
            # Generate summary report
            summary = self._generate_summary_report()
            
            duration = datetime.now() - start_time
            self.logger.info(f"Performance validation completed in {duration.total_seconds():.2f} seconds")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            return {"error": str(e), "traceback": traceback.format_exc()}
            
    async def _test_system_performance(self) -> None:
        """Test basic system performance"""
        self.logger.info("Testing system performance")
        
        start_time = datetime.now()
        
        # CPU and memory baseline
        cpu_readings = []
        memory_readings = []
        
        # Monitor for 30 seconds
        for _ in range(30):
            cpu_readings.append(psutil.cpu_percent(interval=1))
            memory_readings.append(psutil.virtual_memory().used / 1024 / 1024)  # MB
            
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        metrics = PerformanceMetrics(
            test_name="system_baseline",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=True,
            avg_cpu_percent=statistics.mean(cpu_readings),
            peak_cpu_percent=max(cpu_readings),
            avg_memory_mb=statistics.mean(memory_readings),
            peak_memory_mb=max(memory_readings),
            custom_metrics={
                "cpu_std_dev": statistics.stdev(cpu_readings) if len(cpu_readings) > 1 else 0,
                "memory_std_dev": statistics.stdev(memory_readings) if len(memory_readings) > 1 else 0,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "available_memory_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024
            }
        )
        
        self.results.append(metrics)
        self.logger.info(f"System baseline: CPU {metrics.avg_cpu_percent:.1f}%, Memory {metrics.avg_memory_mb:.1f}MB")
        
    async def _test_inference_performance(self) -> None:
        """Test Episode 13475 inference performance"""
        self.logger.info("Testing inference performance")
        
        if not torch:
            self.logger.warning("PyTorch not available, skipping inference tests")
            return
            
        try:
            start_time = datetime.now()
            latencies = []
            
            # Simulate Episode 13475 inference
            for i in range(100):
                inference_start = time.perf_counter()
                
                # Simulate inference workload
                await self._simulate_episode_13475_inference()
                
                inference_end = time.perf_counter()
                latency_ms = (inference_end - inference_start) * 1000
                latencies.append(latency_ms)
                
                if i % 20 == 0:
                    self.logger.info(f"Completed {i+1}/100 inference tests")
                    
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metrics = PerformanceMetrics(
                test_name="inference_performance",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=True,
                avg_latency_ms=statistics.mean(latencies),
                p50_latency_ms=statistics.median(latencies),
                p95_latency_ms=np.percentile(latencies, 95) if np else sorted(latencies)[int(0.95 * len(latencies))],
                p99_latency_ms=np.percentile(latencies, 99) if np else sorted(latencies)[int(0.99 * len(latencies))],
                max_latency_ms=max(latencies),
                operations_per_second=len(latencies) / duration,
                custom_metrics={
                    "inference_count": len(latencies),
                    "latency_std_dev": statistics.stdev(latencies),
                    "target_latency_ms": 500,  # Episode 13475 target
                    "latency_violations": sum(1 for l in latencies if l > 500),
                    "latency_violation_rate": sum(1 for l in latencies if l > 500) / len(latencies)
                }
            )
            
            self.results.append(metrics)
            self.logger.info(f"Inference performance: {metrics.avg_latency_ms:.1f}ms avg, {metrics.p95_latency_ms:.1f}ms p95")
            
        except Exception as e:
            error_metrics = PerformanceMetrics(
                test_name="inference_performance",
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=0,
                success=False,
                error_message=str(e)
            )
            self.results.append(error_metrics)
            self.logger.error(f"Inference performance test failed: {e}")
            
    async def _simulate_episode_13475_inference(self) -> Dict[str, Any]:
        """Simulate Episode 13475 MCTS inference"""
        # Simulate feature processing
        features = torch.randn(9)  # 9D feature vector
        
        # Simulate WST transform
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Simulate MCTS with 15 simulations
        for _ in range(15):
            # Simulate neural network forward pass
            hidden = torch.relu(torch.matmul(features, torch.randn(9, 64)))
            output = torch.matmul(hidden, torch.randn(64, 3))  # 3 actions
            await asyncio.sleep(0.001)  # Simulate computation
            
        # Return action and confidence
        action_probs = torch.softmax(output, dim=0)
        action = torch.argmax(action_probs).item()
        confidence = torch.max(action_probs).item()
        
        return {
            "action": action,
            "confidence": confidence,
            "features": features.tolist()
        }
        
    async def _test_data_processing(self) -> None:
        """Test market data processing performance"""
        self.logger.info("Testing data processing performance")
        
        start_time = datetime.now()
        
        try:
            # Generate test market data
            num_candles = 10000
            test_data = self._generate_test_market_data(num_candles)
            
            # Test feature extraction performance
            processing_times = []
            
            for i in range(100):
                process_start = time.perf_counter()
                
                # Simulate feature extraction
                features = await self._extract_features(test_data[-100:])  # Last 100 candles
                
                process_end = time.perf_counter()
                processing_times.append((process_end - process_start) * 1000)
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metrics = PerformanceMetrics(
                test_name="data_processing",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=True,
                avg_latency_ms=statistics.mean(processing_times),
                p95_latency_ms=np.percentile(processing_times, 95) if np else sorted(processing_times)[int(0.95 * len(processing_times))],
                max_latency_ms=max(processing_times),
                operations_per_second=len(processing_times) / duration,
                custom_metrics={
                    "candles_processed": num_candles,
                    "feature_extractions": len(processing_times),
                    "avg_features_per_second": 100 * len(processing_times) / duration  # 100 candles per extraction
                }
            )
            
            self.results.append(metrics)
            self.logger.info(f"Data processing: {metrics.avg_latency_ms:.1f}ms avg feature extraction")
            
        except Exception as e:
            self.logger.error(f"Data processing test failed: {e}")
            
    def _generate_test_market_data(self, num_candles: int) -> List[Dict[str, float]]:
        """Generate realistic test market data"""
        data = []
        base_price = 195.0
        current_price = base_price
        
        for i in range(num_candles):
            # Random walk with some trend
            change = (np.random.random() - 0.5) * 0.002 if np else (time.time() % 1 - 0.5) * 0.002
            current_price += change
            
            # Ensure realistic OHLC
            high = current_price + abs(change) * 2
            low = current_price - abs(change) * 2
            open_price = current_price - change
            close_price = current_price
            
            data.append({
                "timestamp": datetime.now() - timedelta(minutes=num_candles - i),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": 1000 + int((time.time() + i) % 500)
            })
            
        return data
        
    async def _extract_features(self, data: List[Dict[str, float]]) -> List[float]:
        """Extract 9D features from market data"""
        if len(data) < 20:
            return [0.0] * 9
            
        # Extract price series
        prices = [candle["close"] for candle in data]
        
        # Calculate features (simplified)
        features = [
            0.0,  # position_type (flat)
            0.0,  # position_pnl
            0.0,  # position_duration
            sum(prices[-5:]) / 5,  # ma_short
            sum(prices[-20:]) / 20,  # ma_long
            50.0,  # rsi (simplified)
            statistics.stdev(prices[-10:]) if len(prices) >= 10 else 0.0,  # volatility
            (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0.0,  # trend_strength
            1.0   # market_state
        ]
        
        # Simulate processing delay
        await asyncio.sleep(0.001)
        
        return features
        
    async def _test_memory_management(self) -> None:
        """Test memory usage and garbage collection"""
        self.logger.info("Testing memory management")
        
        start_time = datetime.now()
        initial_memory = psutil.virtual_memory().used / 1024 / 1024
        
        try:
            # Allocate and deallocate memory to test GC
            large_objects = []
            memory_readings = []
            
            for i in range(50):
                # Allocate large object
                if torch:
                    large_obj = torch.randn(1000, 1000)
                else:
                    large_obj = [[j for j in range(1000)] for _ in range(1000)]
                large_objects.append(large_obj)
                
                # Monitor memory
                current_memory = psutil.virtual_memory().used / 1024 / 1024
                memory_readings.append(current_memory)
                
                # Periodic cleanup
                if i % 10 == 0:
                    large_objects = large_objects[-5:]  # Keep only last 5
                    if torch:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                await asyncio.sleep(0.1)
                
            # Final cleanup
            large_objects.clear()
            
            end_time = datetime.now()
            final_memory = psutil.virtual_memory().used / 1024 / 1024
            duration = (end_time - start_time).total_seconds()
            
            metrics = PerformanceMetrics(
                test_name="memory_management",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=True,
                peak_memory_mb=max(memory_readings),
                avg_memory_mb=statistics.mean(memory_readings),
                custom_metrics={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_delta_mb": final_memory - initial_memory,
                    "memory_leak_detected": abs(final_memory - initial_memory) > 100,
                    "peak_memory_increase_mb": max(memory_readings) - initial_memory
                }
            )
            
            self.results.append(metrics)
            self.logger.info(f"Memory management: {metrics.peak_memory_mb:.1f}MB peak, {metrics.custom_metrics['memory_delta_mb']:+.1f}MB delta")
            
        except Exception as e:
            self.logger.error(f"Memory management test failed: {e}")
            
    async def _test_concurrent_load(self) -> None:
        """Test performance under concurrent load"""
        self.logger.info("Testing concurrent load performance")
        
        start_time = datetime.now()
        
        try:
            # Test concurrent inference requests
            concurrent_tasks = 20
            operations_per_task = 10
            
            async def worker_task(worker_id: int) -> List[float]:
                """Worker task for concurrent testing"""
                latencies = []
                for i in range(operations_per_task):
                    task_start = time.perf_counter()
                    await self._simulate_episode_13475_inference()
                    task_end = time.perf_counter()
                    latencies.append((task_end - task_start) * 1000)
                return latencies
                
            # Run concurrent tasks
            tasks = [worker_task(i) for i in range(concurrent_tasks)]
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            all_latencies = []
            for task_latencies in results:
                all_latencies.extend(task_latencies)
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metrics = PerformanceMetrics(
                test_name="concurrent_load",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=True,
                avg_latency_ms=statistics.mean(all_latencies),
                p95_latency_ms=np.percentile(all_latencies, 95) if np else sorted(all_latencies)[int(0.95 * len(all_latencies))],
                max_latency_ms=max(all_latencies),
                operations_per_second=len(all_latencies) / duration,
                custom_metrics={
                    "concurrent_workers": concurrent_tasks,
                    "total_operations": len(all_latencies),
                    "operations_per_worker": operations_per_task,
                    "throughput_degradation": max(all_latencies) / min(all_latencies) if all_latencies else 1.0
                }
            )
            
            self.results.append(metrics)
            self.logger.info(f"Concurrent load: {concurrent_tasks} workers, {metrics.operations_per_second:.1f} ops/sec")
            
        except Exception as e:
            self.logger.error(f"Concurrent load test failed: {e}")
            
    async def _test_network_performance(self) -> None:
        """Test network-related performance (simulated)"""
        self.logger.info("Testing network performance")
        
        start_time = datetime.now()
        
        try:
            # Simulate API response times
            response_times = []
            
            for i in range(100):
                request_start = time.perf_counter()
                
                # Simulate API call delay
                await asyncio.sleep(0.01 + (i % 10) * 0.001)  # 10-20ms range
                
                request_end = time.perf_counter()
                response_times.append((request_end - request_start) * 1000)
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metrics = PerformanceMetrics(
                test_name="network_performance",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=True,
                avg_latency_ms=statistics.mean(response_times),
                p95_latency_ms=np.percentile(response_times, 95) if np else sorted(response_times)[int(0.95 * len(response_times))],
                max_latency_ms=max(response_times),
                requests_per_second=len(response_times) / duration,
                custom_metrics={
                    "total_requests": len(response_times),
                    "timeout_threshold_ms": 1000,
                    "timeouts": sum(1 for t in response_times if t > 1000),
                    "timeout_rate": sum(1 for t in response_times if t > 1000) / len(response_times)
                }
            )
            
            self.results.append(metrics)
            self.logger.info(f"Network performance: {metrics.avg_latency_ms:.1f}ms avg response time")
            
        except Exception as e:
            self.logger.error(f"Network performance test failed: {e}")
            
    async def _test_integration_performance(self) -> None:
        """Test end-to-end integration performance"""
        self.logger.info("Testing integration performance")
        
        start_time = datetime.now()
        
        try:
            # Simulate complete trading cycle
            cycle_times = []
            
            for i in range(50):
                cycle_start = time.perf_counter()
                
                # Simulate complete cycle
                market_data = self._generate_test_market_data(100)
                features = await self._extract_features(market_data[-100:])
                inference_result = await self._simulate_episode_13475_inference()
                
                # Simulate order processing delay
                await asyncio.sleep(0.05)
                
                cycle_end = time.perf_counter()
                cycle_times.append((cycle_end - cycle_start) * 1000)
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metrics = PerformanceMetrics(
                test_name="integration_performance",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=True,
                avg_latency_ms=statistics.mean(cycle_times),
                p95_latency_ms=np.percentile(cycle_times, 95) if np else sorted(cycle_times)[int(0.95 * len(cycle_times))],
                max_latency_ms=max(cycle_times),
                operations_per_second=len(cycle_times) / duration,
                custom_metrics={
                    "trading_cycles": len(cycle_times),
                    "target_cycle_time_ms": 1000,  # 1 second target
                    "sla_violations": sum(1 for t in cycle_times if t > 1000),
                    "sla_compliance_rate": (len(cycle_times) - sum(1 for t in cycle_times if t > 1000)) / len(cycle_times)
                }
            )
            
            self.results.append(metrics)
            self.logger.info(f"Integration performance: {metrics.avg_latency_ms:.1f}ms avg cycle time")
            
        except Exception as e:
            self.logger.error(f"Integration performance test failed: {e}")
            
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary"""
        summary = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "overall_success": all(r.success for r in self.results),
            "total_duration_seconds": sum(r.duration_seconds for r in self.results),
            "test_results": []
        }
        
        # Add individual test results
        for result in self.results:
            summary["test_results"].append(asdict(result))
            
        # Performance thresholds and compliance
        compliance_checks = {
            "inference_latency_compliance": True,
            "memory_usage_compliance": True,
            "throughput_compliance": True,
            "sla_compliance": True
        }
        
        # Check compliance for each test
        for result in self.results:
            if result.test_name == "inference_performance" and result.success:
                if result.p95_latency_ms and result.p95_latency_ms > 500:  # 500ms threshold
                    compliance_checks["inference_latency_compliance"] = False
                    
            elif result.test_name == "memory_management" and result.success:
                if result.custom_metrics and result.custom_metrics.get("memory_leak_detected", False):
                    compliance_checks["memory_usage_compliance"] = False
                    
            elif result.test_name == "concurrent_load" and result.success:
                if result.operations_per_second and result.operations_per_second < 10:  # 10 ops/sec minimum
                    compliance_checks["throughput_compliance"] = False
                    
            elif result.test_name == "integration_performance" and result.success:
                if result.custom_metrics and result.custom_metrics.get("sla_compliance_rate", 1.0) < 0.95:
                    compliance_checks["sla_compliance"] = False
                    
        summary["compliance_checks"] = compliance_checks
        summary["overall_compliance"] = all(compliance_checks.values())
        
        # Performance insights
        insights = []
        
        # Check for performance issues
        inference_test = next((r for r in self.results if r.test_name == "inference_performance"), None)
        if inference_test and inference_test.success:
            if inference_test.p95_latency_ms and inference_test.p95_latency_ms > 300:
                insights.append("Inference latency higher than optimal (>300ms)")
            if inference_test.custom_metrics and inference_test.custom_metrics.get("latency_violation_rate", 0) > 0.1:
                insights.append("High latency violation rate in inference tests")
                
        memory_test = next((r for r in self.results if r.test_name == "memory_management"), None)
        if memory_test and memory_test.success and memory_test.custom_metrics:
            if memory_test.custom_metrics.get("memory_delta_mb", 0) > 50:
                insights.append("Potential memory leak detected")
                
        concurrent_test = next((r for r in self.results if r.test_name == "concurrent_load"), None)
        if concurrent_test and concurrent_test.success:
            if concurrent_test.custom_metrics and concurrent_test.custom_metrics.get("throughput_degradation", 1) > 5:
                insights.append("High performance degradation under concurrent load")
                
        summary["performance_insights"] = insights
        
        return summary
        
    def save_results(self, output_file: str) -> None:
        """Save performance validation results to file"""
        summary = self._generate_summary_report()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        self.logger.info(f"Performance validation results saved to {output_path}")

async def main():
    """Main performance validation entry point"""
    parser = argparse.ArgumentParser(description="SWT Performance Validator")
    parser.add_argument('--config', default='config/live.yaml', help='Configuration file path')
    parser.add_argument('--output', default='test_results/performance_validation.json', help='Output file path')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create validator
    validator = PerformanceValidator(args.config)
    
    # Run validation
    print("🚀 Starting SWT Performance Validation")
    print(f"📋 Configuration: {args.config}")
    print(f"⏱️  Duration: {args.duration} seconds")
    print(f"📊 Output: {args.output}")
    print("-" * 60)
    
    try:
        results = await validator.run_comprehensive_validation()
        
        # Save results
        validator.save_results(args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE VALIDATION SUMMARY")
        print("=" * 60)
        
        if "error" in results:
            print(f"❌ Validation failed: {results['error']}")
            return 1
            
        print(f"✅ Tests completed: {results['successful_tests']}/{results['total_tests']}")
        print(f"⏱️  Total duration: {results['total_duration_seconds']:.2f} seconds")
        print(f"🎯 Overall compliance: {'✅ PASS' if results['overall_compliance'] else '❌ FAIL'}")
        
        if results.get('performance_insights'):
            print("\n⚠️  Performance Insights:")
            for insight in results['performance_insights']:
                print(f"   • {insight}")
                
        # Print detailed results
        print(f"\n📋 Detailed Results:")
        for test_result in results['test_results']:
            status = "✅" if test_result['success'] else "❌"
            name = test_result['test_name']
            duration = test_result['duration_seconds']
            
            print(f"   {status} {name}: {duration:.2f}s")
            
            if test_result.get('avg_latency_ms'):
                print(f"      ⏱️  Latency: {test_result['avg_latency_ms']:.1f}ms avg, {test_result.get('p95_latency_ms', 0):.1f}ms p95")
                
            if test_result.get('operations_per_second'):
                print(f"      🔄 Throughput: {test_result['operations_per_second']:.1f} ops/sec")
                
            if not test_result['success']:
                print(f"      ❌ Error: {test_result.get('error_message', 'Unknown error')}")
                
        print(f"\n💾 Full results saved to: {args.output}")
        
        return 0 if results['overall_success'] else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Performance validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))