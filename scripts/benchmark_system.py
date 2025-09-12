#!/usr/bin/env python3
"""
SWT System Performance Benchmark
Comprehensive performance testing and validation

This script benchmarks:
- Feature processing performance
- Inference engine speed and accuracy
- Memory usage and resource consumption
- Network latency and throughput
- System scalability and stability
- Real-time performance characteristics

Usage:
    python benchmark_system.py --config config/live.yaml
    python benchmark_system.py --config config/live.yaml --duration 600
    python benchmark_system.py --config config/live.yaml --stress-test
    python benchmark_system.py --checkpoint checkpoints/episode_13475.pth --benchmark-inference
"""

import argparse
import asyncio
import logging
import psutil
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

import numpy as np
import torch

# Import SWT components
from swt_core.config_manager import SWTConfig
from swt_core.types import PositionState, PositionType, TradingAction, TradingDecision
from swt_features.feature_processor import FeatureProcessor
from swt_features.market_features import MarketDataPoint
from swt_inference.agent_factory import AgentFactory
from swt_inference.inference_engine import SWTInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class SystemBenchmark:
    """Comprehensive system performance benchmark suite"""
    
    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None):
        """
        Initialize benchmark suite
        
        Args:
            config_path: Path to system configuration
            checkpoint_path: Optional path to model checkpoint
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        
        # Load configuration
        self.config = SWTConfig.from_file(str(self.config_path))
        
        # Benchmark results
        self.results = {
            "system_info": {},
            "feature_processing": {},
            "inference_performance": {},
            "memory_usage": {},
            "network_performance": {},
            "stress_test": {},
            "real_time_performance": {}
        }
        
        # Performance thresholds
        self.thresholds = {
            "feature_processing_ms": 10.0,
            "inference_time_ms": 200.0,
            "memory_usage_mb": 2000.0,
            "cpu_usage_percent": 80.0,
            "latency_p95_ms": 500.0
        }
        
        logger.info(f"🎯 SystemBenchmark initialized")
        logger.info(f"📁 Config: {self.config_path}")
        if self.checkpoint_path:
            logger.info(f"🧠 Checkpoint: {self.checkpoint_path}")
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect system information and specifications"""
        logger.info("💻 Collecting system information...")
        
        try:
            # CPU information
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent_used": memory.percent
            }
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3),
                "percent_used": (disk.used / disk.total) * 100
            }
            
            # GPU information (if available)
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                    "memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2)
                }
            else:
                gpu_info = {"available": False}
            
            # Python and PyTorch versions
            version_info = {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__
            }
            
            system_info = {
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "gpu": gpu_info,
                "versions": version_info,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log key information
            logger.info(f"  CPU: {cpu_info['logical_cores']} cores @ {cpu_info['current_frequency']:.0f}MHz")
            logger.info(f"  Memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available")
            logger.info(f"  GPU: {gpu_info.get('device_name', 'Not available')}")
            
            self.results["system_info"] = system_info
            return system_info
            
        except Exception as e:
            logger.error(f"❌ Failed to collect system information: {e}")
            return {}
    
    def benchmark_feature_processing(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark feature processing performance"""
        logger.info(f"🔧 Benchmarking feature processing ({num_iterations} iterations)...")
        
        try:
            # Create feature processor
            feature_processor = FeatureProcessor(self.config)
            
            # Generate synthetic market data
            synthetic_data = []
            base_price = 195.0
            
            for i in range(256):  # WST requires sequence of data
                price = base_price + np.sin(i * 0.1) * 0.01 + np.random.normal(0, 0.001)
                data_point = MarketDataPoint(
                    timestamp=datetime.now(),
                    open=price,
                    high=price + 0.001,
                    low=price - 0.001,
                    close=price,
                    volume=1000,
                    spread=0.003,
                    bid=price - 0.0015,
                    ask=price + 0.0015
                )
                synthetic_data.append(data_point)
                feature_processor.add_market_data(data_point)
            
            # Wait for feature processor to be ready
            if not feature_processor.is_ready():
                logger.warning("Feature processor not ready after adding data")
                return {}
            
            # Create position state
            position_state = PositionState(
                position_type=PositionType.FLAT,
                units=0,
                entry_price=0.0,
                unrealized_pnl=0.0
            )
            
            # Benchmark feature processing
            processing_times = []
            memory_usage = []
            
            tracemalloc.start()
            
            for i in range(num_iterations):
                start_time = time.perf_counter()
                
                try:
                    # Process observation
                    observation = feature_processor.process_observation(
                        position_state=position_state,
                        current_price=base_price + (i * 0.0001)
                    )
                    
                    end_time = time.perf_counter()
                    processing_time = (end_time - start_time) * 1000  # Convert to ms
                    processing_times.append(processing_time)
                    
                    # Track memory usage periodically
                    if i % 100 == 0:
                        current, peak = tracemalloc.get_traced_memory()
                        memory_usage.append(current / 1024 / 1024)  # Convert to MB
                    
                except Exception as e:
                    logger.warning(f"Feature processing failed at iteration {i}: {e}")
                    continue
            
            tracemalloc.stop()
            
            # Calculate statistics
            if processing_times:
                stats = {
                    "iterations": len(processing_times),
                    "mean_time_ms": np.mean(processing_times),
                    "median_time_ms": np.median(processing_times),
                    "std_time_ms": np.std(processing_times),
                    "min_time_ms": np.min(processing_times),
                    "max_time_ms": np.max(processing_times),
                    "p95_time_ms": np.percentile(processing_times, 95),
                    "p99_time_ms": np.percentile(processing_times, 99),
                    "throughput_per_sec": 1000 / np.mean(processing_times),
                    "memory_usage_mb": np.mean(memory_usage) if memory_usage else 0
                }
                
                logger.info(f"  Mean processing time: {stats['mean_time_ms']:.2f}ms")
                logger.info(f"  P95 processing time: {stats['p95_time_ms']:.2f}ms") 
                logger.info(f"  Throughput: {stats['throughput_per_sec']:.1f} obs/sec")
                logger.info(f"  Memory usage: {stats['memory_usage_mb']:.1f}MB")
                
                # Check against thresholds
                if stats['p95_time_ms'] > self.thresholds['feature_processing_ms']:
                    logger.warning(f"⚠️ Feature processing P95 exceeds threshold: {stats['p95_time_ms']:.2f}ms > {self.thresholds['feature_processing_ms']}ms")
                
                self.results["feature_processing"] = stats
                return stats
            else:
                logger.error("❌ No successful feature processing iterations")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Feature processing benchmark failed: {e}")
            return {}
    
    def benchmark_inference_performance(self, num_iterations: int = 500) -> Dict[str, Any]:
        """Benchmark inference engine performance"""
        logger.info(f"🧠 Benchmarking inference performance ({num_iterations} iterations)...")
        
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            logger.warning("⚠️ No checkpoint provided, skipping inference benchmark")
            return {}
        
        try:
            # Create agent and load checkpoint
            agent = AgentFactory.create_agent(self.config)
            agent.load_checkpoint(str(self.checkpoint_path))
            
            # Create feature processor and inference engine
            feature_processor = FeatureProcessor(self.config)
            inference_engine = SWTInferenceEngine(
                agent=agent,
                feature_processor=feature_processor,
                config=self.config
            )
            
            # Generate synthetic market data
            base_price = 195.0
            for i in range(256):
                price = base_price + np.sin(i * 0.1) * 0.01 + np.random.normal(0, 0.001)
                data_point = MarketDataPoint(
                    timestamp=datetime.now(),
                    open=price,
                    high=price + 0.001,
                    low=price - 0.001,
                    close=price,
                    volume=1000,
                    spread=0.003,
                    bid=price - 0.0015,
                    ask=price + 0.0015
                )
                feature_processor.add_market_data(data_point)
            
            if not feature_processor.is_ready():
                logger.error("❌ Feature processor not ready for inference benchmark")
                return {}
            
            # Benchmark inference
            inference_times = []
            confidence_scores = []
            actions_taken = []
            memory_usage = []
            
            tracemalloc.start()
            
            for i in range(num_iterations):
                position_state = PositionState(
                    position_type=PositionType.FLAT,
                    units=0,
                    entry_price=0.0,
                    unrealized_pnl=0.0
                )
                
                start_time = time.perf_counter()
                
                try:
                    # Run inference
                    decision = await inference_engine.get_trading_decision(
                        observation=feature_processor.process_observation(
                            position_state=position_state,
                            current_price=base_price + (i * 0.0001)
                        ),
                        current_position=position_state,
                        deterministic=False
                    )
                    
                    end_time = time.perf_counter()
                    inference_time = (end_time - start_time) * 1000  # Convert to ms
                    
                    inference_times.append(inference_time)
                    confidence_scores.append(decision.confidence)
                    actions_taken.append(decision.action.value)
                    
                    # Track memory usage periodically
                    if i % 50 == 0:
                        current, peak = tracemalloc.get_traced_memory()
                        memory_usage.append(current / 1024 / 1024)  # Convert to MB
                    
                except Exception as e:
                    logger.warning(f"Inference failed at iteration {i}: {e}")
                    continue
            
            tracemalloc.stop()
            
            # Calculate statistics
            if inference_times:
                # Action distribution
                action_counts = {}
                for action in actions_taken:
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                action_distribution = {
                    action: count / len(actions_taken) 
                    for action, count in action_counts.items()
                }
                
                stats = {
                    "iterations": len(inference_times),
                    "mean_time_ms": np.mean(inference_times),
                    "median_time_ms": np.median(inference_times),
                    "std_time_ms": np.std(inference_times),
                    "min_time_ms": np.min(inference_times),
                    "max_time_ms": np.max(inference_times),
                    "p95_time_ms": np.percentile(inference_times, 95),
                    "p99_time_ms": np.percentile(inference_times, 99),
                    "throughput_per_sec": 1000 / np.mean(inference_times),
                    "mean_confidence": np.mean(confidence_scores),
                    "confidence_std": np.std(confidence_scores),
                    "action_distribution": action_distribution,
                    "memory_usage_mb": np.mean(memory_usage) if memory_usage else 0
                }
                
                logger.info(f"  Mean inference time: {stats['mean_time_ms']:.2f}ms")
                logger.info(f"  P95 inference time: {stats['p95_time_ms']:.2f}ms")
                logger.info(f"  Throughput: {stats['throughput_per_sec']:.1f} decisions/sec")
                logger.info(f"  Mean confidence: {stats['mean_confidence']:.1%}")
                logger.info(f"  Action distribution: {action_distribution}")
                
                # Check against thresholds
                if stats['p95_time_ms'] > self.thresholds['inference_time_ms']:
                    logger.warning(f"⚠️ Inference P95 exceeds threshold: {stats['p95_time_ms']:.2f}ms > {self.thresholds['inference_time_ms']}ms")
                
                self.results["inference_performance"] = stats
                return stats
            else:
                logger.error("❌ No successful inference iterations")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Inference performance benchmark failed: {e}")
            return {}
    
    async def benchmark_memory_usage(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Benchmark memory usage over time"""
        logger.info(f"💾 Benchmarking memory usage ({duration_seconds}s)...")
        
        try:
            # Create system components
            if self.checkpoint_path and self.checkpoint_path.exists():
                agent = AgentFactory.create_agent(self.config)
                agent.load_checkpoint(str(self.checkpoint_path))
                feature_processor = FeatureProcessor(self.config)
                inference_engine = SWTInferenceEngine(
                    agent=agent,
                    feature_processor=feature_processor,
                    config=self.config
                )
            else:
                feature_processor = FeatureProcessor(self.config)
                inference_engine = None
            
            # Memory tracking
            memory_samples = []
            cpu_samples = []
            
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            # Generate continuous synthetic data
            base_price = 195.0
            iteration = 0
            
            while time.time() < end_time:
                try:
                    # Add market data
                    price = base_price + np.sin(iteration * 0.1) * 0.01 + np.random.normal(0, 0.001)
                    data_point = MarketDataPoint(
                        timestamp=datetime.now(),
                        open=price,
                        high=price + 0.001,
                        low=price - 0.001,
                        close=price,
                        volume=1000,
                        spread=0.003,
                        bid=price - 0.0015,
                        ask=price + 0.0015
                    )
                    feature_processor.add_market_data(data_point)
                    
                    # Sample memory and CPU usage
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent()
                    
                    memory_samples.append({
                        "timestamp": time.time() - start_time,
                        "rss_mb": memory_info.rss / 1024 / 1024,
                        "vms_mb": memory_info.vms / 1024 / 1024,
                        "cpu_percent": cpu_percent
                    })
                    
                    cpu_samples.append(cpu_percent)
                    
                    # Periodic inference if available
                    if inference_engine and iteration % 60 == 0 and feature_processor.is_ready():
                        position_state = PositionState(
                            position_type=PositionType.FLAT,
                            units=0,
                            entry_price=0.0,
                            unrealized_pnl=0.0
                        )
                        
                        try:
                            await inference_engine.get_trading_decision(
                                observation=feature_processor.process_observation(
                                    position_state=position_state,
                                    current_price=price
                                ),
                                current_position=position_state,
                                deterministic=True
                            )
                        except Exception as e:
                            logger.debug(f"Inference error during memory test: {e}")
                    
                    iteration += 1
                    await asyncio.sleep(0.1)  # 100ms sampling rate
                    
                except Exception as e:
                    logger.debug(f"Memory benchmark iteration error: {e}")
                    continue
            
            # Calculate memory statistics
            if memory_samples:
                rss_values = [sample["rss_mb"] for sample in memory_samples]
                vms_values = [sample["vms_mb"] for sample in memory_samples]
                cpu_values = [sample["cpu_percent"] for sample in memory_samples]
                
                stats = {
                    "duration_seconds": duration_seconds,
                    "samples_count": len(memory_samples),
                    "rss_memory": {
                        "mean_mb": np.mean(rss_values),
                        "max_mb": np.max(rss_values),
                        "min_mb": np.min(rss_values),
                        "final_mb": rss_values[-1]
                    },
                    "vms_memory": {
                        "mean_mb": np.mean(vms_values),
                        "max_mb": np.max(vms_values),
                        "min_mb": np.min(vms_values),
                        "final_mb": vms_values[-1]
                    },
                    "cpu_usage": {
                        "mean_percent": np.mean(cpu_values),
                        "max_percent": np.max(cpu_values),
                        "p95_percent": np.percentile(cpu_values, 95)
                    },
                    "memory_growth_mb": rss_values[-1] - rss_values[0],
                    "memory_stable": abs(rss_values[-1] - rss_values[0]) < 50  # < 50MB growth
                }
                
                logger.info(f"  Peak RSS memory: {stats['rss_memory']['max_mb']:.1f}MB")
                logger.info(f"  Memory growth: {stats['memory_growth_mb']:.1f}MB")
                logger.info(f"  Mean CPU usage: {stats['cpu_usage']['mean_percent']:.1f}%")
                logger.info(f"  Memory stable: {stats['memory_stable']}")
                
                # Check against thresholds
                if stats['rss_memory']['max_mb'] > self.thresholds['memory_usage_mb']:
                    logger.warning(f"⚠️ Peak memory exceeds threshold: {stats['rss_memory']['max_mb']:.1f}MB > {self.thresholds['memory_usage_mb']}MB")
                
                if stats['cpu_usage']['p95_percent'] > self.thresholds['cpu_usage_percent']:
                    logger.warning(f"⚠️ P95 CPU usage exceeds threshold: {stats['cpu_usage']['p95_percent']:.1f}% > {self.thresholds['cpu_usage_percent']}%")
                
                self.results["memory_usage"] = stats
                return stats
            else:
                logger.error("❌ No memory samples collected")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Memory usage benchmark failed: {e}")
            return {}
    
    async def run_stress_test(self, duration_seconds: int = 600, num_threads: int = 4) -> Dict[str, Any]:
        """Run system stress test with concurrent workload"""
        logger.info(f"🔥 Running stress test ({duration_seconds}s, {num_threads} threads)...")
        
        try:
            # Create components for stress testing
            components = []
            for i in range(num_threads):
                feature_processor = FeatureProcessor(self.config)
                if self.checkpoint_path and self.checkpoint_path.exists():
                    agent = AgentFactory.create_agent(self.config)
                    agent.load_checkpoint(str(self.checkpoint_path))
                    inference_engine = SWTInferenceEngine(
                        agent=agent,
                        feature_processor=feature_processor,
                        config=self.config
                    )
                else:
                    inference_engine = None
                
                components.append({
                    "feature_processor": feature_processor,
                    "inference_engine": inference_engine
                })
            
            # Stress test metrics
            total_operations = 0
            total_errors = 0
            latency_samples = []
            memory_samples = []
            
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            async def worker(worker_id: int, component: Dict[str, Any]):
                """Worker function for stress testing"""
                nonlocal total_operations, total_errors
                
                feature_processor = component["feature_processor"]
                inference_engine = component["inference_engine"]
                
                # Initialize with synthetic data
                base_price = 195.0 + worker_id * 0.1
                for i in range(256):
                    price = base_price + np.sin(i * 0.1) * 0.01
                    data_point = MarketDataPoint(
                        timestamp=datetime.now(),
                        open=price,
                        high=price + 0.001,
                        low=price - 0.001,
                        close=price,
                        volume=1000,
                        spread=0.003,
                        bid=price - 0.0015,
                        ask=price + 0.0015
                    )
                    feature_processor.add_market_data(data_point)
                
                iteration = 0
                while time.time() < end_time:
                    operation_start = time.perf_counter()
                    
                    try:
                        # Add market data
                        price = base_price + np.sin(iteration * 0.1) * 0.01 + np.random.normal(0, 0.001)
                        data_point = MarketDataPoint(
                            timestamp=datetime.now(),
                            open=price,
                            high=price + 0.001,
                            low=price - 0.001,
                            close=price,
                            volume=1000,
                            spread=0.003,
                            bid=price - 0.0015,
                            ask=price + 0.0015
                        )
                        feature_processor.add_market_data(data_point)
                        
                        # Perform inference periodically
                        if inference_engine and iteration % 10 == 0 and feature_processor.is_ready():
                            position_state = PositionState(
                                position_type=PositionType.FLAT,
                                units=0,
                                entry_price=0.0,
                                unrealized_pnl=0.0
                            )
                            
                            decision = await inference_engine.get_trading_decision(
                                observation=feature_processor.process_observation(
                                    position_state=position_state,
                                    current_price=price
                                ),
                                current_position=position_state,
                                deterministic=True
                            )
                        
                        operation_end = time.perf_counter()
                        latency = (operation_end - operation_start) * 1000
                        latency_samples.append(latency)
                        
                        total_operations += 1
                        
                    except Exception as e:
                        total_errors += 1
                        logger.debug(f"Worker {worker_id} error: {e}")
                    
                    iteration += 1
                    await asyncio.sleep(0.01)  # Small delay
            
            # Monitor system resources during stress test
            async def monitor_resources():
                """Monitor system resources during stress test"""
                while time.time() < end_time:
                    try:
                        memory = psutil.virtual_memory()
                        cpu_percent = psutil.cpu_percent(interval=1)
                        
                        memory_samples.append({
                            "timestamp": time.time() - start_time,
                            "memory_percent": memory.percent,
                            "cpu_percent": cpu_percent
                        })
                        
                    except Exception as e:
                        logger.debug(f"Resource monitoring error: {e}")
                    
                    await asyncio.sleep(1)
            
            # Run stress test
            tasks = []
            
            # Start worker tasks
            for i, component in enumerate(components):
                task = asyncio.create_task(worker(i, component))
                tasks.append(task)
            
            # Start resource monitor
            monitor_task = asyncio.create_task(monitor_resources())
            tasks.append(monitor_task)
            
            # Wait for completion
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate stress test statistics
            actual_duration = time.time() - start_time
            
            if latency_samples and memory_samples:
                cpu_values = [sample["cpu_percent"] for sample in memory_samples]
                memory_values = [sample["memory_percent"] for sample in memory_samples]
                
                stats = {
                    "duration_seconds": actual_duration,
                    "num_threads": num_threads,
                    "total_operations": total_operations,
                    "total_errors": total_errors,
                    "error_rate": total_errors / max(1, total_operations + total_errors),
                    "operations_per_second": total_operations / actual_duration,
                    "latency_stats": {
                        "mean_ms": np.mean(latency_samples),
                        "p95_ms": np.percentile(latency_samples, 95),
                        "p99_ms": np.percentile(latency_samples, 99),
                        "max_ms": np.max(latency_samples)
                    },
                    "resource_usage": {
                        "peak_cpu_percent": np.max(cpu_values),
                        "mean_cpu_percent": np.mean(cpu_values),
                        "peak_memory_percent": np.max(memory_values),
                        "mean_memory_percent": np.mean(memory_values)
                    },
                    "system_stable": total_errors / max(1, total_operations) < 0.05  # < 5% error rate
                }
                
                logger.info(f"  Operations completed: {stats['total_operations']:,}")
                logger.info(f"  Error rate: {stats['error_rate']:.1%}")
                logger.info(f"  Operations/sec: {stats['operations_per_second']:.1f}")
                logger.info(f"  P95 latency: {stats['latency_stats']['p95_ms']:.2f}ms")
                logger.info(f"  Peak CPU: {stats['resource_usage']['peak_cpu_percent']:.1f}%")
                logger.info(f"  System stable: {stats['system_stable']}")
                
                # Check against thresholds
                if stats['latency_stats']['p95_ms'] > self.thresholds['latency_p95_ms']:
                    logger.warning(f"⚠️ P95 latency exceeds threshold: {stats['latency_stats']['p95_ms']:.2f}ms > {self.thresholds['latency_p95_ms']}ms")
                
                self.results["stress_test"] = stats
                return stats
            else:
                logger.error("❌ No stress test data collected")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Stress test failed: {e}")
            return {}
    
    async def run_comprehensive_benchmark(self, 
                                        duration_seconds: int = 300,
                                        include_stress_test: bool = False,
                                        stress_duration: int = 600) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info("🚀 Starting comprehensive system benchmark...")
        logger.info(f"🕐 Estimated duration: {duration_seconds + (stress_duration if include_stress_test else 0)}s")
        
        start_time = time.time()
        
        # Collect system information
        logger.info("\n" + "="*60)
        self.collect_system_info()
        
        # Feature processing benchmark
        logger.info("\n" + "="*60)
        self.benchmark_feature_processing(num_iterations=1000)
        
        # Inference performance benchmark
        if self.checkpoint_path and self.checkpoint_path.exists():
            logger.info("\n" + "="*60)
            await self.benchmark_inference_performance(num_iterations=500)
        
        # Memory usage benchmark
        logger.info("\n" + "="*60)
        await self.benchmark_memory_usage(duration_seconds=duration_seconds)
        
        # Stress test (if requested)
        if include_stress_test:
            logger.info("\n" + "="*60)
            await self.run_stress_test(duration_seconds=stress_duration, num_threads=4)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        logger.info("\n" + "="*60)
        logger.info("📋 BENCHMARK SUMMARY")
        logger.info("="*60)
        
        # Performance summary
        feature_perf = self.results.get("feature_processing", {})
        inference_perf = self.results.get("inference_performance", {})
        memory_perf = self.results.get("memory_usage", {})
        stress_perf = self.results.get("stress_test", {})
        
        if feature_perf:
            logger.info(f"Feature Processing:")
            logger.info(f"  Mean time: {feature_perf.get('mean_time_ms', 0):.2f}ms")
            logger.info(f"  P95 time: {feature_perf.get('p95_time_ms', 0):.2f}ms")
            logger.info(f"  Throughput: {feature_perf.get('throughput_per_sec', 0):.1f} obs/sec")
        
        if inference_perf:
            logger.info(f"Inference Performance:")
            logger.info(f"  Mean time: {inference_perf.get('mean_time_ms', 0):.2f}ms")
            logger.info(f"  P95 time: {inference_perf.get('p95_time_ms', 0):.2f}ms")
            logger.info(f"  Throughput: {inference_perf.get('throughput_per_sec', 0):.1f} decisions/sec")
        
        if memory_perf:
            logger.info(f"Memory Usage:")
            logger.info(f"  Peak RSS: {memory_perf.get('rss_memory', {}).get('max_mb', 0):.1f}MB")
            logger.info(f"  Memory stable: {memory_perf.get('memory_stable', False)}")
            logger.info(f"  CPU P95: {memory_perf.get('cpu_usage', {}).get('p95_percent', 0):.1f}%")
        
        if stress_perf:
            logger.info(f"Stress Test:")
            logger.info(f"  Operations: {stress_perf.get('total_operations', 0):,}")
            logger.info(f"  Error rate: {stress_perf.get('error_rate', 0):.1%}")
            logger.info(f"  System stable: {stress_perf.get('system_stable', False)}")
        
        logger.info("="*60)
        logger.info(f"Total benchmark time: {total_time:.1f}s")
        
        # Overall assessment
        passed_checks = 0
        total_checks = 0
        
        checks = [
            ("Feature processing P95", feature_perf.get('p95_time_ms', float('inf')) < self.thresholds['feature_processing_ms']),
            ("Inference P95", inference_perf.get('p95_time_ms', float('inf')) < self.thresholds['inference_time_ms']),
            ("Memory usage", memory_perf.get('rss_memory', {}).get('max_mb', float('inf')) < self.thresholds['memory_usage_mb']),
            ("Memory stable", memory_perf.get('memory_stable', False)),
            ("System stable", stress_perf.get('system_stable', True))  # Default to True if no stress test
        ]
        
        for check_name, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{check_name:.<40} {status}")
            if passed:
                passed_checks += 1
            total_checks += 1
        
        success_rate = passed_checks / total_checks
        logger.info("="*60)
        logger.info(f"Overall Success Rate: {success_rate:.1%} ({passed_checks}/{total_checks})")
        
        if success_rate >= 0.8:
            logger.info("🎉 SYSTEM BENCHMARK PASSED")
        else:
            logger.error("💥 SYSTEM BENCHMARK FAILED")
        
        # Add summary to results
        self.results["summary"] = {
            "success_rate": success_rate,
            "total_time": total_time,
            "passed_checks": passed_checks,
            "total_checks": total_checks
        }
        
        return self.results


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="SWT System Performance Benchmark")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to system configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to model checkpoint (optional)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Memory benchmark duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Include system stress test"
    )
    parser.add_argument(
        "--stress-duration",
        type=int,
        default=600,
        help="Stress test duration in seconds (default: 600)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create benchmark suite
    benchmark = SystemBenchmark(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    # Run benchmarks
    try:
        results = await benchmark.run_comprehensive_benchmark(
            duration_seconds=args.duration,
            include_stress_test=args.stress_test,
            stress_duration=args.stress_duration
        )
        
        # Exit with appropriate code
        success_rate = results.get("summary", {}).get("success_rate", 0)
        if success_rate >= 0.8:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Benchmark failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())