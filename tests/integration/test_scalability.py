"""
Scalability Testing Suite for GridAttention Trading System
Tests horizontal/vertical scaling, resource utilization, and performance at scale
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import time
import psutil
import multiprocessing
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, AsyncMock, patch
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# GridAttention imports - aligned with system structure
from src.grid_attention_layer import GridAttentionLayer
from src.attention_learning_layer import AttentionLearningLayer
from src.market_regime_detector import MarketRegimeDetector
from src.grid_strategy_selector import GridStrategySelector
from src.risk_management_system import RiskManagementSystem
from src.execution_engine import ExecutionEngine
from src.performance_monitor import PerformanceMonitor
from src.feedback_loop import FeedbackLoop
from src.overfitting_detector import OverfittingDetector

# Test utilities
from tests.utils.test_helpers import (
    async_test,
    create_test_config,
    ScalabilityProfiler
)
from tests.mocks.mock_exchange import MockExchange
from tests.fixtures.market_data import generate_market_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScalabilityMetrics:
    """Container for scalability measurements"""
    scale_factor: int
    throughput: float
    latency_mean: float
    latency_p95: float
    latency_p99: float
    cpu_usage: float
    memory_usage_mb: float
    efficiency: float  # Relative to single unit
    resource_efficiency: float  # Throughput per resource unit


class TestHorizontalScaling:
    """Test horizontal scaling capabilities"""
    
    @pytest.fixture
    def scalability_profiler(self):
        """Create scalability profiler"""
        return ScalabilityProfiler()
    
    @async_test
    async def test_multi_instance_scaling(self, scalability_profiler):
        """Test scaling with multiple GridAttention instances"""
        instance_counts = [1, 2, 4, 8, 16]
        results = []
        
        for num_instances in instance_counts:
            logger.info(f"\nTesting with {num_instances} instances...")
            
            # Create instances
            instances = []
            for i in range(num_instances):
                config = create_test_config()
                config['instance_id'] = i
                config['total_instances'] = num_instances
                instances.append(GridAttentionLayer(config))
            
            # Create load balancer
            load_balancer = LoadBalancer(instances)
            
            # Run scalability test
            metrics = await self._run_scaling_test(
                load_balancer,
                num_instances,
                scalability_profiler
            )
            
            results.append(metrics)
            
            # Cleanup
            for instance in instances:
                await instance.shutdown()
        
        # Analyze scaling efficiency
        self._analyze_scaling_results(results, "Multi-Instance Scaling")
        
        # Verify linear scaling up to certain point
        for i in range(1, min(3, len(results))):
            efficiency = results[i].efficiency
            assert efficiency >= 0.7, f"Poor scaling efficiency at {results[i].scale_factor}x: {efficiency:.2f}"
        
        return results
    
    @async_test
    async def test_worker_pool_scaling(self, scalability_profiler):
        """Test scaling with worker pools"""
        worker_counts = [1, 2, 4, 8, 16, 32]
        results = []
        
        for num_workers in worker_counts:
            logger.info(f"\nTesting with {num_workers} workers...")
            
            # Create system with worker pool
            config = create_test_config()
            config['worker_pool_size'] = num_workers
            system = GridAttentionLayer(config)
            
            # Run test
            metrics = await self._run_worker_scaling_test(
                system,
                num_workers,
                scalability_profiler
            )
            
            results.append(metrics)
            
            # Cleanup
            await system.shutdown()
        
        self._analyze_scaling_results(results, "Worker Pool Scaling")
        
        # Verify efficient worker utilization
        for result in results:
            assert result.resource_efficiency > 0.5, f"Poor worker efficiency: {result.resource_efficiency:.2f}"
        
        return results
    
    @async_test
    async def test_distributed_processing(self):
        """Test distributed processing across multiple nodes"""
        # Simulate distributed nodes
        node_configs = [
            {'node_id': 0, 'role': 'primary', 'host': 'localhost', 'port': 8000},
            {'node_id': 1, 'role': 'secondary', 'host': 'localhost', 'port': 8001},
            {'node_id': 2, 'role': 'secondary', 'host': 'localhost', 'port': 8002},
            {'node_id': 3, 'role': 'secondary', 'host': 'localhost', 'port': 8003}
        ]
        
        # Create distributed system
        distributed_system = DistributedGridAttention(node_configs)
        await distributed_system.initialize()
        
        # Test scaling with different data partitioning strategies
        strategies = ['round_robin', 'hash_based', 'load_balanced']
        results = {}
        
        for strategy in strategies:
            distributed_system.set_partitioning_strategy(strategy)
            
            # Run distributed test
            start_time = time.time()
            operations = 0
            errors = 0
            
            # Generate workload
            for i in range(10000):
                symbol = f"SYMBOL_{i % 100}"
                data = {
                    'symbol': symbol,
                    'price': 50000 + np.random.uniform(-100, 100),
                    'volume': np.random.uniform(10, 1000)
                }
                
                try:
                    node = distributed_system.get_node_for_data(data)
                    await node.process(data)
                    operations += 1
                except Exception as e:
                    errors += 1
                    logger.error(f"Distributed processing error: {e}")
            
            duration = time.time() - start_time
            throughput = operations / duration
            
            results[strategy] = {
                'throughput': throughput,
                'errors': errors,
                'efficiency': operations / (operations + errors)
            }
            
            logger.info(f"\nDistributed {strategy}:")
            logger.info(f"  Throughput: {throughput:,.2f} ops/sec")
            logger.info(f"  Errors: {errors}")
            logger.info(f"  Efficiency: {results[strategy]['efficiency']:.2%}")
        
        # Verify distributed processing works
        for strategy, result in results.items():
            assert result['efficiency'] > 0.95, f"Poor distributed efficiency for {strategy}"
            assert result['throughput'] > 1000, f"Low distributed throughput for {strategy}"
        
        await distributed_system.shutdown()
        return results
    
    async def _run_scaling_test(self, load_balancer, scale_factor, profiler):
        """Run scaling test with load balancer"""
        duration = 10
        operations = []
        latencies = []
        
        profiler.start()
        start_time = time.time()
        
        # Generate concurrent load
        async def generate_load():
            while time.time() - start_time < duration:
                data = {
                    'price': 50000 + np.random.uniform(-100, 100),
                    'volume': np.random.uniform(10, 1000),
                    'timestamp': time.time()
                }
                
                op_start = time.perf_counter()
                await load_balancer.process(data)
                op_end = time.perf_counter()
                
                operations.append(1)
                latencies.append((op_end - op_start) * 1000)
        
        # Run with multiple concurrent tasks
        tasks = [generate_load() for _ in range(scale_factor * 2)]
        await asyncio.gather(*tasks)
        
        # Calculate metrics
        profiler.stop()
        profile_data = profiler.get_metrics()
        
        throughput = len(operations) / duration
        latency_sorted = sorted(latencies)
        
        return ScalabilityMetrics(
            scale_factor=scale_factor,
            throughput=throughput,
            latency_mean=np.mean(latencies),
            latency_p95=latency_sorted[int(len(latencies) * 0.95)],
            latency_p99=latency_sorted[int(len(latencies) * 0.99)],
            cpu_usage=profile_data['cpu_usage'],
            memory_usage_mb=profile_data['memory_usage_mb'],
            efficiency=throughput / (scale_factor * (throughput if scale_factor == 1 else 0)),
            resource_efficiency=throughput / (scale_factor * profile_data['cpu_usage'] / 100)
        )
    
    async def _run_worker_scaling_test(self, system, num_workers, profiler):
        """Run worker pool scaling test"""
        duration = 10
        operations = []
        latencies = []
        
        profiler.start()
        start_time = time.time()
        
        # Generate workload
        async def worker_task():
            while time.time() - start_time < duration:
                op_start = time.perf_counter()
                
                await system.process_market_update({
                    'price': 50000 + np.random.uniform(-100, 100),
                    'volume': np.random.uniform(10, 1000)
                })
                
                op_end = time.perf_counter()
                operations.append(1)
                latencies.append((op_end - op_start) * 1000)
        
        # Run concurrent tasks
        tasks = [worker_task() for _ in range(min(num_workers * 2, 64))]
        await asyncio.gather(*tasks)
        
        profiler.stop()
        profile_data = profiler.get_metrics()
        
        throughput = len(operations) / duration
        latency_sorted = sorted(latencies)
        
        return ScalabilityMetrics(
            scale_factor=num_workers,
            throughput=throughput,
            latency_mean=np.mean(latencies),
            latency_p95=latency_sorted[int(len(latencies) * 0.95)],
            latency_p99=latency_sorted[int(len(latencies) * 0.99)],
            cpu_usage=profile_data['cpu_usage'],
            memory_usage_mb=profile_data['memory_usage_mb'],
            efficiency=1.0,  # Will be calculated relative to baseline
            resource_efficiency=throughput / num_workers
        )
    
    def _analyze_scaling_results(self, results: List[ScalabilityMetrics], test_name: str):
        """Analyze and log scaling results"""
        logger.info(f"\n{test_name} Results:")
        logger.info(f"{'Scale':>6} {'Throughput':>12} {'Latency(ms)':>12} {'CPU%':>6} {'Mem(MB)':>8} {'Efficiency':>10}")
        logger.info("-" * 70)
        
        baseline_throughput = results[0].throughput if results else 1
        
        for r in results:
            # Calculate efficiency relative to baseline
            r.efficiency = (r.throughput / baseline_throughput) / r.scale_factor
            
            logger.info(
                f"{r.scale_factor:6d} {r.throughput:12.2f} {r.latency_mean:12.2f} "
                f"{r.cpu_usage:6.1f} {r.memory_usage_mb:8.1f} {r.efficiency:10.2%}"
            )


class TestVerticalScaling:
    """Test vertical scaling capabilities"""
    
    @async_test
    async def test_cpu_scaling(self):
        """Test performance with different CPU allocations"""
        cpu_configs = [
            {'cores': 1, 'threads': 2},
            {'cores': 2, 'threads': 4},
            {'cores': 4, 'threads': 8},
            {'cores': 8, 'threads': 16}
        ]
        
        results = []
        
        for cpu_config in cpu_configs:
            # Create system with CPU limits
            config = create_test_config()
            config['cpu_cores'] = cpu_config['cores']
            config['max_threads'] = cpu_config['threads']
            
            system = GridAttentionLayer(config)
            
            # Run performance test
            profiler = ScalabilityProfiler()
            profiler.start()
            
            operations = 0
            start_time = time.time()
            duration = 10
            
            while time.time() - start_time < duration:
                await system.process_market_update({
                    'price': 50000 + np.random.uniform(-100, 100),
                    'volume': np.random.uniform(10, 1000)
                })
                operations += 1
            
            profiler.stop()
            metrics = profiler.get_metrics()
            
            throughput = operations / duration
            
            results.append({
                'cores': cpu_config['cores'],
                'threads': cpu_config['threads'],
                'throughput': throughput,
                'cpu_efficiency': throughput / cpu_config['cores'],
                'thread_efficiency': throughput / cpu_config['threads']
            })
            
            await system.shutdown()
            
            logger.info(f"\nCPU Scaling - {cpu_config['cores']} cores:")
            logger.info(f"  Throughput: {throughput:,.2f} ops/sec")
            logger.info(f"  Per-core efficiency: {results[-1]['cpu_efficiency']:,.2f} ops/sec/core")
        
        # Verify CPU scaling
        for i in range(1, len(results)):
            # Throughput should increase with more cores
            assert results[i]['throughput'] > results[i-1]['throughput']
        
        return results
    
    @async_test
    async def test_memory_scaling(self):
        """Test performance with different memory allocations"""
        memory_configs = [
            {'memory_mb': 512, 'cache_size': 1000},
            {'memory_mb': 1024, 'cache_size': 5000},
            {'memory_mb': 2048, 'cache_size': 10000},
            {'memory_mb': 4096, 'cache_size': 20000}
        ]
        
        results = []
        
        for mem_config in memory_configs:
            # Create system with memory configuration
            config = create_test_config()
            config['max_memory_mb'] = mem_config['memory_mb']
            config['cache_size'] = mem_config['cache_size']
            
            system = GridAttentionLayer(config)
            
            # Generate dataset that benefits from caching
            test_data = []
            for i in range(1000):
                test_data.append({
                    'symbol': f'SYMBOL_{i % 100}',  # 100 unique symbols
                    'price': 50000 + (i % 1000),
                    'volume': 100 + (i % 100)
                })
            
            # Run test with repeated data access
            start_time = time.time()
            operations = 0
            cache_hits = 0
            
            for _ in range(10):  # Multiple passes through data
                for data in test_data:
                    result = await system.process_market_update(data)
                    operations += 1
                    if result.get('cache_hit'):
                        cache_hits += 1
            
            duration = time.time() - start_time
            throughput = operations / duration
            hit_rate = cache_hits / operations if operations > 0 else 0
            
            results.append({
                'memory_mb': mem_config['memory_mb'],
                'cache_size': mem_config['cache_size'],
                'throughput': throughput,
                'cache_hit_rate': hit_rate,
                'ops_per_mb': throughput / mem_config['memory_mb']
            })
            
            await system.shutdown()
            
            logger.info(f"\nMemory Scaling - {mem_config['memory_mb']}MB:")
            logger.info(f"  Throughput: {throughput:,.2f} ops/sec")
            logger.info(f"  Cache hit rate: {hit_rate:.2%}")
            logger.info(f"  Ops per MB: {results[-1]['ops_per_mb']:.2f}")
        
        # Verify memory scaling benefits
        for i in range(1, len(results)):
            # Cache hit rate should improve with more memory
            assert results[i]['cache_hit_rate'] >= results[i-1]['cache_hit_rate']
        
        return results
    
    @async_test
    async def test_resource_optimization(self):
        """Test optimal resource allocation"""
        # Test different resource combinations
        configurations = [
            {'cores': 2, 'memory_mb': 1024, 'workers': 4},
            {'cores': 4, 'memory_mb': 1024, 'workers': 4},
            {'cores': 2, 'memory_mb': 2048, 'workers': 4},
            {'cores': 4, 'memory_mb': 2048, 'workers': 8},
            {'cores': 8, 'memory_mb': 4096, 'workers': 16}
        ]
        
        results = []
        
        for config_params in configurations:
            config = create_test_config()
            config.update(config_params)
            
            system = GridAttentionLayer(config)
            
            # Run comprehensive workload
            metrics = await self._run_resource_test(system, config_params)
            results.append(metrics)
            
            await system.shutdown()
        
        # Find optimal configuration
        optimal_idx = max(range(len(results)), 
                         key=lambda i: results[i]['efficiency_score'])
        optimal_config = configurations[optimal_idx]
        
        logger.info(f"\nOptimal Configuration:")
        logger.info(f"  Cores: {optimal_config['cores']}")
        logger.info(f"  Memory: {optimal_config['memory_mb']}MB")
        logger.info(f"  Workers: {optimal_config['workers']}")
        logger.info(f"  Efficiency Score: {results[optimal_idx]['efficiency_score']:.2f}")
        
        return results, optimal_config
    
    async def _run_resource_test(self, system, config):
        """Run resource optimization test"""
        duration = 10
        operations = 0
        latencies = []
        resource_samples = []
        
        start_time = time.time()
        
        # Monitor resources
        async def monitor_resources():
            while time.time() - start_time < duration:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                resource_samples.append({
                    'cpu': cpu_percent,
                    'memory': memory_mb
                })
                await asyncio.sleep(0.5)
        
        # Run workload
        async def run_workload():
            nonlocal operations
            while time.time() - start_time < duration:
                op_start = time.perf_counter()
                
                await system.process_market_update({
                    'price': 50000 + np.random.uniform(-100, 100),
                    'volume': np.random.uniform(10, 1000)
                })
                
                op_end = time.perf_counter()
                operations += 1
                latencies.append((op_end - op_start) * 1000)
        
        # Run monitoring and workload concurrently
        await asyncio.gather(
            monitor_resources(),
            *[run_workload() for _ in range(config['workers'])]
        )
        
        # Calculate metrics
        throughput = operations / duration
        avg_latency = np.mean(latencies)
        avg_cpu = np.mean([s['cpu'] for s in resource_samples])
        avg_memory = np.mean([s['memory'] for s in resource_samples])
        
        # Calculate efficiency score (throughput per resource unit)
        resource_cost = (config['cores'] * avg_cpu / 100) + (avg_memory / 1000)
        efficiency_score = throughput / resource_cost if resource_cost > 0 else 0
        
        return {
            'config': config,
            'throughput': throughput,
            'avg_latency': avg_latency,
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'efficiency_score': efficiency_score
        }


class TestDataScaling:
    """Test scaling with increasing data volumes"""
    
    @async_test
    async def test_symbol_scaling(self):
        """Test scaling with increasing number of symbols"""
        symbol_counts = [10, 50, 100, 500, 1000]
        results = []
        
        for num_symbols in symbol_counts:
            # Create system
            system = GridAttentionLayer(create_test_config())
            
            # Generate symbols
            symbols = [f'SYMBOL_{i}' for i in range(num_symbols)]
            
            # Run test
            start_time = time.time()
            operations = 0
            
            # Process updates for all symbols
            for _ in range(1000):
                symbol = symbols[operations % num_symbols]
                await system.process_market_update({
                    'symbol': symbol,
                    'price': 50000 + np.random.uniform(-100, 100),
                    'volume': np.random.uniform(10, 1000)
                })
                operations += 1
            
            duration = time.time() - start_time
            throughput = operations / duration
            
            # Measure memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            results.append({
                'symbols': num_symbols,
                'throughput': throughput,
                'memory_mb': memory_mb,
                'ops_per_symbol': throughput / num_symbols
            })
            
            await system.shutdown()
            
            logger.info(f"\nSymbol Scaling - {num_symbols} symbols:")
            logger.info(f"  Throughput: {throughput:,.2f} ops/sec")
            logger.info(f"  Memory: {memory_mb:.1f} MB")
            logger.info(f"  Ops per symbol: {results[-1]['ops_per_symbol']:.2f}")
        
        # Verify reasonable scaling
        for i in range(1, len(results)):
            # Memory should scale sub-linearly with symbols
            memory_ratio = results[i]['memory_mb'] / results[0]['memory_mb']
            symbol_ratio = results[i]['symbols'] / results[0]['symbols']
            assert memory_ratio < symbol_ratio, "Memory scaling is not efficient"
        
        return results
    
    @async_test
    async def test_history_scaling(self):
        """Test scaling with increasing historical data"""
        history_sizes = [1000, 10000, 50000, 100000, 500000]
        results = []
        
        for history_size in history_sizes:
            # Create system with history configuration
            config = create_test_config()
            config['max_history_size'] = history_size
            system = GridAttentionLayer(config)
            
            # Generate historical data
            logger.info(f"\nGenerating {history_size} historical records...")
            historical_data = generate_market_data(
                'BTC/USDT',
                periods=history_size,
                interval='1m'
            )
            
            # Load historical data
            load_start = time.time()
            await system.load_historical_data(historical_data)
            load_time = time.time() - load_start
            
            # Test operations with historical context
            operations = 0
            op_start = time.time()
            
            for _ in range(100):
                result = await system.analyze_with_history({
                    'price': 50000,
                    'volume': 100
                })
                operations += 1
            
            op_duration = time.time() - op_start
            analysis_throughput = operations / op_duration
            
            # Measure memory
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            results.append({
                'history_size': history_size,
                'load_time': load_time,
                'analysis_throughput': analysis_throughput,
                'memory_mb': memory_mb,
                'memory_per_record': memory_mb / history_size * 1000  # KB per record
            })
            
            await system.shutdown()
            
            logger.info(f"History Scaling - {history_size} records:")
            logger.info(f"  Load time: {load_time:.2f}s")
            logger.info(f"  Analysis throughput: {analysis_throughput:.2f} ops/sec")
            logger.info(f"  Memory: {memory_mb:.1f} MB")
            logger.info(f"  Memory per record: {results[-1]['memory_per_record']:.2f} KB")
        
        return results
    
    @async_test
    async def test_concurrent_stream_scaling(self):
        """Test scaling with multiple concurrent data streams"""
        stream_counts = [1, 5, 10, 20, 50]
        results = []
        
        for num_streams in stream_counts:
            # Create system
            system = GridAttentionLayer(create_test_config())
            
            # Create data streams
            streams = []
            for i in range(num_streams):
                stream = DataStream(
                    stream_id=i,
                    symbol=f'STREAM_{i}',
                    rate=100  # 100 updates per second per stream
                )
                streams.append(stream)
            
            # Run concurrent stream processing
            duration = 10
            total_processed = 0
            errors = 0
            
            async def process_stream(stream):
                nonlocal total_processed, errors
                processed = 0
                
                async for data in stream.generate(duration):
                    try:
                        await system.process_market_update(data)
                        processed += 1
                    except Exception as e:
                        errors += 1
                
                total_processed += processed
            
            # Process all streams concurrently
            start_time = time.time()
            await asyncio.gather(*[process_stream(s) for s in streams])
            actual_duration = time.time() - start_time
            
            throughput = total_processed / actual_duration
            error_rate = errors / (total_processed + errors) if total_processed + errors > 0 else 0
            
            results.append({
                'streams': num_streams,
                'total_processed': total_processed,
                'throughput': throughput,
                'error_rate': error_rate,
                'per_stream_throughput': throughput / num_streams
            })
            
            await system.shutdown()
            
            logger.info(f"\nStream Scaling - {num_streams} streams:")
            logger.info(f"  Total processed: {total_processed:,}")
            logger.info(f"  Throughput: {throughput:,.2f} ops/sec")
            logger.info(f"  Error rate: {error_rate:.2%}")
            logger.info(f"  Per-stream: {results[-1]['per_stream_throughput']:.2f} ops/sec")
        
        # Verify stream scaling
        for result in results:
            assert result['error_rate'] < 0.01, f"High error rate: {result['error_rate']:.2%}"
        
        return results


class TestAutoScaling:
    """Test auto-scaling capabilities"""
    
    @async_test
    async def test_load_based_autoscaling(self):
        """Test automatic scaling based on load"""
        # Create auto-scaling system
        auto_scaler = AutoScalingGridAttention(
            min_instances=1,
            max_instances=10,
            scale_up_threshold=0.8,  # 80% CPU
            scale_down_threshold=0.3,  # 30% CPU
            cooldown_seconds=30
        )
        
        await auto_scaler.initialize()
        
        # Simulate varying load
        load_pattern = [
            (60, 100),   # 60s low load
            (120, 1000), # 120s high load
            (60, 100),   # 60s low load
        ]
        
        scaling_events = []
        
        for duration, ops_per_sec in load_pattern:
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Generate load
                for _ in range(int(ops_per_sec / 10)):  # 100ms batches
                    await auto_scaler.process({
                        'price': 50000 + np.random.uniform(-100, 100),
                        'volume': np.random.uniform(10, 1000)
                    })
                
                # Check scaling events
                current_instances = auto_scaler.get_instance_count()
                if len(scaling_events) == 0 or scaling_events[-1]['instances'] != current_instances:
                    scaling_events.append({
                        'time': time.time() - start_time,
                        'instances': current_instances,
                        'cpu_usage': auto_scaler.get_average_cpu_usage()
                    })
                
                await asyncio.sleep(0.1)
        
        # Analyze scaling behavior
        logger.info("\nAuto-scaling Events:")
        for event in scaling_events:
            logger.info(f"  Time: {event['time']:.1f}s, Instances: {event['instances']}, "
                       f"CPU: {event['cpu_usage']:.1f}%")
        
        # Verify scaling occurred
        max_instances = max(e['instances'] for e in scaling_events)
        min_instances = min(e['instances'] for e in scaling_events)
        
        assert max_instances > min_instances, "No scaling occurred"
        assert max_instances <= 10, "Exceeded max instances"
        
        await auto_scaler.shutdown()
        return scaling_events
    
    @async_test
    async def test_predictive_scaling(self):
        """Test predictive scaling based on patterns"""
        # Create predictive scaler
        predictive_scaler = PredictiveScaler(
            historical_window=7 * 24 * 60,  # 7 days of minutes
            prediction_horizon=60  # Predict 60 minutes ahead
        )
        
        # Train on historical pattern (simulated)
        historical_pattern = self._generate_weekly_pattern()
        await predictive_scaler.train(historical_pattern)
        
        # Test predictions
        current_time = datetime.now()
        predictions = []
        
        for i in range(24):  # Next 24 hours
            future_time = current_time + timedelta(hours=i)
            predicted_load = await predictive_scaler.predict_load(future_time)
            recommended_instances = await predictive_scaler.recommend_instances(
                predicted_load,
                capacity_per_instance=1000
            )
            
            predictions.append({
                'time': future_time,
                'predicted_load': predicted_load,
                'recommended_instances': recommended_instances
            })
        
        # Verify predictions make sense
        logger.info("\nPredictive Scaling (Next 24 hours):")
        for pred in predictions[::4]:  # Every 4 hours
            logger.info(f"  {pred['time'].strftime('%H:%M')}: "
                       f"Load={pred['predicted_load']:.0f}, "
                       f"Instances={pred['recommended_instances']}")
        
        # Verify pattern detection
        day_predictions = [p['predicted_load'] for p in predictions[:12]]
        night_predictions = [p['predicted_load'] for p in predictions[12:]]
        
        assert np.mean(day_predictions) > np.mean(night_predictions), \
            "Predictive scaling failed to detect day/night pattern"
        
        return predictions
    
    def _generate_weekly_pattern(self):
        """Generate realistic weekly load pattern"""
        pattern = []
        
        for day in range(7):
            for hour in range(24):
                # Base load
                base_load = 500
                
                # Time of day factor (peak during business hours)
                if 9 <= hour <= 17:
                    time_factor = 2.0
                elif 6 <= hour <= 9 or 17 <= hour <= 22:
                    time_factor = 1.5
                else:
                    time_factor = 0.5
                
                # Day of week factor (lower on weekends)
                if day in [5, 6]:  # Weekend
                    day_factor = 0.6
                else:
                    day_factor = 1.0
                
                # Add some randomness
                noise = np.random.uniform(0.8, 1.2)
                
                load = base_load * time_factor * day_factor * noise
                
                # Generate minute-level data
                for minute in range(60):
                    pattern.append({
                        'timestamp': datetime.now() - timedelta(days=7-day, hours=23-hour, minutes=59-minute),
                        'load': load + np.random.uniform(-50, 50)
                    })
        
        return pattern


# Helper Classes

class LoadBalancer:
    """Load balancer for multiple instances"""
    
    def __init__(self, instances):
        self.instances = instances
        self.current_index = 0
    
    async def process(self, data):
        """Process data using round-robin load balancing"""
        instance = self.instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instances)
        return await instance.process_market_update(data)


class DistributedGridAttention:
    """Simulated distributed GridAttention system"""
    
    def __init__(self, node_configs):
        self.node_configs = node_configs
        self.nodes = {}
        self.partitioning_strategy = 'round_robin'
    
    async def initialize(self):
        """Initialize distributed nodes"""
        for config in self.node_configs:
            node = GridAttentionNode(config)
            await node.start()
            self.nodes[config['node_id']] = node
    
    def set_partitioning_strategy(self, strategy):
        """Set data partitioning strategy"""
        self.partitioning_strategy = strategy
    
    def get_node_for_data(self, data):
        """Get appropriate node for data"""
        if self.partitioning_strategy == 'round_robin':
            node_id = hash(data['symbol']) % len(self.nodes)
        elif self.partitioning_strategy == 'hash_based':
            node_id = hash(data['symbol']) % len(self.nodes)
        else:  # load_balanced
            # Find least loaded node
            node_id = min(self.nodes.keys(), 
                         key=lambda k: self.nodes[k].get_load())
        
        return self.nodes[node_id]
    
    async def shutdown(self):
        """Shutdown all nodes"""
        for node in self.nodes.values():
            await node.stop()


class GridAttentionNode:
    """Single node in distributed system"""
    
    def __init__(self, config):
        self.config = config
        self.load = 0
    
    async def start(self):
        """Start node"""
        pass
    
    async def stop(self):
        """Stop node"""
        pass
    
    async def process(self, data):
        """Process data on this node"""
        self.load += 1
        # Simulate processing
        await asyncio.sleep(0.001)
        self.load -= 1
    
    def get_load(self):
        """Get current load"""
        return self.load


class DataStream:
    """Simulated data stream"""
    
    def __init__(self, stream_id, symbol, rate):
        self.stream_id = stream_id
        self.symbol = symbol
        self.rate = rate
    
    async def generate(self, duration):
        """Generate data stream"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            yield {
                'stream_id': self.stream_id,
                'symbol': self.symbol,
                'price': 50000 + np.random.uniform(-100, 100),
                'volume': np.random.uniform(10, 1000),
                'timestamp': time.time()
            }
            
            await asyncio.sleep(1 / self.rate)


class AutoScalingGridAttention:
    """Auto-scaling GridAttention system"""
    
    def __init__(self, min_instances, max_instances, scale_up_threshold, scale_down_threshold, cooldown_seconds):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        self.instances = []
        self.last_scale_time = 0
        self.cpu_samples = deque(maxlen=60)  # 1 minute of samples
    
    async def initialize(self):
        """Initialize with minimum instances"""
        for i in range(self.min_instances):
            await self._add_instance()
    
    async def _add_instance(self):
        """Add new instance"""
        config = create_test_config()
        instance = GridAttentionLayer(config)
        self.instances.append(instance)
    
    async def _remove_instance(self):
        """Remove instance"""
        if len(self.instances) > self.min_instances:
            instance = self.instances.pop()
            await instance.shutdown()
    
    async def process(self, data):
        """Process data and monitor load"""
        # Round-robin to instances
        instance = self.instances[len(self.cpu_samples) % len(self.instances)]
        await instance.process_market_update(data)
        
        # Sample CPU usage
        cpu_usage = psutil.cpu_percent(interval=0)
        self.cpu_samples.append(cpu_usage)
        
        # Check scaling conditions
        await self._check_scaling()
    
    async def _check_scaling(self):
        """Check if scaling is needed"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scale_time < self.cooldown_seconds:
            return
        
        avg_cpu = self.get_average_cpu_usage()
        
        # Scale up
        if avg_cpu > self.scale_up_threshold * 100 and len(self.instances) < self.max_instances:
            await self._add_instance()
            self.last_scale_time = current_time
            logger.info(f"Scaled up to {len(self.instances)} instances (CPU: {avg_cpu:.1f}%)")
        
        # Scale down
        elif avg_cpu < self.scale_down_threshold * 100 and len(self.instances) > self.min_instances:
            await self._remove_instance()
            self.last_scale_time = current_time
            logger.info(f"Scaled down to {len(self.instances)} instances (CPU: {avg_cpu:.1f}%)")
    
    def get_instance_count(self):
        """Get current instance count"""
        return len(self.instances)
    
    def get_average_cpu_usage(self):
        """Get average CPU usage"""
        if not self.cpu_samples:
            return 0
        return np.mean(self.cpu_samples)
    
    async def shutdown(self):
        """Shutdown all instances"""
        for instance in self.instances:
            await instance.shutdown()


class PredictiveScaler:
    """Predictive scaling based on historical patterns"""
    
    def __init__(self, historical_window, prediction_horizon):
        self.historical_window = historical_window
        self.prediction_horizon = prediction_horizon
        self.model = None
    
    async def train(self, historical_data):
        """Train predictive model"""
        # Simple pattern-based prediction (in real system, use ML)
        self.pattern_data = historical_data
    
    async def predict_load(self, future_time):
        """Predict load at future time"""
        # Find similar time in historical data
        hour = future_time.hour
        day_of_week = future_time.weekday()
        
        similar_loads = []
        for data_point in self.pattern_data:
            if (data_point['timestamp'].hour == hour and 
                data_point['timestamp'].weekday() == day_of_week):
                similar_loads.append(data_point['load'])
        
        if similar_loads:
            return np.mean(similar_loads)
        return 500  # Default
    
    async def recommend_instances(self, predicted_load, capacity_per_instance):
        """Recommend number of instances"""
        # Add 20% buffer
        required_capacity = predicted_load * 1.2
        return max(1, int(np.ceil(required_capacity / capacity_per_instance)))