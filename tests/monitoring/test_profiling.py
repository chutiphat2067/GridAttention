"""
System profiling tests for GridAttention trading system.

Tests comprehensive profiling capabilities including CPU, memory, I/O profiling,
code hotspot detection, latency analysis, and performance optimization recommendations.
"""

import pytest
import asyncio
import time
import cProfile
import pstats
import io
import tracemalloc
import gc
import sys
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from memory_profiler import profile as memory_profile
import psutil
import threading
from collections import defaultdict, deque
from unittest.mock import Mock, patch, AsyncMock
import line_profiler
import py_spy

# Import core components
from core.profiling_system import ProfilingSystem
from core.performance_analyzer import PerformanceAnalyzer
from core.bottleneck_detector import BottleneckDetector
from core.optimization_advisor import OptimizationAdvisor


class ProfileType(Enum):
    """Types of profiling available"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    ASYNC = "async"
    GPU = "gpu"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


class ProfileMode(Enum):
    """Profiling modes"""
    DEVELOPMENT = "development"  # Detailed profiling
    PRODUCTION = "production"   # Low-overhead profiling
    DIAGNOSTIC = "diagnostic"   # Deep analysis when issues detected


@dataclass
class ProfileResult:
    """Profiling result data"""
    profile_id: str
    profile_type: ProfileType
    start_time: datetime
    end_time: datetime
    duration: timedelta
    metrics: Dict[str, Any]
    hotspots: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    component: str
    operation: str
    baseline_metrics: Dict[str, float]
    thresholds: Dict[str, float]
    recorded_at: datetime


class TestProfilingSystem:
    """Test profiling system functionality"""
    
    @pytest.fixture
    async def profiling_system(self):
        """Create profiling system instance"""
        return ProfilingSystem(
            enable_auto_profiling=True,
            sample_rate=0.1,  # 10% sampling in production
            profile_storage_days=7,
            enable_continuous_monitoring=True
        )
    
    @pytest.fixture
    async def performance_analyzer(self):
        """Create performance analyzer instance"""
        return PerformanceAnalyzer(
            enable_ml_analysis=True,
            anomaly_detection_enabled=True,
            baseline_learning_period=timedelta(days=7)
        )
    
    @pytest.fixture
    async def bottleneck_detector(self):
        """Create bottleneck detector instance"""
        return BottleneckDetector(
            detection_threshold=0.2,  # 20% of total time
            min_samples=100,
            enable_real_time_detection=True
        )
    
    @pytest.mark.asyncio
    async def test_cpu_profiling(self, profiling_system):
        """Test CPU profiling capabilities"""
        # Define CPU-intensive function
        def calculate_indicators(prices: List[float], window: int = 20) -> Dict[str, List[float]]:
            """Simulate technical indicator calculations"""
            results = {
                'sma': [],
                'ema': [],
                'rsi': [],
                'bollinger_upper': [],
                'bollinger_lower': []
            }
            
            for i in range(len(prices)):
                # Simple Moving Average
                if i >= window:
                    sma = sum(prices[i-window:i]) / window
                    results['sma'].append(sma)
                    
                    # Exponential Moving Average (simplified)
                    multiplier = 2 / (window + 1)
                    if results['ema']:
                        ema = prices[i] * multiplier + results['ema'][-1] * (1 - multiplier)
                    else:
                        ema = sma
                    results['ema'].append(ema)
                    
                    # RSI calculation (simplified)
                    gains = [prices[j] - prices[j-1] for j in range(i-window+1, i+1) 
                            if prices[j] > prices[j-1]]
                    losses = [prices[j-1] - prices[j] for j in range(i-window+1, i+1) 
                             if prices[j] < prices[j-1]]
                    
                    avg_gain = sum(gains) / window if gains else 0
                    avg_loss = sum(losses) / window if losses else 0
                    
                    if avg_loss != 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 100 if avg_gain > 0 else 50
                    results['rsi'].append(rsi)
                    
                    # Bollinger Bands
                    std_dev = np.std(prices[i-window:i])
                    results['bollinger_upper'].append(sma + 2 * std_dev)
                    results['bollinger_lower'].append(sma - 2 * std_dev)
            
            return results
        
        # Generate test data
        prices = [50000 + np.random.randn() * 1000 for _ in range(1000)]
        
        # Profile the function
        profile_result = await profiling_system.profile_function(
            func=calculate_indicators,
            args=(prices,),
            kwargs={'window': 20},
            profile_type=ProfileType.CPU
        )
        
        assert profile_result.profile_type == ProfileType.CPU
        assert profile_result.duration > timedelta(0)
        
        # Check CPU metrics
        cpu_metrics = profile_result.metrics
        assert 'total_time' in cpu_metrics
        assert 'function_calls' in cpu_metrics
        assert 'primitive_calls' in cpu_metrics
        
        # Check hotspots
        assert len(profile_result.hotspots) > 0
        
        # Top hotspot should be the main calculation
        top_hotspot = profile_result.hotspots[0]
        assert 'function' in top_hotspot
        assert 'cumulative_time' in top_hotspot
        assert 'self_time' in top_hotspot
        assert 'calls' in top_hotspot
        
        # Check recommendations
        assert len(profile_result.recommendations) > 0
        
        # Test line-by-line profiling for detailed analysis
        detailed_profile = await profiling_system.profile_lines(
            func=calculate_indicators,
            args=(prices[:100],),  # Smaller dataset for line profiling
            focus_on=['sma', 'rsi']  # Focus on specific calculations
        )
        
        assert 'line_timings' in detailed_profile
        assert len(detailed_profile['line_timings']) > 0
        
        # Find most expensive lines
        expensive_lines = sorted(
            detailed_profile['line_timings'],
            key=lambda x: x['time'],
            reverse=True
        )[:5]
        
        assert all('line_number' in line for line in expensive_lines)
        assert all('code' in line for line in expensive_lines)
        assert all('time' in line for line in expensive_lines)
    
    @pytest.mark.asyncio
    async def test_memory_profiling(self, profiling_system):
        """Test memory profiling and leak detection"""
        # Enable memory profiling
        tracemalloc.start()
        
        # Function that might have memory issues
        async def process_large_dataset():
            """Simulate processing that uses significant memory"""
            # Intentional memory accumulation
            accumulated_data = []
            
            for batch in range(10):
                # Simulate loading batch data
                batch_data = {
                    'prices': np.random.rand(10000),
                    'volumes': np.random.rand(10000),
                    'timestamps': [datetime.now() for _ in range(10000)]
                }
                
                # Process batch
                processed = {
                    'avg_price': np.mean(batch_data['prices']),
                    'total_volume': np.sum(batch_data['volumes']),
                    'records': len(batch_data['timestamps'])
                }
                
                # Accumulate (potential memory issue)
                accumulated_data.append(batch_data)  # Keeping all raw data
                
                # Simulate some async work
                await asyncio.sleep(0.01)
            
            return processed, accumulated_data
        
        # Take memory snapshot before
        snapshot1 = tracemalloc.take_snapshot()
        
        # Profile memory usage
        memory_profile = await profiling_system.profile_memory(
            func=process_large_dataset,
            track_allocations=True,
            detect_leaks=True
        )
        
        # Take memory snapshot after
        snapshot2 = tracemalloc.take_snapshot()
        
        # Analyze memory growth
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        assert memory_profile.profile_type == ProfileType.MEMORY
        
        # Check memory metrics
        memory_metrics = memory_profile.metrics
        assert 'peak_memory_mb' in memory_metrics
        assert 'memory_growth_mb' in memory_metrics
        assert 'allocations' in memory_metrics
        assert 'deallocations' in memory_metrics
        
        # Check for memory leaks
        assert 'potential_leaks' in memory_profile.metrics
        if memory_profile.metrics['potential_leaks']:
            leak = memory_profile.metrics['potential_leaks'][0]
            assert 'location' in leak
            assert 'size_mb' in leak
            assert 'count' in leak
        
        # Check memory hotspots
        assert len(memory_profile.hotspots) > 0
        
        # Verify recommendations
        if memory_metrics['memory_growth_mb'] > 10:  # Significant growth
            assert any('memory' in rec.lower() for rec in memory_profile.recommendations)
        
        tracemalloc.stop()
        
        # Test garbage collection analysis
        gc_profile = await profiling_system.analyze_garbage_collection()
        
        assert 'gc_stats' in gc_profile
        assert 'generation_0' in gc_profile['gc_stats']
        assert 'generation_1' in gc_profile['gc_stats']
        assert 'generation_2' in gc_profile['gc_stats']
        
        # Check for circular references
        assert 'circular_references' in gc_profile
        assert 'uncollectable_objects' in gc_profile
    
    @pytest.mark.asyncio
    async def test_async_profiling(self, profiling_system):
        """Test async/await profiling for coroutines"""
        # Define async operations to profile
        async def fetch_market_data(symbol: str) -> Dict[str, Any]:
            """Simulate async market data fetch"""
            await asyncio.sleep(0.05)  # Simulate network delay
            return {
                'symbol': symbol,
                'price': 50000 + np.random.randn() * 1000,
                'volume': np.random.exponential(10)
            }
        
        async def process_orders(orders: List[Dict]) -> List[Dict]:
            """Simulate async order processing"""
            results = []
            
            # Process orders concurrently
            tasks = []
            for order in orders:
                async def process_single_order(o):
                    await asyncio.sleep(0.02)  # Simulate processing
                    
                    # Fetch market data
                    market_data = await fetch_market_data(o['symbol'])
                    
                    return {
                        'order_id': o['id'],
                        'status': 'processed',
                        'executed_price': market_data['price']
                    }
                
                tasks.append(process_single_order(order))
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Test data
        test_orders = [
            {'id': f'ORD_{i}', 'symbol': 'BTC/USDT', 'quantity': 0.1}
            for i in range(20)
        ]
        
        # Profile async operations
        async_profile = await profiling_system.profile_async(
            func=process_orders,
            args=(test_orders,),
            trace_tasks=True,
            measure_wait_time=True
        )
        
        assert async_profile.profile_type == ProfileType.ASYNC
        
        # Check async metrics
        async_metrics = async_profile.metrics
        assert 'total_time' in async_metrics
        assert 'active_time' in async_metrics
        assert 'wait_time' in async_metrics
        assert 'task_count' in async_metrics
        assert 'concurrent_tasks_max' in async_metrics
        
        # Calculate async efficiency
        efficiency = async_metrics['active_time'] / async_metrics['total_time']
        assert efficiency < 1.0  # Should have some wait time
        
        # Check task breakdown
        assert 'task_timings' in async_profile.metrics
        task_timings = async_profile.metrics['task_timings']
        assert len(task_timings) > 0
        
        # Verify hotspots show async bottlenecks
        async_hotspots = [h for h in async_profile.hotspots if 'await' in str(h.get('location', ''))]
        assert len(async_hotspots) > 0
        
        # Test event loop analysis
        loop_analysis = await profiling_system.analyze_event_loop()
        
        assert 'pending_tasks' in loop_analysis
        assert 'running_tasks' in loop_analysis
        assert 'task_queue_size' in loop_analysis
        assert 'loop_utilization' in loop_analysis
    
    @pytest.mark.asyncio
    async def test_io_profiling(self, profiling_system):
        """Test I/O profiling for disk and network operations"""
        # Create test file
        test_file = Path('test_io_profile.dat')
        
        try:
            # Function with I/O operations
            async def io_intensive_operation():
                """Simulate I/O intensive operations"""
                results = {
                    'reads': 0,
                    'writes': 0,
                    'network_calls': 0
                }
                
                # File I/O operations
                # Write test data
                with open(test_file, 'wb') as f:
                    for i in range(100):
                        data = os.urandom(1024 * 10)  # 10KB chunks
                        f.write(data)
                        results['writes'] += 1
                
                # Read test data
                with open(test_file, 'rb') as f:
                    while True:
                        chunk = f.read(1024 * 10)
                        if not chunk:
                            break
                        results['reads'] += 1
                
                # Simulate network I/O
                for i in range(10):
                    await asyncio.sleep(0.01)  # Simulate network delay
                    results['network_calls'] += 1
                
                return results
            
            # Profile I/O operations
            io_profile = await profiling_system.profile_io(
                func=io_intensive_operation,
                track_file_io=True,
                track_network_io=True,
                track_database_io=True
            )
            
            assert io_profile.profile_type == ProfileType.IO
            
            # Check I/O metrics
            io_metrics = io_profile.metrics
            assert 'total_io_time' in io_metrics
            assert 'file_io' in io_metrics
            assert 'network_io' in io_metrics
            
            # File I/O breakdown
            file_io = io_metrics['file_io']
            assert 'read_ops' in file_io
            assert 'write_ops' in file_io
            assert 'read_bytes' in file_io
            assert 'write_bytes' in file_io
            assert 'avg_read_time_ms' in file_io
            assert 'avg_write_time_ms' in file_io
            
            # Check I/O hotspots
            io_hotspots = profile_result.hotspots
            write_hotspots = [h for h in io_hotspots if 'write' in h.get('operation', '').lower()]
            read_hotspots = [h for h in io_hotspots if 'read' in h.get('operation', '').lower()]
            
            assert len(write_hotspots) > 0
            assert len(read_hotspots) > 0
            
            # Verify recommendations for I/O optimization
            io_recommendations = [r for r in io_profile.recommendations if 'I/O' in r or 'buffer' in r]
            assert len(io_recommendations) > 0
            
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()
    
    @pytest.mark.asyncio
    async def test_latency_profiling(self, profiling_system):
        """Test latency profiling for trading operations"""
        # Define trading operations with various latencies
        async def execute_trade_flow(order: Dict) -> Dict:
            """Simulate complete trade execution flow"""
            timestamps = {'start': datetime.now(timezone.utc)}
            
            # Step 1: Validation (fast)
            await asyncio.sleep(0.001)  # 1ms
            timestamps['validated'] = datetime.now(timezone.utc)
            
            # Step 2: Risk check (medium)
            await asyncio.sleep(0.005)  # 5ms
            timestamps['risk_checked'] = datetime.now(timezone.utc)
            
            # Step 3: Market data fetch (variable)
            await asyncio.sleep(0.01 + np.random.exponential(0.005))  # 10-20ms avg
            timestamps['market_data'] = datetime.now(timezone.utc)
            
            # Step 4: Order routing (fast)
            await asyncio.sleep(0.002)  # 2ms
            timestamps['routed'] = datetime.now(timezone.utc)
            
            # Step 5: Exchange execution (variable, potentially slow)
            await asyncio.sleep(0.05 + np.random.exponential(0.02))  # 50-90ms avg
            timestamps['executed'] = datetime.now(timezone.utc)
            
            # Step 6: Confirmation (fast)
            await asyncio.sleep(0.001)  # 1ms
            timestamps['confirmed'] = datetime.now(timezone.utc)
            
            return {
                'order_id': order['id'],
                'status': 'executed',
                'timestamps': timestamps
            }
        
        # Profile latency for multiple orders
        test_orders = [{'id': f'LAT_ORD_{i}'} for i in range(50)]
        
        latency_profile = await profiling_system.profile_latency(
            func=execute_trade_flow,
            inputs=test_orders,
            trace_components=True,
            percentiles=[50, 75, 90, 95, 99]
        )
        
        assert latency_profile.profile_type == ProfileType.LATENCY
        
        # Check latency metrics
        latency_metrics = latency_profile.metrics
        assert 'end_to_end_latency' in latency_metrics
        assert 'component_latencies' in latency_metrics
        assert 'percentiles' in latency_metrics
        
        # Check percentiles
        percentiles = latency_metrics['percentiles']
        assert percentiles['p50'] < percentiles['p95']
        assert percentiles['p95'] < percentiles['p99']
        
        # Check component breakdown
        components = latency_metrics['component_latencies']
        expected_components = [
            'validation', 'risk_check', 'market_data',
            'routing', 'execution', 'confirmation'
        ]
        
        for component in expected_components:
            assert component in components
            assert 'avg_ms' in components[component]
            assert 'min_ms' in components[component]
            assert 'max_ms' in components[component]
        
        # Identify slowest component
        slowest_component = max(
            components.items(),
            key=lambda x: x[1]['avg_ms']
        )
        assert slowest_component[0] == 'execution'  # Should be exchange execution
        
        # Check latency distribution
        assert 'latency_distribution' in latency_profile.metrics
        distribution = latency_profile.metrics['latency_distribution']
        assert 'histogram' in distribution
        assert 'outliers' in distribution
        
        # Verify SLA compliance check
        sla_check = await profiling_system.check_latency_sla(
            latency_profile=latency_profile,
            sla_requirements={
                'p95_max_ms': 100,
                'p99_max_ms': 150
            }
        )
        
        assert 'compliant' in sla_check
        assert 'violations' in sla_check
    
    @pytest.mark.asyncio
    async def test_throughput_profiling(self, profiling_system):
        """Test throughput profiling for high-frequency operations"""
        # Queue for order processing
        order_queue = asyncio.Queue(maxsize=1000)
        processed_count = 0
        processing_lock = asyncio.Lock()
        
        async def order_processor():
            """Process orders from queue"""
            nonlocal processed_count
            
            while True:
                try:
                    order = await asyncio.wait_for(
                        order_queue.get(),
                        timeout=0.1
                    )
                    
                    # Simulate processing
                    await asyncio.sleep(0.001)  # 1ms per order
                    
                    async with processing_lock:
                        processed_count += 1
                    
                except asyncio.TimeoutError:
                    break
        
        async def load_generator(rate_per_second: int, duration_seconds: int):
            """Generate orders at specified rate"""
            interval = 1.0 / rate_per_second
            start_time = asyncio.get_event_loop().time()
            
            order_id = 0
            while asyncio.get_event_loop().time() - start_time < duration_seconds:
                order = {'id': f'TP_ORD_{order_id}', 'timestamp': datetime.now()}
                
                try:
                    await order_queue.put(order)
                    order_id += 1
                except asyncio.QueueFull:
                    pass  # Drop order if queue full
                
                await asyncio.sleep(interval)
        
        # Profile throughput
        throughput_profile = await profiling_system.profile_throughput(
            producer=load_generator,
            consumer=order_processor,
            producer_args=(1000, 5),  # 1000 orders/sec for 5 seconds
            consumer_count=10,  # 10 concurrent processors
            measure_backpressure=True
        )
        
        assert throughput_profile.profile_type == ProfileType.THROUGHPUT
        
        # Check throughput metrics
        throughput_metrics = throughput_profile.metrics
        assert 'input_rate' in throughput_metrics
        assert 'processing_rate' in throughput_metrics
        assert 'queue_metrics' in throughput_metrics
        assert 'dropped_items' in throughput_metrics
        
        # Queue metrics
        queue_metrics = throughput_metrics['queue_metrics']
        assert 'avg_size' in queue_metrics
        assert 'max_size' in queue_metrics
        assert 'utilization' in queue_metrics
        
        # Performance metrics
        assert throughput_metrics['processing_rate'] > 0
        
        # Check for bottlenecks
        if throughput_metrics['processing_rate'] < throughput_metrics['input_rate']:
            assert any('bottleneck' in rec.lower() for rec in throughput_profile.recommendations)
    
    @pytest.mark.asyncio
    async def test_bottleneck_detection(self, profiling_system, bottleneck_detector):
        """Test automatic bottleneck detection"""
        # Create a system with obvious bottlenecks
        async def trading_pipeline(market_data: Dict) -> Dict:
            """Simulate trading pipeline with bottlenecks"""
            # Fast: Data validation
            await asyncio.sleep(0.001)
            
            # BOTTLENECK 1: Complex calculation (CPU bound)
            indicators = []
            for i in range(1000):
                # Intentionally inefficient calculation
                value = sum(j ** 2 for j in range(100))
                indicators.append(value)
            
            # Fast: Signal generation
            await asyncio.sleep(0.002)
            
            # BOTTLENECK 2: Database query (I/O bound)
            await asyncio.sleep(0.1)  # Simulate slow DB
            
            # Fast: Risk check
            await asyncio.sleep(0.001)
            
            # BOTTLENECK 3: External API call (Network bound)
            await asyncio.sleep(0.15)  # Simulate slow API
            
            return {'status': 'completed'}
        
        # Profile with bottleneck detection
        test_data = {'symbol': 'BTC/USDT', 'price': 50000}
        
        bottleneck_analysis = await bottleneck_detector.analyze_pipeline(
            pipeline=trading_pipeline,
            input_data=test_data,
            iterations=10
        )
        
        assert 'bottlenecks' in bottleneck_analysis
        assert len(bottleneck_analysis['bottlenecks']) >= 2
        
        # Check bottleneck details
        for bottleneck in bottleneck_analysis['bottlenecks']:
            assert 'location' in bottleneck
            assert 'type' in bottleneck  # CPU, I/O, Network
            assert 'impact_percent' in bottleneck
            assert 'avg_time_ms' in bottleneck
            assert 'optimization_potential' in bottleneck
        
        # Verify bottleneck types detected
        bottleneck_types = {b['type'] for b in bottleneck_analysis['bottlenecks']}
        assert 'CPU' in bottleneck_types
        assert 'I/O' in bottleneck_types or 'Network' in bottleneck_types
        
        # Check optimization recommendations
        assert 'optimization_plan' in bottleneck_analysis
        plan = bottleneck_analysis['optimization_plan']
        assert len(plan) > 0
        
        for optimization in plan:
            assert 'bottleneck_id' in optimization
            assert 'recommendation' in optimization
            assert 'expected_improvement' in optimization
            assert 'implementation_effort' in optimization
    
    @pytest.mark.asyncio
    async def test_production_profiling(self, profiling_system):
        """Test low-overhead production profiling"""
        # Configure production mode
        await profiling_system.set_mode(ProfileMode.PRODUCTION)
        
        # Simulate production workload
        async def production_workload():
            """Simulate typical production operations"""
            operations = []
            
            for i in range(1000):
                # Various operations that happen in production
                op_type = np.random.choice(['trade', 'quote', 'risk_check', 'market_data'])
                
                if op_type == 'trade':
                    await asyncio.sleep(0.01)
                    operations.append({'type': 'trade', 'latency': 10})
                elif op_type == 'quote':
                    await asyncio.sleep(0.001)
                    operations.append({'type': 'quote', 'latency': 1})
                elif op_type == 'risk_check':
                    await asyncio.sleep(0.005)
                    operations.append({'type': 'risk_check', 'latency': 5})
                else:  # market_data
                    await asyncio.sleep(0.002)
                    operations.append({'type': 'market_data', 'latency': 2})
            
            return operations
        
        # Profile with production settings
        prod_profile = await profiling_system.profile_production(
            workload=production_workload,
            duration_seconds=10,
            sample_rate=0.1  # Only sample 10% to reduce overhead
        )
        
        # Verify low overhead
        overhead_metrics = prod_profile.metrics.get('overhead', {})
        assert overhead_metrics.get('cpu_overhead_percent', 0) < 5  # Less than 5% CPU overhead
        assert overhead_metrics.get('memory_overhead_mb', 0) < 50  # Less than 50MB overhead
        
        # Check sampled metrics are still useful
        assert 'sampled_operations' in prod_profile.metrics
        assert prod_profile.metrics['sampled_operations'] > 0
        
        # Verify statistical accuracy despite sampling
        assert 'operation_latencies' in prod_profile.metrics
        latencies = prod_profile.metrics['operation_latencies']
        assert all(op_type in latencies for op_type in ['trade', 'quote', 'risk_check', 'market_data'])
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, profiling_system, performance_analyzer):
        """Test detection of performance regressions"""
        # Establish baseline performance
        async def test_operation(complexity: int):
            """Operation that can have varying performance"""
            result = 0
            for i in range(complexity * 100):
                result += i ** 2
            await asyncio.sleep(complexity * 0.001)
            return result
        
        # Create baseline
        baseline_results = []
        for _ in range(20):
            profile = await profiling_system.profile_function(
                func=test_operation,
                args=(10,),  # Normal complexity
                profile_type=ProfileType.CPU
            )
            baseline_results.append(profile.metrics['total_time'])
        
        baseline = PerformanceBaseline(
            component='test_operation',
            operation='calculate',
            baseline_metrics={
                'avg_time_ms': np.mean(baseline_results) * 1000,
                'std_dev_ms': np.std(baseline_results) * 1000,
                'p95_ms': np.percentile(baseline_results, 95) * 1000
            },
            thresholds={
                'avg_regression_percent': 20,  # 20% slower is regression
                'p95_regression_percent': 30   # 30% slower at p95
            },
            recorded_at=datetime.now(timezone.utc)
        )
        
        await performance_analyzer.save_baseline(baseline)
        
        # Test with degraded performance
        degraded_results = []
        for _ in range(20):
            profile = await profiling_system.profile_function(
                func=test_operation,
                args=(15,),  # Higher complexity (regression)
                profile_type=ProfileType.CPU
            )
            degraded_results.append(profile.metrics['total_time'])
        
        # Check for regression
        regression_check = await performance_analyzer.check_regression(
            component='test_operation',
            operation='calculate',
            current_metrics={
                'avg_time_ms': np.mean(degraded_results) * 1000,
                'p95_ms': np.percentile(degraded_results, 95) * 1000
            }
        )
        
        assert regression_check['regression_detected'] == True
        assert 'regression_severity' in regression_check
        assert 'affected_metrics' in regression_check
        assert len(regression_check['affected_metrics']) > 0
        
        # Check regression details
        for metric in regression_check['affected_metrics']:
            assert 'metric_name' in metric
            assert 'baseline_value' in metric
            assert 'current_value' in metric
            assert 'regression_percent' in metric
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, profiling_system, optimization_advisor):
        """Test optimization recommendation generation"""
        # Create a function with multiple optimization opportunities
        def suboptimal_function(data: List[Dict]) -> Dict:
            """Function with various optimization opportunities"""
            results = {}
            
            # Inefficiency 1: Repeated expensive operations
            for item in data:
                # Calculate same value multiple times
                expensive_calc = sum(i ** 2 for i in range(1000))
                item['calc1'] = expensive_calc
                item['calc2'] = expensive_calc  # Same calculation
            
            # Inefficiency 2: Inefficient data structure
            lookup_dict = {}
            for item in data:
                # Using string concatenation as key (inefficient)
                key = str(item['id']) + '_' + str(item['type'])
                lookup_dict[key] = item
            
            # Inefficiency 3: Unnecessary list operations
            filtered_items = []
            for item in data:
                if item.get('active', False):
                    filtered_items.append(item)
            
            # Could use list comprehension
            sorted_items = []
            for item in filtered_items:
                sorted_items.append(item)
            sorted_items.sort(key=lambda x: x['id'])  # Inefficient sort
            
            # Inefficiency 4: Multiple passes over data
            total = 0
            for item in data:
                total += item.get('value', 0)
            
            average = 0
            count = 0
            for item in data:
                if 'value' in item:
                    average += item['value']
                    count += 1
            
            if count > 0:
                average = average / count
            
            return {
                'total': total,
                'average': average,
                'filtered_count': len(filtered_items)
            }
        
        # Generate test data
        test_data = [
            {
                'id': i,
                'type': 'A' if i % 2 == 0 else 'B',
                'active': i % 3 == 0,
                'value': i * 10
            }
            for i in range(1000)
        ]
        
        # Profile and get optimization recommendations
        profile_result = await profiling_system.profile_function(
            func=suboptimal_function,
            args=(test_data,),
            profile_type=ProfileType.CPU
        )
        
        optimizations = await optimization_advisor.analyze_profile(
            profile=profile_result,
            code_analysis=True,
            suggest_alternatives=True
        )
        
        assert 'recommendations' in optimizations
        assert len(optimizations['recommendations']) > 0
        
        # Check recommendation categories
        recommendation_categories = {r['category'] for r in optimizations['recommendations']}
        expected_categories = {
            'caching',  # For repeated calculations
            'data_structure',  # For inefficient lookups
            'algorithm',  # For sorting and filtering
            'code_style'  # For list comprehensions
        }
        
        assert len(recommendation_categories.intersection(expected_categories)) >= 2
        
        # Check specific recommendations
        for rec in optimizations['recommendations']:
            assert 'description' in rec
            assert 'impact' in rec  # Expected performance improvement
            assert 'effort' in rec  # Implementation effort
            assert 'code_example' in rec  # How to implement
            
            # High impact recommendations should be prioritized
            if rec['impact'] == 'high':
                assert rec['priority'] <= 3  # Top 3 priority
        
        # Test automated optimization application (for simple cases)
        if optimizations.get('auto_applicable'):
            optimized_code = await optimization_advisor.apply_optimizations(
                original_code=suboptimal_function,
                safe_only=True  # Only apply safe optimizations
            )
            
            assert optimized_code is not None
            assert 'modified_code' in optimized_code
            assert 'applied_optimizations' in optimized_code
            assert len(optimized_code['applied_optimizations']) > 0


class TestProfilingIntegration:
    """Test profiling integration with trading system"""
    
    @pytest.mark.asyncio
    async def test_live_trading_profiling(self, profiling_system):
        """Test profiling integration with live trading components"""
        # Simulate live trading system components
        class TradingSystem:
            def __init__(self):
                self.orders_processed = 0
                self.market_data_updates = 0
                self.risk_checks = 0
            
            async def process_market_data(self, data: Dict):
                """Process incoming market data"""
                self.market_data_updates += 1
                # Simulate processing
                await asyncio.sleep(0.001)
                return {'processed': True}
            
            async def execute_order(self, order: Dict):
                """Execute trading order"""
                self.orders_processed += 1
                
                # Risk check
                await self.check_risk(order)
                
                # Market data check
                await self.process_market_data({'symbol': order['symbol']})
                
                # Execute
                await asyncio.sleep(0.01)  # Simulate execution
                
                return {'status': 'executed', 'order_id': order['id']}
            
            async def check_risk(self, order: Dict):
                """Perform risk checks"""
                self.risk_checks += 1
                await asyncio.sleep(0.002)
                return {'approved': True}
        
        trading_system = TradingSystem()
        
        # Enable continuous profiling
        await profiling_system.start_continuous_profiling(
            components={
                'market_data': trading_system.process_market_data,
                'order_execution': trading_system.execute_order,
                'risk_management': trading_system.check_risk
            },
            interval_seconds=5,
            metrics_to_track=['cpu', 'memory', 'latency']
        )
        
        # Simulate trading activity
        async def simulate_trading():
            tasks = []
            
            # Market data updates
            for i in range(100):
                tasks.append(trading_system.process_market_data({
                    'symbol': 'BTC/USDT',
                    'price': 50000 + i
                }))
            
            # Order executions
            for i in range(20):
                tasks.append(trading_system.execute_order({
                    'id': f'LIVE_ORD_{i}',
                    'symbol': 'BTC/USDT',
                    'quantity': 0.1
                }))
            
            await asyncio.gather(*tasks)
        
        # Run simulation
        await simulate_trading()
        
        # Wait for profiling data
        await asyncio.sleep(1)
        
        # Stop profiling and get report
        profiling_report = await profiling_system.stop_continuous_profiling()
        
        assert 'components' in profiling_report
        assert 'market_data' in profiling_report['components']
        assert 'order_execution' in profiling_report['components']
        assert 'risk_management' in profiling_report['components']
        
        # Check component metrics
        for component, metrics in profiling_report['components'].items():
            assert 'avg_cpu_percent' in metrics
            assert 'avg_memory_mb' in metrics
            assert 'avg_latency_ms' in metrics
            assert 'call_count' in metrics
            assert 'error_rate' in metrics
        
        # Verify system health assessment
        assert 'health_score' in profiling_report
        assert 0 <= profiling_report['health_score'] <= 100
        
        # Check for any warnings
        if profiling_report.get('warnings'):
            for warning in profiling_report['warnings']:
                assert 'component' in warning
                assert 'issue' in warning
                assert 'severity' in warning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])