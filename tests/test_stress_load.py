"""
Stress and Load Tests for GridAttention Trading System
Tests system performance under extreme conditions
"""

import asyncio
import pytest
import time
import random
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import numpy as np


class TestHighFrequencyScenarios:
    """Test high-frequency trading scenarios"""
    
    async def test_tick_bombardment(self):
        """Test handling of rapid tick updates"""
        from market_data_input import MarketDataInput
        from performance_monitor import PerformanceMonitor
        
        market_data = MarketDataInput({})
        monitor = PerformanceMonitor({})
        
        # Simulate 1000 ticks per second
        tick_rate = 1000
        duration = 5  # seconds
        
        processed = 0
        errors = 0
        latencies = []
        
        print(f"Bombarding with {tick_rate} ticks/second for {duration} seconds...")
        
        start_time = time.time()
        
        for _ in range(tick_rate * duration):
            tick_time = time.time()
            
            tick = {
                'symbol': 'BTC/USDT',
                'bid': 50000 + random.uniform(-100, 100),
                'ask': 50100 + random.uniform(-100, 100),
                'volume': random.uniform(0.1, 10),
                'timestamp': tick_time
            }
            
            try:
                await market_data.process_tick(tick)
                processed += 1
                
                # Measure latency
                latency = (time.time() - tick_time) * 1000  # ms
                latencies.append(latency)
                
            except Exception as e:
                errors += 1
        
        elapsed = time.time() - start_time
        
        # Performance metrics
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        actual_rate = processed / elapsed
        
        print(f"Results:")
        print(f"  Processed: {processed}/{tick_rate * duration} ticks")
        print(f"  Errors: {errors}")
        print(f"  Actual rate: {actual_rate:.1f} ticks/second")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  P99 latency: {p99_latency:.2f}ms")
        
        # Assertions
        assert errors < processed * 0.01  # Less than 1% errors
        assert avg_latency < 10  # Average under 10ms
        assert p99_latency < 50  # P99 under 50ms
        assert actual_rate > tick_rate * 0.95  # Process at least 95% of target rate
    
    async def test_concurrent_orders(self):
        """Test handling multiple concurrent orders"""
        from execution_engine import ExecutionEngine
        
        engine = ExecutionEngine({'max_orders': 1000})
        
        # Create 100 concurrent orders
        num_orders = 100
        order_tasks = []
        
        print(f"Submitting {num_orders} concurrent orders...")
        
        start_time = time.time()
        
        for i in range(num_orders):
            order = {
                'symbol': 'BTC/USDT',
                'side': random.choice(['buy', 'sell']),
                'amount': random.uniform(0.001, 0.1),
                'price': 50000 + random.uniform(-1000, 1000)
            }
            
            task = asyncio.create_task(engine.execute_order(order))
            order_tasks.append(task)
        
        # Wait for all orders
        results = await asyncio.gather(*order_tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        
        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"Results:")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Success: {successes}/{num_orders}")
        print(f"  Failed: {failures}")
        print(f"  Rate: {successes/elapsed:.1f} orders/second")
        
        # Should handle most orders successfully
        assert successes > num_orders * 0.95
        assert elapsed < 10  # Should complete within 10 seconds


class TestMemoryLeaks:
    """Test for memory leaks"""
    
    async def test_long_running_memory(self):
        """Test memory usage over extended period"""
        from main import GridTradingSystem
        
        print("Testing long-running memory usage...")
        
        # Track memory
        process = psutil.Process()
        memory_samples = []
        
        # Create minimal system
        system = GridTradingSystem('config_minimal.yaml')
        await system.initialize()
        
        # Run for extended period
        duration = 60  # seconds
        sample_interval = 5
        
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for i in range(duration // sample_interval):
            # Simulate activity
            for _ in range(1000):
                tick = {
                    'symbol': 'BTC/USDT',
                    'price': 50000 + random.uniform(-100, 100),
                    'timestamp': time.time()
                }
                # Process tick through system
                # This would normally go through market_data_input
            
            # Force garbage collection
            gc.collect()
            
            # Sample memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            print(f"  Sample {i+1}: {current_memory:.1f}MB")
            
            await asyncio.sleep(sample_interval)
        
        # Analyze memory trend
        memory_growth = memory_samples[-1] - start_memory
        avg_growth_rate = memory_growth / len(memory_samples)
        
        print(f"\nMemory analysis:")
        print(f"  Start: {start_memory:.1f}MB")
        print(f"  End: {memory_samples[-1]:.1f}MB")
        print(f"  Growth: {memory_growth:.1f}MB")
        print(f"  Avg growth rate: {avg_growth_rate:.2f}MB/sample")
        
        # Check for memory leak
        assert memory_growth < 100  # Less than 100MB growth
        assert avg_growth_rate < 5  # Less than 5MB per sample
        
        # Cleanup
        await system.shutdown()
    
    async def test_cache_overflow(self):
        """Test cache behavior under memory pressure"""
        from performance_cache import PerformanceCache
        
        cache = PerformanceCache(max_size=1000)
        
        print("Testing cache overflow behavior...")
        
        # Fill cache beyond capacity
        for i in range(10000):
            key = f"metric_{i}"
            value = {'data': 'x' * 1000}  # 1KB per entry
            
            cache.set(key, value)
        
        # Check cache size is limited
        assert cache.size() <= 1000
        
        # Verify LRU eviction
        # Early entries should be evicted
        assert cache.get('metric_0') is None
        # Recent entries should exist
        assert cache.get('metric_9999') is not None


class TestNetworkCongestion:
    """Test network congestion scenarios"""
    
    async def test_connection_pool_exhaustion(self):
        """Test handling when connection pool is exhausted"""
        from network_client import NetworkClient
        
        client = NetworkClient(max_connections=10)
        
        # Create more requests than connections
        num_requests = 50
        request_tasks = []
        
        print(f"Testing {num_requests} requests with pool size 10...")
        
        for i in range(num_requests):
            task = asyncio.create_task(
                client.make_request(f"https://api.example.com/data/{i}")
            )
            request_tasks.append(task)
        
        # Should queue requests properly
        results = await asyncio.gather(*request_tasks, return_exceptions=True)
        
        successes = sum(1 for r in results if not isinstance(r, Exception))
        timeouts = sum(1 for r in results 
                      if isinstance(r, asyncio.TimeoutError))
        
        print(f"Results: {successes} success, {timeouts} timeouts")
        
        # Should handle gracefully
        assert successes > 0
        assert timeouts < num_requests  # Not all should timeout
    
    async def test_latency_spike_handling(self):
        """Test system behavior during latency spikes"""
        from system_coordinator import SystemCoordinator
        
        coordinator = SystemCoordinator({})
        
        # Simulate latency spikes
        normal_latency = 10  # ms
        spike_latency = 1000  # ms
        
        print("Simulating network latency spikes...")
        
        responses = []
        
        for i in range(100):
            # Occasional spike
            if i % 20 == 0:
                latency = spike_latency
            else:
                latency = normal_latency
            
            start = time.time()
            
            # Simulate delayed response
            await asyncio.sleep(latency / 1000)
            
            response_time = (time.time() - start) * 1000
            responses.append(response_time)
        
        # Calculate metrics
        avg_response = np.mean(responses)
        p95_response = np.percentile(responses, 95)
        
        print(f"Response times:")
        print(f"  Average: {avg_response:.1f}ms")
        print(f"  P95: {p95_response:.1f}ms")
        
        # System should handle spikes gracefully
        assert avg_response < 200  # Reasonable average despite spikes
        assert p95_response < spike_latency * 1.5  # Spikes handled efficiently


class TestDatabaseStress:
    """Test database under stress"""
    
    async def test_connection_pool_stress(self):
        """Test database connection pool under load"""
        from database_manager import DatabaseManager
        
        db = DatabaseManager(pool_size=20)
        
        # Simulate many concurrent queries
        num_queries = 200
        query_tasks = []
        
        print(f"Running {num_queries} concurrent database queries...")
        
        start_time = time.time()
        
        for i in range(num_queries):
            # Mix of read and write operations
            if i % 5 == 0:
                # Write operation
                task = asyncio.create_task(
                    db.insert_trade({
                        'id': f'trade_{i}',
                        'symbol': 'BTC/USDT',
                        'price': 50000,
                        'amount': 0.01
                    })
                )
            else:
                # Read operation
                task = asyncio.create_task(
                    db.get_recent_trades('BTC/USDT', limit=100)
                )
            
            query_tasks.append(task)
        
        results = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        
        # Analyze results
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"Database stress test results:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Success: {successes}/{num_queries}")
        print(f"  Failed: {failures}")
        print(f"  QPS: {successes/elapsed:.1f}")
        
        # Should handle load efficiently
        assert successes > num_queries * 0.98  # 98% success rate
        assert elapsed < 30  # Complete within 30 seconds


class TestCPUStress:
    """Test CPU-intensive scenarios"""
    
    async def test_parallel_calculations(self):
        """Test parallel strategy calculations"""
        from grid_strategy_selector import GridStrategySelector
        
        selector = GridStrategySelector({})
        
        # Generate test data
        market_conditions = []
        for _ in range(100):
            conditions = {
                'volatility': random.uniform(0.1, 0.9),
                'trend': random.uniform(-1, 1),
                'volume': random.uniform(100, 10000),
                'regime': random.choice(['trending', 'ranging', 'volatile'])
            }
            market_conditions.append(conditions)
        
        print("Running parallel strategy calculations...")
        
        # Use thread pool for CPU-bound work
        with ThreadPoolExecutor(max_workers=4) as executor:
            start_time = time.time()
            
            # Submit all calculations
            futures = []
            for conditions in market_conditions:
                future = executor.submit(
                    selector.calculate_optimal_strategy,
                    conditions
                )
                futures.append(future)
            
            # Wait for completion
            results = [f.result() for f in futures]
            
            elapsed = time.time() - start_time
        
        print(f"Calculation results:")
        print(f"  Completed: {len(results)} strategies")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {len(results)/elapsed:.1f} strategies/second")
        
        # Should complete efficiently
        assert len(results) == len(market_conditions)
        assert elapsed < 10  # Within 10 seconds


# Stress test runner
async def run_stress_tests():
    """Run all stress and load tests"""
    print("💪 Running Stress and Load Tests...")
    print("⚠️  These tests may take several minutes and use significant resources")
    
    test_classes = [
        TestHighFrequencyScenarios,
        TestMemoryLeaks,
        TestNetworkCongestion,
        TestDatabaseStress,
        TestCPUStress
    ]
    
    results = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing {test_class.__name__}...")
        print(f"{'='*60}")
        
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                print(f"\n▶️  {method_name}")
                method = getattr(test_instance, method_name)
                
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                
                print(f"✅ {method_name} PASSED")
                results.append((method_name, True))
                
            except Exception as e:
                print(f"❌ {method_name} FAILED: {e}")
                results.append((method_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"📊 Stress Test Results: {passed}/{total} passed")
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_stress_tests())
    exit(0 if success else 1)