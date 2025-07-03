"""
Stress and Load Testing Suite for GridAttention Trading System
Tests system performance under extreme conditions and high loads
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import psutil
import gc
import time
import threading
import multiprocessing
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional, Tuple
import random
import logging
from memory_profiler import profile
import aiohttp
import websockets

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
    measure_memory_usage,
    measure_execution_time
)
from tests.mocks.mock_exchange import MockExchange
from tests.fixtures.market_data import generate_high_frequency_data

# Configure logging for stress tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHighLoadScenarios:
    """Test system behavior under high load conditions"""
    
    @pytest.fixture
    def grid_attention_system(self):
        """Create full GridAttention system"""
        config = create_test_config()
        return GridAttentionLayer(config)
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange with high-frequency capabilities"""
        exchange = MockExchange()
        exchange.enable_high_frequency_mode()
        return exchange
    
    @async_test
    async def test_concurrent_market_updates(self, grid_attention_system):
        """Test system with concurrent high-frequency market updates"""
        num_concurrent_updates = 1000
        update_tasks = []
        
        async def send_market_update(index):
            """Simulate market data update"""
            data = {
                'symbol': 'BTC/USDT',
                'price': 50000 + random.uniform(-100, 100),
                'volume': random.uniform(0.1, 10),
                'timestamp': datetime.now().timestamp() + index
            }
            await grid_attention_system.process_market_update(data)
            return index
        
        # Create concurrent update tasks
        start_time = time.time()
        
        for i in range(num_concurrent_updates):
            task = asyncio.create_task(send_market_update(i))
            update_tasks.append(task)
        
        # Wait for all updates to complete
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify performance
        successful_updates = sum(1 for r in results if not isinstance(r, Exception))
        throughput = successful_updates / duration
        
        assert successful_updates >= num_concurrent_updates * 0.95  # 95% success rate
        assert throughput >= 100  # At least 100 updates per second
        
        logger.info(f"Processed {successful_updates}/{num_concurrent_updates} updates")
        logger.info(f"Throughput: {throughput:.2f} updates/second")
    
    @async_test
    async def test_massive_order_processing(self, execution_engine, mock_exchange):
        """Test order processing under extreme load"""
        num_orders = 5000
        orders = []
        
        # Generate massive order batch
        for i in range(num_orders):
            order = {
                'id': f'order_{i}',
                'symbol': 'BTC/USDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': 50000 + random.uniform(-500, 500),
                'quantity': random.uniform(0.001, 0.1),
                'type': 'limit'
            }
            orders.append(order)
        
        # Process orders concurrently
        start_time = time.time()
        
        async def process_order_batch(batch):
            tasks = [execution_engine.submit_order(order) for order in batch]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Split into batches for processing
        batch_size = 100
        batches = [orders[i:i+batch_size] for i in range(0, len(orders), batch_size)]
        
        results = []
        for batch in batches:
            batch_results = await process_order_batch(batch)
            results.extend(batch_results)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful_orders = sum(1 for r in results if not isinstance(r, Exception))
        orders_per_second = successful_orders / duration
        
        assert successful_orders >= num_orders * 0.90  # 90% success rate
        assert orders_per_second >= 500  # At least 500 orders/second
        
        logger.info(f"Processed {successful_orders}/{num_orders} orders")
        logger.info(f"Rate: {orders_per_second:.2f} orders/second")
    
    @async_test
    async def test_memory_under_sustained_load(self, grid_attention_system):
        """Test memory usage under sustained high load"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]
        duration_minutes = 2
        
        async def sustained_load():
            """Generate sustained load for specified duration"""
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            operations_count = 0
            
            while datetime.now() < end_time:
                # Rotate through different operations
                operation = operations_count % 4
                
                if operation == 0:
                    # Market update
                    await grid_attention_system.process_market_update({
                        'price': 50000 + random.uniform(-100, 100),
                        'volume': random.uniform(1, 100)
                    })
                elif operation == 1:
                    # Strategy selection
                    await grid_attention_system.grid_strategy_selector.select_strategy()
                elif operation == 2:
                    # Risk calculation
                    await grid_attention_system.risk_management.calculate_position_risk()
                else:
                    # Performance update
                    await grid_attention_system.performance_monitor.update_metrics()
                
                operations_count += 1
                
                # Sample memory every 100 operations
                if operations_count % 100 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    
                    # Force garbage collection periodically
                    if operations_count % 1000 == 0:
                        gc.collect()
                
                # Small delay to prevent CPU saturation
                await asyncio.sleep(0.001)
            
            return operations_count
        
        # Run sustained load test
        total_operations = await sustained_load()
        
        # Analyze memory usage
        peak_memory = max(memory_samples)
        average_memory = np.mean(memory_samples)
        memory_growth = peak_memory - initial_memory
        
        # Memory should not grow indefinitely
        assert memory_growth < 500  # Less than 500MB growth
        assert peak_memory < initial_memory * 3  # Peak less than 3x initial
        
        logger.info(f"Total operations: {total_operations}")
        logger.info(f"Initial memory: {initial_memory:.2f} MB")
        logger.info(f"Peak memory: {peak_memory:.2f} MB")
        logger.info(f"Memory growth: {memory_growth:.2f} MB")
    
    @async_test
    async def test_cpu_bound_operations(self, attention_learning_layer):
        """Test CPU-intensive operations under load"""
        num_calculations = 1000
        matrix_size = 100
        
        async def cpu_intensive_task(task_id):
            """Perform CPU-intensive attention calculations"""
            # Generate random market data
            data = np.random.randn(matrix_size, matrix_size)
            
            # Perform attention calculations
            start = time.time()
            result = await attention_learning_layer.calculate_attention_weights(data)
            duration = time.time() - start
            
            return task_id, duration, result
        
        # Run CPU-bound tasks concurrently
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            loop = asyncio.get_event_loop()
            
            tasks = []
            for i in range(num_calculations):
                task = loop.run_in_executor(
                    executor,
                    asyncio.run,
                    cpu_intensive_task(i)
                )
                tasks.append(task)
            
            # Measure overall performance
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_duration = time.time() - start_time
            
        # Analyze performance
        individual_durations = [r[1] for r in results]
        avg_duration = np.mean(individual_durations)
        calculations_per_second = num_calculations / total_duration
        
        assert avg_duration < 0.1  # Each calculation under 100ms
        assert calculations_per_second > 50  # At least 50 calculations/second
        
        logger.info(f"Completed {num_calculations} calculations")
        logger.info(f"Average duration: {avg_duration*1000:.2f} ms")
        logger.info(f"Rate: {calculations_per_second:.2f} calculations/second")


class TestNetworkStress:
    """Test system behavior under network stress"""
    
    @pytest.fixture
    def network_simulator(self):
        """Create network simulator for stress testing"""
        return NetworkStressSimulator()
    
    @async_test
    async def test_high_latency_conditions(self, execution_engine, network_simulator):
        """Test system with high network latency"""
        # Configure high latency
        network_simulator.set_latency(500, 1000)  # 500-1000ms latency
        
        with patch.object(execution_engine, '_send_order', 
                         side_effect=network_simulator.delayed_response):
            
            # Submit multiple orders
            orders = []
            for i in range(10):
                order = execution_engine.submit_order({
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'price': 50000,
                    'quantity': 0.1
                })
                orders.append(order)
            
            # Wait for orders with timeout
            start_time = time.time()
            results = await asyncio.gather(*orders, return_exceptions=True)
            duration = time.time() - start_time
            
            # Verify system handles high latency gracefully
            successful = sum(1 for r in results if not isinstance(r, Exception))
            assert successful >= len(orders) * 0.8  # 80% success rate
            assert duration < 15  # Should complete within 15 seconds
    
    @async_test
    async def test_connection_drops(self, grid_attention_system, network_simulator):
        """Test system resilience to connection drops"""
        drop_probability = 0.3  # 30% chance of connection drop
        
        async def unreliable_update(data):
            """Simulate unreliable network connection"""
            if random.random() < drop_probability:
                raise ConnectionError("Network connection lost")
            return await grid_attention_system.process_market_update(data)
        
        # Send updates through unreliable connection
        num_updates = 100
        results = []
        
        for i in range(num_updates):
            try:
                result = await unreliable_update({
                    'price': 50000 + random.uniform(-100, 100),
                    'volume': random.uniform(1, 10)
                })
                results.append(('success', result))
            except ConnectionError:
                results.append(('failed', None))
        
        # Analyze results
        successful = sum(1 for r in results if r[0] == 'success')
        failed = sum(1 for r in results if r[0] == 'failed')
        
        # System should handle failures gracefully
        assert successful > 0
        assert failed < num_updates
        assert successful + failed == num_updates
        
        logger.info(f"Network drops: {failed}/{num_updates} ({failed/num_updates*100:.1f}%)")
    
    @async_test
    async def test_websocket_flood(self, mock_exchange):
        """Test WebSocket handling under message flood"""
        num_messages = 10000
        messages_per_second = 1000
        
        async def websocket_flood():
            """Simulate WebSocket message flood"""
            messages_sent = 0
            start_time = time.time()
            
            while messages_sent < num_messages:
                # Generate market data message
                message = {
                    'type': 'ticker',
                    'symbol': 'BTC/USDT',
                    'bid': 50000 + random.uniform(-10, 10),
                    'ask': 50000 + random.uniform(-10, 10),
                    'last': 50000 + random.uniform(-10, 10),
                    'volume': random.uniform(100, 1000),
                    'timestamp': time.time()
                }
                
                # Send through mock WebSocket
                await mock_exchange.websocket_handler(message)
                messages_sent += 1
                
                # Rate limiting
                elapsed = time.time() - start_time
                expected_messages = elapsed * messages_per_second
                if messages_sent > expected_messages:
                    await asyncio.sleep(0.001)
            
            return messages_sent
        
        # Run flood test
        total_sent = await websocket_flood()
        
        # Verify message handling
        processed = mock_exchange.get_processed_message_count()
        drop_rate = (total_sent - processed) / total_sent
        
        assert processed >= total_sent * 0.95  # 95% processing rate
        assert drop_rate < 0.05  # Less than 5% message drop
        
        logger.info(f"WebSocket flood: {processed}/{total_sent} messages processed")


class TestResourceExhaustion:
    """Test system behavior when resources are exhausted"""
    
    @async_test
    async def test_thread_pool_exhaustion(self, grid_attention_system):
        """Test behavior when thread pool is exhausted"""
        max_workers = 10
        num_tasks = 100
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            async def blocking_task(task_id):
                """Simulate blocking I/O operation"""
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    executor,
                    time.sleep,
                    random.uniform(0.1, 0.5)
                )
                return task_id
            
            # Submit more tasks than thread pool capacity
            start_time = time.time()
            tasks = [blocking_task(i) for i in range(num_tasks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            # Verify graceful handling
            successful = sum(1 for r in results if not isinstance(r, Exception))
            assert successful == num_tasks  # All tasks should complete
            assert duration < num_tasks * 0.5  # Should use parallelism
            
            logger.info(f"Thread pool exhaustion test: {successful}/{num_tasks} completed")
    
    @async_test
    async def test_memory_pressure(self, grid_attention_system):
        """Test system under memory pressure"""
        # Create large data structures
        large_dataframes = []
        target_memory_mb = 1000  # Target 1GB memory usage
        
        try:
            while True:
                # Create large DataFrame
                df = pd.DataFrame(
                    np.random.randn(100000, 100),
                    columns=[f'col_{i}' for i in range(100)]
                )
                large_dataframes.append(df)
                
                # Check memory usage
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                if current_memory > target_memory_mb:
                    break
                
                # Try to process data under memory pressure
                await grid_attention_system.process_market_update({
                    'price': 50000,
                    'volume': 100
                })
            
            # System should still function under memory pressure
            result = await grid_attention_system.get_system_status()
            assert result['status'] == 'operational'
            
        finally:
            # Clean up
            large_dataframes.clear()
            gc.collect()
    
    @async_test 
    async def test_file_descriptor_limits(self, execution_engine):
        """Test behavior when file descriptors are exhausted"""
        num_connections = 100
        connections = []
        
        try:
            # Open many connections
            for i in range(num_connections):
                session = aiohttp.ClientSession()
                connections.append(session)
            
            # Try to execute orders with limited resources
            orders = []
            for i in range(10):
                order = await execution_engine.submit_order({
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'price': 50000,
                    'quantity': 0.1
                })
                orders.append(order)
            
            results = await asyncio.gather(*orders, return_exceptions=True)
            
            # Should handle resource limitations gracefully
            successful = sum(1 for r in results if not isinstance(r, Exception))
            assert successful > 0  # At least some orders should succeed
            
        finally:
            # Clean up connections
            for session in connections:
                await session.close()


class TestCascadingFailures:
    """Test system resilience to cascading failures"""
    
    @async_test
    async def test_component_failure_cascade(self, grid_attention_system):
        """Test how component failures affect the system"""
        # Simulate market regime detector failure
        with patch.object(
            grid_attention_system.market_regime_detector,
            'detect_regime',
            side_effect=Exception("Component failure")
        ):
            # System should continue operating
            result = await grid_attention_system.process_market_update({
                'price': 50000,
                'volume': 100
            })
            
            assert result is not None
            assert result.get('regime_detection') == 'failed'
            assert result.get('fallback_mode') is True
    
    @async_test
    async def test_feedback_loop_overload(self, feedback_loop):
        """Test feedback loop under extreme conditions"""
        num_feedback_items = 10000
        
        # Generate massive feedback
        feedback_tasks = []
        for i in range(num_feedback_items):
            feedback = {
                'type': 'trade_result',
                'outcome': 'profit' if random.random() > 0.5 else 'loss',
                'amount': random.uniform(-100, 100),
                'timestamp': time.time()
            }
            task = feedback_loop.process_feedback(feedback)
            feedback_tasks.append(task)
        
        # Process all feedback
        start_time = time.time()
        results = await asyncio.gather(*feedback_tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Verify processing
        successful = sum(1 for r in results if not isinstance(r, Exception))
        processing_rate = successful / duration
        
        assert successful >= num_feedback_items * 0.9  # 90% success rate
        assert processing_rate > 100  # Process > 100 items/second
        
        logger.info(f"Feedback processing rate: {processing_rate:.2f} items/second")


class TestPerformanceDegradation:
    """Test graceful performance degradation"""
    
    @async_test
    async def test_gradual_load_increase(self, grid_attention_system):
        """Test system performance as load gradually increases"""
        load_levels = [10, 50, 100, 500, 1000]  # Operations per second
        performance_metrics = []
        
        for target_ops_per_sec in load_levels:
            # Run load test at this level
            duration_seconds = 5
            delay = 1.0 / target_ops_per_sec
            
            operations = []
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                op_start = time.time()
                
                await grid_attention_system.process_market_update({
                    'price': 50000 + random.uniform(-100, 100),
                    'volume': random.uniform(1, 100)
                })
                
                op_duration = time.time() - op_start
                operations.append(op_duration)
                
                # Rate limiting
                await asyncio.sleep(max(0, delay - op_duration))
            
            # Calculate metrics
            metrics = {
                'target_ops': target_ops_per_sec,
                'avg_latency': np.mean(operations) * 1000,  # ms
                'p95_latency': np.percentile(operations, 95) * 1000,
                'p99_latency': np.percentile(operations, 99) * 1000
            }
            performance_metrics.append(metrics)
            
            logger.info(f"Load {target_ops_per_sec} ops/s: "
                       f"avg={metrics['avg_latency']:.2f}ms, "
                       f"p95={metrics['p95_latency']:.2f}ms")
        
        # Verify graceful degradation
        latencies = [m['avg_latency'] for m in performance_metrics]
        assert latencies == sorted(latencies)  # Latency increases with load
        assert all(m['p99_latency'] < 1000 for m in performance_metrics)  # All under 1s


class NetworkStressSimulator:
    """Simulate various network stress conditions"""
    
    def __init__(self):
        self.min_latency = 0
        self.max_latency = 0
        
    def set_latency(self, min_ms: int, max_ms: int):
        """Set latency range in milliseconds"""
        self.min_latency = min_ms / 1000
        self.max_latency = max_ms / 1000
    
    async def delayed_response(self, *args, **kwargs):
        """Simulate delayed network response"""
        delay = random.uniform(self.min_latency, self.max_latency)
        await asyncio.sleep(delay)
        return {'status': 'success', 'latency': delay}


class StrategyMonitor:
    """Monitor strategy performance during stress tests"""
    
    def __init__(self, config):
        self.config = config
        self.metrics = []
        
    async def start_monitoring(self, strategy_type: str):
        """Start monitoring a strategy"""
        self.current_strategy = strategy_type
        self.start_time = time.time()
        
    async def record_trade(self, trade: Dict):
        """Record trade for monitoring"""
        self.metrics.append({
            'timestamp': time.time(),
            'trade': trade
        })
        
    async def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        if not self.metrics:
            return {}
            
        trades = [m['trade'] for m in self.metrics]
        profits = [t.get('profit', 0) for t in trades]
        
        return {
            'total_trades': len(trades),
            'win_rate': sum(1 for p in profits if p > 0) / len(profits),
            'average_profit': np.mean(profits),
            'sharpe_ratio': self._calculate_sharpe_ratio(profits)
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        return np.mean(returns) / (np.std(returns) + 1e-8)
    
    async def set_adaptation_thresholds(self, thresholds: Dict):
        """Set thresholds for strategy adaptation"""
        self.thresholds = thresholds
        
    async def should_adapt_strategy(self) -> bool:
        """Check if strategy should be adapted"""
        metrics = await self.get_current_metrics()
        
        if metrics.get('win_rate', 1) < self.thresholds.get('min_win_rate', 0):
            return True
        if metrics.get('sharpe_ratio', 1) < self.thresholds.get('min_sharpe_ratio', 0):
            return True
            
        return False
    
    async def get_adaptation_reason(self) -> str:
        """Get reason for adaptation"""
        metrics = await self.get_current_metrics()
        reasons = []
        
        if metrics.get('win_rate', 1) < self.thresholds.get('min_win_rate', 0):
            reasons.append('win_rate')
        if metrics.get('sharpe_ratio', 1) < self.thresholds.get('min_sharpe_ratio', 0):
            reasons.append('sharpe_ratio')
            
        return ' and '.join(reasons)