"""
Throughput Testing Suite for GridAttention Trading System
Tests processing capacity, rate limits, and system throughput
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import time
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, AsyncMock, patch
import logging
import multiprocessing
from collections import defaultdict, deque
import psutil
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
    ThroughputMeasurer
)
from tests.mocks.mock_exchange import MockExchange
from tests.fixtures.market_data import generate_market_data, generate_order_book

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThroughputMetrics:
    """Container for throughput measurements"""
    operations_per_second: float
    total_operations: int
    duration_seconds: float
    success_rate: float
    avg_cpu_usage: float
    avg_memory_usage_mb: float
    peak_operations_per_second: float
    min_operations_per_second: float


class TestMarketDataThroughput:
    """Test market data processing throughput"""
    
    @pytest.fixture
    def throughput_measurer(self):
        """Create throughput measurement tool"""
        return ThroughputMeasurer()
    
    @pytest.fixture
    def grid_attention_system(self):
        """Create GridAttention system optimized for throughput"""
        config = create_test_config()
        config['high_throughput_mode'] = True
        config['batch_processing'] = True
        config['async_workers'] = multiprocessing.cpu_count()
        return GridAttentionLayer(config)
    
    @async_test
    async def test_market_update_throughput(self, grid_attention_system, throughput_measurer):
        """Test maximum market update processing throughput"""
        duration_seconds = 10
        updates_processed = 0
        errors = 0
        
        # Start measurement
        throughput_measurer.start()
        
        async def process_updates():
            nonlocal updates_processed, errors
            
            while throughput_measurer.elapsed_time() < duration_seconds:
                try:
                    # Generate market update
                    update = {
                        'symbol': 'BTC/USDT',
                        'price': 50000 + np.random.uniform(-100, 100),
                        'volume': np.random.uniform(10, 1000),
                        'timestamp': time.time()
                    }
                    
                    # Process update
                    await grid_attention_system.process_market_update(update)
                    updates_processed += 1
                    
                    # Record successful operation
                    throughput_measurer.record_operation()
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"Update processing error: {e}")
        
        # Run with multiple concurrent workers
        workers = 4
        tasks = [process_updates() for _ in range(workers)]
        await asyncio.gather(*tasks)
        
        # Calculate metrics
        metrics = throughput_measurer.get_metrics()
        metrics.success_rate = updates_processed / (updates_processed + errors) if updates_processed > 0 else 0
        
        # Log results
        logger.info(f"\nMarket Update Throughput:")
        logger.info(f"  Total updates: {updates_processed:,}")
        logger.info(f"  Duration: {metrics.duration_seconds:.2f}s")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} updates/sec")
        logger.info(f"  Success rate: {metrics.success_rate:.2%}")
        logger.info(f"  Errors: {errors}")
        
        # Verify throughput meets requirements
        assert metrics.operations_per_second >= 1000, f"Throughput too low: {metrics.operations_per_second}"
        assert metrics.success_rate >= 0.99, f"Success rate too low: {metrics.success_rate}"
        
        return metrics
    
    @async_test
    async def test_order_book_throughput(self, grid_attention_system, throughput_measurer):
        """Test order book update processing throughput"""
        duration_seconds = 5
        book_updates = 0
        
        throughput_measurer.start()
        
        async def process_order_books():
            nonlocal book_updates
            
            while throughput_measurer.elapsed_time() < duration_seconds:
                # Generate order book update
                order_book = generate_order_book(
                    symbol='BTC/USDT',
                    mid_price=50000,
                    levels=10
                )
                
                # Process order book
                await grid_attention_system.process_order_book(order_book)
                book_updates += 1
                throughput_measurer.record_operation()
        
        # Run concurrent processing
        tasks = [process_order_books() for _ in range(2)]
        await asyncio.gather(*tasks)
        
        metrics = throughput_measurer.get_metrics()
        
        logger.info(f"\nOrder Book Throughput:")
        logger.info(f"  Total updates: {book_updates:,}")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} books/sec")
        
        # Order book processing is more complex, expect lower throughput
        assert metrics.operations_per_second >= 100, f"Order book throughput too low"
        
        return metrics
    
    @async_test
    async def test_multi_symbol_throughput(self, grid_attention_system, throughput_measurer):
        """Test throughput with multiple trading symbols"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
        duration_seconds = 10
        updates_by_symbol = defaultdict(int)
        
        throughput_measurer.start()
        
        async def process_symbol_updates(symbol: str):
            while throughput_measurer.elapsed_time() < duration_seconds:
                update = {
                    'symbol': symbol,
                    'price': np.random.uniform(100, 10000),
                    'volume': np.random.uniform(10, 1000),
                    'timestamp': time.time()
                }
                
                await grid_attention_system.process_market_update(update)
                updates_by_symbol[symbol] += 1
                throughput_measurer.record_operation()
        
        # Process all symbols concurrently
        tasks = [process_symbol_updates(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
        
        metrics = throughput_measurer.get_metrics()
        
        logger.info(f"\nMulti-Symbol Throughput:")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Total throughput: {metrics.operations_per_second:,.2f} updates/sec")
        for symbol, count in updates_by_symbol.items():
            logger.info(f"  {symbol}: {count:,} updates")
        
        # Should handle multiple symbols efficiently
        assert metrics.operations_per_second >= 500 * len(symbols)
        
        return metrics


class TestOrderProcessingThroughput:
    """Test order processing and execution throughput"""
    
    @async_test
    async def test_order_submission_throughput(self, execution_engine, mock_exchange):
        """Test order submission throughput"""
        measurer = ThroughputMeasurer()
        duration_seconds = 10
        orders_submitted = 0
        orders_filled = 0
        
        # Configure mock exchange for high throughput
        mock_exchange.set_fill_rate(0.95)  # 95% fill rate
        mock_exchange.set_latency(0.001, 0.005)  # 1-5ms latency
        
        measurer.start()
        
        async def submit_orders():
            nonlocal orders_submitted, orders_filled
            order_id = 0
            
            while measurer.elapsed_time() < duration_seconds:
                order = {
                    'id': f'perf_order_{order_id}',
                    'symbol': 'BTC/USDT',
                    'side': 'buy' if order_id % 2 == 0 else 'sell',
                    'price': 50000 + np.random.uniform(-100, 100),
                    'quantity': np.random.uniform(0.01, 1.0),
                    'type': 'limit'
                }
                
                result = await execution_engine.submit_order(order)
                orders_submitted += 1
                
                if result.get('status') == 'filled':
                    orders_filled += 1
                
                measurer.record_operation()
                order_id += 1
        
        # Run with multiple concurrent submitters
        tasks = [submit_orders() for _ in range(4)]
        await asyncio.gather(*tasks)
        
        metrics = measurer.get_metrics()
        fill_rate = orders_filled / orders_submitted if orders_submitted > 0 else 0
        
        logger.info(f"\nOrder Submission Throughput:")
        logger.info(f"  Orders submitted: {orders_submitted:,}")
        logger.info(f"  Orders filled: {orders_filled:,}")
        logger.info(f"  Fill rate: {fill_rate:.2%}")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} orders/sec")
        
        # Verify order processing throughput
        assert metrics.operations_per_second >= 500, "Order throughput too low"
        assert fill_rate >= 0.90, "Fill rate too low"
        
        return metrics
    
    @async_test
    async def test_order_modification_throughput(self, execution_engine):
        """Test order modification throughput"""
        measurer = ThroughputMeasurer()
        
        # First, create some orders
        order_ids = []
        for i in range(100):
            order = await execution_engine.submit_order({
                'id': f'mod_order_{i}',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 49000,
                'quantity': 0.1,
                'type': 'limit'
            })
            order_ids.append(order['id'])
        
        # Test modification throughput
        modifications = 0
        measurer.start()
        
        while measurer.elapsed_time() < 5:
            order_id = order_ids[modifications % len(order_ids)]
            
            # Modify order
            await execution_engine.modify_order(
                order_id=order_id,
                new_price=49000 + np.random.uniform(-100, 100),
                new_quantity=np.random.uniform(0.05, 0.2)
            )
            
            modifications += 1
            measurer.record_operation()
        
        metrics = measurer.get_metrics()
        
        logger.info(f"\nOrder Modification Throughput:")
        logger.info(f"  Modifications: {modifications:,}")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} mods/sec")
        
        assert metrics.operations_per_second >= 200, "Modification throughput too low"
        
        return metrics
    
    @async_test
    async def test_order_cancellation_throughput(self, execution_engine):
        """Test order cancellation throughput"""
        measurer = ThroughputMeasurer()
        
        # Create orders for cancellation
        order_ids = []
        for i in range(500):
            order = await execution_engine.submit_order({
                'id': f'cancel_order_{i}',
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'price': 51000,
                'quantity': 0.1,
                'type': 'limit'
            })
            order_ids.append(order['id'])
        
        # Test cancellation throughput
        cancellations = 0
        measurer.start()
        
        for order_id in order_ids:
            await execution_engine.cancel_order(order_id)
            cancellations += 1
            measurer.record_operation()
            
            if measurer.elapsed_time() > 5:
                break
        
        metrics = measurer.get_metrics()
        
        logger.info(f"\nOrder Cancellation Throughput:")
        logger.info(f"  Cancellations: {cancellations:,}")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} cancels/sec")
        
        assert metrics.operations_per_second >= 300, "Cancellation throughput too low"
        
        return metrics


class TestAnalyticsThroughput:
    """Test analytics and calculation throughput"""
    
    @async_test
    async def test_risk_calculation_throughput(self, risk_management_system):
        """Test risk calculation throughput"""
        measurer = ThroughputMeasurer()
        
        # Setup portfolio
        positions = [
            {'symbol': 'BTC/USDT', 'size': 1.0, 'entry_price': 50000},
            {'symbol': 'ETH/USDT', 'size': 10.0, 'entry_price': 3000},
            {'symbol': 'BNB/USDT', 'size': 20.0, 'entry_price': 400}
        ]
        
        calculations = 0
        measurer.start()
        
        while measurer.elapsed_time() < 5:
            # Update market prices
            market_prices = {
                'BTC/USDT': 50000 + np.random.uniform(-500, 500),
                'ETH/USDT': 3000 + np.random.uniform(-50, 50),
                'BNB/USDT': 400 + np.random.uniform(-10, 10)
            }
            
            # Calculate risk
            await risk_management_system.calculate_portfolio_risk(
                positions=positions,
                market_prices=market_prices
            )
            
            calculations += 1
            measurer.record_operation()
        
        metrics = measurer.get_metrics()
        
        logger.info(f"\nRisk Calculation Throughput:")
        logger.info(f"  Calculations: {calculations:,}")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} calcs/sec")
        
        assert metrics.operations_per_second >= 1000, "Risk calculation throughput too low"
        
        return metrics
    
    @async_test
    async def test_attention_calculation_throughput(self, attention_learning_layer):
        """Test attention weight calculation throughput"""
        measurer = ThroughputMeasurer()
        matrix_size = 50
        
        calculations = 0
        measurer.start()
        
        while measurer.elapsed_time() < 5:
            # Generate market data matrix
            data = np.random.randn(matrix_size, matrix_size)
            
            # Calculate attention weights
            await attention_learning_layer.calculate_attention_weights(data)
            
            calculations += 1
            measurer.record_operation()
        
        metrics = measurer.get_metrics()
        
        logger.info(f"\nAttention Calculation Throughput:")
        logger.info(f"  Calculations: {calculations:,}")
        logger.info(f"  Matrix size: {matrix_size}x{matrix_size}")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} calcs/sec")
        
        assert metrics.operations_per_second >= 100, "Attention calculation throughput too low"
        
        return metrics
    
    @async_test
    async def test_strategy_evaluation_throughput(self, grid_strategy_selector):
        """Test strategy evaluation throughput"""
        measurer = ThroughputMeasurer()
        
        # Prepare market scenarios
        scenarios = [
            {'volatility': 0.02, 'trend': 'up', 'volume': 'high'},
            {'volatility': 0.05, 'trend': 'down', 'volume': 'low'},
            {'volatility': 0.01, 'trend': 'neutral', 'volume': 'medium'},
            {'volatility': 0.10, 'trend': 'volatile', 'volume': 'high'}
        ]
        
        evaluations = 0
        measurer.start()
        
        while measurer.elapsed_time() < 5:
            scenario = scenarios[evaluations % len(scenarios)]
            
            # Evaluate strategy
            await grid_strategy_selector.evaluate_strategies(
                market_conditions=scenario,
                current_performance={'sharpe': 1.5, 'win_rate': 0.6}
            )
            
            evaluations += 1
            measurer.record_operation()
        
        metrics = measurer.get_metrics()
        
        logger.info(f"\nStrategy Evaluation Throughput:")
        logger.info(f"  Evaluations: {evaluations:,}")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} evals/sec")
        
        assert metrics.operations_per_second >= 500, "Strategy evaluation throughput too low"
        
        return metrics


class TestSystemThroughput:
    """Test overall system throughput under various conditions"""
    
    @async_test
    async def test_mixed_workload_throughput(self, grid_attention_system):
        """Test throughput with mixed operations"""
        measurer = ThroughputMeasurer()
        duration_seconds = 10
        
        operation_counts = defaultdict(int)
        
        async def mixed_operations():
            operation_id = 0
            
            while measurer.elapsed_time() < duration_seconds:
                operation_type = operation_id % 4
                
                try:
                    if operation_type == 0:
                        # Market update
                        await grid_attention_system.process_market_update({
                            'price': 50000 + np.random.uniform(-100, 100),
                            'volume': np.random.uniform(10, 100)
                        })
                        operation_counts['market_update'] += 1
                        
                    elif operation_type == 1:
                        # Strategy selection
                        await grid_attention_system.grid_strategy_selector.select_strategy()
                        operation_counts['strategy_select'] += 1
                        
                    elif operation_type == 2:
                        # Risk calculation
                        await grid_attention_system.risk_management.calculate_position_risk()
                        operation_counts['risk_calc'] += 1
                        
                    else:
                        # Performance update
                        await grid_attention_system.performance_monitor.update_metrics()
                        operation_counts['perf_update'] += 1
                    
                    measurer.record_operation()
                    
                except Exception as e:
                    logger.error(f"Operation error: {e}")
                
                operation_id += 1
        
        # Start measurement
        measurer.start()
        
        # Run with multiple workers
        tasks = [mixed_operations() for _ in range(4)]
        await asyncio.gather(*tasks)
        
        metrics = measurer.get_metrics()
        
        logger.info(f"\nMixed Workload Throughput:")
        logger.info(f"  Total operations: {sum(operation_counts.values()):,}")
        logger.info(f"  Overall throughput: {metrics.operations_per_second:,.2f} ops/sec")
        for op_type, count in operation_counts.items():
            ops_per_sec = count / metrics.duration_seconds
            logger.info(f"  {op_type}: {count:,} ({ops_per_sec:.2f} ops/sec)")
        
        assert metrics.operations_per_second >= 2000, "Mixed workload throughput too low"
        
        return metrics
    
    @async_test
    async def test_burst_throughput(self, grid_attention_system):
        """Test system's ability to handle burst traffic"""
        measurer = ThroughputMeasurer()
        
        # Test parameters
        burst_duration = 2  # seconds
        normal_rate = 100  # ops/sec
        burst_rate = 1000  # ops/sec
        total_duration = 10  # seconds
        
        operations = 0
        burst_operations = 0
        
        measurer.start()
        
        while measurer.elapsed_time() < total_duration:
            # Determine if in burst period
            elapsed = measurer.elapsed_time()
            in_burst = 2 < elapsed < 2 + burst_duration or 6 < elapsed < 6 + burst_duration
            
            target_rate = burst_rate if in_burst else normal_rate
            
            # Generate operations at target rate
            start_time = time.time()
            batch_operations = 0
            
            while time.time() - start_time < 0.1:  # 100ms batches
                await grid_attention_system.process_market_update({
                    'price': 50000 + np.random.uniform(-10, 10),
                    'volume': np.random.uniform(10, 100)
                })
                
                operations += 1
                batch_operations += 1
                if in_burst:
                    burst_operations += 1
                
                measurer.record_operation()
                
                # Rate limiting
                if batch_operations >= target_rate * 0.1:
                    break
            
            # Sleep remainder of 100ms window
            sleep_time = 0.1 - (time.time() - start_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        metrics = measurer.get_metrics()
        burst_throughput = burst_operations / (burst_duration * 2)  # 2 burst periods
        
        logger.info(f"\nBurst Throughput Test:")
        logger.info(f"  Total operations: {operations:,}")
        logger.info(f"  Average throughput: {metrics.operations_per_second:,.2f} ops/sec")
        logger.info(f"  Burst throughput: {burst_throughput:,.2f} ops/sec")
        logger.info(f"  Burst handling: {'PASSED' if burst_throughput >= burst_rate * 0.9 else 'FAILED'}")
        
        assert burst_throughput >= burst_rate * 0.9, "System cannot handle burst traffic"
        
        return metrics
    
    @async_test
    async def test_sustained_throughput(self, grid_attention_system):
        """Test sustained throughput over extended period"""
        measurer = ThroughputMeasurer()
        duration_seconds = 30
        
        # Track throughput over time
        throughput_samples = []
        sample_interval = 1  # seconds
        last_sample_time = 0
        last_sample_ops = 0
        
        measurer.start()
        
        async def sustained_load():
            while measurer.elapsed_time() < duration_seconds:
                # Process market update
                await grid_attention_system.process_market_update({
                    'price': 50000 + np.random.uniform(-50, 50),
                    'volume': np.random.uniform(10, 100)
                })
                
                measurer.record_operation()
                
                # Sample throughput periodically
                current_time = measurer.elapsed_time()
                if current_time - last_sample_time >= sample_interval:
                    current_ops = measurer.total_operations
                    sample_throughput = (current_ops - last_sample_ops) / sample_interval
                    throughput_samples.append(sample_throughput)
                    
                    last_sample_time = current_time
                    last_sample_ops = current_ops
        
        # Run with multiple workers
        tasks = [sustained_load() for _ in range(4)]
        await asyncio.gather(*tasks)
        
        metrics = measurer.get_metrics()
        
        # Analyze throughput stability
        if throughput_samples:
            avg_throughput = statistics.mean(throughput_samples)
            std_throughput = statistics.stdev(throughput_samples)
            cv = std_throughput / avg_throughput  # Coefficient of variation
            
            logger.info(f"\nSustained Throughput Test:")
            logger.info(f"  Duration: {duration_seconds}s")
            logger.info(f"  Total operations: {metrics.total_operations:,}")
            logger.info(f"  Average throughput: {avg_throughput:,.2f} ops/sec")
            logger.info(f"  Throughput std dev: {std_throughput:.2f}")
            logger.info(f"  Coefficient of variation: {cv:.2%}")
            logger.info(f"  CPU usage: {metrics.avg_cpu_usage:.1f}%")
            logger.info(f"  Memory usage: {metrics.avg_memory_usage_mb:.1f} MB")
            
            # Throughput should be stable (low variation)
            assert cv < 0.15, f"Throughput too unstable: CV={cv:.2%}"
            assert avg_throughput >= 1000, f"Sustained throughput too low: {avg_throughput}"
        
        return metrics


class TestThroughputOptimization:
    """Test throughput optimization techniques"""
    
    @async_test
    async def test_batch_processing_optimization(self, grid_attention_system):
        """Test throughput improvement with batch processing"""
        measurer_single = ThroughputMeasurer()
        measurer_batch = ThroughputMeasurer()
        
        duration = 5
        
        # Test single processing
        grid_attention_system.disable_batch_processing()
        measurer_single.start()
        
        while measurer_single.elapsed_time() < duration:
            await grid_attention_system.process_market_update({
                'price': 50000,
                'volume': 100
            })
            measurer_single.record_operation()
        
        single_metrics = measurer_single.get_metrics()
        
        # Test batch processing
        grid_attention_system.enable_batch_processing(batch_size=100)
        measurer_batch.start()
        
        while measurer_batch.elapsed_time() < duration:
            # Create batch
            batch = []
            for _ in range(100):
                batch.append({
                    'price': 50000 + np.random.uniform(-10, 10),
                    'volume': np.random.uniform(10, 100)
                })
            
            # Process batch
            await grid_attention_system.process_market_batch(batch)
            measurer_batch.record_operations(100)
        
        batch_metrics = measurer_batch.get_metrics()
        
        # Calculate improvement
        improvement = batch_metrics.operations_per_second / single_metrics.operations_per_second
        
        logger.info(f"\nBatch Processing Optimization:")
        logger.info(f"  Single: {single_metrics.operations_per_second:,.2f} ops/sec")
        logger.info(f"  Batch: {batch_metrics.operations_per_second:,.2f} ops/sec")
        logger.info(f"  Improvement: {improvement:.2f}x")
        
        assert improvement >= 2.0, "Batch processing should at least double throughput"
        
        return improvement
    
    @async_test
    async def test_pipeline_optimization(self, grid_attention_system):
        """Test throughput with pipeline optimization"""
        measurer = ThroughputMeasurer()
        
        # Enable pipeline processing
        grid_attention_system.enable_pipeline_mode(stages=4)
        
        operations = 0
        measurer.start()
        
        # Create pipeline stages
        async def pipeline_processor():
            nonlocal operations
            
            while measurer.elapsed_time() < 5:
                # Stage 1: Data ingestion
                data = {
                    'price': 50000 + np.random.uniform(-100, 100),
                    'volume': np.random.uniform(10, 100)
                }
                
                # Process through pipeline
                await grid_attention_system.pipeline_process(data)
                operations += 1
                measurer.record_operation()
        
        # Run multiple pipeline processors
        tasks = [pipeline_processor() for _ in range(4)]
        await asyncio.gather(*tasks)
        
        metrics = measurer.get_metrics()
        
        logger.info(f"\nPipeline Optimization:")
        logger.info(f"  Operations: {operations:,}")
        logger.info(f"  Throughput: {metrics.operations_per_second:,.2f} ops/sec")
        
        assert metrics.operations_per_second >= 3000, "Pipeline throughput too low"
        
        return metrics
    
    @async_test
    async def test_cache_impact_on_throughput(self, grid_attention_system):
        """Test throughput improvement with caching"""
        measurer_no_cache = ThroughputMeasurer()
        measurer_cache = ThroughputMeasurer()
        
        # Prepare test data with repeated patterns
        test_patterns = []
        for i in range(10):
            test_patterns.append({
                'price': 50000 + i * 100,
                'volume': 100 + i * 10,
                'pattern_id': i
            })
        
        # Test without cache
        grid_attention_system.disable_cache()
        measurer_no_cache.start()
        
        operations = 0
        while measurer_no_cache.elapsed_time() < 5:
            pattern = test_patterns[operations % len(test_patterns)]
            await grid_attention_system.process_market_update(pattern)
            operations += 1
            measurer_no_cache.record_operation()
        
        no_cache_metrics = measurer_no_cache.get_metrics()
        
        # Test with cache
        grid_attention_system.enable_cache()
        measurer_cache.start()
        
        operations = 0
        while measurer_cache.elapsed_time() < 5:
            pattern = test_patterns[operations % len(test_patterns)]
            await grid_attention_system.process_market_update(pattern)
            operations += 1
            measurer_cache.record_operation()
        
        cache_metrics = measurer_cache.get_metrics()
        
        # Calculate improvement
        improvement = cache_metrics.operations_per_second / no_cache_metrics.operations_per_second
        
        logger.info(f"\nCache Impact on Throughput:")
        logger.info(f"  Without cache: {no_cache_metrics.operations_per_second:,.2f} ops/sec")
        logger.info(f"  With cache: {cache_metrics.operations_per_second:,.2f} ops/sec")
        logger.info(f"  Improvement: {improvement:.2f}x")
        
        assert improvement >= 1.5, "Cache should improve throughput by at least 50%"
        
        return improvement


class TestThroughputMonitoring:
    """Test throughput monitoring and alerting"""
    
    @async_test
    async def test_real_time_throughput_monitoring(self):
        """Test real-time throughput monitoring"""
        monitor = ThroughputMonitor(
            window_size=10,  # 10 second window
            alert_threshold=100  # Alert if below 100 ops/sec
        )
        
        # Simulate varying throughput
        time_periods = [
            (5, 200),   # 5 seconds at 200 ops/sec
            (5, 50),    # 5 seconds at 50 ops/sec (should trigger alert)
            (5, 300),   # 5 seconds at 300 ops/sec
        ]
        
        alerts = []
        
        for duration, rate in time_periods:
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Simulate operations at specified rate
                ops_this_second = 0
                second_start = time.time()
                
                while ops_this_second < rate and time.time() - second_start < 1:
                    monitor.record_operation()
                    ops_this_second += 1
                    await asyncio.sleep(1 / rate)
                
                # Check for alerts
                if monitor.check_alert():
                    current_throughput = monitor.get_current_throughput()
                    alerts.append({
                        'time': time.time(),
                        'throughput': current_throughput,
                        'threshold': monitor.alert_threshold
                    })
                
                # Wait remainder of second
                sleep_time = 1 - (time.time() - second_start)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        # Verify alerts were triggered
        assert len(alerts) > 0, "No alerts triggered for low throughput"
        assert alerts[0]['throughput'] < 100, "Alert triggered incorrectly"
        
        logger.info(f"\nThroughput Monitoring:")
        logger.info(f"  Alerts triggered: {len(alerts)}")
        for alert in alerts:
            logger.info(f"  Alert: {alert['throughput']:.2f} ops/sec (threshold: {alert['threshold']})")


# Helper Classes

class ThroughputMonitor:
    """Monitor throughput and generate alerts"""
    
    def __init__(self, window_size: int = 60, alert_threshold: float = 100):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.operations = deque()
        self.last_alert_time = 0
        self.alert_cooldown = 30  # seconds
    
    def record_operation(self):
        """Record an operation"""
        current_time = time.time()
        self.operations.append(current_time)
        
        # Remove old operations outside window
        cutoff_time = current_time - self.window_size
        while self.operations and self.operations[0] < cutoff_time:
            self.operations.popleft()
    
    def get_current_throughput(self) -> float:
        """Get current throughput in operations per second"""
        if not self.operations:
            return 0
        
        time_span = self.operations[-1] - self.operations[0]
        if time_span == 0:
            return 0
        
        return len(self.operations) / time_span
    
    def check_alert(self) -> bool:
        """Check if alert should be triggered"""
        current_time = time.time()
        current_throughput = self.get_current_throughput()
        
        # Check if below threshold and cooldown has passed
        if (current_throughput < self.alert_threshold and 
            current_time - self.last_alert_time > self.alert_cooldown):
            self.last_alert_time = current_time
            return True
        
        return False