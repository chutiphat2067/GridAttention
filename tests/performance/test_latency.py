"""
Latency Testing Suite for GridAttention Trading System
Tests response times, processing delays, and end-to-end latency
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
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, AsyncMock, patch
import logging
import aiohttp
import websockets
from collections import defaultdict, deque

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
    measure_latency,
    LatencyProfiler
)
from tests.mocks.mock_exchange import MockExchange
from tests.fixtures.market_data import generate_market_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Container for latency measurements"""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    samples: int


class TestComponentLatency:
    """Test latency of individual components"""
    
    @pytest.fixture
    def latency_profiler(self):
        """Create latency profiler"""
        return LatencyProfiler()
    
    @pytest.fixture
    def grid_attention_system(self):
        """Create GridAttention system optimized for low latency"""
        config = create_test_config()
        config['low_latency_mode'] = True
        config['async_processing'] = True
        return GridAttentionLayer(config)
    
    @async_test
    async def test_market_update_latency(self, grid_attention_system, latency_profiler):
        """Test latency of market data updates"""
        num_samples = 1000
        latencies = []
        
        # Warm up
        for _ in range(100):
            await grid_attention_system.process_market_update({
                'price': 50000,
                'volume': 100
            })
        
        # Measure latency
        for i in range(num_samples):
            market_data = {
                'symbol': 'BTC/USDT',
                'price': 50000 + np.random.uniform(-100, 100),
                'volume': np.random.uniform(1, 100),
                'timestamp': time.time()
            }
            
            start_time = time.perf_counter()
            await grid_attention_system.process_market_update(market_data)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate metrics
        metrics = self._calculate_latency_metrics(latencies)
        
        # Verify latency requirements
        assert metrics.mean_ms < 10, f"Mean latency too high: {metrics.mean_ms:.2f}ms"
        assert metrics.p95_ms < 20, f"P95 latency too high: {metrics.p95_ms:.2f}ms"
        assert metrics.p99_ms < 50, f"P99 latency too high: {metrics.p99_ms:.2f}ms"
        
        self._log_latency_metrics("Market Update", metrics)
        return metrics
    
    @async_test
    async def test_order_execution_latency(self, execution_engine, mock_exchange):
        """Test latency of order execution"""
        num_orders = 500
        latencies = []
        
        # Configure mock exchange for low latency
        mock_exchange.set_latency(0.001, 0.005)  # 1-5ms simulated exchange latency
        
        for i in range(num_orders):
            order = {
                'id': f'order_{i}',
                'symbol': 'BTC/USDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': 50000 + np.random.uniform(-100, 100),
                'quantity': 0.1,
                'type': 'limit'
            }
            
            start_time = time.perf_counter()
            result = await execution_engine.submit_order(order)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        metrics = self._calculate_latency_metrics(latencies)
        
        # Order execution latency requirements
        assert metrics.mean_ms < 15, f"Mean order latency too high: {metrics.mean_ms:.2f}ms"
        assert metrics.p95_ms < 30, f"P95 order latency too high: {metrics.p95_ms:.2f}ms"
        
        self._log_latency_metrics("Order Execution", metrics)
        return metrics
    
    @async_test
    async def test_strategy_selection_latency(self, grid_strategy_selector):
        """Test latency of strategy selection"""
        num_selections = 100
        latencies = []
        
        # Prepare market conditions
        market_conditions = [
            {'volatility': 0.02, 'trend': 'up', 'volume': 'high'},
            {'volatility': 0.05, 'trend': 'down', 'volume': 'low'},
            {'volatility': 0.01, 'trend': 'neutral', 'volume': 'medium'},
            {'volatility': 0.10, 'trend': 'volatile', 'volume': 'high'}
        ]
        
        for i in range(num_selections):
            condition = market_conditions[i % len(market_conditions)]
            
            start_time = time.perf_counter()
            strategy = await grid_strategy_selector.select_strategy(
                market_volatility=condition['volatility'],
                market_trend=condition['trend'],
                volume_profile=condition['volume']
            )
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        metrics = self._calculate_latency_metrics(latencies)
        
        # Strategy selection should be fast
        assert metrics.mean_ms < 5, f"Strategy selection too slow: {metrics.mean_ms:.2f}ms"
        assert metrics.p99_ms < 10, f"P99 strategy selection too slow: {metrics.p99_ms:.2f}ms"
        
        self._log_latency_metrics("Strategy Selection", metrics)
        return metrics
    
    @async_test
    async def test_risk_calculation_latency(self, risk_management_system):
        """Test latency of risk calculations"""
        num_calculations = 200
        latencies = []
        
        # Setup portfolio positions
        positions = [
            {'symbol': 'BTC/USDT', 'size': 0.5, 'entry_price': 50000},
            {'symbol': 'ETH/USDT', 'size': 5.0, 'entry_price': 3000},
            {'symbol': 'BNB/USDT', 'size': 10.0, 'entry_price': 400}
        ]
        
        for i in range(num_calculations):
            # Update market prices
            market_prices = {
                'BTC/USDT': 50000 + np.random.uniform(-500, 500),
                'ETH/USDT': 3000 + np.random.uniform(-50, 50),
                'BNB/USDT': 400 + np.random.uniform(-10, 10)
            }
            
            start_time = time.perf_counter()
            risk_metrics = await risk_management_system.calculate_portfolio_risk(
                positions=positions,
                market_prices=market_prices
            )
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        metrics = self._calculate_latency_metrics(latencies)
        
        # Risk calculations should be efficient
        assert metrics.mean_ms < 8, f"Risk calculation too slow: {metrics.mean_ms:.2f}ms"
        assert metrics.p95_ms < 15, f"P95 risk calculation too slow: {metrics.p95_ms:.2f}ms"
        
        self._log_latency_metrics("Risk Calculation", metrics)
        return metrics
    
    def _calculate_latency_metrics(self, latencies: List[float]) -> LatencyMetrics:
        """Calculate comprehensive latency metrics"""
        sorted_latencies = sorted(latencies)
        
        return LatencyMetrics(
            min_ms=min(latencies),
            max_ms=max(latencies),
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p95_ms=sorted_latencies[int(len(latencies) * 0.95)],
            p99_ms=sorted_latencies[int(len(latencies) * 0.99)],
            std_dev_ms=statistics.stdev(latencies),
            samples=len(latencies)
        )
    
    def _log_latency_metrics(self, operation: str, metrics: LatencyMetrics):
        """Log latency metrics"""
        logger.info(f"\n{operation} Latency Metrics:")
        logger.info(f"  Mean: {metrics.mean_ms:.2f}ms")
        logger.info(f"  Median: {metrics.median_ms:.2f}ms")
        logger.info(f"  P95: {metrics.p95_ms:.2f}ms")
        logger.info(f"  P99: {metrics.p99_ms:.2f}ms")
        logger.info(f"  Min: {metrics.min_ms:.2f}ms")
        logger.info(f"  Max: {metrics.max_ms:.2f}ms")
        logger.info(f"  Std Dev: {metrics.std_dev_ms:.2f}ms")


class TestEndToEndLatency:
    """Test end-to-end system latency"""
    
    @async_test
    async def test_market_to_order_latency(self, grid_attention_system, mock_exchange):
        """Test latency from market update to order placement"""
        num_cycles = 100
        latencies = []
        
        # Configure system for automated trading
        await grid_attention_system.enable_auto_trading()
        
        for i in range(num_cycles):
            # Market event timestamp
            market_event_time = time.perf_counter()
            
            # Simulate significant price movement
            market_data = {
                'symbol': 'BTC/USDT',
                'price': 50000 + (100 if i % 2 == 0 else -100),  # Trigger trading
                'volume': 1000,
                'timestamp': market_event_time
            }
            
            # Track order placement
            order_placed_event = asyncio.Event()
            order_timestamp = None
            
            async def order_callback(order):
                nonlocal order_timestamp
                order_timestamp = time.perf_counter()
                order_placed_event.set()
            
            # Set callback
            mock_exchange.set_order_callback(order_callback)
            
            # Process market update
            await grid_attention_system.process_market_update(market_data)
            
            # Wait for order (with timeout)
            try:
                await asyncio.wait_for(order_placed_event.wait(), timeout=0.1)
                
                if order_timestamp:
                    latency_ms = (order_timestamp - market_event_time) * 1000
                    latencies.append(latency_ms)
            except asyncio.TimeoutError:
                logger.warning(f"No order placed for cycle {i}")
        
        if latencies:
            metrics = self._calculate_latency_metrics(latencies)
            
            # End-to-end latency requirements
            assert metrics.mean_ms < 50, f"E2E latency too high: {metrics.mean_ms:.2f}ms"
            assert metrics.p95_ms < 100, f"P95 E2E latency too high: {metrics.p95_ms:.2f}ms"
            
            logger.info(f"\nMarket-to-Order Latency: {metrics.mean_ms:.2f}ms (mean)")
            return metrics
    
    @async_test
    async def test_full_cycle_latency(self, grid_attention_system):
        """Test full trading cycle latency"""
        # Measure: Market Update → Analysis → Decision → Order → Confirmation
        
        cycle_latencies = []
        
        for i in range(50):
            timestamps = {}
            
            # 1. Market update received
            timestamps['market_update'] = time.perf_counter()
            
            market_data = {
                'symbol': 'BTC/USDT',
                'price': 50000 + np.random.uniform(-200, 200),
                'volume': np.random.uniform(100, 1000)
            }
            
            # Hook into processing stages
            async def track_stage(stage_name):
                timestamps[stage_name] = time.perf_counter()
            
            # Set stage callbacks
            grid_attention_system.set_stage_callback('regime_detected', 
                                                   lambda: track_stage('regime_detected'))
            grid_attention_system.set_stage_callback('strategy_selected',
                                                   lambda: track_stage('strategy_selected'))
            grid_attention_system.set_stage_callback('risk_assessed',
                                                   lambda: track_stage('risk_assessed'))
            grid_attention_system.set_stage_callback('order_placed',
                                                   lambda: track_stage('order_placed'))
            
            # Process update
            await grid_attention_system.process_market_update(market_data)
            
            # Calculate stage latencies
            if len(timestamps) > 1:
                stages = ['market_update', 'regime_detected', 'strategy_selected', 
                         'risk_assessed', 'order_placed']
                
                stage_latencies = {}
                for j in range(1, len(stages)):
                    if stages[j] in timestamps and stages[j-1] in timestamps:
                        latency = (timestamps[stages[j]] - timestamps[stages[j-1]]) * 1000
                        stage_latencies[f"{stages[j-1]}_to_{stages[j]}"] = latency
                
                if 'order_placed' in timestamps:
                    total_latency = (timestamps['order_placed'] - timestamps['market_update']) * 1000
                    cycle_latencies.append(total_latency)
                    
                    if i % 10 == 0:
                        logger.info(f"\nCycle {i} Stage Latencies:")
                        for stage, latency in stage_latencies.items():
                            logger.info(f"  {stage}: {latency:.2f}ms")
                        logger.info(f"  Total: {total_latency:.2f}ms")
        
        if cycle_latencies:
            metrics = self._calculate_latency_metrics(cycle_latencies)
            logger.info(f"\nFull Cycle Latency: {metrics.mean_ms:.2f}ms (mean)")
            
            # Full cycle should complete quickly
            assert metrics.mean_ms < 100, f"Full cycle too slow: {metrics.mean_ms:.2f}ms"
            
            return metrics
    
    def _calculate_latency_metrics(self, latencies: List[float]) -> LatencyMetrics:
        """Calculate latency metrics"""
        sorted_latencies = sorted(latencies)
        
        return LatencyMetrics(
            min_ms=min(latencies),
            max_ms=max(latencies),
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p95_ms=sorted_latencies[int(len(latencies) * 0.95)],
            p99_ms=sorted_latencies[int(len(latencies) * 0.99)],
            std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            samples=len(latencies)
        )


class TestLatencyUnderLoad:
    """Test latency behavior under various load conditions"""
    
    @async_test
    async def test_latency_vs_throughput(self, grid_attention_system):
        """Test how latency changes with increasing throughput"""
        throughput_levels = [10, 50, 100, 200, 500]  # Updates per second
        results = []
        
        for target_throughput in throughput_levels:
            latencies = []
            actual_throughput = 0
            duration = 5  # seconds
            
            start_time = time.time()
            update_count = 0
            
            while time.time() - start_time < duration:
                update_start = time.perf_counter()
                
                await grid_attention_system.process_market_update({
                    'price': 50000 + np.random.uniform(-10, 10),
                    'volume': np.random.uniform(10, 100)
                })
                
                update_end = time.perf_counter()
                latency_ms = (update_end - update_start) * 1000
                latencies.append(latency_ms)
                
                update_count += 1
                
                # Rate limiting
                expected_updates = (time.time() - start_time) * target_throughput
                if update_count > expected_updates:
                    sleep_time = (update_count - expected_updates) / target_throughput
                    await asyncio.sleep(sleep_time)
            
            actual_throughput = update_count / duration
            metrics = self._calculate_latency_metrics(latencies)
            
            results.append({
                'target_throughput': target_throughput,
                'actual_throughput': actual_throughput,
                'latency_metrics': metrics
            })
            
            logger.info(f"\nThroughput: {actual_throughput:.1f} updates/sec")
            logger.info(f"  Mean latency: {metrics.mean_ms:.2f}ms")
            logger.info(f"  P95 latency: {metrics.p95_ms:.2f}ms")
        
        # Verify latency doesn't degrade too much
        for result in results:
            if result['target_throughput'] <= 100:
                assert result['latency_metrics'].mean_ms < 20
            else:
                assert result['latency_metrics'].mean_ms < 50
        
        return results
    
    @async_test
    async def test_concurrent_request_latency(self, execution_engine):
        """Test latency with concurrent requests"""
        concurrency_levels = [1, 5, 10, 20, 50]
        results = []
        
        for concurrency in concurrency_levels:
            latencies = []
            
            async def submit_order_batch():
                batch_latencies = []
                
                # Submit concurrent orders
                tasks = []
                for i in range(concurrency):
                    order = {
                        'id': f'concurrent_order_{i}',
                        'symbol': 'BTC/USDT',
                        'side': 'buy' if i % 2 == 0 else 'sell',
                        'price': 50000,
                        'quantity': 0.1
                    }
                    
                    async def timed_order(order):
                        start = time.perf_counter()
                        await execution_engine.submit_order(order)
                        end = time.perf_counter()
                        return (end - start) * 1000
                    
                    tasks.append(timed_order(order))
                
                batch_results = await asyncio.gather(*tasks)
                batch_latencies.extend(batch_results)
                
                return batch_latencies
            
            # Run multiple batches
            for _ in range(10):
                batch_latencies = await submit_order_batch()
                latencies.extend(batch_latencies)
            
            metrics = self._calculate_latency_metrics(latencies)
            
            results.append({
                'concurrency': concurrency,
                'latency_metrics': metrics
            })
            
            logger.info(f"\nConcurrency: {concurrency}")
            logger.info(f"  Mean latency: {metrics.mean_ms:.2f}ms")
            logger.info(f"  P95 latency: {metrics.p95_ms:.2f}ms")
        
        # Latency should scale reasonably with concurrency
        for i in range(1, len(results)):
            latency_increase = results[i]['latency_metrics'].mean_ms / results[0]['latency_metrics'].mean_ms
            concurrency_increase = results[i]['concurrency'] / results[0]['concurrency']
            
            # Latency shouldn't increase linearly with concurrency
            assert latency_increase < concurrency_increase * 0.5
        
        return results
    
    def _calculate_latency_metrics(self, latencies: List[float]) -> LatencyMetrics:
        """Calculate latency metrics"""
        if not latencies:
            return None
            
        sorted_latencies = sorted(latencies)
        
        return LatencyMetrics(
            min_ms=min(latencies),
            max_ms=max(latencies),
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p95_ms=sorted_latencies[int(len(latencies) * 0.95)],
            p99_ms=sorted_latencies[int(len(latencies) * 0.99)],
            std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            samples=len(latencies)
        )


class TestLatencyOptimization:
    """Test latency optimization techniques"""
    
    @async_test
    async def test_caching_impact_on_latency(self, grid_attention_system):
        """Test how caching improves latency"""
        # Test without cache
        grid_attention_system.disable_cache()
        
        no_cache_latencies = []
        for i in range(100):
            start = time.perf_counter()
            await grid_attention_system.market_regime_detector.detect_regime(
                generate_market_data('BTC/USDT', periods=100)
            )
            end = time.perf_counter()
            no_cache_latencies.append((end - start) * 1000)
        
        # Test with cache
        grid_attention_system.enable_cache()
        
        cache_latencies = []
        for i in range(100):
            start = time.perf_counter()
            await grid_attention_system.market_regime_detector.detect_regime(
                generate_market_data('BTC/USDT', periods=100)
            )
            end = time.perf_counter()
            cache_latencies.append((end - start) * 1000)
        
        # Calculate improvement
        no_cache_mean = statistics.mean(no_cache_latencies)
        cache_mean = statistics.mean(cache_latencies)
        improvement = (no_cache_mean - cache_mean) / no_cache_mean * 100
        
        logger.info(f"\nCaching Impact:")
        logger.info(f"  Without cache: {no_cache_mean:.2f}ms")
        logger.info(f"  With cache: {cache_mean:.2f}ms")
        logger.info(f"  Improvement: {improvement:.1f}%")
        
        # Cache should improve latency
        assert cache_mean < no_cache_mean
        assert improvement > 20  # At least 20% improvement
    
    @async_test
    async def test_batch_processing_latency(self, execution_engine):
        """Test batch processing vs individual processing latency"""
        num_orders = 100
        
        # Individual processing
        individual_start = time.perf_counter()
        for i in range(num_orders):
            await execution_engine.submit_order({
                'id': f'individual_{i}',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 50000,
                'quantity': 0.1
            })
        individual_end = time.perf_counter()
        individual_total = (individual_end - individual_start) * 1000
        
        # Batch processing
        batch_orders = []
        for i in range(num_orders):
            batch_orders.append({
                'id': f'batch_{i}',
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'price': 50000,
                'quantity': 0.1
            })
        
        batch_start = time.perf_counter()
        await execution_engine.submit_batch(batch_orders)
        batch_end = time.perf_counter()
        batch_total = (batch_end - batch_start) * 1000
        
        # Calculate improvement
        improvement = (individual_total - batch_total) / individual_total * 100
        
        logger.info(f"\nBatch Processing:")
        logger.info(f"  Individual: {individual_total:.2f}ms total ({individual_total/num_orders:.2f}ms per order)")
        logger.info(f"  Batch: {batch_total:.2f}ms total ({batch_total/num_orders:.2f}ms per order)")
        logger.info(f"  Improvement: {improvement:.1f}%")
        
        # Batch processing should be faster
        assert batch_total < individual_total
        assert improvement > 50  # At least 50% improvement
    
    @async_test
    async def test_async_vs_sync_latency(self, grid_attention_system):
        """Compare async vs sync processing latency"""
        num_operations = 50
        
        # Async processing
        async_latencies = []
        for i in range(num_operations):
            start = time.perf_counter()
            
            # Run multiple operations concurrently
            await asyncio.gather(
                grid_attention_system.market_regime_detector.detect_regime(
                    generate_market_data('BTC/USDT', periods=50)
                ),
                grid_attention_system.grid_strategy_selector.select_strategy(),
                grid_attention_system.risk_management.calculate_position_risk()
            )
            
            end = time.perf_counter()
            async_latencies.append((end - start) * 1000)
        
        # Sync processing (simulated)
        sync_latencies = []
        for i in range(num_operations):
            start = time.perf_counter()
            
            # Run operations sequentially
            await grid_attention_system.market_regime_detector.detect_regime(
                generate_market_data('BTC/USDT', periods=50)
            )
            await grid_attention_system.grid_strategy_selector.select_strategy()
            await grid_attention_system.risk_management.calculate_position_risk()
            
            end = time.perf_counter()
            sync_latencies.append((end - start) * 1000)
        
        # Compare results
        async_mean = statistics.mean(async_latencies)
        sync_mean = statistics.mean(sync_latencies)
        improvement = (sync_mean - async_mean) / sync_mean * 100
        
        logger.info(f"\nAsync vs Sync:")
        logger.info(f"  Async: {async_mean:.2f}ms")
        logger.info(f"  Sync: {sync_mean:.2f}ms")
        logger.info(f"  Improvement: {improvement:.1f}%")
        
        # Async should be faster
        assert async_mean < sync_mean
        assert improvement > 30  # At least 30% improvement


class TestLatencyMonitoring:
    """Test latency monitoring and alerting"""
    
    @async_test
    async def test_latency_spike_detection(self):
        """Test detection of latency spikes"""
        monitor = LatencyMonitor(
            baseline_window=100,
            spike_threshold=3.0  # 3x baseline
        )
        
        # Normal operations
        for i in range(100):
            latency = np.random.normal(10, 2)  # 10ms ± 2ms
            spike_detected = monitor.record_latency('market_update', latency)
            assert not spike_detected
        
        # Inject latency spike
        spike_latency = 50  # 5x normal
        spike_detected = monitor.record_latency('market_update', spike_latency)
        assert spike_detected
        
        # Get spike report
        report = monitor.get_spike_report()
        assert len(report) == 1
        assert report[0]['operation'] == 'market_update'
        assert report[0]['latency'] == spike_latency
    
    @async_test
    async def test_latency_percentile_tracking(self):
        """Test real-time percentile tracking"""
        tracker = PercentileTracker(window_size=1000)
        
        # Generate latency distribution
        latencies = np.random.lognormal(2, 0.5, 1000)  # Log-normal distribution
        
        for latency in latencies:
            tracker.add(latency)
        
        # Check percentiles
        p50 = tracker.get_percentile(50)
        p95 = tracker.get_percentile(95)
        p99 = tracker.get_percentile(99)
        
        # Verify percentile relationships
        assert p50 < p95 < p99
        
        # Verify accuracy (within 5% of numpy calculation)
        np_p95 = np.percentile(latencies, 95)
        assert abs(p95 - np_p95) / np_p95 < 0.05
        
        logger.info(f"\nPercentile Tracking:")
        logger.info(f"  P50: {p50:.2f}ms")
        logger.info(f"  P95: {p95:.2f}ms")
        logger.info(f"  P99: {p99:.2f}ms")


# Helper Classes

class LatencyMonitor:
    """Monitor and detect latency anomalies"""
    
    def __init__(self, baseline_window: int = 100, spike_threshold: float = 3.0):
        self.baseline_window = baseline_window
        self.spike_threshold = spike_threshold
        self.latency_history = defaultdict(lambda: deque(maxlen=baseline_window))
        self.spikes = []
    
    def record_latency(self, operation: str, latency_ms: float) -> bool:
        """Record latency and check for spikes"""
        history = self.latency_history[operation]
        
        # Check for spike
        spike_detected = False
        if len(history) >= self.baseline_window // 2:
            baseline = statistics.mean(history)
            if latency_ms > baseline * self.spike_threshold:
                spike_detected = True
                self.spikes.append({
                    'operation': operation,
                    'latency': latency_ms,
                    'baseline': baseline,
                    'timestamp': time.time()
                })
        
        history.append(latency_ms)
        return spike_detected
    
    def get_spike_report(self) -> List[Dict]:
        """Get report of detected spikes"""
        return self.spikes.copy()


class PercentileTracker:
    """Efficient percentile tracking using t-digest algorithm (simplified)"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def add(self, value: float):
        """Add value to tracker"""
        self.values.append(value)
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile value"""
        if not self.values:
            return 0
        
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]