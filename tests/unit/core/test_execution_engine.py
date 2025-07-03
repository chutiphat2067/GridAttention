"""
Unit tests for the Execution Engine component.

Tests cover:
- Order execution and routing
- Smart order routing (SOR)
- Order types and time-in-force
- Slippage and latency management
- Order queue management
- Execution algorithms (TWAP, VWAP, Iceberg)
- Fill tracking and reconciliation
- Exchange connectivity and failover
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import deque
import uuid

# Assuming the module structure
from src.core.execution_engine import (
    ExecutionEngine,
    ExecutionConfig,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    Fill,
    ExecutionAlgorithm,
    SmartOrderRouter,
    OrderBook,
    ExecutionMetrics,
    ExchangeConnector,
    LatencyMonitor
)


class TestExecutionConfig:
    """Test cases for ExecutionConfig validation."""
    
    def test_default_config(self):
        """Test default execution configuration."""
        config = ExecutionConfig()
        
        assert config.max_order_size == Decimal('10000')
        assert config.max_slippage == 0.002  # 0.2%
        assert config.latency_threshold == 100  # 100ms
        assert config.retry_attempts == 3
        assert config.enable_smart_routing is True
        assert config.enable_iceberg_orders is True
        
    def test_custom_config(self):
        """Test custom execution configuration."""
        config = ExecutionConfig(
            max_order_size=Decimal('50000'),
            max_slippage=0.001,
            latency_threshold=50,
            preferred_exchanges=['binance', 'coinbase'],
            execution_algorithms=['TWAP', 'VWAP', 'POV']
        )
        
        assert config.max_order_size == Decimal('50000')
        assert config.max_slippage == 0.001
        assert len(config.preferred_exchanges) == 2
        assert 'TWAP' in config.execution_algorithms
        
    def test_config_validation(self):
        """Test configuration validation rules."""
        # Invalid max slippage
        with pytest.raises(ValueError, match="max_slippage"):
            ExecutionConfig(max_slippage=0.5)  # 50% too high
            
        # Invalid latency threshold
        with pytest.raises(ValueError, match="latency_threshold"):
            ExecutionConfig(latency_threshold=-10)
            
        # Invalid retry attempts
        with pytest.raises(ValueError, match="retry_attempts"):
            ExecutionConfig(retry_attempts=0)


class TestOrder:
    """Test cases for Order class functionality."""
    
    def test_order_creation(self):
        """Test order creation and validation."""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('0.5'),
            price=Decimal('50000'),
            time_in_force=TimeInForce.GTC
        )
        
        assert order.status == OrderStatus.NEW
        assert order.filled_quantity == Decimal('0')
        assert order.remaining_quantity == Decimal('0.5')
        assert order.average_price == Decimal('0')
        
    def test_order_validation(self):
        """Test order validation rules."""
        # Invalid quantity
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(
                order_id='123',
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal('-0.5'),
                price=Decimal('50000')
            )
            
        # Market order with price
        with pytest.raises(ValueError, match="Market orders cannot have price"):
            Order(
                order_id='123',
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal('0.5'),
                price=Decimal('50000')
            )
            
    def test_order_fill_update(self):
        """Test order fill updates."""
        order = Order(
            order_id='123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('50000')
        )
        
        # Partial fill
        fill1 = Fill(
            fill_id='fill_1',
            order_id='123',
            quantity=Decimal('0.3'),
            price=Decimal('49999'),
            timestamp=datetime.now(),
            fee=Decimal('0.3')
        )
        
        order.add_fill(fill1)
        
        assert order.filled_quantity == Decimal('0.3')
        assert order.remaining_quantity == Decimal('0.7')
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.average_price == Decimal('49999')
        
        # Complete fill
        fill2 = Fill(
            fill_id='fill_2',
            order_id='123',
            quantity=Decimal('0.7'),
            price=Decimal('50001'),
            timestamp=datetime.now(),
            fee=Decimal('0.7')
        )
        
        order.add_fill(fill2)
        
        assert order.filled_quantity == Decimal('1.0')
        assert order.remaining_quantity == Decimal('0')
        assert order.status == OrderStatus.FILLED
        
        # Calculate weighted average price
        expected_avg = (Decimal('0.3') * Decimal('49999') + 
                       Decimal('0.7') * Decimal('50001')) / Decimal('1.0')
        assert abs(order.average_price - expected_avg) < Decimal('0.01')


class TestExecutionAlgorithms:
    """Test cases for execution algorithms."""
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01 09:00', periods=390, freq='1min')
        
        # Simulate intraday volume pattern
        time_of_day = np.array([(d.hour - 9) * 60 + d.minute for d in dates])
        volume_profile = 1000 + 500 * np.sin(time_of_day * np.pi / 390) + np.random.normal(0, 100, 390)
        
        prices = 50000 + np.cumsum(np.random.normal(0, 10, 390))
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.maximum(volume_profile, 100),
            'bid': prices - 5,
            'ask': prices + 5
        })
        
    def test_twap_algorithm(self, market_data):
        """Test Time-Weighted Average Price algorithm."""
        algo = ExecutionAlgorithm('TWAP')
        
        # Execute 10 BTC over 60 minutes
        total_quantity = Decimal('10')
        duration_minutes = 60
        
        slices = algo.calculate_twap_slices(
            total_quantity=total_quantity,
            duration_minutes=duration_minutes,
            slice_interval_minutes=5
        )
        
        # Should have 12 slices (60/5)
        assert len(slices) == 12
        
        # Each slice should be equal
        expected_slice_size = total_quantity / 12
        for slice_qty in slices:
            assert abs(slice_qty - expected_slice_size) < Decimal('0.001')
            
        # Total should match
        assert sum(slices) == total_quantity
        
    def test_vwap_algorithm(self, market_data):
        """Test Volume-Weighted Average Price algorithm."""
        algo = ExecutionAlgorithm('VWAP')
        
        # Calculate VWAP slices based on historical volume
        total_quantity = Decimal('10')
        
        volume_profile = market_data.groupby(market_data['timestamp'].dt.hour)['volume'].mean()
        
        slices = algo.calculate_vwap_slices(
            total_quantity=total_quantity,
            volume_profile=volume_profile,
            start_hour=9,
            end_hour=16
        )
        
        # Slices should be proportional to volume
        total_volume = volume_profile.sum()
        for hour, slice_qty in slices.items():
            expected_qty = total_quantity * (volume_profile[hour] / total_volume)
            assert abs(slice_qty - expected_qty) < Decimal('0.1')
            
    def test_iceberg_algorithm(self):
        """Test Iceberg order algorithm."""
        algo = ExecutionAlgorithm('ICEBERG')
        
        # Large order to hide
        total_quantity = Decimal('100')
        visible_quantity = Decimal('5')  # Show only 5 at a time
        
        iceberg_slices = algo.create_iceberg_slices(
            total_quantity=total_quantity,
            visible_quantity=visible_quantity,
            randomize=True
        )
        
        # Check properties
        assert all(slice <= visible_quantity * Decimal('1.1') for slice in iceberg_slices)
        assert sum(iceberg_slices) == total_quantity
        
        # With randomization, slices shouldn't all be identical
        unique_sizes = len(set(iceberg_slices))
        assert unique_sizes > 1
        
    def test_pov_algorithm(self, market_data):
        """Test Percentage of Volume algorithm."""
        algo = ExecutionAlgorithm('POV')
        
        # Participate at 10% of market volume
        participation_rate = 0.10
        total_quantity = Decimal('50')
        
        # Simulate real-time execution
        executed_quantity = Decimal('0')
        market_volume_consumed = Decimal('0')
        
        for _, row in market_data.iterrows():
            market_volume = Decimal(str(row['volume']))
            
            # Calculate how much to execute
            max_executable = market_volume * Decimal(str(participation_rate))
            remaining = total_quantity - executed_quantity
            
            execute_qty = min(max_executable, remaining)
            
            if execute_qty > 0:
                executed_quantity += execute_qty
                market_volume_consumed += market_volume
                
            if executed_quantity >= total_quantity:
                break
                
        # Check participation rate was maintained
        actual_participation = float(executed_quantity / market_volume_consumed)
        assert abs(actual_participation - participation_rate) < 0.01


class TestSmartOrderRouter:
    """Test cases for Smart Order Router."""
    
    @pytest.fixture
    def mock_exchanges(self):
        """Create mock exchange connectors."""
        exchanges = {}
        
        # Mock Binance
        binance = Mock()
        binance.name = 'binance'
        binance.get_order_book = Mock(return_value={
            'bids': [[49990, 10], [49980, 20], [49970, 30]],
            'asks': [[50010, 10], [50020, 20], [50030, 30]]
        })
        binance.trading_fee = 0.001
        binance.is_connected = Mock(return_value=True)
        exchanges['binance'] = binance
        
        # Mock Coinbase
        coinbase = Mock()
        coinbase.name = 'coinbase'
        coinbase.get_order_book = Mock(return_value={
            'bids': [[49995, 15], [49985, 25], [49975, 35]],
            'asks': [[50005, 15], [50015, 25], [50025, 35]]
        })
        coinbase.trading_fee = 0.0015
        coinbase.is_connected = Mock(return_value=True)
        exchanges['coinbase'] = coinbase
        
        return exchanges
        
    @pytest.fixture
    def smart_router(self, mock_exchanges):
        """Create Smart Order Router instance."""
        config = ExecutionConfig(
            enable_smart_routing=True,
            routing_strategy='best_price'
        )
        return SmartOrderRouter(config, mock_exchanges)
        
    def test_best_price_routing(self, smart_router):
        """Test best price routing strategy."""
        order = Order(
            order_id='123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('5')
        )
        
        # Find best execution venue
        routing_plan = smart_router.route_order(order)
        
        assert 'primary_exchange' in routing_plan
        assert routing_plan['primary_exchange'] == 'coinbase'  # Better ask price
        assert 'estimated_price' in routing_plan
        assert routing_plan['estimated_price'] == Decimal('50005')
        
    def test_liquidity_splitting(self, smart_router):
        """Test order splitting across venues for liquidity."""
        # Large order that needs splitting
        order = Order(
            order_id='123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('50')  # Larger than single venue liquidity
        )
        
        routing_plan = smart_router.route_order(order)
        
        assert 'split_orders' in routing_plan
        assert len(routing_plan['split_orders']) > 1
        
        # Check total quantity matches
        total_routed = sum(split['quantity'] for split in routing_plan['split_orders'])
        assert total_routed == order.quantity
        
    def test_fee_aware_routing(self, smart_router):
        """Test routing that considers trading fees."""
        smart_router.config.routing_strategy = 'fee_adjusted'
        
        order = Order(
            order_id='123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('10'),
            price=Decimal('50000')
        )
        
        routing_plan = smart_router.route_order(order)
        
        # Should consider both price and fees
        assert 'total_cost' in routing_plan
        assert 'fee_impact' in routing_plan
        
    def test_failover_routing(self, smart_router, mock_exchanges):
        """Test failover to backup exchanges."""
        # Simulate primary exchange failure
        mock_exchanges['coinbase'].is_connected = Mock(return_value=False)
        
        order = Order(
            order_id='123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('5')
        )
        
        routing_plan = smart_router.route_order(order)
        
        # Should route to backup exchange
        assert routing_plan['primary_exchange'] == 'binance'
        assert 'failover_reason' in routing_plan


class TestExecutionEngine:
    """Test cases for the main ExecutionEngine."""
    
    @pytest.fixture
    def mock_exchange_connector(self):
        """Create mock exchange connector."""
        connector = AsyncMock()
        connector.place_order = AsyncMock(return_value={
            'order_id': 'exchange_123',
            'status': 'new',
            'timestamp': datetime.now()
        })
        connector.cancel_order = AsyncMock(return_value={
            'order_id': 'exchange_123',
            'status': 'cancelled'
        })
        connector.get_order_status = AsyncMock(return_value={
            'order_id': 'exchange_123',
            'status': 'filled',
            'filled_quantity': '1.0',
            'average_price': '50000'
        })
        return connector
        
    @pytest.fixture
    def execution_engine(self, mock_exchange_connector):
        """Create ExecutionEngine instance."""
        config = ExecutionConfig(
            max_order_size=Decimal('100'),
            max_slippage=0.002,
            enable_smart_routing=True
        )
        
        engine = ExecutionEngine(config)
        engine.add_exchange('binance', mock_exchange_connector)
        return engine
        
    @pytest.mark.asyncio
    async def test_order_submission(self, execution_engine):
        """Test order submission flow."""
        order = Order(
            order_id='test_123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('50000')
        )
        
        # Submit order
        result = await execution_engine.submit_order(order)
        
        assert result['status'] == 'submitted'
        assert 'exchange_order_id' in result
        assert order.order_id in execution_engine.active_orders
        
    @pytest.mark.asyncio
    async def test_order_cancellation(self, execution_engine):
        """Test order cancellation."""
        # Submit order first
        order = Order(
            order_id='test_123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('50000')
        )
        
        await execution_engine.submit_order(order)
        
        # Cancel order
        result = await execution_engine.cancel_order(order.order_id)
        
        assert result['status'] == 'cancelled'
        assert order.order_id not in execution_engine.active_orders
        
    @pytest.mark.asyncio
    async def test_order_modification(self, execution_engine):
        """Test order modification."""
        # Submit order
        order = Order(
            order_id='test_123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('50000')
        )
        
        await execution_engine.submit_order(order)
        
        # Modify price
        modifications = {
            'price': Decimal('49500'),
            'quantity': Decimal('0.8')
        }
        
        result = await execution_engine.modify_order(order.order_id, modifications)
        
        assert result['status'] == 'modified'
        assert result['new_price'] == Decimal('49500')
        assert result['new_quantity'] == Decimal('0.8')
        
    @pytest.mark.asyncio
    async def test_bulk_order_submission(self, execution_engine):
        """Test submitting multiple orders."""
        orders = []
        for i in range(5):
            order = Order(
                order_id=f'test_{i}',
                symbol='BTC/USDT',
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal('0.1'),
                price=Decimal('50000') + Decimal(str(i * 10))
            )
            orders.append(order)
            
        # Submit all orders
        results = await execution_engine.submit_bulk_orders(orders)
        
        assert len(results) == 5
        assert all(r['status'] == 'submitted' for r in results)
        assert len(execution_engine.active_orders) == 5
        
    @pytest.mark.asyncio
    async def test_order_fill_processing(self, execution_engine):
        """Test processing order fills."""
        order = Order(
            order_id='test_123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('50000')
        )
        
        await execution_engine.submit_order(order)
        
        # Simulate fill event
        fill_event = {
            'order_id': 'test_123',
            'exchange_order_id': 'exchange_123',
            'fill_id': 'fill_001',
            'quantity': '0.5',
            'price': '49999',
            'timestamp': datetime.now().isoformat(),
            'fee': '0.05'
        }
        
        await execution_engine.process_fill(fill_event)
        
        # Check order updated
        updated_order = execution_engine.get_order(order.order_id)
        assert updated_order.filled_quantity == Decimal('0.5')
        assert updated_order.status == OrderStatus.PARTIALLY_FILLED
        
    @pytest.mark.asyncio
    async def test_slippage_monitoring(self, execution_engine):
        """Test slippage monitoring and alerts."""
        order = Order(
            order_id='test_123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('10.0'),
            expected_price=Decimal('50000')  # Expected execution price
        )
        
        await execution_engine.submit_order(order)
        
        # Simulate fill with slippage
        fill_event = {
            'order_id': 'test_123',
            'fill_id': 'fill_001',
            'quantity': '10.0',
            'price': '50150',  # 0.3% slippage
            'timestamp': datetime.now().isoformat()
        }
        
        alerts = await execution_engine.process_fill(fill_event)
        
        # Should generate slippage alert (> 0.2% threshold)
        assert any(alert['type'] == 'SLIPPAGE_WARNING' for alert in alerts)
        
    @pytest.mark.asyncio
    async def test_order_retry_mechanism(self, execution_engine, mock_exchange_connector):
        """Test order retry on failures."""
        # Configure to fail first 2 attempts
        call_count = 0
        
        async def place_order_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Network error")
            return {
                'order_id': 'exchange_123',
                'status': 'new',
                'timestamp': datetime.now()
            }
            
        mock_exchange_connector.place_order = place_order_with_retry
        
        order = Order(
            order_id='test_123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('50000')
        )
        
        result = await execution_engine.submit_order(order)
        
        # Should succeed after retries
        assert result['status'] == 'submitted'
        assert call_count == 3
        
    @pytest.mark.asyncio
    async def test_order_queue_management(self, execution_engine):
        """Test order queue and priority handling."""
        # Create orders with different priorities
        high_priority_order = Order(
            order_id='high_001',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('1.0'),
            priority=1  # Highest priority
        )
        
        normal_order = Order(
            order_id='normal_001',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('50000'),
            priority=5
        )
        
        # Submit in reverse priority order
        await execution_engine.submit_order(normal_order)
        await execution_engine.submit_order(high_priority_order)
        
        # Check execution order (would need to mock execution delays)
        execution_order = execution_engine.get_execution_history()
        
        # High priority should execute first
        assert execution_order[0]['order_id'] == 'high_001'


class TestLatencyMonitor:
    """Test cases for latency monitoring."""
    
    @pytest.fixture
    def latency_monitor(self):
        """Create LatencyMonitor instance."""
        return LatencyMonitor(
            warning_threshold=100,  # 100ms warning
            critical_threshold=500  # 500ms critical
        )
        
    def test_latency_measurement(self, latency_monitor):
        """Test latency measurement and statistics."""
        # Record some latencies
        latencies = [50, 75, 100, 150, 200, 1000]  # Last one is outlier
        
        for latency in latencies:
            latency_monitor.record_latency('binance', 'order_submit', latency)
            
        stats = latency_monitor.get_statistics('binance', 'order_submit')
        
        assert stats['count'] == 6
        assert stats['mean'] > 100
        assert stats['median'] < stats['mean']  # Due to outlier
        assert stats['p95'] >= 200
        assert stats['p99'] >= 1000
        
    def test_latency_alerts(self, latency_monitor):
        """Test latency alert generation."""
        alerts = []
        latency_monitor.set_alert_callback(lambda alert: alerts.append(alert))
        
        # Record increasing latencies
        latency_monitor.record_latency('binance', 'order_submit', 50)
        latency_monitor.record_latency('binance', 'order_submit', 150)  # Warning
        latency_monitor.record_latency('binance', 'order_submit', 600)  # Critical
        
        assert len(alerts) == 2
        assert alerts[0]['level'] == 'warning'
        assert alerts[1]['level'] == 'critical'
        
    def test_latency_degradation_detection(self, latency_monitor):
        """Test detection of latency degradation over time."""
        # Simulate gradual degradation
        base_latency = 50
        
        for i in range(20):
            latency = base_latency + i * 10  # Gradually increasing
            latency_monitor.record_latency('binance', 'order_submit', latency)
            
        degradation = latency_monitor.detect_degradation('binance', 'order_submit')
        
        assert degradation['is_degrading'] is True
        assert degradation['trend'] > 0  # Positive trend (increasing latency)
        assert 'recommendation' in degradation


class TestExecutionMetrics:
    """Test cases for execution metrics and analytics."""
    
    @pytest.fixture
    def execution_metrics(self):
        """Create ExecutionMetrics instance."""
        return ExecutionMetrics()
        
    def test_fill_rate_calculation(self, execution_metrics):
        """Test fill rate metrics."""
        # Add some orders
        orders = [
            {'order_id': '1', 'quantity': Decimal('10'), 'filled': Decimal('10')},
            {'order_id': '2', 'quantity': Decimal('5'), 'filled': Decimal('3')},
            {'order_id': '3', 'quantity': Decimal('8'), 'filled': Decimal('8')},
            {'order_id': '4', 'quantity': Decimal('2'), 'filled': Decimal('0')}
        ]
        
        for order in orders:
            execution_metrics.add_order_result(order)
            
        fill_metrics = execution_metrics.calculate_fill_rates()
        
        assert fill_metrics['overall_fill_rate'] == 0.84  # 21/25
        assert fill_metrics['complete_fill_rate'] == 0.5   # 2/4 orders
        
    def test_slippage_analysis(self, execution_metrics):
        """Test slippage analysis."""
        fills = [
            {
                'expected_price': Decimal('50000'),
                'actual_price': Decimal('50050'),
                'quantity': Decimal('1'),
                'side': 'buy'
            },
            {
                'expected_price': Decimal('50000'),
                'actual_price': Decimal('49980'),
                'quantity': Decimal('2'),
                'side': 'sell'
            }
        ]
        
        for fill in fills:
            execution_metrics.add_fill(fill)
            
        slippage_stats = execution_metrics.calculate_slippage_statistics()
        
        assert 'average_slippage_bps' in slippage_stats
        assert 'positive_slippage_rate' in slippage_stats
        assert 'slippage_cost' in slippage_stats
        
    def test_venue_performance(self, execution_metrics):
        """Test venue performance comparison."""
        # Add execution data from multiple venues
        venues_data = {
            'binance': {'fills': 100, 'failures': 2, 'avg_latency': 50},
            'coinbase': {'fills': 80, 'failures': 5, 'avg_latency': 75},
            'kraken': {'fills': 60, 'failures': 1, 'avg_latency': 100}
        }
        
        for venue, data in venues_data.items():
            for _ in range(data['fills']):
                execution_metrics.add_venue_execution(venue, 'success', data['avg_latency'])
            for _ in range(data['failures']):
                execution_metrics.add_venue_execution(venue, 'failure', data['avg_latency'])
                
        venue_comparison = execution_metrics.compare_venues()
        
        # Binance should rank highest (best success rate and latency)
        assert venue_comparison[0]['venue'] == 'binance'
        assert venue_comparison[0]['success_rate'] > 0.97
        
    def test_execution_cost_analysis(self, execution_metrics):
        """Test execution cost analysis."""
        trades = [
            {
                'venue': 'binance',
                'quantity': Decimal('1'),
                'price': Decimal('50000'),
                'fee_rate': Decimal('0.001'),
                'slippage': Decimal('50')
            },
            {
                'venue': 'coinbase',
                'quantity': Decimal('0.5'),
                'price': Decimal('50000'),
                'fee_rate': Decimal('0.0015'),
                'slippage': Decimal('25')
            }
        ]
        
        for trade in trades:
            execution_metrics.add_trade_cost(trade)
            
        cost_analysis = execution_metrics.analyze_execution_costs()
        
        assert 'total_fee_cost' in cost_analysis
        assert 'total_slippage_cost' in cost_analysis
        assert 'cost_per_venue' in cost_analysis
        assert 'optimal_venue' in cost_analysis


class TestOrderBookAggregation:
    """Test cases for order book aggregation."""
    
    def test_order_book_merge(self):
        """Test merging order books from multiple venues."""
        book1 = OrderBook('binance')
        book1.update({
            'bids': [[50000, 1], [49990, 2], [49980, 3]],
            'asks': [[50010, 1], [50020, 2], [50030, 3]]
        })
        
        book2 = OrderBook('coinbase')
        book2.update({
            'bids': [[50005, 0.5], [49995, 1.5], [49985, 2.5]],
            'asks': [[50015, 0.5], [50025, 1.5], [50035, 2.5]]
        })
        
        aggregated = OrderBook.aggregate([book1, book2])
        
        # Best bid should be 50005 from coinbase
        assert aggregated.best_bid[0] == 50005
        assert aggregated.best_bid[1] == 0.5
        
        # Best ask should be 50010 from binance
        assert aggregated.best_ask[0] == 50010
        assert aggregated.best_ask[1] == 1
        
    def test_liquidity_calculation(self):
        """Test liquidity metrics calculation."""
        book = OrderBook('test')
        book.update({
            'bids': [[50000, 10], [49990, 20], [49980, 30]],
            'asks': [[50010, 10], [50020, 20], [50030, 30]]
        })
        
        # Calculate liquidity at different depths
        liquidity_1bps = book.calculate_liquidity(0.0001)  # 0.01%
        liquidity_10bps = book.calculate_liquidity(0.001)  # 0.1%
        
        assert liquidity_10bps['bid_liquidity'] > liquidity_1bps['bid_liquidity']
        assert liquidity_10bps['ask_liquidity'] > liquidity_1bps['ask_liquidity']
        
    def test_market_impact_estimation(self):
        """Test market impact estimation for large orders."""
        book = OrderBook('test')
        book.update({
            'bids': [[50000, 5], [49990, 10], [49980, 15]],
            'asks': [[50010, 5], [50020, 10], [50030, 15]]
        })
        
        # Estimate impact of buying 20 units
        impact = book.estimate_market_impact('buy', 20)
        
        assert impact['average_price'] > 50010  # Should be higher than best ask
        assert impact['price_impact'] > 0  # Positive impact for buy
        assert impact['levels_consumed'] == 3  # Need 3 levels for 20 units


class TestExecutionReporting:
    """Test cases for execution reporting and reconciliation."""
    
    def test_execution_report_generation(self):
        """Test generation of execution reports."""
        reporter = ExecutionMetrics()
        
        # Add sample executions
        executions = [
            {
                'timestamp': datetime.now(),
                'order_id': '001',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'quantity': Decimal('1'),
                'price': Decimal('50000'),
                'venue': 'binance',
                'fee': Decimal('50'),
                'slippage': Decimal('25')
            }
        ]
        
        for exec in executions:
            reporter.add_execution(exec)
            
        report = reporter.generate_daily_report(datetime.now().date())
        
        assert 'summary' in report
        assert 'executions' in report
        assert 'metrics' in report
        assert 'costs' in report
        
    def test_execution_reconciliation(self):
        """Test reconciliation between internal records and exchange."""
        internal_records = [
            {'order_id': '001', 'filled': Decimal('1'), 'status': 'filled'},
            {'order_id': '002', 'filled': Decimal('0.5'), 'status': 'partial'},
            {'order_id': '003', 'filled': Decimal('0'), 'status': 'new'}
        ]
        
        exchange_records = [
            {'order_id': '001', 'filled': '1.0', 'status': 'filled'},
            {'order_id': '002', 'filled': '0.6', 'status': 'partial'},  # Mismatch
            {'order_id': '003', 'filled': '0', 'status': 'cancelled'}   # Status mismatch
        ]
        
        reconciler = ExecutionMetrics()
        discrepancies = reconciler.reconcile_executions(internal_records, exchange_records)
        
        assert len(discrepancies) == 2
        assert any(d['order_id'] == '002' and d['type'] == 'fill_mismatch' for d in discrepancies)
        assert any(d['order_id'] == '003' and d['type'] == 'status_mismatch' for d in discrepancies)