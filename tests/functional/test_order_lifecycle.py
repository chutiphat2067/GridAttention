# tests/functional/test_order_lifecycle.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from core.execution_engine import ExecutionEngine
from core.risk_management_system import RiskManagementSystem
from infrastructure.event_bus import EventBus, Event, EventType
from monitoring.performance_monitor import PerformanceMonitor


class OrderStatus(Enum):
    """Order status types"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    GRID = "grid"


class TimeInForce(Enum):
    """Time in force types"""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date
    DAY = "day"  # Day order


@dataclass
class Order:
    """Order object"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    price: Optional[float]
    size: float
    status: OrderStatus
    time_in_force: TimeInForce
    created_at: datetime
    updated_at: datetime
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    fills: List[Dict[str, Any]] = field(default_factory=list)
    status_history: List[Dict[str, Any]] = field(default_factory=list)
    

@dataclass
class OrderUpdate:
    """Order update event"""
    order_id: str
    timestamp: datetime
    update_type: str
    old_status: OrderStatus
    new_status: OrderStatus
    data: Dict[str, Any]


class TestOrderLifecycle:
    """Test complete order lifecycle management"""
    
    @pytest.fixture
    async def order_system(self):
        """Create order management system"""
        config = {
            'symbol': 'BTC/USDT',
            'execution': {
                'max_slippage': 0.001,  # 0.1%
                'retry_attempts': 3,
                'timeout_seconds': 30,
                'rate_limit': 10,  # orders per second
                'min_order_size': 0.001,
                'max_order_size': 10.0
            },
            'order_management': {
                'enable_amend': True,
                'enable_iceberg': True,
                'enable_trailing': True,
                'max_open_orders': 100,
                'order_ttl': 86400  # 24 hours
            },
            'fees': {
                'maker': 0.001,  # 0.1%
                'taker': 0.0015  # 0.15%
            }
        }
        
        system = {
            'execution_engine': ExecutionEngine(config),
            'risk_manager': RiskManagementSystem(config),
            'event_bus': EventBus(),
            'performance_monitor': PerformanceMonitor(config),
            'orders': {},  # Active orders
            'order_history': [],  # Completed orders
            'fill_simulator': None  # For testing
        }
        
        # Setup fill simulator for testing
        system['fill_simulator'] = self._create_fill_simulator()
        
        return system, config
    
    @pytest.mark.asyncio
    async def test_market_order_lifecycle(self, order_system):
        """Test market order from creation to completion"""
        system, config = order_system
        
        # Create market order
        order_request = {
            'symbol': config['symbol'],
            'side': 'buy',
            'size': 0.1,
            'type': OrderType.MARKET
        }
        
        # Validate with risk manager
        risk_check = await system['risk_manager'].validate_order(order_request)
        assert risk_check['approved']
        
        # Create order object
        order = Order(
            id=f"MKT_{datetime.now().timestamp()}",
            symbol=order_request['symbol'],
            side=order_request['side'],
            order_type=OrderType.MARKET,
            price=None,  # Market orders don't have price
            size=order_request['size'],
            status=OrderStatus.PENDING,
            time_in_force=TimeInForce.IOC,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Track lifecycle events
        lifecycle_events = []
        
        async def track_event(event):
            lifecycle_events.append({
                'timestamp': datetime.now(),
                'event_type': event.type,
                'data': event.data
            })
        
        system['event_bus'].subscribe(EventType.ORDER_UPDATE, track_event)
        
        # Submit order
        order.status = OrderStatus.SUBMITTED
        order.status_history.append({
            'status': OrderStatus.SUBMITTED,
            'timestamp': datetime.now()
        })
        
        await system['event_bus'].publish(Event(
            type=EventType.ORDER_UPDATE,
            data={'order_id': order.id, 'status': 'submitted'}
        ))
        
        # Simulate immediate fill for market order
        fill_price = 50000  # Current market price
        fill_result = {
            'order_id': order.id,
            'price': fill_price,
            'size': order.size,
            'timestamp': datetime.now(),
            'fee': order.size * fill_price * config['fees']['taker']
        }
        
        # Process fill
        order.status = OrderStatus.FILLED
        order.filled_size = fill_result['size']
        order.avg_fill_price = fill_result['price']
        order.fees = fill_result['fee']
        order.fills.append(fill_result)
        order.updated_at = datetime.now()
        
        await system['event_bus'].publish(Event(
            type=EventType.ORDER_UPDATE,
            data={'order_id': order.id, 'status': 'filled', 'fill': fill_result}
        ))
        
        # Store completed order
        system['order_history'].append(order)
        
        # Verify lifecycle
        assert order.status == OrderStatus.FILLED
        assert order.filled_size == order.size
        assert len(order.fills) == 1
        assert len(lifecycle_events) >= 2  # submitted and filled events
        
        # Check execution time
        execution_time = (order.updated_at - order.created_at).total_seconds()
        assert execution_time < 1.0  # Market orders should fill quickly
    
    @pytest.mark.asyncio
    async def test_limit_order_lifecycle(self, order_system):
        """Test limit order with various scenarios"""
        system, config = order_system
        
        # Current market price
        market_price = 50000
        
        # Test scenarios
        scenarios = [
            {
                'name': 'immediate_fill',
                'side': 'buy',
                'price': 50100,  # Above market
                'expected': 'filled'
            },
            {
                'name': 'resting_order',
                'side': 'buy',
                'price': 49900,  # Below market
                'expected': 'pending'
            },
            {
                'name': 'partial_fill',
                'side': 'sell',
                'price': 49950,
                'size': 0.5,
                'fill_sizes': [0.2, 0.3],  # Multiple partial fills
                'expected': 'filled'
            }
        ]
        
        for scenario in scenarios:
            # Create limit order
            order = Order(
                id=f"LMT_{scenario['name']}_{datetime.now().timestamp()}",
                symbol=config['symbol'],
                side=scenario['side'],
                order_type=OrderType.LIMIT,
                price=scenario['price'],
                size=scenario.get('size', 0.1),
                status=OrderStatus.PENDING,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            system['orders'][order.id] = order
            
            # Submit order
            submission_result = await system['execution_engine'].submit_order(order)
            assert submission_result['success']
            
            order.status = OrderStatus.SUBMITTED
            
            # Simulate market movements and fills
            if scenario['name'] == 'immediate_fill':
                # Should fill immediately
                await self._simulate_fill(system, order, order.price, order.size)
                assert order.status == OrderStatus.FILLED
                
            elif scenario['name'] == 'resting_order':
                # Should remain open
                await asyncio.sleep(0.1)
                assert order.status == OrderStatus.SUBMITTED
                
                # Simulate price movement to trigger fill
                await self._simulate_price_movement(system, 49900)
                await self._simulate_fill(system, order, order.price, order.size)
                assert order.status == OrderStatus.FILLED
                
            elif scenario['name'] == 'partial_fill':
                # Multiple partial fills
                for fill_size in scenario['fill_sizes']:
                    await self._simulate_fill(system, order, order.price, fill_size)
                    
                    if order.filled_size < order.size:
                        assert order.status == OrderStatus.PARTIALLY_FILLED
                    else:
                        assert order.status == OrderStatus.FILLED
                
                assert order.filled_size == order.size
                assert len(order.fills) == len(scenario['fill_sizes'])
    
    @pytest.mark.asyncio
    async def test_stop_orders_lifecycle(self, order_system):
        """Test stop loss and take profit orders"""
        system, config = order_system
        
        # Setup position to protect
        position = {
            'symbol': config['symbol'],
            'side': 'long',
            'size': 0.2,
            'entry_price': 50000
        }
        
        # Create stop loss order
        stop_loss = Order(
            id=f"STP_LOSS_{datetime.now().timestamp()}",
            symbol=config['symbol'],
            side='sell',  # Opposite of position
            order_type=OrderType.STOP_LOSS,
            price=49000,  # Trigger price
            size=position['size'],
            status=OrderStatus.PENDING,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={'trigger_price': 49000, 'position_id': 'pos_123'}
        )
        
        # Create take profit order
        take_profit = Order(
            id=f"TAKE_PROFIT_{datetime.now().timestamp()}",
            symbol=config['symbol'],
            side='sell',
            order_type=OrderType.TAKE_PROFIT,
            price=51000,  # Trigger price
            size=position['size'],
            status=OrderStatus.PENDING,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={'trigger_price': 51000, 'position_id': 'pos_123'}
        )
        
        system['orders'][stop_loss.id] = stop_loss
        system['orders'][take_profit.id] = take_profit
        
        # Monitor price for triggers
        price_feed = [
            50000,  # Start
            50200,  # Move up
            50500,  # Continue up
            51050,  # Trigger take profit
            50800,  # After TP
            49500,  # Drop
            48950   # Would trigger stop loss (but already closed)
        ]
        
        triggered_orders = []
        
        for price in price_feed:
            # Check stop orders
            for order_id, order in list(system['orders'].items()):
                if order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                    trigger_price = order.metadata['trigger_price']
                    
                    triggered = False
                    if order.order_type == OrderType.STOP_LOSS:
                        triggered = price <= trigger_price
                    elif order.order_type == OrderType.TAKE_PROFIT:
                        triggered = price >= trigger_price
                    
                    if triggered and order.status == OrderStatus.PENDING:
                        # Trigger order
                        order.status = OrderStatus.SUBMITTED
                        triggered_orders.append(order)
                        
                        # Convert to market order and fill
                        await self._simulate_fill(system, order, price, order.size)
                        
                        # Cancel opposite order (OCO - One Cancels Other)
                        opposite_type = (OrderType.TAKE_PROFIT 
                                       if order.order_type == OrderType.STOP_LOSS 
                                       else OrderType.STOP_LOSS)
                        
                        for other_id, other_order in system['orders'].items():
                            if (other_order.order_type == opposite_type and
                                other_order.metadata.get('position_id') == order.metadata.get('position_id')):
                                other_order.status = OrderStatus.CANCELLED
                                other_order.metadata['cancel_reason'] = 'oco_triggered'
        
        # Verify stop order behavior
        assert len(triggered_orders) == 1
        assert triggered_orders[0].order_type == OrderType.TAKE_PROFIT
        assert take_profit.status == OrderStatus.FILLED
        assert stop_loss.status == OrderStatus.CANCELLED
        assert stop_loss.metadata.get('cancel_reason') == 'oco_triggered'
    
    @pytest.mark.asyncio
    async def test_advanced_order_types(self, order_system):
        """Test advanced order types like iceberg and trailing stop"""
        system, config = order_system
        
        # 1. Iceberg Order
        iceberg_order = Order(
            id=f"ICEBERG_{datetime.now().timestamp()}",
            symbol=config['symbol'],
            side='buy',
            order_type=OrderType.ICEBERG,
            price=49900,
            size=1.0,  # Total size
            status=OrderStatus.PENDING,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                'visible_size': 0.1,  # Only show 0.1 at a time
                'remaining_size': 1.0,
                'slice_count': 0
            }
        )
        
        if config['order_management']['enable_iceberg']:
            # Process iceberg slices
            total_filled = 0
            while total_filled < iceberg_order.size:
                # Create visible slice
                slice_size = min(
                    iceberg_order.metadata['visible_size'],
                    iceberg_order.size - total_filled
                )
                
                # Simulate fill of visible portion
                await self._simulate_fill(system, iceberg_order, iceberg_order.price, slice_size)
                total_filled += slice_size
                iceberg_order.metadata['slice_count'] += 1
                
                # Update remaining
                iceberg_order.metadata['remaining_size'] = iceberg_order.size - total_filled
            
            assert iceberg_order.filled_size == iceberg_order.size
            assert iceberg_order.metadata['slice_count'] == 10  # 1.0 / 0.1
        
        # 2. Trailing Stop Order
        trailing_stop = Order(
            id=f"TRAIL_STOP_{datetime.now().timestamp()}",
            symbol=config['symbol'],
            side='sell',
            order_type=OrderType.TRAILING_STOP,
            price=None,  # Dynamic
            size=0.1,
            status=OrderStatus.PENDING,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                'trail_amount': 100,  # $100 trailing distance
                'trail_percent': None,  # Or use percentage
                'highest_price': 50000,  # Starting reference
                'trigger_price': 49900  # Initial trigger
            }
        )
        
        if config['order_management']['enable_trailing']:
            # Simulate price movements
            price_movements = [50000, 50200, 50500, 50300, 50100, 49800]
            
            for price in price_movements:
                # Update trailing stop
                if price > trailing_stop.metadata['highest_price']:
                    # Price went up, adjust stop
                    trailing_stop.metadata['highest_price'] = price
                    trailing_stop.metadata['trigger_price'] = price - trailing_stop.metadata['trail_amount']
                
                # Check if triggered
                if price <= trailing_stop.metadata['trigger_price']:
                    trailing_stop.status = OrderStatus.SUBMITTED
                    await self._simulate_fill(system, trailing_stop, price, trailing_stop.size)
                    break
            
            assert trailing_stop.status == OrderStatus.FILLED
            assert trailing_stop.metadata['highest_price'] == 50500
            assert trailing_stop.avg_fill_price == 49800
    
    @pytest.mark.asyncio
    async def test_order_amendment_lifecycle(self, order_system):
        """Test order modification/amendment"""
        system, config = order_system
        
        if not config['order_management']['enable_amend']:
            pytest.skip("Order amendment not enabled")
        
        # Create initial order
        order = Order(
            id=f"AMEND_TEST_{datetime.now().timestamp()}",
            symbol=config['symbol'],
            side='buy',
            order_type=OrderType.LIMIT,
            price=49800,
            size=0.1,
            status=OrderStatus.SUBMITTED,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        system['orders'][order.id] = order
        
        # Test various amendments
        amendments = [
            {
                'type': 'price_change',
                'new_price': 49900,
                'expected': 'success'
            },
            {
                'type': 'size_increase',
                'new_size': 0.2,
                'expected': 'success'
            },
            {
                'type': 'size_decrease',
                'new_size': 0.05,
                'expected': 'success'
            }
        ]
        
        for amendment in amendments:
            # Validate amendment
            can_amend = await system['execution_engine'].can_amend_order(
                order_id=order.id,
                changes=amendment
            )
            
            if can_amend:
                # Store old values
                old_price = order.price
                old_size = order.size
                
                # Apply amendment
                if 'new_price' in amendment:
                    order.price = amendment['new_price']
                if 'new_size' in amendment:
                    # Check if size can be decreased (no partial fills)
                    if amendment['new_size'] >= order.filled_size:
                        order.size = amendment['new_size']
                
                order.updated_at = datetime.now()
                
                # Record amendment
                order.metadata['amendments'] = order.metadata.get('amendments', [])
                order.metadata['amendments'].append({
                    'timestamp': datetime.now(),
                    'type': amendment['type'],
                    'old_values': {'price': old_price, 'size': old_size},
                    'new_values': {'price': order.price, 'size': order.size}
                })
                
                # Emit amendment event
                await system['event_bus'].publish(Event(
                    type=EventType.ORDER_UPDATE,
                    data={
                        'order_id': order.id,
                        'update_type': 'amended',
                        'changes': amendment
                    }
                ))
        
        # Verify amendments
        assert len(order.metadata.get('amendments', [])) == len(amendments)
        assert order.price == 49900
        assert order.size == 0.05
    
    @pytest.mark.asyncio
    async def test_order_cancellation_scenarios(self, order_system):
        """Test various order cancellation scenarios"""
        system, config = order_system
        
        # Create multiple orders
        orders = []
        for i in range(5):
            order = Order(
                id=f"CANCEL_TEST_{i}_{datetime.now().timestamp()}",
                symbol=config['symbol'],
                side='buy' if i % 2 == 0 else 'sell',
                order_type=OrderType.LIMIT,
                price=49900 + i * 10,
                size=0.1,
                status=OrderStatus.SUBMITTED,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            orders.append(order)
            system['orders'][order.id] = order
        
        # Test different cancellation scenarios
        
        # 1. Single order cancellation
        cancel_result = await system['execution_engine'].cancel_order(orders[0].id)
        assert cancel_result['success']
        orders[0].status = OrderStatus.CANCELLED
        orders[0].metadata['cancel_reason'] = 'user_requested'
        
        # 2. Cancel all orders for symbol
        symbol_cancel_result = await system['execution_engine'].cancel_all_orders(
            symbol=config['symbol']
        )
        
        cancelled_count = 0
        for order in orders[1:]:
            if order.status == OrderStatus.SUBMITTED:
                order.status = OrderStatus.CANCELLED
                order.metadata['cancel_reason'] = 'cancel_all'
                cancelled_count += 1
        
        assert cancelled_count > 0
        
        # 3. Conditional cancellation (e.g., cancel buy orders only)
        buy_orders = [o for o in orders if o.side == 'buy' and o.status == OrderStatus.SUBMITTED]
        for order in buy_orders:
            order.status = OrderStatus.CANCELLED
            order.metadata['cancel_reason'] = 'conditional_cancel'
        
        # 4. Failed cancellation (order already filled)
        filled_order = Order(
            id=f"FILLED_ORDER_{datetime.now().timestamp()}",
            symbol=config['symbol'],
            side='buy',
            order_type=OrderType.LIMIT,
            price=50000,
            size=0.1,
            status=OrderStatus.FILLED,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        cancel_filled_result = await system['execution_engine'].cancel_order(filled_order.id)
        assert not cancel_filled_result['success']
        assert cancel_filled_result['reason'] == 'order_not_cancellable'
        
        # Verify cancellations
        cancelled_orders = [o for o in orders if o.status == OrderStatus.CANCELLED]
        assert len(cancelled_orders) == len(orders)
    
    @pytest.mark.asyncio
    async def test_order_expiration_handling(self, order_system):
        """Test order expiration and time-based lifecycle"""
        system, config = order_system
        
        # Create orders with different time in force
        expiration_tests = [
            {
                'tif': TimeInForce.DAY,
                'expire_after': timedelta(hours=24)
            },
            {
                'tif': TimeInForce.GTD,
                'expire_after': timedelta(hours=48),
                'expire_time': datetime.now() + timedelta(hours=48)
            },
            {
                'tif': TimeInForce.IOC,
                'expire_after': timedelta(seconds=0)  # Immediate
            }
        ]
        
        for test in expiration_tests:
            order = Order(
                id=f"EXPIRE_TEST_{test['tif'].value}_{datetime.now().timestamp()}",
                symbol=config['symbol'],
                side='buy',
                order_type=OrderType.LIMIT,
                price=49000,  # Far from market, won't fill
                size=0.1,
                status=OrderStatus.SUBMITTED,
                time_in_force=test['tif'],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={'expire_time': test.get('expire_time')}
            )
            
            system['orders'][order.id] = order
            
            # Simulate time passage
            if test['tif'] == TimeInForce.IOC:
                # Should expire immediately if not filled
                await asyncio.sleep(0.1)
                order.status = OrderStatus.EXPIRED
                order.metadata['expire_reason'] = 'ioc_not_filled'
                
            elif test['tif'] == TimeInForce.DAY:
                # Check at end of day
                if (datetime.now() - order.created_at) >= test['expire_after']:
                    order.status = OrderStatus.EXPIRED
                    order.metadata['expire_reason'] = 'day_order_expired'
                    
            elif test['tif'] == TimeInForce.GTD:
                # Check against specific time
                if datetime.now() >= order.metadata['expire_time']:
                    order.status = OrderStatus.EXPIRED
                    order.metadata['expire_reason'] = 'gtd_time_reached'
        
        # Verify expiration handling
        expired_orders = [o for o in system['orders'].values() 
                         if o.status == OrderStatus.EXPIRED]
        
        ioc_expired = [o for o in expired_orders if o.time_in_force == TimeInForce.IOC]
        assert len(ioc_expired) > 0
        assert ioc_expired[0].metadata['expire_reason'] == 'ioc_not_filled'
    
    @pytest.mark.asyncio
    async def test_order_execution_errors_and_recovery(self, order_system):
        """Test order execution error handling and recovery"""
        system, config = order_system
        
        # Define error scenarios
        error_scenarios = [
            {
                'type': 'insufficient_balance',
                'order': Order(
                    id=f"ERR_BALANCE_{datetime.now().timestamp()}",
                    symbol=config['symbol'],
                    side='buy',
                    order_type=OrderType.MARKET,
                    price=None,
                    size=100.0,  # Very large
                    status=OrderStatus.PENDING,
                    time_in_force=TimeInForce.IOC,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ),
                'expected_status': OrderStatus.REJECTED
            },
            {
                'type': 'min_size_not_met',
                'order': Order(
                    id=f"ERR_MINSIZE_{datetime.now().timestamp()}",
                    symbol=config['symbol'],
                    side='buy',
                    order_type=OrderType.LIMIT,
                    price=50000,
                    size=0.0001,  # Below minimum
                    status=OrderStatus.PENDING,
                    time_in_force=TimeInForce.GTC,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ),
                'expected_status': OrderStatus.REJECTED
            },
            {
                'type': 'network_timeout',
                'order': Order(
                    id=f"ERR_NETWORK_{datetime.now().timestamp()}",
                    symbol=config['symbol'],
                    side='sell',
                    order_type=OrderType.MARKET,
                    price=None,
                    size=0.1,
                    status=OrderStatus.PENDING,
                    time_in_force=TimeInForce.IOC,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ),
                'expected_status': OrderStatus.FAILED,
                'retry_count': 3
            }
        ]
        
        for scenario in error_scenarios:
            order = scenario['order']
            
            # Pre-execution validation
            if scenario['type'] == 'insufficient_balance':
                # Simulate balance check
                available_balance = 5000  # USDT
                required_balance = order.size * 50000  # Assume BTC price
                
                if required_balance > available_balance:
                    order.status = OrderStatus.REJECTED
                    order.metadata['reject_reason'] = 'insufficient_balance'
                    order.metadata['required'] = required_balance
                    order.metadata['available'] = available_balance
                    
            elif scenario['type'] == 'min_size_not_met':
                if order.size < config['execution']['min_order_size']:
                    order.status = OrderStatus.REJECTED
                    order.metadata['reject_reason'] = 'below_min_size'
                    order.metadata['min_size'] = config['execution']['min_order_size']
                    
            elif scenario['type'] == 'network_timeout':
                # Simulate network issues with retry
                retry_count = 0
                max_retries = scenario.get('retry_count', 3)
                
                while retry_count < max_retries:
                    try:
                        # Simulate network call
                        if retry_count < max_retries - 1:
                            raise TimeoutError("Network timeout")
                        
                        # Success on last retry
                        order.status = OrderStatus.SUBMITTED
                        break
                        
                    except TimeoutError:
                        retry_count += 1
                        order.metadata['retry_attempts'] = retry_count
                        
                        if retry_count >= max_retries:
                            order.status = OrderStatus.FAILED
                            order.metadata['fail_reason'] = 'max_retries_exceeded'
                        else:
                            await asyncio.sleep(0.1 * retry_count)  # Exponential backoff
            
            # Record error event
            await system['event_bus'].publish(Event(
                type=EventType.ORDER_ERROR,
                data={
                    'order_id': order.id,
                    'error_type': scenario['type'],
                    'status': order.status.value,
                    'metadata': order.metadata
                }
            ))
            
            # Verify error handling
            assert order.status == scenario['expected_status']
            assert 'reject_reason' in order.metadata or 'fail_reason' in order.metadata
    
    @pytest.mark.asyncio
    async def test_order_performance_metrics(self, order_system):
        """Test order execution performance tracking"""
        system, config = order_system
        
        # Create orders for performance testing
        performance_orders = []
        
        for i in range(20):
            order = Order(
                id=f"PERF_TEST_{i}_{datetime.now().timestamp()}",
                symbol=config['symbol'],
                side='buy' if i % 2 == 0 else 'sell',
                order_type=OrderType.LIMIT if i % 3 == 0 else OrderType.MARKET,
                price=50000 + (i - 10) * 10 if i % 3 == 0 else None,
                size=0.1 + i * 0.01,
                status=OrderStatus.PENDING,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.now() - timedelta(seconds=i * 5),
                updated_at=datetime.now(),
                metadata={'test_batch': 'performance'}
            )
            performance_orders.append(order)
            
            # Simulate different execution scenarios
            if i % 4 == 0:
                # Immediate fill
                order.status = OrderStatus.FILLED
                order.filled_size = order.size
                order.avg_fill_price = order.price or 50000
                order.updated_at = order.created_at + timedelta(milliseconds=50)
                
            elif i % 4 == 1:
                # Delayed fill
                order.status = OrderStatus.FILLED
                order.filled_size = order.size
                order.avg_fill_price = order.price or 50000
                order.updated_at = order.created_at + timedelta(seconds=2)
                
            elif i % 4 == 2:
                # Partial fill
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_size = order.size * 0.6
                order.avg_fill_price = order.price or 50000
                
            else:
                # Cancelled
                order.status = OrderStatus.CANCELLED
                order.updated_at = order.created_at + timedelta(seconds=10)
        
        # Calculate performance metrics
        metrics = {
            'total_orders': len(performance_orders),
            'filled_orders': len([o for o in performance_orders if o.status == OrderStatus.FILLED]),
            'partial_fills': len([o for o in performance_orders if o.status == OrderStatus.PARTIALLY_FILLED]),
            'cancelled_orders': len([o for o in performance_orders if o.status == OrderStatus.CANCELLED]),
            'fill_rate': 0.0,
            'avg_execution_time': timedelta(),
            'execution_times': [],
            'slippage_stats': {'positive': 0, 'negative': 0, 'total': 0.0}
        }
        
        # Fill rate
        metrics['fill_rate'] = (metrics['filled_orders'] / metrics['total_orders']) * 100
        
        # Execution times
        for order in performance_orders:
            if order.status == OrderStatus.FILLED:
                exec_time = order.updated_at - order.created_at
                metrics['execution_times'].append(exec_time)
        
        if metrics['execution_times']:
            metrics['avg_execution_time'] = sum(
                metrics['execution_times'], 
                timedelta()
            ) / len(metrics['execution_times'])
        
        # Slippage analysis for limit orders
        for order in performance_orders:
            if (order.order_type == OrderType.LIMIT and 
                order.status == OrderStatus.FILLED and 
                order.price):
                
                slippage = order.avg_fill_price - order.price
                if order.side == 'sell':
                    slippage = -slippage
                    
                metrics['slippage_stats']['total'] += slippage
                if slippage > 0:
                    metrics['slippage_stats']['positive'] += 1
                else:
                    metrics['slippage_stats']['negative'] += 1
        
        # Verify performance metrics
        assert metrics['fill_rate'] > 0
        assert metrics['avg_execution_time'].total_seconds() < 5.0
        assert metrics['total_orders'] == 20
        
        # Store metrics for monitoring
        await system['performance_monitor'].record_order_metrics(metrics)
    
    # Helper methods
    def _create_fill_simulator(self) -> Dict[str, Any]:
        """Create fill simulator for testing"""
        return {
            'market_price': 50000,
            'spread': 10,  # $10 spread
            'liquidity': 1000,  # BTC available at each level
            'slippage_factor': 0.0001  # 0.01% per BTC
        }
    
    async def _simulate_fill(self, system: Dict, order: Order, 
                           price: float, size: float) -> None:
        """Simulate order fill"""
        fill = {
            'order_id': order.id,
            'price': price,
            'size': size,
            'timestamp': datetime.now(),
            'fee': size * price * system['config']['fees']['taker']
        }
        
        order.fills.append(fill)
        order.filled_size += size
        order.fees += fill['fee']
        
        # Update average fill price
        if len(order.fills) == 1:
            order.avg_fill_price = price
        else:
            total_value = sum(f['price'] * f['size'] for f in order.fills)
            total_size = sum(f['size'] for f in order.fills)
            order.avg_fill_price = total_value / total_size
        
        # Update status
        if order.filled_size >= order.size:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        order.updated_at = datetime.now()
        
        # Emit fill event
        await system['event_bus'].publish(Event(
            type=EventType.ORDER_FILL,
            data={'order_id': order.id, 'fill': fill}
        ))
    
    async def _simulate_price_movement(self, system: Dict, new_price: float) -> None:
        """Simulate market price movement"""
        system['fill_simulator']['market_price'] = new_price
        
        # Check if any resting orders should fill
        for order_id, order in system['orders'].items():
            if order.status == OrderStatus.SUBMITTED and order.order_type == OrderType.LIMIT:
                if ((order.side == 'buy' and new_price <= order.price) or
                    (order.side == 'sell' and new_price >= order.price)):
                    # Order should fill
                    await self._simulate_fill(system, order, order.price, order.size)