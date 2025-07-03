
# tests/integration/test_event_flow.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

from core.event_bus import EventBus, Event, EventType
from core.attention_learning_layer import AttentionLearningLayer
from core.market_regime_detector import MarketRegimeDetector
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine


class TestEventFlow:
    """Test event-driven communication between components"""
    
    @pytest.fixture
    async def event_system(self):
        """Create event-driven system with all components"""
        # Initialize event bus
        event_bus = EventBus()
        
        # Initialize components with event bus
        config = {
            'symbol': 'BTC/USDT',
            'event_bus': event_bus
        }
        
        components = {
            'event_bus': event_bus,
            'attention': AttentionLearningSystem(config),
            'regime': MarketRegimeDetector(config),
            'grid': GridStrategyManager(config),
            'risk': RiskManager(config),
            'execution': ExecutionEngine(config)
        }
        
        # Wire up event handlers
        self._setup_event_handlers(components)
        
        return components
    
    def _setup_event_handlers(self, components):
        """Setup event subscriptions between components"""
        event_bus = components['event_bus']
        
        # Attention → Regime
        event_bus.subscribe(
            EventType.ATTENTION_UPDATE,
            components['regime'].handle_attention_update
        )
        
        # Regime → Grid Strategy
        event_bus.subscribe(
            EventType.REGIME_CHANGE,
            components['grid'].handle_regime_change
        )
        
        # Grid → Risk
        event_bus.subscribe(
            EventType.ORDER_CREATED,
            components['risk'].handle_order_request
        )
        
        # Risk → Execution
        event_bus.subscribe(
            EventType.ORDER_APPROVED,
            components['execution'].handle_approved_order
        )
        
        # Execution → All
        event_bus.subscribe(
            EventType.ORDER_FILLED,
            components['attention'].handle_trade_feedback
        )
    
    @pytest.mark.asyncio
    async def test_market_data_event_flow(self, event_system):
        """Test market data triggering cascade of events"""
        components = event_system
        event_bus = components['event_bus']
        
        # Track events
        events_received = []
        
        async def event_tracker(event):
            events_received.append(event)
        
        # Subscribe tracker to all events
        for event_type in EventType:
            event_bus.subscribe(event_type, event_tracker)
        
        # Publish market data event
        market_data_event = Event(
            type=EventType.MARKET_DATA,
            data={
                'symbol': 'BTC/USDT',
                'price': 50000,
                'volume': 100,
                'timestamp': datetime.now()
            }
        )
        
        await event_bus.publish(market_data_event)
        
        # Allow async processing
        await asyncio.sleep(0.1)
        
        # Verify cascade of events
        event_types = [e.type for e in events_received]
        assert EventType.MARKET_DATA in event_types
        
        # Should trigger attention update
        attention_events = [e for e in events_received if e.type == EventType.ATTENTION_UPDATE]
        assert len(attention_events) > 0
    
    @pytest.mark.asyncio
    async def test_regime_change_event_cascade(self, event_system):
        """Test regime change triggering strategy adjustments"""
        components = event_system
        event_bus = components['event_bus']
        
        # Mock regime change detection
        regime_change_event = Event(
            type=EventType.REGIME_CHANGE,
            data={
                'old_regime': 'ranging',
                'new_regime': 'trending',
                'confidence': 0.85,
                'direction': 'up',
                'timestamp': datetime.now()
            }
        )
        
        # Track grid strategy response
        grid_events = []
        
        async def track_grid_events(event):
            if event.type in [EventType.STRATEGY_UPDATE, EventType.ORDER_CREATED]:
                grid_events.append(event)
        
        event_bus.subscribe(EventType.STRATEGY_UPDATE, track_grid_events)
        event_bus.subscribe(EventType.ORDER_CREATED, track_grid_events)
        
        # Publish regime change
        await event_bus.publish(regime_change_event)
        await asyncio.sleep(0.1)
        
        # Grid should respond with strategy update
        assert len(grid_events) > 0
        strategy_updates = [e for e in grid_events if e.type == EventType.STRATEGY_UPDATE]
        assert len(strategy_updates) > 0
        
        # Verify strategy adapted to trending regime
        update_data = strategy_updates[0].data
        assert update_data['grid_spacing'] != components['grid'].default_spacing
    
    @pytest.mark.asyncio
    async def test_order_approval_event_flow(self, event_system):
        """Test order creation through risk approval to execution"""
        components = event_system
        event_bus = components['event_bus']
        
        # Track order lifecycle
        order_events = []
        
        async def track_orders(event):
            if 'ORDER' in event.type.name:
                order_events.append(event)
        
        for event_type in EventType:
            event_bus.subscribe(event_type, track_orders)
        
        # Create order event
        order_event = Event(
            type=EventType.ORDER_CREATED,
            data={
                'id': 'test_order_1',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'size': 0.01,
                'price': 49500,
                'type': 'limit'
            }
        )
        
        # Mock risk approval
        with patch.object(components['risk'], 'validate_order', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {'approved': True, 'risk_score': 0.3}
            
            await event_bus.publish(order_event)
            await asyncio.sleep(0.1)
        
        # Verify order flow
        created = [e for e in order_events if e.type == EventType.ORDER_CREATED]
        approved = [e for e in order_events if e.type == EventType.ORDER_APPROVED]
        
        assert len(created) == 1
        assert len(approved) == 1
        assert approved[0].data['id'] == 'test_order_1'
    
    @pytest.mark.asyncio
    async def test_trade_execution_feedback_loop(self, event_system):
        """Test trade execution feeding back to learning systems"""
        components = event_system
        event_bus = components['event_bus']
        
        # Track feedback events
        feedback_events = []
        
        async def track_feedback(event):
            if event.type in [EventType.ORDER_FILLED, EventType.PERFORMANCE_UPDATE]:
                feedback_events.append(event)
        
        event_bus.subscribe(EventType.ORDER_FILLED, track_feedback)
        event_bus.subscribe(EventType.PERFORMANCE_UPDATE, track_feedback)
        
        # Simulate order fill
        fill_event = Event(
            type=EventType.ORDER_FILLED,
            data={
                'id': 'test_order_1',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'size': 0.01,
                'price': 49500,
                'filled_at': datetime.now(),
                'commission': 0.001
            }
        )
        
        await event_bus.publish(fill_event)
        await asyncio.sleep(0.1)
        
        # Should trigger performance updates
        assert len(feedback_events) > 0
        fill_events = [e for e in feedback_events if e.type == EventType.ORDER_FILLED]
        assert len(fill_events) == 1
    
    @pytest.mark.asyncio
    async def test_error_event_propagation(self, event_system):
        """Test error events and recovery mechanisms"""
        components = event_system
        event_bus = components['event_bus']
        
        # Track errors
        error_events = []
        
        async def track_errors(event):
            error_events.append(event)
        
        event_bus.subscribe(EventType.ERROR, track_errors)
        
        # Simulate execution error
        error_event = Event(
            type=EventType.ERROR,
            data={
                'source': 'execution_engine',
                'error_type': 'connection_lost',
                'message': 'Exchange connection timeout',
                'timestamp': datetime.now(),
                'context': {'order_id': 'test_order_1'}
            }
        )
        
        await event_bus.publish(error_event)
        await asyncio.sleep(0.1)
        
        # Verify error captured
        assert len(error_events) == 1
        assert error_events[0].data['source'] == 'execution_engine'
    
    @pytest.mark.asyncio
    async def test_event_ordering_and_dependencies(self, event_system):
        """Test correct ordering of dependent events"""
        components = event_system
        event_bus = components['event_bus']
        
        # Track event sequence
        event_sequence = []
        
        async def sequence_tracker(event):
            event_sequence.append({
                'type': event.type,
                'timestamp': datetime.now(),
                'data': event.data
            })
        
        # Subscribe to all events
        for event_type in EventType:
            event_bus.subscribe(event_type, sequence_tracker)
        
        # Trigger complex event chain
        # 1. Market data
        await event_bus.publish(Event(
            type=EventType.MARKET_DATA,
            data={'price': 50000}
        ))
        
        # 2. Should trigger attention update
        await asyncio.sleep(0.05)
        
        # 3. Which should trigger regime check
        await asyncio.sleep(0.05)
        
        # 4. Which might trigger strategy update
        await asyncio.sleep(0.05)
        
        # Verify sequence
        event_types = [e['type'] for e in event_sequence]
        
        # Market data should come first
        assert event_types[0] == EventType.MARKET_DATA
        
        # Dependent events should follow
        if EventType.ATTENTION_UPDATE in event_types:
            attention_idx = event_types.index(EventType.ATTENTION_UPDATE)
            market_idx = event_types.index(EventType.MARKET_DATA)
            assert attention_idx > market_idx
    
    @pytest.mark.asyncio
    async def test_event_fan_out(self, event_system):
        """Test single event triggering multiple handlers"""
        components = event_system
        event_bus = components['event_bus']
        
        # Multiple handlers for same event
        handler_calls = []
        
        async def handler_1(event):
            handler_calls.append(('handler_1', event))
        
        async def handler_2(event):
            handler_calls.append(('handler_2', event))
        
        async def handler_3(event):
            handler_calls.append(('handler_3', event))
        
        # Subscribe all to same event type
        event_bus.subscribe(EventType.MARKET_DATA, handler_1)
        event_bus.subscribe(EventType.MARKET_DATA, handler_2)
        event_bus.subscribe(EventType.MARKET_DATA, handler_3)
        
        # Publish event
        test_event = Event(
            type=EventType.MARKET_DATA,
            data={'test': 'data'}
        )
        
        await event_bus.publish(test_event)
        await asyncio.sleep(0.1)
        
        # All handlers should be called
        assert len(handler_calls) == 3
        handler_names = [call[0] for call in handler_calls]
        assert 'handler_1' in handler_names
        assert 'handler_2' in handler_names
        assert 'handler_3' in handler_names
    
    @pytest.mark.asyncio
    async def test_event_filtering_and_routing(self, event_system):
        """Test event filtering based on criteria"""
        components = event_system
        event_bus = components['event_bus']
        
        # Handler with filter
        filtered_events = []
        
        async def filtered_handler(event):
            # Only handle BTC events
            if event.data.get('symbol') == 'BTC/USDT':
                filtered_events.append(event)
        
        event_bus.subscribe(EventType.ORDER_CREATED, filtered_handler)
        
        # Publish multiple orders
        await event_bus.publish(Event(
            type=EventType.ORDER_CREATED,
            data={'symbol': 'BTC/USDT', 'size': 0.01}
        ))
        
        await event_bus.publish(Event(
            type=EventType.ORDER_CREATED,
            data={'symbol': 'ETH/USDT', 'size': 0.1}
        ))
        
        await event_bus.publish(Event(
            type=EventType.ORDER_CREATED,
            data={'symbol': 'BTC/USDT', 'size': 0.02}
        ))
        
        await asyncio.sleep(0.1)
        
        # Should only handle BTC events
        assert len(filtered_events) == 2
        for event in filtered_events:
            assert event.data['symbol'] == 'BTC/USDT'
    
    @pytest.mark.asyncio
    async def test_event_replay_capability(self, event_system):
        """Test ability to replay events for recovery"""
        components = event_system
        event_bus = components['event_bus']
        
        # Store events for replay
        event_store = []
        
        async def store_events(event):
            event_store.append(event)
        
        # Subscribe to all events
        for event_type in EventType:
            event_bus.subscribe(event_type, store_events)
        
        # Generate events
        original_events = [
            Event(type=EventType.MARKET_DATA, data={'price': 50000}),
            Event(type=EventType.ORDER_CREATED, data={'id': '1'}),
            Event(type=EventType.ORDER_FILLED, data={'id': '1'})
        ]
        
        for event in original_events:
            await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Clear and replay
        replay_count = 0
        
        async def count_replays(event):
            nonlocal replay_count
            replay_count += 1
        
        event_bus.subscribe(EventType.MARKET_DATA, count_replays)
        
        # Replay stored events
        for event in event_store:
            if event.type == EventType.MARKET_DATA:
                await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Should have replayed market data events
        assert replay_count > 0