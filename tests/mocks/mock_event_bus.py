#!/usr/bin/env python3
"""
Mock Event Bus for GridAttention Trading System
Provides a realistic event-driven architecture simulation for testing
"""

import asyncio
import time
import uuid
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import inspect
from functools import wraps
import traceback


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in the trading system"""
    # Market events
    MARKET_UPDATE = "market.update"
    PRICE_CHANGE = "market.price_change"
    ORDERBOOK_UPDATE = "market.orderbook_update"
    TRADE_EXECUTED = "market.trade_executed"
    
    # Trading events
    ORDER_PLACED = "trading.order_placed"
    ORDER_FILLED = "trading.order_filled"
    ORDER_CANCELLED = "trading.order_cancelled"
    ORDER_REJECTED = "trading.order_rejected"
    POSITION_OPENED = "trading.position_opened"
    POSITION_CLOSED = "trading.position_closed"
    POSITION_UPDATED = "trading.position_updated"
    
    # Grid events
    GRID_CREATED = "grid.created"
    GRID_LEVEL_FILLED = "grid.level_filled"
    GRID_ADJUSTED = "grid.adjusted"
    GRID_COMPLETED = "grid.completed"
    
    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    COMPONENT_ERROR = "system.component_error"
    HEALTH_CHECK = "system.health_check"
    
    # Risk events
    RISK_ALERT = "risk.alert"
    RISK_LIMIT_REACHED = "risk.limit_reached"
    STOP_LOSS_TRIGGERED = "risk.stop_loss_triggered"
    MARGIN_CALL = "risk.margin_call"
    
    # Performance events
    METRICS_UPDATE = "performance.metrics_update"
    REPORT_GENERATED = "performance.report_generated"
    
    # Learning events
    ATTENTION_PHASE_CHANGE = "learning.phase_change"
    MODEL_UPDATED = "learning.model_updated"
    REGIME_DETECTED = "learning.regime_detected"
    OVERFITTING_DETECTED = "learning.overfitting_detected"


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Event data structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Union[EventType, str] = EventType.SYSTEM_STARTED
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields
    source: Optional[str] = None
    target: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    ttl: Optional[int] = None  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert string to EventType if needed
        if isinstance(self.event_type, str):
            try:
                self.event_type = EventType(self.event_type)
            except ValueError:
                # Keep as string for custom events
                pass
    
    @property
    def is_expired(self) -> bool:
        """Check if event has expired"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "target": self.target,
            "correlation_id": self.correlation_id,
            "priority": self.priority.value,
            "ttl": self.ttl,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict())


@dataclass
class Subscription:
    """Event subscription"""
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_pattern: Union[EventType, str, List[Union[EventType, str]]] = "*"
    handler: Callable = None
    filter_func: Optional[Callable[[Event], bool]] = None
    priority: EventPriority = EventPriority.NORMAL
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    # Statistics
    events_received: int = 0
    events_processed: int = 0
    events_filtered: int = 0
    last_event_time: Optional[datetime] = None
    
    def matches(self, event: Event) -> bool:
        """Check if event matches subscription"""
        if not self.active:
            return False
        
        # Check event type pattern
        if self.event_pattern == "*":
            type_match = True
        elif isinstance(self.event_pattern, list):
            type_match = event.event_type in self.event_pattern
        else:
            type_match = event.event_type == self.event_pattern
        
        if not type_match:
            return False
        
        # Apply custom filter
        if self.filter_func:
            try:
                return self.filter_func(event)
            except Exception:
                return False
        
        return True


class MockEventBus:
    """Mock event bus implementation"""
    
    def __init__(
        self,
        name: str = "MockEventBus",
        enable_persistence: bool = False,
        enable_replay: bool = False,
        max_queue_size: int = 10000
    ):
        self.name = name
        self.enable_persistence = enable_persistence
        self.enable_replay = enable_replay
        self.max_queue_size = max_queue_size
        
        # Event storage
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.event_history: deque = deque(maxlen=10000)
        self.dead_letter_queue: deque = deque(maxlen=1000)
        
        # Subscriptions
        self.subscriptions: Dict[str, Subscription] = {}
        self.subscriptions_by_type: Dict[Union[EventType, str], Set[str]] = defaultdict(set)
        self.wildcard_subscriptions: Set[str] = set()
        
        # Event processing
        self.processing = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_expired": 0,
            "events_filtered": 0,
            "processing_time_total": 0,
            "processing_time_avg": 0
        }
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker()
        
        # Event replay
        self.replay_mode = False
        self.replay_speed = 1.0
        
        # Locks
        self.subscription_lock = threading.RLock()
        
        # Start processing
        self._start_processing()
    
    def _start_processing(self):
        """Start event processing"""
        self.processing = True
        asyncio.create_task(self._process_events())
        asyncio.create_task(self._cleanup_expired_events())
    
    async def _process_events(self):
        """Process events from queue"""
        while self.processing:
            try:
                # Get event from queue
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                # Process event
                await self._dispatch_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    async def _dispatch_event(self, event: Event):
        """Dispatch event to subscribers"""
        start_time = time.time()
        
        # Check if event expired
        if event.is_expired:
            self.stats["events_expired"] += 1
            self.dead_letter_queue.append(event)
            return
        
        # Add to history
        self.event_history.append(event)
        
        # Find matching subscriptions
        matching_subs = self._find_matching_subscriptions(event)
        
        # Sort by priority
        matching_subs.sort(key=lambda s: s.priority.value, reverse=True)
        
        # Dispatch to handlers
        tasks = []
        for subscription in matching_subs:
            subscription.events_received += 1
            subscription.last_event_time = datetime.now()
            
            # Create dispatch task
            task = self._dispatch_to_handler(event, subscription)
            tasks.append(task)
        
        # Wait for all handlers
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update statistics
        self.stats["events_processed"] += 1
        processing_time = time.time() - start_time
        self.stats["processing_time_total"] += processing_time
        self.stats["processing_time_avg"] = (
            self.stats["processing_time_total"] / self.stats["events_processed"]
        )
    
    async def _dispatch_to_handler(self, event: Event, subscription: Subscription):
        """Dispatch event to single handler"""
        try:
            # Check circuit breaker
            if not self.circuit_breaker.is_closed:
                return
            
            # Call handler
            if asyncio.iscoroutinefunction(subscription.handler):
                await subscription.handler(event)
            else:
                # Run sync handler in executor
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    subscription.handler,
                    event
                )
            
            subscription.events_processed += 1
            self.circuit_breaker.record_success()
            
        except Exception as e:
            logger.error(f"Handler error: {e}")
            self.stats["events_failed"] += 1
            self.circuit_breaker.record_failure()
            
            # Send to dead letter queue
            self.dead_letter_queue.append({
                "event": event,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "subscription": subscription.subscription_id
            })
    
    def _find_matching_subscriptions(self, event: Event) -> List[Subscription]:
        """Find subscriptions matching event"""
        matching = []
        
        with self.subscription_lock:
            # Check wildcard subscriptions
            for sub_id in self.wildcard_subscriptions:
                if sub_id in self.subscriptions:
                    sub = self.subscriptions[sub_id]
                    if sub.matches(event):
                        matching.append(sub)
                    else:
                        sub.events_filtered += 1
                        self.stats["events_filtered"] += 1
            
            # Check type-specific subscriptions
            event_type = event.event_type
            if event_type in self.subscriptions_by_type:
                for sub_id in self.subscriptions_by_type[event_type]:
                    if sub_id in self.subscriptions:
                        sub = self.subscriptions[sub_id]
                        if sub.matches(event):
                            matching.append(sub)
                        else:
                            sub.events_filtered += 1
                            self.stats["events_filtered"] += 1
        
        return matching
    
    async def _cleanup_expired_events(self):
        """Clean up expired events periodically"""
        while self.processing:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Clean expired events from history
                current_time = datetime.now()
                self.event_history = deque(
                    (e for e in self.event_history if not e.is_expired),
                    maxlen=10000
                )
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    # Public API
    
    async def publish(
        self,
        event_type: Union[EventType, str],
        data: Dict[str, Any],
        **kwargs
    ) -> str:
        """Publish event"""
        # Create event
        event = Event(
            event_type=event_type,
            data=data,
            **kwargs
        )
        
        # Add to queue
        try:
            await asyncio.wait_for(
                self.event_queue.put(event),
                timeout=1.0
            )
            self.stats["events_published"] += 1
            return event.event_id
            
        except asyncio.TimeoutError:
            logger.error("Event queue full")
            self.dead_letter_queue.append(event)
            raise Exception("Event queue full")
    
    async def publish_batch(self, events: List[Event]) -> List[str]:
        """Publish multiple events"""
        event_ids = []
        
        for event in events:
            try:
                await self.event_queue.put(event)
                event_ids.append(event.event_id)
                self.stats["events_published"] += 1
            except asyncio.QueueFull:
                logger.error(f"Failed to publish event {event.event_id}")
                self.dead_letter_queue.append(event)
        
        return event_ids
    
    def subscribe(
        self,
        event_pattern: Union[EventType, str, List[Union[EventType, str]]],
        handler: Callable,
        filter_func: Optional[Callable[[Event], bool]] = None,
        priority: EventPriority = EventPriority.NORMAL
    ) -> str:
        """Subscribe to events"""
        subscription = Subscription(
            event_pattern=event_pattern,
            handler=handler,
            filter_func=filter_func,
            priority=priority
        )
        
        with self.subscription_lock:
            # Store subscription
            self.subscriptions[subscription.subscription_id] = subscription
            
            # Index by event type
            if event_pattern == "*":
                self.wildcard_subscriptions.add(subscription.subscription_id)
            elif isinstance(event_pattern, list):
                for pattern in event_pattern:
                    self.subscriptions_by_type[pattern].add(subscription.subscription_id)
            else:
                self.subscriptions_by_type[event_pattern].add(subscription.subscription_id)
        
        return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        with self.subscription_lock:
            if subscription_id not in self.subscriptions:
                return False
            
            subscription = self.subscriptions[subscription_id]
            
            # Remove from indexes
            if subscription.event_pattern == "*":
                self.wildcard_subscriptions.discard(subscription_id)
            elif isinstance(subscription.event_pattern, list):
                for pattern in subscription.event_pattern:
                    self.subscriptions_by_type[pattern].discard(subscription_id)
            else:
                self.subscriptions_by_type[subscription.event_pattern].discard(subscription_id)
            
            # Remove subscription
            del self.subscriptions[subscription_id]
            
        return True
    
    def pause_subscription(self, subscription_id: str) -> bool:
        """Pause subscription"""
        with self.subscription_lock:
            if subscription_id in self.subscriptions:
                self.subscriptions[subscription_id].active = False
                return True
        return False
    
    def resume_subscription(self, subscription_id: str) -> bool:
        """Resume subscription"""
        with self.subscription_lock:
            if subscription_id in self.subscriptions:
                self.subscriptions[subscription_id].active = True
                return True
        return False
    
    async def wait_for_event(
        self,
        event_pattern: Union[EventType, str],
        timeout: Optional[float] = None,
        filter_func: Optional[Callable[[Event], bool]] = None
    ) -> Optional[Event]:
        """Wait for specific event"""
        received_event = None
        event_received = asyncio.Event()
        
        async def handler(event: Event):
            nonlocal received_event
            received_event = event
            event_received.set()
        
        # Subscribe temporarily
        sub_id = self.subscribe(event_pattern, handler, filter_func)
        
        try:
            # Wait for event
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return received_event
            
        except asyncio.TimeoutError:
            return None
            
        finally:
            # Unsubscribe
            self.unsubscribe(sub_id)
    
    async def request_response(
        self,
        request_event: Event,
        response_pattern: Union[EventType, str],
        timeout: float = 5.0
    ) -> Optional[Event]:
        """Request-response pattern"""
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request_event.correlation_id = correlation_id
        
        # Wait for response with correlation ID
        response_future = self.wait_for_event(
            response_pattern,
            timeout=timeout,
            filter_func=lambda e: e.correlation_id == correlation_id
        )
        
        # Publish request
        await self.publish(
            request_event.event_type,
            request_event.data,
            correlation_id=correlation_id,
            source=request_event.source,
            target=request_event.target
        )
        
        # Wait for response
        return await response_future
    
    def get_event_history(
        self,
        event_type: Optional[Union[EventType, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get event history"""
        events = list(self.event_history)
        
        # Filter by type
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Filter by time
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Apply limit
        if limit:
            events = events[-limit:]
        
        return events
    
    async def replay_events(
        self,
        events: List[Event],
        speed: float = 1.0,
        real_time: bool = False
    ):
        """Replay historical events"""
        self.replay_mode = True
        self.replay_speed = speed
        
        try:
            if real_time and events:
                # Sort by timestamp
                events.sort(key=lambda e: e.timestamp)
                base_time = events[0].timestamp
                start_real_time = datetime.now()
                
                for event in events:
                    # Calculate delay
                    event_offset = (event.timestamp - base_time).total_seconds()
                    real_offset = (datetime.now() - start_real_time).total_seconds()
                    delay = (event_offset - real_offset) / speed
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
                    
                    # Publish event
                    await self.event_queue.put(event)
            else:
                # Publish all at once
                for event in events:
                    await self.event_queue.put(event)
                    if speed < float('inf'):
                        await asyncio.sleep(1.0 / speed)
                        
        finally:
            self.replay_mode = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        with self.subscription_lock:
            subscription_stats = []
            
            for sub in self.subscriptions.values():
                subscription_stats.append({
                    "subscription_id": sub.subscription_id,
                    "event_pattern": str(sub.event_pattern),
                    "priority": sub.priority.name,
                    "active": sub.active,
                    "events_received": sub.events_received,
                    "events_processed": sub.events_processed,
                    "events_filtered": sub.events_filtered,
                    "last_event_time": sub.last_event_time.isoformat() if sub.last_event_time else None
                })
        
        return {
            "name": self.name,
            "stats": self.stats,
            "queue_size": self.event_queue.qsize(),
            "history_size": len(self.event_history),
            "dead_letter_size": len(self.dead_letter_queue),
            "total_subscriptions": len(self.subscriptions),
            "active_subscriptions": sum(1 for s in self.subscriptions.values() if s.active),
            "subscriptions": subscription_stats,
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
                "success_count": self.circuit_breaker.success_count
            }
        }
    
    def clear_statistics(self):
        """Clear statistics"""
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_expired": 0,
            "events_filtered": 0,
            "processing_time_total": 0,
            "processing_time_avg": 0
        }
        
        with self.subscription_lock:
            for sub in self.subscriptions.values():
                sub.events_received = 0
                sub.events_processed = 0
                sub.events_filtered = 0
    
    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Get dead letter queue contents"""
        return list(self.dead_letter_queue)
    
    def clear_dead_letter_queue(self):
        """Clear dead letter queue"""
        self.dead_letter_queue.clear()
    
    async def flush(self):
        """Flush all pending events"""
        while not self.event_queue.empty():
            await asyncio.sleep(0.1)
    
    async def shutdown(self):
        """Shutdown event bus"""
        self.processing = False
        
        # Wait for queue to empty
        await self.flush()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear subscriptions
        with self.subscription_lock:
            self.subscriptions.clear()
            self.subscriptions_by_type.clear()
            self.wildcard_subscriptions.clear()


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (working)"""
        return self.state == "closed"
    
    def record_success(self):
        """Record successful operation"""
        self.success_count += 1
        
        if self.state == "half_open" and self.success_count >= self.success_threshold:
            self.state = "closed"
            self.failure_count = 0
            self.success_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
        
        # Check for recovery
        if self.state == "open":
            if (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = "half_open"
                self.success_count = 0


# Event decorators

def event_handler(
    event_type: Union[EventType, str],
    priority: EventPriority = EventPriority.NORMAL,
    filter_func: Optional[Callable] = None
):
    """Decorator for event handlers"""
    def decorator(func):
        func._event_handler = True
        func._event_type = event_type
        func._event_priority = priority
        func._event_filter = filter_func
        return func
    return decorator


def async_event_handler(
    event_type: Union[EventType, str],
    priority: EventPriority = EventPriority.NORMAL,
    filter_func: Optional[Callable] = None
):
    """Decorator for async event handlers"""
    def decorator(func):
        @wraps(func)
        async def wrapper(event: Event):
            return await func(event)
        
        wrapper._event_handler = True
        wrapper._event_type = event_type
        wrapper._event_priority = priority
        wrapper._event_filter = filter_func
        return wrapper
    return decorator


class EventBusComponent:
    """Base class for components using event bus"""
    
    def __init__(self, event_bus: MockEventBus):
        self.event_bus = event_bus
        self._subscriptions: List[str] = []
        self._register_handlers()
    
    def _register_handlers(self):
        """Register event handlers from decorated methods"""
        for name, method in inspect.getmembers(self):
            if hasattr(method, '_event_handler'):
                sub_id = self.event_bus.subscribe(
                    method._event_type,
                    method,
                    filter_func=method._event_filter,
                    priority=method._event_priority
                )
                self._subscriptions.append(sub_id)
    
    def cleanup(self):
        """Cleanup subscriptions"""
        for sub_id in self._subscriptions:
            self.event_bus.unsubscribe(sub_id)
        self._subscriptions.clear()


# Helper functions

def create_mock_event_bus(
    enable_persistence: bool = False,
    enable_replay: bool = False
) -> MockEventBus:
    """Create mock event bus instance"""
    return MockEventBus(
        enable_persistence=enable_persistence,
        enable_replay=enable_replay
    )


def create_test_events(count: int = 10) -> List[Event]:
    """Create test events"""
    events = []
    event_types = list(EventType)
    
    for i in range(count):
        event_type = event_types[i % len(event_types)]
        
        event = Event(
            event_type=event_type,
            data={
                "index": i,
                "symbol": "BTC/USDT",
                "price": 45000 + i * 100,
                "timestamp": datetime.now()
            },
            source=f"component_{i % 3}",
            priority=EventPriority(i % 4)
        )
        
        events.append(event)
    
    return events


# Example component using event bus

class TradingComponent(EventBusComponent):
    """Example trading component"""
    
    def __init__(self, event_bus: MockEventBus):
        super().__init__(event_bus)
        self.positions = {}
    
    @async_event_handler(EventType.ORDER_FILLED)
    async def on_order_filled(self, event: Event):
        """Handle order filled event"""
        logger.info(f"Order filled: {event.data}")
        
        # Update positions
        symbol = event.data.get("symbol")
        if symbol:
            self.positions[symbol] = event.data
        
        # Publish position update
        await self.event_bus.publish(
            EventType.POSITION_UPDATED,
            {
                "symbol": symbol,
                "positions": self.positions
            },
            source="TradingComponent"
        )
    
    @event_handler(
        EventType.RISK_ALERT,
        priority=EventPriority.HIGH,
        filter_func=lambda e: e.data.get("severity", "") == "high"
    )
    def on_high_risk_alert(self, event: Event):
        """Handle high severity risk alerts"""
        logger.warning(f"High risk alert: {event.data}")
        
        # Take protective action
        asyncio.create_task(self._close_risky_positions(event.data))
    
    async def _close_risky_positions(self, risk_data: Dict):
        """Close positions based on risk alert"""
        symbol = risk_data.get("symbol")
        if symbol in self.positions:
            await self.event_bus.publish(
                EventType.POSITION_CLOSED,
                {
                    "symbol": symbol,
                    "reason": "risk_alert",
                    "risk_data": risk_data
                },
                source="TradingComponent",
                priority=EventPriority.HIGH
            )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create event bus
        event_bus = create_mock_event_bus()
        
        # Create component
        trading = TradingComponent(event_bus)
        
        # Subscribe to events
        async def market_handler(event: Event):
            print(f"Market update: {event.data}")
        
        sub_id = event_bus.subscribe(
            EventType.MARKET_UPDATE,
            market_handler
        )
        
        # Publish events
        await event_bus.publish(
            EventType.MARKET_UPDATE,
            {"symbol": "BTC/USDT", "price": 45000}
        )
        
        await event_bus.publish(
            EventType.ORDER_FILLED,
            {
                "symbol": "BTC/USDT",
                "order_id": "12345",
                "price": 45000,
                "quantity": 0.1
            }
        )
        
        # Wait for specific event
        risk_event = await event_bus.wait_for_event(
            EventType.RISK_ALERT,
            timeout=2.0
        )
        
        if not risk_event:
            # Publish risk alert
            await event_bus.publish(
                EventType.RISK_ALERT,
                {
                    "symbol": "BTC/USDT",
                    "severity": "high",
                    "message": "Volatility spike detected"
                },
                priority=EventPriority.HIGH
            )
        
        # Request-response pattern
        response = await event_bus.request_response(
            Event(
                event_type="query.position",
                data={"symbol": "BTC/USDT"},
                source="Client"
            ),
            response_pattern="response.position",
            timeout=1.0
        )
        
        # Get statistics
        await asyncio.sleep(1)
        stats = event_bus.get_statistics()
        print(f"\nStatistics: {json.dumps(stats, indent=2)}")
        
        # Replay events
        history = event_bus.get_event_history(limit=5)
        print(f"\nReplaying {len(history)} events...")
        await event_bus.replay_events(history, speed=2.0)
        
        # Cleanup
        event_bus.unsubscribe(sub_id)
        trading.cleanup()
        await event_bus.shutdown()
    
    asyncio.run(main())