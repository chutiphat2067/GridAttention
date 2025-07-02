import asyncio
from typing import Dict, List, Callable, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EventBus:
    """Central event bus for component communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = asyncio.Queue()
        self.running = False
        
    async def start(self):
        """Start event processing"""
        self.running = True
        asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop event processing"""
        self.running = False
        
    async def publish(self, event_type: str, data: Any):
        """Publish event to all subscribers"""
        await self.event_queue.put((event_type, data))
        
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type"""
        self.subscribers[event_type].append(handler)
        
    async def _process_events(self):
        """Process events from queue"""
        while self.running:
            try:
                event_type, data = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                
                # Notify all subscribers
                for handler in self.subscribers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data)
                        else:
                            handler(data)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")

# Global event bus instance
event_bus = EventBus()

# Event types
class Events:
    PHASE_CHANGED = "phase_changed"
    REGIME_DETECTED = "regime_detected"
    OVERFITTING_DETECTED = "overfitting_detected"
    RISK_LIMIT_REACHED = "risk_limit_reached"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    COMPONENT_ERROR = "component_error"
    COMPONENT_RECOVERED = "component_recovered"
    PERFORMANCE_UPDATE = "performance_update"
    FEEDBACK_SIGNAL = "feedback_signal"