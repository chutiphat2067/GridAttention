"""
Memory Manager - Bounded buffers and memory management
Prevents memory leaks by limiting buffer sizes and implementing cleanup
"""

import gc
import time
import logging
import psutil
from collections import deque
from typing import Any, List, Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BoundedBuffer:
    """Buffer with size limit and automatic cleanup"""
    
    def __init__(self, max_size: int = 10000, name: str = "buffer"):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.name = name
        self.total_items = 0
        self.created_at = time.time()
        self.last_cleanup = time.time()
        
    def append(self, item: Any):
        """Add item to buffer"""
        self.buffer.append(item)
        self.total_items += 1
        
        # Auto cleanup every 1000 items
        if self.total_items % 1000 == 0:
            self._auto_cleanup()
    
    def extend(self, items: List[Any]):
        """Add multiple items to buffer"""
        for item in items:
            self.append(item)
    
    def get_recent(self, n: int = 100) -> List[Any]:
        """Get n most recent items"""
        return list(self.buffer)[-n:]
    
    def get_all(self) -> List[Any]:
        """Get all items in buffer"""
        return list(self.buffer)
    
    def clear(self):
        """Clear all items"""
        old_size = len(self.buffer)
        self.buffer.clear()
        logger.debug(f"Cleared {self.name} buffer: {old_size} items removed")
    
    def clear_old(self, keep_recent: int = 1000):
        """Keep only recent items"""
        if len(self.buffer) > keep_recent:
            # Convert to list, slice, convert back
            recent_items = list(self.buffer)[-keep_recent:]
            self.buffer = deque(recent_items, maxlen=self.max_size)
            logger.debug(f"Trimmed {self.name} buffer to {keep_recent} items")
    
    def _auto_cleanup(self):
        """Automatic cleanup based on time"""
        now = time.time()
        
        # Cleanup every 10 minutes
        if now - self.last_cleanup > 600:
            if len(self.buffer) > self.max_size * 0.8:  # 80% full
                self.clear_old(keep_recent=int(self.max_size * 0.5))  # Keep 50%
            self.last_cleanup = now
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            'name': self.name,
            'current_size': len(self.buffer),
            'max_size': self.max_size,
            'total_items': self.total_items,
            'memory_usage_mb': self._estimate_memory_usage(),
            'age_hours': (time.time() - self.created_at) / 3600
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if not self.buffer:
            return 0.0
        
        # Sample first item to estimate size
        import sys
        sample_size = sys.getsizeof(self.buffer[0]) if self.buffer else 0
        estimated_mb = (sample_size * len(self.buffer)) / (1024 * 1024)
        return estimated_mb
    
    def __len__(self):
        return len(self.buffer)
    
    def __iter__(self):
        return iter(self.buffer)


class TimedBuffer(BoundedBuffer):
    """Buffer that expires items based on time"""
    
    def __init__(self, max_size: int = 10000, max_age_seconds: int = 3600, name: str = "timed_buffer"):
        super().__init__(max_size, name)
        self.max_age_seconds = max_age_seconds
        self.item_timestamps = deque(maxlen=max_size)
    
    def append(self, item: Any):
        """Add item with timestamp"""
        current_time = time.time()
        super().append(item)
        self.item_timestamps.append(current_time)
        
        # Clean expired items
        self._clean_expired_items()
    
    def _clean_expired_items(self):
        """Remove items older than max_age_seconds"""
        current_time = time.time()
        expired_count = 0
        
        # Remove from front until we find non-expired item
        while (self.item_timestamps and 
               current_time - self.item_timestamps[0] > self.max_age_seconds):
            self.buffer.popleft()
            self.item_timestamps.popleft()
            expired_count += 1
        
        if expired_count > 0:
            logger.debug(f"Expired {expired_count} items from {self.name}")
    
    def get_recent_by_time(self, seconds: int = 300) -> List[Any]:
        """Get items from last N seconds"""
        current_time = time.time()
        cutoff_time = current_time - seconds
        
        result = []
        for i, timestamp in enumerate(self.item_timestamps):
            if timestamp >= cutoff_time:
                result.extend(list(self.buffer)[i:])
                break
                
        return result


class MemoryManager:
    """System-wide memory management"""
    
    def __init__(self):
        self.buffers: Dict[str, BoundedBuffer] = {}
        self.memory_limit_mb = 2048  # 2GB default
        self.cleanup_threshold = 0.8  # Cleanup at 80% memory usage
        self.last_cleanup = time.time()
        
    def create_buffer(self, name: str, max_size: int = 10000, timed: bool = False, **kwargs) -> BoundedBuffer:
        """Create a new bounded buffer"""
        if timed:
            buffer = TimedBuffer(max_size=max_size, name=name, **kwargs)
        else:
            buffer = BoundedBuffer(max_size=max_size, name=name)
        
        self.buffers[name] = buffer
        logger.info(f"Created buffer '{name}' with max_size={max_size}")
        return buffer
    
    def get_buffer(self, name: str) -> Optional[BoundedBuffer]:
        """Get existing buffer"""
        return self.buffers.get(name)
    
    def patch_component_buffers(self, components: Dict[str, Any]):
        """Replace component data structures with bounded buffers"""
        for name, component in components.items():
            try:
                # Market data buffer
                if hasattr(component, 'data_buffer'):
                    if not isinstance(component.data_buffer, BoundedBuffer):
                        old_data = getattr(component.data_buffer, 'data', []) if hasattr(component.data_buffer, 'data') else []
                        component.data_buffer = self.create_buffer(f"{name}_data", max_size=10000)
                        if old_data:
                            component.data_buffer.extend(old_data[-1000:])  # Keep recent 1000
                        logger.info(f"Patched data_buffer for {name}")
                
                # Performance history
                if hasattr(component, 'performance_history'):
                    if not isinstance(component.performance_history, BoundedBuffer):
                        old_data = component.performance_history if isinstance(component.performance_history, list) else []
                        component.performance_history = self.create_buffer(f"{name}_performance", max_size=50000)
                        if old_data:
                            component.performance_history.extend(old_data[-5000:])  # Keep recent 5000
                        logger.info(f"Patched performance_history for {name}")
                
                # Trade history
                if hasattr(component, 'trade_history'):
                    if not isinstance(component.trade_history, BoundedBuffer):
                        old_data = component.trade_history if isinstance(component.trade_history, list) else []
                        component.trade_history = self.create_buffer(f"{name}_trades", max_size=100000)
                        if old_data:
                            component.trade_history.extend(old_data[-10000:])  # Keep recent 10000
                        logger.info(f"Patched trade_history for {name}")
                
                # Log buffer
                if hasattr(component, 'log_buffer'):
                    if not isinstance(component.log_buffer, BoundedBuffer):
                        component.log_buffer = self.create_buffer(f"{name}_logs", max_size=5000, timed=True, max_age_seconds=3600)
                        logger.info(f"Patched log_buffer for {name}")
                        
            except Exception as e:
                logger.warning(f"Failed to patch buffers for {name}: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        usage = self.get_memory_usage()
        
        # Check if exceeding limit
        if usage['rss_mb'] > self.memory_limit_mb:
            logger.warning(f"Memory usage ({usage['rss_mb']:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)")
            return False
        
        # Check if approaching limit
        if usage['percent'] > 85:
            logger.warning(f"High memory usage: {usage['percent']:.1f}%")
            return False
            
        return True
    
    def cleanup_memory(self, force: bool = False):
        """Perform memory cleanup"""
        now = time.time()
        
        # Don't cleanup too frequently unless forced
        if not force and now - self.last_cleanup < 300:  # 5 minutes
            return
        
        logger.info("Starting memory cleanup...")
        initial_usage = self.get_memory_usage()
        
        # Clean all buffers
        for name, buffer in self.buffers.items():
            if len(buffer) > buffer.max_size * 0.5:  # If more than 50% full
                buffer.clear_old(keep_recent=int(buffer.max_size * 0.3))  # Keep 30%
        
        # Force garbage collection
        collected = gc.collect()
        
        # Final memory check
        final_usage = self.get_memory_usage()
        freed_mb = initial_usage['rss_mb'] - final_usage['rss_mb']
        
        logger.info(f"Memory cleanup complete:")
        logger.info(f"  Freed: {freed_mb:.1f}MB")
        logger.info(f"  Objects collected: {collected}")
        logger.info(f"  Current usage: {final_usage['rss_mb']:.1f}MB ({final_usage['percent']:.1f}%)")
        
        self.last_cleanup = now
        
        return {
            'freed_mb': freed_mb,
            'objects_collected': collected,
            'final_usage_mb': final_usage['rss_mb']
        }
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics for all buffers"""
        stats = {}
        total_items = 0
        total_memory_mb = 0
        
        for name, buffer in self.buffers.items():
            buffer_stats = buffer.get_stats()
            stats[name] = buffer_stats
            total_items += buffer_stats['current_size']
            total_memory_mb += buffer_stats['memory_usage_mb']
        
        return {
            'buffers': stats,
            'summary': {
                'total_buffers': len(self.buffers),
                'total_items': total_items,
                'estimated_memory_mb': total_memory_mb
            }
        }
    
    def optimize_for_production(self):
        """Optimize memory settings for production"""
        # Reduce buffer sizes for production
        production_limits = {
            'data': 5000,      # Market data
            'performance': 20000,  # Performance history
            'trades': 50000,   # Trade history
            'logs': 1000       # Log entries
        }
        
        for name, buffer in self.buffers.items():
            for category, limit in production_limits.items():
                if category in name and buffer.max_size > limit:
                    # Reduce buffer size
                    buffer.clear_old(keep_recent=limit)
                    buffer.max_size = limit
                    logger.info(f"Reduced {name} buffer size to {limit}")
    
    def emergency_cleanup(self):
        """Emergency memory cleanup - more aggressive"""
        logger.warning("Emergency memory cleanup initiated!")
        
        # Clear large buffers more aggressively
        for name, buffer in self.buffers.items():
            if len(buffer) > 1000:
                buffer.clear_old(keep_recent=min(500, len(buffer) // 4))  # Keep only 25%
        
        # Multiple garbage collection passes
        for i in range(3):
            collected = gc.collect()
            logger.info(f"Emergency GC pass {i+1}: {collected} objects collected")
        
        usage = self.get_memory_usage()
        logger.info(f"Emergency cleanup complete. Memory usage: {usage['rss_mb']:.1f}MB")


# Global memory manager instance
memory_manager = MemoryManager()


def patch_system_buffers(system):
    """Patch all system components to use bounded buffers"""
    if hasattr(system, 'components'):
        memory_manager.patch_component_buffers(system.components)
    
    # Store reference in system
    system.memory_manager = memory_manager
    
    logger.info("✓ System memory management enabled")
    return memory_manager