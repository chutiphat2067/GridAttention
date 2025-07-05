"""
Memory management utilities for preventing memory leaks in GridAttention
"""
import asyncio
import gc
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import weakref
from collections import deque

logger = logging.getLogger(__name__)

class MemoryManager:
    """Centralized memory management for GridAttention system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_memory_percent = self.config.get('max_memory_percent', 80)
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5 minutes
        self.data_retention = {
            'feature_stats': timedelta(hours=24),
            'performance_history': timedelta(hours=48),
            'tick_buffer': timedelta(minutes=60),
            'order_history': timedelta(days=7)
        }
        self._cleanup_task = None
        self._object_registry = weakref.WeakValueDictionary()
        
    async def start(self):
        """Start memory monitoring and cleanup"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Memory manager started")
        
    async def stop(self):
        """Stop memory manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
            
    def register_object(self, name: str, obj: Any):
        """Register object for memory tracking"""
        self._object_registry[name] = obj
        
    async def _cleanup_loop(self):
        """Main cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                
    async def cleanup_all(self):
        """Perform comprehensive cleanup"""
        start_memory = self.get_memory_usage()
        
        # Cleanup registered objects
        for name, obj in list(self._object_registry.items()):
            if hasattr(obj, 'cleanup_old_data'):
                await obj.cleanup_old_data()
                
        # Force garbage collection
        gc.collect()
        
        end_memory = self.get_memory_usage()
        freed = start_memory - end_memory
        
        logger.info(f"Memory cleanup completed. Freed: {freed:.2f} MB")
        
        # Check if we need emergency cleanup
        if self.get_memory_percent() > self.max_memory_percent:
            await self.emergency_cleanup()
            
    async def emergency_cleanup(self):
        """Emergency memory cleanup when usage is critical"""
        logger.warning("Emergency memory cleanup triggered")
        
        # Clear all caches
        for name, obj in list(self._object_registry.items()):
            if hasattr(obj, 'clear_cache'):
                obj.clear_cache()
                
        # Force full garbage collection
        gc.collect(2)
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def get_memory_percent(self) -> float:
        """Get memory usage percentage"""
        return psutil.virtual_memory().percent


class DataRetentionMixin:
    """Mixin for adding data retention capabilities to classes"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_delta = timedelta(hours=retention_hours)
        self._last_cleanup = datetime.now()
        
    async def cleanup_old_data(self):
        """Clean up data older than retention period"""
        now = datetime.now()
        cutoff_time = now - self.retention_delta
        
        # Cleanup different data structures
        if hasattr(self, 'feature_stats'):
            self._cleanup_dict_with_timestamps(self.feature_stats, cutoff_time)
            
        if hasattr(self, 'performance_history'):
            self._cleanup_deque_with_timestamps(self.performance_history, cutoff_time)
            
        self._last_cleanup = now
        
    def _cleanup_dict_with_timestamps(self, data_dict: Dict, cutoff_time: datetime):
        """Clean up dictionary entries older than cutoff"""
        keys_to_remove = []
        for key, value in data_dict.items():
            if isinstance(value, dict) and 'timestamp' in value:
                if value['timestamp'] < cutoff_time:
                    keys_to_remove.append(key)
                    
        for key in keys_to_remove:
            del data_dict[key]
            
    def _cleanup_deque_with_timestamps(self, data_deque: deque, cutoff_time: datetime):
        """Clean up deque entries older than cutoff"""
        while data_deque and data_deque[0].get('timestamp', datetime.now()) < cutoff_time:
            data_deque.popleft()