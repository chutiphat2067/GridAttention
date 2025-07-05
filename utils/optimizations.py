"""
Performance optimizations for GridAttention system
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from functools import lru_cache, wraps
import numpy as np
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class PerformanceCache:
    """High-performance caching system"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.cache = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hit_count += 1
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return value
            else:
                # Expired
                del self.cache[key]
                
        self.miss_count += 1
        return None
        
    def set(self, key: str, value: Any):
        """Set value in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
            
        self.cache[key] = (value, time.time())
        
        # Evict oldest if needed
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate
        }
        
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


def cached_async(ttl: int = 60):
    """Decorator for caching async function results"""
    def decorator(func):
        cache = PerformanceCache(ttl=ttl)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
                
            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result)
            
            return result
            
        wrapper.cache = cache
        return wrapper
    return decorator


class BatchProcessor:
    """Process items in batches for efficiency"""
    
    def __init__(
        self,
        process_func: Callable,
        batch_size: int = 100,
        batch_timeout: float = 1.0
    ):
        self.process_func = process_func
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_items = []
        self.batch_event = asyncio.Event()
        self.processor_task = None
        
    async def start(self):
        """Start batch processor"""
        self.processor_task = asyncio.create_task(self._process_loop())
        
    async def stop(self):
        """Stop batch processor"""
        if self.processor_task:
            self.processor_task.cancel()
            await asyncio.gather(self.processor_task, return_exceptions=True)
            
    async def add_item(self, item_id: str, item: Any) -> Any:
        """Add item for processing"""
        future = asyncio.Future()
        self.pending_items.append((item_id, item, future))
        
        # Trigger processing if batch is full
        if len(self.pending_items) >= self.batch_size:
            self.batch_event.set()
            
        return await future
        
    async def _process_loop(self):
        """Main processing loop"""
        while True:
            try:
                # Wait for batch or timeout
                await asyncio.wait_for(
                    self.batch_event.wait(),
                    timeout=self.batch_timeout
                )
            except asyncio.TimeoutError:
                pass
                
            if self.pending_items:
                await self._process_batch()
                
            self.batch_event.clear()
            
    async def _process_batch(self):
        """Process current batch"""
        batch = self.pending_items[:self.batch_size]
        self.pending_items = self.pending_items[self.batch_size:]
        
        if not batch:
            return
            
        try:
            # Extract items
            items = [(item_id, item) for item_id, item, _ in batch]
            
            # Process batch
            results = await self.process_func(items)
            
            # Set results
            for i, (item_id, _, future) in enumerate(batch):
                if i < len(results):
                    future.set_result(results[i])
                else:
                    future.set_exception(Exception("No result"))
                    
        except Exception as e:
            # Set exception for all items
            for _, _, future in batch:
                future.set_exception(e)


class OptimizedFeatureCalculator:
    """Optimized feature calculation with vectorization"""
    
    def __init__(self):
        self.buffer_size = 1000
        self.price_buffer = np.zeros(self.buffer_size)
        self.volume_buffer = np.zeros(self.buffer_size)
        self.timestamp_buffer = np.zeros(self.buffer_size)
        self.current_index = 0
        self.is_full = False
        
    def add_tick(self, price: float, volume: float, timestamp: float):
        """Add tick to buffers"""
        self.price_buffer[self.current_index] = price
        self.volume_buffer[self.current_index] = volume
        self.timestamp_buffer[self.current_index] = timestamp
        
        self.current_index = (self.current_index + 1) % self.buffer_size
        if self.current_index == 0:
            self.is_full = True
            
    def calculate_features_vectorized(self) -> Dict[str, float]:
        """Calculate features using vectorized operations"""
        if not self.is_full and self.current_index < 20:
            return {}
            
        # Get valid data
        if self.is_full:
            prices = self.price_buffer
            volumes = self.volume_buffer
        else:
            prices = self.price_buffer[:self.current_index]
            volumes = self.volume_buffer[:self.current_index]
            
        if len(prices) < 2:
            return {}
            
        # Vectorized calculations
        returns = np.diff(prices) / prices[:-1]
        
        features = {
            'price_mean': np.mean(prices),
            'price_std': np.std(prices),
            'volume_mean': np.mean(volumes),
            'volume_std': np.std(volumes),
            'returns_mean': np.mean(returns),
            'returns_std': np.std(returns),
        }
        
        # Safe momentum calculation
        if len(prices) >= 21:
            features['price_momentum'] = (prices[-1] - prices[-21]) / prices[-21]
            features['volume_ratio'] = volumes[-1] / np.mean(volumes[-21:])
            features['volatility'] = np.std(returns[-21:])
        
        # RSI calculation (vectorized)
        if len(returns) > 0:
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 100 if avg_gain > 0 else 50
            
        return features


class ConnectionPool:
    """Connection pool for exchange connections"""
    
    def __init__(self, create_func: Callable, max_size: int = 10):
        self.create_func = create_func
        self.max_size = max_size
        self.available = asyncio.Queue(maxsize=max_size)
        self.in_use = set()
        
    async def acquire(self):
        """Acquire connection from pool"""
        try:
            conn = await asyncio.wait_for(
                self.available.get(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Create new if pool not full
            if len(self.in_use) < self.max_size:
                conn = await self.create_func()
            else:
                # Wait for available connection
                conn = await self.available.get()
                
        self.in_use.add(conn)
        return conn
        
    async def release(self, conn):
        """Release connection back to pool"""
        self.in_use.discard(conn)
        await self.available.put(conn)
        
    async def close_all(self):
        """Close all connections"""
        # Close available connections
        while not self.available.empty():
            conn = await self.available.get()
            if hasattr(conn, 'close'):
                await conn.close()
            
        # Close in-use connections
        for conn in list(self.in_use):
            if hasattr(conn, 'close'):
                await conn.close()
            self.in_use.discard(conn)


class ThreadSafeDict:
    """Thread-safe dictionary wrapper"""
    import threading
    
    def __init__(self):
        self._dict = {}
        self._lock = self.threading.RLock()
        
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._dict.get(key, default)
            
    def set(self, key: str, value: Any):
        with self._lock:
            self._dict[key] = value
            
    def update(self, updates: Dict[str, Any]):
        with self._lock:
            self._dict.update(updates)
            
    def delete(self, key: str):
        with self._lock:
            if key in self._dict:
                del self._dict[key]
                
    def items(self):
        with self._lock:
            return list(self._dict.items())
            
    def clear(self):
        with self._lock:
            self._dict.clear()


class AsyncLockManager:
    """Centralized async lock management"""
    
    def __init__(self):
        self._locks = {}
        self._lock_creation_lock = asyncio.Lock()
        
    async def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create named lock"""
        async with self._lock_creation_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            return self._locks[name]