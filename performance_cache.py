"""
Performance Cache - Reduce computational overhead
"""
from functools import lru_cache
import time
import hashlib
import json
from typing import Any, Callable, Dict, Optional

class PerformanceCache:
    """Performance caching with TTL (Time To Live)"""
    
    def __init__(self, ttl: int = 5):
        self.cache = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        
    def get_or_compute(self, key: str, compute_func: Callable) -> Any:
        """Get cached value or compute new one"""
        current_time = time.time()
        
        if key in self.cache:
            value, timestamp = self.cache[key]
            if current_time - timestamp < self.ttl:
                self.hits += 1
                return value
                
        # Cache miss - compute new value
        self.misses += 1
        value = compute_func()
        self.cache[key] = (value, current_time)
        
        # Clean old entries periodically
        if len(self.cache) > 1000:
            self._cleanup_old_entries()
            
        return value
        
    def _cleanup_old_entries(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
        
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

class MetricsCache:
    """Specialized cache for performance metrics"""
    
    def __init__(self, ttl: int = 10):
        self.performance_cache = PerformanceCache(ttl)
        self.dashboard_cache = PerformanceCache(ttl * 2)  # Dashboard updates less frequently
        self.health_cache = PerformanceCache(ttl // 2)    # Health checks more frequently
        
    def cache_metrics(self, component_name: str, metrics_func: Callable) -> Any:
        """Cache component metrics"""
        cache_key = f"metrics_{component_name}"
        return self.performance_cache.get_or_compute(cache_key, metrics_func)
        
    def cache_dashboard_data(self, section: str, data_func: Callable) -> Any:
        """Cache dashboard data"""
        cache_key = f"dashboard_{section}"
        return self.dashboard_cache.get_or_compute(cache_key, data_func)
        
    def cache_health_check(self, component_name: str, health_func: Callable) -> Any:
        """Cache health check results"""
        cache_key = f"health_{component_name}"
        return self.health_cache.get_or_compute(cache_key, health_func)
        
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches"""
        return {
            'performance_cache': self.performance_cache.get_stats(),
            'dashboard_cache': self.dashboard_cache.get_stats(),
            'health_cache': self.health_cache.get_stats()
        }

# Global cache instances
perf_cache = PerformanceCache(ttl=5)
metrics_cache = MetricsCache()

# Decorator for easy caching
def cached(ttl: int = 5):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            def compute():
                return func(*args, **kwargs)
                
            cache = PerformanceCache(ttl)
            return cache.get_or_compute(cache_key, compute)
            
        return wrapper
    return decorator

# Memory optimization
def optimize_memory():
    """Clean up memory and optimize performance"""
    import gc
    
    # Clear all caches
    perf_cache.clear()
    metrics_cache.performance_cache.clear()
    metrics_cache.dashboard_cache.clear()
    metrics_cache.health_cache.clear()
    
    # Force garbage collection
    gc.collect()
    
    return {
        'caches_cleared': 4,
        'gc_collected': True
    }