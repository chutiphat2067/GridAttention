"""
Unit tests for Cache System.

Tests cover:
- In-memory caching
- Redis cache integration
- Cache key generation
- TTL management
- Cache invalidation strategies
- Cache statistics and monitoring
- Distributed caching
- Cache warming and preloading
"""

import pytest
import time
import json
import pickle
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Any, Dict, List, Optional
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

# GridAttention project imports
from utils.cache import (
    CacheManager,
    MemoryCache,
    RedisCache,
    CacheKey,
    CacheEntry,
    CacheStats,
    CacheConfig,
    InvalidationStrategy,
    CacheDecorator,
    DistributedCache,
    CacheWarmer
)


class TestCacheConfig:
    """Test cases for cache configuration."""
    
    def test_default_config(self):
        """Test default cache configuration."""
        config = CacheConfig()
        
        assert config.max_size == 1000
        assert config.default_ttl == 3600  # 1 hour
        assert config.eviction_policy == 'LRU'
        assert config.enable_statistics is True
        assert config.compression_enabled is False
        
    def test_custom_config(self):
        """Test custom cache configuration."""
        config = CacheConfig(
            max_size=5000,
            default_ttl=7200,
            eviction_policy='LFU',
            compression_enabled=True,
            compression_threshold=1024  # Compress items > 1KB
        )
        
        assert config.max_size == 5000
        assert config.default_ttl == 7200
        assert config.eviction_policy == 'LFU'
        assert config.compression_enabled is True
        assert config.compression_threshold == 1024
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid max size
        with pytest.raises(ValueError, match="max_size must be positive"):
            CacheConfig(max_size=0)
            
        # Invalid TTL
        with pytest.raises(ValueError, match="default_ttl must be positive"):
            CacheConfig(default_ttl=-1)
            
        # Invalid eviction policy
        with pytest.raises(ValueError, match="eviction_policy must be"):
            CacheConfig(eviction_policy='INVALID')


class TestCacheKey:
    """Test cases for cache key generation."""
    
    @pytest.fixture
    def cache_key(self):
        """Create CacheKey instance."""
        return CacheKey()
        
    def test_simple_key_generation(self, cache_key):
        """Test simple cache key generation."""
        # String key
        key1 = cache_key.generate("user", "123")
        assert key1 == "user:123"
        
        # Multiple parts
        key2 = cache_key.generate("order", "BTC/USDT", "2023-01-01")
        assert key2 == "order:BTC/USDT:2023-01-01"
        
        # With prefix
        cache_key.set_prefix("prod")
        key3 = cache_key.generate("user", "123")
        assert key3 == "prod:user:123"
        
    def test_hash_key_generation(self, cache_key):
        """Test hash-based key generation for complex objects."""
        # Dictionary
        data1 = {"symbol": "BTC/USDT", "interval": "5m", "limit": 100}
        key1 = cache_key.generate_hash("candles", data1)
        
        # Same data should generate same key
        data2 = {"symbol": "BTC/USDT", "interval": "5m", "limit": 100}
        key2 = cache_key.generate_hash("candles", data2)
        assert key1 == key2
        
        # Different data should generate different key
        data3 = {"symbol": "ETH/USDT", "interval": "5m", "limit": 100}
        key3 = cache_key.generate_hash("candles", data3)
        assert key1 != key3
        
        # Order shouldn't matter for dicts
        data4 = {"limit": 100, "interval": "5m", "symbol": "BTC/USDT"}
        key4 = cache_key.generate_hash("candles", data4)
        assert key1 == key4
        
    def test_key_pattern_matching(self, cache_key):
        """Test key pattern matching for invalidation."""
        keys = [
            "user:123:profile",
            "user:123:orders",
            "user:456:profile",
            "order:789:details",
            "product:abc:info"
        ]
        
        # Match by prefix
        user_123_keys = cache_key.match_pattern(keys, "user:123:*")
        assert len(user_123_keys) == 2
        assert "user:123:profile" in user_123_keys
        assert "user:123:orders" in user_123_keys
        
        # Match all user keys
        all_user_keys = cache_key.match_pattern(keys, "user:*")
        assert len(all_user_keys) == 3
        
        # Match by suffix
        profile_keys = cache_key.match_pattern(keys, "*:profile")
        assert len(profile_keys) == 2
        
    def test_key_expiry_embedding(self, cache_key):
        """Test embedding expiry information in keys."""
        # Key with daily expiry
        daily_key = cache_key.generate_with_expiry(
            "stats",
            "daily",
            expiry_type="daily"
        )
        
        today = datetime.now().strftime("%Y%m%d")
        assert today in daily_key
        
        # Key with hourly expiry
        hourly_key = cache_key.generate_with_expiry(
            "stats",
            "hourly",
            expiry_type="hourly"
        )
        
        current_hour = datetime.now().strftime("%Y%m%d%H")
        assert current_hour in hourly_key


class TestMemoryCache:
    """Test cases for in-memory cache implementation."""
    
    @pytest.fixture
    def memory_cache(self):
        """Create MemoryCache instance."""
        config = CacheConfig(max_size=100, eviction_policy='LRU')
        return MemoryCache(config)
        
    def test_basic_operations(self, memory_cache):
        """Test basic cache operations."""
        # Set and get
        memory_cache.set("key1", "value1")
        assert memory_cache.get("key1") == "value1"
        
        # Set with TTL
        memory_cache.set("key2", "value2", ttl=1)
        assert memory_cache.get("key2") == "value2"
        
        time.sleep(1.1)
        assert memory_cache.get("key2") is None  # Expired
        
        # Delete
        memory_cache.set("key3", "value3")
        memory_cache.delete("key3")
        assert memory_cache.get("key3") is None
        
        # Exists
        memory_cache.set("key4", "value4")
        assert memory_cache.exists("key4") is True
        assert memory_cache.exists("nonexistent") is False
        
    def test_data_types(self, memory_cache):
        """Test caching different data types."""
        # String
        memory_cache.set("string", "Hello World")
        assert memory_cache.get("string") == "Hello World"
        
        # Number
        memory_cache.set("int", 42)
        memory_cache.set("float", 3.14)
        memory_cache.set("decimal", Decimal("123.45"))
        
        assert memory_cache.get("int") == 42
        assert memory_cache.get("float") == 3.14
        assert memory_cache.get("decimal") == Decimal("123.45")
        
        # List
        memory_cache.set("list", [1, 2, 3, 4, 5])
        cached_list = memory_cache.get("list")
        assert cached_list == [1, 2, 3, 4, 5]
        
        # Dictionary
        data_dict = {"name": "Alice", "age": 30, "active": True}
        memory_cache.set("dict", data_dict)
        assert memory_cache.get("dict") == data_dict
        
        # Complex object
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        memory_cache.set("dataframe", df)
        cached_df = memory_cache.get("dataframe")
        pd.testing.assert_frame_equal(cached_df, df)
        
    def test_lru_eviction(self, memory_cache):
        """Test LRU eviction policy."""
        # Fill cache to capacity
        for i in range(100):
            memory_cache.set(f"key_{i}", f"value_{i}")
            
        # Access some keys to make them recently used
        memory_cache.get("key_0")
        memory_cache.get("key_1")
        memory_cache.get("key_2")
        
        # Add new items to trigger eviction
        memory_cache.set("new_key_1", "new_value_1")
        memory_cache.set("new_key_2", "new_value_2")
        
        # Recently used keys should still exist
        assert memory_cache.get("key_0") is not None
        assert memory_cache.get("key_1") is not None
        assert memory_cache.get("key_2") is not None
        
        # Least recently used should be evicted
        assert memory_cache.get("key_3") is None
        assert memory_cache.get("key_4") is None
        
    def test_lfu_eviction(self):
        """Test LFU eviction policy."""
        config = CacheConfig(max_size=5, eviction_policy='LFU')
        lfu_cache = MemoryCache(config)
        
        # Add items with different access frequencies
        for i in range(5):
            lfu_cache.set(f"key_{i}", f"value_{i}")
            
        # Access keys different number of times
        for _ in range(5):
            lfu_cache.get("key_0")  # Most frequent
        for _ in range(3):
            lfu_cache.get("key_1")
        for _ in range(2):
            lfu_cache.get("key_2")
        # key_3 and key_4 accessed 0 times
        
        # Add new item to trigger eviction
        lfu_cache.set("new_key", "new_value")
        
        # Least frequently used should be evicted
        assert lfu_cache.get("key_0") is not None  # Most frequent
        assert lfu_cache.get("key_1") is not None
        assert lfu_cache.get("key_3") is None or lfu_cache.get("key_4") is None
        
    def test_ttl_expiration(self, memory_cache):
        """Test TTL expiration handling."""
        # Set items with different TTLs
        memory_cache.set("ttl_1", "value_1", ttl=0.5)
        memory_cache.set("ttl_2", "value_2", ttl=1.0)
        memory_cache.set("ttl_3", "value_3", ttl=2.0)
        memory_cache.set("no_ttl", "value_4")  # No TTL
        
        # Check all exist initially
        assert memory_cache.get("ttl_1") is not None
        assert memory_cache.get("ttl_2") is not None
        assert memory_cache.get("ttl_3") is not None
        assert memory_cache.get("no_ttl") is not None
        
        # After 0.6 seconds
        time.sleep(0.6)
        assert memory_cache.get("ttl_1") is None  # Expired
        assert memory_cache.get("ttl_2") is not None
        assert memory_cache.get("ttl_3") is not None
        assert memory_cache.get("no_ttl") is not None
        
        # After 1.1 seconds total
        time.sleep(0.5)
        assert memory_cache.get("ttl_2") is None  # Expired
        assert memory_cache.get("ttl_3") is not None
        assert memory_cache.get("no_ttl") is not None
        
    def test_bulk_operations(self, memory_cache):
        """Test bulk cache operations."""
        # Bulk set
        items = {
            "bulk_1": "value_1",
            "bulk_2": "value_2",
            "bulk_3": "value_3",
            "bulk_4": "value_4"
        }
        
        memory_cache.set_many(items)
        
        # Bulk get
        keys = ["bulk_1", "bulk_2", "bulk_3", "bulk_4", "nonexistent"]
        values = memory_cache.get_many(keys)
        
        assert values["bulk_1"] == "value_1"
        assert values["bulk_2"] == "value_2"
        assert values["bulk_3"] == "value_3"
        assert values["bulk_4"] == "value_4"
        assert values["nonexistent"] is None
        
        # Bulk delete
        memory_cache.delete_many(["bulk_1", "bulk_3"])
        
        assert memory_cache.get("bulk_1") is None
        assert memory_cache.get("bulk_2") is not None
        assert memory_cache.get("bulk_3") is None
        assert memory_cache.get("bulk_4") is not None
        
    def test_clear_operations(self, memory_cache):
        """Test cache clearing operations."""
        # Add some items
        for i in range(10):
            memory_cache.set(f"key_{i}", f"value_{i}")
            
        # Clear all
        memory_cache.clear()
        
        # Check all cleared
        for i in range(10):
            assert memory_cache.get(f"key_{i}") is None
            
        # Add items with patterns
        memory_cache.set("user:123:profile", "profile_data")
        memory_cache.set("user:123:settings", "settings_data")
        memory_cache.set("user:456:profile", "profile_data")
        memory_cache.set("order:789", "order_data")
        
        # Clear by pattern
        memory_cache.clear_pattern("user:123:*")
        
        assert memory_cache.get("user:123:profile") is None
        assert memory_cache.get("user:123:settings") is None
        assert memory_cache.get("user:456:profile") is not None
        assert memory_cache.get("order:789") is not None
        
    def test_thread_safety(self, memory_cache):
        """Test thread safety of cache operations."""
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    
                    memory_cache.set(key, value)
                    retrieved = memory_cache.get(key)
                    
                    if retrieved != value:
                        errors.append(f"Mismatch in thread {thread_id}")
                        
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Error in thread {thread_id}: {e}")
                
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(worker, i)
                futures.append(future)
                
            # Wait for all to complete
            for future in futures:
                future.result()
                
        assert len(errors) == 0
        assert len(results) == 10


class TestRedisCache:
    """Test cases for Redis cache implementation."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = MagicMock()
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = 1
        mock.exists.return_value = 0
        mock.keys.return_value = []
        mock.pipeline.return_value = mock
        mock.execute.return_value = []
        return mock
        
    @pytest.fixture
    def redis_cache(self, mock_redis):
        """Create RedisCache instance with mock."""
        config = CacheConfig()
        with patch('redis.Redis', return_value=mock_redis):
            cache = RedisCache(config, host='localhost', port=6379)
            cache._client = mock_redis
            return cache
            
    def test_basic_redis_operations(self, redis_cache, mock_redis):
        """Test basic Redis cache operations."""
        # Set
        redis_cache.set("key1", "value1")
        mock_redis.set.assert_called_once()
        
        # Get
        mock_redis.get.return_value = b'"value1"'
        value = redis_cache.get("key1")
        assert value == "value1"
        
        # Delete
        redis_cache.delete("key1")
        mock_redis.delete.assert_called_with("key1")
        
        # Exists
        mock_redis.exists.return_value = 1
        assert redis_cache.exists("key1") is True
        
    def test_redis_serialization(self, redis_cache, mock_redis):
        """Test Redis serialization of different data types."""
        # Complex object
        data = {
            "string": "Hello",
            "number": 42,
            "decimal": Decimal("123.45"),
            "list": [1, 2, 3],
            "datetime": datetime.now()
        }
        
        redis_cache.set("complex", data)
        
        # Check serialization
        call_args = mock_redis.set.call_args[0]
        assert call_args[0] == "complex"
        
        # Should be JSON serialized
        serialized = call_args[1]
        assert isinstance(serialized, str)
        
        # Test deserialization
        mock_redis.get.return_value = serialized.encode()
        retrieved = redis_cache.get("complex")
        
        assert retrieved["string"] == data["string"]
        assert retrieved["number"] == data["number"]
        
    def test_redis_pipeline(self, redis_cache, mock_redis):
        """Test Redis pipeline for bulk operations."""
        # Bulk set
        items = {f"key_{i}": f"value_{i}" for i in range(100)}
        
        redis_cache.set_many(items)
        
        # Should use pipeline
        mock_redis.pipeline.assert_called()
        
        # Check pipeline calls
        assert mock_redis.set.call_count >= 100
        mock_redis.execute.assert_called()
        
    def test_redis_ttl(self, redis_cache, mock_redis):
        """Test Redis TTL handling."""
        # Set with TTL
        redis_cache.set("ttl_key", "ttl_value", ttl=3600)
        
        # Check set was called with expiration
        call_args = mock_redis.set.call_args
        assert 'ex' in call_args[1]
        assert call_args[1]['ex'] == 3600
        
    def test_redis_pattern_operations(self, redis_cache, mock_redis):
        """Test Redis pattern-based operations."""
        # Mock keys return
        mock_redis.keys.return_value = [
            b"user:123:profile",
            b"user:123:settings",
            b"user:456:profile"
        ]
        
        # Clear pattern
        redis_cache.clear_pattern("user:123:*")
        
        # Check keys was called with pattern
        mock_redis.keys.assert_called_with("user:123:*")
        
        # Check delete was called for matching keys
        assert mock_redis.delete.call_count >= 2
        
    def test_redis_connection_handling(self, mock_redis):
        """Test Redis connection error handling."""
        # Simulate connection error
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        config = CacheConfig()
        with patch('redis.Redis', return_value=mock_redis):
            with pytest.raises(Exception, match="Connection failed"):
                cache = RedisCache(config)
                cache.connect()


class TestCacheStats:
    """Test cases for cache statistics."""
    
    @pytest.fixture
    def cache_with_stats(self):
        """Create cache with statistics enabled."""
        config = CacheConfig(enable_statistics=True)
        cache = MemoryCache(config)
        return cache
        
    def test_hit_miss_tracking(self, cache_with_stats):
        """Test cache hit/miss tracking."""
        # Some hits
        cache_with_stats.set("key1", "value1")
        cache_with_stats.set("key2", "value2")
        
        cache_with_stats.get("key1")  # Hit
        cache_with_stats.get("key1")  # Hit
        cache_with_stats.get("key2")  # Hit
        cache_with_stats.get("key3")  # Miss
        cache_with_stats.get("key4")  # Miss
        
        stats = cache_with_stats.get_stats()
        
        assert stats.hits == 3
        assert stats.misses == 2
        assert stats.hit_rate == 0.6  # 3/5
        
    def test_operation_timing(self, cache_with_stats):
        """Test operation timing statistics."""
        # Perform operations
        for i in range(100):
            cache_with_stats.set(f"key_{i}", f"value_{i}")
            cache_with_stats.get(f"key_{i}")
            
        stats = cache_with_stats.get_stats()
        
        assert stats.total_gets == 100
        assert stats.total_sets == 100
        assert stats.avg_get_time > 0
        assert stats.avg_set_time > 0
        
    def test_memory_usage_tracking(self, cache_with_stats):
        """Test memory usage statistics."""
        # Add items of different sizes
        cache_with_stats.set("small", "x")
        cache_with_stats.set("medium", "x" * 1000)
        cache_with_stats.set("large", "x" * 10000)
        
        stats = cache_with_stats.get_stats()
        
        assert stats.memory_usage > 0
        assert stats.item_count == 3
        assert stats.avg_item_size > 0
        
    def test_eviction_tracking(self):
        """Test eviction statistics."""
        config = CacheConfig(max_size=5, enable_statistics=True)
        cache = MemoryCache(config)
        
        # Fill cache
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
            
        stats = cache.get_stats()
        
        assert stats.evictions == 5  # 10 - 5
        assert stats.eviction_rate > 0


class TestCacheDecorator:
    """Test cases for cache decorator."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        config = CacheConfig()
        return MemoryCache(config)
        
    def test_function_caching(self, cache):
        """Test caching function results."""
        call_count = 0
        
        @CacheDecorator(cache, ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate expensive operation
            return x + y
            
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Cached call
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not called again
        
        # Different arguments
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
        
    def test_method_caching(self, cache):
        """Test caching class methods."""
        class Calculator:
            def __init__(self):
                self.call_count = 0
                
            @CacheDecorator(cache, ttl=60)
            def multiply(self, x, y):
                self.call_count += 1
                return x * y
                
        calc = Calculator()
        
        # First call
        result1 = calc.multiply(3, 4)
        assert result1 == 12
        assert calc.call_count == 1
        
        # Cached call
        result2 = calc.multiply(3, 4)
        assert result2 == 12
        assert calc.call_count == 1
        
    def test_async_function_caching(self, cache):
        """Test caching async function results."""
        call_count = 0
        
        @CacheDecorator(cache, ttl=60)
        async def async_expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * y
            
        async def test():
            # First call
            result1 = await async_expensive_function(2, 3)
            assert result1 == 6
            assert call_count == 1
            
            # Cached call
            result2 = await async_expensive_function(2, 3)
            assert result2 == 6
            assert call_count == 1
            
        asyncio.run(test())
        
    def test_cache_key_function(self, cache):
        """Test custom cache key generation."""
        @CacheDecorator(
            cache,
            key_func=lambda fn, *args, **kwargs: f"{fn.__name__}:{args[0]}"
        )
        def process_data(data_id, options=None):
            return f"Processed {data_id}"
            
        # Same data_id, different options should use same cache
        result1 = process_data("123", options={"a": 1})
        result2 = process_data("123", options={"b": 2})
        
        assert result1 == result2  # Same cached result
        
    def test_conditional_caching(self, cache):
        """Test conditional caching based on results."""
        @CacheDecorator(
            cache,
            condition=lambda result: result is not None and result > 0
        )
        def compute(x):
            return x - 5
            
        # Should cache positive results
        result1 = compute(10)  # Returns 5
        assert cache.exists(CacheKey().generate_hash("compute", (10,)))
        
        # Should not cache negative results
        result2 = compute(3)  # Returns -2
        assert not cache.exists(CacheKey().generate_hash("compute", (3,)))


class TestInvalidationStrategy:
    """Test cases for cache invalidation strategies."""
    
    @pytest.fixture
    def cache(self):
        """Create cache with invalidation support."""
        config = CacheConfig()
        cache = MemoryCache(config)
        
        # Add test data
        cache.set("user:123:profile", {"name": "Alice"})
        cache.set("user:123:orders", [1, 2, 3])
        cache.set("user:456:profile", {"name": "Bob"})
        cache.set("product:789:details", {"name": "Widget"})
        cache.set("order:111:details", {"total": 100})
        
        return cache
        
    def test_tag_based_invalidation(self, cache):
        """Test tag-based cache invalidation."""
        # Create cache with tagging
        cache.set("tagged1", "value1", tags=["user", "important"])
        cache.set("tagged2", "value2", tags=["user", "temporary"])
        cache.set("tagged3", "value3", tags=["product", "important"])
        
        # Invalidate by tag
        strategy = InvalidationStrategy()
        strategy.invalidate_by_tags(cache, ["user"])
        
        assert cache.get("tagged1") is None
        assert cache.get("tagged2") is None
        assert cache.get("tagged3") is not None  # Different tag
        
    def test_time_based_invalidation(self, cache):
        """Test time-based invalidation."""
        strategy = InvalidationStrategy()
        
        # Set items with different ages
        old_time = time.time() - 3600  # 1 hour ago
        recent_time = time.time() - 300  # 5 minutes ago
        
        cache.set("old_item", "value", _timestamp=old_time)
        cache.set("recent_item", "value", _timestamp=recent_time)
        
        # Invalidate items older than 30 minutes
        strategy.invalidate_older_than(cache, 1800)
        
        assert cache.get("old_item") is None
        assert cache.get("recent_item") is not None
        
    def test_dependency_invalidation(self, cache):
        """Test dependency-based invalidation."""
        strategy = InvalidationStrategy()
        
        # Set up dependencies
        strategy.add_dependency("user:123:*", ["session:abc", "cache:temp"])
        
        # When user data changes, invalidate dependencies
        strategy.invalidate_dependencies(cache, "user:123:profile")
        
        assert cache.get("session:abc") is None
        assert cache.get("cache:temp") is None
        
    def test_cascade_invalidation(self, cache):
        """Test cascading invalidation."""
        strategy = InvalidationStrategy()
        
        # Define cascade rules
        cascade_rules = {
            "user:*": ["order:*", "session:*"],
            "product:*": ["cart:*", "wishlist:*"]
        }
        
        strategy.set_cascade_rules(cascade_rules)
        
        # Invalidate user should cascade
        strategy.cascade_invalidate(cache, "user:123")
        
        assert cache.get("order:111:details") is None  # Cascaded


class TestDistributedCache:
    """Test cases for distributed cache implementation."""
    
    @pytest.fixture
    def distributed_cache(self):
        """Create distributed cache instance."""
        nodes = [
            {"host": "localhost", "port": 6379},
            {"host": "localhost", "port": 6380},
            {"host": "localhost", "port": 6381}
        ]
        
        config = CacheConfig()
        return DistributedCache(config, nodes)
        
    def test_consistent_hashing(self, distributed_cache):
        """Test consistent hashing for key distribution."""
        # Test key distribution
        key_distribution = {}
        
        for i in range(1000):
            key = f"key_{i}"
            node = distributed_cache.get_node(key)
            node_id = f"{node['host']}:{node['port']}"
            
            if node_id not in key_distribution:
                key_distribution[node_id] = 0
            key_distribution[node_id] += 1
            
        # Check relatively even distribution
        values = list(key_distribution.values())
        assert len(values) == 3  # All nodes used
        
        # Each node should have roughly 1/3 of keys (with some variance)
        for count in values:
            assert 200 < count < 500  # Allow for variance
            
    def test_node_failure_handling(self, distributed_cache):
        """Test handling of node failures."""
        # Mark a node as failed
        failed_node = distributed_cache.nodes[0]
        distributed_cache.mark_node_failed(failed_node)
        
        # Keys should redistribute to remaining nodes
        test_keys = [f"key_{i}" for i in range(100)]
        
        for key in test_keys:
            node = distributed_cache.get_node(key)
            assert node != failed_node
            
        # Test node recovery
        distributed_cache.mark_node_recovered(failed_node)
        
        # Some keys should now map to recovered node
        recovered_keys = 0
        for key in test_keys:
            if distributed_cache.get_node(key) == failed_node:
                recovered_keys += 1
                
        assert recovered_keys > 0
        
    def test_replication(self, distributed_cache):
        """Test cache replication across nodes."""
        # Enable replication
        distributed_cache.set_replication_factor(2)
        
        # Set a value
        key = "replicated_key"
        value = "replicated_value"
        
        nodes = distributed_cache.set_with_replication(key, value)
        
        assert len(nodes) == 2  # Primary + 1 replica
        
        # Simulate primary node failure
        primary_node = nodes[0]
        distributed_cache.mark_node_failed(primary_node)
        
        # Should still be able to get from replica
        retrieved = distributed_cache.get(key)
        assert retrieved == value


class TestCacheWarmer:
    """Test cases for cache warming functionality."""
    
    @pytest.fixture
    def cache_warmer(self):
        """Create cache warmer instance."""
        config = CacheConfig()
        cache = MemoryCache(config)
        return CacheWarmer(cache)
        
    def test_warm_from_function(self, cache_warmer):
        """Test warming cache from function results."""
        def data_loader():
            return {
                "key1": "value1",
                "key2": "value2",
                "key3": "value3"
            }
            
        cache_warmer.warm_from_function(data_loader)
        
        # Check cache is warmed
        assert cache_warmer.cache.get("key1") == "value1"
        assert cache_warmer.cache.get("key2") == "value2"
        assert cache_warmer.cache.get("key3") == "value3"
        
    def test_warm_from_database(self, cache_warmer):
        """Test warming cache from database results."""
        # Mock database results
        mock_results = [
            {"id": 1, "name": "Product 1", "price": 100},
            {"id": 2, "name": "Product 2", "price": 200},
            {"id": 3, "name": "Product 3", "price": 300}
        ]
        
        def key_generator(item):
            return f"product:{item['id']}"
            
        cache_warmer.warm_from_data(
            mock_results,
            key_generator=key_generator
        )
        
        # Check cache is warmed
        assert cache_warmer.cache.get("product:1")["name"] == "Product 1"
        assert cache_warmer.cache.get("product:2")["price"] == 200
        
    def test_scheduled_warming(self, cache_warmer):
        """Test scheduled cache warming."""
        refresh_count = 0
        
        def refresh_function():
            nonlocal refresh_count
            refresh_count += 1
            return {"dynamic_key": f"value_{refresh_count}"}
            
        # Schedule warming every 0.5 seconds
        cache_warmer.schedule_warming(
            refresh_function,
            interval=0.5,
            run_immediately=True
        )
        
        # Check immediate run
        assert cache_warmer.cache.get("dynamic_key") == "value_1"
        
        # Wait for scheduled refresh
        time.sleep(0.6)
        cache_warmer.stop_scheduled_warming()
        
        # Should have refreshed at least once
        assert refresh_count >= 2
        
    def test_parallel_warming(self, cache_warmer):
        """Test parallel cache warming for large datasets."""
        # Large dataset
        large_dataset = [
            {"id": i, "data": f"value_{i}"}
            for i in range(1000)
        ]
        
        def process_item(item):
            # Simulate processing
            time.sleep(0.001)
            return f"item:{item['id']}", item
            
        # Warm cache in parallel
        cache_warmer.warm_parallel(
            large_dataset,
            process_item,
            max_workers=10
        )
        
        # Check random samples
        assert cache_warmer.cache.get("item:0")["data"] == "value_0"
        assert cache_warmer.cache.get("item:500")["data"] == "value_500"
        assert cache_warmer.cache.get("item:999")["data"] == "value_999"


class TestCacheCompression:
    """Test cases for cache compression."""
    
    @pytest.fixture
    def compressed_cache(self):
        """Create cache with compression enabled."""
        config = CacheConfig(
            compression_enabled=True,
            compression_threshold=100  # Compress items > 100 bytes
        )
        return MemoryCache(config)
        
    def test_automatic_compression(self, compressed_cache):
        """Test automatic compression of large values."""
        # Small value (not compressed)
        small_value = "Small"
        compressed_cache.set("small", small_value)
        
        # Large value (compressed)
        large_value = "x" * 1000
        compressed_cache.set("large", large_value)
        
        # Both should retrieve correctly
        assert compressed_cache.get("small") == small_value
        assert compressed_cache.get("large") == large_value
        
        # Check internal storage
        small_entry = compressed_cache._get_entry("small")
        large_entry = compressed_cache._get_entry("large")
        
        assert not small_entry.is_compressed
        assert large_entry.is_compressed
        assert large_entry.compressed_size < len(large_value)
        
    def test_compression_ratio(self, compressed_cache):
        """Test compression ratio tracking."""
        # Highly compressible data
        repetitive_data = "AAAA" * 1000
        compressed_cache.set("repetitive", repetitive_data)
        
        entry = compressed_cache._get_entry("repetitive")
        compression_ratio = len(repetitive_data) / entry.compressed_size
        
        assert compression_ratio > 10  # Should compress well
        
        # Random data (poor compression)
        random_data = ''.join(chr(i % 256) for i in range(1000))
        compressed_cache.set("random", random_data)
        
        entry = compressed_cache._get_entry("random")
        compression_ratio = len(random_data) / entry.compressed_size
        
        assert compression_ratio < 2  # Poor compression