"""
Memory Usage Testing Suite for GridAttention Trading System
Tests memory efficiency, leak detection, and optimization
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import psutil
import gc
import tracemalloc
import weakref
import objgraph
import sys
import time
from datetime import datetime, timedelta
from memory_profiler import profile, memory_usage
from pympler import tracker, muppy, summary
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional, Tuple, Any
import logging

# GridAttention imports - aligned with system structure
from src.grid_attention_layer import GridAttentionLayer
from src.attention_learning_layer import AttentionLearningLayer
from src.market_regime_detector import MarketRegimeDetector
from src.grid_strategy_selector import GridStrategySelector
from src.risk_management_system import RiskManagementSystem
from src.execution_engine import ExecutionEngine
from src.performance_monitor import PerformanceMonitor
from src.feedback_loop import FeedbackLoop
from src.overfitting_detector import OverfittingDetector

# Test utilities
from tests.utils.test_helpers import (
    async_test, 
    create_test_config,
    measure_memory_usage
)
from tests.fixtures.market_data import generate_market_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMemoryEfficiency:
    """Test memory efficiency of core components"""
    
    @pytest.fixture
    def memory_tracker(self):
        """Create memory tracker for detailed analysis"""
        return MemoryTracker()
    
    @pytest.fixture
    def grid_attention_system(self):
        """Create GridAttention system with memory tracking"""
        config = create_test_config()
        config['enable_memory_profiling'] = True
        return GridAttentionLayer(config)
    
    @async_test
    async def test_component_memory_footprint(self, memory_tracker):
        """Test memory footprint of individual components"""
        components = {
            'AttentionLearning': AttentionLearningLayer,
            'MarketRegime': MarketRegimeDetector,
            'GridStrategy': GridStrategySelector,
            'RiskManagement': RiskManagementSystem,
            'ExecutionEngine': ExecutionEngine,
            'PerformanceMonitor': PerformanceMonitor,
            'FeedbackLoop': FeedbackLoop,
            'OverfittingDetector': OverfittingDetector
        }
        
        memory_footprints = {}
        
        for name, ComponentClass in components.items():
            # Measure memory before creation
            gc.collect()
            memory_before = memory_tracker.get_current_memory()
            
            # Create component
            config = create_test_config()
            component = ComponentClass(config)
            
            # Measure memory after creation
            memory_after = memory_tracker.get_current_memory()
            memory_used = memory_after - memory_before
            
            memory_footprints[name] = {
                'memory_mb': memory_used,
                'object_count': len(gc.get_objects())
            }
            
            # Cleanup
            del component
            gc.collect()
            
            logger.info(f"{name}: {memory_used:.2f} MB")
        
        # Verify reasonable memory usage
        for name, footprint in memory_footprints.items():
            assert footprint['memory_mb'] < 50, f"{name} uses too much memory"
        
        return memory_footprints
    
    @async_test
    async def test_data_structure_efficiency(self):
        """Test memory efficiency of data structures"""
        # Test different data structure implementations
        num_records = 100000
        
        # Dictionary-based storage
        gc.collect()
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        dict_storage = {}
        for i in range(num_records):
            dict_storage[f'key_{i}'] = {
                'price': 50000 + i,
                'volume': 100 + i,
                'timestamp': time.time() + i
            }
        
        dict_memory = psutil.Process().memory_info().rss / 1024 / 1024 - mem_before
        
        # Clear memory
        dict_storage.clear()
        gc.collect()
        
        # NumPy array storage
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        numpy_storage = np.zeros((num_records, 3))
        for i in range(num_records):
            numpy_storage[i] = [50000 + i, 100 + i, time.time() + i]
        
        numpy_memory = psutil.Process().memory_info().rss / 1024 / 1024 - mem_before
        
        # Compare efficiency
        efficiency_ratio = numpy_memory / dict_memory
        assert efficiency_ratio < 0.5, "NumPy should be more memory efficient"
        
        logger.info(f"Dict memory: {dict_memory:.2f} MB")
        logger.info(f"NumPy memory: {numpy_memory:.2f} MB")
        logger.info(f"Efficiency ratio: {efficiency_ratio:.2f}")
    
    @async_test
    async def test_circular_buffer_memory(self):
        """Test memory usage of circular buffers"""
        buffer_sizes = [1000, 10000, 100000]
        memory_usage_results = []
        
        for size in buffer_sizes:
            # Create circular buffer
            buffer = CircularBuffer(size)
            
            # Fill buffer beyond capacity
            gc.collect()
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            for i in range(size * 2):  # Add 2x capacity
                buffer.append({
                    'index': i,
                    'data': np.random.randn(100)
                })
            
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = mem_after - mem_before
            
            # Verify buffer maintains size limit
            assert len(buffer) == size
            assert memory_used < size * 0.001  # Less than 1KB per item
            
            memory_usage_results.append({
                'size': size,
                'memory_mb': memory_used,
                'per_item_kb': (memory_used * 1024) / size
            })
            
            logger.info(f"Buffer size {size}: {memory_used:.2f} MB "
                       f"({memory_usage_results[-1]['per_item_kb']:.2f} KB/item)")
        
        return memory_usage_results


class TestMemoryLeaks:
    """Test for memory leaks in various scenarios"""
    
    @pytest.fixture(autouse=True)
    def setup_tracemalloc(self):
        """Setup memory tracing"""
        tracemalloc.start()
        yield
        tracemalloc.stop()
    
    @async_test
    async def test_market_update_memory_leak(self, grid_attention_system):
        """Test for memory leaks in market update processing"""
        num_iterations = 1000
        snapshots = []
        
        for i in range(0, num_iterations, 100):
            # Process market updates
            for j in range(100):
                await grid_attention_system.process_market_update({
                    'symbol': 'BTC/USDT',
                    'price': 50000 + np.random.randn(),
                    'volume': 100 + np.random.randn(),
                    'timestamp': time.time()
                })
            
            # Take memory snapshot
            snapshot = tracemalloc.take_snapshot()
            snapshots.append({
                'iteration': i + 100,
                'snapshot': snapshot,
                'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            })
            
            # Force garbage collection
            gc.collect()
        
        # Analyze memory growth
        memory_growth = snapshots[-1]['memory_mb'] - snapshots[0]['memory_mb']
        growth_per_update = memory_growth / num_iterations
        
        # Check for leaks
        assert growth_per_update < 0.001, f"Memory leak detected: {growth_per_update:.6f} MB/update"
        
        # Analyze top memory consumers
        if memory_growth > 10:  # If significant growth
            self._analyze_memory_diff(snapshots[0]['snapshot'], snapshots[-1]['snapshot'])
        
        logger.info(f"Memory growth: {memory_growth:.2f} MB over {num_iterations} updates")
        logger.info(f"Growth per update: {growth_per_update*1000:.3f} KB")
    
    def _analyze_memory_diff(self, snapshot1, snapshot2):
        """Analyze memory difference between snapshots"""
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        logger.info("\nTop memory increases:")
        for stat in top_stats[:10]:
            logger.info(f"{stat}")
    
    @async_test
    async def test_order_execution_memory_leak(self, execution_engine):
        """Test for memory leaks in order execution"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        order_batches = 10
        orders_per_batch = 100
        
        memory_samples = [initial_memory]
        
        for batch in range(order_batches):
            # Execute orders
            orders = []
            for i in range(orders_per_batch):
                order = await execution_engine.submit_order({
                    'id': f'order_{batch}_{i}',
                    'symbol': 'BTC/USDT',
                    'side': 'buy' if i % 2 == 0 else 'sell',
                    'price': 50000 + np.random.uniform(-100, 100),
                    'quantity': 0.1,
                    'type': 'limit'
                })
                orders.append(order)
            
            # Wait for orders to complete
            await asyncio.gather(*orders, return_exceptions=True)
            
            # Clear order history if exists
            if hasattr(execution_engine, 'clear_completed_orders'):
                await execution_engine.clear_completed_orders()
            
            # Sample memory
            gc.collect()
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            logger.info(f"Batch {batch + 1}: Memory = {current_memory:.2f} MB")
        
        # Analyze memory trend
        memory_growth = memory_samples[-1] - memory_samples[0]
        avg_growth_per_batch = memory_growth / order_batches
        
        assert avg_growth_per_batch < 1.0, f"Memory leak in order execution: {avg_growth_per_batch:.2f} MB/batch"
    
    @async_test
    async def test_attention_matrix_memory_leak(self, attention_learning_layer):
        """Test for memory leaks in attention matrix calculations"""
        num_calculations = 100
        matrix_size = 50
        
        # Track weak references to attention matrices
        weak_refs = []
        
        initial_objects = len(gc.get_objects())
        
        for i in range(num_calculations):
            # Generate market data
            data = np.random.randn(matrix_size, matrix_size)
            
            # Calculate attention weights
            attention_weights = await attention_learning_layer.calculate_attention_weights(data)
            
            # Store weak reference
            weak_refs.append(weakref.ref(attention_weights))
            
            # Explicitly delete reference
            del attention_weights
            
            if i % 10 == 0:
                gc.collect()
                alive_refs = sum(1 for ref in weak_refs if ref() is not None)
                logger.info(f"Iteration {i}: {alive_refs} attention matrices still in memory")
        
        # Final cleanup
        gc.collect()
        
        # Check that attention matrices are properly garbage collected
        alive_refs = sum(1 for ref in weak_refs if ref() is not None)
        assert alive_refs == 0, f"{alive_refs} attention matrices leaked"
        
        # Check object count growth
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Too many objects created: {object_growth}"


class TestMemoryOptimization:
    """Test memory optimization strategies"""
    
    @async_test
    async def test_data_compression(self):
        """Test data compression for memory efficiency"""
        # Generate large dataset
        num_records = 100000
        data = []
        
        for i in range(num_records):
            data.append({
                'timestamp': time.time() + i,
                'price': 50000 + np.random.uniform(-100, 100),
                'volume': np.random.uniform(0.1, 100),
                'bid': 49999 + np.random.uniform(-100, 100),
                'ask': 50001 + np.random.uniform(-100, 100)
            })
        
        # Measure uncompressed memory
        uncompressed_size = sys.getsizeof(data) / 1024 / 1024
        
        # Apply compression strategies
        compressed_data = DataCompressor.compress(data)
        compressed_size = sys.getsizeof(compressed_data) / 1024 / 1024
        
        compression_ratio = compressed_size / uncompressed_size
        
        assert compression_ratio < 0.5, f"Compression ratio too high: {compression_ratio:.2f}"
        
        # Verify data integrity
        decompressed_data = DataCompressor.decompress(compressed_data)
        assert len(decompressed_data) == len(data)
        
        logger.info(f"Uncompressed: {uncompressed_size:.2f} MB")
        logger.info(f"Compressed: {compressed_size:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}")
    
    @async_test
    async def test_object_pooling(self):
        """Test object pooling for memory efficiency"""
        pool = ObjectPool(object_type=dict, max_size=100)
        
        # Measure memory without pooling
        gc.collect()
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        objects_no_pool = []
        for i in range(1000):
            obj = {'id': i, 'data': np.random.randn(100)}
            objects_no_pool.append(obj)
        
        mem_no_pool = psutil.Process().memory_info().rss / 1024 / 1024 - mem_before
        objects_no_pool.clear()
        gc.collect()
        
        # Measure memory with pooling
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        objects_with_pool = []
        for i in range(1000):
            obj = pool.acquire()
            obj['id'] = i
            obj['data'] = np.random.randn(100)
            objects_with_pool.append(obj)
            
            # Return to pool when done
            if i > 100:
                old_obj = objects_with_pool.pop(0)
                old_obj.clear()
                pool.release(old_obj)
        
        mem_with_pool = psutil.Process().memory_info().rss / 1024 / 1024 - mem_before
        
        # Pooling should reduce memory usage
        memory_saved = mem_no_pool - mem_with_pool
        assert memory_saved > 0, "Object pooling should save memory"
        
        logger.info(f"Memory without pooling: {mem_no_pool:.2f} MB")
        logger.info(f"Memory with pooling: {mem_with_pool:.2f} MB")
        logger.info(f"Memory saved: {memory_saved:.2f} MB")
    
    @async_test
    async def test_lazy_loading(self):
        """Test lazy loading for memory efficiency"""
        # Create lazy loader for historical data
        lazy_loader = LazyDataLoader('historical_data.csv')
        
        # Initial memory should be minimal
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Access data in chunks
        chunk_memories = []
        for i in range(10):
            chunk = await lazy_loader.get_chunk(i, size=10000)
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            chunk_memories.append(current_memory)
            
            # Process chunk
            await self._process_data_chunk(chunk)
            
            # Release chunk
            del chunk
            gc.collect()
        
        # Memory should not accumulate
        max_memory_growth = max(chunk_memories) - initial_memory
        assert max_memory_growth < 100, f"Memory grew too much: {max_memory_growth:.2f} MB"
        
        logger.info(f"Max memory growth: {max_memory_growth:.2f} MB")
    
    async def _process_data_chunk(self, chunk):
        """Process a data chunk"""
        # Simulate processing
        await asyncio.sleep(0.01)


class TestMemoryProfiling:
    """Profile memory usage of specific operations"""
    
    @profile
    async def test_profile_market_regime_detection(self):
        """Profile memory usage of market regime detection"""
        detector = MarketRegimeDetector(create_test_config())
        
        # Generate market data
        data = generate_market_data('BTC/USDT', periods=10000)
        
        # Profile regime detection
        regime = await detector.detect_regime(data)
        
        # Profile feature calculation
        features = await detector.calculate_features(data)
        
        return regime, features
    
    @async_test
    async def test_memory_usage_by_phase(self, grid_attention_system):
        """Test memory usage in different system phases"""
        phases = ['learning', 'shadow', 'paper', 'live']
        phase_memory = {}
        
        for phase in phases:
            # Switch to phase
            await grid_attention_system.switch_phase(phase)
            
            # Measure memory during operations
            gc.collect()
            memory_samples = []
            
            for i in range(100):
                # Perform phase-specific operations
                await grid_attention_system.process_market_update({
                    'price': 50000 + np.random.randn(),
                    'volume': 100 + np.random.randn()
                })
                
                if i % 10 == 0:
                    memory_samples.append(
                        psutil.Process().memory_info().rss / 1024 / 1024
                    )
            
            phase_memory[phase] = {
                'avg': np.mean(memory_samples),
                'max': max(memory_samples),
                'min': min(memory_samples)
            }
            
            logger.info(f"Phase {phase}: avg={phase_memory[phase]['avg']:.2f} MB, "
                       f"max={phase_memory[phase]['max']:.2f} MB")
        
        # Verify memory usage is reasonable for each phase
        assert phase_memory['learning']['avg'] < phase_memory['live']['avg']
        assert all(pm['max'] < 500 for pm in phase_memory.values())
    
    @async_test
    async def test_memory_hotspots(self):
        """Identify memory hotspots in the system"""
        # Use pympler to track memory allocations
        tr = tracker.SummaryTracker()
        
        # Run typical operations
        system = GridAttentionLayer(create_test_config())
        
        for i in range(100):
            await system.process_market_update({
                'price': 50000 + np.random.randn(),
                'volume': 100
            })
            
            if i % 20 == 0:
                await system.grid_strategy_selector.select_strategy()
                await system.risk_management.calculate_position_risk()
        
        # Get memory summary
        tr.print_diff()
        
        # Analyze object types
        all_objects = muppy.get_objects()
        sum_obj = summary.summarize(all_objects)
        summary.print_(sum_obj)


# Helper Classes

class MemoryTracker:
    """Track memory usage over time"""
    
    def __init__(self):
        self.baseline = psutil.Process().memory_info().rss / 1024 / 1024
        self.samples = []
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def sample(self) -> float:
        """Take a memory sample"""
        current = self.get_current_memory()
        self.samples.append({
            'timestamp': time.time(),
            'memory_mb': current
        })
        return current
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage"""
        if not self.samples:
            return self.get_current_memory()
        return max(s['memory_mb'] for s in self.samples)
    
    def get_average_memory(self) -> float:
        """Get average memory usage"""
        if not self.samples:
            return self.get_current_memory()
        return np.mean([s['memory_mb'] for s in self.samples])


class CircularBuffer:
    """Memory-efficient circular buffer implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.size = 0
    
    def append(self, item: Any):
        """Add item to buffer"""
        self.buffer[self.head] = item
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def __len__(self) -> int:
        return self.size
    
    def get_all(self) -> List[Any]:
        """Get all items in order"""
        if self.size < self.capacity:
            return self.buffer[:self.size]
        else:
            return self.buffer[self.head:] + self.buffer[:self.head]


class DataCompressor:
    """Data compression utilities"""
    
    @staticmethod
    def compress(data: List[Dict]) -> bytes:
        """Compress data for storage"""
        # Simple implementation - in reality would use proper compression
        import pickle
        import zlib
        pickled = pickle.dumps(data)
        return zlib.compress(pickled)
    
    @staticmethod
    def decompress(compressed: bytes) -> List[Dict]:
        """Decompress data"""
        import pickle
        import zlib
        pickled = zlib.decompress(compressed)
        return pickle.loads(pickled)


class ObjectPool:
    """Object pool for reusing objects"""
    
    def __init__(self, object_type, max_size: int = 100):
        self.object_type = object_type
        self.max_size = max_size
        self.pool = []
    
    def acquire(self):
        """Get object from pool or create new"""
        if self.pool:
            return self.pool.pop()
        return self.object_type()
    
    def release(self, obj):
        """Return object to pool"""
        if len(self.pool) < self.max_size:
            self.pool.append(obj)


class LazyDataLoader:
    """Lazy loading for large datasets"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cache = {}
    
    async def get_chunk(self, chunk_id: int, size: int = 10000) -> pd.DataFrame:
        """Load data chunk on demand"""
        if chunk_id in self.cache:
            return self.cache[chunk_id]
        
        # Simulate loading from file
        start_idx = chunk_id * size
        data = pd.DataFrame({
            'timestamp': range(start_idx, start_idx + size),
            'price': np.random.randn(size) * 100 + 50000,
            'volume': np.random.rand(size) * 100
        })
        
        # Cache with size limit
        if len(self.cache) < 5:  # Keep only 5 chunks in memory
            self.cache[chunk_id] = data
        
        return data