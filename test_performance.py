"""
Performance Test - Measure system resource usage and optimization
"""
import asyncio
import psutil
import time
import logging
from typing import Dict, Any
from resource_limiter import set_resource_limits, monitor_resource_usage, optimize_memory
from performance_cache import metrics_cache

logging.basicConfig(level=logging.INFO)

async def test_memory_usage():
    """Test memory usage with and without optimizations"""
    print("🧪 Testing Memory Usage...")
    
    # Get baseline
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"   Baseline memory: {mem_before:.1f} MB")
    
    try:
        # Apply resource limits
        set_resource_limits({
            'max_memory_mb': 1024,  # 1GB limit for test
            'max_cpu_cores': 2,
            'process_priority': 10
        })
        
        # Test with minimal config
        from main import GridTradingSystem
        
        print("   Initializing system with minimal config...")
        system = GridTradingSystem('config_minimal.yaml')
        
        # Initialize with monitoring
        await system.initialize()
        
        # Monitor for 30 seconds
        print("   Running for 30 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 30:
            usage = monitor_resource_usage()
            print(f"   Memory: {usage['memory_mb']:.1f}MB, CPU: {usage['cpu_percent']:.1f}%, Threads: {usage['num_threads']}")
            await asyncio.sleep(5)
        
        # Check final memory
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_increase = mem_after - mem_before
        
        print(f"\n📊 Memory Test Results:")
        print(f"   Before: {mem_before:.1f} MB")
        print(f"   After: {mem_after:.1f} MB")
        print(f"   Increase: {memory_increase:.1f} MB")
        
        # Test memory optimization
        print("\n🔧 Testing memory optimization...")
        opt_result = optimize_memory()
        print(f"   Freed: {opt_result['memory_freed_mb']:.1f} MB")
        print(f"   Objects collected: {opt_result['objects_collected']}")
        
        # Memory leak detection
        if memory_increase > 500:  # 500MB increase
            print("   ❌ POTENTIAL MEMORY LEAK DETECTED!")
            return False
        else:
            print("   ✅ Memory usage acceptable")
            
        # Cleanup
        if hasattr(system, 'coordinator') and system.coordinator:
            await system.coordinator.stop()
            
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cache_performance():
    """Test performance cache effectiveness"""
    print("\n🚀 Testing Cache Performance...")
    
    # Test cache with repeated operations
    def expensive_operation():
        time.sleep(0.01)  # Simulate 10ms operation
        return {"result": time.time()}
    
    # Without cache
    start_time = time.time()
    for i in range(100):
        expensive_operation()
    uncached_time = time.time() - start_time
    
    # With cache
    start_time = time.time()
    for i in range(100):
        metrics_cache.cache_metrics("test_component", expensive_operation)
    cached_time = time.time() - start_time
    
    # Get cache stats
    stats = metrics_cache.get_all_stats()
    
    print(f"   Uncached time: {uncached_time:.2f}s")
    print(f"   Cached time: {cached_time:.2f}s")
    print(f"   Speedup: {uncached_time / cached_time:.1f}x")
    print(f"   Cache stats: {stats}")
    
    if cached_time < uncached_time / 2:  # At least 2x speedup
        print("   ✅ Cache performance good")
        return True
    else:
        print("   ⚠️ Cache performance could be better")
        return False

async def test_dashboard_throttling():
    """Test dashboard throttling performance"""
    print("\n📊 Testing Dashboard Throttling...")
    
    try:
        from dashboard_integration import ThrottledDashboardCollector
        
        # Mock grid system
        class MockSystem:
            def __init__(self):
                self.components = {}
                
        mock_system = MockSystem()
        
        # Test normal vs throttled collector
        normal_times = []
        throttled_times = []
        
        # Throttled collector (10s interval)
        throttled_collector = ThrottledDashboardCollector(mock_system, update_interval=10)
        
        # Test multiple rapid calls
        for i in range(5):
            start = time.time()
            try:
                await throttled_collector.collect_all_data()
            except:
                pass  # Expected to fail with mock system
            end = time.time()
            throttled_times.append(end - start)
            await asyncio.sleep(0.1)
        
        avg_throttled_time = sum(throttled_times) / len(throttled_times)
        
        print(f"   Throttled collector avg time: {avg_throttled_time:.3f}s")
        
        # Check if subsequent calls are faster (using cache)
        if len(throttled_times) > 1 and throttled_times[1] < throttled_times[0]:
            print("   ✅ Dashboard throttling working")
            return True
        else:
            print("   ⚠️ Dashboard throttling not effective")
            return False
            
    except Exception as e:
        print(f"   ❌ Dashboard test failed: {e}")
        return False

async def run_performance_tests():
    """Run all performance tests"""
    print("🚀 GridAttention Performance Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Memory usage
    try:
        result1 = await test_memory_usage()
        results.append(result1)
    except Exception as e:
        print(f"Memory test failed: {e}")
        results.append(False)
    
    # Test 2: Cache performance
    try:
        result2 = await test_cache_performance()
        results.append(result2)
    except Exception as e:
        print(f"Cache test failed: {e}")
        results.append(False)
    
    # Test 3: Dashboard throttling
    try:
        result3 = await test_dashboard_throttling()
        results.append(result3)
    except Exception as e:
        print(f"Dashboard test failed: {e}")
        results.append(False)
    
    # Overall results
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Performance Test Results: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL PERFORMANCE TESTS PASSED")
        print("   System optimized for production!")
    elif passed >= total * 0.7:  # 70% pass rate
        print("⚠️ MOST TESTS PASSED")
        print("   System mostly optimized, some issues remain")
    else:
        print("❌ PERFORMANCE ISSUES DETECTED")
        print("   System needs optimization")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = asyncio.run(run_performance_tests())
    exit(0 if success else 1)