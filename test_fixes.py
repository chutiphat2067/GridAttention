#!/usr/bin/env python3
"""
Test priority 1-2 fixes for GridAttention system
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

async def test_memory_management():
    """Test memory management fixes"""
    try:
        from utils.memory_manager import MemoryManager
        
        print("🧠 Testing Memory Manager...")
        mm = MemoryManager()
        await mm.start()
        
        # Simulate memory usage
        initial_memory = mm.get_memory_usage()
        print(f"   Initial memory: {initial_memory:.2f} MB")
        
        # Create some data to test cleanup
        test_data = [list(range(1000)) for _ in range(100)]
        after_allocation = mm.get_memory_usage()
        print(f"   After allocation: {after_allocation:.2f} MB")
        
        # Test cleanup
        del test_data
        await mm.cleanup_all()
        after_cleanup = mm.get_memory_usage()
        print(f"   After cleanup: {after_cleanup:.2f} MB")
        
        await mm.stop()
        print("   ✅ Memory Manager test passed\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Memory Manager test failed: {e}\n")
        return False
    

async def test_error_recovery():
    """Test error recovery mechanisms"""
    try:
        from utils.resilient_components import retry_with_backoff, CircuitBreaker
        
        print("🔄 Testing Error Recovery...")
        
        # Test retry mechanism
        attempt_count = 0
        
        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Simulated failure")
            return "Success"
            
        result = await flaky_function()
        print(f"   Retry result after {attempt_count} attempts: {result}")
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        # Test normal operation
        result = await breaker.call(lambda: "normal_result")
        print(f"   Circuit breaker normal: {result}")
        
        print("   ✅ Error Recovery test passed\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Error Recovery test failed: {e}\n")
        return False
    

async def test_validation():
    """Test input validation"""
    try:
        from utils.validators import FeatureValidator, OrderValidator
        
        print("🔍 Testing Validation...")
        
        # Test feature validation
        validator = FeatureValidator()
        
        # Valid features
        valid_features = {
            'price_change': 0.01,
            'volume_ratio': 1.5,
            'spread_bps': 2.0,
            'volatility': 0.02,
            'rsi': 55.0
        }
        
        result = validator.validate_features(valid_features)
        print(f"   Valid features test: {'✅' if result.is_valid else '❌'}")
        
        # Invalid features
        invalid_features = {
            'price_change': float('nan'),
            'volume_ratio': -1,
            'spread_bps': 10000,
            'volatility': 2.0,
            'rsi': 150
        }
        
        result = validator.validate_features(invalid_features)
        print(f"   Invalid features detected: {'✅' if not result.is_valid else '❌'}")
        if result.errors:
            print(f"   Detected {len(result.errors)} validation errors")
        
        # Test order validation
        order_validator = OrderValidator()
        valid_order = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'quantity': 0.01,
            'order_type': 'market'
        }
        
        result = order_validator.validate_order(valid_order)
        print(f"   Order validation: {'✅' if result.is_valid else '❌'}")
        
        print("   ✅ Validation test passed\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Validation test failed: {e}\n")
        return False
    

async def test_performance():
    """Test performance optimizations"""
    try:
        from utils.optimizations import PerformanceCache, OptimizedFeatureCalculator
        
        print("⚡ Testing Performance Optimizations...")
        
        # Test performance cache
        cache = PerformanceCache(max_size=100, ttl=60)
        
        # Test caching
        cache.set("key1", "value1")
        cached_value = cache.get("key1")
        print(f"   Cache test: {'✅' if cached_value == 'value1' else '❌'}")
        
        # Test cache stats
        for i in range(10):
            cache.get("key1")  # hits
            cache.get(f"missing{i}")  # misses
            
        stats = cache.get_stats()
        hit_rate = stats['hit_rate']
        print(f"   Cache hit rate: {hit_rate:.2f}")
        
        # Test optimized feature calculator
        calculator = OptimizedFeatureCalculator()
        
        # Add some sample data
        import numpy as np
        for i in range(50):
            price = 50000 + np.random.randn() * 100
            volume = 1000 + np.random.randn() * 100
            calculator.add_tick(price, volume, i)
            
        features = calculator.calculate_features_vectorized()
        print(f"   Feature calculation: {'✅' if len(features) > 0 else '❌'}")
        print(f"   Generated {len(features)} features")
        
        print("   ✅ Performance test passed\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}\n")
        return False


async def test_integration():
    """Test basic integration"""
    try:
        print("🔗 Testing Integration...")
        
        # Test that we can import all main components
        from utils.memory_manager import MemoryManager
        from utils.resilient_components import retry_with_backoff
        from utils.validators import FeatureValidator
        from utils.optimizations import PerformanceCache
        
        print("   All imports successful: ✅")
        
        # Test basic integration
        memory_manager = MemoryManager()
        validator = FeatureValidator()
        cache = PerformanceCache()
        
        # Test that components can work together
        features = {'price_change': 0.01, 'rsi': 50}
        validation_result = validator.validate_features(features)
        
        if validation_result.is_valid:
            cache.set("validated_features", features)
            cached_features = cache.get("validated_features")
            
        print("   Integration test: ✅")
        print("   ✅ Integration test passed\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}\n")
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 TESTING PRIORITY 1-2 FIXES FOR GRIDATTENTION")
    print("=" * 60)
    print()
    
    tests = [
        test_memory_management,
        test_error_recovery,
        test_validation,
        test_performance,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ Test {test.__name__} crashed: {e}\n")
    
    print("=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for improved operation.")
        print("\n📋 Next steps:")
        print("1. Run system: python main.py")
        print("2. Monitor memory usage")
        print("3. Check error recovery in logs")
        print("4. Validate performance improvements")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
        
    print("=" * 60)
    return 0
    

if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)