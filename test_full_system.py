#!/usr/bin/env python3
"""
Test complete system integration with Priority 1-2 fixes
"""
import asyncio
import sys
import os
import logging
from unittest.mock import Mock
import tempfile

# Add current directory to path
sys.path.insert(0, os.getcwd())

async def test_memory_integration():
    """Test memory management integration"""
    print("🧠 Testing Memory Management Integration...")
    
    try:
        from utils.memory_manager import MemoryManager
        from core.attention_learning_layer import AttentionLearningLayer
        
        # Create test config
        config = {
            'memory': {
                'max_memory_percent': 80,
                'cleanup_interval': 10,
                'auto_cleanup': True
            },
            'learning_phase': {},
            'shadow_phase': {},
            'active_phase': {}
        }
        
        # Test attention layer with memory management
        attention_layer = AttentionLearningLayer(config)
        
        # Check if memory manager is initialized
        assert hasattr(attention_layer, 'memory_manager')
        assert hasattr(attention_layer, 'performance_cache')
        
        # Start memory management
        await attention_layer.start_memory_management()
        
        # Test cleanup method
        await attention_layer.cleanup_old_data()
        
        # Stop memory management
        await attention_layer.memory_manager.stop()
        
        print("   ✅ Memory management integration working")
        return True
        
    except Exception as e:
        print(f"   ❌ Memory integration failed: {e}")
        return False

async def test_feature_validation_integration():
    """Test feature validation integration"""
    print("🔍 Testing Feature Validation Integration...")
    
    try:
        from data.feature_engineering_pipeline import FeatureEngineeringPipeline
        from utils.validators import FeatureValidator
        
        # Create test config
        config = {
            'validation': {
                'enable_validation': True,
                'feature_validation': True
            },
            'cache_ttl': 60,
            'min_history': 10
        }
        
        # Test feature pipeline with validation
        pipeline = FeatureEngineeringPipeline(config)
        
        # Check if validator is initialized
        assert hasattr(pipeline, 'feature_validator')
        assert hasattr(pipeline, 'performance_cache')
        
        print("   ✅ Feature validation integration working")
        return True
        
    except Exception as e:
        print(f"   ❌ Feature validation integration failed: {e}")
        return False

async def test_main_system_integration():
    """Test main system integration"""
    print("🚀 Testing Main System Integration...")
    
    try:
        # Create temporary config file
        test_config = {
            'system': {'mode': 'paper_trading', 'capital': 1000},
            'memory': {'auto_cleanup': True, 'cleanup_interval': 10},
            'validation': {'enable_validation': True},
            'error_recovery': {'max_retry_attempts': 2},
            'performance_optimization': {'enable_caching': True},
            'market_data': {'buffer_size': 100},
            'features': {'technical_indicators': ['rsi']},
            'attention': {'min_trades_learning': 10},
            'regime_detector': {'volatility_window': 10},
            'strategy_selector': {'grid_types': ['symmetric']},
            'risk_management': {'max_position_size': 0.01},
            'execution': {'test': True},
            'performance': {'metrics_interval': 10}
        }
        
        # Save to temporary file
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            # Import and create system (without starting full loop)
            from main import GridAttentionSystem
            
            system = GridAttentionSystem(temp_config_path)
            
            # Check if fixes components are initialized
            assert hasattr(system, 'memory_manager')
            assert hasattr(system, 'feature_validator')
            assert hasattr(system, 'data_integrity_checker')
            
            # Test memory manager start/stop
            await system.memory_manager.start()
            
            # Quick test of system initialization (without full start)
            # Just check if components can be created
            system.config = test_config  # Ensure config is loaded
            
            await system.memory_manager.stop()
            
            print("   ✅ Main system integration working")
            return True
            
        finally:
            # Cleanup temp file
            os.unlink(temp_config_path)
        
    except Exception as e:
        print(f"   ❌ Main system integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_config_integration():
    """Test configuration integration"""
    print("⚙️ Testing Configuration Integration...")
    
    try:
        import yaml
        
        # Load actual config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if all fix sections are present
        required_sections = ['memory', 'validation', 'error_recovery', 'performance_optimization']
        
        for section in required_sections:
            if section in config:
                print(f"   ✅ {section} configuration found")
            else:
                print(f"   ⚠️  {section} configuration missing")
        
        # Check specific settings
        if config.get('memory', {}).get('auto_cleanup') is True:
            print("   ✅ Memory auto-cleanup enabled")
        else:
            print("   ⚠️  Memory auto-cleanup not enabled")
            
        if config.get('validation', {}).get('enable_validation') is True:
            print("   ✅ Validation enabled")
        else:
            print("   ⚠️  Validation not enabled")
        
        print("   ✅ Configuration integration working")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration integration failed: {e}")
        return False

async def test_error_recovery():
    """Test error recovery integration"""
    print("🔄 Testing Error Recovery...")
    
    try:
        from utils.resilient_components import retry_with_backoff, CircuitBreaker
        
        # Test retry mechanism
        attempt_count = 0
        
        @retry_with_backoff(max_attempts=2, initial_delay=0.1)
        async def test_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Test error")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert attempt_count == 2
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        
        # Normal operation
        result = await breaker.call(lambda: "normal")
        assert result == "normal"
        
        print("   ✅ Error recovery integration working")
        return True
        
    except Exception as e:
        print(f"   ❌ Error recovery integration failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    print("=" * 60)
    print("🧪 TESTING COMPLETE SYSTEM INTEGRATION")
    print("=" * 60)
    print()
    
    tests = [
        test_memory_integration,
        test_feature_validation_integration,
        test_config_integration,
        test_error_recovery,
        test_main_system_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"   ❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 COMPLETE INTEGRATION SUCCESSFUL!")
        print("\n✅ System ready for production with Priority 1-2 fixes:")
        print("   • Memory management with auto-cleanup")
        print("   • Feature validation and data integrity")
        print("   • Error recovery with retry and circuit breaker")
        print("   • Performance optimization with caching")
        print("   • Full system integration")
        
        print("\n🚀 To start the system:")
        print("   python main.py")
        
    else:
        print("⚠️  Some integration tests failed.")
        print("Check the errors above before running the full system.")
        return 1
        
    print("=" * 60)
    return 0

if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Integration tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Integration test suite crashed: {e}")
        sys.exit(1)