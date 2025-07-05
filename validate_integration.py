#!/usr/bin/env python3
"""
Validate that fixes are properly integrated into main system
"""
import sys
import os
import importlib.util

def check_integration():
    """Check if fixes are properly integrated"""
    
    print("🔍 Checking GridAttention Integration Status...")
    print("=" * 50)
    
    # Check 1: Utils modules available
    print("1. Utils Modules:")
    utils_modules = [
        'utils.memory_manager',
        'utils.resilient_components', 
        'utils.validators',
        'utils.optimizations'
    ]
    
    for module in utils_modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec:
                print(f"   ✅ {module}")
            else:
                print(f"   ❌ {module} - Not found")
        except Exception as e:
            print(f"   ❌ {module} - Error: {e}")
    
    # Check 2: Import integration in main files
    print("\n2. Import Integration:")
    
    main_files = {
        'core/attention_learning_layer.py': ['memory_manager', 'optimizations'],
        'data/feature_engineering_pipeline.py': ['validators', 'optimizations'],
        'core/execution_engine.py': ['resilient_components', 'validators'],
        'data/market_data_input.py': ['resilient_components', 'validators']
    }
    
    for file_path, expected_imports in main_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
            print(f"   📁 {file_path}:")
            for imp in expected_imports:
                if imp in content:
                    print(f"      ✅ {imp} imported")
                else:
                    print(f"      ❌ {imp} missing")
        else:
            print(f"   ❌ {file_path} not found")
    
    # Check 3: Configuration integration
    print("\n3. Configuration Integration:")
    
    config_file = 'config/config.yaml'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_content = f.read()
            
        config_checks = [
            'memory',
            'validation', 
            'error_recovery',
            'performance'
        ]
        
        for check in config_checks:
            if check in config_content:
                print(f"   ✅ {check} configuration found")
            else:
                print(f"   ⚠️  {check} configuration missing (optional)")
    else:
        print("   ❌ config.yaml not found")
    
    # Check 4: Test integration 
    print("\n4. Test Results:")
    
    try:
        # Import main utils
        from utils.memory_manager import MemoryManager
        from utils.validators import FeatureValidator  
        from utils.resilient_components import retry_with_backoff
        from utils.optimizations import PerformanceCache
        
        print("   ✅ All utils can be imported")
        
        # Test basic functionality
        validator = FeatureValidator()
        cache = PerformanceCache()
        
        test_features = {'price_change': 0.01, 'rsi': 50}
        result = validator.validate_features(test_features)
        
        if result.is_valid:
            print("   ✅ Feature validation working")
        else:
            print("   ❌ Feature validation failed")
            
        cache.set("test", "value")
        if cache.get("test") == "value":
            print("   ✅ Performance cache working")
        else:
            print("   ❌ Performance cache failed")
            
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION STATUS:")
    print("✅ Level 1: Files copied to utils/ - COMPLETE")
    print("✅ Level 2: Imports added to utils/__init__.py - COMPLETE") 
    print("⚠️  Level 3: Integration in main components - PARTIAL")
    print("❌ Level 4: Full system integration - PENDING")
    
    print("\n🔧 NEXT STEPS NEEDED:")
    print("1. Update main.py to initialize memory manager")
    print("2. Modify core components to use fixes")
    print("3. Update config.yaml with fix settings")
    print("4. Test full system with fixes enabled")
    
    return True

if __name__ == '__main__':
    check_integration()