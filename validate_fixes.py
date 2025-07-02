import asyncio
import importlib
import sys
from pathlib import Path

async def validate_system():
    print("🔍 Validating system fixes...\n")
    
    issues = []
    
    # Check required files
    required_files = [
        'essential_fixes.py',
        'event_bus.py',
        'attention_learning_layer.py',
        'market_regime_detector.py',
        'grid_strategy_selector.py',
        'risk_management_system.py',
        'execution_engine.py',
        'performance_monitor.py',
        'overfitting_detector.py',
        'feedback_loop.py',
        'market_data_input.py'
    ]
    
    print("📁 Checking files:")
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING")
            issues.append(f"Missing file: {file}")
    
    # Check methods in components
    print("\n🔧 Checking component methods:")
    component_modules = [
        ('attention_learning_layer', 'AttentionLearningLayer'),
        ('market_regime_detector', 'MarketRegimeDetector'),
        ('grid_strategy_selector', 'GridStrategySelector'),
        ('risk_management_system', 'RiskManagementSystem'),
        ('execution_engine', 'ExecutionEngine'),
        ('performance_monitor', 'PerformanceMonitor'),
        ('market_data_input', 'MarketDataInput')
    ]
    
    required_methods = [
        'health_check',
        'is_healthy',
        'recover',
        'get_state',
        'load_state'
    ]
    
    # Special methods for MarketDataInput
    special_methods = {
        'MarketDataInput': ['get_latest_data']
    }
    
    for module_name, class_name in component_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                print(f"\n  {class_name}:")
                
                methods_to_check = required_methods.copy()
                if class_name in special_methods:
                    methods_to_check.extend(special_methods[class_name])
                
                for method in methods_to_check:
                    if hasattr(cls, method):
                        print(f"    ✓ {method}")
                    else:
                        print(f"    ✗ {method} MISSING")
                        issues.append(f"{class_name} missing {method}")
            else:
                print(f"  ✗ {class_name} not found in {module_name}")
                issues.append(f"Class {class_name} not found")
                
        except ImportError as e:
            print(f"  ✗ Failed to import {module_name}: {e}")
            issues.append(f"Import error: {module_name}")
    
    # Test event bus
    print("\n🌐 Testing event bus:")
    try:
        from event_bus import event_bus, Events
        print("  ✓ Event bus imported successfully")
        
        # Test basic functionality
        test_received = False
        def test_handler(data):
            global test_received
            test_received = True
            
        event_bus.subscribe(Events.PHASE_CHANGED, test_handler)
        await event_bus.start()
        await event_bus.publish(Events.PHASE_CHANGED, {'phase': 'test'})
        await asyncio.sleep(0.1)
        await event_bus.stop()
        
        if test_received:
            print("  ✓ Event bus working correctly")
        else:
            print("  ✗ Event bus not working")
            issues.append("Event bus not functioning")
            
    except Exception as e:
        print(f"  ✗ Event bus test failed: {e}")
        issues.append("Event bus test failed")
    
    # Summary
    print(f"\n{'='*50}")
    if not issues:
        print("✅ All validations passed! System is ready.")
    else:
        print(f"❌ Found {len(issues)} issues:\n")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  Please fix these issues before proceeding.")
    
    return len(issues) == 0

if __name__ == "__main__":
    asyncio.run(validate_system())