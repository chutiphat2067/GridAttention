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
        'system_coordinator.py',
        'attention_learning_layer.py',
        'market_regime_detector.py',
        'grid_strategy_selector.py',
        'risk_management_system.py',
        'execution_engine.py',
        'performance_monitor.py',
        'overfitting_detector.py',
        'feedback_loop.py'
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
        ('performance_monitor', 'PerformanceMonitor')
    ]
    
    required_methods = [
        'health_check',
        'is_healthy',
        'recover',
        'get_state',
        'load_state'
    ]
    
    for module_name, class_name in component_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                print(f"\n  {class_name}:")
                
                for method in required_methods:
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
    
    # Check special requirements
    print("\n🔐 Checking special requirements:")
    
    # Check kill switch in risk management
    try:
        rm_module = importlib.import_module('risk_management_system')
        if hasattr(rm_module, 'RiskManagementSystem'):
            rm_class = getattr(rm_module, 'RiskManagementSystem')
            # Check in source code
            import inspect
            source = inspect.getsource(rm_class)
            if 'kill_switch' in source:
                print("  ✓ Kill switch in RiskManagementSystem")
            else:
                print("  ✗ Kill switch NOT in RiskManagementSystem")
                issues.append("RiskManagementSystem missing kill_switch")
    except Exception as e:
        print(f"  ✗ Error checking kill switch: {e}")
        issues.append("Could not verify kill_switch")
    
    # Summary
    print(f"\n{'='*50}")
    if not issues:
        print("✅ All validations passed\! System is ready.")
    else:
        print(f"❌ Found {len(issues)} issues:\n")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  Please fix these issues before proceeding.")
    
    return len(issues) == 0

if __name__ == "__main__":
    asyncio.run(validate_system())