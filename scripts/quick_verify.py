"""Quick verification without full system start"""
import sys
import traceback

def quick_verify():
    """Quick check of main.py imports and attributes"""
    
    print("⚡ Quick Verification")
    print("=" * 30)
    
    results = {}
    
    try:
        # Test imports
        from main import GridTradingSystem
        print("✅ Main imports successful")
        results['imports'] = True
        
        # Create system object (no initialization)
        system = GridTradingSystem('config/config_production.yaml')
        print("✅ System object created")
        results['creation'] = True
        
        # Check if system has new attributes in __init__
        init_attrs = [
            'unified_monitor',
            'optimized_dashboard_collector', 
            'training_mode',
            'dashboard_enabled'
        ]
        
        found_attrs = 0
        for attr in init_attrs:
            if hasattr(system, attr):
                found_attrs += 1
                print(f"✅ {attr} attribute exists")
            else:
                print(f"❌ {attr} attribute missing")
        
        results['attributes'] = found_attrs / len(init_attrs) >= 0.5
        
        # Check imports in main.py file
        with open('/Users/chutiphatchimphalee/Desktop/GridAttention/main.py', 'r') as f:
            content = f.read()
        
        required_imports = [
            'OptimizedDashboardCollector',
            'patch_system_buffers',
            'unified_monitor'
        ]
        
        found_imports = 0
        for imp in required_imports:
            if imp in content:
                found_imports += 1
                print(f"✅ {imp} import found")
            else:
                print(f"❌ {imp} import missing")
        
        results['file_imports'] = found_imports / len(required_imports) >= 0.8
        
    except Exception as e:
        print(f"❌ Error: {e}")
        results['error'] = str(e)
        traceback.print_exc()
    
    # Summary
    print("\n📊 Quick Results:")
    passed = sum(1 for v in results.values() if v is True)
    total = len([k for k in results.keys() if k != 'error'])
    
    for test, result in results.items():
        if test != 'error':
            status = "✅" if result else "❌"
            print(f"{status} {test.replace('_', ' ').title()}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\n🎯 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = quick_verify()
    print(f"\n{'🎉 Ready for production!' if success else '⚠️ Needs fixes'}")
    sys.exit(0 if success else 1)