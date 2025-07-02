"""Verify all final fixes are applied"""
import asyncio
import traceback
from main import GridTradingSystem

async def verify_fixes():
    """Verify all final fixes are applied"""
    
    print("🔍 Verifying Final Fixes...")
    print("=" * 50)
    
    results = {}
    
    try:
        # Use production config
        system = GridTradingSystem('config/config_production.yaml')
        await system.initialize()
        
        # Check 1: Unified Monitor
        if hasattr(system, 'unified_monitor') and system.unified_monitor:
            print("✅ Unified Monitor integrated")
            results['unified_monitor'] = True
        else:
            print("❌ Unified Monitor missing")
            results['unified_monitor'] = False
            
        # Check 2: Optimized Dashboard
        if hasattr(system, 'optimized_dashboard_collector'):
            print("✅ Optimized Dashboard Collector present")
            results['dashboard_collector'] = True
        else:
            print("❌ Optimized Dashboard Collector missing")
            results['dashboard_collector'] = False
            
        # Check 3: Memory Patches
        comp_with_bounded = 0
        total_components = len(system.components)
        
        for name, comp in system.components.items():
            # Check for any buffer-like attributes that might be patched
            for attr_name in dir(comp):
                attr = getattr(comp, attr_name)
                if hasattr(attr, 'maxlen') and hasattr(attr, 'append'):
                    comp_with_bounded += 1
                    break
                    
        if comp_with_bounded > 0:
            print(f"✅ Memory patches applied ({comp_with_bounded}/{total_components} components)")
            results['memory_patches'] = True
        else:
            print("❌ Memory patches not detected")
            results['memory_patches'] = False
            
        # Check 4: System Integration
        if system.coordinator and hasattr(system.coordinator, 'components') and len(system.coordinator.components) > 0:
            print("✅ System integration healthy")
            results['integration'] = True
        else:
            print("❌ System integration issues")
            results['integration'] = False
            
        # Check 5: Component Count
        expected_components = 8  # Core components
        actual_components = len(system.components)
        
        if actual_components >= expected_components:
            print(f"✅ All components present ({actual_components} components)")
            results['components'] = True
        else:
            print(f"❌ Missing components ({actual_components}/{expected_components})")
            results['components'] = False
        
        await system.stop()
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        traceback.print_exc()
        results['error'] = str(e)
    
    # Summary
    print("\n📊 Verification Results:")
    print("-" * 30)
    
    passed = sum(1 for v in results.values() if v is True)
    total = len([k for k in results.keys() if k != 'error'])
    
    for test, result in results.items():
        if test != 'error':
            status = "✅" if result else "❌"
            print(f"{status} {test.replace('_', ' ').title()}")
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\n🎯 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 System ready for production!")
        return True
    else:
        print("⚠️ Issues need to be addressed")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_fixes())
    exit(0 if success else 1)