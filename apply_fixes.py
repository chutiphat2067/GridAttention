import asyncio
from essential_fixes import apply_essential_fixes
from pathlib import Path
import importlib
import sys

async def fix_system():
    print("🔧 Applying essential fixes...")
    
    # Import all components
    components = {}
    component_names = [
        'attention_learning_layer',
        'market_regime_detector', 
        'grid_strategy_selector',
        'risk_management_system',
        'execution_engine',
        'performance_monitor',
        'overfitting_detector',
        'feedback_loop'
    ]
    
    for name in component_names:
        try:
            module = importlib.import_module(name)
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            if hasattr(module, class_name):
                components[name] = getattr(module, class_name)({})
                print(f"✓ Loaded {name}")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    # Apply fixes
    fixed_components = apply_essential_fixes(components)
    print(f"\n✅ Fixed {len(fixed_components)} components")
    
    # Test health checks
    print("\n🏥 Testing health checks...")
    for name, comp in fixed_components.items():
        if hasattr(comp, 'health_check'):
            health = await comp.health_check()
            print(f"{name}: {'✓' if health.get('healthy') else '✗'}")

if __name__ == "__main__":
    asyncio.run(fix_system())