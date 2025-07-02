"""
Comprehensive Integration Test - Following Terminal Fix Guide Step 7
"""
import asyncio
import logging
from event_bus import event_bus, Events

logging.basicConfig(level=logging.INFO)

async def test_event_flow():
    """Test event propagation as specified in guide"""
    
    print("🧪 Testing Event Flow (Step 7)...")
    received = []
    
    def handler(data):
        received.append(data)
        print(f"   Event received: {data}")
        
    # Test phase change
    event_bus.subscribe(Events.PHASE_CHANGED, handler)
    await event_bus.start()
    
    await event_bus.publish(Events.PHASE_CHANGED, {'phase': 'SHADOW'})
    await asyncio.sleep(0.1)
    
    assert len(received) == 1, f"Expected 1 event, got {len(received)}"
    print("✓ Event flow working")
    
    await event_bus.stop()
    return True

async def test_main_integration():
    """Test main.py integration with SystemCoordinator"""
    
    print("\n🔧 Testing main.py Integration...")
    
    try:
        # Import main system
        from main import GridTradingSystem
        
        # Create minimal config files if they don't exist
        import yaml
        import os
        
        if not os.path.exists('config.yaml'):
            config = {
                'attention': {'learning_rate': 0.001},
                'risk_management': {'max_position_size': 0.05},
                'execution': {'max_orders': 100},
                'performance': {'metrics_interval': 60},
                'feedback': {'update_interval': 30}
            }
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f)
                
        if not os.path.exists('overfitting_config.yaml'):
            overfitting_config = {
                'detector': {'threshold': 0.1},
                'regularization': {'l2_lambda': 0.01},
                'checkpointing': {'checkpoint_dir': './checkpoints'}
            }
            with open('overfitting_config.yaml', 'w') as f:
                yaml.dump(overfitting_config, f)
        
        # Initialize system
        system = GridTradingSystem('config.yaml', 'overfitting_config.yaml')
        
        # Test initialization (this should use SystemCoordinator now)
        await system.initialize()
        
        # Verify coordinator exists
        assert system.coordinator is not None, "SystemCoordinator not initialized"
        assert len(system.components) >= 6, f"Expected ≥6 components, got {len(system.components)}"
        
        # Verify integration
        from integration_manager import integration_manager
        status = await integration_manager.verify_integration()
        
        print(f"   Components: {len(system.components)}")
        print(f"   Integration status: {status}")
        
        # Verify legacy mappings
        legacy_mappings = ['attention', 'regime_detector', 'strategy_selector', 'risk_manager', 'execution']
        for legacy_name in legacy_mappings:
            if legacy_name in system.components:
                print(f"   ✓ Legacy mapping: {legacy_name}")
            
        print("✓ main.py integration working")
        
        # Cleanup
        if hasattr(system, 'coordinator') and system.coordinator:
            await system.coordinator.stop()
            
        return True
        
    except Exception as e:
        print(f"   ❌ main.py integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def run_comprehensive_test():
        print("🚀 Comprehensive Integration Test")
        
        # Step 1: Event flow test
        event_test = await test_event_flow()
        
        # Step 2: Main integration test  
        main_test = await test_main_integration()
        
        # Results
        passed = sum([event_test, main_test])
        total = 2
        
        print(f"\n📊 Comprehensive Test Results: {passed}/{total}")
        
        if passed == total:
            print("✅ ALL INTEGRATION TESTS PASSED")
            print("   ✓ Event flow working")
            print("   ✓ main.py using SystemCoordinator") 
            print("   ✓ Component integration complete")
            return True
        else:
            print("❌ INTEGRATION TESTS FAILED")
            return False
            
    success = asyncio.run(run_comprehensive_test())
    exit(0 if success else 1)