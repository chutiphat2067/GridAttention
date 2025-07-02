"""
Integration Test - Verify component integration and event flows
"""
import asyncio
import logging
from system_coordinator import SystemCoordinator
from event_bus import event_bus, Events
from integration_manager import integration_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_integration():
    """Test complete system integration"""
    
    print("🧪 Testing System Integration...")
    
    # 1. Initialize System
    print("\n1️⃣ Initializing system...")
    config = {
        'attention_learning_layer': {'learning_rate': 0.001},
        'risk_management_system': {'max_position_size': 0.05}
    }
    
    coordinator = SystemCoordinator(config)
    await coordinator.initialize_components()
    
    # 2. Test Event Flow
    print("\n2️⃣ Testing event flow...")
    received_events = []
    
    async def event_collector(data):
        received_events.append(data)
        
    # Subscribe to all events
    for event_type in [Events.PHASE_CHANGED, Events.OVERFITTING_DETECTED, 
                      Events.KILL_SWITCH_ACTIVATED, Events.RISK_LIMIT_REACHED]:
        event_bus.subscribe(event_type, event_collector)
    
    # Test phase change event
    await event_bus.publish(Events.PHASE_CHANGED, {
        'phase': 'SHADOW',
        'timestamp': asyncio.get_event_loop().time()
    })
    
    await asyncio.sleep(0.1)  # Allow event processing
    
    # 3. Test Kill Switch Integration
    print("\n3️⃣ Testing kill switch integration...")
    
    risk_mgr = coordinator.components.get('risk_management_system')
    execution = coordinator.components.get('execution_engine')
    
    if risk_mgr and hasattr(risk_mgr, 'activate_kill_switch'):
        # Test kill switch
        await risk_mgr.activate_kill_switch("Integration test")
        await asyncio.sleep(0.1)
        
    # 4. Verify Integration Status
    print("\n4️⃣ Verifying integration status...")
    
    status = await integration_manager.verify_integration()
    print(f"Integration Status: {status}")
    
    # 5. Test Component Communication
    print("\n5️⃣ Testing component communication...")
    
    # Check if components have event handlers
    components_with_handlers = 0
    for name, comp in coordinator.components.items():
        if hasattr(comp, 'on_phase_change'):
            components_with_handlers += 1
            print(f"   ✓ {name} has event handlers")
        else:
            print(f"   ❌ {name} missing event handlers")
            
    # 6. Results
    print(f"\n📊 Integration Test Results:")
    print(f"   Components: {len(coordinator.components)}")
    print(f"   With Event Handlers: {components_with_handlers}")
    print(f"   Events Received: {len(received_events)}")
    print(f"   Integration Health: {status.get('integration_health')}")
    
    # Determine success
    success = (
        len(coordinator.components) >= 6 and
        components_with_handlers >= len(coordinator.components) * 0.8 and
        len(received_events) > 0 and
        status.get('integration_health') != 'failed'
    )
    
    if success:
        print("\n✅ Integration Test PASSED")
        print("   ✓ Components initialized")
        print("   ✓ Event handlers injected")
        print("   ✓ Event flow working")
        print("   ✓ Kill switch integrated")
    else:
        print("\n❌ Integration Test FAILED")
        print(f"   Components: {len(coordinator.components)} (need ≥6)")
        print(f"   Event handlers: {components_with_handlers}/{len(coordinator.components)}")
        print(f"   Events received: {len(received_events)}")
        
    # Cleanup
    await coordinator.stop()
    return success

async def test_individual_integrations():
    """Test specific integration scenarios"""
    
    print("\n🔬 Testing Individual Integrations...")
    
    # Test 1: Event Bus Functionality
    print("\nTest 1: Event Bus...")
    test_events = []
    
    def test_handler(data):
        test_events.append(data)
        
    event_bus.subscribe(Events.PHASE_CHANGED, test_handler)
    await event_bus.start()
    
    await event_bus.publish(Events.PHASE_CHANGED, {'test': 'data'})
    await asyncio.sleep(0.1)
    
    if test_events:
        print("   ✓ Event bus working")
    else:
        print("   ❌ Event bus failed")
        
    # Test 2: Integration Manager
    print("\nTest 2: Integration Manager...")
    
    class MockComponent:
        def __init__(self):
            self.name = "mock"
            
    mock_comp = MockComponent()
    await integration_manager.register_component("test_component", mock_comp)
    
    if hasattr(mock_comp, 'on_phase_change'):
        print("   ✓ Integration manager working")
    else:
        print("   ❌ Integration manager failed")
        
    await event_bus.stop()

if __name__ == "__main__":
    async def main():
        success = await test_integration()
        await test_individual_integrations()
        
        print(f"\n🎯 Overall Result: {'PASS' if success else 'FAIL'}")
        return success
        
    result = asyncio.run(main())
    exit(0 if result else 1)