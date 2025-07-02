"""
Quick Integration Test - Focus on event integration without full system startup
"""
import asyncio
import logging
from event_bus import event_bus, Events
from integration_manager import integration_manager

logging.basicConfig(level=logging.INFO)

async def quick_test():
    """Quick test of integration components"""
    
    print("🚀 Quick Integration Test")
    
    # Test 1: Event Bus
    print("\n1️⃣ Testing Event Bus...")
    received = []
    
    async def handler(data):
        received.append(data)
        print(f"   Received: {data}")
    
    event_bus.subscribe(Events.PHASE_CHANGED, handler)
    await event_bus.start()
    
    await event_bus.publish(Events.PHASE_CHANGED, {'phase': 'SHADOW'})
    await asyncio.sleep(0.2)  # More time for processing
    
    if received:
        print("   ✅ Event Bus: WORKING")
    else:
        print("   ❌ Event Bus: FAILED")
    
    # Test 2: Integration Manager
    print("\n2️⃣ Testing Integration Manager...")
    
    class MockComponent:
        def __init__(self, name):
            self.name = name
            self.current_phase = None
    
    mock = MockComponent("test")
    await integration_manager.register_component("test", mock)
    
    if hasattr(mock, 'on_phase_change'):
        print("   ✅ Integration Manager: WORKING")
        
        # Test event injection
        await mock.on_phase_change({'phase': 'ACTIVE'})
        if mock.current_phase == 'ACTIVE':
            print("   ✅ Event Handler Injection: WORKING")
        else:
            print("   ⚠️ Event Handler Injection: PARTIAL")
    else:
        print("   ❌ Integration Manager: FAILED")
    
    # Test 3: Event Flow
    print("\n3️⃣ Testing Event Flow...")
    
    phase_received = []
    
    async def phase_handler(data):
        phase_received.append(data)
        print(f"   Phase event: {data}")
    
    event_bus.subscribe(Events.PHASE_CHANGED, phase_handler)
    
    await event_bus.publish(Events.PHASE_CHANGED, {
        'phase': 'ACTIVE',
        'timestamp': asyncio.get_event_loop().time()
    })
    
    await asyncio.sleep(0.2)
    
    if len(phase_received) > 0:
        print("   ✅ Event Flow: WORKING")
    else:
        print("   ❌ Event Flow: FAILED")
    
    await event_bus.stop()
    
    # Results
    total_tests = 3
    passed_tests = sum([
        len(received) > 0,  # Event bus
        hasattr(mock, 'on_phase_change'),  # Integration manager
        len(phase_received) > 0  # Event flow
    ])
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ INTEGRATION READY")
        return True
    else:
        print("❌ INTEGRATION ISSUES")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    print(f"\n🎯 Status: {'PASS' if success else 'FAIL'}")