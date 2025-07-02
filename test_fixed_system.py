import asyncio
from system_coordinator import SystemCoordinator
import logging

logging.basicConfig(level=logging.INFO)

async def test_system():
    print("🧪 Testing fixed system...\n")
    
    # Minimal config
    config = {
        'attention_learning_layer': {'learning_rate': 0.001},
        'risk_management_system': {'max_position_size': 0.05}
    }
    
    coordinator = SystemCoordinator(config)
    
    try:
        # Initialize
        print("1️⃣ Initializing components...")
        await coordinator.initialize_components()
        print("✓ Components initialized\n")
        
        # Start
        print("2️⃣ Starting system...")
        await coordinator.start()
        print("✓ System started\n")
        
        # Run for 5 seconds
        print("3️⃣ Running for 5 seconds...")
        await asyncio.sleep(5)
        
        # Check health
        print("\n4️⃣ Checking component health:")
        for name, comp in coordinator.components.items():
            try:
                if hasattr(comp, 'is_healthy'):
                    healthy = await comp.is_healthy()
                    print(f"  {name}: {'✓' if healthy else '✗'}")
                else:
                    print(f"  {name}: ⚠️ (no health check)")
            except Exception as e:
                print(f"  {name}: ✗ (error: {e})")
        
        # Stop
        print("\n5️⃣ Stopping system...")
        await coordinator.stop()
        print("✓ System stopped\n")
        
        print("✅ Test completed successfully\!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_system())