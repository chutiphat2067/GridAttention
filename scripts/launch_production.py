#!/usr/bin/env python3
"""
Production launcher for GridAttention Trading System
"""
import sys
import asyncio
import logging
import signal
from datetime import datetime
from main import GridTradingSystem

# Production logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ProductionLauncher:
    """Production launcher with proper shutdown handling"""
    
    def __init__(self):
        self.system = None
        self.shutdown_requested = False
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
    async def launch_production(self):
        """Launch system in production mode"""
        
        print("🚀 Launching GridAttention in Production Mode")
        print("=" * 50)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # Create system with production config
            self.system = GridTradingSystem('config/config_production.yaml')
            
            # Verify configuration
            print("📋 Production Configuration:")
            print(f"✓ Training mode: {self.system.training_mode}")
            print(f"✓ Dashboard enabled: {self.system.dashboard_enabled}")
            print(f"✓ Performance mode: {self.system.performance_mode}")
            print()
            
            # Initialize and start system
            logger.info("Initializing production system...")
            await self.system.initialize()
            
            logger.info("Starting production system...")
            await self.system.start()
            
            print("🎯 System Status:")
            print("✅ All components initialized")
            print("✅ System started successfully")
            print("✅ Production mode active")
            print()
            
            print("🌐 Access Points:")
            print("- Dashboard: http://localhost:8080")
            print("- Metrics: http://localhost:9090") 
            print("- Logs: production.log")
            print()
            
            print("🔧 Control Commands:")
            print("- Ctrl+C: Graceful shutdown")
            print("- curl -X POST http://localhost:8080/api/emergency_stop")
            print()
            
            logger.info("Production system running...")
            
            # Keep running until shutdown requested
            while not self.shutdown_requested:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.critical(f"Production launch failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.system:
                logger.info("Shutting down system...")
                try:
                    await self.system.stop()
                    logger.info("System shutdown completed")
                except Exception as e:
                    logger.error(f"Error during shutdown: {e}")
                    
        return True

async def main():
    """Main entry point"""
    launcher = ProductionLauncher()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)
    
    try:
        success = await launcher.launch_production()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())