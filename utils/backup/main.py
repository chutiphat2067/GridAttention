# main.py
import asyncio
import logging
import signal
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from data.market_data_input import MarketDataInput
from data.feature_engineering_pipeline import FeatureEngineeringPipeline
from core.attention_learning_layer import AttentionLearningLayer
from core.market_regime_detector import MarketRegimeDetector
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor
from core.feedback_loop import FeedbackLoop
from monitoring.scaling_monitor import create_scaling_monitor, ScalingMonitor

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grid_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GridTradingSystem:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.components = {}
        self._running = False
        self._shutdown_requested = False
        self._tasks = []
        
        # Performance monitoring
        self.start_time = datetime.now()
        self.error_count = 0
        self.last_heartbeat = datetime.now()
        self.scaling_monitor = None
        
    def _load_config(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
            
    async def initialize(self):
        """Initialize all components with proper error handling"""
        try:
            logger.info("Initializing GridTradingSystem...")
            
            # 1. Market Data Input
            self.components['market_data'] = MarketDataInput(
                self.config.get('market_data', {})
            )
            
            # 2. Feature Engineering
            self.components['features'] = FeatureEngineeringPipeline(
                self.config.get('features', {})
            )
            
            # 3. Attention Layer
            self.components['attention'] = AttentionLearningLayer(
                self.config.get('attention', {})
            )
            
            # 4. Market Regime Detector
            self.components['regime_detector'] = MarketRegimeDetector(
                self.config.get('regime_detector', {})
            )
            
            # 5. Grid Strategy Selector
            self.components['strategy_selector'] = GridStrategySelector(
                self.config.get('strategy_selector', {})
            )
            
            # 6. Risk Management System
            self.components['risk_manager'] = RiskManagementSystem(
                self.config.get('risk_management', {})
            )
            
            # 7. Execution Engine
            self.components['execution'] = ExecutionEngine(
                self.config.get('execution', {})
            )
            
            # 8. Performance Monitor
            self.components['performance'] = PerformanceMonitor(
                self.config.get('performance', {})
            )
            
            # 9. Feedback Loop
            self.components['feedback'] = FeedbackLoop(
                self.config.get('feedback', {})
            )

            # 10. Scaling monitor
            self.scaling_monitor = await create_scaling_monitor(
                self.components,
                self.config.get('scaling_monitor', {})
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
        
    async def start(self):
        """Start the trading system with proper lifecycle management"""
        try:
            await self.initialize()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self._running = True
            logger.info("Starting GridTradingSystem...")
            
            # Start all components sequentially with error handling
            await self._start_components()
            
            # Start monitoring tasks
            self._tasks.extend([
                asyncio.create_task(self._heartbeat_monitor()),
                asyncio.create_task(self._error_monitor()),
                asyncio.create_task(self._performance_monitor())
            ])
            
            # Main trading loop
            while self._running and not self._shutdown_requested:
                try:
                    await self._trading_loop()
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Trading loop error: {e}")
                    
                    # Circuit breaker: stop if too many errors
                    if self.error_count > 10:
                        logger.critical("Too many errors, shutting down")
                        await self.shutdown()
                        break
                        
                    await asyncio.sleep(1)

            # Start scaling monitor
            if self.scaling_monitor:
                await self.scaling_monitor.start()
                
            # scaling monitor task to tasks list
            self._tasks.append(
                asyncio.create_task(self._scaling_report_task())
            )
                    
        except Exception as e:
            logger.critical(f"Critical error in start(): {e}")
            await self.shutdown()
            raise
                
    async def _trading_loop(self):
        """Main trading logic"""
        # 1. Get market data
        tick = await self.components['market_data'].collect_tick()
        
        # 2. Extract features
        features = await self.components['features'].extract_features()
        
        # 3. Process through attention
        features = await self.components['attention'].process(
            features, regime, context
        )
        
        # ... continue with trading logic

    async def _start_components(self):
        """Start all components with error handling"""
        component_order = [
            'market_data', 'risk_manager', 'performance', 
            'execution', 'regime_detector', 'strategy_selector'
        ]
        
        for component_name in component_order:
            try:
                component = self.components.get(component_name)
                if component and hasattr(component, 'start'):
                    await component.start()
                    logger.info(f"Started {component_name}")
            except Exception as e:
                logger.error(f"Failed to start {component_name}: {e}")
                raise
                
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def _heartbeat_monitor(self):
        """Monitor system heartbeat"""
        while self._running:
            self.last_heartbeat = datetime.now()
            await asyncio.sleep(30)  # Heartbeat every 30 seconds
            
    async def _error_monitor(self):
        """Monitor system errors"""
        while self._running:
            if self.error_count > 0:
                logger.warning(f"Current error count: {self.error_count}")
            await asyncio.sleep(60)  # Check every minute
            
    async def _performance_monitor(self):
        """Monitor system performance"""
        while self._running:
            uptime = datetime.now() - self.start_time
            logger.info(f"System uptime: {uptime}, errors: {self.error_count}")
            await asyncio.sleep(300)  # Report every 5 minutes

    async def _scaling_report_task(self):
        """Periodic scaling report generation"""
        while self._running:
            try:
                # Generate scaling report every 5 minutes
                await asyncio.sleep(300)
                
                if self.scaling_monitor:
                    report = await self.scaling_monitor.get_scaling_report()
                    
                    # Log scaling score
                    logger.info(f"Scaling Score: {report['scaling_score']:.1f}/100")
                    
                    # Log any immediate issues
                    if report['immediate_action_required']:
                        logger.critical("IMMEDIATE SCALING ACTION REQUIRED")
                        for issue in report.get('immediate_issues', []):
                            logger.critical(f"  - {issue}")
                            
                    # Save report
                    report_path = f"scaling_report_{int(time.time())}.json"
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2)
                        
            except Exception as e:
                logger.error(f"Error generating scaling report: {e}")
            
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating graceful shutdown...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Stop all components
        for name, component in self.components.items():
            try:
                # Stop scaling monitor
                if self.scaling_monitor:
                    await self.scaling_monitor.stop()
                    logger.info("Stopped scaling monitor")

                if hasattr(component, 'stop'):
                    await component.stop()
                    logger.info(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
                
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Shutdown complete")


if __name__ == "__main__":
    config_path = "config.yaml"  # Fixed path
    
    try:
        system = GridTradingSystem(config_path)
        asyncio.run(system.start())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.critical(f"System failed: {e}")
        sys.exit(1)