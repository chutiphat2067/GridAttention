import asyncio
import logging
from typing import Dict, Any, Optional
from event_bus import event_bus, Events
from essential_fixes import apply_essential_fixes

logger = logging.getLogger(__name__)

class SystemCoordinator:
    """Central coordinator for all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components = {}
        self.is_running = False
        self.health_check_interval = 60  # seconds
        
    async def initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing components...")
        
        # Import and create components
        component_configs = [
            ('attention_learning_layer', 'AttentionLearningLayer'),
            ('market_regime_detector', 'MarketRegimeDetector'),
            ('grid_strategy_selector', 'GridStrategySelector'),
            ('risk_management_system', 'RiskManagementSystem'),
            ('execution_engine', 'ExecutionEngine'),
            ('performance_monitor', 'PerformanceMonitor'),
            ('overfitting_detector', 'OverfittingDetector'),
            ('feedback_loop', 'FeedbackLoop')
        ]
        
        for module_name, class_name in component_configs:
            try:
                module = __import__(module_name)
                component_class = getattr(module, class_name)
                self.components[module_name] = component_class(
                    self.config.get(module_name, {})
                )
                logger.info(f"✓ Initialized {module_name}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {module_name}: {e}")
        
        # Apply fixes
        self.components = apply_essential_fixes(self.components)
        
        # Setup integration
        await self._setup_integration()
        
        # Subscribe to events
        await self._setup_event_handlers()
        
    async def _setup_integration(self):
        """Setup component integration using IntegrationManager"""
        from integration_manager import integration_manager
        
        # Register all components with integration manager
        for name, component in self.components.items():
            await integration_manager.register_component(name, component)
            
        # Setup feedback loops
        await integration_manager.setup_feedback_loops()
        
        # Setup kill switch integration
        await integration_manager.setup_kill_switch_integration()
        
        # Verify integration
        status = await integration_manager.verify_integration()
        logger.info(f"Integration status: {status}")
        
    async def _setup_event_handlers(self):
        """Setup event handlers for coordination"""
        
        # Phase change handler
        async def handle_phase_change(data):
            phase = data['phase']
            logger.info(f"Attention phase changed to: {phase}")
            
            # Notify other components
            for comp in self.components.values():
                if hasattr(comp, 'on_phase_change'):
                    await comp.on_phase_change(phase)
        
        event_bus.subscribe(Events.PHASE_CHANGED, handle_phase_change)
        
        # Overfitting handler
        async def handle_overfitting(data):
            severity = data['severity']
            logger.warning(f"Overfitting detected: {severity}")
            
            if severity == 'CRITICAL':
                # Activate kill switch
                if 'risk_management_system' in self.components:
                    await self.components['risk_management_system'].activate_kill_switch(
                        f"Critical overfitting detected: {data}"
                    )
        
        event_bus.subscribe(Events.OVERFITTING_DETECTED, handle_overfitting)
        
    async def start(self):
        """Start all components"""
        logger.info("Starting system...")
        
        # Start event bus
        await event_bus.start()
        
        # Start components in order
        start_order = [
            'market_data_input',
            'feature_engineering',
            'attention_learning_layer',
            'market_regime_detector',
            'overfitting_detector',
            'grid_strategy_selector',
            'risk_management_system',
            'execution_engine',
            'performance_monitor',
            'feedback_loop'
        ]
        
        for comp_name in start_order:
            if comp_name in self.components:
                comp = self.components[comp_name]
                if hasattr(comp, 'start'):
                    await comp.start()
                    logger.info(f"✓ Started {comp_name}")
        
        self.is_running = True
        
        # Start health monitoring
        asyncio.create_task(self._monitor_health())
        
    async def _monitor_health(self):
        """Monitor component health"""
        while self.is_running:
            try:
                unhealthy = []
                
                for name, comp in self.components.items():
                    if hasattr(comp, 'is_healthy'):
                        if not await comp.is_healthy():
                            unhealthy.append(name)
                
                if unhealthy:
                    logger.warning(f"Unhealthy components: {unhealthy}")
                    
                    # Try recovery
                    for name in unhealthy:
                        comp = self.components[name]
                        if hasattr(comp, 'recover'):
                            if await comp.recover():
                                logger.info(f"✓ Recovered {name}")
                            else:
                                logger.error(f"✗ Failed to recover {name}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
    async def stop(self):
        """Stop all components"""
        logger.info("Stopping system...")
        
        self.is_running = False
        
        # Stop in reverse order
        for comp in reversed(list(self.components.values())):
            if hasattr(comp, 'stop'):
                await comp.stop()
        
        await event_bus.stop()
        
        logger.info("System stopped")

# Usage
async def main():
    config = {
        'attention_learning_layer': {
            'learning_rate': 0.001,
            'window_size': 1000
        },
        'risk_management_system': {
            'max_position_size': 0.05,
            'max_daily_loss': 0.02
        }
    }
    
    coordinator = SystemCoordinator(config)
    await coordinator.initialize_components()
    await coordinator.start()
    
    # Run for a while
    await asyncio.sleep(60)
    
    await coordinator.stop()

if __name__ == "__main__":
    asyncio.run(main())