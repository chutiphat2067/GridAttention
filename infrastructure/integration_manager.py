"""
Integration Manager - Central orchestration for component integration
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from infrastructure.event_bus import event_bus, Events

logger = logging.getLogger(__name__)

class IntegrationManager:
    """Manages component integration and event flows"""
    
    def __init__(self):
        self.components = {}
        self.event_flows = {}
        self.integration_status = {}
        
    async def register_component(self, name: str, component: Any) -> None:
        """Register component and setup integration"""
        self.components[name] = component
        
        # Add event handlers to component
        await self._inject_event_handlers(name, component)
        
        # Setup component-specific subscriptions
        await self._setup_component_subscriptions(name, component)
        
        logger.info(f"✓ {name} integrated")
        
    async def _inject_event_handlers(self, name: str, component: Any) -> None:
        """Inject standard event handlers"""
        
        async def on_phase_change(data):
            """Handle phase change"""
            if hasattr(component, 'current_phase'):
                component.current_phase = data.get('phase')
            logger.info(f"{name} received phase change: {data.get('phase')}")
            
        async def on_overfitting(data):
            """Handle overfitting detection"""
            severity = data.get('severity')
            if severity == 'CRITICAL':
                if hasattr(component, 'enter_safe_mode'):
                    await component.enter_safe_mode()
                elif hasattr(component, 'reduce_risk'):
                    await component.reduce_risk()
                    
        async def on_kill_switch(data):
            """Handle kill switch activation"""
            if hasattr(component, 'emergency_stop'):
                await component.emergency_stop()
            elif hasattr(component, 'stop'):
                await component.stop()
                
        # Inject methods into component
        component.on_phase_change = on_phase_change
        component.on_overfitting = on_overfitting  
        component.on_kill_switch = on_kill_switch
        
        # Subscribe to events
        event_bus.subscribe(Events.PHASE_CHANGED, on_phase_change)
        event_bus.subscribe(Events.OVERFITTING_DETECTED, on_overfitting)
        event_bus.subscribe(Events.KILL_SWITCH_ACTIVATED, on_kill_switch)
        
    async def _setup_component_subscriptions(self, name: str, component: Any) -> None:
        """Setup component-specific event subscriptions"""
        
        if name == 'attention_learning_layer':
            # Attention layer publishes phase changes
            original_transition = getattr(component, 'transition_phase', None)
            if original_transition:
                async def enhanced_transition(new_phase):
                    await original_transition(new_phase)
                    await event_bus.publish(Events.PHASE_CHANGED, {
                        'phase': new_phase,
                        'component': name,
                        'timestamp': asyncio.get_event_loop().time()
                    })
                component.transition_phase = enhanced_transition
                
        elif name == 'overfitting_detector':
            # Overfitting detector publishes overfitting events
            original_detect = getattr(component, 'detect_overfitting', None)
            if original_detect:
                async def enhanced_detect(*args, **kwargs):
                    result = await original_detect(*args, **kwargs)
                    if result.get('overfitting_detected'):
                        await event_bus.publish(Events.OVERFITTING_DETECTED, {
                            'severity': result.get('severity'),
                            'confidence': result.get('confidence'),
                            'timestamp': asyncio.get_event_loop().time()
                        })
                    return result
                component.detect_overfitting = enhanced_detect
                
        elif name == 'risk_management_system':
            # Risk management publishes risk alerts
            original_check = getattr(component, 'check_risk_limits', None)
            if original_check:
                async def enhanced_check(*args, **kwargs):
                    result = await original_check(*args, **kwargs)
                    if not result.get('passed'):
                        await event_bus.publish(Events.RISK_LIMIT_REACHED, {
                            'violations': result.get('violations'),
                            'severity': result.get('severity'),
                            'timestamp': asyncio.get_event_loop().time()
                        })
                    return result
                component.check_risk_limits = enhanced_check
                
    async def setup_feedback_loops(self) -> None:
        """Setup inter-component feedback loops"""
        
        # Performance → Feedback Loop
        perf_monitor = self.components.get('performance_monitor')
        feedback_loop = self.components.get('feedback_loop')
        
        if perf_monitor and feedback_loop:
            async def on_performance_update(data):
                if hasattr(feedback_loop, 'process_performance_data'):
                    await feedback_loop.process_performance_data(data)
                    
            event_bus.subscribe('PERFORMANCE_UPDATE', on_performance_update)
            
        # Feedback Loop → Attention Learning
        attention = self.components.get('attention_learning_layer')
        
        if feedback_loop and attention:
            async def on_feedback_signal(data):
                if hasattr(attention, 'apply_feedback'):
                    await attention.apply_feedback(data)
                    
            event_bus.subscribe('FEEDBACK_SIGNAL', on_feedback_signal)
            
        logger.info("✓ Feedback loops connected")
        
    async def setup_kill_switch_integration(self) -> None:
        """Setup kill switch integration across components"""
        
        risk_mgr = self.components.get('risk_management_system')
        execution = self.components.get('execution_engine')
        
        if risk_mgr and execution:
            # Enhance kill switch to close positions
            original_kill_switch = getattr(risk_mgr, 'activate_kill_switch', None)
            
            if original_kill_switch:
                async def enhanced_kill_switch(reason: str):
                    logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
                    
                    # Activate original kill switch
                    await original_kill_switch(reason)
                    
                    # Publish kill switch event
                    await event_bus.publish(Events.KILL_SWITCH_ACTIVATED, {
                        'reason': reason,
                        'timestamp': asyncio.get_event_loop().time()
                    })
                    
                    # Close all positions
                    if hasattr(execution, 'cancel_all_orders'):
                        await execution.cancel_all_orders()
                        
                risk_mgr.activate_kill_switch = enhanced_kill_switch
                
        logger.info("✓ Kill switch integration complete")
        
    async def verify_integration(self) -> Dict[str, Any]:
        """Verify all integrations are working"""
        
        status = {
            'components_registered': len(self.components),
            'event_handlers_injected': 0,
            'feedback_loops_active': 0,
            'integration_health': 'healthy'
        }
        
        # Check each component has event handlers
        for name, comp in self.components.items():
            if hasattr(comp, 'on_phase_change'):
                status['event_handlers_injected'] += 1
                
        # Check feedback loops
        if (self.components.get('performance_monitor') and 
            self.components.get('feedback_loop')):
            status['feedback_loops_active'] += 1
            
        if (self.components.get('feedback_loop') and
            self.components.get('attention_learning_layer')):
            status['feedback_loops_active'] += 1
            
        # Overall health
        if status['event_handlers_injected'] < len(self.components) * 0.8:
            status['integration_health'] = 'degraded'
            
        return status

# Global integration manager
integration_manager = IntegrationManager()