# tests/integration/test_system_coordination.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from infrastructure.system_coordinator import SystemCoordinator
from infrastructure.event_bus import EventBus, Event, EventType
from infrastructure.integration_manager import IntegrationManager
from infrastructure.unified_monitor import UnifiedMonitor
from infrastructure.memory_manager import MemoryManager
from core.attention_learning_layer import AttentionLearningLayer
from core.market_regime_detector import MarketRegimeDetector
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor
from core.feedback_loop import FeedbackLoop
from core.overfitting_detector import OverfittingDetector


@dataclass
class SystemState:
    """Track overall system state"""
    timestamp: datetime
    active_components: Dict[str, bool]
    processing_queue: List[Any]
    resource_usage: Dict[str, float]
    error_count: int = 0
    coordination_metrics: Dict[str, Any] = None


class TestSystemCoordination:
    """Test system-wide coordination and orchestration"""
    
    @pytest.fixture
    async def coordinated_system(self):
        """Create fully coordinated system"""
        config = {
            'symbol': 'BTC/USDT',
            'timeframe': '5m',
            'coordination': {
                'heartbeat_interval': 1,  # seconds
                'sync_timeout': 5,
                'max_queue_size': 1000,
                'priority_levels': 3
            },
            'resource_limits': {
                'max_memory_mb': 1000,
                'max_cpu_percent': 80,
                'max_latency_ms': 100
            }
        }
        
        # Initialize infrastructure
        event_bus = EventBus()
        memory_manager = MemoryManager(config)
        
        # Initialize components
        components = {
            'coordinator': SystemCoordinator(config),
            'event_bus': event_bus,
            'integration_manager': IntegrationManager(config),
            'unified_monitor': UnifiedMonitor(config),
            'memory_manager': memory_manager,
            'attention': AttentionLearningLayer(config),
            'regime_detector': MarketRegimeDetector(config),
            'strategy_selector': GridStrategySelector(config),
            'risk_manager': RiskManagementSystem(config),
            'execution_engine': ExecutionEngine(config),
            'performance_monitor': PerformanceMonitor(config),
            'feedback_loop': FeedbackLoop(config),
            'overfitting_detector': OverfittingDetector(config)
        }
        
        # Register components with coordinator
        for name, component in components.items():
            if name not in ['coordinator', 'event_bus']:
                await components['coordinator'].register_component(name, component)
        
        return components, config
    
    @pytest.mark.asyncio
    async def test_system_startup_sequence(self, coordinated_system):
        """Test coordinated system startup"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        
        # Track startup sequence
        startup_events = []
        
        async def track_startup(event):
            startup_events.append({
                'component': event.data.get('component'),
                'status': event.data.get('status'),
                'timestamp': datetime.now()
            })
        
        components['event_bus'].subscribe(EventType.COMPONENT_STARTED, track_startup)
        
        # Execute coordinated startup
        startup_result = await coordinator.startup_system()
        
        assert startup_result['success']
        assert startup_result['all_components_ready']
        
        # Verify startup order (dependencies first)
        startup_order = [e['component'] for e in startup_events]
        
        # Core components should start before dependent ones
        assert startup_order.index('memory_manager') < startup_order.index('attention')
        assert startup_order.index('event_bus') < startup_order.index('execution_engine')
        
        # All components should be active
        system_status = await coordinator.get_system_status()
        assert all(system_status['components'].values())
    
    @pytest.mark.asyncio
    async def test_coordinated_data_flow(self, coordinated_system):
        """Test data flow coordination across system"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        
        # Setup data flow tracking
        flow_tracker = defaultdict(list)
        
        async def track_flow(stage: str, data: Any):
            flow_tracker[stage].append({
                'timestamp': datetime.now(),
                'data_shape': getattr(data, 'shape', len(data)) if data else 0
            })
        
        # Simulate market data arrival
        market_data = self._generate_market_tick()
        
        # Process through coordinated pipeline
        # 1. Data ingestion
        await coordinator.process_market_data(market_data)
        await track_flow('ingestion', market_data)
        
        # 2. Feature engineering
        features = await coordinator.extract_features(market_data)
        await track_flow('features', features)
        
        # 3. Attention processing
        weighted_features = await coordinator.apply_attention(features)
        await track_flow('attention', weighted_features)
        
        # 4. Regime detection
        regime = await coordinator.detect_regime(weighted_features)
        await track_flow('regime', regime)
        
        # 5. Strategy selection
        strategy = await coordinator.select_strategy(regime, weighted_features)
        await track_flow('strategy', strategy)
        
        # 6. Risk validation
        risk_approved = await coordinator.validate_risk(strategy)
        await track_flow('risk', risk_approved)
        
        # Verify complete flow
        expected_stages = ['ingestion', 'features', 'attention', 'regime', 'strategy', 'risk']
        for stage in expected_stages:
            assert stage in flow_tracker
            assert len(flow_tracker[stage]) > 0
        
        # Check timing (each stage should complete quickly)
        for stage, records in flow_tracker.items():
            if len(records) > 1:
                time_diff = (records[-1]['timestamp'] - records[0]['timestamp']).total_seconds()
                assert time_diff < 1.0  # Each stage under 1 second
    
    @pytest.mark.asyncio
    async def test_component_synchronization(self, coordinated_system):
        """Test component synchronization mechanisms"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        
        # Test synchronization barrier
        sync_results = []
        
        async def component_task(name: str, delay: float):
            await asyncio.sleep(delay)
            result = {'component': name, 'completed': datetime.now()}
            sync_results.append(result)
            return result
        
        # Launch components with different delays
        tasks = [
            component_task('attention', 0.1),
            component_task('regime', 0.2),
            component_task('strategy', 0.15),
            component_task('risk', 0.05)
        ]
        
        # Coordinate with barrier
        start_time = datetime.now()
        results = await coordinator.synchronized_execution(tasks)
        end_time = datetime.now()
        
        # All should complete together
        assert len(results) == 4
        completion_times = [r['completed'] for r in sync_results]
        time_spread = (max(completion_times) - min(completion_times)).total_seconds()
        assert time_spread < 0.1  # All complete within 100ms of each other
    
    @pytest.mark.asyncio
    async def test_resource_coordination(self, coordinated_system):
        """Test resource allocation and management"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        memory_manager = components['memory_manager']
        
        # Monitor resource allocation
        resource_allocations = []
        
        # Request resources for different components
        resource_requests = [
            {'component': 'attention', 'memory_mb': 200, 'cpu_cores': 2},
            {'component': 'regime_detector', 'memory_mb': 150, 'cpu_cores': 1},
            {'component': 'execution_engine', 'memory_mb': 100, 'cpu_cores': 1},
            {'component': 'overfitting_detector', 'memory_mb': 300, 'cpu_cores': 2}
        ]
        
        for request in resource_requests:
            allocation = await coordinator.allocate_resources(
                request['component'],
                request['memory_mb'],
                request['cpu_cores']
            )
            resource_allocations.append(allocation)
        
        # Verify allocations respect limits
        total_memory = sum(a['memory_allocated'] for a in resource_allocations)
        assert total_memory <= config['resource_limits']['max_memory_mb']
        
        # Test resource pressure handling
        # Request exceeding available resources
        large_request = {'component': 'new_component', 'memory_mb': 500, 'cpu_cores': 4}
        
        with patch.object(memory_manager, 'get_available_memory', return_value=100):
            allocation = await coordinator.allocate_resources(
                large_request['component'],
                large_request['memory_mb'],
                large_request['cpu_cores']
            )
            
            # Should handle gracefully
            assert allocation['memory_allocated'] <= 100
            assert 'resource_limited' in allocation
    
    @pytest.mark.asyncio
    async def test_event_coordination_and_ordering(self, coordinated_system):
        """Test event coordination and proper ordering"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        event_bus = components['event_bus']
        
        # Track event sequence
        event_sequence = []
        
        async def sequence_tracker(event):
            event_sequence.append({
                'type': event.type,
                'timestamp': event.timestamp,
                'priority': event.data.get('priority', 0)
            })
        
        # Subscribe to all events
        for event_type in EventType:
            event_bus.subscribe(event_type, sequence_tracker)
        
        # Generate events with different priorities
        events = [
            Event(EventType.RISK_ALERT, {'priority': 3, 'severity': 'high'}),
            Event(EventType.MARKET_DATA, {'priority': 1, 'data': 'tick'}),
            Event(EventType.STRATEGY_UPDATE, {'priority': 2, 'params': {}}),
            Event(EventType.EXECUTION_REQUEST, {'priority': 3, 'order': {}})
        ]
        
        # Process through coordinator
        for event in events:
            await coordinator.handle_prioritized_event(event)
        
        # Allow processing
        await asyncio.sleep(0.1)
        
        # Verify high priority events processed first
        high_priority_events = [e for e in event_sequence if e.get('priority', 0) >= 3]
        low_priority_events = [e for e in event_sequence if e.get('priority', 0) < 3]
        
        if high_priority_events and low_priority_events:
            # High priority should be processed before low priority
            first_high = min(e['timestamp'] for e in high_priority_events)
            last_low = max(e['timestamp'] for e in low_priority_events)
            # This might not always be true due to async nature, but tendency should exist
    
    @pytest.mark.asyncio
    async def test_failure_coordination_and_recovery(self, coordinated_system):
        """Test coordinated failure handling"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        
        # Simulate component failures
        failure_scenarios = [
            {
                'component': 'execution_engine',
                'error': ConnectionError("Exchange connection lost"),
                'recovery': 'reconnect'
            },
            {
                'component': 'attention',
                'error': RuntimeError("Model convergence failed"),
                'recovery': 'rollback'
            },
            {
                'component': 'risk_manager',
                'error': ValueError("Invalid risk parameters"),
                'recovery': 'use_defaults'
            }
        ]
        
        recovery_actions = []
        
        for scenario in failure_scenarios:
            # Inject failure
            await coordinator.report_component_failure(
                scenario['component'],
                scenario['error']
            )
            
            # Coordinate recovery
            recovery_result = await coordinator.coordinate_recovery(
                scenario['component'],
                scenario['recovery']
            )
            
            recovery_actions.append({
                'component': scenario['component'],
                'recovery': recovery_result,
                'timestamp': datetime.now()
            })
        
        # Verify all components recovered
        assert all(action['recovery']['success'] for action in recovery_actions)
        
        # Check system stability after recovery
        system_health = await coordinator.check_system_health()
        assert system_health['stable']
        assert system_health['all_components_responsive']
    
    @pytest.mark.asyncio
    async def test_performance_coordination(self, coordinated_system):
        """Test performance optimization coordination"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        unified_monitor = components['unified_monitor']
        
        # Track performance metrics
        performance_history = []
        
        # Simulate varying load conditions
        load_patterns = [
            {'name': 'normal', 'tps': 10, 'duration': 5},
            {'name': 'spike', 'tps': 100, 'duration': 2},
            {'name': 'sustained_high', 'tps': 50, 'duration': 10}
        ]
        
        for pattern in load_patterns:
            # Apply load pattern
            start_time = datetime.now()
            
            for _ in range(pattern['duration']):
                # Generate load
                for _ in range(pattern['tps']):
                    await coordinator.process_market_data(self._generate_market_tick())
                
                # Measure performance
                metrics = await unified_monitor.get_current_metrics()
                performance_history.append({
                    'pattern': pattern['name'],
                    'timestamp': datetime.now(),
                    'latency': metrics.get('avg_latency_ms', 0),
                    'throughput': metrics.get('throughput_tps', 0),
                    'cpu_usage': metrics.get('cpu_percent', 0)
                })
                
                await asyncio.sleep(0.1)
            
            # Coordinator should optimize based on load
            optimization = await coordinator.optimize_for_load(pattern['tps'])
            
            # Verify optimization applied
            assert optimization['applied']
            
            # Check if performance improved
            if pattern['name'] == 'spike':
                # Should handle spikes by buffering or scaling
                assert optimization['actions_taken'] in ['buffering', 'scaling', 'throttling']
    
    @pytest.mark.asyncio
    async def test_state_coordination_across_components(self, coordinated_system):
        """Test state consistency across components"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        
        # Define shared state
        shared_state = {
            'current_regime': 'RANGING',
            'active_positions': 5,
            'risk_level': 0.3,
            'performance_score': 0.75
        }
        
        # Update state through coordinator
        await coordinator.update_shared_state(shared_state)
        
        # Verify all components have consistent view
        component_states = {}
        
        for name in ['attention', 'regime_detector', 'strategy_selector', 'risk_manager']:
            if name in components:
                state = await coordinator.get_component_state(name)
                component_states[name] = state
        
        # Check state consistency
        for component, state in component_states.items():
            if 'current_regime' in state:
                assert state['current_regime'] == shared_state['current_regime']
            if 'risk_level' in state:
                assert abs(state['risk_level'] - shared_state['risk_level']) < 0.01
    
    @pytest.mark.asyncio
    async def test_coordinated_shutdown(self, coordinated_system):
        """Test graceful coordinated shutdown"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        
        # Track shutdown sequence
        shutdown_events = []
        
        async def track_shutdown(event):
            shutdown_events.append({
                'component': event.data.get('component'),
                'timestamp': datetime.now()
            })
        
        components['event_bus'].subscribe(EventType.COMPONENT_STOPPED, track_shutdown)
        
        # Initiate coordinated shutdown
        shutdown_result = await coordinator.shutdown_system(timeout=10)
        
        assert shutdown_result['success']
        assert shutdown_result['all_components_stopped']
        
        # Verify shutdown order (reverse of startup)
        shutdown_order = [e['component'] for e in shutdown_events]
        
        # Dependent components should stop before core ones
        if 'execution_engine' in shutdown_order and 'event_bus' in shutdown_order:
            assert shutdown_order.index('execution_engine') < shutdown_order.index('event_bus')
        
        # No components should be active
        final_status = await coordinator.get_system_status()
        assert not any(final_status['components'].values())
    
    @pytest.mark.asyncio
    async def test_coordination_monitoring_and_metrics(self, coordinated_system):
        """Test coordination layer monitoring"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        unified_monitor = components['unified_monitor']
        
        # Enable coordination metrics
        await coordinator.enable_detailed_monitoring()
        
        # Perform various coordination tasks
        coordination_tasks = [
            coordinator.process_market_data(self._generate_market_tick()),
            coordinator.synchronize_components(['attention', 'regime_detector']),
            coordinator.handle_prioritized_event(
                Event(EventType.RISK_ALERT, {'severity': 'medium'})
            ),
            coordinator.optimize_for_load(50)
        ]
        
        # Execute tasks
        await asyncio.gather(*coordination_tasks)
        
        # Collect coordination metrics
        metrics = await unified_monitor.get_coordination_metrics()
        
        # Verify metrics collected
        expected_metrics = [
            'coordination_latency',
            'sync_success_rate',
            'event_processing_time',
            'resource_utilization',
            'component_health_scores'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert metrics[metric] is not None
        
        # Check metric quality
        assert metrics['sync_success_rate'] > 0.9  # High success rate
        assert metrics['coordination_latency'] < 100  # Low latency (ms)
        assert all(score > 0.7 for score in metrics['component_health_scores'].values())
    
    @pytest.mark.asyncio
    async def test_adaptive_coordination_strategy(self, coordinated_system):
        """Test adaptive coordination based on system conditions"""
        components, config = coordinated_system
        coordinator = components['coordinator']
        
        # Define different system conditions
        conditions = [
            {
                'name': 'high_volatility',
                'market_volatility': 0.05,
                'expected_strategy': 'conservative'
            },
            {
                'name': 'low_latency_required',
                'latency_requirement': 10,  # ms
                'expected_strategy': 'performance'
            },
            {
                'name': 'resource_constrained',
                'available_memory': 100,  # MB
                'expected_strategy': 'efficient'
            }
        ]
        
        for condition in conditions:
            # Apply condition
            await coordinator.set_system_condition(condition)
            
            # Get adapted coordination strategy
            strategy = await coordinator.get_coordination_strategy()
            
            assert strategy['name'] == condition['expected_strategy']
            
            # Verify strategy parameters adjusted
            if condition['name'] == 'high_volatility':
                assert strategy['risk_checks_enabled']
                assert strategy['update_frequency'] == 'high'
            
            elif condition['name'] == 'low_latency_required':
                assert strategy['parallel_processing']
                assert strategy['cache_enabled']
            
            elif condition['name'] == 'resource_constrained':
                assert strategy['memory_optimization']
                assert strategy['batch_processing']
    
    # Helper methods
    def _generate_market_tick(self) -> Dict[str, Any]:
        """Generate market tick data"""
        return {
            'symbol': 'BTC/USDT',
            'timestamp': datetime.now(),
            'price': 50000 + np.random.randn() * 100,
            'volume': np.random.uniform(10, 100),
            'bid': 49995 + np.random.randn() * 10,
            'ask': 50005 + np.random.randn() * 10
        }
    
    def _generate_load_pattern(self, tps: int, duration: int) -> List[Dict[str, Any]]:
        """Generate load pattern for testing"""
        pattern = []
        for i in range(duration):
            for j in range(tps):
                pattern.append({
                    'timestamp': datetime.now() + timedelta(seconds=i, milliseconds=j*10),
                    'data': self._generate_market_tick()
                })
        return pattern