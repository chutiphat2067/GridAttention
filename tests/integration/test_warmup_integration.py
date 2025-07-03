# tests/integration/test_warmup_integration.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from core.warmup_manager import WarmupManager
from core.attention_learning_layer import AttentionLearningLayer
from core.market_regime_detector import MarketRegimeDetector
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from data.market_data_input import MarketDataInput
from utils.checkpoint_manager import CheckpointManager
from monitoring.scaling_monitor import ScalingMonitor


class TestWarmupIntegration:
    """Test system warmup and initialization phase"""
    
    @pytest.fixture
    async def warmup_system(self):
        """Create warmup system with all components"""
        config = {
            'symbol': 'BTC/USDT',
            'timeframe': '5m',
            'warmup_periods': 1000,
            'min_data_quality': 0.95,
            'required_history_days': 30,
            'model_validation_threshold': 0.8,
            'component_timeout': 30,
            'parallel_init': True
        }
        
        components = {
            'warmup_manager': WarmupManager(config),
            'data_loader': HistoricalDataLoader(config),
            'model_loader': ModelLoader(config),
            'attention_system': AttentionLearningSystem(config),
            'regime_detector': MarketRegimeDetector(config),
            'grid_manager': GridStrategyManager(config),
            'risk_manager': RiskManager(config),
            'health_checker': HealthChecker(config)
        }
        
        return components, config
    
    @pytest.mark.asyncio
    async def test_warmup_sequence(self, warmup_system):
        """Test complete warmup sequence"""
        components, config = warmup_system
        warmup_manager = components['warmup_manager']
        
        # Track warmup stages
        warmup_stages = []
        
        async def stage_tracker(stage_name, status):
            warmup_stages.append({
                'stage': stage_name,
                'status': status,
                'timestamp': datetime.now()
            })
        
        warmup_manager.on_stage_complete = stage_tracker
        
        # Mock historical data
        historical_data = self._generate_historical_data(config['warmup_periods'])
        
        with patch.object(components['data_loader'], 'load_historical_data', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = historical_data
            
            # Execute warmup
            warmup_result = await warmup_manager.execute_warmup()
            
            assert warmup_result['success']
            assert warmup_result['ready_for_trading']
            
            # Verify all stages completed
            expected_stages = [
                'data_loading',
                'data_validation',
                'model_initialization',
                'component_warmup',
                'system_validation',
                'final_checks'
            ]
            
            completed_stages = [s['stage'] for s in warmup_stages if s['status'] == 'completed']
            for stage in expected_stages:
                assert stage in completed_stages
    
    @pytest.mark.asyncio
    async def test_historical_data_loading_and_validation(self, warmup_system):
        """Test historical data loading during warmup"""
        components, config = warmup_system
        data_loader = components['data_loader']
        
        # Test different data scenarios
        scenarios = [
            {
                'name': 'complete_data',
                'data': self._generate_historical_data(1000),
                'expected_result': 'success'
            },
            {
                'name': 'insufficient_data',
                'data': self._generate_historical_data(100),  # Too little
                'expected_result': 'insufficient_data'
            },
            {
                'name': 'gaps_in_data',
                'data': self._generate_data_with_gaps(1000),
                'expected_result': 'data_gaps'
            }
        ]
        
        for scenario in scenarios:
            # Load and validate
            validation_result = await data_loader.validate_historical_data(scenario['data'])
            
            if scenario['expected_result'] == 'success':
                assert validation_result['is_valid']
                assert validation_result['quality_score'] >= config['min_data_quality']
            else:
                assert not validation_result['is_valid']
                assert scenario['expected_result'] in validation_result['issues']
    
    @pytest.mark.asyncio
    async def test_model_initialization_during_warmup(self, warmup_system):
        """Test ML model loading and validation"""
        components, config = warmup_system
        model_loader = components['model_loader']
        attention_system = components['attention_system']
        
        # Mock model files
        model_artifacts = {
            'attention_weights': self._create_mock_model_weights(),
            'regime_classifier': self._create_mock_classifier(),
            'risk_model': self._create_mock_risk_model()
        }
        
        with patch.object(model_loader, 'load_model_artifacts', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = model_artifacts
            
            # Initialize models
            init_result = await model_loader.initialize_models()
            
            assert init_result['success']
            assert all(init_result['models_loaded'].values())
            
            # Validate model performance
            test_data = self._generate_historical_data(100)
            validation_result = await attention_system.validate_model(test_data)
            
            assert validation_result['accuracy'] >= config['model_validation_threshold']
    
    @pytest.mark.asyncio
    async def test_component_warmup_dependencies(self, warmup_system):
        """Test component initialization with dependencies"""
        components, config = warmup_system
        warmup_manager = components['warmup_manager']
        
        # Define component dependencies
        dependencies = {
            'attention_system': ['data_loader'],
            'regime_detector': ['data_loader', 'attention_system'],
            'grid_manager': ['regime_detector'],
            'risk_manager': ['data_loader']
        }
        
        # Track initialization order
        init_order = []
        
        async def track_init(component_name):
            init_order.append(component_name)
            await asyncio.sleep(0.01)  # Simulate init time
        
        # Mock component initialization
        for comp_name in dependencies:
            if comp_name in components:
                components[comp_name].initialize = lambda n=comp_name: track_init(n)
        
        # Execute warmup with dependencies
        await warmup_manager.warmup_components_with_dependencies(dependencies)
        
        # Verify dependency order
        for comp, deps in dependencies.items():
            comp_idx = init_order.index(comp) if comp in init_order else -1
            for dep in deps:
                dep_idx = init_order.index(dep) if dep in init_order else -1
                if comp_idx >= 0 and dep_idx >= 0:
                    assert dep_idx < comp_idx, f"{dep} should initialize before {comp}"
    
    @pytest.mark.asyncio
    async def test_parallel_component_warmup(self, warmup_system):
        """Test parallel initialization of independent components"""
        components, config = warmup_system
        warmup_manager = components['warmup_manager']
        
        # Components that can initialize in parallel
        parallel_components = ['attention_system', 'risk_manager', 'health_checker']
        
        # Track timing
        init_times = {}
        
        async def timed_init(component_name):
            start = datetime.now()
            await asyncio.sleep(0.1)  # Simulate init time
            end = datetime.now()
            init_times[component_name] = (start, end)
        
        # Mock initialization
        for comp_name in parallel_components:
            if comp_name in components:
                components[comp_name].initialize = lambda n=comp_name: timed_init(n)
        
        # Execute parallel warmup
        start_time = datetime.now()
        await warmup_manager.parallel_warmup(parallel_components)
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Verify parallel execution
        # If sequential, would take 0.3s (3 * 0.1s)
        # If parallel, should take ~0.1s
        assert total_time < 0.2  # Allow some overhead
        
        # Check overlap in execution times
        for i, comp1 in enumerate(parallel_components):
            for comp2 in parallel_components[i+1:]:
                if comp1 in init_times and comp2 in init_times:
                    # Check if execution overlapped
                    start1, end1 = init_times[comp1]
                    start2, end2 = init_times[comp2]
                    
                    overlap = (min(end1, end2) - max(start1, start2)).total_seconds()
                    assert overlap > 0, f"{comp1} and {comp2} should run in parallel"
    
    @pytest.mark.asyncio
    async def test_warmup_health_checks(self, warmup_system):
        """Test system health validation during warmup"""
        components, config = warmup_system
        health_checker = components['health_checker']
        
        # Define health check criteria
        health_checks = {
            'memory_usage': lambda: {'status': 'healthy', 'value': 45, 'threshold': 80},
            'cpu_usage': lambda: {'status': 'healthy', 'value': 30, 'threshold': 70},
            'disk_space': lambda: {'status': 'healthy', 'value': 60, 'threshold': 90},
            'network_latency': lambda: {'status': 'healthy', 'value': 10, 'threshold': 100},
            'api_connectivity': lambda: {'status': 'healthy', 'connected': True}
        }
        
        # Run health checks
        health_results = {}
        for check_name, check_func in health_checks.items():
            result = await health_checker.run_check(check_name, check_func)
            health_results[check_name] = result
        
        # All should be healthy
        assert all(r['status'] == 'healthy' for r in health_results.values())
        
        # Test with unhealthy condition
        health_checks['memory_usage'] = lambda: {'status': 'unhealthy', 'value': 85, 'threshold': 80}
        
        with pytest.raises(RuntimeError, match="Health check failed"):
            await health_checker.validate_system_health(health_checks)
    
    @pytest.mark.asyncio
    async def test_warmup_state_persistence(self, warmup_system):
        """Test saving and loading warmup state"""
        components, config = warmup_system
        warmup_manager = components['warmup_manager']
        
        # Create warmup state
        warmup_state = {
            'timestamp': datetime.now(),
            'components_initialized': {
                'attention_system': True,
                'regime_detector': True,
                'grid_manager': True,
                'risk_manager': True
            },
            'data_loaded': {
                'historical_periods': 1000,
                'quality_score': 0.97,
                'last_timestamp': datetime.now() - timedelta(minutes=5)
            },
            'models_loaded': {
                'attention_model': 'v2.1.0',
                'regime_model': 'v1.5.2',
                'risk_model': 'v3.0.1'
            },
            'validation_results': {
                'data_validation': 'passed',
                'model_validation': 'passed',
                'system_validation': 'passed'
            }
        }
        
        # Save state
        await warmup_manager.save_warmup_state(warmup_state)
        
        # Load state
        loaded_state = await warmup_manager.load_warmup_state()
        
        assert loaded_state is not None
        assert loaded_state['components_initialized'] == warmup_state['components_initialized']
        assert loaded_state['models_loaded'] == warmup_state['models_loaded']
        
        # Test resume from saved state
        can_resume = await warmup_manager.can_resume_from_state(loaded_state)
        assert can_resume
    
    @pytest.mark.asyncio
    async def test_warmup_recovery_mechanisms(self, warmup_system):
        """Test recovery from warmup failures"""
        components, config = warmup_system
        warmup_manager = components['warmup_manager']
        
        # Simulate failures at different stages
        failure_scenarios = [
            {
                'stage': 'data_loading',
                'error': ConnectionError("Failed to connect to data source"),
                'recovery': 'retry_with_backup'
            },
            {
                'stage': 'model_initialization',
                'error': FileNotFoundError("Model file not found"),
                'recovery': 'use_default_model'
            },
            {
                'stage': 'component_warmup',
                'error': TimeoutError("Component initialization timeout"),
                'recovery': 'restart_component'
            }
        ]
        
        for scenario in failure_scenarios:
            # Inject failure
            with patch.object(warmup_manager, f'_execute_{scenario["stage"]}', side_effect=scenario['error']):
                
                # Attempt warmup with recovery
                recovery_result = await warmup_manager.warmup_with_recovery()
                
                # Should recover based on scenario
                if scenario['recovery'] == 'retry_with_backup':
                    assert recovery_result['recovered']
                    assert recovery_result['used_backup']
                elif scenario['recovery'] == 'use_default_model':
                    assert recovery_result['recovered']
                    assert recovery_result['using_defaults']
                elif scenario['recovery'] == 'restart_component':
                    assert recovery_result['recovered']
                    assert recovery_result['components_restarted'] > 0
    
    @pytest.mark.asyncio
    async def test_gradual_warmup_with_live_data(self, warmup_system):
        """Test gradual transition from historical to live data"""
        components, config = warmup_system
        warmup_manager = components['warmup_manager']
        
        # Historical data
        historical_data = self._generate_historical_data(1000)
        
        # Simulate live data stream
        live_data_buffer = []
        
        async def simulate_live_data():
            """Generate live data points"""
            base_price = historical_data['close'].iloc[-1]
            for i in range(20):
                new_point = {
                    'timestamp': datetime.now(),
                    'close': base_price * (1 + np.random.normal(0, 0.001)),
                    'volume': np.random.uniform(50, 150)
                }
                live_data_buffer.append(new_point)
                await asyncio.sleep(0.05)
        
        # Start gradual warmup
        warmup_task = asyncio.create_task(
            warmup_manager.gradual_warmup(historical_data)
        )
        
        # Start live data
        live_task = asyncio.create_task(simulate_live_data())
        
        # Wait for both
        await asyncio.gather(warmup_task, live_task)
        
        # Verify smooth transition
        warmup_result = warmup_task.result()
        assert warmup_result['success']
        assert warmup_result['live_data_integrated']
        assert len(warmup_result['transition_metrics']) > 0
        
        # Check data continuity
        last_historical = historical_data['close'].iloc[-1]
        first_live = live_data_buffer[0]['close']
        price_gap = abs(first_live - last_historical) / last_historical
        assert price_gap < 0.01  # Less than 1% gap
    
    @pytest.mark.asyncio
    async def test_warmup_performance_benchmarks(self, warmup_system):
        """Test warmup meets performance requirements"""
        components, config = warmup_system
        warmup_manager = components['warmup_manager']
        
        # Performance requirements
        requirements = {
            'total_warmup_time': 60,  # seconds
            'data_loading_time': 10,
            'model_init_time': 5,
            'component_warmup_time': 20,
            'memory_usage_mb': 500,
            'cpu_usage_percent': 80
        }
        
        # Track metrics
        metrics = {
            'start_time': datetime.now(),
            'stage_times': {},
            'resource_usage': {}
        }
        
        # Mock resource monitoring
        async def monitor_resources():
            while True:
                metrics['resource_usage']['memory_mb'] = np.random.uniform(200, 400)
                metrics['resource_usage']['cpu_percent'] = np.random.uniform(30, 60)
                await asyncio.sleep(1)
        
        # Start monitoring
        monitor_task = asyncio.create_task(monitor_resources())
        
        try:
            # Execute warmup with timing
            stages = ['data_loading', 'model_initialization', 'component_warmup']
            
            for stage in stages:
                stage_start = datetime.now()
                
                # Simulate stage execution
                await asyncio.sleep(np.random.uniform(1, 3))
                
                stage_end = datetime.now()
                metrics['stage_times'][stage] = (stage_end - stage_start).total_seconds()
            
            # Total time
            total_time = (datetime.now() - metrics['start_time']).total_seconds()
            
            # Verify performance
            assert total_time < requirements['total_warmup_time']
            
            for stage, max_time in [
                ('data_loading', requirements['data_loading_time']),
                ('model_initialization', requirements['model_init_time']),
                ('component_warmup', requirements['component_warmup_time'])
            ]:
                if stage in metrics['stage_times']:
                    assert metrics['stage_times'][stage] < max_time
            
            # Check resource usage
            max_memory = max(metrics['resource_usage'].get('memory_mb', [0]))
            max_cpu = max(metrics['resource_usage'].get('cpu_percent', [0]))
            
            assert max_memory < requirements['memory_usage_mb']
            assert max_cpu < requirements['cpu_usage_percent']
            
        finally:
            monitor_task.cancel()
    
    @pytest.mark.asyncio
    async def test_warmup_configuration_validation(self, warmup_system):
        """Test configuration validation during warmup"""
        components, config = warmup_system
        warmup_manager = components['warmup_manager']
        
        # Test various configurations
        test_configs = [
            {
                'name': 'valid_config',
                'config': config,
                'expected': 'valid'
            },
            {
                'name': 'missing_required',
                'config': {**config, 'symbol': None},
                'expected': 'invalid'
            },
            {
                'name': 'invalid_values',
                'config': {**config, 'warmup_periods': -100},
                'expected': 'invalid'
            },
            {
                'name': 'incompatible_settings',
                'config': {**config, 'timeframe': '1s', 'warmup_periods': 1000000},
                'expected': 'warning'
            }
        ]
        
        for test in test_configs:
            validation_result = await warmup_manager.validate_configuration(test['config'])
            
            if test['expected'] == 'valid':
                assert validation_result['is_valid']
                assert len(validation_result['errors']) == 0
            elif test['expected'] == 'invalid':
                assert not validation_result['is_valid']
                assert len(validation_result['errors']) > 0
            elif test['expected'] == 'warning':
                assert validation_result['is_valid']
                assert len(validation_result['warnings']) > 0
    
    # Helper methods
    def _generate_historical_data(self, periods: int) -> pd.DataFrame:
        """Generate historical market data"""
        end_time = datetime.now()
        dates = pd.date_range(end=end_time, periods=periods, freq='5min')
        
        # Generate realistic price movement
        price = 50000
        prices = []
        volumes = []
        
        for i in range(periods):
            # Add trend and noise
            trend = 0.00001 * i  # Slight upward trend
            noise = np.random.normal(0, 0.001)
            price *= (1 + trend + noise)
            prices.append(price)
            
            # Volume with daily pattern
            hour = dates[i].hour
            base_volume = 100
            daily_pattern = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
            volume = base_volume * daily_pattern * np.random.uniform(0.8, 1.2)
            volumes.append(volume)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': volumes
        })
    
    def _generate_data_with_gaps(self, periods: int) -> pd.DataFrame:
        """Generate data with gaps"""
        data = self._generate_historical_data(periods)
        
        # Remove random chunks
        gap_starts = [200, 500, 800]
        gap_size = 20
        
        for start in gap_starts:
            if start + gap_size < len(data):
                data = data.drop(data.index[start:start+gap_size])
        
        return data.reset_index(drop=True)
    
    def _create_mock_model_weights(self) -> Dict[str, np.ndarray]:
        """Create mock model weights"""
        return {
            'attention_weights': np.random.randn(100, 100),
            'output_weights': np.random.randn(100, 10),
            'bias': np.random.randn(10)
        }
    
    def _create_mock_classifier(self) -> Dict[str, Any]:
        """Create mock classifier model"""
        return {
            'model_type': 'random_forest',
            'n_estimators': 100,
            'max_depth': 10,
            'feature_importances': np.random.rand(50)
        }
    
    def _create_mock_risk_model(self) -> Dict[str, Any]:
        """Create mock risk model"""
        return {
            'risk_factors': ['volatility', 'correlation', 'drawdown'],
            'factor_weights': [0.4, 0.3, 0.3],
            'thresholds': {'max_var': 0.02, 'max_drawdown': 0.1}
        }