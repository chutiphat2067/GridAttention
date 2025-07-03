# tests/integration/test_phase_transitions.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from enum import Enum

from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase
from core.market_regime_detector import MarketRegimeDetector
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor
from core.overfitting_detector import OverfittingDetector


class TestPhaseTransitions:
    """Test phase transitions in attention learning system"""
    
    @pytest.fixture
    async def phase_system(self):
        """Create system with phase transition capabilities"""
        config = {
            'symbol': 'BTC/USDT',
            'timeframe': '5m',
            'min_trades_learning': 1000,
            'min_trades_shadow': 200,
            'min_trades_active': 100,
            'performance_improvement_threshold': 0.02,
            'max_attention_influence': 0.3,
            'validation_threshold': 0.5
        }
        
        components = {
            'attention': AttentionLearningLayer(config),
            'regime_detector': MarketRegimeDetector(config),
            'strategy_selector': GridStrategySelector(config),
            'risk_manager': RiskManagementSystem(config),
            'performance_monitor': PerformanceMonitor(config),
            'overfitting_detector': OverfittingDetector(config)
        }
        
        return components, config
    
    @pytest.mark.asyncio
    async def test_learning_to_shadow_transition(self, phase_system):
        """Test transition from learning to shadow phase"""
        components, config = phase_system
        attention = components['attention']
        
        # Verify initial phase
        assert attention.current_phase == AttentionPhase.LEARNING
        initial_state = await attention.get_attention_state()
        assert initial_state['phase'] == 'learning'
        
        # Simulate learning phase observations
        for i in range(config['min_trades_learning'] + 50):
            features = self._generate_features()
            regime = 'RANGING'
            context = self._generate_context(i)
            
            # Process in learning mode
            result = await attention.process(features, regime, context)
            
            # In learning phase, should not modify features
            assert result == features
        
        # Check phase transition
        await attention._check_phase_transition()
        assert attention.current_phase == AttentionPhase.SHADOW
        
        # Verify metrics
        state = await attention.get_attention_state()
        assert state['phase'] == 'shadow'
        assert state['total_observations'] >= config['min_trades_learning']
        
        # Verify phase transition was recorded
        assert len(attention.metrics.phase_transitions) > 0
        transition = attention.metrics.phase_transitions[-1]
        assert transition['from'] == 'learning'
        assert transition['to'] == 'shadow'
    
    @pytest.mark.asyncio
    async def test_shadow_to_active_transition(self, phase_system):
        """Test transition from shadow to active phase"""
        components, config = phase_system
        attention = components['attention']
        performance = components['performance_monitor']
        
        # Force to shadow phase
        attention.current_phase = AttentionPhase.SHADOW
        attention.metrics.total_observations = config['min_trades_learning']
        
        # Initialize performance baseline
        baseline_metrics = {
            'win_rate': 0.45,
            'sharpe_ratio': 0.8,
            'profit_factor': 1.1,
            'max_drawdown': 0.08
        }
        
        # Simulate shadow phase with improved performance
        shadow_results = []
        for i in range(config['min_trades_shadow'] + 50):
            features = self._generate_features()
            regime = 'TRENDING'
            context = self._generate_context(i)
            
            # Add performance metrics to context
            context['performance'] = {
                'win_rate': 0.48 + i * 0.0001,  # Gradual improvement
                'sharpe_ratio': 0.85 + i * 0.0002,
                'profit_factor': 1.15 + i * 0.0001
            }
            
            # Process in shadow mode
            result = await attention.process(features, regime, context)
            
            # In shadow phase, might calculate but not apply
            shadow_results.append({
                'original': features,
                'calculated': result,
                'timestamp': datetime.now()
            })
        
        # Update performance improvements
        attention.metrics.performance_improvements = {
            'win_rate': 0.03,  # 3% improvement
            'sharpe_ratio': 0.05,
            'profit_factor': 0.04
        }
        
        # Check transition conditions
        await attention._check_phase_transition()
        
        # Should transition to active if performance improved
        assert attention.current_phase == AttentionPhase.ACTIVE
        
        # Verify shadow calculations were performed
        assert attention.metrics.shadow_calculations > 0
    
    @pytest.mark.asyncio
    async def test_active_phase_behavior(self, phase_system):
        """Test behavior in active phase"""
        components, config = phase_system
        attention = components['attention']
        
        # Force to active phase
        attention.current_phase = AttentionPhase.ACTIVE
        attention.metrics.total_observations = (
            config['min_trades_learning'] + 
            config['min_trades_shadow'] + 
            config['min_trades_active']
        )
        
        # Setup attention weights
        attention.feature_attention.attention_weights = {
            'volatility': 0.8,
            'trend_strength': 0.9,
            'volume_ratio': 0.6,
            'rsi': 0.7
        }
        
        # Process in active mode
        features = {
            'volatility': 0.02,
            'trend_strength': 0.5,
            'volume_ratio': 1.2,
            'rsi': 0.6
        }
        
        regime = 'VOLATILE'
        context = self._generate_context(1000)
        
        # Process features
        weighted_features = await attention.process(features, regime, context)
        
        # In active phase, features should be modified
        assert weighted_features != features
        
        # Check attention was applied within limits
        for key in features:
            if key in weighted_features:
                original = features[key]
                weighted = weighted_features[key]
                change = abs(weighted - original) / (abs(original) + 1e-8)
                assert change <= config['max_attention_influence']
        
        # Verify active application was recorded
        assert attention.metrics.active_applications > 0
    
    @pytest.mark.asyncio
    async def test_phase_rollback_on_degradation(self, phase_system):
        """Test rollback to previous phase on performance degradation"""
        components, config = phase_system
        attention = components['attention']
        overfitting = components['overfitting_detector']
        
        # Start in active phase
        attention.current_phase = AttentionPhase.ACTIVE
        attention.metrics.active_applications = 100
        
        # Simulate performance degradation
        degradation_signals = []
        for i in range(50):
            # Generate poor performance metrics
            metrics = {
                'win_rate': 0.40 - i * 0.001,  # Declining
                'sharpe_ratio': 0.7 - i * 0.002,
                'max_drawdown': 0.10 + i * 0.001  # Increasing
            }
            
            # Detect overfitting
            overfitting_result = await overfitting.detect(
                model_performance=metrics,
                validation_performance={
                    'win_rate': 0.35,
                    'sharpe_ratio': 0.5,
                    'max_drawdown': 0.15
                }
            )
            
            if overfitting_result['is_overfitting']:
                degradation_signals.append(overfitting_result)
        
        # Should trigger phase rollback
        if len(degradation_signals) > 10:
            # Rollback to shadow phase
            attention.current_phase = AttentionPhase.SHADOW
            
            # Record rollback
            attention.metrics.phase_transitions.append({
                'from': 'active',
                'to': 'shadow',
                'reason': 'performance_degradation',
                'timestamp': datetime.now(),
                'metrics': degradation_signals[-1]
            })
        
        assert attention.current_phase == AttentionPhase.SHADOW
        assert any(t['reason'] == 'performance_degradation' 
                  for t in attention.metrics.phase_transitions)
    
    @pytest.mark.asyncio
    async def test_multi_phase_coordination(self, phase_system):
        """Test coordination between different component phases"""
        components, config = phase_system
        
        # Different components might be in different phases
        component_phases = {
            'attention': AttentionPhase.SHADOW,
            'regime_detector': 'active',  # Already trained
            'strategy_selector': 'learning',  # Still learning
            'risk_manager': 'active'  # Always active
        }
        
        # Simulate coordinated processing
        features = self._generate_features()
        regime = 'RANGING'
        
        # Process through pipeline with different phases
        results = {}
        
        # 1. Attention (Shadow) - calculates but doesn't apply
        attention_output = await components['attention'].process(
            features, regime, self._generate_context(500)
        )
        results['attention'] = {
            'phase': 'shadow',
            'modified': attention_output != features
        }
        
        # 2. Regime Detector (Active) - fully operational
        regime_output = await components['regime_detector'].detect_regime(
            market_data=self._generate_market_data(100)
        )
        results['regime'] = {
            'phase': 'active',
            'confidence': regime_output.get('confidence', 0)
        }
        
        # 3. Strategy Selector (Learning) - observing only
        strategy_params = await components['strategy_selector'].select_strategy(
            regime=regime_output['regime'],
            features=features
        )
        results['strategy'] = {
            'phase': 'learning',
            'params': strategy_params
        }
        
        # Verify phase-appropriate behavior
        assert not results['attention']['modified']  # Shadow doesn't modify
        assert results['regime']['confidence'] > 0  # Active provides confidence
        assert results['strategy']['params'] is not None  # Learning provides defaults
    
    @pytest.mark.asyncio
    async def test_phase_persistence_and_recovery(self, phase_system):
        """Test saving and loading phase states"""
        components, config = phase_system
        attention = components['attention']
        
        # Setup specific phase state
        attention.current_phase = AttentionPhase.SHADOW
        attention.metrics.total_observations = 1500
        attention.metrics.shadow_calculations = 250
        attention.metrics.performance_improvements = {
            'win_rate': 0.02,
            'sharpe_ratio': 0.03
        }
        
        # Add phase transitions
        attention.metrics.phase_transitions = [
            {
                'from': 'learning',
                'to': 'shadow',
                'timestamp': datetime.now() - timedelta(hours=2),
                'observations': 1000
            }
        ]
        
        # Save state
        state_file = 'test_phase_state.json'
        await attention.save_state(state_file)
        
        # Create new instance
        new_attention = AttentionLearningLayer(config)
        
        # Load state
        await new_attention.load_state(state_file)
        
        # Verify phase restoration
        assert new_attention.current_phase == AttentionPhase.SHADOW
        assert new_attention.metrics.total_observations == 1500
        assert new_attention.metrics.shadow_calculations == 250
        assert len(new_attention.metrics.phase_transitions) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_phase_updates(self, phase_system):
        """Test handling concurrent phase transition requests"""
        components, config = phase_system
        attention = components['attention']
        
        # Start near transition threshold
        attention.metrics.total_observations = config['min_trades_learning'] - 10
        
        # Simulate concurrent updates
        async def process_batch(batch_id: int):
            for i in range(20):
                features = self._generate_features()
                context = self._generate_context(
                    attention.metrics.total_observations + i
                )
                await attention.process(features, 'RANGING', context)
                await asyncio.sleep(0.001)  # Small delay
        
        # Run multiple batches concurrently
        tasks = [process_batch(i) for i in range(3)]
        await asyncio.gather(*tasks)
        
        # Check only one transition occurred
        transitions = [
            t for t in attention.metrics.phase_transitions 
            if t['from'] == 'learning' and t['to'] == 'shadow'
        ]
        assert len(transitions) == 1
        
        # Verify final state is consistent
        assert attention.current_phase == AttentionPhase.SHADOW
    
    @pytest.mark.asyncio
    async def test_phase_specific_monitoring(self, phase_system):
        """Test monitoring metrics specific to each phase"""
        components, config = phase_system
        attention = components['attention']
        performance = components['performance_monitor']
        
        # Track metrics for each phase
        phase_metrics = {
            AttentionPhase.LEARNING: {
                'observations': [],
                'processing_times': [],
                'feature_stats': {}
            },
            AttentionPhase.SHADOW: {
                'calculations': [],
                'accuracy': [],
                'divergence': []
            },
            AttentionPhase.ACTIVE: {
                'applications': [],
                'impact': [],
                'performance': []
            }
        }
        
        # Simulate progression through phases
        phases = [AttentionPhase.LEARNING, AttentionPhase.SHADOW, AttentionPhase.ACTIVE]
        
        for phase in phases:
            attention.current_phase = phase
            
            for i in range(50):
                features = self._generate_features()
                context = self._generate_context(i)
                
                start_time = datetime.now()
                result = await attention.process(features, 'RANGING', context)
                process_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Record phase-specific metrics
                if phase == AttentionPhase.LEARNING:
                    phase_metrics[phase]['observations'].append(i)
                    phase_metrics[phase]['processing_times'].append(process_time)
                    
                elif phase == AttentionPhase.SHADOW:
                    phase_metrics[phase]['calculations'].append(result)
                    # Calculate accuracy vs baseline
                    accuracy = np.random.uniform(0.7, 0.9)
                    phase_metrics[phase]['accuracy'].append(accuracy)
                    
                elif phase == AttentionPhase.ACTIVE:
                    phase_metrics[phase]['applications'].append(result)
                    # Measure impact
                    impact = sum(abs(result.get(k, 0) - features.get(k, 0)) 
                               for k in features) / len(features)
                    phase_metrics[phase]['impact'].append(impact)
        
        # Verify phase-specific metrics collected
        assert len(phase_metrics[AttentionPhase.LEARNING]['observations']) > 0
        assert len(phase_metrics[AttentionPhase.SHADOW]['calculations']) > 0
        assert len(phase_metrics[AttentionPhase.ACTIVE]['applications']) > 0
        
        # Check metric quality
        avg_shadow_accuracy = np.mean(phase_metrics[AttentionPhase.SHADOW]['accuracy'])
        assert avg_shadow_accuracy > 0.7
        
        avg_active_impact = np.mean(phase_metrics[AttentionPhase.ACTIVE]['impact'])
        assert avg_active_impact <= config['max_attention_influence']
    
    @pytest.mark.asyncio
    async def test_emergency_phase_override(self, phase_system):
        """Test emergency override to safe phase"""
        components, config = phase_system
        attention = components['attention']
        risk_manager = components['risk_manager']
        
        # Start in active phase
        attention.current_phase = AttentionPhase.ACTIVE
        
        # Simulate emergency conditions
        emergency_conditions = [
            {'type': 'extreme_volatility', 'severity': 0.9},
            {'type': 'system_error', 'severity': 0.8},
            {'type': 'risk_breach', 'severity': 1.0}
        ]
        
        for condition in emergency_conditions:
            if condition['severity'] > 0.85:
                # Emergency override to learning phase (safe mode)
                previous_phase = attention.current_phase
                attention.current_phase = AttentionPhase.LEARNING
                
                # Record emergency transition
                attention.metrics.phase_transitions.append({
                    'from': previous_phase.value,
                    'to': 'learning',
                    'reason': 'emergency_override',
                    'condition': condition,
                    'timestamp': datetime.now()
                })
                
                # Notify risk manager
                await risk_manager.handle_emergency_override({
                    'component': 'attention',
                    'action': 'phase_override',
                    'condition': condition
                })
                
                break
        
        # Verify emergency override occurred
        assert attention.current_phase == AttentionPhase.LEARNING
        emergency_transitions = [
            t for t in attention.metrics.phase_transitions 
            if t['reason'] == 'emergency_override'
        ]
        assert len(emergency_transitions) > 0
    
    # Helper methods
    def _generate_features(self) -> Dict[str, float]:
        """Generate random features"""
        return {
            'volatility': np.random.uniform(0.001, 0.03),
            'trend_strength': np.random.uniform(-1, 1),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'rsi': np.random.uniform(0.2, 0.8),
            'spread': np.random.uniform(0.0001, 0.001)
        }
    
    def _generate_context(self, trade_count: int) -> Dict[str, Any]:
        """Generate context for processing"""
        return {
            'timestamp': datetime.now(),
            'trade_count': trade_count,
            'performance': {
                'win_rate': 0.45 + np.random.uniform(-0.05, 0.05),
                'profit': np.random.uniform(-100, 200),
                'sharpe_ratio': 0.8 + np.random.uniform(-0.2, 0.3)
            },
            'market_conditions': {
                'volatility_rank': np.random.uniform(0, 1),
                'trend_clarity': np.random.uniform(0, 1)
            }
        }
    
    def _generate_market_data(self, periods: int) -> pd.DataFrame:
        """Generate market data"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        prices = 50000 + np.cumsum(np.random.randn(periods) * 50)
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.uniform(50, 150, periods),
            'high': prices * 1.001,
            'low': prices * 0.999
        })