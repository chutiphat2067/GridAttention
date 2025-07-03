"""
Test suite for GridAttention Strategy Selection functionality
Tests grid strategy selection, adaptation, and optimization
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional, Tuple
import random

# GridAttention imports - aligned with system structure
from core.attention_learning_layer import AttentionLearningLayer
from core.grid_strategy_selector import GridStrategySelector
from core.market_regime_detector import MarketRegimeDetector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor

# Test utilities
from tests.fixtures.market_data import (
    generate_market_data,
    generate_order_book,
    create_price_series
)
from tests.mocks.mock_exchange import MockExchange
from tests.utils.test_helpers import async_test, create_test_config


class TestStrategySelection:
    """Test grid strategy selection and adaptation"""
    
    @pytest.fixture
    def strategy_selector(self):
        """Create strategy selector instance"""
        config = create_test_config()
        return GridStrategySelector(config)
    
    @pytest.fixture
    def market_regime_detector(self):
        """Create market regime detector"""
        config = create_test_config()
        return MarketRegimeDetector(config)
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange"""
        return MockExchange()
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data"""
        return generate_market_data(
            symbol="BTC/USDT",
            periods=1000,
            interval="1h"
        )
    
    # Basic Strategy Selection Tests
    
    @async_test
    async def test_strategy_initialization(self, strategy_selector):
        """Test strategy selector initialization"""
        assert strategy_selector is not None
        assert hasattr(strategy_selector, 'strategies')
        assert hasattr(strategy_selector, 'select_strategy')
        assert len(strategy_selector.strategies) > 0
    
    @async_test
    async def test_default_strategy_selection(self, strategy_selector):
        """Test default strategy selection without market data"""
        strategy = await strategy_selector.select_strategy()
        
        assert strategy is not None
        assert 'type' in strategy
        assert 'parameters' in strategy
        assert strategy['type'] in ['conservative', 'balanced', 'aggressive', 'adaptive']
    
    @async_test
    async def test_strategy_selection_with_market_regime(
        self, strategy_selector, market_regime_detector, sample_market_data
    ):
        """Test strategy selection based on market regime"""
        # Detect market regime
        regime = await market_regime_detector.detect_regime(sample_market_data)
        
        # Select strategy based on regime
        strategy = await strategy_selector.select_strategy(
            market_regime=regime,
            market_data=sample_market_data
        )
        
        assert strategy is not None
        assert strategy['type'] == self._expected_strategy_for_regime(regime)
    
    def _expected_strategy_for_regime(self, regime: str) -> str:
        """Map regime to expected strategy type"""
        regime_strategy_map = {
            'trending_up': 'aggressive',
            'trending_down': 'conservative',
            'ranging': 'balanced',
            'volatile': 'adaptive'
        }
        return regime_strategy_map.get(regime, 'balanced')
    
    # Grid Configuration Tests
    
    @async_test
    async def test_grid_parameters_generation(self, strategy_selector):
        """Test grid parameters generation for different strategies"""
        strategies = ['conservative', 'balanced', 'aggressive', 'adaptive']
        
        for strategy_type in strategies:
            params = await strategy_selector.generate_grid_parameters(
                strategy_type=strategy_type,
                price=50000,
                volatility=0.02
            )
            
            assert 'grid_levels' in params
            assert 'grid_spacing' in params
            assert 'position_size' in params
            assert 'take_profit' in params
            assert 'stop_loss' in params
            
            # Validate parameter ranges
            assert 5 <= params['grid_levels'] <= 50
            assert 0.001 <= params['grid_spacing'] <= 0.1
            assert 0.01 <= params['position_size'] <= 1.0
    
    @async_test
    async def test_adaptive_grid_spacing(self, strategy_selector):
        """Test adaptive grid spacing based on volatility"""
        base_price = 50000
        volatilities = [0.01, 0.02, 0.05, 0.1]  # Low to high volatility
        
        spacings = []
        for vol in volatilities:
            params = await strategy_selector.generate_grid_parameters(
                strategy_type='adaptive',
                price=base_price,
                volatility=vol
            )
            spacings.append(params['grid_spacing'])
        
        # Higher volatility should result in wider spacing
        assert spacings == sorted(spacings)
    
    # Performance-Based Selection Tests
    
    @async_test
    async def test_performance_based_strategy_selection(
        self, strategy_selector, performance_monitor
    ):
        """Test strategy selection based on historical performance"""
        # Mock performance data
        performance_data = {
            'conservative': {'sharpe_ratio': 1.2, 'win_rate': 0.65, 'profit': 1000},
            'balanced': {'sharpe_ratio': 1.5, 'win_rate': 0.60, 'profit': 1500},
            'aggressive': {'sharpe_ratio': 0.8, 'win_rate': 0.45, 'profit': 2000},
            'adaptive': {'sharpe_ratio': 1.8, 'win_rate': 0.70, 'profit': 1800}
        }
        
        with patch.object(performance_monitor, 'get_strategy_performance', 
                         return_value=performance_data):
            strategy = await strategy_selector.select_strategy(
                selection_mode='performance_based',
                performance_monitor=performance_monitor
            )
            
            # Should select adaptive (highest Sharpe ratio)
            assert strategy['type'] == 'adaptive'
    
    @async_test
    async def test_risk_adjusted_strategy_selection(
        self, strategy_selector, risk_management_system
    ):
        """Test strategy selection with risk constraints"""
        # Mock risk metrics
        risk_metrics = {
            'current_drawdown': 0.15,
            'var_95': 0.05,
            'position_limit': 0.3
        }
        
        with patch.object(risk_management_system, 'get_risk_metrics', 
                         return_value=risk_metrics):
            strategy = await strategy_selector.select_strategy(
                risk_management=risk_management_system,
                max_drawdown=0.20
            )
            
            # Should select conservative due to high current drawdown
            assert strategy['type'] in ['conservative', 'balanced']
    
    # Market Condition Adaptation Tests
    
    @async_test
    async def test_volatility_based_adaptation(self, strategy_selector):
        """Test strategy adaptation based on volatility changes"""
        # Simulate changing volatility
        volatility_scenarios = [
            (0.01, 'aggressive'),    # Low volatility
            (0.05, 'balanced'),      # Medium volatility
            (0.15, 'conservative')   # High volatility
        ]
        
        for volatility, expected_strategy in volatility_scenarios:
            strategy = await strategy_selector.select_strategy(
                market_volatility=volatility
            )
            assert strategy['type'] == expected_strategy
    
    @async_test
    async def test_trend_strength_adaptation(self, strategy_selector):
        """Test strategy adaptation based on trend strength"""
        # Simulate different trend strengths
        trend_scenarios = [
            (0.8, 'aggressive'),    # Strong trend
            (0.4, 'balanced'),      # Moderate trend
            (0.1, 'adaptive')       # Weak/no trend
        ]
        
        for trend_strength, expected_strategy in trend_scenarios:
            strategy = await strategy_selector.select_strategy(
                trend_strength=trend_strength,
                market_regime='trending_up'
            )
            assert strategy['type'] == expected_strategy
    
    # Multi-Factor Selection Tests
    
    @async_test
    async def test_multi_factor_strategy_selection(
        self, strategy_selector, market_regime_detector, 
        performance_monitor, risk_management_system
    ):
        """Test strategy selection with multiple factors"""
        # Setup multi-factor inputs
        factors = {
            'market_regime': 'ranging',
            'volatility': 0.03,
            'trend_strength': 0.2,
            'performance_data': {
                'recent_sharpe': 1.5,
                'win_rate': 0.6
            },
            'risk_metrics': {
                'current_drawdown': 0.08,
                'var_95': 0.03
            }
        }
        
        strategy = await strategy_selector.select_strategy_multi_factor(
            factors=factors,
            weights={
                'regime': 0.3,
                'volatility': 0.2,
                'performance': 0.3,
                'risk': 0.2
            }
        )
        
        assert strategy is not None
        assert 'type' in strategy
        assert 'confidence' in strategy
        assert 0 <= strategy['confidence'] <= 1
    
    # Strategy Switching Tests
    
    @async_test
    async def test_strategy_switching_cooldown(self, strategy_selector):
        """Test cooldown period for strategy switching"""
        # Select initial strategy
        strategy1 = await strategy_selector.select_strategy()
        initial_type = strategy1['type']
        
        # Try to switch immediately
        strategy2 = await strategy_selector.select_strategy(
            force_different=True
        )
        
        # Should not switch due to cooldown
        assert strategy2['type'] == initial_type
        
        # Simulate cooldown expiry
        strategy_selector.last_switch_time = datetime.now() - timedelta(hours=2)
        
        # Now should allow switching
        strategy3 = await strategy_selector.select_strategy(
            force_different=True
        )
        assert strategy3['type'] != initial_type
    
    @async_test
    async def test_gradual_strategy_transition(self, strategy_selector):
        """Test gradual transition between strategies"""
        # Start with conservative
        await strategy_selector.set_current_strategy('conservative')
        
        # Request transition to aggressive
        transition_steps = await strategy_selector.plan_strategy_transition(
            target_strategy='aggressive',
            transition_periods=5
        )
        
        assert len(transition_steps) == 5
        assert transition_steps[0]['type'] == 'conservative'
        assert transition_steps[-1]['type'] == 'aggressive'
        
        # Verify gradual parameter changes
        grid_levels = [step['parameters']['grid_levels'] 
                      for step in transition_steps]
        assert grid_levels == sorted(grid_levels, reverse=True)
    
    # Backtesting Strategy Selection Tests
    
    @async_test
    async def test_backtest_strategy_selection(
        self, strategy_selector, sample_market_data
    ):
        """Test strategy selection using backtesting results"""
        # Run backtest for each strategy
        backtest_results = {}
        
        for strategy_type in ['conservative', 'balanced', 'aggressive']:
            results = await strategy_selector.backtest_strategy(
                strategy_type=strategy_type,
                market_data=sample_market_data,
                initial_capital=10000
            )
            backtest_results[strategy_type] = results
        
        # Select best performing strategy
        best_strategy = await strategy_selector.select_from_backtest(
            backtest_results=backtest_results,
            optimization_metric='sharpe_ratio'
        )
        
        assert best_strategy is not None
        assert best_strategy['type'] in backtest_results.keys()
    
    # Edge Cases and Error Handling
    
    @async_test
    async def test_strategy_selection_with_insufficient_data(
        self, strategy_selector
    ):
        """Test strategy selection with insufficient market data"""
        insufficient_data = pd.DataFrame({
            'close': [50000, 50100],  # Only 2 data points
            'volume': [100, 110]
        })
        
        strategy = await strategy_selector.select_strategy(
            market_data=insufficient_data
        )
        
        # Should fall back to default strategy
        assert strategy['type'] == 'balanced'
        assert strategy.get('warning') == 'Insufficient data for full analysis'
    
    @async_test
    async def test_strategy_selection_with_extreme_volatility(
        self, strategy_selector
    ):
        """Test strategy selection under extreme volatility"""
        extreme_volatility = 0.5  # 50% volatility
        
        strategy = await strategy_selector.select_strategy(
            market_volatility=extreme_volatility
        )
        
        # Should select most conservative strategy
        assert strategy['type'] == 'conservative'
        assert strategy['parameters']['grid_levels'] <= 10
        assert strategy['parameters']['position_size'] <= 0.1
    
    @async_test
    async def test_strategy_recovery_after_failure(
        self, strategy_selector
    ):
        """Test strategy recovery after selection failure"""
        # Mock a failure scenario
        with patch.object(strategy_selector, '_calculate_strategy_score',
                         side_effect=Exception("Calculation error")):
            strategy = await strategy_selector.select_strategy(
                fallback_enabled=True
            )
            
            # Should return fallback strategy
            assert strategy['type'] == 'balanced'
            assert strategy.get('fallback') is True
    
    # Integration Tests
    
    @async_test
    async def test_strategy_selection_integration(
        self, strategy_selector, market_regime_detector,
        performance_monitor, risk_management_system,
        mock_exchange, sample_market_data
    ):
        """Test full integration of strategy selection components"""
        # Setup integrated components
        attention_layer = GridAttentionLayer({
            'strategy_selector': strategy_selector,
            'regime_detector': market_regime_detector,
            'performance_monitor': performance_monitor,
            'risk_management': risk_management_system
        })
        
        # Run integrated strategy selection
        selected_strategy = await attention_layer.select_optimal_strategy(
            market_data=sample_market_data,
            current_positions=mock_exchange.get_positions(),
            risk_limits={'max_drawdown': 0.2, 'max_positions': 10}
        )
        
        assert selected_strategy is not None
        assert 'type' in selected_strategy
        assert 'parameters' in selected_strategy
        assert 'metadata' in selected_strategy
        
        # Verify metadata includes decision factors
        metadata = selected_strategy['metadata']
        assert 'regime' in metadata
        assert 'volatility' in metadata
        assert 'risk_score' in metadata
        assert 'confidence' in metadata


class TestStrategyOptimization:
    """Test strategy parameter optimization"""
    
    @pytest.fixture
    def optimizer(self):
        """Create strategy optimizer"""
        config = create_test_config()
        return GridStrategySelector(config).optimizer
    
    @async_test
    async def test_grid_parameter_optimization(self, optimizer):
        """Test optimization of grid parameters"""
        # Define optimization constraints
        constraints = {
            'min_grid_levels': 5,
            'max_grid_levels': 30,
            'min_spacing': 0.001,
            'max_spacing': 0.05,
            'target_sharpe': 1.5
        }
        
        # Run optimization
        optimal_params = await optimizer.optimize_parameters(
            strategy_type='balanced',
            constraints=constraints,
            optimization_periods=100
        )
        
        assert constraints['min_grid_levels'] <= optimal_params['grid_levels'] <= constraints['max_grid_levels']
        assert constraints['min_spacing'] <= optimal_params['grid_spacing'] <= constraints['max_spacing']
    
    @async_test
    async def test_multi_objective_optimization(self, optimizer):
        """Test multi-objective parameter optimization"""
        objectives = {
            'maximize_returns': 0.4,
            'minimize_drawdown': 0.3,
            'maximize_win_rate': 0.3
        }
        
        optimal_params = await optimizer.multi_objective_optimize(
            objectives=objectives,
            strategy_type='adaptive'
        )
        
        assert optimal_params is not None
        assert 'pareto_front' in optimal_params
        assert len(optimal_params['pareto_front']) > 0


class TestStrategyMonitoring:
    """Test strategy performance monitoring and adaptation"""
    
    @pytest.fixture
    def strategy_monitor(self):
        """Create strategy monitor"""
        return StrategyMonitor(create_test_config())
    
    @async_test
    async def test_real_time_strategy_monitoring(self, strategy_monitor):
        """Test real-time monitoring of strategy performance"""
        # Start monitoring
        await strategy_monitor.start_monitoring('aggressive')
        
        # Simulate trades
        for i in range(10):
            trade = {
                'type': 'buy' if i % 2 == 0 else 'sell',
                'price': 50000 + random.randint(-1000, 1000),
                'size': 0.1,
                'profit': random.uniform(-50, 100)
            }
            await strategy_monitor.record_trade(trade)
        
        # Get performance metrics
        metrics = await strategy_monitor.get_current_metrics()
        
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'average_profit' in metrics
        assert 'sharpe_ratio' in metrics
        assert metrics['total_trades'] == 10
    
    @async_test
    async def test_strategy_adaptation_trigger(self, strategy_monitor):
        """Test automatic strategy adaptation triggers"""
        # Set adaptation thresholds
        thresholds = {
            'min_win_rate': 0.5,
            'max_drawdown': 0.15,
            'min_sharpe_ratio': 1.0
        }
        
        await strategy_monitor.set_adaptation_thresholds(thresholds)
        
        # Simulate poor performance
        for _ in range(20):
            await strategy_monitor.record_trade({
                'profit': -100,
                'type': 'loss'
            })
        
        # Check if adaptation is triggered
        should_adapt = await strategy_monitor.should_adapt_strategy()
        assert should_adapt is True
        
        adaptation_reason = await strategy_monitor.get_adaptation_reason()
        assert 'win_rate' in adaptation_reason or 'drawdown' in adaptation_reason