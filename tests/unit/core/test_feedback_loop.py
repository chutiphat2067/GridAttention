"""
Unit tests for the Feedback Loop System.

Tests cover:
- Learning from historical performance
- Parameter optimization and adaptation
- Strategy improvement mechanisms
- Performance pattern recognition
- Adaptive decision making
- Model retraining triggers
- Feedback aggregation and weighting
- Continuous improvement tracking
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
from sklearn.ensemble import RandomForestRegressor
import json

# Assuming the module structure
from src.core.feedback_loop import (
    FeedbackLoop,
    FeedbackConfig,
    PerformanceFeedback,
    ParameterOptimizer,
    LearningEngine,
    AdaptiveController,
    FeedbackMetrics,
    ImprovementTracker,
    FeedbackType,
    AdaptationStrategy
)


class TestFeedbackConfig:
    """Test cases for FeedbackConfig validation."""
    
    def test_default_config(self):
        """Test default feedback configuration."""
        config = FeedbackConfig()
        
        assert config.learning_rate == 0.01
        assert config.adaptation_threshold == 0.05  # 5% performance change
        assert config.feedback_window == 100  # Last 100 trades
        assert config.min_samples_required == 30
        assert config.optimization_frequency == 'daily'
        assert config.enable_online_learning is True
        
    def test_custom_config(self):
        """Test custom feedback configuration."""
        config = FeedbackConfig(
            learning_rate=0.001,
            adaptation_threshold=0.03,
            feedback_window=200,
            optimization_method='bayesian',
            enable_reinforcement_learning=True
        )
        
        assert config.learning_rate == 0.001
        assert config.optimization_method == 'bayesian'
        assert config.enable_reinforcement_learning is True
        
    def test_config_validation(self):
        """Test configuration validation rules."""
        # Invalid learning rate
        with pytest.raises(ValueError, match="learning_rate"):
            FeedbackConfig(learning_rate=-0.01)
            
        # Invalid adaptation threshold
        with pytest.raises(ValueError, match="adaptation_threshold"):
            FeedbackConfig(adaptation_threshold=1.5)  # > 100%
            
        # Invalid feedback window
        with pytest.raises(ValueError, match="feedback_window"):
            FeedbackConfig(feedback_window=0)
            
        # Invalid optimization frequency
        with pytest.raises(ValueError, match="optimization_frequency"):
            FeedbackConfig(optimization_frequency='invalid')


class TestPerformanceFeedback:
    """Test cases for performance feedback collection."""
    
    def test_feedback_creation(self):
        """Test feedback data structure creation."""
        feedback = PerformanceFeedback(
            timestamp=datetime.now(),
            trade_id='trade_001',
            strategy='grid',
            symbol='BTC/USDT',
            outcome='win',
            pnl=Decimal('150'),
            return_pct=Decimal('0.03'),
            market_conditions={
                'volatility': 0.02,
                'trend': 'bullish',
                'volume': 'high'
            },
            parameters_used={
                'grid_spacing': 0.01,
                'num_levels': 10,
                'stop_loss': 0.02
            }
        )
        
        assert feedback.trade_id == 'trade_001'
        assert feedback.outcome == 'win'
        assert feedback.return_pct == Decimal('0.03')
        assert feedback.market_conditions['volatility'] == 0.02
        
    def test_feedback_validation(self):
        """Test feedback data validation."""
        # Invalid outcome
        with pytest.raises(ValueError, match="outcome must be"):
            PerformanceFeedback(
                timestamp=datetime.now(),
                trade_id='001',
                outcome='invalid',
                pnl=Decimal('100')
            )
            
        # Missing required fields
        with pytest.raises(ValueError, match="required fields"):
            PerformanceFeedback(
                timestamp=datetime.now(),
                trade_id='001'
                # Missing outcome and pnl
            )
            
    def test_feedback_aggregation(self):
        """Test aggregating multiple feedback items."""
        feedbacks = []
        
        # Create feedback items
        for i in range(10):
            feedback = PerformanceFeedback(
                timestamp=datetime.now() - timedelta(hours=i),
                trade_id=f'trade_{i}',
                strategy='grid',
                outcome='win' if i % 3 != 0 else 'loss',
                pnl=Decimal('100') if i % 3 != 0 else Decimal('-50'),
                return_pct=Decimal('0.02') if i % 3 != 0 else Decimal('-0.01'),
                parameters_used={'grid_spacing': 0.01}
            )
            feedbacks.append(feedback)
            
        # Aggregate statistics
        aggregator = PerformanceFeedback.aggregate(feedbacks)
        
        assert aggregator['total_trades'] == 10
        assert aggregator['win_rate'] == 0.7  # 7/10
        assert aggregator['average_return'] > 0
        assert 'parameter_performance' in aggregator


class TestParameterOptimizer:
    """Test cases for parameter optimization."""
    
    @pytest.fixture
    def parameter_optimizer(self):
        """Create ParameterOptimizer instance."""
        return ParameterOptimizer(
            optimization_method='grid_search',
            objective='sharpe_ratio'
        )
        
    @pytest.fixture
    def historical_performance(self):
        """Create historical performance data."""
        data = []
        
        # Generate performance data with different parameters
        for grid_spacing in [0.005, 0.01, 0.015, 0.02]:
            for num_levels in [5, 10, 15, 20]:
                # Simulate performance (better with medium values)
                base_performance = 1.0 - abs(grid_spacing - 0.01) * 10 - abs(num_levels - 10) * 0.01
                
                for i in range(10):  # 10 samples per parameter set
                    performance = base_performance + np.random.normal(0, 0.1)
                    data.append({
                        'grid_spacing': grid_spacing,
                        'num_levels': num_levels,
                        'sharpe_ratio': performance,
                        'total_return': performance * 0.1,
                        'max_drawdown': -0.05 - (1 - performance) * 0.1
                    })
                    
        return pd.DataFrame(data)
        
    def test_grid_search_optimization(self, parameter_optimizer, historical_performance):
        """Test grid search parameter optimization."""
        # Define parameter space
        param_space = {
            'grid_spacing': [0.005, 0.01, 0.015, 0.02],
            'num_levels': [5, 10, 15, 20]
        }
        
        # Run optimization
        optimal_params = parameter_optimizer.optimize(
            historical_performance,
            param_space,
            method='grid_search'
        )
        
        assert 'grid_spacing' in optimal_params
        assert 'num_levels' in optimal_params
        assert 'expected_performance' in optimal_params
        
        # Should find near-optimal values
        assert 0.008 <= optimal_params['grid_spacing'] <= 0.012
        assert 8 <= optimal_params['num_levels'] <= 12
        
    def test_bayesian_optimization(self, parameter_optimizer):
        """Test Bayesian optimization for parameters."""
        parameter_optimizer.optimization_method = 'bayesian'
        
        # Define continuous parameter space
        param_space = {
            'grid_spacing': (0.001, 0.05),
            'stop_loss': (0.01, 0.05),
            'take_profit': (0.02, 0.10)
        }
        
        # Mock objective function
        def objective_function(params):
            # Optimal around grid_spacing=0.01, stop_loss=0.02
            score = 1.0
            score -= abs(params['grid_spacing'] - 0.01) * 10
            score -= abs(params['stop_loss'] - 0.02) * 5
            score += params['take_profit'] * 0.5
            return score + np.random.normal(0, 0.05)
            
        # Run optimization
        optimal_params = parameter_optimizer.optimize_bayesian(
            objective_function,
            param_space,
            n_iterations=20
        )
        
        assert all(param in optimal_params for param in param_space)
        assert 0.005 <= optimal_params['grid_spacing'] <= 0.015
        
    def test_evolutionary_optimization(self, parameter_optimizer):
        """Test evolutionary algorithm optimization."""
        parameter_optimizer.optimization_method = 'evolutionary'
        
        # Define parameter genes
        param_genes = {
            'grid_spacing': {'min': 0.001, 'max': 0.05, 'type': 'float'},
            'num_levels': {'min': 5, 'max': 30, 'type': 'int'},
            'use_trailing_stop': {'values': [True, False], 'type': 'bool'}
        }
        
        # Fitness function
        def fitness_function(individual):
            score = 1.0
            score -= abs(individual['grid_spacing'] - 0.015) * 10
            score -= abs(individual['num_levels'] - 15) * 0.02
            if individual['use_trailing_stop']:
                score += 0.1
            return score
            
        # Run evolution
        optimal_params = parameter_optimizer.optimize_evolutionary(
            fitness_function,
            param_genes,
            population_size=50,
            generations=10
        )
        
        assert all(param in optimal_params for param in param_genes)
        assert isinstance(optimal_params['num_levels'], int)
        assert isinstance(optimal_params['use_trailing_stop'], bool)
        
    def test_online_parameter_update(self, parameter_optimizer):
        """Test online parameter updates with new feedback."""
        # Current parameters
        current_params = {
            'grid_spacing': 0.01,
            'stop_loss': 0.02,
            'position_size': 0.02
        }
        
        # New performance feedback
        recent_feedback = [
            {'parameters': current_params, 'performance': 0.8},
            {'parameters': current_params, 'performance': 0.6},
            {'parameters': current_params, 'performance': 0.7}
        ]
        
        # Update parameters based on feedback
        updated_params = parameter_optimizer.update_online(
            current_params,
            recent_feedback,
            learning_rate=0.1
        )
        
        # Should adjust parameters based on suboptimal performance
        assert updated_params != current_params
        assert 'grid_spacing' in updated_params
        
    def test_parameter_sensitivity_analysis(self, parameter_optimizer, historical_performance):
        """Test parameter sensitivity analysis."""
        # Analyze sensitivity
        sensitivity = parameter_optimizer.analyze_sensitivity(
            historical_performance,
            target_metric='sharpe_ratio'
        )
        
        assert 'grid_spacing' in sensitivity
        assert 'num_levels' in sensitivity
        
        # Each parameter should have sensitivity score
        for param, score in sensitivity.items():
            assert isinstance(score, float)
            assert score >= 0  # Non-negative sensitivity


class TestLearningEngine:
    """Test cases for the learning engine."""
    
    @pytest.fixture
    def learning_engine(self):
        """Create LearningEngine instance."""
        return LearningEngine(
            model_type='random_forest',
            learning_rate=0.01
        )
        
    @pytest.fixture
    def training_data(self):
        """Create training data for learning."""
        # Generate synthetic trading data
        n_samples = 1000
        
        features = pd.DataFrame({
            'volatility': np.random.uniform(0.01, 0.05, n_samples),
            'trend_strength': np.random.uniform(-1, 1, n_samples),
            'volume_ratio': np.random.uniform(0.5, 2, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples)
        })
        
        # Target: profitable trades when volatility is moderate and trend is strong
        target = []
        for i in range(n_samples):
            profit_prob = 0.5
            if 0.02 <= features.iloc[i]['volatility'] <= 0.03:
                profit_prob += 0.2
            if abs(features.iloc[i]['trend_strength']) > 0.5:
                profit_prob += 0.2
            if 9 <= features.iloc[i]['hour_of_day'] <= 16:
                profit_prob += 0.1
                
            target.append(1 if np.random.random() < profit_prob else 0)
            
        return features, np.array(target)
        
    def test_model_training(self, learning_engine, training_data):
        """Test model training process."""
        features, target = training_data
        
        # Split data
        train_size = int(0.8 * len(features))
        X_train, X_test = features[:train_size], features[train_size:]
        y_train, y_test = target[:train_size], target[train_size:]
        
        # Train model
        learning_engine.train(X_train, y_train)
        
        # Evaluate
        train_score = learning_engine.evaluate(X_train, y_train)
        test_score = learning_engine.evaluate(X_test, y_test)
        
        assert train_score > 0.6  # Should learn patterns
        assert test_score > 0.5   # Should generalize
        
    def test_feature_importance(self, learning_engine, training_data):
        """Test feature importance extraction."""
        features, target = training_data
        
        learning_engine.train(features, target)
        
        importance = learning_engine.get_feature_importance()
        
        assert len(importance) == len(features.columns)
        assert all(score >= 0 for score in importance.values())
        assert sum(importance.values()) > 0.99  # Should sum to ~1
        
        # Volatility and trend should be important
        assert importance['volatility'] > 0.15
        assert importance['trend_strength'] > 0.15
        
    def test_incremental_learning(self, learning_engine, training_data):
        """Test incremental/online learning."""
        features, target = training_data
        
        # Initial training
        initial_size = 100
        learning_engine.train(features[:initial_size], target[:initial_size])
        
        initial_score = learning_engine.evaluate(features[:initial_size], target[:initial_size])
        
        # Incremental updates
        batch_size = 20
        for i in range(initial_size, len(features), batch_size):
            batch_X = features[i:i+batch_size]
            batch_y = target[i:i+batch_size]
            
            learning_engine.update_incremental(batch_X, batch_y)
            
        # Performance should improve with more data
        final_score = learning_engine.evaluate(features, target)
        assert final_score >= initial_score
        
    def test_concept_drift_detection(self, learning_engine):
        """Test detection of concept drift."""
        # Create data with concept drift
        n_samples = 1000
        
        # First half: one pattern
        features1 = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples//2),
            'feature2': np.random.normal(0, 1, n_samples//2)
        })
        target1 = (features1['feature1'] > 0).astype(int)
        
        # Second half: different pattern
        features2 = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples//2),
            'feature2': np.random.normal(0, 1, n_samples//2)
        })
        target2 = (features2['feature2'] > 0).astype(int)  # Different feature matters
        
        # Combine
        features = pd.concat([features1, features2], ignore_index=True)
        target = np.concatenate([target1, target2])
        
        # Train on first concept
        learning_engine.train(features1, target1)
        
        # Detect drift
        drift_scores = []
        window_size = 50
        
        for i in range(n_samples//2, n_samples, window_size):
            window_features = features[i:i+window_size]
            window_target = target[i:i+window_size]
            
            drift_score = learning_engine.detect_drift(window_features, window_target)
            drift_scores.append(drift_score)
            
        # Should detect increasing drift
        assert drift_scores[-1] > drift_scores[0]
        assert any(score > 0.1 for score in drift_scores)  # Significant drift
        
    def test_ensemble_learning(self, learning_engine, training_data):
        """Test ensemble of multiple models."""
        features, target = training_data
        
        # Create ensemble
        models = [
            'random_forest',
            'gradient_boosting',
            'neural_network'
        ]
        
        ensemble = LearningEngine(model_type='ensemble', base_models=models)
        
        # Train ensemble
        ensemble.train(features, target)
        
        # Predictions should be weighted average
        predictions = ensemble.predict(features[:10])
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)
        
        # Ensemble should perform well
        score = ensemble.evaluate(features, target)
        assert score > 0.6


class TestAdaptiveController:
    """Test cases for adaptive control system."""
    
    @pytest.fixture
    def adaptive_controller(self):
        """Create AdaptiveController instance."""
        config = FeedbackConfig(
            adaptation_threshold=0.05,
            min_samples_required=20
        )
        return AdaptiveController(config)
        
    def test_adaptation_decision(self, adaptive_controller):
        """Test adaptation decision making."""
        # Current performance
        current_metrics = {
            'sharpe_ratio': 1.2,
            'win_rate': 0.55,
            'profit_factor': 1.8,
            'max_drawdown': -0.08
        }
        
        # Historical baseline
        baseline_metrics = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.60,
            'profit_factor': 2.0,
            'max_drawdown': -0.05
        }
        
        # Should trigger adaptation due to performance degradation
        decision = adaptive_controller.should_adapt(current_metrics, baseline_metrics)
        
        assert decision['should_adapt'] is True
        assert 'reasons' in decision
        assert len(decision['reasons']) > 0
        assert 'urgency' in decision
        
    def test_adaptation_strategy_selection(self, adaptive_controller):
        """Test selection of adaptation strategy."""
        # Performance context
        context = {
            'performance_trend': 'declining',
            'volatility_regime': 'high',
            'recent_losses': 5,
            'parameter_staleness': 30  # Days since last update
        }
        
        strategy = adaptive_controller.select_adaptation_strategy(context)
        
        assert isinstance(strategy, AdaptationStrategy)
        assert strategy.name in ['aggressive', 'conservative', 'incremental']
        assert 'actions' in strategy.details
        assert 'priority' in strategy.details
        
    def test_adaptation_execution(self, adaptive_controller):
        """Test execution of adaptation."""
        # Current system state
        current_state = {
            'parameters': {
                'position_size': 0.02,
                'stop_loss': 0.02,
                'take_profit': 0.05
            },
            'models': {
                'market_regime': 'model_v1',
                'signal_generation': 'model_v1'
            },
            'active_strategies': ['grid', 'momentum']
        }
        
        # Adaptation recommendations
        recommendations = {
            'parameter_updates': {
                'position_size': 0.015,  # Reduce risk
                'stop_loss': 0.025      # Wider stops
            },
            'model_updates': {
                'market_regime': 'model_v2'  # New model version
            },
            'strategy_changes': {
                'disable': ['momentum'],  # Disable in high volatility
                'enable': ['mean_reversion']
            }
        }
        
        # Execute adaptation
        new_state = adaptive_controller.execute_adaptation(
            current_state,
            recommendations
        )
        
        assert new_state['parameters']['position_size'] == 0.015
        assert new_state['models']['market_regime'] == 'model_v2'
        assert 'momentum' not in new_state['active_strategies']
        assert 'mean_reversion' in new_state['active_strategies']
        
    def test_gradual_adaptation(self, adaptive_controller):
        """Test gradual parameter adaptation."""
        # Current parameters
        current = {'grid_spacing': 0.01, 'position_size': 0.02}
        
        # Target parameters
        target = {'grid_spacing': 0.015, 'position_size': 0.015}
        
        # Gradual adaptation over multiple steps
        steps = 5
        adapted_params = []
        
        for step in range(steps):
            progress = (step + 1) / steps
            params = adaptive_controller.adapt_gradually(
                current, 
                target, 
                progress
            )
            adapted_params.append(params)
            
        # Should gradually move toward target
        assert adapted_params[0]['grid_spacing'] < adapted_params[-1]['grid_spacing']
        assert adapted_params[-1]['grid_spacing'] == target['grid_spacing']
        
        # Check smooth progression
        for i in range(1, len(adapted_params)):
            prev_spacing = adapted_params[i-1]['grid_spacing']
            curr_spacing = adapted_params[i]['grid_spacing']
            assert curr_spacing >= prev_spacing  # Monotonic increase
            
    def test_rollback_mechanism(self, adaptive_controller):
        """Test rollback on failed adaptation."""
        # State before adaptation
        original_state = {
            'parameters': {'position_size': 0.02},
            'performance': {'sharpe_ratio': 1.5}
        }
        
        # Adapted state
        adapted_state = {
            'parameters': {'position_size': 0.03},
            'performance': {'sharpe_ratio': 0.8}  # Worse performance
        }
        
        # Monitor period
        monitoring_results = {
            'performance_degraded': True,
            'metrics': {'sharpe_ratio': 0.8, 'max_drawdown': -0.15}
        }
        
        # Should trigger rollback
        action = adaptive_controller.evaluate_adaptation(
            original_state,
            adapted_state,
            monitoring_results
        )
        
        assert action['decision'] == 'rollback'
        assert 'reasons' in action
        assert action['restore_state'] == original_state


class TestFeedbackMetrics:
    """Test cases for feedback system metrics."""
    
    @pytest.fixture
    def feedback_metrics(self):
        """Create FeedbackMetrics instance."""
        return FeedbackMetrics()
        
    def test_learning_effectiveness(self, feedback_metrics):
        """Test measurement of learning effectiveness."""
        # Performance over time
        performance_history = [
            {'date': datetime.now() - timedelta(days=30), 'sharpe': 0.8, 'adaptations': 0},
            {'date': datetime.now() - timedelta(days=20), 'sharpe': 1.0, 'adaptations': 1},
            {'date': datetime.now() - timedelta(days=10), 'sharpe': 1.3, 'adaptations': 2},
            {'date': datetime.now(), 'sharpe': 1.5, 'adaptations': 3}
        ]
        
        effectiveness = feedback_metrics.calculate_learning_effectiveness(
            performance_history
        )
        
        assert 'improvement_rate' in effectiveness
        assert 'adaptation_success_rate' in effectiveness
        assert 'performance_trend' in effectiveness
        
        # Should show improvement
        assert effectiveness['improvement_rate'] > 0
        assert effectiveness['performance_trend'] == 'improving'
        
    def test_adaptation_impact(self, feedback_metrics):
        """Test measurement of adaptation impact."""
        # Adaptations with before/after metrics
        adaptations = [
            {
                'timestamp': datetime.now() - timedelta(days=20),
                'type': 'parameter_update',
                'before_metrics': {'sharpe': 0.8, 'win_rate': 0.50},
                'after_metrics': {'sharpe': 1.2, 'win_rate': 0.55}
            },
            {
                'timestamp': datetime.now() - timedelta(days=10),
                'type': 'model_update',
                'before_metrics': {'sharpe': 1.2, 'win_rate': 0.55},
                'after_metrics': {'sharpe': 1.0, 'win_rate': 0.52}
            }
        ]
        
        impact_analysis = feedback_metrics.analyze_adaptation_impact(adaptations)
        
        assert 'average_impact' in impact_analysis
        assert 'success_rate' in impact_analysis
        assert 'by_type' in impact_analysis
        
        # Parameter updates were more successful
        assert impact_analysis['by_type']['parameter_update']['success_rate'] > \
               impact_analysis['by_type']['model_update']['success_rate']
               
    def test_feedback_quality_metrics(self, feedback_metrics):
        """Test feedback quality assessment."""
        feedbacks = []
        
        # Generate feedbacks with varying quality
        for i in range(100):
            feedback = {
                'timestamp': datetime.now() - timedelta(hours=i),
                'completeness': 1.0 if i < 50 else 0.7,  # Older data less complete
                'accuracy': 0.95 if i < 30 else 0.85,
                'latency': i * 10  # Increasing latency
            }
            feedbacks.append(feedback)
            
        quality_metrics = feedback_metrics.assess_feedback_quality(feedbacks)
        
        assert 'average_completeness' in quality_metrics
        assert 'average_accuracy' in quality_metrics
        assert 'average_latency' in quality_metrics
        assert 'quality_score' in quality_metrics
        
        # Recent feedback should have better quality
        recent_quality = feedback_metrics.assess_feedback_quality(feedbacks[:30])
        assert recent_quality['quality_score'] > quality_metrics['quality_score']


class TestImprovementTracker:
    """Test cases for improvement tracking."""
    
    @pytest.fixture
    def improvement_tracker(self):
        """Create ImprovementTracker instance."""
        return ImprovementTracker()
        
    def test_improvement_recording(self, improvement_tracker):
        """Test recording system improvements."""
        # Record improvements
        improvements = [
            {
                'date': datetime.now() - timedelta(days=10),
                'type': 'parameter_optimization',
                'description': 'Optimized grid spacing',
                'impact': {'sharpe_ratio': +0.2, 'max_drawdown': -0.02}
            },
            {
                'date': datetime.now() - timedelta(days=5),
                'type': 'model_update',
                'description': 'Updated market regime model',
                'impact': {'win_rate': +0.05, 'profit_factor': +0.3}
            }
        ]
        
        for improvement in improvements:
            improvement_tracker.record_improvement(improvement)
            
        # Get improvement history
        history = improvement_tracker.get_improvement_history()
        
        assert len(history) == 2
        assert history[0]['type'] == 'parameter_optimization'
        
    def test_cumulative_improvement(self, improvement_tracker):
        """Test tracking cumulative improvements."""
        # Initial baseline
        baseline_metrics = {
            'sharpe_ratio': 0.8,
            'win_rate': 0.50,
            'profit_factor': 1.5,
            'max_drawdown': -0.10
        }
        
        improvement_tracker.set_baseline(baseline_metrics)
        
        # Record series of improvements
        current_metrics = baseline_metrics.copy()
        
        for i in range(5):
            # Simulate improvement
            current_metrics['sharpe_ratio'] += 0.1
            current_metrics['win_rate'] += 0.02
            
            improvement_tracker.update_metrics(current_metrics)
            
        cumulative = improvement_tracker.get_cumulative_improvement()
        
        assert cumulative['sharpe_ratio']['absolute'] == 0.5  # 0.1 * 5
        assert cumulative['sharpe_ratio']['percentage'] == 62.5  # 0.5/0.8
        assert cumulative['win_rate']['absolute'] == 0.10
        
    def test_improvement_velocity(self, improvement_tracker):
        """Test calculation of improvement velocity."""
        # Record improvements over time
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        for i, date in enumerate(dates):
            metrics = {
                'sharpe_ratio': 0.8 + i * 0.01,  # Linear improvement
                'win_rate': 0.50 + i * 0.005
            }
            improvement_tracker.record_metrics(date, metrics)
            
        velocity = improvement_tracker.calculate_improvement_velocity()
        
        assert 'sharpe_ratio' in velocity
        assert 'win_rate' in velocity
        
        # Should detect positive velocity
        assert velocity['sharpe_ratio']['daily'] > 0
        assert velocity['sharpe_ratio']['weekly'] > velocity['sharpe_ratio']['daily'] * 7
        
    def test_improvement_forecast(self, improvement_tracker):
        """Test forecasting future improvements."""
        # Historical improvement data
        history = []
        base_sharpe = 0.8
        
        for i in range(60):
            date = datetime.now() - timedelta(days=60-i)
            # Logarithmic improvement curve
            sharpe = base_sharpe + np.log(i + 1) * 0.1
            history.append({'date': date, 'sharpe_ratio': sharpe})
            
        improvement_tracker.load_history(history)
        
        # Forecast future improvements
        forecast = improvement_tracker.forecast_improvement(days=30)
        
        assert 'expected_metrics' in forecast
        assert 'confidence_interval' in forecast
        assert 'improvement_probability' in forecast
        
        # Should predict continued but slowing improvement
        current_sharpe = history[-1]['sharpe_ratio']
        forecasted_sharpe = forecast['expected_metrics']['sharpe_ratio']
        assert forecasted_sharpe > current_sharpe
        
        # Confidence interval should widen with time
        assert forecast['confidence_interval'][30] > forecast['confidence_interval'][7]


class TestFeedbackLoop:
    """Test cases for the main FeedbackLoop system."""
    
    @pytest.fixture
    def feedback_loop(self):
        """Create FeedbackLoop instance."""
        config = FeedbackConfig(
            learning_rate=0.01,
            adaptation_threshold=0.05,
            optimization_frequency='daily'
        )
        return FeedbackLoop(config)
        
    @pytest.fixture
    def mock_trading_system(self):
        """Create mock trading system."""
        system = Mock()
        system.get_parameters = Mock(return_value={
            'position_size': 0.02,
            'stop_loss': 0.02,
            'grid_spacing': 0.01
        })
        system.update_parameters = Mock()
        system.get_performance_metrics = Mock(return_value={
            'sharpe_ratio': 1.2,
            'win_rate': 0.55,
            'profit_factor': 1.8
        })
        return system
        
    @pytest.mark.asyncio
    async def test_feedback_collection(self, feedback_loop, mock_trading_system):
        """Test continuous feedback collection."""
        # Start feedback collection
        collection_task = asyncio.create_task(
            feedback_loop.start_collection(mock_trading_system)
        )
        
        # Simulate trades
        for i in range(10):
            trade_result = {
                'trade_id': f'trade_{i}',
                'outcome': 'win' if i % 3 != 0 else 'loss',
                'pnl': Decimal('100') if i % 3 != 0 else Decimal('-50'),
                'parameters_used': mock_trading_system.get_parameters()
            }
            
            await feedback_loop.process_trade(trade_result)
            await asyncio.sleep(0.01)
            
        # Check feedback collected
        feedback_count = feedback_loop.get_feedback_count()
        assert feedback_count == 10
        
        # Cancel collection
        collection_task.cancel()
        
    def test_optimization_cycle(self, feedback_loop, mock_trading_system):
        """Test complete optimization cycle."""
        # Add historical feedback
        for i in range(50):
            feedback = PerformanceFeedback(
                timestamp=datetime.now() - timedelta(hours=i),
                trade_id=f'hist_{i}',
                outcome='win' if i % 3 != 0 else 'loss',
                pnl=Decimal('100') if i % 3 != 0 else Decimal('-50'),
                parameters_used={'grid_spacing': 0.01 + (i % 5) * 0.002}
            )
            feedback_loop.add_feedback(feedback)
            
        # Run optimization
        optimization_result = feedback_loop.run_optimization_cycle(
            mock_trading_system
        )
        
        assert 'optimized_parameters' in optimization_result
        assert 'expected_improvement' in optimization_result
        assert 'confidence' in optimization_result
        
        # Should recommend parameter updates
        if optimization_result['expected_improvement'] > 0.05:
            mock_trading_system.update_parameters.assert_called()
            
    @pytest.mark.asyncio
    async def test_adaptive_learning(self, feedback_loop):
        """Test adaptive learning from feedback."""
        # Enable adaptive learning
        feedback_loop.enable_adaptive_learning()
        
        # Simulate changing market conditions
        market_phases = [
            {'volatility': 'low', 'trend': 'bullish', 'optimal_params': {'grid_spacing': 0.005}},
            {'volatility': 'high', 'trend': 'ranging', 'optimal_params': {'grid_spacing': 0.02}},
            {'volatility': 'medium', 'trend': 'bearish', 'optimal_params': {'grid_spacing': 0.01}}
        ]
        
        for phase in market_phases:
            # Generate feedback for this phase
            for i in range(20):
                # Performance based on parameter alignment
                current_spacing = 0.01
                optimal_spacing = phase['optimal_params']['grid_spacing']
                performance = 1.0 - abs(current_spacing - optimal_spacing) * 10
                
                feedback = PerformanceFeedback(
                    timestamp=datetime.now(),
                    trade_id=f'adaptive_{phase["volatility"]}_{i}',
                    outcome='win' if performance > 0.5 else 'loss',
                    pnl=Decimal(str(performance * 100)),
                    market_conditions=phase,
                    parameters_used={'grid_spacing': current_spacing}
                )
                
                await feedback_loop.process_feedback(feedback)
                
            # Check if system learned optimal parameters for this condition
            learned_params = feedback_loop.get_optimal_parameters(phase)
            assert abs(learned_params['grid_spacing'] - optimal_spacing) < 0.005
            
    def test_feedback_loop_monitoring(self, feedback_loop):
        """Test monitoring of feedback loop performance."""
        # Get monitoring metrics
        metrics = feedback_loop.get_monitoring_metrics()
        
        assert 'feedback_rate' in metrics
        assert 'learning_efficiency' in metrics
        assert 'adaptation_frequency' in metrics
        assert 'system_improvement' in metrics
        
        # Add some feedback and check metrics update
        for i in range(10):
            feedback = PerformanceFeedback(
                timestamp=datetime.now(),
                trade_id=f'monitor_{i}',
                outcome='win',
                pnl=Decimal('100')
            )
            feedback_loop.add_feedback(feedback)
            
        new_metrics = feedback_loop.get_monitoring_metrics()
        assert new_metrics['feedback_rate'] > metrics['feedback_rate']
        
    def test_feedback_persistence(self, feedback_loop, tmp_path):
        """Test saving and loading feedback data."""
        # Add feedback data
        for i in range(20):
            feedback = PerformanceFeedback(
                timestamp=datetime.now() - timedelta(hours=i),
                trade_id=f'persist_{i}',
                outcome='win' if i % 2 == 0 else 'loss',
                pnl=Decimal('50')
            )
            feedback_loop.add_feedback(feedback)
            
        # Save state
        save_path = tmp_path / "feedback_state.json"
        feedback_loop.save_state(save_path)
        
        # Create new instance and load
        new_loop = FeedbackLoop(feedback_loop.config)
        new_loop.load_state(save_path)
        
        # Verify data restored
        assert new_loop.get_feedback_count() == 20
        assert new_loop.get_learning_progress() == feedback_loop.get_learning_progress()


class TestFeedbackIntegration:
    """Integration tests for complete feedback system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_feedback_flow(self):
        """Test complete feedback flow from trade to adaptation."""
        # Create integrated system
        config = FeedbackConfig(
            learning_rate=0.1,  # Fast learning for test
            adaptation_threshold=0.03,
            min_samples_required=10
        )
        
        feedback_loop = FeedbackLoop(config)
        mock_system = Mock()
        
        # Initial parameters
        initial_params = {'position_size': 0.02, 'stop_loss': 0.02}
        mock_system.get_parameters = Mock(return_value=initial_params)
        mock_system.update_parameters = Mock()
        
        # Simulate poor performance
        for i in range(15):
            trade = {
                'trade_id': f'e2e_{i}',
                'outcome': 'loss' if i < 10 else 'win',  # Mostly losses
                'pnl': Decimal('-50') if i < 10 else Decimal('100'),
                'parameters_used': initial_params
            }
            
            await feedback_loop.process_trade(trade)
            
        # System should adapt
        await feedback_loop.check_and_adapt(mock_system)
        
        # Parameters should be updated
        mock_system.update_parameters.assert_called()
        updated_params = mock_system.update_parameters.call_args[0][0]
        
        # Risk parameters should be more conservative
        assert updated_params['position_size'] < initial_params['position_size']
        assert updated_params['stop_loss'] != initial_params['stop_loss']