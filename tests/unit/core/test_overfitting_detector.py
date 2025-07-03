"""
Unit tests for the Overfitting Detection System.

Tests cover:
- Statistical overfitting detection
- Walk-forward analysis
- Out-of-sample testing
- Cross-validation techniques
- Performance degradation monitoring
- Curve fitting detection
- Robustness testing
- Model complexity analysis
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import asyncio
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings

# GridAttention project imports
from core.overfitting_detector import (
    OverfittingDetector,
    OverfittingConfig,
    ValidationMethod,
    PerformanceValidator,
    StatisticalTest,
    RobustnessChecker,
    ComplexityAnalyzer,
    DegradationMonitor,
    OverfittingMetrics,
    ValidationResult
)


class TestOverfittingConfig:
    """Test cases for OverfittingConfig validation."""
    
    def test_default_config(self):
        """Test default overfitting detection configuration."""
        config = OverfittingConfig()
        
        assert config.lookback_ratio == 0.7  # 70% training, 30% test
        assert config.min_sample_size == 100
        assert config.confidence_level == 0.95
        assert config.walk_forward_windows == 5
        assert config.monte_carlo_simulations == 1000
        assert config.complexity_penalty == 0.01
        
    def test_custom_config(self):
        """Test custom overfitting configuration."""
        config = OverfittingConfig(
            lookback_ratio=0.8,
            validation_method='walk_forward',
            significance_threshold=0.01,
            robustness_checks=['monte_carlo', 'bootstrap']
        )
        
        assert config.lookback_ratio == 0.8
        assert config.validation_method == 'walk_forward'
        assert config.significance_threshold == 0.01
        assert 'monte_carlo' in config.robustness_checks
        
    def test_config_validation(self):
        """Test configuration validation rules."""
        # Invalid lookback ratio
        with pytest.raises(ValueError, match="lookback_ratio"):
            OverfittingConfig(lookback_ratio=1.5)
            
        # Invalid sample size
        with pytest.raises(ValueError, match="min_sample_size"):
            OverfittingConfig(min_sample_size=10)
            
        # Invalid confidence level
        with pytest.raises(ValueError, match="confidence_level"):
            OverfittingConfig(confidence_level=1.5)


class TestPerformanceValidator:
    """Test cases for performance validation."""
    
    @pytest.fixture
    def performance_validator(self):
        """Create PerformanceValidator instance."""
        config = OverfittingConfig(
            lookback_ratio=0.7,
            validation_method='time_series_split'
        )
        return PerformanceValidator(config)
        
    @pytest.fixture
    def sample_strategy_data(self):
        """Create sample strategy performance data."""
        # Generate realistic trading data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        
        # Create returns with some patterns
        base_returns = np.random.normal(0.0005, 0.02, 500)
        
        # Add some "lucky" periods that won't persist
        base_returns[100:150] += 0.01  # Exceptional period
        
        data = pd.DataFrame({
            'date': dates,
            'returns': base_returns,
            'trades': np.random.poisson(3, 500),  # Number of trades
            'win_rate': 0.5 + np.random.normal(0, 0.1, 500)
        })
        
        return data
        
    def test_train_test_split(self, performance_validator, sample_strategy_data):
        """Test train/test split for validation."""
        train_data, test_data = performance_validator.split_data(
            sample_strategy_data,
            method='simple',
            ratio=0.7
        )
        
        assert len(train_data) == int(len(sample_strategy_data) * 0.7)
        assert len(test_data) == len(sample_strategy_data) - len(train_data)
        
        # Ensure chronological order
        assert train_data['date'].max() <= test_data['date'].min()
        
    def test_walk_forward_analysis(self, performance_validator, sample_strategy_data):
        """Test walk-forward validation."""
        # Configure walk-forward
        n_splits = 5
        train_size = 250
        test_size = 50
        
        results = performance_validator.walk_forward_validate(
            sample_strategy_data,
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size
        )
        
        assert len(results) == n_splits
        
        for i, (train_metrics, test_metrics) in enumerate(results):
            assert 'sharpe_ratio' in train_metrics
            assert 'sharpe_ratio' in test_metrics
            assert 'total_return' in train_metrics
            assert 'total_return' in test_metrics
            
        # Check for performance degradation
        train_sharpes = [r[0]['sharpe_ratio'] for r in results]
        test_sharpes = [r[1]['sharpe_ratio'] for r in results]
        
        # Test performance should generally be worse (sign of overfitting)
        avg_train_sharpe = np.mean(train_sharpes)
        avg_test_sharpe = np.mean(test_sharpes)
        
        degradation = (avg_train_sharpe - avg_test_sharpe) / avg_train_sharpe
        assert degradation > 0  # Some degradation expected
        
    def test_cross_validation(self, performance_validator, sample_strategy_data):
        """Test time series cross-validation."""
        cv_results = performance_validator.cross_validate(
            sample_strategy_data,
            n_splits=5,
            gap=10  # Gap between train and test
        )
        
        assert 'train_scores' in cv_results
        assert 'test_scores' in cv_results
        assert 'score_variance' in cv_results
        
        # High variance between folds might indicate overfitting
        assert cv_results['score_variance'] < 0.5
        
        # Calculate overfitting metric
        overfitting_score = performance_validator.calculate_overfitting_score(
            cv_results['train_scores'],
            cv_results['test_scores']
        )
        
        assert 0 <= overfitting_score <= 1
        
    def test_out_of_sample_testing(self, performance_validator):
        """Test out-of-sample performance validation."""
        # In-sample data (used for optimization)
        in_sample_returns = np.random.normal(0.001, 0.02, 300)
        in_sample_returns[50:100] += 0.005  # Good period
        
        # Out-of-sample data (reality)
        out_sample_returns = np.random.normal(0.0005, 0.02, 100)  # Lower returns
        
        validation = performance_validator.validate_out_of_sample(
            in_sample_returns,
            out_sample_returns
        )
        
        assert 'performance_ratio' in validation
        assert 'degradation_pct' in validation
        assert 'is_significant' in validation
        assert 'confidence_interval' in validation
        
        # Should detect performance degradation
        assert validation['performance_ratio'] < 1.0
        assert validation['degradation_pct'] > 0
        
    def test_monte_carlo_validation(self, performance_validator, sample_strategy_data):
        """Test Monte Carlo simulation for robustness."""
        # Run Monte Carlo simulations
        mc_results = performance_validator.monte_carlo_validate(
            sample_strategy_data,
            n_simulations=100,
            confidence_level=0.95
        )
        
        assert 'simulated_returns' in mc_results
        assert 'confidence_interval' in mc_results
        assert 'probability_profitable' in mc_results
        assert 'expected_sharpe' in mc_results
        
        # Check distribution of results
        assert len(mc_results['simulated_returns']) == 100
        assert mc_results['confidence_interval'][0] < mc_results['confidence_interval'][1]


class TestStatisticalTest:
    """Test cases for statistical overfitting tests."""
    
    @pytest.fixture
    def statistical_test(self):
        """Create StatisticalTest instance."""
        return StatisticalTest(confidence_level=0.95)
        
    @pytest.fixture
    def strategy_results(self):
        """Create strategy backtest results."""
        # Generate results with some overfitting characteristics
        n_trades = 200
        
        # In-sample: artificially good
        in_sample_returns = np.concatenate([
            np.random.normal(0.002, 0.01, 100),  # Good performance
            np.random.normal(0.001, 0.015, 100)  # Moderate performance
        ])
        
        # Out-of-sample: reality
        out_sample_returns = np.random.normal(0.0001, 0.02, 50)  # Poor performance
        
        return {
            'in_sample': in_sample_returns,
            'out_of_sample': out_sample_returns,
            'n_parameters': 10,  # Number of optimized parameters
            'n_trades': n_trades
        }
        
    def test_sharpe_ratio_test(self, statistical_test, strategy_results):
        """Test Sharpe ratio statistical significance."""
        in_sample_sharpe = statistical_test.calculate_sharpe(
            strategy_results['in_sample']
        )
        
        # Test if Sharpe is statistically significant
        is_significant, p_value = statistical_test.test_sharpe_significance(
            strategy_results['in_sample'],
            min_sharpe=0.5
        )
        
        assert isinstance(is_significant, bool)
        assert 0 <= p_value <= 1
        
        # Test degradation
        out_sample_sharpe = statistical_test.calculate_sharpe(
            strategy_results['out_sample']
        )
        
        degradation_significant = statistical_test.test_performance_degradation(
            in_sample_sharpe,
            out_sample_sharpe,
            len(strategy_results['in_sample']),
            len(strategy_results['out_sample'])
        )
        
        assert degradation_significant['is_significant'] == True  # Significant degradation
        
    def test_white_reality_check(self, statistical_test):
        """Test White's Reality Check for data snooping."""
        # Multiple strategy results (from optimization)
        n_strategies = 50
        n_returns = 200
        
        # Generate returns for multiple strategies
        # Best strategy is just lucky
        strategy_returns = []
        for i in range(n_strategies):
            if i == 0:  # "Best" strategy
                returns = np.random.normal(0.001, 0.02, n_returns)
                returns[50:100] += 0.01  # Lucky period
            else:
                returns = np.random.normal(0, 0.02, n_returns)
            strategy_returns.append(returns)
            
        # Run White's Reality Check
        reality_check = statistical_test.whites_reality_check(
            strategy_returns,
            n_bootstrap=100
        )
        
        assert 'p_value' in reality_check
        assert 'is_significant' in reality_check
        assert 'best_strategy_return' in reality_check
        
        # Should detect that best strategy is likely due to chance
        assert reality_check['p_value'] > 0.05  # Not significant
        
    def test_hansen_spa_test(self, statistical_test):
        """Test Hansen's Superior Predictive Ability test."""
        # Benchmark returns
        benchmark_returns = np.random.normal(0.0005, 0.015, 200)
        
        # Strategy returns (slightly better by chance)
        strategy_returns = benchmark_returns + np.random.normal(0.0001, 0.005, 200)
        
        # Run SPA test
        spa_result = statistical_test.hansen_spa_test(
            strategy_returns,
            benchmark_returns,
            n_bootstrap=100
        )
        
        assert 'p_value' in spa_result
        assert 'consistent_p_value' in spa_result
        assert 'is_superior' in spa_result
        
        # Should not show significant superiority
        assert spa_result['p_value'] > 0.05
        
    def test_data_snooping_bias(self, statistical_test):
        """Test for data snooping bias detection."""
        # Simulate parameter optimization process
        n_parameters = 10
        n_combinations = 100  # Parameter combinations tested
        n_returns = 200
        
        # Best combination found by exhaustive search
        best_returns = None
        best_sharpe = -np.inf
        
        for _ in range(n_combinations):
            returns = np.random.normal(0.0001, 0.02, n_returns)
            sharpe = statistical_test.calculate_sharpe(returns)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_returns = returns
                
        # Test for data snooping
        snooping_test = statistical_test.test_data_snooping(
            best_returns,
            n_parameters=n_parameters,
            n_combinations=n_combinations
        )
        
        assert 'adjusted_sharpe' in snooping_test
        assert 'snooping_bias' in snooping_test
        assert 'deflated_p_value' in snooping_test
        
        # Adjusted Sharpe should be lower
        assert snooping_test['adjusted_sharpe'] < best_sharpe
        assert snooping_test['snooping_bias'] > 0


class TestRobustnessChecker:
    """Test cases for strategy robustness checking."""
    
    @pytest.fixture
    def robustness_checker(self):
        """Create RobustnessChecker instance."""
        return RobustnessChecker()
        
    @pytest.fixture
    def base_strategy_params(self):
        """Create base strategy parameters."""
        return {
            'lookback_period': 20,
            'entry_threshold': 0.02,
            'stop_loss': 0.015,
            'take_profit': 0.03,
            'position_size': 0.02
        }
        
    def test_parameter_sensitivity(self, robustness_checker, base_strategy_params):
        """Test parameter sensitivity analysis."""
        # Define parameter ranges for sensitivity test
        param_ranges = {
            'lookback_period': (10, 30, 5),  # min, max, step
            'entry_threshold': (0.01, 0.03, 0.005),
            'stop_loss': (0.01, 0.02, 0.0025)
        }
        
        # Mock strategy performance function
        def mock_strategy_performance(params):
            # Performance decreases as we move away from optimal
            base_performance = 1.5  # Base Sharpe ratio
            
            # Calculate degradation
            degradation = 0
            degradation += abs(params['lookback_period'] - 20) * 0.01
            degradation += abs(params['entry_threshold'] - 0.02) * 10
            degradation += abs(params['stop_loss'] - 0.015) * 20
            
            return base_performance - degradation + np.random.normal(0, 0.1)
            
        # Run sensitivity analysis
        sensitivity_results = robustness_checker.analyze_parameter_sensitivity(
            base_strategy_params,
            param_ranges,
            mock_strategy_performance
        )
        
        assert 'sensitivity_scores' in sensitivity_results
        assert 'stable_regions' in sensitivity_results
        assert 'fragile_parameters' in sensitivity_results
        
        # Stop loss should be most sensitive
        assert sensitivity_results['sensitivity_scores']['stop_loss'] > \
               sensitivity_results['sensitivity_scores']['lookback_period']
               
    def test_noise_robustness(self, robustness_checker):
        """Test robustness to noise in data."""
        # Original price data
        prices = 100 + np.cumsum(np.random.normal(0, 1, 1000))
        
        # Test strategy on original and noisy data
        noise_levels = [0.001, 0.005, 0.01, 0.02]  # 0.1% to 2% noise
        
        results = robustness_checker.test_noise_robustness(
            prices,
            noise_levels,
            strategy_function=lambda p: np.mean(np.diff(p) > 0)  # Simple strategy
        )
        
        assert 'performance_degradation' in results
        assert 'noise_tolerance' in results
        assert 'robustness_score' in results
        
        # Performance should degrade with more noise
        degradations = results['performance_degradation']
        assert degradations[0.001] < degradations[0.02]
        
    def test_market_regime_robustness(self, robustness_checker):
        """Test robustness across different market regimes."""
        # Create different market regimes
        regimes = {
            'trending_up': np.cumsum(np.random.normal(0.001, 0.01, 200)),
            'trending_down': np.cumsum(np.random.normal(-0.001, 0.01, 200)),
            'ranging': np.sin(np.linspace(0, 20*np.pi, 200)) * 0.1,
            'volatile': np.random.normal(0, 0.03, 200)
        }
        
        # Test strategy across regimes
        regime_results = robustness_checker.test_regime_robustness(
            regimes,
            strategy_params={'threshold': 0.01}
        )
        
        assert 'performance_by_regime' in regime_results
        assert 'consistency_score' in regime_results
        assert 'worst_regime' in regime_results
        assert 'regime_adaptability' in regime_results
        
        # Check consistency across regimes
        performances = list(regime_results['performance_by_regime'].values())
        performance_std = np.std(performances)
        
        # Lower std means more consistent
        assert regime_results['consistency_score'] == 1 / (1 + performance_std)
        
    def test_sample_size_robustness(self, robustness_checker):
        """Test performance stability with different sample sizes."""
        # Full dataset
        full_data = np.random.normal(0.0005, 0.02, 1000)
        
        # Test with different sample sizes
        sample_sizes = [50, 100, 200, 500, 1000]
        
        stability_results = robustness_checker.test_sample_size_stability(
            full_data,
            sample_sizes,
            n_bootstrap=50
        )
        
        assert 'mean_performance' in stability_results
        assert 'performance_variance' in stability_results
        assert 'min_stable_size' in stability_results
        assert 'convergence_rate' in stability_results
        
        # Variance should decrease with larger samples
        variances = stability_results['performance_variance']
        assert variances[1000] < variances[50]


class TestComplexityAnalyzer:
    """Test cases for model complexity analysis."""
    
    @pytest.fixture
    def complexity_analyzer(self):
        """Create ComplexityAnalyzer instance."""
        return ComplexityAnalyzer()
        
    def test_parameter_count_penalty(self, complexity_analyzer):
        """Test penalty based on parameter count."""
        # Model with different parameter counts
        models = [
            {'n_params': 5, 'sharpe': 1.5},
            {'n_params': 10, 'sharpe': 1.6},
            {'n_params': 20, 'sharpe': 1.7},
            {'n_params': 50, 'sharpe': 1.8}
        ]
        
        # Apply complexity penalty
        adjusted_performance = []
        for model in models:
            adjusted = complexity_analyzer.apply_complexity_penalty(
                model['sharpe'],
                model['n_params'],
                sample_size=200
            )
            adjusted_performance.append(adjusted)
            
        # Higher parameter models should have larger penalties
        assert adjusted_performance[0] > adjusted_performance[0] * 0.9  # Small penalty
        assert adjusted_performance[3] < models[3]['sharpe'] * 0.8  # Large penalty
        
        # Best model after adjustment might not be the most complex
        best_idx = np.argmax(adjusted_performance)
        assert best_idx < 3  # Not the most complex model
        
    def test_degrees_of_freedom_adjustment(self, complexity_analyzer):
        """Test degrees of freedom adjustment for statistics."""
        n_observations = 200
        n_parameters = 20
        
        # Calculate effective degrees of freedom
        dof = complexity_analyzer.calculate_degrees_of_freedom(
            n_observations,
            n_parameters,
            parameter_interactions=True
        )
        
        assert dof < n_observations
        assert dof < n_observations - n_parameters  # More conservative with interactions
        
        # Adjust statistics
        raw_sharpe = 1.5
        adjusted_sharpe = complexity_analyzer.adjust_sharpe_for_dof(
            raw_sharpe,
            n_observations,
            n_parameters
        )
        
        assert adjusted_sharpe < raw_sharpe
        
    def test_akaike_information_criterion(self, complexity_analyzer):
        """Test AIC for model selection."""
        # Different models with their log-likelihoods
        models = [
            {'name': 'simple', 'n_params': 3, 'log_likelihood': -100},
            {'name': 'medium', 'n_params': 7, 'log_likelihood': -95},
            {'name': 'complex', 'n_params': 15, 'log_likelihood': -92}
        ]
        
        # Calculate AIC for each model
        aic_scores = []
        for model in models:
            aic = complexity_analyzer.calculate_aic(
                model['log_likelihood'],
                model['n_params']
            )
            aic_scores.append(aic)
            
        # Lower AIC is better
        best_model_idx = np.argmin(aic_scores)
        
        # Should prefer simpler model unless complex is much better
        assert models[best_model_idx]['name'] != 'complex'
        
    def test_bayesian_information_criterion(self, complexity_analyzer):
        """Test BIC for model selection."""
        n_observations = 200
        
        models = [
            {'n_params': 3, 'log_likelihood': -100},
            {'n_params': 7, 'log_likelihood': -95},
            {'n_params': 15, 'log_likelihood': -92}
        ]
        
        # Calculate BIC
        bic_scores = []
        for model in models:
            bic = complexity_analyzer.calculate_bic(
                model['log_likelihood'],
                model['n_params'],
                n_observations
            )
            bic_scores.append(bic)
            
        # BIC penalizes complexity more than AIC
        best_model_idx = np.argmin(bic_scores)
        assert best_model_idx == 0  # Simplest model
        
    def test_minimum_description_length(self, complexity_analyzer):
        """Test MDL principle for complexity."""
        # Model complexity vs data fit trade-off
        data_points = 1000
        
        models = []
        for n_params in [5, 10, 20, 40]:
            # More parameters = better fit but higher complexity
            model_cost = complexity_analyzer.calculate_mdl(
                model_params=n_params,
                data_points=data_points,
                prediction_error=100 / n_params  # Better fit with more params
            )
            models.append({'params': n_params, 'mdl': model_cost})
            
        # Find optimal complexity
        optimal_idx = np.argmin([m['mdl'] for m in models])
        optimal_params = models[optimal_idx]['params']
        
        # Should not choose most complex model
        assert optimal_params < 40
        assert optimal_params > 5  # But not too simple


class TestDegradationMonitor:
    """Test cases for performance degradation monitoring."""
    
    @pytest.fixture
    def degradation_monitor(self):
        """Create DegradationMonitor instance."""
        return DegradationMonitor(
            window_size=50,
            alert_threshold=0.2  # 20% degradation
        )
        
    def test_performance_tracking(self, degradation_monitor):
        """Test continuous performance tracking."""
        # Simulate performance over time
        # Good performance initially, then degradation
        performance_data = []
        
        for i in range(200):
            if i < 100:
                # Good performance
                daily_return = np.random.normal(0.001, 0.02)
            else:
                # Degrading performance
                daily_return = np.random.normal(-0.0005, 0.025)
                
            performance_data.append({
                'date': datetime.now() - timedelta(days=200-i),
                'return': daily_return,
                'trades': np.random.poisson(3)
            })
            
        # Feed data to monitor
        for data_point in performance_data:
            degradation_monitor.update(data_point)
            
        # Check degradation detection
        degradation_detected = degradation_monitor.check_degradation()
        
        assert degradation_detected['is_degrading'] == True
        assert degradation_detected['degradation_start'] is not None
        assert degradation_detected['current_performance'] < degradation_detected['baseline_performance']
        
    def test_regime_change_detection(self, degradation_monitor):
        """Test detection of regime changes affecting performance."""
        # Create data with regime change
        regime1_returns = np.random.normal(0.001, 0.015, 100)
        regime2_returns = np.random.normal(-0.0002, 0.025, 100)
        
        all_returns = np.concatenate([regime1_returns, regime2_returns])
        
        # Process returns
        change_points = degradation_monitor.detect_regime_changes(all_returns)
        
        assert len(change_points) > 0
        assert 90 < change_points[0] < 110  # Should detect around position 100
        
    def test_early_warning_system(self, degradation_monitor):
        """Test early warning signals for degradation."""
        # Configure warning levels
        degradation_monitor.set_warning_levels({
            'yellow': 0.1,   # 10% degradation
            'orange': 0.15,  # 15% degradation  
            'red': 0.2      # 20% degradation
        })
        
        warnings = []
        degradation_monitor.set_warning_callback(lambda w: warnings.append(w))
        
        # Simulate gradual degradation
        baseline_sharpe = 1.5
        
        for i in range(100):
            # Gradually decreasing performance
            current_sharpe = baseline_sharpe * (1 - i * 0.003)
            
            degradation_monitor.update_metric('sharpe_ratio', current_sharpe)
            
        # Should have triggered warnings at different levels
        assert len(warnings) >= 3
        assert any(w['level'] == 'yellow' for w in warnings)
        assert any(w['level'] == 'red' for w in warnings)
        
    def test_recovery_detection(self, degradation_monitor):
        """Test detection of performance recovery."""
        # Simulate degradation and recovery
        performance_timeline = []
        
        # Good -> Bad -> Recovery
        for i in range(300):
            if i < 100:
                daily_return = np.random.normal(0.001, 0.02)
            elif i < 200:
                daily_return = np.random.normal(-0.001, 0.025)
            else:
                daily_return = np.random.normal(0.0008, 0.02)
                
            performance_timeline.append(daily_return)
            degradation_monitor.update({'return': daily_return})
            
        # Check recovery detection
        recovery_info = degradation_monitor.check_recovery()
        
        assert recovery_info['is_recovering'] == True
        assert recovery_info['recovery_start'] is not None
        assert recovery_info['recovery_strength'] > 0


class TestOverfittingDetector:
    """Test cases for the main OverfittingDetector class."""
    
    @pytest.fixture
    def overfitting_detector(self):
        """Create OverfittingDetector instance."""
        config = OverfittingConfig(
            lookback_ratio=0.7,
            validation_method='walk_forward',
            confidence_level=0.95
        )
        return OverfittingDetector(config)
        
    @pytest.fixture
    def strategy_backtest_results(self):
        """Create comprehensive backtest results."""
        # Generate realistic backtest data
        np.random.seed(42)
        
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # Create overfitted strategy results
        # In-sample: Jan 2020 - Jun 2023 (used for optimization)
        # Out-sample: Jul 2023 - Dec 2023 (reality check)
        
        in_sample_end = '2023-06-30'
        
        results = pd.DataFrame(index=dates)
        
        # Generate returns
        base_returns = np.random.normal(0.0001, 0.02, len(dates))
        
        # Add "patterns" in training data that won't persist
        in_sample_mask = dates <= in_sample_end
        base_returns[in_sample_mask & (dates.day == 15)] += 0.01  # Lucky day pattern
        
        results['returns'] = base_returns
        results['is_in_sample'] = in_sample_mask
        
        # Add trade information
        results['trades'] = np.random.poisson(2, len(dates))
        results['win_rate'] = 0.5 + np.random.normal(0, 0.1, len(dates))
        
        return results
        
    def test_comprehensive_overfitting_analysis(self, overfitting_detector, strategy_backtest_results):
        """Test complete overfitting analysis pipeline."""
        # Run comprehensive analysis
        analysis_results = overfitting_detector.analyze(
            strategy_backtest_results,
            n_parameters=15,  # Number of optimized parameters
            n_optimization_runs=1000  # Parameter combinations tested
        )
        
        assert 'overfitting_score' in analysis_results
        assert 'validation_results' in analysis_results
        assert 'statistical_tests' in analysis_results
        assert 'robustness_checks' in analysis_results
        assert 'complexity_analysis' in analysis_results
        assert 'recommendations' in analysis_results
        
        # Should detect overfitting
        assert analysis_results['overfitting_score'] > 0.5
        assert analysis_results['is_overfitted'] == True
        
    def test_real_time_monitoring(self, overfitting_detector):
        """Test real-time overfitting monitoring."""
        # Start with good in-sample performance
        in_sample_sharpe = 1.8
        
        overfitting_detector.initialize_monitoring(
            baseline_metrics={'sharpe_ratio': in_sample_sharpe}
        )
        
        # Simulate live trading with degraded performance
        live_returns = []
        for i in range(60):  # 60 days of live trading
            # Poor performance in reality
            daily_return = np.random.normal(-0.0002, 0.025)
            live_returns.append(daily_return)
            
            # Update monitor
            overfitting_detector.update_live_performance({
                'date': datetime.now() + timedelta(days=i),
                'return': daily_return
            })
            
        # Check if overfitting is detected
        live_analysis = overfitting_detector.analyze_live_performance()
        
        assert live_analysis['degradation_detected'] == True
        assert live_analysis['likely_overfitted'] == True
        assert 'confidence' in live_analysis
        assert 'recommendation' in live_analysis
        
    def test_strategy_comparison(self, overfitting_detector):
        """Test comparing multiple strategies for overfitting."""
        strategies = []
        
        # Create multiple strategies with different overfitting levels
        for i in range(5):
            n_params = 5 + i * 10  # Increasing complexity
            
            # More complex strategies have better in-sample performance
            in_sample_returns = np.random.normal(0.0005 + i*0.0001, 0.02, 500)
            out_sample_returns = np.random.normal(0.0001, 0.02 + i*0.002, 100)  # Worse out-sample
            
            strategies.append({
                'name': f'strategy_{i}',
                'n_parameters': n_params,
                'in_sample_returns': in_sample_returns,
                'out_sample_returns': out_sample_returns
            })
            
        # Compare strategies
        comparison = overfitting_detector.compare_strategies(strategies)
        
        assert 'rankings' in comparison
        assert 'overfitting_scores' in comparison
        assert 'recommended_strategy' in comparison
        
        # Should recommend simpler strategy
        recommended = comparison['recommended_strategy']
        assert recommended in ['strategy_0', 'strategy_1', 'strategy_2']
        
    def test_adaptive_complexity_control(self, overfitting_detector):
        """Test adaptive complexity control based on data."""
        # Available data
        data_sizes = [100, 500, 1000, 5000]
        
        recommendations = []
        for size in data_sizes:
            rec = overfitting_detector.recommend_complexity(
                data_size=size,
                feature_count=20
            )
            recommendations.append(rec)
            
        # Should recommend more parameters with more data
        assert recommendations[0]['max_parameters'] < recommendations[-1]['max_parameters']
        assert recommendations[0]['suggested_validation'] == 'leave_one_out'  # Small data
        assert recommendations[-1]['suggested_validation'] == 'walk_forward'  # Large data
        
    def test_reporting(self, overfitting_detector, strategy_backtest_results):
        """Test overfitting analysis report generation."""
        # Run analysis
        analysis = overfitting_detector.analyze(
            strategy_backtest_results,
            n_parameters=10
        )
        
        # Generate report
        report = overfitting_detector.generate_report(analysis)
        
        assert 'summary' in report
        assert 'detailed_results' in report
        assert 'visualizations' in report
        assert 'recommendations' in report
        
        # Check summary content
        summary = report['summary']
        assert 'overfitting_risk' in summary
        assert 'confidence_level' in summary
        assert 'key_findings' in summary
        
        # Check recommendations
        assert len(report['recommendations']) > 0
        assert any('reduce' in r.lower() for r in report['recommendations'])  # Suggest reducing complexity


class TestOverfittingMetrics:
    """Test cases for overfitting metrics calculation."""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create OverfittingMetrics instance."""
        return OverfittingMetrics()
        
    def test_deflated_sharpe_ratio(self, metrics_calculator):
        """Test deflated Sharpe ratio calculation."""
        # Strategy search parameters
        sharpe_ratio = 2.0
        n_strategies_tested = 1000
        n_independent_trials = 50
        
        deflated_sharpe = metrics_calculator.calculate_deflated_sharpe(
            sharpe_ratio,
            n_strategies_tested,
            n_independent_trials
        )
        
        assert deflated_sharpe < sharpe_ratio  # Should be lower
        assert deflated_sharpe > 0  # But still positive for good strategy
        
        # Test probability of backtest overfitting
        pbo = metrics_calculator.probability_of_backtest_overfitting(
            sharpe_ratio,
            n_strategies_tested
        )
        
        assert 0 <= pbo <= 1
        assert pbo > 0.3  # High probability with many trials
        
    def test_information_ratio_adjustment(self, metrics_calculator):
        """Test information ratio adjustment for multiple testing."""
        # Original information ratio
        ir = 1.5
        n_factors_tested = 50
        
        adjusted_ir = metrics_calculator.adjust_information_ratio(
            ir,
            n_factors_tested
        )
        
        assert adjusted_ir < ir
        
        # More factors = larger adjustment
        adjusted_ir_more = metrics_calculator.adjust_information_ratio(
            ir,
            n_factors_tested=200
        )
        
        assert adjusted_ir_more < adjusted_ir