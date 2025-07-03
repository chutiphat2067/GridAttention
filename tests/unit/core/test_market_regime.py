"""
Unit tests for the Market Regime Detection component.

Tests cover:
- Regime identification algorithms
- State transitions and validation
- Feature extraction for regime detection
- Regime persistence and stability
- Multi-timeframe analysis
- Regime change alerts
- Historical regime analysis
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from enum import Enum
import json

# GridAttention project imports
from core.market_regime_detector import (
    MarketRegimeDetector,
    RegimeState,
    RegimeConfig,
    RegimeTransition,
    RegimeMetrics,
    RegimeFeatures,
    RegimeHistory,
    RegimeClassifier
)


class TestRegimeState:
    """Test cases for RegimeState enum and properties."""
    
    def test_regime_states(self):
        """Test all regime states are defined."""
        expected_states = [
            'TRENDING_UP',
            'TRENDING_DOWN', 
            'RANGING',
            'VOLATILE',
            'BREAKOUT',
            'BREAKDOWN',
            'ACCUMULATION',
            'DISTRIBUTION'
        ]
        
        for state in expected_states:
            assert hasattr(RegimeState, state)
            
    def test_regime_state_properties(self):
        """Test regime state properties and characteristics."""
        # Trending states should have directional bias
        assert RegimeState.TRENDING_UP.has_directional_bias()
        assert RegimeState.TRENDING_DOWN.has_directional_bias()
        assert not RegimeState.RANGING.has_directional_bias()
        
        # Volatile states should be marked as high risk
        assert RegimeState.VOLATILE.is_high_risk()
        assert RegimeState.BREAKDOWN.is_high_risk()
        assert not RegimeState.RANGING.is_high_risk()
        
    def test_regime_state_transitions(self):
        """Test valid regime state transitions."""
        # Define valid transitions
        valid_transitions = {
            RegimeState.RANGING: [RegimeState.BREAKOUT, RegimeState.BREAKDOWN, RegimeState.TRENDING_UP, RegimeState.TRENDING_DOWN],
            RegimeState.TRENDING_UP: [RegimeState.DISTRIBUTION, RegimeState.VOLATILE, RegimeState.RANGING],
            RegimeState.TRENDING_DOWN: [RegimeState.ACCUMULATION, RegimeState.VOLATILE, RegimeState.RANGING],
            RegimeState.BREAKOUT: [RegimeState.TRENDING_UP, RegimeState.VOLATILE],
            RegimeState.BREAKDOWN: [RegimeState.TRENDING_DOWN, RegimeState.VOLATILE]
        }
        
        for state, transitions in valid_transitions.items():
            assert all(state.can_transition_to(next_state) for next_state in transitions)


class TestRegimeConfig:
    """Test cases for RegimeConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RegimeConfig()
        
        assert config.lookback_period == 20
        assert config.volatility_window == 14
        assert config.trend_strength_threshold == 0.6
        assert config.ranging_threshold == 0.3
        assert config.min_regime_duration == 5
        assert config.confidence_threshold == 0.7
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = RegimeConfig(
            lookback_period=50,
            volatility_window=20,
            trend_strength_threshold=0.7,
            use_volume_confirmation=True
        )
        
        assert config.lookback_period == 50
        assert config.volatility_window == 20
        assert config.trend_strength_threshold == 0.7
        assert config.use_volume_confirmation is True
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid lookback period
        with pytest.raises(ValueError):
            RegimeConfig(lookback_period=0)
            
        # Invalid threshold values
        with pytest.raises(ValueError):
            RegimeConfig(trend_strength_threshold=1.5)  # Must be between 0 and 1
            
        with pytest.raises(ValueError):
            RegimeConfig(confidence_threshold=-0.1)


class TestRegimeFeatures:
    """Test cases for regime feature extraction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        
        # Create different market scenarios
        trend_up = np.linspace(100, 120, 40) + np.random.randn(40) * 0.5
        ranging = 110 + np.sin(np.linspace(0, 4*np.pi, 30)) * 2 + np.random.randn(30) * 0.3
        trend_down = np.linspace(110, 95, 30) + np.random.randn(30) * 0.5
        
        prices = np.concatenate([trend_up, ranging, trend_down])
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100)) * 0.5,
            'low': prices - np.abs(np.random.randn(100)) * 0.5,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
    def test_trend_features(self, sample_data):
        """Test trend feature extraction."""
        features = RegimeFeatures.extract_trend_features(sample_data)
        
        assert 'trend_strength' in features
        assert 'trend_direction' in features
        assert 'trend_consistency' in features
        assert 'momentum' in features
        
        # Check value ranges
        assert -1 <= features['trend_strength'] <= 1
        assert features['trend_direction'] in [-1, 0, 1]
        assert 0 <= features['trend_consistency'] <= 1
        
    def test_volatility_features(self, sample_data):
        """Test volatility feature extraction."""
        features = RegimeFeatures.extract_volatility_features(sample_data)
        
        assert 'atr' in features
        assert 'volatility_ratio' in features
        assert 'volatility_regime' in features
        assert 'volatility_percentile' in features
        
        # Check values are positive
        assert features['atr'] > 0
        assert features['volatility_ratio'] > 0
        assert 0 <= features['volatility_percentile'] <= 100
        
    def test_volume_features(self, sample_data):
        """Test volume feature extraction."""
        features = RegimeFeatures.extract_volume_features(sample_data)
        
        assert 'volume_trend' in features
        assert 'volume_ratio' in features
        assert 'volume_volatility' in features
        assert 'price_volume_correlation' in features
        
        # Check correlation is bounded
        assert -1 <= features['price_volume_correlation'] <= 1
        
    def test_microstructure_features(self, sample_data):
        """Test market microstructure features."""
        features = RegimeFeatures.extract_microstructure_features(sample_data)
        
        assert 'bid_ask_spread' in features
        assert 'trade_intensity' in features
        assert 'price_efficiency' in features
        assert 'return_autocorrelation' in features
        
    def test_technical_features(self, sample_data):
        """Test technical indicator features."""
        features = RegimeFeatures.extract_technical_features(sample_data)
        
        assert 'rsi' in features
        assert 'macd_signal' in features
        assert 'bollinger_position' in features
        assert 'support_resistance_distance' in features
        
        # RSI should be bounded
        assert 0 <= features['rsi'] <= 100
        assert -1 <= features['bollinger_position'] <= 1


class TestRegimeClassifier:
    """Test cases for regime classification models."""
    
    @pytest.fixture
    def classifier(self):
        """Create a regime classifier instance."""
        config = RegimeConfig()
        return RegimeClassifier(config)
        
    @pytest.fixture
    def feature_vector(self):
        """Create sample feature vector."""
        return {
            'trend_strength': 0.8,
            'trend_direction': 1,
            'volatility_ratio': 1.2,
            'volume_trend': 0.6,
            'rsi': 65,
            'momentum': 0.7
        }
        
    def test_rule_based_classification(self, classifier, feature_vector):
        """Test rule-based regime classification."""
        regime, confidence = classifier.classify_regime_rules(feature_vector)
        
        assert isinstance(regime, RegimeState)
        assert 0 <= confidence <= 1
        
        # Strong uptrend features should classify as TRENDING_UP
        assert regime == RegimeState.TRENDING_UP
        assert confidence > 0.7
        
    def test_ml_classification(self, classifier, feature_vector):
        """Test ML-based regime classification."""
        # Convert features to tensor
        feature_tensor = torch.tensor(list(feature_vector.values()), dtype=torch.float32)
        
        regime, confidence = classifier.classify_regime_ml(feature_tensor)
        
        assert isinstance(regime, RegimeState)
        assert 0 <= confidence <= 1
        
    def test_ensemble_classification(self, classifier, feature_vector):
        """Test ensemble classification combining multiple methods."""
        regime, confidence, method_votes = classifier.classify_regime_ensemble(feature_vector)
        
        assert isinstance(regime, RegimeState)
        assert 0 <= confidence <= 1
        assert len(method_votes) >= 2  # At least 2 methods voting
        
    def test_classification_edge_cases(self, classifier):
        """Test classification with edge case features."""
        # Conflicting features
        conflicting_features = {
            'trend_strength': 0.8,
            'trend_direction': 1,
            'volatility_ratio': 3.0,  # Very high volatility
            'volume_trend': -0.8,     # Declining volume
            'rsi': 50,                # Neutral
            'momentum': 0.0           # No momentum
        }
        
        regime, confidence = classifier.classify_regime_rules(conflicting_features)
        
        # Should have lower confidence due to conflicts
        assert confidence < 0.6


class TestMarketRegimeDetector:
    """Test cases for the main MarketRegimeDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a MarketRegimeDetector instance."""
        config = RegimeConfig(
            lookback_period=30,
            min_regime_duration=5,
            confidence_threshold=0.6
        )
        return MarketRegimeDetector(config)
        
    @pytest.fixture
    def market_data(self):
        """Create comprehensive market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='5min')
        
        # Create a scenario with multiple regime changes
        segments = []
        
        # Ranging market
        ranging = 100 + np.sin(np.linspace(0, 6*np.pi, 50)) * 2 + np.random.randn(50) * 0.3
        segments.append(ranging)
        
        # Breakout to uptrend
        breakout = np.linspace(100, 110, 20) + np.random.randn(20) * 0.4
        segments.append(breakout)
        
        # Strong uptrend
        uptrend = np.linspace(110, 130, 50) + np.random.randn(50) * 0.5
        segments.append(uptrend)
        
        # Volatile top
        volatile = 130 + np.random.randn(30) * 2
        segments.append(volatile)
        
        # Downtrend
        downtrend = np.linspace(130, 115, 50) + np.random.randn(50) * 0.6
        segments.append(downtrend)
        
        prices = np.concatenate(segments)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(200) * 0.1,
            'high': prices + np.abs(np.random.randn(200)) * 0.5,
            'low': prices - np.abs(np.random.randn(200)) * 0.5,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        })
        
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.current_regime is None
        assert detector.regime_start_time is None
        assert len(detector.regime_history) == 0
        assert detector.config.lookback_period == 30
        
    def test_detect_regime(self, detector, market_data):
        """Test regime detection on market data."""
        # Detect regime for the ranging period
        ranging_data = market_data.iloc[:50]
        regime = detector.detect_regime(ranging_data)
        
        assert regime.state == RegimeState.RANGING
        assert regime.confidence > 0.6
        assert regime.features is not None
        
    def test_update_regime(self, detector, market_data):
        """Test regime updates with new data."""
        # Initialize with ranging data
        initial_data = market_data.iloc[:50]
        detector.update(initial_data)
        
        assert detector.current_regime.state == RegimeState.RANGING
        
        # Update with breakout data
        breakout_data = market_data.iloc[50:70]
        detector.update(breakout_data)
        
        # Should detect regime change
        assert detector.current_regime.state in [RegimeState.BREAKOUT, RegimeState.TRENDING_UP]
        
    def test_regime_transitions(self, detector, market_data):
        """Test regime transition detection and validation."""
        # Process data in chunks to simulate real-time updates
        chunk_size = 20
        transitions = []
        
        for i in range(0, len(market_data), chunk_size):
            chunk = market_data.iloc[max(0, i-30):i+chunk_size]
            previous_regime = detector.current_regime.state if detector.current_regime else None
            
            detector.update(chunk)
            
            if previous_regime and previous_regime != detector.current_regime.state:
                transition = RegimeTransition(
                    from_state=previous_regime,
                    to_state=detector.current_regime.state,
                    timestamp=chunk.iloc[-1]['timestamp'],
                    confidence=detector.current_regime.confidence
                )
                transitions.append(transition)
                
        # Should have detected multiple transitions
        assert len(transitions) >= 3
        
        # Verify transition validity
        for transition in transitions:
            assert transition.from_state.can_transition_to(transition.to_state)
            
    def test_regime_persistence(self, detector, market_data):
        """Test minimum regime duration enforcement."""
        detector.config.min_regime_duration = 10
        
        # Create data with brief regime change
        stable_data = market_data.iloc[:30]
        detector.update(stable_data)
        initial_regime = detector.current_regime.state
        
        # Brief spike that should be ignored
        spike_data = stable_data.copy()
        spike_data.loc[:, 'close'] *= 1.1  # 10% spike
        detector.update(spike_data.iloc[:5])  # Only 5 periods
        
        # Should maintain original regime due to minimum duration
        assert detector.current_regime.state == initial_regime
        
    def test_multi_timeframe_analysis(self, detector, market_data):
        """Test multi-timeframe regime analysis."""
        # Get regimes at different timeframes
        regimes = detector.analyze_multi_timeframe(
            market_data,
            timeframes=['5min', '15min', '1H']
        )
        
        assert len(regimes) == 3
        assert '5min' in regimes
        assert '15min' in regimes
        assert '1H' in regimes
        
        # Higher timeframes should be more stable
        assert regimes['1H'].confidence >= regimes['5min'].confidence
        
    def test_regime_strength_calculation(self, detector, market_data):
        """Test regime strength and stability metrics."""
        # Establish a strong trend
        trend_data = market_data.iloc[70:120]  # Strong uptrend portion
        detector.update(trend_data)
        
        strength = detector.calculate_regime_strength()
        
        assert 'strength_score' in strength
        assert 'stability_score' in strength
        assert 'persistence_score' in strength
        assert 'confidence_trend' in strength
        
        # Strong trend should have high scores
        assert strength['strength_score'] > 0.7
        assert strength['stability_score'] > 0.6
        
    def test_regime_history_tracking(self, detector, market_data):
        """Test regime history tracking and analysis."""
        # Process entire dataset
        chunk_size = 20
        for i in range(0, len(market_data), chunk_size):
            chunk = market_data.iloc[max(0, i-30):i+chunk_size]
            detector.update(chunk)
            
        history = detector.get_regime_history()
        
        assert len(history) > 0
        assert all(isinstance(r, RegimeHistory) for r in history)
        
        # Check history consistency
        for i in range(1, len(history)):
            assert history[i].start_time >= history[i-1].end_time
            
    def test_regime_statistics(self, detector, market_data):
        """Test regime statistics calculation."""
        # Process data to build history
        chunk_size = 20
        for i in range(0, len(market_data), chunk_size):
            chunk = market_data.iloc[max(0, i-30):i+chunk_size]
            detector.update(chunk)
            
        stats = detector.calculate_regime_statistics()
        
        assert 'regime_distribution' in stats
        assert 'average_duration' in stats
        assert 'transition_matrix' in stats
        assert 'regime_returns' in stats
        
        # Check distribution sums to 100%
        total_distribution = sum(stats['regime_distribution'].values())
        assert abs(total_distribution - 100.0) < 0.01
        
    def test_regime_forecast(self, detector, market_data):
        """Test regime forecasting capabilities."""
        # Build history
        training_data = market_data.iloc[:150]
        detector.update(training_data)
        
        # Forecast next regime
        forecast = detector.forecast_regime_change(horizon=10)
        
        assert 'probability_change' in forecast
        assert 'most_likely_regime' in forecast
        assert 'confidence' in forecast
        assert 'expected_duration' in forecast
        
        # Probabilities should be valid
        assert 0 <= forecast['probability_change'] <= 1
        assert 0 <= forecast['confidence'] <= 1
        
    def test_regime_alerts(self, detector, market_data):
        """Test regime change alerts."""
        alerts = []
        
        def alert_callback(alert):
            alerts.append(alert)
            
        detector.set_alert_callback(alert_callback)
        
        # Process data
        chunk_size = 20
        for i in range(0, len(market_data), chunk_size):
            chunk = market_data.iloc[max(0, i-30):i+chunk_size]
            detector.update(chunk)
            
        # Should have generated alerts
        assert len(alerts) > 0
        
        # Check alert structure
        for alert in alerts:
            assert 'timestamp' in alert
            assert 'previous_regime' in alert
            assert 'new_regime' in alert
            assert 'confidence' in alert
            assert 'message' in alert
            
    def test_regime_specific_parameters(self, detector):
        """Test regime-specific parameter recommendations."""
        # Set different regimes and get parameters
        regimes_to_test = [
            RegimeState.TRENDING_UP,
            RegimeState.RANGING,
            RegimeState.VOLATILE,
            RegimeState.TRENDING_DOWN
        ]
        
        for regime in regimes_to_test:
            params = detector.get_regime_parameters(regime)
            
            assert 'stop_loss_multiplier' in params
            assert 'take_profit_multiplier' in params
            assert 'position_size_factor' in params
            assert 'grid_spacing' in params
            assert 'order_frequency' in params
            
            # Volatile regime should have wider stops
            if regime == RegimeState.VOLATILE:
                assert params['stop_loss_multiplier'] > 1.5
                assert params['position_size_factor'] < 0.7
                
    def test_regime_confidence_decay(self, detector, market_data):
        """Test confidence decay over time without updates."""
        # Establish initial regime
        initial_data = market_data.iloc[:50]
        detector.update(initial_data)
        initial_confidence = detector.current_regime.confidence
        
        # Simulate time passing without updates
        detector.decay_confidence(periods=10)
        
        # Confidence should decrease
        assert detector.current_regime.confidence < initial_confidence
        assert detector.current_regime.confidence > 0
        
    def test_data_quality_handling(self, detector):
        """Test handling of poor quality data."""
        # Create data with gaps and outliers
        dates = pd.date_range(start='2023-01-01', periods=50, freq='5min')
        prices = 100 + np.random.randn(50) * 0.5
        
        # Add outliers
        prices[20] = 150  # Extreme spike
        prices[21] = 50   # Extreme drop
        
        poor_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        # Add missing data
        poor_data.loc[25:30, 'close'] = np.nan
        
        # Should handle gracefully
        with pytest.warns(UserWarning, match="Data quality issues"):
            regime = detector.detect_regime(poor_data)
            
        assert regime is not None
        assert regime.confidence < 0.5  # Low confidence due to poor data
        
    def test_real_time_performance(self, detector, market_data):
        """Test real-time performance requirements."""
        import time
        
        # Single update should be fast
        chunk = market_data.iloc[-30:]
        
        start_time = time.time()
        detector.update(chunk)
        elapsed = time.time() - start_time
        
        # Should complete within 10ms for real-time trading
        assert elapsed < 0.01
        
    def test_regime_serialization(self, detector, market_data, tmp_path):
        """Test saving and loading regime state."""
        # Build up state
        detector.update(market_data)
        
        # Save state
        save_path = tmp_path / "regime_state.json"
        detector.save_state(save_path)
        
        # Create new detector and load
        new_detector = MarketRegimeDetector(detector.config)
        new_detector.load_state(save_path)
        
        # Compare states
        assert new_detector.current_regime.state == detector.current_regime.state
        assert len(new_detector.regime_history) == len(detector.regime_history)
        
    def test_regime_comparison(self, detector):
        """Test regime comparison and similarity metrics."""
        regime1 = RegimeMetrics(
            state=RegimeState.TRENDING_UP,
            confidence=0.8,
            features={'trend_strength': 0.7, 'volatility': 0.3}
        )
        
        regime2 = RegimeMetrics(
            state=RegimeState.TRENDING_UP,
            confidence=0.75,
            features={'trend_strength': 0.65, 'volatility': 0.35}
        )
        
        regime3 = RegimeMetrics(
            state=RegimeState.VOLATILE,
            confidence=0.9,
            features={'trend_strength': 0.1, 'volatility': 0.9}
        )
        
        # Similar regimes
        similarity_1_2 = detector.calculate_regime_similarity(regime1, regime2)
        assert similarity_1_2 > 0.8
        
        # Different regimes
        similarity_1_3 = detector.calculate_regime_similarity(regime1, regime3)
        assert similarity_1_3 < 0.3


class TestRegimeMetrics:
    """Test regime metrics and analysis."""
    
    def test_metrics_calculation(self):
        """Test comprehensive metrics calculation."""
        metrics = RegimeMetrics()
        
        # Add sample regime periods
        regimes = [
            (RegimeState.TRENDING_UP, 20, 0.015),    # 20 periods, 1.5% return
            (RegimeState.RANGING, 30, 0.002),         # 30 periods, 0.2% return  
            (RegimeState.TRENDING_DOWN, 15, -0.012),  # 15 periods, -1.2% return
            (RegimeState.VOLATILE, 10, -0.005)        # 10 periods, -0.5% return
        ]
        
        for state, duration, returns in regimes:
            metrics.add_regime_period(state, duration, returns)
            
        summary = metrics.get_summary()
        
        assert 'total_periods' in summary
        assert 'regime_percentages' in summary
        assert 'average_returns_by_regime' in summary
        assert 'win_rate_by_regime' in summary
        assert 'sharpe_by_regime' in summary
        
        # Check calculations
        assert summary['total_periods'] == 75
        assert abs(sum(summary['regime_percentages'].values()) - 100.0) < 0.01
        
    def test_regime_performance_tracking(self):
        """Test tracking performance within regimes."""
        tracker = RegimeMetrics()
        
        # Simulate trades in different regimes
        trades = [
            (RegimeState.TRENDING_UP, 0.02, 1.5),     # 2% profit, 1.5 R
            (RegimeState.TRENDING_UP, 0.015, 1.2),    # 1.5% profit, 1.2 R
            (RegimeState.TRENDING_UP, -0.005, -0.5),  # 0.5% loss
            (RegimeState.VOLATILE, -0.01, -1.0),      # 1% loss
            (RegimeState.VOLATILE, 0.025, 2.0),       # 2.5% profit
            (RegimeState.RANGING, 0.005, 0.5),        # 0.5% profit
            (RegimeState.RANGING, 0.003, 0.3),        # 0.3% profit
        ]
        
        for regime, return_pct, r_multiple in trades:
            tracker.add_trade(regime, return_pct, r_multiple)
            
        performance = tracker.get_regime_performance()
        
        # Trending up should show best performance
        assert performance[RegimeState.TRENDING_UP]['win_rate'] > 0.6
        assert performance[RegimeState.TRENDING_UP]['avg_return'] > 0
        assert performance[RegimeState.TRENDING_UP]['avg_r_multiple'] > 0
        
        # Volatile should show mixed results
        assert performance[RegimeState.VOLATILE]['win_rate'] == 0.5
        
        
class TestRegimeIntegration:
    """Integration tests for regime detection system."""
    
    @pytest.fixture
    def full_system(self):
        """Create a full regime detection system."""
        config = RegimeConfig(
            lookback_period=30,
            use_ml_classifier=True,
            use_volume_confirmation=True,
            multi_timeframe=True
        )
        
        detector = MarketRegimeDetector(config)
        return detector
        
    def test_live_market_simulation(self, full_system):
        """Test with simulated live market conditions."""
        # Simulate 1 day of 5-minute bars
        timestamps = pd.date_range(start='2023-01-01 09:30', periods=78, freq='5min')
        
        # Create realistic intraday pattern
        morning_trend = np.linspace(100, 102, 20) + np.random.randn(20) * 0.1
        midday_range = 102 + np.sin(np.linspace(0, 2*np.pi, 20)) * 0.5 + np.random.randn(20) * 0.1
        afternoon_trend = np.linspace(102, 104, 20) + np.random.randn(20) * 0.15
        close_volatility = 104 + np.random.randn(18) * 0.3
        
        prices = np.concatenate([morning_trend, midday_range, afternoon_trend, close_volatility])
        
        # Track regime changes throughout the day
        regime_changes = []
        
        for i in range(30, len(timestamps)):
            data = pd.DataFrame({
                'timestamp': timestamps[:i+1],
                'open': prices[:i+1] + np.random.randn(i+1) * 0.05,
                'high': prices[:i+1] + np.abs(np.random.randn(i+1)) * 0.2,
                'low': prices[:i+1] - np.abs(np.random.randn(i+1)) * 0.2,
                'close': prices[:i+1],
                'volume': np.random.randint(5000, 20000, i+1)
            })
            
            prev_regime = full_system.current_regime.state if full_system.current_regime else None
            full_system.update(data)
            
            if prev_regime != full_system.current_regime.state:
                regime_changes.append({
                    'time': timestamps[i],
                    'from': prev_regime,
                    'to': full_system.current_regime.state,
                    'confidence': full_system.current_regime.confidence
                })
                
        # Should detect multiple intraday regime changes
        assert len(regime_changes) >= 2
        assert len(regime_changes) <= 6  # Not too many (over-sensitive)
        
    def test_stress_conditions(self, full_system):
        """Test under stress market conditions."""
        # Create flash crash scenario
        timestamps = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        
        # Normal market
        normal = 100 + np.random.randn(40) * 0.2
        
        # Flash crash
        crash = np.linspace(100, 92, 10)  # 8% drop in 10 minutes
        
        # Recovery
        recovery = np.linspace(92, 97, 20)
        
        # Volatile aftermath  
        aftermath = 97 + np.random.randn(30) * 1.0
        
        prices = np.concatenate([normal, crash, recovery, aftermath])
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100)) * 0.3,
            'low': prices - np.abs(np.random.randn(100)) * 0.3,
            'close': prices,
            'volume': np.concatenate([
                np.random.randint(5000, 10000, 40),
                np.random.randint(20000, 50000, 30),  # High volume during crash
                np.random.randint(10000, 20000, 30)
            ])
        })
        
        # Process in real-time
        alerts = []
        full_system.set_alert_callback(lambda a: alerts.append(a))
        
        for i in range(30, len(data)):
            full_system.update(data.iloc[:i+1])
            
        # Should detect breakdown and volatile regimes
        regimes_detected = [a['new_regime'] for a in alerts]
        assert RegimeState.BREAKDOWN in regimes_detected or RegimeState.VOLATILE in regimes_detected
        
        # Final regime should reflect volatility
        assert full_system.current_regime.state in [RegimeState.VOLATILE, RegimeState.RECOVERY]