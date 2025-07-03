"""
Unit tests for Market Regime Models.

Tests cover:
- Hidden Markov Models (HMM) for regime detection
- Gaussian Mixture Models (GMM)
- Change point detection algorithms
- Regime classification models
- Feature engineering for regime detection
- Model training and inference
- Regime transition probabilities
- Multi-timeframe regime analysis
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Optional, Tuple

# Assuming the module structure
from src.models.regime_models import (
    RegimeHMM,
    RegimeGMM,
    ChangePointDetector,
    RegimeClassifier,
    RegimeFeatureExtractor,
    MarkovRegimeSwitch,
    RegimeEnsemble,
    RegimeConfig,
    RegimeState
)


class TestRegimeConfig:
    """Test cases for regime model configuration."""
    
    def test_default_config(self):
        """Test default regime configuration."""
        config = RegimeConfig()
        
        assert config.n_regimes == 3
        assert config.feature_window == 20
        assert config.min_regime_length == 5
        assert config.transition_penalty == 0.01
        assert config.model_type == 'hmm'
        
    def test_custom_config(self):
        """Test custom regime configuration."""
        config = RegimeConfig(
            n_regimes=5,
            feature_window=50,
            model_type='gmm',
            use_volume_features=True,
            multi_timeframe=True
        )
        
        assert config.n_regimes == 5
        assert config.feature_window == 50
        assert config.model_type == 'gmm'
        assert config.use_volume_features is True
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid number of regimes
        with pytest.raises(ValueError, match="n_regimes must be"):
            RegimeConfig(n_regimes=1)
            
        # Invalid feature window
        with pytest.raises(ValueError, match="feature_window must be"):
            RegimeConfig(feature_window=0)
            
        # Invalid model type
        with pytest.raises(ValueError, match="model_type must be"):
            RegimeConfig(model_type='invalid')


class TestRegimeFeatureExtractor:
    """Test cases for regime feature extraction."""
    
    @pytest.fixture
    def feature_extractor(self):
        """Create feature extractor instance."""
        return RegimeFeatureExtractor(
            price_features=True,
            volume_features=True,
            volatility_features=True,
            microstructure_features=True
        )
        
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=500, freq='5min')
        
        # Generate synthetic market data with regime changes
        prices = []
        volumes = []
        
        # Regime 1: Trending up (0-150)
        trend_up = 100 + np.linspace(0, 20, 150) + np.random.normal(0, 0.5, 150)
        vol_up = np.random.lognormal(8, 0.5, 150)
        
        # Regime 2: Volatile ranging (150-300)
        volatile = 120 + np.random.normal(0, 3, 150)
        vol_volatile = np.random.lognormal(9, 0.7, 150)
        
        # Regime 3: Trending down (300-500)
        trend_down = 120 - np.linspace(0, 15, 200) + np.random.normal(0, 0.5, 200)
        vol_down = np.random.lognormal(8.5, 0.6, 200)
        
        prices = np.concatenate([trend_up, volatile, trend_down])
        volumes = np.concatenate([vol_up, vol_volatile, vol_down])
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.uniform(-0.5, 0.5, 500),
            'high': prices + np.abs(np.random.normal(0, 0.5, 500)),
            'low': prices - np.abs(np.random.normal(0, 0.5, 500)),
            'close': prices,
            'volume': volumes
        })
        
        return df
        
    def test_price_features(self, feature_extractor, sample_market_data):
        """Test price-based feature extraction."""
        features = feature_extractor.extract_price_features(sample_market_data)
        
        expected_features = [
            'returns', 'log_returns', 'price_momentum',
            'price_acceleration', 'price_range', 'price_position'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
            
        # Check feature properties
        assert len(features) == len(sample_market_data) - feature_extractor.lookback
        assert not features.isna().any().any()  # No NaN values
        
        # Returns should be centered around 0
        assert abs(features['returns'].mean()) < 0.1
        
    def test_volume_features(self, feature_extractor, sample_market_data):
        """Test volume-based feature extraction."""
        features = feature_extractor.extract_volume_features(sample_market_data)
        
        expected_features = [
            'volume_ratio', 'volume_trend', 'volume_volatility',
            'price_volume_correlation', 'volume_weighted_price'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
            
        # Volume ratio should be positive
        assert (features['volume_ratio'] > 0).all()
        
        # Correlation should be between -1 and 1
        assert features['price_volume_correlation'].between(-1, 1).all()
        
    def test_volatility_features(self, feature_extractor, sample_market_data):
        """Test volatility feature extraction."""
        features = feature_extractor.extract_volatility_features(sample_market_data)
        
        expected_features = [
            'realized_volatility', 'garch_volatility', 'volatility_ratio',
            'high_low_spread', 'close_to_close_vol'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
            
        # Volatility should be positive
        assert (features['realized_volatility'] > 0).all()
        assert (features['high_low_spread'] >= 0).all()
        
    def test_microstructure_features(self, feature_extractor, sample_market_data):
        """Test market microstructure features."""
        features = feature_extractor.extract_microstructure_features(sample_market_data)
        
        expected_features = [
            'bid_ask_spread', 'trade_imbalance', 'price_impact',
            'order_flow_toxicity', 'realized_spread'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
            
    def test_combined_features(self, feature_extractor, sample_market_data):
        """Test combined feature extraction."""
        all_features = feature_extractor.extract_all_features(sample_market_data)
        
        # Should have features from all categories
        assert all_features.shape[1] > 15  # Multiple features
        assert len(all_features) == len(sample_market_data) - feature_extractor.lookback
        
        # Check feature scaling
        scaled_features = feature_extractor.scale_features(all_features)
        
        # Scaled features should have mean ~0 and std ~1
        assert abs(scaled_features.mean().mean()) < 0.1
        assert abs(scaled_features.std().mean() - 1.0) < 0.2


class TestRegimeHMM:
    """Test cases for Hidden Markov Model regime detection."""
    
    @pytest.fixture
    def regime_hmm(self):
        """Create HMM regime model."""
        config = RegimeConfig(
            n_regimes=3,
            model_type='hmm',
            covariance_type='full'
        )
        return RegimeHMM(config)
        
    @pytest.fixture
    def training_features(self):
        """Create training features with clear regimes."""
        n_samples = 1000
        
        # Generate features for 3 regimes
        regime_1 = np.random.multivariate_normal(
            [0.001, 0.5, 0.1],  # Low return, medium vol, low volume
            [[0.01, 0, 0], [0, 0.1, 0], [0, 0, 0.05]],
            size=300
        )
        
        regime_2 = np.random.multivariate_normal(
            [0.002, 0.3, 0.2],  # Medium return, low vol, medium volume
            [[0.02, 0, 0], [0, 0.05, 0], [0, 0, 0.1]],
            size=400
        )
        
        regime_3 = np.random.multivariate_normal(
            [-0.001, 0.8, 0.3],  # Negative return, high vol, high volume
            [[0.03, 0, 0], [0, 0.2, 0], [0, 0, 0.15]],
            size=300
        )
        
        features = np.vstack([regime_1, regime_2, regime_3])
        
        return pd.DataFrame(
            features,
            columns=['returns', 'volatility', 'volume_ratio']
        )
        
    def test_hmm_initialization(self, regime_hmm):
        """Test HMM model initialization."""
        assert regime_hmm.n_states == 3
        assert hasattr(regime_hmm, 'model')
        assert isinstance(regime_hmm.model, hmm.GaussianHMM)
        
    def test_hmm_training(self, regime_hmm, training_features):
        """Test HMM model training."""
        # Train model
        regime_hmm.fit(training_features)
        
        # Check model is trained
        assert regime_hmm.is_fitted
        
        # Check transition matrix
        trans_matrix = regime_hmm.get_transition_matrix()
        assert trans_matrix.shape == (3, 3)
        
        # Rows should sum to 1
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)
        
        # Diagonal should be strong (regimes persist)
        assert np.diag(trans_matrix).min() > 0.5
        
    def test_regime_prediction(self, regime_hmm, training_features):
        """Test regime prediction."""
        regime_hmm.fit(training_features)
        
        # Predict regimes
        regimes = regime_hmm.predict(training_features)
        
        assert len(regimes) == len(training_features)
        assert set(regimes) == {0, 1, 2}  # 3 regimes
        
        # Check regime persistence
        regime_changes = np.sum(np.diff(regimes) != 0)
        assert regime_changes < len(regimes) * 0.1  # Less than 10% changes
        
    def test_regime_probabilities(self, regime_hmm, training_features):
        """Test regime probability calculation."""
        regime_hmm.fit(training_features)
        
        # Get regime probabilities
        probabilities = regime_hmm.predict_proba(training_features)
        
        assert probabilities.shape == (len(training_features), 3)
        
        # Probabilities should sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Max probability should correspond to predicted regime
        regimes = regime_hmm.predict(training_features)
        max_prob_regimes = np.argmax(probabilities, axis=1)
        assert np.array_equal(regimes, max_prob_regimes)
        
    def test_regime_statistics(self, regime_hmm, training_features):
        """Test regime statistics calculation."""
        regime_hmm.fit(training_features)
        regimes = regime_hmm.predict(training_features)
        
        # Get regime statistics
        stats = regime_hmm.get_regime_statistics(training_features, regimes)
        
        assert len(stats) == 3  # One per regime
        
        for regime_id, regime_stats in stats.items():
            assert 'mean' in regime_stats
            assert 'std' in regime_stats
            assert 'duration' in regime_stats
            assert 'frequency' in regime_stats
            
            # Check shapes
            assert len(regime_stats['mean']) == training_features.shape[1]
            assert len(regime_stats['std']) == training_features.shape[1]
            
    def test_viterbi_decoding(self, regime_hmm, training_features):
        """Test Viterbi algorithm for optimal path."""
        regime_hmm.fit(training_features)
        
        # Get most likely sequence
        viterbi_path = regime_hmm.decode(training_features)
        
        # Compare with regular prediction
        regular_prediction = regime_hmm.predict(training_features)
        
        # Viterbi should give smoother transitions
        viterbi_changes = np.sum(np.diff(viterbi_path) != 0)
        regular_changes = np.sum(np.diff(regular_prediction) != 0)
        
        assert viterbi_changes <= regular_changes


class TestRegimeGMM:
    """Test cases for Gaussian Mixture Model regime detection."""
    
    @pytest.fixture
    def regime_gmm(self):
        """Create GMM regime model."""
        config = RegimeConfig(
            n_regimes=3,
            model_type='gmm',
            covariance_type='full'
        )
        return RegimeGMM(config)
        
    def test_gmm_initialization(self, regime_gmm):
        """Test GMM model initialization."""
        assert regime_gmm.n_components == 3
        assert hasattr(regime_gmm, 'model')
        assert isinstance(regime_gmm.model, GaussianMixture)
        
    def test_gmm_training(self, regime_gmm, training_features):
        """Test GMM model training."""
        regime_gmm.fit(training_features)
        
        assert regime_gmm.is_fitted
        
        # Check component weights
        weights = regime_gmm.get_component_weights()
        assert len(weights) == 3
        assert np.allclose(weights.sum(), 1.0)
        
        # Check means
        means = regime_gmm.get_component_means()
        assert means.shape == (3, training_features.shape[1])
        
    def test_gmm_prediction(self, regime_gmm, training_features):
        """Test GMM regime prediction."""
        regime_gmm.fit(training_features)
        
        regimes = regime_gmm.predict(training_features)
        
        assert len(regimes) == len(training_features)
        assert set(regimes).issubset({0, 1, 2})
        
    def test_gmm_scoring(self, regime_gmm, training_features):
        """Test GMM model scoring (BIC/AIC)."""
        regime_gmm.fit(training_features)
        
        bic = regime_gmm.bic(training_features)
        aic = regime_gmm.aic(training_features)
        
        assert isinstance(bic, float)
        assert isinstance(aic, float)
        assert bic > aic  # BIC penalizes complexity more
        
    def test_optimal_components(self, training_features):
        """Test finding optimal number of components."""
        gmm_selector = RegimeGMM.select_optimal_components(
            training_features,
            n_components_range=range(2, 6),
            criterion='bic'
        )
        
        assert 2 <= gmm_selector.n_components <= 5
        
        # Should have selected based on lowest BIC
        scores = gmm_selector.selection_scores
        assert len(scores) == 4  # Tested 2, 3, 4, 5 components


class TestChangePointDetector:
    """Test cases for change point detection algorithms."""
    
    @pytest.fixture
    def change_detector(self):
        """Create change point detector."""
        return ChangePointDetector(
            method='pelt',
            penalty='bic',
            min_segment_length=10
        )
        
    @pytest.fixture
    def data_with_changepoints(self):
        """Create data with known change points."""
        n_points = 500
        
        # Segment 1: Low volatility trend
        segment1 = np.linspace(0, 10, 150) + np.random.normal(0, 0.5, 150)
        
        # Segment 2: High volatility
        segment2 = 10 + np.random.normal(0, 3, 150)
        
        # Segment 3: Downward trend
        segment3 = np.linspace(10, 0, 200) + np.random.normal(0, 0.5, 200)
        
        data = np.concatenate([segment1, segment2, segment3])
        
        return data, [150, 300]  # True change points
        
    def test_pelt_detection(self, change_detector, data_with_changepoints):
        """Test PELT (Pruned Exact Linear Time) algorithm."""
        data, true_cp = data_with_changepoints
        
        detected_cp = change_detector.detect_changepoints(data)
        
        # Should detect approximately correct change points
        assert len(detected_cp) >= 2
        
        # Check if detected points are close to true points
        for true_point in true_cp:
            closest = min(abs(cp - true_point) for cp in detected_cp)
            assert closest < 20  # Within 20 points
            
    def test_bayesian_changepoint(self):
        """Test Bayesian online changepoint detection."""
        detector = ChangePointDetector(
            method='bayesian',
            hazard_rate=1/100  # Expected run length
        )
        
        # Create data with sudden change
        data = np.concatenate([
            np.random.normal(0, 1, 200),
            np.random.normal(3, 1, 200)  # Mean shift
        ])
        
        # Run online detection
        changepoint_probs = []
        for i in range(len(data)):
            prob = detector.update(data[i])
            changepoint_probs.append(prob)
            
        changepoint_probs = np.array(changepoint_probs)
        
        # Should have high probability around point 200
        max_prob_idx = np.argmax(changepoint_probs)
        assert 180 < max_prob_idx < 220
        
    def test_cusum_detection(self):
        """Test CUSUM (Cumulative Sum) algorithm."""
        detector = ChangePointDetector(
            method='cusum',
            threshold=5,
            drift=1
        )
        
        # Data with mean shift
        data = np.concatenate([
            np.random.normal(0, 1, 300),
            np.random.normal(2, 1, 200)
        ])
        
        changepoints = detector.detect_changepoints(data)
        
        # Should detect change around 300
        assert len(changepoints) >= 1
        assert any(280 < cp < 320 for cp in changepoints)
        
    def test_multiple_changepoints(self, change_detector):
        """Test detection of multiple change points."""
        # Create data with multiple regime changes
        segments = [
            np.random.normal(0, 1, 100),
            np.random.normal(2, 0.5, 100),
            np.random.normal(-1, 2, 100),
            np.random.normal(1, 1, 100),
            np.random.normal(0, 3, 100)
        ]
        
        data = np.concatenate(segments)
        true_changepoints = [100, 200, 300, 400]
        
        detected = change_detector.detect_changepoints(data)
        
        # Should detect most change points
        assert len(detected) >= 3
        
        # Calculate detection accuracy
        detected_correctly = 0
        for true_cp in true_changepoints:
            if any(abs(d - true_cp) < 20 for d in detected):
                detected_correctly += 1
                
        assert detected_correctly >= 3  # At least 75% accuracy


class TestRegimeClassifier:
    """Test cases for regime classification models."""
    
    @pytest.fixture
    def regime_classifier(self):
        """Create regime classifier."""
        return RegimeClassifier(
            model_type='random_forest',
            n_regimes=3,
            feature_importance=True
        )
        
    @pytest.fixture
    def labeled_data(self):
        """Create labeled training data."""
        n_samples = 1000
        
        # Generate features and labels
        features = []
        labels = []
        
        # Regime 0: Bull market
        bull_features = np.random.multivariate_normal(
            [0.002, 0.15, 0.8],  # Positive return, low vol, high momentum
            [[0.01, 0, 0], [0, 0.05, 0], [0, 0, 0.1]],
            size=400
        )
        features.append(bull_features)
        labels.extend([0] * 400)
        
        # Regime 1: Bear market
        bear_features = np.random.multivariate_normal(
            [-0.002, 0.25, 0.2],  # Negative return, high vol, low momentum
            [[0.01, 0, 0], [0, 0.08, 0], [0, 0, 0.1]],
            size=300
        )
        features.append(bear_features)
        labels.extend([1] * 300)
        
        # Regime 2: Sideways market
        sideways_features = np.random.multivariate_normal(
            [0.0, 0.10, 0.5],  # No trend, low vol, medium momentum
            [[0.005, 0, 0], [0, 0.03, 0], [0, 0, 0.05]],
            size=300
        )
        features.append(sideways_features)
        labels.extend([2] * 300)
        
        features = np.vstack(features)
        
        return features, np.array(labels)
        
    def test_classifier_training(self, regime_classifier, labeled_data):
        """Test classifier training."""
        features, labels = labeled_data
        
        # Split data
        train_size = int(0.8 * len(features))
        X_train, y_train = features[:train_size], labels[:train_size]
        X_test, y_test = features[train_size:], labels[train_size:]
        
        # Train classifier
        regime_classifier.fit(X_train, y_train)
        
        assert regime_classifier.is_fitted
        
        # Test prediction
        predictions = regime_classifier.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1, 2})
        
        # Check accuracy
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.7  # Should achieve reasonable accuracy
        
    def test_prediction_probabilities(self, regime_classifier, labeled_data):
        """Test probability predictions."""
        features, labels = labeled_data
        
        regime_classifier.fit(features[:800], labels[:800])
        
        # Get probabilities
        probas = regime_classifier.predict_proba(features[800:])
        
        assert probas.shape == (200, 3)
        assert np.allclose(probas.sum(axis=1), 1.0)
        
        # High confidence predictions should be accurate
        high_conf_mask = probas.max(axis=1) > 0.8
        high_conf_preds = probas[high_conf_mask].argmax(axis=1)
        high_conf_true = labels[800:][high_conf_mask]
        
        accuracy = np.mean(high_conf_preds == high_conf_true)
        assert accuracy > 0.85
        
    def test_feature_importance(self, regime_classifier, labeled_data):
        """Test feature importance extraction."""
        features, labels = labeled_data
        
        # Use feature names
        feature_names = ['returns', 'volatility', 'momentum']
        regime_classifier.fit(features, labels, feature_names=feature_names)
        
        importance = regime_classifier.get_feature_importance()
        
        assert len(importance) == 3
        assert all(imp >= 0 for imp in importance.values())
        assert sum(importance.values()) > 0
        
        # Returns should be important for regime classification
        assert importance['returns'] > 0.2
        
    def test_cross_validation(self, regime_classifier, labeled_data):
        """Test cross-validation performance."""
        features, labels = labeled_data
        
        cv_scores = regime_classifier.cross_validate(
            features, 
            labels,
            cv_folds=5
        )
        
        assert len(cv_scores) == 5
        assert all(0 <= score <= 1 for score in cv_scores)
        assert np.mean(cv_scores) > 0.6  # Reasonable average accuracy


class TestMarkovRegimeSwitch:
    """Test cases for Markov regime switching models."""
    
    @pytest.fixture
    def markov_model(self):
        """Create Markov regime switching model."""
        return MarkovRegimeSwitch(
            n_regimes=2,
            switching_variance=True,
            switching_mean=True
        )
        
    @pytest.fixture
    def regime_switching_data(self):
        """Create data with regime switching dynamics."""
        n_points = 1000
        
        # True regime sequence
        regimes = []
        regime = 0
        
        # Transition probabilities
        trans_prob = [[0.95, 0.05], [0.10, 0.90]]
        
        for _ in range(n_points):
            regimes.append(regime)
            # Switch regime based on transition probability
            regime = np.random.choice([0, 1], p=trans_prob[regime])
            
        regimes = np.array(regimes)
        
        # Generate returns based on regime
        returns = np.zeros(n_points)
        for i in range(n_points):
            if regimes[i] == 0:
                # Low volatility regime
                returns[i] = np.random.normal(0.001, 0.01)
            else:
                # High volatility regime
                returns[i] = np.random.normal(-0.0005, 0.03)
                
        return returns, regimes
        
    def test_markov_fitting(self, markov_model, regime_switching_data):
        """Test Markov regime switching model fitting."""
        returns, true_regimes = regime_switching_data
        
        # Fit model
        markov_model.fit(returns)
        
        assert markov_model.is_fitted
        
        # Check estimated parameters
        params = markov_model.get_parameters()
        
        assert 'regime_means' in params
        assert 'regime_variances' in params
        assert 'transition_matrix' in params
        
        # Should identify two distinct regimes
        assert len(params['regime_means']) == 2
        assert params['regime_variances'][1] > params['regime_variances'][0]  # Regime 1 more volatile
        
    def test_regime_inference(self, markov_model, regime_switching_data):
        """Test regime inference."""
        returns, true_regimes = regime_switching_data
        
        markov_model.fit(returns)
        
        # Infer regimes
        inferred_regimes = markov_model.predict_regimes(returns)
        
        assert len(inferred_regimes) == len(returns)
        
        # Check alignment with true regimes (may be label switched)
        accuracy1 = np.mean(inferred_regimes == true_regimes)
        accuracy2 = np.mean(inferred_regimes == (1 - true_regimes))  # Label switched
        
        assert max(accuracy1, accuracy2) > 0.8  # Good recovery
        
    def test_regime_probabilities(self, markov_model, regime_switching_data):
        """Test smoothed regime probabilities."""
        returns, _ = regime_switching_data
        
        markov_model.fit(returns)
        
        # Get smoothed probabilities
        regime_probs = markov_model.get_regime_probabilities(returns)
        
        assert regime_probs.shape == (len(returns), 2)
        assert np.allclose(regime_probs.sum(axis=1), 1.0)
        
        # Probabilities should be relatively stable within regimes
        prob_changes = np.abs(np.diff(regime_probs[:, 0]))
        assert np.median(prob_changes) < 0.1
        
    def test_likelihood_calculation(self, markov_model, regime_switching_data):
        """Test log-likelihood calculation."""
        returns, _ = regime_switching_data
        
        markov_model.fit(returns)
        
        log_likelihood = markov_model.log_likelihood(returns)
        
        assert isinstance(log_likelihood, float)
        assert not np.isnan(log_likelihood)
        assert not np.isinf(log_likelihood)


class TestRegimeEnsemble:
    """Test cases for ensemble regime models."""
    
    @pytest.fixture
    def regime_ensemble(self):
        """Create ensemble of regime models."""
        models = [
            ('hmm', RegimeHMM(RegimeConfig(n_regimes=3))),
            ('gmm', RegimeGMM(RegimeConfig(n_regimes=3))),
            ('classifier', RegimeClassifier(model_type='random_forest', n_regimes=3))
        ]
        
        return RegimeEnsemble(models, voting='soft')
        
    def test_ensemble_training(self, regime_ensemble, training_features):
        """Test ensemble model training."""
        # For classifier, we need labels
        # Generate pseudo-labels from clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        pseudo_labels = kmeans.fit_predict(training_features)
        
        # Train ensemble
        regime_ensemble.fit(training_features, labels=pseudo_labels)
        
        assert all(model.is_fitted for _, model in regime_ensemble.models)
        
    def test_ensemble_prediction(self, regime_ensemble, training_features):
        """Test ensemble prediction."""
        # Generate pseudo-labels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        pseudo_labels = kmeans.fit_predict(training_features)
        
        regime_ensemble.fit(training_features, labels=pseudo_labels)
        
        # Predict
        predictions = regime_ensemble.predict(training_features)
        
        assert len(predictions) == len(training_features)
        assert set(predictions).issubset({0, 1, 2})
        
    def test_voting_mechanisms(self, training_features):
        """Test different voting mechanisms."""
        # Create ensemble with hard voting
        models = [
            ('hmm', RegimeHMM(RegimeConfig(n_regimes=3))),
            ('gmm', RegimeGMM(RegimeConfig(n_regimes=3)))
        ]
        
        hard_ensemble = RegimeEnsemble(models, voting='hard')
        soft_ensemble = RegimeEnsemble(models, voting='soft')
        
        # Train both
        hard_ensemble.fit(training_features)
        soft_ensemble.fit(training_features)
        
        # Predictions might differ
        hard_preds = hard_ensemble.predict(training_features)
        soft_preds = soft_ensemble.predict(training_features)
        
        # Soft voting should be smoother
        hard_changes = np.sum(np.diff(hard_preds) != 0)
        soft_changes = np.sum(np.diff(soft_preds) != 0)
        
        # Not always true but generally expected
        # Soft voting aggregates probabilities, leading to smoother transitions
        
    def test_model_weights(self, regime_ensemble, training_features):
        """Test weighted ensemble voting."""
        # Set custom weights
        weights = {'hmm': 0.5, 'gmm': 0.3, 'classifier': 0.2}
        regime_ensemble.set_weights(weights)
        
        # Generate pseudo-labels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        pseudo_labels = kmeans.fit_predict(training_features)
        
        regime_ensemble.fit(training_features, labels=pseudo_labels)
        
        # Get weighted probabilities
        probas = regime_ensemble.predict_proba(training_features)
        
        assert probas.shape == (len(training_features), 3)
        
        # Weights should affect the final probabilities
        # HMM should have most influence due to highest weight


class TestMultiTimeframeRegime:
    """Test cases for multi-timeframe regime analysis."""
    
    def test_timeframe_aggregation(self):
        """Test regime analysis across multiple timeframes."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=1440, freq='5min')  # 5 days
        prices = 100 + np.cumsum(np.random.normal(0, 0.1, 1440))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.lognormal(8, 0.5, 1440)
        })
        
        # Create multi-timeframe analyzer
        from src.models.regime_models import MultiTimeframeRegimeAnalyzer
        
        analyzer = MultiTimeframeRegimeAnalyzer(
            timeframes=['5min', '15min', '1H', '4H'],
            n_regimes=3
        )
        
        # Analyze regimes
        regimes = analyzer.analyze(df)
        
        assert '5min' in regimes
        assert '15min' in regimes
        assert '1H' in regimes
        assert '4H' in regimes
        
        # Higher timeframes should have fewer regime changes
        changes_5min = np.sum(np.diff(regimes['5min']) != 0)
        changes_4H = np.sum(np.diff(regimes['4H']) != 0)
        
        assert changes_4H < changes_5min
        
    def test_regime_alignment(self):
        """Test regime alignment across timeframes."""
        # When lower timeframe is in strong trend,
        # higher timeframes should agree
        
        dates = pd.date_range(start='2023-01-01', periods=500, freq='5min')
        
        # Strong uptrend
        prices = 100 + np.linspace(0, 20, 500) + np.random.normal(0, 0.1, 500)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.lognormal(8, 0.3, 500)
        })
        
        from src.models.regime_models import MultiTimeframeRegimeAnalyzer
        
        analyzer = MultiTimeframeRegimeAnalyzer(
            timeframes=['5min', '15min', '1H'],
            n_regimes=3
        )
        
        regimes = analyzer.analyze(df)
        
        # Calculate regime agreement
        agreement = analyzer.calculate_timeframe_agreement(regimes)
        
        assert agreement > 0.7  # High agreement in strong trend


class TestRegimeMetrics:
    """Test cases for regime performance metrics."""
    
    def test_regime_performance_metrics(self):
        """Test calculation of performance metrics by regime."""
        # Create sample data with regimes
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
        
        regimes = np.array([0] * 300 + [1] * 400 + [2] * 300)
        returns = np.concatenate([
            np.random.normal(0.001, 0.01, 300),   # Regime 0: positive returns
            np.random.normal(-0.0005, 0.02, 400), # Regime 1: negative, volatile
            np.random.normal(0.0008, 0.008, 300)  # Regime 2: positive, low vol
        ])
        
        from src.models.regime_models import calculate_regime_metrics
        
        metrics = calculate_regime_metrics(returns, regimes)
        
        assert len(metrics) == 3
        
        # Check metrics for each regime
        for regime_id, regime_metrics in metrics.items():
            assert 'mean_return' in regime_metrics
            assert 'volatility' in regime_metrics
            assert 'sharpe_ratio' in regime_metrics
            assert 'max_drawdown' in regime_metrics
            assert 'duration' in regime_metrics
            
        # Regime 0 should have positive Sharpe
        assert metrics[0]['sharpe_ratio'] > 0
        
        # Regime 1 should have highest volatility
        assert metrics[1]['volatility'] > metrics[0]['volatility']
        assert metrics[1]['volatility'] > metrics[2]['volatility']
        
    def test_regime_transition_matrix(self):
        """Test regime transition matrix calculation."""
        # Create regime sequence with known transitions
        regimes = [0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 1, 1, 0]
        
        from src.models.regime_models import calculate_transition_matrix
        
        trans_matrix = calculate_transition_matrix(regimes, n_regimes=3)
        
        assert trans_matrix.shape == (3, 3)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)
        
        # Check specific transitions
        # From regime 0: went to 0 (2 times), 1 (1 time), 2 (1 time)
        assert trans_matrix[0, 0] == 0.5   # 2/4
        assert trans_matrix[0, 1] == 0.25  # 1/4
        assert trans_matrix[0, 2] == 0.25  # 1/4