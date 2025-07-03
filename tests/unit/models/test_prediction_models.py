"""
Unit tests for Prediction Models.

Tests cover:
- Time series forecasting models (ARIMA, GARCH, etc.)
- Machine learning models (RF, XGBoost, etc.)
- Deep learning models (LSTM, GRU, CNN)
- Ensemble prediction models
- Feature engineering for predictions
- Model evaluation and backtesting
- Multi-step ahead forecasting
- Uncertainty quantification
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Optional, Tuple

# Assuming the module structure
from src.models.prediction_models import (
    PricePredictionModel,
    ARIMAPredictor,
    GARCHPredictor,
    LSTMPredictor,
    GRUPredictor,
    CNNPredictor,
    TransformerPredictor,
    EnsemblePredictor,
    PredictionConfig,
    FeatureEngineering,
    ModelEvaluator
)


class TestPredictionConfig:
    """Test cases for prediction model configuration."""
    
    def test_default_config(self):
        """Test default prediction configuration."""
        config = PredictionConfig()
        
        assert config.prediction_horizon == 1
        assert config.lookback_window == 60
        assert config.feature_set == 'default'
        assert config.validation_split == 0.2
        assert config.use_ensemble is False
        
    def test_custom_config(self):
        """Test custom prediction configuration."""
        config = PredictionConfig(
            prediction_horizon=5,
            lookback_window=120,
            feature_set='extended',
            use_ensemble=True,
            confidence_intervals=True
        )
        
        assert config.prediction_horizon == 5
        assert config.lookback_window == 120
        assert config.confidence_intervals is True
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid prediction horizon
        with pytest.raises(ValueError, match="prediction_horizon must be"):
            PredictionConfig(prediction_horizon=0)
            
        # Invalid lookback window
        with pytest.raises(ValueError, match="lookback_window must be"):
            PredictionConfig(lookback_window=1)
            
        # Invalid validation split
        with pytest.raises(ValueError, match="validation_split must be"):
            PredictionConfig(validation_split=1.5)


class TestFeatureEngineering:
    """Test cases for feature engineering."""
    
    @pytest.fixture
    def feature_eng(self):
        """Create feature engineering instance."""
        return FeatureEngineering(
            price_features=True,
            technical_features=True,
            statistical_features=True,
            market_features=True
        )
        
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
        
        # Generate realistic price series
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, 1000)
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, 1000)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 1000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 1000))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, 1000)
        })
        
        return df
        
    def test_price_features(self, feature_eng, sample_data):
        """Test price-based feature engineering."""
        features = feature_eng.create_price_features(sample_data)
        
        expected_features = [
            'returns', 'log_returns', 'return_volatility',
            'price_momentum', 'price_acceleration',
            'high_low_ratio', 'close_open_ratio'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
            
        # Check feature properties
        assert abs(features['returns'].mean()) < 0.01
        assert features['return_volatility'].min() >= 0
        
    def test_technical_features(self, feature_eng, sample_data):
        """Test technical indicator features."""
        features = feature_eng.create_technical_features(sample_data)
        
        expected_features = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'atr_14', 'adx_14', 'cci_20'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
            
        # RSI should be between 0 and 100
        assert features['rsi_14'].dropna().between(0, 100).all()
        
    def test_statistical_features(self, feature_eng, sample_data):
        """Test statistical features."""
        features = feature_eng.create_statistical_features(sample_data)
        
        expected_features = [
            'skewness', 'kurtosis', 'autocorrelation',
            'partial_autocorrelation', 'hurst_exponent',
            'sample_entropy', 'permutation_entropy'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
            
        # Hurst exponent should be between 0 and 1
        assert features['hurst_exponent'].dropna().between(0, 1).all()
        
    def test_market_features(self, feature_eng, sample_data):
        """Test market microstructure features."""
        features = feature_eng.create_market_features(sample_data)
        
        expected_features = [
            'bid_ask_spread', 'volume_imbalance', 'trade_intensity',
            'price_impact', 'realized_volatility', 'volume_weighted_price'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
            
    def test_feature_lags(self, feature_eng, sample_data):
        """Test lagged feature creation."""
        base_features = feature_eng.create_price_features(sample_data)
        
        # Create lagged features
        lagged_features = feature_eng.create_lagged_features(
            base_features,
            columns=['returns', 'return_volatility'],
            lags=[1, 5, 10]
        )
        
        expected_lags = [
            'returns_lag_1', 'returns_lag_5', 'returns_lag_10',
            'return_volatility_lag_1', 'return_volatility_lag_5', 'return_volatility_lag_10'
        ]
        
        for feat in expected_lags:
            assert feat in lagged_features.columns
            
    def test_rolling_features(self, feature_eng, sample_data):
        """Test rolling window features."""
        base_features = feature_eng.create_price_features(sample_data)
        
        rolling_features = feature_eng.create_rolling_features(
            base_features,
            columns=['returns'],
            windows=[5, 20],
            functions=['mean', 'std', 'min', 'max']
        )
        
        expected_rolling = [
            'returns_rolling_mean_5', 'returns_rolling_std_5',
            'returns_rolling_mean_20', 'returns_rolling_std_20'
        ]
        
        for feat in expected_rolling:
            assert feat in rolling_features.columns


class TestARIMAPredictor:
    """Test cases for ARIMA prediction model."""
    
    @pytest.fixture
    def arima_predictor(self):
        """Create ARIMA predictor instance."""
        config = PredictionConfig(
            prediction_horizon=5,
            lookback_window=100
        )
        return ARIMAPredictor(config)
        
    @pytest.fixture
    def time_series_data(self):
        """Create time series data for ARIMA."""
        dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
        
        # Generate AR(1) process with trend
        ar_coef = 0.7
        trend = 0.01
        noise_std = 0.5
        
        data = [100]
        for i in range(1, 500):
            next_val = trend + ar_coef * (data[-1] - 100) + 100 + np.random.normal(0, noise_std)
            data.append(next_val)
            
        return pd.Series(data, index=dates, name='price')
        
    def test_arima_order_selection(self, arima_predictor, time_series_data):
        """Test automatic ARIMA order selection."""
        order = arima_predictor.select_order(time_series_data)
        
        assert len(order) == 3  # (p, d, q)
        assert all(isinstance(o, int) for o in order)
        assert all(o >= 0 for o in order)
        
    def test_arima_fitting(self, arima_predictor, time_series_data):
        """Test ARIMA model fitting."""
        arima_predictor.fit(time_series_data)
        
        assert arima_predictor.is_fitted
        assert hasattr(arima_predictor, 'model')
        
        # Check model parameters
        params = arima_predictor.get_parameters()
        assert 'ar_params' in params
        assert 'ma_params' in params
        
    def test_arima_prediction(self, arima_predictor, time_series_data):
        """Test ARIMA predictions."""
        # Fit model
        train_data = time_series_data[:-20]
        test_data = time_series_data[-20:]
        
        arima_predictor.fit(train_data)
        
        # Make predictions
        predictions = arima_predictor.predict(steps=20)
        
        assert len(predictions) == 20
        assert isinstance(predictions, pd.Series)
        
        # Predictions should be reasonable
        assert predictions.min() > train_data.min() * 0.5
        assert predictions.max() < train_data.max() * 1.5
        
    def test_arima_confidence_intervals(self, arima_predictor, time_series_data):
        """Test ARIMA prediction intervals."""
        arima_predictor.fit(time_series_data)
        
        predictions, lower, upper = arima_predictor.predict_with_intervals(
            steps=10,
            alpha=0.05  # 95% confidence
        )
        
        assert len(predictions) == len(lower) == len(upper) == 10
        assert (lower < predictions).all()
        assert (predictions < upper).all()
        
        # Intervals should widen with horizon
        interval_widths = upper - lower
        assert interval_widths.iloc[-1] > interval_widths.iloc[0]


class TestGARCHPredictor:
    """Test cases for GARCH volatility prediction model."""
    
    @pytest.fixture
    def garch_predictor(self):
        """Create GARCH predictor instance."""
        config = PredictionConfig(
            prediction_horizon=10
        )
        return GARCHPredictor(config)
        
    @pytest.fixture
    def returns_data(self):
        """Create returns data with volatility clustering."""
        n_points = 1000
        
        # Generate returns with GARCH(1,1) dynamics
        omega = 0.00001
        alpha = 0.1
        beta = 0.85
        
        returns = []
        variances = [0.0001]
        
        for i in range(n_points):
            variance = omega + alpha * (returns[-1]**2 if returns else 0) + beta * variances[-1]
            variances.append(variance)
            returns.append(np.random.normal(0, np.sqrt(variance)))
            
        return pd.Series(returns, name='returns')
        
    def test_garch_fitting(self, garch_predictor, returns_data):
        """Test GARCH model fitting."""
        garch_predictor.fit(returns_data)
        
        assert garch_predictor.is_fitted
        
        # Check estimated parameters
        params = garch_predictor.get_parameters()
        assert 'omega' in params
        assert 'alpha' in params
        assert 'beta' in params
        
        # Parameters should be reasonable
        assert params['alpha'] + params['beta'] < 1  # Stationarity
        
    def test_volatility_forecast(self, garch_predictor, returns_data):
        """Test volatility forecasting."""
        garch_predictor.fit(returns_data)
        
        # Forecast volatility
        vol_forecast = garch_predictor.forecast_volatility(horizon=10)
        
        assert len(vol_forecast) == 10
        assert (vol_forecast > 0).all()
        
        # Volatility should converge to long-term level
        assert abs(vol_forecast.iloc[-1] - vol_forecast.iloc[-2]) < abs(vol_forecast.iloc[1] - vol_forecast.iloc[0])
        
    def test_garch_variants(self):
        """Test different GARCH model variants."""
        returns = pd.Series(np.random.normal(0, 0.01, 500))
        
        # Test EGARCH
        egarch = GARCHPredictor(
            PredictionConfig(),
            model_type='EGARCH'
        )
        egarch.fit(returns)
        
        # Test GJR-GARCH
        gjr = GARCHPredictor(
            PredictionConfig(),
            model_type='GJR-GARCH'
        )
        gjr.fit(returns)
        
        # Both should fit successfully
        assert egarch.is_fitted
        assert gjr.is_fitted


class TestLSTMPredictor:
    """Test cases for LSTM prediction model."""
    
    @pytest.fixture
    def lstm_predictor(self):
        """Create LSTM predictor instance."""
        config = PredictionConfig(
            prediction_horizon=5,
            lookback_window=50
        )
        return LSTMPredictor(
            config,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
    @pytest.fixture
    def sequence_data(self):
        """Create sequence data for LSTM."""
        # Generate synthetic time series with patterns
        t = np.linspace(0, 100, 1000)
        
        # Combine trend, seasonality, and noise
        trend = 0.5 * t
        seasonal = 10 * np.sin(0.1 * t)
        noise = np.random.normal(0, 2, 1000)
        
        values = 100 + trend + seasonal + noise
        
        return pd.DataFrame({
            'value': values,
            'feature1': np.sin(0.05 * t),
            'feature2': np.cos(0.05 * t)
        })
        
    def test_lstm_architecture(self, lstm_predictor):
        """Test LSTM model architecture."""
        model = lstm_predictor.build_model(input_size=3)
        
        assert isinstance(model.lstm, nn.LSTM)
        assert model.lstm.input_size == 3
        assert model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 2
        assert model.lstm.dropout == 0.2
        
        # Check output layer
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.out_features == 5  # prediction_horizon
        
    def test_sequence_preparation(self, lstm_predictor, sequence_data):
        """Test sequence data preparation."""
        X, y = lstm_predictor.prepare_sequences(
            sequence_data,
            target_col='value'
        )
        
        expected_sequences = len(sequence_data) - lstm_predictor.lookback_window - lstm_predictor.prediction_horizon + 1
        
        assert X.shape[0] == expected_sequences
        assert X.shape[1] == lstm_predictor.lookback_window
        assert X.shape[2] == 3  # number of features
        
        assert y.shape[0] == expected_sequences
        assert y.shape[1] == lstm_predictor.prediction_horizon
        
    def test_lstm_training(self, lstm_predictor, sequence_data):
        """Test LSTM model training."""
        # Prepare data
        X, y = lstm_predictor.prepare_sequences(sequence_data, target_col='value')
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        # Train model
        history = lstm_predictor.train(
            X_train, y_train,
            X_val, y_val,
            epochs=5,
            batch_size=32
        )
        
        assert lstm_predictor.is_fitted
        assert 'train_loss' in history
        assert 'val_loss' in history
        
        # Loss should decrease
        assert history['train_loss'][-1] < history['train_loss'][0]
        
    def test_lstm_prediction(self, lstm_predictor, sequence_data):
        """Test LSTM predictions."""
        # Prepare and train
        X, y = lstm_predictor.prepare_sequences(sequence_data, target_col='value')
        
        train_size = int(0.8 * len(X))
        lstm_predictor.train(
            X[:train_size], y[:train_size],
            X[train_size:], y[train_size:],
            epochs=5
        )
        
        # Make predictions
        test_input = X[-1].unsqueeze(0)  # Last sequence
        predictions = lstm_predictor.predict(test_input)
        
        assert predictions.shape == (1, 5)  # batch_size=1, horizon=5
        
        # Convert to numpy for easier testing
        pred_values = predictions.numpy().flatten()
        
        # Predictions should be reasonable
        recent_values = sequence_data['value'].iloc[-50:].values
        assert pred_values.min() > recent_values.min() * 0.5
        assert pred_values.max() < recent_values.max() * 1.5
        
    def test_lstm_uncertainty(self, lstm_predictor, sequence_data):
        """Test LSTM with uncertainty estimation."""
        # Enable dropout at inference
        lstm_predictor.enable_uncertainty = True
        
        X, y = lstm_predictor.prepare_sequences(sequence_data, target_col='value')
        lstm_predictor.train(X[:400], y[:400], X[400:], y[400:], epochs=5)
        
        # Make multiple predictions
        test_input = X[-1].unsqueeze(0)
        predictions = []
        
        for _ in range(100):
            pred = lstm_predictor.predict(test_input, training=True)  # Keep dropout on
            predictions.append(pred.numpy())
            
        predictions = np.array(predictions)
        
        # Calculate mean and std
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        assert mean_pred.shape == (1, 5)
        assert std_pred.shape == (1, 5)
        assert (std_pred > 0).all()  # Should have uncertainty


class TestGRUPredictor:
    """Test cases for GRU prediction model."""
    
    @pytest.fixture
    def gru_predictor(self):
        """Create GRU predictor instance."""
        config = PredictionConfig(
            prediction_horizon=3,
            lookback_window=30
        )
        return GRUPredictor(
            config,
            hidden_size=32,
            num_layers=1
        )
        
    def test_gru_architecture(self, gru_predictor):
        """Test GRU model architecture."""
        model = gru_predictor.build_model(input_size=5)
        
        assert isinstance(model.gru, nn.GRU)
        assert model.gru.input_size == 5
        assert model.gru.hidden_size == 32
        
        # GRU should have fewer parameters than LSTM
        gru_params = sum(p.numel() for p in model.gru.parameters())
        
        # Compare with equivalent LSTM
        lstm = nn.LSTM(5, 32, 1)
        lstm_params = sum(p.numel() for p in lstm.parameters())
        
        assert gru_params < lstm_params
        
    def test_gru_speed_comparison(self, gru_predictor):
        """Test GRU training speed vs LSTM."""
        # Create equivalent LSTM
        lstm_predictor = LSTMPredictor(
            PredictionConfig(prediction_horizon=3, lookback_window=30),
            hidden_size=32,
            num_layers=1
        )
        
        # Create dummy data
        X = torch.randn(100, 30, 5)
        y = torch.randn(100, 3)
        
        # Time GRU training
        import time
        gru_model = gru_predictor.build_model(5)
        
        start = time.time()
        for _ in range(10):
            output = gru_model(X)
            loss = nn.MSELoss()(output, y)
            loss.backward()
        gru_time = time.time() - start
        
        # Time LSTM training
        lstm_model = lstm_predictor.build_model(5)
        
        start = time.time()
        for _ in range(10):
            output = lstm_model(X)
            loss = nn.MSELoss()(output, y)
            loss.backward()
        lstm_time = time.time() - start
        
        # GRU should be faster
        assert gru_time < lstm_time * 1.2  # Allow some variance


class TestCNNPredictor:
    """Test cases for CNN prediction model."""
    
    @pytest.fixture
    def cnn_predictor(self):
        """Create CNN predictor instance."""
        config = PredictionConfig(
            prediction_horizon=1,
            lookback_window=60
        )
        return CNNPredictor(
            config,
            n_filters=[32, 64, 128],
            kernel_sizes=[3, 5, 7],
            use_residual=True
        )
        
    def test_cnn_architecture(self, cnn_predictor):
        """Test CNN model architecture."""
        model = cnn_predictor.build_model(input_channels=5)
        
        # Check convolutional layers
        assert len(model.conv_layers) == 3
        assert model.conv_layers[0].in_channels == 5
        assert model.conv_layers[0].out_channels == 32
        
        # Check residual connections
        assert hasattr(model, 'residual_connections')
        
    def test_cnn_feature_extraction(self, cnn_predictor):
        """Test CNN feature extraction."""
        # Create input with patterns
        batch_size = 16
        seq_len = 60
        n_features = 5
        
        # Create data with local patterns
        X = torch.zeros(batch_size, n_features, seq_len)
        
        # Add patterns at different scales
        for i in range(batch_size):
            # Short-term pattern
            X[i, 0, :10] = torch.sin(torch.linspace(0, 2*np.pi, 10))
            # Medium-term pattern
            X[i, 1, :30] = torch.cos(torch.linspace(0, 4*np.pi, 30))
            # Long-term pattern
            X[i, 2, :] = torch.linspace(0, 1, seq_len)
            
        model = cnn_predictor.build_model(n_features)
        features = model.extract_features(X)
        
        # Should extract hierarchical features
        assert len(features) == 3  # One per conv layer
        assert features[0].shape[1] == 32  # First layer filters
        assert features[1].shape[1] == 64  # Second layer filters
        assert features[2].shape[1] == 128  # Third layer filters
        
    def test_dilated_convolutions(self):
        """Test CNN with dilated convolutions."""
        config = PredictionConfig(lookback_window=100)
        
        dilated_cnn = CNNPredictor(
            config,
            n_filters=[32, 32, 32],
            kernel_sizes=[3, 3, 3],
            dilation_rates=[1, 2, 4]
        )
        
        model = dilated_cnn.build_model(input_channels=3)
        
        # Check receptive field
        # Dilated convolutions should have larger receptive field
        X = torch.randn(1, 3, 100)
        output = model(X)
        
        # Should capture longer dependencies
        assert output.shape[-1] == 1  # Single prediction


class TestTransformerPredictor:
    """Test cases for Transformer prediction model."""
    
    @pytest.fixture
    def transformer_predictor(self):
        """Create Transformer predictor instance."""
        config = PredictionConfig(
            prediction_horizon=5,
            lookback_window=50
        )
        return TransformerPredictor(
            config,
            d_model=128,
            n_heads=8,
            n_layers=3,
            d_ff=512
        )
        
    def test_transformer_architecture(self, transformer_predictor):
        """Test Transformer model architecture."""
        model = transformer_predictor.build_model(input_size=10)
        
        # Check input projection
        assert model.input_projection.in_features == 10
        assert model.input_projection.out_features == 128
        
        # Check transformer layers
        assert len(model.transformer.layers) == 3
        
        # Check attention mechanism
        layer = model.transformer.layers[0]
        assert layer.self_attn.embed_dim == 128
        assert layer.self_attn.num_heads == 8
        
    def test_positional_encoding(self, transformer_predictor):
        """Test positional encoding in transformer."""
        model = transformer_predictor.build_model(input_size=10)
        
        # Check positional encoding
        seq_len = 50
        d_model = 128
        
        pe = model.pos_encoder.pe[:seq_len, :]
        
        # Should have sinusoidal patterns
        assert pe.shape == (seq_len, d_model)
        
        # Check alternating sin/cos
        assert torch.abs(pe[0, 0]) < 1e-6  # sin(0) = 0
        assert torch.abs(pe[0, 1] - 1.0) < 1e-6  # cos(0) = 1
        
    def test_attention_visualization(self, transformer_predictor):
        """Test attention weight extraction."""
        model = transformer_predictor.build_model(input_size=5)
        model.eval()
        
        # Forward pass with attention
        X = torch.randn(4, 50, 5)
        
        output, attention_weights = model(X, return_attention=True)
        
        assert len(attention_weights) == 3  # One per layer
        assert attention_weights[0].shape == (4, 8, 50, 50)  # batch, heads, seq, seq
        
        # Attention should sum to 1
        assert torch.allclose(
            attention_weights[0].sum(dim=-1),
            torch.ones_like(attention_weights[0].sum(dim=-1))
        )


class TestEnsemblePredictor:
    """Test cases for ensemble prediction models."""
    
    @pytest.fixture
    def ensemble_predictor(self):
        """Create ensemble predictor instance."""
        config = PredictionConfig(
            prediction_horizon=3,
            lookback_window=30
        )
        
        models = [
            ('arima', ARIMAPredictor(config)),
            ('lstm', LSTMPredictor(config, hidden_size=32)),
            ('gru', GRUPredictor(config, hidden_size=32))
        ]
        
        return EnsemblePredictor(models, method='weighted_average')
        
    @pytest.fixture
    def ensemble_data(self):
        """Create data for ensemble testing."""
        dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
        
        # Create data with multiple patterns
        trend = np.linspace(100, 120, 500)
        seasonal = 5 * np.sin(np.linspace(0, 20*np.pi, 500))
        noise = np.random.normal(0, 2, 500)
        
        values = trend + seasonal + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'volume': np.random.lognormal(10, 0.5, 500)
        })
        
    def test_ensemble_training(self, ensemble_predictor, ensemble_data):
        """Test ensemble model training."""
        # Train all models
        ensemble_predictor.fit(ensemble_data, target_col='value')
        
        assert all(model.is_fitted for _, model in ensemble_predictor.models)
        
    def test_ensemble_prediction(self, ensemble_predictor, ensemble_data):
        """Test ensemble predictions."""
        # Train ensemble
        train_data = ensemble_data.iloc[:-50]
        test_data = ensemble_data.iloc[-50:]
        
        ensemble_predictor.fit(train_data, target_col='value')
        
        # Make predictions
        predictions = ensemble_predictor.predict(test_data.iloc[:30])
        
        assert len(predictions) == 3  # prediction_horizon
        
        # Get individual model predictions
        individual_preds = ensemble_predictor.get_individual_predictions(test_data.iloc[:30])
        
        assert len(individual_preds) == 3  # Number of models
        assert all(len(pred) == 3 for pred in individual_preds.values())
        
    def test_weighted_ensemble(self, ensemble_predictor, ensemble_data):
        """Test weighted average ensemble."""
        # Set custom weights
        weights = {'arima': 0.5, 'lstm': 0.3, 'gru': 0.2}
        ensemble_predictor.set_weights(weights)
        
        ensemble_predictor.fit(ensemble_data.iloc[:-50], target_col='value')
        
        # Make predictions
        test_input = ensemble_data.iloc[-50:-47]
        ensemble_pred = ensemble_predictor.predict(test_input)
        individual_preds = ensemble_predictor.get_individual_predictions(test_input)
        
        # Check weighted average
        expected = (
            weights['arima'] * individual_preds['arima'] +
            weights['lstm'] * individual_preds['lstm'] +
            weights['gru'] * individual_preds['gru']
        )
        
        np.testing.assert_allclose(ensemble_pred, expected, rtol=1e-5)
        
    def test_ensemble_stacking(self):
        """Test stacking ensemble method."""
        config = PredictionConfig(prediction_horizon=1)
        
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=10)),
            ('lstm', LSTMPredictor(config, hidden_size=16))
        ]
        
        stacking_ensemble = EnsemblePredictor(
            base_models,
            method='stacking',
            meta_model=RandomForestRegressor(n_estimators=5)
        )
        
        # Create simple data
        X = np.random.randn(200, 10)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.normal(0, 0.1, 200)
        
        # Train stacking ensemble
        stacking_ensemble.fit_stacking(X[:150], y[:150], X[150:], y[150:])
        
        # Predict
        predictions = stacking_ensemble.predict_stacking(X[-10:])
        
        assert len(predictions) == 10
        
    def test_ensemble_uncertainty(self, ensemble_predictor, ensemble_data):
        """Test uncertainty estimation from ensemble."""
        ensemble_predictor.fit(ensemble_data.iloc[:-50], target_col='value')
        
        # Get predictions with uncertainty
        test_input = ensemble_data.iloc[-50:-47]
        mean_pred, std_pred = ensemble_predictor.predict_with_uncertainty(test_input)
        
        assert len(mean_pred) == 3
        assert len(std_pred) == 3
        assert (std_pred > 0).all()
        
        # Uncertainty should increase with horizon
        assert std_pred[2] >= std_pred[0]


class TestModelEvaluator:
    """Test cases for model evaluation and backtesting."""
    
    @pytest.fixture
    def evaluator(self):
        """Create model evaluator instance."""
        return ModelEvaluator(
            metrics=['mse', 'mae', 'mape', 'directional_accuracy']
        )
        
    def test_point_forecast_metrics(self, evaluator):
        """Test point forecast evaluation metrics."""
        y_true = np.array([100, 102, 104, 103, 105])
        y_pred = np.array([101, 103, 103, 104, 106])
        
        metrics = evaluator.evaluate_point_forecasts(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'directional_accuracy' in metrics
        
        # Check metric values
        assert metrics['mse'] == mean_squared_error(y_true, y_pred)
        assert metrics['mae'] == mean_absolute_error(y_true, y_pred)
        assert 0 <= metrics['directional_accuracy'] <= 1
        
    def test_probabilistic_metrics(self, evaluator):
        """Test probabilistic forecast evaluation."""
        y_true = np.array([100, 102, 104, 103, 105])
        y_pred_mean = np.array([101, 103, 103, 104, 106])
        y_pred_std = np.array([1.0, 1.2, 1.5, 1.3, 2.0])
        
        metrics = evaluator.evaluate_probabilistic_forecasts(
            y_true, y_pred_mean, y_pred_std
        )
        
        assert 'crps' in metrics  # Continuous Ranked Probability Score
        assert 'log_score' in metrics
        assert 'interval_coverage_90' in metrics
        assert 'interval_width_90' in metrics
        
    def test_backtesting(self, evaluator):
        """Test backtesting framework."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': 100 + np.cumsum(np.random.normal(0, 1, 200))
        })
        
        # Create simple predictor
        from sklearn.linear_model import LinearRegression
        
        class SimplePredictor:
            def __init__(self):
                self.model = LinearRegression()
                
            def fit(self, data):
                X = np.arange(len(data)).reshape(-1, 1)
                y = data['value'].values
                self.model.fit(X, y)
                
            def predict(self, steps):
                last_index = self.last_index
                X_pred = np.arange(last_index + 1, last_index + steps + 1).reshape(-1, 1)
                return self.model.predict(X_pred)
                
            def update(self, data):
                self.last_index = len(data) - 1
                self.fit(data)
                
        predictor = SimplePredictor()
        
        # Run backtesting
        results = evaluator.backtest(
            predictor,
            data,
            initial_train_size=100,
            step_size=1,
            prediction_horizon=5,
            refit_frequency=20
        )
        
        assert 'predictions' in results
        assert 'actuals' in results
        assert 'metrics' in results
        assert 'timestamps' in results
        
        # Should have multiple evaluation points
        assert len(results['predictions']) > 50
        
    def test_model_comparison(self, evaluator):
        """Test comparing multiple models."""
        # Create predictions from different models
        y_true = np.random.randn(100) * 10 + 100
        
        model_predictions = {
            'model1': y_true + np.random.normal(0, 2, 100),
            'model2': y_true + np.random.normal(0, 3, 100),
            'model3': y_true * 0.95 + np.random.normal(0, 1, 100)
        }
        
        comparison = evaluator.compare_models(y_true, model_predictions)
        
        assert len(comparison) == 3
        
        for model_name, metrics in comparison.items():
            assert 'mse' in metrics
            assert 'mae' in metrics
            
        # Rank models
        ranking = evaluator.rank_models(comparison, metric='mse')
        
        assert len(ranking) == 3
        assert ranking[0][0] in ['model1', 'model2', 'model3']


class TestMultiStepForecasting:
    """Test cases for multi-step ahead forecasting."""
    
    def test_direct_multistep(self):
        """Test direct multi-step forecasting strategy."""
        from src.models.prediction_models import DirectMultiStepPredictor
        
        # Create predictor for each horizon
        predictor = DirectMultiStepPredictor(
            base_model=RandomForestRegressor(n_estimators=10),
            horizons=[1, 3, 5, 10]
        )
        
        # Create data
        X = np.random.randn(200, 10)
        y = np.random.randn(200)
        
        # Train
        predictor.fit(X[:150], y[:150])
        
        # Predict
        predictions = predictor.predict(X[150:160])
        
        assert len(predictions) == 4  # One per horizon
        assert all(len(pred) == 10 for pred in predictions.values())
        
    def test_recursive_multistep(self):
        """Test recursive multi-step forecasting strategy."""
        from src.models.prediction_models import RecursiveMultiStepPredictor
        
        class SimpleModel:
            def __init__(self):
                self.coef = None
                
            def fit(self, X, y):
                # Simple AR(1) model
                self.coef = np.corrcoef(X[:, -1], y)[0, 1]
                
            def predict(self, X):
                return X[:, -1] * self.coef
                
        predictor = RecursiveMultiStepPredictor(
            base_model=SimpleModel(),
            max_horizon=5
        )
        
        # Create AR data
        data = [1.0]
        for _ in range(200):
            data.append(0.8 * data[-1] + np.random.normal(0, 0.1))
            
        X = np.array([data[i:i+10] for i in range(190)]).reshape(190, 10)
        y = np.array([data[i+10] for i in range(190)])
        
        # Train and predict
        predictor.fit(X[:150], y[:150])
        predictions = predictor.predict_multistep(X[150], steps=5)
        
        assert len(predictions) == 5
        
        # Should show decay due to AR coefficient
        assert abs(predictions[-1]) < abs(predictions[0])