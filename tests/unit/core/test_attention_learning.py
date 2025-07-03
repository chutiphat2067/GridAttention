"""
Unit tests for the Attention Learning component.

Tests cover:
- Model initialization and configuration
- Forward pass computations
- Attention weight calculations
- Feature importance scoring
- Model adaptation and learning
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

# GridAttention project imports
from core.attention_learning_layer import (
    AttentionLearningLayer,
    MarketAttentionModel,
    FeatureAttention,
    TemporalAttention,
    AttentionConfig,
    AttentionMetrics,
    AttentionPhase
)


class TestAttentionConfig:
    """Test cases for AttentionConfig validation and initialization."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AttentionConfig()
        
        assert config.hidden_dim == 256
        assert config.num_heads == 8
        assert config.dropout_rate == 0.1
        assert config.learning_rate == 0.001
        assert config.warmup_steps == 1000
        assert config.max_sequence_length == 100
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AttentionConfig(
            hidden_dim=512,
            num_heads=16,
            dropout_rate=0.2,
            learning_rate=0.0001
        )
        
        assert config.hidden_dim == 512
        assert config.num_heads == 16
        assert config.dropout_rate == 0.2
        assert config.learning_rate == 0.0001
        
    def test_invalid_config(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            # Hidden dim must be divisible by num_heads
            AttentionConfig(hidden_dim=250, num_heads=8)
            
        with pytest.raises(ValueError):
            # Dropout rate must be between 0 and 1
            AttentionConfig(dropout_rate=1.5)
            
        with pytest.raises(ValueError):
            # Learning rate must be positive
            AttentionConfig(learning_rate=-0.001)


class TestFeatureAttention:
    """Test cases for FeatureAttention module."""
    
    @pytest.fixture
    def feature_attention(self):
        """Create a FeatureAttention instance."""
        return FeatureAttention(
            input_dim=64,
            hidden_dim=256,
            num_heads=8
        )
        
    def test_initialization(self, feature_attention):
        """Test proper initialization of layers."""
        assert isinstance(feature_attention.query_proj, nn.Linear)
        assert isinstance(feature_attention.key_proj, nn.Linear)
        assert isinstance(feature_attention.value_proj, nn.Linear)
        assert feature_attention.query_proj.in_features == 64
        assert feature_attention.query_proj.out_features == 256
        
    def test_forward_pass(self, feature_attention):
        """Test forward pass computation."""
        batch_size = 32
        seq_len = 50
        input_dim = 64
        
        # Create random input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output, attention_weights = feature_attention(x)
        
        # Check output shapes
        assert output.shape == (batch_size, seq_len, 256)
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)
        
        # Check attention weights sum to 1
        weight_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
        
    def test_masked_attention(self, feature_attention):
        """Test attention with masking."""
        batch_size = 16
        seq_len = 30
        input_dim = 64
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Create mask (mask out last 10 positions)
        mask = torch.ones(batch_size, seq_len)
        mask[:, -10:] = 0
        
        output, attention_weights = feature_attention(x, mask=mask)
        
        # Check masked positions have zero attention
        masked_attention = attention_weights[:, :, :, -10:]
        assert torch.allclose(masked_attention, torch.zeros_like(masked_attention))


class TestTemporalAttention:
    """Test cases for TemporalAttention module."""
    
    @pytest.fixture
    def temporal_attention(self):
        """Create a TemporalAttention instance."""
        return TemporalAttention(
            hidden_dim=256,
            num_heads=8,
            max_seq_length=100
        )
        
    def test_positional_encoding(self, temporal_attention):
        """Test positional encoding generation."""
        seq_len = 50
        hidden_dim = 256
        
        pos_encoding = temporal_attention._generate_positional_encoding(seq_len)
        
        assert pos_encoding.shape == (seq_len, hidden_dim)
        
        # Check that encoding values are bounded
        assert pos_encoding.abs().max() <= 1.0
        
    def test_temporal_forward_pass(self, temporal_attention):
        """Test temporal attention forward pass."""
        batch_size = 16
        seq_len = 40
        hidden_dim = 256
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        output, temporal_weights = temporal_attention(x)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert temporal_weights.shape == (batch_size, 8, seq_len, seq_len)
        
    def test_causal_masking(self, temporal_attention):
        """Test causal masking in temporal attention."""
        batch_size = 8
        seq_len = 20
        hidden_dim = 256
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        output, temporal_weights = temporal_attention(x, causal=True)
        
        # Check that future positions are masked (upper triangular should be zero)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(
                    temporal_weights[:, :, i, j], 
                    torch.zeros_like(temporal_weights[:, :, i, j])
                )


class TestMarketAttentionModel:
    """Test cases for the complete MarketAttentionModel."""
    
    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        return AttentionConfig(
            hidden_dim=128,
            num_heads=4,
            num_features=20,
            num_layers=2
        )
        
    @pytest.fixture
    def market_model(self, model_config):
        """Create a MarketAttentionModel instance."""
        return MarketAttentionModel(config=model_config)
        
    def test_model_initialization(self, market_model):
        """Test model initialization."""
        assert len(market_model.attention_layers) == 2
        assert isinstance(market_model.output_projection, nn.Linear)
        assert market_model.config.num_features == 20
        
    def test_full_forward_pass(self, market_model):
        """Test complete forward pass through the model."""
        batch_size = 16
        seq_len = 50
        num_features = 20
        
        # Create input data
        market_data = torch.randn(batch_size, seq_len, num_features)
        
        # Forward pass
        predictions, attention_maps = market_model(market_data)
        
        # Check outputs
        assert predictions.shape == (batch_size, seq_len, 1)  # Price predictions
        assert len(attention_maps) == 2  # One per layer
        assert attention_maps[0]['feature'].shape == (batch_size, 4, seq_len, seq_len)
        assert attention_maps[0]['temporal'].shape == (batch_size, 4, seq_len, seq_len)
        
    def test_feature_importance_extraction(self, market_model):
        """Test feature importance score extraction."""
        batch_size = 8
        seq_len = 30
        num_features = 20
        
        market_data = torch.randn(batch_size, seq_len, num_features)
        
        predictions, attention_maps = market_model(market_data)
        feature_importance = market_model.get_feature_importance(attention_maps)
        
        assert feature_importance.shape == (num_features,)
        assert torch.all(feature_importance >= 0)
        assert torch.allclose(feature_importance.sum(), torch.tensor(1.0), atol=1e-6)


class TestAttentionLearning:
    """Test cases for the main AttentionLearning class."""
    
    @pytest.fixture
    def attention_learning(self):
        """Create an AttentionLearning instance."""
        config = AttentionConfig(
            hidden_dim=64,
            num_heads=2,
            num_features=10
        )
        return AttentionLearning(config=config)
        
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.randn(100),
            'bb_upper': np.random.randn(100).cumsum() + 102,
            'bb_lower': np.random.randn(100).cumsum() + 98,
            'atr': np.random.uniform(0.5, 2.0, 100)
        })
        return data
        
    def test_preprocessing(self, attention_learning, sample_market_data):
        """Test data preprocessing."""
        processed_data = attention_learning.preprocess_data(sample_market_data)
        
        assert isinstance(processed_data, torch.Tensor)
        assert processed_data.shape[0] == len(sample_market_data)
        assert processed_data.shape[1] == 10  # num_features
        
        # Check normalization
        assert processed_data.abs().max() <= 10  # Reasonable range
        
    def test_training_step(self, attention_learning, sample_market_data):
        """Test single training step."""
        # Prepare data
        processed_data = attention_learning.preprocess_data(sample_market_data)
        
        # Create sequences
        seq_len = 20
        sequences = []
        targets = []
        
        for i in range(len(processed_data) - seq_len):
            sequences.append(processed_data[i:i+seq_len])
            targets.append(processed_data[i+seq_len, 3])  # Close price
            
        sequences = torch.stack(sequences)
        targets = torch.stack(targets).unsqueeze(1)
        
        # Training step
        loss = attention_learning.train_step(sequences[:32], targets[:32])
        
        assert isinstance(loss, float)
        assert loss > 0
        
    def test_prediction(self, attention_learning, sample_market_data):
        """Test making predictions."""
        processed_data = attention_learning.preprocess_data(sample_market_data)
        
        # Use last 20 timesteps for prediction
        input_sequence = processed_data[-20:].unsqueeze(0)
        
        prediction, attention_weights = attention_learning.predict(input_sequence)
        
        assert prediction.shape == (1, 1)
        assert isinstance(attention_weights, dict)
        
    def test_adaptation(self, attention_learning, sample_market_data):
        """Test model adaptation with new data."""
        initial_weights = attention_learning.model.state_dict()
        
        # Adapt with new data
        attention_learning.adapt(sample_market_data, learning_rate=0.01)
        
        updated_weights = attention_learning.model.state_dict()
        
        # Check that weights have changed
        for key in initial_weights:
            assert not torch.allclose(initial_weights[key], updated_weights[key])
            
    def test_feature_importance_tracking(self, attention_learning, sample_market_data):
        """Test feature importance tracking over time."""
        # Process multiple batches
        for _ in range(5):
            processed_data = attention_learning.preprocess_data(sample_market_data)
            input_sequence = processed_data[-30:].unsqueeze(0)
            attention_learning.predict(input_sequence)
            
        # Get feature importance history
        importance_history = attention_learning.get_feature_importance_history()
        
        assert len(importance_history) == 5
        assert all(imp.shape == (10,) for imp in importance_history)
        
    def test_attention_metrics(self, attention_learning):
        """Test attention metrics calculation."""
        metrics = attention_learning.calculate_metrics()
        
        assert isinstance(metrics, AttentionMetrics)
        assert hasattr(metrics, 'avg_attention_entropy')
        assert hasattr(metrics, 'feature_coverage')
        assert hasattr(metrics, 'temporal_consistency')
        assert hasattr(metrics, 'adaptation_rate')
        
    @pytest.mark.parametrize("market_condition", ["trending", "ranging", "volatile"])
    def test_different_market_conditions(self, attention_learning, market_condition):
        """Test model behavior under different market conditions."""
        # Generate data based on market condition
        if market_condition == "trending":
            prices = np.linspace(100, 150, 100) + np.random.randn(100) * 0.5
        elif market_condition == "ranging":
            prices = 100 + np.sin(np.linspace(0, 10, 100)) * 5 + np.random.randn(100) * 0.5
        else:  # volatile
            prices = 100 + np.random.randn(100).cumsum() * 2
            
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='5min'),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Add technical indicators
        data['rsi'] = 50 + np.random.randn(100) * 10
        data['macd'] = np.random.randn(100)
        
        # Test adaptation
        attention_learning.adapt(data)
        
        # Check that model adapts differently
        metrics = attention_learning.calculate_metrics()
        assert metrics.adaptation_rate > 0
        
    def test_memory_efficiency(self, attention_learning):
        """Test memory efficiency with large sequences."""
        # Create large batch
        large_batch = torch.randn(64, 100, 10)
        
        # Should not raise memory errors
        try:
            with torch.no_grad():
                predictions, _ = attention_learning.model(large_batch)
            assert predictions.shape == (64, 100, 1)
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Insufficient memory for large batch test")
                
    def test_gradient_flow(self, attention_learning):
        """Test gradient flow through attention layers."""
        input_data = torch.randn(16, 20, 10, requires_grad=True)
        
        predictions, attention_maps = attention_learning.model(input_data)
        loss = predictions.mean()
        loss.backward()
        
        # Check gradients exist and are not zero
        assert input_data.grad is not None
        assert not torch.allclose(input_data.grad, torch.zeros_like(input_data.grad))
        
    def test_save_and_load(self, attention_learning, tmp_path):
        """Test model saving and loading."""
        # Save model
        save_path = tmp_path / "attention_model.pth"
        attention_learning.save(save_path)
        
        # Create new instance and load
        new_learning = AttentionLearning(config=attention_learning.config)
        new_learning.load(save_path)
        
        # Compare weights
        old_state = attention_learning.model.state_dict()
        new_state = new_learning.model.state_dict()
        
        for key in old_state:
            assert torch.allclose(old_state[key], new_state[key])
            
    def test_error_handling(self, attention_learning):
        """Test error handling for invalid inputs."""
        # Empty data
        with pytest.raises(ValueError):
            attention_learning.preprocess_data(pd.DataFrame())
            
        # Invalid tensor shapes
        with pytest.raises(ValueError):
            attention_learning.predict(torch.randn(10))  # Wrong dimensions
            
        # NaN values
        data_with_nan = torch.randn(1, 20, 10)
        data_with_nan[0, 10, 5] = float('nan')
        
        with pytest.raises(ValueError):
            attention_learning.predict(data_with_nan)
            
    def test_attention_visualization_data(self, attention_learning, sample_market_data):
        """Test attention visualization data generation."""
        processed_data = attention_learning.preprocess_data(sample_market_data)
        input_sequence = processed_data[-20:].unsqueeze(0)
        
        _, attention_weights = attention_learning.predict(input_sequence)
        
        # Generate visualization data
        viz_data = attention_learning.prepare_attention_visualization(attention_weights)
        
        assert 'feature_attention' in viz_data
        assert 'temporal_attention' in viz_data
        assert 'feature_names' in viz_data
        assert len(viz_data['feature_names']) == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def attention_learning(self):
        """Create an AttentionLearning instance for edge case testing."""
        config = AttentionConfig(
            hidden_dim=32,
            num_heads=2,
            num_features=5,
            max_sequence_length=10
        )
        return AttentionLearning(config=config)
        
    def test_single_sample_batch(self, attention_learning):
        """Test with batch size of 1."""
        single_sample = torch.randn(1, 10, 5)
        predictions, attention = attention_learning.model(single_sample)
        
        assert predictions.shape == (1, 10, 1)
        
    def test_minimum_sequence_length(self, attention_learning):
        """Test with minimum viable sequence length."""
        min_sequence = torch.randn(4, 2, 5)  # Sequence length of 2
        predictions, attention = attention_learning.model(min_sequence)
        
        assert predictions.shape == (4, 2, 1)
        
    def test_maximum_sequence_length(self, attention_learning):
        """Test with maximum sequence length."""
        max_sequence = torch.randn(2, 10, 5)  # Max length from config
        predictions, attention = attention_learning.model(max_sequence)
        
        assert predictions.shape == (2, 10, 1)
        
    def test_zero_attention_weights(self, attention_learning):
        """Test handling of zero attention weights."""
        # Create input that might lead to zero attention
        uniform_input = torch.ones(4, 5, 5) * 0.001
        predictions, attention = attention_learning.model(uniform_input)
        
        # Model should still produce valid outputs
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    @pytest.fixture
    def optimized_learning(self):
        """Create an optimized AttentionLearning instance."""
        config = AttentionConfig(
            hidden_dim=128,
            num_heads=4,
            num_features=15,
            use_flash_attention=True,
            gradient_checkpointing=True
        )
        return AttentionLearning(config=config)
        
    def test_flash_attention(self, optimized_learning):
        """Test flash attention implementation."""
        # Should use optimized attention if available
        input_data = torch.randn(32, 50, 15)
        
        start_time = datetime.now()
        predictions, _ = optimized_learning.model(input_data)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        assert predictions.shape == (32, 50, 1)
        # Flash attention should be faster (this is a soft check)
        assert inference_time < 1.0  # Reasonable threshold
        
    def test_gradient_checkpointing(self, optimized_learning):
        """Test gradient checkpointing for memory efficiency."""
        large_input = torch.randn(16, 100, 15, requires_grad=True)
        
        # Forward and backward pass
        predictions, _ = optimized_learning.model(large_input)
        loss = predictions.mean()
        
        # Should not raise memory errors with checkpointing
        loss.backward()
        
        assert large_input.grad is not None
        
    @pytest.mark.benchmark
    def test_inference_speed(self, benchmark, optimized_learning):
        """Benchmark inference speed."""
        input_data = torch.randn(16, 30, 15)
        
        def run_inference():
            with torch.no_grad():
                return optimized_learning.model(input_data)
                
        result = benchmark(run_inference)
        predictions, _ = result
        
        assert predictions.shape == (16, 30, 1)