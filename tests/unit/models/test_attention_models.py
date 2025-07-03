"""
Unit tests for Attention Models.

Tests cover:
- Multi-head attention mechanisms
- Self-attention layers
- Cross-attention layers
- Positional encoding
- Attention weight visualization
- Model architectures (Transformer, etc.)
- Training and inference
- Gradient flow
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Optional
import math

# GridAttention project imports
from models.attention_models import (
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    PositionalEncoding,
    TransformerBlock,
    MarketTransformer,
    AttentionConfig,
    TemporalAttention,
    FeatureAttention,
    HierarchicalAttention
)


class TestAttentionConfig:
    """Test cases for attention model configuration."""
    
    def test_default_config(self):
        """Test default attention configuration."""
        config = AttentionConfig()
        
        assert config.hidden_dim == 256
        assert config.num_heads == 8
        assert config.dropout_rate == 0.1
        assert config.max_sequence_length == 1000
        assert config.use_relative_position is False
        assert config.attention_type == 'scaled_dot_product'
        
    def test_custom_config(self):
        """Test custom attention configuration."""
        config = AttentionConfig(
            hidden_dim=512,
            num_heads=16,
            dropout_rate=0.2,
            use_relative_position=True,
            attention_type='additive'
        )
        
        assert config.hidden_dim == 512
        assert config.num_heads == 16
        assert config.head_dim == 32  # hidden_dim / num_heads
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Hidden dim must be divisible by num_heads
        with pytest.raises(ValueError, match="hidden_dim must be divisible"):
            AttentionConfig(hidden_dim=250, num_heads=8)
            
        # Invalid dropout rate
        with pytest.raises(ValueError, match="dropout_rate must be"):
            AttentionConfig(dropout_rate=1.5)
            
        # Invalid attention type
        with pytest.raises(ValueError, match="attention_type must be"):
            AttentionConfig(attention_type='invalid')


class TestMultiHeadAttention:
    """Test cases for multi-head attention mechanism."""
    
    @pytest.fixture
    def mha_layer(self):
        """Create multi-head attention layer."""
        config = AttentionConfig(
            hidden_dim=256,
            num_heads=8,
            dropout_rate=0.1
        )
        return MultiHeadAttention(config)
        
    def test_initialization(self, mha_layer):
        """Test proper initialization of MHA layers."""
        assert isinstance(mha_layer.q_linear, nn.Linear)
        assert isinstance(mha_layer.k_linear, nn.Linear)
        assert isinstance(mha_layer.v_linear, nn.Linear)
        assert isinstance(mha_layer.out_linear, nn.Linear)
        
        # Check dimensions
        assert mha_layer.q_linear.in_features == 256
        assert mha_layer.q_linear.out_features == 256
        assert mha_layer.out_linear.in_features == 256
        assert mha_layer.out_linear.out_features == 256
        
    def test_forward_pass(self, mha_layer):
        """Test forward pass through MHA."""
        batch_size = 16
        seq_len = 50
        hidden_dim = 256
        
        # Create input tensors
        query = torch.randn(batch_size, seq_len, hidden_dim)
        key = torch.randn(batch_size, seq_len, hidden_dim)
        value = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Forward pass
        output, attention_weights = mha_layer(query, key, value)
        
        # Check output shapes
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)
        
        # Check attention weights sum to 1
        attention_sum = attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6)
        
    def test_masked_attention(self, mha_layer):
        """Test attention with masking."""
        batch_size = 8
        seq_len = 20
        hidden_dim = 256
        
        # Create inputs
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Create padding mask (last 5 positions are padding)
        mask = torch.ones(batch_size, seq_len)
        mask[:, -5:] = 0
        
        # Forward pass with mask
        output, attention_weights = mha_layer(x, x, x, mask=mask)
        
        # Check masked positions have zero attention
        # Attention weights shape: (batch, heads, seq, seq)
        masked_attention = attention_weights[:, :, :, -5:]
        assert torch.allclose(masked_attention, torch.zeros_like(masked_attention), atol=1e-6)
        
    def test_causal_mask(self, mha_layer):
        """Test causal (look-ahead) masking."""
        batch_size = 4
        seq_len = 10
        hidden_dim = 256
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        output, attention_weights = mha_layer(x, x, x, causal_mask=causal_mask)
        
        # Check that future positions are masked
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(
                    attention_weights[:, :, i, j],
                    torch.zeros_like(attention_weights[:, :, i, j]),
                    atol=1e-6
                )
                
    def test_attention_dropout(self):
        """Test dropout in attention."""
        config = AttentionConfig(
            hidden_dim=128,
            num_heads=4,
            dropout_rate=0.5  # High dropout for testing
        )
        mha = MultiHeadAttention(config)
        
        x = torch.randn(8, 20, 128)
        
        # Training mode
        mha.train()
        output_train1, _ = mha(x, x, x)
        output_train2, _ = mha(x, x, x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2)
        
        # Eval mode
        mha.eval()
        output_eval1, _ = mha(x, x, x)
        output_eval2, _ = mha(x, x, x)
        
        # Outputs should be identical without dropout
        assert torch.allclose(output_eval1, output_eval2)


class TestSelfAttention:
    """Test cases for self-attention layer."""
    
    @pytest.fixture
    def self_attention(self):
        """Create self-attention layer."""
        return SelfAttention(
            input_dim=128,
            hidden_dim=256,
            num_heads=8
        )
        
    def test_self_attention_forward(self, self_attention):
        """Test self-attention forward pass."""
        batch_size = 16
        seq_len = 30
        input_dim = 128
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output, attention_weights = self_attention(x)
        
        # Output should have hidden_dim
        assert output.shape == (batch_size, seq_len, 256)
        
        # Self-attention should be square
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)
        
    def test_self_attention_with_projection(self):
        """Test self-attention with input projection."""
        self_att = SelfAttention(
            input_dim=64,
            hidden_dim=128,
            num_heads=4,
            project_input=True
        )
        
        x = torch.randn(8, 20, 64)
        output, _ = self_att(x)
        
        assert output.shape == (8, 20, 128)


class TestCrossAttention:
    """Test cases for cross-attention layer."""
    
    @pytest.fixture
    def cross_attention(self):
        """Create cross-attention layer."""
        return CrossAttention(
            query_dim=128,
            key_value_dim=256,
            hidden_dim=128,
            num_heads=4
        )
        
    def test_cross_attention_forward(self, cross_attention):
        """Test cross-attention forward pass."""
        batch_size = 8
        query_len = 20
        kv_len = 30
        
        query = torch.randn(batch_size, query_len, 128)
        key_value = torch.randn(batch_size, kv_len, 256)
        
        output, attention_weights = cross_attention(query, key_value)
        
        # Output shape matches query
        assert output.shape == (batch_size, query_len, 128)
        
        # Attention maps from query to key/value
        assert attention_weights.shape == (batch_size, 4, query_len, kv_len)
        
    def test_cross_attention_different_lengths(self, cross_attention):
        """Test cross-attention with different sequence lengths."""
        # Short query attending to long context
        query = torch.randn(4, 5, 128)  # 5 query positions
        context = torch.randn(4, 100, 256)  # 100 context positions
        
        output, attention_weights = cross_attention(query, context)
        
        assert output.shape == (4, 5, 128)
        assert attention_weights.shape == (4, 4, 5, 100)
        
        # Each query position should attend to all context
        attention_sum = attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6)


class TestPositionalEncoding:
    """Test cases for positional encoding."""
    
    @pytest.fixture
    def pos_encoding(self):
        """Create positional encoding layer."""
        return PositionalEncoding(
            d_model=256,
            max_len=1000,
            dropout=0.1
        )
        
    def test_encoding_shape(self, pos_encoding):
        """Test positional encoding output shape."""
        batch_size = 16
        seq_len = 50
        d_model = 256
        
        x = torch.randn(batch_size, seq_len, d_model)
        encoded = pos_encoding(x)
        
        # Shape should be preserved
        assert encoded.shape == x.shape
        
    def test_encoding_values(self, pos_encoding):
        """Test positional encoding values."""
        # Check specific positions
        seq_len = 100
        d_model = 256
        
        # Get encoding for position 0
        pos_0 = pos_encoding.pe[0, 0, :]
        
        # First dimension should be sin(0) = 0
        assert abs(pos_0[0].item()) < 1e-6
        
        # Second dimension should be cos(0) = 1
        assert abs(pos_0[1].item() - 1.0) < 1e-6
        
    def test_relative_positional_encoding(self):
        """Test relative positional encoding."""
        rel_pos = PositionalEncoding(
            d_model=128,
            max_len=100,
            use_relative=True
        )
        
        batch_size = 8
        seq_len = 20
        d_model = 128
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Get relative positions
        rel_positions = rel_pos.get_relative_positions(seq_len)
        
        assert rel_positions.shape == (seq_len, seq_len)
        
        # Diagonal should be 0 (same position)
        assert torch.allclose(torch.diag(rel_positions), torch.zeros(seq_len))
        
    def test_learnable_positional_encoding(self):
        """Test learnable positional embeddings."""
        learn_pos = PositionalEncoding(
            d_model=128,
            max_len=100,
            learnable=True
        )
        
        # Check that embeddings are parameters
        assert hasattr(learn_pos, 'pos_embeddings')
        assert isinstance(learn_pos.pos_embeddings, nn.Parameter)
        
        # Test forward pass
        x = torch.randn(4, 50, 128)
        encoded = learn_pos(x)
        
        assert encoded.shape == x.shape


class TestTransformerBlock:
    """Test cases for transformer block."""
    
    @pytest.fixture
    def transformer_block(self):
        """Create transformer block."""
        config = AttentionConfig(
            hidden_dim=256,
            num_heads=8,
            dropout_rate=0.1
        )
        return TransformerBlock(config)
        
    def test_block_forward(self, transformer_block):
        """Test transformer block forward pass."""
        batch_size = 16
        seq_len = 30
        hidden_dim = 256
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        output, attention_weights = transformer_block(x)
        
        # Check output shape preserved
        assert output.shape == x.shape
        
        # Check residual connections work
        assert not torch.allclose(output, x)  # Should be transformed
        
    def test_block_components(self, transformer_block):
        """Test transformer block components."""
        # Check sub-layers exist
        assert hasattr(transformer_block, 'attention')
        assert hasattr(transformer_block, 'feed_forward')
        assert hasattr(transformer_block, 'norm1')
        assert hasattr(transformer_block, 'norm2')
        
        # Check feed-forward dimensions
        assert transformer_block.feed_forward[0].in_features == 256
        assert transformer_block.feed_forward[0].out_features == 1024  # 4x hidden
        assert transformer_block.feed_forward[2].out_features == 256
        
    def test_gradient_flow(self, transformer_block):
        """Test gradient flow through transformer block."""
        x = torch.randn(8, 20, 256, requires_grad=True)
        
        output, _ = transformer_block(x)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist and are not zero
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
        # Check no gradient explosion
        assert x.grad.abs().max() < 10.0


class TestMarketTransformer:
    """Test cases for market-specific transformer model."""
    
    @pytest.fixture
    def market_config(self):
        """Create market transformer configuration."""
        return {
            'input_dim': 10,  # OHLCV + indicators
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 3,
            'dropout_rate': 0.1,
            'max_seq_length': 100
        }
        
    @pytest.fixture
    def market_transformer(self, market_config):
        """Create market transformer model."""
        return MarketTransformer(**market_config)
        
    def test_model_initialization(self, market_transformer):
        """Test market transformer initialization."""
        assert market_transformer.input_projection.in_features == 10
        assert market_transformer.input_projection.out_features == 128
        assert len(market_transformer.transformer_blocks) == 3
        assert market_transformer.output_projection.out_features == 1  # Price prediction
        
    def test_forward_pass(self, market_transformer):
        """Test forward pass through market transformer."""
        batch_size = 8
        seq_len = 50
        input_dim = 10
        
        # Create market data input
        market_data = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        predictions, attention_maps = market_transformer(market_data)
        
        # Check predictions
        assert predictions.shape == (batch_size, seq_len, 1)
        
        # Check attention maps from all layers
        assert len(attention_maps) == 3  # One per layer
        assert all(
            attn.shape == (batch_size, 4, seq_len, seq_len)
            for attn in attention_maps
        )
        
    def test_causal_prediction(self, market_transformer):
        """Test causal (autoregressive) prediction."""
        market_transformer.set_causal(True)
        
        batch_size = 4
        seq_len = 20
        
        market_data = torch.randn(batch_size, seq_len, 10)
        
        predictions, attention_maps = market_transformer(market_data)
        
        # Check causal masking in attention
        for layer_attention in attention_maps:
            # Average over batch and heads
            avg_attention = layer_attention.mean(dim=(0, 1))
            
            # Upper triangle should be near zero (future masked)
            upper_triangle = torch.triu(avg_attention, diagonal=1)
            assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)
            
    def test_feature_importance(self, market_transformer):
        """Test feature importance extraction."""
        batch_size = 16
        seq_len = 30
        
        market_data = torch.randn(batch_size, seq_len, 10)
        
        _, attention_maps = market_transformer(market_data)
        
        # Get feature importance scores
        feature_importance = market_transformer.get_feature_importance(
            market_data,
            attention_maps
        )
        
        assert feature_importance.shape == (10,)  # One score per input feature
        assert torch.allclose(feature_importance.sum(), torch.tensor(1.0))  # Normalized
        
    def test_training_step(self, market_transformer):
        """Test training step with loss calculation."""
        batch_size = 8
        seq_len = 50
        
        # Create training data
        market_data = torch.randn(batch_size, seq_len, 10)
        target_prices = torch.randn(batch_size, seq_len, 1)
        
        # Setup optimizer
        optimizer = optim.Adam(market_transformer.parameters(), lr=0.001)
        
        # Training step
        market_transformer.train()
        predictions, _ = market_transformer(market_data)
        
        # Calculate loss
        loss = nn.MSELoss()(predictions, target_prices)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check loss is reasonable
        assert loss.item() > 0
        assert loss.item() < 100  # Not exploding


class TestTemporalAttention:
    """Test cases for temporal attention mechanisms."""
    
    @pytest.fixture
    def temporal_attention(self):
        """Create temporal attention layer."""
        return TemporalAttention(
            hidden_dim=128,
            num_heads=4,
            window_size=20,
            use_decay=True
        )
        
    def test_temporal_attention_forward(self, temporal_attention):
        """Test temporal attention forward pass."""
        batch_size = 8
        seq_len = 50
        hidden_dim = 128
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        output, attention_weights = temporal_attention(x)
        
        assert output.shape == x.shape
        
        # Check window constraint
        # Attention should be limited to window_size
        assert attention_weights.shape == (batch_size, 4, seq_len, seq_len)
        
    def test_temporal_decay(self, temporal_attention):
        """Test temporal decay in attention weights."""
        batch_size = 4
        seq_len = 30
        
        x = torch.randn(batch_size, seq_len, 128)
        
        _, attention_weights = temporal_attention(x)
        
        # Average attention weights
        avg_attention = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        
        # Check that recent positions have higher attention on average
        for i in range(seq_len - 1):
            recent_avg = avg_attention[i, max(0, i-5):i+1].mean()
            distant_avg = avg_attention[i, max(0, i-20):max(0, i-10)].mean() if i > 20 else 0
            
            if i > 20:
                assert recent_avg > distant_avg  # Recent should have more weight
                
    def test_multi_scale_temporal(self):
        """Test multi-scale temporal attention."""
        multi_temporal = TemporalAttention(
            hidden_dim=128,
            num_heads=4,
            window_sizes=[5, 10, 20],  # Multiple scales
            use_multi_scale=True
        )
        
        x = torch.randn(8, 30, 128)
        
        output, attention_weights = multi_temporal(x)
        
        # Should have attention at multiple scales
        assert isinstance(attention_weights, dict)
        assert len(attention_weights) == 3  # One per scale
        
        for scale, attn in attention_weights.items():
            assert attn.shape == (8, 4, 30, 30)


class TestFeatureAttention:
    """Test cases for feature attention mechanisms."""
    
    @pytest.fixture
    def feature_attention(self):
        """Create feature attention layer."""
        return FeatureAttention(
            num_features=10,
            hidden_dim=64,
            temperature=1.0
        )
        
    def test_feature_attention_forward(self, feature_attention):
        """Test feature attention forward pass."""
        batch_size = 16
        seq_len = 20
        num_features = 10
        
        # Input shape: (batch, seq, features)
        x = torch.randn(batch_size, seq_len, num_features)
        
        output, feature_weights = feature_attention(x)
        
        # Output should preserve shape
        assert output.shape == x.shape
        
        # Feature weights should be (batch, features)
        assert feature_weights.shape == (batch_size, num_features)
        
        # Weights should sum to 1
        assert torch.allclose(feature_weights.sum(dim=1), torch.ones(batch_size))
        
    def test_feature_selection(self, feature_attention):
        """Test feature selection based on importance."""
        batch_size = 8
        seq_len = 30
        num_features = 20
        
        # Create data where some features are more important
        x = torch.randn(batch_size, seq_len, num_features)
        
        # Make first 5 features have larger magnitude
        x[:, :, :5] *= 10
        
        _, feature_weights = feature_attention(x)
        
        # Average weights across batch
        avg_weights = feature_weights.mean(dim=0)
        
        # First 5 features should have higher weights
        top_5_avg = avg_weights[:5].mean()
        rest_avg = avg_weights[5:].mean()
        
        assert top_5_avg > rest_avg
        
    def test_temperature_scaling(self):
        """Test temperature scaling in feature attention."""
        # High temperature (uniform attention)
        high_temp = FeatureAttention(
            num_features=10,
            hidden_dim=64,
            temperature=10.0
        )
        
        # Low temperature (sharp attention)
        low_temp = FeatureAttention(
            num_features=10,
            hidden_dim=64,
            temperature=0.1
        )
        
        x = torch.randn(8, 20, 10)
        
        _, weights_high = high_temp(x)
        _, weights_low = low_temp(x)
        
        # High temperature should give more uniform weights
        std_high = weights_high.std(dim=1).mean()
        std_low = weights_low.std(dim=1).mean()
        
        assert std_high < std_low  # Lower variance with high temperature


class TestHierarchicalAttention:
    """Test cases for hierarchical attention."""
    
    @pytest.fixture
    def hierarchical_attention(self):
        """Create hierarchical attention model."""
        return HierarchicalAttention(
            input_dim=64,
            hidden_dims=[128, 256, 512],
            num_heads=[4, 8, 16],
            pool_sizes=[2, 2, 2]
        )
        
    def test_hierarchical_forward(self, hierarchical_attention):
        """Test hierarchical attention forward pass."""
        batch_size = 8
        seq_len = 64  # Must be divisible by product of pool_sizes
        input_dim = 64
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        outputs, attention_maps = hierarchical_attention(x)
        
        # Check outputs at each level
        assert len(outputs) == 3
        assert len(attention_maps) == 3
        
        # Check progressive pooling
        assert outputs[0].shape == (batch_size, 32, 128)  # seq_len/2
        assert outputs[1].shape == (batch_size, 16, 256)  # seq_len/4
        assert outputs[2].shape == (batch_size, 8, 512)   # seq_len/8
        
    def test_multi_resolution_features(self, hierarchical_attention):
        """Test multi-resolution feature extraction."""
        batch_size = 4
        seq_len = 32
        
        x = torch.randn(batch_size, seq_len, 64)
        
        outputs, _ = hierarchical_attention(x)
        
        # Combine features from all levels
        combined_features = hierarchical_attention.combine_hierarchical_features(outputs)
        
        # Should concatenate features from all levels
        # Upsampled to original sequence length
        expected_dim = 128 + 256 + 512
        assert combined_features.shape == (batch_size, seq_len, expected_dim)
        
    def test_hierarchical_importance(self, hierarchical_attention):
        """Test importance scores at different hierarchical levels."""
        x = torch.randn(4, 32, 64)
        
        _, attention_maps = hierarchical_attention(x)
        
        # Get importance at each level
        importance_scores = hierarchical_attention.get_hierarchical_importance(
            attention_maps
        )
        
        assert len(importance_scores) == 3
        
        # Higher levels should have coarser importance maps
        assert importance_scores[0].shape == (32,)  # Fine-grained
        assert importance_scores[1].shape == (16,)  # Medium
        assert importance_scores[2].shape == (8,)   # Coarse


class TestAttentionVisualization:
    """Test cases for attention visualization utilities."""
    
    def test_attention_matrix_preparation(self):
        """Test preparation of attention matrices for visualization."""
        batch_size = 4
        num_heads = 8
        seq_len = 20
        
        # Create attention weights
        attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Prepare for visualization
        from src.models.attention_models import prepare_attention_for_viz
        
        # Average over batch
        viz_data = prepare_attention_for_viz(
            attention_weights,
            aggregate='mean',
            select_heads=[0, 1, 2]  # Select specific heads
        )
        
        assert viz_data.shape == (3, seq_len, seq_len)  # 3 selected heads
        
        # Max aggregation
        viz_data_max = prepare_attention_for_viz(
            attention_weights,
            aggregate='max'
        )
        
        assert viz_data_max.shape == (num_heads, seq_len, seq_len)
        
    def test_attention_pattern_analysis(self):
        """Test analysis of attention patterns."""
        # Create specific attention patterns
        seq_len = 30
        
        # Diagonal attention (attending to self)
        diagonal_attention = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)
        
        # Uniform attention
        uniform_attention = torch.ones(1, 1, seq_len, seq_len) / seq_len
        
        # Local attention (attending to nearby positions)
        local_attention = torch.zeros(1, 1, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                local_attention[0, 0, i, j] = 1.0
        local_attention = local_attention / local_attention.sum(dim=-1, keepdim=True)
        
        from src.models.attention_models import analyze_attention_pattern
        
        # Analyze patterns
        diagonal_stats = analyze_attention_pattern(diagonal_attention)
        assert diagonal_stats['pattern_type'] == 'diagonal'
        assert diagonal_stats['entropy'] < 0.1  # Low entropy
        
        uniform_stats = analyze_attention_pattern(uniform_attention)
        assert uniform_stats['pattern_type'] == 'uniform'
        assert uniform_stats['entropy'] > 3.0  # High entropy
        
        local_stats = analyze_attention_pattern(local_attention)
        assert local_stats['pattern_type'] == 'local'
        assert 0.5 < local_stats['entropy'] < 2.0  # Medium entropy


class TestAttentionRegularization:
    """Test cases for attention regularization techniques."""
    
    def test_attention_dropout(self):
        """Test structured dropout for attention."""
        from src.models.attention_models import AttentionDropout
        
        dropout = AttentionDropout(
            drop_rate=0.2,
            drop_type='structured'  # Drop entire heads
        )
        
        attention_weights = torch.randn(8, 4, 20, 20)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Training mode
        dropout.train()
        dropped = dropout(attention_weights)
        
        # Some heads should be completely zeroed
        head_sums = dropped.sum(dim=(2, 3))  # Sum over attention matrix
        zero_heads = (head_sums == 0).sum()
        
        assert zero_heads > 0  # At least some heads dropped
        
    def test_attention_regularization_loss(self):
        """Test regularization losses for attention."""
        from src.models.attention_models import attention_regularization_loss
        
        batch_size = 8
        num_heads = 4
        seq_len = 20
        
        attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Entropy regularization (encourage diverse attention)
        entropy_loss = attention_regularization_loss(
            attention_weights,
            reg_type='entropy',
            target_entropy=2.0
        )
        
        assert entropy_loss.item() >= 0
        
        # Sparsity regularization (encourage focused attention)
        sparsity_loss = attention_regularization_loss(
            attention_weights,
            reg_type='sparsity',
            sparsity_target=0.1
        )
        
        assert sparsity_loss.item() >= 0
        
    def test_attention_constraints(self):
        """Test constraints on attention patterns."""
        from src.models.attention_models import apply_attention_constraints
        
        seq_len = 30
        attention_weights = torch.randn(4, 8, seq_len, seq_len)
        
        # Monotonic attention (can only attend to later positions)
        monotonic = apply_attention_constraints(
            attention_weights,
            constraint_type='monotonic'
        )
        
        # Check monotonicity
        for i in range(seq_len - 1):
            assert (monotonic[:, :, :, i+1] >= monotonic[:, :, :, i]).all()
            
        # Local attention window
        local = apply_attention_constraints(
            attention_weights,
            constraint_type='local',
            window_size=5
        )
        
        # Check positions outside window are zero
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > 5:
                    assert torch.allclose(
                        local[:, :, i, j],
                        torch.zeros_like(local[:, :, i, j])
                    )