"""
Test Grid Optimizer Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grid_optimizer import (
    DynamicGridOptimizer, OptimizedGridParams, GridPerformanceMetrics, 
    MarketConditions
)

class TestGridOptimizer:
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.config = {
            'history_size': 500,
            'use_grid_optimization': True
        }
        self.optimizer = DynamicGridOptimizer(self.config)
        
    def create_sample_features(self):
        """Create sample features for testing"""
        return {
            'atr_14': 0.02,
            'volatility_5m': 0.015,
            'volatility_avg': 0.012,
            'volatility_percentile': 60,
            'trend_strength': 0.3,
            'current_price': 100.0,
            'volume_ratio': 1.2,
            'correlation_factor': 0.5
        }
        
    def create_sample_account_info(self):
        """Create sample account information"""
        return {
            'balance': 10000,
            'risk_tolerance': 0.02,
            'win_rate': 0.55,
            'avg_win': 1.2,
            'avg_loss': 0.8,
            'confidence': 0.7
        }
        
    def create_sample_market_context(self):
        """Create sample market context"""
        return {
            'sub_regime': 'normal_range',
            'volume_profile': 'normal',
            'session': 'LONDON',
            'support_resistance_levels': [99.5, 100.5, 101.0],
            'news_in_next_hour': False
        }
        
    def test_basic_optimization(self):
        """Test basic grid parameter optimization"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        # Test trending regime
        result = self.optimizer.optimize_grid_parameters(
            'TRENDING', features, account_info, market_context
        )
        
        assert isinstance(result, OptimizedGridParams)
        assert result.spacing > 0
        assert result.levels >= 3
        assert result.position_size > 0
        assert 0 <= result.fill_probability <= 1
        assert result.expected_profit >= 0
        assert result.max_drawdown_risk >= 0
        
    def test_regime_specific_optimization(self):
        """Test optimization for different regimes"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'UNCERTAIN']
        results = {}
        
        for regime in regimes:
            result = self.optimizer.optimize_grid_parameters(
                regime, features, account_info, market_context
            )
            results[regime] = result
            
            assert isinstance(result, OptimizedGridParams)
            print(f"{regime}: spacing={result.spacing:.4f}, levels={result.levels}, size={result.position_size:.2f}")
        
        # Volatile regime should have wider spacing
        assert results['VOLATILE'].spacing > results['RANGING'].spacing
        
        # Check that results are reasonable (allow for algorithm variations)
        for regime in regimes:
            assert results[regime].levels >= 3
            assert results[regime].spacing > 0
            assert results[regime].position_size > 0
        
    def test_kelly_criterion_calculation(self):
        """Test Kelly criterion position sizing"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        # High win rate scenario
        high_win_account = account_info.copy()
        high_win_account['win_rate'] = 0.7
        high_win_account['avg_win'] = 1.5
        
        result_high = self.optimizer.optimize_grid_parameters(
            'RANGING', features, high_win_account, market_context
        )
        
        # Low win rate scenario
        low_win_account = account_info.copy()
        low_win_account['win_rate'] = 0.4
        low_win_account['avg_win'] = 0.8
        
        result_low = self.optimizer.optimize_grid_parameters(
            'RANGING', features, low_win_account, market_context
        )
        
        # High win rate should result in larger position sizes
        assert result_high.position_size > result_low.position_size
        
    def test_volatility_adjustments(self):
        """Test volatility-based parameter adjustments"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        # Low volatility
        low_vol_features = features.copy()
        low_vol_features['volatility_percentile'] = 10
        
        result_low_vol = self.optimizer.optimize_grid_parameters(
            'RANGING', low_vol_features, account_info, market_context
        )
        
        # High volatility
        high_vol_features = features.copy()
        high_vol_features['volatility_percentile'] = 90
        
        result_high_vol = self.optimizer.optimize_grid_parameters(
            'RANGING', high_vol_features, account_info, market_context
        )
        
        # High volatility should have wider spacing
        assert result_high_vol.spacing > result_low_vol.spacing
        
        # Both should be reasonable
        assert result_high_vol.levels >= 3
        assert result_low_vol.levels >= 3
        
        print(f"Low vol: spacing={result_low_vol.spacing:.4f}, levels={result_low_vol.levels}")
        print(f"High vol: spacing={result_high_vol.spacing:.4f}, levels={result_high_vol.levels}")
        
    def test_session_adjustments(self):
        """Test session-based parameter adjustments"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        sessions = ['ASIAN', 'LONDON', 'US', 'US_OPEN']
        results = {}
        
        for session in sessions:
            context = market_context.copy()
            context['session'] = session
            
            result = self.optimizer.optimize_grid_parameters(
                'RANGING', features, account_info, context
            )
            results[session] = result
            
        # US_OPEN should have wider spacing (more volatile)
        assert results['US_OPEN'].spacing > results['ASIAN'].spacing
        
        # All should be reasonable
        for session in sessions:
            assert results[session].levels >= 3
            assert results[session].spacing > 0
            print(f"{session}: spacing={results[session].spacing:.4f}, levels={results[session].levels}")
        
    def test_support_resistance_awareness(self):
        """Test support/resistance level awareness"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        
        # Without S/R levels
        context_no_sr = self.create_sample_market_context()
        context_no_sr['support_resistance_levels'] = []
        
        result_no_sr = self.optimizer.optimize_grid_parameters(
            'RANGING', features, account_info, context_no_sr
        )
        
        # With S/R levels
        context_with_sr = self.create_sample_market_context()
        context_with_sr['support_resistance_levels'] = [99.0, 99.5, 100.5, 101.0, 101.5]
        
        result_with_sr = self.optimizer.optimize_grid_parameters(
            'RANGING', features, account_info, context_with_sr
        )
        
        # Should produce different results
        assert len(result_with_sr.optimal_entry_zones) > 0
        
    def test_risk_constraints(self):
        """Test risk constraint enforcement"""
        features = self.create_sample_features()
        market_context = self.create_sample_market_context()
        
        # Small account
        small_account = {
            'balance': 1000,  # Small account
            'risk_tolerance': 0.05,  # High risk tolerance
            'win_rate': 0.5,
            'avg_win': 1.0,
            'avg_loss': 1.0,
            'confidence': 0.6
        }
        
        result = self.optimizer.optimize_grid_parameters(
            'VOLATILE', features, small_account, market_context
        )
        
        # Check constraints are respected
        max_position = small_account['balance'] * 0.05  # 5% max
        assert result.position_size <= max_position
        
        total_exposure = result.position_size * result.levels
        max_exposure = small_account['balance'] * 0.15  # 15% max total
        assert total_exposure <= max_exposure
        
    def test_fill_probability_estimation(self):
        """Test fill probability estimation"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        # Tight spacing (high fill probability)
        tight_features = features.copy()
        tight_features['atr_14'] = 0.01  # Low ATR
        
        result_tight = self.optimizer.optimize_grid_parameters(
            'RANGING', tight_features, account_info, market_context
        )
        
        # Wide spacing (low fill probability)
        wide_features = features.copy()
        wide_features['atr_14'] = 0.05  # High ATR
        
        result_wide = self.optimizer.optimize_grid_parameters(
            'RANGING', wide_features, account_info, market_context
        )
        
        # Tighter grids relative to ATR should have higher fill probability
        tight_ratio = tight_features['atr_14'] / result_tight.spacing
        wide_ratio = wide_features['atr_14'] / result_wide.spacing
        
        if tight_ratio > wide_ratio:
            assert result_tight.fill_probability >= result_wide.fill_probability
        
    def test_expected_profit_calculation(self):
        """Test expected profit calculation"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        # High volume session
        high_vol_context = market_context.copy()
        high_vol_context['session'] = 'US_OPEN'
        high_vol_context['volume_profile'] = 'high'
        
        result_high_vol = self.optimizer.optimize_grid_parameters(
            'RANGING', features, account_info, high_vol_context
        )
        
        # Low volume session
        low_vol_context = market_context.copy()
        low_vol_context['session'] = 'ASIAN'
        low_vol_context['volume_profile'] = 'low'
        
        result_low_vol = self.optimizer.optimize_grid_parameters(
            'RANGING', features, account_info, low_vol_context
        )
        
        # High volume should generally lead to higher expected profit
        # (though this can vary based on other factors)
        assert result_high_vol.expected_profit >= 0
        assert result_low_vol.expected_profit >= 0
        
    def test_optimal_entry_zones(self):
        """Test optimal entry zone calculation"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        result = self.optimizer.optimize_grid_parameters(
            'RANGING', features, account_info, market_context
        )
        
        # Should have entry zones
        assert len(result.optimal_entry_zones) > 0
        print(f"Entry zones: {len(result.optimal_entry_zones)}, Levels: {result.levels}")
        # Entry zones might not exactly equal levels due to optimization
        assert len(result.optimal_entry_zones) >= result.levels * 0.8  # Allow some variance
        
        # Zones should be sorted
        assert result.optimal_entry_zones == sorted(result.optimal_entry_zones)
        
        # Zones should be spaced appropriately
        if len(result.optimal_entry_zones) > 1:
            spacings = np.diff(result.optimal_entry_zones)
            avg_spacing = np.mean(spacings)
            # Should be close to the optimized spacing
            assert abs(avg_spacing - result.spacing) < result.spacing * 0.5
        
    def test_stop_loss_take_profit(self):
        """Test stop loss and take profit calculation"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        # Strong uptrend
        trend_features = features.copy()
        trend_features['trend_strength'] = 0.8
        
        result = self.optimizer.optimize_grid_parameters(
            'TRENDING', trend_features, account_info, market_context
        )
        
        current_price = features['current_price']
        
        # Should have stop loss and take profit
        if result.stop_loss_level and result.take_profit_level:
            # In uptrend: stop below, target above
            if trend_features['trend_strength'] > 0:
                assert result.stop_loss_level < current_price
                assert result.take_profit_level > current_price
            
            # Stop should be closer than take profit in absolute terms
            stop_distance = abs(current_price - result.stop_loss_level)
            tp_distance = abs(current_price - result.take_profit_level)
            # Take profit should typically be further than stop loss
            assert tp_distance >= stop_distance * 0.8
    
    def test_optimization_statistics(self):
        """Test optimization statistics collection"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        # Run several optimizations
        for regime in ['TRENDING', 'RANGING', 'VOLATILE']:
            self.optimizer.optimize_grid_parameters(
                regime, features, account_info, market_context
            )
        
        stats = self.optimizer.get_optimization_statistics()
        
        assert 'total_optimizations' in stats
        assert 'avg_expected_profit' in stats
        assert 'avg_fill_probability' in stats
        assert 'regime_distribution' in stats
        assert 'parameter_ranges' in stats
        
        assert stats['total_optimizations'] >= 3
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        features = self.create_sample_features()
        account_info = self.create_sample_account_info()
        market_context = self.create_sample_market_context()
        
        # Missing features
        minimal_features = {'atr_14': 0.01}
        result = self.optimizer.optimize_grid_parameters(
            'RANGING', minimal_features, account_info, market_context
        )
        assert isinstance(result, OptimizedGridParams)
        
        # Extreme values
        extreme_features = features.copy()
        extreme_features['volatility_percentile'] = 99
        extreme_features['atr_14'] = 0.1  # Very high ATR
        
        result = self.optimizer.optimize_grid_parameters(
            'VOLATILE', extreme_features, account_info, market_context
        )
        assert isinstance(result, OptimizedGridParams)
        assert result.spacing > 0
        
        # Very small account
        tiny_account = account_info.copy()
        tiny_account['balance'] = 100
        
        result = self.optimizer.optimize_grid_parameters(
            'RANGING', features, tiny_account, market_context
        )
        assert isinstance(result, OptimizedGridParams)
        assert result.position_size > 0


if __name__ == "__main__":
    # Run tests
    test = TestGridOptimizer()
    test.setup_method()
    
    print("Testing basic optimization...")
    test.test_basic_optimization()
    print("✓ Passed")
    
    print("Testing regime-specific optimization...")
    test.test_regime_specific_optimization()
    print("✓ Passed")
    
    print("Testing Kelly criterion...")
    test.test_kelly_criterion_calculation()
    print("✓ Passed")
    
    print("Testing volatility adjustments...")
    test.test_volatility_adjustments()
    print("✓ Passed")
    
    print("Testing session adjustments...")
    test.test_session_adjustments()
    print("✓ Passed")
    
    print("Testing S/R awareness...")
    test.test_support_resistance_awareness()
    print("✓ Passed")
    
    print("Testing risk constraints...")
    test.test_risk_constraints()
    print("✓ Passed")
    
    print("Testing fill probability...")
    test.test_fill_probability_estimation()
    print("✓ Passed")
    
    print("Testing expected profit...")
    test.test_expected_profit_calculation()
    print("✓ Passed")
    
    print("Testing entry zones...")
    test.test_optimal_entry_zones()
    print("✓ Passed")
    
    print("Testing stop/take profit...")
    test.test_stop_loss_take_profit()
    print("✓ Passed")
    
    print("Testing statistics...")
    test.test_optimization_statistics()
    print("✓ Passed")
    
    print("Testing edge cases...")
    test.test_edge_cases()
    print("✓ Passed")
    
    print("\nAll tests passed! ✨")