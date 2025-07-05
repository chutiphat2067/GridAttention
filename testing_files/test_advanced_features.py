"""
Test Advanced Features Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.advanced_features import AdvancedFeatureEngineer, MicrostructureFeatures, MultiTimeframeFeatures

class TestAdvancedFeatures:
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.config = {
            'use_advanced_features': True,
            'feature_cache_size': 1000
        }
        self.engineer = AdvancedFeatureEngineer(self.config)
        
    def create_sample_data(self, n_rows=100):
        """Create sample market data for testing"""
        np.random.seed(42)
        
        # Generate price data
        price = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)
        
        # Generate volume data
        volume = np.random.exponential(1000, n_rows)
        
        # Generate bid/ask data
        spread = np.random.uniform(0.01, 0.05, n_rows)
        bid_price = price - spread/2
        ask_price = price + spread/2
        
        bid_volume = np.random.exponential(500, n_rows)
        ask_volume = np.random.exponential(500, n_rows)
        
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='1min'),
            'open': price,
            'high': price + np.random.uniform(0, 0.1, n_rows),
            'low': price - np.random.uniform(0, 0.1, n_rows),
            'close': price,
            'volume': volume,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_price_1': bid_price,
            'ask_price_1': ask_price,
            'bid_size_1': bid_volume,
            'ask_size_1': ask_volume
        })
        
        return data
        
    def test_market_microstructure(self):
        """Test microstructure feature calculation"""
        data = self.create_sample_data()
        features = self.engineer.calculate_market_microstructure(data)
        
        # Check all expected features exist
        assert 'bid_ask_imbalance' in features
        assert 'bid_ask_imbalance_ma' in features
        assert 'order_flow_toxicity' in features
        assert 'price_acceleration' in features
        assert 'price_jerk' in features
        assert 'volume_weighted_spread' in features
        assert 'trade_intensity' in features
        assert 'price_impact' in features
        assert 'liquidity_consumption' in features
        assert 'order_book_slope' in features
        
        # Check value ranges
        assert -1 <= features['bid_ask_imbalance'] <= 1
        assert 0 <= features['order_flow_toxicity'] <= 1
        assert features['volume_weighted_spread'] > 0
        assert features['liquidity_consumption'] >= 0
        
    def test_multi_timeframe_features(self):
        """Test multi-timeframe feature calculation"""
        data = self.create_sample_data(500)  # Need more data for longer timeframes
        features = self.engineer.calculate_multi_timeframe_features(data)
        
        # Check trend alignment features
        assert 'trend_alignment_1m_5m' in features
        assert 'trend_alignment_5m_15m' in features
        assert 'trend_alignment_15m_1h' in features
        assert 'trend_alignment_1h_4h' in features
        assert 'overall_trend_alignment' in features
        
        # Check other MTF features
        assert 'momentum_divergence' in features
        assert 'vol_ratio_5_20' in features
        assert 'vol_ratio_20_100' in features
        assert 'vol_expansion' in features
        assert 'regime_consistency' in features
        
        # Check timeframe strengths
        for tf in ['1m', '5m', '15m', '1h']:
            assert f'strength_{tf}' in features
            
        # Check value ranges
        assert -1 <= features['overall_trend_alignment'] <= 1
        assert 0 <= features['regime_consistency'] <= 1
        
    def test_volume_profile_features(self):
        """Test volume profile feature calculation"""
        data = self.create_sample_data()
        features = self.engineer.calculate_volume_profile(data)
        
        # Check volume profile features
        assert 'poc_price' in features
        assert 'poc_distance' in features
        assert 'value_area_high' in features
        assert 'value_area_low' in features
        assert 'in_value_area' in features
        assert 'large_trade_ratio' in features
        assert 'volume_momentum' in features
        assert 'volume_concentration' in features
        
        # Check logical constraints
        assert features['value_area_low'] <= features['poc_price'] <= features['value_area_high']
        assert features['in_value_area'] in [0, 1]
        assert 0 <= features['large_trade_ratio'] <= 1
        assert features['volume_momentum'] > 0
        assert 0 <= features['volume_concentration'] <= 1
        
    def test_volatility_regime_features(self):
        """Test volatility regime feature calculation"""
        data = self.create_sample_data(2500)  # Need long history for volatility
        features = self.engineer.calculate_volatility_regime(data)
        
        # Check volatility features exist (some may not if data insufficient)
        vol_features = ['realized_vol_micro', 'realized_vol_short', 'realized_vol_medium', 'realized_vol_long']
        vol_features_found = [f for f in vol_features if f in features]
        assert len(vol_features_found) >= 2  # At least 2 volatility measures
        
        assert 'garch_forecast' in features
        assert 'volatility_regime' in features
        
        # vol_of_vol may not exist if insufficient data
        if 'vol_of_vol' in features:
            assert features['vol_of_vol'] >= 0
        
        # Check volatility is positive
        for key in features:
            if 'vol' in key and key != 'volatility_regime' and isinstance(features[key], (int, float)):
                assert features[key] >= 0
                
        # Check regime classification
        assert features['volatility_regime'] in ['expanding', 'contracting', 'high', 'low', 'normal', 'unknown']
        
    def test_market_sentiment_features(self):
        """Test market sentiment feature calculation"""
        data = self.create_sample_data()
        
        # Add options data for full test
        data['put_volume'] = np.random.exponential(100, len(data))
        data['call_volume'] = np.random.exponential(120, len(data))
        
        features = self.engineer.calculate_market_sentiment(data)
        
        # Check sentiment features
        assert 'rsi_14' in features
        assert 'sentiment_score' in features
        assert 'put_call_ratio' in features
        assert 'options_sentiment' in features
        assert 'obv_momentum' in features
        assert 'volume_price_confirmation' in features
        
        # Check value ranges
        assert 0 <= features['rsi_14'] <= 100
        assert -1 <= features['sentiment_score'] <= 1
        assert features['put_call_ratio'] >= 0
        assert features['volume_price_confirmation'] in [-1, 0, 1]
        
    def test_get_all_features(self):
        """Test comprehensive feature calculation"""
        data = self.create_sample_data(500)
        all_features = self.engineer.get_all_features(data)
        
        # Check that all feature categories are present
        micro_features = [k for k in all_features if k.startswith('micro_')]
        mtf_features = [k for k in all_features if k.startswith('mtf_')]
        vp_features = [k for k in all_features if k.startswith('vp_')]
        vol_features = [k for k in all_features if k.startswith('vol_')]
        sent_features = [k for k in all_features if k.startswith('sent_')]
        
        assert len(micro_features) > 5
        assert len(mtf_features) > 5
        assert len(vp_features) > 5
        assert len(vol_features) > 3
        assert len(sent_features) > 3
        
        # Check timestamp
        assert 'feature_timestamp' in all_features
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with minimal data
        small_data = self.create_sample_data(10)
        features = self.engineer.get_all_features(small_data)
        assert isinstance(features, dict)
        
        # Test with missing columns
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102, 101, 100],
            'volume': [1000, 1200, 800, 900, 1100]
        })
        features = self.engineer.get_all_features(incomplete_data)
        assert isinstance(features, dict)
        
        # Test with NaN values
        nan_data = self.create_sample_data()
        nan_data.loc[10:20, 'close'] = np.nan
        features = self.engineer.get_all_features(nan_data)
        assert isinstance(features, dict)
        
    def test_feature_stability(self):
        """Test feature calculation stability"""
        data = self.create_sample_data(200)
        
        # Calculate features multiple times
        features1 = self.engineer.get_all_features(data)
        features2 = self.engineer.get_all_features(data)
        
        # Check that features are deterministic
        for key in features1:
            if key != 'feature_timestamp':
                if isinstance(features1[key], (int, float)) and not np.isnan(features1[key]):
                    assert features1[key] == features2[key]
                    
    def test_performance(self):
        """Test feature calculation performance"""
        import time
        
        data = self.create_sample_data(1000)
        
        start_time = time.time()
        features = self.engineer.get_all_features(data)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should complete within reasonable time (1 second)
        assert calculation_time < 1.0
        
        print(f"Feature calculation time: {calculation_time:.3f} seconds")
        print(f"Total features calculated: {len(features)}")


if __name__ == "__main__":
    # Run tests
    test = TestAdvancedFeatures()
    test.setup_method()
    
    # Run each test
    print("Testing market microstructure...")
    test.test_market_microstructure()
    print("✓ Passed")
    
    print("Testing multi-timeframe features...")
    test.test_multi_timeframe_features()
    print("✓ Passed")
    
    print("Testing volume profile...")
    test.test_volume_profile_features()
    print("✓ Passed")
    
    print("Testing volatility regime...")
    test.test_volatility_regime_features()
    print("✓ Passed")
    
    print("Testing market sentiment...")
    test.test_market_sentiment_features()
    print("✓ Passed")
    
    print("Testing all features...")
    test.test_get_all_features()
    print("✓ Passed")
    
    print("Testing edge cases...")
    test.test_edge_cases()
    print("✓ Passed")
    
    print("Testing feature stability...")
    test.test_feature_stability()
    print("✓ Passed")
    
    print("Testing performance...")
    test.test_performance()
    print("✓ Passed")
    
    print("\nAll tests passed! ✨")