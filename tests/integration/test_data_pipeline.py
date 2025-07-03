# tests/integration/test_data_pipeline.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from data.market_data_input import MarketDataInput
from data.feature_engineering_pipeline import FeatureEngineeringPipeline
from data.data_augmentation import MarketDataAugmenter
from data.phase_aware_data_augmenter import PhaseAwareDataAugmenter
from core.attention_learning_layer import AttentionLearningLayer


class TestDataPipeline:
    """Test data flow through the entire pipeline"""
    
    @pytest.fixture
    async def data_pipeline(self):
        """Create complete data pipeline"""
        config = {
            'symbol': 'BTC/USDT',
            'timeframe': '5m',
            'lookback_periods': 100,
            'feature_window': 20,
            'data_validation': True,
            'storage_type': 'memory'
        }
        
        # Initialize pipeline components
        pipeline = {
            'collector': MarketDataCollector(config),
            'validator': DataValidator(config),
            'preprocessor': DataPreprocessor(config),
            'feature_engineer': FeatureEngineer(config),
            'storage': DataStorage(config),
            'consumer': AttentionLearningSystem(config)
        }
        
        return pipeline, config
    
    @pytest.mark.asyncio
    async def test_raw_data_collection_to_storage(self, data_pipeline):
        """Test data collection through validation to storage"""
        pipeline, config = data_pipeline
        
        # Mock exchange data
        mock_exchange_data = self._create_mock_exchange_data(50)
        
        with patch.object(pipeline['collector'], 'fetch_from_exchange', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_exchange_data
            
            # Collect data
            raw_data = await pipeline['collector'].collect_market_data()
            
            # Validate
            validation_result = await pipeline['validator'].validate_data(raw_data)
            assert validation_result['is_valid']
            assert len(validation_result['issues']) == 0
            
            # Store
            storage_result = await pipeline['storage'].store_raw_data(raw_data)
            assert storage_result['success']
            assert storage_result['records_stored'] == 50
    
    @pytest.mark.asyncio
    async def test_data_preprocessing_pipeline(self, data_pipeline):
        """Test preprocessing stages in sequence"""
        pipeline, config = data_pipeline
        
        # Generate raw data with issues
        raw_data = self._create_raw_data_with_issues(100)
        
        # Validate and identify issues
        validation = await pipeline['validator'].validate_data(raw_data)
        assert not validation['is_valid']  # Should detect issues
        
        # Preprocess to fix issues
        cleaned_data = await pipeline['preprocessor'].clean_data(raw_data, validation['issues'])
        
        # Re-validate cleaned data
        revalidation = await pipeline['validator'].validate_data(cleaned_data)
        assert revalidation['is_valid']  # Should be clean now
        
        # Normalize
        normalized_data = await pipeline['preprocessor'].normalize_data(cleaned_data)
        
        # Check normalization
        assert normalized_data['close'].mean() < 1.0  # Should be normalized
        assert normalized_data['close'].std() < 2.0
    
    @pytest.mark.asyncio
    async def test_feature_engineering_pipeline(self, data_pipeline):
        """Test feature generation and validation"""
        pipeline, config = data_pipeline
        
        # Start with clean data
        clean_data = self._create_clean_market_data(200)
        
        # Engineer features
        features = await pipeline['feature_engineer'].create_features(clean_data)
        
        # Verify features created
        expected_features = [
            'returns', 'log_returns', 'volatility',
            'rsi', 'macd', 'bb_upper', 'bb_lower',
            'volume_ratio', 'price_momentum'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
        
        # Validate feature quality
        feature_validation = await pipeline['validator'].validate_features(features)
        assert feature_validation['all_valid']
        
        # Check feature statistics
        assert not features.isnull().any().any()  # No nulls
        assert not np.isinf(features.values).any()  # No infinities
    
    @pytest.mark.asyncio
    async def test_streaming_data_pipeline(self, data_pipeline):
        """Test real-time streaming data processing"""
        pipeline, config = data_pipeline
        
        # Setup streaming
        processed_batches = []
        
        async def process_stream_batch(batch):
            # Validate
            validation = await pipeline['validator'].validate_data(batch)
            if validation['is_valid']:
                # Preprocess
                cleaned = await pipeline['preprocessor'].clean_data(batch)
                # Engineer features
                features = await pipeline['feature_engineer'].create_features(cleaned)
                processed_batches.append(features)
        
        # Simulate streaming data
        for i in range(5):
            # Generate batch
            batch = self._create_streaming_batch(10, base_price=50000 + i*100)
            
            # Process through pipeline
            await process_stream_batch(batch)
            
            # Simulate delay
            await asyncio.sleep(0.1)
        
        # Verify all batches processed
        assert len(processed_batches) == 5
        
        # Check continuity
        for i in range(1, len(processed_batches)):
            prev_last = processed_batches[i-1].iloc[-1]['close']
            curr_first = processed_batches[i].iloc[0]['close']
            # Prices should be continuous
            assert abs(curr_first - prev_last) / prev_last < 0.1
    
    @pytest.mark.asyncio
    async def test_data_pipeline_with_model_consumption(self, data_pipeline):
        """Test complete pipeline feeding into ML model"""
        pipeline, config = data_pipeline
        
        # Collect and process data
        raw_data = self._create_clean_market_data(500)
        
        # Full pipeline processing
        validated = await pipeline['validator'].validate_data(raw_data)
        assert validated['is_valid']
        
        normalized = await pipeline['preprocessor'].normalize_data(raw_data)
        features = await pipeline['feature_engineer'].create_features(normalized)
        
        # Store processed data
        await pipeline['storage'].store_processed_data(features)
        
        # Consumer (attention model) retrieves and uses data
        model_data = await pipeline['storage'].get_latest_features(100)
        
        # Feed to attention model
        attention_output = await pipeline['consumer'].process_market_data(model_data)
        
        # Verify model output
        assert attention_output is not None
        assert 'attention_weights' in attention_output
        assert 'predictions' in attention_output
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, data_pipeline):
        """Test pipeline resilience to various errors"""
        pipeline, config = data_pipeline
        
        # Test various error scenarios
        error_scenarios = [
            {
                'name': 'missing_data',
                'data': pd.DataFrame(),  # Empty
                'expected_error': 'No data'
            },
            {
                'name': 'corrupt_values',
                'data': self._create_corrupt_data(50),
                'expected_error': 'Invalid values'
            },
            {
                'name': 'wrong_schema',
                'data': pd.DataFrame({'wrong_col': [1, 2, 3]}),
                'expected_error': 'Schema mismatch'
            }
        ]
        
        for scenario in error_scenarios:
            # Should handle gracefully
            validation = await pipeline['validator'].validate_data(scenario['data'])
            assert not validation['is_valid']
            assert len(validation['issues']) > 0
            
            # Pipeline should not crash
            try:
                cleaned = await pipeline['preprocessor'].clean_data(
                    scenario['data'], 
                    validation['issues']
                )
                # If cleaning succeeds, data should be valid
                if cleaned is not None and len(cleaned) > 0:
                    revalidation = await pipeline['validator'].validate_data(cleaned)
                    assert revalidation['is_valid']
            except Exception as e:
                # Should provide meaningful error
                assert scenario['expected_error'].lower() in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_monitoring(self, data_pipeline):
        """Test pipeline performance metrics collection"""
        pipeline, config = data_pipeline
        
        # Track metrics
        metrics = {
            'collection_time': [],
            'validation_time': [],
            'preprocessing_time': [],
            'feature_time': [],
            'total_time': []
        }
        
        # Process multiple batches
        for i in range(10):
            start_time = datetime.now()
            
            # Generate data
            data = self._create_clean_market_data(100)
            
            # Time each stage
            t1 = datetime.now()
            validation = await pipeline['validator'].validate_data(data)
            t2 = datetime.now()
            
            cleaned = await pipeline['preprocessor'].clean_data(data)
            t3 = datetime.now()
            
            features = await pipeline['feature_engineer'].create_features(cleaned)
            t4 = datetime.now()
            
            # Record times
            metrics['validation_time'].append((t2 - t1).total_seconds())
            metrics['preprocessing_time'].append((t3 - t2).total_seconds())
            metrics['feature_time'].append((t4 - t3).total_seconds())
            metrics['total_time'].append((t4 - start_time).total_seconds())
        
        # Analyze performance
        avg_total = np.mean(metrics['total_time'])
        assert avg_total < 1.0  # Should process in under 1 second
        
        # Identify bottlenecks
        stage_times = {
            'validation': np.mean(metrics['validation_time']),
            'preprocessing': np.mean(metrics['preprocessing_time']),
            'features': np.mean(metrics['feature_time'])
        }
        
        slowest_stage = max(stage_times, key=stage_times.get)
        print(f"Slowest stage: {slowest_stage} ({stage_times[slowest_stage]:.3f}s)")
    
    @pytest.mark.asyncio
    async def test_data_quality_monitoring(self, data_pipeline):
        """Test continuous data quality monitoring"""
        pipeline, config = data_pipeline
        
        # Quality metrics tracker
        quality_metrics = []
        
        # Process data with varying quality
        quality_levels = [1.0, 0.9, 0.7, 0.5, 0.8, 0.95]
        
        for quality in quality_levels:
            # Generate data with specified quality
            data = self._create_data_with_quality(100, quality)
            
            # Validate and get quality score
            validation = await pipeline['validator'].validate_data(data)
            quality_score = await pipeline['validator'].calculate_quality_score(data)
            
            quality_metrics.append({
                'timestamp': datetime.now(),
                'quality_score': quality_score,
                'is_valid': validation['is_valid'],
                'issue_count': len(validation['issues'])
            })
        
        # Analyze quality trend
        scores = [m['quality_score'] for m in quality_metrics]
        
        # Should detect quality degradation
        assert min(scores) < 0.6
        assert max(scores) > 0.9
        
        # Low quality should correlate with validation issues
        for metric in quality_metrics:
            if metric['quality_score'] < 0.6:
                assert not metric['is_valid'] or metric['issue_count'] > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_data_versioning(self, data_pipeline):
        """Test data versioning and lineage tracking"""
        pipeline, config = data_pipeline
        
        # Process data with versioning
        versions = []
        
        for version in range(3):
            # Generate data
            raw_data = self._create_clean_market_data(100)
            
            # Add version metadata
            raw_data.attrs['version'] = f'v{version}'
            raw_data.attrs['timestamp'] = datetime.now()
            raw_data.attrs['source'] = 'test_exchange'
            
            # Process through pipeline
            features = await pipeline['feature_engineer'].create_features(raw_data)
            
            # Preserve lineage
            features.attrs['raw_version'] = raw_data.attrs['version']
            features.attrs['processing_timestamp'] = datetime.now()
            features.attrs['pipeline_version'] = '1.0.0'
            
            # Store with version
            await pipeline['storage'].store_versioned_data(features, version=f'v{version}')
            versions.append(features)
        
        # Retrieve specific version
        retrieved_v1 = await pipeline['storage'].get_data_version('v1')
        assert retrieved_v1 is not None
        assert retrieved_v1.attrs['raw_version'] == 'v1'
        
        # Get latest version
        latest = await pipeline['storage'].get_latest_version()
        assert latest.attrs['raw_version'] == 'v2'
    
    # Helper methods
    def _create_mock_exchange_data(self, size: int) -> pd.DataFrame:
        """Create mock exchange data"""
        dates = pd.date_range(end=datetime.now(), periods=size, freq='5min')
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(49000, 51000, size),
            'high': np.random.uniform(49500, 51500, size),
            'low': np.random.uniform(48500, 50500, size),
            'close': np.random.uniform(49000, 51000, size),
            'volume': np.random.uniform(10, 100, size)
        })
    
    def _create_raw_data_with_issues(self, size: int) -> pd.DataFrame:
        """Create data with various issues"""
        data = self._create_mock_exchange_data(size)
        
        # Add issues
        data.loc[10:15, 'close'] = np.nan  # Missing values
        data.loc[20:25, 'volume'] = -1  # Negative volumes
        data.loc[30:35, 'high'] = data.loc[30:35, 'low'] - 100  # High < Low
        data.loc[40:45, 'close'] = np.inf  # Infinities
        
        return data
    
    def _create_clean_market_data(self, size: int) -> pd.DataFrame:
        """Create clean market data"""
        dates = pd.date_range(end=datetime.now(), periods=size, freq='5min')
        
        # Generate realistic price movement
        price = 50000
        prices = []
        for _ in range(size):
            price *= (1 + np.random.normal(0, 0.001))
            prices.append(price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': np.random.uniform(50, 150, size)
        })
    
    def _create_streaming_batch(self, size: int, base_price: float) -> pd.DataFrame:
        """Create a batch of streaming data"""
        start = datetime.now()
        dates = pd.date_range(start=start, periods=size, freq='5min')
        
        prices = base_price + np.random.normal(0, 50, size)
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.uniform(50, 150, size)
        })
    
    def _create_corrupt_data(self, size: int) -> pd.DataFrame:
        """Create data with corrupt values"""
        data = self._create_clean_market_data(size)
        
        # Corrupt random values
        corrupt_idx = np.random.choice(size, size//5, replace=False)
        data.loc[corrupt_idx, 'close'] = 'invalid'  # String in numeric column
        
        return data
    
    def _create_data_with_quality(self, size: int, quality: float) -> pd.DataFrame:
        """Create data with specified quality level"""
        data = self._create_clean_market_data(size)
        
        # Degrade quality
        if quality < 1.0:
            # Add missing values
            missing_count = int(size * (1 - quality) * 0.5)
            missing_idx = np.random.choice(size, missing_count, replace=False)
            data.loc[missing_idx, 'close'] = np.nan
            
            # Add outliers
            outlier_count = int(size * (1 - quality) * 0.3)
            outlier_idx = np.random.choice(size, outlier_count, replace=False)
            data.loc[outlier_idx, 'volume'] = data['volume'].mean() * 10
        
        return data