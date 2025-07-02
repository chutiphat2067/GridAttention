"""
data_augmentation.py
Data augmentation system to increase training data diversity and prevent overfitting

Author: Grid Trading System
Date: 2024
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.decomposition import PCA
import random
import hashlib

logger = logging.getLogger(__name__)

# Constants
NOISE_LEVELS = {
    'conservative': 0.0001,  # 0.01% noise
    'moderate': 0.0005,      # 0.05% noise
    'aggressive': 0.001      # 0.1% noise
}

AUGMENTATION_METHODS = [
    'noise_injection',
    'time_warping',
    'magnitude_warping',
    'bootstrap_sampling',
    'synthetic_patterns',
    'regime_mixing',
    'feature_dropout',
    'temporal_shift'
]


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    noise_level: str = 'moderate'
    time_warp_factor: float = 0.1
    magnitude_warp_factor: float = 0.1
    bootstrap_ratio: float = 0.5
    synthetic_ratio: float = 0.3
    feature_dropout_rate: float = 0.1
    preserve_correlations: bool = True
    max_augmentation_factor: int = 3  # Maximum 3x original data
    seed: Optional[int] = None


@dataclass
class AugmentedData:
    """Container for augmented data"""
    original_size: int
    augmented_size: int
    methods_used: List[str]
    augmentation_params: Dict[str, Any]
    quality_score: float
    timestamp: float = field(default_factory=time.time)


class MarketDataAugmenter:
    """Augment market data to increase training diversity"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.augmentation_history = deque(maxlen=100)
        self.quality_metrics = {}
        
    def augment_market_data(self, 
                          data: Union[List['MarketTick'], pd.DataFrame],
                          methods: Optional[List[str]] = None) -> Tuple[Any, AugmentedData]:
        """Main augmentation function"""
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = self._ticks_to_dataframe(data)
        else:
            df = data.copy()
            
        original_size = len(df)
        
        # Select augmentation methods
        if methods is None:
            methods = self._select_augmentation_methods(df)
            
        # Apply augmentations
        augmented_dfs = [df]  # Include original
        
        for method in methods:
            if method == 'noise_injection':
                augmented_dfs.append(self._add_noise(df))
                
            elif method == 'time_warping':
                augmented_dfs.append(self._time_warp(df))
                
            elif method == 'magnitude_warping':
                augmented_dfs.append(self._magnitude_warp(df))
                
            elif method == 'bootstrap_sampling':
                augmented_dfs.append(self._bootstrap_sample(df))
                
            elif method == 'synthetic_patterns':
                augmented_dfs.append(self._generate_synthetic_patterns(df))
                
            elif method == 'regime_mixing':
                augmented_dfs.append(self._mix_regimes(df))
                
            elif method == 'feature_dropout':
                augmented_dfs.append(self._feature_dropout(df))
                
            elif method == 'temporal_shift':
                augmented_dfs.append(self._temporal_shift(df))
                
        # Combine augmented data
        augmented_df = pd.concat(augmented_dfs, ignore_index=True)
        
        # Limit augmentation factor
        max_size = original_size * self.config.max_augmentation_factor
        if len(augmented_df) > max_size:
            augmented_df = augmented_df.sample(n=max_size, random_state=self.rng)
            
        # Shuffle to mix original and augmented
        augmented_df = augmented_df.sample(frac=1, random_state=self.rng).reset_index(drop=True)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, augmented_df)
        
        # Create metadata
        augmented_data = AugmentedData(
            original_size=original_size,
            augmented_size=len(augmented_df),
            methods_used=methods,
            augmentation_params={
                'noise_level': self.config.noise_level,
                'time_warp_factor': self.config.time_warp_factor,
                'magnitude_warp_factor': self.config.magnitude_warp_factor
            },
            quality_score=quality_score
        )
        
        # Store in history
        self.augmentation_history.append(augmented_data)
        
        # Convert back to ticks if needed
        if isinstance(data, list):
            augmented_ticks = self._dataframe_to_ticks(augmented_df)
            return augmented_ticks, augmented_data
        else:
            return augmented_df, augmented_data
            
    def _ticks_to_dataframe(self, ticks: List['MarketTick']) -> pd.DataFrame:
        """Convert list of MarketTick to DataFrame"""
        data = []
        for tick in ticks:
            data.append({
                'timestamp': tick.timestamp,
                'price': tick.price,
                'volume': tick.volume,
                'bid': tick.bid,
                'ask': tick.ask,
                'symbol': tick.symbol
            })
        return pd.DataFrame(data)
        
    def _dataframe_to_ticks(self, df: pd.DataFrame) -> List['MarketTick']:
        """Convert DataFrame back to MarketTick objects"""
        from data.market_data_input import MarketTick
        
        ticks = []
        for _, row in df.iterrows():
            tick = MarketTick(
                symbol=row['symbol'],
                price=row['price'],
                volume=row['volume'],
                timestamp=row['timestamp'],
                bid=row['bid'],
                ask=row['ask'],
                exchange='augmented'
            )
            ticks.append(tick)
        return ticks
        
    def _select_augmentation_methods(self, df: pd.DataFrame) -> List[str]:
        """Intelligently select augmentation methods based on data characteristics"""
        methods = []
        
        # Always add some noise
        methods.append('noise_injection')
        
        # Check data size
        if len(df) > 1000:
            methods.append('bootstrap_sampling')
            
        # Check for volatility
        price_volatility = df['price'].pct_change().std()
        if price_volatility > 0.001:
            methods.append('magnitude_warping')
            
        # Check for trends
        if self._has_trend(df['price']):
            methods.append('time_warping')
            
        # Add synthetic patterns for more diversity
        if len(df) > 500:
            methods.append('synthetic_patterns')
            
        return methods
        
    def _add_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic noise to data"""
        augmented = df.copy()
        noise_scale = NOISE_LEVELS[self.config.noise_level]
        
        # Price noise (correlated)
        price_noise = self.rng.normal(0, augmented['price'].mean() * noise_scale, len(augmented))
        if self.config.preserve_correlations:
            # Apply smoothing to maintain local correlations
            price_noise = signal.savgol_filter(price_noise, window_length=5, polyorder=2, mode='nearest')
        augmented['price'] += price_noise
        
        # Volume noise (always positive)
        volume_noise = np.abs(self.rng.normal(0, augmented['volume'].mean() * noise_scale * 2, len(augmented)))
        augmented['volume'] += volume_noise
        
        # Adjust bid/ask to maintain spread
        spread = augmented['ask'] - augmented['bid']
        mid_price = (augmented['bid'] + augmented['ask']) / 2
        
        # Add noise to mid price
        mid_noise = self.rng.normal(0, mid_price.mean() * noise_scale, len(augmented))
        if self.config.preserve_correlations:
            mid_noise = signal.savgol_filter(mid_noise, window_length=5, polyorder=2, mode='nearest')
            
        augmented['bid'] = mid_price + mid_noise - spread / 2
        augmented['ask'] = mid_price + mid_noise + spread / 2
        
        # Add augmentation marker
        augmented['augmentation_type'] = 'noise'
        
        return augmented
        
    def _time_warp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply time warping to create temporal variations"""
        augmented = df.copy()
        
        # Generate smooth warping function
        warp_points = 10
        warp_values = self.rng.uniform(1 - self.config.time_warp_factor, 
                                      1 + self.config.time_warp_factor, 
                                      warp_points)
        
        # Interpolate to full length
        x = np.linspace(0, len(df) - 1, warp_points)
        x_new = np.arange(len(df))
        warp_function = np.interp(x_new, x, warp_values)
        
        # Apply warping to time-sensitive features
        for col in ['price', 'volume', 'bid', 'ask']:
            if col in augmented.columns:
                # Resample using warped time indices
                original_indices = np.arange(len(df))
                warped_indices = np.cumsum(warp_function)
                warped_indices = warped_indices * (len(df) - 1) / warped_indices[-1]
                
                augmented[col] = np.interp(original_indices, warped_indices, df[col])
                
        augmented['augmentation_type'] = 'time_warp'
        
        return augmented
        
    def _magnitude_warp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply magnitude warping to create amplitude variations"""
        augmented = df.copy()
        
        # Generate smooth magnitude warping
        warp_window = max(20, len(df) // 50)
        
        for col in ['price', 'volume']:
            if col not in augmented.columns:
                continue
                
            # Decompose into trend and deviations
            trend = augmented[col].rolling(window=warp_window, center=True).mean()
            trend = trend.fillna(method='bfill').fillna(method='ffill')
            deviations = augmented[col] - trend
            
            # Warp the deviations
            warp_factor = self.rng.uniform(1 - self.config.magnitude_warp_factor,
                                         1 + self.config.magnitude_warp_factor,
                                         len(augmented))
            
            # Smooth the warp factor
            warp_factor = pd.Series(warp_factor).rolling(window=warp_window, center=True).mean()
            warp_factor = warp_factor.fillna(method='bfill').fillna(method='ffill')
            
            # Apply warping
            augmented[col] = trend + deviations * warp_factor
            
        # Adjust bid/ask proportionally
        price_ratio = augmented['price'] / df['price']
        augmented['bid'] *= price_ratio
        augmented['ask'] *= price_ratio
        
        augmented['augmentation_type'] = 'magnitude_warp'
        
        return augmented
        
    def _bootstrap_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create bootstrap samples with block structure preservation"""
        n_samples = int(len(df) * self.config.bootstrap_ratio)
        
        # Use block bootstrap to preserve temporal structure
        block_size = max(10, len(df) // 100)
        n_blocks = n_samples // block_size + 1
        
        blocks = []
        for _ in range(n_blocks):
            start_idx = self.rng.randint(0, len(df) - block_size)
            block = df.iloc[start_idx:start_idx + block_size].copy()
            blocks.append(block)
            
        augmented = pd.concat(blocks, ignore_index=True)[:n_samples]
        
        # Add small time shifts to avoid exact duplicates
        time_shift = self.rng.uniform(-1, 1, len(augmented))
        augmented['timestamp'] += time_shift
        
        # Sort by timestamp
        augmented = augmented.sort_values('timestamp').reset_index(drop=True)
        
        augmented['augmentation_type'] = 'bootstrap'
        
        return augmented
        
    def _generate_synthetic_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic patterns based on learned characteristics"""
        n_synthetic = int(len(df) * self.config.synthetic_ratio)
        
        # Extract statistical properties
        price_returns = df['price'].pct_change().dropna()
        volume_profile = df['volume'].describe()
        spread_stats = (df['ask'] - df['bid']).describe()
        
        # Generate synthetic price series using GARCH-like model
        synthetic_returns = self._generate_garch_returns(price_returns, n_synthetic)
        
        # Convert returns to prices
        initial_price = df['price'].iloc[-1]
        synthetic_prices = initial_price * (1 + synthetic_returns).cumprod()
        
        # Generate correlated volume
        volume_correlation = df[['price', 'volume']].pct_change().corr().iloc[0, 1]
        volume_noise = self.rng.normal(0, 1, n_synthetic)
        
        synthetic_volume = volume_profile['mean'] + volume_profile['std'] * (
            volume_correlation * synthetic_returns + 
            np.sqrt(1 - volume_correlation**2) * volume_noise
        )
        synthetic_volume = np.maximum(synthetic_volume, volume_profile['min'])
        
        # Generate spread
        synthetic_spread = self.rng.normal(spread_stats['mean'], spread_stats['std'], n_synthetic)
        synthetic_spread = np.maximum(synthetic_spread, spread_stats['min'])
        
        # Create synthetic DataFrame
        augmented = pd.DataFrame({
            'timestamp': np.linspace(df['timestamp'].iloc[-1], 
                                   df['timestamp'].iloc[-1] + n_synthetic * 5, 
                                   n_synthetic),
            'price': synthetic_prices,
            'volume': synthetic_volume,
            'bid': synthetic_prices - synthetic_spread / 2,
            'ask': synthetic_prices + synthetic_spread / 2,
            'symbol': df['symbol'].iloc[0],
            'augmentation_type': 'synthetic'
        })
        
        return augmented
        
    def _generate_garch_returns(self, historical_returns: pd.Series, n_periods: int) -> np.ndarray:
        """Generate returns using simplified GARCH model"""
        # Estimate parameters
        mean_return = historical_returns.mean()
        unconditional_vol = historical_returns.std()
        
        # GARCH(1,1) parameters (simplified)
        omega = unconditional_vol**2 * 0.1
        alpha = 0.1  # Impact of past shocks
        beta = 0.85  # Persistence
        
        # Generate returns
        returns = np.zeros(n_periods)
        variance = np.zeros(n_periods)
        variance[0] = unconditional_vol**2
        
        for t in range(1, n_periods):
            # Update variance
            variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
            
            # Generate return
            returns[t] = mean_return + np.sqrt(variance[t]) * self.rng.normal()
            
        return returns
        
    def _mix_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mix different market regimes to create transitional patterns"""
        if len(df) < 200:
            return df.copy()  # Not enough data
            
        # Identify potential regime segments using simple clustering
        window = 50
        features = []
        
        for i in range(window, len(df) - window):
            segment = df.iloc[i-window:i+window]
            
            feature = {
                'volatility': segment['price'].pct_change().std(),
                'trend': (segment['price'].iloc[-1] - segment['price'].iloc[0]) / segment['price'].iloc[0],
                'volume_mean': segment['volume'].mean()
            }
            features.append(feature)
            
        if not features:
            return df.copy()
            
        # Simple regime identification
        features_df = pd.DataFrame(features)
        
        # Normalize features
        features_normalized = (features_df - features_df.mean()) / features_df.std()
        
        # Find different regimes (simplified k-means)
        n_regimes = min(3, len(features) // 50)
        regime_indices = self._simple_kmeans(features_normalized.values, n_regimes)
        
        # Mix regimes by swapping segments
        augmented = df.copy()
        
        # Randomly swap some segments
        n_swaps = min(5, len(regime_indices) // 10)
        
        for _ in range(n_swaps):
            # Select two different regime segments
            regime1 = self.rng.randint(0, n_regimes)
            regime2 = (regime1 + 1) % n_regimes
            
            indices1 = np.where(regime_indices == regime1)[0]
            indices2 = np.where(regime_indices == regime2)[0]
            
            if len(indices1) > 0 and len(indices2) > 0:
                idx1 = self.rng.choice(indices1)
                idx2 = self.rng.choice(indices2)
                
                # Swap segments with smooth transition
                start1 = idx1 * window
                start2 = idx2 * window
                segment_size = window
                
                if start1 + segment_size < len(augmented) and start2 + segment_size < len(augmented):
                    # Create smooth transition
                    self._smooth_swap(augmented, start1, start2, segment_size)
                    
        augmented['augmentation_type'] = 'regime_mix'
        
        return augmented
        
    def _simple_kmeans(self, data: np.ndarray, k: int, max_iters: int = 50) -> np.ndarray:
        """Simple k-means implementation"""
        n_samples = len(data)
        
        # Initialize centroids
        centroids = data[self.rng.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iters):
            # Assign to nearest centroid
            distances = np.array([np.linalg.norm(data - c, axis=1) for c in centroids])
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
            
        return labels
        
    def _smooth_swap(self, df: pd.DataFrame, start1: int, start2: int, size: int):
        """Swap segments with smooth transition"""
        # Create transition weights
        transition_size = min(10, size // 5)
        weights = np.linspace(0, 1, transition_size)
        
        for col in ['price', 'volume', 'bid', 'ask']:
            if col not in df.columns:
                continue
                
            # Get segments
            segment1 = df[col].iloc[start1:start1+size].values.copy()
            segment2 = df[col].iloc[start2:start2+size].values.copy()
            
            # Apply smooth transition at boundaries
            if transition_size > 0:
                # Blend start
                for i in range(transition_size):
                    segment1[i] = segment1[i] * (1 - weights[i]) + segment2[i] * weights[i]
                    segment2[i] = segment2[i] * (1 - weights[i]) + segment1[i] * weights[i]
                    
                # Blend end
                for i in range(transition_size):
                    idx = size - transition_size + i
                    segment1[idx] = segment1[idx] * weights[i] + segment2[idx] * (1 - weights[i])
                    segment2[idx] = segment2[idx] * weights[i] + segment1[idx] * (1 - weights[i])
                    
            # Swap
            df[col].iloc[start1:start1+size] = segment2
            df[col].iloc[start2:start2+size] = segment1
            
    def _feature_dropout(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomly drop features to increase robustness"""
        augmented = df.copy()
        
        # Randomly mask some features
        for col in ['volume']:  # Don't dropout price
            if col in augmented.columns and self.rng.random() < self.config.feature_dropout_rate:
                # Replace with rolling mean
                window = 20
                augmented[col] = augmented[col].rolling(window=window, center=True).mean()
                augmented[col] = augmented[col].fillna(method='bfill').fillna(method='ffill')
                
        augmented['augmentation_type'] = 'feature_dropout'
        
        return augmented
        
    def _temporal_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal shifts to create lag variations"""
        augmented = df.copy()
        
        # Shift different features by different amounts
        shifts = {
            'volume': self.rng.randint(-5, 5),
            'bid': self.rng.randint(-2, 2),
            'ask': self.rng.randint(-2, 2)
        }
        
        for col, shift in shifts.items():
            if col in augmented.columns and shift != 0:
                augmented[col] = augmented[col].shift(shift)
                
        # Fill NaN values
        augmented = augmented.fillna(method='bfill').fillna(method='ffill')
        
        augmented['augmentation_type'] = 'temporal_shift'
        
        return augmented
        
    def _has_trend(self, series: pd.Series) -> bool:
        """Check if series has significant trend"""
        if len(series) < 50:
            return False
            
        # Simple linear regression test
        x = np.arange(len(series))
        slope, _, r_value, p_value, _ = stats.linregress(x, series)
        
        # Significant trend if p < 0.05 and R² > 0.3
        return p_value < 0.05 and r_value**2 > 0.3
        
    def _calculate_quality_score(self, original: pd.DataFrame, augmented: pd.DataFrame) -> float:
        """Calculate quality score of augmented data"""
        score = 1.0
        
        # Check statistical similarity
        for col in ['price', 'volume']:
            if col not in original.columns:
                continue
                
            # Compare distributions
            original_stats = original[col].describe()
            augmented_stats = augmented[col].describe()
            
            # Penalize large deviations in mean and std
            mean_diff = abs(original_stats['mean'] - augmented_stats['mean']) / original_stats['mean']
            std_diff = abs(original_stats['std'] - augmented_stats['std']) / original_stats['std']
            
            if mean_diff > 0.1:  # More than 10% difference
                score *= (1 - mean_diff)
                
            if std_diff > 0.2:  # More than 20% difference
                score *= (1 - std_diff / 2)
                
        # Check correlation preservation
        if self.config.preserve_correlations and len(original) > 50:
            original_corr = original[['price', 'volume']].corr().iloc[0, 1]
            augmented_corr = augmented[['price', 'volume']].corr().iloc[0, 1]
            
            corr_diff = abs(original_corr - augmented_corr)
            if corr_diff > 0.2:
                score *= (1 - corr_diff)
                
        return max(0.0, min(1.0, score))
        
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about augmentation performance"""
        if not self.augmentation_history:
            return {}
            
        history = list(self.augmentation_history)
        
        return {
            'total_augmentations': len(history),
            'average_augmentation_factor': np.mean([a.augmented_size / a.original_size for a in history]),
            'average_quality_score': np.mean([a.quality_score for a in history]),
            'methods_usage': self._count_methods_usage(history),
            'recent_augmentations': [
                {
                    'timestamp': a.timestamp,
                    'original_size': a.original_size,
                    'augmented_size': a.augmented_size,
                    'quality_score': a.quality_score,
                    'methods': a.methods_used
                }
                for a in history[-5:]
            ]
        }
        
    def _count_methods_usage(self, history: List[AugmentedData]) -> Dict[str, int]:
        """Count usage of each augmentation method"""
        usage = defaultdict(int)
        
        for aug in history:
            for method in aug.methods_used:
                usage[method] += 1
                
        return dict(usage)


class FeatureAugmenter:
    """Augment extracted features for model training"""
    
    def __init__(self, feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        self.feature_ranges = feature_ranges or {}
        self.augmentation_history = deque(maxlen=100)
        self.rng = np.random.RandomState(42)
        
    def augment_features(self, 
                        features: pd.DataFrame,
                        labels: Optional[pd.Series] = None,
                        augmentation_factor: float = 2.0) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Augment feature dataset"""
        
        original_size = len(features)
        n_augmented = int(original_size * (augmentation_factor - 1))
        
        augmented_features = [features]
        augmented_labels = [labels] if labels is not None else None
        
        # Multiple augmentation strategies
        strategies = [
            self._mixup_augmentation,
            self._feature_noise_augmentation,
            self._feature_masking_augmentation,
            self._smote_like_augmentation
        ]
        
        # Apply each strategy
        samples_per_strategy = n_augmented // len(strategies)
        
        for strategy in strategies:
            aug_features, aug_labels = strategy(features, labels, samples_per_strategy)
            augmented_features.append(aug_features)
            
            if augmented_labels is not None and aug_labels is not None:
                augmented_labels.append(aug_labels)
                
        # Combine all augmented data
        final_features = pd.concat(augmented_features, ignore_index=True)
        final_labels = pd.concat(augmented_labels, ignore_index=True) if augmented_labels else None
        
        # Shuffle
        indices = self.rng.permutation(len(final_features))
        final_features = final_features.iloc[indices].reset_index(drop=True)
        
        if final_labels is not None:
            final_labels = final_labels.iloc[indices].reset_index(drop=True)
            
        # Record augmentation
        self.augmentation_history.append({
            'timestamp': time.time(),
            'original_size': original_size,
            'augmented_size': len(final_features),
            'methods': ['mixup', 'noise', 'masking', 'smote']
        })
        
        return final_features, final_labels
        
    def _mixup_augmentation(self, 
                           features: pd.DataFrame, 
                           labels: Optional[pd.Series],
                           n_samples: int) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Mixup augmentation - blend samples"""
        augmented_features = []
        augmented_labels = []
        
        for _ in range(n_samples):
            # Select two random samples
            idx1, idx2 = self.rng.choice(len(features), 2, replace=False)
            
            # Random mixing coefficient
            alpha = self.rng.beta(0.2, 0.2)  # Beta distribution for better mixing
            
            # Mix features
            mixed_features = (
                alpha * features.iloc[idx1] + 
                (1 - alpha) * features.iloc[idx2]
            )
            augmented_features.append(mixed_features)
            
            # Mix labels if available
            if labels is not None:
                mixed_label = alpha * labels.iloc[idx1] + (1 - alpha) * labels.iloc[idx2]
                augmented_labels.append(mixed_label)
                
        aug_df = pd.DataFrame(augmented_features)
        aug_labels = pd.Series(augmented_labels) if augmented_labels else None
        
        return aug_df, aug_labels
        
    def _feature_noise_augmentation(self,
                                   features: pd.DataFrame,
                                   labels: Optional[pd.Series],
                                   n_samples: int) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Add calibrated noise to features"""
        # Sample random indices
        indices = self.rng.choice(len(features), n_samples, replace=True)
        augmented = features.iloc[indices].copy()
        
        # Add noise to each feature
        for col in augmented.columns:
            # Determine noise scale based on feature range
            if col in self.feature_ranges:
                min_val, max_val = self.feature_ranges[col]
                scale = (max_val - min_val) * 0.05  # 5% of range
            else:
                scale = augmented[col].std() * 0.1  # 10% of std
                
            noise = self.rng.normal(0, scale, len(augmented))
            augmented[col] += noise
            
            # Clip to valid range if specified
            if col in self.feature_ranges:
                augmented[col] = np.clip(augmented[col], min_val, max_val)
                
        aug_labels = labels.iloc[indices] if labels is not None else None
        
        return augmented, aug_labels
        
    def _feature_masking_augmentation(self,
                                     features: pd.DataFrame,
                                     labels: Optional[pd.Series],
                                     n_samples: int) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Randomly mask features (set to mean)"""
        indices = self.rng.choice(len(features), n_samples, replace=True)
        augmented = features.iloc[indices].copy()
        
        # Mask each feature with some probability
        mask_prob = 0.1
        
        for col in augmented.columns:
            mask = self.rng.random(len(augmented)) < mask_prob
            if mask.any():
                mean_val = features[col].mean()
                augmented.loc[mask, col] = mean_val
                
        aug_labels = labels.iloc[indices] if labels is not None else None
        
        return augmented, aug_labels
        
    def _smote_like_augmentation(self,
                                 features: pd.DataFrame,
                                 labels: Optional[pd.Series],
                                 n_samples: int) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """SMOTE-like synthetic sample generation"""
        augmented_features = []
        augmented_labels = []
        
        # Use KNN-like approach (simplified)
        for _ in range(n_samples):
            # Select random sample
            idx = self.rng.randint(0, len(features))
            sample = features.iloc[idx]
            
            # Find nearest neighbors (simplified - random selection)
            n_neighbors = min(5, len(features) - 1)
            neighbor_indices = self.rng.choice(
                [i for i in range(len(features)) if i != idx], 
                n_neighbors, 
                replace=False
            )
            
            # Select random neighbor
            neighbor_idx = self.rng.choice(neighbor_indices)
            neighbor = features.iloc[neighbor_idx]
            
            # Create synthetic sample
            alpha = self.rng.random()
            synthetic = sample + alpha * (neighbor - sample)
            
            augmented_features.append(synthetic)
            
            # Interpolate label
            if labels is not None:
                synthetic_label = (
                    labels.iloc[idx] + 
                    alpha * (labels.iloc[neighbor_idx] - labels.iloc[idx])
                )
                augmented_labels.append(synthetic_label)
                
        aug_df = pd.DataFrame(augmented_features)
        aug_labels = pd.Series(augmented_labels) if augmented_labels else None
        
        return aug_df, aug_labels
        
    def set_feature_ranges(self, features: pd.DataFrame):
        """Automatically determine feature ranges from data"""
        for col in features.columns:
            min_val = features[col].quantile(0.01)
            max_val = features[col].quantile(0.99)
            self.feature_ranges[col] = (min_val, max_val)


class AugmentationValidator:
    """Validate augmented data quality"""
    
    def __init__(self):
        self.validation_results = deque(maxlen=100)
        
    def validate_augmentation(self, 
                            original: pd.DataFrame,
                            augmented: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive validation of augmented data"""
        
        results = {
            'timestamp': time.time(),
            'is_valid': True,
            'warnings': [],
            'metrics': {}
        }
        
        # 1. Statistical distribution tests
        distribution_results = self._validate_distributions(original, augmented)
        results['metrics']['distribution'] = distribution_results
        
        if distribution_results['max_ks_statistic'] > 0.2:
            results['warnings'].append("Large distribution shift detected")
            
        # 2. Correlation preservation
        correlation_results = self._validate_correlations(original, augmented)
        results['metrics']['correlation'] = correlation_results
        
        if correlation_results['max_correlation_diff'] > 0.3:
            results['warnings'].append("Correlation structure significantly altered")
            results['is_valid'] = False
            
        # 3. Temporal consistency
        if 'timestamp' in original.columns:
            temporal_results = self._validate_temporal_consistency(original, augmented)
            results['metrics']['temporal'] = temporal_results
            
            if not temporal_results['is_consistent']:
                results['warnings'].append("Temporal inconsistencies detected")
                
        # 4. Feature bounds
        bounds_results = self._validate_feature_bounds(original, augmented)
        results['metrics']['bounds'] = bounds_results
        
        if bounds_results['out_of_bounds_ratio'] > 0.05:
            results['warnings'].append("Too many out-of-bounds values")
            results['is_valid'] = False
            
        # Store results
        self.validation_results.append(results)
        
        return results
        
    def _validate_distributions(self, original: pd.DataFrame, augmented: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical distributions"""
        results = {
            'ks_statistics': {},
            'max_ks_statistic': 0.0
        }
        
        for col in original.select_dtypes(include=[np.number]).columns:
            if col in augmented.columns:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(original[col], augmented[col])
                
                results['ks_statistics'][col] = {
                    'statistic': ks_stat,
                    'p_value': p_value
                }
                
                results['max_ks_statistic'] = max(results['max_ks_statistic'], ks_stat)
                
        return results
        
    def _validate_correlations(self, original: pd.DataFrame, augmented: pd.DataFrame) -> Dict[str, Any]:
        """Validate correlation preservation"""
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'max_correlation_diff': 0.0}
            
        original_corr = original[numeric_cols].corr()
        augmented_corr = augmented[numeric_cols].corr()
        
        # Calculate maximum correlation difference
        corr_diff = np.abs(original_corr - augmented_corr)
        max_diff = corr_diff.max().max()
        
        return {
            'max_correlation_diff': max_diff,
            'correlation_matrix_diff': corr_diff.to_dict()
        }
        
    def _validate_temporal_consistency(self, original: pd.DataFrame, augmented: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal consistency"""
        # Check if timestamps are monotonic
        is_monotonic = augmented['timestamp'].is_monotonic_increasing
        
        # Check for unrealistic time gaps
        time_diffs = augmented['timestamp'].diff()
        max_gap = time_diffs.max()
        min_gap = time_diffs[time_diffs > 0].min() if len(time_diffs[time_diffs > 0]) > 0 else 0
        
        original_diffs = original['timestamp'].diff()
        original_max_gap = original_diffs.max()
        
        return {
            'is_consistent': is_monotonic and max_gap < original_max_gap * 2,
            'is_monotonic': is_monotonic,
            'max_time_gap': max_gap,
            'min_time_gap': min_gap
        }
        
    def _validate_feature_bounds(self, original: pd.DataFrame, augmented: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature bounds"""
        out_of_bounds_count = 0
        total_values = 0
        
        bounds_violations = {}
        
        for col in original.select_dtypes(include=[np.number]).columns:
            if col not in augmented.columns:
                continue
                
            # Calculate bounds from original (99% percentile)
            lower_bound = original[col].quantile(0.005)
            upper_bound = original[col].quantile(0.995)
            
            # Check augmented data
            violations = (
                (augmented[col] < lower_bound) | 
                (augmented[col] > upper_bound)
            ).sum()
            
            if violations > 0:
                bounds_violations[col] = {
                    'violations': int(violations),
                    'ratio': violations / len(augmented)
                }
                
            out_of_bounds_count += violations
            total_values += len(augmented)
            
        return {
            'out_of_bounds_ratio': out_of_bounds_count / total_values if total_values > 0 else 0,
            'violations_by_feature': bounds_violations
        }


# Utility functions
def create_augmentation_pipeline(config: Dict[str, Any]) -> Tuple[MarketDataAugmenter, FeatureAugmenter]:
    """Create configured augmentation pipeline"""
    
    # Market data augmenter
    market_config = AugmentationConfig(
        noise_level=config.get('noise_level', 'moderate'),
        time_warp_factor=config.get('time_warp_factor', 0.1),
        magnitude_warp_factor=config.get('magnitude_warp_factor', 0.1),
        bootstrap_ratio=config.get('bootstrap_ratio', 0.5),
        synthetic_ratio=config.get('synthetic_ratio', 0.3),
        feature_dropout_rate=config.get('feature_dropout_rate', 0.1),
        preserve_correlations=config.get('preserve_correlations', True),
        max_augmentation_factor=config.get('max_augmentation_factor', 3)
    )
    
    market_augmenter = MarketDataAugmenter(market_config)
    
    # Feature augmenter
    feature_augmenter = FeatureAugmenter(
        feature_ranges=config.get('feature_ranges', {})
    )
    
    return market_augmenter, feature_augmenter


# Example usage
async def example_usage():
    """Example of using data augmentation"""
    
    # Create sample data
    n_samples = 1000
    timestamps = np.arange(n_samples) * 5  # 5-second intervals
    
    # Generate synthetic price series
    price = 50000 + np.cumsum(np.random.randn(n_samples) * 50)
    volume = 100 + np.random.exponential(50, n_samples)
    spread = 0.5 + np.random.exponential(0.5, n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': price,
        'volume': volume,
        'bid': price - spread/2,
        'ask': price + spread/2,
        'symbol': 'BTCUSDT'
    })
    
    # Initialize augmenter
    config = AugmentationConfig(
        noise_level='moderate',
        time_warp_factor=0.1,
        magnitude_warp_factor=0.1,
        max_augmentation_factor=3
    )
    
    augmenter = MarketDataAugmenter(config)
    
    # Augment data
    augmented_df, metadata = augmenter.augment_market_data(df)
    
    print(f"Original size: {metadata.original_size}")
    print(f"Augmented size: {metadata.augmented_size}")
    print(f"Methods used: {metadata.methods_used}")
    print(f"Quality score: {metadata.quality_score:.3f}")
    
    # Validate augmentation
    validator = AugmentationValidator()
    validation = validator.validate_augmentation(df, augmented_df)
    
    print(f"\nValidation results:")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Warnings: {validation['warnings']}")
    
    # Get statistics
    stats = augmenter.get_augmentation_stats()
    print(f"\nAugmentation statistics:")
    print(f"Total augmentations: {stats['total_augmentations']}")
    print(f"Average quality score: {stats['average_quality_score']:.3f}")


if __name__ == "__main__":
    asyncio.run(example_usage())