"""
feature_engineering_pipeline.py
Extract features with attention tracking for grid trading system

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
import hashlib
import json

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from numba import jit

# Local imports (these would be from other modules)
from data.market_data_input import MarketTick
from data.advanced_features import AdvancedFeatureEngineer

# Setup logger
logger = logging.getLogger(__name__)


# Constants
FEATURE_CACHE_TTL = 60  # seconds
MIN_HISTORY_REQUIRED = 100  # minimum ticks for feature calculation
FEATURE_VERSION = "1.0.0"  # for cache invalidation


@dataclass
class FeatureSet:
    """Container for extracted features"""
    timestamp: float
    features: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array"""
        return np.array(list(self.features.values()))
    
    def get_feature_names(self) -> List[str]:
        """Get ordered feature names"""
        return list(self.features.keys())


class FeatureCache:
    """LRU cache for computed features"""
    
    def __init__(self, max_size: int = 1000, ttl: int = FEATURE_CACHE_TTL):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[FeatureSet]:
        """Get cached features"""
        async with self._lock:
            if key in self.cache:
                # Check if expired
                if time.time() - self.access_times[key] > self.ttl:
                    del self.cache[key]
                    del self.access_times[key]
                    self.miss_count += 1
                    return None
                    
                # Update access time
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
                
            self.miss_count += 1
            return None
            
    async def set(self, key: str, features: FeatureSet) -> None:
        """Cache features"""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                
            self.cache[key] = features
            self.access_times[key] = time.time()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)
        
        return {
            'size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class FeatureAttentionTracker:
    """Track feature importance for attention learning"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.feature_values = {}  # feature_name -> deque of values
        self.extraction_times = {}  # feature_name -> deque of times
        self.feature_impacts = {}  # feature_name -> deque of trading impacts
        self.correlation_matrix = None
        self._lock = asyncio.Lock()
        
    async def record(self, features: Dict[str, float], extraction_times: Dict[str, float]) -> None:
        """Record feature values and extraction times"""
        async with self._lock:
            for name, value in features.items():
                if name not in self.feature_values:
                    self.feature_values[name] = deque(maxlen=self.window_size)
                    self.extraction_times[name] = deque(maxlen=self.window_size)
                    
                self.feature_values[name].append(value)
                self.extraction_times[name].append(extraction_times.get(name, 0))
                
    async def record_impact(self, feature_name: str, impact: float) -> None:
        """Record feature impact on trading outcome"""
        async with self._lock:
            if feature_name not in self.feature_impacts:
                self.feature_impacts[feature_name] = deque(maxlen=self.window_size)
            self.feature_impacts[feature_name].append(impact)
            
    async def get_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance scores"""
        async with self._lock:
            importance_scores = {}
            
            for feature_name in self.feature_values:
                # Calculate variance (high variance = potentially important)
                if len(self.feature_values[feature_name]) > 10:
                    variance = np.var(self.feature_values[feature_name])
                    
                    # Calculate average extraction time (fast = good)
                    avg_time = np.mean(self.extraction_times[feature_name])
                    
                    # Calculate impact score if available
                    impact_score = 0
                    if feature_name in self.feature_impacts and len(self.feature_impacts[feature_name]) > 0:
                        impact_score = np.mean(np.abs(self.feature_impacts[feature_name]))
                        
                    # Combine scores (normalize each component)
                    importance = (
                        0.4 * (variance / (variance + 1)) +  # Variance contribution
                        0.3 * (1 / (1 + avg_time * 1000)) +  # Speed contribution (ms)
                        0.3 * (impact_score / (impact_score + 1))  # Impact contribution
                    )
                    
                    importance_scores[feature_name] = importance
                    
            return importance_scores
            
    async def update_correlation_matrix(self) -> None:
        """Update feature correlation matrix"""
        async with self._lock:
            if len(self.feature_values) < 2:
                return
                
            # Convert to DataFrame for correlation calculation
            data = {}
            min_length = min(len(values) for values in self.feature_values.values())
            
            if min_length < 10:
                return
                
            for name, values in self.feature_values.items():
                data[name] = list(values)[-min_length:]
                
            df = pd.DataFrame(data)
            self.correlation_matrix = df.corr()


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors"""
    
    def __init__(self, name: str):
        self.name = name
        self.extraction_count = 0
        self.failure_count = 0
        self.total_extraction_time = 0
        
    @abstractmethod
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Extract feature from market data"""
        pass
        
    @abstractmethod
    def get_required_history(self) -> int:
        """Get required history length for this feature"""
        pass
        
    def get_default_value(self) -> float:
        """Get default value when extraction fails"""
        return 0.0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics"""
        avg_time = self.total_extraction_time / max(self.extraction_count, 1)
        success_rate = (self.extraction_count - self.failure_count) / max(self.extraction_count, 1)
        
        return {
            'name': self.name,
            'extractions': self.extraction_count,
            'failures': self.failure_count,
            'success_rate': success_rate,
            'avg_extraction_time': avg_time
        }


class PriceChangeExtractor(BaseFeatureExtractor):
    """Extract price change over specified period"""
    
    def __init__(self, period: int = 5):
        super().__init__(f'price_change_{period}m')
        self.period = period
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate price change percentage"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            if len(market_data) < self.get_required_history():
                raise ValueError(f"Insufficient data: {len(market_data)} < {self.get_required_history()}")
                
            # Get prices at start and end of period
            current_price = market_data[-1].price
            past_price = market_data[-self.period * 12].price  # Assuming 5-second ticks
            
            # Calculate percentage change
            price_change = (current_price - past_price) / past_price
            
            self.total_extraction_time += time.perf_counter() - start_time
            return price_change
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return self.get_default_value()
            
    def get_required_history(self) -> int:
        return self.period * 12  # 5-second ticks


class PricePositionExtractor(BaseFeatureExtractor):
    """Extract price position within recent range"""
    
    def __init__(self, lookback: int = 100):
        super().__init__('price_position')
        self.lookback = lookback
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate where price is within recent range (0-1)"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            recent_data = market_data[-self.lookback:]
            prices = [tick.price for tick in recent_data]
            
            current_price = prices[-1]
            min_price = min(prices)
            max_price = max(prices)
            
            if max_price == min_price:
                position = 0.5
            else:
                position = (current_price - min_price) / (max_price - min_price)
                
            self.total_extraction_time += time.perf_counter() - start_time
            return position
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.5
            
    def get_required_history(self) -> int:
        return self.lookback


class VolumeRatioExtractor(BaseFeatureExtractor):
    """Extract volume ratio vs average"""
    
    def __init__(self, lookback: int = 100):
        super().__init__('volume_ratio')
        self.lookback = lookback
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate current volume vs average volume"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            recent_data = market_data[-self.lookback:]
            volumes = [tick.volume for tick in recent_data]
            
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1])  # Exclude current
            
            if avg_volume == 0:
                ratio = 1.0
            else:
                ratio = current_volume / avg_volume
                
            self.total_extraction_time += time.perf_counter() - start_time
            return ratio
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 1.0
            
    def get_required_history(self) -> int:
        return self.lookback


class VolumeAccelerationExtractor(BaseFeatureExtractor):
    """Extract volume acceleration"""
    
    def __init__(self, short_period: int = 5, long_period: int = 20):
        super().__init__('volume_acceleration')
        self.short_period = short_period
        self.long_period = long_period
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate volume acceleration"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            volumes = [tick.volume for tick in market_data[-self.long_period:]]
            
            short_avg = np.mean(volumes[-self.short_period:])
            long_avg = np.mean(volumes)
            
            if long_avg == 0:
                acceleration = 0.0
            else:
                acceleration = (short_avg - long_avg) / long_avg
                
            self.total_extraction_time += time.perf_counter() - start_time
            return acceleration
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.0
            
    def get_required_history(self) -> int:
        return self.long_period


class SpreadExtractor(BaseFeatureExtractor):
    """Extract bid-ask spread in basis points"""
    
    def __init__(self):
        super().__init__('spread_bps')
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate current spread in basis points"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            current_tick = market_data[-1]
            
            if current_tick.bid <= 0:
                spread_bps = 0.0
            else:
                spread_bps = ((current_tick.ask - current_tick.bid) / current_tick.bid) * 10000
                
            self.total_extraction_time += time.perf_counter() - start_time
            return spread_bps
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.0
            
    def get_required_history(self) -> int:
        return 1


class OrderImbalanceExtractor(BaseFeatureExtractor):
    """Extract order imbalance from bid-ask sizes"""
    
    def __init__(self, lookback: int = 20):
        super().__init__('order_imbalance')
        self.lookback = lookback
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate order imbalance (-1 to 1)"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            recent_data = market_data[-self.lookback:]
            
            # Calculate average bid-ask pressure
            bid_pressure = 0
            ask_pressure = 0
            
            for tick in recent_data:
                mid_price = (tick.bid + tick.ask) / 2
                
                # Use price movement as proxy for order flow
                if tick.price > mid_price:
                    ask_pressure += tick.volume
                else:
                    bid_pressure += tick.volume
                    
            total_pressure = bid_pressure + ask_pressure
            
            if total_pressure == 0:
                imbalance = 0.0
            else:
                imbalance = (bid_pressure - ask_pressure) / total_pressure
                
            self.total_extraction_time += time.perf_counter() - start_time
            return imbalance
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.0
            
    def get_required_history(self) -> int:
        return self.lookback


class RSIExtractor(BaseFeatureExtractor):
    """Extract Relative Strength Index"""
    
    def __init__(self, period: int = 14):
        super().__init__(f'rsi_{period}')
        self.period = period
        
    @staticmethod
    @jit(nopython=True)
    def _calculate_rsi(prices: np.ndarray, period: int) -> float:
        """JIT-compiled RSI calculation"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate RSI"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            prices = np.array([tick.price for tick in market_data[-(self.period + 1):]])
            rsi = self._calculate_rsi(prices, self.period)
            
            self.total_extraction_time += time.perf_counter() - start_time
            return rsi / 100.0  # Normalize to 0-1
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.5
            
    def get_required_history(self) -> int:
        return self.period + 1


class BollingerBandExtractor(BaseFeatureExtractor):
    """Extract Bollinger Band position"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__('bb_position')
        self.period = period
        self.std_dev = std_dev
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate position within Bollinger Bands (-1 to 1)"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            prices = [tick.price for tick in market_data[-self.period:]]
            current_price = prices[-1]
            
            mean = np.mean(prices)
            std = np.std(prices)
            
            upper_band = mean + (self.std_dev * std)
            lower_band = mean - (self.std_dev * std)
            
            if upper_band == lower_band:
                position = 0.0
            else:
                # Normalize to -1 to 1
                position = 2 * (current_price - lower_band) / (upper_band - lower_band) - 1
                position = np.clip(position, -1, 1)
                
            self.total_extraction_time += time.perf_counter() - start_time
            return position
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.0
            
    def get_required_history(self) -> int:
        return self.period


class VolatilityExtractor(BaseFeatureExtractor):
    """Extract price volatility"""
    
    def __init__(self, period: int = 5):
        super().__init__(f'volatility_{period}m')
        self.period = period
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate annualized volatility"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            # Get prices for the period
            ticks_per_period = self.period * 12  # 5-second ticks
            prices = [tick.price for tick in market_data[-ticks_per_period:]]
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Calculate volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252 * 24 * 12)  # Annualize for 5-min bars
            
            self.total_extraction_time += time.perf_counter() - start_time
            return volatility
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.001  # Default low volatility
            
    def get_required_history(self) -> int:
        return self.period * 12


class TrendStrengthExtractor(BaseFeatureExtractor):
    """Extract trend strength using linear regression"""
    
    def __init__(self, period: int = 20):
        super().__init__('trend_strength')
        self.period = period
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate trend strength (R-squared of linear regression)"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            prices = [tick.price for tick in market_data[-self.period:]]
            
            # Linear regression
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            # R-squared indicates trend strength
            r_squared = r_value ** 2
            
            # Add direction information (-1 to 1)
            trend_strength = r_squared if slope > 0 else -r_squared
            
            self.total_extraction_time += time.perf_counter() - start_time
            return trend_strength
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.0
            
    def get_required_history(self) -> int:
        return self.period


class MicrostructureExtractor(BaseFeatureExtractor):
    """Extract microstructure features"""
    
    def __init__(self):
        super().__init__('microstructure_score')
        
    async def extract(self, market_data: List[MarketTick]) -> float:
        """Calculate microstructure health score"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        try:
            recent_ticks = market_data[-20:]
            
            # Calculate various microstructure metrics
            spreads = [(t.ask - t.bid) / t.bid for t in recent_ticks]
            avg_spread = np.mean(spreads)
            spread_stability = 1 / (1 + np.std(spreads))
            
            # Price efficiency (how close trades are to mid price)
            price_efficiency = []
            for tick in recent_ticks:
                mid = (tick.bid + tick.ask) / 2
                efficiency = 1 - abs(tick.price - mid) / mid
                price_efficiency.append(efficiency)
            avg_efficiency = np.mean(price_efficiency)
            
            # Combine into score
            score = 0.5 * spread_stability + 0.5 * avg_efficiency
            
            self.total_extraction_time += time.perf_counter() - start_time
            return score
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Failed to extract {self.name}: {e}")
            return 0.5
            
    def get_required_history(self) -> int:
        return 20


class FeatureEngineeringPipeline:
    """
    Extract features with attention tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_extractors = self._init_extractors()
        self.feature_cache = FeatureCache(
            max_size=self.config.get('cache_size', 1000),
            ttl=self.config.get('cache_ttl', FEATURE_CACHE_TTL)
        )
        self.attention_tracker = FeatureAttentionTracker(
            window_size=self.config.get('attention_window', 1000)
        )
        self.market_data_buffer = deque(maxlen=self.config.get('buffer_size', 500))
        self.min_history_required = self._calculate_min_history()
        
        # Initialize advanced feature engineer (5 Focus Module 1)
        self.advanced_engineer = AdvancedFeatureEngineer(self.config)
        self.use_advanced_features = self.config.get('use_advanced_features', True)
        
        # Performance tracking
        self.extraction_count = 0
        self.cache_hits = 0
        self.total_extraction_time = 0
        
    def _init_extractors(self) -> Dict[str, BaseFeatureExtractor]:
        """Initialize feature extractors"""
        extractors = {
            # Price features
            'price_change_5m': PriceChangeExtractor(period=5),
            'price_position': PricePositionExtractor(),
            
            # Volume features
            'volume_ratio': VolumeRatioExtractor(),
            'volume_acceleration': VolumeAccelerationExtractor(),
            
            # Microstructure features
            'spread_bps': SpreadExtractor(),
            'order_imbalance': OrderImbalanceExtractor(),
            
            # Technical features
            'rsi_14': RSIExtractor(period=14),
            'bb_position': BollingerBandExtractor(),
            
            # Market state features
            'volatility_5m': VolatilityExtractor(period=5),
            'trend_strength': TrendStrengthExtractor(),
            
            # Composite features
            'microstructure_score': MicrostructureExtractor()
        }
        
        # Add custom extractors from config
        custom_extractors = self.config.get('custom_extractors', {})
        extractors.update(custom_extractors)
        
        return extractors
        
    def _calculate_min_history(self) -> int:
        """Calculate minimum history required"""
        return max(
            extractor.get_required_history() 
            for extractor in self.feature_extractors.values()
        )
        
    def _generate_cache_key(self, market_data: List[MarketTick]) -> str:
        """Generate cache key for market data"""
        if not market_data:
            return ""
            
        # Use last N ticks for cache key
        key_ticks = market_data[-10:]
        key_data = {
            'prices': [t.price for t in key_ticks],
            'volumes': [t.volume for t in key_ticks],
            'timestamps': [t.timestamp for t in key_ticks],
            'version': FEATURE_VERSION
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    async def update_buffer(self, tick: MarketTick) -> None:
        """Update internal market data buffer"""
        self.market_data_buffer.append(tick)
        
    async def extract_features(self, market_data: Optional[List[MarketTick]] = None) -> Optional[FeatureSet]:
        """Extract all features with caching"""
        start_time = time.perf_counter()
        self.extraction_count += 1
        
        # Use provided data or internal buffer
        if market_data is None:
            market_data = list(self.market_data_buffer)
            
        # Check if we have enough history
        if len(market_data) < self.min_history_required:
            logger.warning(f"Insufficient data: {len(market_data)} < {self.min_history_required}")
            return None
            
        # Check cache first
        cache_key = self._generate_cache_key(market_data)
        cached_features = await self.feature_cache.get(cache_key)
        
        if cached_features:
            self.cache_hits += 1
            self.total_extraction_time += time.perf_counter() - start_time
            return cached_features
            
        # Extract features
        features = {}
        extraction_times = {}
        quality_scores = {}
        
        # Parallel extraction for independent features
        extraction_tasks = []
        
        for name, extractor in self.feature_extractors.items():
            # Check if we have enough data for this extractor
            if len(market_data) >= extractor.get_required_history():
                task = self._extract_single_feature(name, extractor, market_data)
                extraction_tasks.append(task)
                
        # Wait for all extractions
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Feature extraction failed: {result}")
                continue
                
            name, value, extract_time, quality = result
            features[name] = value
            extraction_times[name] = extract_time
            quality_scores[name] = quality
            
        # Add advanced features (5 Focus Module 1)
        if self.use_advanced_features:
            try:
                # Convert market data to DataFrame for advanced features
                df_data = self._convert_ticks_to_dataframe(market_data)
                advanced_features = self.advanced_engineer.get_all_features(df_data)
                features.update(advanced_features)
                logger.info(f"Added {len(advanced_features)} advanced features")
            except Exception as e:
                logger.warning(f"Failed to extract advanced features: {e}")
        
        # Track for attention learning
        await self.attention_tracker.record(features, extraction_times)
        
        # Create feature set
        feature_set = FeatureSet(
            timestamp=market_data[-1].timestamp,
            features=features,
            metadata={
                'extraction_time': time.perf_counter() - start_time,
                'data_points': len(market_data),
                'cache_key': cache_key
            },
            quality_scores=quality_scores
        )
        
        # Cache results
        await self.feature_cache.set(cache_key, feature_set)
        
        self.total_extraction_time += time.perf_counter() - start_time
        
        return feature_set
        
    async def _extract_single_feature(
        self, 
        name: str, 
        extractor: BaseFeatureExtractor, 
        market_data: List[MarketTick]
    ) -> Tuple[str, float, float, float]:
        """Extract single feature with timing and quality assessment"""
        extract_start = time.perf_counter()
        
        try:
            value = await extractor.extract(market_data)
            extract_time = time.perf_counter() - extract_start
            
            # Assess quality based on extraction time and success
            quality = self._assess_feature_quality(extract_time, extractor)
            
            return name, value, extract_time, quality
            
        except Exception as e:
            logger.error(f"Failed to extract {name}: {e}")
            extract_time = time.perf_counter() - extract_start
            
            return name, extractor.get_default_value(), extract_time, 0.0
            
    def _assess_feature_quality(self, extraction_time: float, extractor: BaseFeatureExtractor) -> float:
        """Assess feature extraction quality"""
        # Start with perfect score
        quality = 1.0
        
        # Penalize slow extraction (>1ms)
        if extraction_time > 0.001:
            time_penalty = min((extraction_time - 0.001) * 100, 0.5)
            quality -= time_penalty
            
        # Consider extractor reliability
        stats = extractor.get_stats()
        quality *= stats['success_rate']
        
        return max(0.0, quality)
    
    def _convert_ticks_to_dataframe(self, market_data: List[MarketTick]) -> pd.DataFrame:
        """Convert market ticks to DataFrame for advanced features"""
        try:
            # Extract data from MarketTick objects
            data = {
                'timestamp': [tick.timestamp for tick in market_data],
                'open': [tick.price for tick in market_data],  # Using price as OHLC for simplicity
                'high': [tick.price * 1.0001 for tick in market_data],  # Mock high
                'low': [tick.price * 0.9999 for tick in market_data],   # Mock low
                'close': [tick.price for tick in market_data],
                'volume': [tick.volume if hasattr(tick, 'volume') else 1000 for tick in market_data]
            }
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to convert ticks to DataFrame: {e}")
            # Return minimal DataFrame
            return pd.DataFrame({
                'close': [tick.price for tick in market_data[-100:]],
                'volume': [1000] * len(market_data[-100:])
            })
        
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get current feature importance scores"""
        return await self.attention_tracker.get_feature_importance()
        
    async def update_correlation_matrix(self) -> None:
        """Update feature correlation matrix"""
        await self.attention_tracker.update_correlation_matrix()
        
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return list(self.feature_extractors.keys())
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        # Get cache stats
        cache_stats = self.feature_cache.get_stats()
        
        # Get extractor stats
        extractor_stats = {}
        for name, extractor in self.feature_extractors.items():
            extractor_stats[name] = extractor.get_stats()
            
        # Get importance scores
        importance_scores = await self.attention_tracker.get_feature_importance()
        
        # Calculate overall metrics
        total_extractions = sum(e.extraction_count for e in self.feature_extractors.values())
        total_failures = sum(e.failure_count for e in self.feature_extractors.values())
        
        return {
            'pipeline_stats': {
                'extraction_count': self.extraction_count,
                'cache_hit_rate': self.cache_hits / max(self.extraction_count, 1),
                'avg_extraction_time': self.total_extraction_time / max(self.extraction_count, 1),
                'buffer_size': len(self.market_data_buffer),
                'min_history_required': self.min_history_required
            },
            'cache_stats': cache_stats,
            'extractor_stats': extractor_stats,
            'feature_importance': importance_scores,
            'total_extractions': total_extractions,
            'total_failures': total_failures,
            'overall_success_rate': (total_extractions - total_failures) / max(total_extractions, 1)
        }


# Example configuration
EXAMPLE_CONFIG = {
    'cache_size': 1000,
    'cache_ttl': 60,
    'attention_window': 1000,
    'buffer_size': 500,
    'custom_extractors': {}
}


# Example usage
async def main():
    """Example usage of FeatureEngineeringPipeline"""
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(EXAMPLE_CONFIG)
    
    # Simulate market data
    market_data = []
    base_price = 50000
    
    for i in range(200):
        tick = MarketTick(
            symbol='BTCUSDT',
            price=base_price + np.random.randn() * 100,
            volume=1000 + np.random.rand() * 500,
            timestamp=time.time() - (200 - i) * 5,  # 5-second intervals
            bid=base_price + np.random.randn() * 100 - 10,
            ask=base_price + np.random.randn() * 100 + 10,
            exchange='binance'
        )
        market_data.append(tick)
        
    # Extract features
    print("Extracting features...")
    features = await pipeline.extract_features(market_data)
    
    if features:
        print(f"\nExtracted {len(features.features)} features:")
        for name, value in features.features.items():
            quality = features.quality_scores.get(name, 0)
            print(f"  {name}: {value:.6f} (quality: {quality:.2f})")
            
        print(f"\nExtraction time: {features.metadata['extraction_time']:.3f}s")
        
    # Get statistics
    stats = await pipeline.get_statistics()
    print(f"\nPipeline Statistics:")
    print(f"  Cache hit rate: {stats['pipeline_stats']['cache_hit_rate']:.2%}")
    print(f"  Avg extraction time: {stats['pipeline_stats']['avg_extraction_time']:.3f}s")
    print(f"  Overall success rate: {stats['overall_success_rate']:.2%}")
    
    # Get feature importance
    importance = await pipeline.get_feature_importance()
    if importance:
        print(f"\nFeature Importance:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_features[:5]:
            print(f"  {name}: {score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
