"""
Advanced Feature Engineering for GridAttention
Implements market microstructure and multi-timeframe features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MicrostructureFeatures:
    """Market microstructure feature container"""
    bid_ask_imbalance: float
    order_flow_toxicity: float
    price_acceleration: float
    volume_weighted_spread: float
    trade_intensity: float
    price_impact: float
    liquidity_consumption: float
    order_book_slope: float

@dataclass
class MultiTimeframeFeatures:
    """Multi-timeframe alignment features"""
    trend_alignment_score: float
    momentum_divergence: float
    volatility_ratio: float
    correlation_matrix: np.ndarray
    timeframe_strength: Dict[str, float]
    regime_consistency: float

class AdvancedFeatureEngineer:
    """Advanced feature calculations for improved trading signals"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_periods = {
            'micro': 20,    # 20 ticks
            'short': 100,   # 100 ticks  
            'medium': 500,  # 500 ticks
            'long': 2000    # 2000 ticks
        }
        
        # Multi-timeframe settings
        self.timeframes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240
        }
        
        # Feature history for calculations
        self.feature_history = deque(maxlen=5000)
        self.microstructure_cache = {}
        
    def calculate_market_microstructure(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure features"""
        features = {}
        
        # Bid-Ask Imbalance
        if 'bid_volume' in data.columns and 'ask_volume' in data.columns:
            total_volume = data['bid_volume'] + data['ask_volume']
            imbalance = (data['bid_volume'] - data['ask_volume']) / (total_volume + 1e-8)
            features['bid_ask_imbalance'] = imbalance.iloc[-1]
            features['bid_ask_imbalance_ma'] = imbalance.rolling(20).mean().iloc[-1]
            
        # Order Flow Toxicity (simplified VPIN)
        if 'volume' in data.columns:
            features['order_flow_toxicity'] = self._calculate_vpin(data)
            
        # Price Acceleration
        if 'close' in data.columns:
            price_velocity = data['close'].diff()
            price_acceleration = price_velocity.diff()
            features['price_acceleration'] = price_acceleration.iloc[-1]
            features['price_jerk'] = price_acceleration.diff().iloc[-1]  # 3rd derivative
            
        # Volume-Weighted Spread
        if all(col in data.columns for col in ['high', 'low', 'volume']):
            spread = data['high'] - data['low']
            features['volume_weighted_spread'] = (
                (spread * data['volume']).sum() / (data['volume'].sum() + 1e-8)
            )
            
        # Trade Intensity
        if 'volume' in data.columns:
            features['trade_intensity'] = self._calculate_trade_intensity(data)
            
        # Price Impact
        if 'close' in data.columns and 'volume' in data.columns:
            features['price_impact'] = self._calculate_price_impact(data)
            
        # Liquidity Consumption Rate
        if 'volume' in data.columns:
            features['liquidity_consumption'] = self._calculate_liquidity_consumption(data)
            
        # Order Book Slope (if level 2 data available)
        if 'bid_price_1' in data.columns and 'ask_price_1' in data.columns:
            features['order_book_slope'] = self._calculate_order_book_slope(data)
            
        return features
    
    def calculate_multi_timeframe_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate multi-timeframe alignment features"""
        features = {}
        
        # Trend alignment across timeframes
        alignment_scores = []
        for tf1, tf2 in [('1m', '5m'), ('5m', '15m'), ('15m', '1h'), ('1h', '4h')]:
            score = self._calculate_trend_alignment(data, tf1, tf2)
            features[f'trend_alignment_{tf1}_{tf2}'] = score
            alignment_scores.append(score)
            
        features['overall_trend_alignment'] = np.mean(alignment_scores)
        
        # Momentum divergence
        features['momentum_divergence'] = self._calculate_momentum_divergence(data)
        
        # Volatility across timeframes
        vol_ratios = self._calculate_volatility_ratios(data)
        features.update(vol_ratios)
        
        # Timeframe strength analysis
        tf_strength = self._analyze_timeframe_strength(data)
        for tf, strength in tf_strength.items():
            features[f'strength_{tf}'] = strength
            
        # Regime consistency
        features['regime_consistency'] = self._calculate_regime_consistency(data)
        
        return features
    
    def calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile features"""
        features = {}
        
        if 'volume' in data.columns and 'close' in data.columns:
            # Volume distribution analysis
            volume_profile = self._build_volume_profile(data)
            
            # Point of Control (POC)
            features['poc_price'] = volume_profile['poc']
            features['poc_distance'] = (data['close'].iloc[-1] - volume_profile['poc']) / data['close'].iloc[-1]
            
            # Value Area
            features['value_area_high'] = volume_profile['vah']
            features['value_area_low'] = volume_profile['val']
            features['in_value_area'] = 1 if volume_profile['val'] <= data['close'].iloc[-1] <= volume_profile['vah'] else 0
            
            # Large trade detection
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()
            large_trade_threshold = volume_mean + 2 * volume_std
            
            features['large_trade_ratio'] = (
                (data['volume'] > large_trade_threshold).sum() / len(data)
            )
            
            # Volume momentum
            features['volume_momentum'] = (
                data['volume'].rolling(20).mean().iloc[-1] / 
                data['volume'].rolling(100).mean().iloc[-1]
            )
            
            # Volume concentration
            features['volume_concentration'] = self._calculate_volume_concentration(data)
            
        return features
    
    def calculate_volatility_regime(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility regime features"""
        features = {}
        
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            
            # Realized volatility over different periods
            for period_name, period in self.lookback_periods.items():
                if len(returns) >= period:
                    features[f'realized_vol_{period_name}'] = (
                        returns.iloc[-period:].std() * np.sqrt(252 * 24 * 12)  # Annualized
                    )
            
            # GARCH volatility forecast
            features['garch_forecast'] = self._calculate_garch_volatility(returns)
            
            # Volatility of volatility
            if len(returns) >= 100:
                rolling_vol = returns.rolling(20).std()
                features['vol_of_vol'] = rolling_vol.iloc[-100:].std()
            
            # Volatility regime classification
            features['volatility_regime'] = self._classify_volatility_regime(features)
            
            # Volatility risk premium
            if 'implied_vol' in data.columns:
                features['vol_risk_premium'] = (
                    data['implied_vol'].iloc[-1] - features.get('realized_vol_short', 0)
                )
                
        return features
    
    def calculate_market_sentiment(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market sentiment features"""
        features = {}
        
        # Price-based sentiment
        if 'close' in data.columns:
            # RSI-based sentiment
            rsi = self._calculate_rsi(data['close'], 14)
            features['rsi_14'] = rsi
            
            # Sentiment score
            if rsi > 70:
                features['sentiment_score'] = (rsi - 70) / 30  # Greed
            elif rsi < 30:
                features['sentiment_score'] = (rsi - 30) / 30  # Fear
            else:
                features['sentiment_score'] = 0  # Neutral
                
            # Put-Call ratio sentiment (if available)
            if 'put_volume' in data.columns and 'call_volume' in data.columns:
                features['put_call_ratio'] = data['put_volume'].iloc[-1] / (data['call_volume'].iloc[-1] + 1e-8)
                features['options_sentiment'] = 1 - features['put_call_ratio']  # Higher = bullish
                
        # Volume-based sentiment
        if 'volume' in data.columns and 'close' in data.columns:
            # On-Balance Volume momentum
            obv = self._calculate_obv(data)
            features['obv_momentum'] = (obv.iloc[-1] - obv.iloc[-20]) / (abs(obv.iloc[-20]) + 1e-8)
            
            # Volume-Price confirmation
            price_change = data['close'].pct_change().iloc[-1]
            volume_change = data['volume'].pct_change().iloc[-1]
            features['volume_price_confirmation'] = np.sign(price_change) * np.sign(volume_change)
            
        # Market breadth (if multiple assets)
        if 'market_breadth' in data.columns:
            features['breadth_momentum'] = data['market_breadth'].rolling(20).mean().iloc[-1]
            
        return features
    
    # Helper methods
    def _calculate_vpin(self, data: pd.DataFrame) -> float:
        """Calculate Volume-Synchronized Probability of Informed Trading"""
        if len(data) < 50:
            return 0.5
            
        # Simplified VPIN calculation
        returns = data['close'].pct_change().dropna()
        volume = data['volume'].iloc[1:]
        
        # Classify volume as buy or sell
        buy_volume = volume[returns > 0].sum()
        sell_volume = volume[returns <= 0].sum()
        total_volume = buy_volume + sell_volume + 1e-8
        
        # Order imbalance
        order_imbalance = abs(buy_volume - sell_volume) / total_volume
        
        return order_imbalance
    
    def _calculate_trade_intensity(self, data: pd.DataFrame) -> float:
        """Calculate trade intensity metric"""
        if len(data) < 20:
            return 0.0
            
        # Volume acceleration
        volume_ma_short = data['volume'].rolling(5).mean()
        volume_ma_long = data['volume'].rolling(20).mean()
        
        intensity = (volume_ma_short.iloc[-1] / (volume_ma_long.iloc[-1] + 1e-8)) - 1
        return np.clip(intensity, -1, 2)
    
    def _calculate_price_impact(self, data: pd.DataFrame) -> float:
        """Calculate price impact of trades"""
        if len(data) < 10:
            return 0.0
            
        # Kyle's lambda approximation
        returns = data['close'].pct_change().dropna()
        signed_volume = data['volume'].iloc[1:] * np.sign(returns)
        
        # Simple regression of returns on signed volume
        if len(returns) >= 10:
            correlation = np.corrcoef(returns[-10:], signed_volume[-10:])[0, 1]
            price_impact = abs(correlation) * returns.std() / signed_volume.std()
            return price_impact
        
        return 0.0
    
    def _calculate_liquidity_consumption(self, data: pd.DataFrame) -> float:
        """Calculate rate of liquidity consumption"""
        if len(data) < 20:
            return 0.0
            
        # Volume relative to recent average
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].iloc[-20:].mean()
        
        consumption_rate = current_volume / (avg_volume + 1e-8)
        return np.clip(consumption_rate, 0, 5)
    
    def _calculate_order_book_slope(self, data: pd.DataFrame) -> float:
        """Calculate order book slope from level 2 data"""
        # Simplified calculation using best bid/ask
        bid_price = data['bid_price_1'].iloc[-1]
        ask_price = data['ask_price_1'].iloc[-1]
        bid_size = data.get('bid_size_1', pd.Series([1])).iloc[-1]
        ask_size = data.get('ask_size_1', pd.Series([1])).iloc[-1]
        
        # Slope approximation
        price_diff = ask_price - bid_price
        size_avg = (bid_size + ask_size) / 2
        
        slope = price_diff / (size_avg + 1e-8)
        return slope
    
    def _calculate_trend_alignment(self, data: pd.DataFrame, tf1: str, tf2: str) -> float:
        """Calculate trend alignment between timeframes"""
        if 'close' not in data.columns or len(data) < self.timeframes[tf2]:
            return 0.0
            
        # Simple moving averages for different timeframes
        ma1 = data['close'].rolling(self.timeframes[tf1]).mean()
        ma2 = data['close'].rolling(self.timeframes[tf2]).mean()
        
        if ma1.iloc[-1] > ma2.iloc[-1]:
            alignment = 1.0  # Bullish alignment
        else:
            alignment = -1.0  # Bearish alignment
            
        # Strength based on distance
        distance = abs(ma1.iloc[-1] - ma2.iloc[-1]) / ma2.iloc[-1]
        strength = min(distance * 100, 1.0)
        
        return alignment * strength
    
    def _calculate_momentum_divergence(self, data: pd.DataFrame) -> float:
        """Calculate momentum divergence across timeframes"""
        if 'close' not in data.columns or len(data) < 60:
            return 0.0
            
        # Short-term momentum
        momentum_short = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
        
        # Long-term momentum
        momentum_long = (data['close'].iloc[-1] - data['close'].iloc[-60]) / data['close'].iloc[-60]
        
        # Divergence
        divergence = momentum_short - momentum_long
        return divergence
    
    def _calculate_volatility_ratios(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility ratios across timeframes"""
        ratios = {}
        
        if 'close' not in data.columns:
            return ratios
            
        returns = data['close'].pct_change().dropna()
        
        # Short-term vs long-term volatility
        if len(returns) >= 100:
            vol_5 = returns.iloc[-5:].std()
            vol_20 = returns.iloc[-20:].std()
            vol_100 = returns.iloc[-100:].std()
            
            ratios['vol_ratio_5_20'] = vol_5 / (vol_20 + 1e-8)
            ratios['vol_ratio_20_100'] = vol_20 / (vol_100 + 1e-8)
            ratios['vol_expansion'] = vol_5 / (vol_100 + 1e-8)
            
        return ratios
    
    def _analyze_timeframe_strength(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze strength of each timeframe"""
        strength = {}
        
        if 'close' not in data.columns:
            return strength
            
        for tf_name, tf_period in self.timeframes.items():
            if len(data) >= tf_period * 2:
                # Trend strength for timeframe
                ma = data['close'].rolling(tf_period).mean()
                price_above_ma = (data['close'] > ma).sum() / len(data)
                
                # Momentum strength
                momentum = (ma.iloc[-1] - ma.iloc[-tf_period]) / ma.iloc[-tf_period]
                
                # Combined strength
                strength[tf_name] = price_above_ma * abs(momentum)
                
        return strength
    
    def _calculate_regime_consistency(self, data: pd.DataFrame) -> float:
        """Calculate regime consistency across timeframes"""
        if 'close' not in data.columns or len(data) < 240:
            return 0.5
            
        # Check trend direction across timeframes
        directions = []
        for tf_period in [5, 15, 60, 240]:
            if len(data) >= tf_period:
                ma = data['close'].rolling(tf_period).mean()
                direction = 1 if ma.iloc[-1] > ma.iloc[-tf_period] else -1
                directions.append(direction)
                
        # Consistency score
        if directions:
            consistency = abs(sum(directions)) / len(directions)
            return consistency
            
        return 0.5
    
    def _build_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Build volume profile and find key levels"""
        prices = data['close'].values
        volumes = data['volume'].values
        
        # Create price bins
        price_range = prices.max() - prices.min()
        n_bins = min(50, len(data) // 10)
        bins = np.linspace(prices.min(), prices.max(), n_bins)
        
        # Accumulate volume in each bin
        volume_profile = np.zeros(n_bins - 1)
        for i in range(len(prices)):
            bin_idx = np.digitize(prices[i], bins) - 1
            if 0 <= bin_idx < n_bins - 1:
                volume_profile[bin_idx] += volumes[i]
                
        # Find Point of Control (highest volume price)
        poc_idx = np.argmax(volume_profile)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Calculate Value Area (70% of volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.7
        
        # Expand from POC until 70% volume captured
        accumulated_volume = volume_profile[poc_idx]
        low_idx, high_idx = poc_idx, poc_idx
        
        while accumulated_volume < target_volume and (low_idx > 0 or high_idx < len(volume_profile) - 1):
            if low_idx > 0 and high_idx < len(volume_profile) - 1:
                if volume_profile[low_idx - 1] > volume_profile[high_idx + 1]:
                    low_idx -= 1
                    accumulated_volume += volume_profile[low_idx]
                else:
                    high_idx += 1
                    accumulated_volume += volume_profile[high_idx]
            elif low_idx > 0:
                low_idx -= 1
                accumulated_volume += volume_profile[low_idx]
            elif high_idx < len(volume_profile) - 1:
                high_idx += 1
                accumulated_volume += volume_profile[high_idx]
                
        val = (bins[low_idx] + bins[low_idx + 1]) / 2
        vah = (bins[high_idx] + bins[high_idx + 1]) / 2
        
        return {
            'poc': poc_price,
            'val': val,
            'vah': vah,
            'profile': volume_profile
        }
    
    def _calculate_volume_concentration(self, data: pd.DataFrame) -> float:
        """Calculate how concentrated volume is in recent bars"""
        if len(data) < 20:
            return 0.5
            
        recent_volume = data['volume'].iloc[-5:].sum()
        total_volume = data['volume'].iloc[-20:].sum()
        
        concentration = recent_volume / (total_volume + 1e-8)
        return concentration
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        """Simple GARCH(1,1) volatility forecast"""
        if len(returns) < 100:
            return returns.std()
            
        # Parameters (simplified)
        omega = 0.00001
        alpha = 0.1
        beta = 0.85
        
        # Initialize
        variance = returns.var()
        
        # GARCH recursion
        for r in returns.iloc[-20:]:
            variance = omega + alpha * r**2 + beta * variance
            
        return np.sqrt(variance)
    
    def _classify_volatility_regime(self, features: Dict[str, float]) -> str:
        """Classify current volatility regime"""
        current_vol = features.get('realized_vol_short', 0)
        long_vol = features.get('realized_vol_long', 0)
        
        if current_vol == 0 or long_vol == 0:
            return 'unknown'
            
        vol_ratio = current_vol / long_vol
        
        if vol_ratio > 1.5:
            return 'expanding'
        elif vol_ratio < 0.7:
            return 'contracting'
        elif current_vol > 0.3:  # 30% annualized
            return 'high'
        elif current_vol < 0.1:  # 10% annualized
            return 'low'
        else:
            return 'normal'
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    def get_all_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all advanced features"""
        all_features = {}
        
        try:
            # Market microstructure features
            microstructure = self.calculate_market_microstructure(market_data)
            all_features.update({f'micro_{k}': v for k, v in microstructure.items()})
            
            # Multi-timeframe features
            mtf = self.calculate_multi_timeframe_features(market_data)
            all_features.update({f'mtf_{k}': v for k, v in mtf.items()})
            
            # Volume profile features
            volume_profile = self.calculate_volume_profile(market_data)
            all_features.update({f'vp_{k}': v for k, v in volume_profile.items()})
            
            # Volatility regime features
            vol_regime = self.calculate_volatility_regime(market_data)
            all_features.update({f'vol_{k}': v for k, v in vol_regime.items()})
            
            # Market sentiment features
            sentiment = self.calculate_market_sentiment(market_data)
            all_features.update({f'sent_{k}': v for k, v in sentiment.items()})
            
            # Add timestamp
            all_features['feature_timestamp'] = pd.Timestamp.now()
            
            # Cache features
            self.feature_history.append(all_features)
            
        except Exception as e:
            logger.error(f"Error calculating advanced features: {e}")
            return {}
            
        return all_features