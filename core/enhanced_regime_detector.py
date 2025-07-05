"""
Enhanced Market Regime Detection with Sub-regimes and Transitions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

class SubRegime(Enum):
    """Detailed sub-regime classifications"""
    # Trending sub-regimes
    STRONG_TREND_UP = "strong_trend_up"
    MODERATE_TREND_UP = "moderate_trend_up"
    WEAK_TREND_UP = "weak_trend_up"
    STRONG_TREND_DOWN = "strong_trend_down"
    MODERATE_TREND_DOWN = "moderate_trend_down"
    WEAK_TREND_DOWN = "weak_trend_down"
    
    # Ranging sub-regimes
    TIGHT_RANGE = "tight_range"
    NORMAL_RANGE = "normal_range"
    WIDE_RANGE = "wide_range"
    
    # Volatile sub-regimes
    INCREASING_VOL = "increasing_vol"
    DECREASING_VOL = "decreasing_vol"
    SPIKE_VOL = "spike_vol"
    
    # Transition states
    BREAKOUT_PENDING = "breakout_pending"
    TREND_EXHAUSTION = "trend_exhaustion"
    RANGE_COMPRESSION = "range_compression"

@dataclass
class RegimeContext:
    """Context information for regime detection"""
    session: str
    hour: int
    day_of_week: int
    volume_profile: str
    news_pending: bool
    correlation_state: str
    vix_level: Optional[float] = None

@dataclass
class TransitionWarning:
    """Regime transition warning information"""
    warning_type: str
    message: str
    confidence: float
    timestamp: pd.Timestamp
    indicators: Dict[str, float]

@dataclass
class RegimeAnalysis:
    """Complete regime analysis result"""
    base_regime: str
    sub_regime: str
    confidence: float
    transition_probability: float
    warnings: List[TransitionWarning]
    context_factors: Dict[str, Any]
    supporting_indicators: Dict[str, float]

class EnhancedRegimeDetector:
    """Advanced regime detection with early transition warnings"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regime_history = deque(maxlen=config.get('history_size', 100))
        self.sub_regime_history = deque(maxlen=config.get('history_size', 100))
        self.transition_warnings = deque(maxlen=config.get('warning_history', 10))
        
        # Thresholds for regime classification
        self.trend_thresholds = {
            'strong': 0.8,
            'moderate': 0.5,
            'weak': 0.3
        }
        
        self.volatility_thresholds = {
            'spike': 2.0,      # 2x normal volatility
            'high': 1.5,       # 1.5x normal
            'normal': 1.0,     # baseline
            'low': 0.7         # 0.7x normal
        }
        
        self.range_thresholds = {
            'tight': 0.5,      # 50% of ATR
            'normal': 1.0,     # 1x ATR
            'wide': 2.0        # 2x ATR
        }
        
        # Market session definitions (UTC)
        self.market_sessions = {
            'ASIAN': (22, 7),
            'LONDON': (7, 16),
            'US': (13, 22),
            'US_OPEN': (13, 14),  # First hour
            'OVERLAP': (13, 16)   # London/US overlap
        }
        
        # Transition detection parameters
        self.transition_indicators = {
            'volume_surge': 2.0,           # Volume multiplier
            'volatility_expansion': 1.5,    # Volatility expansion ratio
            'momentum_acceleration': 0.001, # Momentum change threshold
            'range_break': 0.02,           # Range break threshold
            'correlation_break': 0.3       # Correlation change threshold
        }
        
        # Machine learning models placeholders
        self.ml_models = {}
        self.feature_importance = {}
        
    def detect_regime_with_context(
        self, 
        features: Dict[str, float],
        timestamp: pd.Timestamp,
        market_data: Optional[pd.DataFrame] = None
    ) -> RegimeAnalysis:
        """Detect regime with full context and sub-regime classification"""
        
        # Extract context
        context = self._extract_context(timestamp, features)
        
        # Base regime detection
        base_regime, regime_confidence = self._detect_base_regime(features)
        
        # Sub-regime classification
        sub_regime = self._classify_sub_regime(base_regime, features)
        
        # Apply context adjustments
        adjusted_regime, adjusted_sub_regime = self._apply_context_adjustments(
            base_regime, sub_regime, features, context
        )
        
        # Detect transition warnings
        warnings = self._detect_transitions(features, market_data)
        
        # Calculate transition probability
        transition_prob = self._calculate_transition_probability(warnings, features)
        
        # Get supporting indicators
        supporting_indicators = self._get_supporting_indicators(features, adjusted_regime)
        
        # Update history
        self._update_history(adjusted_regime, adjusted_sub_regime, regime_confidence, timestamp, features)
        
        # Create analysis result
        analysis = RegimeAnalysis(
            base_regime=adjusted_regime,
            sub_regime=adjusted_sub_regime,
            confidence=regime_confidence,
            transition_probability=transition_prob,
            warnings=warnings,
            context_factors={
                'session': context.session,
                'hour': context.hour,
                'day_of_week': context.day_of_week
            },
            supporting_indicators=supporting_indicators
        )
        
        return analysis
    
    def _detect_base_regime(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Enhanced base regime detection with multiple indicators"""
        
        # Calculate regime scores
        regime_scores = {
            'TRENDING': self._calculate_trend_score(features),
            'RANGING': self._calculate_range_score(features),
            'VOLATILE': self._calculate_volatile_score(features)
        }
        
        # Get regime with highest score
        detected_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[detected_regime]
        
        # Apply minimum confidence threshold
        if confidence < 0.4:
            detected_regime = 'UNCERTAIN'
            confidence = 1 - confidence
            
        return detected_regime, confidence
    
    def _calculate_trend_score(self, features: Dict[str, float]) -> float:
        """Calculate trending regime score"""
        score_components = []
        
        # Trend strength
        trend_strength = abs(features.get('trend_strength', 0))
        score_components.append(min(trend_strength / 0.8, 1.0) * 0.3)
        
        # Momentum consistency
        momentum = features.get('momentum', 0)
        momentum_ma = features.get('momentum_ma', 0)
        consistency = 1 - abs(momentum - momentum_ma) / (abs(momentum) + 0.001)
        score_components.append(consistency * 0.2)
        
        # ADX indicator
        adx = features.get('adx', 0)
        adx_score = min(adx / 40, 1.0)  # ADX > 40 indicates strong trend
        score_components.append(adx_score * 0.2)
        
        # Price position relative to moving averages
        price_ma_ratio = features.get('price_ma_ratio', 1.0)
        ma_score = min(abs(price_ma_ratio - 1.0) * 10, 1.0)
        score_components.append(ma_score * 0.15)
        
        # Volume trend confirmation
        volume_trend = features.get('volume_trend_confirmation', 0)
        score_components.append(abs(volume_trend) * 0.15)
        
        return sum(score_components)
    
    def _calculate_range_score(self, features: Dict[str, float]) -> float:
        """Calculate ranging regime score"""
        score_components = []
        
        # Low trend strength
        trend_strength = abs(features.get('trend_strength', 0))
        score_components.append((1 - min(trend_strength / 0.3, 1.0)) * 0.25)
        
        # Price oscillation around mean
        price_std = features.get('price_std_ratio', 0)
        oscillation_score = 1 - abs(price_std - 1.0)
        score_components.append(oscillation_score * 0.25)
        
        # Bollinger Band position
        bb_position = features.get('bb_position', 0.5)
        bb_score = 1 - abs(bb_position - 0.5) * 2  # Closer to middle = higher score
        score_components.append(bb_score * 0.2)
        
        # Low ADX
        adx = features.get('adx', 0)
        low_adx_score = 1 - min(adx / 25, 1.0)  # ADX < 25 indicates ranging
        score_components.append(low_adx_score * 0.15)
        
        # Support/Resistance bounces
        sr_bounces = features.get('support_resistance_bounces', 0)
        score_components.append(min(sr_bounces / 5, 1.0) * 0.15)
        
        return sum(score_components)
    
    def _calculate_volatile_score(self, features: Dict[str, float]) -> float:
        """Calculate volatile regime score"""
        score_components = []
        
        # High volatility
        current_vol = features.get('volatility_5m', 0)
        vol_ma = features.get('volatility_ma', 0.001)
        vol_ratio = current_vol / vol_ma
        score_components.append(min(vol_ratio / 2.0, 1.0) * 0.3)
        
        # Volatility acceleration
        vol_acceleration = features.get('volatility_acceleration', 0)
        score_components.append(min(abs(vol_acceleration) * 100, 1.0) * 0.2)
        
        # Wide Bollinger Bands
        bb_width = features.get('bb_width', 0)
        bb_width_ma = features.get('bb_width_ma', 0.001)
        bb_expansion = bb_width / bb_width_ma
        score_components.append(min(bb_expansion / 1.5, 1.0) * 0.2)
        
        # Price gaps
        gap_frequency = features.get('gap_frequency', 0)
        score_components.append(min(gap_frequency, 1.0) * 0.15)
        
        # Erratic volume
        volume_volatility = features.get('volume_volatility', 0)
        score_components.append(min(volume_volatility / 0.5, 1.0) * 0.15)
        
        return sum(score_components)
    
    def _classify_sub_regime(self, base_regime: str, features: Dict[str, float]) -> str:
        """Classify detailed sub-regime within base regime"""
        
        if base_regime == 'TRENDING':
            return self._classify_trending_sub_regime(features)
        elif base_regime == 'RANGING':
            return self._classify_ranging_sub_regime(features)
        elif base_regime == 'VOLATILE':
            return self._classify_volatile_sub_regime(features)
        else:
            return SubRegime.NORMAL_RANGE.value  # Default
    
    def _classify_trending_sub_regime(self, features: Dict[str, float]) -> str:
        """Classify trending sub-regimes"""
        trend_strength = features.get('trend_strength', 0)
        momentum = features.get('momentum', 0)
        volume_confirmation = features.get('volume_trend_confirmation', 0)
        
        # Calculate trend intensity
        intensity = abs(trend_strength) * abs(momentum) * abs(volume_confirmation)
        
        # Check for exhaustion
        if self._detect_trend_exhaustion(features):
            return SubRegime.TREND_EXHAUSTION.value
        
        # Determine direction and strength
        if trend_strength > 0:  # Uptrend
            if intensity > self.trend_thresholds['strong']:
                return SubRegime.STRONG_TREND_UP.value
            elif intensity > self.trend_thresholds['moderate']:
                return SubRegime.MODERATE_TREND_UP.value
            else:
                return SubRegime.WEAK_TREND_UP.value
        else:  # Downtrend
            if intensity > self.trend_thresholds['strong']:
                return SubRegime.STRONG_TREND_DOWN.value
            elif intensity > self.trend_thresholds['moderate']:
                return SubRegime.MODERATE_TREND_DOWN.value
            else:
                return SubRegime.WEAK_TREND_DOWN.value
    
    def _classify_ranging_sub_regime(self, features: Dict[str, float]) -> str:
        """Classify ranging sub-regimes"""
        atr = features.get('atr_14', 0.001)
        range_width = features.get('range_width', atr)
        bb_width = features.get('bb_width', atr * 2)
        
        # Calculate relative range
        relative_range = range_width / atr
        
        # Check for compression
        if self._detect_range_compression(features):
            return SubRegime.RANGE_COMPRESSION.value
        
        # Check for breakout pending
        if self._detect_breakout_pending(features):
            return SubRegime.BREAKOUT_PENDING.value
        
        # Classify range type
        if relative_range < self.range_thresholds['tight']:
            return SubRegime.TIGHT_RANGE.value
        elif relative_range > self.range_thresholds['wide']:
            return SubRegime.WIDE_RANGE.value
        else:
            return SubRegime.NORMAL_RANGE.value
    
    def _classify_volatile_sub_regime(self, features: Dict[str, float]) -> str:
        """Classify volatile sub-regimes"""
        current_vol = features.get('volatility_5m', 0)
        vol_ma = features.get('volatility_ma', 0.001)
        vol_acceleration = features.get('volatility_acceleration', 0)
        
        vol_ratio = current_vol / vol_ma
        
        # Check volatility direction
        if vol_acceleration > 0.0001:
            return SubRegime.INCREASING_VOL.value
        elif vol_acceleration < -0.0001:
            return SubRegime.DECREASING_VOL.value
        elif vol_ratio > self.volatility_thresholds['spike']:
            return SubRegime.SPIKE_VOL.value
        else:
            return SubRegime.INCREASING_VOL.value  # Default volatile
    
    def _detect_transitions(
        self, 
        features: Dict[str, float],
        market_data: Optional[pd.DataFrame] = None
    ) -> List[TransitionWarning]:
        """Detect potential regime transitions with early warning"""
        warnings = []
        
        # Volume surge detection
        volume_surge_warning = self._detect_volume_surge(features)
        if volume_surge_warning:
            warnings.append(volume_surge_warning)
        
        # Volatility expansion
        vol_expansion_warning = self._detect_volatility_expansion(features)
        if vol_expansion_warning:
            warnings.append(vol_expansion_warning)
        
        # Momentum shift
        momentum_warning = self._detect_momentum_shift(features)
        if momentum_warning:
            warnings.append(momentum_warning)
        
        # Range break
        range_break_warning = self._detect_range_break(features)
        if range_break_warning:
            warnings.append(range_break_warning)
        
        # Correlation breakdown
        correlation_warning = self._detect_correlation_breakdown(features)
        if correlation_warning:
            warnings.append(correlation_warning)
        
        # Pattern completion
        if market_data is not None:
            pattern_warning = self._detect_pattern_completion(features, market_data)
            if pattern_warning:
                warnings.append(pattern_warning)
        
        # Store warnings
        for warning in warnings:
            self.transition_warnings.append(warning)
        
        return warnings
    
    def _detect_volume_surge(self, features: Dict[str, float]) -> Optional[TransitionWarning]:
        """Detect unusual volume that may signal regime change"""
        volume_ratio = features.get('volume_ratio', 1.0)
        volume_acceleration = features.get('volume_acceleration', 0)
        
        if volume_ratio > self.transition_indicators['volume_surge']:
            confidence = min((volume_ratio - 1.5) / 2.0, 1.0)
            
            return TransitionWarning(
                warning_type='volume_surge',
                message=f'Unusual volume detected: {volume_ratio:.1f}x normal',
                confidence=confidence,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'volume_ratio': volume_ratio,
                    'volume_acceleration': volume_acceleration
                }
            )
        
        return None
    
    def _detect_volatility_expansion(self, features: Dict[str, float]) -> Optional[TransitionWarning]:
        """Detect volatility expansion that may precede regime change"""
        bb_width = features.get('bb_width', 0)
        bb_width_ma = features.get('bb_width_ma', 0.001)
        vol_expansion = bb_width / bb_width_ma
        
        if vol_expansion > self.transition_indicators['volatility_expansion']:
            confidence = min((vol_expansion - 1.3) / 1.0, 0.9)
            
            return TransitionWarning(
                warning_type='volatility_expansion',
                message='Volatility expanding - possible breakout',
                confidence=confidence,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'bb_expansion': vol_expansion,
                    'bb_width': bb_width
                }
            )
        
        return None
    
    def _detect_momentum_shift(self, features: Dict[str, float]) -> Optional[TransitionWarning]:
        """Detect momentum acceleration that signals trend change"""
        momentum_acceleration = features.get('momentum_acceleration', 0)
        rsi = features.get('rsi_14', 50)
        
        if abs(momentum_acceleration) > self.transition_indicators['momentum_acceleration']:
            # Higher confidence if RSI supports the shift
            base_confidence = min(abs(momentum_acceleration) * 500, 0.7)
            rsi_factor = 1.0
            
            if (momentum_acceleration > 0 and rsi < 30) or (momentum_acceleration < 0 and rsi > 70):
                rsi_factor = 1.3  # Reversal signal
            
            confidence = min(base_confidence * rsi_factor, 0.9)
            
            direction = "bullish" if momentum_acceleration > 0 else "bearish"
            
            return TransitionWarning(
                warning_type='momentum_shift',
                message=f'Momentum shift detected - turning {direction}',
                confidence=confidence,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'momentum_acceleration': momentum_acceleration,
                    'rsi': rsi
                }
            )
        
        return None
    
    def _detect_range_break(self, features: Dict[str, float]) -> Optional[TransitionWarning]:
        """Detect potential range breakout"""
        price_position = features.get('range_position', 0.5)
        volume_ratio = features.get('volume_ratio', 1.0)
        atr = features.get('atr_14', 0.001)
        
        # Check if price is near range boundary
        if price_position > 0.9 or price_position < 0.1:
            # Higher confidence with volume confirmation
            base_confidence = 0.6
            if volume_ratio > 1.5:
                base_confidence *= 1.2
            
            confidence = min(base_confidence, 0.85)
            direction = "upper" if price_position > 0.9 else "lower"
            
            return TransitionWarning(
                warning_type='range_break',
                message=f'Price testing {direction} range boundary',
                confidence=confidence,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'range_position': price_position,
                    'volume_ratio': volume_ratio,
                    'atr': atr
                }
            )
        
        return None
    
    def _detect_correlation_breakdown(self, features: Dict[str, float]) -> Optional[TransitionWarning]:
        """Detect correlation changes that signal regime shift"""
        correlation_change = features.get('correlation_change', 0)
        
        if abs(correlation_change) > self.transition_indicators['correlation_break']:
            confidence = min(abs(correlation_change) / 0.5, 0.8)
            
            return TransitionWarning(
                warning_type='correlation_breakdown',
                message='Market correlation structure changing',
                confidence=confidence,
                timestamp=pd.Timestamp.now(),
                indicators={
                    'correlation_change': correlation_change
                }
            )
        
        return None
    
    def _detect_pattern_completion(
        self, 
        features: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Optional[TransitionWarning]:
        """Detect chart pattern completions"""
        # Simplified pattern detection
        if len(market_data) < 50:
            return None
        
        # Check for triangle pattern
        highs = market_data['high'].iloc[-20:]
        lows = market_data['low'].iloc[-20:]
        
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Converging lines indicate triangle
        if high_slope < 0 and low_slope > 0 and abs(high_slope + low_slope) < 0.0001:
            current_range = highs.iloc[-1] - lows.iloc[-1]
            initial_range = highs.iloc[0] - lows.iloc[0]
            compression = 1 - (current_range / initial_range)
            
            if compression > 0.5:
                return TransitionWarning(
                    warning_type='pattern_completion',
                    message='Triangle pattern nearing apex',
                    confidence=min(compression, 0.8),
                    timestamp=pd.Timestamp.now(),
                    indicators={
                        'pattern_type': 'triangle',
                        'compression': compression
                    }
                )
        
        return None
    
    def _calculate_transition_probability(
        self, 
        warnings: List[TransitionWarning],
        features: Dict[str, float]
    ) -> float:
        """Calculate overall probability of regime transition"""
        
        if not warnings:
            # Check historical volatility for baseline probability
            vol_percentile = features.get('volatility_percentile', 50)
            return max(0.1, (vol_percentile - 50) / 100)
        
        # Weight warnings by confidence and recency
        current_time = pd.Timestamp.now()
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for warning in warnings:
            # Time decay (warnings older than 10 minutes have less weight)
            time_diff = (current_time - warning.timestamp).total_seconds() / 60
            time_weight = max(0.1, 1.0 - time_diff / 10)
            
            # Type weighting
            type_weights = {
                'volume_surge': 1.2,
                'volatility_expansion': 1.1,
                'momentum_shift': 1.0,
                'range_break': 1.3,
                'correlation_breakdown': 0.9,
                'pattern_completion': 1.4
            }
            type_weight = type_weights.get(warning.warning_type, 1.0)
            
            weight = time_weight * type_weight
            weighted_confidence += warning.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_confidence = weighted_confidence / total_weight
            
            # Adjust for multiple warnings
            multi_warning_factor = min(1.0 + len(warnings) * 0.1, 1.5)
            
            return min(avg_confidence * multi_warning_factor, 0.95)
        
        return 0.1
    
    def _extract_context(
        self, 
        timestamp: pd.Timestamp,
        features: Dict[str, float]
    ) -> RegimeContext:
        """Extract market context for regime detection"""
        
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        # Determine market session
        session = self._get_market_session(hour)
        
        # Volume profile
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            volume_profile = 'high'
        elif volume_ratio < 0.5:
            volume_profile = 'low'
        else:
            volume_profile = 'normal'
        
        # Correlation state
        avg_correlation = features.get('avg_correlation', 0.5)
        if avg_correlation > 0.7:
            correlation_state = 'high'
        elif avg_correlation < 0.3:
            correlation_state = 'low'
        else:
            correlation_state = 'normal'
        
        # News/event pending (simplified)
        news_pending = hour in [13, 14, 15]  # Major news times
        
        # VIX level if available
        vix_level = features.get('vix_level', None)
        
        return RegimeContext(
            session=session,
            hour=hour,
            day_of_week=day_of_week,
            volume_profile=volume_profile,
            news_pending=news_pending,
            correlation_state=correlation_state,
            vix_level=vix_level
        )
    
    def _get_market_session(self, hour: int) -> str:
        """Determine current market session"""
        for session, (start, end) in self.market_sessions.items():
            if start <= hour < end:
                return session
        return 'OFF_HOURS'
    
    def _apply_context_adjustments(
        self,
        base_regime: str,
        sub_regime: str,
        features: Dict[str, float],
        context: RegimeContext
    ) -> Tuple[str, str]:
        """Apply context-based adjustments to regime detection"""
        
        adjusted_regime = base_regime
        adjusted_sub_regime = sub_regime
        
        # Session-based adjustments
        if context.session == 'ASIAN':
            # Asian session typically less volatile
            if base_regime == 'VOLATILE' and features.get('volatility_5m', 0) < features.get('volatility_avg', 0) * 1.5:
                adjusted_regime = 'RANGING'
                adjusted_sub_regime = SubRegime.NORMAL_RANGE.value
        
        elif context.session == 'US_OPEN':
            # US open often breaks overnight ranges
            if base_regime == 'RANGING' and self._calculate_transition_probability([], features) > 0.5:
                adjusted_sub_regime = SubRegime.BREAKOUT_PENDING.value
        
        elif context.session == 'OVERLAP':
            # London/US overlap has highest liquidity
            if base_regime == 'RANGING' and context.volume_profile == 'high':
                adjusted_sub_regime = SubRegime.WIDE_RANGE.value
        
        # Day of week adjustments
        if context.day_of_week == 4:  # Friday
            # Reduced activity on Fridays
            if sub_regime in [SubRegime.STRONG_TREND_UP.value, SubRegime.STRONG_TREND_DOWN.value]:
                adjusted_sub_regime = sub_regime.replace('strong', 'moderate')
        
        # News event adjustments
        if context.news_pending:
            if base_regime == 'RANGING':
                adjusted_sub_regime = SubRegime.BREAKOUT_PENDING.value
            elif base_regime == 'TRENDING' and 'weak' in sub_regime:
                adjusted_sub_regime = SubRegime.TREND_EXHAUSTION.value
        
        # VIX level adjustments
        if context.vix_level:
            if context.vix_level > 30 and base_regime != 'VOLATILE':
                # High VIX suggests volatility
                adjusted_regime = 'VOLATILE'
                adjusted_sub_regime = SubRegime.SPIKE_VOL.value
            elif context.vix_level < 15 and base_regime == 'VOLATILE':
                # Low VIX suggests calm
                adjusted_regime = 'RANGING'
                adjusted_sub_regime = SubRegime.TIGHT_RANGE.value
        
        return adjusted_regime, adjusted_sub_regime
    
    def _get_supporting_indicators(
        self, 
        features: Dict[str, float],
        regime: str
    ) -> Dict[str, float]:
        """Get key indicators supporting the regime classification"""
        
        indicators = {}
        
        if regime == 'TRENDING':
            indicators['trend_strength'] = features.get('trend_strength', 0)
            indicators['adx'] = features.get('adx', 0)
            indicators['momentum'] = features.get('momentum', 0)
            indicators['volume_trend_confirmation'] = features.get('volume_trend_confirmation', 0)
            
        elif regime == 'RANGING':
            indicators['atr'] = features.get('atr_14', 0)
            indicators['bb_position'] = features.get('bb_position', 0.5)
            indicators['support_resistance_bounces'] = features.get('support_resistance_bounces', 0)
            indicators['range_width'] = features.get('range_width', 0)
            
        elif regime == 'VOLATILE':
            indicators['volatility_5m'] = features.get('volatility_5m', 0)
            indicators['bb_width'] = features.get('bb_width', 0)
            indicators['volatility_acceleration'] = features.get('volatility_acceleration', 0)
            indicators['gap_frequency'] = features.get('gap_frequency', 0)
        
        return indicators
    
    def _detect_trend_exhaustion(self, features: Dict[str, float]) -> bool:
        """Detect if trend is showing exhaustion signs"""
        # Divergence between price and momentum
        price_momentum_divergence = features.get('price_momentum_divergence', 0)
        
        # Decreasing volume on trend continuation
        volume_trend_divergence = features.get('volume_trend_divergence', 0)
        
        # Extreme RSI
        rsi = features.get('rsi_14', 50)
        
        exhaustion_signals = 0
        
        if abs(price_momentum_divergence) > 0.2:
            exhaustion_signals += 1
        
        if volume_trend_divergence < -0.3:
            exhaustion_signals += 1
        
        if rsi > 80 or rsi < 20:
            exhaustion_signals += 1
        
        return exhaustion_signals >= 2
    
    def _detect_range_compression(self, features: Dict[str, float]) -> bool:
        """Detect if range is compressing (coiling)"""
        bb_width = features.get('bb_width', 1.0)
        bb_width_ma = features.get('bb_width_ma', 1.0)
        atr = features.get('atr_14', 1.0)
        atr_ma = features.get('atr_ma', 1.0)
        
        # Both BB and ATR contracting
        bb_contracting = bb_width < bb_width_ma * 0.8
        atr_contracting = atr < atr_ma * 0.8
        
        return bb_contracting and atr_contracting
    
    def _detect_breakout_pending(self, features: Dict[str, float]) -> bool:
        """Detect if breakout is imminent"""
        # Range compression
        compression = self._detect_range_compression(features)
        
        # Volume building
        volume_building = features.get('volume_ma_ratio', 1.0) > 1.2
        
        # Price near boundary
        range_position = features.get('range_position', 0.5)
        near_boundary = range_position > 0.8 or range_position < 0.2
        
        return compression and (volume_building or near_boundary)
    
    def _update_history(
        self,
        regime: str,
        sub_regime: str,
        confidence: float,
        timestamp: pd.Timestamp,
        features: Dict[str, float]
    ) -> None:
        """Update regime history for pattern analysis"""
        
        regime_record = {
            'timestamp': timestamp,
            'regime': regime,
            'sub_regime': sub_regime,
            'confidence': confidence,
            'features': features.copy()
        }
        
        self.regime_history.append(regime_record)
        self.sub_regime_history.append(sub_regime)
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime patterns"""
        
        if not self.regime_history:
            return {}
        
        # Regime distribution
        regimes = [r['regime'] for r in self.regime_history]
        regime_counts = pd.Series(regimes).value_counts(normalize=True).to_dict()
        
        # Sub-regime distribution
        sub_regime_counts = pd.Series(self.sub_regime_history).value_counts(normalize=True).to_dict()
        
        # Average confidence by regime
        regime_confidence = {}
        for regime in set(regimes):
            confidences = [r['confidence'] for r in self.regime_history if r['regime'] == regime]
            regime_confidence[regime] = np.mean(confidences) if confidences else 0
        
        # Transition matrix (simplified)
        transitions = self._calculate_transition_matrix()
        
        return {
            'regime_distribution': regime_counts,
            'sub_regime_distribution': sub_regime_counts,
            'average_confidence': regime_confidence,
            'transition_matrix': transitions,
            'total_observations': len(self.regime_history)
        }
    
    def _calculate_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities"""
        
        if len(self.regime_history) < 2:
            return {}
        
        transitions = {}
        regimes = [r['regime'] for r in self.regime_history]
        
        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            
            if from_regime not in transitions:
                transitions[from_regime] = {}
            
            if to_regime not in transitions[from_regime]:
                transitions[from_regime][to_regime] = 0
            
            transitions[from_regime][to_regime] += 1
        
        # Convert to probabilities
        for from_regime in transitions:
            total = sum(transitions[from_regime].values())
            if total > 0:
                for to_regime in transitions[from_regime]:
                    transitions[from_regime][to_regime] /= total
        
        return transitions
    
    def get_transition_probability(self) -> float:
        """Get current transition probability"""
        
        if not self.transition_warnings:
            return 0.0
        
        # Recent warnings (last 10 minutes)
        recent_warnings = [
            w for w in self.transition_warnings 
            if (pd.Timestamp.now() - w.timestamp).total_seconds() < 600
        ]
        
        if not recent_warnings:
            return 0.0
        
        # Use the most recent calculation
        features = self.regime_history[-1]['features'] if self.regime_history else {}
        return self._calculate_transition_probability(recent_warnings, features)