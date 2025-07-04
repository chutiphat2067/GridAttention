"""
market_regime_detector.py
Detect market regime with attention enhancement for grid trading system

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats
import json

# Third-party imports
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Local imports (from other modules)
from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase

# Setup logger
logger = logging.getLogger(__name__)


# Enums and Constants
class MarketRegime(Enum):
    """Market regime types"""
    RANGING = "RANGING"          # Sideways, bounded movement
    TRENDING = "TRENDING"        # Clear directional movement
    VOLATILE = "VOLATILE"        # High volatility, erratic
    DORMANT = "DORMANT"          # Very low activity
    TRANSITIONING = "TRANSITIONING"  # Between regimes
    

class RegimeConfidence(Enum):
    """Confidence levels for regime detection"""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


# Configuration constants
MIN_HISTORY_FOR_DETECTION = 100
REGIME_CHANGE_THRESHOLD = 0.7
TRANSITION_SMOOTHING_PERIOD = 5
REGIME_MEMORY_SIZE = 200


@dataclass
class RegimeState:
    """Current regime state information"""
    regime: MarketRegime
    confidence: float
    timestamp: float
    features: Dict[str, float]
    sub_state: Optional[str] = None  # e.g., "strong_uptrend", "weak_ranging"
    strength: float = 0.5
    duration: int = 0  # Number of periods in this regime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'features': self.features.copy(),
            'sub_state': self.sub_state,
            'strength': self.strength,
            'duration': self.duration,
            'metadata': self.metadata.copy()
        }


@dataclass
class RegimeTransition:
    """Information about regime transitions"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: float
    confidence: float
    trigger_features: Dict[str, float]
    smoothed: bool = False
    
    
class RegimeTransitionDetector:
    """Detects and validates regime transitions"""
    
    def __init__(self, smoothing_period: int = TRANSITION_SMOOTHING_PERIOD):
        self.smoothing_period = smoothing_period
        self.pending_transitions = deque(maxlen=smoothing_period)
        self.transition_history = deque(maxlen=100)
        self.false_transitions = 0
        self.confirmed_transitions = 0
        
    def check_transition(
        self, 
        previous_state: RegimeState, 
        current_regime: MarketRegime, 
        confidence: float
    ) -> Optional[RegimeTransition]:
        """Check if a regime transition should occur"""
        
        # No transition if same regime
        if previous_state.regime == current_regime:
            self.pending_transitions.clear()
            return None
            
        # Create potential transition
        transition = RegimeTransition(
            from_regime=previous_state.regime,
            to_regime=current_regime,
            timestamp=time.time(),
            confidence=confidence,
            trigger_features={}
        )
        
        # Add to pending
        self.pending_transitions.append(transition)
        
        # Check if transition is confirmed
        if self._is_transition_confirmed():
            self.confirmed_transitions += 1
            self.pending_transitions.clear()
            return transition
        else:
            # Not yet confirmed
            return None
            
    def _is_transition_confirmed(self) -> bool:
        """Check if pending transition is confirmed"""
        if len(self.pending_transitions) < self.smoothing_period // 2:
            return False
            
        # Check consistency of pending transitions
        regimes = [t.to_regime for t in self.pending_transitions]
        most_common = max(set(regimes), key=regimes.count)
        consistency = regimes.count(most_common) / len(regimes)
        
        # Require high consistency
        return consistency >= 0.6
        
    def record_false_transition(self) -> None:
        """Record a false transition for learning"""
        self.false_transitions += 1
        self.pending_transitions.clear()
        
    def get_transition_reliability(self) -> float:
        """Get transition detection reliability"""
        total = self.confirmed_transitions + self.false_transitions
        if total == 0:
            return 0.5
        return self.confirmed_transitions / total


class BaseRegimeRule(ABC):
    """Base class for regime detection rules"""
    
    def __init__(self, name: str):
        self.name = name
        self.weight = 1.0
        self.hit_count = 0
        self.total_count = 0
        
    @abstractmethod
    def calculate_score(self, features: Dict[str, float]) -> float:
        """Calculate regime score (0-1)"""
        pass
        
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Get list of required features"""
        pass
        
    def update_statistics(self, was_correct: bool) -> None:
        """Update rule statistics"""
        self.total_count += 1
        if was_correct:
            self.hit_count += 1
            
    def get_accuracy(self) -> float:
        """Get rule accuracy"""
        if self.total_count == 0:
            return 0.5
        return self.hit_count / self.total_count
        
    def adjust_weight(self, performance_feedback: float) -> None:
        """Adjust rule weight based on performance"""
        # Simple exponential moving average
        alpha = 0.1
        self.weight = (1 - alpha) * self.weight + alpha * performance_feedback
        self.weight = np.clip(self.weight, 0.1, 2.0)


class RangingRegimeRule(BaseRegimeRule):
    """Rule for detecting ranging/sideways market"""
    
    def __init__(self):
        super().__init__('ranging_rule')
        self.lookback = 20
        
    def calculate_score(self, features: Dict[str, float]) -> float:
        """Calculate ranging score"""
        score_components = []
        
        # Low trend strength indicates ranging
        trend_strength = abs(features.get('trend_strength', 0))
        trend_score = 1.0 - trend_strength
        score_components.append(('trend', trend_score, 0.3))
        
        # Price within bands
        bb_position = abs(features.get('bb_position', 0))
        bb_score = 1.0 - bb_position  # Closer to middle = higher score
        score_components.append(('bollinger', bb_score, 0.2))
        
        # Low volatility change
        volatility = features.get('volatility_5m', 0.001)
        vol_score = 1.0 / (1.0 + volatility * 100)  # Lower volatility = higher score
        score_components.append(('volatility', vol_score, 0.2))
        
        # Price position stability
        price_position = features.get('price_position', 0.5)
        position_score = 1.0 - abs(price_position - 0.5) * 2  # Middle of range = high score
        score_components.append(('position', position_score, 0.2))
        
        # RSI near 50
        rsi = features.get('rsi_14', 0.5)
        rsi_score = 1.0 - abs(rsi - 0.5) * 2
        score_components.append(('rsi', rsi_score, 0.1))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return np.clip(total_score, 0, 1)
        
    def get_required_features(self) -> List[str]:
        return ['trend_strength', 'bb_position', 'volatility_5m', 'price_position', 'rsi_14']


class TrendingRegimeRule(BaseRegimeRule):
    """Rule for detecting trending market"""
    
    def __init__(self):
        super().__init__('trending_rule')
        
    def calculate_score(self, features: Dict[str, float]) -> float:
        """Calculate trending score"""
        score_components = []
        
        # High trend strength
        trend_strength = abs(features.get('trend_strength', 0))
        trend_score = trend_strength
        score_components.append(('trend', trend_score, 0.4))
        
        # Consistent price movement
        price_change = abs(features.get('price_change_5m', 0))
        change_score = min(price_change * 100, 1.0)  # Cap at 1%
        score_components.append(('change', change_score, 0.2))
        
        # RSI extremes
        rsi = features.get('rsi_14', 0.5)
        rsi_score = abs(rsi - 0.5) * 2  # Further from 50 = higher score
        score_components.append(('rsi', rsi_score, 0.2))
        
        # Volume confirmation
        volume_ratio = features.get('volume_ratio', 1.0)
        volume_score = min((volume_ratio - 1.0) / 2.0, 1.0) if volume_ratio > 1 else 0
        score_components.append(('volume', volume_score, 0.1))
        
        # Bollinger band position
        bb_position = abs(features.get('bb_position', 0))
        bb_score = min(bb_position, 1.0)  # Near bands = trending
        score_components.append(('bollinger', bb_score, 0.1))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return np.clip(total_score, 0, 1)
        
    def get_required_features(self) -> List[str]:
        return ['trend_strength', 'price_change_5m', 'rsi_14', 'volume_ratio', 'bb_position']


class VolatileRegimeRule(BaseRegimeRule):
    """Rule for detecting volatile market"""
    
    def __init__(self):
        super().__init__('volatile_rule')
        
    def calculate_score(self, features: Dict[str, float]) -> float:
        """Calculate volatility score"""
        score_components = []
        
        # High volatility
        volatility = features.get('volatility_5m', 0.001)
        vol_score = min(volatility * 500, 1.0)  # Scale volatility
        score_components.append(('volatility', vol_score, 0.4))
        
        # Wide spread
        spread_bps = features.get('spread_bps', 1)
        spread_score = min(spread_bps / 10, 1.0)  # >10bps = max score
        score_components.append(('spread', spread_score, 0.2))
        
        # Erratic price movement
        price_change = abs(features.get('price_change_5m', 0))
        microstructure = features.get('microstructure_score', 0.5)
        erratic_score = price_change * 50 * (1 - microstructure)
        score_components.append(('erratic', min(erratic_score, 1.0), 0.2))
        
        # Volume spikes
        volume_acceleration = abs(features.get('volume_acceleration', 0))
        volume_score = min(volume_acceleration * 2, 1.0)
        score_components.append(('volume', volume_score, 0.1))
        
        # Order imbalance
        imbalance = abs(features.get('order_imbalance', 0))
        imbalance_score = imbalance
        score_components.append(('imbalance', imbalance_score, 0.1))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return np.clip(total_score, 0, 1)
        
    def get_required_features(self) -> List[str]:
        return ['volatility_5m', 'spread_bps', 'price_change_5m', 'microstructure_score', 
                'volume_acceleration', 'order_imbalance']


class DormantRegimeRule(BaseRegimeRule):
    """Rule for detecting dormant/inactive market"""
    
    def __init__(self):
        super().__init__('dormant_rule')
        
    def calculate_score(self, features: Dict[str, float]) -> float:
        """Calculate dormancy score"""
        score_components = []
        
        # Very low volatility
        volatility = features.get('volatility_5m', 0.001)
        vol_score = 1.0 / (1.0 + volatility * 1000)
        score_components.append(('volatility', vol_score, 0.3))
        
        # Low volume
        volume_ratio = features.get('volume_ratio', 1.0)
        volume_score = 1.0 / (1.0 + volume_ratio)
        score_components.append(('volume', volume_score, 0.3))
        
        # Minimal price change
        price_change = abs(features.get('price_change_5m', 0))
        change_score = 1.0 / (1.0 + price_change * 200)
        score_components.append(('change', change_score, 0.2))
        
        # Tight spread
        spread_bps = features.get('spread_bps', 1)
        spread_score = 1.0 / (1.0 + spread_bps / 2)
        score_components.append(('spread', spread_score, 0.1))
        
        # No trend
        trend_strength = abs(features.get('trend_strength', 0))
        trend_score = 1.0 - trend_strength
        score_components.append(('trend', trend_score, 0.1))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return np.clip(total_score, 0, 1)
        
    def get_required_features(self) -> List[str]:
        return ['volatility_5m', 'volume_ratio', 'price_change_5m', 'spread_bps', 'trend_strength']


class MarketRegimeDetector:
    """
    Detect market regime with attention enhancement
    """
    
    def __init__(self, attention_layer: Optional[AttentionLearningLayer] = None):
        self.attention = attention_layer
        self.regime_rules = self._init_regime_rules()
        self.regime_history = deque(maxlen=REGIME_MEMORY_SIZE)
        self.transition_detector = RegimeTransitionDetector()
        
        # Machine learning components
        self.gmm_model = None  # Gaussian Mixture Model
        self.scaler = StandardScaler()
        self.is_ml_initialized = False
        
        # Performance tracking
        self.detection_count = 0
        self.regime_durations = defaultdict(list)
        self.regime_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Current state
        self.current_state = None
        self._lock = asyncio.Lock()
        
        logger.info("Initialized Market Regime Detector")
        
    def _init_regime_rules(self) -> Dict[MarketRegime, BaseRegimeRule]:
        """Initialize regime detection rules"""
        return {
            MarketRegime.RANGING: RangingRegimeRule(),
            MarketRegime.TRENDING: TrendingRegimeRule(),
            MarketRegime.VOLATILE: VolatileRegimeRule(),
            MarketRegime.DORMANT: DormantRegimeRule()
        }
        
    async def detect_regime(self, features: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Detect current market regime"""
        async with self._lock:
            self.detection_count += 1
            
            # Check if we have enough history
            if len(self.regime_history) < MIN_HISTORY_FOR_DETECTION:
                # Default to ranging with low confidence during warmup
                regime = MarketRegime.RANGING
                confidence = 0.3
                
            else:
                # Apply attention if active
                if self.attention and self.attention.phase == AttentionPhase.ACTIVE:
                    features = await self._apply_attention_to_features(features)
                    
                # Calculate regime scores
                regime_scores = await self._calculate_regime_scores(features)
                
                # Apply ML enhancement if available
                if self.is_ml_initialized:
                    ml_regime, ml_confidence = await self._apply_ml_detection(features)
                    # Blend with rule-based scores
                    regime_scores = self._blend_scores(regime_scores, ml_regime, ml_confidence)
                    
                # Select regime with highest score
                regime, confidence = self._select_regime(regime_scores)
                
                # Check for transition
                if self.current_state:
                    transition = self.transition_detector.check_transition(
                        self.current_state, regime, confidence
                    )
                    
                    if transition:
                        await self._handle_transition(transition)
                        
            # Create new state
            new_state = RegimeState(
                regime=regime,
                confidence=confidence,
                timestamp=time.time(),
                features=features.copy(),
                strength=self._calculate_regime_strength(regime, features),
                duration=self._get_regime_duration(regime)
            )
            
            # Determine sub-state
            new_state.sub_state = self._determine_sub_state(regime, features)
            
            # Update history
            self.regime_history.append(new_state)
            self.current_state = new_state
            
            # Update ML model periodically
            if self.detection_count % 100 == 0:
                await self._update_ml_model()
                
            return regime, confidence
            
    async def _apply_attention_to_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply attention mechanism to features for regime detection"""
        if not self.attention:
            return features
            
        # Let attention process features specifically for regime detection
        context = {
            'task': 'regime_detection',
            'history': [s.features for s in list(self.regime_history)[-20:]]
        }
        
        # Process through attention
        weighted_features = await self.attention.feature_attention.apply_weights(features)
        
        return weighted_features
        
    async def _calculate_regime_scores(self, features: Dict[str, float]) -> Dict[MarketRegime, float]:
        """Calculate scores for each regime"""
        regime_scores = {}
        
        for regime, rule in self.regime_rules.items():
            try:
                # Check if all required features are present
                required_features = rule.get_required_features()
                if all(f in features for f in required_features):
                    score = rule.calculate_score(features)
                    # Apply rule weight
                    weighted_score = score * rule.weight
                    regime_scores[regime] = weighted_score
                else:
                    regime_scores[regime] = 0.0
                    
            except Exception as e:
                logger.error(f"Error calculating score for {regime}: {e}")
                regime_scores[regime] = 0.0
                
        # Normalize scores
        total_score = sum(regime_scores.values())
        if total_score > 0:
            regime_scores = {k: v / total_score for k, v in regime_scores.items()}
            
        return regime_scores
        
    async def _apply_ml_detection(self, features: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Apply machine learning model for regime detection"""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            # Scale features
            scaled_features = self.scaler.transform([feature_vector])
            
            # Predict regime
            prediction = self.gmm_model.predict(scaled_features)[0]
            probabilities = self.gmm_model.predict_proba(scaled_features)[0]
            
            # Map cluster to regime
            regime = self._map_cluster_to_regime(prediction)
            confidence = float(probabilities[prediction])
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"ML detection failed: {e}")
            return MarketRegime.RANGING, 0.5
            
    def _prepare_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Prepare feature vector for ML model"""
        # Use consistent feature ordering
        feature_names = [
            'trend_strength', 'volatility_5m', 'volume_ratio',
            'price_change_5m', 'rsi_14', 'bb_position',
            'spread_bps', 'order_imbalance'
        ]
        
        vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            vector.append(value)
            
        return vector
        
    def _map_cluster_to_regime(self, cluster: int) -> MarketRegime:
        """Map GMM cluster to market regime"""
        # This mapping is learned during training
        cluster_map = {
            0: MarketRegime.RANGING,
            1: MarketRegime.TRENDING,
            2: MarketRegime.VOLATILE,
            3: MarketRegime.DORMANT
        }
        
        return cluster_map.get(cluster, MarketRegime.RANGING)
        
    def _blend_scores(
        self, 
        rule_scores: Dict[MarketRegime, float], 
        ml_regime: MarketRegime, 
        ml_confidence: float
    ) -> Dict[MarketRegime, float]:
        """Blend rule-based and ML scores"""
        # Weight for ML prediction
        ml_weight = 0.3 if ml_confidence > 0.6 else 0.1
        
        # Adjust scores
        blended_scores = {}
        for regime in MarketRegime:
            if regime == ml_regime:
                # Boost ML-predicted regime
                blended_scores[regime] = (
                    rule_scores.get(regime, 0) * (1 - ml_weight) + 
                    ml_confidence * ml_weight
                )
            else:
                # Reduce other regimes proportionally
                blended_scores[regime] = rule_scores.get(regime, 0) * (1 - ml_weight * 0.5)
                
        return blended_scores
        
    def _select_regime(self, regime_scores: Dict[MarketRegime, float]) -> Tuple[MarketRegime, float]:
        """Select regime with highest score"""
        if not regime_scores:
            return MarketRegime.RANGING, 0.5
            
        # Find regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        # Check if transitioning
        if confidence < REGIME_CHANGE_THRESHOLD and self.current_state:
            # Stay in current regime if confidence is low
            best_regime = self.current_state.regime
            confidence = confidence * 0.8  # Reduce confidence
            
        return best_regime, confidence
        
    def _calculate_regime_strength(self, regime: MarketRegime, features: Dict[str, float]) -> float:
        """Calculate how strongly the regime is expressed"""
        if regime == MarketRegime.TRENDING:
            # Trend strength directly indicates regime strength
            return abs(features.get('trend_strength', 0))
            
        elif regime == MarketRegime.VOLATILE:
            # Volatility level indicates strength
            vol = features.get('volatility_5m', 0.001)
            return min(vol * 200, 1.0)
            
        elif regime == MarketRegime.RANGING:
            # Inverse of trend strength
            return 1.0 - abs(features.get('trend_strength', 0))
            
        elif regime == MarketRegime.DORMANT:
            # Inverse of activity metrics
            vol = features.get('volatility_5m', 0.001)
            volume = features.get('volume_ratio', 1.0)
            return 1.0 / (1.0 + vol * 100 + volume)
            
        return 0.5
        
    def _determine_sub_state(self, regime: MarketRegime, features: Dict[str, float]) -> str:
        """Determine regime sub-state for more granular classification"""
        if regime == MarketRegime.TRENDING:
            trend = features.get('trend_strength', 0)
            if trend > 0.7:
                return "strong_uptrend"
            elif trend > 0.3:
                return "moderate_uptrend"
            elif trend > 0:
                return "weak_uptrend"
            elif trend < -0.7:
                return "strong_downtrend"
            elif trend < -0.3:
                return "moderate_downtrend"
            else:
                return "weak_downtrend"
                
        elif regime == MarketRegime.RANGING:
            volatility = features.get('volatility_5m', 0.001)
            if volatility < 0.0005:
                return "tight_range"
            elif volatility < 0.001:
                return "normal_range"
            else:
                return "wide_range"
                
        elif regime == MarketRegime.VOLATILE:
            spread = features.get('spread_bps', 1)
            if spread > 10:
                return "extreme_volatility"
            elif spread > 5:
                return "high_volatility"
            else:
                return "moderate_volatility"
                
        elif regime == MarketRegime.DORMANT:
            volume = features.get('volume_ratio', 1.0)
            if volume < 0.1:
                return "extremely_dormant"
            elif volume < 0.3:
                return "very_dormant"
            else:
                return "moderately_dormant"
                
        return "normal"
        
    def _get_regime_duration(self, regime: MarketRegime) -> int:
        """Get how long we've been in current regime"""
        if not self.regime_history:
            return 0
            
        duration = 0
        for state in reversed(self.regime_history):
            if state.regime == regime:
                duration += 1
            else:
                break
                
        return duration
        
    async def _handle_transition(self, transition: RegimeTransition) -> None:
        """Handle regime transition"""
        logger.info(f"Regime transition: {transition.from_regime.value} -> {transition.to_regime.value}")
        
        # Record duration of previous regime
        if self.current_state:
            duration = self.current_state.duration
            self.regime_durations[transition.from_regime].append(duration)
            
        # Notify attention system if available
        if self.attention:
            context = {
                'previous_regime': transition.from_regime.value,
                'timestamp': transition.timestamp
            }
            await self.attention.regime_attention.observe(transition.to_regime.value, context)
            
    async def _update_ml_model(self) -> None:
        """Update machine learning model with recent data"""
        if len(self.regime_history) < 200:
            return
            
        try:
            # Prepare training data
            X = []
            y = []
            
            for state in list(self.regime_history)[-200:]:
                feature_vector = self._prepare_feature_vector(state.features)
                X.append(feature_vector)
                y.append(state.regime.value)
                
            X = np.array(X)
            
            # Initialize or update model
            if not self.is_ml_initialized:
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                
                self.gmm_model = GaussianMixture(
                    n_components=4,  # 4 regimes
                    covariance_type='full',
                    max_iter=100,
                    random_state=42
                )
                self.gmm_model.fit(X_scaled)
                self.is_ml_initialized = True
                
                logger.info("Initialized ML regime detection model")
                
            else:
                # Incremental update (refit with recent data)
                X_scaled = self.scaler.transform(X)
                self.gmm_model.fit(X_scaled)
                
        except Exception as e:
            logger.error(f"Failed to update ML model: {e}")
            
    async def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics"""
        stats = {
            'current_regime': self.current_state.regime.value if self.current_state else None,
            'current_confidence': self.current_state.confidence if self.current_state else 0,
            'current_duration': self.current_state.duration if self.current_state else 0,
            'detection_count': self.detection_count,
            'regime_distribution': {},
            'average_durations': {},
            'transition_reliability': self.transition_detector.get_transition_reliability(),
            'rule_performance': {}
        }
        
        # Calculate regime distribution
        regime_counts = defaultdict(int)
        for state in self.regime_history:
            regime_counts[state.regime.value] += 1
            
        total = sum(regime_counts.values())
        if total > 0:
            stats['regime_distribution'] = {
                regime: count / total 
                for regime, count in regime_counts.items()
            }
            
        # Calculate average durations
        for regime, durations in self.regime_durations.items():
            if durations:
                stats['average_durations'][regime.value] = np.mean(durations)
                
        # Get rule performance
        for regime, rule in self.regime_rules.items():
            stats['rule_performance'][regime.value] = {
                'accuracy': rule.get_accuracy(),
                'weight': rule.weight
            }
            
        return stats
        
    def get_regime_history(self, n: int = 100) -> List[RegimeState]:
        """Get recent regime history"""
        return list(self.regime_history)[-n:]
        
    async def update_performance_feedback(self, regime: MarketRegime, was_successful: bool) -> None:
        """Update regime detection performance based on trading results"""
        # Update rule statistics
        if regime in self.regime_rules:
            self.regime_rules[regime].update_statistics(was_successful)
            
        # Update accuracy tracking
        self.regime_accuracy[regime]['total'] += 1
        if was_successful:
            self.regime_accuracy[regime]['correct'] += 1
            
        # Adjust rule weights based on performance
        accuracy = self.regime_accuracy[regime]['correct'] / self.regime_accuracy[regime]['total']
        if regime in self.regime_rules:
            self.regime_rules[regime].adjust_weight(accuracy)
            
    async def save_state(self, filepath: str) -> None:
        """Save detector state to file"""
        state = {
            'detection_count': self.detection_count,
            'regime_history': [s.to_dict() for s in list(self.regime_history)[-50:]],
            'regime_durations': {k.value: v for k, v in self.regime_durations.items()},
            'rule_weights': {k.value: v.weight for k, v in self.regime_rules.items()},
            'ml_initialized': self.is_ml_initialized
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved regime detector state to {filepath}")
        
    async def load_state(self, filepath: str) -> None:
        """Load detector state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.detection_count = state['detection_count']
        
        # Restore rule weights
        for regime_name, weight in state['rule_weights'].items():
            regime = MarketRegime(regime_name)
            if regime in self.regime_rules:
                self.regime_rules[regime].weight = weight
                
        logger.info(f"Loaded regime detector state from {filepath}")


# Example usage
async def main():
    """Example usage of MarketRegimeDetector"""
    
    # Initialize with attention layer
    from core.attention_learning_layer import AttentionLearningLayer
    
    attention = AttentionLearningLayer()
    detector = MarketRegimeDetector(attention)
    
    # Simulate market data
    for i in range(300):
        # Generate synthetic features
        trend = np.sin(i * 0.1) * 0.5  # Oscillating trend
        
        features = {
            'trend_strength': trend + np.random.randn() * 0.1,
            'volatility_5m': 0.001 + abs(np.random.randn()) * 0.0005,
            'volume_ratio': 1.0 + np.random.randn() * 0.3,
            'price_change_5m': trend * 0.001 + np.random.randn() * 0.0001,
            'rsi_14': 0.5 + trend * 0.3 + np.random.randn() * 0.1,
            'bb_position': trend + np.random.randn() * 0.2,
            'spread_bps': 1 + abs(np.random.randn()) * 2,
            'order_imbalance': np.random.randn() * 0.3,
            'price_position': 0.5 + trend * 0.3,
            'microstructure_score': 0.7 + np.random.randn() * 0.1,
            'volume_acceleration': np.random.randn() * 0.2
        }
        
        # Detect regime
        regime, confidence = await detector.detect_regime(features)
        
        # Print updates
        if i % 50 == 0:
            print(f"\nIteration {i}:")
            print(f"  Regime: {regime.value}")
            print(f"  Confidence: {confidence:.2f}")
            
            if detector.current_state:
                print(f"  Sub-state: {detector.current_state.sub_state}")
                print(f"  Strength: {detector.current_state.strength:.2f}")
                print(f"  Duration: {detector.current_state.duration}")
                
    # Get statistics
    stats = await detector.get_regime_statistics()
    print(f"\nRegime Statistics:")
    print(f"  Distribution: {stats['regime_distribution']}")
    print(f"  Avg Durations: {stats['average_durations']}")
    print(f"  Transition Reliability: {stats['transition_reliability']:.2%}")
    
    # Save state
    await detector.save_state('regime_detector_state.json')


if __name__ == "__main__":
    asyncio.run(main())
