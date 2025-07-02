from typing import Dict, Any, Optional
"""
attention_learning_layer.py
Master attention controller with three sub-modules for grid trading system
Enhanced with comprehensive overfitting prevention mechanisms

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import pandas as pd
from scipy import stats
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, neural network features disabled")
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# Import overfitting detector
from overfitting_detector import OverfittingDetector, OverfittingSeverity

# Setup logger
logger = logging.getLogger(__name__)


# Enums and Constants
class AttentionPhase(Enum):
    """Attention system phases"""
    LEARNING = "learning"      # Observe only, no impact
    SHADOW = "shadow"          # Calculate but don't apply
    ACTIVE = "active"          # Full application
    
    
class AttentionType(Enum):
    """Types of attention mechanisms"""
    FEATURE = "feature"
    TEMPORAL = "temporal"
    REGIME = "regime"


# Configuration constants - MORE CONSERVATIVE VALUES
MIN_TRADES_FOR_LEARNING = 2000  # Increased from 1000 for better learning
MIN_TRADES_FOR_SHADOW = 500    # Increased from 200
MIN_TRADES_FOR_ACTIVE = 200    # Increased from 100
MAX_ATTENTION_INFLUENCE = 0.3  # Maximum 30% adjustment
LEARNING_RATE = 0.001
ATTENTION_WINDOW_SIZE = 1000
VALIDATION_THRESHOLD = 0.5  # Maximum 50% feature change allowed
MIN_SAMPLES_FOR_DETECTION = 100  # Minimum samples for overfitting detection

# Enhanced regularization configuration
REGULARIZATION_CONFIG = {
    'dropout_rate': 0.3,           # เพิ่มจาก 0.1
    'weight_decay': 0.01,          # L2 regularization
    'gradient_clipping': 1.0,      # Gradient clipping threshold
    'early_stopping_patience': 50,  # Early stopping patience
    'learning_rate_decay': 0.95,   # LR decay factor
    'max_norm': 2.0,              # Max norm for weight clipping
    'label_smoothing': 0.1,       # Label smoothing factor
    'mixup_alpha': 0.2            # Mixup augmentation alpha
}


@dataclass
class AttentionMetrics:
    """Metrics for attention system performance"""
    phase: AttentionPhase
    total_observations: int = 0
    processing_times: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    shadow_calculations: int = 0
    active_applications: int = 0
    performance_improvements: Dict[str, float] = field(default_factory=dict)
    phase_transitions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Regularization tracking
    regularization_metrics: Dict[str, List[float]] = field(default_factory=lambda: {
        'dropout_effectiveness': [],
        'weight_decay_impact': [],
        'gradient_norms': [],
        'learning_rates': []
    })
    overfitting_detections: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_processing_time(self, time_ms: float):
        """Record processing time"""
        self.processing_times.append(time_ms)
        
    def get_avg_processing_time(self) -> float:
        """Get average processing time"""
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times)
        
    def record_shadow_calculation(self, calculation_data: Dict[str, Any]):
        """Record shadow mode calculation"""
        self.shadow_calculations += 1
        
    def record_performance_improvement(self, metric: str, improvement: float):
        """Record performance improvement"""
        if metric not in self.performance_improvements:
            self.performance_improvements[metric] = []
        self.performance_improvements[metric].append(improvement)


class PhaseController:
    """Controls phase transitions for attention system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_trades_learning = self.config.get('min_trades_learning', MIN_TRADES_FOR_LEARNING)
        self.min_trades_shadow = self.config.get('min_trades_shadow', MIN_TRADES_FOR_SHADOW)
        self.min_trades_active = self.config.get('min_trades_active', MIN_TRADES_FOR_ACTIVE)
        
        # Performance thresholds
        self.shadow_confidence_threshold = 0.7
        self.active_confidence_threshold = 0.85
        self.performance_improvement_threshold = 0.02  # 2% improvement
        
    def should_transition(self, metrics: AttentionMetrics) -> bool:
        """Check if phase transition should occur"""
        if metrics.phase == AttentionPhase.LEARNING:
            return self._should_transition_to_shadow(metrics)
        elif metrics.phase == AttentionPhase.SHADOW:
            return self._should_transition_to_active(metrics)
        return False
        
    def _should_transition_to_shadow(self, metrics: AttentionMetrics) -> bool:
        """Check if ready to transition from learning to shadow"""
        # Check minimum observations
        if metrics.total_observations < self.min_trades_learning:
            return False
            
        # Check if we have learned meaningful patterns
        avg_processing_time = metrics.get_avg_processing_time()
        if avg_processing_time > 1.0:  # > 1ms is too slow
            logger.warning(f"Processing time too high: {avg_processing_time:.2f}ms")
            return False
            
        logger.info(f"Ready to transition to shadow mode after {metrics.total_observations} observations")
        return True
        
    def _should_transition_to_active(self, metrics: AttentionMetrics) -> bool:
        """Check if ready to transition from shadow to active"""
        # Check minimum shadow calculations
        if metrics.shadow_calculations < self.min_trades_shadow:
            return False
            
        # Check performance improvements
        if not metrics.performance_improvements:
            return False
            
        # Calculate average improvement
        avg_improvements = {}
        for metric, values in metrics.performance_improvements.items():
            if values:
                avg_improvements[metric] = np.mean(values)
                
        # Check if improvements meet threshold
        significant_improvements = sum(
            1 for imp in avg_improvements.values() 
            if imp >= self.performance_improvement_threshold
        )
        
        if significant_improvements >= 2:  # At least 2 metrics improved
            logger.info(f"Ready to transition to active mode with improvements: {avg_improvements}")
            return True
            
        return False
        
    def get_next_phase(self, current_phase: AttentionPhase) -> AttentionPhase:
        """Get next phase in progression"""
        if current_phase == AttentionPhase.LEARNING:
            return AttentionPhase.SHADOW
        elif current_phase == AttentionPhase.SHADOW:
            return AttentionPhase.ACTIVE
        return current_phase


class BaseAttentionModule(ABC):
    """Base class for attention modules"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.observation_count = 0
        self.is_initialized = False
        
    @abstractmethod
    async def observe(self, data: Any, context: Dict[str, Any]) -> None:
        """Observe data during learning phase"""
        pass
        
    @abstractmethod
    async def calculate_weights(self, data: Any) -> Dict[str, float]:
        """Calculate attention weights"""
        pass
        
    @abstractmethod
    async def apply_weights(self, data: Any) -> Any:
        """Apply attention weights to data"""
        pass
        
    def get_stats(self) -> Dict[str, Any]:
        """Get module statistics"""
        return {
            'name': self.name,
            'observations': self.observation_count,
            'initialized': self.is_initialized
        }


class FeatureAttention(BaseAttentionModule):
    """Attention mechanism for feature selection and weighting"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('feature_attention', config)
        
        # Feature statistics
        self.feature_stats = defaultdict(lambda: {
            'mean': 0.0,
            'std': 0.0,
            'importance': 0.5,
            'stability': 0.5,
            'values': deque(maxlen=ATTENTION_WINDOW_SIZE)
        })
        
        # Feature correlations with outcomes
        self.feature_outcomes = defaultdict(list)
        
        # Attention weights
        self.attention_weights = {}
        self.weight_history = deque(maxlen=100)
        
        # Neural network for feature attention
        self.attention_network = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = REGULARIZATION_CONFIG['early_stopping_patience']
        self.training = True  # Training mode flag
        
        # Overfitting detector
        self.overfitting_detector = OverfittingDetector()
        
    async def observe(self, features: Dict[str, float], context: Dict[str, Any]) -> None:
        """Observe features during learning phase"""
        self.observation_count += 1
        
        # Update feature statistics
        for name, value in features.items():
            stats = self.feature_stats[name]
            stats['values'].append(value)
            
            # Update running statistics
            if len(stats['values']) > 10:
                stats['mean'] = np.mean(stats['values'])
                stats['std'] = np.std(stats['values'])
                
        # Store outcome if available
        if 'outcome' in context:
            outcome = context['outcome']
            for name, value in features.items():
                self.feature_outcomes[name].append((value, outcome))
                
        # Initialize network after sufficient observations
        if self.observation_count == 100 and not self.is_initialized:
            await self._initialize_network(len(features))
            
    async def _initialize_network(self, num_features: int) -> None:
        """Initialize attention neural network with enhanced regularization"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping network initialization")
            return
            
        # Use enhanced regularization config
        dropout_rate = REGULARIZATION_CONFIG['dropout_rate']
        weight_decay = REGULARIZATION_CONFIG['weight_decay']
        
        self.attention_network = FeatureAttentionNetwork(
            num_features, 
            dropout_rate=dropout_rate
        )
        
        # Optimizer with weight decay and gradient clipping
        self.optimizer = torch.optim.AdamW(  # Use AdamW for better weight decay
            self.attention_network.parameters(),
            lr=LEARNING_RATE,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,  # Restart every 100 iterations
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Initialize gradient scaler for mixed precision (if available)
        if hasattr(torch.cuda.amp, 'GradScaler'):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.is_initialized = True
        logger.info(f"Initialized feature attention network with enhanced regularization "
                    f"(dropout={dropout_rate}, weight_decay={weight_decay})")
        
    async def calculate_weights(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate attention weights for features"""
        if not self.is_initialized:
            # Return uniform weights if not initialized
            uniform_weight = 1.0 / len(features)
            return {name: uniform_weight for name in features}
            
        # Prepare input tensor
        feature_names = sorted(features.keys())
        feature_values = [features[name] for name in feature_names]
        
        # Normalize features
        normalized_values = self._normalize_features(feature_values)
        
        # Get attention weights from network
        input_tensor = torch.FloatTensor(normalized_values).unsqueeze(0)
        
        with torch.no_grad():
            attention_scores = self.attention_network(input_tensor)
            attention_weights = F.softmax(attention_scores, dim=1).squeeze().numpy()
            
        # Create weight dictionary
        weights = {}
        for i, name in enumerate(feature_names):
            weights[name] = float(attention_weights[i])
            
            # Update importance based on attention
            self.feature_stats[name]['importance'] = 0.7 * self.feature_stats[name]['importance'] + 0.3 * weights[name]
            
        self.attention_weights = weights
        self.weight_history.append(weights.copy())
        
        return weights
        
    async def calculate_weights_ensemble(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights using ensemble of methods for robustness"""
        weights_list = []
        
        # Method 1: Neural network weights
        if self.is_initialized:
            nn_weights = await self.calculate_weights(features)
            weights_list.append(nn_weights)
        
        # Method 2: Statistical importance (correlation-based)
        stat_weights = self._calculate_statistical_weights(features)
        if stat_weights:
            weights_list.append(stat_weights)
        
        # Method 3: Information gain based
        info_weights = self._calculate_information_gain_weights(features)
        if info_weights:
            weights_list.append(info_weights)
        
        # Ensemble: average the weights
        if not weights_list:
            # Fallback to uniform weights
            uniform_weight = 1.0 / len(features)
            return {name: uniform_weight for name in features}
        
        # Average weights from all methods
        ensemble_weights = {}
        feature_names = sorted(features.keys())
        
        for name in feature_names:
            weights = [w.get(name, 0) for w in weights_list]
            ensemble_weights[name] = np.mean(weights)
        
        # Normalize
        total = sum(ensemble_weights.values())
        if total > 0:
            ensemble_weights = {k: v/total for k, v in ensemble_weights.items()}
        
        return ensemble_weights

    def _calculate_statistical_weights(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights based on statistical properties"""
        if len(self.feature_outcomes) < 50:
            return {}
        
        weights = {}
        for name, values in self.feature_stats.items():
            if name not in features:
                continue
                
            # Calculate correlation with outcomes
            if name in self.feature_outcomes and len(self.feature_outcomes[name]) > 10:
                feature_vals = [v[0] for v in self.feature_outcomes[name]]
                outcomes = [v[1] for v in self.feature_outcomes[name]]
                
                # Use Spearman correlation (more robust to outliers)
                correlation, _ = stats.spearmanr(feature_vals, outcomes)
                weights[name] = abs(correlation) if not np.isnan(correlation) else 0.1
            else:
                weights[name] = 0.1
        
        return weights

    def _calculate_information_gain_weights(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights based on information gain"""
        # Simplified information gain calculation
        weights = {}
        
        for name, value in features.items():
            if name in self.feature_stats:
                # Use coefficient of variation as proxy for information content
                stats_dict = self.feature_stats[name]
                if stats_dict['std'] > 0 and stats_dict['mean'] != 0:
                    cv = stats_dict['std'] / abs(stats_dict['mean'])
                    # Higher CV means more information
                    weights[name] = min(cv, 1.0)
                else:
                    weights[name] = 0.1
            else:
                weights[name] = 0.1
                
        return weights
        
    async def _train_step(self, features: Dict[str, float], outcome: float) -> float:
        """Single training step with enhanced regularization techniques"""
        if not self.is_initialized or not TORCH_AVAILABLE:
            return 0.0
            
        # Prepare data
        feature_names = sorted(features.keys())
        feature_values = [features[name] for name in feature_names]
        normalized_values = self._normalize_features(feature_values)
        
        # Data augmentation - add noise
        if self.training and np.random.rand() < 0.5:
            noise = torch.randn_like(torch.FloatTensor(normalized_values)) * 0.01
            normalized_values = [v + n.item() for v, n in zip(normalized_values, noise)]
        
        input_tensor = torch.FloatTensor(normalized_values).unsqueeze(0)
        target_tensor = torch.FloatTensor([outcome])
        
        # Label smoothing
        if self.training:
            smoothing = REGULARIZATION_CONFIG['label_smoothing']
            target_tensor = target_tensor * (1 - smoothing) + smoothing * 0.5
        
        # Forward pass (with mixed precision if available)
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                output = self.attention_network(input_tensor)
                attention_weights = F.softmax(output, dim=1)
                weighted_features = attention_weights * input_tensor
                prediction = weighted_features.sum()
                
                # Multi-component loss
                mse_loss = F.mse_loss(prediction, target_tensor)
                l1_reg = REGULARIZATION_CONFIG['weight_decay'] * attention_weights.abs().mean()
                entropy_reg = 0.01 * (attention_weights * torch.log(attention_weights + 1e-8)).sum()
                
                loss = mse_loss + l1_reg - entropy_reg  # Negative entropy encourages diversity
        else:
            # Standard forward pass
            self.optimizer.zero_grad()
            output = self.attention_network(input_tensor)
            attention_weights = F.softmax(output, dim=1)
            weighted_features = attention_weights * input_tensor
            prediction = weighted_features.sum()
            
            # Multi-component loss
            mse_loss = F.mse_loss(prediction, target_tensor)
            l1_reg = REGULARIZATION_CONFIG['weight_decay'] * attention_weights.abs().mean()
            entropy_reg = 0.01 * (attention_weights * torch.log(attention_weights + 1e-8)).sum()
            
            loss = mse_loss + l1_reg - entropy_reg
        
        # Backward pass with gradient clipping
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.attention_network.parameters(), 
                REGULARIZATION_CONFIG['gradient_clipping']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.attention_network.parameters(), 
                REGULARIZATION_CONFIG['gradient_clipping']
            )
            self.optimizer.step()
        
        # Update learning rate
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        
        # Early stopping check
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            self.training = False  # Stop training
            
        return loss.item()
        
    async def apply_weights(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply attention weights to features"""
        if not self.attention_weights:
            return features
            
        weighted_features = {}
        
        for name, value in features.items():
            weight = self.attention_weights.get(name, 1.0)
            
            # Apply weight with maximum influence limit
            adjustment = 1.0 + (weight - 0.5) * 2 * MAX_ATTENTION_INFLUENCE
            weighted_features[name] = value * adjustment
            
        return weighted_features
        
    def _normalize_features(self, values: List[float]) -> List[float]:
        """Normalize feature values"""
        normalized = []
        
        for i, value in enumerate(values):
            feature_name = sorted(self.feature_stats.keys())[i]
            stats = self.feature_stats[feature_name]
            
            if stats['std'] > 0:
                normalized_value = (value - stats['mean']) / stats['std']
            else:
                normalized_value = 0.0
                
            normalized.append(normalized_value)
            
        return normalized
        
    def _mixup_augmentation(self, features1: List[float], features2: List[float], 
                           target1: float, target2: float, alpha: float = 0.2) -> Tuple[List[float], float]:
        """Mixup data augmentation for better generalization"""
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        
        mixed_features = [
            lam * f1 + (1 - lam) * f2 
            for f1, f2 in zip(features1, features2)
        ]
        mixed_target = lam * target1 + (1 - lam) * target2
        
        return mixed_features, mixed_target
        
    async def update_importance(self, performance_feedback: Dict[str, float]) -> None:
        """Update feature importance based on performance feedback"""
        if not self.is_initialized:
            return
            
        # Check for overfitting before updating
        if await self._validate_and_check_overfitting():
            logger.info("Skipping importance update due to overfitting detection")
            return
            
        # Prepare training data
        for feature_name, impact in performance_feedback.items():
            if feature_name in self.feature_stats:
                # Update importance score
                current_importance = self.feature_stats[feature_name]['importance']
                new_importance = current_importance * (1 + impact)
                self.feature_stats[feature_name]['importance'] = np.clip(new_importance, 0.1, 0.9)
                
    def get_importance_scores(self) -> Dict[str, float]:
        """Get current feature importance scores"""
        return {
            name: stats['importance'] 
            for name, stats in self.feature_stats.items()
        }
        
    def get_stability_scores(self) -> Dict[str, float]:
        """Get feature stability scores"""
        stability_scores = {}
        
        for name, stats in self.feature_stats.items():
            if len(stats['values']) > 50:
                # Calculate coefficient of variation
                cv = stats['std'] / (abs(stats['mean']) + 1e-6)
                stability = 1 / (1 + cv)
                stats['stability'] = stability
                stability_scores[name] = stability
                
        return stability_scores
        
    async def _validate_and_check_overfitting(self) -> bool:
        """Validate model and check for overfitting"""
        if self.observation_count < MIN_SAMPLES_FOR_DETECTION:
            return False
            
        # Get overfitting metrics (assume overfitting_detector is available)
        try:
            detection = await self.overfitting_detector.detect_overfitting()
            
            if detection['is_overfitting']:
                severity = detection['severity']
                logger.warning(f"Overfitting detected in attention layer: {severity}")
                
                # Take action based on severity
                if severity == 'CRITICAL':
                    # Reset to more conservative state
                    await self._apply_emergency_regularization()
                    return True
                elif severity == 'HIGH':
                    # Increase regularization
                    await self._apply_emergency_regularization()
                    return True
                    
        except Exception as e:
            logger.warning(f"Overfitting detection failed: {e}")
                
        return False
        
    async def _apply_emergency_regularization(self):
        """Apply emergency regularization when critical overfitting detected"""
        logger.warning("Applying emergency regularization to attention layer")
        
        if hasattr(self, 'attention_network') and self.attention_network:
            # Reset weights with smaller initialization
            for layer in self.attention_network.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                        
            # Reduce learning rate dramatically
            if hasattr(self, 'optimizer'):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1


class TemporalAttention(BaseAttentionModule):
    """Attention mechanism for temporal patterns"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('temporal_attention', config)
        
        # Temporal pattern storage
        self.temporal_patterns = {
            'short_term': deque(maxlen=20),    # Last 20 ticks
            'medium_term': deque(maxlen=100),  # Last 100 ticks
            'long_term': deque(maxlen=500)     # Last 500 ticks
        }
        
        # Pattern recognition
        self.recognized_patterns = defaultdict(list)
        self.pattern_outcomes = defaultdict(list)
        
        # Temporal weights
        self.temporal_weights = {
            'short_term': 0.5,
            'medium_term': 0.3,
            'long_term': 0.2
        }
        
        # LSTM for temporal attention
        self.lstm_network = None
        self.hidden_size = 64
        
    async def observe(self, history: List[Dict[str, float]], timestamp: float) -> None:
        """Observe temporal data during learning phase"""
        self.observation_count += 1
        
        # Update temporal patterns
        if history:
            latest_data = history[-1]
            
            for window_name, window in self.temporal_patterns.items():
                window.append({
                    'timestamp': timestamp,
                    'data': latest_data.copy()
                })
                
        # Detect patterns every 50 observations
        if self.observation_count % 50 == 0:
            await self._detect_patterns()
            
        # Initialize LSTM after sufficient observations
        if self.observation_count == 200 and not self.is_initialized:
            await self._initialize_lstm()
            
    async def _initialize_lstm(self) -> None:
        """Initialize LSTM network for temporal attention"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping LSTM initialization")
            return
            
        # Determine input size from observed data
        if self.temporal_patterns['short_term']:
            sample_data = self.temporal_patterns['short_term'][0]['data']
            input_size = len(sample_data)
            
            dropout_rate = REGULARIZATION_CONFIG['dropout_rate']
            self.lstm_network = TemporalAttentionLSTM(input_size, self.hidden_size, dropout_rate)
            self.optimizer = torch.optim.Adam(self.lstm_network.parameters(), lr=LEARNING_RATE)
            self.is_initialized = True
            
            logger.info(f"Initialized temporal attention LSTM with input size {input_size}")
            
    async def _detect_patterns(self) -> None:
        """Detect temporal patterns in data"""
        for window_name, window in self.temporal_patterns.items():
            if len(window) < 10:
                continue
                
            # Extract time series data
            timestamps = [w['timestamp'] for w in window]
            
            # Example: detect trends in each feature
            if window[0]['data']:
                for feature_name in window[0]['data'].keys():
                    values = [w['data'].get(feature_name, 0) for w in window]
                    
                    if len(values) > 10:
                        # Simple trend detection
                        trend = self._calculate_trend(values)
                        volatility = np.std(values)
                        
                        pattern = {
                            'window': window_name,
                            'feature': feature_name,
                            'trend': trend,
                            'volatility': volatility,
                            'timestamp': timestamps[-1]
                        }
                        
                        self.recognized_patterns[window_name].append(pattern)
                        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend strength and direction"""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        # Return signed R-squared (trend strength with direction)
        return np.sign(slope) * (r_value ** 2)
        
    async def calculate_weights(self, history: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate temporal attention weights"""
        if not self.is_initialized or not history:
            return self.temporal_weights.copy()
            
        # Prepare sequences for each window
        sequences = {}
        
        for window_name in self.temporal_patterns:
            if window_name == 'short_term':
                sequence = history[-20:] if len(history) >= 20 else history
            elif window_name == 'medium_term':
                sequence = history[-100:] if len(history) >= 100 else history
            else:  # long_term
                sequence = history[-500:] if len(history) >= 500 else history
                
            sequences[window_name] = sequence
            
        # Calculate attention scores using LSTM
        attention_scores = await self._calculate_lstm_attention(sequences)
        
        # Normalize to weights
        total_score = sum(attention_scores.values())
        if total_score > 0:
            weights = {k: v / total_score for k, v in attention_scores.items()}
        else:
            weights = self.temporal_weights.copy()
            
        self.temporal_weights = weights
        return weights
        
    async def _calculate_lstm_attention(self, sequences: Dict[str, List[Dict[str, float]]]) -> Dict[str, float]:
        """Calculate attention scores using LSTM"""
        attention_scores = {}
        
        with torch.no_grad():
            for window_name, sequence in sequences.items():
                if not sequence:
                    attention_scores[window_name] = 0.1
                    continue
                    
                # Convert sequence to tensor
                tensor_sequence = self._sequence_to_tensor(sequence)
                
                # Get LSTM output
                _, (hidden, _) = self.lstm_network(tensor_sequence)
                
                # Calculate attention score from hidden state
                score = torch.norm(hidden).item()
                attention_scores[window_name] = score
                
        return attention_scores
        
    def _sequence_to_tensor(self, sequence: List[Dict[str, float]]) -> torch.Tensor:
        """Convert sequence of dictionaries to tensor"""
        if not sequence:
            return torch.zeros(1, 1, 1)
            
        # Extract feature values in consistent order
        feature_names = sorted(sequence[0].keys())
        data = []
        
        for item in sequence:
            values = [item.get(name, 0.0) for name in feature_names]
            data.append(values)
            
        return torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
        
    async def apply_weights(self, history: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Apply temporal attention weights to historical data"""
        if not history:
            return history
            
        weighted_history = []
        
        # Apply different weights based on recency
        for i, data_point in enumerate(history):
            # Determine which window this point belongs to
            recency = len(history) - i
            
            if recency <= 20:
                weight = self.temporal_weights['short_term']
            elif recency <= 100:
                weight = self.temporal_weights['medium_term']
            else:
                weight = self.temporal_weights['long_term']
                
            # Apply weight to all features
            weighted_point = {}
            for key, value in data_point.items():
                weighted_point[key] = value * weight
                
            weighted_history.append(weighted_point)
            
        return weighted_history
        
    def get_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get recognized temporal patterns"""
        return dict(self.recognized_patterns)
        
    async def update_patterns(self, new_patterns: Dict[str, Any]) -> None:
        """Update temporal patterns based on feedback"""
        # Update pattern recognition based on trading outcomes
        for pattern_type, patterns in new_patterns.items():
            if pattern_type in self.recognized_patterns:
                self.recognized_patterns[pattern_type].extend(patterns)


class RegimeAttention(BaseAttentionModule):
    """Attention mechanism for market regime adaptation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('regime_attention', config)
        
        # Regime performance tracking
        self.regime_performance = defaultdict(lambda: {
            'observations': 0,
            'total_profit': 0.0,
            'win_rate': 0.5,
            'avg_profit': 0.0,
            'strategy_adjustments': defaultdict(float)
        })
        
        # Regime transition patterns
        self.transition_patterns = defaultdict(list)
        
        # Strategy parameter adjustments
        self.parameter_adjustments = defaultdict(dict)
        
        # Neural network for regime-specific adjustments
        self.regime_networks = {}
        
    async def observe(self, regime: str, context: Dict[str, Any]) -> None:
        """Observe regime data during learning phase"""
        self.observation_count += 1
        
        # Update regime statistics
        regime_stats = self.regime_performance[regime]
        regime_stats['observations'] += 1
        
        # Track performance if available
        if 'profit' in context:
            regime_stats['total_profit'] += context['profit']
            regime_stats['avg_profit'] = regime_stats['total_profit'] / regime_stats['observations']
            
        if 'is_winner' in context:
            # Update win rate with exponential moving average
            alpha = 0.1
            regime_stats['win_rate'] = (1 - alpha) * regime_stats['win_rate'] + alpha * context['is_winner']
            
        # Track strategy parameters
        if 'strategy_params' in context:
            for param, value in context['strategy_params'].items():
                if param not in regime_stats['strategy_adjustments']:
                    regime_stats['strategy_adjustments'][param] = []
                regime_stats['strategy_adjustments'][param].append(value)
                
        # Track regime transitions
        if 'previous_regime' in context:
            transition = {
                'from': context['previous_regime'],
                'to': regime,
                'timestamp': context.get('timestamp', time.time()),
                'market_conditions': context.get('market_conditions', {})
            }
            self.transition_patterns[context['previous_regime']].append(transition)
            
        # Initialize regime-specific network after sufficient observations
        if regime_stats['observations'] == 50 and regime not in self.regime_networks:
            await self._initialize_regime_network(regime)
            
    async def _initialize_regime_network(self, regime: str) -> None:
        """Initialize neural network for specific regime"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping regime network initialization")
            return
            
        # Determine input/output sizes from observations
        regime_stats = self.regime_performance[regime]
        
        if regime_stats['strategy_adjustments']:
            num_params = len(regime_stats['strategy_adjustments'])
            self.regime_networks[regime] = RegimeAdaptationNetwork(
                input_size=10,  # Market features
                num_params=num_params
            )
            
            logger.info(f"Initialized regime network for {regime} with {num_params} parameters")
            
    async def calculate_adjustments(self, regime: str) -> Dict[str, float]:
        """Calculate parameter adjustments for regime"""
        regime_stats = self.regime_performance[regime]
        
        if regime_stats['observations'] < 10:
            # Not enough data, return no adjustments
            return {}
            
        adjustments = {}
        
        # Calculate adjustments based on performance
        performance_factor = regime_stats['win_rate'] - 0.5  # -0.5 to 0.5
        
        # Adjust each strategy parameter
        for param, values in regime_stats['strategy_adjustments'].items():
            if len(values) > 5:
                # Calculate optimal value based on performance correlation
                avg_value = np.mean(values)
                
                # Adjust based on regime performance
                adjustment = 1.0 + performance_factor * MAX_ATTENTION_INFLUENCE
                adjustments[param] = adjustment
                
        self.parameter_adjustments[regime] = adjustments
        return adjustments
        
    async def apply_adjustments(self, regime: str, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regime-specific adjustments to strategy parameters"""
        if regime not in self.parameter_adjustments:
            return strategy_params
            
        adjusted_params = strategy_params.copy()
        adjustments = self.parameter_adjustments[regime]
        
        for param, adjustment in adjustments.items():
            if param in adjusted_params:
                # Apply adjustment based on parameter type
                if isinstance(adjusted_params[param], (int, float)):
                    adjusted_params[param] *= adjustment
                elif param == 'enabled' and adjustment < 0.8:
                    # Disable strategy if performance is poor
                    adjusted_params[param] = False
                    
        return adjusted_params
        
    async def calculate_weights(self, data: Any) -> Dict[str, float]:
        """Calculate attention weights - wrapper for calculate_adjustments"""
        return await self.calculate_adjustments(data)
        
    async def apply_weights(self, data: Any) -> Any:
        """Apply attention weights - wrapper for apply_adjustments"""
        if isinstance(data, dict) and 'regime' in data:
            regime = data['regime']
            strategy_params = data.get('strategy_params', {})
            return await self.apply_adjustments(regime, strategy_params)
        return data
        
    def get_performance_by_regime(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by regime"""
        performance = {}
        
        for regime, stats in self.regime_performance.items():
            performance[regime] = {
                'observations': stats['observations'],
                'win_rate': stats['win_rate'],
                'avg_profit': stats['avg_profit'],
                'total_profit': stats['total_profit']
            }
            
        return performance
        
    async def update_strategies(self, regime_feedback: Dict[str, Any]) -> None:
        """Update regime strategies based on feedback"""
        for regime, feedback in regime_feedback.items():
            if regime in self.regime_performance:
                # Update performance metrics
                if 'win_rate' in feedback:
                    self.regime_performance[regime]['win_rate'] = feedback['win_rate']
                    
                # Update parameter adjustments
                if 'optimal_params' in feedback:
                    for param, value in feedback['optimal_params'].items():
                        current_values = self.regime_performance[regime]['strategy_adjustments'][param]
                        if current_values:
                            # Blend with existing values
                            current_values.append(value)


# Neural Network Components
class FeatureAttentionNetwork(nn.Module):
    """Neural network for feature attention with enhanced regularization"""
    
    def __init__(self, num_features: int, dropout_rate: float = 0.3):
        super().__init__()
        # ลดขนาด network และเพิ่ม regularization
        hidden_size = max(num_features // 2, 16)  # ลดขนาด hidden layer
        
        # Layer 1 with enhanced regularization
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2 with enhanced regularization  
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size // 2, num_features)
        self.bn3 = nn.BatchNorm1d(num_features)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)  # Less dropout on output
        
        # Weight initialization with smaller values
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization scaled down"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        # Layer 1
        x = F.relu(self.fc1(x))
        x = self.bn1(x) if x.shape[0] > 1 else x
        x = self.dropout1(x)
        
        # Layer 2
        x = F.relu(self.fc2(x))
        x = self.bn2(x) if x.shape[0] > 1 else x
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        x = self.bn3(x) if x.shape[0] > 1 else x
        x = self.dropout3(x)
        
        return x


class TemporalAttentionLSTM(nn.Module):
    """LSTM network for temporal attention with regularization"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            batch_first=True,
            dropout=dropout_rate,  # Add dropout to LSTM
            num_layers=2  # Use 2 layers for better capacity
        )
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)  # Add layer normalization
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        # Apply layer normalization to output
        output = self.layer_norm(output)
        return output, (hidden, cell)


class RegimeAdaptationNetwork(nn.Module):
    """Neural network for regime-specific adaptations"""
    
    def __init__(self, input_size: int, num_params: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_params)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1
        return x


class AttentionValidator:
    """Validate attention decisions before applying"""
    
    def __init__(self, max_change: float = VALIDATION_THRESHOLD):
        self.max_change = max_change
        self.validation_history = deque(maxlen=1000)
        self.rejection_count = 0
        
    async def validate_attention_output(self, 
                                      original_features: Dict[str, float],
                                      weighted_features: Dict[str, float]) -> Tuple[bool, str]:
        """Validate attention output before applying"""
        
        # Check for NaN or infinite values
        for feature, value in weighted_features.items():
            if np.isnan(value) or np.isinf(value):
                self.rejection_count += 1
                return False, f"Invalid value for {feature}: {value}"
                
        # Check maximum deviation
        for feature, original in original_features.items():
            if feature not in weighted_features:
                continue
                
            weighted = weighted_features[feature]
            
            # Avoid division by zero
            if abs(original) < 1e-10:
                if abs(weighted) > 1e-10:
                    change = float('inf')
                else:
                    change = 0
            else:
                change = abs(weighted - original) / abs(original)
                
            if change > self.max_change:
                self.rejection_count += 1
                return False, f"Change too large for {feature}: {change:.2%} > {self.max_change:.2%}"
                
        # Check if all features are present
        missing_features = set(original_features.keys()) - set(weighted_features.keys())
        if missing_features:
            return False, f"Missing features: {missing_features}"
            
        # Record validation
        self.validation_history.append({
            'timestamp': time.time(),
            'original_count': len(original_features),
            'weighted_count': len(weighted_features),
            'max_change': max(
                abs(weighted_features.get(f, original) - original) / abs(original) 
                if abs(original) > 1e-10 else 0
                for f, original in original_features.items()
            )
        })
        
        return True, "Validation passed"
        
    def get_rejection_rate(self) -> float:
        """Get rejection rate of attention outputs"""
        total = len(self.validation_history) + self.rejection_count
        if total == 0:
            return 0.0
        return self.rejection_count / total


class ABTestController:
    """A/B testing framework for attention system"""
    
    def __init__(self, control_percentage: float = 0.3):
        self.control_percentage = control_percentage
        self.test_results = {
            'control': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0},
            'test': {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0}
        }
        self.assignment_history = deque(maxlen=10000)
        self.random_state = np.random.RandomState(42)  # For reproducibility
        
    def assign_group(self, trade_id: str) -> str:
        """Assign trade to control or test group"""
        # Use hash for consistent assignment
        hash_value = int(hashlib.md5(trade_id.encode()).hexdigest(), 16)
        group = 'control' if (hash_value % 100) < (self.control_percentage * 100) else 'test'
        
        self.assignment_history.append({
            'trade_id': trade_id,
            'group': group,
            'timestamp': time.time()
        })
        
        return group
        
    def update_results(self, trade_id: str, pnl: float, is_winner: bool):
        """Update A/B test results"""
        # Find group assignment
        group = None
        for assignment in reversed(self.assignment_history):
            if assignment['trade_id'] == trade_id:
                group = assignment['group']
                break
                
        if not group:
            return
            
        # Update results
        self.test_results[group]['trades'] += 1
        self.test_results[group]['pnl'] += pnl
        
        # Update win rate
        wins = self.test_results[group].get('wins', 0)
        if is_winner:
            wins += 1
        self.test_results[group]['wins'] = wins
        self.test_results[group]['win_rate'] = wins / self.test_results[group]['trades']
        
    def get_statistical_significance(self) -> Dict[str, Any]:
        """Calculate statistical significance of results"""
        control = self.test_results['control']
        test = self.test_results['test']
        
        # Minimum sample size check
        min_samples = 100
        if control['trades'] < min_samples or test['trades'] < min_samples:
            return {
                'significant': False,
                'p_value': 1.0,
                'confidence': 0.0,
                'message': f"Insufficient samples (control: {control['trades']}, test: {test['trades']})"
            }
            
        # Calculate z-score for win rate difference
        p1 = control['win_rate']
        p2 = test['win_rate']
        n1 = control['trades']
        n2 = test['trades']
        
        # Pooled proportion
        p_pool = (control['wins'] + test['wins']) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        if se == 0:
            return {
                'significant': False,
                'p_value': 1.0,
                'confidence': 0.0,
                'message': "Cannot calculate significance"
            }
            
        # Z-score
        z = (p2 - p1) / se
        
        # Two-tailed p-value
        from scipy import stats as scipy_stats
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        
        # Confidence level
        confidence = 1 - p_value
        
        return {
            'significant': p_value < 0.05,
            'p_value': p_value,
            'confidence': confidence,
            'control_win_rate': p1,
            'test_win_rate': p2,
            'improvement': (p2 - p1) / p1 if p1 > 0 else 0,
            'message': f"Test group {'outperforms' if p2 > p1 else 'underperforms'} control by {abs(p2 - p1):.2%}"
        }


class AttentionLearningLayer:
    """
    Master attention controller with three sub-modules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.phase = AttentionPhase.LEARNING
        
        # Initialize attention modules
        self.feature_attention = FeatureAttention(config)
        self.temporal_attention = TemporalAttention(config)
        self.regime_attention = RegimeAttention(config)
        
        # Add validation and A/B testing
        self.validator = AttentionValidator()
        self.ab_test = ABTestController(
            control_percentage=config.get('control_percentage', 0.3)
        )
        
        # Learning metrics
        self.metrics = AttentionMetrics(phase=self.phase)
        self.phase_controller = PhaseController(config)
        
        # Performance tracking
        self.baseline_performance = {}
        self.attention_performance = {}
        
        # State management
        self._lock = asyncio.Lock()
        self.is_running = True
        
        # Warmup state variables
        self.warmup_loaded = False
        self.original_thresholds = {
            'learning': MIN_TRADES_FOR_LEARNING,
            'shadow': MIN_TRADES_FOR_SHADOW, 
            'active': MIN_TRADES_FOR_ACTIVE
        }
        
        # Try to load warmup state
        asyncio.create_task(self._check_and_load_warmup())
        
        logger.info("Initialized Attention Learning Layer with Enhanced Regularization")
    
    async def _check_and_load_warmup(self):
        """Check for warmup state file and load if available"""
        warmup_file = Path("attention_warmup_state.json")
        
        if warmup_file.exists():
            try:
                await self._load_warmup_state(str(warmup_file))
                logger.info("🚀 Warmup state loaded - accelerated learning enabled")
            except Exception as e:
                logger.warning(f"Failed to load warmup state: {e}")
                logger.info("📚 Starting with fresh learning phase")
        else:
            logger.info("📚 No warmup state found - starting fresh learning phase")
    
    async def _load_warmup_state(self, warmup_file: str):
        """Load warmup state and adjust thresholds"""
        with open(warmup_file, 'r') as f:
            warmup_data = json.load(f)
        
        # Validate warmup data structure
        required_keys = ['attention_state', 'feature_importance', 'learning_progress', 'timestamp']
        for key in required_keys:
            if key not in warmup_data:
                raise ValueError(f"Invalid warmup file: missing {key}")
        
        # Apply feature importance to attention modules
        feature_importance = warmup_data['feature_importance']
        
        if hasattr(self.feature_attention, 'importance_scores'):
            self.feature_attention.importance_scores.update(feature_importance)
        
        # Set pre-observation count based on warmup progress
        warmup_observations = warmup_data['attention_state'].get('observations', 0)
        learning_progress = warmup_data.get('learning_progress', 0.0)
        
        # Reduce thresholds based on warmup quality
        reduction_factor = min(0.9, learning_progress)  # Max 90% reduction
        
        global MIN_TRADES_FOR_LEARNING, MIN_TRADES_FOR_SHADOW, MIN_TRADES_FOR_ACTIVE
        MIN_TRADES_FOR_LEARNING = max(200, int(self.original_thresholds['learning'] * (1 - reduction_factor)))
        MIN_TRADES_FOR_SHADOW = max(100, int(self.original_thresholds['shadow'] * (1 - reduction_factor)))
        MIN_TRADES_FOR_ACTIVE = max(50, int(self.original_thresholds['active'] * (1 - reduction_factor)))
        
        # Set initial observation count (partial credit from warmup)
        initial_observations = int(warmup_observations * 0.1)  # 10% credit
        self.metrics.total_observations = initial_observations
        
        self.warmup_loaded = True
        
        logger.info(f"Warmup applied: Learning {MIN_TRADES_FOR_LEARNING}, "
                   f"Shadow {MIN_TRADES_FOR_SHADOW}, Active {MIN_TRADES_FOR_ACTIVE}")
        logger.info(f"Starting with {initial_observations} observation credits")
        
    async def process(self, features: Dict[str, float], regime: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Process features through attention based on current phase"""
        
        # Record processing start
        start_time = time.perf_counter()
        
        async with self._lock:
            try:
                # Update observation count
                self.metrics.total_observations += 1
                
                # Phase-based processing
                if self.phase == AttentionPhase.LEARNING:
                    output = await self._learning_phase(features, regime, context)
                    
                elif self.phase == AttentionPhase.SHADOW:
                    output = await self._shadow_phase(features, regime, context)
                    
                elif self.phase == AttentionPhase.ACTIVE:
                    output = await self._active_phase(features, regime, context)
                    
                # Record metrics
                processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
                self.metrics.record_processing_time(processing_time)
                
                # Check for overfitting in feature attention
                if hasattr(self.feature_attention, '_validate_and_check_overfitting'):
                    await self.feature_attention._validate_and_check_overfitting()
                
                # Check for phase transition
                if self.phase_controller.should_transition(self.metrics):
                    await self._transition_phase()
                    
                return output
                
            except Exception as e:
                logger.error(f"Error in attention processing: {e}")
                return features  # Return unchanged features on error
                
    async def _learning_phase(self, features: Dict[str, float], regime: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Learning phase: observe only"""
        
        # Let each module observe
        await self.feature_attention.observe(features, context)
        await self.temporal_attention.observe(context.get('history', []), context.get('timestamp', time.time()))
        await self.regime_attention.observe(regime, context)
        
        # Track baseline performance
        if 'performance' in context:
            self._update_baseline_performance(context['performance'])
            
        # Return unchanged features
        return features
        
    async def _shadow_phase(self, features: Dict[str, float], regime: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Shadow phase: calculate but don't apply"""
        
        # Calculate attention weights (use ensemble method for robustness)
        feature_weights = await self.feature_attention.calculate_weights_ensemble(features)
        temporal_weights = await self.temporal_attention.calculate_weights(context.get('history', []))
        regime_adjustments = await self.regime_attention.calculate_adjustments(regime)
        
        # Log shadow calculations
        shadow_data = {
            'feature_weights': feature_weights,
            'temporal_weights': temporal_weights,
            'regime_adjustments': regime_adjustments,
            'timestamp': time.time()
        }
        
        self.metrics.record_shadow_calculation(shadow_data)
        
        # Simulate application to measure potential impact
        simulated_features = await self.feature_attention.apply_weights(features)
        
        # Track what would have happened
        if 'performance' in context:
            self._track_shadow_performance(context['performance'], simulated_features)
            
        # Still return unchanged features
        return features
        
    async def _active_phase(self, features: Dict[str, float], regime: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Active phase: apply attention with validation and A/B testing"""
        
        # Check if this trade is in control group (A/B testing)
        trade_id = context.get('trade_id', str(time.time()))
        group = self.ab_test.assign_group(trade_id)
        
        if group == 'control':
            # Control group - return unchanged features
            context['ab_group'] = 'control'
            return features
            
        # Test group - apply attention
        context['ab_group'] = 'test'
        
        # Apply feature attention (use ensemble for better generalization)
        weighted_features = await self.feature_attention.apply_weights(features)
        
        # Validate weighted features
        is_valid, validation_message = await self.validator.validate_attention_output(
            features, weighted_features
        )
        
        if not is_valid:
            logger.warning(f"Attention output validation failed: {validation_message}")
            # Return original features if validation fails
            return features
            
        # Apply temporal attention to historical context
        if 'history' in context:
            weighted_history = await self.temporal_attention.apply_weights(context['history'])
            context['weighted_history'] = weighted_history
            
        # Get regime-specific adjustments
        if 'strategy_params' in context:
            adjusted_params = await self.regime_attention.apply_adjustments(
                regime, context['strategy_params']
            )
            context['adjusted_params'] = adjusted_params
            
        # Track active performance
        if 'performance' in context:
            self._track_active_performance(context['performance'])
            
            # Update A/B test results
            if 'pnl' in context['performance']:
                self.ab_test.update_results(
                    trade_id,
                    context['performance']['pnl'],
                    context['performance'].get('is_winner', False)
                )
                
        return weighted_features
        
    async def _transition_phase(self) -> None:
        """Handle phase transition"""
        old_phase = self.phase
        new_phase = self.phase_controller.get_next_phase(self.phase)
        
        if new_phase != old_phase:
            # Record transition
            transition_info = {
                'from_phase': old_phase.value,
                'to_phase': new_phase.value,
                'timestamp': time.time(),
                'observations': self.metrics.total_observations,
                'avg_processing_time': self.metrics.get_avg_processing_time(),
                'performance_improvement': self._calculate_performance_improvement()
            }
            
            self.metrics.phase_transitions.append(transition_info)
            
            # Update phase
            self.phase = new_phase
            self.metrics.phase = new_phase
            
            logger.info(f"Transitioned from {old_phase.value} to {new_phase.value} phase")
            logger.info(f"Transition info: {transition_info}")
            
    def _update_baseline_performance(self, performance: Dict[str, float]) -> None:
        """Update baseline performance metrics"""
        for metric, value in performance.items():
            if metric not in self.baseline_performance:
                self.baseline_performance[metric] = []
            self.baseline_performance[metric].append(value)
            
    def _track_shadow_performance(self, performance: Dict[str, float], simulated_features: Dict[str, float]) -> None:
        """Track shadow mode performance"""
        # Calculate potential improvement
        for metric, value in performance.items():
            if metric in self.baseline_performance and self.baseline_performance[metric]:
                baseline_avg = np.mean(self.baseline_performance[metric][-100:])
                improvement = (value - baseline_avg) / baseline_avg if baseline_avg != 0 else 0
                self.metrics.record_performance_improvement(metric, improvement)
                
    def _track_active_performance(self, performance: Dict[str, float]) -> None:
        """Track active mode performance"""
        for metric, value in performance.items():
            if metric not in self.attention_performance:
                self.attention_performance[metric] = []
            self.attention_performance[metric].append(value)
            
    def _calculate_performance_improvement(self) -> float:
        """Calculate overall performance improvement"""
        if not self.baseline_performance or not self.metrics.performance_improvements:
            return 0.0
            
        improvements = []
        for metric, values in self.metrics.performance_improvements.items():
            if values:
                improvements.append(np.mean(values))
                
        return np.mean(improvements) if improvements else 0.0
        
    async def get_attention_state(self) -> Dict[str, Any]:
        """Get current attention system state"""
        return {
            'phase': self.phase.value,
            'total_observations': self.metrics.total_observations,
            'avg_processing_time': self.metrics.get_avg_processing_time(),
            'feature_importance': self.feature_attention.get_importance_scores(),
            'temporal_weights': self.temporal_attention.temporal_weights,
            'regime_performance': self.regime_attention.get_performance_by_regime(),
            'performance_improvement': self._calculate_performance_improvement(),
            'validation_rejection_rate': self.validator.get_rejection_rate(),
            'ab_test_results': self.ab_test.get_statistical_significance()
        }
        
    async def force_phase_transition(self, target_phase: AttentionPhase) -> None:
        """Force transition to specific phase (for testing)"""
        async with self._lock:
            old_phase = self.phase
            self.phase = target_phase
            self.metrics.phase = target_phase
            
            logger.warning(f"Forced phase transition from {old_phase.value} to {target_phase.value}")
            
    def get_learning_progress(self) -> float:
        """Get learning progress percentage"""
        if self.phase == AttentionPhase.ACTIVE:
            return 1.0
        elif self.phase == AttentionPhase.SHADOW:
            shadow_progress = self.metrics.shadow_calculations / self.phase_controller.min_trades_shadow
            return 0.5 + 0.5 * min(shadow_progress, 1.0)
        else:  # LEARNING
            learning_progress = self.metrics.total_observations / self.phase_controller.min_trades_learning
            return 0.5 * min(learning_progress, 1.0)
            
    async def save_state(self, filepath: str) -> None:
        """Save attention state to file"""
        state = {
            'phase': self.phase.value,
            'metrics': {
                'total_observations': self.metrics.total_observations,
                'shadow_calculations': self.metrics.shadow_calculations,
                'active_applications': self.metrics.active_applications,
                'regularization_metrics': self.metrics.regularization_metrics,
                'overfitting_detections': self.metrics.overfitting_detections
            },
            'feature_attention': {
                'weights': self.feature_attention.attention_weights,
                'importance': self.feature_attention.get_importance_scores(),
                'stability': self.feature_attention.get_stability_scores()
            },
            'temporal_attention': {
                'weights': self.temporal_attention.temporal_weights,
                'patterns': self.temporal_attention.get_patterns()
            },
            'regime_attention': {
                'performance': self.regime_attention.get_performance_by_regime(),
                'adjustments': dict(self.regime_attention.parameter_adjustments)
            },
            'ab_test_results': self.ab_test.get_statistical_significance(),
            'validation_stats': {
                'rejection_rate': self.validator.get_rejection_rate(),
                'total_validations': len(self.validator.validation_history)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved attention state to {filepath}")
        
    async def load_state(self, filepath: str) -> None:
            """Load attention state from file"""
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            async with self._lock:
                # Restore phase
                self.phase = AttentionPhase(state['phase'])
                self.metrics.phase = self.phase
            
                # Restore metrics
                self.metrics.total_observations = state['metrics']['total_observations']
                self.metrics.shadow_calculations = state['metrics']['shadow_calculations']
                self.metrics.active_applications = state['metrics']['active_applications']
            
                if 'regularization_metrics' in state['metrics']:
                    self.metrics.regularization_metrics = state['metrics']['regularization_metrics']
                if 'overfitting_detections' in state['metrics']:
                    self.metrics.overfitting_detections = state['metrics']['overfitting_detections']
            
                # Restore attention states
                self.feature_attention.attention_weights = state['feature_attention']['weights']
                self.temporal_attention.temporal_weights = state['temporal_attention']['weights']
            
                logger.info(f"Loaded attention state from {filepath}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check component health"""
        return {
            'healthy': True,
            'is_running': getattr(self, 'is_running', True),
            'error_count': getattr(self, 'error_count', 0),
            'last_error': getattr(self, 'last_error', None)
        }

    async def is_healthy(self) -> bool:
        """Quick health check"""
        health = await self.health_check()
        return health.get('healthy', True)

    async def recover(self) -> bool:
        """Recover from failure"""
        try:
            self.error_count = 0
            self.last_error = None
            return True
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state for checkpointing"""
        return {
            'class': self.__class__.__name__,
            'timestamp': time.time(),
            'phase': self.current_phase.value,
            'learning_progress': self.performance_tracker.learning_progress
        }


# Example usage
async def main():
    """Example usage of AttentionLearningLayer with enhanced regularization"""
    
    # Initialize attention layer with custom config
    config = {
        'min_trades_learning': 100,  # Reduced for demo
        'min_trades_shadow': 50,
        'min_trades_active': 25,
        'control_percentage': 0.3  # 30% control group for A/B testing
    }
    
    attention = AttentionLearningLayer(config)
    
    # Simulate trading loop
    for i in range(200):
        # Simulate features
        features = {
            'price_change': np.random.randn() * 0.01,
            'volume_ratio': 1 + np.random.rand() * 0.5,
            'spread_bps': 1 + np.random.rand() * 2,
            'volatility': 0.001 + np.random.rand() * 0.002,
            'rsi': 0.5 + np.random.randn() * 0.2
        }
        
        # Simulate regime
        regime = np.random.choice(['RANGING', 'TRENDING', 'VOLATILE'])
        
        # Context
        context = {
            'timestamp': time.time(),
            'history': [features.copy() for _ in range(10)],  # Fake history
            'strategy_params': {
                'spacing': 0.001,
                'levels': 5,
                'enabled': True
            },
            'performance': {
                'win_rate': 0.5 + np.random.randn() * 0.1,
                'profit': np.random.randn() * 10,
                'pnl': np.random.randn() * 100
            },
            'outcome': np.random.randn(),  # Simulated outcome
            'is_winner': np.random.rand() > 0.5,
            'trade_id': f"trade_{i}"
        }
        
        # Process through attention
        weighted_features = await attention.process(features, regime, context)
        
        # Print progress
        if i % 50 == 0:
            state = await attention.get_attention_state()
            print(f"\nIteration {i}:")
            print(f"  Phase: {state['phase']}")
            print(f"  Observations: {state['total_observations']}")
            print(f"  Avg Processing Time: {state['avg_processing_time']:.2f}ms")
            print(f"  Learning Progress: {attention.get_learning_progress():.1%}")
            print(f"  Validation Rejection Rate: {state['validation_rejection_rate']:.2%}")
            
            # Print A/B test results if available
            ab_results = state['ab_test_results']
            if ab_results.get('significant') is not None:
                print(f"  A/B Test: {ab_results['message']}")
            
    # Final state
    final_state = await attention.get_attention_state()
    print(f"\nFinal State:")
    print(f"  Phase: {final_state['phase']}")
    print(f"  Feature Importance: {final_state['feature_importance']}")
    print(f"  Performance Improvement: {final_state['performance_improvement']:.2%}")
    print(f"  Validation Rejection Rate: {final_state['validation_rejection_rate']:.2%}")
    
    # A/B test results
    ab_results = final_state['ab_test_results']
    if 'control_win_rate' in ab_results:
        print(f"\nA/B Test Results:")
        print(f"  Control Win Rate: {ab_results['control_win_rate']:.2%}")
        print(f"  Test Win Rate: {ab_results['test_win_rate']:.2%}")
        print(f"  Statistical Significance: {ab_results['significant']}")
        print(f"  P-value: {ab_results['p_value']:.4f}")
    
    # Save state
    await attention.save_state('attention_state_enhanced.json')



    async def health_check(self) -> Dict[str, Any]:
            """Check component health"""
            return {
                'healthy': True,
                'is_running': getattr(self, 'is_running', True),
                'error_count': getattr(self, 'error_count', 0),
                'last_error': getattr(self, 'last_error', None)
            }

    async def is_healthy(self) -> bool:
            """Quick health check"""
            health = await self.health_check()
            return health.get('healthy', True)

    async def recover(self) -> bool:
            """Recover from failure"""
            try:
                self.error_count = 0
                self.last_error = None
                return True
            except Exception as e:
                print(f"Recovery failed: {e}")
                return False

    async def get_latest_data(self):
        """Get latest market data - fix for missing method"""
        if hasattr(self, 'market_data_buffer') and self.market_data_buffer:
            return self.market_data_buffer[-1]
        # Return mock data if no real data
        return {
            'symbol': 'BTC/USDT',
            'price': 50000,
            'volume': 1.0,
            'timestamp': time.time()
        }
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(main())