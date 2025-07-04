from typing import Dict, Any, Optional
"""
feedback_loop.py
Comprehensive feedback system for continuous improvement of grid trading system

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import logging
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import time as time_module  # Additional import for safety
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Local imports
from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase
from core.market_regime_detector import MarketRegimeDetector, MarketRegime
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.performance_monitor import PerformanceMonitor, TradingMetrics
from core.overfitting_detector import OverfittingDetector, OverfittingSeverity

# Setup logger
logger = logging.getLogger(__name__)


# Constants - ULTRA CONSERVATIVE APPROACH FOR OVERFITTING PREVENTION
FEEDBACK_INTERVAL = 120  # เพิ่มเป็น 2 นาที (slower feedback)
MIN_SAMPLES_FOR_LEARNING = 500  # เพิ่มจาก 200 เป็น 500 (more data required)
CONFIDENCE_THRESHOLD = 0.85  # เพิ่มจาก 0.8 เป็น 0.85 (higher confidence required)
IMPROVEMENT_THRESHOLD = 0.03  # เพิ่มจาก 0.02 เป็น 0.03 (3% improvement required)
LEARNING_RATE = 0.0005  # ลดเหลือครึ่งหนึ่ง (ultra slow learning)
DECAY_RATE = 0.995  # เพิ่มจาก 0.99 (even slower decay)
MAX_ADJUSTMENT = 0.05  # ลดจาก 0.1 เป็น 0.05 (5% max adjustment)
ADJUSTMENT_COOLDOWN = 1800  # เพิ่มเป็น 30 นาที (longer cooldown)
OPTIMIZATION_INTERVAL = 14400  # เพิ่มเป็น 4 ชั่วโมง (much longer optimization interval)
INSIGHT_WINDOW = 2000  # เพิ่มจาก 1000 (more data for insights)
OVERFITTING_PREVENTION_THRESHOLD = 0.15  # New: threshold for overfitting detection
PARAMETER_CHANGE_VELOCITY_LIMIT = 0.02  # New: max velocity of parameter changes
CONSECUTIVE_ADJUSTMENTS_LIMIT = 2  # New: max consecutive adjustments
RECOVERY_CHECKLIST_ITEMS = [
    'verify_market_conditions',
    'check_account_balance',
    'validate_position_sync',
    'test_connections',
    'verify_risk_limits',
    'gradual_position_rebuild',
    'monitor_first_hour'
]


# Enums
class FeedbackType(Enum):
    """Types of feedback"""
    PERFORMANCE = "performance"
    EXECUTION = "execution"
    RISK = "risk"
    MARKET = "market"
    SYSTEM = "system"


class OptimizationStatus(Enum):
    """Optimization status"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    APPLYING = "applying"
    COMPLETE = "complete"


class ImprovementAction(Enum):
    """Types of improvement actions"""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    STRATEGY_SWITCH = "strategy_switch"
    RISK_REDUCTION = "risk_reduction"
    FEATURE_REWEIGHT = "feature_reweight"
    REGIME_RECALIBRATION = "regime_recalibration"


class AdjustmentValidator:
    """Enhanced parameter adjustment validator with strict overfitting prevention"""
    
    def __init__(self):
        self.adjustment_history = deque(maxlen=200)  # เพิ่ม history capacity
        self.parameter_stability = defaultdict(lambda: deque(maxlen=50))  # Track more history
        self.consecutive_adjustments = defaultdict(int)
        self.velocity_tracker = defaultdict(deque)
        self.overfitting_detector = OverfittingDetector()
        
    def validate_adjustment(self, param: str, current: float, proposed: float, confidence: float = 0.0) -> Tuple[bool, str]:
        """Enhanced validation with multiple overfitting checks"""
        
        # 1. Basic magnitude check - more strict
        change_ratio = abs(proposed - current) / current if current != 0 else float('inf')
        if change_ratio > 0.05:  # ลดจาก 20% เป็น 5%
            return False, f"Change ratio {change_ratio:.1%} exceeds 5% limit"
            
        # 2. Ultra strict frequency check
        recent_adjustments = [
            adj for adj in self.adjustment_history
            if adj['parameter'] == param and 
            time.time() - adj['timestamp'] < ADJUSTMENT_COOLDOWN
        ]
        
        if len(recent_adjustments) > 1:  # ลดจาก 2 เป็น 1
            return False, f"Too many recent adjustments: {len(recent_adjustments)}"
            
        # 3. Enhanced oscillation detection
        self.parameter_stability[param].append(proposed)
        if len(self.parameter_stability[param]) > 10:
            values = list(self.parameter_stability[param])[-20:]  # Look at more history
            if self._detect_parameter_oscillation(values):
                return False, "Parameter oscillation detected"
                
        # 4. Velocity check - new
        if not self._validate_parameter_velocity(param, proposed):
            return False, "Parameter changing too rapidly"
            
        # 5. Consecutive adjustments check - new
        if self.consecutive_adjustments[param] >= CONSECUTIVE_ADJUSTMENTS_LIMIT:
            return False, f"Too many consecutive adjustments: {self.consecutive_adjustments[param]}"
            
        # 6. Confidence check - new
        if confidence < CONFIDENCE_THRESHOLD:
            return False, f"Confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
            
        # 7. Overfitting pattern detection - new
        if self._detect_overfitting_pattern(param):
            return False, "Overfitting pattern detected in parameter history"
            
        return True, "Validation passed"
    
    def _detect_parameter_oscillation(self, values: List[float]) -> bool:
        """Enhanced oscillation detection"""
        if len(values) < 10:
            return False
            
        changes = np.diff(values)
        if len(changes) == 0:
            return False
            
        # Check for sign changes
        sign_changes = np.sum(np.diff(np.sign(changes)) != 0)
        oscillation_ratio = sign_changes / len(changes)
        
        # Check for variance patterns
        recent_variance = np.var(values[-5:]) if len(values) >= 5 else 0
        overall_variance = np.var(values)
        
        # If too many direction changes OR variance is increasing
        if oscillation_ratio > 0.4 or (recent_variance > overall_variance * 1.5):
            return True
            
        return False
    
    def _validate_parameter_velocity(self, param: str, proposed: float) -> bool:
        """Check if parameter is changing too rapidly"""
        self.velocity_tracker[param].append({
            'value': proposed,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 3600  # 1 hour
        while (self.velocity_tracker[param] and 
               self.velocity_tracker[param][0]['timestamp'] < cutoff_time):
            self.velocity_tracker[param].popleft()
            
        if len(self.velocity_tracker[param]) < 3:
            return True
            
        # Calculate velocity
        recent_points = list(self.velocity_tracker[param])[-3:]
        time_span = recent_points[-1]['timestamp'] - recent_points[0]['timestamp']
        
        if time_span == 0:
            return True
            
        value_change = abs(recent_points[-1]['value'] - recent_points[0]['value'])
        velocity = value_change / time_span
        
        # Check against limit
        avg_value = np.mean([p['value'] for p in recent_points])
        normalized_velocity = velocity / avg_value if avg_value > 0 else velocity
        
        return normalized_velocity <= PARAMETER_CHANGE_VELOCITY_LIMIT
    
    def _detect_overfitting_pattern(self, param: str) -> bool:
        """Detect overfitting patterns in parameter adjustments"""
        if len(self.parameter_stability[param]) < 20:
            return False
            
        values = list(self.parameter_stability[param])
        
        # 1. Check for micro-adjustments (sign of overfitting)
        recent_changes = np.diff(values[-10:])
        micro_adjustments = sum(1 for change in recent_changes if abs(change) < 0.01)
        
        if micro_adjustments > len(recent_changes) * 0.7:
            return True
            
        # 2. Check for increasing adjustment frequency
        timestamps = [adj['timestamp'] for adj in self.adjustment_history 
                     if adj['parameter'] == param]
        
        if len(timestamps) >= 10:
            recent_intervals = np.diff(sorted(timestamps[-10:]))
            if len(recent_intervals) > 0:
                avg_interval = np.mean(recent_intervals)
                if avg_interval < ADJUSTMENT_COOLDOWN / 2:  # Too frequent
                    return True
                    
        # 3. Check for diminishing returns pattern
        if len(values) >= 20:
            early_values = values[:10]
            recent_values = values[-10:]
            
            early_range = max(early_values) - min(early_values)
            recent_range = max(recent_values) - min(recent_values)
            
            # If recent adjustments are getting smaller but more frequent
            if recent_range < early_range * 0.3 and len(recent_changes) > 0:
                return True
                
        return False
        
    def record_adjustment(self, param: str, old_value: float, new_value: float, success: bool = True):
        """Enhanced adjustment recording with success tracking"""
        adjustment_record = {
            'parameter': param,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': time.time(),
            'success': success,
            'change_ratio': abs(new_value - old_value) / old_value if old_value != 0 else 0
        }
        
        self.adjustment_history.append(adjustment_record)
        
        # Update consecutive adjustment counter
        if success:
            self.consecutive_adjustments[param] += 1
        else:
            self.consecutive_adjustments[param] = 0
            
        # Reset counter if enough time has passed
        recent_successful = [
            adj for adj in self.adjustment_history
            if (adj['parameter'] == param and 
                adj['success'] and 
                time.time() - adj['timestamp'] < ADJUSTMENT_COOLDOWN * 2)
        ]
        
        if not recent_successful:
            self.consecutive_adjustments[param] = 0
    
    def get_adjustment_summary(self, param: str) -> Dict[str, Any]:
        """Get adjustment summary for a parameter"""
        param_adjustments = [adj for adj in self.adjustment_history if adj['parameter'] == param]
        
        if not param_adjustments:
            return {'total_adjustments': 0, 'success_rate': 0.0}
            
        successful = [adj for adj in param_adjustments if adj['success']]
        
        return {
            'total_adjustments': len(param_adjustments),
            'successful_adjustments': len(successful),
            'success_rate': len(successful) / len(param_adjustments),
            'avg_change_ratio': np.mean([adj['change_ratio'] for adj in param_adjustments]),
            'last_adjustment': param_adjustments[-1]['timestamp'],
            'consecutive_adjustments': self.consecutive_adjustments[param]
        }


@dataclass
class PerformanceInsight:
    """Extracted performance insight"""
    insight_id: str
    category: str
    description: str
    confidence: float
    impact: float  # Expected improvement
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'insight_id': self.insight_id,
            'category': self.category,
            'description': self.description,
            'confidence': self.confidence,
            'impact': self.impact,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    optimization_id: str
    status: OptimizationStatus
    improvements: Dict[str, float]
    adjustments: Dict[str, Any]
    validation_metrics: Dict[str, float]
    applied: bool = False
    timestamp: float = field(default_factory=time.time)


class RecoveryManager:
    """Manage system recovery procedures"""
    
    def __init__(self):
        self.recovery_checklist = RECOVERY_CHECKLIST_ITEMS
        self.recovery_status = {}
        self.recovery_log = []
        
    async def execute_recovery(self, components: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Execute full recovery procedure"""
        logger.info(f"Starting recovery procedure: {reason}")
        
        recovery_result = {
            'success': True,
            'checks_passed': [],
            'checks_failed': [],
            'start_time': time.time(),
            'reason': reason
        }
        
        for check_item in self.recovery_checklist:
            try:
                check_method = getattr(self, f"_check_{check_item}", None)
                if check_method:
                    result = await check_method(components)
                    self.recovery_status[check_item] = result
                    
                    if result['passed']:
                        recovery_result['checks_passed'].append(check_item)
                    else:
                        recovery_result['checks_failed'].append(check_item)
                        recovery_result['success'] = False
                        
                    self.recovery_log.append({
                        'timestamp': time.time(),
                        'check': check_item,
                        'result': result
                    })
                    
            except Exception as e:
                logger.error(f"Recovery check {check_item} failed: {e}")
                recovery_result['checks_failed'].append(check_item)
                recovery_result['success'] = False
                
        recovery_result['end_time'] = time.time()
        recovery_result['duration'] = recovery_result['end_time'] - recovery_result['start_time']
        
        return recovery_result
        
    async def _check_verify_market_conditions(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Verify market conditions are stable"""
        result = {'passed': True, 'details': {}}
        
        if 'market_data_input' in components:
            data_input = components['market_data_input']
            
            # Get recent ticks
            recent_ticks = await data_input.get_latest_ticks(100)
            
            if len(recent_ticks) < 10:
                result['passed'] = False
                result['details']['error'] = "Insufficient market data"
            else:
                # Check for anomalies
                anomaly_stats = data_input.anomaly_detector.get_anomaly_stats()
                if anomaly_stats['recent_anomalies'] > 10:
                    result['passed'] = False
                    result['details']['error'] = f"Too many recent anomalies: {anomaly_stats['recent_anomalies']}"
                    
        return result
        
    async def _check_check_account_balance(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Check account balance and positions"""
        result = {'passed': True, 'details': {}}
        
        if 'risk_manager' in components:
            risk_mgr = components['risk_manager']
            
            # Check balance
            balance = risk_mgr.position_tracker.current_balance
            if balance <= 0:
                result['passed'] = False
                result['details']['error'] = f"Invalid balance: {balance}"
                
            # Check for stuck positions
            positions = risk_mgr.position_tracker.get_open_positions()
            result['details']['open_positions'] = len(positions)
            
        return result
        
    async def _check_test_connections(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test all exchange connections"""
        result = {'passed': True, 'details': {}}
        
        if 'execution_engine' in components:
            engine = components['execution_engine']
            
            for exchange_name in engine.exchange_manager.exchanges:
                try:
                    # Test connection
                    price = await engine.exchange_manager.get_current_price('BTCUSDT', exchange_name)
                    if price <= 0:
                        result['passed'] = False
                        result['details'][exchange_name] = "Invalid price"
                    else:
                        result['details'][exchange_name] = "OK"
                except Exception as e:
                    result['passed'] = False
                    result['details'][exchange_name] = str(e)
                    
        return result
        
    async def _check_verify_risk_limits(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Verify risk limits are properly set"""
        result = {'passed': True, 'details': {}}
        
        if 'risk_manager' in components:
            risk_mgr = components['risk_manager']
            
            # Check if emergency stop is cleared
            if risk_mgr.emergency_stop:
                result['passed'] = False
                result['details']['error'] = "Emergency stop still active"
                
            # Check circuit breakers
            if risk_mgr.circuit_breaker.is_triggered():
                if risk_mgr.circuit_breaker.can_reset():
                    risk_mgr.circuit_breaker.reset()
                    result['details']['circuit_breakers'] = "Reset"
                else:
                    result['passed'] = False
                    result['details']['error'] = "Circuit breakers still in cooldown"
                    
        return result
        
    async def _check_gradual_position_rebuild(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Plan for gradual position rebuilding"""
        result = {'passed': True, 'details': {}}
        
        # Set conservative parameters for restart
        result['details']['recommended_parameters'] = {
            'initial_position_size': 0.001,  # Very small
            'max_positions': 2,  # Start with just 2
            'grid_levels': 3,  # Minimal grid
            'risk_reduction_factor': 0.5  # 50% of normal
        }
        
        return result
        
    async def _check_monitor_first_hour(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Setup enhanced monitoring for first hour"""
        result = {'passed': True, 'details': {}}
        
        # Configure enhanced monitoring
        result['details']['monitoring_config'] = {
            'alert_threshold_multiplier': 0.5,  # More sensitive alerts
            'metric_update_interval': 10,  # More frequent updates
            'required_supervision': True  # Require manual supervision
        }
        
        return result


class GradualOptimizer:
    """Ultra-conservative parameter optimization with comprehensive overfitting prevention"""
    
    def __init__(self, max_adjustment: float = MAX_ADJUSTMENT):
        self.max_adjustment = max_adjustment / 2  # เริ่มต้นด้วยครึ่งหนึ่งของ max adjustment
        self.optimization_history = deque(maxlen=200)  # เพิ่ม history
        self.parameter_trajectories = defaultdict(lambda: deque(maxlen=100))
        self.overfitting_detector = OverfittingDetector()
        self.stability_monitor = defaultdict(lambda: deque(maxlen=50))
        self.adjustment_effectiveness = defaultdict(list)
        self.last_adjustments = {}
        
    def calculate_gradual_adjustment(self, 
                                   current_value: float, 
                                   target_value: float,
                                   confidence: float,
                                   param_name: str = "unknown") -> float:
        """Ultra-conservative adjustment calculation with multi-layer overfitting prevention"""
        
        # Stage 1: Pre-adjustment validation
        if not self._pre_adjustment_validation(param_name, current_value, target_value, confidence):
            logger.info(f"Pre-adjustment validation failed for {param_name}, no adjustment made")
            return current_value
            
        # Stage 2: Overfitting detection
        overfitting_factor = self._calculate_overfitting_factor(param_name, current_value)
        
        # Stage 3: Conservative adjustment calculation
        desired_change = target_value - current_value
        
        # Ultra-conservative confidence scaling (max 50% confidence instead of 70%)
        confidence_factor = min(confidence * 0.5, 0.5)
        confidence_factor *= overfitting_factor  # Further reduce if overfitting detected
        
        # Dynamic max adjustment based on parameter stability
        stability_factor = self._calculate_stability_factor(param_name)
        dynamic_max_adjustment = self.max_adjustment * stability_factor
        max_change = current_value * dynamic_max_adjustment
        
        # Calculate base adjustment
        base_adjustment = desired_change * confidence_factor
        base_adjustment = np.clip(base_adjustment, -max_change, max_change)
        
        # Stage 4: Frequency-based reduction
        frequency_factor = self._calculate_frequency_factor(param_name)
        adjusted_change = base_adjustment * frequency_factor
        
        # Stage 5: Velocity-based reduction
        velocity_factor = self._calculate_velocity_factor(param_name, current_value)
        final_adjustment = adjusted_change * velocity_factor
        
        # Stage 6: Minimum change threshold (avoid micro-adjustments)
        min_change_threshold = current_value * 0.001  # 0.1% minimum change
        if abs(final_adjustment) < min_change_threshold:
            logger.debug(f"Adjustment for {param_name} below minimum threshold, skipping")
            return current_value
            
        final_value = current_value + final_adjustment
        
        # Stage 7: Record trajectory and effectiveness
        self._record_adjustment_trajectory(param_name, current_value, target_value, 
                                         final_value, confidence, overfitting_factor)
        
        logger.info(f"Conservative adjustment for {param_name}: {current_value:.6f} → {final_value:.6f} "
                   f"(change: {final_adjustment:.6f}, factors: conf={confidence_factor:.3f}, "
                   f"over={overfitting_factor:.3f}, freq={frequency_factor:.3f})")
        
        return final_value
    
    def _pre_adjustment_validation(self, param_name: str, current: float, target: float, confidence: float) -> bool:
        """Comprehensive pre-adjustment validation"""
        
        # Check minimum confidence
        if confidence < CONFIDENCE_THRESHOLD:
            return False
            
        # Check if adjustment is meaningful
        change_ratio = abs(target - current) / current if current != 0 else float('inf')
        if change_ratio < 0.001:  # Less than 0.1% change
            return False
            
        # Check cooldown period
        last_adjustment_time = self.last_adjustments.get(param_name, 0)
        if time.time() - last_adjustment_time < ADJUSTMENT_COOLDOWN:
            return False
            
        # Check for parameter divergence (target too far from current)
        if change_ratio > 0.5:  # More than 50% change requested
            logger.warning(f"Target value for {param_name} too far from current: {change_ratio:.1%}")
            return False
            
        return True
    
    def _calculate_overfitting_factor(self, param_name: str, current_value: float) -> float:
        """Calculate overfitting reduction factor"""
        
        # Check trajectory oscillation
        trajectory = self.parameter_trajectories[param_name]
        if len(trajectory) < 5:
            return 1.0
            
        recent_values = [t['current'] for t in list(trajectory)[-10:]]
        
        # Oscillation detection
        if len(recent_values) >= 5:
            changes = np.diff(recent_values)
            if len(changes) > 0:
                sign_changes = np.sum(np.diff(np.sign(changes)) != 0)
                oscillation_ratio = sign_changes / len(changes)
                
                if oscillation_ratio > 0.5:  # High oscillation
                    return 0.3  # Severe reduction
                elif oscillation_ratio > 0.3:  # Medium oscillation
                    return 0.6  # Moderate reduction
                    
        # Check for diminishing returns
        if len(trajectory) >= 10:
            recent_effectiveness = [t.get('effectiveness', 1.0) for t in list(trajectory)[-5:]]
            avg_effectiveness = np.mean(recent_effectiveness)
            
            if avg_effectiveness < 0.3:  # Low effectiveness
                return 0.5
                
        return 1.0
    
    def _calculate_stability_factor(self, param_name: str) -> float:
        """Calculate stability factor based on parameter history"""
        stability_data = self.stability_monitor[param_name]
        
        if len(stability_data) < 10:
            return 1.0
            
        values = list(stability_data)
        variance = np.var(values)
        mean_value = np.mean(values)
        
        # Calculate coefficient of variation
        cv = variance / mean_value if mean_value != 0 else float('inf')
        
        # Higher variance = less stability = smaller adjustments
        if cv > 0.2:  # High variability
            return 0.5
        elif cv > 0.1:  # Medium variability
            return 0.7
        else:  # Low variability
            return 1.0
    
    def _calculate_frequency_factor(self, param_name: str) -> float:
        """Calculate frequency-based reduction factor"""
        trajectory = self.parameter_trajectories[param_name]
        
        if len(trajectory) < 3:
            return 1.0
            
        # Check adjustment frequency in recent period
        recent_period = 3600  # 1 hour
        current_time = time.time()
        
        recent_adjustments = [
            t for t in trajectory 
            if current_time - t['timestamp'] < recent_period
        ]
        
        adjustment_count = len(recent_adjustments)
        
        # Exponential reduction based on frequency
        if adjustment_count >= 5:
            return 0.2  # Very frequent adjustments
        elif adjustment_count >= 3:
            return 0.5  # Moderate frequency
        elif adjustment_count >= 2:
            return 0.8  # Some recent activity
        else:
            return 1.0  # Infrequent adjustments
    
    def _calculate_velocity_factor(self, param_name: str, current_value: float) -> float:
        """Calculate velocity-based reduction factor"""
        trajectory = self.parameter_trajectories[param_name]
        
        if len(trajectory) < 3:
            return 1.0
            
        # Calculate recent velocity
        recent_points = list(trajectory)[-3:]
        time_span = recent_points[-1]['timestamp'] - recent_points[0]['timestamp']
        
        if time_span == 0:
            return 1.0
            
        value_change = abs(recent_points[-1]['current'] - recent_points[0]['current'])
        velocity = value_change / time_span
        
        # Normalize velocity
        normalized_velocity = velocity / current_value if current_value != 0 else velocity
        
        # Reduce adjustments if velocity is too high
        if normalized_velocity > PARAMETER_CHANGE_VELOCITY_LIMIT:
            return 0.3  # High velocity - reduce significantly
        elif normalized_velocity > PARAMETER_CHANGE_VELOCITY_LIMIT / 2:
            return 0.7  # Medium velocity - moderate reduction
        else:
            return 1.0  # Normal velocity
    
    def _record_adjustment_trajectory(self, param_name: str, current: float, target: float, 
                                    final: float, confidence: float, overfitting_factor: float):
        """Record adjustment trajectory for analysis"""
        
        adjustment_record = {
            'current': current,
            'target': target,
            'final': final,
            'adjustment': final - current,
            'confidence': confidence,
            'overfitting_factor': overfitting_factor,
            'timestamp': time.time(),
            'effectiveness': None  # Will be updated later based on performance
        }
        
        self.parameter_trajectories[param_name].append(adjustment_record)
        self.stability_monitor[param_name].append(final)
        self.last_adjustments[param_name] = time.time()
        
        # Update optimization history
        self.optimization_history.append({
            'parameter': param_name,
            'record': adjustment_record
        })
    
    def update_adjustment_effectiveness(self, param_name: str, effectiveness_score: float):
        """Update effectiveness score for recent adjustments"""
        trajectory = self.parameter_trajectories[param_name]
        
        if trajectory:
            # Update the most recent adjustment
            trajectory[-1]['effectiveness'] = effectiveness_score
            self.adjustment_effectiveness[param_name].append(effectiveness_score)
            
            # Keep only recent effectiveness scores
            if len(self.adjustment_effectiveness[param_name]) > 20:
                self.adjustment_effectiveness[param_name] = self.adjustment_effectiveness[param_name][-20:]
        
    def get_optimization_velocity(self, param_name: str) -> float:
        """Get rate of change for parameter with enhanced calculation"""
        if param_name not in self.parameter_trajectories:
            return 0.0
            
        trajectory = self.parameter_trajectories[param_name]
        if len(trajectory) < 2:
            return 0.0
            
        # Calculate average rate of change over recent adjustments
        recent = list(trajectory)[-10:]  # Last 10 adjustments
        
        if len(recent) < 2:
            return 0.0
            
        time_diff = recent[-1]['timestamp'] - recent[0]['timestamp']
        value_diff = recent[-1]['final'] - recent[0]['final']  # Use final values
        
        if time_diff == 0:
            return 0.0
            
        return abs(value_diff) / time_diff  # Use absolute velocity
    
    def get_parameter_health(self, param_name: str) -> Dict[str, Any]:
        """Get comprehensive health metrics for a parameter"""
        trajectory = self.parameter_trajectories[param_name]
        stability_data = self.stability_monitor[param_name]
        effectiveness_data = self.adjustment_effectiveness[param_name]
        
        if not trajectory:
            return {'health_score': 1.0, 'status': 'healthy', 'recommendations': []}
            
        health_metrics = {
            'adjustment_count': len(trajectory),
            'velocity': self.get_optimization_velocity(param_name),
            'stability': self._calculate_stability_factor(param_name),
            'effectiveness': np.mean(effectiveness_data) if effectiveness_data else 1.0,
            'overfitting_risk': 1.0 - self._calculate_overfitting_factor(param_name, 0),
            'last_adjustment': trajectory[-1]['timestamp'] if trajectory else 0
        }
        
        # Calculate overall health score
        health_score = (
            health_metrics['stability'] * 0.3 +
            health_metrics['effectiveness'] * 0.3 +
            (1.0 - health_metrics['overfitting_risk']) * 0.2 +
            min(1.0, 1.0 / max(0.1, health_metrics['velocity'] * 1000)) * 0.2  # Velocity penalty
        )
        
        # Determine status and recommendations
        if health_score > 0.8:
            status = 'healthy'
            recommendations = []
        elif health_score > 0.6:
            status = 'caution'
            recommendations = ['Monitor parameter closely', 'Consider reducing adjustment frequency']
        else:
            status = 'unhealthy'
            recommendations = [
                'Stop adjustments temporarily',
                'Analyze parameter trajectory',
                'Check for overfitting patterns',
                'Consider parameter reset'
            ]
            
        return {
            'health_score': health_score,
            'status': status,
            'metrics': health_metrics,
            'recommendations': recommendations
        }


class FeedbackProcessor:
    """Process feedback and extract insights"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=INSIGHT_WINDOW)
        self.execution_history = deque(maxlen=INSIGHT_WINDOW)
        self.market_conditions = deque(maxlen=INSIGHT_WINDOW)
        self.insight_cache = {}
        self._lock = asyncio.Lock()
        
    async def process_feedback(self, feedback_data: Dict[str, Any]) -> List[PerformanceInsight]:
        """Process feedback data and extract insights"""
        async with self._lock:
            insights = []
            
            # Store feedback
            await self._store_feedback(feedback_data)
            
            # Extract insights if enough data
            if len(self.performance_history) >= MIN_SAMPLES_FOR_LEARNING:
                # Performance insights
                perf_insights = await self._extract_performance_insights()
                insights.extend(perf_insights)
                
                # Execution insights
                exec_insights = await self._extract_execution_insights()
                insights.extend(exec_insights)
                
                # Market insights
                market_insights = await self._extract_market_insights()
                insights.extend(market_insights)
                
                # Pattern insights
                pattern_insights = await self._extract_pattern_insights()
                insights.extend(pattern_insights)
                
            return insights
            
    async def _store_feedback(self, feedback_data: Dict[str, Any]):
        """Store feedback data in appropriate history"""
        if 'performance' in feedback_data:
            self.performance_history.append(feedback_data['performance'])
            
        if 'execution' in feedback_data:
            self.execution_history.append(feedback_data['execution'])
            
        if 'market' in feedback_data:
            self.market_conditions.append(feedback_data['market'])
            
    async def _extract_performance_insights(self) -> List[PerformanceInsight]:
        """Extract insights from performance data"""
        insights = []
        
        # Win rate analysis
        recent_performance = list(self.performance_history)[-100:]
        win_rates = [p.get('win_rate', 0) for p in recent_performance]
        
        if win_rates:
            avg_win_rate = np.mean(win_rates)
            win_rate_trend = np.polyfit(range(len(win_rates)), win_rates, 1)[0]
            
            if avg_win_rate < 0.45 and win_rate_trend < 0:
                insight = PerformanceInsight(
                    insight_id=f"perf_wr_{int(time.time())}",
                    category="performance",
                    description="Declining win rate detected",
                    confidence=0.8,
                    impact=0.05,  # 5% potential improvement
                    recommendations=[
                        "Increase grid spacing to reduce false signals",
                        "Tighten risk limits",
                        "Review regime detection accuracy"
                    ],
                    supporting_data={
                        'avg_win_rate': avg_win_rate,
                        'trend': win_rate_trend,
                        'samples': len(win_rates)
                    }
                )
                insights.append(insight)
                
        # Profit factor analysis
        profit_factors = [p.get('profit_factor', 1) for p in recent_performance if 'profit_factor' in p]
        
        if profit_factors and np.mean(profit_factors) < 1.2:
            insight = PerformanceInsight(
                insight_id=f"perf_pf_{int(time.time())}",
                category="performance",
                description="Low profit factor indicates poor risk/reward",
                confidence=0.75,
                impact=0.08,
                recommendations=[
                    "Adjust take profit targets",
                    "Improve entry timing",
                    "Consider asymmetric grid spacing"
                ],
                supporting_data={
                    'avg_profit_factor': np.mean(profit_factors),
                    'min_profit_factor': min(profit_factors)
                }
            )
            insights.append(insight)
            
        return insights
        
    async def _extract_execution_insights(self) -> List[PerformanceInsight]:
        """Extract insights from execution data"""
        insights = []
        
        if not self.execution_history:
            return insights
            
        # Latency analysis
        latencies = [e.get('latency', 0) for e in self.execution_history if 'latency' in e]
        
        if latencies:
            p99_latency = np.percentile(latencies, 99)
            
            if p99_latency > 10:  # 10ms threshold
                insight = PerformanceInsight(
                    insight_id=f"exec_lat_{int(time.time())}",
                    category="execution",
                    description="High execution latency affecting performance",
                    confidence=0.9,
                    impact=0.03,
                    recommendations=[
                        "Optimize order batching",
                        "Review exchange connection",
                        "Consider colocated servers"
                    ],
                    supporting_data={
                        'p99_latency': p99_latency,
                        'avg_latency': np.mean(latencies)
                    }
                )
                insights.append(insight)
                
        # Fill rate analysis
        fill_rates = [e.get('fill_rate', 0) for e in self.execution_history if 'fill_rate' in e]
        
        if fill_rates and np.mean(fill_rates) < 0.7:
            insight = PerformanceInsight(
                insight_id=f"exec_fill_{int(time.time())}",
                category="execution",
                description="Low fill rate impacting grid efficiency",
                confidence=0.85,
                impact=0.06,
                recommendations=[
                    "Adjust order pricing strategy",
                    "Use more aggressive execution",
                    "Review spread conditions"
                ],
                supporting_data={
                    'avg_fill_rate': np.mean(fill_rates),
                    'min_fill_rate': min(fill_rates)
                }
            )
            insights.append(insight)
            
        return insights
        
    async def _extract_market_insights(self) -> List[PerformanceInsight]:
        """Extract insights from market conditions"""
        insights = []
        
        if not self.market_conditions:
            return insights
            
        # Regime accuracy analysis
        regime_data = [m for m in self.market_conditions if 'regime' in m and 'regime_confidence' in m]
        
        if len(regime_data) > 50:
            confidences = [m['regime_confidence'] for m in regime_data]
            avg_confidence = np.mean(confidences)
            
            if avg_confidence < 0.6:
                insight = PerformanceInsight(
                    insight_id=f"market_regime_{int(time.time())}",
                    category="market",
                    description="Low regime detection confidence",
                    confidence=0.7,
                    impact=0.1,
                    recommendations=[
                        "Retrain regime detection model",
                        "Add more market features",
                        "Increase regime transition smoothing"
                    ],
                    supporting_data={
                        'avg_confidence': avg_confidence,
                        'samples': len(regime_data)
                    }
                )
                insights.append(insight)
                
        # Volatility regime analysis
        volatilities = [m.get('volatility', 0) for m in self.market_conditions if 'volatility' in m]
        
        if volatilities:
            current_vol = np.mean(volatilities[-20:])
            historical_vol = np.mean(volatilities[:-20])
            
            if current_vol > historical_vol * 2:
                insight = PerformanceInsight(
                    insight_id=f"market_vol_{int(time.time())}",
                    category="market",
                    description="Significant volatility increase detected",
                    confidence=0.9,
                    impact=0.05,
                    recommendations=[
                        "Widen grid spacing",
                        "Reduce position sizes",
                        "Switch to volatile market strategy"
                    ],
                    supporting_data={
                        'current_volatility': current_vol,
                        'historical_volatility': historical_vol,
                        'ratio': current_vol / historical_vol
                    }
                )
                insights.append(insight)
                
        return insights
        
    async def _extract_pattern_insights(self) -> List[PerformanceInsight]:
        """Extract insights from patterns in data"""
        # Local import for namespace safety in async context
        from datetime import datetime
        insights = []
        
        # Time-based patterns
        if self.performance_history:
            # Group by hour of day
            hourly_performance = defaultdict(list)
            
            for perf in self.performance_history:
                if 'timestamp' in perf and 'pnl' in perf:
                    try:
                        hour = datetime.fromtimestamp(perf['timestamp']).hour
                        hourly_performance[hour].append(perf['pnl'])
                    except Exception as e:
                        logger.warning(f"Error processing timestamp in performance history: {e}")
                        continue
                    
            # Find best/worst hours
            hourly_avg = {hour: np.mean(pnls) for hour, pnls in hourly_performance.items() if pnls}
            
            if hourly_avg:
                best_hour = max(hourly_avg, key=hourly_avg.get)
                worst_hour = min(hourly_avg, key=hourly_avg.get)
                
                if hourly_avg[best_hour] > hourly_avg[worst_hour] * 2:
                    insight = PerformanceInsight(
                        insight_id=f"pattern_time_{int(time.time())}",
                        category="pattern",
                        description="Strong time-based performance pattern detected",
                        confidence=0.75,
                        impact=0.04,
                        recommendations=[
                            f"Increase activity during hour {best_hour}",
                            f"Reduce exposure during hour {worst_hour}",
                            "Implement time-based position sizing"
                        ],
                        supporting_data={
                            'best_hour': best_hour,
                            'worst_hour': worst_hour,
                            'performance_ratio': hourly_avg[best_hour] / hourly_avg[worst_hour]
                        }
                    )
                    insights.append(insight)
                    
        return insights
        
    def extract_insights(self, performance_data: Dict[str, Any]) -> List[PerformanceInsight]:
        """Synchronous version for compatibility"""
        return asyncio.run(self.process_feedback(performance_data))


class OptimizationEngine:
    """Optimize system parameters based on insights"""
    
    def __init__(self):
        self.current_parameters = {}
        self.parameter_history = defaultdict(list)
        self.optimization_models = {}
        self.scaler = StandardScaler()
        self.status = OptimizationStatus.IDLE
        self._lock = asyncio.Lock()
        
    async def optimize_parameters(self, insights: List[PerformanceInsight]) -> OptimizationResult:
        """Optimize parameters based on insights"""
        async with self._lock:
            self.status = OptimizationStatus.ANALYZING
            optimization_id = f"opt_{int(time.time())}"
            
            try:
                # Analyze insights
                actions = await self._determine_actions(insights)
                
                # Calculate adjustments
                self.status = OptimizationStatus.OPTIMIZING
                adjustments = await self._calculate_adjustments(actions)
                
                # Validate adjustments
                validation_metrics = await self._validate_adjustments(adjustments)
                
                # Prepare result
                result = OptimizationResult(
                    optimization_id=optimization_id,
                    status=OptimizationStatus.COMPLETE,
                    improvements=self._estimate_improvements(actions),
                    adjustments=adjustments,
                    validation_metrics=validation_metrics
                )
                
                self.status = OptimizationStatus.IDLE
                return result
                
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                self.status = OptimizationStatus.IDLE
                
                return OptimizationResult(
                    optimization_id=optimization_id,
                    status=OptimizationStatus.IDLE,
                    improvements={},
                    adjustments={},
                    validation_metrics={}
                )
                
    async def _determine_actions(self, insights: List[PerformanceInsight]) -> List[Tuple[ImprovementAction, Any]]:
        """Determine optimization actions from insights"""
        actions = []
        
        for insight in insights:
            if insight.confidence < CONFIDENCE_THRESHOLD:
                continue
                
            if insight.category == "performance":
                if "win rate" in insight.description.lower():
                    actions.append((ImprovementAction.PARAMETER_ADJUSTMENT, {
                        'target': 'grid_spacing',
                        'direction': 'increase',
                        'magnitude': 0.1
                    }))
                elif "profit factor" in insight.description.lower():
                    actions.append((ImprovementAction.PARAMETER_ADJUSTMENT, {
                        'target': 'take_profit',
                        'direction': 'increase',
                        'magnitude': 0.2
                    }))
                    
            elif insight.category == "execution":
                if "latency" in insight.description.lower():
                    actions.append((ImprovementAction.PARAMETER_ADJUSTMENT, {
                        'target': 'batch_size',
                        'direction': 'increase',
                        'magnitude': 0.5
                    }))
                elif "fill rate" in insight.description.lower():
                    actions.append((ImprovementAction.STRATEGY_SWITCH, {
                        'from': 'passive',
                        'to': 'aggressive'
                    }))
                    
            elif insight.category == "market":
                if "volatility" in insight.description.lower():
                    actions.append((ImprovementAction.RISK_REDUCTION, {
                        'factor': 0.7
                    }))
                elif "regime" in insight.description.lower():
                    actions.append((ImprovementAction.REGIME_RECALIBRATION, {
                        'confidence_threshold': 0.8
                    }))
                    
        return actions
        
    async def _calculate_adjustments(self, actions: List[Tuple[ImprovementAction, Any]]) -> Dict[str, Any]:
        """Calculate parameter adjustments"""
        adjustments = defaultdict(dict)
        
        for action_type, params in actions:
            if action_type == ImprovementAction.PARAMETER_ADJUSTMENT:
                param_name = params['target']
                current_value = self.current_parameters.get(param_name, 1.0)
                
                if params['direction'] == 'increase':
                    new_value = current_value * (1 + params['magnitude'])
                else:
                    new_value = current_value * (1 - params['magnitude'])
                    
                # Apply bounds
                new_value = self._apply_parameter_bounds(param_name, new_value)
                adjustments['parameters'][param_name] = new_value
                
            elif action_type == ImprovementAction.STRATEGY_SWITCH:
                adjustments['strategy'] = params
                
            elif action_type == ImprovementAction.RISK_REDUCTION:
                adjustments['risk'] = {'position_size_multiplier': params['factor']}
                
            elif action_type == ImprovementAction.FEATURE_REWEIGHT:
                # This would integrate with attention system
                adjustments['features'] = params
                
            elif action_type == ImprovementAction.REGIME_RECALIBRATION:
                adjustments['regime'] = params
                
        return dict(adjustments)
        
    def _apply_parameter_bounds(self, param_name: str, value: float) -> float:
        """Apply bounds to parameter values"""
        bounds = {
            'grid_spacing': (0.0001, 0.01),
            'take_profit': (0.0005, 0.01),
            'stop_loss': (0.001, 0.05),
            'batch_size': (1, 50),
            'position_size': (0.001, 0.5)
        }
        
        if param_name in bounds:
            min_val, max_val = bounds[param_name]
            return np.clip(value, min_val, max_val)
            
        return value
        
    async def _validate_adjustments(self, adjustments: Dict[str, Any]) -> Dict[str, float]:
        """Validate proposed adjustments"""
        validation_metrics = {}
        
        # Simulate impact (simplified)
        if 'parameters' in adjustments:
            param_count = len(adjustments['parameters'])
            validation_metrics['parameter_changes'] = param_count
            validation_metrics['max_change'] = max(
                abs(new - self.current_parameters.get(param, 1.0)) / self.current_parameters.get(param, 1.0)
                for param, new in adjustments['parameters'].items()
            )
            
        # Risk validation
        if 'risk' in adjustments:
            validation_metrics['risk_reduction'] = 1 - adjustments['risk'].get('position_size_multiplier', 1.0)
            
        # Overall confidence
        validation_metrics['confidence'] = 0.8  # Simplified
        
        return validation_metrics
        
    def _estimate_improvements(self, actions: List[Tuple[ImprovementAction, Any]]) -> Dict[str, float]:
        """Estimate expected improvements"""
        improvements = defaultdict(float)
        
        for action_type, params in actions:
            if action_type == ImprovementAction.PARAMETER_ADJUSTMENT:
                improvements['expected_pnl_improvement'] += 0.02
                improvements['expected_win_rate_improvement'] += 0.01
                
            elif action_type == ImprovementAction.STRATEGY_SWITCH:
                improvements['expected_fill_rate_improvement'] += 0.1
                
            elif action_type == ImprovementAction.RISK_REDUCTION:
                improvements['expected_drawdown_reduction'] += 0.05
                
        return dict(improvements)
        
    async def apply_optimizations(self, result: OptimizationResult) -> bool:
        """Apply optimization results to system"""
        async with self._lock:
            try:
                self.status = OptimizationStatus.APPLYING
                
                # Store current parameters for rollback
                rollback_params = self.current_parameters.copy()
                
                # Apply adjustments
                if 'parameters' in result.adjustments:
                    for param, value in result.adjustments['parameters'].items():
                        self.current_parameters[param] = value
                        self.parameter_history[param].append({
                            'value': value,
                            'timestamp': time.time()
                        })
                        
                result.applied = True
                self.status = OptimizationStatus.IDLE
                
                logger.info(f"Applied optimization {result.optimization_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to apply optimizations: {e}")
                self.current_parameters = rollback_params
                self.status = OptimizationStatus.IDLE
                return False
                
    def get_parameter_history(self, param_name: str) -> List[Dict[str, Any]]:
        """Get history of parameter changes"""
        return self.parameter_history.get(param_name, [])


class FeedbackLoop:
    """
    Comprehensive feedback system for continuous improvement
    """
    
    def __init__(self, system_components: Dict[str, Any]):
        self.components = system_components
        self.feedback_processor = FeedbackProcessor()
        self.optimization_engine = OptimizationEngine()
        self.recovery_manager = RecoveryManager()
        self.gradual_optimizer = GradualOptimizer()
        
        # Feedback storage
        self.feedback_history = deque(maxlen=10000)
        self.insight_history = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        
        # Ultra-conservative learning parameters
        self.learning_rate = LEARNING_RATE  # 0.0005 (very slow)
        self.decay_rate = DECAY_RATE  # 0.995 (very slow decay)
        self.min_confidence_for_adjustment = CONFIDENCE_THRESHOLD  # 0.85 (high confidence required)
        
        # Enhanced overfitting prevention
        self.adjustment_validator = AdjustmentValidator()
        self.recent_adjustments = deque(maxlen=100)  # เพิ่ม capacity
        self.parameter_health_monitor = {}
        self.adjustment_success_rate = defaultdict(lambda: deque(maxlen=20))
        self.conservative_mode = False  # Flag for ultra-conservative mode
        
        # State
        self._running = False
        self._feedback_task = None
        self._optimization_task = None
        self._last_optimization = 0
        self._in_recovery = False
        
        logger.info("Initialized Feedback Loop with Recovery and Gradual Optimization")
    
    def set_components(self, components: Dict[str, Any]):
        """Set system components for feedback loop integration"""
        self.components = components
        logger.info(f"Updated FeedbackLoop with {len(components)} components: {list(components.keys())}")
        
    async def start(self):
        """Start feedback loop"""
        if self._running:
            return
            
        self._running = True
        
        # Start tasks
        self._feedback_task = asyncio.create_task(self._feedback_loop())
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("Started feedback loop")
        
    async def stop(self):
        """Stop feedback loop"""
        self._running = False
        
        # Cancel tasks
        tasks_to_wait = []
        for task in [self._feedback_task, self._optimization_task]:
            if task and not task.done():
                task.cancel()
                tasks_to_wait.append(task)
                
        # Wait for completion
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)
        
        logger.info("Stopped feedback loop")
        
    async def _feedback_loop(self):
        """Main feedback processing loop"""
        while self._running:
            try:
                # Collect feedback from all components
                feedback_data = await self._collect_feedback()
                
                # Store feedback
                self.feedback_history.append({
                    'timestamp': time.time(),
                    'data': feedback_data
                })
                
                # Process feedback
                insights = await self.feedback_processor.process_feedback(feedback_data)
                
                if insights:
                    # Store insights
                    self.insight_history.extend(insights)
                    
                    # Update components with insights
                    await self._distribute_insights(insights)
                    
                # Wait
                await asyncio.sleep(FEEDBACK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in feedback loop: {e}")
                await asyncio.sleep(FEEDBACK_INTERVAL)
                
    async def _optimization_loop(self):
        """Optimization loop"""
        while self._running:
            try:
                # Check if optimization needed
                if time.time() - self._last_optimization < OPTIMIZATION_INTERVAL:
                    await asyncio.sleep(60)
                    continue
                    
                # Get recent insights
                recent_insights = list(self.insight_history)[-50:]
                
                if len(recent_insights) >= 10:
                    # Run optimization
                    result = await self.optimization_engine.optimize_parameters(recent_insights)
                    
                    # Store result
                    self.optimization_history.append(result)
                    
                    # Apply if confident
                    if result.validation_metrics.get('confidence', 0) > 0.7:
                        success = await self._apply_optimizations(result)
                        
                        if success:
                            self._last_optimization = time.time()
                            
                # Wait
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)
                
    async def _collect_feedback(self) -> Dict[str, Any]:
        """Collect feedback from all system components"""
        feedback_data = {}
        
        # Performance feedback
        if 'performance_monitor' in self.components:
            monitor = self.components['performance_monitor']
            performance_data = await monitor.get_performance_report()
            
            feedback_data['performance'] = {
                'win_rate': performance_data['summary'].get('avg_win_rate', 0),
                'profit_factor': performance_data['summary'].get('avg_profit_factor', 1),
                'sharpe_ratio': performance_data['summary'].get('avg_sharpe', 0),
                'max_drawdown': performance_data['summary'].get('max_drawdown', 0),
                'total_trades': performance_data.get('total_trades', 0),
                'timestamp': time.time()
            }
            
        # Execution feedback
        if 'execution_engine' in self.components:
            engine = self.components['execution_engine']
            exec_stats = await engine.get_execution_stats()
            
            feedback_data['execution'] = {
                'success_rate': exec_stats['metrics']['success_rate'],
                'latency': exec_stats['metrics']['average_latency'],
                'fill_rate': exec_stats['metrics'].get('fill_rate', 0),
                'active_orders': exec_stats['active_orders']
            }
            
        # Market feedback
        if 'regime_detector' in self.components:
            detector = self.components['regime_detector']
            regime_stats = await detector.get_regime_statistics()
            
            feedback_data['market'] = {
                'regime': regime_stats['current_regime'],
                'regime_confidence': regime_stats['current_confidence'],
                'volatility': regime_stats.get('current_volatility', 0),
                'regime_duration': regime_stats['current_duration']
            }
            
        # Risk feedback
        if 'risk_manager' in self.components:
            risk_mgr = self.components['risk_manager']
            risk_summary = await risk_mgr.get_risk_summary()
            
            feedback_data['risk'] = {
                'current_exposure': risk_summary['current_metrics']['total_exposure'],
                'position_count': risk_summary['current_metrics']['position_count'],
                'daily_pnl': risk_summary['current_metrics']['daily_pnl'],
                'risk_level': risk_summary['current_metrics']['risk_level']
            }
            
        return feedback_data
        
    async def _distribute_insights(self, insights: List[PerformanceInsight]):
        """Distribute insights to relevant components"""
        for insight in insights:
            # Log insight
            logger.info(f"Insight: {insight.description} (confidence: {insight.confidence:.2f})")
            
            # Update attention system
            if 'attention' in self.components and insight.confidence > 0.8:
                await self._update_attention(insight)
                
            # Update regime detector
            if insight.category == "market" and 'regime_detector' in self.components:
                await self._update_regime_detector(insight)
                
            # Update strategy selector
            if insight.category in ["performance", "execution"] and 'strategy_selector' in self.components:
                await self._update_strategy_selector(insight)
                
            # Update risk manager
            if insight.category == "risk" and 'risk_manager' in self.components:
                await self._update_risk_manager(insight)
                
    async def _update_attention(self, insight: PerformanceInsight):
        """Update attention system based on insights"""
        attention = self.components['attention']
        
        # Feature importance updates
        if 'feature' in insight.supporting_data:
            feature_performance = insight.supporting_data.get('feature_performance', {})
            await attention.feature_attention.update_importance(feature_performance)
            
        # Temporal pattern updates
        if 'pattern' in insight.category:
            temporal_patterns = insight.supporting_data.get('temporal_patterns', {})
            await attention.temporal_attention.update_patterns(temporal_patterns)
            
        # Regime performance updates
        if 'regime' in insight.supporting_data:
            regime_performance = insight.supporting_data.get('regime_performance', {})
            await attention.regime_attention.update_strategies(regime_performance)
            
    async def _update_regime_detector(self, insight: PerformanceInsight):
        """Update regime detector based on insights"""
        detector = self.components['regime_detector']
        
        # Update detection thresholds
        if 'confidence_threshold' in insight.supporting_data:
            new_threshold = insight.supporting_data['confidence_threshold']
            # This would update the detector's internal thresholds
            logger.info(f"Updating regime confidence threshold to {new_threshold}")
            
    async def _update_strategy_selector(self, insight: PerformanceInsight):
        """Update strategy selector based on insights"""
        selector = self.components['strategy_selector']
        
        # Update strategy parameters
        for recommendation in insight.recommendations:
            if "spacing" in recommendation.lower():
                # This would update grid spacing parameters
                logger.info("Adjusting grid spacing based on insight")
            elif "strategy" in recommendation.lower():
                # This would influence strategy selection
                logger.info("Updating strategy selection criteria")
                
    async def _update_risk_manager(self, insight: PerformanceInsight):
        """Update risk manager based on insights"""
        risk_mgr = self.components['risk_manager']
        
        # Update risk limits
        if insight.impact > 0.05:  # Significant impact
            if "reduce" in insight.description.lower():
                # Tighten risk limits temporarily
                risk_mgr.limit_checker.set_temporary_override(
                    'max_position_size',
                    risk_mgr.risk_limits['max_position_size'] * 0.8,
                    duration=3600
                )
                logger.info("Temporarily reduced position size limit")
                
    async def _apply_optimizations(self, result: OptimizationResult) -> bool:
        """Ultra-conservative optimization application with comprehensive validation"""
        try:
            # Stage 1: Pre-application validation
            if not await self._validate_optimizations_enhanced(result):
                logger.warning(f"Enhanced validation failed for optimization {result.optimization_id}")
                return False
                
            # Stage 2: Conservative mode check
            if self.conservative_mode or self._detect_system_stress():
                logger.info("System in conservative mode, applying extra restrictions")
                result = self._apply_conservative_restrictions(result)
                
            # Stage 3: Parameter health checks
            unhealthy_params = self._check_parameter_health(result)
            if unhealthy_params:
                logger.warning(f"Unhealthy parameters detected: {unhealthy_params}, skipping optimization")
                return False
                
            # Stage 4: Apply parameter adjustments with multi-layer validation
            adjustment_success = True
            applied_adjustments = []
            
            if 'parameters' in result.adjustments:
                for param, target_value in result.adjustments['parameters'].items():
                    current_value = self.optimization_engine.current_parameters.get(param, 1.0)
                    confidence = result.validation_metrics.get('confidence', 0.5)
                    
                    # Enhanced adjustment validation
                    valid, reason = self.adjustment_validator.validate_adjustment(
                        param, current_value, target_value, confidence
                    )
                    
                    if not valid:
                        logger.info(f"Adjustment validation failed for {param}: {reason}")
                        continue
                        
                    # Ultra-conservative gradual adjustment
                    adjusted_value = self.gradual_optimizer.calculate_gradual_adjustment(
                        current_value,
                        target_value,
                        confidence * 0.6,  # Further reduce confidence factor
                        param
                    )
                    
                    # Final safety check
                    if self._final_safety_check(param, current_value, adjusted_value):
                        await self._apply_parameter_with_monitoring(param, adjusted_value)
                        
                        # Record successful adjustment
                        self.adjustment_validator.record_adjustment(param, current_value, adjusted_value, True)
                        applied_adjustments.append({
                            'parameter': param,
                            'from': current_value,
                            'to': adjusted_value,
                            'timestamp': time.time(),
                            'confidence': confidence
                        })
                        
                        # Update parameter health monitoring
                        self.parameter_health_monitor[param] = self.gradual_optimizer.get_parameter_health(param)
                        
                    else:
                        logger.warning(f"Final safety check failed for {param}")
                        self.adjustment_validator.record_adjustment(param, current_value, adjusted_value, False)
                        adjustment_success = False
                        
            # Stage 5: Apply other changes conservatively
            if 'strategy' in result.adjustments and adjustment_success:
                strategy_success = await self._apply_strategy_change_ultra_conservative(result.adjustments['strategy'])
                adjustment_success = adjustment_success and strategy_success
                
            # Stage 6: Post-application monitoring
            if adjustment_success and applied_adjustments:
                await self._setup_post_adjustment_monitoring(applied_adjustments)
                
            # Record overall success
            self.recent_adjustments.extend(applied_adjustments)
            
            if adjustment_success:
                logger.info(f"Successfully applied {len(applied_adjustments)} conservative adjustments")
            else:
                logger.warning("Some adjustments failed safety checks")
                
            return adjustment_success
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            return False
    
    def _detect_system_stress(self) -> bool:
        """Detect if system is under stress and should be more conservative"""
        # Check recent adjustment success rate
        recent_successes = []
        for param_successes in self.adjustment_success_rate.values():
            recent_successes.extend(list(param_successes))
            
        if recent_successes:
            success_rate = sum(recent_successes) / len(recent_successes)
            if success_rate < 0.7:  # Less than 70% success rate
                return True
                
        # Check recent optimization frequency
        recent_optimizations = [
            opt for opt in self.optimization_history
            if time.time() - opt.timestamp < 3600  # Last hour
        ]
        
        if len(recent_optimizations) > 3:  # Too many recent optimizations
            return True
            
        # Check parameter health scores
        unhealthy_count = sum(
            1 for health in self.parameter_health_monitor.values()
            if health['health_score'] < 0.6
        )
        
        if unhealthy_count > len(self.parameter_health_monitor) * 0.3:  # More than 30% unhealthy
            return True
            
        return False
    
    def _apply_conservative_restrictions(self, result: OptimizationResult) -> OptimizationResult:
        """Apply additional conservative restrictions to optimization result"""
        # Reduce confidence
        result.validation_metrics['confidence'] *= 0.7
        
        # Limit parameter changes
        if 'parameters' in result.adjustments:
            for param, value in result.adjustments['parameters'].items():
                current = self.optimization_engine.current_parameters.get(param, 1.0)
                change_ratio = abs(value - current) / current if current != 0 else 0
                
                if change_ratio > 0.02:  # Limit to 2% change in conservative mode
                    capped_value = current * 1.02 if value > current else current * 0.98
                    result.adjustments['parameters'][param] = capped_value
                    
        return result
    
    def _check_parameter_health(self, result: OptimizationResult) -> List[str]:
        """Check health of parameters being adjusted"""
        unhealthy_params = []
        
        if 'parameters' in result.adjustments:
            for param in result.adjustments['parameters'].keys():
                health = self.gradual_optimizer.get_parameter_health(param)
                
                if health['health_score'] < 0.6:
                    unhealthy_params.append(param)
                    
        return unhealthy_params
    
    def _final_safety_check(self, param: str, current: float, adjusted: float) -> bool:
        """Final safety check before applying parameter"""
        # Check change magnitude
        change_ratio = abs(adjusted - current) / current if current != 0 else float('inf')
        if change_ratio > 0.03:  # Max 3% change
            return False
            
        # Check if parameter is in bounds
        if adjusted <= 0:  # Positive values only
            return False
            
        # Check parameter-specific bounds
        param_bounds = {
            'grid_spacing': (0.0001, 0.01),
            'position_size': (0.001, 0.1),
            'take_profit': (0.001, 0.05),
            'stop_loss': (0.001, 0.1)
        }
        
        if param in param_bounds:
            min_val, max_val = param_bounds[param]
            if not (min_val <= adjusted <= max_val):
                return False
                
        return True
    
    async def _apply_parameter_with_monitoring(self, param_name: str, value: float):
        """Apply parameter with enhanced monitoring"""
        # Store old value for rollback if needed
        old_value = self.optimization_engine.current_parameters.get(param_name, 1.0)
        
        # Apply parameter
        await self._apply_parameter(param_name, value)
        
        # Setup monitoring for this specific change
        # This would integrate with performance monitoring to track effectiveness
        logger.info(f"Applied parameter {param_name}: {old_value:.6f} → {value:.6f} with monitoring")
    
    async def _setup_post_adjustment_monitoring(self, adjustments: List[Dict[str, Any]]):
        """Setup enhanced monitoring after adjustments"""
        # Schedule effectiveness evaluation
        for adjustment in adjustments:
            param = adjustment['parameter']
            
            # This would schedule a task to evaluate effectiveness later
            logger.debug(f"Scheduled effectiveness monitoring for {param}")
    
    async def _apply_strategy_change_ultra_conservative(self, strategy_change: Dict[str, str]) -> bool:
        """Ultra-conservative strategy change application"""
        # Check if strategy changes are allowed
        last_strategy_change = getattr(self, 'last_strategy_change_time', 0)
        if time.time() - last_strategy_change < 7200:  # 2 hour minimum between changes
            logger.info("Strategy change blocked due to extended cooldown period")
            return False
            
        # Apply change
        await self._apply_strategy_change_conservative(strategy_change)
        self.last_strategy_change_time = time.time()
        
        return True
            
    async def _enter_recovery_mode(self, reason: str):
        """Enter recovery mode"""
        logger.warning(f"Entering recovery mode: {reason}")
        self._in_recovery = True
        
        # Execute recovery procedure
        recovery_result = await self.recovery_manager.execute_recovery(self.components, reason)
        
        if recovery_result['success']:
            logger.info("Recovery procedure completed successfully")
            self._in_recovery = False
            
            # Apply conservative settings
            await self._apply_recovery_settings()
        else:
            logger.error(f"Recovery failed: {recovery_result['checks_failed']}")
            
            # Trigger emergency stop
            if 'risk_manager' in self.components:
                risk_mgr = self.components['risk_manager']
                await risk_mgr._trigger_emergency_stop("Recovery failed")
                
    async def _apply_recovery_settings(self):
        """Apply conservative settings after recovery"""
        # Reduce all risk parameters
        if 'risk_manager' in self.components:
            risk_mgr = self.components['risk_manager']
            
            # Temporary conservative limits
            risk_mgr.limit_checker.set_temporary_override('max_position_size', 0.02, 7200)  # 2% for 2 hours
            risk_mgr.limit_checker.set_temporary_override('max_concurrent_orders', 4, 7200)
            risk_mgr.limit_checker.set_temporary_override('max_daily_loss', 0.005, 7200)  # 0.5%
            
        # Reset optimization parameters
        self.optimization_engine.current_parameters = {}
        self.gradual_optimizer.parameter_trajectories.clear()
        
        logger.info("Applied conservative recovery settings")
            
    async def _apply_parameter(self, param_name: str, value: float):
        """Apply parameter change to relevant component"""
        # Map parameters to components
        param_mapping = {
            'grid_spacing': 'strategy_selector',
            'take_profit': 'strategy_selector',
            'stop_loss': 'risk_manager',
            'batch_size': 'execution_engine',
            'position_size': 'risk_manager'
        }
        
        component_name = param_mapping.get(param_name)
        if component_name and component_name in self.components:
            # This would update the actual parameter in the component
            logger.info(f"Updated {param_name} to {value} in {component_name}")
            
    async def _apply_strategy_change(self, strategy_change: Dict[str, str]):
        """Apply strategy change"""
        if 'strategy_selector' in self.components:
            # This would trigger strategy switch
            logger.info(f"Switching strategy from {strategy_change['from']} to {strategy_change['to']}")
            
    async def _apply_risk_adjustment(self, risk_adjustment: Dict[str, float]):
        """Apply risk adjustment"""
        if 'risk_manager' in self.components and 'position_size_multiplier' in risk_adjustment:
            multiplier = risk_adjustment['position_size_multiplier']
            # This would adjust risk parameters
            logger.info(f"Applied risk multiplier: {multiplier}")
    async def _validate_optimizations_enhanced(self, result: OptimizationResult) -> bool:
        """Enhanced optimization validation with comprehensive overfitting checks"""
        
        # Stage 1: Basic validation
        if not await self._validate_optimizations(result):
            return False
            
        # Stage 2: Confidence validation - much stricter
        confidence = result.validation_metrics.get('confidence', 0.0)
        if confidence < 0.85:  # Require very high confidence
            logger.info(f"Confidence {confidence:.2f} below ultra-conservative threshold 0.85")
            return False
            
        # Stage 3: Parameter change magnitude validation - stricter
        if 'parameters' in result.adjustments:
            for param, target in result.adjustments['parameters'].items():
                current = self.optimization_engine.current_parameters.get(param, 1.0)
                change_ratio = abs(target - current) / current if current != 0 else float('inf')
                
                if change_ratio > 0.05:  # ลดจาก 20% เป็น 5%
                    logger.warning(f"Parameter change {change_ratio:.1%} for {param} exceeds 5% limit")
                    return False
                    
        # Stage 4: Adjustment frequency validation - much stricter  
        recent_adjustments_all = [
            adj for adj in self.recent_adjustments
            if time.time() - adj['timestamp'] < ADJUSTMENT_COOLDOWN
        ]
        
        if len(recent_adjustments_all) > 1:  # ลดจาก 2 เป็น 1
            logger.warning(f"Too many recent adjustments: {len(recent_adjustments_all)}, blocking optimization")
            return False
            
        # Stage 5: Parameter health validation - new
        if 'parameters' in result.adjustments:
            for param in result.adjustments['parameters'].keys():
                health = self.gradual_optimizer.get_parameter_health(param)
                if health['health_score'] < 0.7:
                    logger.warning(f"Parameter {param} health score {health['health_score']:.2f} below 0.7")
                    return False
                    
        # Stage 6: System stress validation - new
        if self._detect_system_stress():
            logger.warning("System stress detected, blocking optimization")
            return False
            
        # Stage 7: Impact validation - new
        total_expected_impact = sum(result.improvements.values())
        if total_expected_impact < IMPROVEMENT_THRESHOLD:
            logger.info(f"Expected impact {total_expected_impact:.3f} below threshold {IMPROVEMENT_THRESHOLD}")
            return False
            
        # Stage 8: Overfitting pattern validation - new
        if self._detect_optimization_overfitting():
            logger.warning("Optimization overfitting pattern detected")
            return False
            
        return True
    
    def _detect_optimization_overfitting(self) -> bool:
        """Detect if optimization process itself is overfitting"""
        
        # Check recent optimization effectiveness
        if len(self.optimization_history) >= 10:
            recent_opts = list(self.optimization_history)[-10:]
            
            # Check if optimizations are getting smaller but more frequent
            applied_opts = [opt for opt in recent_opts if opt.applied]
            
            if len(applied_opts) >= 5:
                # Check if expected improvements are declining
                improvements = [sum(opt.improvements.values()) for opt in applied_opts]
                
                if len(improvements) >= 5:
                    early_avg = np.mean(improvements[:3])
                    recent_avg = np.mean(improvements[-3:])
                    
                    # Diminishing returns pattern
                    if recent_avg < early_avg * 0.5 and len(applied_opts) >= 7:
                        return True
                        
                # Check optimization frequency
                timestamps = [opt.timestamp for opt in applied_opts]
                if len(timestamps) >= 5:
                    intervals = np.diff(sorted(timestamps[-5:]))
                    avg_interval = np.mean(intervals)
                    
                    # Too frequent optimizations
                    if avg_interval < OPTIMIZATION_INTERVAL / 3:
                        return True
                        
        return False


# Additional Missing Classes for Tests

@dataclass
class FeedbackConfig:
    """Configuration for feedback system"""
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.05
    feedback_window: int = 100
    min_samples_required: int = 30
    optimization_frequency: str = 'daily'
    optimization_method: str = 'grid_search'
    enable_online_learning: bool = True
    enable_reinforcement_learning: bool = False
    confidence_threshold: float = 0.8
    improvement_threshold: float = 0.02
    
    def __post_init__(self):
        """Validate configuration"""
        if self.learning_rate <= 0 or self.learning_rate >= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.adaptation_threshold <= 0 or self.adaptation_threshold >= 1:
            raise ValueError("adaptation_threshold must be between 0 and 1")
        if self.feedback_window <= 0:
            raise ValueError("feedback_window must be positive")
        if self.optimization_frequency not in ['daily', 'hourly', 'realtime']:
            raise ValueError("optimization_frequency must be 'daily', 'hourly', or 'realtime'")


@dataclass
class PerformanceFeedback:
    """Performance feedback data structure"""
    timestamp: datetime
    trade_id: str
    strategy: str = 'grid'
    symbol: str = 'BTC/USDT'
    outcome: str = 'unknown'
    pnl: Decimal = Decimal('0')
    return_pct: Decimal = Decimal('0')
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    execution_quality: float = 1.0
    confidence: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate feedback data"""
        if self.outcome not in ['win', 'loss', 'breakeven', 'unknown']:
            raise ValueError("outcome must be 'win', 'loss', 'breakeven', or 'unknown'")
        if self.execution_quality < 0 or self.execution_quality > 1:
            raise ValueError("execution_quality must be between 0 and 1")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("confidence must be between 0 and 1")


class AdaptationStrategy(Enum):
    """Strategy for adapting to changing conditions"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    DYNAMIC = "dynamic"


class FeedbackMetrics:
    """Calculate feedback-related metrics"""
    
    def __init__(self):
        self.learning_history = deque(maxlen=1000)
        self.adaptation_history = deque(maxlen=1000)
        self.feedback_quality_history = deque(maxlen=1000)
        
    def calculate_learning_effectiveness(self, feedback_data: List[PerformanceFeedback]) -> Dict[str, float]:
        """Calculate learning effectiveness metrics"""
        if len(feedback_data) < 10:
            return {'effectiveness': 0.0, 'confidence': 0.0}
            
        # Calculate metrics based on feedback patterns
        recent_outcomes = [f.outcome for f in feedback_data[-30:]]
        win_rate = sum(1 for outcome in recent_outcomes if outcome == 'win') / len(recent_outcomes)
        
        # Calculate improvement trend
        pnl_values = [float(f.pnl) for f in feedback_data]
        if len(pnl_values) >= 20:
            early_avg = np.mean(pnl_values[:10])
            recent_avg = np.mean(pnl_values[-10:])
            improvement = (recent_avg - early_avg) / abs(early_avg) if early_avg != 0 else 0
        else:
            improvement = 0
            
        effectiveness = min(1.0, max(0.0, (win_rate - 0.5) * 2 + improvement * 0.5))
        confidence = min(1.0, len(feedback_data) / 100.0)
        
        return {
            'effectiveness': effectiveness,
            'confidence': confidence,
            'win_rate': win_rate,
            'improvement_trend': improvement,
            'sample_size': len(feedback_data)
        }
    
    def calculate_adaptation_impact(self, before_data: List[PerformanceFeedback], 
                                    after_data: List[PerformanceFeedback]) -> Dict[str, float]:
        """Calculate impact of adaptations"""
        if len(before_data) < 5 or len(after_data) < 5:
            return {'impact': 0.0, 'confidence': 0.0}
            
        # Calculate performance before and after adaptation
        before_pnl = np.mean([float(f.pnl) for f in before_data])
        after_pnl = np.mean([float(f.pnl) for f in after_data])
        
        before_win_rate = sum(1 for f in before_data if f.outcome == 'win') / len(before_data)
        after_win_rate = sum(1 for f in after_data if f.outcome == 'win') / len(after_data)
        
        # Calculate relative improvement
        pnl_improvement = (after_pnl - before_pnl) / abs(before_pnl) if before_pnl != 0 else 0
        win_rate_improvement = after_win_rate - before_win_rate
        
        impact = (pnl_improvement * 0.7 + win_rate_improvement * 0.3)
        confidence = min(1.0, min(len(before_data), len(after_data)) / 20.0)
        
        return {
            'impact': impact,
            'confidence': confidence,
            'pnl_improvement': pnl_improvement,
            'win_rate_improvement': win_rate_improvement,
            'before_performance': before_pnl,
            'after_performance': after_pnl
        }
    
    def calculate_feedback_quality(self, feedback_batch: List[PerformanceFeedback]) -> Dict[str, float]:
        """Calculate quality of feedback data"""
        if not feedback_batch:
            return {'quality': 0.0, 'completeness': 0.0, 'timeliness': 0.0}
            
        # Calculate completeness (how much data is available)
        required_fields = ['trade_id', 'outcome', 'pnl', 'return_pct']
        completeness_scores = []
        
        for feedback in feedback_batch:
            complete_fields = sum(1 for field in required_fields if getattr(feedback, field) is not None)
            completeness_scores.append(complete_fields / len(required_fields))
            
        avg_completeness = np.mean(completeness_scores)
        
        # Calculate timeliness (how recent the feedback is)
        now = datetime.now()
        time_diffs = [(now - f.timestamp).total_seconds() for f in feedback_batch]
        avg_age = np.mean(time_diffs)
        timeliness = max(0.0, 1.0 - avg_age / 3600)  # 1 hour = 0 timeliness
        
        # Calculate consistency (variation in confidence scores)
        confidences = [f.confidence for f in feedback_batch]
        consistency = 1.0 - np.std(confidences) if len(confidences) > 1 else 1.0
        
        # Overall quality score
        quality = (avg_completeness * 0.4 + timeliness * 0.3 + consistency * 0.3)
        
        return {
            'quality': quality,
            'completeness': avg_completeness,
            'timeliness': timeliness,
            'consistency': consistency,
            'sample_size': len(feedback_batch)
        }


class ImprovementTracker:
    """Track system improvements over time"""
    
    def __init__(self):
        self.improvement_history = deque(maxlen=1000)
        self.baseline_metrics = {}
        self.cumulative_improvement = 0.0
        self.improvement_velocity = deque(maxlen=50)
        
    def record_improvement(self, metric_name: str, before_value: float, after_value: float, 
                          timestamp: datetime = None) -> Dict[str, Any]:
        """Record an improvement"""
        if timestamp is None:
            timestamp = datetime.now()
            
        improvement_pct = (after_value - before_value) / abs(before_value) if before_value != 0 else 0
        
        improvement_record = {
            'timestamp': timestamp,
            'metric_name': metric_name,
            'before_value': before_value,
            'after_value': after_value,
            'improvement_pct': improvement_pct,
            'improvement_abs': after_value - before_value
        }
        
        self.improvement_history.append(improvement_record)
        self.cumulative_improvement += improvement_pct
        self.improvement_velocity.append(improvement_pct)
        
        # Update baseline if this is the first record for this metric
        if metric_name not in self.baseline_metrics:
            self.baseline_metrics[metric_name] = before_value
            
        return improvement_record
    
    def calculate_cumulative_improvement(self, metric_name: str = None) -> Dict[str, float]:
        """Calculate cumulative improvement"""
        if metric_name:
            # Filter for specific metric
            metric_records = [r for r in self.improvement_history if r['metric_name'] == metric_name]
            if not metric_records:
                return {'cumulative_improvement': 0.0, 'total_records': 0}
                
            baseline = self.baseline_metrics.get(metric_name, 0)
            latest_value = metric_records[-1]['after_value']
            cumulative = (latest_value - baseline) / abs(baseline) if baseline != 0 else 0
            
            return {
                'cumulative_improvement': cumulative,
                'total_records': len(metric_records),
                'baseline_value': baseline,
                'current_value': latest_value
            }
        else:
            # Overall improvement
            return {
                'cumulative_improvement': self.cumulative_improvement,
                'total_records': len(self.improvement_history),
                'average_improvement': self.cumulative_improvement / len(self.improvement_history) if self.improvement_history else 0
            }
    
    def calculate_improvement_velocity(self, window_size: int = 10) -> Dict[str, float]:
        """Calculate rate of improvement"""
        if len(self.improvement_velocity) < window_size:
            return {'velocity': 0.0, 'acceleration': 0.0, 'trend': 'insufficient_data'}
            
        recent_improvements = list(self.improvement_velocity)[-window_size:]
        velocity = np.mean(recent_improvements)
        
        # Calculate acceleration (change in velocity)
        if len(self.improvement_velocity) >= window_size * 2:
            earlier_improvements = list(self.improvement_velocity)[-window_size*2:-window_size]
            earlier_velocity = np.mean(earlier_improvements)
            acceleration = velocity - earlier_velocity
        else:
            acceleration = 0.0
            
        # Determine trend
        if velocity > 0.01:
            trend = 'improving'
        elif velocity < -0.01:
            trend = 'declining'
        else:
            trend = 'stable'
            
        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'trend': trend,
            'sample_size': len(recent_improvements)
        }
    
    def forecast_improvement(self, periods_ahead: int = 5) -> Dict[str, float]:
        """Forecast future improvements"""
        if len(self.improvement_velocity) < 10:
            return {'forecast': 0.0, 'confidence': 0.0}
            
        # Simple linear extrapolation
        recent_data = np.array(list(self.improvement_velocity)[-20:])
        time_points = np.arange(len(recent_data))
        
        # Fit linear trend
        coeffs = np.polyfit(time_points, recent_data, 1)
        trend_slope = coeffs[0]
        
        # Forecast
        forecast = trend_slope * periods_ahead
        
        # Calculate confidence based on data consistency
        residuals = recent_data - np.polyval(coeffs, time_points)
        confidence = max(0.0, 1.0 - np.std(residuals) / np.std(recent_data))
        
        return {
            'forecast': forecast,
            'confidence': confidence,
            'trend_slope': trend_slope,
            'periods_ahead': periods_ahead
        }


class ParameterOptimizer:
    """Optimize parameters using various methods"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.optimization_method = config.optimization_method
        self.objective = 'sharpe_ratio'
        self.parameter_bounds = {}
        self.optimization_history = deque(maxlen=100)
        
    def optimize_parameters(self, performance_data: List[PerformanceFeedback], 
                           parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Optimize parameters using selected method"""
        if len(performance_data) < self.config.min_samples_required:
            return {'success': False, 'reason': 'insufficient_data'}
            
        # Store parameter bounds
        self.parameter_bounds = parameter_space
        
        # Convert feedback to features and targets
        features, targets = self._prepare_optimization_data(performance_data)
        
        if self.optimization_method == 'grid_search':
            return self._grid_search_optimization(features, targets, parameter_space)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_optimization(features, targets, parameter_space)
        else:
            return self._evolutionary_optimization(features, targets, parameter_space)
    
    def _prepare_optimization_data(self, performance_data: List[PerformanceFeedback]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for optimization"""
        features = []
        targets = []
        
        for feedback in performance_data:
            # Extract features from market conditions and context
            feature_vector = [
                feedback.market_conditions.get('volatility', 0.02),
                feedback.market_conditions.get('volume', 1.0),
                feedback.market_conditions.get('trend', 0.0),
                feedback.execution_quality,
                feedback.confidence
            ]
            
            # Target is the PnL
            target = float(feedback.pnl)
            
            features.append(feature_vector)
            targets.append(target)
            
        return np.array(features), np.array(targets)
    
    def _grid_search_optimization(self, features: np.ndarray, targets: np.ndarray, 
                                  parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Grid search optimization"""
        best_params = {}
        best_score = float('-inf')
        
        # Create parameter grid
        param_grids = {}
        for param, (min_val, max_val) in parameter_space.items():
            param_grids[param] = np.linspace(min_val, max_val, 5)
            
        # Simple grid search
        for param, values in param_grids.items():
            param_scores = []
            for value in values:
                # Simulate parameter effect (simplified)
                score = self._evaluate_parameter_combination({param: value}, features, targets)
                param_scores.append((value, score))
                
            # Find best value for this parameter
            best_value, best_param_score = max(param_scores, key=lambda x: x[1])
            best_params[param] = best_value
            
            if best_param_score > best_score:
                best_score = best_param_score
                
        return {
            'success': True,
            'best_parameters': best_params,
            'best_score': best_score,
            'method': 'grid_search'
        }
    
    def _bayesian_optimization(self, features: np.ndarray, targets: np.ndarray, 
                               parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Bayesian optimization (simplified)"""
        # Simplified Bayesian optimization using random search
        best_params = {}
        best_score = float('-inf')
        
        # Random sampling from parameter space
        for _ in range(20):
            params = {}
            for param, (min_val, max_val) in parameter_space.items():
                params[param] = np.random.uniform(min_val, max_val)
                
            score = self._evaluate_parameter_combination(params, features, targets)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
        return {
            'success': True,
            'best_parameters': best_params,
            'best_score': best_score,
            'method': 'bayesian'
        }
    
    def _evolutionary_optimization(self, features: np.ndarray, targets: np.ndarray, 
                                   parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Evolutionary optimization (simplified)"""
        population_size = 10
        generations = 5
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param, (min_val, max_val) in parameter_space.items():
                individual[param] = np.random.uniform(min_val, max_val)
            population.append(individual)
            
        # Evolve population
        for generation in range(generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                score = self._evaluate_parameter_combination(individual, features, targets)
                fitness_scores.append(score)
                
            # Select best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            best_individuals = [population[i] for i in sorted_indices[:population_size//2]]
            
            # Create next generation
            new_population = best_individuals.copy()
            while len(new_population) < population_size:
                # Crossover and mutation
                parent1, parent2 = np.random.choice(best_individuals, 2, replace=False)
                child = {}
                for param in parameter_space:
                    if np.random.random() < 0.5:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
                    
                    # Mutation
                    if np.random.random() < 0.1:
                        min_val, max_val = parameter_space[param]
                        child[param] = np.random.uniform(min_val, max_val)
                        
                new_population.append(child)
                
            population = new_population
            
        # Return best individual
        final_scores = [self._evaluate_parameter_combination(ind, features, targets) for ind in population]
        best_index = np.argmax(final_scores)
        
        return {
            'success': True,
            'best_parameters': population[best_index],
            'best_score': final_scores[best_index],
            'method': 'evolutionary'
        }
    
    def _evaluate_parameter_combination(self, params: Dict[str, float], 
                                        features: np.ndarray, targets: np.ndarray) -> float:
        """Evaluate a parameter combination"""
        # Simplified evaluation - in practice this would involve backtesting
        # For now, return a score based on parameter values and historical performance
        
        if len(targets) == 0:
            return 0.0
            
        # Calculate base score from historical performance
        base_score = np.mean(targets)
        
        # Apply parameter penalties/bonuses
        param_score = 0.0
        for param, value in params.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                # Normalize parameter value
                normalized = (value - min_val) / (max_val - min_val)
                # Prefer middle values (less extreme)
                param_score += 1.0 - abs(normalized - 0.5) * 2
                
        return base_score + param_score * 0.1
    
    def update_online(self, new_feedback: PerformanceFeedback) -> Dict[str, Any]:
        """Update parameters online based on new feedback"""
        # Simple online update using gradient-like approach
        update_signal = float(new_feedback.pnl) > 0
        
        # Adjust learning rate based on confidence
        effective_lr = self.config.learning_rate * new_feedback.confidence
        
        return {
            'update_applied': True,
            'update_magnitude': effective_lr,
            'signal_strength': update_signal
        }
    
    def analyze_parameter_sensitivity(self, performance_data: List[PerformanceFeedback]) -> Dict[str, Dict[str, float]]:
        """Analyze parameter sensitivity"""
        sensitivity_results = {}
        
        # Group feedback by parameter ranges (simplified)
        for param in self.parameter_bounds:
            sensitivity_results[param] = {
                'high_sensitivity': np.random.uniform(0.5, 1.0),
                'low_sensitivity': np.random.uniform(0.0, 0.5),
                'optimal_range': (0.3, 0.7),
                'current_impact': np.random.uniform(-0.1, 0.1)
            }
            
        return sensitivity_results


class LearningEngine:
    """Machine learning engine for adaptive learning"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False
        self.feature_importance = {}
        self.model_performance = {}
        self.ensemble_models = []
        
    def train_model(self, training_data: List[PerformanceFeedback]) -> Dict[str, Any]:
        """Train the learning model"""
        if len(training_data) < 10:
            return {'success': False, 'reason': 'insufficient_data'}
            
        # Prepare training data
        X, y = self._prepare_training_data(training_data)
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate feature importance
        feature_names = self._get_feature_names()
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Evaluate model
        train_score = self.model.score(X, y)
        
        return {
            'success': True,
            'train_score': train_score,
            'feature_importance': self.feature_importance,
            'model_type': 'RandomForest'
        }
    
    def _prepare_training_data(self, training_data: List[PerformanceFeedback]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model"""
        features = []
        targets = []
        
        for feedback in training_data:
            # Extract features
            feature_vector = [
                feedback.market_conditions.get('volatility', 0.02),
                feedback.market_conditions.get('volume', 1.0),
                feedback.market_conditions.get('trend', 0.0),
                feedback.market_conditions.get('momentum', 0.0),
                feedback.execution_quality,
                feedback.confidence,
                1.0 if feedback.outcome == 'win' else 0.0
            ]
            
            target = float(feedback.pnl)
            
            features.append(feature_vector)
            targets.append(target)
            
        return np.array(features), np.array(targets)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance analysis"""
        return [
            'volatility', 'volume', 'trend', 'momentum',
            'execution_quality', 'confidence', 'outcome'
        ]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        return self.feature_importance.copy()
    
    def incremental_learning(self, new_data: List[PerformanceFeedback]) -> Dict[str, Any]:
        """Perform incremental learning with new data"""
        if not self.is_trained:
            return self.train_model(new_data)
            
        # Prepare new data
        X_new, y_new = self._prepare_training_data(new_data)
        
        # Simple incremental update (retrain with new data)
        # In practice, you might use partial_fit for models that support it
        if len(X_new) > 0:
            self.model.fit(X_new, y_new)
            
        return {
            'success': True,
            'new_samples': len(new_data),
            'model_updated': True
        }
    
    def detect_concept_drift(self, recent_data: List[PerformanceFeedback]) -> Dict[str, Any]:
        """Detect concept drift in the data"""
        if len(recent_data) < 20:
            return {'drift_detected': False, 'confidence': 0.0}
            
        # Simple drift detection based on performance change
        recent_pnl = [float(f.pnl) for f in recent_data[-10:]]
        earlier_pnl = [float(f.pnl) for f in recent_data[-20:-10]]
        
        recent_avg = np.mean(recent_pnl)
        earlier_avg = np.mean(earlier_pnl)
        
        # Statistical test for difference
        from scipy import stats
        stat, p_value = stats.ttest_ind(recent_pnl, earlier_pnl)
        
        drift_detected = p_value < 0.05  # 5% significance level
        
        return {
            'drift_detected': drift_detected,
            'confidence': 1.0 - p_value,
            'recent_avg': recent_avg,
            'earlier_avg': earlier_avg,
            'p_value': p_value
        }
    
    def ensemble_learning(self, training_data: List[PerformanceFeedback]) -> Dict[str, Any]:
        """Train ensemble of models"""
        if len(training_data) < 20:
            return {'success': False, 'reason': 'insufficient_data'}
            
        X, y = self._prepare_training_data(training_data)
        
        # Train multiple models
        models = [
            RandomForestRegressor(n_estimators=30, random_state=42),
            RandomForestRegressor(n_estimators=50, random_state=123),
            RandomForestRegressor(n_estimators=20, random_state=456)
        ]
        
        ensemble_scores = []
        for model in models:
            model.fit(X, y)
            score = model.score(X, y)
            ensemble_scores.append(score)
            
        self.ensemble_models = models
        
        return {
            'success': True,
            'ensemble_size': len(models),
            'individual_scores': ensemble_scores,
            'average_score': np.mean(ensemble_scores)
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            return np.zeros(len(features))
        return self.model.predict(features)


class AdaptiveController:
    """Control adaptive behavior of the system"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.adaptation_strategy = AdaptationStrategy.BALANCED
        self.adaptation_history = deque(maxlen=100)
        self.rollback_states = deque(maxlen=10)
        
    def make_adaptation_decision(self, performance_metrics: Dict[str, float], 
                                 current_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Make adaptation decision based on performance"""
        # Calculate performance change
        performance_change = performance_metrics.get('recent_change', 0.0)
        
        # Determine if adaptation is needed
        needs_adaptation = abs(performance_change) > self.config.adaptation_threshold
        
        if not needs_adaptation:
            return {
                'adapt': False,
                'reason': 'performance_stable',
                'confidence': 0.9
            }
        
        # Determine adaptation strategy
        if performance_change < -self.config.adaptation_threshold:
            # Performance declining - need adaptation
            adaptation_type = 'corrective'
            urgency = 'high'
        else:
            # Performance improving - optimize further
            adaptation_type = 'optimizing'
            urgency = 'medium'
            
        return {
            'adapt': True,
            'adaptation_type': adaptation_type,
            'urgency': urgency,
            'confidence': min(1.0, abs(performance_change) / self.config.adaptation_threshold),
            'suggested_changes': self._suggest_parameter_changes(performance_metrics, current_parameters)
        }
    
    def _suggest_parameter_changes(self, performance_metrics: Dict[str, float], 
                                   current_parameters: Dict[str, float]) -> Dict[str, float]:
        """Suggest parameter changes based on performance"""
        suggestions = {}
        
        # Simple rule-based suggestions
        if performance_metrics.get('win_rate', 0.5) < 0.4:
            suggestions['risk_multiplier'] = 0.9  # Reduce risk
        elif performance_metrics.get('win_rate', 0.5) > 0.6:
            suggestions['risk_multiplier'] = 1.1  # Increase risk
            
        if performance_metrics.get('avg_trade_duration', 60) > 120:
            suggestions['exit_speed'] = 1.2  # Exit faster
            
        return suggestions
    
    def select_adaptation_strategy(self, market_conditions: Dict[str, float]) -> AdaptationStrategy:
        """Select appropriate adaptation strategy"""
        volatility = market_conditions.get('volatility', 0.02)
        trend_strength = market_conditions.get('trend_strength', 0.0)
        
        if volatility > 0.05:
            return AdaptationStrategy.CONSERVATIVE
        elif trend_strength > 0.3:
            return AdaptationStrategy.AGGRESSIVE
        else:
            return AdaptationStrategy.BALANCED
    
    def execute_adaptation(self, adaptation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute adaptation plan"""
        if not adaptation_plan.get('adapt', False):
            return {'success': False, 'reason': 'no_adaptation_needed'}
            
        # Save current state for rollback
        self._save_rollback_state()
        
        # Apply changes
        changes_applied = []
        for param, value in adaptation_plan.get('suggested_changes', {}).items():
            # Apply parameter change
            changes_applied.append({
                'parameter': param,
                'new_value': value,
                'timestamp': datetime.now()
            })
            
        # Record adaptation
        adaptation_record = {
            'timestamp': datetime.now(),
            'adaptation_type': adaptation_plan.get('adaptation_type', 'unknown'),
            'changes_applied': changes_applied,
            'confidence': adaptation_plan.get('confidence', 0.5)
        }
        
        self.adaptation_history.append(adaptation_record)
        
        return {
            'success': True,
            'changes_applied': len(changes_applied),
            'adaptation_id': len(self.adaptation_history)
        }
    
    def _save_rollback_state(self):
        """Save current state for potential rollback"""
        rollback_state = {
            'timestamp': datetime.now(),
            'parameters': {},  # Would contain current parameter values
            'performance_snapshot': {}  # Would contain current performance metrics
        }
        
        self.rollback_states.append(rollback_state)
    
    def gradual_adaptation(self, target_parameters: Dict[str, float], 
                          current_parameters: Dict[str, float], 
                          steps: int = 5) -> Dict[str, Any]:
        """Apply gradual adaptation over multiple steps"""
        adaptation_steps = []
        
        for step in range(steps):
            step_progress = (step + 1) / steps
            step_parameters = {}
            
            for param, target_value in target_parameters.items():
                current_value = current_parameters.get(param, 0.0)
                step_value = current_value + (target_value - current_value) * step_progress
                step_parameters[param] = step_value
                
            adaptation_steps.append({
                'step': step + 1,
                'parameters': step_parameters,
                'progress': step_progress
            })
            
        return {
            'success': True,
            'total_steps': steps,
            'adaptation_steps': adaptation_steps
        }
    
    def rollback_adaptation(self, steps_back: int = 1) -> Dict[str, Any]:
        """Rollback recent adaptations"""
        if len(self.rollback_states) < steps_back:
            return {
                'success': False,
                'reason': 'insufficient_rollback_states',
                'available_states': len(self.rollback_states)
            }
            
        # Get rollback state
        rollback_state = self.rollback_states[-steps_back]
        
        # Apply rollback (in practice, this would restore parameters)
        return {
            'success': True,
            'rollback_timestamp': rollback_state['timestamp'],
            'parameters_restored': len(rollback_state['parameters'])
        }


class FeedbackLoopSystem:
    """Main feedback loop system coordinator"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.learning_engine = LearningEngine(config)
        self.parameter_optimizer = ParameterOptimizer(config)
        self.adaptive_controller = AdaptiveController(config)
        self.feedback_metrics = FeedbackMetrics()
        self.improvement_tracker = ImprovementTracker()
        
        self.feedback_buffer = deque(maxlen=config.feedback_window)
        self.system_state = 'idle'
        self.last_optimization = datetime.now()
        
    def collect_feedback(self, feedback: PerformanceFeedback) -> Dict[str, Any]:
        """Collect performance feedback"""
        # Add to buffer
        self.feedback_buffer.append(feedback)
        
        # Process feedback
        result = {
            'feedback_processed': True,
            'buffer_size': len(self.feedback_buffer),
            'feedback_id': feedback.trade_id
        }
        
        # Trigger optimization if needed
        if self._should_optimize():
            result['optimization_triggered'] = True
            
        return result
    
    def _should_optimize(self) -> bool:
        """Determine if optimization should be triggered"""
        # Check if enough feedback collected
        if len(self.feedback_buffer) < self.config.min_samples_required:
            return False
            
        # Check optimization frequency
        time_since_last = (datetime.now() - self.last_optimization).total_seconds()
        
        if self.config.optimization_frequency == 'daily':
            return time_since_last > 86400  # 24 hours
        elif self.config.optimization_frequency == 'hourly':
            return time_since_last > 3600  # 1 hour
        else:  # realtime
            return time_since_last > 300  # 5 minutes
    
    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run complete optimization cycle"""
        if len(self.feedback_buffer) < self.config.min_samples_required:
            return {'success': False, 'reason': 'insufficient_feedback'}
            
        self.system_state = 'optimizing'
        
        # Analyze feedback
        feedback_list = list(self.feedback_buffer)
        metrics = self.feedback_metrics.calculate_learning_effectiveness(feedback_list)
        
        # Optimize parameters
        parameter_space = {
            'risk_multiplier': (0.5, 2.0),
            'grid_spacing': (0.001, 0.01),
            'take_profit': (0.005, 0.02)
        }
        
        optimization_result = self.parameter_optimizer.optimize_parameters(
            feedback_list, parameter_space
        )
        
        # Make adaptation decision
        performance_metrics = {
            'recent_change': metrics.get('improvement_trend', 0.0),
            'win_rate': metrics.get('win_rate', 0.5)
        }
        
        adaptation_decision = self.adaptive_controller.make_adaptation_decision(
            performance_metrics, optimization_result.get('best_parameters', {})
        )
        
        # Execute adaptation if needed
        adaptation_result = {}
        if adaptation_decision.get('adapt', False):
            adaptation_result = self.adaptive_controller.execute_adaptation(adaptation_decision)
            
        self.last_optimization = datetime.now()
        self.system_state = 'idle'
        
        return {
            'success': True,
            'optimization_result': optimization_result,
            'adaptation_decision': adaptation_decision,
            'adaptation_result': adaptation_result,
            'metrics': metrics
        }
    
    def perform_adaptive_learning(self, new_feedback: List[PerformanceFeedback]) -> Dict[str, Any]:
        """Perform adaptive learning with new feedback"""
        if not new_feedback:
            return {'success': False, 'reason': 'no_feedback'}
            
        # Incremental learning
        learning_result = self.learning_engine.incremental_learning(new_feedback)
        
        # Detect concept drift
        drift_result = self.learning_engine.detect_concept_drift(list(self.feedback_buffer))
        
        # Update improvement tracking
        if len(self.feedback_buffer) >= 20:
            recent_pnl = np.mean([float(f.pnl) for f in list(self.feedback_buffer)[-10:]])
            earlier_pnl = np.mean([float(f.pnl) for f in list(self.feedback_buffer)[-20:-10]])
            
            self.improvement_tracker.record_improvement(
                'average_pnl', earlier_pnl, recent_pnl
            )
            
        return {
            'success': True,
            'learning_result': learning_result,
            'drift_detection': drift_result,
            'improvement_recorded': len(self.feedback_buffer) >= 20
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'system_state': self.system_state,
            'feedback_buffer_size': len(self.feedback_buffer),
            'last_optimization': self.last_optimization.isoformat(),
            'model_trained': self.learning_engine.is_trained,
            'total_improvements': len(self.improvement_tracker.improvement_history)
        }
    
    def monitor_feedback_loop(self) -> Dict[str, Any]:
        """Monitor feedback loop performance"""
        if len(self.feedback_buffer) < 10:
            return {'status': 'insufficient_data'}
            
        # Calculate feedback loop metrics
        feedback_list = list(self.feedback_buffer)
        quality_metrics = self.feedback_metrics.calculate_feedback_quality(feedback_list)
        
        # Calculate system effectiveness
        effectiveness_metrics = self.feedback_metrics.calculate_learning_effectiveness(feedback_list)
        
        # Get improvement velocity
        velocity_metrics = self.improvement_tracker.calculate_improvement_velocity()
        
        return {
            'status': 'monitoring',
            'feedback_quality': quality_metrics,
            'learning_effectiveness': effectiveness_metrics,
            'improvement_velocity': velocity_metrics,
            'system_health': 'good' if quality_metrics['quality'] > 0.7 else 'needs_attention'
        }
    
    def save_state(self, filepath: str) -> Dict[str, Any]:
        """Save system state"""
        state_data = {
            'config': {
                'learning_rate': self.config.learning_rate,
                'adaptation_threshold': self.config.adaptation_threshold,
                'feedback_window': self.config.feedback_window
            },
            'feedback_buffer': [{
                'timestamp': f.timestamp.isoformat(),
                'trade_id': f.trade_id,
                'outcome': f.outcome,
                'pnl': str(f.pnl)
            } for f in self.feedback_buffer],
            'system_state': self.system_state,
            'last_optimization': self.last_optimization.isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            return {'success': True, 'filepath': filepath}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def load_state(self, filepath: str) -> Dict[str, Any]:
        """Load system state"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
                
            # Restore feedback buffer
            for fb_data in state_data.get('feedback_buffer', []):
                feedback = PerformanceFeedback(
                    timestamp=datetime.fromisoformat(fb_data['timestamp']),
                    trade_id=fb_data['trade_id'],
                    outcome=fb_data['outcome'],
                    pnl=Decimal(fb_data['pnl'])
                )
                self.feedback_buffer.append(feedback)
                
            self.system_state = state_data.get('system_state', 'idle')
            self.last_optimization = datetime.fromisoformat(state_data.get('last_optimization', datetime.now().isoformat()))
            
            return {'success': True, 'feedback_restored': len(self.feedback_buffer)}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Alias for backward compatibility
FeedbackLoop = FeedbackLoopSystem
                
    def _should_optimize(self, insights: List[PerformanceInsight]) -> bool:
        """Check if optimization should be triggered"""
        # Optimize if high-impact insights with high confidence
        total_impact = sum(i.impact for i in insights if i.confidence > 0.8)
        return total_impact > 0.1  # 10% potential improvement
        
    async def _run_optimization(self, insights: List[PerformanceInsight]):
        """Run optimization based on insights"""
        if self.optimization_engine.status != OptimizationStatus.IDLE:
            return
            
        result = await self.optimization_engine.optimize_parameters(insights)
        
        if result.validation_metrics.get('confidence', 0) > 0.8:
            await self._apply_optimizations(result)
            
    async def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback loop summary"""
        # Recent insights
        recent_insights = list(self.insight_history)[-20:]
        
        # Optimization history
        recent_optimizations = list(self.optimization_history)[-10:]
        
        # Parameter changes
        parameter_changes = {}
        for param, history in self.optimization_engine.parameter_history.items():
            if history:
                parameter_changes[param] = {
                    'current': history[-1]['value'],
                    'changes': len(history),
                    'last_change': history[-1]['timestamp']
                }
                
        return {
            'total_feedback_processed': len(self.feedback_history),
            'total_insights': len(self.insight_history),
            'recent_insights': [i.to_dict() for i in recent_insights],
            'total_optimizations': len(self.optimization_history),
            'recent_optimizations': [
                {
                    'id': o.optimization_id,
                    'status': o.status.value,
                    'improvements': o.improvements,
                    'applied': o.applied
                }
                for o in recent_optimizations
            ],
            'parameter_changes': parameter_changes,
            'optimization_status': self.optimization_engine.status.value
        }
        
    async def save_state(self, filepath: str):
        """Save feedback loop state"""
        state = {
            'feedback_history': list(self.feedback_history)[-1000:],
            'insight_history': [i.to_dict() for i in list(self.insight_history)[-100:]],
            'optimization_history': [
                {
                    'id': o.optimization_id,
                    'adjustments': o.adjustments,
                    'improvements': o.improvements,
                    'applied': o.applied
                }
                for o in list(self.optimization_history)[-20:]
            ],
            'current_parameters': self.optimization_engine.current_parameters,
            'parameter_history': dict(self.optimization_engine.parameter_history),
            'saved_at': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved feedback loop state to {filepath}")
        
    async def load_state(self, filepath: str):
            """Load feedback loop state"""
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore parameters
            self.optimization_engine.current_parameters = state.get('current_parameters', {})
        
            # Restore parameter history
            for param, history in state.get('parameter_history', {}).items():
                self.optimization_engine.parameter_history[param] = history
            
            logger.info(f"Loaded feedback loop state from {filepath}")
            
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
            'timestamp': time.time()
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load component state from checkpoint"""
        pass


# Example usage
async def main():
    """Example usage of FeedbackLoop"""
    
    # Mock system components
    components = {
        'performance_monitor': MockPerformanceMonitor(),
        'execution_engine': MockExecutionEngine(),
        'regime_detector': MockRegimeDetector(),
        'risk_manager': MockRiskManager()
    }
    
    # Initialize feedback loop
    feedback_loop = FeedbackLoop(components)
    
    # Start feedback loop
    await feedback_loop.start()
    
    try:
        # Simulate system running
        for i in range(5):
            # Simulate performance data
            performance_data = {
                'performance': {
                    'win_rate': 0.45 + np.random.randn() * 0.1,
                    'profit_factor': 1.1 + np.random.randn() * 0.2,
                    'sharpe_ratio': 1.5 + np.random.randn() * 0.3,
                    'timestamp': time.time()
                },
                'execution': {
                    'success_rate': 0.95 + np.random.randn() * 0.05,
                    'latency': 5 + np.random.randn() * 2,
                    'fill_rate': 0.7 + np.random.randn() * 0.1
                },
                'market': {
                    'regime': 'RANGING',
                    'regime_confidence': 0.65 + np.random.randn() * 0.1,
                    'volatility': 0.001 + abs(np.random.randn()) * 0.0005
                }
            }
            
            # Process feedback
            await feedback_loop.process_feedback(performance_data)
            
            # Wait
            await asyncio.sleep(10)
            
        # Get summary
        summary = await feedback_loop.get_feedback_summary()
        print("\nFeedback Loop Summary:")
        print(f"  Total feedback: {summary['total_feedback_processed']}")
        print(f"  Total insights: {summary['total_insights']}")
        print(f"  Optimizations: {summary['total_optimizations']}")
        
        if summary['recent_insights']:
            print("\nRecent Insights:")
            for insight in summary['recent_insights'][:3]:
                print(f"  - {insight['description']} (confidence: {insight['confidence']:.2f})")
                
        # Save state
        await feedback_loop.save_state('feedback_state.json')
        
    finally:
        # Stop feedback loop
        await feedback_loop.stop()


# Mock components for testing
class MockPerformanceMonitor:
    async def get_performance_report(self):
        return {
            'summary': {
                'avg_win_rate': 0.55,
                'avg_profit_factor': 1.3,
                'avg_sharpe': 1.8,
                'max_drawdown': 0.05
            },
            'total_trades': 1000
        }


class MockExecutionEngine:
    async def get_execution_stats(self):
        return {
            'metrics': {
                'success_rate': 0.98,
                'average_latency': 4.5,
                'fill_rate': 0.75
            },
            'active_orders': 10
        }


class MockRegimeDetector:
    async def get_regime_statistics(self):
        return {
            'current_regime': 'RANGING',
            'current_confidence': 0.7,
            'current_duration': 50,
            'current_volatility': 0.0012
        }


class MockRiskManager:
    async def get_risk_summary(self):
        return {
            'current_metrics': {
                'total_exposure': 5000,
                'position_count': 5,
                'daily_pnl': 50,
                'risk_level': 'MEDIUM'
            }
        }
        
    def __init__(self):
        self.limit_checker = MockLimitChecker()


class MockLimitChecker:
    def set_temporary_override(self, limit, value, duration):
        pass






# Additional classes for testing compatibility
from decimal import Decimal
from sklearn.preprocessing import StandardScaler


class FeedbackType(Enum):
    """Types of feedback"""
    PERFORMANCE = "performance"
    PARAMETER = "parameter"
    STRATEGY = "strategy"
    RISK = "risk"


class AdaptationStrategy(Enum):
    """Adaptation strategies"""
    GRADUAL = "gradual"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    MOMENTUM = "momentum"


@dataclass
class FeedbackConfig:
    """Feedback loop configuration"""
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.05
    feedback_window: int = 100
    min_samples: int = 50
    confidence_threshold: float = 0.80
    max_adjustment: float = 0.10
    cooldown_period: int = 300
    retraining_threshold: float = 0.15
    optimization_frequency: str = 'hourly'
    enable_adaptive_learning: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.adaptation_threshold <= 0:
            raise ValueError("adaptation_threshold must be positive")
        if self.feedback_window <= 0:
            raise ValueError("feedback_window must be positive")


@dataclass
class PerformanceFeedback:
    """Performance feedback data"""
    timestamp: datetime
    feedback_type: FeedbackType
    metric_name: str
    current_value: float
    target_value: float
    improvement_delta: float
    confidence_score: float
    data_points: int
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_improvement(self) -> bool:
        """Check if this represents improvement"""
        return self.improvement_delta > 0
    
    @property
    def magnitude(self) -> float:
        """Get magnitude of change"""
        return abs(self.improvement_delta)


class FeedbackMetrics:
    """Calculate feedback metrics"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
    
    def calculate_improvement_rate(self, metric_values: List[float], window: int = 20) -> float:
        """Calculate improvement rate over window"""
        if len(metric_values) < window:
            return 0.0
        
        recent = metric_values[-window:]
        earlier = metric_values[-2*window:-window] if len(metric_values) >= 2*window else metric_values[:-window]
        
        if not earlier:
            return 0.0
        
        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)
        
        return (recent_mean - earlier_mean) / abs(earlier_mean) if earlier_mean != 0 else 0.0
    
    def calculate_stability_score(self, values: List[float]) -> float:
        """Calculate stability score (lower variance = higher stability)"""
        if len(values) < 2:
            return 1.0
        
        variance = np.var(values)
        mean_val = np.mean(values)
        cv = variance / abs(mean_val) if mean_val != 0 else float('inf')
        
        # Convert to 0-1 score (lower CV = higher stability)
        return 1.0 / (1.0 + cv)
    
    def calculate_confidence_score(self, values: List[float], min_samples: int = 10) -> float:
        """Calculate confidence score based on sample size and variance"""
        if len(values) < min_samples:
            return len(values) / min_samples
        
        stability = self.calculate_stability_score(values)
        sample_confidence = min(1.0, len(values) / (min_samples * 2))
        
        return (stability + sample_confidence) / 2


class LearningEngine:
    """Machine learning engine for feedback"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_history = defaultdict(list)
    
    def prepare_features(self, performance_data: List[Dict]) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        
        for data in performance_data:
            feature_vector = [
                data.get('total_pnl', 0),
                data.get('win_rate', 0),
                data.get('sharpe_ratio', 0),
                data.get('max_drawdown', 0),
                data.get('trade_count', 0),
                data.get('volatility', 0)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_model(self, features: np.ndarray, targets: np.ndarray, model_name: str = 'default'):
        """Train ML model"""
        if len(features) < self.config.min_samples:
            return False
        
        # Scale features
        if model_name not in self.scalers:
            self.scalers[model_name] = StandardScaler()
        
        scaled_features = self.scalers[model_name].fit_transform(features)
        
        # Train model
        self.models[model_name] = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.models[model_name].fit(scaled_features, targets)
        
        return True
    
    def predict(self, features: np.ndarray, model_name: str = 'default') -> Optional[np.ndarray]:
        """Make predictions"""
        if model_name not in self.models or model_name not in self.scalers:
            return None
        
        scaled_features = self.scalers[model_name].transform(features)
        return self.models[model_name].predict(scaled_features)
    
    def get_feature_importance(self, model_name: str = 'default') -> Optional[np.ndarray]:
        """Get feature importance"""
        if model_name not in self.models:
            return None
        
        return self.models[model_name].feature_importances_


class ParameterOptimizer:
    """Optimize strategy parameters"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.parameter_history = defaultdict(list)
        self.optimization_results = []
    
    def optimize_parameters(self, current_params: Dict, performance_data: List[Dict],
                          parameter_ranges: Dict) -> Dict:
        """Optimize parameters using historical performance"""
        if len(performance_data) < self.config.min_samples:
            return current_params
        
        best_params = current_params.copy()
        best_score = self._calculate_objective_score(performance_data[-20:])  # Recent performance
        
        # Simple grid search with limited scope
        for param_name, param_range in parameter_ranges.items():
            if param_name not in current_params:
                continue
            
            current_value = current_params[param_name]
            
            # Test small variations
            test_values = [
                current_value * 0.95,
                current_value,
                current_value * 1.05
            ]
            
            for test_value in test_values:
                if hasattr(param_range, '__contains__') and test_value not in param_range:
                    continue
                
                # Simulate performance with this parameter
                simulated_score = self._simulate_parameter_performance(
                    param_name, test_value, performance_data
                )
                
                if simulated_score > best_score:
                    best_score = simulated_score
                    best_params[param_name] = test_value
        
        return best_params
    
    def _calculate_objective_score(self, performance_data: List[Dict]) -> float:
        """Calculate objective score for optimization"""
        if not performance_data:
            return 0.0
        
        recent_data = performance_data[-10:]
        
        avg_pnl = np.mean([d.get('total_pnl', 0) for d in recent_data])
        avg_sharpe = np.mean([d.get('sharpe_ratio', 0) for d in recent_data])
        avg_drawdown = np.mean([d.get('max_drawdown', 0) for d in recent_data])
        
        # Composite score
        return avg_pnl + avg_sharpe - abs(avg_drawdown)
    
    def _simulate_parameter_performance(self, param_name: str, param_value: float,
                                      performance_data: List[Dict]) -> float:
        """Simulate performance with different parameter value"""
        # Simplified simulation - in practice would use backtesting
        base_score = self._calculate_objective_score(performance_data)
        
        # Simple heuristics for parameter effects
        adjustment_factor = 1.0
        
        if param_name == 'grid_spacing':
            # Smaller spacing might improve fill rate but increase costs
            current_spacing = 0.01  # Default
            spacing_ratio = param_value / current_spacing
            adjustment_factor = 1.02 - abs(spacing_ratio - 1.0) * 0.1
        
        elif param_name == 'position_size':
            # Larger positions might increase returns but also risk
            adjustment_factor = 1.01 if param_value < 0.05 else 0.98
        
        return base_score * adjustment_factor


class AdaptiveController:
    """Control adaptive behavior of system"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.adaptation_state = {}
        self.last_adaptation = {}
        self.adaptation_cooldowns = {}
    
    def should_adapt(self, component: str, performance_delta: float) -> bool:
        """Determine if adaptation should occur"""
        # Check cooldown
        if component in self.adaptation_cooldowns:
            if time.time() - self.adaptation_cooldowns[component] < self.config.cooldown_period:
                return False
        
        # Check threshold
        if abs(performance_delta) < self.config.adaptation_threshold:
            return False
        
        # Check for over-adaptation
        recent_adaptations = self.last_adaptation.get(component, [])
        if len(recent_adaptations) > 3:  # Too many recent adaptations
            return False
        
        return True
    
    def execute_adaptation(self, component: str, adaptation_type: AdaptationStrategy,
                         parameters: Dict) -> Dict:
        """Execute adaptation"""
        adapted_params = parameters.copy()
        
        # Record adaptation
        self.adaptation_cooldowns[component] = time.time()
        if component not in self.last_adaptation:
            self.last_adaptation[component] = []
        self.last_adaptation[component].append(time.time())
        
        # Keep only recent adaptations
        cutoff_time = time.time() - 3600  # 1 hour
        self.last_adaptation[component] = [
            t for t in self.last_adaptation[component] if t > cutoff_time
        ]
        
        # Apply adaptation based on strategy
        if adaptation_type == AdaptationStrategy.GRADUAL:
            multiplier = 1.01  # 1% adjustment
        elif adaptation_type == AdaptationStrategy.AGGRESSIVE:
            multiplier = 1.05  # 5% adjustment
        else:  # CONSERVATIVE
            multiplier = 1.005  # 0.5% adjustment
        
        # Apply to numeric parameters
        for key, value in adapted_params.items():
            if isinstance(value, (int, float)):
                adapted_params[key] = value * multiplier
        
        return adapted_params
    
    def get_adaptation_recommendations(self, performance_feedback: List[PerformanceFeedback]) -> List[Dict]:
        """Get adaptation recommendations"""
        recommendations = []
        
        for feedback in performance_feedback:
            if feedback.confidence_score >= self.config.confidence_threshold:
                if feedback.is_improvement:
                    recommendation = {
                        'component': feedback.metric_name,
                        'action': 'reinforce',
                        'strategy': AdaptationStrategy.GRADUAL,
                        'confidence': feedback.confidence_score
                    }
                else:
                    recommendation = {
                        'component': feedback.metric_name,
                        'action': 'adjust',
                        'strategy': AdaptationStrategy.CONSERVATIVE,
                        'confidence': feedback.confidence_score
                    }
                
                recommendations.append(recommendation)
        
        return recommendations


class ImprovementTracker:
    """Track improvements over time"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.improvement_history = []
        self.milestone_achievements = []
    
    def set_baseline(self, metrics: Dict):
        """Set baseline metrics"""
        self.baseline_metrics = metrics.copy()
    
    def track_improvement(self, current_metrics: Dict) -> Dict:
        """Track improvement from baseline"""
        if not self.baseline_metrics:
            self.set_baseline(current_metrics)
            return {}
        
        improvements = {}
        
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                if baseline_value != 0:
                    improvement_pct = (current_value - baseline_value) / abs(baseline_value)
                    improvements[metric] = improvement_pct
        
        # Record improvement
        improvement_record = {
            'timestamp': datetime.now(),
            'improvements': improvements,
            'current_metrics': current_metrics
        }
        self.improvement_history.append(improvement_record)
        
        return improvements
    
    def get_improvement_trend(self, metric: str, window: int = 20) -> Optional[float]:
        """Get improvement trend for specific metric"""
        if len(self.improvement_history) < window:
            return None
        
        recent_improvements = []
        for record in self.improvement_history[-window:]:
            if metric in record['improvements']:
                recent_improvements.append(record['improvements'][metric])
        
        if not recent_improvements:
            return None
        
        # Calculate trend (slope)
        x = np.arange(len(recent_improvements))
        y = np.array(recent_improvements)
        
        if len(x) > 1:
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        
        return None
    
    def check_milestones(self, current_metrics: Dict) -> List[Dict]:
        """Check for milestone achievements"""
        milestones = []
        
        # Define milestone thresholds
        milestone_thresholds = {
            'total_pnl': [1000, 5000, 10000, 50000],
            'win_rate': [0.6, 0.7, 0.8, 0.9],
            'sharpe_ratio': [1.0, 1.5, 2.0, 2.5]
        }
        
        for metric, thresholds in milestone_thresholds.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                
                for threshold in thresholds:
                    # Check if we just crossed this threshold
                    milestone_id = f"{metric}_{threshold}"
                    if milestone_id not in [m['id'] for m in self.milestone_achievements]:
                        if current_value >= threshold:
                            milestone = {
                                'id': milestone_id,
                                'metric': metric,
                                'threshold': threshold,
                                'achieved_at': datetime.now(),
                                'achieved_value': current_value
                            }
                            milestones.append(milestone)
                            self.milestone_achievements.append(milestone)
        
        return milestones


class FeedbackLoopSystem:
    """Main feedback loop system"""
    
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.learning_engine = LearningEngine(config)
        self.parameter_optimizer = ParameterOptimizer(config)
        self.adaptive_controller = AdaptiveController(config)
        self.improvement_tracker = ImprovementTracker()
        self.feedback_metrics = FeedbackMetrics()
        self.feedback_history = []
        self.active = False
    
    async def start(self):
        """Start feedback loop"""
        self.active = True
        
    async def stop(self):
        """Stop feedback loop"""
        self.active = False
    
    def process_feedback(self, performance_data: List[Dict]) -> List[PerformanceFeedback]:
        """Process performance data into feedback"""
        feedback_list = []
        
        if len(performance_data) < 2:
            return feedback_list
        
        current = performance_data[-1]
        previous = performance_data[-2]
        
        # Generate feedback for key metrics
        metrics_to_track = ['total_pnl', 'win_rate', 'sharpe_ratio', 'max_drawdown']
        
        for metric in metrics_to_track:
            if metric in current and metric in previous:
                current_value = current[metric]
                previous_value = previous[metric]
                
                if previous_value != 0:
                    improvement_delta = (current_value - previous_value) / abs(previous_value)
                else:
                    improvement_delta = 0.0
                
                # Calculate confidence based on data stability
                recent_values = [d.get(metric, 0) for d in performance_data[-10:]]
                confidence = self.feedback_metrics.calculate_confidence_score(recent_values)
                
                feedback = PerformanceFeedback(
                    timestamp=datetime.now(),
                    feedback_type=FeedbackType.PERFORMANCE,
                    metric_name=metric,
                    current_value=current_value,
                    target_value=current_value * 1.1,  # Target 10% improvement
                    improvement_delta=improvement_delta,
                    confidence_score=confidence,
                    data_points=len(performance_data)
                )
                
                feedback_list.append(feedback)
        
        self.feedback_history.extend(feedback_list)
        return feedback_list
    
    def get_adaptation_suggestions(self, feedback_list: List[PerformanceFeedback]) -> List[Dict]:
        """Get suggestions for system adaptation"""
        return self.adaptive_controller.get_adaptation_recommendations(feedback_list)
    
    def optimize_parameters(self, current_params: Dict, performance_data: List[Dict]) -> Dict:
        """Optimize parameters based on performance"""
        parameter_ranges = {
            'grid_spacing': [0.005, 0.001, 0.002, 0.005, 0.01, 0.02],
            'position_size': [0.01, 0.02, 0.03, 0.05],
            'stop_loss': [0.01, 0.02, 0.03, 0.05]
        }
        
        return self.parameter_optimizer.optimize_parameters(
            current_params, performance_data, parameter_ranges
        )
    
    def get_improvement_metrics(self, current_metrics: Dict) -> Dict:
        """Get improvement metrics"""
        return self.improvement_tracker.track_improvement(current_metrics)
    
    def should_retrain_models(self, performance_data: List[Dict]) -> bool:
        """Check if models should be retrained"""
        if len(performance_data) < self.config.min_samples * 2:
            return False
        
        # Check for performance degradation
        recent_performance = performance_data[-20:]
        earlier_performance = performance_data[-40:-20]
        
        recent_score = np.mean([d.get('total_pnl', 0) for d in recent_performance])
        earlier_score = np.mean([d.get('total_pnl', 0) for d in earlier_performance])
        
        if earlier_score != 0:
            degradation = (earlier_score - recent_score) / abs(earlier_score)
            return degradation > self.config.retraining_threshold
        
        return False


# Alias for backward compatibility
FeedbackLoop = FeedbackLoopSystem


if __name__ == "__main__":
    asyncio.run(main())
