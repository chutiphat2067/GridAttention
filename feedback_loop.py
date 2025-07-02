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
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Local imports
from attention_learning_layer import AttentionLearningLayer, AttentionPhase
from market_regime_detector import MarketRegimeDetector, MarketRegime
from grid_strategy_selector import GridStrategySelector
from risk_management_system import RiskManagementSystem
from performance_monitor import PerformanceMonitor, TradingMetrics
from overfitting_detector import OverfittingDetector, OverfittingSeverity

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
        insights = []
        
        # Time-based patterns
        if self.performance_history:
            # Group by hour of day
            hourly_performance = defaultdict(list)
            
            for perf in self.performance_history:
                if 'timestamp' in perf and 'pnl' in perf:
                    hour = datetime.fromtimestamp(perf['timestamp']).hour
                    hourly_performance[hour].append(perf['pnl'])
                    
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
        for task in [self._feedback_task, self._optimization_task]:
            if task:
                task.cancel()
                
        # Wait for completion
        await asyncio.gather(
            self._feedback_task,
            self._optimization_task,
            return_exceptions=True
        )
        
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
            
    async def _validate_optimizations(self, result: OptimizationResult) -> bool:
        """Basic optimization validation (legacy method for compatibility)"""
        # Check if adjustments are too aggressive
        if 'parameters' in result.adjustments:
            for param, target in result.adjustments['parameters'].items():
                current = self.optimization_engine.current_parameters.get(param, 1.0)
                change_ratio = abs(target - current) / current if current != 0 else float('inf')
                
                if change_ratio > 0.1:  # ลดจาก 20% เป็น 10%
                    logger.warning(f"Optimization suggests {change_ratio:.1%} change for {param}, too aggressive")
                    return False
                    
        # Check recent adjustment frequency
        recent_param_adjustments = [
            adj for adj in self.recent_adjustments
            if time.time() - adj['timestamp'] < ADJUSTMENT_COOLDOWN
        ]
        
        if len(recent_param_adjustments) > 1:  # ลดจาก 2 เป็น 1
            logger.warning("Recent adjustments detected, skipping to prevent overfitting")
            return False
            
        return True
        
    async def _apply_strategy_change_conservative(self, strategy_change: Dict[str, str]):
        """Apply strategy change with validation"""
        if 'strategy_selector' in self.components:
            # Don't switch strategies too frequently
            last_switch = getattr(self, 'last_strategy_switch', 0)
            if time.time() - last_switch < 3600:  # 1 hour minimum between switches
                logger.info("Strategy switch blocked due to cooldown")
                return
                
            logger.info(f"Conservative strategy switch: {strategy_change['from']} to {strategy_change['to']}")
            self.last_strategy_switch = time.time()
            
    async def process_feedback(self, performance_data: Dict[str, Any]):
        """Process immediate feedback (called by components)"""
        # Extract insights
        insights = await self.feedback_processor.process_feedback(performance_data)
        
        # Update components if high-impact insights
        high_impact_insights = [i for i in insights if i.impact > IMPROVEMENT_THRESHOLD]
        
        if high_impact_insights:
            await self._distribute_insights(high_impact_insights)
            
            # Trigger optimization if needed
            if self._should_optimize(high_impact_insights):
                asyncio.create_task(self._run_optimization(high_impact_insights))
                
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






if __name__ == "__main__":
    asyncio.run(main())
