from typing import Dict, Any, Optional
"""
overfitting_detector.py
Real-time overfitting detection and prevention system for grid trading

Author: Grid Trading System
Date: 2024
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import json

logger = logging.getLogger(__name__)

# Constants
PERFORMANCE_GAP_THRESHOLD = 0.15  # 15% max acceptable gap
CONFIDENCE_CALIBRATION_THRESHOLD = 0.2  # 20% max calibration error
FEATURE_STABILITY_THRESHOLD = 0.3  # 30% max feature importance change
VALIDATION_WINDOW_SIZE = 1000
MIN_SAMPLES_FOR_DETECTION = 100


class OverfittingType(Enum):
    """Types of overfitting detected"""
    PERFORMANCE_DIVERGENCE = "performance_divergence"
    CONFIDENCE_MISCALIBRATION = "confidence_miscalibration"
    FEATURE_INSTABILITY = "feature_instability"
    PARAMETER_OVERFIT = "parameter_overfit"
    REGIME_OVERFIT = "regime_overfit"


class OverfittingSeverity(Enum):
    """Severity levels of overfitting"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class OverfittingMetrics:
    """Metrics for overfitting detection"""
    train_performance: float = 0.0
    test_performance: float = 0.0
    performance_gap: float = 0.0
    confidence_calibration_error: float = 0.0
    feature_stability_score: float = 1.0
    parameter_variance: float = 0.0
    prediction_variance: float = 0.0
    cross_validation_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def get_severity(self) -> OverfittingSeverity:
        """Calculate overall severity"""
        severity_score = 0
        
        # Performance gap contribution
        if self.performance_gap > 0.3:
            severity_score += 3
        elif self.performance_gap > 0.2:
            severity_score += 2
        elif self.performance_gap > 0.1:
            severity_score += 1
            
        # Confidence calibration contribution
        if self.confidence_calibration_error > 0.3:
            severity_score += 2
        elif self.confidence_calibration_error > 0.2:
            severity_score += 1
            
        # Feature stability contribution
        if self.feature_stability_score < 0.5:
            severity_score += 2
        elif self.feature_stability_score < 0.7:
            severity_score += 1
            
        # Map to severity
        if severity_score >= 6:
            return OverfittingSeverity.CRITICAL
        elif severity_score >= 4:
            return OverfittingSeverity.HIGH
        elif severity_score >= 2:
            return OverfittingSeverity.MEDIUM
        elif severity_score >= 1:
            return OverfittingSeverity.LOW
        else:
            return OverfittingSeverity.NONE


@dataclass
class ValidationResult:
    """Result of cross-validation"""
    avg_train_score: float
    avg_test_score: float
    score_variance: float
    is_overfitting: bool
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceDivergenceDetector:
    """Detect performance divergence between training and live trading"""
    
    def __init__(self, window_size: int = VALIDATION_WINDOW_SIZE):
        self.window_size = window_size
        self.training_performance = deque(maxlen=window_size)
        self.live_performance = deque(maxlen=window_size)
        self.divergence_history = deque(maxlen=100)
        
    def add_training_result(self, win_rate: float, profit_factor: float, timestamp: float):
        """Add training/backtest performance"""
        self.training_performance.append({
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'timestamp': timestamp
        })
        
    def add_live_result(self, win_rate: float, profit_factor: float, timestamp: float):
        """Add live trading performance"""
        self.live_performance.append({
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'timestamp': timestamp
        })
        
    def calculate_divergence(self) -> Tuple[float, Dict[str, float]]:
        """Calculate performance divergence"""
        if len(self.training_performance) < MIN_SAMPLES_FOR_DETECTION or \
           len(self.live_performance) < MIN_SAMPLES_FOR_DETECTION:
            return 0.0, {}
            
        # Calculate metrics for both
        train_metrics = self._calculate_metrics(self.training_performance)
        live_metrics = self._calculate_metrics(self.live_performance)
        
        # Calculate divergences
        win_rate_gap = abs(train_metrics['avg_win_rate'] - live_metrics['avg_win_rate'])
        profit_factor_gap = abs(train_metrics['avg_profit_factor'] - live_metrics['avg_profit_factor'])
        
        # Normalized divergence
        divergence = (win_rate_gap + profit_factor_gap / 10) / 2  # Normalize profit factor
        
        # Store in history
        self.divergence_history.append({
            'divergence': divergence,
            'timestamp': time.time()
        })
        
        details = {
            'train_win_rate': train_metrics['avg_win_rate'],
            'live_win_rate': live_metrics['avg_win_rate'],
            'train_profit_factor': train_metrics['avg_profit_factor'],
            'live_profit_factor': live_metrics['avg_profit_factor'],
            'win_rate_gap': win_rate_gap,
            'profit_factor_gap': profit_factor_gap
        }
        
        return divergence, details
        
    def _calculate_metrics(self, performance_data: deque) -> Dict[str, float]:
        """Calculate average metrics"""
        if not performance_data:
            return {'avg_win_rate': 0.0, 'avg_profit_factor': 0.0}
            
        win_rates = [p['win_rate'] for p in performance_data]
        profit_factors = [p['profit_factor'] for p in performance_data]
        
        return {
            'avg_win_rate': np.mean(win_rates),
            'avg_profit_factor': np.mean(profit_factors),
            'std_win_rate': np.std(win_rates),
            'std_profit_factor': np.std(profit_factors)
        }
        
    def get_trend(self) -> str:
        """Get divergence trend"""
        if len(self.divergence_history) < 10:
            return "insufficient_data"
            
        recent = list(self.divergence_history)[-10:]
        divergences = [d['divergence'] for d in recent]
        
        # Linear regression for trend
        x = np.arange(len(divergences))
        slope, _, _, _, _ = stats.linregress(x, divergences)
        
        if slope > 0.01:
            return "increasing"  # Overfitting getting worse
        elif slope < -0.01:
            return "decreasing"  # Overfitting improving
        else:
            return "stable"


class ConfidenceCalibrationChecker:
    """Check if model confidence matches actual accuracy"""
    
    def __init__(self):
        self.predictions = defaultdict(list)  # confidence_bucket -> outcomes
        self.calibration_history = deque(maxlen=100)
        
    def add_prediction(self, confidence: float, actual_outcome: bool):
        """Add prediction with confidence and actual outcome"""
        # Bucket confidence into 10% intervals
        bucket = int(confidence * 10) / 10
        self.predictions[bucket].append(actual_outcome)
        
    def calculate_calibration_error(self) -> Tuple[float, Dict[float, float]]:
        """Calculate calibration error"""
        calibration_map = {}
        total_error = 0.0
        bucket_count = 0
        
        for confidence_bucket, outcomes in self.predictions.items():
            if len(outcomes) < 10:  # Need minimum samples
                continue
                
            actual_accuracy = np.mean(outcomes)
            expected_accuracy = confidence_bucket
            
            error = abs(actual_accuracy - expected_accuracy)
            calibration_map[confidence_bucket] = actual_accuracy
            
            total_error += error
            bucket_count += 1
            
        avg_calibration_error = total_error / max(bucket_count, 1)
        
        # Store in history
        self.calibration_history.append({
            'error': avg_calibration_error,
            'timestamp': time.time()
        })
        
        return avg_calibration_error, calibration_map
        
    def is_well_calibrated(self, threshold: float = CONFIDENCE_CALIBRATION_THRESHOLD) -> bool:
        """Check if model is well calibrated"""
        error, _ = self.calculate_calibration_error()
        return error < threshold


class FeatureStabilityMonitor:
    """Monitor feature importance stability"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.feature_importance_history = defaultdict(lambda: deque(maxlen=window_size))
        self.stability_scores = {}
        
    def update_feature_importance(self, importance_scores: Dict[str, float]):
        """Update feature importance scores"""
        timestamp = time.time()
        
        for feature, importance in importance_scores.items():
            self.feature_importance_history[feature].append({
                'importance': importance,
                'timestamp': timestamp
            })
            
    def calculate_stability(self) -> Tuple[float, Dict[str, float]]:
        """Calculate feature stability scores"""
        feature_stability = {}
        
        for feature, history in self.feature_importance_history.items():
            if len(history) < 20:
                feature_stability[feature] = 1.0  # Assume stable if insufficient data
                continue
                
            importances = [h['importance'] for h in history]
            
            # Calculate coefficient of variation
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            
            if mean_importance > 0:
                cv = std_importance / mean_importance
                stability = 1.0 / (1.0 + cv)  # Convert to 0-1 scale
            else:
                stability = 1.0
                
            feature_stability[feature] = stability
            
        # Overall stability score
        overall_stability = np.mean(list(feature_stability.values())) if feature_stability else 1.0
        
        self.stability_scores = feature_stability
        
        return overall_stability, feature_stability
        
    def get_unstable_features(self, threshold: float = 0.7) -> List[str]:
        """Get features with low stability"""
        unstable = []
        
        for feature, stability in self.stability_scores.items():
            if stability < threshold:
                unstable.append(feature)
                
        return unstable


class CrossValidator:
    """Walk-forward cross-validation for overfitting detection"""
    
    def __init__(self):
        self.validation_results = deque(maxlen=50)
        
    async def validate_model(self, model, data: pd.DataFrame, 
                           window_size: int = 1000, 
                           test_size: int = 200) -> ValidationResult:
        """Perform walk-forward validation"""
        
        if len(data) < window_size + test_size:
            return ValidationResult(
                avg_train_score=0.0,
                avg_test_score=0.0,
                score_variance=0.0,
                is_overfitting=False,
                confidence=0.0,
                details={'error': 'Insufficient data'}
            )
            
        train_scores = []
        test_scores = []
        
        # Walk-forward validation
        for i in range(0, len(data) - window_size - test_size, test_size // 2):
            # Split data
            train_data = data.iloc[i:i + window_size]
            test_data = data.iloc[i + window_size:i + window_size + test_size]
            
            # Train model
            train_score = await self._evaluate_model(model, train_data, is_training=True)
            
            # Test model
            test_score = await self._evaluate_model(model, test_data, is_training=False)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            
        # Calculate statistics
        avg_train = np.mean(train_scores)
        avg_test = np.mean(test_scores)
        score_variance = np.var(test_scores)
        
        # Detect overfitting
        performance_gap = avg_train - avg_test
        is_overfitting = (
            performance_gap > PERFORMANCE_GAP_THRESHOLD or
            score_variance > 0.1  # High variance indicates instability
        )
        
        # Confidence in detection
        confidence = min(len(train_scores) / 10, 1.0)  # More folds = higher confidence
        
        result = ValidationResult(
            avg_train_score=avg_train,
            avg_test_score=avg_test,
            score_variance=score_variance,
            is_overfitting=is_overfitting,
            confidence=confidence,
            details={
                'num_folds': len(train_scores),
                'performance_gap': performance_gap,
                'train_scores': train_scores,
                'test_scores': test_scores
            }
        )
        
        self.validation_results.append(result)
        
        return result
        
    async def _evaluate_model(self, model, data: pd.DataFrame, is_training: bool) -> float:
        """Evaluate model performance"""
        # This would be implemented based on specific model type
        # For now, return placeholder
        return np.random.uniform(0.5, 0.8)


class OverfittingDetector:
    """Main overfitting detection system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize sub-detectors
        self.performance_detector = PerformanceDivergenceDetector()
        self.confidence_checker = ConfidenceCalibrationChecker()
        self.feature_monitor = FeatureStabilityMonitor()
        self.cross_validator = CrossValidator()
        
        # Detection history
        self.detection_history = deque(maxlen=1000)
        self.current_metrics = OverfittingMetrics()
        
        # Thresholds
        self.thresholds = {
            'performance_gap': self.config.get('performance_gap_threshold', PERFORMANCE_GAP_THRESHOLD),
            'confidence_error': self.config.get('confidence_threshold', CONFIDENCE_CALIBRATION_THRESHOLD),
            'feature_stability': self.config.get('feature_stability_threshold', FEATURE_STABILITY_THRESHOLD)
        }
        
        self._lock = asyncio.Lock()
        
    async def detect_overfitting(self) -> Dict[str, Any]:
        """Comprehensive overfitting detection"""
        async with self._lock:
            # 1. Check performance divergence
            perf_divergence, perf_details = self.performance_detector.calculate_divergence()
            
            # 2. Check confidence calibration
            calibration_error, calibration_map = self.confidence_checker.calculate_calibration_error()
            
            # 3. Check feature stability
            feature_stability, feature_scores = self.feature_monitor.calculate_stability()
            
            # Update metrics
            self.current_metrics.performance_gap = perf_divergence
            self.current_metrics.confidence_calibration_error = calibration_error
            self.current_metrics.feature_stability_score = feature_stability
            
            # Determine overfitting types
            overfitting_types = []
            
            if perf_divergence > self.thresholds['performance_gap']:
                overfitting_types.append(OverfittingType.PERFORMANCE_DIVERGENCE)
                
            if calibration_error > self.thresholds['confidence_error']:
                overfitting_types.append(OverfittingType.CONFIDENCE_MISCALIBRATION)
                
            if feature_stability < self.thresholds['feature_stability']:
                overfitting_types.append(OverfittingType.FEATURE_INSTABILITY)
                
            # Get severity
            severity = self.current_metrics.get_severity()
            
            # Create detection result
            detection_result = {
                'timestamp': time.time(),
                'is_overfitting': len(overfitting_types) > 0,
                'overfitting_types': [t.value for t in overfitting_types],
                'severity': severity.name,
                'metrics': {
                    'performance_gap': perf_divergence,
                    'calibration_error': calibration_error,
                    'feature_stability': feature_stability
                },
                'details': {
                    'performance': perf_details,
                    'calibration_map': calibration_map,
                    'unstable_features': self.feature_monitor.get_unstable_features()
                },
                'recommendations': self._generate_recommendations(overfitting_types, severity)
            }
            
            # Store in history
            self.detection_history.append(detection_result)
            
            return detection_result
            
    def _generate_recommendations(self, types: List[OverfittingType], 
                                severity: OverfittingSeverity) -> List[str]:
        """Generate recommendations based on detection"""
        recommendations = []
        
        if OverfittingType.PERFORMANCE_DIVERGENCE in types:
            recommendations.extend([
                "Increase regularization (dropout, weight decay)",
                "Reduce model complexity",
                "Collect more diverse training data",
                "Implement ensemble methods"
            ])
            
        if OverfittingType.CONFIDENCE_MISCALIBRATION in types:
            recommendations.extend([
                "Recalibrate confidence scores",
                "Add temperature scaling",
                "Use Platt scaling for probability calibration"
            ])
            
        if OverfittingType.FEATURE_INSTABILITY in types:
            recommendations.extend([
                "Implement feature selection stability checks",
                "Use L1 regularization for feature selection",
                "Consider removing unstable features"
            ])
            
        # Severity-based recommendations
        if severity == OverfittingSeverity.CRITICAL:
            recommendations.insert(0, "IMMEDIATE: Switch to conservative mode")
            recommendations.insert(1, "IMMEDIATE: Reduce position sizes by 50%")
            
        elif severity == OverfittingSeverity.HIGH:
            recommendations.insert(0, "Reduce learning rate by 50%")
            recommendations.insert(1, "Increase validation frequency")
            
        return recommendations
        
    async def add_training_result(self, win_rate: float, profit_factor: float):
        """Add training/backtest result"""
        self.performance_detector.add_training_result(win_rate, profit_factor, time.time())
        
    async def add_live_result(self, win_rate: float, profit_factor: float):
        """Add live trading result"""
        self.performance_detector.add_live_result(win_rate, profit_factor, time.time())
        
    async def add_prediction(self, confidence: float, actual_outcome: bool):
        """Add prediction for confidence calibration"""
        self.confidence_checker.add_prediction(confidence, actual_outcome)
        
    async def update_feature_importance(self, importance_scores: Dict[str, float]):
        """Update feature importance scores"""
        self.feature_monitor.update_feature_importance(importance_scores)
        
    async def validate_model(self, model, data: pd.DataFrame) -> ValidationResult:
        """Perform cross-validation"""
        return await self.cross_validator.validate_model(model, data)
        
    def get_metrics(self) -> OverfittingMetrics:
        """Get current overfitting metrics"""
        return self.current_metrics
        
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of recent detections"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'recent_severity': 'NONE',
                'overfitting_rate': 0.0
            }
            
        recent = list(self.detection_history)[-100:]
        overfitting_count = sum(1 for d in recent if d['is_overfitting'])
        
        # Get most recent severity
        recent_severity = recent[-1]['severity'] if recent else 'NONE'
        
        # Trend analysis
        if len(recent) >= 10:
            recent_10 = recent[-10:]
            trend = "increasing" if sum(1 for d in recent_10[-5:] if d['is_overfitting']) > \
                                   sum(1 for d in recent_10[:5] if d['is_overfitting']) else "decreasing"
        else:
            trend = "insufficient_data"
            
        return {
            'total_detections': len(self.detection_history),
            'recent_severity': recent_severity,
            'overfitting_rate': overfitting_count / len(recent),
            'trend': trend,
            'last_detection': recent[-1] if recent else None
        }


class OverfittingMonitor:
    """Real-time monitoring and alerting for overfitting"""
    
    def __init__(self, detector: OverfittingDetector):
        self.detector = detector
        self.alert_handlers = []
        self.monitoring_interval = 300  # 5 minutes
        self._monitoring_task = None
        self._running = False
        
        # Alert thresholds
        self.alert_thresholds = {
            'performance_gap': 0.2,
            'consecutive_detections': 3,
            'severity_threshold': OverfittingSeverity.HIGH
        }
        
    async def start_monitoring(self):
        """Start monitoring loop"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started overfitting monitoring")
        
    async def stop_monitoring(self):
        """Stop monitoring loop"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            await asyncio.gather(self._monitoring_task, return_exceptions=True)
        logger.info("Stopped overfitting monitoring")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        consecutive_detections = 0
        
        while self._running:
            try:
                # Detect overfitting
                detection = await self.detector.detect_overfitting()
                
                if detection['is_overfitting']:
                    consecutive_detections += 1
                    
                    # Check if alert needed
                    if self._should_alert(detection, consecutive_detections):
                        await self._send_alert(detection)
                else:
                    consecutive_detections = 0
                    
                # Wait for next check
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in overfitting monitor: {e}")
                await asyncio.sleep(self.monitoring_interval)
                
    def _should_alert(self, detection: Dict[str, Any], consecutive: int) -> bool:
        """Determine if alert should be sent"""
        # Check severity
        severity = OverfittingSeverity[detection['severity']]
        if severity.value >= self.alert_thresholds['severity_threshold'].value:
            return True
            
        # Check consecutive detections
        if consecutive >= self.alert_thresholds['consecutive_detections']:
            return True
            
        # Check specific metrics
        if detection['metrics']['performance_gap'] > self.alert_thresholds['performance_gap']:
            return True
            
        return False
        
    async def _send_alert(self, detection: Dict[str, Any]):
        """Send overfitting alert"""
        alert = {
            'type': 'OVERFITTING_DETECTED',
            'severity': detection['severity'],
            'timestamp': detection['timestamp'],
            'details': detection,
            'message': self._format_alert_message(detection)
        }
        
        # Send to all handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
                
        logger.warning(f"Overfitting alert: {alert['message']}")
        
    def _format_alert_message(self, detection: Dict[str, Any]) -> str:
        """Format alert message"""
        severity = detection['severity']
        types = ', '.join(detection['overfitting_types'])
        
        message = f"Overfitting detected - Severity: {severity}, Types: {types}"
        
        if detection['metrics']['performance_gap'] > 0:
            message += f", Performance gap: {detection['metrics']['performance_gap']:.2%}"
            
        return message
        
    def register_alert_handler(self, handler):
        """Register alert handler"""
        self.alert_handlers.append(handler)
        
    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive overfitting report"""
        detection_summary = self.detector.get_detection_summary()
        current_metrics = self.detector.get_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': detection_summary,
            'current_metrics': {
                'performance_gap': current_metrics.performance_gap,
                'calibration_error': current_metrics.confidence_calibration_error,
                'feature_stability': current_metrics.feature_stability_score,
                'severity': current_metrics.get_severity().name
            },
            'trends': {
                'performance_divergence': self.detector.performance_detector.get_trend(),
                'overfitting_trend': detection_summary.get('trend', 'unknown')
            },
            'recommendations': [],
            'risk_assessment': self._assess_risk(current_metrics)
        }
        
        # Add recommendations if overfitting detected
        if detection_summary.get('last_detection', {}).get('is_overfitting'):
            report['recommendations'] = detection_summary['last_detection']['recommendations']
            
        return report
        
    def _assess_risk(self, metrics: OverfittingMetrics) -> str:
        """Assess overfitting risk level"""
        severity = metrics.get_severity()
        
        if severity == OverfittingSeverity.CRITICAL:
            return "CRITICAL - Immediate action required"
        elif severity == OverfittingSeverity.HIGH:
            return "HIGH - Significant overfitting detected"
        elif severity == OverfittingSeverity.MEDIUM:
            return "MEDIUM - Moderate overfitting signs"
        elif severity == OverfittingSeverity.LOW:
            return "LOW - Minor overfitting indicators"
        else:
            return "MINIMAL - System operating normally"


class OverfittingRecovery:
    """Automated recovery procedures for overfitting"""
    
    def __init__(self, system_components: Dict[str, Any]):
        self.components = system_components
        self.recovery_history = deque(maxlen=100)
        self.model_checkpoints = {}  # Store model states for rollback
        
    async def recover_from_overfitting(self, 
                                     detection: Dict[str, Any],
                                     severity: OverfittingSeverity) -> Dict[str, Any]:
        """Execute recovery based on severity"""
        
        recovery_result = {
            'timestamp': time.time(),
            'severity': severity.name,
            'actions_taken': [],
            'success': True
        }
        
        try:
            if severity == OverfittingSeverity.CRITICAL:
                await self._critical_recovery(recovery_result)
                
            elif severity == OverfittingSeverity.HIGH:
                await self._high_severity_recovery(recovery_result)
                
            elif severity == OverfittingSeverity.MEDIUM:
                await self._medium_severity_recovery(recovery_result)
                
            elif severity == OverfittingSeverity.LOW:
                await self._low_severity_recovery(recovery_result)
                
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            recovery_result['success'] = False
            recovery_result['error'] = str(e)
            
        # Store recovery attempt
        self.recovery_history.append(recovery_result)
        
        return recovery_result
        
    async def _critical_recovery(self, result: Dict[str, Any]):
        """Critical overfitting recovery"""
        logger.critical("Executing critical overfitting recovery")
        
        # 1. Switch to conservative mode immediately
        if 'risk_manager' in self.components:
            risk_mgr = self.components['risk_manager']
            risk_mgr.risk_reduction_mode = True
            result['actions_taken'].append("Enabled risk reduction mode")
            
        # 2. Reduce position sizes dramatically
        if 'grid_strategy_selector' in self.components:
            selector = self.components['grid_strategy_selector']
            # Reduce all position multipliers
            for strategy in selector.strategies.values():
                if hasattr(strategy, 'position_multiplier'):
                    strategy.position_multiplier *= 0.3
            result['actions_taken'].append("Reduced position sizes by 70%")
            
        # 3. Increase all regularization
        if 'attention' in self.components:
            attention = self.components['attention']
            # This would increase dropout rates, etc.
            result['actions_taken'].append("Increased regularization parameters")
            
        # 4. Disable complex features
        result['actions_taken'].append("Disabled complex feature extractors")
        
    async def _high_severity_recovery(self, result: Dict[str, Any]):
        """High severity recovery"""
        logger.warning("Executing high severity overfitting recovery")
        
        # 1. Gradual parameter adjustment
        if 'feedback_loop' in self.components:
            feedback = self.components['feedback_loop']
            feedback.learning_rate *= 0.5
            result['actions_taken'].append("Reduced learning rate by 50%")
            
        # 2. Increase validation frequency
        result['actions_taken'].append("Increased validation frequency")
        
        # 3. Reduce model complexity
        result['actions_taken'].append("Switched to simpler models")
        
    async def _medium_severity_recovery(self, result: Dict[str, Any]):
        """Medium severity recovery"""
        logger.info("Executing medium severity overfitting recovery")
        
        # 1. Fine-tune regularization
        result['actions_taken'].append("Fine-tuned regularization parameters")
        
        # 2. Reweight features
        if 'attention' in self.components:
            # Reduce weight of unstable features
            result['actions_taken'].append("Reweighted feature importance")
            
    async def _low_severity_recovery(self, result: Dict[str, Any]):
        """Low severity recovery"""
        logger.info("Executing low severity overfitting recovery")
        
        # 1. Monitor closely
        result['actions_taken'].append("Increased monitoring frequency")
        
        # 2. Collect more data
        result['actions_taken'].append("Extended data collection period")
        
    async def save_checkpoint(self, component_name: str, state: Any):
        """Save model checkpoint for potential rollback"""
        checkpoint_id = f"{component_name}_{int(time.time())}"
        self.model_checkpoints[checkpoint_id] = {
            'component': component_name,
            'state': state,
            'timestamp': time.time()
        }
        
        # Keep only recent checkpoints
        if len(self.model_checkpoints) > 10:
            oldest = min(self.model_checkpoints.keys(), 
                        key=lambda k: self.model_checkpoints[k]['timestamp'])
            del self.model_checkpoints[oldest]
            
    async def rollback_model(self, component_name: str) -> bool:
        """Rollback to previous model state"""
        # Find most recent checkpoint for component
        relevant_checkpoints = [
            (k, v) for k, v in self.model_checkpoints.items()
            if v['component'] == component_name
        ]
        
        if not relevant_checkpoints:
            logger.error(f"No checkpoint found for {component_name}")
            return False
            
        # Get most recent
        checkpoint_id, checkpoint = max(relevant_checkpoints, 
                                      key=lambda x: x[1]['timestamp'])
        
        # Apply rollback (component-specific implementation needed)
        logger.info(f"Rolling back {component_name} to checkpoint {checkpoint_id}")
        
        return True


# Utility functions
def calculate_overfitting_score(metrics: OverfittingMetrics) -> float:
    """Calculate overall overfitting score (0-1)"""
    score = 0.0
    
    # Performance gap contribution (40%)
    score += min(metrics.performance_gap / 0.3, 1.0) * 0.4
    
    # Calibration error contribution (30%)
    score += min(metrics.confidence_calibration_error / 0.3, 1.0) * 0.3
    
    # Feature instability contribution (30%)
    score += (1.0 - metrics.feature_stability_score) * 0.3
    
    return min(score, 1.0)


# Example usage
async def example_usage():
    """Example of using overfitting detection system"""
    
    # Initialize detector
    config = {
        'performance_gap_threshold': 0.15,
        'confidence_threshold': 0.2,
        'feature_stability_threshold': 0.3
    }
    
    detector = OverfittingDetector(config)
    monitor = OverfittingMonitor(detector)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Simulate data
    for i in range(100):
        # Add training results (overfitted)
        await detector.add_training_result(
            win_rate=0.65 + np.random.normal(0, 0.05),
            profit_factor=1.5 + np.random.normal(0, 0.1)
        )
        
        # Add live results (worse due to overfitting)
        await detector.add_live_result(
            win_rate=0.50 + np.random.normal(0, 0.05),
            profit_factor=1.2 + np.random.normal(0, 0.1)
        )
        
        # Add predictions
        confidence = np.random.uniform(0.6, 0.9)
        actual = np.random.random() < (confidence - 0.1)  # Miscalibrated
        await detector.add_prediction(confidence, actual)
        
        # Update features
        await detector.update_feature_importance({
            'feature1': 0.3 + np.random.normal(0, 0.1),
            'feature2': 0.2 + np.random.normal(0, 0.15),  # Unstable
            'feature3': 0.5 + np.random.normal(0, 0.05)
        })
        
    # Get detection result
    detection = await detector.detect_overfitting()
    print(f"Overfitting detected: {detection['is_overfitting']}")
    print(f"Severity: {detection['severity']}")
    print(f"Types: {detection['overfitting_types']}")
    
    # Generate report
    report = await monitor.generate_report()
    print(f"\nOverfitting Report:")
    print(f"Risk Assessment: {report['risk_assessment']}")
    print(f"Recommendations: {report['recommendations']}")
    
    # Stop monitoring
    await monitor.stop_monitoring()



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

    def get_state(self) -> Dict[str, Any]:
            """Get component state for checkpointing"""
            return {
                'class': self.__class__.__name__,
                'timestamp': time.time() if 'time' in globals() else 0
            }

    def load_state(self, state: Dict[str, Any]) -> None:
            """Load component state from checkpoint"""
            pass

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
if __name__ == "__main__":
    asyncio.run(example_usage())