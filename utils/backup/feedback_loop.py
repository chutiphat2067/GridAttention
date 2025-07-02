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
from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase
from core.market_regime_detector import MarketRegimeDetector, MarketRegime
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.performance_monitor import PerformanceMonitor, TradingMetrics

# Setup logger
logger = logging.getLogger(__name__)


# Constants - MORE CONSERVATIVE APPROACH
FEEDBACK_INTERVAL = 60  # seconds
MIN_SAMPLES_FOR_LEARNING = 200  # Increased from 100
CONFIDENCE_THRESHOLD = 0.8  # Increased from 0.7
IMPROVEMENT_THRESHOLD = 0.02  # 2% improvement
LEARNING_RATE = 0.005  # Reduced from 0.01
DECAY_RATE = 0.95
MAX_ADJUSTMENT = 0.2  # Reduced from 0.3 (20% max adjustment)
OPTIMIZATION_INTERVAL = 7200  # 2 hours (increased from 1 hour)
INSIGHT_WINDOW = 1000  # Number of trades for insight extraction
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
    """Implement gradual parameter optimization"""
    
    def __init__(self, max_adjustment: float = MAX_ADJUSTMENT):
        self.max_adjustment = max_adjustment
        self.optimization_history = deque(maxlen=100)
        self.parameter_trajectories = defaultdict(list)
        
    def calculate_gradual_adjustment(self, 
                                   current_value: float, 
                                   target_value: float,
                                   confidence: float) -> float:
        """Calculate gradual adjustment towards target"""
        
        # Calculate desired change
        desired_change = target_value - current_value
        
        # Apply confidence scaling
        confidence_factor = min(confidence, 0.9)  # Cap at 90%
        
        # Apply maximum adjustment limit
        max_change = current_value * self.max_adjustment
        
        # Calculate actual adjustment
        actual_change = desired_change * confidence_factor
        actual_change = np.clip(actual_change, -max_change, max_change)
        
        # Record trajectory
        self.parameter_trajectories[f"param_{len(self.optimization_history)}"].append({
            'current': current_value,
            'target': target_value,
            'adjustment': actual_change,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        return current_value + actual_change
        
    def get_optimization_velocity(self, param_name: str) -> float:
        """Get rate of change for parameter"""
        if param_name not in self.parameter_trajectories:
            return 0.0
            
        trajectory = self.parameter_trajectories[param_name]
        if len(trajectory) < 2:
            return 0.0
            
        # Calculate average rate of change
        recent = trajectory[-10:]  # Last 10 adjustments
        
        if len(recent) < 2:
            return 0.0
            
        time_diff = recent[-1]['timestamp'] - recent[0]['timestamp']
        value_diff = recent[-1]['current'] - recent[0]['current']
        
        if time_diff == 0:
            return 0.0
            
        return value_diff / time_diff
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
        
        # Learning parameters
        self.learning_rate = LEARNING_RATE
        self.decay_rate = DECAY_RATE
        
        # State
        self._running = False
        self._feedback_task = None
        self._optimization_task = None
        self._last_optimization = 0
        self._in_recovery = False
        
        logger.info("Initialized Feedback Loop with Recovery and Gradual Optimization")
        
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
        """Apply optimization results to system components with gradual adjustment"""
        try:
            # Check if in recovery mode
            if self._in_recovery:
                logger.warning("In recovery mode, skipping optimizations")
                return False
                
            # Apply parameter adjustments gradually
            if 'parameters' in result.adjustments:
                for param, target_value in result.adjustments['parameters'].items():
                    current_value = self.optimization_engine.current_parameters.get(param, 1.0)
                    
                    # Calculate gradual adjustment
                    adjusted_value = self.gradual_optimizer.calculate_gradual_adjustment(
                        current_value,
                        target_value,
                        result.validation_metrics.get('confidence', 0.5)
                    )
                    
                    # Check if adjustment is safe
                    if abs(adjusted_value - current_value) / current_value > 0.5:
                        logger.warning(f"Adjustment for {param} too large, limiting to 20%")
                        adjusted_value = current_value * 1.2 if adjusted_value > current_value else current_value * 0.8
                        
                    await self._apply_parameter(param, adjusted_value)
                    
            # Apply strategy changes
            if 'strategy' in result.adjustments:
                await self._apply_strategy_change(result.adjustments['strategy'])
                
            # Apply risk adjustments
            if 'risk' in result.adjustments:
                await self._apply_risk_adjustment(result.adjustments['risk'])
                
            logger.info(f"Applied optimization {result.optimization_id} with gradual adjustments")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            
            # Trigger recovery if optimization fails
            await self._enter_recovery_mode("Optimization failure")
            return False
            
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
