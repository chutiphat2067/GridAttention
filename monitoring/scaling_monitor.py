"""
scaling_monitor.py
Monitor and predict scaling needs for grid trading system

Author: Grid Trading System
Date: 2024
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from scipy import stats
import psutil
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Constants
METRIC_HISTORY_SIZE = 10080  # 1 week of minute-by-minute data
PREDICTION_CONFIDENCE_THRESHOLD = 0.8
SCALING_CHECK_INTERVAL = 60  # seconds

class ScalingMetric(Enum):
    """Metrics tracked for scaling decisions"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    EXECUTION_LATENCY = "execution_latency"
    QUEUE_DEPTH = "queue_depth"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    ORDERS_PER_SECOND = "orders_per_second"
    TICKS_PER_SECOND = "ticks_per_second"
    ML_INFERENCE_TIME = "ml_inference_time"
    FEATURE_EXTRACTION_TIME = "feature_extraction_time"

class ScalingAction(Enum):
    """Possible scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_OUT = "scale_out"
    OPTIMIZE = "optimize"
    MIGRATE = "migrate"
    ARCHIVE = "archive"
    NONE = "none"

@dataclass
class ScalingRecommendation:
    """Scaling recommendation with details"""
    priority: str  # IMMEDIATE, HIGH, MEDIUM, LOW
    action: ScalingAction
    component: str
    reason: str
    steps: List[str]
    estimated_cost: float = 0.0
    estimated_impact: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'priority': self.priority,
            'action': self.action.value,
            'component': self.component,
            'reason': self.reason,
            'steps': self.steps,
            'estimated_cost': self.estimated_cost,
            'estimated_impact': self.estimated_impact,
            'confidence': self.confidence
        }

@dataclass
class ScalingThresholds:
    """Configurable thresholds for scaling decisions"""
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    latency_warning: float = 20.0  # ms
    latency_critical: float = 50.0  # ms
    queue_warning: int = 500
    queue_critical: int = 1000
    error_rate_warning: float = 0.01  # 1%
    error_rate_critical: float = 0.05  # 5%

class GrowthPredictor:
    """Predict resource usage growth using statistical methods"""
    
    def __init__(self):
        self.min_data_points = 100
        
    def predict(self, metric_history: Dict[str, deque]) -> Dict[str, Dict[str, Any]]:
        """Predict future values for all metrics"""
        predictions = {}
        
        for metric_name, history in metric_history.items():
            if len(history) < self.min_data_points:
                continue
                
            # Extract time series
            data = list(history)
            times = np.array([h[0] for h in data])
            values = np.array([h[1] for h in data])
            
            # Remove outliers
            values_clean = self._remove_outliers(values)
            
            # Multiple prediction methods
            linear_pred = self._linear_prediction(times, values_clean)
            exp_pred = self._exponential_prediction(times, values_clean)
            
            # Choose best model
            best_pred = linear_pred if linear_pred['r_squared'] > exp_pred['r_squared'] else exp_pred
            
            predictions[metric_name] = best_pred
            
        return predictions
        
    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """Remove outliers using IQR method"""
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return values[(values >= lower_bound) & (values <= upper_bound)]
        
    def _linear_prediction(self, times: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """Linear regression prediction"""
        slope, intercept, r_value, _, _ = stats.linregress(times, values)
        
        current_time = time.time()
        
        return {
            'model': 'linear',
            'current': values[-1] if len(values) > 0 else 0,
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'r_squared': r_value ** 2,
            '1_hour': intercept + slope * (current_time + 3600),
            '1_day': intercept + slope * (current_time + 86400),
            '1_week': intercept + slope * (current_time + 604800),
            'days_until_critical': self._calculate_days_to_threshold(
                slope, intercept, current_time, 90  # 90% as critical
            )
        }
        
    def _exponential_prediction(self, times: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """Exponential growth prediction"""
        # Log transform for exponential fit
        log_values = np.log(values + 1)  # +1 to avoid log(0)
        slope, intercept, r_value, _, _ = stats.linregress(times, log_values)
        
        current_time = time.time()
        
        return {
            'model': 'exponential',
            'current': values[-1] if len(values) > 0 else 0,
            'trend': 'exponential_growth' if slope > 0 else 'exponential_decay',
            'growth_rate': slope,
            'r_squared': r_value ** 2,
            '1_hour': np.exp(intercept + slope * (current_time + 3600)) - 1,
            '1_day': np.exp(intercept + slope * (current_time + 86400)) - 1,
            '1_week': np.exp(intercept + slope * (current_time + 604800)) - 1,
            'days_until_critical': self._calculate_days_to_threshold_exp(
                slope, intercept, current_time, 90
            )
        }
        
    def _calculate_days_to_threshold(self, slope: float, intercept: float, 
                                   current_time: float, threshold: float) -> Optional[float]:
        """Calculate days until reaching threshold (linear)"""
        if slope <= 0:
            return None  # Never reaches threshold
            
        time_to_threshold = (threshold - intercept) / slope
        days = (time_to_threshold - current_time) / 86400
        
        return days if days > 0 else 0
        
    def _calculate_days_to_threshold_exp(self, slope: float, intercept: float,
                                       current_time: float, threshold: float) -> Optional[float]:
        """Calculate days until reaching threshold (exponential)"""
        if slope <= 0:
            return None
            
        time_to_threshold = (np.log(threshold + 1) - intercept) / slope
        days = (time_to_threshold - current_time) / 86400
        
        return days if days > 0 else 0

class BottleneckDetector:
    """Detect system bottlenecks"""
    
    def __init__(self, thresholds: ScalingThresholds):
        self.thresholds = thresholds
        
    def detect(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect all bottlenecks in the system"""
        bottlenecks = []
        
        # CPU bottleneck
        cpu_usage = metrics.get('cpu_usage', 0)
        if cpu_usage > self.thresholds.cpu_warning:
            bottlenecks.append({
                'component': 'CPU',
                'metric': 'cpu_usage',
                'current_value': cpu_usage,
                'threshold': self.thresholds.cpu_critical,
                'severity': self._get_severity(cpu_usage, 
                                              self.thresholds.cpu_warning,
                                              self.thresholds.cpu_critical),
                'impact': 'Slow processing, increased latency across all operations',
                'suggestions': [
                    'Profile CPU hotspots using cProfile',
                    'Optimize algorithm complexity',
                    'Use NumPy/Numba for numerical operations',
                    'Scale horizontally with multiprocessing'
                ]
            })
            
        # Memory bottleneck
        memory_usage = metrics.get('memory_usage', 0)
        if memory_usage > self.thresholds.memory_warning:
            bottlenecks.append({
                'component': 'Memory',
                'metric': 'memory_usage',
                'current_value': memory_usage,
                'threshold': self.thresholds.memory_critical,
                'severity': self._get_severity(memory_usage,
                                              self.thresholds.memory_warning,
                                              self.thresholds.memory_critical),
                'impact': 'Risk of OOM kills, slow garbage collection, swapping',
                'suggestions': [
                    'Implement data archiving to disk/S3',
                    'Reduce in-memory buffer sizes',
                    'Use memory-mapped files for large datasets',
                    'Add more RAM or scale vertically'
                ]
            })
            
        # Latency bottleneck
        latency = metrics.get('execution_latency_p99', 0)
        if latency > self.thresholds.latency_warning:
            bottlenecks.append({
                'component': 'Execution',
                'metric': 'execution_latency_p99',
                'current_value': latency,
                'threshold': self.thresholds.latency_critical,
                'severity': self._get_severity(latency,
                                              self.thresholds.latency_warning,
                                              self.thresholds.latency_critical),
                'impact': 'Missed trading opportunities, poor fill rates',
                'suggestions': [
                    'Add connection pooling',
                    'Implement request batching',
                    'Use faster networking (10Gb)',
                    'Consider colocated servers'
                ]
            })
            
        # Queue bottleneck
        queue_depth = metrics.get('execution_queue_depth', 0)
        if queue_depth > self.thresholds.queue_warning:
            bottlenecks.append({
                'component': 'Queue',
                'metric': 'execution_queue_depth',
                'current_value': queue_depth,
                'threshold': self.thresholds.queue_critical,
                'severity': self._get_severity(queue_depth,
                                              self.thresholds.queue_warning,
                                              self.thresholds.queue_critical),
                'impact': 'Delayed order execution, accumulating backlog',
                'suggestions': [
                    'Add more execution workers',
                    'Implement priority queuing',
                    'Optimize order processing logic',
                    'Consider parallel execution pipelines'
                ]
            })
            
        # ML bottleneck
        ml_time = metrics.get('ml_inference_time', 0)
        if ml_time > 10:  # 10ms threshold
            bottlenecks.append({
                'component': 'ML_Pipeline',
                'metric': 'ml_inference_time',
                'current_value': ml_time,
                'threshold': 10,
                'severity': 'MEDIUM' if ml_time < 20 else 'HIGH',
                'impact': 'Slow feature extraction and decision making',
                'suggestions': [
                    'Use GPU acceleration for inference',
                    'Implement model quantization',
                    'Cache frequent predictions',
                    'Use TensorFlow Lite or ONNX Runtime'
                ]
            })
            
        return bottlenecks
        
    def _get_severity(self, value: float, warning: float, critical: float) -> str:
        """Determine severity level"""
        if value >= critical:
            return 'CRITICAL'
        elif value >= warning:
            return 'HIGH'
        else:
            return 'MEDIUM'

class ScalingMonitor:
    """Main scaling monitor class"""
    
    def __init__(self, performance_monitor, execution_engine, market_data_input,
                 attention_layer=None, config: Optional[Dict[str, Any]] = None):
        self.perf_monitor = performance_monitor
        self.execution_engine = execution_engine
        self.market_data = market_data_input
        self.attention = attention_layer
        self.config = config or {}
        
        # Initialize components
        self.thresholds = ScalingThresholds(**self.config.get('thresholds', {}))
        self.growth_predictor = GrowthPredictor()
        self.bottleneck_detector = BottleneckDetector(self.thresholds)
        
        # Historical data
        self.metric_history = defaultdict(lambda: deque(maxlen=METRIC_HISTORY_SIZE))
        self.scaling_events = deque(maxlen=100)
        self.recommendations_history = deque(maxlen=50)
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = self.config.get('auto_scaling', {}).get('enabled', False)
        self.last_scaling_action = 0
        self.scaling_cooldown = 300  # 5 minutes between scaling actions
        
        # Monitoring state
        self._running = False
        self._monitor_task = None
        
    async def start(self):
        """Start scaling monitor"""
        if self._running:
            return
            
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started scaling monitor")
        
    async def stop(self):
        """Stop scaling monitor"""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)
            
        logger.info("Stopped scaling monitor")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Collect metrics
                metrics = await self.collect_scaling_metrics()
                
                # Analyze scaling needs
                analysis = await self.analyze_scaling_needs()
                
                # Log critical issues
                for issue in analysis.get('immediate_issues', []):
                    logger.critical(f"Scaling issue: {issue}")
                    
                # Auto-scale if enabled
                if self.auto_scaling_enabled:
                    await self._handle_auto_scaling(analysis)
                    
                # Wait for next iteration
                await asyncio.sleep(SCALING_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling monitor: {e}")
                await asyncio.sleep(SCALING_CHECK_INTERVAL)
                
    async def collect_scaling_metrics(self) -> Dict[str, float]:
        """Collect all metrics relevant for scaling decisions"""
        metrics = {}
        
        # System metrics
        metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
        metrics['memory_usage'] = psutil.virtual_memory().percent
        metrics['disk_usage'] = psutil.disk_usage('/').percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        metrics['network_bytes_sent'] = net_io.bytes_sent
        metrics['network_bytes_recv'] = net_io.bytes_recv
        
        # Process-specific metrics
        process = psutil.Process()
        metrics['process_memory_mb'] = process.memory_info().rss / 1024 / 1024
        metrics['process_threads'] = process.num_threads()
        metrics['process_fds'] = len(process.open_files())
        
        # Application metrics from execution engine
        if self.execution_engine:
            exec_stats = await self.execution_engine.get_execution_stats()
            metrics['execution_latency_avg'] = exec_stats['metrics'].get('average_latency', 0)
            metrics['execution_latency_p99'] = exec_stats['metrics'].get('max_latency', 0)
            metrics['active_orders'] = exec_stats.get('active_orders', 0)
            metrics['execution_queue_depth'] = getattr(
                self.execution_engine.execution_queue, 'qsize', lambda: 0
            )()
            metrics['orders_per_second'] = exec_stats['metrics'].get('total_orders', 0) / max(
                (time.time() - self.execution_engine.start_time), 1
            ) if hasattr(self.execution_engine, 'start_time') else 0
            
        # Market data metrics
        if self.market_data:
            data_stats = await self.market_data.get_statistics()
            metrics['ticks_per_second'] = data_stats.get('tick_rate', 0)
            metrics['buffer_usage_percent'] = (
                data_stats.get('buffer_size', 0) / 
                self.market_data.config.get('buffer_size', 1000) * 100
            )
            metrics['validation_failure_rate'] = data_stats.get('failure_rate', 0)
            
        # ML/Attention metrics
        if self.attention:
            attention_state = await self.attention.get_attention_state()
            metrics['attention_processing_time'] = attention_state.get('avg_processing_time', 0)
            metrics['total_observations'] = attention_state.get('total_observations', 0)
            metrics['learning_progress'] = self.attention.get_learning_progress() * 100
            
        # Performance metrics
        if self.perf_monitor:
            perf_report = await self.perf_monitor.get_performance_report()
            metrics['total_trades'] = perf_report.get('total_trades', 0)
            metrics['error_rate'] = perf_report.get('system_health', {}).get('error_rate', 0)
            
        # Calculate derived metrics
        metrics['memory_pressure'] = metrics['memory_usage'] * (
            1 + metrics.get('execution_queue_depth', 0) / 1000
        )
        
        # Store in history
        timestamp = time.time()
        for key, value in metrics.items():
            self.metric_history[key].append((timestamp, value))
            
        return metrics
        
    async def analyze_scaling_needs(self) -> Dict[str, Any]:
        """Comprehensive scaling analysis"""
        current_metrics = await self.collect_scaling_metrics()
        
        # Detect bottlenecks
        bottlenecks = self.bottleneck_detector.detect(current_metrics)
        
        # Predict future resource usage
        predictions = self.growth_predictor.predict(self.metric_history)
        
        # Check immediate issues
        immediate_issues = self._check_immediate_needs(current_metrics)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            current_metrics, bottlenecks, predictions
        )
        
        # Calculate scaling score
        scaling_score = self._calculate_scaling_score(
            current_metrics, bottlenecks, predictions
        )
        
        analysis = {
            'timestamp': time.time(),
            'current_metrics': current_metrics,
            'bottlenecks': bottlenecks,
            'predictions': predictions,
            'immediate_issues': immediate_issues,
            'recommendations': recommendations,
            'scaling_score': scaling_score,
            'requires_immediate_action': len(immediate_issues) > 0
        }
        
        return analysis
        
    def _check_immediate_needs(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for immediate scaling needs"""
        issues = []
        
        # CPU critical
        if metrics.get('cpu_usage', 0) > self.thresholds.cpu_critical:
            issues.append({
                'type': 'CPU_CRITICAL',
                'metric': 'cpu_usage',
                'current': metrics['cpu_usage'],
                'threshold': self.thresholds.cpu_critical,
                'action_required': 'IMMEDIATE_SCALE_UP'
            })
            
        # Memory critical
        if metrics.get('memory_usage', 0) > self.thresholds.memory_critical:
            issues.append({
                'type': 'MEMORY_CRITICAL',
                'metric': 'memory_usage',
                'current': metrics['memory_usage'],
                'threshold': self.thresholds.memory_critical,
                'action_required': 'ADD_MEMORY_OR_RESTART'
            })
            
        # Queue overflow imminent
        queue_depth = metrics.get('execution_queue_depth', 0)
        if queue_depth > self.thresholds.queue_critical:
            issues.append({
                'type': 'QUEUE_OVERFLOW',
                'metric': 'execution_queue_depth',
                'current': queue_depth,
                'threshold': self.thresholds.queue_critical,
                'action_required': 'ADD_WORKERS_IMMEDIATELY'
            })
            
        # Error rate spike
        error_rate = metrics.get('error_rate', 0)
        if error_rate > self.thresholds.error_rate_critical:
            issues.append({
                'type': 'ERROR_RATE_CRITICAL',
                'metric': 'error_rate',
                'current': error_rate,
                'threshold': self.thresholds.error_rate_critical,
                'action_required': 'INVESTIGATE_AND_SCALE'
            })
            
        return issues
        
    async def _generate_recommendations(self, metrics: Dict[str, float],
                                      bottlenecks: List[Dict[str, Any]],
                                      predictions: Dict[str, Dict[str, Any]]) -> List[ScalingRecommendation]:
        """Generate scaling recommendations"""
        recommendations = []
        
        # Immediate bottleneck recommendations
        for bottleneck in bottlenecks:
            if bottleneck['severity'] == 'CRITICAL':
                priority = 'IMMEDIATE'
            elif bottleneck['severity'] == 'HIGH':
                priority = 'HIGH'
            else:
                priority = 'MEDIUM'
                
            rec = ScalingRecommendation(
                priority=priority,
                action=self._determine_scaling_action(bottleneck),
                component=bottleneck['component'],
                reason=f"{bottleneck['metric']} at {bottleneck['current_value']:.1f}",
                steps=bottleneck['suggestions'],
                confidence=0.9 if bottleneck['severity'] == 'CRITICAL' else 0.7
            )
            recommendations.append(rec)
            
        # Predictive recommendations
        for metric_name, prediction in predictions.items():
            if prediction.get('days_until_critical') is not None:
                days = prediction['days_until_critical']
                
                if days < 1:
                    priority = 'HIGH'
                elif days < 7:
                    priority = 'MEDIUM'
                else:
                    priority = 'LOW'
                    
                if days < 7:  # Only recommend if within a week
                    rec = ScalingRecommendation(
                        priority=priority,
                        action=ScalingAction.SCALE_UP,
                        component=self._get_component_for_metric(metric_name),
                        reason=f"{metric_name} will reach critical in {days:.1f} days",
                        steps=[
                            f"Current {metric_name}: {prediction['current']:.1f}",
                            f"Predicted in 1 week: {prediction['1_week']:.1f}",
                            "Plan scaling before threshold is reached"
                        ],
                        confidence=prediction['r_squared']
                    )
                    recommendations.append(rec)
                    
        # Architecture recommendations based on patterns
        if self._should_recommend_architecture_change(metrics, predictions):
            rec = ScalingRecommendation(
                priority='MEDIUM',
                action=ScalingAction.MIGRATE,
                component='Architecture',
                reason='Current architecture reaching limits',
                steps=[
                    'Consider microservices architecture',
                    'Implement message queue (Kafka/RabbitMQ)',
                    'Add caching layer (Redis)',
                    'Split into specialized workers'
                ],
                estimated_impact='10x capacity increase',
                confidence=0.8
            )
            recommendations.append(rec)
            
        # Sort by priority
        priority_order = {'IMMEDIATE': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 4))
        
        return recommendations
        
    def _determine_scaling_action(self, bottleneck: Dict[str, Any]) -> ScalingAction:
        """Determine appropriate scaling action for bottleneck"""
        component = bottleneck['component']
        
        if component == 'CPU':
            return ScalingAction.SCALE_OUT  # Horizontal scaling for CPU
        elif component == 'Memory':
            return ScalingAction.SCALE_UP   # Vertical scaling for memory
        elif component == 'Queue':
            return ScalingAction.SCALE_OUT  # More workers
        elif component == 'ML_Pipeline':
            return ScalingAction.OPTIMIZE   # Optimization first
        else:
            return ScalingAction.OPTIMIZE
            
    def _get_component_for_metric(self, metric_name: str) -> str:
        """Map metric to component"""
        mapping = {
            'cpu_usage': 'CPU',
            'memory_usage': 'Memory',
            'execution_latency_p99': 'Execution Engine',
            'execution_queue_depth': 'Queue',
            'ml_inference_time': 'ML Pipeline',
            'ticks_per_second': 'Market Data',
            'error_rate': 'System'
        }
        return mapping.get(metric_name, 'System')
        
    def _should_recommend_architecture_change(self, metrics: Dict[str, float],
                                            predictions: Dict[str, Dict[str, Any]]) -> bool:
        """Determine if architecture change is needed"""
        indicators = 0
        
        # Multiple resources near limit
        if metrics.get('cpu_usage', 0) > 60 and metrics.get('memory_usage', 0) > 60:
            indicators += 1
            
        # High sustained load
        if metrics.get('orders_per_second', 0) > 50:
            indicators += 1
            
        # Multiple components predicted to hit limits
        critical_predictions = sum(
            1 for p in predictions.values()
            if p.get('days_until_critical', float('inf')) < 30
        )
        if critical_predictions >= 3:
            indicators += 1
            
        return indicators >= 2
        
    def _calculate_scaling_score(self, metrics: Dict[str, float],
                                bottlenecks: List[Dict[str, Any]],
                                predictions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall scaling urgency score (0-100)"""
        score = 0.0
        
        # Current resource usage (40% weight)
        cpu_score = min(metrics.get('cpu_usage', 0) / 100 * 100, 40)
        memory_score = min(metrics.get('memory_usage', 0) / 100 * 100, 40)
        score += (cpu_score + memory_score) * 0.4
        
        # Bottleneck severity (30% weight)
        bottleneck_score = 0
        for bottleneck in bottlenecks:
            if bottleneck['severity'] == 'CRITICAL':
                bottleneck_score = 100
                break
            elif bottleneck['severity'] == 'HIGH':
                bottleneck_score = max(bottleneck_score, 70)
            elif bottleneck['severity'] == 'MEDIUM':
                bottleneck_score = max(bottleneck_score, 40)
        score += bottleneck_score * 0.3
        
        # Growth rate (30% weight)
        growth_score = 0
        for prediction in predictions.values():
            if prediction.get('days_until_critical'):
                days = prediction['days_until_critical']
                if days < 7:
                    growth_score = max(growth_score, 100)
                elif days < 30:
                    growth_score = max(growth_score, 70)
                elif days < 90:
                    growth_score = max(growth_score, 40)
        score += growth_score * 0.3
        
        return min(score, 100)
        
    async def _handle_auto_scaling(self, analysis: Dict[str, Any]):
        """Handle automatic scaling actions"""
        # Check cooldown
        if time.time() - self.last_scaling_action < self.scaling_cooldown:
            return
            
        # Only auto-scale for immediate issues
        if not analysis.get('requires_immediate_action'):
            return
            
        # Log scaling event
        scaling_event = {
            'timestamp': time.time(),
            'trigger': analysis['immediate_issues'],
            'metrics': analysis['current_metrics'],
            'action': 'auto_scale'
        }
        self.scaling_events.append(scaling_event)
        
        # Trigger scaling action (would integrate with cloud provider API)
        logger.critical(f"AUTO-SCALING TRIGGERED: {analysis['immediate_issues']}")
        
        # Update last scaling time
        self.last_scaling_action = time.time()
        
    async def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling report"""
        analysis = await self.analyze_scaling_needs()
        
        # Calculate trends
        trends = {}
        for metric_name, history in self.metric_history.items():
            if len(history) > 100:
                values = [h[1] for h in list(history)[-100:]]
                trends[metric_name] = {
                    'current': values[-1] if values else 0,
                    'avg_1h': np.mean(values[-60:]) if len(values) >= 60 else 0,
                    'avg_24h': np.mean(values) if values else 0,
                    'trend': 'up' if values[-1] > np.mean(values) else 'down'
                }
                
        report = {
            'timestamp': datetime.now().isoformat(),
            'scaling_score': analysis['scaling_score'],
            'immediate_action_required': analysis['requires_immediate_action'],
            'current_metrics': analysis['current_metrics'],
            'bottlenecks': analysis['bottlenecks'],
            'recommendations': [r.to_dict() for r in analysis['recommendations']],
            'predictions': analysis['predictions'],
            'trends': trends,
            'recent_scaling_events': list(self.scaling_events)[-10:],
            'auto_scaling_enabled': self.auto_scaling_enabled
        }
        
        return report
        
    async def export_metrics_history(self, filepath: str):
        """Export metrics history for analysis"""
        data = {
            'export_time': datetime.now().isoformat(),
            'metrics': {}
        }
        
        for metric_name, history in self.metric_history.items():
            data['metrics'][metric_name] = [
                {'timestamp': h[0], 'value': h[1]}
                for h in list(history)
            ]
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Exported metrics history to {filepath}")

# Utility functions for integration
async def create_scaling_monitor(system_components: Dict[str, Any],
                               config: Optional[Dict[str, Any]] = None) -> ScalingMonitor:
    """Create and configure scaling monitor"""
    monitor = ScalingMonitor(
        performance_monitor=system_components.get('performance'),
        execution_engine=system_components.get('execution'),
        market_data_input=system_components.get('market_data'),
        attention_layer=system_components.get('attention'),
        config=config
    )
    
    return monitor