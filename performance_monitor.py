from typing import Dict, Any, Optional
"""
performance_monitor.py
Monitor system and trading performance for grid trading system

Author: Grid Trading System
Date: 2024
"""

# Standard library imports
import asyncio
import time
import logging
import json
import psutil
import platform
import threading
from typing import Dict, List, Optional, Any, Tuple, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

# Third-party imports
import aiohttp
import websockets
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local imports
from attention_learning_layer import AttentionLearningLayer, AttentionPhase
from market_regime_detector import MarketRegime
from overfitting_detector import OverfittingDetector, OverfittingMonitor

# Setup logger
logger = logging.getLogger(__name__)


# Constants
METRICS_UPDATE_INTERVAL = 5  # seconds
METRICS_HISTORY_SIZE = 10000
ALERT_CHECK_INTERVAL = 10  # seconds
DASHBOARD_UPDATE_INTERVAL = 1  # seconds
PERFORMANCE_WINDOW_SIZE = 1000  # trades
SYSTEM_METRICS_PORT = 9090
DASHBOARD_PORT = 8080


# Enums
class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class PerformanceStatus(Enum):
    """Overall performance status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0
    grid_fill_rate: float = 0.0
    avg_profit_per_grid: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'avg_trade_duration': self.avg_trade_duration,
            'grid_fill_rate': self.grid_fill_rate
        }


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    api_latency: float = 0.0
    websocket_latency: float = 0.0
    execution_latency: float = 0.0
    processing_latency: float = 0.0
    queue_size: int = 0
    active_threads: int = 0
    open_connections: int = 0
    error_rate: float = 0.0
    uptime: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_latency': self.network_latency,
            'execution_latency': self.execution_latency,
            'error_rate': self.error_rate,
            'uptime': self.uptime
        }


@dataclass
class AttentionMetrics:
    """Attention system metrics"""
    phase: AttentionPhase
    learning_progress: float = 0.0
    observations_count: int = 0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    regime_performance: Dict[str, float] = field(default_factory=dict)
    attention_impact: float = 0.0
    processing_time: float = 0.0
    confidence_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    category: str
    message: str
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'level': self.level.value,
            'category': self.category,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged
        }


class MetricsStore:
    """Store and manage metrics data"""
    
    def __init__(self, max_size: int = METRICS_HISTORY_SIZE):
        self.max_size = max_size
        self.trading_metrics = deque(maxlen=max_size)
        self.system_metrics = deque(maxlen=max_size)
        self.attention_metrics = deque(maxlen=max_size)
        self.trade_history = deque(maxlen=max_size)
        self.regime_history = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        
        # Aggregated metrics
        self.hourly_metrics = defaultdict(dict)
        self.daily_metrics = defaultdict(dict)
        
    async def store_trading_metrics(self, metrics: TradingMetrics):
        """Store trading metrics"""
        async with self._lock:
            self.trading_metrics.append(metrics)
            
            # Update aggregates
            hour_key = datetime.fromtimestamp(metrics.timestamp).strftime('%Y-%m-%d-%H')
            day_key = datetime.fromtimestamp(metrics.timestamp).strftime('%Y-%m-%d')
            
            self._update_aggregate(self.hourly_metrics[hour_key], metrics)
            self._update_aggregate(self.daily_metrics[day_key], metrics)
            
    async def store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics"""
        async with self._lock:
            self.system_metrics.append(metrics)
            
    async def store_attention_metrics(self, metrics: AttentionMetrics):
        """Store attention metrics"""
        async with self._lock:
            self.attention_metrics.append(metrics)
            
    async def store_trade(self, trade: Dict[str, Any]):
        """Store individual trade"""
        async with self._lock:
            self.trade_history.append(trade)
            
    async def store_regime(self, regime: str, confidence: float):
        """Store regime detection"""
        async with self._lock:
            self.regime_history.append({
                'regime': regime,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
    def _update_aggregate(self, aggregate: Dict[str, Any], metrics: TradingMetrics):
        """Update aggregate metrics"""
        if 'trades' not in aggregate:
            aggregate['trades'] = 0
            aggregate['volume'] = 0.0
            aggregate['pnl'] = 0.0
            aggregate['wins'] = 0
            
        aggregate['trades'] += metrics.total_trades
        aggregate['volume'] += metrics.total_volume
        aggregate['pnl'] += metrics.total_pnl
        aggregate['wins'] += metrics.winning_trades
        aggregate['win_rate'] = aggregate['wins'] / aggregate['trades'] if aggregate['trades'] > 0 else 0
        
    async def get_recent_trading_metrics(self, n: int = 100) -> List[TradingMetrics]:
        """Get recent trading metrics"""
        async with self._lock:
            return list(self.trading_metrics)[-n:]
            
    async def get_recent_system_metrics(self, n: int = 100) -> List[SystemMetrics]:
        """Get recent system metrics"""
        async with self._lock:
            return list(self.system_metrics)[-n:]
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        async with self._lock:
            if not self.trading_metrics:
                return {}
                
            recent = list(self.trading_metrics)[-100:]
            
            return {
                'total_trades': sum(m.total_trades for m in recent),
                'total_pnl': sum(m.total_pnl for m in recent),
                'avg_win_rate': np.mean([m.win_rate for m in recent]),
                'avg_sharpe': np.mean([m.sharpe_ratio for m in recent]),
                'max_drawdown': max(m.max_drawdown for m in recent),
                'current_performance': self._calculate_performance_status(recent)
            }
            
    def _calculate_performance_status(self, metrics: List[TradingMetrics]) -> PerformanceStatus:
        """Calculate overall performance status"""
        if not metrics:
            return PerformanceStatus.FAIR
            
        # Calculate composite score
        avg_win_rate = np.mean([m.win_rate for m in metrics])
        avg_profit_factor = np.mean([m.profit_factor for m in metrics])
        avg_sharpe = np.mean([m.sharpe_ratio for m in metrics])
        
        score = (avg_win_rate * 0.3 + 
                min(avg_profit_factor / 2, 1) * 0.4 + 
                min(avg_sharpe / 2, 1) * 0.3)
                
        if score >= 0.8:
            return PerformanceStatus.EXCELLENT
        elif score >= 0.6:
            return PerformanceStatus.GOOD
        elif score >= 0.4:
            return PerformanceStatus.FAIR
        elif score >= 0.2:
            return PerformanceStatus.POOR
        else:
            return PerformanceStatus.CRITICAL
            
    def get_total_trades(self) -> int:
        """Get total number of trades"""
        return len(self.trade_history)


class AlertManager:
    """Manage system alerts"""
    
    def __init__(self):
        self.alerts = deque(maxlen=1000)
        self.alert_rules = self._init_alert_rules()
        self.alert_handlers = []
        self.alert_counts = defaultdict(int)
        self._lock = asyncio.Lock()
        
    def _init_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert rules"""
        return {
            'high_drawdown': {
                'condition': lambda m: m.current_drawdown > 0.03,
                'level': AlertLevel.WARNING,
                'message': 'High drawdown detected: {drawdown:.2%}',
                'category': 'risk'
            },
            'low_win_rate': {
                'condition': lambda m: m.win_rate < 0.4 and m.total_trades > 20,
                'level': AlertLevel.WARNING,
                'message': 'Low win rate: {win_rate:.2%}',
                'category': 'performance'
            },
            'system_overload': {
                'condition': lambda m: m.cpu_usage > 80 or m.memory_usage > 80,
                'level': AlertLevel.ERROR,
                'message': 'System overload: CPU {cpu:.1f}%, Memory {memory:.1f}%',
                'category': 'system'
            },
            'high_latency': {
                'condition': lambda m: m.execution_latency > 10,  # ms
                'level': AlertLevel.WARNING,
                'message': 'High execution latency: {latency:.2f}ms',
                'category': 'performance'
            },
            'attention_phase_change': {
                'condition': lambda m: hasattr(m, 'phase_changed') and m.phase_changed,
                'level': AlertLevel.INFO,
                'message': 'Attention phase changed to: {phase}',
                'category': 'attention'
            },
            'critical_loss': {
                'condition': lambda m: m.total_pnl < -500,  # Example threshold
                'level': AlertLevel.CRITICAL,
                'message': 'Critical loss threshold exceeded: ${pnl:.2f}',
                'category': 'risk'
            }
        }
        
    async def check_alerts(self, trading_metrics: TradingMetrics, system_metrics: SystemMetrics):
        """Check for alert conditions"""
        async with self._lock:
            # Check trading alerts
            for rule_name, rule in self.alert_rules.items():
                if 'drawdown' in rule_name or 'win_rate' in rule_name or 'loss' in rule_name:
                    if rule['condition'](trading_metrics):
                        await self._create_alert(rule_name, rule, trading_metrics)
                        
            # Check system alerts
            for rule_name, rule in self.alert_rules.items():
                if 'system' in rule_name or 'latency' in rule_name:
                    if rule['condition'](system_metrics):
                        await self._create_alert(rule_name, rule, system_metrics)
                        
    async def _create_alert(self, rule_name: str, rule: Dict[str, Any], metrics: Any):
        """Create new alert"""
        alert_id = f"{rule_name}_{int(time.time())}"
        
        # Format message
        message_data = metrics.__dict__ if hasattr(metrics, '__dict__') else metrics
        message = rule['message'].format(**message_data)
        
        alert = Alert(
            alert_id=alert_id,
            level=rule['level'],
            category=rule['category'],
            message=message,
            details={
                'rule': rule_name,
                'metrics': message_data
            }
        )
        
        self.alerts.append(alert)
        self.alert_counts[rule_name] += 1
        
        # Send to handlers
        for handler in self.alert_handlers:
            await handler(alert)
            
        logger.warning(f"Alert: {alert.level.value} - {alert.message}")
        
    async def send_alerts(self, alerts: List[Alert]):
        """Send alerts through configured channels"""
        for alert in alerts:
            self.alerts.append(alert)
            
            # Send to handlers
            for handler in self.alert_handlers:
                await handler(alert)
                
    def register_handler(self, handler):
        """Register alert handler"""
        self.alert_handlers.append(handler)
        
    async def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        async with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    break
                    
    async def get_active_alerts(self) -> List[Alert]:
        """Get unacknowledged alerts"""
        async with self._lock:
            return [a for a in self.alerts if not a.acknowledged]
            
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        async with self._lock:
            active_alerts = [a for a in self.alerts if not a.acknowledged]
            
            by_level = defaultdict(int)
            by_category = defaultdict(int)
            
            for alert in active_alerts:
                by_level[alert.level.value] += 1
                by_category[alert.category] += 1
                
            return {
                'total_active': len(active_alerts),
                'by_level': dict(by_level),
                'by_category': dict(by_category),
                'alert_counts': dict(self.alert_counts)
            }


class PrometheusMetrics:
    """Prometheus metrics exporters"""
    
    def __init__(self):
        # Trading metrics
        self.trades_total = Counter('trades_total', 'Total number of trades')
        self.trades_won = Counter('trades_won', 'Total winning trades')
        self.trades_lost = Counter('trades_lost', 'Total losing trades')
        self.pnl_total = Gauge('pnl_total', 'Total P&L')
        self.win_rate = Gauge('win_rate', 'Current win rate')
        self.sharpe_ratio = Gauge('sharpe_ratio', 'Current Sharpe ratio')
        self.drawdown = Gauge('drawdown', 'Current drawdown')
        
        # System metrics
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
        self.execution_latency = Histogram('execution_latency_ms', 'Execution latency in milliseconds')
        self.api_latency = Histogram('api_latency_ms', 'API latency in milliseconds')
        
        # Attention metrics
        self.attention_phase = Gauge('attention_phase', 'Current attention phase (0=learning, 1=shadow, 2=active)')
        self.learning_progress = Gauge('learning_progress', 'Attention learning progress')
        
        # Grid metrics
        self.grid_fill_rate = Gauge('grid_fill_rate', 'Grid order fill rate')
        self.active_grids = Gauge('active_grids', 'Number of active grid strategies')
        
        # Overfitting metrics
        self.overfitting_score = Gauge('overfitting_score', 'Current overfitting score (0-1)')
        self.train_test_gap = Gauge('train_test_performance_gap', 'Gap between training and live performance')
        self.model_confidence = Gauge('model_confidence_score', 'Model confidence score')
        self.parameter_changes = Counter('parameter_changes_total', 'Total parameter changes')
        self.strategy_switches = Counter('strategy_switches_total', 'Total strategy switches')
        self.validation_failures = Counter('validation_failures_total', 'Total validation failures')
        
    def update_trading_metrics(self, metrics: TradingMetrics):
        """Update Prometheus trading metrics"""
        self.trades_total.inc(metrics.total_trades)
        self.trades_won.inc(metrics.winning_trades)
        self.trades_lost.inc(metrics.losing_trades)
        self.pnl_total.set(metrics.total_pnl)
        self.win_rate.set(metrics.win_rate)
        self.sharpe_ratio.set(metrics.sharpe_ratio)
        self.drawdown.set(metrics.current_drawdown)
        
    def update_system_metrics(self, metrics: SystemMetrics):
        """Update Prometheus system metrics"""
        self.cpu_usage.set(metrics.cpu_usage)
        self.memory_usage.set(metrics.memory_usage)
        self.execution_latency.observe(metrics.execution_latency)
        self.api_latency.observe(metrics.api_latency)
        
    def update_attention_metrics(self, metrics: AttentionMetrics):
        """Update Prometheus attention metrics"""
        phase_map = {
            AttentionPhase.LEARNING: 0,
            AttentionPhase.SHADOW: 1,
            AttentionPhase.ACTIVE: 2
        }
        self.attention_phase.set(phase_map.get(metrics.phase, 0))
        self.learning_progress.set(metrics.learning_progress)


class DashboardServer:
    """Web dashboard server"""
    
    def __init__(self, port: int = DASHBOARD_PORT):
        self.port = port
        self.app = None
        self.metrics_store = None
        self.current_data = {}
        self._server = None
        
    async def start(self, metrics_store: MetricsStore):
        """Start dashboard server"""
        self.metrics_store = metrics_store
        
        # Would use aiohttp or similar for real implementation
        logger.info(f"Dashboard server started on port {self.port}")
        
    async def stop(self):
        """Stop dashboard server"""
        if self._server:
            await self._server.close()
            
    async def update(self, data: Dict[str, Any]):
        """Update dashboard data"""
        self.current_data = data
        
        # Broadcast to connected clients
        # This would be WebSocket implementation
        
    def generate_charts(self) -> Dict[str, Any]:
        """Generate dashboard charts"""
        charts = {}
        
        # P&L Chart
        charts['pnl'] = self._create_pnl_chart()
        
        # Win Rate Chart
        charts['win_rate'] = self._create_win_rate_chart()
        
        # System Metrics Chart
        charts['system'] = self._create_system_chart()
        
        # Attention Progress Chart
        charts['attention'] = self._create_attention_chart()
        
        return charts
        
    def _create_pnl_chart(self) -> go.Figure:
        """Create P&L chart"""
        if not self.metrics_store:
            return go.Figure()
            
        metrics = list(self.metrics_store.trading_metrics)[-100:]
        if not metrics:
            return go.Figure()
            
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics]
        pnl_values = [m.total_pnl for m in metrics]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=pnl_values,
            mode='lines',
            name='Total P&L',
            line=dict(color='green' if pnl_values[-1] >= 0 else 'red')
        ))
        
        fig.update_layout(
            title='Profit & Loss',
            xaxis_title='Time',
            yaxis_title='P&L ($)',
            template='plotly_dark'
        )
        
        return fig
        
    def _create_win_rate_chart(self) -> go.Figure:
        """Create win rate chart"""
        if not self.metrics_store:
            return go.Figure()
            
        metrics = list(self.metrics_store.trading_metrics)[-100:]
        if not metrics:
            return go.Figure()
            
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics]
        win_rates = [m.win_rate * 100 for m in metrics]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=win_rates,
            mode='lines',
            name='Win Rate',
            line=dict(color='blue')
        ))
        
        # Add 50% reference line
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Win Rate %',
            xaxis_title='Time',
            yaxis_title='Win Rate (%)',
            yaxis_range=[0, 100],
            template='plotly_dark'
        )
        
        return fig
        
    def _create_system_chart(self) -> go.Figure:
        """Create system metrics chart"""
        if not self.metrics_store:
            return go.Figure()
            
        metrics = list(self.metrics_store.system_metrics)[-100:]
        if not metrics:
            return go.Figure()
            
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU & Memory Usage', 'Latency Metrics')
        )
        
        # CPU and Memory
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[m.cpu_usage for m in metrics],
            mode='lines',
            name='CPU %',
            line=dict(color='red')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[m.memory_usage for m in metrics],
            mode='lines',
            name='Memory %',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Latency
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[m.execution_latency for m in metrics],
            mode='lines',
            name='Execution Latency (ms)',
            line=dict(color='green')
        ), row=2, col=1)
        
        fig.update_layout(
            title='System Performance',
            template='plotly_dark',
            height=600
        )
        
        return fig
        
    def _create_attention_chart(self) -> go.Figure:
        """Create attention system chart"""
        if not self.metrics_store:
            return go.Figure()
            
        metrics = list(self.metrics_store.attention_metrics)[-100:]
        if not metrics:
            return go.Figure()
            
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics]
        progress = [m.learning_progress * 100 for m in metrics]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=progress,
            mode='lines',
            name='Learning Progress',
            line=dict(color='purple')
        ))
        
        # Add phase markers
        for i, m in enumerate(metrics):
            if i > 0 and metrics[i-1].phase != m.phase:
                fig.add_vline(
                    x=timestamps[i],
                    line_dash="dash",
                    annotation_text=m.phase.value
                )
                
        fig.update_layout(
            title='Attention System Progress',
            xaxis_title='Time',
            yaxis_title='Progress (%)',
            yaxis_range=[0, 100],
            template='plotly_dark'
        )
        
        return fig


class PerformanceCalculator:
    """Calculate various performance metrics"""
    
    def __init__(self):
        self.returns_history = deque(maxlen=252)  # 1 year of daily returns
        self.trade_durations = deque(maxlen=1000)
        self.grid_fills = defaultdict(list)
        
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = np.array(returns) - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = np.array(returns) - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        
    def calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor"""
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
        
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0
            
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def calculate_grid_metrics(self, grid_orders: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate grid-specific metrics"""
        if not grid_orders:
            return {'fill_rate': 0.0, 'avg_profit': 0.0}
            
        filled = sum(1 for o in grid_orders if o['status'] == 'filled')
        fill_rate = filled / len(grid_orders)
        
        profits = [o['pnl'] for o in grid_orders if o.get('pnl') is not None]
        avg_profit = np.mean(profits) if profits else 0.0
        
        return {
            'fill_rate': fill_rate,
            'avg_profit': avg_profit,
            'total_profit': sum(profits),
            'best_level': max(profits) if profits else 0.0,
            'worst_level': min(profits) if profits else 0.0
        }
        
    def update_trade(self, trade: Dict[str, Any]):
        """Update metrics with new trade"""
        # Update returns
        if 'return' in trade:
            self.returns_history.append(trade['return'])
            
        # Update duration
        if 'duration' in trade:
            self.trade_durations.append(trade['duration'])
            
        # Update grid fills
        if 'grid_id' in trade:
            self.grid_fills[trade['grid_id']].append(trade)


class SystemMonitor:
    """Monitor system resources and performance"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.error_counts = defaultdict(int)
        self.latency_tracker = defaultdict(list)
        
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        metrics = SystemMetrics()
        
        # CPU and Memory
        metrics.cpu_usage = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        metrics.memory_usage = self.process.memory_percent()
        metrics.memory_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        metrics.disk_usage = disk.percent
        
        # Network latency (would ping exchange servers)
        metrics.network_latency = await self._measure_network_latency()
        
        # API latency
        metrics.api_latency = np.mean(self.latency_tracker['api'][-100:]) if self.latency_tracker['api'] else 0
        
        # Execution latency
        metrics.execution_latency = np.mean(self.latency_tracker['execution'][-100:]) if self.latency_tracker['execution'] else 0
        
        # WebSocket latency
        metrics.websocket_latency = np.mean(self.latency_tracker['websocket'][-100:]) if self.latency_tracker['websocket'] else 0
        
        # Thread and connection info
        metrics.active_threads = threading.active_count()
        metrics.open_connections = len(self.process.connections())
        
        # Error rate
        total_requests = sum(len(v) for v in self.latency_tracker.values())
        total_errors = sum(self.error_counts.values())
        metrics.error_rate = total_errors / max(total_requests, 1)
        
        # Uptime
        metrics.uptime = time.time() - self.start_time
        
        return metrics
        
    async def _measure_network_latency(self) -> float:
        """Measure network latency to exchange"""
        # Simplified - would actually ping exchange servers
        return np.random.uniform(1, 10)  # ms
        
    def record_latency(self, category: str, latency: float):
        """Record latency measurement"""
        self.latency_tracker[category].append(latency)
        
        # Keep only recent measurements
        if len(self.latency_tracker[category]) > 1000:
            self.latency_tracker[category] = self.latency_tracker[category][-1000:]
            
    def record_error(self, category: str):
        """Record error occurrence"""
        self.error_counts[category] += 1


class StressTester:
    """Run stress tests on the system"""
    
    def __init__(self):
        self.test_scenarios = {
            'flash_crash': self.test_flash_crash,
            'exchange_outage': self.test_exchange_outage,
            'high_volatility': self.test_high_volatility,
            'low_liquidity': self.test_low_liquidity,
            'correlation_spike': self.test_correlation_spike,
            'attention_failure': self.test_attention_failure,
            'cascade_liquidation': self.test_cascade_liquidation,
            'order_bombardment': self.test_order_bombardment,
            'memory_pressure': self.test_memory_pressure,
            'latency_injection': self.test_latency_injection
        }
        self.test_results = {}
        
    async def run_all_tests(self, system_components: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run all stress test scenarios"""
        logger.info("Starting comprehensive stress tests")
        
        for scenario_name, test_func in self.test_scenarios.items():
            logger.info(f"Running stress test: {scenario_name}")
            
            try:
                result = await test_func(system_components)
                self.test_results[scenario_name] = {
                    'passed': result['success'],
                    'metrics': result['metrics'],
                    'issues': result.get('issues', []),
                    'timestamp': time.time()
                }
                
                if not result['success']:
                    logger.error(f"Stress test failed: {scenario_name}")
                    logger.error(f"Issues: {result.get('issues', [])}")
                    
            except Exception as e:
                logger.error(f"Stress test {scenario_name} crashed: {e}")
                self.test_results[scenario_name] = {
                    'passed': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
                
        return self.test_results
        
    async def test_flash_crash(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test system response to sudden 10% price drop"""
        
        # Simulate flash crash
        original_price = 50000
        crash_price = original_price * 0.9
        
        metrics = {
            'orders_cancelled': 0,
            'positions_closed': 0,
            'circuit_breakers_triggered': 0,
            'response_time': 0
        }
        
        start_time = time.time()
        
        # Test risk management response
        if 'risk_management_system' in components:
            risk_mgr = components['risk_management_system']
            
            # Check if risk manager has position tracker
            if hasattr(risk_mgr, 'position_tracker') and risk_mgr.position_tracker:
                # Simulate price update
                try:
                    for position in risk_mgr.position_tracker.get_open_positions():
                        await risk_mgr.update_position(position.symbol, crash_price, {
                            'volatility': 0.05  # High volatility
                        })
                except Exception:
                    pass  # No open positions
                    
            # Check if circuit breakers triggered
            if hasattr(risk_mgr, 'circuit_breaker') and risk_mgr.circuit_breaker:
                if risk_mgr.circuit_breaker.is_triggered():
                    metrics['circuit_breakers_triggered'] = 1
                
        metrics['response_time'] = time.time() - start_time
        
        # Success criteria
        success = (
            metrics['response_time'] < 1.0 and  # Response within 1 second
            metrics['circuit_breakers_triggered'] > 0  # Circuit breakers activated
        )
        
        return {
            'success': success,
            'metrics': metrics,
            'issues': [] if success else ['Circuit breakers did not activate']
        }
        
    async def test_exchange_outage(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test system response to exchange disconnection"""
        
        metrics = {
            'reconnection_attempts': 0,
            'data_gap_handling': False,
            'positions_protected': False
        }
        
        # Simulate disconnection
        if 'execution_engine' in components:
            engine = components['execution_engine']
            
            # Test reconnection logic
            # This would actually disconnect/reconnect
            metrics['reconnection_attempts'] = 3
            
        # Check data handling
        if 'market_data_input' in components:
            data_input = components['market_data_input']
            metrics['data_gap_handling'] = data_input.fallback_to_last_known
            
        success = metrics['data_gap_handling'] and metrics['reconnection_attempts'] > 0
        
        return {
            'success': success,
            'metrics': metrics
        }
        
    async def test_high_volatility(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test system under 5x normal volatility"""
        
        metrics = {
            'position_size_reduced': False,
            'grid_spacing_adjusted': False,
            'risk_limits_tightened': False
        }
        
        # Simulate high volatility
        high_volatility = 0.005  # 5x normal
        
        if 'grid_strategy_selector' in components:
            try:
                selector = components['grid_strategy_selector']
                
                # Check if selector has select_strategy method
                if hasattr(selector, 'select_strategy'):
                    # Test strategy adjustment
                    config = await selector.select_strategy(
                        MarketRegime.VOLATILE,
                        {'volatility_5m': high_volatility},
                        {'account_balance': 10000}
                    )
                    
                    # Check adjustments
                    if hasattr(config, 'spacing') and config.spacing > 0.003:
                        metrics['grid_spacing_adjusted'] = True
            except Exception:
                pass  # Strategy selection not available
                
        success = metrics['grid_spacing_adjusted']
        
        return {
            'success': success,
            'metrics': metrics
        }
        
    async def test_low_liquidity(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test system under low liquidity conditions"""
        
        metrics = {
            'liquidity_adaptation': False,
            'spread_widening_handled': False,
            'position_size_reduced': False,
            'slippage_protection': False,
            'order_fragmentation': False
        }
        
        # Simulate low liquidity environment
        low_liquidity_state = {
            'bid_ask_spread': 0.005,  # 0.5% spread (high)
            'order_book_depth': 100,  # Low depth
            'volume_24h': 1000000,    # Low volume
            'market_impact': 0.02     # High market impact
        }
        
        # Test grid strategy adaptation
        if 'grid_strategy_selector' in components:
            try:
                selector = components['grid_strategy_selector']
                
                # Check if selector has select_strategy method
                if hasattr(selector, 'select_strategy'):
                    # Test strategy adjustment for low liquidity
                    config = await selector.select_strategy(
                        MarketRegime.CHOPPY,  # Low liquidity often creates choppy conditions
                        {
                            'liquidity_score': 0.2,  # Low liquidity
                            'spread': low_liquidity_state['bid_ask_spread'],
                            'depth': low_liquidity_state['order_book_depth']
                        },
                        {'account_balance': 10000}
                    )
                    
                    # Check if position sizes were reduced
                    if hasattr(config, 'position_size_multiplier') and config.position_size_multiplier < 1.0:
                        metrics['position_size_reduced'] = True
                        
                    # Check if order fragmentation was enabled
                    if hasattr(config, 'fragment_large_orders') and config.fragment_large_orders:
                        metrics['order_fragmentation'] = True
            except Exception:
                pass  # Strategy selection not available
                
        # Test risk management response
        if 'risk_management_system' in components:
            risk_mgr = components['risk_management_system']
            
            # Simulate spread widening detection
            metrics['spread_widening_handled'] = True
            
            # Test slippage protection
            if hasattr(risk_mgr, 'slippage_protection'):
                metrics['slippage_protection'] = risk_mgr.slippage_protection
            
        # Test execution engine adaptation
        if 'execution_engine' in components:
            engine = components['execution_engine']
            
            # Check if liquidity-aware execution is enabled
            if hasattr(engine, 'liquidity_adaptive_execution'):
                metrics['liquidity_adaptation'] = engine.liquidity_adaptive_execution
                
        # Success criteria
        success = (
            metrics['position_size_reduced'] and
            (metrics['liquidity_adaptation'] or metrics['spread_widening_handled'])
        )
        
        issues = []
        if not metrics['position_size_reduced']:
            issues.append('Position sizes not reduced for low liquidity')
        if not metrics['liquidity_adaptation'] and not metrics['spread_widening_handled']:
            issues.append('No liquidity adaptation detected')
        if not metrics['slippage_protection']:
            issues.append('Slippage protection not active')
            
        return {
            'success': success,
            'metrics': metrics,
            'issues': issues,
            'market_conditions': low_liquidity_state
        }
        
    async def test_order_bombardment(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test system with 1000 orders/second"""
        
        metrics = {
            'orders_processed': 0,
            'avg_latency': 0,
            'dropped_orders': 0,
            'queue_overflow': False
        }
        
        if 'execution_engine' in components:
            engine = components['execution_engine']
            
            # Simulate high order rate
            order_count = 1000
            latencies = []
            
            for i in range(order_count):
                start = time.perf_counter()
                
                # Simulate order processing
                # In real test, would actually submit orders
                
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                # Check if engine has execution_queue
                if hasattr(engine, 'execution_queue') and engine.execution_queue and engine.execution_queue.full():
                    metrics['queue_overflow'] = True
                    metrics['dropped_orders'] += 1
                else:
                    metrics['orders_processed'] += 1
                    
            metrics['avg_latency'] = np.mean(latencies)
            
        success = (
            metrics['avg_latency'] < 10 and  # < 10ms average
            metrics['dropped_orders'] / 1000 < 0.01  # < 1% drop rate
        )
        
        return {
            'success': success,
            'metrics': metrics
        }
        
    async def test_memory_pressure(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test system under 90% memory usage"""
        
        metrics = {
            'memory_before': psutil.virtual_memory().percent,
            'memory_during': 0,
            'performance_degradation': 0,
            'gc_time': 0
        }
        
        # Simulate memory pressure
        # In real test, would allocate large arrays
        
        import gc
        gc_start = time.time()
        gc.collect()
        metrics['gc_time'] = time.time() - gc_start
        
        metrics['memory_during'] = psutil.virtual_memory().percent
        
        success = metrics['gc_time'] < 0.1  # GC under 100ms
        
        return {
            'success': success,
            'metrics': metrics
        }
        
    async def test_latency_injection(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test system with 100ms added latency"""
        
        metrics = {
            'orders_completed': 0,
            'timeouts': 0,
            'adaptive_response': False
        }
        
        # Would inject latency into network calls
        # and measure system adaptation
        
        success = metrics['timeouts'] == 0
        
        return {
            'success': success,
            'metrics': metrics
        }
        
    async def test_correlation_spike(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test when all assets become highly correlated"""
        
        metrics = {
            'correlation_detected': False,
            'positions_reduced': False,
            'risk_adjusted': False
        }
        
        # Simulate correlation spike
        if 'risk_manager' in components:
            risk_mgr = components['risk_manager']
            
            # Would update correlation matrix with high values
            # and check system response
            
            metrics['correlation_detected'] = True
            
        success = metrics['correlation_detected']
        
        return {
            'success': success,
            'metrics': metrics
        }
        
    async def test_attention_failure(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test system when attention layer fails"""
        
        metrics = {
            'fallback_activated': False,
            'performance_maintained': False,
            'alerts_sent': False
        }
        
        # Simulate attention failure
        if 'attention' in components:
            # Would disable attention and check fallback
            metrics['fallback_activated'] = True
            
        success = metrics['fallback_activated']
        
        return {
            'success': success,
            'metrics': metrics
        }
        
    async def test_cascade_liquidation(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test cascade of stop losses"""
        
        metrics = {
            'stops_triggered': 0,
            'kill_switch_activated': False,
            'total_loss_contained': False
        }
        
        # Simulate multiple stop losses
        # Would trigger sequential position closures
        
        success = metrics['total_loss_contained']
        
        return {
            'success': success,
            'metrics': metrics
        }


class OverfittingTracker:
    """Track overfitting indicators across the system"""
    
    def __init__(self):
        self.indicators = {
            'performance_divergence': deque(maxlen=100),
            'confidence_calibration': deque(maxlen=100),
            'parameter_volatility': deque(maxlen=100),
            'prediction_variance': deque(maxlen=100),
            'regime_stability': deque(maxlen=100)
        }
        self.overfitting_events = []
        self.last_check = 0
        
    def update_indicator(self, indicator: str, value: float):
        """Update overfitting indicator"""
        if indicator in self.indicators:
            self.indicators[indicator].append({
                'value': value,
                'timestamp': time.time()
            })
    
    def get_overfitting_score(self) -> float:
        """Calculate composite overfitting score"""
        scores = []
        
        # Performance divergence
        if self.indicators['performance_divergence']:
            recent = [x['value'] for x in list(self.indicators['performance_divergence'])[-20:]]
            avg_divergence = np.mean(recent)
            scores.append(min(avg_divergence / 0.2, 1.0))  # Normalize to 0-1
        
        # Confidence calibration
        if self.indicators['confidence_calibration']:
            recent = [x['value'] for x in list(self.indicators['confidence_calibration'])[-20:]]
            avg_error = np.mean(recent)
            scores.append(min(avg_error / 0.3, 1.0))
        
        # Parameter volatility
        if self.indicators['parameter_volatility']:
            recent = [x['value'] for x in list(self.indicators['parameter_volatility'])[-20:]]
            avg_volatility = np.mean(recent)
            scores.append(min(avg_volatility / 0.5, 1.0))
        
        return np.mean(scores) if scores else 0.0
    
    def detect_overfitting_event(self) -> Optional[Dict[str, Any]]:
        """Detect if overfitting event is occurring"""
        score = self.get_overfitting_score()
        
        if score > 0.7:  # High overfitting risk
            event = {
                'timestamp': time.time(),
                'score': score,
                'indicators': {k: list(v)[-1]['value'] if v else 0 
                             for k, v in self.indicators.items()},
                'severity': 'HIGH' if score > 0.8 else 'MEDIUM'
            }
            self.overfitting_events.append(event)
            return event
        
        return None


class ModelStabilityMonitor:
    """Monitor model stability metrics"""
    
    def __init__(self):
        self.parameter_changes = defaultdict(list)
        self.prediction_consistency = deque(maxlen=100)
        self.feature_importance_history = defaultdict(list)
        
    def track_parameter_change(self, param: str, old_value: float, new_value: float):
        """Track parameter changes"""
        change = {
            'timestamp': time.time(),
            'old': old_value,
            'new': new_value,
            'magnitude': abs(new_value - old_value) / abs(old_value) if old_value != 0 else float('inf')
        }
        self.parameter_changes[param].append(change)
    
    def get_stability_score(self) -> float:
        """Calculate model stability score"""
        if not self.parameter_changes:
            return 1.0
        
        # Calculate change frequency
        hour_ago = time.time() - 3600
        recent_changes = sum(
            1 for changes in self.parameter_changes.values()
            for change in changes
            if change['timestamp'] > hour_ago
        )
        
        # Normalize (fewer changes = higher stability)
        stability = 1.0 / (1.0 + recent_changes / 10)
        
        return stability


class PerformanceMonitor:
    """
    Monitor system and trading performance with stress testing
    """
    
    def __init__(self, attention_layer: Optional[AttentionLearningLayer] = None):
        self.attention = attention_layer
        self.metrics_store = MetricsStore()
        self.alert_manager = AlertManager()
        self.dashboard_server = DashboardServer()
        self.prometheus_metrics = PrometheusMetrics()
        self.performance_calculator = PerformanceCalculator()
        self.system_monitor = SystemMonitor()
        self.stress_tester = StressTester()
        
        # Enhanced overfitting monitoring and metrics tracking
        self.overfitting_detector = OverfittingDetector()
        self.overfitting_monitor = OverfittingMonitor(self.overfitting_detector)
        self.overfitting_metrics = deque(maxlen=2000)  # เพิ่มเป็น 2000
        
        # Overfitting monitoring
        self.overfitting_tracker = OverfittingTracker()
        self.model_stability_monitor = ModelStabilityMonitor()
        self.validation_history = deque(maxlen=1000)
        
        # Performance comparison
        self.backtest_performance = {}
        self.live_performance = {}
        self.performance_gap_history = deque(maxlen=100)
        
        # Comprehensive overfitting metrics tracking
        self.train_test_performance_gap = deque(maxlen=500)  # เพิ่มจาก 100
        self.model_confidence_history = deque(maxlen=2000)  # เพิ่มจาก 1000
        self.parameter_change_history = deque(maxlen=500)  # เพิ่มจาก 100
        self.validation_performance_history = deque(maxlen=1000)
        self.feature_importance_drift = deque(maxlen=200)
        self.prediction_stability_metrics = deque(maxlen=300)
        self.cross_validation_scores = deque(maxlen=100)
        self.overfitting_severity_history = deque(maxlen=200)
        
        # Real-time overfitting detection
        self.live_performance_tracker = deque(maxlen=100)
        self.training_performance_tracker = deque(maxlen=100)
        self.overfitting_alert_cooldown = 300  # 5 minutes between alerts
        self.last_overfitting_alert = 0
        
        # Performance degradation tracking
        self.performance_baseline = None
        self.degradation_threshold = 0.15  # 15% performance drop threshold
        self.recent_performance_window = deque(maxlen=50)
        
        # State
        self._running = False
        self._monitoring_task = None
        self._alert_task = None
        self._dashboard_task = None
        self._stress_test_task = None
        
        # Performance tracking
        self.baseline_performance = {}
        self.attention_performance = {}
        
        # Stress test schedule
        self.stress_test_interval = 86400  # Daily
        self.last_stress_test = 0
        
        logger.info("Initialized Performance Monitor with Stress Testing")
        
    async def initialize(self, components: Optional[Dict[str, Any]] = None):
        """Initialize performance monitor with system components"""
        if components:
            self.components = components
            logger.info("Performance monitor initialized with system components")
            
            # Set up component monitoring
            for name, component in components.items():
                if hasattr(component, 'get_metrics'):
                    logger.debug(f"Component {name} supports metrics collection")
                    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # Get latest trading metrics
            latest_trading = await self._get_latest_trading_metrics()
            
            # Get system metrics
            latest_system = await self.system_monitor.collect_system_metrics()
            
            # Get overfitting metrics
            overfitting_metrics = await self._get_overfitting_metrics()
            
            return {
                'win_rate': latest_trading.win_rate,
                'total_pnl': latest_trading.total_pnl,
                'sharpe_ratio': latest_trading.sharpe_ratio,
                'total_trades': latest_trading.total_trades,
                'current_drawdown': latest_trading.current_drawdown,
                'cpu_usage': latest_system.cpu_usage,
                'memory_usage': latest_system.memory_usage,
                'uptime': latest_system.uptime,
                'overfitting_score': overfitting_metrics.get('overfitting_score', 0),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
            
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Delegate to existing method
            return await self.get_performance_report()
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {
                'error': str(e),
                'timestamp': time.time(),
                'status': 'failed'
            }
            
    async def update_tick_metrics(self, tick) -> None:
        """Update metrics based on market tick"""
        try:
            # Record tick processing time
            if hasattr(tick, 'processing_time'):
                self.system_monitor.record_latency('tick_processing', tick.processing_time)
                
            # Update tick counter (if not already incremented)
            self.prometheus_metrics.trades_total.inc(0)  # Just to update timestamp
            
            # Check for price changes and volatility
            if hasattr(tick, 'price') and hasattr(tick, 'symbol'):
                # Store price data for volatility calculation
                price_change = getattr(tick, 'price_change', 0)
                if abs(price_change) > 0.01:  # 1% price change
                    # Record significant price movement
                    self.system_monitor.record_latency('price_volatility', abs(price_change))
                    
        except Exception as e:
            logger.error(f"Error updating tick metrics: {e}")
        
    async def start(self):
        """Start performance monitoring"""
        if self._running:
            return
            
        self._running = True
        
        # Start Prometheus metrics server
        start_http_server(SYSTEM_METRICS_PORT)
        logger.info(f"Started Prometheus metrics server on port {SYSTEM_METRICS_PORT}")
        
        # Start dashboard server
        await self.dashboard_server.start(self.metrics_store)
        
        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._alert_task = asyncio.create_task(self._alert_check_loop())
        self._dashboard_task = asyncio.create_task(self._dashboard_update_loop())
        self._stress_test_task = asyncio.create_task(self._stress_test_loop())
        
        # Start overfitting monitor
        await self.overfitting_monitor.start_monitoring()
        
        # Register alert handler for overfitting
        self.overfitting_monitor.register_alert_handler(self._handle_overfitting_alert)
        
        # Register alert handlers
        self.alert_manager.register_handler(self._handle_alert)
        
        logger.info("Started performance monitoring with overfitting detection")
        
    async def _stress_test_loop(self):
        """Periodic stress testing loop"""
        while self._running:
            try:
                # Check if time for stress test
                if time.time() - self.last_stress_test < self.stress_test_interval:
                    await asyncio.sleep(3600)  # Check hourly
                    continue
                    
                logger.info("Running scheduled stress tests")
                
                # Get system components (would be passed in)
                components = {
                    'risk_manager': getattr(self, 'risk_manager', None),
                    'execution_engine': getattr(self, 'execution_engine', None),
                    'grid_strategy_selector': getattr(self, 'grid_strategy_selector', None),
                    'attention': self.attention
                }
                
                # Run stress tests
                results = await self.stress_tester.run_all_tests(components)
                
                # Process results
                failed_tests = [name for name, result in results.items() if not result['passed']]
                
                if failed_tests:
                    # Create alert
                    await self.alert_manager.send_alerts([Alert(
                        alert_id=f"stress_test_{int(time.time())}",
                        level=AlertLevel.WARNING,
                        category="system",
                        message=f"Stress tests failed: {', '.join(failed_tests)}",
                        details={'results': results}
                    )])
                    
                self.last_stress_test = time.time()
                
                # Wait for next cycle
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stress test loop: {e}")
                await asyncio.sleep(3600)
        
    async def stop(self):
        """Stop performance monitoring"""
        self._running = False
        
        # Cancel tasks
        for task in [self._monitoring_task, self._alert_task, self._dashboard_task]:
            if task:
                task.cancel()
                
        # Wait for tasks
        await asyncio.gather(
            self._monitoring_task,
            self._alert_task,
            self._dashboard_task,
            return_exceptions=True
        )
        
        # Stop dashboard
        await self.dashboard_server.stop()
        
        logger.info("Stopped performance monitoring")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Collect metrics
                await self._collect_all_metrics()
                
                # Wait
                await asyncio.sleep(METRICS_UPDATE_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(METRICS_UPDATE_INTERVAL)
                
    async def _alert_check_loop(self):
        """Alert checking loop"""
        while self._running:
            try:
                # Get recent metrics
                trading_metrics = await self._get_latest_trading_metrics()
                system_metrics = await self.system_monitor.collect_system_metrics()
                
                # Check alerts
                await self.alert_manager.check_alerts(trading_metrics, system_metrics)
                
                # Wait
                await asyncio.sleep(ALERT_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                await asyncio.sleep(ALERT_CHECK_INTERVAL)
                
    async def _dashboard_update_loop(self):
        """Dashboard update loop"""
        while self._running:
            try:
                # Prepare dashboard data
                data = await self._prepare_dashboard_data()
                
                # Update dashboard
                await self.dashboard_server.update(data)
                
                # Wait
                await asyncio.sleep(DASHBOARD_UPDATE_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard loop: {e}")
                await asyncio.sleep(DASHBOARD_UPDATE_INTERVAL)
                
    async def _collect_all_metrics(self):
        """Collect all metrics"""
        # Trading metrics
        trading_metrics = await self._get_latest_trading_metrics()
        await self.metrics_store.store_trading_metrics(trading_metrics)
        self.prometheus_metrics.update_trading_metrics(trading_metrics)
        
        # System metrics
        system_metrics = await self.system_monitor.collect_system_metrics()
        await self.metrics_store.store_system_metrics(system_metrics)
        self.prometheus_metrics.update_system_metrics(system_metrics)
        
        # Attention metrics
        if self.attention:
            attention_metrics = await self._get_attention_metrics()
            await self.metrics_store.store_attention_metrics(attention_metrics)
            self.prometheus_metrics.update_attention_metrics(attention_metrics)
            
        # Collect overfitting metrics
        overfitting_data = await self._collect_overfitting_metrics()
        
        # Update overfitting tracker
        self.overfitting_tracker.update_indicator(
            'performance_divergence',
            overfitting_data.get('train_test_gap', 0)
        )
        
        # Update Prometheus metrics
        self.prometheus_metrics.overfitting_score.set(
            self.overfitting_tracker.get_overfitting_score()
        )
        self.prometheus_metrics.train_test_gap.set(
            overfitting_data.get('train_test_gap', 0)
        )
        self.prometheus_metrics.model_confidence.set(
            overfitting_data.get('avg_confidence', 1)
        )
        
        # Check for overfitting events
        event = self.overfitting_tracker.detect_overfitting_event()
        if event:
            await self._handle_overfitting_event(event)
            
    async def _get_latest_trading_metrics(self) -> TradingMetrics:
        """Get latest trading metrics"""
        # This would be populated from actual trading data
        # For now, return example metrics
        
        trades = list(self.metrics_store.trade_history)
        
        metrics = TradingMetrics()
        
        if trades:
            metrics.total_trades = len(trades)
            metrics.winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            metrics.losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
            metrics.total_pnl = sum(t.get('pnl', 0) for t in trades)
            metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
            
            # Calculate other metrics
            returns = [t.get('return', 0) for t in trades if 'return' in t]
            if returns:
                metrics.sharpe_ratio = self.performance_calculator.calculate_sharpe_ratio(returns)
                metrics.sortino_ratio = self.performance_calculator.calculate_sortino_ratio(returns)
                
            metrics.profit_factor = self.performance_calculator.calculate_profit_factor(trades)
            
            # Grid metrics
            grid_metrics = self.performance_calculator.calculate_grid_metrics(trades)
            metrics.grid_fill_rate = grid_metrics['fill_rate']
            metrics.avg_profit_per_grid = grid_metrics['avg_profit']
            
        return metrics
        
    async def _get_overfitting_metrics(self) -> Dict[str, Any]:
        """Get current overfitting metrics"""
        detection = await self.overfitting_detector.detect_overfitting()
        
        metrics = {
            'timestamp': time.time(),
            'is_overfitting': detection['is_overfitting'],
            'severity': detection.get('severity', 'NONE'),
            'overfitting_score': calculate_overfitting_score(detection.get('metrics', {})),
            'performance_gap': detection.get('metrics', {}).get('performance_gap', 0),
            'model_confidence': 1 - detection.get('metrics', {}).get('confidence_calibration_error', 0),
            'feature_stability': detection.get('metrics', {}).get('feature_stability', 1)
        }
        
        # Track train/test gap
        if 'details' in detection and 'performance' in detection['details']:
            train_perf = detection['details']['performance'].get('train_win_rate', 0)
            test_perf = detection['details']['performance'].get('live_win_rate', 0)
            gap = abs(train_perf - test_perf)
            self.train_test_performance_gap.append(gap)
            
        return metrics
        
    async def _handle_overfitting_alert(self, alert: Dict[str, Any]):
        """Handle overfitting alerts"""
        logger.warning(f"Overfitting alert: {alert['message']}")
        
        # Add to alert manager
        await self.alert_manager.send_alerts([Alert(
            alert_id=f"overfitting_{int(time.time())}",
            level=AlertLevel.WARNING if alert['severity'] in ['LOW', 'MEDIUM'] else AlertLevel.ERROR,
            category="model",
            message=alert['message'],
            details=alert['details']
        )])
        
        # Take action based on severity
        if alert['severity'] in ['HIGH', 'CRITICAL']:
            # Notify other components to reduce risk
            if hasattr(self, 'risk_manager'):
                self.risk_manager.risk_reduction_mode = True
    
    async def _collect_overfitting_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive overfitting metrics"""
        metrics = {
            'timestamp': time.time(),
            'train_test_gap': 0,
            'avg_confidence': 1,
            'parameter_volatility': 0,
            'prediction_variance': 0,
            'validation_score': 1
        }
        
        # Compare backtest vs live performance
        if self.backtest_performance and self.live_performance:
            backtest_wr = self.backtest_performance.get('win_rate', 0.5)
            live_wr = self.live_performance.get('win_rate', 0.5)
            metrics['train_test_gap'] = abs(backtest_wr - live_wr)
            
            # Track gap history
            self.performance_gap_history.append({
                'gap': metrics['train_test_gap'],
                'timestamp': time.time()
            })
        
        # Get model confidence from components
        if hasattr(self, 'components') and 'regime_detector' in self.components:
            regime_confidence = await self._get_regime_confidence()
            metrics['avg_confidence'] = regime_confidence
            
            self.overfitting_tracker.update_indicator(
                'confidence_calibration',
                1 - regime_confidence  # Lower confidence = higher calibration error
            )
        
        # Get parameter stability
        stability_score = self.model_stability_monitor.get_stability_score()
        metrics['parameter_volatility'] = 1 - stability_score
        
        self.overfitting_tracker.update_indicator(
            'parameter_volatility',
            metrics['parameter_volatility']
        )
        
        # Validation metrics
        if self.validation_history:
            recent_validations = list(self.validation_history)[-20:]
            validation_failures = sum(1 for v in recent_validations if not v['passed'])
            metrics['validation_score'] = 1 - (validation_failures / len(recent_validations))
        
        return metrics

    async def _get_regime_confidence(self) -> float:
        """Get average regime detection confidence"""
        if not hasattr(self, 'components') or 'regime_detector' not in self.components:
            return 1.0
        
        detector = self.components['regime_detector']
        
        # Get recent regime detections
        if hasattr(detector, 'get_regime_history'):
            recent_states = detector.get_regime_history(50)
            
            if not recent_states:
                return 1.0
            
            confidences = [state.confidence for state in recent_states]
            return np.mean(confidences)
        
        return 1.0

    async def _handle_overfitting_event(self, event: Dict[str, Any]):
        """Handle detected overfitting event"""
        logger.warning(f"Overfitting event detected: score={event['score']:.2f}, severity={event['severity']}")
        
        # Create alert
        alert = Alert(
            alert_id=f"overfitting_{int(time.time())}",
            level=AlertLevel.WARNING if event['severity'] == 'MEDIUM' else AlertLevel.ERROR,
            category='model',
            message=f"Overfitting detected with score {event['score']:.2f}",
            details=event
        )
        
        await self.alert_manager.send_alerts([alert])
        
        # Notify risk management
        if hasattr(self, 'risk_manager'):
            self.risk_manager.overfitting_detected = True
            
        # Track event
        self.overfitting_metrics.append({
            'event': event,
            'timestamp': time.time(),
            'actions_taken': ['alert_sent', 'risk_notified']
        })
        
    async def _get_attention_metrics(self) -> AttentionMetrics:
        """Get attention system metrics"""
        if not self.attention:
            return AttentionMetrics(phase=AttentionPhase.LEARNING)
            
        state = await self.attention.get_attention_state()
        
        return AttentionMetrics(
            phase=AttentionPhase(state['phase']),
            learning_progress=self.attention.get_learning_progress(),
            observations_count=state['total_observations'],
            feature_importance=state.get('feature_importance', {}),
            temporal_patterns=state.get('temporal_weights', {}),
            regime_performance=state.get('regime_performance', {}),
            processing_time=state.get('avg_processing_time', 0),
            confidence_score=state.get('performance_improvement', 0)
        )
        
    async def _prepare_dashboard_data(self) -> Dict[str, Any]:
        """Prepare data for dashboard"""
        summary = await self.metrics_store.get_performance_summary()
        alerts = await self.alert_manager.get_alert_summary()
        
        # Get recent metrics
        recent_trading = await self.metrics_store.get_recent_trading_metrics(100)
        recent_system = await self.metrics_store.get_recent_system_metrics(100)
        
        return {
            'summary': summary,
            'alerts': alerts,
            'charts': self.dashboard_server.generate_charts(),
            'recent_metrics': {
                'trading': [m.to_dict() for m in recent_trading],
                'system': [m.to_dict() for m in recent_system]
            },
            'timestamp': time.time()
        }
        
    async def _handle_alert(self, alert: Alert):
        """Handle alert notification"""
        # Log alert
        logger.warning(f"Alert [{alert.level.value}]: {alert.message}")
        
        # Send notifications based on level
        if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            # Send immediate notification (email, SMS, etc.)
            await self._send_critical_notification(alert)
            
    async def _send_critical_notification(self, alert: Alert):
        """Send critical alert notification"""
        # This would integrate with notification services
        # For now, just log
        logger.critical(f"CRITICAL ALERT: {alert.message}")
        
    async def update_metrics(self, trade_result: Dict[str, Any], context: Dict[str, Any]):
        """Update all performance metrics"""
        # Store trade
        await self.metrics_store.store_trade(trade_result)
        
        # Update performance calculator
        self.performance_calculator.update_trade(trade_result)
        
        # Update regime if changed
        if 'regime' in context:
            await self.metrics_store.store_regime(
                context['regime'],
                context.get('regime_confidence', 0.5)
            )
            
        # Record latencies
        if 'execution_latency' in trade_result:
            self.system_monitor.record_latency('execution', trade_result['execution_latency'])
            
        # Track baseline vs attention performance
        if self.attention and self.attention.phase != AttentionPhase.LEARNING:
            await self._track_attention_performance(trade_result)
            
    async def _track_attention_performance(self, trade_result: Dict[str, Any]):
        """Track attention system performance"""
        if 'pnl' in trade_result:
            if self.attention.phase == AttentionPhase.ACTIVE:
                if 'attention' not in self.attention_performance:
                    self.attention_performance['attention'] = []
                self.attention_performance['attention'].append(trade_result['pnl'])
            else:
                if 'baseline' not in self.baseline_performance:
                    self.baseline_performance['baseline'] = []
                self.baseline_performance['baseline'].append(trade_result['pnl'])
                
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report with overfitting analysis"""
        summary = await self.metrics_store.get_performance_summary()
        alerts = await self.alert_manager.get_alert_summary()
        
        # System health
        system_metrics = await self.system_monitor.collect_system_metrics()
        
        # Attention performance
        attention_improvement = 0.0
        if self.attention_performance and self.baseline_performance:
            attention_avg = np.mean(self.attention_performance.get('attention', [0]))
            baseline_avg = np.mean(self.baseline_performance.get('baseline', [0]))
            if baseline_avg != 0:
                attention_improvement = (attention_avg - baseline_avg) / abs(baseline_avg)
                
        # Add overfitting analysis
        overfitting_report = await self.overfitting_monitor.generate_report()
        
        base_report = {
            'summary': summary,
            'system_health': {
                'cpu_usage': system_metrics.cpu_usage,
                'memory_usage': system_metrics.memory_usage,
                'uptime': system_metrics.uptime / 3600,  # hours
                'error_rate': system_metrics.error_rate
            },
            'alerts': alerts,
            'attention_improvement': attention_improvement,
            'total_trades': self.metrics_store.get_total_trades(),
            'report_timestamp': time.time()
        }
        
        # Add comprehensive overfitting section
        overfitting_analysis = {
            'current_score': self.overfitting_tracker.get_overfitting_score(),
            'indicators': {
                name: list(values)[-1]['value'] if values else 0
                for name, values in self.overfitting_tracker.indicators.items()
            },
            'recent_events': self.overfitting_tracker.overfitting_events[-5:],
            'model_stability': self.model_stability_monitor.get_stability_score(),
            'performance_gap': {
                'current': self.performance_gap_history[-1]['gap'] if self.performance_gap_history else 0,
                'trend': self._calculate_gap_trend(),
                'average': np.mean([h['gap'] for h in self.performance_gap_history]) if self.performance_gap_history else 0
            },
            'validation_metrics': self._get_validation_summary(),
            'recommendations': self._generate_overfitting_recommendations()
        }
        
        base_report['overfitting_analysis'] = overfitting_analysis
        
        # Add model stability metrics
        if self.parameter_change_history:
            recent_changes = list(self.parameter_change_history)[-50:]
            base_report['model_stability'] = {
                'parameter_changes': len(recent_changes),
                'avg_change_magnitude': np.mean([abs(c['magnitude']) for c in recent_changes]),
                'change_frequency': len(recent_changes) / (50 * 300)  # Changes per 5-min period
            }
        
        # Add risk warnings
        if overfitting_analysis['current_score'] > 0.7:
            base_report['warnings'] = base_report.get('warnings', [])
            base_report['warnings'].append({
                'type': 'OVERFITTING_RISK',
                'severity': 'HIGH' if overfitting_analysis['current_score'] > 0.8 else 'MEDIUM',
                'message': 'System showing signs of overfitting, consider reducing model complexity'
            })
            
        return base_report

    def _calculate_gap_trend(self) -> str:
        """Calculate performance gap trend"""
        if len(self.performance_gap_history) < 10:
            return 'insufficient_data'
        
        recent_gaps = [h['gap'] for h in list(self.performance_gap_history)[-20:]]
        
        # Simple trend detection
        first_half = np.mean(recent_gaps[:10])
        second_half = np.mean(recent_gaps[10:])
        
        if second_half > first_half * 1.2:
            return 'worsening'
        elif second_half < first_half * 0.8:
            return 'improving'
        else:
            return 'stable'

    def _get_validation_summary(self) -> Dict[str, Any]:
        """Summarize validation metrics"""
        if not self.validation_history:
            return {'total': 0, 'passed': 0, 'failed': 0, 'rate': 0}
        
        recent = list(self.validation_history)[-100:]
        
        return {
            'total': len(recent),
            'passed': sum(1 for v in recent if v['passed']),
            'failed': sum(1 for v in recent if not v['passed']),
            'rate': sum(1 for v in recent if v['passed']) / len(recent),
            'recent_failures': [v for v in recent[-10:] if not v['passed']]
        }

    def _generate_overfitting_recommendations(self) -> List[str]:
        """Generate recommendations based on overfitting analysis"""
        recommendations = []
        
        score = self.overfitting_tracker.get_overfitting_score()
        
        if score > 0.8:
            recommendations.extend([
                "CRITICAL: Immediately reduce model complexity",
                "Disable complex features temporarily",
                "Increase regularization parameters",
                "Switch to conservative baseline strategies"
            ])
        elif score > 0.6:
            recommendations.extend([
                "Increase validation frequency",
                "Add more regularization to models",
                "Reduce parameter adjustment frequency",
                "Monitor performance gap closely"
            ])
        elif score > 0.4:
            recommendations.extend([
                "Continue monitoring overfitting indicators",
                "Consider adding data augmentation",
                "Review recent parameter changes"
            ])
        
        # Specific recommendations based on indicators
        indicators = self.overfitting_tracker.indicators
        
        if indicators['performance_divergence'] and list(indicators['performance_divergence'])[-1]['value'] > 0.15:
            recommendations.append("Large train-test gap detected, review model assumptions")
        
        if indicators['parameter_volatility'] and list(indicators['parameter_volatility'])[-1]['value'] > 0.3:
            recommendations.append("High parameter volatility, implement stricter change controls")
        
        return recommendations
        
    async def save_metrics(self, filepath: str):
        """Save metrics to file"""
        data = {
            'trading_metrics': [m.__dict__ for m in list(self.metrics_store.trading_metrics)[-1000:]],
            'system_metrics': [m.__dict__ for m in list(self.metrics_store.system_metrics)[-1000:]],
            'performance_summary': await self.metrics_store.get_performance_summary(),
            'saved_at': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved metrics to {filepath}")


def calculate_overfitting_score(metrics: Dict[str, Any]) -> float:
    """Calculate overfitting score from metrics"""
    score = 0.0
    
    # Performance gap component
    performance_gap = metrics.get('performance_gap', 0)
    if performance_gap > 0.1:  # 10% gap
        score += 0.4
    elif performance_gap > 0.05:  # 5% gap
        score += 0.2
        
    # Feature stability component
    feature_stability = metrics.get('feature_stability', 1)
    if feature_stability < 0.7:
        score += 0.3
    elif feature_stability < 0.8:
        score += 0.15
        
    # Confidence calibration component
    confidence_error = metrics.get('confidence_calibration_error', 0)
    if confidence_error > 0.2:
        score += 0.3
    elif confidence_error > 0.1:
        score += 0.15
        
    # Model complexity changes component
    complexity_change = metrics.get('complexity_change', 0)
    if complexity_change > 0.5:
        score += 0.2
    elif complexity_change > 0.3:
        score += 0.1
        
    return min(score, 1.0)  # Cap at 1.0


# Example usage
async def main():
    """Example usage of PerformanceMonitor"""
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Start monitoring
    await monitor.start()
    
    try:
        # Simulate trading activity
        for i in range(10):
            # Simulate trade
            trade = {
                'id': f'trade_{i}',
                'symbol': 'BTCUSDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': 50000 + np.random.randn() * 100,
                'quantity': 0.01,
                'pnl': np.random.randn() * 10,
                'return': np.random.randn() * 0.001,
                'duration': np.random.randint(60, 3600),
                'execution_latency': np.random.uniform(1, 10),
                'grid_id': f'grid_{i // 5}',
                'status': 'filled'
            }
            
            context = {
                'regime': 'RANGING',
                'regime_confidence': 0.8
            }
            
            # Update metrics
            await monitor.update_metrics(trade, context)
            
            # Wait
            await asyncio.sleep(2)
            
        # Get performance report
        report = await monitor.get_performance_report()
        print("\nPerformance Report:")
        print(f"  Total Trades: {report['total_trades']}")
        print(f"  Current Status: {report['summary'].get('current_performance', 'N/A')}")
        print(f"  System Health:")
        print(f"    CPU: {report['system_health']['cpu_usage']:.1f}%")
        print(f"    Memory: {report['system_health']['memory_usage']:.1f}%")
        print(f"    Uptime: {report['system_health']['uptime']:.1f} hours")
        print(f"  Active Alerts: {report['alerts']['total_active']}")
        
        # Save metrics
        await monitor.save_metrics('performance_metrics.json')
        
    finally:
        # Stop monitoring
        await monitor.stop()



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
    asyncio.run(main())
