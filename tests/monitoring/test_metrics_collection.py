"""
Metrics collection and monitoring tests for GridAttention trading system.

Tests comprehensive metrics gathering, performance monitoring, system health tracking,
and observability features for algorithmic trading operations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
import time
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from unittest.mock import Mock, patch, AsyncMock
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
import json

# Import core components
from core.metrics_collector import MetricsCollector
from core.performance_monitor import PerformanceMonitor
from core.system_health import SystemHealthMonitor
from core.trade_analytics import TradeAnalytics


class MetricType(Enum):
    """Types of metrics collected"""
    # Trading Metrics
    TRADES_EXECUTED = "trades_executed"
    ORDER_LATENCY = "order_latency"
    FILL_RATE = "fill_rate"
    SLIPPAGE = "slippage"
    PNL = "profit_and_loss"
    
    # System Metrics
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_LATENCY = "network_latency"
    DISK_IO = "disk_io"
    
    # Algorithm Metrics
    SIGNAL_ACCURACY = "signal_accuracy"
    REGIME_ACCURACY = "regime_accuracy"
    GRID_EFFICIENCY = "grid_efficiency"
    
    # Risk Metrics
    POSITION_EXPOSURE = "position_exposure"
    VAR = "value_at_risk"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    
    # Market Metrics
    MARKET_VOLATILITY = "market_volatility"
    BID_ASK_SPREAD = "bid_ask_spread"
    ORDER_BOOK_DEPTH = "order_book_depth"
    MARKET_IMPACT = "market_impact"


@dataclass
class MetricPoint:
    """Single metric data point"""
    metric_type: MetricType
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricAggregation:
    """Aggregated metric statistics"""
    metric_type: MetricType
    period: timedelta
    count: int
    sum: float
    mean: float
    min: float
    max: float
    std: float
    percentiles: Dict[int, float]  # 50, 75, 90, 95, 99
    

class TestMetricsCollection:
    """Test metrics collection functionality"""
    
    @pytest.fixture
    async def metrics_collector(self):
        """Create metrics collector instance"""
        return MetricsCollector(
            backend='prometheus',
            flush_interval_seconds=10,
            retention_days=30,
            enable_aggregation=True,
            enable_alerts=True
        )
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitor instance"""
        return PerformanceMonitor(
            sample_rate_hz=10,
            enable_profiling=True,
            track_memory=True,
            track_latency=True
        )
    
    @pytest.fixture
    def prometheus_registry(self):
        """Create Prometheus registry for testing"""
        return prometheus_client.CollectorRegistry()
    
    @pytest.mark.asyncio
    async def test_trading_metrics_collection(self, metrics_collector):
        """Test collection of trading-related metrics"""
        # Define Prometheus metrics
        trades_counter = Counter(
            'trades_total',
            'Total number of trades executed',
            ['instrument', 'side', 'strategy']
        )
        
        order_latency_histogram = Histogram(
            'order_latency_seconds',
            'Order execution latency',
            ['instrument', 'order_type'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )
        
        fill_rate_gauge = Gauge(
            'fill_rate_ratio',
            'Order fill rate',
            ['instrument', 'venue']
        )
        
        # Simulate trading activity
        trading_session = {
            'start_time': datetime.now(timezone.utc),
            'trades': []
        }
        
        # Execute trades and collect metrics
        for i in range(100):
            # Simulate trade execution
            trade = {
                'id': f'TRADE_{i}',
                'instrument': 'BTC/USDT' if i % 2 == 0 else 'ETH/USDT',
                'side': 'BUY' if i % 3 == 0 else 'SELL',
                'strategy': 'grid' if i % 4 == 0 else 'momentum',
                'order_sent': time.time(),
                'order_filled': time.time() + np.random.exponential(0.01),  # Exponential latency
                'price_ordered': Decimal('50000') + Decimal(str(np.random.randn() * 100)),
                'price_filled': Decimal('50000') + Decimal(str(np.random.randn() * 100)),
                'quantity': Decimal(str(abs(np.random.randn() * 0.1))),
                'status': 'FILLED' if np.random.random() > 0.1 else 'CANCELLED'
            }
            
            # Record metrics
            if trade['status'] == 'FILLED':
                # Trade counter
                trades_counter.labels(
                    instrument=trade['instrument'],
                    side=trade['side'],
                    strategy=trade['strategy']
                ).inc()
                
                # Order latency
                latency = trade['order_filled'] - trade['order_sent']
                order_latency_histogram.labels(
                    instrument=trade['instrument'],
                    order_type='MARKET'
                ).observe(latency)
                
                # Collect in our system
                await metrics_collector.record_metric(
                    metric_type=MetricType.ORDER_LATENCY,
                    value=latency * 1000,  # Convert to ms
                    labels={
                        'instrument': trade['instrument'],
                        'strategy': trade['strategy']
                    }
                )
                
                # Calculate slippage
                slippage = float(abs(trade['price_filled'] - trade['price_ordered']))
                await metrics_collector.record_metric(
                    metric_type=MetricType.SLIPPAGE,
                    value=slippage,
                    labels={'instrument': trade['instrument']}
                )
            
            trading_session['trades'].append(trade)
        
        # Calculate and record fill rate
        fill_rates = {}
        for instrument in ['BTC/USDT', 'ETH/USDT']:
            instrument_trades = [t for t in trading_session['trades'] 
                               if t['instrument'] == instrument]
            filled = len([t for t in instrument_trades if t['status'] == 'FILLED'])
            total = len(instrument_trades)
            
            fill_rate = filled / total if total > 0 else 0
            fill_rates[instrument] = fill_rate
            
            fill_rate_gauge.labels(
                instrument=instrument,
                venue='BINANCE'
            ).set(fill_rate)
            
            await metrics_collector.record_metric(
                metric_type=MetricType.FILL_RATE,
                value=fill_rate,
                labels={'instrument': instrument, 'venue': 'BINANCE'}
            )
        
        # Verify metrics were collected
        latency_metrics = await metrics_collector.get_metrics(
            metric_type=MetricType.ORDER_LATENCY,
            start_time=trading_session['start_time'],
            end_time=datetime.now(timezone.utc)
        )
        
        assert len(latency_metrics) > 0
        assert all(m.value > 0 for m in latency_metrics)
        
        # Check fill rates
        assert fill_rates['BTC/USDT'] > 0.8  # Expect >80% fill rate
        assert fill_rates['ETH/USDT'] > 0.8
    
    @pytest.mark.asyncio
    async def test_system_performance_metrics(self, performance_monitor):
        """Test system performance metrics collection"""
        # Start monitoring
        await performance_monitor.start_monitoring()
        
        # Simulate system load
        start_time = time.time()
        metrics_log = []
        
        for i in range(30):  # 30 seconds of monitoring
            # Collect system metrics
            system_metrics = await performance_monitor.collect_system_metrics()
            
            metrics_log.append({
                'timestamp': datetime.now(timezone.utc),
                'cpu_percent': system_metrics['cpu_percent'],
                'memory_percent': system_metrics['memory_percent'],
                'memory_mb': system_metrics['memory_mb'],
                'disk_io_read_mb': system_metrics['disk_io_read_mb'],
                'disk_io_write_mb': system_metrics['disk_io_write_mb'],
                'network_sent_mb': system_metrics['network_sent_mb'],
                'network_recv_mb': system_metrics['network_recv_mb'],
                'open_file_descriptors': system_metrics['open_file_descriptors'],
                'thread_count': system_metrics['thread_count']
            })
            
            # Simulate some CPU load
            if i % 10 == 0:
                # Intensive calculation
                _ = sum(j**2 for j in range(100000))
            
            await asyncio.sleep(1)
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        
        # Analyze collected metrics
        df = pd.DataFrame(metrics_log)
        
        # CPU metrics
        assert df['cpu_percent'].mean() > 0
        assert df['cpu_percent'].max() < 100
        
        # Memory metrics
        assert df['memory_percent'].mean() > 0
        assert df['memory_mb'].mean() > 0
        
        # Check for memory leaks
        memory_start = df['memory_mb'].iloc[:5].mean()
        memory_end = df['memory_mb'].iloc[-5:].mean()
        memory_growth = (memory_end - memory_start) / memory_start
        
        assert memory_growth < 0.1, f"Memory growth {memory_growth:.2%} exceeds 10%"
        
        # Resource usage alerts
        alerts = await performance_monitor.check_resource_alerts(
            cpu_threshold=80,
            memory_threshold=90,
            disk_io_threshold_mbps=100
        )
        
        if alerts:
            for alert in alerts:
                assert alert['severity'] in ['WARNING', 'CRITICAL']
                assert 'resource' in alert
                assert 'current_value' in alert
                assert 'threshold' in alert
    
    @pytest.mark.asyncio
    async def test_pnl_metrics(self, metrics_collector):
        """Test P&L metrics collection and calculation"""
        # Initialize P&L tracking
        pnl_tracker = {
            'realized_pnl': Decimal('0'),
            'unrealized_pnl': Decimal('0'),
            'positions': {},
            'trades': []
        }
        
        # Simulate trading with P&L
        base_time = datetime.now(timezone.utc)
        
        # Open positions
        positions = [
            {
                'instrument': 'BTC/USDT',
                'side': 'LONG',
                'quantity': Decimal('0.5'),
                'entry_price': Decimal('50000'),
                'current_price': Decimal('51000')
            },
            {
                'instrument': 'ETH/USDT',
                'side': 'SHORT',
                'quantity': Decimal('5.0'),
                'entry_price': Decimal('3000'),
                'current_price': Decimal('2950')
            }
        ]
        
        # Calculate and record P&L
        for position in positions:
            # Calculate unrealized P&L
            if position['side'] == 'LONG':
                unrealized = position['quantity'] * (position['current_price'] - position['entry_price'])
            else:
                unrealized = position['quantity'] * (position['entry_price'] - position['current_price'])
            
            await metrics_collector.record_metric(
                metric_type=MetricType.PNL,
                value=float(unrealized),
                labels={
                    'instrument': position['instrument'],
                    'type': 'unrealized',
                    'side': position['side']
                }
            )
            
            pnl_tracker['unrealized_pnl'] += unrealized
            pnl_tracker['positions'][position['instrument']] = position
        
        # Simulate closed trades
        closed_trades = [
            {
                'instrument': 'BTC/USDT',
                'side': 'LONG',
                'quantity': Decimal('0.2'),
                'entry_price': Decimal('49000'),
                'exit_price': Decimal('50000'),
                'realized_pnl': Decimal('200')  # 0.2 * (50000 - 49000)
            },
            {
                'instrument': 'ETH/USDT',
                'side': 'SHORT',
                'quantity': Decimal('3.0'),
                'entry_price': Decimal('3100'),
                'exit_price': Decimal('3000'),
                'realized_pnl': Decimal('300')  # 3.0 * (3100 - 3000)
            }
        ]
        
        for trade in closed_trades:
            await metrics_collector.record_metric(
                metric_type=MetricType.PNL,
                value=float(trade['realized_pnl']),
                labels={
                    'instrument': trade['instrument'],
                    'type': 'realized',
                    'side': trade['side']
                }
            )
            
            pnl_tracker['realized_pnl'] += trade['realized_pnl']
            pnl_tracker['trades'].append(trade)
        
        # Record total P&L
        total_pnl = pnl_tracker['realized_pnl'] + pnl_tracker['unrealized_pnl']
        await metrics_collector.record_metric(
            metric_type=MetricType.PNL,
            value=float(total_pnl),
            labels={'type': 'total'}
        )
        
        # Generate P&L report
        pnl_report = await metrics_collector.generate_pnl_report(
            start_time=base_time,
            end_time=datetime.now(timezone.utc)
        )
        
        assert pnl_report['total_pnl'] == float(total_pnl)
        assert pnl_report['realized_pnl'] == float(pnl_tracker['realized_pnl'])
        assert pnl_report['unrealized_pnl'] == float(pnl_tracker['unrealized_pnl'])
        assert 'by_instrument' in pnl_report
        assert 'by_side' in pnl_report
    
    @pytest.mark.asyncio
    async def test_risk_metrics(self, metrics_collector):
        """Test risk metrics calculation and monitoring"""
        # Portfolio for risk calculations
        portfolio = {
            'positions': [
                {'instrument': 'BTC/USDT', 'value': Decimal('100000'), 'weight': 0.4},
                {'instrument': 'ETH/USDT', 'value': Decimal('75000'), 'weight': 0.3},
                {'instrument': 'SOL/USDT', 'value': Decimal('50000'), 'weight': 0.2},
                {'instrument': 'CASH', 'value': Decimal('25000'), 'weight': 0.1}
            ],
            'total_value': Decimal('250000')
        }
        
        # Historical returns for risk calculations (daily returns)
        returns_data = np.random.normal(0.001, 0.02, (252, 4))  # 1 year of daily returns
        returns_df = pd.DataFrame(
            returns_data,
            columns=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'CASH']
        )
        returns_df['CASH'] = 0.0001  # Minimal cash returns
        
        # Calculate risk metrics
        # 1. Portfolio volatility
        portfolio_returns = returns_df @ [0.4, 0.3, 0.2, 0.1]  # Weighted returns
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        await metrics_collector.record_metric(
            metric_type=MetricType.MARKET_VOLATILITY,
            value=portfolio_volatility,
            labels={'portfolio': 'main'}
        )
        
        # 2. Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(portfolio_returns, 5) * float(portfolio['total_value'])
        
        await metrics_collector.record_metric(
            metric_type=MetricType.VAR,
            value=abs(var_95),
            labels={'confidence': '95', 'horizon': '1day'}
        )
        
        # 3. Sharpe Ratio
        risk_free_rate = 0.02  # 2% annual
        excess_returns = portfolio_returns - risk_free_rate/252
        sharpe_ratio = (excess_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
        
        await metrics_collector.record_metric(
            metric_type=MetricType.SHARPE_RATIO,
            value=sharpe_ratio,
            labels={'period': '1year'}
        )
        
        # 4. Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        await metrics_collector.record_metric(
            metric_type=MetricType.MAX_DRAWDOWN,
            value=abs(max_drawdown),
            labels={'period': '1year'}
        )
        
        # 5. Position concentration risk
        position_concentrations = {}
        for position in portfolio['positions']:
            concentration = float(position['value'] / portfolio['total_value'])
            position_concentrations[position['instrument']] = concentration
            
            await metrics_collector.record_metric(
                metric_type=MetricType.POSITION_EXPOSURE,
                value=concentration,
                labels={'instrument': position['instrument']}
            )
        
        # Verify risk metrics
        assert portfolio_volatility > 0 and portfolio_volatility < 1  # Reasonable volatility
        assert var_95 < 0  # VaR should be negative (potential loss)
        assert sharpe_ratio > -2 and sharpe_ratio < 5  # Reasonable Sharpe ratio
        assert max_drawdown < 0 and max_drawdown > -0.5  # Less than 50% drawdown
        
        # Check concentration limits
        max_concentration = max(position_concentrations.values())
        assert max_concentration < 0.5, "Single position exceeds 50% concentration"
    
    @pytest.mark.asyncio
    async def test_market_microstructure_metrics(self, metrics_collector):
        """Test market microstructure metrics collection"""
        # Simulate order book data
        order_book_snapshots = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(60):  # 60 seconds of data
            # Generate order book
            mid_price = Decimal('50000') + Decimal(str(np.random.randn() * 10))
            
            order_book = {
                'timestamp': base_time + timedelta(seconds=i),
                'instrument': 'BTC/USDT',
                'bids': [
                    {'price': mid_price - Decimal('1') * (j+1), 'quantity': Decimal(str(np.random.exponential(1)))}
                    for j in range(10)
                ],
                'asks': [
                    {'price': mid_price + Decimal('1') * (j+1), 'quantity': Decimal(str(np.random.exponential(1)))}
                    for j in range(10)
                ]
            }
            
            # Calculate metrics
            # 1. Bid-ask spread
            best_bid = order_book['bids'][0]['price']
            best_ask = order_book['asks'][0]['price']
            spread = float(best_ask - best_bid)
            spread_bps = (spread / float(mid_price)) * 10000  # Basis points
            
            await metrics_collector.record_metric(
                metric_type=MetricType.BID_ASK_SPREAD,
                value=spread_bps,
                labels={'instrument': 'BTC/USDT'}
            )
            
            # 2. Order book depth
            bid_depth = sum(float(bid['quantity']) for bid in order_book['bids'][:5])  # Top 5 levels
            ask_depth = sum(float(ask['quantity']) for ask in order_book['asks'][:5])
            total_depth = bid_depth + ask_depth
            
            await metrics_collector.record_metric(
                metric_type=MetricType.ORDER_BOOK_DEPTH,
                value=total_depth,
                labels={'instrument': 'BTC/USDT', 'levels': '5'}
            )
            
            # 3. Order book imbalance
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if total_depth > 0 else 0
            
            await metrics_collector.record_metric(
                metric_type=MetricType.ORDER_BOOK_DEPTH,
                value=imbalance,
                labels={'instrument': 'BTC/USDT', 'metric': 'imbalance'}
            )
            
            order_book_snapshots.append({
                'timestamp': order_book['timestamp'],
                'spread_bps': spread_bps,
                'total_depth': total_depth,
                'imbalance': imbalance
            })
        
        # Analyze microstructure metrics
        df = pd.DataFrame(order_book_snapshots)
        
        # Check spread statistics
        avg_spread = df['spread_bps'].mean()
        assert avg_spread > 0 and avg_spread < 50  # Reasonable spread < 50 bps
        
        # Check depth consistency
        avg_depth = df['total_depth'].mean()
        assert avg_depth > 0
        
        # Market impact estimation
        # Simulate a large order
        order_size = Decimal('5.0')  # 5 BTC
        
        market_impact = await metrics_collector.estimate_market_impact(
            instrument='BTC/USDT',
            order_size=order_size,
            side='BUY',
            order_book=order_book_snapshots[-1]  # Use latest snapshot
        )
        
        await metrics_collector.record_metric(
            metric_type=MetricType.MARKET_IMPACT,
            value=market_impact['impact_bps'],
            labels={
                'instrument': 'BTC/USDT',
                'order_size': str(order_size),
                'side': 'BUY'
            }
        )
        
        assert market_impact['impact_bps'] > 0
        assert market_impact['expected_fill_price'] > best_ask
    
    @pytest.mark.asyncio
    async def test_algorithm_performance_metrics(self, metrics_collector):
        """Test algorithm-specific performance metrics"""
        # Simulate algorithm decisions and outcomes
        algorithm_metrics = {
            'signals': [],
            'regime_predictions': [],
            'grid_performance': []
        }
        
        # 1. Signal accuracy tracking
        for i in range(100):
            signal = {
                'timestamp': datetime.now(timezone.utc) + timedelta(minutes=i),
                'predicted_direction': 'UP' if np.random.random() > 0.5 else 'DOWN',
                'confidence': np.random.uniform(0.5, 1.0),
                'actual_direction': 'UP' if np.random.random() > 0.45 else 'DOWN'  # Slight edge
            }
            
            signal['correct'] = signal['predicted_direction'] == signal['actual_direction']
            algorithm_metrics['signals'].append(signal)
            
            # Record signal accuracy
            await metrics_collector.record_metric(
                metric_type=MetricType.SIGNAL_ACCURACY,
                value=1.0 if signal['correct'] else 0.0,
                labels={
                    'confidence_bucket': f"{int(signal['confidence']*10)/10:.1f}",
                    'direction': signal['predicted_direction']
                }
            )
        
        # Calculate overall accuracy
        accuracy = sum(s['correct'] for s in algorithm_metrics['signals']) / len(algorithm_metrics['signals'])
        
        await metrics_collector.record_metric(
            metric_type=MetricType.SIGNAL_ACCURACY,
            value=accuracy,
            labels={'type': 'overall'}
        )
        
        # 2. Regime detection accuracy
        regimes = ['TRENDING', 'RANGING', 'VOLATILE']
        for i in range(50):
            regime_pred = {
                'timestamp': datetime.now(timezone.utc) + timedelta(hours=i),
                'predicted_regime': np.random.choice(regimes),
                'actual_regime': np.random.choice(regimes, p=[0.4, 0.4, 0.2]),  # Weighted
                'confidence': np.random.uniform(0.6, 1.0)
            }
            
            regime_pred['correct'] = regime_pred['predicted_regime'] == regime_pred['actual_regime']
            algorithm_metrics['regime_predictions'].append(regime_pred)
            
            await metrics_collector.record_metric(
                metric_type=MetricType.REGIME_ACCURACY,
                value=1.0 if regime_pred['correct'] else 0.0,
                labels={'regime': regime_pred['predicted_regime']}
            )
        
        # 3. Grid trading efficiency
        for i in range(30):
            grid_metrics = {
                'timestamp': datetime.now(timezone.utc) + timedelta(hours=i),
                'grid_id': f'GRID_{i}',
                'total_trades': np.random.randint(10, 50),
                'profitable_trades': np.random.randint(5, 40),
                'grid_profit': Decimal(str(np.random.uniform(-100, 500))),
                'grid_utilization': np.random.uniform(0.3, 0.9)  # How many grid levels used
            }
            
            grid_metrics['efficiency'] = (
                grid_metrics['profitable_trades'] / grid_metrics['total_trades']
                if grid_metrics['total_trades'] > 0 else 0
            )
            
            algorithm_metrics['grid_performance'].append(grid_metrics)
            
            await metrics_collector.record_metric(
                metric_type=MetricType.GRID_EFFICIENCY,
                value=grid_metrics['efficiency'],
                labels={'grid_id': grid_metrics['grid_id']}
            )
            
            # Grid utilization
            await metrics_collector.record_metric(
                metric_type=MetricType.GRID_EFFICIENCY,
                value=grid_metrics['grid_utilization'],
                labels={'metric': 'utilization', 'grid_id': grid_metrics['grid_id']}
            )
        
        # Verify algorithm metrics
        assert accuracy > 0.5  # Better than random
        
        regime_accuracy = sum(r['correct'] for r in algorithm_metrics['regime_predictions']) / len(algorithm_metrics['regime_predictions'])
        assert regime_accuracy > 0.3  # Better than random (3 regimes)
        
        avg_grid_efficiency = np.mean([g['efficiency'] for g in algorithm_metrics['grid_performance']])
        assert avg_grid_efficiency > 0.4  # At least 40% profitable trades
    
    @pytest.mark.asyncio
    async def test_metric_aggregation_and_rollups(self, metrics_collector):
        """Test metric aggregation and time-based rollups"""
        # Generate high-frequency metrics
        base_time = datetime.now(timezone.utc)
        
        # Record metrics at different frequencies
        for i in range(3600):  # 1 hour of second-level data
            timestamp = base_time + timedelta(seconds=i)
            
            # High-frequency order latency (every second)
            latency = np.random.exponential(10)  # 10ms average
            await metrics_collector.record_metric(
                metric_type=MetricType.ORDER_LATENCY,
                value=latency,
                timestamp=timestamp,
                labels={'instrument': 'BTC/USDT'}
            )
            
            # Medium frequency P&L (every 10 seconds)
            if i % 10 == 0:
                pnl = np.random.normal(0, 100)  # Mean 0, std 100
                await metrics_collector.record_metric(
                    metric_type=MetricType.PNL,
                    value=pnl,
                    timestamp=timestamp,
                    labels={'type': 'realized'}
                )
        
        # Test aggregation at different time windows
        # 1. 1-minute aggregation
        minute_agg = await metrics_collector.aggregate_metrics(
            metric_type=MetricType.ORDER_LATENCY,
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            aggregation_window=timedelta(minutes=1),
            aggregation_functions=['mean', 'min', 'max', 'p50', 'p95', 'p99']
        )
        
        assert len(minute_agg) == 60  # 60 minutes
        
        for agg in minute_agg:
            assert agg.count == 60  # 60 data points per minute
            assert agg.mean > 0
            assert agg.min < agg.mean < agg.max
            assert agg.percentiles[50] > 0  # Median
            assert agg.percentiles[95] > agg.percentiles[50]  # 95th > median
        
        # 2. 5-minute aggregation
        five_min_agg = await metrics_collector.aggregate_metrics(
            metric_type=MetricType.ORDER_LATENCY,
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            aggregation_window=timedelta(minutes=5),
            aggregation_functions=['mean', 'std', 'count']
        )
        
        assert len(five_min_agg) == 12  # 12 five-minute windows
        
        # 3. Hourly aggregation
        hourly_agg = await metrics_collector.aggregate_metrics(
            metric_type=MetricType.PNL,
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            aggregation_window=timedelta(hours=1),
            aggregation_functions=['sum', 'mean', 'std']
        )
        
        assert len(hourly_agg) == 1  # 1 hour
        assert hourly_agg[0].count == 360  # 360 ten-second intervals
        
        # Test downsampling for storage efficiency
        downsampled = await metrics_collector.downsample_metrics(
            metric_type=MetricType.ORDER_LATENCY,
            original_frequency=timedelta(seconds=1),
            target_frequency=timedelta(minutes=1),
            method='mean'
        )
        
        assert len(downsampled) < 3600  # Less than original
    
    @pytest.mark.asyncio
    async def test_real_time_dashboards(self, metrics_collector):
        """Test real-time dashboard data generation"""
        # Define dashboard configuration
        dashboard_config = {
            'trading_overview': {
                'update_frequency': timedelta(seconds=1),
                'metrics': [
                    MetricType.TRADES_EXECUTED,
                    MetricType.PNL,
                    MetricType.POSITION_EXPOSURE,
                    MetricType.ORDER_LATENCY
                ]
            },
            'risk_monitor': {
                'update_frequency': timedelta(seconds=5),
                'metrics': [
                    MetricType.VAR,
                    MetricType.MAX_DRAWDOWN,
                    MetricType.POSITION_EXPOSURE,
                    MetricType.SHARPE_RATIO
                ]
            },
            'system_health': {
                'update_frequency': timedelta(seconds=10),
                'metrics': [
                    MetricType.CPU_USAGE,
                    MetricType.MEMORY_USAGE,
                    MetricType.NETWORK_LATENCY,
                    MetricType.DISK_IO
                ]
            }
        }
        
        # Generate dashboard data
        dashboard_data = {}
        
        for dashboard_name, config in dashboard_config.items():
            data = await metrics_collector.generate_dashboard_data(
                dashboard_name=dashboard_name,
                config=config,
                time_range=timedelta(minutes=5)
            )
            
            dashboard_data[dashboard_name] = data
            
            # Verify dashboard data structure
            assert 'metrics' in data
            assert 'last_updated' in data
            assert 'time_series' in data
            
            # Check all requested metrics are present
            for metric_type in config['metrics']:
                assert metric_type.value in data['metrics']
                
                metric_data = data['metrics'][metric_type.value]
                assert 'current_value' in metric_data
                assert 'trend' in metric_data  # up/down/stable
                assert 'change_1m' in metric_data  # 1-minute change
                assert 'change_5m' in metric_data  # 5-minute change
        
        # Test real-time updates
        update_stream = await metrics_collector.create_dashboard_stream(
            dashboard_name='trading_overview',
            config=dashboard_config['trading_overview']
        )
        
        updates_received = []
        
        async def consume_updates():
            async for update in update_stream:
                updates_received.append(update)
                if len(updates_received) >= 5:
                    break
        
        # Simulate metric updates while consuming stream
        async def generate_metrics():
            for i in range(10):
                await metrics_collector.record_metric(
                    metric_type=MetricType.TRADES_EXECUTED,
                    value=1,
                    labels={'instrument': 'BTC/USDT'}
                )
                await asyncio.sleep(0.5)
        
        await asyncio.gather(
            consume_updates(),
            generate_metrics()
        )
        
        assert len(updates_received) >= 5
        
        # Verify updates contain fresh data
        for i in range(1, len(updates_received)):
            assert updates_received[i]['timestamp'] > updates_received[i-1]['timestamp']
    
    @pytest.mark.asyncio
    async def test_alerting_on_metrics(self, metrics_collector):
        """Test metric-based alerting system"""
        # Define alert rules
        alert_rules = [
            {
                'name': 'high_latency',
                'metric': MetricType.ORDER_LATENCY,
                'condition': 'greater_than',
                'threshold': 100,  # 100ms
                'duration': timedelta(minutes=1),
                'severity': 'WARNING'
            },
            {
                'name': 'low_fill_rate',
                'metric': MetricType.FILL_RATE,
                'condition': 'less_than',
                'threshold': 0.8,  # 80%
                'duration': timedelta(minutes=5),
                'severity': 'CRITICAL'
            },
            {
                'name': 'high_cpu',
                'metric': MetricType.CPU_USAGE,
                'condition': 'greater_than',
                'threshold': 80,  # 80%
                'duration': timedelta(seconds=30),
                'severity': 'WARNING'
            },
            {
                'name': 'large_drawdown',
                'metric': MetricType.MAX_DRAWDOWN,
                'condition': 'greater_than',
                'threshold': 0.1,  # 10%
                'duration': timedelta(minutes=1),
                'severity': 'CRITICAL'
            }
        ]
        
        # Configure alerts
        for rule in alert_rules:
            await metrics_collector.configure_alert(rule)
        
        # Simulate metrics that trigger alerts
        alerts_triggered = []
        
        # High latency scenario
        for i in range(70):  # 70 seconds
            await metrics_collector.record_metric(
                metric_type=MetricType.ORDER_LATENCY,
                value=150 if i > 10 else 50,  # High latency after 10 seconds
                labels={'instrument': 'BTC/USDT'}
            )
            await asyncio.sleep(0.1)
        
        # Check for triggered alerts
        alerts = await metrics_collector.get_triggered_alerts(
            since=datetime.now(timezone.utc) - timedelta(minutes=2)
        )
        
        high_latency_alerts = [a for a in alerts if a['rule_name'] == 'high_latency']
        assert len(high_latency_alerts) > 0
        
        alert = high_latency_alerts[0]
        assert alert['severity'] == 'WARNING'
        assert alert['metric_value'] > 100
        assert 'triggered_at' in alert
        assert 'description' in alert
        
        # Test alert suppression (avoid alert fatigue)
        suppression_config = {
            'suppression_window': timedelta(minutes=10),
            'max_alerts_per_rule': 3
        }
        
        await metrics_collector.configure_alert_suppression(suppression_config)
        
        # Generate many violations
        for i in range(100):
            await metrics_collector.record_metric(
                metric_type=MetricType.ORDER_LATENCY,
                value=200,
                labels={'instrument': 'BTC/USDT'}
            )
        
        # Check suppression worked
        recent_alerts = await metrics_collector.get_triggered_alerts(
            since=datetime.now(timezone.utc) - timedelta(minutes=1)
        )
        
        latency_alerts_count = len([a for a in recent_alerts if a['rule_name'] == 'high_latency'])
        assert latency_alerts_count <= suppression_config['max_alerts_per_rule']
    
    @pytest.mark.asyncio
    async def test_metric_export_formats(self, metrics_collector):
        """Test exporting metrics in various formats"""
        # Generate sample metrics
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        # Record various metrics
        for i in range(100):
            timestamp = start_time + timedelta(minutes=i*0.6)
            
            await metrics_collector.record_metric(
                metric_type=MetricType.TRADES_EXECUTED,
                value=np.random.randint(1, 10),
                timestamp=timestamp,
                labels={'strategy': 'grid'}
            )
            
            await metrics_collector.record_metric(
                metric_type=MetricType.PNL,
                value=np.random.normal(0, 100),
                timestamp=timestamp,
                labels={'type': 'realized'}
            )
        
        # Test different export formats
        # 1. Prometheus format
        prometheus_export = await metrics_collector.export_metrics(
            format='prometheus',
            start_time=start_time,
            end_time=end_time
        )
        
        assert '# HELP' in prometheus_export
        assert '# TYPE' in prometheus_export
        assert 'trades_executed{strategy="grid"}' in prometheus_export
        
        # 2. JSON format
        json_export = await metrics_collector.export_metrics(
            format='json',
            start_time=start_time,
            end_time=end_time
        )
        
        json_data = json.loads(json_export)
        assert 'metrics' in json_data
        assert 'metadata' in json_data
        assert len(json_data['metrics']) > 0
        
        # 3. CSV format
        csv_export = await metrics_collector.export_metrics(
            format='csv',
            start_time=start_time,
            end_time=end_time,
            metric_types=[MetricType.TRADES_EXECUTED, MetricType.PNL]
        )
        
        assert 'timestamp,metric_type,value,labels' in csv_export
        lines = csv_export.strip().split('\n')
        assert len(lines) > 1  # Header + data
        
        # 4. Time-series database format (InfluxDB line protocol)
        influx_export = await metrics_collector.export_metrics(
            format='influxdb',
            start_time=start_time,
            end_time=end_time
        )
        
        assert 'trades_executed,strategy=grid value=' in influx_export
        
        # 5. Binary format for efficiency
        binary_export = await metrics_collector.export_metrics(
            format='binary',
            start_time=start_time,
            end_time=end_time,
            compression='gzip'
        )
        
        assert len(binary_export) > 0
        assert isinstance(binary_export, bytes)


class TestMetricsIntegration:
    """Test metrics integration with other systems"""
    
    @pytest.mark.asyncio
    async def test_prometheus_integration(self, metrics_collector, prometheus_registry):
        """Test Prometheus metrics integration"""
        # Define custom metrics
        custom_metrics = {
            'grid_trades_total': Counter(
                'grid_trades_total',
                'Total grid trades executed',
                ['instrument', 'direction'],
                registry=prometheus_registry
            ),
            'active_grids': Gauge(
                'active_grids',
                'Number of active grid strategies',
                ['instrument'],
                registry=prometheus_registry
            ),
            'order_execution_duration': Histogram(
                'order_execution_duration_seconds',
                'Order execution duration',
                ['instrument', 'order_type'],
                buckets=(0.001, 0.01, 0.1, 1.0, 10.0),
                registry=prometheus_registry
            ),
            'strategy_profit': Summary(
                'strategy_profit_dollars',
                'Strategy profit summary',
                ['strategy_type'],
                registry=prometheus_registry
            )
        }
        
        # Simulate metric updates
        # Grid trades
        for _ in range(50):
            custom_metrics['grid_trades_total'].labels(
                instrument='BTC/USDT',
                direction='BUY'
            ).inc()
        
        # Active grids
        custom_metrics['active_grids'].labels(instrument='BTC/USDT').set(5)
        custom_metrics['active_grids'].labels(instrument='ETH/USDT').set(3)
        
        # Order execution times
        for _ in range(100):
            duration = np.random.exponential(0.05)  # 50ms average
            custom_metrics['order_execution_duration'].labels(
                instrument='BTC/USDT',
                order_type='LIMIT'
            ).observe(duration)
        
        # Strategy profits
        for _ in range(20):
            profit = np.random.normal(100, 50)
            custom_metrics['strategy_profit'].labels(
                strategy_type='grid'
            ).observe(profit)
        
        # Generate Prometheus metrics output
        from prometheus_client import generate_latest
        metrics_output = generate_latest(prometheus_registry).decode('utf-8')
        
        # Verify metrics are present
        assert 'grid_trades_total' in metrics_output
        assert 'active_grids' in metrics_output
        assert 'order_execution_duration' in metrics_output
        assert 'strategy_profit' in metrics_output
        
        # Verify metric values
        assert 'grid_trades_total{direction="BUY",instrument="BTC/USDT"} 50.0' in metrics_output
        assert 'active_grids{instrument="BTC/USDT"} 5.0' in metrics_output
    
    @pytest.mark.asyncio
    async def test_grafana_dashboard_compatibility(self, metrics_collector):
        """Test metrics format compatibility with Grafana dashboards"""
        # Generate metrics in Grafana-compatible format
        grafana_query_response = await metrics_collector.query_metrics_for_grafana(
            query={
                'metric': 'trades_executed',
                'aggregation': 'sum',
                'group_by': ['instrument'],
                'interval': '1m',
                'from': 'now-1h',
                'to': 'now'
            }
        )
        
        # Verify Grafana response format
        assert 'data' in grafana_query_response
        assert 'result' in grafana_query_response['data']
        
        for series in grafana_query_response['data']['result']:
            assert 'metric' in series
            assert 'values' in series
            assert isinstance(series['values'], list)
            
            # Each value should be [timestamp, value]
            for value in series['values']:
                assert len(value) == 2
                assert isinstance(value[0], (int, float))  # Timestamp
                assert isinstance(value[1], (int, float, str))  # Value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])