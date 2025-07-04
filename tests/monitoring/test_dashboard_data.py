"""
Dashboard data tests for GridAttention trading system.

Tests real-time dashboard data generation, aggregation, visualization preparation,
and WebSocket streaming for trading system monitoring dashboards.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import websockets
from unittest.mock import Mock, patch, AsyncMock
import aiohttp

# Import core components
from core.dashboard_service import DashboardService
from core.data_aggregator import DataAggregator
from core.websocket_manager import WebSocketManager
from core.visualization_engine import VisualizationEngine


class DashboardType(Enum):
    """Types of dashboards in the system"""
    TRADING_OVERVIEW = "trading_overview"
    RISK_MONITOR = "risk_monitor"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    SYSTEM_HEALTH = "system_health"
    MARKET_ANALYSIS = "market_analysis"
    POSITION_TRACKER = "position_tracker"
    ORDER_FLOW = "order_flow"
    COMPLIANCE_MONITOR = "compliance_monitor"


class WidgetType(Enum):
    """Types of dashboard widgets"""
    LINE_CHART = "line_chart"
    CANDLESTICK_CHART = "candlestick_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    TABLE = "table"
    METRIC_CARD = "metric_card"
    ALERT_FEED = "alert_feed"
    ORDER_BOOK = "order_book"


class UpdateFrequency(Enum):
    """Dashboard update frequencies"""
    REAL_TIME = 0.1  # 100ms
    FAST = 1.0       # 1 second
    NORMAL = 5.0     # 5 seconds
    SLOW = 30.0      # 30 seconds
    MINUTE = 60.0    # 1 minute


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    dashboard_id: str
    dashboard_type: DashboardType
    name: str
    description: str
    layout: Dict[str, Any]
    widgets: List['WidgetConfig']
    update_frequency: UpdateFrequency
    permissions: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    theme: str = 'dark'


@dataclass
class WidgetConfig:
    """Widget configuration"""
    widget_id: str
    widget_type: WidgetType
    title: str
    data_source: str
    metrics: List[str]
    update_frequency: UpdateFrequency
    position: Dict[str, int]  # x, y, width, height
    options: Dict[str, Any] = field(default_factory=dict)
    thresholds: Optional[Dict[str, float]] = None


@dataclass
class DashboardData:
    """Dashboard data packet"""
    dashboard_id: str
    timestamp: datetime
    widgets: Dict[str, 'WidgetData']
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WidgetData:
    """Widget data packet"""
    widget_id: str
    timestamp: datetime
    data: Any  # Can be various types depending on widget
    format: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestDashboardData:
    """Test dashboard data functionality"""
    
    @pytest.fixture
    async def dashboard_service(self):
        """Create dashboard service instance"""
        return DashboardService(
            enable_caching=True,
            cache_ttl_seconds=60,
            enable_compression=True,
            max_connections_per_dashboard=100
        )
    
    @pytest.fixture
    async def data_aggregator(self):
        """Create data aggregator instance"""
        return DataAggregator(
            aggregation_windows=[1, 5, 15, 60, 300],  # seconds
            enable_downsampling=True,
            retention_periods={
                '1s': timedelta(hours=1),
                '1m': timedelta(days=1),
                '5m': timedelta(days=7),
                '1h': timedelta(days=30)
            }
        )
    
    @pytest.fixture
    async def websocket_manager(self):
        """Create WebSocket manager"""
        return WebSocketManager(
            ping_interval=30,
            ping_timeout=10,
            max_message_size=1024 * 1024,  # 1MB
            enable_compression=True
        )
    
    @pytest.fixture
    def sample_dashboard_configs(self) -> List[DashboardConfig]:
        """Generate sample dashboard configurations"""
        configs = []
        
        # Trading Overview Dashboard
        trading_widgets = [
            WidgetConfig(
                widget_id='pnl_chart',
                widget_type=WidgetType.LINE_CHART,
                title='P&L Over Time',
                data_source='trading_metrics',
                metrics=['realized_pnl', 'unrealized_pnl', 'total_pnl'],
                update_frequency=UpdateFrequency.FAST,
                position={'x': 0, 'y': 0, 'width': 6, 'height': 4},
                options={
                    'show_grid': True,
                    'animation': True,
                    'colors': ['#00ff00', '#ffff00', '#0080ff']
                }
            ),
            WidgetConfig(
                widget_id='position_table',
                widget_type=WidgetType.TABLE,
                title='Open Positions',
                data_source='position_manager',
                metrics=['positions'],
                update_frequency=UpdateFrequency.FAST,
                position={'x': 6, 'y': 0, 'width': 6, 'height': 4},
                options={
                    'sortable': True,
                    'show_totals': True,
                    'highlight_pnl': True
                }
            ),
            WidgetConfig(
                widget_id='trade_volume',
                widget_type=WidgetType.METRIC_CARD,
                title='24h Trade Volume',
                data_source='trading_metrics',
                metrics=['volume_24h'],
                update_frequency=UpdateFrequency.NORMAL,
                position={'x': 0, 'y': 4, 'width': 3, 'height': 2},
                options={'format': 'currency', 'trend': True}
            ),
            WidgetConfig(
                widget_id='win_rate',
                widget_type=WidgetType.GAUGE,
                title='Win Rate',
                data_source='performance_metrics',
                metrics=['win_rate'],
                update_frequency=UpdateFrequency.SLOW,
                position={'x': 3, 'y': 4, 'width': 3, 'height': 2},
                options={'min': 0, 'max': 100, 'units': '%'},
                thresholds={'danger': 40, 'warning': 50, 'success': 60}
            )
        ]
        
        configs.append(DashboardConfig(
            dashboard_id='trading_overview',
            dashboard_type=DashboardType.TRADING_OVERVIEW,
            name='Trading Overview',
            description='Real-time trading performance and positions',
            layout={'columns': 12, 'row_height': 50},
            widgets=trading_widgets,
            update_frequency=UpdateFrequency.FAST,
            permissions=['trading', 'view_positions']
        ))
        
        # Risk Monitor Dashboard
        risk_widgets = [
            WidgetConfig(
                widget_id='var_chart',
                widget_type=WidgetType.LINE_CHART,
                title='Value at Risk',
                data_source='risk_metrics',
                metrics=['var_95', 'var_99', 'portfolio_value'],
                update_frequency=UpdateFrequency.NORMAL,
                position={'x': 0, 'y': 0, 'width': 8, 'height': 4},
                options={'show_limits': True, 'fill_area': True}
            ),
            WidgetConfig(
                widget_id='exposure_heatmap',
                widget_type=WidgetType.HEATMAP,
                title='Position Exposure Heatmap',
                data_source='risk_metrics',
                metrics=['exposure_matrix'],
                update_frequency=UpdateFrequency.SLOW,
                position={'x': 8, 'y': 0, 'width': 4, 'height': 4},
                options={'color_scheme': 'RdYlGn', 'show_values': True}
            ),
            WidgetConfig(
                widget_id='risk_alerts',
                widget_type=WidgetType.ALERT_FEED,
                title='Risk Alerts',
                data_source='alert_system',
                metrics=['risk_alerts'],
                update_frequency=UpdateFrequency.FAST,
                position={'x': 0, 'y': 4, 'width': 12, 'height': 3},
                options={'max_alerts': 10, 'show_severity': True}
            )
        ]
        
        configs.append(DashboardConfig(
            dashboard_id='risk_monitor',
            dashboard_type=DashboardType.RISK_MONITOR,
            name='Risk Monitor',
            description='Risk metrics and exposure monitoring',
            layout={'columns': 12, 'row_height': 50},
            widgets=risk_widgets,
            update_frequency=UpdateFrequency.NORMAL,
            permissions=['risk_management']
        ))
        
        return configs
    
    @pytest.mark.asyncio
    async def test_dashboard_data_generation(self, dashboard_service, sample_dashboard_configs):
        """Test dashboard data generation for different widget types"""
        # Register dashboards
        for config in sample_dashboard_configs:
            await dashboard_service.register_dashboard(config)
        
        # Generate data for trading overview dashboard
        trading_config = sample_dashboard_configs[0]
        dashboard_data = await dashboard_service.generate_dashboard_data(
            dashboard_id=trading_config.dashboard_id,
            user_context={'user_id': 'test_user', 'permissions': ['trading', 'view_positions']}
        )
        
        assert dashboard_data.dashboard_id == trading_config.dashboard_id
        assert len(dashboard_data.widgets) == len(trading_config.widgets)
        
        # Verify P&L chart data
        pnl_data = dashboard_data.widgets['pnl_chart']
        assert pnl_data.widget_id == 'pnl_chart'
        assert pnl_data.format == 'time_series'
        assert 'series' in pnl_data.data
        
        series_data = pnl_data.data['series']
        assert 'realized_pnl' in series_data
        assert 'unrealized_pnl' in series_data
        assert 'total_pnl' in series_data
        
        # Each series should have timestamps and values
        for series_name, series in series_data.items():
            assert 'timestamps' in series
            assert 'values' in series
            assert len(series['timestamps']) == len(series['values'])
            assert len(series['values']) > 0
        
        # Verify position table data
        position_data = dashboard_data.widgets['position_table']
        assert position_data.format == 'table'
        assert 'columns' in position_data.data
        assert 'rows' in position_data.data
        
        expected_columns = ['instrument', 'side', 'quantity', 'entry_price', 
                          'current_price', 'unrealized_pnl', 'pnl_percent']
        assert all(col in position_data.data['columns'] for col in expected_columns)
        
        # Verify metric card data
        volume_data = dashboard_data.widgets['trade_volume']
        assert volume_data.format == 'metric'
        assert 'value' in volume_data.data
        assert 'trend' in volume_data.data
        assert 'change_percent' in volume_data.data
        
        # Verify gauge data
        win_rate_data = dashboard_data.widgets['win_rate']
        assert win_rate_data.format == 'gauge'
        assert 'value' in win_rate_data.data
        assert 0 <= win_rate_data.data['value'] <= 100
        assert 'status' in win_rate_data.data  # danger/warning/success
    
    @pytest.mark.asyncio
    async def test_real_time_data_streaming(self, dashboard_service, websocket_manager):
        """Test real-time data streaming via WebSocket"""
        # Create test dashboard
        dashboard_config = DashboardConfig(
            dashboard_id='test_streaming',
            dashboard_type=DashboardType.TRADING_OVERVIEW,
            name='Test Streaming Dashboard',
            description='Test real-time streaming',
            layout={'columns': 12, 'row_height': 50},
            widgets=[
                WidgetConfig(
                    widget_id='price_chart',
                    widget_type=WidgetType.CANDLESTICK_CHART,
                    title='BTC/USDT Price',
                    data_source='market_data',
                    metrics=['ohlcv'],
                    update_frequency=UpdateFrequency.REAL_TIME,
                    position={'x': 0, 'y': 0, 'width': 12, 'height': 6}
                )
            ],
            update_frequency=UpdateFrequency.REAL_TIME
        )
        
        await dashboard_service.register_dashboard(dashboard_config)
        
        # Simulate WebSocket connection
        connection_id = 'test_conn_001'
        subscription = {
            'dashboard_id': 'test_streaming',
            'widgets': ['price_chart'],
            'quality': 'high'  # full updates
        }
        
        # Create data stream
        data_stream = await dashboard_service.create_data_stream(
            connection_id=connection_id,
            subscription=subscription
        )
        
        # Collect streamed data
        streamed_data = []
        stream_duration = 5  # seconds
        start_time = asyncio.get_event_loop().time()
        
        async for data_packet in data_stream:
            streamed_data.append(data_packet)
            
            # Check if we've streamed for long enough
            if asyncio.get_event_loop().time() - start_time > stream_duration:
                break
            
            # Simulate some processing delay
            await asyncio.sleep(0.05)
        
        # Verify streamed data
        assert len(streamed_data) > 0
        
        # Check data packet structure
        for packet in streamed_data:
            assert 'timestamp' in packet
            assert 'widget_id' in packet
            assert 'data' in packet
            assert packet['widget_id'] == 'price_chart'
            
            # Verify OHLCV data
            ohlcv_data = packet['data']
            assert 'open' in ohlcv_data
            assert 'high' in ohlcv_data
            assert 'low' in ohlcv_data
            assert 'close' in ohlcv_data
            assert 'volume' in ohlcv_data
            
            # Verify price relationships
            assert ohlcv_data['low'] <= ohlcv_data['open'] <= ohlcv_data['high']
            assert ohlcv_data['low'] <= ohlcv_data['close'] <= ohlcv_data['high']
        
        # Check update frequency
        if len(streamed_data) > 1:
            time_diffs = []
            for i in range(1, len(streamed_data)):
                diff = (streamed_data[i]['timestamp'] - streamed_data[i-1]['timestamp']).total_seconds()
                time_diffs.append(diff)
            
            avg_update_interval = np.mean(time_diffs)
            assert avg_update_interval < 1.0  # Should be updating faster than 1 second
    
    @pytest.mark.asyncio
    async def test_data_aggregation_for_charts(self, data_aggregator):
        """Test data aggregation for different chart types"""
        # Generate sample trading data
        base_time = datetime.now(timezone.utc) - timedelta(hours=24)
        raw_data = []
        
        # Generate 24 hours of 1-second data
        for i in range(86400):  # 24 * 60 * 60
            timestamp = base_time + timedelta(seconds=i)
            price = Decimal('50000') + Decimal(str(np.sin(i/3600) * 1000 + np.random.randn() * 50))
            volume = Decimal(str(abs(np.random.exponential(0.1))))
            
            raw_data.append({
                'timestamp': timestamp,
                'price': price,
                'volume': volume,
                'trades': np.random.poisson(5)
            })
        
        # Store raw data
        await data_aggregator.store_raw_data('market_data', raw_data)
        
        # Test different aggregation levels
        # 1. 1-minute aggregation for line chart
        minute_data = await data_aggregator.aggregate_data(
            data_source='market_data',
            metrics=['price', 'volume'],
            aggregation_window='1m',
            aggregation_functions={
                'price': ['open', 'high', 'low', 'close', 'mean'],
                'volume': ['sum']
            },
            start_time=base_time,
            end_time=base_time + timedelta(hours=1)
        )
        
        assert len(minute_data) == 60  # 60 minutes
        
        for point in minute_data:
            assert 'timestamp' in point
            assert 'price_open' in point
            assert 'price_high' in point
            assert 'price_low' in point
            assert 'price_close' in point
            assert 'price_mean' in point
            assert 'volume_sum' in point
            
            # Verify price relationships
            assert point['price_low'] <= point['price_open'] <= point['price_high']
            assert point['price_low'] <= point['price_close'] <= point['price_high']
            assert point['price_low'] <= point['price_mean'] <= point['price_high']
        
        # 2. 5-minute aggregation for candlestick chart
        candle_data = await data_aggregator.aggregate_data(
            data_source='market_data',
            metrics=['price', 'volume'],
            aggregation_window='5m',
            aggregation_functions={
                'price': ['ohlc'],
                'volume': ['sum', 'mean']
            },
            start_time=base_time,
            end_time=base_time + timedelta(hours=4)
        )
        
        assert len(candle_data) == 48  # 4 hours * 12 (5-minute candles per hour)
        
        # 3. Hourly aggregation for bar chart
        hourly_data = await data_aggregator.aggregate_data(
            data_source='market_data',
            metrics=['volume', 'trades'],
            aggregation_window='1h',
            aggregation_functions={
                'volume': ['sum'],
                'trades': ['sum', 'mean']
            },
            start_time=base_time,
            end_time=base_time + timedelta(hours=24)
        )
        
        assert len(hourly_data) == 24  # 24 hours
    
    @pytest.mark.asyncio
    async def test_heatmap_data_generation(self, dashboard_service):
        """Test heatmap data generation for correlation and exposure"""
        # Generate correlation matrix data
        instruments = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT']
        
        # Create correlation matrix
        correlation_matrix = np.random.rand(len(instruments), len(instruments))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
        
        # Convert to heatmap format
        heatmap_data = await dashboard_service.format_heatmap_data(
            matrix=correlation_matrix,
            row_labels=instruments,
            col_labels=instruments,
            title='Asset Correlation Matrix',
            color_scale='RdBu',
            center_value=0,
            annotations=True
        )
        
        assert heatmap_data['type'] == 'heatmap'
        assert 'data' in heatmap_data
        assert 'layout' in heatmap_data
        
        # Verify data structure
        data = heatmap_data['data'][0]
        assert 'z' in data  # Matrix values
        assert 'x' in data  # Column labels
        assert 'y' in data  # Row labels
        assert len(data['z']) == len(instruments)
        assert all(len(row) == len(instruments) for row in data['z'])
        
        # Verify layout
        layout = heatmap_data['layout']
        assert 'title' in layout
        assert 'xaxis' in layout
        assert 'yaxis' in layout
        assert 'colorscale' in layout
        
        # Test exposure heatmap
        exposure_data = {
            'BTC/USDT': {'long': 50000, 'short': 10000},
            'ETH/USDT': {'long': 30000, 'short': 20000},
            'SOL/USDT': {'long': 15000, 'short': 5000}
        }
        
        exposure_heatmap = await dashboard_service.format_exposure_heatmap(
            exposure_data=exposure_data,
            max_exposure=100000,
            show_net=True
        )
        
        assert exposure_heatmap['type'] == 'heatmap'
        assert 'net_exposure' in exposure_heatmap
    
    @pytest.mark.asyncio
    async def test_performance_metrics_dashboard(self, dashboard_service):
        """Test performance metrics dashboard data"""
        # Create performance dashboard widgets
        performance_widgets = [
            WidgetConfig(
                widget_id='sharpe_ratio_trend',
                widget_type=WidgetType.LINE_CHART,
                title='Sharpe Ratio Trend',
                data_source='performance_metrics',
                metrics=['sharpe_ratio_30d', 'sharpe_ratio_90d'],
                update_frequency=UpdateFrequency.SLOW,
                position={'x': 0, 'y': 0, 'width': 6, 'height': 4}
            ),
            WidgetConfig(
                widget_id='strategy_comparison',
                widget_type=WidgetType.BAR_CHART,
                title='Strategy Performance Comparison',
                data_source='performance_metrics',
                metrics=['strategy_returns'],
                update_frequency=UpdateFrequency.SLOW,
                position={'x': 6, 'y': 0, 'width': 6, 'height': 4}
            ),
            WidgetConfig(
                widget_id='drawdown_chart',
                widget_type=WidgetType.LINE_CHART,
                title='Maximum Drawdown',
                data_source='performance_metrics',
                metrics=['drawdown', 'underwater_curve'],
                update_frequency=UpdateFrequency.NORMAL,
                position={'x': 0, 'y': 4, 'width': 12, 'height': 4},
                options={'fill_negative': True, 'show_recovery_time': True}
            ),
            WidgetConfig(
                widget_id='monthly_returns',
                widget_type=WidgetType.HEATMAP,
                title='Monthly Returns Heatmap',
                data_source='performance_metrics',
                metrics=['monthly_returns'],
                update_frequency=UpdateFrequency.MINUTE,
                position={'x': 0, 'y': 8, 'width': 12, 'height': 6},
                options={'color_scale': 'RdYlGn', 'show_totals': True}
            )
        ]
        
        # Generate performance data
        performance_data = {}
        
        # Sharpe ratio trend
        sharpe_trend_data = []
        base_time = datetime.now(timezone.utc) - timedelta(days=90)
        
        for i in range(90):
            timestamp = base_time + timedelta(days=i)
            sharpe_30d = 1.5 + np.random.normal(0, 0.3)
            sharpe_90d = 1.2 + np.random.normal(0, 0.2)
            
            sharpe_trend_data.append({
                'timestamp': timestamp,
                'sharpe_ratio_30d': max(sharpe_30d, -2),  # Cap at -2
                'sharpe_ratio_90d': max(sharpe_90d, -2)
            })
        
        performance_data['sharpe_ratio_trend'] = {
            'format': 'time_series',
            'data': {
                'series': {
                    'sharpe_ratio_30d': {
                        'timestamps': [d['timestamp'] for d in sharpe_trend_data],
                        'values': [d['sharpe_ratio_30d'] for d in sharpe_trend_data]
                    },
                    'sharpe_ratio_90d': {
                        'timestamps': [d['timestamp'] for d in sharpe_trend_data],
                        'values': [d['sharpe_ratio_90d'] for d in sharpe_trend_data]
                    }
                }
            }
        }
        
        # Strategy comparison
        strategies = ['Grid Trading', 'Momentum', 'Mean Reversion', 'Arbitrage']
        strategy_returns = {
            strategy: {
                'daily': np.random.normal(0.1, 0.5),
                'weekly': np.random.normal(0.5, 2),
                'monthly': np.random.normal(2, 5),
                'ytd': np.random.normal(15, 10)
            }
            for strategy in strategies
        }
        
        performance_data['strategy_comparison'] = {
            'format': 'bar_chart',
            'data': {
                'categories': strategies,
                'series': [
                    {
                        'name': 'Daily %',
                        'data': [strategy_returns[s]['daily'] for s in strategies]
                    },
                    {
                        'name': 'Weekly %',
                        'data': [strategy_returns[s]['weekly'] for s in strategies]
                    },
                    {
                        'name': 'Monthly %',
                        'data': [strategy_returns[s]['monthly'] for s in strategies]
                    },
                    {
                        'name': 'YTD %',
                        'data': [strategy_returns[s]['ytd'] for s in strategies]
                    }
                ]
            }
        }
        
        # Drawdown chart
        equity_curve = [100000]  # Starting equity
        peak = 100000
        drawdown_data = []
        underwater_data = []
        
        for i in range(1, 365):  # 1 year of daily data
            # Simulate returns
            daily_return = np.random.normal(0.001, 0.02)
            new_equity = equity_curve[-1] * (1 + daily_return)
            equity_curve.append(new_equity)
            
            # Update peak
            if new_equity > peak:
                peak = new_equity
            
            # Calculate drawdown
            drawdown = (peak - new_equity) / peak
            underwater = -drawdown * 100  # Negative percentage
            
            drawdown_data.append({
                'timestamp': base_time + timedelta(days=i),
                'drawdown': drawdown * 100,  # Percentage
                'underwater': underwater,
                'equity': new_equity
            })
        
        performance_data['drawdown_chart'] = {
            'format': 'time_series',
            'data': {
                'series': {
                    'drawdown': {
                        'timestamps': [d['timestamp'] for d in drawdown_data],
                        'values': [d['drawdown'] for d in drawdown_data]
                    },
                    'underwater_curve': {
                        'timestamps': [d['timestamp'] for d in drawdown_data],
                        'values': [d['underwater'] for d in drawdown_data]
                    }
                },
                'annotations': {
                    'max_drawdown': max(d['drawdown'] for d in drawdown_data),
                    'current_drawdown': drawdown_data[-1]['drawdown'],
                    'recovery_periods': []  # Would calculate recovery periods
                }
            }
        }
        
        # Monthly returns heatmap
        monthly_returns = {}
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = [2023, 2024]
        
        for year in years:
            monthly_returns[year] = {}
            for month in months:
                # Skip future months
                if year == 2024 and months.index(month) > 6:  # Assuming we're in July 2024
                    monthly_returns[year][month] = None
                else:
                    monthly_returns[year][month] = np.random.normal(2, 5)
        
        performance_data['monthly_returns'] = {
            'format': 'heatmap',
            'data': {
                'years': years,
                'months': months,
                'values': [[monthly_returns[year][month] for month in months] for year in years],
                'totals': {
                    'yearly': {year: sum(v for v in monthly_returns[year].values() if v) for year in years},
                    'monthly': {month: sum(monthly_returns[year][month] for year in years 
                              if monthly_returns[year][month]) for month in months}
                }
            }
        }
        
        # Verify all performance metrics
        assert len(performance_data) == len(performance_widgets)
        
        for widget in performance_widgets:
            assert widget.widget_id in performance_data
            widget_data = performance_data[widget.widget_id]
            assert 'format' in widget_data
            assert 'data' in widget_data
    
    @pytest.mark.asyncio
    async def test_order_book_visualization(self, dashboard_service):
        """Test order book visualization data"""
        # Generate order book data
        mid_price = Decimal('50000')
        
        # Generate realistic order book
        order_book = {
            'bids': [],
            'asks': [],
            'timestamp': datetime.now(timezone.utc),
            'instrument': 'BTC/USDT'
        }
        
        # Generate bids (buy orders)
        for i in range(20):
            price = mid_price - Decimal(str((i + 1) * 10))
            # Larger quantities further from mid price
            quantity = Decimal(str(np.random.exponential(0.5 * (i + 1))))
            order_book['bids'].append({
                'price': price,
                'quantity': quantity,
                'total': sum(b['quantity'] for b in order_book['bids']) + quantity
            })
        
        # Generate asks (sell orders)
        for i in range(20):
            price = mid_price + Decimal(str((i + 1) * 10))
            quantity = Decimal(str(np.random.exponential(0.5 * (i + 1))))
            order_book['asks'].append({
                'price': price,
                'quantity': quantity,
                'total': sum(a['quantity'] for a in order_book['asks']) + quantity
            })
        
        # Format for visualization
        order_book_viz = await dashboard_service.format_order_book_data(
            order_book=order_book,
            depth_levels=10,
            group_by_price=True,
            show_cumulative=True
        )
        
        assert order_book_viz['format'] == 'order_book'
        assert 'bids' in order_book_viz['data']
        assert 'asks' in order_book_viz['data']
        assert 'spread' in order_book_viz['data']
        assert 'mid_price' in order_book_viz['data']
        
        # Verify bid/ask structure
        bids_data = order_book_viz['data']['bids']
        asks_data = order_book_viz['data']['asks']
        
        assert len(bids_data) <= 10  # Depth levels
        assert len(asks_data) <= 10
        
        # Verify price ordering
        bid_prices = [b['price'] for b in bids_data]
        ask_prices = [a['price'] for a in asks_data]
        
        assert bid_prices == sorted(bid_prices, reverse=True)  # Descending
        assert ask_prices == sorted(ask_prices)  # Ascending
        
        # Verify spread calculation
        best_bid = float(bids_data[0]['price'])
        best_ask = float(asks_data[0]['price'])
        spread = order_book_viz['data']['spread']
        
        assert spread['absolute'] == best_ask - best_bid
        assert spread['percentage'] == ((best_ask - best_bid) / mid_price) * 100
        
        # Test order book imbalance indicator
        total_bid_volume = sum(float(b['quantity']) for b in bids_data)
        total_ask_volume = sum(float(a['quantity']) for a in asks_data)
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        
        assert 'imbalance' in order_book_viz['data']
        assert abs(order_book_viz['data']['imbalance'] - imbalance) < 0.001
    
    @pytest.mark.asyncio
    async def test_alert_feed_widget(self, dashboard_service):
        """Test alert feed widget data generation"""
        # Generate sample alerts
        alerts = []
        alert_types = ['RISK', 'TRADING', 'SYSTEM', 'MARKET']
        severities = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        base_time = datetime.now(timezone.utc)
        
        for i in range(20):
            alert = {
                'alert_id': f'ALERT_{i:03d}',
                'timestamp': base_time - timedelta(minutes=i),
                'type': np.random.choice(alert_types),
                'severity': np.random.choice(severities, p=[0.4, 0.3, 0.2, 0.1]),
                'title': f'Alert {i}',
                'description': f'This is alert number {i}',
                'acknowledged': i > 15,  # Older alerts acknowledged
                'resolved': i > 18
            }
            alerts.append(alert)
        
        # Format alert feed data
        alert_feed_data = await dashboard_service.format_alert_feed(
            alerts=alerts,
            max_alerts=10,
            group_by_type=True,
            show_resolved=False
        )
        
        assert alert_feed_data['format'] == 'alert_feed'
        assert 'alerts' in alert_feed_data['data']
        assert 'summary' in alert_feed_data['data']
        
        # Check filtered alerts
        displayed_alerts = alert_feed_data['data']['alerts']
        assert len(displayed_alerts) <= 10
        assert all(not alert['resolved'] for alert in displayed_alerts)
        
        # Check summary statistics
        summary = alert_feed_data['data']['summary']
        assert 'total_active' in summary
        assert 'by_severity' in summary
        assert 'by_type' in summary
        assert 'unacknowledged' in summary
        
        # Verify severity counts
        severity_counts = summary['by_severity']
        for severity in severities:
            assert severity in severity_counts
    
    @pytest.mark.asyncio
    async def test_dashboard_caching(self, dashboard_service):
        """Test dashboard data caching for performance"""
        # Create a dashboard with expensive calculations
        dashboard_config = DashboardConfig(
            dashboard_id='cache_test',
            dashboard_type=DashboardType.PERFORMANCE_ANALYTICS,
            name='Cache Test Dashboard',
            description='Testing caching behavior',
            layout={'columns': 12, 'row_height': 50},
            widgets=[
                WidgetConfig(
                    widget_id='expensive_calc',
                    widget_type=WidgetType.LINE_CHART,
                    title='Expensive Calculation',
                    data_source='performance_metrics',
                    metrics=['complex_metric'],
                    update_frequency=UpdateFrequency.SLOW,
                    position={'x': 0, 'y': 0, 'width': 12, 'height': 6}
                )
            ],
            update_frequency=UpdateFrequency.SLOW
        )
        
        await dashboard_service.register_dashboard(dashboard_config)
        
        # First request - should calculate and cache
        start_time = asyncio.get_event_loop().time()
        data1 = await dashboard_service.generate_dashboard_data(
            dashboard_id='cache_test',
            user_context={'user_id': 'test_user'}
        )
        first_request_time = asyncio.get_event_loop().time() - start_time
        
        # Second request - should use cache
        start_time = asyncio.get_event_loop().time()
        data2 = await dashboard_service.generate_dashboard_data(
            dashboard_id='cache_test',
            user_context={'user_id': 'test_user'}
        )
        second_request_time = asyncio.get_event_loop().time() - start_time
        
        # Cache should make second request much faster
        assert second_request_time < first_request_time * 0.5
        
        # Data should be identical
        assert data1.widgets['expensive_calc'].data == data2.widgets['expensive_calc'].data
        
        # Check cache metadata
        cache_info = await dashboard_service.get_cache_info('cache_test')
        assert cache_info['hits'] >= 1
        assert cache_info['misses'] >= 1
        assert cache_info['hit_rate'] > 0
        
        # Test cache invalidation
        await dashboard_service.invalidate_cache('cache_test')
        
        # Next request should recalculate
        start_time = asyncio.get_event_loop().time()
        data3 = await dashboard_service.generate_dashboard_data(
            dashboard_id='cache_test',
            user_context={'user_id': 'test_user'}
        )
        third_request_time = asyncio.get_event_loop().time() - start_time
        
        # Should be slow again after invalidation
        assert third_request_time > second_request_time * 2
    
    @pytest.mark.asyncio
    async def test_dashboard_permissions(self, dashboard_service):
        """Test dashboard access control and permissions"""
        # Create dashboards with different permission requirements
        dashboards = [
            DashboardConfig(
                dashboard_id='public_dash',
                dashboard_type=DashboardType.MARKET_ANALYSIS,
                name='Public Market Dashboard',
                description='Publicly accessible',
                layout={'columns': 12, 'row_height': 50},
                widgets=[],
                update_frequency=UpdateFrequency.NORMAL,
                permissions=[]  # No permissions required
            ),
            DashboardConfig(
                dashboard_id='trading_dash',
                dashboard_type=DashboardType.TRADING_OVERVIEW,
                name='Trading Dashboard',
                description='Requires trading permission',
                layout={'columns': 12, 'row_height': 50},
                widgets=[],
                update_frequency=UpdateFrequency.FAST,
                permissions=['trading', 'view_positions']
            ),
            DashboardConfig(
                dashboard_id='admin_dash',
                dashboard_type=DashboardType.SYSTEM_HEALTH,
                name='Admin Dashboard',
                description='Admin only',
                layout={'columns': 12, 'row_height': 50},
                widgets=[],
                update_frequency=UpdateFrequency.NORMAL,
                permissions=['admin', 'system_monitoring']
            )
        ]
        
        for config in dashboards:
            await dashboard_service.register_dashboard(config)
        
        # Test different user contexts
        user_contexts = [
            {'user_id': 'public_user', 'permissions': []},
            {'user_id': 'trader', 'permissions': ['trading', 'view_positions']},
            {'user_id': 'admin', 'permissions': ['admin', 'system_monitoring', 'trading', 'view_positions']}
        ]
        
        access_results = {}
        
        for user in user_contexts:
            access_results[user['user_id']] = {}
            
            for dashboard in dashboards:
                can_access = await dashboard_service.check_access(
                    dashboard_id=dashboard.dashboard_id,
                    user_context=user
                )
                access_results[user['user_id']][dashboard.dashboard_id] = can_access
        
        # Verify access control
        # Public user can only access public dashboard
        assert access_results['public_user']['public_dash'] == True
        assert access_results['public_user']['trading_dash'] == False
        assert access_results['public_user']['admin_dash'] == False
        
        # Trader can access public and trading dashboards
        assert access_results['trader']['public_dash'] == True
        assert access_results['trader']['trading_dash'] == True
        assert access_results['trader']['admin_dash'] == False
        
        # Admin can access all dashboards
        assert access_results['admin']['public_dash'] == True
        assert access_results['admin']['trading_dash'] == True
        assert access_results['admin']['admin_dash'] == True
    
    @pytest.mark.asyncio
    async def test_responsive_dashboard_layouts(self, dashboard_service):
        """Test responsive dashboard layouts for different screen sizes"""
        # Define responsive breakpoints
        breakpoints = {
            'mobile': {'width': 375, 'columns': 4},
            'tablet': {'width': 768, 'columns': 8},
            'desktop': {'width': 1920, 'columns': 12}
        }
        
        # Create dashboard with responsive layout
        dashboard_config = DashboardConfig(
            dashboard_id='responsive_test',
            dashboard_type=DashboardType.TRADING_OVERVIEW,
            name='Responsive Dashboard',
            description='Tests responsive behavior',
            layout={'columns': 12, 'row_height': 50, 'breakpoints': breakpoints},
            widgets=[
                WidgetConfig(
                    widget_id='main_chart',
                    widget_type=WidgetType.LINE_CHART,
                    title='Main Chart',
                    data_source='market_data',
                    metrics=['price'],
                    update_frequency=UpdateFrequency.FAST,
                    position={
                        'desktop': {'x': 0, 'y': 0, 'width': 8, 'height': 6},
                        'tablet': {'x': 0, 'y': 0, 'width': 8, 'height': 4},
                        'mobile': {'x': 0, 'y': 0, 'width': 4, 'height': 3}
                    }
                ),
                WidgetConfig(
                    widget_id='side_panel',
                    widget_type=WidgetType.TABLE,
                    title='Positions',
                    data_source='position_manager',
                    metrics=['positions'],
                    update_frequency=UpdateFrequency.FAST,
                    position={
                        'desktop': {'x': 8, 'y': 0, 'width': 4, 'height': 6},
                        'tablet': {'x': 0, 'y': 4, 'width': 8, 'height': 4},
                        'mobile': {'x': 0, 'y': 3, 'width': 4, 'height': 3}
                    }
                )
            ],
            update_frequency=UpdateFrequency.FAST
        )
        
        await dashboard_service.register_dashboard(dashboard_config)
        
        # Test layout generation for different screen sizes
        for device, specs in breakpoints.items():
            layout = await dashboard_service.generate_layout(
                dashboard_id='responsive_test',
                screen_width=specs['width'],
                device_type=device
            )
            
            assert layout['columns'] == specs['columns']
            assert len(layout['widgets']) == len(dashboard_config.widgets)
            
            # Verify widget positions are adjusted
            for widget in layout['widgets']:
                widget_config = next(w for w in dashboard_config.widgets if w.widget_id == widget['widget_id'])
                expected_position = widget_config.position.get(device, widget_config.position)
                
                if isinstance(expected_position, dict) and device in expected_position:
                    assert widget['position'] == expected_position[device]
                
                # Ensure widgets fit within column constraints
                assert widget['position']['x'] + widget['position']['width'] <= specs['columns']


class TestDashboardIntegration:
    """Test dashboard integration with other systems"""
    
    @pytest.mark.asyncio
    async def test_multi_user_dashboard_sync(self, dashboard_service, websocket_manager):
        """Test multi-user dashboard synchronization"""
        dashboard_id = 'shared_dashboard'
        
        # Simulate multiple users viewing same dashboard
        users = [
            {'user_id': 'user1', 'connection_id': 'conn1'},
            {'user_id': 'user2', 'connection_id': 'conn2'},
            {'user_id': 'user3', 'connection_id': 'conn3'}
        ]
        
        # Connect all users
        connections = []
        for user in users:
            conn = await websocket_manager.create_connection(
                connection_id=user['connection_id'],
                user_id=user['user_id']
            )
            
            # Subscribe to dashboard
            await dashboard_service.subscribe_to_dashboard(
                dashboard_id=dashboard_id,
                connection_id=user['connection_id']
            )
            
            connections.append(conn)
        
        # Verify all users are subscribed
        subscribers = await dashboard_service.get_dashboard_subscribers(dashboard_id)
        assert len(subscribers) == 3
        
        # Broadcast update to all users
        update_data = {
            'widget_id': 'shared_widget',
            'data': {'value': 12345},
            'timestamp': datetime.now(timezone.utc)
        }
        
        broadcast_result = await dashboard_service.broadcast_update(
            dashboard_id=dashboard_id,
            update_data=update_data
        )
        
        assert broadcast_result['delivered_to'] == 3
        assert broadcast_result['failed_deliveries'] == 0
        
        # Test selective updates based on permissions
        restricted_update = {
            'widget_id': 'admin_widget',
            'data': {'sensitive_value': 99999},
            'required_permissions': ['admin'],
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Only admin should receive this
        selective_result = await dashboard_service.broadcast_update(
            dashboard_id=dashboard_id,
            update_data=restricted_update,
            check_permissions=True
        )
        
        # Assuming only one user has admin permissions
        assert selective_result['delivered_to'] < 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])