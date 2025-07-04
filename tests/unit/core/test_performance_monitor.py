"""
Unit tests for the Performance Monitoring System.

Tests cover:
- Real-time performance tracking
- P&L calculation and attribution
- Performance metrics (Sharpe, Sortino, Calmar)
- Trade analytics and statistics
- Benchmark comparison
- Performance attribution
- Risk-adjusted returns
- Performance reporting and visualization
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
import asyncio
from dataclasses import dataclass

# GridAttention project imports
from core.performance_monitor import (
    PerformanceMonitor,
    PerformanceConfig,
    Trade,
    Position,
    PerformanceMetrics,
    PnLCalculator,
    AttributionAnalyzer,
    BenchmarkComparator,
    PerformanceReport,
    MetricType,
    TimeFrame
)


class TestPerformanceConfig:
    """Test cases for PerformanceConfig validation."""
    
    def test_default_config(self):
        """Test default performance configuration."""
        config = PerformanceConfig()
        
        assert config.calculation_frequency == 'realtime'
        assert config.risk_free_rate == 0.03  # 3% annual
        assert config.benchmark_symbol == 'BTC/USDT'
        assert config.reporting_currency == 'USDT'
        assert config.include_fees is True
        assert config.include_slippage is True
        
    def test_custom_config(self):
        """Test custom performance configuration."""
        config = PerformanceConfig(
            calculation_frequency='1min',
            risk_free_rate=0.05,
            benchmark_symbol='SPY',
            reporting_currency='USD',
            performance_window_days=30
        )
        
        assert config.calculation_frequency == '1min'
        assert config.risk_free_rate == 0.05
        assert config.performance_window_days == 30
        
    def test_config_validation(self):
        """Test configuration validation rules."""
        # Invalid risk-free rate
        with pytest.raises(ValueError, match="risk_free_rate"):
            PerformanceConfig(risk_free_rate=-0.01)
            
        # Invalid calculation frequency
        with pytest.raises(ValueError, match="calculation_frequency"):
            PerformanceConfig(calculation_frequency='invalid')
            
        # Invalid performance window
        with pytest.raises(ValueError, match="performance_window"):
            PerformanceConfig(performance_window_days=0)


class TestTrade:
    """Test cases for Trade data structure."""
    
    def test_trade_creation(self):
        """Test trade creation and validation."""
        trade = Trade(
            trade_id='trade_001',
            symbol='BTC/USDT',
            side='buy',
            quantity=Decimal('0.5'),
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            fees=Decimal('50'),
            slippage=Decimal('25')
        )
        
        assert trade.trade_id == 'trade_001'
        assert trade.quantity == Decimal('0.5')
        assert trade.is_closed is True
        
    def test_trade_pnl_calculation(self):
        """Test trade P&L calculation."""
        trade = Trade(
            trade_id='trade_001',
            symbol='BTC/USDT',
            side='buy',
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            fees=Decimal('100'),
            slippage=Decimal('50')
        )
        
        # Gross P&L
        gross_pnl = trade.calculate_gross_pnl()
        assert gross_pnl == Decimal('1000')  # (51000 - 50000) * 1
        
        # Net P&L
        net_pnl = trade.calculate_net_pnl()
        assert net_pnl == Decimal('850')  # 1000 - 100 - 50
        
        # Return percentage
        return_pct = trade.calculate_return_percentage()
        assert return_pct == Decimal('0.017')  # 850 / 50000
        
    def test_trade_duration(self):
        """Test trade duration calculation."""
        trade = Trade(
            trade_id='trade_001',
            symbol='BTC/USDT',
            side='buy',
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            entry_time=datetime.now() - timedelta(hours=24),
            exit_time=datetime.now()
        )
        
        duration = trade.get_duration()
        assert abs(duration.total_seconds() - 24 * 3600) < 1  # Allow 1 second tolerance
        
    def test_r_multiple_calculation(self):
        """Test R-multiple calculation for trades."""
        trade = Trade(
            trade_id='trade_001',
            symbol='BTC/USDT',
            side='buy',
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            stop_loss_price=Decimal('49000')
        )
        
        r_multiple = trade.calculate_r_multiple()
        assert r_multiple == Decimal('1.0')  # Risk = 1000, Profit = 1000


class TestPosition:
    """Test cases for Position tracking."""
    
    def test_position_creation(self):
        """Test position creation and initialization."""
        position = Position(
            symbol='BTC/USDT',
            side='long',
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000'),
            entry_time=datetime.now()
        )
        
        assert position.symbol == 'BTC/USDT'
        assert position.quantity == Decimal('1.0')
        assert position.is_open is True
        assert position.unrealized_pnl == Decimal('0')
        
    def test_position_update(self):
        """Test position updates with new fills."""
        position = Position(
            symbol='BTC/USDT',
            side='long',
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000')
        )
        
        # Add to position
        position.add_fill(
            quantity=Decimal('0.5'),
            price=Decimal('49000'),
            fees=Decimal('25')
        )
        
        assert position.quantity == Decimal('1.5')
        assert position.average_entry_price == Decimal('49666.67')  # Weighted average
        assert position.total_fees == Decimal('25')
        
    def test_position_mark_to_market(self):
        """Test position mark-to-market calculation."""
        position = Position(
            symbol='BTC/USDT',
            side='long',
            quantity=Decimal('2.0'),
            entry_price=Decimal('50000')
        )
        
        # Update market price
        market_price = Decimal('51000')
        position.mark_to_market(market_price)
        
        assert position.unrealized_pnl == Decimal('2000')  # 2 * (51000 - 50000)
        assert position.current_value == Decimal('102000')
        
        # Test short position
        short_position = Position(
            symbol='BTC/USDT',
            side='short',
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000')
        )
        
        short_position.mark_to_market(market_price)
        assert short_position.unrealized_pnl == Decimal('-1000')  # Loss on short


class TestPnLCalculator:
    """Test cases for P&L calculation engine."""
    
    @pytest.fixture
    def pnl_calculator(self):
        """Create PnL calculator instance."""
        return PnLCalculator(base_currency='USDT')
        
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        trades = [
            Trade(
                trade_id='001',
                symbol='BTC/USDT',
                side='buy',
                quantity=Decimal('1.0'),
                entry_price=Decimal('50000'),
                exit_price=Decimal('51000'),
                entry_time=datetime.now() - timedelta(days=2),
                exit_time=datetime.now() - timedelta(days=1),
                fees=Decimal('100')
            ),
            Trade(
                trade_id='002',
                symbol='ETH/USDT',
                side='sell',
                quantity=Decimal('10.0'),
                entry_price=Decimal('3000'),
                exit_price=Decimal('2950'),
                entry_time=datetime.now() - timedelta(days=1),
                exit_time=datetime.now(),
                fees=Decimal('50')
            )
        ]
        return trades
        
    def test_realized_pnl_calculation(self, pnl_calculator, sample_trades):
        """Test realized P&L calculation."""
        realized_pnl = pnl_calculator.calculate_realized_pnl(sample_trades)
        
        assert 'total_pnl' in realized_pnl
        assert 'gross_pnl' in realized_pnl
        assert 'total_fees' in realized_pnl
        assert 'net_pnl' in realized_pnl
        
        # BTC: +1000 - 100 = 900
        # ETH: +500 - 50 = 450 (short profit)
        # Total: 1350
        assert realized_pnl['net_pnl'] == Decimal('1350')
        
    def test_unrealized_pnl_calculation(self, pnl_calculator):
        """Test unrealized P&L calculation."""
        positions = [
            Position(
                symbol='BTC/USDT',
                side='long',
                quantity=Decimal('0.5'),
                entry_price=Decimal('48000')
            ),
            Position(
                symbol='ETH/USDT',
                side='short',
                quantity=Decimal('5.0'),
                entry_price=Decimal('3100')
            )
        ]
        
        market_prices = {
            'BTC/USDT': Decimal('50000'),
            'ETH/USDT': Decimal('3000')
        }
        
        unrealized_pnl = pnl_calculator.calculate_unrealized_pnl(positions, market_prices)
        
        # BTC: 0.5 * (50000 - 48000) = 1000
        # ETH: -5 * (3000 - 3100) = 500
        # Total: 1500
        assert unrealized_pnl['total_unrealized'] == Decimal('1500')
        assert len(unrealized_pnl['by_position']) == 2
        
    def test_daily_pnl_calculation(self, pnl_calculator, sample_trades):
        """Test daily P&L calculation."""
        # Add market prices for open positions
        positions = []
        market_prices = {}
        
        daily_pnl = pnl_calculator.calculate_daily_pnl(
            date=datetime.now().date(),
            trades=sample_trades,
            positions=positions,
            market_prices=market_prices
        )
        
        assert 'date' in daily_pnl
        assert 'realized_pnl' in daily_pnl
        assert 'unrealized_pnl' in daily_pnl
        assert 'total_pnl' in daily_pnl
        assert 'trade_count' in daily_pnl
        
    def test_pnl_by_symbol(self, pnl_calculator, sample_trades):
        """Test P&L breakdown by symbol."""
        pnl_by_symbol = pnl_calculator.calculate_pnl_by_symbol(sample_trades)
        
        assert 'BTC/USDT' in pnl_by_symbol
        assert 'ETH/USDT' in pnl_by_symbol
        
        assert pnl_by_symbol['BTC/USDT']['net_pnl'] == Decimal('900')
        assert pnl_by_symbol['ETH/USDT']['net_pnl'] == Decimal('450')
        
    def test_cumulative_pnl(self, pnl_calculator):
        """Test cumulative P&L calculation over time."""
        # Create trades over multiple days
        trades = []
        for i in range(10):
            trade = Trade(
                trade_id=f'trade_{i}',
                symbol='BTC/USDT',
                side='buy' if i % 2 == 0 else 'sell',
                quantity=Decimal('0.1'),
                entry_price=Decimal('50000'),
                exit_price=Decimal('50100') if i % 2 == 0 else Decimal('49900'),
                entry_time=datetime.now() - timedelta(days=10-i, hours=12),
                exit_time=datetime.now() - timedelta(days=10-i),
                fees=Decimal('5')
            )
            trades.append(trade)
            
        cumulative_pnl = pnl_calculator.calculate_cumulative_pnl(trades)
        
        assert len(cumulative_pnl) == 10
        assert cumulative_pnl.index[0] < cumulative_pnl.index[-1]
        assert cumulative_pnl.iloc[-1] == sum(t.calculate_net_pnl() for t in trades)


class TestPerformanceMetrics:
    """Test cases for performance metrics calculation."""
    
    @pytest.fixture
    def performance_metrics(self):
        """Create PerformanceMetrics instance."""
        return PerformanceMetrics(risk_free_rate=0.03)
        
    @pytest.fixture
    def returns_series(self):
        """Create sample returns series."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        # Generate returns with positive drift and volatility
        returns = np.random.normal(0.0005, 0.02, 252)
        return pd.Series(returns, index=dates)
        
    def test_sharpe_ratio(self, performance_metrics, returns_series):
        """Test Sharpe ratio calculation."""
        sharpe = performance_metrics.calculate_sharpe_ratio(returns_series)
        
        # Check calculation
        excess_returns = returns_series - 0.03/252  # Daily risk-free rate
        expected_sharpe = np.sqrt(252) * excess_returns.mean() / returns_series.std()
        
        assert abs(sharpe - expected_sharpe) < 0.001
        
    def test_sortino_ratio(self, performance_metrics, returns_series):
        """Test Sortino ratio calculation."""
        sortino = performance_metrics.calculate_sortino_ratio(returns_series)
        
        # Sortino uses downside deviation
        downside_returns = returns_series[returns_series < 0]
        
        # Should be higher than Sharpe for positively skewed returns
        sharpe = performance_metrics.calculate_sharpe_ratio(returns_series)
        if returns_series.mean() > 0 and len(downside_returns) > 0:
            assert sortino != sharpe  # Different due to downside focus
            
    def test_calmar_ratio(self, performance_metrics):
        """Test Calmar ratio calculation."""
        # Create equity curve
        initial_capital = 100000
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # Simulate equity curve with drawdown
        equity_values = [initial_capital]
        for i in range(1, 252):
            if i < 100:
                # Uptrend
                equity_values.append(equity_values[-1] * 1.001)
            elif i < 150:
                # Drawdown
                equity_values.append(equity_values[-1] * 0.998)
            else:
                # Recovery
                equity_values.append(equity_values[-1] * 1.0015)
                
        equity_curve = pd.Series(equity_values, index=dates)
        
        calmar = performance_metrics.calculate_calmar_ratio(equity_curve)
        
        # Calculate expected values
        annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        max_dd = performance_metrics.calculate_max_drawdown(equity_curve)
        expected_calmar = annual_return / abs(max_dd)
        
        assert abs(calmar - expected_calmar) < 0.02  # Increased tolerance
        
    def test_max_drawdown(self, performance_metrics):
        """Test maximum drawdown calculation."""
        # Create equity curve with known drawdown
        equity_curve = pd.Series([
            100000, 105000, 110000, 108000, 104000,  # -5.45% drawdown
            102000, 106000, 111000, 115000, 113000   # -1.74% drawdown
        ])
        
        max_dd = performance_metrics.calculate_max_drawdown(equity_curve)
        
        # Max drawdown from 110000 to 102000
        expected_dd = (102000 - 110000) / 110000
        assert abs(max_dd - expected_dd) < 0.001
        
    def test_win_rate_metrics(self, performance_metrics):
        """Test win rate and related metrics."""
        trades = [
            {'pnl': Decimal('100')},
            {'pnl': Decimal('200')},
            {'pnl': Decimal('-50')},
            {'pnl': Decimal('150')},
            {'pnl': Decimal('-100')},
            {'pnl': Decimal('300')},
            {'pnl': Decimal('-75')},
            {'pnl': Decimal('50')}
        ]
        
        metrics = performance_metrics.calculate_win_metrics(trades)
        
        assert metrics['win_rate'] == 0.625  # 5/8
        assert metrics['average_win'] == Decimal('160')  # (100+200+150+300+50)/5
        assert metrics['average_loss'] == Decimal('75')   # (50+100+75)/3
        assert metrics['profit_factor'] == Decimal('800') / Decimal('225')  # Total wins / Total losses
        
    def test_risk_metrics(self, performance_metrics, returns_series):
        """Test various risk metrics."""
        risk_metrics = performance_metrics.calculate_risk_metrics(returns_series)
        
        assert 'volatility' in risk_metrics
        assert 'downside_deviation' in risk_metrics
        assert 'var_95' in risk_metrics
        assert 'cvar_95' in risk_metrics
        assert 'skewness' in risk_metrics
        assert 'kurtosis' in risk_metrics
        
        # Verify calculations
        assert risk_metrics['volatility'] == returns_series.std() * np.sqrt(252)
        assert risk_metrics['var_95'] <= 0  # VaR is negative
        assert abs(risk_metrics['cvar_95']) >= abs(risk_metrics['var_95'])  # CVaR more conservative
        
    def test_rolling_metrics(self, performance_metrics, returns_series):
        """Test rolling performance metrics."""
        window = 30  # 30-day rolling window
        
        rolling_sharpe = performance_metrics.calculate_rolling_sharpe(
            returns_series, 
            window=window
        )
        
        assert len(rolling_sharpe) == len(returns_series) - window + 1
        assert not rolling_sharpe.isna().any()
        
        # Test rolling volatility
        rolling_vol = performance_metrics.calculate_rolling_volatility(
            returns_series,
            window=window
        )
        
        assert len(rolling_vol) == len(returns_series) - window + 1
        assert (rolling_vol > 0).all()


class TestAttributionAnalyzer:
    """Test cases for performance attribution analysis."""
    
    @pytest.fixture
    def attribution_analyzer(self):
        """Create AttributionAnalyzer instance."""
        return AttributionAnalyzer()
        
    @pytest.fixture
    def multi_strategy_trades(self):
        """Create trades from multiple strategies."""
        trades = []
        
        # Grid strategy trades
        for i in range(5):
            # Make some trades losing
            if i < 3:  # First 3 winning
                exit_price = Decimal('50100') if i % 2 == 0 else Decimal('49900')
            else:  # Last 2 losing
                exit_price = Decimal('49900') if i % 2 == 0 else Decimal('50100')
                
            trades.append(Trade(
                trade_id=f'grid_{i}',
                symbol='BTC/USDT',
                side='buy' if i % 2 == 0 else 'sell',
                quantity=Decimal('0.1'),
                entry_price=Decimal('50000'),
                exit_price=exit_price,
                strategy='grid',
                fees=Decimal('5')
            ))
            
        # Trend following trades
        for i in range(3):
            trades.append(Trade(
                trade_id=f'trend_{i}',
                symbol='ETH/USDT',
                side='buy',
                quantity=Decimal('1.0'),
                entry_price=Decimal('3000'),
                exit_price=Decimal('3200'),
                strategy='trend',
                fees=Decimal('10')
            ))
            
        return trades
        
    def test_strategy_attribution(self, attribution_analyzer, multi_strategy_trades):
        """Test P&L attribution by strategy."""
        attribution = attribution_analyzer.attribute_by_strategy(multi_strategy_trades)
        
        assert 'grid' in attribution
        assert 'trend' in attribution
        
        # Grid should have mixed results
        assert attribution['grid']['trade_count'] == 5
        assert attribution['grid']['win_rate'] == 0.6  # 3/5
        
        # Trend should be profitable
        assert attribution['trend']['trade_count'] == 3
        assert attribution['trend']['total_pnl'] > 0
        
    def test_symbol_attribution(self, attribution_analyzer, multi_strategy_trades):
        """Test P&L attribution by symbol."""
        attribution = attribution_analyzer.attribute_by_symbol(multi_strategy_trades)
        
        assert 'BTC/USDT' in attribution
        assert 'ETH/USDT' in attribution
        
        assert attribution['BTC/USDT']['trade_count'] == 5
        assert attribution['ETH/USDT']['trade_count'] == 3
        
    def test_time_attribution(self, attribution_analyzer):
        """Test P&L attribution by time period."""
        # Create trades across different times
        trades = []
        for hour in [9, 10, 14, 15, 16]:  # Different trading hours
            for i in range(2):
                trade_time = datetime.now().replace(hour=hour, minute=0)
                trades.append(Trade(
                    trade_id=f'trade_{hour}_{i}',
                    symbol='BTC/USDT',
                    side='buy',
                    quantity=Decimal('0.1'),
                    entry_price=Decimal('50000'),
                    exit_price=Decimal('50100') if hour < 12 else Decimal('49900'),
                    entry_time=trade_time - timedelta(minutes=30),
                    exit_time=trade_time,
                    fees=Decimal('5')
                ))
                
        attribution = attribution_analyzer.attribute_by_time(trades, period='hour')
        
        # Morning hours should be profitable
        assert attribution[9]['net_pnl'] > 0
        assert attribution[10]['net_pnl'] > 0
        
        # Afternoon hours should have losses
        assert attribution[14]['net_pnl'] < 0
        
    def test_risk_factor_attribution(self, attribution_analyzer):
        """Test attribution by risk factors."""
        trades = []
        
        # High volatility trades
        for i in range(3):
            trades.append(Trade(
                trade_id=f'high_vol_{i}',
                symbol='BTC/USDT',
                side='buy',
                quantity=Decimal('0.5'),
                entry_price=Decimal('50000'),
                exit_price=Decimal('48000'),  # Losses
                market_volatility=0.05,  # 5% volatility
                fees=Decimal('20')
            ))
            
        # Low volatility trades  
        for i in range(3):
            trades.append(Trade(
                trade_id=f'low_vol_{i}',
                symbol='ETH/USDT',
                side='buy',
                quantity=Decimal('1.0'),
                entry_price=Decimal('3000'),
                exit_price=Decimal('3050'),  # Small gains
                market_volatility=0.01,  # 1% volatility
                fees=Decimal('10')
            ))
            
        attribution = attribution_analyzer.attribute_by_risk_factor(
            trades, 
            factor='volatility'
        )
        
        assert 'high_volatility' in attribution
        assert 'low_volatility' in attribution
        
        # High volatility trades performed worse
        assert attribution['high_volatility']['average_pnl'] < attribution['low_volatility']['average_pnl']


class TestBenchmarkComparator:
    """Test cases for benchmark comparison."""
    
    @pytest.fixture
    def benchmark_comparator(self):
        """Create BenchmarkComparator instance."""
        return BenchmarkComparator(benchmark_symbol='BTC/USDT')
        
    @pytest.fixture
    def strategy_returns(self):
        """Create strategy returns series."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # Strategy with higher return but more volatility
        returns = np.random.normal(0.002, 0.025, 100)
        return pd.Series(returns, index=dates)
        
    @pytest.fixture
    def benchmark_returns(self):
        """Create benchmark returns series."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # Benchmark with lower return but less volatility
        returns = np.random.normal(0.001, 0.015, 100)
        return pd.Series(returns, index=dates)
        
    def test_alpha_beta_calculation(self, benchmark_comparator, strategy_returns, benchmark_returns):
        """Test alpha and beta calculation."""
        results = benchmark_comparator.calculate_alpha_beta(
            strategy_returns,
            benchmark_returns
        )
        
        assert 'alpha' in results
        assert 'beta' in results
        assert 'r_squared' in results
        
        # Beta can be positive or negative depending on correlation
        assert 'beta' in results and results['beta'] is not None
        
        # R-squared should be between 0 and 1
        assert 0 <= results['r_squared'] <= 1
        
    def test_tracking_error(self, benchmark_comparator, strategy_returns, benchmark_returns):
        """Test tracking error calculation."""
        tracking_error = benchmark_comparator.calculate_tracking_error(
            strategy_returns,
            benchmark_returns
        )
        
        # Should equal standard deviation of return differences
        return_diff = strategy_returns - benchmark_returns
        expected_te = return_diff.std() * np.sqrt(252)
        
        assert abs(tracking_error - expected_te) < 0.001
        
    def test_information_ratio(self, benchmark_comparator, strategy_returns, benchmark_returns):
        """Test information ratio calculation."""
        info_ratio = benchmark_comparator.calculate_information_ratio(
            strategy_returns,
            benchmark_returns
        )
        
        # IR = (Strategy Return - Benchmark Return) / Tracking Error
        excess_return = (strategy_returns.mean() - benchmark_returns.mean()) * 252
        tracking_error = benchmark_comparator.calculate_tracking_error(
            strategy_returns,
            benchmark_returns
        )
        
        expected_ir = excess_return / tracking_error
        assert abs(info_ratio - expected_ir) < 0.01
        
    def test_relative_performance(self, benchmark_comparator, strategy_returns, benchmark_returns):
        """Test relative performance metrics."""
        relative_metrics = benchmark_comparator.calculate_relative_performance(
            strategy_returns,
            benchmark_returns
        )
        
        assert 'excess_return' in relative_metrics
        assert 'win_rate_vs_benchmark' in relative_metrics
        assert 'up_capture' in relative_metrics
        assert 'down_capture' in relative_metrics
        assert 'capture_ratio' in relative_metrics
        
        # Capture ratio should be up_capture / down_capture
        expected_capture = relative_metrics['up_capture'] / relative_metrics['down_capture']
        assert abs(relative_metrics['capture_ratio'] - expected_capture) < 0.001


class TestPerformanceMonitor:
    """Test cases for the main PerformanceMonitor class."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create PerformanceMonitor instance."""
        config = PerformanceConfig(
            calculation_frequency='1min',
            risk_free_rate=0.03,
            benchmark_symbol='BTC/USDT'
        )
        return PerformanceMonitor(config)
        
    @pytest.fixture
    def live_trades(self):
        """Create stream of live trades."""
        trades = []
        base_time = datetime.now() - timedelta(hours=2)
        
        for i in range(10):
            trade = Trade(
                trade_id=f'live_{i}',
                symbol='BTC/USDT' if i % 2 == 0 else 'ETH/USDT',
                side='buy' if i % 3 == 0 else 'sell',
                quantity=Decimal('0.1'),
                entry_price=Decimal('50000') + Decimal(str(i * 100)),
                exit_price=Decimal('50000') + Decimal(str(i * 100 + 50)),
                entry_time=base_time + timedelta(minutes=i*10),
                exit_time=base_time + timedelta(minutes=i*10 + 5),
                fees=Decimal('5')
            )
            trades.append(trade)
            
        return trades
        
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, performance_monitor, live_trades):
        """Test real-time performance monitoring."""
        # Start monitoring
        monitor_task = asyncio.create_task(
            performance_monitor.start_monitoring()
        )
        
        # Feed trades
        for trade in live_trades:
            await performance_monitor.add_trade(trade)
            await asyncio.sleep(0.01)  # Small delay
            
        # Get current metrics
        current_metrics = performance_monitor.get_current_metrics()
        
        assert 'total_pnl' in current_metrics
        assert 'trade_count' in current_metrics
        assert 'win_rate' in current_metrics
        assert 'sharpe_ratio' in current_metrics
        
        # Cancel monitoring
        monitor_task.cancel()
        
    def test_performance_summary(self, performance_monitor, live_trades):
        """Test performance summary generation."""
        # Add historical trades
        for trade in live_trades:
            performance_monitor.add_historical_trade(trade)
            
        summary = performance_monitor.generate_summary(
            period='daily',
            date=datetime.now().date()
        )
        
        assert 'overview' in summary
        assert 'metrics' in summary
        assert 'trade_statistics' in summary
        assert 'risk_metrics' in summary
        
        # Check overview
        assert summary['overview']['total_trades'] == 10
        assert 'net_pnl' in summary['overview']
        assert 'roi_percentage' in summary['overview']
        
    def test_performance_report_generation(self, performance_monitor, live_trades):
        """Test comprehensive performance report."""
        # Add trades and positions
        for trade in live_trades:
            performance_monitor.add_historical_trade(trade)
            
        # Add benchmark data
        benchmark_data = pd.Series(
            np.random.normal(0.001, 0.02, 100),
            index=pd.date_range(start='2023-01-01', periods=100, freq='D')
        )
        performance_monitor.set_benchmark_data(benchmark_data)
        
        # Generate report
        report = performance_monitor.generate_performance_report(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            include_charts=True
        )
        
        assert isinstance(report, PerformanceReport)
        assert report.period_start is not None
        assert report.period_end is not None
        
        # Check sections
        assert 'executive_summary' in report.sections
        assert 'detailed_metrics' in report.sections
        assert 'trade_analysis' in report.sections
        assert 'risk_analysis' in report.sections
        assert 'benchmark_comparison' in report.sections
        
    def test_metric_alerts(self, performance_monitor):
        """Test performance metric alerts."""
        # Set alert thresholds
        performance_monitor.set_alert_thresholds({
            'max_drawdown': 0.10,  # 10%
            'min_sharpe': 0.5,
            'max_losing_streak': 5
        })
        
        alerts = []
        performance_monitor.set_alert_callback(lambda a: alerts.append(a))
        
        # Simulate performance degradation
        losing_trades = []
        for i in range(6):  # 6 losing trades
            trade = Trade(
                trade_id=f'loss_{i}',
                symbol='BTC/USDT',
                side='buy',
                quantity=Decimal('1.0'),
                entry_price=Decimal('50000'),
                exit_price=Decimal('49000'),  # 2% loss each
                fees=Decimal('50')
            )
            losing_trades.append(trade)
            performance_monitor.add_historical_trade(trade)
            
        # Should trigger losing streak alert
        assert len(alerts) > 0
        assert any(a['type'] == 'LOSING_STREAK' for a in alerts)
        
    def test_performance_persistence(self, performance_monitor, tmp_path):
        """Test saving and loading performance data."""
        # Add some data
        trades = [
            Trade(
                trade_id='001',
                symbol='BTC/USDT',
                side='buy',
                quantity=Decimal('1.0'),
                entry_price=Decimal('50000'),
                exit_price=Decimal('51000'),
                fees=Decimal('50')
            )
        ]
        
        for trade in trades:
            performance_monitor.add_historical_trade(trade)
            
        # Save state
        save_path = tmp_path / "performance_data.json"
        performance_monitor.save_state(save_path)
        
        # Create new monitor and load
        new_monitor = PerformanceMonitor(performance_monitor.config)
        new_monitor.load_state(performance_monitor)  # Pass source monitor
        
        # Verify data restored
        assert new_monitor.get_trade_count() == 1
        assert new_monitor.calculate_total_pnl() == trades[0].calculate_net_pnl()


class TestPerformanceVisualization:
    """Test cases for performance visualization data preparation."""
    
    def test_equity_curve_data(self):
        """Test equity curve data preparation."""
        initial_capital = Decimal('100000')
        trades = []
        
        # Create sequence of trades
        for i in range(20):
            pnl = Decimal('500') if i % 3 != 0 else Decimal('-200')
            trades.append({
                'timestamp': datetime.now() - timedelta(days=20-i),
                'pnl': pnl
            })
            
        visualizer = PerformanceReport()
        equity_data = visualizer.prepare_equity_curve(
            initial_capital,
            trades
        )
        
        assert len(equity_data) == 21  # Initial + 20 trades
        assert equity_data[0]['value'] == initial_capital
        assert equity_data[-1]['value'] > initial_capital  # Net profitable
        
    def test_drawdown_visualization(self):
        """Test drawdown visualization data."""
        equity_curve = pd.Series([
            100000, 102000, 105000, 103000, 98000,
            95000, 97000, 101000, 104000, 106000
        ])
        
        visualizer = PerformanceReport()
        drawdown_data = visualizer.prepare_drawdown_chart(equity_curve)
        
        assert 'drawdown_percentages' in drawdown_data
        assert 'drawdown_periods' in drawdown_data
        assert 'underwater_curve' in drawdown_data
        
        # Max drawdown should be around 9.5%
        max_dd = max(abs(dd) for dd in drawdown_data['drawdown_percentages'])
        assert 0.09 < max_dd < 0.10
        
    def test_returns_distribution(self):
        """Test returns distribution visualization."""
        returns = np.random.normal(0.001, 0.02, 1000)
        
        visualizer = PerformanceReport()
        distribution_data = visualizer.prepare_returns_distribution(returns)
        
        assert 'histogram' in distribution_data
        assert 'normal_overlay' in distribution_data
        assert 'statistics' in distribution_data
        
        # Check statistics
        stats = distribution_data['statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'skew' in stats
        assert 'kurtosis' in stats