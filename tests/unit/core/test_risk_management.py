"""
Unit tests for the Risk Management System.

Tests cover:
- Position sizing and Kelly criterion
- Stop loss and take profit management
- Portfolio risk metrics (VaR, CVaR, Sharpe)
- Correlation and diversification
- Drawdown protection
- Risk limits and alerts
- Dynamic risk adjustment
- Multi-asset portfolio management
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
from scipy import stats

# Assuming the module structure
from src.core.risk_management import (
    RiskManager,
    RiskConfig,
    RiskMetrics,
    PositionSizer,
    StopLossManager,
    PortfolioRisk,
    RiskAlert,
    RiskLevel,
    DrawdownMonitor,
    CorrelationTracker,
    RiskAdjustmentEngine
)


class TestRiskConfig:
    """Test cases for RiskConfig validation and initialization."""
    
    def test_default_config(self):
        """Test default risk configuration."""
        config = RiskConfig()
        
        assert config.max_position_size == 0.02  # 2% per position
        assert config.max_portfolio_risk == 0.06  # 6% total risk
        assert config.max_drawdown == 0.15  # 15% max drawdown
        assert config.stop_loss_pct == 0.02  # 2% stop loss
        assert config.risk_free_rate == 0.03  # 3% risk-free rate
        assert config.confidence_level == 0.95  # 95% VaR confidence
        assert config.correlation_window == 30  # 30 periods
        
    def test_custom_config(self):
        """Test custom risk configuration."""
        config = RiskConfig(
            max_position_size=0.01,
            max_portfolio_risk=0.05,
            max_drawdown=0.10,
            use_kelly_criterion=True,
            dynamic_adjustment=True
        )
        
        assert config.max_position_size == 0.01
        assert config.max_portfolio_risk == 0.05
        assert config.use_kelly_criterion is True
        assert config.dynamic_adjustment is True
        
    def test_config_validation(self):
        """Test configuration validation rules."""
        # Invalid position size
        with pytest.raises(ValueError, match="max_position_size"):
            RiskConfig(max_position_size=0.5)  # 50% too high
            
        # Invalid portfolio risk
        with pytest.raises(ValueError, match="max_portfolio_risk"):
            RiskConfig(max_portfolio_risk=1.5)  # > 100%
            
        # Invalid confidence level
        with pytest.raises(ValueError, match="confidence_level"):
            RiskConfig(confidence_level=1.5)  # Must be < 1
            
        # Stop loss greater than position size
        with pytest.raises(ValueError, match="stop_loss"):
            RiskConfig(max_position_size=0.02, stop_loss_pct=0.05)


class TestPositionSizer:
    """Test cases for position sizing algorithms."""
    
    @pytest.fixture
    def position_sizer(self):
        """Create a PositionSizer instance."""
        config = RiskConfig(
            max_position_size=0.02,
            use_kelly_criterion=True
        )
        return PositionSizer(config)
        
    @pytest.fixture
    def portfolio_data(self):
        """Create sample portfolio data."""
        return {
            'total_capital': Decimal('100000'),
            'available_capital': Decimal('50000'),
            'current_positions': 5,
            'max_positions': 10
        }
        
    def test_fixed_position_sizing(self, position_sizer, portfolio_data):
        """Test fixed percentage position sizing."""
        size = position_sizer.calculate_fixed_size(
            portfolio_data['total_capital'],
            risk_percentage=0.02
        )
        
        assert size == Decimal('2000')  # 2% of 100k
        
        # Test with available capital constraint
        size_constrained = position_sizer.calculate_fixed_size(
            portfolio_data['total_capital'],
            risk_percentage=0.02,
            available_capital=portfolio_data['available_capital']
        )
        
        assert size_constrained <= portfolio_data['available_capital']
        
    def test_kelly_criterion_sizing(self, position_sizer):
        """Test Kelly criterion position sizing."""
        # Historical win/loss data
        win_rate = 0.6
        avg_win = Decimal('1.5')  # 1.5R average win
        avg_loss = Decimal('1.0')  # 1R average loss
        
        kelly_fraction = position_sizer.calculate_kelly_fraction(
            win_rate, avg_win, avg_loss
        )
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        expected = (Decimal(str(win_rate)) * avg_win - Decimal(str(1-win_rate))) / avg_win
        
        assert abs(kelly_fraction - expected) < Decimal('0.001')
        
        # Apply Kelly fraction to capital
        capital = Decimal('100000')
        position_size = position_sizer.apply_kelly_criterion(
            capital, kelly_fraction, max_fraction=0.25
        )
        
        # Should be capped at 25% max
        assert position_size <= capital * Decimal('0.25')
        
    def test_volatility_adjusted_sizing(self, position_sizer):
        """Test volatility-adjusted position sizing."""
        base_size = Decimal('2000')
        current_volatility = 0.02  # 2% daily volatility
        target_volatility = 0.01   # 1% target
        
        adjusted_size = position_sizer.adjust_for_volatility(
            base_size,
            current_volatility,
            target_volatility
        )
        
        # Should reduce size when current vol > target vol
        assert adjusted_size < base_size
        assert adjusted_size == base_size * Decimal(str(target_volatility / current_volatility))
        
    def test_correlation_adjusted_sizing(self, position_sizer):
        """Test correlation-adjusted position sizing."""
        base_size = Decimal('2000')
        
        # Existing positions with correlations
        existing_positions = [
            {'symbol': 'BTC', 'size': Decimal('5000'), 'correlation': 0.8},
            {'symbol': 'ETH', 'size': Decimal('3000'), 'correlation': 0.6}
        ]
        
        adjusted_size = position_sizer.adjust_for_correlation(
            base_size,
            existing_positions,
            max_correlated_exposure=0.5
        )
        
        # Should reduce size due to high correlations
        assert adjusted_size < base_size
        
    def test_risk_parity_sizing(self, position_sizer):
        """Test risk parity position sizing."""
        assets = [
            {'symbol': 'BTC', 'volatility': 0.03, 'sharpe': 1.2},
            {'symbol': 'ETH', 'volatility': 0.04, 'sharpe': 1.0},
            {'symbol': 'BNB', 'volatility': 0.02, 'sharpe': 0.8}
        ]
        
        total_capital = Decimal('100000')
        
        allocations = position_sizer.calculate_risk_parity(
            assets,
            total_capital,
            target_risk=0.01
        )
        
        # Check equal risk contribution
        risk_contributions = []
        for asset, allocation in zip(assets, allocations):
            risk = allocation * Decimal(str(asset['volatility']))
            risk_contributions.append(risk)
            
        # All should contribute roughly equal risk
        avg_risk = sum(risk_contributions) / len(risk_contributions)
        for risk in risk_contributions:
            assert abs(risk - avg_risk) / avg_risk < 0.1  # Within 10%
            
    def test_maximum_position_limits(self, position_sizer, portfolio_data):
        """Test maximum position size limits."""
        requested_size = Decimal('10000')
        
        # Apply various limits
        final_size = position_sizer.apply_position_limits(
            requested_size,
            portfolio_data['total_capital'],
            max_position_pct=0.02,
            max_position_value=Decimal('5000'),
            available_capital=portfolio_data['available_capital']
        )
        
        # Should be limited by max percentage (2% of 100k = 2000)
        assert final_size == Decimal('2000')
        
    def test_dynamic_position_sizing(self, position_sizer):
        """Test dynamic position sizing based on market conditions."""
        base_size = Decimal('2000')
        
        # Market conditions
        market_conditions = {
            'trend_strength': 0.8,  # Strong trend
            'volatility_regime': 'low',
            'market_regime': 'bull',
            'drawdown_level': 0.05  # 5% current drawdown
        }
        
        dynamic_size = position_sizer.calculate_dynamic_size(
            base_size,
            market_conditions
        )
        
        # Should increase size in favorable conditions
        assert dynamic_size > base_size
        
        # Test with unfavorable conditions
        adverse_conditions = {
            'trend_strength': 0.2,
            'volatility_regime': 'high',
            'market_regime': 'bear',
            'drawdown_level': 0.12  # 12% drawdown
        }
        
        reduced_size = position_sizer.calculate_dynamic_size(
            base_size,
            adverse_conditions
        )
        
        # Should decrease size
        assert reduced_size < base_size


class TestStopLossManager:
    """Test cases for stop loss management."""
    
    @pytest.fixture
    def stop_loss_manager(self):
        """Create a StopLossManager instance."""
        config = RiskConfig(
            stop_loss_pct=0.02,
            use_trailing_stop=True,
            use_time_stop=True
        )
        return StopLossManager(config)
        
    def test_fixed_stop_loss(self, stop_loss_manager):
        """Test fixed percentage stop loss."""
        entry_price = Decimal('50000')
        stop_percentage = Decimal('0.02')  # 2%
        
        stop_price = stop_loss_manager.calculate_fixed_stop(
            entry_price,
            stop_percentage,
            side='long'
        )
        
        assert stop_price == Decimal('49000')  # 2% below entry
        
        # Test for short position
        stop_price_short = stop_loss_manager.calculate_fixed_stop(
            entry_price,
            stop_percentage,
            side='short'
        )
        
        assert stop_price_short == Decimal('51000')  # 2% above entry
        
    def test_atr_based_stop_loss(self, stop_loss_manager):
        """Test ATR-based stop loss."""
        entry_price = Decimal('50000')
        atr = Decimal('500')  # $500 ATR
        atr_multiplier = Decimal('2')
        
        stop_price = stop_loss_manager.calculate_atr_stop(
            entry_price,
            atr,
            atr_multiplier,
            side='long'
        )
        
        assert stop_price == Decimal('49000')  # Entry - (2 * ATR)
        
    def test_trailing_stop_loss(self, stop_loss_manager):
        """Test trailing stop loss functionality."""
        # Initialize position
        position = {
            'entry_price': Decimal('50000'),
            'current_stop': Decimal('49000'),
            'highest_price': Decimal('50000'),
            'trailing_percentage': Decimal('0.02')
        }
        
        # Price moves up
        new_price = Decimal('52000')
        
        updated_stop = stop_loss_manager.update_trailing_stop(
            position,
            new_price,
            side='long'
        )
        
        # Stop should trail up
        assert updated_stop > position['current_stop']
        assert updated_stop == Decimal('50960')  # 2% below 52000
        
        # Price moves down but above stop
        lower_price = Decimal('51000')
        
        updated_stop_2 = stop_loss_manager.update_trailing_stop(
            position,
            lower_price,
            side='long'
        )
        
        # Stop should not move down
        assert updated_stop_2 == updated_stop
        
    def test_time_based_stop(self, stop_loss_manager):
        """Test time-based stop loss."""
        position = {
            'entry_time': datetime.now() - timedelta(hours=48),
            'entry_price': Decimal('50000'),
            'current_price': Decimal('50500'),
            'time_limit_hours': 24
        }
        
        should_exit = stop_loss_manager.check_time_stop(position)
        
        # Should exit as position exceeded time limit
        assert should_exit is True
        
    def test_profit_target_management(self, stop_loss_manager):
        """Test profit target and R-multiple targets."""
        entry_price = Decimal('50000')
        stop_price = Decimal('49000')
        risk_amount = entry_price - stop_price
        
        # Calculate profit targets at different R-multiples
        targets = stop_loss_manager.calculate_profit_targets(
            entry_price,
            risk_amount,
            r_multiples=[1, 2, 3]
        )
        
        assert len(targets) == 3
        assert targets[0] == Decimal('51000')  # 1R target
        assert targets[1] == Decimal('52000')  # 2R target
        assert targets[2] == Decimal('53000')  # 3R target
        
    def test_break_even_stop(self, stop_loss_manager):
        """Test break-even stop adjustment."""
        position = {
            'entry_price': Decimal('50000'),
            'current_stop': Decimal('49000'),
            'current_price': Decimal('51000'),
            'break_even_threshold': Decimal('0.01')  # Move to BE at 1% profit
        }
        
        new_stop = stop_loss_manager.adjust_to_break_even(position)
        
        # Should move stop to break-even
        assert new_stop >= position['entry_price']
        
    def test_volatility_adjusted_stops(self, stop_loss_manager):
        """Test volatility-adjusted stop distances."""
        entry_price = Decimal('50000')
        
        # Different volatility scenarios
        low_vol_stop = stop_loss_manager.calculate_volatility_stop(
            entry_price,
            volatility=0.01,  # 1% daily vol
            confidence_level=2  # 2 standard deviations
        )
        
        high_vol_stop = stop_loss_manager.calculate_volatility_stop(
            entry_price,
            volatility=0.03,  # 3% daily vol
            confidence_level=2
        )
        
        # High volatility should have wider stop
        low_vol_distance = entry_price - low_vol_stop
        high_vol_distance = entry_price - high_vol_stop
        
        assert high_vol_distance > low_vol_distance


class TestPortfolioRisk:
    """Test cases for portfolio-wide risk management."""
    
    @pytest.fixture
    def portfolio_risk(self):
        """Create a PortfolioRisk instance."""
        config = RiskConfig(
            max_portfolio_risk=0.06,
            correlation_window=30
        )
        return PortfolioRisk(config)
        
    @pytest.fixture
    def portfolio_positions(self):
        """Create sample portfolio positions."""
        return [
            {
                'symbol': 'BTC',
                'size': Decimal('10000'),
                'entry_price': Decimal('50000'),
                'current_price': Decimal('51000'),
                'quantity': Decimal('0.2')
            },
            {
                'symbol': 'ETH',
                'size': Decimal('5000'),
                'entry_price': Decimal('3000'),
                'current_price': Decimal('3100'),
                'quantity': Decimal('1.67')
            },
            {
                'symbol': 'BNB',
                'size': Decimal('3000'),
                'entry_price': Decimal('400'),
                'current_price': Decimal('390'),
                'quantity': Decimal('7.5')
            }
        ]
        
    @pytest.fixture
    def returns_data(self):
        """Create sample returns data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Correlated returns
        btc_returns = np.random.normal(0.001, 0.03, 100)
        eth_returns = btc_returns * 0.8 + np.random.normal(0, 0.02, 100)
        bnb_returns = btc_returns * 0.6 + np.random.normal(0, 0.025, 100)
        
        return pd.DataFrame({
            'BTC': btc_returns,
            'ETH': eth_returns,
            'BNB': bnb_returns
        }, index=dates)
        
    def test_portfolio_var_calculation(self, portfolio_risk, portfolio_positions, returns_data):
        """Test Value at Risk (VaR) calculation."""
        # Calculate portfolio VaR
        var_95 = portfolio_risk.calculate_var(
            portfolio_positions,
            returns_data,
            confidence_level=0.95,
            time_horizon=1
        )
        
        assert var_95 < 0  # VaR is negative (loss)
        
        # Calculate for different time horizons
        var_weekly = portfolio_risk.calculate_var(
            portfolio_positions,
            returns_data,
            confidence_level=0.95,
            time_horizon=5
        )
        
        # Weekly VaR should be larger than daily
        assert abs(var_weekly) > abs(var_95)
        
    def test_portfolio_cvar_calculation(self, portfolio_risk, portfolio_positions, returns_data):
        """Test Conditional Value at Risk (CVaR) calculation."""
        var_95 = portfolio_risk.calculate_var(
            portfolio_positions,
            returns_data,
            confidence_level=0.95
        )
        
        cvar_95 = portfolio_risk.calculate_cvar(
            portfolio_positions,
            returns_data,
            confidence_level=0.95
        )
        
        # CVaR should be more conservative than VaR
        assert abs(cvar_95) > abs(var_95)
        
    def test_correlation_matrix(self, portfolio_risk, returns_data):
        """Test correlation matrix calculation."""
        correlation_matrix = portfolio_risk.calculate_correlation_matrix(returns_data)
        
        # Check matrix properties
        assert correlation_matrix.shape == (3, 3)
        
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(
            np.diag(correlation_matrix),
            np.ones(3)
        )
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            correlation_matrix,
            correlation_matrix.T
        )
        
        # Check expected correlations
        assert correlation_matrix.loc['BTC', 'ETH'] > 0.7  # High correlation
        assert correlation_matrix.loc['BTC', 'BNB'] > 0.5  # Moderate correlation
        
    def test_portfolio_beta(self, portfolio_risk, portfolio_positions, returns_data):
        """Test portfolio beta calculation."""
        # Add market returns
        market_returns = returns_data.mean(axis=1)  # Simple average as market
        
        portfolio_beta = portfolio_risk.calculate_portfolio_beta(
            portfolio_positions,
            returns_data,
            market_returns
        )
        
        # Beta should be close to 1 for this portfolio
        assert 0.8 <= portfolio_beta <= 1.2
        
    def test_risk_concentration(self, portfolio_risk, portfolio_positions):
        """Test risk concentration metrics."""
        concentration = portfolio_risk.calculate_concentration_risk(portfolio_positions)
        
        assert 'herfindahl_index' in concentration
        assert 'largest_position_pct' in concentration
        assert 'top_3_concentration' in concentration
        
        # Check calculations
        total_value = sum(p['size'] for p in portfolio_positions)
        largest_pct = max(p['size'] for p in portfolio_positions) / total_value
        
        assert concentration['largest_position_pct'] == float(largest_pct)
        
    def test_stress_testing(self, portfolio_risk, portfolio_positions):
        """Test portfolio stress testing."""
        # Define stress scenarios
        stress_scenarios = [
            {
                'name': 'Market Crash',
                'shocks': {'BTC': -0.20, 'ETH': -0.25, 'BNB': -0.30}
            },
            {
                'name': 'Flash Crash',
                'shocks': {'BTC': -0.10, 'ETH': -0.15, 'BNB': -0.12}
            },
            {
                'name': 'Black Swan',
                'shocks': {'BTC': -0.40, 'ETH': -0.45, 'BNB': -0.50}
            }
        ]
        
        stress_results = portfolio_risk.run_stress_tests(
            portfolio_positions,
            stress_scenarios
        )
        
        # Check results
        for scenario in stress_scenarios:
            result = stress_results[scenario['name']]
            assert 'portfolio_loss' in result
            assert 'portfolio_value_after' in result
            assert result['portfolio_loss'] < 0  # Losses are negative
            
    def test_margin_requirements(self, portfolio_risk, portfolio_positions):
        """Test margin requirement calculations."""
        # Calculate initial and maintenance margin
        margins = portfolio_risk.calculate_margin_requirements(
            portfolio_positions,
            initial_margin_pct=0.10,  # 10% initial
            maintenance_margin_pct=0.05  # 5% maintenance
        )
        
        assert 'initial_margin' in margins
        assert 'maintenance_margin' in margins
        assert 'current_margin_used' in margins
        assert 'margin_available' in margins
        
        # Initial margin should be higher than maintenance
        assert margins['initial_margin'] > margins['maintenance_margin']


class TestDrawdownMonitor:
    """Test cases for drawdown monitoring."""
    
    @pytest.fixture
    def drawdown_monitor(self):
        """Create a DrawdownMonitor instance."""
        return DrawdownMonitor(max_drawdown=0.15)  # 15% max
        
    def test_drawdown_calculation(self, drawdown_monitor):
        """Test drawdown calculation from equity curve."""
        # Create equity curve with drawdown
        equity_curve = [
            100000, 102000, 105000, 103000, 98000,  # Drawdown starts
            95000, 97000, 99000, 101000, 104000      # Recovery
        ]
        
        timestamps = pd.date_range(start='2023-01-01', periods=10, freq='D')
        equity_series = pd.Series(equity_curve, index=timestamps)
        
        drawdown_data = drawdown_monitor.calculate_drawdown(equity_series)
        
        assert 'current_drawdown' in drawdown_data
        assert 'max_drawdown' in drawdown_data
        assert 'drawdown_duration' in drawdown_data
        assert 'time_to_recovery' in drawdown_data
        
        # Max drawdown should be from 105000 to 95000
        expected_max_dd = (105000 - 95000) / 105000
        assert abs(drawdown_data['max_drawdown'] - expected_max_dd) < 0.001
        
    def test_drawdown_alerts(self, drawdown_monitor):
        """Test drawdown alert generation."""
        # Set alert thresholds
        drawdown_monitor.set_alert_thresholds([0.05, 0.10, 0.15])
        
        alerts = []
        
        # Simulate increasing drawdown
        drawdowns = [0.02, 0.04, 0.06, 0.08, 0.11, 0.13, 0.16]
        
        for dd in drawdowns:
            alert = drawdown_monitor.check_drawdown_alerts(dd)
            if alert:
                alerts.append(alert)
                
        # Should have triggered 3 alerts
        assert len(alerts) == 3
        assert alerts[0].level == RiskLevel.WARNING  # 5%
        assert alerts[1].level == RiskLevel.HIGH    # 10%
        assert alerts[2].level == RiskLevel.CRITICAL # 15%
        
    def test_recovery_metrics(self, drawdown_monitor):
        """Test drawdown recovery metrics."""
        # Historical drawdowns
        drawdown_history = [
            {
                'start_date': datetime(2023, 1, 1),
                'end_date': datetime(2023, 1, 15),
                'max_drawdown': 0.08,
                'recovery_date': datetime(2023, 1, 20)
            },
            {
                'start_date': datetime(2023, 2, 1),
                'end_date': datetime(2023, 2, 10),
                'max_drawdown': 0.12,
                'recovery_date': datetime(2023, 2, 25)
            }
        ]
        
        recovery_stats = drawdown_monitor.calculate_recovery_statistics(drawdown_history)
        
        assert 'avg_drawdown' in recovery_stats
        assert 'avg_recovery_time' in recovery_stats
        assert 'worst_drawdown' in recovery_stats
        assert 'recovery_rate' in recovery_stats
        
        # Check calculations
        assert recovery_stats['avg_drawdown'] == 0.10  # (0.08 + 0.12) / 2
        assert recovery_stats['worst_drawdown'] == 0.12


class TestRiskAdjustmentEngine:
    """Test cases for dynamic risk adjustment."""
    
    @pytest.fixture
    def risk_engine(self):
        """Create a RiskAdjustmentEngine instance."""
        config = RiskConfig(
            dynamic_adjustment=True,
            adjustment_frequency='daily'
        )
        return RiskAdjustmentEngine(config)
        
    def test_volatility_regime_detection(self, risk_engine):
        """Test volatility regime detection."""
        # Create returns with changing volatility
        low_vol_returns = np.random.normal(0, 0.01, 50)
        high_vol_returns = np.random.normal(0, 0.03, 50)
        returns = np.concatenate([low_vol_returns, high_vol_returns])
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns_series = pd.Series(returns, index=dates)
        
        # Detect regime at different points
        regime_start = risk_engine.detect_volatility_regime(returns_series[:30])
        regime_end = risk_engine.detect_volatility_regime(returns_series[70:])
        
        assert regime_start == 'low'
        assert regime_end == 'high'
        
    def test_risk_scaling(self, risk_engine):
        """Test risk parameter scaling based on conditions."""
        base_params = {
            'position_size': Decimal('2000'),
            'stop_loss': Decimal('0.02'),
            'max_positions': 10
        }
        
        # Scale for high volatility
        high_vol_params = risk_engine.scale_risk_parameters(
            base_params,
            volatility_regime='high',
            drawdown_level=0.08
        )
        
        # Should reduce risk
        assert high_vol_params['position_size'] < base_params['position_size']
        assert high_vol_params['stop_loss'] > base_params['stop_loss']  # Wider stops
        assert high_vol_params['max_positions'] < base_params['max_positions']
        
        # Scale for low volatility, no drawdown
        low_risk_params = risk_engine.scale_risk_parameters(
            base_params,
            volatility_regime='low',
            drawdown_level=0.02
        )
        
        # Can increase risk slightly
        assert low_risk_params['position_size'] >= base_params['position_size']
        
    def test_correlation_regime_adjustment(self, risk_engine):
        """Test adjustments based on correlation regimes."""
        # High correlation regime
        high_corr_adjustment = risk_engine.adjust_for_correlation_regime(
            avg_correlation=0.8,
            base_exposure=Decimal('10000')
        )
        
        # Should reduce exposure
        assert high_corr_adjustment < Decimal('10000')
        
        # Low correlation regime
        low_corr_adjustment = risk_engine.adjust_for_correlation_regime(
            avg_correlation=0.2,
            base_exposure=Decimal('10000')
        )
        
        # Can maintain or increase exposure
        assert low_corr_adjustment >= Decimal('10000')
        
    def test_adaptive_stop_loss(self, risk_engine):
        """Test adaptive stop loss based on market conditions."""
        base_stop = Decimal('0.02')  # 2%
        
        # Trending market
        trend_stop = risk_engine.calculate_adaptive_stop(
            base_stop,
            market_regime='trending',
            volatility=0.01,
            win_rate=0.65
        )
        
        # Can use tighter stops in trending market with good win rate
        assert trend_stop <= base_stop
        
        # Choppy market
        choppy_stop = risk_engine.calculate_adaptive_stop(
            base_stop,
            market_regime='ranging',
            volatility=0.02,
            win_rate=0.45
        )
        
        # Need wider stops in choppy market
        assert choppy_stop > base_stop


class TestRiskMetrics:
    """Test cases for risk metrics calculation."""
    
    @pytest.fixture
    def risk_metrics(self):
        """Create a RiskMetrics instance."""
        return RiskMetrics()
        
    def test_sharpe_ratio_calculation(self, risk_metrics):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.025])
        risk_free_rate = 0.03 / 252  # Daily risk-free rate
        
        sharpe = risk_metrics.calculate_sharpe_ratio(
            returns,
            risk_free_rate,
            periods_per_year=252
        )
        
        # Calculate expected
        excess_returns = returns - risk_free_rate
        expected_sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        assert abs(sharpe - expected_sharpe) < 0.001
        
    def test_sortino_ratio_calculation(self, risk_metrics):
        """Test Sortino ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.025])
        risk_free_rate = 0.03 / 252
        
        sortino = risk_metrics.calculate_sortino_ratio(
            returns,
            risk_free_rate,
            periods_per_year=252
        )
        
        # Sortino only uses downside deviation
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        # Should be higher than Sharpe for positive skew
        sharpe = risk_metrics.calculate_sharpe_ratio(returns, risk_free_rate)
        if returns.mean() > 0:
            assert sortino > sharpe
            
    def test_calmar_ratio_calculation(self, risk_metrics):
        """Test Calmar ratio calculation."""
        # Create equity curve
        equity_curve = pd.Series(
            [100000, 102000, 105000, 103000, 98000, 101000, 106000],
            index=pd.date_range(start='2023-01-01', periods=7, freq='D')
        )
        
        calmar = risk_metrics.calculate_calmar_ratio(equity_curve, periods_per_year=252)
        
        # Calculate components
        returns = equity_curve.pct_change().dropna()
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = risk_metrics.calculate_max_drawdown(equity_curve)
        
        expected_calmar = annual_return / abs(max_dd)
        assert abs(calmar - expected_calmar) < 0.001
        
    def test_risk_adjusted_returns(self, risk_metrics):
        """Test various risk-adjusted return metrics."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year of returns
        
        metrics = risk_metrics.calculate_all_metrics(returns)
        
        # Check all metrics are calculated
        expected_metrics = [
            'total_return', 'annual_return', 'volatility', 
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'var_95', 'cvar_95', 'win_rate', 'profit_factor'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert metrics[metric] is not None


class TestRiskIntegration:
    """Integration tests for the complete risk management system."""
    
    @pytest.fixture
    def risk_system(self):
        """Create a complete risk management system."""
        config = RiskConfig(
            max_position_size=0.02,
            max_portfolio_risk=0.06,
            max_drawdown=0.15,
            use_kelly_criterion=True,
            dynamic_adjustment=True
        )
        
        return RiskManager(config)
        
    @pytest.mark.asyncio
    async def test_pre_trade_risk_check(self, risk_system):
        """Test pre-trade risk validation."""
        # Proposed trade
        trade = {
            'symbol': 'BTC',
            'side': 'buy',
            'quantity': Decimal('0.5'),
            'price': Decimal('50000'),
            'value': Decimal('25000')
        }
        
        # Current portfolio state
        portfolio = {
            'total_capital': Decimal('100000'),
            'available_capital': Decimal('40000'),
            'current_drawdown': 0.08,
            'open_positions': 3
        }
        
        # Run pre-trade checks
        risk_check = await risk_system.validate_trade(trade, portfolio)
        
        assert 'approved' in risk_check
        assert 'adjusted_size' in risk_check
        assert 'risk_warnings' in risk_check
        assert 'position_score' in risk_check
        
        # Check if size was adjusted
        if not risk_check['approved']:
            assert risk_check['rejection_reason'] is not None
            
    @pytest.mark.asyncio  
    async def test_real_time_risk_monitoring(self, risk_system):
        """Test real-time risk monitoring."""
        # Start monitoring
        monitor_task = asyncio.create_task(
            risk_system.start_monitoring(interval=1)
        )
        
        # Simulate portfolio updates
        for i in range(5):
            update = {
                'timestamp': datetime.now(),
                'total_value': Decimal('100000') - (i * 1000),
                'positions': [],
                'realized_pnl': Decimal('-500') * i
            }
            
            risk_system.update_portfolio(update)
            await asyncio.sleep(0.1)
            
        # Check alerts generated
        alerts = risk_system.get_alerts()
        
        # Should have drawdown alerts
        assert len(alerts) > 0
        assert any(a.type == 'DRAWDOWN' for a in alerts)
        
        # Cancel monitoring
        monitor_task.cancel()
        
    def test_risk_report_generation(self, risk_system):
        """Test comprehensive risk report generation."""
        # Add sample data
        risk_system.add_historical_data(
            returns=pd.Series(np.random.normal(0.001, 0.02, 252)),
            positions=[],
            trades=[]
        )
        
        # Generate report
        report = risk_system.generate_risk_report()
        
        # Check report sections
        assert 'summary' in report
        assert 'portfolio_metrics' in report
        assert 'risk_metrics' in report
        assert 'exposure_analysis' in report
        assert 'recommendations' in report
        
        # Verify recommendations based on risk levels
        if report['risk_metrics']['current_drawdown'] > 0.10:
            assert 'reduce_position_sizes' in report['recommendations']