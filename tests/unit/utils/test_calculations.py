"""
Unit tests for Calculation Utilities.

Tests cover:
- Financial calculations (returns, volatility, ratios)
- Statistical calculations (mean, std, correlations)
- Technical indicators (SMA, EMA, RSI, MACD)
- Position calculations (P&L, fees, slippage)
- Risk calculations (VaR, CVaR, max drawdown)
- Portfolio calculations (weights, rebalancing)
- Mathematical utilities (rounding, precision)
- Performance optimizations
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import math
from unittest.mock import Mock, patch

# GridAttention project imports
from utils.calculations import (
    FinancialCalculator,
    StatisticalCalculator,
    TechnicalIndicators,
    PositionCalculator,
    RiskCalculator,
    PortfolioCalculator,
    MathUtils,
    CalculationError,
    PrecisionConfig
)


class TestFinancialCalculator:
    """Test cases for financial calculations."""
    
    @pytest.fixture
    def financial_calc(self):
        """Create FinancialCalculator instance."""
        return FinancialCalculator(precision=8)
        
    @pytest.fixture
    def price_series(self):
        """Create sample price series."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        # Generate realistic price movement
        returns = np.random.normal(0.0005, 0.02, 252)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)
        
    def test_simple_returns(self, financial_calc, price_series):
        """Test simple returns calculation."""
        returns = financial_calc.calculate_simple_returns(price_series)
        
        assert len(returns) == len(price_series) - 1
        assert returns.index[0] == price_series.index[1]
        
        # Verify calculation
        manual_return = (price_series.iloc[1] - price_series.iloc[0]) / price_series.iloc[0]
        assert abs(returns.iloc[0] - manual_return) < 1e-10
        
        # Test with different periods
        returns_weekly = financial_calc.calculate_simple_returns(price_series, periods=5)
        assert len(returns_weekly) == len(price_series) - 5
        
    def test_log_returns(self, financial_calc, price_series):
        """Test logarithmic returns calculation."""
        log_returns = financial_calc.calculate_log_returns(price_series)
        
        assert len(log_returns) == len(price_series) - 1
        
        # Verify calculation
        manual_log_return = np.log(price_series.iloc[1] / price_series.iloc[0])
        assert abs(log_returns.iloc[0] - manual_log_return) < 1e-10
        
        # Log returns should be additive
        cumulative_log_return = log_returns.sum()
        total_return = np.log(price_series.iloc[-1] / price_series.iloc[0])
        assert abs(cumulative_log_return - total_return) < 1e-6
        
    def test_annualized_return(self, financial_calc):
        """Test annualized return calculation."""
        # Daily returns for one year
        daily_returns = pd.Series(np.random.normal(0.0004, 0.01, 252))
        
        # Annualize daily returns
        annual_return = financial_calc.annualize_return(
            daily_returns, 
            periods_per_year=252
        )
        
        # Should be approximately mean daily return * 252
        expected = (1 + daily_returns.mean()) ** 252 - 1
        assert abs(annual_return - expected) < 0.001
        
        # Test with monthly returns
        monthly_returns = pd.Series(np.random.normal(0.01, 0.03, 12))
        annual_return_monthly = financial_calc.annualize_return(
            monthly_returns,
            periods_per_year=12
        )
        
        expected_monthly = (1 + monthly_returns.mean()) ** 12 - 1
        assert abs(annual_return_monthly - expected_monthly) < 0.001
        
    def test_volatility_calculation(self, financial_calc, price_series):
        """Test volatility calculation."""
        returns = financial_calc.calculate_simple_returns(price_series)
        
        # Calculate volatility
        volatility = financial_calc.calculate_volatility(
            returns,
            annualize=True,
            periods_per_year=252
        )
        
        # Manual calculation
        expected_vol = returns.std() * np.sqrt(252)
        assert abs(volatility - expected_vol) < 1e-6
        
        # Test rolling volatility
        rolling_vol = financial_calc.calculate_rolling_volatility(
            returns,
            window=30,
            annualize=True
        )
        
        assert len(rolling_vol) == len(returns) - 30 + 1
        assert all(v > 0 for v in rolling_vol)
        
    def test_sharpe_ratio(self, financial_calc):
        """Test Sharpe ratio calculation."""
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        risk_free_rate = 0.03  # 3% annual
        
        sharpe = financial_calc.calculate_sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=252
        )
        
        # Manual calculation
        excess_returns = returns - risk_free_rate / 252
        expected_sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        assert abs(sharpe - expected_sharpe) < 1e-6
        
        # Test with zero risk-free rate
        sharpe_zero_rf = financial_calc.calculate_sharpe_ratio(
            returns,
            risk_free_rate=0,
            periods_per_year=252
        )
        
        assert sharpe_zero_rf > sharpe  # Should be higher without risk-free rate
        
    def test_sortino_ratio(self, financial_calc):
        """Test Sortino ratio calculation."""
        # Returns with positive skew
        returns = pd.Series(
            np.concatenate([
                np.random.normal(0.002, 0.01, 200),  # Mostly positive
                np.random.normal(-0.001, 0.015, 52)  # Some negative
            ])
        )
        
        sortino = financial_calc.calculate_sortino_ratio(
            returns,
            target_return=0,
            periods_per_year=252
        )
        
        # Sortino should be higher than Sharpe for positively skewed returns
        sharpe = financial_calc.calculate_sharpe_ratio(returns, periods_per_year=252)
        assert sortino > sharpe
        
        # Test with different target return
        sortino_high_target = financial_calc.calculate_sortino_ratio(
            returns,
            target_return=0.001,
            periods_per_year=252
        )
        
        assert sortino_high_target < sortino  # Higher target = lower ratio
        
    def test_calmar_ratio(self, financial_calc):
        """Test Calmar ratio calculation."""
        # Create equity curve with drawdown
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        equity_curve = pd.Series(index=dates)
        equity_curve.iloc[0] = 100000
        
        for i in range(1, 252):
            if i < 100:
                # Growth phase
                equity_curve.iloc[i] = equity_curve.iloc[i-1] * 1.001
            elif i < 150:
                # Drawdown phase
                equity_curve.iloc[i] = equity_curve.iloc[i-1] * 0.998
            else:
                # Recovery phase
                equity_curve.iloc[i] = equity_curve.iloc[i-1] * 1.0015
                
        calmar = financial_calc.calculate_calmar_ratio(equity_curve)
        
        # Calculate components
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        max_dd = financial_calc.calculate_max_drawdown(equity_curve)
        
        expected_calmar = total_return / abs(max_dd)
        assert abs(calmar - expected_calmar) < 1e-6
        
    def test_compound_return(self, financial_calc):
        """Test compound return calculation."""
        # Series of returns
        returns = pd.Series([0.05, -0.02, 0.03, 0.01, -0.01])
        
        compound_return = financial_calc.calculate_compound_return(returns)
        
        # Manual calculation
        expected = np.prod(1 + returns) - 1
        assert abs(compound_return - expected) < 1e-10
        
        # Test with large returns series
        large_returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        compound_large = financial_calc.calculate_compound_return(large_returns)
        
        # Should handle numerical precision
        assert not np.isnan(compound_large)
        assert not np.isinf(compound_large)


class TestStatisticalCalculator:
    """Test cases for statistical calculations."""
    
    @pytest.fixture
    def stat_calc(self):
        """Create StatisticalCalculator instance."""
        return StatisticalCalculator()
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for statistical tests."""
        np.random.seed(42)
        return {
            'returns_a': pd.Series(np.random.normal(0.001, 0.02, 500)),
            'returns_b': pd.Series(np.random.normal(0.0005, 0.025, 500)),
            'multivariate': pd.DataFrame({
                'x1': np.random.normal(0, 1, 1000),
                'x2': np.random.normal(0, 1, 1000),
                'x3': np.random.normal(0, 1, 1000)
            })
        }
        
    def test_descriptive_statistics(self, stat_calc, sample_data):
        """Test descriptive statistics calculation."""
        stats = stat_calc.calculate_descriptive_stats(sample_data['returns_a'])
        
        required_stats = [
            'mean', 'median', 'std', 'variance', 'skewness', 'kurtosis',
            'min', 'max', 'q1', 'q3', 'iqr'
        ]
        
        for stat in required_stats:
            assert stat in stats
            assert not np.isnan(stats[stat])
            
        # Verify calculations
        assert abs(stats['mean'] - sample_data['returns_a'].mean()) < 1e-10
        assert abs(stats['std'] - sample_data['returns_a'].std()) < 1e-10
        
    def test_correlation_calculation(self, stat_calc, sample_data):
        """Test correlation calculations."""
        # Pearson correlation
        corr_pearson = stat_calc.calculate_correlation(
            sample_data['returns_a'],
            sample_data['returns_b'],
            method='pearson'
        )
        
        assert -1 <= corr_pearson <= 1
        
        # Spearman correlation
        corr_spearman = stat_calc.calculate_correlation(
            sample_data['returns_a'],
            sample_data['returns_b'],
            method='spearman'
        )
        
        assert -1 <= corr_spearman <= 1
        
        # Correlation matrix
        data = pd.DataFrame({
            'a': sample_data['returns_a'],
            'b': sample_data['returns_b']
        })
        
        corr_matrix = stat_calc.calculate_correlation_matrix(data)
        
        assert corr_matrix.shape == (2, 2)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
        assert corr_matrix.iloc[0, 1] == corr_matrix.iloc[1, 0]  # Symmetric
        
    def test_rolling_correlation(self, stat_calc, sample_data):
        """Test rolling correlation calculation."""
        window = 60
        
        rolling_corr = stat_calc.calculate_rolling_correlation(
            sample_data['returns_a'],
            sample_data['returns_b'],
            window=window
        )
        
        assert len(rolling_corr) == len(sample_data['returns_a']) - window + 1
        assert all(-1 <= c <= 1 for c in rolling_corr)
        
        # Test stability
        assert rolling_corr.std() < 0.5  # Should not vary too wildly
        
    def test_autocorrelation(self, stat_calc):
        """Test autocorrelation calculation."""
        # Create data with autocorrelation
        n = 500
        ar_data = np.zeros(n)
        ar_data[0] = np.random.normal()
        
        for i in range(1, n):
            ar_data[i] = 0.7 * ar_data[i-1] + np.random.normal(0, 0.5)
            
        ar_series = pd.Series(ar_data)
        
        # Calculate autocorrelation
        acf = stat_calc.calculate_autocorrelation(ar_series, lags=20)
        
        assert len(acf) == 21  # Including lag 0
        assert acf[0] == 1.0  # Lag 0 is always 1
        assert acf[1] > 0.5  # Should show positive autocorrelation
        
        # Test partial autocorrelation
        pacf = stat_calc.calculate_partial_autocorrelation(ar_series, lags=20)
        
        assert len(pacf) == 21
        assert abs(pacf[1] - 0.7) < 0.1  # Should be close to AR coefficient
        
    def test_distribution_tests(self, stat_calc, sample_data):
        """Test distribution testing functions."""
        # Test for normality
        normal_data = np.random.normal(0, 1, 1000)
        normality_result = stat_calc.test_normality(normal_data)
        
        assert 'statistic' in normality_result
        assert 'p_value' in normality_result
        assert 'is_normal' in normality_result
        assert normality_result['p_value'] > 0.05  # Should not reject normality
        
        # Test with non-normal data
        uniform_data = np.random.uniform(0, 1, 1000)
        non_normal_result = stat_calc.test_normality(uniform_data)
        
        assert non_normal_result['p_value'] < 0.05  # Should reject normality
        
    def test_outlier_detection(self, stat_calc):
        """Test statistical outlier detection."""
        # Data with outliers
        data = np.concatenate([
            np.random.normal(0, 1, 95),
            [10, -10, 15, -8, 12]  # Outliers
        ])
        
        # Z-score method
        outliers_zscore = stat_calc.detect_outliers(
            data,
            method='zscore',
            threshold=3
        )
        
        assert len(outliers_zscore) >= 3  # Should detect most outliers
        assert all(i >= 95 for i in outliers_zscore)  # Should be from end
        
        # IQR method
        outliers_iqr = stat_calc.detect_outliers(
            data,
            method='iqr',
            multiplier=1.5
        )
        
        assert len(outliers_iqr) >= 3
        
        # Isolation Forest method
        outliers_iforest = stat_calc.detect_outliers(
            data.reshape(-1, 1),
            method='isolation_forest',
            contamination=0.05
        )
        
        assert len(outliers_iforest) == int(len(data) * 0.05)
        
    def test_regression_analysis(self, stat_calc, sample_data):
        """Test regression analysis functions."""
        # Simple linear regression
        x = np.array(range(100))
        y = 2 * x + 5 + np.random.normal(0, 10, 100)
        
        regression_result = stat_calc.linear_regression(x, y)
        
        assert 'slope' in regression_result
        assert 'intercept' in regression_result
        assert 'r_squared' in regression_result
        assert 'p_value' in regression_result
        
        # Coefficients should be close to true values
        assert abs(regression_result['slope'] - 2) < 0.5
        assert abs(regression_result['intercept'] - 5) < 5
        
        # Multiple regression
        X = sample_data['multivariate']
        y = 2 * X['x1'] - 1.5 * X['x2'] + 0.5 * X['x3'] + np.random.normal(0, 1, 1000)
        
        multi_regression = stat_calc.multiple_regression(X, y)
        
        assert len(multi_regression['coefficients']) == 3
        assert multi_regression['r_squared'] > 0.8  # Should explain most variance


class TestTechnicalIndicators:
    """Test cases for technical indicator calculations."""
    
    @pytest.fixture
    def tech_indicators(self):
        """Create TechnicalIndicators instance."""
        return TechnicalIndicators()
        
    @pytest.fixture
    def ohlcv_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Generate realistic price data
        close_prices = 100
        data = {'date': dates}
        
        closes = []
        for _ in range(200):
            change = np.random.normal(0, 2)
            close_prices = max(close_prices + change, 10)  # Prevent negative
            closes.append(close_prices)
            
        data['close'] = closes
        data['open'] = [c + np.random.uniform(-1, 1) for c in closes]
        data['high'] = [max(o, c) + abs(np.random.normal(0, 0.5)) 
                       for o, c in zip(data['open'], closes)]
        data['low'] = [min(o, c) - abs(np.random.normal(0, 0.5)) 
                      for o, c in zip(data['open'], closes)]
        data['volume'] = np.random.randint(1000, 10000, 200)
        
        return pd.DataFrame(data).set_index('date')
        
    def test_moving_averages(self, tech_indicators, ohlcv_data):
        """Test moving average calculations."""
        # Simple Moving Average
        sma_20 = tech_indicators.calculate_sma(ohlcv_data['close'], period=20)
        
        assert len(sma_20) == len(ohlcv_data)
        assert sma_20[:19].isna().all()  # First 19 should be NaN
        assert not sma_20[19:].isna().any()  # Rest should have values
        
        # Verify calculation
        assert sma_20.iloc[19] == ohlcv_data['close'].iloc[:20].mean()
        
        # Exponential Moving Average
        ema_20 = tech_indicators.calculate_ema(ohlcv_data['close'], period=20)
        
        assert len(ema_20) == len(ohlcv_data)
        assert not ema_20[19:].isna().any()
        
        # EMA should react faster to recent prices
        if ohlcv_data['close'].iloc[-5:].mean() > ohlcv_data['close'].iloc[-25:-5].mean():
            assert ema_20.iloc[-1] > sma_20.iloc[-1]
            
        # Weighted Moving Average
        wma_20 = tech_indicators.calculate_wma(ohlcv_data['close'], period=20)
        
        assert len(wma_20) == len(ohlcv_data)
        assert not wma_20[19:].isna().any()
        
    def test_rsi_calculation(self, tech_indicators, ohlcv_data):
        """Test RSI (Relative Strength Index) calculation."""
        rsi = tech_indicators.calculate_rsi(ohlcv_data['close'], period=14)
        
        assert len(rsi) == len(ohlcv_data)
        assert rsi[:14].isna().all()  # First 14 should be NaN
        assert all(0 <= r <= 100 for r in rsi[14:] if not np.isnan(r))
        
        # Test extreme cases
        # All up moves
        up_prices = pd.Series(range(100, 200))
        rsi_up = tech_indicators.calculate_rsi(up_prices, period=14)
        assert all(r > 70 for r in rsi_up[14:])  # Should be overbought
        
        # All down moves
        down_prices = pd.Series(range(200, 100, -1))
        rsi_down = tech_indicators.calculate_rsi(down_prices, period=14)
        assert all(r < 30 for r in rsi_down[14:])  # Should be oversold
        
    def test_macd_calculation(self, tech_indicators, ohlcv_data):
        """Test MACD calculation."""
        macd_result = tech_indicators.calculate_macd(
            ohlcv_data['close'],
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result
        
        # Check lengths
        assert len(macd_result['macd']) == len(ohlcv_data)
        assert len(macd_result['signal']) == len(ohlcv_data)
        assert len(macd_result['histogram']) == len(ohlcv_data)
        
        # Verify histogram calculation
        histogram_calc = macd_result['macd'] - macd_result['signal']
        assert np.allclose(macd_result['histogram'].dropna(), 
                          histogram_calc.dropna(), rtol=1e-10)
                          
    def test_bollinger_bands(self, tech_indicators, ohlcv_data):
        """Test Bollinger Bands calculation."""
        bb_result = tech_indicators.calculate_bollinger_bands(
            ohlcv_data['close'],
            period=20,
            std_dev=2
        )
        
        assert 'middle' in bb_result
        assert 'upper' in bb_result
        assert 'lower' in bb_result
        assert 'bandwidth' in bb_result
        assert 'percent_b' in bb_result
        
        # Check relationships
        assert all(bb_result['upper'] > bb_result['middle'])
        assert all(bb_result['middle'] > bb_result['lower'])
        
        # Verify calculations
        sma = ohlcv_data['close'].rolling(20).mean()
        std = ohlcv_data['close'].rolling(20).std()
        
        assert np.allclose(bb_result['middle'].dropna(), sma.dropna(), rtol=1e-10)
        assert np.allclose(bb_result['upper'].dropna(), 
                          (sma + 2 * std).dropna(), rtol=1e-10)
                          
    def test_atr_calculation(self, tech_indicators, ohlcv_data):
        """Test Average True Range calculation."""
        atr = tech_indicators.calculate_atr(
            ohlcv_data['high'],
            ohlcv_data['low'],
            ohlcv_data['close'],
            period=14
        )
        
        assert len(atr) == len(ohlcv_data)
        assert atr[:14].isna().all()
        assert all(a > 0 for a in atr[14:])  # ATR should be positive
        
        # Test with increasing volatility
        volatile_data = ohlcv_data.copy()
        volatile_data['high'] = volatile_data['close'] * 1.05
        volatile_data['low'] = volatile_data['close'] * 0.95
        
        atr_volatile = tech_indicators.calculate_atr(
            volatile_data['high'],
            volatile_data['low'],
            volatile_data['close'],
            period=14
        )
        
        assert atr_volatile[14:].mean() > atr[14:].mean()  # Higher ATR
        
    def test_stochastic_oscillator(self, tech_indicators, ohlcv_data):
        """Test Stochastic Oscillator calculation."""
        stoch_result = tech_indicators.calculate_stochastic(
            ohlcv_data['high'],
            ohlcv_data['low'],
            ohlcv_data['close'],
            period=14,
            smooth_k=3,
            smooth_d=3
        )
        
        assert 'k' in stoch_result
        assert 'd' in stoch_result
        
        # Values should be between 0 and 100
        assert all(0 <= k <= 100 for k in stoch_result['k'].dropna())
        assert all(0 <= d <= 100 for d in stoch_result['d'].dropna())
        
        # D is smoothed version of K
        assert stoch_result['d'].iloc[-1] == stoch_result['k'].iloc[-3:].mean()
        
    def test_volume_indicators(self, tech_indicators, ohlcv_data):
        """Test volume-based indicators."""
        # On-Balance Volume
        obv = tech_indicators.calculate_obv(
            ohlcv_data['close'],
            ohlcv_data['volume']
        )
        
        assert len(obv) == len(ohlcv_data)
        assert obv.iloc[0] == ohlcv_data['volume'].iloc[0]  # First OBV equals first volume
        
        # Volume Weighted Average Price
        vwap = tech_indicators.calculate_vwap(
            ohlcv_data['high'],
            ohlcv_data['low'],
            ohlcv_data['close'],
            ohlcv_data['volume']
        )
        
        assert len(vwap) == len(ohlcv_data)
        assert all(vwap >= ohlcv_data['low'])
        assert all(vwap <= ohlcv_data['high'])
        
    def test_custom_indicators(self, tech_indicators):
        """Test custom indicator creation."""
        # Create custom indicator (e.g., price rate of change)
        def price_roc(prices, period):
            return ((prices - prices.shift(period)) / prices.shift(period)) * 100
            
        prices = pd.Series([100, 102, 101, 105, 103, 107, 110])
        
        roc = tech_indicators.add_custom_indicator(
            'ROC',
            price_roc,
            prices,
            period=3
        )
        
        assert len(roc) == len(prices)
        assert roc.iloc[3] == 5.0  # (105-100)/100 * 100


class TestPositionCalculator:
    """Test cases for position-related calculations."""
    
    @pytest.fixture
    def position_calc(self):
        """Create PositionCalculator instance."""
        return PositionCalculator(precision=8)
        
    def test_position_size_calculation(self, position_calc):
        """Test position size calculation."""
        # Fixed risk position sizing
        position_size = position_calc.calculate_position_size(
            account_balance=Decimal('100000'),
            risk_percentage=Decimal('0.02'),  # 2% risk
            stop_loss_pips=Decimal('50'),
            pip_value=Decimal('10')
        )
        
        # Risk amount = 100000 * 0.02 = 2000
        # Position size = 2000 / (50 * 10) = 4
        assert position_size == Decimal('4')
        
        # Kelly criterion sizing
        kelly_size = position_calc.calculate_kelly_position(
            account_balance=Decimal('100000'),
            win_probability=Decimal('0.6'),
            win_loss_ratio=Decimal('1.5'),
            max_position_pct=Decimal('0.25')
        )
        
        # Kelly % = (0.6 * 1.5 - 0.4) / 1.5 = 0.33...
        # But capped at 25%
        assert kelly_size == Decimal('25000')
        
    def test_pnl_calculation(self, position_calc):
        """Test P&L calculation for positions."""
        # Long position
        long_pnl = position_calc.calculate_pnl(
            position_type='long',
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            quantity=Decimal('0.5'),
            fees=Decimal('50')
        )
        
        assert long_pnl['gross_pnl'] == Decimal('500')  # (51000-50000) * 0.5
        assert long_pnl['net_pnl'] == Decimal('450')    # 500 - 50
        assert long_pnl['return_pct'] == Decimal('0.018')  # 450 / 25000
        
        # Short position
        short_pnl = position_calc.calculate_pnl(
            position_type='short',
            entry_price=Decimal('50000'),
            exit_price=Decimal('49000'),
            quantity=Decimal('0.5'),
            fees=Decimal('50')
        )
        
        assert short_pnl['gross_pnl'] == Decimal('500')  # (50000-49000) * 0.5
        assert short_pnl['net_pnl'] == Decimal('450')
        
        # Losing trade
        loss_pnl = position_calc.calculate_pnl(
            position_type='long',
            entry_price=Decimal('50000'),
            exit_price=Decimal('49000'),
            quantity=Decimal('0.5'),
            fees=Decimal('50')
        )
        
        assert loss_pnl['gross_pnl'] == Decimal('-500')
        assert loss_pnl['net_pnl'] == Decimal('-550')
        
    def test_fee_calculation(self, position_calc):
        """Test trading fee calculations."""
        # Percentage-based fees
        pct_fee = position_calc.calculate_trading_fee(
            notional_value=Decimal('10000'),
            fee_type='percentage',
            fee_rate=Decimal('0.001')  # 0.1%
        )
        
        assert pct_fee == Decimal('10')
        
        # Fixed fees
        fixed_fee = position_calc.calculate_trading_fee(
            notional_value=Decimal('10000'),
            fee_type='fixed',
            fee_amount=Decimal('5')
        )
        
        assert fixed_fee == Decimal('5')
        
        # Tiered fees
        tiered_fee = position_calc.calculate_trading_fee(
            notional_value=Decimal('1000000'),
            fee_type='tiered',
            fee_tiers=[
                {'threshold': Decimal('0'), 'rate': Decimal('0.001')},
                {'threshold': Decimal('100000'), 'rate': Decimal('0.0008')},
                {'threshold': Decimal('1000000'), 'rate': Decimal('0.0006')}
            ]
        )
        
        assert tiered_fee == Decimal('600')  # 1M * 0.0006
        
    def test_slippage_calculation(self, position_calc):
        """Test slippage calculation."""
        # Market order slippage
        slippage = position_calc.calculate_slippage(
            expected_price=Decimal('50000'),
            actual_price=Decimal('50050'),
            quantity=Decimal('1.0'),
            side='buy'
        )
        
        assert slippage['slippage_amount'] == Decimal('50')
        assert slippage['slippage_pct'] == Decimal('0.001')  # 0.1%
        assert slippage['slippage_cost'] == Decimal('50')
        
        # Sell order slippage (favorable)
        sell_slippage = position_calc.calculate_slippage(
            expected_price=Decimal('50000'),
            actual_price=Decimal('50050'),
            quantity=Decimal('1.0'),
            side='sell'
        )
        
        assert sell_slippage['slippage_amount'] == Decimal('50')
        assert sell_slippage['slippage_cost'] == Decimal('-50')  # Favorable
        
    def test_break_even_calculation(self, position_calc):
        """Test break-even price calculation."""
        # Long position break-even
        long_be = position_calc.calculate_break_even(
            position_type='long',
            entry_price=Decimal('50000'),
            quantity=Decimal('0.5'),
            total_fees=Decimal('100')
        )
        
        # Break-even = 50000 + (100 / 0.5) = 50200
        assert long_be == Decimal('50200')
        
        # Short position break-even
        short_be = position_calc.calculate_break_even(
            position_type='short',
            entry_price=Decimal('50000'),
            quantity=Decimal('0.5'),
            total_fees=Decimal('100')
        )
        
        # Break-even = 50000 - (100 / 0.5) = 49800
        assert short_be == Decimal('49800')
        
    def test_risk_reward_calculation(self, position_calc):
        """Test risk/reward ratio calculation."""
        rr_ratio = position_calc.calculate_risk_reward(
            entry_price=Decimal('50000'),
            stop_loss=Decimal('49000'),
            take_profit=Decimal('52000')
        )
        
        assert rr_ratio == Decimal('2.0')  # Risk: 1000, Reward: 2000
        
        # With fees
        rr_with_fees = position_calc.calculate_risk_reward(
            entry_price=Decimal('50000'),
            stop_loss=Decimal('49000'),
            take_profit=Decimal('52000'),
            entry_fees=Decimal('50'),
            exit_fees=Decimal('50')
        )
        
        assert rr_with_fees < Decimal('2.0')  # Fees reduce reward


class TestRiskCalculator:
    """Test cases for risk calculations."""
    
    @pytest.fixture
    def risk_calc(self):
        """Create RiskCalculator instance."""
        return RiskCalculator(confidence_level=0.95)
        
    @pytest.fixture
    def returns_data(self):
        """Create sample returns data."""
        np.random.seed(42)
        # Generate returns with fat tails
        normal_returns = np.random.normal(0.0005, 0.02, 800)
        extreme_returns = np.random.normal(0, 0.05, 200)
        
        returns = np.concatenate([normal_returns, extreme_returns])
        np.random.shuffle(returns)
        
        return pd.Series(returns)
        
    def test_var_calculation(self, risk_calc, returns_data):
        """Test Value at Risk calculation."""
        # Parametric VaR
        var_parametric = risk_calc.calculate_var(
            returns_data,
            method='parametric',
            confidence_level=0.95
        )
        
        assert var_parametric < 0  # VaR is a loss, should be negative
        
        # Historical VaR
        var_historical = risk_calc.calculate_var(
            returns_data,
            method='historical',
            confidence_level=0.95
        )
        
        assert var_historical < 0
        assert var_historical == np.percentile(returns_data, 5)
        
        # Monte Carlo VaR
        var_montecarlo = risk_calc.calculate_var(
            returns_data,
            method='montecarlo',
            confidence_level=0.95,
            n_simulations=10000
        )
        
        assert var_montecarlo < 0
        # Should be similar to other methods
        assert abs(var_montecarlo - var_historical) / abs(var_historical) < 0.2
        
    def test_cvar_calculation(self, risk_calc, returns_data):
        """Test Conditional Value at Risk (CVaR) calculation."""
        var_95 = risk_calc.calculate_var(
            returns_data,
            method='historical',
            confidence_level=0.95
        )
        
        cvar_95 = risk_calc.calculate_cvar(
            returns_data,
            confidence_level=0.95
        )
        
        assert cvar_95 < var_95  # CVaR should be more conservative
        
        # Manual verification
        worst_returns = returns_data[returns_data <= var_95]
        expected_cvar = worst_returns.mean()
        
        assert abs(cvar_95 - expected_cvar) < 1e-10
        
    def test_max_drawdown_calculation(self, risk_calc):
        """Test maximum drawdown calculation."""
        # Create price series with known drawdown
        prices = pd.Series([100, 110, 105, 95, 90, 95, 100, 105, 110, 100])
        
        dd_result = risk_calc.calculate_max_drawdown(prices)
        
        assert dd_result['max_drawdown'] == -0.1818  # (90-110)/110
        assert dd_result['peak_idx'] == 1
        assert dd_result['trough_idx'] == 4
        assert dd_result['recovery_idx'] == 8
        assert dd_result['duration'] == 7  # From peak to recovery
        
        # Test with no drawdown
        increasing_prices = pd.Series(range(100, 110))
        dd_no_loss = risk_calc.calculate_max_drawdown(increasing_prices)
        
        assert dd_no_loss['max_drawdown'] == 0
        
    def test_downside_deviation(self, risk_calc, returns_data):
        """Test downside deviation calculation."""
        # Downside deviation with zero threshold
        downside_dev = risk_calc.calculate_downside_deviation(
            returns_data,
            threshold=0
        )
        
        # Manual calculation
        negative_returns = returns_data[returns_data < 0]
        expected_dd = np.sqrt(np.mean(negative_returns ** 2))
        
        assert abs(downside_dev - expected_dd) < 1e-10
        
        # With custom threshold
        threshold = 0.001  # 0.1% daily return target
        downside_dev_custom = risk_calc.calculate_downside_deviation(
            returns_data,
            threshold=threshold
        )
        
        assert downside_dev_custom > downside_dev  # More returns below higher threshold
        
    def test_beta_calculation(self, risk_calc):
        """Test beta calculation."""
        # Create correlated returns
        market_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        asset_returns = 1.5 * market_returns + np.random.normal(0, 0.005, 252)
        
        beta = risk_calc.calculate_beta(asset_returns, market_returns)
        
        assert 1.3 < beta < 1.7  # Should be close to 1.5
        
        # Test with negative correlation
        inverse_returns = -0.8 * market_returns + np.random.normal(0, 0.005, 252)
        beta_negative = risk_calc.calculate_beta(inverse_returns, market_returns)
        
        assert beta_negative < 0
        
    def test_treynor_ratio(self, risk_calc):
        """Test Treynor ratio calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        market_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
        risk_free_rate = 0.03  # Annual
        
        treynor = risk_calc.calculate_treynor_ratio(
            returns,
            market_returns,
            risk_free_rate,
            periods_per_year=252
        )
        
        # Manual calculation
        excess_return = returns.mean() * 252 - risk_free_rate
        beta = risk_calc.calculate_beta(returns, market_returns)
        expected_treynor = excess_return / beta
        
        assert abs(treynor - expected_treynor) < 1e-6


class TestPortfolioCalculator:
    """Test cases for portfolio calculations."""
    
    @pytest.fixture
    def portfolio_calc(self):
        """Create PortfolioCalculator instance."""
        return PortfolioCalculator()
        
    @pytest.fixture
    def portfolio_data(self):
        """Create sample portfolio data."""
        positions = pd.DataFrame({
            'symbol': ['BTC', 'ETH', 'BNB', 'SOL'],
            'quantity': [0.5, 10.0, 50.0, 100.0],
            'entry_price': [50000, 3000, 400, 150],
            'current_price': [52000, 3200, 380, 160]
        })
        
        # Historical returns
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        returns = pd.DataFrame({
            'BTC': np.random.normal(0.001, 0.03, 100),
            'ETH': np.random.normal(0.0015, 0.04, 100),
            'BNB': np.random.normal(0.0008, 0.025, 100),
            'SOL': np.random.normal(0.002, 0.05, 100)
        }, index=dates)
        
        return {'positions': positions, 'returns': returns}
        
    def test_portfolio_weights(self, portfolio_calc, portfolio_data):
        """Test portfolio weight calculation."""
        weights = portfolio_calc.calculate_weights(portfolio_data['positions'])
        
        assert len(weights) == 4
        assert abs(weights.sum() - 1.0) < 1e-10  # Should sum to 1
        
        # Check individual weights
        total_value = (portfolio_data['positions']['quantity'] * 
                      portfolio_data['positions']['current_price']).sum()
        
        btc_value = 0.5 * 52000
        expected_btc_weight = btc_value / total_value
        
        assert abs(weights['BTC'] - expected_btc_weight) < 1e-10
        
    def test_portfolio_returns(self, portfolio_calc, portfolio_data):
        """Test portfolio returns calculation."""
        weights = portfolio_calc.calculate_weights(portfolio_data['positions'])
        
        # Calculate portfolio returns
        portfolio_returns = portfolio_calc.calculate_portfolio_returns(
            portfolio_data['returns'],
            weights
        )
        
        assert len(portfolio_returns) == len(portfolio_data['returns'])
        
        # Verify calculation for first day
        expected_return = sum(
            portfolio_data['returns'].iloc[0][symbol] * weight 
            for symbol, weight in weights.items()
        )
        
        assert abs(portfolio_returns.iloc[0] - expected_return) < 1e-10
        
    def test_portfolio_volatility(self, portfolio_calc, portfolio_data):
        """Test portfolio volatility calculation."""
        weights = portfolio_calc.calculate_weights(portfolio_data['positions'])
        
        # Calculate portfolio volatility
        portfolio_vol = portfolio_calc.calculate_portfolio_volatility(
            portfolio_data['returns'],
            weights,
            annualize=True,
            periods_per_year=252
        )
        
        assert portfolio_vol > 0
        assert portfolio_vol < 1.0  # Reasonable annual volatility
        
        # Should be less than weighted average of individual volatilities (diversification)
        individual_vols = portfolio_data['returns'].std() * np.sqrt(252)
        weighted_avg_vol = sum(
            individual_vols[symbol] * weight 
            for symbol, weight in weights.items()
        )
        
        assert portfolio_vol < weighted_avg_vol
        
    def test_portfolio_sharpe(self, portfolio_calc, portfolio_data):
        """Test portfolio Sharpe ratio calculation."""
        weights = portfolio_calc.calculate_weights(portfolio_data['positions'])
        
        sharpe = portfolio_calc.calculate_portfolio_sharpe(
            portfolio_data['returns'],
            weights,
            risk_free_rate=0.03,
            periods_per_year=252
        )
        
        # Verify against manual calculation
        portfolio_returns = portfolio_calc.calculate_portfolio_returns(
            portfolio_data['returns'],
            weights
        )
        
        excess_returns = portfolio_returns - 0.03/252
        expected_sharpe = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std()
        
        assert abs(sharpe - expected_sharpe) < 1e-6
        
    def test_efficient_frontier(self, portfolio_calc, portfolio_data):
        """Test efficient frontier calculation."""
        # Calculate efficient frontier
        frontier = portfolio_calc.calculate_efficient_frontier(
            portfolio_data['returns'],
            n_portfolios=20,
            risk_free_rate=0.03
        )
        
        assert len(frontier) == 20
        
        # Check that frontier is efficient (higher return for higher risk)
        frontier_sorted = sorted(frontier, key=lambda x: x['volatility'])
        
        for i in range(1, len(frontier_sorted)):
            assert frontier_sorted[i]['return'] >= frontier_sorted[i-1]['return']
            
        # Maximum Sharpe portfolio should be included
        max_sharpe_portfolio = max(frontier, key=lambda x: x['sharpe_ratio'])
        assert max_sharpe_portfolio['sharpe_ratio'] > 0
        
    def test_portfolio_rebalancing(self, portfolio_calc):
        """Test portfolio rebalancing calculations."""
        # Current positions
        current_positions = pd.DataFrame({
            'symbol': ['BTC', 'ETH', 'BNB'],
            'quantity': [0.5, 10.0, 50.0],
            'price': [52000, 3200, 400]
        })
        
        # Target weights
        target_weights = pd.Series({
            'BTC': 0.5,
            'ETH': 0.3,
            'BNB': 0.2
        })
        
        # Calculate rebalancing trades
        rebalance_trades = portfolio_calc.calculate_rebalancing_trades(
            current_positions,
            target_weights
        )
        
        assert len(rebalance_trades) == 3
        assert all(col in rebalance_trades.columns for col in 
                  ['current_value', 'target_value', 'trade_value', 'trade_quantity'])
                  
        # Verify total value preserved
        total_value = (current_positions['quantity'] * current_positions['price']).sum()
        target_total = rebalance_trades['target_value'].sum()
        
        assert abs(total_value - target_total) < 1  # Allow small rounding


class TestMathUtils:
    """Test cases for mathematical utility functions."""
    
    @pytest.fixture
    def math_utils(self):
        """Create MathUtils instance."""
        return MathUtils()
        
    def test_rounding_functions(self, math_utils):
        """Test various rounding functions."""
        # Round to decimal places
        assert math_utils.round_decimal(Decimal('1.23456'), 2) == Decimal('1.23')
        assert math_utils.round_decimal(Decimal('1.23556'), 2) == Decimal('1.24')
        
        # Round to significant figures
        assert math_utils.round_significant(1234.5678, 3) == 1230
        assert math_utils.round_significant(0.0012345, 3) == 0.00123
        
        # Round to nearest multiple
        assert math_utils.round_to_multiple(17, 5) == 15
        assert math_utils.round_to_multiple(18, 5) == 20
        assert math_utils.round_to_multiple(Decimal('0.123'), Decimal('0.05')) == Decimal('0.10')
        
    def test_precision_handling(self, math_utils):
        """Test precision handling for financial calculations."""
        # Test decimal context
        with math_utils.decimal_context(precision=4, rounding=ROUND_HALF_UP):
            result = Decimal('1.23456') * Decimal('2.34567')
            assert len(str(result).split('.')[-1]) <= 4
            
        # Test float to decimal conversion
        float_val = 1.23456789
        decimal_val = math_utils.float_to_decimal(float_val, precision=6)
        
        assert isinstance(decimal_val, Decimal)
        assert decimal_val == Decimal('1.234568')
        
    def test_percentage_calculations(self, math_utils):
        """Test percentage calculation utilities."""
        # Percentage change
        assert math_utils.percentage_change(100, 110) == 10.0
        assert math_utils.percentage_change(100, 90) == -10.0
        assert math_utils.percentage_change(0, 100) == float('inf')
        
        # Percentage of total
        assert math_utils.percentage_of_total(25, 100) == 25.0
        assert math_utils.percentage_of_total(0, 100) == 0.0
        
        # Compound percentage
        changes = [0.1, -0.05, 0.08, -0.02]  # 10%, -5%, 8%, -2%
        compound = math_utils.compound_percentage(changes)
        expected = (1.1 * 0.95 * 1.08 * 0.98) - 1
        
        assert abs(compound - expected) < 1e-10
        
    def test_statistical_utilities(self, math_utils):
        """Test statistical utility functions."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Moving average
        ma = math_utils.moving_average(data, window=3)
        assert ma == [2, 3, 4, 5, 6, 7, 8, 9]
        
        # Exponential smoothing
        smoothed = math_utils.exponential_smoothing(data, alpha=0.3)
        assert len(smoothed) == len(data)
        assert smoothed[0] == data[0]
        
        # Z-score normalization
        z_scores = math_utils.z_score_normalize(data)
        assert abs(np.mean(z_scores)) < 1e-10  # Mean should be 0
        assert abs(np.std(z_scores) - 1) < 1e-10  # Std should be 1
        
    def test_interpolation(self, math_utils):
        """Test interpolation functions."""
        # Linear interpolation
        x = [0, 1, 2, 3, 4]
        y = [0, 2, 4, 6, 8]
        
        assert math_utils.linear_interpolate(x, y, 1.5) == 3.0
        assert math_utils.linear_interpolate(x, y, 2.5) == 5.0
        
        # Extrapolation
        assert math_utils.linear_interpolate(x, y, 5, extrapolate=True) == 10.0
        
        # Log interpolation
        x_log = [1, 10, 100]
        y_log = [0, 1, 2]
        
        assert abs(math_utils.log_interpolate(x_log, y_log, 31.6) - 1.5) < 0.1
        
    def test_array_operations(self, math_utils):
        """Test array operation utilities."""
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([2, 2, 2, 2, 2])
        
        # Element-wise operations
        assert np.array_equal(
            math_utils.safe_divide(arr1, arr2),
            np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        )
        
        # Safe divide with zeros
        arr3 = np.array([1, 0, 3, 0, 5])
        result = math_utils.safe_divide(arr1, arr3, fill_value=0)
        
        assert result[1] == 0  # Division by zero handled
        assert result[3] == 0
        
        # Cumulative product
        cum_prod = math_utils.cumulative_product([1.1, 0.9, 1.05, 0.95])
        expected = [1.1, 0.99, 1.0395, 0.98752]
        
        assert np.allclose(cum_prod, expected)
        
    def test_financial_math(self, math_utils):
        """Test financial mathematics functions."""
        # Present value
        pv = math_utils.present_value(
            future_value=1000,
            rate=0.05,
            periods=3
        )
        
        assert abs(pv - 863.84) < 0.01
        
        # Future value
        fv = math_utils.future_value(
            present_value=1000,
            rate=0.05,
            periods=3
        )
        
        assert abs(fv - 1157.63) < 0.01
        
        # Compound annual growth rate
        cagr = math_utils.calculate_cagr(
            beginning_value=1000,
            ending_value=1500,
            periods=3
        )
        
        assert abs(cagr - 0.1447) < 0.0001  # ~14.47%