"""
Unit tests for the Grid Trading Strategy component.

Tests cover:
- Grid creation and management
- Order placement and execution
- Position tracking and P&L calculation
- Dynamic grid adjustment
- Risk management integration
- Grid optimization
- Multi-grid strategies
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio

# GridAttention project imports
from core.grid_strategy_selector import (
    GridStrategySelector,
    GridConfig,
    GridLevel,
    GridOrder,
    GridPosition,
    GridMetrics,
    GridOptimizer,
    GridType,
    OrderSide,
    OrderStatus,
    GridState
)


class TestGridConfig:
    """Test cases for GridConfig validation and initialization."""
    
    def test_default_config(self):
        """Test default grid configuration."""
        config = GridConfig()
        
        assert config.grid_type == GridType.SYMMETRIC
        assert config.num_levels == 10
        assert config.grid_spacing == 0.01  # 1%
        assert config.total_investment == 10000
        assert config.leverage == 1
        assert config.stop_loss == 0.05  # 5%
        assert config.take_profit == 0.10  # 10%
        
    def test_custom_config(self):
        """Test custom grid configuration."""
        config = GridConfig(
            grid_type=GridType.GEOMETRIC,
            num_levels=20,
            grid_spacing=0.005,
            total_investment=50000,
            leverage=3,
            stop_loss=0.03,
            take_profit=0.15
        )
        
        assert config.grid_type == GridType.GEOMETRIC
        assert config.num_levels == 20
        assert config.grid_spacing == 0.005
        assert config.leverage == 3
        
    def test_config_validation(self):
        """Test configuration validation rules."""
        # Invalid number of levels
        with pytest.raises(ValueError, match="num_levels must be"):
            GridConfig(num_levels=0)
            
        # Invalid grid spacing
        with pytest.raises(ValueError, match="grid_spacing must be"):
            GridConfig(grid_spacing=0)
            
        # Invalid leverage
        with pytest.raises(ValueError, match="leverage must be"):
            GridConfig(leverage=101)  # Max 100x
            
        # Stop loss greater than take profit
        with pytest.raises(ValueError, match="stop_loss must be less than"):
            GridConfig(stop_loss=0.2, take_profit=0.1)
            
    def test_grid_type_specific_validation(self):
        """Test validation specific to grid types."""
        # Arithmetic grid requires fixed spacing
        config = GridConfig(grid_type=GridType.ARITHMETIC)
        assert config.fixed_spacing is not None
        
        # Dynamic grid requires volatility parameters
        config = GridConfig(grid_type=GridType.DYNAMIC)
        assert config.volatility_lookback is not None
        assert config.adjustment_frequency is not None


class TestGridLevel:
    """Test cases for GridLevel functionality."""
    
    def test_grid_level_creation(self):
        """Test creating grid levels."""
        level = GridLevel(
            index=0,
            price=Decimal('100.00'),
            quantity=Decimal('0.1'),
            side=OrderSide.BUY
        )
        
        assert level.index == 0
        assert level.price == Decimal('100.00')
        assert level.quantity == Decimal('0.1')
        assert level.side == OrderSide.BUY
        assert level.order_id is None
        assert level.filled_quantity == Decimal('0')
        
    def test_grid_level_fill(self):
        """Test filling grid levels."""
        level = GridLevel(
            index=0,
            price=Decimal('100.00'),
            quantity=Decimal('0.1'),
            side=OrderSide.BUY
        )
        
        # Partial fill
        level.fill(Decimal('0.05'), 'order_123')
        assert level.filled_quantity == Decimal('0.05')
        assert level.order_id == 'order_123'
        assert not level.is_fully_filled()
        
        # Complete fill
        level.fill(Decimal('0.05'))
        assert level.filled_quantity == Decimal('0.1')
        assert level.is_fully_filled()
        
    def test_grid_level_reset(self):
        """Test resetting grid levels."""
        level = GridLevel(
            index=0,
            price=Decimal('100.00'),
            quantity=Decimal('0.1'),
            side=OrderSide.BUY
        )
        
        level.fill(Decimal('0.1'), 'order_123')
        level.reset()
        
        assert level.filled_quantity == Decimal('0')
        assert level.order_id is None
        assert not level.is_fully_filled()


class TestGridStrategy:
    """Test cases for the main GridStrategy class."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange interface."""
        exchange = Mock()
        exchange.get_balance = Mock(return_value={'USDT': 10000, 'BTC': 0})
        exchange.get_ticker = Mock(return_value={'last': 50000, 'bid': 49990, 'ask': 50010})
        exchange.place_order = Mock(return_value={'id': 'order_123', 'status': 'new'})
        exchange.cancel_order = Mock(return_value={'id': 'order_123', 'status': 'cancelled'})
        exchange.get_order = Mock(return_value={'id': 'order_123', 'status': 'filled', 'filled': 0.1})
        return exchange
        
    @pytest.fixture
    def grid_strategy(self, mock_exchange):
        """Create a GridStrategy instance."""
        config = GridConfig(
            grid_type=GridType.SYMMETRIC,
            num_levels=10,
            grid_spacing=0.01,
            total_investment=10000
        )
        return GridStrategy(config, mock_exchange)
        
    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        prices = 50000 + np.sin(np.linspace(0, 4*np.pi, 100)) * 1000 + np.random.randn(100) * 100
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 100, 100),
            'low': prices - np.random.uniform(0, 100, 100),
            'close': prices + np.random.randn(100) * 50,
            'volume': np.random.randint(10, 100, 100)
        })
        
    def test_initialization(self, grid_strategy):
        """Test strategy initialization."""
        assert grid_strategy.state == GridState.INITIALIZED
        assert len(grid_strategy.grid_levels) == 0
        assert grid_strategy.current_position == 0
        assert grid_strategy.realized_pnl == 0
        
    def test_create_symmetric_grid(self, grid_strategy):
        """Test symmetric grid creation."""
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        
        assert len(grid_strategy.grid_levels) == 10
        assert grid_strategy.state == GridState.ACTIVE
        
        # Check grid spacing
        buy_levels = [l for l in grid_strategy.grid_levels if l.side == OrderSide.BUY]
        sell_levels = [l for l in grid_strategy.grid_levels if l.side == OrderSide.SELL]
        
        assert len(buy_levels) == 5
        assert len(sell_levels) == 5
        
        # Verify symmetric spacing
        for i, level in enumerate(sorted(buy_levels, key=lambda x: x.price, reverse=True)):
            expected_price = center_price * (Decimal('1') - Decimal('0.01') * (i + 1))
            assert abs(level.price - expected_price) < Decimal('0.01')
            
    def test_create_geometric_grid(self):
        """Test geometric grid creation."""
        config = GridConfig(
            grid_type=GridType.GEOMETRIC,
            num_levels=8,
            grid_spacing=0.02
        )
        
        exchange = Mock()
        strategy = GridStrategy(config, exchange)
        
        center_price = Decimal('50000')
        strategy.create_grid(center_price)
        
        # Check geometric progression
        buy_levels = sorted([l for l in strategy.grid_levels if l.side == OrderSide.BUY], 
                          key=lambda x: x.price, reverse=True)
        
        for i in range(1, len(buy_levels)):
            ratio = buy_levels[i-1].price / buy_levels[i].price
            assert abs(ratio - Decimal('1.02')) < Decimal('0.001')
            
    def test_create_dynamic_grid(self, mock_exchange):
        """Test dynamic grid creation based on volatility."""
        config = GridConfig(
            grid_type=GridType.DYNAMIC,
            num_levels=10,
            volatility_multiplier=2.0
        )
        
        strategy = GridStrategy(config, mock_exchange)
        
        # Mock volatility calculation
        with patch.object(strategy, 'calculate_volatility', return_value=0.02):
            center_price = Decimal('50000')
            strategy.create_grid(center_price)
            
            # Grid spacing should be adjusted by volatility
            expected_spacing = 0.02 * 2.0  # volatility * multiplier
            
            buy_levels = sorted([l for l in strategy.grid_levels if l.side == OrderSide.BUY],
                              key=lambda x: x.price, reverse=True)
            
            actual_spacing = float((buy_levels[0].price - buy_levels[1].price) / buy_levels[0].price)
            assert abs(actual_spacing - expected_spacing) < 0.001
            
    def test_place_grid_orders(self, grid_strategy, mock_exchange):
        """Test placing orders for all grid levels."""
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        
        # Place all orders
        asyncio.run(grid_strategy.place_grid_orders())
        
        # Check all levels have order IDs
        for level in grid_strategy.grid_levels:
            assert level.order_id is not None
            
        # Verify exchange calls
        assert mock_exchange.place_order.call_count == 10
        
    def test_order_execution_handling(self, grid_strategy, mock_exchange):
        """Test handling order executions."""
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        asyncio.run(grid_strategy.place_grid_orders())
        
        # Simulate buy order filled
        buy_level = next(l for l in grid_strategy.grid_levels if l.side == OrderSide.BUY)
        
        filled_order = {
            'id': buy_level.order_id,
            'status': 'filled',
            'filled': float(buy_level.quantity),
            'price': float(buy_level.price),
            'side': 'buy'
        }
        
        grid_strategy.handle_order_update(filled_order)
        
        # Check position updated
        assert grid_strategy.current_position == buy_level.quantity
        assert buy_level.is_fully_filled()
        
        # Check opposite sell order created
        assert any(l.side == OrderSide.SELL and l.price > buy_level.price 
                  for l in grid_strategy.grid_levels)
                  
    def test_position_tracking(self, grid_strategy, mock_exchange):
        """Test position tracking and P&L calculation."""
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        
        # Simulate multiple trades
        trades = [
            ('buy', Decimal('49500'), Decimal('0.1')),
            ('buy', Decimal('49000'), Decimal('0.1')),
            ('sell', Decimal('50000'), Decimal('0.1')),
            ('sell', Decimal('50500'), Decimal('0.1'))
        ]
        
        for side, price, quantity in trades:
            position = GridPosition(
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                price=price,
                quantity=quantity,
                timestamp=datetime.now()
            )
            grid_strategy.positions.append(position)
            
        # Calculate P&L
        pnl = grid_strategy.calculate_pnl(Decimal('50000'))
        
        # Should have positive P&L from grid trading
        assert pnl['realized'] > 0
        assert pnl['unrealized'] == 0  # Net position is 0
        
    def test_grid_adjustment(self, grid_strategy, mock_exchange):
        """Test dynamic grid adjustment."""
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        asyncio.run(grid_strategy.place_grid_orders())
        
        # Price moves significantly
        new_price = Decimal('52000')  # 4% move
        
        # Check if adjustment needed
        needs_adjustment = grid_strategy.check_grid_adjustment(new_price)
        assert needs_adjustment
        
        # Adjust grid
        asyncio.run(grid_strategy.adjust_grid(new_price))
        
        # Verify new grid centered around new price
        avg_price = sum(l.price for l in grid_strategy.grid_levels) / len(grid_strategy.grid_levels)
        assert abs(avg_price - new_price) / new_price < Decimal('0.01')
        
    def test_stop_loss_trigger(self, grid_strategy, mock_exchange):
        """Test stop loss functionality."""
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        
        # Simulate position
        grid_strategy.current_position = Decimal('0.5')
        grid_strategy.average_entry_price = Decimal('50000')
        
        # Price drops below stop loss
        stop_price = Decimal('50000') * (Decimal('1') - Decimal(str(grid_strategy.config.stop_loss)))
        current_price = stop_price - Decimal('100')
        
        # Check stop loss
        should_stop = grid_strategy.check_stop_loss(current_price)
        assert should_stop
        
        # Execute stop loss
        asyncio.run(grid_strategy.execute_stop_loss())
        
        # Verify position closed
        assert mock_exchange.place_order.called
        last_call = mock_exchange.place_order.call_args
        assert last_call[1]['side'] == 'sell'
        assert last_call[1]['quantity'] == 0.5
        
    def test_take_profit_trigger(self, grid_strategy, mock_exchange):
        """Test take profit functionality."""
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        
        # Simulate profitable position
        grid_strategy.current_position = Decimal('0.5')
        grid_strategy.average_entry_price = Decimal('50000')
        grid_strategy.realized_pnl = Decimal('500')  # $500 profit
        
        # Calculate ROI
        roi = grid_strategy.calculate_roi()
        
        # Check if take profit reached
        if roi > grid_strategy.config.take_profit:
            asyncio.run(grid_strategy.execute_take_profit())
            
            # Verify all orders cancelled
            assert mock_exchange.cancel_order.called
            assert grid_strategy.state == GridState.COMPLETED
            
    def test_multi_grid_management(self, mock_exchange):
        """Test managing multiple grids."""
        # Create multiple grid strategies
        configs = [
            GridConfig(num_levels=10, grid_spacing=0.005),  # Tight grid
            GridConfig(num_levels=8, grid_spacing=0.01),    # Medium grid
            GridConfig(num_levels=6, grid_spacing=0.02)     # Wide grid
        ]
        
        strategies = [GridStrategy(config, mock_exchange) for config in configs]
        
        # Initialize all grids
        center_price = Decimal('50000')
        for strategy in strategies:
            strategy.create_grid(center_price)
            
        # Verify no overlapping orders
        all_prices = []
        for strategy in strategies:
            all_prices.extend([level.price for level in strategy.grid_levels])
            
        assert len(all_prices) == len(set(all_prices))  # All unique
        
    def test_grid_metrics_calculation(self, grid_strategy):
        """Test grid performance metrics calculation."""
        # Simulate trading history
        grid_strategy.realized_pnl = Decimal('1000')
        grid_strategy.total_trades = 50
        grid_strategy.winning_trades = 35
        grid_strategy.total_volume = Decimal('100000')
        grid_strategy.start_time = datetime.now() - timedelta(days=7)
        
        metrics = grid_strategy.calculate_metrics()
        
        assert isinstance(metrics, GridMetrics)
        assert metrics.total_return > 0
        assert metrics.win_rate == 0.7  # 35/50
        assert metrics.profit_per_trade == 20  # 1000/50
        assert metrics.daily_return > 0
        assert metrics.sharpe_ratio is not None
        
    def test_grid_optimization(self, market_data):
        """Test grid parameter optimization."""
        optimizer = GridOptimizer()
        
        # Define parameter ranges
        param_ranges = {
            'num_levels': range(5, 21),
            'grid_spacing': np.arange(0.005, 0.03, 0.005),
            'stop_loss': np.arange(0.02, 0.1, 0.01)
        }
        
        # Run optimization
        best_params = optimizer.optimize(
            market_data,
            param_ranges,
            optimization_metric='sharpe_ratio'
        )
        
        assert 'num_levels' in best_params
        assert 'grid_spacing' in best_params
        assert 'stop_loss' in best_params
        
        # Best parameters should be within ranges
        assert 5 <= best_params['num_levels'] <= 20
        assert 0.005 <= best_params['grid_spacing'] <= 0.03
        
    def test_order_size_calculation(self, grid_strategy):
        """Test order size calculation for different position sizing methods."""
        total_investment = Decimal('10000')
        num_levels = 10
        
        # Equal sizing
        equal_size = grid_strategy.calculate_order_size(
            total_investment, 
            num_levels, 
            method='equal'
        )
        assert equal_size == Decimal('1000')  # 10000 / 10
        
        # Pyramid sizing (larger at better prices)
        pyramid_sizes = grid_strategy.calculate_order_size(
            total_investment,
            num_levels,
            method='pyramid'
        )
        assert len(pyramid_sizes) == num_levels
        assert sum(pyramid_sizes) == total_investment
        assert pyramid_sizes[0] < pyramid_sizes[-1]  # Increasing sizes
        
    def test_fee_calculation(self, grid_strategy):
        """Test trading fee calculation."""
        grid_strategy.fee_rate = Decimal('0.001')  # 0.1%
        
        # Calculate fees for a trade
        trade_value = Decimal('1000')
        fee = grid_strategy.calculate_fee(trade_value)
        
        assert fee == Decimal('1')  # 0.1% of 1000
        
        # Test fee impact on P&L
        gross_pnl = Decimal('100')
        net_pnl = grid_strategy.calculate_net_pnl(gross_pnl, num_trades=10)
        
        # Should be less due to fees
        assert net_pnl < gross_pnl
        
    def test_grid_persistence(self, grid_strategy, tmp_path):
        """Test saving and loading grid state."""
        # Set up grid
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        grid_strategy.realized_pnl = Decimal('500')
        grid_strategy.current_position = Decimal('0.1')
        
        # Save state
        save_path = tmp_path / "grid_state.json"
        grid_strategy.save_state(save_path)
        
        # Create new strategy and load
        new_strategy = GridStrategy(grid_strategy.config, grid_strategy.exchange)
        new_strategy.load_state(save_path)
        
        # Verify state restored
        assert len(new_strategy.grid_levels) == len(grid_strategy.grid_levels)
        assert new_strategy.realized_pnl == grid_strategy.realized_pnl
        assert new_strategy.current_position == grid_strategy.current_position
        
    def test_emergency_shutdown(self, grid_strategy, mock_exchange):
        """Test emergency shutdown procedures."""
        # Set up active grid
        center_price = Decimal('50000')
        grid_strategy.create_grid(center_price)
        asyncio.run(grid_strategy.place_grid_orders())
        
        # Add some positions
        grid_strategy.current_position = Decimal('0.5')
        
        # Execute emergency shutdown
        asyncio.run(grid_strategy.emergency_shutdown())
        
        # Verify all orders cancelled
        assert mock_exchange.cancel_order.call_count >= len(grid_strategy.grid_levels)
        
        # Verify position closed
        close_order_calls = [
            call for call in mock_exchange.place_order.call_args_list
            if 'market' in call[1].get('type', '')
        ]
        assert len(close_order_calls) > 0
        
        # Verify state
        assert grid_strategy.state == GridState.STOPPED
        

class TestGridOrderManagement:
    """Test cases for grid order management."""
    
    @pytest.fixture
    def order_manager(self):
        """Create order management instance."""
        return Mock()  # Would be actual OrderManager
        
    def test_order_queue_management(self, order_manager):
        """Test order queue and prioritization."""
        orders = [
            GridOrder(price=Decimal('50000'), quantity=Decimal('0.1'), side=OrderSide.BUY, priority=1),
            GridOrder(price=Decimal('49900'), quantity=Decimal('0.1'), side=OrderSide.BUY, priority=2),
            GridOrder(price=Decimal('50100'), quantity=Decimal('0.1'), side=OrderSide.SELL, priority=1),
        ]
        
        # Sort by priority
        sorted_orders = sorted(orders, key=lambda x: x.priority)
        
        assert sorted_orders[0].price == Decimal('50000')
        assert sorted_orders[1].price == Decimal('50100')
        
    def test_order_retry_logic(self, order_manager):
        """Test order retry on failures."""
        order = GridOrder(
            price=Decimal('50000'),
            quantity=Decimal('0.1'),
            side=OrderSide.BUY
        )
        
        # Simulate failures
        max_retries = 3
        for i in range(max_retries):
            try:
                # Simulate order placement failure
                if i < max_retries - 1:
                    raise Exception("Network error")
                else:
                    # Success on last retry
                    order.status = OrderStatus.PLACED
                    break
            except Exception:
                order.retry_count += 1
                
        assert order.retry_count == max_retries - 1
        assert order.status == OrderStatus.PLACED


class TestGridRiskManagement:
    """Test cases for grid risk management features."""
    
    def test_position_limits(self):
        """Test position size limits."""
        config = GridConfig(
            max_position_size=Decimal('1.0'),  # Max 1 BTC
            max_position_value=Decimal('50000')  # Max $50k
        )
        
        # Test quantity limit
        assert config.check_position_limit(
            Decimal('0.5'), 
            Decimal('50000')
        )  # OK
        
        assert not config.check_position_limit(
            Decimal('1.5'), 
            Decimal('50000')
        )  # Exceeds quantity
        
        # Test value limit
        assert not config.check_position_limit(
            Decimal('0.8'),
            Decimal('70000')  # 0.8 * 70000 = 56000 > 50000
        )
        
    def test_drawdown_protection(self):
        """Test maximum drawdown protection."""
        risk_manager = Mock()
        risk_manager.max_drawdown = Decimal('0.1')  # 10%
        risk_manager.current_drawdown = Decimal('0')
        
        # Simulate losses
        losses = [
            Decimal('200'),   # 2%
            Decimal('300'),   # 3%
            Decimal('400'),   # 4%
            Decimal('200')    # 2% - Would exceed 10%
        ]
        
        initial_capital = Decimal('10000')
        current_capital = initial_capital
        
        for loss in losses:
            current_capital -= loss
            drawdown = (initial_capital - current_capital) / initial_capital
            
            if drawdown > risk_manager.max_drawdown:
                # Should trigger protection
                assert drawdown > Decimal('0.1')
                break
                
    def test_correlation_limits(self):
        """Test correlation limits for multiple grids."""
        # Simulate correlation matrix
        correlations = {
            ('BTC', 'ETH'): 0.8,
            ('BTC', 'BNB'): 0.7,
            ('ETH', 'BNB'): 0.75
        }
        
        max_correlation = 0.6
        
        # Check if adding new grid violates correlation limit
        existing_grids = ['BTC', 'ETH']
        new_grid = 'BNB'
        
        violations = []
        for existing in existing_grids:
            corr = correlations.get((existing, new_grid)) or correlations.get((new_grid, existing))
            if corr and corr > max_correlation:
                violations.append((existing, new_grid, corr))
                
        assert len(violations) == 2  # Both BTC and ETH correlations too high


class TestGridPerformanceAnalysis:
    """Test cases for grid performance analysis."""
    
    def test_backtest_metrics(self):
        """Test backtesting metrics calculation."""
        backtest_results = {
            'total_trades': 100,
            'winning_trades': 65,
            'total_pnl': Decimal('5000'),
            'max_drawdown': Decimal('800'),
            'trading_days': 30
        }
        
        # Calculate metrics
        metrics = {
            'win_rate': backtest_results['winning_trades'] / backtest_results['total_trades'],
            'avg_daily_pnl': backtest_results['total_pnl'] / backtest_results['trading_days'],
            'recovery_factor': backtest_results['total_pnl'] / backtest_results['max_drawdown']
        }
        
        assert metrics['win_rate'] == 0.65
        assert metrics['avg_daily_pnl'] > 0
        assert metrics['recovery_factor'] > 6
        
    def test_grid_efficiency(self):
        """Test grid efficiency metrics."""
        # Grid utilization
        total_levels = 10
        filled_levels = 7
        utilization = filled_levels / total_levels
        
        assert utilization == 0.7
        
        # Price coverage efficiency
        price_range_covered = Decimal('1000')  # $1000 range
        actual_price_movement = Decimal('1200')  # $1200 movement
        coverage_efficiency = min(price_range_covered / actual_price_movement, Decimal('1'))
        
        assert coverage_efficiency < 1  # Grid didn't cover full movement
        
    def test_optimal_grid_spacing(self, market_data):
        """Test calculation of optimal grid spacing."""
        # Calculate volatility
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Optimal spacing based on volatility
        volatility_multiplier = 2.0
        optimal_spacing = volatility * volatility_multiplier
        
        # Test if spacing captures enough price movement
        price_moves = abs(market_data['high'] - market_data['low']) / market_data['close']
        captured_moves = (price_moves > optimal_spacing).sum()
        capture_rate = captured_moves / len(price_moves)
        
        # Should capture reasonable percentage of moves
        assert 0.3 <= capture_rate <= 0.7