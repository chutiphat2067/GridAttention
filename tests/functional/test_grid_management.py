# tests/functional/test_grid_management.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.grid_strategy_selector import GridStrategySelector
from core.market_regime_detector import MarketRegimeDetector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from infrastructure.event_bus import EventBus, Event, EventType


class GridType(Enum):
    """Types of grid configurations"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"


@dataclass
class GridLevel:
    """Individual grid level"""
    price: float
    side: str  # 'buy' or 'sell'
    size: float
    status: str  # 'pending', 'filled', 'cancelled'
    order_id: Optional[str] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None


@dataclass
class GridConfiguration:
    """Grid configuration parameters"""
    center_price: float
    levels: int
    spacing: float  # As percentage
    size_per_level: float
    grid_type: GridType
    bounds: Tuple[float, float]  # (lower, upper)
    

class TestGridManagement:
    """Test grid trading management functionality"""
    
    @pytest.fixture
    async def grid_system(self):
        """Create grid management system"""
        config = {
            'symbol': 'BTC/USDT',
            'grid': {
                'default_levels': 10,
                'min_levels': 5,
                'max_levels': 20,
                'default_spacing': 0.1,  # 0.1%
                'min_spacing': 0.05,
                'max_spacing': 0.5,
                'default_size': 0.01,
                'min_size': 0.001,
                'max_size': 0.1,
                'rebalance_threshold': 0.02  # 2%
            },
            'execution': {
                'order_timeout': 30,
                'retry_attempts': 3
            }
        }
        
        system = {
            'grid_selector': GridStrategySelector(config),
            'regime_detector': MarketRegimeDetector(config),
            'risk_manager': RiskManagementSystem(config),
            'execution_engine': ExecutionEngine(config),
            'event_bus': EventBus()
        }
        
        return system, config
    
    @pytest.mark.asyncio
    async def test_grid_initialization(self, grid_system):
        """Test initial grid setup"""
        system, config = grid_system
        
        # Initialize grid configuration
        grid_config = GridConfiguration(
            center_price=50000,
            levels=10,
            spacing=0.1,
            size_per_level=0.01,
            grid_type=GridType.SYMMETRIC,
            bounds=(49000, 51000)
        )
        
        # Create grid levels
        grid_levels = await system['grid_selector'].initialize_grid(grid_config)
        
        # Verify grid structure
        assert len(grid_levels) == grid_config.levels
        
        # Check symmetric distribution
        buy_levels = [l for l in grid_levels if l.side == 'buy']
        sell_levels = [l for l in grid_levels if l.side == 'sell']
        assert len(buy_levels) == len(sell_levels)
        
        # Verify spacing
        sorted_levels = sorted(grid_levels, key=lambda x: x.price)
        for i in range(1, len(sorted_levels)):
            spacing_pct = (sorted_levels[i].price - sorted_levels[i-1].price) / sorted_levels[i-1].price
            assert abs(spacing_pct - grid_config.spacing/100) < 0.001
        
        # Check bounds
        assert all(grid_config.bounds[0] <= l.price <= grid_config.bounds[1] for l in grid_levels)
    
    @pytest.mark.asyncio
    async def test_dynamic_grid_adjustment(self, grid_system):
        """Test dynamic grid adjustments based on market conditions"""
        system, config = grid_system
        
        # Initial grid
        initial_grid = GridConfiguration(
            center_price=50000,
            levels=10,
            spacing=0.1,
            size_per_level=0.01,
            grid_type=GridType.DYNAMIC,
            bounds=(49000, 51000)
        )
        
        grid_levels = await system['grid_selector'].initialize_grid(initial_grid)
        
        # Simulate market movements
        market_scenarios = [
            {'price': 50500, 'volatility': 0.002, 'regime': 'TRENDING_UP'},
            {'price': 50200, 'volatility': 0.005, 'regime': 'VOLATILE'},
            {'price': 49800, 'volatility': 0.001, 'regime': 'RANGING'}
        ]
        
        adjustments = []
        
        for scenario in market_scenarios:
            # Detect need for adjustment
            adjustment_needed = await system['grid_selector'].check_adjustment_needed(
                current_price=scenario['price'],
                grid_center=initial_grid.center_price,
                volatility=scenario['volatility']
            )
            
            if adjustment_needed:
                # Calculate new parameters
                new_params = await system['grid_selector'].calculate_dynamic_params(
                    current_price=scenario['price'],
                    volatility=scenario['volatility'],
                    regime=scenario['regime']
                )
                
                # Apply adjustment
                new_grid = await system['grid_selector'].adjust_grid(
                    current_levels=grid_levels,
                    new_params=new_params
                )
                
                adjustments.append({
                    'scenario': scenario,
                    'old_center': initial_grid.center_price,
                    'new_center': new_params['center_price'],
                    'spacing_change': new_params['spacing'] - initial_grid.spacing
                })
                
                grid_levels = new_grid
        
        # Verify adjustments
        assert len(adjustments) > 0
        
        # Check volatility-based spacing
        volatile_adj = [a for a in adjustments if a['scenario']['regime'] == 'VOLATILE']
        if volatile_adj:
            assert volatile_adj[0]['spacing_change'] > 0  # Wider spacing in volatile markets
        
        # Check trend following
        trend_adj = [a for a in adjustments if 'TRENDING' in a['scenario']['regime']]
        if trend_adj:
            price_move = trend_adj[0]['scenario']['price'] - initial_grid.center_price
            center_move = trend_adj[0]['new_center'] - trend_adj[0]['old_center']
            assert price_move * center_move > 0  # Center moves in price direction
    
    @pytest.mark.asyncio
    async def test_grid_order_management(self, grid_system):
        """Test grid order placement and management"""
        system, config = grid_system
        
        # Setup grid
        grid_config = GridConfiguration(
            center_price=50000,
            levels=6,
            spacing=0.1,
            size_per_level=0.01,
            grid_type=GridType.SYMMETRIC,
            bounds=(49700, 50300)
        )
        
        grid_levels = await system['grid_selector'].initialize_grid(grid_config)
        
        # Place grid orders
        placed_orders = []
        failed_orders = []
        
        for level in grid_levels:
            try:
                # Validate with risk manager
                risk_check = await system['risk_manager'].validate_grid_order({
                    'price': level.price,
                    'side': level.side,
                    'size': level.size
                })
                
                if risk_check['approved']:
                    # Place order
                    order = await system['execution_engine'].place_limit_order({
                        'symbol': config['symbol'],
                        'side': level.side,
                        'price': level.price,
                        'size': level.size,
                        'type': 'grid'
                    })
                    
                    level.order_id = order['id']
                    level.status = 'pending'
                    placed_orders.append(order)
                else:
                    failed_orders.append({
                        'level': level,
                        'reason': risk_check.get('reason', 'risk_rejected')
                    })
                    
            except Exception as e:
                failed_orders.append({
                    'level': level,
                    'reason': str(e)
                })
        
        # Verify order placement
        assert len(placed_orders) > 0
        assert len(placed_orders) + len(failed_orders) == len(grid_levels)
        
        # Simulate order fills
        filled_orders = []
        market_price = 50000
        
        for _ in range(5):  # Simulate 5 price movements
            # Random price movement
            market_price *= (1 + np.random.normal(0, 0.001))
            
            # Check for fills
            for level in grid_levels:
                if level.status == 'pending':
                    if (level.side == 'buy' and market_price <= level.price) or \
                       (level.side == 'sell' and market_price >= level.price):
                        # Order filled
                        level.status = 'filled'
                        level.filled_at = datetime.now()
                        level.filled_price = level.price
                        filled_orders.append(level)
        
        # Verify some orders filled
        assert len(filled_orders) > 0
        
        # Test order replacement for filled levels
        for filled_level in filled_orders:
            # Calculate replacement level
            if filled_level.side == 'buy':
                # Place new sell order above
                new_price = filled_level.price * (1 + grid_config.spacing/100)
                new_side = 'sell'
            else:
                # Place new buy order below
                new_price = filled_level.price * (1 - grid_config.spacing/100)
                new_side = 'buy'
            
            # Create replacement order
            replacement = GridLevel(
                price=new_price,
                side=new_side,
                size=filled_level.size,
                status='pending'
            )
            
            # Verify within bounds
            if grid_config.bounds[0] <= replacement.price <= grid_config.bounds[1]:
                grid_levels.append(replacement)
    
    @pytest.mark.asyncio
    async def test_grid_rebalancing(self, grid_system):
        """Test grid rebalancing mechanisms"""
        system, config = grid_system
        
        # Initial balanced grid
        grid_config = GridConfiguration(
            center_price=50000,
            levels=10,
            spacing=0.1,
            size_per_level=0.01,
            grid_type=GridType.SYMMETRIC,
            bounds=(49500, 50500)
        )
        
        grid_levels = await system['grid_selector'].initialize_grid(grid_config)
        
        # Track grid state
        grid_state = {
            'total_buy_orders': len([l for l in grid_levels if l.side == 'buy']),
            'total_sell_orders': len([l for l in grid_levels if l.side == 'sell']),
            'filled_buys': 0,
            'filled_sells': 0,
            'net_position': 0.0
        }
        
        # Simulate market movement causing imbalance
        # Strong upward movement
        for i in range(8):
            # Fill buy orders
            buy_levels = [l for l in grid_levels if l.side == 'buy' and l.status == 'pending']
            if buy_levels and i < 6:
                buy_levels[0].status = 'filled'
                grid_state['filled_buys'] += 1
                grid_state['net_position'] += buy_levels[0].size
        
        # Check if rebalancing needed
        rebalance_needed = await system['grid_selector'].check_rebalance_needed(
            grid_state=grid_state,
            current_price=50300
        )
        
        assert rebalance_needed
        
        # Execute rebalancing
        rebalance_actions = await system['grid_selector'].calculate_rebalance_actions(
            current_grid=grid_levels,
            grid_state=grid_state,
            current_price=50300
        )
        
        # Verify rebalancing actions
        assert 'cancel_orders' in rebalance_actions
        assert 'new_orders' in rebalance_actions
        assert 'adjust_center' in rebalance_actions
        
        # Apply rebalancing
        # Cancel far orders
        for order_id in rebalance_actions['cancel_orders']:
            level = next((l for l in grid_levels if l.order_id == order_id), None)
            if level:
                level.status = 'cancelled'
        
        # Add new orders
        for new_order in rebalance_actions['new_orders']:
            new_level = GridLevel(
                price=new_order['price'],
                side=new_order['side'],
                size=new_order['size'],
                status='pending'
            )
            grid_levels.append(new_level)
        
        # Verify balance restored
        active_buys = len([l for l in grid_levels if l.side == 'buy' and l.status == 'pending'])
        active_sells = len([l for l in grid_levels if l.side == 'sell' and l.status == 'pending'])
        
        assert abs(active_buys - active_sells) <= 2  # Reasonably balanced
    
    @pytest.mark.asyncio
    async def test_adaptive_grid_strategies(self, grid_system):
        """Test adaptive grid strategies for different market conditions"""
        system, config = grid_system
        
        # Define market conditions and expected adaptations
        market_conditions = [
            {
                'name': 'tight_range',
                'volatility': 0.0005,
                'price_range': (49900, 50100),
                'expected_strategy': 'narrow_grid'
            },
            {
                'name': 'trending',
                'volatility': 0.002,
                'trend': 0.001,  # 0.1% per period
                'expected_strategy': 'asymmetric_grid'
            },
            {
                'name': 'high_volatility',
                'volatility': 0.01,
                'price_range': (49000, 51000),
                'expected_strategy': 'wide_grid'
            },
            {
                'name': 'low_liquidity',
                'volatility': 0.003,
                'volume_factor': 0.3,
                'expected_strategy': 'reduced_size_grid'
            }
        ]
        
        strategies_applied = []
        
        for condition in market_conditions:
            # Analyze market condition
            market_analysis = await system['regime_detector'].analyze_conditions({
                'volatility': condition['volatility'],
                'trend': condition.get('trend', 0),
                'volume_factor': condition.get('volume_factor', 1.0)
            })
            
            # Select adaptive strategy
            strategy = await system['grid_selector'].select_adaptive_strategy(
                market_analysis=market_analysis,
                current_price=50000
            )
            
            strategies_applied.append({
                'condition': condition['name'],
                'strategy': strategy,
                'expected': condition['expected_strategy']
            })
            
            # Verify strategy matches expectation
            if condition['name'] == 'tight_range':
                assert strategy['spacing'] < config['grid']['default_spacing']
                assert strategy['levels'] >= config['grid']['default_levels']
                
            elif condition['name'] == 'trending':
                assert strategy['buy_bias'] != strategy['sell_bias']  # Asymmetric
                
            elif condition['name'] == 'high_volatility':
                assert strategy['spacing'] > config['grid']['default_spacing']
                assert strategy['stop_distance'] > 0  # Risk protection
                
            elif condition['name'] == 'low_liquidity':
                assert strategy['size_per_level'] < config['grid']['default_size']
    
    @pytest.mark.asyncio
    async def test_grid_performance_tracking(self, grid_system):
        """Test grid performance monitoring and optimization"""
        system, config = grid_system
        
        # Initialize grid with tracking
        grid_config = GridConfiguration(
            center_price=50000,
            levels=10,
            spacing=0.1,
            size_per_level=0.01,
            grid_type=GridType.SYMMETRIC,
            bounds=(49500, 50500)
        )
        
        grid_levels = await system['grid_selector'].initialize_grid(grid_config)
        
        # Performance tracking
        performance_metrics = {
            'fills': [],
            'pnl': 0.0,
            'win_rate': 0.0,
            'avg_fill_time': timedelta(),
            'grid_efficiency': 0.0
        }
        
        # Simulate trading session
        start_time = datetime.now()
        fills = []
        
        # Generate price movements
        prices = self._generate_price_series(
            start=50000,
            periods=100,
            volatility=0.002
        )
        
        for i, price in enumerate(prices):
            # Check for fills
            for level in grid_levels:
                if level.status == 'pending':
                    if self._check_fill_condition(level, price):
                        # Record fill
                        fill = {
                            'level': level,
                            'fill_price': price,
                            'fill_time': start_time + timedelta(minutes=i),
                            'slippage': abs(price - level.price)
                        }
                        fills.append(fill)
                        level.status = 'filled'
                        level.filled_at = fill['fill_time']
                        
                        # Calculate P&L for completed round trips
                        opposite_fills = [f for f in fills 
                                        if f['level'].side != level.side 
                                        and f['fill_time'] < fill['fill_time']]
                        
                        if opposite_fills:
                            # Match with last opposite fill
                            match = opposite_fills[-1]
                            if level.side == 'sell':
                                profit = (level.price - match['fill_price']) * level.size
                            else:
                                profit = (match['fill_price'] - level.price) * level.size
                            
                            performance_metrics['pnl'] += profit
        
        # Calculate performance metrics
        if fills:
            performance_metrics['fills'] = fills
            performance_metrics['win_rate'] = sum(1 for f in fills 
                                                 if self._is_profitable_fill(f, fills)) / len(fills)
            
            fill_times = [f['fill_time'] - start_time for f in fills]
            performance_metrics['avg_fill_time'] = sum(fill_times, timedelta()) / len(fill_times)
            
            # Grid efficiency: filled levels / total levels
            unique_levels_filled = len(set(f['level'].price for f in fills))
            performance_metrics['grid_efficiency'] = unique_levels_filled / len(grid_levels)
        
        # Verify performance tracking
        assert len(performance_metrics['fills']) > 0
        assert performance_metrics['grid_efficiency'] > 0
        
        # Test performance-based optimization
        if performance_metrics['win_rate'] < 0.4:
            # Poor performance - suggest adjustments
            optimization = await system['grid_selector'].optimize_from_performance(
                performance_metrics=performance_metrics,
                current_config=grid_config
            )
            
            assert 'suggested_spacing' in optimization
            assert 'suggested_levels' in optimization
            assert optimization['suggested_spacing'] != grid_config.spacing
    
    @pytest.mark.asyncio
    async def test_grid_risk_controls(self, grid_system):
        """Test grid-specific risk management"""
        system, config = grid_system
        
        # Setup grid with risk parameters
        risk_params = {
            'max_grid_exposure': 0.5,  # BTC
            'max_loss_per_grid': 100,  # USDT
            'position_limit': 0.3,  # BTC net position
            'drawdown_threshold': 0.02  # 2%
        }
        
        grid_config = GridConfiguration(
            center_price=50000,
            levels=15,
            spacing=0.1,
            size_per_level=0.05,  # Larger size to test limits
            grid_type=GridType.SYMMETRIC,
            bounds=(49000, 51000)
        )
        
        # Test exposure limits
        grid_levels = await system['grid_selector'].initialize_grid(grid_config)
        
        # Calculate total exposure
        total_buy_exposure = sum(l.price * l.size for l in grid_levels if l.side == 'buy')
        total_sell_exposure = sum(l.size for l in grid_levels if l.side == 'sell')
        
        # Risk manager should limit exposure
        validated_levels = []
        rejected_levels = []
        
        current_exposure = 0.0
        for level in grid_levels:
            exposure_after = current_exposure + level.size
            
            risk_check = await system['risk_manager'].validate_grid_exposure(
                current_exposure=current_exposure,
                additional_exposure=level.size,
                max_exposure=risk_params['max_grid_exposure']
            )
            
            if risk_check['approved']:
                validated_levels.append(level)
                current_exposure = exposure_after
            else:
                rejected_levels.append(level)
        
        # Verify risk limits enforced
        assert len(rejected_levels) > 0  # Some should be rejected
        assert current_exposure <= risk_params['max_grid_exposure']
        
        # Test drawdown protection
        # Simulate losses
        cumulative_loss = 0.0
        grid_active = True
        
        for i in range(10):
            # Simulate loss
            loss = np.random.uniform(10, 50)
            cumulative_loss += loss
            
            # Check drawdown
            drawdown = cumulative_loss / (grid_config.center_price * risk_params['max_grid_exposure'])
            
            if drawdown > risk_params['drawdown_threshold']:
                # Trigger protection
                protection_action = await system['risk_manager'].handle_grid_drawdown({
                    'drawdown': drawdown,
                    'cumulative_loss': cumulative_loss,
                    'active_levels': len(validated_levels)
                })
                
                if protection_action['action'] == 'halt_grid':
                    grid_active = False
                    break
                elif protection_action['action'] == 'reduce_size':
                    # Reduce all level sizes
                    for level in validated_levels:
                        level.size *= protection_action['size_multiplier']
        
        # Verify protection triggered if needed
        if cumulative_loss > risk_params['max_loss_per_grid']:
            assert not grid_active
    
    @pytest.mark.asyncio
    async def test_multi_grid_coordination(self, grid_system):
        """Test coordination of multiple grid strategies"""
        system, config = grid_system
        
        # Define multiple grids for different purposes
        grids = [
            {
                'name': 'main_grid',
                'config': GridConfiguration(
                    center_price=50000,
                    levels=10,
                    spacing=0.1,
                    size_per_level=0.01,
                    grid_type=GridType.SYMMETRIC,
                    bounds=(49500, 50500)
                ),
                'purpose': 'primary_trading'
            },
            {
                'name': 'scalp_grid',
                'config': GridConfiguration(
                    center_price=50000,
                    levels=5,
                    spacing=0.05,
                    size_per_level=0.02,
                    grid_type=GridType.SYMMETRIC,
                    bounds=(49900, 50100)
                ),
                'purpose': 'scalping'
            },
            {
                'name': 'hedge_grid',
                'config': GridConfiguration(
                    center_price=50000,
                    levels=4,
                    spacing=0.5,
                    size_per_level=0.05,
                    grid_type=GridType.ASYMMETRIC,
                    bounds=(49000, 51000)
                ),
                'purpose': 'hedging'
            }
        ]
        
        # Initialize all grids
        active_grids = {}
        total_exposure = 0.0
        
        for grid_def in grids:
            # Check if can add grid
            can_add = await system['risk_manager'].can_add_grid(
                existing_exposure=total_exposure,
                new_grid_exposure=grid_def['config'].levels * grid_def['config'].size_per_level
            )
            
            if can_add:
                grid_levels = await system['grid_selector'].initialize_grid(grid_def['config'])
                active_grids[grid_def['name']] = {
                    'definition': grid_def,
                    'levels': grid_levels,
                    'active': True
                }
                total_exposure += sum(l.size for l in grid_levels)
        
        # Verify grids initialized
        assert len(active_grids) >= 2  # At least 2 grids should be active
        
        # Test coordination rules
        # 1. No overlapping orders at same price
        all_order_prices = []
        for grid_name, grid_data in active_grids.items():
            for level in grid_data['levels']:
                all_order_prices.append((level.price, grid_name))
        
        # Check for conflicts
        price_counts = {}
        for price, grid in all_order_prices:
            if price not in price_counts:
                price_counts[price] = []
            price_counts[price].append(grid)
        
        conflicts = {p: grids for p, grids in price_counts.items() if len(grids) > 1}
        
        # Resolve conflicts
        if conflicts:
            for price, conflicting_grids in conflicts.items():
                # Prioritize based on purpose
                priority_order = ['primary_trading', 'hedging', 'scalping']
                
                for grid_name in conflicting_grids[1:]:  # Keep first, remove others
                    # Cancel conflicting order
                    grid_data = active_grids[grid_name]
                    for level in grid_data['levels']:
                        if level.price == price:
                            level.status = 'cancelled'
        
        # 2. Coordinate position limits
        net_positions = {}
        for grid_name, grid_data in active_grids.items():
            net_position = sum(l.size if l.side == 'buy' else -l.size 
                             for l in grid_data['levels'] 
                             if l.status == 'filled')
            net_positions[grid_name] = net_position
        
        total_net_position = sum(net_positions.values())
        
        # Verify within limits
        assert abs(total_net_position) <= config['grid']['max_size'] * 10
        
        # 3. Performance-based grid priority
        # Simulate performance
        grid_performance = {
            'main_grid': {'pnl': 100, 'win_rate': 0.6},
            'scalp_grid': {'pnl': -20, 'win_rate': 0.4},
            'hedge_grid': {'pnl': 50, 'win_rate': 0.5}
        }
        
        # Adjust grid resources based on performance
        resource_allocation = await system['grid_selector'].allocate_resources_by_performance(
            grid_performance=grid_performance,
            total_capital=10000
        )
        
        # Best performing grid should get more resources
        assert resource_allocation['main_grid'] > resource_allocation['scalp_grid']
    
    # Helper methods
    def _generate_price_series(self, start: float, periods: int, volatility: float) -> List[float]:
        """Generate price series for testing"""
        prices = [start]
        for _ in range(periods - 1):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        return prices
    
    def _check_fill_condition(self, level: GridLevel, market_price: float) -> bool:
        """Check if order should be filled"""
        if level.side == 'buy':
            return market_price <= level.price
        else:  # sell
            return market_price >= level.price
    
    def _is_profitable_fill(self, fill: Dict, all_fills: List[Dict]) -> bool:
        """Check if fill is part of profitable round trip"""
        level = fill['level']
        opposite_fills = [f for f in all_fills 
                         if f['level'].side != level.side 
                         and f['fill_time'] != fill['fill_time']]
        
        if not opposite_fills:
            return False
        
        # Find matching opposite fill
        for opp in opposite_fills:
            if level.side == 'sell':
                if fill['fill_price'] > opp['fill_price']:
                    return True
            else:  # buy
                if fill['fill_price'] < opp['fill_price']:
                    return True
        
        return False