# tests/functional/test_position_handling.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor
from infrastructure.event_bus import EventBus, Event, EventType


class PositionStatus(Enum):
    """Position status types"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"
    LIQUIDATED = "liquidated"


class PositionType(Enum):
    """Position types"""
    SPOT = "spot"
    MARGIN = "margin"
    GRID = "grid"
    HEDGE = "hedge"


@dataclass
class Position:
    """Trading position"""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float
    position_type: PositionType
    status: PositionStatus
    opened_at: datetime
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionUpdate:
    """Position update event"""
    position_id: str
    update_type: str
    timestamp: datetime
    data: Dict[str, Any]


class TestPositionHandling:
    """Test position management functionality"""
    
    @pytest.fixture
    async def position_system(self):
        """Create position management system"""
        config = {
            'symbol': 'BTC/USDT',
            'position': {
                'max_positions': 10,
                'max_position_size': 1.0,
                'max_total_exposure': 5.0,
                'default_stop_loss': 0.02,  # 2%
                'default_take_profit': 0.05,  # 5%
                'trailing_stop': True,
                'partial_close_enabled': True
            },
            'risk': {
                'max_loss_per_position': 0.01,  # 1% of capital
                'max_correlation': 0.7,
                'position_sizing_method': 'kelly'
            }
        }
        
        system = {
            'risk_manager': RiskManagementSystem(config),
            'execution_engine': ExecutionEngine(config),
            'performance_monitor': PerformanceMonitor(config),
            'event_bus': EventBus(),
            'positions': {},  # Position tracker
            'position_history': []
        }
        
        return system, config
    
    @pytest.mark.asyncio
    async def test_position_opening(self, position_system):
        """Test opening various types of positions"""
        system, config = position_system
        
        # Test different position types
        position_requests = [
            {
                'type': PositionType.SPOT,
                'side': 'long',
                'size': 0.1,
                'entry_price': 50000
            },
            {
                'type': PositionType.GRID,
                'side': 'long',
                'size': 0.05,
                'entry_price': 49900,
                'metadata': {'grid_level': 1}
            },
            {
                'type': PositionType.HEDGE,
                'side': 'short',
                'size': 0.08,
                'entry_price': 50100,
                'metadata': {'hedge_ratio': 0.8}
            }
        ]
        
        opened_positions = []
        
        for request in position_requests:
            # Risk validation
            risk_check = await system['risk_manager'].validate_new_position({
                'symbol': config['symbol'],
                'side': request['side'],
                'size': request['size'],
                'price': request['entry_price'],
                'type': request['type'].value,
                'existing_positions': list(system['positions'].values())
            })
            
            if risk_check['approved']:
                # Execute position opening
                execution_result = await system['execution_engine'].open_position({
                    'symbol': config['symbol'],
                    'side': request['side'],
                    'size': request['size'],
                    'type': 'market',  # or limit
                    'price': request['entry_price']
                })
                
                if execution_result['success']:
                    # Create position object
                    position = Position(
                        id=execution_result['order_id'],
                        symbol=config['symbol'],
                        side=request['side'],
                        entry_price=execution_result['fill_price'],
                        size=request['size'],
                        position_type=request['type'],
                        status=PositionStatus.OPEN,
                        opened_at=datetime.now(),
                        fees_paid=execution_result.get('fees', 0),
                        metadata=request.get('metadata', {})
                    )
                    
                    # Set stops if configured
                    if config['position']['default_stop_loss']:
                        position.stop_loss = position.entry_price * (
                            1 - config['position']['default_stop_loss'] 
                            if position.side == 'long' 
                            else 1 + config['position']['default_stop_loss']
                        )
                    
                    if config['position']['default_take_profit']:
                        position.take_profit = position.entry_price * (
                            1 + config['position']['default_take_profit']
                            if position.side == 'long'
                            else 1 - config['position']['default_take_profit']
                        )
                    
                    system['positions'][position.id] = position
                    opened_positions.append(position)
                    
                    # Emit position opened event
                    await system['event_bus'].publish(Event(
                        type=EventType.POSITION_OPENED,
                        data={'position': position.__dict__}
                    ))
        
        # Verify positions opened
        assert len(opened_positions) > 0
        assert len(system['positions']) == len(opened_positions)
        
        # Check position diversity
        position_types = set(p.position_type for p in opened_positions)
        assert len(position_types) >= 2  # Multiple types
        
        # Verify stops set
        for position in opened_positions:
            assert position.stop_loss is not None
            assert position.take_profit is not None
    
    @pytest.mark.asyncio
    async def test_position_monitoring_and_updates(self, position_system):
        """Test real-time position monitoring and P&L updates"""
        system, config = position_system
        
        # Create test positions
        test_positions = [
            Position(
                id='pos_1',
                symbol=config['symbol'],
                side='long',
                entry_price=50000,
                size=0.1,
                position_type=PositionType.SPOT,
                status=PositionStatus.OPEN,
                opened_at=datetime.now() - timedelta(minutes=30)
            ),
            Position(
                id='pos_2',
                symbol=config['symbol'],
                side='short',
                entry_price=50200,
                size=0.05,
                position_type=PositionType.HEDGE,
                status=PositionStatus.OPEN,
                opened_at=datetime.now() - timedelta(minutes=20)
            )
        ]
        
        for pos in test_positions:
            system['positions'][pos.id] = pos
        
        # Simulate price movements
        price_updates = [
            {'price': 50100, 'timestamp': datetime.now()},
            {'price': 50300, 'timestamp': datetime.now() + timedelta(minutes=5)},
            {'price': 50150, 'timestamp': datetime.now() + timedelta(minutes=10)},
            {'price': 49800, 'timestamp': datetime.now() + timedelta(minutes=15)}
        ]
        
        position_updates = []
        
        for update in price_updates:
            current_price = update['price']
            
            # Update all positions
            for position in system['positions'].values():
                if position.status == PositionStatus.OPEN:
                    # Calculate unrealized P&L
                    if position.side == 'long':
                        position.unrealized_pnl = (current_price - position.entry_price) * position.size
                    else:  # short
                        position.unrealized_pnl = (position.entry_price - current_price) * position.size
                    
                    # Check stop loss
                    if position.stop_loss:
                        if (position.side == 'long' and current_price <= position.stop_loss) or \
                           (position.side == 'short' and current_price >= position.stop_loss):
                            # Trigger stop loss
                            await self._close_position(system, position, current_price, 'stop_loss')
                    
                    # Check take profit
                    elif position.take_profit:
                        if (position.side == 'long' and current_price >= position.take_profit) or \
                           (position.side == 'short' and current_price <= position.take_profit):
                            # Trigger take profit
                            await self._close_position(system, position, current_price, 'take_profit')
                    
                    # Trailing stop update
                    elif config['position']['trailing_stop'] and position.unrealized_pnl > 0:
                        await self._update_trailing_stop(position, current_price, config)
                    
                    # Record update
                    position_updates.append(PositionUpdate(
                        position_id=position.id,
                        update_type='pnl_update',
                        timestamp=update['timestamp'],
                        data={
                            'price': current_price,
                            'unrealized_pnl': position.unrealized_pnl,
                            'pnl_percent': position.unrealized_pnl / (position.entry_price * position.size) * 100
                        }
                    ))
        
        # Verify monitoring
        assert len(position_updates) > 0
        
        # Check P&L calculations
        long_positions = [p for p in system['positions'].values() if p.side == 'long']
        short_positions = [p for p in system['positions'].values() if p.side == 'short']
        
        # At price 49800 (last update), longs should have negative P&L
        for pos in long_positions:
            if pos.status == PositionStatus.OPEN:
                assert pos.unrealized_pnl < 0
        
        # Shorts should have positive P&L
        for pos in short_positions:
            if pos.status == PositionStatus.OPEN:
                assert pos.unrealized_pnl > 0
    
    @pytest.mark.asyncio
    async def test_position_closing_strategies(self, position_system):
        """Test various position closing strategies"""
        system, config = position_system
        
        # Create position for testing
        position = Position(
            id='test_pos',
            symbol=config['symbol'],
            side='long',
            entry_price=50000,
            size=0.2,
            position_type=PositionType.SPOT,
            status=PositionStatus.OPEN,
            opened_at=datetime.now() - timedelta(hours=1),
            stop_loss=49000,
            take_profit=52000
        )
        system['positions'][position.id] = position
        
        # Test different closing strategies
        
        # 1. Market close
        market_close_result = await system['execution_engine'].close_position({
            'position_id': position.id,
            'close_type': 'market',
            'size': position.size  # Full close
        })
        
        if market_close_result['success']:
            position.status = PositionStatus.CLOSED
            position.exit_price = market_close_result['fill_price']
            position.closed_at = datetime.now()
            position.realized_pnl = (position.exit_price - position.entry_price) * position.size
            position.fees_paid += market_close_result.get('fees', 0)
        
        # Reset for next test
        position.status = PositionStatus.OPEN
        
        # 2. Partial close
        if config['position']['partial_close_enabled']:
            partial_size = position.size * 0.5  # Close 50%
            
            partial_close_result = await system['execution_engine'].close_position({
                'position_id': position.id,
                'close_type': 'market',
                'size': partial_size
            })
            
            if partial_close_result['success']:
                # Update position
                position.status = PositionStatus.PARTIALLY_CLOSED
                position.size -= partial_size
                partial_pnl = (partial_close_result['fill_price'] - position.entry_price) * partial_size
                position.realized_pnl += partial_pnl
                
                # Record partial close
                system['position_history'].append({
                    'position_id': position.id,
                    'action': 'partial_close',
                    'size': partial_size,
                    'price': partial_close_result['fill_price'],
                    'pnl': partial_pnl,
                    'timestamp': datetime.now()
                })
        
        # 3. Scaled exit
        scale_out_levels = [
            {'price': 50500, 'percent': 0.3},  # Take 30% at +1%
            {'price': 51000, 'percent': 0.3},  # Take 30% at +2%
            {'price': 51500, 'percent': 0.4}   # Take 40% at +3%
        ]
        
        remaining_size = position.size
        for level in scale_out_levels:
            if remaining_size > 0:
                exit_size = position.size * level['percent']
                
                # Place limit order for scaled exit
                scale_order = await system['execution_engine'].place_limit_order({
                    'symbol': position.symbol,
                    'side': 'sell' if position.side == 'long' else 'buy',
                    'size': exit_size,
                    'price': level['price'],
                    'type': 'scale_out',
                    'position_id': position.id
                })
                
                if scale_order['success']:
                    position.metadata['scale_orders'] = position.metadata.get('scale_orders', [])
                    position.metadata['scale_orders'].append(scale_order['order_id'])
        
        # Verify closing strategies work
        assert position.status in [PositionStatus.CLOSED, PositionStatus.PARTIALLY_CLOSED]
        assert len(system['position_history']) > 0
    
    @pytest.mark.asyncio
    async def test_position_risk_management(self, position_system):
        """Test position-level risk management"""
        system, config = position_system
        
        # Create positions with different risk profiles
        positions = [
            Position(
                id='high_risk',
                symbol=config['symbol'],
                side='long',
                entry_price=50000,
                size=0.5,  # Large position
                position_type=PositionType.MARGIN,
                status=PositionStatus.OPEN,
                opened_at=datetime.now(),
                metadata={'leverage': 5}
            ),
            Position(
                id='correlated_1',
                symbol='ETH/USDT',
                side='long',
                entry_price=3000,
                size=2.0,
                position_type=PositionType.SPOT,
                status=PositionStatus.OPEN,
                opened_at=datetime.now()
            ),
            Position(
                id='correlated_2',
                symbol='BTC/USDT',
                side='long',
                entry_price=49500,
                size=0.1,
                position_type=PositionType.SPOT,
                status=PositionStatus.OPEN,
                opened_at=datetime.now()
            )
        ]
        
        for pos in positions:
            system['positions'][pos.id] = pos
        
        # Test risk checks
        risk_alerts = []
        
        # 1. Position size risk
        for position in positions:
            size_risk = await system['risk_manager'].check_position_size_risk(
                position=position,
                max_size=config['position']['max_position_size']
            )
            
            if size_risk['risk_level'] == 'high':
                risk_alerts.append({
                    'position_id': position.id,
                    'risk_type': 'size',
                    'action': size_risk['recommended_action']
                })
        
        # 2. Correlation risk
        correlation_matrix = {
            ('BTC/USDT', 'ETH/USDT'): 0.85,  # High correlation
            ('BTC/USDT', 'BTC/USDT'): 1.0
        }
        
        correlation_risk = await system['risk_manager'].check_correlation_risk(
            positions=list(system['positions'].values()),
            correlation_matrix=correlation_matrix,
            max_correlation=config['risk']['max_correlation']
        )
        
        if correlation_risk['high_correlation_pairs']:
            for pair in correlation_risk['high_correlation_pairs']:
                risk_alerts.append({
                    'positions': pair,
                    'risk_type': 'correlation',
                    'correlation': correlation_matrix.get(tuple(sorted([p.symbol for p in pair])), 0),
                    'action': 'reduce_exposure'
                })
        
        # 3. Drawdown protection
        # Simulate adverse price movement
        adverse_prices = {
            'BTC/USDT': 48000,  # -4% from 50000
            'ETH/USDT': 2850    # -5% from 3000
        }
        
        total_unrealized_loss = 0
        for position in positions:
            if position.symbol in adverse_prices:
                current_price = adverse_prices[position.symbol]
                if position.side == 'long':
                    loss = (current_price - position.entry_price) * position.size
                else:
                    loss = (position.entry_price - current_price) * position.size
                
                total_unrealized_loss += loss
        
        # Check if emergency close needed
        if abs(total_unrealized_loss) > 1000:  # Threshold
            emergency_action = await system['risk_manager'].handle_emergency_close({
                'total_loss': total_unrealized_loss,
                'positions': positions,
                'reason': 'max_drawdown_exceeded'
            })
            
            if emergency_action['execute']:
                for position_id in emergency_action['positions_to_close']:
                    await self._close_position(
                        system, 
                        system['positions'][position_id],
                        adverse_prices.get(system['positions'][position_id].symbol, 50000),
                        'emergency'
                    )
        
        # Verify risk management
        assert len(risk_alerts) > 0
        assert any(alert['risk_type'] == 'size' for alert in risk_alerts)
        assert any(alert['risk_type'] == 'correlation' for alert in risk_alerts)
    
    @pytest.mark.asyncio
    async def test_position_recovery_and_reconciliation(self, position_system):
        """Test position recovery and reconciliation mechanisms"""
        system, config = position_system
        
        # Simulate system restart with open positions
        stored_positions = [
            {
                'id': 'recovered_1',
                'symbol': 'BTC/USDT',
                'side': 'long',
                'entry_price': 49800,
                'size': 0.15,
                'status': 'open',
                'opened_at': '2024-01-01T10:00:00'
            },
            {
                'id': 'recovered_2',
                'symbol': 'BTC/USDT',
                'side': 'short',
                'entry_price': 50200,
                'size': 0.1,
                'status': 'open',
                'opened_at': '2024-01-01T11:00:00'
            }
        ]
        
        # Exchange reported positions (might differ)
        exchange_positions = [
            {
                'id': 'recovered_1',
                'size': 0.15,
                'side': 'long',
                'avg_price': 49800
            },
            {
                'id': 'recovered_2',
                'size': 0.08,  # Different size - partial close happened
                'side': 'short',
                'avg_price': 50200
            },
            {
                'id': 'unknown_pos',  # Position not in our records
                'size': 0.05,
                'side': 'long',
                'avg_price': 50000
            }
        ]
        
        # Reconciliation process
        reconciliation_report = {
            'matched': [],
            'discrepancies': [],
            'unknown': [],
            'missing': []
        }
        
        # Match stored vs exchange positions
        for stored in stored_positions:
            exchange_match = next(
                (ep for ep in exchange_positions if ep['id'] == stored['id']),
                None
            )
            
            if exchange_match:
                if abs(exchange_match['size'] - stored['size']) < 0.001:
                    reconciliation_report['matched'].append(stored['id'])
                else:
                    reconciliation_report['discrepancies'].append({
                        'id': stored['id'],
                        'stored_size': stored['size'],
                        'exchange_size': exchange_match['size'],
                        'difference': exchange_match['size'] - stored['size']
                    })
            else:
                reconciliation_report['missing'].append(stored['id'])
        
        # Check for unknown positions
        stored_ids = [sp['id'] for sp in stored_positions]
        for exchange_pos in exchange_positions:
            if exchange_pos['id'] not in stored_ids:
                reconciliation_report['unknown'].append(exchange_pos)
        
        # Recovery actions
        recovery_actions = []
        
        # Handle discrepancies
        for discrepancy in reconciliation_report['discrepancies']:
            action = {
                'position_id': discrepancy['id'],
                'action_type': 'adjust_size',
                'old_size': discrepancy['stored_size'],
                'new_size': discrepancy['exchange_size'],
                'reason': 'reconciliation'
            }
            recovery_actions.append(action)
            
            # Update position
            if discrepancy['id'] in system['positions']:
                system['positions'][discrepancy['id']].size = discrepancy['exchange_size']
        
        # Handle unknown positions
        for unknown in reconciliation_report['unknown']:
            action = {
                'position_id': unknown['id'],
                'action_type': 'import_position',
                'data': unknown,
                'reason': 'found_on_exchange'
            }
            recovery_actions.append(action)
            
            # Create position object
            new_position = Position(
                id=unknown['id'],
                symbol=config['symbol'],
                side=unknown['side'],
                entry_price=unknown['avg_price'],
                size=unknown['size'],
                position_type=PositionType.SPOT,
                status=PositionStatus.OPEN,
                opened_at=datetime.now(),
                metadata={'recovered': True, 'source': 'exchange'}
            )
            system['positions'][new_position.id] = new_position
        
        # Verify reconciliation
        assert len(reconciliation_report['discrepancies']) > 0
        assert len(reconciliation_report['unknown']) > 0
        assert len(recovery_actions) >= 2
        
        # All positions should now match exchange
        assert 'unknown_pos' in system['positions']
    
    @pytest.mark.asyncio
    async def test_multi_strategy_position_coordination(self, position_system):
        """Test coordination between positions from different strategies"""
        system, config = position_system
        
        # Positions from different strategies
        strategy_positions = {
            'grid_strategy': [
                Position(
                    id='grid_1',
                    symbol='BTC/USDT',
                    side='long',
                    entry_price=49900,
                    size=0.05,
                    position_type=PositionType.GRID,
                    status=PositionStatus.OPEN,
                    opened_at=datetime.now(),
                    metadata={'strategy': 'grid', 'level': 1}
                ),
                Position(
                    id='grid_2',
                    symbol='BTC/USDT',
                    side='long',
                    entry_price=49800,
                    size=0.05,
                    position_type=PositionType.GRID,
                    status=PositionStatus.OPEN,
                    opened_at=datetime.now(),
                    metadata={'strategy': 'grid', 'level': 2}
                )
            ],
            'trend_strategy': [
                Position(
                    id='trend_1',
                    symbol='BTC/USDT',
                    side='long',
                    entry_price=50000,
                    size=0.2,
                    position_type=PositionType.SPOT,
                    status=PositionStatus.OPEN,
                    opened_at=datetime.now(),
                    metadata={'strategy': 'trend_following'}
                )
            ],
            'arbitrage_strategy': [
                Position(
                    id='arb_1',
                    symbol='BTC/USDT',
                    side='short',
                    entry_price=50100,
                    size=0.15,
                    position_type=PositionType.HEDGE,
                    status=PositionStatus.OPEN,
                    opened_at=datetime.now(),
                    metadata={'strategy': 'arbitrage', 'pair': 'BTC-PERP'}
                )
            ]
        }
        
        # Add all positions
        for strategy, positions in strategy_positions.items():
            for pos in positions:
                system['positions'][pos.id] = pos
        
        # Calculate net exposure by strategy
        strategy_exposure = {}
        for strategy, positions in strategy_positions.items():
            net_long = sum(p.size for p in positions if p.side == 'long')
            net_short = sum(p.size for p in positions if p.side == 'short')
            strategy_exposure[strategy] = {
                'net_position': net_long - net_short,
                'gross_exposure': net_long + net_short,
                'position_count': len(positions)
            }
        
        # Test coordination rules
        
        # 1. Total exposure limit
        total_exposure = sum(exp['gross_exposure'] for exp in strategy_exposure.values())
        assert total_exposure <= config['position']['max_total_exposure']
        
        # 2. Strategy conflict detection
        conflicts = []
        
        # Check if strategies are working against each other
        long_strategies = [s for s, exp in strategy_exposure.items() if exp['net_position'] > 0]
        short_strategies = [s for s, exp in strategy_exposure.items() if exp['net_position'] < 0]
        
        if long_strategies and short_strategies:
            conflicts.append({
                'type': 'directional_conflict',
                'long_strategies': long_strategies,
                'short_strategies': short_strategies
            })
        
        # 3. Position netting opportunities
        all_positions = [p for positions in strategy_positions.values() for p in positions]
        
        # Find positions that could be netted
        netting_opportunities = []
        for i, pos1 in enumerate(all_positions):
            for pos2 in all_positions[i+1:]:
                if (pos1.symbol == pos2.symbol and 
                    pos1.side != pos2.side and
                    abs(pos1.entry_price - pos2.entry_price) / pos1.entry_price < 0.001):  # Very close prices
                    
                    netting_opportunities.append({
                        'positions': [pos1.id, pos2.id],
                        'potential_saving': min(pos1.size, pos2.size) * 0.001 * pos1.entry_price  # Fee saving
                    })
        
        # 4. Strategy priority in constrained resources
        if total_exposure > config['position']['max_total_exposure'] * 0.8:  # 80% utilized
            # Prioritize strategies
            strategy_performance = {
                'grid_strategy': {'win_rate': 0.65, 'sharpe': 1.2},
                'trend_strategy': {'win_rate': 0.45, 'sharpe': 0.8},
                'arbitrage_strategy': {'win_rate': 0.9, 'sharpe': 2.5}
            }
            
            # Rank strategies
            strategy_ranking = sorted(
                strategy_performance.items(),
                key=lambda x: x[1]['sharpe'],
                reverse=True
            )
            
            # Reduce exposure for lower-ranked strategies
            reduction_plan = []
            for idx, (strategy, _) in enumerate(strategy_ranking[1:], 1):  # Skip best
                reduction_plan.append({
                    'strategy': strategy,
                    'reduction_percent': idx * 10,  # 10%, 20%, etc.
                    'reason': 'resource_constraint'
                })
        
        # Verify coordination
        assert len(strategy_exposure) == 3
        if conflicts:
            assert conflicts[0]['type'] == 'directional_conflict'
        
        # Net position across all strategies
        total_net = sum(exp['net_position'] for exp in strategy_exposure.values())
        assert abs(total_net) < config['position']['max_position_size'] * 2
    
    # Helper methods
    async def _close_position(self, system: Dict, position: Position, 
                            exit_price: float, reason: str) -> None:
        """Close a position"""
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.closed_at = datetime.now()
        
        if position.side == 'long':
            position.realized_pnl = (exit_price - position.entry_price) * position.size
        else:
            position.realized_pnl = (position.entry_price - exit_price) * position.size
        
        position.realized_pnl -= position.fees_paid
        
        # Record in history
        system['position_history'].append({
            'position': position,
            'close_reason': reason,
            'timestamp': position.closed_at
        })
        
        # Emit event
        await system['event_bus'].publish(Event(
            type=EventType.POSITION_CLOSED,
            data={
                'position_id': position.id,
                'reason': reason,
                'pnl': position.realized_pnl
            }
        ))
    
    async def _update_trailing_stop(self, position: Position, current_price: float, 
                                  config: Dict) -> None:
        """Update trailing stop for profitable position"""
        if position.side == 'long':
            # For longs, move stop up but never down
            new_stop = current_price * (1 - config['position']['default_stop_loss'])
            if position.stop_loss is None or new_stop > position.stop_loss:
                position.stop_loss = new_stop
        else:  # short
            # For shorts, move stop down but never up
            new_stop = current_price * (1 + config['position']['default_stop_loss'])
            if position.stop_loss is None or new_stop < position.stop_loss:
                position.stop_loss = new_stop