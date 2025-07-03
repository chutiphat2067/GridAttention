# tests/functional/test_trading_scenarios.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from core.attention_learning_layer import AttentionLearningLayer
from core.market_regime_detector import MarketRegimeDetector
from core.grid_strategy_selector import GridStrategySelector
from core.risk_management_system import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor
from infrastructure.system_coordinator import SystemCoordinator


class MarketCondition(Enum):
    """Market conditions for testing"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    FLASH_CRASH = "flash_crash"
    SQUEEZE = "squeeze"


@dataclass
class TradingScenario:
    """Define a trading scenario"""
    name: str
    market_condition: MarketCondition
    duration_minutes: int
    expected_behavior: Dict[str, Any]
    risk_events: List[Dict[str, Any]] = None


class TestTradingScenarios:
    """Test various real-world trading scenarios"""
    
    @pytest.fixture
    async def trading_system(self):
        """Create complete trading system for scenario testing"""
        config = {
            'symbol': 'BTC/USDT',
            'timeframe': '5m',
            'grid': {
                'levels': 10,
                'spacing_percent': 0.1,
                'size_per_level': 0.01
            },
            'risk': {
                'max_position': 1.0,
                'max_drawdown': 0.05,
                'stop_loss_percent': 2.0
            },
            'scenarios': {
                'enable_adaptations': True,
                'emergency_threshold': 0.1
            }
        }
        
        system = {
            'coordinator': SystemCoordinator(config),
            'attention': AttentionLearningLayer(config),
            'regime_detector': MarketRegimeDetector(config),
            'strategy_selector': GridStrategySelector(config),
            'risk_manager': RiskManagementSystem(config),
            'execution_engine': ExecutionEngine(config),
            'performance_monitor': PerformanceMonitor(config)
        }
        
        return system, config
    
    @pytest.mark.asyncio
    async def test_bull_trend_scenario(self, trading_system):
        """Test system behavior during bull trend"""
        system, config = trading_system
        
        # Define bull trend scenario
        scenario = TradingScenario(
            name="Strong Bull Trend",
            market_condition=MarketCondition.BULL_TREND,
            duration_minutes=60,
            expected_behavior={
                'regime': 'TRENDING_UP',
                'grid_adjustment': 'expand_upward',
                'position_bias': 'long',
                'profit_expectation': 'positive'
            }
        )
        
        # Generate bull market data
        market_data = self._generate_trending_market(
            start_price=50000,
            end_price=52000,
            periods=scenario.duration_minutes,
            volatility=0.001
        )
        
        # Track system decisions
        decisions = []
        trades = []
        
        # Run scenario
        for i, (_, row) in enumerate(market_data.iterrows()):
            # Update market data
            tick = {
                'price': row['close'],
                'volume': row['volume'],
                'timestamp': row['timestamp']
            }
            
            # Process through system
            features = await system['coordinator'].extract_features(tick)
            regime = await system['regime_detector'].detect_regime(market_data[:i+1])
            strategy = await system['strategy_selector'].select_strategy(regime, features)
            
            # Record decisions
            decisions.append({
                'timestamp': row['timestamp'],
                'regime': regime['regime'],
                'strategy': strategy,
                'price': row['close']
            })
            
            # Execute strategy
            if strategy['action'] in ['buy', 'sell']:
                if await system['risk_manager'].validate_order(strategy):
                    trade = await system['execution_engine'].execute_order(strategy)
                    trades.append(trade)
        
        # Verify expected behavior
        # 1. Regime detection
        regime_counts = self._count_regimes(decisions)
        assert regime_counts.get('TRENDING_UP', 0) > len(decisions) * 0.7
        
        # 2. Grid expansion
        grid_adjustments = [d['strategy'].get('grid_adjustment') for d in decisions]
        assert grid_adjustments.count('expand_upward') > grid_adjustments.count('contract')
        
        # 3. Position bias
        buy_trades = [t for t in trades if t['side'] == 'buy']
        sell_trades = [t for t in trades if t['side'] == 'sell']
        assert len(buy_trades) > len(sell_trades) * 1.2  # Long bias
        
        # 4. Profitability
        pnl = await system['performance_monitor'].calculate_pnl(trades)
        assert pnl['total'] > 0
    
    @pytest.mark.asyncio
    async def test_ranging_market_scenario(self, trading_system):
        """Test system behavior in ranging market"""
        system, config = trading_system
        
        scenario = TradingScenario(
            name="Tight Range",
            market_condition=MarketCondition.RANGING,
            duration_minutes=120,
            expected_behavior={
                'regime': 'RANGING',
                'grid_strategy': 'standard',
                'trade_frequency': 'high',
                'position_management': 'balanced'
            }
        )
        
        # Generate ranging market
        market_data = self._generate_ranging_market(
            center_price=50000,
            range_percent=1.0,
            periods=scenario.duration_minutes
        )
        
        # Track grid performance
        grid_fills = []
        grid_levels = []
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            tick = {
                'price': row['close'],
                'volume': row['volume'],
                'timestamp': row['timestamp']
            }
            
            # Update grid levels
            current_grid = await system['strategy_selector'].update_grid_levels(
                center_price=row['close'],
                volatility=row.get('volatility', 0.001)
            )
            grid_levels.append(current_grid)
            
            # Check for grid fills
            for level in current_grid['levels']:
                if abs(row['close'] - level['price']) < level['trigger_distance']:
                    fill = {
                        'timestamp': row['timestamp'],
                        'price': level['price'],
                        'side': level['side'],
                        'size': level['size']
                    }
                    grid_fills.append(fill)
        
        # Verify ranging market behavior
        # 1. High trade frequency
        assert len(grid_fills) > scenario.duration_minutes * 0.1
        
        # 2. Balanced positions
        buy_fills = [f for f in grid_fills if f['side'] == 'buy']
        sell_fills = [f for f in grid_fills if f['side'] == 'sell']
        balance_ratio = len(buy_fills) / (len(sell_fills) + 1)
        assert 0.8 < balance_ratio < 1.2
        
        # 3. Grid efficiency
        unique_levels_hit = len(set(f['price'] for f in grid_fills))
        assert unique_levels_hit > config['grid']['levels'] * 0.5
    
    @pytest.mark.asyncio
    async def test_high_volatility_scenario(self, trading_system):
        """Test system behavior during high volatility"""
        system, config = trading_system
        
        scenario = TradingScenario(
            name="High Volatility Period",
            market_condition=MarketCondition.HIGH_VOLATILITY,
            duration_minutes=30,
            expected_behavior={
                'risk_reduction': True,
                'grid_widening': True,
                'position_sizing': 'reduced',
                'stop_losses': 'tightened'
            }
        )
        
        # Generate volatile market
        market_data = self._generate_volatile_market(
            base_price=50000,
            volatility=0.05,  # 5% volatility
            periods=scenario.duration_minutes
        )
        
        # Track risk adjustments
        risk_adjustments = []
        position_sizes = []
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            # Calculate current volatility
            if i >= 10:
                recent_data = market_data.iloc[i-10:i+1]
                current_vol = recent_data['close'].pct_change().std()
                
                # Risk adjustment decision
                risk_adj = await system['risk_manager'].adjust_for_volatility(current_vol)
                risk_adjustments.append(risk_adj)
                
                # Position sizing
                base_size = config['grid']['size_per_level']
                adjusted_size = base_size * risk_adj['size_multiplier']
                position_sizes.append(adjusted_size)
        
        # Verify volatility adaptations
        # 1. Risk reduction
        avg_risk_multiplier = np.mean([r['size_multiplier'] for r in risk_adjustments])
        assert avg_risk_multiplier < 0.7  # Reduced from base
        
        # 2. Grid widening
        grid_spacings = [r['grid_spacing_multiplier'] for r in risk_adjustments]
        assert np.mean(grid_spacings) > 1.3  # Wider grids
        
        # 3. Position size reduction
        assert np.mean(position_sizes) < config['grid']['size_per_level'] * 0.7
    
    @pytest.mark.asyncio
    async def test_flash_crash_scenario(self, trading_system):
        """Test system behavior during flash crash"""
        system, config = trading_system
        
        scenario = TradingScenario(
            name="Flash Crash",
            market_condition=MarketCondition.FLASH_CRASH,
            duration_minutes=5,
            expected_behavior={
                'emergency_stop': True,
                'position_closing': 'immediate',
                'trading_halt': True,
                'recovery_protocol': 'activated'
            },
            risk_events=[
                {'time': 2, 'type': 'price_drop', 'magnitude': 0.1}  # 10% drop
            ]
        )
        
        # Generate flash crash data
        market_data = self._generate_flash_crash(
            start_price=50000,
            crash_percent=10,
            crash_minute=2,
            recovery_minutes=3
        )
        
        # Track emergency responses
        emergency_actions = []
        trading_halted = False
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            # Detect abnormal movement
            if i > 0:
                price_change = abs(row['close'] - market_data.iloc[i-1]['close']) / market_data.iloc[i-1]['close']
                
                if price_change > 0.05:  # 5% in one candle
                    # Emergency protocol
                    emergency = await system['risk_manager'].handle_emergency({
                        'type': 'flash_crash',
                        'price_change': price_change,
                        'current_price': row['close']
                    })
                    emergency_actions.append(emergency)
                    
                    # Halt trading
                    if emergency['action'] == 'halt_trading':
                        trading_halted = True
                        await system['execution_engine'].cancel_all_orders()
                        await system['execution_engine'].close_all_positions()
        
        # Verify emergency handling
        assert len(emergency_actions) > 0
        assert trading_halted
        assert any(a['action'] == 'close_positions' for a in emergency_actions)
    
    @pytest.mark.asyncio
    async def test_squeeze_scenario(self, trading_system):
        """Test system behavior during short/long squeeze"""
        system, config = trading_system
        
        scenario = TradingScenario(
            name="Short Squeeze",
            market_condition=MarketCondition.SQUEEZE,
            duration_minutes=15,
            expected_behavior={
                'rapid_adjustment': True,
                'momentum_following': True,
                'risk_monitoring': 'heightened',
                'profit_taking': 'aggressive'
            }
        )
        
        # Generate squeeze market data
        market_data = self._generate_squeeze_market(
            start_price=50000,
            squeeze_percent=20,
            squeeze_duration=10
        )
        
        # Track momentum and decisions
        momentum_scores = []
        profit_taking_actions = []
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            if i >= 5:
                # Calculate momentum
                momentum = self._calculate_momentum(market_data.iloc[i-5:i+1])
                momentum_scores.append(momentum)
                
                # Check for profit taking
                if momentum > 0.8:  # High momentum
                    positions = await system['risk_manager'].get_open_positions()
                    for pos in positions:
                        if pos['unrealized_pnl_percent'] > 5:  # 5% profit
                            take_profit = {
                                'position': pos,
                                'action': 'close_partial',
                                'percent': 0.5,  # Take 50% profit
                                'reason': 'squeeze_momentum'
                            }
                            profit_taking_actions.append(take_profit)
        
        # Verify squeeze handling
        assert max(momentum_scores) > 0.8  # Detected high momentum
        assert len(profit_taking_actions) > 0  # Took profits
        
        # Check rapid adjustments
        strategy_changes = []
        for i in range(1, len(momentum_scores)):
            if abs(momentum_scores[i] - momentum_scores[i-1]) > 0.2:
                strategy_changes.append(i)
        
        assert len(strategy_changes) >= 2  # Multiple rapid adjustments
    
    @pytest.mark.asyncio
    async def test_multi_scenario_sequence(self, trading_system):
        """Test system handling sequence of different scenarios"""
        system, config = trading_system
        
        # Define scenario sequence
        scenarios = [
            TradingScenario("Morning Range", MarketCondition.RANGING, 30, {}),
            TradingScenario("Breakout", MarketCondition.BULL_TREND, 20, {}),
            TradingScenario("Volatility Spike", MarketCondition.HIGH_VOLATILITY, 15, {}),
            TradingScenario("Afternoon Range", MarketCondition.RANGING, 25, {})
        ]
        
        # Track transitions
        regime_transitions = []
        strategy_adaptations = []
        cumulative_pnl = []
        
        current_price = 50000
        all_trades = []
        
        for scenario in scenarios:
            # Generate market data for scenario
            if scenario.market_condition == MarketCondition.RANGING:
                market_data = self._generate_ranging_market(
                    center_price=current_price,
                    range_percent=0.5,
                    periods=scenario.duration_minutes
                )
            elif scenario.market_condition == MarketCondition.BULL_TREND:
                market_data = self._generate_trending_market(
                    start_price=current_price,
                    end_price=current_price * 1.02,
                    periods=scenario.duration_minutes,
                    volatility=0.001
                )
            else:  # HIGH_VOLATILITY
                market_data = self._generate_volatile_market(
                    base_price=current_price,
                    volatility=0.03,
                    periods=scenario.duration_minutes
                )
            
            # Process scenario
            scenario_trades = []
            for _, row in market_data.iterrows():
                # Detect regime
                regime = await system['regime_detector'].detect_regime(market_data)
                
                # Adapt strategy
                strategy = await system['strategy_selector'].select_strategy(
                    regime, 
                    {'scenario': scenario.name}
                )
                
                # Execute if valid
                if strategy['action'] != 'hold':
                    trade = {'price': row['close'], 'side': strategy['action']}
                    scenario_trades.append(trade)
            
            # Record transition
            regime_transitions.append({
                'scenario': scenario.name,
                'regime': regime['regime'],
                'trades': len(scenario_trades)
            })
            
            all_trades.extend(scenario_trades)
            current_price = market_data.iloc[-1]['close']
            
            # Calculate running PnL
            if all_trades:
                pnl = self._calculate_simple_pnl(all_trades)
                cumulative_pnl.append(pnl)
        
        # Verify multi-scenario handling
        # 1. Detected different regimes
        unique_regimes = set(t['regime'] for t in regime_transitions)
        assert len(unique_regimes) >= 2
        
        # 2. Adapted strategies
        trade_counts = [t['trades'] for t in regime_transitions]
        assert max(trade_counts) > min(trade_counts) * 1.5  # Different activity levels
        
        # 3. Overall profitability
        if cumulative_pnl:
            final_pnl = cumulative_pnl[-1]
            assert final_pnl > -config['risk']['max_drawdown'] * current_price
    
    @pytest.mark.asyncio
    async def test_scenario_with_external_events(self, trading_system):
        """Test system response to external events during scenarios"""
        system, config = trading_system
        
        # Define scenario with events
        external_events = [
            {'time': 10, 'type': 'news_release', 'impact': 'high'},
            {'time': 25, 'type': 'large_order', 'side': 'buy', 'size': 1000},
            {'time': 40, 'type': 'exchange_maintenance', 'duration': 5}
        ]
        
        market_data = self._generate_trending_market(
            start_price=50000,
            end_price=51000,
            periods=60,
            volatility=0.002
        )
        
        # Track event responses
        event_responses = []
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            # Check for events
            current_events = [e for e in external_events if e['time'] == i]
            
            for event in current_events:
                # Process event
                response = await system['coordinator'].handle_external_event(event)
                event_responses.append({
                    'event': event,
                    'response': response,
                    'timestamp': row['timestamp']
                })
                
                # Apply event impact
                if event['type'] == 'news_release':
                    # Increase caution
                    await system['risk_manager'].set_risk_multiplier(0.5)
                    
                elif event['type'] == 'large_order':
                    # Adjust grid to capture momentum
                    await system['strategy_selector'].adjust_grid_bias(event['side'])
                    
                elif event['type'] == 'exchange_maintenance':
                    # Pause trading
                    await system['execution_engine'].pause_trading(event['duration'])
        
        # Verify event handling
        assert len(event_responses) == len(external_events)
        
        # Check appropriate responses
        news_responses = [r for r in event_responses 
                         if r['event']['type'] == 'news_release']
        assert any('risk_reduced' in r['response'] for r in news_responses)
        
        maintenance_responses = [r for r in event_responses 
                               if r['event']['type'] == 'exchange_maintenance']
        assert any('trading_paused' in r['response'] for r in maintenance_responses)
    
    # Helper methods
    def _generate_trending_market(self, start_price: float, end_price: float, 
                                 periods: int, volatility: float) -> pd.DataFrame:
        """Generate trending market data"""
        timestamps = pd.date_range(start=datetime.now(), periods=periods, freq='1min')
        
        # Linear trend with noise
        trend = np.linspace(start_price, end_price, periods)
        noise = np.random.normal(0, start_price * volatility, periods)
        prices = trend + noise
        
        # Ensure positive prices
        prices = np.maximum(prices, start_price * 0.5)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * 0.999,
            'high': prices * 1.001,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(50, 150, periods)
        })
    
    def _generate_ranging_market(self, center_price: float, range_percent: float,
                                periods: int) -> pd.DataFrame:
        """Generate ranging market data"""
        timestamps = pd.date_range(start=datetime.now(), periods=periods, freq='1min')
        
        # Sine wave pattern
        range_amplitude = center_price * range_percent / 100
        prices = center_price + range_amplitude * np.sin(np.linspace(0, 4*np.pi, periods))
        prices += np.random.normal(0, center_price * 0.0001, periods)  # Small noise
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * 0.9995,
            'high': prices * 1.0005,
            'low': prices * 0.9995,
            'close': prices,
            'volume': np.random.uniform(40, 120, periods)
        })
    
    def _generate_volatile_market(self, base_price: float, volatility: float,
                                 periods: int) -> pd.DataFrame:
        """Generate volatile market data"""
        timestamps = pd.date_range(start=datetime.now(), periods=periods, freq='1min')
        
        # Random walk with high volatility
        returns = np.random.normal(0, volatility, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * 0.995,
            'high': prices * 1.01,  # Wider ranges
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(100, 300, periods)  # Higher volume
        })
    
    def _generate_flash_crash(self, start_price: float, crash_percent: float,
                             crash_minute: int, recovery_minutes: int) -> pd.DataFrame:
        """Generate flash crash scenario"""
        total_minutes = crash_minute + recovery_minutes + 5
        timestamps = pd.date_range(start=datetime.now(), periods=total_minutes, freq='1min')
        
        prices = []
        for i in range(total_minutes):
            if i < crash_minute:
                # Normal market
                price = start_price * (1 + np.random.normal(0, 0.001))
            elif i == crash_minute:
                # Crash
                price = start_price * (1 - crash_percent/100)
            elif i <= crash_minute + recovery_minutes:
                # Recovery
                recovery_progress = (i - crash_minute) / recovery_minutes
                price = start_price * (1 - crash_percent/100 * (1 - recovery_progress))
            else:
                # Post-recovery
                price = start_price * (1 + np.random.normal(0, 0.002))
            
            prices.append(price)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': np.random.uniform(200, 500, total_minutes)
        })
    
    def _generate_squeeze_market(self, start_price: float, squeeze_percent: float,
                                squeeze_duration: int) -> pd.DataFrame:
        """Generate squeeze scenario"""
        total_minutes = squeeze_duration + 10
        timestamps = pd.date_range(start=datetime.now(), periods=total_minutes, freq='1min')
        
        prices = []
        for i in range(total_minutes):
            if i < 5:
                # Build up
                price = start_price * (1 + i * 0.002)
            elif i < squeeze_duration:
                # Squeeze
                progress = (i - 5) / (squeeze_duration - 5)
                price = start_price * (1 + squeeze_percent/100 * progress)
            else:
                # Cool down
                price = start_price * (1 + squeeze_percent/100) * (1 - (i - squeeze_duration) * 0.01)
            
            prices.append(price)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': np.random.uniform(300, 800, total_minutes)
        })
    
    def _count_regimes(self, decisions: List[Dict]) -> Dict[str, int]:
        """Count regime occurrences"""
        regime_counts = {}
        for decision in decisions:
            regime = decision.get('regime', 'UNKNOWN')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        return regime_counts
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum score (0-1)"""
        if len(data) < 2:
            return 0.5
        
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.5
        
        # Momentum based on returns and trend
        avg_return = returns.mean()
        trend_strength = abs(returns.sum())
        
        # Normalize to 0-1
        momentum = min(1.0, max(0.0, 0.5 + avg_return * 100 + trend_strength))
        return momentum
    
    def _calculate_simple_pnl(self, trades: List[Dict]) -> float:
        """Calculate simple P&L from trades"""
        if not trades:
            return 0.0
        
        pnl = 0.0
        position = 0.0
        
        for trade in trades:
            if trade['side'] == 'buy':
                position += 1
                pnl -= trade['price']
            else:  # sell
                position -= 1
                pnl += trade['price']
        
        # Close remaining position at last price
        if position != 0 and trades:
            pnl += position * trades[-1]['price']
        
        return pnl