"""
Market Extreme Conditions Testing Suite for GridAttention Trading System
Tests system behavior during flash crashes, circuit breakers, extreme volatility, and black swan events
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, AsyncMock, patch
import logging
import random
from dataclasses import dataclass
from enum import Enum

# GridAttention imports - aligned with system structure
from src.grid_attention_layer import GridAttentionLayer
from src.market_regime_detector import MarketRegimeDetector
from src.risk_management_system import RiskManagementSystem
from src.execution_engine import ExecutionEngine
from src.grid_strategy_selector import GridStrategySelector
from src.performance_monitor import PerformanceMonitor
from src.circuit_breaker import CircuitBreaker
from src.emergency_shutdown import EmergencyShutdownManager

# Test utilities
from tests.utils.test_helpers import async_test, create_test_config
from tests.utils.market_simulator import (
    simulate_flash_crash,
    simulate_volatility_spike,
    simulate_liquidity_crisis,
    simulate_black_swan_event
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Extreme market conditions"""
    FLASH_CRASH = "flash_crash"
    CIRCUIT_BREAKER = "circuit_breaker"
    EXTREME_VOLATILITY = "extreme_volatility"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    BLACK_SWAN = "black_swan"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    THIN_MARKET = "thin_market"
    HALTED_TRADING = "halted_trading"


@dataclass
class MarketEvent:
    """Market event data"""
    timestamp: datetime
    condition: MarketCondition
    severity: float  # 0-1 scale
    duration_seconds: int
    price_impact_percent: float
    volume_impact_percent: float
    affected_symbols: List[str]


class TestFlashCrashScenarios:
    """Test flash crash scenarios"""
    
    @pytest.fixture
    def grid_attention(self):
        """Create GridAttention system with extreme market handling"""
        config = create_test_config()
        config['extreme_markets'] = {
            'flash_crash_detection': True,
            'circuit_breaker_enabled': True,
            'emergency_shutdown_enabled': True,
            'max_price_change_percent': 10,
            'max_spread_percent': 5,
            'min_liquidity_threshold': 1000
        }
        return GridAttentionLayer(config)
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker"""
        config = create_test_config()
        config['circuit_breaker'] = {
            'level1_threshold': 5,   # 5% drop
            'level2_threshold': 10,  # 10% drop
            'level3_threshold': 20,  # 20% drop
            'cooldown_periods': [300, 900, 3600],  # 5min, 15min, 1hour
            'reference_window': 300  # 5 minutes
        }
        return CircuitBreaker(config)
    
    @async_test
    async def test_flash_crash_detection(self, grid_attention):
        """Test flash crash detection and response"""
        # Initial market state
        initial_price = 50000
        symbol = 'BTC/USDT'
        
        # Simulate flash crash - 30% drop in 1 minute
        crash_data = simulate_flash_crash(
            symbol=symbol,
            initial_price=initial_price,
            drop_percent=30,
            duration_seconds=60,
            recovery_seconds=300
        )
        
        responses = []
        for tick in crash_data:
            response = await grid_attention.process_market_update(tick)
            responses.append(response)
            
            # Check if flash crash was detected
            if response.get('alert_type') == 'flash_crash':
                assert response['action'] in ['pause_trading', 'reduce_exposure', 'emergency_stop']
                assert response['severity'] == 'critical'
                logger.info(f"Flash crash detected at price {tick['price']}")
                break
        
        # Verify system took protective action
        flash_crash_detected = any(r.get('alert_type') == 'flash_crash' for r in responses)
        assert flash_crash_detected, "Flash crash was not detected"
        
        # Check risk management response
        risk_state = await grid_attention.risk_management.get_current_state()
        assert risk_state['mode'] == 'emergency'
        assert risk_state['trading_allowed'] is False
        assert risk_state['max_position_size'] == 0
    
    @async_test
    async def test_circuit_breaker_triggers(self, circuit_breaker):
        """Test circuit breaker level triggers"""
        initial_price = 50000
        
        # Test Level 1 trigger (5% drop)
        level1_price = initial_price * 0.95
        
        triggered = await circuit_breaker.check_trigger(
            current_price=level1_price,
            reference_price=initial_price,
            symbol='BTC/USDT'
        )
        
        assert triggered['triggered'] is True
        assert triggered['level'] == 1
        assert triggered['cooldown_seconds'] == 300
        assert triggered['can_resume_at'] is not None
        
        # Test Level 2 trigger (10% drop)
        level2_price = initial_price * 0.90
        
        triggered = await circuit_breaker.check_trigger(
            current_price=level2_price,
            reference_price=initial_price,
            symbol='BTC/USDT'
        )
        
        assert triggered['level'] == 2
        assert triggered['cooldown_seconds'] == 900
        
        # Test Level 3 trigger (20% drop)
        level3_price = initial_price * 0.80
        
        triggered = await circuit_breaker.check_trigger(
            current_price=level3_price,
            reference_price=initial_price,
            symbol='BTC/USDT'
        )
        
        assert triggered['level'] == 3
        assert triggered['cooldown_seconds'] == 3600
        assert triggered['halt_all_trading'] is True
    
    @async_test
    async def test_multi_asset_flash_crash(self, grid_attention):
        """Test flash crash affecting multiple assets"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        initial_prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000, 'BNB/USDT': 400}
        
        # Simulate correlated crash
        crash_events = []
        for symbol in symbols:
            crash_data = simulate_flash_crash(
                symbol=symbol,
                initial_price=initial_prices[symbol],
                drop_percent=25,
                duration_seconds=45,
                correlation=0.8  # High correlation
            )
            crash_events.append(crash_data)
        
        # Process simultaneous crashes
        alerts = []
        for i in range(len(crash_events[0])):
            for j, symbol in enumerate(symbols):
                tick = crash_events[j][i]
                response = await grid_attention.process_market_update(tick)
                
                if response.get('alert_type'):
                    alerts.append(response)
        
        # Verify correlated crash detection
        correlated_crash = any(a.get('alert_type') == 'correlated_crash' for a in alerts)
        assert correlated_crash, "Correlated crash not detected"
        
        # Check system-wide response
        system_state = await grid_attention.get_system_state()
        assert system_state['emergency_mode'] is True
        assert system_state['all_trading_halted'] is True
    
    @async_test
    async def test_flash_crash_recovery(self, grid_attention):
        """Test system recovery after flash crash"""
        # Simulate flash crash and recovery
        crash_and_recovery = simulate_flash_crash(
            symbol='BTC/USDT',
            initial_price=50000,
            drop_percent=20,
            duration_seconds=60,
            recovery_seconds=600,
            recovery_percent=15  # Recover to -5% from initial
        )
        
        states = []
        for tick in crash_and_recovery:
            response = await grid_attention.process_market_update(tick)
            
            if response.get('state_change'):
                states.append({
                    'timestamp': tick['timestamp'],
                    'state': response['new_state'],
                    'price': tick['price']
                })
        
        # Verify state transitions
        state_sequence = [s['state'] for s in states]
        
        assert 'normal' in state_sequence
        assert 'emergency' in state_sequence
        assert 'recovery' in state_sequence
        assert state_sequence[-1] in ['normal', 'cautious']
        
        # Check gradual position reopening
        final_risk_state = await grid_attention.risk_management.get_current_state()
        assert final_risk_state['trading_allowed'] is True
        assert 0 < final_risk_state['max_position_size'] <= 1.0


class TestExtremeVolatility:
    """Test extreme volatility conditions"""
    
    @pytest.fixture
    def volatility_manager(self):
        """Create volatility manager"""
        config = create_test_config()
        config['volatility'] = {
            'normal_range': [0.01, 0.05],  # 1-5% daily
            'high_threshold': 0.10,         # 10% daily
            'extreme_threshold': 0.20,      # 20% daily
            'measurement_window': 3600,     # 1 hour
            'adjustment_factor': 0.5        # Reduce position by 50% in extreme volatility
        }
        return VolatilityManager(config)
    
    @async_test
    async def test_volatility_spike_detection(self, grid_attention, volatility_manager):
        """Test detection of sudden volatility spikes"""
        # Generate high volatility market data
        volatility_data = simulate_volatility_spike(
            symbol='BTC/USDT',
            base_price=50000,
            normal_volatility=0.02,
            spike_volatility=0.25,  # 25% volatility
            spike_duration=300,     # 5 minutes
            total_duration=1800     # 30 minutes
        )
        
        volatility_readings = []
        for tick in volatility_data:
            # Update volatility calculation
            volatility = await volatility_manager.calculate_realtime_volatility(
                price=tick['price'],
                timestamp=tick['timestamp']
            )
            volatility_readings.append(volatility)
            
            # Process market update
            response = await grid_attention.process_market_update(tick)
            
            if volatility > volatility_manager.extreme_threshold:
                assert response.get('volatility_alert') == 'extreme'
                assert response.get('position_adjustment') <= 0.5
        
        # Verify volatility spike was captured
        max_volatility = max(volatility_readings)
        assert max_volatility > 0.20, "Volatility spike not properly measured"
    
    @async_test
    async def test_whipsaw_market_protection(self, grid_attention):
        """Test protection against whipsaw markets"""
        # Generate whipsaw pattern - rapid reversals
        whipsaw_data = []
        base_price = 50000
        timestamp = datetime.now()
        
        for i in range(100):
            # Create rapid price reversals
            if i % 4 == 0:
                price = base_price * 1.02  # +2%
            elif i % 4 == 1:
                price = base_price * 0.98  # -2%
            elif i % 4 == 2:
                price = base_price * 1.015 # +1.5%
            else:
                price = base_price * 0.985 # -1.5%
            
            whipsaw_data.append({
                'symbol': 'BTC/USDT',
                'price': price,
                'volume': random.uniform(10, 100),
                'timestamp': timestamp + timedelta(seconds=i*3)
            })
        
        # Process whipsaw market
        whipsaw_detected = False
        for tick in whipsaw_data:
            response = await grid_attention.process_market_update(tick)
            
            if response.get('market_condition') == 'whipsaw':
                whipsaw_detected = True
                assert response.get('action') in ['pause_grid', 'widen_spread', 'reduce_frequency']
                break
        
        assert whipsaw_detected, "Whipsaw market condition not detected"
        
        # Verify protective measures
        strategy_state = await grid_attention.grid_strategy_selector.get_current_strategy()
        assert strategy_state['type'] == 'defensive'
        assert strategy_state['grid_spacing'] > strategy_state['normal_spacing']
    
    @async_test
    async def test_volatility_regime_adaptation(self, grid_attention):
        """Test strategy adaptation to different volatility regimes"""
        # Define volatility regimes
        volatility_regimes = [
            {'name': 'low', 'volatility': 0.01, 'duration': 300},
            {'name': 'normal', 'volatility': 0.03, 'duration': 300},
            {'name': 'high', 'volatility': 0.10, 'duration': 300},
            {'name': 'extreme', 'volatility': 0.25, 'duration': 300}
        ]
        
        regime_strategies = {}
        
        for regime in volatility_regimes:
            # Generate market data for regime
            regime_data = generate_volatility_regime_data(
                volatility=regime['volatility'],
                duration=regime['duration'],
                base_price=50000
            )
            
            # Process regime data
            for tick in regime_data:
                await grid_attention.process_market_update(tick)
            
            # Get adapted strategy
            strategy = await grid_attention.grid_strategy_selector.get_current_strategy()
            regime_strategies[regime['name']] = strategy
        
        # Verify appropriate adaptations
        assert regime_strategies['low']['grid_levels'] > regime_strategies['extreme']['grid_levels']
        assert regime_strategies['low']['position_size'] > regime_strategies['extreme']['position_size']
        assert regime_strategies['extreme']['stop_loss_percent'] < regime_strategies['low']['stop_loss_percent']


class TestLiquidityCrisis:
    """Test liquidity crisis scenarios"""
    
    @async_test
    async def test_liquidity_dry_up(self, grid_attention):
        """Test system behavior when liquidity dries up"""
        # Simulate liquidity crisis
        liquidity_crisis = simulate_liquidity_crisis(
            symbol='BTC/USDT',
            normal_volume=10000,
            crisis_volume=100,  # 99% volume drop
            normal_spread=0.01,  # 0.01%
            crisis_spread=1.0,   # 1% spread
            duration_seconds=600
        )
        
        liquidity_alerts = []
        for tick in liquidity_crisis:
            response = await grid_attention.process_market_update(tick)
            
            if response.get('liquidity_alert'):
                liquidity_alerts.append(response)
                
            # Check order book depth
            if 'order_book' in tick:
                depth = calculate_order_book_depth(tick['order_book'])
                if depth < 1000:  # Minimum depth threshold
                    assert response.get('action') in ['halt_trading', 'reduce_size', 'market_maker_mode']
        
        assert len(liquidity_alerts) > 0, "Liquidity crisis not detected"
        
        # Verify protective measures
        execution_state = await grid_attention.execution_engine.get_state()
        assert execution_state['use_market_orders'] is False
        assert execution_state['max_order_size'] < execution_state['normal_max_size']
    
    @async_test
    async def test_wide_spread_handling(self, grid_attention):
        """Test handling of extremely wide spreads"""
        # Normal spread: 0.01%, Crisis spread: 5%
        wide_spread_data = [
            {
                'symbol': 'BTC/USDT',
                'bid': 47500,  # 5% spread
                'ask': 50000,
                'mid': 48750,
                'spread_percent': 5.0,
                'timestamp': datetime.now()
            }
        ]
        
        for tick in wide_spread_data:
            response = await grid_attention.process_market_update(tick)
            
            assert response.get('spread_alert') == 'extreme'
            assert response.get('trading_allowed') is False
            assert 'spread too wide' in response.get('reason', '').lower()
        
        # Verify grid adjustment
        grid_state = await grid_attention.get_grid_state()
        assert grid_state['active_orders'] == 0
        assert grid_state['reason'] == 'spread_exceeds_threshold'
    
    @async_test
    async def test_order_book_imbalance(self, grid_attention):
        """Test extreme order book imbalances"""
        # Create severely imbalanced order book
        imbalanced_book = {
            'symbol': 'BTC/USDT',
            'bids': [
                {'price': 49900, 'size': 0.1},
                {'price': 49800, 'size': 0.1},
                {'price': 49700, 'size': 0.1}
            ],
            'asks': [
                {'price': 50000, 'size': 100},  # 1000x more on sell side
                {'price': 50100, 'size': 150},
                {'price': 50200, 'size': 200}
            ],
            'timestamp': datetime.now()
        }
        
        response = await grid_attention.process_order_book(imbalanced_book)
        
        assert response.get('imbalance_detected') is True
        assert response.get('imbalance_ratio') > 100
        assert response.get('bias_direction') == 'sell'
        assert response.get('recommended_action') in ['avoid_buys', 'sell_only', 'halt']
        
        # Check strategy adjustment
        strategy = await grid_attention.grid_strategy_selector.get_current_strategy()
        assert strategy['allow_buys'] is False or strategy['buy_size_multiplier'] < 0.5


class TestBlackSwanEvents:
    """Test black swan events"""
    
    @async_test
    async def test_exchange_hack_scenario(self, grid_attention):
        """Test response to exchange hack news"""
        # Simulate black swan event - exchange hack
        hack_event = MarketEvent(
            timestamp=datetime.now(),
            condition=MarketCondition.BLACK_SWAN,
            severity=1.0,  # Maximum severity
            duration_seconds=3600,
            price_impact_percent=-30,
            volume_impact_percent=-80,
            affected_symbols=['BTC/USDT', 'ETH/USDT', 'ALL']
        )
        
        response = await grid_attention.process_emergency_event(hack_event)
        
        assert response['action'] == 'emergency_shutdown'
        assert response['withdraw_orders'] is True
        assert response['close_positions'] is True
        assert response['reason'] == 'black_swan_event'
        
        # Verify complete shutdown
        system_state = await grid_attention.get_system_state()
        assert system_state['status'] == 'emergency_shutdown'
        assert system_state['trading_enabled'] is False
        assert system_state['accepting_market_data'] is False
    
    @async_test
    async def test_regulatory_halt(self, grid_attention):
        """Test regulatory trading halt"""
        # Simulate regulatory halt
        halt_notice = {
            'type': 'regulatory_halt',
            'symbols': ['BTC/USDT'],
            'reason': 'pending_investigation',
            'expected_duration': 'indefinite',
            'timestamp': datetime.now()
        }
        
        response = await grid_attention.process_trading_halt(halt_notice)
        
        assert response['acknowledged'] is True
        assert response['orders_cancelled'] > 0 or response['no_active_orders'] is True
        assert response['positions_marked'] is True
        
        # Try to place order during halt
        order_attempt = await grid_attention.execution_engine.place_order({
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'size': 0.1,
            'price': 50000
        })
        
        assert order_attempt['success'] is False
        assert 'halted' in order_attempt['error'].lower()
    
    @async_test
    async def test_infrastructure_failure(self, grid_attention):
        """Test response to infrastructure failures"""
        # Simulate data feed failure
        feed_failure = {
            'type': 'data_feed_failure',
            'affected_feeds': ['primary', 'backup1'],
            'working_feeds': ['backup2'],
            'data_quality': 'degraded',
            'timestamp': datetime.now()
        }
        
        response = await grid_attention.handle_infrastructure_failure(feed_failure)
        
        assert response['fallback_activated'] is True
        assert response['using_feed'] == 'backup2'
        assert response['trading_mode'] == 'conservative'
        assert response['risk_limits_reduced'] is True
        
        # Test complete feed loss
        total_failure = {
            'type': 'data_feed_failure',
            'affected_feeds': ['primary', 'backup1', 'backup2'],
            'working_feeds': [],
            'timestamp': datetime.now()
        }
        
        response = await grid_attention.handle_infrastructure_failure(total_failure)
        
        assert response['trading_suspended'] is True
        assert response['reason'] == 'no_data_feeds_available'


class TestPriceGaps:
    """Test price gap scenarios"""
    
    @async_test
    async def test_gap_up_opening(self, grid_attention):
        """Test handling of gap up openings"""
        # Friday close price
        friday_close = 50000
        
        # Monday open with 10% gap up
        monday_open = 55000
        
        gap_event = {
            'symbol': 'BTC/USDT',
            'previous_close': friday_close,
            'current_price': monday_open,
            'gap_percent': 10.0,
            'gap_direction': 'up',
            'timestamp': datetime.now()
        }
        
        response = await grid_attention.process_price_gap(gap_event)
        
        assert response['gap_detected'] is True
        assert response['gap_size'] == 'large'
        assert response['grid_adjustment'] == 'recalibrate'
        
        # Check if existing orders were cancelled
        assert response['orders_cancelled'] is True
        assert response['new_grid_center'] == monday_open
        
        # Verify risk adjustment for gap
        risk_params = await grid_attention.risk_management.get_gap_risk_parameters()
        assert risk_params['position_size_multiplier'] < 1.0
        assert risk_params['require_confirmation'] is True
    
    @async_test
    async def test_gap_down_protection(self, grid_attention):
        """Test protection against gap down events"""
        # Simulate overnight gap down
        previous_price = 50000
        gap_down_price = 42000  # 16% gap down
        
        # Check stop loss handling through gap
        position = {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry_price': 48000,
            'size': 1.0,
            'stop_loss': 46000  # Stop at 46000, but gap to 42000
        }
        
        gap_result = await grid_attention.process_gap_down(
            position=position,
            gap_price=gap_down_price,
            previous_price=previous_price
        )
        
        assert gap_result['stop_triggered'] is True
        assert gap_result['execution_price'] == gap_down_price  # Executed at gap price
        assert gap_result['slippage'] == 4000  # 46000 - 42000
        assert gap_result['slippage_percent'] > 8
        
        # Verify post-gap risk management
        assert gap_result['trading_paused'] is True
        assert gap_result['reassessment_required'] is True


class TestThinMarkets:
    """Test thin market conditions"""
    
    @async_test
    async def test_holiday_thin_market(self, grid_attention):
        """Test handling of holiday thin markets"""
        # Simulate Christmas/New Year thin market
        thin_market_data = {
            'symbol': 'BTC/USDT',
            'average_volume': 100,  # vs normal 10000
            'bid_ask_spread': 0.5,  # vs normal 0.01%
            'order_book_depth': 10,  # vs normal 1000
            'active_traders': 50,    # vs normal 5000
            'is_holiday': True,
            'timestamp': datetime.now()
        }
        
        response = await grid_attention.analyze_market_conditions(thin_market_data)
        
        assert response['market_type'] == 'thin'
        assert response['liquidity_score'] < 0.2
        assert response['recommended_strategy'] == 'conservative'
        
        # Check adjusted parameters
        params = await grid_attention.get_thin_market_parameters()
        assert params['max_position_size'] < params['normal_max_size'] * 0.3
        assert params['use_market_orders'] is False
        assert params['min_order_spacing'] > params['normal_spacing'] * 2
    
    @async_test
    async def test_weekend_volatility(self, grid_attention):
        """Test weekend low liquidity volatility"""
        # Simulate weekend pump and dump
        weekend_pump_dump = simulate_weekend_manipulation(
            base_price=50000,
            pump_percent=15,
            dump_percent=20,
            low_volume=True
        )
        
        manipulation_detected = False
        for tick in weekend_pump_dump:
            response = await grid_attention.process_market_update(tick)
            
            if response.get('manipulation_suspected'):
                manipulation_detected = True
                assert response['action'] in ['ignore_price', 'halt_trading', 'alert_only']
                assert response['reason'] == 'abnormal_price_movement_low_volume'
        
        assert manipulation_detected, "Market manipulation not detected"


class TestRecoveryMechanisms:
    """Test recovery from extreme conditions"""
    
    @async_test
    async def test_gradual_recovery_after_crash(self, grid_attention):
        """Test gradual recovery mechanism after crash"""
        # Simulate crash and gradual recovery phases
        recovery_phases = [
            {'name': 'crash', 'price_level': 0.70, 'volatility': 0.30, 'duration': 60},
            {'name': 'stabilization', 'price_level': 0.72, 'volatility': 0.20, 'duration': 300},
            {'name': 'early_recovery', 'price_level': 0.75, 'volatility': 0.15, 'duration': 600},
            {'name': 'recovery', 'price_level': 0.80, 'volatility': 0.10, 'duration': 900},
            {'name': 'normalization', 'price_level': 0.85, 'volatility': 0.05, 'duration': 1200}
        ]
        
        base_price = 50000
        position_sizes = []
        
        for phase in recovery_phases:
            phase_data = generate_recovery_phase_data(
                base_price=base_price,
                price_level=phase['price_level'],
                volatility=phase['volatility'],
                duration=phase['duration']
            )
            
            for tick in phase_data:
                await grid_attention.process_market_update(tick)
            
            # Get position size for phase
            risk_params = await grid_attention.risk_management.get_current_parameters()
            position_sizes.append({
                'phase': phase['name'],
                'max_position': risk_params['max_position_size'],
                'confidence': risk_params['market_confidence']
            })
        
        # Verify gradual increase in position sizes
        for i in range(1, len(position_sizes)):
            assert position_sizes[i]['max_position'] >= position_sizes[i-1]['max_position']
            assert position_sizes[i]['confidence'] >= position_sizes[i-1]['confidence']
        
        # Final state should be normalized but cautious
        assert position_sizes[-1]['max_position'] < 1.0  # Not full size
        assert position_sizes[-1]['confidence'] > 0.7    # Reasonable confidence
    
    @async_test
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test recovery after circuit breaker activation"""
        # Trigger circuit breaker
        await circuit_breaker.trigger(level=2, symbol='BTC/USDT')
        
        # Verify cooldown
        assert circuit_breaker.is_active('BTC/USDT') is True
        assert circuit_breaker.can_trade('BTC/USDT') is False
        
        # Wait for cooldown (simulate)
        await asyncio.sleep(0.1)  # In real test would wait actual cooldown
        
        # Simulate cooldown expiry
        circuit_breaker._cooldown_end = datetime.now() - timedelta(seconds=1)
        
        # Verify can resume trading
        assert circuit_breaker.can_trade('BTC/USDT') is True
        
        # Test gradual resumption
        resumption_plan = await circuit_breaker.get_resumption_plan('BTC/USDT')
        assert resumption_plan['phases'] > 1
        assert resumption_plan['initial_size_limit'] < 1.0
        assert resumption_plan['monitoring_period'] > 0


# Helper Functions

def calculate_order_book_depth(order_book: Dict) -> float:
    """Calculate total order book depth in USD"""
    depth = 0
    
    for side in ['bids', 'asks']:
        if side in order_book:
            for order in order_book[side]:
                depth += order['price'] * order['size']
    
    return depth


def generate_volatility_regime_data(volatility: float, duration: int, base_price: float) -> List[Dict]:
    """Generate market data for specific volatility regime"""
    data = []
    timestamp = datetime.now()
    price = base_price
    
    for i in range(duration):
        # Random walk with specified volatility
        returns = np.random.normal(0, volatility / np.sqrt(duration))
        price = price * (1 + returns)
        
        data.append({
            'symbol': 'BTC/USDT',
            'price': price,
            'volume': np.random.uniform(100, 1000),
            'timestamp': timestamp + timedelta(seconds=i)
        })
    
    return data


def simulate_weekend_manipulation(base_price: float, pump_percent: float, 
                                 dump_percent: float, low_volume: bool = True) -> List[Dict]:
    """Simulate weekend pump and dump manipulation"""
    data = []
    timestamp = datetime.now()
    
    # Normal price
    for i in range(10):
        data.append({
            'symbol': 'BTC/USDT',
            'price': base_price + np.random.uniform(-50, 50),
            'volume': 10 if low_volume else 1000,
            'timestamp': timestamp + timedelta(minutes=i)
        })
    
    # Pump phase
    pump_price = base_price * (1 + pump_percent / 100)
    for i in range(5):
        price = base_price + (pump_price - base_price) * (i + 1) / 5
        data.append({
            'symbol': 'BTC/USDT',
            'price': price,
            'volume': 5 if low_volume else 500,  # Low volume pump
            'timestamp': timestamp + timedelta(minutes=10 + i)
        })
    
    # Dump phase
    dump_price = base_price * (1 - dump_percent / 100)
    for i in range(3):
        price = pump_price - (pump_price - dump_price) * (i + 1) / 3
        data.append({
            'symbol': 'BTC/USDT',
            'price': price,
            'volume': 20 if low_volume else 2000,  # Higher volume dump
            'timestamp': timestamp + timedelta(minutes=15 + i)
        })
    
    return data


def generate_recovery_phase_data(base_price: float, price_level: float, 
                                volatility: float, duration: int) -> List[Dict]:
    """Generate data for recovery phase"""
    data = []
    timestamp = datetime.now()
    target_price = base_price * price_level
    current_price = target_price * 0.95  # Start slightly below target
    
    for i in range(duration):
        # Gradual move toward target with decreasing volatility
        drift = (target_price - current_price) / (duration - i)
        noise = np.random.normal(0, volatility * target_price)
        current_price += drift + noise
        
        # Ensure price doesn't go negative
        current_price = max(current_price, target_price * 0.5)
        
        data.append({
            'symbol': 'BTC/USDT',
            'price': current_price,
            'volume': np.random.uniform(100, 1000) * (1 + i / duration),  # Increasing volume
            'timestamp': timestamp + timedelta(seconds=i),
            'volatility': volatility * (1 - i / duration / 2)  # Decreasing volatility
        })
    
    return data


class VolatilityManager:
    """Manage volatility calculations and responses"""
    
    def __init__(self, config):
        self.config = config
        self.price_history = []
        self.window_size = config['volatility']['measurement_window']
        
    async def calculate_realtime_volatility(self, price: float, timestamp: datetime) -> float:
        """Calculate real-time volatility"""
        self.price_history.append({'price': price, 'timestamp': timestamp})
        
        # Keep only recent history
        cutoff_time = timestamp - timedelta(seconds=self.window_size)
        self.price_history = [p for p in self.price_history if p['timestamp'] > cutoff_time]
        
        if len(self.price_history) < 2:
            return 0.0
        
        # Calculate returns
        prices = [p['price'] for p in self.price_history]
        returns = np.diff(np.log(prices))
        
        # Annualized volatility
        if len(returns) > 0:
            volatility = np.std(returns) * np.sqrt(365 * 24 * 3600 / self.window_size)
            return volatility
        
        return 0.0