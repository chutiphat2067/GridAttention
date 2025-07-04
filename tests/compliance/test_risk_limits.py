"""
Risk limits compliance tests for GridAttention trading system.

Ensures all risk management controls, position limits, and exposure controls
meet regulatory requirements and protect against excessive losses.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd

# Import core components
from core.risk_management import RiskManagementSystem
from core.position_manager import PositionManager
from core.exposure_calculator import ExposureCalculator
from core.compliance_manager import ComplianceManager


class RiskLimitType(Enum):
    """Types of risk limits"""
    # Position limits
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_POSITION_VALUE = "MAX_POSITION_VALUE"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    
    # Loss limits
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    TRAILING_LOSS_LIMIT = "TRAILING_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    
    # Exposure limits
    GROSS_EXPOSURE = "GROSS_EXPOSURE"
    NET_EXPOSURE = "NET_EXPOSURE"
    SECTOR_EXPOSURE = "SECTOR_EXPOSURE"
    
    # Operational limits
    ORDER_RATE_LIMIT = "ORDER_RATE_LIMIT"
    MESSAGE_RATE_LIMIT = "MESSAGE_RATE_LIMIT"
    TRADE_FREQUENCY = "TRADE_FREQUENCY"
    
    # Leverage limits
    MAX_LEVERAGE = "MAX_LEVERAGE"
    MARGIN_USAGE = "MARGIN_USAGE"
    
    # Volatility limits
    VAR_LIMIT = "VAR_LIMIT"
    STRESS_TEST_LIMIT = "STRESS_TEST_LIMIT"


@dataclass
class RiskLimit:
    """Risk limit configuration"""
    limit_type: RiskLimitType
    limit_value: Decimal
    current_value: Decimal = Decimal('0')
    warning_threshold: Decimal = Decimal('0.8')  # 80% of limit
    critical_threshold: Decimal = Decimal('0.95')  # 95% of limit
    hard_limit: bool = True  # If True, blocks trading when breached
    time_window: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskBreach:
    """Risk limit breach event"""
    breach_id: str
    timestamp: datetime
    limit_type: RiskLimitType
    limit_value: Decimal
    actual_value: Decimal
    severity: str  # WARNING, CRITICAL, BREACH
    action_taken: str
    details: Dict[str, Any]


class TestRiskLimits:
    """Test risk limit compliance"""
    
    @pytest.fixture
    async def risk_system(self):
        """Create risk management system"""
        return RiskManagementSystem(
            enable_hard_limits=True,
            enable_soft_limits=True,
            real_time_monitoring=True,
            alert_on_warning=True
        )
    
    @pytest.fixture
    def risk_limits_config(self) -> Dict[RiskLimitType, RiskLimit]:
        """Standard risk limits configuration"""
        return {
            RiskLimitType.MAX_POSITION_SIZE: RiskLimit(
                limit_type=RiskLimitType.MAX_POSITION_SIZE,
                limit_value=Decimal('10.0'),  # 10 BTC max per position
                warning_threshold=Decimal('0.8')
            ),
            RiskLimitType.MAX_POSITION_VALUE: RiskLimit(
                limit_type=RiskLimitType.MAX_POSITION_VALUE,
                limit_value=Decimal('500000'),  # $500K max position value
                warning_threshold=Decimal('0.85')
            ),
            RiskLimitType.DAILY_LOSS_LIMIT: RiskLimit(
                limit_type=RiskLimitType.DAILY_LOSS_LIMIT,
                limit_value=Decimal('10000'),  # $10K daily loss limit
                warning_threshold=Decimal('0.7'),
                time_window=timedelta(days=1)
            ),
            RiskLimitType.MAX_DRAWDOWN: RiskLimit(
                limit_type=RiskLimitType.MAX_DRAWDOWN,
                limit_value=Decimal('0.15'),  # 15% max drawdown
                warning_threshold=Decimal('0.8')
            ),
            RiskLimitType.MAX_LEVERAGE: RiskLimit(
                limit_type=RiskLimitType.MAX_LEVERAGE,
                limit_value=Decimal('3.0'),  # 3x max leverage
                warning_threshold=Decimal('0.9')
            ),
            RiskLimitType.ORDER_RATE_LIMIT: RiskLimit(
                limit_type=RiskLimitType.ORDER_RATE_LIMIT,
                limit_value=Decimal('100'),  # 100 orders per minute
                warning_threshold=Decimal('0.8'),
                time_window=timedelta(minutes=1)
            ),
            RiskLimitType.VAR_LIMIT: RiskLimit(
                limit_type=RiskLimitType.VAR_LIMIT,
                limit_value=Decimal('25000'),  # $25K 95% VaR
                warning_threshold=Decimal('0.85')
            )
        }
    
    @pytest.mark.asyncio
    async def test_position_size_limits(self, risk_system, risk_limits_config):
        """Test position size limit enforcement"""
        # Configure limits
        await risk_system.configure_limits(risk_limits_config)
        
        # Test progressive position building
        position_updates = [
            {'symbol': 'BTC/USDT', 'quantity': Decimal('2.0'), 'side': 'BUY'},
            {'symbol': 'BTC/USDT', 'quantity': Decimal('3.0'), 'side': 'BUY'},
            {'symbol': 'BTC/USDT', 'quantity': Decimal('3.0'), 'side': 'BUY'},
            {'symbol': 'BTC/USDT', 'quantity': Decimal('2.5'), 'side': 'BUY'},  # Would exceed limit
        ]
        
        results = []
        current_position = Decimal('0')
        
        for update in position_updates:
            check_result = await risk_system.check_position_limit(
                symbol=update['symbol'],
                current_position=current_position,
                proposed_change=update['quantity'],
                side=update['side']
            )
            
            results.append(check_result)
            
            if check_result['approved']:
                current_position += update['quantity']
        
        # Verify limit enforcement
        assert results[0]['approved'] == True  # 2.0 BTC - OK
        assert results[1]['approved'] == True  # 5.0 BTC total - OK
        assert results[1]['warning'] == False  # Below warning threshold
        
        assert results[2]['approved'] == True  # 8.0 BTC total - Warning level
        assert results[2]['warning'] == True   # At 80% of limit
        assert 'WARNING' in results[2]['severity']
        
        assert results[3]['approved'] == False  # Would be 10.5 BTC - Exceeds limit
        assert results[3]['breach_type'] == RiskLimitType.MAX_POSITION_SIZE
        assert current_position == Decimal('8.0')  # Position unchanged after rejection
    
    @pytest.mark.asyncio
    async def test_position_value_limits(self, risk_system, risk_limits_config):
        """Test position value limit with price changes"""
        await risk_system.configure_limits(risk_limits_config)
        
        # Initial position
        position = {
            'symbol': 'BTC/USDT',
            'quantity': Decimal('5.0'),
            'entry_price': Decimal('50000')
        }
        
        # Test with different market prices
        test_prices = [
            Decimal('50000'),  # $250K value - OK
            Decimal('80000'),  # $400K value - Warning
            Decimal('100000'), # $500K value - At limit
            Decimal('110000')  # $550K value - Breach
        ]
        
        for price in test_prices:
            check_result = await risk_system.check_position_value_limit(
                position=position,
                current_price=price
            )
            
            position_value = position['quantity'] * price
            
            if position_value <= Decimal('500000'):
                assert check_result['within_limit'] == True
                
                if position_value >= Decimal('425000'):  # 85% warning threshold
                    assert check_result['warning'] == True
                    assert 'reduce_position' in check_result['recommendations']
            else:
                assert check_result['within_limit'] == False
                assert check_result['action_required'] == 'REDUCE_POSITION'
                assert 'suggested_reduction' in check_result
    
    @pytest.mark.asyncio
    async def test_daily_loss_limit(self, risk_system, risk_limits_config):
        """Test daily loss limit tracking and enforcement"""
        await risk_system.configure_limits(risk_limits_config)
        
        # Simulate trading day with P&L updates
        trading_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        pnl_updates = []
        
        # Generate P&L events
        pnl_events = [
            -2000,   # Loss
            1500,    # Profit
            -3000,   # Loss (total -3500)
            -2000,   # Loss (total -5500)
            -1500,   # Loss (total -7000) - Warning level
            500,     # Small profit (total -6500)
            -3000,   # Loss (total -9500) - Near limit
            -1000    # Would breach limit
        ]
        
        cumulative_pnl = Decimal('0')
        
        for i, pnl in enumerate(pnl_events):
            timestamp = trading_day + timedelta(hours=i)
            
            # Check if trade is allowed
            can_trade = await risk_system.check_daily_loss_limit(
                current_daily_pnl=cumulative_pnl,
                potential_loss=Decimal(str(abs(pnl))) if pnl < 0 else Decimal('0')
            )
            
            if can_trade['allowed']:
                cumulative_pnl += Decimal(str(pnl))
                pnl_updates.append({
                    'timestamp': timestamp,
                    'pnl': pnl,
                    'cumulative': cumulative_pnl,
                    'status': can_trade.get('status', 'OK')
                })
            else:
                pnl_updates.append({
                    'timestamp': timestamp,
                    'pnl': 0,  # Trade blocked
                    'cumulative': cumulative_pnl,
                    'status': 'BLOCKED',
                    'reason': 'Daily loss limit would be breached'
                })
        
        # Verify loss limit enforcement
        assert cumulative_pnl > Decimal('-10000')  # Should not exceed limit
        
        # Check warning was triggered
        warning_updates = [u for u in pnl_updates if u.get('status') == 'WARNING']
        assert len(warning_updates) > 0
        
        # Check trade was blocked
        blocked_updates = [u for u in pnl_updates if u.get('status') == 'BLOCKED']
        assert len(blocked_updates) > 0
    
    @pytest.mark.asyncio
    async def test_leverage_limits(self, risk_system, risk_limits_config):
        """Test leverage limit calculation and enforcement"""
        await risk_system.configure_limits(risk_limits_config)
        
        # Test scenarios with different account values and positions
        test_scenarios = [
            {
                'account_equity': Decimal('100000'),
                'positions': [
                    {'symbol': 'BTC/USDT', 'value': Decimal('150000'), 'side': 'LONG'},
                    {'symbol': 'ETH/USDT', 'value': Decimal('50000'), 'side': 'LONG'}
                ],
                'expected_leverage': Decimal('2.0'),
                'should_allow_more': True
            },
            {
                'account_equity': Decimal('100000'),
                'positions': [
                    {'symbol': 'BTC/USDT', 'value': Decimal('200000'), 'side': 'LONG'},
                    {'symbol': 'ETH/USDT', 'value': Decimal('80000'), 'side': 'LONG'}
                ],
                'expected_leverage': Decimal('2.8'),
                'should_allow_more': True  # But with warning
            },
            {
                'account_equity': Decimal('100000'),
                'positions': [
                    {'symbol': 'BTC/USDT', 'value': Decimal('250000'), 'side': 'LONG'},
                    {'symbol': 'ETH/USDT', 'value': Decimal('100000'), 'side': 'SHORT'}
                ],
                'expected_leverage': Decimal('3.5'),
                'should_allow_more': False  # Exceeds limit
            }
        ]
        
        for scenario in test_scenarios:
            leverage_check = await risk_system.check_leverage_limit(
                account_equity=scenario['account_equity'],
                positions=scenario['positions']
            )
            
            # Calculate gross leverage
            gross_exposure = sum(pos['value'] for pos in scenario['positions'])
            actual_leverage = gross_exposure / scenario['account_equity']
            
            assert abs(leverage_check['current_leverage'] - actual_leverage) < Decimal('0.01')
            assert leverage_check['allow_new_positions'] == scenario['should_allow_more']
            
            if actual_leverage > Decimal('2.7'):  # 90% of 3.0 limit
                assert leverage_check['warning'] == True
                assert 'reduce_leverage' in leverage_check['recommendations']
    
    @pytest.mark.asyncio
    async def test_order_rate_limits(self, risk_system, risk_limits_config):
        """Test order rate limiting for market abuse prevention"""
        await risk_system.configure_limits(risk_limits_config)
        
        # Simulate rapid order submission
        order_timestamps = []
        results = []
        
        # Generate 150 order attempts in 90 seconds
        start_time = datetime.now(timezone.utc)
        
        for i in range(150):
            # Accelerate orders over time
            if i < 50:
                delay = 2.0  # 30 orders/minute pace
            elif i < 100:
                delay = 0.8  # 75 orders/minute pace
            else:
                delay = 0.4  # 150 orders/minute pace - exceeds limit
            
            timestamp = start_time + timedelta(seconds=i * delay)
            order_timestamps.append(timestamp)
            
            # Check rate limit
            rate_check = await risk_system.check_order_rate_limit(
                timestamps=order_timestamps,
                current_time=timestamp
            )
            
            results.append({
                'order_num': i,
                'timestamp': timestamp,
                'allowed': rate_check['allowed'],
                'current_rate': rate_check['current_rate'],
                'status': rate_check.get('status', 'OK')
            })
        
        # Analyze results
        allowed_orders = [r for r in results if r['allowed']]
        blocked_orders = [r for r in results if not r['allowed']]
        warning_orders = [r for r in results if r.get('status') == 'WARNING']
        
        assert len(blocked_orders) > 0  # Some orders should be blocked
        assert len(warning_orders) > 0  # Warnings before blocking
        
        # Verify rate calculation
        for result in results[-10:]:  # Check last 10 orders
            if result['current_rate'] > 100:
                assert result['allowed'] == False
    
    @pytest.mark.asyncio
    async def test_var_limits(self, risk_system, risk_limits_config):
        """Test Value at Risk (VaR) limit calculations"""
        await risk_system.configure_limits(risk_limits_config)
        
        # Portfolio for VaR calculation
        portfolio = {
            'positions': [
                {
                    'symbol': 'BTC/USDT',
                    'quantity': Decimal('2.0'),
                    'current_price': Decimal('50000'),
                    'volatility': Decimal('0.04')  # 4% daily volatility
                },
                {
                    'symbol': 'ETH/USDT',
                    'quantity': Decimal('20.0'),
                    'current_price': Decimal('3000'),
                    'volatility': Decimal('0.05')  # 5% daily volatility
                }
            ],
            'correlation_matrix': np.array([[1.0, 0.7], [0.7, 1.0]])  # BTC-ETH correlation
        }
        
        # Calculate VaR
        var_result = await risk_system.calculate_portfolio_var(
            portfolio=portfolio,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        assert 'var_95' in var_result
        assert 'var_99' in var_result
        assert 'component_var' in var_result
        assert 'marginal_var' in var_result
        
        # Check against limit
        limit_check = await risk_system.check_var_limit(
            current_var=var_result['var_95'],
            var_limit=risk_limits_config[RiskLimitType.VAR_LIMIT].limit_value
        )
        
        assert 'within_limit' in limit_check
        assert 'utilization' in limit_check
        
        # Test with increased position that would breach VaR limit
        large_portfolio = portfolio.copy()
        large_portfolio['positions'][0]['quantity'] = Decimal('10.0')  # 5x BTC position
        
        large_var_result = await risk_system.calculate_portfolio_var(
            portfolio=large_portfolio,
            confidence_level=0.95,
            time_horizon_days=1
        )
        
        large_limit_check = await risk_system.check_var_limit(
            current_var=large_var_result['var_95'],
            var_limit=risk_limits_config[RiskLimitType.VAR_LIMIT].limit_value
        )
        
        if large_var_result['var_95'] > Decimal('25000'):
            assert large_limit_check['within_limit'] == False
            assert 'risk_reduction_required' in large_limit_check
    
    @pytest.mark.asyncio
    async def test_concentration_limits(self, risk_system):
        """Test position concentration limits"""
        concentration_limits = {
            'max_single_position': Decimal('0.25'),  # 25% of portfolio
            'max_sector_exposure': Decimal('0.40'),  # 40% per sector
            'max_correlated_exposure': Decimal('0.50')  # 50% for correlated assets
        }
        
        await risk_system.configure_concentration_limits(concentration_limits)
        
        # Test portfolio
        portfolio_value = Decimal('1000000')
        positions = [
            {
                'symbol': 'BTC/USDT',
                'value': Decimal('200000'),  # 20% - OK
                'sector': 'CRYPTO',
                'correlation_group': 'CRYPTO_MAJOR'
            },
            {
                'symbol': 'ETH/USDT',
                'value': Decimal('150000'),  # 15% - OK individually
                'sector': 'CRYPTO',
                'correlation_group': 'CRYPTO_MAJOR'
            },
            {
                'symbol': 'SOL/USDT',
                'value': Decimal('100000'),  # 10% - OK individually
                'sector': 'CRYPTO',
                'correlation_group': 'CRYPTO_MAJOR'
            }
        ]
        
        # Check concentration
        concentration_check = await risk_system.check_concentration_limits(
            positions=positions,
            portfolio_value=portfolio_value
        )
        
        # Individual position check
        assert concentration_check['single_position_breaches'] == []
        
        # Sector concentration (45% in CRYPTO - breach)
        assert len(concentration_check['sector_breaches']) > 0
        assert 'CRYPTO' in concentration_check['sector_breaches'][0]['sector']
        
        # Correlation group (45% in CRYPTO_MAJOR - warning but not breach)
        assert concentration_check['correlation_warnings'] != []
        
        # Test adding position that would breach single position limit
        new_position = {
            'symbol': 'AAPL',
            'proposed_value': Decimal('300000'),  # 30% - would breach
            'sector': 'TECH',
            'correlation_group': 'US_EQUITY'
        }
        
        can_add = await risk_system.check_new_position_concentration(
            new_position=new_position,
            existing_positions=positions,
            portfolio_value=portfolio_value
        )
        
        assert can_add['allowed'] == False
        assert can_add['breach_reason'] == 'SINGLE_POSITION_LIMIT'
        assert 'max_allowed_value' in can_add
    
    @pytest.mark.asyncio
    async def test_margin_and_liquidation_limits(self, risk_system):
        """Test margin usage and liquidation risk limits"""
        margin_config = {
            'initial_margin_requirement': Decimal('0.20'),  # 20%
            'maintenance_margin_requirement': Decimal('0.15'),  # 15%
            'margin_call_level': Decimal('0.17'),  # 17%
            'max_margin_usage': Decimal('0.70')  # 70% of available margin
        }
        
        await risk_system.configure_margin_limits(margin_config)
        
        # Test account
        account = {
            'equity': Decimal('100000'),
            'used_margin': Decimal('0'),
            'free_margin': Decimal('100000'),
            'positions': []
        }
        
        # Progressive margin usage
        test_trades = [
            {'symbol': 'BTC/USDT', 'notional': Decimal('200000'), 'margin_req': Decimal('0.20')},
            {'symbol': 'ETH/USDT', 'notional': Decimal('100000'), 'margin_req': Decimal('0.20')},
            {'symbol': 'EUR/USD', 'notional': Decimal('50000'), 'margin_req': Decimal('0.10')}
        ]
        
        for trade in test_trades:
            # Calculate margin impact
            required_margin = trade['notional'] * trade['margin_req']
            
            margin_check = await risk_system.check_margin_availability(
                account=account,
                required_margin=required_margin
            )
            
            if margin_check['sufficient_margin']:
                # Update account
                account['used_margin'] += required_margin
                account['free_margin'] -= required_margin
                account['positions'].append(trade)
                
                # Check margin health
                health_check = await risk_system.check_margin_health(account)
                
                print(f"After {trade['symbol']}: Margin usage: {health_check['margin_usage_ratio']:.2%}")
                
                if health_check['margin_level'] < Decimal('0.17'):
                    assert health_check['margin_call'] == True
                    assert 'required_deposit' in health_check
        
        # Test liquidation scenarios
        # Simulate market move against positions
        adverse_move = Decimal('0.10')  # 10% adverse move
        
        liquidation_check = await risk_system.check_liquidation_risk(
            account=account,
            market_scenario={'adverse_move': adverse_move}
        )
        
        assert 'liquidation_price_levels' in liquidation_check
        assert 'margin_buffer' in liquidation_check
        assert 'risk_score' in liquidation_check
    
    @pytest.mark.asyncio
    async def test_stress_test_limits(self, risk_system, risk_limits_config):
        """Test portfolio stress testing against extreme scenarios"""
        await risk_system.configure_limits(risk_limits_config)
        
        # Define stress scenarios
        stress_scenarios = [
            {
                'name': 'Market Crash',
                'market_moves': {
                    'BTC/USDT': Decimal('-0.30'),  # -30%
                    'ETH/USDT': Decimal('-0.35'),  # -35%
                    'STOCK': Decimal('-0.20')      # -20%
                },
                'volatility_multiplier': Decimal('3.0'),
                'correlation_breakdown': True
            },
            {
                'name': 'Flash Crash',
                'market_moves': {
                    'BTC/USDT': Decimal('-0.50'),  # -50%
                    'ETH/USDT': Decimal('-0.60'),  # -60%
                    'STOCK': Decimal('-0.40')      # -40%
                },
                'volatility_multiplier': Decimal('5.0'),
                'liquidity_discount': Decimal('0.70')  # 30% liquidity discount
            },
            {
                'name': 'Black Swan',
                'market_moves': {
                    'BTC/USDT': Decimal('-0.80'),  # -80%
                    'ETH/USDT': Decimal('-0.90'),  # -90%
                    'STOCK': Decimal('-0.50')      # -50%
                },
                'volatility_multiplier': Decimal('10.0'),
                'correlation_to_one': True  # All correlations go to 1
            }
        ]
        
        # Test portfolio
        portfolio = {
            'positions': [
                {'symbol': 'BTC/USDT', 'quantity': Decimal('2.0'), 'price': Decimal('50000')},
                {'symbol': 'ETH/USDT', 'quantity': Decimal('10.0'), 'price': Decimal('3000')},
                {'symbol': 'STOCK', 'quantity': Decimal('100'), 'price': Decimal('150')}
            ],
            'cash': Decimal('50000'),
            'total_equity': Decimal('195000')
        }
        
        stress_results = []
        
        for scenario in stress_scenarios:
            result = await risk_system.run_stress_test(
                portfolio=portfolio,
                scenario=scenario
            )
            
            stress_results.append({
                'scenario': scenario['name'],
                'portfolio_loss': result['total_loss'],
                'loss_percentage': result['loss_percentage'],
                'margin_call': result['would_trigger_margin_call'],
                'liquidation': result['would_trigger_liquidation'],
                'survival': result['portfolio_survives']
            })
            
            # Check against stress test limits
            if result['loss_percentage'] > Decimal('0.50'):  # 50% loss
                assert result['exceeds_stress_limit'] == True
                assert 'risk_reduction_required' in result
        
        # Verify at least one scenario triggers limit breach
        severe_breaches = [r for r in stress_results if r['loss_percentage'] > Decimal('0.50')]
        assert len(severe_breaches) > 0
        
        # Generate stress test report
        stress_report = await risk_system.generate_stress_test_report(
            portfolio=portfolio,
            scenarios=stress_scenarios,
            results=stress_results
        )
        
        assert 'worst_case_scenario' in stress_report
        assert 'recommendations' in stress_report
        assert 'risk_metrics' in stress_report
    
    @pytest.mark.asyncio
    async def test_dynamic_limit_adjustment(self, risk_system):
        """Test dynamic risk limit adjustments based on market conditions"""
        # Base limits for normal market
        base_limits = {
            RiskLimitType.MAX_POSITION_SIZE: Decimal('10.0'),
            RiskLimitType.DAILY_LOSS_LIMIT: Decimal('10000'),
            RiskLimitType.MAX_LEVERAGE: Decimal('3.0')
        }
        
        # Market condition indicators
        market_conditions = [
            {
                'volatility': Decimal('0.02'),  # 2% - Normal
                'regime': 'NORMAL',
                'adjustment_factor': Decimal('1.0')
            },
            {
                'volatility': Decimal('0.04'),  # 4% - Elevated
                'regime': 'VOLATILE',
                'adjustment_factor': Decimal('0.7')  # Reduce limits by 30%
            },
            {
                'volatility': Decimal('0.08'),  # 8% - Extreme
                'regime': 'CRISIS',
                'adjustment_factor': Decimal('0.3')  # Reduce limits by 70%
            }
        ]
        
        for condition in market_conditions:
            # Adjust limits based on market conditions
            adjusted_limits = await risk_system.adjust_limits_for_market_conditions(
                base_limits=base_limits,
                market_volatility=condition['volatility'],
                market_regime=condition['regime']
            )
            
            # Verify adjustments
            for limit_type, base_value in base_limits.items():
                adjusted_value = adjusted_limits[limit_type]
                expected_value = base_value * condition['adjustment_factor']
                
                assert abs(adjusted_value - expected_value) < Decimal('0.01')
            
            # Test trading with adjusted limits
            test_order = {
                'symbol': 'BTC/USDT',
                'quantity': Decimal('8.0'),  # Would be OK in normal, blocked in crisis
                'notional': Decimal('400000')
            }
            
            order_check = await risk_system.check_order_against_dynamic_limits(
                order=test_order,
                current_limits=adjusted_limits
            )
            
            if condition['regime'] == 'CRISIS':
                assert order_check['allowed'] == False
                assert 'reduced_limits_in_effect' in order_check
    
    @pytest.mark.asyncio
    async def test_limit_breach_notifications(self, risk_system):
        """Test risk limit breach notification system"""
        # Configure notification handlers
        notifications_received = []
        
        async def notification_handler(breach: RiskBreach):
            notifications_received.append(breach)
        
        await risk_system.register_breach_handler(notification_handler)
        
        # Configure limits with different thresholds
        limits = {
            RiskLimitType.DAILY_LOSS_LIMIT: RiskLimit(
                limit_type=RiskLimitType.DAILY_LOSS_LIMIT,
                limit_value=Decimal('10000'),
                warning_threshold=Decimal('0.7'),  # Warn at $7K loss
                critical_threshold=Decimal('0.9')   # Critical at $9K loss
            )
        }
        
        await risk_system.configure_limits(limits)
        
        # Simulate progressive losses
        loss_levels = [
            Decimal('-5000'),   # 50% - No alert
            Decimal('-7500'),   # 75% - Warning
            Decimal('-9200'),   # 92% - Critical
            Decimal('-10500')   # 105% - Breach
        ]
        
        for loss in loss_levels:
            await risk_system.update_daily_pnl(loss)
            await asyncio.sleep(0.1)  # Allow notifications to process
        
        # Verify notifications
        assert len(notifications_received) >= 3  # Warning, Critical, Breach
        
        # Check notification escalation
        severities = [n.severity for n in notifications_received]
        assert 'WARNING' in severities
        assert 'CRITICAL' in severities
        assert 'BREACH' in severities
        
        # Verify breach details
        breach_notification = next(n for n in notifications_received if n.severity == 'BREACH')
        assert breach_notification.limit_type == RiskLimitType.DAILY_LOSS_LIMIT
        assert breach_notification.actual_value == Decimal('-10500')
        assert breach_notification.limit_value == Decimal('10000')
        assert 'HALT_TRADING' in breach_notification.action_taken
    
    @pytest.mark.asyncio
    async def test_multi_level_risk_controls(self, risk_system):
        """Test cascading risk controls at different levels"""
        # Configure multi-level controls
        risk_hierarchy = {
            'account_level': {
                'max_daily_loss': Decimal('50000'),
                'max_leverage': Decimal('5.0')
            },
            'strategy_level': {
                'GRID_STRATEGY': {
                    'max_daily_loss': Decimal('20000'),
                    'max_positions': 10,
                    'max_position_size': Decimal('5.0')
                },
                'MOMENTUM_STRATEGY': {
                    'max_daily_loss': Decimal('15000'),
                    'max_positions': 5,
                    'max_position_size': Decimal('3.0')
                }
            },
            'instrument_level': {
                'BTC/USDT': {
                    'max_position': Decimal('10.0'),
                    'max_daily_volume': Decimal('50.0')
                },
                'ETH/USDT': {
                    'max_position': Decimal('100.0'),
                    'max_daily_volume': Decimal('500.0')
                }
            }
        }
        
        await risk_system.configure_hierarchical_limits(risk_hierarchy)
        
        # Test order against all levels
        test_order = {
            'strategy': 'GRID_STRATEGY',
            'instrument': 'BTC/USDT',
            'quantity': Decimal('4.0'),
            'side': 'BUY',
            'notional': Decimal('200000')
        }
        
        # Check at each level
        checks = await risk_system.check_hierarchical_limits(test_order)
        
        assert 'account_level' in checks
        assert 'strategy_level' in checks
        assert 'instrument_level' in checks
        
        # All levels should pass for this order
        assert all(check['passed'] for check in checks.values())
        
        # Test order that fails at strategy level
        large_order = test_order.copy()
        large_order['quantity'] = Decimal('6.0')  # Exceeds strategy limit of 5.0
        
        large_checks = await risk_system.check_hierarchical_limits(large_order)
        
        assert large_checks['account_level']['passed'] == True
        assert large_checks['strategy_level']['passed'] == False
        assert large_checks['strategy_level']['failed_limit'] == 'max_position_size'


class TestRiskLimitCompliance:
    """Test regulatory compliance for risk limits"""
    
    @pytest.mark.asyncio
    async def test_mifid_ii_risk_controls(self, risk_system):
        """Test MiFID II required risk controls"""
        # MiFID II Article 17 requirements
        mifid_controls = {
            'pre_trade_controls': {
                'price_collar': Decimal('0.05'),  # 5% from reference price
                'max_order_value': Decimal('1000000'),
                'max_order_quantity': Decimal('1000'),
                'fat_finger_check': True
            },
            'post_trade_controls': {
                'max_position_value': Decimal('5000000'),
                'max_daily_turnover': Decimal('10000000'),
                'max_net_exposure': Decimal('3000000')
            },
            'kill_switch': {
                'enabled': True,
                'triggers': {
                    'max_messages_per_second': 100,
                    'max_orders_per_second': 50,
                    'error_rate_threshold': Decimal('0.05')
                }
            }
        }
        
        await risk_system.configure_mifid_controls(mifid_controls)
        
        # Test pre-trade controls
        test_order = {
            'symbol': 'EUR/USD',
            'price': Decimal('1.2000'),
            'quantity': Decimal('1000000'),  # 1M EUR
            'reference_price': Decimal('1.1800')
        }
        
        pre_trade_check = await risk_system.check_mifid_pre_trade_controls(test_order)
        
        # Price is more than 5% from reference - should fail
        assert pre_trade_check['passed'] == False
        assert 'price_collar' in pre_trade_check['failed_checks']
        
        # Test kill switch
        message_burst = [datetime.now(timezone.utc) + timedelta(milliseconds=i) 
                        for i in range(150)]  # 150 messages in ~150ms
        
        kill_switch_check = await risk_system.check_kill_switch(
            message_timestamps=message_burst
        )
        
        assert kill_switch_check['triggered'] == True
        assert kill_switch_check['reason'] == 'EXCESSIVE_MESSAGE_RATE'
    
    @pytest.mark.asyncio
    async def test_sec_rule_15c3_5_compliance(self, risk_system):
        """Test SEC Rule 15c3-5 (Market Access Rule) compliance"""
        # Configure controls required by Rule 15c3-5
        sec_controls = {
            'financial_controls': {
                'credit_limits': True,
                'capital_thresholds': True,
                'erroneous_order_prevention': True
            },
            'regulatory_controls': {
                'short_sale_compliance': True,
                'trading_halt_compliance': True,
                'position_limits': True
            },
            'system_controls': {
                'message_throttling': True,
                'execution_throttling': True,
                'automated_rejection': True
            }
        }
        
        await risk_system.configure_sec_controls(sec_controls)
        
        # Test erroneous order detection
        erroneous_order = {
            'symbol': 'AAPL',
            'price': Decimal('15000'),  # AAPL at $15,000? Likely error
            'quantity': Decimal('10000'),
            'expected_price_range': (Decimal('150'), Decimal('200'))
        }
        
        error_check = await risk_system.check_erroneous_order(erroneous_order)
        
        assert error_check['is_erroneous'] == True
        assert error_check['reason'] == 'PRICE_OUT_OF_RANGE'
        assert error_check['suggested_action'] == 'REJECT_ORDER'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])