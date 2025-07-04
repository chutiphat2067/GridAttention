"""
Market surveillance compliance tests for GridAttention trading system.

Ensures compliance with market abuse regulations (MAR, MAD II) and detection
of manipulative trading behaviors, insider trading, and market manipulation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics
from unittest.mock import Mock, patch, AsyncMock
import json

# Import core components
from core.market_surveillance import MarketSurveillanceSystem
from core.pattern_detector import PatternDetector
from core.alert_manager import AlertManager
from core.compliance_manager import ComplianceManager


class MarketAbuseType(Enum):
    """Types of market abuse behaviors"""
    # Market Manipulation
    WASH_TRADING = "WASH_TRADING"
    SPOOFING = "SPOOFING"
    LAYERING = "LAYERING"
    QUOTE_STUFFING = "QUOTE_STUFFING"
    MOMENTUM_IGNITION = "MOMENTUM_IGNITION"
    MARKING_THE_CLOSE = "MARKING_THE_CLOSE"
    
    # Price Manipulation
    RAMPING = "RAMPING"
    BEAR_RAID = "BEAR_RAID"
    PUMP_AND_DUMP = "PUMP_AND_DUMP"
    
    # Information-based
    INSIDER_TRADING = "INSIDER_TRADING"
    FRONT_RUNNING = "FRONT_RUNNING"
    
    # Cross-Market
    CROSS_MARKET_MANIPULATION = "CROSS_MARKET_MANIPULATION"
    INTER_TRADING_VENUE_MANIPULATION = "INTER_TRADING_VENUE_MANIPULATION"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class TradingActivity:
    """Trading activity record"""
    activity_id: str
    timestamp: datetime
    trader_id: str
    instrument: str
    order_type: str  # LIMIT, MARKET, STOP
    side: str  # BUY, SELL
    price: Decimal
    quantity: Decimal
    venue: str
    status: str  # NEW, FILLED, CANCELLED, MODIFIED
    fill_price: Optional[Decimal] = None
    fill_quantity: Optional[Decimal] = None
    cancel_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurveillanceAlert:
    """Market surveillance alert"""
    alert_id: str
    timestamp: datetime
    alert_type: MarketAbuseType
    severity: AlertSeverity
    trader_id: str
    instrument: str
    evidence: List[Dict[str, Any]]
    confidence_score: float
    description: str
    recommended_action: str
    regulatory_requirement: Optional[str] = None
    false_positive: Optional[bool] = None


class TestMarketSurveillance:
    """Test market surveillance functionality"""
    
    @pytest.fixture
    async def surveillance_system(self):
        """Create market surveillance system"""
        return MarketSurveillanceSystem(
            real_time_monitoring=True,
            pattern_detection_enabled=True,
            ml_detection_enabled=True,
            alert_threshold_confidence=0.7
        )
    
    @pytest.fixture
    async def pattern_detector(self):
        """Create pattern detection engine"""
        return PatternDetector(
            algorithms=['statistical', 'ml_based', 'rule_based'],
            sensitivity='high',
            lookback_periods={'short': 60, 'medium': 300, 'long': 1800}  # seconds
        )
    
    @pytest.fixture
    def sample_trading_data(self) -> List[TradingActivity]:
        """Generate sample trading data with various patterns"""
        activities = []
        base_time = datetime.now(timezone.utc)
        
        # Normal trading pattern
        for i in range(100):
            activities.append(TradingActivity(
                activity_id=f'NORMAL_{i}',
                timestamp=base_time + timedelta(seconds=i*30),
                trader_id='TRADER_001',
                instrument='BTC/USDT',
                order_type='LIMIT',
                side='BUY' if i % 2 == 0 else 'SELL',
                price=Decimal('50000') + Decimal(str(np.random.randn() * 100)),
                quantity=Decimal(str(abs(np.random.randn() * 0.1))),
                venue='BINANCE',
                status='FILLED',
                fill_price=Decimal('50000') + Decimal(str(np.random.randn() * 100)),
                fill_quantity=Decimal(str(abs(np.random.randn() * 0.1)))
            ))
        
        return activities
    
    @pytest.mark.asyncio
    async def test_wash_trading_detection(self, surveillance_system):
        """Test detection of wash trading (self-dealing)"""
        # Generate wash trading pattern
        trader_id = 'WASH_TRADER_001'
        base_time = datetime.now(timezone.utc)
        
        wash_trades = []
        for i in range(10):
            # Buy order
            buy_order = TradingActivity(
                activity_id=f'WASH_BUY_{i}',
                timestamp=base_time + timedelta(seconds=i*60),
                trader_id=trader_id,
                instrument='ETH/USDT',
                order_type='LIMIT',
                side='BUY',
                price=Decimal('3000'),
                quantity=Decimal('1.0'),
                venue='EXCHANGE_A',
                status='FILLED',
                fill_price=Decimal('3000'),
                fill_quantity=Decimal('1.0')
            )
            wash_trades.append(buy_order)
            
            # Matching sell order (wash trade)
            sell_order = TradingActivity(
                activity_id=f'WASH_SELL_{i}',
                timestamp=base_time + timedelta(seconds=i*60 + 5),  # 5 seconds later
                trader_id=trader_id,
                instrument='ETH/USDT',
                order_type='LIMIT',
                side='SELL',
                price=Decimal('3000'),
                quantity=Decimal('1.0'),
                venue='EXCHANGE_A',
                status='FILLED',
                fill_price=Decimal('3000'),
                fill_quantity=Decimal('1.0')
            )
            wash_trades.append(sell_order)
        
        # Process trades through surveillance
        alerts = []
        for trade in wash_trades:
            alert = await surveillance_system.process_trading_activity(trade)
            if alert:
                alerts.append(alert)
        
        # Verify wash trading detected
        wash_alerts = [a for a in alerts if a.alert_type == MarketAbuseType.WASH_TRADING]
        assert len(wash_alerts) > 0
        
        # Check alert details
        for alert in wash_alerts:
            assert alert.trader_id == trader_id
            assert alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            assert alert.confidence_score >= 0.8
            assert 'matching_trades' in alert.evidence[0]
            assert 'no_change_in_beneficial_ownership' in alert.description.lower()
        
        # Test wash trading across venues
        cross_venue_wash = [
            TradingActivity(
                activity_id='CROSS_WASH_BUY',
                timestamp=base_time,
                trader_id=trader_id,
                instrument='BTC/USDT',
                order_type='MARKET',
                side='BUY',
                price=Decimal('50000'),
                quantity=Decimal('0.5'),
                venue='EXCHANGE_A',
                status='FILLED'
            ),
            TradingActivity(
                activity_id='CROSS_WASH_SELL',
                timestamp=base_time + timedelta(seconds=30),
                trader_id=trader_id,
                instrument='BTC/USDT',
                order_type='MARKET',
                side='SELL',
                price=Decimal('50000'),
                quantity=Decimal('0.5'),
                venue='EXCHANGE_B',  # Different venue
                status='FILLED'
            )
        ]
        
        cross_alerts = []
        for trade in cross_venue_wash:
            alert = await surveillance_system.process_trading_activity(trade)
            if alert:
                cross_alerts.append(alert)
        
        # Cross-venue wash trading should also be detected
        assert any(a.alert_type == MarketAbuseType.WASH_TRADING for a in cross_alerts)
    
    @pytest.mark.asyncio
    async def test_spoofing_detection(self, surveillance_system, pattern_detector):
        """Test detection of spoofing (fake orders)"""
        trader_id = 'SPOOFER_001'
        base_time = datetime.now(timezone.utc)
        instrument = 'BTC/USDT'
        
        spoofing_pattern = []
        
        # Phase 1: Place large fake sell orders to push price down
        for i in range(5):
            fake_sell = TradingActivity(
                activity_id=f'SPOOF_SELL_{i}',
                timestamp=base_time + timedelta(seconds=i),
                trader_id=trader_id,
                instrument=instrument,
                order_type='LIMIT',
                side='SELL',
                price=Decimal('49900') - Decimal(str(i * 10)),  # Stacked sells
                quantity=Decimal('5.0'),  # Large size
                venue='BINANCE',
                status='NEW'
            )
            spoofing_pattern.append(fake_sell)
        
        # Phase 2: Place genuine buy order at depressed price
        genuine_buy = TradingActivity(
            activity_id='SPOOF_GENUINE_BUY',
            timestamp=base_time + timedelta(seconds=6),
            trader_id=trader_id,
            instrument=instrument,
            order_type='LIMIT',
            side='BUY',
            price=Decimal('49800'),
            quantity=Decimal('0.5'),
            venue='BINANCE',
            status='FILLED',
            fill_price=Decimal('49800'),
            fill_quantity=Decimal('0.5')
        )
        spoofing_pattern.append(genuine_buy)
        
        # Phase 3: Cancel all fake orders
        for i in range(5):
            cancel = TradingActivity(
                activity_id=f'SPOOF_CANCEL_{i}',
                timestamp=base_time + timedelta(seconds=7 + i*0.1),  # Rapid cancellation
                trader_id=trader_id,
                instrument=instrument,
                order_type='CANCEL',
                side='SELL',
                price=Decimal('49900') - Decimal(str(i * 10)),
                quantity=Decimal('5.0'),
                venue='BINANCE',
                status='CANCELLED',
                cancel_reason='USER_CANCELLED',
                metadata={'original_order_id': f'SPOOF_SELL_{i}'}
            )
            spoofing_pattern.append(cancel)
        
        # Analyze pattern
        analysis_result = await pattern_detector.analyze_order_pattern(
            activities=spoofing_pattern,
            trader_id=trader_id
        )
        
        assert analysis_result['pattern_detected'] == True
        assert MarketAbuseType.SPOOFING in analysis_result['detected_patterns']
        
        # Check spoofing indicators
        indicators = analysis_result['spoofing_indicators']
        assert indicators['high_cancel_rate'] == True
        assert indicators['one_sided_pressure'] == True
        assert indicators['rapid_cancellation'] == True
        assert indicators['fill_on_opposite_side'] == True
        
        # Process through surveillance system
        alerts = []
        for activity in spoofing_pattern:
            alert = await surveillance_system.process_trading_activity(activity)
            if alert:
                alerts.append(alert)
        
        # Verify spoofing alert generated
        spoof_alerts = [a for a in alerts if a.alert_type == MarketAbuseType.SPOOFING]
        assert len(spoof_alerts) > 0
        
        alert = spoof_alerts[0]
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.confidence_score >= 0.85
        assert 'cancel_rate' in str(alert.evidence)
        assert alert.regulatory_requirement == 'MAR_SPOOFING_PROHIBITION'
    
    @pytest.mark.asyncio
    async def test_layering_detection(self, surveillance_system):
        """Test detection of layering (similar to spoofing but more complex)"""
        trader_id = 'LAYERING_TRADER_001'
        base_time = datetime.now(timezone.utc)
        
        layering_activities = []
        
        # Create multiple layers of orders
        layers = [
            {'price_offset': -50, 'quantity': 2.0, 'layer': 1},
            {'price_offset': -100, 'quantity': 3.0, 'layer': 2},
            {'price_offset': -150, 'quantity': 4.0, 'layer': 3},
            {'price_offset': -200, 'quantity': 5.0, 'layer': 4}
        ]
        
        # Place layered sell orders
        for i, layer in enumerate(layers):
            for j in range(3):  # Multiple orders per layer
                order = TradingActivity(
                    activity_id=f'LAYER_{i}_{j}',
                    timestamp=base_time + timedelta(seconds=i + j*0.1),
                    trader_id=trader_id,
                    instrument='ETH/USDT',
                    order_type='LIMIT',
                    side='SELL',
                    price=Decimal('3000') + Decimal(str(layer['price_offset'])),
                    quantity=Decimal(str(layer['quantity'])),
                    venue='KRAKEN',
                    status='NEW',
                    metadata={'layer': layer['layer']}
                )
                layering_activities.append(order)
        
        # Execute buy order on opposite side
        execution = TradingActivity(
            activity_id='LAYER_EXECUTION',
            timestamp=base_time + timedelta(seconds=5),
            trader_id=trader_id,
            instrument='ETH/USDT',
            order_type='MARKET',
            side='BUY',
            price=Decimal('2900'),
            quantity=Decimal('1.0'),
            venue='KRAKEN',
            status='FILLED'
        )
        layering_activities.append(execution)
        
        # Cancel layers
        for i, layer in enumerate(layers):
            for j in range(3):
                cancel = TradingActivity(
                    activity_id=f'LAYER_CANCEL_{i}_{j}',
                    timestamp=base_time + timedelta(seconds=6 + i*0.1),
                    trader_id=trader_id,
                    instrument='ETH/USDT',
                    order_type='CANCEL',
                    side='SELL',
                    price=Decimal('3000') + Decimal(str(layer['price_offset'])),
                    quantity=Decimal(str(layer['quantity'])),
                    venue='KRAKEN',
                    status='CANCELLED',
                    metadata={'original_order_id': f'LAYER_{i}_{j}'}
                )
                layering_activities.append(cancel)
        
        # Analyze layering pattern
        alerts = []
        pattern_buffer = []
        
        for activity in layering_activities:
            pattern_buffer.append(activity)
            
            # Analyze when we have enough data
            if len(pattern_buffer) >= 10:
                alert = await surveillance_system.analyze_layering_pattern(
                    recent_activities=pattern_buffer,
                    lookback_seconds=300
                )
                if alert:
                    alerts.append(alert)
        
        # Verify layering detected
        layering_alerts = [a for a in alerts if a.alert_type == MarketAbuseType.LAYERING]
        assert len(layering_alerts) > 0
        
        alert = layering_alerts[0]
        assert alert.severity == AlertSeverity.CRITICAL
        assert 'multiple_price_levels' in str(alert.evidence)
        assert 'opposite_side_execution' in str(alert.evidence)
    
    @pytest.mark.asyncio
    async def test_momentum_ignition_detection(self, surveillance_system):
        """Test detection of momentum ignition strategies"""
        trader_id = 'MOMENTUM_TRADER_001'
        base_time = datetime.now(timezone.utc)
        
        momentum_trades = []
        
        # Phase 1: Aggressive buying to trigger momentum
        for i in range(10):
            aggressive_buy = TradingActivity(
                activity_id=f'MOMENTUM_BUY_{i}',
                timestamp=base_time + timedelta(seconds=i*2),
                trader_id=trader_id,
                instrument='SOL/USDT',
                order_type='MARKET',  # Market orders for immediate impact
                side='BUY',
                price=Decimal('100') + Decimal(str(i * 0.5)),  # Rising prices
                quantity=Decimal('10.0'),  # Large size
                venue='FTX',
                status='FILLED',
                fill_price=Decimal('100') + Decimal(str(i * 0.5))
            )
            momentum_trades.append(aggressive_buy)
        
        # Phase 2: Wait for others to follow (simulated by other traders)
        for i in range(5):
            follower_trade = TradingActivity(
                activity_id=f'FOLLOWER_{i}',
                timestamp=base_time + timedelta(seconds=25 + i),
                trader_id=f'OTHER_TRADER_{i}',
                instrument='SOL/USDT',
                order_type='MARKET',
                side='BUY',
                price=Decimal('105') + Decimal(str(i * 0.3)),
                quantity=Decimal('5.0'),
                venue='FTX',
                status='FILLED'
            )
            momentum_trades.append(follower_trade)
        
        # Phase 3: Original trader sells into the momentum
        for i in range(10):
            profit_taking = TradingActivity(
                activity_id=f'MOMENTUM_SELL_{i}',
                timestamp=base_time + timedelta(seconds=35 + i),
                trader_id=trader_id,
                instrument='SOL/USDT',
                order_type='LIMIT',
                side='SELL',
                price=Decimal('107') - Decimal(str(i * 0.1)),
                quantity=Decimal('10.0'),
                venue='FTX',
                status='FILLED',
                fill_price=Decimal('107') - Decimal(str(i * 0.1))
            )
            momentum_trades.append(profit_taking)
        
        # Analyze momentum ignition pattern
        analysis = await surveillance_system.analyze_momentum_pattern(
            trades=momentum_trades,
            time_window=timedelta(minutes=5)
        )
        
        assert analysis['momentum_ignition_detected'] == True
        assert analysis['initiator'] == trader_id
        assert analysis['price_impact'] > Decimal('5.0')  # 5% price movement
        assert analysis['profit_taking_detected'] == True
        
        # Generate alert
        alert = await surveillance_system.generate_momentum_alert(analysis)
        assert alert.alert_type == MarketAbuseType.MOMENTUM_IGNITION
        assert alert.severity == AlertSeverity.HIGH
        assert 'artificial_price_movement' in alert.description
    
    @pytest.mark.asyncio
    async def test_marking_the_close_detection(self, surveillance_system):
        """Test detection of marking the close manipulation"""
        trader_id = 'CLOSE_MARKER_001'
        
        # Define market close time (e.g., 4:00 PM)
        close_time = datetime.now(timezone.utc).replace(hour=16, minute=0, second=0)
        
        closing_trades = []
        
        # Normal trading during the day
        for i in range(20):
            normal_trade = TradingActivity(
                activity_id=f'NORMAL_DAY_{i}',
                timestamp=close_time - timedelta(hours=3, minutes=i*5),
                trader_id=trader_id,
                instrument='AAPL',
                order_type='LIMIT',
                side='BUY' if i % 2 == 0 else 'SELL',
                price=Decimal('150') + Decimal(str(np.random.randn() * 0.5)),
                quantity=Decimal('100'),
                venue='NASDAQ',
                status='FILLED'
            )
            closing_trades.append(normal_trade)
        
        # Suspicious activity near close
        # Large trades in last 10 minutes
        for i in range(5):
            closing_trade = TradingActivity(
                activity_id=f'CLOSE_TRADE_{i}',
                timestamp=close_time - timedelta(minutes=10-i*2),
                trader_id=trader_id,
                instrument='AAPL',
                order_type='MARKET',  # Market orders near close
                side='BUY',  # All buys to push price up
                price=Decimal('151') + Decimal(str(i * 0.2)),
                quantity=Decimal('5000'),  # Large size
                venue='NASDAQ',
                status='FILLED',
                metadata={'near_close': True}
            )
            closing_trades.append(closing_trade)
        
        # Analyze for marking the close
        analysis = await surveillance_system.analyze_closing_activity(
            trades=closing_trades,
            closing_window_minutes=30,
            market_close_time=close_time
        )
        
        assert analysis['marking_the_close_suspected'] == True
        assert analysis['price_impact_last_30min'] > Decimal('1.0')  # 1% impact
        assert analysis['volume_concentration_near_close'] > 0.5  # >50% of volume
        assert analysis['directional_trading'] == True  # One-sided trading
        
        # Check regulatory flags
        assert analysis['regulatory_concern'] == 'CLOSING_PRICE_MANIPULATION'
        assert analysis['affects_benchmarks'] == True  # Affects closing price benchmarks
    
    @pytest.mark.asyncio
    async def test_cross_market_manipulation(self, surveillance_system):
        """Test detection of cross-market/cross-product manipulation"""
        trader_id = 'CROSS_MARKET_001'
        base_time = datetime.now(timezone.utc)
        
        # Scenario: Manipulate spot to profit from futures position
        cross_market_trades = []
        
        # Step 1: Build futures position
        futures_position = TradingActivity(
            activity_id='FUTURES_POSITION',
            timestamp=base_time,
            trader_id=trader_id,
            instrument='BTC-FUTURES-MAR25',
            order_type='LIMIT',
            side='BUY',
            price=Decimal('51000'),  # Futures trading at premium
            quantity=Decimal('10.0'),  # Large position
            venue='CME',
            status='FILLED',
            metadata={'product_type': 'futures', 'underlying': 'BTC'}
        )
        cross_market_trades.append(futures_position)
        
        # Step 2: Manipulate spot market
        for i in range(15):
            spot_manipulation = TradingActivity(
                activity_id=f'SPOT_MANIP_{i}',
                timestamp=base_time + timedelta(minutes=5+i),
                trader_id=trader_id,
                instrument='BTC/USDT',
                order_type='MARKET',
                side='BUY',
                price=Decimal('50000') + Decimal(str(i * 50)),
                quantity=Decimal('2.0'),
                venue='COINBASE',
                status='FILLED',
                metadata={'product_type': 'spot'}
            )
            cross_market_trades.append(spot_manipulation)
        
        # Step 3: Close futures position at profit
        futures_close = TradingActivity(
            activity_id='FUTURES_CLOSE',
            timestamp=base_time + timedelta(minutes=25),
            trader_id=trader_id,
            instrument='BTC-FUTURES-MAR25',
            order_type='LIMIT',
            side='SELL',
            price=Decimal('51500'),  # Profit from convergence
            quantity=Decimal('10.0'),
            venue='CME',
            status='FILLED',
            metadata={'product_type': 'futures', 'underlying': 'BTC'}
        )
        cross_market_trades.append(futures_close)
        
        # Analyze cross-market patterns
        analysis = await surveillance_system.analyze_cross_market_activity(
            trades=cross_market_trades,
            correlation_threshold=0.8
        )
        
        assert analysis['cross_market_manipulation_detected'] == True
        assert analysis['linked_products'] == ['BTC-FUTURES-MAR25', 'BTC/USDT']
        assert analysis['manipulation_type'] == 'SPOT_FUTURES_MANIPULATION'
        assert analysis['estimated_profit'] > Decimal('1000')
        
        # Generate alert
        alert = await surveillance_system.generate_cross_market_alert(
            analysis=analysis,
            trader_id=trader_id
        )
        
        assert alert.alert_type == MarketAbuseType.CROSS_MARKET_MANIPULATION
        assert alert.severity == AlertSeverity.CRITICAL
        assert 'multiple_venues' in str(alert.evidence)
        assert alert.regulatory_requirement == 'MAR_CROSS_PRODUCT_MANIPULATION'
    
    @pytest.mark.asyncio
    async def test_insider_trading_detection(self, surveillance_system):
        """Test detection of potential insider trading"""
        # Historical price data
        price_history = []
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Normal trading pattern for 25 days
        for day in range(25):
            for hour in range(8):  # Trading hours
                price_history.append({
                    'timestamp': base_time + timedelta(days=day, hours=9+hour),
                    'price': Decimal('100') + Decimal(str(np.random.randn() * 2)),
                    'volume': Decimal('1000000')  # Normal volume
                })
        
        # Suspicious trading before announcement
        insider_trades = []
        announcement_time = base_time + timedelta(days=26, hours=16)  # After market
        
        # Unusual activity 2 days before announcement
        suspicious_trader = 'INSIDER_001'
        
        for i in range(10):
            insider_trade = TradingActivity(
                activity_id=f'INSIDER_{i}',
                timestamp=announcement_time - timedelta(days=2, hours=i),
                trader_id=suspicious_trader,
                instrument='TECH_CORP',
                order_type='LIMIT',
                side='BUY',
                price=Decimal('100') + Decimal(str(i * 0.1)),
                quantity=Decimal('10000'),  # Large position
                venue='NYSE',
                status='FILLED',
                metadata={
                    'unusual_size': True,
                    'first_time_trader': True,  # Never traded this stock before
                    'account_age_days': 400
                }
            )
            insider_trades.append(insider_trade)
        
        # Price spike after announcement
        post_announcement_price = Decimal('120')  # 20% gain
        
        # Analyze for insider trading
        analysis = await surveillance_system.analyze_insider_trading_pattern(
            trades=insider_trades,
            price_history=price_history,
            material_event={
                'timestamp': announcement_time,
                'type': 'merger_announcement',
                'price_impact': Decimal('20.0')  # 20% impact
            }
        )
        
        assert analysis['insider_trading_suspected'] == True
        assert analysis['timing_suspicious'] == True
        assert analysis['size_unusual'] == True
        assert analysis['trader_profile_suspicious'] == True
        assert analysis['profit_from_event'] > Decimal('100000')
        
        # Check statistical significance
        assert analysis['statistical_anomaly'] == True
        assert analysis['standard_deviations_from_mean'] > 3.0
        
        # Generate insider trading alert
        alert = await surveillance_system.generate_insider_trading_alert(
            analysis=analysis,
            trader_id=suspicious_trader
        )
        
        assert alert.alert_type == MarketAbuseType.INSIDER_TRADING
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.regulatory_requirement == 'MAR_INSIDER_DEALING'
        assert 'immediate_escalation' in alert.recommended_action
    
    @pytest.mark.asyncio
    async def test_quote_stuffing_detection(self, surveillance_system):
        """Test detection of quote stuffing (overwhelming the market with orders)"""
        trader_id = 'QUOTE_STUFFER_001'
        base_time = datetime.now(timezone.utc)
        
        quote_stuffing_orders = []
        
        # Generate massive number of orders in short time
        for i in range(1000):  # 1000 orders
            order = TradingActivity(
                activity_id=f'STUFF_{i}',
                timestamp=base_time + timedelta(milliseconds=i*10),  # 10ms apart
                trader_id=trader_id,
                instrument='SPY',
                order_type='LIMIT',
                side='BUY' if i % 2 == 0 else 'SELL',
                price=Decimal('400') + Decimal(str((i % 10) * 0.01)),
                quantity=Decimal('100'),
                venue='NASDAQ',
                status='NEW'
            )
            quote_stuffing_orders.append(order)
            
            # Cancel most orders quickly
            if i % 10 != 0:  # Cancel 90% of orders
                cancel = TradingActivity(
                    activity_id=f'STUFF_CANCEL_{i}',
                    timestamp=base_time + timedelta(milliseconds=i*10 + 5),  # 5ms later
                    trader_id=trader_id,
                    instrument='SPY',
                    order_type='CANCEL',
                    side='BUY' if i % 2 == 0 else 'SELL',
                    price=Decimal('400') + Decimal(str((i % 10) * 0.01)),
                    quantity=Decimal('100'),
                    venue='NASDAQ',
                    status='CANCELLED',
                    metadata={'original_order_id': f'STUFF_{i}'}
                )
                quote_stuffing_orders.append(cancel)
        
        # Analyze message rate
        rate_analysis = await surveillance_system.analyze_message_rate(
            activities=quote_stuffing_orders,
            window_seconds=10
        )
        
        assert rate_analysis['quote_stuffing_detected'] == True
        assert rate_analysis['message_rate_per_second'] > 100
        assert rate_analysis['cancel_rate'] > 0.85  # >85% cancellation
        assert rate_analysis['order_to_trade_ratio'] > 100  # Very high ratio
        
        # Check system impact
        assert rate_analysis['potential_system_impact'] == 'HIGH'
        assert rate_analysis['latency_injection_suspected'] == True
        
        # Generate alert
        alert = await surveillance_system.generate_quote_stuffing_alert(
            analysis=rate_analysis,
            trader_id=trader_id
        )
        
        assert alert.alert_type == MarketAbuseType.QUOTE_STUFFING
        assert alert.severity == AlertSeverity.HIGH
        assert 'message_rate_limit' in alert.recommended_action
    
    @pytest.mark.asyncio
    async def test_ml_based_anomaly_detection(self, surveillance_system):
        """Test machine learning based anomaly detection"""
        # Train ML model on normal patterns
        normal_patterns = []
        
        # Generate normal trading patterns
        for day in range(30):
            daily_trades = []
            base_time = datetime.now(timezone.utc) - timedelta(days=30-day)
            
            for hour in range(8):
                for trade_num in range(50):
                    trade = {
                        'timestamp': base_time + timedelta(hours=9+hour, minutes=trade_num),
                        'price_return': np.random.normal(0, 0.001),  # 0.1% volatility
                        'volume': np.random.lognormal(10, 1),
                        'bid_ask_spread': np.random.uniform(0.01, 0.05),
                        'order_imbalance': np.random.normal(0, 0.1),
                        'trade_size': np.random.lognormal(8, 1.5),
                        'message_rate': np.random.poisson(10)
                    }
                    normal_patterns.append(trade)
        
        # Train anomaly detection model
        await surveillance_system.train_anomaly_model(
            training_data=normal_patterns,
            features=['price_return', 'volume', 'bid_ask_spread', 
                     'order_imbalance', 'trade_size', 'message_rate']
        )
        
        # Test with anomalous patterns
        anomalous_patterns = [
            {
                'timestamp': datetime.now(timezone.utc),
                'price_return': 0.05,  # 5% return - highly anomalous
                'volume': 1e8,  # Extremely high volume
                'bid_ask_spread': 0.5,  # Very wide spread
                'order_imbalance': 0.9,  # Extreme imbalance
                'trade_size': 1e6,  # Huge trade
                'message_rate': 1000  # Extremely high message rate
            }
        ]
        
        # Detect anomalies
        anomaly_results = await surveillance_system.detect_anomalies(
            data=anomalous_patterns,
            contamination=0.01  # Expect 1% anomalies
        )
        
        assert len(anomaly_results['anomalies']) > 0
        
        anomaly = anomaly_results['anomalies'][0]
        assert anomaly['anomaly_score'] > 0.9  # High anomaly score
        assert 'contributing_features' in anomaly
        assert 'price_return' in anomaly['contributing_features']
        assert 'volume' in anomaly['contributing_features']
        
        # Generate ML-based alert
        ml_alert = await surveillance_system.generate_ml_anomaly_alert(
            anomaly=anomaly,
            confidence_threshold=0.8
        )
        
        assert ml_alert is not None
        assert ml_alert.description.startswith('ML-detected anomaly')
        assert ml_alert.confidence_score > 0.8
    
    @pytest.mark.asyncio
    async def test_regulatory_reporting(self, surveillance_system):
        """Test regulatory reporting requirements for detected market abuse"""
        # Create a critical alert that requires regulatory reporting
        critical_alert = SurveillanceAlert(
            alert_id='CRITICAL_001',
            timestamp=datetime.now(timezone.utc),
            alert_type=MarketAbuseType.INSIDER_TRADING,
            severity=AlertSeverity.CRITICAL,
            trader_id='SUSPICIOUS_TRADER',
            instrument='TECH_STOCK',
            evidence=[
                {
                    'type': 'unusual_trading',
                    'details': 'Large position before material announcement',
                    'profit': '500000'
                }
            ],
            confidence_score=0.95,
            description='Suspected insider trading before merger announcement',
            recommended_action='Immediate regulatory notification required',
            regulatory_requirement='MAR Article 16'
        )
        
        # Generate STOR (Suspicious Transaction and Order Report)
        stor_report = await surveillance_system.generate_stor_report(critical_alert)
        
        # Verify STOR contains required fields
        assert stor_report['report_type'] == 'STOR'
        assert stor_report['submission_deadline'] == 'T+2'  # 2 business days
        assert 'transaction_details' in stor_report
        assert 'suspicion_indicators' in stor_report
        assert 'supporting_evidence' in stor_report
        
        # Check transaction details
        tx_details = stor_report['transaction_details']
        assert 'order_identification' in tx_details
        assert 'trader_identification' in tx_details
        assert 'instrument_identification' in tx_details
        assert 'venue_identification' in tx_details
        assert 'timestamps' in tx_details
        
        # Verify regulatory format compliance
        assert stor_report['format'] == 'ISO20022_XML'
        assert 'regulatory_technical_standards' in stor_report
        
        # Test automatic submission
        submission_result = await surveillance_system.submit_regulatory_report(
            report=stor_report,
            regulator='FCA',  # Or appropriate regulator
            test_mode=True
        )
        
        assert submission_result['status'] == 'SUBMITTED'
        assert 'submission_id' in submission_result
        assert 'acknowledgment' in submission_result
    
    @pytest.mark.asyncio
    async def test_alert_management_workflow(self, surveillance_system):
        """Test alert management and investigation workflow"""
        # Generate test alert
        test_alert = SurveillanceAlert(
            alert_id='WORKFLOW_001',
            timestamp=datetime.now(timezone.utc),
            alert_type=MarketAbuseType.SPOOFING,
            severity=AlertSeverity.HIGH,
            trader_id='TRADER_123',
            instrument='EUR/USD',
            evidence=[{'type': 'high_cancel_rate', 'rate': 0.95}],
            confidence_score=0.85,
            description='Potential spoofing detected',
            recommended_action='Review required'
        )
        
        # Alert triage
        triage_result = await surveillance_system.triage_alert(test_alert)
        
        assert triage_result['priority'] in ['HIGH', 'MEDIUM', 'LOW']
        assert 'assigned_to' in triage_result
        assert 'investigation_required' in triage_result
        
        # Investigation workflow
        investigation = await surveillance_system.create_investigation(
            alert=test_alert,
            investigator='compliance_analyst_001'
        )
        
        assert investigation['investigation_id'] is not None
        assert investigation['status'] == 'OPEN'
        assert 'checklist' in investigation
        
        # Add investigation findings
        findings = {
            'trader_history': 'No previous violations',
            'pattern_analysis': 'Confirmed spoofing pattern',
            'intent_assessment': 'Likely intentional',
            'market_impact': 'Minimal impact observed',
            'recommendation': 'Warning letter'
        }
        
        updated_investigation = await surveillance_system.update_investigation(
            investigation_id=investigation['investigation_id'],
            findings=findings,
            status='UNDER_REVIEW'
        )
        
        assert updated_investigation['status'] == 'UNDER_REVIEW'
        assert 'findings' in updated_investigation
        
        # Close investigation
        closure = await surveillance_system.close_investigation(
            investigation_id=investigation['investigation_id'],
            outcome='WARNING_ISSUED',
            false_positive=False,
            lessons_learned='Enhanced spoofing detection parameters'
        )
        
        assert closure['status'] == 'CLOSED'
        assert closure['outcome'] == 'WARNING_ISSUED'
        assert 'closure_report' in closure
        
        # Update ML models with feedback
        await surveillance_system.update_models_with_feedback(
            alert_id=test_alert.alert_id,
            was_true_positive=True,
            severity_accurate=True
        )


class TestMarketSurveillanceCompliance:
    """Test regulatory compliance aspects of market surveillance"""
    
    @pytest.mark.asyncio
    async def test_mar_compliance(self, surveillance_system):
        """Test compliance with Market Abuse Regulation (MAR)"""
        # Test MAR Article 12 - Market manipulation prohibitions
        mar_prohibitions = [
            MarketAbuseType.SPOOFING,
            MarketAbuseType.LAYERING,
            MarketAbuseType.WASH_TRADING,
            MarketAbuseType.MOMENTUM_IGNITION
        ]
        
        compliance_check = await surveillance_system.verify_mar_compliance(
            monitored_behaviors=mar_prohibitions
        )
        
        assert compliance_check['all_prohibitions_monitored'] == True
        assert compliance_check['detection_mechanisms_active'] == True
        assert compliance_check['reporting_mechanism_available'] == True
        
        # Test MAR Article 16 - Reporting obligations
        reporting_check = await surveillance_system.verify_reporting_compliance()
        
        assert reporting_check['stor_capability'] == True
        assert reporting_check['submission_timeline_compliant'] == True
        assert reporting_check['data_retention_compliant'] == True  # 5 years
    
    @pytest.mark.asyncio
    async def test_mifid_ii_algo_surveillance(self, surveillance_system):
        """Test MiFID II requirements for algorithmic trading surveillance"""
        # Test RTS 6 requirements
        algo_surveillance_check = await surveillance_system.verify_algo_surveillance_compliance()
        
        required_capabilities = [
            'real_time_monitoring',
            'kill_switch_capability',
            'audit_trail_complete',
            'parameter_recording',
            'testing_documentation'
        ]
        
        for capability in required_capabilities:
            assert algo_surveillance_check[capability] == True
        
        # Test market making obligations monitoring
        mm_compliance = await surveillance_system.verify_market_making_compliance(
            trader_id='MM_ALGO_001',
            instrument='EUR/USD',
            requirements={
                'minimum_presence': 0.5,  # 50% of trading day
                'maximum_spread': Decimal('0.001'),
                'minimum_size': Decimal('100000')
            }
        )
        
        assert 'presence_percentage' in mm_compliance
        assert 'spread_compliance' in mm_compliance
        assert 'size_compliance' in mm_compliance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])