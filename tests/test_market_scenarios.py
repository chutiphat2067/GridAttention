"""
Market Scenarios Tests for GridAttention Trading System
Tests real-world market conditions and events
"""

import asyncio
import pytest
import random
import numpy as np
from datetime import datetime, timedelta, time as datetime_time
from typing import List, Dict, Any
import pandas as pd


class TestMarketEvents:
    """Test various market events and conditions"""
    
    async def test_market_open_volatility(self):
        """Test handling of market open volatility"""
        from market_regime_detector import MarketRegimeDetector
        from grid_strategy_selector import GridStrategySelector
        
        detector = MarketRegimeDetector({})
        selector = GridStrategySelector({})
        
        print("Testing market open volatility...")
        
        # Simulate pre-market to market open transition
        prices = []
        timestamps = []
        
        # Pre-market: low volatility
        base_price = 50000
        current_time = datetime.now().replace(hour=9, minute=0, second=0)
        
        # 30 minutes before open
        for i in range(30):
            price = base_price + random.uniform(-50, 50)
            prices.append(price)
            timestamps.append(current_time - timedelta(minutes=30-i))
        
        # Market open: high volatility spike
        for i in range(30):
            # Volatility increases significantly
            volatility_multiplier = 5 if i < 10 else 3
            price = base_price + random.uniform(-200*volatility_multiplier, 200*volatility_multiplier)
            prices.append(price)
            timestamps.append(current_time + timedelta(minutes=i))
        
        # Detect regime change
        market_data = {
            'prices': prices,
            'timestamps': timestamps,
            'volume_spike': 10.0  # 10x normal volume
        }
        
        regime = await detector.detect_regime(market_data)
        
        # Should detect opening volatility
        assert regime['type'] in ['opening_volatility', 'high_volatility']
        assert regime['confidence'] > 0.7
        
        # Strategy should adapt
        strategy = await selector.select_strategy(regime)
        
        # Should use wider grids and smaller positions
        assert strategy['grid_spacing'] > 0.02  # Wider than normal
        assert strategy['position_size'] < 0.02  # Smaller than normal
        assert strategy['use_time_stops'] is True  # Time-based stops
        
        print(f"  Detected regime: {regime['type']}")
        print(f"  Grid spacing: {strategy['grid_spacing']}")
        print(f"  Position size: {strategy['position_size']}")
    
    async def test_weekend_gap(self):
        """Test handling of weekend gaps"""
        from grid_management_system import GridManagementSystem
        from risk_management_system import RiskManagementSystem
        
        grid_mgr = GridManagementSystem({})
        risk_mgr = RiskManagementSystem({})
        
        print("Testing weekend gap scenario...")
        
        # Friday close price
        friday_close = 50000
        
        # Setup positions before weekend
        positions = [
            {'symbol': 'BTC/USDT', 'side': 'long', 'entry': 49500, 'size': 0.1},
            {'symbol': 'BTC/USDT', 'side': 'long', 'entry': 49800, 'size': 0.1},
            {'symbol': 'BTC/USDT', 'side': 'short', 'entry': 50500, 'size': 0.1}
        ]
        
        for pos in positions:
            await risk_mgr.add_position(pos)
        
        # Simulate weekend gap (3% gap up)
        monday_open = friday_close * 1.03
        gap_percentage = (monday_open - friday_close) / friday_close
        
        print(f"  Friday close: ${friday_close}")
        print(f"  Monday open: ${monday_open}")
        print(f"  Gap: {gap_percentage*100:.1f}%")
        
        # Handle gap
        gap_result = await grid_mgr.handle_weekend_gap({
            'friday_close': friday_close,
            'monday_open': monday_open,
            'gap_percentage': gap_percentage
        })
        
        # Should trigger gap protection
        assert gap_result['gap_protection_triggered']
        assert gap_result['positions_adjusted'] > 0
        
        # Check risk adjustments
        risk_adjustments = await risk_mgr.adjust_for_gap(gap_percentage)
        
        assert risk_adjustments['stop_losses_adjusted'] > 0
        assert risk_adjustments['position_sizes_reduced'] > 0
        assert risk_adjustments['new_orders_blocked'] is True
    
    async def test_news_event_spike(self):
        """Test handling of news-driven price spikes"""
        from event_detector import EventDetector
        from execution_engine import ExecutionEngine
        
        detector = EventDetector({})
        engine = ExecutionEngine({})
        
        print("Testing news event spike...")
        
        # Normal market conditions
        normal_volatility = 0.01  # 1% hourly volatility
        normal_volume = 1000
        
        # News event causes spike
        event_data = {
            'timestamp': datetime.now(),
            'price_change': 0.05,  # 5% in 1 minute
            'volume_spike': 20,  # 20x normal volume
            'social_mentions': 1000,  # High social activity
            'news_sources': 10  # Multiple news sources
        }
        
        # Detect event
        event_detected = await detector.detect_news_event(event_data)
        
        assert event_detected['is_news_event']
        assert event_detected['severity'] == 'high'
        assert event_detected['confidence'] > 0.8
        
        # Execution should adapt
        execution_rules = await engine.get_news_event_rules(event_detected)
        
        # Should implement protective measures
        assert execution_rules['pause_new_orders'] is True
        assert execution_rules['reduce_position_sizes'] is True
        assert execution_rules['widen_spreads'] is True
        assert execution_rules['increase_confirmations'] is True
        
        print(f"  Event severity: {event_detected['severity']}")
        print(f"  Protective measures: {list(execution_rules.keys())}")
    
    async def test_thin_holiday_trading(self):
        """Test handling of thin holiday trading"""
        from market_analyzer import MarketAnalyzer
        
        analyzer = MarketAnalyzer({})
        
        print("Testing thin holiday trading...")
        
        # Simulate holiday trading conditions
        holiday_data = {
            'date': datetime(2024, 12, 25),  # Christmas
            'volume_ratio': 0.2,  # 20% of normal volume
            'spread_widening': 3.0,  # 3x normal spread
            'participants': 100,  # vs normal 1000
            'order_book_depth': 0.3  # 30% of normal depth
        }
        
        # Analyze conditions
        market_condition = await analyzer.analyze_holiday_conditions(holiday_data)
        
        assert market_condition['is_thin_market']
        assert market_condition['liquidity_score'] < 0.3
        assert market_condition['risk_level'] == 'high'
        
        # Get recommended adjustments
        adjustments = await analyzer.get_thin_market_adjustments(market_condition)
        
        assert adjustments['reduce_position_sizes_by'] > 0.5  # 50%+ reduction
        assert adjustments['increase_min_spread_by'] > 2.0  # 2x minimum spread
        assert adjustments['disable_market_orders'] is True
        assert adjustments['require_limit_orders'] is True
        
        print(f"  Liquidity score: {market_condition['liquidity_score']}")
        print(f"  Position reduction: {adjustments['reduce_position_sizes_by']*100}%")
    
    async def test_correlated_asset_movement(self):
        """Test handling of correlated asset movements"""
        from correlation_analyzer import CorrelationAnalyzer
        from portfolio_manager import PortfolioManager
        
        analyzer = CorrelationAnalyzer({})
        portfolio = PortfolioManager({})
        
        print("Testing correlated asset movements...")
        
        # Define correlated assets
        assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # Simulate correlation spike
        price_movements = {
            'BTC/USDT': -0.10,  # -10%
            'ETH/USDT': -0.12,  # -12%
            'SOL/USDT': -0.15,  # -15%
        }
        
        # Calculate correlations
        correlations = await analyzer.calculate_correlations(price_movements)
        
        # Should detect high correlation
        assert correlations['BTC-ETH'] > 0.8
        assert correlations['BTC-SOL'] > 0.7
        assert correlations['correlation_spike'] is True
        
        # Portfolio adjustments
        portfolio_positions = {
            'BTC/USDT': 0.5,  # 50% of portfolio
            'ETH/USDT': 0.3,  # 30% of portfolio
            'SOL/USDT': 0.2   # 20% of portfolio
        }
        
        risk_assessment = await portfolio.assess_correlation_risk(
            portfolio_positions,
            correlations
        )
        
        assert risk_assessment['concentration_risk'] == 'high'
        assert risk_assessment['recommended_reduction'] > 0.3
        assert risk_assessment['diversification_needed'] is True
        
        print(f"  BTC-ETH correlation: {correlations['BTC-ETH']:.2f}")
        print(f"  Portfolio risk: {risk_assessment['concentration_risk']}")
    
    async def test_market_manipulation_detection(self):
        """Test detection of potential market manipulation"""
        from manipulation_detector import ManipulationDetector
        
        detector = ManipulationDetector({})
        
        print("Testing market manipulation detection...")
        
        # Simulate potential manipulation patterns
        
        # 1. Spoofing pattern
        order_book_history = []
        for i in range(10):
            if i < 5:
                # Large orders appear
                order_book_history.append({
                    'timestamp': datetime.now() + timedelta(seconds=i),
                    'bid_size': 100,
                    'ask_size': 1000,  # Large sell wall
                    'spread': 0.001
                })
            else:
                # Large orders disappear without execution
                order_book_history.append({
                    'timestamp': datetime.now() + timedelta(seconds=i),
                    'bid_size': 100,
                    'ask_size': 100,  # Wall disappeared
                    'spread': 0.001
                })
        
        spoofing_result = await detector.detect_spoofing(order_book_history)
        
        assert spoofing_result['potential_spoofing'] is True
        assert spoofing_result['confidence'] > 0.7
        
        # 2. Wash trading pattern
        trades = []
        for i in range(20):
            trades.append({
                'price': 50000 + random.uniform(-10, 10),
                'size': 1.0,  # Uniform size
                'time': datetime.now() + timedelta(seconds=i*3),  # Regular intervals
                'buyer_id': 'USER_A' if i % 2 == 0 else 'USER_B',
                'seller_id': 'USER_B' if i % 2 == 0 else 'USER_A'
            })
        
        wash_result = await detector.detect_wash_trading(trades)
        
        assert wash_result['potential_wash_trading'] is True
        assert wash_result['suspicious_accounts'] == ['USER_A', 'USER_B']
        
        # 3. Pump and dump pattern
        price_volume_data = {
            'phase1': {'price_increase': 0.5, 'volume_increase': 10},  # Pump
            'phase2': {'price_stable': True, 'volume_decrease': 0.8},   # Distribution
            'phase3': {'price_decrease': 0.4, 'volume_spike': 5}        # Dump
        }
        
        pump_dump_result = await detector.detect_pump_dump(price_volume_data)
        
        assert pump_dump_result['potential_pump_dump'] is True
        assert pump_dump_result['current_phase'] == 'dump'
        
        print("  Manipulation patterns detected:")
        print(f"    - Spoofing: {spoofing_result['potential_spoofing']}")
        print(f"    - Wash trading: {wash_result['potential_wash_trading']}")
        print(f"    - Pump & dump: {pump_dump_result['potential_pump_dump']}")


class TestMarketMicrostructure:
    """Test market microstructure scenarios"""
    
    async def test_order_book_imbalance(self):
        """Test handling of severe order book imbalances"""
        from order_book_analyzer import OrderBookAnalyzer
        
        analyzer = OrderBookAnalyzer({})
        
        print("Testing order book imbalance...")
        
        # Severe buy-side imbalance
        order_book = {
            'bids': [
                {'price': 49900, 'size': 50},
                {'price': 49800, 'size': 100},
                {'price': 49700, 'size': 200},
            ],
            'asks': [
                {'price': 50100, 'size': 0.5},  # Very thin asks
                {'price': 50200, 'size': 0.3},
                {'price': 50300, 'size': 0.2},
            ]
        }
        
        imbalance = await analyzer.calculate_imbalance(order_book)
        
        assert imbalance['ratio'] > 100  # 100:1 buy/sell ratio
        assert imbalance['direction'] == 'buy_pressure'
        assert imbalance['severity'] == 'extreme'
        
        # Get trading adjustments
        adjustments = await analyzer.get_imbalance_adjustments(imbalance)
        
        assert adjustments['avoid_market_sells'] is True
        assert adjustments['use_iceberg_orders'] is True
        assert adjustments['reduce_sell_size_by'] > 0.8
    
    async def test_latency_arbitrage_protection(self):
        """Test protection against latency arbitrage"""
        from latency_protector import LatencyProtector
        
        protector = LatencyProtector({})
        
        print("Testing latency arbitrage protection...")
        
        # Detect potential latency arbitrage
        order_flow = []
        
        # Suspicious pattern: rapid order placement/cancellation
        base_time = datetime.now()
        for i in range(10):
            order_flow.append({
                'action': 'place' if i % 2 == 0 else 'cancel',
                'timestamp': base_time + timedelta(microseconds=i*100),  # 100μs intervals
                'size': 10.0,
                'price': 50000
            })
        
        detection = await protector.detect_latency_arbitrage(order_flow)
        
        assert detection['suspicious_pattern'] is True
        assert detection['pattern_type'] == 'quote_stuffing'
        
        # Implement protection
        protection_measures = await protector.get_protection_measures(detection)
        
        assert protection_measures['add_random_delay'] is True
        assert protection_measures['min_delay_ms'] > 0
        assert protection_measures['rate_limit_orders'] is True


# Market scenario test runner
async def run_market_scenario_tests():
    """Run all market scenario tests"""
    print("📈 Running Market Scenario Tests...")
    
    test_classes = [
        TestMarketEvents,
        TestMarketMicrostructure
    ]
    
    results = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing {test_class.__name__}...")
        print(f"{'='*60}")
        
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                print(f"\n▶️  {method_name}")
                method = getattr(test_instance, method_name)
                
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                
                print(f"✅ {method_name} PASSED")
                results.append((method_name, True))
                
            except Exception as e:
                print(f"❌ {method_name} FAILED: {e}")
                results.append((method_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"📊 Market Scenario Test Results: {passed}/{total} passed")
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_market_scenario_tests())
    exit(0 if success else 1)