"""
Compliance and Regulatory Tests for GridAttention Trading System
Tests regulatory compliance, audit trails, and reporting requirements
"""

import asyncio
import pytest
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from decimal import Decimal


class TestTradeReporting:
    """Test trade reporting compliance"""
    
    async def test_trade_record_completeness(self):
        """Test that all required trade fields are recorded"""
        from trade_recorder import TradeRecorder
        
        recorder = TradeRecorder({})
        
        print("Testing trade record completeness...")
        
        # Execute a trade
        trade = {
            'id': 'TRD_123456',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': Decimal('50000.00'),
            'amount': Decimal('0.01'),
            'timestamp': datetime.now(),
            'exchange': 'binance',
            'order_type': 'limit'
        }
        
        # Record trade
        recorded = await recorder.record_trade(trade)
        
        # Verify all required fields for compliance
        required_fields = [
            'trade_id',
            'symbol',
            'side',
            'price',
            'amount',
            'value',
            'timestamp',
            'exchange',
            'order_type',
            'fees',
            'fee_currency',
            'execution_venue',
            'counterparty',
            'settlement_date',
            'regulatory_status'
        ]
        
        for field in required_fields:
            assert field in recorded
            assert recorded[field] is not None
        
        # Verify calculations
        assert recorded['value'] == trade['price'] * trade['amount']
        assert recorded['timestamp'].tzinfo is not None  # Must have timezone
        
        print(f"  ✓ All {len(required_fields)} required fields present")
        print(f"  ✓ Trade value: ${recorded['value']}")
    
    async def test_reporting_accuracy(self):
        """Test accuracy of regulatory reports"""
        from reporting_engine import ReportingEngine
        
        engine = ReportingEngine({})
        
        print("Testing reporting accuracy...")
        
        # Generate test trades
        trades = []
        for i in range(100):
            trades.append({
                'timestamp': datetime.now() - timedelta(days=i),
                'symbol': 'BTC/USDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': Decimal('50000') + Decimal(str(i * 10)),
                'amount': Decimal('0.01'),
                'fees': Decimal('0.0001')
            })
        
        # Generate daily report
        report = await engine.generate_daily_report(trades, datetime.now().date())
        
        # Verify report accuracy
        assert report['total_trades'] == len(trades)
        assert report['total_volume'] == sum(t['amount'] for t in trades)
        assert report['total_fees'] == sum(t['fees'] for t in trades)
        
        # Verify P&L calculation
        buys = [t for t in trades if t['side'] == 'buy']
        sells = [t for t in trades if t['side'] == 'sell']
        
        buy_value = sum(t['price'] * t['amount'] for t in buys)
        sell_value = sum(t['price'] * t['amount'] for t in sells)
        expected_pnl = sell_value - buy_value - report['total_fees']
        
        assert abs(report['realized_pnl'] - expected_pnl) < Decimal('0.01')
        
        print(f"  ✓ Report covers {report['total_trades']} trades")
        print(f"  ✓ P&L accuracy: ${report['realized_pnl']}")
    
    async def test_audit_trail_integrity(self):
        """Test audit trail integrity and immutability"""
        from audit_logger import AuditLogger
        
        logger = AuditLogger({})
        
        print("Testing audit trail integrity...")
        
        # Log series of events
        events = []
        for i in range(10):
            event = {
                'type': 'order_placed',
                'timestamp': datetime.now(),
                'user': 'system',
                'details': {
                    'order_id': f'ORD_{i}',
                    'symbol': 'BTC/USDT',
                    'amount': 0.01
                }
            }
            
            logged_event = await logger.log_event(event)
            events.append(logged_event)
        
        # Verify integrity
        for event in events:
            # Each event should have a hash
            assert 'hash' in event
            assert 'previous_hash' in event
            
            # Verify hash chain
            if event != events[0]:
                idx = events.index(event)
                assert event['previous_hash'] == events[idx-1]['hash']
            
            # Verify event cannot be modified
            original_hash = event['hash']
            event['details']['amount'] = 0.02  # Try to modify
            
            # Recalculate hash
            calculated_hash = await logger.calculate_hash(event)
            assert calculated_hash != original_hash  # Modification detected
        
        print(f"  ✓ Audit trail contains {len(events)} immutable events")
        print(f"  ✓ Hash chain integrity verified")


class TestRiskCompliance:
    """Test risk management compliance"""
    
    async def test_position_limit_enforcement(self):
        """Test position limit enforcement"""
        from risk_compliance import RiskComplianceManager
        
        manager = RiskComplianceManager({
            'max_position_size': Decimal('0.05'),  # 5% max
            'max_leverage': Decimal('3.0'),
            'max_concentration': Decimal('0.25')  # 25% in single asset
        })
        
        print("Testing position limit enforcement...")
        
        # Test position size limit
        portfolio_value = Decimal('100000')
        
        # Try to open position exceeding limit
        order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': Decimal('0.1'),  # Would be 10% at $50k
            'price': Decimal('50000')
        }
        
        compliance_check = await manager.check_order_compliance(order, portfolio_value)
        
        assert not compliance_check['compliant']
        assert 'position_size_limit' in compliance_check['violations']
        
        # Verify suggested adjustment
        assert compliance_check['suggested_amount'] <= Decimal('0.05')
        
        print(f"  ✓ Position limit enforced: {compliance_check['violations']}")
        print(f"  ✓ Suggested adjustment: {compliance_check['suggested_amount']}")
    
    async def test_leverage_monitoring(self):
        """Test leverage monitoring and alerts"""
        from leverage_monitor import LeverageMonitor
        
        monitor = LeverageMonitor({})
        
        print("Testing leverage monitoring...")
        
        # Current positions
        positions = [
            {'symbol': 'BTC/USDT', 'notional': Decimal('50000'), 'margin': Decimal('10000')},
            {'symbol': 'ETH/USDT', 'notional': Decimal('30000'), 'margin': Decimal('10000')},
        ]
        
        account_balance = Decimal('25000')
        
        # Calculate leverage
        leverage_report = await monitor.calculate_leverage(positions, account_balance)
        
        assert leverage_report['total_notional'] == Decimal('80000')
        assert leverage_report['total_margin'] == Decimal('20000')
        assert leverage_report['account_leverage'] == Decimal('3.2')  # 80k/25k
        
        # Check if alert triggered
        alerts = await monitor.check_leverage_alerts(leverage_report)
        
        assert len(alerts) > 0
        assert any(a['type'] == 'leverage_warning' for a in alerts)
        
        print(f"  ✓ Leverage calculated: {leverage_report['account_leverage']}x")
        print(f"  ✓ Alerts generated: {len(alerts)}")


class TestDataPrivacy:
    """Test data privacy and protection compliance"""
    
    async def test_pii_protection(self):
        """Test personally identifiable information protection"""
        from data_protector import DataProtector
        
        protector = DataProtector({})
        
        print("Testing PII protection...")
        
        # User data with PII
        user_data = {
            'user_id': 'USR_12345',
            'email': 'trader@example.com',
            'phone': '+1-555-123-4567',
            'api_key': 'sk_live_abcdef123456',
            'trading_history': [
                {'date': '2024-01-01', 'profit': 1000}
            ]
        }
        
        # Anonymize for analytics
        anonymized = await protector.anonymize_user_data(user_data)
        
        # Verify PII is protected
        assert anonymized['user_id'] != user_data['user_id']
        assert '@' not in anonymized.get('email', '')
        assert 'phone' not in anonymized
        assert 'api_key' not in anonymized
        
        # Trading data should be preserved
        assert 'trading_history' in anonymized
        assert anonymized['trading_history'][0]['profit'] == 1000
        
        print(f"  ✓ PII fields removed/anonymized")
        print(f"  ✓ Trading data preserved")
    
    async def test_data_retention_policy(self):
        """Test data retention policy compliance"""
        from data_retention import DataRetentionManager
        
        manager = DataRetentionManager({
            'trade_retention_days': 2555,  # 7 years
            'log_retention_days': 90,
            'tick_retention_days': 30
        })
        
        print("Testing data retention policy...")
        
        # Check data for deletion
        data_types = {
            'trades': datetime.now() - timedelta(days=2600),  # Over 7 years
            'logs': datetime.now() - timedelta(days=100),     # Over 90 days
            'ticks': datetime.now() - timedelta(days=40),     # Over 30 days
            'recent_trades': datetime.now() - timedelta(days=100)  # Recent
        }
        
        retention_check = await manager.check_retention_compliance(data_types)
        
        assert retention_check['trades']['should_delete'] is True
        assert retention_check['logs']['should_delete'] is True
        assert retention_check['ticks']['should_delete'] is True
        assert retention_check['recent_trades']['should_delete'] is False
        
        # Verify archival before deletion
        for data_type, check in retention_check.items():
            if check['should_delete']:
                assert check['archive_required'] is True
        
        print(f"  ✓ Retention policy checked for {len(data_types)} data types")
        print(f"  ✓ Archival required before deletion")
    
    async def test_gdpr_compliance(self):
        """Test GDPR compliance features"""
        from gdpr_manager import GDPRManager
        
        manager = GDPRManager({})
        
        print("Testing GDPR compliance...")
        
        user_id = 'USR_12345'
        
        # Test right to access
        user_data = await manager.export_user_data(user_id)
        
        assert 'personal_data' in user_data
        assert 'trading_data' in user_data
        assert 'processing_history' in user_data
        assert user_data['export_format'] == 'json'
        
        # Test right to erasure
        erasure_request = await manager.process_erasure_request(user_id)
        
        assert erasure_request['status'] == 'scheduled'
        assert erasure_request['retention_obligations'] is not None
        assert erasure_request['estimated_completion'] is not None
        
        # Test consent management
        consent_status = await manager.get_consent_status(user_id)
        
        assert 'data_processing' in consent_status
        assert 'marketing' in consent_status
        assert consent_status['last_updated'] is not None
        
        print(f"  ✓ Data export available")
        print(f"  ✓ Erasure request processed")
        print(f"  ✓ Consent tracking active")


class TestMarketAbuseDetection:
    """Test market abuse detection and prevention"""
    
    async def test_insider_trading_detection(self):
        """Test detection of potential insider trading patterns"""
        from market_surveillance import MarketSurveillance
        
        surveillance = MarketSurveillance({})
        
        print("Testing insider trading detection...")
        
        # Suspicious pattern: large trades before news
        trades = [
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'symbol': 'XYZ/USDT',
                'side': 'buy',
                'amount': 10000,  # Unusually large
                'price': 10.0,
                'user_id': 'USR_SUSPECT'
            }
        ]
        
        news_event = {
            'timestamp': datetime.now(),
            'symbol': 'XYZ/USDT',
            'type': 'earnings_beat',
            'price_impact': 0.20  # 20% price increase
        }
        
        # Analyze pattern
        analysis = await surveillance.analyze_pre_news_trading(trades, news_event)
        
        assert analysis['suspicious_activity'] is True
        assert analysis['confidence'] > 0.7
        assert 'USR_SUSPECT' in analysis['flagged_users']
        
        # Generate report
        report = await surveillance.generate_surveillance_report(analysis)
        
        assert report['report_type'] == 'potential_insider_trading'
        assert report['regulatory_action_required'] is True
        
        print(f"  ✓ Suspicious pattern detected")
        print(f"  ✓ Report generated for regulatory review")
    
    async def test_market_manipulation_reporting(self):
        """Test market manipulation reporting"""
        from compliance_reporter import ComplianceReporter
        
        reporter = ComplianceReporter({})
        
        print("Testing market manipulation reporting...")
        
        # Detected manipulation
        manipulation_event = {
            'type': 'spoofing',
            'timestamp': datetime.now(),
            'symbol': 'BTC/USDT',
            'evidence': {
                'orders_placed': 50,
                'orders_cancelled': 49,
                'execution_rate': 0.02,
                'pattern_duration': 300  # seconds
            },
            'suspected_account': 'ACC_12345'
        }
        
        # File regulatory report
        report = await reporter.file_manipulation_report(manipulation_event)
        
        assert report['report_id'] is not None
        assert report['filed_with'] == ['exchange', 'regulator']
        assert report['status'] == 'submitted'
        assert report['follow_up_required'] is True
        
        print(f"  ✓ Manipulation report filed: {report['report_id']}")
        print(f"  ✓ Submitted to: {report['filed_with']}")


# Compliance test runner
async def run_compliance_tests():
    """Run all compliance and regulatory tests"""
    print("📋 Running Compliance and Regulatory Tests...")
    
    test_classes = [
        TestTradeReporting,
        TestRiskCompliance,
        TestDataPrivacy,
        TestMarketAbuseDetection
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
    print(f"📊 Compliance Test Results: {passed}/{total} passed")
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_compliance_tests())
    exit(0 if success else 1)