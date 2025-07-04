"""
Trade reporting compliance tests for GridAttention trading system.

Ensures all trades are properly reported according to regulatory requirements,
including MiFID II, EMIR, and other relevant regulations for algorithmic trading.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import json
import hashlib
import uuid
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from collections import defaultdict

# Import core components (adjust based on actual module structure)
from core.trade_reporter import TradeReporter
from core.execution_engine import ExecutionEngine
from core.compliance_manager import ComplianceManager


@dataclass
class TradeReport:
    """Trade report structure for regulatory compliance"""
    # Transaction Reference Number (unique identifier)
    transaction_id: str
    
    # Execution details
    execution_timestamp: datetime
    trading_day: str
    
    # Instrument identification
    instrument_id: str
    instrument_type: str
    isin: Optional[str]
    
    # Trade details
    price: Decimal
    quantity: Decimal
    currency: str
    side: str  # BUY/SELL
    
    # Venue information
    venue: str
    venue_transaction_id: str
    
    # Counterparty information
    counterparty_id: Optional[str]
    
    # Algorithm identification
    algorithm_id: str
    algorithm_version: str
    
    # Flags and indicators
    short_selling_indicator: bool
    otc_post_trade_indicator: bool
    commodity_derivative_indicator: bool
    securities_financing_transaction: bool
    
    # Capacity and client info
    trading_capacity: str  # DEAL/MATCH/AOTC
    client_id: Optional[str]
    investment_decision_maker: str
    execution_decision_maker: str
    
    # Regulatory flags
    waiver_indicator: Optional[str]
    deferral_indicator: Optional[str]
    
    # Additional compliance fields
    best_execution_flags: Dict[str, bool]
    market_abuse_flags: Dict[str, bool]


class TestTradeReporting:
    """Test trade reporting compliance"""
    
    @pytest.fixture
    async def trade_reporter(self):
        """Create trade reporter instance"""
        return TradeReporter(
            reporting_delay_ms=100,
            batch_size=50,
            enable_real_time=True
        )
    
    @pytest.fixture
    def sample_trades(self) -> List[Dict]:
        """Generate sample trades for testing"""
        trades = []
        base_time = datetime.now(timezone.utc)
        
        instruments = [
            {'id': 'BTC/USDT', 'type': 'CRYPTO', 'isin': None},
            {'id': 'ETH/USDT', 'type': 'CRYPTO', 'isin': None},
            {'id': 'AAPL', 'type': 'EQUITY', 'isin': 'US0378331005'},
            {'id': 'EUR/USD', 'type': 'FX', 'isin': None}
        ]
        
        for i in range(100):
            instrument = instruments[i % len(instruments)]
            
            trade = {
                'transaction_id': f'TRD{uuid.uuid4().hex[:12].upper()}',
                'execution_timestamp': base_time + timedelta(seconds=i),
                'instrument_id': instrument['id'],
                'instrument_type': instrument['type'],
                'isin': instrument['isin'],
                'price': Decimal(str(50000 + np.random.randn() * 1000)),
                'quantity': Decimal(str(abs(np.random.randn() * 0.1))),
                'currency': 'USD',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'venue': 'BINANCE' if 'CRYPTO' in instrument['type'] else 'NASDAQ',
                'algorithm_id': 'GRIDATTENTION_V1',
                'algorithm_version': '1.2.3'
            }
            
            trades.append(trade)
        
        return trades
    
    @pytest.mark.asyncio
    async def test_trade_report_generation(self, trade_reporter, sample_trades):
        """Test generation of compliant trade reports"""
        reports = []
        
        for trade in sample_trades[:10]:
            report = await trade_reporter.generate_report(trade)
            reports.append(report)
            
            # Verify required fields
            assert report.transaction_id == trade['transaction_id']
            assert report.execution_timestamp == trade['execution_timestamp']
            assert report.instrument_id == trade['instrument_id']
            assert report.price == trade['price']
            assert report.quantity == trade['quantity']
            
            # Verify regulatory fields
            assert report.algorithm_id == trade['algorithm_id']
            assert report.algorithm_version == trade['algorithm_version']
            assert report.trading_capacity in ['DEAL', 'MATCH', 'AOTC']
            assert report.investment_decision_maker == 'ALGORITHM'
            assert report.execution_decision_maker == 'ALGORITHM'
            
            # Verify timestamps
            assert isinstance(report.execution_timestamp, datetime)
            assert report.trading_day == trade['execution_timestamp'].strftime('%Y%m%d')
    
    @pytest.mark.asyncio
    async def test_real_time_reporting(self, trade_reporter, sample_trades):
        """Test real-time trade reporting within regulatory timeframes"""
        reporting_times = []
        
        async def report_trade(trade):
            start_time = datetime.now()
            report = await trade_reporter.report_trade(trade)
            end_time = datetime.now()
            
            reporting_time_ms = (end_time - start_time).total_seconds() * 1000
            reporting_times.append(reporting_time_ms)
            
            return report
        
        # Report multiple trades concurrently
        tasks = [report_trade(trade) for trade in sample_trades[:20]]
        reports = await asyncio.gather(*tasks)
        
        # Verify reporting timeframes
        avg_reporting_time = np.mean(reporting_times)
        max_reporting_time = np.max(reporting_times)
        
        # MiFID II requires reporting "as quickly as possible" and no later than T+1
        # For algorithmic trading, this typically means < 1 second
        assert max_reporting_time < 1000, f"Max reporting time {max_reporting_time}ms exceeds limit"
        assert avg_reporting_time < 500, f"Average reporting time {avg_reporting_time}ms too high"
        
        print(f"Reporting times - Avg: {avg_reporting_time:.2f}ms, Max: {max_reporting_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_batch_reporting(self, trade_reporter):
        """Test batch reporting for high-volume scenarios"""
        batch_size = 50
        num_batches = 5
        
        all_reports = []
        
        for batch_num in range(num_batches):
            # Generate batch of trades
            trades = []
            base_time = datetime.now(timezone.utc)
            
            for i in range(batch_size):
                trade = {
                    'transaction_id': f'BATCH{batch_num}_{i:04d}',
                    'execution_timestamp': base_time + timedelta(milliseconds=i * 10),
                    'instrument_id': 'BTC/USDT',
                    'price': Decimal('50000') + Decimal(str(i)),
                    'quantity': Decimal('0.1'),
                    'side': 'BUY' if i % 2 == 0 else 'SELL'
                }
                trades.append(trade)
            
            # Submit batch
            batch_reports = await trade_reporter.report_batch(trades)
            all_reports.extend(batch_reports)
            
            # Verify batch integrity
            assert len(batch_reports) == batch_size
            
            # Verify batch sequence
            for i, report in enumerate(batch_reports):
                assert report.transaction_id == f'BATCH{batch_num}_{i:04d}'
        
        # Verify all trades reported
        assert len(all_reports) == batch_size * num_batches
        
        # Check for duplicates
        transaction_ids = [r.transaction_id for r in all_reports]
        assert len(transaction_ids) == len(set(transaction_ids)), "Duplicate transaction IDs found"
    
    @pytest.mark.asyncio
    async def test_failed_trade_reporting(self, trade_reporter):
        """Test handling of failed trade reports"""
        failed_reports = []
        retry_attempts = defaultdict(int)
        
        # Mock reporting service with intermittent failures
        async def mock_submit_report(report):
            if np.random.random() < 0.3:  # 30% failure rate
                retry_attempts[report.transaction_id] += 1
                raise Exception("Reporting service unavailable")
            return {'status': 'success', 'report_id': uuid.uuid4().hex}
        
        trade_reporter.submit_report = mock_submit_report
        
        # Attempt to report trades
        trades = [
            {
                'transaction_id': f'FAIL_TEST_{i}',
                'execution_timestamp': datetime.now(timezone.utc),
                'instrument_id': 'BTC/USDT',
                'price': Decimal('50000'),
                'quantity': Decimal('0.1'),
                'side': 'BUY'
            }
            for i in range(50)
        ]
        
        results = await trade_reporter.report_trades_with_retry(
            trades,
            max_retries=3,
            retry_delay_ms=100
        )
        
        # Analyze results
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"Successful reports: {len(successful)}, Failed: {len(failed)}")
        print(f"Retry attempts: {dict(retry_attempts)}")
        
        # Verify retry logic
        for trade_id, attempts in retry_attempts.items():
            assert attempts <= 3, f"Trade {trade_id} exceeded max retries"
        
        # Ensure failed trades are logged for manual intervention
        assert len(failed) > 0  # Some should fail given 30% failure rate
        for failed_report in failed:
            assert 'error' in failed_report
            assert 'trade_id' in failed_report
    
    @pytest.mark.asyncio
    async def test_regulatory_field_validation(self, trade_reporter):
        """Test validation of regulatory required fields"""
        invalid_trades = [
            # Missing transaction ID
            {
                'execution_timestamp': datetime.now(timezone.utc),
                'instrument_id': 'BTC/USDT',
                'price': Decimal('50000'),
                'quantity': Decimal('0.1')
            },
            # Invalid price
            {
                'transaction_id': 'INVALID_PRICE',
                'execution_timestamp': datetime.now(timezone.utc),
                'instrument_id': 'BTC/USDT',
                'price': Decimal('-100'),
                'quantity': Decimal('0.1')
            },
            # Missing timestamp
            {
                'transaction_id': 'NO_TIMESTAMP',
                'instrument_id': 'BTC/USDT',
                'price': Decimal('50000'),
                'quantity': Decimal('0.1')
            },
            # Invalid quantity
            {
                'transaction_id': 'ZERO_QTY',
                'execution_timestamp': datetime.now(timezone.utc),
                'instrument_id': 'BTC/USDT',
                'price': Decimal('50000'),
                'quantity': Decimal('0')
            }
        ]
        
        validation_errors = []
        
        for trade in invalid_trades:
            try:
                report = await trade_reporter.validate_and_generate_report(trade)
            except ValueError as e:
                validation_errors.append({
                    'trade': trade.get('transaction_id', 'UNKNOWN'),
                    'error': str(e)
                })
        
        # All invalid trades should raise validation errors
        assert len(validation_errors) == len(invalid_trades)
        
        # Verify specific validation messages
        error_messages = [e['error'] for e in validation_errors]
        assert any('transaction_id' in msg for msg in error_messages)
        assert any('price' in msg for msg in error_messages)
        assert any('timestamp' in msg for msg in error_messages)
        assert any('quantity' in msg for msg in error_messages)
    
    @pytest.mark.asyncio
    async def test_best_execution_reporting(self, trade_reporter):
        """Test best execution compliance reporting"""
        # Generate trades with various execution qualities
        trades_with_benchmarks = []
        
        for i in range(50):
            benchmark_price = Decimal('50000')
            slippage = Decimal(str(np.random.randn() * 10))
            execution_price = benchmark_price + slippage
            
            trade = {
                'transaction_id': f'BEST_EX_{i}',
                'execution_timestamp': datetime.now(timezone.utc),
                'instrument_id': 'BTC/USDT',
                'price': execution_price,
                'quantity': Decimal('0.1'),
                'side': 'BUY',
                'benchmark_price': benchmark_price,
                'venue': 'BINANCE',
                'alternative_venues': ['KRAKEN', 'COINBASE']
            }
            
            trades_with_benchmarks.append(trade)
        
        # Generate best execution reports
        best_ex_reports = []
        
        for trade in trades_with_benchmarks:
            report = await trade_reporter.generate_best_execution_report(trade)
            best_ex_reports.append(report)
            
            # Verify best execution fields
            assert 'execution_quality' in report
            assert 'slippage_bps' in report
            assert 'venue_comparison' in report
            assert 'best_execution_factors' in report
            
            # Check execution quality metrics
            slippage_bps = report['slippage_bps']
            if abs(slippage_bps) > 10:  # More than 10 basis points
                assert report['execution_quality'] == 'REVIEW_REQUIRED'
            else:
                assert report['execution_quality'] in ['GOOD', 'ACCEPTABLE']
        
        # Generate summary statistics
        summary = await trade_reporter.generate_best_execution_summary(best_ex_reports)
        
        assert 'average_slippage_bps' in summary
        assert 'execution_quality_distribution' in summary
        assert 'venue_performance' in summary
        assert 'recommendations' in summary
    
    @pytest.mark.asyncio
    async def test_transaction_reporting_fields(self, trade_reporter):
        """Test complete transaction reporting field set for MiFID II"""
        trade = {
            'transaction_id': 'MIFID_TEST_001',
            'execution_timestamp': datetime.now(timezone.utc),
            'instrument_id': 'AAPL',
            'instrument_type': 'EQUITY',
            'isin': 'US0378331005',
            'price': Decimal('150.25'),
            'quantity': Decimal('100'),
            'currency': 'USD',
            'side': 'BUY',
            'venue': 'XNAS',
            'order_id': 'ORD123456',
            'client_id': 'CLIENT001',
            'desk_id': 'ALGO_DESK_01'
        }
        
        # Generate full MiFID II report
        mifid_report = await trade_reporter.generate_mifid_report(trade)
        
        # Verify all required MiFID II fields
        required_fields = [
            'trading_date_time',
            'trading_capacity',
            'quantity',
            'price',
            'price_currency',
            'net_amount',
            'venue',
            'instrument_identification_code',
            'instrument_identification_code_type',
            'investment_decision_within_firm',
            'executing_trader',
            'waiver_indicator',
            'short_selling_indicator',
            'otc_post_trade_indicator',
            'commodity_derivative_indicator',
            'securities_financing_transaction_indicator'
        ]
        
        for field in required_fields:
            assert field in mifid_report, f"Missing required field: {field}"
        
        # Verify field formats
        assert len(mifid_report['trading_date_time']) == 28  # ISO 8601 with microseconds
        assert mifid_report['quantity'] >= 0
        assert mifid_report['price'] > 0
        assert mifid_report['instrument_identification_code_type'] in ['ISIN', 'AII', 'MIC']
        assert isinstance(mifid_report['short_selling_indicator'], bool)
    
    @pytest.mark.asyncio
    async def test_aggregated_reporting(self, trade_reporter, sample_trades):
        """Test aggregated reporting for regulatory requirements"""
        # Group trades by instrument and time window
        time_window = timedelta(minutes=1)
        
        aggregated_reports = await trade_reporter.generate_aggregated_reports(
            sample_trades,
            time_window=time_window,
            aggregation_fields=['instrument_id', 'side']
        )
        
        for report in aggregated_reports:
            # Verify aggregation fields
            assert 'instrument_id' in report
            assert 'side' in report
            assert 'total_quantity' in report
            assert 'total_value' in report
            assert 'average_price' in report
            assert 'trade_count' in report
            assert 'time_window_start' in report
            assert 'time_window_end' in report
            
            # Verify calculations
            assert report['total_quantity'] > 0
            assert report['total_value'] > 0
            assert report['average_price'] > 0
            assert report['trade_count'] > 0
            
            # Verify time window
            window_duration = report['time_window_end'] - report['time_window_start']
            assert window_duration <= time_window
    
    @pytest.mark.asyncio
    async def test_audit_trail_generation(self, trade_reporter):
        """Test generation of complete audit trails for trades"""
        # Create a trade with full lifecycle
        trade_lifecycle = {
            'order_creation': {
                'timestamp': datetime.now(timezone.utc) - timedelta(seconds=5),
                'order_id': 'ORD_AUDIT_001',
                'type': 'LIMIT',
                'price': Decimal('50000'),
                'quantity': Decimal('0.5')
            },
            'order_modification': {
                'timestamp': datetime.now(timezone.utc) - timedelta(seconds=3),
                'new_price': Decimal('49950'),
                'reason': 'Price improvement'
            },
            'partial_fill': {
                'timestamp': datetime.now(timezone.utc) - timedelta(seconds=2),
                'filled_quantity': Decimal('0.3'),
                'fill_price': Decimal('49948')
            },
            'final_fill': {
                'timestamp': datetime.now(timezone.utc) - timedelta(seconds=1),
                'filled_quantity': Decimal('0.2'),
                'fill_price': Decimal('49945')
            },
            'trade_report': {
                'timestamp': datetime.now(timezone.utc),
                'transaction_id': 'TRD_AUDIT_001',
                'status': 'COMPLETED'
            }
        }
        
        # Generate audit trail
        audit_trail = await trade_reporter.generate_audit_trail(trade_lifecycle)
        
        # Verify audit trail completeness
        assert len(audit_trail['events']) == 5
        
        # Verify chronological order
        timestamps = [event['timestamp'] for event in audit_trail['events']]
        assert timestamps == sorted(timestamps)
        
        # Verify event details
        for event in audit_trail['events']:
            assert 'event_type' in event
            assert 'timestamp' in event
            assert 'details' in event
            assert 'actor' in event
            
        # Verify traceability
        assert audit_trail['order_id'] == 'ORD_AUDIT_001'
        assert audit_trail['transaction_id'] == 'TRD_AUDIT_001'
        assert 'checksum' in audit_trail  # For integrity verification
    
    @pytest.mark.asyncio
    async def test_cross_venue_reporting(self, trade_reporter):
        """Test reporting for trades across multiple venues"""
        venues = ['BINANCE', 'KRAKEN', 'COINBASE', 'FTX']
        cross_venue_trades = []
        
        # Generate trades across venues
        for i in range(40):
            venue = venues[i % len(venues)]
            trade = {
                'transaction_id': f'CROSS_VENUE_{i}',
                'execution_timestamp': datetime.now(timezone.utc),
                'instrument_id': 'BTC/USDT',
                'price': Decimal('50000') + Decimal(str(np.random.randn() * 100)),
                'quantity': Decimal('0.1'),
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'venue': venue,
                'venue_transaction_id': f'{venue}_TXN_{i}'
            }
            cross_venue_trades.append(trade)
        
        # Generate consolidated report
        consolidated_report = await trade_reporter.generate_cross_venue_report(
            cross_venue_trades,
            reporting_period='daily'
        )
        
        # Verify venue breakdown
        assert 'venue_summary' in consolidated_report
        for venue in venues:
            assert venue in consolidated_report['venue_summary']
            venue_data = consolidated_report['venue_summary'][venue]
            assert 'trade_count' in venue_data
            assert 'total_volume' in venue_data
            assert 'average_price' in venue_data
        
        # Verify consolidated metrics
        assert 'total_trades' in consolidated_report
        assert 'total_volume' in consolidated_report
        assert 'volume_weighted_average_price' in consolidated_report
        
        # Verify regulatory identifiers
        assert 'consolidated_report_id' in consolidated_report
        assert 'reporting_entity' in consolidated_report
        assert 'reporting_period' in consolidated_report
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_replay(self, trade_reporter):
        """Test ability to recover and replay failed reports"""
        # Simulate a batch of trades with some reporting failures
        failed_trades = []
        successful_trades = []
        
        for i in range(30):
            trade = {
                'transaction_id': f'RECOVERY_TEST_{i}',
                'execution_timestamp': datetime.now(timezone.utc) - timedelta(hours=i),
                'instrument_id': 'ETH/USDT',
                'price': Decimal('3000'),
                'quantity': Decimal('1.0'),
                'side': 'BUY'
            }
            
            # Simulate some trades failing to report
            if i % 5 == 0:
                failed_trades.append(trade)
            else:
                successful_trades.append(trade)
        
        # Store failed trades for recovery
        await trade_reporter.store_failed_reports(failed_trades)
        
        # Simulate recovery process
        recovered_reports = await trade_reporter.recover_and_replay_failed_reports(
            lookback_hours=48,
            batch_size=10
        )
        
        # Verify recovery
        assert len(recovered_reports) == len(failed_trades)
        
        # Verify recovered reports maintain original timestamps
        for original, recovered in zip(failed_trades, recovered_reports):
            assert recovered['original_transaction_id'] == original['transaction_id']
            assert recovered['original_timestamp'] == original['execution_timestamp']
            assert 'recovery_timestamp' in recovered
            assert 'recovery_attempt' in recovered
        
        # Verify deduplication
        all_transaction_ids = [t['transaction_id'] for t in successful_trades + failed_trades]
        reported_ids = await trade_reporter.get_reported_transaction_ids(
            start_time=datetime.now(timezone.utc) - timedelta(hours=50)
        )
        
        assert len(reported_ids) == len(set(all_transaction_ids))


class TestReportingCompliance:
    """Test reporting compliance with various regulations"""
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance(self, trade_reporter):
        """Test GDPR compliance in trade reports"""
        sensitive_trade = {
            'transaction_id': 'GDPR_TEST_001',
            'execution_timestamp': datetime.now(timezone.utc),
            'instrument_id': 'EUR/USD',
            'price': Decimal('1.1850'),
            'quantity': Decimal('10000'),
            'client_id': 'CLIENT_PII_001',
            'client_name': 'John Doe',  # PII
            'client_email': 'john@example.com',  # PII
            'client_phone': '+1234567890'  # PII
        }
        
        # Generate anonymized report
        anonymized_report = await trade_reporter.generate_gdpr_compliant_report(
            sensitive_trade,
            anonymize_pii=True
        )
        
        # Verify PII is not in report
        assert 'client_name' not in anonymized_report
        assert 'client_email' not in anonymized_report
        assert 'client_phone' not in anonymized_report
        
        # Verify pseudonymized ID is present
        assert 'client_id_hash' in anonymized_report
        assert anonymized_report['client_id_hash'] != sensitive_trade['client_id']
        
        # Verify ability to retrieve original with proper authorization
        with pytest.raises(PermissionError):
            await trade_reporter.retrieve_pii_data(
                anonymized_report['client_id_hash'],
                authorization_level='basic'
            )
        
        # Should work with proper authorization
        pii_data = await trade_reporter.retrieve_pii_data(
            anonymized_report['client_id_hash'],
            authorization_level='compliance_officer'
        )
        
        assert pii_data['client_name'] == 'John Doe'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])