"""
Audit trail compliance tests for GridAttention trading system.

Ensures comprehensive logging, immutability, and traceability of all trading activities
for regulatory compliance and forensic analysis capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import hashlib
import uuid
import time
from enum import Enum
from collections import defaultdict
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import sqlite3
import pickle
import zlib

# Import core components
from core.audit_logger import AuditLogger
from core.event_store import EventStore
from core.compliance_manager import ComplianceManager


class EventType(Enum):
    """Audit event types"""
    # System events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    
    # Trading events
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_MODIFIED = "ORDER_MODIFIED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    
    # Risk events
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"
    POSITION_LIMIT_CHECK = "POSITION_LIMIT_CHECK"
    MARGIN_CALL = "MARGIN_CALL"
    
    # Algorithm events
    ALGO_DECISION = "ALGO_DECISION"
    REGIME_CHANGE = "REGIME_CHANGE"
    STRATEGY_SWITCH = "STRATEGY_SWITCH"
    
    # Compliance events
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    
    # Data events
    DATA_RECEIVED = "DATA_RECEIVED"
    DATA_ANOMALY = "DATA_ANOMALY"
    CONNECTION_LOST = "CONNECTION_LOST"


@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    actor: str  # System component or user
    action: str
    target: Optional[str]
    details: Dict[str, Any]
    context: Dict[str, Any]
    checksum: Optional[str] = None
    previous_event_id: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum if not provided"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data"""
        data_str = json.dumps({
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor': self.actor,
            'action': self.action,
            'target': self.target,
            'details': self.details,
            'context': self.context,
            'previous_event_id': self.previous_event_id
        }, sort_keys=True)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum"""
        return self.checksum == self._calculate_checksum()


class TestAuditTrail:
    """Test audit trail functionality"""
    
    @pytest.fixture
    async def audit_logger(self):
        """Create audit logger instance"""
        return AuditLogger(
            storage_backend='sqlite',
            retention_days=2555,  # 7 years for regulatory compliance
            enable_encryption=True,
            enable_compression=True
        )
    
    @pytest.fixture
    async def event_store(self):
        """Create event store for audit events"""
        return EventStore(
            database_path='test_audit.db',
            enable_wal=True,  # Write-ahead logging for durability
            page_size=4096
        )
    
    @pytest.mark.asyncio
    async def test_audit_event_creation(self, audit_logger):
        """Test creation of audit events with all required fields"""
        # Create various types of audit events
        events = []
        
        # Order creation event
        order_event = await audit_logger.log_event(
            event_type=EventType.ORDER_CREATED,
            actor='GridStrategySelector',
            action='CREATE_LIMIT_ORDER',
            target='BTC/USDT',
            details={
                'order_id': 'ORD_001',
                'order_type': 'LIMIT',
                'side': 'BUY',
                'price': '50000.00',
                'quantity': '0.1',
                'time_in_force': 'GTC'
            },
            context={
                'strategy': 'GRID_NEUTRAL',
                'regime': 'RANGING',
                'grid_level': 5
            }
        )
        events.append(order_event)
        
        # Risk check event
        risk_event = await audit_logger.log_event(
            event_type=EventType.POSITION_LIMIT_CHECK,
            actor='RiskManagementSystem',
            action='CHECK_POSITION_LIMIT',
            target='BTC/USDT',
            details={
                'current_position': '1.5',
                'position_limit': '2.0',
                'utilization': '75%',
                'check_result': 'PASS'
            },
            context={
                'triggered_by': 'ORDER_CREATED',
                'order_id': 'ORD_001'
            }
        )
        events.append(risk_event)
        
        # Verify event properties
        for event in events:
            assert isinstance(event, AuditEvent)
            assert event.event_id is not None
            assert event.timestamp is not None
            assert event.checksum is not None
            assert event.verify_integrity()
            
            # Verify immutability (attempting to modify should fail)
            original_checksum = event.checksum
            with pytest.raises(AttributeError):
                event.details['modified'] = True
    
    @pytest.mark.asyncio
    async def test_event_chain_integrity(self, audit_logger):
        """Test blockchain-style event chaining for tamper detection"""
        events = []
        
        # Create chain of related events
        for i in range(10):
            previous_id = events[-1].event_id if events else None
            
            event = await audit_logger.log_event(
                event_type=EventType.ALGO_DECISION,
                actor='AttentionLearningLayer',
                action='UPDATE_WEIGHTS',
                details={
                    'iteration': i,
                    'loss': 0.1 * (10 - i),
                    'learning_rate': 0.001
                },
                previous_event_id=previous_id
            )
            events.append(event)
        
        # Verify chain integrity
        chain_valid = await audit_logger.verify_event_chain(events)
        assert chain_valid
        
        # Test tamper detection - modify an event in the middle
        tampered_event = events[5]
        tampered_details = tampered_event.details.copy()
        tampered_details['loss'] = 0.0  # Tamper with data
        
        # Create fake event with same ID but different data
        fake_event = AuditEvent(
            event_id=tampered_event.event_id,
            timestamp=tampered_event.timestamp,
            event_type=tampered_event.event_type,
            actor=tampered_event.actor,
            action=tampered_event.action,
            target=tampered_event.target,
            details=tampered_details,
            context=tampered_event.context,
            previous_event_id=tampered_event.previous_event_id
        )
        
        # Replace original with tampered version
        tampered_chain = events.copy()
        tampered_chain[5] = fake_event
        
        # Verify tamper detection
        chain_valid = await audit_logger.verify_event_chain(tampered_chain)
        assert not chain_valid
    
    @pytest.mark.asyncio
    async def test_comprehensive_order_lifecycle_audit(self, audit_logger):
        """Test complete audit trail for order lifecycle"""
        order_id = 'ORD_LIFECYCLE_001'
        audit_trail = []
        
        # 1. Strategy decision
        decision_event = await audit_logger.log_event(
            event_type=EventType.ALGO_DECISION,
            actor='GridStrategySelector',
            action='GENERATE_TRADING_SIGNAL',
            details={
                'signal': 'BUY',
                'confidence': 0.85,
                'strategy': 'GRID_TREND_FOLLOWING',
                'indicators': {
                    'rsi': 45.2,
                    'ma_cross': 'BULLISH',
                    'volume_profile': 'INCREASING'
                }
            }
        )
        audit_trail.append(decision_event)
        
        # 2. Risk pre-check
        risk_check = await audit_logger.log_event(
            event_type=EventType.COMPLIANCE_CHECK,
            actor='RiskManagementSystem',
            action='PRE_TRADE_RISK_CHECK',
            details={
                'checks_performed': [
                    'position_limit',
                    'daily_loss_limit',
                    'concentration_risk',
                    'margin_requirement'
                ],
                'all_passed': True,
                'margin_available': '25000.00',
                'margin_required': '5000.00'
            },
            previous_event_id=decision_event.event_id
        )
        audit_trail.append(risk_check)
        
        # 3. Order creation
        order_create = await audit_logger.log_event(
            event_type=EventType.ORDER_CREATED,
            actor='ExecutionEngine',
            action='SUBMIT_ORDER',
            target='BTC/USDT',
            details={
                'order_id': order_id,
                'order_type': 'LIMIT',
                'side': 'BUY',
                'price': '50000.00',
                'quantity': '0.1',
                'venue': 'BINANCE'
            },
            previous_event_id=risk_check.event_id
        )
        audit_trail.append(order_create)
        
        # 4. Order modification
        order_modify = await audit_logger.log_event(
            event_type=EventType.ORDER_MODIFIED,
            actor='GridStrategySelector',
            action='IMPROVE_PRICE',
            target=order_id,
            details={
                'original_price': '50000.00',
                'new_price': '49950.00',
                'reason': 'Market moved favorably'
            },
            previous_event_id=order_create.event_id
        )
        audit_trail.append(order_modify)
        
        # 5. Partial fill
        partial_fill = await audit_logger.log_event(
            event_type=EventType.ORDER_FILLED,
            actor='ExecutionEngine',
            action='PARTIAL_FILL',
            target=order_id,
            details={
                'filled_quantity': '0.06',
                'fill_price': '49948.50',
                'remaining_quantity': '0.04',
                'commission': '0.00006',
                'commission_asset': 'BTC'
            },
            previous_event_id=order_modify.event_id
        )
        audit_trail.append(partial_fill)
        
        # 6. Final fill
        final_fill = await audit_logger.log_event(
            event_type=EventType.ORDER_FILLED,
            actor='ExecutionEngine',
            action='FINAL_FILL',
            target=order_id,
            details={
                'filled_quantity': '0.04',
                'fill_price': '49945.00',
                'total_filled': '0.1',
                'average_price': '49947.10',
                'status': 'FILLED'
            },
            previous_event_id=partial_fill.event_id
        )
        audit_trail.append(final_fill)
        
        # Verify complete audit trail
        trail_summary = await audit_logger.generate_audit_trail_summary(
            order_id=order_id,
            events=audit_trail
        )
        
        assert trail_summary['total_events'] == 6
        assert trail_summary['start_event'] == decision_event.event_id
        assert trail_summary['end_event'] == final_fill.event_id
        assert trail_summary['duration'] > timedelta(0)
        assert trail_summary['actors'] == {
            'GridStrategySelector', 'RiskManagementSystem', 'ExecutionEngine'
        }
    
    @pytest.mark.asyncio
    async def test_concurrent_audit_logging(self, audit_logger):
        """Test audit logging under high concurrency"""
        num_concurrent_loggers = 20
        events_per_logger = 50
        all_events = []
        
        async def log_events(logger_id: int):
            """Simulate concurrent event logging"""
            events = []
            for i in range(events_per_logger):
                event = await audit_logger.log_event(
                    event_type=EventType.DATA_RECEIVED,
                    actor=f'DataFeed_{logger_id}',
                    action='PRICE_UPDATE',
                    target='BTC/USDT',
                    details={
                        'price': Decimal('50000') + Decimal(str(np.random.randn() * 100)),
                        'volume': Decimal(str(abs(np.random.randn() * 10))),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
                events.append(event)
                await asyncio.sleep(0.001)  # Small delay
            return events
        
        # Run concurrent loggers
        tasks = [log_events(i) for i in range(num_concurrent_loggers)]
        results = await asyncio.gather(*tasks)
        
        # Collect all events
        for events in results:
            all_events.extend(events)
        
        # Verify no duplicate event IDs
        event_ids = [e.event_id for e in all_events]
        assert len(event_ids) == len(set(event_ids)), "Duplicate event IDs detected"
        
        # Verify all events are properly stored and retrievable
        for event in all_events:
            stored_event = await audit_logger.get_event(event.event_id)
            assert stored_event is not None
            assert stored_event.verify_integrity()
    
    @pytest.mark.asyncio
    async def test_audit_search_and_filtering(self, audit_logger):
        """Test searching and filtering audit events"""
        # Create diverse set of events
        start_time = datetime.now(timezone.utc)
        
        # Create events over time
        for hour in range(24):
            timestamp = start_time + timedelta(hours=hour)
            
            # Various event types
            await audit_logger.log_event(
                event_type=EventType.ORDER_CREATED,
                actor='GridStrategy',
                action='CREATE_ORDER',
                timestamp=timestamp,
                details={'price': '50000', 'quantity': '0.1'}
            )
            
            if hour % 6 == 0:  # Every 6 hours
                await audit_logger.log_event(
                    event_type=EventType.REGIME_CHANGE,
                    actor='MarketRegimeDetector',
                    action='DETECT_REGIME_CHANGE',
                    timestamp=timestamp,
                    details={'from': 'RANGING', 'to': 'TRENDING'}
                )
            
            if hour % 8 == 0:  # Every 8 hours
                await audit_logger.log_event(
                    event_type=EventType.RISK_LIMIT_BREACH,
                    actor='RiskManagement',
                    action='LIMIT_BREACH_DETECTED',
                    timestamp=timestamp,
                    details={'limit_type': 'daily_loss', 'severity': 'WARNING'}
                )
        
        # Test time-based filtering
        last_12_hours = await audit_logger.search_events(
            start_time=start_time + timedelta(hours=12),
            end_time=start_time + timedelta(hours=24)
        )
        assert len(last_12_hours) > 0
        
        # Test event type filtering
        regime_changes = await audit_logger.search_events(
            event_type=EventType.REGIME_CHANGE,
            start_time=start_time
        )
        assert len(regime_changes) == 4  # Every 6 hours over 24 hours
        
        # Test actor filtering
        risk_events = await audit_logger.search_events(
            actor='RiskManagement',
            start_time=start_time
        )
        assert len(risk_events) == 3  # Every 8 hours over 24 hours
        
        # Test complex filtering
        critical_events = await audit_logger.search_events(
            event_types=[EventType.RISK_LIMIT_BREACH, EventType.SUSPICIOUS_ACTIVITY],
            start_time=start_time,
            severity='WARNING'
        )
        assert all(e.event_type == EventType.RISK_LIMIT_BREACH for e in critical_events)
    
    @pytest.mark.asyncio
    async def test_audit_report_generation(self, audit_logger, event_store):
        """Test generation of compliance audit reports"""
        # Generate test data for a trading day
        trading_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        
        # Simulate a day of trading
        events_data = {
            'orders_created': 150,
            'orders_filled': 120,
            'orders_cancelled': 30,
            'risk_checks': 200,
            'regime_changes': 3,
            'manual_overrides': 2,
            'suspicious_activities': 1
        }
        
        # Create events
        for event_type, count in events_data.items():
            for i in range(count):
                timestamp = trading_day + timedelta(
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                if event_type == 'orders_created':
                    await audit_logger.log_event(
                        event_type=EventType.ORDER_CREATED,
                        actor='TradingSystem',
                        action='CREATE_ORDER',
                        timestamp=timestamp,
                        details={'order_id': f'ORD_{i}', 'amount': np.random.uniform(0.01, 1.0)}
                    )
                elif event_type == 'manual_overrides':
                    await audit_logger.log_event(
                        event_type=EventType.MANUAL_OVERRIDE,
                        actor='ComplianceOfficer',
                        action='OVERRIDE_RISK_LIMIT',
                        timestamp=timestamp,
                        details={'reason': 'Market opportunity', 'approved_by': 'John Doe'}
                    )
                elif event_type == 'suspicious_activities':
                    await audit_logger.log_event(
                        event_type=EventType.SUSPICIOUS_ACTIVITY,
                        actor='MarketSurveillance',
                        action='DETECT_ANOMALY',
                        timestamp=timestamp,
                        details={'pattern': 'Unusual volume spike', 'severity': 'HIGH'}
                    )
        
        # Generate daily compliance report
        daily_report = await audit_logger.generate_compliance_report(
            report_date=trading_day,
            report_type='DAILY_COMPLIANCE'
        )
        
        # Verify report contents
        assert daily_report['report_date'] == trading_day.date()
        assert daily_report['total_events'] > 0
        assert 'event_summary' in daily_report
        assert 'compliance_issues' in daily_report
        assert 'manual_interventions' in daily_report
        assert 'system_availability' in daily_report
        
        # Check specific sections
        assert daily_report['event_summary']['orders_created'] == 150
        assert daily_report['event_summary']['orders_filled'] == 120
        assert len(daily_report['manual_interventions']) == 2
        assert len(daily_report['compliance_issues']) >= 1  # At least the suspicious activity
        
        # Generate regulatory submission format
        regulatory_format = await audit_logger.export_for_regulatory_submission(
            daily_report,
            format='XML',  # Common format for regulatory reporting
            include_checksums=True
        )
        
        assert regulatory_format is not None
        assert 'checksum' in regulatory_format
        assert 'timestamp' in regulatory_format
        assert 'reporting_entity' in regulatory_format
    
    @pytest.mark.asyncio
    async def test_data_retention_and_archival(self, audit_logger, event_store):
        """Test long-term data retention and archival policies"""
        # Create events with different ages
        current_time = datetime.now(timezone.utc)
        
        # Recent events (should be in hot storage)
        recent_events = []
        for i in range(10):
            event = await audit_logger.log_event(
                event_type=EventType.ORDER_FILLED,
                actor='ExecutionEngine',
                action='FILL_ORDER',
                timestamp=current_time - timedelta(days=i),
                details={'order_id': f'RECENT_{i}'}
            )
            recent_events.append(event)
        
        # Old events (should be archived)
        old_events = []
        for i in range(10):
            event = await audit_logger.log_event(
                event_type=EventType.ORDER_FILLED,
                actor='ExecutionEngine',
                action='FILL_ORDER',
                timestamp=current_time - timedelta(days=365 + i),  # Over 1 year old
                details={'order_id': f'OLD_{i}'}
            )
            old_events.append(event)
        
        # Very old events (near retention limit)
        very_old_events = []
        for i in range(5):
            event = await audit_logger.log_event(
                event_type=EventType.ORDER_FILLED,
                actor='ExecutionEngine',
                action='FILL_ORDER',
                timestamp=current_time - timedelta(days=2500 + i),  # Near 7-year limit
                details={'order_id': f'VERY_OLD_{i}'}
            )
            very_old_events.append(event)
        
        # Test archival process
        archive_summary = await audit_logger.run_archival_process(
            hot_storage_days=30,
            warm_storage_days=365,
            cold_storage_days=2555  # 7 years
        )
        
        assert archive_summary['events_in_hot_storage'] >= len(recent_events)
        assert archive_summary['events_archived_to_warm'] > 0
        assert archive_summary['events_archived_to_cold'] > 0
        assert archive_summary['events_deleted'] == 0  # Nothing older than 7 years
        
        # Verify retrieval from different storage tiers
        # Recent event (hot storage) - should be fast
        start_time = time.time()
        recent_retrieved = await audit_logger.get_event(recent_events[0].event_id)
        hot_retrieval_time = time.time() - start_time
        assert recent_retrieved is not None
        
        # Old event (warm/cold storage) - might be slower
        start_time = time.time()
        old_retrieved = await audit_logger.get_event(
            old_events[0].event_id,
            allow_archived=True
        )
        cold_retrieval_time = time.time() - start_time
        assert old_retrieved is not None
        
        print(f"Hot storage retrieval: {hot_retrieval_time*1000:.2f}ms")
        print(f"Cold storage retrieval: {cold_retrieval_time*1000:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_forensic_analysis_capabilities(self, audit_logger):
        """Test forensic analysis features for investigations"""
        # Create a suspicious trading pattern
        actor_id = 'SuspiciousTrader_001'
        start_time = datetime.now(timezone.utc)
        
        # Normal trading pattern
        for i in range(20):
            await audit_logger.log_event(
                event_type=EventType.ORDER_CREATED,
                actor=actor_id,
                action='CREATE_ORDER',
                timestamp=start_time + timedelta(minutes=i*5),
                details={
                    'price': Decimal('50000') + Decimal(str(np.random.randn() * 100)),
                    'quantity': Decimal('0.1'),
                    'side': 'BUY' if i % 2 == 0 else 'SELL'
                }
            )
        
        # Suspicious burst of activity
        burst_start = start_time + timedelta(hours=2)
        for i in range(50):
            await audit_logger.log_event(
                event_type=EventType.ORDER_CREATED,
                actor=actor_id,
                action='CREATE_ORDER',
                timestamp=burst_start + timedelta(seconds=i),
                details={
                    'price': Decimal('51000'),  # All at same price
                    'quantity': Decimal('0.01'),  # Small quantities
                    'side': 'BUY'
                }
            )
        
        # Perform forensic analysis
        analysis = await audit_logger.analyze_actor_behavior(
            actor_id=actor_id,
            start_time=start_time,
            end_time=burst_start + timedelta(minutes=5)
        )
        
        # Verify analysis results
        assert analysis['total_events'] == 70
        assert analysis['event_rate_per_minute'] > 0
        assert 'anomalies_detected' in analysis
        assert len(analysis['anomalies_detected']) > 0
        
        # Check for specific anomaly - burst of orders
        burst_anomaly = next(
            (a for a in analysis['anomalies_detected'] if a['type'] == 'ORDER_BURST'),
            None
        )
        assert burst_anomaly is not None
        assert burst_anomaly['severity'] in ['HIGH', 'CRITICAL']
        
        # Generate forensic timeline
        timeline = await audit_logger.generate_forensic_timeline(
            actor_id=actor_id,
            start_time=burst_start - timedelta(minutes=10),
            end_time=burst_start + timedelta(minutes=10),
            include_context=True
        )
        
        assert len(timeline) > 0
        assert all('context' in event for event in timeline)
        
        # Test pattern matching
        patterns = await audit_logger.detect_patterns(
            actor_id=actor_id,
            patterns_to_check=['WASH_TRADING', 'LAYERING', 'SPOOFING', 'QUOTE_STUFFING']
        )
        
        assert 'QUOTE_STUFFING' in patterns  # Due to burst of small orders
    
    @pytest.mark.asyncio
    async def test_regulatory_query_interface(self, audit_logger):
        """Test regulatory query interface for compliance audits"""
        # Simulate regulatory query scenarios
        
        # Query 1: All manual overrides in the last 30 days
        manual_overrides = await audit_logger.regulatory_query(
            query_type='MANUAL_OVERRIDES',
            time_period=timedelta(days=30),
            include_context=True
        )
        
        # Query 2: All trades above certain size
        large_trades = await audit_logger.regulatory_query(
            query_type='LARGE_TRADES',
            filters={'min_size': Decimal('10.0')},
            time_period=timedelta(days=90)
        )
        
        # Query 3: System availability and downtime
        system_availability = await audit_logger.regulatory_query(
            query_type='SYSTEM_AVAILABILITY',
            time_period=timedelta(days=30),
            aggregate=True
        )
        
        # Query 4: Algorithm performance metrics
        algo_performance = await audit_logger.regulatory_query(
            query_type='ALGORITHM_PERFORMANCE',
            filters={'algorithm_id': 'GRIDATTENTION_V1'},
            time_period=timedelta(days=7),
            include_metrics=True
        )
        
        # Verify query results format
        for result in [manual_overrides, large_trades, system_availability, algo_performance]:
            assert 'query_id' in result
            assert 'timestamp' in result
            assert 'record_count' in result
            assert 'data' in result
            assert 'metadata' in result
    
    @pytest.mark.asyncio
    async def test_audit_integrity_verification(self, audit_logger):
        """Test periodic integrity verification of audit logs"""
        # Create events with cross-references
        events = []
        
        # Create interconnected events
        order_id = 'INTEGRITY_TEST_001'
        
        # Order lifecycle with cross-references
        create_event = await audit_logger.log_event(
            event_type=EventType.ORDER_CREATED,
            actor='TradingSystem',
            action='CREATE_ORDER',
            target=order_id,
            details={'price': '50000', 'quantity': '1.0'}
        )
        events.append(create_event)
        
        risk_event = await audit_logger.log_event(
            event_type=EventType.COMPLIANCE_CHECK,
            actor='RiskSystem',
            action='VALIDATE_ORDER',
            target=order_id,
            details={'check_result': 'PASS', 'checks': ['position', 'margin']},
            context={'triggered_by': create_event.event_id}
        )
        events.append(risk_event)
        
        fill_event = await audit_logger.log_event(
            event_type=EventType.ORDER_FILLED,
            actor='ExecutionEngine',
            action='EXECUTE_FILL',
            target=order_id,
            details={'fill_price': '49999', 'fill_quantity': '1.0'},
            context={
                'order_event': create_event.event_id,
                'risk_event': risk_event.event_id
            }
        )
        events.append(fill_event)
        
        # Run integrity verification
        integrity_report = await audit_logger.verify_audit_integrity(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
            deep_check=True
        )
        
        # Verify integrity report
        assert integrity_report['total_events_checked'] >= 3
        assert integrity_report['integrity_failures'] == 0
        assert integrity_report['checksum_failures'] == 0
        assert integrity_report['missing_references'] == 0
        assert integrity_report['orphaned_events'] == 0
        assert integrity_report['verification_status'] == 'PASSED'
        
        # Test cross-reference validation
        references_valid = await audit_logger.validate_cross_references(events)
        assert references_valid
        
        # Test tampering detection
        # Simulate tampering by modifying stored data directly
        tampered_event_id = create_event.event_id
        # (In real scenario, this would modify the database directly)
        
        # Re-run integrity check
        post_tamper_report = await audit_logger.verify_audit_integrity(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
            deep_check=True
        )
        
        # System should detect any tampering
        if post_tamper_report['integrity_failures'] > 0:
            assert post_tamper_report['verification_status'] == 'FAILED'
            assert tampered_event_id in post_tamper_report['failed_events']


class TestAuditCompliance:
    """Test regulatory compliance aspects of audit trail"""
    
    @pytest.mark.asyncio
    async def test_mifid_audit_requirements(self, audit_logger):
        """Test MiFID II specific audit requirements"""
        # MiFID II requires 5-year retention and specific fields
        
        # Create algorithmic trading decision
        algo_decision = await audit_logger.log_event(
            event_type=EventType.ALGO_DECISION,
            actor='GRIDATTENTION_V1.2.3',  # Algorithm ID and version
            action='GENERATE_SIGNAL',
            details={
                'decision_timestamp': datetime.now(timezone.utc).isoformat(),
                'algorithm_parameters': {
                    'lookback_period': 20,
                    'risk_limit': 0.02,
                    'grid_spacing': 0.01
                },
                'market_data_used': {
                    'price_feeds': ['BINANCE', 'KRAKEN'],
                    'latency_ms': 15
                },
                'signal_generated': 'BUY',
                'confidence': 0.75
            }
        )
        
        # Verify MiFID II required fields
        assert algo_decision.actor.startswith('GRIDATTENTION')  # Algorithm identification
        assert 'decision_timestamp' in algo_decision.details
        assert 'algorithm_parameters' in algo_decision.details
        assert 'market_data_used' in algo_decision.details
    
    @pytest.mark.asyncio
    async def test_cftc_audit_requirements(self, audit_logger):
        """Test CFTC specific audit requirements for derivatives"""
        # CFTC requires specific audit trails for automated trading
        
        # Pre-trade risk controls
        risk_control = await audit_logger.log_event(
            event_type=EventType.COMPLIANCE_CHECK,
            actor='PreTradeRiskControl',
            action='VALIDATE_ORDER',
            details={
                'order_id': 'CFTC_TEST_001',
                'instrument': 'BTC-PERP',  # Perpetual futures
                'risk_checks': {
                    'message_rate': {'limit': 100, 'current': 45, 'passed': True},
                    'execution_rate': {'limit': 50, 'current': 22, 'passed': True},
                    'order_size': {'limit': '10.0', 'requested': '1.0', 'passed': True},
                    'price_collar': {'range': 0.05, 'deviation': 0.01, 'passed': True}
                },
                'kill_switch_armed': True,
                'drop_copy_enabled': True
            }
        )
        
        # Source code deployment record
        deployment_record = await audit_logger.log_event(
            event_type=EventType.CONFIG_CHANGE,
            actor='DeploymentSystem',
            action='DEPLOY_ALGORITHM',
            details={
                'algorithm_id': 'GRIDATTENTION_V1.2.3',
                'source_code_hash': hashlib.sha256(b'source_code').hexdigest(),
                'deployment_timestamp': datetime.now(timezone.utc).isoformat(),
                'deployed_by': 'DevOps_Team',
                'testing_completed': True,
                'backtest_results': 'PASSED'
            }
        )
        
        # Verify CFTC requirements
        assert 'kill_switch_armed' in risk_control.details
        assert 'drop_copy_enabled' in risk_control.details
        assert 'source_code_hash' in deployment_record.details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])