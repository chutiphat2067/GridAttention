"""
Logging system tests for GridAttention trading system.

Tests comprehensive logging capabilities including structured logging,
log aggregation, correlation, search functionality, and compliance requirements.
"""

import pytest
import asyncio
import logging
import json
import re
import gzip
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from collections import defaultdict
import structlog
import uuid

# Import core components
from core.logging_system import LoggingSystem
from core.log_aggregator import LogAggregator
from core.log_processor import LogProcessor
from core.correlation_engine import CorrelationEngine


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogCategory(Enum):
    """Log categories for different system components"""
    TRADING = "trading"
    RISK = "risk"
    MARKET_DATA = "market_data"
    EXECUTION = "execution"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AUDIT = "audit"
    COMPLIANCE = "compliance"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None
    exception: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class LogConfig:
    """Logging configuration"""
    min_level: LogLevel
    output_format: str  # json, text, structured
    outputs: List[Dict[str, Any]]  # file, console, remote
    retention_days: int
    rotation_size_mb: int
    compression: bool
    enable_sampling: bool
    sampling_rate: float
    enable_correlation: bool
    enable_metrics: bool
    sensitive_fields: List[str]
    compliance_mode: bool


class TestLoggingSystem:
    """Test logging system functionality"""
    
    @pytest.fixture
    async def logging_system(self):
        """Create logging system instance"""
        return LoggingSystem(
            enable_structured_logging=True,
            enable_async_logging=True,
            buffer_size=10000,
            flush_interval_seconds=1
        )
    
    @pytest.fixture
    async def log_aggregator(self):
        """Create log aggregator instance"""
        return LogAggregator(
            aggregation_window_seconds=60,
            enable_pattern_detection=True,
            enable_anomaly_detection=True
        )
    
    @pytest.fixture
    def log_config(self):
        """Create logging configuration"""
        return LogConfig(
            min_level=LogLevel.INFO,
            output_format='json',
            outputs=[
                {'type': 'file', 'path': 'logs/trading.log'},
                {'type': 'console', 'format': 'colored'},
                {'type': 'remote', 'endpoint': 'http://log-aggregator:8080'}
            ],
            retention_days=30,
            rotation_size_mb=100,
            compression=True,
            enable_sampling=True,
            sampling_rate=0.1,  # Sample 10% of DEBUG logs
            enable_correlation=True,
            enable_metrics=True,
            sensitive_fields=['password', 'api_key', 'private_key'],
            compliance_mode=True
        )
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for log files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_structured_logging(self, logging_system):
        """Test structured logging with context"""
        # Configure structured logging
        logger = await logging_system.get_logger('trading')
        
        # Log with context
        trade_context = {
            'order_id': 'ORD_123456',
            'instrument': 'BTC/USDT',
            'quantity': Decimal('0.5'),
            'price': Decimal('50000'),
            'side': 'BUY',
            'strategy': 'grid_trading'
        }
        
        await logger.info(
            "Order placed successfully",
            **trade_context
        )
        
        # Retrieve logged entry
        logs = await logging_system.get_recent_logs(count=1)
        assert len(logs) > 0
        
        log_entry = logs[0]
        assert log_entry.message == "Order placed successfully"
        assert log_entry.context['order_id'] == 'ORD_123456'
        assert log_entry.context['instrument'] == 'BTC/USDT'
        assert str(log_entry.context['quantity']) == '0.5'
        
        # Test structured exception logging
        try:
            raise ValueError("Invalid order parameters")
        except Exception as e:
            await logger.error(
                "Order validation failed",
                exc_info=True,
                order_id='ORD_123457',
                error_type='validation'
            )
        
        # Retrieve error log
        error_logs = await logging_system.get_logs_by_level(
            level=LogLevel.ERROR,
            limit=1
        )
        
        assert len(error_logs) > 0
        error_log = error_logs[0]
        assert error_log.exception is not None
        assert 'traceback' in error_log.exception
        assert 'type' in error_log.exception
        assert error_log.exception['type'] == 'ValueError'
    
    @pytest.mark.asyncio
    async def test_log_correlation(self, logging_system):
        """Test log correlation across system components"""
        # Create correlation context
        correlation_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        # Simulate a trading flow with correlation
        async with logging_system.correlation_context(
            correlation_id=correlation_id,
            trace_id=trace_id
        ) as context:
            # Order placement
            order_logger = await logging_system.get_logger('execution')
            await order_logger.info(
                "Placing order",
                order_id='ORD_CORR_001',
                instrument='ETH/USDT'
            )
            
            # Risk check
            risk_logger = await logging_system.get_logger('risk')
            await risk_logger.info(
                "Performing risk check",
                order_id='ORD_CORR_001',
                check_type='position_limit'
            )
            
            # Market data fetch
            market_logger = await logging_system.get_logger('market_data')
            await market_logger.info(
                "Fetching market data",
                instrument='ETH/USDT',
                data_type='order_book'
            )
            
            # Execution
            await order_logger.info(
                "Order executed",
                order_id='ORD_CORR_001',
                fill_price='3000.50'
            )
        
        # Retrieve correlated logs
        correlated_logs = await logging_system.get_correlated_logs(
            correlation_id=correlation_id
        )
        
        assert len(correlated_logs) == 4
        
        # Verify all logs have same correlation ID
        for log in correlated_logs:
            assert log.correlation_id == correlation_id
            assert log.trace_id == trace_id
        
        # Verify log sequence
        log_messages = [log.message for log in correlated_logs]
        expected_sequence = [
            "Placing order",
            "Performing risk check",
            "Fetching market data",
            "Order executed"
        ]
        assert log_messages == expected_sequence
        
        # Test trace analysis
        trace_analysis = await logging_system.analyze_trace(trace_id)
        assert trace_analysis['total_logs'] == 4
        assert trace_analysis['duration'] > timedelta(0)
        assert 'component_breakdown' in trace_analysis
        assert len(trace_analysis['component_breakdown']) == 3  # execution, risk, market_data
    
    @pytest.mark.asyncio
    async def test_log_filtering_and_search(self, logging_system):
        """Test log filtering and search capabilities"""
        # Generate diverse log entries
        base_time = datetime.now(timezone.utc)
        
        test_logs = [
            {
                'level': LogLevel.INFO,
                'category': LogCategory.TRADING,
                'message': 'Order placed',
                'context': {'order_id': 'ORD_001', 'instrument': 'BTC/USDT', 'price': 50000}
            },
            {
                'level': LogLevel.WARNING,
                'category': LogCategory.RISK,
                'message': 'Position limit approaching',
                'context': {'position': 8.5, 'limit': 10.0, 'instrument': 'BTC/USDT'}
            },
            {
                'level': LogLevel.ERROR,
                'category': LogCategory.EXECUTION,
                'message': 'Order rejected by exchange',
                'context': {'order_id': 'ORD_002', 'reason': 'insufficient_balance'}
            },
            {
                'level': LogLevel.INFO,
                'category': LogCategory.TRADING,
                'message': 'Order filled',
                'context': {'order_id': 'ORD_001', 'fill_price': 49950}
            },
            {
                'level': LogLevel.DEBUG,
                'category': LogCategory.SYSTEM,
                'message': 'Cache updated',
                'context': {'cache_key': 'market_data_BTC', 'ttl': 60}
            }
        ]
        
        # Log all entries
        for i, log_data in enumerate(test_logs):
            logger = await logging_system.get_logger(log_data['category'].value)
            await logger.log(
                level=log_data['level'],
                message=log_data['message'],
                timestamp=base_time + timedelta(seconds=i),
                **log_data['context']
            )
        
        # Test filtering by level
        info_logs = await logging_system.search_logs(
            filters={'level': LogLevel.INFO}
        )
        assert len(info_logs) == 2
        
        # Test filtering by category
        trading_logs = await logging_system.search_logs(
            filters={'category': LogCategory.TRADING}
        )
        assert len(trading_logs) == 2
        
        # Test filtering by time range
        recent_logs = await logging_system.search_logs(
            filters={
                'start_time': base_time + timedelta(seconds=2),
                'end_time': base_time + timedelta(seconds=4)
            }
        )
        assert len(recent_logs) == 2
        
        # Test text search
        order_logs = await logging_system.search_logs(
            query='order',
            case_sensitive=False
        )
        assert len(order_logs) >= 3
        
        # Test context field search
        btc_logs = await logging_system.search_logs(
            filters={'context.instrument': 'BTC/USDT'}
        )
        assert len(btc_logs) == 2
        
        # Test complex query
        complex_results = await logging_system.search_logs(
            query='order',
            filters={
                'level': [LogLevel.INFO, LogLevel.ERROR],
                'category': [LogCategory.TRADING, LogCategory.EXECUTION],
                'context.order_id': {'$exists': True}
            }
        )
        assert len(complex_results) >= 2
    
    @pytest.mark.asyncio
    async def test_log_rotation_and_compression(self, logging_system, temp_log_dir):
        """Test log file rotation and compression"""
        # Configure file logging with rotation
        file_config = {
            'path': temp_log_dir / 'test.log',
            'max_size_mb': 1,  # Small size for testing
            'backup_count': 3,
            'compression': True
        }
        
        await logging_system.configure_file_output(file_config)
        logger = await logging_system.get_logger('test')
        
        # Generate logs to trigger rotation
        large_message = 'x' * 1024  # 1KB message
        
        for i in range(1500):  # Should trigger rotation
            await logger.info(
                f"Large message {i}: {large_message}",
                index=i
            )
        
        # Force flush
        await logging_system.flush()
        
        # Check for rotated files
        log_files = list(temp_log_dir.glob('test.log*'))
        assert len(log_files) > 1  # Should have rotated files
        
        # Check for compressed files
        compressed_files = list(temp_log_dir.glob('test.log.*.gz'))
        assert len(compressed_files) > 0
        
        # Verify compressed file can be read
        if compressed_files:
            with gzip.open(compressed_files[0], 'rt') as f:
                content = f.read()
                assert 'Large message' in content
        
        # Verify backup count limit
        all_backups = list(temp_log_dir.glob('test.log.*'))
        assert len(all_backups) <= file_config['backup_count']
    
    @pytest.mark.asyncio
    async def test_performance_logging(self, logging_system):
        """Test performance metric logging"""
        perf_logger = await logging_system.get_performance_logger('execution')
        
        # Log operation timing
        async with perf_logger.timer('order_execution') as timer:
            # Simulate order execution
            await asyncio.sleep(0.1)
            timer.add_context(order_id='ORD_PERF_001')
        
        # Log custom metrics
        await perf_logger.record_metric(
            'order_latency_ms',
            value=15.5,
            tags={'exchange': 'binance', 'order_type': 'market'}
        )
        
        await perf_logger.record_metric(
            'fill_rate',
            value=0.95,
            tags={'instrument': 'BTC/USDT', 'strategy': 'grid'}
        )
        
        # Log histogram data
        for i in range(100):
            latency = 10 + (i % 10) * 2  # 10-28ms
            await perf_logger.record_histogram(
                'api_latency_ms',
                value=latency,
                buckets=[10, 20, 50, 100, 200, 500]
            )
        
        # Retrieve performance logs
        perf_logs = await logging_system.get_performance_logs(
            start_time=datetime.now(timezone.utc) - timedelta(minutes=1)
        )
        
        # Verify timer log
        timer_logs = [log for log in perf_logs if 'duration_ms' in log.metrics]
        assert len(timer_logs) > 0
        assert timer_logs[0].metrics['duration_ms'] >= 100
        
        # Verify custom metrics
        metric_logs = [log for log in perf_logs if 'order_latency_ms' in log.metrics]
        assert len(metric_logs) > 0
        assert metric_logs[0].metrics['order_latency_ms'] == 15.5
        
        # Verify histogram data
        histogram_logs = [log for log in perf_logs if 'api_latency_ms' in log.metrics]
        assert len(histogram_logs) == 100
        
        # Generate performance summary
        perf_summary = await logging_system.generate_performance_summary(
            time_window=timedelta(minutes=5)
        )
        
        assert 'metrics' in perf_summary
        assert 'order_latency_ms' in perf_summary['metrics']
        assert 'percentiles' in perf_summary['metrics']['api_latency_ms']
    
    @pytest.mark.asyncio
    async def test_security_logging(self, logging_system):
        """Test security-sensitive logging"""
        security_logger = await logging_system.get_security_logger()
        
        # Test sensitive data masking
        sensitive_data = {
            'user_id': 'USER_123',
            'api_key': 'sk_live_abcdef123456789',
            'password': 'super_secret_password',
            'credit_card': '4111-1111-1111-1111',
            'email': 'user@example.com'
        }
        
        await security_logger.info(
            "User authentication attempt",
            **sensitive_data
        )
        
        # Retrieve log and verify masking
        logs = await logging_system.get_recent_logs(count=1)
        log_entry = logs[0]
        
        # Sensitive fields should be masked
        assert log_entry.context['api_key'] == 'sk_live_******'
        assert log_entry.context['password'] == '***'
        assert log_entry.context['credit_card'] == '4111-****-****-1111'
        
        # Non-sensitive fields should be preserved
        assert log_entry.context['user_id'] == 'USER_123'
        assert log_entry.context['email'] == 'user@example.com'
        
        # Test security event logging
        security_events = [
            {
                'event': 'login_success',
                'user_id': 'USER_123',
                'ip_address': '192.168.1.100',
                'user_agent': 'Mozilla/5.0...'
            },
            {
                'event': 'login_failure',
                'user_id': 'USER_456',
                'ip_address': '10.0.0.50',
                'reason': 'invalid_password',
                'attempt_count': 3
            },
            {
                'event': 'permission_denied',
                'user_id': 'USER_789',
                'resource': '/api/admin/users',
                'required_permission': 'admin'
            }
        ]
        
        for event in security_events:
            await security_logger.security_event(**event)
        
        # Verify security events are properly categorized
        security_logs = await logging_system.get_logs_by_category(
            category=LogCategory.SECURITY
        )
        
        assert len(security_logs) >= len(security_events)
        
        # Check for failed login tracking
        failed_logins = [
            log for log in security_logs 
            if log.context.get('event') == 'login_failure'
        ]
        assert len(failed_logins) > 0
        assert failed_logins[0].context['attempt_count'] == 3
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, logging_system):
        """Test audit trail logging for compliance"""
        audit_logger = await logging_system.get_audit_logger()
        
        # Log trading actions for audit
        trading_actions = [
            {
                'action': 'order_placed',
                'order_id': 'AUD_ORD_001',
                'user_id': 'TRADER_001',
                'instrument': 'BTC/USDT',
                'side': 'BUY',
                'quantity': '0.5',
                'price': '50000',
                'timestamp': datetime.now(timezone.utc)
            },
            {
                'action': 'order_modified',
                'order_id': 'AUD_ORD_001',
                'user_id': 'TRADER_001',
                'old_price': '50000',
                'new_price': '49500',
                'modification_reason': 'price_improvement',
                'timestamp': datetime.now(timezone.utc)
            },
            {
                'action': 'order_cancelled',
                'order_id': 'AUD_ORD_001',
                'user_id': 'TRADER_001',
                'cancellation_reason': 'user_requested',
                'timestamp': datetime.now(timezone.utc)
            }
        ]
        
        for action in trading_actions:
            await audit_logger.log_action(**action)
        
        # Verify audit logs are immutable
        audit_logs = await logging_system.get_audit_logs(
            start_time=datetime.now(timezone.utc) - timedelta(minutes=1)
        )
        
        assert len(audit_logs) == len(trading_actions)
        
        # Verify audit log properties
        for log in audit_logs:
            assert log.category == LogCategory.AUDIT
            assert log.level == LogLevel.INFO
            assert 'action' in log.context
            assert 'user_id' in log.context
            assert 'timestamp' in log.context
            
            # Verify checksum for integrity
            assert 'checksum' in log.context
            calculated_checksum = await logging_system.calculate_log_checksum(log)
            assert log.context['checksum'] == calculated_checksum
        
        # Test audit report generation
        audit_report = await logging_system.generate_audit_report(
            user_id='TRADER_001',
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc)
        )
        
        assert audit_report['total_actions'] == 3
        assert 'actions_by_type' in audit_report
        assert audit_report['actions_by_type']['order_placed'] == 1
        assert audit_report['actions_by_type']['order_modified'] == 1
        assert audit_report['actions_by_type']['order_cancelled'] == 1
    
    @pytest.mark.asyncio
    async def test_log_aggregation_and_patterns(self, logging_system, log_aggregator):
        """Test log aggregation and pattern detection"""
        # Generate logs with patterns
        base_time = datetime.now(timezone.utc)
        
        # Pattern 1: Repeated errors
        error_logger = await logging_system.get_logger('execution')
        for i in range(20):
            await error_logger.error(
                "Connection timeout",
                exchange='binance',
                endpoint='order_submit',
                attempt=i+1,
                timestamp=base_time + timedelta(seconds=i*5)
            )
        
        # Pattern 2: Spike in order rejections
        order_logger = await logging_system.get_logger('trading')
        for i in range(50):
            if 20 <= i <= 30:  # Spike period
                await order_logger.warning(
                    "Order rejected",
                    reason='insufficient_margin',
                    timestamp=base_time + timedelta(seconds=i*2)
                )
        
        # Run aggregation
        aggregation_result = await log_aggregator.aggregate_logs(
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            group_by=['level', 'message'],
            time_bucket='1m'
        )
        
        # Verify aggregation results
        assert 'buckets' in aggregation_result
        assert len(aggregation_result['buckets']) > 0
        
        # Check for error pattern
        error_bucket = next(
            (b for b in aggregation_result['buckets'] 
             if b['level'] == LogLevel.ERROR and 'Connection timeout' in b['message']),
            None
        )
        assert error_bucket is not None
        assert error_bucket['count'] >= 10
        
        # Detect patterns
        patterns = await log_aggregator.detect_patterns(
            logs=await logging_system.get_recent_logs(count=100),
            pattern_types=['repeated_errors', 'anomaly_spike', 'correlation']
        )
        
        assert len(patterns) > 0
        
        # Check for repeated error pattern
        repeated_error_pattern = next(
            (p for p in patterns if p['type'] == 'repeated_errors'),
            None
        )
        assert repeated_error_pattern is not None
        assert repeated_error_pattern['severity'] == 'high'
        assert 'Connection timeout' in repeated_error_pattern['description']
        
        # Check for spike pattern
        spike_pattern = next(
            (p for p in patterns if p['type'] == 'anomaly_spike'),
            None
        )
        assert spike_pattern is not None
    
    @pytest.mark.asyncio
    async def test_log_shipping_and_remote_aggregation(self, logging_system):
        """Test log shipping to remote aggregation services"""
        # Configure remote log shipping
        remote_config = {
            'endpoints': [
                {
                    'url': 'http://elasticsearch:9200',
                    'type': 'elasticsearch',
                    'index_pattern': 'trading-logs-%Y.%m.%d',
                    'batch_size': 100,
                    'flush_interval': 5
                },
                {
                    'url': 'http://fluentd:24224',
                    'type': 'fluentd',
                    'tag': 'trading.gridattention'
                }
            ],
            'retry_policy': {
                'max_retries': 3,
                'backoff_multiplier': 2,
                'max_backoff': 60
            },
            'buffer_size': 10000,
            'compression': True
        }
        
        await logging_system.configure_remote_shipping(remote_config)
        
        # Generate logs for shipping
        logger = await logging_system.get_logger('trading')
        
        for i in range(200):
            await logger.info(
                f"Test log for shipping {i}",
                index=i,
                category='test_shipping'
            )
        
        # Force ship logs
        ship_result = await logging_system.ship_buffered_logs()
        
        assert ship_result['total_logs'] == 200
        assert ship_result['shipped_successfully'] > 0
        assert 'failed_endpoints' in ship_result
        
        # Test retry mechanism for failed shipments
        if ship_result['failed_logs'] > 0:
            retry_result = await logging_system.retry_failed_shipments()
            assert retry_result['retried'] > 0
    
    @pytest.mark.asyncio
    async def test_log_compliance_requirements(self, logging_system, log_config):
        """Test compliance-specific logging requirements"""
        # Enable compliance mode
        log_config.compliance_mode = True
        await logging_system.configure(log_config)
        
        compliance_logger = await logging_system.get_compliance_logger()
        
        # Log regulatory events
        regulatory_events = [
            {
                'event_type': 'trade_execution',
                'trade_id': 'TRD_COMP_001',
                'instrument': 'BTC/USDT',
                'quantity': '1.5',
                'price': '50000',
                'venue': 'BINANCE',
                'execution_time': datetime.now(timezone.utc),
                'algorithm_id': 'GRID_ALGO_V1',
                'client_id': 'CLIENT_001'
            },
            {
                'event_type': 'risk_limit_breach',
                'limit_type': 'position_limit',
                'current_value': '15.5',
                'limit_value': '10.0',
                'action_taken': 'orders_blocked',
                'timestamp': datetime.now(timezone.utc)
            },
            {
                'event_type': 'suspicious_activity',
                'activity_type': 'unusual_trading_pattern',
                'trader_id': 'TRADER_999',
                'description': 'Rapid order placement and cancellation',
                'severity': 'high',
                'timestamp': datetime.now(timezone.utc)
            }
        ]
        
        for event in regulatory_events:
            await compliance_logger.log_regulatory_event(**event)
        
        # Verify compliance logs meet requirements
        compliance_logs = await logging_system.get_compliance_logs(
            start_time=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        for log in compliance_logs:
            # Verify immutability
            assert 'checksum' in log.context
            assert 'timestamp' in log.context
            
            # Verify required fields based on event type
            if log.context.get('event_type') == 'trade_execution':
                required_fields = [
                    'trade_id', 'instrument', 'quantity', 'price',
                    'venue', 'execution_time', 'algorithm_id'
                ]
                for field in required_fields:
                    assert field in log.context
            
            # Verify retention marking
            assert log.context.get('retention_years') >= 5  # Regulatory requirement
            assert log.context.get('deletion_prohibited') == True
        
        # Test compliance report generation
        compliance_report = await logging_system.generate_compliance_log_report(
            report_type='daily',
            date=datetime.now(timezone.utc).date()
        )
        
        assert 'total_events' in compliance_report
        assert 'events_by_type' in compliance_report
        assert 'high_severity_events' in compliance_report
        assert compliance_report['report_complete'] == True
        assert 'signature' in compliance_report  # Digital signature for integrity
    
    @pytest.mark.asyncio
    async def test_log_context_propagation(self, logging_system):
        """Test context propagation across async operations"""
        # Set up context propagation
        await logging_system.enable_context_propagation()
        
        # Define a complex async workflow
        async def process_order(order_id: str):
            async with logging_system.context(order_id=order_id) as ctx:
                logger = await logging_system.get_logger('workflow')
                
                # Step 1: Validate order
                await logger.info("Validating order")
                await asyncio.sleep(0.01)
                
                # Step 2: Check risk (nested context)
                async with logging_system.context(risk_check='position_limit'):
                    risk_logger = await logging_system.get_logger('risk')
                    await risk_logger.info("Checking position limits")
                    
                    # Nested async call
                    await check_margin_requirements(order_id)
                
                # Step 3: Execute order
                await logger.info("Executing order")
                
                return True
        
        async def check_margin_requirements(order_id: str):
            # Should inherit context
            margin_logger = await logging_system.get_logger('margin')
            await margin_logger.info("Checking margin requirements")
        
        # Execute workflow
        await process_order('CTX_ORD_001')
        
        # Retrieve logs and verify context propagation
        workflow_logs = await logging_system.search_logs(
            filters={'context.order_id': 'CTX_ORD_001'}
        )
        
        assert len(workflow_logs) >= 4
        
        # All logs should have the order_id context
        for log in workflow_logs:
            assert log.context.get('order_id') == 'CTX_ORD_001'
        
        # Risk check logs should have additional context
        risk_logs = [log for log in workflow_logs if log.source == 'risk']
        assert all(log.context.get('risk_check') == 'position_limit' for log in risk_logs)
    
    @pytest.mark.asyncio
    async def test_log_sampling_and_throttling(self, logging_system):
        """Test log sampling and throttling for high-frequency logs"""
        # Configure sampling
        sampling_config = {
            'rules': [
                {
                    'level': LogLevel.DEBUG,
                    'sample_rate': 0.1  # 10% of debug logs
                },
                {
                    'level': LogLevel.INFO,
                    'category': LogCategory.MARKET_DATA,
                    'sample_rate': 0.01  # 1% of market data logs
                }
            ],
            'throttle_rules': [
                {
                    'pattern': 'Price update',
                    'max_per_second': 10
                },
                {
                    'pattern': 'Cache hit',
                    'max_per_minute': 100
                }
            ]
        }
        
        await logging_system.configure_sampling(sampling_config)
        
        # Generate high-frequency logs
        debug_logger = await logging_system.get_logger('system')
        market_logger = await logging_system.get_logger('market_data')
        
        # Debug logs (should be sampled)
        debug_logged = 0
        for i in range(1000):
            result = await debug_logger.debug(f"Debug message {i}")
            if result.get('logged'):
                debug_logged += 1
        
        # Should be approximately 10% (±5%)
        assert 50 <= debug_logged <= 150
        
        # Market data logs (heavy sampling)
        market_logged = 0
        for i in range(10000):
            result = await market_logger.info(
                f"Price update {i}",
                instrument='BTC/USDT',
                price=50000 + i
            )
            if result.get('logged'):
                market_logged += 1
        
        # Should be approximately 1% (±0.5%)
        assert 50 <= market_logged <= 150
        
        # Test throttling
        throttle_logger = await logging_system.get_logger('cache')
        throttled_count = 0
        
        # Rapid logging that should be throttled
        start_time = asyncio.get_event_loop().time()
        for i in range(200):
            result = await throttle_logger.info("Cache hit for key XYZ")
            if result.get('throttled'):
                throttled_count += 1
            
            # Small delay to spread over time
            if i % 10 == 0:
                await asyncio.sleep(0.1)
        
        # Should have throttled some messages
        assert throttled_count > 0
        
        # Verify throttle statistics
        throttle_stats = await logging_system.get_throttle_statistics()
        assert 'Cache hit' in throttle_stats
        assert throttle_stats['Cache hit']['throttled'] > 0


class TestLogAnalytics:
    """Test log analytics and visualization"""
    
    @pytest.mark.asyncio
    async def test_log_analytics_dashboard(self, logging_system):
        """Test log analytics dashboard data generation"""
        # Generate diverse logs for analytics
        await self._generate_analytics_logs(logging_system)
        
        # Generate analytics dashboard data
        dashboard_data = await logging_system.generate_analytics_dashboard(
            time_range=timedelta(hours=1),
            refresh_interval=timedelta(seconds=30)
        )
        
        # Verify dashboard components
        assert 'summary' in dashboard_data
        assert 'time_series' in dashboard_data
        assert 'top_errors' in dashboard_data
        assert 'performance_metrics' in dashboard_data
        assert 'log_volume_heatmap' in dashboard_data
        
        # Check summary statistics
        summary = dashboard_data['summary']
        assert summary['total_logs'] > 0
        assert 'logs_per_level' in summary
        assert 'logs_per_category' in summary
        assert 'error_rate' in summary
        assert 'avg_log_size_bytes' in summary
        
        # Check time series data
        time_series = dashboard_data['time_series']
        assert len(time_series['timestamps']) > 0
        assert len(time_series['values']) == len(time_series['timestamps'])
        
        # Check top errors
        top_errors = dashboard_data['top_errors']
        assert isinstance(top_errors, list)
        if top_errors:
            assert 'message' in top_errors[0]
            assert 'count' in top_errors[0]
            assert 'first_seen' in top_errors[0]
            assert 'last_seen' in top_errors[0]
    
    async def _generate_analytics_logs(self, logging_system):
        """Helper to generate logs for analytics testing"""
        categories = list(LogCategory)
        levels = list(LogLevel)
        
        for i in range(500):
            logger = await logging_system.get_logger(
                np.random.choice(categories).value
            )
            level = np.random.choice(levels)
            
            if level == LogLevel.ERROR:
                await logger.error(
                    f"Error message type {i % 5}",
                    error_code=f"ERR_{i % 10:03d}"
                )
            elif level == LogLevel.WARNING:
                await logger.warning(
                    f"Warning about {i % 3}",
                    threshold_exceeded=i % 2 == 0
                )
            else:
                await logger.log(
                    level=level,
                    message=f"Normal operation {i}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])