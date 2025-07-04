"""
Alert system tests for GridAttention trading system.

Tests comprehensive alerting capabilities including real-time notifications,
escalation procedures, alert routing, and integration with various channels.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from collections import defaultdict, deque
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import core components
from core.alert_system import AlertSystem
from core.alert_manager import AlertManager
from core.notification_service import NotificationService
from core.escalation_manager import EscalationManager


class AlertType(Enum):
    """Types of alerts in the system"""
    # Trading Alerts
    LARGE_LOSS = "large_loss"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    UNUSUAL_TRADING_ACTIVITY = "unusual_trading_activity"
    ORDER_REJECTION = "order_rejection"
    FILL_QUALITY_DEGRADATION = "fill_quality_degradation"
    
    # Risk Alerts
    RISK_LIMIT_BREACH = "risk_limit_breach"
    MARGIN_CALL = "margin_call"
    VAR_BREACH = "var_breach"
    CONCENTRATION_WARNING = "concentration_warning"
    
    # System Alerts
    HIGH_LATENCY = "high_latency"
    SYSTEM_ERROR = "system_error"
    CONNECTIVITY_LOSS = "connectivity_loss"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    
    # Market Alerts
    MARKET_VOLATILITY_SPIKE = "market_volatility_spike"
    LIQUIDITY_WARNING = "liquidity_warning"
    PRICE_DEVIATION = "price_deviation"
    MARKET_HOURS_CHANGE = "market_hours_change"
    
    # Compliance Alerts
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    REGULATORY_BREACH = "regulatory_breach"
    AUDIT_REQUIREMENT = "audit_requirement"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    EMERGENCY = 5


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    TELEGRAM = "telegram"
    DASHBOARD = "dashboard"
    LOG = "log"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    auto_resolve: bool = False
    ttl: Optional[timedelta] = None
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]
    alert_type: AlertType
    severity: AlertSeverity
    enabled: bool = True
    cooldown_period: Optional[timedelta] = None
    aggregation_window: Optional[timedelta] = None
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    escalation_policy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    policy_id: str
    name: str
    levels: List[Dict[str, Any]]  # Each level has delay, recipients, channels
    repeat_interval: Optional[timedelta] = None
    max_escalations: int = 3


class TestAlertSystem:
    """Test alert system functionality"""
    
    @pytest.fixture
    async def alert_system(self):
        """Create alert system instance"""
        return AlertSystem(
            enable_deduplication=True,
            enable_correlation=True,
            enable_auto_resolution=True,
            retention_days=30
        )
    
    @pytest.fixture
    async def notification_service(self):
        """Create notification service"""
        return NotificationService(
            channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
                NotificationChannel.WEBHOOK,
                NotificationChannel.PAGERDUTY
            ],
            rate_limiting_enabled=True,
            batch_notifications=True
        )
    
    @pytest.fixture
    async def escalation_manager(self):
        """Create escalation manager"""
        return EscalationManager(
            enable_auto_escalation=True,
            track_acknowledgments=True
        )
    
    @pytest.fixture
    def sample_alert_rules(self) -> List[AlertRule]:
        """Generate sample alert rules"""
        return [
            AlertRule(
                rule_id='RULE_001',
                name='Large Loss Alert',
                description='Alert when daily loss exceeds threshold',
                condition={
                    'metric': 'daily_pnl',
                    'operator': 'less_than',
                    'threshold': -10000,
                    'duration': timedelta(minutes=1)
                },
                alert_type=AlertType.LARGE_LOSS,
                severity=AlertSeverity.ERROR,
                cooldown_period=timedelta(minutes=30),
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                recipients=['trading@example.com', '#trading-alerts'],
                escalation_policy='financial_critical'
            ),
            AlertRule(
                rule_id='RULE_002',
                name='High System Latency',
                description='Alert on high order execution latency',
                condition={
                    'metric': 'order_latency_p95',
                    'operator': 'greater_than',
                    'threshold': 100,  # 100ms
                    'duration': timedelta(minutes=5)
                },
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.WARNING,
                cooldown_period=timedelta(minutes=15),
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.DASHBOARD],
                recipients=['#tech-alerts']
            ),
            AlertRule(
                rule_id='RULE_003',
                name='Risk Limit Breach',
                description='Alert when risk limits are breached',
                condition={
                    'metric': 'position_exposure',
                    'operator': 'greater_than',
                    'threshold': 1000000,  # $1M
                    'immediate': True
                },
                alert_type=AlertType.RISK_LIMIT_BREACH,
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PAGERDUTY],
                recipients=['risk@example.com', '+1234567890'],
                escalation_policy='risk_critical'
            )
        ]
    
    @pytest.mark.asyncio
    async def test_alert_creation_and_routing(self, alert_system, notification_service):
        """Test alert creation and routing to appropriate channels"""
        # Create test alert
        alert = Alert(
            alert_id='TEST_ALERT_001',
            alert_type=AlertType.LARGE_LOSS,
            severity=AlertSeverity.ERROR,
            title='Large Trading Loss Detected',
            description='Daily P&L has exceeded loss threshold of $10,000',
            timestamp=datetime.now(timezone.utc),
            source='risk_management_system',
            metadata={
                'current_pnl': -12500,
                'threshold': -10000,
                'instruments': ['BTC/USDT', 'ETH/USDT'],
                'strategy': 'grid_trading'
            },
            tags=['trading', 'risk', 'pnl'],
            affected_resources=['grid_strategy_001', 'trading_account_main'],
            recommended_actions=[
                'Review current positions',
                'Consider reducing exposure',
                'Check market conditions'
            ]
        )
        
        # Process alert
        result = await alert_system.process_alert(alert)
        
        assert result['processed'] == True
        assert result['alert_id'] == alert.alert_id
        assert 'routing_decisions' in result
        
        # Verify routing based on severity and type
        routing = result['routing_decisions']
        assert NotificationChannel.EMAIL in routing['channels']
        assert NotificationChannel.SLACK in routing['channels']
        
        # Test notification delivery
        notifications_sent = await notification_service.send_alert(
            alert=alert,
            channels=routing['channels'],
            recipients=['trading@example.com', '#trading-alerts']
        )
        
        assert len(notifications_sent) > 0
        
        for notification in notifications_sent:
            assert notification['status'] == 'sent'
            assert notification['channel'] in routing['channels']
            assert 'delivery_time' in notification
            assert 'message_id' in notification
    
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, alert_system):
        """Test alert deduplication to prevent alert fatigue"""
        # Create similar alerts
        base_alert = Alert(
            alert_id='DUP_TEST_001',
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            title='High Order Latency',
            description='Order latency exceeds threshold',
            timestamp=datetime.now(timezone.utc),
            source='performance_monitor',
            metadata={'latency_ms': 150, 'instrument': 'BTC/USDT'}
        )
        
        # Send first alert
        result1 = await alert_system.process_alert(base_alert)
        assert result1['processed'] == True
        assert result1['deduplicated'] == False
        
        # Send duplicate alerts
        duplicate_alerts = []
        for i in range(5):
            dup_alert = Alert(
                alert_id=f'DUP_TEST_{i+2:03d}',
                alert_type=base_alert.alert_type,
                severity=base_alert.severity,
                title=base_alert.title,
                description=base_alert.description,
                timestamp=base_alert.timestamp + timedelta(seconds=i+1),
                source=base_alert.source,
                metadata={'latency_ms': 150 + i*5, 'instrument': 'BTC/USDT'}
            )
            duplicate_alerts.append(dup_alert)
        
        # Process duplicates
        results = []
        for alert in duplicate_alerts:
            result = await alert_system.process_alert(alert)
            results.append(result)
        
        # Verify deduplication
        deduplicated_count = sum(1 for r in results if r['deduplicated'])
        assert deduplicated_count >= 3  # Most should be deduplicated
        
        # Check deduplication summary
        dedup_summary = await alert_system.get_deduplication_summary(
            alert_type=AlertType.HIGH_LATENCY,
            time_window=timedelta(minutes=5)
        )
        
        assert dedup_summary['total_alerts'] == 6  # 1 original + 5 duplicates
        assert dedup_summary['unique_alerts'] <= 3  # Should be deduplicated
        assert dedup_summary['suppressed_alerts'] >= 3
        assert 'deduplication_rules' in dedup_summary
    
    @pytest.mark.asyncio
    async def test_alert_correlation(self, alert_system):
        """Test alert correlation to identify related issues"""
        # Create related alerts
        correlation_group = []
        base_time = datetime.now(timezone.utc)
        
        # System error alert
        system_error = Alert(
            alert_id='CORR_001',
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.ERROR,
            title='Database Connection Failed',
            description='Unable to connect to trading database',
            timestamp=base_time,
            source='database_monitor',
            metadata={'error_code': 'DB_CONN_FAIL', 'database': 'trading_db'}
        )
        correlation_group.append(system_error)
        
        # Subsequent connectivity issues
        connectivity_alerts = [
            Alert(
                alert_id='CORR_002',
                alert_type=AlertType.CONNECTIVITY_LOSS,
                severity=AlertSeverity.ERROR,
                title='Exchange Connection Lost',
                description='Lost connection to Binance',
                timestamp=base_time + timedelta(seconds=5),
                source='exchange_connector',
                metadata={'exchange': 'Binance', 'error': 'timeout'}
            ),
            Alert(
                alert_id='CORR_003',
                alert_type=AlertType.ORDER_REJECTION,
                severity=AlertSeverity.WARNING,
                title='Orders Rejected',
                description='Multiple orders rejected due to connectivity',
                timestamp=base_time + timedelta(seconds=10),
                source='order_manager',
                metadata={'rejected_count': 15, 'reason': 'no_connection'}
            ),
            Alert(
                alert_id='CORR_004',
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.WARNING,
                title='Extreme Latency Detected',
                description='API latency exceeds 5000ms',
                timestamp=base_time + timedelta(seconds=8),
                source='latency_monitor',
                metadata={'latency_ms': 5500, 'endpoint': 'order_submit'}
            )
        ]
        
        correlation_group.extend(connectivity_alerts)
        
        # Process alerts and check correlation
        for alert in correlation_group:
            await alert_system.process_alert(alert)
        
        # Get correlation analysis
        correlation_result = await alert_system.analyze_correlations(
            time_window=timedelta(minutes=1),
            correlation_threshold=0.7
        )
        
        assert len(correlation_result['correlation_groups']) > 0
        
        # Check the main correlation group
        main_group = correlation_result['correlation_groups'][0]
        assert len(main_group['alerts']) >= 3
        assert main_group['root_cause_probability'] > 0.7
        assert 'suggested_root_cause' in main_group
        assert main_group['suggested_root_cause'] == 'Database Connection Failed'
        
        # Verify correlation scores
        assert 'correlation_matrix' in correlation_result
        assert correlation_result['correlation_strength']['CORR_001']['CORR_002'] > 0.7
    
    @pytest.mark.asyncio
    async def test_alert_escalation(self, alert_system, escalation_manager):
        """Test alert escalation procedures"""
        # Define escalation policy
        escalation_policy = EscalationPolicy(
            policy_id='risk_critical',
            name='Critical Risk Escalation',
            levels=[
                {
                    'level': 1,
                    'delay': timedelta(minutes=0),
                    'recipients': ['trader@example.com'],
                    'channels': [NotificationChannel.EMAIL, NotificationChannel.SLACK]
                },
                {
                    'level': 2,
                    'delay': timedelta(minutes=5),
                    'recipients': ['trader@example.com', 'risk_manager@example.com'],
                    'channels': [NotificationChannel.EMAIL, NotificationChannel.SMS]
                },
                {
                    'level': 3,
                    'delay': timedelta(minutes=15),
                    'recipients': ['trader@example.com', 'risk_manager@example.com', 'cto@example.com'],
                    'channels': [NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.PAGERDUTY]
                }
            ],
            repeat_interval=timedelta(minutes=30),
            max_escalations=5
        )
        
        await escalation_manager.register_policy(escalation_policy)
        
        # Create critical alert
        critical_alert = Alert(
            alert_id='ESC_TEST_001',
            alert_type=AlertType.RISK_LIMIT_BREACH,
            severity=AlertSeverity.CRITICAL,
            title='Position Limit Exceeded',
            description='BTC position exceeds maximum allowed',
            timestamp=datetime.now(timezone.utc),
            source='risk_management',
            metadata={'position_size': 15.0, 'limit': 10.0, 'instrument': 'BTC/USDT'}
        )
        
        # Start escalation
        escalation_id = await escalation_manager.start_escalation(
            alert=critical_alert,
            policy_id='risk_critical'
        )
        
        assert escalation_id is not None
        
        # Check initial notification (Level 1)
        initial_status = await escalation_manager.get_escalation_status(escalation_id)
        assert initial_status['current_level'] == 1
        assert initial_status['notifications_sent'] > 0
        
        # Simulate no acknowledgment for 5 minutes
        await asyncio.sleep(0.1)  # In real scenario, wait 5 minutes
        
        # Trigger escalation to Level 2
        await escalation_manager.check_escalations()
        
        level2_status = await escalation_manager.get_escalation_status(escalation_id)
        assert level2_status['current_level'] == 2
        assert 'risk_manager@example.com' in level2_status['notified_recipients']
        
        # Test acknowledgment
        ack_result = await escalation_manager.acknowledge_alert(
            escalation_id=escalation_id,
            acknowledged_by='risk_manager@example.com',
            notes='Reviewing position and taking action'
        )
        
        assert ack_result['acknowledged'] == True
        assert ack_result['escalation_stopped'] == True
        
        # Verify escalation history
        history = await escalation_manager.get_escalation_history(escalation_id)
        assert len(history['events']) >= 3  # Start, Level 2, Acknowledgment
        assert history['total_duration'] > timedelta(0)
        assert history['final_level'] == 2
    
    @pytest.mark.asyncio
    async def test_alert_rules_engine(self, alert_system, sample_alert_rules):
        """Test alert rules engine with complex conditions"""
        # Register alert rules
        for rule in sample_alert_rules:
            await alert_system.register_rule(rule)
        
        # Test data that should trigger alerts
        test_metrics = [
            # Should trigger large loss alert
            {
                'metric': 'daily_pnl',
                'value': -15000,
                'timestamp': datetime.now(timezone.utc),
                'labels': {'account': 'main', 'strategy': 'grid'}
            },
            # Should trigger high latency alert
            {
                'metric': 'order_latency_p95',
                'value': 150,  # 150ms
                'timestamp': datetime.now(timezone.utc),
                'labels': {'exchange': 'binance', 'instrument': 'BTC/USDT'}
            },
            # Should trigger risk limit breach
            {
                'metric': 'position_exposure',
                'value': 1500000,  # $1.5M
                'timestamp': datetime.now(timezone.utc),
                'labels': {'instrument': 'BTC/USDT', 'side': 'long'}
            },
            # Should NOT trigger any alert
            {
                'metric': 'daily_pnl',
                'value': 5000,  # Positive P&L
                'timestamp': datetime.now(timezone.utc),
                'labels': {'account': 'main'}
            }
        ]
        
        triggered_alerts = []
        
        # Process metrics through rules engine
        for metric in test_metrics:
            alerts = await alert_system.evaluate_rules(metric)
            triggered_alerts.extend(alerts)
        
        # Verify correct alerts were triggered
        assert len(triggered_alerts) == 3  # 3 metrics should trigger alerts
        
        alert_types = [alert.alert_type for alert in triggered_alerts]
        assert AlertType.LARGE_LOSS in alert_types
        assert AlertType.HIGH_LATENCY in alert_types
        assert AlertType.RISK_LIMIT_BREACH in alert_types
        
        # Check alert details
        large_loss_alert = next(a for a in triggered_alerts if a.alert_type == AlertType.LARGE_LOSS)
        assert large_loss_alert.severity == AlertSeverity.ERROR
        assert abs(large_loss_alert.metadata['current_value']) == 15000
        assert large_loss_alert.metadata['threshold'] == -10000
        
        # Test cooldown period
        # Try to trigger the same alert again
        repeat_metric = {
            'metric': 'daily_pnl',
            'value': -16000,
            'timestamp': datetime.now(timezone.utc) + timedelta(minutes=10),
            'labels': {'account': 'main', 'strategy': 'grid'}
        }
        
        repeat_alerts = await alert_system.evaluate_rules(repeat_metric)
        
        # Should be blocked by cooldown
        large_loss_repeats = [a for a in repeat_alerts if a.alert_type == AlertType.LARGE_LOSS]
        assert len(large_loss_repeats) == 0  # Cooldown should prevent alert
    
    @pytest.mark.asyncio
    async def test_notification_channels(self, notification_service):
        """Test different notification channel implementations"""
        test_alert = Alert(
            alert_id='NOTIF_TEST_001',
            alert_type=AlertType.MARKET_VOLATILITY_SPIKE,
            severity=AlertSeverity.WARNING,
            title='High Market Volatility',
            description='BTC volatility exceeds normal range',
            timestamp=datetime.now(timezone.utc),
            source='market_monitor',
            metadata={'volatility': 0.15, 'normal_range': [0.02, 0.08]}
        )
        
        # Test Email notification
        email_result = await notification_service.send_email(
            alert=test_alert,
            recipients=['trading@example.com', 'risk@example.com'],
            smtp_config={
                'host': 'smtp.example.com',
                'port': 587,
                'username': 'alerts@example.com',
                'password': 'test_password',
                'use_tls': True
            }
        )
        
        assert email_result['sent'] == True
        assert len(email_result['recipients']) == 2
        assert 'message_id' in email_result
        
        # Verify email content
        assert test_alert.title in email_result['subject']
        assert test_alert.description in email_result['body']
        assert 'Severity: WARNING' in email_result['body']
        
        # Test Slack notification
        slack_result = await notification_service.send_slack(
            alert=test_alert,
            webhook_url='https://hooks.slack.com/services/TEST/WEBHOOK',
            channel='#trading-alerts',
            username='Trading Bot',
            icon_emoji=':warning:'
        )
        
        assert slack_result['sent'] == True
        assert slack_result['channel'] == '#trading-alerts'
        
        # Verify Slack message format
        slack_message = slack_result['message']
        assert 'attachments' in slack_message
        assert slack_message['attachments'][0]['color'] == 'warning'  # Yellow for WARNING
        assert 'fields' in slack_message['attachments'][0]
        
        # Test Webhook notification
        webhook_result = await notification_service.send_webhook(
            alert=test_alert,
            webhook_url='https://api.example.com/alerts',
            headers={'Authorization': 'Bearer test_token'},
            method='POST'
        )
        
        assert webhook_result['sent'] == True
        assert webhook_result['status_code'] == 200
        assert 'response_time_ms' in webhook_result
        
        # Test PagerDuty integration
        pagerduty_result = await notification_service.send_pagerduty(
            alert=test_alert,
            routing_key='test_routing_key',
            dedup_key=test_alert.alert_id
        )
        
        assert pagerduty_result['sent'] == True
        assert pagerduty_result['incident_key'] == test_alert.alert_id
        assert pagerduty_result['action'] == 'trigger'
    
    @pytest.mark.asyncio
    async def test_alert_aggregation(self, alert_system):
        """Test alert aggregation to reduce noise"""
        # Generate multiple similar alerts
        base_time = datetime.now(timezone.utc)
        order_rejection_alerts = []
        
        for i in range(50):
            alert = Alert(
                alert_id=f'AGG_TEST_{i:03d}',
                alert_type=AlertType.ORDER_REJECTION,
                severity=AlertSeverity.WARNING,
                title='Order Rejected',
                description=f'Order rejected by exchange',
                timestamp=base_time + timedelta(seconds=i),
                source='order_manager',
                metadata={
                    'order_id': f'ORD_{i:06d}',
                    'reason': 'insufficient_balance' if i % 3 == 0 else 'price_out_of_range',
                    'instrument': 'BTC/USDT' if i % 2 == 0 else 'ETH/USDT'
                }
            )
            order_rejection_alerts.append(alert)
        
        # Configure aggregation rules
        aggregation_rule = {
            'rule_id': 'order_rejection_aggregation',
            'alert_type': AlertType.ORDER_REJECTION,
            'aggregation_window': timedelta(minutes=1),
            'min_count': 5,  # Aggregate if more than 5 alerts
            'group_by': ['reason', 'instrument']
        }
        
        await alert_system.configure_aggregation(aggregation_rule)
        
        # Process alerts
        for alert in order_rejection_alerts:
            await alert_system.process_alert(alert)
        
        # Get aggregated alerts
        aggregated = await alert_system.get_aggregated_alerts(
            start_time=base_time,
            end_time=base_time + timedelta(minutes=1)
        )
        
        assert len(aggregated) < len(order_rejection_alerts)  # Should be aggregated
        
        # Check aggregation groups
        for agg_alert in aggregated:
            assert agg_alert.alert_type == AlertType.ORDER_REJECTION
            assert 'aggregation_count' in agg_alert.metadata
            assert agg_alert.metadata['aggregation_count'] >= 5
            assert 'grouped_by' in agg_alert.metadata
            
            # Verify title indicates aggregation
            assert 'Multiple' in agg_alert.title or 'Aggregated' in agg_alert.title
            
            # Check breakdown by reason
            if 'reason_breakdown' in agg_alert.metadata:
                breakdown = agg_alert.metadata['reason_breakdown']
                assert sum(breakdown.values()) == agg_alert.metadata['aggregation_count']
    
    @pytest.mark.asyncio
    async def test_alert_auto_resolution(self, alert_system):
        """Test automatic alert resolution"""
        # Create alert with auto-resolution enabled
        auto_resolve_alert = Alert(
            alert_id='AUTO_RESOLVE_001',
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            title='High Latency Detected',
            description='API latency exceeds threshold',
            timestamp=datetime.now(timezone.utc),
            source='latency_monitor',
            metadata={'latency_ms': 250, 'threshold_ms': 100},
            auto_resolve=True,
            ttl=timedelta(minutes=10)  # Auto-resolve after 10 minutes
        )
        
        # Process alert
        await alert_system.process_alert(auto_resolve_alert)
        
        # Verify alert is active
        active_alerts = await alert_system.get_active_alerts()
        assert any(a.alert_id == 'AUTO_RESOLVE_001' for a in active_alerts)
        
        # Simulate condition improvement
        recovery_metric = {
            'metric': 'api_latency',
            'value': 50,  # Below threshold
            'timestamp': datetime.now(timezone.utc) + timedelta(minutes=2),
            'labels': {'endpoint': 'order_submit'}
        }
        
        # Process recovery
        await alert_system.check_auto_resolution(recovery_metric)
        
        # Verify alert is resolved
        resolved_alerts = await alert_system.get_resolved_alerts(
            start_time=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        resolved = next((a for a in resolved_alerts if a.alert_id == 'AUTO_RESOLVE_001'), None)
        assert resolved is not None
        assert resolved.metadata['resolution_reason'] == 'condition_cleared'
        assert 'resolution_timestamp' in resolved.metadata
        
        # Test TTL-based auto-resolution
        ttl_alert = Alert(
            alert_id='TTL_RESOLVE_001',
            alert_type=AlertType.CONNECTIVITY_LOSS,
            severity=AlertSeverity.ERROR,
            title='Connection Lost',
            description='Lost connection to exchange',
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=15),
            source='connection_monitor',
            auto_resolve=True,
            ttl=timedelta(minutes=10)
        )
        
        await alert_system.process_alert(ttl_alert)
        
        # Run TTL check
        await alert_system.check_ttl_expiration()
        
        # Should be auto-resolved due to TTL
        ttl_resolved = await alert_system.get_alert_status(ttl_alert.alert_id)
        assert ttl_resolved['status'] == 'resolved'
        assert ttl_resolved['resolution_reason'] == 'ttl_expired'
    
    @pytest.mark.asyncio
    async def test_alert_rate_limiting(self, alert_system, notification_service):
        """Test alert rate limiting to prevent spam"""
        # Configure rate limits
        rate_limits = {
            'global': {
                'max_alerts_per_minute': 100,
                'max_alerts_per_hour': 1000
            },
            'per_type': {
                AlertType.HIGH_LATENCY: {
                    'max_per_minute': 10,
                    'max_per_hour': 50
                }
            },
            'per_channel': {
                NotificationChannel.SMS: {
                    'max_per_minute': 5,
                    'max_per_hour': 20
                }
            }
        }
        
        await notification_service.configure_rate_limits(rate_limits)
        
        # Generate burst of alerts
        burst_alerts = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(150):  # Exceeds per-minute limit
            alert = Alert(
                alert_id=f'RATE_TEST_{i:03d}',
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.WARNING,
                title='Latency Alert',
                description='High latency detected',
                timestamp=base_time + timedelta(seconds=i*0.1),  # 15 seconds total
                source='monitor'
            )
            burst_alerts.append(alert)
        
        # Process alerts
        results = []
        rate_limited_count = 0
        
        for alert in burst_alerts:
            result = await alert_system.process_alert(alert)
            results.append(result)
            
            if result.get('rate_limited', False):
                rate_limited_count += 1
        
        # Verify rate limiting
        assert rate_limited_count > 0  # Some alerts should be rate limited
        
        # Check rate limit stats
        stats = await notification_service.get_rate_limit_stats()
        
        assert stats['global']['current_minute_count'] <= 100
        assert stats['per_type'][AlertType.HIGH_LATENCY]['dropped_count'] > 0
        
        # Test rate limit headers in responses
        for result in results[-10:]:  # Check last 10
            if result.get('rate_limited'):
                assert 'rate_limit_reset' in result
                assert 'retry_after' in result
    
    @pytest.mark.asyncio
    async def test_alert_templates(self, notification_service):
        """Test alert message templates for different channels"""
        # Define templates
        templates = {
            NotificationChannel.EMAIL: {
                'subject': '[{severity}] {title} - Trading System Alert',
                'body': '''
                <html>
                <body>
                    <h2 style="color: {color}">{title}</h2>
                    <p>{description}</p>
                    <h3>Details:</h3>
                    <ul>
                        {details}
                    </ul>
                    <h3>Recommended Actions:</h3>
                    <ol>
                        {actions}
                    </ol>
                    <hr>
                    <p><small>Alert ID: {alert_id} | Time: {timestamp}</small></p>
                </body>
                </html>
                '''
            },
            NotificationChannel.SLACK: {
                'format': 'blocks',
                'blocks': [
                    {
                        'type': 'header',
                        'text': {
                            'type': 'plain_text',
                            'text': '{title}'
                        }
                    },
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': '{description}'
                        }
                    },
                    {
                        'type': 'section',
                        'fields': '{fields}'
                    }
                ]
            },
            NotificationChannel.SMS: {
                'format': 'text',
                'template': '{severity}: {title}. {description}. ID: {alert_id}'
            }
        }
        
        await notification_service.register_templates(templates)
        
        # Test alert with templates
        test_alert = Alert(
            alert_id='TMPL_TEST_001',
            alert_type=AlertType.LARGE_LOSS,
            severity=AlertSeverity.CRITICAL,
            title='Critical Loss Alert',
            description='Trading loss exceeds critical threshold',
            timestamp=datetime.now(timezone.utc),
            source='risk_system',
            metadata={
                'current_loss': -25000,
                'threshold': -20000,
                'instruments': ['BTC/USDT', 'ETH/USDT']
            },
            recommended_actions=[
                'Immediately review all open positions',
                'Consider closing losing positions',
                'Contact risk management team'
            ]
        )
        
        # Generate messages for each channel
        messages = {}
        
        for channel in [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS]:
            message = await notification_service.format_alert_message(
                alert=test_alert,
                channel=channel,
                template=templates.get(channel)
            )
            messages[channel] = message
        
        # Verify Email formatting
        email_msg = messages[NotificationChannel.EMAIL]
        assert '[CRITICAL]' in email_msg['subject']
        assert 'Critical Loss Alert' in email_msg['subject']
        assert '<h2' in email_msg['body']  # HTML formatting
        assert 'color:' in email_msg['body']  # Color coding
        assert all(action in email_msg['body'] for action in test_alert.recommended_actions)
        
        # Verify Slack formatting
        slack_msg = messages[NotificationChannel.SLACK]
        assert len(slack_msg['blocks']) >= 3
        assert slack_msg['blocks'][0]['type'] == 'header'
        assert 'Critical Loss Alert' in slack_msg['blocks'][0]['text']['text']
        
        # Verify SMS formatting (shortened)
        sms_msg = messages[NotificationChannel.SMS]
        assert len(sms_msg['text']) <= 160  # SMS length limit
        assert 'CRITICAL:' in sms_msg['text']
        assert 'TMPL_TEST_001' in sms_msg['text']
    
    @pytest.mark.asyncio
    async def test_alert_analytics(self, alert_system):
        """Test alert analytics and reporting"""
        # Generate historical alert data
        alert_history = []
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        # Create alerts over 7 days
        for day in range(7):
            daily_time = base_time + timedelta(days=day)
            
            # Different alert patterns per day
            if day % 2 == 0:  # High activity days
                alert_count = 50
            else:  # Low activity days
                alert_count = 20
            
            for i in range(alert_count):
                alert_type = np.random.choice(list(AlertType))
                severity = np.random.choice(list(AlertSeverity))
                
                alert = Alert(
                    alert_id=f'HIST_{day:02d}_{i:03d}',
                    alert_type=alert_type,
                    severity=severity,
                    title=f'{alert_type.value} Alert',
                    description='Historical alert for analytics',
                    timestamp=daily_time + timedelta(hours=np.random.randint(0, 24)),
                    source='various_systems'
                )
                alert_history.append(alert)
        
        # Store historical alerts
        for alert in alert_history:
            await alert_system.store_alert(alert)
        
        # Generate analytics report
        analytics = await alert_system.generate_analytics_report(
            start_time=base_time,
            end_time=datetime.now(timezone.utc),
            group_by=['day', 'alert_type', 'severity']
        )
        
        # Verify analytics content
        assert 'summary' in analytics
        assert 'total_alerts' in analytics['summary']
        assert analytics['summary']['total_alerts'] == len(alert_history)
        
        assert 'by_type' in analytics
        assert len(analytics['by_type']) > 0
        
        assert 'by_severity' in analytics
        assert all(sev.value in analytics['by_severity'] for sev in AlertSeverity)
        
        assert 'time_series' in analytics
        assert len(analytics['time_series']['daily']) == 7
        
        # Check patterns
        assert 'patterns' in analytics
        patterns = analytics['patterns']
        assert 'peak_hours' in patterns
        assert 'quiet_hours' in patterns
        assert 'most_common_alerts' in patterns
        
        # Verify trend analysis
        assert 'trends' in analytics
        trends = analytics['trends']
        assert 'alert_volume_trend' in trends  # increasing/decreasing/stable
        
        # Check alert resolution metrics
        assert 'resolution_metrics' in analytics
        resolution = analytics['resolution_metrics']
        assert 'mean_time_to_resolve' in resolution
        assert 'auto_resolved_percentage' in resolution
        assert 'escalation_rate' in resolution


class TestAlertSystemIntegration:
    """Test alert system integration with other components"""
    
    @pytest.mark.asyncio
    async def test_trading_system_integration(self, alert_system):
        """Test integration with trading system components"""
        # Simulate trading system events
        trading_events = [
            {
                'event': 'position_opened',
                'data': {
                    'instrument': 'BTC/USDT',
                    'side': 'LONG',
                    'size': 2.5,
                    'entry_price': 50000
                }
            },
            {
                'event': 'stop_loss_triggered',
                'data': {
                    'instrument': 'BTC/USDT',
                    'loss_amount': -5000,
                    'exit_price': 48000
                }
            },
            {
                'event': 'margin_level_low',
                'data': {
                    'current_margin_level': 0.15,
                    'required_margin_level': 0.20,
                    'additional_margin_needed': 10000
                }
            }
        ]
        
        # Process events and generate appropriate alerts
        generated_alerts = []
        
        for event in trading_events:
            if event['event'] == 'stop_loss_triggered':
                alert = Alert(
                    alert_id=f"TRADE_ALERT_{event['event']}",
                    alert_type=AlertType.LARGE_LOSS,
                    severity=AlertSeverity.WARNING,
                    title='Stop Loss Triggered',
                    description=f"Stop loss executed on {event['data']['instrument']}",
                    timestamp=datetime.now(timezone.utc),
                    source='trading_system',
                    metadata=event['data']
                )
                generated_alerts.append(alert)
                
            elif event['event'] == 'margin_level_low':
                alert = Alert(
                    alert_id=f"TRADE_ALERT_{event['event']}",
                    alert_type=AlertType.MARGIN_CALL,
                    severity=AlertSeverity.CRITICAL,
                    title='Margin Call Warning',
                    description='Margin level below required minimum',
                    timestamp=datetime.now(timezone.utc),
                    source='risk_management',
                    metadata=event['data'],
                    recommended_actions=[
                        f"Deposit ${event['data']['additional_margin_needed']} immediately",
                        'Reduce position sizes',
                        'Close losing positions'
                    ]
                )
                generated_alerts.append(alert)
        
        # Process alerts
        for alert in generated_alerts:
            result = await alert_system.process_alert(alert)
            assert result['processed'] == True
        
        # Verify trading-specific alert handling
        margin_alert = next(a for a in generated_alerts if a.alert_type == AlertType.MARGIN_CALL)
        assert margin_alert.severity == AlertSeverity.CRITICAL
        assert len(margin_alert.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_system_integration(self, alert_system):
        """Test integration with monitoring and metrics systems"""
        # Simulate monitoring metrics that trigger alerts
        monitoring_data = {
            'cpu_usage': 85.5,
            'memory_usage': 92.3,
            'disk_usage': 78.2,
            'api_latency_p99': 2500,  # 2.5 seconds
            'error_rate': 0.15,  # 15% error rate
            'active_connections': 950
        }
        
        # Define thresholds
        thresholds = {
            'cpu_usage': 80,
            'memory_usage': 90,
            'api_latency_p99': 1000,
            'error_rate': 0.05
        }
        
        # Check thresholds and generate alerts
        monitoring_alerts = []
        
        for metric, value in monitoring_data.items():
            if metric in thresholds and value > thresholds[metric]:
                severity = AlertSeverity.WARNING
                if metric == 'memory_usage' and value > 90:
                    severity = AlertSeverity.CRITICAL
                elif metric == 'error_rate' and value > 0.10:
                    severity = AlertSeverity.ERROR
                
                alert = Alert(
                    alert_id=f'MON_{metric.upper()}_{int(time.time())}',
                    alert_type=AlertType.RESOURCE_EXHAUSTION if 'usage' in metric else AlertType.SYSTEM_ERROR,
                    severity=severity,
                    title=f'High {metric.replace("_", " ").title()}',
                    description=f'{metric} is at {value}, exceeding threshold of {thresholds[metric]}',
                    timestamp=datetime.now(timezone.utc),
                    source='monitoring_system',
                    metadata={
                        'metric': metric,
                        'current_value': value,
                        'threshold': thresholds[metric],
                        'percentage_over': ((value - thresholds[metric]) / thresholds[metric]) * 100
                    }
                )
                monitoring_alerts.append(alert)
        
        # Process monitoring alerts
        for alert in monitoring_alerts:
            await alert_system.process_alert(alert)
        
        # Test system health dashboard
        health_summary = await alert_system.get_system_health_summary()
        
        assert health_summary['status'] in ['healthy', 'degraded', 'critical']
        assert health_summary['active_alerts'] >= len(monitoring_alerts)
        assert 'components' in health_summary
        assert 'recommendations' in health_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])