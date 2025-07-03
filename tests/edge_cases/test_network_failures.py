"""
Network Failure Testing Suite for GridAttention Trading System
Tests connection failures, latency issues, packet loss, and network recovery mechanisms
"""

import pytest
import asyncio
import time
import random
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import logging
from dataclasses import dataclass
from enum import Enum
import aiohttp
import websockets
from asyncio import TimeoutError, CancelledError
import numpy as np

# GridAttention imports - aligned with system structure
from src.grid_attention_layer import GridAttentionLayer
from src.network.connection_manager import ConnectionManager
from src.network.failover_manager import FailoverManager
from src.network.heartbeat_monitor import HeartbeatMonitor
from src.network.reconnection_strategy import ReconnectionStrategy
from src.network.data_buffer import DataBuffer
from src.network.latency_monitor import LatencyMonitor
from src.execution_engine import ExecutionEngine
from src.market_data_manager import MarketDataManager

# Test utilities
from tests.utils.test_helpers import async_test, create_test_config
from tests.utils.network_simulator import (
    NetworkSimulator,
    simulate_packet_loss,
    simulate_latency_spike,
    simulate_connection_drop
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkFailureType(Enum):
    """Types of network failures"""
    CONNECTION_LOST = "connection_lost"
    HIGH_LATENCY = "high_latency"
    PACKET_LOSS = "packet_loss"
    DNS_FAILURE = "dns_failure"
    SSL_FAILURE = "ssl_failure"
    TIMEOUT = "timeout"
    PARTIAL_DISCONNECT = "partial_disconnect"
    BANDWIDTH_THROTTLE = "bandwidth_throttle"
    ROUTE_FAILURE = "route_failure"


@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    latency_ms: float
    packet_loss_percent: float
    bandwidth_mbps: float
    jitter_ms: float
    connection_stability: float
    last_disconnect: Optional[datetime]
    reconnect_attempts: int
    uptime_percent: float


class TestConnectionFailures:
    """Test various connection failure scenarios"""
    
    @pytest.fixture
    def connection_manager(self):
        """Create connection manager with failover"""
        config = create_test_config()
        config['network'] = {
            'primary_endpoint': 'wss://api.exchange.com/stream',
            'backup_endpoints': [
                'wss://backup1.exchange.com/stream',
                'wss://backup2.exchange.com/stream',
                'wss://backup3.exchange.com/stream'
            ],
            'connection_timeout': 5,
            'heartbeat_interval': 30,
            'reconnect_attempts': 5,
            'reconnect_delay': 1,
            'exponential_backoff': True
        }
        return ConnectionManager(config)
    
    @pytest.fixture
    def network_simulator(self):
        """Create network simulator for failure injection"""
        return NetworkSimulator()
    
    @async_test
    async def test_primary_connection_failure(self, connection_manager, network_simulator):
        """Test failover when primary connection fails"""
        # Simulate primary connection failure
        with patch('websockets.connect') as mock_connect:
            # Primary fails, backup succeeds
            mock_connect.side_effect = [
                ConnectionError("Primary connection failed"),
                AsyncMock()  # Backup connection succeeds
            ]
            
            result = await connection_manager.connect()
            
            assert result['connected'] is True
            assert result['endpoint'] != connection_manager.primary_endpoint
            assert result['failover_used'] is True
            assert mock_connect.call_count == 2
            
            # Verify connection state
            state = connection_manager.get_connection_state()
            assert state['active_endpoint'] in connection_manager.backup_endpoints
            assert state['primary_failed'] is True
    
    @async_test
    async def test_cascading_connection_failures(self, connection_manager):
        """Test multiple endpoint failures in sequence"""
        failure_sequence = []
        
        with patch('websockets.connect') as mock_connect:
            # All endpoints fail except the last one
            async def connect_with_failures(uri, **kwargs):
                failure_sequence.append(uri)
                if len(failure_sequence) < 3:
                    raise ConnectionError(f"Connection to {uri} failed")
                return AsyncMock()
            
            mock_connect.side_effect = connect_with_failures
            
            result = await connection_manager.connect()
            
            assert result['connected'] is True
            assert len(failure_sequence) == 3
            assert result['attempts'] == 3
            assert result['time_to_connect'] > 0
    
    @async_test
    async def test_total_connection_failure(self, connection_manager):
        """Test when all endpoints fail"""
        with patch('websockets.connect') as mock_connect:
            mock_connect.side_effect = ConnectionError("All connections failed")
            
            result = await connection_manager.connect()
            
            assert result['connected'] is False
            assert result['error'] == 'all_endpoints_failed'
            assert result['attempts'] == 4  # Primary + 3 backups
            
            # Verify system enters offline mode
            state = connection_manager.get_connection_state()
            assert state['status'] == 'offline'
            assert state['last_attempt'] is not None
    
    @async_test
    async def test_connection_drop_during_trading(self, grid_attention, network_simulator):
        """Test connection drop while orders are active"""
        # Place some orders
        orders = []
        for i in range(5):
            order = await grid_attention.execution_engine.place_order({
                'symbol': 'BTC/USDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': 50000 + i * 100,
                'size': 0.1
            })
            orders.append(order)
        
        # Simulate connection drop
        await network_simulator.drop_connection()
        
        # Verify protective actions
        response = await grid_attention.handle_connection_loss()
        
        assert response['orders_cancelled'] >= 0  # May or may not cancel depending on strategy
        assert response['positions_protected'] is True
        assert response['data_buffer_activated'] is True
        assert response['reconnection_initiated'] is True
        
        # Check if critical data is being buffered
        buffer_state = await grid_attention.get_buffer_state()
        assert buffer_state['buffering_active'] is True
        assert buffer_state['buffer_size'] >= 0
    
    @async_test
    async def test_websocket_reconnection(self, connection_manager):
        """Test WebSocket reconnection with exponential backoff"""
        reconnect_delays = []
        
        class MockWebSocket:
            def __init__(self, uri):
                self.closed = False
                
            async def __aenter__(self):
                return self
                
            async def __aexit__(self, *args):
                pass
                
            async def recv(self):
                await asyncio.sleep(0.1)
                raise websockets.ConnectionClosed(None, None)
        
        with patch('websockets.connect') as mock_connect:
            attempt_count = 0
            
            async def connect_with_tracking(uri, **kwargs):
                nonlocal attempt_count
                attempt_count += 1
                
                if attempt_count < 3:
                    # Track reconnection delay
                    if attempt_count > 1:
                        reconnect_delays.append(time.time())
                    raise ConnectionError("Connection failed")
                
                return MockWebSocket(uri)
            
            mock_connect.side_effect = connect_with_tracking
            
            # Attempt connection with retries
            result = await connection_manager.connect_with_retry()
            
            # Verify exponential backoff
            if len(reconnect_delays) > 1:
                for i in range(1, len(reconnect_delays)):
                    delay_diff = reconnect_delays[i] - reconnect_delays[i-1]
                    assert delay_diff > reconnect_delays[i-1] - reconnect_delays[i-2] if i > 1 else 0


class TestLatencyIssues:
    """Test high latency and timeout scenarios"""
    
    @pytest.fixture
    def latency_monitor(self):
        """Create latency monitoring service"""
        config = create_test_config()
        config['latency'] = {
            'warning_threshold_ms': 100,
            'critical_threshold_ms': 500,
            'timeout_threshold_ms': 5000,
            'measurement_interval': 1,
            'averaging_window': 60
        }
        return LatencyMonitor(config)
    
    @async_test
    async def test_high_latency_detection(self, grid_attention, latency_monitor):
        """Test detection of high latency conditions"""
        # Simulate increasing latency
        latency_sequence = [50, 100, 200, 500, 1000, 2000]  # ms
        
        alerts = []
        for latency in latency_sequence:
            # Inject latency
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0, latency / 1000]  # Convert to seconds
                
                response = await grid_attention.process_market_update({
                    'symbol': 'BTC/USDT',
                    'price': 50000,
                    'timestamp': datetime.now()
                })
                
                # Record latency
                measured_latency = await latency_monitor.record_latency(
                    operation='market_update',
                    latency_ms=latency
                )
                
                if measured_latency['alert']:
                    alerts.append(measured_latency)
        
        # Verify alerts were generated
        assert len(alerts) > 0
        assert any(a['level'] == 'warning' for a in alerts)
        assert any(a['level'] == 'critical' for a in alerts)
        
        # Check if system adjusted for high latency
        state = await grid_attention.get_latency_compensation_state()
        assert state['compensation_active'] is True
        assert state['order_lead_time_ms'] > 0
    
    @async_test
    async def test_timeout_handling(self, execution_engine):
        """Test handling of request timeouts"""
        # Configure short timeout
        execution_engine.set_timeout(1.0)  # 1 second
        
        with patch.object(execution_engine, '_send_order') as mock_send:
            # Simulate slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(2.0)  # Longer than timeout
                return {'status': 'filled'}
            
            mock_send.side_effect = slow_response
            
            # Attempt to place order
            start_time = time.time()
            try:
                result = await execution_engine.place_order({
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'price': 50000,
                    'size': 0.1
                })
            except TimeoutError:
                result = {'error': 'timeout'}
            
            duration = time.time() - start_time
            
            assert result.get('error') == 'timeout'
            assert duration < 1.5  # Should timeout at ~1 second
            
            # Verify timeout is logged and handled
            timeout_state = await execution_engine.get_timeout_state()
            assert timeout_state['last_timeout'] is not None
            assert timeout_state['timeout_count'] > 0
    
    @async_test
    async def test_variable_latency_adaptation(self, grid_attention, network_simulator):
        """Test adaptation to variable network latency"""
        # Simulate variable latency pattern
        latency_pattern = simulate_latency_spike(
            base_latency=50,
            spike_latency=1000,
            spike_duration=30,
            variation=0.3
        )
        
        adaptations = []
        for latency_data in latency_pattern:
            # Apply latency
            await network_simulator.set_latency(latency_data['latency_ms'])
            
            # Process with latency
            response = await grid_attention.process_with_latency_compensation({
                'symbol': 'BTC/USDT',
                'price': 50000,
                'timestamp': datetime.now()
            })
            
            if response.get('adaptation_applied'):
                adaptations.append({
                    'latency': latency_data['latency_ms'],
                    'compensation': response['compensation_ms'],
                    'strategy': response['strategy_adjustment']
                })
        
        # Verify adaptations
        assert len(adaptations) > 0
        
        # High latency should trigger more conservative strategies
        high_latency_adaptations = [a for a in adaptations if a['latency'] > 500]
        for adaptation in high_latency_adaptations:
            assert adaptation['strategy'] in ['conservative', 'latency_aware']
            assert adaptation['compensation'] > 0
    
    @async_test
    async def test_order_synchronization_with_latency(self, execution_engine):
        """Test order synchronization under high latency"""
        # Simulate high latency environment
        with patch('aiohttp.ClientSession.post') as mock_post:
            order_sequence = []
            
            async def delayed_response(url, **kwargs):
                order_data = kwargs.get('json', {})
                order_sequence.append(order_data['client_order_id'])
                
                # Simulate variable latency
                delay = random.uniform(0.1, 0.5)
                await asyncio.sleep(delay)
                
                return AsyncMock(
                    status=200,
                    json=AsyncMock(return_value={
                        'order_id': f"server_{order_data['client_order_id']}",
                        'status': 'accepted'
                    })
                )
            
            mock_post.side_effect = delayed_response
            
            # Place multiple orders quickly
            order_tasks = []
            for i in range(10):
                task = execution_engine.place_order({
                    'client_order_id': f'order_{i}',
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'price': 50000 + i,
                    'size': 0.1
                })
                order_tasks.append(task)
            
            # Wait for all orders
            results = await asyncio.gather(*order_tasks)
            
            # Verify order integrity despite latency
            assert len(results) == 10
            assert all(r.get('status') == 'accepted' for r in results)
            assert len(set(order_sequence)) == 10  # No duplicates


class TestPacketLoss:
    """Test packet loss scenarios"""
    
    @async_test
    async def test_packet_loss_detection(self, connection_manager, network_simulator):
        """Test detection of packet loss"""
        # Configure packet loss simulation
        packet_loss_rates = [0.0, 0.05, 0.10, 0.25, 0.50]  # 0%, 5%, 10%, 25%, 50%
        
        loss_metrics = []
        for loss_rate in packet_loss_rates:
            await network_simulator.set_packet_loss(loss_rate)
            
            # Send test messages
            sent = 100
            received = 0
            
            for i in range(sent):
                try:
                    response = await connection_manager.send_with_ack(
                        {'type': 'ping', 'id': i},
                        timeout=0.5
                    )
                    if response:
                        received += 1
                except TimeoutError:
                    pass
            
            actual_loss = (sent - received) / sent
            loss_metrics.append({
                'configured_loss': loss_rate,
                'actual_loss': actual_loss,
                'detection_accurate': abs(actual_loss - loss_rate) < 0.1
            })
        
        # Verify packet loss detection accuracy
        accurate_detections = sum(1 for m in loss_metrics if m['detection_accurate'])
        assert accurate_detections >= 3  # At least 3 out of 5 should be accurate
    
    @async_test
    async def test_message_retransmission(self, connection_manager):
        """Test message retransmission on packet loss"""
        retransmission_count = 0
        
        with patch.object(connection_manager, '_send_raw') as mock_send:
            # Simulate packet loss on first attempt
            async def send_with_loss(message):
                nonlocal retransmission_count
                if retransmission_count == 0:
                    retransmission_count += 1
                    raise TimeoutError("Packet lost")
                return {'ack': True, 'id': message.get('id')}
            
            mock_send.side_effect = send_with_loss
            
            # Send critical message with retransmission
            result = await connection_manager.send_critical_message(
                {'type': 'order', 'action': 'cancel_all'},
                max_retries=3,
                retry_delay=0.1
            )
            
            assert result['success'] is True
            assert result['attempts'] == 2  # Original + 1 retry
            assert retransmission_count == 1
    
    @async_test
    async def test_order_integrity_with_packet_loss(self, execution_engine, network_simulator):
        """Test order integrity under packet loss conditions"""
        # Set 20% packet loss
        await network_simulator.set_packet_loss(0.20)
        
        # Track order states
        order_states = {}
        
        # Place orders with acknowledgment tracking
        orders_placed = 0
        orders_confirmed = 0
        
        for i in range(20):
            try:
                order = await execution_engine.place_order_with_confirmation({
                    'order_id': f'order_{i}',
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'price': 50000,
                    'size': 0.1
                })
                
                orders_placed += 1
                if order.get('confirmed'):
                    orders_confirmed += 1
                    order_states[order['order_id']] = 'confirmed'
                else:
                    order_states[order['order_id']] = 'unconfirmed'
                    
            except Exception as e:
                order_states[f'order_{i}'] = 'failed'
        
        # Verify order integrity
        assert orders_confirmed >= orders_placed * 0.7  # At least 70% confirmed
        
        # Check for duplicate orders
        server_order_ids = [v for v in order_states.values() if v not in ['failed', 'unconfirmed']]
        assert len(server_order_ids) == len(set(server_order_ids))  # No duplicates


class TestDataSynchronization:
    """Test data synchronization during network issues"""
    
    @pytest.fixture
    def data_buffer(self):
        """Create data buffer for network failures"""
        config = create_test_config()
        config['buffer'] = {
            'max_size': 10000,
            'ttl_seconds': 300,
            'compression': True,
            'persistence': True
        }
        return DataBuffer(config)
    
    @async_test
    async def test_market_data_buffering(self, market_data_manager, data_buffer, network_simulator):
        """Test market data buffering during disconnection"""
        # Start receiving market data
        market_data = []
        async def collect_data():
            async for tick in market_data_manager.stream_market_data('BTC/USDT'):
                market_data.append(tick)
                if len(market_data) >= 100:
                    break
        
        # Run data collection with network interruption
        collect_task = asyncio.create_task(collect_data())
        
        # Wait a bit then disconnect
        await asyncio.sleep(0.5)
        await network_simulator.drop_connection()
        
        # Data should be buffered
        await asyncio.sleep(1.0)
        
        # Reconnect
        await network_simulator.restore_connection()
        
        # Complete data collection
        await collect_task
        
        # Check buffer statistics
        buffer_stats = await data_buffer.get_statistics()
        assert buffer_stats['buffered_count'] > 0
        assert buffer_stats['data_loss'] == 0
        
        # Verify data continuity
        timestamps = [tick['timestamp'] for tick in market_data]
        gaps = []
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]
            if gap > timedelta(seconds=2):  # Assuming 1-second ticks
                gaps.append(gap)
        
        assert len(gaps) == 0 or all(gap < timedelta(seconds=10) for gap in gaps)
    
    @async_test
    async def test_order_state_reconciliation(self, execution_engine, connection_manager):
        """Test order state reconciliation after reconnection"""
        # Place orders before disconnection
        orders_before = []
        for i in range(5):
            order = await execution_engine.place_order({
                'client_order_id': f'order_before_{i}',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 50000 + i * 100,
                'size': 0.1
            })
            orders_before.append(order)
        
        # Simulate disconnection
        await connection_manager.disconnect()
        
        # Orders might be filled during disconnection
        # Simulate some fills on exchange side
        exchange_states = {
            'order_before_0': 'filled',
            'order_before_1': 'partially_filled',
            'order_before_2': 'cancelled',
            'order_before_3': 'open',
            'order_before_4': 'filled'
        }
        
        # Reconnect
        await connection_manager.connect()
        
        # Reconcile order states
        reconciliation = await execution_engine.reconcile_orders()
        
        assert reconciliation['success'] is True
        assert reconciliation['orders_checked'] == 5
        assert reconciliation['discrepancies_found'] >= 0
        assert reconciliation['discrepancies_resolved'] == reconciliation['discrepancies_found']
        
        # Verify local state matches exchange state
        for order_id, expected_state in exchange_states.items():
            local_state = await execution_engine.get_order_state(order_id)
            assert local_state['status'] == expected_state
    
    @async_test
    async def test_position_synchronization(self, grid_attention, connection_manager):
        """Test position synchronization after network recovery"""
        # Get initial positions
        initial_positions = await grid_attention.get_positions()
        
        # Simulate network failure during trading
        await connection_manager.disconnect()
        
        # Simulate position changes on exchange during disconnect
        exchange_positions = {
            'BTC/USDT': {'size': 1.5, 'side': 'long', 'entry_price': 50000},
            'ETH/USDT': {'size': 10.0, 'side': 'short', 'entry_price': 3000}
        }
        
        # Reconnect
        await connection_manager.connect()
        
        # Sync positions
        sync_result = await grid_attention.sync_positions()
        
        assert sync_result['success'] is True
        assert sync_result['positions_synced'] > 0
        
        # Verify risk calculations are updated
        risk_state = await grid_attention.risk_management.recalculate_after_sync()
        assert risk_state['positions_updated'] is True
        assert risk_state['risk_metrics_recalculated'] is True


class TestNetworkRecovery:
    """Test network recovery mechanisms"""
    
    @pytest.fixture
    def failover_manager(self):
        """Create failover manager"""
        config = create_test_config()
        config['failover'] = {
            'health_check_interval': 30,
            'failover_threshold': 3,  # 3 consecutive failures
            'recovery_check_interval': 60,
            'auto_failback': True
        }
        return FailoverManager(config)
    
    @async_test
    async def test_automatic_failover(self, failover_manager, connection_manager):
        """Test automatic failover to backup endpoints"""
        # Monitor health checks
        health_history = []
        
        async def mock_health_check(endpoint):
            if endpoint == connection_manager.primary_endpoint:
                # Primary fails after 3 checks
                if len(health_history) >= 3:
                    return {'healthy': False, 'reason': 'timeout'}
            return {'healthy': True, 'latency_ms': 50}
        
        with patch.object(failover_manager, 'check_endpoint_health', mock_health_check):
            # Run health monitoring
            for i in range(5):
                health = await failover_manager.monitor_health()
                health_history.append(health)
                
                if health.get('failover_triggered'):
                    break
            
            # Verify failover was triggered
            assert any(h.get('failover_triggered') for h in health_history)
            
            # Check new active endpoint
            active = await failover_manager.get_active_endpoint()
            assert active != connection_manager.primary_endpoint
            assert active in connection_manager.backup_endpoints
    
    @async_test
    async def test_connection_recovery_queue(self, grid_attention, data_buffer):
        """Test queued operations during recovery"""
        # Queue operations during disconnect
        queued_operations = []
        
        # Disconnect
        await grid_attention.connection_manager.disconnect()
        
        # Try to perform operations (should be queued)
        for i in range(10):
            operation = {
                'type': 'order',
                'data': {
                    'symbol': 'BTC/USDT',
                    'side': 'buy' if i % 2 == 0 else 'sell',
                    'price': 50000 + i * 10,
                    'size': 0.1
                }
            }
            
            result = await grid_attention.queue_operation(operation)
            assert result['queued'] is True
            queued_operations.append(result['queue_id'])
        
        # Verify operations are queued
        queue_state = await data_buffer.get_queue_state()
        assert queue_state['queued_count'] == 10
        assert queue_state['queue_status'] == 'buffering'
        
        # Reconnect
        await grid_attention.connection_manager.connect()
        
        # Process queued operations
        processing_result = await grid_attention.process_queued_operations()
        
        assert processing_result['processed'] == 10
        assert processing_result['failed'] == 0
        assert processing_result['queue_cleared'] is True
    
    @async_test
    async def test_gradual_traffic_restoration(self, connection_manager, failover_manager):
        """Test gradual traffic restoration after recovery"""
        # Simulate recovery of primary endpoint
        primary_recovered = False
        
        async def mock_health_check(endpoint):
            if endpoint == connection_manager.primary_endpoint:
                return {'healthy': primary_recovered, 'latency_ms': 30}
            return {'healthy': True, 'latency_ms': 80}
        
        # Start on backup
        await failover_manager.failover_to_backup()
        
        # Primary recovers
        primary_recovered = True
        
        with patch.object(failover_manager, 'check_endpoint_health', mock_health_check):
            # Gradual failback process
            failback_stages = []
            
            for i in range(10):
                stage = await failover_manager.gradual_failback_step(i * 10)  # 0%, 10%, 20%...
                failback_stages.append(stage)
            
            # Verify gradual traffic shift
            traffic_percentages = [s['primary_traffic_percent'] for s in failback_stages]
            
            # Should gradually increase
            for i in range(1, len(traffic_percentages)):
                assert traffic_percentages[i] >= traffic_percentages[i-1]
            
            # Should reach 100% eventually
            assert traffic_percentages[-1] == 100


class TestHeartbeatMonitoring:
    """Test heartbeat and connection monitoring"""
    
    @pytest.fixture
    def heartbeat_monitor(self):
        """Create heartbeat monitor"""
        config = create_test_config()
        config['heartbeat'] = {
            'interval_seconds': 30,
            'timeout_seconds': 60,
            'max_missed': 2,
            'auto_reconnect': True
        }
        return HeartbeatMonitor(config)
    
    @async_test
    async def test_heartbeat_timeout_detection(self, heartbeat_monitor, connection_manager):
        """Test detection of missed heartbeats"""
        missed_count = 0
        
        async def mock_heartbeat():
            nonlocal missed_count
            missed_count += 1
            if missed_count > 2:
                raise TimeoutError("Heartbeat timeout")
            return {'pong': True, 'timestamp': time.time()}
        
        with patch.object(connection_manager, 'send_heartbeat', mock_heartbeat):
            # Monitor heartbeats
            disconnected = False
            
            for i in range(5):
                try:
                    result = await heartbeat_monitor.check_heartbeat()
                    if result.get('connection_lost'):
                        disconnected = True
                        break
                except TimeoutError:
                    disconnected = True
                    break
            
            assert disconnected is True
            assert missed_count > 2
    
    @async_test
    async def test_heartbeat_based_reconnection(self, heartbeat_monitor, connection_manager):
        """Test automatic reconnection on heartbeat failure"""
        heartbeat_failures = 0
        reconnect_attempted = False
        
        async def failing_heartbeat():
            nonlocal heartbeat_failures
            heartbeat_failures += 1
            raise TimeoutError("No heartbeat response")
        
        async def mock_reconnect():
            nonlocal reconnect_attempted
            reconnect_attempted = True
            return {'success': True}
        
        with patch.object(connection_manager, 'send_heartbeat', failing_heartbeat):
            with patch.object(connection_manager, 'reconnect', mock_reconnect):
                # Run heartbeat monitoring
                await heartbeat_monitor.monitor_connection()
                
                # Should detect failure and attempt reconnection
                assert heartbeat_failures >= heartbeat_monitor.max_missed
                assert reconnect_attempted is True


# Helper Classes and Functions

class NetworkQualityMonitor:
    """Monitor network quality metrics"""
    
    def __init__(self):
        self.latency_samples = []
        self.packet_loss_samples = []
        self.bandwidth_samples = []
        
    async def measure_network_quality(self) -> NetworkMetrics:
        """Measure current network quality"""
        # Measure latency
        latency = await self._measure_latency()
        
        # Measure packet loss
        packet_loss = await self._measure_packet_loss()
        
        # Measure bandwidth
        bandwidth = await self._measure_bandwidth()
        
        # Calculate jitter
        jitter = self._calculate_jitter()
        
        return NetworkMetrics(
            latency_ms=latency,
            packet_loss_percent=packet_loss,
            bandwidth_mbps=bandwidth,
            jitter_ms=jitter,
            connection_stability=self._calculate_stability(),
            last_disconnect=None,
            reconnect_attempts=0,
            uptime_percent=100.0
        )
    
    async def _measure_latency(self) -> float:
        """Measure round-trip latency"""
        start = time.perf_counter()
        # Simulate ping
        await asyncio.sleep(0.001)  # Minimal processing
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        self.latency_samples.append(latency_ms)
        
        # Keep last 100 samples
        if len(self.latency_samples) > 100:
            self.latency_samples.pop(0)
        
        return latency_ms
    
    async def _measure_packet_loss(self) -> float:
        """Measure packet loss rate"""
        # Simulate packet loss measurement
        return random.uniform(0, 0.05)  # 0-5% loss
    
    async def _measure_bandwidth(self) -> float:
        """Measure available bandwidth"""
        # Simulate bandwidth test
        return random.uniform(10, 100)  # 10-100 Mbps
    
    def _calculate_jitter(self) -> float:
        """Calculate latency variation (jitter)"""
        if len(self.latency_samples) < 2:
            return 0.0
        
        differences = []
        for i in range(1, len(self.latency_samples)):
            diff = abs(self.latency_samples[i] - self.latency_samples[i-1])
            differences.append(diff)
        
        return np.mean(differences) if differences else 0.0
    
    def _calculate_stability(self) -> float:
        """Calculate connection stability score (0-1)"""
        if not self.latency_samples:
            return 1.0
        
        # Factors: low latency, low jitter, no packet loss
        avg_latency = np.mean(self.latency_samples)
        jitter = self._calculate_jitter()
        
        # Simple stability score
        latency_score = max(0, 1 - avg_latency / 1000)  # Penalty for >1s latency
        jitter_score = max(0, 1 - jitter / 100)  # Penalty for >100ms jitter
        
        return (latency_score + jitter_score) / 2


def simulate_network_degradation(duration_seconds: int, 
                               degradation_type: str = 'gradual') -> List[NetworkMetrics]:
    """Simulate network degradation over time"""
    metrics = []
    monitor = NetworkQualityMonitor()
    
    for i in range(duration_seconds):
        if degradation_type == 'gradual':
            # Gradually worsen
            degradation_factor = i / duration_seconds
        elif degradation_type == 'sudden':
            # Sudden degradation at midpoint
            degradation_factor = 0 if i < duration_seconds / 2 else 1
        else:
            degradation_factor = 0
        
        # Apply degradation
        base_metrics = asyncio.run(monitor.measure_network_quality())
        
        # Worsen metrics based on degradation
        base_metrics.latency_ms *= (1 + degradation_factor * 10)
        base_metrics.packet_loss_percent *= (1 + degradation_factor * 20)
        base_metrics.bandwidth_mbps *= (1 - degradation_factor * 0.9)
        base_metrics.jitter_ms *= (1 + degradation_factor * 5)
        base_metrics.connection_stability *= (1 - degradation_factor * 0.8)
        
        metrics.append(base_metrics)
    
    return metrics