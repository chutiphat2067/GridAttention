"""
Monitoring and Alerts Tests for GridAttention Trading System
Tests monitoring accuracy, alert systems, and observability features
"""

import asyncio
import pytest
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
from unittest.mock import Mock, patch


class TestMetricsCollection:
    """Test metrics collection and accuracy"""
    
    async def test_real_time_metrics(self):
        """Test real-time metrics collection"""
        from metrics_collector import MetricsCollector
        
        collector = MetricsCollector({})
        
        print("Testing real-time metrics collection...")
        
        # Start collection
        await collector.start()
        
        # Generate test activity
        for i in range(100):
            # Trade execution metric
            await collector.record_metric('trade_execution', {
                'latency_ms': random.uniform(5, 50),
                'status': 'success' if random.random() > 0.05 else 'failed',
                'timestamp': time.time()
            })
            
            # Order book metric
            await collector.record_metric('order_book', {
                'spread': random.uniform(0.0001, 0.001),
                'depth': random.randint(10, 100),
                'timestamp': time.time()
            })
            
            await asyncio.sleep(0.01)
        
        # Get aggregated metrics
        metrics = await collector.get_metrics('1m')  # 1 minute window
        
        # Verify metrics accuracy
        assert 'trade_execution' in metrics
        assert metrics['trade_execution']['count'] == 100
        assert metrics['trade_execution']['success_rate'] > 0.9
        assert metrics['trade_execution']['avg_latency_ms'] < 50
        
        assert 'order_book' in metrics
        assert metrics['order_book']['avg_spread'] > 0
        assert metrics['order_book']['avg_depth'] > 0
        
        print(f"  ✓ Collected {metrics['trade_execution']['count']} trade metrics")
        print(f"  ✓ Success rate: {metrics['trade_execution']['success_rate']:.2%}")
        print(f"  ✓ Avg latency: {metrics['trade_execution']['avg_latency_ms']:.1f}ms")
    
    async def test_performance_metrics_accuracy(self):
        """Test accuracy of performance metrics"""
        from performance_tracker import PerformanceTracker
        
        tracker = PerformanceTracker({})
        
        print("Testing performance metrics accuracy...")
        
        # Simulate trades with known P&L
        trades = []
        expected_pnl = 0
        
        for i in range(50):
            entry_price = 50000 + random.uniform(-1000, 1000)
            exit_price = entry_price * (1 + random.uniform(-0.02, 0.03))
            amount = 0.1
            
            pnl = (exit_price - entry_price) * amount
            expected_pnl += pnl
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'amount': amount,
                'pnl': pnl,
                'timestamp': datetime.now() - timedelta(hours=i)
            })
        
        # Track trades
        for trade in trades:
            await tracker.record_trade(trade)
        
        # Get performance metrics
        performance = await tracker.get_performance_summary()
        
        # Verify accuracy
        assert abs(performance['total_pnl'] - expected_pnl) < 0.01
        assert performance['total_trades'] == len(trades)
        assert performance['win_rate'] == len([t for t in trades if t['pnl'] > 0]) / len(trades)
        
        # Verify Sharpe ratio calculation
        returns = [t['pnl'] / (t['entry_price'] * t['amount']) for t in trades]
        expected_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        assert abs(performance['sharpe_ratio'] - expected_sharpe) < 0.1
        
        print(f"  ✓ Total P&L accuracy: ${performance['total_pnl']:.2f}")
        print(f"  ✓ Win rate: {performance['win_rate']:.2%}")
        print(f"  ✓ Sharpe ratio: {performance['sharpe_ratio']:.2f}")
    
    async def test_system_resource_monitoring(self):
        """Test system resource monitoring"""
        from resource_monitor import ResourceMonitor
        
        monitor = ResourceMonitor({})
        
        print("Testing system resource monitoring...")
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate resource usage
        memory_allocations = []
        for i in range(10):
            # Allocate some memory
            data = [0] * (1024 * 1024)  # 1MB
            memory_allocations.append(data)
            
            # CPU intensive task
            sum(i**2 for i in range(10000))
            
            await asyncio.sleep(0.1)
        
        # Get resource metrics
        resources = await monitor.get_resource_metrics()
        
        # Verify monitoring
        assert 'cpu_percent' in resources
        assert 'memory_mb' in resources
        assert 'disk_io' in resources
        assert 'network_io' in resources
        
        assert resources['cpu_percent'] >= 0
        assert resources['memory_mb'] > 0
        assert resources['memory_available_mb'] > 0
        
        print(f"  ✓ CPU usage: {resources['cpu_percent']:.1f}%")
        print(f"  ✓ Memory usage: {resources['memory_mb']:.1f}MB")
        print(f"  ✓ Available memory: {resources['memory_available_mb']:.1f}MB")


class TestAlertSystem:
    """Test alert generation and delivery"""
    
    async def test_alert_triggers(self):
        """Test various alert trigger conditions"""
        from alert_manager import AlertManager
        
        manager = AlertManager({
            'thresholds': {
                'high_latency_ms': 100,
                'low_liquidity_ratio': 0.5,
                'position_limit_percent': 80,
                'daily_loss_percent': 2
            }
        })
        
        print("Testing alert triggers...")
        
        triggered_alerts = []
        
        # Test latency alert
        await manager.check_latency(150)  # 150ms > 100ms threshold
        
        # Test liquidity alert
        await manager.check_liquidity({
            'bid_volume': 100,
            'ask_volume': 300,
            'ratio': 0.33  # Below 0.5 threshold
        })
        
        # Test position limit alert
        await manager.check_position_usage(0.85)  # 85% > 80% threshold
        
        # Test daily loss alert
        await manager.check_daily_pnl(-0.025)  # -2.5% < -2% threshold
        
        # Get triggered alerts
        alerts = await manager.get_active_alerts()
        
        # Verify all alerts triggered
        alert_types = [a['type'] for a in alerts]
        assert 'high_latency' in alert_types
        assert 'low_liquidity' in alert_types
        assert 'position_limit_warning' in alert_types
        assert 'daily_loss_exceeded' in alert_types
        
        print(f"  ✓ Triggered {len(alerts)} alerts")
        for alert in alerts:
            print(f"    - {alert['type']}: {alert['message']}")
    
    async def test_alert_prioritization(self):
        """Test alert prioritization and deduplication"""
        from alert_manager import AlertManager
        
        manager = AlertManager({})
        
        print("Testing alert prioritization...")
        
        # Generate multiple alerts
        alerts_to_send = [
            {'type': 'system_error', 'severity': 'critical', 'message': 'Database connection lost'},
            {'type': 'high_latency', 'severity': 'warning', 'message': 'Latency spike detected'},
            {'type': 'system_error', 'severity': 'critical', 'message': 'Database connection lost'},  # Duplicate
            {'type': 'position_limit', 'severity': 'high', 'message': 'Position limit near'},
            {'type': 'info', 'severity': 'low', 'message': 'Daily report ready'},
        ]
        
        for alert in alerts_to_send:
            await manager.send_alert(alert)
        
        # Get prioritized alerts
        prioritized = await manager.get_prioritized_alerts()
        
        # Verify prioritization
        assert len(prioritized) == 4  # One duplicate removed
        assert prioritized[0]['severity'] == 'critical'  # Highest priority first
        assert prioritized[-1]['severity'] == 'low'  # Lowest priority last
        
        # Verify deduplication
        critical_alerts = [a for a in prioritized if a['severity'] == 'critical']
        assert len(critical_alerts) == 1  # Duplicate removed
        
        print(f"  ✓ Prioritized {len(prioritized)} alerts")
        print(f"  ✓ Removed duplicates")
        print(f"  ✓ Order: {[a['severity'] for a in prioritized]}")
    
    async def test_alert_delivery_channels(self):
        """Test multiple alert delivery channels"""
        from alert_dispatcher import AlertDispatcher
        
        dispatcher = AlertDispatcher({
            'channels': {
                'email': {'enabled': True, 'rate_limit': 10},
                'slack': {'enabled': True, 'rate_limit': 30},
                'sms': {'enabled': True, 'rate_limit': 5},
                'webhook': {'enabled': True, 'rate_limit': 100}
            }
        })
        
        print("Testing alert delivery channels...")
        
        # Critical alert - should go to all channels
        critical_alert = {
            'type': 'system_critical',
            'severity': 'critical',
            'message': 'Trading system emergency',
            'timestamp': datetime.now()
        }
        
        delivery_report = await dispatcher.dispatch_alert(critical_alert)
        
        # Verify delivery to all channels
        assert delivery_report['email']['sent'] is True
        assert delivery_report['slack']['sent'] is True
        assert delivery_report['sms']['sent'] is True
        assert delivery_report['webhook']['sent'] is True
        
        # Low priority alert - limited channels
        info_alert = {
            'type': 'daily_summary',
            'severity': 'info',
            'message': 'Daily trading summary',
            'timestamp': datetime.now()
        }
        
        info_delivery = await dispatcher.dispatch_alert(info_alert)
        
        # Should only go to some channels
        assert info_delivery['email']['sent'] is True
        assert info_delivery['slack']['sent'] is True
        assert info_delivery['sms']['sent'] is False  # Not for info alerts
        
        print(f"  ✓ Critical alert sent to {sum(1 for c in delivery_report.values() if c['sent'])} channels")
        print(f"  ✓ Info alert sent to {sum(1 for c in info_delivery.values() if c['sent'])} channels")


class TestDashboardIntegration:
    """Test dashboard and visualization integration"""
    
    async def test_dashboard_data_freshness(self):
        """Test dashboard data update frequency"""
        from dashboard_provider import DashboardProvider
        
        provider = DashboardProvider({
            'update_interval': 1  # 1 second
        })
        
        print("Testing dashboard data freshness...")
        
        # Get initial data
        initial_data = await provider.get_dashboard_data()
        initial_timestamp = initial_data['timestamp']
        
        # Wait for update
        await asyncio.sleep(1.5)
        
        # Get updated data
        updated_data = await provider.get_dashboard_data()
        updated_timestamp = updated_data['timestamp']
        
        # Verify update occurred
        assert updated_timestamp > initial_timestamp
        assert (updated_timestamp - initial_timestamp).total_seconds() >= 1
        
        # Verify data completeness
        required_sections = [
            'portfolio_summary',
            'active_positions',
            'recent_trades',
            'performance_metrics',
            'risk_metrics',
            'system_status'
        ]
        
        for section in required_sections:
            assert section in updated_data
            assert updated_data[section] is not None
        
        print(f"  ✓ Data updated after {(updated_timestamp - initial_timestamp).total_seconds():.1f}s")
        print(f"  ✓ All {len(required_sections)} sections present")
    
    async def test_real_time_chart_updates(self):
        """Test real-time chart data streaming"""
        from chart_streamer import ChartStreamer
        
        streamer = ChartStreamer({})
        
        print("Testing real-time chart updates...")
        
        received_updates = []
        
        # Subscribe to chart updates
        async def update_handler(data):
            received_updates.append(data)
        
        await streamer.subscribe('BTC/USDT', '1m', update_handler)
        
        # Generate some price updates
        for i in range(10):
            price_update = {
                'symbol': 'BTC/USDT',
                'price': 50000 + random.uniform(-100, 100),
                'volume': random.uniform(0.1, 1.0),
                'timestamp': time.time()
            }
            
            await streamer.process_price_update(price_update)
            await asyncio.sleep(0.1)
        
        # Verify updates received
        assert len(received_updates) >= 5  # Should aggregate to fewer candles
        
        # Verify candle structure
        for update in received_updates:
            assert 'open' in update
            assert 'high' in update
            assert 'low' in update
            assert 'close' in update
            assert 'volume' in update
            assert 'timestamp' in update
        
        print(f"  ✓ Received {len(received_updates)} chart updates")
        print(f"  ✓ Candle structure verified")


class TestLoggingAndAuditing:
    """Test logging and audit functionality"""
    
    async def test_structured_logging(self):
        """Test structured logging format"""
        from structured_logger import StructuredLogger
        
        logger = StructuredLogger({})
        
        print("Testing structured logging...")
        
        # Log various events
        events = [
            {
                'event': 'trade_executed',
                'level': 'info',
                'data': {
                    'trade_id': 'TRD_123',
                    'symbol': 'BTC/USDT',
                    'amount': 0.01,
                    'price': 50000
                }
            },
            {
                'event': 'error_occurred',
                'level': 'error',
                'data': {
                    'error_type': 'connection_timeout',
                    'service': 'exchange_api',
                    'retry_count': 3
                }
            }
        ]
        
        logged_events = []
        for event in events:
            logged = await logger.log(event['level'], event['event'], event['data'])
            logged_events.append(logged)
        
        # Verify structured format
        for logged in logged_events:
            assert 'timestamp' in logged
            assert 'level' in logged
            assert 'event' in logged
            assert 'data' in logged
            assert 'trace_id' in logged  # For distributed tracing
            assert 'host' in logged
            assert 'service' in logged
        
        # Verify log searchability
        search_results = await logger.search({
            'event': 'trade_executed',
            'time_range': '1h'
        })
        
        assert len(search_results) >= 1
        assert search_results[0]['data']['trade_id'] == 'TRD_123'
        
        print(f"  ✓ Logged {len(logged_events)} structured events")
        print(f"  ✓ All required fields present")
        print(f"  ✓ Log search functionality working")
    
    async def test_performance_profiling(self):
        """Test performance profiling capabilities"""
        from performance_profiler import PerformanceProfiler
        
        profiler = PerformanceProfiler({})
        
        print("Testing performance profiling...")
        
        # Profile different operations
        
        # Profile trade execution
        with profiler.profile('trade_execution'):
            # Simulate trade execution
            await asyncio.sleep(0.05)
            
        # Profile strategy calculation
        with profiler.profile('strategy_calculation'):
            # Simulate complex calculation
            sum(i**2 for i in range(10000))
            
        # Profile database query
        with profiler.profile('database_query'):
            # Simulate DB query
            await asyncio.sleep(0.02)
        
        # Get profiling results
        profile_data = await profiler.get_profile_summary()
        
        # Verify profiling data
        assert 'trade_execution' in profile_data
        assert profile_data['trade_execution']['avg_duration_ms'] >= 50
        assert profile_data['trade_execution']['count'] == 1
        
        assert 'strategy_calculation' in profile_data
        assert profile_data['strategy_calculation']['avg_duration_ms'] > 0
        
        assert 'database_query' in profile_data
        assert profile_data['database_query']['avg_duration_ms'] >= 20
        
        # Get bottlenecks
        bottlenecks = await profiler.identify_bottlenecks()
        
        assert len(bottlenecks) > 0
        assert bottlenecks[0]['operation'] == 'trade_execution'  # Slowest
        
        print(f"  ✓ Profiled {len(profile_data)} operations")
        print(f"  ✓ Identified {len(bottlenecks)} bottlenecks")
        print(f"  ✓ Slowest operation: {bottlenecks[0]['operation']} ({bottlenecks[0]['avg_duration_ms']:.1f}ms)")


# Monitoring test runner
async def run_monitoring_tests():
    """Run all monitoring and alert tests"""
    print("🚨 Running Monitoring and Alert Tests...")
    
    test_classes = [
        TestMetricsCollection,
        TestAlertSystem,
        TestDashboardIntegration,
        TestLoggingAndAuditing
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
    print(f"📊 Monitoring Test Results: {passed}/{total} passed")
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_monitoring_tests())
    exit(0 if success else 1)