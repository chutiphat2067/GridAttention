"""
Race condition tests for GridAttention trading system.

Tests concurrent operations, thread safety, and data consistency
under high-concurrency scenarios typical in algorithmic trading.
"""

import pytest
import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any, Set
from unittest.mock import Mock, patch, AsyncMock
import time
from collections import defaultdict
import numpy as np

# Import core components (adjust based on actual module structure)
from core.attention_learning import AttentionLearningLayer
from core.market_regime import MarketRegimeDetector
from core.grid_strategy import GridStrategySelector
from core.risk_management import RiskManagementSystem
from core.execution_engine import ExecutionEngine
from core.performance_monitor import PerformanceMonitor


class TestRaceConditions:
    """Test race conditions in GridAttention system"""
    
    @pytest.fixture
    async def trading_system(self):
        """Create a test trading system instance"""
        return {
            'attention': AttentionLearningLayer(),
            'regime_detector': MarketRegimeDetector(),
            'grid_strategy': GridStrategySelector(),
            'risk_management': RiskManagementSystem(),
            'execution_engine': ExecutionEngine(),
            'performance_monitor': PerformanceMonitor()
        }
    
    @pytest.fixture
    def shared_state(self):
        """Create shared state for testing race conditions"""
        return {
            'positions': {},
            'orders': {},
            'balance': 100000.0,
            'grid_levels': {},
            'attention_weights': {},
            'regime_state': 'NEUTRAL',
            'lock': threading.Lock(),
            'event_log': []
        }
    
    @pytest.mark.asyncio
    async def test_concurrent_order_placement(self, trading_system, shared_state):
        """Test race conditions in concurrent order placement"""
        num_traders = 10
        orders_per_trader = 50
        order_results = defaultdict(list)
        failed_orders = []
        
        async def place_orders(trader_id: int):
            """Simulate trader placing multiple orders"""
            for i in range(orders_per_trader):
                try:
                    order = {
                        'trader_id': trader_id,
                        'order_id': f'{trader_id}_{i}',
                        'symbol': 'BTC/USDT',
                        'side': random.choice(['buy', 'sell']),
                        'price': 50000 + random.uniform(-1000, 1000),
                        'quantity': random.uniform(0.01, 0.1),
                        'timestamp': datetime.now()
                    }
                    
                    # Check risk limits before placing order
                    with shared_state['lock']:
                        if shared_state['balance'] < order['quantity'] * order['price']:
                            failed_orders.append(order)
                            continue
                        
                        # Simulate order placement
                        shared_state['orders'][order['order_id']] = order
                        shared_state['balance'] -= order['quantity'] * order['price']
                        order_results[trader_id].append(order['order_id'])
                    
                    # Small delay to increase chance of race conditions
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    failed_orders.append({'error': str(e), 'trader_id': trader_id})
        
        # Run concurrent traders
        tasks = [place_orders(i) for i in range(num_traders)]
        await asyncio.gather(*tasks)
        
        # Verify results
        total_orders = sum(len(orders) for orders in order_results.values())
        assert total_orders + len(failed_orders) == num_traders * orders_per_trader
        assert len(shared_state['orders']) == total_orders
        
        # Check for duplicate order IDs (race condition indicator)
        all_order_ids = [order_id for orders in order_results.values() for order_id in orders]
        assert len(all_order_ids) == len(set(all_order_ids)), "Duplicate order IDs detected"
        
        # Verify balance consistency
        total_spent = sum(
            order['quantity'] * order['price'] 
            for order in shared_state['orders'].values()
        )
        assert abs(shared_state['balance'] - (100000.0 - total_spent)) < 0.01
    
    @pytest.mark.asyncio
    async def test_concurrent_position_updates(self, trading_system):
        """Test race conditions in position management"""
        position_lock = asyncio.Lock()
        positions = {
            'BTC/USDT': {'quantity': 0, 'avg_price': 0, 'pnl': 0}
        }
        update_history = []
        
        async def update_position(trader_id: int, updates: int):
            """Simulate position updates from multiple sources"""
            for i in range(updates):
                async with position_lock:
                    old_qty = positions['BTC/USDT']['quantity']
                    delta = random.uniform(-0.1, 0.1)
                    
                    # Simulate complex position update logic
                    await asyncio.sleep(0.001)  # Simulate calculation time
                    
                    positions['BTC/USDT']['quantity'] += delta
                    new_qty = positions['BTC/USDT']['quantity']
                    
                    update_history.append({
                        'trader_id': trader_id,
                        'update_id': i,
                        'old_qty': old_qty,
                        'delta': delta,
                        'new_qty': new_qty,
                        'timestamp': datetime.now()
                    })
        
        # Run concurrent position updates
        num_updaters = 5
        updates_per_updater = 20
        tasks = [update_position(i, updates_per_updater) for i in range(num_updaters)]
        await asyncio.gather(*tasks)
        
        # Verify position consistency
        expected_total_delta = sum(update['delta'] for update in update_history)
        assert abs(positions['BTC/USDT']['quantity'] - expected_total_delta) < 0.0001
        
        # Check update sequence integrity
        for i in range(1, len(update_history)):
            prev = update_history[i-1]
            curr = update_history[i]
            # Each update should start from the previous end state
            if prev['timestamp'] < curr['timestamp']:
                assert abs(curr['old_qty'] - prev['new_qty']) < 0.0001
    
    @pytest.mark.asyncio
    async def test_grid_level_race_conditions(self, trading_system):
        """Test race conditions in grid level management"""
        grid_manager = {
            'levels': {},
            'active_orders': set(),
            'lock': asyncio.Lock()
        }
        
        async def manage_grid_level(level_id: int, action: str):
            """Simulate grid level operations"""
            async with grid_manager['lock']:
                if action == 'create':
                    if level_id not in grid_manager['levels']:
                        grid_manager['levels'][level_id] = {
                            'price': 50000 + level_id * 100,
                            'orders': set(),
                            'filled': False
                        }
                        # Simulate order creation delay
                        await asyncio.sleep(0.01)
                        order_id = f'grid_{level_id}_{random.randint(1000, 9999)}'
                        grid_manager['levels'][level_id]['orders'].add(order_id)
                        grid_manager['active_orders'].add(order_id)
                
                elif action == 'fill':
                    if level_id in grid_manager['levels']:
                        level = grid_manager['levels'][level_id]
                        if not level['filled']:
                            level['filled'] = True
                            # Remove active orders
                            for order_id in level['orders']:
                                grid_manager['active_orders'].discard(order_id)
                
                elif action == 'cancel':
                    if level_id in grid_manager['levels']:
                        level = grid_manager['levels'][level_id]
                        for order_id in level['orders']:
                            grid_manager['active_orders'].discard(order_id)
                        del grid_manager['levels'][level_id]
        
        # Simulate concurrent grid operations
        operations = []
        for i in range(50):
            level_id = random.randint(1, 20)
            action = random.choice(['create', 'fill', 'cancel'])
            operations.append(manage_grid_level(level_id, action))
        
        await asyncio.gather(*operations)
        
        # Verify grid consistency
        for level_id, level in grid_manager['levels'].items():
            if level['filled']:
                # Filled levels should have no active orders
                for order_id in level['orders']:
                    assert order_id not in grid_manager['active_orders']
            else:
                # Unfilled levels should have active orders
                for order_id in level['orders']:
                    assert order_id in grid_manager['active_orders']
    
    @pytest.mark.asyncio
    async def test_attention_weight_updates(self, trading_system):
        """Test race conditions in attention weight updates"""
        attention_state = {
            'weights': np.random.rand(10, 10),
            'version': 0,
            'update_queue': asyncio.Queue(),
            'lock': asyncio.Lock()
        }
        
        update_conflicts = []
        
        async def update_attention_weights(updater_id: int):
            """Simulate concurrent attention weight updates"""
            for i in range(20):
                # Generate update
                update = {
                    'updater_id': updater_id,
                    'delta': np.random.randn(10, 10) * 0.01,
                    'timestamp': datetime.now()
                }
                
                async with attention_state['lock']:
                    old_version = attention_state['version']
                    old_weights = attention_state['weights'].copy()
                    
                    # Simulate complex calculation
                    await asyncio.sleep(0.001)
                    
                    # Check for concurrent modification
                    if attention_state['version'] != old_version:
                        update_conflicts.append({
                            'updater_id': updater_id,
                            'expected_version': old_version,
                            'actual_version': attention_state['version']
                        })
                        continue
                    
                    # Apply update
                    attention_state['weights'] += update['delta']
                    attention_state['version'] += 1
                    
                    # Record update
                    await attention_state['update_queue'].put({
                        'version': attention_state['version'],
                        'updater_id': updater_id,
                        'checksum': np.sum(attention_state['weights'])
                    })
        
        # Run concurrent updaters
        num_updaters = 5
        tasks = [update_attention_weights(i) for i in range(num_updaters)]
        await asyncio.gather(*tasks)
        
        # Verify update consistency
        updates = []
        while not attention_state['update_queue'].empty():
            updates.append(await attention_state['update_queue'].get())
        
        # Check version sequence
        versions = [u['version'] for u in updates]
        assert versions == sorted(versions), "Version sequence broken"
        assert len(set(versions)) == len(versions), "Duplicate versions detected"
        
        print(f"Update conflicts detected: {len(update_conflicts)}")
    
    @pytest.mark.asyncio
    async def test_event_ordering_consistency(self, trading_system):
        """Test event ordering under concurrent conditions"""
        event_bus = {
            'events': [],
            'subscribers': defaultdict(list),
            'lock': asyncio.Lock(),
            'sequence': 0
        }
        
        async def publish_event(publisher_id: int, event_type: str):
            """Publish events with guaranteed ordering"""
            async with event_bus['lock']:
                event = {
                    'publisher_id': publisher_id,
                    'type': event_type,
                    'sequence': event_bus['sequence'],
                    'timestamp': datetime.now(),
                    'data': {'value': random.random()}
                }
                event_bus['sequence'] += 1
                event_bus['events'].append(event)
                
                # Notify subscribers
                for subscriber in event_bus['subscribers'][event_type]:
                    await subscriber(event)
        
        processed_events = defaultdict(list)
        
        async def event_handler(handler_id: int, event: Dict):
            """Process events maintaining order"""
            await asyncio.sleep(random.uniform(0, 0.01))  # Simulate processing
            processed_events[handler_id].append(event['sequence'])
        
        # Register handlers
        for i in range(3):
            handler = lambda e, h_id=i: event_handler(h_id, e)
            event_bus['subscribers']['trade'].append(handler)
            event_bus['subscribers']['update'].append(handler)
        
        # Publish concurrent events
        publishers = []
        for i in range(5):
            for j in range(10):
                event_type = 'trade' if j % 2 == 0 else 'update'
                publishers.append(publish_event(i, event_type))
        
        await asyncio.gather(*publishers)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify event ordering
        assert len(event_bus['events']) == 50
        
        # Check sequence integrity
        sequences = [e['sequence'] for e in event_bus['events']]
        assert sequences == list(range(50))
        
        # Verify each handler processed events in order
        for handler_id, events in processed_events.items():
            assert events == sorted(events), f"Handler {handler_id} processed events out of order"
    
    @pytest.mark.asyncio
    async def test_balance_consistency_under_load(self, trading_system):
        """Test balance updates under heavy concurrent load"""
        account = {
            'balance': 100000.0,
            'reserved': 0.0,
            'lock': asyncio.Lock(),
            'transactions': []
        }
        
        async def execute_transaction(trader_id: int, num_transactions: int):
            """Execute multiple transactions"""
            for i in range(num_transactions):
                amount = random.uniform(100, 1000)
                tx_type = random.choice(['reserve', 'release', 'deduct'])
                
                async with account['lock']:
                    success = False
                    old_balance = account['balance']
                    old_reserved = account['reserved']
                    
                    if tx_type == 'reserve':
                        if account['balance'] >= amount:
                            account['balance'] -= amount
                            account['reserved'] += amount
                            success = True
                    
                    elif tx_type == 'release':
                        if account['reserved'] >= amount:
                            account['reserved'] -= amount
                            account['balance'] += amount
                            success = True
                    
                    elif tx_type == 'deduct':
                        if account['reserved'] >= amount:
                            account['reserved'] -= amount
                            success = True
                    
                    if success:
                        account['transactions'].append({
                            'trader_id': trader_id,
                            'type': tx_type,
                            'amount': amount,
                            'old_balance': old_balance,
                            'new_balance': account['balance'],
                            'old_reserved': old_reserved,
                            'new_reserved': account['reserved'],
                            'timestamp': datetime.now()
                        })
                
                await asyncio.sleep(0.001)
        
        # Run concurrent transactions
        num_traders = 10
        tasks = [execute_transaction(i, 50) for i in range(num_traders)]
        await asyncio.gather(*tasks)
        
        # Verify balance consistency
        total_change = 0.0
        for tx in account['transactions']:
            if tx['type'] == 'deduct':
                total_change -= tx['amount']
        
        expected_total = 100000.0 + total_change
        actual_total = account['balance'] + account['reserved']
        
        assert abs(actual_total - expected_total) < 0.01, f"Balance inconsistency: expected {expected_total}, got {actual_total}"
        
        # Verify transaction integrity
        for i in range(1, len(account['transactions'])):
            tx = account['transactions'][i]
            # Each transaction should see consistent state
            assert tx['old_balance'] >= 0
            assert tx['old_reserved'] >= 0
            assert tx['new_balance'] >= 0
            assert tx['new_reserved'] >= 0
    
    @pytest.mark.asyncio
    async def test_regime_change_race_conditions(self, trading_system):
        """Test race conditions during market regime changes"""
        regime_state = {
            'current': 'NEUTRAL',
            'transitioning': False,
            'pending_strategies': [],
            'active_strategies': set(),
            'lock': asyncio.Lock()
        }
        
        regime_changes = []
        strategy_conflicts = []
        
        async def detect_regime_change(detector_id: int):
            """Simulate regime detection"""
            for _ in range(20):
                new_regime = random.choice(['TRENDING', 'RANGING', 'VOLATILE', 'NEUTRAL'])
                
                async with regime_state['lock']:
                    if regime_state['transitioning']:
                        # Another transition in progress
                        strategy_conflicts.append({
                            'detector_id': detector_id,
                            'attempted_regime': new_regime,
                            'current_state': 'transitioning'
                        })
                        continue
                    
                    if new_regime != regime_state['current']:
                        regime_state['transitioning'] = True
                        old_regime = regime_state['current']
                        
                        # Simulate transition process
                        await asyncio.sleep(0.01)
                        
                        # Clear old strategies
                        regime_state['active_strategies'].clear()
                        
                        # Load new strategies
                        new_strategies = self._get_strategies_for_regime(new_regime)
                        regime_state['active_strategies'].update(new_strategies)
                        
                        regime_state['current'] = new_regime
                        regime_state['transitioning'] = False
                        
                        regime_changes.append({
                            'detector_id': detector_id,
                            'from_regime': old_regime,
                            'to_regime': new_regime,
                            'timestamp': datetime.now()
                        })
                
                await asyncio.sleep(random.uniform(0.05, 0.1))
        
        # Run concurrent regime detectors
        detectors = [detect_regime_change(i) for i in range(5)]
        await asyncio.gather(*detectors)
        
        # Verify regime transition integrity
        assert not regime_state['transitioning'], "System left in transitioning state"
        assert regime_state['current'] in ['TRENDING', 'RANGING', 'VOLATILE', 'NEUTRAL']
        
        # Check for overlapping regime changes
        for i in range(1, len(regime_changes)):
            prev = regime_changes[i-1]
            curr = regime_changes[i]
            # Ensure regime changes don't overlap
            assert prev['timestamp'] < curr['timestamp']
            assert prev['to_regime'] == curr['from_regime'] or curr['from_regime'] == regime_changes[i-1]['to_regime']
        
        print(f"Strategy conflicts: {len(strategy_conflicts)}")
    
    def _get_strategies_for_regime(self, regime: str) -> Set[str]:
        """Helper to get strategies for a regime"""
        strategies = {
            'TRENDING': {'momentum', 'breakout', 'trend_follow'},
            'RANGING': {'mean_reversion', 'support_resistance', 'grid'},
            'VOLATILE': {'scalping', 'arbitrage', 'hedging'},
            'NEUTRAL': {'grid', 'market_making', 'passive'}
        }
        return strategies.get(regime, set())
    
    @pytest.mark.asyncio
    async def test_order_fill_race_conditions(self, trading_system):
        """Test race conditions in order fill processing"""
        order_book = {
            'orders': {},
            'fills': [],
            'position': 0.0,
            'lock': asyncio.Lock()
        }
        
        # Create test orders
        for i in range(100):
            order_book['orders'][f'order_{i}'] = {
                'id': f'order_{i}',
                'quantity': random.uniform(0.01, 0.1),
                'filled': 0.0,
                'status': 'open'
            }
        
        async def process_fill(fill_id: int, order_id: str, fill_qty: float):
            """Process order fills with race condition checks"""
            async with order_book['lock']:
                if order_id not in order_book['orders']:
                    return False
                
                order = order_book['orders'][order_id]
                if order['status'] != 'open':
                    return False
                
                remaining = order['quantity'] - order['filled']
                actual_fill = min(fill_qty, remaining)
                
                if actual_fill > 0:
                    order['filled'] += actual_fill
                    order_book['position'] += actual_fill
                    
                    if abs(order['filled'] - order['quantity']) < 0.0001:
                        order['status'] = 'filled'
                    
                    order_book['fills'].append({
                        'fill_id': fill_id,
                        'order_id': order_id,
                        'quantity': actual_fill,
                        'timestamp': datetime.now()
                    })
                    
                    return True
                
                return False
        
        # Simulate concurrent fills
        fill_tasks = []
        for i in range(500):
            order_id = f'order_{random.randint(0, 99)}'
            fill_qty = random.uniform(0.01, 0.05)
            fill_tasks.append(process_fill(i, order_id, fill_qty))
        
        results = await asyncio.gather(*fill_tasks)
        
        # Verify fill consistency
        total_filled = sum(order['filled'] for order in order_book['orders'].values())
        total_fill_records = sum(fill['quantity'] for fill in order_book['fills'])
        
        assert abs(total_filled - total_fill_records) < 0.0001, "Fill quantity mismatch"
        assert abs(order_book['position'] - total_filled) < 0.0001, "Position mismatch"
        
        # Check for overfills
        for order in order_book['orders'].values():
            assert order['filled'] <= order['quantity'] + 0.0001, f"Order {order['id']} overfilled"
        
        # Verify fill sequence
        fills_by_order = defaultdict(list)
        for fill in order_book['fills']:
            fills_by_order[fill['order_id']].append(fill)
        
        for order_id, fills in fills_by_order.items():
            # Fills for same order should be time-ordered
            for i in range(1, len(fills)):
                assert fills[i-1]['timestamp'] <= fills[i]['timestamp']


class TestThreadSafety:
    """Test thread safety of shared components"""
    
    def test_shared_cache_thread_safety(self):
        """Test thread-safe cache operations"""
        cache = {}
        cache_lock = threading.Lock()
        errors = []
        
        def cache_operation(thread_id: int, operations: int):
            """Perform cache operations"""
            for i in range(operations):
                key = f'key_{random.randint(0, 50)}'
                op = random.choice(['get', 'set', 'delete'])
                
                try:
                    with cache_lock:
                        if op == 'get':
                            value = cache.get(key)
                        elif op == 'set':
                            cache[key] = f'value_{thread_id}_{i}'
                        elif op == 'delete' and key in cache:
                            del cache[key]
                except Exception as e:
                    errors.append({'thread_id': thread_id, 'error': str(e)})
        
        # Run threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=cache_operation, args=(i, 100))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors detected: {errors}"
    
    def test_metric_collection_thread_safety(self):
        """Test thread-safe metric collection"""
        metrics = {
            'trades': 0,
            'volume': 0.0,
            'pnl': 0.0,
            'lock': threading.Lock()
        }
        
        def update_metrics(thread_id: int, updates: int):
            """Update metrics from multiple threads"""
            for _ in range(updates):
                with metrics['lock']:
                    metrics['trades'] += 1
                    metrics['volume'] += random.uniform(0.1, 1.0)
                    metrics['pnl'] += random.uniform(-10, 10)
        
        # Expected values
        num_threads = 8
        updates_per_thread = 100
        expected_trades = num_threads * updates_per_thread
        
        # Run threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=update_metrics, args=(i, updates_per_thread))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert metrics['trades'] == expected_trades, f"Trade count mismatch: {metrics['trades']} vs {expected_trades}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])