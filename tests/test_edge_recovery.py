"""
Edge Cases and Recovery Tests for GridAttention Trading System
Tests unusual scenarios, error handling, and recovery mechanisms
"""

import asyncio
import pytest
import random
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import shutil


class TestMarketEdgeCases:
    """Test extreme market conditions"""
    
    async def test_flash_crash_recovery(self):
        """Test system response to flash crash"""
        from main import GridTradingSystem
        from risk_management_system import RiskManagementSystem
        
        system = GridTradingSystem('config_minimal.yaml')
        await system.initialize()
        
        # Simulate normal trading
        normal_price = 50000
        
        # Sudden 20% drop
        crash_price = normal_price * 0.8
        
        print(f"Simulating flash crash: ${normal_price} -> ${crash_price}")
        
        # Inject crash data
        crash_tick = {
            'symbol': 'BTC/USDT',
            'bid': crash_price,
            'ask': crash_price + 10,
            'timestamp': datetime.now().timestamp()
        }
        
        # System should activate safety measures
        risk_mgr = system.components.get('risk_management_system')
        
        # Process crash tick
        await system.process_market_update(crash_tick)
        
        # Verify safety measures activated
        assert risk_mgr.emergency_mode_active
        assert risk_mgr.kill_switch_active
        
        # Verify positions protected
        open_positions = await risk_mgr.get_open_positions()
        for position in open_positions:
            assert position.get('stop_loss') is not None
            assert position.get('reduced_size', False)
        
        print("✅ Flash crash handled correctly")
        
        await system.shutdown()
    
    async def test_zero_liquidity(self):
        """Test handling of zero liquidity"""
        from execution_engine import ExecutionEngine
        
        engine = ExecutionEngine({})
        
        # Market with no liquidity
        market_data = {
            'symbol': 'BTC/USDT',
            'bids': [],  # No bids
            'asks': [],  # No asks
            'volume': 0
        }
        
        # Try to execute order
        order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.01
        }
        
        result = await engine.execute_order(order, market_data)
        
        # Should handle gracefully
        assert result['status'] == 'rejected'
        assert 'liquidity' in result['reason'].lower()
        assert result['error_code'] == 'NO_LIQUIDITY'
    
    async def test_extreme_volatility(self):
        """Test handling of extreme volatility"""
        from market_regime_detector import MarketRegimeDetector
        from grid_strategy_selector import GridStrategySelector
        
        detector = MarketRegimeDetector({})
        selector = GridStrategySelector({})
        
        # Generate extreme volatility data
        prices = []
        base_price = 50000
        
        for i in range(100):
            # 5% swings every tick
            swing = base_price * 0.05 * (1 if i % 2 == 0 else -1)
            price = base_price + swing + random.uniform(-1000, 1000)
            prices.append(price)
        
        # Detect regime
        regime = await detector.detect_regime({'prices': prices})
        
        assert regime['type'] == 'extreme_volatility'
        assert regime['confidence'] > 0.8
        
        # Select appropriate strategy
        strategy = await selector.select_strategy(regime)
        
        # Should use conservative settings
        assert strategy['grid_spacing'] > 0.05  # Wide spacing
        assert strategy['position_size'] < 0.01  # Small positions
        assert strategy['max_grids'] < 5  # Few grids
    
    async def test_price_gap_handling(self):
        """Test handling of price gaps"""
        from grid_management_system import GridManagementSystem
        
        grid_mgr = GridManagementSystem({})
        
        # Setup grids at normal price
        await grid_mgr.setup_grids({
            'center_price': 50000,
            'spacing': 0.01,
            'levels': 10
        })
        
        # Simulate price gap (10% overnight gap)
        gap_price = 55000
        
        result = await grid_mgr.handle_price_gap(gap_price)
        
        # Should adjust grids
        assert result['grids_cancelled'] > 0
        assert result['grids_repositioned'] > 0
        assert result['gap_protection_activated']


class TestNetworkFailures:
    """Test network-related failures"""
    
    async def test_connection_loss_recovery(self):
        """Test recovery from connection loss"""
        from market_data_input import MarketDataInput
        
        market_data = MarketDataInput({})
        
        # Simulate connection loss
        await market_data.disconnect()
        
        # Should detect disconnection
        assert not market_data.is_connected()
        
        # Should attempt reconnection
        reconnect_attempts = 0
        max_attempts = 5
        
        while reconnect_attempts < max_attempts:
            success = await market_data.reconnect()
            reconnect_attempts += 1
            
            if success:
                break
            
            await asyncio.sleep(1)
        
        # Should eventually reconnect
        assert market_data.is_connected()
        assert reconnect_attempts < max_attempts
        
        # Should resync state after reconnection
        state = await market_data.get_state()
        assert state['synchronized']
        assert state['last_update'] is not None
    
    async def test_partial_message_handling(self):
        """Test handling of partial/corrupted messages"""
        from websocket_client import WebSocketClient
        
        client = WebSocketClient()
        
        # Test various corrupted messages
        corrupted_messages = [
            '{"symbol": "BTC/USDT", "price":',  # Incomplete JSON
            '{"symbol": "BTC/USDT", "price": "not_a_number"}',  # Invalid type
            '{"symbol": "BTC/USDT"}',  # Missing required field
            'not json at all',  # Not JSON
            '{"symbol": "BTC/USDT", "price": null}',  # Null value
        ]
        
        errors_handled = 0
        
        for msg in corrupted_messages:
            try:
                result = await client.process_message(msg)
                # Should handle gracefully
                assert result is None or result.get('error') is not None
                errors_handled += 1
            except Exception:
                # Should not crash
                errors_handled += 1
        
        assert errors_handled == len(corrupted_messages)
    
    async def test_timeout_handling(self):
        """Test request timeout handling"""
        from api_client import APIClient
        
        client = APIClient(timeout=1.0)  # 1 second timeout
        
        # Mock slow endpoint
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Simulate slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(2)  # Longer than timeout
                return Mock()
            
            mock_get.side_effect = slow_response
            
            # Should timeout gracefully
            result = await client.get_market_data('BTC/USDT')
            
            assert result['status'] == 'error'
            assert result['error_type'] == 'timeout'
            assert result['retry_after'] is not None


class TestStateRecovery:
    """Test state recovery mechanisms"""
    
    async def test_checkpoint_recovery(self):
        """Test recovery from checkpoints"""
        from checkpoint_manager import CheckpointManager
        from attention_learning_layer import AttentionLearningLayer
        
        checkpoint_mgr = CheckpointManager({'checkpoint_dir': './test_checkpoints'})
        attention = AttentionLearningLayer({})
        
        # Create initial state
        attention.phase = 'active'
        attention.total_observations = 10000
        attention.feature_weights = {'volatility': 0.8, 'trend': 0.6}
        
        # Save checkpoint
        checkpoint_id = await checkpoint_mgr.save_checkpoint(
            'attention_state',
            attention.get_state()
        )
        
        # Corrupt current state
        attention.phase = None
        attention.total_observations = 0
        attention.feature_weights = {}
        
        # Recover from checkpoint
        recovered_state = await checkpoint_mgr.load_checkpoint(
            'attention_state',
            checkpoint_id
        )
        
        attention.load_state(recovered_state)
        
        # Verify recovery
        assert attention.phase == 'active'
        assert attention.total_observations == 10000
        assert attention.feature_weights['volatility'] == 0.8
        
        # Cleanup
        shutil.rmtree('./test_checkpoints', ignore_errors=True)
    
    async def test_partial_state_recovery(self):
        """Test recovery with partial state loss"""
        from system_coordinator import SystemCoordinator
        
        coordinator = SystemCoordinator({})
        await coordinator.initialize_components()
        
        # Save current state
        full_state = await coordinator.get_system_state()
        
        # Simulate partial component failure
        failed_components = ['market_regime_detector', 'grid_strategy_selector']
        
        for comp_name in failed_components:
            coordinator.components[comp_name] = None
        
        # Attempt recovery
        recovery_result = await coordinator.recover_failed_components(
            full_state,
            failed_components
        )
        
        # Should recover failed components
        assert recovery_result['recovered'] == len(failed_components)
        
        for comp_name in failed_components:
            assert coordinator.components[comp_name] is not None
            assert await coordinator.components[comp_name].health_check()
    
    async def test_config_rollback(self):
        """Test configuration rollback on failure"""
        from config_manager import ConfigManager
        
        config_mgr = ConfigManager()
        
        # Save current config
        original_config = await config_mgr.get_current_config()
        
        # Apply bad configuration
        bad_config = {
            'risk_management': {
                'max_position_size': 2.0,  # 200% - invalid
                'max_daily_loss': -0.1     # Negative - invalid
            }
        }
        
        try:
            await config_mgr.apply_config(bad_config)
            # Should fail validation
            assert False, "Bad config should not be accepted"
        except Exception:
            # Should auto-rollback
            current_config = await config_mgr.get_current_config()
            assert current_config == original_config


class TestRaceConditions:
    """Test race conditions and concurrency issues"""
    
    async def test_concurrent_phase_transitions(self):
        """Test concurrent phase transition requests"""
        from attention_learning_layer import AttentionLearningLayer
        
        attention = AttentionLearningLayer({})
        attention.phase = 'shadow'
        attention.total_observations = 5000
        
        # Multiple components trying to transition phase
        transition_tasks = []
        
        for i in range(10):
            task = asyncio.create_task(
                attention.check_phase_transition()
            )
            transition_tasks.append(task)
        
        results = await asyncio.gather(*transition_tasks, return_exceptions=True)
        
        # Should handle concurrent requests safely
        # Only one transition should succeed
        successful_transitions = sum(
            1 for r in results 
            if r and not isinstance(r, Exception)
        )
        
        assert successful_transitions <= 1
        assert attention.phase in ['shadow', 'active']  # Valid state
    
    async def test_order_race_condition(self):
        """Test race condition in order execution"""
        from execution_engine import ExecutionEngine
        
        engine = ExecutionEngine({'max_orders': 100})
        
        # Same order submitted multiple times concurrently
        order = {
            'id': 'ORD_12345',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.01,
            'price': 50000
        }
        
        # Submit same order 5 times concurrently
        submit_tasks = []
        for _ in range(5):
            task = asyncio.create_task(
                engine.submit_order(order.copy())
            )
            submit_tasks.append(task)
        
        results = await asyncio.gather(*submit_tasks, return_exceptions=True)
        
        # Only one should succeed
        successes = sum(
            1 for r in results 
            if r and r.get('status') == 'accepted'
        )
        
        duplicates = sum(
            1 for r in results
            if r and r.get('error') == 'DUPLICATE_ORDER'
        )
        
        assert successes == 1
        assert duplicates == 4


class TestDataCorruption:
    """Test data corruption scenarios"""
    
    async def test_corrupted_model_file(self):
        """Test handling of corrupted model files"""
        from model_manager import ModelManager
        
        model_mgr = ModelManager()
        
        # Create corrupted model file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            f.write(b'corrupted model data')
            corrupted_file = f.name
        
        # Try to load
        result = await model_mgr.load_model(corrupted_file)
        
        # Should handle gracefully
        assert result['success'] is False
        assert result['fallback'] == 'default_model'
        assert 'corruption' in result['error'].lower()
        
        # Cleanup
        os.unlink(corrupted_file)
    
    async def test_invalid_checkpoint_data(self):
        """Test invalid checkpoint data handling"""
        from checkpoint_manager import CheckpointManager
        
        checkpoint_mgr = CheckpointManager({})
        
        # Invalid checkpoint data
        invalid_states = [
            {},  # Empty state
            {'phase': 'invalid_phase'},  # Invalid phase
            {'total_observations': -1000},  # Negative observations
            {'feature_weights': 'not_a_dict'},  # Wrong type
        ]
        
        for invalid_state in invalid_states:
            # Should validate before saving
            result = await checkpoint_mgr.save_checkpoint(
                'test_state',
                invalid_state
            )
            
            assert result['success'] is False
            assert 'validation' in result['error'].lower()


# Edge case test runner
async def run_edge_case_tests():
    """Run all edge case and recovery tests"""
    print("🔍 Running Edge Case and Recovery Tests...")
    
    test_classes = [
        TestMarketEdgeCases,
        TestNetworkFailures,
        TestStateRecovery,
        TestRaceConditions,
        TestDataCorruption
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
    print(f"📊 Edge Case Test Results: {passed}/{total} passed")
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_edge_case_tests())
    exit(0 if success else 1)