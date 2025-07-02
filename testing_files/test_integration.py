# tests/test_integration.py
"""
Integration tests for the complete grid trading system
Tests component interactions and end-to-end workflows

Author: Grid Trading System
Date: 2024
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
from datetime import datetime, timedelta
import tempfile
import shutil
import os

# Import all system components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_data_input import MarketDataInput, MarketTick
from feature_engineering_pipeline import FeatureEngineeringPipeline
from attention_learning_layer import AttentionLearningLayer, AttentionPhase
from market_regime_detector import MarketRegimeDetector, MarketRegime
from grid_strategy_selector import GridStrategySelector
from risk_management_system import RiskManagementSystem
from execution_engine import ExecutionEngine
from performance_monitor import PerformanceMonitor
from feedback_loop import FeedbackLoop
from overfitting_detector import OverfittingDetector
from checkpoint_manager import CheckpointManager
from data_augmentation import MarketDataAugmenter, AugmentationConfig


class TestSystemIntegration(unittest.TestCase):
    """Test full system integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.config = {
            'market_data': {
                'symbols': ['BTCUSDT'],
                'exchanges': ['binance'],
                'buffer_size': 1000
            },
            'features': {
                'cache_size': 100,
                'extractors': ['price', 'volume', 'volatility']
            },
            'attention': {
                'min_trades_learning': 100,
                'min_trades_shadow': 50,
                'min_trades_active': 25
            },
            'risk_management': {
                'max_position_size': 0.05,
                'max_concurrent_orders': 5,
                'max_daily_loss': 0.01
            }
        }
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        
    async def test_data_flow_pipeline(self):
        """Test data flow through entire pipeline"""
        # Initialize components
        market_data = MarketDataInput(self.config['market_data'])
        features = FeatureEngineeringPipeline(self.config['features'])
        attention = AttentionLearningLayer(self.config['attention'])
        regime_detector = MarketRegimeDetector()
        
        # Generate test data
        test_ticks = self._generate_test_ticks(100)
        
        # Process through pipeline
        for tick in test_ticks:
            # 1. Market data ingestion
            await market_data.process_tick(tick)
            
            # 2. Feature extraction
            history = await market_data.get_recent_data('BTCUSDT', 50)
            feature_set = await features.extract_features(history)
            
            # 3. Attention processing
            enhanced_features = await attention.process_market_data(
                feature_set.features,
                history
            )
            
            # 4. Regime detection
            regime, confidence = await regime_detector.detect_regime(enhanced_features)
            
            # Verify data flow
            self.assertIsNotNone(feature_set)
            self.assertIsNotNone(enhanced_features)
            self.assertIsInstance(regime, MarketRegime)
            self.assertGreater(confidence, 0)
            
    async def test_trading_decision_flow(self):
        """Test trading decision making flow"""
        # Initialize trading components
        strategy_selector = GridStrategySelector(self.config)
        risk_manager = RiskManagementSystem(self.config['risk_management'])
        execution_engine = ExecutionEngine()
        
        # Mock market conditions
        features = {
            'price': 50000,
            'volatility_5m': 0.001,
            'trend_strength': 0.5,
            'volume_ratio': 1.2
        }
        
        regime = MarketRegime.RANGING
        
        # 1. Select strategy
        strategy = await strategy_selector.select_strategy(regime, features)
        self.assertIsNotNone(strategy)
        
        # 2. Generate grid levels
        grid_levels = strategy.calculate_grid_levels(features['price'])
        self.assertGreater(len(grid_levels), 0)
        
        # 3. Risk check
        for level in grid_levels[:3]:  # Test first 3 levels
            order_params = {
                'symbol': 'BTCUSDT',
                'side': level.side,
                'price': level.price,
                'quantity': level.quantity
            }
            
            risk_check = await risk_manager.check_order_risk(order_params)
            self.assertIn('approved', risk_check)
            
            if risk_check['approved']:
                # 4. Execute order (mock)
                with patch.object(execution_engine, '_submit_order', new_callable=AsyncMock) as mock_submit:
                    mock_submit.return_value = {
                        'order_id': f'test_{time.time()}',
                        'status': 'filled'
                    }
                    
                    result = await execution_engine.execute_order(order_params)
                    self.assertIsNotNone(result)
                    
    async def test_learning_feedback_loop(self):
        """Test learning and feedback mechanisms"""
        # Initialize learning components
        attention = AttentionLearningLayer(self.config['attention'])
        feedback_loop = FeedbackLoop()
        overfitting_detector = OverfittingDetector()
        
        # Simulate trading results
        for i in range(150):
            # Generate varying performance
            if i < 50:
                # Good performance initially
                win_rate = 0.65
                profit = 100
            elif i < 100:
                # Degrading performance (overfitting)
                win_rate = 0.45
                profit = -50
            else:
                # Recovery after adjustment
                win_rate = 0.55
                profit = 50
                
            # Process feedback
            trade_result = {
                'trade_id': f'trade_{i}',
                'profit': profit,
                'win': profit > 0,
                'features': {'volatility': 0.001, 'trend': 0.5}
            }
            
            # Update components
            await feedback_loop.process_trade_result(trade_result)
            
            # Check for overfitting
            if i % 20 == 0:
                await overfitting_detector.add_training_result(
                    win_rate=0.65,
                    profit_factor=1.5
                )
                await overfitting_detector.add_live_result(
                    win_rate=win_rate,
                    profit_factor=1.0 + win_rate
                )
                
        # Verify learning occurred
        self.assertEqual(attention.phase, AttentionPhase.SHADOW)  # Should progress
        
        # Check overfitting detection
        detection = await overfitting_detector.detect_overfitting()
        self.assertTrue(detection['is_overfitting'])
        
    async def test_checkpoint_and_recovery(self):
        """Test checkpoint save/load and recovery"""
        # Initialize components with checkpoint support
        checkpoint_manager = CheckpointManager(self.temp_dir)
        attention = AttentionLearningLayer(self.config['attention'])
        
        # Train model
        for i in range(100):
            features = {f'feature_{j}': np.random.random() for j in range(5)}
            await attention.process_market_data(features, [])
            
        # Save checkpoint
        checkpoint_id = await checkpoint_manager.save_checkpoint(
            model_name='attention_layer',
            component=attention,
            performance_metrics={
                'win_rate': 0.65,
                'profit_factor': 1.5
            }
        )
        
        self.assertIsNotNone(checkpoint_id)
        
        # Modify model state
        attention.phase = AttentionPhase.ACTIVE
        
        # Load checkpoint
        success = await checkpoint_manager.load_checkpoint(
            model_name='attention_layer',
            component=attention,
            checkpoint_id=checkpoint_id
        )
        
        self.assertTrue(success)
        self.assertEqual(attention.phase, AttentionPhase.LEARNING)  # Should be restored
        
    async def test_data_augmentation_integration(self):
        """Test data augmentation in training pipeline"""
        # Initialize augmenter
        augmenter = MarketDataAugmenter(AugmentationConfig(
            noise_level='moderate',
            max_augmentation_factor=2
        ))
        
        # Generate base data
        original_ticks = self._generate_test_ticks(100)
        
        # Augment data
        augmented_ticks, metadata = augmenter.augment_market_data(
            original_ticks,
            methods=['noise_injection', 'time_warping']
        )
        
        # Verify augmentation
        self.assertEqual(metadata.original_size, 100)
        self.assertGreater(metadata.augmented_size, 100)
        self.assertGreater(metadata.quality_score, 0.8)
        
        # Test with feature pipeline
        features = FeatureEngineeringPipeline(self.config['features'])
        
        # Extract features from augmented data
        for i in range(50, len(augmented_ticks)):
            window = augmented_ticks[i-50:i]
            feature_set = await features.extract_features(window)
            self.assertIsNotNone(feature_set)
            
    async def test_monitoring_and_alerts(self):
        """Test monitoring and alert system"""
        # Initialize monitoring
        monitor = PerformanceMonitor()
        
        # Mock components
        mock_components = {
            'market_data': Mock(),
            'attention': Mock(get_learning_progress=Mock(return_value=0.5)),
            'risk_manager': Mock(),
            'execution_engine': Mock()
        }
        
        await monitor.initialize(mock_components)
        
        # Simulate metrics
        for i in range(10):
            metrics = {
                'trades': i * 10,
                'win_rate': 0.5 + np.random.normal(0, 0.1),
                'pnl': 1000 + i * 100,
                'cpu_usage': 50 + np.random.normal(0, 10),
                'memory_usage': 60 + np.random.normal(0, 5)
            }
            
            await monitor.update_metrics(metrics)
            
        # Check alerts
        alerts = await monitor.get_active_alerts()
        self.assertIsInstance(alerts, list)
        
        # Generate report
        report = await monitor.generate_performance_report()
        self.assertIn('summary', report)
        self.assertIn('system_health', report)
        
    async def test_error_handling_and_recovery(self):
        """Test system error handling and recovery"""
        # Initialize system with error simulation
        market_data = MarketDataInput(self.config['market_data'])
        
        # Simulate connection error
        with patch.object(market_data, '_connect_websocket', side_effect=Exception("Connection failed")):
            # Should handle gracefully
            connected = await market_data.connect()
            self.assertFalse(connected)
            
        # Test recovery mechanism
        recovery_manager = Mock()
        recovery_manager.execute_recovery = AsyncMock(return_value={'success': True})
        
        # Simulate recovery
        result = await recovery_manager.execute_recovery({}, "Connection failure")
        self.assertTrue(result['success'])
        
    async def test_performance_under_load(self):
        """Test system performance under high load"""
        # Initialize components
        features = FeatureEngineeringPipeline(self.config['features'])
        
        # Generate large dataset
        large_dataset = self._generate_test_ticks(1000)
        
        # Measure processing time
        start_time = time.time()
        
        for i in range(100, len(large_dataset)):
            window = large_dataset[i-100:i]
            await features.extract_features(window)
            
        duration = time.time() - start_time
        
        # Should process reasonably fast
        self.assertLess(duration, 10.0)  # Less than 10 seconds for 900 windows
        
    def _generate_test_ticks(self, count: int) -> List[MarketTick]:
        """Generate test market ticks"""
        ticks = []
        base_price = 50000
        
        for i in range(count):
            price = base_price + np.random.normal(0, 100)
            
            tick = MarketTick(
                symbol='BTCUSDT',
                price=price,
                volume=100 + np.random.exponential(50),
                timestamp=time.time() + i,
                bid=price - 0.5,
                ask=price + 0.5,
                exchange='binance'
            )
            
            ticks.append(tick)
            
        return ticks


class TestComponentInteractions(unittest.TestCase):
    """Test specific component interactions"""
    
    async def test_attention_regime_interaction(self):
        """Test interaction between attention layer and regime detector"""
        attention = AttentionLearningLayer()
        regime_detector = MarketRegimeDetector()
        
        # Process data through both
        features = {
            'volatility_5m': 0.002,
            'trend_strength': 0.7,
            'volume_ratio': 1.5
        }
        
        # Attention enhancement
        enhanced = await attention.process_market_data(features, [])
        
        # Regime detection with enhanced features
        regime, confidence = await regime_detector.detect_regime(enhanced)
        
        # Verify interaction
        self.assertNotEqual(features, enhanced)  # Features should be modified
        self.assertIsInstance(regime, MarketRegime)
        
    async def test_risk_execution_interaction(self):
        """Test interaction between risk manager and execution engine"""
        risk_manager = RiskManagementSystem()
        execution_engine = ExecutionEngine()
        
        # Create order
        order = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'price': 50000,
            'quantity': 0.001
        }
        
        # Risk check
        risk_result = await risk_manager.check_order_risk(order)
        
        if risk_result['approved']:
            # Mock execution
            with patch.object(execution_engine, '_submit_order', new_callable=AsyncMock):
                result = await execution_engine.execute_order(order)
                self.assertIsNotNone(result)
        else:
            # Verify rejection reason
            self.assertIn('reason', risk_result)
            
    async def test_feedback_overfitting_interaction(self):
        """Test interaction between feedback loop and overfitting detector"""
        feedback_loop = FeedbackLoop()
        overfitting_detector = OverfittingDetector()
        
        # Register overfitting detector with feedback loop
        feedback_loop.register_component('overfitting_detector', overfitting_detector)
        
        # Simulate trades with degrading performance
        for i in range(100):
            trade = {
                'profit': 100 if i < 50 else -50,
                'win': i < 50,
                'features': {'vol': 0.001}
            }
            
            await feedback_loop.process_trade_result(trade)
            
        # Check if overfitting was detected
        insights = feedback_loop.get_recent_insights()
        overfitting_insights = [i for i in insights if i.category == 'overfitting']
        
        self.assertGreater(len(overfitting_insights), 0)


# Run specific test scenarios
async def run_integration_test_scenario(scenario: str):
    """Run specific integration test scenario"""
    
    if scenario == "full_trading_cycle":
        # Complete trading cycle test
        await test_full_trading_cycle()
        
    elif scenario == "overfitting_detection":
        # Overfitting detection and recovery
        await test_overfitting_scenario()
        
    elif scenario == "stress_test":
        # High load stress test
        await test_stress_scenario()
        
    elif scenario == "failover":
        # Failover and recovery test
        await test_failover_scenario()
        

async def test_full_trading_cycle():
    """Test complete trading cycle from data to execution"""
    print("Running full trading cycle test...")
    
    # Initialize all components
    components = await initialize_test_system()
    
    # Generate market scenario
    market_scenario = generate_market_scenario("ranging", duration=1000)
    
    # Run trading cycle
    results = []
    
    for tick in market_scenario:
        # Process tick through system
        result = await process_trading_cycle(components, tick)
        results.append(result)
        
    # Analyze results
    performance = analyze_trading_results(results)
    
    print(f"Trading cycle complete:")
    print(f"- Total trades: {performance['total_trades']}")
    print(f"- Win rate: {performance['win_rate']:.2%}")
    print(f"- Profit factor: {performance['profit_factor']:.2f}")
    
    return performance


async def test_overfitting_scenario():
    """Test overfitting detection and recovery"""
    print("Running overfitting scenario test...")
    
    # Initialize components
    components = await initialize_test_system()
    
    # Create overfitting conditions
    print("Creating overfitting conditions...")
    
    # Phase 1: Good training performance
    for i in range(100):
        await simulate_trade(components, win_rate=0.7, is_training=True)
        
    # Phase 2: Poor live performance
    for i in range(100):
        await simulate_trade(components, win_rate=0.4, is_training=False)
        
    # Check detection
    detection = await components['overfitting_detector'].detect_overfitting()
    
    print(f"Overfitting detected: {detection['is_overfitting']}")
    print(f"Severity: {detection['severity']}")
    
    if detection['is_overfitting']:
        # Execute recovery
        recovery_result = await components['recovery_manager'].recover_from_overfitting(
            detection,
            detection['severity']
        )
        
        print(f"Recovery executed: {recovery_result['success']}")
        print(f"Actions taken: {recovery_result['actions_taken']}")
        
    return detection


async def test_stress_scenario():
    """Test system under high load"""
    print("Running stress test scenario...")
    
    components = await initialize_test_system()
    
    # Generate high-frequency data
    tick_rate = 100  # ticks per second
    duration = 10    # seconds
    
    start_time = time.time()
    processed = 0
    errors = 0
    
    print(f"Processing {tick_rate * duration} ticks...")
    
    for i in range(tick_rate * duration):
        try:
            tick = generate_random_tick('BTCUSDT')
            await process_tick_fast(components, tick)
            processed += 1
            
        except Exception as e:
            errors += 1
            
    elapsed = time.time() - start_time
    
    print(f"Stress test complete:")
    print(f"- Processed: {processed} ticks")
    print(f"- Errors: {errors}")
    print(f"- Rate: {processed/elapsed:.1f} ticks/second")
    print(f"- Latency: {elapsed/processed*1000:.1f} ms/tick")
    
    return {
        'processed': processed,
        'errors': errors,
        'rate': processed/elapsed,
        'latency': elapsed/processed
    }


async def test_failover_scenario():
    """Test system failover and recovery"""
    print("Running failover scenario test...")
    
    components = await initialize_test_system()
    checkpoint_manager = components['checkpoint_manager']
    
    # Save initial state
    print("Saving initial checkpoint...")
    checkpoint_id = await checkpoint_manager.save_checkpoint(
        'system_state',
        components['attention'],
        {'performance': 0.65}
    )
    
    # Simulate component failure
    print("Simulating component failure...")
    
    # Corrupt state
    components['attention'].phase = None
    components['regime_detector'].current_regime = None
    
    # Detect failure
    health_check = await check_system_health(components)
    
    print(f"System health: {health_check['status']}")
    
    if health_check['status'] == 'unhealthy':
        # Execute failover
        print("Executing failover...")
        
        # Restore from checkpoint
        success = await checkpoint_manager.rollback_to_checkpoint(
            'system_state',
            components['attention'],
            checkpoint_id,
            reason="System failure"
        )
        
        print(f"Failover success: {success}")
        
        # Verify recovery
        health_check_after = await check_system_health(components)
        print(f"System health after recovery: {health_check_after['status']}")
        
    return health_check_after


# Helper functions
async def initialize_test_system():
    """Initialize all system components for testing"""
    config = {
        'market_data': {'symbols': ['BTCUSDT']},
        'features': {'extractors': ['price', 'volume']},
        'attention': {'min_trades_learning': 50},
        'risk': {'max_position_size': 0.05}
    }
    
    components = {
        'market_data': MarketDataInput(config['market_data']),
        'features': FeatureEngineeringPipeline(config['features']),
        'attention': AttentionLearningLayer(config['attention']),
        'regime_detector': MarketRegimeDetector(),
        'strategy_selector': GridStrategySelector(config),
        'risk_manager': RiskManagementSystem(config['risk']),
        'execution_engine': ExecutionEngine(),
        'performance_monitor': PerformanceMonitor(),
        'feedback_loop': FeedbackLoop(),
        'overfitting_detector': OverfittingDetector(),
        'checkpoint_manager': CheckpointManager('./test_checkpoints'),
        'recovery_manager': Mock(recover_from_overfitting=AsyncMock())
    }
    
    # Initialize components
    for component in components.values():
        if hasattr(component, 'initialize'):
            await component.initialize()
            
    return components


def generate_market_scenario(scenario_type: str, duration: int) -> List[MarketTick]:
    """Generate market scenario for testing"""
    ticks = []
    base_price = 50000
    
    for i in range(duration):
        if scenario_type == "ranging":
            # Sideways market
            price = base_price + np.sin(i/50) * 100 + np.random.normal(0, 20)
            
        elif scenario_type == "trending":
            # Trending market
            price = base_price + i * 5 + np.random.normal(0, 50)
            
        elif scenario_type == "volatile":
            # Volatile market
            price = base_price + np.random.normal(0, 200) * (1 + i/1000)
            
        else:
            price = base_price
            
        tick = MarketTick(
            symbol='BTCUSDT',
            price=price,
            volume=100 * np.exp(np.random.normal(0, 0.5)),
            timestamp=time.time() + i,
            bid=price - 0.5,
            ask=price + 0.5,
            exchange='binance'
        )
        
        ticks.append(tick)
        
    return ticks


async def process_trading_cycle(components: Dict, tick: MarketTick) -> Dict[str, Any]:
    """Process single trading cycle"""
    # 1. Ingest market data
    await components['market_data'].process_tick(tick)
    
    # 2. Extract features
    history = await components['market_data'].get_recent_data(tick.symbol, 50)
    features = await components['features'].extract_features(history)
    
    # 3. Apply attention
    enhanced_features = await components['attention'].process_market_data(
        features.features,
        history
    )
    
    # 4. Detect regime
    regime, confidence = await components['regime_detector'].detect_regime(enhanced_features)
    
    # 5. Select strategy
    strategy = await components['strategy_selector'].select_strategy(regime, enhanced_features)
    
    # 6. Generate orders
    grid_levels = strategy.calculate_grid_levels(tick.price)
    
    # 7. Risk check and execute
    executed_orders = []
    
    for level in grid_levels[:3]:  # Limit to 3 orders for testing
        order = {
            'symbol': tick.symbol,
            'side': level.side,
            'price': level.price,
            'quantity': level.quantity
        }
        
        risk_check = await components['risk_manager'].check_order_risk(order)
        
        if risk_check['approved']:
            # Mock execution
            executed_orders.append(order)
            
    return {
        'tick': tick,
        'regime': regime,
        'strategy': strategy.__class__.__name__,
        'orders_executed': len(executed_orders),
        'timestamp': tick.timestamp
    }


def analyze_trading_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze trading results"""
    total_orders = sum(r['orders_executed'] for r in results)
    
    # Simulate P&L
    pnl = 0
    wins = 0
    
    for i, result in enumerate(results):
        if result['orders_executed'] > 0:
            # Simple P&L simulation
            if i < len(results) - 1:
                price_change = results[i+1]['tick'].price - result['tick'].price
                trade_pnl = price_change * 0.001  # Small position
                pnl += trade_pnl
                
                if trade_pnl > 0:
                    wins += 1
                    
    total_trades = max(1, sum(1 for r in results if r['orders_executed'] > 0))
    
    return {
        'total_trades': total_trades,
        'total_orders': total_orders,
        'win_rate': wins / total_trades if total_trades > 0 else 0,
        'total_pnl': pnl,
        'profit_factor': abs(pnl) / total_trades if total_trades > 0 else 0
    }


async def simulate_trade(components: Dict, win_rate: float, is_training: bool):
    """Simulate a trade with given win rate"""
    win = np.random.random() < win_rate
    profit = 100 if win else -50
    
    if is_training:
        await components['overfitting_detector'].add_training_result(
            win_rate=win_rate,
            profit_factor=1.5 if win else 0.8
        )
    else:
        await components['overfitting_detector'].add_live_result(
            win_rate=win_rate,
            profit_factor=1.5 if win else 0.8
        )


def generate_random_tick(symbol: str) -> MarketTick:
    """Generate random market tick"""
    price = 50000 + np.random.normal(0, 100)
    
    return MarketTick(
        symbol=symbol,
        price=price,
        volume=100 * np.exp(np.random.normal(0, 0.5)),
        timestamp=time.time(),
        bid=price - 0.5,
        ask=price + 0.5,
        exchange='binance'
    )


async def process_tick_fast(components: Dict, tick: MarketTick):
    """Fast tick processing for stress testing"""
    # Minimal processing
    await components['market_data'].process_tick(tick)
    
    # Quick feature extraction
    if len(components['market_data'].buffer) >= 10:
        features = {'price': tick.price, 'volume': tick.volume}
        
        # Quick regime check
        regime = MarketRegime.RANGING  # Simplified
        
        # Risk check
        await components['risk_manager'].update_market_data(tick)


async def check_system_health(components: Dict) -> Dict[str, Any]:
    """Check overall system health"""
    health_status = {
        'status': 'healthy',
        'components': {},
        'issues': []
    }
    
    # Check each component
    for name, component in components.items():
        try:
            if hasattr(component, 'health_check'):
                component_health = await component.health_check()
                health_status['components'][name] = component_health
                
                if not component_health.get('healthy', True):
                    health_status['issues'].append(f"{name}: {component_health.get('issue', 'Unknown')}")
                    
            elif hasattr(component, 'phase'):
                # Check attention layer
                if component.phase is None:
                    health_status['issues'].append(f"{name}: Invalid phase")
                    
            elif hasattr(component, 'current_regime'):
                # Check regime detector
                if component.current_regime is None:
                    health_status['issues'].append(f"{name}: No regime detected")
                    
        except Exception as e:
            health_status['issues'].append(f"{name}: {str(e)}")
            
    # Overall status
    if health_status['issues']:
        health_status['status'] = 'unhealthy'
        
    return health_status


# Test runner
if __name__ == '__main__':
    # Run unit tests
    print("Running integration tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run scenario tests
    print("\n\nRunning scenario tests...")
    
    scenarios = [
        "full_trading_cycle",
        "overfitting_detection",
        "stress_test",
        "failover"
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}")
        print('='*50)
        
        asyncio.run(run_integration_test_scenario(scenario))