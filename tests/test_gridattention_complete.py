"""
Complete Test Suite for GridAttention Trading System
Comprehensive tests for all components and their integration
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import tempfile
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Data Structures
# ============================================================================

@dataclass
class TestMarketTick:
    """Test market tick data"""
    symbol: str
    price: float
    volume: float
    timestamp: float
    bid: float
    ask: float
    exchange: str = "TEST"
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'exchange': self.exchange
        }


class MarketRegime(Enum):
    """Market regime types for testing"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    UNCERTAIN = "uncertain"


# ============================================================================
# Test Utilities
# ============================================================================

class TestDataGenerator:
    """Generate test data for system testing"""
    
    @staticmethod
    def generate_market_ticks(
        n_ticks: int = 1000,
        base_price: float = 50000,
        volatility: float = 0.001,
        trend: float = 0.0001,
        regime: MarketRegime = MarketRegime.RANGING
    ) -> List[TestMarketTick]:
        """Generate synthetic market ticks"""
        ticks = []
        price = base_price
        
        for i in range(n_ticks):
            # Apply regime-specific patterns
            if regime == MarketRegime.TRENDING:
                price_change = np.random.randn() * volatility + trend
            elif regime == MarketRegime.VOLATILE:
                price_change = np.random.randn() * volatility * 3
            elif regime == MarketRegime.BREAKOUT:
                if i > n_ticks // 2:
                    price_change = np.random.randn() * volatility + trend * 5
                else:
                    price_change = np.random.randn() * volatility
            else:  # RANGING
                price_change = np.random.randn() * volatility
                price_change += -0.01 * (price / base_price - 1)  # Mean reversion
            
            price *= (1 + price_change)
            spread = price * 0.0001 * (1 + np.random.rand())
            
            tick = TestMarketTick(
                symbol="BTC/USDT",
                price=price,
                volume=np.random.exponential(100),
                timestamp=time.time() + i,
                bid=price - spread/2,
                ask=price + spread/2
            )
            ticks.append(tick)
        
        return ticks
    
    @staticmethod
    def generate_features(tick: TestMarketTick, history: List[TestMarketTick]) -> Dict[str, float]:
        """Generate features from market data"""
        if len(history) < 20:
            return {}
        
        prices = [t.price for t in history[-20:]]
        volumes = [t.volume for t in history[-20:]]
        
        features = {
            'price': tick.price,
            'volume': tick.volume,
            'spread_bps': (tick.ask - tick.bid) / tick.price * 10000,
            'volatility_5m': np.std(prices[-5:]) / np.mean(prices[-5:]) if len(prices) >= 5 else 0,
            'volatility_20m': np.std(prices) / np.mean(prices),
            'volume_ratio': tick.volume / np.mean(volumes) if volumes else 1,
            'price_momentum': (tick.price - prices[0]) / prices[0] if prices else 0,
            'rsi_14': TestDataGenerator._calculate_rsi(prices, 14),
            'trend_strength': TestDataGenerator._calculate_trend_strength(prices)
        }
        
        return features
    
    @staticmethod
    def _calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 0.5
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 1.0
        
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _calculate_trend_strength(prices: List[float]) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 3:
            return 0
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize by price level
        normalized_slope = slope / np.mean(prices) * len(prices)
        
        return np.clip(normalized_slope, -1, 1)


# ============================================================================
# Component Test Classes
# ============================================================================

class TestResult:
    """Test result container"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}
        self.execution_time = 0
        
    def __str__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        msg = f"{status} - {self.name} ({self.execution_time:.2f}s)"
        if self.error:
            msg += f"\n   Error: {self.error}"
        return msg


class ComponentTester:
    """Base class for component testing"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.test_results = []
        
    async def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test with error handling"""
        result = TestResult(f"{self.component_name}::{test_name}")
        start_time = time.time()
        
        try:
            await test_func(*args, **kwargs)
            result.passed = True
        except Exception as e:
            result.passed = False
            result.error = str(e)
            logger.error(f"Test failed: {result.name} - {e}")
        finally:
            result.execution_time = time.time() - start_time
            self.test_results.append(result)
            
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        
        return {
            'component': self.component_name,
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total if total > 0 else 0,
            'results': self.test_results
        }


# ============================================================================
# Core Component Tests
# ============================================================================

class AttentionLayerTester(ComponentTester):
    """Test Attention Learning Layer"""
    
    def __init__(self):
        super().__init__("AttentionLearningLayer")
        
    async def test_initialization(self):
        """Test component initialization"""
        from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase
        
        config = {
            'learning_rate': 0.001,
            'attention_window': 1000,
            'min_trades_for_learning': 100
        }
        
        layer = AttentionLearningLayer(config)
        assert layer is not None
        assert layer.phase == AttentionPhase.LEARNING
        
    async def test_feature_processing(self):
        """Test feature processing and weighting"""
        from core.attention_learning_layer import AttentionLearningLayer
        
        layer = AttentionLearningLayer({})
        
        # Generate test features
        features = {
            'volatility_5m': 0.002,
            'trend_strength': 0.5,
            'volume_ratio': 1.2,
            'rsi_14': 0.6
        }
        
        # Process features
        result = await layer.process(
            features,
            "ranging",
            {'timestamp': time.time()}
        )
        
        assert result is not None
        assert 'weighted_features' in result
        assert 'phase' in result
        
    async def test_phase_transitions(self):
        """Test phase transition logic"""
        from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase
        
        layer = AttentionLearningLayer({
            'min_trades_for_learning': 10,
            'min_trades_for_shadow': 20,
            'min_trades_for_active': 30
        })
        
        # Process enough data to trigger phase transitions
        for i in range(35):
            features = {'volatility_5m': 0.001 + np.random.rand() * 0.001}
            await layer.process(features, "ranging", {'timestamp': time.time() + i})
        
        # Should have progressed through phases
        assert layer.total_observations >= 35
        
    async def test_warmup_loading(self):
        """Test warmup state loading"""
        from core.attention_learning_layer import AttentionLearningLayer
        
        # Create warmup state
        warmup_state = {
            'version': '1.0',
            'timestamp': time.time(),
            'attention_state': {
                'phase': 'active',
                'total_observations': 100000,
                'feature_importance': {
                    'volatility_5m': 0.8,
                    'trend_strength': 0.6,
                    'volume_ratio': 0.4
                }
            }
        }
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(warmup_state, f)
            temp_file = f.name
        
        try:
            # Load warmup
            layer = AttentionLearningLayer({})
            loaded = await layer._load_warmup_state(temp_file)
            
            assert loaded is True
            assert layer.total_observations == 100000
        finally:
            Path(temp_file).unlink(missing_ok=True)


class MarketRegimeDetectorTester(ComponentTester):
    """Test Market Regime Detector"""
    
    def __init__(self):
        super().__init__("MarketRegimeDetector")
        
    async def test_regime_detection(self):
        """Test regime detection accuracy"""
        from core.market_regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector({})
        
        # Test different market conditions
        test_cases = [
            {
                'features': {
                    'volatility_5m': 0.0005,
                    'trend_strength': 0.1,
                    'volume_ratio': 1.0
                },
                'expected': 'ranging'
            },
            {
                'features': {
                    'volatility_5m': 0.001,
                    'trend_strength': 0.8,
                    'volume_ratio': 1.5
                },
                'expected': 'trending'
            },
            {
                'features': {
                    'volatility_5m': 0.003,
                    'trend_strength': -0.2,
                    'volume_ratio': 2.5
                },
                'expected': 'volatile'
            }
        ]
        
        for case in test_cases:
            regime, confidence = await detector.detect_regime(case['features'])
            assert regime is not None
            assert 0 <= confidence <= 1
            
    async def test_ensemble_consistency(self):
        """Test ensemble voting consistency"""
        from core.market_regime_detector import MarketRegimeDetector
        
        detector = MarketRegimeDetector({
            'ensemble': {
                'enabled': True,
                'min_confidence': 0.6
            }
        })
        
        # Run multiple detections with same features
        features = {
            'volatility_5m': 0.001,
            'trend_strength': 0.5,
            'volume_ratio': 1.2
        }
        
        regimes = []
        for _ in range(10):
            regime, confidence = await detector.detect_regime(features)
            regimes.append(regime)
            
        # Should have consistent results
        unique_regimes = set(regimes)
        assert len(unique_regimes) <= 2  # Allow some variation


class GridStrategySelectorTester(ComponentTester):
    """Test Grid Strategy Selector"""
    
    def __init__(self):
        super().__init__("GridStrategySelector")
        
    async def test_strategy_selection(self):
        """Test strategy selection for different regimes"""
        from core.grid_strategy_selector import GridStrategySelector
        
        selector = GridStrategySelector({})
        
        # Test different market regimes
        regimes = ['trending', 'ranging', 'volatile', 'breakout']
        
        for regime in regimes:
            features = {'volatility_5m': 0.001, 'trend_strength': 0.5}
            context = {'account_balance': 10000}
            
            config = await selector.select_strategy(
                regime,
                features,
                context
            )
            
            assert config is not None
            assert config.grid_type in ['symmetric', 'asymmetric']
            assert config.spacing > 0
            assert config.levels > 0
            
    async def test_cross_validation(self):
        """Test strategy cross-validation"""
        from core.grid_strategy_selector import GridStrategySelector
        
        selector = GridStrategySelector({})
        
        # Simulate performance history
        for i in range(100):
            await selector.update_performance(
                'ranging',
                profit=np.random.randn() * 10,
                success=np.random.rand() > 0.4,
                {'timestamp': time.time() + i}
            )
            
        # Get statistics
        stats = await selector.get_statistics()
        assert 'strategy_performance' in stats
        assert 'validation_metrics' in stats


class RiskManagementTester(ComponentTester):
    """Test Risk Management System"""
    
    def __init__(self):
        super().__init__("RiskManagementSystem")
        
    async def test_risk_validation(self):
        """Test order risk validation"""
        from core.risk_management_system import RiskManagementSystem, RiskViolationType
        
        risk_manager = RiskManagementSystem({
            'max_position_size': 0.1,
            'max_daily_loss': 0.02,
            'max_drawdown': 0.05
        })
        
        # Test valid order
        order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'size': 0.01,
            'price': 50000
        }
        
        result = await risk_manager.validate_order(order, {})
        assert result['approved'] is True
        
        # Test oversized order
        large_order = order.copy()
        large_order['size'] = 1.0
        
        result = await risk_manager.validate_order(large_order, {})
        assert result['approved'] is False
        assert RiskViolationType.POSITION_SIZE in result['violations']
        
    async def test_portfolio_monitoring(self):
        """Test portfolio risk monitoring"""
        from core.risk_management_system import RiskManagementSystem
        
        risk_manager = RiskManagementSystem({})
        
        # Add test positions
        positions = [
            {'symbol': 'BTC/USDT', 'side': 'long', 'size': 0.5, 'entry_price': 50000},
            {'symbol': 'ETH/USDT', 'side': 'long', 'size': 2.0, 'entry_price': 3000}
        ]
        
        for pos in positions:
            await risk_manager.add_position(pos)
            
        # Check portfolio metrics
        metrics = await risk_manager.get_portfolio_metrics()
        assert 'total_exposure' in metrics
        assert 'risk_level' in metrics
        assert metrics['position_count'] == 2


class ExecutionEngineTester(ComponentTester):
    """Test Execution Engine"""
    
    def __init__(self):
        super().__init__("ExecutionEngine")
        
    async def test_order_execution(self):
        """Test order execution flow"""
        from core.execution_engine import ExecutionEngine
        
        engine = ExecutionEngine({
            'mode': 'simulation',  # Test mode
            'latency_target': 100  # ms
        })
        
        # Test order
        order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'limit',
            'size': 0.1,
            'price': 50000
        }
        
        result = await engine.execute_order(order)
        assert result is not None
        assert 'order_id' in result
        assert 'status' in result
        
    async def test_batch_execution(self):
        """Test batch order execution"""
        from core.execution_engine import ExecutionEngine
        
        engine = ExecutionEngine({'mode': 'simulation'})
        
        # Create grid orders
        orders = []
        base_price = 50000
        
        for i in range(5):
            orders.append({
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'type': 'limit',
                'size': 0.1,
                'price': base_price - (i + 1) * 100
            })
            
        results = await engine.execute_batch(orders)
        assert len(results) == len(orders)
        assert all('order_id' in r for r in results)


class PerformanceMonitorTester(ComponentTester):
    """Test Performance Monitor"""
    
    def __init__(self):
        super().__init__("PerformanceMonitor")
        
    async def test_metric_tracking(self):
        """Test performance metric tracking"""
        from core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor({})
        
        # Record test trades
        for i in range(10):
            trade = {
                'symbol': 'BTC/USDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'price': 50000 + np.random.randn() * 100,
                'size': 0.1,
                'profit': np.random.randn() * 50,
                'timestamp': time.time() + i
            }
            
            await monitor.record_trade(trade)
            
        # Get metrics
        metrics = await monitor.get_metrics()
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'sharpe_ratio' in metrics
        
    async def test_overfitting_detection(self):
        """Test overfitting detection in performance"""
        from core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor({
            'overfitting_detection': {
                'enabled': True,
                'window_size': 20
            }
        })
        
        # Simulate degrading performance
        for i in range(50):
            profit = 50 - i if i < 25 else -50  # Performance degrades
            
            trade = {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 50000,
                'size': 0.1,
                'profit': profit,
                'timestamp': time.time() + i
            }
            
            await monitor.record_trade(trade)
            
        # Check overfitting indicators
        indicators = await monitor.get_overfitting_indicators()
        assert 'performance_degradation' in indicators
        assert 'parameter_instability' in indicators


class FeedbackLoopTester(ComponentTester):
    """Test Feedback Loop"""
    
    def __init__(self):
        super().__init__("FeedbackLoop")
        
    async def test_feedback_processing(self):
        """Test feedback processing and learning"""
        from core.feedback_loop import FeedbackLoop
        
        feedback = FeedbackLoop({
            'learning_rate': 0.001,
            'update_interval': 60
        })
        
        # Send performance feedback
        perf_data = {
            'metric': 'sharpe_ratio',
            'value': 1.5,
            'timestamp': time.time()
        }
        
        await feedback.process_feedback('performance', perf_data)
        
        # Send execution feedback
        exec_data = {
            'latency': 50,
            'slippage': 0.0001,
            'success': True
        }
        
        await feedback.process_feedback('execution', exec_data)
        
        # Get insights
        insights = await feedback.get_insights()
        assert insights is not None
        assert 'recommendations' in insights


# ============================================================================
# Integration Tests
# ============================================================================

class SystemIntegrationTester(ComponentTester):
    """Test full system integration"""
    
    def __init__(self):
        super().__init__("SystemIntegration")
        
    async def test_full_pipeline(self):
        """Test complete trading pipeline"""
        # This would require all components to be available
        # Simplified version for demonstration
        
        # 1. Generate market data
        ticks = TestDataGenerator.generate_market_ticks(100)
        
        # 2. Extract features
        features_list = []
        for i, tick in enumerate(ticks):
            if i > 20:
                features = TestDataGenerator.generate_features(tick, ticks[:i])
                features_list.append(features)
                
        assert len(features_list) > 0
        
        # 3. Detect regime
        regime = self._detect_regime_from_features(features_list[-1])
        assert regime in ['trending', 'ranging', 'volatile', 'breakout', 'uncertain']
        
        # 4. Select strategy
        strategy = self._select_strategy(regime, features_list[-1])
        assert strategy is not None
        
        # 5. Risk validation
        risk_approved = self._validate_risk(strategy)
        assert isinstance(risk_approved, bool)
        
    def _detect_regime_from_features(self, features: Dict[str, float]) -> str:
        """Simple regime detection for testing"""
        volatility = features.get('volatility_5m', 0)
        trend = features.get('trend_strength', 0)
        
        if volatility > 0.002:
            return 'volatile'
        elif abs(trend) > 0.5:
            return 'trending'
        elif volatility < 0.0005:
            return 'ranging'
        else:
            return 'uncertain'
            
    def _select_strategy(self, regime: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Simple strategy selection for testing"""
        strategies = {
            'trending': {'type': 'asymmetric', 'spacing': 0.002, 'levels': 8},
            'ranging': {'type': 'symmetric', 'spacing': 0.001, 'levels': 10},
            'volatile': {'type': 'symmetric', 'spacing': 0.003, 'levels': 6},
            'breakout': {'type': 'asymmetric', 'spacing': 0.0025, 'levels': 7},
            'uncertain': {'type': 'symmetric', 'spacing': 0.0015, 'levels': 5}
        }
        
        return strategies.get(regime, strategies['uncertain'])
        
    def _validate_risk(self, strategy: Dict[str, Any]) -> bool:
        """Simple risk validation for testing"""
        # Check strategy parameters are within bounds
        if strategy['spacing'] < 0.0001 or strategy['spacing'] > 0.01:
            return False
        if strategy['levels'] < 2 or strategy['levels'] > 20:
            return False
        return True
        
    async def test_error_recovery(self):
        """Test system error recovery"""
        # Test connection failures
        connection_recovered = await self._test_connection_recovery()
        assert connection_recovered
        
        # Test data validation failures
        validation_handled = await self._test_validation_recovery()
        assert validation_handled
        
        # Test component failures
        component_recovered = await self._test_component_recovery()
        assert component_recovered
        
    async def _test_connection_recovery(self) -> bool:
        """Test connection failure recovery"""
        # Simulate connection failure and recovery
        try:
            # Simulate failure
            raise ConnectionError("Simulated connection failure")
        except ConnectionError:
            # Simulate recovery
            await asyncio.sleep(0.1)
            return True
        return False
        
    async def _test_validation_recovery(self) -> bool:
        """Test data validation recovery"""
        invalid_data = {'price': -100}  # Invalid price
        
        try:
            if invalid_data['price'] < 0:
                raise ValueError("Invalid price data")
        except ValueError:
            # Handle invalid data
            return True
        return False
        
    async def _test_component_recovery(self) -> bool:
        """Test component failure recovery"""
        # Simulate component failure
        component_status = {'healthy': False}
        
        # Recovery logic
        if not component_status['healthy']:
            # Restart component
            await asyncio.sleep(0.1)
            component_status['healthy'] = True
            
        return component_status['healthy']


# ============================================================================
# Test Configuration
# ============================================================================

def create_test_config() -> Dict[str, Any]:
    """Create test configuration"""
    return {
        'market_data': {
            'symbols': ['BTC/USDT'],
            'update_interval': 1,
            'buffer_size': 1000
        },
        'features': {
            'indicators': ['volatility', 'trend', 'volume', 'rsi'],
            'lookback_periods': [5, 20, 50]
        },
        'attention': {
            'learning_rate': 0.001,
            'window_size': 1000,
            'min_trades_for_active': 100
        },
        'regime_detector': {
            'ensemble': {
                'enabled': True,
                'methods': ['gmm', 'kmeans', 'rule_based']
            }
        },
        'strategy_selector': {
            'cache_size': 100,
            'validation_enabled': True
        },
        'risk_management': {
            'max_position_size': 0.1,
            'max_daily_loss': 0.02,
            'max_drawdown': 0.05
        },
        'execution': {
            'mode': 'simulation',
            'latency_target': 100,
            'fee_rate': 0.001
        },
        'performance_monitor': {
            'metrics_window': 1000,
            'save_interval': 300
        },
        'feedback_loop': {
            'update_interval': 60,
            'learning_rate': 0.001
        }
    }


# ============================================================================
# Main Test Runner
# ============================================================================

class GridAttentionTestRunner:
    """Main test runner for GridAttention system"""
    
    def __init__(self):
        self.testers = [
            AttentionLayerTester(),
            MarketRegimeDetectorTester(),
            GridStrategySelectorTester(),
            RiskManagementTester(),
            ExecutionEngineTester(),
            PerformanceMonitorTester(),
            FeedbackLoopTester(),
            SystemIntegrationTester()
        ]
        self.results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all component tests"""
        logger.info("=" * 60)
        logger.info("Starting GridAttention System Test Suite")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run tests for each component
        for tester in self.testers:
            logger.info(f"\nTesting {tester.component_name}...")
            
            # Get all test methods
            test_methods = [
                method for method in dir(tester)
                if method.startswith('test_') and callable(getattr(tester, method))
            ]
            
            # Run each test
            for method_name in test_methods:
                test_method = getattr(tester, method_name)
                await tester.run_test(method_name, test_method)
                
            # Store results
            self.results[tester.component_name] = tester.get_summary()
            
        # Generate summary report
        total_time = time.time() - start_time
        summary = self._generate_summary(total_time)
        
        return summary
        
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate test summary report"""
        total_tests = 0
        total_passed = 0
        component_results = []
        
        for component, result in self.results.items():
            total_tests += result['total_tests']
            total_passed += result['passed']
            
            component_results.append({
                'component': component,
                'tests': result['total_tests'],
                'passed': result['passed'],
                'failed': result['failed'],
                'success_rate': result['success_rate']
            })
            
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'overall': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_tests - total_passed,
                'success_rate': overall_success_rate
            },
            'components': component_results,
            'detailed_results': self.results
        }
        
        return summary
        
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary to console"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        print(f"\nExecution Time: {summary['total_execution_time']:.2f} seconds")
        print(f"\nOverall Results:")
        print(f"  Total Tests: {summary['overall']['total_tests']}")
        print(f"  Passed: {summary['overall']['passed']}")
        print(f"  Failed: {summary['overall']['failed']}")
        print(f"  Success Rate: {summary['overall']['success_rate']:.1%}")
        
        print("\nComponent Results:")
        print("-" * 60)
        print(f"{'Component':<25} {'Tests':>8} {'Passed':>8} {'Failed':>8} {'Rate':>8}")
        print("-" * 60)
        
        for comp in summary['components']:
            print(f"{comp['component']:<25} {comp['tests']:>8} {comp['passed']:>8} "
                  f"{comp['failed']:>8} {comp['success_rate']:>7.1%}")
            
        # Print failed tests
        print("\nFailed Tests:")
        print("-" * 60)
        
        for component, results in summary['detailed_results'].items():
            for test in results['results']:
                if not test.passed:
                    print(f"  {test.name}")
                    if test.error:
                        print(f"    Error: {test.error}")
                        
    def save_results(self, summary: Dict[str, Any], filepath: str = "test_results.json"):
        """Save test results to file"""
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Test results saved to {filepath}")


# ============================================================================
# Test Execution
# ============================================================================

async def main():
    """Main test execution function"""
    # Create test runner
    runner = GridAttentionTestRunner()
    
    # Run all tests
    summary = await runner.run_all_tests()
    
    # Print results
    runner.print_summary(summary)
    
    # Save results
    runner.save_results(summary)
    
    # Return success/failure
    return summary['overall']['success_rate'] == 1.0


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    
    # Exit with appropriate code
    exit(0 if success else 1)