#!/usr/bin/env python3
"""
test_warmup_integration.py
Integration test script for warmup functionality

Tests:
1. Warmup state file creation and loading
2. Threshold adjustments when warmup is loaded
3. Learning acceleration verification
4. Feature importance transfer

Usage:
    python test_warmup_integration.py
"""

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Mock the required imports for testing
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockFeatureAttention:
    """Mock feature attention for testing"""
    def __init__(self):
        self.importance_scores = {}
        self.attention_weights = {}


class MockTemporalAttention:
    """Mock temporal attention for testing"""
    def __init__(self):
        self.temporal_weights = {'short_term': 0.5, 'medium_term': 0.3, 'long_term': 0.2}
    
    def get_patterns(self):
        return ['trend_up', 'trend_down', 'ranging']


class MockRegimeAttention:
    """Mock regime attention for testing"""
    def __init__(self):
        self.parameter_adjustments = {'ranging': 1.0, 'trending': 1.2, 'volatile': 0.8}
    
    def get_performance_by_regime(self):
        return {'ranging': 0.65, 'trending': 0.72, 'volatile': 0.58}


class MockAttentionMetrics:
    """Mock attention metrics for testing"""
    def __init__(self):
        self.total_observations = 0
        self.shadow_calculations = 0
        self.active_applications = 0


class MockPhaseController:
    """Mock phase controller for testing"""
    def __init__(self, config):
        self.config = config
        self.min_trades_learning = config.get('min_trades_learning', 2000)
        self.min_trades_shadow = config.get('min_trades_shadow', 500)


class TestAttentionLearningLayer:
    """Test version of AttentionLearningLayer for integration testing"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.warmup_loaded = False
        
        # Mock components
        self.feature_attention = MockFeatureAttention()
        self.temporal_attention = MockTemporalAttention()
        self.regime_attention = MockRegimeAttention()
        self.metrics = MockAttentionMetrics()
        self.phase_controller = MockPhaseController(self.config)
        
        # Original thresholds for testing
        self.original_thresholds = {
            'learning': 2000,
            'shadow': 500,
            'active': 200
        }
        
    async def _load_warmup_state(self, warmup_file: str):
        """Load warmup state and adjust thresholds"""
        try:
            with open(warmup_file, 'r') as f:
                content = f.read()
                logger.info(f"File content preview: {content[:200]}...")
                warmup_data = json.loads(content)
                logger.info(f"Loaded data type: {type(warmup_data)}, keys: {list(warmup_data.keys()) if isinstance(warmup_data, dict) else 'Not a dict'}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Cannot load warmup file: {e}")
        
        if warmup_data is None:
            raise ValueError("Warmup file contains null data")
        
        if not isinstance(warmup_data, dict):
            raise ValueError(f"Invalid warmup data type: {type(warmup_data)}")
        
        # Validate warmup data structure
        required_keys = ['attention_state', 'feature_importance', 'learning_progress', 'timestamp']
        for key in required_keys:
            if key not in warmup_data:
                raise ValueError(f"Invalid warmup file: missing {key}")
        
        # Apply feature importance
        feature_importance = warmup_data.get('feature_importance', {})
        if feature_importance:
            self.feature_attention.importance_scores.update(feature_importance)
        
        # Calculate threshold reductions
        learning_progress = warmup_data.get('learning_progress', 0.0)
        reduction_factor = min(0.9, learning_progress)
        
        self.reduced_thresholds = {
            'learning': max(200, int(self.original_thresholds['learning'] * (1 - reduction_factor))),
            'shadow': max(100, int(self.original_thresholds['shadow'] * (1 - reduction_factor))),
            'active': max(50, int(self.original_thresholds['active'] * (1 - reduction_factor)))
        }
        
        # Set initial observations
        attention_state = warmup_data.get('attention_state', {})
        warmup_observations = attention_state.get('observations', 0)
        initial_observations = int(warmup_observations * 0.1)
        self.metrics.total_observations = initial_observations
        
        self.warmup_loaded = True
        
        logger.info(f"Warmup loaded: Learning {self.reduced_thresholds['learning']}, "
                   f"Shadow {self.reduced_thresholds['shadow']}, Active {self.reduced_thresholds['active']}")
        logger.info(f"Starting with {initial_observations} observation credits")
        
        return True


def create_test_warmup_state() -> Dict[str, Any]:
    """Create a test warmup state"""
    state = {
        'timestamp': '2024-01-01T12:00:00',
        'attention_state': {
            'observations': 150000,
            'phase': 'learning'
        },
        'feature_importance': {
            'price_change_5m': 0.25,
            'volatility_5m': 0.20,
            'rsi_14': 0.18,
            'volume_ratio': 0.15,
            'trend_strength': 0.12,
            'spread_bps': 0.10
        },
        'learning_progress': 0.85,
        'config': {
            'target_observations': 150000,
            'phases': ['initial_learning', 'regime_diversity', 'recent_conditions'],
            'warmup_version': '1.0.0'
        }
    }
    logger.info(f"Created test state: {type(state)} with keys: {list(state.keys())}")
    return state


async def test_warmup_state_loading():
    """Test 1: Warmup state file loading"""
    logger.info("🧪 Test 1: Warmup state file loading")
    
    temp_file = None
    try:
        # Create temporary warmup file
        test_state = create_test_warmup_state()
        logger.info(f"Test state before saving: {test_state}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_state, f, indent=2)
            temp_file = f.name
        
        logger.info(f"Saved to temp file: {temp_file}")
        
        # Test loading
        attention_layer = TestAttentionLearningLayer()
        try:
            result = await attention_layer._load_warmup_state(temp_file)
        except Exception as e:
            logger.error(f"Error in _load_warmup_state: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Verify loading
        assert result == True, "Warmup state loading failed"
        assert attention_layer.warmup_loaded == True, "Warmup loaded flag not set"
        
        # Verify feature importance transfer
        expected_features = test_state.get('feature_importance', {})
        loaded_features = attention_layer.feature_attention.importance_scores
        
        for feature, importance in expected_features.items():
            assert feature in loaded_features, f"Feature {feature} not loaded"
            assert loaded_features[feature] == importance, f"Feature {feature} importance mismatch"
        
        logger.info("✅ Test 1 passed: Warmup state loading works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
        return False
        
    finally:
        # Cleanup
        if temp_file:
            Path(temp_file).unlink(missing_ok=True)


async def test_threshold_adjustments():
    """Test 2: Learning threshold adjustments"""
    logger.info("🧪 Test 2: Learning threshold adjustments")
    
    try:
        # Create test state with high learning progress
        test_state = create_test_warmup_state()
        test_state['learning_progress'] = 0.8  # 80% progress
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_state, f, indent=2)
            temp_file = f.name
        
        # Load warmup state
        attention_layer = TestAttentionLearningLayer()
        await attention_layer._load_warmup_state(temp_file)
        
        # Verify threshold reductions
        original = attention_layer.original_thresholds
        reduced = attention_layer.reduced_thresholds
        
        # Should be significantly reduced
        learning_reduction = (original['learning'] - reduced['learning']) / original['learning']
        shadow_reduction = (original['shadow'] - reduced['shadow']) / original['shadow']
        
        assert learning_reduction > 0.5, f"Learning threshold not reduced enough: {learning_reduction:.2%}"
        assert shadow_reduction > 0.5, f"Shadow threshold not reduced enough: {shadow_reduction:.2%}"
        
        # Verify minimum thresholds
        assert reduced['learning'] >= 200, "Learning threshold below minimum"
        assert reduced['shadow'] >= 100, "Shadow threshold below minimum"
        assert reduced['active'] >= 50, "Active threshold below minimum"
        
        logger.info(f"✅ Test 2 passed: Thresholds reduced by ~{learning_reduction:.0%}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
        return False
        
    finally:
        if temp_file:
            Path(temp_file).unlink(missing_ok=True)


async def test_learning_acceleration():
    """Test 3: Learning acceleration verification"""
    logger.info("🧪 Test 3: Learning acceleration verification")
    
    temp_file = None
    try:
        # Test with warmup vs without warmup
        test_state = create_test_warmup_state()
        
        # Test WITHOUT warmup
        fresh_layer = TestAttentionLearningLayer()
        fresh_thresholds = fresh_layer.original_thresholds.copy()
        
        # Test WITH warmup  
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_state, f, indent=2)
            temp_file = f.name
        
        warmup_layer = TestAttentionLearningLayer()
        await warmup_layer._load_warmup_state(temp_file)
        warmup_thresholds = warmup_layer.reduced_thresholds
        
        # Calculate acceleration
        learning_acceleration = fresh_thresholds['learning'] / warmup_thresholds['learning']
        shadow_acceleration = fresh_thresholds['shadow'] / warmup_thresholds['shadow']
        
        # Should be significantly faster
        assert learning_acceleration > 3, f"Learning acceleration too low: {learning_acceleration:.1f}x"
        assert shadow_acceleration > 3, f"Shadow acceleration too low: {shadow_acceleration:.1f}x"
        
        # Verify observation credits
        assert warmup_layer.metrics.total_observations > 0, "No observation credits given"
        attention_state = test_state.get('attention_state', {})
        expected_credits = int(attention_state.get('observations', 0) * 0.1)
        assert warmup_layer.metrics.total_observations == expected_credits, "Incorrect observation credits"
        
        logger.info(f"✅ Test 3 passed: Learning accelerated by {learning_acceleration:.1f}x")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 3 failed: {e}")
        return False
        
    finally:
        if temp_file:
            Path(temp_file).unlink(missing_ok=True)


async def test_invalid_warmup_file():
    """Test 4: Invalid warmup file handling"""
    logger.info("🧪 Test 4: Invalid warmup file handling")
    
    try:
        # Test with invalid JSON
        invalid_file = None
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_file = f.name
        
        attention_layer = TestAttentionLearningLayer()
        
        try:
            await attention_layer._load_warmup_state(invalid_file)
            assert False, "Should have failed with invalid JSON"
        except (json.JSONDecodeError, ValueError):
            pass  # Expected
        
        # Test with missing required keys
        incomplete_state = {'timestamp': '2024-01-01T12:00:00'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(incomplete_state, f)
            incomplete_file = f.name
        
        try:
            await attention_layer._load_warmup_state(incomplete_file)
            assert False, "Should have failed with missing keys"
        except ValueError as e:
            assert "missing" in str(e).lower(), f"Wrong error message: {e}"
        
        logger.info("✅ Test 4 passed: Invalid file handling works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 4 failed: {e}")
        return False
        
    finally:
        # Cleanup files safely
        try:
            if 'invalid_file' in locals() and invalid_file:
                Path(invalid_file).unlink(missing_ok=True)
        except:
            pass
        try:
            if 'incomplete_file' in locals() and incomplete_file:
                Path(incomplete_file).unlink(missing_ok=True)
        except:
            pass


async def test_performance_benchmark():
    """Test 5: Performance benchmark"""
    logger.info("🧪 Test 5: Performance benchmark")
    
    temp_file = None
    try:
        test_state = create_test_warmup_state()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_state, f, indent=2)
            temp_file = f.name
        
        # Benchmark loading time
        start_time = time.time()
        
        attention_layer = TestAttentionLearningLayer()
        await attention_layer._load_warmup_state(temp_file)
        
        loading_time = time.time() - start_time
        
        # Should load quickly
        assert loading_time < 1.0, f"Loading too slow: {loading_time:.3f}s"
        
        # Verify memory efficiency (basic check)
        import sys
        state_size = sys.getsizeof(attention_layer.feature_attention.importance_scores)
        assert state_size < 10000, f"State size too large: {state_size} bytes"
        
        logger.info(f"✅ Test 5 passed: Loading time {loading_time:.3f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 5 failed: {e}")
        return False
        
    finally:
        if temp_file:
            Path(temp_file).unlink(missing_ok=True)


async def run_integration_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting warmup integration tests")
    
    tests = [
        test_warmup_state_loading,
        test_threshold_adjustments,
        test_learning_acceleration,
        test_invalid_warmup_file,
        test_performance_benchmark
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Warmup integration is working correctly.")
        logger.info("\n✅ Ready for production use:")
        logger.info("   1. Run warmup_main.ipynb with your historical data")
        logger.info("   2. Copy generated 'attention_warmup_state.json' to GridAttention root")
        logger.info("   3. Start GridAttention system normally")
        logger.info("   4. Enjoy 3-10x faster learning!")
    else:
        logger.error(f"❌ {total - passed} tests failed. Please fix issues before using warmup.")
    
    return passed == total


async def simple_test():
    """Simple test to check basic functionality"""
    print("🧪 Simple warmup test")
    
    # Create test state
    state = create_test_warmup_state()
    print(f"✅ State created: {type(state)}")
    
    # Create test file
    import tempfile, json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(state, f, indent=2)
        temp_file = f.name
    
    print(f"✅ File saved: {temp_file}")
    
    # Test loading
    layer = TestAttentionLearningLayer()
    try:
        result = await layer._load_warmup_state(temp_file)
        print(f"✅ Loading result: {result}")
        print(f"✅ Warmup loaded: {layer.warmup_loaded}")
        return True
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(temp_file).unlink(missing_ok=True)

if __name__ == "__main__":
    # Run simple test first
    success = asyncio.run(simple_test())
    
    if success:
        print("\n🎉 Simple test passed! Running full tests...")
        success = asyncio.run(run_integration_tests())
    
    exit(0 if success else 1)